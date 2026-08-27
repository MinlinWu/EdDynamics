#!/usr/bin/env python3
from __future__ import annotations

"""Null-referenced recovery of construction-excess M-Psi dynamics.

This script is an evaluation-only extension of the frozen EdNet-KT4 pipeline.
It does not refit the semantic coordinates, construction-matched null,
minimal-mechanism family or coefficients, Event-SSL models, grid, support mask,
convergence core, or mesostate partition.

For each requested split it:
  1. reconstructs the formal Stage-1 empirical field under the exact Stage-1
     user-balanced weighting contract;
  2. rebuilds the frozen construction-matched permutation layouts and computes
     the exact expectation of the cyclic-shift null field;
  3. evaluates the frozen mechanism and one or more already-evaluated Event-SSL
     prediction tables from the same empirical current-state anchors;
  4. tests whether each model is closer to the empirical field than the exact
     construction-null field and whether its null-referenced correction recovers
     the empirical excess field;
  5. optionally performs a paired user-level multiplier bootstrap conditional on the
     frozen matching groups and exact row-level null expectations.

A_val may be used to fit non-negative scalar and per-coordinate rescalings of
the exact null field. Those coefficients are then frozen and applied to
B_confirm as stronger rescaled-scaffold benchmarks; they are not treated as
additional formal null distributions. B_confirm remains output-only: it is never used to
update model parameters, coordinates, masks, matching cutpoints, or regions.
"""

import argparse
import dataclasses
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

EPS = 1e-12
PRIMARY_COORDINATE = "MR_PsiA"
DEFAULT_STAGE1_ROOT = Path("/data/datasets/KT4/outputs_KT4/stage1")
DEFAULT_CMN_OUTPUT_ROOT = Path(
    "/data/datasets/KT4/outputs_KT4/stage1_construction_matched_null"
)
DEFAULT_FROZEN_MECHANISM_MANIFEST = Path(
    "/data/datasets/KT4/outputs_KT4/stage2_phase2_freeze/metadata/"
    "phase2_frozen_model_manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/datasets/KT4/outputs_KT4/null_referenced_downstream_recovery"
)
DEFAULT_BOOTSTRAP_SEED = 20260731
AUDIT_ATOL = 2e-6


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def json_safe(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return json_safe(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(key): json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(obj), handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(obj: Any) -> str:
    payload = json.dumps(
        json_safe(obj),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def import_module(path: Path, module_name: str) -> Any:
    source = path.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location(module_name, str(source))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def resolve_script(explicit: Optional[Path], sibling_name: str) -> Path:
    if explicit is not None:
        return explicit.resolve()
    sibling = Path(__file__).resolve().with_name(sibling_name)
    if sibling.exists():
        return sibling
    raise FileNotFoundError(
        f"Pass the explicit path for {sibling_name}; no sibling copy was found."
    )


def find_table(base_or_path: Path) -> Path:
    path = Path(base_or_path)
    if path.exists() and path.is_file():
        return path
    for extension in (".parquet", ".csv.gz", ".csv"):
        candidate = path.with_suffix(extension)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find table for {base_or_path}")


def read_table(base_or_path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    path = find_table(base_or_path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    return pd.read_csv(
        path,
        usecols=list(columns) if columns is not None else None,
        low_memory=False,
    )


def write_table(frame: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    parquet = base.with_suffix(".parquet")
    parquet_tmp = parquet.with_name(parquet.name + ".tmp")
    try:
        frame.to_parquet(parquet_tmp, index=False)
        os.replace(parquet_tmp, parquet)
        return parquet
    except Exception:
        if parquet_tmp.exists():
            parquet_tmp.unlink()
        csv_path = base.with_suffix(".csv.gz")
        csv_tmp = csv_path.with_name(csv_path.name + ".tmp")
        frame.to_csv(csv_tmp, index=False, compression="gzip")
        os.replace(csv_tmp, csv_path)
        return csv_path


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label).strip())
    return cleaned.strip("_") or "model"


def numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise KeyError(column)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def integer_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise KeyError(column)
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise RuntimeError(f"Non-finite identifiers in {column}")
    return values.to_numpy(dtype=np.int64)


def max_abs_difference(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    if not np.any(valid):
        return float("nan")
    return float(np.max(np.abs(first[valid] - second[valid])))


def bounded_state(values: np.ndarray, label: str) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64)
    if not np.isfinite(output).all():
        raise RuntimeError(f"{label} contains non-finite values.")
    excess = float(np.max(np.maximum(np.abs(output) - 1.0, 0.0))) if output.size else 0.0
    if excess > AUDIT_ATOL:
        raise RuntimeError(f"{label} leaves [-1,1] by {excess:.3e}.")
    return np.clip(output, -1.0, 1.0)


def parse_named_roots(values: Sequence[str]) -> Dict[str, Path]:
    output: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                "Each --event-ssl-root value must be LABEL=/path/to/evaluation_root"
            )
        label, raw_path = value.split("=", 1)
        label = sanitize_label(label)
        if label in output:
            raise ValueError(f"Duplicate Event-SSL label: {label}")
        output[label] = Path(raw_path).resolve()
    if not output:
        raise ValueError("At least one --event-ssl-root is required.")
    return output


# -----------------------------------------------------------------------------
# Exact expectation of the frozen cyclic-shift construction null
# -----------------------------------------------------------------------------
@dataclass
class ExactNullExpectation:
    expected_z_m: np.ndarray
    expected_z_psi: np.ndarray
    expected_pair_product: np.ndarray
    audit: Dict[str, Any]


def exact_expected_donor_pairs(
    layouts: Sequence[Any],
    effective_randomizable: np.ndarray,
    z_m: np.ndarray,
    z_psi: np.ndarray,
) -> ExactNullExpectation:
    """Return the exact rowwise donor expectation under random non-zero shifts.

    Within a frozen permutation group of size m, one common cyclic shift is
    drawn uniformly from 1,...,m-1. For each recipient, its donor therefore
    ranges exactly once over every other row in that group. The exact expected
    donor vector is the leave-one-out group mean. Rows excluded from randomized
    layouts remain self-mapped, matching the formal null implementation.
    """

    z_m = np.asarray(z_m, dtype=np.float64)
    z_psi = np.asarray(z_psi, dtype=np.float64)
    if z_m.shape != z_psi.shape:
        raise ValueError("z_m and z_psi shapes differ.")
    n = len(z_m)
    randomizable = np.asarray(effective_randomizable, dtype=bool)
    if len(randomizable) != n:
        raise ValueError("effective_randomizable length differs from innovations.")

    expected_m = z_m.copy()
    expected_psi = z_psi.copy()
    expected_product = (z_m * z_psi).copy()
    assigned = np.zeros(n, dtype=bool)
    group_sizes: List[int] = []
    layout_rows: List[dict] = []

    for layout in layouts:
        sorted_rows = np.asarray(layout.original_indices, dtype=np.int64)[
            np.asarray(layout.order, dtype=np.int64)
        ]
        group_id = np.asarray(layout.sorted_group, dtype=np.int64)
        starts = np.asarray(layout.starts, dtype=np.int64)
        counts = np.asarray(layout.counts, dtype=np.int64)
        if len(sorted_rows) == 0:
            continue
        if np.any(counts < 2):
            raise RuntimeError(f"Layout {layout.name} contains a singleton group.")
        if group_id.min(initial=0) < 0 or group_id.max(initial=-1) >= len(counts):
            raise RuntimeError(f"Layout {layout.name} has invalid group identifiers.")

        zm_sorted = z_m[sorted_rows]
        zp_sorted = z_psi[sorted_rows]
        product_sorted = zm_sorted * zp_sorted
        sum_m = np.add.reduceat(zm_sorted, starts)
        sum_psi = np.add.reduceat(zp_sorted, starts)
        sum_product = np.add.reduceat(product_sorted, starts)
        denominator = counts[group_id] - 1

        expected_m[sorted_rows] = (sum_m[group_id] - zm_sorted) / denominator
        expected_psi[sorted_rows] = (sum_psi[group_id] - zp_sorted) / denominator
        expected_product[sorted_rows] = (
            sum_product[group_id] - product_sorted
        ) / denominator
        assigned[sorted_rows] = True
        group_sizes.extend(counts.tolist())
        layout_rows.append(
            {
                "layout": str(layout.name),
                "rows": int(len(sorted_rows)),
                "groups": int(len(counts)),
                "minimum_group_size": int(np.min(counts)),
                "median_group_size": float(np.median(counts)),
                "maximum_group_size": int(np.max(counts)),
            }
        )

    if not np.array_equal(assigned, randomizable):
        mismatch = int(np.sum(assigned != randomizable))
        raise RuntimeError(
            f"Exact-expectation assignment differs from frozen randomizable mask on {mismatch} rows."
        )

    assigned_rows = np.flatnonzero(randomizable)
    if assigned_rows.size:
        mean_audit = {
            "mean_Z_M_original_randomized": float(np.mean(z_m[assigned_rows])),
            "mean_Z_M_expected_donor_randomized": float(
                np.mean(expected_m[assigned_rows])
            ),
            "mean_Z_Psi_original_randomized": float(np.mean(z_psi[assigned_rows])),
            "mean_Z_Psi_expected_donor_randomized": float(
                np.mean(expected_psi[assigned_rows])
            ),
            "mean_pair_product_original_randomized": float(
                np.mean((z_m * z_psi)[assigned_rows])
            ),
            "mean_pair_product_expected_donor_randomized": float(
                np.mean(expected_product[assigned_rows])
            ),
        }
        for before, after in (
            (
                mean_audit["mean_Z_M_original_randomized"],
                mean_audit["mean_Z_M_expected_donor_randomized"],
            ),
            (
                mean_audit["mean_Z_Psi_original_randomized"],
                mean_audit["mean_Z_Psi_expected_donor_randomized"],
            ),
            (
                mean_audit["mean_pair_product_original_randomized"],
                mean_audit["mean_pair_product_expected_donor_randomized"],
            ),
        ):
            if not np.isclose(before, after, atol=1e-10, rtol=1e-10):
                raise RuntimeError(
                    "Exact donor expectation does not preserve the randomized-row marginal."
                )
    else:
        mean_audit = {}

    audit = {
        "rows": int(n),
        "randomized_rows": int(np.sum(randomizable)),
        "self_mapped_rows": int(np.sum(~randomizable)),
        "layouts": layout_rows,
        "total_groups": int(len(group_sizes)),
        "minimum_group_size": int(min(group_sizes)) if group_sizes else None,
        "median_group_size": float(np.median(group_sizes)) if group_sizes else None,
        "maximum_group_size": int(max(group_sizes)) if group_sizes else None,
        "expectation_contract": (
            "leave-one-out group mean under a uniformly random non-zero cyclic shift; "
            "joint M/Psi donor pairs remain coupled"
        ),
        **mean_audit,
    }
    return ExactNullExpectation(expected_m, expected_psi, expected_product, audit)


def exact_null_increments(
    prepared: Any,
    expectation: ExactNullExpectation,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    m_denominator = np.asarray(prepared.m_denominator, dtype=np.float64)
    psi_denominator = np.asarray(prepared.psi_denominator, dtype=np.float64)
    if np.any(~np.isfinite(m_denominator)) or np.any(m_denominator <= EPS):
        raise RuntimeError("Exact-null response denominators are not strictly positive and finite.")
    if np.any(~np.isfinite(psi_denominator)) or np.any(psi_denominator <= EPS):
        raise RuntimeError("Exact-null exposure denominators are not strictly positive and finite.")
    next_m = (
        np.asarray(prepared.s_pre, dtype=np.float64)
        + np.asarray(prepared.a_m, dtype=np.float64) * expectation.expected_z_m
    ) / m_denominator
    next_psi = (
        np.asarray(prepared.g_pre, dtype=np.float64)
        + np.asarray(prepared.a_psi, dtype=np.float64) * expectation.expected_z_psi
    ) / psi_denominator
    next_m = bounded_state(next_m, "exact null next M")
    next_psi = bounded_state(next_psi, "exact null next Psi")
    dx = next_m - np.asarray(prepared.x, dtype=np.float64)
    dy = next_psi - np.asarray(prepared.y, dtype=np.float64)
    return dx, dy, {
        "max_abs_exact_null_next_M": float(np.max(np.abs(next_m))),
        "max_abs_exact_null_next_Psi": float(np.max(np.abs(next_psi))),
        "finite_exact_null_increments": bool(np.isfinite(dx).all() and np.isfinite(dy).all()),
    }


# -----------------------------------------------------------------------------
# Field metrics and rescaled-scaffold benchmarks
# -----------------------------------------------------------------------------
def pearson(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(valid)) < 3:
        return float("nan")
    aa = a[valid] - float(np.mean(a[valid]))
    bb = b[valid] - float(np.mean(b[valid]))
    denominator = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denominator <= EPS:
        return float("nan")
    return float(np.clip(np.dot(aa, bb) / denominator, -1.0, 1.0))


def common_field_mask(mask: np.ndarray, *arrays: np.ndarray) -> np.ndarray:
    valid = np.asarray(mask, dtype=bool).copy()
    for array in arrays:
        valid &= np.isfinite(np.asarray(array, dtype=np.float64))
    return valid


def normalized_cell_weights(weight: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(weight, dtype=np.float64)[mask]
    values = np.where(np.isfinite(values) & (values >= 0), values, 0.0)
    total = float(np.sum(values))
    if total <= EPS:
        return np.full(len(values), 1.0 / max(len(values), 1), dtype=np.float64)
    return values / total


def weighted_field_sse(
    first_u: np.ndarray,
    first_v: np.ndarray,
    second_u: np.ndarray,
    second_v: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> float:
    valid = common_field_mask(mask, first_u, first_v, second_u, second_v)
    if not np.any(valid):
        return float("nan")
    w = normalized_cell_weights(weight, valid)
    squared = (
        (np.asarray(first_u)[valid] - np.asarray(second_u)[valid]) ** 2
        + (np.asarray(first_v)[valid] - np.asarray(second_v)[valid]) ** 2
    )
    return float(np.sum(w * squared))


def weighted_component_sse(
    first: np.ndarray,
    second: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> float:
    valid = common_field_mask(mask, first, second)
    if not np.any(valid):
        return float("nan")
    w = normalized_cell_weights(weight, valid)
    return float(np.sum(w * (np.asarray(first)[valid] - np.asarray(second)[valid]) ** 2))


def vector_correlation(
    first_u: np.ndarray,
    first_v: np.ndarray,
    second_u: np.ndarray,
    second_v: np.ndarray,
    mask: np.ndarray,
) -> float:
    valid = common_field_mask(mask, first_u, first_v, second_u, second_v)
    if int(np.sum(valid)) < 3:
        return float("nan")
    first = np.column_stack(
        [np.asarray(first_u)[valid], np.asarray(first_v)[valid]]
    ).ravel()
    second = np.column_stack(
        [np.asarray(second_u)[valid], np.asarray(second_v)[valid]]
    ).ravel()
    return pearson(first, second)


def weighted_local_cosine(
    first_u: np.ndarray,
    first_v: np.ndarray,
    second_u: np.ndarray,
    second_v: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float]:
    valid = common_field_mask(mask, first_u, first_v, second_u, second_v)
    first_speed = np.hypot(np.asarray(first_u), np.asarray(first_v))
    second_speed = np.hypot(np.asarray(second_u), np.asarray(second_v))
    valid &= (first_speed > EPS) & (second_speed > EPS)
    total_mask_weight = float(
        np.sum(np.where(np.asarray(mask, dtype=bool), np.maximum(weight, 0.0), 0.0))
    )
    valid_weight = float(np.sum(np.asarray(weight)[valid])) if np.any(valid) else 0.0
    coverage = valid_weight / max(total_mask_weight, EPS)
    if not np.any(valid):
        return float("nan"), float(coverage)
    cosine = (
        np.asarray(first_u)[valid] * np.asarray(second_u)[valid]
        + np.asarray(first_v)[valid] * np.asarray(second_v)[valid]
    ) / (first_speed[valid] * second_speed[valid])
    w = normalized_cell_weights(weight, valid)
    return float(np.sum(w * np.clip(cosine, -1.0, 1.0))), float(coverage)


def weighted_inner_product(
    first_u: np.ndarray,
    first_v: np.ndarray,
    second_u: np.ndarray,
    second_v: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> float:
    valid = common_field_mask(mask, first_u, first_v, second_u, second_v)
    if not np.any(valid):
        return float("nan")
    w = normalized_cell_weights(weight, valid)
    return float(
        np.sum(
            w
            * (
                np.asarray(first_u)[valid] * np.asarray(second_u)[valid]
                + np.asarray(first_v)[valid] * np.asarray(second_v)[valid]
            )
        )
    )


@dataclass(frozen=True)
class ScaffoldRescalingCalibration:
    source_split: str
    scalar_alpha: float
    alpha_M: float
    alpha_Psi: float
    fitting_contract: str


def fit_scaffold_rescaling(
    empirical_u: np.ndarray,
    empirical_v: np.ndarray,
    null_u: np.ndarray,
    null_v: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
    source_split: str,
) -> ScaffoldRescalingCalibration:
    scalar_num = weighted_inner_product(
        empirical_u, empirical_v, null_u, null_v, weight, mask
    )
    scalar_den = weighted_inner_product(null_u, null_v, null_u, null_v, weight, mask)
    scalar = max(scalar_num / max(scalar_den, EPS), 0.0)

    valid_m = common_field_mask(mask, empirical_u, null_u)
    wm = normalized_cell_weights(weight, valid_m)
    num_m = float(np.sum(wm * np.asarray(empirical_u)[valid_m] * np.asarray(null_u)[valid_m]))
    den_m = float(np.sum(wm * np.asarray(null_u)[valid_m] ** 2))
    alpha_m = max(num_m / max(den_m, EPS), 0.0)

    valid_p = common_field_mask(mask, empirical_v, null_v)
    wp = normalized_cell_weights(weight, valid_p)
    num_p = float(np.sum(wp * np.asarray(empirical_v)[valid_p] * np.asarray(null_v)[valid_p]))
    den_p = float(np.sum(wp * np.asarray(null_v)[valid_p] ** 2))
    alpha_p = max(num_p / max(den_p, EPS), 0.0)

    return ScaffoldRescalingCalibration(
        source_split=str(source_split),
        scalar_alpha=float(scalar),
        alpha_M=float(alpha_m),
        alpha_Psi=float(alpha_p),
        fitting_contract=(
            "occupancy-weighted non-negative least-squares amplitude scaling of the exact "
            "construction-null field; no intercept; coefficients fitted on A_val and frozen "
            "before B_confirm evaluation; rescaled-scaffold benchmark only, not a formal null"
        ),
    )


def recovery_metrics(
    split: str,
    model_label: str,
    empirical_u: np.ndarray,
    empirical_v: np.ndarray,
    null_u: np.ndarray,
    null_v: np.ndarray,
    model_u: np.ndarray,
    model_v: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
    rescaling: Optional[ScaffoldRescalingCalibration],
    rescaling_role: str,
) -> Dict[str, Any]:
    valid = common_field_mask(
        mask,
        empirical_u,
        empirical_v,
        null_u,
        null_v,
        model_u,
        model_v,
    )
    if int(np.sum(valid)) < 3:
        raise RuntimeError(f"Too few common field cells for {model_label} on {split}.")

    null_sse = weighted_field_sse(
        empirical_u, empirical_v, null_u, null_v, weight, valid
    )
    model_sse = weighted_field_sse(
        empirical_u, empirical_v, model_u, model_v, weight, valid
    )
    if not np.isfinite(null_sse) or null_sse <= EPS:
        raise RuntimeError(
            f"The empirical field has no numerically resolved departure from the exact null "
            f"for {model_label} on {split}; null-referenced recovery is undefined."
        )
    delta_sse = model_sse - null_sse
    fraction_recovered = 1.0 - model_sse / null_sse
    rmse_ratio = math.sqrt(max(model_sse, 0.0) / null_sse)

    empirical_excess_u = np.asarray(empirical_u) - np.asarray(null_u)
    empirical_excess_v = np.asarray(empirical_v) - np.asarray(null_v)
    model_correction_u = np.asarray(model_u) - np.asarray(null_u)
    model_correction_v = np.asarray(model_v) - np.asarray(null_v)

    excess_corr = vector_correlation(
        model_correction_u,
        model_correction_v,
        empirical_excess_u,
        empirical_excess_v,
        valid,
    )
    excess_cosine, excess_cosine_coverage = weighted_local_cosine(
        model_correction_u,
        model_correction_v,
        empirical_excess_u,
        empirical_excess_v,
        weight,
        valid,
    )
    excess_speed_corr = pearson(
        np.hypot(model_correction_u[valid], model_correction_v[valid]),
        np.hypot(empirical_excess_u[valid], empirical_excess_v[valid]),
    )
    slope_num = weighted_inner_product(
        model_correction_u,
        model_correction_v,
        empirical_excess_u,
        empirical_excess_v,
        weight,
        valid,
    )
    slope_den = weighted_inner_product(
        empirical_excess_u,
        empirical_excess_v,
        empirical_excess_u,
        empirical_excess_v,
        weight,
        valid,
    )
    amplitude_slope = slope_num / max(slope_den, EPS)

    raw_cosine, raw_cosine_coverage = weighted_local_cosine(
        model_u, model_v, empirical_u, empirical_v, weight, valid
    )
    null_cosine, null_cosine_coverage = weighted_local_cosine(
        null_u, null_v, empirical_u, empirical_v, weight, valid
    )

    m_null_sse = weighted_component_sse(empirical_u, null_u, weight, valid)
    m_model_sse = weighted_component_sse(empirical_u, model_u, weight, valid)
    p_null_sse = weighted_component_sse(empirical_v, null_v, weight, valid)
    p_model_sse = weighted_component_sse(empirical_v, model_v, weight, valid)

    result: Dict[str, Any] = {
        "split": split,
        "model": model_label,
        "evaluation_view": "empirical_current_state_anchor",
        "supported_cells": int(np.sum(valid)),
        "supported_occupancy_mass": float(np.sum(np.asarray(weight)[valid])),
        "exact_null_sse_to_empirical": float(null_sse),
        "model_sse_to_empirical": float(model_sse),
        "primary_delta_sse_model_minus_exact_null": float(delta_sse),
        "null_relative_field_skill": float(fraction_recovered),
        "null_normalized_rmse_ratio": float(rmse_ratio),
        "model_better_than_exact_null_point_estimate": bool(delta_sse < 0),
        "model_vs_empirical_vector_corr": vector_correlation(
            model_u, model_v, empirical_u, empirical_v, valid
        ),
        "model_vs_empirical_speed_corr": pearson(
            np.hypot(np.asarray(model_u)[valid], np.asarray(model_v)[valid]),
            np.hypot(np.asarray(empirical_u)[valid], np.asarray(empirical_v)[valid]),
        ),
        "model_vs_empirical_weighted_local_cosine": raw_cosine,
        "model_vs_empirical_local_cosine_occupancy_coverage": raw_cosine_coverage,
        "exact_null_vs_empirical_vector_corr": vector_correlation(
            null_u, null_v, empirical_u, empirical_v, valid
        ),
        "exact_null_vs_empirical_weighted_local_cosine": null_cosine,
        "exact_null_vs_empirical_local_cosine_occupancy_coverage": null_cosine_coverage,
        "null_referenced_correction_vs_empirical_excess_vector_corr": excess_corr,
        "null_referenced_correction_vs_empirical_excess_speed_corr": excess_speed_corr,
        "null_referenced_correction_vs_empirical_excess_weighted_local_cosine": excess_cosine,
        "excess_local_cosine_occupancy_coverage": excess_cosine_coverage,
        "null_referenced_excess_amplitude_slope": float(amplitude_slope),
        "M_null_relative_field_skill": (
            float(1.0 - m_model_sse / m_null_sse)
            if np.isfinite(m_null_sse) and m_null_sse > EPS
            else np.nan
        ),
        "Psi_null_relative_field_skill": (
            float(1.0 - p_model_sse / p_null_sse)
            if np.isfinite(p_null_sse) and p_null_sse > EPS
            else np.nan
        ),
        "M_model_minus_null_sse": float(m_model_sse - m_null_sse),
        "Psi_model_minus_null_sse": float(p_model_sse - p_null_sse),
        "inference_note": (
            "The paired squared-error difference is the primary comparison. "
            "Ratios, correlations, cosines and slopes are effect-size diagnostics; "
            "the correction-versus-excess comparisons share the same subtracted null baseline "
            "and are therefore not primary tests."
        ),
    }

    if rescaling is not None:
        scalar_u = rescaling.scalar_alpha * np.asarray(null_u)
        scalar_v = rescaling.scalar_alpha * np.asarray(null_v)
        diagonal_u = rescaling.alpha_M * np.asarray(null_u)
        diagonal_v = rescaling.alpha_Psi * np.asarray(null_v)
        scalar_sse = weighted_field_sse(
            empirical_u, empirical_v, scalar_u, scalar_v, weight, valid
        )
        diagonal_sse = weighted_field_sse(
            empirical_u, empirical_v, diagonal_u, diagonal_v, weight, valid
        )
        result.update(
            {
                "scaffold_rescaling_source_split": rescaling.source_split,
                "scaffold_rescaling_role": rescaling_role,
                "frozen_scalar_scaffold_alpha": rescaling.scalar_alpha,
                "frozen_diagonal_scaffold_alpha_M": rescaling.alpha_M,
                "frozen_diagonal_scaffold_alpha_Psi": rescaling.alpha_Psi,
                "scalar_rescaled_scaffold_sse_to_empirical": scalar_sse,
                "diagonal_rescaled_scaffold_sse_to_empirical": diagonal_sse,
                "model_minus_scalar_rescaled_scaffold_sse": model_sse - scalar_sse,
                "model_minus_diagonal_rescaled_scaffold_sse": model_sse - diagonal_sse,
                "model_better_than_frozen_diagonal_scaffold_point_estimate": bool(
                    model_sse < diagonal_sse
                ),
            }
        )
    return result


def pairwise_field_metrics(
    split: str,
    first_label: str,
    second_label: str,
    first_u: np.ndarray,
    first_v: np.ndarray,
    second_u: np.ndarray,
    second_v: np.ndarray,
    null_u: np.ndarray,
    null_v: np.ndarray,
    empirical_u: np.ndarray,
    empirical_v: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    valid = common_field_mask(
        mask,
        first_u,
        first_v,
        second_u,
        second_v,
        null_u,
        null_v,
        empirical_u,
        empirical_v,
    )
    raw_cos, raw_cov = weighted_local_cosine(
        first_u, first_v, second_u, second_v, weight, valid
    )
    first_q_u = np.asarray(first_u) - np.asarray(null_u)
    first_q_v = np.asarray(first_v) - np.asarray(null_v)
    second_q_u = np.asarray(second_u) - np.asarray(null_u)
    second_q_v = np.asarray(second_v) - np.asarray(null_v)
    correction_cos, correction_cov = weighted_local_cosine(
        first_q_u,
        first_q_v,
        second_q_u,
        second_q_v,
        weight,
        valid,
    )
    return {
        "split": split,
        "first_model": first_label,
        "second_model": second_label,
        "raw_field_vector_corr": vector_correlation(
            first_u, first_v, second_u, second_v, valid
        ),
        "raw_field_speed_corr": pearson(
            np.hypot(np.asarray(first_u)[valid], np.asarray(first_v)[valid]),
            np.hypot(np.asarray(second_u)[valid], np.asarray(second_v)[valid]),
        ),
        "raw_field_weighted_local_cosine": raw_cos,
        "raw_field_local_cosine_occupancy_coverage": raw_cov,
        "shared_null_subtracted_correction_vector_corr_descriptive": vector_correlation(
            first_q_u, first_q_v, second_q_u, second_q_v, valid
        ),
        "shared_null_subtracted_correction_weighted_local_cosine_descriptive": correction_cos,
        "shared_null_subtracted_local_cosine_occupancy_coverage": correction_cov,
        "interpretation_guardrail": (
            "Cross-model correction agreement is descriptive because both corrections subtract "
            "the same null field. Independent evidence comes from each model's recovery of the "
            "empirical excess target."
        ),
    }


# -----------------------------------------------------------------------------
# Row alignment and model outputs
# -----------------------------------------------------------------------------
def assert_exact_alignment(
    reference_user: np.ndarray,
    reference_step: np.ndarray,
    candidate_user: np.ndarray,
    candidate_step: np.ndarray,
    label: str,
) -> None:
    reference_user = np.asarray(reference_user, dtype=np.int64)
    reference_step = np.asarray(reference_step, dtype=np.int64)
    candidate_user = np.asarray(candidate_user, dtype=np.int64)
    candidate_step = np.asarray(candidate_step, dtype=np.int64)
    if len(reference_user) != len(candidate_user):
        raise RuntimeError(
            f"{label} has {len(candidate_user)} rows; the formal drift panel has "
            f"{len(reference_user)}. No intersection is taken silently."
        )

    def audit_sorted_unique(user: np.ndarray, step: np.ndarray, source: str) -> None:
        if len(user) <= 1:
            return
        duplicate = (user[1:] == user[:-1]) & (step[1:] == step[:-1])
        if np.any(duplicate):
            index = int(np.flatnonzero(duplicate)[0] + 1)
            raise RuntimeError(
                f"{source} contains a duplicate (user_id, bundle_step_index) key "
                f"at sorted row {index}: ({user[index]}, {step[index]})."
            )
        out_of_order = (user[1:] < user[:-1]) | (
            (user[1:] == user[:-1]) & (step[1:] < step[:-1])
        )
        if np.any(out_of_order):
            index = int(np.flatnonzero(out_of_order)[0] + 1)
            raise RuntimeError(f"{source} keys are not lexicographically sorted at row {index}.")

    audit_sorted_unique(reference_user, reference_step, "formal drift panel")
    audit_sorted_unique(candidate_user, candidate_step, label)
    mismatch = (reference_user != candidate_user) | (reference_step != candidate_step)
    if np.any(mismatch):
        index = int(np.flatnonzero(mismatch)[0])
        raise RuntimeError(
            f"{label} key mismatch at sorted row {index}: formal="
            f"({reference_user[index]}, {reference_step[index]}), candidate="
            f"({candidate_user[index]}, {candidate_step[index]})."
        )


def audit_state_targets(
    label: str,
    reference_m: np.ndarray,
    reference_psi: np.ndarray,
    reference_next_m: np.ndarray,
    reference_next_psi: np.ndarray,
    current_m: np.ndarray,
    current_psi: np.ndarray,
    target_m: np.ndarray,
    target_psi: np.ndarray,
) -> Dict[str, Any]:
    arrays = {
        "reference_M": np.asarray(reference_m, dtype=np.float64),
        "reference_Psi": np.asarray(reference_psi, dtype=np.float64),
        "reference_next_M": np.asarray(reference_next_m, dtype=np.float64),
        "reference_next_Psi": np.asarray(reference_next_psi, dtype=np.float64),
        "candidate_M": np.asarray(current_m, dtype=np.float64),
        "candidate_Psi": np.asarray(current_psi, dtype=np.float64),
        "candidate_next_M": np.asarray(target_m, dtype=np.float64),
        "candidate_next_Psi": np.asarray(target_psi, dtype=np.float64),
    }
    lengths = {name: len(value) for name, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"{label} state/target lengths differ: {lengths}")
    nonfinite = {name: int(np.sum(~np.isfinite(value))) for name, value in arrays.items()}
    failed_nonfinite = {name: count for name, count in nonfinite.items() if count > 0}
    if failed_nonfinite:
        raise RuntimeError(f"{label} state/target arrays contain non-finite values: {failed_nonfinite}")
    differences = {
        "max_abs_current_M_difference": max_abs_difference(reference_m, current_m),
        "max_abs_current_Psi_difference": max_abs_difference(reference_psi, current_psi),
        "max_abs_target_M_difference": max_abs_difference(reference_next_m, target_m),
        "max_abs_target_Psi_difference": max_abs_difference(reference_next_psi, target_psi),
    }
    if any(value > AUDIT_ATOL for value in differences.values() if np.isfinite(value)):
        raise RuntimeError(f"{label} state/target audit failed: {differences}")
    return {"label": label, **differences, "tolerance": AUDIT_ATOL}


def load_event_ssl_predictions(
    root: Path,
    split: str,
    keep_users: Optional[np.ndarray],
    require_manifest: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    base = root / "predictions" / f"stage4_event_ssl_predictions_{split}"
    columns = [
        "user_id",
        "bundle_step_index",
        "M",
        "Psi",
        "target_M_next",
        "target_Psi_next",
        "pred_next_M",
        "pred_next_Psi",
    ]
    frame = read_table(base, columns=columns)
    if keep_users is not None:
        frame = frame[pd.to_numeric(frame["user_id"], errors="coerce").isin(keep_users)].copy()
    frame = frame.sort_values(
        ["user_id", "bundle_step_index"], kind="mergesort"
    ).reset_index(drop=True)
    manifest_path = root / "metadata" / "stage4_event_ssl_evaluation_manifest.json"
    prediction_path = find_table(base).resolve()
    manifest_audit: Dict[str, Any] = {
        "evaluation_root": str(root),
        "prediction_path": str(prediction_path),
        "prediction_sha256": file_sha256(prediction_path),
        "rows_loaded": int(len(frame)),
        "users_loaded": int(frame["user_id"].nunique()),
        "manifest_required": bool(require_manifest),
    }
    if not manifest_path.exists() and require_manifest:
        raise FileNotFoundError(
            f"Formal Event-SSL evaluation manifest is missing: {manifest_path}"
        )
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest.get("primary_coordinates") != ["M", "Psi"]:
            raise RuntimeError(f"Event-SSL manifest at {root} does not use exactly M and Psi.")
        guardrails = dict(manifest.get("guardrails", {}))
        if bool(guardrails.get("B_confirm_used_for_update", False)):
            raise RuntimeError(f"Event-SSL manifest at {root} reports B_confirm model updating.")
        views = dict(manifest.get("evaluation_views", {}))
        if "empirical_anchor" not in views:
            raise RuntimeError(
                f"Event-SSL manifest at {root} does not document the empirical-anchor view."
            )
        split_manifest = dict(manifest.get("splits", {}).get(split, {}))
        if keep_users is None and split_manifest:
            if int(split_manifest.get("rows", -1)) != len(frame):
                raise RuntimeError(f"Event-SSL manifest row count differs for {split} at {root}.")
        manifest_audit.update(
            {
                "evaluation_manifest": str(manifest_path.resolve()),
                "evaluation_manifest_sha256": file_sha256(manifest_path),
                "checkpoint": manifest.get("checkpoint"),
                "checkpoint_sha256": manifest.get("checkpoint_sha256"),
                "model_kind": manifest.get("model_kind"),
                "evaluation_views": manifest.get("evaluation_views"),
            }
        )
    return frame, manifest_audit


def frozen_mechanism_predictions(
    p1: Any,
    stage1_root: Path,
    split: str,
    params: Mapping[str, float],
    calibration: Any,
    keep_users: Optional[np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    dynamics = p1.stage1_dynamics_root(stage1_root)
    raw = p1.read_core_panel(dynamics, split)
    raw_rows = int(len(raw))
    if keep_users is not None:
        raw = raw[pd.to_numeric(raw["user_id"], errors="coerce").isin(keep_users)].copy()
    panel = p1.prepare_panel(raw, split, float(calibration.eta))
    del raw
    gc.collect()
    if panel.empty:
        raise RuntimeError(f"Frozen mechanism panel is empty for {split}.")
    cache = p1.make_metric_cache(panel, [], split)
    del panel
    gc.collect()
    parameter_hash_before = stable_json_hash(params)
    calibration_hash_before = stable_json_hash(dataclasses.asdict(calibration))
    simulation = p1.simulate_arrays(cache, dict(params), calibration)
    parameter_hash_after = stable_json_hash(params)
    calibration_hash_after = stable_json_hash(dataclasses.asdict(calibration))
    if parameter_hash_before != parameter_hash_after:
        raise RuntimeError("Frozen mechanism parameters changed during evaluation.")
    if calibration_hash_before != calibration_hash_after:
        raise RuntimeError("Frozen mechanism calibration changed during evaluation.")
    arrays = {
        "user_id": np.asarray(cache.uid, dtype=np.int64),
        "step": np.asarray(cache.steps, dtype=np.int64),
        "M": np.asarray(cache.M, dtype=np.float64),
        "Psi": np.asarray(cache.Psi, dtype=np.float64),
        "target_M_next": np.asarray(cache.target_M_next, dtype=np.float64),
        "target_Psi_next": np.asarray(cache.target_Psi_next, dtype=np.float64),
        "pred_next_M": bounded_state(simulation.pred_next_M, "mechanism predicted next M"),
        "pred_next_Psi": bounded_state(
            simulation.pred_next_Psi, "mechanism predicted next Psi"
        ),
    }
    audit = {
        "split": split,
        "raw_rows": raw_rows,
        "valid_rows": int(cache.n_rows),
        "valid_users": int(cache.n_users),
        "parameter_hash_before": parameter_hash_before,
        "parameter_hash_after": parameter_hash_after,
        "calibration_hash_before": calibration_hash_before,
        "calibration_hash_after": calibration_hash_after,
        "parameter_search_opened": False,
        "calibration_reestimated": False,
        "mechanism_family_reselected": False,
        "mechanism_parameters_refit": False,
        "region_redefinition": False,
    }
    del cache
    del simulation
    gc.collect()
    return arrays, audit


# -----------------------------------------------------------------------------
# User-cell sufficient statistics and conditional user-level multiplier bootstrap
# -----------------------------------------------------------------------------
@dataclass
class GroupedUserCell:
    user_index: np.ndarray
    cell: np.ndarray
    sums: List[np.ndarray]


def group_user_cell(
    user_index: np.ndarray,
    cell: np.ndarray,
    values: Sequence[np.ndarray],
    n_cells: int,
) -> GroupedUserCell:
    user_index = np.asarray(user_index, dtype=np.int64)
    cell = np.asarray(cell, dtype=np.int64)
    if len(user_index) != len(cell):
        raise ValueError("user_index and cell lengths differ.")
    if any(len(np.asarray(value)) != len(cell) for value in values):
        raise ValueError("A contribution array has the wrong length.")
    if len(cell) == 0:
        return GroupedUserCell(
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            [np.asarray([], dtype=np.float64) for _ in values],
        )
    key = user_index * int(n_cells) + cell
    order = np.argsort(key, kind="mergesort")
    sorted_key = key[order]
    starts = np.flatnonzero(
        np.concatenate([[True], sorted_key[1:] != sorted_key[:-1]])
    )
    unique_key = sorted_key[starts]
    grouped_values = [
        np.add.reduceat(np.asarray(value, dtype=np.float64)[order], starts)
        for value in values
    ]
    return GroupedUserCell(
        user_index=(unique_key // int(n_cells)).astype(np.int64),
        cell=(unique_key % int(n_cells)).astype(np.int64),
        sums=grouped_values,
    )


def grouped_to_csr(
    grouped: GroupedUserCell,
    value_index: int,
    n_users: int,
    n_cells: int,
) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (
            np.asarray(grouped.sums[value_index], dtype=np.float64),
            (grouped.user_index, grouped.cell),
        ),
        shape=(int(n_users), int(n_cells)),
        dtype=np.float64,
    )


def occupancy_user_cell_contributions(
    frame: pd.DataFrame,
    stage1: Any,
    specification: Any,
) -> Tuple[np.ndarray, GroupedUserCell, Dict[str, Any]]:
    field_columns = ["user_id", specification.xcol, specification.ycol]
    field_frame = stage1.downcast_frame(frame[field_columns].copy())
    user_id = integer_array(field_frame, "user_id")
    all_users = np.unique(user_id)
    user_index = np.searchsorted(all_users, user_id)
    x = numeric_array(field_frame, specification.xcol)
    y = numeric_array(field_frame, specification.ycol)
    weights = np.asarray(stage1.user_balanced_weights(field_frame), dtype=np.float64)
    ix = stage1.digitize_closed_right(x, specification.xbins)
    iy = stage1.digitize_closed_right(y, specification.ybins)
    nx = len(specification.xbins) - 1
    ny = len(specification.ybins) - 1
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (ix >= 0)
        & (ix < nx)
        & (iy >= 0)
        & (iy < ny)
    )
    cell = (ix[valid] * ny + iy[valid]).astype(np.int64)
    grouped = group_user_cell(
        user_index[valid], cell, [weights[valid]], nx * ny
    )
    audit = {
        "panel_users": int(len(all_users)),
        "panel_rows": int(len(field_frame)),
        "finite_state_rows": int(np.sum(valid)),
        "unique_user_cell_occupancy_pairs": int(len(grouped.cell)),
        "weight_contract": "1/N_u computed before state-validity filtering",
    }
    return all_users, grouped, audit


def build_bootstrap_matrix(
    all_users: np.ndarray,
    occupancy_grouped: GroupedUserCell,
    prepared: Any,
    empirical_dx: np.ndarray,
    empirical_dy: np.ndarray,
    null_dx: np.ndarray,
    null_dy: np.ndarray,
    mechanism_dx: np.ndarray,
    mechanism_dy: np.ndarray,
    event_dx: np.ndarray,
    event_dy: np.ndarray,
) -> Tuple[sparse.csr_matrix, Dict[str, int], Dict[str, Any]]:
    n_users = int(len(all_users))
    shape = prepared.formal_field.drift_u.shape
    n_cells = int(np.prod(shape))
    drift_user_index = np.searchsorted(all_users, np.asarray(prepared.user_id, dtype=np.int64))
    if not np.array_equal(all_users[drift_user_index], np.asarray(prepared.user_id, dtype=np.int64)):
        raise RuntimeError("A formal drift user is absent from the full split user set.")
    weight = np.asarray(prepared.weight, dtype=np.float64)
    values = [
        weight,
        weight * np.asarray(empirical_dx, dtype=np.float64),
        weight * np.asarray(empirical_dy, dtype=np.float64),
        weight * np.asarray(null_dx, dtype=np.float64),
        weight * np.asarray(null_dy, dtype=np.float64),
        weight * np.asarray(mechanism_dx, dtype=np.float64),
        weight * np.asarray(mechanism_dy, dtype=np.float64),
        weight * np.asarray(event_dx, dtype=np.float64),
        weight * np.asarray(event_dy, dtype=np.float64),
    ]
    drift_grouped = group_user_cell(
        drift_user_index,
        np.asarray(prepared.cell, dtype=np.int64),
        values,
        n_cells,
    )

    block_names = [
        "occupancy",
        "drift_denominator",
        "empirical_u_sum",
        "empirical_v_sum",
        "null_u_sum",
        "null_v_sum",
        "mechanism_u_sum",
        "mechanism_v_sum",
        "event_u_sum",
        "event_v_sum",
    ]
    blocks = [
        grouped_to_csr(occupancy_grouped, 0, n_users, n_cells),
        *[
            grouped_to_csr(drift_grouped, index, n_users, n_cells)
            for index in range(len(values))
        ],
    ]
    matrix = sparse.hstack(blocks, format="csr", dtype=np.float64)
    block_index = {name: index for index, name in enumerate(block_names)}

    totals = np.asarray(matrix.sum(axis=0)).ravel().reshape(len(block_names), n_cells)
    occupancy = totals[block_index["occupancy"]]
    occupancy = occupancy / max(float(np.sum(occupancy)), EPS)
    denominator = np.maximum(totals[block_index["drift_denominator"]], EPS)
    reproduced = {
        "occupancy": occupancy,
        "empirical_u": totals[block_index["empirical_u_sum"]] / denominator,
        "empirical_v": totals[block_index["empirical_v_sum"]] / denominator,
        "null_u": totals[block_index["null_u_sum"]] / denominator,
        "null_v": totals[block_index["null_v_sum"]] / denominator,
        "mechanism_u": totals[block_index["mechanism_u_sum"]] / denominator,
        "mechanism_v": totals[block_index["mechanism_v_sum"]] / denominator,
        "event_u": totals[block_index["event_u_sum"]] / denominator,
        "event_v": totals[block_index["event_v_sum"]] / denominator,
    }
    audit = {
        "n_users": n_users,
        "n_cells": n_cells,
        "matrix_shape": list(matrix.shape),
        "matrix_nonzero_entries": int(matrix.nnz),
        "unique_user_cell_drift_pairs": int(len(drift_grouped.cell)),
        "block_names": block_names,
        "reproduced": reproduced,
    }
    return matrix, block_index, audit


def block_view(
    totals: np.ndarray,
    block_index: Mapping[str, int],
    name: str,
    n_cells: int,
) -> np.ndarray:
    start = int(block_index[name]) * int(n_cells)
    return np.asarray(totals[start : start + int(n_cells)], dtype=np.float64)


def bootstrap_recovery(
    matrix: sparse.csr_matrix,
    block_index: Mapping[str, int],
    n_users: int,
    n_cells: int,
    shape: Tuple[int, int],
    mask: np.ndarray,
    rescaling: Optional[ScaffoldRescalingCalibration],
    event_label: str,
    replicates: int,
    batch_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if replicates <= 0:
        return pd.DataFrame(), pd.DataFrame(), {"skipped": True}
    rng = np.random.default_rng(int(seed))
    transposed = matrix.T.tocsr()
    rows: List[dict] = []
    fixed_mask = np.asarray(mask, dtype=bool).reshape(shape)
    minimum_cells = int(np.sum(fixed_mask))

    completed = 0
    while completed < int(replicates):
        current_batch = min(int(batch_size), int(replicates) - completed)
        cluster_weights = rng.exponential(
            scale=1.0, size=(current_batch, int(n_users))
        ).astype(np.float64, copy=False)
        cluster_weights /= np.maximum(
            np.mean(cluster_weights, axis=1, keepdims=True), EPS
        )
        aggregated = (transposed @ cluster_weights.T).T
        aggregated = np.asarray(aggregated, dtype=np.float64)
        for local in range(current_batch):
            replicate = completed + local
            vector = aggregated[local]
            occupancy = block_view(vector, block_index, "occupancy", n_cells).reshape(shape)
            occupancy = occupancy / max(float(np.sum(occupancy)), EPS)
            denominator = block_view(
                vector, block_index, "drift_denominator", n_cells
            ).reshape(shape)
            positive = denominator > EPS

            def field(prefix: str) -> Tuple[np.ndarray, np.ndarray]:
                u = block_view(vector, block_index, f"{prefix}_u_sum", n_cells).reshape(shape)
                v = block_view(vector, block_index, f"{prefix}_v_sum", n_cells).reshape(shape)
                out_u = np.full(shape, np.nan, dtype=np.float64)
                out_v = np.full(shape, np.nan, dtype=np.float64)
                out_u[positive] = u[positive] / denominator[positive]
                out_v[positive] = v[positive] / denominator[positive]
                return out_u, out_v

            empirical_u, empirical_v = field("empirical")
            null_u, null_v = field("null")
            mechanism_u, mechanism_v = field("mechanism")
            event_u, event_v = field("event")
            available_mask = fixed_mask & positive
            available_cells = int(np.sum(available_mask))
            if available_cells != minimum_cells:
                raise RuntimeError(
                    f"Multiplier-bootstrap replicate {replicate} changed frozen support: "
                    f"{available_cells}/{minimum_cells} cells remain."
                )

            for model_label, model_u, model_v in (
                ("minimal_mechanism", mechanism_u, mechanism_v),
                (event_label, event_u, event_v),
            ):
                metrics = recovery_metrics(
                    split="bootstrap",
                    model_label=model_label,
                    empirical_u=empirical_u,
                    empirical_v=empirical_v,
                    null_u=null_u,
                    null_v=null_v,
                    model_u=model_u,
                    model_v=model_v,
                    weight=occupancy,
                    mask=available_mask,
                    rescaling=rescaling,
                    rescaling_role=(
                        "A_val-frozen rescaled-scaffold benchmark"
                        if rescaling is not None
                        else "not available"
                    ),
                )
                rows.append(
                    {
                        "replicate": int(replicate),
                        "model": model_label,
                        "available_frozen_support_cells": available_cells,
                        "primary_delta_sse_model_minus_exact_null": metrics[
                            "primary_delta_sse_model_minus_exact_null"
                        ],
                        "null_relative_field_skill": metrics[
                            "null_relative_field_skill"
                        ],
                        "null_normalized_rmse_ratio": metrics[
                            "null_normalized_rmse_ratio"
                        ],
                        "excess_vector_corr": metrics[
                            "null_referenced_correction_vs_empirical_excess_vector_corr"
                        ],
                        "excess_weighted_local_cosine": metrics[
                            "null_referenced_correction_vs_empirical_excess_weighted_local_cosine"
                        ],
                        "excess_amplitude_slope": metrics[
                            "null_referenced_excess_amplitude_slope"
                        ],
                        "model_minus_scalar_rescaled_scaffold_sse": metrics.get(
                            "model_minus_scalar_rescaled_scaffold_sse", np.nan
                        ),
                        "model_minus_diagonal_rescaled_scaffold_sse": metrics.get(
                            "model_minus_diagonal_rescaled_scaffold_sse", np.nan
                        ),
                    }
                )
        completed += current_batch
        print(
            f"[null-referenced multiplier bootstrap] {completed}/{replicates} replicates complete",
            flush=True,
        )

    table = pd.DataFrame(rows)
    summary_rows: List[dict] = []
    for model_label, group in table.groupby("model", sort=False):
        delta = pd.to_numeric(
            group["primary_delta_sse_model_minus_exact_null"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        delta = delta[np.isfinite(delta)]
        tail_fraction = float(np.mean(delta >= 0.0)) if delta.size else float("nan")
        row: Dict[str, Any] = {
            "model": str(model_label),
            "bootstrap_replicates": int(len(group)),
            "descriptive_bootstrap_tail_fraction_delta_ge_zero": tail_fraction,
            "bootstrap_contract": (
                "paired positive user-level multiplier bootstrap conditional on the full-split "
                "matching groups and exact row-level donor expectations"
            ),
        }
        for metric in (
            "primary_delta_sse_model_minus_exact_null",
            "null_relative_field_skill",
            "null_normalized_rmse_ratio",
            "excess_vector_corr",
            "excess_weighted_local_cosine",
            "excess_amplitude_slope",
            "model_minus_scalar_rescaled_scaffold_sse",
            "model_minus_diagonal_rescaled_scaffold_sse",
        ):
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_2p5"] = float(np.quantile(values, 0.025))
                row[f"{metric}_50"] = float(np.quantile(values, 0.50))
                row[f"{metric}_97p5"] = float(np.quantile(values, 0.975))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty and "primary_delta_sse_model_minus_exact_null_97p5" in summary.columns:
        simultaneous = bool(
            np.all(
                pd.to_numeric(
                    summary["primary_delta_sse_model_minus_exact_null_97p5"],
                    errors="coerce",
                ).to_numpy(dtype=float)
                < 0.0
            )
        )
        summary["two_model_simultaneous_97p5_upper_bounds_below_zero"] = simultaneous
    audit = {
        "skipped": False,
        "replicates": int(replicates),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "weighted_clusters": "all split-panel users, including users with state rows but no valid drift row",
        "null_reestimation_boundary": (
            "Matching groups and exact row-level donor expectations are held fixed. "
            "The bootstrap captures learner-composition uncertainty conditional on that null estimate; "
            "it does not rebuild donor pools inside each resample."
        ),
        "primary_test": (
            "paired weighted-SSE difference, model minus exact null; evidence is based on "
            "the 97.5th-percentile upper bound, not on the descriptive bootstrap tail fraction. "
            "Using 97.5% for each of the two primary models is a conservative Bonferroni-style "
            "simultaneous one-sided criterion."
        ),
        "multiplier_distribution": (
            "independent Exp(1) user-level weights, normalized to mean one within each replicate; "
            "strictly positive weights preserve the frozen field support"
        ),
    }
    return table, summary, audit


# -----------------------------------------------------------------------------
# External contract loading
# -----------------------------------------------------------------------------
@dataclass
class Contracts:
    cmn: Any
    stage1: Any
    mechanism_confirm: Any
    mechanism_phase1: Any
    stage1_script: Path
    cmn_script: Path
    mechanism_confirm_script: Path
    mechanism_phase1_script: Path
    frozen_manifest_path: Path
    frozen_manifest: Dict[str, Any]
    frozen_manifest_checksum: Dict[str, Any]
    params: Dict[str, float]
    calibration: Any
    stage1_root: Path
    audits: Dict[str, Any]


def load_contracts(args: argparse.Namespace) -> Contracts:
    cmn_script = resolve_script(
        args.construction_null_script, "run_construction_matched_null.py"
    )
    cmn = import_module(cmn_script, "null_recovery_construction_null")
    required_cmn = (
        "required_columns_for_split",
        "fit_matching_cutpoints",
        "reconstruct_innovations",
        "prepare_analysis",
        "build_matching_keys",
        "build_hierarchical_layouts",
        "aggregate_mean_field",
        "copy_field_with_drift",
        "field_geometry_metrics",
        "audit_saved_field",
        "read_table",
    )
    missing_cmn = [name for name in required_cmn if not hasattr(cmn, name)]
    if missing_cmn:
        raise RuntimeError(f"Construction-null script is missing: {missing_cmn}")

    stage1_script = (
        args.stage1_script.resolve()
        if args.stage1_script is not None
        else cmn.resolve_stage1_script(None)
    )
    stage1 = cmn.import_stage1_module(stage1_script)

    mechanism_confirm_script = resolve_script(
        args.mechanism_confirm_script, "confirm_offset_dual_channel_phase3.py"
    )
    mechanism_confirm = import_module(
        mechanism_confirm_script, "null_recovery_mechanism_confirm"
    )
    frozen_manifest_path = args.frozen_mechanism_manifest.resolve()
    checksum = mechanism_confirm.verify_manifest_checksum(
        frozen_manifest_path, bool(args.require_mechanism_manifest_checksum)
    )
    frozen_manifest = mechanism_confirm.load_json(frozen_manifest_path)
    params, calibration_payload, _ = mechanism_confirm.validate_frozen_manifest(
        frozen_manifest
    )
    mechanism_phase1_script, phase1_audit = mechanism_confirm.resolve_phase1_script(
        args.mechanism_phase1_script, frozen_manifest
    )
    mechanism_confirm.prepare_runtime_cache(mechanism_phase1_script)
    mechanism_phase1 = mechanism_confirm.import_phase1_module(mechanism_phase1_script)
    mechanism_confirm.validate_phase1_module_contract(mechanism_phase1)
    mechanism_confirm.configure_frozen_phase1_module(mechanism_phase1, params)
    calibration = mechanism_confirm.calibration_from_manifest(
        mechanism_phase1, calibration_payload
    )

    stage1_root = args.stage1_root.resolve()
    manifest_stage1 = str(frozen_manifest.get("stage1_root", "") or "").strip()
    if manifest_stage1 and Path(manifest_stage1).resolve() != stage1_root:
        raise RuntimeError(
            "The requested Stage-1 root differs from the Phase-2 frozen mechanism manifest."
        )
    kmeans_current = mechanism_phase1.audit_stage1_kmeans_contract(stage1_root)
    mechanism_confirm.compare_kmeans_contracts(
        dict(frozen_manifest.get("stage1_fixed_k6_contract", {})), kmeans_current
    )

    audits = {
        "construction_null_script_sha256": file_sha256(cmn_script),
        "stage1_script_sha256": file_sha256(stage1_script),
        "mechanism_confirm_script_sha256": file_sha256(mechanism_confirm_script),
        "mechanism_phase1_implementation": phase1_audit,
        "frozen_mechanism_manifest_checksum": checksum,
        "frozen_parameter_hash": stable_json_hash(params),
        "frozen_calibration_hash": stable_json_hash(dataclasses.asdict(calibration)),
        "stage1_fixed_k6_contract": kmeans_current,
    }
    return Contracts(
        cmn=cmn,
        stage1=stage1,
        mechanism_confirm=mechanism_confirm,
        mechanism_phase1=mechanism_phase1,
        stage1_script=stage1_script,
        cmn_script=cmn_script,
        mechanism_confirm_script=mechanism_confirm_script,
        mechanism_phase1_script=mechanism_phase1_script,
        frozen_manifest_path=frozen_manifest_path,
        frozen_manifest=frozen_manifest,
        frozen_manifest_checksum=checksum,
        params=params,
        calibration=calibration,
        stage1_root=stage1_root,
        audits=audits,
    )


# -----------------------------------------------------------------------------
# Existing construction-null contract audits
# -----------------------------------------------------------------------------
def compare_matching_cutpoints(current: Any, archived: Mapping[str, Any]) -> Dict[str, Any]:
    current_payload = json_safe(current)
    archived_payload = dict(archived)
    array_fields = (
        "log_a_m",
        "log_a_psi",
        "support_share",
        "idle_share",
        "sequence_length",
    )
    maximum_differences: Dict[str, float] = {}
    failures: Dict[str, Any] = {}
    for field in array_fields:
        if field not in current_payload or field not in archived_payload:
            failures[field] = "missing"
            continue
        first = np.asarray(current_payload[field], dtype=np.float64)
        second = np.asarray(archived_payload[field], dtype=np.float64)
        if first.shape != second.shape:
            failures[field] = {"current_shape": list(first.shape), "archived_shape": list(second.shape)}
            continue
        difference = float(np.max(np.abs(first - second))) if first.size else 0.0
        maximum_differences[field] = difference
        if not np.allclose(first, second, atol=1e-12, rtol=1e-12, equal_nan=True):
            failures[field] = difference
    for field in ("fit_rows_sampled", "fit_users"):
        if int(current_payload.get(field, -1)) != int(archived_payload.get(field, -2)):
            failures[field] = {
                "current": current_payload.get(field),
                "archived": archived_payload.get(field),
            }
    if str(current_payload.get("fit_split", "")) != str(archived_payload.get("fit_split", "")):
        failures["fit_split"] = {
            "current": current_payload.get("fit_split"),
            "archived": archived_payload.get("fit_split"),
        }
    return {
        "match": not failures,
        "maximum_absolute_differences": maximum_differences,
        "failures": failures,
        "current_contract_hash": stable_json_hash(current_payload),
        "archived_contract_hash": stable_json_hash(archived_payload),
    }


def audit_existing_cmn_contract(
    root: Optional[Path],
    splits: Sequence[str],
    contracts: Contracts,
    args: argparse.Namespace,
    cutpoints: Any,
) -> Dict[str, Any]:
    required = bool(args.require_existing_construction_null_output)
    if root is None:
        if required:
            raise FileNotFoundError(
                "A construction-null output root is required for the formal analysis."
            )
        return {"available": False, "required": False, "reason": "no output root supplied"}
    root = root.resolve()
    if not root.exists():
        if required:
            raise FileNotFoundError(f"Construction-null output root does not exist: {root}")
        return {"available": False, "required": False, "reason": f"missing {root}"}

    metadata_root = root / "metadata"
    cutpoint_path = metadata_root / "matching_cutpoints_A_train.json"
    if not cutpoint_path.exists():
        if required:
            raise FileNotFoundError(f"Published matching-cutpoint audit is missing: {cutpoint_path}")
        return {"available": False, "required": False, "reason": f"missing {cutpoint_path}"}
    existing_cutpoint_payload = load_json(cutpoint_path)
    existing_cutpoints = existing_cutpoint_payload.get("cutpoints", existing_cutpoint_payload)
    cutpoint_comparison = compare_matching_cutpoints(cutpoints, existing_cutpoints)
    if not bool(cutpoint_comparison["match"]):
        raise RuntimeError(
            "Recomputed A_train matching cutpoints differ from the published construction-null contract: "
            f"{cutpoint_comparison['failures']}"
        )
    current_cutpoint_hash = str(cutpoint_comparison["current_contract_hash"])

    core_path = (
        contracts.stage1_root
        / "dynamics"
        / "candidate_regions"
        / PRIMARY_COORDINATE
        / "A_train_primary_convergence_core_mask.npy"
    )
    thresholds_path = (
        contracts.stage1_root
        / "dynamics"
        / "candidate_regions"
        / PRIMARY_COORDINATE
        / "training_convergence_thresholds.json"
    )
    current_core_sha = file_sha256(core_path)
    current_thresholds_sha = file_sha256(thresholds_path)
    current_stage1_sha = file_sha256(contracts.stage1_script)
    manifest_audits: Dict[str, Any] = {}
    missing_manifests: List[str] = []
    for split in splits:
        if split == "B_confirm":
            confirm_root = Path(str(root) + "_confirm")
            split_metadata_root = confirm_root / "metadata"
        else:
            split_metadata_root = metadata_root
        manifest_path = split_metadata_root / f"{split}_construction_null_manifest.json"
        if not manifest_path.exists():
            missing_manifests.append(str(manifest_path))
            continue
        manifest = load_json(manifest_path)
        failures: Dict[str, Any] = {}
        if str(manifest.get("analysis_split", "")) != str(split):
            failures["analysis_split"] = manifest.get("analysis_split")
        if int(manifest.get("base_seed", -1)) != int(args.seed):
            failures["base_seed"] = manifest.get("base_seed")
        if str(manifest.get("formal_stage1_script_sha256", "")) != current_stage1_sha:
            failures["formal_stage1_script_sha256"] = manifest.get(
                "formal_stage1_script_sha256"
            )
        if list(manifest.get("primary_coordinates", [])) != ["M", "Psi"]:
            failures["primary_coordinates"] = manifest.get("primary_coordinates")
        if str(manifest.get("frozen_core_sha256", "")) != current_core_sha:
            failures["frozen_core_sha256"] = manifest.get("frozen_core_sha256")
        if str(manifest.get("frozen_thresholds_sha256", "")) != current_thresholds_sha:
            failures["frozen_thresholds_sha256"] = manifest.get(
                "frozen_thresholds_sha256"
            )
        manifest_cutpoints = manifest.get("matching_cutpoints")
        if manifest_cutpoints is not None:
            manifest_cutpoint_comparison = compare_matching_cutpoints(
                cutpoints, manifest_cutpoints
            )
            if not bool(manifest_cutpoint_comparison["match"]):
                failures["matching_cutpoints"] = manifest_cutpoint_comparison["failures"]
        cutpoint_audit = dict(manifest.get("matching_cutpoint_audit", {}))
        if cutpoint_audit and int(cutpoint_audit.get("max_sample_rows", -1)) != int(
            args.matching_fit_max_rows
        ):
            failures["matching_fit_max_rows"] = cutpoint_audit.get("max_sample_rows")
        if split == "B_confirm" and not bool(manifest.get("confirmation_output_only", False)):
            failures["confirmation_output_only"] = manifest.get(
                "confirmation_output_only"
            )
        if failures:
            raise RuntimeError(
                f"Published construction-null manifest differs for {split}: {failures}"
            )
        manifest_audits[split] = {
            "path": str(manifest_path.resolve()),
            "sha256": file_sha256(manifest_path),
            "replicates": int(manifest.get("replicates", 0)),
            "base_seed": int(manifest.get("base_seed", -1)),
            "contract_match": True,
        }

    if missing_manifests and required:
        raise FileNotFoundError(
            "Published construction-null manifests are missing: " + "; ".join(missing_manifests)
        )
    return {
        "available": True,
        "required": required,
        "output_root": str(root),
        "matching_cutpoints_path": str(cutpoint_path.resolve()),
        "matching_cutpoints_sha256": file_sha256(cutpoint_path),
        "matching_cutpoints_contract_hash": current_cutpoint_hash,
        "matching_cutpoints_comparison": cutpoint_comparison,
        "formal_stage1_script_sha256": current_stage1_sha,
        "frozen_core_sha256": current_core_sha,
        "frozen_thresholds_sha256": current_thresholds_sha,
        "manifests": manifest_audits,
        "missing_manifests": missing_manifests,
    }


# -----------------------------------------------------------------------------
# Existing finite-replicate null audit
# -----------------------------------------------------------------------------
def audit_existing_null_arrays(
    cmn_output_root: Optional[Path],
    split: str,
    empirical_u: np.ndarray,
    empirical_v: np.ndarray,
    exact_null_u: np.ndarray,
    exact_null_v: np.ndarray,
    mask: np.ndarray,
    weight: np.ndarray,
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
    if cmn_output_root is None:
        return None, {"available": False, "reason": "no construction-null output root supplied"}
    path = cmn_output_root.resolve() / "arrays" / f"{split}_construction_null_fields.npz"
    if not path.exists():
        return None, {"available": False, "reason": f"missing {path}"}
    with np.load(path) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    saved_mask = np.asarray(arrays["drift_mask"], dtype=bool)
    if not np.array_equal(saved_mask, np.asarray(mask, dtype=bool)):
        raise RuntimeError(f"Existing construction-null mask differs for {split}.")
    observed_u_diff = max_abs_difference(arrays["observed_u"], empirical_u)
    observed_v_diff = max_abs_difference(arrays["observed_v"], empirical_v)
    if max(observed_u_diff, observed_v_diff) > 1e-10:
        raise RuntimeError(
            f"Existing construction-null observed field differs for {split}: "
            f"M={observed_u_diff:.3e}, Psi={observed_v_diff:.3e}."
        )
    mc_mean_u = np.asarray(arrays["null_mean_u"], dtype=np.float64)
    mc_mean_v = np.asarray(arrays["null_mean_v"], dtype=np.float64)
    exact_to_mc_sse = weighted_field_sse(
        exact_null_u, exact_null_v, mc_mean_u, mc_mean_v, weight, mask
    )
    replicate_count = int(np.asarray(arrays.get("null_u", np.empty((0,)))).shape[0])
    standardized_parts: List[np.ndarray] = []
    if replicate_count > 1 and "null_sd_u" in arrays and "null_sd_v" in arrays:
        se_u = np.asarray(arrays["null_sd_u"], dtype=np.float64) / math.sqrt(float(replicate_count))
        se_v = np.asarray(arrays["null_sd_v"], dtype=np.float64) / math.sqrt(float(replicate_count))
        valid_u = np.asarray(mask, dtype=bool) & np.isfinite(se_u) & (se_u > EPS)
        valid_v = np.asarray(mask, dtype=bool) & np.isfinite(se_v) & (se_v > EPS)
        if np.any(valid_u):
            standardized_parts.append(np.abs(exact_null_u[valid_u] - mc_mean_u[valid_u]) / se_u[valid_u])
        if np.any(valid_v):
            standardized_parts.append(np.abs(exact_null_v[valid_v] - mc_mean_v[valid_v]) / se_v[valid_v])
    standardized = np.concatenate(standardized_parts) if standardized_parts else np.asarray([], dtype=np.float64)
    return arrays, {
        "available": True,
        "path": str(path.resolve()),
        "replicates": replicate_count,
        "max_abs_observed_M_difference": observed_u_diff,
        "max_abs_observed_Psi_difference": observed_v_diff,
        "max_abs_exact_vs_MC_mean_M": max_abs_difference(exact_null_u, mc_mean_u),
        "max_abs_exact_vs_MC_mean_Psi": max_abs_difference(exact_null_v, mc_mean_v),
        "occupancy_weighted_exact_vs_MC_mean_RMSE": math.sqrt(max(exact_to_mc_sse, 0.0)),
        "exact_vs_MC_mean_standardized_difference_median": (
            float(np.median(standardized)) if standardized.size else None
        ),
        "exact_vs_MC_mean_standardized_difference_95th_percentile": (
            float(np.quantile(standardized, 0.95)) if standardized.size else None
        ),
        "exact_vs_MC_mean_standardized_difference_maximum": (
            float(np.max(standardized)) if standardized.size else None
        ),
        "standardization": "finite-null field standard deviation divided by sqrt(number of replicates)",
        "role": (
            "The exact expectation is used for decomposition. Existing finite replicates remain "
            "the source of the original construction-null Monte Carlo tests."
        ),
    }




def audit_existing_null_protocol(
    cmn_output_root: Optional[Path],
    split: str,
    cutpoints: Any,
    coverage_table: pd.DataFrame,
    base_seed: int,
    stage1_script: Path,
    smoke_test: bool,
) -> Dict[str, Any]:
    """Verify that the rebuilt exact expectation uses the archived formal protocol."""
    if cmn_output_root is None:
        return {"available": False, "reason": "no construction-null output root supplied"}
    root = cmn_output_root.resolve()
    manifest_path = root / "metadata" / f"{split}_construction_null_manifest.json"
    if not manifest_path.exists():
        return {"available": False, "reason": f"missing {manifest_path}"}
    manifest = load_json(manifest_path)
    if str(manifest.get("analysis_split", "")) != split:
        raise RuntimeError(f"Archived construction-null manifest split differs for {split}.")
    if int(manifest.get("base_seed", -1)) != int(base_seed):
        raise RuntimeError(
            f"Archived construction-null seed={manifest.get('base_seed')} differs from requested seed={base_seed}."
        )
    archived_stage1_sha = str(manifest.get("formal_stage1_script_sha256", "") or "")
    current_stage1_sha = file_sha256(stage1_script)
    if archived_stage1_sha and archived_stage1_sha != current_stage1_sha:
        raise RuntimeError("Formal Stage-1 script differs from the archived construction-null protocol.")
    archived_cutpoints = manifest.get("matching_cutpoints", {})
    cutpoint_comparison = compare_matching_cutpoints(cutpoints, archived_cutpoints)
    if not bool(cutpoint_comparison["match"]):
        raise RuntimeError(
            "Recomputed A_train matching cutpoints differ from the archived construction-null protocol: "
            f"{cutpoint_comparison['failures']}"
        )

    coverage_audit: Dict[str, Any]
    if smoke_test:
        coverage_audit = {
            "skipped": True,
            "reason": "a user-subset smoke test changes split-specific group coverage",
        }
    else:
        raw_coverage = str(manifest.get("matching_fallback_coverage_table", "") or "")
        archived_coverage_path = Path(raw_coverage) if raw_coverage else None
        if archived_coverage_path is None or not archived_coverage_path.exists():
            archived_coverage_path = root / "tables" / f"{split}_matching_fallback_coverage"
        archived = read_table(archived_coverage_path)
        columns = ["level", "matching_keys", "rows_assigned", "rows_remaining_after_level"]
        if any(column not in archived.columns or column not in coverage_table.columns for column in columns):
            raise RuntimeError("Archived or rebuilt matching coverage lacks required audit columns.")
        left = coverage_table[columns].reset_index(drop=True)
        right = archived[columns].reset_index(drop=True)
        exact = (
            len(left) == len(right)
            and left["level"].astype(str).equals(right["level"].astype(str))
            and left["matching_keys"].astype(str).equals(right["matching_keys"].astype(str))
            and np.array_equal(
                pd.to_numeric(left["rows_assigned"], errors="coerce").to_numpy(dtype=np.int64),
                pd.to_numeric(right["rows_assigned"], errors="coerce").to_numpy(dtype=np.int64),
            )
            and np.array_equal(
                pd.to_numeric(left["rows_remaining_after_level"], errors="coerce").to_numpy(dtype=np.int64),
                pd.to_numeric(right["rows_remaining_after_level"], errors="coerce").to_numpy(dtype=np.int64),
            )
        )
        if not exact:
            raise RuntimeError("Rebuilt matching-group coverage differs from the archived formal construction-null run.")
        coverage_path = find_table(archived_coverage_path)
        coverage_audit = {
            "skipped": False,
            "exact_match": True,
            "path": str(coverage_path.resolve()),
            "sha256": file_sha256(coverage_path),
            "levels": int(len(left)),
        }
    return {
        "available": True,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "analysis_split": split,
        "base_seed_exact_match": True,
        "formal_stage1_sha256_exact_match": True,
        "matching_cutpoints_exact_match": True,
        "matching_cutpoints_comparison": cutpoint_comparison,
        "matching_coverage": coverage_audit,
    }


# -----------------------------------------------------------------------------
# Split execution
# -----------------------------------------------------------------------------
def sample_split_users(
    frame: pd.DataFrame,
    max_users: int,
    seed: int,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    if max_users <= 0:
        return frame, None
    users = np.asarray(
        sorted(pd.to_numeric(frame["user_id"], errors="coerce").dropna().astype(np.int64).unique())
    )
    if len(users) <= max_users:
        return frame, users
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(users, size=int(max_users), replace=False))
    output = frame[pd.to_numeric(frame["user_id"], errors="coerce").isin(selected)].copy()
    return output, selected


def fit_cutpoints_once(
    contracts: Contracts,
    args: argparse.Namespace,
    metadata_root: Path,
) -> Any:
    train_base = contracts.stage1_root / "dynamics" / "student_dynamics_panel_core_A_train"
    train_columns, _ = contracts.cmn.required_columns_for_split(train_base)
    cutpoints, audit = contracts.cmn.fit_matching_cutpoints(
        train_base=train_base,
        train_columns=train_columns,
        tau_response_days=float(contracts.stage1.TAU_RESPONSE_DAYS),
        tau_activity_days=float(contracts.stage1.TAU_ACTIVITY_DAYS),
        max_sample_rows=int(args.matching_fit_max_rows),
        chunk_rows=int(args.read_chunk_rows),
        seed=int(args.seed),
    )
    save_json(
        {"cutpoints": cutpoints, "audit": audit},
        metadata_root / "matching_cutpoints_A_train.json",
    )
    return cutpoints


def run_split(
    split: str,
    contracts: Contracts,
    args: argparse.Namespace,
    event_roots: Mapping[str, Path],
    primary_event_label: str,
    cutpoints: Any,
    output_root: Path,
    rescaling: Optional[ScaffoldRescalingCalibration],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Optional[ScaffoldRescalingCalibration],
    Dict[str, Any],
]:
    started = time.time()
    cmn_audit_root = args.construction_null_output_root
    if split == "B_confirm" and cmn_audit_root is not None:
        cmn_audit_root = Path(str(cmn_audit_root) + "_confirm")
    cmn = contracts.cmn
    stage1 = contracts.stage1
    p1 = contracts.mechanism_phase1
    table_root = output_root / "tables"
    array_root = output_root / "arrays"
    metadata_root = output_root / "metadata"
    for directory in (table_root, array_root, metadata_root):
        directory.mkdir(parents=True, exist_ok=True)

    analysis_base = (
        contracts.stage1_root / "dynamics" / f"student_dynamics_panel_core_{split}"
    )
    analysis_columns, aliases = cmn.required_columns_for_split(analysis_base)
    frame = cmn.read_table(analysis_base, columns=analysis_columns)
    frame, keep_users = sample_split_users(
        frame, int(args.max_users), int(args.seed) + (11 if split == "A_val" else 23)
    )
    frame = frame.sort_values(
        ["user_id", "bundle_step_index"], kind="mergesort"
    ).reset_index(drop=True)
    panel_rows = int(len(frame))
    panel_users = int(frame["user_id"].nunique())

    innovations = cmn.reconstruct_innovations(
        frame,
        tau_response_days=float(stage1.TAU_RESPONSE_DAYS),
        tau_activity_days=float(stage1.TAU_ACTIVITY_DAYS),
        require_next_audit=True,
    )
    specification = stage1.coordinate_specs()[0]
    if specification.name != PRIMARY_COORDINATE:
        raise RuntimeError(
            f"Expected coordinate {PRIMARY_COORDINATE}, found {specification.name}."
        )
    prepared = cmn.prepare_analysis(frame, innovations, stage1, specification)
    reference_step = np.asarray(innovations.step[prepared.drift_row_indices], dtype=np.int64)
    reference_user = np.asarray(prepared.user_id, dtype=np.int64)
    reference_m = np.asarray(prepared.x, dtype=np.float64)
    reference_psi = np.asarray(prepared.y, dtype=np.float64)
    reference_next_m = reference_m + np.asarray(prepared.observed_dx, dtype=np.float64)
    reference_next_psi = reference_psi + np.asarray(prepared.observed_dy, dtype=np.float64)

    saved_field_audit = cmn.audit_saved_field(
        contracts.stage1_root,
        split,
        prepared.formal_field,
        skip=bool(args.max_users > 0),
    )
    all_users, occupancy_grouped, occupancy_audit = occupancy_user_cell_contributions(
        frame, stage1, specification
    )
    del frame
    gc.collect()

    keys, randomizable, composition_audit = cmn.build_matching_keys(prepared, cutpoints)
    layouts, coverage_table, effective_randomizable = cmn.build_hierarchical_layouts(
        keys,
        randomizable,
        seed=int(args.seed) + 200003,
        max_last_resort_fraction=float(args.max_last_resort_fraction),
    )
    existing_protocol_audit = audit_existing_null_protocol(
        cmn_audit_root,
        split,
        cutpoints,
        coverage_table,
        int(args.seed),
        contracts.stage1_script,
        smoke_test=bool(args.max_users > 0),
    )
    if bool(args.require_existing_construction_null_output) and not bool(
        existing_protocol_audit.get("available", False)
    ):
        raise RuntimeError(
            f"Published construction-null protocol audit is unavailable for {split}: "
            f"{existing_protocol_audit}"
        )
    del keys
    del randomizable
    gc.collect()

    expectation = exact_expected_donor_pairs(
        layouts,
        effective_randomizable,
        np.asarray(prepared.z_m, dtype=np.float64),
        np.asarray(prepared.z_psi, dtype=np.float64),
    )
    null_dx, null_dy, exact_null_row_audit = exact_null_increments(
        prepared, expectation
    )
    exact_null_u, exact_null_v = cmn.aggregate_mean_field(prepared, null_dx, null_dy)
    empirical_u = np.asarray(prepared.formal_field.drift_u, dtype=np.float64)
    empirical_v = np.asarray(prepared.formal_field.drift_v, dtype=np.float64)
    mask = np.asarray(prepared.formal_field.drift_mask, dtype=bool)
    occupancy_weight = np.asarray(
        prepared.formal_field.occupancy_probability, dtype=np.float64
    )

    if int(args.max_users) > 0:
        existing_null_arrays = None
        existing_null_audit = {
            "available": False,
            "skipped": True,
            "reason": "smoke-test user subset is not comparable with full-split archived arrays",
        }
    else:
        existing_null_arrays, existing_null_audit = audit_existing_null_arrays(
            cmn_audit_root,
            split,
            empirical_u,
            empirical_v,
            exact_null_u,
            exact_null_v,
            mask,
            occupancy_weight,
        )
        if bool(args.require_existing_construction_null_output) and not bool(
            existing_null_audit.get("available", False)
        ):
            raise FileNotFoundError(
                f"The archived finite-replicate construction-null arrays are required for {split}: "
                f"{existing_null_audit}"
            )

    if split == "A_val":
        fitted = fit_scaffold_rescaling(
            empirical_u,
            empirical_v,
            exact_null_u,
            exact_null_v,
            occupancy_weight,
            mask,
            source_split="A_val",
        )
        if rescaling is not None:
            differences = {
                "scalar_alpha": abs(fitted.scalar_alpha - rescaling.scalar_alpha),
                "alpha_M": abs(fitted.alpha_M - rescaling.alpha_M),
                "alpha_Psi": abs(fitted.alpha_Psi - rescaling.alpha_Psi),
            }
            if max(differences.values()) > 1e-12:
                raise RuntimeError(
                    f"Loaded and recomputed A_val scaffold-rescaling calibration differ: {differences}"
                )
        rescaling = fitted
        save_json(
            {"calibration": fitted},
            metadata_root / "A_val_frozen_scaffold_rescaling_calibration.json",
        )
        rescaling_role = "A_val in-sample amplitude-only scaffold fit; frozen for later B_confirm use"
    elif rescaling is not None:
        rescaling_role = "A_val-frozen rescaled-scaffold benchmark applied without B_confirm refitting"
    else:
        rescaling_role = "not available because A_val scaffold calibration was not supplied"

    mechanism_arrays, mechanism_audit = frozen_mechanism_predictions(
        p1,
        contracts.stage1_root,
        split,
        contracts.params,
        contracts.calibration,
        keep_users,
    )
    assert_exact_alignment(
        reference_user,
        reference_step,
        mechanism_arrays["user_id"],
        mechanism_arrays["step"],
        "frozen mechanism",
    )
    mechanism_state_audit = audit_state_targets(
        "frozen mechanism",
        reference_m,
        reference_psi,
        reference_next_m,
        reference_next_psi,
        mechanism_arrays["M"],
        mechanism_arrays["Psi"],
        mechanism_arrays["target_M_next"],
        mechanism_arrays["target_Psi_next"],
    )
    mechanism_dx = mechanism_arrays["pred_next_M"] - reference_m
    mechanism_dy = mechanism_arrays["pred_next_Psi"] - reference_psi
    mechanism_u, mechanism_v = cmn.aggregate_mean_field(
        prepared, mechanism_dx, mechanism_dy
    )
    del mechanism_arrays
    gc.collect()

    model_fields: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "minimal_mechanism": (
            np.asarray(mechanism_u, dtype=np.float64),
            np.asarray(mechanism_v, dtype=np.float64),
        )
    }
    event_audits: Dict[str, Any] = {}
    primary_event_dx: Optional[np.ndarray] = None
    primary_event_dy: Optional[np.ndarray] = None

    for event_label, root in event_roots.items():
        event_frame, event_manifest_audit = load_event_ssl_predictions(
            root, split, keep_users, bool(args.require_event_ssl_manifest)
        )
        event_user = integer_array(event_frame, "user_id")
        event_step = integer_array(event_frame, "bundle_step_index")
        assert_exact_alignment(
            reference_user,
            reference_step,
            event_user,
            event_step,
            f"Event-SSL {event_label}",
        )
        state_audit = audit_state_targets(
            f"Event-SSL {event_label}",
            reference_m,
            reference_psi,
            reference_next_m,
            reference_next_psi,
            numeric_array(event_frame, "M"),
            numeric_array(event_frame, "Psi"),
            numeric_array(event_frame, "target_M_next"),
            numeric_array(event_frame, "target_Psi_next"),
        )
        pred_next_m = bounded_state(
            numeric_array(event_frame, "pred_next_M"),
            f"Event-SSL {event_label} predicted next M",
        )
        pred_next_psi = bounded_state(
            numeric_array(event_frame, "pred_next_Psi"),
            f"Event-SSL {event_label} predicted next Psi",
        )
        event_dx = pred_next_m - reference_m
        event_dy = pred_next_psi - reference_psi
        event_u, event_v = cmn.aggregate_mean_field(prepared, event_dx, event_dy)
        model_fields[event_label] = (
            np.asarray(event_u, dtype=np.float64),
            np.asarray(event_v, dtype=np.float64),
        )
        event_audits[event_label] = {
            "manifest": event_manifest_audit,
            "state_target_alignment": state_audit,
        }
        if event_label == primary_event_label:
            model_kind = str(event_manifest_audit.get("model_kind", "") or "")
            if bool(args.require_event_ssl_manifest) and model_kind != "predictive_state":
                raise RuntimeError(
                    f"Primary Event-SSL root {event_label!r} has model_kind={model_kind!r}; "
                    "the primary downstream test requires the frozen predictive_state model."
                )
            primary_event_dx = np.asarray(event_dx, dtype=np.float64)
            primary_event_dy = np.asarray(event_dy, dtype=np.float64)
        del event_frame
        del event_dx
        del event_dy
        del pred_next_m
        del pred_next_psi
        gc.collect()

    if primary_event_dx is None or primary_event_dy is None:
        raise RuntimeError(
            f"Primary Event-SSL label {primary_event_label!r} was not evaluated."
        )

    point_rows: List[Dict[str, Any]] = []
    for model_label, (model_u, model_v) in model_fields.items():
        point_rows.append(
            recovery_metrics(
                split,
                model_label,
                empirical_u,
                empirical_v,
                exact_null_u,
                exact_null_v,
                model_u,
                model_v,
                occupancy_weight,
                mask,
                rescaling,
                rescaling_role,
            )
        )

    cross_rows: List[Dict[str, Any]] = []
    mechanism_field = model_fields["minimal_mechanism"]
    for model_label, (model_u, model_v) in model_fields.items():
        if model_label == "minimal_mechanism":
            continue
        cross_rows.append(
            pairwise_field_metrics(
                split,
                "minimal_mechanism",
                model_label,
                mechanism_field[0],
                mechanism_field[1],
                model_u,
                model_v,
                exact_null_u,
                exact_null_v,
                empirical_u,
                empirical_v,
                occupancy_weight,
                mask,
            )
        )

    core_mask_path = (
        contracts.stage1_root
        / "dynamics"
        / "candidate_regions"
        / PRIMARY_COORDINATE
        / "A_train_primary_convergence_core_mask.npy"
    )
    thresholds_path = (
        contracts.stage1_root
        / "dynamics"
        / "candidate_regions"
        / PRIMARY_COORDINATE
        / "training_convergence_thresholds.json"
    )
    core_mask = np.load(core_mask_path).astype(bool)
    thresholds = load_json(thresholds_path)
    shell_radius = float(thresholds["shell_radius"])
    geometry_rows: List[dict] = []
    geometry_fields: Dict[str, Tuple[np.ndarray, np.ndarray, str]] = {
        "empirical": (empirical_u, empirical_v, "raw empirical field"),
        "exact_construction_null": (
            exact_null_u,
            exact_null_v,
            "exact expectation of matched construction null",
        ),
        "empirical_excess": (
            empirical_u - exact_null_u,
            empirical_v - exact_null_v,
            "empirical minus exact construction null",
        ),
    }
    for model_label, (model_u, model_v) in model_fields.items():
        geometry_fields[f"{model_label}_raw"] = (
            model_u,
            model_v,
            "raw empirical-anchor model field",
        )
        geometry_fields[f"{model_label}_null_referenced_correction"] = (
            model_u - exact_null_u,
            model_v - exact_null_v,
            "model field minus exact construction null",
        )
    for field_label, (field_u, field_v, role) in geometry_fields.items():
        field = cmn.copy_field_with_drift(prepared.formal_field, field_u, field_v)
        metrics = cmn.field_geometry_metrics(
            stage1,
            field,
            core_mask,
            f"{split}_{field_label}",
            shell_radius,
        )
        geometry_rows.append(
            {
                "split": split,
                "field": field_label,
                "role": role,
                **metrics,
            }
        )

    grid = stage1.field_grid_table(prepared.formal_field, split).copy()
    grid["exact_null_drift_M"] = exact_null_u.ravel(order="C")
    grid["exact_null_drift_Psi"] = exact_null_v.ravel(order="C")
    grid["empirical_excess_drift_M"] = (empirical_u - exact_null_u).ravel(order="C")
    grid["empirical_excess_drift_Psi"] = (empirical_v - exact_null_v).ravel(order="C")
    for model_label, (model_u, model_v) in model_fields.items():
        safe = sanitize_label(model_label)
        grid[f"{safe}_drift_M"] = model_u.ravel(order="C")
        grid[f"{safe}_drift_Psi"] = model_v.ravel(order="C")
        grid[f"{safe}_null_referenced_correction_M"] = (
            model_u - exact_null_u
        ).ravel(order="C")
        grid[f"{safe}_null_referenced_correction_Psi"] = (
            model_v - exact_null_v
        ).ravel(order="C")
        grid[f"{safe}_empirical_residual_magnitude"] = np.hypot(
            model_u - empirical_u, model_v - empirical_v
        ).ravel(order="C")
    grid_path = write_table(
        grid, table_root / f"{split}_null_referenced_field_grid"
    )
    point_path = write_table(
        pd.DataFrame(point_rows),
        table_root / f"{split}_null_referenced_recovery_metrics",
    )
    cross_path = write_table(
        pd.DataFrame(cross_rows),
        table_root / f"{split}_cross_model_null_referenced_metrics_descriptive",
    )
    geometry_path = write_table(
        pd.DataFrame(geometry_rows),
        table_root / f"{split}_null_referenced_geometry_metrics",
    )
    coverage_path = write_table(
        coverage_table,
        table_root / f"{split}_matching_fallback_coverage",
    )
    composition_path = write_table(
        composition_audit,
        table_root / f"{split}_opportunity_composition_audit",
    )

    npz_payload: Dict[str, np.ndarray] = {
        "empirical_u": empirical_u,
        "empirical_v": empirical_v,
        "exact_null_u": exact_null_u,
        "exact_null_v": exact_null_v,
        "empirical_excess_u": empirical_u - exact_null_u,
        "empirical_excess_v": empirical_v - exact_null_v,
        "drift_mask": mask,
        "state_mask": np.asarray(prepared.formal_field.state_mask, dtype=bool),
        "occupancy_probability": occupancy_weight,
        "core_mask": core_mask,
        "xcenters": np.asarray(prepared.formal_field.xcenters, dtype=np.float64),
        "ycenters": np.asarray(prepared.formal_field.ycenters, dtype=np.float64),
    }
    for model_label, (model_u, model_v) in model_fields.items():
        safe = sanitize_label(model_label)
        npz_payload[f"{safe}_u"] = model_u
        npz_payload[f"{safe}_v"] = model_v
        npz_payload[f"{safe}_correction_u"] = model_u - exact_null_u
        npz_payload[f"{safe}_correction_v"] = model_v - exact_null_v
    array_path = array_root / f"{split}_null_referenced_fields.npz"
    np.savez_compressed(array_path, **npz_payload)

    bootstrap_table = pd.DataFrame()
    bootstrap_summary = pd.DataFrame()
    bootstrap_audit: Dict[str, Any] = {"skipped": True}
    sparse_reproduction_audit: Dict[str, Any] = {"skipped": True}
    if split in set(args.bootstrap_splits) and int(args.bootstrap_replicates) > 0:
        matrix, block_index, matrix_audit = build_bootstrap_matrix(
            all_users,
            occupancy_grouped,
            prepared,
            np.asarray(prepared.observed_dx, dtype=np.float64),
            np.asarray(prepared.observed_dy, dtype=np.float64),
            null_dx,
            null_dy,
            mechanism_dx,
            mechanism_dy,
            primary_event_dx,
            primary_event_dy,
        )
        reproduced = matrix_audit.pop("reproduced")
        sparse_reproduction_audit = {
            **matrix_audit,
            "max_abs_occupancy_difference": max_abs_difference(
                reproduced["occupancy"].reshape(mask.shape), occupancy_weight
            ),
            "max_abs_empirical_M_difference": max_abs_difference(
                reproduced["empirical_u"].reshape(mask.shape), empirical_u
            ),
            "max_abs_empirical_Psi_difference": max_abs_difference(
                reproduced["empirical_v"].reshape(mask.shape), empirical_v
            ),
            "max_abs_null_M_difference": max_abs_difference(
                reproduced["null_u"].reshape(mask.shape), exact_null_u
            ),
            "max_abs_null_Psi_difference": max_abs_difference(
                reproduced["null_v"].reshape(mask.shape), exact_null_v
            ),
            "max_abs_mechanism_M_difference": max_abs_difference(
                reproduced["mechanism_u"].reshape(mask.shape), mechanism_field[0]
            ),
            "max_abs_mechanism_Psi_difference": max_abs_difference(
                reproduced["mechanism_v"].reshape(mask.shape), mechanism_field[1]
            ),
            "max_abs_event_M_difference": max_abs_difference(
                reproduced["event_u"].reshape(mask.shape), model_fields[primary_event_label][0]
            ),
            "max_abs_event_Psi_difference": max_abs_difference(
                reproduced["event_v"].reshape(mask.shape), model_fields[primary_event_label][1]
            ),
        }
        if max(
            value
            for key, value in sparse_reproduction_audit.items()
            if key.startswith("max_abs_") and np.isfinite(value)
        ) > 1e-10:
            raise RuntimeError(
                f"Sparse user-cell sufficient statistics do not reproduce point fields: "
                f"{sparse_reproduction_audit}"
            )
        bootstrap_table, bootstrap_summary, bootstrap_audit = bootstrap_recovery(
            matrix=matrix,
            block_index=block_index,
            n_users=len(all_users),
            n_cells=int(np.prod(mask.shape)),
            shape=mask.shape,
            mask=mask,
            rescaling=rescaling,
            event_label=primary_event_label,
            replicates=int(args.bootstrap_replicates),
            batch_size=int(args.bootstrap_batch_size),
            seed=int(args.bootstrap_seed) + (0 if split == "A_val" else 100003),
        )
        write_table(
            bootstrap_table,
            table_root / f"{split}_user_multiplier_bootstrap_replicates",
        )
        write_table(
            bootstrap_summary,
            table_root / f"{split}_user_multiplier_bootstrap_summary",
        )
        del matrix
        gc.collect()

    baseline_sensitivity_rows: List[dict] = []
    if existing_null_arrays is not None:
        mc_u = np.asarray(existing_null_arrays["null_mean_u"], dtype=np.float64)
        mc_v = np.asarray(existing_null_arrays["null_mean_v"], dtype=np.float64)
        for model_label, (model_u, model_v) in model_fields.items():
            exact_metrics = recovery_metrics(
                split,
                model_label,
                empirical_u,
                empirical_v,
                exact_null_u,
                exact_null_v,
                model_u,
                model_v,
                occupancy_weight,
                mask,
                None,
                "not used",
            )
            mc_metrics = recovery_metrics(
                split,
                model_label,
                empirical_u,
                empirical_v,
                mc_u,
                mc_v,
                model_u,
                model_v,
                occupancy_weight,
                mask,
                None,
                "not used",
            )
            baseline_sensitivity_rows.extend(
                [
                    {
                        "split": split,
                        "model": model_label,
                        "null_baseline": "exact_cyclic_shift_expectation",
                        "primary_delta_sse_model_minus_null": exact_metrics[
                            "primary_delta_sse_model_minus_exact_null"
                        ],
                        "null_relative_field_skill": exact_metrics[
                            "null_relative_field_skill"
                        ],
                        "excess_vector_corr": exact_metrics[
                            "null_referenced_correction_vs_empirical_excess_vector_corr"
                        ],
                    },
                    {
                        "split": split,
                        "model": model_label,
                        "null_baseline": "existing_finite_replicate_mean",
                        "primary_delta_sse_model_minus_null": mc_metrics[
                            "primary_delta_sse_model_minus_exact_null"
                        ],
                        "null_relative_field_skill": mc_metrics[
                            "null_relative_field_skill"
                        ],
                        "excess_vector_corr": mc_metrics[
                            "null_referenced_correction_vs_empirical_excess_vector_corr"
                        ],
                    },
                ]
            )
        write_table(
            pd.DataFrame(baseline_sensitivity_rows),
            table_root / f"{split}_exact_vs_finite_null_mean_sensitivity",
        )

    split_audit = {
        "split": split,
        "created_at": now_string(),
        "runtime_seconds": float(time.time() - started),
        "panel_rows_before_drift_filter": panel_rows,
        "panel_users_before_drift_filter": panel_users,
        "formal_drift_rows": int(len(prepared.x)),
        "formal_drift_users": int(pd.Series(prepared.user_id).nunique()),
        "smoke_test_max_users": int(args.max_users),
        "resolved_activity_aliases": aliases,
        "stage1_reconstruction_audit": innovations.reconstruction_audit,
        "saved_stage1_field_audit": saved_field_audit,
        "occupancy_user_cell_audit": occupancy_audit,
        "exact_null_expectation_audit": expectation.audit,
        "exact_null_row_audit": exact_null_row_audit,
        "existing_construction_null_protocol_audit": existing_protocol_audit,
        "existing_finite_null_audit": existing_null_audit,
        "mechanism_audit": mechanism_audit,
        "mechanism_state_target_alignment": mechanism_state_audit,
        "event_ssl_audits": event_audits,
        "sparse_bootstrap_reproduction_audit": sparse_reproduction_audit,
        "bootstrap_audit": bootstrap_audit,
        "scaffold_rescaling_calibration": rescaling,
        "scaffold_rescaling_role": rescaling_role,
        "outputs": {
            "point_metrics": str(point_path),
            "cross_model_descriptive_metrics": str(cross_path),
            "geometry_metrics": str(geometry_path),
            "field_grid": str(grid_path),
            "field_arrays": str(array_path),
            "matching_coverage": str(coverage_path),
            "opportunity_composition": str(composition_path),
        },
        "guardrails": {
            "coordinates_refit": False,
            "grid_or_support_refit": False,
            "construction_null_matching_cutpoints_refit_outside_A_train": False,
            "construction_null_groups_redefined_from_model_outputs": False,
            "mechanism_family_reselected": False,
            "mechanism_parameters_refit": False,
            "mechanism_calibration_reestimated": False,
            "event_ssl_retrained": False,
            "learned_plane_subtracted_from_empirical_null": False,
            "model_rows_intersected_silently": False,
            "B_confirm_used_for_model_update": False,
        },
    }
    save_json(split_audit, metadata_root / f"{split}_null_referenced_recovery_audit.json")

    del innovations
    del prepared
    del occupancy_grouped
    del primary_event_dx
    del primary_event_dy
    del mechanism_dx
    del mechanism_dy
    gc.collect()
    return point_rows, cross_rows, rescaling, split_audit


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------
def run_self_test(cmn: Any) -> None:
    n = 9
    indices = np.arange(n, dtype=np.int64)
    group_key = np.asarray([0, 0, 0, 1, 1, 1, 1, 2, 2], dtype=np.int16)
    layout, remaining = cmn.make_layout(
        "self_test", indices, [group_key], layout_seed=1234
    )
    if layout is None or len(remaining):
        raise AssertionError("Synthetic layout construction failed.")
    z_m = np.linspace(-0.8, 0.8, n)
    z_psi = np.cos(np.arange(n)) * 0.5
    exact = exact_expected_donor_pairs([layout], np.ones(n, dtype=bool), z_m, z_psi)

    brute_m = np.zeros(n, dtype=float)
    brute_p = np.zeros(n, dtype=float)
    brute_product = np.zeros(n, dtype=float)
    sorted_rows = layout.original_indices[layout.order]
    for start, count in zip(layout.starts, layout.counts):
        positions = np.arange(start, start + count)
        recipients = sorted_rows[positions]
        for shift in range(1, int(count)):
            donors = sorted_rows[start + ((positions - start + shift) % count)]
            brute_m[recipients] += z_m[donors]
            brute_p[recipients] += z_psi[donors]
            brute_product[recipients] += z_m[donors] * z_psi[donors]
        brute_m[recipients] /= count - 1
        brute_p[recipients] /= count - 1
        brute_product[recipients] /= count - 1
    if not np.allclose(exact.expected_z_m, brute_m, atol=1e-14, rtol=0):
        raise AssertionError("Exact Z_M expectation failed brute-force enumeration.")
    if not np.allclose(exact.expected_z_psi, brute_p, atol=1e-14, rtol=0):
        raise AssertionError("Exact Z_Psi expectation failed brute-force enumeration.")
    if not np.allclose(
        exact.expected_pair_product, brute_product, atol=1e-14, rtol=0
    ):
        raise AssertionError("Exact joint-pair expectation failed brute-force enumeration.")

    shape = (2, 2)
    mask = np.ones(shape, dtype=bool)
    weight = np.full(shape, 0.25)
    null_u = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    null_v = -0.5 * null_u
    excess_u = np.asarray([[0.01, -0.02], [0.03, -0.01]])
    excess_v = np.asarray([[0.02, 0.01], [-0.01, 0.03]])
    empirical_u = null_u + excess_u
    empirical_v = null_v + excess_v
    perfect = recovery_metrics(
        "test",
        "perfect",
        empirical_u,
        empirical_v,
        null_u,
        null_v,
        empirical_u,
        empirical_v,
        weight,
        mask,
        None,
        "none",
    )
    null_model = recovery_metrics(
        "test",
        "null",
        empirical_u,
        empirical_v,
        null_u,
        null_v,
        null_u,
        null_v,
        weight,
        mask,
        None,
        "none",
    )
    if not np.isclose(perfect["null_relative_field_skill"], 1.0):
        raise AssertionError("Perfect model did not attain unit null-relative field skill.")
    if not np.isclose(null_model["null_relative_field_skill"], 0.0):
        raise AssertionError("Null model did not yield zero null-relative field skill.")

    anti = fit_scaffold_rescaling(
        -null_u,
        -null_v,
        null_u,
        null_v,
        weight,
        mask,
        source_split="test",
    )
    if min(anti.scalar_alpha, anti.alpha_M, anti.alpha_Psi) < 0.0:
        raise AssertionError("Rescaled-scaffold coefficients must be non-negative.")

    block_names = [
        "occupancy",
        "drift_denominator",
        "empirical_u_sum",
        "empirical_v_sum",
        "null_u_sum",
        "null_v_sum",
        "mechanism_u_sum",
        "mechanism_v_sum",
        "event_u_sum",
        "event_v_sum",
    ]
    n_users = 3
    n_cells = 4
    dense = np.zeros((n_users, len(block_names) * n_cells), dtype=float)
    for user in range(n_users):
        dense[user, 0 * n_cells : 1 * n_cells] = 1.0 / n_cells
        dense[user, 1 * n_cells : 2 * n_cells] = 1.0
        dense[user, 2 * n_cells : 3 * n_cells] = empirical_u.ravel()
        dense[user, 3 * n_cells : 4 * n_cells] = empirical_v.ravel()
        dense[user, 4 * n_cells : 5 * n_cells] = null_u.ravel()
        dense[user, 5 * n_cells : 6 * n_cells] = null_v.ravel()
        dense[user, 6 * n_cells : 7 * n_cells] = empirical_u.ravel()
        dense[user, 7 * n_cells : 8 * n_cells] = empirical_v.ravel()
        dense[user, 8 * n_cells : 9 * n_cells] = empirical_u.ravel()
        dense[user, 9 * n_cells : 10 * n_cells] = empirical_v.ravel()
    bootstrap_table, bootstrap_summary, bootstrap_audit = bootstrap_recovery(
        matrix=sparse.csr_matrix(dense),
        block_index={name: index for index, name in enumerate(block_names)},
        n_users=n_users,
        n_cells=n_cells,
        shape=shape,
        mask=mask,
        rescaling=None,
        event_label="event_test",
        replicates=4,
        batch_size=2,
        seed=17,
    )
    if len(bootstrap_table) != 8 or len(bootstrap_summary) != 2:
        raise AssertionError("Multiplier-bootstrap self-test produced unexpected output sizes.")
    if bootstrap_audit.get("skipped") is not False:
        raise AssertionError("Multiplier-bootstrap self-test was unexpectedly skipped.")
    print(
        "[self-test] exact expectation, metric algebra and multiplier bootstrap passed",
        flush=True,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen downstream recovery relative to the construction-matched null."
    )
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--stage1-script", type=Path, default=None)
    parser.add_argument("--construction-null-script", type=Path, default=None)
    parser.add_argument(
        "--construction-null-output-root",
        type=Path,
        default=DEFAULT_CMN_OUTPUT_ROOT,
        help=("Published 100-replicate construction-null output root used for contract "
              "verification and exact-vs-finite-mean audit."),
    )
    parser.add_argument(
        "--require-existing-construction-null-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the published construction-null manifests, matching cutpoints, "
            "fallback tables and finite-replicate arrays to match this reconstruction."
        ),
    )
    parser.add_argument("--mechanism-confirm-script", type=Path, default=None)
    parser.add_argument("--mechanism-phase1-script", type=Path, default=None)
    parser.add_argument(
        "--frozen-mechanism-manifest",
        type=Path,
        default=DEFAULT_FROZEN_MECHANISM_MANIFEST,
    )
    parser.add_argument(
        "--require-mechanism-manifest-checksum",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--event-ssl-root",
        action="append",
        default=[],
        metavar="LABEL=ROOT",
        help=(
            "May be repeated. ROOT must contain predictions/"
            "stage4_event_ssl_predictions_{split}.[parquet|csv.gz|csv]."
        ),
    )
    parser.add_argument(
        "--require-event-ssl-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require each Event-SSL prediction root to contain its formal evaluation manifest.",
    )
    parser.add_argument("--primary-event-ssl-label", type=str, required=False)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["A_val", "B_confirm"],
        default=["A_val", "B_confirm"],
    )
    parser.add_argument(
        "--confirmation-output-only",
        action="store_true",
        help="Required whenever B_confirm is requested.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--matching-fit-max-rows", type=int, default=500000)
    parser.add_argument("--read-chunk-rows", type=int, default=250000)
    parser.add_argument("--max-last-resort-fraction", type=float, default=0.01)
    parser.add_argument(
        "--scaffold-rescaling-calibration",
        "--null-rescaling-calibration",
        dest="scaffold_rescaling_calibration",
        type=Path,
        default=None,
        help=(
            "Optional previously frozen A_val non-negative rescaled-scaffold "
            "calibration JSON; the legacy option name is accepted as an alias."
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-batch-size", type=int, default=20)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--bootstrap-splits",
        nargs="+",
        choices=["A_val", "B_confirm"],
        default=["B_confirm"],
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="Positive values are smoke-test only and skip archived-field equality.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def load_scaffold_rescaling_calibration(path: Optional[Path]) -> Optional[ScaffoldRescalingCalibration]:
    if path is None:
        return None
    payload = load_json(path.resolve())
    source = payload.get("calibration", payload)
    calibration = ScaffoldRescalingCalibration(
        source_split=str(source["source_split"]),
        scalar_alpha=float(source["scalar_alpha"]),
        alpha_M=float(source["alpha_M"]),
        alpha_Psi=float(source["alpha_Psi"]),
        fitting_contract=str(source["fitting_contract"]),
    )
    if calibration.source_split != "A_val":
        raise RuntimeError(
            "A supplied rescaled-scaffold calibration must have been fitted on A_val."
        )
    coefficients = np.asarray(
        [calibration.scalar_alpha, calibration.alpha_M, calibration.alpha_Psi],
        dtype=float,
    )
    if not np.isfinite(coefficients).all() or np.any(coefficients < 0.0):
        raise RuntimeError(
            "A supplied rescaled-scaffold calibration has non-finite or negative coefficients."
        )
    return calibration


def main() -> None:
    args = build_parser().parse_args()
    cmn_script = resolve_script(
        args.construction_null_script, "run_construction_matched_null.py"
    )
    if args.self_test:
        cmn = import_module(cmn_script, "null_recovery_self_test_cmn")
        run_self_test(cmn)
        return

    if "B_confirm" in args.splits and not args.confirmation_output_only:
        raise RuntimeError(
            "B_confirm requires --confirmation-output-only after the analysis protocol is frozen."
        )
    if args.matching_fit_max_rows < 10000:
        raise ValueError("--matching-fit-max-rows must be at least 10000.")
    if not 0.0 <= args.max_last_resort_fraction <= 1.0:
        raise ValueError("--max-last-resort-fraction must lie in [0,1].")
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates cannot be negative.")
    if args.bootstrap_batch_size < 1:
        raise ValueError("--bootstrap-batch-size must be positive.")

    event_roots = parse_named_roots(args.event_ssl_root)
    if args.primary_event_ssl_label is None:
        if len(event_roots) != 1:
            raise ValueError(
                "Pass --primary-event-ssl-label when more than one Event-SSL root is supplied."
            )
        primary_event_label = next(iter(event_roots))
    else:
        primary_event_label = sanitize_label(args.primary_event_ssl_label)
    if primary_event_label not in event_roots:
        raise ValueError(
            f"Primary Event-SSL label {primary_event_label!r} is absent from --event-ssl-root."
        )
    if "minimal_mechanism" in event_roots:
        raise ValueError("The Event-SSL label 'minimal_mechanism' is reserved.")

    started = time.time()
    output_root = args.output_root.resolve()
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    for directory in (output_root, table_root, metadata_root, output_root / "arrays"):
        directory.mkdir(parents=True, exist_ok=True)

    contracts = load_contracts(args)
    cutpoints = fit_cutpoints_once(contracts, args, metadata_root)
    rescaling = load_scaffold_rescaling_calibration(args.scaffold_rescaling_calibration)

    requested = set(args.splits)
    ordered_splits = [split for split in ("A_val", "B_confirm") if split in requested]
    existing_cmn_contract_audit = audit_existing_cmn_contract(
        args.construction_null_output_root,
        ordered_splits,
        contracts,
        args,
        cutpoints,
    )
    save_json(
        existing_cmn_contract_audit,
        metadata_root / "existing_construction_null_contract_audit.json",
    )
    contracts.audits["existing_construction_null_contract"] = existing_cmn_contract_audit
    all_point_rows: List[Dict[str, Any]] = []
    all_cross_rows: List[Dict[str, Any]] = []
    split_audits: Dict[str, Any] = {}
    for split in ordered_splits:
        print(f"[null-referenced recovery] starting split={split}", flush=True)
        point_rows, cross_rows, rescaling, split_audit = run_split(
            split,
            contracts,
            args,
            event_roots,
            primary_event_label,
            cutpoints,
            output_root,
            rescaling,
        )
        all_point_rows.extend(point_rows)
        all_cross_rows.extend(cross_rows)
        split_audits[split] = split_audit
        print(f"[null-referenced recovery] completed split={split}", flush=True)

    combined_point_path = write_table(
        pd.DataFrame(all_point_rows), table_root / "null_referenced_recovery_metrics_all_splits"
    )
    combined_cross_path = write_table(
        pd.DataFrame(all_cross_rows),
        table_root / "cross_model_null_referenced_metrics_all_splits_descriptive",
    )
    if rescaling is not None:
        save_json(
            {"calibration": rescaling},
            metadata_root / "final_frozen_scaffold_rescaling_calibration.json",
        )

    manifest = {
        "script": Path(__file__).name,
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "created_at": now_string(),
        "runtime_seconds": float(time.time() - started),
        "output_root": str(output_root),
        "analysis_name": "construction-null-referenced downstream excess-field recovery",
        "splits": ordered_splits,
        "primary_event_ssl_label": primary_event_label,
        "event_ssl_roots": event_roots,
        "contracts": contracts.audits,
        "existing_construction_null_contract_audit": existing_cmn_contract_audit,
        "stage1_root": str(contracts.stage1_root),
        "construction_null_script": str(contracts.cmn_script),
        "stage1_script": str(contracts.stage1_script),
        "frozen_mechanism_manifest": str(contracts.frozen_manifest_path),
        "mechanism_phase1_script": str(contracts.mechanism_phase1_script),
        "point_metrics_table": str(combined_point_path),
        "cross_model_descriptive_table": str(combined_cross_path),
        "scaffold_rescaling_calibration": rescaling,
        "primary_endpoint": (
            "occupancy-weighted paired squared-field-error difference: "
            "SSE(model, empirical) - SSE(exact construction null, empirical)"
        ),
        "success_rule": (
            "For each primary model, the B_confirm point estimate and the 97.5th-percentile "
            "upper bound from the paired user-level multiplier bootstrap should be below zero. Recovery against the A_val-frozen "
            "per-coordinate rescaled-scaffold benchmark is a stronger secondary specificity check."
        ),
        "inference_boundary": (
            "This is a post-freeze reviewer-motivated secondary analysis. B_confirm remains "
            "held out from model fitting and is output-only, but the derived analysis itself "
            "is not described as preregistered."
        ),
        "bootstrap_boundary": (
            "User-level multiplier-bootstrap uncertainty is conditional on matching groups and exact donor "
            "expectations estimated once from the full split. Existing finite null replicates "
            "continue to support the original permutation tests."
        ),
        "guardrails": {
            "model_training": False,
            "model_selection": False,
            "mechanism_refit": False,
            "coordinate_refit": False,
            "grid_or_mask_refit": False,
            "construction_null_protocol_changed": False,
            "learned_plane_used_for_null_subtraction": False,
            "cross_model_correction_correlation_used_as_primary_test": False,
            "B_confirm_used_for_update": False,
        },
        "split_audits": {
            split: str(
                metadata_root / f"{split}_null_referenced_recovery_audit.json"
            )
            for split in ordered_splits
        },
        "smoke_test_max_users": int(args.max_users),
        "visualization_outputs": "none; figures should be generated separately from saved grids and arrays",
    }
    save_json(manifest, metadata_root / "null_referenced_recovery_manifest.json")
    print(f"[null-referenced recovery] completed: {output_root}", flush=True)


if __name__ == "__main__":
    main()
