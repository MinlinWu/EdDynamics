#!/usr/bin/env python3
"""Audit empirical excess-field reliability and convergence-core selection stability."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

EPS = 1e-12
PRIMARY_COORDINATE = "MR_PsiA"
DEFAULT_STAGE1_ROOT = Path("/data/datasets/KT4/outputs_KT4/stage1")
DEFAULT_CONSTRUCTION_NULL_ROOT = Path(
    "/data/datasets/KT4/outputs_KT4/stage1_construction_matched_null"
)
DEFAULT_CONSTRUCTION_NULL_CONFIRM_ROOT = Path(
    "/data/datasets/KT4/outputs_KT4/stage1_construction_matched_null_confirm"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/datasets/KT4/outputs_KT4/empirical_excess_reliability_soft_core"
)
DEFAULT_STAGE1_SCRIPT = "build_effective_dynamics_kt4_stage1_empirical.py"
DEFAULT_CONSTRUCTION_NULL_SCRIPT = "run_construction_matched_null.py"
FORMAL_EXCESS_CELLS = 917
FORMAL_EXCESS_VECTOR_CORR = 0.7164
FORMAL_EXCESS_SPEED_CORR = 0.7362
FORMAL_EXCESS_LOCAL_COSINE = 0.9604
EXPECTED_FORMAL_CONSTRUCTION_NULL_SHA256 = "c0b4149a65a7ba155950914ef0936d31b1812d30d4ce9f8dad2ec5d02636d0f9"
AUDIT_ATOL = 1e-10


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
        json_safe(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def import_module(path: Path, name: str) -> Any:
    source = path.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location(name, str(source))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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
    raise FileNotFoundError(f"Pass the path to {sibling_name}; no sibling copy was found.")



def validate_module_contracts(stage1: Any, cmn: Any) -> None:
    stage1_required = (
        "coordinate_specs",
        "occupancy_drift_stats",
        "field_stats_from_dict",
        "identify_convergence_regions",
        "convergence_region_reproducibility",
        "user_balanced_weights",
        "digitize_closed_right",
        "downcast_frame",
        "FieldStats",
        "MIN_DRIFT_BIN_COUNT",
        "MIN_CELL_USERS",
    )
    cmn_required = (
        "required_columns_for_split",
        "reconstruct_innovations",
        "prepare_analysis",
        "audit_saved_field",
        "build_matching_keys",
        "build_hierarchical_layouts",
        "aggregate_mean_field",
        "MatchingCutpoints",
    )
    missing_stage1 = [name for name in stage1_required if not hasattr(stage1, name)]
    missing_cmn = [name for name in cmn_required if not hasattr(cmn, name)]
    if missing_stage1 or missing_cmn:
        raise RuntimeError(
            "Formal implementation contract is incomplete: "
            f"Stage-1={missing_stage1}, construction-null={missing_cmn}."
        )

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


def vector_correlation(
    first_u: np.ndarray,
    first_v: np.ndarray,
    second_u: np.ndarray,
    second_v: np.ndarray,
    mask: np.ndarray,
) -> float:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(first_u)
        & np.isfinite(first_v)
        & np.isfinite(second_u)
        & np.isfinite(second_v)
    )
    if int(np.sum(valid)) < 3:
        return float("nan")
    first = np.column_stack([np.asarray(first_u)[valid], np.asarray(first_v)[valid]]).ravel()
    second = np.column_stack([np.asarray(second_u)[valid], np.asarray(second_v)[valid]]).ravel()
    return pearson(first, second)


def weighted_local_cosine(
    first_u: np.ndarray,
    first_v: np.ndarray,
    second_u: np.ndarray,
    second_v: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float]:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(first_u)
        & np.isfinite(first_v)
        & np.isfinite(second_u)
        & np.isfinite(second_v)
    )
    first_speed = np.hypot(first_u, first_v)
    second_speed = np.hypot(second_u, second_v)
    valid &= (first_speed > EPS) & (second_speed > EPS)
    all_weight = np.where(np.asarray(mask, dtype=bool), np.maximum(weights, 0.0), 0.0)
    coverage = float(np.sum(np.asarray(weights)[valid]) / max(float(np.sum(all_weight)), EPS))
    if not np.any(valid):
        return float("nan"), coverage
    cosine = (
        np.asarray(first_u)[valid] * np.asarray(second_u)[valid]
        + np.asarray(first_v)[valid] * np.asarray(second_v)[valid]
    ) / (first_speed[valid] * second_speed[valid])
    w = np.maximum(np.asarray(weights, dtype=np.float64)[valid], 0.0)
    w = w / max(float(np.sum(w)), EPS)
    return float(np.sum(w * np.clip(cosine, -1.0, 1.0))), coverage


def field_agreement(
    first_u: np.ndarray,
    first_v: np.ndarray,
    second_u: np.ndarray,
    second_v: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, float]:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(first_u)
        & np.isfinite(first_v)
        & np.isfinite(second_u)
        & np.isfinite(second_v)
    )
    local_cosine, coverage = weighted_local_cosine(
        first_u, first_v, second_u, second_v, weights, valid
    )
    return {
        "supported_cells": int(np.sum(valid)),
        "vector_correlation": vector_correlation(
            first_u, first_v, second_u, second_v, valid
        ),
        "speed_correlation": pearson(
            np.hypot(np.asarray(first_u)[valid], np.asarray(first_v)[valid]),
            np.hypot(np.asarray(second_u)[valid], np.asarray(second_v)[valid]),
        ),
        "M_component_correlation": pearson(
            np.asarray(first_u)[valid], np.asarray(second_u)[valid]
        ),
        "Psi_component_correlation": pearson(
            np.asarray(first_v)[valid], np.asarray(second_v)[valid]
        ),
        "weighted_local_cosine": local_cosine,
        "local_cosine_weight_coverage": coverage,
    }


def spearman_brown(half_correlation: float) -> float:
    value = float(half_correlation)
    if not np.isfinite(value) or value <= -1.0:
        return float("nan")
    return float(2.0 * value / (1.0 + value))


def quantile_summary(values: Sequence[float], prefix: str) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_2p5": np.nan,
            f"{prefix}_50": np.nan,
            f"{prefix}_97p5": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_n": int(array.size),
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_2p5": float(np.quantile(array, 0.025)),
        f"{prefix}_50": float(np.quantile(array, 0.50)),
        f"{prefix}_97p5": float(np.quantile(array, 0.975)),
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
    }


def stable_uint64(values: np.ndarray, seed: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint64) ^ np.uint64(int(seed) & ((1 << 64) - 1))
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


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
    z_m = np.asarray(z_m, dtype=np.float64)
    z_psi = np.asarray(z_psi, dtype=np.float64)
    randomizable = np.asarray(effective_randomizable, dtype=bool)
    expected_m = z_m.copy()
    expected_psi = z_psi.copy()
    expected_product = (z_m * z_psi).copy()
    assigned = np.zeros(len(z_m), dtype=bool)
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
        zm = z_m[sorted_rows]
        zp = z_psi[sorted_rows]
        product = zm * zp
        sum_m = np.add.reduceat(zm, starts)
        sum_p = np.add.reduceat(zp, starts)
        sum_product = np.add.reduceat(product, starts)
        denominator = counts[group_id] - 1
        expected_m[sorted_rows] = (sum_m[group_id] - zm) / denominator
        expected_psi[sorted_rows] = (sum_p[group_id] - zp) / denominator
        expected_product[sorted_rows] = (
            sum_product[group_id] - product
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
        raise RuntimeError(
            "Exact-expectation assignment differs from the frozen randomizable mask."
        )
    randomized_rows = np.flatnonzero(randomizable)
    moment_errors: Dict[str, float] = {}
    if randomized_rows.size:
        pairs = (
            ("Z_M", z_m, expected_m),
            ("Z_Psi", z_psi, expected_psi),
            ("pair_product", z_m * z_psi, expected_product),
        )
        for label, before, after in pairs:
            error = float(
                abs(np.mean(before[randomized_rows]) - np.mean(after[randomized_rows]))
            )
            moment_errors[f"mean_{label}_preservation_error"] = error
            if error > 1e-10:
                raise RuntimeError(
                    f"Exact donor expectation changed the {label} randomized-row mean."
                )
    return ExactNullExpectation(
        expected_z_m=expected_m,
        expected_z_psi=expected_psi,
        expected_pair_product=expected_product,
        audit={
            "rows": int(len(z_m)),
            "randomized_rows": int(np.sum(randomizable)),
            "self_mapped_rows": int(np.sum(~randomizable)),
            "layouts": layout_rows,
            "total_groups": int(len(group_sizes)),
            "minimum_group_size": int(min(group_sizes)) if group_sizes else None,
            "median_group_size": float(np.median(group_sizes)) if group_sizes else None,
            "maximum_group_size": int(max(group_sizes)) if group_sizes else None,
            "preservation_scope": "randomized-row first moments of Z_M, Z_Psi and their paired product; not full marginal-distribution preservation",
            **moment_errors,
        },
    )


def exact_null_increments(prepared: Any, expectation: ExactNullExpectation) -> Tuple[np.ndarray, np.ndarray]:
    next_m = (
        np.asarray(prepared.s_pre, dtype=np.float64)
        + np.asarray(prepared.a_m, dtype=np.float64) * expectation.expected_z_m
    ) / np.asarray(prepared.m_denominator, dtype=np.float64)
    next_psi = (
        np.asarray(prepared.g_pre, dtype=np.float64)
        + np.asarray(prepared.a_psi, dtype=np.float64) * expectation.expected_z_psi
    ) / np.asarray(prepared.psi_denominator, dtype=np.float64)
    if not np.isfinite(next_m).all() or not np.isfinite(next_psi).all():
        raise RuntimeError("Exact construction-null next states contain non-finite values.")
    excess = max(
        float(np.max(np.maximum(np.abs(next_m) - 1.0, 0.0))),
        float(np.max(np.maximum(np.abs(next_psi) - 1.0, 0.0))),
    )
    if excess > 2e-6:
        raise RuntimeError(f"Exact construction-null next state left [-1,1] by {excess:.3e}.")
    next_m = np.clip(next_m, -1.0, 1.0)
    next_psi = np.clip(next_psi, -1.0, 1.0)
    return next_m - prepared.x, next_psi - prepared.y


def subset_innovations(innovations: Any, mask: np.ndarray) -> Any:
    mask = np.asarray(mask, dtype=bool)
    row_count = len(mask)
    payload: Dict[str, Any] = {}
    for field in dataclasses.fields(innovations):
        value = getattr(innovations, field.name)
        if isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) == row_count:
            payload[field.name] = value[mask]
        elif field.name == "reconstruction_audit":
            payload[field.name] = {
                "source": "full-split reconstruction",
                "subset_rows": int(np.sum(mask)),
            }
        else:
            payload[field.name] = value
    return type(innovations)(**payload)


def load_matching_cutpoints(cmn: Any, root: Path) -> Tuple[Any, Dict[str, Any], Path]:
    path = root / "metadata" / "matching_cutpoints_A_train.json"
    if path.exists():
        payload = load_json(path)
        values = dict(payload.get("cutpoints", {}))
    else:
        path = root / "metadata" / "A_val_construction_null_manifest.json"
        payload = load_json(path)
        values = dict(payload.get("matching_cutpoints", {}))
    if not values:
        raise RuntimeError("The formal construction-null output has no matching cutpoints.")
    cutpoints = cmn.MatchingCutpoints(
        log_a_m=np.asarray(values["log_a_m"], dtype=np.float64),
        log_a_psi=np.asarray(values["log_a_psi"], dtype=np.float64),
        support_share=np.asarray(values["support_share"], dtype=np.float64),
        idle_share=np.asarray(values["idle_share"], dtype=np.float64),
        sequence_length=np.asarray(values["sequence_length"], dtype=np.float64),
        fit_rows_sampled=int(values["fit_rows_sampled"]),
        fit_users=int(values["fit_users"]),
        fit_split=str(values.get("fit_split", "A_train")),
    )
    return cutpoints, {"cutpoints": values, "source": str(path.resolve())}, path


def load_cmn_manifest(root: Path, split: str) -> Tuple[Dict[str, Any], Path]:
    path = root / "metadata" / f"{split}_construction_null_manifest.json"
    return load_json(path), path


def load_finite_null_arrays(root: Path, split: str) -> Tuple[Dict[str, np.ndarray], Path]:
    path = root / "arrays" / f"{split}_construction_null_fields.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return arrays, path


def drift_user_count(prepared: Any) -> np.ndarray:
    n_cells = int(np.prod(prepared.formal_field.drift_u.shape))
    key = np.asarray(prepared.user_id, dtype=np.int64) * n_cells + np.asarray(
        prepared.cell, dtype=np.int64
    )
    unique_key = np.unique(key)
    return np.bincount(unique_key % n_cells, minlength=n_cells).reshape(
        prepared.formal_field.drift_u.shape
    )


def activity_cutpoints(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    return np.unique(np.quantile(values, np.arange(0.1, 1.0, 0.1)))


def assign_complementary_halves(
    users: np.ndarray,
    activity: np.ndarray,
    cutpoints: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    users = np.asarray(users, dtype=np.int64)
    activity = np.asarray(activity, dtype=np.float64)
    bins = np.searchsorted(np.asarray(cutpoints, dtype=np.float64), activity, side="right")
    side = np.full(len(users), -1, dtype=np.int8)
    rows: List[dict] = []
    for activity_bin in np.unique(bins):
        indices = np.flatnonzero(bins == activity_bin)
        priority = stable_uint64(users[indices], seed + 1009 * (int(activity_bin) + 1))
        ordered = indices[np.argsort(priority, kind="mergesort")]
        side[ordered[::2]] = 0
        side[ordered[1::2]] = 1
        rows.append(
            {
                "activity_bin": int(activity_bin),
                "users": int(len(indices)),
                "half_0_users": int(np.sum(side[indices] == 0)),
                "half_1_users": int(np.sum(side[indices] == 1)),
                "activity_min": float(np.min(activity[indices])) if len(indices) else np.nan,
                "activity_max": float(np.max(activity[indices])) if len(indices) else np.nan,
            }
        )
    if np.any(side < 0):
        raise RuntimeError("Some users were not assigned to a complementary half.")
    return side, pd.DataFrame(rows)


@dataclass
class ReliabilitySplitResult:
    split: str
    full_observed_u: np.ndarray
    full_observed_v: np.ndarray
    full_exact_null_u: np.ndarray
    full_exact_null_v: np.ndarray
    full_exact_excess_u: np.ndarray
    full_exact_excess_v: np.ndarray
    full_occupancy: np.ndarray
    drift_mask: np.ndarray
    drift_count: np.ndarray
    drift_user_count: np.ndarray
    finite_arrays: Dict[str, np.ndarray]
    half_excess_u: np.ndarray
    half_excess_v: np.ndarray
    half_drift_count: np.ndarray
    half_drift_user_count: np.ndarray
    partition_balance: pd.DataFrame
    partition_bin_balance: pd.DataFrame
    matching_quality: pd.DataFrame
    formal_audit: Dict[str, Any]


def coverage_metrics(table: pd.DataFrame) -> Dict[str, float]:
    result: Dict[str, float] = {}
    randomized_levels = table[table["level"] != "deterministic_zero_innovation"]
    total = float(
        pd.to_numeric(randomized_levels["rows_assigned"], errors="coerce").fillna(0).sum()
    )
    for label in (
        "within_user_fine",
        "within_user_coarse",
        "across_user_fine",
        "across_user_coarse",
        "global_opportunity",
        "global_last_resort",
        "unmatched_singleton_self_exempt",
    ):
        value = float(
            pd.to_numeric(
                table.loc[table["level"] == label, "rows_assigned"], errors="coerce"
            ).fillna(0).sum()
        )
        result[f"{label}_rows"] = value
        result[f"{label}_fraction_of_assigned"] = value / max(total, 1.0)
    weak = result.get("global_last_resort_rows", 0.0) + result.get(
        "unmatched_singleton_self_exempt_rows", 0.0
    )
    result["weak_fallback_rows"] = weak
    result["weak_fallback_fraction_of_assigned"] = weak / max(total, 1.0)
    return result


def compute_exact_excess(
    prepared: Any,
    cmn: Any,
    cutpoints: Any,
    seed: int,
    max_last_resort_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    keys, randomizable, composition = cmn.build_matching_keys(prepared, cutpoints)
    layouts, coverage, effective = cmn.build_hierarchical_layouts(
        keys,
        randomizable,
        seed=int(seed) + 200003,
        max_last_resort_fraction=float(max_last_resort_fraction),
    )
    expectation = exact_expected_donor_pairs(
        layouts,
        effective,
        prepared.z_m,
        prepared.z_psi,
    )
    null_dx, null_dy = exact_null_increments(prepared, expectation)
    null_u, null_v = cmn.aggregate_mean_field(prepared, null_dx, null_dy)
    audit = {
        "matching_coverage": coverage_metrics(coverage),
        "composition": composition.iloc[0].to_dict() if not composition.empty else {},
        "expectation": expectation.audit,
    }
    return null_u, null_v, audit


def run_split_reliability(
    split: str,
    stage1_root: Path,
    stage1: Any,
    cmn: Any,
    specification: Any,
    cutpoints: Any,
    cmn_root: Path,
    partition_cutpoints: Optional[np.ndarray],
    partitions: int,
    partition_seed: int,
    max_last_resort_fraction: float,
    confirmation_output_only: bool,
    progress_every: int,
) -> Tuple[ReliabilitySplitResult, np.ndarray]:
    if split == "B_confirm" and not confirmation_output_only:
        raise RuntimeError("B_confirm reliability requires --confirmation-output-only.")
    dynamics = stage1_root / "dynamics"
    base = dynamics / f"student_dynamics_panel_core_{split}"
    columns, _ = cmn.required_columns_for_split(base)
    frame = cmn.read_table(base, columns=columns)
    frame = frame.sort_values(["user_id", "bundle_step_index"], kind="mergesort").reset_index(drop=True)
    innovations = cmn.reconstruct_innovations(
        frame,
        tau_response_days=float(stage1.TAU_RESPONSE_DAYS),
        tau_activity_days=float(stage1.TAU_ACTIVITY_DAYS),
        require_next_audit=True,
    )
    prepared = cmn.prepare_analysis(frame, innovations, stage1, specification)
    saved_field_audit = cmn.audit_saved_field(stage1_root, split, prepared.formal_field, skip=False)
    null_u, null_v, exact_audit = compute_exact_excess(
        prepared,
        cmn,
        cutpoints,
        seed=int(load_cmn_manifest(cmn_root, split)[0]["base_seed"]),
        max_last_resort_fraction=max_last_resort_fraction,
    )
    exact_excess_u = prepared.formal_field.drift_u - null_u
    exact_excess_v = prepared.formal_field.drift_v - null_v
    finite_arrays, finite_path = load_finite_null_arrays(cmn_root, split)
    if not np.array_equal(
        np.asarray(finite_arrays["drift_mask"], dtype=bool),
        np.asarray(prepared.formal_field.drift_mask, dtype=bool),
    ):
        raise RuntimeError(f"{split} finite-null drift support differs from Stage 1.")
    finite_excess_identity_error = max(
        float(
            np.nanmax(
                np.abs(
                    finite_arrays["excess_u"]
                    - (finite_arrays["observed_u"] - finite_arrays["null_mean_u"])
                )
            )
        ),
        float(
            np.nanmax(
                np.abs(
                    finite_arrays["excess_v"]
                    - (finite_arrays["observed_v"] - finite_arrays["null_mean_v"])
                )
            )
        ),
    )
    if finite_excess_identity_error > AUDIT_ATOL:
        raise RuntimeError(f"{split} finite-null excess arrays fail the subtraction identity.")
    finite_observed_error = max(
        float(
            np.max(
                np.abs(
                    finite_arrays["observed_u"][prepared.formal_field.drift_mask]
                    - prepared.formal_field.drift_u[prepared.formal_field.drift_mask]
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    finite_arrays["observed_v"][prepared.formal_field.drift_mask]
                    - prepared.formal_field.drift_v[prepared.formal_field.drift_mask]
                )
            )
        ),
    )
    if finite_observed_error > AUDIT_ATOL:
        raise RuntimeError(f"{split} finite-null arrays do not match the formal field.")

    all_users = np.asarray(sorted(pd.to_numeric(frame["user_id"], errors="raise").astype(np.int64).unique()))
    drift_counts_series = pd.Series(prepared.user_id).value_counts(sort=False)
    activity = np.asarray(
        [float(drift_counts_series.get(int(user), 0.0)) for user in all_users],
        dtype=np.float64,
    )
    if partition_cutpoints is None:
        partition_cutpoints = activity_cutpoints(activity)

    frame_user = pd.to_numeric(frame["user_id"], errors="raise").to_numpy(dtype=np.int64)
    frame_user_index = np.searchsorted(all_users, frame_user)
    if not np.array_equal(all_users[frame_user_index], frame_user):
        raise RuntimeError("User indexing failed during split-half construction.")
    nx, ny = prepared.formal_field.drift_u.shape
    half_u = np.full((partitions, 2, nx, ny), np.nan, dtype=np.float64)
    half_v = np.full_like(half_u, np.nan)
    half_count = np.zeros((partitions, 2, nx, ny), dtype=np.int32)
    half_user_count = np.zeros((partitions, 2, nx, ny), dtype=np.int32)
    balance_rows: List[dict] = []
    bin_balance_rows: List[dict] = []
    matching_rows: List[dict] = []

    for partition in range(partitions):
        seed = int(partition_seed) + 1_000_003 * (partition + 1)
        user_side, bin_balance = assign_complementary_halves(
            all_users, activity, partition_cutpoints, seed
        )
        bin_balance = bin_balance.assign(
            split=split,
            partition=int(partition),
            partition_seed=int(seed),
        )
        bin_balance_rows.extend(bin_balance.to_dict("records"))
        row_side = user_side[frame_user_index]
        for half in (0, 1):
            row_mask = row_side == half
            half_frame = frame.loc[row_mask].reset_index(drop=True)
            half_innovations = subset_innovations(innovations, row_mask)
            half_prepared = cmn.prepare_analysis(
                half_frame, half_innovations, stage1, specification
            )
            half_null_u, half_null_v, audit = compute_exact_excess(
                half_prepared,
                cmn,
                cutpoints,
                seed=seed + 10007 * (half + 1),
                max_last_resort_fraction=max_last_resort_fraction,
            )
            excess_u = half_prepared.formal_field.drift_u - half_null_u
            excess_v = half_prepared.formal_field.drift_v - half_null_v
            positive_weight = np.asarray(half_prepared.formal_field.drift_weight) > EPS
            excess_u = np.where(positive_weight, excess_u, np.nan)
            excess_v = np.where(positive_weight, excess_v, np.nan)
            half_u[partition, half] = excess_u
            half_v[partition, half] = excess_v
            half_count[partition, half] = np.asarray(
                half_prepared.formal_field.drift_count, dtype=np.int32
            )
            half_user_count[partition, half] = np.asarray(
                drift_user_count(half_prepared), dtype=np.int32
            )
            balance_rows.append(
                {
                    "split": split,
                    "partition": int(partition),
                    "partition_seed": int(seed),
                    "half": int(half),
                    "users": int(half_frame["user_id"].nunique()),
                    "panel_rows": int(len(half_frame)),
                    "valid_drift_rows": int(len(half_prepared.user_id)),
                    "drift_cells_with_positive_weight": int(
                        np.sum(half_prepared.formal_field.drift_weight > EPS)
                    ),
                }
            )
            matching_rows.append(
                {
                    "split": split,
                    "partition": int(partition),
                    "half": int(half),
                    **audit["matching_coverage"],
                    **{
                        key: value
                        for key, value in audit["expectation"].items()
                        if key.startswith("mean_")
                    },
                }
            )
            del half_frame, half_innovations, half_prepared
            gc.collect()
        if (partition + 1) % max(1, progress_every) == 0 or partition == 0:
            print(
                f"[excess reliability] {split} partition {partition + 1}/{partitions}",
                flush=True,
            )

    formal_audit = {
        "split": split,
        "panel_rows": int(len(frame)),
        "panel_users": int(len(all_users)),
        "valid_drift_rows": int(len(prepared.user_id)),
        "saved_field_audit": saved_field_audit,
        "finite_null_array_path": str(finite_path.resolve()),
        "finite_null_array_sha256": file_sha256(finite_path),
        "finite_observed_max_abs_error": finite_observed_error,
        "finite_excess_subtraction_max_abs_error": finite_excess_identity_error,
        "finite_drift_mask_exact_match": True,
        "full_exact_null_audit": exact_audit,
        "activity_partition_cutpoints": partition_cutpoints,
        "partition_contract": (
            "A_val-fitted valid-drift-count cutpoints; stable-hash alternating assignment "
            "within fixed bins; B_confirm applies the A_val-frozen cutpoints"
        ),
    }
    result = ReliabilitySplitResult(
        split=split,
        full_observed_u=np.asarray(prepared.formal_field.drift_u, dtype=np.float64),
        full_observed_v=np.asarray(prepared.formal_field.drift_v, dtype=np.float64),
        full_exact_null_u=np.asarray(null_u, dtype=np.float64),
        full_exact_null_v=np.asarray(null_v, dtype=np.float64),
        full_exact_excess_u=np.asarray(exact_excess_u, dtype=np.float64),
        full_exact_excess_v=np.asarray(exact_excess_v, dtype=np.float64),
        full_occupancy=np.asarray(prepared.formal_field.occupancy_probability, dtype=np.float64),
        drift_mask=np.asarray(prepared.formal_field.drift_mask, dtype=bool),
        drift_count=np.asarray(prepared.formal_field.drift_count, dtype=np.float64),
        drift_user_count=drift_user_count(prepared),
        finite_arrays=finite_arrays,
        half_excess_u=half_u,
        half_excess_v=half_v,
        half_drift_count=half_count,
        half_drift_user_count=half_user_count,
        partition_balance=pd.DataFrame(balance_rows),
        partition_bin_balance=pd.DataFrame(bin_balance_rows),
        matching_quality=pd.DataFrame(matching_rows),
        formal_audit=formal_audit,
    )
    del prepared, innovations, frame
    gc.collect()
    return result, partition_cutpoints


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
    if len(cell) == 0:
        return GroupedUserCell(
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            [np.asarray([], dtype=np.float64) for _ in values],
        )
    key = user_index * int(n_cells) + cell
    order = np.argsort(key, kind="mergesort")
    sorted_key = key[order]
    starts = np.flatnonzero(np.concatenate([[True], sorted_key[1:] != sorted_key[:-1]]))
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


def grouped_matrix(
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
        shape=(n_users, n_cells),
        dtype=np.float64,
    )


@dataclass
class CoreMatrix:
    split: str
    users: np.ndarray
    linear: sparse.csr_matrix
    squared_weight: sparse.csr_matrix
    block_index: Dict[str, int]
    formal_field: Any
    formal_drift_user_count: np.ndarray
    audit: Dict[str, Any]


def build_core_matrix(
    split: str,
    stage1_root: Path,
    stage1: Any,
    specification: Any,
) -> CoreMatrix:
    base = stage1_root / "dynamics" / f"student_dynamics_panel_core_{split}"
    columns = [
        "user_id",
        specification.xcol,
        specification.ycol,
        specification.dxcol,
        specification.dycol,
    ]
    frame = read_table(base, columns=columns)
    frame = stage1.downcast_frame(frame)
    stats = stage1.occupancy_drift_stats(frame, specification)
    formal = stage1.field_stats_from_dict(stats)
    users = np.asarray(sorted(pd.to_numeric(frame["user_id"], errors="raise").astype(np.int64).unique()))
    user_id = pd.to_numeric(frame["user_id"], errors="raise").to_numpy(dtype=np.int64)
    user_index = np.searchsorted(users, user_id)
    weights = np.asarray(stage1.user_balanced_weights(frame), dtype=np.float64)
    x = pd.to_numeric(frame[specification.xcol], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(frame[specification.ycol], errors="coerce").to_numpy(dtype=np.float64)
    dx = pd.to_numeric(frame[specification.dxcol], errors="coerce").to_numpy(dtype=np.float64)
    dy = pd.to_numeric(frame[specification.dycol], errors="coerce").to_numpy(dtype=np.float64)
    ix = stage1.digitize_closed_right(x, specification.xbins)
    iy = stage1.digitize_closed_right(y, specification.ybins)
    nx = len(specification.xbins) - 1
    ny = len(specification.ybins) - 1
    n_cells = nx * ny
    state_valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (ix >= 0)
        & (ix < nx)
        & (iy >= 0)
        & (iy < ny)
    )
    drift_valid = state_valid & np.isfinite(dx) & np.isfinite(dy)
    state_cell = (ix[state_valid] * ny + iy[state_valid]).astype(np.int64)
    drift_cell = (ix[drift_valid] * ny + iy[drift_valid]).astype(np.int64)
    occupancy_grouped = group_user_cell(
        user_index[state_valid], state_cell, [weights[state_valid]], n_cells
    )
    drift_values = [
        weights[drift_valid],
        weights[drift_valid] * dx[drift_valid],
        weights[drift_valid] * dy[drift_valid],
        weights[drift_valid] * dx[drift_valid] ** 2,
        weights[drift_valid] * dy[drift_valid] ** 2,
        weights[drift_valid] * dx[drift_valid] * dy[drift_valid],
        weights[drift_valid] ** 2,
    ]
    drift_grouped = group_user_cell(
        user_index[drift_valid], drift_cell, drift_values, n_cells
    )
    names = ["occupancy", "drift_weight", "sx", "sy", "sxx", "syy", "sxy"]
    matrices = [
        grouped_matrix(occupancy_grouped, 0, len(users), n_cells),
        *[
            grouped_matrix(drift_grouped, index, len(users), n_cells)
            for index in range(6)
        ],
    ]
    linear = sparse.hstack(matrices, format="csr", dtype=np.float64)
    squared = grouped_matrix(drift_grouped, 6, len(users), n_cells)
    drift_users = np.bincount(
        drift_grouped.cell, minlength=n_cells
    ).reshape(formal.drift_u.shape)
    matrix = CoreMatrix(
        split=split,
        users=users,
        linear=linear,
        squared_weight=squared,
        block_index={name: index for index, name in enumerate(names)},
        formal_field=formal,
        formal_drift_user_count=drift_users,
        audit={
            "split": split,
            "panel_rows": int(len(frame)),
            "panel_users": int(len(users)),
            "finite_state_rows": int(np.sum(state_valid)),
            "finite_drift_rows": int(np.sum(drift_valid)),
            "unique_occupancy_user_cells": int(len(occupancy_grouped.cell)),
            "unique_drift_user_cells": int(len(drift_grouped.cell)),
            "linear_matrix_shape": list(linear.shape),
            "linear_matrix_nnz": int(linear.nnz),
            "squared_weight_matrix_nnz": int(squared.nnz),
        },
    )
    del frame
    gc.collect()
    return matrix


def block_view(vector: np.ndarray, index: Mapping[str, int], name: str, n_cells: int) -> np.ndarray:
    start = int(index[name]) * n_cells
    return np.asarray(vector[start : start + n_cells], dtype=np.float64)


def field_from_aggregates(
    stage1: Any,
    core: CoreMatrix,
    linear_vector: np.ndarray,
    squared_vector: np.ndarray,
) -> Any:
    formal = core.formal_field
    shape = formal.drift_u.shape
    n_cells = int(np.prod(shape))
    occupancy = block_view(linear_vector, core.block_index, "occupancy", n_cells).reshape(shape)
    drift_weight = block_view(linear_vector, core.block_index, "drift_weight", n_cells).reshape(shape)
    sx = block_view(linear_vector, core.block_index, "sx", n_cells).reshape(shape)
    sy = block_view(linear_vector, core.block_index, "sy", n_cells).reshape(shape)
    sxx = block_view(linear_vector, core.block_index, "sxx", n_cells).reshape(shape)
    syy = block_view(linear_vector, core.block_index, "syy", n_cells).reshape(shape)
    sxy = block_view(linear_vector, core.block_index, "sxy", n_cells).reshape(shape)
    drift_weight_sq = np.asarray(squared_vector, dtype=np.float64).reshape(shape)
    denominator = np.maximum(drift_weight, EPS)
    drift_u = sx / denominator
    drift_v = sy / denominator
    diff_x = np.maximum(sxx / denominator - drift_u * drift_u, 0.0)
    diff_y = np.maximum(syy / denominator - drift_v * drift_v, 0.0)
    diff_xy = sxy / denominator - drift_u * drift_v
    effective_n = drift_weight * drift_weight / np.maximum(drift_weight_sq, EPS)
    occupancy_probability = occupancy / max(float(np.sum(occupancy)), EPS)
    return stage1.FieldStats(
        xbins=formal.xbins,
        ybins=formal.ybins,
        xcenters=formal.xcenters,
        ycenters=formal.ycenters,
        occupancy_weighted=occupancy,
        occupancy_count=formal.occupancy_count,
        user_count=formal.user_count,
        occupancy_probability=occupancy_probability,
        potential=-np.log(occupancy_probability + EPS),
        drift_u=drift_u,
        drift_v=drift_v,
        drift_count=formal.drift_count,
        drift_weight=drift_weight,
        drift_weight_sq=drift_weight_sq,
        drift_effective_sample_size=effective_n,
        drift_se_u=np.sqrt(diff_x / np.maximum(effective_n, 1.0)),
        drift_se_v=np.sqrt(diff_y / np.maximum(effective_n, 1.0)),
        diff_x=diff_x,
        diff_y=diff_y,
        diff_xy=diff_xy,
        state_mask=formal.state_mask,
        drift_mask=formal.drift_mask,
        valid_state_rows=formal.valid_state_rows,
        valid_drift_rows=formal.valid_drift_rows,
        users=formal.users,
    )


def aggregate_core_batch(
    core: CoreMatrix, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    linear = (core.linear.T @ np.asarray(weights, dtype=np.float64).T).T
    squared = (
        core.squared_weight.T @ (np.asarray(weights, dtype=np.float64) ** 2).T
    ).T
    return np.asarray(linear, dtype=np.float64), np.asarray(squared, dtype=np.float64)


def jaccard(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    union = int(np.sum(first | second))
    return float(np.sum(first & second) / union) if union else float("nan")


def overlap_coefficient(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    denominator = min(int(np.sum(first)), int(np.sum(second)))
    return float(np.sum(first & second) / denominator) if denominator else float("nan")


def fuzzy_jaccard(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.sum(np.maximum(first, second)))
    return float(np.sum(np.minimum(first, second)) / denominator) if denominator > EPS else float("nan")


def fuzzy_overlap(first: np.ndarray, second: np.ndarray) -> float:
    denominator = min(float(np.sum(first)), float(np.sum(second)))
    return float(np.sum(np.minimum(first, second)) / denominator) if denominator > EPS else float("nan")


def load_formal_core_contract(stage1_root: Path) -> Dict[str, Any]:
    root = stage1_root / "dynamics" / "candidate_regions" / PRIMARY_COORDINATE
    core_path = root / "A_train_primary_convergence_core_mask.npy"
    thresholds_path = root / "training_convergence_thresholds.json"
    return {
        "root": root,
        "core_path": core_path,
        "core_mask": np.load(core_path).astype(bool),
        "thresholds_path": thresholds_path,
        "thresholds": load_json(thresholds_path),
        "train_regions": read_table(root / "training_flow_defined_convergence_regions"),
        "val_regions": read_table(root / "validation_flow_defined_convergence_regions_fixed_thresholds"),
        "reproducibility": read_table(root / "training_validation_convergence_region_reproducibility"),
    }


def formal_core_audit(
    stage1: Any,
    train_core: CoreMatrix,
    val_core: CoreMatrix,
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    ones_train = np.ones((1, len(train_core.users)), dtype=np.float64)
    ones_val = np.ones((1, len(val_core.users)), dtype=np.float64)
    train_linear, train_squared = aggregate_core_batch(train_core, ones_train)
    val_linear, val_squared = aggregate_core_batch(val_core, ones_val)
    train_field = field_from_aggregates(stage1, train_core, train_linear[0], train_squared[0])
    val_field = field_from_aggregates(stage1, val_core, val_linear[0], val_squared[0])
    formal_train = train_core.formal_field
    formal_val = val_core.formal_field
    max_field_error = max(
        float(np.max(np.abs(train_field.drift_u - formal_train.drift_u))),
        float(np.max(np.abs(train_field.drift_v - formal_train.drift_v))),
        float(np.max(np.abs(train_field.diff_x - formal_train.diff_x))),
        float(np.max(np.abs(train_field.diff_y - formal_train.diff_y))),
        float(np.max(np.abs(train_field.occupancy_probability - formal_train.occupancy_probability))),
        float(np.max(np.abs(val_field.drift_u - formal_val.drift_u))),
        float(np.max(np.abs(val_field.drift_v - formal_val.drift_v))),
    )
    if max_field_error > 1e-10:
        raise RuntimeError(f"Core sufficient statistics fail the all-ones field audit: {max_field_error:.3e}")
    thresholds_json = dict(contract["thresholds"])
    train_regions, train_masks, _, thresholds = stage1.identify_convergence_regions(
        train_field,
        "A_train_soft_core_audit",
        float(thresholds_json["speed_quantile"]),
        float(thresholds_json["negative_divergence_quantile"]),
        float(thresholds_json["drift_to_diffusion_quantile"]),
        int(thresholds_json["min_region_cells"]),
        float(thresholds_json["shell_radius"]),
        thresholds=None,
        allow_fallback=False,
    )
    if not train_masks:
        raise RuntimeError("The all-ones core audit did not identify the formal training core.")
    threshold_error = max(
        abs(float(thresholds.speed_threshold) - float(thresholds_json["speed_threshold"])),
        abs(float(thresholds.negative_divergence_threshold) - float(thresholds_json["negative_divergence_threshold"])),
        abs(float(thresholds.drift_to_diffusion_threshold) - float(thresholds_json["drift_to_diffusion_threshold"])),
    )
    mask_match = np.array_equal(train_masks[0], np.asarray(contract["core_mask"], dtype=bool))
    if threshold_error > 1e-10 or not mask_match:
        raise RuntimeError("The all-ones core-selection audit does not reproduce Stage 1.")
    val_regions, val_masks, _, _ = stage1.identify_convergence_regions(
        val_field,
        "A_val_soft_core_audit",
        float(thresholds_json["speed_quantile"]),
        float(thresholds_json["negative_divergence_quantile"]),
        float(thresholds_json["drift_to_diffusion_quantile"]),
        int(thresholds_json["min_region_cells"]),
        float(thresholds_json["shell_radius"]),
        thresholds=thresholds,
        allow_fallback=False,
    )
    reproducibility = stage1.convergence_region_reproducibility(
        train_masks[0], train_regions, val_masks, val_regions
    )
    archived = contract["reproducibility"].iloc[0]
    reproduced = reproducibility.iloc[0]
    jaccard_error = abs(float(archived["mask_jaccard"]) - float(reproduced["mask_jaccard"]))
    center_error = abs(
        float(archived["convergence_center_distance"])
        - float(reproduced["convergence_center_distance"])
    )
    if max(jaccard_error, center_error) > 1e-10:
        raise RuntimeError("The all-ones train-validation core audit does not reproduce Stage 1.")
    val_mask = val_masks[int(reproduced["validation_region_id"])]
    return {
        "max_abs_all_ones_field_error": max_field_error,
        "max_abs_threshold_error": threshold_error,
        "training_primary_mask_exact_match": mask_match,
        "formal_train_validation_jaccard": float(reproduced["mask_jaccard"]),
        "formal_train_validation_overlap_coefficient": float(reproduced["mask_overlap_coefficient"]),
        "formal_train_validation_center_distance": float(reproduced["convergence_center_distance"]),
        "formal_validation_matching_mask": val_mask,
    }


def run_soft_core(
    stage1: Any,
    stage1_root: Path,
    specification: Any,
    output_root: Path,
    replicates: int,
    batch_size: int,
    seed: int,
    progress_every: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    train_core = build_core_matrix("A_train", stage1_root, stage1, specification)
    val_core = build_core_matrix("A_val", stage1_root, stage1, specification)
    contract = load_formal_core_contract(stage1_root)
    audit = formal_core_audit(stage1, train_core, val_core, contract)
    formal_train_mask = np.asarray(contract["core_mask"], dtype=bool)
    formal_val_mask = np.asarray(audit.pop("formal_validation_matching_mask"), dtype=bool)
    thresholds_json = dict(contract["thresholds"])

    train_unconditional_sum = np.zeros_like(formal_train_mask, dtype=np.float64)
    val_unconditional_sum = np.zeros_like(formal_val_mask, dtype=np.float64)
    train_paired_available_sum = np.zeros_like(formal_train_mask, dtype=np.float64)
    val_paired_available_sum = np.zeros_like(formal_val_mask, dtype=np.float64)
    train_paired_qualified_sum = np.zeros_like(formal_train_mask, dtype=np.float64)
    val_paired_qualified_sum = np.zeros_like(formal_val_mask, dtype=np.float64)
    paired_available_count = 0
    paired_qualified_count = 0

    rows: List[dict] = []
    rng_train = np.random.default_rng(int(seed) + 100003)
    rng_val = np.random.default_rng(int(seed) + 200003)
    completed = 0

    while completed < replicates:
        current = min(batch_size, replicates - completed)
        weights_train = rng_train.exponential(1.0, size=(current, len(train_core.users)))
        weights_val = rng_val.exponential(1.0, size=(current, len(val_core.users)))
        weights_train /= np.maximum(np.mean(weights_train, axis=1, keepdims=True), EPS)
        weights_val /= np.maximum(np.mean(weights_val, axis=1, keepdims=True), EPS)
        train_linear, train_squared = aggregate_core_batch(train_core, weights_train)
        val_linear, val_squared = aggregate_core_batch(val_core, weights_val)

        for local in range(current):
            replicate = completed + local
            train_field = field_from_aggregates(
                stage1, train_core, train_linear[local], train_squared[local]
            )
            val_field = field_from_aggregates(
                stage1, val_core, val_linear[local], val_squared[local]
            )
            train_regions, train_masks, _, thresholds = stage1.identify_convergence_regions(
                train_field,
                f"A_train_soft_core_{replicate:04d}",
                float(thresholds_json["speed_quantile"]),
                float(thresholds_json["negative_divergence_quantile"]),
                float(thresholds_json["drift_to_diffusion_quantile"]),
                int(thresholds_json["min_region_cells"]),
                float(thresholds_json["shell_radius"]),
                thresholds=None,
                allow_fallback=False,
            )
            row: Dict[str, Any] = {
                "replicate": int(replicate),
                "training_region_available": bool(train_masks),
                "training_primary_dynamically_qualified": False,
                "validation_matching_region_available": False,
                "validation_matching_region_dynamically_qualified": False,
                "paired_region_available": False,
                "paired_dynamically_qualified": False,
                "training_primary_is_best_formal_overlap_candidate": False,
                "speed_threshold": float(thresholds.speed_threshold),
                "negative_divergence_threshold": float(
                    thresholds.negative_divergence_threshold
                ),
                "drift_to_diffusion_threshold": float(
                    thresholds.drift_to_diffusion_threshold
                ),
            }
            if not train_masks:
                rows.append(row)
                continue

            train_primary = train_masks[0]
            train_unconditional_sum += train_primary.astype(np.float64)
            primary_row = train_regions.iloc[0]
            train_qualified = bool(
                primary_row.get("dynamically_qualified", False)
            )
            formal_scores: List[Tuple[float, float, int, int]] = []
            formal_train_row = contract["train_regions"].iloc[0]
            for region_id, mask in enumerate(train_masks):
                candidate = train_regions.iloc[region_id]
                distance = math.hypot(
                    float(candidate["convergence_center_M"])
                    - float(formal_train_row["convergence_center_M"]),
                    float(candidate["convergence_center_Psi"])
                    - float(formal_train_row["convergence_center_Psi"]),
                )
                formal_scores.append(
                    (jaccard(mask, formal_train_mask), -distance, -region_id, region_id)
                )
            best_formal = max(formal_scores, key=lambda item: (item[0], item[1], item[2]))
            row.update(
                {
                    "training_primary_cells": int(np.sum(train_primary)),
                    "training_primary_dynamically_qualified": train_qualified,
                    "training_primary_center_M": float(
                        primary_row["convergence_center_M"]
                    ),
                    "training_primary_center_Psi": float(
                        primary_row["convergence_center_Psi"]
                    ),
                    "training_primary_vs_formal_jaccard": jaccard(
                        train_primary, formal_train_mask
                    ),
                    "training_primary_vs_formal_overlap": overlap_coefficient(
                        train_primary, formal_train_mask
                    ),
                    "training_primary_is_best_formal_overlap_candidate": bool(
                        best_formal[3] == 0 and best_formal[0] > 0.0
                    ),
                    "best_formal_basin_candidate_jaccard": float(best_formal[0]),
                }
            )

            val_regions, val_masks, _, _ = stage1.identify_convergence_regions(
                val_field,
                f"A_val_soft_core_{replicate:04d}",
                float(thresholds_json["speed_quantile"]),
                float(thresholds_json["negative_divergence_quantile"]),
                float(thresholds_json["drift_to_diffusion_quantile"]),
                int(thresholds_json["min_region_cells"]),
                float(thresholds_json["shell_radius"]),
                thresholds=thresholds,
                allow_fallback=False,
            )
            reproducibility = stage1.convergence_region_reproducibility(
                train_primary, train_regions, val_masks, val_regions
            )
            if reproducibility.empty or not bool(
                reproducibility.iloc[0].get(
                    "validation_matching_region_available", False
                )
            ):
                rows.append(row)
                continue

            match = reproducibility.iloc[0]
            region_id = int(match["validation_region_id"])
            val_mask = val_masks[region_id]
            val_row = val_regions.iloc[region_id]
            val_qualified = bool(val_row.get("dynamically_qualified", False))
            val_unconditional_sum += val_mask.astype(np.float64)
            train_paired_available_sum += train_primary.astype(np.float64)
            val_paired_available_sum += val_mask.astype(np.float64)
            paired_available_count += 1
            paired_qualified = bool(train_qualified and val_qualified)
            if paired_qualified:
                train_paired_qualified_sum += train_primary.astype(np.float64)
                val_paired_qualified_sum += val_mask.astype(np.float64)
                paired_qualified_count += 1
            row.update(
                {
                    "validation_matching_region_available": True,
                    "validation_matching_region_cells": int(np.sum(val_mask)),
                    "validation_matching_region_dynamically_qualified": val_qualified,
                    "paired_region_available": True,
                    "paired_dynamically_qualified": paired_qualified,
                    "train_validation_mask_jaccard": float(match["mask_jaccard"]),
                    "train_validation_mask_overlap": float(
                        match["mask_overlap_coefficient"]
                    ),
                    "train_validation_center_distance": float(
                        match["convergence_center_distance"]
                    ),
                    "validation_matching_center_M": float(
                        val_row["convergence_center_M"]
                    ),
                    "validation_matching_center_Psi": float(
                        val_row["convergence_center_Psi"]
                    ),
                    "validation_matching_vs_formal_jaccard": jaccard(
                        val_mask, formal_val_mask
                    ),
                }
            )
            rows.append(row)
        completed += current
        if completed % max(1, progress_every) == 0 or completed == replicates:
            print(f"[soft core] {completed}/{replicates} replicates complete", flush=True)

    replicate_table = pd.DataFrame(rows)
    train_unconditional = train_unconditional_sum / float(replicates)
    val_unconditional = val_unconditional_sum / float(replicates)
    train_paired_available = (
        train_paired_available_sum / float(paired_available_count)
        if paired_available_count > 0
        else np.full_like(train_unconditional, np.nan)
    )
    val_paired_available = (
        val_paired_available_sum / float(paired_available_count)
        if paired_available_count > 0
        else np.full_like(val_unconditional, np.nan)
    )
    train_paired_qualified = (
        train_paired_qualified_sum / float(paired_qualified_count)
        if paired_qualified_count > 0
        else np.full_like(train_unconditional, np.nan)
    )
    val_paired_qualified = (
        val_paired_qualified_sum / float(paired_qualified_count)
        if paired_qualified_count > 0
        else np.full_like(val_unconditional, np.nan)
    )

    x_grid, y_grid = np.meshgrid(
        train_core.formal_field.xcenters,
        train_core.formal_field.ycenters,
        indexing="ij",
    )
    cell_table = pd.DataFrame(
        {
            "x_bin": np.repeat(
                np.arange(train_unconditional.shape[0]),
                train_unconditional.shape[1],
            ),
            "y_bin": np.tile(
                np.arange(train_unconditional.shape[1]),
                train_unconditional.shape[0],
            ),
            "M_center": x_grid.ravel(order="C"),
            "Psi_center": y_grid.ravel(order="C"),
            "training_core_selection_frequency_unconditional": train_unconditional.ravel(
                order="C"
            ),
            "validation_matching_core_selection_frequency_unconditional": val_unconditional.ravel(
                order="C"
            ),
            "training_core_selection_frequency_paired_available": train_paired_available.ravel(
                order="C"
            ),
            "validation_core_selection_frequency_paired_available": val_paired_available.ravel(
                order="C"
            ),
            "training_core_selection_frequency_paired_qualified": train_paired_qualified.ravel(
                order="C"
            ),
            "validation_core_selection_frequency_paired_qualified": val_paired_qualified.ravel(
                order="C"
            ),
            "formal_training_core": formal_train_mask.ravel(order="C"),
            "formal_validation_matching_core": formal_val_mask.ravel(order="C"),
        }
    )

    train_available = replicate_table["training_region_available"].fillna(False)
    val_available = replicate_table[
        "validation_matching_region_available"
    ].fillna(False)
    paired_available = replicate_table["paired_region_available"].fillna(False)
    paired_qualified_series = replicate_table[
        "paired_dynamically_qualified"
    ].fillna(False)
    training_qualified = replicate_table[
        "training_primary_dynamically_qualified"
    ].fillna(False)
    validation_qualified = replicate_table[
        "validation_matching_region_dynamically_qualified"
    ].fillna(False)
    formal_primary = replicate_table["training_primary_is_best_formal_overlap_candidate"].fillna(False)

    summary = {
        "replicates": int(replicates),
        "training_region_detection_rate": float(train_available.mean()),
        "training_primary_dynamically_qualified_rate_unconditional": float(
            training_qualified.mean()
        ),
        "training_primary_dynamically_qualified_rate_given_detection": float(
            training_qualified[train_available].mean()
        )
        if bool(train_available.any())
        else np.nan,
        "validation_matching_region_availability_rate": float(val_available.mean()),
        "validation_matching_dynamically_qualified_rate_unconditional": float(
            validation_qualified.mean()
        ),
        "validation_matching_dynamically_qualified_rate_given_availability": float(
            validation_qualified[val_available].mean()
        )
        if bool(val_available.any())
        else np.nan,
        "paired_region_availability_rate": float(paired_available.mean()),
        "paired_dynamically_qualified_rate_unconditional": float(
            paired_qualified_series.mean()
        ),
        "paired_dynamically_qualified_rate_given_pair_availability": float(
            paired_qualified_series[paired_available].mean()
        )
        if bool(paired_available.any())
        else np.nan,
        "training_primary_is_best_formal_overlap_candidate_rate_unconditional": float(
            formal_primary.mean()
        ),
        "training_primary_is_best_formal_overlap_candidate_rate_given_training_detection": float(
            formal_primary[train_available].mean()
        )
        if bool(train_available.any())
        else np.nan,
        "paired_available_replicates": int(paired_available_count),
        "paired_qualified_replicates": int(paired_qualified_count),
        "soft_jaccard_unconditional": fuzzy_jaccard(
            train_unconditional, val_unconditional
        ),
        "soft_overlap_coefficient_unconditional": fuzzy_overlap(
            train_unconditional, val_unconditional
        ),
        "soft_jaccard_paired_available": fuzzy_jaccard(
            train_paired_available, val_paired_available
        ),
        "soft_overlap_coefficient_paired_available": fuzzy_overlap(
            train_paired_available, val_paired_available
        ),
        "soft_jaccard_paired_qualified": fuzzy_jaccard(
            train_paired_qualified, val_paired_qualified
        ),
        "soft_overlap_coefficient_paired_qualified": fuzzy_overlap(
            train_paired_qualified, val_paired_qualified
        ),
        "formal_training_core_mean_selection_frequency_unconditional": float(
            np.mean(train_unconditional[formal_train_mask])
        ),
        "formal_validation_core_mean_selection_frequency_unconditional": float(
            np.mean(val_unconditional[formal_val_mask])
        ),
        "formal_training_core_mean_selection_frequency_paired_available": float(
            np.nanmean(train_paired_available[formal_train_mask])
        ),
        "formal_validation_core_mean_selection_frequency_paired_available": float(
            np.nanmean(val_paired_available[formal_val_mask])
        ),
        "formal_training_core_mean_selection_frequency_paired_qualified": float(
            np.nanmean(train_paired_qualified[formal_train_mask])
        ),
        "formal_validation_core_mean_selection_frequency_paired_qualified": float(
            np.nanmean(val_paired_qualified[formal_val_mask])
        ),
        **quantile_summary(
            replicate_table.get(
                "train_validation_mask_jaccard", pd.Series(dtype=float)
            ),
            "hard_jaccard",
        ),
        **quantile_summary(
            replicate_table.get(
                "train_validation_mask_overlap", pd.Series(dtype=float)
            ),
            "hard_overlap",
        ),
        **quantile_summary(
            replicate_table.get(
                "train_validation_center_distance", pd.Series(dtype=float)
            ),
            "center_distance",
        ),
        **quantile_summary(
            replicate_table.get("training_primary_center_M", pd.Series(dtype=float)),
            "training_center_M",
        ),
        **quantile_summary(
            replicate_table.get(
                "training_primary_center_Psi", pd.Series(dtype=float)
            ),
            "training_center_Psi",
        ),
        **quantile_summary(
            replicate_table.get(
                "validation_matching_center_M", pd.Series(dtype=float)
            ),
            "validation_center_M",
        ),
        **quantile_summary(
            replicate_table.get(
                "validation_matching_center_Psi", pd.Series(dtype=float)
            ),
            "validation_center_Psi",
        ),
    }
    threshold_rows = []
    for metric in (
        "speed_threshold",
        "negative_divergence_threshold",
        "drift_to_diffusion_threshold",
    ):
        threshold_rows.append(
            {
                "metric": metric,
                **quantile_summary(replicate_table[metric], "value"),
            }
        )
    np.savez_compressed(
        output_root / "arrays" / "soft_core_selection_frequency.npz",
        training_frequency_unconditional=train_unconditional,
        validation_frequency_unconditional=val_unconditional,
        training_frequency_paired_available=train_paired_available,
        validation_frequency_paired_available=val_paired_available,
        training_frequency_paired_qualified=train_paired_qualified,
        validation_frequency_paired_qualified=val_paired_qualified,
        formal_training_core=formal_train_mask,
        formal_validation_matching_core=formal_val_mask,
        xcenters=train_core.formal_field.xcenters,
        ycenters=train_core.formal_field.ycenters,
    )
    audit.update(
        {
            "A_train_matrix": train_core.audit,
            "A_val_matrix": val_core.audit,
            "support_contract": (
                "formal full-split state and drift masks fixed; positive Exp(1) learner weights "
                "recompute weighted moments, A_train thresholds and core selection"
            ),
            "frequency_denominators": {
                "unconditional": int(replicates),
                "paired_available": int(paired_available_count),
                "paired_qualified": int(paired_qualified_count),
            },
            "frequency_interpretation": (
                "unconditional maps combine detection and location stability; paired-available "
                "and paired-qualified maps condition on explicit detectability/qualification "
                "events and are not posterior probabilities"
            ),
            "fallback_allowed": False,
            "multiplier_bank_relation_to_prior_analyses": (
                "new deterministic bank under the same Exp(1) learner-composition contract; "
                "not claimed to reproduce an unpublished prior bank"
            ),
            "B_confirm_read": False,
        }
    )
    return replicate_table, pd.DataFrame([summary]), cell_table, {
        "formal_audit": audit,
        "threshold_summary": pd.DataFrame(threshold_rows),
    }

def reliability_tables(
    val: ReliabilitySplitResult,
    confirm: ReliabilitySplitResult,
    stage1: Any,
    minimum_cell_coverage: float,
    minimum_occupancy_coverage: float,
    minimum_valid_partition_fraction: float,
    minimum_defined_benchmark_fraction: float,
    minimum_half_drift_count: int,
    minimum_half_cell_users: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    formal_mask = val.drift_mask & confirm.drift_mask
    formal_cells = int(np.sum(formal_mask))
    if formal_cells != FORMAL_EXCESS_CELLS:
        raise RuntimeError(
            f"Formal common excess-field support has {formal_cells} cells; expected {FORMAL_EXCESS_CELLS}."
        )

    finite_val_u = val.finite_arrays["excess_u"]
    finite_val_v = val.finite_arrays["excess_v"]
    finite_confirm_u = confirm.finite_arrays["excess_u"]
    finite_confirm_v = confirm.finite_arrays["excess_v"]
    point_rows: List[dict] = []
    for estimator, first_u, first_v, second_u, second_v in (
        (
            "formal_100_permutation_mean",
            finite_val_u,
            finite_val_v,
            finite_confirm_u,
            finite_confirm_v,
        ),
        (
            "exact_cyclic_shift_expectation",
            val.full_exact_excess_u,
            val.full_exact_excess_v,
            confirm.full_exact_excess_u,
            confirm.full_exact_excess_v,
        ),
    ):
        metrics = field_agreement(
            first_u,
            first_v,
            second_u,
            second_v,
            formal_mask,
            val.full_occupancy,
        )
        point_rows.append(
            {
                "estimator": estimator,
                "support": "formal_common_support",
                "support_definition_split": "formal A_val/B_confirm common drift support",
                **metrics,
            }
        )
    points = pd.DataFrame(point_rows)
    formal_point = points[
        points["estimator"] == "formal_100_permutation_mean"
    ].iloc[0]
    if abs(float(formal_point["vector_correlation"]) - FORMAL_EXCESS_VECTOR_CORR) > 5e-4:
        raise RuntimeError("The archived finite-null excess-field vector correlation was not reproduced.")
    if abs(float(formal_point["speed_correlation"]) - FORMAL_EXCESS_SPEED_CORR) > 5e-4:
        raise RuntimeError("The archived finite-null excess-field speed correlation was not reproduced.")
    if abs(float(formal_point["weighted_local_cosine"]) - FORMAL_EXCESS_LOCAL_COSINE) > 1e-3:
        raise RuntimeError("The archived finite-null excess-field local cosine was not reproduced.")

    def half_available(result: ReliabilitySplitResult, partition: int) -> np.ndarray:
        return (
            formal_mask
            & np.isfinite(result.half_excess_u[partition, 0])
            & np.isfinite(result.half_excess_v[partition, 0])
            & np.isfinite(result.half_excess_u[partition, 1])
            & np.isfinite(result.half_excess_v[partition, 1])
            & (result.half_drift_count[partition, 0] >= int(minimum_half_drift_count))
            & (result.half_drift_count[partition, 1] >= int(minimum_half_drift_count))
            & (result.half_drift_user_count[partition, 0] >= int(minimum_half_cell_users))
            & (result.half_drift_user_count[partition, 1] >= int(minimum_half_cell_users))
        )

    def occupancy_coverage(result: ReliabilitySplitResult, mask: np.ndarray) -> float:
        denominator = float(np.sum(result.full_occupancy[formal_mask]))
        return float(np.sum(result.full_occupancy[mask]) / max(denominator, EPS))

    partitions = int(val.half_excess_u.shape[0])
    if int(confirm.half_excess_u.shape[0]) != partitions:
        raise RuntimeError("A_val and B_confirm have different split-half partition counts.")
    split_masks: Dict[str, np.ndarray] = {
        "A_val": np.zeros((partitions,) + formal_mask.shape, dtype=bool),
        "B_confirm": np.zeros((partitions,) + formal_mask.shape, dtype=bool),
    }

    reliability_rows: List[dict] = []
    for split_result in (val, confirm):
        for partition in range(partitions):
            mask = half_available(split_result, partition)
            split_masks[split_result.split][partition] = mask
            cell_coverage = float(np.sum(mask) / max(formal_cells, 1))
            occupancy_mass_coverage = occupancy_coverage(split_result, mask)
            coverage_valid = bool(
                cell_coverage >= float(minimum_cell_coverage)
                and occupancy_mass_coverage >= float(minimum_occupancy_coverage)
            )
            metrics = field_agreement(
                split_result.half_excess_u[partition, 0],
                split_result.half_excess_v[partition, 0],
                split_result.half_excess_u[partition, 1],
                split_result.half_excess_v[partition, 1],
                mask,
                split_result.full_occupancy,
            )
            reliability_rows.append(
                {
                    "split": split_result.split,
                    "partition": int(partition),
                    "formal_target_cells": formal_cells,
                    "cell_coverage_fraction": cell_coverage,
                    "occupancy_mass_coverage_fraction": occupancy_mass_coverage,
                    "coverage_valid": coverage_valid,
                    "minimum_half_drift_count_on_available_cells": (
                        int(np.min(split_result.half_drift_count[partition, :, mask]))
                        if np.any(mask)
                        else 0
                    ),
                    "minimum_half_drift_user_count_on_available_cells": (
                        int(np.min(split_result.half_drift_user_count[partition, :, mask]))
                        if np.any(mask)
                        else 0
                    ),
                    **metrics,
                    "vector_spearman_brown": spearman_brown(
                        metrics["vector_correlation"]
                    ),
                    "speed_spearman_brown": spearman_brown(
                        metrics["speed_correlation"]
                    ),
                    "M_spearman_brown": spearman_brown(
                        metrics["M_component_correlation"]
                    ),
                    "Psi_spearman_brown": spearman_brown(
                        metrics["Psi_component_correlation"]
                    ),
                }
            )
    reliability = pd.DataFrame(reliability_rows)

    summary_rows: List[dict] = []
    for split, group in reliability.groupby("split", sort=False):
        valid_group = group[group["coverage_valid"]].copy()
        row: Dict[str, Any] = {
            "split": split,
            "partitions": int(len(group)),
            "coverage_valid_partitions": int(len(valid_group)),
            "coverage_valid_fraction": float(len(valid_group) / max(len(group), 1)),
            "formal_target_cells": formal_cells,
            **quantile_summary(group["cell_coverage_fraction"], "cell_coverage"),
            **quantile_summary(
                group["occupancy_mass_coverage_fraction"],
                "occupancy_mass_coverage",
            ),
            **quantile_summary(
                group["minimum_half_drift_count_on_available_cells"],
                "minimum_half_drift_count_on_available_cells",
            ),
            **quantile_summary(
                group["minimum_half_drift_user_count_on_available_cells"],
                "minimum_half_drift_user_count_on_available_cells",
            ),
        }
        for metric in (
            "supported_cells",
            "vector_correlation",
            "vector_spearman_brown",
            "speed_correlation",
            "speed_spearman_brown",
            "M_component_correlation",
            "M_spearman_brown",
            "Psi_component_correlation",
            "Psi_spearman_brown",
            "weighted_local_cosine",
            "local_cosine_weight_coverage",
        ):
            row.update(quantile_summary(valid_group[metric], metric))
        summary_rows.append(row)
    reliability_summary = pd.DataFrame(summary_rows)

    attenuation_rows: List[dict] = []
    common_masks = np.zeros((partitions,) + formal_mask.shape, dtype=bool)
    for partition in range(partitions):
        common_mask = (
            split_masks["A_val"][partition]
            & split_masks["B_confirm"][partition]
        )
        common_masks[partition] = common_mask
        cell_coverage = float(np.sum(common_mask) / max(formal_cells, 1))
        val_occupancy_coverage = occupancy_coverage(val, common_mask)
        confirm_occupancy_coverage = occupancy_coverage(confirm, common_mask)
        coverage_valid = bool(
            cell_coverage >= float(minimum_cell_coverage)
            and min(val_occupancy_coverage, confirm_occupancy_coverage)
            >= float(minimum_occupancy_coverage)
        )
        val_metrics = field_agreement(
            val.half_excess_u[partition, 0],
            val.half_excess_v[partition, 0],
            val.half_excess_u[partition, 1],
            val.half_excess_v[partition, 1],
            common_mask,
            val.full_occupancy,
        )
        confirm_metrics = field_agreement(
            confirm.half_excess_u[partition, 0],
            confirm.half_excess_v[partition, 0],
            confirm.half_excess_u[partition, 1],
            confirm.half_excess_v[partition, 1],
            common_mask,
            confirm.full_occupancy,
        )
        observed_metrics = field_agreement(
            val.full_exact_excess_u,
            val.full_exact_excess_v,
            confirm.full_exact_excess_u,
            confirm.full_exact_excess_v,
            common_mask,
            val.full_occupancy,
        )
        row: Dict[str, Any] = {
            "partition": int(partition),
            "formal_target_cells": formal_cells,
            "supported_cells": int(np.sum(common_mask)),
            "cell_coverage_fraction": cell_coverage,
            "A_val_occupancy_mass_coverage_fraction": val_occupancy_coverage,
            "B_confirm_occupancy_mass_coverage_fraction": confirm_occupancy_coverage,
            "coverage_valid": coverage_valid,
            "observed_exact_weighted_local_cosine": observed_metrics[
                "weighted_local_cosine"
            ],
        }
        for label, half_column, point_column in (
            ("vector", "vector_correlation", "vector_correlation"),
            ("speed", "speed_correlation", "speed_correlation"),
            ("M", "M_component_correlation", "M_component_correlation"),
            ("Psi", "Psi_component_correlation", "Psi_component_correlation"),
        ):
            r_val = spearman_brown(val_metrics[half_column])
            r_confirm = spearman_brown(confirm_metrics[half_column])
            ceiling = (
                math.sqrt(r_val * r_confirm)
                if np.isfinite(r_val)
                and np.isfinite(r_confirm)
                and r_val > 0
                and r_confirm > 0
                else np.nan
            )
            observed = float(observed_metrics[point_column])
            row[f"{label}_A_val_half_correlation"] = float(val_metrics[half_column])
            row[f"{label}_B_confirm_half_correlation"] = float(
                confirm_metrics[half_column]
            )
            row[f"{label}_A_val_full_sample_reliability"] = r_val
            row[f"{label}_B_confirm_full_sample_reliability"] = r_confirm
            row[f"{label}_attenuation_benchmark"] = ceiling
            row[f"{label}_observed_exact_cross_split_agreement"] = observed
            row[f"{label}_agreement_to_benchmark_ratio"] = (
                observed / ceiling
                if np.isfinite(ceiling) and ceiling > EPS
                else np.nan
            )
        attenuation_rows.append(row)
    attenuation = pd.DataFrame(attenuation_rows)

    minimum_valid_partitions = int(
        math.ceil(partitions * float(minimum_valid_partition_fraction))
    )
    valid_attenuation = attenuation[attenuation["coverage_valid"]].copy()
    if len(valid_attenuation) < minimum_valid_partitions:
        raise RuntimeError(
            "Too few complementary partitions met the prespecified support-coverage gate: "
            f"{len(valid_attenuation)}/{partitions}; required {minimum_valid_partitions}."
        )

    minimum_defined_partitions = int(
        math.ceil(partitions * float(minimum_defined_benchmark_fraction))
    )
    attenuation_summary_rows: List[dict] = []
    for label in ("vector", "speed", "M", "Psi"):
        benchmark_values = pd.to_numeric(
            valid_attenuation[f"{label}_attenuation_benchmark"], errors="coerce"
        )
        ratio_values = pd.to_numeric(
            valid_attenuation[f"{label}_agreement_to_benchmark_ratio"], errors="coerce"
        )
        benchmark_defined = benchmark_values.notna() & ratio_values.notna()
        benchmark_defined_partitions = int(benchmark_defined.sum())
        benchmark_reporting_eligible = bool(
            benchmark_defined_partitions >= minimum_defined_partitions
        )
        benchmark_summary = (
            quantile_summary(
                benchmark_values[benchmark_defined], "attenuation_benchmark"
            )
            if benchmark_reporting_eligible
            else quantile_summary([], "attenuation_benchmark")
        )
        ratio_summary = (
            quantile_summary(
                ratio_values[benchmark_defined], "agreement_to_benchmark_ratio"
            )
            if benchmark_reporting_eligible
            else quantile_summary([], "agreement_to_benchmark_ratio")
        )
        attenuation_summary_rows.append(
            {
                "metric": label,
                "partitions": partitions,
                "coverage_valid_partitions": int(len(valid_attenuation)),
                "benchmark_defined_partitions": benchmark_defined_partitions,
                "minimum_defined_partitions": minimum_defined_partitions,
                "benchmark_reporting_eligible": benchmark_reporting_eligible,
                "formal_target_cells": formal_cells,
                **quantile_summary(
                    valid_attenuation[f"{label}_observed_exact_cross_split_agreement"],
                    "observed_exact_cross_split_agreement",
                ),
                **benchmark_summary,
                **ratio_summary,
            }
        )
    attenuation_summary = pd.DataFrame(attenuation_summary_rows)
    audit = {
        "formal_common_support_cells": formal_cells,
        "minimum_cell_coverage": float(minimum_cell_coverage),
        "minimum_occupancy_coverage": float(minimum_occupancy_coverage),
        "minimum_valid_partition_fraction": float(minimum_valid_partition_fraction),
        "minimum_defined_benchmark_fraction": float(minimum_defined_benchmark_fraction),
        "partition_exclusion_policy": (
            "all prespecified partitions remain in the raw output; summary benchmarks use the "
            "coverage-valid subset defined only by prespecified row, learner, cell and occupancy "
            "adequacy, and at least the declared partition fraction must pass"
        ),
        "minimum_valid_partitions": minimum_valid_partitions,
        "minimum_defined_partitions": minimum_defined_partitions,
        "minimum_half_drift_count": int(minimum_half_drift_count),
        "minimum_half_cell_users": int(minimum_half_cell_users),
        "half_sample_adequacy_source": (
            "ceil(one-half of the formal Stage-1 drift-row and per-cell learner support thresholds) "
            "unless explicitly overridden"
        ),
        "coverage_valid_common_partitions": int(len(valid_attenuation)),
        "reliability_support_definition": (
            "the formal 917-cell A_val/B_confirm common drift support is the fixed target; "
            "each complementary partition is evaluated on cells with finite estimates in both "
            "halves that meet the declared half-sample row and learner adequacy thresholds; "
            "cell and full-split occupancy coverage are prespecified implementation-quality gates"
        ),
        "B_confirm_role_in_support": (
            "B_confirm contributes only output-side half-estimate availability on the already "
            "formal common support; it does not define partition strata, thresholds or the "
            "formal target support"
        ),
        "local_cosine_weight_contract": "A_val full-split user-balanced occupancy",
        "attenuation_interpretation": (
            "Spearman-Brown split-half attenuation benchmark; repeated complementary partitions "
            "are sensitivity replicates, not independent confidence-interval draws; ratios are "
            "not truncated and are undefined when either estimated reliability is non-positive; "
            "benchmark and ratio summaries are withheld unless the declared number of partitions is defined"
        ),
    }
    return points, reliability, reliability_summary, attenuation, {
        "summary": attenuation_summary,
        "audit": audit,
        "formal_mask": formal_mask,
        "A_val_partition_masks": split_masks["A_val"],
        "B_confirm_partition_masks": split_masks["B_confirm"],
        "common_partition_masks": common_masks,
    }

def source_hash_table(paths: Sequence[Path]) -> pd.DataFrame:
    rows = []
    seen = set()
    for source in paths:
        path = Path(source).resolve()
        if path in seen:
            continue
        seen.add(path)
        rows.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "bytes": int(path.stat().st_size),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output root is not empty: {output_root}; pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    table_root = output_root / "tables"
    array_root = output_root / "arrays"
    metadata_root = output_root / "metadata"
    for directory in (table_root, array_root, metadata_root):
        directory.mkdir(parents=True, exist_ok=True)

    stage1_script = resolve_script(args.stage1_script, DEFAULT_STAGE1_SCRIPT)
    cmn_script = resolve_script(
        args.construction_null_script, DEFAULT_CONSTRUCTION_NULL_SCRIPT
    )
    stage1 = import_module(stage1_script, "formal_stage1_empirical_reliability")
    cmn = import_module(cmn_script, "formal_construction_null_reliability")
    validate_module_contracts(stage1, cmn)
    specification = stage1.coordinate_specs()[0]
    if specification.name != PRIMARY_COORDINATE:
        raise RuntimeError(f"Expected {PRIMARY_COORDINATE}, found {specification.name}.")
    stage1_root = args.stage1_root.resolve()
    cmn_root = args.construction_null_output_root.resolve()
    cmn_confirm_root = args.construction_null_confirm_output_root.resolve()

    cutpoints, cutpoint_payload, cutpoint_path = load_matching_cutpoints(cmn, cmn_root)
    val_manifest, val_manifest_path = load_cmn_manifest(cmn_root, "A_val")
    confirm_manifest, confirm_manifest_path = load_cmn_manifest(
        cmn_confirm_root, "B_confirm"
    )
    stage1_script_sha = file_sha256(stage1_script)
    cmn_script_sha = file_sha256(cmn_script)
    expected_cmn_sha = str(args.expected_construction_null_script_sha256).strip().lower()
    if expected_cmn_sha and cmn_script_sha.lower() != expected_cmn_sha:
        raise RuntimeError(
            "The supplied construction-null implementation does not match the reviewed formal source: "
            f"expected {expected_cmn_sha}, found {cmn_script_sha}."
        )
    for split, payload in (("A_val", val_manifest), ("B_confirm", confirm_manifest)):
        if str(payload.get("analysis_split", "")) != split:
            raise RuntimeError(f"Construction-null manifest split mismatch for {split}.")
        if int(payload.get("replicates", -1)) != 100:
            raise RuntimeError(f"The formal {split} construction-null output does not use 100 permutations.")
        if list(payload.get("primary_coordinates", [])) != ["M", "Psi"]:
            raise RuntimeError(f"Construction-null coordinate contract changed for {split}.")
        if str(payload.get("formal_stage1_script_sha256", "")) != stage1_script_sha:
            raise RuntimeError(f"The supplied Stage-1 script does not match the formal {split} null manifest.")
    if bool(confirm_manifest.get("confirmation_output_only", False)) is not True:
        raise RuntimeError("The B_confirm construction-null manifest is not output-only.")
    if stable_json_hash(val_manifest["matching_cutpoints"]) != stable_json_hash(
        confirm_manifest["matching_cutpoints"]
    ):
        raise RuntimeError("A_val and B_confirm construction-null cutpoints differ.")
    if stable_json_hash(val_manifest["matching_cutpoints"]) != stable_json_hash(
        cutpoint_payload["cutpoints"]
    ):
        raise RuntimeError("Archived matching-cutpoint metadata differs from the formal manifests.")

    print("[analysis] running A_val split-half exact-null fields", flush=True)
    val, partition_cutpoints = run_split_reliability(
        split="A_val",
        stage1_root=stage1_root,
        stage1=stage1,
        cmn=cmn,
        specification=specification,
        cutpoints=cutpoints,
        cmn_root=cmn_root,
        partition_cutpoints=None,
        partitions=args.split_half_partitions,
        partition_seed=args.split_half_seed,
        max_last_resort_fraction=args.max_last_resort_fraction,
        confirmation_output_only=args.confirmation_output_only,
        progress_every=args.progress_every,
    )
    print("[analysis] running output-only B_confirm split-half exact-null fields", flush=True)
    confirm, _ = run_split_reliability(
        split="B_confirm",
        stage1_root=stage1_root,
        stage1=stage1,
        cmn=cmn,
        specification=specification,
        cutpoints=cutpoints,
        cmn_root=cmn_confirm_root,
        partition_cutpoints=partition_cutpoints,
        partitions=args.split_half_partitions,
        partition_seed=args.split_half_seed,
        max_last_resort_fraction=args.max_last_resort_fraction,
        confirmation_output_only=args.confirmation_output_only,
        progress_every=args.progress_every,
    )

    minimum_half_drift_count = (
        int(args.minimum_half_drift_count)
        if int(args.minimum_half_drift_count) > 0
        else int(math.ceil(float(stage1.MIN_DRIFT_BIN_COUNT) / 2.0))
    )
    minimum_half_cell_users = (
        int(args.minimum_half_cell_users)
        if int(args.minimum_half_cell_users) > 0
        else int(math.ceil(float(stage1.MIN_CELL_USERS) / 2.0))
    )
    points, reliability, reliability_summary, attenuation, attenuation_payload = reliability_tables(
        val,
        confirm,
        stage1,
        minimum_cell_coverage=args.minimum_half_cell_coverage,
        minimum_occupancy_coverage=args.minimum_half_occupancy_coverage,
        minimum_valid_partition_fraction=args.minimum_valid_partition_fraction,
        minimum_defined_benchmark_fraction=args.minimum_defined_benchmark_fraction,
        minimum_half_drift_count=minimum_half_drift_count,
        minimum_half_cell_users=minimum_half_cell_users,
    )
    attenuation_summary = attenuation_payload["summary"]
    reliability_audit = attenuation_payload["audit"]
    formal_mask = attenuation_payload["formal_mask"]
    val_partition_masks = attenuation_payload["A_val_partition_masks"]
    confirm_partition_masks = attenuation_payload["B_confirm_partition_masks"]
    common_partition_masks = attenuation_payload["common_partition_masks"]

    partition_balance = pd.concat(
        [val.partition_balance, confirm.partition_balance], ignore_index=True
    )
    partition_bin_balance = pd.concat(
        [val.partition_bin_balance, confirm.partition_bin_balance], ignore_index=True
    )
    matching_quality = pd.concat(
        [val.matching_quality, confirm.matching_quality], ignore_index=True
    )
    write_table(points, table_root / "formal_finite_and_exact_excess_field_points")
    write_table(partition_balance, table_root / "split_half_partition_balance")
    write_table(partition_bin_balance, table_root / "split_half_activity_bin_balance")
    write_table(matching_quality, table_root / "split_half_matching_quality")
    write_table(reliability, table_root / "split_half_reliability_replicates")
    write_table(reliability_summary, table_root / "split_half_reliability_summary")
    write_table(attenuation, table_root / "attenuation_benchmark_replicates")
    write_table(attenuation_summary, table_root / "attenuation_benchmark_summary")
    np.savez_compressed(
        array_root / "excess_split_half_fields.npz",
        A_val_half_excess_u=val.half_excess_u,
        A_val_half_excess_v=val.half_excess_v,
        B_confirm_half_excess_u=confirm.half_excess_u,
        B_confirm_half_excess_v=confirm.half_excess_v,
        A_val_half_drift_count=val.half_drift_count,
        A_val_half_drift_user_count=val.half_drift_user_count,
        B_confirm_half_drift_count=confirm.half_drift_count,
        B_confirm_half_drift_user_count=confirm.half_drift_user_count,
        A_val_full_exact_excess_u=val.full_exact_excess_u,
        A_val_full_exact_excess_v=val.full_exact_excess_v,
        B_confirm_full_exact_excess_u=confirm.full_exact_excess_u,
        B_confirm_full_exact_excess_v=confirm.full_exact_excess_v,
        formal_common_support=formal_mask,
        A_val_partition_availability=val_partition_masks,
        B_confirm_partition_availability=confirm_partition_masks,
        common_partition_availability=common_partition_masks,
        A_val_occupancy=val.full_occupancy,
        B_confirm_occupancy=confirm.full_occupancy,
    )

    print("[analysis] running learner-composition soft-core audit", flush=True)
    soft_replicates, soft_summary, soft_cells, soft_payload = run_soft_core(
        stage1=stage1,
        stage1_root=stage1_root,
        specification=specification,
        output_root=output_root,
        replicates=args.soft_core_replicates,
        batch_size=args.soft_core_batch_size,
        seed=args.soft_core_seed,
        progress_every=args.progress_every,
    )
    write_table(soft_replicates, table_root / "soft_core_replicates")
    write_table(soft_summary, table_root / "soft_core_summary")
    write_table(soft_cells, table_root / "soft_core_cell_selection_frequency")
    write_table(
        soft_payload["threshold_summary"], table_root / "soft_core_threshold_summary"
    )

    region_root = (
        stage1_root
        / "dynamics"
        / "candidate_regions"
        / PRIMARY_COORDINATE
    )
    source_paths = [
        stage1_script,
        cmn_script,
        cutpoint_path,
        val_manifest_path,
        confirm_manifest_path,
        cmn_root / "arrays" / "A_val_construction_null_fields.npz",
        cmn_confirm_root / "arrays" / "B_confirm_construction_null_fields.npz",
        region_root / "training_convergence_thresholds.json",
        find_table(region_root / "training_flow_defined_convergence_regions"),
        find_table(region_root / "validation_flow_defined_convergence_regions_fixed_thresholds"),
        find_table(region_root / "training_validation_convergence_region_reproducibility"),
        region_root / "A_train_primary_convergence_core_mask.npy",
    ]
    source_hashes = source_hash_table(source_paths)
    write_table(source_hashes, table_root / "source_file_hashes")

    weak_max = float(matching_quality["weak_fallback_fraction_of_assigned"].max())
    exact_moment_max = float(
        matching_quality[
            [
                column
                for column in matching_quality.columns
                if column.startswith("mean_") and column.endswith("preservation_error")
            ]
        ].max().max()
    )
    quality_gates = {
        "formal_stage1_script_matches_construction_null_manifests_passed": True,
        "formal_construction_null_100_permutations_passed": True,
        "formal_construction_null_script_reviewed_sha256_passed": (
            not expected_cmn_sha or cmn_script_sha.lower() == expected_cmn_sha
        ),
        "formal_common_support_reproduced_passed": int(np.sum(formal_mask)) == FORMAL_EXCESS_CELLS,
        "partition_availability_masks_within_formal_support_passed": bool(
            np.all(~val_partition_masks | formal_mask[None, :, :])
            and np.all(~confirm_partition_masks | formal_mask[None, :, :])
            and np.all(~common_partition_masks | formal_mask[None, :, :])
        ),
        "formal_finite_vector_correlation_reproduced": abs(
            float(
                points[
                    (points["estimator"] == "formal_100_permutation_mean")
                    & (points["support"] == "formal_common_support")
                ].iloc[0]["vector_correlation"]
            )
            - FORMAL_EXCESS_VECTOR_CORR
        )
        <= 5e-4,
        "formal_finite_speed_correlation_reproduced": abs(
            float(
                points[
                    (points["estimator"] == "formal_100_permutation_mean")
                    & (points["support"] == "formal_common_support")
                ].iloc[0]["speed_correlation"]
            )
            - FORMAL_EXCESS_SPEED_CORR
        )
        <= 5e-4,
        "formal_finite_local_cosine_reproduced": abs(
            float(
                points[
                    (points["estimator"] == "formal_100_permutation_mean")
                    & (points["support"] == "formal_common_support")
                ].iloc[0]["weighted_local_cosine"]
            )
            - FORMAL_EXCESS_LOCAL_COSINE
        )
        <= 1e-3,
        "split_half_partition_coverage_gate_passed": (
            int(reliability_audit["coverage_valid_common_partitions"])
            >= int(reliability_audit["minimum_valid_partitions"])
        ),
        "vector_attenuation_benchmark_reporting_eligible": bool(
            attenuation_summary.loc[
                attenuation_summary["metric"] == "vector",
                "benchmark_reporting_eligible",
            ].iloc[0]
        ),
        "speed_attenuation_benchmark_reporting_eligible": bool(
            attenuation_summary.loc[
                attenuation_summary["metric"] == "speed",
                "benchmark_reporting_eligible",
            ].iloc[0]
        ),
        "maximum_half_weak_fallback_fraction": weak_max,
        "half_weak_fallback_gate_passed": weak_max <= args.max_last_resort_fraction,
        "maximum_exact_donor_moment_error": exact_moment_max,
        "exact_donor_moment_gate_passed": exact_moment_max <= 1e-10,
        "core_all_ones_field_gate_passed": soft_payload["formal_audit"][
            "max_abs_all_ones_field_error"
        ]
        <= 1e-10,
        "core_all_ones_mask_gate_passed": bool(
            soft_payload["formal_audit"]["training_primary_mask_exact_match"]
        ),
        "B_confirm_used_for_partition_cutpoint_fitting": False,
        "B_confirm_used_for_core_definition": False,
        "model_outputs_read": False,
        "formal_outputs_modified": False,
    }
    failed = [
        key
        for key, value in quality_gates.items()
        if key.endswith("passed") and not bool(value)
    ]
    if failed:
        raise RuntimeError(f"Analysis quality gates failed: {failed}")

    script_path = Path(__file__).resolve()
    manifest = {
        "script": script_path.name,
        "script_sha256": file_sha256(script_path),
        "created_at": now_string(),
        "runtime_seconds": float(time.time() - started),
        "analysis": "empirical excess-field split-half attenuation benchmark and soft-core selection stability",
        "analysis_status": "post hoc supplementary robustness audit",
        "stage1_root": str(stage1_root),
        "construction_null_output_root": str(cmn_root),
        "construction_null_confirm_output_root": str(cmn_confirm_root),
        "construction_null_script_sha256": cmn_script_sha,
        "expected_construction_null_script_sha256": expected_cmn_sha or None,
        "primary_coordinates": ["M", "Psi"],
        "data_splits": ["A_train", "A_val", "B_confirm"],
        "B_confirm_policy": (
            "output-only reliability evaluation; partition algorithm and activity cutpoints frozen from A_val; "
            "no threshold, coordinate, core, null protocol or model update"
        ),
        "excess_reliability_contract": {
            "partitions": int(args.split_half_partitions),
            "base_seed": int(args.split_half_seed),
            "partition_activity_cutpoints_fit_split": "A_val",
            "partition_activity_variable": "valid drift rows per learner",
            "partition_cutpoints": partition_cutpoints,
            "whole_user_complementary_halves": True,
            "matching_cutpoints_source": str(cutpoint_path.resolve()),
            "null_rebuilt_within_every_half": True,
            "exact_null_role": "removes finite-permutation Monte Carlo noise from the reliability estimator",
            "formal_finite_permutation_inference_modified": False,
            **reliability_audit,
        },
        "soft_core_contract": {
            "replicates": int(args.soft_core_replicates),
            "batch_size": int(args.soft_core_batch_size),
            "seed": int(args.soft_core_seed),
            "multiplier_distribution": "independent Exp(1) learner weights normalized to mean one within split and replicate",
            "formal_support_masks_fixed": True,
            "A_train_thresholds_reestimated_per_replicate": True,
            "A_train_primary_core_reselected_per_replicate": True,
            "A_val_threshold_transfer": "replicate A_train thresholds applied unchanged",
            "fallback_allowed": False,
            "frequency_denominators": soft_payload["formal_audit"].get(
                "frequency_denominators", {}
            ),
            "frequency_interpretation": (
                "unconditional frequencies combine region detectability and spatial stability; "
                "paired-available and paired-qualified maps condition on explicit availability "
                "and dynamical-qualification events and are not posterior probabilities"
            ),
        },
        "formal_reconstruction_audits": {
            "A_val": val.formal_audit,
            "B_confirm": confirm.formal_audit,
            "soft_core": soft_payload["formal_audit"],
        },
        "quality_gates": quality_gates,
        "source_hashes_table": str(find_table(table_root / "source_file_hashes").resolve()),
        "outputs": {
            "tables": sorted(str(path.resolve()) for path in table_root.iterdir()),
            "arrays": sorted(str(path.resolve()) for path in array_root.iterdir()),
        },
        "interpretation_boundary": (
            "The split-half correction is an attenuation benchmark under repeated user partitions, not a latent-truth estimate. "
            "Soft-core maps quantify selection stability under fixed-support learner reweighting and do not redefine the formal core."
        ),
        "no_preregistration_claim": True,
    }
    manifest_path = metadata_root / "empirical_excess_reliability_soft_core_manifest.json"
    save_json(manifest, manifest_path)
    save_json(
        {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": file_sha256(manifest_path),
        },
        metadata_root / "empirical_excess_reliability_soft_core_manifest.sha256.json",
    )
    print(f"[analysis] completed: {output_root}", flush=True)


def self_test() -> None:
    class Layout:
        name = "test"
        original_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
        order = np.asarray([0, 1, 2, 3], dtype=np.int64)
        sorted_group = np.asarray([0, 0, 1, 1], dtype=np.int32)
        starts = np.asarray([0, 2], dtype=np.int64)
        counts = np.asarray([2, 2], dtype=np.int64)

    z_m = np.asarray([1.0, 3.0, 5.0, 7.0])
    z_p = np.asarray([2.0, 4.0, 6.0, 8.0])
    expectation = exact_expected_donor_pairs([Layout()], np.ones(4, dtype=bool), z_m, z_p)
    if not np.allclose(expectation.expected_z_m, [3.0, 1.0, 7.0, 5.0]):
        raise RuntimeError("Exact-expectation self-test failed.")
    users = np.arange(20, dtype=np.int64)
    activity = np.arange(20, dtype=np.float64)
    sides, balance = assign_complementary_halves(
        users, activity, activity_cutpoints(activity), 123
    )
    if set(np.unique(sides)) != {0, 1} or int(balance["users"].sum()) != 20:
        raise RuntimeError("Complementary-half self-test failed.")
    first_u = np.asarray([[1.0, 0.0], [0.0, -1.0]])
    first_v = np.asarray([[0.0, 1.0], [-1.0, 0.0]])
    metrics = field_agreement(
        first_u,
        first_v,
        first_u,
        first_v,
        np.ones((2, 2), dtype=bool),
        np.ones((2, 2), dtype=float),
    )
    if not np.isclose(metrics["vector_correlation"], 1.0):
        raise RuntimeError("Field-agreement self-test failed.")
    print("[self-test] exact expectation, complementary partition and field metrics passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the empirical excess-field reliability and soft-core stability audit."
    )
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--stage1-script", type=Path, default=None)
    parser.add_argument("--construction-null-script", type=Path, default=None)
    parser.add_argument(
        "--expected-construction-null-script-sha256",
        type=str,
        default=EXPECTED_FORMAL_CONSTRUCTION_NULL_SHA256,
    )
    parser.add_argument(
        "--construction-null-output-root",
        type=Path,
        default=DEFAULT_CONSTRUCTION_NULL_ROOT,
        help="Root containing the formal A_val construction-null manifest and arrays.",
    )
    parser.add_argument(
        "--construction-null-confirm-output-root",
        type=Path,
        default=DEFAULT_CONSTRUCTION_NULL_CONFIRM_ROOT,
        help="Root containing the output-only B_confirm construction-null manifest and arrays.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split-half-partitions", type=int, default=32)
    parser.add_argument("--split-half-seed", type=int, default=20260804)
    parser.add_argument("--soft-core-replicates", type=int, default=1000)
    parser.add_argument("--soft-core-batch-size", type=int, default=10)
    parser.add_argument("--soft-core-seed", type=int, default=20260731)
    parser.add_argument("--max-last-resort-fraction", type=float, default=0.01)
    parser.add_argument("--minimum-half-cell-coverage", type=float, default=0.95)
    parser.add_argument("--minimum-half-occupancy-coverage", type=float, default=0.98)
    parser.add_argument("--minimum-valid-partition-fraction", type=float, default=0.90)
    parser.add_argument("--minimum-defined-benchmark-fraction", type=float, default=0.90)
    parser.add_argument("--minimum-half-drift-count", type=int, default=0)
    parser.add_argument("--minimum-half-cell-users", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--confirmation-output-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return
    if args.split_half_partitions < 8:
        raise ValueError("Use at least eight complementary split-half partitions.")
    if args.soft_core_replicates < 100:
        raise ValueError("Use at least 100 soft-core learner-composition replicates.")
    if args.soft_core_batch_size < 1:
        raise ValueError("--soft-core-batch-size must be positive.")
    for name, value in (
        ("--minimum-half-cell-coverage", args.minimum_half_cell_coverage),
        ("--minimum-half-occupancy-coverage", args.minimum_half_occupancy_coverage),
        ("--minimum-valid-partition-fraction", args.minimum_valid_partition_fraction),
        ("--minimum-defined-benchmark-fraction", args.minimum_defined_benchmark_fraction),
    ):
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{name} must lie in (0,1].")
    if int(args.minimum_half_drift_count) < 0 or int(args.minimum_half_cell_users) < 0:
        raise ValueError("Half-sample adequacy overrides must be non-negative.")
    if not args.confirmation_output_only:
        raise ValueError("Use --confirmation-output-only for B_confirm reliability evaluation.")
    run(args)


if __name__ == "__main__":
    main()
