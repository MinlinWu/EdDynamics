#!/usr/bin/env python3
from __future__ import annotations

"""Construction-matched null for the frozen EdNet-KT4 M-Psi dynamics.

The analysis preserves each observed current state, response/exposure denominator
increment, user-balanced weight, grid, support mask and A_train-defined
convergence core. It jointly reassigns the normalized signed response and
exposure innovations away from their observed states under a hierarchical,
opportunity-matched permutation. No coordinate, threshold, region, mesostate or
model is refitted.

Primary manuscript use is A_val. B_confirm is available only through the
explicit output-only flag after the null protocol and interpretation have been
frozen.
"""

import argparse
import dataclasses
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
RECONSTRUCTION_ATOL = 2e-6
RECONSTRUCTION_RTOL = 2e-6
Z_BOUND_TOL = 2e-6
DEFAULT_STAGE1_ROOT = Path("/data/datasets/KT4/outputs_KT4/stage1")
DEFAULT_OUTPUT_ROOT = Path("/data/datasets/KT4/outputs_KT4/stage1_construction_matched_null")
DEFAULT_STAGE1_SCRIPT_BASENAME = "build_effective_dynamics_kt4_stage1_empirical.py"
PRIMARY_COORDINATE = "MR_PsiA"


# -----------------------------------------------------------------------------
# Generic I/O and audit helpers
# -----------------------------------------------------------------------------
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_table(base: Path) -> Path:
    for extension in (".parquet", ".csv.gz", ".csv"):
        path = base.with_suffix(extension)
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find table for {base}.[parquet|csv.gz|csv]")


def available_columns(base: Path) -> List[str]:
    path = find_table(base)
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            return list(pq.read_schema(path).names)
        except Exception:
            return list(pd.read_parquet(path).columns)
    return list(pd.read_csv(path, nrows=0).columns)


def read_table(base: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    path = find_table(base)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    return pd.read_csv(path, usecols=list(columns) if columns is not None else None, low_memory=False)


def iter_table_chunks(
    base: Path,
    columns: Sequence[str],
    chunk_rows: int,
) -> Iterable[pd.DataFrame]:
    path = find_table(base)
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(columns=list(columns), batch_size=int(chunk_rows)):
                yield batch.to_pandas()
            return
        except Exception:
            yield pd.read_parquet(path, columns=list(columns))
            return
    yield from pd.read_csv(
        path,
        usecols=list(columns),
        low_memory=False,
        chunksize=int(chunk_rows),
    )


def write_table(df: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    parquet = base.with_suffix(".parquet")
    tmp = parquet.with_name(parquet.name + ".tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, parquet)
        return parquet
    except Exception:
        if tmp.exists():
            tmp.unlink()
        csv_path = base.with_suffix(".csv.gz")
        tmp_csv = csv_path.with_name(csv_path.name + ".tmp")
        df.to_csv(tmp_csv, index=False, compression="gzip")
        os.replace(tmp_csv, csv_path)
        return csv_path


def import_stage1_module(path: Path):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Formal Stage-1 script not found: {path}")
    spec = importlib.util.spec_from_file_location("formal_stage1_empirical", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import formal Stage-1 script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = {
        "CoordinateSpec",
        "FieldStats",
        "coordinate_specs",
        "occupancy_drift_stats",
        "field_stats_from_dict",
        "field_grid_table",
        "global_field_contraction_summary",
        "evaluate_frozen_convergence_region",
        "user_balanced_weights",
        "interior_divergence",
        "downcast_frame",
    }
    missing = sorted(name for name in required if not hasattr(module, name))
    if missing:
        raise RuntimeError(f"Formal Stage-1 script is missing required functions: {missing}")
    return module


def resolve_stage1_script(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit.resolve()
    sibling = Path(__file__).resolve().with_name(DEFAULT_STAGE1_SCRIPT_BASENAME)
    if sibling.exists():
        return sibling
    raise FileNotFoundError(
        "Pass --stage1-script with the formal empirical-dynamics script. "
        f"No sibling {DEFAULT_STAGE1_SCRIPT_BASENAME!r} was found."
    )


def numeric_array(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        raise KeyError(f"Required column not found: {column}")
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)


def integer_array(df: pd.DataFrame, column: str, default: int = -1) -> np.ndarray:
    if column not in df.columns:
        return np.full(len(df), int(default), dtype=np.int64)
    values = pd.to_numeric(df[column], errors="coerce").fillna(default)
    return values.to_numpy(dtype=np.int64)


def resolve_off_target_column(columns: Sequence[str], suffix: str) -> str:
    preferred = f"activity_off_target_mass_{suffix}"
    fallback = f"activity_non_aligned_mass_{suffix}"
    if preferred in columns:
        return preferred
    if fallback in columns:
        return fallback
    raise KeyError(f"Neither {preferred!r} nor {fallback!r} is present.")


def stable_uint64_priority(user_id: np.ndarray, step: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic splitmix64 priority for streaming A_train sampling."""
    x = np.asarray(user_id, dtype=np.uint64)
    y = np.asarray(step, dtype=np.uint64)
    z = x ^ (y * np.uint64(0x9E3779B97F4A7C15)) ^ np.uint64(int(seed) & ((1 << 64) - 1))
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


# -----------------------------------------------------------------------------
# Exact same-row reconstruction of Stage-1 innovations
# -----------------------------------------------------------------------------
@dataclass
class InnovationArrays:
    user_id: np.ndarray
    step: np.ndarray
    part: np.ndarray
    gap_days: np.ndarray
    m_pre: np.ndarray
    psi_pre: np.ndarray
    e_pre: np.ndarray
    s_pre: np.ndarray
    b_pre: np.ndarray
    g_pre: np.ndarray
    a_m: np.ndarray
    j_m: np.ndarray
    z_m: np.ndarray
    a_psi: np.ndarray
    j_psi: np.ndarray
    z_psi: np.ndarray
    response_active: np.ndarray
    support_active: np.ndarray
    idle_mass: np.ndarray
    next_m: np.ndarray
    next_psi: np.ndarray
    observed_dx: np.ndarray
    observed_dy: np.ndarray
    formal_drift_valid: np.ndarray
    reconstruction_audit: Dict[str, Any]


def _max_abs_difference(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.max(np.abs(np.asarray(first)[mask] - np.asarray(second)[mask])))


def _max_relative_difference(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    numerator = np.abs(np.asarray(first)[mask] - np.asarray(second)[mask])
    denominator = np.maximum(np.maximum(np.abs(np.asarray(first)[mask]), np.abs(np.asarray(second)[mask])), EPS)
    return float(np.max(numerator / denominator))


def reconstruct_innovations(
    df: pd.DataFrame,
    tau_response_days: float,
    tau_activity_days: float,
    require_next_audit: bool,
) -> InnovationArrays:
    columns = list(df.columns)
    off_pre = resolve_off_target_column(columns, "pre")
    off_post = resolve_off_target_column(columns, "post")
    next_off = (
        "next_activity_off_target_mass"
        if "next_activity_off_target_mass" in columns
        else "next_activity_non_aligned_mass"
    )
    if require_next_audit and next_off not in columns:
        raise KeyError("Next activity off-target/non-aligned mass is required for the reconstruction audit.")

    user_id = integer_array(df, "user_id")
    step = integer_array(df, "bundle_step_index")
    part = integer_array(df, "part")
    gap_days = numeric_array(df, "next_gap_days")

    m_pre = numeric_array(df, "M_response_prebalanced_pre")
    psi_pre = numeric_array(df, "activity_alignment_order_Psi_pre")
    e_pre = numeric_array(df, "response_evidence_mass_pre")
    m_resp = numeric_array(df, "M_response_prebalanced_resp")
    e_resp = numeric_array(df, "response_evidence_mass_resp")

    active_pre = numeric_array(df, "activity_active_mass_pre")
    aligned_pre = numeric_array(df, "activity_aligned_mass_pre")
    off_pre_values = numeric_array(df, off_pre)
    idle_pre = numeric_array(df, "activity_idle_mass_pre")
    active_post = numeric_array(df, "activity_active_mass_post")
    aligned_post = numeric_array(df, "activity_aligned_mass_post")
    off_post_values = numeric_array(df, off_post)
    idle_post = numeric_array(df, "activity_idle_mass_post")

    next_m = numeric_array(df, "next_M_response_prebalanced")
    next_psi = numeric_array(df, "next_activity_alignment_order_Psi")
    observed_dx = numeric_array(df, "delta_M_response_prebalanced_next")
    observed_dy = numeric_array(df, "delta_activity_alignment_order_Psi_next")

    response_active = np.maximum(numeric_array(df, "response_active_mass_interval"), 0.0)
    support_active = np.maximum(numeric_array(df, "support_active_total_interval"), 0.0)
    idle_mass = np.maximum(numeric_array(df, "idle_mass_interval"), 0.0)

    formal_drift_valid = (
        np.isfinite(m_pre)
        & np.isfinite(psi_pre)
        & np.isfinite(observed_dx)
        & np.isfinite(observed_dy)
    )

    s_pre = m_pre * e_pre
    s_resp = m_resp * e_resp
    b_pre = active_pre + idle_pre
    g_pre = aligned_pre - off_pre_values
    b_post = active_post + idle_post
    g_post = aligned_post - off_post_values

    a_m = e_resp - e_pre
    j_m = s_resp - s_pre
    a_psi = b_post - b_pre
    j_psi = g_post - g_pre

    ingredient_valid = (
        np.isfinite(e_pre)
        & np.isfinite(m_resp)
        & np.isfinite(e_resp)
        & np.isfinite(b_pre)
        & np.isfinite(g_pre)
        & np.isfinite(b_post)
        & np.isfinite(g_post)
        & np.isfinite(a_m)
        & np.isfinite(j_m)
        & np.isfinite(a_psi)
        & np.isfinite(j_psi)
    )
    missing_formal = formal_drift_valid & ~ingredient_valid
    if np.any(missing_formal):
        raise RuntimeError(
            "Formal drift rows are missing phase-resolved accounting ingredients: "
            f"{int(np.sum(missing_formal))} rows."
        )

    strongly_negative_m = formal_drift_valid & (a_m < -RECONSTRUCTION_ATOL)
    strongly_negative_psi = formal_drift_valid & (a_psi < -RECONSTRUCTION_ATOL)
    if np.any(strongly_negative_m) or np.any(strongly_negative_psi):
        raise RuntimeError(
            "Negative denominator increments exceed tolerance: "
            f"response={int(np.sum(strongly_negative_m))}, exposure={int(np.sum(strongly_negative_psi))}."
        )
    a_m = np.where((a_m < 0) & (a_m >= -RECONSTRUCTION_ATOL), 0.0, a_m)
    a_psi = np.where((a_psi < 0) & (a_psi >= -RECONSTRUCTION_ATOL), 0.0, a_psi)

    z_m = np.zeros(len(df), dtype=np.float64)
    z_psi = np.zeros(len(df), dtype=np.float64)
    positive_m = formal_drift_valid & (a_m > EPS)
    positive_psi = formal_drift_valid & (a_psi > EPS)
    z_m[positive_m] = j_m[positive_m] / a_m[positive_m]
    z_psi[positive_psi] = j_psi[positive_psi] / a_psi[positive_psi]

    bad_zero_m = formal_drift_valid & (a_m <= EPS) & (np.abs(j_m) > RECONSTRUCTION_ATOL)
    bad_zero_psi = formal_drift_valid & (a_psi <= EPS) & (np.abs(j_psi) > RECONSTRUCTION_ATOL)
    if np.any(bad_zero_m) or np.any(bad_zero_psi):
        raise RuntimeError(
            "Non-zero signed innovation with zero denominator increment: "
            f"response={int(np.sum(bad_zero_m))}, exposure={int(np.sum(bad_zero_psi))}."
        )

    z_m_excess = np.maximum(np.abs(z_m[formal_drift_valid]) - 1.0, 0.0)
    z_psi_excess = np.maximum(np.abs(z_psi[formal_drift_valid]) - 1.0, 0.0)
    max_z_m_excess = float(np.max(z_m_excess)) if z_m_excess.size else 0.0
    max_z_psi_excess = float(np.max(z_psi_excess)) if z_psi_excess.size else 0.0
    if max_z_m_excess > Z_BOUND_TOL or max_z_psi_excess > Z_BOUND_TOL:
        raise RuntimeError(
            "Normalized innovations exceed [-1,1] beyond numerical tolerance: "
            f"response_excess={max_z_m_excess:.3e}, exposure_excess={max_z_psi_excess:.3e}."
        )
    z_m = np.clip(z_m, -1.0, 1.0)
    z_psi = np.clip(z_psi, -1.0, 1.0)

    response_denominator = e_pre + a_m
    exposure_denominator = b_pre + a_psi
    reconstructed_next_m = np.full(len(df), np.nan, dtype=np.float64)
    reconstructed_next_psi = np.full(len(df), np.nan, dtype=np.float64)
    valid_m_denominator = formal_drift_valid & (response_denominator > EPS)
    valid_psi_denominator = formal_drift_valid & (exposure_denominator > EPS)
    reconstructed_next_m[valid_m_denominator] = (
        s_pre[valid_m_denominator] + j_m[valid_m_denominator]
    ) / response_denominator[valid_m_denominator]
    reconstructed_next_psi[valid_psi_denominator] = (
        g_pre[valid_psi_denominator] + j_psi[valid_psi_denominator]
    ) / exposure_denominator[valid_psi_denominator]

    coordinate_audit_mask = formal_drift_valid & valid_m_denominator & valid_psi_denominator
    max_abs_m = _max_abs_difference(reconstructed_next_m, next_m, coordinate_audit_mask)
    max_abs_psi = _max_abs_difference(reconstructed_next_psi, next_psi, coordinate_audit_mask)
    max_abs_dx = _max_abs_difference(reconstructed_next_m - m_pre, observed_dx, coordinate_audit_mask)
    max_abs_dy = _max_abs_difference(reconstructed_next_psi - psi_pre, observed_dy, coordinate_audit_mask)
    if any(value > RECONSTRUCTION_ATOL for value in (max_abs_m, max_abs_psi, max_abs_dx, max_abs_dy)):
        raise RuntimeError(
            "Same-row innovation reconstruction does not reproduce the frozen next coordinates: "
            f"M={max_abs_m:.3e}, Psi={max_abs_psi:.3e}, dM={max_abs_dx:.3e}, dPsi={max_abs_dy:.3e}."
        )

    audit: Dict[str, Any] = {
        "rows": int(len(df)),
        "formal_drift_rows": int(np.sum(formal_drift_valid)),
        "response_positive_increment_rows": int(np.sum(positive_m)),
        "exposure_positive_increment_rows": int(np.sum(positive_psi)),
        "max_abs_next_M_reconstruction_error": max_abs_m,
        "max_abs_next_Psi_reconstruction_error": max_abs_psi,
        "max_abs_delta_M_reconstruction_error": max_abs_dx,
        "max_abs_delta_Psi_reconstruction_error": max_abs_dy,
        "max_response_Z_bound_excess_before_clipping": max_z_m_excess,
        "max_exposure_Z_bound_excess_before_clipping": max_z_psi_excess,
        "innovation_source": "same-row pre-to-response and pre-to-post phase accounting",
    }

    # Secondary audit: post-state masses decay to the next pre-state masses.
    if require_next_audit:
        next_e = numeric_array(df, "next_response_evidence_mass")
        next_active = numeric_array(df, "next_activity_active_mass")
        next_idle = numeric_array(df, "next_activity_idle_mass")
        next_aligned = numeric_array(df, "next_activity_aligned_mass")
        next_off_values = numeric_array(df, next_off)
        next_b = next_active + next_idle
        next_g = next_aligned - next_off_values
        finite_gap = coordinate_audit_mask & np.isfinite(gap_days) & (gap_days >= 0)
        rho_m = np.exp(-np.maximum(gap_days, 0.0) / max(float(tau_response_days), EPS))
        rho_psi = np.exp(-np.maximum(gap_days, 0.0) / max(float(tau_activity_days), EPS))
        predicted_next_e = rho_m * response_denominator
        predicted_next_b = rho_psi * exposure_denominator
        predicted_next_g = rho_psi * (g_pre + j_psi)
        mass_mask_m = finite_gap & np.isfinite(next_e)
        mass_mask_psi = finite_gap & np.isfinite(next_b) & np.isfinite(next_g)
        mass_abs_e = _max_abs_difference(predicted_next_e, next_e, mass_mask_m)
        mass_rel_e = _max_relative_difference(predicted_next_e, next_e, mass_mask_m)
        mass_abs_b = _max_abs_difference(predicted_next_b, next_b, mass_mask_psi)
        mass_rel_b = _max_relative_difference(predicted_next_b, next_b, mass_mask_psi)
        mass_abs_g = _max_abs_difference(predicted_next_g, next_g, mass_mask_psi)
        mass_rel_g = _max_relative_difference(predicted_next_g, next_g, mass_mask_psi)
        audit.update(
            {
                "max_abs_next_response_mass_decay_error": mass_abs_e,
                "max_rel_next_response_mass_decay_error": mass_rel_e,
                "max_abs_next_exposure_denominator_decay_error": mass_abs_b,
                "max_rel_next_exposure_denominator_decay_error": mass_rel_b,
                "max_abs_next_exposure_numerator_decay_error": mass_abs_g,
                "max_rel_next_exposure_numerator_decay_error": mass_rel_g,
            }
        )
        # Relative and absolute gates are paired to avoid false failures near zero.
        if (
            (mass_abs_e > RECONSTRUCTION_ATOL and mass_rel_e > RECONSTRUCTION_RTOL)
            or (mass_abs_b > RECONSTRUCTION_ATOL and mass_rel_b > RECONSTRUCTION_RTOL)
            or (mass_abs_g > RECONSTRUCTION_ATOL and mass_rel_g > RECONSTRUCTION_RTOL)
        ):
            raise RuntimeError(
                "Phase masses do not decay to the frozen next pre-state within tolerance. "
                f"E(abs={mass_abs_e:.3e}, rel={mass_rel_e:.3e}), "
                f"B(abs={mass_abs_b:.3e}, rel={mass_rel_b:.3e}), "
                f"G(abs={mass_abs_g:.3e}, rel={mass_rel_g:.3e})."
            )

    return InnovationArrays(
        user_id=user_id,
        step=step,
        part=part,
        gap_days=gap_days,
        m_pre=m_pre,
        psi_pre=psi_pre,
        e_pre=e_pre,
        s_pre=s_pre,
        b_pre=b_pre,
        g_pre=g_pre,
        a_m=a_m,
        j_m=j_m,
        z_m=z_m,
        a_psi=a_psi,
        j_psi=j_psi,
        z_psi=z_psi,
        response_active=response_active,
        support_active=support_active,
        idle_mass=idle_mass,
        next_m=next_m,
        next_psi=next_psi,
        observed_dx=observed_dx,
        observed_dy=observed_dy,
        formal_drift_valid=formal_drift_valid,
        reconstruction_audit=audit,
    )


# -----------------------------------------------------------------------------
# A_train-only matching cutpoints
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MatchingCutpoints:
    log_a_m: np.ndarray
    log_a_psi: np.ndarray
    support_share: np.ndarray
    idle_share: np.ndarray
    sequence_length: np.ndarray
    fit_rows_sampled: int
    fit_users: int
    fit_split: str = "A_train"


def unique_quantiles(values: np.ndarray, probabilities: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.asarray([], dtype=np.float64)
    cuts = np.quantile(array, np.asarray(probabilities, dtype=np.float64))
    return np.unique(cuts[np.isfinite(cuts)]).astype(np.float64)


def fit_matching_cutpoints(
    train_base: Path,
    train_columns: Sequence[str],
    tau_response_days: float,
    tau_activity_days: float,
    max_sample_rows: int,
    chunk_rows: int,
    seed: int,
) -> Tuple[MatchingCutpoints, Dict[str, Any]]:
    sample_priority = np.asarray([], dtype=np.uint64)
    sample_values = np.empty((0, 4), dtype=np.float64)
    user_counts: Dict[int, int] = {}
    scanned_rows = 0
    eligible_rows = 0

    for chunk in iter_table_chunks(train_base, train_columns, chunk_rows):
        scanned_rows += len(chunk)
        reconstructed = reconstruct_innovations(
            chunk,
            tau_response_days=tau_response_days,
            tau_activity_days=tau_activity_days,
            require_next_audit=False,
        )
        valid = reconstructed.formal_drift_valid
        if not np.any(valid):
            continue
        eligible_rows += int(np.sum(valid))
        uid = reconstructed.user_id[valid]
        step = reconstructed.step[valid]
        counts = pd.Series(uid).value_counts(sort=False)
        for user, count in counts.items():
            key = int(user)
            user_counts[key] = user_counts.get(key, 0) + int(count)

        a_m = np.maximum(reconstructed.a_m[valid], 0.0)
        a_psi = np.maximum(reconstructed.a_psi[valid], 0.0)
        support_share = np.divide(
            reconstructed.support_active[valid],
            a_psi,
            out=np.zeros_like(a_psi),
            where=a_psi > EPS,
        )
        idle_share = np.divide(
            reconstructed.idle_mass[valid],
            a_psi,
            out=np.zeros_like(a_psi),
            where=a_psi > EPS,
        )
        values = np.column_stack(
            [
                np.log1p(a_m),
                np.log1p(a_psi),
                np.clip(support_share, 0.0, 1.0),
                np.clip(idle_share, 0.0, 1.0),
            ]
        )
        priority = stable_uint64_priority(uid, step, seed + 7919)
        if len(priority) > max_sample_rows:
            keep = np.argpartition(priority, max_sample_rows - 1)[:max_sample_rows]
            priority = priority[keep]
            values = values[keep]
        if sample_priority.size == 0:
            sample_priority = priority
            sample_values = values
        else:
            sample_priority = np.concatenate([sample_priority, priority])
            sample_values = np.concatenate([sample_values, values], axis=0)
        if len(sample_priority) > max_sample_rows:
            keep = np.argpartition(sample_priority, max_sample_rows - 1)[:max_sample_rows]
            sample_priority = sample_priority[keep]
            sample_values = sample_values[keep]

    if sample_values.shape[0] < 1000:
        raise RuntimeError(
            f"Only {sample_values.shape[0]} A_train rows were available for matching cutpoints."
        )
    sequence_lengths = np.asarray(list(user_counts.values()), dtype=np.float64)
    cutpoints = MatchingCutpoints(
        log_a_m=unique_quantiles(sample_values[:, 0], (0.25, 0.50, 0.75)),
        log_a_psi=unique_quantiles(sample_values[:, 1], (0.25, 0.50, 0.75)),
        support_share=unique_quantiles(sample_values[:, 2], (1.0 / 3.0, 2.0 / 3.0)),
        idle_share=unique_quantiles(sample_values[:, 3], (1.0 / 3.0, 2.0 / 3.0)),
        sequence_length=unique_quantiles(sequence_lengths, tuple(np.arange(0.1, 1.0, 0.1))),
        fit_rows_sampled=int(sample_values.shape[0]),
        fit_users=int(len(user_counts)),
    )
    audit = {
        "fit_split": "A_train",
        "rows_scanned": int(scanned_rows),
        "eligible_transition_rows": int(eligible_rows),
        "priority_sample_rows": int(sample_values.shape[0]),
        "users_counted": int(len(user_counts)),
        "sample_policy": "deterministic smallest splitmix64 priorities over user_id and bundle_step_index",
        "max_sample_rows": int(max_sample_rows),
        "cutpoints": json_safe(cutpoints),
    }
    return cutpoints, audit


def bin_by_cutpoints(values: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    output = np.searchsorted(np.asarray(cutpoints, dtype=np.float64), array, side="right").astype(np.int16)
    output[~np.isfinite(array)] = -1
    return output


def gap_bins(gap_days: np.ndarray) -> np.ndarray:
    values = np.asarray(gap_days, dtype=np.float64)
    output = np.full(len(values), 3, dtype=np.int8)  # missing/invalid
    finite = np.isfinite(values) & (values >= 0)
    output[finite & (values <= 1.0)] = 0
    output[finite & (values > 1.0) & (values <= 7.0)] = 1
    output[finite & (values > 7.0)] = 2
    return output


# -----------------------------------------------------------------------------
# Disjoint hierarchical permutation layouts
# -----------------------------------------------------------------------------
@dataclass
class PermutationLayout:
    name: str
    original_indices: np.ndarray
    order: np.ndarray
    sorted_group: np.ndarray
    starts: np.ndarray
    counts: np.ndarray

    def donor_indices(self, rng: np.random.Generator) -> np.ndarray:
        group_count = len(self.counts)
        random_fraction = rng.random(group_count)
        shifts = 1 + np.floor(random_fraction * (self.counts - 1)).astype(np.int64)
        positions = np.arange(len(self.order), dtype=np.int64)
        starts_for_position = self.starts[self.sorted_group]
        counts_for_position = self.counts[self.sorted_group]
        shifted_position = starts_for_position + (
            (positions - starts_for_position + shifts[self.sorted_group]) % counts_for_position
        )
        donor_local_sorted = self.order[shifted_position]
        donor_local = np.empty(len(self.order), dtype=np.int64)
        donor_local[self.order] = donor_local_sorted
        return self.original_indices[donor_local]


def structured_group_codes(key_arrays: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not key_arrays:
        n = 0
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int64)
    n = len(key_arrays[0])
    if any(len(array) != n for array in key_arrays):
        raise ValueError("Grouping key lengths differ.")
    dtype = []
    for index, array in enumerate(key_arrays):
        arr = np.asarray(array)
        if arr.dtype.kind not in "iu":
            raise TypeError("Permutation grouping keys must be integer arrays.")
        dtype.append((f"k{index}", arr.dtype.str))
    structured = np.empty(n, dtype=dtype)
    for index, array in enumerate(key_arrays):
        structured[f"k{index}"] = np.asarray(array)
    _, inverse, counts = np.unique(structured, return_inverse=True, return_counts=True)
    return inverse.astype(np.int32, copy=False), counts.astype(np.int64, copy=False)


def make_layout(
    name: str,
    original_indices: np.ndarray,
    key_arrays: Sequence[np.ndarray],
    layout_seed: int,
) -> Tuple[Optional[PermutationLayout], np.ndarray]:
    if len(original_indices) == 0:
        return None, np.asarray([], dtype=np.int64)
    local_keys = [np.asarray(array)[original_indices] for array in key_arrays]
    codes, counts = structured_group_codes(local_keys)
    eligible_local = counts[codes] >= 2
    selected_original = original_indices[eligible_local]
    remaining_original = original_indices[~eligible_local]
    if len(selected_original) == 0:
        return None, remaining_original

    selected_keys = [np.asarray(array)[selected_original] for array in key_arrays]
    selected_codes, selected_counts = structured_group_codes(selected_keys)
    if np.any(selected_counts < 2):
        raise RuntimeError(f"Internal grouping error in layout {name}.")
    rng = np.random.default_rng(int(layout_seed))
    random_key = rng.random(len(selected_original))
    order = np.lexsort((random_key, selected_codes)).astype(np.int64, copy=False)
    sorted_group = selected_codes[order]
    starts = np.flatnonzero(
        np.concatenate([[True], sorted_group[1:] != sorted_group[:-1]])
    ).astype(np.int64)
    counts_check = np.diff(np.concatenate([starts, [len(order)]])).astype(np.int64)
    if not np.array_equal(counts_check, selected_counts):
        raise RuntimeError(f"Group-layout count mismatch in {name}.")
    return (
        PermutationLayout(
            name=name,
            original_indices=selected_original.astype(np.int64, copy=False),
            order=order,
            sorted_group=sorted_group.astype(np.int32, copy=False),
            starts=starts,
            counts=counts_check,
        ),
        remaining_original,
    )


def build_hierarchical_layouts(
    keys: Mapping[str, np.ndarray],
    randomizable: np.ndarray,
    seed: int,
    max_last_resort_fraction: float,
) -> Tuple[List[PermutationLayout], pd.DataFrame, np.ndarray]:
    n = len(randomizable)
    remaining = np.flatnonzero(np.asarray(randomizable, dtype=bool)).astype(np.int64)
    constant = np.zeros(n, dtype=np.int8)
    level_specs: List[Tuple[str, List[str]]] = [
        (
            "within_user_fine",
            ["user_id", "response_present", "support_present", "idle_present", "gap_bin"],
        ),
        (
            "within_user_coarse",
            ["user_id", "response_present", "exposure_present"],
        ),
        (
            "across_user_fine",
            [
                "part",
                "response_present",
                "support_present",
                "idle_present",
                "gap_bin",
                "a_m_bin",
                "a_psi_bin",
                "support_share_bin",
                "idle_share_bin",
                "sequence_length_bin",
            ],
        ),
        (
            "across_user_coarse",
            ["part", "response_present", "support_present", "idle_present", "gap_bin"],
        ),
        (
            "global_opportunity",
            ["response_present", "support_present", "idle_present", "exposure_present"],
        ),
        ("global_last_resort", ["constant"]),
    ]
    all_keys = dict(keys)
    all_keys["constant"] = constant
    layouts: List[PermutationLayout] = []
    coverage_rows: List[dict] = []
    randomizable_count = int(np.sum(randomizable))

    for level_index, (name, key_names) in enumerate(level_specs):
        if len(remaining) == 0:
            break
        layout, new_remaining = make_layout(
            name,
            remaining,
            [all_keys[key] for key in key_names],
            layout_seed=int(seed) + 1009 * (level_index + 1),
        )
        assigned_count = 0 if layout is None else len(layout.original_indices)
        if layout is not None:
            layouts.append(layout)
        coverage_rows.append(
            {
                "level": name,
                "matching_keys": ";".join(key_names),
                "rows_assigned": int(assigned_count),
                "fraction_of_randomizable_rows": (
                    float(assigned_count / randomizable_count) if randomizable_count > 0 else 0.0
                ),
                "rows_remaining_after_level": int(len(new_remaining)),
            }
        )
        remaining = new_remaining

    effective_randomizable = np.asarray(randomizable, dtype=bool).copy()
    unmatched_count = int(len(remaining))
    unmatched_fraction = unmatched_count / max(randomizable_count, 1)
    if unmatched_count > 0:
        effective_randomizable[remaining] = False
        coverage_rows.append(
            {
                "level": "unmatched_singleton_self_exempt",
                "matching_keys": "self-mapped only because no non-singleton donor group exists",
                "rows_assigned": unmatched_count,
                "fraction_of_randomizable_rows": float(unmatched_fraction),
                "rows_remaining_after_level": 0,
            }
        )
    last_resort_rows = sum(
        len(layout.original_indices) for layout in layouts if layout.name == "global_last_resort"
    )
    weak_fallback_fraction = (last_resort_rows + unmatched_count) / max(randomizable_count, 1)
    if weak_fallback_fraction > float(max_last_resort_fraction):
        raise RuntimeError(
            "The weakly matched fallback share is too large for a credible matched null: "
            f"{weak_fallback_fraction:.4%} > {float(max_last_resort_fraction):.4%}."
        )
    coverage_rows.append(
        {
            "level": "deterministic_zero_innovation",
            "matching_keys": "not randomized because both denominator increments are zero",
            "rows_assigned": int(n - randomizable_count),
            "fraction_of_randomizable_rows": np.nan,
            "rows_remaining_after_level": 0,
        }
    )
    return layouts, pd.DataFrame(coverage_rows), effective_randomizable


def generate_joint_donor_mapping(
    n_rows: int,
    layouts: Sequence[PermutationLayout],
    randomizable: np.ndarray,
    seed: int,
) -> np.ndarray:
    donor = np.arange(n_rows, dtype=np.int64)
    assigned = np.zeros(n_rows, dtype=bool)
    rng = np.random.default_rng(int(seed))
    for layout in layouts:
        mapped = layout.donor_indices(rng)
        recipient = layout.original_indices
        donor[recipient] = mapped
        assigned[recipient] = True
    if not np.array_equal(assigned, np.asarray(randomizable, dtype=bool)):
        mismatch = int(np.sum(assigned != np.asarray(randomizable, dtype=bool)))
        raise RuntimeError(f"Permutation assignment mismatch on {mismatch} rows.")
    return donor


# -----------------------------------------------------------------------------
# Frozen field estimator and metrics
# -----------------------------------------------------------------------------
@dataclass
class PreparedAnalysis:
    drift_row_indices: np.ndarray
    x: np.ndarray
    y: np.ndarray
    user_id: np.ndarray
    part: np.ndarray
    gap_days: np.ndarray
    weight: np.ndarray
    cell: np.ndarray
    observed_dx: np.ndarray
    observed_dy: np.ndarray
    e_pre: np.ndarray
    s_pre: np.ndarray
    b_pre: np.ndarray
    g_pre: np.ndarray
    a_m: np.ndarray
    z_m: np.ndarray
    a_psi: np.ndarray
    z_psi: np.ndarray
    response_active: np.ndarray
    support_active: np.ndarray
    idle_mass: np.ndarray
    m_denominator: np.ndarray
    psi_denominator: np.ndarray
    formal_field: Any
    specification: Any


def copy_field_with_drift(field: Any, drift_u: np.ndarray, drift_v: np.ndarray) -> Any:
    return dataclasses.replace(
        field,
        drift_u=np.asarray(drift_u, dtype=np.float64),
        drift_v=np.asarray(drift_v, dtype=np.float64),
    )


def aggregate_mean_field(
    prepared: PreparedAnalysis,
    dx: np.ndarray,
    dy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    shape = prepared.formal_field.drift_u.shape
    n_cells = int(np.prod(shape))
    sx = np.bincount(
        prepared.cell,
        weights=prepared.weight * np.asarray(dx, dtype=np.float64),
        minlength=n_cells,
    ).reshape(shape)
    sy = np.bincount(
        prepared.cell,
        weights=prepared.weight * np.asarray(dy, dtype=np.float64),
        minlength=n_cells,
    ).reshape(shape)
    denominator = np.maximum(prepared.formal_field.drift_weight, EPS)
    return sx / denominator, sy / denominator


def field_geometry_metrics(stage1: Any, field: Any, core_mask: np.ndarray, split_label: str, shell_radius: float) -> Dict[str, float]:
    contraction = stage1.global_field_contraction_summary(field, split_label)
    region_table, _ = stage1.evaluate_frozen_convergence_region(
        field,
        core_mask,
        split_label,
        float(shell_radius),
    )
    if region_table.empty:
        raise RuntimeError(f"No frozen-core metrics were produced for {split_label}.")
    region = region_table.iloc[0]
    return {
        "negative_divergence_occupancy_fraction": float(
            contraction["weighted_negative_divergence_fraction_interior_only"]
        ),
        "weighted_mean_divergence": float(
            contraction["weighted_mean_local_divergence_interior_only"]
        ),
        "flow_weighted_shell_fraction_inward": float(
            region["flow_weighted_shell_fraction_inward"]
        ),
        "flow_weighted_shell_inward_cosine": float(
            region["flow_weighted_shell_inward_cosine"]
        ),
        "flow_core_to_shell_speed_ratio": float(
            region["flow_core_to_shell_speed_ratio"]
        ),
        "occupancy_core_to_shell_speed_ratio": float(region["core_to_shell_speed_ratio"]),
    }


def weighted_field_distance(
    u_first: np.ndarray,
    v_first: np.ndarray,
    u_second: np.ndarray,
    v_second: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> float:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(u_first)
        & np.isfinite(v_first)
        & np.isfinite(u_second)
        & np.isfinite(v_second)
    )
    if not np.any(valid):
        return float("nan")
    w = np.asarray(weight, dtype=np.float64)[valid]
    w = w / max(float(np.sum(w)), EPS)
    squared = (u_first[valid] - u_second[valid]) ** 2 + (v_first[valid] - v_second[valid]) ** 2
    return float(np.sqrt(np.sum(w * squared)))


def bh_qvalues(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    q = np.full_like(p, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(p))
    if finite_indices.size == 0:
        return q
    order_local = np.argsort(p[finite_indices])
    ordered_indices = finite_indices[order_local]
    m = len(ordered_indices)
    adjusted = p[ordered_indices] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[ordered_indices] = np.clip(adjusted, 0.0, 1.0)
    return q


def monte_carlo_p(observed: float, null_values: np.ndarray, direction: str) -> float:
    values = np.asarray(null_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or not np.isfinite(observed):
        return float("nan")
    if direction == "greater":
        exceed = int(np.sum(values >= observed))
    elif direction == "less":
        exceed = int(np.sum(values <= observed))
    else:
        raise ValueError(direction)
    return float((1 + exceed) / (values.size + 1))


def prepare_analysis(
    frame: pd.DataFrame,
    innovations: InnovationArrays,
    stage1: Any,
    specification: Any,
) -> PreparedAnalysis:
    field_columns = [
        "user_id",
        specification.xcol,
        specification.ycol,
        specification.dxcol,
        specification.dycol,
    ]
    field_frame = stage1.downcast_frame(frame[field_columns].copy())

    formal_stats_dict = stage1.occupancy_drift_stats(
        field_frame,
        specification,
    )
    formal_field = stage1.field_stats_from_dict(formal_stats_dict)
    weights_full = stage1.user_balanced_weights(field_frame)

    x_full = numeric_array(field_frame, specification.xcol)
    y_full = numeric_array(field_frame, specification.ycol)
    observed_dx_full = numeric_array(field_frame, specification.dxcol)
    observed_dy_full = numeric_array(field_frame, specification.dycol)

    ix = stage1.digitize_closed_right(
        x_full,
        specification.xbins,
    )
    iy = stage1.digitize_closed_right(
        y_full,
        specification.ybins,
    )
    nx = len(specification.xbins) - 1
    ny = len(specification.ybins) - 1
    drift_valid = innovations.formal_drift_valid & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    row_indices = np.flatnonzero(drift_valid).astype(np.int64)
    cell = (ix[row_indices] * ny + iy[row_indices]).astype(np.int64)
    prepared = PreparedAnalysis(
        drift_row_indices=row_indices,
        x=x_full[row_indices],
        y=y_full[row_indices],
        user_id=innovations.user_id[row_indices],
        part=innovations.part[row_indices].astype(np.int16, copy=False),
        gap_days=innovations.gap_days[row_indices],
        weight=np.asarray(weights_full, dtype=np.float64)[row_indices],
        cell=cell,
        observed_dx=observed_dx_full[row_indices],
        observed_dy=observed_dy_full[row_indices],
        e_pre=innovations.e_pre[row_indices],
        s_pre=innovations.s_pre[row_indices],
        b_pre=innovations.b_pre[row_indices],
        g_pre=innovations.g_pre[row_indices],
        a_m=innovations.a_m[row_indices],
        z_m=innovations.z_m[row_indices],
        a_psi=innovations.a_psi[row_indices],
        z_psi=innovations.z_psi[row_indices],
        response_active=innovations.response_active[row_indices],
        support_active=innovations.support_active[row_indices],
        idle_mass=innovations.idle_mass[row_indices],
        m_denominator=(innovations.e_pre + innovations.a_m)[row_indices],
        psi_denominator=(innovations.b_pre + innovations.a_psi)[row_indices],
        formal_field=formal_field,
        specification=specification,
    )
    custom_u, custom_v = aggregate_mean_field(prepared, prepared.observed_dx, prepared.observed_dy)
    mask = formal_field.drift_mask
    max_u = float(np.max(np.abs(custom_u[mask] - formal_field.drift_u[mask]))) if np.any(mask) else 0.0
    max_v = float(np.max(np.abs(custom_v[mask] - formal_field.drift_v[mask]))) if np.any(mask) else 0.0
    if max(max_u, max_v) > 1e-12:
        raise RuntimeError(
            "The optimized null field estimator does not reproduce the formal Stage-1 field: "
            f"max_dM={max_u:.3e}, max_dPsi={max_v:.3e}."
        )
    return prepared


def audit_saved_field(
    stage1_root: Path,
    split: str,
    field: Any,
    skip: bool,
) -> Dict[str, Any]:
    if skip:
        return {"skipped": True, "reason": "smoke-test user subsample"}
    suffix = "_output_only" if split == "B_confirm" else ""
    base = (
        stage1_root
        / "dynamics"
        / "coordinate_analysis"
        / PRIMARY_COORDINATE
        / f"{split}_publication_field_grid{suffix}"
    )
    table = read_table(base).sort_values(["x_bin", "y_bin"], kind="mergesort")
    shape = field.drift_u.shape
    expected_rows = int(np.prod(shape))
    if len(table) != expected_rows:
        raise RuntimeError(f"Saved field grid has {len(table)} rows; expected {expected_rows}.")
    saved_u = pd.to_numeric(table["drift_M"], errors="coerce").to_numpy(dtype=float).reshape(shape)
    saved_v = pd.to_numeric(table["drift_Psi"], errors="coerce").to_numpy(dtype=float).reshape(shape)
    saved_p = pd.to_numeric(table["occupancy_probability"], errors="coerce").to_numpy(dtype=float).reshape(shape)
    saved_mask = table["drift_supported"].astype(bool).to_numpy().reshape(shape)
    if not np.array_equal(saved_mask, field.drift_mask):
        raise RuntimeError("Recomputed and saved Stage-1 drift-support masks differ.")
    max_u = float(np.max(np.abs(saved_u[field.drift_mask] - field.drift_u[field.drift_mask])))
    max_v = float(np.max(np.abs(saved_v[field.drift_mask] - field.drift_v[field.drift_mask])))
    max_p = float(np.max(np.abs(saved_p - field.occupancy_probability)))
    if max(max_u, max_v, max_p) > 1e-10:
        raise RuntimeError(
            "Recomputed Stage-1 field does not match the archived publication grid: "
            f"dM={max_u:.3e}, dPsi={max_v:.3e}, occupancy={max_p:.3e}."
        )
    return {
        "skipped": False,
        "saved_field_path": str(find_table(base).resolve()),
        "max_abs_drift_M_difference": max_u,
        "max_abs_drift_Psi_difference": max_v,
        "max_abs_occupancy_probability_difference": max_p,
        "drift_mask_exact_match": True,
    }


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------
def required_columns_for_split(base: Path) -> Tuple[List[str], Dict[str, str]]:
    available = available_columns(base)
    off_pre = resolve_off_target_column(available, "pre")
    off_post = resolve_off_target_column(available, "post")
    next_off = (
        "next_activity_off_target_mass"
        if "next_activity_off_target_mass" in available
        else "next_activity_non_aligned_mass"
    )
    required = [
        "user_id",
        "bundle_step_index",
        "part",
        "next_gap_days",
        "M_response_prebalanced_pre",
        "response_evidence_mass_pre",
        "M_response_prebalanced_resp",
        "response_evidence_mass_resp",
        "activity_alignment_order_Psi_pre",
        "activity_active_mass_pre",
        "activity_aligned_mass_pre",
        off_pre,
        "activity_idle_mass_pre",
        "activity_alignment_order_Psi_post",
        "activity_active_mass_post",
        "activity_aligned_mass_post",
        off_post,
        "activity_idle_mass_post",
        "next_M_response_prebalanced",
        "next_response_evidence_mass",
        "next_activity_alignment_order_Psi",
        "next_activity_active_mass",
        "next_activity_aligned_mass",
        next_off,
        "next_activity_idle_mass",
        "delta_M_response_prebalanced_next",
        "delta_activity_alignment_order_Psi_next",
        "response_active_mass_interval",
        "support_active_total_interval",
        "idle_mass_interval",
    ]
    missing = sorted(set(required).difference(available))
    if missing:
        raise RuntimeError(f"Core panel is missing required construction-null columns: {missing}")
    return sorted(set(required)), {"off_pre": off_pre, "off_post": off_post, "next_off": next_off}


def build_matching_keys(
    prepared: PreparedAnalysis,
    cutpoints: MatchingCutpoints,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    uid = prepared.user_id
    part = prepared.part
    gap = prepared.gap_days
    a_m = np.maximum(prepared.a_m, 0.0)
    a_psi = np.maximum(prepared.a_psi, 0.0)
    response_active = np.maximum(prepared.response_active, 0.0)
    support_active = np.maximum(prepared.support_active, 0.0)
    idle_mass = np.maximum(prepared.idle_mass, 0.0)

    response_present = (a_m > EPS).astype(np.int8)
    exposure_present = (a_psi > EPS).astype(np.int8)
    support_present = (support_active > EPS).astype(np.int8)
    idle_present = (idle_mass > EPS).astype(np.int8)
    support_share = np.divide(support_active, a_psi, out=np.zeros_like(a_psi), where=a_psi > EPS)
    idle_share = np.divide(idle_mass, a_psi, out=np.zeros_like(a_psi), where=a_psi > EPS)
    support_share = np.clip(support_share, 0.0, 1.0)
    idle_share = np.clip(idle_share, 0.0, 1.0)

    user_lengths = pd.Series(uid).groupby(pd.Series(uid), sort=False).transform("count").to_numpy(dtype=float)
    keys: Dict[str, np.ndarray] = {
        "user_id": uid.astype(np.int64, copy=False),
        "part": part,
        "response_present": response_present,
        "exposure_present": exposure_present,
        "support_present": support_present,
        "idle_present": idle_present,
        "gap_bin": gap_bins(gap),
        "a_m_bin": bin_by_cutpoints(np.log1p(a_m), cutpoints.log_a_m),
        "a_psi_bin": bin_by_cutpoints(np.log1p(a_psi), cutpoints.log_a_psi),
        "support_share_bin": bin_by_cutpoints(support_share, cutpoints.support_share),
        "idle_share_bin": bin_by_cutpoints(idle_share, cutpoints.idle_share),
        "sequence_length_bin": bin_by_cutpoints(user_lengths, cutpoints.sequence_length),
    }
    randomizable = (a_m > EPS) | (a_psi > EPS)
    composition_audit = pd.DataFrame(
        [
            {
                "analysis_rows": int(len(prepared.x)),
                "randomizable_rows": int(np.sum(randomizable)),
                "zero_innovation_rows": int(np.sum(~randomizable)),
                "response_increment_present_fraction": float(np.mean(response_present)),
                "exposure_increment_present_fraction": float(np.mean(exposure_present)),
                "support_present_fraction": float(np.mean(support_present)),
                "idle_present_fraction": float(np.mean(idle_present)),
                "mean_response_active_mass": float(np.mean(response_active)),
                "mean_support_active_mass": float(np.mean(support_active)),
                "mean_idle_mass": float(np.mean(idle_mass)),
            }
        ]
    )
    return keys, randomizable, composition_audit


def field_comparison_table(
    stage1: Any,
    observed_field: Any,
    null_mean_u: np.ndarray,
    null_mean_v: np.ndarray,
    null_sd_u: np.ndarray,
    null_sd_v: np.ndarray,
    ratio_u: np.ndarray,
    ratio_v: np.ndarray,
    split: str,
) -> pd.DataFrame:
    table = stage1.field_grid_table(observed_field, split).copy()
    table["null_mean_drift_M"] = null_mean_u.ravel(order="C")
    table["null_mean_drift_Psi"] = null_mean_v.ravel(order="C")
    table["null_sd_drift_M"] = null_sd_u.ravel(order="C")
    table["null_sd_drift_Psi"] = null_sd_v.ravel(order="C")
    table["excess_drift_M"] = (observed_field.drift_u - null_mean_u).ravel(order="C")
    table["excess_drift_Psi"] = (observed_field.drift_v - null_mean_v).ravel(order="C")
    table["pure_ratio_drift_M"] = ratio_u.ravel(order="C")
    table["pure_ratio_drift_Psi"] = ratio_v.ravel(order="C")
    table["null_role"] = "matched joint signed-innovation permutation; current anchors and denominator increments fixed"
    return table


def run_analysis(args: argparse.Namespace) -> None:
    started = time.time()
    stage1_root = args.stage1_root.resolve()
    dynamics_root = stage1_root / "dynamics"
    if not dynamics_root.is_dir():
        raise FileNotFoundError(f"Stage-1 dynamics directory not found: {dynamics_root}")
    if args.analysis_split == "B_confirm" and not args.confirmation_output_only:
        raise RuntimeError(
            "B_confirm requires --confirmation-output-only after the A_val protocol and interpretation are frozen."
        )

    stage1_script = resolve_stage1_script(args.stage1_script)
    stage1 = import_stage1_module(stage1_script)
    specification = stage1.coordinate_specs()[0]
    if specification.name != PRIMARY_COORDINATE:
        raise RuntimeError(f"Expected primary coordinate {PRIMARY_COORDINATE}, got {specification.name}.")

    output_root = args.output_root.resolve()
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    array_root = output_root / "arrays"
    for directory in (table_root, metadata_root, array_root):
        directory.mkdir(parents=True, exist_ok=True)

    train_base = dynamics_root / "student_dynamics_panel_core_A_train"
    analysis_base = dynamics_root / f"student_dynamics_panel_core_{args.analysis_split}"
    train_columns, _ = required_columns_for_split(train_base)
    analysis_columns, resolved_aliases = required_columns_for_split(analysis_base)

    cutpoints, cutpoint_audit = fit_matching_cutpoints(
        train_base=train_base,
        train_columns=train_columns,
        tau_response_days=float(stage1.TAU_RESPONSE_DAYS),
        tau_activity_days=float(stage1.TAU_ACTIVITY_DAYS),
        max_sample_rows=int(args.matching_fit_max_rows),
        chunk_rows=int(args.read_chunk_rows),
        seed=int(args.seed),
    )
    save_json({"cutpoints": cutpoints, "audit": cutpoint_audit}, metadata_root / "matching_cutpoints_A_train.json")

    frame = read_table(analysis_base, columns=analysis_columns)
    if args.max_users > 0:
        users = np.asarray(sorted(pd.to_numeric(frame["user_id"], errors="coerce").dropna().astype(np.int64).unique()))
        if len(users) > args.max_users:
            rng = np.random.default_rng(int(args.seed))
            keep = set(rng.choice(users, size=int(args.max_users), replace=False).tolist())
            frame = frame[pd.to_numeric(frame["user_id"], errors="coerce").isin(keep)].copy()
    frame = frame.sort_values(["user_id", "bundle_step_index"], kind="mergesort").reset_index(drop=True)

    innovations = reconstruct_innovations(
        frame,
        tau_response_days=float(stage1.TAU_RESPONSE_DAYS),
        tau_activity_days=float(stage1.TAU_ACTIVITY_DAYS),
        require_next_audit=True,
    )
    panel_row_count = int(len(frame))
    panel_user_count = int(pd.Series(innovations.user_id).nunique())
    reconstruction_audit = dict(innovations.reconstruction_audit)
    prepared = prepare_analysis(frame, innovations, stage1, specification)

    saved_field_audit = audit_saved_field(
        stage1_root,
        args.analysis_split,
        prepared.formal_field,
        skip=bool(args.max_users > 0),
    )
    core_mask_path = (
        stage1_root
        / "dynamics"
        / "candidate_regions"
        / PRIMARY_COORDINATE
        / "A_train_primary_convergence_core_mask.npy"
    )
    thresholds_path = (
        stage1_root
        / "dynamics"
        / "candidate_regions"
        / PRIMARY_COORDINATE
        / "training_convergence_thresholds.json"
    )
    if not core_mask_path.exists() or not thresholds_path.exists():
        raise FileNotFoundError("Frozen A_train convergence core or thresholds file is missing.")
    core_mask = np.load(core_mask_path).astype(bool)
    with thresholds_path.open("r", encoding="utf-8") as handle:
        thresholds = json.load(handle)
    shell_radius = float(thresholds["shell_radius"])
    if core_mask.shape != prepared.formal_field.drift_u.shape:
        raise RuntimeError(
            f"Frozen core shape {core_mask.shape} does not match field shape {prepared.formal_field.drift_u.shape}."
        )

    del frame
    del innovations
    gc.collect()

    keys, randomizable, composition_audit = build_matching_keys(
        prepared,
        cutpoints,
    )
    layouts, coverage_table, randomizable = build_hierarchical_layouts(
        keys,
        randomizable,
        seed=int(args.seed) + 200003,
        max_last_resort_fraction=float(args.max_last_resort_fraction),
    )
    del keys
    gc.collect()

    observed_metrics = field_geometry_metrics(
        stage1,
        prepared.formal_field,
        core_mask,
        f"{args.analysis_split}_observed",
        shell_radius,
    )

    e_after = prepared.m_denominator
    b_after = prepared.psi_denominator
    ratio_next_m = prepared.s_pre / e_after
    ratio_next_psi = prepared.g_pre / b_after
    ratio_dx = ratio_next_m - prepared.x
    ratio_dy = ratio_next_psi - prepared.y
    ratio_u, ratio_v = aggregate_mean_field(prepared, ratio_dx, ratio_dy)
    ratio_field = copy_field_with_drift(prepared.formal_field, ratio_u, ratio_v)
    ratio_metrics = field_geometry_metrics(
        stage1,
        ratio_field,
        core_mask,
        f"{args.analysis_split}_pure_ratio_contraction",
        shell_radius,
    )

    replicate_rows: List[Dict[str, float]] = []
    null_u = np.empty((int(args.replicates),) + prepared.formal_field.drift_u.shape, dtype=np.float64)
    null_v = np.empty_like(null_u)
    z_m = prepared.z_m
    z_psi = prepared.z_psi
    a_m = prepared.a_m
    a_psi = prepared.a_psi
    s_pre = prepared.s_pre
    g_pre = prepared.g_pre

    first_permutation_audit: Dict[str, Any] = {}
    for replicate in range(int(args.replicates)):
        donor = generate_joint_donor_mapping(
            len(prepared.x),
            layouts,
            randomizable,
            seed=int(args.seed) + 1_000_003 * (replicate + 1),
        )
        donor_z_m = z_m[donor]
        donor_z_psi = z_psi[donor]
        null_next_m = (s_pre + a_m * donor_z_m) / e_after
        null_next_psi = (g_pre + a_psi * donor_z_psi) / b_after
        max_bound = max(
            float(np.max(np.maximum(np.abs(null_next_m) - 1.0, 0.0))),
            float(np.max(np.maximum(np.abs(null_next_psi) - 1.0, 0.0))),
        )
        if max_bound > Z_BOUND_TOL:
            raise RuntimeError(f"Null next state left [-1,1] by {max_bound:.3e} in replicate {replicate}.")
        null_next_m = np.clip(null_next_m, -1.0, 1.0)
        null_next_psi = np.clip(null_next_psi, -1.0, 1.0)
        dx = null_next_m - prepared.x
        dy = null_next_psi - prepared.y
        u, v = aggregate_mean_field(prepared, dx, dy)
        null_u[replicate] = u
        null_v[replicate] = v
        field = copy_field_with_drift(prepared.formal_field, u, v)
        metrics = field_geometry_metrics(
            stage1,
            field,
            core_mask,
            f"{args.analysis_split}_null_{replicate:03d}",
            shell_radius,
        )
        replicate_rows.append({"replicate": int(replicate), "seed": int(args.seed) + 1_000_003 * (replicate + 1), **metrics})

        if replicate == 0:
            randomized_rows = np.asarray(randomizable, dtype=bool)
            first_permutation_audit = {
                "randomized_rows": int(np.sum(randomized_rows)),
                "fixed_points_among_randomized_rows": int(np.sum(donor[randomized_rows] == np.flatnonzero(randomized_rows))),
                "mean_Z_M_before": float(np.mean(z_m)),
                "mean_Z_M_after": float(np.mean(donor_z_m)),
                "mean_Z_Psi_before": float(np.mean(z_psi)),
                "mean_Z_Psi_after": float(np.mean(donor_z_psi)),
                "mean_product_Z_before": float(np.mean(z_m * z_psi)),
                "mean_product_Z_after": float(np.mean(donor_z_m * donor_z_psi)),
                "joint_pair_moved_together": True,
                "overall_mapping_bijective_by_disjoint_group_permutations": True,
            }
            if first_permutation_audit["fixed_points_among_randomized_rows"] != 0:
                raise RuntimeError("The first null replicate contains fixed points among randomized rows.")
            for key in (
                "mean_Z_M_before",
                "mean_Z_M_after",
                "mean_Z_Psi_before",
                "mean_Z_Psi_after",
                "mean_product_Z_before",
                "mean_product_Z_after",
            ):
                if not np.isfinite(first_permutation_audit[key]):
                    raise RuntimeError("Permutation-marginal audit produced a non-finite value.")
            if abs(first_permutation_audit["mean_Z_M_before"] - first_permutation_audit["mean_Z_M_after"]) > 1e-12:
                raise RuntimeError("The joint permutation changed the global Z_M marginal mean.")
            if abs(first_permutation_audit["mean_Z_Psi_before"] - first_permutation_audit["mean_Z_Psi_after"]) > 1e-12:
                raise RuntimeError("The joint permutation changed the global Z_Psi marginal mean.")
            if abs(first_permutation_audit["mean_product_Z_before"] - first_permutation_audit["mean_product_Z_after"]) > 1e-12:
                raise RuntimeError("The joint permutation did not preserve paired Z_M-Z_Psi coupling.")

        if (replicate + 1) % max(1, int(args.progress_every)) == 0 or replicate == 0:
            print(
                f"[construction null] {replicate + 1}/{args.replicates} replicates complete",
                flush=True,
            )

    null_mean_u = np.mean(null_u, axis=0)
    null_mean_v = np.mean(null_v, axis=0)
    null_sd_u = np.std(null_u, axis=0, ddof=1) if int(args.replicates) > 1 else np.zeros_like(null_mean_u)
    null_sd_v = np.std(null_v, axis=0, ddof=1) if int(args.replicates) > 1 else np.zeros_like(null_mean_v)
    excess_u = prepared.formal_field.drift_u - null_mean_u
    excess_v = prepared.formal_field.drift_v - null_mean_v
    excess_field = copy_field_with_drift(prepared.formal_field, excess_u, excess_v)
    excess_metrics = field_geometry_metrics(
        stage1,
        excess_field,
        core_mask,
        f"{args.analysis_split}_excess_field",
        shell_radius,
    )

    replicate_table = pd.DataFrame(replicate_rows)
    primary_specs = [
        ("negative_divergence_occupancy_fraction", "greater"),
        ("flow_weighted_shell_fraction_inward", "greater"),
        ("flow_core_to_shell_speed_ratio", "less"),
    ]
    summary_rows: List[dict] = []
    p_values: List[float] = []
    for metric, direction in primary_specs:
        values = pd.to_numeric(replicate_table[metric], errors="coerce").to_numpy(dtype=float)
        observed = float(observed_metrics[metric])
        p_value = monte_carlo_p(observed, values, direction)
        p_values.append(p_value)
        summary_rows.append(
            {
                "metric": metric,
                "direction_supporting_excess_structure": direction,
                "observed": observed,
                "pure_ratio_contraction": float(ratio_metrics[metric]),
                "null_mean": float(np.nanmean(values)),
                "null_sd": float(np.nanstd(values, ddof=1)),
                "null_2p5": float(np.nanquantile(values, 0.025)),
                "null_50": float(np.nanquantile(values, 0.50)),
                "null_97p5": float(np.nanquantile(values, 0.975)),
                "monte_carlo_p": p_value,
                "excess_field_value_descriptive": float(excess_metrics[metric]),
            }
        )
    q_values = bh_qvalues(p_values)
    for row, q_value in zip(summary_rows, q_values):
        row["BH_q_across_three_basin_metrics"] = float(q_value)

    field_mask = np.asarray(prepared.formal_field.drift_mask, dtype=bool)
    field_weight = np.asarray(prepared.formal_field.occupancy_probability, dtype=float)
    t_observed = weighted_field_distance(
        prepared.formal_field.drift_u,
        prepared.formal_field.drift_v,
        null_mean_u,
        null_mean_v,
        field_weight,
        field_mask,
    )
    null_sum_u = np.sum(null_u, axis=0)
    null_sum_v = np.sum(null_v, axis=0)
    t_null = np.empty(int(args.replicates), dtype=float)
    for replicate in range(int(args.replicates)):
        if int(args.replicates) > 1:
            loo_u = (null_sum_u - null_u[replicate]) / (int(args.replicates) - 1)
            loo_v = (null_sum_v - null_v[replicate]) / (int(args.replicates) - 1)
        else:
            loo_u = null_mean_u
            loo_v = null_mean_v
        t_null[replicate] = weighted_field_distance(
            null_u[replicate],
            null_v[replicate],
            loo_u,
            loo_v,
            field_weight,
            field_mask,
        )
    full_field_p = float((1 + np.sum(t_null >= t_observed)) / (len(t_null) + 1))
    full_field_row = {
        "metric": "occupancy_weighted_full_field_distance_from_null_mean",
        "direction_supporting_excess_structure": "greater",
        "observed": t_observed,
        "pure_ratio_contraction": weighted_field_distance(
            ratio_u,
            ratio_v,
            null_mean_u,
            null_mean_v,
            field_weight,
            field_mask,
        ),
        "null_mean": float(np.mean(t_null)),
        "null_sd": float(np.std(t_null, ddof=1)) if len(t_null) > 1 else 0.0,
        "null_2p5": float(np.quantile(t_null, 0.025)),
        "null_50": float(np.quantile(t_null, 0.50)),
        "null_97p5": float(np.quantile(t_null, 0.975)),
        "monte_carlo_p": full_field_p,
        "BH_q_across_three_basin_metrics": np.nan,
        "excess_field_value_descriptive": t_observed,
    }
    summary_table = pd.DataFrame([full_field_row, *summary_rows])

    comparison = field_comparison_table(
        stage1,
        prepared.formal_field,
        null_mean_u,
        null_mean_v,
        null_sd_u,
        null_sd_v,
        ratio_u,
        ratio_v,
        args.analysis_split,
    )
    write_table(comparison, table_root / f"{args.analysis_split}_construction_null_field_comparison")
    write_table(replicate_table, table_root / f"{args.analysis_split}_construction_null_replicate_metrics")
    write_table(summary_table, table_root / f"{args.analysis_split}_construction_null_summary")
    write_table(coverage_table, table_root / f"{args.analysis_split}_matching_fallback_coverage")
    write_table(composition_audit, table_root / f"{args.analysis_split}_opportunity_composition_audit")
    write_table(
        pd.DataFrame(
            [
                {"field": "observed", **observed_metrics},
                {"field": "pure_ratio_contraction", **ratio_metrics},
                {"field": "excess_observed_minus_null_mean", **excess_metrics},
            ]
        ),
        table_root / f"{args.analysis_split}_observed_ratio_excess_metrics",
    )
    np.savez_compressed(
        array_root / f"{args.analysis_split}_construction_null_fields.npz",
        observed_u=np.asarray(prepared.formal_field.drift_u, dtype=np.float64),
        observed_v=np.asarray(prepared.formal_field.drift_v, dtype=np.float64),
        null_u=null_u,
        null_v=null_v,
        null_mean_u=null_mean_u,
        null_mean_v=null_mean_v,
        null_sd_u=null_sd_u,
        null_sd_v=null_sd_v,
        excess_u=excess_u,
        excess_v=excess_v,
        pure_ratio_u=ratio_u,
        pure_ratio_v=ratio_v,
        drift_mask=field_mask,
        state_mask=np.asarray(prepared.formal_field.state_mask, dtype=bool),
        core_mask=core_mask,
        xcenters=np.asarray(prepared.formal_field.xcenters, dtype=np.float64),
        ycenters=np.asarray(prepared.formal_field.ycenters, dtype=np.float64),
        t_null=t_null,
    )

    reconstruction_audit.update(
        {
            "formal_field_estimator_reproduced_to_1e-12": True,
            "archived_field_audit": saved_field_audit,
            "resolved_activity_off_target_aliases": resolved_aliases,
        }
    )
    save_json(reconstruction_audit, metadata_root / f"{args.analysis_split}_reconstruction_and_field_audit.json")
    save_json(first_permutation_audit, metadata_root / f"{args.analysis_split}_first_permutation_audit.json")

    manifest = {
        "script": Path(__file__).name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": float(time.time() - started),
        "stage1_root": str(stage1_root),
        "formal_stage1_script": str(stage1_script),
        "formal_stage1_script_sha256": file_sha256(stage1_script),
        "analysis_split": args.analysis_split,
        "primary_manuscript_split": "A_val",
        "confirmation_output_only": bool(args.confirmation_output_only),
        "rows_in_analysis_panel": panel_row_count,
        "users_in_analysis_panel": panel_user_count,
        "valid_drift_rows": int(len(prepared.drift_row_indices)),
        "replicates": int(args.replicates),
        "base_seed": int(args.seed),
        "primary_coordinates": ["M", "Psi"],
        "null_definition": {
            "preserved": [
                "observed current M/Psi anchors",
                "response and exposure denominator increments",
                "user-balanced row weights",
                "current-state occupancy and grid support",
                "A_train-defined convergence core and shell radius",
                "joint marginal distribution of normalized response/exposure innovations",
            ],
            "randomized": "joint normalized signed-innovation pair reassigned away from its observed current state",
            "permutation": "disjoint hierarchical opportunity-matched cyclic permutations in randomized within-group base orders",
            "refitted_objects": [],
            "mesostate_or_model_use": "none",
        },
        "matching_cutpoints": cutpoints,
        "matching_cutpoint_audit": cutpoint_audit,
        "matching_fallback_coverage_table": str(
            find_table(table_root / f"{args.analysis_split}_matching_fallback_coverage").resolve()
        ),
        "frozen_core_path": str(core_mask_path.resolve()),
        "frozen_core_sha256": file_sha256(core_mask_path),
        "frozen_thresholds_path": str(thresholds_path.resolve()),
        "frozen_thresholds_sha256": file_sha256(thresholds_path),
        "shell_radius": shell_radius,
        "full_field_primary_test": full_field_row,
        "basin_metric_tests": summary_rows,
        "pure_ratio_baseline_role": "descriptive denominator-growth-only baseline; not the formal permutation null",
        "quality_gates": {
            "same_row_phase_reconstruction": True,
            "next_state_coordinate_reconstruction": True,
            "next_mass_decay_audit": True,
            "formal_field_estimator_reproduced": True,
            "archived_stage1_field_matched": bool(saved_field_audit.get("skipped") is False),
            "joint_innovation_pairs_permuted_together": True,
            "global_innovation_marginals_preserved": True,
            "frozen_A_train_core_reused": True,
            "coordinate_or_region_refit": False,
            "B_confirm_used_for_definition_or_selection": False,
        },
        "smoke_test_max_users": int(args.max_users),
        "visualization_outputs": "none; publication figures should be generated separately from saved tables and arrays",
    }
    save_json(manifest, metadata_root / f"{args.analysis_split}_construction_null_manifest.json")
    print(f"[construction null] completed: {output_root}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the accounting-matched construction null for frozen Stage-1 M-Psi dynamics."
    )
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--stage1-script", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--analysis-split", choices=["A_val", "B_confirm"], default="A_val")
    parser.add_argument(
        "--confirmation-output-only",
        action="store_true",
        help="Required to run B_confirm after the A_val protocol and interpretation are frozen.",
    )
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--matching-fit-max-rows", type=int, default=500000)
    parser.add_argument("--read-chunk-rows", type=int, default=250000)
    parser.add_argument("--max-last-resort-fraction", type=float, default=0.01)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="0 uses the full split; positive values are smoke-test only and skip archived-field equality.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.replicates < 20:
        raise ValueError("Use at least 20 null replicates; the publication default is 100.")
    if args.matching_fit_max_rows < 10000:
        raise ValueError("--matching-fit-max-rows must be at least 10000.")
    if not (0.0 <= args.max_last_resort_fraction <= 1.0):
        raise ValueError("--max-last-resort-fraction must lie in [0,1].")
    run_analysis(args)


if __name__ == "__main__":
    main()
