#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
CONTROL_COORDINATE = "MR_PhiAI"
CONTROL_SECOND_AXIS = "Phi_activity_idle"
CONTROL_STATE_LABEL = "alignment_free_M_Phi"
CONTROL_INTERPRETATION = "denominator-matched content-alignment-free activity--idle comparator"
CONTROL_Y_COLUMN = "activity_idle_balance_Phi_pre"
CONTROL_Y_POST_COLUMN = "activity_idle_balance_Phi_post"
CONTROL_Y_NEXT_COLUMN = "next_activity_idle_balance_Phi"
CONTROL_DY_COLUMN = "delta_activity_idle_balance_Phi_next"

BASE_COLUMNS = [
    "user_id",
    "bundle_step_index",
    "part",
    "next_gap_days",
    "has_next_submitted_bundle",
    "has_next_within_observation_horizon",
    "long_gap_or_no_next",
    "M_response_prebalanced_pre",
    "activity_alignment_order_Psi_pre",
    "next_M_response_prebalanced",
    "next_activity_alignment_order_Psi",
    "delta_M_response_prebalanced_next",
    "delta_activity_alignment_order_Psi_next",
    "activity_active_mass_pre",
    "activity_idle_mass_pre",
    "activity_active_mass_post",
    "activity_idle_mass_post",
    "next_activity_active_mass",
    "next_activity_idle_mass",
    "response_active_mass_interval",
    "support_active_total_interval",
    "idle_mass_interval",
]


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_module(path: Path, name: str) -> Any:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    specification = importlib.util.spec_from_file_location(name, str(resolved))
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not import {resolved}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def table_path(base: Path) -> Path:
    path = Path(base)
    if path.suffix in {".parquet", ".csv", ".gz"} and path.exists():
        return path
    for extension in (".parquet", ".csv.gz", ".csv"):
        candidate = path.with_suffix(extension)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find table for {base}.[parquet|csv.gz|csv]")


def read_table(base: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    path = table_path(base)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    return pd.read_csv(path, usecols=list(columns) if columns is not None else None, low_memory=False)


def write_table(frame: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = base.with_suffix(".parquet")
        frame.to_parquet(path, index=False)
        return path
    except Exception:
        path = base.with_suffix(".csv.gz")
        frame.to_csv(path, index=False, compression="gzip")
        return path


def numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise KeyError(column)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return bool(int(value))
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return bool(default) if not math.isfinite(number) else bool(number)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "", "nan", "none", "<na>"}:
        return False
    return bool(default)


def safe_max_abs(first: np.ndarray, second: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    return float(np.max(np.abs(left[valid] - right[valid]))) if np.any(valid) else 0.0


def bounded_balance(active: np.ndarray, idle: np.ndarray) -> np.ndarray:
    active_values = np.asarray(active, dtype=np.float64)
    idle_values = np.asarray(idle, dtype=np.float64)
    denominator = active_values + idle_values
    output = np.full(len(denominator), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(active_values)
        & np.isfinite(idle_values)
        & (active_values >= -1e-9)
        & (idle_values >= -1e-9)
        & (denominator > EPS)
    )
    output[valid] = (active_values[valid] - idle_values[valid]) / denominator[valid]
    return np.clip(output, -1.0, 1.0)


def distribution_summary(values: np.ndarray) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "finite_rows": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "q2p5": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "q97p5": float("nan"),
            "iqr": float("nan"),
            "fraction_abs_ge_0p95": float("nan"),
        }
    quantiles = np.quantile(array, [0.025, 0.25, 0.50, 0.75, 0.975])
    return {
        "finite_rows": int(array.size),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "q2p5": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q97p5": float(quantiles[4]),
        "iqr": float(quantiles[3] - quantiles[1]),
        "fraction_abs_ge_0p95": float(np.mean(np.abs(array) >= 0.95)),
    }


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    first = np.asarray(values_a, dtype=np.float64)
    second = np.asarray(values_b, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    if int(np.sum(valid)) < 3:
        return float("nan")
    first_rank = pd.Series(first[valid]).rank(method="average").to_numpy(dtype=np.float64)
    second_rank = pd.Series(second[valid]).rank(method="average").to_numpy(dtype=np.float64)
    return pearson(first_rank, second_rank)


def axis_distribution_audit(
    psi: np.ndarray,
    phi: np.ndarray,
    active: np.ndarray,
    idle: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, Any]:
    valid = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(psi)
        & np.isfinite(phi)
        & np.isfinite(active)
        & np.isfinite(idle)
        & ((np.asarray(active, dtype=float) + np.asarray(idle, dtype=float)) > EPS)
    )
    denominator = np.asarray(active, dtype=np.float64) + np.asarray(idle, dtype=np.float64)
    idle_share = np.full(len(denominator), np.nan, dtype=np.float64)
    idle_share[valid] = np.asarray(idle, dtype=np.float64)[valid] / denominator[valid]
    return {
        "formal_Psi": distribution_summary(np.asarray(psi, dtype=np.float64)[valid]),
        "alignment_free_Phi": distribution_summary(np.asarray(phi, dtype=np.float64)[valid]),
        "idle_share": distribution_summary(idle_share[valid]),
        "pearson_Psi_Phi": pearson(np.asarray(psi, dtype=np.float64)[valid], np.asarray(phi, dtype=np.float64)[valid]),
        "spearman_Psi_Phi": spearman(np.asarray(psi, dtype=np.float64)[valid], np.asarray(phi, dtype=np.float64)[valid]),
        "pearson_Phi_idle_share": pearson(np.asarray(phi, dtype=np.float64)[valid], idle_share[valid]),
        "pearson_Psi_idle_share": pearson(np.asarray(psi, dtype=np.float64)[valid], idle_share[valid]),
        "same_finite_rows_used": True,
    }


def add_nonsemantic_coordinate(frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    missing = [column for column in BASE_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Stage-1 panel is missing alignment-free comparator columns: {missing}")

    output = frame.copy()
    current_m = numeric_array(output, "M_response_prebalanced_pre")
    current_psi = numeric_array(output, "activity_alignment_order_Psi_pre")
    next_m = numeric_array(output, "next_M_response_prebalanced")
    next_psi = numeric_array(output, "next_activity_alignment_order_Psi")
    delta_m = numeric_array(output, "delta_M_response_prebalanced_next")
    delta_psi = numeric_array(output, "delta_activity_alignment_order_Psi_next")

    active_pre = numeric_array(output, "activity_active_mass_pre")
    idle_pre = numeric_array(output, "activity_idle_mass_pre")
    active_post = numeric_array(output, "activity_active_mass_post")
    idle_post = numeric_array(output, "activity_idle_mass_post")
    active_next = numeric_array(output, "next_activity_active_mass")
    idle_next = numeric_array(output, "next_activity_idle_mass")

    formal_state_valid = np.isfinite(current_m) & np.isfinite(current_psi)
    formal_drift_valid = formal_state_valid & np.isfinite(delta_m) & np.isfinite(delta_psi)

    phi_pre_raw = bounded_balance(active_pre, idle_pre)
    phi_post_raw = bounded_balance(active_post, idle_post)
    phi_next_raw = bounded_balance(active_next, idle_next)
    phi_pre = np.where(formal_state_valid, phi_pre_raw, np.nan)
    phi_post = np.where(formal_drift_valid, phi_post_raw, np.nan)
    phi_next = np.where(formal_drift_valid, phi_next_raw, np.nan)
    delta_phi = np.where(formal_drift_valid, phi_next - phi_pre, np.nan)

    output[CONTROL_Y_COLUMN] = phi_pre
    output[CONTROL_Y_POST_COLUMN] = phi_post
    output[CONTROL_Y_NEXT_COLUMN] = phi_next
    output[CONTROL_DY_COLUMN] = delta_phi

    recomputed_delta_m = next_m - current_m
    delta_m_error = safe_max_abs(delta_m, recomputed_delta_m)
    if delta_m_error > 2e-6:
        raise RuntimeError(f"Frozen M increment mismatch: {delta_m_error:.3e}")

    recomputed_delta_psi = next_psi - current_psi
    delta_psi_error = safe_max_abs(delta_psi, recomputed_delta_psi)
    if delta_psi_error > 2e-6:
        raise RuntimeError(f"Frozen Psi increment mismatch: {delta_psi_error:.3e}")

    b_pre = active_pre + idle_pre
    b_post = active_post + idle_post
    g_pre = active_pre - idle_pre
    g_post = active_post - idle_post
    a_phi = b_post - b_pre
    j_phi = g_post - g_pre
    finite_increment = np.isfinite(a_phi) & np.isfinite(j_phi)
    if np.any(a_phi[finite_increment] < -2e-6):
        minimum = float(np.min(a_phi[finite_increment]))
        raise RuntimeError(f"Activity denominator decreased within an interval: minimum={minimum:.3e}")
    a_phi = np.where(np.isfinite(a_phi), np.maximum(a_phi, 0.0), np.nan)

    z_phi = np.zeros(len(output), dtype=np.float64)
    positive = finite_increment & (a_phi > EPS)
    z_phi[positive] = j_phi[positive] / a_phi[positive]
    zero_violation = finite_increment & (a_phi <= EPS) & (np.abs(j_phi) > 2e-6)
    if np.any(zero_violation):
        raise RuntimeError(
            "Non-zero activity--idle innovation with zero denominator increment: "
            f"{int(np.sum(zero_violation))}"
        )
    bound_excess = np.maximum(np.abs(z_phi[finite_increment]) - 1.0, 0.0)
    max_bound_excess = float(np.max(bound_excess)) if bound_excess.size else 0.0
    if max_bound_excess > 2e-6:
        raise RuntimeError(f"Activity--idle normalized innovation exceeds [-1,1]: {max_bound_excess:.3e}")
    z_phi = np.clip(z_phi, -1.0, 1.0)

    post_denominator = b_pre + a_phi
    reconstructed_post = np.full(len(output), np.nan, dtype=np.float64)
    valid_reconstruction = np.isfinite(g_pre) & np.isfinite(post_denominator) & (post_denominator > EPS)
    reconstructed_post[valid_reconstruction] = (
        g_pre[valid_reconstruction] + a_phi[valid_reconstruction] * z_phi[valid_reconstruction]
    ) / post_denominator[valid_reconstruction]
    post_error = safe_max_abs(reconstructed_post, phi_post_raw, formal_drift_valid)
    decay_invariance_error = safe_max_abs(phi_post_raw, phi_next_raw, formal_drift_valid)
    next_error = safe_max_abs(reconstructed_post, phi_next_raw, formal_drift_valid)
    if max(post_error, decay_invariance_error, next_error) > 2e-6:
        raise RuntimeError(
            "Activity--idle state reconstruction failed: "
            f"post={post_error:.3e}, decay={decay_invariance_error:.3e}, next={next_error:.3e}"
        )

    output["activity_idle_balance_denominator_pre"] = b_pre
    output["activity_idle_balance_numerator_pre"] = g_pre
    output["activity_idle_balance_denominator_increment"] = a_phi
    output["activity_idle_balance_signed_innovation"] = j_phi
    output["activity_idle_balance_normalized_innovation"] = z_phi

    audit = {
        "rows": int(len(output)),
        "formal_state_valid_rows": int(np.sum(formal_state_valid)),
        "formal_drift_valid_rows": int(np.sum(formal_drift_valid)),
        "finite_current_control_state_rows": int(np.sum(np.isfinite(phi_pre))),
        "finite_control_increment_rows": int(np.sum(np.isfinite(delta_phi))),
        "positive_control_denominator_increment_rows": int(np.sum(positive & formal_drift_valid)),
        "maximum_normalized_innovation_bound_excess": max_bound_excess,
        "maximum_post_state_reconstruction_error": post_error,
        "maximum_post_to_next_ratio_invariance_error": decay_invariance_error,
        "maximum_next_state_reconstruction_error": next_error,
        "maximum_M_increment_reconstruction_error": delta_m_error,
        "maximum_Psi_increment_reconstruction_error": delta_psi_error,
        "state_eligibility_matched_to_formal_M_Psi": bool(
            np.array_equal(np.isfinite(phi_pre), formal_state_valid)
        ),
        "drift_eligibility_matched_to_formal_M_Psi": bool(
            np.array_equal(np.isfinite(delta_phi), formal_drift_valid)
        ),
        "axis_distribution_audit": axis_distribution_audit(
            current_psi, phi_pre, active_pre, idle_pre, formal_state_valid
        ),
        "definition": "Phi=(total active memory-idle memory)/(total active memory+idle memory)",
        "content_demand_alignment_used_by_second_axis": False,
        "event_semantics_still_used_to_define_active_and_idle_mass": True,
        "interpretation": CONTROL_INTERPRETATION,
    }
    return output, audit


def control_spec(stage1: Any) -> Any:
    return stage1.CoordinateSpec(
        name=CONTROL_COORDINATE,
        xcol="M_response_prebalanced_pre",
        ycol=CONTROL_Y_COLUMN,
        dxcol="delta_M_response_prebalanced_next",
        dycol=CONTROL_DY_COLUMN,
        xbins=np.asarray(stage1.GRID_BINS_SIGNED, dtype=float),
        ybins=np.asarray(stage1.GRID_BINS_SIGNED, dtype=float),
        y_short="PhiAI",
        role="denominator-matched content-alignment-free activity--idle comparator",
    )


def coerce_cutpoints(payload: Mapping[str, Any], cmn: Optional[Any] = None) -> Any:
    converted: Dict[str, Any] = {}
    for key, value in payload.items():
        converted[key] = np.asarray(value, dtype=float) if isinstance(value, list) else value
    if cmn is not None and hasattr(cmn, "MatchingCutpoints"):
        try:
            return cmn.MatchingCutpoints(**converted)
        except TypeError:
            pass
    return SimpleNamespace(**converted)


def compare_coverage_tables(current: pd.DataFrame, archived: pd.DataFrame) -> pd.DataFrame:
    required = ["level", "rows_assigned", "rows_remaining_after_level"]
    for label, frame in (("current", current), ("archived", archived)):
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise RuntimeError(f"{label} matching coverage is missing columns: {missing}")
    columns = required + (["matching_keys"] if "matching_keys" in current.columns and "matching_keys" in archived.columns else [])
    first = current[columns].copy()
    second = archived[columns].copy()
    first["level"] = first["level"].astype(str)
    second["level"] = second["level"].astype(str)
    merged = first.merge(second, on="level", how="outer", suffixes=("_current", "_archived"), indicator=True)
    passed = (
        (merged["_merge"] == "both")
        & (
            pd.to_numeric(merged["rows_assigned_current"], errors="coerce")
            == pd.to_numeric(merged["rows_assigned_archived"], errors="coerce")
        )
        & (
            pd.to_numeric(merged["rows_remaining_after_level_current"], errors="coerce")
            == pd.to_numeric(merged["rows_remaining_after_level_archived"], errors="coerce")
        )
    )
    if "matching_keys_current" in merged.columns:
        passed &= merged["matching_keys_current"].astype(str) == merged["matching_keys_archived"].astype(str)
    merged["passed"] = passed
    return merged


def custom_prepared_analysis(
    cmn: Any,
    stage1: Any,
    frame: pd.DataFrame,
    innovations: Any,
    specification: Any,
) -> Tuple[Any, Dict[str, Any]]:
    transformed, coordinate_audit = add_nonsemantic_coordinate(frame)
    field_columns = ["user_id", specification.xcol, specification.ycol, specification.dxcol, specification.dycol]
    field_frame = stage1.downcast_frame(transformed[field_columns].copy())
    formal_stats = stage1.occupancy_drift_stats(field_frame, specification)
    formal_field = stage1.field_stats_from_dict(formal_stats)
    weights_full = np.asarray(stage1.user_balanced_weights(field_frame), dtype=np.float64)

    x_full = numeric_array(field_frame, specification.xcol)
    y_full = numeric_array(field_frame, specification.ycol)
    observed_dx_full = numeric_array(field_frame, specification.dxcol)
    observed_dy_full = numeric_array(field_frame, specification.dycol)
    ix = stage1.digitize_closed_right(x_full, specification.xbins)
    iy = stage1.digitize_closed_right(y_full, specification.ybins)
    nx = len(specification.xbins) - 1
    ny = len(specification.ybins) - 1
    custom_finite = (
        np.isfinite(x_full)
        & np.isfinite(y_full)
        & np.isfinite(observed_dx_full)
        & np.isfinite(observed_dy_full)
    )
    formal_drift_valid = np.asarray(innovations.formal_drift_valid, dtype=bool)
    drift_valid = formal_drift_valid & custom_finite & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    missing_formal_rows = int(np.sum(formal_drift_valid & ~custom_finite))
    if missing_formal_rows:
        raise RuntimeError(f"The control coordinate lost {missing_formal_rows} formal drift rows.")
    row_indices = np.flatnonzero(drift_valid).astype(np.int64)
    cell = (ix[row_indices] * ny + iy[row_indices]).astype(np.int64)

    b_pre_all = numeric_array(transformed, "activity_idle_balance_denominator_pre")
    g_pre_all = numeric_array(transformed, "activity_idle_balance_numerator_pre")
    a_phi_all = numeric_array(transformed, "activity_idle_balance_denominator_increment")
    z_phi_all = numeric_array(transformed, "activity_idle_balance_normalized_innovation")
    b_difference = safe_max_abs(np.asarray(innovations.b_pre, dtype=float), b_pre_all, formal_drift_valid)
    a_difference = safe_max_abs(np.asarray(innovations.a_psi, dtype=float), a_phi_all, formal_drift_valid)
    if b_difference > 2e-6 or a_difference > 2e-6:
        raise RuntimeError(
            "The alignment-free control did not preserve the formal exposure denominator path: "
            f"pre={b_difference:.3e}, increment={a_difference:.3e}"
        )

    prepared = cmn.PreparedAnalysis(
        drift_row_indices=row_indices,
        x=x_full[row_indices],
        y=y_full[row_indices],
        user_id=np.asarray(innovations.user_id, dtype=np.int64)[row_indices],
        part=np.asarray(innovations.part)[row_indices],
        gap_days=np.asarray(innovations.gap_days, dtype=np.float64)[row_indices],
        weight=weights_full[row_indices],
        cell=cell,
        observed_dx=observed_dx_full[row_indices],
        observed_dy=observed_dy_full[row_indices],
        e_pre=np.asarray(innovations.e_pre, dtype=np.float64)[row_indices],
        s_pre=np.asarray(innovations.s_pre, dtype=np.float64)[row_indices],
        b_pre=b_pre_all[row_indices],
        g_pre=g_pre_all[row_indices],
        a_m=np.asarray(innovations.a_m, dtype=np.float64)[row_indices],
        z_m=np.asarray(innovations.z_m, dtype=np.float64)[row_indices],
        a_psi=a_phi_all[row_indices],
        z_psi=z_phi_all[row_indices],
        response_active=np.asarray(innovations.response_active, dtype=np.float64)[row_indices],
        support_active=np.asarray(innovations.support_active, dtype=np.float64)[row_indices],
        idle_mass=np.asarray(innovations.idle_mass, dtype=np.float64)[row_indices],
        m_denominator=(
            np.asarray(innovations.e_pre, dtype=np.float64)
            + np.asarray(innovations.a_m, dtype=np.float64)
        )[row_indices],
        psi_denominator=(b_pre_all + a_phi_all)[row_indices],
        formal_field=formal_field,
        specification=specification,
    )

    reconstructed_u, reconstructed_v = cmn.aggregate_mean_field(
        prepared,
        prepared.observed_dx,
        prepared.observed_dy,
    )
    mask = np.asarray(formal_field.drift_mask, dtype=bool)
    max_u = safe_max_abs(reconstructed_u, np.asarray(formal_field.drift_u), mask)
    max_v = safe_max_abs(reconstructed_v, np.asarray(formal_field.drift_v), mask)
    if max(max_u, max_v) > 1e-12:
        raise RuntimeError(
            "Optimized control field aggregation does not reproduce Stage-1: "
            f"M={max_u:.3e}, Phi={max_v:.3e}"
        )

    audit = {
        **coordinate_audit,
        "formal_drift_rows": int(len(row_indices)),
        "formal_field_M_max_abs_reproduction_error": max_u,
        "formal_field_Phi_max_abs_reproduction_error": max_v,
        "maximum_formal_control_B_pre_difference": b_difference,
        "maximum_formal_control_B_increment_difference": a_difference,
        "formal_and_control_drift_row_sets_identical": bool(
            len(row_indices) == int(np.sum(formal_drift_valid))
        ),
    }
    return prepared, audit


def pearson(values_a: np.ndarray, values_b: np.ndarray) -> float:
    first = np.asarray(values_a, dtype=float)
    second = np.asarray(values_b, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    if int(np.sum(valid)) < 3:
        return float("nan")
    first = first[valid] - float(np.mean(first[valid]))
    second = second[valid] - float(np.mean(second[valid]))
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > EPS else float("nan")


def field_replication_metrics(train_grid: pd.DataFrame, held_grid: pd.DataFrame) -> Dict[str, Any]:
    keys = ["x_bin", "y_bin"]
    required = set(keys + ["drift_M", "drift_Psi", "drift_supported", "occupancy_probability"])
    for label, frame in (("train", train_grid), ("held", held_grid)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise RuntimeError(f"{label} field grid is missing columns: {missing}")
    merged = train_grid.merge(held_grid, on=keys, suffixes=("_train", "_held"), validate="one_to_one")
    train_support = merged["drift_supported_train"].map(coerce_bool).to_numpy(dtype=bool)
    held_support = merged["drift_supported_held"].map(coerce_bool).to_numpy(dtype=bool)
    common = train_support & held_support
    if int(np.sum(common)) < 3:
        raise RuntimeError("Too few common supported cells for field replication.")

    train_u = pd.to_numeric(merged.loc[common, "drift_M_train"], errors="coerce").to_numpy(dtype=float)
    train_v = pd.to_numeric(merged.loc[common, "drift_Psi_train"], errors="coerce").to_numpy(dtype=float)
    held_u = pd.to_numeric(merged.loc[common, "drift_M_held"], errors="coerce").to_numpy(dtype=float)
    held_v = pd.to_numeric(merged.loc[common, "drift_Psi_held"], errors="coerce").to_numpy(dtype=float)
    train_speed = np.sqrt(train_u * train_u + train_v * train_v)
    held_speed = np.sqrt(held_u * held_u + held_v * held_v)
    cosine = np.full(len(train_u), np.nan, dtype=float)
    nonzero = (train_speed > EPS) & (held_speed > EPS)
    cosine[nonzero] = (
        train_u[nonzero] * held_u[nonzero] + train_v[nonzero] * held_v[nonzero]
    ) / (train_speed[nonzero] * held_speed[nonzero])

    weights = pd.to_numeric(
        merged.loc[common, "occupancy_probability_held"], errors="coerce"
    ).to_numpy(dtype=float)
    weights = np.where(np.isfinite(weights) & (weights >= 0), weights, 0.0)
    cosine_valid = np.isfinite(cosine)
    cosine_weights = weights[cosine_valid]
    cosine_weights = cosine_weights / max(float(np.sum(cosine_weights)), EPS)

    occupancy_train = pd.to_numeric(
        merged["occupancy_probability_train"], errors="coerce"
    ).to_numpy(dtype=float)
    occupancy_held = pd.to_numeric(
        merged["occupancy_probability_held"], errors="coerce"
    ).to_numpy(dtype=float)
    occupancy_train = np.where(np.isfinite(occupancy_train), np.maximum(occupancy_train, 0.0), 0.0)
    occupancy_held = np.where(np.isfinite(occupancy_held), np.maximum(occupancy_held, 0.0), 0.0)
    occupancy_train = occupancy_train / max(float(np.sum(occupancy_train)), EPS)
    occupancy_held = occupancy_held / max(float(np.sum(occupancy_held)), EPS)
    mixture = 0.5 * (occupancy_train + occupancy_held)

    def kl(first: np.ndarray, second: np.ndarray) -> float:
        valid = first > 0
        return float(np.sum(first[valid] * np.log((first[valid] + EPS) / (second[valid] + EPS))))

    union = train_support | held_support
    vector_train = np.column_stack([train_u, train_v]).ravel()
    vector_held = np.column_stack([held_u, held_v]).ravel()
    return {
        "train_supported_cells": int(np.sum(train_support)),
        "held_supported_cells": int(np.sum(held_support)),
        "common_supported_cells": int(np.sum(common)),
        "support_jaccard": float(np.sum(common) / max(int(np.sum(union)), 1)),
        "occupancy_js": 0.5 * kl(occupancy_train, mixture) + 0.5 * kl(occupancy_held, mixture),
        "drift_vector_r": pearson(vector_train, vector_held),
        "drift_M_r": pearson(train_u, held_u),
        "drift_second_axis_r": pearson(train_v, held_v),
        "mean_local_cosine": float(np.nanmean(cosine)),
        "occupancy_weighted_local_cosine": float(
            np.sum(cosine_weights * cosine[cosine_valid])
        ) if np.any(cosine_valid) else float("nan"),
        "drift_speed_r": pearson(train_speed, held_speed),
        "drift_component_rmse": float(
            np.sqrt(np.mean(np.concatenate([train_u - held_u, train_v - held_v]) ** 2))
        ),
        "held_supported_occupancy_mass": float(np.sum(occupancy_held[held_support])),
    }


def weighted_component_distance(
    first: np.ndarray,
    second: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> float:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(first) & np.isfinite(second)
    if not np.any(valid):
        return float("nan")
    selected_weight = np.asarray(weight, dtype=float)[valid]
    selected_weight = selected_weight / max(float(np.sum(selected_weight)), EPS)
    return float(
        np.sqrt(
            np.sum(
                selected_weight
                * (np.asarray(first, dtype=float)[valid] - np.asarray(second, dtype=float)[valid]) ** 2
            )
        )
    )


def null_separation_row(
    metric: str,
    observed: float,
    null_values: np.ndarray,
    primary: bool,
    direction: str = "greater",
) -> Dict[str, Any]:
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError(f"No finite null values for {metric}")
    standard_deviation = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    if direction == "greater":
        exceed = int(np.sum(values >= observed))
    elif direction == "less":
        exceed = int(np.sum(values <= observed))
    else:
        raise ValueError(direction)
    null_mean = float(np.mean(values))
    null_median = float(np.median(values))
    return {
        "metric": metric,
        "primary_endpoint": bool(primary),
        "direction_supporting_departure": direction,
        "observed": float(observed),
        "null_mean": null_mean,
        "null_sd": standard_deviation,
        "null_2p5": float(np.quantile(values, 0.025)),
        "null_50": null_median,
        "null_97p5": float(np.quantile(values, 0.975)),
        "null_standardized_separation": float((observed - null_mean) / standard_deviation)
        if standard_deviation > EPS
        else float("nan"),
        "observed_to_null_median_ratio": float(observed / null_median)
        if abs(null_median) > EPS
        else float("nan"),
        "monte_carlo_p": float((1 + exceed) / (values.size + 1)) if primary else float("nan"),
        "finite_replicates": int(values.size),
    }
