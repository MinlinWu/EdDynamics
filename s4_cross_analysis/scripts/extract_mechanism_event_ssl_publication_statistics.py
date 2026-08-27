#!/usr/bin/env python3
"""Extract cross-stage publication statistics for the frozen minimal mechanism and predictive-state Event-SSL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
GRID_BINS_SIGNED = np.linspace(-1.0, 1.0, 41)
MIN_DRIFT_BIN_COUNT = 30
EXPECTED_MACROSTATE_K = 6
EXPECTED_KMEANS_N_INIT = 20
EXPECTED_KMEANS_FIT_MAX_ROWS = 500000
EXPECTED_RANDOM_STATE = 42
EXPECTED_FAMILY = "offset_dual_channel"
EXPECTED_PRIMARY_MACROSTATE = ("M", "Psi")
EXPECTED_FEATURES = (
    "M_response_prebalanced_pre",
    "activity_alignment_order_Psi_pre",
)

DEFAULT_ROOT = Path(os.environ.get("EDNET_KT4_OUTPUT_ROOT", "/data/datasets/KT4/outputs_KT4"))
DEFAULT_STAGE1_ROOT = DEFAULT_ROOT / "stage1"
DEFAULT_PHASE2_ROOT = DEFAULT_ROOT / "stage2_phase2_freeze"
DEFAULT_PHASE3_ROOT = DEFAULT_ROOT / "stage2_phase3_confirm"
DEFAULT_EVENT_SSL_ROOT = DEFAULT_ROOT / "stage4_event_ssl" / "evaluation_predictive_state"
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "cross_stage_mechanism_event_ssl_comparison"
DEFAULT_SPLIT = "B_confirm"

MAIN_REQUIRED = "main_text_required"
MAIN_RECOMMENDED = "main_text_recommended"
SUPPLEMENT_REQUIRED = "supplement_required"


@dataclass(frozen=True)
class SourceRecord:
    name: str
    path: Optional[Path]
    status: str
    sha256: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    note: str = ""


@dataclass(frozen=True)
class FrozenPartition:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    ordered_centers: np.ndarray
    ordered_centers_scaled: np.ndarray
    raw_centers_scaled: np.ndarray
    raw_to_ordered: np.ndarray
    centers_table: pd.DataFrame
    metadata: Mapping[str, Any]
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class FieldStats:
    u: np.ndarray
    v: np.ndarray
    mask: np.ndarray
    count: np.ndarray
    weight: np.ndarray


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(obj: Any) -> Any:
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
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(obj), handle, indent=2, ensure_ascii=False, allow_nan=False)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def table_path(base_or_path: Path) -> Path:
    path = Path(base_or_path)
    if path.suffix and path.exists():
        return path
    for extension in (".parquet", ".csv.gz", ".csv", ".tsv"):
        candidate = path.with_suffix(extension)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find table for {path}")


def read_table(base_or_path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    path = table_path(base_or_path)
    selected = list(columns) if columns is not None else None
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=selected)
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t", usecols=selected, low_memory=False)
    return pd.read_csv(path, usecols=selected, low_memory=False)


def write_table(df: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = base.with_suffix(".parquet")
        df.to_parquet(path, index=False)
        return path
    except Exception:
        path = base.with_suffix(".csv")
        df.to_csv(path, index=False)
        return path


def record_file(name: str, path: Path, sources: List[SourceRecord], rows: Optional[int] = None, columns: Optional[int] = None, note: str = "") -> None:
    sources.append(SourceRecord(
        name=name,
        path=path.resolve(),
        status="ok",
        sha256=file_sha256(path),
        rows=rows,
        columns=columns,
        note=note,
    ))


def source_table(sources: Sequence[SourceRecord]) -> pd.DataFrame:
    return pd.DataFrame([{
        "name": item.name,
        "path": str(item.path) if item.path is not None else None,
        "status": item.status,
        "sha256": item.sha256,
        "rows": item.rows,
        "columns": item.columns,
        "note": item.note,
    } for item in sources])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    x0 = x[valid] - float(np.mean(x[valid]))
    y0 = y[valid] - float(np.mean(y[valid]))
    denominator = float(np.linalg.norm(x0) * np.linalg.norm(y0))
    return float(np.dot(x0, y0) / denominator) if denominator > EPS else float("nan")


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    if not valid.any():
        return float("nan")
    total = float(np.sum(w[valid]))
    if total <= EPS:
        return float("nan")
    return float(np.sum(w[valid] * x[valid]) / total)


def weighted_rmse(error: np.ndarray, weights: np.ndarray) -> float:
    value = weighted_mean(np.asarray(error, dtype=float) ** 2, weights)
    return float(np.sqrt(value)) if np.isfinite(value) else float("nan")


def weighted_mae(error: np.ndarray, weights: np.ndarray) -> float:
    return weighted_mean(np.abs(np.asarray(error, dtype=float)), weights)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    return float(np.sqrt(np.mean((x[valid] - y[valid]) ** 2))) if valid.any() else float("nan")


def mae(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    return float(np.mean(np.abs(x[valid] - y[valid]))) if valid.any() else float("nan")


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    a = np.asarray(p, dtype=float).ravel()
    b = np.asarray(q, dtype=float).ravel()
    a = a / max(float(np.nansum(a)), EPS)
    b = b / max(float(np.nansum(b)), EPS)
    midpoint = 0.5 * (a + b)

    def kl(x: np.ndarray, y: np.ndarray) -> float:
        valid = x > 0
        return float(np.sum(x[valid] * np.log((x[valid] + EPS) / (y[valid] + EPS))))

    return 0.5 * kl(a, midpoint) + 0.5 * kl(b, midpoint)


def user_balanced_weights(user_id: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(user_id))
    counts = series.groupby(series).transform("count").to_numpy(dtype=float)
    return 1.0 / np.maximum(counts, 1.0)


def digitize_closed_right(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    adjusted = np.where(array == bins[-1], np.nextafter(bins[-1], bins[0]), array)
    return np.digitize(adjusted, bins) - 1


def occupancy_grid(x: np.ndarray, y: np.ndarray, user_id: np.ndarray) -> np.ndarray:
    n = len(GRID_BINS_SIGNED) - 1
    ix = digitize_closed_right(x, GRID_BINS_SIGNED)
    iy = digitize_closed_right(y, GRID_BINS_SIGNED)
    valid = np.isfinite(x) & np.isfinite(y) & (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    weights = user_balanced_weights(user_id)
    flat = ix[valid] * n + iy[valid]
    histogram = np.bincount(flat, weights=weights[valid], minlength=n * n).reshape(n, n).astype(float)
    return histogram / max(float(histogram.sum()), EPS)


def field_stats(x: np.ndarray, y: np.ndarray, dx: np.ndarray, dy: np.ndarray, user_id: np.ndarray) -> FieldStats:
    n = len(GRID_BINS_SIGNED) - 1
    ix = digitize_closed_right(x, GRID_BINS_SIGNED)
    iy = digitize_closed_right(y, GRID_BINS_SIGNED)
    valid = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(dx) & np.isfinite(dy)
        & (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    )
    weights = user_balanced_weights(user_id)
    flat = ix[valid] * n + iy[valid]
    count = np.bincount(flat, minlength=n * n).reshape(n, n).astype(float)
    weight = np.bincount(flat, weights=weights[valid], minlength=n * n).reshape(n, n).astype(float)
    sum_x = np.bincount(flat, weights=weights[valid] * dx[valid], minlength=n * n).reshape(n, n).astype(float)
    sum_y = np.bincount(flat, weights=weights[valid] * dy[valid], minlength=n * n).reshape(n, n).astype(float)
    u = sum_x / np.maximum(weight, EPS)
    v = sum_y / np.maximum(weight, EPS)
    return FieldStats(u=u, v=v, mask=count >= MIN_DRIFT_BIN_COUNT, count=count, weight=weight)


def vector_corr(first: FieldStats, second: FieldStats) -> float:
    mask = first.mask & second.mask
    if mask.sum() < 3:
        return float("nan")
    a = np.column_stack([first.u[mask], first.v[mask]]).ravel()
    b = np.column_stack([second.u[mask], second.v[mask]]).ravel()
    return pearson(a, b)


def speed_corr(first: FieldStats, second: FieldStats) -> float:
    mask = first.mask & second.mask
    if mask.sum() < 3:
        return float("nan")
    speed_first = np.sqrt(first.u[mask] ** 2 + first.v[mask] ** 2)
    speed_second = np.sqrt(second.u[mask] ** 2 + second.v[mask] ** 2)
    return pearson(speed_first, speed_second)


def local_cosine_array(first: FieldStats, second: FieldStats) -> np.ndarray:
    dot = first.u * second.u + first.v * second.v
    speed_first = np.sqrt(first.u ** 2 + first.v ** 2)
    speed_second = np.sqrt(second.u ** 2 + second.v ** 2)
    cosine = dot / np.maximum(speed_first * speed_second, EPS)
    cosine[~(first.mask & second.mask)] = np.nan
    return np.clip(cosine, -1.0, 1.0)


def occupancy_weighted_cosine(first: FieldStats, second: FieldStats, reference_weight: np.ndarray) -> float:
    cosine = local_cosine_array(first, second)
    valid = np.isfinite(cosine)
    if not valid.any():
        return float("nan")
    weights = np.asarray(reference_weight, dtype=float)
    total = float(np.nansum(weights[valid]))
    if total <= EPS:
        return float(np.nanmean(cosine[valid]))
    return float(np.nansum(weights[valid] * cosine[valid]) / total)


def field_rmse(first: FieldStats, second: FieldStats) -> float:
    mask = first.mask & second.mask
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.nanmean((second.u[mask] - first.u[mask]) ** 2 + (second.v[mask] - first.v[mask]) ** 2)))


def transition_matrix(current: np.ndarray, next_state: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    cur = np.asarray(current, dtype=np.int64)
    nxt = np.asarray(next_state, dtype=np.int64)
    valid = (cur >= 0) & (cur < k) & (nxt >= 0) & (nxt < k)
    flat = cur[valid] * k + nxt[valid]
    counts = np.bincount(flat, minlength=k * k).reshape(k, k).astype(float)
    row_sum = counts.sum(axis=1, keepdims=True)
    matrix = np.divide(counts, np.maximum(row_sum, EPS), out=np.zeros_like(counts), where=row_sum > 0)
    return counts, matrix


def transition_metrics(prefix: str, first: np.ndarray, second: np.ndarray) -> Dict[str, float]:
    row_tv = 0.5 * np.nansum(np.abs(second - first), axis=1)
    self_first = np.diag(first)
    self_second = np.diag(second)
    k = int(first.shape[0])
    top_n = min(max(k, 1), k * k)
    top_first = set(np.argsort(first.ravel())[::-1][:top_n].tolist())
    top_second = set(np.argsort(second.ravel())[::-1][:top_n].tolist())
    diagonal_first = np.argmax(first, axis=1) == np.arange(k)
    diagonal_second = np.argmax(second, axis=1) == np.arange(k)
    return {
        f"{prefix}_mean_row_tv": float(np.nanmean(row_tv)),
        f"{prefix}_max_row_tv": float(np.nanmax(row_tv)),
        f"{prefix}_self_transition_corr": pearson(self_first, self_second),
        f"{prefix}_self_transition_rmse": rmse(self_first, self_second),
        f"{prefix}_self_transition_mae": mae(self_first, self_second),
        f"{prefix}_top_edge_overlap": float(len(top_first.intersection(top_second)) / max(len(top_first), 1)),
        f"{prefix}_diagonal_match_fraction": float(np.mean(diagonal_first == diagonal_second)),
    }


def load_fixed_partition(stage1_root: Path, sources: List[SourceRecord]) -> FrozenPartition:
    root = Path(stage1_root).resolve() / "dynamics" / "fixed_k6_mesostates"
    metadata_path = root / "fixed_k6_model_metadata.json"
    centers_path = table_path(root / "fixed_k6_centers")
    metadata = load_json(metadata_path)
    centers = read_table(centers_path).sort_values("macrostate", kind="mergesort").reset_index(drop=True)
    record_file("stage1_fixed_k6_metadata", metadata_path, sources)
    record_file("stage1_fixed_k6_centers", centers_path, sources, rows=len(centers), columns=len(centers.columns))

    checks = {
        "coordinate": metadata.get("coordinate") == "MR_PsiA",
        "macrostate_k": int(metadata.get("macrostate_k", -1)) == EXPECTED_MACROSTATE_K,
        "macrostate_k_rule": metadata.get("macrostate_k_rule") == "fixed a priori",
        "features": tuple(metadata.get("features", [])) == EXPECTED_FEATURES,
        "fit_split": metadata.get("fit_split") == "A_train",
        "user_balanced_sampling": metadata.get("user_balanced_sampling") is True,
        "user_balanced_kmeans_fit": metadata.get("user_balanced_kmeans_fit") is True,
        "kmeans_n_init": int(metadata.get("kmeans_n_init", -1)) == EXPECTED_KMEANS_N_INIT,
        "fit_max_rows": int(metadata.get("fit_max_rows", -1)) == EXPECTED_KMEANS_FIT_MAX_ROWS,
        "random_state": int(metadata.get("random_state", -1)) == EXPECTED_RANDOM_STATE,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Stage-1 fixed K=6 contract failed: " + ", ".join(failed))

    required = {"macrostate", "center_M", "center_Psi"}
    missing = sorted(required.difference(centers.columns))
    if missing:
        raise RuntimeError(f"Fixed K=6 center table is missing columns: {missing}")
    ids = pd.to_numeric(centers["macrostate"], errors="coerce").to_numpy(dtype=float)
    ordered_centers = centers[["center_M", "center_Psi"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if len(centers) != EXPECTED_MACROSTATE_K or not np.array_equal(ids, np.arange(EXPECTED_MACROSTATE_K, dtype=float)) or not np.isfinite(ordered_centers).all():
        raise RuntimeError("Fixed K=6 centers are not ordered S0-S5.")

    scaler_mean = np.asarray(metadata.get("scaler_mean", []), dtype=float)
    scaler_scale = np.asarray(metadata.get("scaler_scale", []), dtype=float)
    if scaler_mean.shape != (2,) or scaler_scale.shape != (2,) or not np.isfinite(scaler_mean).all() or not np.isfinite(scaler_scale).all() or not np.all(scaler_scale > 0):
        raise RuntimeError("Invalid fixed K=6 scaler metadata.")
    if {"center_M_standardized", "center_Psi_standardized"}.issubset(centers.columns):
        ordered_scaled = centers[["center_M_standardized", "center_Psi_standardized"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    else:
        ordered_scaled = (ordered_centers - scaler_mean) / scaler_scale

    mapping = {int(key): int(value) for key, value in dict(metadata.get("raw_to_ordered_label", {})).items()}
    labels = list(range(EXPECTED_MACROSTATE_K))
    if sorted(mapping) != labels or sorted(mapping.values()) != labels:
        raise RuntimeError("Invalid raw-to-ordered KMeans label map.")
    raw_to_ordered = np.asarray([mapping[index] for index in labels], dtype=np.int64)
    raw_scaled = np.empty_like(ordered_scaled)
    for raw, ordered in enumerate(raw_to_ordered):
        raw_scaled[raw] = ordered_scaled[ordered]

    audit = {
        "source": "frozen Stage-1 fixed K=6 partition",
        "coordinate": "MR_PsiA",
        "macrostate_k": EXPECTED_MACROSTATE_K,
        "macrostate_k_rule": "fixed a priori",
        "fit_split": "A_train",
        "features": list(EXPECTED_FEATURES),
        "user_balanced_sampling": True,
        "user_balanced_kmeans_fit": True,
        "fit_max_rows": EXPECTED_KMEANS_FIT_MAX_ROWS,
        "kmeans_n_init": EXPECTED_KMEANS_N_INIT,
        "random_state": EXPECTED_RANDOM_STATE,
        "metadata_path": str(metadata_path),
        "metadata_sha256": file_sha256(metadata_path),
        "centers_path": str(centers_path),
        "centers_sha256": file_sha256(centers_path),
        "kmeans_refit": False,
        "macrostate_k_selected": False,
        "confirmation_data_used_for_partition": False,
        "checks": checks,
    }
    return FrozenPartition(
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        ordered_centers=ordered_centers,
        ordered_centers_scaled=ordered_scaled,
        raw_centers_scaled=raw_scaled,
        raw_to_ordered=raw_to_ordered,
        centers_table=centers,
        metadata=metadata,
        audit=audit,
    )


def predict_macro_labels(partition: FrozenPartition, points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    labels = np.full(values.shape[0], -1, dtype=np.int64)
    valid = np.isfinite(values).all(axis=1)
    if valid.any():
        standardized = (values[valid] - partition.scaler_mean[None, :]) / partition.scaler_scale[None, :]
        distances = np.sum((standardized[:, None, :] - partition.raw_centers_scaled[None, :, :]) ** 2, axis=2)
        raw = np.argmin(distances, axis=1)
        labels[valid] = partition.raw_to_ordered[raw]
    return labels


def validate_contracts(
    stage1_root: Path,
    phase2_root: Path,
    phase3_root: Path,
    event_ssl_root: Path,
    split: str,
    partition: FrozenPartition,
    sources: List[SourceRecord],
) -> Dict[str, Any]:
    phase2_path = phase2_root / "metadata" / "phase2_frozen_model_manifest.json"
    phase3_path = phase3_root / "metadata" / "phase3_confirmation_manifest.json"
    no_update_path = phase3_root / "metadata" / "phase3_no_update_audit.json"
    event_path = event_ssl_root / "metadata" / "stage4_event_ssl_evaluation_manifest.json"
    event_partition_path = event_ssl_root / "metadata" / "stage4_event_ssl_fixed_k6_partition_audit.json"

    phase2 = load_json(phase2_path)
    phase3 = load_json(phase3_path)
    no_update = load_json(no_update_path)
    event = load_json(event_path)
    event_partition = load_json(event_partition_path)
    for name, path in (
        ("phase2_frozen_model_manifest", phase2_path),
        ("phase3_confirmation_manifest", phase3_path),
        ("phase3_no_update_audit", no_update_path),
        ("event_ssl_evaluation_manifest", event_path),
        ("event_ssl_fixed_k6_partition_audit", event_partition_path),
    ):
        record_file(name, path, sources)

    failures: List[str] = []
    if tuple(phase2.get("primary_macrostate", [])) != EXPECTED_PRIMARY_MACROSTATE:
        failures.append("Phase 2 primary macrostate")
    if tuple(phase3.get("primary_macrostate", [])) != EXPECTED_PRIMARY_MACROSTATE:
        failures.append("Phase 3 primary macrostate")
    if tuple(event.get("primary_coordinates", [])) != EXPECTED_PRIMARY_MACROSTATE:
        failures.append("Event-SSL primary coordinates")
    frozen = dict(phase2.get("frozen_parameters", {}))
    if str(frozen.get("family_key", "")) != EXPECTED_FAMILY:
        failures.append("Phase 2 family")
    if str(dict(phase3.get("frozen_parameters", {})).get("family_key", "")) != EXPECTED_FAMILY:
        failures.append("Phase 3 family")
    if str(event.get("model_kind", "")) != "predictive_state":
        failures.append("Event-SSL model kind")
    if str(phase3.get("confirm_split", "")) != str(split):
        failures.append("Phase 3 confirmation split")
    if str(split) not in dict(event.get("splits", {})):
        failures.append("Event-SSL split")

    phase2_guardrails = dict(phase2.get("guardrails", {}))
    phase3_guardrails = dict(phase3.get("guardrails", {}))
    event_guardrails = dict(event.get("guardrails", {}))
    for name in ("parameter_search_opened", "mechanism_family_reselected", "mechanism_parameters_refit", "kmeans_refit", "macrostate_k_selected", "region_redefinition"):
        if bool(phase2_guardrails.get(name, False)):
            failures.append(f"Phase 2 guardrail {name}")
    for name in ("parameter_search_opened", "calibration_reestimated", "mechanism_family_reselected", "mechanism_parameters_refit", "kmeans_refit", "macrostate_k_selected", "region_redefinition", "B_confirm_used_for_update"):
        if bool(phase3_guardrails.get(name, False)):
            failures.append(f"Phase 3 guardrail {name}")
    for name in ("kmeans_refit", "macrostate_k_selected", "B_confirm_used_for_update"):
        if bool(event_guardrails.get(name, False)):
            failures.append(f"Event-SSL guardrail {name}")
    if str(phase3_guardrails.get("confirmation_mode", "")) != "output_only":
        failures.append("Phase 3 confirmation mode")

    for before, after in (
        ("frozen_parameter_hash_before_confirmation", "frozen_parameter_hash_after_confirmation"),
        ("frozen_calibration_hash_before_confirmation", "frozen_calibration_hash_after_confirmation"),
    ):
        if str(no_update.get(before, "")) != str(no_update.get(after, "")):
            failures.append(f"Phase 3 hash mismatch {before}")

    metadata_sha = str(partition.audit["metadata_sha256"])
    centers_sha = str(partition.audit["centers_sha256"])
    contracts = (
        ("Phase 2", dict(phase2.get("stage1_fixed_k6_contract", {}))),
        ("Phase 3", dict(phase3.get("stage1_fixed_k6_contract", {}))),
        ("Event-SSL", dict(event.get("macro_partition", {}))),
        ("Event-SSL audit", event_partition),
    )
    for label, contract in contracts:
        contract_metadata = dict(contract.get("metadata") or {})
    
        macrostate_k = contract.get("macrostate_k")
        if macrostate_k is None:
            macrostate_k = contract_metadata.get("macrostate_k", -1)
    
        if int(macrostate_k) != EXPECTED_MACROSTATE_K:
            failures.append(f"{label} macrostate K")
        if str(contract.get("metadata_sha256", "")) != metadata_sha:
            failures.append(f"{label} metadata hash")
        if str(contract.get("centers_sha256", "")) != centers_sha:
            failures.append(f"{label} centers hash")
        if bool(contract.get("kmeans_refit", False)):
            failures.append(f"{label} KMeans refit")
        if bool(contract.get("macrostate_k_selected", False)):
            failures.append(f"{label} K selection")

    phase2_vector = dict(frozen.get("full_parameter_vector", {}))
    phase3_vector = dict(dict(phase3.get("frozen_parameters", {})).get("full_parameter_vector", {}))
    if phase2_vector != phase3_vector:
        failures.append("frozen parameter vector mismatch")
    if dict(phase2.get("frozen_calibration", {})) != dict(phase3.get("frozen_calibration", {})):
        failures.append("frozen calibration mismatch")

    if failures:
        raise RuntimeError("Cross-stage contract failed: " + "; ".join(failures))

    return {
        "stage1_root": str(stage1_root.resolve()),
        "phase2_manifest": phase2,
        "phase3_manifest": phase3,
        "phase3_no_update_audit": no_update,
        "event_ssl_manifest": event,
        "event_ssl_partition_audit": event_partition,
        "family": EXPECTED_FAMILY,
        "event_ssl_model_kind": "predictive_state",
        "primary_macrostate": list(EXPECTED_PRIMARY_MACROSTATE),
        "split": split,
        "fixed_k6_contract_verified": True,
        "cross_model_fitting": False,
        "confirmation_data_used_for_update": False,
    }


def clean_prediction_frame(df: pd.DataFrame, prediction_columns: Sequence[str], label: str) -> pd.DataFrame:
    output = df.copy()
    required = ["user_id", "bundle_step_index", "M", "Psi", "target_M_next", "target_Psi_next", *prediction_columns]
    missing = [name for name in required if name not in output.columns]
    if missing:
        raise RuntimeError(f"{label} prediction table is missing columns: {missing}")
    for name in required:
        output[name] = pd.to_numeric(output[name], errors="coerce")
    valid = output[required].notna().all(axis=1)
    for name in [value for value in required if value not in ("user_id", "bundle_step_index")]:
        valid &= np.isfinite(output[name].to_numpy(dtype=float))
    output = output.loc[valid, required].copy()
    output["user_id"] = output["user_id"].astype(np.int64)
    output["bundle_step_index"] = output["bundle_step_index"].astype(np.int64)
    if output.duplicated(["user_id", "bundle_step_index"]).any():
        raise RuntimeError(f"{label} prediction keys are not unique.")
    return output.sort_values(["user_id", "bundle_step_index"], kind="mergesort").reset_index(drop=True)


def load_minimal_predictions(phase3_root: Path, phase3_manifest: Mapping[str, Any], split: str, sources: List[SourceRecord]) -> pd.DataFrame:
    output_path = dict(phase3_manifest.get("outputs", {})).get("full_prediction_table")
    path = Path(output_path) if output_path else table_path(phase3_root / "tables" / f"phase3_{split}_full_predictions")
    if not path.exists():
        path = table_path(phase3_root / "tables" / f"phase3_{split}_full_predictions")
    columns = ["user_id", "bundle_step_index", "M", "Psi", "target_M_next", "target_Psi_next", "pred_next_M", "pred_next_Psi"]
    frame = read_table(path, columns=columns).rename(columns={"pred_next_M": "mech_next_M", "pred_next_Psi": "mech_next_Psi"})
    frame = clean_prediction_frame(frame, ("mech_next_M", "mech_next_Psi"), "minimal mechanism")
    record_file("minimal_mechanism_full_predictions", path, sources, rows=len(frame), columns=len(frame.columns))
    return frame


def load_event_ssl_predictions(event_root: Path, event_manifest: Mapping[str, Any], split: str, sources: List[SourceRecord]) -> pd.DataFrame:
    split_payload = dict(dict(event_manifest.get("splits", {})).get(split, {}))
    output_path = split_payload.get("prediction_path")
    path = Path(output_path) if output_path else table_path(event_root / "predictions" / f"stage4_event_ssl_predictions_{split}")
    if not path.exists():
        path = table_path(event_root / "predictions" / f"stage4_event_ssl_predictions_{split}")
    columns = ["user_id", "bundle_step_index", "M", "Psi", "target_M_next", "target_Psi_next", "pred_M", "pred_Psi", "pred_next_M", "pred_next_Psi"]
    frame = read_table(path, columns=columns).rename(columns={
        "pred_M": "ssl_M",
        "pred_Psi": "ssl_Psi",
        "pred_next_M": "ssl_next_M",
        "pred_next_Psi": "ssl_next_Psi",
    })
    frame = clean_prediction_frame(frame, ("ssl_M", "ssl_Psi", "ssl_next_M", "ssl_next_Psi"), "Event-SSL")
    record_file("event_ssl_predictions", path, sources, rows=len(frame), columns=len(frame.columns))
    return frame


def align_predictions(
    mechanism: pd.DataFrame,
    event_ssl: pd.DataFrame,
    anchor_tolerance: float,
    minimum_join_fraction: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    keys = ["user_id", "bundle_step_index"]
    mechanism_columns = keys + ["M", "Psi", "target_M_next", "target_Psi_next", "mech_next_M", "mech_next_Psi"]
    ssl_columns = keys + ["M", "Psi", "target_M_next", "target_Psi_next", "ssl_M", "ssl_Psi", "ssl_next_M", "ssl_next_Psi"]
    joined = pd.merge(
        mechanism[mechanism_columns],
        event_ssl[ssl_columns],
        on=keys,
        how="inner",
        suffixes=("_mech_emp", "_ssl_emp"),
        validate="one_to_one",
    )
    if joined.empty:
        raise RuntimeError("No overlapping rows between the minimal mechanism and Event-SSL predictions.")
    mechanism_fraction = float(len(joined) / max(len(mechanism), 1))
    ssl_fraction = float(len(joined) / max(len(event_ssl), 1))
    if mechanism_fraction < minimum_join_fraction or ssl_fraction < minimum_join_fraction:
        raise RuntimeError(
            f"Prediction join coverage is below the required fraction: mechanism={mechanism_fraction}, Event-SSL={ssl_fraction}."
        )

    audit: Dict[str, Any] = {
        "mechanism_rows": int(len(mechanism)),
        "event_ssl_rows": int(len(event_ssl)),
        "joined_rows": int(len(joined)),
        "joined_users": int(joined["user_id"].nunique()),
        "join_fraction_of_mechanism": mechanism_fraction,
        "join_fraction_of_event_ssl": ssl_fraction,
        "minimum_required_join_fraction": float(minimum_join_fraction),
        "anchor_tolerance": float(anchor_tolerance),
        "empirical_anchor_policy": "mean of the numerically matching formal mechanism and Event-SSL empirical columns",
    }
    for name in ("M", "Psi", "target_M_next", "target_Psi_next"):
        first = joined[f"{name}_mech_emp"].to_numpy(dtype=float)
        second = joined[f"{name}_ssl_emp"].to_numpy(dtype=float)
        maximum = float(np.nanmax(np.abs(first - second)))
        audit[f"max_abs_difference_{name}_between_sources"] = maximum
        audit[f"mean_abs_difference_{name}_between_sources"] = float(np.nanmean(np.abs(first - second)))
        if maximum > anchor_tolerance:
            raise RuntimeError(f"Empirical anchor mismatch for {name}: {maximum} > {anchor_tolerance}")
        joined[name] = 0.5 * (first + second)
    joined = joined.drop(columns=[name for name in joined.columns if name.endswith("_mech_emp") or name.endswith("_ssl_emp")])
    return joined, audit


def load_stage1_assignments(stage1_root: Path, split: str, sources: List[SourceRecord]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    root = Path(stage1_root).resolve() / "dynamics" / "fixed_k6_mesostates"
    assignment_path = table_path(root / f"{split}_fixed_k6_assignments")
    counts_path = table_path(root / f"{split}_fixed_k6_transition_counts")
    matrix_path = table_path(root / f"{split}_fixed_k6_transition_matrix")
    assignments = read_table(assignment_path)
    state_column = "macrostate" if "macrostate" in assignments.columns else "state" if "state" in assignments.columns else None
    if state_column is None:
        raise RuntimeError("Stage-1 assignment table has no macrostate column.")
    assignments = assignments[["user_id", "bundle_step_index", state_column]].rename(columns={state_column: "current_macrostate"})
    for name in ("user_id", "bundle_step_index", "current_macrostate"):
        assignments[name] = pd.to_numeric(assignments[name], errors="coerce")
    assignments = assignments.dropna().copy()
    assignments[["user_id", "bundle_step_index", "current_macrostate"]] = assignments[["user_id", "bundle_step_index", "current_macrostate"]].astype(np.int64)
    if assignments.duplicated(["user_id", "bundle_step_index"]).any():
        raise RuntimeError("Stage-1 fixed K=6 assignment keys are not unique.")
    counts = read_table(counts_path).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    matrix = read_table(matrix_path).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if counts.shape != (EXPECTED_MACROSTATE_K, EXPECTED_MACROSTATE_K) or matrix.shape != counts.shape:
        raise RuntimeError("Unexpected Stage-1 transition matrix shape.")
    record_file("stage1_fixed_k6_assignments", assignment_path, sources, rows=len(assignments), columns=len(assignments.columns))
    record_file("stage1_fixed_k6_transition_counts", counts_path, sources, rows=counts.shape[0], columns=counts.shape[1])
    record_file("stage1_fixed_k6_transition_matrix", matrix_path, sources, rows=matrix.shape[0], columns=matrix.shape[1])
    return assignments, counts, matrix


def attach_current_macrostates(
    joined: pd.DataFrame,
    assignments: pd.DataFrame,
    partition: FrozenPartition,
    require_full_match: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    output = pd.merge(joined, assignments, on=["user_id", "bundle_step_index"], how="left", validate="one_to_one")
    missing = int(output["current_macrostate"].isna().sum())
    if require_full_match and missing:
        raise RuntimeError(f"Stage-1 current macrostate assignments are missing for {missing} joined rows.")
    output["current_macrostate"] = pd.to_numeric(output["current_macrostate"], errors="coerce").fillna(-1).astype(np.int64)
    recomputed = predict_macro_labels(partition, output[["M", "Psi"]].to_numpy(dtype=float))
    valid = output["current_macrostate"].to_numpy(dtype=np.int64) >= 0
    match = float(np.mean(recomputed[valid] == output.loc[valid, "current_macrostate"].to_numpy(dtype=np.int64))) if valid.any() else float("nan")
    return output, {
        "joined_rows_missing_stage1_macrostate": missing,
        "stage1_assignment_match_fraction": match,
        "current_macrostate_source": "frozen Stage-1 fixed K=6 assignment table",
    }


def build_field_bundle(frame: pd.DataFrame) -> Dict[str, FieldStats]:
    user_id = frame["user_id"].to_numpy(dtype=np.int64)
    current_m = frame["M"].to_numpy(dtype=float)
    current_psi = frame["Psi"].to_numpy(dtype=float)
    empirical = field_stats(
        current_m,
        current_psi,
        frame["target_M_next"].to_numpy(dtype=float) - current_m,
        frame["target_Psi_next"].to_numpy(dtype=float) - current_psi,
        user_id,
    )
    mechanism = field_stats(
        current_m,
        current_psi,
        frame["mech_next_M"].to_numpy(dtype=float) - current_m,
        frame["mech_next_Psi"].to_numpy(dtype=float) - current_psi,
        user_id,
    )
    ssl_anchor = field_stats(
        current_m,
        current_psi,
        frame["ssl_next_M"].to_numpy(dtype=float) - current_m,
        frame["ssl_next_Psi"].to_numpy(dtype=float) - current_psi,
        user_id,
    )
    ssl_learned = field_stats(
        frame["ssl_M"].to_numpy(dtype=float),
        frame["ssl_Psi"].to_numpy(dtype=float),
        frame["ssl_next_M"].to_numpy(dtype=float) - frame["ssl_M"].to_numpy(dtype=float),
        frame["ssl_next_Psi"].to_numpy(dtype=float) - frame["ssl_Psi"].to_numpy(dtype=float),
        user_id,
    )
    return {
        "empirical": empirical,
        "mechanism": mechanism,
        "event_ssl_anchor": ssl_anchor,
        "event_ssl_learned": ssl_learned,
    }


def compute_metrics(
    joined: pd.DataFrame,
    partition: FrozenPartition,
    stage1_counts: np.ndarray,
    stage1_matrix: np.ndarray,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    user_id = joined["user_id"].to_numpy(dtype=np.int64)
    row_metrics: List[Dict[str, Any]] = []
    weighted_rows: List[Dict[str, Any]] = []
    pairs = (
        ("mechanism_vs_empirical", "mech_next_M", "mech_next_Psi", "target_M_next", "target_Psi_next"),
        ("event_ssl_vs_empirical", "ssl_next_M", "ssl_next_Psi", "target_M_next", "target_Psi_next"),
        ("mechanism_vs_event_ssl", "mech_next_M", "mech_next_Psi", "ssl_next_M", "ssl_next_Psi"),
    )
    weights = user_balanced_weights(user_id)
    current_m = joined["M"].to_numpy(dtype=float)
    current_psi = joined["Psi"].to_numpy(dtype=float)
    for comparison, a_m_name, a_psi_name, b_m_name, b_psi_name in pairs:
        a_m = joined[a_m_name].to_numpy(dtype=float)
        a_psi = joined[a_psi_name].to_numpy(dtype=float)
        b_m = joined[b_m_name].to_numpy(dtype=float)
        b_psi = joined[b_psi_name].to_numpy(dtype=float)
        row_metrics.extend([
            {"comparison": comparison, "metric": "next_M_corr", "value": pearson(a_m, b_m)},
            {"comparison": comparison, "metric": "next_Psi_corr", "value": pearson(a_psi, b_psi)},
            {"comparison": comparison, "metric": "next_M_rmse", "value": rmse(a_m, b_m)},
            {"comparison": comparison, "metric": "next_Psi_rmse", "value": rmse(a_psi, b_psi)},
            {"comparison": comparison, "metric": "next_M_mae", "value": mae(a_m, b_m)},
            {"comparison": comparison, "metric": "next_Psi_mae", "value": mae(a_psi, b_psi)},
        ])
        for coordinate, first, second in (("M", a_m, b_m), ("Psi", a_psi, b_psi)):
            error = first - second
            weighted_rows.extend([
                {"comparison": comparison, "metric": f"interval_weighted_rmse_{coordinate}", "value": float(np.sqrt(np.mean(error ** 2)))},
                {"comparison": comparison, "metric": f"interval_weighted_mae_{coordinate}", "value": float(np.mean(np.abs(error)))},
                {"comparison": comparison, "metric": f"interval_weighted_bias_{coordinate}", "value": float(np.mean(error))},
                {"comparison": comparison, "metric": f"user_balanced_rmse_{coordinate}", "value": weighted_rmse(error, weights)},
                {"comparison": comparison, "metric": f"user_balanced_mae_{coordinate}", "value": weighted_mae(error, weights)},
                {"comparison": comparison, "metric": f"user_balanced_bias_{coordinate}", "value": weighted_mean(error, weights)},
            ])
        displacement_a = np.column_stack([a_m - current_m, a_psi - current_psi])
        displacement_b = np.column_stack([b_m - current_m, b_psi - current_psi])
        displacement_corr = pearson(displacement_a.ravel(), displacement_b.ravel())
        norm_a = np.linalg.norm(displacement_a, axis=1)
        norm_b = np.linalg.norm(displacement_b, axis=1)
        cosine = np.sum(displacement_a * displacement_b, axis=1) / np.maximum(norm_a * norm_b, EPS)
        valid_cosine = np.isfinite(cosine)
        row_metrics.extend([
            {"comparison": comparison, "metric": "displacement_vector_corr", "value": displacement_corr},
            {"comparison": comparison, "metric": "mean_displacement_cosine", "value": float(np.nanmean(cosine[valid_cosine])) if valid_cosine.any() else np.nan},
        ])
        weighted_rows.extend([
            {"comparison": comparison, "metric": "median_displacement_cosine", "value": float(np.nanmedian(cosine[valid_cosine])) if valid_cosine.any() else np.nan},
            {"comparison": comparison, "metric": "displacement_cosine_q25", "value": float(np.nanquantile(cosine[valid_cosine], 0.25)) if valid_cosine.any() else np.nan},
            {"comparison": comparison, "metric": "displacement_cosine_q75", "value": float(np.nanquantile(cosine[valid_cosine], 0.75)) if valid_cosine.any() else np.nan},
            {"comparison": comparison, "metric": "fraction_positive_displacement_cosine", "value": float(np.nanmean(cosine[valid_cosine] > 0)) if valid_cosine.any() else np.nan},
            {"comparison": comparison, "metric": "mean_displacement_norm_first", "value": float(np.nanmean(norm_a))},
            {"comparison": comparison, "metric": "mean_displacement_norm_second", "value": float(np.nanmean(norm_b))},
        ])
    row_df = pd.DataFrame(row_metrics)
    weighted_df = pd.DataFrame(weighted_rows)

    occupancy = {
        "empirical": occupancy_grid(joined["target_M_next"].to_numpy(dtype=float), joined["target_Psi_next"].to_numpy(dtype=float), user_id),
        "mechanism": occupancy_grid(joined["mech_next_M"].to_numpy(dtype=float), joined["mech_next_Psi"].to_numpy(dtype=float), user_id),
        "event_ssl": occupancy_grid(joined["ssl_next_M"].to_numpy(dtype=float), joined["ssl_next_Psi"].to_numpy(dtype=float), user_id),
    }
    landscape_rows: List[Dict[str, Any]] = []
    for first, second in (("mechanism", "empirical"), ("event_ssl", "empirical"), ("mechanism", "event_ssl")):
        landscape_rows.extend([
            {"comparison": f"{first}_vs_{second}", "metric": "next_occupancy_js", "value": js_divergence(occupancy[first], occupancy[second])},
            {"comparison": f"{first}_vs_{second}", "metric": "next_occupancy_overlap", "value": float(np.minimum(occupancy[first], occupancy[second]).sum())},
            {"comparison": f"{first}_vs_{second}", "metric": "next_occupancy_tv", "value": 0.5 * float(np.abs(occupancy[first] - occupancy[second]).sum())},
            {"comparison": f"{first}_vs_{second}", "metric": "next_occupancy_corr", "value": pearson(occupancy[first].ravel(), occupancy[second].ravel())},
        ])
    landscape_df = pd.DataFrame(landscape_rows)

    fields = build_field_bundle(joined)
    field_rows: List[Dict[str, Any]] = []
    for first, second in (
        ("mechanism", "empirical"),
        ("event_ssl_anchor", "empirical"),
        ("event_ssl_learned", "empirical"),
        ("mechanism", "event_ssl_anchor"),
    ):
        first_field = fields[first]
        second_field = fields[second]
        reference_weight = fields["empirical"].weight if second == "empirical" else first_field.weight
        comparison = f"{first}_vs_{second}"
        field_rows.extend([
            {"comparison": comparison, "metric": "common_drift_cells", "value": float(np.sum(first_field.mask & second_field.mask))},
            {"comparison": comparison, "metric": "drift_vector_corr", "value": vector_corr(first_field, second_field)},
            {"comparison": comparison, "metric": "drift_speed_corr", "value": speed_corr(first_field, second_field)},
            {"comparison": comparison, "metric": "occupancy_weighted_local_drift_cosine", "value": occupancy_weighted_cosine(first_field, second_field, reference_weight)},
            {"comparison": comparison, "metric": "field_rmse", "value": field_rmse(first_field, second_field)},
        ])
    field_df = pd.DataFrame(field_rows)

    common_residual_mask = fields["empirical"].mask & fields["mechanism"].mask & fields["event_ssl_anchor"].mask
    residual_mech_u = fields["mechanism"].u - fields["empirical"].u
    residual_mech_v = fields["mechanism"].v - fields["empirical"].v
    residual_ssl_u = fields["event_ssl_anchor"].u - fields["empirical"].u
    residual_ssl_v = fields["event_ssl_anchor"].v - fields["empirical"].v
    residual_mech = FieldStats(residual_mech_u, residual_mech_v, common_residual_mask, fields["empirical"].count, fields["empirical"].weight)
    residual_ssl = FieldStats(residual_ssl_u, residual_ssl_v, common_residual_mask, fields["empirical"].count, fields["empirical"].weight)
    residual_rows = pd.DataFrame([{
        "comparison": "mechanism_residual_vs_event_ssl_residual",
        "common_drift_cells": int(common_residual_mask.sum()),
        "residual_vector_corr": vector_corr(residual_mech, residual_ssl),
        "residual_speed_corr": speed_corr(residual_mech, residual_ssl),
        "occupancy_weighted_residual_cosine": occupancy_weighted_cosine(residual_mech, residual_ssl, fields["empirical"].weight),
        "residual_field_rmse_between_models": field_rmse(residual_mech, residual_ssl),
        "mechanism_residual_magnitude_mean": float(np.nanmean(np.sqrt(residual_mech_u[common_residual_mask] ** 2 + residual_mech_v[common_residual_mask] ** 2))) if common_residual_mask.any() else np.nan,
        "event_ssl_residual_magnitude_mean": float(np.nanmean(np.sqrt(residual_ssl_u[common_residual_mask] ** 2 + residual_ssl_v[common_residual_mask] ** 2))) if common_residual_mask.any() else np.nan,
    }])

    current = joined["current_macrostate"].to_numpy(dtype=np.int64)
    next_labels = {
        "empirical": predict_macro_labels(partition, joined[["target_M_next", "target_Psi_next"]].to_numpy(dtype=float)),
        "mechanism": predict_macro_labels(partition, joined[["mech_next_M", "mech_next_Psi"]].to_numpy(dtype=float)),
        "event_ssl": predict_macro_labels(partition, joined[["ssl_next_M", "ssl_next_Psi"]].to_numpy(dtype=float)),
    }
    transition_counts: Dict[str, np.ndarray] = {}
    transition_matrices: Dict[str, np.ndarray] = {}
    for label, next_state in next_labels.items():
        counts, matrix = transition_matrix(current, next_state, EXPECTED_MACROSTATE_K)
        transition_counts[label] = counts
        transition_matrices[label] = matrix

    transition_rows: List[Dict[str, Any]] = []
    for first, second in (("empirical", "mechanism"), ("empirical", "event_ssl"), ("mechanism", "event_ssl")):
        prefix = f"{second}_vs_{first}"
        transition_rows.extend([
            {"comparison": prefix, "metric": key.replace(f"{prefix}_", ""), "value": value}
            for key, value in transition_metrics(prefix, transition_matrices[first], transition_matrices[second]).items()
        ])
    transition_df = pd.DataFrame(transition_rows)

    state_rows: List[Dict[str, Any]] = []
    for state in range(EXPECTED_MACROSTATE_K):
        state_rows.append({
            "macrostate": state,
            "empirical_self_transition": float(transition_matrices["empirical"][state, state]),
            "mechanism_self_transition": float(transition_matrices["mechanism"][state, state]),
            "event_ssl_self_transition": float(transition_matrices["event_ssl"][state, state]),
            "mechanism_minus_empirical_self_transition": float(transition_matrices["mechanism"][state, state] - transition_matrices["empirical"][state, state]),
            "event_ssl_minus_empirical_self_transition": float(transition_matrices["event_ssl"][state, state] - transition_matrices["empirical"][state, state]),
            "event_ssl_minus_mechanism_self_transition": float(transition_matrices["event_ssl"][state, state] - transition_matrices["mechanism"][state, state]),
            "mechanism_vs_empirical_row_tv": 0.5 * float(np.abs(transition_matrices["mechanism"][state] - transition_matrices["empirical"][state]).sum()),
            "event_ssl_vs_empirical_row_tv": 0.5 * float(np.abs(transition_matrices["event_ssl"][state] - transition_matrices["empirical"][state]).sum()),
            "event_ssl_vs_mechanism_row_tv": 0.5 * float(np.abs(transition_matrices["event_ssl"][state] - transition_matrices["mechanism"][state]).sum()),
        })
    state_df = pd.DataFrame(state_rows)

    stage1_count_diff = float(np.nanmax(np.abs(transition_counts["empirical"] - stage1_counts)))
    stage1_matrix_diff = float(np.nanmax(np.abs(transition_matrices["empirical"] - stage1_matrix)))
    reconstruction_df = pd.DataFrame([{
        "joined_empirical_transition_count": float(transition_counts["empirical"].sum()),
        "stage1_empirical_transition_count": float(stage1_counts.sum()),
        "transition_count_max_abs_difference": stage1_count_diff,
        "transition_matrix_max_abs_difference": stage1_matrix_diff,
        "exact_full_stage1_reconstruction": bool(stage1_count_diff <= 0 and stage1_matrix_diff <= 1e-12),
    }])

    def lookup(table: pd.DataFrame, comparison: str, metric: str) -> float:
        subset = table[(table["comparison"] == comparison) & (table["metric"] == metric)]
        return float(subset["value"].iloc[0]) if not subset.empty else float("nan")

    scores = [
        ("row_next_M_corr", lookup(row_df, "mechanism_vs_event_ssl", "next_M_corr")),
        ("row_next_Psi_corr", lookup(row_df, "mechanism_vs_event_ssl", "next_Psi_corr")),
        ("row_displacement_vector_corr", lookup(row_df, "mechanism_vs_event_ssl", "displacement_vector_corr")),
        ("landscape_similarity_1minusJS", 1.0 - lookup(landscape_df, "mechanism_vs_event_ssl", "next_occupancy_js")),
        ("field_vector_corr", lookup(field_df, "mechanism_vs_event_ssl_anchor", "drift_vector_corr")),
        ("field_weighted_cosine", lookup(field_df, "mechanism_vs_event_ssl_anchor", "occupancy_weighted_local_drift_cosine")),
        ("transition_similarity_1minusTV", 1.0 - lookup(transition_df, "event_ssl_vs_mechanism", "mean_row_tv")),
        ("transition_self_corr", lookup(transition_df, "event_ssl_vs_mechanism", "self_transition_corr")),
    ]
    score_df = pd.DataFrame([{
        "domain": name,
        "score": float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else np.nan,
        "raw_value": value,
    } for name, value in scores])
    finite_scores = score_df["score"].dropna()
    composite = float(finite_scores.mean()) if not finite_scores.empty else np.nan
    score_df.loc[len(score_df)] = {
        "domain": "mechanistic_surrogate_composite",
        "score": composite,
        "raw_value": composite,
    }

    matrices: Dict[str, np.ndarray] = {
        **{f"H_{name}": value for name, value in occupancy.items()},
        **{f"P_{name}": value for name, value in transition_matrices.items()},
        **{f"C_{name}": value for name, value in transition_counts.items()},
        "field_emp_u": fields["empirical"].u,
        "field_emp_v": fields["empirical"].v,
        "field_emp_mask": fields["empirical"].mask,
        "field_mech_u": fields["mechanism"].u,
        "field_mech_v": fields["mechanism"].v,
        "field_mech_mask": fields["mechanism"].mask,
        "field_ssl_anchor_u": fields["event_ssl_anchor"].u,
        "field_ssl_anchor_v": fields["event_ssl_anchor"].v,
        "field_ssl_anchor_mask": fields["event_ssl_anchor"].mask,
        "field_ssl_learned_u": fields["event_ssl_learned"].u,
        "field_ssl_learned_v": fields["event_ssl_learned"].v,
        "field_ssl_learned_mask": fields["event_ssl_learned"].mask,
        "residual_mech_u": residual_mech_u,
        "residual_mech_v": residual_mech_v,
        "residual_ssl_u": residual_ssl_u,
        "residual_ssl_v": residual_ssl_v,
        "residual_common_mask": common_residual_mask,
    }
    tables = {
        "row_level": row_df,
        "row_level_weighted": weighted_df,
        "landscape": landscape_df,
        "field": field_df,
        "residual_field": residual_rows,
        "transition": transition_df,
        "statewise_transition": state_df,
        "transition_reconstruction_audit": reconstruction_df,
        "decomposition_scores": score_df,
    }
    return tables, matrices


def table_value(tables: Mapping[str, pd.DataFrame], table: str, comparison: str, metric: str) -> float:
    frame = tables.get(table, pd.DataFrame())
    if frame.empty or not {"comparison", "metric", "value"}.issubset(frame.columns):
        return float("nan")
    subset = frame[(frame["comparison"] == comparison) & (frame["metric"] == metric)]
    return float(subset["value"].iloc[0]) if not subset.empty else float("nan")


def score_value(tables: Mapping[str, pd.DataFrame], domain: str) -> float:
    frame = tables.get("decomposition_scores", pd.DataFrame())
    if frame.empty:
        return float("nan")
    subset = frame[frame["domain"] == domain]
    return float(subset["score"].iloc[0]) if not subset.empty else float("nan")


def formal_context_table(contracts: Mapping[str, Any], event_root: Path, phase3_root: Path, split: str, sources: List[SourceRecord]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    phase3_metrics_path = table_path(phase3_root / "tables" / f"phase3_{split}_structural_alignment_metrics")
    phase3_metrics = read_table(phase3_metrics_path)
    record_file("phase3_structural_alignment_metrics", phase3_metrics_path, sources, rows=len(phase3_metrics), columns=len(phase3_metrics.columns))
    event_metrics_path = table_path(event_root / "tables" / "stage4_event_ssl_structural_metrics_all_splits")
    event_metrics = read_table(event_metrics_path)
    record_file("event_ssl_structural_metrics_all_splits", event_metrics_path, sources, rows=len(event_metrics), columns=len(event_metrics.columns))
    event_row = event_metrics[event_metrics["split"].astype(str) == split]
    mechanism_row = phase3_metrics.iloc[0] if not phase3_metrics.empty else pd.Series(dtype=object)
    ssl_row = event_row.iloc[0] if not event_row.empty else pd.Series(dtype=object)

    mechanism_metrics = (
        "one_step_rmse_M",
        "one_step_rmse_Psi",
        "occupancy_js_MR_PsiA",
        "drift_vector_corr_MR_PsiA",
        "drift_local_rmse_loss_MR_PsiA",
        "drift_direction_loss_MR_PsiA",
        "drift_magnitude_loss_MR_PsiA",
        "objective_primary_score",
    )
    event_metrics_names = (
        "coordinate_corr_M",
        "coordinate_corr_Psi",
        "coordinate_rmse_M",
        "coordinate_rmse_Psi",
        "one_step_rmse_M",
        "one_step_rmse_Psi",
        "next_state_occupancy_js",
        "anchor_drift_vector_corr",
        "anchor_occupancy_weighted_local_drift_cosine",
        "learned_plane_drift_vector_corr",
        "learned_plane_occupancy_weighted_local_drift_cosine",
        "learned_plane_transition_mean_row_tv",
        "learned_plane_self_transition_corr",
    )
    for metric in mechanism_metrics:
        if metric in mechanism_row.index:
            rows.append({"model": "minimal_mechanism", "metric": metric, "value": pd.to_numeric(mechanism_row.get(metric), errors="coerce"), "contract": "formal Phase-3 confirmation contract"})
    for metric in event_metrics_names:
        if metric in ssl_row.index:
            rows.append({"model": "predictive_state_event_ssl", "metric": metric, "value": pd.to_numeric(ssl_row.get(metric), errors="coerce"), "contract": "formal Stage-4 evaluation contract"})
    rows.extend([
        {"model": "minimal_mechanism", "metric": "family", "value": contracts["family"], "contract": "frozen Phase-2/Phase-3 manifest"},
        {"model": "predictive_state_event_ssl", "metric": "model_kind", "value": contracts["event_ssl_model_kind"], "contract": "frozen Stage-4 evaluation manifest"},
    ])
    return pd.DataFrame(rows)


def build_metric_ledger(tables: Mapping[str, pd.DataFrame], join_audit: Mapping[str, Any], partition_audit: Mapping[str, Any], split: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add(priority: str, category: str, metric: str, value: Any, estimator: str, weighting: str, support: str, manuscript_use: str) -> None:
        rows.append({
            "priority": priority,
            "category": category,
            "metric": metric,
            "value": value,
            "estimator": estimator,
            "weighting": weighting,
            "support": support,
            "uncertainty_status": "point estimate from frozen models; no cross-model bootstrap interval is inferred",
            "manuscript_use": manuscript_use,
        })

    add(MAIN_REQUIRED, "analysis contract", "confirmation split", split, "frozen output-only comparison", "not applicable", f"{join_audit.get('joined_rows')} joined intervals; {join_audit.get('joined_users')} users", "State the independent confirmation cohort.")
    add(MAIN_REQUIRED, "analysis contract", "join fraction of mechanism rows", join_audit.get("join_fraction_of_mechanism"), "one-to-one key join", "not applicable", "all formal prediction rows", "Report or retain in audit.")
    add(MAIN_REQUIRED, "analysis contract", "join fraction of Event-SSL rows", join_audit.get("join_fraction_of_event_ssl"), "one-to-one key join", "not applicable", "all formal prediction rows", "Report or retain in audit.")
    add(MAIN_REQUIRED, "analysis contract", "fixed mesostate K", partition_audit.get("macrostate_k"), "frozen Stage-1 partition", "A_train user-balanced KMeans fit", "K=6; no refit", "Methods or comparison caption.")

    core = (
        ("next-state M correlation", "row_level", "mechanism_vs_event_ssl", "next_M_corr", "interval-level Pearson correlation", "interval", MAIN_REQUIRED),
        ("next-state Psi correlation", "row_level", "mechanism_vs_event_ssl", "next_Psi_corr", "interval-level Pearson correlation", "interval", MAIN_REQUIRED),
        ("displacement-vector correlation", "row_level", "mechanism_vs_event_ssl", "displacement_vector_corr", "Pearson correlation of flattened two-coordinate displacements", "interval", MAIN_REQUIRED),
        ("mean displacement cosine", "row_level", "mechanism_vs_event_ssl", "mean_displacement_cosine", "mean interval-level vector cosine", "interval", MAIN_RECOMMENDED),
        ("next-state occupancy JS", "landscape", "mechanism_vs_event_ssl", "next_occupancy_js", "Jensen-Shannon divergence", "user-balanced occupancy", MAIN_REQUIRED),
        ("population drift vector correlation", "field", "mechanism_vs_event_ssl_anchor", "drift_vector_corr", "Pearson correlation of flattened supported drift components", "user-balanced field", MAIN_REQUIRED),
        ("population drift speed correlation", "field", "mechanism_vs_event_ssl_anchor", "drift_speed_corr", "Pearson correlation of cellwise drift speeds", "user-balanced field", MAIN_REQUIRED),
        ("occupancy-weighted local drift cosine", "field", "mechanism_vs_event_ssl_anchor", "occupancy_weighted_local_drift_cosine", "local vector cosine averaged with empirical-anchor field weights", "user-balanced field", MAIN_REQUIRED),
        ("common supported drift cells", "field", "mechanism_vs_event_ssl_anchor", "common_drift_cells", "common support count", "minimum 30 intervals per cell", MAIN_REQUIRED),
        ("transition mean row-wise TV", "transition", "event_ssl_vs_mechanism", "mean_row_tv", "mean row-wise total variation", "interval transition counts", MAIN_REQUIRED),
        ("transition max row-wise TV", "transition", "event_ssl_vs_mechanism", "max_row_tv", "maximum row-wise total variation", "interval transition counts", MAIN_RECOMMENDED),
        ("self-transition correlation", "transition", "event_ssl_vs_mechanism", "self_transition_corr", "Pearson correlation across six statewise self-transition probabilities", "six fixed states", MAIN_REQUIRED),
    )
    for label, table, comparison, metric, estimator, weighting, priority in core:
        add(priority, "cross-model agreement", label, table_value(tables, table, comparison, metric), estimator, weighting, "common confirmation rows and frozen K=6 partition", "Main Results unless marked recommended.")
    add(MAIN_RECOMMENDED, "cross-model agreement", "mechanistic surrogate composite", score_value(tables, "mechanistic_surrogate_composite"), "descriptive mean of eight normalized agreement signals", "mixed descriptive scales", "not a training or selection target", "Use only as a descriptive summary.")
    residual = tables.get("residual_field", pd.DataFrame())
    if not residual.empty:
        for metric in ("residual_vector_corr", "residual_speed_corr", "occupancy_weighted_residual_cosine", "residual_field_rmse_between_models"):
            add(SUPPLEMENT_REQUIRED, "residual-field diagnostics", metric, residual.iloc[0].get(metric), "comparison after subtracting the same empirical field", "user-balanced field", "common supported cells", "Additional information; do not substitute for primary field agreement.")
    return pd.DataFrame(rows)


def build_quality_gates(
    contracts: Mapping[str, Any],
    join_audit: Mapping[str, Any],
    assignment_audit: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    max_rows: int,
) -> pd.DataFrame:
    reconstruction = tables["transition_reconstruction_audit"].iloc[0].to_dict()
    checks = [
        ("primary macrostate is exactly M and Psi", tuple(contracts["primary_macrostate"]) == EXPECTED_PRIMARY_MACROSTATE),
        ("minimal mechanism family is offset_dual_channel", contracts["family"] == EXPECTED_FAMILY),
        ("Event-SSL model is predictive_state", contracts["event_ssl_model_kind"] == "predictive_state"),
        ("fixed K=6 contract is verified", contracts["fixed_k6_contract_verified"] is True),
        ("no cross-model fitting", contracts["cross_model_fitting"] is False),
        ("confirmation data were not used for updates", contracts["confirmation_data_used_for_update"] is False),
        ("prediction join covers the mechanism output", float(join_audit["join_fraction_of_mechanism"]) >= float(join_audit["minimum_required_join_fraction"])),
        ("prediction join covers the Event-SSL output", float(join_audit["join_fraction_of_event_ssl"]) >= float(join_audit["minimum_required_join_fraction"])),
        ("empirical anchors agree within tolerance", max(float(join_audit[key]) for key in join_audit if key.startswith("max_abs_difference_") and key.endswith("_between_sources")) <= float(join_audit["anchor_tolerance"])),
        ("Stage-1 current labels match frozen-centre assignment", float(assignment_audit["stage1_assignment_match_fraction"]) >= 0.999),
        ("publication run uses all joined rows", int(max_rows) == 0),
        ("Stage-1 transition reconstruction audit is available", np.isfinite(float(reconstruction.get("transition_matrix_max_abs_difference", np.nan)))),
    ]
    return pd.DataFrame([{"quality_gate": name, "passed": bool(passed)} for name, passed in checks])


def markdown_table(frame: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if frame is None or frame.empty:
        return "_No rows available._"
    output = frame.head(max_rows).copy() if max_rows is not None else frame.copy()
    text = output.to_markdown(index=False)
    if max_rows is not None and len(frame) > max_rows:
        text += f"\n\n_Table truncated to the first {max_rows} rows._"
    return text


def make_markdown_report(
    tables: Mapping[str, pd.DataFrame],
    join_audit: Mapping[str, Any],
    partition_audit: Mapping[str, Any],
    assignment_audit: Mapping[str, Any],
    output_root: Path,
) -> Path:
    path = output_root / "reports" / "mechanism_event_ssl_macro_closure_comparison.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Minimal mechanism versus predictive-state Event-SSL macro-closure comparison\n",
        "This report compares the frozen offset dual-channel mechanism with predictive-state Event-SSL at the common `(M, Psi)` macrostate level. It does not retrain either model, fit either model to the other, redefine macrostates, or use confirmation data for model updates.\n",
        "## Join and partition audit\n",
        markdown_table(pd.DataFrame([{**join_audit, **assignment_audit}])),
        "\n",
        markdown_table(pd.DataFrame([partition_audit])),
        "\n",
        "## Key mechanistic-surrogate signals\n",
    ]
    key_rows = [
        ("next-state M correlation", table_value(tables, "row_level", "mechanism_vs_event_ssl", "next_M_corr")),
        ("next-state Psi correlation", table_value(tables, "row_level", "mechanism_vs_event_ssl", "next_Psi_corr")),
        ("displacement-vector correlation", table_value(tables, "row_level", "mechanism_vs_event_ssl", "displacement_vector_corr")),
        ("mean interval displacement cosine", table_value(tables, "row_level", "mechanism_vs_event_ssl", "mean_displacement_cosine")),
        ("mechanism-EventSSL occupancy JS", table_value(tables, "landscape", "mechanism_vs_event_ssl", "next_occupancy_js")),
        ("mechanism-EventSSL field vector correlation", table_value(tables, "field", "mechanism_vs_event_ssl_anchor", "drift_vector_corr")),
        ("mechanism-EventSSL field speed correlation", table_value(tables, "field", "mechanism_vs_event_ssl_anchor", "drift_speed_corr")),
        ("mechanism-EventSSL weighted local drift cosine", table_value(tables, "field", "mechanism_vs_event_ssl_anchor", "occupancy_weighted_local_drift_cosine")),
        ("mechanism-EventSSL transition mean row-TV", table_value(tables, "transition", "event_ssl_vs_mechanism", "mean_row_tv")),
        ("mechanism-EventSSL transition max row-TV", table_value(tables, "transition", "event_ssl_vs_mechanism", "max_row_tv")),
        ("mechanism-EventSSL self-transition correlation", table_value(tables, "transition", "event_ssl_vs_mechanism", "self_transition_corr")),
        ("mechanistic surrogate composite", score_value(tables, "mechanistic_surrogate_composite")),
    ]
    lines.extend([
        markdown_table(pd.DataFrame(key_rows, columns=["signal", "value"])),
        "\n",
        "## Numerical rigor and interpretation boundary\n",
        "The cross-model values are point estimates from two independently frozen models evaluated on exactly joined confirmation intervals. Row-level errors are reported with both interval and user-balanced weighting. Population fields use user-balanced conditional means on a common empirical current-state grid, and transition comparisons use the frozen Stage-1 K=6 partition. No cross-model bootstrap interval is inferred by this report. Strong population-field agreement does not imply interval-level identity, complete transition-operator equivalence, or mechanistic sufficiency of the full Event-SSL hidden state.\n",
        "## Publication metric ledger\n",
        markdown_table(tables["publication_metric_ledger"]),
        "\n",
        "## Scientific quality gates\n",
        markdown_table(tables["scientific_quality_gates"]),
        "\n",
        "## Formal single-model context\n",
        markdown_table(tables["formal_context"]),
        "\n",
        "## Residual-field diagnostics\n",
        markdown_table(tables["residual_field"]),
        "\n",
        "## Statewise transition diagnostics\n",
        markdown_table(tables["statewise_transition"]),
        "\n",
    ])
    for name in ("row_level", "row_level_weighted", "landscape", "field", "transition", "decomposition_scores", "transition_reconstruction_audit"):
        lines.extend([f"## {name.replace('_', ' ').title()}\n", markdown_table(tables[name]), "\n"])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen minimal-mechanism versus predictive-state Event-SSL publication statistics.")
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--phase2-root", type=Path, default=DEFAULT_PHASE2_ROOT)
    parser.add_argument("--phase3-root", type=Path, default=DEFAULT_PHASE3_ROOT)
    parser.add_argument("--event-ssl-eval-root", type=Path, default=DEFAULT_EVENT_SSL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional deterministic post-join subsample; use 0 for publication outputs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchor-tolerance", type=float, default=1e-6)
    parser.add_argument("--minimum-join-fraction", type=float, default=0.999999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    report_root = output_root / "reports"
    for directory in (table_root, metadata_root, report_root):
        directory.mkdir(parents=True, exist_ok=True)

    sources: List[SourceRecord] = []
    record_file("collector_script", Path(__file__).resolve(), sources)
    partition = load_fixed_partition(args.stage1_root, sources)
    contracts = validate_contracts(
        args.stage1_root,
        args.phase2_root,
        args.phase3_root,
        args.event_ssl_eval_root,
        args.split,
        partition,
        sources,
    )
    phase3_manifest = contracts["phase3_manifest"]
    event_manifest = contracts["event_ssl_manifest"]
    mechanism = load_minimal_predictions(args.phase3_root, phase3_manifest, args.split, sources)
    event_ssl = load_event_ssl_predictions(args.event_ssl_eval_root, event_manifest, args.split, sources)
    joined, join_audit = align_predictions(mechanism, event_ssl, args.anchor_tolerance, args.minimum_join_fraction)

    if args.max_rows > 0 and len(joined) > args.max_rows:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(len(joined), size=int(args.max_rows), replace=False))
        joined = joined.iloc[indices].reset_index(drop=True)
    join_audit["analysis_rows_after_subsample"] = int(len(joined))
    join_audit["analysis_users_after_subsample"] = int(joined["user_id"].nunique())
    join_audit["max_rows_argument"] = int(args.max_rows)

    assignments, stage1_counts, stage1_matrix = load_stage1_assignments(args.stage1_root, args.split, sources)
    joined, assignment_audit = attach_current_macrostates(joined, assignments, partition, require_full_match=args.max_rows == 0)
    tables, matrices = compute_metrics(joined, partition, stage1_counts, stage1_matrix)
    tables["formal_context"] = formal_context_table(contracts, args.event_ssl_eval_root, args.phase3_root, args.split, sources)
    tables["publication_metric_ledger"] = build_metric_ledger(tables, join_audit, partition.audit, args.split)
    tables["scientific_quality_gates"] = build_quality_gates(contracts, join_audit, assignment_audit, tables, args.max_rows)

    if not bool(tables["scientific_quality_gates"]["passed"].all()):
        failed = tables["scientific_quality_gates"].loc[~tables["scientific_quality_gates"]["passed"], "quality_gate"].tolist()
        raise RuntimeError("Scientific quality gates failed: " + "; ".join(failed))

    output_paths: Dict[str, str] = {}
    output_paths["join_audit"] = str(write_table(pd.DataFrame([{**join_audit, **assignment_audit}]), table_root / "join_audit"))
    output_paths["macro_partition_audit"] = str(write_table(pd.DataFrame([partition.audit]), table_root / "macro_partition_audit"))
    output_paths["macro_partition_centers"] = str(write_table(partition.centers_table, table_root / "macro_partition_centers"))
    for name, frame in tables.items():
        output_paths[name] = str(write_table(frame, table_root / f"mechanism_event_ssl_{name}"))
    matrix_path = table_root / "mechanism_event_ssl_matrices.npz"
    np.savez_compressed(matrix_path, **matrices)
    output_paths["matrices"] = str(matrix_path)

    source_df = source_table(sources)
    source_path = write_table(source_df, table_root / "mechanism_event_ssl_source_audit")
    output_paths["source_audit"] = str(source_path)
    report_path = make_markdown_report(tables, join_audit, partition.audit, assignment_audit, output_root)
    output_paths["report"] = str(report_path)

    manifest = {
        "script": Path(__file__).name,
        "created_at": now_iso(),
        "stage1_root": str(args.stage1_root.resolve()),
        "phase2_root": str(args.phase2_root.resolve()),
        "phase3_root": str(args.phase3_root.resolve()),
        "event_ssl_eval_root": str(args.event_ssl_eval_root.resolve()),
        "output_root": str(output_root),
        "split": args.split,
        "primary_macrostate": list(EXPECTED_PRIMARY_MACROSTATE),
        "minimal_mechanism_family": EXPECTED_FAMILY,
        "event_ssl_model_kind": "predictive_state",
        "fixed_k6_partition": partition.audit,
        "analysis_boundary": {
            "model_retraining": False,
            "cross_model_fitting": False,
            "mechanism_refit_to_event_ssl": False,
            "event_ssl_refit_to_mechanism": False,
            "macrostate_redefinition": False,
            "kmeans_refit": False,
            "macrostate_k_selected": False,
            "confirmation_data_used_for_update": False,
            "visualization_output": False,
        },
        "join_audit": join_audit,
        "assignment_audit": assignment_audit,
        "outputs": output_paths,
        "source_audit": str(source_path),
    }
    manifest_path = metadata_root / "mechanism_event_ssl_comparison_manifest.json"
    save_json(manifest, manifest_path)
    print(f"[cross-stage report] wrote {report_path}")
    print(f"[cross-stage report] wrote {manifest_path}")


if __name__ == "__main__":
    main()
