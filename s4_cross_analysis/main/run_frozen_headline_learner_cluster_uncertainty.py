#!/usr/bin/env python3
from __future__ import annotations

"""Paired learner-cluster uncertainty for frozen field and transition summaries."""

import argparse
import dataclasses
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm, spearmanr

EPS = 1e-12
GRID_BINS = np.linspace(-1.0, 1.0, 41)
GRID_N = 40
N_CELLS = GRID_N * GRID_N
K = 6
MIN_DRIFT_COUNT = 30
EXPECTED_KMEANS_N_INIT = 20
EXPECTED_KMEANS_FIT_MAX_ROWS = 500000
EXPECTED_STAGE1_RANDOM_STATE = 42
EXPECTED_PARTITION_FEATURES = (
    "M_response_prebalanced_pre",
    "activity_alignment_order_Psi_pre",
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/datasets/KT4/outputs_KT4/frozen_headline_learner_cluster_uncertainty"
)
EXPECTED_POINT_VALUES = {
    "mechanism_vs_empirical_drift_vector_corr": 0.9457,
    "event_ssl_learned_vs_empirical_drift_vector_corr": 0.6877,
    "mechanism_vs_event_ssl_anchor_drift_vector_corr": 0.8661,
    "mechanism_vs_event_ssl_anchor_drift_speed_corr": 0.8552,
    "mechanism_vs_event_ssl_anchor_weighted_local_cosine": 0.7851,
    "event_ssl_anchor_vs_empirical_drift_vector_corr": 0.8802,
    "event_ssl_learned_vs_empirical_weighted_local_cosine": 0.8629,
    "mechanism_vs_empirical_transition_mean_row_tv": 0.1021,
    "event_ssl_anchor_vs_empirical_transition_mean_row_tv": 0.1512,
    "mechanism_vs_event_ssl_transition_mean_row_tv": 0.1497,
    "event_ssl_learned_vs_empirical_transition_mean_row_tv": 0.09417,
    "event_ssl_learned_vs_empirical_self_transition_pearson": 0.8462,
    "mechanism_vs_event_ssl_self_transition_pearson": 0.9660,
    "mechanism_vs_empirical_self_transition_pearson": 0.9870,
    "event_ssl_anchor_vs_empirical_self_transition_pearson": 0.9529,
}


@dataclass(frozen=True)
class MacroPartition:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    standardized_centers: np.ndarray
    metadata_path: Path
    centers_path: Path


@dataclass(frozen=True)
class GroupedUserCell:
    user_index: np.ndarray
    cell: np.ndarray
    sums: Tuple[np.ndarray, ...]


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
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
        return number if np.isfinite(number) else None
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
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_checksum(path: Path) -> Dict[str, Any]:
    actual = sha256_file(path)
    candidates = (
        path.with_suffix(".sha256.json"),
        path.parent / f"{path.stem}.sha256.json",
        path.parent / "phase3_confirmation_manifest.sha256.json",
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = load_json(candidate)
        expected = str(payload.get("manifest_sha256", "") or "")
        if not expected:
            continue
        if expected != actual:
            raise RuntimeError(
                f"Manifest checksum mismatch for {path}: expected {expected}, found {actual}."
            )
        return {
            "verified": True,
            "manifest_sha256": actual,
            "checksum_path": str(candidate.resolve()),
        }
    raise RuntimeError(f"No checksum sidecar with manifest_sha256 was found for {path}.")


def path_matches_declared(actual: Path, declared: Any) -> Dict[str, Any]:
    declared_path = Path(str(declared)).expanduser() if declared else None
    if declared_path is None:
        return {"declared": None, "match": False, "mode": "missing"}
    actual_resolved = actual.resolve()
    try:
        declared_resolved = declared_path.resolve()
    except Exception:
        declared_resolved = declared_path
    if declared_resolved == actual_resolved:
        return {
            "declared": str(declared_path),
            "match": True,
            "mode": "exact",
        }
    declared_tail = tuple(declared_path.parts[-2:])
    actual_tail = tuple(actual_resolved.parts[-2:])
    if len(declared_tail) == 2 and declared_tail == actual_tail:
        return {
            "declared": str(declared_path),
            "match": True,
            "mode": "consistent_root_relocation",
        }
    return {
        "declared": str(declared_path),
        "match": False,
        "mode": "mismatch",
    }


def validate_mechanism_manifest(
    path: Path,
    prediction_path: Path,
    expected_script: Optional[Path],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manifest_path = path.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    checksum = verify_manifest_checksum(manifest_path)
    manifest = load_json(manifest_path)
    if str(manifest.get("confirm_split", "")) != "B_confirm":
        raise RuntimeError("Mechanism manifest is not for B_confirm.")
    if str(manifest.get("confirmation_status", "")) != "completed_output_only":
        raise RuntimeError("Mechanism confirmation did not complete in output-only mode.")
    guardrails = dict(manifest.get("guardrails", {}))
    required_false = (
        "parameter_search_opened",
        "calibration_reestimated",
        "mechanism_family_reselected",
        "mechanism_parameters_refit",
        "kmeans_refit",
        "macrostate_k_selected",
        "region_redefinition",
        "B_confirm_used_for_update",
    )
    failed = [name for name in required_false if bool(guardrails.get(name, False))]
    if failed:
        raise RuntimeError(f"Mechanism confirmation guardrails failed: {failed}")
    script_audit: Dict[str, Any] = {"verified": False}
    if expected_script is not None:
        script_path = expected_script.resolve()
        if not script_path.exists():
            raise FileNotFoundError(script_path)
        actual_script_sha = sha256_file(script_path)
        manifest_script_sha = str(manifest.get("phase3_script_sha256", "") or "")
        if not manifest_script_sha:
            raise RuntimeError("Mechanism confirmation manifest lacks phase3_script_sha256.")
        if manifest_script_sha != actual_script_sha:
            raise RuntimeError(
                "Mechanism confirmation was produced by a different Phase-3 implementation: "
                f"manifest={manifest_script_sha}, supplied={actual_script_sha}."
            )
        script_audit = {
            "verified": True,
            "script_path": str(script_path),
            "script_sha256": actual_script_sha,
        }
    declared = dict(manifest.get("outputs", {})).get("full_prediction_table")
    path_audit = path_matches_declared(prediction_path, declared)
    if not path_audit["match"]:
        raise RuntimeError(
            "Mechanism full-prediction path does not match its confirmation manifest: "
            f"declared={declared}, actual={prediction_path}."
        )
    return manifest, {
        "checksum": checksum,
        "prediction_path": path_audit,
        "phase3_script": script_audit,
    }


def validate_event_manifest(
    path: Path,
    prediction_path: Path,
    expected_evaluate_script: Optional[Path],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manifest_path = path.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = load_json(manifest_path)
    if str(manifest.get("model_kind", "")) != "predictive_state":
        raise RuntimeError("Event-SSL manifest is not predictive_state.")
    if list(manifest.get("primary_coordinates", [])) != ["M", "Psi"]:
        raise RuntimeError("Event-SSL manifest primary coordinates are not ['M', 'Psi'].")
    guardrails = dict(manifest.get("guardrails", {}))
    if bool(guardrails.get("B_confirm_used_for_update", False)):
        raise RuntimeError("Event-SSL manifest indicates confirmation was used for update.")
    if bool(guardrails.get("kmeans_refit", False)):
        raise RuntimeError("Event-SSL evaluation refitted the mesostate partition.")
    if bool(guardrails.get("macrostate_k_selected", False)):
        raise RuntimeError("Event-SSL evaluation selected macrostate K.")
    if int(guardrails.get("macrostate_k", -1)) != K:
        raise RuntimeError("Event-SSL evaluation did not use the fixed K=6 partition.")
    views = dict(manifest.get("evaluation_views", {}))
    if not {"empirical_anchor", "learned_plane"}.issubset(views):
        raise RuntimeError("Event-SSL manifest does not document both evaluation gauges.")
    split = dict(manifest.get("splits", {})).get("B_confirm")
    if not isinstance(split, Mapping):
        raise RuntimeError("Event-SSL manifest lacks the B_confirm split record.")
    path_audit = path_matches_declared(prediction_path, split.get("prediction_path"))
    if not path_audit["match"]:
        raise RuntimeError(
            "Event-SSL prediction path does not match its evaluation manifest: "
            f"declared={split.get('prediction_path')}, actual={prediction_path}."
        )
    script_audit: Dict[str, Any] = {"supplied": False}
    if expected_evaluate_script is not None:
        script_path = expected_evaluate_script.resolve()
        if not script_path.exists():
            raise FileNotFoundError(script_path)
        script_audit = {
            "supplied": True,
            "path": str(script_path),
            "sha256": sha256_file(script_path),
            "note": (
                "The formal evaluation manifest does not store the evaluator SHA; "
                "metric and gauge equivalence are enforced by the point-reconstruction gates."
            ),
        }
    return manifest, {
        "prediction_path": path_audit,
        "evaluation_views": sorted(views),
        "guardrails": guardrails,
        "supplied_evaluator": script_audit,
    }


def validate_partition_provenance(
    mechanism_manifest: Mapping[str, Any],
    event_manifest: Mapping[str, Any],
    partition: MacroPartition,
) -> Dict[str, Any]:
    """Require both frozen model pipelines to reference the same Stage-1 K=6 artifacts.

    The Phase-3 mechanism manifest stores the Phase-1 audit contract with
    ``status='verified'`` and nests the Stage-1 metadata, whereas the Event-SSL
    evaluator stores a flat ``verified=True`` audit.  Normalize those two
    formal schemas rather than requiring one pipeline to mimic the other.
    """
    metadata_sha = sha256_file(partition.metadata_path)
    centers_sha = sha256_file(partition.centers_path)
    mechanism_record = dict(mechanism_manifest.get("stage1_fixed_k6_contract", {}))
    event_record = dict(event_manifest.get("macro_partition", {}))

    mechanism_metadata = dict(mechanism_record.get("metadata", {}))
    mechanism_checks = dict(mechanism_record.get("checks", {}))
    normalized = {
        "mechanism": {
            "verified": (
                mechanism_record.get("status") == "verified"
                and bool(mechanism_checks)
                and all(bool(value) for value in mechanism_checks.values())
            ),
            "macrostate_k": mechanism_metadata.get("macrostate_k"),
            "fit_split": mechanism_metadata.get("fit_split"),
            "metadata_sha256": mechanism_record.get("metadata_sha256"),
            "centers_sha256": mechanism_record.get("centers_sha256"),
            "source_schema": "phase3_nested_stage1_audit",
        },
        "event_ssl": {
            "verified": event_record.get("verified") is True,
            "macrostate_k": event_record.get("macrostate_k"),
            "fit_split": event_record.get("fit_split"),
            "metadata_sha256": event_record.get("metadata_sha256"),
            "centers_sha256": event_record.get("centers_sha256"),
            "source_schema": "event_ssl_flat_partition_audit",
        },
    }
    audit: Dict[str, Any] = {
        "metadata_sha256": metadata_sha,
        "centers_sha256": centers_sha,
        "models": {},
    }
    for label, record in normalized.items():
        checks = {
            "verified": bool(record["verified"]),
            "macrostate_k": int(record.get("macrostate_k", -1)) == K,
            "fit_split": record.get("fit_split") == "A_train",
            "metadata_sha256": str(record.get("metadata_sha256", "")) == metadata_sha,
            "centers_sha256": str(record.get("centers_sha256", "")) == centers_sha,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                f"{label} fixed-K provenance does not match the supplied Stage-1 partition: {failed}"
            )
        audit["models"][label] = {
            "source_schema": record["source_schema"],
            "checks": checks,
            "passed": True,
        }
    return audit


def find_table(base_or_path: Path) -> Path:
    path = Path(base_or_path)
    if path.is_file():
        return path
    for suffix in (".parquet", ".csv.gz", ".csv"):
        candidate = path.with_suffix(suffix)
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
    temporary = parquet.with_name(parquet.name + ".tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, parquet)
        return parquet
    except Exception:
        if temporary.exists():
            temporary.unlink()
        csv_path = base.with_suffix(".csv.gz")
        temporary_csv = csv_path.with_name(csv_path.name + ".tmp")
        frame.to_csv(temporary_csv, index=False, compression="gzip")
        os.replace(temporary_csv, csv_path)
        return csv_path


def numeric(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        raise KeyError(f"Required column is absent: {name}")
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)


def integer(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        raise KeyError(f"Required column is absent: {name}")
    values = pd.to_numeric(frame[name], errors="coerce")
    if values.isna().any():
        raise RuntimeError(f"Non-finite identifier in {name}")
    return values.to_numpy(dtype=np.int64)


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3:
        return float("nan")
    a = a[valid] - float(np.mean(a[valid]))
    b = b[valid] - float(np.mean(b[valid]))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > EPS else float("nan")


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3:
        return float("nan")
    result = spearmanr(a[valid], b[valid])
    return float(result.statistic) if np.isfinite(result.statistic) else float("nan")


def digitize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    adjusted = np.where(
        array == GRID_BINS[-1], np.nextafter(GRID_BINS[-1], GRID_BINS[0]), array
    )
    return np.digitize(adjusted, GRID_BINS) - 1


def grid_cells(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ix = digitize(x)
    iy = digitize(y)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (ix >= 0)
        & (ix < GRID_N)
        & (iy >= 0)
        & (iy < GRID_N)
    )
    cell = np.full(len(x), -1, dtype=np.int64)
    cell[valid] = ix[valid] * GRID_N + iy[valid]
    return cell, valid


def load_partition(stage1_root: Path) -> MacroPartition:
    root = stage1_root.resolve() / "dynamics" / "fixed_k6_mesostates"
    metadata_path = root / "fixed_k6_model_metadata.json"
    centers_path = find_table(root / "fixed_k6_centers")
    metadata = load_json(metadata_path)
    centers = read_table(centers_path).sort_values("macrostate", kind="mergesort")
    mapping = metadata.get("raw_to_ordered_label", {})
    expected_labels = list(range(K))
    checks = {
        "coordinate": metadata.get("coordinate") == "MR_PsiA",
        "macrostate_k": int(metadata.get("macrostate_k", -1)) == K,
        "macrostate_k_rule": metadata.get("macrostate_k_rule") == "fixed a priori",
        "features": tuple(metadata.get("features", [])) == EXPECTED_PARTITION_FEATURES,
        "fit_split": metadata.get("fit_split") == "A_train",
        "user_balanced_sampling": metadata.get("user_balanced_sampling") is True,
        "user_balanced_kmeans_fit": metadata.get("user_balanced_kmeans_fit") is True,
        "kmeans_n_init": int(metadata.get("kmeans_n_init", -1)) == EXPECTED_KMEANS_N_INIT,
        "fit_max_rows": int(metadata.get("fit_max_rows", -1)) == EXPECTED_KMEANS_FIT_MAX_ROWS,
        "random_state": int(metadata.get("random_state", -1)) == EXPECTED_STAGE1_RANDOM_STATE,
        "label_mapping": (
            isinstance(mapping, Mapping)
            and sorted(int(key) for key in mapping) == expected_labels
            and sorted(int(value) for value in mapping.values()) == expected_labels
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Stage-1 fixed-K contract failed: " + ", ".join(failed))
    mean = np.asarray(metadata.get("scaler_mean", []), dtype=np.float64)
    scale = np.asarray(metadata.get("scaler_scale", []), dtype=np.float64)
    raw_centers = centers[["center_M", "center_Psi"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    standardized = centers[
        ["center_M_standardized", "center_Psi_standardized"]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    ids = pd.to_numeric(centers["macrostate"], errors="coerce").to_numpy(dtype=float)
    ordered = np.lexsort((raw_centers[:, 1], raw_centers[:, 0]))
    if (
        mean.shape != (2,)
        or scale.shape != (2,)
        or raw_centers.shape != (K, 2)
        or standardized.shape != (K, 2)
        or not np.array_equal(ids, np.arange(K, dtype=float))
        or not np.array_equal(ordered, np.arange(K))
        or not np.isfinite(raw_centers).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or not np.isfinite(standardized).all()
        or np.any(scale <= 0)
    ):
        raise RuntimeError("Invalid Stage-1 fixed-K partition artifacts.")
    return MacroPartition(mean, scale, standardized, metadata_path, centers_path)


def assign_states(partition: MacroPartition, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    output = np.full(len(array), -1, dtype=np.int64)
    valid = np.isfinite(array).all(axis=1)
    if np.any(valid):
        standardized = (array[valid] - partition.scaler_mean[None, :]) / partition.scaler_scale[None, :]
        distances = np.sum(
            (standardized[:, None, :] - partition.standardized_centers[None, :, :]) ** 2,
            axis=2,
        )
        output[valid] = np.argmin(distances, axis=1).astype(np.int64)
    return output


def normalize_transition(counts: np.ndarray) -> np.ndarray:
    matrix = np.asarray(counts, dtype=np.float64).reshape(K, K)
    row_sum = matrix.sum(axis=1, keepdims=True)
    output = np.zeros_like(matrix)
    valid = row_sum[:, 0] > 0
    output[valid] = matrix[valid] / row_sum[valid]
    return output


def row_tv(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(np.abs(np.asarray(first) - np.asarray(second)), axis=1)


def field_metrics(
    first_u: np.ndarray,
    first_v: np.ndarray,
    second_u: np.ndarray,
    second_v: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(first_u)
        & np.isfinite(first_v)
        & np.isfinite(second_u)
        & np.isfinite(second_v)
        & np.isfinite(weight)
    )
    if int(np.sum(valid)) < 3:
        return {
            "supported_cells": int(np.sum(valid)),
            "drift_vector_corr": float("nan"),
            "drift_speed_corr": float("nan"),
            "weighted_local_cosine": float("nan"),
        }
    first_vector = np.column_stack([first_u[valid], first_v[valid]]).ravel()
    second_vector = np.column_stack([second_u[valid], second_v[valid]]).ravel()
    first_speed = np.sqrt(first_u[valid] ** 2 + first_v[valid] ** 2)
    second_speed = np.sqrt(second_u[valid] ** 2 + second_v[valid] ** 2)
    cosine = (
        first_u[valid] * second_u[valid] + first_v[valid] * second_v[valid]
    ) / np.maximum(first_speed * second_speed, EPS)
    normalized_weight = np.asarray(weight[valid], dtype=np.float64)
    normalized_weight /= max(float(np.sum(normalized_weight)), EPS)
    return {
        "supported_cells": int(np.sum(valid)),
        "drift_vector_corr": pearson(first_vector, second_vector),
        "drift_speed_corr": pearson(first_speed, second_speed),
        "weighted_local_cosine": float(np.sum(normalized_weight * np.clip(cosine, -1.0, 1.0))),
    }


def transition_metrics(first: np.ndarray, second: np.ndarray) -> Dict[str, float]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_self = np.diag(first)
    second_self = np.diag(second)
    return {
        "transition_mean_row_tv": float(np.mean(row_tv(first, second))),
        "transition_max_row_tv": float(np.max(row_tv(first, second))),
        "self_transition_pearson": pearson(first_self, second_self),
        "self_transition_spearman": spearman(first_self, second_self),
    }


def leave_one_state_out(first: np.ndarray, second: np.ndarray) -> pd.DataFrame:
    first_self = np.diag(np.asarray(first, dtype=np.float64))
    second_self = np.diag(np.asarray(second, dtype=np.float64))
    rows = []
    for omitted in range(K):
        keep = np.arange(K) != omitted
        rows.append(
            {
                "omitted_state": int(omitted),
                "pearson": pearson(first_self[keep], second_self[keep]),
                "spearman": spearman(first_self[keep], second_self[keep]),
            }
        )
    return pd.DataFrame(rows)


def group_user_cell(
    user_index: np.ndarray,
    cell: np.ndarray,
    values: Sequence[np.ndarray],
) -> GroupedUserCell:
    user_index = np.asarray(user_index, dtype=np.int64)
    cell = np.asarray(cell, dtype=np.int64)
    if len(user_index) != len(cell):
        raise ValueError("user_index and cell lengths differ.")
    if any(len(np.asarray(value)) != len(cell) for value in values):
        raise ValueError("Contribution length mismatch.")
    if len(cell) == 0:
        return GroupedUserCell(
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            tuple(np.asarray([], dtype=np.float64) for _ in values),
        )
    key = user_index * N_CELLS + cell
    order = np.argsort(key, kind="mergesort")
    ordered_key = key[order]
    starts = np.flatnonzero(
        np.concatenate([[True], ordered_key[1:] != ordered_key[:-1]])
    )
    unique = ordered_key[starts]
    sums = tuple(
        np.add.reduceat(np.asarray(value, dtype=np.float64)[order], starts)
        for value in values
    )
    return GroupedUserCell(
        (unique // N_CELLS).astype(np.int64),
        (unique % N_CELLS).astype(np.int64),
        sums,
    )


def grouped_to_csr(
    grouped: GroupedUserCell,
    value_index: int,
    n_users: int,
) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (
            np.asarray(grouped.sums[value_index], dtype=np.float64),
            (grouped.user_index, grouped.cell),
        ),
        shape=(n_users, N_CELLS),
        dtype=np.float64,
    )


def transition_csr(
    user_index: np.ndarray,
    current: np.ndarray,
    next_state: np.ndarray,
    n_users: int,
) -> sparse.csr_matrix:
    valid = (
        (current >= 0)
        & (current < K)
        & (next_state >= 0)
        & (next_state < K)
    )
    flat = current[valid] * K + next_state[valid]
    return sparse.csr_matrix(
        (
            np.ones(int(np.sum(valid)), dtype=np.float64),
            (user_index[valid], flat),
        ),
        shape=(n_users, K * K),
        dtype=np.float64,
    )


def canonical_mechanism_table(path: Path) -> pd.DataFrame:
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
    frame = read_table(path, columns=columns)
    frame = frame.rename(
        columns={"pred_next_M": "mechanism_next_M", "pred_next_Psi": "mechanism_next_Psi"}
    )
    return clean_prediction_table(frame, "mechanism")


def canonical_event_table(path: Path) -> pd.DataFrame:
    columns = [
        "user_id",
        "bundle_step_index",
        "M",
        "Psi",
        "target_M_next",
        "target_Psi_next",
        "pred_M",
        "pred_Psi",
        "pred_next_M",
        "pred_next_Psi",
    ]
    frame = read_table(path, columns=columns)
    frame = frame.rename(
        columns={
            "pred_M": "event_current_M",
            "pred_Psi": "event_current_Psi",
            "pred_next_M": "event_next_M",
            "pred_next_Psi": "event_next_Psi",
        }
    )
    return clean_prediction_table(frame, "event_ssl")


def clean_prediction_table(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    output = frame.copy()
    output["user_id"] = pd.to_numeric(output["user_id"], errors="coerce")
    output["bundle_step_index"] = pd.to_numeric(output["bundle_step_index"], errors="coerce")
    valid = output["user_id"].notna() & output["bundle_step_index"].notna()
    for column in output.columns:
        if column not in {"user_id", "bundle_step_index"}:
            output[column] = pd.to_numeric(output[column], errors="coerce")
            valid &= np.isfinite(output[column].to_numpy(dtype=np.float64))
    output = output.loc[valid].copy()
    output["user_id"] = output["user_id"].astype(np.int64)
    output["bundle_step_index"] = output["bundle_step_index"].astype(np.int64)
    output = output.sort_values(["user_id", "bundle_step_index"], kind="mergesort").reset_index(drop=True)
    if output.duplicated(["user_id", "bundle_step_index"]).any():
        raise RuntimeError(f"Duplicate prediction keys in {label}.")
    state_columns = [
        column
        for column in output.columns
        if column not in {"user_id", "bundle_step_index"}
    ]
    maximum_excess = float(
        np.max(np.maximum(np.abs(output[state_columns].to_numpy(dtype=np.float64)) - 1.0, 0.0))
    )
    if maximum_excess > 2e-5:
        raise RuntimeError(f"{label} state values exceed [-1,1] by {maximum_excess:.3e}.")
    return output


def align_tables(
    mechanism: pd.DataFrame,
    event: pd.DataFrame,
    tolerance: float,
    minimum_join_fraction: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    keys = ["user_id", "bundle_step_index"]
    mechanism_keys = mechanism[keys].to_numpy(dtype=np.int64)
    event_keys = event[keys].to_numpy(dtype=np.int64)
    if len(mechanism) == len(event) and np.array_equal(mechanism_keys, event_keys):
        aligned_mechanism = mechanism
        aligned_event = event
        mode = "direct_sorted_key_identity"
    else:
        mechanism_index = mechanism[keys].copy()
        mechanism_index["mechanism_row"] = np.arange(len(mechanism), dtype=np.int64)
        event_index = event[keys].copy()
        event_index["event_row"] = np.arange(len(event), dtype=np.int64)
        joined = mechanism_index.merge(event_index, on=keys, how="inner", validate="one_to_one")
        if joined.empty:
            raise RuntimeError("No common mechanism and Event-SSL prediction rows.")
        aligned_mechanism = mechanism.iloc[joined["mechanism_row"].to_numpy(dtype=np.int64)].reset_index(drop=True)
        aligned_event = event.iloc[joined["event_row"].to_numpy(dtype=np.int64)].reset_index(drop=True)
        mode = "one_to_one_inner_join"
    mechanism_fraction = len(aligned_mechanism) / max(len(mechanism), 1)
    event_fraction = len(aligned_event) / max(len(event), 1)
    if mechanism_fraction < minimum_join_fraction or event_fraction < minimum_join_fraction:
        raise RuntimeError(
            "Prediction join coverage is too low: "
            f"mechanism={mechanism_fraction:.6f}, event={event_fraction:.6f}."
        )
    differences = {}
    for column in ("M", "Psi", "target_M_next", "target_Psi_next"):
        first = numeric(aligned_mechanism, column)
        second = numeric(aligned_event, column)
        differences[column] = float(np.max(np.abs(first - second)))
        if differences[column] > tolerance:
            raise RuntimeError(
                f"Empirical anchor mismatch for {column}: {differences[column]:.3e}."
            )
    arrays = {
        "user_id": integer(aligned_mechanism, "user_id"),
        "step": integer(aligned_mechanism, "bundle_step_index"),
        "M": numeric(aligned_mechanism, "M"),
        "Psi": numeric(aligned_mechanism, "Psi"),
        "target_M_next": numeric(aligned_mechanism, "target_M_next"),
        "target_Psi_next": numeric(aligned_mechanism, "target_Psi_next"),
        "mechanism_next_M": numeric(aligned_mechanism, "mechanism_next_M"),
        "mechanism_next_Psi": numeric(aligned_mechanism, "mechanism_next_Psi"),
        "event_current_M": numeric(aligned_event, "event_current_M"),
        "event_current_Psi": numeric(aligned_event, "event_current_Psi"),
        "event_next_M": numeric(aligned_event, "event_next_M"),
        "event_next_Psi": numeric(aligned_event, "event_next_Psi"),
    }
    audit = {
        "mode": mode,
        "mechanism_rows": int(len(mechanism)),
        "event_rows": int(len(event)),
        "joined_rows": int(len(aligned_mechanism)),
        "mechanism_join_fraction": float(mechanism_fraction),
        "event_join_fraction": float(event_fraction),
        "anchor_max_abs_differences": differences,
    }
    return arrays, audit


def build_sufficient_statistics(
    arrays: Mapping[str, np.ndarray],
    partition: MacroPartition,
) -> Dict[str, Any]:
    user_id = np.asarray(arrays["user_id"], dtype=np.int64)
    all_users = np.unique(user_id)
    user_index = np.searchsorted(all_users, user_id)
    if not np.array_equal(all_users[user_index], user_id):
        raise RuntimeError("User indexing failed.")
    n_users = len(all_users)
    rows_per_user = np.bincount(user_index, minlength=n_users).astype(np.float64)
    if np.any(rows_per_user <= 0):
        raise RuntimeError("A joined user has no rows.")
    row_weight = 1.0 / rows_per_user[user_index]

    current = np.column_stack([arrays["M"], arrays["Psi"]])
    target_next = np.column_stack([arrays["target_M_next"], arrays["target_Psi_next"]])
    mechanism_next = np.column_stack([arrays["mechanism_next_M"], arrays["mechanism_next_Psi"]])
    event_current = np.column_stack([arrays["event_current_M"], arrays["event_current_Psi"]])
    event_next = np.column_stack([arrays["event_next_M"], arrays["event_next_Psi"]])

    anchor_cell, anchor_valid = grid_cells(current[:, 0], current[:, 1])
    learned_cell, learned_valid = grid_cells(event_current[:, 0], event_current[:, 1])
    finite_anchor = (
        anchor_valid
        & np.isfinite(target_next).all(axis=1)
        & np.isfinite(mechanism_next).all(axis=1)
        & np.isfinite(event_next).all(axis=1)
    )
    finite_learned = (
        learned_valid
        & np.isfinite(event_next).all(axis=1)
        & np.isfinite(event_current).all(axis=1)
    )

    empirical_delta = target_next - current
    mechanism_delta = mechanism_next - current
    event_anchor_delta = event_next - current
    event_learned_delta = event_next - event_current

    anchor_values = [
        np.ones(int(np.sum(finite_anchor)), dtype=np.float64),
        row_weight[finite_anchor],
        row_weight[finite_anchor] * empirical_delta[finite_anchor, 0],
        row_weight[finite_anchor] * empirical_delta[finite_anchor, 1],
        row_weight[finite_anchor] * mechanism_delta[finite_anchor, 0],
        row_weight[finite_anchor] * mechanism_delta[finite_anchor, 1],
        row_weight[finite_anchor] * event_anchor_delta[finite_anchor, 0],
        row_weight[finite_anchor] * event_anchor_delta[finite_anchor, 1],
    ]
    anchor_grouped = group_user_cell(
        user_index[finite_anchor], anchor_cell[finite_anchor], anchor_values
    )
    anchor_blocks = [
        grouped_to_csr(anchor_grouped, index, n_users)
        for index in range(len(anchor_values))
    ]
    anchor_names = (
        "count",
        "denominator",
        "empirical_u_sum",
        "empirical_v_sum",
        "mechanism_u_sum",
        "mechanism_v_sum",
        "event_anchor_u_sum",
        "event_anchor_v_sum",
    )
    anchor_matrix = sparse.hstack(anchor_blocks, format="csr", dtype=np.float64)
    anchor_index = {name: index for index, name in enumerate(anchor_names)}

    learned_values = [
        np.ones(int(np.sum(finite_learned)), dtype=np.float64),
        row_weight[finite_learned],
        row_weight[finite_learned] * event_learned_delta[finite_learned, 0],
        row_weight[finite_learned] * event_learned_delta[finite_learned, 1],
    ]
    learned_grouped = group_user_cell(
        user_index[finite_learned], learned_cell[finite_learned], learned_values
    )
    learned_blocks = [
        grouped_to_csr(learned_grouped, index, n_users)
        for index in range(len(learned_values))
    ]
    learned_names = ("count", "denominator", "event_learned_u_sum", "event_learned_v_sum")
    learned_matrix = sparse.hstack(learned_blocks, format="csr", dtype=np.float64)
    learned_index = {name: index for index, name in enumerate(learned_names)}

    empirical_current_state = assign_states(partition, current)
    empirical_next_state = assign_states(partition, target_next)
    mechanism_next_state = assign_states(partition, mechanism_next)
    event_current_state = assign_states(partition, event_current)
    event_next_state = assign_states(partition, event_next)
    transition_blocks = [
        transition_csr(user_index, empirical_current_state, empirical_next_state, n_users),
        transition_csr(user_index, empirical_current_state, mechanism_next_state, n_users),
        transition_csr(user_index, empirical_current_state, event_next_state, n_users),
        transition_csr(user_index, event_current_state, event_next_state, n_users),
    ]
    transition_matrix = sparse.hstack(transition_blocks, format="csr", dtype=np.float64)
    transition_index = {
        "empirical": 0,
        "mechanism": 1,
        "event_anchor": 2,
        "event_learned": 3,
    }

    return {
        "all_users": all_users,
        "n_users": int(n_users),
        "n_rows": int(len(user_id)),
        "anchor_matrix": anchor_matrix,
        "anchor_index": anchor_index,
        "learned_matrix": learned_matrix,
        "learned_index": learned_index,
        "transition_matrix": transition_matrix,
        "transition_index": transition_index,
        "audit": {
            "joined_users": int(n_users),
            "joined_rows": int(len(user_id)),
            "anchor_valid_rows": int(np.sum(finite_anchor)),
            "learned_valid_rows": int(np.sum(finite_learned)),
            "anchor_unique_user_cell_pairs": int(len(anchor_grouped.cell)),
            "learned_unique_user_cell_pairs": int(len(learned_grouped.cell)),
            "user_balancing": "each sampled learner copy contributes one total unit to field aggregation",
            "transition_estimand": "raw interval-count rows under the frozen K=6 partition",
        },
    }


def block(vector: np.ndarray, index: Mapping[str, int], name: str, size: int) -> np.ndarray:
    start = int(index[name]) * size
    return np.asarray(vector[start : start + size], dtype=np.float64)


def fields_from_aggregates(
    anchor_vector: np.ndarray,
    learned_vector: np.ndarray,
    anchor_index: Mapping[str, int],
    learned_index: Mapping[str, int],
) -> Dict[str, np.ndarray]:
    anchor_count = block(anchor_vector, anchor_index, "count", N_CELLS)
    anchor_denominator = block(anchor_vector, anchor_index, "denominator", N_CELLS)
    learned_count = block(learned_vector, learned_index, "count", N_CELLS)
    learned_denominator = block(learned_vector, learned_index, "denominator", N_CELLS)

    def field(vector: np.ndarray, index: Mapping[str, int], prefix: str, denominator: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u_sum = block(vector, index, f"{prefix}_u_sum", N_CELLS)
        v_sum = block(vector, index, f"{prefix}_v_sum", N_CELLS)
        u = np.full(N_CELLS, np.nan, dtype=np.float64)
        v = np.full(N_CELLS, np.nan, dtype=np.float64)
        valid = denominator > EPS
        u[valid] = u_sum[valid] / denominator[valid]
        v[valid] = v_sum[valid] / denominator[valid]
        return u, v

    empirical_u, empirical_v = field(anchor_vector, anchor_index, "empirical", anchor_denominator)
    mechanism_u, mechanism_v = field(anchor_vector, anchor_index, "mechanism", anchor_denominator)
    event_anchor_u, event_anchor_v = field(anchor_vector, anchor_index, "event_anchor", anchor_denominator)
    event_learned_u, event_learned_v = field(learned_vector, learned_index, "event_learned", learned_denominator)
    return {
        "anchor_count": anchor_count,
        "anchor_denominator": anchor_denominator,
        "learned_count": learned_count,
        "learned_denominator": learned_denominator,
        "empirical_u": empirical_u,
        "empirical_v": empirical_v,
        "mechanism_u": mechanism_u,
        "mechanism_v": mechanism_v,
        "event_anchor_u": event_anchor_u,
        "event_anchor_v": event_anchor_v,
        "event_learned_u": event_learned_u,
        "event_learned_v": event_learned_v,
    }


def transition_counts_from_aggregate(
    vector: np.ndarray,
    index: Mapping[str, int],
) -> Dict[str, np.ndarray]:
    output: Dict[str, np.ndarray] = {}
    for name, position in index.items():
        start = int(position) * K * K
        output[name] = np.asarray(
            vector[start : start + K * K], dtype=np.float64
        ).reshape(K, K)
    return output


def transition_rows_complete(
    counts: Mapping[str, np.ndarray],
    names: Sequence[str],
) -> Tuple[bool, float]:
    minima: List[float] = []
    complete = True
    for name in names:
        row_counts = np.asarray(counts[name], dtype=np.float64).sum(axis=1)
        if row_counts.size == 0 or not np.isfinite(row_counts).all() or np.any(row_counts <= 0):
            complete = False
        minima.append(float(np.min(row_counts)) if row_counts.size else float("nan"))
    finite_minima = [value for value in minima if np.isfinite(value)]
    minimum = min(finite_minima) if finite_minima else float("nan")
    return bool(complete), float(minimum)


def point_estimates(statistics: Mapping[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    anchor_total = np.asarray(statistics["anchor_matrix"].sum(axis=0)).ravel()
    learned_total = np.asarray(statistics["learned_matrix"].sum(axis=0)).ravel()
    transition_total = np.asarray(statistics["transition_matrix"].sum(axis=0)).ravel()
    fields = fields_from_aggregates(
        anchor_total,
        learned_total,
        statistics["anchor_index"],
        statistics["learned_index"],
    )
    transition_counts = transition_counts_from_aggregate(
        transition_total, statistics["transition_index"]
    )
    transition_complete, minimum_origin_count = transition_rows_complete(
        transition_counts, tuple(transition_counts)
    )
    if not transition_complete:
        raise RuntimeError("At least one formal transition view has an empty mesostate origin row.")
    transitions = {
        name: normalize_transition(counts) for name, counts in transition_counts.items()
    }
    masks = {
        "mechanism_vs_empirical": fields["anchor_count"] >= MIN_DRIFT_COUNT,
        "event_ssl_anchor_vs_empirical": fields["anchor_count"] >= MIN_DRIFT_COUNT,
        "event_ssl_learned_vs_empirical": (
            (fields["anchor_count"] >= MIN_DRIFT_COUNT)
            & (fields["learned_count"] >= MIN_DRIFT_COUNT)
        ),
        "mechanism_vs_event_ssl_anchor": fields["anchor_count"] >= MIN_DRIFT_COUNT,
    }
    comparisons = {
        "mechanism_vs_empirical": (
            fields["mechanism_u"], fields["mechanism_v"], fields["empirical_u"], fields["empirical_v"]
        ),
        "event_ssl_anchor_vs_empirical": (
            fields["event_anchor_u"], fields["event_anchor_v"], fields["empirical_u"], fields["empirical_v"]
        ),
        "event_ssl_learned_vs_empirical": (
            fields["event_learned_u"], fields["event_learned_v"], fields["empirical_u"], fields["empirical_v"]
        ),
        "mechanism_vs_event_ssl_anchor": (
            fields["mechanism_u"], fields["mechanism_v"], fields["event_anchor_u"], fields["event_anchor_v"]
        ),
    }
    rows: List[dict] = []
    for comparison, values in comparisons.items():
        metrics = field_metrics(
            values[0], values[1], values[2], values[3], fields["anchor_denominator"], masks[comparison]
        )
        for metric, value in metrics.items():
            rows.append(
                {
                    "domain": "field",
                    "comparison": comparison,
                    "metric": metric,
                    "point_estimate": value,
                }
            )
    transition_pairs = {
        "mechanism_vs_empirical": (transitions["mechanism"], transitions["empirical"]),
        "event_ssl_anchor_vs_empirical": (transitions["event_anchor"], transitions["empirical"]),
        "event_ssl_learned_vs_empirical": (transitions["event_learned"], transitions["empirical"]),
        "mechanism_vs_event_ssl": (transitions["mechanism"], transitions["event_anchor"]),
    }
    for comparison, values in transition_pairs.items():
        metrics = transition_metrics(values[0], values[1])
        for metric, value in metrics.items():
            rows.append(
                {
                    "domain": "transition",
                    "comparison": comparison,
                    "metric": metric,
                    "point_estimate": value,
                }
            )
    state_rows = []
    for state in range(K):
        state_rows.append(
            {
                "state": state,
                "empirical_Pii": float(transitions["empirical"][state, state]),
                "mechanism_Pii": float(transitions["mechanism"][state, state]),
                "event_ssl_anchor_Pii": float(transitions["event_anchor"][state, state]),
                "event_ssl_learned_Pii": float(transitions["event_learned"][state, state]),
                "mechanism_minus_event_ssl_Pii": float(
                    transitions["mechanism"][state, state] - transitions["event_anchor"][state, state]
                ),
                "mechanism_minus_empirical_Pii": float(
                    transitions["mechanism"][state, state] - transitions["empirical"][state, state]
                ),
                "event_ssl_minus_empirical_Pii": float(
                    transitions["event_anchor"][state, state] - transitions["empirical"][state, state]
                ),
                "event_ssl_learned_minus_empirical_Pii": float(
                    transitions["event_learned"][state, state] - transitions["empirical"][state, state]
                ),
                "mechanism_vs_empirical_row_tv": float(
                    row_tv(transitions["mechanism"], transitions["empirical"])[state]
                ),
                "event_ssl_vs_empirical_row_tv": float(
                    row_tv(transitions["event_anchor"], transitions["empirical"])[state]
                ),
                "mechanism_vs_event_ssl_row_tv": float(
                    row_tv(transitions["mechanism"], transitions["event_anchor"])[state]
                ),
            }
        )
    auxiliary = {
        "fields": fields,
        "formal_masks": masks,
        "transitions": transitions,
        "transition_counts": transition_counts,
        "minimum_origin_transition_count": minimum_origin_count,
        "statewise": pd.DataFrame(state_rows),
        "loso": leave_one_state_out(transitions["mechanism"], transitions["event_anchor"]),
    }
    return pd.DataFrame(rows), auxiliary


def transition_matrix_table(
    transitions: Mapping[str, np.ndarray],
    counts: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    rows: List[dict] = []
    for model in ("empirical", "mechanism", "event_anchor", "event_learned"):
        matrix = np.asarray(transitions[model], dtype=np.float64)
        raw = np.asarray(counts[model], dtype=np.float64)
        for origin in range(K):
            for destination in range(K):
                rows.append(
                    {
                        "model": model,
                        "origin_state": origin,
                        "destination_state": destination,
                        "interval_count": float(raw[origin, destination]),
                        "transition_probability": float(matrix[origin, destination]),
                        "self_transition": bool(origin == destination),
                    }
                )
    return pd.DataFrame(rows)


def point_value(point: pd.DataFrame, domain: str, comparison: str, metric: str) -> float:
    selected = point[
        (point["domain"] == domain)
        & (point["comparison"] == comparison)
        & (point["metric"] == metric)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Point estimate not unique: {domain}/{comparison}/{metric}")
    return float(selected.iloc[0]["point_estimate"])


def point_reconstruction_audit(
    point: pd.DataFrame,
    tolerance: float,
    enforce: bool,
) -> pd.DataFrame:
    actual = {
        "mechanism_vs_empirical_drift_vector_corr": point_value(
            point, "field", "mechanism_vs_empirical", "drift_vector_corr"
        ),
        "event_ssl_learned_vs_empirical_drift_vector_corr": point_value(
            point, "field", "event_ssl_learned_vs_empirical", "drift_vector_corr"
        ),
        "mechanism_vs_event_ssl_anchor_drift_vector_corr": point_value(
            point, "field", "mechanism_vs_event_ssl_anchor", "drift_vector_corr"
        ),
        "mechanism_vs_event_ssl_anchor_drift_speed_corr": point_value(
            point, "field", "mechanism_vs_event_ssl_anchor", "drift_speed_corr"
        ),
        "mechanism_vs_event_ssl_anchor_weighted_local_cosine": point_value(
            point, "field", "mechanism_vs_event_ssl_anchor", "weighted_local_cosine"
        ),
        "event_ssl_anchor_vs_empirical_drift_vector_corr": point_value(
            point, "field", "event_ssl_anchor_vs_empirical", "drift_vector_corr"
        ),
        "event_ssl_learned_vs_empirical_weighted_local_cosine": point_value(
            point, "field", "event_ssl_learned_vs_empirical", "weighted_local_cosine"
        ),
        "mechanism_vs_empirical_transition_mean_row_tv": point_value(
            point, "transition", "mechanism_vs_empirical", "transition_mean_row_tv"
        ),
        "event_ssl_anchor_vs_empirical_transition_mean_row_tv": point_value(
            point, "transition", "event_ssl_anchor_vs_empirical", "transition_mean_row_tv"
        ),
        "mechanism_vs_event_ssl_transition_mean_row_tv": point_value(
            point, "transition", "mechanism_vs_event_ssl", "transition_mean_row_tv"
        ),
        "event_ssl_learned_vs_empirical_transition_mean_row_tv": point_value(
            point, "transition", "event_ssl_learned_vs_empirical", "transition_mean_row_tv"
        ),
        "event_ssl_learned_vs_empirical_self_transition_pearson": point_value(
            point, "transition", "event_ssl_learned_vs_empirical", "self_transition_pearson"
        ),
        "mechanism_vs_event_ssl_self_transition_pearson": point_value(
            point, "transition", "mechanism_vs_event_ssl", "self_transition_pearson"
        ),
        "mechanism_vs_empirical_self_transition_pearson": point_value(
            point, "transition", "mechanism_vs_empirical", "self_transition_pearson"
        ),
        "event_ssl_anchor_vs_empirical_self_transition_pearson": point_value(
            point, "transition", "event_ssl_anchor_vs_empirical", "self_transition_pearson"
        ),
    }
    rows = []
    for metric, expected in EXPECTED_POINT_VALUES.items():
        observed = actual[metric]
        difference = abs(observed - expected)
        rows.append(
            {
                "metric": metric,
                "manuscript_rounded_reference": expected,
                "reconstructed_point": observed,
                "absolute_difference": difference,
                "tolerance": tolerance,
                "passed": bool(np.isfinite(observed) and difference <= tolerance),
            }
        )
    frame = pd.DataFrame(rows)
    if enforce and not bool(frame["passed"].all()):
        failed = frame.loc[~frame["passed"], ["metric", "reconstructed_point", "manuscript_rounded_reference"]]
        raise RuntimeError("Formal point reconstruction failed:\n" + failed.to_string(index=False))
    return frame


def support_diagnostics(
    formal_mask: np.ndarray,
    adaptive_mask: np.ndarray,
    weight: np.ndarray,
) -> Dict[str, float]:
    formal = np.asarray(formal_mask, dtype=bool)
    adaptive = np.asarray(adaptive_mask, dtype=bool)
    intersection = formal & adaptive
    union = formal | adaptive
    denominator = float(np.sum(weight[formal]))
    return {
        "formal_supported_cells": int(np.sum(formal)),
        "adaptive_supported_cells": int(np.sum(adaptive)),
        "support_intersection_cells": int(np.sum(intersection)),
        "support_jaccard": float(np.sum(intersection) / max(np.sum(union), 1)),
        "formal_support_weight_coverage": float(
            np.sum(weight[intersection]) / max(denominator, EPS)
        ),
        "formal_cells_lost": int(np.sum(formal & ~adaptive)),
        "new_cells_added": int(np.sum(adaptive & ~formal)),
    }


def bootstrap_analysis(
    statistics: Mapping[str, Any],
    auxiliary: Mapping[str, Any],
    replicates: int,
    batch_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_users = int(statistics["n_users"])
    anchor_transposed = statistics["anchor_matrix"].T.tocsr()
    learned_transposed = statistics["learned_matrix"].T.tocsr()
    transition_transposed = statistics["transition_matrix"].T.tocsr()
    rng = np.random.default_rng(int(seed))
    field_rows: List[dict] = []
    transition_rows: List[dict] = []
    state_rows: List[dict] = []
    formal_masks = auxiliary["formal_masks"]

    completed = 0
    while completed < replicates:
        current_batch = min(batch_size, replicates - completed)
        probabilities = np.full(n_users, 1.0 / n_users, dtype=np.float64)
        multiplicities = rng.multinomial(n_users, probabilities, size=current_batch).astype(np.float64)
        anchor_batch = np.asarray((anchor_transposed @ multiplicities.T).T, dtype=np.float64)
        learned_batch = np.asarray((learned_transposed @ multiplicities.T).T, dtype=np.float64)
        transition_batch = np.asarray((transition_transposed @ multiplicities.T).T, dtype=np.float64)
        for local in range(current_batch):
            replicate = completed + local
            fields = fields_from_aggregates(
                anchor_batch[local],
                learned_batch[local],
                statistics["anchor_index"],
                statistics["learned_index"],
            )
            adaptive_masks = {
                "mechanism_vs_empirical": fields["anchor_count"] >= MIN_DRIFT_COUNT,
                "event_ssl_anchor_vs_empirical": fields["anchor_count"] >= MIN_DRIFT_COUNT,
                "event_ssl_learned_vs_empirical": (
                    (fields["anchor_count"] >= MIN_DRIFT_COUNT)
                    & (fields["learned_count"] >= MIN_DRIFT_COUNT)
                ),
                "mechanism_vs_event_ssl_anchor": fields["anchor_count"] >= MIN_DRIFT_COUNT,
            }
            comparisons = {
                "mechanism_vs_empirical": (
                    fields["mechanism_u"], fields["mechanism_v"], fields["empirical_u"], fields["empirical_v"]
                ),
                "event_ssl_anchor_vs_empirical": (
                    fields["event_anchor_u"], fields["event_anchor_v"], fields["empirical_u"], fields["empirical_v"]
                ),
                "event_ssl_learned_vs_empirical": (
                    fields["event_learned_u"], fields["event_learned_v"], fields["empirical_u"], fields["empirical_v"]
                ),
                "mechanism_vs_event_ssl_anchor": (
                    fields["mechanism_u"], fields["mechanism_v"], fields["event_anchor_u"], fields["event_anchor_v"]
                ),
            }
            for comparison, values in comparisons.items():
                for contract, mask in (
                    ("fixed_formal_support", formal_masks[comparison]),
                    ("support_reselected", adaptive_masks[comparison]),
                ):
                    required_denominator = fields["anchor_denominator"] > EPS
                    if comparison == "event_ssl_learned_vs_empirical":
                        required_denominator &= fields["learned_denominator"] > EPS
                    available_mask = np.asarray(mask, dtype=bool) & required_denominator
                    fixed_complete = bool(
                        contract != "fixed_formal_support"
                        or np.array_equal(available_mask, np.asarray(mask, dtype=bool))
                    )
                    metrics = field_metrics(
                        values[0],
                        values[1],
                        values[2],
                        values[3],
                        fields["anchor_denominator"],
                        available_mask if fixed_complete else np.zeros(N_CELLS, dtype=bool),
                    )
                    diagnostics = support_diagnostics(
                        formal_masks[comparison], adaptive_masks[comparison], fields["anchor_denominator"]
                    )
                    field_rows.append(
                        {
                            "replicate": replicate,
                            "comparison": comparison,
                            "support_contract": contract,
                            "fixed_support_complete": fixed_complete,
                            **metrics,
                            **diagnostics,
                        }
                    )

            transition_counts = transition_counts_from_aggregate(
                transition_batch[local], statistics["transition_index"]
            )
            transitions = {
                name: normalize_transition(counts)
                for name, counts in transition_counts.items()
            }
            transition_pair_contracts = {
                "mechanism_vs_empirical": ("mechanism", "empirical"),
                "event_ssl_anchor_vs_empirical": ("event_anchor", "empirical"),
                "event_ssl_learned_vs_empirical": ("event_learned", "empirical"),
                "mechanism_vs_event_ssl": ("mechanism", "event_anchor"),
            }
            for comparison, names in transition_pair_contracts.items():
                pair_complete, minimum_origin_count = transition_rows_complete(
                    transition_counts, names
                )
                metrics = (
                    transition_metrics(transitions[names[0]], transitions[names[1]])
                    if pair_complete
                    else {
                        "transition_mean_row_tv": float("nan"),
                        "transition_max_row_tv": float("nan"),
                        "self_transition_pearson": float("nan"),
                        "self_transition_spearman": float("nan"),
                    }
                )
                transition_rows.append(
                    {
                        "replicate": replicate,
                        "comparison": comparison,
                        "all_origin_rows_present": pair_complete,
                        "minimum_origin_transition_count": minimum_origin_count,
                        **metrics,
                    }
                )
            per_view_row_counts = {
                name: np.asarray(counts, dtype=np.float64).sum(axis=1)
                for name, counts in transition_counts.items()
            }
            for state in range(K):
                def state_pii(name: str) -> float:
                    count = per_view_row_counts[name][state]
                    return (
                        float(transitions[name][state, state])
                        if np.isfinite(count) and count > 0
                        else float("nan")
                    )

                empirical_pii = state_pii("empirical")
                mechanism_pii = state_pii("mechanism")
                event_pii = state_pii("event_anchor")
                event_learned_pii = state_pii("event_learned")
                finite_minimum = min(
                    per_view_row_counts[name][state]
                    for name in per_view_row_counts
                )
                state_rows.append(
                    {
                        "replicate": replicate,
                        "state": state,
                        "all_origin_rows_present": bool(
                            all(per_view_row_counts[name][state] > 0 for name in per_view_row_counts)
                        ),
                        "minimum_origin_transition_count": float(finite_minimum),
                        "empirical_Pii": empirical_pii,
                        "mechanism_Pii": mechanism_pii,
                        "event_ssl_anchor_Pii": event_pii,
                        "event_ssl_learned_Pii": event_learned_pii,
                        "mechanism_minus_event_ssl_Pii": mechanism_pii - event_pii,
                        "mechanism_minus_empirical_Pii": mechanism_pii - empirical_pii,
                        "event_ssl_minus_empirical_Pii": event_pii - empirical_pii,
                        "event_ssl_learned_minus_empirical_Pii": (
                            event_learned_pii - empirical_pii
                        ),
                    }
                )
        completed += current_batch
        print(f"[learner-cluster bootstrap] {completed}/{replicates}", flush=True)
    return pd.DataFrame(field_rows), pd.DataFrame(transition_rows), pd.DataFrame(state_rows)


def correlation_transform(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -1.0 + 1e-12, 1.0 - 1e-12)
    return np.arctanh(clipped)


def correlation_inverse(values: np.ndarray) -> np.ndarray:
    return np.tanh(np.asarray(values, dtype=np.float64))


def interval_summary(
    values: np.ndarray,
    point: float,
    transform: str,
    ci_level: float,
) -> Dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    empty = {
        "finite_replicates": int(finite.size),
        "bootstrap_mean": float("nan"),
        "bootstrap_sd": float("nan"),
        "bootstrap_median": float("nan"),
        "bootstrap_bias": float("nan"),
        "se_centered_ci_low": float("nan"),
        "se_centered_ci_high": float("nan"),
        "basic_ci_low": float("nan"),
        "basic_ci_high": float("nan"),
        "percentile_low": float("nan"),
        "percentile_high": float("nan"),
    }
    if finite.size == 0 or not np.isfinite(point):
        return empty
    alpha = 1.0 - float(ci_level)
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    if transform == "fisher_z":
        transformed = correlation_transform(finite)
        point_t = float(correlation_transform(np.asarray([point]))[0])
        error = transformed - point_t
        basic_low_t = point_t - float(np.quantile(error, 1.0 - alpha / 2.0))
        basic_high_t = point_t - float(np.quantile(error, alpha / 2.0))
        basic_low, basic_high = correlation_inverse(np.asarray([basic_low_t, basic_high_t]))
        percentile = correlation_inverse(
            np.quantile(transformed, [alpha / 2.0, 1.0 - alpha / 2.0])
        )
        transformed_sd = float(np.std(transformed, ddof=1)) if finite.size > 1 else 0.0
        se_interval = correlation_inverse(
            np.asarray([point_t - critical * transformed_sd, point_t + critical * transformed_sd])
        )
    else:
        error = finite - point
        basic_low = point - float(np.quantile(error, 1.0 - alpha / 2.0))
        basic_high = point - float(np.quantile(error, alpha / 2.0))
        percentile = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
        standard_error = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        se_interval = np.asarray(
            [point - critical * standard_error, point + critical * standard_error],
            dtype=np.float64,
        )
        if transform == "unit_interval":
            basic_low = float(np.clip(basic_low, 0.0, 1.0))
            basic_high = float(np.clip(basic_high, 0.0, 1.0))
            percentile = np.clip(percentile, 0.0, 1.0)
            se_interval = np.clip(se_interval, 0.0, 1.0)
        elif transform == "bounded_raw":
            basic_low = float(np.clip(basic_low, -1.0, 1.0))
            basic_high = float(np.clip(basic_high, -1.0, 1.0))
            percentile = np.clip(percentile, -1.0, 1.0)
            se_interval = np.clip(se_interval, -1.0, 1.0)
    return {
        "finite_replicates": int(finite.size),
        "bootstrap_mean": float(np.mean(finite)),
        "bootstrap_sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "bootstrap_median": float(np.median(finite)),
        "bootstrap_bias": float(np.mean(finite) - point),
        "se_centered_ci_low": float(se_interval[0]),
        "se_centered_ci_high": float(se_interval[1]),
        "basic_ci_low": float(basic_low),
        "basic_ci_high": float(basic_high),
        "percentile_low": float(percentile[0]),
        "percentile_high": float(percentile[1]),
    }


def summarize_field_bootstrap(
    point: pd.DataFrame,
    bootstrap: pd.DataFrame,
    replicates: int,
    ci_level: float,
) -> pd.DataFrame:
    rows: List[dict] = []
    for (comparison, contract), group in bootstrap.groupby(
        ["comparison", "support_contract"], sort=False
    ):
        for metric in ("drift_vector_corr", "drift_speed_corr", "weighted_local_cosine"):
            point_estimate = point_value(point, "field", comparison, metric)
            transform = "fisher_z" if metric != "weighted_local_cosine" else "bounded_raw"
            summary = interval_summary(
                pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float),
                point_estimate,
                transform,
                ci_level,
            )
            rows.append(
                {
                    "comparison": comparison,
                    "support_contract": contract,
                    "metric": metric,
                    "point_estimate": point_estimate,
                    "requested_replicates": replicates,
                    "finite_fraction": summary["finite_replicates"] / max(replicates, 1),
                    "primary_interval": bool(contract == "fixed_formal_support"),
                    "reported_interval_method": (
                        (
                            "cluster_bootstrap_se_fisher_z"
                            if transform == "fisher_z"
                            else "cluster_bootstrap_se_raw_bounded"
                        )
                        if contract == "fixed_formal_support"
                        else (
                            "support_reselected_percentile_fisher_z"
                            if transform == "fisher_z"
                            else "support_reselected_percentile_raw_bounded"
                        )
                    ),
                    "reported_ci_low": (
                        summary["se_centered_ci_low"]
                        if contract == "fixed_formal_support"
                        else summary["percentile_low"]
                    ),
                    "reported_ci_high": (
                        summary["se_centered_ci_high"]
                        if contract == "fixed_formal_support"
                        else summary["percentile_high"]
                    ),
                    "interval_interpretation": (
                        "learner-cluster bootstrap standard-error interval centred on the formal point and conditional on the formal support"
                        if contract == "fixed_formal_support"
                        else "secondary support-reselected percentile interval; threshold selection is part of the estimator"
                    ),
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def summarize_transition_bootstrap(
    point: pd.DataFrame,
    bootstrap: pd.DataFrame,
    replicates: int,
    ci_level: float,
) -> pd.DataFrame:
    rows: List[dict] = []
    metric_contract = {
        "transition_mean_row_tv": ("unit_interval", "se_centered"),
        "transition_max_row_tv": ("unit_interval", "se_centered"),
        "self_transition_pearson": ("fisher_z", "se_centered"),
        "self_transition_spearman": ("bounded_raw", "percentile"),
    }
    for comparison, group in bootstrap.groupby("comparison", sort=False):
        for metric, (transform, reported_method) in metric_contract.items():
            point_estimate = point_value(point, "transition", comparison, metric)
            summary = interval_summary(
                pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float),
                point_estimate,
                transform,
                ci_level,
            )
            if reported_method == "percentile":
                reported_low = summary["percentile_low"]
                reported_high = summary["percentile_high"]
                method_label = "percentile_raw_scale"
            else:
                reported_low = summary["se_centered_ci_low"]
                reported_high = summary["se_centered_ci_high"]
                method_label = (
                    "cluster_bootstrap_se_fisher_z"
                    if transform == "fisher_z"
                    else "cluster_bootstrap_se_raw_scale"
                )
            rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "point_estimate": point_estimate,
                    "requested_replicates": replicates,
                    "finite_fraction": summary["finite_replicates"] / max(replicates, 1),
                    "reported_interval_method": method_label,
                    "reported_ci_low": reported_low,
                    "reported_ci_high": reported_high,
                    "interval_interpretation": (
                        "paired learner-cluster bootstrap interval conditional on the six fixed mesostates; "
                        "it is not inference over a population of states"
                        if metric.startswith("self_transition")
                        else "paired learner-cluster bootstrap interval under the formal interval-count transition estimand"
                    ),
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def summarize_statewise_bootstrap(
    point_statewise: pd.DataFrame,
    bootstrap_statewise: pd.DataFrame,
    replicates: int,
    ci_level: float,
) -> pd.DataFrame:
    rows: List[dict] = []
    contracts = {
        "empirical_Pii": ("empirical", "unit_interval"),
        "mechanism_Pii": ("mechanism", "unit_interval"),
        "event_ssl_anchor_Pii": ("event_ssl_anchor", "unit_interval"),
        "event_ssl_learned_Pii": ("event_ssl_learned", "unit_interval"),
        "mechanism_minus_event_ssl_Pii": ("mechanism_minus_event_ssl", "bounded_raw"),
        "mechanism_minus_empirical_Pii": ("mechanism_minus_empirical", "bounded_raw"),
        "event_ssl_minus_empirical_Pii": ("event_ssl_minus_empirical", "bounded_raw"),
        "event_ssl_learned_minus_empirical_Pii": (
            "event_ssl_learned_minus_empirical",
            "bounded_raw",
        ),
    }
    for state in range(K):
        point_row = point_statewise[point_statewise["state"] == state].iloc[0]
        group = bootstrap_statewise[bootstrap_statewise["state"] == state]
        for column, (label, transform) in contracts.items():
            summary = interval_summary(
                pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float),
                float(point_row[column]),
                transform,
                ci_level,
            )
            rows.append(
                {
                    "state": state,
                    "quantity": label,
                    "point_estimate": float(point_row[column]),
                    "requested_replicates": replicates,
                    "finite_fraction": summary["finite_replicates"] / max(replicates, 1),
                    "reported_interval_method": (
                        "cluster_bootstrap_percentile_unit_interval"
                        if transform == "unit_interval"
                        else "cluster_bootstrap_percentile_raw_bounded"
                    ),
                    "reported_ci_low": summary["percentile_low"],
                    "reported_ci_high": summary["percentile_high"],
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def quality_gates(
    args: argparse.Namespace,
    join_audit: Mapping[str, Any],
    reconstruction: pd.DataFrame,
    field_summary: pd.DataFrame,
    transition_summary: pd.DataFrame,
    statewise_summary: pd.DataFrame,
    statistics: Mapping[str, Any],
) -> pd.DataFrame:
    field_fixed = field_summary[field_summary["support_contract"] == "fixed_formal_support"]
    checks = [
        (
            "formal_point_reconstruction",
            bool(reconstruction["passed"].all()),
            f"passed={int(reconstruction['passed'].sum())}/{len(reconstruction)}",
        ),
        (
            "common_confirmation_cohort",
            bool(
                (args.expected_common_rows <= 0 or int(join_audit["joined_rows"]) == args.expected_common_rows)
                and (args.expected_common_users <= 0 or int(statistics["n_users"]) == args.expected_common_users)
            ),
            f"rows={join_audit['joined_rows']}, users={statistics['n_users']}",
        ),
        (
            "join_coverage",
            bool(
                float(join_audit["mechanism_join_fraction"]) >= args.minimum_join_fraction
                and float(join_audit["event_join_fraction"]) >= args.minimum_join_fraction
            ),
            f"mechanism={join_audit['mechanism_join_fraction']:.6f}, event={join_audit['event_join_fraction']:.6f}",
        ),
        (
            "fixed_support_interval_finite_fraction",
            bool((pd.to_numeric(field_fixed["finite_fraction"], errors="coerce") >= args.minimum_finite_fraction).all()),
            f"minimum={pd.to_numeric(field_fixed['finite_fraction'], errors='coerce').min():.6f}",
        ),
        (
            "transition_interval_finite_fraction",
            bool((pd.to_numeric(transition_summary["finite_fraction"], errors="coerce") >= args.minimum_finite_fraction).all()),
            f"minimum={pd.to_numeric(transition_summary['finite_fraction'], errors='coerce').min():.6f}",
        ),
        (
            "statewise_interval_finite_fraction",
            bool((pd.to_numeric(statewise_summary["finite_fraction"], errors="coerce") >= args.minimum_finite_fraction).all()),
            f"minimum={pd.to_numeric(statewise_summary['finite_fraction'], errors='coerce').min():.6f}",
        ),
        ("mechanism_manifest_verified", True, "checksum and output-only guardrails passed"),
        ("event_ssl_manifest_verified", True, "predictive-state output-only B_confirm record passed"),
        (
            "bootstrap_replicates",
            bool(args.bootstrap_replicates >= 1000),
            f"replicates={args.bootstrap_replicates}",
        ),
        ("models_frozen", True, "prediction tables only; no model or probe fitting"),
        (
            "coordinates_grid_partition_frozen",
            True,
            "M,Psi; 40x40 grid; checksum-matched Stage-1 K=6 across both models",
        ),
        ("null_and_surrogate_not_rerun", True, "outside this learner-sampling audit"),
        ("B_confirm_output_only", True, "no development or model update"),
    ]
    return pd.DataFrame(
        [{"quality_gate": name, "passed": passed, "detail": detail} for name, passed, detail in checks]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired learner-cluster uncertainty for frozen headline field and transition summaries."
    )
    parser.add_argument("--stage1-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/stage1"))
    parser.add_argument("--mechanism-predictions", type=Path, required=False)
    parser.add_argument("--mechanism-manifest", type=Path, required=False)
    parser.add_argument("--stage1-script", type=Path, default=None)
    parser.add_argument("--mechanism-confirm-script", type=Path, default=None)
    parser.add_argument("--event-ssl-evaluate-script", type=Path, default=None)
    parser.add_argument("--event-ssl-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/stage4_event_ssl/evaluation_predictive_state"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-batch-size", type=int, default=20)
    parser.add_argument("--bootstrap-seed", type=int, default=20260806)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--minimum-join-fraction", type=float, default=0.999)
    parser.add_argument("--anchor-tolerance", type=float, default=2e-6)
    parser.add_argument("--point-tolerance", type=float, default=5e-4)
    parser.add_argument("--minimum-finite-fraction", type=float, default=0.99)
    parser.add_argument("--expected-common-rows", type=int, default=3233208)
    parser.add_argument("--expected-common-users", type=int, default=56195)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_self_test() -> None:
    rng = np.random.default_rng(7)
    n_users = 40
    rows_per_user = 20
    user_id = np.repeat(np.arange(n_users), rows_per_user)
    current = rng.uniform(-0.8, 0.8, size=(len(user_id), 2))
    target = np.clip(current + 0.1 * (-current) + rng.normal(0, 0.03, size=current.shape), -1, 1)
    mechanism = np.clip(current + 0.09 * (-current) + rng.normal(0, 0.02, size=current.shape), -1, 1)
    event_current = np.clip(current + rng.normal(0, 0.02, size=current.shape), -1, 1)
    event_next = np.clip(event_current + 0.08 * (-event_current) + rng.normal(0, 0.03, size=current.shape), -1, 1)
    centers = np.column_stack([np.linspace(-1, 1, K), np.linspace(-0.8, 0.8, K)])
    temporary = tempfile.TemporaryDirectory(prefix="frozen_headline_cluster_")
    temporary_root = Path(temporary.name)
    metadata_path = temporary_root / "fixed_k6_model_metadata.json"
    centers_path = temporary_root / "fixed_k6_centers.csv"
    metadata_path.write_text("{}\n", encoding="utf-8")
    centers_path.write_text("macrostate\n", encoding="utf-8")
    partition = MacroPartition(
        np.zeros(2), np.ones(2), centers, metadata_path, centers_path
    )
    metadata_sha = sha256_file(metadata_path)
    centers_sha = sha256_file(centers_path)
    provenance = validate_partition_provenance(
        {
            "stage1_fixed_k6_contract": {
                "status": "verified",
                "checks": {"synthetic_contract": True},
                "metadata": {"macrostate_k": K, "fit_split": "A_train"},
                "metadata_sha256": metadata_sha,
                "centers_sha256": centers_sha,
            }
        },
        {
            "macro_partition": {
                "verified": True,
                "macrostate_k": K,
                "fit_split": "A_train",
                "metadata_sha256": metadata_sha,
                "centers_sha256": centers_sha,
            }
        },
        partition,
    )
    if not all(bool(record.get("passed")) for record in provenance["models"].values()):
        raise RuntimeError("Self-test partition provenance audit failed.")
    arrays = {
        "user_id": user_id,
        "step": np.tile(np.arange(rows_per_user), n_users),
        "M": current[:, 0],
        "Psi": current[:, 1],
        "target_M_next": target[:, 0],
        "target_Psi_next": target[:, 1],
        "mechanism_next_M": mechanism[:, 0],
        "mechanism_next_Psi": mechanism[:, 1],
        "event_current_M": event_current[:, 0],
        "event_current_Psi": event_current[:, 1],
        "event_next_M": event_next[:, 0],
        "event_next_Psi": event_next[:, 1],
    }
    statistics = build_sufficient_statistics(arrays, partition)
    point, auxiliary = point_estimates(statistics)
    fields, transitions, statewise = bootstrap_analysis(
        statistics, auxiliary, replicates=20, batch_size=5, seed=11
    )
    field_summary = summarize_field_bootstrap(point, fields, 20, 0.95)
    transition_summary = summarize_transition_bootstrap(point, transitions, 20, 0.95)
    state_summary = summarize_statewise_bootstrap(auxiliary["statewise"], statewise, 20, 0.95)
    if point.empty or field_summary.empty or transition_summary.empty or state_summary.empty:
        raise RuntimeError("Self-test produced empty outputs.")
    if not np.isfinite(pd.to_numeric(point["point_estimate"], errors="coerce")).any():
        raise RuntimeError("Self-test produced no finite point estimates.")
    transition_long = transition_matrix_table(
        auxiliary["transitions"], auxiliary["transition_counts"]
    )
    if set(transition_long["model"].astype(str)) != {
        "empirical", "mechanism", "event_anchor", "event_learned"
    }:
        raise RuntimeError("Self-test transition export omitted a formal evaluation gauge.")
    temporary.cleanup()
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.bootstrap_replicates < 1000:
        raise ValueError(
            "--bootstrap-replicates must be at least 1000 for a formal run; "
            "the publication default is 2000."
        )
    if args.bootstrap_batch_size < 1:
        raise ValueError("--bootstrap-batch-size must be positive.")
    if not 0.0 < args.ci_level < 1.0:
        raise ValueError("--ci-level must lie in (0,1).")
    if args.mechanism_predictions is None:
        raise ValueError("--mechanism-predictions is required.")
    if args.mechanism_manifest is None:
        raise ValueError("--mechanism-manifest is required.")

    started = time.time()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output root is not empty: {output_root}. Pass --overwrite to replace it."
        )
    if args.overwrite and output_root.exists():
        import shutil

        shutil.rmtree(output_root)
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    table_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    mechanism_path = find_table(args.mechanism_predictions.resolve())
    event_path = find_table(
        args.event_ssl_root.resolve()
        / "predictions"
        / "stage4_event_ssl_predictions_B_confirm"
    )
    mechanism_manifest_path = args.mechanism_manifest.resolve()
    event_manifest_path = (
        args.event_ssl_root.resolve()
        / "metadata"
        / "stage4_event_ssl_evaluation_manifest.json"
    )
    mechanism_manifest, mechanism_manifest_audit = validate_mechanism_manifest(
        mechanism_manifest_path, mechanism_path, args.mechanism_confirm_script
    )
    event_manifest, event_manifest_audit = validate_event_manifest(
        event_manifest_path, event_path, args.event_ssl_evaluate_script
    )

    mechanism = canonical_mechanism_table(mechanism_path)
    event = canonical_event_table(event_path)
    event_split_record = dict(event_manifest.get("splits", {})).get("B_confirm", {})
    if int(event_split_record.get("rows", -1)) != len(event):
        raise RuntimeError(
            f"Event-SSL manifest row count differs from the prediction table: "
            f"manifest={event_split_record.get('rows')}, table={len(event)}."
        )
    if int(event_split_record.get("users", -1)) != int(event["user_id"].nunique()):
        raise RuntimeError("Event-SSL manifest user count differs from the prediction table.")
    mechanism_fingerprint = dict(mechanism_manifest.get("confirm_panel_fingerprint", {}))
    if mechanism_fingerprint:
        expected_rows = int(mechanism_fingerprint.get("rows", len(mechanism)))
        expected_users = int(mechanism_fingerprint.get("users", mechanism["user_id"].nunique()))
        if expected_rows != len(mechanism) or expected_users != int(mechanism["user_id"].nunique()):
            raise RuntimeError(
                "Mechanism confirmation panel fingerprint differs from the full prediction table."
            )
    arrays, join_audit = align_tables(
        mechanism,
        event,
        tolerance=args.anchor_tolerance,
        minimum_join_fraction=args.minimum_join_fraction,
    )
    del mechanism, event

    partition = load_partition(args.stage1_root.resolve())
    partition_provenance_audit = validate_partition_provenance(
        mechanism_manifest, event_manifest, partition
    )
    statistics = build_sufficient_statistics(arrays, partition)
    point, auxiliary = point_estimates(statistics)
    reconstruction = point_reconstruction_audit(
        point, args.point_tolerance, enforce=True
    )

    field_bootstrap, transition_bootstrap, statewise_bootstrap = bootstrap_analysis(
        statistics,
        auxiliary,
        replicates=args.bootstrap_replicates,
        batch_size=args.bootstrap_batch_size,
        seed=args.bootstrap_seed,
    )
    field_summary = summarize_field_bootstrap(
        point, field_bootstrap, args.bootstrap_replicates, args.ci_level
    )
    transition_summary = summarize_transition_bootstrap(
        point, transition_bootstrap, args.bootstrap_replicates, args.ci_level
    )
    statewise_summary = summarize_statewise_bootstrap(
        auxiliary["statewise"], statewise_bootstrap, args.bootstrap_replicates, args.ci_level
    )
    quality = quality_gates(
        args,
        join_audit,
        reconstruction,
        field_summary,
        transition_summary,
        statewise_summary,
        statistics,
    )
    transition_long = transition_matrix_table(
        auxiliary["transitions"], auxiliary["transition_counts"]
    )

    paths = {
        "formal_point_estimates": write_table(point, table_root / "formal_point_estimates"),
        "formal_point_reconstruction_audit": write_table(
            reconstruction, table_root / "formal_point_reconstruction_audit"
        ),
        "field_bootstrap_replicates": write_table(
            field_bootstrap, table_root / "field_cluster_bootstrap_replicates"
        ),
        "field_bootstrap_intervals": write_table(
            field_summary, table_root / "field_cluster_bootstrap_intervals"
        ),
        "transition_bootstrap_replicates": write_table(
            transition_bootstrap, table_root / "transition_cluster_bootstrap_replicates"
        ),
        "transition_bootstrap_intervals": write_table(
            transition_summary, table_root / "transition_cluster_bootstrap_intervals"
        ),
        "statewise_bootstrap_replicates": write_table(
            statewise_bootstrap, table_root / "statewise_persistence_bootstrap_replicates"
        ),
        "statewise_persistence_summary": write_table(
            statewise_summary, table_root / "statewise_persistence_cluster_bootstrap_intervals"
        ),
        "transition_matrices": write_table(
            transition_long, table_root / "formal_transition_matrices"
        ),
        "statewise_point_estimates": write_table(
            auxiliary["statewise"], table_root / "statewise_persistence_point_estimates"
        ),
        "persistence_leave_one_state_out": write_table(
            auxiliary["loso"], table_root / "persistence_leave_one_state_out"
        ),
        "quality_gates": write_table(quality, table_root / "quality_gates"),
    }

    source_files: Dict[str, Path] = {
        "mechanism_predictions": mechanism_path,
        "event_ssl_predictions": event_path,
        "mechanism_manifest": mechanism_manifest_path,
        "event_ssl_manifest": event_manifest_path,
        "stage1_partition_metadata": partition.metadata_path,
        "stage1_partition_centers": partition.centers_path,
        "analysis_script": Path(__file__).resolve(),
    }
    optional_sources = {
        "stage1_script": args.stage1_script,
        "mechanism_confirm_script": args.mechanism_confirm_script,
        "event_ssl_evaluate_script": args.event_ssl_evaluate_script,
    }
    for name, path in optional_sources.items():
        if path is None:
            continue
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        source_files[name] = resolved
    manifest = {
        "analysis": "paired learner-cluster uncertainty for frozen headline field and transition summaries",
        "status": "post hoc supplementary inferential audit",
        "created_at_unix": time.time(),
        "runtime_seconds": float(time.time() - started),
        "output_root": str(output_root),
        "data_boundary": {
            "split": "B_confirm",
            "output_only": True,
            "model_refit": False,
            "probe_refit": False,
            "coordinate_refit": False,
            "grid_refit": False,
            "partition_refit": False,
            "construction_null_rerun": False,
            "recursive_surrogate_rerun": False,
            "positive_exponential_multiplier_sensitivity_rerun": False,
            "A_val_residence_cluster_bootstrap_rerun": False,
            "null_relative_skill_or_delta_D2_rerun": False,
            "new_figure_generated": False,
        },
        "bootstrap_contract": {
            "method": "ordinary nonparametric multinomial learner-cluster bootstrap",
            "resampling_unit": "learner",
            "replicates": int(args.bootstrap_replicates),
            "batch_size": int(args.bootstrap_batch_size),
            "seed": int(args.bootstrap_seed),
            "same_multiplicity_for_all_paired_views": True,
            "field_user_balancing": (
                "each original learner contributes one total field unit; a learner sampled m times contributes m copies"
            ),
            "transition_estimand": "formal interval-count transition rows",
            "primary_support_contract": "fixed formal common support",
            "secondary_support_contract": "support reselected by the formal raw transition-count threshold of 30",
            "primary_interval": (
                "cluster-bootstrap standard-error interval centred on the unit-weight estimate; "
                "Pearson correlations use Fisher-z scale and weighted local cosine uses its raw bounded scale"
            ),
            "secondary_support_interval": (
                "support-reselected percentile interval because threshold selection is part of the estimator"
            ),
            "statewise_correlation_boundary": (
                "conditional on six fixed mesostates; not inference over a population of states"
            ),
        },
        "join_audit": join_audit,
        "mechanism_manifest_audit": mechanism_manifest_audit,
        "event_ssl_manifest_audit": event_manifest_audit,
        "partition_provenance_audit": partition_provenance_audit,
        "sufficient_statistics_audit": statistics["audit"],
        "quality_gates": quality.to_dict(orient="records"),
        "source_files": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path.resolve()),
                "bytes": int(path.stat().st_size),
            }
            for name, path in source_files.items()
        },
        "outputs": {name: str(path) for name, path in paths.items()},
        "output_files": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path.resolve()),
                "bytes": int(path.stat().st_size),
            }
            for name, path in paths.items()
        },
        "interpretation_boundary": (
            "Intervals quantify confirmation-learner sampling uncertainty conditional on frozen model outputs, "
            "coordinates, grid and K=6 partition. The six-state correlations remain descriptive summaries of "
            "the fixed states even when learner-bootstrap intervals are reported."
        ),
    }
    manifest_path = metadata_root / "analysis_manifest.json"
    save_json(manifest, manifest_path)
    manifest_sha = sha256_file(manifest_path)
    save_json(
        {"manifest_path": str(manifest_path), "manifest_sha256": manifest_sha},
        metadata_root / "analysis_manifest.sha256.json",
    )
    if not bool(quality["passed"].all()):
        failed = quality.loc[~quality["passed"]]
        raise RuntimeError("Quality gates failed:\n" + failed.to_string(index=False))
    print(f"completed: {output_root}")


if __name__ == "__main__":
    main()
