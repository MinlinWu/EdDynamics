#!/usr/bin/env python3
"""Render publication figures for the frozen EdNet-KT4 minimal mechanism."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

EPS = 1e-12
GRID_BINS_SIGNED = np.linspace(-1.0, 1.0, int(os.environ.get("EDNET_STAGE1_SIGNED_GRID_N", "41")))
POTENTIAL_CLIP_Q = float(os.environ.get("EDNET_PUB_POTENTIAL_CLIP_Q", "0.98"))
POTENTIAL_RESIDUAL_VLIM = float(os.environ.get("EDNET_MINMECH_PUB_POTENTIAL_RESIDUAL_VLIM", "25"))
DRIFT_RESIDUAL_VMAX = float(os.environ.get("EDNET_MINMECH_PUB_DRIFT_RESIDUAL_VMAX", "0.20"))
TRANSITION_RESIDUAL_VLIM = float(os.environ.get("EDNET_MINMECH_PUB_TRANSITION_RESIDUAL_VLIM", "0.30"))
MIN_RESIDENCE_AT_RISK = int(os.environ.get("EDNET_PUB_MIN_RESIDENCE_AT_RISK", "20"))
MAX_RESIDENCE_LEN_FOR_PLOT = int(os.environ.get("EDNET_PUB_MAX_RESIDENCE_LEN", "10000"))
MECHANISM_POTENTIAL_SCALE_SPLIT = os.environ.get("EDNET_MINMECH_PUB_POTENTIAL_SCALE_SPLIT", "B_confirm")

DEFAULT_ROOT = Path(os.environ.get("EDNET_KT4_OUTPUT_ROOT", "/data/datasets/KT4/outputs_KT4"))
DEFAULT_STAGE1_ROOT = DEFAULT_ROOT / "stage1"
DEFAULT_PHASE2_ROOT = DEFAULT_ROOT / "stage2_phase2_freeze"
DEFAULT_PHASE3_ROOT = DEFAULT_ROOT / "stage2_phase3_confirm"
DEFAULT_MINIMALITY_ROOT = DEFAULT_ROOT / "stage2_phase1_unified_minimality"

EXPECTED_MACROSTATE_K = 6
EXPECTED_FEATURES = [
    "M_response_prebalanced_pre",
    "activity_alignment_order_Psi_pre",
]

BLUE_CMAP = LinearSegmentedColormap.from_list(
    "ednet_light_to_deep_blue",
    ["#f7fbff", "#deebf7", "#9ecae1", "#4292c6", "#08519c", "#08306b"],
)
BLUE_CMAP_REVERSED = LinearSegmentedColormap.from_list(
    "ednet_deep_to_light_blue",
    ["#08306b", "#08519c", "#4292c6", "#9ecae1", "#deebf7", "#f7fbff"],
)
STATE_LINE_CMAP = LinearSegmentedColormap.from_list(
    "ednet_state_blues",
    ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"],
)
RESIDUAL_CMAP = LinearSegmentedColormap.from_list(
    "ednet_blue_orange_residual",
    ["#8c2d04", "#fdd0a2", "#f7fbff", "#bdd7e7", "#08519c"],
)

plt.rcParams.update({
    "font.size": 9.2,
    "axes.titlesize": 10.8,
    "axes.labelsize": 9.2,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 8.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.85,
})


@dataclass(frozen=True)
class FrozenMacroPartition:
    k: int
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    ordered_centers: np.ndarray
    ordered_centers_scaled: np.ndarray
    raw_centers_scaled: np.ndarray
    raw_to_ordered: np.ndarray
    centers_table: pd.DataFrame
    fit_table: pd.DataFrame
    audit: Mapping[str, object]


@dataclass(frozen=True)
class FieldStats:
    u: np.ndarray
    v: np.ndarray
    mask: np.ndarray
    count: np.ndarray
    weight: np.ndarray


def path_candidates(base_or_path: Path) -> List[Path]:
    path = Path(base_or_path)
    if path.suffix:
        return [path]
    return [path.with_suffix(".parquet"), path.with_suffix(".csv.gz"), path.with_suffix(".csv")]


def read_table(base_or_path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    for path in path_candidates(base_or_path):
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
        return pd.read_csv(path, usecols=list(columns) if columns is not None else None, low_memory=False)
    raise FileNotFoundError(f"Could not find table for {base_or_path}")


def existing_table_path(base_or_path: Path) -> Path:
    for path in path_candidates(base_or_path):
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Could not find table for {base_or_path}")


def write_table(df: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = base.with_suffix(".parquet")
        df.to_parquet(path, index=False)
        return path
    except Exception:
        path = base.with_suffix(".csv.gz")
        df.to_csv(path, index=False, compression="gzip")
        return path


def load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def json_sanitize(obj):
    if isinstance(obj, Mapping):
        return {str(key): json_sanitize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return json_sanitize(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_sanitize(obj), handle, indent=2, ensure_ascii=False, allow_nan=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def savefig(fig: plt.Figure, path: Path, formats: Sequence[str] = ("png", "pdf", "svg")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_phase3_no_update_audit(phase3_root: Path) -> Mapping[str, object]:
    path = phase3_root / "metadata" / "phase3_no_update_audit.json"
    if not path.exists():
        raise FileNotFoundError(f"Phase-3 no-update audit not found: {path}")
    audit = dict(load_json(path))
    required_false = (
        "parameter_search_opened",
        "calibration_reestimated",
        "mechanism_family_reselected",
        "mechanism_parameters_refit",
        "kmeans_refit",
        "kmeans_k_selection",
        "macrostate_k_selected",
        "region_redefinition",
        "B_confirm_used_for_model_update",
    )
    failures = {name: audit.get(name) for name in required_false if bool(audit.get(name, False))}
    for before, after in (
        ("frozen_parameter_hash_before_confirmation", "frozen_parameter_hash_after_confirmation"),
        ("frozen_calibration_hash_before_confirmation", "frozen_calibration_hash_after_confirmation"),
    ):
        if before in audit and after in audit and str(audit[before]) != str(audit[after]):
            failures[f"{before}!={after}"] = True
    if failures:
        raise RuntimeError(f"Phase-3 no-update audit failed: {failures}")
    audit["path"] = str(path.resolve())
    audit["sha256"] = file_sha256(path)
    return audit


def load_phase3_confirmation_manifest(phase3_root: Path) -> Mapping[str, object]:
    path = phase3_root / "metadata" / "phase3_confirmation_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Phase-3 confirmation manifest not found: {path}")
    manifest = dict(load_json(path))
    if tuple(manifest.get("primary_macrostate", [])) != ("M", "Psi"):
        raise RuntimeError("Phase-3 primary macrostate must be ['M', 'Psi'].")
    if str(manifest.get("confirm_split", "")) != "B_confirm":
        raise RuntimeError("Publication figures require the B_confirm output-only split.")
    guardrails = dict(manifest.get("guardrails", {}))
    required_false = (
        "parameter_search_opened",
        "calibration_reestimated",
        "mechanism_family_reselected",
        "mechanism_parameters_refit",
        "kmeans_refit",
        "kmeans_k_selection",
        "macrostate_k_selected",
        "region_redefinition",
        "B_confirm_used_for_update",
    )
    failures = {name: guardrails.get(name) for name in required_false if bool(guardrails.get(name, False))}
    if failures:
        raise RuntimeError(f"Phase-3 manifest guardrails failed: {failures}")
    manifest["path"] = str(path.resolve())
    manifest["sha256"] = file_sha256(path)
    return manifest


def load_phase2_frozen_manifest(phase2_root: Path) -> Mapping[str, object]:
    path = phase2_root / "metadata" / "phase2_frozen_model_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Phase-2 frozen manifest not found: {path}")
    manifest = dict(load_json(path))
    if tuple(manifest.get("primary_macrostate", [])) != ("M", "Psi"):
        raise RuntimeError("Phase-2 primary macrostate must be ['M', 'Psi'].")
    frozen = dict(manifest.get("frozen_parameters", {}))
    if str(frozen.get("family_key", "")) != "offset_dual_channel":
        raise RuntimeError("Phase-2 frozen family is not offset_dual_channel.")
    manifest["path"] = str(path.resolve())
    manifest["sha256"] = file_sha256(path)
    return manifest

def validate_source_roots(
    stage1_root: Path,
    minimality_root: Path,
    phase2_manifest: Mapping[str, object],
    phase3_manifest: Mapping[str, object],
) -> Mapping[str, object]:
    for label, manifest in (("Phase 2", phase2_manifest), ("Phase 3", phase3_manifest)):
        recorded = Path(str(manifest.get("stage1_root", ""))).resolve()
        if recorded != stage1_root:
            raise RuntimeError(f"{label} Stage-1 root differs from the requested publication source.")
    handoff_path = minimality_root / "metadata" / "phase1_minimal_mechanism_handoff.json"
    handoff = dict(load_json(handoff_path))
    if handoff.get("ready_for_phase2_freeze") is not True or handoff.get("final_family_key") != "offset_dual_channel":
        raise RuntimeError("The publication minimality handoff is not the ready offset_dual_channel result.")
    actual_handoff_sha = file_sha256(handoff_path)
    expected_handoff_sha = str(phase2_manifest.get("phase1_minimality_handoff_sha256", "") or "")
    if expected_handoff_sha and expected_handoff_sha != actual_handoff_sha:
        raise RuntimeError("The publication minimality handoff differs from the Phase-2 frozen source.")
    return {
        "stage1_root": str(stage1_root),
        "minimality_handoff_path": str(handoff_path.resolve()),
        "minimality_handoff_sha256": actual_handoff_sha,
    }


def load_phase3_formal_metrics(phase3_root: Path, confirm_split: str) -> Tuple[Dict[str, float], Path]:
    base = phase3_root / "tables" / f"phase3_{confirm_split}_structural_alignment_metrics"
    table = read_table(base)
    if table.empty:
        raise RuntimeError(f"Phase-3 structural metric table is empty: {base}")
    values: Dict[str, float] = {}
    for key, value in table.iloc[0].to_dict().items():
        try:
            number = float(value)
        except Exception:
            continue
        if np.isfinite(number):
            values[str(key)] = number
    return values, existing_table_path(base)


def locate_full_predictions(phase3_root: Path, confirm_split: str, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    base = phase3_root / "tables" / f"phase3_{confirm_split}_full_predictions"
    try:
        return existing_table_path(base)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Full Phase-3 predictions are required for the one-step closure panel. "
            "Rerun Phase 3 with --write-full-predictions."
        ) from exc


def load_confirmation_predictions(phase3_root: Path, confirm_split: str, explicit: Optional[Path]) -> Tuple[pd.DataFrame, Path]:
    path = locate_full_predictions(phase3_root, confirm_split, explicit)
    required = [
        "user_id",
        "bundle_step_index",
        "M",
        "Psi",
        "target_M_next",
        "target_Psi_next",
        "pred_next_M",
        "pred_next_Psi",
    ]
    table = read_table(path)
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise RuntimeError(f"Phase-3 prediction table is missing columns: {missing}")
    table = table[required].copy()
    for name in required:
        table[name] = pd.to_numeric(table[name], errors="coerce")
    valid = table["user_id"].notna() & table["bundle_step_index"].notna()
    valid &= np.isfinite(table[["M", "Psi", "target_M_next", "target_Psi_next", "pred_next_M", "pred_next_Psi"]]).all(axis=1)
    table = table.loc[valid].copy()
    table["user_id"] = table["user_id"].astype(np.int64)
    table["bundle_step_index"] = table["bundle_step_index"].astype(np.int64)
    table = table.sort_values(["user_id", "bundle_step_index"], kind="mergesort").reset_index(drop=True)
    return table, path


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(float(np.nansum(p)), eps)
    q = q / max(float(np.nansum(q)), eps)
    middle = 0.5 * (p + q)
    mask_p = p > 0
    mask_q = q > 0
    kl_p = float(np.sum(p[mask_p] * np.log((p[mask_p] + eps) / (middle[mask_p] + eps))))
    kl_q = float(np.sum(q[mask_q] * np.log((q[mask_q] + eps) / (middle[mask_q] + eps))))
    return 0.5 * (kl_p + kl_q)


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return np.nan
    aa = a[valid] - float(np.mean(a[valid]))
    bb = b[valid] - float(np.mean(b[valid]))
    denominator = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denominator) if denominator > EPS else np.nan


def row_tv(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    if p.shape != q.shape or p.size == 0:
        return np.asarray([], dtype=float)
    return 0.5 * np.nansum(np.abs(p - q), axis=1)


def clip_for_potential(values: np.ndarray, q: float = POTENTIAL_CLIP_Q) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    finite = output[np.isfinite(output)]
    if finite.size:
        cap = float(np.nanquantile(finite, min(max(q, 0.50), 1.0)))
        output = np.minimum(output, cap)
    return output


def _pivot_grid(table: pd.DataFrame, value: str) -> np.ndarray:
    pivot = table.pivot(index="M_center", columns="Psi_center", values=value)
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    expected = 0.5 * (GRID_BINS_SIGNED[:-1] + GRID_BINS_SIGNED[1:])
    if pivot.shape != (len(expected), len(expected)):
        raise RuntimeError(f"Unexpected field-grid shape for {value}: {pivot.shape}")
    if not np.allclose(pivot.index.to_numpy(dtype=float), expected, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"Unexpected M grid for {value}.")
    if not np.allclose(pivot.columns.to_numpy(dtype=float), expected, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"Unexpected Psi grid for {value}.")
    return pivot.to_numpy(dtype=float)


def load_confirmation_fields(phase3_root: Path, confirm_split: str) -> Tuple[Dict[str, object], Path]:
    base = phase3_root / "tables" / f"phase3_{confirm_split}_field_grid"
    table = read_table(base)
    required = {
        "M_center",
        "Psi_center",
        "empirical_next_occupancy",
        "mechanism_next_occupancy",
        "empirical_drift_M",
        "empirical_drift_Psi",
        "mechanism_drift_M",
        "mechanism_drift_Psi",
        "empirical_supported",
        "mechanism_supported",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise RuntimeError(f"Phase-3 field table is missing columns: {missing}")
    empirical_mask = _pivot_grid(table, "empirical_supported").astype(bool)
    mechanism_mask = _pivot_grid(table, "mechanism_supported").astype(bool)
    empirical = FieldStats(
        u=_pivot_grid(table, "empirical_drift_M"),
        v=_pivot_grid(table, "empirical_drift_Psi"),
        mask=empirical_mask,
        count=empirical_mask.astype(float),
        weight=empirical_mask.astype(float),
    )
    mechanism = FieldStats(
        u=_pivot_grid(table, "mechanism_drift_M"),
        v=_pivot_grid(table, "mechanism_drift_Psi"),
        mask=mechanism_mask,
        count=mechanism_mask.astype(float),
        weight=mechanism_mask.astype(float),
    )
    empirical_occupancy = _pivot_grid(table, "empirical_next_occupancy")
    mechanism_occupancy = _pivot_grid(table, "mechanism_next_occupancy")
    common = empirical.mask & mechanism.mask
    if common.any():
        empirical_vector = np.column_stack([empirical.u[common], empirical.v[common]]).ravel()
        mechanism_vector = np.column_stack([mechanism.u[common], mechanism.v[common]]).ravel()
        residual = np.sqrt((mechanism.u[common] - empirical.u[common]) ** 2 + (mechanism.v[common] - empirical.v[common]) ** 2)
        speed_emp = np.sqrt(empirical.u[common] ** 2 + empirical.v[common] ** 2)
        speed_mech = np.sqrt(mechanism.u[common] ** 2 + mechanism.v[common] ** 2)
        direct_metrics = {
            "common_drift_cells": int(common.sum()),
            "drift_vector_corr": pearson_safe(empirical_vector, mechanism_vector),
            "drift_local_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "drift_speed_corr": pearson_safe(speed_emp, speed_mech),
        }
    else:
        direct_metrics = {
            "common_drift_cells": 0,
            "drift_vector_corr": np.nan,
            "drift_local_rmse": np.nan,
            "drift_speed_corr": np.nan,
        }
    return {
        "f_emp": empirical,
        "f_mech": mechanism,
        "H_emp_next": empirical_occupancy,
        "H_mech_next": mechanism_occupancy,
        "metrics": {
            "next_state_occupancy_js": js_divergence(empirical_occupancy + EPS, mechanism_occupancy + EPS),
            **direct_metrics,
        },
    }, existing_table_path(base)


def load_empirical_current_potential_clim(stage1_root: Path, split: str) -> Tuple[float, float, Path]:
    suffix = "_output_only" if split == "B_confirm" else ""
    base = (
        stage1_root
        / "dynamics"
        / "coordinate_analysis"
        / "MR_PsiA"
        / f"{split}_publication_field_grid{suffix}"
    )
    table = read_table(base)
    if "empirical_quasi_potential" not in table.columns:
        raise RuntimeError(f"Stage-1 publication field grid lacks empirical_quasi_potential: {base}")
    values = pd.to_numeric(table["empirical_quasi_potential"], errors="coerce").to_numpy(dtype=float)
    values = clip_for_potential(values, POTENTIAL_CLIP_Q)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise RuntimeError(f"Stage-1 potential grid is empty: {base}")
    return float(np.min(finite)), float(np.max(finite)), existing_table_path(base)


def _contract_from_phase3_manifest(manifest: Mapping[str, object]) -> Mapping[str, object]:
    contract = manifest.get("stage1_fixed_k6_contract", {})
    if isinstance(contract, Mapping) and isinstance(contract.get("current_contract"), Mapping):
        return contract["current_contract"]
    return contract if isinstance(contract, Mapping) else {}


def load_frozen_macro_partition(
    stage1_root: Path,
    phase2_manifest: Mapping[str, object],
    phase3_manifest: Mapping[str, object],
) -> FrozenMacroPartition:
    root = stage1_root / "dynamics" / "fixed_k6_mesostates"
    metadata_path = root / "fixed_k6_model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = dict(load_json(metadata_path))
    checks = {
        "coordinate": metadata.get("coordinate") == "MR_PsiA",
        "macrostate_k": int(metadata.get("macrostate_k", -1)) == EXPECTED_MACROSTATE_K,
        "macrostate_k_rule": metadata.get("macrostate_k_rule") == "fixed a priori",
        "features": list(metadata.get("features", [])) == EXPECTED_FEATURES,
        "fit_split": metadata.get("fit_split") == "A_train",
        "user_balanced_sampling": metadata.get("user_balanced_sampling") is True,
        "user_balanced_kmeans_fit": metadata.get("user_balanced_kmeans_fit") is True,
        "kmeans_n_init": int(metadata.get("kmeans_n_init", -1)) == 20,
        "fit_max_rows": int(metadata.get("fit_max_rows", -1)) == 500000,
        "random_state": int(metadata.get("random_state", -1)) == 42,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Stage-1 fixed K=6 contract failed: {failed}")

    centers_path = existing_table_path(root / "fixed_k6_centers")
    centers = read_table(centers_path).sort_values("macrostate", kind="mergesort").reset_index(drop=True)
    required = {"macrostate", "center_M", "center_Psi"}
    missing = sorted(required.difference(centers.columns))
    if missing:
        raise RuntimeError(f"Fixed K=6 center table is missing columns: {missing}")
    ids = pd.to_numeric(centers["macrostate"], errors="coerce").to_numpy(dtype=int)
    if not np.array_equal(ids, np.arange(EXPECTED_MACROSTATE_K)):
        raise RuntimeError("Fixed K=6 centers are not ordered S0-S5.")
    ordered_centers = centers[["center_M", "center_Psi"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    scaler_mean = np.asarray(metadata.get("scaler_mean", []), dtype=float)
    scaler_scale = np.asarray(metadata.get("scaler_scale", []), dtype=float)
    if scaler_mean.shape != (2,) or scaler_scale.shape != (2,) or not np.all(scaler_scale > 0):
        raise RuntimeError("Invalid fixed K=6 scaler metadata.")
    if {"center_M_standardized", "center_Psi_standardized"}.issubset(centers.columns):
        ordered_scaled = centers[["center_M_standardized", "center_Psi_standardized"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    else:
        ordered_scaled = (ordered_centers - scaler_mean) / scaler_scale
    mapping_dict = {int(key): int(value) for key, value in dict(metadata.get("raw_to_ordered_label", {})).items()}
    if sorted(mapping_dict) != list(range(EXPECTED_MACROSTATE_K)) or sorted(mapping_dict.values()) != list(range(EXPECTED_MACROSTATE_K)):
        raise RuntimeError("Invalid raw-to-ordered KMeans label map.")
    raw_to_ordered = np.asarray([mapping_dict[index] for index in range(EXPECTED_MACROSTATE_K)], dtype=int)
    raw_scaled = np.empty_like(ordered_scaled)
    for raw, ordered in enumerate(raw_to_ordered):
        raw_scaled[raw] = ordered_scaled[ordered]

    metadata_sha = file_sha256(metadata_path)
    centers_sha = file_sha256(centers_path)
    for label, contract in (
        ("Phase 2", phase2_manifest.get("stage1_fixed_k6_contract", {})),
        ("Phase 3", _contract_from_phase3_manifest(phase3_manifest)),
    ):
        if isinstance(contract, Mapping):
            expected_metadata = str(contract.get("metadata_sha256", "") or "")
            expected_centers = str(contract.get("centers_sha256", "") or "")
            if expected_metadata and expected_metadata != metadata_sha:
                raise RuntimeError(f"{label} fixed-K metadata hash does not match Stage 1.")
            if expected_centers and expected_centers != centers_sha:
                raise RuntimeError(f"{label} fixed-K center hash does not match Stage 1.")

    fit_table = read_table(root / "fixed_k6_fit_table")
    audit = {
        "source": "frozen Stage-1 fixed K=6 partition",
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": metadata_sha,
        "centers_path": str(centers_path),
        "centers_sha256": centers_sha,
        "checks": checks,
        "kmeans_refit": False,
        "macrostate_k_selected": False,
    }
    return FrozenMacroPartition(
        k=EXPECTED_MACROSTATE_K,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        ordered_centers=ordered_centers,
        ordered_centers_scaled=ordered_scaled,
        raw_centers_scaled=raw_scaled,
        raw_to_ordered=raw_to_ordered,
        centers_table=centers,
        fit_table=fit_table,
        audit=audit,
    )


def predict_macro_labels(partition: FrozenMacroPartition, M: np.ndarray, Psi: np.ndarray) -> np.ndarray:
    values = np.column_stack([np.asarray(M, dtype=float), np.asarray(Psi, dtype=float)])
    labels = np.full(values.shape[0], -1, dtype=np.int64)
    valid = np.isfinite(values).all(axis=1)
    if not valid.any():
        return labels
    standardized = (values[valid] - partition.scaler_mean) / partition.scaler_scale
    distances = np.sum((standardized[:, None, :] - partition.raw_centers_scaled[None, :, :]) ** 2, axis=2)
    raw = np.argmin(distances, axis=1)
    labels[valid] = partition.raw_to_ordered[raw]
    return labels


def normalize_transition(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    row_sum = counts.sum(axis=1, keepdims=True)
    output = np.zeros_like(counts, dtype=float)
    valid = row_sum[:, 0] > 0
    output[valid] = counts[valid] / row_sum[valid]
    return output


def transition_counts_from_current_next(current: np.ndarray, next_state: np.ndarray, k: int) -> np.ndarray:
    current = np.asarray(current, dtype=np.int64)
    next_state = np.asarray(next_state, dtype=np.int64)
    valid = (current >= 0) & (current < k) & (next_state >= 0) & (next_state < k)
    if not valid.any():
        return np.zeros((k, k), dtype=float)
    flat = current[valid] * k + next_state[valid]
    return np.bincount(flat, minlength=k * k).reshape(k, k).astype(float)


def load_stage1_mesostate_tables(stage1_root: Path) -> Dict[str, object]:
    root = stage1_root / "dynamics" / "fixed_k6_mesostates"
    assignments_path = existing_table_path(root / "B_confirm_fixed_k6_assignments")
    assignments = read_table(assignments_path)
    state_column = "macrostate" if "macrostate" in assignments.columns else "state" if "state" in assignments.columns else None
    if state_column is None:
        raise RuntimeError("B_confirm fixed-K assignment table lacks macrostate labels.")
    assignments = assignments[["user_id", "bundle_step_index", state_column]].rename(columns={state_column: "current_macrostate"})
    assignments["user_id"] = pd.to_numeric(assignments["user_id"], errors="coerce")
    assignments["bundle_step_index"] = pd.to_numeric(assignments["bundle_step_index"], errors="coerce")
    assignments["current_macrostate"] = pd.to_numeric(assignments["current_macrostate"], errors="coerce")
    assignments = assignments.dropna().copy()
    assignments[["user_id", "bundle_step_index", "current_macrostate"]] = assignments[["user_id", "bundle_step_index", "current_macrostate"]].astype(np.int64)

    counts_path = existing_table_path(root / "B_confirm_fixed_k6_transition_counts")
    matrix_path = existing_table_path(root / "B_confirm_fixed_k6_transition_matrix")
    counts = read_table(counts_path).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    matrix = read_table(matrix_path).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if counts.shape != (EXPECTED_MACROSTATE_K, EXPECTED_MACROSTATE_K) or matrix.shape != counts.shape:
        raise RuntimeError("Unexpected B_confirm fixed-K transition shape.")

    curves_path = existing_table_path(root / "B_confirm_fixed_k6_residence_curves")
    curves = read_table(curves_path)
    required_curves = {"macrostate", "residence_length", "km_ccdf", "at_risk", "geometric_ccdf"}
    missing_curves = sorted(required_curves.difference(curves.columns))
    if missing_curves:
        raise RuntimeError(f"Residence curve table is missing columns: {missing_curves}")
    summary_path = existing_table_path(root / "B_confirm_fixed_k6_residence_summary")
    summary = read_table(summary_path)
    return {
        "assignments": assignments,
        "assignments_path": assignments_path,
        "transition_counts": counts,
        "transition_counts_path": counts_path,
        "transition_matrix": matrix,
        "transition_matrix_path": matrix_path,
        "residence_curves": curves,
        "residence_curves_path": curves_path,
        "residence_summary": summary,
        "residence_summary_path": summary_path,
    }


def kinetic_recovery(
    predictions: pd.DataFrame,
    partition: FrozenMacroPartition,
    stage1_tables: Mapping[str, object],
    max_len: int,
    residence_metric_max_len: int,
) -> Dict[str, object]:
    assignments = stage1_tables["assignments"]
    joined = predictions.merge(
        assignments,
        on=["user_id", "bundle_step_index"],
        how="left",
        validate="one_to_one",
    )
    join_fraction = float(joined["current_macrostate"].notna().mean()) if len(joined) else 0.0
    if join_fraction < 1.0 - 1e-12:
        raise RuntimeError(f"Only {join_fraction:.6%} of Phase-3 rows matched Stage-1 fixed-K assignments.")
    current = joined["current_macrostate"].to_numpy(dtype=np.int64)
    empirical_next = predict_macro_labels(
        partition,
        joined["target_M_next"].to_numpy(dtype=float),
        joined["target_Psi_next"].to_numpy(dtype=float),
    )
    mechanism_next = predict_macro_labels(
        partition,
        joined["pred_next_M"].to_numpy(dtype=float),
        joined["pred_next_Psi"].to_numpy(dtype=float),
    )
    empirical_counts = transition_counts_from_current_next(current, empirical_next, partition.k)
    empirical_matrix = normalize_transition(empirical_counts)
    stage1_empirical_counts = np.asarray(stage1_tables["transition_counts"], dtype=float)
    stage1_empirical_matrix = np.asarray(stage1_tables["transition_matrix"], dtype=float)
    reconstruction_difference = float(np.max(np.abs(empirical_matrix - stage1_empirical_matrix)))
    mechanism_counts = transition_counts_from_current_next(current, mechanism_next, partition.k)
    mechanism_matrix = normalize_transition(mechanism_counts)

    tv = row_tv(empirical_matrix, mechanism_matrix)
    diag_emp = np.diag(empirical_matrix)
    diag_mech = np.diag(mechanism_matrix)
    diag_rmse = float(np.sqrt(np.mean((diag_emp - diag_mech) ** 2)))
    diag_mae = float(np.mean(np.abs(diag_emp - diag_mech)))
    diag_corr = pearson_safe(diag_emp, diag_mech)
    top_n = partition.k
    top_emp = np.argsort(empirical_matrix.ravel())[::-1][:top_n]
    top_mech = np.argsort(mechanism_matrix.ravel())[::-1][:top_n]
    top_overlap = float(len(set(top_emp.tolist()).intersection(set(top_mech.tolist()))) / top_n)

    row_emp = empirical_counts.sum(axis=1) > 0
    row_mech = mechanism_counts.sum(axis=1) > 0
    emp_flags = row_emp & (np.diag(empirical_matrix) >= np.max(empirical_matrix, axis=1) - 1e-12)
    mech_flags = row_mech & (np.diag(mechanism_matrix) >= np.max(mechanism_matrix, axis=1) - 1e-12)

    curves = stage1_tables["residence_curves"].copy()
    curves["macrostate"] = pd.to_numeric(curves["macrostate"], errors="coerce")
    curves["residence_length"] = pd.to_numeric(curves["residence_length"], errors="coerce")
    curves["at_risk"] = pd.to_numeric(curves["at_risk"], errors="coerce")
    curves = curves.dropna(subset=["macrostate", "residence_length", "at_risk"]).copy()
    observed_max = int(curves["residence_length"].max()) if not curves.empty else 1
    metric_max = int(max(1, min(max_len, residence_metric_max_len, observed_max)))
    log_differences: List[float] = []
    points_used = 0
    for state in range(partition.k):
        state_curves = curves[
            (curves["macrostate"].astype(int) == state)
            & (curves["residence_length"] <= metric_max)
            & (curves["at_risk"] >= MIN_RESIDENCE_AT_RISK)
        ]
        if state_curves.empty:
            continue
        lengths = state_curves["residence_length"].to_numpy(dtype=int)
        pe = float(np.clip(empirical_matrix[state, state], 1e-9, 1.0 - 1e-9))
        pm = float(np.clip(mechanism_matrix[state, state], 1e-9, 1.0 - 1e-9))
        log_differences.extend(np.abs((lengths - 1) * (math.log(pm) - math.log(pe))).tolist())
        points_used += int(len(lengths))
    residence_log_difference = float(np.mean(log_differences)) if log_differences else np.nan

    metrics = {
        "macrostate_k": int(partition.k),
        "transition_count": int(empirical_counts.sum()),
        "transition_mean_row_tv": float(np.mean(tv)),
        "transition_max_row_tv": float(np.max(tv)),
        "self_transition_rmse": diag_rmse,
        "self_transition_mae": diag_mae,
        "self_transition_correlation": diag_corr,
        "diagonal_dominant_states_empirical": int(emp_flags.sum()),
        "diagonal_dominant_states_mechanism": int(mech_flags.sum()),
        "diagonal_dominance_recall": float(np.sum(emp_flags & mech_flags) / max(int(emp_flags.sum()), 1)),
        "diagonal_dominance_match_fraction": float(np.mean(emp_flags == mech_flags)),
        "top_transition_edge_overlap": top_overlap,
        "residence_reference_mean_abs_log_difference": residence_log_difference,
        "residence_reference_concordance": float(np.exp(-residence_log_difference)) if np.isfinite(residence_log_difference) else np.nan,
        "residence_reference_points_used": int(points_used),
        "residence_observed_max_length": int(observed_max),
        "residence_metric_max_length": int(metric_max),
        "residence_source": "Stage-1 B_confirm fixed-K Kaplan-Meier curves",
        "partition_assignment_join_fraction": join_fraction,
        "stage1_full_vs_phase3_valid_empirical_transition_max_abs_difference": reconstruction_difference,
        "stage1_full_transition_count": int(stage1_empirical_counts.sum()),
    }
    return {
        "current_states": current,
        "emp_next_states": empirical_next,
        "mech_next_states": mechanism_next,
        "C_emp": empirical_counts,
        "C_mech": mechanism_counts,
        "P_emp": empirical_matrix,
        "P_mech": mechanism_matrix,
        "residence_curves": curves,
        "metrics": metrics,
    }


PARAMETER_MATH_LABELS = {
    "theta0": r"$\theta_0$",
    "thetaM": r"$\theta_M$",
    "thetaPsi": r"$\theta_\Psi$",
    "thetaMPsi": r"$\theta_{M\Psi}$",
    "phi0": r"$\phi_0$",
    "deltaS": r"$\Delta_S$",
    "phiPsi": r"$\phi_\Psi$",
    "lambdaR": r"$\lambda_R$",
    "lambdaA": r"$\lambda_A$",
    "lambdaI": r"$\lambda_I$",
}

STABILITY_PANEL_METRICS = [
    ("one_step_mse_main_norm", "Closure loss"),
    ("occupancy_js_MR_PsiA", "Landscape JS"),
    ("drift_local_rmse_loss_MR_PsiA", "Flow residual"),
    ("drift_direction_loss_MR_PsiA", "Flow direction"),
    ("drift_magnitude_loss_MR_PsiA", "Flow speed"),
]



def publication_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#334155")
    ax.tick_params(length=3, width=0.7)


def aligned_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    mappable,
    label: str,
    *,
    width: str = "3.0%",
    pad: float = 0.055,
    labelpad: float = 1.6,
    shrink_ticks: bool = True,
):
    """Colorbar with height aligned to the parent axes, matching Figure 4 style."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=width, pad=pad)
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label, labelpad=labelpad)
    if shrink_ticks:
        cb.ax.tick_params(labelsize=7.6)
    return cb


def draw_potential_difference(ax, H_emp: np.ndarray, H_mech: np.ndarray, title: str, colorbar: bool = True, residual_vlim: float = POTENTIAL_RESIDUAL_VLIM) -> None:
    U_emp = -np.log(H_emp + EPS)
    U_mech = -np.log(H_mech + EPS)
    D = U_mech - U_emp
    positive = np.concatenate([H_emp[H_emp > 0].ravel(), H_mech[H_mech > 0].ravel()])
    if positive.size:
        min_mass = max(float(np.nanquantile(positive, 0.01)), EPS)
        support = (H_emp >= min_mass) | (H_mech >= min_mass)
    else:
        support = np.zeros_like(H_emp, dtype=bool)
    D = np.where(support, D, np.nan)
    finite = D[np.isfinite(D)]
    if residual_vlim is not None and np.isfinite(residual_vlim) and residual_vlim > 0:
        lim = float(residual_vlim)
    else:
        lim = max(float(np.nanquantile(np.abs(finite), 0.98)) if finite.size else 1.0, 1e-6)
    cmap = RESIDUAL_CMAP.copy(); cmap.set_bad("#f1f5f9")
    ax.set_facecolor("#f1f5f9")
    mesh = ax.pcolormesh(GRID_BINS_SIGNED, GRID_BINS_SIGNED, D.T, shading="auto", cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_title(title)
    ax.set_xlabel(r"Response evidence order $M$")
    ax.set_ylabel(r"Exposure-alignment order $\Psi$")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    publication_axes(ax)
    if colorbar:
        cb = aligned_colorbar(ax.figure, ax, mesh, r"mechanism $-$ empirical $U_{\mathrm{next}}$", width="4.0%", pad=0.050, labelpad=1.0)
        cb.set_ticks([-lim, 0.0, lim])


def draw_mechanism_drift(ax, f_mech: FieldStats, H_mech_next: np.ndarray, title: str, colorbar: bool = True, potential_clim: Optional[Tuple[float, float]] = None) -> None:
    U = clip_for_potential(-np.log(H_mech_next + EPS), POTENTIAL_CLIP_Q)
    vmin = vmax = None
    if potential_clim is not None:
        lo, hi = potential_clim
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            vmin, vmax = float(lo), float(hi)
    mesh = ax.pcolormesh(GRID_BINS_SIGNED, GRID_BINS_SIGNED, U.T, shading="auto", cmap=BLUE_CMAP_REVERSED, alpha=0.62, vmin=vmin, vmax=vmax)
    xc = 0.5 * (GRID_BINS_SIGNED[:-1] + GRID_BINS_SIGNED[1:])
    yc = 0.5 * (GRID_BINS_SIGNED[:-1] + GRID_BINS_SIGNED[1:])
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    mask = f_mech.mask
    if np.any(mask):
        ax.quiver(
            X[mask], Y[mask], f_mech.u[mask], f_mech.v[mask],
            angles="xy", scale_units="xy", scale=1.0, width=0.0025,
            color="#111827", alpha=0.88,
        )
    ax.set_title(title)
    ax.set_xlabel(r"Response evidence order $M$")
    ax.set_ylabel(r"Exposure-alignment order $\Psi$")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    publication_axes(ax)
    if colorbar:
        cb = aligned_colorbar(ax.figure, ax, mesh, r"$U_{\mathrm{mech,next}}=-\log p_{\mathrm{mech}}$", width="4.0%", pad=0.050, labelpad=1.0)
        cb.ax.invert_yaxis()


def draw_drift_residual(ax, f_emp: FieldStats, f_mech: FieldStats, title: str, colorbar: bool = True, vmax: float = DRIFT_RESIDUAL_VMAX) -> None:
    residual = np.sqrt((f_mech.u - f_emp.u) ** 2 + (f_mech.v - f_emp.v) ** 2)
    mask = f_emp.mask & f_mech.mask
    residual = np.where(mask, residual, np.nan)
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else DRIFT_RESIDUAL_VMAX
    cmap = BLUE_CMAP.copy(); cmap.set_bad("#f1f5f9")
    ax.set_facecolor("#f1f5f9")
    mesh = ax.pcolormesh(GRID_BINS_SIGNED, GRID_BINS_SIGNED, residual.T, shading="auto", cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(r"Response evidence order $M$")
    ax.set_ylabel(r"Exposure-alignment order $\Psi$")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    publication_axes(ax)
    if colorbar:
        cb = aligned_colorbar(ax.figure, ax, mesh, r"drift residual magnitude $\|\hat b-b\|$", width="4.0%", pad=0.050, labelpad=1.0)
        cb.set_ticks([0.0, 0.5 * vmax, vmax])
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def weighted_prediction_hist(obs: np.ndarray, pred: np.ndarray, weights: Optional[np.ndarray] = None, bins: int = 90) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(pred)
    ww = None
    if weights is not None:
        ww_all = np.asarray(weights, dtype=float)
        ww = ww_all[ok]
    H, xedges, yedges = np.histogram2d(obs[ok], pred[ok], bins=bins, range=[[-1, 1], [-1, 1]], weights=ww)
    return H.astype(float), xedges, yedges


def draw_prediction_density(
    fig: plt.Figure,
    ax: plt.Axes,
    H: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    label: str,
    norm: Normalize,
    title: str,
    *,
    colorbar: bool = True,
) -> plt.cm.ScalarMappable:
    """Figure-4-style binned closure density with aligned sample-count colorbar."""
    Hplot = np.where(H > 0, H, np.nan)
    cmap = BLUE_CMAP.copy(); cmap.set_bad("#f1f5f9")
    ax.set_facecolor("#f1f5f9")
    mesh = ax.pcolormesh(xedges, yedges, Hplot.T, shading="auto", cmap=cmap, norm=norm)
    ax.plot([-1, 1], [-1, 1], linestyle="--", color="#08306b", linewidth=1.0, alpha=0.88)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"empirical next {label}")
    ax.set_ylabel(f"mechanism next {label}", labelpad=1.2)
    ax.set_title(title, pad=6)
    publication_axes(ax)
    if colorbar:
        cb = aligned_colorbar(fig, ax, mesh, "sample count", width="3.0%", pad=0.040, labelpad=0.8, shrink_ticks=True)
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    return mesh


def prediction_closure_panels(fig: plt.Figure, parent_gs, pred: pd.DataFrame, title: str) -> None:
    # Align closure panels with the other Figure 3 panels.
    sub = parent_gs.subgridspec(1, 2, wspace=0.46)
    ax_m = fig.add_subplot(sub[0, 0])
    ax_p = fig.add_subplot(sub[0, 1])
    ax_m.text(-0.115, 1.185, title, transform=ax_m.transAxes, fontsize=10.8, va="top", ha="left")

    obs_m = pred["target_M_next"].to_numpy(dtype=float)
    pred_m = pred["pred_next_M"].to_numpy(dtype=float)
    obs_p = pred["target_Psi_next"].to_numpy(dtype=float)
    pred_p = pred["pred_next_Psi"].to_numpy(dtype=float)
    Hm, xm, ym = weighted_prediction_hist(obs_m, pred_m, bins=85)
    Hp, xp, yp = weighted_prediction_hist(obs_p, pred_p, bins=85)
    max_count = np.nanmax([
        np.nanmax(Hm) if np.isfinite(Hm).any() else 1.0,
        np.nanmax(Hp) if np.isfinite(Hp).any() else 1.0,
    ])
    norm = LogNorm(vmin=1.0, vmax=max(float(max_count), 1.0))

    draw_prediction_density(fig, ax_m, Hm, xm, ym, r"$M$", norm, r"$M$ closure", colorbar=True)
    draw_prediction_density(fig, ax_p, Hp, xp, yp, r"$\Psi$", norm, r"$\Psi$ closure", colorbar=True)



def load_minimality_summary(minimality_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = read_table(minimality_root / "tables" / "model_family_results")
    deletion = read_table(minimality_root / "tables" / "global_scalar_deletion_audit")
    return summary, deletion

def _plot_deletion_panel(ax, deletion: pd.DataFrame) -> bool:
    required = {"tested_removed_parameter", "paired_difference_mean", "paired_difference_ci95_lower", "paired_difference_ci95_upper"}
    if deletion.empty or not required.issubset(deletion.columns):
        return False
    df = deletion.copy()
    labels = [parameter_math_label(x) for x in df["tested_removed_parameter"].astype(str).tolist()]
    y = np.arange(len(df))
    x = pd.to_numeric(df["paired_difference_mean"], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(df["paired_difference_ci95_lower"], errors="coerce").to_numpy(dtype=float)
    hi = pd.to_numeric(df["paired_difference_ci95_upper"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
    if not ok.any():
        return False
    ax.errorbar(x[ok], y[ok], xerr=[x[ok] - lo[ok], hi[ok] - x[ok]], fmt="o", capsize=2.5,
                color="#08306b", ecolor="#4292c6", linewidth=0.9, markersize=4.0)
    ax.axvline(0.0, linestyle="--", linewidth=0.8, color="#334155", alpha=0.75)
    ax.set_yticks(y[ok]); ax.set_yticklabels([labels[i] for i in np.where(ok)[0]])
    ax.set_xlabel("score increase")
    ax.set_ylabel("removed term")
    ax.text(0.985, 0.92, "Term necessity", transform=ax.transAxes, ha="right", va="top", fontsize=9.3, fontweight="semibold", color="#0f172a", bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85))
    ax.grid(axis="x", alpha=0.18)
    publication_axes(ax)
    return True


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    x = df[col]
    if x.dtype == bool:
        return x.fillna(False)
    return x.astype(str).str.lower().isin(["true", "1", "yes"])


def parameter_math_label(name: object) -> str:
    key = str(name)
    return PARAMETER_MATH_LABELS.get(key, key)


def _one_se_threshold_from_summary(df: pd.DataFrame) -> float:
    required = {"Bootstrap mean primary score", "Bootstrap standard error"}
    if df.empty or not required.issubset(df.columns):
        raise RuntimeError("Formal model-family results lack bootstrap score or standard error.")
    score = pd.to_numeric(df["Bootstrap mean primary score"], errors="coerce").to_numpy(dtype=float)
    standard_error = pd.to_numeric(df["Bootstrap standard error"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(score).any():
        raise RuntimeError("Formal model-family bootstrap scores are unavailable.")
    best_index = int(np.nanargmin(score))
    if not np.isfinite(standard_error[best_index]) or standard_error[best_index] < 0:
        raise RuntimeError("The best-family bootstrap standard error is invalid.")
    return float(score[best_index] + standard_error[best_index])

def draw_minimality_panel_stack(fig: plt.Figure, parent_gs, summary: pd.DataFrame, deletion: pd.DataFrame, title: str) -> None:
    outer = parent_gs.subgridspec(2, 1, height_ratios=[2.70, 1.00], hspace=0.10)
    ax_top = fig.add_subplot(outer[0, 0])
    ax_bottom = fig.add_subplot(outer[1, 0])

    if summary.empty or "Bootstrap mean primary score" not in summary.columns:
        ax_top.text(0.5, 0.5, "Minimality summary unavailable", transform=ax_top.transAxes, ha="center", va="center")
        ax_top.set_title(title)
        ax_bottom.set_axis_off()
        return

    df = summary.copy()
    if "Free mechanism parameters" not in df.columns:
        df["Free mechanism parameters"] = np.arange(len(df))
    if "family_key" not in df.columns:
        df["family_key"] = [f"family_{i}" for i in range(len(df))]
    df = df.sort_values(["Free mechanism parameters", "Bootstrap mean primary score"], kind="mergesort").reset_index(drop=True)

    x_raw = pd.to_numeric(df["Free mechanism parameters"], errors="coerce").to_numpy(dtype=float)
    y_raw = pd.to_numeric(df["Bootstrap mean primary score"], errors="coerce").to_numpy(dtype=float)
    lo_raw = pd.to_numeric(df.get("Bootstrap 95% CI lower", pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(dtype=float)
    hi_raw = pd.to_numeric(df.get("Bootstrap 95% CI upper", pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x_raw) & np.isfinite(y_raw) & (y_raw > 0)
    if ok.any():
        # Separate families with equal parameter counts.
        x_plot = x_raw.copy()
        for xv in sorted(set(x_raw[ok].astype(int).tolist())):
            idx = np.where(ok & np.isclose(x_raw, float(xv)))[0]
            if len(idx) > 1:
                x_plot[idx] = x_raw[idx] + np.linspace(-0.075, 0.075, len(idx))
        y_plot = np.clip(y_raw, 0.0, 1.0)
        yerr_low = np.where(np.isfinite(lo_raw), y_plot - np.clip(lo_raw, 0.0, 1.0), np.nan)
        yerr_high = np.where(np.isfinite(hi_raw), np.clip(hi_raw, 0.0, 1.0) - y_plot, np.nan)
        ci_ok = ok & np.isfinite(yerr_low) & np.isfinite(yerr_high) & (yerr_low >= 0) & (yerr_high >= 0)
        if ci_ok.any():
            ax_top.errorbar(
                x_plot[ci_ok], y_plot[ci_ok], yerr=[yerr_low[ci_ok], yerr_high[ci_ok]],
                fmt="none", ecolor="#6baed6", elinewidth=0.9, capsize=2.5, alpha=0.80, zorder=1,
            )
        ax_top.scatter(x_plot[ok], y_plot[ok], s=22, facecolor="#9ecae1", edgecolor="#08519c", linewidth=0.55, alpha=0.92, zorder=3)

        # Plot the best score at each complexity.
        env_x, env_y = [], []
        for xv in sorted(set(x_raw[ok].astype(int).tolist())):
            idx = np.where(ok & np.isclose(x_raw, float(xv)))[0]
            if len(idx):
                best_local = idx[int(np.nanargmin(y_raw[idx]))]
                env_x.append(float(xv)); env_y.append(float(np.clip(y_raw[best_local], 0.0, 1.0)))
        if len(env_x) >= 2:
            ax_top.plot(env_x, env_y, color="#08306b", linewidth=1.1, alpha=0.72, zorder=2)

        best_idx = int(np.nanargmin(np.where(ok, y_raw, np.inf)))
        ax_top.scatter(x_plot[best_idx], np.clip(y_raw[best_idx], 0.0, 1.0), s=56, marker="*", color="#08306b", edgecolor="#f7fbff", linewidth=0.65, zorder=6)

        pars_mask = _bool_series(df, "Parsimonious family selected").to_numpy(dtype=bool)
        final_mask = _bool_series(df, "Final scalar-minimal family").to_numpy(dtype=bool)
        if pars_mask.any():
            idx = np.where(pars_mask & ok)[0]
            ax_top.scatter(x_plot[idx], np.clip(y_raw[idx], 0.0, 1.0), s=58, facecolor="none", edgecolor="#111827", linewidth=1.15, zorder=7)
        if final_mask.any():
            idx = np.where(final_mask & ok)[0]
            ax_top.scatter(x_plot[idx], np.clip(y_raw[idx], 0.0, 1.0), s=44, marker="D", facecolor="#4292c6", edgecolor="#111827", linewidth=0.85, zorder=8)

        one_se = _one_se_threshold_from_summary(df)
        if np.isfinite(one_se) and one_se > 0:
            ax_top.axhline(float(np.clip(one_se, 0.0, 1.0)), linestyle="--", linewidth=1.0, color="#334155", alpha=0.85, zorder=0)

        xmax = max(1.0, float(np.nanmax(x_raw[ok])))
        xmin = min(0.0, float(np.nanmin(x_raw[ok])))
        ax_top.set_xlim(xmin - 0.35, xmax + 0.45)
    ax_top.set_ylim(0.0, 1.0)
    ax_top.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax_top.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_top.set_xlabel("free mechanism parameters")
    ax_top.set_ylabel("primary-structure score")
    ax_top.set_title(title)
    ax_top.grid(axis="both", alpha=0.16, which="both")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9ecae1", markeredgecolor="#08519c", markersize=4.6, label="candidate family"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#08306b", markeredgecolor="#f7fbff", markersize=7.0, label="best-scoring family"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#4292c6", markeredgecolor="#111827", markersize=4.8, label="final minimal family"),
        Line2D([0], [0], color="#334155", linestyle="--", linewidth=1.0, label="one-SE threshold"),
    ]
    ax_top.legend(handles=handles, frameon=False, loc="upper right", ncol=1, columnspacing=0.70, handletextpad=0.35)
    publication_axes(ax_top)

    if not _plot_deletion_panel(ax_bottom, deletion):
        ax_bottom.text(0.5, 0.5, "Term necessity unavailable", transform=ax_bottom.transAxes, ha="center", va="center", color="#475569")
        ax_bottom.set_xlabel("score increase")
        ax_bottom.set_ylabel("removed term")
        ax_bottom.text(0.985, 0.92, "Term necessity", transform=ax_bottom.transAxes, ha="right", va="top", fontsize=9.3, fontweight="semibold", color="#0f172a", bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85))
        publication_axes(ax_bottom)


def draw_transition_matrix(ax, P: np.ndarray, title: str, colorbar: bool = True, diff: bool = False, annotate: bool = True) -> None:
    if P.size == 0:
        ax.text(0.5, 0.5, "No transition data", transform=ax.transAxes, ha="center", va="center")
        return
    if diff:
        lim = float(TRANSITION_RESIDUAL_VLIM) if np.isfinite(TRANSITION_RESIDUAL_VLIM) and TRANSITION_RESIDUAL_VLIM > 0 else max(float(np.nanquantile(np.abs(P[np.isfinite(P)]), 0.98)) if np.isfinite(P).any() else 1.0, 1e-6)
        im = ax.imshow(P, origin="lower", vmin=-lim, vmax=lim, cmap=RESIDUAL_CMAP, aspect="equal")
    else:
        vmax = max(1.0, float(np.nanmax(P)) if np.isfinite(P).any() else 1.0)
        im = ax.imshow(P, origin="lower", vmin=0.0, vmax=vmax, cmap=BLUE_CMAP, aspect="equal")
    k = P.shape[0]
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels([f"S{i}" for i in range(k)]); ax.set_yticklabels([f"S{i}" for i in range(k)])
    try:
        ax.set_box_aspect(1.0)
    except Exception:
        pass
    ax.set_xlabel("Next macrostate"); ax.set_ylabel("Current macrostate"); ax.set_title(title)
    if annotate and k <= 10:
        threshold = 0.55 * (float(np.nanmax(np.abs(P))) if diff else float(np.nanmax(P))) if np.isfinite(P).any() else 0.5
        for i in range(k):
            for j in range(k):
                val = float(P[i, j])
                txt_color = "white" if abs(val) >= threshold else "#08306b"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=txt_color)
    publication_axes(ax)
    if colorbar:
        cb = aligned_colorbar(ax.figure, ax, im, "mechanism - empirical" if diff else "transition probability", width="5.0%", pad=0.055, labelpad=1.0)
        if diff:
            cb.set_ticks([-lim, 0.0, lim])
            cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        else:
            cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def load_stability_for_reference(phase3_root: Path, reference_label: str = "A_train_plus_A_val") -> pd.DataFrame:
    """Load split-to-confirmation stability metrics.

    The requested reference label determines the comparison displayed in the
    publication panel. Figure 3c passes A_val so that the visual comparison is
    Validation set versus Confirmation set, while older tables may still fall back to an
    available development reference for exploratory plotting.
    """
    try:
        d = read_table(Path(phase3_root) / "tables" / "phase3_development_vs_confirmation_metric_stability")
    except FileNotFoundError:
        return pd.DataFrame()
    if d.empty or "metric" not in d.columns:
        return pd.DataFrame()
    chosen = str(reference_label)
    if "reference_label" in d.columns:
        available = set(d["reference_label"].astype(str))
        if chosen not in available and "A_train_plus_A_val" in available:
            chosen = "A_train_plus_A_val"
        elif chosen not in available and "A_val" in available:
            chosen = "A_val"
        d = d[d["reference_label"].astype(str) == chosen].copy()
        d["_publication_reference_label"] = chosen
    keep = [m for m, _ in STABILITY_PANEL_METRICS]
    d = d[d["metric"].astype(str).isin(keep)].copy()
    order = {m: i for i, (m, _) in enumerate(STABILITY_PANEL_METRICS)}
    if not d.empty:
        d["_order"] = d["metric"].astype(str).map(order)
        d = d.sort_values("_order", kind="mergesort")
    return d


def draw_confirmation_stability(ax, stability: pd.DataFrame, title: str) -> None:
    if stability is None or stability.empty or not {"metric", "development_value", "confirmation_value"}.issubset(stability.columns):
        ax.text(0.5, 0.5, "Metric stability unavailable", transform=ax.transAxes, ha="center", va="center", color="#475569")
        ax.set_title(title)
        publication_axes(ax)
        return
    label_map = {m: lab for m, lab in STABILITY_PANEL_METRICS}
    metrics = stability["metric"].astype(str).tolist()
    labels = [label_map.get(m, m.replace("_MR_PsiA", "").replace("_", " ")) for m in metrics]
    dev = pd.to_numeric(stability["development_value"], errors="coerce").to_numpy(dtype=float)
    con = pd.to_numeric(stability["confirmation_value"], errors="coerce").to_numpy(dtype=float)
    x = np.arange(len(labels), dtype=float)
    width = 0.38
    ax.bar(x - width / 2, dev, width=width, color="#c6dbef", edgecolor="#08519c", linewidth=0.65, label="Validation set")
    ax.bar(x + width / 2, con, width=width, color="#4292c6", edgecolor="#08306b", linewidth=0.65, label="Confirmation set")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylim(0.0, 0.5)
    ax.set_yticks([0.0, 0.10, 0.20, 0.30, 0.40, 0.50])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_ylabel("metric value")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, loc="upper right", ncol=1, handlelength=1.4)
    publication_axes(ax)


def plot_figure2(fields: Mapping[str, object], minimality_root: Path, out_path: Path, formats: Sequence[str], formal_metrics: Optional[Mapping[str, float]] = None, potential_clim: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
    summary, deletion = load_minimality_summary(minimality_root)
    f_emp: FieldStats = fields["f_emp"]
    f_mech: FieldStats = fields["f_mech"]
    H_emp = fields["H_emp_next"]
    H_mech = fields["H_mech_next"]
    fig = plt.figure(figsize=(16.4, 11.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[1.0, 1.0])
    draw_minimality_panel_stack(fig, gs[0, 0], summary, deletion, "(a) Model minimality")
    ax_b = fig.add_subplot(gs[0, 1]); draw_potential_difference(ax_b, H_emp, H_mech, "(b) Next-state quasi-potential residual")
    ax_c = fig.add_subplot(gs[1, 0]); draw_mechanism_drift(ax_c, f_mech, H_mech, "(c) Mechanism-predicted landscape and drift", potential_clim=potential_clim)
    ax_d = fig.add_subplot(gs[1, 1]); draw_drift_residual(ax_d, f_emp, f_mech, "(d) Drift residual magnitude")
    savefig(fig, out_path, formats=formats)
    out = dict(fields["metrics"])
    if formal_metrics:
        out.update({f"formal_{k}": v for k, v in formal_metrics.items()})
    return out


def plot_figure3(predictions: pd.DataFrame, kinetic: Mapping[str, object], out_path: Path, formats: Sequence[str], phase3_root: Path) -> Dict[str, float]:
    P_emp = kinetic["P_emp"]; P_mech = kinetic["P_mech"]
    metrics = kinetic["metrics"]
    stability = load_stability_for_reference(phase3_root, reference_label="A_val")

    # Keep transition-implied residence references outside the main Figure 3.
    fig = plt.figure(figsize=(21.2, 5.72), constrained_layout=False)
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[3.08, 1.22, 1.16],
        left=0.042, right=0.990, bottom=0.175, top=0.905,
        wspace=0.285,
    )
    prediction_closure_panels(fig, gs[0, 0], predictions, "(a) One-step closure")
    ax_b = fig.add_subplot(gs[0, 1])
    draw_transition_matrix(ax_b, P_mech - P_emp, "(b) Transition residual", colorbar=True, diff=True, annotate=True)
    ax_c = fig.add_subplot(gs[0, 2])
    draw_confirmation_stability(ax_c, stability, "(c) Confirmation stability")
    savefig(fig, out_path, formats=formats)

    out = dict(metrics)
    if not stability.empty:
        for _, row in stability.iterrows():
            metric = str(row.get("metric", "metric"))
            out[f"stability_development_{metric}"] = float(row.get("development_value", np.nan)) if np.isfinite(pd.to_numeric(row.get("development_value", np.nan), errors="coerce")) else np.nan
            out[f"stability_B_confirm_{metric}"] = float(row.get("confirmation_value", np.nan)) if np.isfinite(pd.to_numeric(row.get("confirmation_value", np.nan), errors="coerce")) else np.nan
    return out


def plot_standalone_transition(kinetic: Mapping[str, object], out_path: Path, formats: Sequence[str]) -> None:
    P_emp = kinetic["P_emp"]; P_mech = kinetic["P_mech"]
    fig = plt.figure(figsize=(12.8, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, wspace=0.24)
    draw_transition_matrix(fig.add_subplot(gs[0, 0]), P_emp, "Empirical transition", colorbar=True, annotate=True)
    draw_transition_matrix(fig.add_subplot(gs[0, 1]), P_mech, "Mechanism transition", colorbar=True, annotate=True)
    draw_transition_matrix(fig.add_subplot(gs[0, 2]), P_mech - P_emp, "Mechanism $-$ empirical", colorbar=True, diff=True, annotate=True)
    savefig(fig, out_path, formats=formats)


def plot_supplementary_metric_stability(phase3_root: Path, out_path: Path, formats: Sequence[str]) -> None:
    try:
        d = read_table(phase3_root / "tables" / "phase3_development_vs_confirmation_metric_stability")
    except FileNotFoundError:
        warnings.warn("Metric stability table unavailable; skipping supplementary metric stability figure.")
        return
    metric_col = "metric" if "metric" in d.columns else "metric_name" if "metric_name" in d.columns else None
    if metric_col is None or "development_value" not in d.columns or "confirmation_value" not in d.columns:
        warnings.warn("Metric stability table has an unexpected schema; skipping.")
        return
    keep = ["one_step_mse_main_norm", "occupancy_js_MR_PsiA", "drift_vector_corr_MR_PsiA", "drift_local_rmse_loss_MR_PsiA", "drift_magnitude_loss_MR_PsiA"]
    dd = d[d[metric_col].astype(str).isin(keep)].copy()
    if dd.empty:
        dd = d.head(8).copy()
    labels = dd[metric_col].astype(str).str.replace("_MR_PsiA", "", regex=False).str.replace("_", " ", regex=False)
    x = np.arange(len(dd)); width = 0.38
    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    ax.bar(x - width / 2, pd.to_numeric(dd["development_value"], errors="coerce"), width=width, label="Development")
    ax.bar(x + width / 2, pd.to_numeric(dd["confirmation_value"], errors="coerce"), width=width, label="Confirmation set")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("metric value"); ax.set_title("Development-to-confirmation metric stability")
    ax.legend(frameon=False); publication_axes(ax)
    savefig(fig, out_path, formats=formats)


def draw_residence_small_multiples(
    fig: plt.Figure,
    parent_gs,
    curves: pd.DataFrame,
    P_mech: np.ndarray,
    max_len: int,
    title: str,
) -> None:
    k = int(P_mech.shape[0])
    ncols = 3
    nrows = int(math.ceil(k / ncols))
    outer = parent_gs.subgridspec(
        nrows + 1,
        ncols,
        height_ratios=[0.08] + [1.0] * nrows,
        wspace=0.34,
        hspace=0.48,
    )
    title_ax = fig.add_subplot(outer[0, :])
    title_ax.set_axis_off()
    title_ax.patch.set_alpha(0.0)
    title_ax.set_title(title, pad=0)
    handles = [
        Line2D([0], [0], color="#08306b", linewidth=1.20, linestyle="-", label="empirical residence"),
        Line2D([0], [0], color="#08306b", linewidth=0.95, linestyle="--", label="empirical geometric"),
        Line2D([0], [0], color="#111827", linewidth=1.25, linestyle=":", label="mechanism reference"),
    ]
    title_ax.legend(handles=handles, frameon=False, ncol=3, loc="center", bbox_to_anchor=(0.5, 0.30), columnspacing=1.0, handlelength=2.0)
    colors = [STATE_LINE_CMAP(index / max(k - 1, 1)) for index in range(k)]
    for state in range(k):
        ax = fig.add_subplot(outer[1 + state // ncols, state % ncols])
        data = curves[curves["macrostate"].astype(int) == state].copy()
        data = data[pd.to_numeric(data["residence_length"], errors="coerce") <= max_len]
        data = data.sort_values("residence_length", kind="mergesort")
        color = colors[state]
        if not data.empty:
            ax.plot(data["residence_length"], data["km_ccdf"], color=color, linewidth=1.20)
            ax.plot(data["residence_length"], data["geometric_ccdf"], color=color, linestyle="--", linewidth=0.95, alpha=0.84)
            lengths = pd.to_numeric(data["residence_length"], errors="coerce").to_numpy(dtype=int)
            mechanism_reference = float(np.clip(P_mech[state, state], 0.0, 1.0 - 1e-9)) ** (lengths - 1)
            ax.plot(lengths, mechanism_reference, color="#111827", linestyle=":", linewidth=1.25, alpha=0.92)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, max_len)
        ax.set_ylim(1e-5, 1.05)
        ax.set_title(f"S{state}", fontsize=8.5)
        if state // ncols == nrows - 1:
            ax.set_xlabel("residence length")
        if state % ncols == 0:
            ax.set_ylabel("CCDF")
        publication_axes(ax)


def plot_standalone_residence(
    kinetic: Mapping[str, object],
    out_path: Path,
    max_len: int,
    formats: Sequence[str],
) -> None:
    fig = plt.figure(figsize=(12.8, 7.4), constrained_layout=False)
    grid = fig.add_gridspec(1, 1)
    draw_residence_small_multiples(
        fig,
        grid[0, 0],
        kinetic["residence_curves"],
        kinetic["P_mech"],
        max_len=max_len,
        title="Residence-time references under frozen mesostates",
    )
    savefig(fig, out_path, formats=formats)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render Figures 2 and 3 for the frozen EdNet-KT4 minimal mechanism."
    )
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--phase2-root", type=Path, default=DEFAULT_PHASE2_ROOT)
    parser.add_argument("--phase3-root", type=Path, default=DEFAULT_PHASE3_ROOT)
    parser.add_argument("--minimality-root", type=Path, default=DEFAULT_MINIMALITY_ROOT)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--confirm-split", type=str, default="B_confirm")
    parser.add_argument("--max-residence-len", type=int, default=MAX_RESIDENCE_LEN_FOR_PLOT)
    parser.add_argument("--residence-metric-max-len", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"], choices=["png", "pdf", "svg"])
    parser.add_argument("--skip-standalone", action="store_true")
    parser.add_argument("--make-supplementary-metric-stability", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.confirm_split != "B_confirm":
        raise RuntimeError("Formal publication figures require confirm_split='B_confirm'.")
    formats = list(dict.fromkeys(args.formats))
    stage1_root = args.stage1_root.resolve()
    phase2_root = args.phase2_root.resolve()
    phase3_root = args.phase3_root.resolve()
    minimality_root = args.minimality_root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else phase3_root / "figures_publication_minimal_mechanism"
    table_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    phase2_manifest = load_phase2_frozen_manifest(phase2_root)
    phase3_audit = load_phase3_no_update_audit(phase3_root)
    phase3_manifest = load_phase3_confirmation_manifest(phase3_root)
    source_root_audit = validate_source_roots(stage1_root, minimality_root, phase2_manifest, phase3_manifest)
    formal_metrics, formal_metrics_path = load_phase3_formal_metrics(phase3_root, args.confirm_split)
    predictions, predictions_path = load_confirmation_predictions(phase3_root, args.confirm_split, args.predictions)
    fields, field_grid_path = load_confirmation_fields(phase3_root, args.confirm_split)
    fields["metrics"].update({
        "one_step_rmse_M": float(np.sqrt(np.mean((predictions["pred_next_M"] - predictions["target_M_next"]) ** 2))),
        "one_step_rmse_Psi": float(np.sqrt(np.mean((predictions["pred_next_Psi"] - predictions["target_Psi_next"]) ** 2))),
        "n_rows": int(len(predictions)),
        "n_users": int(predictions["user_id"].nunique()),
    })

    partition = load_frozen_macro_partition(stage1_root, phase2_manifest, phase3_manifest)
    stage1_tables = load_stage1_mesostate_tables(stage1_root)
    kinetic = kinetic_recovery(
        predictions,
        partition,
        stage1_tables,
        int(args.max_residence_len),
        int(args.residence_metric_max_len),
    )
    potential_vmin, potential_vmax, potential_grid_path = load_empirical_current_potential_clim(
        stage1_root,
        MECHANISM_POTENTIAL_SCALE_SPLIT,
    )

    figure2_metrics = plot_figure2(
        fields,
        minimality_root,
        out_dir / "figure2_minimal_mechanism_field_recovery",
        formats=formats,
        formal_metrics=formal_metrics,
        potential_clim=(potential_vmin, potential_vmax),
    )
    figure3_metrics = plot_figure3(
        predictions,
        kinetic,
        out_dir / "figure3_minimal_mechanism_kinetic_recovery",
        formats=formats,
        phase3_root=phase3_root,
    )
    if not args.skip_standalone:
        plot_standalone_transition(
            kinetic,
            out_dir / "figure3_transition_comparison_standalone",
            formats=formats,
        )
        plot_standalone_residence(
            kinetic,
            out_dir / "figure3_residence_comparison_standalone",
            max_len=int(args.max_residence_len),
            formats=formats,
        )
    if args.make_supplementary_metric_stability:
        plot_supplementary_metric_stability(
            phase3_root,
            out_dir / "supplementary_development_confirmation_metric_stability",
            formats=formats,
        )

    write_table(pd.DataFrame([figure2_metrics]), table_dir / "figure2_field_recovery_metrics")
    write_table(pd.DataFrame([figure3_metrics]), table_dir / "figure3_kinetic_recovery_metrics")
    write_table(pd.DataFrame([formal_metrics]), table_dir / "phase3_formal_confirmation_metrics")
    write_table(partition.fit_table, table_dir / "empirical_kmeans_partition_fit_table")
    write_table(partition.centers_table, table_dir / "empirical_kmeans_partition_centers")
    write_table(pd.DataFrame(kinetic["C_emp"]), table_dir / "B_confirm_empirical_mesostate_transition_counts")
    write_table(pd.DataFrame(kinetic["C_mech"]), table_dir / "B_confirm_mechanism_mesostate_transition_counts")
    write_table(pd.DataFrame(kinetic["P_emp"]), table_dir / "B_confirm_empirical_mesostate_transition_matrix")
    write_table(pd.DataFrame(kinetic["P_mech"]), table_dir / "B_confirm_mechanism_mesostate_transition_matrix")
    write_table(kinetic["residence_curves"], table_dir / "B_confirm_empirical_residence_curves_fixed_k6")
    residence_references = kinetic["residence_curves"].copy()
    residence_references["mechanism_geometric_ccdf"] = [
        float(kinetic["P_mech"][int(state), int(state)]) ** (int(length) - 1)
        for state, length in zip(
            residence_references["macrostate"],
            residence_references["residence_length"],
        )
    ]
    write_table(residence_references, table_dir / "B_confirm_mechanism_residence_reference_curves")
    write_table(stage1_tables["residence_summary"], table_dir / "B_confirm_empirical_residence_summary_fixed_k6")
    assignment_audit = {
        key: kinetic["metrics"].get(key)
        for key in (
            "partition_assignment_join_fraction",
            "stage1_full_vs_phase3_valid_empirical_transition_max_abs_difference",
            "stage1_full_transition_count",
            "transition_count",
        )
    }
    write_table(pd.DataFrame([assignment_audit]), table_dir / "B_confirm_mesostate_assignment_audit")

    script_path = Path(__file__).resolve()
    manifest = {
        "script": script_path.name,
        "script_sha256": file_sha256(script_path),
        "phase2_root": str(phase2_root),
        "phase3_root": str(phase3_root),
        "minimality_root": str(minimality_root),
        "stage1_root": str(stage1_root),
        "confirm_split": args.confirm_split,
        "primary_macrostate": ["M", "Psi"],
        "prediction_rows": int(len(predictions)),
        "prediction_users": int(predictions["user_id"].nunique()),
        "sources": {
            "phase3_predictions": {"path": str(predictions_path), "sha256": file_sha256(predictions_path)},
            "phase3_field_grid": {"path": str(field_grid_path), "sha256": file_sha256(field_grid_path)},
            "phase3_formal_metrics": {"path": str(formal_metrics_path), "sha256": file_sha256(formal_metrics_path)},
            "stage1_current_potential_grid": {"path": str(potential_grid_path), "sha256": file_sha256(potential_grid_path)},
            "stage1_assignments": {"path": str(stage1_tables["assignments_path"]), "sha256": file_sha256(stage1_tables["assignments_path"])},
            "stage1_transition_matrix": {"path": str(stage1_tables["transition_matrix_path"]), "sha256": file_sha256(stage1_tables["transition_matrix_path"])},
            "stage1_residence_curves": {"path": str(stage1_tables["residence_curves_path"]), "sha256": file_sha256(stage1_tables["residence_curves_path"])},
        },
        "phase2_frozen_manifest": phase2_manifest,
        "phase3_no_update_audit": phase3_audit,
        "phase3_confirmation_manifest": phase3_manifest,
        "source_root_audit": source_root_audit,
        "macrostate_partition": partition.audit,
        "guardrails": {
            "KMeans_not_refit": True,
            "macrostate_k_fixed_at_6": True,
            "macrostate_k_selected_in_figure_code": False,
            "mechanism_outputs_do_not_define_partition": True,
            "transition_and_residence_not_phase1_targets": True,
            "mechanism_residence_curve": "transition-implied geometric reference, not a free-running residence distribution",
            "V_main_figure_policy": "V is not a primary macrostate coordinate and is not plotted.",
        },
        "figure2_metrics": figure2_metrics,
        "figure3_metrics": figure3_metrics,
        "outputs": {
            "figure2": str((out_dir / "figure2_minimal_mechanism_field_recovery.png").resolve()),
            "figure3": str((out_dir / "figure3_minimal_mechanism_kinetic_recovery.png").resolve()),
        },
    }
    save_json(manifest, out_dir / "publication_minimal_mechanism_figure_manifest.json")
    print(f"[publication-minimal-mechanism] figures written to {out_dir}")


if __name__ == "__main__":
    main()
