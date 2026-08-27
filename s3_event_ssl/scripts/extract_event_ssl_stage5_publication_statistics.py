#!/usr/bin/env python3
"""Collect publication statistics from frozen Stage-5 Event-SSL analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
RMSE_SCORE_SCALE = 0.15
DEFAULT_BASE = Path(os.environ.get("EDNET_KT4_OUTPUT_ROOT", "/data/datasets/KT4/outputs_KT4"))
DEFAULT_MACRO_ROOT = DEFAULT_BASE / "stage5_macro_sufficiency" / "evaluation"
DEFAULT_MACRO_TRAIN_ROOT = DEFAULT_BASE / "stage5_macro_sufficiency"
DEFAULT_GEOMETRY_ROOT = DEFAULT_BASE / "stage5_representation_geometry" / "evaluation"
DEFAULT_GEOMETRY_TRAIN_ROOT = DEFAULT_BASE / "stage5_representation_geometry"
DEFAULT_OUTPUT_ROOT = DEFAULT_BASE / "stage5_joint_macro_geometry_analysis"
DEFAULT_SPLITS = ("A_val", "B_confirm")
EXPECTED_K = 6

MAIN_REQUIRED = "main_text_required"
MAIN_RECOMMENDED = "main_text_recommended"
SUPPLEMENT_REQUIRED = "supplement_required"

MACRO_REPRESENTATIONS = ("full_hidden", "macro_only", "residual_hidden")
GEOMETRY_REPRESENTATIONS = ("model_readout", "linear_hidden", "residual_hidden", "nonlinear_hidden")

HIGHER_IS_BETTER = {
    "coordinate_corr_M", "coordinate_corr_Psi", "one_step_corr_M", "one_step_corr_Psi",
    "anchor_drift_vector_corr", "learned_plane_drift_vector_corr",
    "anchor_occupancy_weighted_local_drift_cosine", "learned_plane_occupancy_weighted_local_drift_cosine",
    "learned_plane_self_transition_corr", "learned_plane_diagonal_dominance_match_fraction",
    "learned_plane_top_transition_edge_overlap", "task_auc", "task_accuracy_0p5",
    "representation_nmi_with_empirical_macrostate", "representation_ari_with_empirical_macrostate",
    "cca_corr_1", "cca_corr_2",
}
LOWER_IS_BETTER = {
    "coordinate_rmse_M", "coordinate_rmse_Psi", "one_step_rmse_M", "one_step_rmse_Psi",
    "current_state_occupancy_js", "next_state_occupancy_js",
    "learned_plane_transition_mean_row_tv", "anchor_transition_mean_row_tv",
    "task_bce", "task_rmse",
}

KEY_RAW_METRICS = [
    "coordinate_corr_M", "coordinate_corr_Psi",
    "coordinate_rmse_M", "coordinate_rmse_Psi",
    "one_step_rmse_M", "one_step_rmse_Psi",
    "next_state_occupancy_js",
    "learned_plane_drift_vector_corr",
    "learned_plane_occupancy_weighted_local_drift_cosine",
    "learned_plane_transition_mean_row_tv",
    "learned_plane_self_transition_corr",
    "task_auc", "task_bce",
    "representation_nmi_with_empirical_macrostate",
    "representation_ari_with_empirical_macrostate",
]


@dataclass
class SourceRecord:
    name: str
    path: Optional[Path]
    status: str
    rows: Optional[int] = None
    columns: Optional[int] = None
    sha256: Optional[str] = None
    note: str = ""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
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


def locate_table(base: Path) -> Optional[Path]:
    candidates = [base] if base.suffix else [base.with_suffix(".parquet"), base.with_suffix(".csv.gz"), base.with_suffix(".csv")]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def read_table(base: Path, *, required: bool = True) -> Tuple[pd.DataFrame, Optional[Path]]:
    path = locate_table(base)
    if path is None:
        if required:
            raise FileNotFoundError(f"Required table not found: {base}")
        return pd.DataFrame(), None
    if path.suffix == ".parquet":
        return pd.read_parquet(path), path
    return pd.read_csv(path, low_memory=False), path


def write_table(df: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = base.with_suffix(".parquet")
        df.to_parquet(path, index=False)
        df.to_csv(base.with_suffix(".csv"), index=False)
        return path
    except Exception:
        path = base.with_suffix(".csv")
        df.to_csv(path, index=False)
        return path


def load_json(path: Path, *, required: bool = True) -> Dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required JSON not found: {path}")
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return data


def finite_float(value: Any) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except Exception:
        return np.nan


def value(row: Optional[pd.Series], name: str) -> float:
    return np.nan if row is None else finite_float(row.get(name))




def mean_finite(values: Sequence[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else np.nan

def bounded_corr_score(x: float) -> float:
    if not np.isfinite(x):
        return np.nan
    return float(np.clip((x + 1.0) / 2.0, 0.0, 1.0))


def positive_score_from_loss(x: float, scale: float) -> float:
    if not np.isfinite(x):
        return np.nan
    return float(1.0 / (1.0 + max(x, 0.0) / max(scale, EPS)))


def score_from_metric(metric: str, raw_value: float) -> float:
    if not np.isfinite(raw_value):
        return np.nan
    if metric in HIGHER_IS_BETTER:
        if "corr" in metric or "cosine" in metric or metric.endswith("ari_with_empirical_macrostate"):
            return bounded_corr_score(raw_value)
        return float(np.clip(raw_value, 0.0, 1.0))
    if metric in LOWER_IS_BETTER:
        if "row_tv" in metric or "js" in metric:
            return float(np.clip(1.0 - raw_value, 0.0, 1.0))
        if "bce" in metric:
            return positive_score_from_loss(raw_value, 1.0)
        return positive_score_from_loss(raw_value, RMSE_SCORE_SCALE)
    return raw_value


def ratio_safe(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= EPS:
        return np.nan
    return float(numerator / denominator)


def row_for(df: pd.DataFrame, split: str, representation: str) -> Optional[pd.Series]:
    if df.empty or "split" not in df.columns or "representation" not in df.columns:
        return None
    subset = df[(df["split"].astype(str) == split) & (df["representation"].astype(str) == representation)]
    return None if subset.empty else subset.iloc[0]


def compute_domain_scores(row: Optional[pd.Series]) -> Dict[str, float]:
    if row is None:
        return {name: np.nan for name in (
            "coordinate_score", "closure_score", "landscape_score", "drift_score",
            "transition_score", "task_score", "macro_label_score",
            "macrostructure_composite_descriptive",
        )}
    coordinate = mean_finite([
        score_from_metric("coordinate_corr_M", value(row, "coordinate_corr_M")),
        score_from_metric("coordinate_corr_Psi", value(row, "coordinate_corr_Psi")),
    ])
    closure = mean_finite([
        score_from_metric("one_step_rmse_M", value(row, "one_step_rmse_M")),
        score_from_metric("one_step_rmse_Psi", value(row, "one_step_rmse_Psi")),
    ])
    landscape = mean_finite([
        score_from_metric("current_state_occupancy_js", value(row, "current_state_occupancy_js")),
        score_from_metric("next_state_occupancy_js", value(row, "next_state_occupancy_js")),
    ])
    drift = mean_finite([
        score_from_metric("learned_plane_drift_vector_corr", value(row, "learned_plane_drift_vector_corr")),
        score_from_metric("learned_plane_occupancy_weighted_local_drift_cosine", value(row, "learned_plane_occupancy_weighted_local_drift_cosine")),
    ])
    transition = mean_finite([
        score_from_metric("learned_plane_transition_mean_row_tv", value(row, "learned_plane_transition_mean_row_tv")),
        score_from_metric("learned_plane_self_transition_corr", value(row, "learned_plane_self_transition_corr")),
        score_from_metric("learned_plane_diagonal_dominance_match_fraction", value(row, "learned_plane_diagonal_dominance_match_fraction")),
        score_from_metric("learned_plane_top_transition_edge_overlap", value(row, "learned_plane_top_transition_edge_overlap")),
    ])
    task = np.nan
    if np.isfinite(value(row, "task_auc")):
        task = score_from_metric("task_auc", value(row, "task_auc"))
    elif np.isfinite(value(row, "task_bce")):
        task = score_from_metric("task_bce", value(row, "task_bce"))
    macro_label = mean_finite([
        score_from_metric("representation_nmi_with_empirical_macrostate", value(row, "representation_nmi_with_empirical_macrostate")),
        score_from_metric("representation_ari_with_empirical_macrostate", value(row, "representation_ari_with_empirical_macrostate")),
    ])
    composite = mean_finite([coordinate, closure, drift, transition])
    return {
        "coordinate_score": float(coordinate),
        "closure_score": float(closure),
        "landscape_score": float(landscape),
        "drift_score": float(drift),
        "transition_score": float(transition),
        "task_score": float(task) if np.isfinite(task) else np.nan,
        "macro_label_score": float(macro_label) if np.isfinite(macro_label) else np.nan,
        "macrostructure_composite_descriptive": float(composite),
    }


def add_domain_scores(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        output = {
            "experiment": experiment,
            "split": str(row.get("split", "")),
            "representation": str(row.get("representation", "")),
        }
        output.update(compute_domain_scores(row))
        rows.append(output)
    return pd.DataFrame(rows)


def build_macro_retention(macro_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    domains = (
        "coordinate_score", "closure_score", "landscape_score", "drift_score",
        "transition_score", "task_score", "macro_label_score",
        "macrostructure_composite_descriptive",
    )
    for split in sorted(macro_df["split"].dropna().astype(str).unique()):
        full = compute_domain_scores(row_for(macro_df, split, "full_hidden"))
        macro = compute_domain_scores(row_for(macro_df, split, "macro_only"))
        residual = compute_domain_scores(row_for(macro_df, split, "residual_hidden"))
        for domain in domains:
            full_value = full[domain]
            macro_value = macro[domain]
            residual_value = residual[domain]
            rows.append({
                "split": split,
                "domain": domain,
                "full_hidden_score": full_value,
                "macro_only_score": macro_value,
                "residual_hidden_score": residual_value,
                "macro_retention_vs_full": ratio_safe(macro_value, full_value),
                "residual_retention_vs_full": ratio_safe(residual_value, full_value),
                "residual_retention_vs_macro": ratio_safe(residual_value, macro_value),
                "macro_minus_residual": macro_value - residual_value if np.isfinite(macro_value) and np.isfinite(residual_value) else np.nan,
            })
        full_row = row_for(macro_df, split, "full_hidden")
        macro_row = row_for(macro_df, split, "macro_only")
        residual_row = row_for(macro_df, split, "residual_hidden")
        for metric in KEY_RAW_METRICS:
            full_raw = value(full_row, metric)
            macro_raw = value(macro_row, metric)
            residual_raw = value(residual_row, metric)
            rows.append({
                "split": split,
                "domain": f"raw::{metric}",
                "full_hidden_score": full_raw,
                "macro_only_score": macro_raw,
                "residual_hidden_score": residual_raw,
                "macro_retention_vs_full": ratio_safe(score_from_metric(metric, macro_raw), score_from_metric(metric, full_raw)),
                "residual_retention_vs_full": ratio_safe(score_from_metric(metric, residual_raw), score_from_metric(metric, full_raw)),
                "residual_retention_vs_macro": ratio_safe(score_from_metric(metric, residual_raw), score_from_metric(metric, macro_raw)),
                "macro_minus_residual": macro_raw - residual_raw if np.isfinite(macro_raw) and np.isfinite(residual_raw) else np.nan,
            })
    return pd.DataFrame(rows)


def build_geometry_retention(geometry_df: pd.DataFrame, gain_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    domains = (
        "coordinate_score", "closure_score", "landscape_score", "drift_score",
        "transition_score", "macrostructure_composite_descriptive",
    )
    for split in sorted(geometry_df["split"].dropna().astype(str).unique()):
        model_row = row_for(geometry_df, split, "model_readout")
        linear_row = row_for(geometry_df, split, "linear_hidden")
        residual_row = row_for(geometry_df, split, "residual_hidden")
        nonlinear_row = row_for(geometry_df, split, "nonlinear_hidden")
        model = compute_domain_scores(model_row)
        linear = compute_domain_scores(linear_row)
        residual = compute_domain_scores(residual_row)
        nonlinear = compute_domain_scores(nonlinear_row)
        for domain in domains:
            model_value = model[domain]
            linear_value = linear[domain]
            residual_value = residual[domain]
            nonlinear_value = nonlinear[domain]
            rows.append({
                "split": split,
                "domain": domain,
                "model_readout_score": model_value,
                "linear_hidden_score": linear_value,
                "residual_hidden_score": residual_value,
                "nonlinear_hidden_score": nonlinear_value,
                "linear_retention_vs_model": ratio_safe(linear_value, model_value),
                "residual_retention_vs_model": ratio_safe(residual_value, model_value),
                "residual_retention_vs_linear": ratio_safe(residual_value, linear_value),
                "linear_minus_residual": linear_value - residual_value if np.isfinite(linear_value) and np.isfinite(residual_value) else np.nan,
                "nonlinear_retention_vs_linear": ratio_safe(nonlinear_value, linear_value),
            })
        base = model_row if model_row is not None else linear_row
        if base is not None:
            for metric in ("cca_corr_1", "cca_corr_2", "twonn_dimension", "participation_ratio_train", "effective_rank_train", "sample_rows_geometry"):
                rows.append({"split": split, "domain": f"geometry::{metric}", "model_readout_score": value(base, metric)})
        gain = gain_df[gain_df["split"].astype(str) == split] if not gain_df.empty and "split" in gain_df.columns else pd.DataFrame()
        if not gain.empty:
            for metric in ("nonlinear_gain_corr_M", "nonlinear_gain_corr_Psi", "nonlinear_gain_rmse_reduction_M", "nonlinear_gain_rmse_reduction_Psi"):
                rows.append({"split": split, "domain": f"nonlinear_gain::{metric}", "model_readout_score": value(gain.iloc[0], metric)})
    return pd.DataFrame(rows)


def retention_value(df: pd.DataFrame, split: str, domain: str, column: str) -> float:
    subset = df[(df["split"].astype(str) == split) & (df["domain"].astype(str) == domain)]
    return np.nan if subset.empty or column not in subset.columns else finite_float(subset.iloc[0][column])


def build_joint_claims(macro_retention: pd.DataFrame, geometry_retention: pd.DataFrame, macro_df: pd.DataFrame, geometry_df: pd.DataFrame, pc_df: pd.DataFrame, gain_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    splits = sorted(set(macro_retention["split"].astype(str)) | set(geometry_retention["split"].astype(str)))
    for split in splits:
        macro_row = row_for(macro_df, split, "macro_only")
        model_row = row_for(geometry_df, split, "model_readout")
        rows.extend([
            {
                "split": split,
                "claim": "macro_bottleneck_retains_structure",
                "primary_value": retention_value(macro_retention, split, "macrostructure_composite_descriptive", "macro_retention_vs_full"),
                "secondary_value": retention_value(macro_retention, split, "drift_score", "macro_retention_vs_full"),
                "contrast_value": retention_value(macro_retention, split, "macrostructure_composite_descriptive", "macro_minus_residual"),
                "interpretation": "Approximate retention under the prespecified descriptive macrostructure score; drift retention is reported separately.",
            },
            {
                "split": split,
                "claim": "residual_hidden_loses_macrostructure",
                "primary_value": retention_value(macro_retention, split, "macrostructure_composite_descriptive", "residual_retention_vs_full"),
                "secondary_value": retention_value(macro_retention, split, "drift_score", "residual_retention_vs_full"),
                "contrast_value": retention_value(macro_retention, split, "macrostructure_composite_descriptive", "residual_retention_vs_macro"),
                "interpretation": "Residual-hidden recovery after linearly removing the predicted M/Psi component.",
            },
            {
                "split": split,
                "claim": "macro_bottleneck_task_retention",
                "primary_value": retention_value(macro_retention, split, "task_score", "macro_retention_vs_full"),
                "secondary_value": value(macro_row, "task_auc"),
                "contrast_value": retention_value(macro_retention, split, "task_score", "macro_minus_residual"),
                "interpretation": "Task information retained by the macro-only bottleneck; this is distinct from macrostructure retention.",
            },
            {
                "split": split,
                "claim": "linear_hidden_accesses_macrostate",
                "primary_value": retention_value(geometry_retention, split, "coordinate_score", "linear_retention_vs_model"),
                "secondary_value": retention_value(geometry_retention, split, "macrostructure_composite_descriptive", "linear_retention_vs_model"),
                "contrast_value": retention_value(geometry_retention, split, "coordinate_score", "linear_minus_residual"),
                "interpretation": "Linear hidden-state recovery relative to the trained model readout.",
            },
            {
                "split": split,
                "claim": "canonical_macro_alignment",
                "primary_value": value(model_row, "cca_corr_1"),
                "secondary_value": value(model_row, "cca_corr_2"),
                "contrast_value": np.nan,
                "interpretation": "Canonical correlations between the frozen hidden state and M/Psi.",
            },
        ])
        gain = gain_df[gain_df["split"].astype(str) == split] if not gain_df.empty else pd.DataFrame()
        rows.append({
            "split": split,
            "claim": "nonlinear_probe_limited_gain",
            "primary_value": value(gain.iloc[0], "nonlinear_gain_corr_M") if not gain.empty else np.nan,
            "secondary_value": value(gain.iloc[0], "nonlinear_gain_corr_Psi") if not gain.empty else np.nan,
            "contrast_value": np.nan,
            "interpretation": "Nonlinear-minus-linear coordinate recovery gain.",
        })
        pc_split = pc_df[pc_df["split"].astype(str) == split] if not pc_df.empty and "split" in pc_df.columns else pd.DataFrame()
        if not pc_split.empty:
            rows.append({
                "split": split,
                "claim": "leading_pc_macro_alignment",
                "primary_value": float(pd.to_numeric(pc_split["abs_corr_M"], errors="coerce").max()),
                "secondary_value": float(pd.to_numeric(pc_split["abs_corr_Psi"], errors="coerce").max()),
                "contrast_value": np.nan,
                "interpretation": "Largest absolute correlation of a reported leading PC with each macro-coordinate.",
            })
    return pd.DataFrame(rows)


def melt_numeric(df: pd.DataFrame, experiment: str, table_name: str, id_columns: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    ids = [column for column in id_columns if column in df.columns]
    numeric = [column for column in df.columns if column not in ids and pd.api.types.is_numeric_dtype(df[column])]
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        identity = {column: row.get(column) for column in ids}
        for metric in numeric:
            metric_value = finite_float(row.get(metric))
            if np.isfinite(metric_value):
                rows.append({"experiment": experiment, "table": table_name, **identity, "metric": metric, "value": metric_value})
    return pd.DataFrame(rows)


def validate_primary_coordinates(name: str, manifest: Mapping[str, Any]) -> None:
    if manifest.get("primary_coordinates") != ["M", "Psi"]:
        raise RuntimeError(f"{name}: primary_coordinates must be ['M', 'Psi'].")
    contract = manifest.get("stage1_fixed_k6_contract", manifest.get("fixed_k6_partition", {}))
    if isinstance(contract, Mapping) and contract:
        if contract.get("verified") is not True or int(contract.get("macrostate_k", -1)) != EXPECTED_K:
            raise RuntimeError(f"{name}: embedded fixed-K contract is invalid.")
        if contract.get("kmeans_refit") not in (None, False):
            raise RuntimeError(f"{name}: embedded contract reports kmeans_refit=true.")
        if contract.get("macrostate_k_selected") not in (None, False):
            raise RuntimeError(f"{name}: embedded contract reports macrostate_k_selected=true.")


def validate_fixed_k6_audit(name: str, audit: Mapping[str, Any]) -> Dict[str, Any]:
    k = int(audit.get("macrostate_k", audit.get("k", -1)))
    checks = {
        "verified": audit.get("verified") is True,
        "macrostate_k": k == EXPECTED_K,
        "macrostate_k_rule": audit.get("macrostate_k_rule") == "fixed a priori",
        "fit_split": audit.get("fit_split") == "A_train",
        "kmeans_refit": audit.get("kmeans_refit") is False,
        "macrostate_k_selected": audit.get("macrostate_k_selected") is False,
        "metadata_sha256": bool(str(audit.get("metadata_sha256", ""))),
        "centers_sha256": bool(str(audit.get("centers_sha256", ""))),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{name}: fixed-K audit failed: {failed}.")
    return {
        "macrostate_k": k,
        "metadata_sha256": audit.get("metadata_sha256"),
        "centers_sha256": audit.get("centers_sha256"),
        "fit_split": audit.get("fit_split"),
        "macrostate_k_rule": audit.get("macrostate_k_rule"),
        "checks": checks,
    }


def validate_representation_rows(df: pd.DataFrame, splits: Sequence[str], expected: Sequence[str], name: str) -> None:
    if df.empty or not {"split", "representation"}.issubset(df.columns):
        raise RuntimeError(f"{name}: metrics table lacks split/representation.")
    audit_column = "macrostate_partition_verified_against_stage1_fixed_k6"
    if audit_column not in df.columns:
        raise RuntimeError(f"{name}: metrics table lacks {audit_column}.")
    for split in splits:
        split_rows = df[df["split"].astype(str) == split]
        observed = set(split_rows["representation"].astype(str))
        missing = sorted(set(expected).difference(observed))
        if missing:
            raise RuntimeError(f"{name}: split {split} is missing representations {missing}.")
        verified = pd.to_numeric(split_rows[audit_column], errors="coerce")
        if not bool((verified == 1.0).all()):
            raise RuntimeError(f"{name}: split {split} contains rows not verified against Stage-1 fixed K=6.")


def read_source_table(name: str, base: Path, sources: List[SourceRecord], *, required: bool) -> pd.DataFrame:
    table, path = read_table(base, required=required)
    if path is None:
        sources.append(SourceRecord(name, None, "missing_allowed", note=str(base)))
        return table
    sources.append(SourceRecord(name, path, "ok", len(table), len(table.columns), file_sha256(path)))
    return table


def build_quality_gates(macro_df: pd.DataFrame, geometry_df: pd.DataFrame, macro_audit: Mapping[str, Any], geometry_audit: Mapping[str, Any], splits: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [
        {"scope": "macro_sufficiency", "gate": "fixed_k6_partition", "passed": int(macro_audit.get("macrostate_k", -1)) == EXPECTED_K, "detail": macro_audit.get("macrostate_k")},
        {"scope": "representation_geometry", "gate": "fixed_k6_partition", "passed": int(geometry_audit.get("macrostate_k", -1)) == EXPECTED_K, "detail": geometry_audit.get("macrostate_k")},
        {"scope": "score_contract", "gate": "rmse_scale_0p15", "passed": np.isclose(RMSE_SCORE_SCALE, 0.15), "detail": RMSE_SCORE_SCALE},
    ]
    for split in splits:
        macro_observed = set(macro_df[macro_df["split"].astype(str) == split]["representation"].astype(str))
        geometry_observed = set(geometry_df[geometry_df["split"].astype(str) == split]["representation"].astype(str))
        rows.append({"scope": f"macro_sufficiency/{split}", "gate": "required_representations", "passed": set(MACRO_REPRESENTATIONS).issubset(macro_observed), "detail": ";".join(sorted(macro_observed))})
        rows.append({"scope": f"representation_geometry/{split}", "gate": "required_representations", "passed": set(GEOMETRY_REPRESENTATIONS).issubset(geometry_observed), "detail": ";".join(sorted(geometry_observed))})
    hash_pairs = {
        (str(macro_audit.get("metadata_sha256")), str(macro_audit.get("centers_sha256"))),
        (str(geometry_audit.get("metadata_sha256")), str(geometry_audit.get("centers_sha256"))),
    }
    hash_pairs.discard(("None", "None"))
    rows.append({"scope": "joint", "gate": "same_stage1_partition", "passed": len(hash_pairs) <= 1, "detail": str(sorted(hash_pairs))})
    return pd.DataFrame(rows)


def validation_confirmation_stability(macro_df: pd.DataFrame, geometry_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metrics = KEY_RAW_METRICS + ["cca_corr_1", "cca_corr_2", "twonn_dimension", "participation_ratio_train", "effective_rank_train"]
    for experiment, df in (("macro_sufficiency", macro_df), ("representation_geometry", geometry_df)):
        for representation in df["representation"].dropna().astype(str).unique():
            validation = row_for(df, "A_val", representation)
            confirmation = row_for(df, "B_confirm", representation)
            for metric in metrics:
                validation_value = value(validation, metric)
                confirmation_value = value(confirmation, metric)
                delta = confirmation_value - validation_value if np.isfinite(validation_value) and np.isfinite(confirmation_value) else np.nan
                rows.append({
                    "experiment": experiment,
                    "representation": representation,
                    "metric": metric,
                    "validation_value": validation_value,
                    "confirmation_value": confirmation_value,
                    "confirmation_minus_validation": delta,
                    "absolute_gap": abs(delta) if np.isfinite(delta) else np.nan,
                })
    return pd.DataFrame(rows)


def add_ledger(rows: List[Dict[str, Any]], priority: str, category: str, metric: str, metric_value: Any, source: str, interpretation: str, manuscript_use: str, split: str = "B_confirm") -> None:
    rows.append({
        "priority": priority,
        "category": category,
        "split": split,
        "metric": metric,
        "value": metric_value,
        "source": source,
        "interpretation": interpretation,
        "manuscript_use": manuscript_use,
    })


def publication_metric_ledger(macro_retention: pd.DataFrame, geometry_retention: pd.DataFrame, macro_df: pd.DataFrame, geometry_df: pd.DataFrame, pc_df: pd.DataFrame, gain_df: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    split = "B_confirm"
    for domain, label in (
        ("macrostructure_composite_descriptive", "Macro-only composite retention"),
        ("coordinate_score", "Macro-only coordinate retention"),
        ("closure_score", "Macro-only closure retention"),
        ("transition_score", "Macro-only transition retention"),
        ("drift_score", "Macro-only drift retention"),
    ):
        add_ledger(rows, MAIN_REQUIRED if domain in {"macrostructure_composite_descriptive", "drift_score"} else MAIN_RECOMMENDED, "macro-sufficiency", label, retention_value(macro_retention, split, domain, "macro_retention_vs_full"), "Stage-5 macro retention table", "Retention of the prespecified descriptive domain score relative to full hidden state.", "Report with the domain definition; drift must be shown separately from the composite.")
    for domain, label in (
        ("macrostructure_composite_descriptive", "Residual-hidden composite retention"),
        ("drift_score", "Residual-hidden drift retention"),
        ("coordinate_score", "Residual-hidden coordinate retention"),
        ("transition_score", "Residual-hidden transition retention"),
    ):
        add_ledger(rows, MAIN_RECOMMENDED, "residual hidden", label, retention_value(macro_retention, split, domain, "residual_retention_vs_full"), "Stage-5 macro retention table", "Residual-hidden recovery after removing the predicted macrostate component.", "Use to delimit the bottleneck claim.")
    add_ledger(rows, MAIN_RECOMMENDED, "task retention", "Macro-only task retention", retention_value(macro_retention, split, "task_score", "macro_retention_vs_full"), "Stage-5 macro retention table", "Task score retained by macro-only features.", "Report separately from structural retention.")

    for domain, label in (
        ("coordinate_score", "Linear-hidden coordinate retention"),
        ("macrostructure_composite_descriptive", "Linear-hidden composite retention"),
        ("drift_score", "Linear-hidden drift retention"),
        ("transition_score", "Linear-hidden transition retention"),
    ):
        add_ledger(rows, MAIN_REQUIRED if domain in {"coordinate_score", "macrostructure_composite_descriptive"} else MAIN_RECOMMENDED, "representation geometry", label, retention_value(geometry_retention, split, domain, "linear_retention_vs_model"), "Stage-5 geometry retention table", "Linear hidden-state recovery relative to the trained model readout.", "Report in the hidden-space organization paragraph.")

    model_row = row_for(geometry_df, split, "model_readout")
    for metric, label in (("cca_corr_1", "First canonical correlation"), ("cca_corr_2", "Second canonical correlation"), ("twonn_dimension", "TwoNN intrinsic-dimension estimate"), ("participation_ratio_train", "Participation ratio"), ("effective_rank_train", "Effective rank"), ("sample_rows_geometry", "Geometry sample rows")):
        add_ledger(rows, MAIN_REQUIRED if metric in {"cca_corr_1", "cca_corr_2"} else MAIN_RECOMMENDED, "representation geometry", label, value(model_row, metric), "Stage-5 representation-geometry metrics", "Frozen hidden-space geometry diagnostic.", "Report CCA in the main text; keep dimension estimators in Additional information.")

    pc_split = pc_df[pc_df["split"].astype(str) == split] if not pc_df.empty else pd.DataFrame()
    if not pc_split.empty:
        add_ledger(rows, MAIN_RECOMMENDED, "principal components", "Maximum absolute PC correlation with M", float(pd.to_numeric(pc_split["abs_corr_M"], errors="coerce").max()), "Stage-5 PC correlation table", "Largest reported leading-PC alignment with M.", "Report with the corresponding PC index if space permits.")
        add_ledger(rows, MAIN_RECOMMENDED, "principal components", "Maximum absolute PC correlation with Psi", float(pd.to_numeric(pc_split["abs_corr_Psi"], errors="coerce").max()), "Stage-5 PC correlation table", "Largest reported leading-PC alignment with Psi.", "Report with the corresponding PC index if space permits.")

    gain = gain_df[gain_df["split"].astype(str) == split] if not gain_df.empty else pd.DataFrame()
    if not gain.empty:
        for metric in ("nonlinear_gain_corr_M", "nonlinear_gain_corr_Psi", "nonlinear_gain_rmse_reduction_M", "nonlinear_gain_rmse_reduction_Psi"):
            add_ledger(rows, MAIN_RECOMMENDED, "nonlinear probe", metric, value(gain.iloc[0], metric), "Stage-5 nonlinear-gain table", "Nonlinear probe improvement over the linear hidden-state probe.", "Report to support approximate linear accessibility.")

    for experiment, representation, metric in (
        ("macro_sufficiency", "macro_only", "macrostructure_composite_descriptive"),
        ("representation_geometry", "linear_hidden", "coordinate_corr_M"),
        ("representation_geometry", "linear_hidden", "coordinate_corr_Psi"),
    ):
        subset = stability[(stability["experiment"] == experiment) & (stability["representation"] == representation) & (stability["metric"] == metric)]
        if not subset.empty:
            add_ledger(rows, MAIN_RECOMMENDED, "validation-confirmation stability", f"absolute gap {experiment}/{representation}/{metric}", subset.iloc[0].get("absolute_gap"), "Stage-5 stability table", "Absolute validation-to-confirmation gap.", "Use with both split values when discussing stability.", "A_val_to_B_confirm")
    return pd.DataFrame(rows)


def source_audit_table(sources: Sequence[SourceRecord]) -> pd.DataFrame:
    return pd.DataFrame([{
        "name": source.name,
        "status": source.status,
        "rows": source.rows,
        "columns": source.columns,
        "path": str(source.path) if source.path else source.note,
        "sha256": source.sha256,
    } for source in sources])


def markdown_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "_No rows available._"
    return df.head(max_rows).to_markdown(index=False)


def write_report(path: Path, audit: pd.DataFrame, quality: pd.DataFrame, ledger: pd.DataFrame, joint: pd.DataFrame, macro_retention: pd.DataFrame, geometry_retention: pd.DataFrame, gain_df: pd.DataFrame, pc_df: pd.DataFrame, stability: pd.DataFrame) -> None:
    lines: List[str] = [
        "# Stage 5 joint macro-sufficiency and representation-geometry report",
        "",
        "This report is generated from frozen Stage-5 outputs. It does not retrain Event-SSL, refit probes, recompute predictions, refit the mesostate partition, or redefine the primary `(M, Psi)` state.",
        "",
        "## Numerical rigor and interpretation boundary",
        "",
        f"- Correlations and cosines are mapped to `(r+1)/2`; RMSE values are mapped to `1/(1+RMSE/{RMSE_SCORE_SCALE})`; JS and row-TV are mapped to `1-x`.",
        "- Domain scores, composite scores and retention ratios are descriptive summaries. Raw metrics remain the primary numerical evidence.",
        "- The composite is the mean of coordinate, closure, drift and transition scores. Drift retention is always reported separately.",
        "- Macro-only uses the fixed feature basis `(M, Psi, M^2, Psi^2, M*Psi)` derived from the same two-coordinate state; it does not introduce extra state variables.",
        "- Residual-hidden analyses are regression residual diagnostics, not causal interventions or proofs of statistical sufficiency.",
        "- A_val and B_confirm are frozen point estimates. Random-seed uncertainty is not inferred from a single trained checkpoint.",
        "- All transition metrics use the same Stage-1 fixed `K=6` partition; KMeans is not refit and `K` is not reselected.",
        "",
        "## Input audit",
        "",
        markdown_table(audit, 40),
        "",
        "## Scientific quality gates",
        "",
        markdown_table(quality, 60),
        "",
        "## Main-text numerical ledger",
        "",
        markdown_table(ledger[ledger["priority"].isin([MAIN_REQUIRED, MAIN_RECOMMENDED])], 200),
        "",
        "## Strong conclusion logic",
        "",
        "- **Macro bottleneck:** assess domain-specific retention relative to full hidden state; do not treat the composite alone as full dynamical sufficiency.",
        "- **Residual hidden:** assess how much macrostructure remains after removing the predicted macrostate component.",
        "- **Representation geometry:** compare linear probes with the model readout, then report CCA, leading-PC alignment and nonlinear gain.",
    ]
    for split in sorted(joint["split"].dropna().astype(str).unique()):
        lines.extend([
            "", f"## {split}: claim contrasts", "", markdown_table(joint[joint["split"] == split], 50),
            "", f"### {split}: macro-sufficiency retention", "",
            markdown_table(macro_retention[(macro_retention["split"] == split) & macro_retention["domain"].isin(["coordinate_score", "closure_score", "drift_score", "transition_score", "task_score", "macro_label_score", "macrostructure_composite_descriptive"])], 50),
            "", f"### {split}: representation-geometry retention", "",
            markdown_table(geometry_retention[(geometry_retention["split"] == split) & geometry_retention["domain"].isin(["coordinate_score", "closure_score", "drift_score", "transition_score", "macrostructure_composite_descriptive"])], 50),
        ])
    lines.extend([
        "", "## Nonlinear probe gain", "", markdown_table(gain_df, 20),
        "", "## PC macro correlations", "", markdown_table(pc_df, 80),
        "", "## Validation-confirmation stability", "", markdown_table(stability, 300),
        "", "## Supplementary numerical ledger", "", markdown_table(ledger[ledger["priority"] == SUPPLEMENT_REQUIRED], 200),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract formal Stage-5 Event-SSL statistics.")
    parser.add_argument("--macro-root", type=Path, default=DEFAULT_MACRO_ROOT)
    parser.add_argument("--macro-train-root", type=Path, default=DEFAULT_MACRO_TRAIN_ROOT)
    parser.add_argument("--geometry-root", type=Path, default=DEFAULT_GEOMETRY_ROOT)
    parser.add_argument("--geometry-train-root", type=Path, default=DEFAULT_GEOMETRY_TRAIN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--allow-missing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    table_root = output_root / "tables"
    report_root = output_root / "reports"
    metadata_root = output_root / "metadata"
    for directory in (table_root, report_root, metadata_root):
        directory.mkdir(parents=True, exist_ok=True)

    sources: List[SourceRecord] = []
    macro_df = read_source_table("macro_sufficiency_metrics", args.macro_root / "tables" / "stage5_macro_sufficiency_metrics_all_splits", sources, required=not args.allow_missing)
    macro_train_df = read_source_table("macro_sufficiency_training_metrics", args.macro_train_root / "tables" / "stage5_macro_sufficiency_training_probe_metrics", sources, required=False)
    geometry_df = read_source_table("representation_geometry_metrics", args.geometry_root / "tables" / "stage5_representation_geometry_metrics_all_splits", sources, required=not args.allow_missing)
    pc_df = read_source_table("representation_geometry_pc_correlations", args.geometry_root / "tables" / "stage5_representation_geometry_pc_macro_correlations", sources, required=False)
    gain_df = read_source_table("representation_geometry_nonlinear_gain", args.geometry_root / "tables" / "stage5_representation_geometry_nonlinear_probe_gain", sources, required=False)
    pc_train_df = read_source_table("representation_geometry_train_pc_correlations", args.geometry_train_root / "tables" / "stage5_pc_macro_correlations_train", sources, required=False)
    if macro_df.empty or geometry_df.empty:
        raise RuntimeError("Stage-5 macro-sufficiency and representation-geometry metrics are required.")

    manifest_paths = {
        "macro_eval": args.macro_root / "metadata" / "stage5_macro_sufficiency_evaluation_manifest.json",
        "macro_train": args.macro_train_root / "metadata" / "stage5_macro_sufficiency_training_manifest.json",
        "geometry_eval": args.geometry_root / "metadata" / "stage5_representation_geometry_evaluation_manifest.json",
        "geometry_train": args.geometry_train_root / "metadata" / "stage5_representation_geometry_training_manifest.json",
        "macro_partition": args.macro_root / "metadata" / "stage5_macro_sufficiency_fixed_k6_partition_audit.json",
        "geometry_partition": args.geometry_root / "metadata" / "stage5_representation_geometry_fixed_k6_partition_audit.json",
    }
    manifests: Dict[str, Dict[str, Any]] = {}
    for name, path in manifest_paths.items():
        required = name in {"macro_eval", "macro_train", "geometry_eval", "geometry_train", "macro_partition", "geometry_partition"}
        manifest = load_json(path, required=required and not args.allow_missing)
        manifests[name] = manifest
        if manifest:
            sources.append(SourceRecord(name, path, "ok", sha256=file_sha256(path)))

    for name in ("macro_eval", "macro_train", "geometry_eval", "geometry_train"):
        if manifests[name]:
            validate_primary_coordinates(name, manifests[name])
    macro_partition = validate_fixed_k6_audit("macro_sufficiency", manifests["macro_partition"])
    geometry_partition = validate_fixed_k6_audit("representation_geometry", manifests["geometry_partition"])
    validate_representation_rows(macro_df, args.splits, MACRO_REPRESENTATIONS, "macro_sufficiency")
    validate_representation_rows(geometry_df, args.splits, GEOMETRY_REPRESENTATIONS, "representation_geometry")

    macro_domain = add_domain_scores(macro_df, "macro_sufficiency")
    geometry_domain = add_domain_scores(geometry_df, "representation_geometry")
    macro_retention = build_macro_retention(macro_df)
    geometry_retention = build_geometry_retention(geometry_df, gain_df)
    joint = build_joint_claims(macro_retention, geometry_retention, macro_df, geometry_df, pc_df, gain_df)
    stability = validation_confirmation_stability(macro_df, geometry_df)
    quality = build_quality_gates(macro_df, geometry_df, macro_partition, geometry_partition, args.splits)
    ledger = publication_metric_ledger(macro_retention, geometry_retention, macro_df, geometry_df, pc_df, gain_df, stability)

    long_parts = [
        melt_numeric(macro_df, "macro_sufficiency", "metrics_all_splits", ["split", "representation"]),
        melt_numeric(macro_train_df, "macro_sufficiency", "training_probe_metrics", ["split", "representation"]),
        melt_numeric(geometry_df, "representation_geometry", "metrics_all_splits", ["split", "representation"]),
        melt_numeric(pc_df, "representation_geometry", "pc_macro_correlations", ["split", "component"]),
        melt_numeric(gain_df, "representation_geometry", "nonlinear_probe_gain", ["split"]),
        melt_numeric(pc_train_df, "representation_geometry", "training_pc_macro_correlations", ["component"]),
    ]
    long_df = pd.concat([part for part in long_parts if not part.empty], ignore_index=True, sort=False) if any(not part.empty for part in long_parts) else pd.DataFrame()
    source_audit = source_audit_table(sources)
    score_contract = pd.DataFrame([
        {"metric_class": "correlation_or_cosine", "transformation": "(r+1)/2", "role": "descriptive score"},
        {"metric_class": "RMSE", "transformation": f"1/(1+RMSE/{RMSE_SCORE_SCALE})", "role": "descriptive score"},
        {"metric_class": "JS_or_row_TV", "transformation": "1-x", "role": "descriptive score"},
        {"metric_class": "task_BCE", "transformation": "1/(1+BCE)", "role": "descriptive score"},
        {"metric_class": "macrostructure_composite", "transformation": "mean(coordinate, closure, drift, transition)", "role": "descriptive score"},
    ])

    outputs = {
        "macro_wide": write_table(macro_df, table_root / "stage5_macro_sufficiency_metrics_wide"),
        "macro_train": write_table(macro_train_df, table_root / "stage5_macro_sufficiency_training_metrics"),
        "geometry_wide": write_table(geometry_df, table_root / "stage5_representation_geometry_metrics_wide"),
        "pc": write_table(pc_df, table_root / "stage5_representation_geometry_pc_macro_correlations"),
        "pc_train": write_table(pc_train_df, table_root / "stage5_representation_geometry_train_pc_macro_correlations"),
        "gain": write_table(gain_df, table_root / "stage5_representation_geometry_nonlinear_gain"),
        "macro_domain": write_table(macro_domain, table_root / "stage5_macro_sufficiency_domain_scores"),
        "geometry_domain": write_table(geometry_domain, table_root / "stage5_representation_geometry_domain_scores"),
        "macro_retention": write_table(macro_retention, table_root / "stage5_macro_sufficiency_retention"),
        "geometry_retention": write_table(geometry_retention, table_root / "stage5_representation_geometry_retention"),
        "joint_claims": write_table(joint, table_root / "stage5_joint_claim_contrasts"),
        "long": write_table(long_df, table_root / "stage5_joint_long_metrics"),
        "input_audit": write_table(source_audit, table_root / "stage5_joint_input_audit"),
        "stability": write_table(stability, table_root / "stage5_validation_confirmation_stability"),
        "quality": write_table(quality, table_root / "stage5_scientific_quality_gates"),
        "ledger": write_table(ledger, table_root / "stage5_publication_metric_ledger"),
        "score_contract": write_table(score_contract, table_root / "stage5_descriptive_score_contract"),
    }
    report_path = report_root / "stage5_joint_macro_geometry_report.md"
    write_report(report_path, source_audit, quality, ledger, joint, macro_retention, geometry_retention, gain_df, pc_df, stability)

    if not quality.empty and not bool(quality["passed"].all()):
        failed = quality[~quality["passed"]]
        raise RuntimeError(f"Stage-5 quality gates failed:\n{failed.to_string(index=False)}")

    save_json({
        "script": Path(__file__).name,
        "macro_root": str(args.macro_root.resolve()),
        "macro_train_root": str(args.macro_train_root.resolve()),
        "geometry_root": str(args.geometry_root.resolve()),
        "geometry_train_root": str(args.geometry_train_root.resolve()),
        "output_root": str(output_root),
        "primary_coordinates": ["M", "Psi"],
        "fixed_macrostate_k": EXPECTED_K,
        "kmeans_refit": False,
        "macrostate_k_selected": False,
        "score_contract": score_contract.to_dict(orient="records"),
        "generated_tables": {name: str(path.resolve()) for name, path in outputs.items()},
        "report": str(report_path.resolve()),
        "numerical_contract": {
            "raw_metrics_primary": True,
            "derived_scores_descriptive": True,
            "single_checkpoint_point_estimates": True,
            "seed_uncertainty_inferred": False,
            "residualization_is_diagnostic_not_causal": True,
        },
    }, metadata_root / "stage5_joint_macro_geometry_collection_manifest.json")
    print(f"[Stage5 statistics] wrote {report_path}")


if __name__ == "__main__":
    main()
