#!/usr/bin/env python3
"""Collect publication statistics from frozen Stage-4 Event-SSL evaluations."""

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
DEFAULT_BASE = Path(os.environ.get("EDNET_KT4_OUTPUT_ROOT", "/data/datasets/KT4/outputs_KT4"))
DEFAULT_OUTPUT_ROOT = DEFAULT_BASE / "stage4_event_ssl" / "all_experiment_comparison"
DEFAULT_SPLITS = ("A_val", "B_confirm")
EXPECTED_K = 6

MAIN_REQUIRED = "main_text_required"
MAIN_RECOMMENDED = "main_text_recommended"
SUPPLEMENT_REQUIRED = "supplement_required"

HIGHER_IS_BETTER = {
    "coordinate_corr_M", "coordinate_corr_Psi", "one_step_corr_M", "one_step_corr_Psi",
    "current_state_occupancy_overlap", "next_state_occupancy_overlap",
    "anchor_drift_vector_corr", "anchor_drift_speed_corr",
    "anchor_mean_local_drift_cosine", "anchor_occupancy_weighted_local_drift_cosine",
    "anchor_fraction_cells_cosine_gt_0p8",
    "learned_plane_drift_vector_corr", "learned_plane_drift_speed_corr",
    "learned_plane_mean_local_drift_cosine", "learned_plane_occupancy_weighted_local_drift_cosine",
    "learned_plane_fraction_cells_cosine_gt_0p8",
    "empirical_negative_divergence_weighted_fraction", "empirical_inward_fraction_to_reference",
    "anchor_negative_divergence_weighted_fraction", "anchor_inward_fraction_to_reference",
    "learned_plane_negative_divergence_weighted_fraction", "learned_plane_inward_fraction_to_reference",
    "anchor_self_transition_corr", "anchor_diagonal_dominance_match_fraction",
    "anchor_top_transition_edge_overlap", "learned_plane_self_transition_corr",
    "learned_plane_diagonal_dominance_match_fraction", "learned_plane_top_transition_edge_overlap",
    "task_auc_binary_rows", "task_auc_thresholded_all_rows",
    "task_accuracy_at_0p5_binary_rows", "task_accuracy_at_0p5_thresholded_all_rows",
    "task_prob_target_corr",
}
LOWER_IS_BETTER = {
    "coordinate_rmse_M", "coordinate_rmse_Psi", "coordinate_mae_M", "coordinate_mae_Psi",
    "one_step_rmse_M", "one_step_rmse_Psi",
    "current_state_occupancy_js", "next_state_occupancy_js",
    "anchor_drift_local_rmse", "learned_plane_drift_local_rmse",
    "anchor_high_support_residual_mean", "anchor_low_support_residual_mean",
    "learned_plane_high_support_residual_mean", "learned_plane_low_support_residual_mean",
    "anchor_transition_mean_row_tv", "anchor_transition_max_row_tv",
    "anchor_self_transition_rmse", "anchor_self_transition_mae",
    "learned_plane_transition_mean_row_tv", "learned_plane_transition_max_row_tv",
    "learned_plane_self_transition_rmse", "learned_plane_self_transition_mae",
    "task_bce", "task_rmse", "task_mae",
}

CORE_METRICS = [
    "n_rows", "n_users",
    "coordinate_corr_M", "coordinate_corr_Psi",
    "coordinate_rmse_M", "coordinate_rmse_Psi",
    "one_step_rmse_M", "one_step_rmse_Psi",
    "current_state_occupancy_js", "next_state_occupancy_js",
    "anchor_drift_vector_corr", "anchor_occupancy_weighted_local_drift_cosine",
    "learned_plane_drift_vector_corr", "learned_plane_occupancy_weighted_local_drift_cosine",
    "anchor_inward_fraction_to_reference", "learned_plane_inward_fraction_to_reference",
    "anchor_negative_divergence_weighted_fraction", "learned_plane_negative_divergence_weighted_fraction",
    "learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr",
    "learned_plane_diagonal_dominance_match_fraction", "learned_plane_top_transition_edge_overlap",
    "task_auc_binary_rows", "task_auc_thresholded_all_rows", "task_bce",
]

DOMAIN_METRICS = {
    "coordinate": ["coordinate_corr_M", "coordinate_corr_Psi", "coordinate_rmse_M", "coordinate_rmse_Psi"],
    "one_step_closure": ["one_step_rmse_M", "one_step_rmse_Psi", "one_step_corr_M", "one_step_corr_Psi"],
    "landscape": ["current_state_occupancy_js", "next_state_occupancy_js", "current_state_occupancy_overlap", "next_state_occupancy_overlap"],
    "anchor_drift": ["anchor_drift_vector_corr", "anchor_drift_speed_corr", "anchor_occupancy_weighted_local_drift_cosine", "anchor_drift_local_rmse"],
    "learned_plane_drift": ["learned_plane_drift_vector_corr", "learned_plane_drift_speed_corr", "learned_plane_occupancy_weighted_local_drift_cosine", "learned_plane_drift_local_rmse"],
    "convergence": ["anchor_inward_fraction_to_reference", "learned_plane_inward_fraction_to_reference", "anchor_negative_divergence_weighted_fraction", "learned_plane_negative_divergence_weighted_fraction"],
    "transition": ["learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr", "learned_plane_diagonal_dominance_match_fraction", "learned_plane_top_transition_edge_overlap"],
    "task": ["task_auc_binary_rows", "task_auc_thresholded_all_rows", "task_accuracy_at_0p5_binary_rows", "task_bce"],
}

MODEL_SPECS = [
    {
        "label": "predictive_state_event_ssl",
        "default_root": DEFAULT_BASE / "stage4_event_ssl" / "evaluation_predictive_state",
        "metrics_base": "stage4_event_ssl_structural_metrics_all_splits",
        "matrix_prefix": "stage4_event_ssl_transition_matrices",
        "manifest_name": "stage4_event_ssl_evaluation_manifest.json",
        "audit_name": "stage4_event_ssl_fixed_k6_partition_audit.json",
        "expected_kind": "predictive_state",
        "role": "main predictive-state Event-SSL",
    },
    {
        "label": "pure_event_ssl_probe",
        "default_root": DEFAULT_BASE / "stage4_event_ssl" / "evaluation_pure_ssl_probe",
        "metrics_base": "stage4_event_ssl_structural_metrics_all_splits",
        "matrix_prefix": "stage4_event_ssl_transition_matrices",
        "manifest_name": "stage4_event_ssl_evaluation_manifest.json",
        "audit_name": "stage4_event_ssl_fixed_k6_partition_audit.json",
        "expected_kind": "pure_ssl",
        "role": "future-predictive SSL with a frozen development probe",
    },
    {
        "label": "task_only",
        "default_root": DEFAULT_BASE / "stage4_event_ssl" / "controls" / "task_only" / "evaluation",
        "metrics_base": "stage4_task_only_structural_metrics_all_splits",
        "task_metrics_base": "stage4_task_only_task_metrics_all_splits",
        "matrix_prefix": "stage4_task_only_transition_matrices",
        "manifest_name": "stage4_task_only_evaluation_manifest.json",
        "audit_name": "stage4_task_only_fixed_k6_partition_audit.json",
        "expected_kind": "task_only",
        "role": "response-task-only sequential control",
    },
    {
        "label": "time_shuffle_control",
        "default_root": DEFAULT_BASE / "stage4_event_ssl_time_shuffle_control" / "evaluation_on_ordered_inputs",
        "metrics_base": "stage4_event_ssl_structural_metrics_all_splits",
        "matrix_prefix": "stage4_event_ssl_transition_matrices",
        "manifest_name": "stage4_event_ssl_evaluation_manifest.json",
        "audit_name": "stage4_event_ssl_fixed_k6_partition_audit.json",
        "control_manifest_name": "stage4_time_shuffle_control_manifest.json",
        "expected_control_type": "within_user_time_shuffle",
        "expected_kind": "predictive_state",
        "role": "within-user time-shuffle control evaluated on ordered inputs",
    },
    {
        "label": "tag_support_randomized",
        "default_root": DEFAULT_BASE / "stage4_event_ssl_tag_support_randomized_control" / "evaluation",
        "metrics_base": "stage4_event_ssl_structural_metrics_all_splits",
        "matrix_prefix": "stage4_event_ssl_transition_matrices",
        "manifest_name": "stage4_event_ssl_evaluation_manifest.json",
        "audit_name": "stage4_event_ssl_fixed_k6_partition_audit.json",
        "control_manifest_name": "stage4_tag_support_randomization_control_manifest.json",
        "expected_control_type": "tag_support_alignment_randomization",
        "expected_kind": "predictive_state",
        "role": "tag/support-alignment randomization control",
    },
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


def metric_direction(metric: str) -> str:
    if metric in HIGHER_IS_BETTER:
        return "higher"
    if metric in LOWER_IS_BETTER:
        return "lower"
    if metric.endswith("_corr") or "cosine" in metric or "overlap" in metric or "fraction" in metric:
        return "higher"
    if any(token in metric for token in ("rmse", "mae", "js", "tv", "loss", "bce")):
        return "lower"
    return "unknown"


def normalized_metric_value(value: float, metric: str) -> float:
    if not np.isfinite(value):
        return np.nan
    direction = metric_direction(metric)
    if direction == "higher":
        return float(np.clip(value, 0.0, 1.0))
    if direction == "lower":
        if "js" in metric or "bce" in metric:
            return float(np.exp(-max(value, 0.0)))
        if "tv" in metric:
            return float(np.clip(1.0 - value, 0.0, 1.0))
        return float(1.0 / (1.0 + max(value, 0.0)))
    return np.nan


def relative_to_reference(reference: float, value: float, metric: str) -> float:
    if not np.isfinite(reference) or not np.isfinite(value):
        return np.nan
    direction = metric_direction(metric)
    if direction == "higher":
        return float(value / max(abs(reference), EPS))
    if direction == "lower":
        return float(reference / max(abs(value), EPS))
    return np.nan


def validate_manifest_contract(label: str, manifest: Mapping[str, Any], metrics: pd.DataFrame, expected_kind: str) -> Dict[str, Any]:
    primary = manifest.get("primary_coordinates")
    if primary != ["M", "Psi"]:
        raise RuntimeError(f"{label}: primary_coordinates must be ['M', 'Psi']; found {primary!r}.")
    kind = str(manifest.get("model_kind") or manifest.get("control_name", ""))
    if expected_kind == "task_only":
        if kind not in {"task_only", "task_only_control"}:
            raise RuntimeError(f"{label}: task-only manifest kind is {kind!r}.")
    elif kind != expected_kind:
        raise RuntimeError(f"{label}: expected model_kind={expected_kind!r}; found {kind!r}.")
    guardrails = dict(manifest.get("guardrails", {}))
    checks = {
        "primary_coordinates_M_Psi": primary == ["M", "Psi"],
        "kmeans_refit_false": guardrails.get("kmeans_refit") is False,
        "macrostate_k_selected_false": guardrails.get("macrostate_k_selected") is False,
        "macrostate_k_fixed_6": int(guardrails.get("macrostate_k", -1)) == EXPECTED_K,
        "B_confirm_not_used_for_update": guardrails.get("B_confirm_used_for_update") is False,
    }
    required_metric_flags = {
        "macrostate_partition_verified_against_stage1_fixed_k6": 1.0,
        "macrostate_k_fixed_a_priori": 1.0,
    }
    for column, expected in required_metric_flags.items():
        present = column in metrics.columns
        checks[f"metrics_{column}"] = present and bool((pd.to_numeric(metrics[column], errors="coerce") == expected).all())
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{label}: formal evaluation contract failed: {failed}")
    return {"model_kind": kind, "guardrails": guardrails, "checks": checks}


def validate_partition_audit(label: str, audit: Mapping[str, Any]) -> Dict[str, Any]:
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
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{label}: fixed partition audit failed: {failed}.")
    return {
        "macrostate_k": k,
        "metadata_sha256": audit.get("metadata_sha256"),
        "centers_sha256": audit.get("centers_sha256"),
        "fit_split": audit.get("fit_split"),
        "macrostate_k_rule": audit.get("macrostate_k_rule"),
        "checks": checks,
    }


def load_model(spec: Mapping[str, Any], root: Path, splits: Sequence[str], allow_missing: bool) -> Tuple[pd.DataFrame, Dict[str, Any], List[SourceRecord]]:
    sources: List[SourceRecord] = []
    metrics, metrics_path = read_table(root / "tables" / str(spec["metrics_base"]), required=not allow_missing)
    if metrics_path is None or metrics.empty:
        return pd.DataFrame(), {"model_label": spec["label"], "status": "missing_allowed"}, sources
    sources.append(SourceRecord(str(spec["metrics_base"]), metrics_path, "ok", len(metrics), len(metrics.columns), file_sha256(metrics_path)))
    if "split" not in metrics.columns:
        raise RuntimeError(f"{spec['label']}: metrics table lacks split.")
    if metrics["split"].astype(str).duplicated().any():
        raise RuntimeError(f"{spec['label']}: duplicate split rows in metrics table.")
    missing_splits = sorted(set(map(str, splits)).difference(set(metrics["split"].astype(str))))
    if missing_splits:
        raise RuntimeError(f"{spec['label']}: missing requested splits {missing_splits}.")

    task_metrics = pd.DataFrame()
    if spec.get("task_metrics_base"):
        task_metrics, task_path = read_table(root / "tables" / str(spec["task_metrics_base"]), required=False)
        if task_path is not None:
            sources.append(SourceRecord(str(spec["task_metrics_base"]), task_path, "ok", len(task_metrics), len(task_metrics.columns), file_sha256(task_path)))
        if not task_metrics.empty and "split" in task_metrics.columns:
            extra = [c for c in task_metrics.columns if c != "split" and c not in metrics.columns]
            metrics = metrics.merge(task_metrics[["split"] + extra], on="split", how="left")

    manifest_path = root / "metadata" / str(spec["manifest_name"])
    audit_path = root / "metadata" / str(spec["audit_name"])
    manifest = load_json(manifest_path, required=not allow_missing)
    audit = load_json(audit_path, required=not allow_missing)
    if not manifest or not audit:
        return pd.DataFrame(), {"model_label": spec["label"], "status": "missing_allowed"}, sources
    sources.extend([
        SourceRecord(str(spec["manifest_name"]), manifest_path, "ok", sha256=file_sha256(manifest_path)),
        SourceRecord(str(spec["audit_name"]), audit_path, "ok", sha256=file_sha256(audit_path)),
    ])
    contract = validate_manifest_contract(str(spec["label"]), manifest, metrics, str(spec["expected_kind"]))
    partition = validate_partition_audit(str(spec["label"]), audit)

    control_manifest = {}
    if spec.get("control_manifest_name"):
        control_path = root / "metadata" / str(spec["control_manifest_name"])
        control_manifest = load_json(control_path, required=not allow_missing)
        if control_manifest:
            if control_manifest.get("primary_coordinates") != ["M", "Psi"]:
                raise RuntimeError(f"{spec['label']}: control manifest does not use M/Psi.")
            expected_control_type = str(spec.get("expected_control_type", ""))
            if expected_control_type and str(control_manifest.get("control_type", "")) != expected_control_type:
                raise RuntimeError(f"{spec['label']}: unexpected control_type={control_manifest.get('control_type')!r}.")
            control_guardrails = dict(control_manifest.get("guardrails", {}))
            if control_guardrails.get("kmeans_refit") is not False or control_guardrails.get("macrostate_k_selected") is not False:
                raise RuntimeError(f"{spec['label']}: control manifest violates fixed-K guardrails.")
            if int(control_guardrails.get("macrostate_k", -1)) != EXPECTED_K:
                raise RuntimeError(f"{spec['label']}: control manifest does not retain fixed K=6.")
            sources.append(SourceRecord(str(spec["control_manifest_name"]), control_path, "ok", sha256=file_sha256(control_path)))

    metrics = metrics.copy()
    metrics.insert(0, "model_label", str(spec["label"]))
    metrics.insert(1, "experiment_root", str(root.resolve()))
    model_audit = {
        "model_label": spec["label"],
        "scientific_role": spec["role"],
        "root": str(root.resolve()),
        "metrics_rows": int(len(metrics)),
        "metrics_columns": int(len(metrics.columns)),
        "manifest_path": str(manifest_path.resolve()),
        "partition_audit_path": str(audit_path.resolve()),
        "contract": contract,
        "partition": partition,
        "control_manifest_present": bool(control_manifest),
        "status": "verified",
    }
    return metrics, model_audit, sources


def collect_all(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, List[SourceRecord]]:
    root_map = {
        "predictive_state_event_ssl": args.main_root,
        "pure_event_ssl_probe": args.pure_root,
        "task_only": args.task_root,
        "time_shuffle_control": args.time_shuffle_root,
        "tag_support_randomized": args.tag_support_root,
    }
    frames: List[pd.DataFrame] = []
    audits: List[Dict[str, Any]] = []
    sources: List[SourceRecord] = []
    for spec in MODEL_SPECS:
        root = Path(root_map[str(spec["label"])])
        metrics, audit, model_sources = load_model(spec, root, args.splits, args.allow_missing)
        sources.extend(model_sources)
        audits.append(audit)
        if not metrics.empty:
            frames.append(metrics)
    if not frames:
        raise RuntimeError("No Stage-4 evaluation metrics were loaded.")
    all_metrics = pd.concat(frames, ignore_index=True, sort=False)
    partition_pairs = {
        (str(row.get("partition", {}).get("metadata_sha256")), str(row.get("partition", {}).get("centers_sha256")))
        for row in audits if row.get("status") == "verified"
    }
    partition_pairs.discard(("None", "None"))
    if len(partition_pairs) > 1:
        raise RuntimeError("Stage-4 evaluations do not use the same frozen Stage-1 K=6 partition.")
    return all_metrics, pd.DataFrame(audits), sources


def make_long_metrics(all_metrics: pd.DataFrame, splits: Sequence[str]) -> pd.DataFrame:
    id_columns = {"model_label", "experiment_root", "split"}
    numeric = [c for c in all_metrics.columns if c not in id_columns and pd.api.types.is_numeric_dtype(all_metrics[c])]
    rows: List[Dict[str, Any]] = []
    for _, row in all_metrics.iterrows():
        split = str(row.get("split", ""))
        if split not in splits:
            continue
        for metric in numeric:
            value = finite_float(row.get(metric))
            if not np.isfinite(value):
                continue
            rows.append({
                "model_label": row["model_label"],
                "split": split,
                "metric": metric,
                "value": value,
                "direction": metric_direction(metric),
                "normalized_value": normalized_metric_value(value, metric),
            })
    return pd.DataFrame(rows)


def add_relative_to_main(long_metrics: pd.DataFrame) -> pd.DataFrame:
    if long_metrics.empty:
        return long_metrics.copy()
    main = long_metrics[long_metrics["model_label"] == "predictive_state_event_ssl"]
    lookup = {(str(r.split), str(r.metric)): float(r.value) for r in main.itertuples(index=False)}
    rows: List[Dict[str, Any]] = []
    for row in long_metrics.itertuples(index=False):
        reference = lookup.get((str(row.split), str(row.metric)), np.nan)
        relative = relative_to_reference(reference, float(row.value), str(row.metric))
        rows.append({
            **row._asdict(),
            "main_value": reference,
            "relative_to_main": relative,
            "relative_to_main_clipped_0_2": float(np.clip(relative, 0.0, 2.0)) if np.isfinite(relative) else np.nan,
        })
    return pd.DataFrame(rows)


def domain_scores(all_metrics: pd.DataFrame, splits: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in all_metrics.iterrows():
        split = str(row.get("split", ""))
        if split not in splits:
            continue
        model = str(row.get("model_label", ""))
        model_rows: List[Dict[str, Any]] = []
        for domain, metrics in DOMAIN_METRICS.items():
            values: List[float] = []
            used: List[str] = []
            for metric in metrics:
                if metric not in row.index:
                    continue
                score = normalized_metric_value(finite_float(row.get(metric)), metric)
                if np.isfinite(score):
                    values.append(score)
                    used.append(metric)
            if values:
                model_rows.append({
                    "split": split,
                    "model_label": model,
                    "domain": domain,
                    "domain_score": float(np.mean(values)),
                    "metrics_used": ";".join(used),
                    "n_metrics_used": len(used),
                })
        rows.extend(model_rows)
        composite_domains = {"coordinate", "one_step_closure", "landscape", "anchor_drift", "learned_plane_drift", "convergence", "transition"}
        composite = [r["domain_score"] for r in model_rows if r["domain"] in composite_domains]
        if composite:
            rows.append({
                "split": split,
                "model_label": model,
                "domain": "macrostructure_composite",
                "domain_score": float(np.mean(composite)),
                "metrics_used": "domain_average_without_task",
                "n_metrics_used": len(composite),
            })
    return pd.DataFrame(rows)


def row_for(all_metrics: pd.DataFrame, model: str, split: str) -> Optional[pd.Series]:
    subset = all_metrics[(all_metrics["model_label"].astype(str) == model) & (all_metrics["split"].astype(str) == split)]
    return None if subset.empty else subset.iloc[0]


def value_for(row: Optional[pd.Series], metric: str) -> float:
    return np.nan if row is None else finite_float(row.get(metric))


def contrast_rows(all_df: pd.DataFrame, score_df: pd.DataFrame, splits: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    def add(split: str, contrast: str, model_a: str, model_b: str, metric: str, claim: str, expected_signal: str) -> None:
        ra = row_for(all_df, model_a, split)
        rb = row_for(all_df, model_b, split)
        va = value_for(ra, metric)
        vb = value_for(rb, metric)
        rows.append({
            "split": split,
            "contrast": contrast,
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "model_a_value": va,
            "model_b_value": vb,
            "direction": metric_direction(metric),
            "model_b_relative_to_model_a": relative_to_reference(va, vb, metric),
            "difference_b_minus_a": float(vb - va) if np.isfinite(va) and np.isfinite(vb) else np.nan,
            "claim_if_signal_matches": claim,
            "expected_signal": expected_signal,
        })
    for split in splits:
        # 1. Main vs pure SSL: explicit alignment vs spontaneous predictive representation.
        for m in ["coordinate_corr_M", "coordinate_corr_Psi", "learned_plane_drift_vector_corr", "learned_plane_self_transition_corr", "next_state_occupancy_js"]:
            add(split, "main_vs_pure_ssl", "predictive_state_event_ssl", "pure_event_ssl_probe", m,
                "Tests whether future-predictive SSL already contains the macrostate structure and how much explicit M/Psi alignment adds.",
                "Pure SSL retains part of the structure; main model is equal or stronger, especially for learned-plane dynamics.")
        # 2. Main vs task-only: prediction accuracy is not macrostructure recovery.
        for m in ["coordinate_corr_M", "coordinate_corr_Psi", "learned_plane_drift_vector_corr", "learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr", "task_auc_binary_rows", "task_auc_thresholded_all_rows", "task_bce"]:
            add(split, "main_vs_task_only", "predictive_state_event_ssl", "task_only", m,
                "Tests whether response-task prediction alone implies macrostructure recovery.",
                "Task-only may have good task discrimination but lower coordinate/drift/transition recovery.")
        # 3. Main vs time shuffle: temporal adjacency is required.
        for m in ["coordinate_corr_M", "coordinate_corr_Psi", "anchor_drift_vector_corr", "learned_plane_drift_vector_corr", "learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr", "learned_plane_inward_fraction_to_reference"]:
            add(split, "main_vs_time_shuffle", "predictive_state_event_ssl", "time_shuffle_control", m,
                "Tests whether real submitted-bundle temporal adjacency is required for effective-field recovery.",
                "Coordinate marginals may partly remain, while drift, convergence and transition metrics degrade.")
        # 4. Main vs tag/support randomization: semantic alignment is required.
        for m in ["coordinate_corr_M", "coordinate_corr_Psi", "coordinate_rmse_M", "coordinate_rmse_Psi", "learned_plane_drift_vector_corr", "learned_plane_inward_fraction_to_reference", "learned_plane_transition_mean_row_tv"]:
            add(split, "main_vs_tag_support_randomized", "predictive_state_event_ssl", "tag_support_randomized", m,
                "Tests whether exposure-alignment recovery depends on content/support semantics.",
                "M recovery should be relatively more preserved than Psi/convergence/transition recovery.")
        # 5. Pure SSL vs task-only: future-predictive objective vs discriminative task objective.
        for m in ["coordinate_corr_M", "coordinate_corr_Psi", "learned_plane_drift_vector_corr", "learned_plane_self_transition_corr"]:
            add(split, "pure_ssl_vs_task_only", "pure_event_ssl_probe", "task_only", m,
                "Compares unsupervised future-predictive representation with task-only representation under the same structural evaluator.",
                "Pure SSL should recover more macrostructure if future-state prediction is closer to effective dynamics than correctness prediction.")
        # 6. Time vs tag/support randomization: temporal vs semantic failure signatures.
        for m in ["coordinate_corr_M", "coordinate_corr_Psi", "learned_plane_drift_vector_corr", "learned_plane_transition_mean_row_tv", "learned_plane_inward_fraction_to_reference"]:
            add(split, "time_shuffle_vs_tag_support_randomized", "time_shuffle_control", "tag_support_randomized", m,
                "Disentangles temporal-adjacency dependence from support/content semantic-alignment dependence.",
                "Time-shuffle should more globally disrupt dynamics; tag/support randomization should show stronger Psi-specific damage.")
        # 7. Anchor vs learned-plane within model.
        for model in ["predictive_state_event_ssl", "pure_event_ssl_probe", "task_only", "time_shuffle_control", "tag_support_randomized"]:
            r = row_for(all_df, model, split)
            for m_anchor, m_learned in [
                ("anchor_drift_vector_corr", "learned_plane_drift_vector_corr"),
                ("anchor_occupancy_weighted_local_drift_cosine", "learned_plane_occupancy_weighted_local_drift_cosine"),
                ("anchor_transition_mean_row_tv", "learned_plane_transition_mean_row_tv"),
                ("anchor_self_transition_corr", "learned_plane_self_transition_corr"),
            ]:
                va = value_for(r, m_anchor); vb = value_for(r, m_learned)
                rows.append({
                    "split": split,
                    "contrast": "anchor_vs_learned_plane_within_model",
                    "model_a": f"{model}: empirical-anchor",
                    "model_b": f"{model}: learned-plane",
                    "metric": f"{m_anchor} vs {m_learned}",
                    "model_a_value": va,
                    "model_b_value": vb,
                    "direction": metric_direction(m_learned),
                    "model_b_relative_to_model_a": relative_to_reference(va, vb, m_learned),
                    "difference_b_minus_a": float(vb - va) if np.isfinite(va) and np.isfinite(vb) else np.nan,
                    "claim_if_signal_matches": "Tests whether the model only predicts next state from empirical anchors or forms its own phase plane.",
                    "expected_signal": "Main model should remain strong in learned-plane view; weaker controls may look better only in anchor view.",
                })
        # 8. M/Psi asymmetry within controls.
        for model in ["predictive_state_event_ssl", "pure_event_ssl_probe", "task_only", "time_shuffle_control", "tag_support_randomized"]:
            r = row_for(all_df, model, split)
            m_corr = value_for(r, "coordinate_corr_M")
            psi_corr = value_for(r, "coordinate_corr_Psi")
            rows.append({
                "split": split,
                "contrast": "M_vs_Psi_coordinate_asymmetry",
                "model_a": model,
                "model_b": model,
                "metric": "coordinate_corr_M - coordinate_corr_Psi",
                "model_a_value": m_corr,
                "model_b_value": psi_corr,
                "direction": "diagnostic",
                "model_b_relative_to_model_a": psi_corr / max(abs(m_corr), EPS) if np.isfinite(m_corr) and np.isfinite(psi_corr) else np.nan,
                "difference_b_minus_a": psi_corr - m_corr if np.isfinite(m_corr) and np.isfinite(psi_corr) else np.nan,
                "claim_if_signal_matches": "Tests whether support/content randomization selectively damages exposure-alignment relative to response order.",
                "expected_signal": "Tag/support randomization should show Psi weaker than M; task/time controls may show different asymmetry.",
            })
    # 9. Development-to-confirmation stability.
    for model in all_df["model_label"].dropna().unique():
        rv = row_for(all_df, str(model), "A_val")
        rc = row_for(all_df, str(model), "B_confirm")
        for m in ["coordinate_corr_M", "coordinate_corr_Psi", "next_state_occupancy_js", "learned_plane_drift_vector_corr", "learned_plane_transition_mean_row_tv"]:
            va = value_for(rv, m); vb = value_for(rc, m)
            rows.append({
                "split": "A_val_to_B_confirm",
                "contrast": "development_to_confirmation_stability",
                "model_a": f"{model}: A_val",
                "model_b": f"{model}: B_confirm",
                "metric": m,
                "model_a_value": va,
                "model_b_value": vb,
                "direction": metric_direction(m),
                "model_b_relative_to_model_a": relative_to_reference(va, vb, m),
                "difference_b_minus_a": float(vb - va) if np.isfinite(va) and np.isfinite(vb) else np.nan,
                "claim_if_signal_matches": "Tests whether each model/control generalizes structurally from validation to independent confirmation.",
                "expected_signal": "Main model should be stable; unstable controls indicate split-specific artifacts.",
            })
    return pd.DataFrame(rows)


def validation_confirmation_stability(all_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "coordinate_corr_M", "coordinate_corr_Psi", "coordinate_rmse_M", "coordinate_rmse_Psi",
        "one_step_rmse_M", "one_step_rmse_Psi", "next_state_occupancy_js",
        "anchor_drift_vector_corr", "learned_plane_drift_vector_corr",
        "learned_plane_occupancy_weighted_local_drift_cosine",
        "learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr",
        "learned_plane_inward_fraction_to_reference",
    ]
    rows: List[Dict[str, Any]] = []
    for model in all_metrics["model_label"].dropna().astype(str).unique():
        validation = row_for(all_metrics, model, "A_val")
        confirmation = row_for(all_metrics, model, "B_confirm")
        for metric in metrics:
            value_validation = value_for(validation, metric)
            value_confirmation = value_for(confirmation, metric)
            direction = metric_direction(metric)
            if np.isfinite(value_validation) and np.isfinite(value_confirmation):
                delta = value_confirmation - value_validation
                degradation = value_validation - value_confirmation if direction == "higher" else value_confirmation - value_validation if direction == "lower" else np.nan
            else:
                delta = degradation = np.nan
            rows.append({
                "model_label": model,
                "metric": metric,
                "direction": direction,
                "validation_value": value_validation,
                "confirmation_value": value_confirmation,
                "confirmation_minus_validation": delta,
                "absolute_gap": abs(delta) if np.isfinite(delta) else np.nan,
                "degradation_amount": degradation,
            })
    return pd.DataFrame(rows)


def load_transition_npz(root: Path, prefix: str, split: str) -> Optional[Mapping[str, np.ndarray]]:
    path = root / "tables" / f"{prefix}_{split}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return {name: data[name] for name in data.files}


def collect_transition_tables(roots: Mapping[str, Path], splits: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    spec_map = {str(spec["label"]): spec for spec in MODEL_SPECS}
    for model, root in roots.items():
        prefix = str(spec_map[model]["matrix_prefix"])
        for split in splits:
            matrices = load_transition_npz(root, prefix, split)
            if matrices is None:
                continue
            empirical = matrices.get("P_emp")
            for view, key in (("learned_plane", "P_learned"), ("anchor", "P_anchor")):
                predicted = matrices.get(key)
                if empirical is None or predicted is None or empirical.shape != predicted.shape:
                    continue
                residual = predicted - empirical
                for source_state in range(residual.shape[0]):
                    for target_state in range(residual.shape[1]):
                        rows.append({
                            "model_label": model,
                            "split": split,
                            "view": view,
                            "from_state": source_state,
                            "to_state": target_state,
                            "P_emp": float(empirical[source_state, target_state]),
                            "P_model": float(predicted[source_state, target_state]),
                            "residual_model_minus_emp": float(residual[source_state, target_state]),
                        })
    return pd.DataFrame(rows)


def build_quality_gates(audit_df: pd.DataFrame, all_metrics: pd.DataFrame, splits: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, audit in audit_df.iterrows():
        model = str(audit.get("model_label", ""))
        verified = audit.get("status") == "verified"
        rows.append({"scope": model, "gate": "formal_manifest_and_fixed_k6_contract", "passed": bool(verified), "detail": audit.get("status")})
        subset = all_metrics[all_metrics["model_label"].astype(str) == model]
        rows.append({"scope": model, "gate": "requested_splits_present", "passed": set(splits).issubset(set(subset["split"].astype(str))), "detail": ";".join(sorted(set(subset["split"].astype(str))))})
        for split in splits:
            row = row_for(all_metrics, model, split)
            finite_count = sum(np.isfinite(value_for(row, metric)) for metric in CORE_METRICS if metric in all_metrics.columns)
            rows.append({"scope": f"{model}/{split}", "gate": "core_metrics_finite", "passed": finite_count >= 10, "detail": f"finite_core_metrics={finite_count}"})
    return pd.DataFrame(rows)


def add_ledger(rows: List[Dict[str, Any]], priority: str, category: str, metric: str, value: Any, source: str, interpretation: str, manuscript_use: str, split: str = "", model: str = "") -> None:
    rows.append({
        "priority": priority,
        "category": category,
        "model_label": model,
        "split": split,
        "metric": metric,
        "value": value,
        "source": source,
        "interpretation": interpretation,
        "manuscript_use": manuscript_use,
    })


def publication_metric_ledger(all_metrics: pd.DataFrame, contrasts: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    main = row_for(all_metrics, "predictive_state_event_ssl", "B_confirm")
    required = [
        ("n_rows", "Confirmation intervals"), ("n_users", "Confirmation users"),
        ("coordinate_corr_M", "Coordinate correlation M"), ("coordinate_corr_Psi", "Coordinate correlation Psi"),
        ("coordinate_rmse_M", "Coordinate RMSE M"), ("coordinate_rmse_Psi", "Coordinate RMSE Psi"),
        ("one_step_rmse_M", "One-step RMSE M"), ("one_step_rmse_Psi", "One-step RMSE Psi"),
        ("next_state_occupancy_js", "Next-state occupancy JS"),
        ("anchor_drift_vector_corr", "Empirical-anchor drift correlation"),
        ("learned_plane_drift_vector_corr", "Learned-plane drift correlation"),
        ("learned_plane_occupancy_weighted_local_drift_cosine", "Learned-plane local drift cosine"),
        ("learned_plane_transition_mean_row_tv", "Learned-plane transition mean row TV"),
        ("learned_plane_self_transition_corr", "Learned-plane self-transition correlation"),
        ("learned_plane_inward_fraction_to_reference", "Learned-plane inward-flow fraction"),
    ]
    for key, label in required:
        add_ledger(rows, MAIN_REQUIRED, "main model confirmation", label, value_for(main, key), "predictive-state B_confirm structural metrics", "Frozen confirmation point estimate under the fixed M/Psi and K=6 evaluation contract.", "Report in Results or the Figure 4 caption.", "B_confirm", "predictive_state_event_ssl")

    time_row = row_for(all_metrics, "time_shuffle_control", "B_confirm")
    tag_row = row_for(all_metrics, "tag_support_randomized", "B_confirm")
    for key, label in (
        ("learned_plane_drift_vector_corr", "Time-shuffle learned-plane drift correlation"),
        ("learned_plane_occupancy_weighted_local_drift_cosine", "Time-shuffle learned-plane local cosine"),
    ):
        add_ledger(rows, MAIN_REQUIRED, "temporal control", label, value_for(time_row, key), "time-shuffle B_confirm metrics", "Temporal adjacency control evaluated on ordered confirmation sequences.", "Report with the main-model value.", "B_confirm", "time_shuffle_control")
    main_inward = value_for(main, "learned_plane_inward_fraction_to_reference")
    tag_inward = value_for(tag_row, "learned_plane_inward_fraction_to_reference")
    add_ledger(rows, MAIN_REQUIRED, "semantic control", "Tag/support-randomized inward-flow fraction", tag_inward, "tag/support B_confirm metrics", "Support-semantic randomization effect on inward transport.", "Report with the main-model inward fraction.", "B_confirm", "tag_support_randomized")
    add_ledger(rows, MAIN_RECOMMENDED, "semantic control", "Relative inward-flow change", (tag_inward - main_inward) / abs(main_inward) if np.isfinite(main_inward) and abs(main_inward) > EPS and np.isfinite(tag_inward) else np.nan, "derived from main and tag/support B_confirm metrics", "Signed relative change; negative values indicate weakened inward transport.", "Use only with both raw fractions.", "B_confirm", "tag_support_randomized")

    for model in ("pure_event_ssl_probe", "task_only"):
        row = row_for(all_metrics, model, "B_confirm")
        for key in ("coordinate_corr_M", "coordinate_corr_Psi", "learned_plane_drift_vector_corr", "learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr"):
            add_ledger(rows, MAIN_RECOMMENDED, "objective controls", key, value_for(row, key), f"{model} B_confirm metrics", "Control-model structural recovery under the same frozen evaluation contract.", "Report selectively in the controls paragraph.", "B_confirm", model)
    task_row = row_for(all_metrics, "task_only", "B_confirm")
    for key in ("task_auc_binary_rows", "task_bce"):
        add_ledger(rows, MAIN_RECOMMENDED, "task-only performance", key, value_for(task_row, key), "task-only B_confirm metrics", "Task performance of the task-only control.", "Use when contrasting task discrimination with macrostructure recovery.", "B_confirm", "task_only")

    main_stability = stability[stability["model_label"] == "predictive_state_event_ssl"]
    for metric in ("coordinate_corr_M", "coordinate_corr_Psi", "next_state_occupancy_js", "learned_plane_drift_vector_corr", "learned_plane_transition_mean_row_tv"):
        subset = main_stability[main_stability["metric"] == metric]
        if subset.empty:
            continue
        row = subset.iloc[0]
        add_ledger(rows, MAIN_RECOMMENDED, "validation-confirmation stability", f"absolute gap {metric}", row.get("absolute_gap"), "Stage-4 stability table", "Absolute validation-to-confirmation difference for the frozen main model.", "Report when making a split-stability statement.", "A_val_to_B_confirm", "predictive_state_event_ssl")

    if not contrasts.empty:
        selected = contrasts[
            (contrasts["split"] == "B_confirm")
            & (contrasts["contrast"].isin(["main_vs_time_shuffle", "main_vs_tag_support_randomized"]))
        ]
        for _, row in selected.iterrows():
            add_ledger(rows, SUPPLEMENT_REQUIRED, "control contrasts", f"{row['contrast']}::{row['metric']}", row.get("difference_b_minus_a"), "Stage-4 contrast table", "Control-minus-main raw metric difference.", "Additional information.", "B_confirm", str(row.get("model_b", "")))
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


def markdown_table(df: pd.DataFrame, max_rows: int = 400) -> str:
    if df.empty:
        return "_No rows available._"
    return df.head(max_rows).to_markdown(index=False)


def write_report(path: Path, audits: pd.DataFrame, quality: pd.DataFrame, ledger: pd.DataFrame, all_metrics: pd.DataFrame, scores: pd.DataFrame, contrasts: pd.DataFrame, stability: pd.DataFrame, splits: Sequence[str]) -> None:
    audit_display = audits.copy()
    if not audit_display.empty:
        audit_display["model_kind"] = audit_display["contract"].map(lambda item: item.get("model_kind") if isinstance(item, Mapping) else None)
        audit_display["macrostate_k"] = audit_display["partition"].map(lambda item: item.get("macrostate_k") if isinstance(item, Mapping) else None)
        audit_columns = ["model_label", "model_kind", "metrics_rows", "metrics_columns", "macrostate_k", "status", "root", "scientific_role"]
        audit_display = audit_display[[column for column in audit_columns if column in audit_display.columns]]
    lines: List[str] = [
        "# Stage 4 all-experiment comparison report",
        "",
        "This report is generated from frozen Stage-4 evaluations. It does not train models, recompute predictions, refit the mesostate partition, or redefine the primary `(M, Psi)` state.",
        "",
        "## Numerical rigor and reporting contract",
        "",
        "- Raw metrics are primary; domain scores and relative-retention quantities are descriptive transformations.",
        "- Empirical-anchor and learned-plane drift statistics are reported separately.",
        "- Every model uses the same Stage-1 fixed `K=6` partition; KMeans is not refit and `K` is not reselected.",
        "- A_val and B_confirm values are point estimates from frozen model runs. Random-seed uncertainty requires separate multi-seed runs and is not inferred here.",
        "- Control-minus-main differences are reported with both raw values; no significance claim is generated from a single trained seed.",
        "",
        "## Input audit",
        "",
        markdown_table(audit_display, 50),
        "",
        "## Scientific quality gates",
        "",
        markdown_table(quality, 100),
        "",
        "## Main-text numerical ledger",
        "",
        markdown_table(ledger[ledger["priority"].isin([MAIN_REQUIRED, MAIN_RECOMMENDED])], 200),
        "",
        "## Recommended comparison hierarchy",
        "",
        pd.DataFrame([
            ("Main vs pure SSL", "Tests future-predictive structure without explicit state/closure supervision."),
            ("Main vs task-only", "Tests whether response prediction alone implies macrostructure recovery."),
            ("Main vs time-shuffle", "Tests whether real temporal adjacency is required for learned-plane dynamics."),
            ("Main vs tag/support randomization", "Tests whether support semantics selectively affect inward transport."),
            ("Anchor vs learned plane", "Separates next-state prediction from self-consistent phase-plane recovery."),
            ("A_val vs B_confirm", "Tests split stability under a frozen evaluation contract."),
        ], columns=["comparison", "scientific use"]).to_markdown(index=False),
    ]
    for split in splits:
        subset = all_metrics[all_metrics["split"].astype(str) == split]
        columns = ["model_label"] + [metric for metric in CORE_METRICS if metric in subset.columns]
        lines.extend(["", f"## Key numerical summary: {split}", "", markdown_table(subset[columns], 20)])
        score_subset = scores[(scores["split"].astype(str) == split) & scores["domain"].isin(["coordinate", "landscape", "anchor_drift", "learned_plane_drift", "convergence", "transition", "macrostructure_composite", "task"])]
        if not score_subset.empty:
            pivot = score_subset.pivot_table(index="model_label", columns="domain", values="domain_score", aggfunc="first").reset_index()
            lines.extend(["", f"### Domain scores: {split}", "", markdown_table(pivot, 20)])
    lines.extend([
        "",
        "## Validation-confirmation stability",
        "",
        markdown_table(stability, 200),
        "",
        "## Contrast table",
        "",
        markdown_table(contrasts, 400),
        "",
        "## Supplementary numerical ledger",
        "",
        markdown_table(ledger[ledger["priority"] == SUPPLEMENT_REQUIRED], 300),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract formal Stage-4 Event-SSL and control statistics.")
    parser.add_argument("--main-root", type=Path, default=MODEL_SPECS[0]["default_root"])
    parser.add_argument("--pure-root", type=Path, default=MODEL_SPECS[1]["default_root"])
    parser.add_argument("--task-root", type=Path, default=MODEL_SPECS[2]["default_root"])
    parser.add_argument("--time-shuffle-root", type=Path, default=MODEL_SPECS[3]["default_root"])
    parser.add_argument("--tag-support-root", type=Path, default=MODEL_SPECS[4]["default_root"])
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

    all_metrics, model_audits, sources = collect_all(args)
    long_metrics = make_long_metrics(all_metrics, args.splits)
    relative_metrics = add_relative_to_main(long_metrics)
    scores = domain_scores(all_metrics, args.splits)
    contrasts = contrast_rows(all_metrics, scores, args.splits)
    stability = validation_confirmation_stability(all_metrics)
    quality = build_quality_gates(model_audits, all_metrics, args.splits)
    ledger = publication_metric_ledger(all_metrics, contrasts, stability)
    roots = {
        "predictive_state_event_ssl": Path(args.main_root),
        "pure_event_ssl_probe": Path(args.pure_root),
        "task_only": Path(args.task_root),
        "time_shuffle_control": Path(args.time_shuffle_root),
        "tag_support_randomized": Path(args.tag_support_root),
    }
    transition_residuals = collect_transition_tables(roots, args.splits)
    source_audit = source_audit_table(sources)

    outputs = {
        "wide": write_table(all_metrics, table_root / "stage4_all_experiments_wide_metrics_all_splits"),
        "long": write_table(long_metrics, table_root / "stage4_all_experiments_long_metrics"),
        "relative": write_table(relative_metrics, table_root / "stage4_all_experiments_relative_to_main"),
        "domain": write_table(scores, table_root / "stage4_all_experiments_domain_scores"),
        "contrasts": write_table(contrasts, table_root / "stage4_all_experiments_claim_contrasts"),
        "transitions": write_table(transition_residuals, table_root / "stage4_all_experiments_transition_residuals_long"),
        "stability": write_table(stability, table_root / "stage4_all_experiments_validation_confirmation_stability"),
        "ledger": write_table(ledger, table_root / "stage4_publication_metric_ledger"),
        "quality": write_table(quality, table_root / "stage4_scientific_quality_gates"),
        "input_audit": write_table(source_audit, table_root / "stage4_all_experiments_input_audit"),
    }
    report_path = report_root / "stage4_all_experiments_comparison_report.md"
    write_report(report_path, model_audits, quality, ledger, all_metrics, scores, contrasts, stability, args.splits)

    if not quality.empty and not bool(quality["passed"].all()):
        failed = quality[~quality["passed"]]
        raise RuntimeError(f"Stage-4 quality gates failed:\n{failed.to_string(index=False)}")

    save_json({
        "script": Path(__file__).name,
        "output_root": str(output_root),
        "splits": list(args.splits),
        "primary_coordinates": ["M", "Psi"],
        "fixed_macrostate_k": EXPECTED_K,
        "kmeans_refit": False,
        "macrostate_k_selected": False,
        "model_audits": model_audits.to_dict(orient="records"),
        "generated_tables": {name: str(path.resolve()) for name, path in outputs.items()},
        "report": str(report_path.resolve()),
        "numerical_contract": {
            "raw_metrics_primary": True,
            "domain_scores_descriptive": True,
            "single_seed_point_estimates": True,
            "seed_uncertainty_inferred": False,
        },
    }, metadata_root / "stage4_all_experiments_collection_manifest.json")
    print(f"[Stage4 statistics] wrote {report_path}")


if __name__ == "__main__":
    main()
