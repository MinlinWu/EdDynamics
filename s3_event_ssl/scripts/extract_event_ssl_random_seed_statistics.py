#!/usr/bin/env python3
from __future__ import annotations

"""Collect Event-SSL random-seed statistics for Additional Information."""

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

EPS = 1e-12
EXPECTED_MACROSTATE_K = 6
DEFAULT_SEEDS = (42, 2026, 666, 606, 37, 4669)
DEFAULT_SPLITS = ("A_val", "B_confirm")
DEFAULT_REFERENCE_SEED = 42
DEFAULT_MAIN_OUTPUT_ROOT = Path("/data/datasets/KT4/outputs_KT4")

STAGE4_MODEL_ORDER = (
    "predictive_state_event_ssl",
    "pure_event_ssl_probe",
    "task_only",
    "time_shuffle_control",
    "tag_support_randomized",
)

STAGE4_MODEL_LABELS = {
    "predictive_state_event_ssl": "Predictive-state Event-SSL",
    "pure_event_ssl_probe": "Pure SSL with frozen probe",
    "task_only": "Task-only",
    "time_shuffle_control": "Within-user time shuffle",
    "tag_support_randomized": "Tag/support randomization",
}

STAGE4_SPECS = {
    "predictive_state_event_ssl": {
        "relative_root": "stage4_event_ssl/evaluation_predictive_state",
        "metrics_base": "stage4_event_ssl_structural_metrics_all_splits",
        "matrix_prefix": "stage4_event_ssl_transition_matrices",
        "manifest": "metadata/stage4_event_ssl_evaluation_manifest.json",
        "partition_audit": "metadata/stage4_event_ssl_fixed_k6_partition_audit.json",
        "training_root": "stage4_event_ssl/models/predictive_state",
        "expected_model_kind": "predictive_state",
    },
    "pure_event_ssl_probe": {
        "relative_root": "stage4_event_ssl/evaluation_pure_ssl_probe",
        "metrics_base": "stage4_event_ssl_structural_metrics_all_splits",
        "matrix_prefix": "stage4_event_ssl_transition_matrices",
        "manifest": "metadata/stage4_event_ssl_evaluation_manifest.json",
        "partition_audit": "metadata/stage4_event_ssl_fixed_k6_partition_audit.json",
        "training_root": "stage4_event_ssl/models/pure_ssl",
        "expected_model_kind": "pure_ssl",
    },
    "task_only": {
        "relative_root": "stage4_event_ssl/controls/task_only/evaluation",
        "metrics_base": "stage4_task_only_structural_metrics_all_splits",
        "task_metrics_base": "stage4_task_only_task_metrics_all_splits",
        "matrix_prefix": "stage4_task_only_transition_matrices",
        "manifest": "metadata/stage4_task_only_evaluation_manifest.json",
        "partition_audit": "metadata/stage4_task_only_fixed_k6_partition_audit.json",
        "training_root": "stage4_event_ssl/controls/task_only/model",
        "expected_model_kind": "task_only_control",
    },
    "time_shuffle_control": {
        "relative_root": "stage4_event_ssl_time_shuffle_control/evaluation_on_ordered_inputs",
        "metrics_base": "stage4_event_ssl_structural_metrics_all_splits",
        "matrix_prefix": "stage4_event_ssl_transition_matrices",
        "manifest": "metadata/stage4_event_ssl_evaluation_manifest.json",
        "control_manifest": "metadata/stage4_time_shuffle_control_manifest.json",
        "partition_audit": "metadata/stage4_event_ssl_fixed_k6_partition_audit.json",
        "training_root": "stage4_event_ssl_time_shuffle_control/model",
        "expected_model_kind": "predictive_state",
    },
    "tag_support_randomized": {
        "relative_root": "stage4_event_ssl_tag_support_randomized_control/evaluation",
        "metrics_base": "stage4_event_ssl_structural_metrics_all_splits",
        "matrix_prefix": "stage4_event_ssl_transition_matrices",
        "manifest": "metadata/stage4_event_ssl_evaluation_manifest.json",
        "control_manifest": "metadata/stage4_tag_support_randomization_control_manifest.json",
        "partition_audit": "metadata/stage4_event_ssl_fixed_k6_partition_audit.json",
        "training_root": "stage4_event_ssl_tag_support_randomized_control/model",
        "expected_model_kind": "predictive_state",
    },
}

STAGE4_HIGHER_IS_BETTER = {
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

STAGE4_LOWER_IS_BETTER = {
    "coordinate_rmse_M", "coordinate_rmse_Psi", "coordinate_mae_M", "coordinate_mae_Psi",
    "one_step_rmse_M", "one_step_rmse_Psi", "current_state_occupancy_js",
    "next_state_occupancy_js", "anchor_drift_local_rmse", "learned_plane_drift_local_rmse",
    "anchor_high_support_residual_mean", "anchor_low_support_residual_mean",
    "learned_plane_high_support_residual_mean", "learned_plane_low_support_residual_mean",
    "anchor_transition_mean_row_tv", "anchor_transition_max_row_tv",
    "anchor_self_transition_rmse", "anchor_self_transition_mae",
    "learned_plane_transition_mean_row_tv", "learned_plane_transition_max_row_tv",
    "learned_plane_self_transition_rmse", "learned_plane_self_transition_mae",
    "task_bce", "task_rmse", "task_mae",
}

STAGE4_DOMAIN_METRICS = {
    "coordinate": ("coordinate_corr_M", "coordinate_corr_Psi", "coordinate_rmse_M", "coordinate_rmse_Psi"),
    "one_step_closure": ("one_step_rmse_M", "one_step_rmse_Psi", "one_step_corr_M", "one_step_corr_Psi"),
    "landscape": ("current_state_occupancy_js", "next_state_occupancy_js", "current_state_occupancy_overlap", "next_state_occupancy_overlap"),
    "anchor_drift": ("anchor_drift_vector_corr", "anchor_drift_speed_corr", "anchor_occupancy_weighted_local_drift_cosine", "anchor_drift_local_rmse"),
    "learned_plane_drift": ("learned_plane_drift_vector_corr", "learned_plane_drift_speed_corr", "learned_plane_occupancy_weighted_local_drift_cosine", "learned_plane_drift_local_rmse"),
    "convergence": ("anchor_inward_fraction_to_reference", "learned_plane_inward_fraction_to_reference", "anchor_negative_divergence_weighted_fraction", "learned_plane_negative_divergence_weighted_fraction"),
    "transition": ("learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr", "learned_plane_diagonal_dominance_match_fraction", "learned_plane_top_transition_edge_overlap"),
    "task": ("task_auc_binary_rows", "task_auc_thresholded_all_rows", "task_accuracy_at_0p5_binary_rows", "task_bce"),
}

STAGE4_KEY_METRICS = (
    "coordinate_corr_M", "coordinate_corr_Psi", "coordinate_rmse_M", "coordinate_rmse_Psi",
    "one_step_corr_M", "one_step_corr_Psi", "one_step_rmse_M", "one_step_rmse_Psi",
    "current_state_occupancy_js", "next_state_occupancy_js",
    "anchor_drift_vector_corr", "anchor_drift_speed_corr",
    "anchor_occupancy_weighted_local_drift_cosine", "anchor_inward_fraction_to_reference",
    "learned_plane_drift_vector_corr", "learned_plane_drift_speed_corr",
    "learned_plane_occupancy_weighted_local_drift_cosine",
    "learned_plane_inward_fraction_to_reference",
    "learned_plane_transition_mean_row_tv", "learned_plane_transition_max_row_tv",
    "learned_plane_self_transition_corr", "learned_plane_diagonal_dominance_match_fraction",
    "learned_plane_top_transition_edge_overlap", "task_auc_binary_rows",
    "task_auc_thresholded_all_rows", "task_bce",
)

STAGE5_SCORE_DOMAINS = (
    "coordinate_score", "closure_score", "landscape_score", "drift_score",
    "transition_score", "task_score", "macro_label_score", "macrostructure_composite_descriptive",
)

EXPECTED_MARKERS = (
    "prepare", "predictive", "pure_ssl", "evaluate_pred", "evaluate_pure",
    "time_shuffle", "task_only", "tag_support", "macro_suff", "representation_geo",
)


@dataclass
class SourceRecord:
    seed: int
    component: str
    status: str
    path: str
    sha256: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    note: str = ""


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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def locate_table(base: Path) -> Path:
    for suffix in (".parquet", ".csv.gz", ".csv"):
        path = base.with_suffix(suffix)
        if path.exists():
            return path
    raise FileNotFoundError(f"Required table not found: {base}.[parquet|csv.gz|csv]")


def read_table(base: Path) -> Tuple[pd.DataFrame, Path]:
    path = locate_table(base)
    if path.suffix == ".parquet":
        return pd.read_parquet(path), path
    return pd.read_csv(path, low_memory=False), path


def write_table(df: pd.DataFrame, base: Path) -> Dict[str, str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    outputs = {"csv": str(csv_path.resolve())}
    try:
        parquet_path = base.with_suffix(".parquet")
        df.to_parquet(parquet_path, index=False)
        outputs["parquet"] = str(parquet_path.resolve())
    except Exception:
        pass
    return outputs


def parse_seed_list(text: str) -> Tuple[int, ...]:
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError("At least one seed is required.")
    if len(set(values)) != len(values):
        raise ValueError("Seed list contains duplicates.")
    return tuple(values)


def parse_runtime_file(path: Path) -> Dict[str, str]:
    output: Dict[str, str] = {}
    if not path.exists():
        return output
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            output[key] = value.strip()
    return output


def parse_sha256_file(path: Path) -> Dict[str, str]:
    output: Dict[str, str] = {}
    if not path.exists():
        return output
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2:
            output[fields[1].lstrip("*./")] = fields[0]
    return output


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def metric_direction(metric: str) -> str:
    if metric in STAGE4_HIGHER_IS_BETTER:
        return "higher"
    if metric in STAGE4_LOWER_IS_BETTER:
        return "lower"
    if metric.endswith("_corr") or "cosine" in metric or "overlap" in metric or "fraction" in metric:
        return "higher"
    if any(token in metric for token in ("rmse", "mae", "js", "tv", "loss", "bce")):
        return "lower"
    return "unknown"


def normalized_stage4_metric(value: float, metric: str) -> float:
    if not np.isfinite(value):
        return float("nan")
    direction = metric_direction(metric)
    if direction == "higher":
        return float(np.clip(value, 0.0, 1.0))
    if direction == "lower":
        if "js" in metric or "bce" in metric:
            return float(np.exp(-max(value, 0.0)))
        if "tv" in metric:
            return float(np.clip(1.0 - value, 0.0, 1.0))
        return float(1.0 / (1.0 + max(value, 0.0)))
    return float("nan")


def stage4_domain_scores(row: pd.Series) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for domain, metrics in STAGE4_DOMAIN_METRICS.items():
        values = []
        for metric in metrics:
            score = normalized_stage4_metric(finite_float(row.get(metric)), metric)
            if np.isfinite(score):
                values.append(score)
        output[domain] = float(np.mean(values)) if values else float("nan")
    composite_domains = (
        "coordinate", "one_step_closure", "landscape", "anchor_drift",
        "learned_plane_drift", "convergence", "transition",
    )
    values = [output[name] for name in composite_domains if np.isfinite(output.get(name, np.nan))]
    output["macrostructure_composite"] = float(np.mean(values)) if values else float("nan")
    return output


def stage5_corr_score(value: float) -> float:
    return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0)) if np.isfinite(value) else float("nan")


def stage5_rmse_score(value: float) -> float:
    return float(1.0 / (1.0 + max(value, 0.0) / 0.15)) if np.isfinite(value) else float("nan")


def stage5_one_minus_score(value: float) -> float:
    return float(np.clip(1.0 - value, 0.0, 1.0)) if np.isfinite(value) else float("nan")


def stage5_direct_score(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else float("nan")


def stage5_bce_score(value: float) -> float:
    return float(1.0 / (1.0 + max(value, 0.0))) if np.isfinite(value) else float("nan")


def mean_finite(values: Iterable[float]) -> float:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return float(np.mean(array)) if array.size else float("nan")


def stage5_domain_scores(row: pd.Series) -> Dict[str, float]:
    coordinate = mean_finite([
        stage5_corr_score(finite_float(row.get("coordinate_corr_M"))),
        stage5_corr_score(finite_float(row.get("coordinate_corr_Psi"))),
    ])
    closure = mean_finite([
        stage5_rmse_score(finite_float(row.get("one_step_rmse_M"))),
        stage5_rmse_score(finite_float(row.get("one_step_rmse_Psi"))),
    ])
    landscape = mean_finite([
        stage5_one_minus_score(finite_float(row.get("current_state_occupancy_js"))),
        stage5_one_minus_score(finite_float(row.get("next_state_occupancy_js"))),
    ])
    drift = mean_finite([
        stage5_corr_score(finite_float(row.get("learned_plane_drift_vector_corr"))),
        stage5_corr_score(finite_float(row.get("learned_plane_occupancy_weighted_local_drift_cosine"))),
    ])
    transition = mean_finite([
        stage5_one_minus_score(finite_float(row.get("learned_plane_transition_mean_row_tv"))),
        stage5_corr_score(finite_float(row.get("learned_plane_self_transition_corr"))),
        stage5_direct_score(finite_float(row.get("learned_plane_diagonal_dominance_match_fraction"))),
        stage5_direct_score(finite_float(row.get("learned_plane_top_transition_edge_overlap"))),
    ])
    task_auc = finite_float(row.get("task_auc"))
    task_bce = finite_float(row.get("task_bce"))
    task = stage5_direct_score(task_auc) if np.isfinite(task_auc) else stage5_bce_score(task_bce)
    nmi = stage5_direct_score(finite_float(row.get("representation_nmi_with_empirical_macrostate")))
    ari = stage5_corr_score(finite_float(row.get("representation_ari_with_empirical_macrostate")))
    macro_label = mean_finite([nmi, ari])
    composite = mean_finite([coordinate, closure, drift, transition])
    return {
        "coordinate_score": coordinate,
        "closure_score": closure,
        "landscape_score": landscape,
        "drift_score": drift,
        "transition_score": transition,
        "task_score": task,
        "macro_label_score": macro_label,
        "macrostructure_composite_descriptive": composite,
    }


def ratio_safe(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= EPS:
        return float("nan")
    return float(numerator / denominator)


def summarize_values(values: Sequence[float], ci_level: float) -> Dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    count = int(array.size)
    if count == 0:
        return {
            "n_seeds": 0, "mean": np.nan, "sd": np.nan, "sem": np.nan,
            "median": np.nan, "q25": np.nan, "q75": np.nan,
            "min": np.nan, "max": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
            "positive_fraction": np.nan, "negative_fraction": np.nan,
        }
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if count > 1 else float("nan")
    sem = float(sd / math.sqrt(count)) if count > 1 else float("nan")
    if count > 1 and np.isfinite(sem):
        critical = float(student_t.ppf(0.5 + ci_level / 2.0, df=count - 1))
        ci_lower = mean - critical * sem
        ci_upper = mean + critical * sem
    else:
        ci_lower = float("nan")
        ci_upper = float("nan")
    return {
        "n_seeds": count,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "positive_fraction": float(np.mean(array > 0)),
        "negative_fraction": float(np.mean(array < 0)),
    }


def summarize_long_table(df: pd.DataFrame, group_cols: Sequence[str], ci_level: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for keys, group in df.groupby(list(group_cols), dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        payload = {column: value for column, value in zip(group_cols, keys)}
        payload.update(summarize_values(pd.to_numeric(group["value"], errors="coerce").to_numpy(dtype=float), ci_level))
        rows.append(payload)
    return pd.DataFrame(rows)


def fmt_number(value: Any, digits: int = 4) -> str:
    number = finite_float(value)
    if not np.isfinite(number):
        return "NA"
    magnitude = abs(number)
    if magnitude != 0 and (magnitude < 1e-3 or magnitude >= 1e4):
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def fmt_mean_sd(row: pd.Series) -> str:
    mean = finite_float(row.get("mean"))
    sd = finite_float(row.get("sd"))
    minimum = finite_float(row.get("min"))
    maximum = finite_float(row.get("max"))
    if not np.isfinite(mean):
        return "NA"
    if np.isfinite(sd):
        return f"{fmt_number(mean)} ± {fmt_number(sd)} [{fmt_number(minimum)}, {fmt_number(maximum)}]"
    return fmt_number(mean)


def markdown_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "_No data available._"
    shown = df.head(max_rows).copy()
    columns = [str(column) for column in shown.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in shown.iterrows():
        values = []
        for column in shown.columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                text = fmt_number(value)
            else:
                text = str(value).replace("|", "\\|").replace("\n", " ")
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Only the first {max_rows} of {len(df)} rows are shown._")
    return "\n".join(lines)


def resolve_experiment_root(
    seed_root: Path,
    main_output_root: Path,
    seed: int,
    reference_seed: int,
) -> Tuple[Path, str]:
    if int(seed) == int(reference_seed):
        return main_output_root.resolve(), "main_experiment_reference"
    return (seed_root / f"seed_{seed}").resolve(), "random_seed_directory"


def audit_seed_directory(
    experiment_root: Path,
    seed: int,
    strict: bool,
    run_source: str,
) -> Tuple[Dict[str, Any], List[SourceRecord]]:
    records: List[SourceRecord] = []
    experiment_root = experiment_root.resolve()
    if not experiment_root.exists():
        raise FileNotFoundError(f"Experiment root not found: {experiment_root}")

    workflow_managed = run_source == "random_seed_directory"
    runtime_path = experiment_root / "environment" / "runtime.txt"
    runtime = parse_runtime_file(runtime_path) if runtime_path.exists() else {}
    records.append(SourceRecord(
        seed,
        "runtime",
        "ok" if runtime_path.exists() else "not_applicable",
        str(runtime_path),
        file_sha256(runtime_path) if runtime_path.exists() else None,
        note="workflow runtime metadata" if workflow_managed else "main experiment output",
    ))

    if workflow_managed:
        marker_status = {name: (experiment_root / "markers" / f"{name}.done").exists() for name in EXPECTED_MARKERS}
        if strict and not all(marker_status.values()):
            missing = [name for name, present in marker_status.items() if not present]
            raise RuntimeError(f"Seed {seed} is missing completion markers: {missing}")
        completion_verified = bool(all(marker_status.values()))
        completion_evidence = "workflow marker files"
        code_sums_path = experiment_root / "environment" / "code_SHA256SUMS.txt"
        code_sums = parse_sha256_file(code_sums_path)
        records.append(SourceRecord(
            seed,
            "code_snapshot_hashes",
            "ok" if code_sums else "missing",
            str(code_sums_path),
            file_sha256(code_sums_path) if code_sums_path.exists() else None,
            note=f"{len(code_sums)} files",
        ))
    else:
        marker_status = {}
        completion_verified = True
        completion_evidence = "formal main-experiment manifests and result tables"
        code_sums = {}
        records.append(SourceRecord(
            seed,
            "code_snapshot_hashes",
            "not_applicable",
            str(experiment_root),
            note="main experiment is audited through formal manifests and source hashes",
        ))

    input_manifest_path = experiment_root / "stage4_event_ssl" / "prepared_inputs" / "metadata" / "stage4_input_manifest.json"
    input_manifest = load_json(input_manifest_path)
    records.append(SourceRecord(seed, "main_input_manifest", "ok", str(input_manifest_path.resolve()), file_sha256(input_manifest_path)))
    if input_manifest.get("primary_coordinates") != ["M", "Psi"]:
        raise RuntimeError(f"Seed {seed} prepared inputs do not use M and Psi.")
    fixed_k = dict(input_manifest.get("stage1_fixed_k6_contract", {}))
    if fixed_k.get("verified") is not True or int(fixed_k.get("macrostate_k", -1)) != EXPECTED_MACROSTATE_K:
        raise RuntimeError(f"Seed {seed} prepared inputs do not carry the verified fixed K=6 contract.")
    schema_payload = {
        "targets": input_manifest.get("targets"),
        "numeric_input_source_columns": input_manifest.get("numeric_input_source_columns"),
        "numeric_feature_names_after_expansion": input_manifest.get("numeric_feature_names_after_expansion"),
        "categorical_input_source_columns": input_manifest.get("categorical_input_source_columns"),
        "categorical_hash_buckets": input_manifest.get("categorical_hash_buckets"),
        "normalization_fit_scope": input_manifest.get("normalization_fit_scope"),
        "sequence_boundary_policy": input_manifest.get("sequence_boundary_policy"),
        "split_shapes": {
            split: {
                "rows": payload.get("rows"),
                "numeric_shape": payload.get("numeric_shape"),
                "categorical_shape": payload.get("categorical_shape"),
            }
            for split, payload in dict(input_manifest.get("split_summaries", {})).items()
        },
        "stage1_metadata_sha256": fixed_k.get("metadata_sha256"),
        "stage1_centers_sha256": fixed_k.get("centers_sha256"),
    }
    input_schema_signature = hashlib.sha256(json.dumps(schema_payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    normalizer_path = experiment_root / "stage4_event_ssl" / "prepared_inputs" / "metadata" / "normalizer.json"
    if normalizer_path.exists():
        records.append(SourceRecord(seed, "main_input_normalizer", "ok", str(normalizer_path.resolve()), file_sha256(normalizer_path)))

    output = {
        "seed": seed,
        "run_source": run_source,
        "experiment_root": str(experiment_root),
        "runtime_seed": int(runtime.get("seed", seed)),
        "model_seed": int(runtime.get("model_seed", seed)),
        "control_seed": int(runtime.get("control_seed", seed)),
        "probe_seed": int(runtime.get("probe_seed", seed)),
        "eval_seed": int(runtime.get("eval_seed", seed)),
        "prep_seed": int(runtime.get("prep_seed", seed)),
        "completion_verified": completion_verified,
        "completion_evidence": completion_evidence,
        "all_markers_complete": bool(all(marker_status.values())) if marker_status else None,
        "missing_markers": ";".join(name for name, present in marker_status.items() if not present),
        "code_snapshot_signature": hashlib.sha256(json.dumps(code_sums, sort_keys=True).encode("utf-8")).hexdigest() if code_sums else "",
        "input_schema_signature": input_schema_signature,
        "stage1_partition_metadata_sha256": str(fixed_k.get("metadata_sha256", "")),
        "stage1_partition_centers_sha256": str(fixed_k.get("centers_sha256", "")),
    }
    return output, records

def validate_fixed_k_metrics(metrics: pd.DataFrame, model: str, seed: int, splits: Sequence[str]) -> None:
    required = {
        "split", "macrostate_partition_verified_against_stage1_fixed_k6",
        "macrostate_k_fixed_a_priori", "macrostate_k",
    }
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise RuntimeError(f"Seed {seed} {model} metrics are missing fixed-K fields: {missing}")
    selected = metrics[metrics["split"].astype(str).isin(splits)]
    observed = set(selected["split"].astype(str))
    absent = sorted(set(splits).difference(observed))
    if absent:
        raise RuntimeError(f"Seed {seed} {model} is missing splits: {absent}")
    verified = pd.to_numeric(selected["macrostate_partition_verified_against_stage1_fixed_k6"], errors="coerce")
    fixed = pd.to_numeric(selected["macrostate_k_fixed_a_priori"], errors="coerce")
    macro_k = pd.to_numeric(selected["macrostate_k"], errors="coerce")
    if not bool((verified == 1.0).all() and (fixed == 1.0).all() and (macro_k == EXPECTED_MACROSTATE_K).all()):
        raise RuntimeError(f"Seed {seed} {model} changed the fixed K=6 evaluation contract.")


def validate_partition_audit(audit: Mapping[str, Any], model: str, seed: int) -> None:
    if audit.get("verified") is not True:
        raise RuntimeError(f"Seed {seed} {model} partition audit is not verified.")
    if int(audit.get("macrostate_k", -1)) != EXPECTED_MACROSTATE_K:
        raise RuntimeError(f"Seed {seed} {model} partition is not K=6.")
    if audit.get("kmeans_refit") is not False or audit.get("macrostate_k_selected") is not False:
        raise RuntimeError(f"Seed {seed} {model} refit KMeans or selected K.")


def collect_stage4_for_seed(
    experiment_root: Path,
    seed: int,
    splits: Sequence[str],
    sources: List[SourceRecord],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    experiment_root = experiment_root.resolve()
    metric_frames: List[pd.DataFrame] = []
    domain_rows: List[Dict[str, Any]] = []
    transition_rows: List[Dict[str, Any]] = []
    training_rows: List[Dict[str, Any]] = []
    partition_hashes: List[Tuple[str, str]] = []

    for model in STAGE4_MODEL_ORDER:
        spec = STAGE4_SPECS[model]
        eval_root = experiment_root / str(spec["relative_root"])
        metrics, metrics_path = read_table(eval_root / "tables" / str(spec["metrics_base"]))
        sources.append(SourceRecord(seed, f"{model}_metrics", "ok", str(metrics_path.resolve()), file_sha256(metrics_path), len(metrics), len(metrics.columns)))
        task_base = spec.get("task_metrics_base")
        if task_base:
            task_metrics, task_path = read_table(eval_root / "tables" / str(task_base))
            sources.append(SourceRecord(seed, f"{model}_task_metrics", "ok", str(task_path.resolve()), file_sha256(task_path), len(task_metrics), len(task_metrics.columns)))
            if "split" in task_metrics.columns:
                extra = [column for column in task_metrics.columns if column != "split" and column not in metrics.columns]
                metrics = metrics.merge(task_metrics[["split"] + extra], on="split", how="left")
        validate_fixed_k_metrics(metrics, model, seed, splits)

        manifest_path = eval_root / str(spec["manifest"])
        manifest = load_json(manifest_path)
        sources.append(SourceRecord(seed, f"{model}_manifest", "ok", str(manifest_path.resolve()), file_sha256(manifest_path)))
        if manifest.get("primary_coordinates") != ["M", "Psi"]:
            raise RuntimeError(f"Seed {seed} {model} does not use primary coordinates M and Psi.")
        expected_kind = str(spec.get("expected_model_kind", ""))
        manifest_kind = str(manifest.get("model_kind", manifest.get("control_name", "")))
        if model != "task_only" and expected_kind and manifest_kind and manifest_kind != expected_kind:
            raise RuntimeError(f"Seed {seed} {model} model_kind={manifest_kind!r}, expected {expected_kind!r}.")
        guardrails = dict(manifest.get("guardrails", {}))
        if guardrails:
            if guardrails.get("kmeans_refit") is not False or guardrails.get("macrostate_k_selected") is not False:
                raise RuntimeError(f"Seed {seed} {model} evaluation changed the KMeans contract.")
            if guardrails.get("B_confirm_used_for_update") is not False:
                raise RuntimeError(f"Seed {seed} {model} used B_confirm for update.")

        audit_path = eval_root / str(spec["partition_audit"])
        audit = load_json(audit_path)
        validate_partition_audit(audit, model, seed)
        sources.append(SourceRecord(seed, f"{model}_partition_audit", "ok", str(audit_path.resolve()), file_sha256(audit_path)))
        partition_hashes.append((str(audit.get("metadata_sha256", "")), str(audit.get("centers_sha256", ""))))

        control_manifest_rel = spec.get("control_manifest")
        if control_manifest_rel:
            control_path = eval_root / str(control_manifest_rel)
            control_manifest = load_json(control_path)
            sources.append(SourceRecord(seed, f"{model}_control_manifest", "ok", str(control_path.resolve()), file_sha256(control_path)))
            control_guardrails = dict(control_manifest.get("guardrails", {}))
            if control_guardrails:
                if control_guardrails.get("kmeans_refit") is not False or control_guardrails.get("macrostate_k_selected") is not False:
                    raise RuntimeError(f"Seed {seed} {model} control changed the KMeans contract.")

        selected = metrics[metrics["split"].astype(str).isin(splits)].copy()
        selected.insert(0, "seed", seed)
        selected.insert(1, "model_label", model)
        selected.insert(2, "model_display", STAGE4_MODEL_LABELS[model])
        metric_frames.append(selected)
        for _, row in selected.iterrows():
            scores = stage4_domain_scores(row)
            for domain, value in scores.items():
                domain_rows.append({
                    "seed": seed,
                    "split": str(row["split"]),
                    "model_label": model,
                    "model_display": STAGE4_MODEL_LABELS[model],
                    "domain": domain,
                    "value": value,
                })

        training_root = experiment_root / str(spec["training_root"])
        history_path = training_root / "training_history.json"
        training_manifest_path = training_root / "training_manifest.json"
        if history_path.exists():
            history = load_json(history_path)
            rows = list(history.get("history", []))
            if rows:
                if model == "task_only":
                    values = [finite_float(entry.get("val", {}).get("task_bce")) for entry in rows]
                    selection_name = "val_task_bce"
                else:
                    values = [finite_float(entry.get("selection_metric")) for entry in rows]
                    selection_name = "selection_metric"
                finite_indices = [index for index, value in enumerate(values) if np.isfinite(value)]
                best_index = min(finite_indices, key=lambda index: values[index]) if finite_indices else None
                training_rows.append({
                    "seed": seed,
                    "model_label": model,
                    "epochs_recorded": len(rows),
                    "best_epoch": int(rows[best_index].get("epoch", best_index + 1)) if best_index is not None else np.nan,
                    "selection_metric_name": selection_name,
                    "best_selection_metric": values[best_index] if best_index is not None else np.nan,
                    "final_selection_metric": values[-1] if values else np.nan,
                })
            sources.append(SourceRecord(seed, f"{model}_training_history", "ok", str(history_path.resolve()), file_sha256(history_path)))
        if training_manifest_path.exists():
            training_manifest = load_json(training_manifest_path)
            configured_seed = training_manifest.get("seed", dict(training_manifest.get("config", {})).get("seed"))
            if configured_seed is not None and int(configured_seed) != int(seed):
                raise RuntimeError(f"Seed {seed} {model} training manifest reports seed={configured_seed}.")
            sources.append(SourceRecord(seed, f"{model}_training_manifest", "ok", str(training_manifest_path.resolve()), file_sha256(training_manifest_path)))

        for split in splits:
            matrix_path = eval_root / "tables" / f"{spec['matrix_prefix']}_{split}.npz"
            if not matrix_path.exists():
                raise FileNotFoundError(f"Transition matrix archive not found: {matrix_path}")
            data = np.load(matrix_path, allow_pickle=False)
            required_keys = {"P_emp", "P_learned", "C_emp", "C_learned"}
            missing_keys = sorted(required_keys.difference(data.files))
            if missing_keys:
                raise RuntimeError(f"Seed {seed} {model} {split} matrix archive is missing: {missing_keys}")
            p_emp = np.asarray(data["P_emp"], dtype=float)
            p_learned = np.asarray(data["P_learned"], dtype=float)
            if p_emp.shape != (EXPECTED_MACROSTATE_K, EXPECTED_MACROSTATE_K) or p_learned.shape != p_emp.shape:
                raise RuntimeError(f"Seed {seed} {model} {split} transition matrix shape is invalid.")
            for state in range(EXPECTED_MACROSTATE_K):
                row_tv = 0.5 * float(np.sum(np.abs(p_learned[state] - p_emp[state])))
                transition_rows.append({
                    "seed": seed,
                    "split": split,
                    "model_label": model,
                    "state": state,
                    "empirical_self_transition": float(p_emp[state, state]),
                    "model_self_transition": float(p_learned[state, state]),
                    "self_transition_difference": float(p_learned[state, state] - p_emp[state, state]),
                    "row_total_variation": row_tv,
                    "empirical_dominant_next_state": int(np.argmax(p_emp[state])),
                    "model_dominant_next_state": int(np.argmax(p_learned[state])),
                    "dominant_next_state_match": float(np.argmax(p_emp[state]) == np.argmax(p_learned[state])),
                    "model_diagonal_dominant": float(int(np.argmax(p_learned[state])) == state),
                })
                for next_state in range(EXPECTED_MACROSTATE_K):
                    transition_rows.append({
                        "seed": seed,
                        "split": split,
                        "model_label": model,
                        "state": state,
                        "next_state": next_state,
                        "transition_cell_empirical": float(p_emp[state, next_state]),
                        "transition_cell_model": float(p_learned[state, next_state]),
                        "transition_cell_residual": float(p_learned[state, next_state] - p_emp[state, next_state]),
                        "record_type": "cell",
                    })
            sources.append(SourceRecord(seed, f"{model}_{split}_matrices", "ok", str(matrix_path.resolve()), file_sha256(matrix_path)))

    if len(set(partition_hashes)) != 1:
        raise RuntimeError(f"Seed {seed} Stage-4 experiments do not share one frozen K=6 partition.")
    return (
        pd.concat(metric_frames, ignore_index=True, sort=False),
        pd.DataFrame(domain_rows),
        pd.DataFrame(transition_rows),
        training_rows,
    )


def stage4_metrics_to_long(metrics: pd.DataFrame) -> pd.DataFrame:
    id_columns = {"seed", "model_label", "model_display", "split"}
    rows: List[Dict[str, Any]] = []
    for _, row in metrics.iterrows():
        for column in metrics.columns:
            if column in id_columns:
                continue
            value = finite_float(row.get(column))
            if np.isfinite(value):
                rows.append({
                    "seed": int(row["seed"]),
                    "model_label": str(row["model_label"]),
                    "model_display": str(row["model_display"]),
                    "split": str(row["split"]),
                    "metric": column,
                    "value": value,
                    "direction": metric_direction(column),
                })
    return pd.DataFrame(rows)


def build_stage4_paired_contrasts(metrics: pd.DataFrame) -> pd.DataFrame:
    controls = [model for model in STAGE4_MODEL_ORDER if model != "predictive_state_event_ssl"]
    rows: List[Dict[str, Any]] = []
    numeric_columns = [
        column for column in metrics.columns
        if column not in {"seed", "model_label", "model_display", "split"}
        and pd.api.types.is_numeric_dtype(metrics[column])
    ]
    for seed in sorted(metrics["seed"].unique()):
        for split in sorted(metrics["split"].astype(str).unique()):
            main = metrics[(metrics["seed"] == seed) & (metrics["split"].astype(str) == split) & (metrics["model_label"] == "predictive_state_event_ssl")]
            if main.empty:
                continue
            main_row = main.iloc[0]
            for control in controls:
                subset = metrics[(metrics["seed"] == seed) & (metrics["split"].astype(str) == split) & (metrics["model_label"] == control)]
                if subset.empty:
                    continue
                control_row = subset.iloc[0]
                for metric in numeric_columns:
                    main_value = finite_float(main_row.get(metric))
                    control_value = finite_float(control_row.get(metric))
                    if not np.isfinite(main_value) or not np.isfinite(control_value):
                        continue
                    direction = metric_direction(metric)
                    raw_difference = main_value - control_value
                    if direction == "higher":
                        improvement = raw_difference
                    elif direction == "lower":
                        improvement = control_value - main_value
                    else:
                        improvement = float("nan")
                    rows.append({
                        "seed": int(seed),
                        "split": split,
                        "control_model": control,
                        "control_display": STAGE4_MODEL_LABELS[control],
                        "metric": metric,
                        "direction": direction,
                        "main_value": main_value,
                        "control_value": control_value,
                        "raw_main_minus_control": raw_difference,
                        "main_improvement_in_expected_direction": improvement,
                    })
    return pd.DataFrame(rows)


def build_stage4_claim_metrics(metrics: pd.DataFrame, domains: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    domain_lookup = {
        (int(row.seed), str(row.split), str(row.model_label), str(row.domain)): float(row.value)
        for row in domains.itertuples(index=False)
        if np.isfinite(float(row.value))
    }
    for seed in sorted(metrics["seed"].unique()):
        for split in sorted(metrics["split"].astype(str).unique()):
            lookup: Dict[Tuple[str, str], float] = {}
            subset = metrics[(metrics["seed"] == seed) & (metrics["split"].astype(str) == split)]
            for _, row in subset.iterrows():
                model = str(row["model_label"])
                for metric in STAGE4_KEY_METRICS:
                    value = finite_float(row.get(metric))
                    if np.isfinite(value):
                        lookup[(model, metric)] = value
            main_drift = lookup.get(("predictive_state_event_ssl", "learned_plane_drift_vector_corr"), np.nan)
            time_drift = lookup.get(("time_shuffle_control", "learned_plane_drift_vector_corr"), np.nan)
            time_cosine = lookup.get(("time_shuffle_control", "learned_plane_occupancy_weighted_local_drift_cosine"), np.nan)
            main_inward = lookup.get(("predictive_state_event_ssl", "learned_plane_inward_fraction_to_reference"), np.nan)
            tag_inward = lookup.get(("tag_support_randomized", "learned_plane_inward_fraction_to_reference"), np.nan)
            tag_reduction = (main_inward - tag_inward) / main_inward if np.isfinite(main_inward) and np.isfinite(tag_inward) and abs(main_inward) > EPS else np.nan
            main_composite = domain_lookup.get((int(seed), split, "predictive_state_event_ssl", "macrostructure_composite"), np.nan)
            control_composites = [
                domain_lookup.get((int(seed), split, control, "macrostructure_composite"), np.nan)
                for control in STAGE4_MODEL_ORDER if control != "predictive_state_event_ssl"
            ]
            finite_controls = [value for value in control_composites if np.isfinite(value)]
            rows.extend([
                {
                    "seed": int(seed), "split": split,
                    "claim": "predictive_state_learned_plane_flow_positive",
                    "value": main_drift,
                    "condition_satisfied": float(np.isfinite(main_drift) and main_drift > 0),
                },
                {
                    "seed": int(seed), "split": split,
                    "claim": "time_shuffle_reverses_learned_plane_field",
                    "value": time_drift,
                    "condition_satisfied": float(np.isfinite(time_drift) and time_drift < 0),
                },
                {
                    "seed": int(seed), "split": split,
                    "claim": "time_shuffle_local_direction_nonpositive",
                    "value": time_cosine,
                    "condition_satisfied": float(np.isfinite(time_cosine) and time_cosine <= 0),
                },
                {
                    "seed": int(seed), "split": split,
                    "claim": "tag_support_randomization_reduces_inward_transport",
                    "value": tag_reduction,
                    "condition_satisfied": float(np.isfinite(tag_reduction) and tag_reduction > 0),
                },
                {
                    "seed": int(seed), "split": split,
                    "claim": "main_macrostructure_exceeds_all_controls",
                    "value": main_composite - max(finite_controls) if np.isfinite(main_composite) and finite_controls else np.nan,
                    "condition_satisfied": float(np.isfinite(main_composite) and finite_controls and main_composite > max(finite_controls)),
                },
            ])
    return pd.DataFrame(rows)


def flatten_numeric_mapping(
    obj: Any,
    prefix: str = "",
) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_numeric_mapping(value, name))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            name = f"{prefix}[{index}]"
            rows.extend(flatten_numeric_mapping(value, name))
    elif isinstance(obj, (bool, np.bool_)):
        rows.append((prefix, float(bool(obj))))
    else:
        value = finite_float(obj)
        if np.isfinite(value):
            rows.append((prefix, value))
    return rows


def collect_control_randomization_audit(
    experiment_root: Path,
    seed: int,
    sources: List[SourceRecord],
    strict: bool,
    run_source: str,
) -> pd.DataFrame:
    experiment_root = experiment_root.resolve()
    specs = {
        "time_shuffle": experiment_root / "stage4_event_ssl_time_shuffle_control" / "prepared_inputs" / "metadata" / "stage4_input_manifest.json",
        "tag_support_randomization": experiment_root / "stage4_event_ssl_tag_support_randomized_control" / "prepared_inputs" / "metadata" / "stage4_input_manifest.json",
    }
    rows: List[Dict[str, Any]] = []
    for control, path in specs.items():
        if not path.exists():
            sources.append(SourceRecord(seed, f"{control}_input_manifest", "missing", str(path), note="optional for the main experiment reference; required for workflow-managed additional runs"))
            if strict and run_source == "random_seed_directory":
                raise FileNotFoundError(f"Control input manifest not found: {path}")
            continue
        payload = load_json(path)
        sources.append(SourceRecord(seed, f"{control}_input_manifest", "ok", str(path.resolve()), file_sha256(path)))
        section_name = "shuffle_summaries" if control == "time_shuffle" else "randomization_summaries"
        summaries = dict(payload.get(section_name, {}))
        for split, summary in summaries.items():
            randomized_columns = summary.get("randomized_columns", []) if isinstance(summary, Mapping) else []
            rows.append({
                "seed": seed,
                "control": control,
                "split": str(split),
                "metric": "randomized_column_count",
                "value": float(len(randomized_columns)),
            })
            for metric, value in flatten_numeric_mapping(summary):
                rows.append({
                    "seed": seed,
                    "control": control,
                    "split": str(split),
                    "metric": metric,
                    "value": value,
                })
    return pd.DataFrame(rows)


def collect_stage5_for_seed(
    experiment_root: Path,
    seed: int,
    splits: Sequence[str],
    sources: List[SourceRecord],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    experiment_root = experiment_root.resolve()
    macro_root = experiment_root / "stage5_macro_sufficiency"
    macro_eval = macro_root / "evaluation"
    geometry_root = experiment_root / "stage5_representation_geometry"
    geometry_eval = geometry_root / "evaluation"

    macro_metrics, macro_path = read_table(macro_eval / "tables" / "stage5_macro_sufficiency_metrics_all_splits")
    macro_train_metrics, macro_train_path = read_table(macro_root / "tables" / "stage5_macro_sufficiency_training_probe_metrics")
    geometry_metrics, geometry_path = read_table(geometry_eval / "tables" / "stage5_representation_geometry_metrics_all_splits")
    pc_metrics, pc_path = read_table(geometry_eval / "tables" / "stage5_representation_geometry_pc_macro_correlations")
    gain_metrics, gain_path = read_table(geometry_eval / "tables" / "stage5_representation_geometry_nonlinear_probe_gain")
    train_pc_path = geometry_root / "tables" / "stage5_pc_macro_correlations_train.csv"
    train_pc = pd.read_csv(train_pc_path, low_memory=False)

    for component, path, frame in (
        ("stage5_macro_metrics", macro_path, macro_metrics),
        ("stage5_macro_training_metrics", macro_train_path, macro_train_metrics),
        ("stage5_geometry_metrics", geometry_path, geometry_metrics),
        ("stage5_geometry_pc_metrics", pc_path, pc_metrics),
        ("stage5_geometry_nonlinear_gain", gain_path, gain_metrics),
        ("stage5_geometry_train_pc_metrics", train_pc_path, train_pc),
    ):
        sources.append(SourceRecord(seed, component, "ok", str(path.resolve()), file_sha256(path), len(frame), len(frame.columns)))

    macro_manifest_path = macro_eval / "metadata" / "stage5_macro_sufficiency_evaluation_manifest.json"
    geometry_manifest_path = geometry_eval / "metadata" / "stage5_representation_geometry_evaluation_manifest.json"
    macro_training_manifest_path = macro_root / "metadata" / "stage5_macro_sufficiency_training_manifest.json"
    macro_partition_path = macro_eval / "metadata" / "stage5_macro_sufficiency_fixed_k6_partition_audit.json"
    geometry_partition_path = geometry_eval / "metadata" / "stage5_representation_geometry_fixed_k6_partition_audit.json"
    for component, path in (
        ("stage5_macro_manifest", macro_manifest_path),
        ("stage5_geometry_manifest", geometry_manifest_path),
        ("stage5_macro_training_manifest", macro_training_manifest_path),
        ("stage5_macro_partition", macro_partition_path),
        ("stage5_geometry_partition", geometry_partition_path),
    ):
        payload = load_json(path)
        sources.append(SourceRecord(seed, component, "ok", str(path.resolve()), file_sha256(path)))
        if payload.get("primary_coordinates") not in (None, ["M", "Psi"]):
            raise RuntimeError(f"Seed {seed} {component} does not use M and Psi.")
        if "partition" in component:
            validate_partition_audit(payload, component, seed)

    macro_metrics = macro_metrics[macro_metrics["split"].astype(str).isin(splits)].copy()
    geometry_metrics = geometry_metrics[geometry_metrics["split"].astype(str).isin(splits)].copy()
    observed_macro = set(macro_metrics["split"].astype(str))
    observed_geometry = set(geometry_metrics["split"].astype(str))
    if set(splits).difference(observed_macro) or set(splits).difference(observed_geometry):
        raise RuntimeError(f"Seed {seed} Stage-5 outputs omit requested splits.")
    macro_metrics.insert(0, "seed", seed)
    macro_train_metrics.insert(0, "seed", seed)
    geometry_metrics.insert(0, "seed", seed)
    pc_metrics.insert(0, "seed", seed)
    gain_metrics.insert(0, "seed", seed)
    train_pc.insert(0, "seed", seed)

    training_manifest_path = geometry_root / "metadata" / "stage5_representation_geometry_training_manifest.json"
    training_manifest = load_json(training_manifest_path)
    sources.append(SourceRecord(seed, "stage5_geometry_training_manifest", "ok", str(training_manifest_path.resolve()), file_sha256(training_manifest_path)))
    return macro_metrics, macro_train_metrics, geometry_metrics, pc_metrics, gain_metrics, train_pc, training_manifest


def build_stage5_domain_rows(macro_metrics: pd.DataFrame, geometry_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for experiment, frame in (("macro_sufficiency", macro_metrics), ("representation_geometry", geometry_metrics)):
        for _, row in frame.iterrows():
            scores = stage5_domain_scores(row)
            for domain, value in scores.items():
                rows.append({
                    "seed": int(row["seed"]),
                    "experiment": experiment,
                    "split": str(row["split"]),
                    "representation": str(row.get("representation", "")),
                    "domain": domain,
                    "value": value,
                })
    return pd.DataFrame(rows)


def score_lookup(domains: pd.DataFrame, seed: int, experiment: str, split: str, representation: str, domain: str) -> float:
    subset = domains[
        (domains["seed"] == seed)
        & (domains["experiment"] == experiment)
        & (domains["split"].astype(str) == split)
        & (domains["representation"].astype(str) == representation)
        & (domains["domain"] == domain)
    ]
    return finite_float(subset.iloc[0]["value"]) if not subset.empty else float("nan")


def build_stage5_retention_tables(domains: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    macro_rows: List[Dict[str, Any]] = []
    geometry_rows: List[Dict[str, Any]] = []
    seeds = sorted(domains["seed"].unique())
    splits = sorted(domains["split"].astype(str).unique())
    for seed in seeds:
        for split in splits:
            for domain in STAGE5_SCORE_DOMAINS:
                full = score_lookup(domains, seed, "macro_sufficiency", split, "full_hidden", domain)
                macro = score_lookup(domains, seed, "macro_sufficiency", split, "macro_only", domain)
                residual = score_lookup(domains, seed, "macro_sufficiency", split, "residual_hidden", domain)
                macro_rows.append({
                    "seed": int(seed), "split": split, "domain": domain,
                    "full_hidden_score": full,
                    "macro_only_score": macro,
                    "residual_hidden_score": residual,
                    "macro_retention_vs_full": ratio_safe(macro, full),
                    "residual_retention_vs_full": ratio_safe(residual, full),
                    "residual_retention_vs_macro": ratio_safe(residual, macro),
                    "macro_minus_residual": macro - residual if np.isfinite(macro) and np.isfinite(residual) else np.nan,
                })
                model = score_lookup(domains, seed, "representation_geometry", split, "model_readout", domain)
                linear = score_lookup(domains, seed, "representation_geometry", split, "linear_hidden", domain)
                geo_residual = score_lookup(domains, seed, "representation_geometry", split, "residual_hidden", domain)
                nonlinear = score_lookup(domains, seed, "representation_geometry", split, "nonlinear_hidden", domain)
                geometry_rows.append({
                    "seed": int(seed), "split": split, "domain": domain,
                    "model_readout_score": model,
                    "linear_hidden_score": linear,
                    "residual_hidden_score": geo_residual,
                    "nonlinear_hidden_score": nonlinear,
                    "linear_retention_vs_model": ratio_safe(linear, model),
                    "residual_retention_vs_model": ratio_safe(geo_residual, model),
                    "residual_retention_vs_linear": ratio_safe(geo_residual, linear),
                    "linear_minus_residual": linear - geo_residual if np.isfinite(linear) and np.isfinite(geo_residual) else np.nan,
                    "nonlinear_retention_vs_linear": ratio_safe(nonlinear, linear),
                })
    return pd.DataFrame(macro_rows), pd.DataFrame(geometry_rows)


def build_stage5_claim_metrics(
    macro_retention: pd.DataFrame,
    geometry_retention: pd.DataFrame,
    geometry_metrics: pd.DataFrame,
    pc_metrics: pd.DataFrame,
    gain_metrics: pd.DataFrame,
    training_manifests: Mapping[int, Mapping[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    seeds = sorted(macro_retention["seed"].unique())
    splits = sorted(macro_retention["split"].astype(str).unique())
    for seed in seeds:
        train_manifest = dict(training_manifests.get(int(seed), {}))
        explained = list(train_manifest.get("pca_explained_variance_ratio_first_10", []))
        pc12_variance = float(sum(finite_float(value) for value in explained[:2])) if explained else np.nan
        for split in splits:
            def macro_value(domain: str, column: str) -> float:
                subset = macro_retention[(macro_retention["seed"] == seed) & (macro_retention["split"].astype(str) == split) & (macro_retention["domain"] == domain)]
                return finite_float(subset.iloc[0].get(column)) if not subset.empty else np.nan

            def geometry_value(domain: str, column: str) -> float:
                subset = geometry_retention[(geometry_retention["seed"] == seed) & (geometry_retention["split"].astype(str) == split) & (geometry_retention["domain"] == domain)]
                return finite_float(subset.iloc[0].get(column)) if not subset.empty else np.nan

            model_row = geometry_metrics[(geometry_metrics["seed"] == seed) & (geometry_metrics["split"].astype(str) == split) & (geometry_metrics["representation"].astype(str) == "model_readout")]
            model_row = model_row.iloc[0] if not model_row.empty else pd.Series(dtype=float)
            pc_subset = pc_metrics[(pc_metrics["seed"] == seed) & (pc_metrics["split"].astype(str) == split)] if "split" in pc_metrics.columns else pd.DataFrame()
            top_pc_m = finite_float(pc_subset["abs_corr_M"].max()) if not pc_subset.empty and "abs_corr_M" in pc_subset.columns else np.nan
            top_pc_psi = finite_float(pc_subset["abs_corr_Psi"].max()) if not pc_subset.empty and "abs_corr_Psi" in pc_subset.columns else np.nan
            gain_subset = gain_metrics[(gain_metrics["seed"] == seed) & (gain_metrics["split"].astype(str) == split)] if "split" in gain_metrics.columns else pd.DataFrame()
            gain_row = gain_subset.iloc[0] if not gain_subset.empty else pd.Series(dtype=float)
            overall_macro = macro_value("macrostructure_composite_descriptive", "macro_retention_vs_full")
            overall_residual = macro_value("macrostructure_composite_descriptive", "residual_retention_vs_full")
            overall_linear = geometry_value("macrostructure_composite_descriptive", "linear_retention_vs_model")
            overall_geo_residual = geometry_value("macrostructure_composite_descriptive", "residual_retention_vs_model")
            claims = {
                "macro_bottleneck_overall_retention": overall_macro,
                "macro_bottleneck_coordinate_retention": macro_value("coordinate_score", "macro_retention_vs_full"),
                "macro_bottleneck_closure_retention": macro_value("closure_score", "macro_retention_vs_full"),
                "macro_bottleneck_drift_retention": macro_value("drift_score", "macro_retention_vs_full"),
                "macro_bottleneck_transition_retention": macro_value("transition_score", "macro_retention_vs_full"),
                "residual_hidden_overall_retention": overall_residual,
                "macro_minus_residual_overall_score": macro_value("macrostructure_composite_descriptive", "macro_minus_residual"),
                "macro_task_retention": macro_value("task_score", "macro_retention_vs_full"),
                "residual_task_retention": macro_value("task_score", "residual_retention_vs_full"),
                "linear_hidden_overall_retention": overall_linear,
                "linear_hidden_coordinate_retention": geometry_value("coordinate_score", "linear_retention_vs_model"),
                "residual_geometry_overall_retention": overall_geo_residual,
                "linear_minus_residual_overall_score": geometry_value("macrostructure_composite_descriptive", "linear_minus_residual"),
                "cca_corr_1": finite_float(model_row.get("cca_corr_1")),
                "cca_corr_2": finite_float(model_row.get("cca_corr_2")),
                "top_abs_pc_corr_M": top_pc_m,
                "top_abs_pc_corr_Psi": top_pc_psi,
                "nonlinear_gain_corr_M": finite_float(gain_row.get("nonlinear_gain_corr_M")),
                "nonlinear_gain_corr_Psi": finite_float(gain_row.get("nonlinear_gain_corr_Psi")),
                "nonlinear_gain_rmse_reduction_M": finite_float(gain_row.get("nonlinear_gain_rmse_reduction_M")),
                "nonlinear_gain_rmse_reduction_Psi": finite_float(gain_row.get("nonlinear_gain_rmse_reduction_Psi")),
                "twonn_dimension": finite_float(model_row.get("twonn_dimension")),
                "participation_ratio_train": finite_float(model_row.get("participation_ratio_train")),
                "effective_rank_train": finite_float(model_row.get("effective_rank_train")),
                "pca_pc1_pc2_explained_variance": pc12_variance,
            }
            for claim, value in claims.items():
                if claim == "macro_minus_residual_overall_score":
                    condition = np.isfinite(value) and value > 0
                elif claim == "linear_minus_residual_overall_score":
                    condition = np.isfinite(value) and value > 0
                elif claim.startswith("nonlinear_gain_corr"):
                    condition = np.isfinite(value) and abs(value) <= 0.02
                else:
                    condition = np.nan
                rows.append({
                    "seed": int(seed), "split": split, "claim_metric": claim,
                    "value": value,
                    "descriptive_condition_satisfied": float(condition) if isinstance(condition, (bool, np.bool_)) else np.nan,
                })
            rows.extend([
                {
                    "seed": int(seed), "split": split,
                    "claim_metric": "macro_retention_exceeds_residual_retention",
                    "value": overall_macro - overall_residual if np.isfinite(overall_macro) and np.isfinite(overall_residual) else np.nan,
                    "descriptive_condition_satisfied": float(np.isfinite(overall_macro) and np.isfinite(overall_residual) and overall_macro > overall_residual),
                },
                {
                    "seed": int(seed), "split": split,
                    "claim_metric": "linear_retention_exceeds_residual_geometry",
                    "value": overall_linear - overall_geo_residual if np.isfinite(overall_linear) and np.isfinite(overall_geo_residual) else np.nan,
                    "descriptive_condition_satisfied": float(np.isfinite(overall_linear) and np.isfinite(overall_geo_residual) and overall_linear > overall_geo_residual),
                },
            ])
    return pd.DataFrame(rows)


def generic_numeric_long(frame: pd.DataFrame, experiment: str, id_columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    ids = [column for column in id_columns if column in frame.columns]
    for _, row in frame.iterrows():
        identity = {column: row.get(column) for column in ids}
        for column in frame.columns:
            if column in ids:
                continue
            value = finite_float(row.get(column))
            if np.isfinite(value):
                rows.append({"experiment": experiment, **identity, "metric": column, "value": value})
    return pd.DataFrame(rows)


def build_split_differences(
    df: pd.DataFrame,
    validation_split: str,
    confirmation_split: str,
    identity_columns: Sequence[str],
) -> pd.DataFrame:
    if df.empty or "seed" not in df.columns or "split" not in df.columns or "value" not in df.columns:
        return pd.DataFrame()
    identities = [column for column in identity_columns if column in df.columns]
    validation = df[df["split"].astype(str) == validation_split].copy()
    confirmation = df[df["split"].astype(str) == confirmation_split].copy()
    lookup = {
        (int(row["seed"]),) + tuple(row.get(column) for column in identities): finite_float(row.get("value"))
        for _, row in validation.iterrows()
    }
    rows: List[Dict[str, Any]] = []
    for _, row in confirmation.iterrows():
        key = (int(row["seed"]),) + tuple(row.get(column) for column in identities)
        validation_value = lookup.get(key, np.nan)
        confirmation_value = finite_float(row.get("value"))
        if not np.isfinite(validation_value) or not np.isfinite(confirmation_value):
            continue
        rows.append({
            "seed": int(row["seed"]),
            **{column: row.get(column) for column in identities},
            "validation_split": validation_split,
            "confirmation_split": confirmation_split,
            "validation_value": validation_value,
            "confirmation_value": confirmation_value,
            "confirmation_minus_validation": confirmation_value - validation_value,
            "absolute_gap": abs(confirmation_value - validation_value),
        })
    return pd.DataFrame(rows)


def build_reference_seed_differences(
    df: pd.DataFrame,
    reference_seed: int,
    identity_columns: Sequence[str],
) -> pd.DataFrame:
    if df.empty or "seed" not in df.columns or "value" not in df.columns:
        return pd.DataFrame()
    identities = [column for column in identity_columns if column in df.columns]
    reference = df[df["seed"] == reference_seed].copy()
    lookup = {
        tuple(row.get(column) for column in identities): finite_float(row.get("value"))
        for _, row in reference.iterrows()
    }
    rows: List[Dict[str, Any]] = []
    for _, row in df[df["seed"] != reference_seed].iterrows():
        key = tuple(row.get(column) for column in identities)
        reference_value = lookup.get(key, np.nan)
        value = finite_float(row.get("value"))
        if not np.isfinite(value) or not np.isfinite(reference_value):
            continue
        rows.append({
            "seed": int(row["seed"]),
            "reference_seed": int(reference_seed),
            **{column: row.get(column) for column in identities},
            "value": value,
            "reference_value": reference_value,
            "difference_from_reference": value - reference_value,
            "absolute_difference_from_reference": abs(value - reference_value),
        })
    return pd.DataFrame(rows)


def key_metric_ledger(
    stage4_summary: pd.DataFrame,
    stage4_claim_summary: pd.DataFrame,
    stage5_claim_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    stage4_keys = {
        "predictive_state_event_ssl": (
            "coordinate_corr_M", "coordinate_corr_Psi", "coordinate_rmse_M", "coordinate_rmse_Psi",
            "one_step_rmse_M", "one_step_rmse_Psi", "next_state_occupancy_js",
            "anchor_drift_vector_corr", "learned_plane_drift_vector_corr",
            "learned_plane_occupancy_weighted_local_drift_cosine",
            "learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr",
            "learned_plane_inward_fraction_to_reference",
        ),
        "pure_event_ssl_probe": (
            "coordinate_corr_M", "coordinate_corr_Psi", "learned_plane_drift_vector_corr",
            "learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr",
        ),
        "task_only": (
            "coordinate_corr_M", "coordinate_corr_Psi", "learned_plane_drift_vector_corr",
            "learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr", "task_bce",
        ),
        "time_shuffle_control": (
            "coordinate_corr_M", "coordinate_corr_Psi", "anchor_drift_vector_corr",
            "learned_plane_drift_vector_corr", "learned_plane_occupancy_weighted_local_drift_cosine",
            "learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr",
        ),
        "tag_support_randomized": (
            "coordinate_corr_M", "coordinate_corr_Psi", "learned_plane_drift_vector_corr",
            "learned_plane_inward_fraction_to_reference", "learned_plane_transition_mean_row_tv",
            "learned_plane_self_transition_corr",
        ),
    }
    for model, metrics in stage4_keys.items():
        subset = stage4_summary[(stage4_summary["model_label"] == model) & (stage4_summary["split"] == "B_confirm") & (stage4_summary["metric"].isin(metrics))]
        for _, row in subset.iterrows():
            rows.append({
                "section": "Stage 4 random-seed robustness",
                "experiment": model,
                "split": "B_confirm",
                "metric": row["metric"],
                "n_seeds": row["n_seeds"],
                "mean": row["mean"], "sd": row["sd"], "ci_lower": row["ci_lower"], "ci_upper": row["ci_upper"],
                "min": row["min"], "max": row["max"],
                "manuscript_use": "Additional Information seed-summary table",
            })
    for _, row in stage4_claim_summary[stage4_claim_summary["split"] == "B_confirm"].iterrows():
        rows.append({
            "section": "Stage 4 claim consistency",
            "experiment": "cross-seed claim",
            "split": "B_confirm",
            "metric": row["claim"],
            "n_seeds": row["n_seeds"], "mean": row["mean"], "sd": row["sd"],
            "ci_lower": row["ci_lower"], "ci_upper": row["ci_upper"], "min": row["min"], "max": row["max"],
            "condition_fraction": row.get("condition_fraction", np.nan),
            "manuscript_use": "Report direction consistency across seeds",
        })
    stage5_keys = {
        "macro_bottleneck_overall_retention", "macro_bottleneck_coordinate_retention",
        "macro_bottleneck_closure_retention", "macro_bottleneck_drift_retention",
        "macro_bottleneck_transition_retention", "residual_hidden_overall_retention",
        "macro_task_retention", "residual_task_retention", "linear_hidden_overall_retention",
        "linear_hidden_coordinate_retention", "residual_geometry_overall_retention",
        "cca_corr_1", "cca_corr_2", "top_abs_pc_corr_M", "top_abs_pc_corr_Psi",
        "nonlinear_gain_corr_M", "nonlinear_gain_corr_Psi", "pca_pc1_pc2_explained_variance",
    }
    subset = stage5_claim_summary[(stage5_claim_summary["split"] == "B_confirm") & (stage5_claim_summary["claim_metric"].isin(stage5_keys))]
    for _, row in subset.iterrows():
        rows.append({
            "section": "Stage 5 random-seed robustness",
            "experiment": "representation organization",
            "split": "B_confirm",
            "metric": row["claim_metric"],
            "n_seeds": row["n_seeds"], "mean": row["mean"], "sd": row["sd"],
            "ci_lower": row["ci_lower"], "ci_upper": row["ci_upper"], "min": row["min"], "max": row["max"],
            "condition_fraction": row.get("condition_fraction", np.nan),
            "manuscript_use": "Additional Information representation-seed table",
        })
    return pd.DataFrame(rows)


def build_quality_gates(
    seeds: Sequence[int],
    seed_audit: pd.DataFrame,
    stage4_summary: pd.DataFrame,
    stage4_claim_summary: pd.DataFrame,
    stage5_claim_summary: pd.DataFrame,
) -> pd.DataFrame:
    expected_n = len(seeds)
    rows: List[Dict[str, Any]] = []

    def add(gate: str, passed: bool, observed: Any, expected: Any, role: str) -> None:
        rows.append({
            "gate": gate,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
            "role": role,
        })

    add("requested_seed_count", len(seed_audit) == expected_n, len(seed_audit), expected_n, "Completeness of the random-seed panel.")
    add("all_runs_completion_verified", bool(seed_audit["completion_verified"].all()), int(seed_audit["completion_verified"].sum()), expected_n, "Every run is supported by workflow markers or formal main-experiment outputs.")
    additional_runs = seed_audit[seed_audit["run_source"].astype(str) == "random_seed_directory"]
    code_signature_count = int(additional_runs.loc[additional_runs["code_snapshot_signature"].astype(str) != "", "code_snapshot_signature"].nunique(dropna=False))
    add("single_code_snapshot", code_signature_count == 1, code_signature_count, 1, "The five additional seed runs use one identical formal code snapshot.")
    add("main_reference_present", int((seed_audit["run_source"].astype(str) == "main_experiment_reference").sum()) == 1, int((seed_audit["run_source"].astype(str) == "main_experiment_reference").sum()), 1, "The main-text seed is read from the formal main experiment output.")
    add("single_input_schema", seed_audit["input_schema_signature"].nunique(dropna=False) == 1, int(seed_audit["input_schema_signature"].nunique(dropna=False)), 1, "All runs use the same prepared-input schema and split shapes.")
    add("single_fixed_k6_partition", seed_audit[["stage1_partition_metadata_sha256", "stage1_partition_centers_sha256"]].drop_duplicates().shape[0] == 1, int(seed_audit[["stage1_partition_metadata_sha256", "stage1_partition_centers_sha256"]].drop_duplicates().shape[0]), 1, "All seeds use the same Stage-1 fixed K=6 partition.")

    required_stage4 = {
        "predictive_state_event_ssl": {
            "coordinate_corr_M", "coordinate_corr_Psi", "coordinate_rmse_M", "coordinate_rmse_Psi",
            "one_step_rmse_M", "one_step_rmse_Psi", "next_state_occupancy_js",
            "anchor_drift_vector_corr", "learned_plane_drift_vector_corr",
            "learned_plane_occupancy_weighted_local_drift_cosine",
            "learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr",
            "learned_plane_inward_fraction_to_reference",
        },
        "pure_event_ssl_probe": {"coordinate_corr_M", "coordinate_corr_Psi", "learned_plane_drift_vector_corr", "learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr"},
        "task_only": {"coordinate_corr_M", "coordinate_corr_Psi", "learned_plane_drift_vector_corr", "learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr", "task_bce"},
        "time_shuffle_control": {"coordinate_corr_M", "coordinate_corr_Psi", "anchor_drift_vector_corr", "learned_plane_drift_vector_corr", "learned_plane_occupancy_weighted_local_drift_cosine", "learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr"},
        "tag_support_randomized": {"coordinate_corr_M", "coordinate_corr_Psi", "learned_plane_drift_vector_corr", "learned_plane_inward_fraction_to_reference", "learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr"},
    }
    for model, metrics in required_stage4.items():
        subset = stage4_summary[(stage4_summary["model_label"] == model) & (stage4_summary["split"] == "B_confirm") & (stage4_summary["metric"].isin(metrics))]
        complete = set(subset.loc[subset["n_seeds"] == expected_n, "metric"].astype(str))
        missing = sorted(metrics.difference(complete))
        add(f"stage4_key_metrics_complete::{model}", not missing, ";".join(missing) if missing else "complete", "all metrics with n_seeds=" + str(expected_n), "Complete B_confirm random-seed reporting for the model/control.")

    required_stage5 = {
        "macro_bottleneck_overall_retention", "macro_bottleneck_coordinate_retention",
        "macro_bottleneck_closure_retention", "macro_bottleneck_drift_retention",
        "macro_bottleneck_transition_retention", "residual_hidden_overall_retention",
        "macro_task_retention", "residual_task_retention", "linear_hidden_overall_retention",
        "linear_hidden_coordinate_retention", "residual_geometry_overall_retention",
        "cca_corr_1", "cca_corr_2", "top_abs_pc_corr_M", "top_abs_pc_corr_Psi",
        "nonlinear_gain_corr_M", "nonlinear_gain_corr_Psi", "pca_pc1_pc2_explained_variance",
    }
    subset = stage5_claim_summary[(stage5_claim_summary["split"] == "B_confirm") & (stage5_claim_summary["claim_metric"].isin(required_stage5))]
    complete = set(subset.loc[subset["n_seeds"] == expected_n, "claim_metric"].astype(str))
    missing = sorted(required_stage5.difference(complete))
    add("stage5_key_metrics_complete", not missing, ";".join(missing) if missing else "complete", "all metrics with n_seeds=" + str(expected_n), "Complete Stage-5 random-seed reporting.")

    for claim in (
        "time_shuffle_reverses_learned_plane_field",
        "tag_support_randomization_reduces_inward_transport",
    ):
        subset = stage4_claim_summary[(stage4_claim_summary["split"] == "B_confirm") & (stage4_claim_summary["claim"] == claim)]
        fraction = finite_float(subset.iloc[0].get("condition_fraction")) if not subset.empty else np.nan
        add(f"claim_direction::{claim}", np.isfinite(fraction), fraction, "descriptive fraction reported", "Direction consistency is reported without forcing the scientific conclusion.")
    return pd.DataFrame(rows)


def write_additional_information_report(
    path: Path,
    seeds: Sequence[int],
    seed_audit: pd.DataFrame,
    stage4_summary: pd.DataFrame,
    stage4_contrast_summary: pd.DataFrame,
    stage4_claim_summary: pd.DataFrame,
    stage5_claim_summary: pd.DataFrame,
    training_summary: pd.DataFrame,
    stage4_reference_summary: pd.DataFrame,
    stage5_reference_summary: pd.DataFrame,
    stage4_split_gap_summary: pd.DataFrame,
    stage5_split_gap_summary: pd.DataFrame,
    ci_level: float,
    reference_seed: int,
) -> None:
    lines: List[str] = []
    lines.append("# Event-SSL random-seed robustness")
    lines.append("")
    lines.append(
        f"The complete Event-SSL pipeline was evaluated for {len(seeds)} seeds: "
        + ", ".join(str(seed) for seed in seeds)
        + f". Seed {reference_seed} is the main-text run read directly from the formal main experiment output; the remaining seeds are independent additional runs. "
        + "Values are reported as mean ± sample standard deviation [minimum, maximum]. "
        + f"The interval columns use a {ci_level:.0%} Student-t interval across seeds."
    )
    lines.append("")
    lines.append(
        "These intervals characterize pipeline random-seed variation, not sampling uncertainty in the learner population. "
        "The time-shuffle and tag/support controls also vary the corresponding randomization instance; Stage-5 results include probe, sampling and representation-clustering randomness."
    )
    lines.append("")
    lines.append("## Reproducibility audit")
    lines.append("")
    audit_cols = ["seed", "run_source", "model_seed", "control_seed", "probe_seed", "eval_seed", "prep_seed", "completion_verified", "completion_evidence", "code_snapshot_signature"]
    lines.append(markdown_table(seed_audit[audit_cols]))
    lines.append("")

    lines.append("## Predictive-state Event-SSL")
    lines.append("")
    key_main = [
        "coordinate_corr_M", "coordinate_corr_Psi", "coordinate_rmse_M", "coordinate_rmse_Psi",
        "one_step_rmse_M", "one_step_rmse_Psi", "next_state_occupancy_js",
        "anchor_drift_vector_corr", "learned_plane_drift_vector_corr",
        "learned_plane_occupancy_weighted_local_drift_cosine",
        "learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr",
        "learned_plane_inward_fraction_to_reference",
    ]
    table = stage4_summary[(stage4_summary["model_label"] == "predictive_state_event_ssl") & (stage4_summary["split"] == "B_confirm") & (stage4_summary["metric"].isin(key_main))].copy()
    table["seed_summary"] = table.apply(fmt_mean_sd, axis=1)
    lines.append(markdown_table(table[["metric", "seed_summary", "ci_lower", "ci_upper", "n_seeds"]]))
    lines.append("")

    lines.append("## Objective and perturbation controls")
    lines.append("")
    control_metrics = [
        "coordinate_corr_M", "coordinate_corr_Psi", "next_state_occupancy_js",
        "anchor_drift_vector_corr", "learned_plane_drift_vector_corr",
        "learned_plane_occupancy_weighted_local_drift_cosine",
        "learned_plane_inward_fraction_to_reference",
        "learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr", "task_bce",
    ]
    controls = stage4_summary[(stage4_summary["model_label"] != "predictive_state_event_ssl") & (stage4_summary["split"] == "B_confirm") & (stage4_summary["metric"].isin(control_metrics))].copy()
    controls["seed_summary"] = controls.apply(fmt_mean_sd, axis=1)
    lines.append(markdown_table(controls[["model_display", "metric", "seed_summary", "ci_lower", "ci_upper", "n_seeds"]], max_rows=80))
    lines.append("")

    lines.append("## Paired same-seed contrasts")
    lines.append("")
    paired_keys = {
        "learned_plane_drift_vector_corr", "learned_plane_inward_fraction_to_reference",
        "learned_plane_transition_mean_row_tv", "coordinate_corr_M", "coordinate_corr_Psi",
    }
    paired = stage4_contrast_summary[(stage4_contrast_summary["split"] == "B_confirm") & (stage4_contrast_summary["metric"].isin(paired_keys))].copy()
    paired["seed_summary"] = paired.apply(fmt_mean_sd, axis=1)
    lines.append(markdown_table(paired[["control_display", "metric", "seed_summary", "ci_lower", "ci_upper", "positive_fraction", "n_seeds"]], max_rows=80))
    lines.append("")

    lines.append("## Claim-direction consistency")
    lines.append("")
    claims = stage4_claim_summary[stage4_claim_summary["split"] == "B_confirm"].copy()
    claims["seed_summary"] = claims.apply(fmt_mean_sd, axis=1)
    lines.append(markdown_table(claims[["claim", "seed_summary", "condition_fraction", "n_seeds"]]))
    lines.append("")

    lines.append("## Macro-sufficiency and representation geometry")
    lines.append("")
    stage5_keys = [
        "macro_bottleneck_overall_retention", "macro_bottleneck_coordinate_retention",
        "macro_bottleneck_closure_retention", "macro_bottleneck_drift_retention",
        "macro_bottleneck_transition_retention", "residual_hidden_overall_retention",
        "macro_task_retention", "residual_task_retention", "linear_hidden_overall_retention",
        "linear_hidden_coordinate_retention", "residual_geometry_overall_retention",
        "cca_corr_1", "cca_corr_2", "top_abs_pc_corr_M", "top_abs_pc_corr_Psi",
        "nonlinear_gain_corr_M", "nonlinear_gain_corr_Psi", "pca_pc1_pc2_explained_variance",
    ]
    stage5 = stage5_claim_summary[(stage5_claim_summary["split"] == "B_confirm") & (stage5_claim_summary["claim_metric"].isin(stage5_keys))].copy()
    stage5["seed_summary"] = stage5.apply(fmt_mean_sd, axis=1)
    lines.append(markdown_table(stage5[["claim_metric", "seed_summary", "ci_lower", "ci_upper", "condition_fraction", "n_seeds"]], max_rows=80))
    lines.append("")

    lines.append("## Validation-confirmation stability across seeds")
    lines.append("")
    gap4 = stage4_split_gap_summary[
        (stage4_split_gap_summary.get("model_label", pd.Series(dtype=str)) == "predictive_state_event_ssl")
        & (stage4_split_gap_summary.get("metric", pd.Series(dtype=str)).isin([
            "coordinate_corr_M", "coordinate_corr_Psi", "learned_plane_drift_vector_corr",
            "learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr",
        ]))
    ].copy()
    if not gap4.empty:
        gap4["seed_summary"] = gap4.apply(fmt_mean_sd, axis=1)
        lines.append(markdown_table(gap4[["metric", "seed_summary", "ci_lower", "ci_upper", "n_seeds"]], max_rows=30))
    gap5 = stage5_split_gap_summary[
        stage5_split_gap_summary.get("claim_metric", pd.Series(dtype=str)).isin([
            "macro_bottleneck_overall_retention", "residual_hidden_overall_retention",
            "linear_hidden_overall_retention", "cca_corr_1", "cca_corr_2",
        ])
    ].copy()
    if not gap5.empty:
        gap5["seed_summary"] = gap5.apply(fmt_mean_sd, axis=1)
        lines.append("")
        lines.append(markdown_table(gap5[["claim_metric", "seed_summary", "ci_lower", "ci_upper", "n_seeds"]], max_rows=30))
    lines.append("")

    lines.append(f"## Difference from the main-text seed {reference_seed} run")
    lines.append("")
    ref4 = stage4_reference_summary[
        (stage4_reference_summary.get("split", pd.Series(dtype=str)).astype(str) == "B_confirm")
        & (stage4_reference_summary.get("metric", pd.Series(dtype=str)).isin([
            "coordinate_corr_M", "coordinate_corr_Psi", "learned_plane_drift_vector_corr",
            "learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr",
        ]))
    ].copy()
    if not ref4.empty:
        ref4["seed_summary"] = ref4.apply(fmt_mean_sd, axis=1)
        lines.append(markdown_table(ref4[["model_label", "metric", "seed_summary", "n_seeds"]], max_rows=60))
    ref5 = stage5_reference_summary[
        (stage5_reference_summary.get("split", pd.Series(dtype=str)).astype(str) == "B_confirm")
        & (stage5_reference_summary.get("claim_metric", pd.Series(dtype=str)).isin([
            "macro_bottleneck_overall_retention", "residual_hidden_overall_retention",
            "linear_hidden_overall_retention", "cca_corr_1", "cca_corr_2",
        ]))
    ].copy()
    if not ref5.empty:
        ref5["seed_summary"] = ref5.apply(fmt_mean_sd, axis=1)
        lines.append("")
        lines.append(markdown_table(ref5[["claim_metric", "seed_summary", "n_seeds"]], max_rows=30))
    lines.append("")

    lines.append("## Training and checkpoint selection")
    lines.append("")
    training = training_summary.copy()
    training["seed_summary"] = training.apply(fmt_mean_sd, axis=1)
    lines.append(markdown_table(training[["model_label", "metric", "seed_summary", "ci_lower", "ci_upper", "n_seeds"]], max_rows=60))
    lines.append("")

    lines.append("## Data files")
    lines.append("")
    lines.append(
        "The accompanying tables retain every numeric column emitted by the formal Stage-4 and Stage-5 evaluators, "
        "the per-seed fixed-K=6 transition diagnostics, all same-seed paired contrasts, and the descriptive Stage-5 score/retention calculations."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Event-SSL random-seed statistics for Additional Information.")
    parser.add_argument(
        "--seed-root", type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/random_seed_experiments"),
        help="Directory containing seed_<seed> experiment roots.",
    )
    parser.add_argument("--main-output-root", type=Path, default=DEFAULT_MAIN_OUTPUT_ROOT, help="Formal main experiment output root used for the reference seed.")
    parser.add_argument("--reference-seed", type=int, default=DEFAULT_REFERENCE_SEED)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    splits = tuple(str(split) for split in args.splits)
    if not 0 < args.ci_level < 1:
        raise ValueError("--ci-level must lie between 0 and 1.")
    seed_root = args.seed_root.resolve()
    main_output_root = args.main_output_root.resolve()
    reference_seed = int(args.reference_seed)
    if reference_seed not in seeds:
        raise ValueError("--reference-seed must be included in --seeds.")
    output_root = (args.output_root or (seed_root / "additional_information_random_seed_summary")).resolve()
    table_root = output_root / "tables"
    report_root = output_root / "reports"
    metadata_root = output_root / "metadata"
    for directory in (table_root, report_root, metadata_root):
        directory.mkdir(parents=True, exist_ok=True)

    all_sources: List[SourceRecord] = []
    seed_audit_rows: List[Dict[str, Any]] = []
    stage4_frames: List[pd.DataFrame] = []
    stage4_domain_frames: List[pd.DataFrame] = []
    transition_frames: List[pd.DataFrame] = []
    training_rows: List[Dict[str, Any]] = []
    macro_frames: List[pd.DataFrame] = []
    macro_train_frames: List[pd.DataFrame] = []
    geometry_frames: List[pd.DataFrame] = []
    control_audit_frames: List[pd.DataFrame] = []
    pc_frames: List[pd.DataFrame] = []
    gain_frames: List[pd.DataFrame] = []
    train_pc_frames: List[pd.DataFrame] = []
    geometry_training_manifests: Dict[int, Mapping[str, Any]] = {}

    for seed in seeds:
        experiment_root, run_source = resolve_experiment_root(seed_root, main_output_root, seed, reference_seed)
        audit, records = audit_seed_directory(experiment_root, seed, args.strict, run_source)
        seed_audit_rows.append(audit)
        all_sources.extend(records)
        stage4, domains, transitions, seed_training = collect_stage4_for_seed(experiment_root, seed, splits, all_sources)
        stage4_frames.append(stage4)
        stage4_domain_frames.append(domains)
        transition_frames.append(transitions)
        training_rows.extend(seed_training)
        control_audit_frames.append(collect_control_randomization_audit(experiment_root, seed, all_sources, args.strict, run_source))
        macro, macro_train, geometry, pc, gain, train_pc, train_manifest = collect_stage5_for_seed(experiment_root, seed, splits, all_sources)
        macro_frames.append(macro)
        macro_train_frames.append(macro_train)
        geometry_frames.append(geometry)
        pc_frames.append(pc)
        gain_frames.append(gain)
        train_pc_frames.append(train_pc)
        geometry_training_manifests[int(seed)] = train_manifest

    seed_audit = pd.DataFrame(seed_audit_rows).sort_values("seed").reset_index(drop=True)
    additional_seed_audit = seed_audit[seed_audit["run_source"].astype(str) == "random_seed_directory"]
    additional_code_signatures = additional_seed_audit.loc[additional_seed_audit["code_snapshot_signature"].astype(str) != "", "code_snapshot_signature"]
    if args.strict and additional_code_signatures.nunique(dropna=False) != 1:
        raise RuntimeError("Additional seed runs do not share one identical formal code snapshot.")
    if args.strict and seed_audit["input_schema_signature"].nunique(dropna=False) != 1:
        raise RuntimeError("Seed runs do not share one identical prepared-input schema and split shape contract.")
    if args.strict and seed_audit["stage1_partition_metadata_sha256"].nunique(dropna=False) != 1:
        raise RuntimeError("Seed runs do not reference one Stage-1 fixed-K metadata hash.")
    if args.strict and seed_audit["stage1_partition_centers_sha256"].nunique(dropna=False) != 1:
        raise RuntimeError("Seed runs do not reference one Stage-1 fixed-K centres hash.")
    if args.strict and len(seed_audit) != len(seeds):
        raise RuntimeError("Not all requested seeds were collected.")

    stage4_metrics = pd.concat(stage4_frames, ignore_index=True, sort=False)
    stage4_domains = pd.concat(stage4_domain_frames, ignore_index=True, sort=False)
    transition_all = pd.concat(transition_frames, ignore_index=True, sort=False)
    training = pd.DataFrame(training_rows)
    macro_metrics = pd.concat(macro_frames, ignore_index=True, sort=False)
    macro_train_metrics = pd.concat(macro_train_frames, ignore_index=True, sort=False)
    geometry_metrics = pd.concat(geometry_frames, ignore_index=True, sort=False)
    control_randomization_audit = pd.concat(control_audit_frames, ignore_index=True, sort=False)
    control_randomization_summary = summarize_long_table(control_randomization_audit, ["control", "split", "metric"], args.ci_level)
    pc_metrics = pd.concat(pc_frames, ignore_index=True, sort=False)
    gain_metrics = pd.concat(gain_frames, ignore_index=True, sort=False)
    train_pc_metrics = pd.concat(train_pc_frames, ignore_index=True, sort=False)

    stage4_long = stage4_metrics_to_long(stage4_metrics)
    stage4_summary = summarize_long_table(stage4_long, ["model_label", "model_display", "split", "metric", "direction"], args.ci_level)
    stage4_domain_summary = summarize_long_table(stage4_domains, ["model_label", "model_display", "split", "domain"], args.ci_level)
    stage4_contrasts = build_stage4_paired_contrasts(stage4_metrics)
    contrast_long = stage4_contrasts.rename(columns={"main_improvement_in_expected_direction": "value"})
    stage4_contrast_summary = summarize_long_table(
        contrast_long[np.isfinite(pd.to_numeric(contrast_long["value"], errors="coerce"))],
        ["control_model", "control_display", "split", "metric", "direction"],
        args.ci_level,
    )
    stage4_claims = build_stage4_claim_metrics(stage4_metrics, stage4_domains)
    stage4_claim_summary = summarize_long_table(stage4_claims, ["split", "claim"], args.ci_level)
    condition_lookup = stage4_claims.groupby(["split", "claim"], as_index=False)["condition_satisfied"].mean().rename(columns={"condition_satisfied": "condition_fraction"})
    stage4_claim_summary = stage4_claim_summary.merge(condition_lookup, on=["split", "claim"], how="left")

    statewise = transition_all[transition_all.get("record_type", pd.Series(index=transition_all.index, dtype=object)).isna()].copy()
    cells = transition_all[transition_all.get("record_type", pd.Series(index=transition_all.index, dtype=object)).astype(str) == "cell"].copy()
    statewise_long = generic_numeric_long(statewise, "stage4_transition_statewise", ["seed", "split", "model_label", "state"])
    cell_long = generic_numeric_long(cells, "stage4_transition_cells", ["seed", "split", "model_label", "state", "next_state", "record_type"])
    statewise_summary = summarize_long_table(statewise_long, ["split", "model_label", "state", "metric"], args.ci_level)
    cell_summary = summarize_long_table(cell_long, ["split", "model_label", "state", "next_state", "metric"], args.ci_level)

    stage5_domains = build_stage5_domain_rows(macro_metrics, geometry_metrics)
    macro_retention, geometry_retention = build_stage5_retention_tables(stage5_domains)
    stage5_claims = build_stage5_claim_metrics(
        macro_retention, geometry_retention, geometry_metrics, pc_metrics, gain_metrics, geometry_training_manifests,
    )
    stage5_claim_summary = summarize_long_table(stage5_claims, ["split", "claim_metric"], args.ci_level)
    stage5_condition_lookup = stage5_claims.groupby(["split", "claim_metric"], as_index=False)["descriptive_condition_satisfied"].mean().rename(columns={"descriptive_condition_satisfied": "condition_fraction"})
    stage5_claim_summary = stage5_claim_summary.merge(stage5_condition_lookup, on=["split", "claim_metric"], how="left")

    stage4_split_gaps = build_split_differences(
        stage4_long, "A_val", "B_confirm", ["model_label", "model_display", "metric", "direction"],
    )
    stage4_split_gap_summary_input = stage4_split_gaps.drop(columns=["validation_value", "confirmation_value"], errors="ignore").rename(columns={"confirmation_minus_validation": "value"})
    stage4_split_gap_summary = summarize_long_table(
        stage4_split_gap_summary_input, ["model_label", "model_display", "metric", "direction"], args.ci_level,
    )
    stage5_split_gaps = build_split_differences(
        stage5_claims.rename(columns={"claim_metric": "metric"}), "A_val", "B_confirm", ["metric"],
    ).rename(columns={"metric": "claim_metric"})
    stage5_split_gap_summary_input = stage5_split_gaps.drop(columns=["validation_value", "confirmation_value"], errors="ignore").rename(columns={"confirmation_minus_validation": "value"})
    stage5_split_gap_summary = summarize_long_table(
        stage5_split_gap_summary_input, ["claim_metric"], args.ci_level,
    )

    reference_seed = int(args.reference_seed)
    stage4_reference = build_reference_seed_differences(
        stage4_long, reference_seed, ["model_label", "model_display", "split", "metric", "direction"],
    )
    stage4_reference_summary_input = stage4_reference.drop(columns=["value"], errors="ignore").rename(columns={"difference_from_reference": "value"})
    stage4_reference_summary = summarize_long_table(
        stage4_reference_summary_input, ["reference_seed", "model_label", "model_display", "split", "metric", "direction"], args.ci_level,
    )
    stage5_reference = build_reference_seed_differences(
        stage5_claims.rename(columns={"claim_metric": "metric"}), reference_seed, ["split", "metric"],
    ).rename(columns={"metric": "claim_metric"})
    stage5_reference_summary_input = stage5_reference.drop(columns=["value"], errors="ignore").rename(columns={"difference_from_reference": "value"})
    stage5_reference_summary = summarize_long_table(
        stage5_reference_summary_input, ["reference_seed", "split", "claim_metric"], args.ci_level,
    )

    all_numeric_parts = [
        stage4_long.assign(stage="stage4", experiment="stage4_model_metrics"),
        generic_numeric_long(macro_metrics, "stage5_macro_sufficiency", ["seed", "split", "representation"]).assign(stage="stage5"),
        generic_numeric_long(macro_train_metrics, "stage5_macro_sufficiency_training", ["seed", "representation"]).assign(stage="stage5"),
        generic_numeric_long(geometry_metrics, "stage5_representation_geometry", ["seed", "split", "representation"]).assign(stage="stage5"),
        control_randomization_audit.assign(stage="stage4_controls", experiment="control_randomization_audit"),
        generic_numeric_long(pc_metrics, "stage5_pc_correlations", ["seed", "split", "component"]).assign(stage="stage5"),
        generic_numeric_long(gain_metrics, "stage5_nonlinear_gain", ["seed", "split"]).assign(stage="stage5"),
        generic_numeric_long(train_pc_metrics, "stage5_train_pc_correlations", ["seed", "component"]).assign(stage="stage5"),
    ]
    all_numeric = pd.concat([frame for frame in all_numeric_parts if not frame.empty], ignore_index=True, sort=False)
    all_numeric_summary = summarize_long_table(
        all_numeric.rename(columns={"experiment": "source_experiment"}),
        [column for column in ("stage", "source_experiment", "model_label", "control", "split", "representation", "component", "metric") if column in all_numeric.columns or column == "source_experiment"],
        args.ci_level,
    )

    training_long = generic_numeric_long(training, "training", ["seed", "model_label", "selection_metric_name"])
    training_summary = summarize_long_table(training_long, ["model_label", "metric"], args.ci_level)

    quality_gates = build_quality_gates(seeds, seed_audit, stage4_summary, stage4_claim_summary, stage5_claim_summary)
    if args.strict and not bool(quality_gates[quality_gates["gate"].str.contains("complete|requested_seed_count|single_code_snapshot|single_input_schema|single_fixed_k6_partition|all_runs_completion_verified|main_reference_present", regex=True)]["passed"].all()):
        failed = quality_gates[~quality_gates["passed"]]["gate"].tolist()
        raise RuntimeError(f"Random-seed quality gates failed: {failed}")
    ledger = key_metric_ledger(stage4_summary, stage4_claim_summary, stage5_claim_summary)
    source_audit = pd.DataFrame([asdict(record) for record in all_sources])

    outputs: Dict[str, Any] = {}
    tables = {
        "random_seed_run_audit": seed_audit,
        "random_seed_source_audit": source_audit,
        "random_seed_stage4_metrics_wide": stage4_metrics,
        "random_seed_stage4_metrics_long": stage4_long,
        "random_seed_stage4_metric_summary": stage4_summary,
        "random_seed_stage4_domain_scores": stage4_domains,
        "random_seed_stage4_domain_summary": stage4_domain_summary,
        "random_seed_stage4_paired_control_contrasts": stage4_contrasts,
        "random_seed_stage4_paired_control_contrast_summary": stage4_contrast_summary,
        "random_seed_stage4_claim_metrics": stage4_claims,
        "random_seed_stage4_claim_summary": stage4_claim_summary,
        "random_seed_stage4_transition_statewise": statewise,
        "random_seed_stage4_transition_statewise_summary": statewise_summary,
        "random_seed_stage4_transition_cell_summary": cell_summary,
        "random_seed_stage5_macro_metrics": macro_metrics,
        "random_seed_stage5_macro_training_metrics": macro_train_metrics,
        "random_seed_stage5_geometry_metrics": geometry_metrics,
        "random_seed_control_randomization_audit": control_randomization_audit,
        "random_seed_control_randomization_summary": control_randomization_summary,
        "random_seed_stage5_pc_correlations": pc_metrics,
        "random_seed_stage5_train_pc_correlations": train_pc_metrics,
        "random_seed_stage5_nonlinear_gain": gain_metrics,
        "random_seed_stage5_domain_scores": stage5_domains,
        "random_seed_stage5_macro_retention": macro_retention,
        "random_seed_stage5_geometry_retention": geometry_retention,
        "random_seed_stage5_claim_metrics": stage5_claims,
        "random_seed_stage5_claim_summary": stage5_claim_summary,
        "random_seed_stage4_difference_from_reference_seed": stage4_reference,
        "random_seed_stage4_difference_from_reference_seed_summary": stage4_reference_summary,
        "random_seed_stage5_difference_from_reference_seed": stage5_reference,
        "random_seed_stage5_difference_from_reference_seed_summary": stage5_reference_summary,
        "random_seed_stage4_validation_confirmation_gaps": stage4_split_gaps,
        "random_seed_stage4_validation_confirmation_gap_summary": stage4_split_gap_summary,
        "random_seed_stage5_validation_confirmation_gaps": stage5_split_gaps,
        "random_seed_stage5_validation_confirmation_gap_summary": stage5_split_gap_summary,
        "random_seed_training_metrics": training,
        "random_seed_training_summary": training_summary,
        "random_seed_all_numeric_metrics_long": all_numeric,
        "random_seed_all_numeric_metrics_summary": all_numeric_summary,
        "random_seed_publication_metric_ledger": ledger,
        "random_seed_scientific_quality_gates": quality_gates,
    }
    for name, frame in tables.items():
        outputs[name] = write_table(frame, table_root / name)

    report_path = report_root / "event_ssl_random_seed_additional_information.md"
    write_additional_information_report(
        report_path, seeds, seed_audit, stage4_summary, stage4_contrast_summary,
        stage4_claim_summary, stage5_claim_summary,
        training_summary, stage4_reference_summary, stage5_reference_summary,
        stage4_split_gap_summary, stage5_split_gap_summary, args.ci_level, reference_seed,
    )
    outputs["additional_information_report"] = str(report_path.resolve())

    manifest = {
        "script": Path(__file__).name,
        "seed_root": str(seed_root),
        "main_output_root": str(main_output_root),
        "reference_seed_source": "formal main experiment output",
        "output_root": str(output_root),
        "seeds": list(seeds),
        "splits": list(splits),
        "ci_level": args.ci_level,
        "strict": bool(args.strict),
        "reference_seed": reference_seed,
        "primary_coordinates": ["M", "Psi"],
        "macrostate_k": EXPECTED_MACROSTATE_K,
        "randomness_scope": {
            "predictive_and_pure": "network initialization, minibatch order and configured probe/evaluation sampling",
            "task_only": "network initialization, minibatch order and A_train probe sampling",
            "time_shuffle": "network randomness plus a seed-specific within-user temporal permutation",
            "tag_support": "network randomness plus a seed-specific support-semantics donor permutation",
            "stage5": "seed-matched predictive checkpoint plus probe, subsampling and representation-clustering randomness",
        },
        "uncertainty_interpretation": "Student-t intervals summarize variation across complete random-seed pipelines and are not learner-sampling confidence intervals.",
        "stage5_score_contract": {
            "correlations_and_cosines": "(r + 1) / 2",
            "rmse": "1 / (1 + RMSE / 0.15)",
            "js_and_row_tv": "1 - x",
            "task_bce": "1 / (1 + BCE)",
            "macrostructure_composite": "mean of coordinate, closure, drift and transition scores",
        },
        "outputs": outputs,
    }
    save_json(manifest, metadata_root / "event_ssl_random_seed_collection_manifest.json")
    print(f"[random-seed report] wrote {report_path}")


if __name__ == "__main__":
    main()
