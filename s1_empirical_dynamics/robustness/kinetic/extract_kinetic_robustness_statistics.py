#!/usr/bin/env python3
"""Extract the kinetic-null, partition-sensitivity and learner-cluster numerical report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_OUTPUT_BASE = Path(
    os.environ.get("EDNET_KT4_OUTPUT_ROOT", "/data/datasets/KT4/outputs_KT4")
)
EXPECTED_K_VALUES = (4, 5, 6, 7, 8)
EXPECTED_STATES = tuple(range(6))
MIN_CLUSTER_REPLICATES = 1000
MIN_NULL_REPLICATES = 100
ATOL = 1e-10
STATEWISE_FWER_ALPHA = 0.05


class SourceRegistry:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self.names: set[str] = set()

    def add(
        self,
        name: str,
        path: Path,
        rows: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> None:
        if name in self.names:
            return
        resolved = path.resolve()
        self.records.append(
            {
                "source": name,
                "path": str(resolved),
                "sha256": sha256_file(resolved),
                "bytes": int(resolved.stat().st_size),
                "rows_loaded": rows,
                "columns_loaded": ",".join(columns or []),
            }
        )
        self.names.add(name)


class GateRegistry:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(self, gate: str, passed: bool, detail: Any, hard: bool = True) -> None:
        self.rows.append(
            {
                "gate": str(gate),
                "passed": bool(passed),
                "hard_gate": bool(hard),
                "detail": json.dumps(json_safe(detail), ensure_ascii=False, sort_keys=True),
            }
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def failed_hard(self) -> pd.DataFrame:
        frame = self.frame()
        if frame.empty:
            return frame
        return frame[frame["hard_gate"].astype(bool) & ~frame["passed"].astype(bool)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the formal two-table kinetic robustness numerical report."
    )
    parser.add_argument(
        "--partition-root",
        type=Path,
        default=DEFAULT_OUTPUT_BASE / "stage1_kinetic_robustness" / "partition_cluster",
    )
    parser.add_argument(
        "--recursive-null-root",
        type=Path,
        default=DEFAULT_OUTPUT_BASE / "stage1_kinetic_robustness" / "recursive_null",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_BASE / "stage1_kinetic_robustness" / "summary",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
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
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int)):
        return obj
    return str(obj)


def save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)


def load_json(path: Path, name: str, registry: SourceRegistry) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON source: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    registry.add(name, path)
    return payload


def table_path(base: Path) -> Path:
    if base.exists() and base.is_file():
        return base
    for suffix in (".parquet", ".csv.gz", ".csv"):
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find table for {base}")


def read_table(path_or_base: Path, name: str, registry: SourceRegistry) -> pd.DataFrame:
    path = table_path(path_or_base)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, low_memory=False)
    registry.add(name, path, len(frame), list(frame.columns))
    return frame


def output_path(
    manifest: Mapping[str, Any],
    key: str,
    fallback: Path,
) -> Path:
    raw = str(manifest.get("outputs", {}).get(key, "") or "")
    if raw:
        candidate = Path(raw)
        if candidate.exists():
            return candidate
    return table_path(fallback)


def optional_output_path(
    manifest: Mapping[str, Any],
    key: str,
    fallback: Path,
) -> Optional[Path]:
    raw = str(manifest.get("outputs", {}).get(key, "") or "").strip()
    candidates = [Path(raw)] if raw else []
    candidates.append(fallback)
    for candidate in candidates:
        try:
            return table_path(candidate)
        except FileNotFoundError:
            continue
    return None


def source_file_path(manifest: Mapping[str, Any], key: str) -> Path:
    payload = manifest.get("source_files", {}).get(key, {})
    raw = str(payload.get("path", "") or "") if isinstance(payload, Mapping) else ""
    if not raw:
        raise RuntimeError(f"Partition manifest does not identify source file {key!r}.")
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"Source file does not exist: {path}")
    return path


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing columns: {missing}")


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        try:
            return bool(int(value)) if np.isfinite(value) else False
        except Exception:
            return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def bool_series(series: pd.Series) -> pd.Series:
    return series.map(as_bool)


def numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def scalar_close(first: Any, second: Any, atol: float = ATOL) -> bool:
    try:
        a = float(first)
        b = float(second)
    except Exception:
        return False
    if not np.isfinite(a) and not np.isfinite(b):
        return True
    return bool(np.isclose(a, b, atol=atol, rtol=0.0, equal_nan=True))


def quantile(values: np.ndarray, probability: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, probability)) if finite.size else np.nan


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite = np.where(np.isfinite(values))[0]
    if finite.size == 0:
        return adjusted
    order = finite[np.argsort(values[finite])]
    ranked = values[order]
    corrected = ranked * finite.size / np.arange(1, finite.size + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    adjusted[order] = np.clip(corrected, 0.0, 1.0)
    return adjusted


def normal_survival(z_score: float) -> float:
    if not np.isfinite(z_score):
        return np.nan
    return float(0.5 * math.erfc(float(z_score) / math.sqrt(2.0)))


def recompute_statewise_maxT(
    statewise: pd.DataFrame,
    statewise_replicates: pd.DataFrame,
    alpha: float = STATEWISE_FWER_ALPHA,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    observed = statewise[["macrostate", "observed_tail_excess_fixed10"]].copy()
    observed["macrostate"] = pd.to_numeric(
        observed["macrostate"], errors="raise"
    ).astype(int)
    observed["observed_tail_excess_fixed10"] = pd.to_numeric(
        observed["observed_tail_excess_fixed10"], errors="raise"
    )
    observed = observed.sort_values("macrostate", kind="mergesort")
    states = observed["macrostate"].to_numpy(dtype=int)
    replicates = statewise_replicates[
        ["replicate", "macrostate", "tail_excess_fixed10"]
    ].copy()
    replicates["replicate"] = pd.to_numeric(
        replicates["replicate"], errors="raise"
    ).astype(int)
    replicates["macrostate"] = pd.to_numeric(
        replicates["macrostate"], errors="raise"
    ).astype(int)
    replicates["tail_excess_fixed10"] = pd.to_numeric(
        replicates["tail_excess_fixed10"], errors="raise"
    )
    matrix_frame = replicates.pivot(
        index="replicate", columns="macrostate", values="tail_excess_fixed10"
    ).sort_index()
    matrix_frame = matrix_frame.reindex(columns=states)
    if matrix_frame.isna().any().any():
        raise RuntimeError("Statewise maxT audit requires a complete replicate-by-state matrix.")
    null_values = matrix_frame.to_numpy(dtype=float)
    observed_values = observed["observed_tail_excess_fixed10"].to_numpy(dtype=float)
    if not np.isfinite(null_values).all() or not np.isfinite(observed_values).all():
        raise RuntimeError("Statewise maxT audit requires finite observed and null values.")
    replicate_count = int(null_values.shape[0])
    if replicate_count < 3:
        raise RuntimeError("Statewise maxT audit has insufficient recursive replicates.")
    null_mean = np.mean(null_values, axis=0)
    null_sd = np.std(null_values, axis=0, ddof=1)
    if np.any(~np.isfinite(null_sd)) or np.any(null_sd <= 0.0):
        raise RuntimeError("Statewise maxT audit requires positive null variance in every state.")
    observed_standardized = (observed_values - null_mean) / null_sd
    total = np.sum(null_values, axis=0)
    total_squared = np.sum(null_values * null_values, axis=0)
    leave_count = replicate_count - 1
    leave_sum = total[None, :] - null_values
    leave_mean = leave_sum / float(leave_count)
    leave_squared = total_squared[None, :] - null_values * null_values
    leave_sse = leave_squared - (leave_sum * leave_sum) / float(leave_count)
    leave_variance = np.maximum(leave_sse, 0.0) / float(leave_count - 1)
    leave_sd = np.sqrt(leave_variance)
    if np.any(~np.isfinite(leave_sd)) or np.any(leave_sd <= 0.0):
        raise RuntimeError("Leave-one-out maxT standardization is undefined in at least one state.")
    null_standardized = (null_values - leave_mean) / leave_sd
    null_maximum = np.max(null_standardized, axis=1)
    observed_maximum = float(np.max(observed_standardized))
    global_p = float(
        (1 + np.sum(null_maximum >= observed_maximum)) / (replicate_count + 1)
    )
    raw_p = (
        1 + np.sum(null_values >= observed_values[None, :], axis=0)
    ) / float(replicate_count + 1)
    adjusted_p = (
        1 + np.sum(null_maximum[:, None] >= observed_standardized[None, :], axis=0)
    ) / float(replicate_count + 1)
    positive = (
        (observed_values > 0.0)
        & (observed_standardized > 0.0)
        & (adjusted_p < float(alpha))
    )
    state_rows: List[Dict[str, Any]] = []
    for index, state in enumerate(states):
        state_rows.append(
            {
                "macrostate": int(state),
                "tail_excess_fixed10_null_mean": float(null_mean[index]),
                "tail_excess_fixed10_null_sd": float(null_sd[index]),
                "tail_excess_fixed10_standardized_excess": float(
                    observed_standardized[index]
                ),
                "tail_excess_fixed10_raw_monte_carlo_p": float(raw_p[index]),
                "tail_excess_fixed10_maxT_fwer_p": float(adjusted_p[index]),
                "tail_excess_fixed10_maxT_fwer_positive": bool(positive[index]),
            }
        )
    supported_states = [int(state) for state, flag in zip(states, positive) if flag]
    if len(supported_states) >= 2:
        interpretation = (
            "At least two frozen states show positive fixed-10 tail excess beyond the "
            "conditional recursive surrogate after single-step maxT family-wise control."
        )
    elif len(supported_states) == 1:
        interpretation = (
            "Construction-aware fixed-10 tail excess is localized to one frozen state; "
            "a multi-state metastable-like interpretation is not supported by this endpoint alone."
        )
    else:
        interpretation = (
            "No frozen state shows positive fixed-10 tail excess beyond the conditional recursive "
            "surrogate after family-wise control."
        )
    familywise = {
        "metric": "statewise_tail_excess_fixed10_studentized_maxT",
        "direction_supporting_observed_excess": "greater",
        "analysis_status": "reviewer-motivated post hoc family-wise endpoint",
        "alpha": float(alpha),
        "observed_maxT": observed_maximum,
        "null_mean": float(np.mean(null_maximum)),
        "null_sd": float(np.std(null_maximum, ddof=1)),
        "null_2p5": float(np.quantile(null_maximum, 0.025)),
        "null_median": float(np.median(null_maximum)),
        "null_97p5": float(np.quantile(null_maximum, 0.975)),
        "monte_carlo_p": global_p,
        "finite_replicates": int(replicate_count),
        "fwer_positive_state_count": int(len(supported_states)),
        "fwer_positive_states": ",".join(f"S{state}" for state in supported_states),
        "multi_state_support": bool(len(supported_states) >= 2),
        "interpretation": interpretation,
    }
    return familywise, pd.DataFrame(state_rows)


def format_number(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return "–"
    if not np.isfinite(number):
        return "–"
    if number != 0.0 and abs(number) < 1e-3:
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def format_integer(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "–"
    return str(int(round(number))) if np.isfinite(number) else "–"


def format_interval(point: Any, lower: Any, upper: Any, digits: int = 4) -> str:
    return (
        f"{format_number(point, digits)} "
        f"[{format_number(lower, digits)}, {format_number(upper, digits)}]"
    )


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(
                str(value).replace("|", "\\|").replace("\n", " ") for value in row
            )
            + " |"
        )
    return "\n".join(lines)


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def validate_manifests(
    partition: Mapping[str, Any],
    recursive: Mapping[str, Any],
    gates: GateRegistry,
) -> Tuple[int, int]:
    partition_quality = dict(partition.get("quality_gates", {}))
    recursive_quality = dict(recursive.get("quality_gates", {}))
    partition_required = (
        "formal_k6_fit_reproduced",
        "cluster_unit_weight_point_reproduced",
        "bootstrap_replicates_at_least_1000",
        "cluster_bootstrap_finite_support_at_least_95_percent",
        "full_split_used",
    )
    recursive_required = (
        "full_split_used",
        "matching_coverage_exactly_reproduced",
        "observed_innovation_recursion_reproduces_formal_assignments_and_kinetics",
        "all_recursive_edges_are_adjacent",
        "target_coordinates_reproduced_within_2e_6",
        "fixed10_all_states_meet_at_risk_threshold_in_observed_and_null",
        "no_invalid_denominators",
        "no_non_numerical_clipping",
        "first_permutation_fixed_point_free",
        "replicates_at_least_100",
    )
    for key in partition_required:
        gates.add(f"partition_manifest:{key}", partition_quality.get(key) is True, partition_quality.get(key))
    for key in recursive_required:
        gates.add(f"recursive_manifest:{key}", recursive_quality.get(key) is True, recursive_quality.get(key))
    candidate_values = tuple(
        sorted(int(value) for value in partition.get("partition_sensitivity", {}).get("candidate_values", []))
    )
    gates.add(
        "partition_candidate_values",
        candidate_values == EXPECTED_K_VALUES,
        {"observed": candidate_values, "expected": EXPECTED_K_VALUES},
    )
    gates.add(
        "formal_partition_k6",
        int(partition.get("formal_partition", {}).get("k", -1)) == 6,
        partition.get("formal_partition", {}),
    )
    gates.add(
        "partition_validation_not_used_for_fit_or_selection",
        partition.get("partition_sensitivity", {}).get("validation_used_for_fit_or_selection") is False,
        partition.get("partition_sensitivity", {}).get("validation_used_for_fit_or_selection"),
    )
    gates.add(
        "partition_B_confirm_not_read",
        partition.get("partition_sensitivity", {}).get("B_confirm_read") is False,
        partition.get("partition_sensitivity", {}).get("B_confirm_read"),
    )
    gates.add(
        "recursive_A_val_only",
        recursive.get("analysis_split") == "A_val" and recursive.get("B_confirm_read") is False,
        {"analysis_split": recursive.get("analysis_split"), "B_confirm_read": recursive.get("B_confirm_read")},
    )
    gates.add(
        "recursive_fixed_partition_k6",
        int(recursive.get("fixed_partition", {}).get("k", -1)) == 6
        and recursive.get("fixed_partition", {}).get("kmeans_refit") is False
        and recursive.get("fixed_partition", {}).get("partition_selected") is False,
        recursive.get("fixed_partition", {}),
    )
    gates.add(
        "recursive_primary_endpoint",
        recursive.get("primary_endpoint", {}).get("metric")
        == "aggregate_mean_log_rmst_lift_fixed10",
        recursive.get("primary_endpoint", {}),
    )
    statewise_endpoint = dict(recursive.get("statewise_familywise_endpoint", {}))
    gates.add(
        "recursive_statewise_familywise_endpoint",
        not statewise_endpoint
        or (
            statewise_endpoint.get("metric") == "tail_excess_fixed10"
            and "maxT" in str(statewise_endpoint.get("global_test", ""))
            and scalar_close(statewise_endpoint.get("alpha"), STATEWISE_FWER_ALPHA)
        ),
        statewise_endpoint or {"status": "recomputed from archived statewise replicates"},
    )
    cluster_replicates = int(partition.get("learner_cluster_bootstrap", {}).get("replicates", 0))
    null_replicates = int(recursive.get("replicates", 0))
    gates.add(
        "cluster_replicate_count",
        cluster_replicates >= MIN_CLUSTER_REPLICATES,
        cluster_replicates,
    )
    gates.add(
        "recursive_null_replicate_count",
        null_replicates >= MIN_NULL_REPLICATES,
        null_replicates,
    )
    cluster_contract = dict(partition.get("learner_cluster_bootstrap", {}))
    gates.add(
        "cluster_joint_transition_residence_resampling",
        cluster_contract.get("resampling_unit") == "learner"
        and cluster_contract.get("same_learner_multiplicity_for_transitions_and_all_residence_runs") is True
        and cluster_contract.get("Pii_and_Kaplan_Meier_recomputed_jointly") is True
        and cluster_contract.get("state_specific_rmst_horizons_frozen_from_formal_analysis") is True,
        cluster_contract,
    )
    partition_sha = str(partition.get("stage1_script_sha256", "") or "")
    recursive_sha = str(recursive.get("stage1_script_sha256", "") or "")
    gates.add(
        "common_formal_stage1_implementation",
        bool(partition_sha) and partition_sha == recursive_sha,
        {"partition": partition_sha, "recursive": recursive_sha},
    )
    duplication = dict(partition.get("duplication_guardrails", {}))
    gates.add(
        "partition_duplication_guardrails",
        all(value is False for value in duplication.values()),
        duplication,
    )
    relationship = dict(recursive.get("relationship_to_existing_experiments", {}))
    gates.add(
        "recursive_no_field_or_downstream_rerun",
        relationship.get("existing_construction_matched_field_null_rerun") is False
        and relationship.get("new_field_drift_divergence_or_core_inference") is False
        and relationship.get("coordinate_or_grid_sensitivity_rerun") is False
        and relationship.get("learner_multiplier_sensitivity_rerun") is False
        and relationship.get("downstream_model_evaluation") is False,
        relationship,
    )
    return cluster_replicates, null_replicates


def validate_recursive_outputs(
    summary: pd.DataFrame,
    replicates: pd.DataFrame,
    statewise: pd.DataFrame,
    statewise_replicates: pd.DataFrame,
    familywise_summary: pd.DataFrame,
    manifest_replicates: int,
    gates: GateRegistry,
) -> None:
    summary_columns = (
        "metric",
        "direction_supporting_observed_excess",
        "primary_endpoint",
        "observed",
        "null_mean",
        "null_sd",
        "null_2p5",
        "null_median",
        "null_97p5",
        "monte_carlo_p",
        "inferential_test_performed",
        "finite_replicates",
        "interpretation",
    )
    replicate_columns = (
        "replicate",
        "donor_seed",
        "transition_count",
        "residence_run_count",
        "right_censored_run_count",
        "aggregate_mean_log_rmst_lift_fixed10",
        "diagonal_margin",
        "mean_self_transition",
        "diagonal_dominant_rows",
        "minimum_reference_at_risk",
        "coordinate_bound_excess_before_numerical_clipping",
    )
    statewise_columns = (
        "macrostate",
        "observed_self_transition",
        "observed_rmst_lift_fixed10",
        "observed_tail_excess_fixed10",
        "formal_state_specific_rmst_lift",
        "null_self_transition_2p5",
        "null_self_transition_97p5",
        "null_rmst_lift_fixed10_2p5",
        "null_rmst_lift_fixed10_97p5",
        "null_tail_excess_fixed10_2p5",
        "null_tail_excess_fixed10_97p5",
        "tail_excess_fixed10_null_mean",
        "tail_excess_fixed10_null_sd",
        "tail_excess_fixed10_standardized_excess",
        "tail_excess_fixed10_raw_monte_carlo_p",
        "tail_excess_fixed10_maxT_fwer_p",
        "tail_excess_fixed10_maxT_fwer_positive",
    )
    familywise_columns = (
        "metric",
        "direction_supporting_observed_excess",
        "analysis_status",
        "alpha",
        "observed_maxT",
        "null_mean",
        "null_sd",
        "null_2p5",
        "null_median",
        "null_97p5",
        "monte_carlo_p",
        "finite_replicates",
        "fwer_positive_state_count",
        "fwer_positive_states",
        "multi_state_support",
        "interpretation",
    )
    statewise_replicate_columns = (
        "replicate",
        "macrostate",
        "self_transition",
        "rmst_lift_fixed10",
        "tail_excess_fixed10",
        "reference_at_risk",
        "run_count",
    )
    require_columns(summary, summary_columns, "recursive null summary")
    require_columns(replicates, replicate_columns, "recursive null replicates")
    require_columns(statewise, statewise_columns, "recursive null statewise summary")
    require_columns(
        statewise_replicates,
        statewise_replicate_columns,
        "recursive null statewise replicates",
    )
    require_columns(
        familywise_summary,
        familywise_columns,
        "recursive statewise family-wise summary",
    )
    metrics = {
        "aggregate_mean_log_rmst_lift_fixed10",
        "diagonal_margin",
        "mean_self_transition",
        "diagonal_dominant_rows",
    }
    gates.add("recursive_summary_metric_set", set(summary["metric"].astype(str)) == metrics, sorted(summary["metric"].astype(str)))
    primary = summary[bool_series(summary["primary_endpoint"])]
    gates.add(
        "recursive_exactly_one_primary_endpoint",
        len(primary) == 1
        and str(primary.iloc[0]["metric"]) == "aggregate_mean_log_rmst_lift_fixed10"
        if len(primary) else False,
        primary["metric"].astype(str).tolist(),
    )
    replicate_ids = pd.to_numeric(replicates["replicate"], errors="raise").astype(int)
    gates.add(
        "recursive_replicate_rows",
        len(replicates) == manifest_replicates
        and replicate_ids.nunique() == manifest_replicates
        and set(replicate_ids.tolist()) == set(range(manifest_replicates)),
        {"rows": len(replicates), "unique": int(replicate_ids.nunique())},
    )
    state_ids = pd.to_numeric(statewise["macrostate"], errors="raise").astype(int)
    gates.add(
        "recursive_statewise_states",
        len(statewise) == 6 and tuple(sorted(state_ids.tolist())) == EXPECTED_STATES,
        state_ids.tolist(),
    )
    pair_frame = statewise_replicates[["replicate", "macrostate"]].apply(
        pd.to_numeric, errors="raise"
    ).astype(int)
    gates.add(
        "recursive_statewise_replicate_rows",
        len(statewise_replicates) == manifest_replicates * 6
        and not pair_frame.duplicated().any()
        and set(pair_frame["replicate"].tolist()) == set(range(manifest_replicates))
        and set(pair_frame["macrostate"].tolist()) == set(EXPECTED_STATES),
        {"rows": len(statewise_replicates), "expected": manifest_replicates * 6},
    )
    summary_index = summary.set_index("metric")
    recomputation_checks: List[bool] = []
    for metric in metrics:
        values = numeric(replicates, metric)
        finite = values[np.isfinite(values)]
        archived = summary_index.loc[metric]
        expected = {
            "null_mean": float(np.mean(finite)),
            "null_sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
            "null_2p5": quantile(finite, 0.025),
            "null_median": quantile(finite, 0.5),
            "null_97p5": quantile(finite, 0.975),
            "finite_replicates": int(finite.size),
        }
        checks = [
            scalar_close(archived[key], value, 1e-10)
            for key, value in expected.items()
        ]
        if as_bool(archived["primary_endpoint"]):
            observed = float(archived["observed"])
            expected_p = float((1 + np.sum(finite >= observed)) / (finite.size + 1))
            checks.append(scalar_close(archived["monte_carlo_p"], expected_p, 1e-12))
        else:
            checks.append(not np.isfinite(float(archived["monte_carlo_p"])))
        recomputation_checks.extend(checks)
    gates.add(
        "recursive_summary_recomputed_from_replicates",
        all(recomputation_checks),
        {"checks": len(recomputation_checks), "passed": int(sum(recomputation_checks))},
    )
    statewise_index = statewise.set_index("macrostate")
    statewise_checks: List[bool] = []
    for state in EXPECTED_STATES:
        subset = statewise_replicates[
            pd.to_numeric(statewise_replicates["macrostate"], errors="coerce") == state
        ]
        row = statewise_index.loc[state]
        definitions = (
            ("self_transition", "null_self_transition_2p5", "null_self_transition_97p5"),
            ("rmst_lift_fixed10", "null_rmst_lift_fixed10_2p5", "null_rmst_lift_fixed10_97p5"),
            ("tail_excess_fixed10", "null_tail_excess_fixed10_2p5", "null_tail_excess_fixed10_97p5"),
        )
        for source, lower, upper in definitions:
            values = numeric(subset, source)
            statewise_checks.append(scalar_close(row[lower], quantile(values, 0.025), 1e-10))
            statewise_checks.append(scalar_close(row[upper], quantile(values, 0.975), 1e-10))
    primary_observed = float(
        summary_index.loc["aggregate_mean_log_rmst_lift_fixed10", "observed"]
    )
    observed_lifts = numeric(statewise, "observed_rmst_lift_fixed10")
    statewise_checks.append(
        scalar_close(primary_observed, float(np.mean(np.log(observed_lifts))), 1e-10)
    )
    statewise_checks.append(
        scalar_close(
            summary_index.loc["mean_self_transition", "observed"],
            float(np.mean(numeric(statewise, "observed_self_transition"))),
            1e-10,
        )
    )
    gates.add(
        "recursive_statewise_summary_recomputed",
        all(statewise_checks),
        {"checks": len(statewise_checks), "passed": int(sum(statewise_checks))},
    )
    expected_familywise, expected_statewise = recompute_statewise_maxT(
        statewise, statewise_replicates, STATEWISE_FWER_ALPHA
    )
    familywise_checks: List[bool] = [len(familywise_summary) == 1]
    if len(familywise_summary) == 1:
        archived_familywise = familywise_summary.iloc[0]
        numeric_fields = (
            "alpha",
            "observed_maxT",
            "null_mean",
            "null_sd",
            "null_2p5",
            "null_median",
            "null_97p5",
            "monte_carlo_p",
            "finite_replicates",
            "fwer_positive_state_count",
        )
        familywise_checks.extend(
            scalar_close(archived_familywise[field], expected_familywise[field], 1e-10)
            for field in numeric_fields
        )
        familywise_checks.extend(
            [
                str(archived_familywise["metric"]) == str(expected_familywise["metric"]),
                str(archived_familywise["direction_supporting_observed_excess"])
                == str(expected_familywise["direction_supporting_observed_excess"]),
                (
                    ""
                    if pd.isna(archived_familywise["fwer_positive_states"])
                    else str(archived_familywise["fwer_positive_states"])
                )
                == str(expected_familywise["fwer_positive_states"]),
                as_bool(archived_familywise["multi_state_support"])
                == bool(expected_familywise["multi_state_support"]),
            ]
        )
    expected_statewise_index = expected_statewise.set_index("macrostate")
    for state in EXPECTED_STATES:
        archived = statewise_index.loc[state]
        expected = expected_statewise_index.loc[state]
        for field in (
            "tail_excess_fixed10_null_mean",
            "tail_excess_fixed10_null_sd",
            "tail_excess_fixed10_standardized_excess",
            "tail_excess_fixed10_raw_monte_carlo_p",
            "tail_excess_fixed10_maxT_fwer_p",
        ):
            familywise_checks.append(
                scalar_close(archived[field], expected[field], 1e-10)
            )
        familywise_checks.append(
            as_bool(archived["tail_excess_fixed10_maxT_fwer_positive"])
            == as_bool(expected["tail_excess_fixed10_maxT_fwer_positive"])
        )
    gates.add(
        "recursive_statewise_maxT_recomputed",
        all(familywise_checks),
        {"checks": len(familywise_checks), "passed": int(sum(familywise_checks))},
    )
    gates.add(
        "recursive_replicate_count_integrity",
        bool((pd.to_numeric(replicates["transition_count"], errors="coerce") > 0).all())
        and bool((pd.to_numeric(replicates["residence_run_count"], errors="coerce") > 0).all())
        and bool(
            (
                pd.to_numeric(replicates["right_censored_run_count"], errors="coerce")
                <= pd.to_numeric(replicates["residence_run_count"], errors="coerce")
            ).all()
        ),
        {
            "min_transition_count": float(pd.to_numeric(replicates["transition_count"], errors="coerce").min()),
            "min_residence_run_count": float(pd.to_numeric(replicates["residence_run_count"], errors="coerce").min()),
        },
    )


def validate_partition_outputs(
    summary: pd.DataFrame,
    statewise: pd.DataFrame,
    partition_manifest: Mapping[str, Any],
    partition_root: Path,
    gates: GateRegistry,
    registry: SourceRegistry,
) -> None:
    summary_columns = (
        "k",
        "role",
        "A_val_minimum_user_balanced_state_occupancy",
        "A_train_transition_count",
        "A_val_transition_count",
        "A_val_residence_run_count",
        "A_val_right_censoring_fraction",
        "A_val_diagonal_dominant_rows",
        "A_val_mean_self_transition",
        "A_val_min_self_transition",
        "A_val_max_self_transition",
        "A_val_states_with_rmst_lift_above_one",
        "A_val_states_meeting_fixed10_at_risk_threshold",
        "A_val_states_with_positive_fixed10_tail_excess",
        "A_train_A_val_transition_mean_row_tv",
        "A_train_A_val_transition_max_row_tv",
        "A_train_A_val_rmst_lift_mean_abs_log_difference",
    )
    statewise_columns = (
        "k",
        "split",
        "macrostate",
        "center_M",
        "center_Psi",
        "user_balanced_occupancy",
        "self_transition",
        "diagonal_dominant",
        "run_count",
        "right_censoring_fraction",
        "rmst_tau",
        "restricted_mean_residence_lift",
        "reference_at_risk",
        "fixed_reference_reliable",
        "tail_excess_at_reference",
        "tail_ratio_at_reference",
        "greenwood_q_formal_K6_only",
        "formal_inference_performed",
    )
    require_columns(summary, summary_columns, "partition robustness summary")
    require_columns(statewise, statewise_columns, "partition statewise kinetics")
    k_values = tuple(sorted(pd.to_numeric(summary["k"], errors="raise").astype(int).tolist()))
    gates.add(
        "partition_summary_K_rows",
        k_values == EXPECTED_K_VALUES and len(summary) == len(EXPECTED_K_VALUES),
        k_values,
    )
    role_check = all(
        (int(row.k) == 6 and str(row.role) == "formal")
        or (int(row.k) != 6 and str(row.role) == "bounded resolution sensitivity")
        for row in summary.itertuples(index=False)
    )
    gates.add("partition_role_contract", role_check, summary[["k", "role"]].to_dict("records"))
    aggregate_checks: List[bool] = []
    for k in EXPECTED_K_VALUES:
        summary_row = summary[pd.to_numeric(summary["k"], errors="coerce") == k].iloc[0]
        train_state = statewise[
            (pd.to_numeric(statewise["k"], errors="coerce") == k)
            & (statewise["split"].astype(str) == "A_train")
        ].sort_values("macrostate")
        val_state = statewise[
            (pd.to_numeric(statewise["k"], errors="coerce") == k)
            & (statewise["split"].astype(str) == "A_val")
        ].sort_values("macrostate")
        aggregate_checks.extend(
            [
                len(train_state) == k,
                len(val_state) == k,
                tuple(pd.to_numeric(train_state["macrostate"], errors="raise").astype(int)) == tuple(range(k)),
                tuple(pd.to_numeric(val_state["macrostate"], errors="raise").astype(int)) == tuple(range(k)),
                scalar_close(
                    summary_row["A_val_minimum_user_balanced_state_occupancy"],
                    pd.to_numeric(val_state["user_balanced_occupancy"], errors="coerce").min(),
                ),
                scalar_close(
                    summary_row["A_val_residence_run_count"],
                    pd.to_numeric(val_state["run_count"], errors="coerce").sum(),
                ),
                scalar_close(
                    summary_row["A_val_diagonal_dominant_rows"],
                    bool_series(val_state["diagonal_dominant"]).sum(),
                ),
                scalar_close(
                    summary_row["A_val_mean_self_transition"],
                    pd.to_numeric(val_state["self_transition"], errors="coerce").mean(),
                ),
                scalar_close(
                    summary_row["A_val_min_self_transition"],
                    pd.to_numeric(val_state["self_transition"], errors="coerce").min(),
                ),
                scalar_close(
                    summary_row["A_val_max_self_transition"],
                    pd.to_numeric(val_state["self_transition"], errors="coerce").max(),
                ),
                scalar_close(
                    summary_row["A_val_states_with_rmst_lift_above_one"],
                    np.sum(pd.to_numeric(val_state["restricted_mean_residence_lift"], errors="coerce") > 1.0),
                ),
                scalar_close(
                    summary_row["A_val_states_meeting_fixed10_at_risk_threshold"],
                    bool_series(val_state["fixed_reference_reliable"]).sum(),
                ),
                scalar_close(
                    summary_row["A_val_states_with_positive_fixed10_tail_excess"],
                    np.sum(
                        bool_series(val_state["fixed_reference_reliable"]).to_numpy()
                        & (pd.to_numeric(val_state["tail_excess_at_reference"], errors="coerce").to_numpy() > 0)
                    ),
                ),
            ]
        )
        weights = np.maximum(pd.to_numeric(val_state["run_count"], errors="coerce").to_numpy(dtype=float), 0.0)
        censor = pd.to_numeric(val_state["right_censoring_fraction"], errors="coerce").to_numpy(dtype=float)
        weighted_censor = float(np.average(censor, weights=weights)) if np.sum(weights) > 0 else np.nan
        aggregate_checks.append(
            scalar_close(summary_row["A_val_right_censoring_fraction"], weighted_censor)
        )
        train_lift = pd.to_numeric(train_state["restricted_mean_residence_lift"], errors="coerce").to_numpy(dtype=float)
        val_lift = pd.to_numeric(val_state["restricted_mean_residence_lift"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(train_lift) & np.isfinite(val_lift) & (train_lift > 0) & (val_lift > 0)
        mean_log_difference = float(np.mean(np.abs(np.log(val_lift[valid] / train_lift[valid])))) if np.any(valid) else np.nan
        aggregate_checks.append(
            scalar_close(
                summary_row["A_train_A_val_rmst_lift_mean_abs_log_difference"],
                mean_log_difference,
            )
        )
        output_paths = partition_manifest.get("outputs", {})
        def declared_path(key: str) -> Optional[Path]:
            raw = str(output_paths.get(key, "") or "").strip()
            if not raw:
                return None
            candidate = Path(raw)
            return candidate if candidate.is_file() else None
        train_counts_path = declared_path(f"K{k}_A_train_transition_counts")
        val_counts_path = declared_path(f"K{k}_A_val_transition_counts")
        train_matrix_path = declared_path(f"K{k}_A_train_transition_matrix")
        val_matrix_path = declared_path(f"K{k}_A_val_transition_matrix")
        if k == 6:
            stage1_root = Path(str(partition_manifest.get("stage1_root", "") or ""))
            if not stage1_root.exists():
                raise FileNotFoundError(f"Partition manifest Stage-1 root does not exist: {stage1_root}")
            formal_root = stage1_root / "dynamics" / "fixed_k6_mesostates"
            if train_counts_path is None:
                train_counts_path = table_path(formal_root / "A_train_fixed_k6_transition_counts")
            if val_counts_path is None:
                val_counts_path = table_path(formal_root / "A_val_fixed_k6_transition_counts")
            if train_matrix_path is None:
                train_matrix_path = table_path(formal_root / "A_train_fixed_k6_transition_matrix")
            if val_matrix_path is None:
                val_matrix_path = table_path(formal_root / "A_val_fixed_k6_transition_matrix")
        else:
            if train_counts_path is None:
                train_counts_path = table_path(partition_root / "tables" / f"K{k}_A_train_transition_counts")
            if val_counts_path is None:
                val_counts_path = table_path(partition_root / "tables" / f"K{k}_A_val_transition_counts")
            if train_matrix_path is None:
                train_matrix_path = table_path(partition_root / "tables" / f"K{k}_A_train_transition_matrix")
            if val_matrix_path is None:
                val_matrix_path = table_path(partition_root / "tables" / f"K{k}_A_val_transition_matrix")
        train_counts = read_table(train_counts_path, f"K{k}_A_train_transition_counts", registry).to_numpy(dtype=float)
        val_counts = read_table(val_counts_path, f"K{k}_A_val_transition_counts", registry).to_numpy(dtype=float)
        train_matrix = read_table(train_matrix_path, f"K{k}_A_train_transition_matrix", registry).to_numpy(dtype=float)
        val_matrix = read_table(val_matrix_path, f"K{k}_A_val_transition_matrix", registry).to_numpy(dtype=float)
        tv = 0.5 * np.sum(np.abs(train_matrix - val_matrix), axis=1)
        aggregate_checks.extend(
            [
                scalar_close(summary_row["A_train_transition_count"], np.sum(train_counts)),
                scalar_close(summary_row["A_val_transition_count"], np.sum(val_counts)),
                scalar_close(summary_row["A_train_A_val_transition_mean_row_tv"], np.mean(tv)),
                scalar_close(summary_row["A_train_A_val_transition_max_row_tv"], np.max(tv)),
            ]
        )
        aggregate_checks.extend(
            [
                scalar_close(pd.to_numeric(train_state["user_balanced_occupancy"], errors="coerce").sum(), 1.0, 1e-8),
                scalar_close(pd.to_numeric(val_state["user_balanced_occupancy"], errors="coerce").sum(), 1.0, 1e-8),
            ]
        )
    gates.add(
        "partition_summary_recomputed",
        all(aggregate_checks),
        {"checks": len(aggregate_checks), "passed": int(sum(aggregate_checks))},
    )


def validate_cluster_outputs(
    summary: pd.DataFrame,
    replicates: pd.DataFrame,
    formal_summary: pd.DataFrame,
    manifest_replicates: int,
    gates: GateRegistry,
) -> None:
    summary_columns = (
        "macrostate",
        "self_transition_point",
        "self_transition_ci_2p5",
        "self_transition_ci_97p5",
        "diagonal_dominance_bootstrap_probability",
        "restricted_mean_residence_lift_point",
        "restricted_mean_residence_lift_ci_2p5",
        "restricted_mean_residence_lift_ci_97p5",
        "rmst_lift_lower_bound_above_one",
        "tail_excess_point",
        "tail_excess_ci_2p5",
        "tail_excess_ci_97p5",
        "tail_excess_cluster_bootstrap_se",
        "tail_excess_cluster_z",
        "tail_excess_cluster_one_sided_p",
        "greenwood_one_sided_p_formal",
        "greenwood_bh_q_formal",
        "finite_self_transition_bootstrap_replicates",
        "finite_rmst_bootstrap_replicates",
        "finite_tail_bootstrap_replicates",
        "tail_excess_cluster_bh_q",
        "tail_excess_cluster_bh_positive",
    )
    replicate_columns = (
        "replicate",
        "macrostate",
        "self_transition",
        "diagonal_dominant",
        "restricted_mean_residence_lift",
        "tail_excess_at_reference",
    )
    formal_columns = (
        "macrostate",
        "n_runs",
        "n_completed_exits",
        "n_right_censored",
        "right_censoring_fraction",
        "self_transition",
        "rmst_tau",
        "restricted_mean_residence_lift",
        "reference_length",
        "reference_at_risk",
        "observed_tail_probability_at_reference",
        "geometric_tail_probability_at_reference",
        "tail_excess_pvalue_greenwood",
        "tail_excess_qvalue_bh",
    )
    require_columns(summary, summary_columns, "cluster-bootstrap statewise summary")
    require_columns(replicates, replicate_columns, "cluster-bootstrap replicates")
    require_columns(formal_summary, formal_columns, "formal K=6 residence summary")
    state_ids = tuple(sorted(pd.to_numeric(summary["macrostate"], errors="raise").astype(int).tolist()))
    gates.add("cluster_summary_states", len(summary) == 6 and state_ids == EXPECTED_STATES, state_ids)
    pairs = replicates[["replicate", "macrostate"]].apply(pd.to_numeric, errors="raise").astype(int)
    gates.add(
        "cluster_replicate_rows",
        len(replicates) == manifest_replicates * 6
        and not pairs.duplicated().any()
        and set(pairs["replicate"].tolist()) == set(range(manifest_replicates))
        and set(pairs["macrostate"].tolist()) == set(EXPECTED_STATES),
        {"rows": len(replicates), "expected": manifest_replicates * 6},
    )
    formal_index = formal_summary.set_index("macrostate")
    summary_index = summary.set_index("macrostate")
    pvalues: List[float] = []
    checks: List[bool] = []
    for state in EXPECTED_STATES:
        subset = replicates[pd.to_numeric(replicates["macrostate"], errors="coerce") == state]
        row = summary_index.loc[state]
        formal = formal_index.loc[state]
        pii = numeric(subset, "self_transition")
        lift = numeric(subset, "restricted_mean_residence_lift")
        tail = numeric(subset, "tail_excess_at_reference")
        finite_pii = pii[np.isfinite(pii)]
        finite_lift = lift[np.isfinite(lift)]
        finite_tail = tail[np.isfinite(tail)]
        tail_se = float(np.std(finite_tail, ddof=1)) if finite_tail.size > 1 else np.nan
        tail_point = float(
            formal["observed_tail_probability_at_reference"]
            - formal["geometric_tail_probability_at_reference"]
        )
        z_score = tail_point / tail_se if np.isfinite(tail_se) and tail_se > 0 else np.nan
        pvalue = normal_survival(z_score)
        pvalues.append(pvalue)
        checks.extend(
            [
                scalar_close(row["self_transition_point"], formal["self_transition"]),
                scalar_close(row["self_transition_ci_2p5"], quantile(finite_pii, 0.025)),
                scalar_close(row["self_transition_ci_97p5"], quantile(finite_pii, 0.975)),
                scalar_close(
                    row["diagonal_dominance_bootstrap_probability"],
                    pd.to_numeric(subset["diagonal_dominant"], errors="coerce").mean(),
                ),
                scalar_close(
                    row["restricted_mean_residence_lift_point"],
                    formal["restricted_mean_residence_lift"],
                ),
                scalar_close(row["restricted_mean_residence_lift_ci_2p5"], quantile(finite_lift, 0.025)),
                scalar_close(row["restricted_mean_residence_lift_ci_97p5"], quantile(finite_lift, 0.975)),
                as_bool(row["rmst_lift_lower_bound_above_one"])
                == bool(quantile(finite_lift, 0.025) > 1.0),
                scalar_close(row["tail_excess_point"], tail_point),
                scalar_close(row["tail_excess_ci_2p5"], quantile(finite_tail, 0.025)),
                scalar_close(row["tail_excess_ci_97p5"], quantile(finite_tail, 0.975)),
                scalar_close(row["tail_excess_cluster_bootstrap_se"], tail_se),
                scalar_close(row["tail_excess_cluster_z"], z_score),
                scalar_close(row["tail_excess_cluster_one_sided_p"], pvalue),
                scalar_close(row["greenwood_one_sided_p_formal"], formal["tail_excess_pvalue_greenwood"]),
                scalar_close(row["greenwood_bh_q_formal"], formal["tail_excess_qvalue_bh"]),
                int(row["finite_self_transition_bootstrap_replicates"]) == int(finite_pii.size),
                int(row["finite_rmst_bootstrap_replicates"]) == int(finite_lift.size),
                int(row["finite_tail_bootstrap_replicates"]) == int(finite_tail.size),
            ]
        )
    qvalues = benjamini_hochberg(pvalues)
    for state, qvalue in zip(EXPECTED_STATES, qvalues):
        row = summary_index.loc[state]
        checks.append(scalar_close(row["tail_excess_cluster_bh_q"], qvalue))
        checks.append(
            as_bool(row["tail_excess_cluster_bh_positive"])
            == bool(np.isfinite(qvalue) and qvalue < 0.05 and float(row["tail_excess_point"]) > 0)
        )
    gates.add(
        "cluster_summary_recomputed_from_replicates",
        all(checks),
        {"checks": len(checks), "passed": int(sum(checks))},
    )
    finite_minimum = int(
        summary[
            [
                "finite_self_transition_bootstrap_replicates",
                "finite_rmst_bootstrap_replicates",
                "finite_tail_bootstrap_replicates",
            ]
        ]
        .apply(pd.to_numeric, errors="coerce")
        .min()
        .min()
    )
    gates.add(
        "cluster_finite_replicates_at_least_95_percent",
        finite_minimum >= math.ceil(0.95 * manifest_replicates),
        {"minimum": finite_minimum, "required": math.ceil(0.95 * manifest_replicates)},
    )


def table1_numeric(
    recursive_summary: pd.DataFrame,
    recursive_statewise: pd.DataFrame,
    recursive_familywise: pd.DataFrame,
    recursive_replicates: pd.DataFrame,
    partition_summary: pd.DataFrame,
    formal_summary: pd.DataFrame,
    recursive_manifest: Mapping[str, Any],
) -> pd.DataFrame:
    columns = [
        "panel",
        "row_type",
        "analysis_unit",
        "status",
        "k",
        "macrostate",
        "observed",
        "null_mean",
        "null_sd",
        "null_2p5",
        "null_median",
        "null_97p5",
        "monte_carlo_p",
        "finite_null_replicates",
        "fwer_positive_state_count",
        "fwer_positive_states",
        "multi_state_support",
        "recursive_panel_rows",
        "recursive_panel_users",
        "recursive_valid_edges",
        "null_minimum_reference_at_risk_min",
        "null_max_coordinate_bound_excess",
        "observed_self_transition",
        "null_self_transition_2p5",
        "null_self_transition_97p5",
        "observed_rmst_lift_fixed10",
        "null_rmst_lift_fixed10_2p5",
        "null_rmst_lift_fixed10_97p5",
        "observed_tail_excess_fixed10",
        "null_tail_excess_fixed10_2p5",
        "null_tail_excess_fixed10_97p5",
        "tail_excess_fixed10_null_mean",
        "tail_excess_fixed10_null_sd",
        "tail_excess_fixed10_standardized_excess",
        "tail_excess_fixed10_raw_monte_carlo_p",
        "tail_excess_fixed10_maxT_fwer_p",
        "tail_excess_fixed10_maxT_fwer_positive",
        "formal_state_specific_rmst_lift",
        "formal_rmst_tau",
        "formal_reference_at_risk",
        "A_val_minimum_user_balanced_state_occupancy",
        "A_train_transition_count",
        "A_val_transition_count",
        "A_val_residence_run_count",
        "A_val_right_censoring_fraction",
        "A_val_diagonal_dominant_rows",
        "A_val_mean_self_transition",
        "A_val_min_self_transition",
        "A_val_max_self_transition",
        "A_val_states_with_rmst_lift_above_one",
        "A_val_states_meeting_fixed10_at_risk_threshold",
        "A_val_states_with_positive_fixed10_tail_excess",
        "A_train_A_val_transition_mean_row_tv",
        "A_train_A_val_transition_max_row_tv",
        "A_train_A_val_rmst_lift_mean_abs_log_difference",
        "interpretation",
    ]
    rows: List[Dict[str, Any]] = []
    labels = {
        "aggregate_mean_log_rmst_lift_fixed10": "Mean log fixed-10 RMST lift",
        "diagonal_margin": "Mean diagonal margin",
        "mean_self_transition": "Mean self-transition probability",
        "diagonal_dominant_rows": "Diagonal-dominant rows",
    }
    for source in recursive_summary.itertuples(index=False):
        row = {column: np.nan for column in columns}
        row.update(
            {
                "panel": "A1. Recursive aggregate null",
                "row_type": "recursive_aggregate",
                "analysis_unit": labels.get(str(source.metric), str(source.metric)),
                "status": "primary" if as_bool(source.primary_endpoint) else "descriptive",
                "k": 6,
                "observed": source.observed,
                "null_mean": source.null_mean,
                "null_sd": source.null_sd,
                "null_2p5": source.null_2p5,
                "null_median": source.null_median,
                "null_97p5": source.null_97p5,
                "monte_carlo_p": source.monte_carlo_p,
                "finite_null_replicates": source.finite_replicates,
                "interpretation": source.interpretation,
            }
        )
        rows.append(row)
    if len(recursive_familywise) != 1:
        raise RuntimeError("The recursive statewise family-wise summary must contain one row.")
    source = recursive_familywise.iloc[0]
    row = {column: np.nan for column in columns}
    row.update(
        {
            "panel": "A2. Statewise construction-aware family-wise test",
            "row_type": "recursive_familywise",
            "analysis_unit": "Fixed-10 tail-excess studentized maxT",
            "status": str(source["analysis_status"]),
            "k": 6,
            "observed": source["observed_maxT"],
            "null_mean": source["null_mean"],
            "null_sd": source["null_sd"],
            "null_2p5": source["null_2p5"],
            "null_median": source["null_median"],
            "null_97p5": source["null_97p5"],
            "monte_carlo_p": source["monte_carlo_p"],
            "finite_null_replicates": source["finite_replicates"],
            "fwer_positive_state_count": source["fwer_positive_state_count"],
            "fwer_positive_states": source["fwer_positive_states"],
            "multi_state_support": as_bool(source["multi_state_support"]),
            "interpretation": source["interpretation"],
        }
    )
    rows.append(row)
    formal_index = formal_summary.set_index("macrostate")
    for source in recursive_statewise.sort_values("macrostate").itertuples(index=False):
        state = int(source.macrostate)
        formal = formal_index.loc[state]
        row = {column: np.nan for column in columns}
        row.update(
            {
                "panel": "A3. Recursive statewise diagnostics",
                "row_type": "recursive_statewise",
                "analysis_unit": f"S{state}",
                "status": "family-wise tested fixed-10 tail",
                "k": 6,
                "macrostate": state,
                "observed_self_transition": source.observed_self_transition,
                "null_self_transition_2p5": source.null_self_transition_2p5,
                "null_self_transition_97p5": source.null_self_transition_97p5,
                "observed_rmst_lift_fixed10": source.observed_rmst_lift_fixed10,
                "null_rmst_lift_fixed10_2p5": source.null_rmst_lift_fixed10_2p5,
                "null_rmst_lift_fixed10_97p5": source.null_rmst_lift_fixed10_97p5,
                "observed_tail_excess_fixed10": source.observed_tail_excess_fixed10,
                "null_tail_excess_fixed10_2p5": source.null_tail_excess_fixed10_2p5,
                "null_tail_excess_fixed10_97p5": source.null_tail_excess_fixed10_97p5,
                "tail_excess_fixed10_null_mean": source.tail_excess_fixed10_null_mean,
                "tail_excess_fixed10_null_sd": source.tail_excess_fixed10_null_sd,
                "tail_excess_fixed10_standardized_excess": source.tail_excess_fixed10_standardized_excess,
                "tail_excess_fixed10_raw_monte_carlo_p": source.tail_excess_fixed10_raw_monte_carlo_p,
                "tail_excess_fixed10_maxT_fwer_p": source.tail_excess_fixed10_maxT_fwer_p,
                "tail_excess_fixed10_maxT_fwer_positive": as_bool(
                    source.tail_excess_fixed10_maxT_fwer_positive
                ),
                "formal_state_specific_rmst_lift": source.formal_state_specific_rmst_lift,
                "formal_rmst_tau": formal["rmst_tau"],
                "formal_reference_at_risk": formal["reference_at_risk"],
                "interpretation": (
                    "Positive fixed-10 tail excess beyond the recursive surrogate after maxT family-wise control."
                    if as_bool(source.tail_excess_fixed10_maxT_fwer_positive)
                    else "No positive construction-aware fixed-10 tail excess after maxT family-wise control."
                ),
            }
        )
        rows.append(row)
    for source in partition_summary.sort_values("k").itertuples(index=False):
        row = {column: np.nan for column in columns}
        row.update(
            {
                "panel": "B. Partition-resolution sensitivity",
                "row_type": "partition",
                "analysis_unit": f"K={int(source.k)}",
                "status": str(source.role),
                "k": int(source.k),
                "A_val_minimum_user_balanced_state_occupancy": source.A_val_minimum_user_balanced_state_occupancy,
                "A_train_transition_count": source.A_train_transition_count,
                "A_val_transition_count": source.A_val_transition_count,
                "A_val_residence_run_count": source.A_val_residence_run_count,
                "A_val_right_censoring_fraction": source.A_val_right_censoring_fraction,
                "A_val_diagonal_dominant_rows": source.A_val_diagonal_dominant_rows,
                "A_val_mean_self_transition": source.A_val_mean_self_transition,
                "A_val_min_self_transition": source.A_val_min_self_transition,
                "A_val_max_self_transition": source.A_val_max_self_transition,
                "A_val_states_with_rmst_lift_above_one": source.A_val_states_with_rmst_lift_above_one,
                "A_val_states_meeting_fixed10_at_risk_threshold": source.A_val_states_meeting_fixed10_at_risk_threshold,
                "A_val_states_with_positive_fixed10_tail_excess": source.A_val_states_with_positive_fixed10_tail_excess,
                "A_train_A_val_transition_mean_row_tv": source.A_train_A_val_transition_mean_row_tv,
                "A_train_A_val_transition_max_row_tv": source.A_train_A_val_transition_max_row_tv,
                "A_train_A_val_rmst_lift_mean_abs_log_difference": source.A_train_A_val_rmst_lift_mean_abs_log_difference,
                "interpretation": "K=6 is formal; all other rows are bounded sensitivity analyses without K selection or statewise inference.",
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows, columns=columns)
    result.insert(
        14,
        "null_transition_count_median_2p5_97p5",
        "",
    )
    result.insert(
        15,
        "null_residence_run_count_median_2p5_97p5",
        "",
    )
    result.insert(
        16,
        "null_right_censoring_fraction_median_2p5_97p5",
        "",
    )
    primary_mask = (result["row_type"] == "recursive_aggregate") & (result["status"] == "primary")
    transition_values = numeric(recursive_replicates, "transition_count")
    run_values = numeric(recursive_replicates, "residence_run_count")
    censored_values = numeric(recursive_replicates, "right_censored_run_count")
    censor_fraction = censored_values / np.maximum(run_values, 1.0)
    support_strings = {
        "null_transition_count_median_2p5_97p5": (
            f"{format_integer(quantile(transition_values, 0.5))} "
            f"[{format_integer(quantile(transition_values, 0.025))}, "
            f"{format_integer(quantile(transition_values, 0.975))}]"
        ),
        "null_residence_run_count_median_2p5_97p5": (
            f"{format_integer(quantile(run_values, 0.5))} "
            f"[{format_integer(quantile(run_values, 0.025))}, "
            f"{format_integer(quantile(run_values, 0.975))}]"
        ),
        "null_right_censoring_fraction_median_2p5_97p5": (
            f"{format_number(quantile(censor_fraction, 0.5))} "
            f"[{format_number(quantile(censor_fraction, 0.025))}, "
            f"{format_number(quantile(censor_fraction, 0.975))}]"
        ),
    }
    for column, value in support_strings.items():
        result.loc[primary_mask, column] = value
    result.loc[primary_mask, "recursive_panel_rows"] = int(recursive_manifest.get("panel_rows", 0))
    result.loc[primary_mask, "recursive_panel_users"] = int(recursive_manifest.get("panel_users", 0))
    result.loc[primary_mask, "recursive_valid_edges"] = int(recursive_manifest.get("valid_recursive_edges", 0))
    result.loc[primary_mask, "null_minimum_reference_at_risk_min"] = float(
        np.nanmin(numeric(recursive_replicates, "minimum_reference_at_risk"))
    )
    result.loc[primary_mask, "null_max_coordinate_bound_excess"] = float(
        np.nanmax(numeric(recursive_replicates, "coordinate_bound_excess_before_numerical_clipping"))
    )
    return result


def table2_numeric(
    cluster_summary: pd.DataFrame,
    formal_summary: pd.DataFrame,
) -> pd.DataFrame:
    formal_index = formal_summary.set_index("macrostate")
    rows: List[Dict[str, Any]] = []
    for source in cluster_summary.sort_values("macrostate").itertuples(index=False):
        state = int(source.macrostate)
        formal = formal_index.loc[state]
        greenwood_positive = bool(
            np.isfinite(float(source.greenwood_bh_q_formal))
            and float(source.greenwood_bh_q_formal) < 0.05
            and float(source.tail_excess_point) > 0
        )
        cluster_positive = as_bool(source.tail_excess_cluster_bh_positive)
        if as_bool(source.rmst_lift_lower_bound_above_one) and cluster_positive:
            conclusion = "Integrated and fixed-10 persistence supported"
        elif as_bool(source.rmst_lift_lower_bound_above_one):
            conclusion = "Integrated residence excess supported; fixed-10 evidence not BH-positive"
        elif cluster_positive:
            conclusion = "Fixed-10 evidence positive; RMST lower bound does not exceed one"
        else:
            conclusion = "Neither learner-cluster criterion retained"
        rows.append(
            {
                "macrostate": state,
                "n_runs": int(formal["n_runs"]),
                "n_completed_exits": int(formal["n_completed_exits"]),
                "n_right_censored": int(formal["n_right_censored"]),
                "right_censoring_fraction": float(formal["right_censoring_fraction"]),
                "formal_rmst_tau": int(formal["rmst_tau"]),
                "fixed_reference_length": int(formal["reference_length"]),
                "reference_at_risk": int(formal["reference_at_risk"]),
                "self_transition_point": source.self_transition_point,
                "self_transition_ci_2p5": source.self_transition_ci_2p5,
                "self_transition_ci_97p5": source.self_transition_ci_97p5,
                "diagonal_dominance_bootstrap_probability": source.diagonal_dominance_bootstrap_probability,
                "restricted_mean_residence_lift_point": source.restricted_mean_residence_lift_point,
                "restricted_mean_residence_lift_ci_2p5": source.restricted_mean_residence_lift_ci_2p5,
                "restricted_mean_residence_lift_ci_97p5": source.restricted_mean_residence_lift_ci_97p5,
                "rmst_lift_lower_bound_above_one": as_bool(source.rmst_lift_lower_bound_above_one),
                "tail_excess_point": source.tail_excess_point,
                "tail_excess_ci_2p5": source.tail_excess_ci_2p5,
                "tail_excess_ci_97p5": source.tail_excess_ci_97p5,
                "tail_excess_cluster_bootstrap_se": source.tail_excess_cluster_bootstrap_se,
                "tail_excess_cluster_z": source.tail_excess_cluster_z,
                "tail_excess_cluster_one_sided_p": source.tail_excess_cluster_one_sided_p,
                "tail_excess_cluster_bh_q": source.tail_excess_cluster_bh_q,
                "tail_excess_cluster_bh_positive": cluster_positive,
                "greenwood_one_sided_p_formal": source.greenwood_one_sided_p_formal,
                "greenwood_bh_q_formal": source.greenwood_bh_q_formal,
                "greenwood_bh_positive_formal": greenwood_positive,
                "fixed_time_inference_changed_from_greenwood": cluster_positive != greenwood_positive,
                "finite_self_transition_bootstrap_replicates": int(source.finite_self_transition_bootstrap_replicates),
                "finite_rmst_bootstrap_replicates": int(source.finite_rmst_bootstrap_replicates),
                "finite_tail_bootstrap_replicates": int(source.finite_tail_bootstrap_replicates),
                "conclusion": conclusion,
            }
        )
    return pd.DataFrame(rows)


def table1_display(table: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in table.itertuples(index=False):
        if row.row_type == "recursive_aggregate":
            rows.append(
                {
                    "Panel": row.panel,
                    "Analysis": row.analysis_unit,
                    "Status": row.status,
                    "Observed": format_number(row.observed),
                    "Recursive-null reference": (
                        f"mean {format_number(row.null_mean)}; median {format_number(row.null_median)}; "
                        f"95% [{format_number(row.null_2p5)}, {format_number(row.null_97p5)}]"
                    ),
                    "Inference": (
                        f"pMC={format_number(row.monte_carlo_p)}; R={format_integer(row.finite_null_replicates)}"
                        if str(row.status) == "primary"
                        else f"descriptive; R={format_integer(row.finite_null_replicates)}"
                    ),
                    "Support / reproducibility": (
                        (
                            f"rows/users/edges={format_integer(row.recursive_panel_rows)}/"
                            f"{format_integer(row.recursive_panel_users)}/"
                            f"{format_integer(row.recursive_valid_edges)}; "
                            f"transitions {row.null_transition_count_median_2p5_97p5}; "
                            f"runs {row.null_residence_run_count_median_2p5_97p5}; "
                            f"censoring {row.null_right_censoring_fraction_median_2p5_97p5}; "
                            f"min at risk={format_integer(row.null_minimum_reference_at_risk_min)}; "
                            f"max bound excess={format_number(row.null_max_coordinate_bound_excess)}"
                        )
                        if str(row.status) == "primary"
                        else "same recursive ensemble as the primary endpoint"
                    ),
                }
            )
        elif row.row_type == "recursive_familywise":
            rows.append(
                {
                    "Panel": row.panel,
                    "Analysis": row.analysis_unit,
                    "Status": row.status,
                    "Observed": f"maxT={format_number(row.observed)}",
                    "Recursive-null reference": (
                        f"mean {format_number(row.null_mean)}; median {format_number(row.null_median)}; "
                        f"95% [{format_number(row.null_2p5)}, {format_number(row.null_97p5)}]"
                    ),
                    "Inference": (
                        f"global pMC={format_number(row.monte_carlo_p)}; "
                        f"FWER-positive states={format_integer(row.fwer_positive_state_count)}/6 "
                        f"({str(row.fwer_positive_states) if str(row.fwer_positive_states) not in {'', 'nan'} else 'none'})"
                    ),
                    "Support / reproducibility": str(row.interpretation),
                }
            )
        elif row.row_type == "recursive_statewise":
            rows.append(
                {
                    "Panel": row.panel,
                    "Analysis": row.analysis_unit,
                    "Status": row.status,
                    "Observed": (
                        f"Pii={format_number(row.observed_self_transition)}; "
                        f"L10={format_number(row.observed_rmst_lift_fixed10)}; "
                        f"D10={format_number(row.observed_tail_excess_fixed10)}; "
                        f"formal RMST lift={format_number(row.formal_state_specific_rmst_lift)}"
                    ),
                    "Recursive-null reference": (
                        f"Pii [{format_number(row.null_self_transition_2p5)}, {format_number(row.null_self_transition_97p5)}]; "
                        f"L10 [{format_number(row.null_rmst_lift_fixed10_2p5)}, {format_number(row.null_rmst_lift_fixed10_97p5)}]; "
                        f"D10 [{format_number(row.null_tail_excess_fixed10_2p5)}, {format_number(row.null_tail_excess_fixed10_97p5)}]"
                    ),
                    "Inference": (
                        f"z={format_number(row.tail_excess_fixed10_standardized_excess)}; "
                        f"raw pMC={format_number(row.tail_excess_fixed10_raw_monte_carlo_p)}; "
                        f"maxT pFWER={format_number(row.tail_excess_fixed10_maxT_fwer_p)}; "
                        f"supported={'yes' if as_bool(row.tail_excess_fixed10_maxT_fwer_positive) else 'no'}"
                    ),
                    "Support / reproducibility": (
                        f"formal tau={format_integer(row.formal_rmst_tau)}; "
                        f"at risk at 10={format_integer(row.formal_reference_at_risk)}; "
                        f"null D10 mean/sd={format_number(row.tail_excess_fixed10_null_mean)}/"
                        f"{format_number(row.tail_excess_fixed10_null_sd)}"
                    ),
                }
            )
        else:
            rows.append(
                {
                    "Panel": row.panel,
                    "Analysis": row.analysis_unit,
                    "Status": row.status,
                    "Observed": (
                        f"diagonal {format_integer(row.A_val_diagonal_dominant_rows)}/{format_integer(row.k)}; "
                        f"mean Pii={format_number(row.A_val_mean_self_transition)} "
                        f"[{format_number(row.A_val_min_self_transition)}, {format_number(row.A_val_max_self_transition)}]; "
                        f"RMST>1 {format_integer(row.A_val_states_with_rmst_lift_above_one)}/{format_integer(row.k)}; "
                        f"D10>0 {format_integer(row.A_val_states_with_positive_fixed10_tail_excess)}/"
                        f"{format_integer(row.A_val_states_meeting_fixed10_at_risk_threshold)} supported"
                    ),
                    "Recursive-null reference": "not applicable",
                    "Inference": "bounded partition sensitivity; no K selection",
                    "Support / reproducibility": (
                        f"min occupancy={format_number(row.A_val_minimum_user_balanced_state_occupancy)}; "
                        f"A_val transitions={format_integer(row.A_val_transition_count)}; "
                        f"runs={format_integer(row.A_val_residence_run_count)}; "
                        f"censoring={format_number(row.A_val_right_censoring_fraction)}; "
                        f"row TV mean/max={format_number(row.A_train_A_val_transition_mean_row_tv)}/"
                        f"{format_number(row.A_train_A_val_transition_max_row_tv)}; "
                        f"|log RMST lift|={format_number(row.A_train_A_val_rmst_lift_mean_abs_log_difference)}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def table2_display(table: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in table.sort_values("macrostate").itertuples(index=False):
        rows.append(
            {
                "State": f"S{int(row.macrostate)}",
                "Formal support": (
                    f"runs={format_integer(row.n_runs)}; censored={format_number(row.right_censoring_fraction)}; "
                    f"tau={format_integer(row.formal_rmst_tau)}; at risk(10)={format_integer(row.reference_at_risk)}"
                ),
                "Pii [95% CI]": format_interval(
                    row.self_transition_point,
                    row.self_transition_ci_2p5,
                    row.self_transition_ci_97p5,
                ),
                "Pr(diagonal)": format_number(row.diagonal_dominance_bootstrap_probability),
                "RMST lift [95% CI]": format_interval(
                    row.restricted_mean_residence_lift_point,
                    row.restricted_mean_residence_lift_ci_2p5,
                    row.restricted_mean_residence_lift_ci_97p5,
                ),
                "D10 [95% CI]": format_interval(
                    row.tail_excess_point,
                    row.tail_excess_ci_2p5,
                    row.tail_excess_ci_97p5,
                ),
                "Cluster inference": (
                    f"SE={format_number(row.tail_excess_cluster_bootstrap_se)}; "
                    f"z={format_number(row.tail_excess_cluster_z)}; "
                    f"p={format_number(row.tail_excess_cluster_one_sided_p)}; "
                    f"q={format_number(row.tail_excess_cluster_bh_q)}"
                ),
                "Original Greenwood": (
                    f"p={format_number(row.greenwood_one_sided_p_formal)}; "
                    f"q={format_number(row.greenwood_bh_q_formal)}"
                ),
                "Finite bootstrap reps": (
                    f"Pii/RMST/D10={format_integer(row.finite_self_transition_bootstrap_replicates)}/"
                    f"{format_integer(row.finite_rmst_bootstrap_replicates)}/"
                    f"{format_integer(row.finite_tail_bootstrap_replicates)}"
                ),
                "Conclusion": row.conclusion,
            }
        )
    return pd.DataFrame(rows)


def scientific_summary(
    recursive_summary: pd.DataFrame,
    recursive_familywise: pd.DataFrame,
    recursive_statewise: pd.DataFrame,
    partition_summary: pd.DataFrame,
    cluster_table: pd.DataFrame,
) -> Dict[str, Any]:
    primary = recursive_summary[bool_series(recursive_summary["primary_endpoint"])]
    if len(primary) != 1 or len(recursive_familywise) != 1:
        raise RuntimeError("Kinetic scientific summary requires one aggregate and one family-wise row.")
    primary_row = primary.iloc[0]
    familywise_row = recursive_familywise.iloc[0]
    partition = partition_summary.sort_values("k")
    recursive_states = recursive_statewise.copy()
    recursive_states["macrostate"] = pd.to_numeric(
        recursive_states["macrostate"], errors="raise"
    ).astype(int)
    cluster_states = cluster_table.copy()
    cluster_states["macrostate"] = pd.to_numeric(
        cluster_states["macrostate"], errors="raise"
    ).astype(int)
    merged = recursive_states[
        ["macrostate", "tail_excess_fixed10_maxT_fwer_positive"]
    ].merge(
        cluster_states[
            [
                "macrostate",
                "tail_excess_cluster_bh_positive",
                "rmst_lift_lower_bound_above_one",
            ]
        ],
        on="macrostate",
        how="inner",
        validate="one_to_one",
    )
    merged["recursive_supported"] = bool_series(
        merged["tail_excess_fixed10_maxT_fwer_positive"]
    )
    merged["cluster_supported"] = bool_series(
        merged["tail_excess_cluster_bh_positive"]
    )
    merged["rmst_supported"] = bool_series(
        merged["rmst_lift_lower_bound_above_one"]
    )
    merged["joint_supported"] = (
        merged["recursive_supported"]
        & merged["cluster_supported"]
        & merged["rmst_supported"]
    )
    recursive_supported_states = [
        int(value)
        for value in merged.loc[merged["recursive_supported"], "macrostate"].tolist()
    ]
    joint_supported_states = [
        int(value)
        for value in merged.loc[merged["joint_supported"], "macrostate"].tolist()
    ]
    partition_all_rmst = {
        str(int(row.k)): bool(int(row.A_val_states_with_rmst_lift_above_one) == int(row.k))
        for row in partition.itertuples(index=False)
    }
    partition_all_diagonal = {
        str(int(row.k)): bool(int(row.A_val_diagonal_dominant_rows) == int(row.k))
        for row in partition.itertuples(index=False)
    }
    all_cluster_rmst = bool(merged["rmst_supported"].all())
    all_partition_rmst = bool(all(partition_all_rmst.values()))
    global_statewise_supported = bool(
        float(familywise_row["monte_carlo_p"]) < STATEWISE_FWER_ALPHA
    )
    if (
        global_statewise_supported
        and len(joint_supported_states) >= 2
        and all_cluster_rmst
        and all_partition_rmst
    ):
        conclusion_category = "operational_metastable_like_supported"
        recommended_claim = (
            "The frozen coarse-graining supports operationally defined metastable-like "
            "organisation: multiple states retain positive fixed-10 tail excess beyond the "
            "construction-aware recursive surrogate after family-wise control, while reliable-horizon "
            "RMST excess is learner-cluster robust and persists across the bounded K=4--8 partitions."
        )
    elif global_statewise_supported and len(joint_supported_states) >= 2:
        conclusion_category = "multistate_construction_aware_persistence_supported"
        recommended_claim = (
            "Multiple frozen states show construction-aware fixed-10 persistence after family-wise "
            "control, but the broader operational metastable-like claim should be limited to the "
            "robustness dimensions that remain positive."
        )
    elif global_statewise_supported and len(joint_supported_states) == 1:
        conclusion_category = "localized_construction_aware_persistence_only"
        recommended_claim = (
            "Construction-aware persistence is localized to one state; report persistent mesostate "
            "organisation and avoid a population-level metastable-like claim."
        )
    else:
        conclusion_category = "construction_aware_metastable_like_not_supported"
        recommended_claim = (
            "The data support reproducible persistent mesostate organisation, but no state retains "
            "positive fixed-10 tail excess under the joint construction-aware and learner-cluster "
            "criteria; the metastable-like wording should therefore be removed."
        )
    return {
        "recursive_primary_observed": float(primary_row["observed"]),
        "recursive_primary_null_median": float(primary_row["null_median"]),
        "recursive_primary_null_interval": [
            float(primary_row["null_2p5"]),
            float(primary_row["null_97p5"]),
        ],
        "recursive_primary_monte_carlo_p": float(primary_row["monte_carlo_p"]),
        "recursive_statewise_maxT_observed": float(familywise_row["observed_maxT"]),
        "recursive_statewise_maxT_null_interval": [
            float(familywise_row["null_2p5"]),
            float(familywise_row["null_97p5"]),
        ],
        "recursive_statewise_maxT_monte_carlo_p": float(
            familywise_row["monte_carlo_p"]
        ),
        "recursive_statewise_fwer_positive_states": recursive_supported_states,
        "joint_construction_cluster_supported_states": joint_supported_states,
        "partition_K_values": pd.to_numeric(
            partition["k"], errors="raise"
        ).astype(int).tolist(),
        "partition_all_states_rmst_above_one": partition_all_rmst,
        "partition_all_rows_diagonal_dominant": partition_all_diagonal,
        "formal_K6_cluster_RMST_lower_bounds_above_one": int(
            cluster_table["rmst_lift_lower_bound_above_one"].astype(bool).sum()
        ),
        "formal_K6_cluster_fixed10_BH_positive_states": int(
            cluster_table["tail_excess_cluster_bh_positive"].astype(bool).sum()
        ),
        "formal_K6_original_Greenwood_BH_positive_states": int(
            cluster_table["greenwood_bh_positive_formal"].astype(bool).sum()
        ),
        "formal_K6_states_with_changed_fixed_time_inference": int(
            cluster_table["fixed_time_inference_changed_from_greenwood"].astype(bool).sum()
        ),
        "metastable_like_conclusion_category": conclusion_category,
        "recommended_main_text_claim": recommended_claim,
    }


def main() -> None:
    args = parse_args()
    partition_root = args.partition_root.resolve()
    recursive_root = args.recursive_null_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = SourceRegistry()
    gates = GateRegistry()

    partition_manifest_path = partition_root / "metadata" / "partition_cluster_kinetic_manifest.json"
    recursive_manifest_path = recursive_root / "metadata" / "recursive_construction_inertia_null_manifest.json"
    partition_manifest = load_json(partition_manifest_path, "partition_manifest", registry)
    recursive_manifest = load_json(recursive_manifest_path, "recursive_null_manifest", registry)
    cluster_replicates_expected, null_replicates_expected = validate_manifests(
        partition_manifest,
        recursive_manifest,
        gates,
    )

    partition_summary = read_table(
        output_path(
            partition_manifest,
            "partition_robustness_summary",
            partition_root / "tables" / "partition_robustness_summary",
        ),
        "partition_robustness_summary",
        registry,
    )
    partition_statewise = read_table(
        output_path(
            partition_manifest,
            "partition_statewise_kinetics",
            partition_root / "tables" / "partition_statewise_kinetics",
        ),
        "partition_statewise_kinetics",
        registry,
    )
    cluster_summary = read_table(
        output_path(
            partition_manifest,
            "cluster_bootstrap_statewise_summary",
            partition_root / "tables" / "learner_cluster_bootstrap_statewise_summary",
        ),
        "learner_cluster_bootstrap_statewise_summary",
        registry,
    )
    cluster_replicates = read_table(
        output_path(
            partition_manifest,
            "cluster_bootstrap_replicates",
            partition_root / "tables" / "learner_cluster_bootstrap_replicates",
        ),
        "learner_cluster_bootstrap_replicates",
        registry,
    )
    k6_audit = read_table(
        output_path(
            partition_manifest,
            "formal_k6_reconstruction_audit",
            partition_root / "tables" / "formal_k6_reconstruction_audit",
        ),
        "formal_k6_reconstruction_audit",
        registry,
    )
    cluster_equivalence = read_table(
        output_path(
            partition_manifest,
            "cluster_formal_point_equivalence_audit",
            partition_root / "tables" / "cluster_bootstrap_formal_point_equivalence_audit",
        ),
        "cluster_bootstrap_formal_point_equivalence_audit",
        registry,
    )
    formal_summary_path = source_file_path(partition_manifest, "formal_A_val_residence_summary")
    formal_summary = read_table(formal_summary_path, "formal_A_val_residence_summary", registry)

    recursive_summary = read_table(
        output_path(
            recursive_manifest,
            "formal_summary",
            recursive_root / "tables" / "recursive_construction_inertia_null_summary",
        ),
        "recursive_construction_inertia_null_summary",
        registry,
    )
    recursive_replicates = read_table(
        output_path(
            recursive_manifest,
            "replicate_metrics",
            recursive_root / "tables" / "recursive_construction_inertia_null_replicates",
        ),
        "recursive_construction_inertia_null_replicates",
        registry,
    )
    recursive_statewise = read_table(
        output_path(
            recursive_manifest,
            "statewise_summary",
            recursive_root / "tables" / "recursive_construction_inertia_statewise_summary",
        ),
        "recursive_construction_inertia_statewise_summary",
        registry,
    )
    recursive_statewise_replicates = read_table(
        output_path(
            recursive_manifest,
            "statewise_replicates",
            recursive_root / "tables" / "recursive_construction_inertia_null_statewise_replicates",
        ),
        "recursive_construction_inertia_null_statewise_replicates",
        registry,
    )
    computed_familywise, computed_statewise_inference = recompute_statewise_maxT(
        recursive_statewise,
        recursive_statewise_replicates,
        STATEWISE_FWER_ALPHA,
    )
    inference_columns = [
        column
        for column in computed_statewise_inference.columns
        if column != "macrostate"
    ]
    missing_inference_columns = [
        column for column in inference_columns if column not in recursive_statewise.columns
    ]
    if missing_inference_columns:
        recursive_statewise = recursive_statewise.drop(
            columns=[
                column for column in inference_columns if column in recursive_statewise.columns
            ],
            errors="ignore",
        ).merge(
            computed_statewise_inference,
            on="macrostate",
            how="left",
            validate="one_to_one",
        )
        gates.add(
            "recursive_statewise_maxT_materialized_from_archived_replicates",
            True,
            {"added_columns": missing_inference_columns},
            hard=False,
        )
    familywise_path = optional_output_path(
        recursive_manifest,
        "statewise_familywise_summary",
        recursive_root
        / "tables"
        / "recursive_construction_inertia_statewise_familywise_summary",
    )
    if familywise_path is None:
        recursive_familywise = pd.DataFrame([computed_familywise])
        gates.add(
            "recursive_familywise_summary_materialized_from_archived_replicates",
            True,
            {"archived_table": False},
            hard=False,
        )
    else:
        recursive_familywise = read_table(
            familywise_path,
            "recursive_construction_inertia_statewise_familywise_summary",
            registry,
        )
    identity_audit = read_table(
        output_path(
            recursive_manifest,
            "identity_reconstruction_audit",
            recursive_root / "tables" / "recursive_identity_reconstruction_audit",
        ),
        "recursive_identity_reconstruction_audit",
        registry,
    )
    coverage_audit = read_table(
        output_path(
            recursive_manifest,
            "matching_coverage_audit",
            recursive_root / "tables" / "recursive_null_matching_coverage_audit",
        ),
        "recursive_null_matching_coverage_audit",
        registry,
    )

    require_columns(k6_audit, ("check", "passed"), "formal K=6 reconstruction audit")
    require_columns(cluster_equivalence, ("check", "passed"), "cluster point-equivalence audit")
    require_columns(identity_audit, ("check", "passed"), "recursive identity audit")
    require_columns(coverage_audit, ("level", "passed"), "matching coverage audit")
    gates.add("formal_k6_reconstruction_audit", bool(bool_series(k6_audit["passed"]).all()), k6_audit.to_dict("records"))
    gates.add("cluster_point_equivalence_audit", bool(bool_series(cluster_equivalence["passed"]).all()), cluster_equivalence.to_dict("records"))
    gates.add("recursive_identity_audit", bool(bool_series(identity_audit["passed"]).all()), identity_audit.to_dict("records"))
    gates.add("recursive_matching_coverage_audit", bool(bool_series(coverage_audit["passed"]).all()), coverage_audit.to_dict("records"))

    validate_recursive_outputs(
        recursive_summary,
        recursive_replicates,
        recursive_statewise,
        recursive_statewise_replicates,
        recursive_familywise,
        null_replicates_expected,
        gates,
    )
    validate_partition_outputs(
        partition_summary,
        partition_statewise,
        partition_manifest,
        partition_root,
        gates,
        registry,
    )
    validate_cluster_outputs(
        cluster_summary,
        cluster_replicates,
        formal_summary,
        cluster_replicates_expected,
        gates,
    )

    gate_frame = gates.frame()
    quality_path = write_csv(
        gate_frame,
        output_dir / "kinetic_robustness_quality_gates.csv",
    )
    failed = gates.failed_hard()
    if not failed.empty:
        names = failed["gate"].astype(str).tolist()
        raise RuntimeError(f"Kinetic robustness extraction failed quality gates: {names}")

    table1 = table1_numeric(
        recursive_summary,
        recursive_statewise,
        recursive_familywise,
        recursive_replicates,
        partition_summary,
        formal_summary,
        recursive_manifest,
    )
    table2 = table2_numeric(cluster_summary, formal_summary)
    table1_path = write_csv(
        table1,
        output_dir / "kinetic_robustness_table1_null_partition.csv",
    )
    table2_path = write_csv(
        table2,
        output_dir / "kinetic_robustness_table2_cluster_inference.csv",
    )
    source_audit = pd.DataFrame(registry.records)
    source_audit_path = write_csv(
        source_audit,
        output_dir / "kinetic_robustness_source_audit.csv",
    )

    display1 = table1_display(table1)
    display2 = table2_display(table2)
    summary = scientific_summary(
        recursive_summary,
        recursive_familywise,
        recursive_statewise,
        partition_summary,
        table2,
    )
    recursive_states_text = ", ".join(
        f"S{state}" for state in summary["recursive_statewise_fwer_positive_states"]
    ) or "none"
    joint_states_text = ", ".join(
        f"S{state}" for state in summary["joint_construction_cluster_supported_states"]
    ) or "none"
    report_lines = [
        "# Kinetic robustness numerical report",
        "",
        "The report is confined to the empirical coarse-kinetic branch. The formal K=6 partition remains unchanged. The recursive surrogate is conditional on the archived denominator and matching-stratum paths, and the K=4--8 analyses are bounded resolution checks rather than model selection.",
        "",
        (
            "The original all-state fixed-10 aggregate endpoint was "
            f"{format_number(summary['recursive_primary_observed'])} in the data and "
            f"{format_number(summary['recursive_primary_null_median'])} under the recursive surrogate "
            f"(95% null interval [{format_number(summary['recursive_primary_null_interval'][0])}, "
            f"{format_number(summary['recursive_primary_null_interval'][1])}], "
            f"Monte Carlo p={format_number(summary['recursive_primary_monte_carlo_p'])})."
        ),
        "",
        (
            "The complementary statewise family-wise endpoint gave "
            f"maxT={format_number(summary['recursive_statewise_maxT_observed'])} "
            f"against a 95% null interval [{format_number(summary['recursive_statewise_maxT_null_interval'][0])}, "
            f"{format_number(summary['recursive_statewise_maxT_null_interval'][1])}] "
            f"(global Monte Carlo p={format_number(summary['recursive_statewise_maxT_monte_carlo_p'])}). "
            f"Positive fixed-10 tail excess survived maxT family-wise control in: {recursive_states_text}."
        ),
        "",
        (
            "Under formal K=6 learner-cluster inference, "
            f"{summary['formal_K6_cluster_RMST_lower_bounds_above_one']}/6 RMST-lift lower bounds exceeded one and "
            f"{summary['formal_K6_cluster_fixed10_BH_positive_states']}/6 fixed-10 tail tests were BH-positive, "
            f"compared with {summary['formal_K6_original_Greenwood_BH_positive_states']}/6 under the original Greenwood analysis. "
            f"States satisfying the recursive maxT, learner-cluster fixed-10 and learner-cluster RMST criteria jointly were: {joint_states_text}."
        ),
        "",
        f"Conclusion contract: {summary['recommended_main_text_claim']}",
        "",
        "The aggregate and statewise endpoints test different alternatives and are both retained: the first asks whether all-state early-horizon persistence shifts upward on average, whereas the second asks whether any frozen state shows construction-aware positive tail excess under family-wise control.",
        "",
        "## Supplementary Table 1. Recursive kinetic surrogate and bounded partition-resolution sensitivity",
        "",
        markdown_table(display1),
        "",
        "The equal-state mean log fixed-10 RMST lift is the original aggregate Monte Carlo endpoint. The fixed-10 statewise tail-excess rows use a single-step studentized maxT test across all six frozen states; raw and family-wise adjusted Monte Carlo p values are shown. K=6 is the formal partition; K=4,5,7,8 are sensitivity analyses and do not reselect K.",
        "",
        "## Supplementary Table 2. Formal K=6 learner-cluster kinetic inference",
        "",
        markdown_table(display2),
        "",
        "The same learner multiplicity is applied jointly to transitions and all residence runs. State-specific RMST horizons remain fixed at their formal values. Percentile intervals summarize effect-size uncertainty; fixed-10 p values use the learner-cluster bootstrap standard error and BH adjustment across the six frozen states.",
        "",
    ]
    report_path = output_dir / "kinetic_robustness_numerical_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    outputs = {
        "numerical_report": str(report_path),
        "table1_null_partition": str(table1_path),
        "table2_cluster_inference": str(table2_path),
        "quality_gates": str(quality_path),
        "source_audit": str(source_audit_path),
    }
    manifest = {
        "script": Path(__file__).name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "partition_root": str(partition_root),
        "recursive_null_root": str(recursive_root),
        "output_dir": str(output_dir),
        "typeset_result_tables": 2,
        "table_contract": {
            "table1": "recursive aggregate endpoint, statewise maxT family-wise inference and bounded K=4–8 partition sensitivity",
            "table2": "formal K=6 learner-cluster Pii, RMST and fixed-10 tail inference",
        },
        "analysis_boundary": {
            "B_confirm_read": False,
            "formal_K6_reselected": False,
            "construction_matched_field_null_rerun": False,
            "coordinate_or_grid_sensitivity_rerun": False,
            "strict_user_equal_transition_rerun": False,
            "positive_exponential_multiplier_rerun": False,
            "mechanism_or_Event_SSL_evaluated": False,
        },
        "replicates": {
            "recursive_null": null_replicates_expected,
            "learner_cluster_bootstrap": cluster_replicates_expected,
        },
        "scientific_summary": summary,
        "quality_gates_passed": True,
        "source_count": int(len(source_audit)),
        "outputs": outputs,
        "output_sha256": {
            key: sha256_file(Path(path)) for key, path in outputs.items()
        },
    }
    manifest_path = output_dir / "kinetic_robustness_report_manifest.json"
    save_json(manifest, manifest_path)
    print(f"[kinetic robustness extraction] report: {report_path}", flush=True)
    print(f"[kinetic robustness extraction] table 1: {table1_path}", flush=True)
    print(f"[kinetic robustness extraction] table 2: {table2_path}", flush=True)
    print(f"[kinetic robustness extraction] manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
