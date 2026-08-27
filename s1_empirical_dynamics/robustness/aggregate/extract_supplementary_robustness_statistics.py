#!/usr/bin/env python3
"""Extract the supplementary learner-level and evaluation-gauge robustness report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_OUTPUT_BASE = Path(
    os.environ.get("EDNET_KT4_OUTPUT_ROOT", "/data/datasets/KT4/outputs_KT4")
)
DEFAULT_COORDINATE_ROOT = Path(
    os.environ.get(
        "EDNET_STAGE1_SENSITIVITY_SUMMARY_ROOT",
        "/data/datasets/KT4/outputs_KT4_stage1_sensitivity/summary",
    )
)
MIN_BOOTSTRAP_REPLICATES = 1000
MIN_PERMUTATION_REPLICATES = 50
EXPECTED_COORDINATE_SETTINGS = {
    "formal_10d",
    "memory_5d",
    "memory_20d",
    "activity_fast",
    "activity_slow",
}
REPRESENTATION_ORDER = ("full_hidden", "macro_only", "residual_hidden")
REPRESENTATION_LABELS = {
    "full_hidden": "Full hidden state",
    "macro_only": "Two-coordinate bottleneck",
    "residual_hidden": "Residual hidden representation",
    "macro_only_minus_full_hidden": "Bottleneck minus full hidden",
}
EMPIRICAL_METRICS = (
    "occupancy_js",
    "train_validation_mean_local_drift_cosine",
    "train_validation_drift_vector_corr",
    "train_validation_drift_speed_corr",
    "validation_negative_divergence_fraction",
    "validation_weighted_mean_divergence",
    "validation_shell_inward_fraction",
    "validation_shell_inward_cosine",
    "validation_core_to_shell_speed_ratio",
    "validation_core_occupancy_mass",
)
MODEL_METRICS = (
    "mechanism_drift_vector_corr",
    "event_ssl_learned_drift_vector_corr",
    "event_ssl_learned_weighted_local_cosine",
    "time_shuffle_learned_drift_vector_corr",
    "tag_support_learned_drift_vector_corr",
    "event_ssl_inward_fraction",
    "tag_support_inward_fraction",
    "main_minus_tag_inward_fraction",
    "cross_model_anchor_drift_vector_corr",
    "cross_model_anchor_drift_speed_corr",
    "cross_model_anchor_weighted_local_cosine",
    "mechanism_self_transition_corr",
    "event_ssl_self_transition_corr",
    "time_shuffle_self_transition_corr",
    "tag_support_self_transition_corr",
    "cross_model_self_transition_corr",
    "cross_model_transition_mean_row_tv",
    "event_ssl_coordinate_corr_M",
    "event_ssl_coordinate_corr_Psi",
    "event_ssl_one_step_rmse_M",
    "event_ssl_one_step_rmse_Psi",
)
STAGE5_METRICS = (
    "coordinate_corr_M",
    "coordinate_corr_Psi",
    "one_step_rmse_M",
    "one_step_rmse_Psi",
    "learned_plane_drift_vector_corr",
    "learned_plane_occupancy_weighted_local_drift_cosine",
    "learned_plane_transition_mean_row_tv",
    "learned_plane_self_transition_corr",
    "learned_plane_diagonal_dominance_match_fraction",
    "learned_plane_top_transition_edge_overlap",
)
STAGE5_FIELD_METRICS = {
    "learned_plane_drift_vector_corr",
    "learned_plane_occupancy_weighted_local_drift_cosine",
}
MODEL_FIELD_METRICS = {
    "mechanism_drift_vector_corr",
    "event_ssl_learned_drift_vector_corr",
    "event_ssl_learned_weighted_local_cosine",
    "time_shuffle_learned_drift_vector_corr",
    "tag_support_learned_drift_vector_corr",
    "event_ssl_inward_fraction",
    "tag_support_inward_fraction",
    "main_minus_tag_inward_fraction",
    "cross_model_anchor_drift_vector_corr",
    "cross_model_anchor_drift_speed_corr",
    "cross_model_anchor_weighted_local_cosine",
}
LOWER_IS_BETTER = {
    "occupancy_js",
    "one_step_rmse_M",
    "one_step_rmse_Psi",
    "event_ssl_one_step_rmse_M",
    "event_ssl_one_step_rmse_Psi",
    "cross_model_transition_mean_row_tv",
    "learned_plane_transition_mean_row_tv",
    "drift_component_rmse",
}
METRIC_LABELS = {
    "occupancy_js": "Training–validation occupancy JS divergence",
    "train_validation_mean_local_drift_cosine": "Training–validation mean local drift cosine",
    "train_validation_drift_vector_corr": "Training–validation drift-vector correlation",
    "train_validation_drift_speed_corr": "Training–validation drift-speed correlation",
    "validation_negative_divergence_fraction": "Validation negative-divergence occupancy fraction",
    "validation_weighted_mean_divergence": "Validation occupancy-weighted mean divergence",
    "validation_shell_inward_fraction": "Validation frozen-shell inward fraction",
    "validation_shell_inward_cosine": "Validation frozen-shell inward cosine",
    "validation_core_to_shell_speed_ratio": "Validation core-to-shell speed ratio",
    "validation_core_occupancy_mass": "Validation frozen-core occupancy mass",
    "rmst_lift": "Restricted-mean residence lift",
    "self_transition": "Self-transition probability",
    "tail_excess": "Fixed-reference tail excess",
    "mechanism_drift_vector_corr": "Minimal-mechanism drift-vector correlation",
    "event_ssl_learned_drift_vector_corr": "Event-SSL learned-plane drift-vector correlation",
    "event_ssl_learned_weighted_local_cosine": "Event-SSL learned-plane occupancy-weighted local cosine",
    "time_shuffle_learned_drift_vector_corr": "Time-shuffle learned-plane drift-vector correlation",
    "tag_support_learned_drift_vector_corr": "Tag/support-randomized learned-plane drift-vector correlation",
    "event_ssl_inward_fraction": "Event-SSL inward-flow fraction",
    "tag_support_inward_fraction": "Tag/support-randomized inward-flow fraction",
    "main_minus_tag_inward_fraction": "Event-SSL minus tag/support-randomized inward-flow fraction",
    "cross_model_anchor_drift_vector_corr": "Mechanism–Event-SSL empirical-anchor field correlation",
    "cross_model_anchor_drift_speed_corr": "Mechanism–Event-SSL empirical-anchor speed correlation",
    "cross_model_anchor_weighted_local_cosine": "Mechanism–Event-SSL empirical-anchor weighted local cosine",
    "mechanism_self_transition_corr": "Minimal-mechanism self-transition correlation",
    "event_ssl_self_transition_corr": "Event-SSL self-transition correlation",
    "time_shuffle_self_transition_corr": "Time-shuffle self-transition correlation",
    "tag_support_self_transition_corr": "Tag/support-randomized self-transition correlation",
    "cross_model_self_transition_corr": "Mechanism–Event-SSL self-transition correlation",
    "cross_model_transition_mean_row_tv": "Mechanism–Event-SSL mean transition row TV",
    "event_ssl_coordinate_corr_M": "Event-SSL coordinate correlation M",
    "event_ssl_coordinate_corr_Psi": "Event-SSL coordinate correlation Ψ",
    "event_ssl_one_step_rmse_M": "Event-SSL one-step RMSE M",
    "event_ssl_one_step_rmse_Psi": "Event-SSL one-step RMSE Ψ",
    "coordinate_corr_M": "Coordinate correlation M",
    "coordinate_corr_Psi": "Coordinate correlation Ψ",
    "one_step_rmse_M": "One-step RMSE M",
    "one_step_rmse_Psi": "One-step RMSE Ψ",
    "learned_plane_drift_vector_corr": "Learned-plane drift-vector correlation",
    "learned_plane_occupancy_weighted_local_drift_cosine": "Learned-plane occupancy-weighted local cosine",
    "learned_plane_transition_mean_row_tv": "Learned-plane mean transition row TV",
    "learned_plane_self_transition_corr": "Learned-plane self-transition correlation",
    "learned_plane_diagonal_dominance_match_fraction": "Learned-plane diagonal-dominance agreement",
    "learned_plane_top_transition_edge_overlap": "Learned-plane top-edge overlap",
}
MODEL_TRANSITION_VIEW = {
    "mechanism_self_transition_corr": ("mechanism",),
    "event_ssl_self_transition_corr": ("main_learned",),
    "time_shuffle_self_transition_corr": ("time_learned",),
    "tag_support_self_transition_corr": ("tag_learned",),
    "cross_model_self_transition_corr": ("mechanism", "main_anchor"),
    "cross_model_transition_mean_row_tv": ("mechanism", "main_anchor"),
}
CLAIM_RULES = {
    "train_validation_mean_local_drift_cosine": ("greater", 0.0),
    "train_validation_drift_vector_corr": ("greater", 0.0),
    "train_validation_drift_speed_corr": ("greater", 0.0),
    "validation_negative_divergence_fraction": ("greater", 0.5),
    "validation_weighted_mean_divergence": ("less", 0.0),
    "validation_shell_inward_fraction": ("greater", 0.5),
    "validation_shell_inward_cosine": ("greater", 0.0),
    "validation_core_to_shell_speed_ratio": ("less", 1.0),
    "rmst_lift": ("greater", 1.0),
    "tail_excess": ("greater", 0.0),
    "mechanism_drift_vector_corr": ("greater", 0.0),
    "event_ssl_learned_drift_vector_corr": ("greater", 0.0),
    "event_ssl_learned_weighted_local_cosine": ("greater", 0.0),
    "time_shuffle_learned_drift_vector_corr": ("less", 0.0),
    "main_minus_tag_inward_fraction": ("greater", 0.0),
    "cross_model_anchor_drift_vector_corr": ("greater", 0.0),
    "cross_model_anchor_drift_speed_corr": ("greater", 0.0),
    "cross_model_anchor_weighted_local_cosine": ("greater", 0.0),
    "cross_model_self_transition_corr": ("greater", 0.0),
}


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


def parse_args() -> argparse.Namespace:
    default_root = DEFAULT_OUTPUT_BASE / "supplementary_robustness"
    parser = argparse.ArgumentParser()
    parser.add_argument("--robustness-root", type=Path, default=default_root)
    parser.add_argument("--empirical-root", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--representation-root", type=Path, default=None)
    parser.add_argument("--coordinate-summary-root", type=Path, default=DEFAULT_COORDINATE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--logs-root", type=Path, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_table(base: Path) -> Path:
    if base.exists() and base.is_file():
        return base
    for suffix in (".parquet", ".csv.gz", ".csv"):
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find table for {base}")


def read_table(base: Path, name: str, registry: SourceRegistry) -> pd.DataFrame:
    path = resolve_table(base)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
    registry.add(name, path, len(frame), list(frame.columns))
    return frame


def read_json(path: Path, name: str, registry: SourceRegistry) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON source: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    registry.add(name, path)
    return payload


def resolve_branch_root(
    explicit: Optional[Path],
    robustness_root: Path,
    default_name: str,
    manifest_name: str,
) -> Path:
    if explicit is not None:
        return explicit.resolve()
    expected = (robustness_root / default_name).resolve()
    if (expected / "metadata" / manifest_name).is_file():
        return expected
    matches = sorted(robustness_root.rglob(manifest_name)) if robustness_root.exists() else []
    roots = sorted({match.parent.parent.resolve() for match in matches})
    if len(roots) == 1:
        return roots[0]
    if len(roots) > 1:
        listed = "\n".join(f"  - {root}" for root in roots)
        raise RuntimeError(
            f"Multiple {manifest_name} files were found under {robustness_root}. "
            f"Pass the corresponding branch root explicitly:\n{listed}"
        )
    return expected


def tail_text(path: Path, lines: int = 80) -> str:
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-max(int(lines), 1):])


def preflight_sources(
    empirical_root: Path,
    model_root: Path,
    representation_root: Path,
    coordinate_root: Path,
    logs_root: Path,
) -> None:
    required = [
        (
            "empirical robustness branch",
            empirical_root / "metadata" / "empirical_robustness_manifest.json",
            logs_root / "empirical.log",
        ),
        (
            "frozen-model robustness branch",
            model_root / "metadata" / "model_robustness_manifest.json",
            logs_root / "models.log",
        ),
        (
            "representation robustness branch",
            representation_root / "metadata" / "representation_robustness_manifest.json",
            logs_root / "representations.log",
        ),
        (
            "coordinate sensitivity manifest",
            coordinate_root / "empirical_coordinate_sensitivity_manifest.json",
            Path(),
        ),
        (
            "coordinate sensitivity statistics",
            coordinate_root / "empirical_coordinate_sensitivity_statistics.csv",
            Path(),
        ),
    ]
    missing = [(label, path, log) for label, path, log in required if not path.is_file()]
    if not missing:
        return
    sections = [
        "Supplementary robustness outputs are incomplete. The numerical report cannot be extracted "
        "until every upstream branch has written its completion manifest.",
    ]
    for label, path, log in missing:
        sections.append(f"\nMissing {label}: {path}")
        if str(log) not in {"", "."}:
            excerpt = tail_text(log)
            if excerpt:
                sections.append(f"Last lines from {log}:\n{excerpt}")
            else:
                sections.append(f"No branch log was found at {log}.")
    sections.append(
        "\nRun run_supplementary_robustness.sh with the same ROBUSTNESS_ROOT, or pass "
        "--robustness-root/branch-root arguments that identify the completed outputs."
    )
    raise FileNotFoundError("\n".join(sections))


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing columns: {missing}")


def bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", "", "nan", "none"}:
        return False
    raise RuntimeError(f"Cannot interpret Boolean value: {value!r}")


def finite(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    number = finite(value)
    if not np.isfinite(number):
        return "—"
    if number == 0:
        return "0"
    if abs(number) >= 10000:
        return f"{number:,.0f}"
    if abs(number) < 1e-3:
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def fmt_count(value: Any) -> str:
    number = finite(value)
    return f"{int(round(number)):,}" if np.isfinite(number) else "—"


def fmt_interval(point: Any, lower: Any, upper: Any) -> str:
    return f"{fmt(point)} [{fmt(lower)}, {fmt(upper)}]"


def markdown_table(frame: pd.DataFrame) -> str:
    def escape(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(escape(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(escape(value) for value in row) + " |")
    return "\n".join(lines)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)


def add_gate(gates: List[Dict[str, Any]], category: str, gate: str, passed: bool, detail: str) -> None:
    gates.append({"category": category, "gate": gate, "passed": bool(passed), "detail": detail})


def assert_gate(gates: List[Dict[str, Any]], category: str, gate: str, passed: bool, detail: str) -> None:
    add_gate(gates, category, gate, passed, detail)
    if not passed:
        raise RuntimeError(f"Quality gate failed: {gate}. {detail}")


def nested(payload: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def validate_audit(frame: pd.DataFrame, label: str, gates: List[Dict[str, Any]]) -> None:
    require_columns(frame, ["passed"], label)
    passed = bool(frame["passed"].map(bool_value).all()) and not frame.empty
    assert_gate(gates, "equivalence", label, passed, f"rows={len(frame)}")


def validate_inputs(
    empirical_manifest: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    representation_manifest: Mapping[str, Any],
    coordinate_statistics: pd.DataFrame,
    coordinate_gates: Optional[pd.DataFrame],
    empirical_bootstrap: pd.DataFrame,
    residence_bootstrap: pd.DataFrame,
    model_bootstrap: pd.DataFrame,
    stage5_bootstrap: pd.DataFrame,
    permutation_floor: pd.DataFrame,
    model_strict: pd.DataFrame,
    stage5_strict: pd.DataFrame,
    gates: List[Dict[str, Any]],
) -> Tuple[int, int, int, int]:
    empirical_replicates = int(nested(empirical_manifest, "bootstrap", "replicates", default=0))
    model_replicates = int(nested(model_manifest, "bootstrap", "replicates", default=0))
    stage5_replicates = int(nested(representation_manifest, "bootstrap", "replicates", default=0))
    permutation_replicates = int(nested(representation_manifest, "permutation_floor", "replicates", default=0))
    assert_gate(gates, "contract", "empirical_bootstrap_replicates", empirical_replicates >= MIN_BOOTSTRAP_REPLICATES, f"replicates={empirical_replicates}")
    assert_gate(gates, "contract", "model_bootstrap_replicates", model_replicates >= MIN_BOOTSTRAP_REPLICATES, f"replicates={model_replicates}")
    assert_gate(gates, "contract", "stage5_bootstrap_replicates", stage5_replicates >= MIN_BOOTSTRAP_REPLICATES, f"replicates={stage5_replicates}")
    assert_gate(gates, "contract", "permutation_replicates", permutation_replicates >= MIN_PERMUTATION_REPLICATES, f"replicates={permutation_replicates}")
    assert_gate(gates, "contract", "empirical_support_and_core_frozen", bool_value(nested(empirical_manifest, "bootstrap", "support_masks_and_training_core_frozen", default=False)), "Empirical support masks and the training core must remain frozen.")
    assert_gate(gates, "contract", "model_same_learner_multipliers", bool_value(nested(model_manifest, "bootstrap", "same_multiplier_for_all_frozen_models_and_controls", default=False)), "Frozen models and controls must share each learner multiplier.")
    assert_gate(gates, "contract", "stage5_same_learner_multipliers", bool_value(nested(representation_manifest, "bootstrap", "same_multiplier_for_full_macro_and_residual", default=False)), "Stage-5 representations must share each learner multiplier.")
    assert_gate(gates, "contract", "no_model_or_probe_refit", not bool_value(nested(model_manifest, "bootstrap", "models_refit", default=True)) and not bool_value(nested(model_manifest, "bootstrap", "probes_refit", default=True)) and not bool_value(nested(representation_manifest, "bootstrap", "model_retrained", default=True)) and not bool_value(nested(representation_manifest, "bootstrap", "probes_refit", default=True)), "No frozen model or probe may be refitted.")
    assert_gate(gates, "contract", "no_partition_or_core_refit", not bool_value(nested(model_manifest, "bootstrap", "KMeans_refit", default=True)) and not bool_value(nested(empirical_manifest, "grid_sensitivity", "core_reselected", default=True)) and not bool_value(nested(model_manifest, "grid_sensitivity", "core_reselected", default=True)) and not bool_value(nested(representation_manifest, "grid_sensitivity", "core_reselected", default=True)) and not bool_value(nested(representation_manifest, "grid_sensitivity", "partition_refit", default=True)), "The convergence core and fixed K=6 partition must remain frozen.")
    assert_gate(gates, "contract", "no_new_p_values", not bool_value(nested(empirical_manifest, "bootstrap", "new_p_values", default=True)) and not bool_value(nested(model_manifest, "bootstrap", "new_p_values", default=True)) and not bool_value(nested(representation_manifest, "bootstrap", "new_p_values", default=True)) and not bool_value(nested(representation_manifest, "permutation_floor", "p_values", default=True)), "The analyses must remain interval and descriptive sensitivity analyses.")
    assert_gate(gates, "contract", "permutation_floor_descriptive", bool_value(nested(representation_manifest, "permutation_floor", "descriptive_only", default=False)) and bool_value(nested(representation_manifest, "permutation_floor", "same_permutation_indices_across_representations", default=False)) and bool_value(nested(representation_manifest, "permutation_floor", "empirical_targets_fixed", default=False)), "The permutation floor must preserve the declared paired descriptive contract.")
    assert_gate(gates, "contract", "raw_metrics_primary", bool_value(nested(representation_manifest, "score_sensitivity", "raw_metrics_primary", default=False)) and bool_value(nested(representation_manifest, "score_sensitivity", "formal_90p8_percent_not_used_as_primary_evidence", default=False)), "Raw Stage-5 metrics must remain primary.")
    coordinate_ids = set(coordinate_statistics["setting_id"].astype(str)) if "setting_id" in coordinate_statistics.columns else set()
    assert_gate(gates, "contract", "coordinate_settings_complete", coordinate_ids == EXPECTED_COORDINATE_SETTINGS, f"settings={sorted(coordinate_ids)}")
    if coordinate_gates is not None:
        require_columns(coordinate_gates, ["category", "passed"], "coordinate quality gates")
        contract = coordinate_gates[coordinate_gates["category"].astype(str) == "contract"]
        assert_gate(gates, "contract", "coordinate_contract_gates", not contract.empty and bool(contract["passed"].map(bool_value).all()), f"contract_rows={len(contract)}")
    for label, frame, expected in (
        ("empirical bootstrap", empirical_bootstrap, empirical_replicates),
        ("residence bootstrap", residence_bootstrap, empirical_replicates),
        ("model bootstrap", model_bootstrap, model_replicates),
        ("Stage-5 bootstrap", stage5_bootstrap, stage5_replicates),
    ):
        require_columns(frame, ["replicates_finite", "formal_point_estimate"], label)
        finite_formal = pd.to_numeric(frame["formal_point_estimate"], errors="coerce").notna()
        adequate = pd.to_numeric(frame.loc[finite_formal, "replicates_finite"], errors="coerce") >= int(np.floor(0.95 * expected))
        assert_gate(gates, "coverage", label.replace(" ", "_"), bool(adequate.all()) and bool(finite_formal.any()), f"finite_formal_rows={int(finite_formal.sum())}; expected={expected}")
    require_columns(permutation_floor, ["null_replicates_finite"], "permutation floor")
    adequate_floor = pd.to_numeric(permutation_floor["null_replicates_finite"], errors="coerce") >= permutation_replicates
    assert_gate(gates, "coverage", "permutation_floor_complete", bool(adequate_floor.all()) and not permutation_floor.empty, f"rows={len(permutation_floor)}")
    for label, frame in (("model", model_strict), ("Stage-5", stage5_strict)):
        selected = frame[frame["metric"].astype(str) == "maximum_absolute_user_mass_minus_one"]
        require_columns(frame, ["metric", "formal_value"], f"{label} strict sensitivity")
        passed = len(selected) == 1 and abs(finite(selected.iloc[0]["formal_value"])) <= 1e-10
        assert_gate(gates, "weighting", f"{label.lower()}_field_user_mass", passed, f"formal_max_abs_mass_minus_one={fmt(selected.iloc[0]['formal_value']) if len(selected) else 'missing'}")
    return empirical_replicates, model_replicates, stage5_replicates, permutation_replicates


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").strip().capitalize())


def claim_status(metric: str, lower: Any, upper: Any, strict_value: Any = np.nan) -> str:
    if metric not in CLAIM_RULES:
        return "Descriptive"
    direction, threshold = CLAIM_RULES[metric]
    lo = finite(lower)
    hi = finite(upper)
    strict = finite(strict_value)
    if direction == "greater":
        interval_pass = np.isfinite(lo) and lo > threshold
        strict_pass = not np.isfinite(strict) or strict > threshold
        symbol = ">"
    else:
        interval_pass = np.isfinite(hi) and hi < threshold
        strict_pass = not np.isfinite(strict) or strict < threshold
        symbol = "<"
    return f"Retained ({symbol} {fmt(threshold)})" if interval_pass and strict_pass else f"Not retained ({symbol} {fmt(threshold)})"


def support_range(frame: pd.DataFrame, view: str, count_column: str) -> str:
    selected = frame[frame["transition_view"].astype(str) == view]
    if selected.empty:
        return ""
    values = pd.to_numeric(selected[count_column], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return ""
    return f"users/state={int(values.min()):,}–{int(values.max()):,}"


def table1_rows(
    empirical_bootstrap: pd.DataFrame,
    empirical_strict: pd.DataFrame,
    empirical_transition: pd.DataFrame,
    residence_bootstrap: pd.DataFrame,
    residence_formal: pd.DataFrame,
    model_bootstrap: pd.DataFrame,
    model_strict: pd.DataFrame,
    model_transition_users: pd.DataFrame,
    stage5_bootstrap: pd.DataFrame,
    stage5_strict: pd.DataFrame,
    stage5_transition_users: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "Section",
        "Analysis object",
        "Metric",
        "Formal estimate",
        "95% learner interval",
        "Strict user-equal estimate",
        "Strict − formal",
        "Support / replicates",
        "Result",
    ]
    rows: List[Dict[str, str]] = []
    empirical_index = empirical_bootstrap.set_index("metric")
    strict_empirical_index = empirical_strict.set_index("metric")
    for metric in EMPIRICAL_METRICS:
        if metric not in empirical_index.index or metric not in strict_empirical_index.index:
            raise RuntimeError(f"Required empirical metric is absent: {metric}")
        boot = empirical_index.loc[metric]
        strict = strict_empirical_index.loc[metric]
        strict_value = finite(strict["strict_user_equal_value"])
        rows.append(
            {
                "Section": "Empirical field",
                "Analysis object": "Training–validation effective dynamics",
                "Metric": metric_label(metric),
                "Formal estimate": fmt(boot["formal_point_estimate"]),
                "95% learner interval": f"[{fmt(boot['ci_2p5'])}, {fmt(boot['ci_97p5'])}]",
                "Strict user-equal estimate": fmt(strict_value),
                "Strict − formal": fmt(strict["strict_minus_formal"]),
                "Support / replicates": f"finite replicates={fmt_count(boot['replicates_finite'])}",
                "Result": claim_status(metric, boot["ci_2p5"], boot["ci_97p5"], strict_value),
            }
        )

    require_columns(empirical_transition, ["macrostate", "formal_self_transition", "strict_user_equal_self_transition", "strict_minus_formal", "strict_contributing_users", "formal_diagonal_dominant", "strict_diagonal_dominant"], "empirical transition sensitivity")
    transition_index = empirical_transition.copy()
    transition_index["macrostate"] = pd.to_numeric(transition_index["macrostate"], errors="raise").astype(int)
    transition_index = transition_index.set_index("macrostate")
    require_columns(residence_bootstrap, ["macrostate", "metric", "formal_point_estimate", "ci_2p5", "ci_97p5", "replicates_finite"], "residence bootstrap")
    residence_lookup = residence_bootstrap.set_index(["macrostate", "metric"])
    residence_formal = residence_formal.copy()
    residence_formal["macrostate"] = pd.to_numeric(residence_formal["macrostate"], errors="raise").astype(int)
    residence_formal = residence_formal.set_index("macrostate")
    for state in range(6):
        for metric in ("rmst_lift", "self_transition", "tail_excess"):
            key = (state, metric)
            if key not in residence_lookup.index:
                raise RuntimeError(f"Required residence metric is absent: state={state}, metric={metric}")
            boot = residence_lookup.loc[key]
            formal_row = residence_formal.loc[state] if state in residence_formal.index else pd.Series(dtype=object)
            support_parts = []
            for column, label in (
                ("n_runs", "runs"),
                ("n_right_censored", "censored"),
                ("rmst_tau", "τ"),
                ("reference_length", "reference"),
                ("reference_at_risk", "at risk"),
            ):
                if column in formal_row.index and np.isfinite(finite(formal_row[column])):
                    support_parts.append(f"{label}={fmt_count(formal_row[column])}")
            support_parts.append(f"finite replicates={fmt_count(boot['replicates_finite'])}")
            strict_value = np.nan
            strict_delta = np.nan
            result = claim_status(metric, boot["ci_2p5"], boot["ci_97p5"])
            if metric == "self_transition":
                if state not in transition_index.index:
                    raise RuntimeError(f"Missing strict transition state: {state}")
                transition = transition_index.loc[state]
                strict_value = finite(transition["strict_user_equal_self_transition"])
                strict_delta = finite(transition["strict_minus_formal"])
                support_parts.append(f"strict users={fmt_count(transition['strict_contributing_users'])}")
                formal_diag = bool_value(transition["formal_diagonal_dominant"])
                strict_diag = bool_value(transition["strict_diagonal_dominant"])
                result = f"Diagonal dominance {'retained' if formal_diag == strict_diag else 'changed'} ({'yes' if formal_diag else 'no'}→{'yes' if strict_diag else 'no'})"
            rows.append(
                {
                    "Section": "Residence kinetics",
                    "Analysis object": f"Empirical mesostate S{state}",
                    "Metric": metric_label(metric),
                    "Formal estimate": fmt(boot["formal_point_estimate"]),
                    "95% learner interval": f"[{fmt(boot['ci_2p5'])}, {fmt(boot['ci_97p5'])}]",
                    "Strict user-equal estimate": fmt(strict_value),
                    "Strict − formal": fmt(strict_delta),
                    "Support / replicates": "; ".join(support_parts),
                    "Result": result,
                }
            )

    model_index = model_bootstrap.set_index("metric")
    model_strict_index = model_strict[model_strict["metric"].astype(str) != "maximum_absolute_user_mass_minus_one"].set_index("metric")
    for metric in MODEL_METRICS:
        if metric not in model_index.index:
            raise RuntimeError(f"Required model metric is absent: {metric}")
        boot = model_index.loc[metric]
        strict_value = np.nan
        strict_delta = np.nan
        strict_text = "—"
        delta_text = "—"
        if metric in model_strict_index.index:
            strict_row = model_strict_index.loc[metric]
            strict_value = finite(strict_row["strict_user_equal_value"])
            strict_delta = finite(strict_row["strict_minus_formal"])
            strict_text = fmt(strict_value)
            delta_text = fmt(strict_delta)
        elif metric in MODEL_FIELD_METRICS:
            strict_value = finite(boot["formal_point_estimate"])
            strict_delta = 0.0
            strict_text = f"{fmt(strict_value)} (same by construction)"
            delta_text = "0"
        support_parts = [f"finite replicates={fmt_count(boot['replicates_finite'])}"]
        for view in MODEL_TRANSITION_VIEW.get(metric, ()):
            text = support_range(model_transition_users, view, "strict_contributing_users")
            if text:
                support_parts.append(f"{view}: {text}")
        if "self_transition" in metric:
            support_parts.append("six frozen state probabilities")
        rows.append(
            {
                "Section": "Frozen models and controls",
                "Analysis object": "Confirmation-set frozen outputs",
                "Metric": metric_label(metric),
                "Formal estimate": fmt(boot["formal_point_estimate"]),
                "95% learner interval": f"[{fmt(boot['ci_2p5'])}, {fmt(boot['ci_97p5'])}]",
                "Strict user-equal estimate": strict_text,
                "Strict − formal": delta_text,
                "Support / replicates": "; ".join(support_parts),
                "Result": claim_status(metric, boot["ci_2p5"], boot["ci_97p5"], strict_value),
            }
        )

    require_columns(stage5_bootstrap, ["comparison", "representation", "metric", "formal_point_estimate", "ci_2p5", "ci_97p5", "replicates_finite"], "Stage-5 bootstrap")
    stage5_strict_index = stage5_strict[stage5_strict["metric"].astype(str) != "maximum_absolute_user_mass_minus_one"].set_index(["representation", "metric"])
    absolute = stage5_bootstrap[stage5_bootstrap["comparison"].astype(str) == "absolute"].set_index(["representation", "metric"])
    for representation in REPRESENTATION_ORDER:
        for metric in STAGE5_METRICS:
            key = (representation, metric)
            if key not in absolute.index:
                raise RuntimeError(f"Required Stage-5 metric is absent: {key}")
            boot = absolute.loc[key]
            strict_value = np.nan
            strict_delta = np.nan
            if key in stage5_strict_index.index:
                strict_row = stage5_strict_index.loc[key]
                strict_value = finite(strict_row["strict_user_equal_value"])
                strict_delta = finite(strict_row["strict_minus_formal"])
                strict_text = fmt(strict_value)
                delta_text = fmt(strict_delta)
            elif metric in STAGE5_FIELD_METRICS:
                strict_value = finite(boot["formal_point_estimate"])
                strict_delta = 0.0
                strict_text = f"{fmt(strict_value)} (same by construction)"
                delta_text = "0"
            else:
                strict_text = "—"
                delta_text = "—"
            support_parts = [f"finite replicates={fmt_count(boot['replicates_finite'])}"]
            if metric.startswith("learned_plane_") and metric not in STAGE5_FIELD_METRICS:
                text = support_range(stage5_transition_users, representation, "contributing_users")
                if text:
                    support_parts.append(text)
            if metric == "learned_plane_self_transition_corr":
                support_parts.append("six frozen state probabilities")
            rows.append(
                {
                    "Section": "Stage-5 raw metrics",
                    "Analysis object": REPRESENTATION_LABELS[representation],
                    "Metric": metric_label(metric),
                    "Formal estimate": fmt(boot["formal_point_estimate"]),
                    "95% learner interval": f"[{fmt(boot['ci_2p5'])}, {fmt(boot['ci_97p5'])}]",
                    "Strict user-equal estimate": strict_text,
                    "Strict − formal": delta_text,
                    "Support / replicates": "; ".join(support_parts),
                    "Result": "Raw metric; primary evidence",
                }
            )

    contrast = stage5_bootstrap[stage5_bootstrap["comparison"].astype(str) == "macro_only_minus_full_hidden"].set_index("metric")
    for metric in STAGE5_METRICS:
        if metric not in contrast.index:
            raise RuntimeError(f"Required Stage-5 paired contrast is absent: {metric}")
        boot = contrast.loc[metric]
        macro_key = ("macro_only", metric)
        full_key = ("full_hidden", metric)
        if macro_key in stage5_strict_index.index and full_key in stage5_strict_index.index:
            strict_value = finite(stage5_strict_index.loc[macro_key]["strict_user_equal_value"]) - finite(stage5_strict_index.loc[full_key]["strict_user_equal_value"])
        elif metric in STAGE5_FIELD_METRICS:
            strict_value = finite(boot["formal_point_estimate"])
        else:
            strict_value = np.nan
        strict_delta = strict_value - finite(boot["formal_point_estimate"]) if np.isfinite(strict_value) else np.nan
        direction = "Positive means greater error" if metric in LOWER_IS_BETTER else "Positive means greater recovery"
        rows.append(
            {
                "Section": "Stage-5 paired contrast",
                "Analysis object": REPRESENTATION_LABELS["macro_only_minus_full_hidden"],
                "Metric": metric_label(metric),
                "Formal estimate": fmt(boot["formal_point_estimate"]),
                "95% learner interval": f"[{fmt(boot['ci_2p5'])}, {fmt(boot['ci_97p5'])}]",
                "Strict user-equal estimate": fmt(strict_value),
                "Strict − formal": fmt(strict_delta),
                "Support / replicates": f"paired finite replicates={fmt_count(boot['replicates_finite'])}",
                "Result": direction,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def coordinate_setting_text(row: Mapping[str, Any]) -> str:
    return (
        f"τR/τA={fmt(row.get('tau_response_days'))}/{fmt(row.get('tau_activity_days'))} d; "
        f"response/explanation/lecture={fmt(row.get('response_half_sat_min'))}/"
        f"{fmt(row.get('explanation_half_sat_min'))}/{fmt(row.get('lecture_half_sat_min'))} min; "
        f"idle={fmt(row.get('idle_half_sat_days'))} d"
    )


def grid_result_text(row: Mapping[str, Any]) -> str:
    return (
        f"cells={fmt_count(row.get('common_supported_cells'))}; "
        f"r={fmt(row.get('drift_vector_corr'))}; "
        f"cos={fmt(row.get('mean_local_drift_cosine'))}; "
        f"cos_w={fmt(row.get('occupancy_weighted_local_drift_cosine'))}; "
        f"r_speed={fmt(row.get('drift_speed_corr'))}; "
        f"RMSE={fmt(row.get('drift_component_rmse'))}"
    )


def grid_baseline_lookup(frame: pd.DataFrame, object_column: Optional[str]) -> Dict[str, Mapping[str, Any]]:
    output: Dict[str, Mapping[str, Any]] = {}
    selected = frame[(frame["setting"].astype(str) == "40x40") & (~frame.get("interior_only", frame.get("interior_only_comparison", False)).map(bool_value))]
    for row in selected.to_dict(orient="records"):
        key = str(row.get(object_column, "single")) if object_column else "single"
        output[key] = row
    return output


def table2_rows(
    coordinate_statistics: pd.DataFrame,
    empirical_grid: pd.DataFrame,
    model_grid: pd.DataFrame,
    stage5_grid: pd.DataFrame,
    permutation_floor: pd.DataFrame,
    score_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "Section",
        "Analysis object",
        "Setting",
        "Metric family",
        "Observed result",
        "Reference / floor",
        "Sensitivity contrast",
        "Support",
        "Interpretation",
    ]
    rows: List[Dict[str, str]] = []
    coordinate_required = [
        "setting_id",
        "setting",
        "analysis",
        "A_val_users",
        "A_val_valid_state_rows",
        "A_val_valid_drift_rows",
        "A_val_state_supported_cells",
        "A_val_drift_supported_cells",
        "A_val_interior_divergence_cells",
        "A_train_A_val_occupancy_js",
        "A_train_A_val_common_drift_cells",
        "A_train_A_val_field_vector_correlation",
        "A_train_A_val_mean_local_drift_cosine",
        "A_train_A_val_occupancy_weighted_local_drift_cosine",
        "A_train_A_val_drift_speed_correlation",
        "A_train_A_val_drift_component_rmse",
        "A_val_occupancy_js_vs_formal",
        "A_val_common_drift_cells_vs_formal",
        "A_val_field_vector_correlation_vs_formal",
        "A_val_mean_local_drift_cosine_vs_formal",
        "A_val_occupancy_weighted_local_drift_cosine_vs_formal",
        "A_val_drift_speed_correlation_vs_formal",
        "A_val_drift_component_rmse_vs_formal",
        "A_val_negative_divergence_occupancy_fraction",
        "A_val_weighted_mean_local_divergence",
        "A_val_frozen_core_occupancy_mass_fraction",
        "A_val_frozen_shell_inward_fraction",
        "A_val_frozen_shell_inward_cosine",
        "A_val_frozen_core_to_shell_speed_ratio",
        "A_train_core_center_M",
        "A_train_core_center_Psi",
        "A_train_core_cells",
        "A_train_core_dynamically_qualified",
        "A_train_core_fallback_used",
        "directional_result_retained",
    ]
    require_columns(coordinate_statistics, coordinate_required, "coordinate sensitivity statistics")
    order = {name: index for index, name in enumerate(("formal_10d", "memory_5d", "memory_20d", "activity_fast", "activity_slow"))}
    coordinate_statistics = coordinate_statistics.assign(_order=coordinate_statistics["setting_id"].astype(str).map(order)).sort_values("_order")
    for row in coordinate_statistics.to_dict(orient="records"):
        replication = (
            f"train–validation JS={fmt(row['A_train_A_val_occupancy_js'])}; "
            f"r={fmt(row['A_train_A_val_field_vector_correlation'])}; "
            f"cos={fmt(row['A_train_A_val_mean_local_drift_cosine'])}; "
            f"cos_w={fmt(row['A_train_A_val_occupancy_weighted_local_drift_cosine'])}; "
            f"r_speed={fmt(row['A_train_A_val_drift_speed_correlation'])}; "
            f"RMSE={fmt(row['A_train_A_val_drift_component_rmse'])}"
        )
        versus_formal = (
            f"validation vs formal JS={fmt(row['A_val_occupancy_js_vs_formal'])}; "
            f"r={fmt(row['A_val_field_vector_correlation_vs_formal'])}; "
            f"cos={fmt(row['A_val_mean_local_drift_cosine_vs_formal'])}; "
            f"cos_w={fmt(row['A_val_occupancy_weighted_local_drift_cosine_vs_formal'])}; "
            f"r_speed={fmt(row['A_val_drift_speed_correlation_vs_formal'])}; "
            f"RMSE={fmt(row['A_val_drift_component_rmse_vs_formal'])}"
        )
        support = (
            f"users={fmt_count(row['A_val_users'])}; state/drift rows={fmt_count(row['A_val_valid_state_rows'])}/"
            f"{fmt_count(row['A_val_valid_drift_rows'])}; state/drift cells={fmt_count(row['A_val_state_supported_cells'])}/"
            f"{fmt_count(row['A_val_drift_supported_cells'])}; common cells={fmt_count(row['A_train_A_val_common_drift_cells'])}; "
            f"interior cells={fmt_count(row['A_val_interior_divergence_cells'])}"
        )
        rows.append(
            {
                "Section": "Coordinate sensitivity",
                "Analysis object": "Empirical field replication",
                "Setting": f"{row['setting']} ({coordinate_setting_text(row)})",
                "Metric family": "Occupancy and drift replication",
                "Observed result": replication,
                "Reference / floor": "Formal coordinate setting" if row["setting_id"] != "formal_10d" else "Reference setting",
                "Sensitivity contrast": versus_formal,
                "Support": support,
                "Interpretation": "Predeclared coordinate setting; no downstream refit",
            }
        )
        basin = (
            f"f_neg={fmt(row['A_val_negative_divergence_occupancy_fraction'])}; "
            f"mean div={fmt(row['A_val_weighted_mean_local_divergence'])}; "
            f"core mass={fmt(row['A_val_frozen_core_occupancy_mass_fraction'])}; "
            f"f_in={fmt(row['A_val_frozen_shell_inward_fraction'])}; "
            f"cos_in={fmt(row['A_val_frozen_shell_inward_cosine'])}; "
            f"R_core/shell={fmt(row['A_val_frozen_core_to_shell_speed_ratio'])}"
        )
        core = (
            f"A_train core=({fmt(row['A_train_core_center_M'])}, {fmt(row['A_train_core_center_Psi'])}); "
            f"cells={fmt_count(row['A_train_core_cells'])}; qualified={'yes' if bool_value(row['A_train_core_dynamically_qualified']) else 'no'}; "
            f"fallback={'yes' if bool_value(row['A_train_core_fallback_used']) else 'no'}"
        )
        rows.append(
            {
                "Section": "Coordinate sensitivity",
                "Analysis object": "Empirical contraction and basin",
                "Setting": f"{row['setting']} ({coordinate_setting_text(row)})",
                "Metric family": "Global contraction and frozen validation basin",
                "Observed result": basin,
                "Reference / floor": "f_neg>0.5; f_in>0.5; R_core/shell<1",
                "Sensitivity contrast": core,
                "Support": support,
                "Interpretation": "Retained" if bool_value(row["directional_result_retained"]) else "Not retained",
            }
        )

    grid_frames = (
        ("Empirical grid", empirical_grid, None, {"single": "A_train–A_val empirical field"}),
        (
            "Frozen-model grid",
            model_grid,
            "comparison",
            {
                "mechanism_vs_empirical": "Minimal mechanism vs empirical field",
                "event_ssl_learned_vs_empirical": "Event-SSL learned plane vs empirical field",
                "mechanism_vs_event_ssl_anchor": "Minimal mechanism vs Event-SSL empirical-anchor field",
            },
        ),
        (
            "Stage-5 grid",
            stage5_grid,
            "representation",
            {name: REPRESENTATION_LABELS[name] for name in REPRESENTATION_ORDER},
        ),
    )
    for section, frame, object_column, labels in grid_frames:
        require_columns(frame, ["setting", "common_supported_cells", "drift_vector_corr", "mean_local_drift_cosine", "occupancy_weighted_local_drift_cosine", "drift_speed_corr", "drift_component_rmse"], section)
        baselines = grid_baseline_lookup(frame, object_column)
        for row in frame.to_dict(orient="records"):
            key = str(row.get(object_column, "single")) if object_column else "single"
            baseline = baselines.get(key)
            if baseline is None:
                raise RuntimeError(f"Missing 40x40 baseline for {section}: {key}")
            contrast = (
                f"Δr={fmt(finite(row['drift_vector_corr']) - finite(baseline['drift_vector_corr']))}; "
                f"Δcos_w={fmt(finite(row['occupancy_weighted_local_drift_cosine']) - finite(baseline['occupancy_weighted_local_drift_cosine']))}; "
                f"ΔRMSE={fmt(finite(row['drift_component_rmse']) - finite(baseline['drift_component_rmse']))}"
            )
            support_parts = [f"common cells={fmt_count(row['common_supported_cells'])}"]
            if section == "Empirical grid":
                for column, label in (
                    ("validation_supported_cells", "validation cells"),
                    ("validation_supported_occupancy_mass", "supported occupancy"),
                ):
                    if column in row:
                        support_parts.append(f"{label}={fmt(row[column]) if 'mass' in column else fmt_count(row[column])}")
                observed = grid_result_text(row)
                observed += f"; f_neg={fmt(row.get('validation_negative_divergence_fraction'))}; mean div={fmt(row.get('validation_weighted_mean_divergence'))}"
            else:
                observed = grid_result_text(row)
            sign_stable = np.sign(finite(row["drift_vector_corr"])) == np.sign(finite(baseline["drift_vector_corr"])) and np.sign(finite(row["occupancy_weighted_local_drift_cosine"])) == np.sign(finite(baseline["occupancy_weighted_local_drift_cosine"]))
            rows.append(
                {
                    "Section": section,
                    "Analysis object": labels.get(key, key),
                    "Setting": str(row["setting"]),
                    "Metric family": "Field agreement under fixed support thresholds",
                    "Observed result": observed,
                    "Reference / floor": grid_result_text(baseline),
                    "Sensitivity contrast": contrast,
                    "Support": "; ".join(support_parts),
                    "Interpretation": "Direction retained" if sign_stable else "Direction changed",
                }
            )

    require_columns(permutation_floor, ["representation", "metric", "metric_direction", "observed", "null_replicates_finite", "null_median", "null_5pct", "null_95pct", "observed_improvement_over_null_median"], "permutation floor")
    representation_rank = {name: index for index, name in enumerate(REPRESENTATION_ORDER)}
    metric_rank = {name: index for index, name in enumerate(STAGE5_METRICS)}
    floor_sorted = permutation_floor.assign(
        _r=permutation_floor["representation"].astype(str).map(representation_rank),
        _m=permutation_floor["metric"].astype(str).map(metric_rank),
    ).sort_values(["_r", "_m"])
    for row in floor_sorted.to_dict(orient="records"):
        improvement = finite(row["observed_improvement_over_null_median"])
        rows.append(
            {
                "Section": "Stage-5 permutation floor",
                "Analysis object": REPRESENTATION_LABELS.get(str(row["representation"]), str(row["representation"])),
                "Setting": "Within-user marginal-preserving permutation",
                "Metric family": metric_label(str(row["metric"])),
                "Observed result": fmt(row["observed"]),
                "Reference / floor": f"median={fmt(row['null_median'])}; 5–95%=[{fmt(row['null_5pct'])}, {fmt(row['null_95pct'])}]",
                "Sensitivity contrast": f"oriented improvement={fmt(improvement)}",
                "Support": f"finite permutations={fmt_count(row['null_replicates_finite'])}",
                "Interpretation": "Above descriptive floor" if np.isfinite(improvement) and improvement > 0 else "Not above descriptive floor",
            }
        )

    require_columns(score_sensitivity, ["contract", "rmse_scale", "included_domains", "quantity", "representation", "value"], "score sensitivity")
    score_lookup = score_sensitivity.set_index(["contract", "quantity", "representation"])["value"]
    contracts = []
    for contract in score_sensitivity["contract"].astype(str).drop_duplicates().tolist():
        selected = score_sensitivity[score_sensitivity["contract"].astype(str) == contract]
        contracts.append((contract, finite(selected.iloc[0]["rmse_scale"]), str(selected.iloc[0]["included_domains"])))
    for contract, scale, domains in contracts:
        def lookup(quantity: str, representation: str) -> float:
            key = (contract, quantity, representation)
            return finite(score_lookup.loc[key]) if key in score_lookup.index else np.nan

        observed = {representation: lookup("observed_composite", representation) for representation in REPRESENTATION_ORDER}
        null_median = {representation: lookup("null_median_composite", representation) for representation in REPRESENTATION_ORDER}
        null_5 = {representation: lookup("null_5pct_composite", representation) for representation in REPRESENTATION_ORDER}
        null_95 = {representation: lookup("null_95pct_composite", representation) for representation in REPRESENTATION_ORDER}
        raw_retention = lookup("macro_retention_vs_full", "macro_only_vs_full_hidden")
        headroom_retention = lookup("macro_null_headroom_retention_vs_full", "macro_only_vs_full_hidden")
        if not all(np.isfinite(value) for value in observed.values()) or not np.isfinite(raw_retention) or not np.isfinite(headroom_retention):
            raise RuntimeError(f"Incomplete score contract: {contract}")
        rows.append(
            {
                "Section": "Descriptive-score sensitivity",
                "Analysis object": "Full hidden / bottleneck / residual hidden",
                "Setting": f"{contract}; RMSE scale={fmt(scale)}; domains={domains}",
                "Metric family": "Composite and bottleneck retention",
                "Observed result": f"full={fmt(observed['full_hidden'])}; bottleneck={fmt(observed['macro_only'])}; residual={fmt(observed['residual_hidden'])}",
                "Reference / floor": (
                    f"full null={fmt(null_median['full_hidden'])} [{fmt(null_5['full_hidden'])}, {fmt(null_95['full_hidden'])}]; "
                    f"bottleneck null={fmt(null_median['macro_only'])} [{fmt(null_5['macro_only'])}, {fmt(null_95['macro_only'])}]; "
                    f"residual null={fmt(null_median['residual_hidden'])} [{fmt(null_5['residual_hidden'])}, {fmt(null_95['residual_hidden'])}]"
                ),
                "Sensitivity contrast": f"raw retention={fmt(raw_retention)}; null-headroom retention={fmt(headroom_retention)}",
                "Support": "Descriptive score; no model selection or p value",
                "Interpretation": "Raw metrics remain primary",
            }
        )

    formal_contract = "S0_formal"
    for domain in ("coordinate_score", "closure_score", "drift_score", "transition_score"):
        values = {}
        for representation in REPRESENTATION_ORDER:
            key = (formal_contract, domain, representation)
            if key not in score_lookup.index:
                raise RuntimeError(f"Missing formal domain score: {domain}, {representation}")
            values[representation] = finite(score_lookup.loc[key])
        ratio = values["macro_only"] / values["full_hidden"] if abs(values["full_hidden"]) > 1e-12 else np.nan
        rows.append(
            {
                "Section": "Formal descriptive domains",
                "Analysis object": "Full hidden / bottleneck / residual hidden",
                "Setting": "S0 formal descriptive mapping",
                "Metric family": domain.replace("_", " "),
                "Observed result": f"full={fmt(values['full_hidden'])}; bottleneck={fmt(values['macro_only'])}; residual={fmt(values['residual_hidden'])}",
                "Reference / floor": "Full-hidden domain score",
                "Sensitivity contrast": f"bottleneck/full={fmt(ratio)}",
                "Support": "Derived from raw held-out metrics",
                "Interpretation": "Descriptive decomposition",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_report(table1: pd.DataFrame, table2: pd.DataFrame, gates: pd.DataFrame, manifests: Mapping[str, Mapping[str, Any]]) -> str:
    contract_passed = bool(gates["passed"].map(bool_value).all()) if not gates.empty else False
    empirical_replicates = nested(manifests["empirical"], "bootstrap", "replicates", default=0)
    model_replicates = nested(manifests["model"], "bootstrap", "replicates", default=0)
    stage5_replicates = nested(manifests["representation"], "bootstrap", "replicates", default=0)
    permutation_replicates = nested(manifests["representation"], "permutation_floor", "replicates", default=0)
    return "\n".join(
        [
            "# Supplementary learner-level and evaluation-gauge robustness numerical report",
            "",
            "The report uses frozen empirical coordinates, models, probes, the training-defined convergence core and the fixed K=6 partition. Learner-multiplier intervals quantify learner-composition uncertainty conditional on that frozen contract and are not combined with random-seed intervals, mechanism-family selection bootstrap intervals or construction-null inference. Strict user-equal values are alternative estimands; Stage-4 and Stage-5 fields that are already user-equal are marked as identical by construction.",
            "",
            f"The empirical, frozen-model and Stage-5 analyses used {fmt_count(empirical_replicates)}, {fmt_count(model_replicates)} and {fmt_count(stage5_replicates)} learner-multiplier replicates, respectively. The Stage-5 floor used {fmt_count(permutation_replicates)} descriptive within-user permutations. No new p value is reported. Statewise self-transition intervals reflect learner-composition variation in six frozen probabilities and do not treat the six states as independent observations.",
            "",
            f"All extraction and formal-point equivalence gates passed: {'yes' if contract_passed else 'no'}. Raw coordinate, closure, drift and transition metrics are primary; composite retention and permutation-floor quantities are descriptive diagnostics.",
            "",
            "## Supplementary Table 1. Learner-level uncertainty and strict user-equal sensitivity",
            "",
            markdown_table(table1),
            "",
            "## Supplementary Table 2. Coordinate, grid, permutation-floor and descriptive-score sensitivity",
            "",
            markdown_table(table2),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    robustness_root = args.robustness_root.resolve()
    empirical_root = resolve_branch_root(
        args.empirical_root, robustness_root, "empirical", "empirical_robustness_manifest.json"
    )
    model_root = resolve_branch_root(
        args.model_root, robustness_root, "models", "model_robustness_manifest.json"
    )
    representation_root = resolve_branch_root(
        args.representation_root, robustness_root, "representations", "representation_robustness_manifest.json"
    )
    coordinate_root = args.coordinate_summary_root.resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir is not None else (robustness_root / "summary").resolve())
    logs_root = (args.logs_root.resolve() if args.logs_root is not None else (robustness_root / "logs").resolve())
    preflight_sources(
        empirical_root, model_root, representation_root, coordinate_root, logs_root
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = SourceRegistry()
    gates: List[Dict[str, Any]] = []

    empirical_manifest = read_json(empirical_root / "metadata" / "empirical_robustness_manifest.json", "empirical_manifest", registry)
    model_manifest = read_json(model_root / "metadata" / "model_robustness_manifest.json", "model_manifest", registry)
    representation_manifest = read_json(representation_root / "metadata" / "representation_robustness_manifest.json", "representation_manifest", registry)
    coordinate_manifest = read_json(coordinate_root / "empirical_coordinate_sensitivity_manifest.json", "coordinate_manifest", registry)

    empirical_equivalence = read_table(empirical_root / "tables" / "empirical_formal_point_equivalence_audit", "empirical_equivalence", registry)
    residence_equivalence = read_table(empirical_root / "tables" / "empirical_residence_formal_point_equivalence_audit", "residence_equivalence", registry)
    model_equivalence = read_table(model_root / "tables" / "model_formal_point_equivalence_audit", "model_equivalence", registry)
    stage5_equivalence = read_table(representation_root / "tables" / "stage5_formal_point_equivalence_audit", "stage5_equivalence", registry)
    validate_audit(empirical_equivalence, "empirical_formal_point_equivalence", gates)
    validate_audit(residence_equivalence, "residence_formal_point_equivalence", gates)
    validate_audit(model_equivalence, "model_formal_point_equivalence", gates)
    validate_audit(stage5_equivalence, "stage5_formal_point_equivalence", gates)

    empirical_bootstrap = read_table(empirical_root / "tables" / "empirical_user_multiplier_bootstrap_summary", "empirical_bootstrap_summary", registry)
    empirical_strict = read_table(empirical_root / "tables" / "empirical_strict_user_equal_sensitivity", "empirical_strict_user_equal", registry)
    empirical_transition = read_table(empirical_root / "tables" / "empirical_transition_user_equal_sensitivity", "empirical_transition_user_equal", registry)
    residence_bootstrap = read_table(empirical_root / "tables" / "empirical_residence_user_multiplier_bootstrap_summary", "residence_bootstrap_summary", registry)
    residence_source = Path(str(nested(empirical_manifest, "source_audit", "A_val_residence_summary", "path", default="")))
    if not residence_source.is_file():
        stage1_root = Path(str(empirical_manifest.get("stage1_root", "")))
        residence_source = resolve_table(stage1_root / "dynamics" / "fixed_k6_mesostates" / "A_val_fixed_k6_residence_summary")
    residence_formal = read_table(residence_source, "formal_residence_summary", registry)

    model_bootstrap = read_table(model_root / "tables" / "model_user_multiplier_bootstrap_summary", "model_bootstrap_summary", registry)
    model_strict = read_table(model_root / "tables" / "model_strict_user_equal_sensitivity", "model_strict_user_equal", registry)
    model_transition_users = read_table(model_root / "tables" / "model_strict_transition_contributing_users", "model_transition_users", registry)

    stage5_bootstrap = read_table(representation_root / "tables" / "stage5_user_multiplier_bootstrap_summary", "stage5_bootstrap_summary", registry)
    stage5_strict = read_table(representation_root / "tables" / "stage5_strict_user_equal_sensitivity", "stage5_strict_user_equal", registry)
    stage5_transition_users = read_table(representation_root / "tables" / "stage5_strict_transition_contributing_users", "stage5_transition_users", registry)
    permutation_floor = read_table(representation_root / "tables" / "stage5_within_user_permutation_floor_summary", "stage5_permutation_floor", registry)
    score_sensitivity = read_table(representation_root / "tables" / "stage5_descriptive_score_sensitivity", "stage5_score_sensitivity", registry)

    coordinate_statistics_path = coordinate_root / "empirical_coordinate_sensitivity_statistics.csv"
    coordinate_statistics = pd.read_csv(coordinate_statistics_path, low_memory=False)
    if "setting" not in coordinate_statistics.columns and "label" in coordinate_statistics.columns:
        coordinate_statistics = coordinate_statistics.rename(columns={"label": "setting"})
    registry.add("coordinate_statistics", coordinate_statistics_path, len(coordinate_statistics), list(coordinate_statistics.columns))
    coordinate_gates_path = coordinate_root / "empirical_coordinate_sensitivity_quality_gates.csv"
    coordinate_gates = None
    if coordinate_gates_path.exists():
        coordinate_gates = pd.read_csv(coordinate_gates_path, low_memory=False)
        registry.add("coordinate_quality_gates", coordinate_gates_path, len(coordinate_gates), list(coordinate_gates.columns))

    empirical_grid = read_table(empirical_root / "tables" / "empirical_grid_sensitivity", "empirical_grid_sensitivity", registry)
    model_grid = read_table(model_root / "tables" / "model_grid_sensitivity", "model_grid_sensitivity", registry)
    stage5_grid = read_table(representation_root / "tables" / "stage5_grid_sensitivity", "stage5_grid_sensitivity", registry)

    validate_inputs(
        empirical_manifest,
        model_manifest,
        representation_manifest,
        coordinate_statistics,
        coordinate_gates,
        empirical_bootstrap,
        residence_bootstrap,
        model_bootstrap,
        stage5_bootstrap,
        permutation_floor,
        model_strict,
        stage5_strict,
        gates,
    )

    table1 = table1_rows(
        empirical_bootstrap,
        empirical_strict,
        empirical_transition,
        residence_bootstrap,
        residence_formal,
        model_bootstrap,
        model_strict,
        model_transition_users,
        stage5_bootstrap,
        stage5_strict,
        stage5_transition_users,
    )
    table2 = table2_rows(
        coordinate_statistics,
        empirical_grid,
        model_grid,
        stage5_grid,
        permutation_floor,
        score_sensitivity,
    )

    assert_gate(gates, "output", "table1_nonempty", not table1.empty, f"rows={len(table1)}")
    assert_gate(gates, "output", "table2_nonempty", not table2.empty, f"rows={len(table2)}")
    gate_frame = pd.DataFrame(gates)
    if not bool(gate_frame["passed"].map(bool_value).all()):
        raise RuntimeError("One or more report quality gates failed.")

    table1_path = output_dir / "supplementary_robustness_table1_learner_uncertainty.csv"
    table2_path = output_dir / "supplementary_robustness_table2_sensitivity.csv"
    report_path = output_dir / "supplementary_robustness_numerical_report.md"
    gates_path = output_dir / "supplementary_robustness_quality_gates.csv"
    source_path = output_dir / "supplementary_robustness_source_audit.csv"
    manifest_path = output_dir / "supplementary_robustness_report_manifest.json"

    table1.to_csv(table1_path, index=False)
    table2.to_csv(table2_path, index=False)
    gate_frame.to_csv(gates_path, index=False)
    source_frame = pd.DataFrame(registry.records)
    source_frame.to_csv(source_path, index=False)
    report_path.write_text(
        build_report(
            table1,
            table2,
            gate_frame,
            {
                "empirical": empirical_manifest,
                "model": model_manifest,
                "representation": representation_manifest,
                "coordinate": coordinate_manifest,
            },
        ),
        encoding="utf-8",
    )
    manifest = {
        "script": Path(__file__).name,
        "analysis_boundary": {
            "models_retrained": False,
            "probes_refit": False,
            "mechanism_refit": False,
            "fixed_k6_partition_refit": False,
            "convergence_core_reselected": False,
            "construction_matched_null_rerun": False,
            "new_p_values": False,
            "coordinate_sensitivity_rerun": False,
            "raw_stage5_metrics_primary": True,
            "report_table_count": 2,
        },
        "input_manifests": {
            "empirical": str((empirical_root / "metadata" / "empirical_robustness_manifest.json").resolve()),
            "models": str((model_root / "metadata" / "model_robustness_manifest.json").resolve()),
            "representations": str((representation_root / "metadata" / "representation_robustness_manifest.json").resolve()),
            "coordinate_sensitivity": str((coordinate_root / "empirical_coordinate_sensitivity_manifest.json").resolve()),
        },
        "outputs": {
            "table1": str(table1_path.resolve()),
            "table2": str(table2_path.resolve()),
            "report": str(report_path.resolve()),
            "quality_gates": str(gates_path.resolve()),
            "source_audit": str(source_path.resolve()),
        },
        "row_counts": {"table1": int(len(table1)), "table2": int(len(table2))},
        "source_count": int(len(source_frame)),
    }
    save_json(manifest, manifest_path)
    print(f"[supplementary robustness report] wrote {report_path}")


if __name__ == "__main__":
    main()
