#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from semantic_specificity_common import (
    CONTROL_COORDINATE,
    CONTROL_INTERPRETATION,
    CONTROL_STATE_LABEL,
    coerce_bool,
    field_replication_metrics,
    json_safe,
    load_json,
    read_table,
    save_json,
    sha256_file,
    table_path,
    write_table,
)

FORMAL_COORDINATE = "MR_PsiA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the exposure-alignment state against the denominator-matched content-alignment-free activity--idle comparator."
    )
    parser.add_argument(
        "--stage1-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/stage1"),
    )
    parser.add_argument(
        "--formal-null-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/stage1_construction_matched_null"),
    )
    parser.add_argument(
        "--formal-confirm-null-root",
        type=Path,
        default=None,
        help="Root containing the B_confirm formal construction-null manifest; defaults to --formal-null-root.",
    )
    parser.add_argument(
        "--coordinate-output-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/semantic_specificity_control/coordinate"),
    )
    parser.add_argument(
        "--nonsemantic-null-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/semantic_specificity_control/null"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/semantic_specificity_control/summary"),
    )
    parser.add_argument(
        "--protocol-freeze",
        type=Path,
        required=False,
        help="Protocol freeze created before B_confirm was read by the new control.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def formal_manifest_path(args: argparse.Namespace, split: str) -> Path:
    root = (
        args.formal_confirm_null_root.resolve()
        if split == "B_confirm" and args.formal_confirm_null_root is not None
        else args.formal_null_root.resolve()
    )
    return root / "metadata" / f"{split}_construction_null_manifest.json"


def input_roots(args: argparse.Namespace) -> Dict[str, str]:
    confirm_root = (
        args.formal_confirm_null_root.resolve()
        if args.formal_confirm_null_root is not None
        else args.formal_null_root.resolve()
    )
    return {
        "stage1_root": str(args.stage1_root.resolve()),
        "formal_null_root": str(args.formal_null_root.resolve()),
        "formal_confirm_null_root": str(confirm_root),
        "coordinate_output_root": str(args.coordinate_output_root.resolve()),
        "nonsemantic_null_root": str(args.nonsemantic_null_root.resolve()),
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def require_manifest_source(
    consumer_manifest: Mapping[str, Any],
    path_key: str,
    sha_key: str,
    expected_path: Path,
    label: str,
) -> None:
    recorded_path = str(consumer_manifest.get(path_key, "") or "")
    recorded_sha = str(consumer_manifest.get(sha_key, "") or "")
    if not recorded_path or Path(recorded_path).resolve() != expected_path.resolve():
        raise RuntimeError(f"{label} references a different source path.")
    if not recorded_sha or recorded_sha != sha256_file(expected_path):
        raise RuntimeError(f"{label} references a different source checksum.")


def field_path(root: Path, coordinate: str, split: str) -> Path:
    suffix = "_output_only" if split == "B_confirm" else ""
    return (
        root
        / "dynamics"
        / "coordinate_analysis"
        / coordinate
        / f"{split}_publication_field_grid{suffix}"
    )


def region_root(root: Path, coordinate: str) -> Path:
    return root / "dynamics" / "candidate_regions" / coordinate


def primary_core_metadata(root: Path, coordinate: str) -> Dict[str, Any]:
    table = read_table(region_root(root, coordinate) / "training_flow_defined_convergence_regions")
    if table.empty:
        raise RuntimeError(f"No training core for {coordinate}")
    if "primary_convergence_region" in table.columns:
        selected = table[table["primary_convergence_region"].map(coerce_bool)]
        row = selected.iloc[0] if not selected.empty else table.iloc[0]
    else:
        row = table.iloc[0]
    return {
        "fallback_used": coerce_bool(row.get("flow_defined_fallback_used", False)),
        "dynamically_qualified": coerce_bool(row.get("dynamically_qualified", False)),
        "center_M": finite(row.get("convergence_center_M")),
        "center_second_axis": finite(row.get("convergence_center_Psi")),
        "region_cells": finite(row.get("region_cells_total")),
    }


def normalize_full_field_row(row: Mapping[str, Any]) -> Dict[str, float]:
    observed = finite(row.get("observed"))
    null_mean = finite(row.get("null_mean"))
    null_sd = finite(row.get("null_sd"))
    null_median = finite(row.get("null_50"))
    ratio = finite(row.get("observed_to_null_median_ratio"))
    if not math.isfinite(ratio) and math.isfinite(observed) and math.isfinite(null_median) and abs(null_median) > 1e-12:
        ratio = observed / null_median
    standardized = finite(row.get("null_standardized_separation"))
    if not math.isfinite(standardized) and math.isfinite(observed) and math.isfinite(null_mean) and math.isfinite(null_sd) and null_sd > 1e-12:
        standardized = (observed - null_mean) / null_sd
    return {
        "full_field_observed_distance": observed,
        "full_field_null_mean": null_mean,
        "full_field_null_sd": null_sd,
        "full_field_null_2p5": finite(row.get("null_2p5")),
        "full_field_null_median": null_median,
        "full_field_null_97p5": finite(row.get("null_97p5")),
        "full_field_observed_to_null_median_ratio": ratio,
        "full_field_standardized_separation": standardized,
        "full_field_monte_carlo_p": finite(row.get("monte_carlo_p")),
    }


def basin_row(manifest: Mapping[str, Any], metric: str) -> Dict[str, float]:
    rows = manifest.get("basin_metric_tests", [])
    if not isinstance(rows, list):
        rows = []
    selected = next((row for row in rows if str(row.get("metric")) == metric), {})
    return {
        "observed": finite(selected.get("observed")),
        "null_mean": finite(selected.get("null_mean")),
        "null_2p5": finite(selected.get("null_2p5")),
        "null_97p5": finite(selected.get("null_97p5")),
        "p": finite(selected.get("monte_carlo_p")),
        "q": finite(selected.get("BH_q_across_three_basin_metrics")),
    }


def supported_field_rms(field_grid: pd.DataFrame) -> float:
    required = {"drift_M", "drift_Psi", "drift_supported", "occupancy_probability"}
    missing = sorted(required.difference(field_grid.columns))
    if missing:
        raise RuntimeError(f"Field grid is missing columns for RMS normalization: {missing}")
    supported = field_grid["drift_supported"].map(coerce_bool).to_numpy(dtype=bool)
    drift_m = pd.to_numeric(field_grid["drift_M"], errors="coerce").to_numpy(dtype=float)
    drift_second = pd.to_numeric(field_grid["drift_Psi"], errors="coerce").to_numpy(dtype=float)
    weight = pd.to_numeric(field_grid["occupancy_probability"], errors="coerce").to_numpy(dtype=float)
    valid = supported & np.isfinite(drift_m) & np.isfinite(drift_second) & np.isfinite(weight) & (weight >= 0)
    if not np.any(valid):
        return float("nan")
    normalized = weight[valid] / max(float(np.sum(weight[valid])), 1e-12)
    return float(np.sqrt(np.sum(normalized * (drift_m[valid] ** 2 + drift_second[valid] ** 2))))


def build_state_row(
    state_name: str,
    second_axis: str,
    semantic_alignment: bool,
    split: str,
    coordinate_root: Path,
    coordinate: str,
    null_manifest: Mapping[str, Any],
    core: Mapping[str, Any],
    coordinate_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    training = read_table(field_path(coordinate_root, coordinate, "A_train"))
    held = read_table(field_path(coordinate_root, coordinate, split))
    replication = field_replication_metrics(training, held)
    full = normalize_full_field_row(null_manifest.get("full_field_primary_test", {}))
    held_rms = supported_field_rms(held)
    departure_fraction = (
        full["full_field_observed_distance"] / held_rms
        if math.isfinite(full["full_field_observed_distance"]) and math.isfinite(held_rms) and held_rms > 1e-12
        else float("nan")
    )
    negative = basin_row(null_manifest, "negative_divergence_occupancy_fraction")
    inward = basin_row(null_manifest, "flow_weighted_shell_fraction_inward")
    speed = basin_row(null_manifest, "flow_core_to_shell_speed_ratio")

    split_audit = coordinate_manifest.get("split_coordinate_audits", {}).get(split, {})
    axis_audit = split_audit.get("axis_distribution_audit", {})
    axis_key = "formal_Psi" if semantic_alignment else "alignment_free_Phi"
    axis_summary = axis_audit.get(axis_key, {})
    permutation_summary = null_manifest.get("permutation_value_change_summary", {})
    innovation_audit = null_manifest.get("innovation_informativeness_audit", {})

    return {
        "state_definition": state_name,
        "coordinate": coordinate,
        "first_axis": "M response order",
        "second_axis": second_axis,
        "second_axis_uses_content_demand_alignment": bool(semantic_alignment),
        "comparator_removes_all_event_semantics": False if not semantic_alignment else np.nan,
        "split": split,
        "split_role": "primary post hoc validation comparison" if split == "A_val" else "post hoc output-only replication after protocol freeze",
        **{f"train_held_{key}": value for key, value in replication.items()},
        **full,
        "held_supported_field_rms": held_rms,
        "full_field_departure_fraction_of_held_field_rms": departure_fraction,
        "second_axis_sd": finite(axis_summary.get("sd")),
        "second_axis_iqr": finite(axis_summary.get("iqr")),
        "second_axis_fraction_abs_ge_0p95": finite(axis_summary.get("fraction_abs_ge_0p95")),
        "Psi_Phi_pearson_same_rows": finite(axis_audit.get("pearson_Psi_Phi")),
        "Psi_Phi_spearman_same_rows": finite(axis_audit.get("spearman_Psi_Phi")),
        "second_axis_idle_share_pearson": finite(
            axis_audit.get("pearson_Psi_idle_share" if semantic_alignment else "pearson_Phi_idle_share")
        ),
        "Z_second_value_change_fraction_median": finite(
            permutation_summary.get("median_Z_Phi_value_change_fraction")
        ) if not semantic_alignment else np.nan,
        "Z_second_value_change_fraction_minimum": finite(
            permutation_summary.get("minimum_Z_Phi_value_change_fraction")
        ) if not semantic_alignment else np.nan,
        "fraction_rows_in_matching_groups_with_Z_second_variation": finite(
            innovation_audit.get("fraction_randomizable_rows_in_groups_with_Z_Phi_variation")
        ) if not semantic_alignment else np.nan,
        "negative_divergence_observed": negative["observed"],
        "negative_divergence_null_mean": negative["null_mean"],
        "negative_divergence_q": negative["q"],
        "shell_inward_observed": inward["observed"],
        "shell_inward_null_mean": inward["null_mean"],
        "shell_inward_q": inward["q"],
        "core_to_shell_speed_observed": speed["observed"],
        "core_to_shell_speed_null_mean": speed["null_mean"],
        "core_to_shell_speed_q": speed["q"],
        "training_core_fallback_used": coerce_bool(core["fallback_used"]),
        "training_core_dynamically_qualified": coerce_bool(core["dynamically_qualified"]),
        "training_core_center_M": core["center_M"],
        "training_core_center_second_axis": core["center_second_axis"],
        "training_core_cells": core["region_cells"],
        "core_metrics_between_coordinate_winner_interpretation_allowed": False,
    }


def contrast_table(summary: pd.DataFrame) -> pd.DataFrame:
    specifications = [
        ("full_field_departure_fraction_of_held_field_rms", "higher", "primary_descriptive_effect_size"),
        ("full_field_observed_to_null_median_ratio", "higher", "secondary_null_dispersion_sensitive"),
        ("full_field_standardized_separation", "higher", "secondary_null_dispersion_sensitive"),
        ("train_held_drift_vector_r", "higher", "supporting_gauge_dependent_replication"),
        ("train_held_occupancy_weighted_local_cosine", "higher", "supporting_gauge_dependent_replication"),
        ("train_held_drift_second_axis_r", "higher", "supporting_axis_specific_replication"),
    ]
    rows: List[Dict[str, Any]] = []
    for split in ("A_val", "B_confirm"):
        selected = summary[summary["split"] == split].set_index("state_definition")
        if "alignment_based_M_Psi" not in selected.index or CONTROL_STATE_LABEL not in selected.index:
            continue
        semantic = selected.loc["alignment_based_M_Psi"]
        control = selected.loc[CONTROL_STATE_LABEL]
        for metric, direction, role in specifications:
            semantic_value = finite(semantic.get(metric))
            control_value = finite(control.get(metric))
            difference = semantic_value - control_value if math.isfinite(semantic_value) and math.isfinite(control_value) else float("nan")
            rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "comparison_role": role,
                    "alignment_based_value": semantic_value,
                    "alignment_free_value": control_value,
                    "alignment_based_minus_alignment_free": difference,
                    "direction_descriptively_favouring_alignment_based": direction,
                    "between_coordinate_p_value": np.nan,
                    "between_coordinate_inference_performed": False,
                    "caution": (
                        "Raw replication metrics depend on coordinate gauge and support; null-normalized contrasts are descriptive "
                        "because no paired learner-level between-coordinate inferential test was specified."
                    ),
                }
            )
    return pd.DataFrame(rows)


def markdown_number(value: Any) -> str:
    number = finite(value)
    if not math.isfinite(number):
        return "--"
    if abs(number) < 1e-3 and number != 0:
        return f"{number:.3e}"
    return f"{number:.4f}"


def markdown_table(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    selected = frame[list(columns)].copy()
    for column in selected.columns:
        if pd.api.types.is_numeric_dtype(selected[column]):
            selected[column] = selected[column].map(markdown_number)
        else:
            selected[column] = selected[column].astype(str)
    header = "| " + " | ".join(selected.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(selected.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in selected.to_numpy()]
    return "\n".join([header, separator, *rows])


def source_record(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def run_self_test() -> None:
    summary = pd.DataFrame(
        [
            {"state_definition": "alignment_based_M_Psi", "split": "A_val", "train_held_drift_vector_r": 0.9},
            {"state_definition": CONTROL_STATE_LABEL, "split": "A_val", "train_held_drift_vector_r": 0.7},
        ]
    )
    contrasts = contrast_table(summary)
    row = contrasts[contrasts["metric"] == "train_held_drift_vector_r"]
    if row.empty or abs(float(row.iloc[0]["alignment_based_minus_alignment_free"]) - 0.2) > 1e-12:
        raise RuntimeError("Summary contrast self-test failed.")
    normalized = normalize_full_field_row(
        {"observed": 1.0, "null_mean": 0.1, "null_sd": 0.1, "null_50": 0.1}
    )
    if abs(normalized["full_field_standardized_separation"] - 9.0) > 1e-12:
        raise RuntimeError("Full-field normalization self-test failed.")
    print("self-test passed")


def code_contract() -> Dict[str, str]:
    common_path = Path(__file__).resolve().with_name("semantic_specificity_common.py")
    return {
        "summary_script_sha256": sha256_file(Path(__file__).resolve()),
        "semantic_specificity_common_sha256": sha256_file(common_path),
    }


def validate_existing_summary(
    completion: Path,
    protocol_freeze_path: Path,
    args: argparse.Namespace,
) -> None:
    manifest = load_json(completion)
    failures = []
    for key, value in code_contract().items():
        if str(manifest.get(key, "") or "") != value:
            failures.append(key)
    freeze_record = manifest.get("protocol_freeze", {})
    if str(freeze_record.get("sha256", "") or "") != sha256_file(protocol_freeze_path):
        failures.append("protocol_freeze")
    recorded_roots = manifest.get("input_roots", {})
    for key, value in input_roots(args).items():
        if str(recorded_roots.get(key, "") or "") != value:
            failures.append(f"input_roots.{key}")
    for key in ("summary_table", "contrast_table", "report"):
        record = manifest.get(key, {})
        path = Path(str(record.get("path", "") or ""))
        expected_sha = str(record.get("sha256", "") or "")
        if not path.exists() or not expected_sha or sha256_file(path) != expected_sha:
            failures.append(key)
    for index, record in enumerate(manifest.get("source_integrity", [])):
        path = Path(str(record.get("path", "") or ""))
        expected = str(record.get("sha256", "") or "")
        if not path.exists() or not expected or sha256_file(path) != expected:
            failures.append(f"source_integrity[{index}]")
    if failures:
        raise RuntimeError(
            "Existing alignment-specificity summary is incompatible or incomplete: "
            + ", ".join(sorted(set(failures)))
            + ". Re-run with --overwrite."
        )


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    started = time.time()
    output_root = args.output_root.resolve()
    completion = output_root / "semantic_specificity_summary_manifest.json"
    if output_root.exists() and any(output_root.iterdir()) and not completion.exists():
        if not args.overwrite:
            raise FileExistsError(f"Non-empty summary output without completion manifest: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.protocol_freeze is None:
        raise ValueError("--protocol-freeze is required for the final A_val/B_confirm summary.")
    protocol_freeze_path = args.protocol_freeze.resolve()
    if completion.exists() and not args.overwrite:
        validate_existing_summary(completion, protocol_freeze_path, args)
        print(f"[alignment specificity summary] compatible summary already complete: {completion}")
        return
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
    protocol_freeze = load_json(protocol_freeze_path)
    if not coerce_bool(protocol_freeze.get("confirmation_locked_before_run", False)):
        raise RuntimeError("Protocol freeze does not certify a pre-confirmation lock.")
    expected_summary_sha = str(
        protocol_freeze.get("code_contract", {}).get("summary", {}).get("sha256", "") or ""
    )
    if expected_summary_sha and expected_summary_sha != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("The summary script changed after protocol freeze.")

    coordinate_manifest_path = (
        args.coordinate_output_root.resolve()
        / "metadata"
        / "semantic_specificity_coordinate_manifest.json"
    )
    coordinate_manifest = load_json(coordinate_manifest_path)
    confirmation_state = coordinate_manifest.get("confirmation_state", {})
    if not coerce_bool(confirmation_state.get("appended_after_protocol_freeze", False)):
        raise RuntimeError("Alignment-free B_confirm coordinate was not appended after protocol freeze.")
    if str(confirmation_state.get("protocol_freeze_sha256", "") or "") != sha256_file(protocol_freeze_path):
        raise RuntimeError("Coordinate manifest references a different protocol freeze.")

    formal_coordinate_root = args.stage1_root.resolve()
    control_coordinate_root = args.coordinate_output_root.resolve() / "stage1"
    formal_core = primary_core_metadata(formal_coordinate_root, FORMAL_COORDINATE)
    control_core = primary_core_metadata(control_coordinate_root, CONTROL_COORDINATE)

    validation_formal_manifest_path = formal_manifest_path(args, "A_val")
    validation_formal_manifest = load_json(validation_formal_manifest_path)
    frozen_formal_validation = protocol_freeze.get("formal_A_val_null_manifest", {})
    frozen_formal_validation_sha = str(frozen_formal_validation.get("sha256", "") or "")
    if (
        not frozen_formal_validation_sha
        or frozen_formal_validation_sha != sha256_file(validation_formal_manifest_path)
    ):
        raise RuntimeError(
            "The A_val formal construction-null manifest differs from the manifest recorded at protocol freeze."
        )

    rows: List[Dict[str, Any]] = []
    source_paths: List[Path] = []
    for split in ("A_val", "B_confirm"):
        current_formal_manifest_path = formal_manifest_path(args, split)
        control_manifest_path = (
            args.nonsemantic_null_root.resolve()
            / "metadata"
            / f"{split}_nonsemantic_construction_null_manifest.json"
        )
        formal_manifest = load_json(current_formal_manifest_path)
        control_manifest = load_json(control_manifest_path)
        require_manifest_source(
            control_manifest,
            "formal_construction_null_manifest",
            "formal_construction_null_manifest_sha256",
            current_formal_manifest_path,
            f"{split} alignment-free null manifest",
        )
        require_manifest_source(
            control_manifest,
            "formal_A_val_construction_null_manifest",
            "formal_A_val_construction_null_manifest_sha256",
            validation_formal_manifest_path,
            f"{split} alignment-free null manifest",
        )
        if canonical_json(control_manifest.get("matching_cutpoints", {})) != canonical_json(
            validation_formal_manifest.get("matching_cutpoints", {})
        ):
            raise RuntimeError(
                f"{split} alignment-free null manifest does not retain the frozen A_val matching cutpoints."
            )
        if canonical_json(formal_manifest.get("matching_cutpoints", {})) != canonical_json(
            validation_formal_manifest.get("matching_cutpoints", {})
        ):
            raise RuntimeError(
                f"{split} formal null manifest does not retain the frozen A_val matching cutpoints."
            )
        if split == "B_confirm":
            manifest_freeze_sha = str(control_manifest.get("protocol_freeze_sha256", "") or "")
            if manifest_freeze_sha != sha256_file(protocol_freeze_path):
                raise RuntimeError("B_confirm null manifest does not match the frozen protocol.")
            if not coerce_bool(
                control_manifest.get("quality_gates", {}).get(
                    "confirmation_locked_by_protocol_before_control_specific_read", False
                )
            ):
                raise RuntimeError("B_confirm null manifest lacks the confirmation-lock quality gate.")
        source_paths.extend([current_formal_manifest_path, control_manifest_path])
        rows.append(
            build_state_row(
                "alignment_based_M_Psi",
                "Psi: aligned-minus-off-target activity over total activity plus idle",
                True,
                split,
                formal_coordinate_root,
                FORMAL_COORDINATE,
                formal_manifest,
                formal_core,
                coordinate_manifest,
            )
        )
        rows.append(
            build_state_row(
                CONTROL_STATE_LABEL,
                "Phi: total activity-minus-idle over total activity plus idle",
                False,
                split,
                control_coordinate_root,
                CONTROL_COORDINATE,
                control_manifest,
                control_core,
                coordinate_manifest,
            )
        )

    summary = pd.DataFrame(rows)
    contrasts = contrast_table(summary)
    summary_path = write_table(summary, output_root / "semantic_specificity_state_comparison")
    contrast_path = write_table(contrasts, output_root / "semantic_specificity_metric_contrasts")

    report_columns = [
        "state_definition",
        "split",
        "train_held_drift_vector_r",
        "train_held_drift_second_axis_r",
        "train_held_mean_local_cosine",
        "train_held_occupancy_weighted_local_cosine",
        "train_held_train_supported_cells",
        "train_held_held_supported_cells",
        "train_held_common_supported_cells",
        "train_held_held_supported_occupancy_mass",
        "train_held_support_jaccard",
        "held_supported_field_rms",
        "full_field_departure_fraction_of_held_field_rms",
        "full_field_observed_to_null_median_ratio",
        "full_field_standardized_separation",
        "full_field_monte_carlo_p",
        "second_axis_sd",
        "second_axis_iqr",
        "second_axis_fraction_abs_ge_0p95",
        "Psi_Phi_pearson_same_rows",
        "second_axis_idle_share_pearson",
        "Z_second_value_change_fraction_median",
        "fraction_rows_in_matching_groups_with_Z_second_variation",
        "negative_divergence_observed",
        "negative_divergence_q",
        "shell_inward_observed",
        "shell_inward_q",
        "core_to_shell_speed_observed",
        "core_to_shell_speed_q",
        "training_core_fallback_used",
    ]
    contrast_columns = [
        "split",
        "metric",
        "comparison_role",
        "alignment_based_value",
        "alignment_free_value",
        "alignment_based_minus_alignment_free",
        "between_coordinate_inference_performed",
    ]
    report = "\n".join(
        [
            "# Alignment-specificity control numerical report",
            "",
            "This reviewer-motivated comparison was specified post hoc. The alignment-based state and one denominator-matched activity--idle comparator use the same response-order axis, 10-day exposure memory, active-plus-idle denominator, activity and idle mappings, formal eligible rows, user weights, grid thresholds and construction-null opportunity strata. The comparator removes content--demand alignment from the second axis but does not remove all event semantics and also changes the numerator roles of neutral activity and idle. Each state defines its own A_train core. A_val is the primary comparison; B_confirm was appended only after the protocol freeze.",
            "",
            "## State-level results",
            "",
            markdown_table(summary, report_columns),
            "",
            "## Alignment-based minus alignment-free contrasts",
            "",
            markdown_table(contrasts, contrast_columns),
            "",
            "## Interpretation boundary",
            "",
            "The comparison does not provide a between-coordinate p value. The primary descriptive effect size is the observed departure from the construction-null mean divided by the held-out supported field RMS. Ratios to null dispersion are secondary because the activity--idle innovation is partly constrained by idle-share matching. Raw field replication depends on coordinate gauge and is supporting context. Axis scale, boundary mass, support and donor-value-change audits must be reported. Separately selected core, inward-flow and core-speed metrics are within-state diagnostics and cannot define a winner. A positive statement may only say that the alignment-based state outperformed this single alignment-free comparator, not that alignment is causally necessary or that all semantic information was removed. Mixed or similar performance supports a construction-defined effective state without semantic-specificity attribution.",
            "",
        ]
    )
    report_path = output_root / "semantic_specificity_control_report.md"
    report_path.write_text(report, encoding="utf-8")

    source_paths.extend([coordinate_manifest_path, protocol_freeze_path])
    for coordinate_root, coordinate in (
        (formal_coordinate_root, FORMAL_COORDINATE),
        (control_coordinate_root, CONTROL_COORDINATE),
    ):
        for split in ("A_train", "A_val", "B_confirm"):
            source_paths.append(table_path(field_path(coordinate_root, coordinate, split)))

    manifest = {
        "script": Path(__file__).name,
        **code_contract(),
        "runtime_seconds": float(time.time() - started),
        "input_roots": input_roots(args),
        "summary_table": source_record(summary_path),
        "contrast_table": source_record(contrast_path),
        "report": source_record(report_path),
        "source_integrity": [source_record(path) for path in source_paths],
        "comparison_contract": {
            "single_alignment_free_comparator": True,
            "comparator_interpretation": CONTROL_INTERPRETATION,
            "same_first_axis_M": True,
            "same_memory_denominator_activity_idle_and_row_eligibility": True,
            "all_event_semantics_removed": False,
            "state_specific_A_train_cores_frozen_to_held_out_splits": True,
            "core_metrics_used_as_between_state_winner_metrics": False,
            "raw_field_replication_treated_as_gauge_dependent": True,
            "primary_descriptive_effect_size": "observed construction-null departure divided by held supported field RMS",
            "null_dispersion_ratios_treated_as_secondary": True,
            "between_coordinate_inference_performed": False,
            "analysis_status": "post hoc reviewer-motivated comparison",
            "A_val_primary_post_hoc_comparison": True,
            "protocol_frozen_before_control_specific_confirmation_read": True,
            "B_confirm_post_hoc_output_only_replication": True,
            "no_downstream_model_or_mesostate_rerun": True,
            "no_composite_winner_score": True,
        },
        "protocol_freeze": source_record(protocol_freeze_path),
        "interpretation_boundary": (
            "The comparison tests the alignment-based state against one denominator-matched content-alignment-free activity--idle comparator. "
            "It does not isolate a causal marginal effect, remove all event semantics, or establish uniqueness or optimality."
        ),
    }
    save_json(manifest, completion)
    print(f"[alignment specificity summary] complete: {completion}")


if __name__ == "__main__":
    main()
