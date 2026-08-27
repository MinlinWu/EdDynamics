#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

from semantic_specificity_common import coerce_bool, load_json, save_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the post hoc alignment-specificity comparison before B_confirm is read by the new control."
    )
    parser.add_argument("--coordinate-output-root", type=Path, required=False)
    parser.add_argument("--nonsemantic-null-root", type=Path, required=False)
    parser.add_argument("--formal-null-root", type=Path, required=False)
    parser.add_argument("--stage1-script", type=Path, required=False)
    parser.add_argument("--construction-null-script", type=Path, required=False)
    parser.add_argument("--common-script", type=Path, required=False)
    parser.add_argument("--coordinate-script", type=Path, required=False)
    parser.add_argument("--null-script", type=Path, required=False)
    parser.add_argument("--summary-script", type=Path, required=False)
    parser.add_argument("--output-path", type=Path, required=False)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_self_test() -> None:
    payload: Dict[str, Any] = {
        "confirmation_locked_before_run": True,
        "primary_metrics": ["drift_vector_r", "full_field_observed_to_null_median_ratio"],
    }
    if not coerce_bool(payload["confirmation_locked_before_run"]) or len(payload["primary_metrics"]) != 2:
        raise RuntimeError("Protocol-freeze self-test failed.")
    print("self-test passed")


def source_record(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    required_values = {
        "--coordinate-output-root": args.coordinate_output_root,
        "--nonsemantic-null-root": args.nonsemantic_null_root,
        "--formal-null-root": args.formal_null_root,
        "--stage1-script": args.stage1_script,
        "--construction-null-script": args.construction_null_script,
        "--common-script": args.common_script,
        "--coordinate-script": args.coordinate_script,
        "--null-script": args.null_script,
        "--summary-script": args.summary_script,
        "--output-path": args.output_path,
    }
    missing = [name for name, value in required_values.items() if value is None]
    if missing:
        raise ValueError("Missing required arguments: " + ", ".join(missing))

    output_path = args.output_path.resolve()

    coordinate_manifest_path = (
        args.coordinate_output_root.resolve()
        / "metadata"
        / "semantic_specificity_coordinate_manifest.json"
    )
    validation_null_manifest_path = (
        args.nonsemantic_null_root.resolve()
        / "metadata"
        / "A_val_nonsemantic_construction_null_manifest.json"
    )
    formal_validation_manifest_path = (
        args.formal_null_root.resolve()
        / "metadata"
        / "A_val_construction_null_manifest.json"
    )
    coordinate_manifest = load_json(coordinate_manifest_path)
    validation_null_manifest = load_json(validation_null_manifest_path)
    formal_validation_manifest = load_json(formal_validation_manifest_path)

    confirmation_state = coordinate_manifest.get("confirmation_state", {})
    if coerce_bool(confirmation_state.get("read", False)):
        raise RuntimeError(
            "B_confirm was already read by the alignment-free coordinate control. Recreate the coordinate stage with --skip-confirmation."
        )
    coordinate_quality = coordinate_manifest.get("quality_gates", {})
    if not coerce_bool(coordinate_quality.get("confirmation_not_read_before_freeze", False)):
        raise RuntimeError("Coordinate manifest does not certify that confirmation remained unread.")
    null_quality = validation_null_manifest.get("quality_gates", {})
    required_null_gates = [
        "formal_stage1_script_matched",
        "control_saved_field_reproduced",
        "formal_and_control_state_drift_rows_matched",
        "formal_exposure_denominator_path_preserved",
        "formal_matching_cutpoints_reused",
        "formal_matching_coverage_reproduced",
        "joint_innovation_pairs_permuted_together",
        "no_fixed_points_among_randomized_rows",
        "control_core_frozen_from_A_train",
        "coordinate_or_region_refit_within_null",
    ]
    failed = []
    for gate in required_null_gates:
        value = coerce_bool(null_quality.get(gate, False))
        if gate == "coordinate_or_region_refit_within_null":
            value = not value
        if not value:
            failed.append(gate)
    if failed:
        raise RuntimeError(f"A_val alignment-free null failed freeze gates: {failed}")

    innovation_audit = validation_null_manifest.get("innovation_informativeness_audit", {})
    value_change = validation_null_manifest.get("permutation_value_change_summary", {})
    phi_group_variation = float(
        innovation_audit.get("fraction_randomizable_rows_in_groups_with_Z_Phi_variation", 0.0) or 0.0
    )
    phi_value_change = float(
        value_change.get("median_Z_Phi_value_change_fraction", 0.0) or 0.0
    )
    null_informativeness = {
        "fraction_randomizable_rows_in_groups_with_Z_Phi_variation": phi_group_variation,
        "median_Z_Phi_value_change_fraction": phi_value_change,
        "informative_value_reassignment_observed": bool(
            phi_group_variation > 0.0 and phi_value_change > 0.0
        ),
        "interpretation": (
            "This is an audit rather than a tuned threshold. If matched permutations move donor rows but do not change "
            "Z_Phi values, null-normalized specificity claims are not informative."
        ),
    }

    current_stage1_sha = sha256_file(args.stage1_script.resolve())
    expected_stage1_sha = str(formal_validation_manifest.get("formal_stage1_script_sha256", "") or "")
    if expected_stage1_sha and current_stage1_sha != expected_stage1_sha:
        raise RuntimeError("The Stage-1 script differs from the formal validation null contract.")
    if str(coordinate_manifest.get("formal_stage1_script_sha256", "") or "") != current_stage1_sha:
        raise RuntimeError("Coordinate and protocol-freeze Stage-1 checksums differ.")
    if str(validation_null_manifest.get("stage1_script_sha256", "") or "") != current_stage1_sha:
        raise RuntimeError("A_val null and protocol-freeze Stage-1 checksums differ.")

    current_coordinate_sha = sha256_file(coordinate_manifest_path)
    preconfirmation_coordinate_sha = str(
        coordinate_manifest.get("confirmation_state", {}).get("pre_confirmation_manifest_sha256", "") or ""
    ) or current_coordinate_sha

    if output_path.exists() and not args.overwrite:
        existing = load_json(output_path)
        existing_failures = []
        if str(existing.get("coordinate_manifest_sha256_at_freeze", "") or "") != preconfirmation_coordinate_sha:
            existing_failures.append("coordinate_manifest_sha256_at_freeze")
        if str(existing.get("A_val_nonsemantic_null_manifest_sha256_at_freeze", "") or "") != sha256_file(validation_null_manifest_path):
            existing_failures.append("A_val_nonsemantic_null_manifest_sha256_at_freeze")
        existing_contract = existing.get("code_contract", {})
        expected_sources = {
            "formal_stage1": args.stage1_script,
            "formal_construction_null": args.construction_null_script,
            "semantic_specificity_common": args.common_script,
            "coordinate_control": args.coordinate_script,
            "nonsemantic_null": args.null_script,
            "summary": args.summary_script,
        }
        for key, path in expected_sources.items():
            if str(existing_contract.get(key, {}).get("sha256", "") or "") != sha256_file(path.resolve()):
                existing_failures.append(f"code_contract.{key}")
        if existing_failures:
            raise RuntimeError(
                "Existing protocol freeze is incompatible with current inputs: "
                + ", ".join(existing_failures)
                + ". Re-run with --overwrite before any new confirmation access."
            )
        print(f"[alignment specificity freeze] compatible protocol already frozen: {output_path}")
        return

    payload = {
        "script": Path(__file__).name,
        "created_at_unix": float(time.time()),
        "analysis_status": "post hoc reviewer-motivated alignment-specificity comparison",
        "confirmation_locked_before_run": True,
        "coordinate_manifest": source_record(coordinate_manifest_path),
        "coordinate_manifest_sha256_at_freeze": preconfirmation_coordinate_sha,
        "A_val_nonsemantic_null_manifest": source_record(validation_null_manifest_path),
        "A_val_nonsemantic_null_manifest_sha256_at_freeze": sha256_file(validation_null_manifest_path),
        "formal_A_val_null_manifest": source_record(formal_validation_manifest_path),
        "formal_stage1_script_sha256": current_stage1_sha,
        "code_contract": {
            "formal_stage1": source_record(args.stage1_script),
            "formal_construction_null": source_record(args.construction_null_script),
            "semantic_specificity_common": source_record(args.common_script),
            "coordinate_control": source_record(args.coordinate_script),
            "nonsemantic_null": source_record(args.null_script),
            "summary": source_record(args.summary_script),
        },
        "state_comparison": {
            "alignment_based": "(M,Psi), where Psi is aligned-minus-off-target activity over active-plus-idle memory",
            "alignment_free_comparator": "(M,Phi), where Phi is total-active-minus-idle activity over the identical active-plus-idle memory",
            "same_first_axis_M": True,
            "same_memory_denominator_activity_idle_and_row_eligibility": True,
            "all_event_semantics_removed": False,
            "difference_not_isolated_to_a_single_numerator_sign": True,
            "separate_A_train_defined_cores": True,
        },
        "primary_validation_metrics": [
            "within-state construction-null Monte Carlo departure",
            "observed full-field departure divided by the held-out supported field RMS",
        ],
        "secondary_null_dispersion_metrics": [
            "observed-to-construction-null-median distance ratio",
            "construction-null standardized separation",
        ],
        "supporting_validation_metrics": [
            "training-to-A_val drift-vector correlation",
            "training-to-A_val occupancy-weighted local drift cosine",
            "drift-component correlations and speed correlation",
            "support Jaccard, supported-cell counts and held-out supported occupancy mass",
            "second-axis scale, boundary mass, Psi--Phi association and idle-share association",
        ],
        "within_state_geometry_diagnostics": [
            "negative-divergence occupancy",
            "frozen-shell inward fraction",
            "frozen-core-to-shell speed ratio",
        ],
        "null_informativeness_audit": null_informativeness,
        "confirmation_rule": (
            "B_confirm is output-only. It may assess directional replication of the frozen primary and secondary comparisons, "
            "but may not change the coordinate, support thresholds, A_train core, null matching strata, metrics or reporting rule."
        ),
        "reporting_rule": {
            "report_both_states_side_by_side": True,
            "no_cross_coordinate_comparison_of_raw_full_field_distance": True,
            "field_rms_normalized_departure_is_primary_descriptive_effect_size": True,
            "null_dispersion_normalized_departures_are_secondary": True,
            "field_replication_metrics_are_gauge_dependent_supporting_diagnostics": True,
            "report_axis_scale_boundary_support_and_coverage": True,
            "report_null_innovation_value_change_not_only_donor_row_movement": True,
            "shell_and_core_metrics_are_within_state_diagnostics_not_winner_metrics": True,
            "shell_and_core_speed_metrics_eligible_only_for_nonfallback_dynamically_qualified_training_cores": True,
            "no_composite_winner_score": True,
            "no_binary_significance_claim_for_the_between_coordinate_difference": True,
        },
        "interpretation_rule": {
            "alignment_specificity_supported_descriptively": (
                "A positive statement may say only that the alignment-based state showed a larger field-RMS-normalized departure than this "
                "single alignment-free comparator, provided its own construction-null departure is supported, the comparison is not "
                "explained by axis compression, boundary mass or support loss, donor permutations materially change Z_Phi values, and "
                "the direction is retained in output-only confirmation. Ratios to null dispersion, raw cross-coordinate field correlations "
                "and separately selected core metrics are not hard winner criteria."
            ),
            "mixed_or_similar": (
                "If advantages are mixed, small, gauge-dependent or not retained in confirmation, describe (M,Psi) as an auditable "
                "construction-defined effective state without attributing the organization specifically to content--demand alignment."
            ),
            "alignment_free_stronger": (
                "If the alignment-free comparator is stronger, explicitly narrow the semantic interpretation; the empirical field may remain "
                "reproducible but cannot be attributed to content--demand alignment."
            ),
            "causal_boundary": (
                "Neither outcome identifies a causal marginal effect of alignment or removes all event semantics from the comparator."
            ),
        },
        "scope_exclusions": [
            "no item-tag remapping",
            "no mechanism rerun",
            "no Event-SSL rerun",
            "no K-state, transition or residence analysis",
            "no alternative-coordinate search",
            "no claim that the comparator removes all event semantics",
        ],
    }
    save_json(payload, output_path)
    print(f"[alignment specificity freeze] protocol frozen before confirmation: {output_path}")


if __name__ == "__main__":
    main()
