#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from semantic_specificity_common import (
    BASE_COLUMNS,
    CONTROL_COORDINATE,
    CONTROL_INTERPRETATION,
    CONTROL_STATE_LABEL,
    add_nonsemantic_coordinate,
    coerce_bool,
    control_spec,
    import_module,
    load_json,
    read_table,
    save_json,
    sha256_file,
    table_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the denominator-matched content-alignment-free activity--idle comparator and its frozen empirical field."
    )
    parser.add_argument(
        "--stage1-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/stage1"),
    )
    parser.add_argument("--stage1-script", type=Path, required=False)
    parser.add_argument(
        "--construction-null-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/stage1_construction_matched_null"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/semantic_specificity_control/coordinate"),
    )
    parser.add_argument(
        "--skip-confirmation",
        action="store_true",
        help="Do not read B_confirm. Use this for the pre-confirmation A_train/A_val stage.",
    )
    parser.add_argument(
        "--append-confirmation",
        action="store_true",
        help="After a protocol freeze, append output-only B_confirm evaluation to an existing A_train/A_val result.",
    )
    parser.add_argument(
        "--protocol-freeze",
        type=Path,
        default=None,
        help="Required with --append-confirmation; records the pre-confirmation analysis contract.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def formal_environment_audit(stage1: Any) -> Dict[str, Any]:
    expected = {
        "TAU_RESPONSE_DAYS": 10.0,
        "TAU_ACTIVITY_DAYS": 10.0,
        "RESPONSE_DURATION_HALF_SAT_MIN": 3.0,
        "EXPLANATION_HALF_SAT_MIN": 2.5,
        "LECTURE_HALF_SAT_MIN": 4.0,
        "IDLE_HALF_SAT_DAYS": 1.0,
        "MIN_STATE_BIN_COUNT": 50,
        "MIN_DRIFT_BIN_COUNT": 30,
        "MIN_CELL_USERS": 5,
        "CONVERGENCE_SPEED_QUANTILE": 0.60,
        "CONVERGENCE_NEGATIVE_DIVERGENCE_QUANTILE": 0.80,
        "CONVERGENCE_RATIO_QUANTILE": 0.60,
        "CONVERGENCE_MIN_CELLS": 4,
        "CONVERGENCE_SHELL_RADIUS": 0.35,
    }
    observed: Dict[str, Any] = {}
    failed = []
    for name, value in expected.items():
        actual = getattr(stage1, name, None)
        observed[name] = actual
        if actual is None or abs(float(actual) - float(value)) > 1e-12:
            failed.append(name)
    observed["GRID_EDGE_COUNT"] = int(len(stage1.GRID_BINS_SIGNED))
    if observed["GRID_EDGE_COUNT"] != 41:
        failed.append("GRID_EDGE_COUNT")
    if failed:
        raise RuntimeError(f"Formal Stage-1 parameter contract failed: {failed}")
    return {"expected": expected, "observed": observed, "passed": True}


def run_self_test() -> None:
    frame = pd.DataFrame(
        {
            "user_id": [1, 1],
            "bundle_step_index": [1, 2],
            "part": [1, 1],
            "next_gap_days": [1.0, float("nan")],
            "has_next_submitted_bundle": [True, False],
            "has_next_within_observation_horizon": [True, False],
            "long_gap_or_no_next": [False, True],
            "M_response_prebalanced_pre": [0.1, 0.2],
            "activity_alignment_order_Psi_pre": [-0.2, -0.1],
            "next_M_response_prebalanced": [0.2, float("nan")],
            "next_activity_alignment_order_Psi": [-0.1, float("nan")],
            "delta_M_response_prebalanced_next": [0.1, float("nan")],
            "delta_activity_alignment_order_Psi_next": [0.1, float("nan")],
            "activity_active_mass_pre": [2.0, 2.5],
            "activity_idle_mass_pre": [1.0, 1.5],
            "activity_active_mass_post": [3.0, 2.5],
            "activity_idle_mass_post": [1.0, 1.5],
            "next_activity_active_mass": [1.5, float("nan")],
            "next_activity_idle_mass": [0.5, float("nan")],
            "response_active_mass_interval": [1.0, 0.0],
            "support_active_total_interval": [0.0, 0.0],
            "idle_mass_interval": [0.0, 0.0],
        }
    )
    transformed, audit = add_nonsemantic_coordinate(frame)
    current = float(transformed.loc[0, "activity_idle_balance_Phi_pre"])
    next_value = float(transformed.loc[0, "next_activity_idle_balance_Phi"])
    if abs(current - 1.0 / 3.0) > 1e-12 or abs(next_value - 0.5) > 1e-12:
        raise RuntimeError("Alignment-free coordinate self-test failed.")
    if audit["maximum_next_state_reconstruction_error"] > 1e-12:
        raise RuntimeError("Alignment-free coordinate reconstruction self-test failed.")
    if not audit["drift_eligibility_matched_to_formal_M_Psi"]:
        raise RuntimeError("Formal-row matching self-test failed.")
    print("self-test passed")


def field_base(output_root: Path, split: str) -> Path:
    suffix = "_output_only" if split == "B_confirm" else ""
    return (
        output_root
        / "stage1"
        / "dynamics"
        / "coordinate_analysis"
        / CONTROL_COORDINATE
        / f"{split}_publication_field_grid{suffix}"
    )


def region_root(output_root: Path) -> Path:
    return (
        output_root
        / "stage1"
        / "dynamics"
        / "candidate_regions"
        / CONTROL_COORDINATE
    )


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_stage1_and_contract(args: argparse.Namespace, module_name: str) -> Tuple[Any, Dict[str, Any], Path, str]:
    output_root = args.output_root.resolve()
    os.environ["EDNET_KT4_OUTPUT_ROOT"] = str(output_root)
    stage1 = import_module(args.stage1_script, module_name)
    stage1.ensure_dirs()
    environment_audit = formal_environment_audit(stage1)

    formal_null_manifest_path = (
        args.construction_null_root.resolve()
        / "metadata"
        / "A_val_construction_null_manifest.json"
    )
    formal_null_manifest = load_json(formal_null_manifest_path)
    expected_stage1_sha = str(formal_null_manifest.get("formal_stage1_script_sha256", "") or "")
    current_stage1_sha = sha256_file(args.stage1_script.resolve())
    if expected_stage1_sha and current_stage1_sha != expected_stage1_sha:
        raise RuntimeError(
            "The Stage-1 implementation differs from the implementation audited by the formal construction null."
        )
    return stage1, environment_audit, formal_null_manifest_path, current_stage1_sha


def load_transformed_split(stage1: Any, stage1_root: Path, split: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    source = stage1_root / "dynamics" / f"student_dynamics_panel_core_{split}"
    frame = stage1.read_table(source, columns=BASE_COLUMNS)
    transformed, audit = add_nonsemantic_coordinate(frame)
    output = stage1.downcast_frame(transformed)
    del frame, transformed
    return output, audit


def extract_core_metadata(output_root: Path, stage1: Any) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    root = region_root(output_root)
    regions = stage1.read_table(root / "training_flow_defined_convergence_regions")
    if regions.empty:
        raise RuntimeError("The alignment-free control produced no training convergence region.")
    if "primary_convergence_region" in regions.columns:
        mask = regions["primary_convergence_region"].map(coerce_bool)
        selected = regions.loc[mask]
        primary = selected.iloc[0] if not selected.empty else regions.iloc[0]
    else:
        primary = regions.iloc[0]
    paths = {
        "core": root / "A_train_primary_convergence_core_mask.npy",
        "thresholds": root / "training_convergence_thresholds.json",
        "regions": table_path(root / "training_flow_defined_convergence_regions"),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    metadata = {
        "core_path": str(paths["core"]),
        "core_sha256": sha256_file(paths["core"]),
        "thresholds_path": str(paths["thresholds"]),
        "thresholds_sha256": sha256_file(paths["thresholds"]),
        "regions_path": str(paths["regions"]),
        "regions_sha256": sha256_file(paths["regions"]),
        "fallback_used": coerce_bool(primary.get("flow_defined_fallback_used", False)),
        "dynamically_qualified": coerce_bool(primary.get("dynamically_qualified", False)),
    }
    return metadata, paths


def compare_frames(before: pd.DataFrame, after: pd.DataFrame, label: str) -> None:
    sort_columns = [column for column in ("x_bin", "y_bin") if column in before.columns]
    first = before.sort_values(sort_columns, kind="mergesort").reset_index(drop=True) if sort_columns else before.reset_index(drop=True)
    second = after.sort_values(sort_columns, kind="mergesort").reset_index(drop=True) if sort_columns else after.reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            first,
            second,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise RuntimeError(f"{label} changed when confirmation was appended: {error}") from error


def code_contract() -> Dict[str, str]:
    common_path = Path(__file__).resolve().with_name("semantic_specificity_common.py")
    return {
        "coordinate_control_script_sha256": sha256_file(Path(__file__).resolve()),
        "semantic_specificity_common_sha256": sha256_file(common_path),
    }


def validate_existing_coordinate_manifest(
    manifest_path: Path,
    args: argparse.Namespace,
    require_preconfirmation: bool,
) -> Dict[str, Any]:
    manifest = load_json(manifest_path)
    current = code_contract()
    failures = []
    expected_stage1 = sha256_file(args.stage1_script.resolve())
    if str(manifest.get("formal_stage1_script_sha256", "") or "") != expected_stage1:
        failures.append("formal_stage1_script_sha256")
    formal_null_manifest_path = (
        args.construction_null_root.resolve()
        / "metadata"
        / "A_val_construction_null_manifest.json"
    )
    if not formal_null_manifest_path.exists():
        failures.append("formal_construction_null_manifest_missing")
    elif str(manifest.get("formal_construction_null_manifest_sha256", "") or "") != sha256_file(
        formal_null_manifest_path
    ):
        failures.append("formal_construction_null_manifest_sha256")
    for key, value in current.items():
        if str(manifest.get(key, "") or "") != value:
            failures.append(key)
    if manifest.get("coordinate", {}).get("name") != CONTROL_COORDINATE:
        failures.append("coordinate_name")
    quality = manifest.get("quality_gates", {})
    for gate in (
        "same_exposure_denominator_path",
        "coordinate_reconstructed_from_frozen_stage1_accounting",
        "formal_state_and_drift_row_eligibility_preserved",
        "A_train_core_frozen_to_held_out_splits",
    ):
        if not coerce_bool(quality.get(gate, False)):
            failures.append(gate)
    output_root = args.output_root.resolve()
    for split in ("A_train", "A_val"):
        try:
            table_path(field_base(output_root, split))
        except FileNotFoundError:
            failures.append(f"{split}_field_missing")
    core_meta = manifest.get("training_core", {})
    for key in ("core", "thresholds", "regions"):
        path_value = str(core_meta.get(f"{key}_path", "") or "")
        sha_value = str(core_meta.get(f"{key}_sha256", "") or "")
        path = Path(path_value) if path_value else Path()
        if not path_value or not path.exists() or not sha_value or sha256_file(path) != sha_value:
            failures.append(f"training_core.{key}")

    state = manifest.get("confirmation_state", {})
    if coerce_bool(state.get("read", False)):
        try:
            table_path(field_base(output_root, "B_confirm"))
        except FileNotFoundError:
            failures.append("B_confirm_field_missing")
    if require_preconfirmation and coerce_bool(state.get("read", False)):
        completed_after_freeze = (
            coerce_bool(state.get("appended_after_protocol_freeze", False))
            and coerce_bool(
                quality.get("A_train_A_val_outputs_unchanged_during_confirmation_append", False)
            )
        )
        if not completed_after_freeze:
            failures.append("confirmation_read_without_certified_post_freeze_append")
    if failures:
        raise RuntimeError(
            "Existing alignment-free coordinate output is incompatible with the current formal contract: "
            + ", ".join(sorted(set(failures)))
            + ". Re-run with --overwrite."
        )
    return manifest


def initial_analysis(args: argparse.Namespace) -> None:
    if args.append_confirmation:
        raise RuntimeError("Internal dispatch error.")
    started = time.time()
    output_root = args.output_root.resolve()
    completion = output_root / "metadata" / "semantic_specificity_coordinate_manifest.json"
    if completion.exists() and not args.overwrite:
        validate_existing_coordinate_manifest(
            completion, args, require_preconfirmation=bool(args.skip_confirmation)
        )
        print(f"[semantic specificity] compatible coordinate output already complete: {completion}")
        return
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Non-empty output without completion manifest: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage1, environment_audit, formal_null_manifest_path, current_stage1_sha = load_stage1_and_contract(
        args, "semantic_specificity_stage1_initial"
    )
    stage1_root = args.stage1_root.resolve()
    splits = ("A_train", "A_val") if args.skip_confirmation else ("A_train", "A_val", "B_confirm")
    split_frames: Dict[str, pd.DataFrame] = {}
    split_audits: Dict[str, Any] = {}
    for split in splits:
        split_frames[split], split_audits[split] = load_transformed_split(stage1, stage1_root, split)

    summary = stage1.analyze_coordinate(
        split_frames["A_train"],
        split_frames["A_val"],
        split_frames.get("B_confirm", pd.DataFrame()),
        control_spec(stage1),
    )
    core_metadata, _ = extract_core_metadata(output_root, stage1)

    confirmation_read = "B_confirm" in split_frames
    manifest = {
        "script": Path(__file__).name,
        **code_contract(),
        "runtime_seconds": float(time.time() - started),
        "formal_stage1_root": str(stage1_root),
        "formal_stage1_script": str(args.stage1_script.resolve()),
        "formal_stage1_script_sha256": current_stage1_sha,
        "formal_construction_null_manifest": str(formal_null_manifest_path),
        "formal_construction_null_manifest_sha256": sha256_file(formal_null_manifest_path),
        "output_root": str(output_root),
        "coordinate": {
            "name": CONTROL_COORDINATE,
            "first_axis": "formal response order M, unchanged",
            "second_axis": "Phi=(total active memory-idle memory)/(total active memory+idle memory)",
            "state_label": CONTROL_STATE_LABEL,
            "equivalent_accumulator_form": "Phi=(H+ + H- + H0 - I)/(H+ + H- + H0 + I)",
            "content_demand_alignment_used_by_second_axis": False,
            "event_semantics_still_define_active_and_idle_mass": True,
            "same_exposure_denominator_as_formal_Psi": True,
            "same_memory_and_activity_mappings_as_formal_state": True,
            "same_state_and_drift_eligible_rows_as_formal_state": True,
            "bounded_range": [-1.0, 1.0],
        },
        "analysis_contract": {
            "control_status": "post hoc semantic-specificity analysis",
            "A_train_defines_field_thresholds_and_core": True,
            "A_val_role": "primary post hoc validation comparison fixed before control-specific confirmation evaluation",
            "B_confirm_role": (
                "output-only replication evaluated before protocol freeze; not recommended for the formal two-phase workflow"
                if confirmation_read
                else "not read before protocol freeze"
            ),
            "grid_and_support_thresholds_unchanged": True,
            "downstream_models_rerun": False,
            "mesostate_or_residence_analysis": False,
        },
        "environment_audit": environment_audit,
        "split_coordinate_audits": split_audits,
        "coordinate_summary": summary,
        "training_core": core_metadata,
        "confirmation_state": {
            "read": confirmation_read,
            "appended_after_protocol_freeze": False,
            "protocol_freeze": None,
        },
        "quality_gates": {
            "formal_stage1_script_matches_existing_null": True,
            "same_exposure_denominator_path": True,
            "coordinate_reconstructed_from_frozen_stage1_accounting": True,
            "formal_state_and_drift_row_eligibility_preserved": True,
            "A_train_core_frozen_to_held_out_splits": True,
            "confirmation_used_for_definition_or_selection": False,
            "confirmation_not_read_before_freeze": not confirmation_read,
            "raw_preprocessing_rerun": False,
            "downstream_model_rerun": False,
        },
        "interpretation_boundary": (
            "This post hoc analysis compares the exposure-alignment state with one denominator-matched content-alignment-free "
            "activity--idle comparator. It does not isolate a causal marginal effect of alignment, remove all event semantics, "
            "or establish uniqueness or optimality among possible coordinates."
        ),
        "comparator_interpretation": CONTROL_INTERPRETATION,
    }
    save_json(manifest, completion)
    print(f"[semantic specificity] alignment-free coordinate complete: {completion}")


def append_confirmation(args: argparse.Namespace) -> None:
    if args.protocol_freeze is None:
        raise ValueError("--append-confirmation requires --protocol-freeze.")
    if args.skip_confirmation:
        raise ValueError("--skip-confirmation cannot be combined with --append-confirmation.")
    output_root = args.output_root.resolve()
    completion = output_root / "metadata" / "semantic_specificity_coordinate_manifest.json"
    if not completion.exists():
        raise FileNotFoundError("Run the A_train/A_val coordinate stage before appending confirmation.")
    existing = validate_existing_coordinate_manifest(
        completion, args, require_preconfirmation=False
    )
    confirmation_state = existing.get("confirmation_state", {})
    if coerce_bool(confirmation_state.get("appended_after_protocol_freeze", False)):
        freeze_path = args.protocol_freeze.resolve()
        recorded_freeze_sha = str(confirmation_state.get("protocol_freeze_sha256", "") or "")
        if recorded_freeze_sha != sha256_file(freeze_path):
            raise RuntimeError("Existing confirmation coordinate references a different protocol freeze.")
        b_path = table_path(field_base(output_root, "B_confirm"))
        print(f"[semantic specificity] compatible confirmation coordinate already appended: {b_path}")
        return
    if coerce_bool(confirmation_state.get("read", False)):
        raise RuntimeError(
            "The existing coordinate run already read B_confirm before a protocol freeze; recreate the result with --skip-confirmation."
        )

    freeze_path = args.protocol_freeze.resolve()
    freeze = load_json(freeze_path)
    if not coerce_bool(freeze.get("confirmation_locked_before_run", False)):
        raise RuntimeError("Protocol freeze does not certify a pre-confirmation lock.")
    expected_manifest_sha = str(freeze.get("coordinate_manifest_sha256_at_freeze", "") or "")
    current_manifest_sha = sha256_file(completion)
    if not expected_manifest_sha or current_manifest_sha != expected_manifest_sha:
        raise RuntimeError("The pre-confirmation coordinate manifest changed after the protocol freeze.")
    expected_stage1_sha = str(freeze.get("formal_stage1_script_sha256", "") or "")
    current_stage1_sha = sha256_file(args.stage1_script.resolve())
    if expected_stage1_sha and current_stage1_sha != expected_stage1_sha:
        raise RuntimeError("The Stage-1 script changed after the protocol freeze.")
    code_contract = freeze.get("code_contract", {})
    expected_coordinate_sha = str(code_contract.get("coordinate_control", {}).get("sha256", "") or "")
    if expected_coordinate_sha and expected_coordinate_sha != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("The coordinate-control script changed after protocol freeze.")
    common_path = Path(__file__).resolve().with_name("semantic_specificity_common.py")
    expected_common_sha = str(code_contract.get("semantic_specificity_common", {}).get("sha256", "") or "")
    if expected_common_sha and expected_common_sha != sha256_file(common_path):
        raise RuntimeError("The shared semantic-specificity implementation changed after protocol freeze.")

    before_fields = {
        split: read_table(field_base(output_root, split)).copy()
        for split in ("A_train", "A_val")
    }
    root = region_root(output_root)
    core_before = np.load(root / "A_train_primary_convergence_core_mask.npy")
    thresholds_before = load_json(root / "training_convergence_thresholds.json")
    regions_before = read_table(root / "training_flow_defined_convergence_regions").copy()

    started = time.time()
    stage1, environment_audit, formal_null_manifest_path, checked_stage1_sha = load_stage1_and_contract(
        args, "semantic_specificity_stage1_confirmation"
    )
    if checked_stage1_sha != current_stage1_sha:
        raise RuntimeError("Stage-1 checksum changed during confirmation append.")
    stage1_root = args.stage1_root.resolve()
    split_frames: Dict[str, pd.DataFrame] = {}
    split_audits: Dict[str, Any] = {}
    for split in ("A_train", "A_val", "B_confirm"):
        split_frames[split], split_audits[split] = load_transformed_split(stage1, stage1_root, split)
    summary = stage1.analyze_coordinate(
        split_frames["A_train"],
        split_frames["A_val"],
        split_frames["B_confirm"],
        control_spec(stage1),
    )

    for split in ("A_train", "A_val"):
        compare_frames(before_fields[split], read_table(field_base(output_root, split)), f"{split} field")
    core_after = np.load(root / "A_train_primary_convergence_core_mask.npy")
    if not np.array_equal(core_before, core_after):
        raise RuntimeError("The frozen A_train core changed when confirmation was appended.")
    thresholds_after = load_json(root / "training_convergence_thresholds.json")
    if canonical_json(thresholds_before) != canonical_json(thresholds_after):
        raise RuntimeError("The frozen A_train thresholds changed when confirmation was appended.")
    compare_frames(regions_before, read_table(root / "training_flow_defined_convergence_regions"), "training core table")
    core_metadata, _ = extract_core_metadata(output_root, stage1)

    existing["runtime_seconds"] = float(existing.get("runtime_seconds", 0.0)) + float(time.time() - started)
    existing["formal_construction_null_manifest"] = str(formal_null_manifest_path)
    existing["formal_construction_null_manifest_sha256"] = sha256_file(formal_null_manifest_path)
    existing["environment_audit"] = environment_audit
    existing.setdefault("split_coordinate_audits", {})["B_confirm"] = split_audits["B_confirm"]
    existing["coordinate_summary"] = summary
    existing["training_core"] = core_metadata
    existing["analysis_contract"]["B_confirm_role"] = "post hoc output-only replication appended only after protocol freeze"
    existing["confirmation_state"] = {
        "read": True,
        "appended_after_protocol_freeze": True,
        "protocol_freeze": str(freeze_path),
        "protocol_freeze_sha256": sha256_file(freeze_path),
        "pre_confirmation_manifest_sha256": current_manifest_sha,
        "A_train_A_val_fields_unchanged": True,
        "A_train_core_and_thresholds_unchanged": True,
    }
    existing["quality_gates"]["confirmation_not_read_before_freeze"] = True
    existing["quality_gates"]["confirmation_appended_after_protocol_freeze"] = True
    existing["quality_gates"]["A_train_A_val_outputs_unchanged_during_confirmation_append"] = True
    save_json(existing, completion)
    print(f"[semantic specificity] output-only alignment-free confirmation coordinate appended: {completion}")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.stage1_script is None:
        raise ValueError("--stage1-script is required.")
    if args.append_confirmation:
        append_confirmation(args)
    else:
        initial_analysis(args)


if __name__ == "__main__":
    main()
