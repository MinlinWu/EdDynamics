#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from semantic_specificity_common import (
    BASE_COLUMNS,
    CONTROL_COORDINATE,
    CONTROL_INTERPRETATION,
    CONTROL_STATE_LABEL,
    coerce_bool,
    coerce_cutpoints,
    compare_coverage_tables,
    control_spec,
    custom_prepared_analysis,
    distribution_summary,
    import_module,
    load_json,
    null_separation_row,
    read_table,
    safe_max_abs,
    save_json,
    sha256_file,
    table_path,
    weighted_component_distance,
    write_table,
)

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the construction-matched null for the denominator-matched content-alignment-free activity--idle comparator."
    )
    parser.add_argument(
        "--stage1-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/stage1"),
    )
    parser.add_argument("--stage1-script", type=Path, required=False)
    parser.add_argument("--construction-null-script", type=Path, required=False)
    parser.add_argument(
        "--construction-null-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/stage1_construction_matched_null"),
        help="Root containing the formal construction-null manifest for --analysis-split.",
    )
    parser.add_argument(
        "--validation-construction-null-root",
        type=Path,
        default=None,
        help="Root containing the frozen A_val construction-null manifest; defaults to --construction-null-root.",
    )
    parser.add_argument(
        "--coordinate-output-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/semantic_specificity_control/coordinate"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/semantic_specificity_control/null"),
    )
    parser.add_argument("--analysis-split", choices=["A_val", "B_confirm"], default="A_val")
    parser.add_argument("--confirmation-output-only", action="store_true")
    parser.add_argument(
        "--protocol-freeze",
        type=Path,
        default=None,
        help="Required for B_confirm; freezes the new control before confirmation is read.",
    )
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-last-resort-fraction", type=float, default=0.01)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def required_contract(module: Any, names: Sequence[str], label: str) -> None:
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"{label} is missing required objects: {missing}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def construction_null_manifest_paths(
    args: argparse.Namespace,
) -> Tuple[Path, Path, Path]:
    split_root = args.construction_null_root.resolve()
    validation_root = (
        args.validation_construction_null_root.resolve()
        if args.validation_construction_null_root is not None
        else split_root
    )
    split_manifest_path = (
        split_root
        / "metadata"
        / f"{args.analysis_split}_construction_null_manifest.json"
    )
    validation_manifest_path = (
        validation_root
        / "metadata"
        / "A_val_construction_null_manifest.json"
    )
    return split_root, split_manifest_path, validation_manifest_path


def audit_saved_control_field(
    coordinate_output_root: Path,
    split: str,
    field: Any,
) -> Dict[str, Any]:
    suffix = "_output_only" if split == "B_confirm" else ""
    base = (
        coordinate_output_root
        / "stage1"
        / "dynamics"
        / "coordinate_analysis"
        / CONTROL_COORDINATE
        / f"{split}_publication_field_grid{suffix}"
    )
    table = read_table(base).sort_values(["x_bin", "y_bin"], kind="mergesort")
    shape = np.asarray(field.drift_u).shape
    if len(table) != int(np.prod(shape)):
        raise RuntimeError(f"Saved control field has {len(table)} rows; expected {int(np.prod(shape))}.")
    saved_u = pd.to_numeric(table["drift_M"], errors="coerce").to_numpy(dtype=float).reshape(shape)
    saved_v = pd.to_numeric(table["drift_Psi"], errors="coerce").to_numpy(dtype=float).reshape(shape)
    saved_p = pd.to_numeric(table["occupancy_probability"], errors="coerce").to_numpy(dtype=float).reshape(shape)
    saved_mask = table["drift_supported"].map(coerce_bool).to_numpy(dtype=bool).reshape(shape)
    field_mask = np.asarray(field.drift_mask, dtype=bool)
    if not np.array_equal(saved_mask, field_mask):
        raise RuntimeError("Recomputed and saved control drift-support masks differ.")
    max_u = safe_max_abs(saved_u, np.asarray(field.drift_u), field_mask)
    max_v = safe_max_abs(saved_v, np.asarray(field.drift_v), field_mask)
    max_p = safe_max_abs(saved_p, np.asarray(field.occupancy_probability))
    if max(max_u, max_v, max_p) > 1e-10:
        raise RuntimeError(
            "Recomputed control field does not match the archived grid: "
            f"M={max_u:.3e}, Phi={max_v:.3e}, occupancy={max_p:.3e}."
        )
    return {
        "saved_field_path": str(table_path(base).resolve()),
        "saved_field_sha256": sha256_file(table_path(base)),
        "drift_mask_exact_match": True,
        "maximum_drift_M_difference": max_u,
        "maximum_drift_Phi_difference": max_v,
        "maximum_occupancy_difference": max_p,
    }


def leave_one_out_field_distances(
    null_u: np.ndarray,
    null_v: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
    distance_function: Any,
) -> np.ndarray:
    replicates = int(null_u.shape[0])
    total_u = np.sum(null_u, axis=0)
    total_v = np.sum(null_v, axis=0)
    output = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        if replicates > 1:
            mean_u = (total_u - null_u[index]) / (replicates - 1)
            mean_v = (total_v - null_v[index]) / (replicates - 1)
        else:
            mean_u = null_u[index]
            mean_v = null_v[index]
        output[index] = distance_function(
            null_u[index],
            null_v[index],
            mean_u,
            mean_v,
            weight,
            mask,
        )
    return output


def leave_one_out_component_distances(
    values: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    replicates = int(values.shape[0])
    total = np.sum(values, axis=0)
    output = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        mean = (total - values[index]) / (replicates - 1) if replicates > 1 else values[index]
        output[index] = weighted_component_distance(values[index], mean, weight, mask)
    return output


def run_self_test() -> None:
    observed_u = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    observed_v = np.asarray([[0.2, 0.1], [0.4, 0.3]])
    null_u = np.stack([observed_u * 0.1, observed_u * 0.2, observed_u * 0.3])
    null_v = np.stack([observed_v * 0.1, observed_v * 0.2, observed_v * 0.3])
    weight = np.full((2, 2), 0.25)
    mask = np.ones((2, 2), dtype=bool)

    def distance(a_u, a_v, b_u, b_v, w, m):
        selected = np.asarray(m, dtype=bool)
        ww = w[selected] / np.sum(w[selected])
        return float(np.sqrt(np.sum(ww * ((a_u[selected] - b_u[selected]) ** 2 + (a_v[selected] - b_v[selected]) ** 2))))

    values = leave_one_out_field_distances(null_u, null_v, weight, mask, distance)
    components = leave_one_out_component_distances(null_u, weight, mask)
    if values.shape != (3,) or components.shape != (3,) or not np.isfinite(values).all():
        raise RuntimeError("Null-distance self-test failed.")
    row = null_separation_row("test", 1.0, values, primary=True)
    if row["finite_replicates"] != 3:
        raise RuntimeError("Null-summary self-test failed.")
    print("self-test passed")


def code_contract() -> Dict[str, str]:
    common_path = Path(__file__).resolve().with_name("semantic_specificity_common.py")
    return {
        "alignment_free_null_script_sha256": sha256_file(Path(__file__).resolve()),
        "semantic_specificity_common_sha256": sha256_file(common_path),
    }


def coordinate_manifest_compatible_with_stored_sha(
    coordinate_manifest_path: Path,
    stored_sha: str,
) -> bool:
    current_sha = sha256_file(coordinate_manifest_path)
    if current_sha == stored_sha:
        return True
    manifest = load_json(coordinate_manifest_path)
    pre_sha = str(
        manifest.get("confirmation_state", {}).get("pre_confirmation_manifest_sha256", "") or ""
    )
    return bool(pre_sha and pre_sha == stored_sha)


def validate_existing_null_manifest(
    completion: Path,
    args: argparse.Namespace,
    coordinate_manifest_path: Path,
) -> Dict[str, Any]:
    manifest = load_json(completion)
    failures = []
    if str(manifest.get("analysis_split", "")) != str(args.analysis_split):
        failures.append("analysis_split")
    if int(manifest.get("replicates", -1)) != int(args.replicates):
        failures.append("replicates")
    if int(manifest.get("base_seed", -1)) != int(args.seed):
        failures.append("base_seed")
    _, formal_manifest_path, validation_manifest_path = construction_null_manifest_paths(args)
    for path_label, sha_label, current_path in (
        (
            "formal_construction_null_manifest",
            "formal_construction_null_manifest_sha256",
            formal_manifest_path,
        ),
        (
            "formal_A_val_construction_null_manifest",
            "formal_A_val_construction_null_manifest_sha256",
            validation_manifest_path,
        ),
    ):
        if not current_path.exists():
            failures.append(f"{path_label}_missing")
            continue
        if str(manifest.get(path_label, "") or "") != str(current_path):
            failures.append(path_label)
        if str(manifest.get(sha_label, "") or "") != sha256_file(current_path):
            failures.append(sha_label)
    if args.analysis_split == "B_confirm":
        if not coerce_bool(manifest.get("confirmation_output_only", False)):
            failures.append("confirmation_output_only")
        protocol_freeze_path = args.protocol_freeze.resolve()
        if not protocol_freeze_path.exists():
            failures.append("protocol_freeze_missing")
        else:
            current_freeze_sha = sha256_file(protocol_freeze_path)
            if str(manifest.get("protocol_freeze_sha256", "") or "") != current_freeze_sha:
                failures.append("protocol_freeze_sha256")
            protocol_freeze = load_json(protocol_freeze_path)
            if not coerce_bool(protocol_freeze.get("confirmation_locked_before_run", False)):
                failures.append("confirmation_locked_before_run")
            frozen_formal_A_val = protocol_freeze.get("formal_A_val_null_manifest", {})
            expected_formal_A_val_sha = str(frozen_formal_A_val.get("sha256", "") or "")
            if (
                not validation_manifest_path.exists()
                or not expected_formal_A_val_sha
                or sha256_file(validation_manifest_path) != expected_formal_A_val_sha
            ):
                failures.append("formal_A_val_manifest_at_protocol_freeze")
            frozen_A_val_path = (
                args.output_root.resolve()
                / "metadata"
                / "A_val_nonsemantic_construction_null_manifest.json"
            )
            expected_A_val_sha = str(
                protocol_freeze.get("A_val_nonsemantic_null_manifest_sha256_at_freeze", "") or ""
            )
            if (
                not frozen_A_val_path.exists()
                or not expected_A_val_sha
                or sha256_file(frozen_A_val_path) != expected_A_val_sha
            ):
                failures.append("A_val_nonsemantic_manifest_at_protocol_freeze")
    if str(manifest.get("stage1_script_sha256", "") or "") != sha256_file(args.stage1_script.resolve()):
        failures.append("stage1_script_sha256")
    if str(manifest.get("construction_null_script_sha256", "") or "") != sha256_file(
        args.construction_null_script.resolve()
    ):
        failures.append("construction_null_script_sha256")
    for key, value in code_contract().items():
        if str(manifest.get(key, "") or "") != value:
            failures.append(key)
    stored_coordinate_sha = str(manifest.get("coordinate_control_manifest_sha256", "") or "")
    if not stored_coordinate_sha or not coordinate_manifest_compatible_with_stored_sha(
        coordinate_manifest_path, stored_coordinate_sha
    ):
        failures.append("coordinate_control_manifest_sha256")
    arrays_path = Path(str(manifest.get("arrays_path", "") or ""))
    arrays_sha = str(manifest.get("arrays_sha256", "") or "")
    if not arrays_path.exists() or not arrays_sha or sha256_file(arrays_path) != arrays_sha:
        failures.append("arrays_path")
    split_table_root = args.output_root.resolve() / "tables" / str(args.analysis_split)
    for base_name in (
        "nonsemantic_null_replicate_metrics",
        "nonsemantic_null_basin_tests",
        "nonsemantic_null_field_departure",
        "nonsemantic_matching_fallback_coverage",
        "formal_nonsemantic_matching_coverage_audit",
    ):
        try:
            table_path(split_table_root / base_name)
        except FileNotFoundError:
            failures.append(base_name)

    quality = manifest.get("quality_gates", {})
    for gate in (
        "formal_stage1_script_matched",
        "control_saved_field_reproduced",
        "formal_and_control_state_drift_rows_matched",
        "formal_exposure_denominator_path_preserved",
        "formal_matching_cutpoints_reused",
        "formal_matching_coverage_reproduced",
        "joint_innovation_pairs_permuted_together",
        "no_fixed_points_among_randomized_rows",
        "control_core_frozen_from_A_train",
    ):
        if not coerce_bool(quality.get(gate, False)):
            failures.append(gate)
    if coerce_bool(quality.get("coordinate_or_region_refit_within_null", True)):
        failures.append("coordinate_or_region_refit_within_null")
    if args.analysis_split == "B_confirm" and not coerce_bool(
        quality.get("confirmation_locked_by_protocol_before_control_specific_read", False)
    ):
        failures.append("confirmation_locked_by_protocol_before_control_specific_read")
    if failures:
        raise RuntimeError(
            "Existing alignment-free construction-null output is incompatible with the current formal contract: "
            + ", ".join(sorted(set(failures)))
            + ". Re-run with --overwrite."
        )
    return manifest


def innovation_informativeness(
    prepared: Any,
    layouts: Sequence[Any],
    randomizable: np.ndarray,
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    mask = np.asarray(randomizable, dtype=bool)
    z_m = np.asarray(prepared.z_m, dtype=np.float64)
    z_phi = np.asarray(prepared.z_psi, dtype=np.float64)
    rows_with_m_variation = 0
    rows_with_phi_variation = 0
    groups_with_m_variation = 0
    groups_with_phi_variation = 0
    nontrivial_groups = 0
    for layout in layouts:
        indices = np.asarray(layout.original_indices, dtype=np.int64)
        if indices.size < 2:
            continue
        nontrivial_groups += 1
        if float(np.ptp(z_m[indices])) > tolerance:
            groups_with_m_variation += 1
            rows_with_m_variation += int(indices.size)
        if float(np.ptp(z_phi[indices])) > tolerance:
            groups_with_phi_variation += 1
            rows_with_phi_variation += int(indices.size)
    randomizable_rows = int(np.sum(mask))
    return {
        "randomizable_rows": randomizable_rows,
        "Z_M_distribution": distribution_summary(z_m[mask]),
        "Z_Phi_distribution": distribution_summary(z_phi[mask]),
        "nontrivial_matching_groups": int(nontrivial_groups),
        "groups_with_Z_M_variation": int(groups_with_m_variation),
        "groups_with_Z_Phi_variation": int(groups_with_phi_variation),
        "fraction_randomizable_rows_in_groups_with_Z_M_variation": float(
            rows_with_m_variation / max(randomizable_rows, 1)
        ),
        "fraction_randomizable_rows_in_groups_with_Z_Phi_variation": float(
            rows_with_phi_variation / max(randomizable_rows, 1)
        ),
        "interpretation": (
            "A donor-row permutation is scientifically informative only to the extent that matched groups contain "
            "different normalized innovation values; row movement alone is insufficient."
        ),
    }


def run_analysis(args: argparse.Namespace) -> None:
    if args.stage1_script is None or args.construction_null_script is None:
        raise ValueError("--stage1-script and --construction-null-script are required.")
    if args.analysis_split == "B_confirm" and not args.confirmation_output_only:
        raise RuntimeError("B_confirm requires --confirmation-output-only.")
    if args.analysis_split == "B_confirm" and args.protocol_freeze is None:
        raise RuntimeError("B_confirm requires --protocol-freeze created before the control reads confirmation.")
    if args.replicates < 20:
        raise ValueError("Use at least 20 null replicates; the matched publication setting uses 100.")
    if not 0.0 <= args.max_last_resort_fraction <= 1.0:
        raise ValueError("--max-last-resort-fraction must lie in [0,1].")

    started = time.time()
    split = args.analysis_split
    output_root = args.output_root.resolve()
    table_root = output_root / "tables"
    array_root = output_root / "arrays"
    metadata_root = output_root / "metadata"
    completion = metadata_root / f"{split}_nonsemantic_construction_null_manifest.json"
    coordinate_output_root = args.coordinate_output_root.resolve()
    coordinate_manifest_path = coordinate_output_root / "metadata" / "semantic_specificity_coordinate_manifest.json"
    if completion.exists() and not args.overwrite:
        validate_existing_null_manifest(completion, args, coordinate_manifest_path)
        print(f"[semantic specificity null] compatible output already complete: {completion}")
        return
    if args.overwrite:
        for path in (
            table_root / split,
            array_root / f"{split}_nonsemantic_construction_null_fields.npz",
            completion,
            metadata_root / f"{split}_nonsemantic_first_permutation_audit.json",
            metadata_root / f"{split}_nonsemantic_reconstruction_audit.json",
        ):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    split_table_root = table_root / split
    split_table_root.mkdir(parents=True, exist_ok=True)
    array_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    coordinate_manifest = load_json(coordinate_manifest_path)
    if coordinate_manifest.get("coordinate", {}).get("name") != CONTROL_COORDINATE:
        raise RuntimeError("Coordinate-control manifest does not describe the required alignment-free comparator.")
    coordinate_script_path = Path(__file__).resolve().with_name("run_nonsemantic_coordinate_control.py")
    common_script_path = Path(__file__).resolve().with_name("semantic_specificity_common.py")
    if str(coordinate_manifest.get("coordinate_control_script_sha256", "") or "") != sha256_file(coordinate_script_path):
        raise RuntimeError("Coordinate-control script differs from the script recorded by the coordinate manifest.")
    if str(coordinate_manifest.get("semantic_specificity_common_sha256", "") or "") != sha256_file(common_script_path):
        raise RuntimeError("Shared semantic-specificity implementation differs from the coordinate manifest.")

    protocol_freeze_path = args.protocol_freeze.resolve() if args.protocol_freeze is not None else None
    protocol_freeze = None
    if split == "B_confirm":
        protocol_freeze = load_json(protocol_freeze_path)
        if not coerce_bool(protocol_freeze.get("confirmation_locked_before_run", False)):
            raise RuntimeError("Protocol freeze does not certify a pre-confirmation lock.")
        confirmation_state = coordinate_manifest.get("confirmation_state", {})
        if not coerce_bool(confirmation_state.get("appended_after_protocol_freeze", False)):
            raise RuntimeError("The B_confirm coordinate was not appended after protocol freeze.")
        frozen_pre_manifest_sha = str(protocol_freeze.get("coordinate_manifest_sha256_at_freeze", "") or "")
        recorded_pre_manifest_sha = str(confirmation_state.get("pre_confirmation_manifest_sha256", "") or "")
        if not frozen_pre_manifest_sha or recorded_pre_manifest_sha != frozen_pre_manifest_sha:
            raise RuntimeError("Coordinate confirmation append does not match the frozen pre-confirmation manifest.")
        recorded_freeze_sha = str(confirmation_state.get("protocol_freeze_sha256", "") or "")
        if recorded_freeze_sha and recorded_freeze_sha != sha256_file(protocol_freeze_path):
            raise RuntimeError("Coordinate manifest references a different protocol freeze.")
        freeze_code_contract = protocol_freeze.get("code_contract", {})
        expected_null_sha = str(freeze_code_contract.get("nonsemantic_null", {}).get("sha256", "") or "")
        if expected_null_sha and expected_null_sha != sha256_file(Path(__file__).resolve()):
            raise RuntimeError("The alignment-free null script changed after protocol freeze.")
        expected_formal_null_sha = str(freeze_code_contract.get("formal_construction_null", {}).get("sha256", "") or "")
        if expected_formal_null_sha and expected_formal_null_sha != sha256_file(args.construction_null_script.resolve()):
            raise RuntimeError("The formal construction-null script changed after protocol freeze.")

    formal_root, formal_manifest_path, validation_manifest_path = construction_null_manifest_paths(args)
    formal_manifest = load_json(formal_manifest_path)
    validation_manifest = load_json(validation_manifest_path)
    if split == "B_confirm" and protocol_freeze is not None:
        frozen_formal_A_val = protocol_freeze.get("formal_A_val_null_manifest", {})
        expected_formal_A_val_sha = str(frozen_formal_A_val.get("sha256", "") or "")
        if not expected_formal_A_val_sha or sha256_file(validation_manifest_path) != expected_formal_A_val_sha:
            raise RuntimeError(
                "The A_val formal construction-null manifest differs from the manifest recorded at protocol freeze."
            )
        frozen_A_val_path = (
            args.output_root.resolve()
            / "metadata"
            / "A_val_nonsemantic_construction_null_manifest.json"
        )
        if not frozen_A_val_path.exists():
            raise FileNotFoundError("The frozen A_val alignment-free null manifest is missing.")
        expected_A_val_sha = str(
            protocol_freeze.get("A_val_nonsemantic_null_manifest_sha256_at_freeze", "") or ""
        )
        if not expected_A_val_sha or sha256_file(frozen_A_val_path) != expected_A_val_sha:
            raise RuntimeError("The A_val alignment-free null result changed after protocol freeze.")
    if canonical_json(formal_manifest.get("matching_cutpoints", {})) != canonical_json(
        validation_manifest.get("matching_cutpoints", {})
    ):
        raise RuntimeError("The formal validation and requested-split nulls do not share frozen A_train cutpoints.")
    formal_replicates = int(formal_manifest.get("replicates", 0))
    formal_seed = int(formal_manifest.get("base_seed", args.seed))
    if formal_replicates and int(args.replicates) != formal_replicates:
        raise RuntimeError(
            f"Use the same replicate count as the formal null: requested={args.replicates}, formal={formal_replicates}."
        )
    if int(args.seed) != formal_seed:
        raise RuntimeError(f"Use the formal null base seed {formal_seed}; received {args.seed}.")

    current_stage1_sha = sha256_file(args.stage1_script.resolve())
    expected_stage1_sha = str(formal_manifest.get("formal_stage1_script_sha256", "") or "")
    if expected_stage1_sha and current_stage1_sha != expected_stage1_sha:
        raise RuntimeError("The Stage-1 script differs from the implementation audited by the formal null.")
    coordinate_stage1_sha = str(coordinate_manifest.get("formal_stage1_script_sha256", "") or "")
    if coordinate_stage1_sha and coordinate_stage1_sha != current_stage1_sha:
        raise RuntimeError("The coordinate control and null runner use different Stage-1 implementations.")

    os.environ["EDNET_KT4_OUTPUT_ROOT"] = str(coordinate_output_root)
    stage1 = import_module(args.stage1_script, f"semantic_specificity_stage1_{split}")
    cmn = import_module(args.construction_null_script, f"semantic_specificity_cmn_{split}")
    required_contract(
        stage1,
        [
            "TAU_RESPONSE_DAYS",
            "TAU_ACTIVITY_DAYS",
            "occupancy_drift_stats",
            "field_stats_from_dict",
            "user_balanced_weights",
            "digitize_closed_right",
            "downcast_frame",
        ],
        "Stage-1 module",
    )
    required_contract(
        cmn,
        [
            "required_columns_for_split",
            "read_table",
            "reconstruct_innovations",
            "PreparedAnalysis",
            "build_matching_keys",
            "build_hierarchical_layouts",
            "generate_joint_donor_mapping",
            "aggregate_mean_field",
            "copy_field_with_drift",
            "field_geometry_metrics",
            "weighted_field_distance",
            "monte_carlo_p",
            "bh_qvalues",
        ],
        "Construction-null module",
    )

    source_base = args.stage1_root.resolve() / "dynamics" / f"student_dynamics_panel_core_{split}"
    formal_columns, resolved_aliases = cmn.required_columns_for_split(source_base)
    columns = sorted(set(formal_columns).union(BASE_COLUMNS))
    frame = cmn.read_table(source_base, columns=columns)
    frame = frame.sort_values(["user_id", "bundle_step_index"], kind="mergesort").reset_index(drop=True)
    panel_rows = int(len(frame))
    panel_users = int(pd.to_numeric(frame["user_id"], errors="coerce").nunique())
    innovations = cmn.reconstruct_innovations(
        frame,
        tau_response_days=float(stage1.TAU_RESPONSE_DAYS),
        tau_activity_days=float(stage1.TAU_ACTIVITY_DAYS),
        require_next_audit=True,
    )
    specification = control_spec(stage1)
    prepared, reconstruction_audit = custom_prepared_analysis(
        cmn,
        stage1,
        frame,
        innovations,
        specification,
    )
    saved_field_audit = audit_saved_control_field(coordinate_output_root, split, prepared.formal_field)

    region_root = (
        coordinate_output_root
        / "stage1"
        / "dynamics"
        / "candidate_regions"
        / CONTROL_COORDINATE
    )
    core_path = region_root / "A_train_primary_convergence_core_mask.npy"
    thresholds_path = region_root / "training_convergence_thresholds.json"
    if not core_path.exists() or not thresholds_path.exists():
        raise FileNotFoundError("Frozen alignment-free A_train core or thresholds are missing.")
    core_mask = np.load(core_path).astype(bool)
    thresholds = load_json(thresholds_path)
    shell_radius = float(thresholds["shell_radius"])
    if core_mask.shape != np.asarray(prepared.formal_field.drift_u).shape:
        raise RuntimeError("Frozen alignment-free core shape differs from the control field.")

    cutpoint_payload = dict(validation_manifest.get("matching_cutpoints", {}))
    if not cutpoint_payload:
        raise RuntimeError("The formal construction-null manifest does not contain frozen matching cutpoints.")
    cutpoints = coerce_cutpoints(cutpoint_payload, cmn)
    keys, randomizable, composition_audit = cmn.build_matching_keys(prepared, cutpoints)
    layouts, coverage_table, randomizable = cmn.build_hierarchical_layouts(
        keys,
        randomizable,
        seed=formal_seed + 200003,
        max_last_resort_fraction=float(args.max_last_resort_fraction),
    )
    innovation_audit = innovation_informativeness(prepared, layouts, randomizable)
    del keys, frame, innovations
    gc.collect()

    archived_coverage_path = Path(str(formal_manifest.get("matching_fallback_coverage_table", "")))
    if not archived_coverage_path.exists():
        archived_coverage_path = table_path(
            formal_root / "tables" / f"{split}_matching_fallback_coverage"
        )
    archived_coverage = read_table(archived_coverage_path)
    coverage_comparison = compare_coverage_tables(coverage_table, archived_coverage)
    if not bool(coverage_comparison["passed"].all()):
        failed = coverage_comparison.loc[~coverage_comparison["passed"], "level"].astype(str).tolist()
        raise RuntimeError(f"Alignment-free and formal null matching coverage differs: {failed}")

    observed_metrics = cmn.field_geometry_metrics(
        stage1,
        prepared.formal_field,
        core_mask,
        f"{split}_nonsemantic_observed",
        shell_radius,
    )

    e_after = prepared.m_denominator
    b_after = prepared.psi_denominator
    ratio_next_m = prepared.s_pre / e_after
    ratio_next_phi = prepared.g_pre / b_after
    ratio_u, ratio_v = cmn.aggregate_mean_field(
        prepared,
        ratio_next_m - prepared.x,
        ratio_next_phi - prepared.y,
    )
    ratio_field = cmn.copy_field_with_drift(prepared.formal_field, ratio_u, ratio_v)
    ratio_metrics = cmn.field_geometry_metrics(
        stage1,
        ratio_field,
        core_mask,
        f"{split}_nonsemantic_pure_ratio",
        shell_radius,
    )

    replicates = int(args.replicates)
    field_shape = np.asarray(prepared.formal_field.drift_u).shape
    null_u = np.empty((replicates,) + field_shape, dtype=np.float64)
    null_v = np.empty_like(null_u)
    replicate_rows: List[Dict[str, Any]] = []
    first_permutation_audit: Dict[str, Any] = {}
    z_bound_tolerance = float(getattr(cmn, "Z_BOUND_TOL", 2e-6))

    for replicate in range(replicates):
        replicate_seed = formal_seed + 1_000_003 * (replicate + 1)
        donor = cmn.generate_joint_donor_mapping(
            len(prepared.x),
            layouts,
            randomizable,
            seed=replicate_seed,
        )
        donor_z_m = prepared.z_m[donor]
        donor_z_phi = prepared.z_psi[donor]
        null_next_m = (prepared.s_pre + prepared.a_m * donor_z_m) / e_after
        null_next_phi = (prepared.g_pre + prepared.a_psi * donor_z_phi) / b_after
        max_bound = max(
            float(np.max(np.maximum(np.abs(null_next_m) - 1.0, 0.0))),
            float(np.max(np.maximum(np.abs(null_next_phi) - 1.0, 0.0))),
        )
        if max_bound > z_bound_tolerance:
            raise RuntimeError(
                f"Null next state left [-1,1] by {max_bound:.3e} in replicate {replicate}."
            )
        null_next_m = np.clip(null_next_m, -1.0, 1.0)
        null_next_phi = np.clip(null_next_phi, -1.0, 1.0)
        u, v = cmn.aggregate_mean_field(
            prepared,
            null_next_m - prepared.x,
            null_next_phi - prepared.y,
        )
        null_u[replicate] = u
        null_v[replicate] = v
        null_field = cmn.copy_field_with_drift(prepared.formal_field, u, v)
        metrics = cmn.field_geometry_metrics(
            stage1,
            null_field,
            core_mask,
            f"{split}_alignment_free_null_{replicate:03d}",
            shell_radius,
        )
        randomized_rows = np.asarray(randomizable, dtype=bool)
        delta_z_m = np.abs(donor_z_m[randomized_rows] - prepared.z_m[randomized_rows])
        delta_z_phi = np.abs(donor_z_phi[randomized_rows] - prepared.z_psi[randomized_rows])
        change_metrics = {
            "Z_M_value_change_fraction": float(np.mean(delta_z_m > 1e-12)) if delta_z_m.size else 0.0,
            "Z_Phi_value_change_fraction": float(np.mean(delta_z_phi > 1e-12)) if delta_z_phi.size else 0.0,
            "joint_value_change_fraction": float(np.mean((delta_z_m > 1e-12) | (delta_z_phi > 1e-12))) if delta_z_m.size else 0.0,
            "median_abs_Z_M_change": float(np.median(delta_z_m)) if delta_z_m.size else 0.0,
            "median_abs_Z_Phi_change": float(np.median(delta_z_phi)) if delta_z_phi.size else 0.0,
            "q90_abs_Z_M_change": float(np.quantile(delta_z_m, 0.90)) if delta_z_m.size else 0.0,
            "q90_abs_Z_Phi_change": float(np.quantile(delta_z_phi, 0.90)) if delta_z_phi.size else 0.0,
        }
        replicate_rows.append({
            "replicate": replicate,
            "seed": replicate_seed,
            **metrics,
            **change_metrics,
        })

        if replicate == 0:
            recipient = np.flatnonzero(randomized_rows)
            first_permutation_audit = {
                "randomized_rows": int(np.sum(randomized_rows)),
                "fixed_points_among_randomized_rows": int(
                    np.sum(donor[randomized_rows] == recipient)
                ),
                "mean_Z_M_before": float(np.mean(prepared.z_m)),
                "mean_Z_M_after": float(np.mean(donor_z_m)),
                "mean_Z_Phi_before": float(np.mean(prepared.z_psi)),
                "mean_Z_Phi_after": float(np.mean(donor_z_phi)),
                "mean_product_Z_before": float(np.mean(prepared.z_m * prepared.z_psi)),
                "mean_product_Z_after": float(np.mean(donor_z_m * donor_z_phi)),
                "joint_pair_moved_together": True,
                "overall_mapping_bijective_by_disjoint_group_permutations": True,
                **change_metrics,
            }
            if first_permutation_audit["fixed_points_among_randomized_rows"] != 0:
                raise RuntimeError("The first null replicate contains fixed points among randomized rows.")
            for before, after, name in (
                ("mean_Z_M_before", "mean_Z_M_after", "Z_M"),
                ("mean_Z_Phi_before", "mean_Z_Phi_after", "Z_Phi"),
                ("mean_product_Z_before", "mean_product_Z_after", "joint product"),
            ):
                if abs(first_permutation_audit[before] - first_permutation_audit[after]) > 1e-12:
                    raise RuntimeError(f"The first joint permutation did not preserve {name}.")

        if (replicate + 1) % max(int(args.progress_every), 1) == 0 or replicate == 0:
            print(
                f"[semantic specificity null] {split}: {replicate + 1}/{replicates}",
                flush=True,
            )

    null_mean_u = np.mean(null_u, axis=0)
    null_mean_v = np.mean(null_v, axis=0)
    null_sd_u = np.std(null_u, axis=0, ddof=1) if replicates > 1 else np.zeros(field_shape)
    null_sd_v = np.std(null_v, axis=0, ddof=1) if replicates > 1 else np.zeros(field_shape)
    excess_u = np.asarray(prepared.formal_field.drift_u) - null_mean_u
    excess_v = np.asarray(prepared.formal_field.drift_v) - null_mean_v
    excess_field = cmn.copy_field_with_drift(prepared.formal_field, excess_u, excess_v)
    excess_metrics = cmn.field_geometry_metrics(
        stage1,
        excess_field,
        core_mask,
        f"{split}_nonsemantic_excess",
        shell_radius,
    )

    replicate_table = pd.DataFrame(replicate_rows)
    value_change_summary = {
        "median_Z_M_value_change_fraction": float(np.median(replicate_table["Z_M_value_change_fraction"])),
        "minimum_Z_M_value_change_fraction": float(np.min(replicate_table["Z_M_value_change_fraction"])),
        "median_Z_Phi_value_change_fraction": float(np.median(replicate_table["Z_Phi_value_change_fraction"])),
        "minimum_Z_Phi_value_change_fraction": float(np.min(replicate_table["Z_Phi_value_change_fraction"])),
        "median_joint_value_change_fraction": float(np.median(replicate_table["joint_value_change_fraction"])),
        "median_abs_Z_M_change": float(np.median(replicate_table["median_abs_Z_M_change"])),
        "median_abs_Z_Phi_change": float(np.median(replicate_table["median_abs_Z_Phi_change"])),
    }
    basin_specs = [
        ("negative_divergence_occupancy_fraction", "greater"),
        ("flow_weighted_shell_fraction_inward", "greater"),
        ("flow_core_to_shell_speed_ratio", "less"),
    ]
    basin_rows: List[Dict[str, Any]] = []
    p_values: List[float] = []
    for metric, direction in basin_specs:
        values = pd.to_numeric(replicate_table[metric], errors="coerce").to_numpy(dtype=float)
        observed = float(observed_metrics[metric])
        p_value = float(cmn.monte_carlo_p(observed, values, direction))
        p_values.append(p_value)
        basin_rows.append(
            {
                "metric": metric,
                "direction_supporting_excess_structure": direction,
                "observed": observed,
                "pure_ratio_contraction": float(ratio_metrics[metric]),
                "null_mean": float(np.nanmean(values)),
                "null_sd": float(np.nanstd(values, ddof=1)),
                "null_2p5": float(np.nanquantile(values, 0.025)),
                "null_50": float(np.nanquantile(values, 0.50)),
                "null_97p5": float(np.nanquantile(values, 0.975)),
                "monte_carlo_p": p_value,
                "excess_field_value_descriptive": float(excess_metrics[metric]),
            }
        )
    q_values = np.asarray(cmn.bh_qvalues(p_values), dtype=float)
    for row, q_value in zip(basin_rows, q_values):
        row["BH_q_across_three_basin_metrics"] = float(q_value)

    field_mask = np.asarray(prepared.formal_field.drift_mask, dtype=bool)
    field_weight = np.asarray(prepared.formal_field.occupancy_probability, dtype=float)
    observed_full = float(
        cmn.weighted_field_distance(
            np.asarray(prepared.formal_field.drift_u),
            np.asarray(prepared.formal_field.drift_v),
            null_mean_u,
            null_mean_v,
            field_weight,
            field_mask,
        )
    )
    full_null = leave_one_out_field_distances(
        null_u,
        null_v,
        field_weight,
        field_mask,
        cmn.weighted_field_distance,
    )
    observed_m = weighted_component_distance(
        np.asarray(prepared.formal_field.drift_u),
        null_mean_u,
        field_weight,
        field_mask,
    )
    observed_phi = weighted_component_distance(
        np.asarray(prepared.formal_field.drift_v),
        null_mean_v,
        field_weight,
        field_mask,
    )
    null_m = leave_one_out_component_distances(null_u, field_weight, field_mask)
    null_phi = leave_one_out_component_distances(null_v, field_weight, field_mask)
    departure_rows = [
        null_separation_row(
            "occupancy_weighted_full_field_distance_from_null_mean",
            observed_full,
            full_null,
            primary=True,
        ),
        null_separation_row(
            "occupancy_weighted_M_component_distance_from_null_mean",
            observed_m,
            null_m,
            primary=False,
        ),
        null_separation_row(
            "occupancy_weighted_Phi_component_distance_from_null_mean",
            observed_phi,
            null_phi,
            primary=False,
        ),
    ]

    write_table(replicate_table, split_table_root / "nonsemantic_null_replicate_metrics")
    write_table(pd.DataFrame(basin_rows), split_table_root / "nonsemantic_null_basin_tests")
    write_table(pd.DataFrame(departure_rows), split_table_root / "nonsemantic_null_field_departure")
    write_table(coverage_table, split_table_root / "nonsemantic_matching_fallback_coverage")
    write_table(coverage_comparison, split_table_root / "formal_nonsemantic_matching_coverage_audit")
    if isinstance(composition_audit, pd.DataFrame):
        write_table(composition_audit, split_table_root / "nonsemantic_opportunity_composition")
    else:
        save_json(
            {"composition_audit": composition_audit},
            metadata_root / f"{split}_nonsemantic_opportunity_composition.json",
        )

    array_path = array_root / f"{split}_nonsemantic_construction_null_fields.npz"
    np.savez_compressed(
        array_path,
        observed_u=np.asarray(prepared.formal_field.drift_u, dtype=np.float64),
        observed_v=np.asarray(prepared.formal_field.drift_v, dtype=np.float64),
        null_u=null_u,
        null_v=null_v,
        null_mean_u=null_mean_u,
        null_mean_v=null_mean_v,
        null_sd_u=null_sd_u,
        null_sd_v=null_sd_v,
        excess_u=excess_u,
        excess_v=excess_v,
        drift_mask=field_mask,
        state_mask=np.asarray(prepared.formal_field.state_mask, dtype=bool),
        occupancy_probability=field_weight,
        core_mask=core_mask,
        xcenters=np.asarray(prepared.formal_field.xcenters, dtype=np.float64),
        ycenters=np.asarray(prepared.formal_field.ycenters, dtype=np.float64),
        t_null_full=full_null,
        t_null_M=null_m,
        t_null_Phi=null_phi,
    )

    reconstruction_audit.update(
        {
            "resolved_formal_activity_aliases": resolved_aliases,
            "saved_control_field_audit": saved_field_audit,
            "formal_matching_coverage_reproduced": True,
        }
    )
    save_json(
        reconstruction_audit,
        metadata_root / f"{split}_nonsemantic_reconstruction_audit.json",
    )
    save_json(
        first_permutation_audit,
        metadata_root / f"{split}_nonsemantic_first_permutation_audit.json",
    )

    manifest = {
        "script": Path(__file__).name,
        **code_contract(),
        "runtime_seconds": float(time.time() - started),
        "analysis_split": split,
        "primary_manuscript_split": "A_val",
        "analysis_status": "post hoc reviewer-motivated semantic-specificity control",
        "confirmation_output_only": bool(args.confirmation_output_only),
        "protocol_freeze": str(protocol_freeze_path) if protocol_freeze_path is not None else None,
        "protocol_freeze_sha256": sha256_file(protocol_freeze_path) if protocol_freeze_path is not None else None,
        "stage1_root": str(args.stage1_root.resolve()),
        "stage1_script": str(args.stage1_script.resolve()),
        "stage1_script_sha256": current_stage1_sha,
        "construction_null_script": str(args.construction_null_script.resolve()),
        "construction_null_script_sha256": sha256_file(args.construction_null_script.resolve()),
        "formal_construction_null_manifest": str(formal_manifest_path),
        "formal_construction_null_manifest_sha256": sha256_file(formal_manifest_path),
        "formal_A_val_construction_null_manifest": str(validation_manifest_path),
        "formal_A_val_construction_null_manifest_sha256": sha256_file(validation_manifest_path),
        "coordinate_control_manifest": str(coordinate_manifest_path),
        "coordinate_control_manifest_sha256": sha256_file(coordinate_manifest_path),
        "rows_in_analysis_panel": panel_rows,
        "users_in_analysis_panel": panel_users,
        "valid_drift_rows": int(len(prepared.drift_row_indices)),
        "replicates": replicates,
        "base_seed": formal_seed,
        "coordinate": CONTROL_COORDINATE,
        "state_label": CONTROL_STATE_LABEL,
        "second_axis_definition": "Phi=(H+ + H- + H0 - I)/(H+ + H- + H0 + I)",
        "comparator_interpretation": CONTROL_INTERPRETATION,
        "null_definition": {
            "preserved": [
                "formal response-order M and its accounting",
                "current M/Phi anchors",
                "response and exposure denominator increments",
                "formal state and drift eligible rows",
                "user-balanced row weights",
                "40x40 grid and unchanged count thresholds",
                "alignment-free A_train-defined convergence core and shell radius",
                "joint marginal distribution of normalized response/activity-idle innovations",
                "formal A_train opportunity strata and donor seeds",
            ],
            "randomized": "joint normalized signed-innovation pair (Z_M,Z_Phi) reassigned away from its observed current state",
            "refitted_objects": [],
            "downstream_models_or_mesostates": "none",
        },
        "matching_cutpoints": cutpoint_payload,
        "matching_coverage_table": str(
            table_path(split_table_root / "nonsemantic_matching_fallback_coverage").resolve()
        ),
        "matching_coverage_audit_table": str(
            table_path(split_table_root / "formal_nonsemantic_matching_coverage_audit").resolve()
        ),
        "core_path": str(core_path.resolve()),
        "core_sha256": sha256_file(core_path),
        "thresholds_path": str(thresholds_path.resolve()),
        "thresholds_sha256": sha256_file(thresholds_path),
        "shell_radius": shell_radius,
        "observed_geometry": observed_metrics,
        "pure_ratio_geometry": ratio_metrics,
        "excess_geometry": excess_metrics,
        "full_field_primary_test": departure_rows[0],
        "component_departures_descriptive": departure_rows[1:],
        "basin_metric_tests": basin_rows,
        "arrays_path": str(array_path.resolve()),
        "arrays_sha256": sha256_file(array_path),
        "innovation_informativeness_audit": innovation_audit,
        "permutation_value_change_summary": value_change_summary,
        "quality_gates": {
            "formal_stage1_script_matched": True,
            "control_saved_field_reproduced": True,
            "formal_and_control_state_drift_rows_matched": True,
            "formal_exposure_denominator_path_preserved": True,
            "formal_matching_cutpoints_reused": True,
            "formal_matching_coverage_reproduced": True,
            "joint_innovation_pairs_permuted_together": True,
            "no_fixed_points_among_randomized_rows": True,
            "donor_permutations_changed_Z_Phi_values": bool(
                value_change_summary["median_Z_Phi_value_change_fraction"] > 0.0
            ),
            "matched_groups_contained_Z_Phi_variation": bool(
                innovation_audit["fraction_randomizable_rows_in_groups_with_Z_Phi_variation"] > 0.0
            ),
            "control_core_frozen_from_A_train": True,
            "coordinate_or_region_refit_within_null": False,
            "confirmation_used_for_definition_or_selection": False,
            "confirmation_locked_by_protocol_before_control_specific_read": bool(
                split != "B_confirm" or protocol_freeze is not None
            ),
        },
        "interpretation_boundary": (
            "The analysis compares the exposure-alignment state with one denominator-matched content-alignment-free activity--idle comparator. "
            "It does not remove all event semantics, identify a causal marginal effect of alignment, or establish uniqueness among coordinates. "
            "Null-normalized comparisons are uninformative if matched donor permutations move rows but rarely change Z_Phi values."
        ),
    }
    save_json(manifest, completion)
    print(f"[semantic specificity null] complete: {completion}")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    run_analysis(args)


if __name__ == "__main__":
    main()
