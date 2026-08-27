#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from kinetic_robustness_common import (
    EPS,
    coerce_matching_cutpoints,
    compare_coverage_tables,
    fixed_horizon_kinetics_from_histograms,
    import_module,
    load_json,
    normalize_transition,
    read_table,
    recursive_backend_audit,
    recursive_kernel_backend,
    recursive_labels,
    run_histograms,
    save_json,
    sha256_file,
    table_path,
    transition_counts_from_assignments,
    write_table,
)

STATEWISE_FWER_ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursive construction- and denominator-inertia-matched kinetic surrogate for the frozen K=6 validation partition."
    )
    parser.add_argument("--stage1-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/stage1"))
    parser.add_argument("--stage1-script", type=Path, required=True)
    parser.add_argument("--construction-null-script", type=Path, required=True)
    parser.add_argument(
        "--construction-null-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/stage1_construction_matched_null"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/stage1_kinetic_robustness/recursive_null"),
    )
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-last-resort-fraction", type=float, default=0.01)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--max-users", type=int, default=0)
    return parser.parse_args()


def sample_users(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum <= 0:
        return frame
    users = np.asarray(sorted(pd.to_numeric(frame["user_id"], errors="raise").astype(np.int64).unique()), dtype=np.int64)
    if len(users) <= maximum:
        return frame
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(users, size=int(maximum), replace=False))
    return frame[frame["user_id"].isin(selected)].copy().reset_index(drop=True)


def load_existing_null_manifest(root: Path) -> Tuple[Dict[str, Any], Path]:
    path = root / "metadata" / "A_val_construction_null_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            "The existing formal A_val construction-null manifest is required so the kinetic surrogate reuses its frozen A_train cutpoints and matching contract: "
            f"{path}"
        )
    manifest = load_json(path)
    if manifest.get("analysis_split") != "A_val":
        raise RuntimeError("The supplied construction-null manifest is not the formal A_val run.")
    if manifest.get("primary_coordinates") != ["M", "Psi"]:
        raise RuntimeError("The existing construction null does not use the formal M-Psi state.")
    if bool(manifest.get("confirmation_output_only", False)):
        raise RuntimeError("The kinetic surrogate must be based on the formal A_val null, not B_confirm.")
    null_definition = dict(manifest.get("null_definition", {}))
    if "joint normalized signed-innovation pair" not in str(
        null_definition.get("randomized", "")
    ):
        raise RuntimeError("The supplied construction null does not randomize the formal joint innovation pair.")
    if list(null_definition.get("refitted_objects", [])) != []:
        raise RuntimeError("The supplied construction null refits objects and cannot be reused here.")
    if str(null_definition.get("mesostate_or_model_use", "")).lower() != "none":
        raise RuntimeError("The supplied construction null already uses a mesostate or trained model.")
    return manifest, path


def archived_coverage_path(manifest: Mapping[str, Any], root: Path) -> Path:
    raw = str(manifest.get("matching_fallback_coverage_table", "") or "")
    if raw:
        path = Path(raw)
        if path.exists():
            return path
    return table_path(root / "tables" / "A_val_matching_fallback_coverage")


def transition_counts_from_labels(
    labels: np.ndarray,
    adjacent_sources: np.ndarray,
    k: int,
) -> np.ndarray:
    current = np.asarray(labels[adjacent_sources], dtype=np.int64)
    next_state = np.asarray(labels[adjacent_sources + 1], dtype=np.int64)
    valid = (
        (current >= 0)
        & (current < int(k))
        & (next_state >= 0)
        & (next_state < int(k))
    )
    if not np.any(valid):
        return np.zeros((int(k), int(k)), dtype=float)
    encoded = current[valid] * int(k) + next_state[valid]
    return np.bincount(encoded, minlength=int(k) * int(k)).reshape(int(k), int(k)).astype(float)


def identity_reconstruction_audit(
    identity_labels: np.ndarray,
    archived_assignments: pd.DataFrame,
    archived_transition: np.ndarray,
    archived_runs: pd.DataFrame,
    user_id: np.ndarray,
    step: np.ndarray,
    adjacent_sources: np.ndarray,
    k: int,
    maximum_length: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    archived = archived_assignments.sort_values(["user_id", "bundle_step_index"], kind="mergesort").reset_index(drop=True)
    archived_user = pd.to_numeric(archived["user_id"], errors="raise").to_numpy(dtype=np.int64)
    archived_step = pd.to_numeric(archived["bundle_step_index"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(archived_user, user_id) or not np.array_equal(archived_step, step):
        raise RuntimeError("Archived K=6 assignments do not align with the construction-null panel rows.")
    archived_labels = pd.to_numeric(archived["macrostate"], errors="coerce").to_numpy(dtype=float)
    reconstructed = identity_labels.astype(float)
    reconstructed[identity_labels < 0] = np.nan
    label_match = (
        (np.isnan(archived_labels) & np.isnan(reconstructed))
        | (archived_labels == reconstructed)
    )
    mismatch_fraction = float(np.mean(~label_match))
    counts = transition_counts_from_labels(identity_labels, adjacent_sources, k)
    transition = normalize_transition(counts)
    total_histogram, event_histogram = run_histograms(
        user_id,
        step,
        identity_labels,
        k,
        maximum_length,
    )
    archived_state = pd.to_numeric(archived_runs["macrostate"], errors="coerce").to_numpy(dtype=float)
    archived_event = archived_runs["event_observed"]
    if pd.api.types.is_bool_dtype(archived_event) or pd.api.types.is_numeric_dtype(archived_event):
        archived_event_count = int(pd.to_numeric(archived_event, errors="coerce").fillna(0).astype(bool).sum())
    else:
        archived_event_count = int(
            archived_event.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"}).sum()
        )
    rows = [
        {
            "check": "identity_assignment_mismatch_fraction",
            "observed_difference": mismatch_fraction,
            "tolerance": 0.0,
        },
        {
            "check": "identity_transition_matrix_max_abs_difference",
            "observed_difference": float(np.max(np.abs(transition - archived_transition))),
            "tolerance": 1e-12,
        },
        {
            "check": "identity_residence_run_count_difference",
            "observed_difference": abs(float(np.sum(total_histogram)) - float(len(archived_runs))),
            "tolerance": 0.0,
        },
        {
            "check": "identity_completed_exit_count_difference",
            "observed_difference": abs(float(np.sum(event_histogram)) - float(archived_event_count)),
            "tolerance": 0.0,
        },
        {
            "check": "archived_run_states_finite",
            "observed_difference": float(np.sum(~np.isfinite(archived_state))),
            "tolerance": 0.0,
        },
    ]
    audit = pd.DataFrame(rows)
    audit["passed"] = audit["observed_difference"] <= audit["tolerance"]
    return audit, total_histogram, event_histogram


def summarize_null_distribution(
    observed: Mapping[str, Any],
    replicates: pd.DataFrame,
) -> pd.DataFrame:
    definitions = [
        (
            "aggregate_mean_log_rmst_lift_fixed10",
            "greater",
            True,
            "Primary kinetic endpoint: equal-state mean log fixed-10-step RMST lift relative to each trajectory's own Pii-matched geometric reference.",
        ),
        (
            "diagonal_margin",
            "greater",
            False,
            "Secondary descriptive endpoint: mean Pii minus the strongest off-diagonal destination probability.",
        ),
        (
            "mean_self_transition",
            "greater",
            False,
            "Descriptive one-step persistence scale.",
        ),
        (
            "diagonal_dominant_rows",
            "greater",
            False,
            "Descriptive number of diagonal-dominant rows.",
        ),
    ]
    rows: List[Dict[str, Any]] = []
    for metric, direction, primary, note in definitions:
        values = pd.to_numeric(replicates[metric], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        point = float(observed[metric])
        if values.size:
            if primary and direction == "greater":
                pvalue = float((1 + np.sum(values >= point)) / (values.size + 1))
            elif primary:
                pvalue = float((1 + np.sum(values <= point)) / (values.size + 1))
            else:
                pvalue = np.nan
            row = {
                "metric": metric,
                "direction_supporting_observed_excess": direction,
                "primary_endpoint": bool(primary),
                "observed": point,
                "null_mean": float(np.mean(values)),
                "null_sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "null_2p5": float(np.quantile(values, 0.025)),
                "null_median": float(np.median(values)),
                "null_97p5": float(np.quantile(values, 0.975)),
                "monte_carlo_p": pvalue,
                "inferential_test_performed": bool(primary),
                "finite_replicates": int(values.size),
                "interpretation": note,
            }
        else:
            row = {
                "metric": metric,
                "direction_supporting_observed_excess": direction,
                "primary_endpoint": bool(primary),
                "observed": point,
                "null_mean": np.nan,
                "null_sd": np.nan,
                "null_2p5": np.nan,
                "null_median": np.nan,
                "null_97p5": np.nan,
                "monte_carlo_p": np.nan,
                "inferential_test_performed": bool(primary),
                "finite_replicates": 0,
                "interpretation": note,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def statewise_maxT_inference(
    observed_statewise: pd.DataFrame,
    replicate_statewise: pd.DataFrame,
    metric: str,
    alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required_observed = {"macrostate", metric}
    required_replicates = {"replicate", "macrostate", metric}
    if not required_observed.issubset(observed_statewise.columns):
        missing = sorted(required_observed.difference(observed_statewise.columns))
        raise RuntimeError(f"Observed statewise table is missing columns: {missing}")
    if not required_replicates.issubset(replicate_statewise.columns):
        missing = sorted(required_replicates.difference(replicate_statewise.columns))
        raise RuntimeError(f"Recursive statewise replicates are missing columns: {missing}")
    observed = observed_statewise[["macrostate", metric]].copy()
    observed["macrostate"] = pd.to_numeric(
        observed["macrostate"], errors="raise"
    ).astype(int)
    observed[metric] = pd.to_numeric(observed[metric], errors="raise")
    observed = observed.sort_values("macrostate", kind="mergesort")
    states = observed["macrostate"].to_numpy(dtype=int)
    if len(states) < 2 or len(np.unique(states)) != len(states):
        raise RuntimeError("Statewise maxT inference requires unique frozen state labels.")
    replicate = replicate_statewise[["replicate", "macrostate", metric]].copy()
    replicate["replicate"] = pd.to_numeric(
        replicate["replicate"], errors="raise"
    ).astype(int)
    replicate["macrostate"] = pd.to_numeric(
        replicate["macrostate"], errors="raise"
    ).astype(int)
    replicate[metric] = pd.to_numeric(replicate[metric], errors="raise")
    matrix_frame = replicate.pivot(
        index="replicate", columns="macrostate", values=metric
    ).sort_index()
    matrix_frame = matrix_frame.reindex(columns=states)
    if matrix_frame.isna().any().any():
        raise RuntimeError("Statewise maxT inference requires a complete replicate-by-state matrix.")
    null_values = matrix_frame.to_numpy(dtype=float)
    observed_values = observed[metric].to_numpy(dtype=float)
    if not np.isfinite(null_values).all() or not np.isfinite(observed_values).all():
        raise RuntimeError("Statewise maxT inference requires finite observed and null values.")
    replicate_count, state_count = null_values.shape
    if replicate_count < 3 or state_count != len(states):
        raise RuntimeError("Statewise maxT inference has insufficient null support.")
    null_mean = np.mean(null_values, axis=0)
    null_sd = np.std(null_values, axis=0, ddof=1)
    if np.any(~np.isfinite(null_sd)) or np.any(null_sd <= 0.0):
        raise RuntimeError("Statewise maxT inference requires positive null variance in every state.")
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
    familywise = pd.DataFrame(
        [
            {
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
                "fwer_positive_states": ",".join(
                    f"S{state}" for state in supported_states
                ),
                "multi_state_support": bool(len(supported_states) >= 2),
                "interpretation": interpretation,
            }
        ]
    )
    return familywise, pd.DataFrame(state_rows)


def main() -> None:
    args = parse_args()
    if args.replicates < 100:
        raise ValueError("Use at least 100 recursive null replicates; the formal default is 100.")
    if not 0.0 <= args.max_last_resort_fraction <= 1.0:
        raise ValueError("--max-last-resort-fraction must lie in [0,1].")
    if args.max_users > 0:
        raise RuntimeError("Formal recursive kinetic-null outputs require --max-users=0; positive values are smoke-test only.")

    started = time.time()
    stage1_root = args.stage1_root.resolve()
    construction_root = args.construction_null_root.resolve()
    output_root = args.output_root.resolve()
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    table_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    print(f"[recursive kinetic null] numerical backend: {recursive_kernel_backend()}", flush=True)

    stage1 = import_module(args.stage1_script, "kinetic_recursive_formal_stage1")
    cmn = import_module(args.construction_null_script, "kinetic_recursive_construction_null")
    required_cmn = (
        "required_columns_for_split",
        "read_table",
        "reconstruct_innovations",
        "prepare_analysis",
        "build_matching_keys",
        "build_hierarchical_layouts",
        "generate_joint_donor_mapping",
    )
    missing = [name for name in required_cmn if not hasattr(cmn, name)]
    if missing:
        raise RuntimeError(f"Construction-null script is missing required objects: {missing}")
    required_stage1 = (
        "coordinate_specs",
        "TAU_RESPONSE_DAYS",
        "TAU_ACTIVITY_DAYS",
        "MAX_RESIDENCE_LENGTH",
        "RESIDENCE_REFERENCE_LENGTH",
        "MIN_RESIDENCE_AT_RISK",
    )
    missing_stage1 = [name for name in required_stage1 if not hasattr(stage1, name)]
    if missing_stage1:
        raise RuntimeError(f"Formal Stage-1 script is missing required objects: {missing_stage1}")

    existing_manifest, existing_manifest_path = load_existing_null_manifest(construction_root)
    current_stage1_sha = sha256_file(args.stage1_script.resolve())
    expected_stage1_sha = str(existing_manifest.get("formal_stage1_script_sha256", "") or "")
    if expected_stage1_sha and current_stage1_sha != expected_stage1_sha:
        raise RuntimeError(
            "The Stage-1 script differs from the implementation audited by the existing construction null: "
            f"expected {expected_stage1_sha}, found {current_stage1_sha}."
        )

    cutpoint_payload = dict(existing_manifest.get("matching_cutpoints", {}))
    if not cutpoint_payload:
        raise RuntimeError("The existing construction-null manifest does not contain frozen A_train matching cutpoints.")
    if str(cutpoint_payload.get("fit_split", existing_manifest.get("matching_cutpoint_audit", {}).get("fit_split", ""))) not in {"", "A_train"}:
        raise RuntimeError("Construction-null matching cutpoints were not fitted on A_train.")
    cutpoints = coerce_matching_cutpoints(cutpoint_payload)
    base_seed = int(existing_manifest.get("base_seed", args.seed))

    analysis_base = stage1_root / "dynamics" / "student_dynamics_panel_core_A_val"
    required_columns, aliases = cmn.required_columns_for_split(analysis_base)
    frame = cmn.read_table(analysis_base, columns=required_columns)
    frame = sample_users(frame, args.max_users, args.seed + 19)
    frame = frame.sort_values(["user_id", "bundle_step_index"], kind="mergesort").reset_index(drop=True)
    panel_rows = int(len(frame))
    panel_users = int(frame["user_id"].nunique())

    innovations = cmn.reconstruct_innovations(
        frame,
        tau_response_days=float(stage1.TAU_RESPONSE_DAYS),
        tau_activity_days=float(stage1.TAU_ACTIVITY_DAYS),
        require_next_audit=True,
    )
    specification = stage1.coordinate_specs()[0]
    if specification.name != "MR_PsiA":
        raise RuntimeError(f"Expected the formal MR_PsiA coordinate, found {specification.name!r}.")
    prepared = cmn.prepare_analysis(frame, innovations, stage1, specification)
    keys, randomizable, composition_audit = cmn.build_matching_keys(prepared, cutpoints)
    layouts, coverage, effective_randomizable = cmn.build_hierarchical_layouts(
        keys,
        randomizable,
        seed=base_seed + 200003,
        max_last_resort_fraction=float(args.max_last_resort_fraction),
    )
    archived_coverage = read_table(archived_coverage_path(existing_manifest, construction_root))
    coverage_audit = compare_coverage_tables(coverage, archived_coverage)
    coverage_path = write_table(coverage, table_root / "recursive_null_matching_coverage")
    coverage_audit_path = write_table(
        coverage_audit, table_root / "recursive_null_matching_coverage_audit"
    )
    if not bool(coverage_audit["passed"].all()):
        failed = coverage_audit.loc[~coverage_audit["passed"], "level"].astype(str).tolist()
        raise RuntimeError(f"The reconstructed matching layout differs from the existing formal null: {failed}")
    del keys
    gc.collect()

    user_id = pd.to_numeric(frame["user_id"], errors="raise").to_numpy(dtype=np.int64)
    step = pd.to_numeric(frame["bundle_step_index"], errors="raise").to_numpy(dtype=np.int64)
    observed_m = pd.to_numeric(frame[specification.xcol], errors="coerce").to_numpy(dtype=float)
    observed_psi = pd.to_numeric(frame[specification.ycol], errors="coerce").to_numpy(dtype=float)
    finite_observed = np.isfinite(observed_m) & np.isfinite(observed_psi)
    edge_rows = np.asarray(prepared.drift_row_indices, dtype=np.int64)
    if len(edge_rows) != len(prepared.x):
        raise RuntimeError("Prepared drift-row indices do not match the construction-null arrays.")
    if np.any(edge_rows < 0) or np.any(edge_rows >= panel_rows - 1):
        raise RuntimeError("A valid recursive edge points outside the panel or lacks a following row.")
    target_rows = edge_rows + 1
    adjacency = (user_id[target_rows] == user_id[edge_rows]) & (step[target_rows] == step[edge_rows] + 1)
    if not bool(np.all(adjacency)):
        raise RuntimeError(
            f"The construction-null valid transition rows contain {int(np.sum(~adjacency))} non-adjacent targets; they cannot define coherent residence trajectories."
        )
    if not bool(np.all(finite_observed[target_rows])):
        raise RuntimeError("At least one construction-null edge targets an unobserved macrostate.")
    reconstructed_target_m = np.asarray(prepared.x, dtype=float) + np.asarray(prepared.observed_dx, dtype=float)
    reconstructed_target_psi = np.asarray(prepared.y, dtype=float) + np.asarray(prepared.observed_dy, dtype=float)
    target_m_error = float(np.max(np.abs(reconstructed_target_m - observed_m[target_rows])))
    target_psi_error = float(np.max(np.abs(reconstructed_target_psi - observed_psi[target_rows])))
    if target_m_error > 2e-6 or target_psi_error > 2e-6:
        raise RuntimeError(
            "The panel row sequence does not reproduce the existing one-step construction-null targets: "
            f"M={target_m_error:.3e}, Psi={target_psi_error:.3e}."
        )
    edge_for_row = np.full(panel_rows, -1, dtype=np.int64)
    edge_for_row[edge_rows] = np.arange(len(edge_rows), dtype=np.int64)
    if np.unique(edge_rows).size != len(edge_rows):
        raise RuntimeError("Construction-null drift rows are not unique.")

    minimums = {
        "e_pre": float(np.min(np.asarray(prepared.e_pre, dtype=float))),
        "b_pre": float(np.min(np.asarray(prepared.b_pre, dtype=float))),
        "a_m": float(np.min(np.asarray(prepared.a_m, dtype=float))),
        "a_psi": float(np.min(np.asarray(prepared.a_psi, dtype=float))),
    }
    if min(minimums.values()) < -2e-8:
        raise RuntimeError(f"Recursive denominator components contain negative values: {minimums}")

    mesostate_root = stage1_root / "dynamics" / "fixed_k6_mesostates"
    metadata = load_json(mesostate_root / "fixed_k6_model_metadata.json")
    centers = read_table(mesostate_root / "fixed_k6_centers").sort_values(
        "macrostate", kind="mergesort"
    )
    expected_partition = {
        "coordinate": "MR_PsiA",
        "macrostate_k": 6,
        "macrostate_k_rule": "fixed a priori",
        "features": [specification.xcol, specification.ycol],
        "fit_split": "A_train",
        "kmeans_n_init": int(getattr(stage1, "KMEANS_N_INIT", 20)),
        "random_state": int(getattr(stage1, "RANDOM_STATE", 42)),
    }
    failed_partition = [
        key for key, expected in expected_partition.items()
        if metadata.get(key) != expected
    ]
    if failed_partition:
        raise RuntimeError(
            "The formal K=6 partition metadata failed: " + ", ".join(failed_partition)
        )
    scaler_mean = np.asarray(metadata["scaler_mean"], dtype=float)
    scaler_scale = np.asarray(metadata["scaler_scale"], dtype=float)
    standardized_centers = centers[
        ["center_M_standardized", "center_Psi_standardized"]
    ].to_numpy(dtype=float)
    k = int(len(standardized_centers))

    archived_assignments = read_table(mesostate_root / "A_val_fixed_k6_assignments")
    archived_transition = read_table(
        mesostate_root / "A_val_fixed_k6_transition_matrix"
    ).to_numpy(dtype=float)
    archived_runs = read_table(mesostate_root / "A_val_fixed_k6_residence_runs")
    archived_summary = read_table(mesostate_root / "A_val_fixed_k6_residence_summary")
    adjacent_sources = np.flatnonzero(
        (user_id[1:] == user_id[:-1]) & (step[1:] == step[:-1] + 1)
    ).astype(np.int64)

    identity_labels, identity_bound_excess, identity_invalid = recursive_labels(
        observed_m,
        observed_psi,
        finite_observed,
        edge_for_row,
        target_rows,
        prepared.e_pre,
        prepared.b_pre,
        prepared.a_m,
        prepared.a_psi,
        prepared.z_m,
        prepared.z_psi,
        scaler_mean,
        scaler_scale,
        standardized_centers,
    )
    identity_audit, observed_total_histogram, observed_event_histogram = identity_reconstruction_audit(
        identity_labels,
        archived_assignments,
        archived_transition,
        archived_runs,
        user_id,
        step,
        adjacent_sources,
        k,
        int(stage1.MAX_RESIDENCE_LENGTH),
    )
    identity_audit = pd.concat(
        [
            identity_audit,
            pd.DataFrame(
                [
                    {
                        "check": "identity_recursive_bound_excess",
                        "observed_difference": float(identity_bound_excess),
                        "tolerance": 2e-6,
                        "passed": bool(identity_bound_excess <= 2e-6),
                    },
                    {
                        "check": "identity_invalid_denominator_count",
                        "observed_difference": float(identity_invalid),
                        "tolerance": 0.0,
                        "passed": bool(identity_invalid == 0),
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    identity_audit_path = write_table(
        identity_audit, table_root / "recursive_identity_reconstruction_audit"
    )
    if not bool(identity_audit["passed"].all()):
        failed = identity_audit.loc[~identity_audit["passed"], "check"].tolist()
        raise RuntimeError(f"Observed-innovation recursion does not reproduce formal kinetics: {failed}")

    observed_fixed = fixed_horizon_kinetics_from_histograms(
        archived_transition,
        observed_total_histogram,
        observed_event_histogram,
        int(stage1.RESIDENCE_REFERENCE_LENGTH),
    )
    observed_lifts = np.asarray(observed_fixed["rmst_lift_fixed"], dtype=float)
    observed_at_risk = np.asarray(observed_fixed["reference_at_risk"], dtype=float)
    if not np.all(np.isfinite(observed_lifts) & (observed_lifts > 0)):
        raise RuntimeError("The formal fixed-10 endpoint is not defined for all six states.")
    if np.any(observed_at_risk < int(stage1.MIN_RESIDENCE_AT_RISK)):
        raise RuntimeError(
            "The formal fixed-10 endpoint violates the declared at-risk threshold in at least one state."
        )
    observed_metrics = {
        "aggregate_mean_log_rmst_lift_fixed10": float(
            observed_fixed["aggregate_mean_log_rmst_lift_fixed"]
        ),
        "diagonal_margin": float(observed_fixed["diagonal_margin"]),
        "mean_self_transition": float(observed_fixed["mean_self_transition"]),
        "diagonal_dominant_rows": int(observed_fixed["diagonal_dominant_rows"]),
    }

    replicate_rows: List[Dict[str, Any]] = []
    statewise_rows: List[Dict[str, Any]] = []
    first_permutation_audit: Dict[str, Any] = {}
    maximum_bound_excess = 0.0
    for replicate in range(int(args.replicates)):
        donor_seed = int(base_seed + 1000003 * (replicate + 1))
        donor = cmn.generate_joint_donor_mapping(
            len(prepared.x),
            layouts,
            effective_randomizable,
            donor_seed,
        )
        donor_z_m = np.asarray(prepared.z_m, dtype=float)[donor]
        donor_z_psi = np.asarray(prepared.z_psi, dtype=float)[donor]
        labels, bound_excess, invalid_denominator = recursive_labels(
            observed_m,
            observed_psi,
            finite_observed,
            edge_for_row,
            target_rows,
            prepared.e_pre,
            prepared.b_pre,
            prepared.a_m,
            prepared.a_psi,
            donor_z_m,
            donor_z_psi,
            scaler_mean,
            scaler_scale,
            standardized_centers,
        )
        maximum_bound_excess = max(maximum_bound_excess, float(bound_excess))
        if invalid_denominator != 0:
            raise RuntimeError(
                f"Recursive null replicate {replicate} encountered {invalid_denominator} invalid denominators."
            )
        if bound_excess > 2e-6:
            raise RuntimeError(
                f"Recursive null replicate {replicate} required non-numerical clipping: excess={bound_excess:.3e}."
            )
        counts = transition_counts_from_labels(labels, adjacent_sources, k)
        transition = normalize_transition(counts)
        total_histogram, event_histogram = run_histograms(
            user_id,
            step,
            labels,
            k,
            int(stage1.MAX_RESIDENCE_LENGTH),
        )
        fixed = fixed_horizon_kinetics_from_histograms(
            transition,
            total_histogram,
            event_histogram,
            int(stage1.RESIDENCE_REFERENCE_LENGTH),
        )
        replicate_lifts = np.asarray(fixed["rmst_lift_fixed"], dtype=float)
        replicate_at_risk = np.asarray(fixed["reference_at_risk"], dtype=float)
        if not np.all(np.isfinite(replicate_lifts) & (replicate_lifts > 0)):
            raise RuntimeError(
                f"Recursive null replicate {replicate} does not define the fixed all-six-state residence endpoint."
            )
        if np.any(replicate_at_risk < int(stage1.MIN_RESIDENCE_AT_RISK)):
            raise RuntimeError(
                f"Recursive null replicate {replicate} has fewer than "
                f"{int(stage1.MIN_RESIDENCE_AT_RISK)} runs at risk at the fixed reference in at least one state."
            )
        replicate_rows.append(
            {
                "replicate": int(replicate),
                "donor_seed": donor_seed,
                "transition_count": int(np.sum(counts)),
                "residence_run_count": int(np.sum(total_histogram)),
                "right_censored_run_count": int(
                    np.sum(total_histogram) - np.sum(event_histogram)
                ),
                "aggregate_mean_log_rmst_lift_fixed10": float(
                    fixed["aggregate_mean_log_rmst_lift_fixed"]
                ),
                "diagonal_margin": float(fixed["diagonal_margin"]),
                "mean_self_transition": float(fixed["mean_self_transition"]),
                "diagonal_dominant_rows": int(fixed["diagonal_dominant_rows"]),
                "minimum_reference_at_risk": float(np.min(replicate_at_risk)),
                "coordinate_bound_excess_before_numerical_clipping": float(bound_excess),
            }
        )
        for state in range(k):
            statewise_rows.append(
                {
                    "replicate": int(replicate),
                    "macrostate": int(state),
                    "self_transition": float(fixed["self_transition"][state]),
                    "rmst_lift_fixed10": float(fixed["rmst_lift_fixed"][state]),
                    "tail_excess_fixed10": float(fixed["tail_excess_fixed"][state]),
                    "reference_at_risk": float(fixed["reference_at_risk"][state]),
                    "run_count": float(fixed["run_count"][state]),
                }
            )
        if replicate == 0:
            randomized_rows = np.asarray(effective_randomizable, dtype=bool)
            recipient = np.flatnonzero(randomized_rows)
            first_permutation_audit = {
                "randomized_rows": int(np.sum(randomized_rows)),
                "fixed_points_among_randomized_rows": int(np.sum(donor[randomized_rows] == recipient)),
                "mean_Z_M_before": float(np.mean(prepared.z_m)),
                "mean_Z_M_after": float(np.mean(donor_z_m)),
                "mean_Z_Psi_before": float(np.mean(prepared.z_psi)),
                "mean_Z_Psi_after": float(np.mean(donor_z_psi)),
                "mean_product_Z_before": float(np.mean(np.asarray(prepared.z_m) * np.asarray(prepared.z_psi))),
                "mean_product_Z_after": float(np.mean(donor_z_m * donor_z_psi)),
            }
            if first_permutation_audit["fixed_points_among_randomized_rows"] != 0:
                raise RuntimeError("The first recursive null replicate contains fixed donor points.")
            if max(
                abs(first_permutation_audit["mean_Z_M_before"] - first_permutation_audit["mean_Z_M_after"]),
                abs(first_permutation_audit["mean_Z_Psi_before"] - first_permutation_audit["mean_Z_Psi_after"]),
                abs(first_permutation_audit["mean_product_Z_before"] - first_permutation_audit["mean_product_Z_after"]),
            ) > 1e-12:
                raise RuntimeError("The recursive null donor mapping changed a preserved global innovation marginal.")
        if (replicate + 1) % max(int(args.progress_every), 1) == 0 or replicate == 0:
            print(f"[recursive kinetic null] {replicate + 1}/{args.replicates}", flush=True)

    replicate_frame = pd.DataFrame(replicate_rows)
    statewise_frame = pd.DataFrame(statewise_rows)
    summary = summarize_null_distribution(observed_metrics, replicate_frame)
    observed_statewise_rows: List[Dict[str, Any]] = []
    formal_summary_index = archived_summary.set_index("macrostate")
    for state in range(k):
        observed_statewise_rows.append(
            {
                "macrostate": int(state),
                "observed_self_transition": float(observed_fixed["self_transition"][state]),
                "observed_rmst_lift_fixed10": float(observed_fixed["rmst_lift_fixed"][state]),
                "observed_tail_excess_fixed10": float(observed_fixed["tail_excess_fixed"][state]),
                "formal_state_specific_rmst_lift": float(
                    formal_summary_index.loc[state, "restricted_mean_residence_lift"]
                ),
                "null_self_transition_2p5": float(
                    np.nanquantile(
                        statewise_frame.loc[statewise_frame["macrostate"] == state, "self_transition"],
                        0.025,
                    )
                ),
                "null_self_transition_97p5": float(
                    np.nanquantile(
                        statewise_frame.loc[statewise_frame["macrostate"] == state, "self_transition"],
                        0.975,
                    )
                ),
                "null_rmst_lift_fixed10_2p5": float(
                    np.nanquantile(
                        statewise_frame.loc[statewise_frame["macrostate"] == state, "rmst_lift_fixed10"],
                        0.025,
                    )
                ),
                "null_rmst_lift_fixed10_97p5": float(
                    np.nanquantile(
                        statewise_frame.loc[statewise_frame["macrostate"] == state, "rmst_lift_fixed10"],
                        0.975,
                    )
                ),
                "null_tail_excess_fixed10_2p5": float(
                    np.nanquantile(
                        statewise_frame.loc[statewise_frame["macrostate"] == state, "tail_excess_fixed10"],
                        0.025,
                    )
                ),
                "null_tail_excess_fixed10_97p5": float(
                    np.nanquantile(
                        statewise_frame.loc[statewise_frame["macrostate"] == state, "tail_excess_fixed10"],
                        0.975,
                    )
                ),
            }
        )
    observed_statewise = pd.DataFrame(observed_statewise_rows)
    familywise_summary, statewise_inference = statewise_maxT_inference(
        observed_statewise[["macrostate", "observed_tail_excess_fixed10"]].rename(
            columns={"observed_tail_excess_fixed10": "tail_excess_fixed10"}
        ),
        statewise_frame,
        metric="tail_excess_fixed10",
        alpha=STATEWISE_FWER_ALPHA,
    )
    observed_statewise = observed_statewise.merge(
        statewise_inference, on="macrostate", how="left", validate="one_to_one"
    )

    replicate_path = write_table(
        replicate_frame, table_root / "recursive_construction_inertia_null_replicates"
    )
    statewise_replicate_path = write_table(
        statewise_frame, table_root / "recursive_construction_inertia_null_statewise_replicates"
    )
    summary_path = write_table(
        summary, table_root / "recursive_construction_inertia_null_summary"
    )
    observed_statewise_path = write_table(
        observed_statewise, table_root / "recursive_construction_inertia_statewise_summary"
    )
    familywise_summary_path = write_table(
        familywise_summary,
        table_root / "recursive_construction_inertia_statewise_familywise_summary",
    )

    manifest = {
        "script": Path(__file__).name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": float(time.time() - started),
        "stage1_root": str(stage1_root),
        "stage1_script": str(args.stage1_script.resolve()),
        "stage1_script_sha256": current_stage1_sha,
        "construction_null_script": str(args.construction_null_script.resolve()),
        "construction_null_script_sha256": sha256_file(args.construction_null_script.resolve()),
        "existing_construction_null_manifest": str(existing_manifest_path.resolve()),
        "existing_construction_null_manifest_sha256": sha256_file(existing_manifest_path.resolve()),
        "analysis_split": "A_val",
        "B_confirm_read": False,
        "panel_rows": panel_rows,
        "panel_users": panel_users,
        "valid_recursive_edges": int(len(edge_rows)),
        "replicates": int(args.replicates),
        "base_seed": base_seed,
        "primary_coordinates": ["M", "Psi"],
        "numerical_backend": recursive_backend_audit(),
        "fixed_partition": {
            "k": 6,
            "source": "formal A_train-fitted Stage-1 scaler and ordered centres",
            "metadata_path": str(
                (mesostate_root / "fixed_k6_model_metadata.json").resolve()
            ),
            "metadata_sha256": sha256_file(
                mesostate_root / "fixed_k6_model_metadata.json"
            ),
            "centers_path": str(
                table_path(mesostate_root / "fixed_k6_centers").resolve()
            ),
            "centers_sha256": sha256_file(
                table_path(mesostate_root / "fixed_k6_centers")
            ),
            "kmeans_refit": False,
            "partition_selected": False,
        },
        "surrogate_definition": {
            "name": "frozen-observed-strata recursive construction-and-denominator-inertia surrogate",
            "initialization": "each contiguous observable segment starts at its empirical M-Psi state",
            "recursion": (
                "the surrogate state is propagated through the empirical response/exposure denominator path using the existing formal null's jointly permuted opportunity-matched normalized innovation pair"
            ),
            "preserved": [
                "A_val user and bundle-step skeleton",
                "observable/unobservable state positions",
                "segment boundaries and censoring opportunities",
                "empirical response and exposure denominator paths",
                "A_train-frozen opportunity-matching cutpoints",
                "existing hierarchical donor groups and joint Z_M-Z_Psi pairing",
                "formal A_train-fitted K=6 partition",
            ],
            "randomized": "joint normalized response/exposure innovation donors under the existing fixed-point-free cyclic-shift matching protocol",
            "not_preserved": [
                "the empirical current-state anchor after the first row of each segment",
                "empirical Pii",
                "empirical residence runs",
                "state-conditioned temporal innovation ordering",
            ],
            "conditional_boundary": (
                "denominator and matching-stratum paths remain empirical and exogenous; this is a conditional recursive surrogate, not an autonomous learner-platform simulator"
            ),
        },
        "primary_endpoint": {
            "metric": "aggregate_mean_log_rmst_lift_fixed10",
            "reference": "each observed or surrogate trajectory's own Pii-matched geometric process",
            "test": "+1 one-sided Monte Carlo comparison against the declared recursive surrogates",
            "status": "reviewer-motivated post hoc aggregate robustness endpoint; the 10-step horizon was inherited unchanged from the formal analysis",
        },
        "statewise_familywise_endpoint": {
            "metric": "tail_excess_fixed10",
            "global_test": "single-step studentized maxT across all six frozen states",
            "statewise_adjustment": "single-step maxT family-wise adjusted Monte Carlo p values",
            "standardization": "observed statewise effects use the full recursive ensemble; each recursive maxT replicate uses leave-one-out statewise means and standard deviations",
            "positive_state_rule": "observed fixed-10 tail excess greater than zero and maxT-adjusted p below 0.05",
            "multi_state_support_rule": "at least two frozen states satisfy the positive-state rule",
            "alpha": STATEWISE_FWER_ALPHA,
            "status": "reviewer-motivated post hoc statewise family-wise endpoint; it complements rather than replaces the aggregate endpoint",
        },
        "secondary_outputs": [
            "diagonal margin",
            "mean Pii",
            "diagonal-dominant row count",
            "statewise fixed-10-step RMST lift",
        ],
        "relationship_to_existing_experiments": {
            "existing_construction_matched_field_null_rerun": False,
            "existing_matching_cutpoints_and_groups_reused": True,
            "existing_Pii_matched_geometric_reference_retained": True,
            "new_field_drift_divergence_or_core_inference": False,
            "coordinate_or_grid_sensitivity_rerun": False,
            "learner_multiplier_sensitivity_rerun": False,
            "downstream_model_evaluation": False,
        },
        "quality_gates": {
            "full_split_used": bool(args.max_users == 0),
            "matching_coverage_exactly_reproduced": bool(coverage_audit["passed"].all()),
            "observed_innovation_recursion_reproduces_formal_assignments_and_kinetics": bool(
                identity_audit["passed"].all()
            ),
            "all_recursive_edges_are_adjacent": bool(np.all(adjacency)),
            "target_coordinates_reproduced_within_2e_6": bool(
                target_m_error <= 2e-6 and target_psi_error <= 2e-6
            ),
            "fixed10_all_states_meet_at_risk_threshold_in_observed_and_null": True,
            "no_invalid_denominators": True,
            "maximum_coordinate_bound_excess_before_numerical_clipping": float(
                maximum_bound_excess
            ),
            "no_non_numerical_clipping": bool(maximum_bound_excess <= 2e-6),
            "first_permutation_fixed_point_free": bool(
                first_permutation_audit.get("fixed_points_among_randomized_rows", -1) == 0
            ),
            "replicates_at_least_100": bool(args.replicates >= 100),
            "statewise_maxT_all_states_finite": bool(
                len(observed_statewise) == k
                and observed_statewise[
                    [
                        "tail_excess_fixed10_standardized_excess",
                        "tail_excess_fixed10_raw_monte_carlo_p",
                        "tail_excess_fixed10_maxT_fwer_p",
                    ]
                ]
                .apply(pd.to_numeric, errors="coerce")
                .notna()
                .all()
                .all()
            ),
            "statewise_maxT_global_summary_finite": bool(
                len(familywise_summary) == 1
                and familywise_summary[
                    ["observed_maxT", "null_2p5", "null_median", "null_97p5", "monte_carlo_p"]
                ]
                .apply(pd.to_numeric, errors="coerce")
                .notna()
                .all()
                .all()
            ),
        },
        "audits": {
            "matching_aliases": aliases,
            "matching_composition": composition_audit,
            "denominator_component_minima": minimums,
            "target_M_max_abs_error": target_m_error,
            "target_Psi_max_abs_error": target_psi_error,
            "first_permutation": first_permutation_audit,
        },
        "outputs": {
            "matching_coverage": str(coverage_path),
            "matching_coverage_audit": str(coverage_audit_path),
            "identity_reconstruction_audit": str(identity_audit_path),
            "replicate_metrics": str(replicate_path),
            "statewise_replicates": str(statewise_replicate_path),
            "formal_summary": str(summary_path),
            "statewise_summary": str(observed_statewise_path),
            "statewise_familywise_summary": str(familywise_summary_path),
        },
        "interpretation_boundary": (
            "The aggregate endpoint tests a common all-state early-horizon shift, whereas the maxT endpoint tests localized statewise excess with family-wise control. Both are reported. The surrogate tests whether normalized-memory construction, the empirical opportunity/denominator path and matched innovation marginals are sufficient for the observed coarse kinetics. It does not establish an autonomous Markov process, a spectral metastability criterion or causal platform dynamics."
        ),
    }
    manifest_path = metadata_root / "recursive_construction_inertia_null_manifest.json"
    save_json(manifest, manifest_path)
    print(f"[recursive kinetic null] completed: {output_root}", flush=True)


if __name__ == "__main__":
    main()
