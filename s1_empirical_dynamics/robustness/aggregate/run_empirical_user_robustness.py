#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from robustness_common import (
    EPS,
    SparseFieldAccumulator,
    TransitionAccumulator,
    WeightedResidenceAccumulator,
    contraction_metrics,
    direct_field,
    drift_comparison,
    import_module,
    interior_cell_mask,
    percentile_summary,
    read_table,
    resolve_table,
    save_json,
    sha256_file,
    user_equal_row_weights,
    write_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learner-level uncertainty, strict user weighting and grid sensitivity for empirical effective dynamics.")
    parser.add_argument("--stage1-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/stage1"))
    parser.add_argument("--stage1-script", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/supplementary_robustness/empirical"))
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-chunk", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def load_panel(root: Path, split: str, module: Any) -> pd.DataFrame:
    columns = [
        "user_id",
        "bundle_step_index",
        "M_response_prebalanced_pre",
        "activity_alignment_order_Psi_pre",
        "delta_M_response_prebalanced_next",
        "delta_activity_alignment_order_Psi_next",
    ]
    frame = read_table(root / "dynamics" / f"student_dynamics_panel_core_{split}", columns=columns)
    frame["user_id"] = pd.to_numeric(frame["user_id"], errors="raise").astype(np.int64)
    frame["bundle_step_index"] = pd.to_numeric(frame["bundle_step_index"], errors="raise").astype(np.int64)
    for column in columns[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = module.downcast_frame(frame)
    return frame.sort_values(["user_id", "bundle_step_index"], kind="mergesort").reset_index(drop=True)


def panel_arrays(frame: pd.DataFrame) -> Tuple[np.ndarray, ...]:
    return (
        frame["user_id"].to_numpy(dtype=np.int64),
        numeric(frame, "M_response_prebalanced_pre"),
        numeric(frame, "activity_alignment_order_Psi_pre"),
        numeric(frame, "delta_M_response_prebalanced_next"),
        numeric(frame, "delta_activity_alignment_order_Psi_next"),
    )


def to_formal_field(module: Any, field, users: int, valid_state_rows: int, valid_drift_rows: int):
    shape = field.u.shape
    zeros = np.zeros(shape, dtype=float)
    return module.FieldStats(
        xbins=np.asarray(field.bins, dtype=float),
        ybins=np.asarray(field.bins, dtype=float),
        xcenters=np.asarray(field.centers, dtype=float),
        ycenters=np.asarray(field.centers, dtype=float),
        occupancy_weighted=np.asarray(field.occupancy_weight, dtype=float),
        occupancy_count=np.asarray(field.occupancy_count, dtype=float),
        user_count=np.asarray(field.user_count, dtype=float),
        occupancy_probability=np.asarray(field.occupancy_probability, dtype=float),
        potential=-np.log(np.asarray(field.occupancy_probability, dtype=float) + EPS),
        drift_u=np.asarray(field.u, dtype=float),
        drift_v=np.asarray(field.v, dtype=float),
        drift_count=np.asarray(field.drift_count, dtype=float),
        drift_weight=np.asarray(field.drift_weight, dtype=float),
        drift_weight_sq=zeros,
        drift_effective_sample_size=zeros,
        drift_se_u=zeros,
        drift_se_v=zeros,
        diff_x=np.asarray(field.diff_x, dtype=float),
        diff_y=np.asarray(field.diff_y, dtype=float),
        diff_xy=np.asarray(field.diff_xy, dtype=float),
        state_mask=np.asarray(field.state_mask, dtype=bool),
        drift_mask=np.asarray(field.drift_mask, dtype=bool),
        valid_state_rows=int(valid_state_rows),
        valid_drift_rows=int(valid_drift_rows),
        users=int(users),
    )


def core_metrics(module: Any, field, core_mask: np.ndarray, shell_radius: float, users: int, valid_state_rows: int, valid_drift_rows: int) -> Dict[str, float]:
    formal = to_formal_field(module, field, users, valid_state_rows, valid_drift_rows)
    divergence, interior = module.interior_divergence(formal)
    summary, _, _, _ = module._region_metrics(formal, divergence, interior, core_mask, shell_radius, "A_val", 0)
    return {
        "shell_inward_fraction": float(summary.get("flow_weighted_shell_fraction_inward", np.nan)),
        "shell_inward_cosine": float(summary.get("flow_weighted_shell_inward_cosine", np.nan)),
        "core_to_shell_speed_ratio": float(summary.get("flow_core_to_shell_speed_ratio", np.nan)),
        "core_occupancy_mass": float(summary.get("occupancy_mass_fraction", np.nan)),
    }


def transition_rows(assignments: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = assignments[["user_id", "bundle_step_index", "macrostate"]].copy()
    data["user_id"] = pd.to_numeric(data["user_id"], errors="coerce")
    data["bundle_step_index"] = pd.to_numeric(data["bundle_step_index"], errors="coerce")
    data["macrostate"] = pd.to_numeric(data["macrostate"], errors="coerce")
    data = data.dropna(subset=["user_id", "bundle_step_index"]).sort_values(["user_id", "bundle_step_index"], kind="mergesort")
    uid = data["user_id"].to_numpy(dtype=np.int64)
    step = data["bundle_step_index"].to_numpy(dtype=np.int64)
    state = data["macrostate"].to_numpy(dtype=float)
    adjacent = (uid[1:] == uid[:-1]) & (step[1:] == step[:-1] + 1) & np.isfinite(state[:-1]) & np.isfinite(state[1:])
    return uid[:-1][adjacent], state[:-1][adjacent].astype(np.int64), state[1:][adjacent].astype(np.int64)



def point_metrics(module: Any, train_field, val_field, core_mask: np.ndarray, shell_radius: float, train_users: int, val_users: int, train_valid: Tuple[int, int], val_valid: Tuple[int, int]) -> Dict[str, float]:
    comparison = drift_comparison(train_field, val_field)
    contraction = contraction_metrics(val_field)
    core = core_metrics(module, val_field, core_mask, shell_radius, val_users, val_valid[0], val_valid[1])
    return {
        "occupancy_js": float(module.js_divergence(
            np.asarray(train_field.occupancy_probability, dtype=float).ravel() + module.EPS,
            np.asarray(val_field.occupancy_probability, dtype=float).ravel() + module.EPS,
        )),
        "train_validation_mean_local_drift_cosine": comparison["mean_local_drift_cosine"],
        "train_validation_drift_vector_corr": comparison["drift_vector_corr"],
        "train_validation_drift_speed_corr": comparison["drift_speed_corr"],
        "validation_negative_divergence_fraction": contraction["weighted_negative_divergence_fraction"],
        "validation_weighted_mean_divergence": contraction["weighted_mean_divergence"],
        "validation_shell_inward_fraction": core["shell_inward_fraction"],
        "validation_shell_inward_cosine": core["shell_inward_cosine"],
        "validation_core_to_shell_speed_ratio": core["core_to_shell_speed_ratio"],
        "validation_core_occupancy_mass": core["core_occupancy_mass"],
    }



def formal_equivalence_audit(stage1_root: Path, formal_point: Dict[str, float], tolerance: float = 1e-8) -> pd.DataFrame:
    reproducibility = read_table(
        stage1_root / "dynamics" / "fixed_k6_mesostates" / "A_train_A_val_reproducibility_summary"
    ).iloc[0]
    contraction = read_table(
        stage1_root / "dynamics" / "coordinate_analysis" / "MR_PsiA" / "global_field_contraction_summaries"
    )
    contraction = contraction[contraction["split"].astype(str) == "A_val"].iloc[0]
    region = read_table(
        stage1_root / "dynamics" / "candidate_regions" / "MR_PsiA" / "validation_frozen_training_convergence_region"
    ).iloc[0]
    checks = [
        ("occupancy_js", reproducibility, "occupancy_js_divergence"),
        ("train_validation_mean_local_drift_cosine", reproducibility, "mean_local_drift_cosine"),
        ("train_validation_drift_speed_corr", reproducibility, "drift_speed_pearson"),
        ("validation_negative_divergence_fraction", contraction, "weighted_negative_divergence_fraction_interior_only"),
        ("validation_weighted_mean_divergence", contraction, "weighted_mean_local_divergence_interior_only"),
        ("validation_shell_inward_fraction", region, "flow_weighted_shell_fraction_inward"),
        ("validation_shell_inward_cosine", region, "flow_weighted_shell_inward_cosine"),
        ("validation_core_to_shell_speed_ratio", region, "flow_core_to_shell_speed_ratio"),
        ("validation_core_occupancy_mass", region, "occupancy_mass_fraction"),
    ]
    rows = []
    for metric, source, column in checks:
        if column not in source.index:
            raise RuntimeError(f"Formal Stage-1 output is missing {column}.")
        recomputed = float(formal_point[metric])
        archived = float(source[column])
        difference = abs(recomputed - archived)
        passed = bool(np.isclose(recomputed, archived, atol=tolerance, rtol=tolerance, equal_nan=True))
        rows.append({
            "metric": metric,
            "archived_formal_value": archived,
            "recomputed_formal_value": recomputed,
            "absolute_difference": difference,
            "tolerance": tolerance,
            "passed": passed,
        })
    audit = pd.DataFrame(rows)
    return audit

def main() -> None:
    args = parse_args()
    started = time.time()
    stage1_root = args.stage1_root.resolve()
    output_root = args.output_root.resolve()
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    table_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    module = import_module(args.stage1_script, "formal_stage1_empirical_for_robustness")
    required = {
        "FieldStats",
        "downcast_frame",
        "user_balanced_weights",
        "js_divergence",
        "interior_divergence",
        "_region_metrics",
    }
    missing = sorted(name for name in required if not hasattr(module, name))
    if missing:
        raise RuntimeError(f"Formal Stage-1 module is missing: {missing}")

    train = load_panel(stage1_root, "A_train", module)
    val = load_panel(stage1_root, "A_val", module)
    train_all_weights = np.asarray(module.user_balanced_weights(train), dtype=float)
    val_all_weights = np.asarray(module.user_balanced_weights(val), dtype=float)
    train_uid, train_m, train_psi, train_dm, train_dp = panel_arrays(train)
    val_uid, val_m, val_psi, val_dm, val_dp = panel_arrays(val)
    del train, val
    bins = np.linspace(-1.0, 1.0, 41)
    train_state_valid = np.isfinite(train_m) & np.isfinite(train_psi)
    train_drift_valid = train_state_valid & np.isfinite(train_dm) & np.isfinite(train_dp)
    val_state_valid = np.isfinite(val_m) & np.isfinite(val_psi)
    val_drift_valid = val_state_valid & np.isfinite(val_dm) & np.isfinite(val_dp)

    train_formal = SparseFieldAccumulator(train_uid, train_m, train_psi, train_dm, train_dp, bins, train_all_weights, train_all_weights)
    val_formal = SparseFieldAccumulator(val_uid, val_m, val_psi, val_dm, val_dp, bins, val_all_weights, val_all_weights)
    train_strict = SparseFieldAccumulator(
        train_uid, train_m, train_psi, train_dm, train_dp, bins,
        user_equal_row_weights(train_uid, train_state_valid),
        user_equal_row_weights(train_uid, train_drift_valid),
    )
    val_strict = SparseFieldAccumulator(
        val_uid, val_m, val_psi, val_dm, val_dp, bins,
        user_equal_row_weights(val_uid, val_state_valid),
        user_equal_row_weights(val_uid, val_drift_valid),
    )

    field_kwargs = {"min_state_count": 50, "min_cell_users": 5, "min_drift_count": 30}
    train_point = train_formal.point_field(**field_kwargs)
    val_point = val_formal.point_field(**field_kwargs)
    train_strict_point = train_strict.point_field(**field_kwargs)
    val_strict_point = val_strict.point_field(**field_kwargs)

    core_mask_path = stage1_root / "dynamics" / "candidate_regions" / "MR_PsiA" / "A_train_primary_convergence_core_mask.npy"
    core_mask = np.load(core_mask_path).astype(bool)
    shell_radius = 0.35
    formal_point = point_metrics(
        module, train_point, val_point, core_mask, shell_radius,
        train_formal.n_users, val_formal.n_users,
        (int(train_state_valid.sum()), int(train_drift_valid.sum())),
        (int(val_state_valid.sum()), int(val_drift_valid.sum())),
    )
    strict_point = point_metrics(
        module, train_strict_point, val_strict_point, core_mask, shell_radius,
        train_strict.n_users, val_strict.n_users,
        (int(train_state_valid.sum()), int(train_drift_valid.sum())),
        (int(val_state_valid.sum()), int(val_drift_valid.sum())),
    )
    equivalence = formal_equivalence_audit(stage1_root, formal_point)
    equivalence_path = write_table(equivalence, table_root / "empirical_formal_point_equivalence_audit")
    if not bool(equivalence["passed"].all()):
        failed = equivalence.loc[~equivalence["passed"], [
            "metric",
            "archived_formal_value",
            "recomputed_formal_value",
            "absolute_difference",
        ]].to_dict(orient="records")
        raise RuntimeError(f"Formal Stage-1 point-estimate equivalence failed: {failed}")

    strict_rows = []
    for metric in formal_point:
        strict_rows.append({
            "metric": metric,
            "formal_value": formal_point[metric],
            "strict_user_equal_value": strict_point[metric],
            "strict_minus_formal": strict_point[metric] - formal_point[metric],
        })
    strict_path = write_table(pd.DataFrame(strict_rows), table_root / "empirical_strict_user_equal_sensitivity")

    assignments = read_table(stage1_root / "dynamics" / "fixed_k6_mesostates" / "A_val_fixed_k6_assignments")
    trans_uid, trans_cur, trans_next = transition_rows(assignments)
    transition_accumulator = TransitionAccumulator(trans_uid, trans_cur, trans_next, 6, user_values=val_formal.user_values)
    formal_transition = transition_accumulator.point_matrix()
    strict_transition, contributing_users = transition_accumulator.strict_user_equal_matrix()
    transition_rows_out = []
    for state in range(6):
        transition_rows_out.append({
            "macrostate": state,
            "formal_self_transition": formal_transition[state, state],
            "strict_user_equal_self_transition": strict_transition[state, state],
            "strict_minus_formal": strict_transition[state, state] - formal_transition[state, state],
            "strict_contributing_users": int(contributing_users[state]),
            "formal_diagonal_dominant": bool(np.argmax(formal_transition[state]) == state),
            "strict_diagonal_dominant": bool(np.argmax(strict_transition[state]) == state),
        })
    transition_path = write_table(pd.DataFrame(transition_rows_out), table_root / "empirical_transition_user_equal_sensitivity")

    runs = read_table(stage1_root / "dynamics" / "fixed_k6_mesostates" / "A_val_fixed_k6_residence_runs")
    residence_summary = read_table(stage1_root / "dynamics" / "fixed_k6_mesostates" / "A_val_fixed_k6_residence_summary")
    residence_accumulator = WeightedResidenceAccumulator(runs, residence_summary, transition_accumulator, val_formal.user_values)
    residence_point = residence_accumulator.evaluate_chunk(np.ones(val_formal.n_users, dtype=float))
    residence_index_audit = residence_summary.copy()
    residence_index_audit["macrostate"] = pd.to_numeric(residence_index_audit["macrostate"], errors="raise").astype(int)
    residence_index_audit = residence_index_audit.set_index("macrostate")
    residence_audit_rows = []
    for state in range(6):
        archived_values = {
            "rmst_lift": float(residence_index_audit.loc[state, "restricted_mean_residence_lift"]),
            "self_transition": float(residence_index_audit.loc[state, "self_transition"]),
            "tail_excess": float(
                residence_index_audit.loc[state, "observed_tail_probability_at_reference"]
                - residence_index_audit.loc[state, "geometric_tail_probability_at_reference"]
            ),
        }
        for metric, archived in archived_values.items():
            recomputed = float(residence_point[state][metric][0])
            passed = bool(np.isclose(recomputed, archived, atol=1e-8, rtol=1e-8, equal_nan=True))
            residence_audit_rows.append({
                "macrostate": state,
                "metric": metric,
                "archived_formal_value": archived,
                "recomputed_formal_value": recomputed,
                "absolute_difference": abs(recomputed - archived),
                "passed": passed,
            })
    residence_equivalence = pd.DataFrame(residence_audit_rows)
    if not bool(residence_equivalence["passed"].all()):
        failed = residence_equivalence.loc[~residence_equivalence["passed"], ["macrostate", "metric"]].astype(str).agg(":".join, axis=1).tolist()
        raise RuntimeError(f"Formal residence point-estimate equivalence failed: {failed}")
    residence_equivalence_path = write_table(
        residence_equivalence, table_root / "empirical_residence_formal_point_equivalence_audit"
    )

    rng_train = np.random.default_rng(args.seed + 1009)
    rng_val = np.random.default_rng(args.seed + 2003)
    replicate_rows: List[Dict[str, float]] = []
    residence_rows: List[Dict[str, float]] = []
    total_replicates = int(args.bootstrap_replicates)
    for start in range(0, total_replicates, int(args.bootstrap_chunk)):
        batch = min(int(args.bootstrap_chunk), total_replicates - start)
        g_train = rng_train.exponential(1.0, size=(train_formal.n_users, batch))
        g_val = rng_val.exponential(1.0, size=(val_formal.n_users, batch))
        train_totals = train_formal.totals(g_train)
        val_totals = val_formal.totals(g_val)
        residence_batch = residence_accumulator.evaluate_chunk(g_val)
        for column in range(batch):
            replicate = start + column
            train_field = train_formal.field_from_totals(train_totals, column=column, **field_kwargs)
            val_field = val_formal.field_from_totals(val_totals, column=column, **field_kwargs)
            metrics = point_metrics(
                module, train_field, val_field, core_mask, shell_radius,
                train_formal.n_users, val_formal.n_users,
                (int(train_state_valid.sum()), int(train_drift_valid.sum())),
                (int(val_state_valid.sum()), int(val_drift_valid.sum())),
            )
            for metric, value in metrics.items():
                replicate_rows.append({"replicate": replicate, "metric": metric, "value": value})
            for state in range(6):
                for metric, values in residence_batch[state].items():
                    residence_rows.append({
                        "replicate": replicate,
                        "macrostate": state,
                        "metric": metric,
                        "value": float(values[column]),
                    })

    replicate_frame = pd.DataFrame(replicate_rows)
    bootstrap_replicates_path = write_table(replicate_frame, table_root / "empirical_user_multiplier_bootstrap_replicates")
    bootstrap_summary = percentile_summary(replicate_frame, ["metric"])
    bootstrap_summary["formal_point_estimate"] = bootstrap_summary["metric"].map(formal_point)
    bootstrap_summary_path = write_table(bootstrap_summary, table_root / "empirical_user_multiplier_bootstrap_summary")

    residence_frame = pd.DataFrame(residence_rows)
    residence_replicates_path = write_table(residence_frame, table_root / "empirical_residence_user_multiplier_bootstrap_replicates")
    residence_bootstrap_summary = percentile_summary(residence_frame, ["macrostate", "metric"])
    residence_point_map = {}
    residence_index = residence_summary.set_index("macrostate")
    for state in range(6):
        residence_point_map[(state, "rmst_lift")] = float(residence_index.loc[state, "restricted_mean_residence_lift"])
        residence_point_map[(state, "self_transition")] = float(residence_index.loc[state, "self_transition"])
        residence_point_map[(state, "tail_excess")] = float(
            residence_index.loc[state, "observed_tail_probability_at_reference"]
            - residence_index.loc[state, "geometric_tail_probability_at_reference"]
        )
    residence_bootstrap_summary["formal_point_estimate"] = [
        residence_point_map.get((int(row.macrostate), str(row.metric)), np.nan)
        for row in residence_bootstrap_summary.itertuples(index=False)
    ]
    residence_summary_path = write_table(residence_bootstrap_summary, table_root / "empirical_residence_user_multiplier_bootstrap_summary")

    grid_rows: List[Dict[str, float]] = []
    for n_bins in (30, 40, 50):
        grid_bins = np.linspace(-1.0, 1.0, n_bins + 1)
        train_grid = direct_field(
            train_uid, train_m, train_psi, train_dm, train_dp, grid_bins,
            train_all_weights, train_all_weights,
            min_state_count=50, min_cell_users=5, min_drift_count=30,
        )
        val_grid = direct_field(
            val_uid, val_m, val_psi, val_dm, val_dp, grid_bins,
            val_all_weights, val_all_weights,
            min_state_count=50, min_cell_users=5, min_drift_count=30,
        )
        comparison = drift_comparison(train_grid, val_grid)
        contraction = contraction_metrics(val_grid)
        grid_rows.append({
            "setting": f"{n_bins}x{n_bins}",
            "grid_bins_per_axis": n_bins,
            "interior_only_comparison": False,
            **comparison,
            "validation_negative_divergence_fraction": contraction["weighted_negative_divergence_fraction"],
            "validation_weighted_mean_divergence": contraction["weighted_mean_divergence"],
            "validation_supported_cells": int(np.sum(val_grid.drift_mask)),
            "validation_supported_occupancy_mass": float(np.sum(val_grid.occupancy_probability[val_grid.drift_mask])),
        })
        if n_bins == 40:
            comparison_interior = drift_comparison(train_grid, val_grid, interior_cell_mask(40))
            grid_rows.append({
                "setting": "40x40_interior_only",
                "grid_bins_per_axis": 40,
                "interior_only_comparison": True,
                **comparison_interior,
                "validation_negative_divergence_fraction": contraction["weighted_negative_divergence_fraction"],
                "validation_weighted_mean_divergence": contraction["weighted_mean_divergence"],
                "validation_supported_cells": int(np.sum(val_grid.drift_mask & interior_cell_mask(40))),
                "validation_supported_occupancy_mass": float(np.sum(val_grid.occupancy_probability[val_grid.drift_mask & interior_cell_mask(40)])),
            })
    grid_path = write_table(pd.DataFrame(grid_rows), table_root / "empirical_grid_sensitivity")

    source_paths = {
        "A_train_panel": resolve_table(stage1_root / "dynamics" / "student_dynamics_panel_core_A_train"),
        "A_val_panel": resolve_table(stage1_root / "dynamics" / "student_dynamics_panel_core_A_val"),
        "core_mask": core_mask_path,
        "A_val_assignments": resolve_table(stage1_root / "dynamics" / "fixed_k6_mesostates" / "A_val_fixed_k6_assignments"),
        "A_val_residence_runs": resolve_table(stage1_root / "dynamics" / "fixed_k6_mesostates" / "A_val_fixed_k6_residence_runs"),
        "A_val_residence_summary": resolve_table(stage1_root / "dynamics" / "fixed_k6_mesostates" / "A_val_fixed_k6_residence_summary"),
    }
    manifest = {
        "script": Path(__file__).name,
        "stage1_script": str(args.stage1_script.resolve()),
        "stage1_script_sha256": sha256_file(args.stage1_script.resolve()),
        "stage1_root": str(stage1_root),
        "output_root": str(output_root),
        "bootstrap": {
            "method": "paired positive exponential learner multipliers within each split",
            "replicates": total_replicates,
            "A_train_and_A_val_multipliers_independent": True,
            "support_masks_and_training_core_frozen": True,
            "new_p_values": False,
        },
        "strict_user_equal_estimand": {
            "occupancy": "one total unit per user with a finite in-range state",
            "drift": "one total unit per user with a finite in-range transition",
            "role": "sensitivity estimand; formal all-row user balancing remains primary",
        },
        "grid_sensitivity": {
            "settings": ["30x30", "40x40", "50x50", "40x40_interior_only"],
            "range": [-1.0, 1.0],
            "drift_count_threshold": 30,
            "state_count_threshold": 50,
            "cell_user_threshold": 5,
            "core_reselected": False,
            "K_reselected": False,
        },
        "source_audit": {name: {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())} for name, path in source_paths.items()},
        "outputs": {
            "formal_point_equivalence_audit": str(equivalence_path),
            "strict_user_equal": str(strict_path),
            "transition_user_equal": str(transition_path),
            "bootstrap_replicates": str(bootstrap_replicates_path),
            "bootstrap_summary": str(bootstrap_summary_path),
            "residence_formal_point_equivalence_audit": str(residence_equivalence_path),
            "residence_bootstrap_replicates": str(residence_replicates_path),
            "residence_bootstrap_summary": str(residence_summary_path),
            "grid_sensitivity": str(grid_path),
        },
        "elapsed_seconds": float(time.time() - started),
    }
    save_json(manifest, metadata_root / "empirical_robustness_manifest.json")
    print(f"[empirical robustness] completed in {time.time() - started:.1f} seconds")


if __name__ == "__main__":
    main()
