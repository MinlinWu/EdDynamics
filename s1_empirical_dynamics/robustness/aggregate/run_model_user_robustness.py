#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from robustness_common import (
    SparseDriftAccumulator,
    TransitionAccumulator,
    UserPairMoments,
    direct_drift_field,
    drift_comparison,
    import_module,
    interior_cell_mask,
    inward_fraction,
    load_frozen_partition,
    load_json,
    normalize_transition,
    percentile_summary,
    read_table,
    resolve_table,
    save_json,
    sha256_file,
    transition_metrics,
    user_equal_row_weights,
    weighted_pearson,
    weighted_rmse,
    write_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learner-level uncertainty, user weighting and grid sensitivity for frozen mechanism and Event-SSL outputs.")
    parser.add_argument("--stage1-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/stage1"))
    parser.add_argument("--phase3-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/stage2_phase3_confirm"))
    parser.add_argument("--frozen-manifest", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/stage2_phase2_freeze/metadata/phase2_frozen_model_manifest.json"))
    parser.add_argument("--phase3-script", type=Path, required=True)
    parser.add_argument("--phase1-script", type=Path, default=None)
    parser.add_argument("--main-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/stage4_event_ssl/evaluation_predictive_state"))
    parser.add_argument("--time-shuffle-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/stage4_event_ssl_time_shuffle_control/evaluation_on_ordered_inputs"))
    parser.add_argument("--tag-support-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/stage4_event_ssl_tag_support_randomized_control/evaluation"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/supplementary_robustness/models"))
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-chunk", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prediction_path(root: Path, stem: str) -> Path:
    return resolve_table(root / "predictions" / stem)


def load_prediction(root: Path, stem: str) -> pd.DataFrame:
    columns = [
        "user_id", "bundle_step_index", "M", "Psi", "target_M_next", "target_Psi_next",
        "pred_M", "pred_Psi", "pred_next_M", "pred_next_Psi",
    ]
    frame = read_table(root / "predictions" / stem, columns=columns)
    frame["user_id"] = pd.to_numeric(frame["user_id"], errors="raise").astype(np.int64)
    frame["bundle_step_index"] = pd.to_numeric(frame["bundle_step_index"], errors="raise").astype(np.int64)
    return frame.sort_values(["user_id", "bundle_step_index"], kind="mergesort").reset_index(drop=True)


def load_or_reconstruct_mechanism(args: argparse.Namespace, cache_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    phase3 = import_module(args.phase3_script, "formal_phase3_for_robustness")
    frozen_manifest_path = args.frozen_manifest.resolve()
    checksum = phase3.verify_manifest_checksum(frozen_manifest_path, True)
    frozen_manifest = phase3.load_json(frozen_manifest_path)
    params, calibration_payload, _ = phase3.validate_frozen_manifest(frozen_manifest)

    confirmation_manifest_path = args.phase3_root / "metadata" / "phase3_confirmation_manifest.json"
    if not confirmation_manifest_path.exists():
        raise FileNotFoundError(f"Phase-3 confirmation manifest not found: {confirmation_manifest_path}")
    confirmation_manifest = load_json(confirmation_manifest_path)
    if str(confirmation_manifest.get("confirm_split", "")) != "B_confirm":
        raise RuntimeError("Phase-3 confirmation manifest is not for B_confirm.")
    guardrails = dict(confirmation_manifest.get("guardrails", {}))
    forbidden_true = (
        "parameter_search_opened",
        "calibration_reestimated",
        "mechanism_family_reselected",
        "mechanism_parameters_refit",
        "kmeans_refit",
        "macrostate_k_selected",
        "region_redefinition",
        "B_confirm_used_for_update",
    )
    violated = [name for name in forbidden_true if bool(guardrails.get(name, False))]
    if violated:
        raise RuntimeError(f"Phase-3 confirmation guardrails were violated: {violated}")
    recorded_frozen_sha = str(
        dict(confirmation_manifest.get("frozen_manifest_checksum", {})).get("manifest_sha256", "")
    )
    if recorded_frozen_sha and recorded_frozen_sha != str(checksum["manifest_sha256"]):
        raise RuntimeError("Phase-3 confirmation manifest does not match the frozen Phase-2 manifest.")

    formal_path: Optional[Path] = None
    raw = confirmation_manifest.get("outputs", {}).get("full_prediction_table")
    if raw:
        candidate = Path(str(raw))
        if not candidate.is_absolute():
            candidate = (confirmation_manifest_path.parent / candidate).resolve()
        if candidate.exists():
            formal_path = candidate

    if formal_path is not None:
        frame = read_table(formal_path)
        source = {
            "mode": "formal_full_prediction_table",
            "path": str(formal_path.resolve()),
            "sha256": sha256_file(formal_path.resolve()),
            "confirmation_manifest": str(confirmation_manifest_path.resolve()),
            "confirmation_manifest_sha256": sha256_file(confirmation_manifest_path.resolve()),
            "frozen_manifest_checksum": checksum,
        }
    else:
        phase1_path, implementation = phase3.resolve_phase1_script(args.phase1_script, frozen_manifest)
        phase3.prepare_runtime_cache(phase1_path)
        phase1 = phase3.import_phase1_module(phase1_path)
        phase3.validate_phase1_module_contract(phase1)
        phase3.configure_frozen_phase1_module(phase1, params)
        calibration = phase3.calibration_from_manifest(phase1, calibration_payload)
        stage1_root = args.stage1_root.resolve()
        current_kmeans = phase1.audit_stage1_kmeans_contract(stage1_root)
        phase3.compare_kmeans_contracts(
            dict(frozen_manifest.get("stage1_fixed_k6_contract", {})),
            current_kmeans,
        )
        panel, load_manifest = phase3.load_confirm_panel(
            phase1,
            stage1_root,
            "B_confirm",
            float(calibration.eta),
        )
        _, cache, simulation = phase3.evaluate_panel(
            phase1,
            panel,
            params,
            calibration,
            "B_confirm",
        )
        frame = phase1.prediction_frame_from_cache(cache, simulation, "B_confirm", 0, args.seed)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path = write_table(frame, cache_path.with_suffix(""))
        source = {
            "mode": "deterministic_reconstruction_from_frozen_manifest",
            "path": str(cache_path.resolve()),
            "sha256": sha256_file(cache_path.resolve()),
            "confirmation_manifest": str(confirmation_manifest_path.resolve()),
            "confirmation_manifest_sha256": sha256_file(confirmation_manifest_path.resolve()),
            "phase1_implementation": implementation,
            "frozen_manifest_checksum": checksum,
            "stage1_fixed_k6_contract": current_kmeans,
            "confirm_load_manifest": load_manifest,
            "parameter_updates": False,
            "calibration_updates": False,
        }
    required = [
        "user_id",
        "bundle_step_index",
        "M",
        "Psi",
        "target_M_next",
        "target_Psi_next",
        "pred_next_M",
        "pred_next_Psi",
    ]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise RuntimeError(f"Mechanism predictions are missing: {missing}")
    frame = frame[required].copy()
    frame["user_id"] = pd.to_numeric(frame["user_id"], errors="raise").astype(np.int64)
    frame["bundle_step_index"] = pd.to_numeric(
        frame["bundle_step_index"], errors="raise"
    ).astype(np.int64)
    frame = frame.sort_values(
        ["user_id", "bundle_step_index"], kind="mergesort"
    ).reset_index(drop=True)
    return frame, source


def align(reference: pd.DataFrame, other: pd.DataFrame, label: str) -> pd.DataFrame:
    ref_keys = reference[["user_id", "bundle_step_index"]].to_numpy(dtype=np.int64)
    other_keys = other[["user_id", "bundle_step_index"]].to_numpy(dtype=np.int64)
    if ref_keys.shape != other_keys.shape or not np.array_equal(ref_keys, other_keys):
        raise RuntimeError(f"{label} rows do not match the frozen B_confirm row contract.")
    for column in ("M", "Psi", "target_M_next", "target_Psi_next"):
        if column in other.columns:
            first = pd.to_numeric(reference[column], errors="coerce").to_numpy(dtype=float)
            second = pd.to_numeric(other[column], errors="coerce").to_numpy(dtype=float)
            if not np.allclose(first, second, atol=1e-7, rtol=0.0, equal_nan=True):
                raise RuntimeError(f"{label} empirical column changed: {column}")
    return other


def numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def transition_point(accumulator: TransitionAccumulator) -> np.ndarray:
    return accumulator.point_matrix()



def field_point_metrics(fields: Dict[str, Any], reference: Tuple[float, float]) -> Dict[str, float]:
    mechanism = drift_comparison(fields["empirical"], fields["mechanism"])
    main = drift_comparison(fields["empirical"], fields["main_learned"])
    time_shuffle = drift_comparison(fields["empirical"], fields["time_learned"])
    tag = drift_comparison(fields["empirical"], fields["tag_learned"])
    cross = drift_comparison(fields["mechanism"], fields["main_anchor"])
    main_inward = inward_fraction(fields["main_learned"], reference)
    tag_inward = inward_fraction(fields["tag_learned"], reference)
    return {
        "mechanism_drift_vector_corr": mechanism["drift_vector_corr"],
        "event_ssl_learned_drift_vector_corr": main["drift_vector_corr"],
        "event_ssl_learned_weighted_local_cosine": main["occupancy_weighted_local_drift_cosine"],
        "time_shuffle_learned_drift_vector_corr": time_shuffle["drift_vector_corr"],
        "tag_support_learned_drift_vector_corr": tag["drift_vector_corr"],
        "event_ssl_inward_fraction": main_inward,
        "tag_support_inward_fraction": tag_inward,
        "main_minus_tag_inward_fraction": main_inward - tag_inward,
        "cross_model_anchor_drift_vector_corr": cross["drift_vector_corr"],
        "cross_model_anchor_drift_speed_corr": cross["drift_speed_corr"],
        "cross_model_anchor_weighted_local_cosine": cross["occupancy_weighted_local_drift_cosine"],
    }


def transition_point_metrics(accumulators: Dict[str, TransitionAccumulator]) -> Dict[str, float]:
    matrices = {name: transition_point(accumulator) for name, accumulator in accumulators.items()}
    return transition_metrics_from_matrices(matrices)


def transition_metrics_from_matrices(matrices: Dict[str, np.ndarray]) -> Dict[str, float]:
    empirical = matrices["empirical"]
    mechanism = transition_metrics(empirical, matrices["mechanism"])
    main = transition_metrics(empirical, matrices["main_learned"])
    time_shuffle = transition_metrics(empirical, matrices["time_learned"])
    tag = transition_metrics(empirical, matrices["tag_learned"])
    cross = transition_metrics(matrices["mechanism"], matrices["main_anchor"])
    return {
        "mechanism_self_transition_corr": mechanism["self_transition_corr"],
        "event_ssl_self_transition_corr": main["self_transition_corr"],
        "time_shuffle_self_transition_corr": time_shuffle["self_transition_corr"],
        "tag_support_self_transition_corr": tag["self_transition_corr"],
        "cross_model_self_transition_corr": cross["self_transition_corr"],
        "cross_model_transition_mean_row_tv": cross["transition_mean_row_tv"],
    }



def stage4_metric_row(root: Path) -> pd.Series:
    frame = read_table(root / "tables" / "stage4_event_ssl_structural_metrics_all_splits")
    if "split" not in frame.columns:
        raise RuntimeError(f"Stage-4 metric table has no split column: {root}")
    selected = frame[frame["split"].astype(str) == "B_confirm"]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one B_confirm metric row under {root}; found {len(selected)}.")
    return selected.iloc[0]


def formal_equivalence_audit(
    args: argparse.Namespace,
    point: Dict[str, float],
    tolerance: float = 2e-6,
) -> pd.DataFrame:
    main = stage4_metric_row(args.main_root.resolve())
    time_shuffle = stage4_metric_row(args.time_shuffle_root.resolve())
    tag = stage4_metric_row(args.tag_support_root.resolve())
    mechanism = read_table(
        args.phase3_root.resolve() / "tables" / "phase3_B_confirm_structural_alignment_metrics"
    ).iloc[0]
    checks = [
        ("event_ssl_coordinate_corr_M", main, "coordinate_corr_M"),
        ("event_ssl_coordinate_corr_Psi", main, "coordinate_corr_Psi"),
        ("event_ssl_one_step_rmse_M", main, "one_step_rmse_M"),
        ("event_ssl_one_step_rmse_Psi", main, "one_step_rmse_Psi"),
        ("event_ssl_learned_drift_vector_corr", main, "learned_plane_drift_vector_corr"),
        ("event_ssl_learned_weighted_local_cosine", main, "learned_plane_occupancy_weighted_local_drift_cosine"),
        ("event_ssl_inward_fraction", main, "learned_plane_inward_fraction_to_reference"),
        ("event_ssl_self_transition_corr", main, "learned_plane_self_transition_corr"),
        ("time_shuffle_learned_drift_vector_corr", time_shuffle, "learned_plane_drift_vector_corr"),
        ("time_shuffle_self_transition_corr", time_shuffle, "learned_plane_self_transition_corr"),
        ("tag_support_learned_drift_vector_corr", tag, "learned_plane_drift_vector_corr"),
        ("tag_support_inward_fraction", tag, "learned_plane_inward_fraction_to_reference"),
        ("tag_support_self_transition_corr", tag, "learned_plane_self_transition_corr"),
        ("mechanism_drift_vector_corr", mechanism, "drift_vector_corr_MR_PsiA"),
    ]
    rows = []
    for metric, source, column in checks:
        if column not in source.index:
            raise RuntimeError(f"Formal output is missing {column} for {metric}.")
        recomputed = float(point[metric])
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
    if not bool(audit["passed"].all()):
        failed = audit.loc[~audit["passed"], "metric"].astype(str).tolist()
        raise RuntimeError(f"Formal model point-estimate equivalence failed: {failed}")
    return audit

def main() -> None:
    args = parse_args()
    started = time.time()
    output_root = args.output_root.resolve()
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    cache_root = output_root / "cache"
    for directory in (table_root, metadata_root, cache_root):
        directory.mkdir(parents=True, exist_ok=True)

    main_frame = load_prediction(args.main_root.resolve(), "stage4_event_ssl_predictions_B_confirm")
    time_frame = align(main_frame, load_prediction(args.time_shuffle_root.resolve(), "stage4_event_ssl_predictions_B_confirm"), "time-shuffle")
    tag_frame = align(main_frame, load_prediction(args.tag_support_root.resolve(), "stage4_event_ssl_predictions_B_confirm"), "tag/support")
    mechanism_frame, mechanism_source = load_or_reconstruct_mechanism(args, cache_root / "mechanism_B_confirm_full_predictions.parquet")
    mechanism_frame = align(main_frame, mechanism_frame, "mechanism")

    uid = main_frame["user_id"].to_numpy(dtype=np.int64)
    m = numeric(main_frame, "M")
    psi = numeric(main_frame, "Psi")
    next_m = numeric(main_frame, "target_M_next")
    next_psi = numeric(main_frame, "target_Psi_next")
    main_m = numeric(main_frame, "pred_M")
    main_psi = numeric(main_frame, "pred_Psi")
    main_next_m = numeric(main_frame, "pred_next_M")
    main_next_psi = numeric(main_frame, "pred_next_Psi")
    time_m = numeric(time_frame, "pred_M")
    time_psi = numeric(time_frame, "pred_Psi")
    time_next_m = numeric(time_frame, "pred_next_M")
    time_next_psi = numeric(time_frame, "pred_next_Psi")
    tag_m = numeric(tag_frame, "pred_M")
    tag_psi = numeric(tag_frame, "pred_Psi")
    tag_next_m = numeric(tag_frame, "pred_next_M")
    tag_next_psi = numeric(tag_frame, "pred_next_Psi")
    mech_next_m = numeric(mechanism_frame, "pred_next_M")
    mech_next_psi = numeric(mechanism_frame, "pred_next_Psi")
    del main_frame, time_frame, tag_frame, mechanism_frame

    user_values = np.unique(uid)
    formal_weights = user_equal_row_weights(uid)
    bins = np.linspace(-1.0, 1.0, 41)
    field_specs = {
        "empirical": (m, psi, next_m - m, next_psi - psi),
        "mechanism": (m, psi, mech_next_m - m, mech_next_psi - psi),
        "main_anchor": (m, psi, main_next_m - m, main_next_psi - psi),
        "main_learned": (main_m, main_psi, main_next_m - main_m, main_next_psi - main_psi),
        "time_learned": (time_m, time_psi, time_next_m - time_m, time_next_psi - time_psi),
        "tag_learned": (tag_m, tag_psi, tag_next_m - tag_m, tag_next_psi - tag_psi),
    }
    field_accumulators = {
        name: SparseDriftAccumulator(uid, x, y, dx, dy, bins, formal_weights, user_values=user_values)
        for name, (x, y, dx, dy) in field_specs.items()
    }
    point_fields = {name: accumulator.point_field(min_drift_count=30) for name, accumulator in field_accumulators.items()}

    main_manifest = load_json(args.main_root.resolve() / "metadata" / "stage4_event_ssl_evaluation_manifest.json")
    convergence = dict(main_manifest.get("convergence_reference", {}))
    reference = (float(convergence["reference_M"]), float(convergence["reference_Psi"]))
    point = field_point_metrics(point_fields, reference)

    partition = load_frozen_partition(args.stage1_root.resolve())
    empirical_cur = partition.labels(np.column_stack([m, psi]))
    empirical_next = partition.labels(np.column_stack([next_m, next_psi]))
    mechanism_next = partition.labels(np.column_stack([mech_next_m, mech_next_psi]))
    main_cur = partition.labels(np.column_stack([main_m, main_psi]))
    main_next = partition.labels(np.column_stack([main_next_m, main_next_psi]))
    time_cur = partition.labels(np.column_stack([time_m, time_psi]))
    time_next = partition.labels(np.column_stack([time_next_m, time_next_psi]))
    tag_cur = partition.labels(np.column_stack([tag_m, tag_psi]))
    tag_next = partition.labels(np.column_stack([tag_next_m, tag_next_psi]))
    transition_accumulators = {
        "empirical": TransitionAccumulator(uid, empirical_cur, empirical_next, 6, user_values=user_values),
        "mechanism": TransitionAccumulator(uid, empirical_cur, mechanism_next, 6, user_values=user_values),
        "main_anchor": TransitionAccumulator(uid, empirical_cur, main_next, 6, user_values=user_values),
        "main_learned": TransitionAccumulator(uid, main_cur, main_next, 6, user_values=user_values),
        "time_learned": TransitionAccumulator(uid, time_cur, time_next, 6, user_values=user_values),
        "tag_learned": TransitionAccumulator(uid, tag_cur, tag_next, 6, user_values=user_values),
    }
    point.update(transition_point_metrics(transition_accumulators))

    moment_accumulators = {
        "event_ssl_coordinate_corr_M": UserPairMoments(uid, main_m, m, user_values=user_values),
        "event_ssl_coordinate_corr_Psi": UserPairMoments(uid, main_psi, psi, user_values=user_values),
        "event_ssl_one_step_rmse_M": UserPairMoments(uid, main_next_m, next_m, user_values=user_values),
        "event_ssl_one_step_rmse_Psi": UserPairMoments(uid, main_next_psi, next_psi, user_values=user_values),
    }
    ones = np.ones(len(user_values), dtype=float)
    for name, accumulator in moment_accumulators.items():
        corr, rmse = accumulator.evaluate(ones)
        point[name] = float(corr[0]) if "corr" in name else float(rmse[0])
    equivalence = formal_equivalence_audit(args, point)
    equivalence_path = write_table(equivalence, table_root / "model_formal_point_equivalence_audit")

    rng = np.random.default_rng(args.seed + 3011)
    replicate_rows: List[Dict[str, float]] = []
    total_replicates = int(args.bootstrap_replicates)
    for start in range(0, total_replicates, int(args.bootstrap_chunk)):
        batch = min(int(args.bootstrap_chunk), total_replicates - start)
        multipliers = rng.exponential(1.0, size=(len(user_values), batch))
        field_totals = {name: accumulator.totals(multipliers) for name, accumulator in field_accumulators.items()}
        transition_totals = {name: accumulator.counts(multipliers) for name, accumulator in transition_accumulators.items()}
        moment_values = {name: accumulator.evaluate(multipliers) for name, accumulator in moment_accumulators.items()}
        for column in range(batch):
            replicate = start + column
            fields = {
                name: accumulator.field_from_totals(field_totals[name], column=column, min_drift_count=30)
                for name, accumulator in field_accumulators.items()
            }
            metrics = field_point_metrics(fields, reference)
            matrices = {
                name: normalize_transition(transition_totals[name][:, column].reshape(6, 6))
                for name in transition_accumulators
            }
            metrics.update(transition_metrics_from_matrices(matrices))
            for name, (correlation, rmse) in moment_values.items():
                metrics[name] = float(correlation[column]) if "corr" in name else float(rmse[column])
            for metric, value in metrics.items():
                replicate_rows.append({"replicate": replicate, "metric": metric, "value": value})

    replicate_frame = pd.DataFrame(replicate_rows)
    replicate_path = write_table(replicate_frame, table_root / "model_user_multiplier_bootstrap_replicates")
    summary = percentile_summary(replicate_frame, ["metric"])
    summary["formal_point_estimate"] = summary["metric"].map(point)
    summary_path = write_table(summary, table_root / "model_user_multiplier_bootstrap_summary")

    strict_weights = user_equal_row_weights(uid)
    strict_rows: List[Dict[str, Any]] = []
    strict_metric_values = {
        "event_ssl_coordinate_corr_M": weighted_pearson(main_m, m, strict_weights),
        "event_ssl_coordinate_corr_Psi": weighted_pearson(main_psi, psi, strict_weights),
        "event_ssl_one_step_rmse_M": weighted_rmse(main_next_m, next_m, strict_weights),
        "event_ssl_one_step_rmse_Psi": weighted_rmse(main_next_psi, next_psi, strict_weights),
    }
    for metric, strict_value in strict_metric_values.items():
        strict_rows.append({
            "analysis": "interval_metric",
            "metric": metric,
            "formal_value": point[metric],
            "strict_user_equal_value": strict_value,
            "strict_minus_formal": strict_value - point[metric],
        })

    strict_matrices = {}
    strict_contributing = {}
    formal_matrices = {}
    for name, accumulator in transition_accumulators.items():
        formal_matrices[name] = accumulator.point_matrix()
        strict_matrices[name], strict_contributing[name] = accumulator.strict_user_equal_matrix()
    formal_transition_metrics = transition_metrics_from_matrices(formal_matrices)
    strict_transition_metrics = transition_metrics_from_matrices(strict_matrices)
    for metric, formal_value in formal_transition_metrics.items():
        strict_value = strict_transition_metrics[metric]
        strict_rows.append({
            "analysis": "transition_metric",
            "metric": metric,
            "formal_value": formal_value,
            "strict_user_equal_value": strict_value,
            "strict_minus_formal": strict_value - formal_value,
        })
    user_mass = pd.Series(formal_weights).groupby(pd.Series(uid)).sum().to_numpy(dtype=float)
    strict_rows.append({
        "analysis": "field_weight_audit",
        "metric": "maximum_absolute_user_mass_minus_one",
        "formal_value": float(np.max(np.abs(user_mass - 1.0))),
        "strict_user_equal_value": 0.0,
        "strict_minus_formal": -float(np.max(np.abs(user_mass - 1.0))),
    })
    strict_path = write_table(pd.DataFrame(strict_rows), table_root / "model_strict_user_equal_sensitivity")

    contributing_rows = []
    for name, counts in strict_contributing.items():
        for state, count in enumerate(counts):
            contributing_rows.append({"transition_view": name, "macrostate": state, "strict_contributing_users": int(count)})
    contributing_path = write_table(pd.DataFrame(contributing_rows), table_root / "model_strict_transition_contributing_users")

    grid_rows: List[Dict[str, Any]] = []
    for n_bins in (30, 40, 50):
        grid_bins = np.linspace(-1.0, 1.0, n_bins + 1)
        fields = {
            name: direct_drift_field(x, y, dx, dy, grid_bins, formal_weights, min_drift_count=30)
            for name, (x, y, dx, dy) in field_specs.items()
            if name in {"empirical", "mechanism", "main_anchor", "main_learned"}
        }
        comparisons = {
            "mechanism_vs_empirical": drift_comparison(fields["empirical"], fields["mechanism"]),
            "event_ssl_learned_vs_empirical": drift_comparison(fields["empirical"], fields["main_learned"]),
            "mechanism_vs_event_ssl_anchor": drift_comparison(fields["mechanism"], fields["main_anchor"]),
        }
        for comparison_name, values in comparisons.items():
            grid_rows.append({
                "setting": f"{n_bins}x{n_bins}",
                "grid_bins_per_axis": n_bins,
                "interior_only": False,
                "comparison": comparison_name,
                **values,
            })
        if n_bins == 40:
            interior = interior_cell_mask(40)
            comparisons = {
                "mechanism_vs_empirical": drift_comparison(fields["empirical"], fields["mechanism"], interior),
                "event_ssl_learned_vs_empirical": drift_comparison(fields["empirical"], fields["main_learned"], interior),
                "mechanism_vs_event_ssl_anchor": drift_comparison(fields["mechanism"], fields["main_anchor"], interior),
            }
            for comparison_name, values in comparisons.items():
                grid_rows.append({
                    "setting": "40x40_interior_only",
                    "grid_bins_per_axis": 40,
                    "interior_only": True,
                    "comparison": comparison_name,
                    **values,
                })
    grid_path = write_table(pd.DataFrame(grid_rows), table_root / "model_grid_sensitivity")

    source_paths = {
        "main_predictions": prediction_path(args.main_root.resolve(), "stage4_event_ssl_predictions_B_confirm"),
        "time_shuffle_predictions": prediction_path(args.time_shuffle_root.resolve(), "stage4_event_ssl_predictions_B_confirm"),
        "tag_support_predictions": prediction_path(args.tag_support_root.resolve(), "stage4_event_ssl_predictions_B_confirm"),
        "frozen_manifest": args.frozen_manifest.resolve(),
        "phase3_script": args.phase3_script.resolve(),
        "fixed_k6_metadata": args.stage1_root.resolve() / "dynamics" / "fixed_k6_mesostates" / "fixed_k6_model_metadata.json",
        "fixed_k6_centers": resolve_table(args.stage1_root.resolve() / "dynamics" / "fixed_k6_mesostates" / "fixed_k6_centers"),
    }
    manifest = {
        "script": Path(__file__).name,
        "output_root": str(output_root),
        "primary_state": ["M", "Psi"],
        "confirmation_split": "B_confirm",
        "mechanism_prediction_source": mechanism_source,
        "bootstrap": {
            "method": "positive exponential learner multipliers",
            "replicates": total_replicates,
            "same_multiplier_for_all_frozen_models_and_controls": True,
            "models_refit": False,
            "probes_refit": False,
            "KMeans_refit": False,
            "new_p_values": False,
        },
        "cross_model_contract": "mechanism and Event-SSL anchor fields use the same empirical current M-Psi states",
        "statewise_correlation_boundary": (
            "Multiplier intervals for self-transition correlations reflect learner-composition variation in "
            "the six frozen state probabilities; the six states are not treated as independent samples."
        ),
        "strict_user_equal_contract": {
            "interval_metrics": "one total weight unit per evaluated user",
            "transitions": "per-user row-normalized transition rows averaged over users visiting each origin state",
            "field_note": "field rows are already strictly user balanced by construction and are audited rather than duplicated",
        },
        "grid_sensitivity": {
            "settings": ["30x30", "40x40", "50x50", "40x40_interior_only"],
            "drift_count_threshold": 30,
            "core_reselected": False,
            "partition_refit": False,
        },
        "source_audit": {name: {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())} for name, path in source_paths.items()},
        "outputs": {
            "formal_point_equivalence_audit": str(equivalence_path),
            "bootstrap_replicates": str(replicate_path),
            "bootstrap_summary": str(summary_path),
            "strict_user_equal": str(strict_path),
            "strict_transition_users": str(contributing_path),
            "grid_sensitivity": str(grid_path),
        },
        "elapsed_seconds": float(time.time() - started),
    }
    save_json(manifest, metadata_root / "model_robustness_manifest.json")
    print(f"[model robustness] completed in {time.time() - started:.1f} seconds")


if __name__ == "__main__":
    main()
