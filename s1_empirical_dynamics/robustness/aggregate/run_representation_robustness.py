#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from robustness_common import (
    SparseDriftAccumulator,
    TransitionAccumulator,
    UserPairMoments,
    direct_drift_field,
    drift_comparison,
    interior_cell_mask,
    load_frozen_partition,
    normalize_transition,
    pearson,
    percentile_summary,
    read_table,
    resolve_table,
    save_json,
    sha256_file,
    stage5_domain_scores,
    transition_metrics,
    user_equal_row_weights,
    user_slices,
    weighted_pearson,
    weighted_rmse,
    write_table,
)

REPRESENTATIONS = ("full_hidden", "macro_only", "residual_hidden")
RAW_METRICS = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learner-level and descriptive-score robustness for frozen Stage-5 representations.")
    parser.add_argument("--stage1-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/stage1"))
    parser.add_argument("--macro-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/stage5_event_ssl_macro_sufficiency/evaluation"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/supplementary_robustness/representations"))
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-chunk", type=int, default=25)
    parser.add_argument("--permutation-replicates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prediction_path(root: Path, representation: str) -> Path:
    return resolve_table(root / "predictions" / f"stage5_macro_sufficiency_predictions_{representation}_B_confirm")


def load_prediction(root: Path, representation: str) -> pd.DataFrame:
    columns = [
        "user_id",
        "bundle_step_index",
        "M",
        "Psi",
        "target_M_next",
        "target_Psi_next",
        "pred_M",
        "pred_Psi",
        "pred_next_M",
        "pred_next_Psi",
    ]
    frame = read_table(root / "predictions" / f"stage5_macro_sufficiency_predictions_{representation}_B_confirm", columns=columns)
    frame["user_id"] = pd.to_numeric(frame["user_id"], errors="raise").astype(np.int64)
    frame["bundle_step_index"] = pd.to_numeric(frame["bundle_step_index"], errors="raise").astype(np.int64)
    return frame.sort_values(["user_id", "bundle_step_index"], kind="mergesort").reset_index(drop=True)


def align(reference: pd.DataFrame, other: pd.DataFrame, label: str) -> pd.DataFrame:
    keys = reference[["user_id", "bundle_step_index"]].to_numpy(dtype=np.int64)
    other_keys = other[["user_id", "bundle_step_index"]].to_numpy(dtype=np.int64)
    if keys.shape != other_keys.shape or not np.array_equal(keys, other_keys):
        raise RuntimeError(f"{label} rows do not match the frozen Stage-5 B_confirm contract.")
    for column in ("M", "Psi", "target_M_next", "target_Psi_next"):
        first = pd.to_numeric(reference[column], errors="coerce").to_numpy(dtype=float)
        second = pd.to_numeric(other[column], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(first, second, atol=1e-7, rtol=0.0, equal_nan=True):
            raise RuntimeError(f"{label} changed the empirical column {column}.")
    return other


def numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def transition_metric_values(empirical: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    values = transition_metrics(empirical, predicted)
    return {
        "learned_plane_transition_mean_row_tv": values["transition_mean_row_tv"],
        "learned_plane_self_transition_corr": values["self_transition_corr"],
        "learned_plane_diagonal_dominance_match_fraction": values["diagonal_dominance_match_fraction"],
        "learned_plane_top_transition_edge_overlap": values["top_transition_edge_overlap"],
    }


def raw_point_metrics(
    empirical_field: Any,
    predicted_field: Any,
    empirical_transition: np.ndarray,
    predicted_transition: np.ndarray,
    current_m: np.ndarray,
    current_psi: np.ndarray,
    next_m: np.ndarray,
    next_psi: np.ndarray,
    pred_m: np.ndarray,
    pred_psi: np.ndarray,
    pred_next_m: np.ndarray,
    pred_next_psi: np.ndarray,
    row_weights: np.ndarray | None = None,
) -> Dict[str, float]:
    field = drift_comparison(empirical_field, predicted_field)
    if row_weights is None:
        coordinate_corr_m = pearson(pred_m, current_m)
        coordinate_corr_psi = pearson(pred_psi, current_psi)
        one_step_rmse_m = float(np.sqrt(np.nanmean((pred_next_m - next_m) ** 2)))
        one_step_rmse_psi = float(np.sqrt(np.nanmean((pred_next_psi - next_psi) ** 2)))
    else:
        coordinate_corr_m = weighted_pearson(pred_m, current_m, row_weights)
        coordinate_corr_psi = weighted_pearson(pred_psi, current_psi, row_weights)
        one_step_rmse_m = weighted_rmse(pred_next_m, next_m, row_weights)
        one_step_rmse_psi = weighted_rmse(pred_next_psi, next_psi, row_weights)
    output = {
        "coordinate_corr_M": coordinate_corr_m,
        "coordinate_corr_Psi": coordinate_corr_psi,
        "one_step_rmse_M": one_step_rmse_m,
        "one_step_rmse_Psi": one_step_rmse_psi,
        "learned_plane_drift_vector_corr": field["drift_vector_corr"],
        "learned_plane_occupancy_weighted_local_drift_cosine": field["occupancy_weighted_local_drift_cosine"],
    }
    output.update(transition_metric_values(empirical_transition, predicted_transition))
    return output



def formal_equivalence_audit(
    macro_root: Path,
    point_metrics: Mapping[str, Mapping[str, float]],
    tolerance: float = 2e-6,
) -> pd.DataFrame:
    formal = read_table(macro_root / "tables" / "stage5_macro_sufficiency_metrics_all_splits")
    required = {"split", "representation"}
    if not required.issubset(formal.columns):
        raise RuntimeError(f"Formal Stage-5 metric table is missing: {sorted(required.difference(formal.columns))}")
    formal = formal[formal["split"].astype(str) == "B_confirm"].copy()
    rows = []
    for representation in REPRESENTATIONS:
        selected = formal[formal["representation"].astype(str) == representation]
        if len(selected) != 1:
            raise RuntimeError(f"Expected one formal B_confirm row for {representation}; found {len(selected)}.")
        source = selected.iloc[0]
        for metric in RAW_METRICS:
            if metric not in source.index:
                raise RuntimeError(f"Formal Stage-5 output is missing {metric} for {representation}.")
            archived = float(source[metric])
            recomputed = float(point_metrics[representation][metric])
            difference = abs(recomputed - archived)
            passed = bool(np.isclose(recomputed, archived, atol=tolerance, rtol=tolerance, equal_nan=True))
            rows.append({
                "representation": representation,
                "metric": metric,
                "archived_formal_value": archived,
                "recomputed_formal_value": recomputed,
                "absolute_difference": difference,
                "tolerance": tolerance,
                "passed": passed,
            })
    audit = pd.DataFrame(rows)
    if not bool(audit["passed"].all()):
        failed = audit.loc[~audit["passed"], ["representation", "metric"]].astype(str).agg(":".join, axis=1).tolist()
        raise RuntimeError(f"Formal Stage-5 point-estimate equivalence failed: {failed}")
    return audit

def score_contracts() -> Sequence[Dict[str, Any]]:
    return (
        {"contract": "S0_formal", "rmse_scale": 0.15, "domains": ("coordinate_score", "closure_score", "drift_score", "transition_score")},
        {"contract": "S1_rmse_0p10", "rmse_scale": 0.10, "domains": ("coordinate_score", "closure_score", "drift_score", "transition_score")},
        {"contract": "S2_rmse_0p20", "rmse_scale": 0.20, "domains": ("coordinate_score", "closure_score", "drift_score", "transition_score")},
        {"contract": "S3_leave_coordinate_out", "rmse_scale": 0.15, "domains": ("closure_score", "drift_score", "transition_score")},
        {"contract": "S4_leave_closure_out", "rmse_scale": 0.15, "domains": ("coordinate_score", "drift_score", "transition_score")},
        {"contract": "S5_leave_drift_out", "rmse_scale": 0.15, "domains": ("coordinate_score", "closure_score", "transition_score")},
        {"contract": "S6_leave_transition_out", "rmse_scale": 0.15, "domains": ("coordinate_score", "closure_score", "drift_score")},
    )


def contract_score(metrics: Mapping[str, float], contract: Mapping[str, Any]) -> Tuple[float, Dict[str, float]]:
    domains = stage5_domain_scores(metrics, rmse_scale=float(contract["rmse_scale"]))
    selected = [float(domains[name]) for name in contract["domains"] if np.isfinite(domains.get(name, np.nan))]
    return (float(np.mean(selected)) if selected else float("nan")), domains


def within_user_permutation_indices(user_id: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, int]:
    current = np.arange(len(user_id), dtype=np.int64)
    next_index = np.arange(len(user_id), dtype=np.int64)
    singleton_rows = 0
    for start, stop in user_slices(user_id):
        length = stop - start
        if length <= 1:
            singleton_rows += length
            continue
        source = np.arange(start, stop, dtype=np.int64)
        current[start:stop] = rng.permutation(source)
        next_index[start:stop] = rng.permutation(source)
    return current, next_index, singleton_rows


def main() -> None:
    args = parse_args()
    started = time.time()
    root = args.macro_root.resolve()
    output_root = args.output_root.resolve()
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    table_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    frames: Dict[str, pd.DataFrame] = {}
    frames["full_hidden"] = load_prediction(root, "full_hidden")
    for representation in ("macro_only", "residual_hidden"):
        frames[representation] = align(frames["full_hidden"], load_prediction(root, representation), representation)

    reference = frames["full_hidden"]
    uid = reference["user_id"].to_numpy(dtype=np.int64)
    user_values = np.unique(uid)
    current_m = numeric(reference, "M")
    current_psi = numeric(reference, "Psi")
    next_m = numeric(reference, "target_M_next")
    next_psi = numeric(reference, "target_Psi_next")
    formal_weights = user_equal_row_weights(uid)
    bins = np.linspace(-1.0, 1.0, 41)

    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    for representation, frame in frames.items():
        predictions[representation] = {
            "m": numeric(frame, "pred_M"),
            "psi": numeric(frame, "pred_Psi"),
            "next_m": numeric(frame, "pred_next_M"),
            "next_psi": numeric(frame, "pred_next_Psi"),
        }
    del frames, reference

    empirical_accumulator = SparseDriftAccumulator(
        uid,
        current_m,
        current_psi,
        next_m - current_m,
        next_psi - current_psi,
        bins,
        formal_weights,
        user_values=user_values,
    )
    representation_accumulators = {
        representation: SparseDriftAccumulator(
            uid,
            values["m"],
            values["psi"],
            values["next_m"] - values["m"],
            values["next_psi"] - values["psi"],
            bins,
            formal_weights,
            user_values=user_values,
        )
        for representation, values in predictions.items()
    }
    empirical_field = empirical_accumulator.point_field(min_drift_count=30)
    representation_fields = {
        representation: accumulator.point_field(min_drift_count=30)
        for representation, accumulator in representation_accumulators.items()
    }

    partition = load_frozen_partition(args.stage1_root.resolve())
    empirical_current_label = partition.labels(np.column_stack([current_m, current_psi]))
    empirical_next_label = partition.labels(np.column_stack([next_m, next_psi]))
    empirical_transition_accumulator = TransitionAccumulator(
        uid, empirical_current_label, empirical_next_label, 6, user_values=user_values
    )
    representation_transition_accumulators: Dict[str, TransitionAccumulator] = {}
    for representation, values in predictions.items():
        representation_transition_accumulators[representation] = TransitionAccumulator(
            uid,
            partition.labels(np.column_stack([values["m"], values["psi"]])),
            partition.labels(np.column_stack([values["next_m"], values["next_psi"]])),
            6,
            user_values=user_values,
        )
    empirical_transition = empirical_transition_accumulator.point_matrix()
    representation_transitions = {
        representation: accumulator.point_matrix()
        for representation, accumulator in representation_transition_accumulators.items()
    }

    moment_accumulators: Dict[Tuple[str, str], UserPairMoments] = {}
    for representation, values in predictions.items():
        moment_accumulators[(representation, "coordinate_corr_M")] = UserPairMoments(uid, values["m"], current_m, user_values=user_values)
        moment_accumulators[(representation, "coordinate_corr_Psi")] = UserPairMoments(uid, values["psi"], current_psi, user_values=user_values)
        moment_accumulators[(representation, "one_step_rmse_M")] = UserPairMoments(uid, values["next_m"], next_m, user_values=user_values)
        moment_accumulators[(representation, "one_step_rmse_Psi")] = UserPairMoments(uid, values["next_psi"], next_psi, user_values=user_values)

    point_metrics: Dict[str, Dict[str, float]] = {}
    point_rows: List[Dict[str, Any]] = []
    for representation, values in predictions.items():
        point = raw_point_metrics(
            empirical_field,
            representation_fields[representation],
            empirical_transition,
            representation_transitions[representation],
            current_m,
            current_psi,
            next_m,
            next_psi,
            values["m"],
            values["psi"],
            values["next_m"],
            values["next_psi"],
        )
        point_metrics[representation] = point
        point_rows.extend({"representation": representation, "metric": metric, "value": value} for metric, value in point.items())
    point_path = write_table(pd.DataFrame(point_rows), table_root / "stage5_raw_metric_point_estimates")
    equivalence = formal_equivalence_audit(root, point_metrics)
    equivalence_path = write_table(equivalence, table_root / "stage5_formal_point_equivalence_audit")

    rng = np.random.default_rng(args.seed + 4013)
    bootstrap_rows: List[Dict[str, Any]] = []
    total_replicates = int(args.bootstrap_replicates)
    for start in range(0, total_replicates, int(args.bootstrap_chunk)):
        batch = min(int(args.bootstrap_chunk), total_replicates - start)
        multipliers = rng.exponential(1.0, size=(len(user_values), batch))
        empirical_field_totals = empirical_accumulator.totals(multipliers)
        representation_field_totals = {
            representation: accumulator.totals(multipliers)
            for representation, accumulator in representation_accumulators.items()
        }
        empirical_transition_totals = empirical_transition_accumulator.counts(multipliers)
        representation_transition_totals = {
            representation: accumulator.counts(multipliers)
            for representation, accumulator in representation_transition_accumulators.items()
        }
        moment_values = {
            key: accumulator.evaluate(multipliers)
            for key, accumulator in moment_accumulators.items()
        }
        for column in range(batch):
            replicate = start + column
            empirical_field_b = empirical_accumulator.field_from_totals(
                empirical_field_totals, column=column, min_drift_count=30
            )
            empirical_transition_b = normalize_transition(
                empirical_transition_totals[:, column].reshape(6, 6)
            )
            replicate_metrics: Dict[str, Dict[str, float]] = {}
            for representation in REPRESENTATIONS:
                predicted_field_b = representation_accumulators[representation].field_from_totals(
                    representation_field_totals[representation], column=column, min_drift_count=30
                )
                predicted_transition_b = normalize_transition(
                    representation_transition_totals[representation][:, column].reshape(6, 6)
                )
                field_values = drift_comparison(empirical_field_b, predicted_field_b)
                transition_values = transition_metric_values(empirical_transition_b, predicted_transition_b)
                values = {
                    "coordinate_corr_M": float(moment_values[(representation, "coordinate_corr_M")][0][column]),
                    "coordinate_corr_Psi": float(moment_values[(representation, "coordinate_corr_Psi")][0][column]),
                    "one_step_rmse_M": float(moment_values[(representation, "one_step_rmse_M")][1][column]),
                    "one_step_rmse_Psi": float(moment_values[(representation, "one_step_rmse_Psi")][1][column]),
                    "learned_plane_drift_vector_corr": field_values["drift_vector_corr"],
                    "learned_plane_occupancy_weighted_local_drift_cosine": field_values["occupancy_weighted_local_drift_cosine"],
                    **transition_values,
                }
                replicate_metrics[representation] = values
                bootstrap_rows.extend(
                    {
                        "replicate": replicate,
                        "comparison": "absolute",
                        "representation": representation,
                        "metric": metric,
                        "value": value,
                    }
                    for metric, value in values.items()
                )
            for metric in RAW_METRICS:
                bootstrap_rows.append({
                    "replicate": replicate,
                    "comparison": "macro_only_minus_full_hidden",
                    "representation": "macro_only_minus_full_hidden",
                    "metric": metric,
                    "value": replicate_metrics["macro_only"][metric] - replicate_metrics["full_hidden"][metric],
                })

    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    bootstrap_replicates_path = write_table(bootstrap_frame, table_root / "stage5_user_multiplier_bootstrap_replicates")
    bootstrap_summary = percentile_summary(bootstrap_frame, ["comparison", "representation", "metric"])
    formal_map = {
        ("absolute", representation, metric): point_metrics[representation][metric]
        for representation in REPRESENTATIONS
        for metric in RAW_METRICS
    }
    formal_map.update({
        ("macro_only_minus_full_hidden", "macro_only_minus_full_hidden", metric): point_metrics["macro_only"][metric] - point_metrics["full_hidden"][metric]
        for metric in RAW_METRICS
    })
    bootstrap_summary["formal_point_estimate"] = [
        formal_map.get((str(row.comparison), str(row.representation), str(row.metric)), np.nan)
        for row in bootstrap_summary.itertuples(index=False)
    ]
    bootstrap_summary_path = write_table(bootstrap_summary, table_root / "stage5_user_multiplier_bootstrap_summary")

    strict_weights = user_equal_row_weights(uid)
    strict_rows: List[Dict[str, Any]] = []
    formal_user_mass = pd.Series(formal_weights).groupby(pd.Series(uid)).sum().to_numpy(dtype=float)
    strict_rows.append({
        "analysis": "field_weight_audit",
        "representation": "all",
        "metric": "maximum_absolute_user_mass_minus_one",
        "formal_value": float(np.max(np.abs(formal_user_mass - 1.0))),
        "strict_user_equal_value": 0.0,
        "strict_minus_formal": -float(np.max(np.abs(formal_user_mass - 1.0))),
    })
    strict_empirical_transition, empirical_contributing = empirical_transition_accumulator.strict_user_equal_matrix()
    strict_transition_users: List[Dict[str, Any]] = []
    for state, count in enumerate(empirical_contributing):
        strict_transition_users.append({"transition_view": "empirical", "macrostate": state, "contributing_users": int(count)})
    for representation, values in predictions.items():
        strict_transition, contributing = representation_transition_accumulators[representation].strict_user_equal_matrix()
        strict_values = {
            "coordinate_corr_M": weighted_pearson(values["m"], current_m, strict_weights),
            "coordinate_corr_Psi": weighted_pearson(values["psi"], current_psi, strict_weights),
            "one_step_rmse_M": weighted_rmse(values["next_m"], next_m, strict_weights),
            "one_step_rmse_Psi": weighted_rmse(values["next_psi"], next_psi, strict_weights),
            **transition_metric_values(strict_empirical_transition, strict_transition),
        }
        for metric, strict_value in strict_values.items():
            formal_value = point_metrics[representation][metric]
            strict_rows.append({
                "analysis": "interval_metric" if metric.startswith(("coordinate", "one_step")) else "transition_metric",
                "representation": representation,
                "metric": metric,
                "formal_value": formal_value,
                "strict_user_equal_value": strict_value,
                "strict_minus_formal": strict_value - formal_value,
            })
        for state, count in enumerate(contributing):
            strict_transition_users.append({"transition_view": representation, "macrostate": state, "contributing_users": int(count)})
    strict_path = write_table(pd.DataFrame(strict_rows), table_root / "stage5_strict_user_equal_sensitivity")
    strict_users_path = write_table(pd.DataFrame(strict_transition_users), table_root / "stage5_strict_transition_contributing_users")

    grid_rows: List[Dict[str, Any]] = []
    for n_bins in (30, 40, 50):
        grid_bins = np.linspace(-1.0, 1.0, n_bins + 1)
        empirical_grid = direct_drift_field(
            current_m,
            current_psi,
            next_m - current_m,
            next_psi - current_psi,
            grid_bins,
            formal_weights,
            min_drift_count=30,
        )
        for representation, values in predictions.items():
            predicted_grid = direct_drift_field(
                values["m"],
                values["psi"],
                values["next_m"] - values["m"],
                values["next_psi"] - values["psi"],
                grid_bins,
                formal_weights,
                min_drift_count=30,
            )
            comparison = drift_comparison(empirical_grid, predicted_grid)
            grid_rows.append({
                "setting": f"{n_bins}x{n_bins}",
                "grid_bins_per_axis": n_bins,
                "interior_only": False,
                "representation": representation,
                **comparison,
            })
            if n_bins == 40:
                comparison_interior = drift_comparison(empirical_grid, predicted_grid, interior_cell_mask(40))
                grid_rows.append({
                    "setting": "40x40_interior_only",
                    "grid_bins_per_axis": 40,
                    "interior_only": True,
                    "representation": representation,
                    **comparison_interior,
                })
    grid_path = write_table(pd.DataFrame(grid_rows), table_root / "stage5_grid_sensitivity")

    permutation_rng = np.random.default_rng(args.seed + 5003)
    permutation_rows: List[Dict[str, Any]] = []
    singleton_rows = None
    for replicate in range(int(args.permutation_replicates)):
        current_index, next_index, singleton_count = within_user_permutation_indices(uid, permutation_rng)
        singleton_rows = singleton_count if singleton_rows is None else singleton_rows
        for representation, values in predictions.items():
            perm_m = values["m"][current_index]
            perm_psi = values["psi"][current_index]
            perm_next_m = values["next_m"][next_index]
            perm_next_psi = values["next_psi"][next_index]
            perm_field = direct_drift_field(
                perm_m,
                perm_psi,
                perm_next_m - perm_m,
                perm_next_psi - perm_psi,
                bins,
                formal_weights,
                min_drift_count=30,
            )
            perm_current_label = partition.labels(np.column_stack([perm_m, perm_psi]))
            perm_next_label = partition.labels(np.column_stack([perm_next_m, perm_next_psi]))
            pair = perm_current_label * 6 + perm_next_label
            valid = (perm_current_label >= 0) & (perm_next_label >= 0)
            counts = np.bincount(pair[valid], minlength=36).reshape(6, 6).astype(float)
            perm_transition = normalize_transition(counts)
            metrics = raw_point_metrics(
                empirical_field,
                perm_field,
                empirical_transition,
                perm_transition,
                current_m,
                current_psi,
                next_m,
                next_psi,
                perm_m,
                perm_psi,
                perm_next_m,
                perm_next_psi,
            )
            permutation_rows.extend(
                {
                    "replicate": replicate,
                    "representation": representation,
                    "metric": metric,
                    "value": value,
                }
                for metric, value in metrics.items()
            )
    permutation_frame = pd.DataFrame(permutation_rows)
    permutation_replicates_path = write_table(permutation_frame, table_root / "stage5_within_user_permutation_floor_replicates")
    floor_rows: List[Dict[str, Any]] = []
    for (representation, metric), group in permutation_frame.groupby(["representation", "metric"], sort=False):
        values = pd.to_numeric(group["value"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        observed = point_metrics[str(representation)][str(metric)]
        lower_is_better = str(metric) in {
            "one_step_rmse_M",
            "one_step_rmse_Psi",
            "learned_plane_transition_mean_row_tv",
        }
        null_median = float(np.median(values)) if values.size else np.nan
        floor_rows.append({
            "representation": representation,
            "metric": metric,
            "metric_direction": "lower_is_better" if lower_is_better else "higher_is_better",
            "observed": observed,
            "null_replicates_finite": int(values.size),
            "null_median": null_median,
            "null_5pct": float(np.quantile(values, 0.05)) if values.size else np.nan,
            "null_95pct": float(np.quantile(values, 0.95)) if values.size else np.nan,
            "observed_minus_null_median": observed - null_median if values.size else np.nan,
            "observed_improvement_over_null_median": (
                null_median - observed if lower_is_better else observed - null_median
            ) if values.size else np.nan,
        })
    floor_path = write_table(pd.DataFrame(floor_rows), table_root / "stage5_within_user_permutation_floor_summary")

    permutation_metric_maps: Dict[Tuple[int, str], Dict[str, float]] = {}
    for (replicate, representation), group in permutation_frame.groupby(["replicate", "representation"], sort=False):
        permutation_metric_maps[(int(replicate), str(representation))] = {
            str(row.metric): float(row.value) for row in group.itertuples(index=False)
        }
    score_rows: List[Dict[str, Any]] = []
    for contract in score_contracts():
        observed_scores: Dict[str, float] = {}
        null_scores: Dict[str, np.ndarray] = {}
        for representation in REPRESENTATIONS:
            observed_score, observed_domains = contract_score(point_metrics[representation], contract)
            observed_scores[representation] = observed_score
            score_rows.append({
                "contract": contract["contract"],
                "rmse_scale": contract["rmse_scale"],
                "included_domains": ",".join(contract["domains"]),
                "quantity": "observed_composite",
                "representation": representation,
                "value": observed_score,
            })
            if contract["contract"] == "S0_formal":
                for domain, value in observed_domains.items():
                    score_rows.append({
                        "contract": contract["contract"],
                        "rmse_scale": contract["rmse_scale"],
                        "included_domains": ",".join(contract["domains"]),
                        "quantity": domain,
                        "representation": representation,
                        "value": value,
                    })
            values = []
            for replicate in range(int(args.permutation_replicates)):
                score, _ = contract_score(permutation_metric_maps[(replicate, representation)], contract)
                values.append(score)
            null_scores[representation] = np.asarray(values, dtype=float)
            score_rows.extend([
                {
                    "contract": contract["contract"],
                    "rmse_scale": contract["rmse_scale"],
                    "included_domains": ",".join(contract["domains"]),
                    "quantity": "null_median_composite",
                    "representation": representation,
                    "value": float(np.nanmedian(null_scores[representation])),
                },
                {
                    "contract": contract["contract"],
                    "rmse_scale": contract["rmse_scale"],
                    "included_domains": ",".join(contract["domains"]),
                    "quantity": "null_5pct_composite",
                    "representation": representation,
                    "value": float(np.nanquantile(null_scores[representation], 0.05)),
                },
                {
                    "contract": contract["contract"],
                    "rmse_scale": contract["rmse_scale"],
                    "included_domains": ",".join(contract["domains"]),
                    "quantity": "null_95pct_composite",
                    "representation": representation,
                    "value": float(np.nanquantile(null_scores[representation], 0.95)),
                },
            ])
        full = observed_scores["full_hidden"]
        macro = observed_scores["macro_only"]
        full_floor = float(np.nanmedian(null_scores["full_hidden"]))
        macro_floor = float(np.nanmedian(null_scores["macro_only"]))
        score_rows.extend([
            {
                "contract": contract["contract"],
                "rmse_scale": contract["rmse_scale"],
                "included_domains": ",".join(contract["domains"]),
                "quantity": "macro_retention_vs_full",
                "representation": "macro_only_vs_full_hidden",
                "value": macro / full if np.isfinite(macro) and np.isfinite(full) and abs(full) > 1e-12 else np.nan,
            },
            {
                "contract": contract["contract"],
                "rmse_scale": contract["rmse_scale"],
                "included_domains": ",".join(contract["domains"]),
                "quantity": "macro_null_headroom_retention_vs_full",
                "representation": "macro_only_vs_full_hidden",
                "value": (macro - macro_floor) / (full - full_floor) if np.isfinite(full - full_floor) and abs(full - full_floor) > 1e-12 else np.nan,
            },
        ])
    score_path = write_table(pd.DataFrame(score_rows), table_root / "stage5_descriptive_score_sensitivity")

    source_paths = {representation: prediction_path(root, representation) for representation in REPRESENTATIONS}
    source_paths.update({
        "fixed_k6_metadata": args.stage1_root.resolve() / "dynamics" / "fixed_k6_mesostates" / "fixed_k6_model_metadata.json",
        "fixed_k6_centers": resolve_table(args.stage1_root.resolve() / "dynamics" / "fixed_k6_mesostates" / "fixed_k6_centers"),
    })
    manifest = {
        "script": Path(__file__).name,
        "output_root": str(output_root),
        "macro_root": str(root),
        "primary_state": ["M", "Psi"],
        "confirmation_split": "B_confirm",
        "bootstrap": {
            "method": "positive exponential learner multipliers",
            "replicates": total_replicates,
            "same_multiplier_for_full_macro_and_residual": True,
            "probes_refit": False,
            "model_retrained": False,
            "partition_refit": False,
            "new_p_values": False,
        },
        "statewise_correlation_boundary": (
            "Multiplier intervals for self-transition correlations reflect learner-composition variation in "
            "the six frozen state probabilities; the six states are not treated as independent samples."
        ),
        "strict_user_equal_contract": {
            "interval_metrics": "one total weight unit per evaluated user",
            "transitions": "per-user row-normalized rows averaged over users visiting each origin state",
            "field_note": "Stage-5 fields are already user-equal by construction and are audited rather than duplicated",
        },
        "grid_sensitivity": {
            "settings": ["30x30", "40x40", "50x50", "40x40_interior_only"],
            "drift_count_threshold": 30,
            "field_only": True,
            "core_reselected": False,
            "partition_refit": False,
        },
        "permutation_floor": {
            "method": "independent within-user permutations of predicted current and next rows",
            "replicates": int(args.permutation_replicates),
            "same_permutation_indices_across_representations": True,
            "empirical_targets_fixed": True,
            "descriptive_only": True,
            "p_values": False,
            "singleton_rows": int(singleton_rows or 0),
            "singleton_row_fraction": float((singleton_rows or 0) / max(len(uid), 1)),
        },
        "score_sensitivity": {
            "contracts": [contract["contract"] for contract in score_contracts()],
            "raw_metrics_primary": True,
            "formal_90p8_percent_not_used_as_primary_evidence": True,
        },
        "source_audit": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
            for name, path in source_paths.items()
        },
        "outputs": {
            "raw_point_estimates": str(point_path),
            "formal_point_equivalence_audit": str(equivalence_path),
            "bootstrap_replicates": str(bootstrap_replicates_path),
            "bootstrap_summary": str(bootstrap_summary_path),
            "strict_user_equal": str(strict_path),
            "strict_transition_users": str(strict_users_path),
            "grid_sensitivity": str(grid_path),
            "permutation_replicates": str(permutation_replicates_path),
            "permutation_floor": str(floor_path),
            "score_sensitivity": str(score_path),
        },
        "elapsed_seconds": float(time.time() - started),
    }
    save_json(manifest, metadata_root / "representation_robustness_manifest.json")
    print(f"[representation robustness] completed in {time.time() - started:.1f} seconds")


if __name__ == "__main__":
    main()
