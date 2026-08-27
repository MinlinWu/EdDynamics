#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from kinetic_robustness_common import (
    ClusterKineticAccumulator,
    EPS,
    benjamini_hochberg,
    frame_identifier_hash,
    import_module,
    load_json,
    normal_survival,
    normalize_transition,
    read_table,
    save_json,
    sha256_file,
    table_path,
    transition_counts_from_assignments,
    user_balanced_state_occupancy,
    write_table,
)

PRIMARY_COORDINATE_COLUMNS = (
    "M_response_prebalanced_pre",
    "activity_alignment_order_Psi_pre",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjacent-partition sensitivity and learner-cluster kinetic inference for the frozen EdNet-KT4 mesostates."
    )
    parser.add_argument("--stage1-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/stage1"))
    parser.add_argument("--stage1-script", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/stage1_kinetic_robustness/partition_cluster"),
    )
    parser.add_argument(
        "--partition-k-values",
        type=str,
        default="4,5,6,7,8",
        help="Bounded resolution-sensitivity values; K=6 must be included and remains the only formal partition.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-batch", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-users", type=int, default=0)
    return parser.parse_args()


def parse_partition_k_values(text: str) -> Tuple[int, ...]:
    values = sorted({int(token.strip()) for token in str(text).split(",") if token.strip()})
    if 6 not in values:
        raise ValueError("--partition-k-values must include the formal K=6 partition.")
    if any(value < 2 for value in values):
        raise ValueError("Every partition K must be at least 2.")
    return tuple(values)


def load_coordinate_panel(stage1_root: Path, split: str, stage1: Any) -> pd.DataFrame:
    columns = ["user_id", "bundle_step_index", *PRIMARY_COORDINATE_COLUMNS]
    frame = read_table(stage1_root / "dynamics" / f"student_dynamics_panel_core_{split}", columns=columns)
    frame["user_id"] = pd.to_numeric(frame["user_id"], errors="raise").astype(np.int64)
    frame["bundle_step_index"] = pd.to_numeric(frame["bundle_step_index"], errors="raise").astype(np.int64)
    for column in PRIMARY_COORDINATE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = stage1.downcast_frame(frame)
    return frame.reset_index(drop=True)


def sample_users(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum <= 0:
        return frame
    users = np.asarray(sorted(frame["user_id"].unique()), dtype=np.int64)
    if len(users) <= maximum:
        return frame
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(users, size=int(maximum), replace=False))
    return frame[frame["user_id"].isin(selected)].copy().reset_index(drop=True)


def ordered_partition(
    train_fit: pd.DataFrame,
    scaler: StandardScaler,
    k: int,
    stage1: Any,
    seed: int,
) -> Tuple[KMeans, np.ndarray, np.ndarray]:
    matrix = scaler.transform(train_fit[list(PRIMARY_COORDINATE_COLUMNS)].to_numpy(dtype=float))
    weights = np.asarray(stage1.user_balanced_weights(train_fit), dtype=float)
    model = KMeans(n_clusters=int(k), n_init=int(stage1.KMEANS_N_INIT), random_state=int(seed))
    try:
        model.fit(matrix, sample_weight=weights)
    except TypeError:
        model.fit(matrix)
    centers_original = scaler.inverse_transform(model.cluster_centers_)
    order = np.lexsort((centers_original[:, 1], centers_original[:, 0]))
    ordered_original = centers_original[order]
    ordered_standardized = model.cluster_centers_[order]
    return model, ordered_original, ordered_standardized


def assignment_frame(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    ordered_standardized_centers: np.ndarray,
) -> pd.DataFrame:
    output = frame[["user_id", "bundle_step_index"]].copy()
    output["macrostate"] = np.nan
    values = frame[list(PRIMARY_COORDINATE_COLUMNS)].to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1)
    if np.any(valid):
        standardized = scaler.transform(values[valid])
        distance = np.sum(
            (standardized[:, None, :] - ordered_standardized_centers[None, :, :]) ** 2,
            axis=2,
        )
        output.loc[valid, "macrostate"] = np.argmin(distance, axis=1).astype(float)
    output["macrostate_observed"] = output["macrostate"].notna()
    return output


def centers_table(centers: np.ndarray, standardized: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "macrostate": np.arange(len(centers), dtype=int),
            "center_M": centers[:, 0],
            "center_Psi": centers[:, 1],
            "center_M_standardized": standardized[:, 0],
            "center_Psi_standardized": standardized[:, 1],
        }
    )


def k6_reconstruction_audit(
    reconstructed_scaler: StandardScaler,
    reconstructed_centers: np.ndarray,
    reconstructed_standardized: np.ndarray,
    reconstructed_val: pd.DataFrame,
    archived_metadata: Mapping[str, Any],
    archived_centers: pd.DataFrame,
    archived_val: pd.DataFrame,
) -> pd.DataFrame:
    archived_centers = archived_centers.sort_values("macrostate", kind="mergesort").reset_index(drop=True)
    for frame in (archived_val, reconstructed_val):
        frame["user_id"] = pd.to_numeric(frame["user_id"], errors="raise").astype(np.int64)
        frame["bundle_step_index"] = pd.to_numeric(
            frame["bundle_step_index"], errors="raise"
        ).astype(np.int64)
    archived_val = archived_val.sort_values(["user_id", "bundle_step_index"], kind="mergesort").reset_index(drop=True)
    reconstructed_val = reconstructed_val.sort_values(["user_id", "bundle_step_index"], kind="mergesort").reset_index(drop=True)
    if not reconstructed_val[["user_id", "bundle_step_index"]].equals(
        archived_val[["user_id", "bundle_step_index"]]
    ):
        raise RuntimeError("Reconstructed and archived K=6 assignment row identifiers differ.")
    archived_labels = pd.to_numeric(archived_val["macrostate"], errors="coerce").to_numpy(dtype=float)
    reconstructed_labels = pd.to_numeric(reconstructed_val["macrostate"], errors="coerce").to_numpy(dtype=float)
    comparable = np.isfinite(archived_labels) | np.isfinite(reconstructed_labels)
    label_match = (
        (np.isnan(archived_labels) & np.isnan(reconstructed_labels))
        | (archived_labels == reconstructed_labels)
    )
    mismatch_fraction = float(np.mean(~label_match[comparable])) if np.any(comparable) else 0.0
    rows = [
        {
            "check": "scaler_mean",
            "maximum_absolute_difference": float(
                np.max(
                    np.abs(
                        np.asarray(reconstructed_scaler.mean_, dtype=float)
                        - np.asarray(archived_metadata["scaler_mean"], dtype=float)
                    )
                )
            ),
            "tolerance": 1e-12,
        },
        {
            "check": "scaler_scale",
            "maximum_absolute_difference": float(
                np.max(
                    np.abs(
                        np.asarray(reconstructed_scaler.scale_, dtype=float)
                        - np.asarray(archived_metadata["scaler_scale"], dtype=float)
                    )
                )
            ),
            "tolerance": 1e-12,
        },
        {
            "check": "ordered_centers_original",
            "maximum_absolute_difference": float(
                np.max(
                    np.abs(
                        reconstructed_centers
                        - archived_centers[["center_M", "center_Psi"]].to_numpy(dtype=float)
                    )
                )
            ),
            "tolerance": 1e-8,
        },
        {
            "check": "ordered_centers_standardized",
            "maximum_absolute_difference": float(
                np.max(
                    np.abs(
                        reconstructed_standardized
                        - archived_centers[
                            ["center_M_standardized", "center_Psi_standardized"]
                        ].to_numpy(dtype=float)
                    )
                )
            ),
            "tolerance": 1e-8,
        },
        {
            "check": "A_val_assignment_mismatch_fraction",
            "maximum_absolute_difference": mismatch_fraction,
            "tolerance": 1e-7,
        },
    ]
    audit = pd.DataFrame(rows)
    audit["passed"] = audit["maximum_absolute_difference"] <= audit["tolerance"]
    return audit


def noninferential_residence_summary(
    stage1: Any,
    transition: np.ndarray,
    runs: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for state in range(int(k)):
        state_runs = runs[
            pd.to_numeric(runs["macrostate"], errors="coerce") == state
        ] if not runs.empty else runs
        if state_runs.empty:
            rows.append({
                "macrostate": int(state),
                "n_runs": 0,
                "n_completed_exits": 0,
                "n_right_censored": 0,
                "right_censoring_fraction": np.nan,
                "self_transition": np.nan,
                "rmst_tau": np.nan,
                "obs_restricted_mean_residence": np.nan,
                "geo_null_restricted_mean": np.nan,
                "restricted_mean_residence_lift": np.nan,
                "reference_length": int(stage1.RESIDENCE_REFERENCE_LENGTH),
                "reference_at_risk": 0,
                "observed_tail_probability_at_reference": np.nan,
                "geometric_tail_probability_at_reference": np.nan,
                "tail_ratio_at_reference": np.nan,
                "inference_performed": False,
            })
            continue
        km = stage1.kaplan_meier_ccdf(
            state_runs, max_length=int(stage1.MAX_RESIDENCE_LENGTH)
        )
        self_transition = float(np.clip(transition[state, state], 1e-6, 1.0 - 1e-6))
        reliable = km[km["at_risk"] >= max(int(stage1.MIN_RESIDENCE_AT_RISK), 1)]
        tau = int(reliable["residence_length"].max()) if not reliable.empty else int(km["residence_length"].max())
        observed_rmst, _ = stage1.restricted_mean_from_km(km, tau)
        lengths = np.arange(1, tau + 1, dtype=int)
        geometric_rmst = float(np.sum(self_transition ** (lengths - 1)))
        lift = float(observed_rmst / geometric_rmst) if geometric_rmst > 0 else np.nan
        reference = max(1, int(stage1.RESIDENCE_REFERENCE_LENGTH))
        reference_row = km[km["residence_length"] == reference]
        if reference_row.empty:
            observed_tail = np.nan
            at_risk = 0
        else:
            observed_tail = float(reference_row["km_ccdf"].iloc[0])
            at_risk = int(reference_row["at_risk"].iloc[0])
        geometric_tail = float(self_transition ** (reference - 1))
        tail_ratio = (
            float(observed_tail / geometric_tail)
            if np.isfinite(observed_tail) and geometric_tail > 0
            else np.nan
        )
        observed = state_runs["event_observed"]
        if pd.api.types.is_bool_dtype(observed) or pd.api.types.is_numeric_dtype(observed):
            event_observed = pd.to_numeric(observed, errors="coerce").fillna(0).astype(bool)
        else:
            event_observed = observed.astype(str).str.strip().str.lower().isin(
                {"1", "true", "t", "yes", "y"}
            )
        n_runs = int(len(state_runs))
        n_completed = int(event_observed.sum())
        n_censored = int(n_runs - n_completed)
        rows.append({
            "macrostate": int(state),
            "n_runs": n_runs,
            "n_completed_exits": n_completed,
            "n_right_censored": n_censored,
            "right_censoring_fraction": float(n_censored / n_runs),
            "self_transition": self_transition,
            "rmst_tau": tau,
            "obs_restricted_mean_residence": observed_rmst,
            "geo_null_restricted_mean": geometric_rmst,
            "restricted_mean_residence_lift": lift,
            "reference_length": reference,
            "reference_at_risk": at_risk,
            "observed_tail_probability_at_reference": observed_tail,
            "geometric_tail_probability_at_reference": geometric_tail,
            "tail_ratio_at_reference": tail_ratio,
            "inference_performed": False,
        })
    return pd.DataFrame(rows)


def residence_point_table(
    stage1: Any,
    assignments: pd.DataFrame,
    k: int,
    inferential: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    counts = transition_counts_from_assignments(assignments, k)
    transition = normalize_transition(counts)
    runs = stage1.censored_residence_runs(assignments)
    summary = (
        stage1.residence_significance(transition, runs)
        if inferential
        else noninferential_residence_summary(stage1, transition, runs, k)
    )
    return counts, transition, runs, summary


def partition_statewise_table(
    stage1: Any,
    k: int,
    split: str,
    centers: pd.DataFrame,
    assignments: pd.DataFrame,
    transition: np.ndarray,
    runs: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    occupancy = user_balanced_state_occupancy(assignments, k)
    center_index = centers.set_index("macrostate")
    summary_index = summary.set_index("macrostate")
    rows: List[Dict[str, Any]] = []
    for state in range(int(k)):
        state_runs = runs[pd.to_numeric(runs["macrostate"], errors="coerce") == state]
        row = summary_index.loc[state]
        observed_tail = float(row.get("observed_tail_probability_at_reference", np.nan))
        geometric_tail = float(row.get("geometric_tail_probability_at_reference", np.nan))
        rows.append(
            {
                "k": int(k),
                "split": split,
                "macrostate": int(state),
                "center_M": float(center_index.loc[state, "center_M"]),
                "center_Psi": float(center_index.loc[state, "center_Psi"]),
                "user_balanced_occupancy": float(occupancy[state]),
                "self_transition": float(transition[state, state]),
                "diagonal_dominant": bool(
                    transition[state, state] >= np.max(transition[state]) - 1e-15
                ),
                "run_count": int(len(state_runs)),
                "right_censoring_fraction": float(row.get("right_censoring_fraction", np.nan)),
                "rmst_tau": int(float(row.get("rmst_tau", 0))),
                "restricted_mean_residence_lift": float(
                    row.get("restricted_mean_residence_lift", np.nan)
                ),
                "reference_at_risk": int(float(row.get("reference_at_risk", 0))),
                "fixed_reference_reliable": bool(
                    float(row.get("reference_at_risk", 0))
                    >= int(getattr(stage1, "MIN_RESIDENCE_AT_RISK", 20))
                ),
                "tail_excess_at_reference": (
                    observed_tail - geometric_tail
                    if np.isfinite(observed_tail) and np.isfinite(geometric_tail)
                    else np.nan
                ),
                "tail_ratio_at_reference": float(row.get("tail_ratio_at_reference", np.nan)),
                "greenwood_q_formal_K6_only": float(
                    row.get("tail_excess_qvalue_bh", np.nan)
                ),
                "formal_inference_performed": bool(
                    row.get("inference_performed", int(k) == 6)
                ),
            }
        )
    return pd.DataFrame(rows)


def partition_summary_row(
    k: int,
    statewise_train: pd.DataFrame,
    statewise_val: pd.DataFrame,
    counts_train: np.ndarray,
    counts_val: np.ndarray,
    transition_train: np.ndarray,
    transition_val: np.ndarray,
) -> Dict[str, Any]:
    row_tv = 0.5 * np.sum(np.abs(transition_train - transition_val), axis=1)
    train_lift = statewise_train["restricted_mean_residence_lift"].to_numpy(dtype=float)
    val_lift = statewise_val["restricted_mean_residence_lift"].to_numpy(dtype=float)
    valid_lift = np.isfinite(train_lift) & np.isfinite(val_lift) & (train_lift > 0) & (val_lift > 0)
    return {
        "k": int(k),
        "role": "formal" if int(k) == 6 else "bounded resolution sensitivity",
        "A_val_minimum_user_balanced_state_occupancy": float(
            statewise_val["user_balanced_occupancy"].min()
        ),
        "A_train_transition_count": int(np.sum(counts_train)),
        "A_val_transition_count": int(np.sum(counts_val)),
        "A_val_residence_run_count": int(statewise_val["run_count"].sum()),
        "A_val_right_censoring_fraction": float(
            np.average(
                statewise_val["right_censoring_fraction"].to_numpy(dtype=float),
                weights=np.maximum(statewise_val["run_count"].to_numpy(dtype=float), 0.0),
            )
        ),
        "A_val_diagonal_dominant_rows": int(statewise_val["diagonal_dominant"].sum()),
        "A_val_mean_self_transition": float(statewise_val["self_transition"].mean()),
        "A_val_min_self_transition": float(statewise_val["self_transition"].min()),
        "A_val_max_self_transition": float(statewise_val["self_transition"].max()),
        "A_val_states_with_rmst_lift_above_one": int(
            np.sum(statewise_val["restricted_mean_residence_lift"].to_numpy(dtype=float) > 1.0)
        ),
        "A_val_states_meeting_fixed10_at_risk_threshold": int(
            statewise_val["fixed_reference_reliable"].astype(bool).sum()
        ),
        "A_val_states_with_positive_fixed10_tail_excess": int(
            np.sum(
                statewise_val["fixed_reference_reliable"].astype(bool).to_numpy()
                & (statewise_val["tail_excess_at_reference"].to_numpy(dtype=float) > 0.0)
            )
        ),
        "A_train_A_val_transition_mean_row_tv": float(np.mean(row_tv)),
        "A_train_A_val_transition_max_row_tv": float(np.max(row_tv)),
        "A_train_A_val_rmst_lift_mean_abs_log_difference": (
            float(np.mean(np.abs(np.log(val_lift[valid_lift] / train_lift[valid_lift]))))
            if np.any(valid_lift)
            else np.nan
        ),
    }


def cluster_point_equivalence(
    point: Dict[str, np.ndarray],
    archived_transition: np.ndarray,
    archived_summary: pd.DataFrame,
) -> pd.DataFrame:
    summary_index = archived_summary.set_index("macrostate")
    rows: List[Dict[str, Any]] = [
        {
            "check": "transition_matrix",
            "maximum_absolute_difference": float(
                np.max(np.abs(point["transition"][:, :, 0] - archived_transition))
            ),
            "tolerance": 1e-12,
        }
    ]
    for state in range(archived_transition.shape[0]):
        archived_tail_excess = float(
            summary_index.loc[state, "observed_tail_probability_at_reference"]
            - summary_index.loc[state, "geometric_tail_probability_at_reference"]
        )
        comparisons = {
            "self_transition": (
                float(point["self_transition"][state, 0]),
                float(summary_index.loc[state, "self_transition"]),
            ),
            "rmst_lift": (
                float(point["rmst_lift"][state, 0]),
                float(summary_index.loc[state, "restricted_mean_residence_lift"]),
            ),
            "tail_excess": (
                float(point["tail_excess"][state, 0]),
                archived_tail_excess,
            ),
        }
        for metric, (recomputed, archived) in comparisons.items():
            rows.append(
                {
                    "check": f"state_{state}_{metric}",
                    "maximum_absolute_difference": abs(recomputed - archived),
                    "tolerance": 1e-10,
                }
            )
    audit = pd.DataFrame(rows)
    audit["passed"] = audit["maximum_absolute_difference"] <= audit["tolerance"]
    return audit


def run_cluster_bootstrap(
    accumulator: ClusterKineticAccumulator,
    archived_summary: pd.DataFrame,
    replicates: int,
    batch_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    point = accumulator.evaluate(np.ones(accumulator.n_users, dtype=float))
    rng = np.random.default_rng(int(seed))
    probability = np.full(accumulator.n_users, 1.0 / accumulator.n_users, dtype=float)
    replicate_rows: List[Dict[str, Any]] = []
    completed = 0
    while completed < int(replicates):
        current_batch = min(int(batch_size), int(replicates) - completed)
        multiplicities = rng.multinomial(
            accumulator.n_users,
            probability,
            size=current_batch,
        ).T.astype(float)
        result = accumulator.evaluate(multiplicities)
        for local in range(current_batch):
            replicate = completed + local
            for state in range(accumulator.k):
                replicate_rows.append(
                    {
                        "replicate": int(replicate),
                        "macrostate": int(state),
                        "self_transition": float(result["self_transition"][state, local]),
                        "diagonal_dominant": (
                            bool(result["diagonal_dominant"][state, local] > 0.5)
                            if np.isfinite(result["diagonal_dominant"][state, local])
                            else np.nan
                        ),
                        "restricted_mean_residence_lift": float(result["rmst_lift"][state, local]),
                        "tail_excess_at_reference": float(result["tail_excess"][state, local]),
                    }
                )
        completed += current_batch
        print(f"[cluster bootstrap] {completed}/{replicates}", flush=True)

    replicates_frame = pd.DataFrame(replicate_rows)
    summary_index = archived_summary.set_index("macrostate")
    summary_rows: List[Dict[str, Any]] = []
    for state in range(accumulator.k):
        subset = replicates_frame[replicates_frame["macrostate"] == state]
        pii_values = subset["self_transition"].to_numpy(dtype=float)
        lift_values = subset["restricted_mean_residence_lift"].to_numpy(dtype=float)
        tail_values = subset["tail_excess_at_reference"].to_numpy(dtype=float)
        point_tail = float(
            summary_index.loc[state, "observed_tail_probability_at_reference"]
            - summary_index.loc[state, "geometric_tail_probability_at_reference"]
        )
        tail_se = float(np.std(tail_values[np.isfinite(tail_values)], ddof=1))
        z_score = point_tail / tail_se if tail_se > 0 else np.nan
        pvalue = normal_survival(z_score)
        summary_rows.append(
            {
                "macrostate": int(state),
                "self_transition_point": float(summary_index.loc[state, "self_transition"]),
                "self_transition_ci_2p5": float(np.nanquantile(pii_values, 0.025)),
                "self_transition_ci_97p5": float(np.nanquantile(pii_values, 0.975)),
                "diagonal_dominance_bootstrap_probability": float(
                    pd.to_numeric(
                        subset["diagonal_dominant"], errors="coerce"
                    ).mean()
                ),
                "restricted_mean_residence_lift_point": float(
                    summary_index.loc[state, "restricted_mean_residence_lift"]
                ),
                "restricted_mean_residence_lift_ci_2p5": float(
                    np.nanquantile(lift_values, 0.025)
                ),
                "restricted_mean_residence_lift_ci_97p5": float(
                    np.nanquantile(lift_values, 0.975)
                ),
                "rmst_lift_lower_bound_above_one": bool(
                    np.nanquantile(lift_values, 0.025) > 1.0
                ),
                "tail_excess_point": point_tail,
                "tail_excess_ci_2p5": float(np.nanquantile(tail_values, 0.025)),
                "tail_excess_ci_97p5": float(np.nanquantile(tail_values, 0.975)),
                "tail_excess_cluster_bootstrap_se": tail_se,
                "tail_excess_cluster_z": z_score,
                "tail_excess_cluster_one_sided_p": pvalue,
                "greenwood_one_sided_p_formal": float(
                    summary_index.loc[state, "tail_excess_pvalue_greenwood"]
                ),
                "greenwood_bh_q_formal": float(
                    summary_index.loc[state, "tail_excess_qvalue_bh"]
                ),
                "finite_self_transition_bootstrap_replicates": int(
                    np.isfinite(pii_values).sum()
                ),
                "finite_rmst_bootstrap_replicates": int(
                    np.isfinite(lift_values).sum()
                ),
                "finite_tail_bootstrap_replicates": int(
                    np.isfinite(tail_values).sum()
                ),
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame["tail_excess_cluster_bh_q"] = benjamini_hochberg(
        summary_frame["tail_excess_cluster_one_sided_p"].to_numpy(dtype=float)
    )
    summary_frame["tail_excess_cluster_bh_positive"] = (
        np.isfinite(summary_frame["tail_excess_cluster_bh_q"])
        & (summary_frame["tail_excess_cluster_bh_q"] < 0.05)
        & (summary_frame["tail_excess_point"] > 0)
    )
    minimum_finite = int(math.ceil(0.95 * int(replicates)))
    finite_columns = [
        "finite_self_transition_bootstrap_replicates",
        "finite_rmst_bootstrap_replicates",
        "finite_tail_bootstrap_replicates",
    ]
    if any(
        bool((pd.to_numeric(summary_frame[column], errors="coerce") < minimum_finite).any())
        for column in finite_columns
    ):
        raise RuntimeError(
            "At least one formal state has fewer than 95% finite learner-cluster bootstrap replicates."
        )
    return replicates_frame, summary_frame, point


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 200:
        raise ValueError("Use at least 200 learner-cluster bootstrap replicates; the formal default is 1000.")
    if args.bootstrap_batch < 1:
        raise ValueError("--bootstrap-batch must be positive.")
    partition_k_values = parse_partition_k_values(args.partition_k_values)
    started = time.time()
    stage1_root = args.stage1_root.resolve()
    output_root = args.output_root.resolve()
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    table_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    stage1 = import_module(args.stage1_script, "kinetic_partition_formal_stage1")
    required = (
        "downcast_frame",
        "user_balanced_weights",
        "KMEANS_N_INIT",
        "KMEANS_FIT_MAX_ROWS",
        "RANDOM_STATE",
        "censored_residence_runs",
        "residence_significance",
        "kaplan_meier_ccdf",
        "restricted_mean_from_km",
        "MAX_RESIDENCE_LENGTH",
        "MIN_RESIDENCE_AT_RISK",
        "RESIDENCE_REFERENCE_LENGTH",
    )
    missing = [name for name in required if not hasattr(stage1, name)]
    if missing:
        raise RuntimeError(f"Formal Stage-1 script is missing required objects: {missing}")

    train = sample_users(load_coordinate_panel(stage1_root, "A_train", stage1), args.max_users, args.seed + 11)
    val = sample_users(load_coordinate_panel(stage1_root, "A_val", stage1), args.max_users, args.seed + 23)
    if args.max_users > 0:
        raise RuntimeError("Formal partition and cluster-bootstrap outputs require --max-users=0; positive values are smoke-test only.")

    mesostate_root = stage1_root / "dynamics" / "fixed_k6_mesostates"
    archived_metadata_path = mesostate_root / "fixed_k6_model_metadata.json"
    archived_metadata = load_json(archived_metadata_path)
    archived_centers = read_table(mesostate_root / "fixed_k6_centers")
    archived_val_assignments = read_table(mesostate_root / "A_val_fixed_k6_assignments")

    train_valid = train.dropna(subset=list(PRIMARY_COORDINATE_COLUMNS)).copy()
    values = train_valid[list(PRIMARY_COORDINATE_COLUMNS)].to_numpy(dtype=float)
    train_valid = train_valid[np.isfinite(values).all(axis=1)].copy()
    if int(stage1.KMEANS_FIT_MAX_ROWS) > 0 and len(train_valid) > int(stage1.KMEANS_FIT_MAX_ROWS):
        train_fit = train_valid.sample(
            n=int(stage1.KMEANS_FIT_MAX_ROWS),
            random_state=int(stage1.RANDOM_STATE),
            weights=np.asarray(stage1.user_balanced_weights(train_valid), dtype=float),
            replace=False,
        ).copy()
    else:
        train_fit = train_valid.copy()
    sample_hash = frame_identifier_hash(train_fit, ["user_id", "bundle_step_index"])

    reconstructed_scaler = StandardScaler().fit(
        train_fit[list(PRIMARY_COORDINATE_COLUMNS)].to_numpy(dtype=float)
    )
    _, reconstructed_centers, reconstructed_standardized = ordered_partition(
        train_fit,
        reconstructed_scaler,
        6,
        stage1,
        int(stage1.RANDOM_STATE),
    )
    reconstructed_val = assignment_frame(val, reconstructed_scaler, reconstructed_standardized)
    k6_audit = k6_reconstruction_audit(
        reconstructed_scaler,
        reconstructed_centers,
        reconstructed_standardized,
        reconstructed_val,
        archived_metadata,
        archived_centers,
        archived_val_assignments,
    )
    k6_audit_path = write_table(k6_audit, table_root / "formal_k6_reconstruction_audit")
    if not bool(k6_audit["passed"].all()):
        failed = k6_audit.loc[~k6_audit["passed"], "check"].tolist()
        raise RuntimeError(f"The formal K=6 fit sample could not be reproduced: {failed}")

    archived_scaler = StandardScaler()
    archived_scaler.mean_ = np.asarray(archived_metadata["scaler_mean"], dtype=float)
    archived_scaler.scale_ = np.asarray(archived_metadata["scaler_scale"], dtype=float)
    archived_scaler.var_ = archived_scaler.scale_ ** 2
    archived_scaler.n_features_in_ = 2
    archived_scaler.n_samples_seen_ = int(archived_metadata["fit_rows"])

    statewise_tables: List[pd.DataFrame] = []
    partition_rows: List[Dict[str, Any]] = []
    output_paths: Dict[str, Any] = {}
    for k in partition_k_values:
        if k == 6:
            centers = archived_centers.sort_values("macrostate", kind="mergesort").reset_index(drop=True)
            assignments = {
                split: read_table(mesostate_root / f"{split}_fixed_k6_assignments")
                for split in ("A_train", "A_val")
            }
            counts = {
                split: read_table(mesostate_root / f"{split}_fixed_k6_transition_counts").to_numpy(dtype=float)
                for split in ("A_train", "A_val")
            }
            transitions = {
                split: read_table(mesostate_root / f"{split}_fixed_k6_transition_matrix").to_numpy(dtype=float)
                for split in ("A_train", "A_val")
            }
            runs = {
                split: read_table(mesostate_root / f"{split}_fixed_k6_residence_runs")
                for split in ("A_train", "A_val")
            }
            summaries = {
                split: read_table(mesostate_root / f"{split}_fixed_k6_residence_summary")
                for split in ("A_train", "A_val")
            }
        else:
            _, original_centers, standardized_centers = ordered_partition(
                train_fit,
                archived_scaler,
                k,
                stage1,
                int(stage1.RANDOM_STATE),
            )
            centers = centers_table(original_centers, standardized_centers)
            assignments = {
                "A_train": assignment_frame(train, archived_scaler, standardized_centers),
                "A_val": assignment_frame(val, archived_scaler, standardized_centers),
            }
            counts = {}
            transitions = {}
            runs = {}
            summaries = {}
            for split in ("A_train", "A_val"):
                counts[split], transitions[split], runs[split], summaries[split] = residence_point_table(
                    stage1, assignments[split], k, inferential=False
                )
                output_paths[f"K{k}_{split}_transition_counts"] = str(
                    write_table(
                        pd.DataFrame(counts[split]),
                        table_root / f"K{k}_{split}_transition_counts",
                    )
                )
                output_paths[f"K{k}_{split}_transition_matrix"] = str(
                    write_table(
                        pd.DataFrame(transitions[split]),
                        table_root / f"K{k}_{split}_transition_matrix",
                    )
                )
                output_paths[f"K{k}_{split}_residence_summary"] = str(
                    write_table(
                        summaries[split],
                        table_root / f"K{k}_{split}_residence_summary",
                    )
                )
            output_paths[f"K{k}_centers"] = str(
                write_table(centers, table_root / f"K{k}_centers")
            )

        statewise_by_split = {}
        for split in ("A_train", "A_val"):
            statewise = partition_statewise_table(
                stage1,
                k,
                split,
                centers,
                assignments[split],
                transitions[split],
                runs[split],
                summaries[split],
            )
            statewise_tables.append(statewise)
            statewise_by_split[split] = statewise
        partition_rows.append(
            partition_summary_row(
                k,
                statewise_by_split["A_train"],
                statewise_by_split["A_val"],
                counts["A_train"],
                counts["A_val"],
                transitions["A_train"],
                transitions["A_val"],
            )
        )

    partition_statewise = pd.concat(statewise_tables, ignore_index=True)
    partition_summary = pd.DataFrame(partition_rows)
    partition_statewise_path = write_table(
        partition_statewise, table_root / "partition_statewise_kinetics"
    )
    partition_summary_path = write_table(
        partition_summary, table_root / "partition_robustness_summary"
    )

    formal_assignments = read_table(mesostate_root / "A_val_fixed_k6_assignments")
    formal_runs = read_table(mesostate_root / "A_val_fixed_k6_residence_runs")
    formal_summary = read_table(mesostate_root / "A_val_fixed_k6_residence_summary")
    formal_transition = read_table(mesostate_root / "A_val_fixed_k6_transition_matrix").to_numpy(dtype=float)
    accumulator = ClusterKineticAccumulator(
        formal_assignments,
        formal_runs,
        formal_summary,
        k=6,
        maximum_length=int(stage1.MAX_RESIDENCE_LENGTH),
    )
    cluster_point_preflight = accumulator.evaluate(np.ones(accumulator.n_users, dtype=float))
    equivalence = cluster_point_equivalence(
        cluster_point_preflight, formal_transition, formal_summary
    )
    equivalence_path = write_table(
        equivalence, table_root / "cluster_bootstrap_formal_point_equivalence_audit"
    )
    if not bool(equivalence["passed"].all()):
        failed = equivalence.loc[~equivalence["passed"], "check"].tolist()
        raise RuntimeError(f"Cluster-bootstrap unit weights do not reproduce formal K=6 kinetics: {failed}")
    cluster_replicates, cluster_summary, cluster_point = run_cluster_bootstrap(
        accumulator,
        formal_summary,
        int(args.bootstrap_replicates),
        int(args.bootstrap_batch),
        int(args.seed) + 9001,
    )
    cluster_replicates_path = write_table(
        cluster_replicates, table_root / "learner_cluster_bootstrap_replicates"
    )
    cluster_summary_path = write_table(
        cluster_summary, table_root / "learner_cluster_bootstrap_statewise_summary"
    )

    source_paths = {
        "stage1_script": args.stage1_script.resolve(),
        "formal_k6_metadata": archived_metadata_path,
        "formal_k6_centers": table_path(mesostate_root / "fixed_k6_centers"),
        "formal_A_val_assignments": table_path(mesostate_root / "A_val_fixed_k6_assignments"),
        "formal_A_val_transition": table_path(mesostate_root / "A_val_fixed_k6_transition_matrix"),
        "formal_A_val_residence_runs": table_path(mesostate_root / "A_val_fixed_k6_residence_runs"),
        "formal_A_val_residence_summary": table_path(mesostate_root / "A_val_fixed_k6_residence_summary"),
    }
    manifest = {
        "script": Path(__file__).name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": float(time.time() - started),
        "stage1_root": str(stage1_root),
        "stage1_script": str(args.stage1_script.resolve()),
        "stage1_script_sha256": sha256_file(args.stage1_script.resolve()),
        "primary_coordinates": ["M", "Psi"],
        "formal_partition": {
            "k": 6,
            "fit_split": "A_train",
            "fit_sample_rows": int(len(train_fit)),
            "fit_sample_identifier_sha256": sample_hash,
            "scaler_and_sample_reconstructed_before_adjacent_K_fits": True,
            "formal_k6_equivalence_required": True,
        },
        "partition_sensitivity": {
            "candidate_values": list(partition_k_values),
            "role": "bounded partition-resolution sensitivity only; no K selection",
            "shared_fit_sample": True,
            "shared_formal_scaler": True,
            "shared_n_init": int(stage1.KMEANS_N_INIT),
            "shared_random_state": int(stage1.RANDOM_STATE),
            "validation_used_for_fit_or_selection": False,
            "B_confirm_read": False,
            "downstream_models_read": False,
            "statewise_cross_K_matching_attempted": False,
            "new_statewise_pvalues_for_K5_or_K7": False,
        },
        "learner_cluster_bootstrap": {
            "split": "A_val",
            "partition": "formal frozen K=6",
            "replicates": int(args.bootstrap_replicates),
            "resampling_unit": "learner",
            "method": "multinomial nonparametric cluster bootstrap",
            "same_learner_multiplicity_for_transitions_and_all_residence_runs": True,
            "Pii_and_Kaplan_Meier_recomputed_jointly": True,
            "state_specific_rmst_horizons_frozen_from_formal_analysis": True,
            "fixed_reference_length": int(stage1.RESIDENCE_REFERENCE_LENGTH),
            "tail_inference": "one-sided normal approximation using learner-cluster bootstrap standard error, followed by BH across six formal states",
            "relationship_to_existing_multiplier_sensitivity": "replaces the kinetic residence/Pii inferential summary; does not rerun or duplicate the positive-exponential field/model sensitivity analyses",
        },
        "duplication_guardrails": {
            "construction_matched_field_null_rerun": False,
            "coordinate_or_grid_sensitivity_rerun": False,
            "strict_user_equal_transition_rerun": False,
            "positive_exponential_learner_multiplier_rerun": False,
            "mechanism_or_event_ssl_evaluation": False,
            "alternative_partition_row_level_assignments_or_runs_written": False,
            "new_figure_generated": False,
        },
        "quality_gates": {
            "formal_k6_fit_reproduced": bool(k6_audit["passed"].all()),
            "cluster_unit_weight_point_reproduced": bool(equivalence["passed"].all()),
            "bootstrap_replicates_at_least_1000": bool(args.bootstrap_replicates >= 1000),
            "cluster_bootstrap_finite_support_at_least_95_percent": bool(
                (
                    cluster_summary[[
                        "finite_self_transition_bootstrap_replicates",
                        "finite_rmst_bootstrap_replicates",
                        "finite_tail_bootstrap_replicates",
                    ]]
                    .apply(pd.to_numeric, errors="coerce")
                    .min()
                    .min()
                    >= math.ceil(0.95 * int(args.bootstrap_replicates))
                )
            ),
            "full_split_used": bool(args.max_users == 0),
        },
        "source_files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "outputs": {
            "formal_k6_reconstruction_audit": str(k6_audit_path),
            "partition_statewise_kinetics": str(partition_statewise_path),
            "partition_robustness_summary": str(partition_summary_path),
            "cluster_formal_point_equivalence_audit": str(equivalence_path),
            "cluster_bootstrap_replicates": str(cluster_replicates_path),
            "cluster_bootstrap_statewise_summary": str(cluster_summary_path),
            **output_paths,
        },
        "interpretation_boundary": (
            "The non-K=6 partitions are bounded resolution-sensitivity analyses, not candidate selection. "
            "The learner-cluster intervals and tail tests apply only to the formal K=6 validation kinetics."
        ),
    }
    manifest_path = metadata_root / "partition_cluster_kinetic_manifest.json"
    save_json(manifest, manifest_path)
    print(f"[partition/cluster] completed: {output_root}", flush=True)


if __name__ == "__main__":
    main()
