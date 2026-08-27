#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from extract_kinetic_robustness_statistics import recompute_statewise_maxT
from kinetic_robustness_common import (
    load_json,
    read_table,
    save_json,
    sha256_file,
    table_path,
    write_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect the recursive kinetic null, adjacent-partition sensitivity and learner-cluster inference outputs."
    )
    parser.add_argument(
        "--partition-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/stage1_kinetic_robustness/partition_cluster"),
    )
    parser.add_argument(
        "--recursive-null-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/stage1_kinetic_robustness/recursive_null"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/stage1_kinetic_robustness/summary"),
    )
    return parser.parse_args()


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(number):
        return "–"
    if number != 0 and abs(number) < 1e-3:
        return f"{number:.2e}"
    return f"{number:.{digits}f}"



def bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)) and np.isfinite(value):
        return bool(int(value))
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def bool_series(series: pd.Series) -> pd.Series:
    return series.map(bool_value)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row)
            + " |"
        )
    return "\n".join(lines)


def required_true(mapping: Mapping[str, Any], keys: List[str], label: str) -> None:
    failed = [key for key in keys if mapping.get(key) is not True]
    if failed:
        raise RuntimeError(f"{label} quality gates failed: {failed}")


def partition_display(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in frame.sort_values("k").itertuples(index=False):
        rows.append(
            {
                "K": int(row.k),
                "Role": str(row.role),
                "Min. occupancy": fmt(row.A_val_minimum_user_balanced_state_occupancy),
                "Diagonal rows": f"{int(row.A_val_diagonal_dominant_rows)}/{int(row.k)}",
                "Mean Pii": fmt(row.A_val_mean_self_transition),
                "Pii range": f"{fmt(row.A_val_min_self_transition)}–{fmt(row.A_val_max_self_transition)}",
                "RMST lift > 1": f"{int(row.A_val_states_with_rmst_lift_above_one)}/{int(row.k)}",
                "10-step supported": f"{int(row.A_val_states_meeting_fixed10_at_risk_threshold)}/{int(row.k)}",
                "Positive D10": f"{int(row.A_val_states_with_positive_fixed10_tail_excess)}/{int(row.k)}",
                "Train–val row TV": fmt(row.A_train_A_val_transition_mean_row_tv),
                "Train–val |log lift|": fmt(
                    row.A_train_A_val_rmst_lift_mean_abs_log_difference
                ),
            }
        )
    return pd.DataFrame(rows)


def recursive_display(
    aggregate: pd.DataFrame,
    familywise: pd.DataFrame,
    statewise: pd.DataFrame,
) -> pd.DataFrame:
    labels = {
        "aggregate_mean_log_rmst_lift_fixed10": "Mean log fixed-10 RMST lift",
        "diagonal_margin": "Mean diagonal margin",
        "mean_self_transition": "Mean Pii",
        "diagonal_dominant_rows": "Diagonal-dominant rows",
    }
    rows: List[Dict[str, Any]] = []
    for row in aggregate.itertuples(index=False):
        rows.append(
            {
                "Panel": "Aggregate recursive surrogate",
                "Endpoint / state": labels.get(str(row.metric), str(row.metric)),
                "Status": "aggregate test" if bool_value(row.primary_endpoint) else "descriptive",
                "Observed": fmt(row.observed),
                "Null reference": (
                    f"median {fmt(row.null_median)}; "
                    f"95% [{fmt(row.null_2p5)}, {fmt(row.null_97p5)}]"
                ),
                "Inference": (
                    f"pMC={fmt(row.monte_carlo_p)}"
                    if bool_value(row.primary_endpoint)
                    else "descriptive"
                ),
            }
        )
    if len(familywise) != 1:
        raise RuntimeError("The recursive statewise family-wise summary must contain one row.")
    row = familywise.iloc[0]
    states = "" if pd.isna(row["fwer_positive_states"]) else str(row["fwer_positive_states"])
    rows.append(
        {
            "Panel": "Statewise construction-aware inference",
            "Endpoint / state": "Fixed-10 tail-excess studentized maxT",
            "Status": str(row["analysis_status"]),
            "Observed": f"maxT={fmt(row['observed_maxT'])}",
            "Null reference": (
                f"median {fmt(row['null_median'])}; "
                f"95% [{fmt(row['null_2p5'])}, {fmt(row['null_97p5'])}]"
            ),
            "Inference": (
                f"global pMC={fmt(row['monte_carlo_p'])}; "
                f"FWER-positive={int(row['fwer_positive_state_count'])}/6 "
                f"({states or 'none'})"
            ),
        }
    )
    for row in statewise.sort_values("macrostate").itertuples(index=False):
        rows.append(
            {
                "Panel": "Statewise construction-aware inference",
                "Endpoint / state": f"S{int(row.macrostate)}",
                "Status": "maxT family-wise tested",
                "Observed": (
                    f"Pii={fmt(row.observed_self_transition)}; "
                    f"L10={fmt(row.observed_rmst_lift_fixed10)}; "
                    f"D10={fmt(row.observed_tail_excess_fixed10)}"
                ),
                "Null reference": (
                    f"D10 mean={fmt(row.tail_excess_fixed10_null_mean)}; "
                    f"95% [{fmt(row.null_tail_excess_fixed10_2p5)}, "
                    f"{fmt(row.null_tail_excess_fixed10_97p5)}]"
                ),
                "Inference": (
                    f"z={fmt(row.tail_excess_fixed10_standardized_excess)}; "
                    f"raw pMC={fmt(row.tail_excess_fixed10_raw_monte_carlo_p)}; "
                    f"maxT pFWER={fmt(row.tail_excess_fixed10_maxT_fwer_p)}; "
                    f"supported={'yes' if bool_value(row.tail_excess_fixed10_maxT_fwer_positive) else 'no'}"
                ),
            }
        )
    return pd.DataFrame(rows)


def cluster_display(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in frame.sort_values("macrostate").itertuples(index=False):
        rows.append(
            {
                "State": f"S{int(row.macrostate)}",
                "Pii [95% CI]": (
                    f"{fmt(row.self_transition_point)} "
                    f"[{fmt(row.self_transition_ci_2p5)}, {fmt(row.self_transition_ci_97p5)}]"
                ),
                "Pr(diagonal)": fmt(row.diagonal_dominance_bootstrap_probability),
                "RMST lift [95% CI]": (
                    f"{fmt(row.restricted_mean_residence_lift_point)} "
                    f"[{fmt(row.restricted_mean_residence_lift_ci_2p5)}, "
                    f"{fmt(row.restricted_mean_residence_lift_ci_97p5)}]"
                ),
                "D10 [95% CI]": (
                    f"{fmt(row.tail_excess_point)} "
                    f"[{fmt(row.tail_excess_ci_2p5)}, {fmt(row.tail_excess_ci_97p5)}]"
                ),
                "Cluster p": fmt(row.tail_excess_cluster_one_sided_p),
                "BH q": fmt(row.tail_excess_cluster_bh_q),
                "BH positive": "yes" if bool_value(row.tail_excess_cluster_bh_positive) else "no",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    partition_root = args.partition_root.resolve()
    null_root = args.recursive_null_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    partition_manifest_path = partition_root / "metadata" / "partition_cluster_kinetic_manifest.json"
    null_manifest_path = null_root / "metadata" / "recursive_construction_inertia_null_manifest.json"
    partition_manifest = load_json(partition_manifest_path)
    null_manifest = load_json(null_manifest_path)
    partition_values = [
        int(value)
        for value in partition_manifest.get("partition_sensitivity", {}).get(
            "candidate_values", [6]
        )
    ]
    sensitivity_values = [value for value in partition_values if value != 6]
    sensitivity_label = ", ".join(f"K={value}" for value in sensitivity_values)
    required_true(
        partition_manifest.get("quality_gates", {}),
        [
            "formal_k6_fit_reproduced",
            "cluster_unit_weight_point_reproduced",
            "bootstrap_replicates_at_least_1000",
            "cluster_bootstrap_finite_support_at_least_95_percent",
            "full_split_used",
        ],
        "partition/cluster",
    )
    required_true(
        null_manifest.get("quality_gates", {}),
        [
            "full_split_used",
            "matching_coverage_exactly_reproduced",
            "observed_innovation_recursion_reproduces_formal_assignments_and_kinetics",
            "all_recursive_edges_are_adjacent",
            "target_coordinates_reproduced_within_2e_6",
            "fixed10_all_states_meet_at_risk_threshold_in_observed_and_null",
            "no_invalid_denominators",
            "no_non_numerical_clipping",
            "first_permutation_fixed_point_free",
            "replicates_at_least_100",
        ],
        "recursive null",
    )

    partition = read_table(partition_root / "tables" / "partition_robustness_summary")
    cluster = read_table(
        partition_root / "tables" / "learner_cluster_bootstrap_statewise_summary"
    )
    recursive = read_table(
        null_root / "tables" / "recursive_construction_inertia_null_summary"
    )
    recursive_statewise = read_table(
        null_root / "tables" / "recursive_construction_inertia_statewise_summary"
    )
    recursive_statewise_replicates = read_table(
        null_root
        / "tables"
        / "recursive_construction_inertia_null_statewise_replicates"
    )
    computed_familywise, computed_statewise = recompute_statewise_maxT(
        recursive_statewise, recursive_statewise_replicates
    )
    inference_columns = [
        column for column in computed_statewise.columns if column != "macrostate"
    ]
    if any(column not in recursive_statewise.columns for column in inference_columns):
        recursive_statewise = recursive_statewise.drop(
            columns=[
                column for column in inference_columns if column in recursive_statewise.columns
            ],
            errors="ignore",
        ).merge(
            computed_statewise, on="macrostate", how="left", validate="one_to_one"
        )
    familywise_base = (
        null_root
        / "tables"
        / "recursive_construction_inertia_statewise_familywise_summary"
    )
    try:
        recursive_familywise = read_table(familywise_base)
        familywise_source = table_path(familywise_base)
    except FileNotFoundError:
        recursive_familywise = pd.DataFrame([computed_familywise])
        familywise_source = None
    table1a = recursive_display(
        recursive, recursive_familywise, recursive_statewise
    )
    table1b = partition_display(partition)
    table2 = cluster_display(cluster)

    table1a_path = write_table(table1a, output_root / "kinetic_table1a_recursive_null")
    table1b_path = write_table(table1b, output_root / "kinetic_table1b_partition_sensitivity")
    table2_path = write_table(table2, output_root / "kinetic_table2_cluster_inference")

    primary = recursive[bool_series(recursive["primary_endpoint"])]
    if len(primary) != 1 or len(recursive_familywise) != 1:
        raise RuntimeError("The recursive null must declare one aggregate and one family-wise endpoint.")
    primary_row = primary.iloc[0]
    familywise_row = recursive_familywise.iloc[0]
    formal_k6 = partition[pd.to_numeric(partition["k"], errors="coerce") == 6]
    if len(formal_k6) != 1:
        raise RuntimeError("Partition summary does not contain exactly one formal K=6 row.")
    cluster_positive = int(bool_series(cluster["tail_excess_cluster_bh_positive"]).sum())
    rmst_robust = int(bool_series(cluster["rmst_lift_lower_bound_above_one"]).sum())
    recursive_support = recursive_statewise[
        ["macrostate", "tail_excess_fixed10_maxT_fwer_positive"]
    ].copy()
    recursive_support["macrostate"] = pd.to_numeric(
        recursive_support["macrostate"], errors="raise"
    ).astype(int)
    recursive_support["recursive_supported"] = bool_series(
        recursive_support["tail_excess_fixed10_maxT_fwer_positive"]
    )
    cluster_support = cluster[
        [
            "macrostate",
            "tail_excess_cluster_bh_positive",
            "rmst_lift_lower_bound_above_one",
        ]
    ].copy()
    cluster_support["macrostate"] = pd.to_numeric(
        cluster_support["macrostate"], errors="raise"
    ).astype(int)
    cluster_support["cluster_supported"] = bool_series(
        cluster_support["tail_excess_cluster_bh_positive"]
    )
    cluster_support["rmst_supported"] = bool_series(
        cluster_support["rmst_lift_lower_bound_above_one"]
    )
    joint = recursive_support.merge(
        cluster_support, on="macrostate", how="inner", validate="one_to_one"
    )
    joint["joint_supported"] = (
        joint["recursive_supported"]
        & joint["cluster_supported"]
        & joint["rmst_supported"]
    )
    recursive_states = [
        int(value)
        for value in joint.loc[joint["recursive_supported"], "macrostate"].tolist()
    ]
    joint_states = [
        int(value)
        for value in joint.loc[joint["joint_supported"], "macrostate"].tolist()
    ]
    all_partition_rmst = bool(
        all(
            int(row.A_val_states_with_rmst_lift_above_one) == int(row.k)
            for row in partition.itertuples(index=False)
        )
    )
    global_statewise_supported = bool(
        float(familywise_row["monte_carlo_p"]) < 0.05
    )
    if global_statewise_supported and len(joint_states) >= 2 and rmst_robust == 6 and all_partition_rmst:
        conclusion_category = "operational_metastable_like_supported"
        conclusion = (
            "The frozen coarse-graining supports operationally defined metastable-like organisation: "
            "multiple states retain positive fixed-10 tail excess beyond the construction-aware "
            "recursive surrogate after family-wise control, while reliable-horizon RMST excess is "
            "learner-cluster robust and persists across K=4--8."
        )
    elif global_statewise_supported and len(joint_states) >= 2:
        conclusion_category = "multistate_construction_aware_persistence_supported"
        conclusion = (
            "Multiple states show construction-aware fixed-10 persistence, but the broader "
            "metastable-like wording should be limited to the robustness dimensions that remain positive."
        )
    elif global_statewise_supported and len(joint_states) == 1:
        conclusion_category = "localized_construction_aware_persistence_only"
        conclusion = (
            "Construction-aware persistence is localized to one state; report persistent mesostate "
            "organisation rather than population-level metastable-like organisation."
        )
    else:
        conclusion_category = "construction_aware_metastable_like_not_supported"
        conclusion = (
            "No state satisfies the joint construction-aware and learner-cluster fixed-10 criteria; "
            "retain persistent mesostate organisation and remove metastable-like wording."
        )
    recursive_state_text = ", ".join(f"S{state}" for state in recursive_states) or "none"
    joint_state_text = ", ".join(f"S{state}" for state in joint_states) or "none"


    report = "\n".join(
        [
            "# Kinetic null, partition sensitivity and learner-cluster inference",
            "",
            "The analysis is confined to the empirical coarse-kinetic branch. It does not rerun the construction-matched field null, coordinate/grid sensitivity, strict-user-equal transition analysis, positive-exponential learner reweighting, the minimal mechanism or Event-SSL.",
            "",
            "The recursive surrogate reuses the existing A_train-frozen opportunity matching and jointly permuted innovation pairs, but propagates a coherent state sequence along the empirical denominator and observation skeleton. It is conditional on that denominator path and is not an autonomous learner-platform simulator.",
            "",
            f"The original all-state fixed-10 statistic was {fmt(primary_row['observed'])} in the data versus a recursive-null median of {fmt(primary_row['null_median'])} (Monte Carlo p={fmt(primary_row['monte_carlo_p'])}).",
            "",
            f"The complementary statewise studentized maxT statistic was {fmt(familywise_row['observed_maxT'])} versus a null interval {fmt(familywise_row['null_2p5'])}--{fmt(familywise_row['null_97p5'])} (global Monte Carlo p={fmt(familywise_row['monte_carlo_p'])}); maxT-FWER-positive states were {recursive_state_text}.",
            "",
            f"Under formal K=6 learner-cluster inference, {rmst_robust}/6 statewise RMST-lift intervals had lower bounds above one and {cluster_positive}/6 fixed-10 tail tests remained positive after BH adjustment. States satisfying the recursive maxT, learner-cluster fixed-10 and learner-cluster RMST criteria jointly were {joint_state_text}.",
            "",
            f"Conclusion contract ({conclusion_category}): {conclusion}",
            "",
            f"{sensitivity_label} are bounded partition-resolution sensitivity analyses only; they do not reselect or replace the a priori K=6 partition.",
            "",
            "## Supplementary Table 1a. Recursive construction- and denominator-inertia-matched kinetic surrogate",
            "",
            markdown_table(table1a),
            "",
            "The aggregate and statewise endpoints are both retained. The aggregate endpoint tests a common all-state shift; the statewise fixed-10 rows use single-step studentized maxT family-wise inference across all six frozen states.",
            "",
            "## Supplementary Table 1b. Bounded partition-resolution sensitivity",
            "",
            markdown_table(table1b),
            "",
            "## Supplementary Table 2. Formal K=6 learner-cluster kinetic inference",
            "",
            markdown_table(table2),
            "",
            "The cluster-bootstrap tail p values use the bootstrap standard error in the same one-sided normal-approximation form as the original Greenwood analysis, followed by BH adjustment across the six frozen states. Percentile intervals are reported separately. The existing positive-exponential residence ranges should be removed or cross-referenced rather than presented as an additional parallel inferential result.",
            "",
        ]
    )
    report_path = output_root / "kinetic_robustness_report.md"
    report_path.write_text(report, encoding="utf-8")

    source_rows = []
    source_items = [
        ("partition_manifest", partition_manifest_path),
        ("recursive_null_manifest", null_manifest_path),
        ("partition_summary", table_path(partition_root / "tables" / "partition_robustness_summary")),
        ("cluster_summary", table_path(partition_root / "tables" / "learner_cluster_bootstrap_statewise_summary")),
        ("recursive_null_summary", table_path(null_root / "tables" / "recursive_construction_inertia_null_summary")),
        ("recursive_statewise_summary", table_path(null_root / "tables" / "recursive_construction_inertia_statewise_summary")),
        ("recursive_statewise_replicates", table_path(null_root / "tables" / "recursive_construction_inertia_null_statewise_replicates")),
    ]
    if familywise_source is not None:
        source_items.append(("recursive_statewise_familywise_summary", familywise_source))
    for label, path in source_items:
        source_rows.append(
            {
                "source": label,
                "path": str(path.resolve()),
                "sha256": sha256_file(path.resolve()),
                "bytes": int(path.stat().st_size),
            }
        )
    source_audit = pd.DataFrame(source_rows)
    source_audit_path = write_table(source_audit, output_root / "kinetic_source_audit")

    manifest = {
        "script": Path(__file__).name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "partition_root": str(partition_root),
        "recursive_null_root": str(null_root),
        "typeset_tables": 2,
        "table_structure": {
            "table1": [
                "recursive aggregate kinetic null",
                "statewise maxT family-wise construction-aware inference",
                "bounded partition-resolution sensitivity with formal K=6 retained",
            ],
            "table2": ["formal K=6 learner-cluster Pii, RMST and fixed-10-step inference"],
        },
        "scientific_summary": {
            "aggregate_monte_carlo_p": float(primary_row["monte_carlo_p"]),
            "statewise_maxT_global_p": float(familywise_row["monte_carlo_p"]),
            "recursive_fwer_positive_states": recursive_states,
            "joint_supported_states": joint_states,
            "conclusion_category": conclusion_category,
            "recommended_claim": conclusion,
        },
        "duplication_policy": {
            "existing_multiplier_residence_row": "replace with or cross-reference the learner-cluster kinetic table",
            "existing_strict_user_equal_transition_result": "retain only in the earlier estimand-sensitivity table; do not duplicate here",
            "existing_construction_matched_field_null": "unchanged and not repeated",
            "existing_coordinate_and_grid_sensitivity": "unchanged and not repeated",
        },
        "outputs": {
            "report": str(report_path),
            "table1a": str(table1a_path),
            "table1b": str(table1b_path),
            "table2": str(table2_path),
            "source_audit": str(source_audit_path),
        },
    }
    save_json(manifest, output_root / "kinetic_robustness_summary_manifest.json")
    print(f"[kinetic summary] completed: {output_root}", flush=True)


if __name__ == "__main__":
    main()
