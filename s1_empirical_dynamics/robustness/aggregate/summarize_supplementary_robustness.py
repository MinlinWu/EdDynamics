#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from robustness_common import read_table, save_json, sha256_file, write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect the frozen supplementary robustness analyses into manuscript-oriented tables.")
    parser.add_argument("--empirical-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/supplementary_robustness/empirical"))
    parser.add_argument("--model-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/supplementary_robustness/models"))
    parser.add_argument("--representation-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/supplementary_robustness/representations"))
    parser.add_argument("--coordinate-summary-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4_stage1_sensitivity/summary"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/supplementary_robustness/summary"))
    return parser.parse_args()


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return ""
    if not np.isfinite(number):
        return ""
    if number != 0 and (abs(number) < 1e-3 or abs(number) >= 1e4):
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def markdown_table(frame: pd.DataFrame, max_rows: int = 200) -> str:
    data = frame.head(max_rows).copy()
    columns = [str(column) for column in data.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in data.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", "<br>") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing columns: {missing}")


def bootstrap_table(empirical_root: Path, model_root: Path) -> pd.DataFrame:
    empirical = read_table(empirical_root / "tables" / "empirical_user_multiplier_bootstrap_summary")
    model = read_table(model_root / "tables" / "model_user_multiplier_bootstrap_summary")
    require_columns(empirical, ["metric", "formal_point_estimate", "ci_2p5", "ci_97p5"], "empirical bootstrap summary")
    require_columns(model, ["metric", "formal_point_estimate", "ci_2p5", "ci_97p5"], "model bootstrap summary")
    empirical_metrics = {
        "train_validation_mean_local_drift_cosine": "Empirical train-validation local drift cosine",
        "validation_negative_divergence_fraction": "Validation negative-divergence occupancy",
        "validation_shell_inward_fraction": "Validation frozen-shell inward fraction",
        "validation_core_to_shell_speed_ratio": "Validation core-to-shell speed ratio",
    }
    model_metrics = {
        "mechanism_drift_vector_corr": "Mechanism confirmation drift correlation",
        "event_ssl_learned_drift_vector_corr": "Event-SSL learned-plane drift correlation",
        "event_ssl_learned_weighted_local_cosine": "Event-SSL learned-plane weighted local cosine",
        "time_shuffle_learned_drift_vector_corr": "Time-shuffle learned-plane drift correlation",
        "main_minus_tag_inward_fraction": "Main-minus-randomized inward fraction",
        "cross_model_anchor_drift_vector_corr": "Mechanism-Event-SSL anchor-field correlation",
        "cross_model_self_transition_corr": "Mechanism-Event-SSL self-transition correlation",
    }
    rows: List[Dict[str, Any]] = []
    for source, frame, mapping in (
        ("empirical", empirical, empirical_metrics),
        ("frozen_models", model, model_metrics),
    ):
        indexed = frame.set_index("metric")
        for metric, label in mapping.items():
            if metric not in indexed.index:
                raise RuntimeError(f"Required bootstrap metric is absent: {metric}")
            row = indexed.loc[metric]
            rows.append({
                "source": source,
                "quantity": label,
                "metric": metric,
                "point": float(row["formal_point_estimate"]),
                "ci_2p5": float(row["ci_2p5"]),
                "ci_97p5": float(row["ci_97p5"]),
                "finite_replicates": int(row["replicates_finite"]),
            })
    return pd.DataFrame(rows)


def residence_table(empirical_root: Path) -> pd.DataFrame:
    frame = read_table(empirical_root / "tables" / "empirical_residence_user_multiplier_bootstrap_summary")
    require_columns(frame, ["macrostate", "metric", "formal_point_estimate", "ci_2p5", "ci_97p5"], "residence bootstrap summary")
    selected = frame[frame["metric"].astype(str).isin(["rmst_lift", "tail_excess"])].copy()
    return selected[["macrostate", "metric", "formal_point_estimate", "ci_2p5", "ci_97p5", "replicates_finite"]].sort_values(["macrostate", "metric"])


def representation_raw_table(representation_root: Path) -> pd.DataFrame:
    point = read_table(representation_root / "tables" / "stage5_raw_metric_point_estimates")
    bootstrap = read_table(representation_root / "tables" / "stage5_user_multiplier_bootstrap_summary")
    require_columns(point, ["representation", "metric", "value"], "Stage-5 point estimates")
    require_columns(bootstrap, ["comparison", "representation", "metric", "ci_2p5", "ci_97p5"], "Stage-5 bootstrap summary")
    absolute = bootstrap[bootstrap["comparison"].astype(str) == "absolute"].copy()
    merged = point.merge(
        absolute[["representation", "metric", "ci_2p5", "ci_97p5", "replicates_finite"]],
        on=["representation", "metric"],
        how="left",
        validate="one_to_one",
    )
    return merged.sort_values(["metric", "representation"])


def representation_contrast_table(representation_root: Path) -> pd.DataFrame:
    bootstrap = read_table(representation_root / "tables" / "stage5_user_multiplier_bootstrap_summary")
    require_columns(
        bootstrap,
        ["comparison", "representation", "metric", "formal_point_estimate", "ci_2p5", "ci_97p5", "replicates_finite"],
        "Stage-5 bootstrap summary",
    )
    contrast = bootstrap[
        bootstrap["comparison"].astype(str) == "macro_only_minus_full_hidden"
    ].copy()
    lower_is_better = {
        "one_step_rmse_M",
        "one_step_rmse_Psi",
        "learned_plane_transition_mean_row_tv",
    }
    contrast["metric_direction"] = [
        "lower_is_better" if str(metric) in lower_is_better else "higher_is_better"
        for metric in contrast["metric"]
    ]
    contrast["macro_minus_full_interpretation"] = [
        "positive_is_worse" if str(metric) in lower_is_better else "positive_is_better"
        for metric in contrast["metric"]
    ]
    columns = [
        "metric",
        "metric_direction",
        "macro_minus_full_interpretation",
        "formal_point_estimate",
        "ci_2p5",
        "ci_97p5",
        "replicates_finite",
    ]
    return contrast[columns].sort_values("metric")


def compact_coordinate_table(coordinate_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(coordinate_root / "empirical_coordinate_sensitivity_statistics.csv", low_memory=False)
    columns = [
        "setting_id",
        "setting",
        "analysis",
        "A_train_A_val_mean_local_drift_cosine",
        "A_val_negative_divergence_occupancy_fraction",
        "A_val_frozen_shell_inward_fraction",
        "A_val_frozen_core_to_shell_speed_ratio",
        "directional_result_retained",
    ]
    require_columns(frame, columns, "coordinate-sensitivity statistics")
    return frame[columns].rename(columns={"setting": "label"}).copy()


def raw_floor_table(representation_root: Path) -> pd.DataFrame:
    return read_table(representation_root / "tables" / "stage5_within_user_permutation_floor_summary")


def score_table(representation_root: Path) -> pd.DataFrame:
    frame = read_table(representation_root / "tables" / "stage5_descriptive_score_sensitivity")
    selected = frame[frame["quantity"].astype(str).isin([
        "macro_retention_vs_full",
        "macro_null_headroom_retention_vs_full",
    ])].copy()
    return selected.sort_values(["contract", "quantity"])


def build_report(
    bootstrap: pd.DataFrame,
    residence: pd.DataFrame,
    representation: pd.DataFrame,
    representation_contrasts: pd.DataFrame,
    coordinate: pd.DataFrame,
    strict: pd.DataFrame,
    grids: pd.DataFrame,
    floor: pd.DataFrame,
    scores: pd.DataFrame,
) -> str:
    bootstrap_display = bootstrap.copy()
    bootstrap_display["estimate (95% learner interval)"] = [
        f"{fmt(row.point)} ({fmt(row.ci_2p5)}, {fmt(row.ci_97p5)})"
        for row in bootstrap_display.itertuples(index=False)
    ]
    representation_display = representation.copy()
    representation_display["estimate (95% learner interval)"] = [
        f"{fmt(row.value)} ({fmt(row.ci_2p5)}, {fmt(row.ci_97p5)})"
        for row in representation_display.itertuples(index=False)
    ]
    lines = [
        "# Learner-level and evaluation-gauge robustness report",
        "",
        "Learner-bootstrap, strict-weighting and grid analyses use the formal frozen coordinates, models, probes, convergence core and fixed K=6 partition. Their multiplier intervals condition on this frozen analysis contract. No model is retrained, no probe or mechanism parameter is refitted, no construction-matched null is rerun and no new p value is produced.",
        "",
        "The existing predeclared memory and activity-quality coordinate variants are collected here but are not rerun or crossed with the new analyses. Grid analyses are restricted to field-level quantities and do not redefine convergence regions or mesostates. Stage-5 raw metrics remain primary; composite-score and permutation-floor quantities are descriptive diagnostics.",
        "",
        "Statewise self-transition correlations use six frozen states. Their learner-multiplier intervals describe how the six probabilities change with learner composition; they are not independent-state sampling intervals and do not generate p values.",
        "",
        "## Learner-level intervals for manuscript-key quantities",
        "",
        markdown_table(bootstrap_display[["source", "quantity", "estimate (95% learner interval)", "finite_replicates"]], 50),
        "",
        "## Residence-time learner-cluster intervals",
        "",
        markdown_table(residence, 30),
        "",
        "## Stage-5 raw representation metrics",
        "",
        markdown_table(representation_display[["representation", "metric", "estimate (95% learner interval)", "replicates_finite"]], 80),
        "",
        "## Paired macro-only minus full-hidden contrasts",
        "",
        markdown_table(representation_contrasts, 30),
        "",
        "## Memory and activity-quality coordinate sensitivity",
        "",
        markdown_table(coordinate, 20),
        "",
        "## Strict user-equal sensitivity",
        "",
        markdown_table(strict, 120),
        "",
        "## Finite field-grid and boundary sensitivity",
        "",
        markdown_table(grids, 120),
        "",
        "## Within-user descriptive permutation floor",
        "",
        markdown_table(floor, 80),
        "",
        "## Descriptive-score sensitivity",
        "",
        markdown_table(scores, 30),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    started = time.time()
    empirical_root = args.empirical_root.resolve()
    model_root = args.model_root.resolve()
    representation_root = args.representation_root.resolve()
    coordinate_root = args.coordinate_summary_root.resolve()
    output_root = args.output_root.resolve()
    table_root = output_root / "tables"
    report_root = output_root / "reports"
    metadata_root = output_root / "metadata"
    for directory in (table_root, report_root, metadata_root):
        directory.mkdir(parents=True, exist_ok=True)

    bootstrap = bootstrap_table(empirical_root, model_root)
    residence = residence_table(empirical_root)
    representation = representation_raw_table(representation_root)
    representation_contrasts = representation_contrast_table(representation_root)
    coordinate = compact_coordinate_table(coordinate_root)
    strict_empirical = read_table(empirical_root / "tables" / "empirical_strict_user_equal_sensitivity")
    strict_empirical_transition = read_table(
        empirical_root / "tables" / "empirical_transition_user_equal_sensitivity"
    )
    strict_model = read_table(model_root / "tables" / "model_strict_user_equal_sensitivity")
    strict_representation = read_table(representation_root / "tables" / "stage5_strict_user_equal_sensitivity")
    strict_empirical.insert(0, "source", "empirical_field")
    strict_empirical_transition.insert(0, "source", "empirical_transition")
    strict_model.insert(0, "source", "frozen_models")
    strict_representation.insert(0, "source", "stage5_representations")
    strict = pd.concat(
        [strict_empirical, strict_empirical_transition, strict_model, strict_representation],
        ignore_index=True,
        sort=False,
    )
    grid_empirical = read_table(empirical_root / "tables" / "empirical_grid_sensitivity")
    grid_model = read_table(model_root / "tables" / "model_grid_sensitivity")
    grid_representation = read_table(representation_root / "tables" / "stage5_grid_sensitivity")
    grid_empirical.insert(0, "source", "empirical")
    grid_model.insert(0, "source", "frozen_models")
    grid_representation.insert(0, "source", "stage5_representations")
    grids = pd.concat([grid_empirical, grid_model, grid_representation], ignore_index=True, sort=False)
    floor = raw_floor_table(representation_root)
    scores = score_table(representation_root)

    outputs = {
        "bootstrap_key": write_table(bootstrap, table_root / "supplementary_learner_bootstrap_key_metrics"),
        "residence": write_table(residence, table_root / "supplementary_residence_cluster_bootstrap"),
        "representation_raw": write_table(representation, table_root / "supplementary_stage5_raw_metrics"),
        "representation_paired_contrasts": write_table(
            representation_contrasts,
            table_root / "supplementary_stage5_macro_minus_full_paired_contrasts",
        ),
        "coordinate_sensitivity": write_table(coordinate, table_root / "supplementary_coordinate_sensitivity"),
        "strict_user_equal": write_table(strict, table_root / "supplementary_strict_user_equal_sensitivity"),
        "grid_sensitivity": write_table(grids, table_root / "supplementary_grid_sensitivity"),
        "permutation_floor": write_table(floor, table_root / "supplementary_stage5_permutation_floor"),
        "score_sensitivity": write_table(scores, table_root / "supplementary_stage5_score_sensitivity"),
    }
    report_path = report_root / "supplementary_robustness_report.md"
    report_path.write_text(
        build_report(
            bootstrap,
            residence,
            representation,
            representation_contrasts,
            coordinate,
            strict,
            grids,
            floor,
            scores,
        ),
        encoding="utf-8",
    )

    source_paths = {
        "empirical_manifest": empirical_root / "metadata" / "empirical_robustness_manifest.json",
        "model_manifest": model_root / "metadata" / "model_robustness_manifest.json",
        "representation_manifest": representation_root / "metadata" / "representation_robustness_manifest.json",
        "coordinate_manifest": coordinate_root / "empirical_coordinate_sensitivity_manifest.json",
    }
    manifest = {
        "script": Path(__file__).name,
        "output_root": str(output_root),
        "analysis_boundary": {
            "models_retrained": False,
            "probes_refit": False,
            "mechanism_refit": False,
            "KMeans_refit": False,
            "convergence_core_reselected": False,
            "construction_matched_null_rerun": False,
            "new_p_values": False,
            "coordinate_sensitivity_rerun": False,
        },
        "source_audit": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
            for name, path in source_paths.items()
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
        "report": str(report_path),
        "elapsed_seconds": float(time.time() - started),
    }
    save_json(manifest, metadata_root / "supplementary_robustness_collection_manifest.json")
    print(f"[supplementary robustness summary] wrote {report_path}")


if __name__ == "__main__":
    main()
