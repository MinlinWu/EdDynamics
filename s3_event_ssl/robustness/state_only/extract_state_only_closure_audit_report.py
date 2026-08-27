#!/usr/bin/env python3
from __future__ import annotations

"""Extract the numerical report for the state-only closure adequacy audit."""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(key): json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(obj), handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_table(base_or_path: Path) -> Path:
    path = Path(base_or_path)
    if path.exists() and path.is_file():
        return path
    for extension in (".parquet", ".csv.gz", ".csv"):
        candidate = path.with_suffix(extension)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(base_or_path)


def read_table(base_or_path: Path) -> pd.DataFrame:
    path = find_table(base_or_path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def write_table(frame: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    parquet = base.with_suffix(".parquet")
    temporary = parquet.with_name(parquet.name + ".tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, parquet)
        return parquet
    except Exception:
        if temporary.exists():
            temporary.unlink()
        csv_path = base.with_suffix(".csv.gz")
        temporary_csv = csv_path.with_name(csv_path.name + ".tmp")
        frame.to_csv(temporary_csv, index=False, compression="gzip")
        os.replace(temporary_csv, csv_path)
        return csv_path


def markdown_table(frame: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> str:
    selected = frame.copy()
    if columns is not None:
        selected = selected[[column for column in columns if column in selected.columns]]
    if selected.empty:
        return "_No rows._"
    display = selected.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if not np.isfinite(value) else f"{float(value):.6g}"
            )
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def selected_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "selected" not in frame.columns:
        return pd.DataFrame()
    return frame[frame["selected"].astype(bool)].copy()


def metric_subset(frame: pd.DataFrame, seed: int, split: str) -> pd.DataFrame:
    selected = frame[(frame["seed"] == seed) & (frame["split"] == split)].copy()
    columns = [
        "model",
        "one_step_rmse_M",
        "one_step_rmse_Psi",
        "next_state_occupancy_js",
        "learned_plane_drift_vector_corr",
        "learned_plane_drift_speed_corr",
        "learned_plane_occupancy_weighted_local_drift_cosine",
        "matched_origin_drift_vector_corr",
        "matched_origin_drift_speed_corr",
        "matched_origin_occupancy_weighted_local_drift_cosine",
        "learned_plane_transition_mean_row_tv",
        "learned_plane_self_transition_corr",
        "matched_origin_transition_mean_row_tv",
        "matched_origin_self_transition_corr",
        "transition_mean_row_tv",
        "self_transition_corr",
        "diagonal_dominance_predicted_states",
        "top_transition_edge_overlap",
        "matched_origin_diagonal_dominance_predicted_states",
        "matched_origin_top_transition_edge_overlap",
        "gaussian_nll",
        "output_clip_fraction",
        "rho",
        "boundary_censoring_mass_any",
        "boundary_censoring_mass_M",
        "boundary_censoring_mass_Psi",
    ]
    return selected[[column for column in columns if column in selected.columns]]


def floor_key_rows(frame: pd.DataFrame) -> pd.DataFrame:
    desired = {
        "learned_plane_drift_vector_corr",
        "matched_origin_drift_vector_corr",
        "learned_plane_occupancy_weighted_local_drift_cosine",
        "matched_origin_occupancy_weighted_local_drift_cosine",
        "learned_plane_transition_mean_row_tv",
        "matched_origin_transition_mean_row_tv",
        "learned_plane_self_transition_corr",
        "matched_origin_self_transition_corr",
        "transition_mean_row_tv",
        "self_transition_corr",
        "learned_plane_diagonal_dominance_match_fraction",
        "matched_origin_diagonal_dominance_match_fraction",
        "diagonal_dominance_match_fraction",
        "learned_plane_top_transition_edge_overlap",
        "matched_origin_top_transition_edge_overlap",
        "top_transition_edge_overlap",
    }
    return frame[frame["metric"].isin(desired)].copy()


def seed_summary(seed_differences: pd.DataFrame) -> pd.DataFrame:
    numeric = [column for column in seed_differences.columns if column not in {"seed", "split"}]
    rows: List[Dict[str, Any]] = []
    for split, subset in seed_differences.groupby("split", sort=True):
        for metric in numeric:
            values = pd.to_numeric(subset[metric], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            lower_is_better_delta = "transition_tv" in metric
            favourable = values < 0 if lower_is_better_delta else values > 0
            mean = float(np.mean(values))
            sample_sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            if values.size > 1:
                half_width = float(
                    stats.t.ppf(0.975, df=values.size - 1)
                    * sample_sd / np.sqrt(values.size)
                )
                interval_low = mean - half_width
                interval_high = mean + half_width
            else:
                interval_low = mean
                interval_high = mean
            rows.append({
                "split": split,
                "metric": metric,
                "favourable_direction": "negative" if lower_is_better_delta else "positive",
                "n_seeds": int(values.size),
                "mean": mean,
                "sample_sd": sample_sd,
                "t_interval_2p5": float(interval_low),
                "t_interval_97p5": float(interval_high),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "favourable_seeds": int(np.sum(favourable)),
                "unfavourable_seeds": int(np.sum(~favourable)),
                "positive_seeds": int(np.sum(values > 0)),
                "negative_seeds": int(np.sum(values < 0)),
            })
    return pd.DataFrame(rows)


def interpretation_flags(
    split_metrics: pd.DataFrame,
    floor_summary: pd.DataFrame,
    primary_seed: int,
) -> Dict[str, Any]:
    confirm = split_metrics[
        (split_metrics["seed"] == primary_seed) & (split_metrics["split"] == "B_confirm")
    ]
    models = {str(row["model"]): row for _, row in confirm.iterrows()}

    def metric(model: str, name: str) -> float:
        row = models.get(model)
        if row is None:
            return float("nan")
        value = pd.to_numeric(pd.Series([row.get(name, np.nan)]), errors="coerce").iloc[0]
        return float(value) if np.isfinite(value) else float("nan")

    def compare(first_model: str, second_model: str, name: str, lower_is_better: bool = False) -> Optional[bool]:
        first = metric(first_model, name)
        second = metric(second_model, name)
        if not np.isfinite(first) or not np.isfinite(second):
            return None
        return bool(first < second) if lower_is_better else bool(first > second)

    def floor_flag(model: str, name: str) -> Optional[bool]:
        rows = floor_summary[(floor_summary["model"] == model) & (floor_summary["metric"] == name)]
        if len(rows) != 1:
            return None
        return bool(rows.iloc[0]["favourable_side_of_floor"])

    return {
        "primary_seed": primary_seed,
        "tuned_quadratic_improves_formal_drift": compare(
            "quadratic_tuned", "quadratic_mean", "learned_plane_drift_vector_corr"
        ),
        "spline_improves_tuned_quadratic_formal_drift": compare(
            "spline_mean", "quadratic_tuned", "learned_plane_drift_vector_corr"
        ),
        "spline_improves_tuned_quadratic_matched_drift": compare(
            "spline_mean", "quadratic_tuned", "matched_origin_drift_vector_corr"
        ),
        "spline_improves_tuned_quadratic_formal_transition_tv": compare(
            "spline_mean", "quadratic_tuned", "learned_plane_transition_mean_row_tv", True
        ),
        "spline_improves_tuned_quadratic_formal_self_transition": compare(
            "spline_mean", "quadratic_tuned", "learned_plane_self_transition_corr"
        ),
        "spline_improves_tuned_quadratic_matched_transition_tv": compare(
            "spline_mean", "quadratic_tuned", "matched_origin_transition_mean_row_tv", True
        ),
        "spline_improves_tuned_quadratic_matched_self_transition": compare(
            "spline_mean", "quadratic_tuned", "matched_origin_self_transition_corr"
        ),
        "gaussian_improves_tuned_quadratic_formal_transition_tv": compare(
            "gaussian_distribution", "quadratic_tuned", "transition_mean_row_tv", True
        ),
        "gaussian_improves_tuned_quadratic_formal_self_transition": compare(
            "gaussian_distribution", "quadratic_tuned", "self_transition_corr"
        ),
        "gaussian_improves_tuned_quadratic_matched_transition_tv": compare(
            "gaussian_distribution", "quadratic_tuned", "matched_origin_transition_mean_row_tv", True
        ),
        "gaussian_improves_tuned_quadratic_matched_self_transition": compare(
            "gaussian_distribution", "quadratic_tuned", "matched_origin_self_transition_corr"
        ),
        "spline_formal_drift_above_own_floor": floor_flag(
            "spline_mean", "learned_plane_drift_vector_corr"
        ),
        "spline_matched_drift_above_own_floor": floor_flag(
            "spline_mean", "matched_origin_drift_vector_corr"
        ),
        "spline_formal_self_transition_above_own_floor": floor_flag(
            "spline_mean", "learned_plane_self_transition_corr"
        ),
        "spline_matched_self_transition_above_own_floor": floor_flag(
            "spline_mean", "matched_origin_self_transition_corr"
        ),
        "gaussian_formal_transition_tv_above_own_floor": floor_flag(
            "gaussian_distribution", "transition_mean_row_tv"
        ),
        "gaussian_formal_self_transition_above_own_floor": floor_flag(
            "gaussian_distribution", "self_transition_corr"
        ),
        "gaussian_matched_transition_tv_above_own_floor": floor_flag(
            "gaussian_distribution", "matched_origin_transition_mean_row_tv"
        ),
        "gaussian_matched_self_transition_above_own_floor": floor_flag(
            "gaussian_distribution", "matched_origin_self_transition_corr"
        ),
    }


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the state-only closure audit report.")
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    analysis_root = args.analysis_root.resolve()
    manifest_path = analysis_root / "metadata" / "state_only_closure_audit_manifest.json"
    checksum_path = analysis_root / "metadata" / "state_only_closure_audit_manifest.sha256.json"
    manifest = load_json(manifest_path)
    checksum = load_json(checksum_path)
    actual_manifest_hash = file_sha256(manifest_path)
    if actual_manifest_hash != str(checksum.get("manifest_sha256", "")):
        raise RuntimeError("Analysis manifest checksum failed.")
    if manifest.get("completed") is not True:
        raise RuntimeError("Analysis manifest is not complete.")
    if manifest.get("quadratic_metric_reconstruction_passed") is not True:
        raise RuntimeError("Formal quadratic metric reconstruction did not pass.")
    sample_audit = dict(manifest.get("quadratic_sample_reconstruction", {}))
    if sample_audit.get("passed") is not True:
        raise RuntimeError("Formal quadratic sample reconstruction did not pass.")

    output_root = args.output_root.resolve() if args.output_root is not None else analysis_root / "numeric_report"
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Report directory is not empty: {output_root}")
        shutil.rmtree(output_root)
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    table_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    quadratic_candidates = read_table(analysis_root / "tables" / "quadratic_alpha_control_selection")
    mean_candidates = read_table(analysis_root / "tables" / "mean_closure_candidate_selection")
    distribution_candidates = read_table(analysis_root / "tables" / "distributional_closure_candidate_selection")
    variance_crossfit = read_table(analysis_root / "tables" / "distributional_variance_crossfit_audit")
    split_metrics = read_table(analysis_root / "tables" / "state_only_closure_split_metrics")
    statewise = read_table(analysis_root / "tables" / "state_only_closure_statewise_transitions")
    seed_differences = read_table(analysis_root / "tables" / "state_only_closure_seed_differences")
    reconstruction = read_table(analysis_root / "tables" / "quadratic_reconstruction_audit")
    quadrature_audit = read_table(analysis_root / "tables" / "gauss_hermite_order_audit")
    quadrature_audit_meta = load_json(analysis_root / "metadata" / "gauss_hermite_order_audit.json")
    floor_replicates = read_table(analysis_root / "tables" / "state_only_closure_permutation_floor_replicates")
    floor_summary = read_table(analysis_root / "tables" / "state_only_closure_permutation_floor_summary")

    if not reconstruction["passed"].astype(bool).all():
        raise RuntimeError("At least one quadratic reconstruction row failed.")
    if quadrature_audit.empty or quadrature_audit_meta.get("passed") is not True:
        raise RuntimeError("Gaussian quadrature convergence audit failed.")

    primary_seed = int(manifest["contract"]["primary_seed"])
    selected_quadratic = selected_rows(quadratic_candidates)
    selected_mean = selected_rows(mean_candidates)
    selected_distribution = selected_rows(distribution_candidates)
    if len(selected_quadratic) != 1 or len(selected_mean) != 1 or len(selected_distribution) != 1:
        raise RuntimeError("Exactly one selected quadratic, mean and distributional closure are required.")

    seed_stats = seed_summary(seed_differences)
    floor_keys = floor_key_rows(floor_summary)
    flags = interpretation_flags(split_metrics, floor_summary, primary_seed)
    primary_statewise = statewise[
        (statewise["seed"] == primary_seed) & (statewise["split"] == "B_confirm")
    ].copy()

    copied_tables = {
        "quadratic_alpha_control_selection": write_table(
            quadratic_candidates, table_root / "quadratic_alpha_control_selection"
        ),
        "mean_candidate_selection": write_table(
            mean_candidates, table_root / "mean_closure_candidate_selection"
        ),
        "distributional_candidate_selection": write_table(
            distribution_candidates, table_root / "distributional_closure_candidate_selection"
        ),
        "distributional_variance_crossfit_audit": write_table(
            variance_crossfit, table_root / "distributional_variance_crossfit_audit"
        ),
        "split_metrics": write_table(split_metrics, table_root / "state_only_closure_split_metrics"),
        "statewise_transitions": write_table(
            statewise, table_root / "state_only_closure_statewise_transitions"
        ),
        "seed_differences": write_table(
            seed_differences, table_root / "state_only_closure_seed_differences"
        ),
        "seed_summary": write_table(seed_stats, table_root / "state_only_closure_seed_summary"),
        "quadratic_reconstruction_audit": write_table(
            reconstruction, table_root / "quadratic_reconstruction_audit"
        ),
        "gauss_hermite_order_audit": write_table(
            quadrature_audit, table_root / "gauss_hermite_order_audit"
        ),
        "permutation_floor_replicates": write_table(
            floor_replicates, table_root / "state_only_closure_permutation_floor_replicates"
        ),
        "permutation_floor_summary": write_table(
            floor_summary, table_root / "state_only_closure_permutation_floor_summary"
        ),
        "primary_floor_key_metrics": write_table(
            floor_keys, table_root / "primary_floor_key_metrics"
        ),
    }

    report_payload = {
        "analysis_manifest_sha256": actual_manifest_hash,
        "analysis_contract": manifest.get("contract", {}),
        "selected_contract": manifest.get("selected_contract", {}),
        "quadratic_sample_reconstruction": sample_audit,
        "quadratic_metric_reconstruction": reconstruction.to_dict(orient="records"),
        "selected_matched_quadratic": selected_quadratic.to_dict(orient="records")[0],
        "selected_mean_candidate": selected_mean.to_dict(orient="records")[0],
        "selected_distributional_candidate": selected_distribution.to_dict(orient="records")[0],
        "quadratic_alpha_control_candidates": quadratic_candidates.to_dict(orient="records"),
        "mean_closure_candidates": mean_candidates.to_dict(orient="records"),
        "distributional_closure_candidates": distribution_candidates.to_dict(orient="records"),
        "distributional_variance_crossfit_audit": variance_crossfit.to_dict(orient="records"),
        "gauss_hermite_order_audit": quadrature_audit.to_dict(orient="records"),
        "gauss_hermite_order_audit_metadata": quadrature_audit_meta,
        "primary_A_val_metrics": metric_subset(split_metrics, primary_seed, "A_val").to_dict(orient="records"),
        "primary_B_confirm_metrics": metric_subset(split_metrics, primary_seed, "B_confirm").to_dict(orient="records"),
        "all_split_metrics": split_metrics.to_dict(orient="records"),
        "all_statewise_transitions": statewise.to_dict(orient="records"),
        "all_seed_differences": seed_differences.to_dict(orient="records"),
        "all_seed_summary": seed_stats.to_dict(orient="records"),
        "all_permutation_floor_summary": floor_summary.to_dict(orient="records"),
        "all_permutation_floor_replicates": floor_replicates.to_dict(orient="records"),
        "primary_floor_key_metrics": floor_keys.to_dict(orient="records"),
        "six_seed_summary": seed_stats.to_dict(orient="records"),
        "primary_confirmation_statewise_transitions": primary_statewise.to_dict(orient="records"),
        "input_fingerprints": manifest.get("input_fingerprints", {}),
        "input_inventory": manifest.get("input_inventory", []),
        "source_hashes": manifest.get("source_hashes", {}),
        "interpretation_flags": flags,
        "interpretation_boundary": manifest.get("contract", {}).get("interpretation_boundary"),
    }

    lines: List[str] = []
    lines.append("# State-only closure adequacy audit")
    lines.append("")
    lines.append("## Analysis contract")
    lines.append("")
    lines.append(f"- Status: {manifest['contract']['status']}")
    lines.append(f"- Primary seed: {primary_seed}")
    lines.append(f"- Seeds: {', '.join(str(value) for value in manifest['contract']['seeds'])}")
    lines.append(f"- Input features: {', '.join(manifest['contract']['input_features'])}")
    lines.append(
        "- Mesostate labels used for fitting or hyperparameter selection: "
        f"{manifest['contract'].get('mesostate_labels_used_for_fitting_or_hyperparameter_selection')}"
    )
    lines.append(
        "- Empirical transition labels used for quadrature-order resolution: "
        f"{manifest['contract'].get('empirical_transition_labels_used_for_quadrature_order_resolution')}"
    )
    lines.append(
        "- Confirmation used for selection: "
        f"{manifest['contract']['fixed_contracts']['B_confirm_used_for_selection']}"
    )
    lines.append(
        "- Permutation-floor relationship to Supplementary Table 8: "
        f"{manifest['contract'].get('permutation_floor', {}).get('relationship_to_existing_table8_floor')}"
    )
    lines.append(f"- Interpretation boundary: {manifest['contract']['interpretation_boundary']}")
    lines.append("")
    lines.append("## Integrity gates")
    lines.append("")
    lines.append(markdown_table(reconstruction))
    lines.append("")
    lines.append("## Matched quadratic control")
    lines.append("")
    lines.append(markdown_table(quadratic_candidates))
    lines.append("")
    lines.append("## Selected nonlinear mean closure")
    lines.append("")
    lines.append(markdown_table(selected_mean))
    lines.append("")
    lines.append("## Selected continuous distributional closure")
    lines.append("")
    lines.append(markdown_table(selected_distribution))
    lines.append("")
    lines.append("## Distributional variance cross-fitting")
    lines.append("")
    lines.append(markdown_table(variance_crossfit))
    lines.append("")
    lines.append("## Gaussian quadrature convergence")
    lines.append("")
    lines.append(f"- Requested order: {quadrature_audit_meta.get('requested_order')}")
    lines.append(f"- Resolved order: {quadrature_audit_meta.get('resolved_order')}")
    lines.append(f"- Audit passed: {quadrature_audit_meta.get('passed')}")
    lines.append("")
    lines.append(markdown_table(quadrature_audit))
    lines.append("")
    lines.append("## Primary-seed validation metrics")
    lines.append("")
    lines.append(markdown_table(metric_subset(split_metrics, primary_seed, "A_val")))
    lines.append("")
    lines.append("## Primary-seed confirmation metrics")
    lines.append("")
    lines.append(markdown_table(metric_subset(split_metrics, primary_seed, "B_confirm")))
    lines.append("")
    lines.append("## Model-specific within-user permutation floors")
    lines.append("")
    lines.append(markdown_table(floor_keys, [
        "model", "metric", "observed", "floor_median", "floor_5p", "floor_95p",
        "oriented_improvement", "favourable_side_of_floor", "permutations",
    ]))
    lines.append("")
    lines.append("## Six-seed differences under matched controls")
    lines.append("")
    lines.append(markdown_table(seed_stats))
    lines.append("")
    lines.append("## Primary-seed confirmation statewise transitions")
    lines.append("")
    lines.append(markdown_table(primary_statewise, [
        "model", "gauge", "state", "empirical_self_transition", "predicted_self_transition",
        "self_transition_difference", "row_tv", "empirical_top_destination",
        "predicted_top_destination", "origin_intervals",
    ]))
    lines.append("")
    lines.append("## Interpretation flags")
    lines.append("")
    lines.append(markdown_table(pd.DataFrame([flags])))
    lines.append("")

    report_markdown = output_root / "state_only_closure_numeric_report.md"
    report_json = output_root / "state_only_closure_numeric_report.json"
    report_markdown.write_text("\n".join(lines), encoding="utf-8")
    save_json(report_payload, report_json)

    source_inventory: List[Dict[str, Any]] = []
    for path in sorted(analysis_root.rglob("*")):
        if path.is_file() and not is_within(path, output_root):
            source_inventory.append({
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(analysis_root)),
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            })
    output_inventory: List[Dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            output_inventory.append({
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(output_root)),
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            })
    report_manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "analysis_root": str(analysis_root),
        "analysis_manifest": str(manifest_path.resolve()),
        "analysis_manifest_sha256": actual_manifest_hash,
        "report_markdown": str(report_markdown.resolve()),
        "report_json": str(report_json.resolve()),
        "tables": {name: str(path.resolve()) for name, path in copied_tables.items()},
        "source_inventory": source_inventory,
        "output_inventory": output_inventory,
        "completed": True,
    }
    report_manifest_path = metadata_root / "state_only_closure_numeric_report_manifest.json"
    save_json(report_manifest, report_manifest_path)
    save_json(
        {"manifest_sha256": file_sha256(report_manifest_path)},
        metadata_root / "state_only_closure_numeric_report_manifest.sha256.json",
    )
    print(f"[state-only closure report] wrote {report_markdown}")


if __name__ == "__main__":
    main()
