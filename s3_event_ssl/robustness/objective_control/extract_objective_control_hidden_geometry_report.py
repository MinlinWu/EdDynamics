#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

MODEL_ORDER = ("predictive_state", "pure_ssl", "task_only")
MODEL_LABELS = {
    "predictive_state": "Predictive-state Event-SSL",
    "pure_ssl": "Pure SSL",
    "task_only": "Task-only",
}
SPLIT_ORDER = ("A_train", "A_val", "B_confirm")
KEY_METRICS = (
    "cca_corr_1",
    "cca_corr_2",
    "raw_ridge_corr_M",
    "raw_ridge_corr_Psi",
    "raw_ridge_rmse_M",
    "raw_ridge_rmse_Psi",
    "assigned_pc_abs_corr_M",
    "assigned_pc_abs_corr_Psi",
    "pca_pc1_pc2_explained_variance",
    "participation_ratio",
    "effective_rank",
    "matched_nonlinear_gain_corr_M",
    "matched_nonlinear_gain_corr_Psi",
    "matched_nonlinear_gain_rmse_reduction_M",
    "matched_nonlinear_gain_rmse_reduction_Psi",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_table(base: Path) -> Path:
    if base.exists() and base.is_file():
        return base
    for suffix in (".parquet", ".csv.gz", ".csv"):
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Table not found: {base}.[parquet|csv.gz|csv]")


def read_table(base: Path) -> Tuple[pd.DataFrame, Path]:
    path = locate_table(base)
    if path.suffix == ".parquet":
        return pd.read_parquet(path), path
    return pd.read_csv(path, low_memory=False), path


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


def finite_float(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def format_number(value: Any) -> str:
    number = finite_float(value)
    if not np.isfinite(number):
        return "--"
    absolute = abs(number)
    if absolute >= 10000:
        return f"{number:,.0f}"
    if absolute >= 100:
        return f"{number:.2f}"
    if absolute >= 1:
        return f"{number:.4f}"
    if absolute >= 0.001:
        return f"{number:.4f}"
    return f"{number:.3e}"


def markdown_table(frame: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if frame.empty:
        return "_No rows._"
    data = frame.copy()
    if max_rows is not None and len(data) > max_rows:
        data = data.head(max_rows)
    for column in data.columns:
        if pd.api.types.is_float_dtype(data[column]):
            data[column] = data[column].map(format_number)
        elif pd.api.types.is_bool_dtype(data[column]):
            data[column] = data[column].map(lambda value: "True" if bool(value) else "False")
        else:
            data[column] = data[column].map(lambda value: "--" if pd.isna(value) else str(value))
    headers = [str(column) for column in data.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in data.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    if max_rows is not None and len(frame) > max_rows:
        lines.append(f"\n_Showing the first {max_rows} of {len(frame)} rows._")
    return "\n".join(lines)


def require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} is missing columns: {missing}")


def build_quality_gates(
    manifest: Mapping[str, Any],
    metrics: pd.DataFrame,
    sample: pd.DataFrame,
    contracts: pd.DataFrame,
    identity: pd.DataFrame,
    reconstruction: pd.DataFrame,
    seed_manifests: Mapping[int, Mapping[str, Any]],
) -> pd.DataFrame:
    requested = [int(value) for value in manifest.get("requested_seeds", [])]
    completed = [int(value) for value in manifest.get("completed_seeds", [])]
    gates: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        gates.append({"gate": name, "passed": bool(passed), "detail": detail})

    add("all_requested_seeds_completed", set(requested) == set(completed), f"requested={requested}; completed={completed}")
    add("three_models_per_seed", len(contracts) == 3 * len(completed), f"rows={len(contracts)}")
    add(
        "all_models_have_all_splits",
        len(metrics[["seed", "model", "split"]].drop_duplicates()) == 3 * 3 * len(completed),
        f"unique model-seed-split rows={len(metrics[['seed', 'model', 'split']].drop_duplicates())}",
    )
    add("cross_seed_identity_gates", bool(identity["passed"].all()), f"failed={identity.loc[~identity['passed'], 'gate'].tolist()}")
    add(
        "shared_sample_seed",
        sample["sample_seed"].nunique(dropna=False) == 1,
        f"sample seeds={sorted(sample['sample_seed'].dropna().unique().tolist())}",
    )
    add(
        "shared_sample_indices_across_seeds",
        all(sample.loc[sample["split"] == split, "sample_index_hash"].nunique(dropna=False) == 1 for split in SPLIT_ORDER),
        "checked A_train, A_val and B_confirm",
    )
    add(
        "shared_sample_identifiers_across_seeds",
        all(sample.loc[sample["split"] == split, "sample_key_hash"].nunique(dropna=False) == 1 for split in SPLIT_ORDER),
        "checked A_train, A_val and B_confirm",
    )
    add(
        "prepared_input_content_identical_across_seeds",
        all(
            sample.loc[sample["split"] == split, "prepared_content_signature"].nunique(dropna=False) == 1
            for split in SPLIT_ORDER
        ),
        "full x_num, x_cat, targets, identifiers and sequence-offset files",
    )
    add(
        "current_and_next_targets_identical_across_seeds",
        all(
            sample.loc[sample["split"] == split, "current_target_hash"].nunique(dropna=False) == 1
            and sample.loc[sample["split"] == split, "next_target_hash"].nunique(dropna=False) == 1
            for split in SPLIT_ORDER
        ),
        "checked A_train, A_val and B_confirm",
    )
    add(
        "encoder_architecture_matched_within_seed",
        all(bool(seed_manifests[seed].get("source_audit", {}).get("architecture_matched")) for seed in completed),
        "hidden/input dimensions, layers, dropout and input schema",
    )
    add(
        "B_confirm_output_only",
        all(
            seed_manifests[seed].get("analysis_boundaries", {}).get("B_confirm_used_for_fitting") is False
            and seed_manifests[seed].get("analysis_boundaries", {}).get("B_confirm_used_for_selection") is False
            for seed in completed
        ),
        "all seed manifests",
    )
    add(
        "checkpoints_frozen",
        all(seed_manifests[seed].get("analysis_boundaries", {}).get("checkpoints_frozen") is True for seed in completed),
        "all seed manifests",
    )
    add(
        "no_macro_bottleneck_residual_or_dynamics_repeat",
        all(
            seed_manifests[seed].get("geometry_contract", {}).get("macro_bottleneck_or_residualisation_run") is False
            and seed_manifests[seed].get("geometry_contract", {}).get("drift_or_transition_metrics_run") is False
            and seed_manifests[seed].get("geometry_contract", {}).get("clustering_run") is False
            for seed in completed
        ),
        "all seed manifests",
    )
    if not reconstruction.empty:
        hard = reconstruction[reconstruction["hard_gate"] == True]
        add("formal_seed42_geometry_reconstruction", bool(hard["passed"].all()), f"hard rows={len(hard)}")
    else:
        add("formal_seed42_geometry_reconstruction", False, "reconstruction table missing")
    nonlinear = metrics[
        metrics[[column for column in metrics.columns if column.startswith("matched_nonlinear_gain")]].notna().any(axis=1)
    ] if any(column.startswith("matched_nonlinear_gain") for column in metrics.columns) else pd.DataFrame()
    nonlinear_seeds = sorted(pd.to_numeric(nonlinear.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist())
    add("nonlinear_capacity_limited_to_reference_seed", nonlinear_seeds in ([42], []), f"seeds={nonlinear_seeds}")
    return pd.DataFrame(gates)


def compact_six_seed_summary(summary: pd.DataFrame, split: str) -> pd.DataFrame:
    selected = summary[
        (summary["split"].astype(str) == split)
        & (summary["metric"].isin(KEY_METRICS))
    ].copy()
    selected["model_order"] = selected["model"].map({name: index for index, name in enumerate(MODEL_ORDER)})
    selected["metric_order"] = selected["metric"].map({name: index for index, name in enumerate(KEY_METRICS)})
    return selected.sort_values(["metric_order", "model_order"], kind="mergesort").drop(columns=["model_order", "metric_order"])


def seedwise_key_metrics(metrics: pd.DataFrame, split: str) -> pd.DataFrame:
    columns = ["seed", "model", "model_display", "split"] + [
        metric for metric in KEY_METRICS if metric in metrics.columns
    ]
    selected = metrics[metrics["split"].astype(str) == split][columns].copy()
    selected["model_order"] = selected["model"].map({name: index for index, name in enumerate(MODEL_ORDER)})
    return selected.sort_values(["seed", "model_order"], kind="mergesort").drop(columns="model_order")


def validation_confirmation_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [metric for metric in KEY_METRICS if metric in metrics.columns]
    rows: List[Dict[str, Any]] = []
    for (seed, model, display), group in metrics.groupby(["seed", "model", "model_display"], sort=True):
        validation = group[group["split"].astype(str) == "A_val"]
        confirmation = group[group["split"].astype(str) == "B_confirm"]
        if validation.empty or confirmation.empty:
            continue
        val_row = validation.iloc[0]
        confirm_row = confirmation.iloc[0]
        for metric in metric_columns:
            first = finite_float(val_row.get(metric))
            second = finite_float(confirm_row.get(metric))
            if not np.isfinite(first) or not np.isfinite(second):
                continue
            rows.append({
                "seed": int(seed),
                "model": model,
                "model_display": display,
                "metric": metric,
                "A_val": first,
                "B_confirm": second,
                "B_confirm_minus_A_val": second - first,
                "absolute_split_gap": abs(second - first),
            })
    return pd.DataFrame(rows)


def pc_assignment_table(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "seed",
        "model",
        "model_display",
        "split",
        "assigned_pc_M",
        "assigned_pc_Psi",
        "assigned_pc_corr_M",
        "assigned_pc_corr_Psi",
        "assigned_pc_abs_corr_M",
        "assigned_pc_abs_corr_Psi",
        "descriptive_split_max_pc_M",
        "descriptive_split_max_pc_Psi",
        "descriptive_split_max_abs_pc_corr_M",
        "descriptive_split_max_abs_pc_corr_Psi",
    ]
    available = [column for column in columns if column in metrics.columns]
    selected = metrics[available].copy()
    selected["model_order"] = selected["model"].map({name: index for index, name in enumerate(MODEL_ORDER)})
    selected["split_order"] = selected["split"].map({name: index for index, name in enumerate(SPLIT_ORDER)})
    return selected.sort_values(["seed", "model_order", "split_order"], kind="mergesort").drop(columns=["model_order", "split_order"])


def nonlinear_capacity_table(metrics: pd.DataFrame) -> pd.DataFrame:
    nonlinear_columns = [column for column in metrics.columns if column.startswith("pca64_hgb_") or column.startswith("matched_nonlinear_gain_")]
    columns = [
        "seed",
        "model",
        "model_display",
        "split",
        "pca64_ridge_corr_M",
        "pca64_ridge_corr_Psi",
        "pca64_ridge_rmse_M",
        "pca64_ridge_rmse_Psi",
    ] + nonlinear_columns
    available = [column for column in columns if column in metrics.columns]
    selected = metrics[available].copy()
    if nonlinear_columns:
        selected = selected[selected[nonlinear_columns].notna().any(axis=1)]
    return selected.sort_values(["seed", "model", "split"], kind="mergesort")


def reconstruction_summary(reconstruction: pd.DataFrame) -> pd.DataFrame:
    if reconstruction.empty:
        return reconstruction
    output = reconstruction.copy()
    output["status"] = np.where(output["passed"] == True, "passed", "failed")
    return output.sort_values(["category", "metric"], kind="mergesort")


def source_inventory(result_root: Path, seeds: Sequence[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        path = result_root / "seeds" / f"seed{seed}" / "metadata" / "source_hashes.json"
        if not path.exists():
            continue
        payload = load_json(path)
        for source in payload.get("sources", []):
            rows.append({"seed": int(seed), **dict(source)})
    return pd.DataFrame(rows)


def build_markdown(
    manifest: Mapping[str, Any],
    quality: pd.DataFrame,
    sample: pd.DataFrame,
    contracts: pd.DataFrame,
    reconstruction: pd.DataFrame,
    summary_b: pd.DataFrame,
    summary_v: pd.DataFrame,
    paired: pd.DataFrame,
    seedwise_b: pd.DataFrame,
    nonlinear: pd.DataFrame,
    pc_assignments: pd.DataFrame,
    split_deltas: pd.DataFrame,
    sources: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append("# Hidden-state geometry in objective-control representations")
    lines.append("")
    lines.append("## Analysis contract")
    lines.append("")
    lines.append(f"- Status: {manifest.get('status')}")
    lines.append(f"- Completed seeds: {', '.join(str(value) for value in manifest.get('completed_seeds', []))}")
    lines.append(f"- Reference seed: {manifest.get('reference_seed')}")
    lines.append(f"- Confidence level for seed-level Student-t intervals: {100.0 * float(manifest.get('confidence_level', 0.95)):.1f}%")
    lines.append(r"- Geometry target: current empirical \(M,\Psi\) from the pre-interval recurrent hidden state.")
    lines.append("- Fitting scope: scalers, PCA, CCA, Ridge and HGB use A_train only; A_val and B_confirm are output-only.")
    lines.append("- Interpretation: CCA is a supervised diagnostic. Objective controls differ in training objective and checkpoint selection and are not single-factor causal ablations.")
    lines.append("")

    lines.append("## Quality and integrity gates")
    lines.append("")
    lines.append(markdown_table(quality))
    lines.append("")

    lines.append("## Shared sample and input identity")
    lines.append("")
    sample_compact = sample[[
        "seed", "split", "rows", "users", "sample_rows", "sample_seed",
        "identifier_hash", "sample_index_hash", "sample_key_hash",
    ]].copy()
    lines.append(markdown_table(sample_compact, max_rows=24))
    lines.append("")

    lines.append("## Frozen objective and architecture contracts")
    lines.append("")
    contract_columns = [
        "seed", "model_display", "model_kind", "hidden_dim", "input_dim", "num_layers",
        "dropout", "n_num", "n_cat", "checkpoint_sha256", "nonlinear_probe_fitted",
    ]
    lines.append(markdown_table(contracts[[column for column in contract_columns if column in contracts.columns]], max_rows=24))
    lines.append("")

    lines.append("## Formal seed-42 reconstruction audit")
    lines.append("")
    if reconstruction.empty:
        lines.append("_No formal reconstruction rows were found._")
    else:
        compact = reconstruction[[
            "category", "metric", "observed", "archived", "absolute_difference", "tolerance", "passed",
        ]]
        lines.append(markdown_table(compact, max_rows=80))
    lines.append("")

    lines.append("## B_confirm six-seed geometry summary")
    lines.append("")
    summary_columns = [
        "model_display", "metric", "mean", "sample_sd", "ci_lower", "ci_upper", "minimum", "maximum", "n_seeds",
    ]
    lines.append(markdown_table(summary_b[[column for column in summary_columns if column in summary_b.columns]], max_rows=80))
    lines.append("")

    lines.append("## A_val six-seed geometry summary")
    lines.append("")
    lines.append(markdown_table(summary_v[[column for column in summary_columns if column in summary_v.columns]], max_rows=80))
    lines.append("")

    lines.append("## Same-seed objective-control contrasts")
    lines.append("")
    paired_columns = [
        "split", "control_display", "metric", "difference_definition", "mean", "sample_sd",
        "ci_lower", "ci_upper", "minimum", "maximum", "positive_fraction", "n_seeds",
    ]
    lines.append(markdown_table(paired[[column for column in paired_columns if column in paired.columns]], max_rows=100))
    lines.append("")

    lines.append("## B_confirm seed-level key metrics")
    lines.append("")
    lines.append(markdown_table(seedwise_b, max_rows=30))
    lines.append("")

    lines.append("## Feature-matched nonlinear-capacity audit")
    lines.append("")
    lines.append(markdown_table(nonlinear, max_rows=12))
    lines.append("")

    lines.append("## A_train-selected leading-PC assignments")
    lines.append("")
    lines.append(markdown_table(pc_assignments, max_rows=60))
    lines.append("")

    lines.append("## Validation-confirmation stability")
    lines.append("")
    split_key = split_deltas[split_deltas["metric"].isin(KEY_METRICS)] if not split_deltas.empty else split_deltas
    lines.append(markdown_table(split_key, max_rows=100))
    lines.append("")

    lines.append("## Source inventory")
    lines.append("")
    if sources.empty:
        lines.append("_No source inventory was found._")
    else:
        lines.append(markdown_table(sources[["seed", "path", "sha256", "bytes"]], max_rows=80))
    lines.append("")
    lines.append("Complete first-16 PC correlations, all per-seed split metrics, sample hashes and machine-readable manifests are retained with the report.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the objective-control hidden-geometry numerical report.")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    manifest_path = result_root / "metadata" / "objective_control_hidden_geometry_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = load_json(manifest_path)
    completed_seeds = [int(value) for value in manifest.get("completed_seeds", [])]
    if not completed_seeds:
        raise RuntimeError("Analysis manifest contains no completed seeds.")

    metrics, metrics_path = read_table(result_root / "tables" / "geometry_metrics_all_seeds")
    summary, summary_path = read_table(result_root / "tables" / "six_seed_geometry_summary")
    paired, paired_path = read_table(result_root / "tables" / "same_seed_objective_contrasts")
    pc, pc_path = read_table(result_root / "tables" / "pc_macro_correlations_all_seeds")
    sample, sample_path = read_table(result_root / "tables" / "shared_sample_audit_all_seeds")
    contracts, contracts_path = read_table(result_root / "tables" / "model_objective_contracts_all_seeds")
    identity, identity_path = read_table(result_root / "tables" / "cross_seed_identity_gates")
    try:
        reconstruction, reconstruction_path = read_table(result_root / "tables" / "formal_geometry_reconstruction_audit")
    except FileNotFoundError:
        reconstruction = pd.DataFrame()
        reconstruction_path = None

    require_columns(metrics, ["seed", "model", "model_display", "split", "cca_corr_1", "cca_corr_2"], "geometry metrics")
    require_columns(summary, ["model", "model_display", "split", "metric", "mean", "n_seeds"], "six-seed summary")
    require_columns(paired, ["split", "metric", "control", "mean", "n_seeds"], "paired contrasts")
    require_columns(sample, ["seed", "split", "sample_index_hash", "sample_key_hash"], "sample audit")
    require_columns(contracts, ["seed", "model", "checkpoint_sha256"], "model contracts")
    require_columns(identity, ["split", "gate", "passed"], "identity gates")

    seed_manifests: Dict[int, Dict[str, Any]] = {}
    for seed in completed_seeds:
        path = result_root / "seeds" / f"seed{seed}" / "metadata" / "seed_geometry_manifest.json"
        if not path.exists():
            raise FileNotFoundError(path)
        seed_manifests[seed] = load_json(path)

    quality = build_quality_gates(
        manifest,
        metrics,
        sample,
        contracts,
        identity,
        reconstruction,
        seed_manifests,
    )
    if not bool(quality["passed"].all()):
        failed = quality.loc[~quality["passed"], "gate"].astype(str).tolist()
        raise RuntimeError(f"Numerical-report quality gates failed: {failed}")

    summary_b = compact_six_seed_summary(summary, "B_confirm")
    summary_v = compact_six_seed_summary(summary, "A_val")
    seedwise_b = seedwise_key_metrics(metrics, "B_confirm")
    split_deltas = validation_confirmation_deltas(metrics)
    assignments = pc_assignment_table(metrics)
    nonlinear = nonlinear_capacity_table(metrics)
    reconstruction_compact = reconstruction_summary(reconstruction)
    sources = source_inventory(result_root, completed_seeds)

    output_root = (args.output_root or (result_root / "numeric_report")).resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Report output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    output_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    report_tables = {
        "quality_gates": write_table(quality, table_root / "quality_gates"),
        "B_confirm_six_seed_summary": write_table(summary_b, table_root / "B_confirm_six_seed_summary"),
        "A_val_six_seed_summary": write_table(summary_v, table_root / "A_val_six_seed_summary"),
        "same_seed_objective_contrasts": write_table(paired, table_root / "same_seed_objective_contrasts"),
        "B_confirm_seedwise_key_metrics": write_table(seedwise_b, table_root / "B_confirm_seedwise_key_metrics"),
        "validation_confirmation_deltas": write_table(split_deltas, table_root / "validation_confirmation_deltas"),
        "pc_assignment_summary": write_table(assignments, table_root / "pc_assignment_summary"),
        "nonlinear_capacity_audit": write_table(nonlinear, table_root / "nonlinear_capacity_audit"),
        "formal_reconstruction_audit": write_table(reconstruction_compact, table_root / "formal_reconstruction_audit"),
        "shared_sample_audit": write_table(sample, table_root / "shared_sample_audit"),
        "model_objective_contracts": write_table(contracts, table_root / "model_objective_contracts"),
        "source_inventory": write_table(sources, table_root / "source_inventory"),
        "all_geometry_metrics": write_table(metrics, table_root / "all_geometry_metrics"),
        "all_pc_correlations": write_table(pc, table_root / "all_pc_correlations"),
    }

    markdown = build_markdown(
        manifest=manifest,
        quality=quality,
        sample=sample,
        contracts=contracts,
        reconstruction=reconstruction_compact,
        summary_b=summary_b,
        summary_v=summary_v,
        paired=paired,
        seedwise_b=seedwise_b,
        nonlinear=nonlinear,
        pc_assignments=assignments,
        split_deltas=split_deltas,
        sources=sources,
    )
    markdown_path = output_root / "objective_control_hidden_geometry_numeric_report.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    input_files = [
        manifest_path,
        metrics_path,
        summary_path,
        paired_path,
        pc_path,
        sample_path,
        contracts_path,
        identity_path,
    ]
    if reconstruction_path is not None:
        input_files.append(reconstruction_path)
    input_inventory = [
        {
            "path": str(path.resolve()),
            "sha256": file_sha256(path.resolve()),
            "bytes": int(path.stat().st_size),
        }
        for path in input_files
    ]
    report_payload = {
        "analysis_manifest": manifest,
        "quality_gates": quality.to_dict(orient="records"),
        "B_confirm_six_seed_summary": summary_b.to_dict(orient="records"),
        "A_val_six_seed_summary": summary_v.to_dict(orient="records"),
        "same_seed_objective_contrasts": paired.to_dict(orient="records"),
        "B_confirm_seedwise_key_metrics": seedwise_b.to_dict(orient="records"),
        "validation_confirmation_deltas": split_deltas.to_dict(orient="records"),
        "pc_assignment_summary": assignments.to_dict(orient="records"),
        "nonlinear_capacity_audit": nonlinear.to_dict(orient="records"),
        "formal_reconstruction_audit": reconstruction_compact.to_dict(orient="records"),
        "input_inventory": input_inventory,
        "report_tables": {name: str(path.resolve()) for name, path in report_tables.items()},
    }
    json_path = output_root / "objective_control_hidden_geometry_numeric_report.json"
    save_json(report_payload, json_path)

    output_inventory = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            output_inventory.append({
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(output_root)),
                "sha256": file_sha256(path),
                "bytes": int(path.stat().st_size),
            })
    report_manifest = {
        "analysis": "hidden-state geometry in objective-control representations",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_result_root": str(result_root),
        "source_analysis_manifest": str(manifest_path),
        "source_analysis_manifest_sha256": file_sha256(manifest_path),
        "report_markdown": str(markdown_path),
        "report_json": str(json_path),
        "input_inventory": input_inventory,
        "output_inventory": output_inventory,
        "scientific_recomputation": False,
        "model_selection_or_update": False,
    }
    report_manifest_path = metadata_root / "objective_control_hidden_geometry_numeric_report_manifest.json"
    save_json(report_manifest, report_manifest_path)
    checksum_path = metadata_root / "objective_control_hidden_geometry_numeric_report_manifest.sha256.json"
    save_json({
        "manifest": str(report_manifest_path),
        "manifest_sha256": file_sha256(report_manifest_path),
    }, checksum_path)
    print(f"[objective geometry report] wrote {markdown_path}")


if __name__ == "__main__":
    main()
