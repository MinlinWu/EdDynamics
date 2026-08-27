#!/usr/bin/env python3
from __future__ import annotations

"""Extract non-duplicative Event-SSL supplementary diagnostics from frozen outputs."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

PRIMARY_COORDINATES = ["M", "Psi"]
MACROSTATE_K = 6
HELD_OUT_SPLITS = ("A_val", "B_confirm")
REPRESENTATIONS = ("full_hidden", "macro_only", "residual_hidden")
REPRESENTATION_LABELS = {
    "full_hidden": "Full hidden state",
    "macro_only": "Two-coordinate bottleneck",
    "residual_hidden": "Residual hidden representation",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def save_json(value: Mapping[str, Any], path: Path) -> None:
    atomic_write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        path,
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_path(base: Path) -> Path:
    candidates = [base] if base.suffix else [
        base.with_suffix(".parquet"),
        base.with_suffix(".csv.gz"),
        base.with_suffix(".csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find table for {base}")


def read_table(base: Path) -> pd.DataFrame:
    path = table_path(base)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def write_table(frame: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        parquet = base.with_suffix(".parquet")
        temporary = parquet.with_name(parquet.name + ".tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, parquet)
        frame.to_csv(base.with_suffix(".csv"), index=False)
        return parquet
    except Exception:
        if temporary.exists():
            temporary.unlink()
        compressed = base.with_suffix(".csv.gz")
        temporary = compressed.with_name(compressed.name + ".tmp")
        frame.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, compressed)
        return compressed


def finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} is missing columns: {missing}")


def fixed_partition_signature(contract: Mapping[str, Any], label: str) -> tuple[str, str]:
    if not contract:
        raise RuntimeError(f"{label} has no fixed-partition contract")
    if int(contract.get("macrostate_k", contract.get("k", -1))) != MACROSTATE_K:
        raise RuntimeError(f"{label} does not use fixed K=6")
    if contract.get("verified") is False:
        raise RuntimeError(f"{label} fixed-partition audit is not verified")
    if bool(contract.get("kmeans_refit", False)):
        raise RuntimeError(f"{label} refitted K-means")
    if bool(contract.get("macrostate_k_selected", False)):
        raise RuntimeError(f"{label} reselected K")
    fit_split = str(contract.get("fit_split", contract.get("training_split", "")))
    if fit_split and fit_split != "A_train":
        raise RuntimeError(f"{label} partition was not fitted on A_train")
    metadata_hash = str(contract.get("metadata_sha256", ""))
    centres_hash = str(contract.get("centers_sha256", contract.get("centres_sha256", "")))
    if not metadata_hash or not centres_hash:
        raise RuntimeError(f"{label} does not record the frozen partition hashes")
    return metadata_hash, centres_hash


def validate_training_manifest(manifest: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    if manifest.get("primary_coordinates") != PRIMARY_COORDINATES:
        raise RuntimeError(f"{label} does not use the M-Psi state")
    if str(manifest.get("train_split", "A_train")) != "A_train":
        raise RuntimeError(f"{label} was not fitted on A_train")
    return dict(manifest.get("stage1_fixed_k6_contract", {}))


def validate_evaluation_manifest(manifest: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    if manifest.get("primary_coordinates") != PRIMARY_COORDINATES:
        raise RuntimeError(f"{label} does not use the M-Psi state")
    return dict(manifest.get("fixed_k6_partition", {}))


def validate_cross_model_manifest(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    primary = manifest.get("primary_macrostate", manifest.get("primary_coordinates"))
    if primary != PRIMARY_COORDINATES:
        raise RuntimeError("Cross-model output does not use the M-Psi state")
    if str(manifest.get("split", manifest.get("split_internal", ""))) != "B_confirm":
        raise RuntimeError("Cross-model supplementary diagnostics must use B_confirm")
    family = str(manifest.get("minimal_mechanism_family", "offset_dual_channel"))
    if family != "offset_dual_channel":
        raise RuntimeError(f"Unexpected minimal-mechanism family: {family}")
    model_kind = str(manifest.get("event_ssl_model_kind", "predictive_state"))
    if model_kind != "predictive_state":
        raise RuntimeError(f"Unexpected Event-SSL model kind: {model_kind}")
    boundary = manifest.get("analysis_boundary", {})
    required_false = (
        "model_retraining",
        "cross_model_fitting",
        "mechanism_refit_to_event_ssl",
        "event_ssl_refit_to_mechanism",
        "macrostate_redefinition",
        "kmeans_refit",
        "macrostate_k_selected",
        "confirmation_data_used_for_update",
    )
    if not isinstance(boundary, Mapping):
        raise RuntimeError("Cross-model manifest has no formal analysis boundary")
    missing = [name for name in required_false if name not in boundary]
    failed = [name for name in required_false if bool(boundary.get(name, False))]
    if missing or failed:
        raise RuntimeError(f"Cross-model analysis-boundary failure: missing={missing}, failed={failed}")
    return dict(manifest.get("fixed_k6_partition", manifest.get("macro_partition_audit", {})))


def row_for(frame: pd.DataFrame, split: str, representation: str | None = None) -> pd.Series:
    selected = frame[frame["split"].astype(str) == split]
    if representation is not None:
        selected = selected[selected["representation"].astype(str) == representation]
    if selected.empty:
        raise RuntimeError(f"Missing metrics for split={split}, representation={representation}")
    return selected.iloc[0]


def build_representation_diagnostics(
    geometry_training: Mapping[str, Any],
    geometry_metrics: pd.DataFrame,
    macro_metrics: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(geometry_metrics, ["split", "representation", "twonn_dimension"], "geometry metrics")
    require_columns(
        macro_metrics,
        [
            "split",
            "representation",
            "representation_nmi_with_empirical_macrostate",
            "representation_ari_with_empirical_macrostate",
        ],
        "macrostate-sufficiency metrics",
    )

    rows: list[dict[str, Any]] = [
        {
            "category": "Hidden-state dimensionality",
            "diagnostic": "Participation ratio",
            "A_train": finite_float(geometry_training.get("participation_ratio")),
            "A_val": np.nan,
            "B_confirm": np.nan,
        },
        {
            "category": "Hidden-state dimensionality",
            "diagnostic": "Effective rank",
            "A_train": finite_float(geometry_training.get("effective_rank")),
            "A_val": np.nan,
            "B_confirm": np.nan,
        },
        {
            "category": "Hidden-state dimensionality",
            "diagnostic": "TwoNN intrinsic dimension",
            "A_train": finite_float(geometry_training.get("twonn_intrinsic_dimension_estimate")),
            "A_val": finite_float(row_for(geometry_metrics, "A_val", "model_readout")["twonn_dimension"]),
            "B_confirm": finite_float(row_for(geometry_metrics, "B_confirm", "model_readout")["twonn_dimension"]),
        },
    ]

    for representation in REPRESENTATIONS:
        values = {
            split: row_for(macro_metrics, split, representation)
            for split in HELD_OUT_SPLITS
        }
        for column, label in (
            ("representation_nmi_with_empirical_macrostate", "NMI with empirical mesostates"),
            ("representation_ari_with_empirical_macrostate", "ARI with empirical mesostates"),
        ):
            rows.append(
                {
                    "category": "Diagnostic representation clustering",
                    "diagnostic": f"{label}: {REPRESENTATION_LABELS[representation]}",
                    "A_train": np.nan,
                    "A_val": finite_float(values["A_val"][column]),
                    "B_confirm": finite_float(values["B_confirm"][column]),
                }
            )
    return pd.DataFrame(rows)


def format_value(value: Any) -> str:
    number = finite_float(value)
    if not np.isfinite(number):
        return "--"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.4g}"


def compact_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for split in ("A_train", "A_val", "B_confirm"):
        output[split] = output[split].map(format_value)
    return output


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def write_latex_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Hidden-representation diagnostics not displayed in the main figures. Participation ratio and effective rank summarize the training-set PCA spectrum, while TwoNN is estimated separately by split. NMI and ARI compare diagnostic six-cluster partitions of each representation with the frozen empirical mesostates; these diagnostic clusters do not define transition states.}",
        r"\label{tab:event_ssl_representation_diagnostics}",
        r"\begin{tabular}{p{0.56\textwidth}ccc}",
        r"\toprule",
        r"Diagnostic & Training & Validation & Confirmation \\",
        r"\midrule",
    ]
    previous_category: str | None = None
    for _, row in frame.iterrows():
        category = str(row["category"])
        if previous_category is not None and category != previous_category:
            lines.append(r"\addlinespace")
        lines.append(
            f"{latex_escape(row['diagnostic'])} & {latex_escape(row['A_train'])} & "
            f"{latex_escape(row['A_val'])} & {latex_escape(row['B_confirm'])}" + r" \\"
        )
        previous_category = category
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    atomic_write_text("\n".join(lines), path)


def source_audit(paths: Mapping[str, Path]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": name,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        ]
    )


def build_quality_gates(
    signatures: Sequence[tuple[str, str]],
    diagnostics: pd.DataFrame,
    residual_field: pd.DataFrame,
    sources: pd.DataFrame,
) -> pd.DataFrame:
    held_out = pd.to_numeric(
        diagnostics[["A_val", "B_confirm"]].stack(), errors="coerce"
    )
    training_values = pd.to_numeric(
        diagnostics.loc[
            diagnostics["diagnostic"].isin(
                ["Participation ratio", "Effective rank", "TwoNN intrinsic dimension"]
            ),
            "A_train",
        ],
        errors="coerce",
    )
    residual_columns = (
        "residual_vector_corr",
        "residual_speed_corr",
        "occupancy_weighted_residual_cosine",
    )
    residual_values = residual_field.loc[:, residual_columns].apply(pd.to_numeric, errors="coerce")
    checks = [
        ("single_frozen_partition", len(set(signatures)) == 1, f"signatures={len(set(signatures))}"),
        ("expected_diagnostic_rows", len(diagnostics) == 9, f"rows={len(diagnostics)}"),
        ("training_dimensionality_complete", bool(training_values.notna().all()), f"values={len(training_values)}"),
        ("held_out_diagnostics_complete", bool(held_out.notna().all()), f"values={len(held_out)}"),
        ("cross_model_residual_complete", bool(residual_values.notna().all().all()), f"rows={len(residual_field)}"),
        ("source_ledger_complete", bool(sources["sha256"].astype(str).str.len().eq(64).all()), f"sources={len(sources)}"),
        ("random_seed_outputs_not_consumed", True, "no random-seed input is accepted"),
    ]
    return pd.DataFrame(
        [{"quality_gate": name, "passed": passed, "detail": detail} for name, passed, detail in checks]
    )


def markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_csv(index=False)


def build_report(
    diagnostics: pd.DataFrame,
    residual_field: pd.DataFrame,
    quality: pd.DataFrame,
) -> str:
    residual = residual_field.iloc[0]
    lines = [
        "# Event-SSL supplementary numerical report",
        "",
        "## Scope",
        "",
        "The report reads frozen representation and cross-model outputs only. It does not train a model, refit a probe, recompute predictions, redefine the primary state, or refit the fixed six-state partition.",
        "",
        "## Hidden-representation diagnostics",
        "",
        markdown_table(compact_table(diagnostics)),
        "",
        "## Residual-field comparison",
        "",
        f"Residual-field vector correlation: {finite_float(residual['residual_vector_corr']):.4g}",
        f"Residual-speed correlation: {finite_float(residual['residual_speed_corr']):.4g}",
        f"Occupancy-weighted residual cosine: {finite_float(residual['occupancy_weighted_residual_cosine']):.4g}",
        "",
        "These values compare the errors remaining after both frozen closures are referenced to the same empirical field. They are descriptive and do not imply cross-model fitting.",
        "",
        "## Scientific quality gates",
        "",
        markdown_table(quality),
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract non-duplicative Event-SSL supplementary diagnostics from frozen outputs."
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4"),
    )
    parser.add_argument("--macro-root", type=Path, default=None)
    parser.add_argument("--geometry-root", type=Path, default=None)
    parser.add_argument("--cross-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--no-strict", action="store_false", dest="strict")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = args.output_base.resolve()
    macro_root = (args.macro_root or base / "stage5_macro_sufficiency").resolve()
    geometry_root = (args.geometry_root or base / "stage5_representation_geometry").resolve()
    cross_root = (args.cross_root or base / "cross_stage_mechanism_event_ssl_comparison").resolve()
    output_root = args.output_root.resolve()

    paths = {
        "event_ssl_input_manifest": base / "stage4_event_ssl" / "prepared_inputs" / "metadata" / "stage4_input_manifest.json",
        "macro_training_manifest": macro_root / "metadata" / "stage5_macro_sufficiency_training_manifest.json",
        "macro_evaluation_manifest": macro_root / "evaluation" / "metadata" / "stage5_macro_sufficiency_evaluation_manifest.json",
        "macro_metrics": table_path(macro_root / "evaluation" / "tables" / "stage5_macro_sufficiency_metrics_all_splits"),
        "geometry_training_manifest": geometry_root / "metadata" / "stage5_representation_geometry_training_manifest.json",
        "geometry_evaluation_manifest": geometry_root / "evaluation" / "metadata" / "stage5_representation_geometry_evaluation_manifest.json",
        "geometry_metrics": table_path(geometry_root / "evaluation" / "tables" / "stage5_representation_geometry_metrics_all_splits"),
        "cross_model_manifest": cross_root / "metadata" / "mechanism_event_ssl_comparison_manifest.json",
        "cross_model_residual_field": table_path(cross_root / "tables" / "mechanism_event_ssl_residual_field"),
    }

    input_manifest = load_json(paths["event_ssl_input_manifest"])
    macro_training = load_json(paths["macro_training_manifest"])
    macro_evaluation = load_json(paths["macro_evaluation_manifest"])
    geometry_training = load_json(paths["geometry_training_manifest"])
    geometry_evaluation = load_json(paths["geometry_evaluation_manifest"])
    cross_manifest = load_json(paths["cross_model_manifest"])

    if input_manifest.get("primary_coordinates") != PRIMARY_COORDINATES:
        raise RuntimeError("Event-SSL input manifest does not use the M-Psi state")

    contracts = [
        ("Event-SSL input preparation", dict(input_manifest.get("stage1_fixed_k6_contract", {}))),
        ("Macrostate-sufficiency training", validate_training_manifest(macro_training, "Macrostate-sufficiency training")),
        ("Macrostate-sufficiency evaluation", validate_evaluation_manifest(macro_evaluation, "Macrostate-sufficiency evaluation")),
        ("Representation-geometry training", validate_training_manifest(geometry_training, "Representation-geometry training")),
        ("Representation-geometry evaluation", validate_evaluation_manifest(geometry_evaluation, "Representation-geometry evaluation")),
        ("Cross-model comparison", validate_cross_model_manifest(cross_manifest)),
    ]
    signatures = [fixed_partition_signature(contract, label) for label, contract in contracts]
    if len(set(signatures)) != 1:
        raise RuntimeError("The supplementary inputs do not share one frozen Stage-1 partition")

    clustering_contract = dict(macro_evaluation.get("representation_clustering", {}))
    if clustering_contract and bool(clustering_contract.get("used_for_transition_metrics", True)):
        raise RuntimeError("Diagnostic representation clusters were used as transition states")

    macro_metrics = read_table(paths["macro_metrics"])
    geometry_metrics = read_table(paths["geometry_metrics"])
    residual_field = read_table(paths["cross_model_residual_field"])
    require_columns(
        residual_field,
        ["residual_vector_corr", "residual_speed_corr", "occupancy_weighted_residual_cosine"],
        "cross-model residual-field table",
    )
    if residual_field.empty:
        raise RuntimeError("Cross-model residual-field table is empty")

    diagnostics = build_representation_diagnostics(
        geometry_training,
        geometry_metrics,
        macro_metrics,
    )
    compact = compact_table(diagnostics)
    sources = source_audit(paths)
    quality = build_quality_gates(signatures, diagnostics, residual_field, sources)
    if args.strict and not bool(quality["passed"].all()):
        failed = quality.loc[~quality["passed"], "quality_gate"].astype(str).tolist()
        raise RuntimeError(f"Supplementary quality gates failed: {failed}")

    table_root = output_root / "tables"
    latex_root = output_root / "latex"
    report_root = output_root / "reports"
    metadata_root = output_root / "metadata"
    for directory in (table_root, latex_root, report_root, metadata_root):
        directory.mkdir(parents=True, exist_ok=True)

    outputs = {
        "representation_diagnostics": write_table(
            diagnostics,
            table_root / "supplementary_event_ssl_representation_diagnostics",
        ),
        "representation_diagnostics_compact": write_table(
            compact,
            table_root / "supplementary_event_ssl_representation_diagnostics_compact",
        ),
        "cross_model_residual_field": write_table(
            residual_field,
            table_root / "supplementary_event_ssl_cross_model_residual_field",
        ),
        "source_audit": write_table(
            sources,
            table_root / "supplementary_event_ssl_source_audit",
        ),
        "quality_gates": write_table(
            quality,
            table_root / "supplementary_event_ssl_quality_gates",
        ),
    }

    latex_path = latex_root / "supplementary_event_ssl_representation_diagnostics.tex"
    write_latex_table(compact, latex_path)
    report_path = report_root / "event_ssl_supplementary_numerical_report.md"
    atomic_write_text(build_report(diagnostics, residual_field, quality), report_path)

    manifest = {
        "script": Path(__file__).name,
        "primary_coordinates": PRIMARY_COORDINATES,
        "macrostate_k": MACROSTATE_K,
        "frozen_partition_metadata_sha256": signatures[0][0],
        "frozen_partition_centers_sha256": signatures[0][1],
        "random_seed_outputs_consumed": False,
        "typeset_outputs": [str(latex_path.resolve())],
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
        "report": str(report_path.resolve()),
        "quality_gates": quality.to_dict(orient="records"),
    }
    save_json(manifest, metadata_root / "event_ssl_supplementary_manifest.json")
    print(f"Supplementary Event-SSL diagnostics written to {output_root}")


if __name__ == "__main__":
    main()
