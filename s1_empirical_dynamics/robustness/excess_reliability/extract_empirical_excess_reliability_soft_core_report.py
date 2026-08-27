#!/usr/bin/env python3
"""Extract the empirical excess-field reliability and soft-core numerical report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_ANALYSIS_ROOT = Path(
    "/data/datasets/KT4/outputs_KT4/empirical_excess_reliability_soft_core"
)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(key): json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.integer):
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


def save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
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
    raise FileNotFoundError(f"Could not find table for {base_or_path}")


def read_table(base_or_path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    path = find_table(base_or_path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    return pd.read_csv(
        path,
        usecols=list(columns) if columns is not None else None,
        low_memory=False,
    )


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
        csv_temporary = csv_path.with_name(csv_path.name + ".tmp")
        frame.to_csv(csv_temporary, index=False, compression="gzip")
        os.replace(csv_temporary, csv_path)
        return csv_path


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing required columns: {missing}")


def finite_summary(values: Sequence[Any]) -> Dict[str, Any]:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "min": None,
            "2p5": None,
            "50": None,
            "97p5": None,
            "max": None,
        }
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "2p5": float(np.quantile(array, 0.025)),
        "50": float(np.quantile(array, 0.50)),
        "97p5": float(np.quantile(array, 0.975)),
        "max": float(np.max(array)),
    }


def bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(float) != 0
    return values.fillna("").astype(str).str.lower().isin({"true", "yes", "1"})


def markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.{digits}g}"
            )
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )
    headers = [str(column).replace("|", "\\|") for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        values = [
            str(value).replace("|", "\\|").replace("\n", " ")
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def table_inventory(table_root: Path) -> pd.DataFrame:
    rows = []
    seen = set()
    for path in sorted(table_root.iterdir()):
        if not path.is_file():
            continue
        key = path.name
        if key in seen:
            continue
        seen.add(key)
        try:
            rows_count = int(len(read_table(path)))
            columns = ";".join(read_table(path).columns)
        except Exception:
            rows_count = None
            columns = None
        rows.append(
            {
                "path": str(path.resolve()),
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
                "rows": rows_count,
                "columns": columns,
            }
        )
    return pd.DataFrame(rows)


def array_inventory(array_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(array_root.glob("*.npz")):
        with np.load(path) as archive:
            members = [
                {
                    "name": name,
                    "shape": list(np.asarray(archive[name]).shape),
                    "dtype": str(np.asarray(archive[name]).dtype),
                }
                for name in archive.files
            ]
        rows.append(
            {
                "path": str(path.resolve()),
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
                "members": members,
            }
        )
    return pd.DataFrame(rows)


def build_report(analysis_root: Path) -> Dict[str, Any]:
    metadata_root = analysis_root / "metadata"
    table_root = analysis_root / "tables"
    array_root = analysis_root / "arrays"
    manifest_path = metadata_root / "empirical_excess_reliability_soft_core_manifest.json"
    checksum_path = metadata_root / "empirical_excess_reliability_soft_core_manifest.sha256.json"
    manifest = load_json(manifest_path)
    checksum = load_json(checksum_path)
    actual_manifest_sha = file_sha256(manifest_path)
    if str(checksum.get("manifest_sha256", "")) != actual_manifest_sha:
        raise RuntimeError("Analysis manifest checksum verification failed.")

    points = read_table(table_root / "formal_finite_and_exact_excess_field_points")
    partition_balance = read_table(table_root / "split_half_partition_balance")
    activity_balance = read_table(table_root / "split_half_activity_bin_balance")
    matching_quality = read_table(table_root / "split_half_matching_quality")
    reliability = read_table(table_root / "split_half_reliability_replicates")
    reliability_summary = read_table(table_root / "split_half_reliability_summary")
    attenuation = read_table(table_root / "attenuation_benchmark_replicates")
    attenuation_summary = read_table(table_root / "attenuation_benchmark_summary")
    soft_replicates = read_table(table_root / "soft_core_replicates")
    soft_summary = read_table(table_root / "soft_core_summary")
    soft_cells = read_table(table_root / "soft_core_cell_selection_frequency")
    soft_thresholds = read_table(table_root / "soft_core_threshold_summary")
    source_hashes = read_table(table_root / "source_file_hashes")

    require_columns(
        points,
        [
            "estimator",
            "support",
            "supported_cells",
            "vector_correlation",
            "speed_correlation",
            "M_component_correlation",
            "Psi_component_correlation",
            "weighted_local_cosine",
        ],
        "formal point table",
    )
    require_columns(
        reliability,
        [
            "split",
            "partition",
            "formal_target_cells",
            "cell_coverage_fraction",
            "occupancy_mass_coverage_fraction",
            "coverage_valid",
            "vector_correlation",
            "vector_spearman_brown",
        ],
        "split-half replicate table",
    )
    require_columns(
        reliability_summary,
        [
            "split",
            "partitions",
            "coverage_valid_partitions",
            "coverage_valid_fraction",
            "formal_target_cells",
        ],
        "split-half reliability summary",
    )
    require_columns(
        attenuation,
        [
            "partition",
            "formal_target_cells",
            "supported_cells",
            "cell_coverage_fraction",
            "coverage_valid",
            "vector_attenuation_benchmark",
            "vector_observed_exact_cross_split_agreement",
        ],
        "attenuation replicate table",
    )
    require_columns(
        attenuation_summary,
        [
            "metric",
            "partitions",
            "coverage_valid_partitions",
            "benchmark_defined_partitions",
            "minimum_defined_partitions",
            "benchmark_reporting_eligible",
            "formal_target_cells",
            "attenuation_benchmark_50",
            "agreement_to_benchmark_ratio_50",
        ],
        "attenuation summary",
    )
    require_columns(
        soft_summary,
        [
            "replicates",
            "training_region_detection_rate",
            "validation_matching_region_availability_rate",
            "paired_region_availability_rate",
            "soft_jaccard_unconditional",
            "soft_jaccard_paired_available",
            "soft_jaccard_paired_qualified",
        ],
        "soft-core summary",
    )
    require_columns(
        soft_cells,
        [
            "training_core_selection_frequency_unconditional",
            "validation_matching_core_selection_frequency_unconditional",
            "training_core_selection_frequency_paired_available",
            "validation_core_selection_frequency_paired_available",
            "training_core_selection_frequency_paired_qualified",
            "validation_core_selection_frequency_paired_qualified",
        ],
        "soft-core cell table",
    )

    contracts = manifest.get("excess_reliability_contract", {})
    soft_contract = manifest.get("soft_core_contract", {})
    minimum_valid_fraction = float(
        contracts.get("minimum_valid_partition_fraction", np.nan)
    )
    expected_partitions = int(contracts.get("partitions", -1))
    expected_replicates = int(soft_contract.get("replicates", -1))
    if not np.isfinite(minimum_valid_fraction) or not 0.0 < minimum_valid_fraction <= 1.0:
        raise RuntimeError("The analysis manifest has an invalid partition-coverage contract.")
    if expected_partitions <= 0 or expected_replicates <= 0:
        raise RuntimeError("Analysis manifest is missing replicate counts.")
    minimum_valid_partitions = int(np.ceil(expected_partitions * minimum_valid_fraction))
    observed_valid_counts = pd.to_numeric(
        attenuation_summary["coverage_valid_partitions"], errors="coerce"
    )
    if observed_valid_counts.isna().any() or bool(
        (observed_valid_counts < minimum_valid_partitions).any()
    ):
        raise RuntimeError(
            "The attenuation summary does not meet the prespecified valid-partition gate."
        )
    eligible = bool_series(attenuation_summary, "benchmark_reporting_eligible")
    if bool(
        (
            pd.to_numeric(
                attenuation_summary.loc[eligible, "benchmark_defined_partitions"],
                errors="coerce",
            )
            < pd.to_numeric(
                attenuation_summary.loc[eligible, "minimum_defined_partitions"],
                errors="coerce",
            )
        ).any()
    ):
        raise RuntimeError("An attenuation summary is marked eligible without enough defined partitions.")
    for column in ("attenuation_benchmark_50", "agreement_to_benchmark_ratio_50"):
        if column in attenuation_summary.columns and bool(
            pd.to_numeric(
                attenuation_summary.loc[~eligible, column], errors="coerce"
            ).notna().any()
        ):
            raise RuntimeError(
                "An ineligible attenuation benchmark contains a selectively summarized point estimate."
            )

    if len(reliability) != 2 * expected_partitions:
        raise RuntimeError("Unexpected split-half reliability row count.")
    if len(attenuation) != expected_partitions:
        raise RuntimeError("Unexpected attenuation row count.")
    if len(soft_replicates) != expected_replicates:
        raise RuntimeError("Unexpected soft-core replicate row count.")
    if len(partition_balance) != 4 * expected_partitions:
        raise RuntimeError("Unexpected complementary-half balance row count.")
    if len(matching_quality) != 4 * expected_partitions:
        raise RuntimeError("Unexpected half-null matching row count.")

    gates = dict(manifest.get("quality_gates", {}))
    failed_gates = [
        key
        for key, value in gates.items()
        if key.endswith("passed") and not bool(value)
    ]
    if failed_gates:
        raise RuntimeError(f"Analysis quality gates failed: {failed_gates}")

    source_rows = []
    for row in source_hashes.to_dict("records"):
        path = Path(str(row["path"]))
        current = file_sha256(path)
        recorded = str(row["sha256"])
        source_rows.append(
            {
                "path": str(path),
                "bytes": int(row["bytes"]),
                "recorded_sha256": recorded,
                "current_sha256": current,
                "checksum_matches": current == recorded,
            }
        )
    if not all(row["checksum_matches"] for row in source_rows):
        raise RuntimeError("A recorded source file changed after the analysis.")

    balance_summary = []
    for (split, half), group in partition_balance.groupby(["split", "half"], sort=False):
        balance_summary.append(
            {
                "split": str(split),
                "half": int(half),
                "users": finite_summary(group["users"]),
                "panel_rows": finite_summary(group["panel_rows"]),
                "valid_drift_rows": finite_summary(group["valid_drift_rows"]),
                "positive_weight_cells": finite_summary(
                    group["drift_cells_with_positive_weight"]
                ),
            }
        )

    matching_summary = []
    moment_columns = [
        column
        for column in matching_quality.columns
        if column.startswith("mean_") and column.endswith("preservation_error")
    ]
    for split, group in matching_quality.groupby("split", sort=False):
        matching_summary.append(
            {
                "split": str(split),
                "half_estimates": int(len(group)),
                "weak_fallback_fraction": finite_summary(
                    group["weak_fallback_fraction_of_assigned"]
                ),
                "within_user_fine_fraction": finite_summary(
                    group["within_user_fine_fraction_of_assigned"]
                ),
                "within_user_coarse_fraction": finite_summary(
                    group["within_user_coarse_fraction_of_assigned"]
                ),
                "maximum_exact_moment_error": float(
                    group[moment_columns].max().max()
                )
                if moment_columns
                else None,
            }
        )

    activity_bin_audit = {
        "rows": int(len(activity_balance)),
        "activity_bins": int(activity_balance["activity_bin"].nunique()),
        "maximum_absolute_user_imbalance": int(
            np.max(
                np.abs(
                    pd.to_numeric(activity_balance["half_0_users"], errors="raise")
                    - pd.to_numeric(activity_balance["half_1_users"], errors="raise")
                )
            )
        ),
    }

    soft_replicate_audit = {
        "rows": int(len(soft_replicates)),
        "training_region_available": int(
            bool_series(soft_replicates, "training_region_available").sum()
        ),
        "validation_matching_region_available": int(
            bool_series(soft_replicates, "validation_matching_region_available").sum()
        ),
        "paired_region_available": int(
            bool_series(soft_replicates, "paired_region_available").sum()
        ),
        "paired_dynamically_qualified": int(
            bool_series(soft_replicates, "paired_dynamically_qualified").sum()
        ),
        "training_primary_is_best_formal_overlap_candidate": int(
            bool_series(
                soft_replicates,
                "training_primary_is_best_formal_overlap_candidate",
            ).sum()
        ),
        "hard_jaccard": finite_summary(
            soft_replicates.get("train_validation_mask_jaccard", pd.Series(dtype=float))
        ),
        "hard_overlap": finite_summary(
            soft_replicates.get("train_validation_mask_overlap", pd.Series(dtype=float))
        ),
        "center_distance": finite_summary(
            soft_replicates.get("train_validation_center_distance", pd.Series(dtype=float))
        ),
        "training_primary_vs_formal_jaccard": finite_summary(
            soft_replicates.get(
                "training_primary_vs_formal_jaccard", pd.Series(dtype=float)
            )
        ),
        "validation_matching_vs_formal_jaccard": finite_summary(
            soft_replicates.get(
                "validation_matching_vs_formal_jaccard", pd.Series(dtype=float)
            )
        ),
    }

    frequency_columns = [
        column
        for column in soft_cells.columns
        if "selection_frequency" in column
    ]
    frequency_summary = {
        column: {
            "distribution": finite_summary(soft_cells[column]),
            "cells_ge_0p25": int(
                np.sum(pd.to_numeric(soft_cells[column], errors="coerce") >= 0.25)
            ),
            "cells_ge_0p50": int(
                np.sum(pd.to_numeric(soft_cells[column], errors="coerce") >= 0.50)
            ),
            "cells_ge_0p75": int(
                np.sum(pd.to_numeric(soft_cells[column], errors="coerce") >= 0.75)
            ),
        }
        for column in frequency_columns
    }

    report = {
        "analysis_root": str(analysis_root.resolve()),
        "analysis_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": actual_manifest_sha,
            "status": manifest.get("analysis_status"),
            "runtime_seconds": manifest.get("runtime_seconds"),
            "no_preregistration_claim": manifest.get("no_preregistration_claim"),
        },
        "contracts": {
            "excess_reliability": contracts,
            "soft_core": soft_contract,
            "B_confirm_policy": manifest.get("B_confirm_policy"),
            "interpretation_boundary": manifest.get("interpretation_boundary"),
        },
        "quality_gates": gates,
        "formal_reconstruction_audits": manifest.get(
            "formal_reconstruction_audits", {}
        ),
        "formal_and_exact_points": points.to_dict("records"),
        "split_half": {
            "summary": reliability_summary.to_dict("records"),
            "replicates": reliability.to_dict("records"),
            "partition_balance": partition_balance.to_dict("records"),
            "activity_bin_balance": activity_balance.to_dict("records"),
            "matching_quality": matching_quality.to_dict("records"),
            "balance_summary": balance_summary,
            "matching_summary": matching_summary,
            "activity_bin_audit": activity_bin_audit,
        },
        "attenuation_benchmark": {
            "summary": attenuation_summary.to_dict("records"),
            "replicates": attenuation.to_dict("records"),
        },
        "soft_core": {
            "summary": soft_summary.to_dict("records"),
            "replicates": soft_replicates.to_dict("records"),
            "threshold_summary": soft_thresholds.to_dict("records"),
            "cell_selection_frequencies": soft_cells.to_dict("records"),
            "replicate_audit": soft_replicate_audit,
            "frequency_summary": frequency_summary,
        },
        "source_files": source_rows,
        "input_table_inventory": table_inventory(table_root).to_dict("records"),
        "input_array_inventory": array_inventory(array_root).to_dict("records"),
    }
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    points = pd.DataFrame(report["formal_and_exact_points"])
    reliability = pd.DataFrame(report["split_half"]["summary"])
    attenuation = pd.DataFrame(report["attenuation_benchmark"]["summary"])
    soft_summary = pd.DataFrame(report["soft_core"]["summary"])
    thresholds = pd.DataFrame(report["soft_core"]["threshold_summary"])
    gates = pd.DataFrame(
        [{"gate": key, "value": value} for key, value in report["quality_gates"].items()]
    )
    sources = pd.DataFrame(report["source_files"])

    point_columns = [
        "estimator",
        "support",
        "supported_cells",
        "vector_correlation",
        "speed_correlation",
        "M_component_correlation",
        "Psi_component_correlation",
        "weighted_local_cosine",
        "local_cosine_weight_coverage",
    ]
    attenuation_columns = [
        column
        for column in (
            "metric",
            "partitions",
            "coverage_valid_partitions",
            "benchmark_defined_partitions",
            "minimum_defined_partitions",
            "benchmark_reporting_eligible",
            "formal_target_cells",
            "observed_exact_cross_split_agreement_50",
            "attenuation_benchmark_50",
            "agreement_to_benchmark_ratio_50",
            "attenuation_benchmark_2p5",
            "attenuation_benchmark_97p5",
            "agreement_to_benchmark_ratio_2p5",
            "agreement_to_benchmark_ratio_97p5",
        )
        if column in attenuation.columns
    ]

    lines = [
        "# Empirical excess-field reliability and soft-core stability",
        "",
        "## Analysis contract",
        "",
        f"- Status: {report['analysis_manifest'].get('status')}",
        f"- Manifest SHA-256: `{report['analysis_manifest'].get('sha256')}`",
        f"- Runtime: {report['analysis_manifest'].get('runtime_seconds')} seconds",
        f"- B-confirm policy: {report['contracts'].get('B_confirm_policy')}",
        f"- Interpretation boundary: {report['contracts'].get('interpretation_boundary')}",
        "",
        "## Quality and integrity gates",
        "",
        markdown_table(gates),
        "",
        "## Formal and exact excess-field agreement",
        "",
        markdown_table(points[point_columns]),
        "",
        "## Repeated complementary split-half reliability",
        "",
        markdown_table(reliability),
        "",
        "## Cross-cohort attenuation benchmark",
        "",
        markdown_table(attenuation[attenuation_columns]),
        "",
        "## Soft-core selection stability",
        "",
        markdown_table(soft_summary),
        "",
        "### Threshold sensitivity",
        "",
        markdown_table(thresholds),
        "",
        "### Matching, balance and frequency audit",
        "",
        "```json",
        json.dumps(
            json_safe(
                {
                    "split_half_balance": report["split_half"]["balance_summary"],
                    "activity_bin_audit": report["split_half"]["activity_bin_audit"],
                    "matching_summary": report["split_half"]["matching_summary"],
                    "soft_core_replicate_audit": report["soft_core"]["replicate_audit"],
                    "soft_core_frequency_summary": report["soft_core"]["frequency_summary"],
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        "```",
        "",
        "## Source integrity",
        "",
        markdown_table(
            sources[
                [
                    "path",
                    "recorded_sha256",
                    "current_sha256",
                    "checksum_matches",
                    "bytes",
                ]
            ],
            digits=8,
        ),
        "",
    ]
    return "\n".join(lines)


def output_file_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(root)),
                    "bytes": int(path.stat().st_size),
                    "sha256": file_sha256(path),
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    analysis_root = args.analysis_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else analysis_root / "numeric_report"
    )
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Report directory exists: {output_root}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    table_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    report = build_report(analysis_root)
    report_json = output_root / "empirical_excess_reliability_soft_core_report.json"
    report_markdown = output_root / "empirical_excess_reliability_soft_core_report.md"
    save_json(report, report_json)
    report_markdown.write_text(render_markdown(report), encoding="utf-8")

    write_table(
        pd.DataFrame(report["formal_and_exact_points"]),
        table_root / "formal_and_exact_excess_field_points",
    )
    write_table(
        pd.DataFrame(report["split_half"]["summary"]),
        table_root / "split_half_reliability_summary",
    )
    write_table(
        pd.DataFrame(report["attenuation_benchmark"]["summary"]),
        table_root / "attenuation_benchmark_summary",
    )
    write_table(
        pd.DataFrame(report["soft_core"]["summary"]),
        table_root / "soft_core_summary",
    )
    write_table(
        pd.DataFrame(report["soft_core"]["threshold_summary"]),
        table_root / "soft_core_threshold_summary",
    )
    write_table(
        pd.DataFrame(
            [{"gate": key, "value": value} for key, value in report["quality_gates"].items()]
        ),
        table_root / "quality_gates",
    )
    write_table(
        pd.DataFrame(report["input_table_inventory"]),
        table_root / "analysis_table_inventory",
    )
    write_table(
        pd.DataFrame(report["input_array_inventory"]),
        table_root / "analysis_array_inventory",
    )

    manifest = {
        "report_script": Path(__file__).name,
        "report_script_sha256": file_sha256(Path(__file__).resolve()),
        "analysis_root": str(analysis_root),
        "output_root": str(output_root),
        "analysis_manifest_sha256": report["analysis_manifest"]["sha256"],
        "report_json": str(report_json.resolve()),
        "report_json_sha256": file_sha256(report_json),
        "report_markdown": str(report_markdown.resolve()),
        "report_markdown_sha256": file_sha256(report_markdown),
        "scientific_values_recomputed": False,
        "source_files_verified": True,
    }
    manifest_path = metadata_root / "numeric_report_manifest.json"
    save_json(manifest, manifest_path)
    write_table(output_file_manifest(output_root), table_root / "numeric_report_file_manifest")
    save_json(
        {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": file_sha256(manifest_path),
        },
        metadata_root / "numeric_report_manifest.sha256.json",
    )
    print(f"[report] completed: {output_root}")


def self_test() -> None:
    summary = finite_summary([1.0, 2.0, 3.0, np.nan])
    if summary["n"] != 3 or not np.isclose(float(summary["50"]), 2.0):
        raise RuntimeError("Finite-summary self-test failed.")
    frame = pd.DataFrame({"flag": [True, False, 1, 0]})
    if int(bool_series(frame, "flag").sum()) != 2:
        raise RuntimeError("Boolean coercion self-test failed.")
    print("[self-test] numerical report helpers passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract the empirical excess-field reliability and soft-core numerical report."
    )
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
