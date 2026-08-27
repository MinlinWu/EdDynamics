#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd


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
        return number if np.isfinite(number) else None
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_table(base: Path) -> Path:
    for extension in (".parquet", ".csv.gz", ".csv"):
        path = base.with_suffix(extension)
        if path.exists():
            return path
    raise FileNotFoundError(base)


def read_table(base: Path) -> pd.DataFrame:
    path = find_table(base)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def markdown_table(frame: pd.DataFrame, columns: Optional[list[str]] = None) -> str:
    view = frame if columns is None else frame[columns]
    display = view.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: fmt(value))
    return display.to_markdown(index=False, disable_numparse=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/six_seed_null_common_target_audit"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def run_self_test() -> None:
    frame = pd.DataFrame({"seed": [42, 2026], "skill": [-1.2, -0.7]})
    text = markdown_table(frame)
    if "42" not in text or "-1.2" not in text:
        raise AssertionError("Markdown formatting self-test failed.")
    print("report extractor self-test passed")


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    result_root = args.result_root.resolve()
    manifest_path = result_root / "metadata" / "six_seed_null_common_target_audit_manifest.json"
    sidecar_path = result_root / "metadata" / "six_seed_null_common_target_audit_manifest.sha256.json"
    if not manifest_path.exists() or not sidecar_path.exists():
        raise FileNotFoundError("Audit manifest or checksum sidecar is missing.")
    expected = str(load_json(sidecar_path).get("manifest_sha256", ""))
    actual = file_sha256(manifest_path)
    if expected != actual:
        raise RuntimeError("Audit manifest checksum mismatch.")
    manifest = load_json(manifest_path)

    report_root = result_root / "numeric_report"
    if report_root.exists() and args.overwrite:
        shutil.rmtree(report_root)
    if report_root.exists() and any(report_root.iterdir()):
        raise FileExistsError(f"Numeric report root is not empty: {report_root}")
    table_out = report_root / "tables"
    metadata_out = report_root / "metadata"
    table_out.mkdir(parents=True, exist_ok=True)
    metadata_out.mkdir(parents=True, exist_ok=True)

    tables = {
        "null_metrics": read_table(result_root / "tables" / "event_ssl_seed_null_calibration_metrics"),
        "null_summary": read_table(result_root / "tables" / "event_ssl_seed_null_calibration_summary"),
        "sign_counts": read_table(result_root / "tables" / "event_ssl_seed_null_calibration_sign_counts"),
        "common_metrics": read_table(result_root / "tables" / "common_target_conditioned_metrics_by_seed"),
        "common_summary": read_table(result_root / "tables" / "common_target_conditioned_seed_summary"),
        "provenance": read_table(result_root / "tables" / "event_ssl_seed_provenance"),
        "quality": read_table(result_root / "tables" / "quality_gates"),
    }
    optional_bases = {
        "bootstrap_replicates": result_root / "tables" / "seed42_common_target_learner_bootstrap_replicates",
        "bootstrap_summary": result_root / "tables" / "seed42_common_target_learner_bootstrap_summary",
    }
    for name, base in optional_bases.items():
        try:
            tables[name] = read_table(base)
        except FileNotFoundError:
            pass

    failed = tables["quality"].loc[~tables["quality"]["passed"].astype(bool)]
    if not failed.empty:
        raise RuntimeError("Quality gates are not all passed.")

    for name, frame in tables.items():
        output = table_out / f"{name}.csv"
        frame.to_csv(output, index=False)

    confirm_null = tables["null_metrics"][tables["null_metrics"]["split"] == "B_confirm"].sort_values("seed")
    confirm_common = tables["common_metrics"][tables["common_metrics"]["split"] == "B_confirm"].sort_values("seed")
    confirm_signs = tables["sign_counts"][tables["sign_counts"]["split"] == "B_confirm"].iloc[0]

    lines = [
        "# Six-seed null calibration and common-target-conditioned cross-model audit",
        "",
        "## Analysis boundary",
        "",
        f"- Status: {manifest.get('status')}",
        f"- Seeds: {', '.join(str(seed) for seed in manifest.get('seeds', []))}",
        "- Event-SSL null subtraction uses empirical-anchor fields only; learned-plane dynamics remain a separate self-consistency result.",
        "- The common-target product and partial correlation are descriptive linear calibrations, not cell-level inferential nulls.",
        "- No model, coordinate, grid, support, construction-null protocol or scaffold calibration was refitted per seed.",
        "",
        "## Confirmation-set Event-SSL null calibration",
        "",
        markdown_table(
            confirm_null,
            [
                "seed",
                "event_ssl_distance",
                "primary_delta_sse_model_minus_exact_null",
                "null_relative_field_skill",
                "M_null_relative_field_skill",
                "Psi_null_relative_field_skill",
                "model_minus_diagonal_rescaled_scaffold_sse",
            ],
        ),
        "",
        "### Direction counts",
        "",
        markdown_table(pd.DataFrame([confirm_signs])),
        "",
        "## Confirmation-set common-target calibration",
        "",
        markdown_table(
            confirm_common,
            [
                "seed",
                "mechanism_vs_empirical_vector_corr",
                "event_ssl_vs_empirical_vector_corr",
                "mechanism_vs_event_ssl_raw_vector_corr",
                "linear_common_target_product_benchmark",
                "raw_minus_target_product",
                "linear_partial_corr_given_empirical",
                "direct_empirical_error_vector_corr",
                "direct_empirical_error_weighted_local_cosine",
            ],
        ),
        "",
        "## Across-seed summaries",
        "",
        "### Null calibration",
        "",
        markdown_table(tables["null_summary"][tables["null_summary"]["group"] == "B_confirm"]),
        "",
        "### Common-target-conditioned agreement",
        "",
        markdown_table(tables["common_summary"][tables["common_summary"]["group"] == "B_confirm"]),
    ]
    if "bootstrap_summary" in tables:
        bootstrap_summary = tables["bootstrap_summary"]
        exact_match = bool(
            bootstrap_summary.get(
                "point_estimands_exactly_matched", pd.Series([True])
            ).astype(bool).all()
        )
        interval_note = (
            "These intervals are derived from the previously completed fixed-support ordinary "
            "learner-cluster bootstrap; no new resampling was performed."
            if exact_match
            else
            "These intervals remain centred on the archived joined-cohort fixed-support headline "
            "points. The Stage-1-weighted null-audit points use a distinct aggregation contract; "
            "their differences are reported in the table, and the intervals are not recentered."
        )
        lines.extend(
            [
                "",
                "## Seed-42 learner-cluster uncertainty for common-target metrics",
                "",
                markdown_table(bootstrap_summary),
                "",
                interval_note,
            ]
        )

    all_failed_null = int(confirm_signs["overall_skill_above_zero"]) == 0
    all_failed_diagonal = int(confirm_signs["better_than_diagonal_rescaled_scaffold"]) == 0
    lines.extend(
        [
            "",
            "## Result-dependent interpretation checks",
            "",
            f"- Positive overall null-relative skill: {int(confirm_signs['overall_skill_above_zero'])}/{int(confirm_signs['seeds'])} seeds.",
            f"- Better than the frozen coordinatewise-rescaled scaffold: {int(confirm_signs['better_than_diagonal_rescaled_scaffold'])}/{int(confirm_signs['seeds'])} seeds.",
            (
                "- The single-seed null-calibration failure reproduces across all six seeds."
                if all_failed_null
                else "- Null calibration is seed-dependent; the single-seed result cannot be generalized as a uniform model-class failure."
            ),
            (
                "- No seed exceeds the stronger coordinatewise-rescaled scaffold."
                if all_failed_diagonal
                else "- At least one seed exceeds the stronger coordinatewise-rescaled scaffold; seed-level heterogeneity must be retained in the manuscript."
            ),
            "- Raw cross-model agreement must be interpreted together with the target-product benchmark, partial correlation and direct empirical-error diagnostics.",
        ]
    )

    report_path = report_root / "six_seed_null_common_target_audit_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_json = {
        "manifest": manifest,
        "confirmation_null_metrics": confirm_null.to_dict(orient="records"),
        "confirmation_common_target_metrics": confirm_common.to_dict(orient="records"),
        "confirmation_sign_counts": json_safe(confirm_signs.to_dict()),
        "tables": {name: str((table_out / f"{name}.csv").resolve()) for name in tables},
        "report": str(report_path.resolve()),
    }
    json_path = report_root / "six_seed_null_common_target_audit_report.json"
    save_json(report_json, json_path)

    inventory = {}
    for path in sorted(report_root.rglob("*")):
        if path.is_file():
            inventory[str(path.relative_to(report_root))] = {
                "sha256": file_sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
    report_manifest = {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": actual,
        "report_root": str(report_root),
        "inventory": inventory,
    }
    report_manifest_path = metadata_out / "numeric_report_manifest.json"
    save_json(report_manifest, report_manifest_path)
    save_json(
        {"manifest_sha256": file_sha256(report_manifest_path)},
        metadata_out / "numeric_report_manifest.sha256.json",
    )
    print(f"Numerical report: {report_path}")


if __name__ == "__main__":
    main()
