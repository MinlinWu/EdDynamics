#!/usr/bin/env python3
from __future__ import annotations

"""Extract a numerical report from the frozen headline learner-cluster audit."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_RESULT_ROOT = Path(
    "/data/datasets/KT4/outputs_KT4/frozen_headline_learner_cluster_uncertainty"
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
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_table(base_or_path: Path) -> Path:
    path = Path(base_or_path)
    if path.is_file():
        return path
    for suffix in (".parquet", ".csv.gz", ".csv"):
        candidate = path.with_suffix(suffix)
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
        temporary_csv = csv_path.with_name(csv_path.name + ".tmp")
        frame.to_csv(temporary_csv, index=False, compression="gzip")
        os.replace(temporary_csv, csv_path)
        return csv_path


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_csv(index=False)


def coerce_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, float) and np.isfinite(value):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    raise ValueError(f"Cannot coerce boolean value: {value!r}")


def bool_series(values: pd.Series) -> pd.Series:
    return values.map(coerce_bool).astype(bool)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return "--"
    if not np.isfinite(number):
        return "--"
    if abs(number) >= 100000:
        return f"{number:,.0f}"
    if abs(number) >= 1000:
        return f"{number:,.1f}"
    return f"{number:.{digits}f}"


def verify_manifest(result_root: Path) -> Tuple[Dict[str, Any], Path, Dict[str, Any]]:
    manifest_path = result_root / "metadata" / "analysis_manifest.json"
    sidecar_path = result_root / "metadata" / "analysis_manifest.sha256.json"
    if not manifest_path.exists() or not sidecar_path.exists():
        raise FileNotFoundError("Analysis manifest or checksum sidecar is missing.")
    manifest = load_json(manifest_path)
    sidecar = load_json(sidecar_path)
    actual = sha256_file(manifest_path)
    expected = str(sidecar.get("manifest_sha256", "") or "")
    if expected != actual:
        raise RuntimeError(
            f"Analysis manifest checksum mismatch: expected {expected}, found {actual}."
        )
    return manifest, manifest_path, {
        "manifest_sha256": actual,
        "sidecar_path": str(sidecar_path.resolve()),
        "verified": True,
    }


def resolve_output_path(
    result_root: Path,
    name: str,
    manifest: Mapping[str, Any],
) -> Path:
    record = dict(manifest.get("output_files", {})).get(name)
    if not isinstance(record, Mapping):
        raw = dict(manifest.get("outputs", {})).get(name)
        if raw is None:
            raise KeyError(f"Manifest output is absent: {name}")
        record = {"path": raw}
    declared = Path(str(record.get("path", "")))
    candidates = [
        declared,
        result_root / "tables" / declared.name,
        result_root / "metadata" / declared.name,
    ]
    actual = None
    for candidate in candidates:
        if candidate.exists():
            actual = candidate.resolve()
            break
    if actual is None:
        raise FileNotFoundError(f"Could not resolve manifest output {name}: {declared}")
    expected_sha = str(record.get("sha256", "") or "")
    if expected_sha:
        observed_sha = sha256_file(actual)
        if observed_sha != expected_sha:
            raise RuntimeError(
                f"Output checksum mismatch for {name}: expected {expected_sha}, found {observed_sha}."
            )
    expected_bytes = record.get("bytes")
    if expected_bytes is not None and int(expected_bytes) != int(actual.stat().st_size):
        raise RuntimeError(f"Output byte-size mismatch for {name}.")
    return actual


def load_outputs(result_root: Path, manifest: Mapping[str, Any]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Path]]:
    names = (
        "formal_point_estimates",
        "formal_point_reconstruction_audit",
        "field_bootstrap_replicates",
        "field_bootstrap_intervals",
        "transition_bootstrap_replicates",
        "transition_bootstrap_intervals",
        "statewise_bootstrap_replicates",
        "statewise_persistence_summary",
        "transition_matrices",
        "statewise_point_estimates",
        "persistence_leave_one_state_out",
        "quality_gates",
    )
    paths = {name: resolve_output_path(result_root, name, manifest) for name in names}
    frames = {name: read_table(path) for name, path in paths.items()}
    return frames, paths


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing required columns: {missing}")


def fixed_and_reselected_field_table(intervals: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        intervals,
        (
            "comparison",
            "support_contract",
            "metric",
            "point_estimate",
            "reported_ci_low",
            "reported_ci_high",
            "reported_interval_method",
            "finite_fraction",
        ),
        "field interval table",
    )
    fixed = intervals[intervals["support_contract"] == "fixed_formal_support"].copy()
    adaptive = intervals[intervals["support_contract"] == "support_reselected"].copy()
    fixed = fixed.rename(
        columns={
            "reported_ci_low": "fixed_support_ci_low",
            "reported_ci_high": "fixed_support_ci_high",
            "reported_interval_method": "fixed_support_interval_method",
            "finite_fraction": "fixed_support_finite_fraction",
        }
    )
    adaptive = adaptive.rename(
        columns={
            "reported_ci_low": "reselected_support_ci_low",
            "reported_ci_high": "reselected_support_ci_high",
            "reported_interval_method": "reselected_support_interval_method",
            "finite_fraction": "reselected_support_finite_fraction",
        }
    )
    columns = [
        "comparison",
        "metric",
        "point_estimate",
        "fixed_support_ci_low",
        "fixed_support_ci_high",
        "fixed_support_interval_method",
        "fixed_support_finite_fraction",
    ]
    output = fixed[columns].merge(
        adaptive[
            [
                "comparison",
                "metric",
                "reselected_support_ci_low",
                "reselected_support_ci_high",
                "reselected_support_interval_method",
                "reselected_support_finite_fraction",
            ]
        ],
        on=["comparison", "metric"],
        how="left",
        validate="one_to_one",
    )
    return output.sort_values(["comparison", "metric"], kind="mergesort").reset_index(drop=True)


def transition_uncertainty_table(intervals: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        intervals,
        (
            "comparison",
            "metric",
            "point_estimate",
            "reported_ci_low",
            "reported_ci_high",
            "reported_interval_method",
            "finite_fraction",
        ),
        "transition interval table",
    )
    columns = [
        "comparison",
        "metric",
        "point_estimate",
        "reported_ci_low",
        "reported_ci_high",
        "reported_interval_method",
        "finite_fraction",
        "bootstrap_mean",
        "bootstrap_sd",
        "bootstrap_bias",
        "percentile_low",
        "percentile_high",
    ]
    return intervals[columns].sort_values(["comparison", "metric"], kind="mergesort").reset_index(drop=True)


def statewise_persistence_table(
    points: pd.DataFrame,
    intervals: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        points,
        (
            "state",
            "empirical_Pii",
            "mechanism_Pii",
            "event_ssl_anchor_Pii",
            "event_ssl_learned_Pii",
        ),
        "statewise points",
    )
    require_columns(
        intervals,
        ("state", "quantity", "point_estimate", "reported_ci_low", "reported_ci_high"),
        "statewise intervals",
    )
    output = points.copy().sort_values("state", kind="mergesort").reset_index(drop=True)
    for quantity in intervals["quantity"].dropna().astype(str).unique():
        subset = intervals[intervals["quantity"].astype(str) == quantity][
            ["state", "reported_ci_low", "reported_ci_high", "finite_fraction"]
        ].copy()
        subset = subset.rename(
            columns={
                "reported_ci_low": f"{quantity}_ci_low",
                "reported_ci_high": f"{quantity}_ci_high",
                "finite_fraction": f"{quantity}_finite_fraction",
            }
        )
        output = output.merge(subset, on="state", how="left", validate="one_to_one")
    return output


def persistence_influence_table(
    point: pd.DataFrame,
    transition_intervals: pd.DataFrame,
    loso: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(loso, ("omitted_state", "pearson", "spearman"), "LOSO table")
    target = transition_intervals[
        transition_intervals["comparison"] == "mechanism_vs_event_ssl"
    ].copy()
    pearson_row = target[target["metric"] == "self_transition_pearson"]
    spearman_row = target[target["metric"] == "self_transition_spearman"]
    if len(pearson_row) != 1 or len(spearman_row) != 1:
        raise RuntimeError("Cross-model persistence summaries are not unique.")
    pearson_loso = pd.to_numeric(loso["pearson"], errors="coerce")
    spearman_loso = pd.to_numeric(loso["spearman"], errors="coerce")
    finite_p = loso[np.isfinite(pearson_loso)].copy()
    finite_s = loso[np.isfinite(spearman_loso)].copy()

    def loso_summary(frame: pd.DataFrame, column: str) -> Dict[str, Any]:
        numeric_values = pd.to_numeric(loso[column], errors="coerce")
        undefined = loso.loc[~np.isfinite(numeric_values), "omitted_state"].astype(int).tolist()
        if frame.empty:
            return {
                "loso_finite_n": 0,
                "loso_undefined_states": (
                    ",".join(str(value) for value in undefined) if undefined else "none"
                ),
                "loso_min": np.nan,
                "loso_max": np.nan,
                "loso_min_omitted_state": np.nan,
                "loso_max_omitted_state": np.nan,
            }
        return {
            "loso_finite_n": int(len(frame)),
            "loso_undefined_states": (
                ",".join(str(value) for value in undefined) if undefined else "none"
            ),
            "loso_min": float(frame[column].min()),
            "loso_max": float(frame[column].max()),
            "loso_min_omitted_state": int(frame.loc[frame[column].idxmin(), "omitted_state"]),
            "loso_max_omitted_state": int(frame.loc[frame[column].idxmax(), "omitted_state"]),
        }

    rows = [
        {
            "summary": "mechanism_vs_event_ssl_self_transition_pearson",
            "point_estimate": float(pearson_row.iloc[0]["point_estimate"]),
            "learner_bootstrap_ci_low": float(pearson_row.iloc[0]["reported_ci_low"]),
            "learner_bootstrap_ci_high": float(pearson_row.iloc[0]["reported_ci_high"]),
            "interval_method": str(pearson_row.iloc[0]["reported_interval_method"]),
            **loso_summary(finite_p, "pearson"),
            "interpretation": "descriptive across six fixed mesostates; learner interval conditions on those states",
        },
        {
            "summary": "mechanism_vs_event_ssl_self_transition_spearman",
            "point_estimate": float(spearman_row.iloc[0]["point_estimate"]),
            "learner_bootstrap_ci_low": float(spearman_row.iloc[0]["reported_ci_low"]),
            "learner_bootstrap_ci_high": float(spearman_row.iloc[0]["reported_ci_high"]),
            "interval_method": str(spearman_row.iloc[0]["reported_interval_method"]),
            **loso_summary(finite_s, "spearman"),
            "interpretation": "descriptive rank agreement across six fixed mesostates",
        },
    ]
    return pd.DataFrame(rows)


def describe_series(values: pd.Series, prefix: str) -> Dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_2p5": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_97p5": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_n": int(len(finite)),
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_2p5": float(finite.quantile(0.025)),
        f"{prefix}_median": float(finite.median()),
        f"{prefix}_97p5": float(finite.quantile(0.975)),
        f"{prefix}_min": float(finite.min()),
        f"{prefix}_max": float(finite.max()),
    }


def support_summary_table(field_replicates: pd.DataFrame) -> pd.DataFrame:
    required = (
        "comparison",
        "support_contract",
        "fixed_support_complete",
        "formal_supported_cells",
        "adaptive_supported_cells",
        "support_jaccard",
        "formal_support_weight_coverage",
        "formal_cells_lost",
        "new_cells_added",
    )
    require_columns(field_replicates, required, "field bootstrap replicates")
    source = field_replicates[
        field_replicates["support_contract"] == "support_reselected"
    ].copy()
    rows: List[dict] = []
    for comparison, group in source.groupby("comparison", sort=False):
        row: Dict[str, Any] = {
            "comparison": comparison,
            "replicates": int(group["replicate"].nunique()),
            "fixed_support_complete_fraction": float(
                bool_series(pd.Series(group["fixed_support_complete"])).mean()
            ),
            "formal_supported_cells": int(
                pd.to_numeric(group["formal_supported_cells"], errors="coerce").dropna().iloc[0]
            ),
        }
        for column in (
            "adaptive_supported_cells",
            "support_jaccard",
            "formal_support_weight_coverage",
            "formal_cells_lost",
            "new_cells_added",
        ):
            row.update(describe_series(group[column], column))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_bias_table(
    field_intervals: pd.DataFrame,
    transition_intervals: pd.DataFrame,
    statewise_intervals: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    field = field_intervals.copy()
    field.insert(0, "domain", "field")
    field["label"] = field["comparison"].astype(str) + "/" + field["support_contract"].astype(str) + "/" + field["metric"].astype(str)
    rows.append(field)
    transition = transition_intervals.copy()
    transition.insert(0, "domain", "transition")
    transition["label"] = transition["comparison"].astype(str) + "/" + transition["metric"].astype(str)
    rows.append(transition)
    statewise = statewise_intervals.copy()
    statewise.insert(0, "domain", "statewise_persistence")
    statewise["label"] = "S" + statewise["state"].astype(str) + "/" + statewise["quantity"].astype(str)
    rows.append(statewise)
    common = [
        "domain",
        "label",
        "point_estimate",
        "bootstrap_mean",
        "bootstrap_sd",
        "bootstrap_median",
        "bootstrap_bias",
        "reported_interval_method",
        "reported_ci_low",
        "reported_ci_high",
        "se_centered_ci_low",
        "se_centered_ci_high",
        "percentile_low",
        "percentile_high",
        "finite_fraction",
    ]
    output = pd.concat([frame.reindex(columns=common) for frame in rows], ignore_index=True)
    return output


def quality_and_integrity_table(
    quality: pd.DataFrame,
    reconstruction: pd.DataFrame,
    paths: Mapping[str, Path],
) -> pd.DataFrame:
    require_columns(quality, ("quality_gate", "passed", "detail"), "quality gates")
    rows = [
        {
            "check": str(row["quality_gate"]),
            "passed": coerce_bool(row["passed"]),
            "detail": str(row["detail"]),
        }
        for _, row in quality.iterrows()
    ]
    for _, row in reconstruction.iterrows():
        rows.append(
            {
                "check": "point_reconstruction/" + str(row["metric"]),
                "passed": coerce_bool(row["passed"]),
                "detail": (
                    f"reconstructed={fmt(row['reconstructed_point'], 6)}, "
                    f"reference={fmt(row['manuscript_rounded_reference'], 6)}, "
                    f"abs_diff={fmt(row['absolute_difference'], 6)}"
                ),
            }
        )
    for name, path in paths.items():
        rows.append(
            {
                "check": "output_integrity/" + name,
                "passed": True,
                "detail": f"sha256={sha256_file(path)}, bytes={path.stat().st_size}",
            }
        )
    return pd.DataFrame(rows)


def synthesis_flags(
    field: pd.DataFrame,
    transition: pd.DataFrame,
    influence: pd.DataFrame,
    quality: pd.DataFrame,
) -> Dict[str, Any]:
    def row_value(frame: pd.DataFrame, comparison: str, metric: str, column: str) -> float:
        selected = frame[(frame["comparison"] == comparison) & (frame["metric"] == metric)]
        if len(selected) != 1:
            return float("nan")
        return float(selected.iloc[0][column])

    field_flags: Dict[str, Any] = {}
    for comparison in field["comparison"].astype(str).unique():
        low = row_value(field, comparison, "drift_vector_corr", "fixed_support_ci_low")
        field_flags[comparison] = {
            "fixed_support_vector_ci_excludes_zero": bool(np.isfinite(low) and low > 0),
            "fixed_support_vector_ci_low": low,
        }
    cross_tv_high = row_value(
        transition,
        "mechanism_vs_event_ssl",
        "transition_mean_row_tv",
        "reported_ci_high",
    )
    pearson = influence[influence["summary"].str.endswith("pearson")].iloc[0]
    return {
        "all_quality_gates_passed": bool(bool_series(pd.Series(quality["passed"])).all()),
        "field_direction_flags": field_flags,
        "cross_model_transition_row_tv_ci_high": cross_tv_high,
        "cross_model_persistence_pearson_loso_min": float(pearson["loso_min"]),
        "cross_model_persistence_pearson_loso_max": float(pearson["loso_max"]),
        "cross_model_persistence_pearson_loso_finite_n": int(pearson["loso_finite_n"]),
        "cross_model_persistence_pearson_loso_undefined_states": str(
            pearson["loso_undefined_states"]
        ),
        "interpretation_boundary": (
            "Learner-cluster intervals quantify confirmation-learner sampling under frozen outputs. "
            "Six-state correlations remain descriptive because the six mesostates are fixed, not sampled states."
        ),
    }


def build_report(
    manifest: Mapping[str, Any],
    manifest_audit: Mapping[str, Any],
    field: pd.DataFrame,
    transition: pd.DataFrame,
    statewise: pd.DataFrame,
    influence: pd.DataFrame,
    loso: pd.DataFrame,
    support: pd.DataFrame,
    quality: pd.DataFrame,
    synthesis: Mapping[str, Any],
) -> str:
    lines = [
        "# Frozen headline learner-cluster uncertainty",
        "",
        "## Analysis contract",
        "",
        f"- Status: {manifest.get('status', '--')}",
        f"- Resampling unit: {dict(manifest.get('bootstrap_contract', {})).get('resampling_unit', '--')}",
        f"- Bootstrap replicates: {dict(manifest.get('bootstrap_contract', {})).get('replicates', '--')}",
        f"- Primary field interval: fixed formal support with {dict(manifest.get('bootstrap_contract', {})).get('primary_interval', '--')}",
        "- Secondary field interval: support reselected with the unchanged raw transition-count threshold of 30.",
        "- Models, coordinates, grid and the fixed six-state partition were not refitted.",
        "- Construction-null, recursive-surrogate and existing null-relative skill analyses were not rerun.",
        f"- Manifest SHA-256: `{manifest_audit.get('manifest_sha256', '--')}`",
        "",
        "## Quality and reconstruction gates",
        "",
        markdown_table(quality),
        "",
        "## Headline field uncertainty",
        "",
        markdown_table(
            field[
                [
                    "comparison",
                    "metric",
                    "point_estimate",
                    "fixed_support_ci_low",
                    "fixed_support_ci_high",
                    "reselected_support_ci_low",
                    "reselected_support_ci_high",
                ]
            ]
        ),
        "",
        "Fixed-support intervals quantify learner sampling conditional on the manuscript evaluation cells. Support-reselected intervals additionally propagate which cells meet the formal count threshold.",
        "",
        "## Transition and persistence uncertainty",
        "",
        markdown_table(
            transition[
                [
                    "comparison",
                    "metric",
                    "point_estimate",
                    "reported_ci_low",
                    "reported_ci_high",
                    "reported_interval_method",
                ]
            ]
        ),
        "",
        "## Six-state persistence values",
        "",
        markdown_table(statewise),
        "",
        "## Six-state influence diagnostics",
        "",
        markdown_table(influence),
        "",
        "### Leave-one-state-out values",
        "",
        markdown_table(loso),
        "",
        "The Pearson and Spearman summaries are descriptive across six fixed mesostates. Learner-bootstrap intervals condition on those states and do not constitute inference over a state population.",
        "",
        "## Support reselection diagnostics",
        "",
        markdown_table(support),
        "",
        "## Machine-readable synthesis",
        "",
        "```json",
        json.dumps(json_safe(synthesis), indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the frozen headline learner-cluster numerical report."
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_extraction(result_root: Path, output_root: Path, overwrite: bool) -> Dict[str, Any]:
    result_root = result_root.resolve()
    manifest, manifest_path, manifest_audit = verify_manifest(result_root)
    frames, paths = load_outputs(result_root, manifest)
    quality = frames["quality_gates"].copy()
    reconstruction = frames["formal_point_reconstruction_audit"].copy()
    if not bool(bool_series(pd.Series(quality["passed"])).all()):
        raise RuntimeError("The analysis quality gates did not all pass.")
    if not bool(bool_series(pd.Series(reconstruction["passed"])).all()):
        raise RuntimeError("The formal point reconstruction audit did not pass.")

    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"Output root is not empty: {output_root}")
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    table_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    field = fixed_and_reselected_field_table(frames["field_bootstrap_intervals"])
    transition = transition_uncertainty_table(frames["transition_bootstrap_intervals"])
    statewise = statewise_persistence_table(
        frames["statewise_point_estimates"],
        frames["statewise_persistence_summary"],
    )
    influence = persistence_influence_table(
        frames["formal_point_estimates"],
        frames["transition_bootstrap_intervals"],
        frames["persistence_leave_one_state_out"],
    )
    support = support_summary_table(frames["field_bootstrap_replicates"])
    bias = bootstrap_bias_table(
        frames["field_bootstrap_intervals"],
        frames["transition_bootstrap_intervals"],
        frames["statewise_persistence_summary"],
    )
    integrity = quality_and_integrity_table(quality, reconstruction, paths)
    synthesis = synthesis_flags(field, transition, influence, quality)

    output_paths = {
        "headline_field_uncertainty": write_table(
            field, table_root / "headline_field_uncertainty"
        ),
        "headline_transition_uncertainty": write_table(
            transition, table_root / "headline_transition_uncertainty"
        ),
        "statewise_persistence": write_table(
            statewise, table_root / "statewise_persistence"
        ),
        "persistence_influence": write_table(
            influence, table_root / "persistence_influence"
        ),
        "persistence_leave_one_state_out": write_table(
            frames["persistence_leave_one_state_out"],
            table_root / "persistence_leave_one_state_out",
        ),
        "support_reselection_summary": write_table(
            support, table_root / "support_reselection_summary"
        ),
        "bootstrap_bias_diagnostics": write_table(
            bias, table_root / "bootstrap_bias_diagnostics"
        ),
        "formal_transition_matrices": write_table(
            frames["transition_matrices"], table_root / "formal_transition_matrices"
        ),
        "quality_and_integrity": write_table(
            integrity, table_root / "quality_and_integrity"
        ),
    }

    report_text = build_report(
        manifest,
        manifest_audit,
        field,
        transition,
        statewise,
        influence,
        frames["persistence_leave_one_state_out"],
        support,
        integrity,
        synthesis,
    )
    report_path = output_root / "frozen_headline_learner_cluster_uncertainty_report.md"
    atomic_write_text(report_text, report_path)

    payload = {
        "analysis_manifest": manifest,
        "analysis_manifest_audit": manifest_audit,
        "headline_field_uncertainty": field.to_dict(orient="records"),
        "headline_transition_uncertainty": transition.to_dict(orient="records"),
        "statewise_persistence": statewise.to_dict(orient="records"),
        "persistence_influence": influence.to_dict(orient="records"),
        "persistence_leave_one_state_out": frames["persistence_leave_one_state_out"].to_dict(orient="records"),
        "support_reselection_summary": support.to_dict(orient="records"),
        "quality_and_integrity": integrity.to_dict(orient="records"),
        "synthesis": synthesis,
    }
    json_path = output_root / "frozen_headline_learner_cluster_uncertainty_report.json"
    save_json(payload, json_path)

    output_inventory = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path.resolve()),
            "bytes": int(path.stat().st_size),
        }
        for name, path in {
            **output_paths,
            "markdown_report": report_path,
            "json_report": json_path,
        }.items()
    }
    report_manifest = {
        "report": "frozen headline learner-cluster uncertainty numerical report",
        "source_result_root": str(result_root),
        "source_analysis_manifest": str(manifest_path.resolve()),
        "source_analysis_manifest_sha256": manifest_audit["manifest_sha256"],
        "report_script": str(Path(__file__).resolve()),
        "report_script_sha256": sha256_file(Path(__file__).resolve()),
        "outputs": output_inventory,
        "scientific_boundary": (
            "The extractor does not recompute model predictions, fields, transitions, bootstrap replicates, "
            "support masks or intervals."
        ),
    }
    report_manifest_path = metadata_root / "report_manifest.json"
    save_json(report_manifest, report_manifest_path)
    save_json(
        {
            "manifest_path": str(report_manifest_path),
            "manifest_sha256": sha256_file(report_manifest_path),
        },
        metadata_root / "report_manifest.sha256.json",
    )
    return {
        "output_root": str(output_root),
        "markdown_report": str(report_path),
        "json_report": str(json_path),
        "tables": {name: str(path) for name, path in output_paths.items()},
    }


def make_self_test_root(root: Path) -> None:
    tables = root / "tables"
    metadata = root / "metadata"
    tables.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    comparisons = [
        "mechanism_vs_empirical",
        "event_ssl_anchor_vs_empirical",
        "event_ssl_learned_vs_empirical",
        "mechanism_vs_event_ssl_anchor",
    ]
    field_intervals = []
    field_reps = []
    for comparison in comparisons:
        for contract in ("fixed_formal_support", "support_reselected"):
            for metric, value in (
                ("drift_vector_corr", 0.8),
                ("drift_speed_corr", 0.75),
                ("weighted_local_cosine", 0.85),
            ):
                field_intervals.append(
                    {
                        "comparison": comparison,
                        "support_contract": contract,
                        "metric": metric,
                        "point_estimate": value,
                        "reported_ci_low": value - 0.05,
                        "reported_ci_high": value + 0.04,
                        "reported_interval_method": "cluster_bootstrap_se_fisher_z",
                        "finite_fraction": 1.0,
                        "bootstrap_mean": value - 0.01,
                        "bootstrap_sd": 0.02,
                        "bootstrap_median": value,
                        "bootstrap_bias": -0.01,
                        "percentile_low": value - 0.06,
                        "percentile_high": value + 0.03,
                    }
                )
        for replicate in range(20):
            field_reps.append(
                {
                    "replicate": replicate,
                    "comparison": comparison,
                    "support_contract": "support_reselected",
                    "fixed_support_complete": True,
                    "formal_supported_cells": 100,
                    "adaptive_supported_cells": 98 + replicate % 3,
                    "support_jaccard": 0.95,
                    "formal_support_weight_coverage": 0.99,
                    "formal_cells_lost": 2,
                    "new_cells_added": 1,
                }
            )
    transition_intervals = []
    transition_reps = []
    for comparison in (
        "mechanism_vs_empirical",
        "event_ssl_anchor_vs_empirical",
        "event_ssl_learned_vs_empirical",
        "mechanism_vs_event_ssl",
    ):
        for metric, value, method in (
            ("transition_mean_row_tv", 0.1, "cluster_bootstrap_se_raw_scale"),
            ("transition_max_row_tv", 0.2, "cluster_bootstrap_se_raw_scale"),
            ("self_transition_pearson", 0.95, "cluster_bootstrap_se_fisher_z"),
            ("self_transition_spearman", 0.89, "percentile_raw_scale"),
        ):
            transition_intervals.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "point_estimate": value,
                    "reported_ci_low": value - 0.05,
                    "reported_ci_high": min(value + 0.04, 1.0),
                    "reported_interval_method": method,
                    "finite_fraction": 1.0,
                    "bootstrap_mean": value,
                    "bootstrap_sd": 0.02,
                    "bootstrap_median": value,
                    "bootstrap_bias": 0.0,
                    "percentile_low": value - 0.05,
                    "percentile_high": min(value + 0.04, 1.0),
                }
            )
        for replicate in range(20):
            transition_reps.append(
                {
                    "replicate": replicate,
                    "comparison": comparison,
                    "transition_mean_row_tv": 0.1,
                }
            )
    point_statewise = []
    state_intervals = []
    state_reps = []
    for state in range(6):
        point_statewise.append(
            {
                "state": state,
                "empirical_Pii": 0.5 + 0.05 * state,
                "mechanism_Pii": 0.51 + 0.05 * state,
                "event_ssl_anchor_Pii": 0.49 + 0.05 * state,
                "event_ssl_learned_Pii": 0.48 + 0.05 * state,
                "mechanism_minus_event_ssl_Pii": 0.02,
                "mechanism_minus_empirical_Pii": 0.01,
                "event_ssl_minus_empirical_Pii": -0.01,
                "event_ssl_learned_minus_empirical_Pii": -0.02,
                "mechanism_vs_empirical_row_tv": 0.1,
                "event_ssl_vs_empirical_row_tv": 0.15,
                "mechanism_vs_event_ssl_row_tv": 0.12,
            }
        )
        for quantity, value in (
            ("empirical", 0.5 + 0.05 * state),
            ("mechanism", 0.51 + 0.05 * state),
            ("event_ssl_anchor", 0.49 + 0.05 * state),
            ("event_ssl_learned", 0.48 + 0.05 * state),
            ("mechanism_minus_event_ssl", 0.02),
            ("mechanism_minus_empirical", 0.01),
            ("event_ssl_minus_empirical", -0.01),
            ("event_ssl_learned_minus_empirical", -0.02),
        ):
            state_intervals.append(
                {
                    "state": state,
                    "quantity": quantity,
                    "point_estimate": value,
                    "reported_ci_low": value - 0.02,
                    "reported_ci_high": value + 0.02,
                    "reported_interval_method": "cluster_bootstrap_percentile_raw_bounded",
                    "finite_fraction": 1.0,
                    "bootstrap_mean": value,
                    "bootstrap_sd": 0.01,
                    "bootstrap_median": value,
                    "bootstrap_bias": 0.0,
                    "percentile_low": value - 0.02,
                    "percentile_high": value + 0.02,
                }
            )
        for replicate in range(20):
            state_reps.append({"replicate": replicate, "state": state, "empirical_Pii": 0.5})
    loso = pd.DataFrame(
        {
            "omitted_state": np.arange(6),
            "pearson": np.linspace(0.88, 0.96, 6),
            "spearman": np.linspace(0.77, 0.94, 6),
        }
    )
    point = pd.DataFrame(
        [
            {"domain": "transition", "comparison": "mechanism_vs_event_ssl", "metric": "self_transition_pearson", "point_estimate": 0.95}
        ]
    )
    reconstruction = pd.DataFrame(
        [
            {
                "metric": "test",
                "manuscript_rounded_reference": 0.8,
                "reconstructed_point": 0.8,
                "absolute_difference": 0.0,
                "passed": True,
            }
        ]
    )
    quality = pd.DataFrame(
        [{"quality_gate": "test", "passed": True, "detail": "ok"}]
    )
    transition_matrix = pd.DataFrame(
        [
            {
                "model": model,
                "origin_state": i,
                "destination_state": j,
                "interval_count": 100,
                "transition_probability": 1.0 if i == j else 0.0,
                "self_transition": i == j,
            }
            for model in ("empirical", "mechanism", "event_anchor", "event_learned")
            for i in range(6)
            for j in range(6)
        ]
    )
    frames = {
        "formal_point_estimates": point,
        "formal_point_reconstruction_audit": reconstruction,
        "field_bootstrap_replicates": pd.DataFrame(field_reps),
        "field_bootstrap_intervals": pd.DataFrame(field_intervals),
        "transition_bootstrap_replicates": pd.DataFrame(transition_reps),
        "transition_bootstrap_intervals": pd.DataFrame(transition_intervals),
        "statewise_bootstrap_replicates": pd.DataFrame(state_reps),
        "statewise_persistence_summary": pd.DataFrame(state_intervals),
        "transition_matrices": transition_matrix,
        "statewise_point_estimates": pd.DataFrame(point_statewise),
        "persistence_leave_one_state_out": loso,
        "quality_gates": quality,
    }
    output_files = {}
    for name, frame in frames.items():
        path = write_table(frame, tables / name)
        output_files[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
    manifest = {
        "status": "self-test",
        "bootstrap_contract": {"resampling_unit": "learner", "replicates": 20},
        "outputs": {name: record["path"] for name, record in output_files.items()},
        "output_files": output_files,
    }
    manifest_path = metadata / "analysis_manifest.json"
    save_json(manifest, manifest_path)
    save_json(
        {"manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path)},
        metadata / "analysis_manifest.sha256.json",
    )


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="headline_cluster_report_") as temporary:
        root = Path(temporary) / "analysis"
        report_root = Path(temporary) / "report"
        make_self_test_root(root)
        result = run_extraction(root, report_root, overwrite=False)
        if not Path(result["markdown_report"]).exists() or not Path(result["json_report"]).exists():
            raise RuntimeError("Self-test report outputs were not created.")
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    result_root = args.result_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else result_root / "numeric_report"
    )
    result = run_extraction(result_root, output_root, args.overwrite)
    print(f"completed: {result['output_root']}")


if __name__ == "__main__":
    main()
