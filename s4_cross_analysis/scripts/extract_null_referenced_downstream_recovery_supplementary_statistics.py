#!/usr/bin/env python3
from __future__ import annotations

"""Extract a compact Supplementary Information report for null-referenced recovery.

The upstream analysis is expected to have been produced by
``evaluate_null_referenced_downstream_recovery.py``. This extractor does not
recompute fields, refit models, alter the construction-matched null, redefine
support, or perform additional selection. It reads frozen numerical outputs,
validates their contracts, and creates exactly two typeset-ready supplementary
tables:

1. primary recovery relative to the exact construction-null field, including
   the paired user-level multiplier-bootstrap interval on the output-only
   confirmation cohort;
2. confirmation-set specificity and frozen-core geometry, presented as two
   panels in one table.

Raw model-versus-empirical correlations, original construction-null Monte Carlo
results, matching fractions, transition matrices, and random-seed metrics that
are already reported elsewhere in the manuscript are deliberately excluded
from the two display tables. They remain available in machine-readable audit
outputs when present.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import t as student_t
except Exception:  # pragma: no cover - optional for seed summaries only
    student_t = None


EPS = 1e-12
EXPECTED_ANALYSIS_NAME = "construction-null-referenced downstream excess-field recovery"
EXPECTED_SPLITS = ("A_val", "B_confirm")
MECHANISM_LABEL = "minimal_mechanism"
DEFAULT_ANALYSIS_ROOT = Path(
    os.environ.get(
        "EDNET_NULL_REFERENCED_RECOVERY_ROOT",
        "/data/datasets/KT4/outputs_KT4/null_referenced_downstream_recovery",
    )
)
PRIMARY_POINT_COLUMNS = (
    "split",
    "model",
    "evaluation_view",
    "supported_cells",
    "supported_occupancy_mass",
    "exact_null_sse_to_empirical",
    "model_sse_to_empirical",
    "primary_delta_sse_model_minus_exact_null",
    "null_relative_field_skill",
    "null_normalized_rmse_ratio",
    "M_null_relative_field_skill",
    "Psi_null_relative_field_skill",
    "null_referenced_correction_vs_empirical_excess_vector_corr",
    "null_referenced_correction_vs_empirical_excess_speed_corr",
    "null_referenced_correction_vs_empirical_excess_weighted_local_cosine",
    "excess_local_cosine_occupancy_coverage",
    "null_referenced_excess_amplitude_slope",
    "model_minus_diagonal_rescaled_scaffold_sse",
)
BOOTSTRAP_COLUMNS = (
    "model",
    "bootstrap_replicates",
    "primary_delta_sse_model_minus_exact_null_2p5",
    "primary_delta_sse_model_minus_exact_null_50",
    "primary_delta_sse_model_minus_exact_null_97p5",
    "null_relative_field_skill_2p5",
    "null_relative_field_skill_50",
    "null_relative_field_skill_97p5",
    "null_normalized_rmse_ratio_2p5",
    "null_normalized_rmse_ratio_50",
    "null_normalized_rmse_ratio_97p5",
    "model_minus_diagonal_rescaled_scaffold_sse_2p5",
    "model_minus_diagonal_rescaled_scaffold_sse_50",
    "model_minus_diagonal_rescaled_scaffold_sse_97p5",
)
GEOMETRY_COLUMNS = (
    "split",
    "field",
    "role",
    "negative_divergence_occupancy_fraction",
    "weighted_mean_divergence",
    "flow_weighted_shell_fraction_inward",
    "flow_weighted_shell_inward_cosine",
    "flow_core_to_shell_speed_ratio",
    "occupancy_core_to_shell_speed_ratio",
)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
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


def find_table(base_or_path: Path) -> Path:
    path = Path(base_or_path)
    if path.exists() and path.is_file():
        return path
    for extension in (".parquet", ".csv.gz", ".csv"):
        candidate = path.with_suffix(extension)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find table for {base_or_path}")


def find_optional_table(base_or_path: Path) -> Optional[Path]:
    try:
        return find_table(base_or_path)
    except FileNotFoundError:
        return None


def read_table(base_or_path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    path = find_table(base_or_path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    return pd.read_csv(
        path,
        usecols=list(columns) if columns is not None else None,
        low_memory=False,
    )


def read_optional_table(base_or_path: Path) -> Tuple[pd.DataFrame, Optional[Path]]:
    path = find_optional_table(base_or_path)
    if path is None:
        return pd.DataFrame(), None
    return read_table(path), path


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


def require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing required columns: {missing}")


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return number if np.isfinite(number) else default


def sanitize_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("_") or "model"


def human_model_label(label: str) -> str:
    value = str(label)
    if value == MECHANISM_LABEL:
        return "Minimal mechanism"
    seed_match = re.search(r"seed[_-]?(\d+)", value, flags=re.IGNORECASE)
    if seed_match:
        return f"Predictive-state Event-SSL (seed {seed_match.group(1)})"
    if "event" in value.lower() and "ssl" in value.lower():
        return "Predictive-state Event-SSL"
    return value.replace("_", " ")


def split_display(split: str) -> str:
    if split == "A_val":
        return r"$A_{\mathrm{val}}$"
    if split == "B_confirm":
        return r"$B_{\mathrm{confirm}}$"
    return str(split)


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    output = str(text)
    for source, target in replacements.items():
        output = output.replace(source, target)
    return output


def format_number(value: Any, digits: int = 4, scientific_threshold: float = 1e-3) -> str:
    number = finite_float(value)
    if not np.isfinite(number):
        return "--"
    absolute = abs(number)
    if absolute > 0 and (absolute < scientific_threshold or absolute >= 1e4):
        return f"{number:.{digits - 1}e}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def format_interval(low: Any, high: Any, digits: int = 4) -> str:
    lo = finite_float(low)
    hi = finite_float(high)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "--"
    return f"[{format_number(lo, digits)}, {format_number(hi, digits)}]"


def format_point_interval(point: Any, low: Any, high: Any, digits: int = 4) -> str:
    point_text = format_number(point, digits)
    interval = format_interval(low, high, digits)
    return point_text if interval == "--" else f"{point_text} {interval}"


def markdown_table(frame: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> str:
    selected = frame.copy() if columns is None else frame[[c for c in columns if c in frame.columns]].copy()
    try:
        return selected.to_markdown(index=False)
    except Exception:
        return selected.to_csv(index=False)


# -----------------------------------------------------------------------------
# Source and quality-gate bookkeeping
# -----------------------------------------------------------------------------
@dataclass
class SourceRecord:
    name: str
    path: str
    sha256: str
    rows: Optional[int]
    columns: Optional[int]


class SourceRegistry:
    def __init__(self) -> None:
        self.records: List[SourceRecord] = []

    def add_file(
        self,
        name: str,
        path: Path,
        frame: Optional[pd.DataFrame] = None,
    ) -> None:
        resolved = path.resolve()
        self.records.append(
            SourceRecord(
                name=name,
                path=str(resolved),
                sha256=file_sha256(resolved),
                rows=None if frame is None else int(len(frame)),
                columns=None if frame is None else int(len(frame.columns)),
            )
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([dataclasses.asdict(record) for record in self.records])


class GateCollector:
    def __init__(self, strict: bool) -> None:
        self.strict = bool(strict)
        self.rows: List[Dict[str, Any]] = []

    def add(
        self,
        gate: str,
        passed: bool,
        details: str,
        critical: bool = True,
    ) -> None:
        self.rows.append(
            {
                "gate": str(gate),
                "passed": bool(passed),
                "critical": bool(critical),
                "details": str(details),
            }
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def enforce(self) -> None:
        failed = [row for row in self.rows if row["critical"] and not row["passed"]]
        if failed and self.strict:
            message = "\n".join(f"- {row['gate']}: {row['details']}" for row in failed)
            raise RuntimeError(f"Critical supplementary-report quality gates failed:\n{message}")


# -----------------------------------------------------------------------------
# Input loading
# -----------------------------------------------------------------------------
@dataclass
class LoadedInputs:
    analysis_root: Path
    manifest_path: Path
    manifest: Dict[str, Any]
    primary_event_label: str
    point_metrics: pd.DataFrame
    cross_model: pd.DataFrame
    geometry: Dict[str, pd.DataFrame]
    bootstrap_summary: Dict[str, pd.DataFrame]
    bootstrap_replicates: Dict[str, pd.DataFrame]
    null_sensitivity: Dict[str, pd.DataFrame]
    matching_coverage: Dict[str, pd.DataFrame]
    opportunity_composition: Dict[str, pd.DataFrame]
    split_audits: Dict[str, Dict[str, Any]]
    sources: SourceRegistry


def manifest_table_path(manifest: Mapping[str, Any], key: str, fallback: Path) -> Path:
    raw = str(manifest.get(key, "") or "").strip()
    if raw:
        path = Path(raw)
        if path.exists():
            return path.resolve()
    return find_table(fallback).resolve()


def load_inputs(
    analysis_root: Path,
    primary_override: Optional[str],
    registry: SourceRegistry,
) -> LoadedInputs:
    root = analysis_root.resolve()
    manifest_path = root / "metadata" / "null_referenced_recovery_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Analysis manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    registry.add_file("analysis_manifest", manifest_path)

    primary_event_label = sanitize_label(
        primary_override or str(manifest.get("primary_event_ssl_label", "") or "")
    )
    if not primary_event_label:
        raise RuntimeError("The primary Event-SSL label is missing from the analysis manifest.")

    table_root = root / "tables"
    point_path = manifest_table_path(
        manifest,
        "point_metrics_table",
        table_root / "null_referenced_recovery_metrics_all_splits",
    )
    point_metrics = read_table(point_path)
    registry.add_file("point_metrics_all_splits", point_path, point_metrics)

    cross_path = find_optional_table(table_root / "cross_model_null_referenced_metrics_all_splits_descriptive")
    if cross_path is not None:
        cross_model = read_table(cross_path)
        registry.add_file("cross_model_descriptive", cross_path, cross_model)
    else:
        cross_model = pd.DataFrame()

    geometry: Dict[str, pd.DataFrame] = {}
    bootstrap_summary: Dict[str, pd.DataFrame] = {}
    bootstrap_replicates: Dict[str, pd.DataFrame] = {}
    null_sensitivity: Dict[str, pd.DataFrame] = {}
    matching_coverage: Dict[str, pd.DataFrame] = {}
    opportunity_composition: Dict[str, pd.DataFrame] = {}
    split_audits: Dict[str, Dict[str, Any]] = {}

    for split in EXPECTED_SPLITS:
        frame, path = read_optional_table(table_root / f"{split}_null_referenced_geometry_metrics")
        geometry[split] = frame
        if path is not None:
            registry.add_file(f"{split}_geometry", path, frame)

        frame, path = read_optional_table(table_root / f"{split}_user_multiplier_bootstrap_summary")
        if not frame.empty:
            frame = frame.copy()
            frame["split"] = split
        bootstrap_summary[split] = frame
        if path is not None:
            registry.add_file(f"{split}_bootstrap_summary", path, frame)

        frame, path = read_optional_table(table_root / f"{split}_user_multiplier_bootstrap_replicates")
        if not frame.empty:
            frame = frame.copy()
            frame["split"] = split
        bootstrap_replicates[split] = frame
        if path is not None:
            registry.add_file(f"{split}_bootstrap_replicates", path, frame)

        frame, path = read_optional_table(table_root / f"{split}_exact_vs_finite_null_mean_sensitivity")
        null_sensitivity[split] = frame
        if path is not None:
            registry.add_file(f"{split}_exact_vs_finite_null_sensitivity", path, frame)

        frame, path = read_optional_table(table_root / f"{split}_matching_fallback_coverage")
        matching_coverage[split] = frame
        if path is not None:
            registry.add_file(f"{split}_matching_coverage", path, frame)

        frame, path = read_optional_table(table_root / f"{split}_opportunity_composition_audit")
        opportunity_composition[split] = frame
        if path is not None:
            registry.add_file(f"{split}_opportunity_composition", path, frame)

        audit_path = root / "metadata" / f"{split}_null_referenced_recovery_audit.json"
        if audit_path.exists():
            split_audits[split] = load_json(audit_path)
            registry.add_file(f"{split}_analysis_audit", audit_path)
        else:
            split_audits[split] = {}

    return LoadedInputs(
        analysis_root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        primary_event_label=primary_event_label,
        point_metrics=point_metrics,
        cross_model=cross_model,
        geometry=geometry,
        bootstrap_summary=bootstrap_summary,
        bootstrap_replicates=bootstrap_replicates,
        null_sensitivity=null_sensitivity,
        matching_coverage=matching_coverage,
        opportunity_composition=opportunity_composition,
        split_audits=split_audits,
        sources=registry,
    )


# -----------------------------------------------------------------------------
# Scientific contract validation
# -----------------------------------------------------------------------------
def nested_max_abs(mapping: Mapping[str, Any], prefix: str = "max_abs_") -> float:
    values: List[float] = []
    for key, value in mapping.items():
        if str(key).startswith(prefix):
            number = finite_float(value)
            if np.isfinite(number):
                values.append(abs(number))
    return max(values) if values else float("nan")


def validate_inputs(
    data: LoadedInputs,
    gates: GateCollector,
    min_bootstrap_replicates: int,
    require_bootstrap: bool,
) -> None:
    manifest = data.manifest
    gates.add(
        "analysis identity",
        manifest.get("analysis_name") == EXPECTED_ANALYSIS_NAME,
        f"analysis_name={manifest.get('analysis_name')!r}",
    )
    declared_splits = tuple(str(value) for value in manifest.get("splits", []))
    gates.add(
        "validation and confirmation splits present",
        all(split in declared_splits for split in EXPECTED_SPLITS),
        f"declared_splits={declared_splits}",
    )
    gates.add(
        "primary Event-SSL label present",
        data.primary_event_label in set(data.point_metrics.get("model", pd.Series(dtype=str)).astype(str)),
        f"primary_event_ssl_label={data.primary_event_label}",
    )

    guardrails = dict(manifest.get("guardrails", {}))
    forbidden_true = (
        "model_training",
        "model_selection",
        "mechanism_refit",
        "coordinate_refit",
        "grid_or_mask_refit",
        "construction_null_protocol_changed",
        "learned_plane_used_for_null_subtraction",
        "cross_model_correction_correlation_used_as_primary_test",
        "B_confirm_used_for_update",
    )
    violated = [name for name in forbidden_true if bool(guardrails.get(name, False))]
    gates.add(
        "global no-update guardrails",
        not violated,
        "none violated" if not violated else f"violated={violated}",
    )

    point = data.point_metrics.copy()
    require_columns(point, PRIMARY_POINT_COLUMNS, "combined point-metrics table")
    point["split"] = point["split"].astype(str)
    point["model"] = point["model"].astype(str)
    gates.add(
        "empirical-anchor evaluation only",
        set(point["evaluation_view"].dropna().astype(str)) == {"empirical_current_state_anchor"},
        f"views={sorted(set(point['evaluation_view'].dropna().astype(str)))}",
    )

    primary_models = {MECHANISM_LABEL, data.primary_event_label}
    for split in EXPECTED_SPLITS:
        subset = point[(point["split"] == split) & (point["model"].isin(primary_models))].copy()
        gates.add(
            f"{split} primary models available",
            set(subset["model"]) == primary_models and len(subset) == 2,
            f"models={sorted(set(subset['model']))}, rows={len(subset)}",
        )
        if subset.empty:
            continue
        null_values = numeric(subset, "exact_null_sse_to_empirical").to_numpy(dtype=float)
        cell_values = numeric(subset, "supported_cells").to_numpy(dtype=float)
        null_spread = float(np.nanmax(null_values) - np.nanmin(null_values))
        cell_match = bool(np.all(cell_values == cell_values[0])) if len(cell_values) else False
        gates.add(
            f"{split} common exact-null estimand",
            np.isfinite(null_values).all() and null_spread <= 1e-12,
            f"null_sse_spread={null_spread:.3e}",
        )
        gates.add(
            f"{split} common frozen support",
            cell_match,
            f"supported_cells={cell_values.tolist()}",
        )
        consistency = []
        for row in subset.itertuples(index=False):
            delta = finite_float(getattr(row, "primary_delta_sse_model_minus_exact_null"))
            flag = bool(getattr(row, "model_better_than_exact_null_point_estimate", delta < 0))
            consistency.append(flag == (delta < 0))
        gates.add(
            f"{split} point-estimate flags consistent",
            all(consistency),
            f"checks={consistency}",
        )

    b_boot = data.bootstrap_summary.get("B_confirm", pd.DataFrame()).copy()
    if require_bootstrap:
        gates.add(
            "B_confirm multiplier-bootstrap summary available",
            not b_boot.empty,
            "missing" if b_boot.empty else f"rows={len(b_boot)}",
        )
    if not b_boot.empty:
        require_columns(b_boot, BOOTSTRAP_COLUMNS, "B_confirm bootstrap summary")
        selected = b_boot[b_boot["model"].astype(str).isin(primary_models)]
        gates.add(
            "B_confirm bootstrap covers both primary models",
            set(selected["model"].astype(str)) == primary_models and len(selected) == 2,
            f"models={sorted(set(selected['model'].astype(str)))}",
        )
        minimum = int(pd.to_numeric(selected["bootstrap_replicates"], errors="coerce").min()) if not selected.empty else 0
        gates.add(
            "B_confirm bootstrap replicate count",
            minimum >= int(min_bootstrap_replicates),
            f"minimum={minimum}, required={min_bootstrap_replicates}",
        )

    for split in EXPECTED_SPLITS:
        geometry = data.geometry.get(split, pd.DataFrame())
        gates.add(
            f"{split} geometry table available",
            not geometry.empty,
            "missing" if geometry.empty else f"rows={len(geometry)}",
        )
        if not geometry.empty:
            require_columns(geometry, GEOMETRY_COLUMNS, f"{split} geometry table")

        audit = data.split_audits.get(split, {})
        gates.add(
            f"{split} analysis audit available",
            bool(audit),
            "missing" if not audit else "available",
        )
        if not audit:
            continue
        split_guardrails = dict(audit.get("guardrails", {}))
        split_violations = [name for name, value in split_guardrails.items() if bool(value)]
        gates.add(
            f"{split} split-level no-update guardrails",
            not split_violations,
            "none violated" if not split_violations else f"violated={split_violations}",
        )
        protocol = dict(audit.get("existing_construction_null_protocol_audit", {}))
        finite_null = dict(audit.get("existing_finite_null_audit", {}))
        gates.add(
            f"{split} archived construction-null protocol audited",
            bool(protocol.get("available", False)),
            f"available={protocol.get('available')}",
        )
        gates.add(
            f"{split} finite-replicate null retained for sensitivity",
            bool(finite_null.get("available", False)),
            f"available={finite_null.get('available')}",
            critical=False,
        )
        sparse_audit = dict(audit.get("sparse_bootstrap_reproduction_audit", {}))
        if sparse_audit and not bool(sparse_audit.get("skipped", False)):
            maximum = nested_max_abs(sparse_audit)
            gates.add(
                f"{split} user-cell sufficient statistics reproduce point fields",
                np.isfinite(maximum) and maximum <= 1e-10,
                f"max_abs_difference={maximum:.3e}",
            )
        mechanism_audit = dict(audit.get("mechanism_audit", {}))
        if mechanism_audit:
            hash_ok = (
                mechanism_audit.get("parameter_hash_before") == mechanism_audit.get("parameter_hash_after")
                and mechanism_audit.get("calibration_hash_before") == mechanism_audit.get("calibration_hash_after")
            )
            gates.add(
                f"{split} frozen mechanism unchanged",
                hash_ok,
                "parameter and calibration hashes unchanged" if hash_ok else "hash mismatch",
            )
        event_audits = dict(audit.get("event_ssl_audits", {}))
        primary_event_audit = dict(event_audits.get(data.primary_event_label, {}))
        manifest_audit = dict(primary_event_audit.get("manifest", {}))
        if primary_event_audit:
            model_kind = str(manifest_audit.get("model_kind", "") or "")
            gates.add(
                f"{split} primary Event-SSL is predictive-state model",
                model_kind == "predictive_state",
                f"model_kind={model_kind!r}",
            )
            state_audit = dict(primary_event_audit.get("state_target_alignment", {}))
            maximum = nested_max_abs(state_audit)
            gates.add(
                f"{split} Event-SSL rows and empirical targets align",
                np.isfinite(maximum) and maximum <= 2e-6,
                f"max_abs_difference={maximum:.3e}",
            )

    b_rows = point[(point["split"] == "B_confirm") & (point["model"].isin(primary_models))]
    if not b_rows.empty:
        source = set(b_rows.get("scaffold_rescaling_source_split", pd.Series(dtype=str)).dropna().astype(str))
        gates.add(
            "B_confirm scaffold benchmark frozen from A_val",
            source == {"A_val"},
            f"source_splits={sorted(source)}",
        )


# -----------------------------------------------------------------------------
# Table construction
# -----------------------------------------------------------------------------
def bootstrap_lookup(data: LoadedInputs) -> pd.DataFrame:
    frames = [frame for frame in data.bootstrap_summary.values() if not frame.empty]
    if not frames:
        return pd.DataFrame()
    output = pd.concat(frames, ignore_index=True, sort=False)
    output["split"] = output["split"].astype(str)
    output["model"] = output["model"].astype(str)
    return output


def build_table1(data: LoadedInputs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    primary_models = [MECHANISM_LABEL, data.primary_event_label]
    point = data.point_metrics[
        data.point_metrics["split"].astype(str).isin(EXPECTED_SPLITS)
        & data.point_metrics["model"].astype(str).isin(primary_models)
    ].copy()
    point["split"] = point["split"].astype(str)
    point["model"] = point["model"].astype(str)
    bootstrap = bootstrap_lookup(data)
    if not bootstrap.empty:
        merge_columns = ["split", "model"] + [
            column for column in BOOTSTRAP_COLUMNS if column not in {"model"} and column in bootstrap.columns
        ]
        point = point.merge(
            bootstrap[merge_columns],
            how="left",
            on=["split", "model"],
            validate="one_to_one",
        )

    point["analysis_role"] = point["split"].map(
        {
            "A_val": "development/validation; descriptive consistency",
            "B_confirm": "post-freeze output-only confirmation",
        }
    )
    point["model_display"] = point["model"].map(human_model_label)
    point["exact_null_weighted_rmse"] = np.sqrt(
        np.maximum(numeric(point, "exact_null_sse_to_empirical"), 0.0)
    )
    point["model_weighted_rmse"] = np.sqrt(
        np.maximum(numeric(point, "model_sse_to_empirical"), 0.0)
    )
    point["bootstrap_exact_null_success"] = (
        numeric(point, "primary_delta_sse_model_minus_exact_null_97p5") < 0.0
    )
    point["point_exact_null_success"] = (
        numeric(point, "primary_delta_sse_model_minus_exact_null") < 0.0
    )
    point["row_order"] = point["split"].map({"A_val": 0, "B_confirm": 1}) * 10 + point["model"].map(
        {MECHANISM_LABEL: 0, data.primary_event_label: 1}
    ).fillna(9)
    point = point.sort_values("row_order", kind="mergesort").reset_index(drop=True)

    display_rows: List[Dict[str, Any]] = []
    for row in point.itertuples(index=False):
        display_rows.append(
            {
                "Split": split_display(str(row.split)),
                "Model": human_model_label(str(row.model)),
                "Cells": int(finite_float(row.supported_cells, 0)),
                "$D_0$": format_number(row.exact_null_weighted_rmse),
                "$D_k$": format_number(row.model_weighted_rmse),
                "$\\Delta D^2$ [95%]": format_point_interval(
                    row.primary_delta_sse_model_minus_exact_null,
                    getattr(row, "primary_delta_sse_model_minus_exact_null_2p5", np.nan),
                    getattr(row, "primary_delta_sse_model_minus_exact_null_97p5", np.nan),
                ),
                "$S_k$": format_number(row.null_relative_field_skill),
                "$R_k$": format_number(row.null_normalized_rmse_ratio),
                "$S_M$": format_number(row.M_null_relative_field_skill),
                "$S_{\\Psi}$": format_number(row.Psi_null_relative_field_skill),
            }
        )
    display = pd.DataFrame(display_rows)
    return point, display


def build_table2_specificity(data: LoadedInputs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    primary_models = [MECHANISM_LABEL, data.primary_event_label]
    point = data.point_metrics[
        (data.point_metrics["split"].astype(str) == "B_confirm")
        & data.point_metrics["model"].astype(str).isin(primary_models)
    ].copy()
    point["split"] = "B_confirm"
    point["model"] = point["model"].astype(str)
    bootstrap = data.bootstrap_summary.get("B_confirm", pd.DataFrame()).copy()
    if not bootstrap.empty:
        bootstrap["model"] = bootstrap["model"].astype(str)
        merge_columns = [
            "model",
            "model_minus_diagonal_rescaled_scaffold_sse_2p5",
            "model_minus_diagonal_rescaled_scaffold_sse_50",
            "model_minus_diagonal_rescaled_scaffold_sse_97p5",
        ]
        point = point.merge(
            bootstrap[[column for column in merge_columns if column in bootstrap.columns]],
            how="left",
            on="model",
            validate="one_to_one",
        )
    point["model_display"] = point["model"].map(human_model_label)
    point["diagonal_scaffold_success_point"] = (
        numeric(point, "model_minus_diagonal_rescaled_scaffold_sse") < 0.0
    )
    point["diagonal_scaffold_success_97p5"] = (
        numeric(point, "model_minus_diagonal_rescaled_scaffold_sse_97p5") < 0.0
    )
    point["row_order"] = point["model"].map(
        {MECHANISM_LABEL: 0, data.primary_event_label: 1}
    ).fillna(9)
    point = point.sort_values("row_order", kind="mergesort").reset_index(drop=True)

    display_rows: List[Dict[str, Any]] = []
    for row in point.itertuples(index=False):
        display_rows.append(
            {
                "Model": human_model_label(str(row.model)),
                "$\\Delta D^2_{\\mathrm{diag}}$ [95%]": format_point_interval(
                    row.model_minus_diagonal_rescaled_scaffold_sse,
                    getattr(row, "model_minus_diagonal_rescaled_scaffold_sse_2p5", np.nan),
                    getattr(row, "model_minus_diagonal_rescaled_scaffold_sse_97p5", np.nan),
                ),
                "$r_{\\mathrm{exc}}$": format_number(
                    row.null_referenced_correction_vs_empirical_excess_vector_corr
                ),
                "$r_{\\lVert b\\rVert}$": format_number(
                    row.null_referenced_correction_vs_empirical_excess_speed_corr
                ),
                "$c_{\\mathrm{local}}$": format_number(
                    row.null_referenced_correction_vs_empirical_excess_weighted_local_cosine
                ),
                "Occupancy coverage": format_number(row.excess_local_cosine_occupancy_coverage),
                "$\\beta_{\\mathrm{exc}}$": format_number(
                    row.null_referenced_excess_amplitude_slope
                ),
            }
        )
    return point, pd.DataFrame(display_rows)


def build_table2_geometry(data: LoadedInputs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    geometry = data.geometry.get("B_confirm", pd.DataFrame()).copy()
    if geometry.empty:
        return geometry, pd.DataFrame()
    target_fields = [
        "empirical_excess",
        f"{MECHANISM_LABEL}_null_referenced_correction",
        f"{data.primary_event_label}_null_referenced_correction",
    ]
    geometry = geometry[geometry["field"].astype(str).isin(target_fields)].copy()
    labels = {
        "empirical_excess": "Empirical excess field",
        f"{MECHANISM_LABEL}_null_referenced_correction": "Minimal-mechanism correction",
        f"{data.primary_event_label}_null_referenced_correction": "Event-SSL correction",
    }
    geometry["field_display"] = geometry["field"].astype(str).map(labels)
    order = {field: index for index, field in enumerate(target_fields)}
    geometry["row_order"] = geometry["field"].astype(str).map(order)
    geometry = geometry.sort_values("row_order", kind="mergesort").reset_index(drop=True)

    display_rows: List[Dict[str, Any]] = []
    for row in geometry.itertuples(index=False):
        display_rows.append(
            {
                "Field": str(row.field_display),
                "$f_{\\nabla\\!\\cdot b<0}$": format_number(
                    row.negative_divergence_occupancy_fraction
                ),
                "$\\langle\\nabla\\!\\cdot b\\rangle$": format_number(
                    row.weighted_mean_divergence
                ),
                "$f_{\\mathrm{in}}$": format_number(
                    row.flow_weighted_shell_fraction_inward
                ),
                "$c_{\\mathrm{in}}$": format_number(
                    row.flow_weighted_shell_inward_cosine
                ),
                "$R_{\\mathrm{core/shell}}$": format_number(
                    row.flow_core_to_shell_speed_ratio
                ),
            }
        )
    return geometry, pd.DataFrame(display_rows)


# -----------------------------------------------------------------------------
# Repository-only supplementary data
# -----------------------------------------------------------------------------
def event_seed_level_table(data: LoadedInputs) -> pd.DataFrame:
    point = data.point_metrics.copy()
    declared_roots = dict(data.manifest.get("event_ssl_roots", {}))
    declared_labels = {sanitize_label(label) for label in declared_roots}
    if declared_labels:
        event_rows = point[point["model"].astype(str).isin(declared_labels)].copy()
    else:
        event_rows = point[point["model"].astype(str) != MECHANISM_LABEL].copy()
    if event_rows.empty:
        return event_rows
    event_rows["model_display"] = event_rows["model"].astype(str).map(human_model_label)
    seed = event_rows["model"].astype(str).str.extract(r"seed[_-]?(\d+)", flags=re.IGNORECASE)[0]
    event_rows["seed"] = pd.to_numeric(seed, errors="coerce").astype("Int64")
    keep = [
        "split",
        "model",
        "model_display",
        "seed",
        "supported_cells",
        "primary_delta_sse_model_minus_exact_null",
        "null_relative_field_skill",
        "null_normalized_rmse_ratio",
        "model_minus_diagonal_rescaled_scaffold_sse",
        "null_referenced_correction_vs_empirical_excess_vector_corr",
        "null_referenced_correction_vs_empirical_excess_speed_corr",
        "null_referenced_correction_vs_empirical_excess_weighted_local_cosine",
        "null_referenced_excess_amplitude_slope",
        "M_null_relative_field_skill",
        "Psi_null_relative_field_skill",
    ]
    return event_rows[[column for column in keep if column in event_rows.columns]].sort_values(
        ["split", "seed", "model"], kind="mergesort"
    ).reset_index(drop=True)


def t_interval(values: np.ndarray) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2 or student_t is None:
        return float("nan"), float("nan")
    mean = float(np.mean(array))
    standard_error = float(np.std(array, ddof=1) / math.sqrt(array.size))
    critical = float(student_t.ppf(0.975, df=array.size - 1))
    return mean - critical * standard_error, mean + critical * standard_error


def event_seed_summary(seed_level: pd.DataFrame) -> pd.DataFrame:
    if seed_level.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    metrics = [
        "primary_delta_sse_model_minus_exact_null",
        "null_relative_field_skill",
        "null_normalized_rmse_ratio",
        "model_minus_diagonal_rescaled_scaffold_sse",
        "null_referenced_correction_vs_empirical_excess_vector_corr",
        "null_referenced_correction_vs_empirical_excess_weighted_local_cosine",
        "null_referenced_excess_amplitude_slope",
    ]
    for split, group in seed_level.groupby("split", sort=False):
        row: Dict[str, Any] = {
            "split": str(split),
            "seed_count": int(group["model"].nunique()),
            "all_seeds_better_than_exact_null": bool(
                np.all(numeric(group, "primary_delta_sse_model_minus_exact_null") < 0.0)
            ),
            "all_seeds_better_than_diagonal_scaffold_point_estimate": bool(
                np.all(numeric(group, "model_minus_diagonal_rescaled_scaffold_sse") < 0.0)
            )
            if "model_minus_diagonal_rescaled_scaffold_sse" in group.columns
            else False,
            "reporting_role": (
                "machine-readable new-metric seed robustness; not a third typeset table because "
                "the manuscript already contains a comprehensive Event-SSL seed table"
            ),
        }
        for metric in metrics:
            values = numeric(group, metric).to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            low, high = t_interval(values)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
            row[f"{metric}_t95_low"] = low
            row[f"{metric}_t95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def combine_optional_tables(mapping: Mapping[str, pd.DataFrame], split_column: bool = True) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for split, frame in mapping.items():
        if frame.empty:
            continue
        current = frame.copy()
        if split_column and "split" not in current.columns:
            current["split"] = split
        frames.append(current)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def null_sensitivity_summary(data: LoadedInputs) -> pd.DataFrame:
    full = combine_optional_tables(data.null_sensitivity)
    if full.empty:
        return pd.DataFrame()
    require_columns(
        full,
        [
            "split",
            "model",
            "null_baseline",
            "primary_delta_sse_model_minus_null",
            "null_relative_field_skill",
            "excess_vector_corr",
        ],
        "exact-versus-finite null sensitivity",
    )
    rows: List[Dict[str, Any]] = []
    for (split, model), group in full.groupby(["split", "model"], sort=False):
        indexed = group.set_index("null_baseline")
        exact_key = "exact_cyclic_shift_expectation"
        finite_key = "existing_finite_replicate_mean"
        if exact_key not in indexed.index or finite_key not in indexed.index:
            continue
        exact = indexed.loc[exact_key]
        finite = indexed.loc[finite_key]
        if isinstance(exact, pd.DataFrame):
            exact = exact.iloc[0]
        if isinstance(finite, pd.DataFrame):
            finite = finite.iloc[0]
        rows.append(
            {
                "split": split,
                "model": model,
                "delta_sse_exact": finite_float(exact["primary_delta_sse_model_minus_null"]),
                "delta_sse_finite_mean": finite_float(finite["primary_delta_sse_model_minus_null"]),
                "absolute_delta_sse_difference": abs(
                    finite_float(exact["primary_delta_sse_model_minus_null"])
                    - finite_float(finite["primary_delta_sse_model_minus_null"])
                ),
                "skill_exact": finite_float(exact["null_relative_field_skill"]),
                "skill_finite_mean": finite_float(finite["null_relative_field_skill"]),
                "absolute_skill_difference": abs(
                    finite_float(exact["null_relative_field_skill"])
                    - finite_float(finite["null_relative_field_skill"])
                ),
                "excess_corr_exact": finite_float(exact["excess_vector_corr"]),
                "excess_corr_finite_mean": finite_float(finite["excess_vector_corr"]),
                "absolute_excess_corr_difference": abs(
                    finite_float(exact["excess_vector_corr"])
                    - finite_float(finite["excess_vector_corr"])
                ),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# LaTeX and Markdown report generation
# -----------------------------------------------------------------------------
def latex_row(values: Sequence[str]) -> str:
    return " & ".join(values) + r" \\"


def build_latex_tables(
    table1_display: pd.DataFrame,
    table2a_display: pd.DataFrame,
    table2b_display: pd.DataFrame,
) -> str:
    lines: List[str] = [
        "% Generated by extract_null_referenced_downstream_recovery_supplementary_statistics.py",
        "% Requires \\usepackage{booktabs}.",
        "",
        r"\begin{table*}[t]",
        r"\caption{Construction-null-referenced recovery of the frozen downstream models. $D_0$ is the occupancy-weighted root-mean-square distance between the exact construction-null mean field and the empirical field; $D_k$ is the corresponding distance for model $k$. $\Delta D^2=D_k^2-D_0^2$, $S_k=1-D_k^2/D_0^2$, and $R_k=D_k/D_0$. Negative $\Delta D^2$ and positive $S_k$ indicate improvement over the exact matched-accounting mean. Brackets give the paired user-level multiplier-bootstrap 95\% interval where available; validation rows are descriptive and confirmation is post-freeze and output-only. $S_M$ and $S_{\Psi}$ are coordinate-specific null-relative skills.}",
        r"\label{tab:supp_null_referenced_recovery}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.1pt}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        latex_row([
            "Split",
            "Model",
            "Cells",
            "$D_0$",
            "$D_k$",
            "$\\Delta D^2$ [95\\%]",
            "$S_k$",
            "$R_k$",
            "$S_M$",
            "$S_{\\Psi}$",
        ]),
        r"\midrule",
    ]
    for row in table1_display.to_dict(orient="records"):
        lines.append(
            latex_row(
                [
                    str(row["Split"]),
                    latex_escape(str(row["Model"])),
                    str(row["Cells"]),
                    str(row["$D_0$"]),
                    str(row["$D_k$"]),
                    str(row["$\\Delta D^2$ [95%]"]),
                    str(row["$S_k$"]),
                    str(row["$R_k$"]),
                    str(row["$S_M$"]),
                    str(row["$S_{\\Psi}$"]),
                ]
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\caption{Specificity and geometry of construction-excess recovery on the output-only confirmation cohort. Panel a compares each model with the stronger per-coordinate construction-scaffold rescaling fitted on $A_{\mathrm{val}}$ and frozen before confirmation; $\Delta D^2_{\mathrm{diag}}<0$ favours the model. The remaining quantities compare the model correction $b_k-b_0$ with the empirical excess field $b-b_0$ and are effect-size diagnostics rather than independent tests because both subtract the same $b_0$. Panel b evaluates global supported-stencil contraction together with the unchanged training-defined core and shell for the empirical excess field and model corrections. These basin diagnostics are descriptive; the original construction-null permutation inference remains reported separately.}",
            r"\label{tab:supp_null_referenced_specificity}",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\textbf{a. Specificity beyond an amplitude-rescaled construction scaffold}\\[2pt]",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            latex_row([
                "Model",
                "$\\Delta D^2_{\\mathrm{diag}}$ [95\\%]",
                "$r_{\\mathrm{exc}}$",
                "$r_{\\lVert b\\rVert}$",
                "$c_{\\mathrm{local}}$",
                "Coverage",
                "$\\beta_{\\mathrm{exc}}$",
            ]),
            r"\midrule",
        ]
    )
    for row in table2a_display.to_dict(orient="records"):
        lines.append(
            latex_row(
                [
                    latex_escape(str(row["Model"])),
                    str(row["$\\Delta D^2_{\\mathrm{diag}}$ [95%]"]),
                    str(row["$r_{\\mathrm{exc}}$"]),
                    str(row["$r_{\\lVert b\\rVert}$"]),
                    str(row["$c_{\\mathrm{local}}$"]),
                    str(row["Occupancy coverage"]),
                    str(row["$\\beta_{\\mathrm{exc}}$"]),
                ]
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{5pt}",
            r"\textbf{b. Frozen-support and core/shell geometry of the empirical excess and model corrections}\\[2pt]",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            latex_row([
                "Field",
                "$f_{\\nabla\\!\\cdot b<0}$",
                "$\\langle\\nabla\\!\\cdot b\\rangle$",
                "$f_{\\mathrm{in}}$",
                "$c_{\\mathrm{in}}$",
                "$R_{\\mathrm{core/shell}}$",
            ]),
            r"\midrule",
        ]
    )
    for row in table2b_display.to_dict(orient="records"):
        lines.append(
            latex_row(
                [
                    latex_escape(str(row["Field"])),
                    str(row["$f_{\\nabla\\!\\cdot b<0}$"]),
                    str(row["$\\langle\\nabla\\!\\cdot b\\rangle$"]),
                    str(row["$f_{\\mathrm{in}}$"]),
                    str(row["$c_{\\mathrm{in}}$"]),
                    str(row["$R_{\\mathrm{core/shell}}$"]),
                ]
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def narrative_lines(
    table1_full: pd.DataFrame,
    table2a_full: pd.DataFrame,
    table2b_full: pd.DataFrame,
    seed_summary: pd.DataFrame,
) -> List[str]:
    lines: List[str] = []
    b_confirm = table1_full[table1_full["split"] == "B_confirm"]
    for row in b_confirm.itertuples(index=False):
        low = getattr(row, "primary_delta_sse_model_minus_exact_null_2p5", np.nan)
        high = getattr(row, "primary_delta_sse_model_minus_exact_null_97p5", np.nan)
        lines.append(
            f"- **{human_model_label(str(row.model))}:** weighted field RMSE changed from "
            f"{format_number(row.exact_null_weighted_rmse)} for the exact construction-null mean "
            f"to {format_number(row.model_weighted_rmse)} for the frozen model. The paired squared-error "
            f"difference was {format_point_interval(row.primary_delta_sse_model_minus_exact_null, low, high)}, "
            f"with null-relative skill {format_number(row.null_relative_field_skill)} and RMSE ratio "
            f"{format_number(row.null_normalized_rmse_ratio)}."
        )
    for row in table2a_full.itertuples(index=False):
        low = getattr(row, "model_minus_diagonal_rescaled_scaffold_sse_2p5", np.nan)
        high = getattr(row, "model_minus_diagonal_rescaled_scaffold_sse_97p5", np.nan)
        lines.append(
            f"- **{human_model_label(str(row.model))} specificity:** relative to the $A_{{\\mathrm{{val}}}}$-fitted "
            f"per-coordinate scaffold rescaling, the squared-error difference was "
            f"{format_point_interval(row.model_minus_diagonal_rescaled_scaffold_sse, low, high)}. "
            f"Its null-referenced correction had excess-field vector correlation "
            f"{format_number(row.null_referenced_correction_vs_empirical_excess_vector_corr)}, weighted local cosine "
            f"{format_number(row.null_referenced_correction_vs_empirical_excess_weighted_local_cosine)}, and amplitude slope "
            f"{format_number(row.null_referenced_excess_amplitude_slope)}."
        )
    if not seed_summary.empty:
        for row in seed_summary.itertuples(index=False):
            lines.append(
                f"- **Event-SSL seed-level supplementary data ({row.split}):** {int(row.seed_count)} frozen "
                f"pipelines were available; all improved over the exact null: "
                f"{bool(row.all_seeds_better_than_exact_null)}. These new-metric seed values remain machine-readable "
                f"rather than forming a third typeset table."
            )
    return lines


def build_markdown_report(
    table1_display: pd.DataFrame,
    table2a_display: pd.DataFrame,
    table2b_display: pd.DataFrame,
    table1_full: pd.DataFrame,
    table2a_full: pd.DataFrame,
    table2b_full: pd.DataFrame,
    seed_level: pd.DataFrame,
    seed_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    quality_gates: pd.DataFrame,
    source_audit: pd.DataFrame,
) -> str:
    lines: List[str] = [
        "# Construction-null-referenced downstream recovery: Supplementary Information numerical report",
        "",
        "## Typeset inclusion boundary",
        "",
        "Use exactly two supplementary tables. The display tables contain only the new null-referenced quantities needed to answer whether the frozen mechanism and predictive-state Event-SSL recover empirical dynamics beyond the matched-accounting scaffold.",
        "",
        "The following are deliberately not repeated in the display tables because they are already shown or reported in the main manuscript or the existing construction-null/random-seed sections: raw model-versus-empirical drift correlations, next-state occupancy divergence, transition metrics, original construction-null Monte Carlo values, matching fractions, raw cross-model field agreement, and the established six-seed structural-recovery panel.",
        "",
        "## Supplementary Table 1 — Primary null-referenced field recovery",
        "",
        markdown_table(table1_display),
        "",
        "## Supplementary Table 2a — Specificity beyond a rescaled scaffold",
        "",
        markdown_table(table2a_display),
        "",
        "## Supplementary Table 2b — Frozen-support and core/shell geometry",
        "",
        markdown_table(table2b_display),
        "",
        "## Narrative-ready numerical statements",
        "",
        *narrative_lines(table1_full, table2a_full, table2b_full, seed_summary),
        "",
        "## Interpretation boundaries",
        "",
        "- The exact cyclic-shift expectation is used only as the deterministic mean field for decomposition; the original finite permutation ensemble remains the source of the construction-null Monte Carlo tests.",
        "- The primary endpoint is the paired occupancy-weighted squared-field-error difference, model minus exact null. Correction-versus-excess correlations and cosines share the same subtracted null field and are secondary effect-size diagnostics.",
        "- The per-coordinate rescaled scaffold is fitted on validation only, has no intercept and no sign reversal, and is frozen before confirmation. It is a stronger specificity benchmark, not a second formal null distribution.",
        "- Confirmation is output-only and post-freeze, but this reviewer-motivated derived endpoint should not be described as preregistered.",
        "- User-level multiplier intervals are conditional on the frozen matching groups and exact row-level donor expectations; donor pools are not rebuilt inside each multiplier replicate.",
        "- Only empirical-anchor model fields enter the null subtraction. Learned-plane dynamics remain a separate Event-SSL self-consistency analysis.",
        "- Frozen-core geometry is descriptive and should not be presented as a new multiplicity-adjusted basin test.",
        "",
        "## Repository-only extended outputs",
        "",
        f"- Event-SSL null-referenced seed-level rows: {len(seed_level)}.",
        f"- Event-SSL seed-summary rows: {len(seed_summary)}.",
        f"- Exact-versus-finite null-sensitivity rows: {len(sensitivity_summary)}.",
        "- Cross-model correction agreement, complete point metrics, bootstrap replicates, matching coverage, opportunity composition, and source hashes remain machine-readable and are not typeset.",
        "",
        "## Scientific quality gates",
        "",
        markdown_table(quality_gates),
        "",
        "## Source audit",
        "",
        markdown_table(source_audit),
        "",
        "## Suggested captions",
        "",
        "**Supplementary Table X | Construction-null-referenced recovery of frozen downstream models.** The exact construction-null mean, frozen minimal mechanism and predictive-state Event-SSL were aggregated from identical empirical current-state anchors, Stage-1 user weights, grid cells and support. The primary quantity is the paired occupancy-weighted squared-field-error difference relative to the exact matched-accounting mean. Validation is descriptive; confirmation is post-freeze and output-only. Brackets report paired user-level multiplier-bootstrap intervals where available.",
        "",
        "**Supplementary Table Y | Specificity and frozen-core geometry of construction-excess recovery.** Panel a compares each frozen model with a per-coordinate amplitude rescaling of the construction scaffold fitted on validation and frozen before confirmation, then reports secondary alignment of the model correction with the empirical excess field. Panel b evaluates the empirical excess and model corrections under the unchanged training-defined core and shell. Geometry values are descriptive and do not replace the original construction-null permutation tests.",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------
def make_synthetic_analysis_root(root: Path) -> Path:
    table_root = root / "tables"
    metadata_root = root / "metadata"
    table_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    primary = "event_ssl_seed42"

    rows: List[Dict[str, Any]] = []
    for split, null_sse in (("A_val", 0.0100), ("B_confirm", 0.0120)):
        for model, model_sse in ((MECHANISM_LABEL, 0.0030), (primary, 0.0060)):
            rows.append(
                {
                    "split": split,
                    "model": model,
                    "evaluation_view": "empirical_current_state_anchor",
                    "supported_cells": 940 if split == "A_val" else 954,
                    "supported_occupancy_mass": 0.98,
                    "exact_null_sse_to_empirical": null_sse,
                    "model_sse_to_empirical": model_sse,
                    "primary_delta_sse_model_minus_exact_null": model_sse - null_sse,
                    "null_relative_field_skill": 1.0 - model_sse / null_sse,
                    "null_normalized_rmse_ratio": math.sqrt(model_sse / null_sse),
                    "model_better_than_exact_null_point_estimate": True,
                    "model_vs_empirical_vector_corr": 0.9,
                    "model_vs_empirical_speed_corr": 0.8,
                    "model_vs_empirical_weighted_local_cosine": 0.85,
                    "model_vs_empirical_local_cosine_occupancy_coverage": 0.95,
                    "exact_null_vs_empirical_vector_corr": 0.84,
                    "exact_null_vs_empirical_weighted_local_cosine": 0.75,
                    "exact_null_vs_empirical_local_cosine_occupancy_coverage": 0.95,
                    "null_referenced_correction_vs_empirical_excess_vector_corr": 0.72,
                    "null_referenced_correction_vs_empirical_excess_speed_corr": 0.68,
                    "null_referenced_correction_vs_empirical_excess_weighted_local_cosine": 0.70,
                    "excess_local_cosine_occupancy_coverage": 0.90,
                    "null_referenced_excess_amplitude_slope": 0.88,
                    "M_null_relative_field_skill": 0.60,
                    "Psi_null_relative_field_skill": 0.50,
                    "M_model_minus_null_sse": -0.004,
                    "Psi_model_minus_null_sse": -0.003,
                    "scaffold_rescaling_source_split": "A_val",
                    "frozen_scalar_scaffold_alpha": 1.1,
                    "frozen_diagonal_scaffold_alpha_M": 1.2,
                    "frozen_diagonal_scaffold_alpha_Psi": 0.9,
                    "scalar_rescaled_scaffold_sse_to_empirical": 0.008,
                    "diagonal_rescaled_scaffold_sse_to_empirical": 0.007,
                    "model_minus_scalar_rescaled_scaffold_sse": model_sse - 0.008,
                    "model_minus_diagonal_rescaled_scaffold_sse": model_sse - 0.007,
                    "model_better_than_frozen_diagonal_scaffold_point_estimate": model_sse < 0.007,
                    "inference_note": "synthetic",
                }
            )
    point = pd.DataFrame(rows)
    point_path = write_table(point, table_root / "null_referenced_recovery_metrics_all_splits")

    bootstrap_rows = []
    for model in (MECHANISM_LABEL, primary):
        bootstrap_rows.append(
            {
                "model": model,
                "bootstrap_replicates": 1000,
                "descriptive_bootstrap_tail_fraction_delta_ge_zero": 0.0,
                "primary_delta_sse_model_minus_exact_null_mean": -0.006,
                "primary_delta_sse_model_minus_exact_null_2p5": -0.008,
                "primary_delta_sse_model_minus_exact_null_50": -0.006,
                "primary_delta_sse_model_minus_exact_null_97p5": -0.003,
                "null_relative_field_skill_mean": 0.5,
                "null_relative_field_skill_2p5": 0.3,
                "null_relative_field_skill_50": 0.5,
                "null_relative_field_skill_97p5": 0.7,
                "null_normalized_rmse_ratio_mean": 0.7,
                "null_normalized_rmse_ratio_2p5": 0.5,
                "null_normalized_rmse_ratio_50": 0.7,
                "null_normalized_rmse_ratio_97p5": 0.85,
                "model_minus_diagonal_rescaled_scaffold_sse_mean": -0.003,
                "model_minus_diagonal_rescaled_scaffold_sse_2p5": -0.005,
                "model_minus_diagonal_rescaled_scaffold_sse_50": -0.003,
                "model_minus_diagonal_rescaled_scaffold_sse_97p5": -0.001,
                "two_model_simultaneous_97p5_upper_bounds_below_zero": True,
                "bootstrap_contract": "synthetic",
            }
        )
    write_table(
        pd.DataFrame(bootstrap_rows),
        table_root / "B_confirm_user_multiplier_bootstrap_summary",
    )
    # Synthetic replicate table is present only for source-audit coverage.
    replicate_rows = [
        {
            "replicate": index,
            "model": model,
            "primary_delta_sse_model_minus_exact_null": -0.006,
        }
        for index in range(5)
        for model in (MECHANISM_LABEL, primary)
    ]
    write_table(
        pd.DataFrame(replicate_rows),
        table_root / "B_confirm_user_multiplier_bootstrap_replicates",
    )

    for split in EXPECTED_SPLITS:
        geometry_rows = []
        fields = [
            "empirical",
            "exact_construction_null",
            "empirical_excess",
            f"{MECHANISM_LABEL}_raw",
            f"{MECHANISM_LABEL}_null_referenced_correction",
            f"{primary}_raw",
            f"{primary}_null_referenced_correction",
        ]
        for index, field in enumerate(fields):
            geometry_rows.append(
                {
                    "split": split,
                    "field": field,
                    "role": "synthetic",
                    "negative_divergence_occupancy_fraction": 0.7 - 0.02 * index,
                    "weighted_mean_divergence": -0.2 + 0.01 * index,
                    "flow_weighted_shell_fraction_inward": 0.6 - 0.01 * index,
                    "flow_weighted_shell_inward_cosine": 0.4 - 0.01 * index,
                    "flow_core_to_shell_speed_ratio": 0.6 + 0.01 * index,
                    "occupancy_core_to_shell_speed_ratio": 0.65 + 0.01 * index,
                }
            )
        write_table(
            pd.DataFrame(geometry_rows),
            table_root / f"{split}_null_referenced_geometry_metrics",
        )
        sensitivity_rows = []
        for model in (MECHANISM_LABEL, primary):
            for baseline, shift in (
                ("exact_cyclic_shift_expectation", 0.0),
                ("existing_finite_replicate_mean", 1e-5),
            ):
                sensitivity_rows.append(
                    {
                        "split": split,
                        "model": model,
                        "null_baseline": baseline,
                        "primary_delta_sse_model_minus_null": -0.006 + shift,
                        "null_relative_field_skill": 0.5 + shift,
                        "excess_vector_corr": 0.72 + shift,
                    }
                )
        write_table(
            pd.DataFrame(sensitivity_rows),
            table_root / f"{split}_exact_vs_finite_null_mean_sensitivity",
        )
        write_table(
            pd.DataFrame(
                [
                    {
                        "level": "within_user_fine",
                        "rows_assigned": 100,
                        "fraction_of_randomizable_rows": 1.0,
                    }
                ]
            ),
            table_root / f"{split}_matching_fallback_coverage",
        )
        write_table(
            pd.DataFrame([{"analysis_rows": 100, "randomizable_rows": 100}]),
            table_root / f"{split}_opportunity_composition_audit",
        )

        audit = {
            "split": split,
            "existing_construction_null_protocol_audit": {"available": True},
            "existing_finite_null_audit": {"available": True},
            "sparse_bootstrap_reproduction_audit": {
                "skipped": False,
                "max_abs_occupancy_difference": 0.0,
                "max_abs_empirical_M_difference": 0.0,
            },
            "mechanism_audit": {
                "parameter_hash_before": "a",
                "parameter_hash_after": "a",
                "calibration_hash_before": "b",
                "calibration_hash_after": "b",
            },
            "event_ssl_audits": {
                primary: {
                    "manifest": {"model_kind": "predictive_state"},
                    "state_target_alignment": {
                        "max_abs_current_M_difference": 0.0,
                        "max_abs_current_Psi_difference": 0.0,
                        "max_abs_target_M_difference": 0.0,
                        "max_abs_target_Psi_difference": 0.0,
                    },
                }
            },
            "guardrails": {
                "coordinates_refit": False,
                "grid_or_support_refit": False,
                "construction_null_matching_cutpoints_refit_outside_A_train": False,
                "construction_null_groups_redefined_from_model_outputs": False,
                "mechanism_family_reselected": False,
                "mechanism_parameters_refit": False,
                "mechanism_calibration_reestimated": False,
                "event_ssl_retrained": False,
                "learned_plane_subtracted_from_empirical_null": False,
                "model_rows_intersected_silently": False,
                "B_confirm_used_for_model_update": False,
            },
        }
        save_json(audit, metadata_root / f"{split}_null_referenced_recovery_audit.json")

    cross = pd.DataFrame(
        [
            {
                "split": split,
                "first_model": MECHANISM_LABEL,
                "second_model": primary,
                "raw_field_vector_corr": 0.8,
                "shared_null_subtracted_correction_vector_corr_descriptive": 0.6,
            }
            for split in EXPECTED_SPLITS
        ]
    )
    write_table(cross, table_root / "cross_model_null_referenced_metrics_all_splits_descriptive")

    manifest = {
        "script": "evaluate_null_referenced_downstream_recovery.py",
        "analysis_name": EXPECTED_ANALYSIS_NAME,
        "splits": list(EXPECTED_SPLITS),
        "primary_event_ssl_label": primary,
        "point_metrics_table": str(point_path.resolve()),
        "guardrails": {
            "model_training": False,
            "model_selection": False,
            "mechanism_refit": False,
            "coordinate_refit": False,
            "grid_or_mask_refit": False,
            "construction_null_protocol_changed": False,
            "learned_plane_used_for_null_subtraction": False,
            "cross_model_correction_correlation_used_as_primary_test": False,
            "B_confirm_used_for_update": False,
        },
        "inference_boundary": "synthetic",
    }
    save_json(manifest, metadata_root / "null_referenced_recovery_manifest.json")
    return root


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="null_referenced_report_selftest_") as temporary:
        root = make_synthetic_analysis_root(Path(temporary) / "analysis")
        output = Path(temporary) / "report"
        run_extraction(
            analysis_root=root,
            output_root=output,
            primary_event_ssl_label=None,
            min_bootstrap_replicates=1000,
            require_bootstrap=True,
            strict=True,
        )
        required = [
            output / "report" / "null_referenced_downstream_recovery_supplementary_report.md",
            output / "report" / "null_referenced_downstream_recovery_tables.tex",
            output / "metadata" / "supplementary_extraction_manifest.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise AssertionError(f"Self-test outputs are missing: {missing}")
    print("[self-test] supplementary extractor passed", flush=True)


# -----------------------------------------------------------------------------
# Main extraction
# -----------------------------------------------------------------------------
def run_extraction(
    analysis_root: Path,
    output_root: Path,
    primary_event_ssl_label: Optional[str],
    min_bootstrap_replicates: int,
    require_bootstrap: bool,
    strict: bool,
) -> None:
    output = output_root.resolve()
    table_root = output / "tables"
    report_root = output / "report"
    metadata_root = output / "metadata"
    for directory in (output, table_root, report_root, metadata_root):
        directory.mkdir(parents=True, exist_ok=True)

    sources = SourceRegistry()
    data = load_inputs(analysis_root, primary_event_ssl_label, sources)
    gates = GateCollector(strict=strict)
    validate_inputs(data, gates, min_bootstrap_replicates, require_bootstrap)

    table1_full, table1_display = build_table1(data)
    table2a_full, table2a_display = build_table2_specificity(data)
    table2b_full, table2b_display = build_table2_geometry(data)

    primary_model_set = {MECHANISM_LABEL, data.primary_event_label}
    if strict:
        if set(table1_full["model"].astype(str)) != primary_model_set:
            raise RuntimeError("Table 1 does not contain exactly the two primary frozen models.")
        if set(table2a_full["model"].astype(str)) != primary_model_set:
            raise RuntimeError("Table 2a does not contain exactly the two primary frozen models.")
        expected_geometry = {
            "empirical_excess",
            f"{MECHANISM_LABEL}_null_referenced_correction",
            f"{data.primary_event_label}_null_referenced_correction",
        }
        if set(table2b_full["field"].astype(str)) != expected_geometry:
            raise RuntimeError(
                "Table 2b does not contain the empirical excess and both model corrections."
            )

    seed_level = event_seed_level_table(data)
    seed_summary = event_seed_summary(seed_level)
    sensitivity_summary = null_sensitivity_summary(data)
    cross_model = data.cross_model.copy()
    matching_coverage = combine_optional_tables(data.matching_coverage)
    opportunity_composition = combine_optional_tables(data.opportunity_composition)
    bootstrap_full = combine_optional_tables(data.bootstrap_replicates)
    bootstrap_summary_full = combine_optional_tables(data.bootstrap_summary)
    geometry_full = combine_optional_tables(data.geometry)

    quality = gates.frame()
    sources_frame = sources.frame()

    output_paths: Dict[str, str] = {}
    for name, frame in (
        ("supplementary_table_1_null_relative_recovery_full", table1_full),
        ("supplementary_table_1_null_relative_recovery_display", table1_display),
        ("supplementary_table_2a_specificity_full", table2a_full),
        ("supplementary_table_2a_specificity_display", table2a_display),
        ("supplementary_table_2b_geometry_full", table2b_full),
        ("supplementary_table_2b_geometry_display", table2b_display),
        ("supplementary_data_all_point_metrics", data.point_metrics),
        ("supplementary_data_event_ssl_seed_level_null_referenced", seed_level),
        ("supplementary_data_event_ssl_seed_summary_null_referenced", seed_summary),
        ("supplementary_data_exact_vs_finite_null_sensitivity", sensitivity_summary),
        ("supplementary_data_exact_vs_finite_null_sensitivity_raw", combine_optional_tables(data.null_sensitivity)),
        ("supplementary_data_cross_model_descriptive", cross_model),
        ("supplementary_data_all_geometry_metrics", geometry_full),
        ("supplementary_data_bootstrap_summary", bootstrap_summary_full),
        ("supplementary_data_bootstrap_replicates", bootstrap_full),
        ("supplementary_data_matching_coverage", matching_coverage),
        ("supplementary_data_opportunity_composition", opportunity_composition),
        ("scientific_quality_gates", quality),
        ("source_audit", sources_frame),
    ):
        if frame.empty and name.startswith("supplementary_data_"):
            continue
        path = write_table(frame, table_root / name)
        output_paths[name] = str(path.resolve())

    latex = build_latex_tables(table1_display, table2a_display, table2b_display)
    latex_path = report_root / "null_referenced_downstream_recovery_tables.tex"
    latex_path.write_text(latex, encoding="utf-8")
    output_paths["latex_tables"] = str(latex_path.resolve())

    report = build_markdown_report(
        table1_display=table1_display,
        table2a_display=table2a_display,
        table2b_display=table2b_display,
        table1_full=table1_full,
        table2a_full=table2a_full,
        table2b_full=table2b_full,
        seed_level=seed_level,
        seed_summary=seed_summary,
        sensitivity_summary=sensitivity_summary,
        quality_gates=quality,
        source_audit=sources_frame,
    )
    report_path = report_root / "null_referenced_downstream_recovery_supplementary_report.md"
    report_path.write_text(report, encoding="utf-8")
    output_paths["markdown_report"] = str(report_path.resolve())

    gates.enforce()

    manifest = {
        "script": Path(__file__).name,
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "analysis_root": str(data.analysis_root),
        "analysis_manifest": str(data.manifest_path),
        "analysis_manifest_sha256": file_sha256(data.manifest_path),
        "primary_event_ssl_label": data.primary_event_label,
        "typeset_table_count": 2,
        "typeset_tables": {
            "table_1": "primary exact-null-referenced field recovery across validation and output-only confirmation",
            "table_2": "confirmation specificity and frozen-support/core-shell geometry, presented as panels a and b",
        },
        "anti_duplication_policy": {
            "excluded_from_display_tables": [
                "raw model-versus-empirical drift correlations already reported in the main text",
                "next-state occupancy and transition metrics already reported in the main figures",
                "original construction-null Monte Carlo tests and matching fractions already reported in Supplementary Note 1",
                "raw cross-model field agreement already reported in the main cross-model section",
                "complete Event-SSL random-seed structural metrics already reported in the existing seed table",
            ],
            "repository_only": [
                "additional Event-SSL seed values for the new null-referenced metrics",
                "cross-model shared-null-subtracted descriptive metrics",
                "exact-versus-finite null-mean sensitivity",
                "bootstrap replicates",
                "complete geometry fields and audit tables",
            ],
        },
        "quality_gate_summary": {
            "total": int(len(quality)),
            "passed": int(quality["passed"].sum()) if not quality.empty else 0,
            "critical_failed": int((~quality["passed"] & quality["critical"]).sum()) if not quality.empty else 0,
            "strict": bool(strict),
        },
        "outputs": output_paths,
        "source_records": [dataclasses.asdict(record) for record in sources.records],
    }
    manifest_path = metadata_root / "supplementary_extraction_manifest.json"
    save_json(manifest, manifest_path)
    print(f"[supplementary extractor] completed: {output}", flush=True)
    print(f"[supplementary extractor] report: {report_path}", flush=True)
    print(f"[supplementary extractor] LaTeX: {latex_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract exactly two compact Supplementary Information tables from the frozen "
            "construction-null-referenced downstream-recovery analysis."
        )
    )
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: <analysis-root>/supplementary_report",
    )
    parser.add_argument(
        "--primary-event-ssl-label",
        type=str,
        default=None,
        help="Optional override; otherwise read from the analysis manifest.",
    )
    parser.add_argument("--min-bootstrap-replicates", type=int, default=1000)
    parser.add_argument(
        "--allow-missing-bootstrap",
        action="store_true",
        help="Permit report generation without B_confirm multiplier-bootstrap intervals.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write diagnostic outputs even when a critical quality gate fails.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.min_bootstrap_replicates < 1:
        raise ValueError("--min-bootstrap-replicates must be positive.")
    analysis_root = args.analysis_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (analysis_root / "supplementary_report").resolve()
    )
    run_extraction(
        analysis_root=analysis_root,
        output_root=output_root,
        primary_event_ssl_label=args.primary_event_ssl_label,
        min_bootstrap_replicates=int(args.min_bootstrap_replicates),
        require_bootstrap=not bool(args.allow_missing_bootstrap),
        strict=not bool(args.allow_incomplete),
    )


if __name__ == "__main__":
    main()
