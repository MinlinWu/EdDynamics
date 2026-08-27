#!/usr/bin/env python3
from __future__ import annotations

"""Extract and audit a manuscript-ready numerical report for the
construction-matched null experiment.

The extractor is output-only. It does not read the learner panel, rebuild the
macrostates, redefine the frozen convergence region, rerun permutations, refit
any model, or create new inferential tests. It verifies the numerical contract
of ``run_construction_matched_null.py`` and organizes its frozen outputs for the
Additional Information section.

A_val is the formal manuscript split. B_confirm can be added only as a separate
output-only replication; p-values are never pooled across splits.
"""

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
RECONSTRUCTION_ATOL = 2e-6
RECONSTRUCTION_RTOL = 2e-6
ARCHIVED_FIELD_ATOL = 1e-10
PRIMARY_SPLIT = "A_val"
CONFIRMATION_SPLIT = "B_confirm"
EXPECTED_PRIMARY_COORDINATES = ["M", "Psi"]
EXPECTED_PRIMARY_METRICS: Tuple[Tuple[str, str], ...] = (
    ("negative_divergence_occupancy_fraction", "greater"),
    ("flow_weighted_shell_fraction_inward", "greater"),
    ("flow_core_to_shell_speed_ratio", "less"),
)
FULL_FIELD_METRIC = "occupancy_weighted_full_field_distance_from_null_mean"
GEOMETRY_METRICS: Tuple[str, ...] = (
    "negative_divergence_occupancy_fraction",
    "weighted_mean_divergence",
    "flow_weighted_shell_fraction_inward",
    "flow_weighted_shell_inward_cosine",
    "flow_core_to_shell_speed_ratio",
    "occupancy_core_to_shell_speed_ratio",
)
FORMAL_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "metric",
    "direction_supporting_excess_structure",
    "observed",
    "pure_ratio_contraction",
    "null_mean",
    "null_sd",
    "null_2p5",
    "null_50",
    "null_97p5",
    "monte_carlo_p",
    "BH_q_across_three_basin_metrics",
    "excess_field_value_descriptive",
)
ARRAY_KEYS: Tuple[str, ...] = (
    "observed_u",
    "observed_v",
    "null_u",
    "null_v",
    "null_mean_u",
    "null_mean_v",
    "null_sd_u",
    "null_sd_v",
    "excess_u",
    "excess_v",
    "pure_ratio_u",
    "pure_ratio_v",
    "drift_mask",
    "state_mask",
    "core_mask",
    "xcenters",
    "ycenters",
    "t_null",
)
METRIC_LABELS: Dict[str, str] = {
    FULL_FIELD_METRIC: "Occupancy-weighted full-field distance from the matched-null mean",
    "negative_divergence_occupancy_fraction": "Interior occupancy fraction with negative local divergence",
    "weighted_mean_divergence": "Interior occupancy-weighted mean local divergence",
    "flow_weighted_shell_fraction_inward": "Flow-weighted inward fraction in the frozen shell",
    "flow_weighted_shell_inward_cosine": "Flow-weighted shell inward cosine",
    "flow_core_to_shell_speed_ratio": "Flow-weighted core-to-shell drift-speed ratio",
    "occupancy_core_to_shell_speed_ratio": "Occupancy-weighted core-to-shell drift-speed ratio",
}
METRIC_ESTIMATORS: Dict[str, str] = {
    FULL_FIELD_METRIC: (
        "occupancy-weighted root mean squared vector distance between the observed field "
        "and the mean of the matched permutation fields"
    ),
    "negative_divergence_occupancy_fraction": (
        "user-balanced occupancy fraction over complete supported five-cell stencils with negative divergence"
    ),
    "weighted_mean_divergence": (
        "user-balanced occupancy-weighted mean divergence over complete supported five-cell stencils"
    ),
    "flow_weighted_shell_fraction_inward": (
        "flow-magnitude-weighted fraction of frozen-shell cells directed toward the A_train core"
    ),
    "flow_weighted_shell_inward_cosine": (
        "flow-magnitude-weighted radial inward cosine in the frozen shell"
    ),
    "flow_core_to_shell_speed_ratio": (
        "flow-weighted mean drift speed in the frozen core divided by that in its frozen shell"
    ),
    "occupancy_core_to_shell_speed_ratio": (
        "occupancy-weighted mean drift speed in the frozen core divided by that in its frozen shell"
    ),
}


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
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
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return payload


def save_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(obj), handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


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
    raise FileNotFoundError(f"Could not find table for {base}.[parquet|csv.gz|csv]")


def read_table(base: Path) -> Tuple[pd.DataFrame, Path]:
    path = find_table(base)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, low_memory=False)
    return frame, path


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing required columns: {missing}")


def numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise KeyError(column)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def bool_array(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    text = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    converted = text.map(mapping)
    if converted.isna().any():
        bad = sorted(text[converted.isna()].unique().tolist())[:5]
        raise RuntimeError(f"Could not parse boolean values: {bad}")
    return converted.to_numpy(dtype=bool)


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        output = float(value)
    except Exception:
        return float(default)
    return output if np.isfinite(output) else float(default)


def values_close(first: Any, second: Any, atol: float = 1e-10, rtol: float = 1e-10) -> bool:
    a = finite_float(first)
    b = finite_float(second)
    if np.isnan(a) and np.isnan(b):
        return True
    if not np.isfinite(a) or not np.isfinite(b):
        return False
    return bool(np.isclose(a, b, atol=atol, rtol=rtol))


def max_abs_difference(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        return float("nan")
    return float(np.max(np.abs(a[valid] - b[valid])))


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=float).ravel()
    b = np.asarray(second, dtype=float).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(valid)) < 3:
        return float("nan")
    aa = a[valid] - float(np.mean(a[valid]))
    bb = b[valid] - float(np.mean(b[valid]))
    denominator = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denominator) if denominator > EPS else float("nan")


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    if not np.any(valid):
        return float("nan")
    x = x[valid]
    w = w[valid]
    if float(np.sum(w)) <= EPS:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    w = w[order]
    cumulative = np.cumsum(w) / float(np.sum(w))
    index = int(np.searchsorted(cumulative, float(probability), side="left"))
    return float(x[min(index, len(x) - 1)])


def weighted_field_distance(
    first_u: np.ndarray,
    first_v: np.ndarray,
    second_u: np.ndarray,
    second_v: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> float:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(first_u)
        & np.isfinite(first_v)
        & np.isfinite(second_u)
        & np.isfinite(second_v)
        & np.isfinite(weight)
        & (np.asarray(weight, dtype=float) >= 0)
    )
    if not np.any(valid):
        return float("nan")
    w = np.asarray(weight, dtype=float)[valid]
    w = w / max(float(np.sum(w)), EPS)
    squared = (
        (np.asarray(first_u, dtype=float)[valid] - np.asarray(second_u, dtype=float)[valid]) ** 2
        + (np.asarray(first_v, dtype=float)[valid] - np.asarray(second_v, dtype=float)[valid]) ** 2
    )
    return float(np.sqrt(np.sum(w * squared)))


def monte_carlo_p(observed: float, null_values: np.ndarray, direction: str) -> float:
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or not np.isfinite(observed):
        return float("nan")
    if direction == "greater":
        extreme = int(np.sum(values >= observed))
    elif direction == "less":
        extreme = int(np.sum(values <= observed))
    else:
        raise ValueError(direction)
    return float((1 + extreme) / (values.size + 1))


def bh_qvalues(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(p))
    if finite_indices.size == 0:
        return q
    order = finite_indices[np.argsort(p[finite_indices])]
    adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0.0, 1.0)
    return q


def format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(number):
        return "NA"
    absolute = abs(number)
    if absolute == 0:
        return "0"
    if number.is_integer() and absolute < 1e15:
        return f"{int(number):,}"
    if absolute >= 1e6 or absolute < 1e-4:
        return f"{number:.4e}"
    if absolute >= 1000:
        return f"{number:,.3f}"
    if absolute >= 10:
        return f"{number:.4f}"
    return f"{number:.6f}"


def markdown_escape(value: Any) -> str:
    text = format_value(value)
    return text.replace("|", "\\|").replace("\n", " ")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    header = "| " + " | ".join(markdown_escape(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(markdown_escape(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def append_section(lines: List[str], heading: str, body: str) -> None:
    lines.extend(["", heading, "", body.rstrip(), ""])


# -----------------------------------------------------------------------------
# Input bundle
# -----------------------------------------------------------------------------
@dataclass
class SplitBundle:
    split: str
    root: Path
    manifest: Dict[str, Any]
    reconstruction_audit: Dict[str, Any]
    permutation_audit: Dict[str, Any]
    cutpoint_metadata: Dict[str, Any]
    summary: pd.DataFrame
    replicates: pd.DataFrame
    coverage: pd.DataFrame
    composition: pd.DataFrame
    observed_ratio_excess: pd.DataFrame
    field: pd.DataFrame
    arrays: Dict[str, np.ndarray]
    input_audit: pd.DataFrame


def table_metadata_row(name: str, path: Path, frame: pd.DataFrame) -> Dict[str, Any]:
    return {
        "name": name,
        "status": "ok",
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
    }


def json_metadata_row(name: str, path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "status": "ok",
        "rows": np.nan,
        "columns": int(len(payload)),
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
    }


def load_split_bundle(root: Path, split: str) -> SplitBundle:
    root = root.resolve()
    table_root = root / "tables"
    metadata_root = root / "metadata"
    array_root = root / "arrays"
    if not root.is_dir():
        raise FileNotFoundError(f"Construction-null output root not found: {root}")

    table_specs = {
        "construction_null_summary": table_root / f"{split}_construction_null_summary",
        "construction_null_replicate_metrics": table_root / f"{split}_construction_null_replicate_metrics",
        "matching_fallback_coverage": table_root / f"{split}_matching_fallback_coverage",
        "opportunity_composition_audit": table_root / f"{split}_opportunity_composition_audit",
        "observed_ratio_excess_metrics": table_root / f"{split}_observed_ratio_excess_metrics",
        "construction_null_field_comparison": table_root / f"{split}_construction_null_field_comparison",
    }
    loaded_tables: Dict[str, pd.DataFrame] = {}
    input_rows: List[Dict[str, Any]] = []
    for name, base in table_specs.items():
        frame, path = read_table(base)
        loaded_tables[name] = frame
        input_rows.append(table_metadata_row(name, path, frame))

    json_specs = {
        "construction_null_manifest": metadata_root / f"{split}_construction_null_manifest.json",
        "reconstruction_and_field_audit": metadata_root / f"{split}_reconstruction_and_field_audit.json",
        "first_permutation_audit": metadata_root / f"{split}_first_permutation_audit.json",
        "matching_cutpoints_A_train": metadata_root / "matching_cutpoints_A_train.json",
    }
    loaded_json: Dict[str, Dict[str, Any]] = {}
    for name, path in json_specs.items():
        payload = load_json(path)
        loaded_json[name] = payload
        input_rows.append(json_metadata_row(name, path, payload))

    array_path = array_root / f"{split}_construction_null_fields.npz"
    if not array_path.exists():
        raise FileNotFoundError(array_path)
    with np.load(array_path, allow_pickle=False) as archive:
        missing = sorted(set(ARRAY_KEYS).difference(archive.files))
        if missing:
            raise RuntimeError(f"Construction-null array archive is missing keys: {missing}")
        arrays = {key: np.asarray(archive[key]) for key in ARRAY_KEYS}
        array_shape_summary = {key: list(arrays[key].shape) for key in ARRAY_KEYS}
    input_rows.append(
        {
            "name": "construction_null_fields",
            "status": "ok",
            "rows": int(arrays["null_u"].shape[0]),
            "columns": int(len(arrays)),
            "path": str(array_path.resolve()),
            "sha256": file_sha256(array_path),
            "shape_summary": json.dumps(array_shape_summary, sort_keys=True),
        }
    )

    return SplitBundle(
        split=split,
        root=root,
        manifest=loaded_json["construction_null_manifest"],
        reconstruction_audit=loaded_json["reconstruction_and_field_audit"],
        permutation_audit=loaded_json["first_permutation_audit"],
        cutpoint_metadata=loaded_json["matching_cutpoints_A_train"],
        summary=loaded_tables["construction_null_summary"],
        replicates=loaded_tables["construction_null_replicate_metrics"],
        coverage=loaded_tables["matching_fallback_coverage"],
        composition=loaded_tables["opportunity_composition_audit"],
        observed_ratio_excess=loaded_tables["observed_ratio_excess_metrics"],
        field=loaded_tables["construction_null_field_comparison"],
        arrays=arrays,
        input_audit=pd.DataFrame(input_rows),
    )


# -----------------------------------------------------------------------------
# Scientific-contract audits
# -----------------------------------------------------------------------------
class GateLedger:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.failures: List[str] = []

    def add(self, scope: str, gate: str, passed: bool, detail: Any, required: bool = True) -> None:
        passed_bool = bool(passed)
        self.rows.append(
            {
                "scope": scope,
                "gate": gate,
                "passed": passed_bool,
                "required": bool(required),
                "detail": detail,
            }
        )
        if required and not passed_bool:
            self.failures.append(f"{scope}: {gate} ({detail})")

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def raise_if_failed(self) -> None:
        if self.failures:
            raise RuntimeError("Scientific report gates failed:\n- " + "\n- ".join(self.failures))


def verify_external_hash(path_value: Any, expected_sha: Any) -> Tuple[bool, str]:
    if not path_value or not expected_sha:
        return False, "path or expected SHA-256 missing"
    path = Path(str(path_value))
    if not path.exists():
        return False, f"source path unavailable in this runtime: {path}"
    actual = file_sha256(path)
    return actual == str(expected_sha), f"expected={expected_sha}; actual={actual}"


def sorted_field_table(field: pd.DataFrame) -> pd.DataFrame:
    require_columns(field, ["x_bin", "y_bin"], "field comparison table")
    return field.sort_values(["x_bin", "y_bin"], kind="mergesort").reset_index(drop=True)


def reshape_field_column(field: pd.DataFrame, column: str, shape: Tuple[int, int]) -> np.ndarray:
    require_columns(field, [column], "field comparison table")
    values = pd.to_numeric(field[column], errors="coerce").to_numpy(dtype=float)
    if values.size != int(np.prod(shape)):
        raise RuntimeError(
            f"Field column {column} has {values.size} values; expected {int(np.prod(shape))}."
        )
    return values.reshape(shape)


def field_pair_diagnostics(
    label: str,
    first_u: np.ndarray,
    first_v: np.ndarray,
    second_u: np.ndarray,
    second_v: np.ndarray,
    mask: np.ndarray,
    occupancy: np.ndarray,
) -> Dict[str, Any]:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(first_u)
        & np.isfinite(first_v)
        & np.isfinite(second_u)
        & np.isfinite(second_v)
        & np.isfinite(occupancy)
        & (np.asarray(occupancy, dtype=float) >= 0)
    )
    if not np.any(valid):
        return {"comparison": label, "common_supported_cells": 0}
    weight = np.asarray(occupancy, dtype=float)[valid]
    occupancy_mass = float(np.sum(weight))
    weight = weight / max(occupancy_mass, EPS)
    first_vector = np.column_stack([np.asarray(first_u)[valid], np.asarray(first_v)[valid]])
    second_vector = np.column_stack([np.asarray(second_u)[valid], np.asarray(second_v)[valid]])
    first_speed = np.linalg.norm(first_vector, axis=1)
    second_speed = np.linalg.norm(second_vector, axis=1)
    denominator = first_speed * second_speed
    local_cosine = np.full(len(first_speed), np.nan, dtype=float)
    nonzero = denominator > EPS
    local_cosine[nonzero] = np.sum(first_vector[nonzero] * second_vector[nonzero], axis=1) / denominator[nonzero]
    cosine_valid = np.isfinite(local_cosine)
    weighted_cosine = (
        float(np.sum(weight[cosine_valid] * local_cosine[cosine_valid]) / max(float(np.sum(weight[cosine_valid])), EPS))
        if np.any(cosine_valid)
        else float("nan")
    )
    difference = first_vector - second_vector
    return {
        "comparison": label,
        "common_supported_cells": int(np.sum(valid)),
        "occupancy_mass_on_common_support": occupancy_mass,
        "field_vector_correlation": pearson(first_vector.ravel(), second_vector.ravel()),
        "field_speed_correlation": pearson(first_speed, second_speed),
        "occupancy_weighted_local_cosine": weighted_cosine,
        "occupancy_weighted_rms_vector_distance": float(
            np.sqrt(np.sum(weight * np.sum(difference * difference, axis=1)))
        ),
        "occupancy_weighted_mean_speed_first": float(np.sum(weight * first_speed)),
        "occupancy_weighted_mean_speed_second": float(np.sum(weight * second_speed)),
        "occupancy_weighted_fraction_positive_local_cosine": (
            float(np.sum(weight[cosine_valid] * (local_cosine[cosine_valid] > 0)) / max(float(np.sum(weight[cosine_valid])), EPS))
            if np.any(cosine_valid)
            else float("nan")
        ),
    }


def audit_split_bundle(
    bundle: SplitBundle,
    gates: GateLedger,
    minimum_replicates: int,
    maximum_weak_fallback_fraction: float,
    allow_smoke_test: bool,
    alpha: float,
) -> Dict[str, pd.DataFrame | Dict[str, Any]]:
    split = bundle.split
    scope = split
    manifest = bundle.manifest
    summary = bundle.summary.copy()
    replicates = bundle.replicates.copy()
    coverage = bundle.coverage.copy()
    composition = bundle.composition.copy()
    observed_ratio = bundle.observed_ratio_excess.copy()
    field = sorted_field_table(bundle.field.copy())
    arrays = bundle.arrays

    gates.add(scope, "manifest split matches requested split", manifest.get("analysis_split") == split, manifest.get("analysis_split"))
    gates.add(
        scope,
        "primary macrostate is exactly M and Psi",
        manifest.get("primary_coordinates") == EXPECTED_PRIMARY_COORDINATES,
        manifest.get("primary_coordinates"),
    )
    gates.add(
        scope,
        "A_val is the formal primary manuscript split",
        manifest.get("primary_manuscript_split") == PRIMARY_SPLIT,
        manifest.get("primary_manuscript_split"),
    )
    if split == CONFIRMATION_SPLIT:
        gates.add(
            scope,
            "confirmation run is explicitly output-only",
            manifest.get("confirmation_output_only") is True,
            manifest.get("confirmation_output_only"),
        )
    else:
        gates.add(
            scope,
            "validation run is not labelled confirmation output-only",
            manifest.get("confirmation_output_only") is False,
            manifest.get("confirmation_output_only"),
        )

    smoke_users = int(manifest.get("smoke_test_max_users", 0) or 0)
    gates.add(
        scope,
        "publication report uses the full split",
        allow_smoke_test or smoke_users == 0,
        f"smoke_test_max_users={smoke_users}",
    )
    declared_replicates = int(manifest.get("replicates", -1))
    gates.add(scope, "replicate count is declared", declared_replicates > 0, declared_replicates)
    gates.add(
        scope,
        f"replicate count is at least {minimum_replicates}",
        allow_smoke_test or declared_replicates >= int(minimum_replicates),
        declared_replicates,
    )

    null_definition = dict(manifest.get("null_definition", {}))
    preserved = list(null_definition.get("preserved", []))
    required_preserved_tokens = (
        "observed current M/Psi anchors",
        "response and exposure denominator increments",
        "user-balanced row weights",
        "current-state occupancy and grid support",
        "A_train-defined convergence core and shell radius",
        "joint marginal distribution of normalized response/exposure innovations",
    )
    gates.add(
        scope,
        "null preserves all declared accounting and frozen-field objects",
        all(token in preserved for token in required_preserved_tokens),
        "; ".join(preserved),
    )
    gates.add(
        scope,
        "joint signed-innovation pair is the randomized object",
        "joint normalized signed-innovation pair" in str(null_definition.get("randomized", "")),
        null_definition.get("randomized"),
    )
    gates.add(
        scope,
        "no object is refitted by the null",
        list(null_definition.get("refitted_objects", [])) == [],
        null_definition.get("refitted_objects"),
    )
    gates.add(
        scope,
        "null does not use a mesostate or trained model",
        str(null_definition.get("mesostate_or_model_use", "")).lower() == "none",
        null_definition.get("mesostate_or_model_use"),
    )

    quality = dict(manifest.get("quality_gates", {}))
    for key in (
        "same_row_phase_reconstruction",
        "next_state_coordinate_reconstruction",
        "next_mass_decay_audit",
        "formal_field_estimator_reproduced",
        "joint_innovation_pairs_permuted_together",
        "global_innovation_marginals_preserved",
        "frozen_A_train_core_reused",
    ):
        gates.add(scope, key.replace("_", " "), quality.get(key) is True, quality.get(key))
    gates.add(
        scope,
        "coordinate or region was not refitted",
        quality.get("coordinate_or_region_refit") is False,
        quality.get("coordinate_or_region_refit"),
    )
    gates.add(
        scope,
        "B_confirm was not used for definition or selection",
        quality.get("B_confirm_used_for_definition_or_selection") is False,
        quality.get("B_confirm_used_for_definition_or_selection"),
    )

    # Source-hash checks are required when the source path is available. They are
    # recorded but not fatal when a report is generated on another machine.
    stage1_ok, stage1_detail = verify_external_hash(
        manifest.get("formal_stage1_script"), manifest.get("formal_stage1_script_sha256")
    )
    stage1_path_available = bool(manifest.get("formal_stage1_script")) and Path(str(manifest.get("formal_stage1_script"))).exists()
    gates.add(scope, "formal Stage-1 script hash matches", stage1_ok, stage1_detail, required=stage1_path_available)
    for label, path_key, sha_key in (
        ("frozen A_train core hash matches", "frozen_core_path", "frozen_core_sha256"),
        ("frozen convergence-threshold hash matches", "frozen_thresholds_path", "frozen_thresholds_sha256"),
    ):
        ok, detail = verify_external_hash(manifest.get(path_key), manifest.get(sha_key))
        available = bool(manifest.get(path_key)) and Path(str(manifest.get(path_key))).exists()
        gates.add(scope, label, ok, detail, required=available)

    require_columns(summary, FORMAL_SUMMARY_COLUMNS, "construction-null summary")
    require_columns(replicates, ["replicate", "seed", *GEOMETRY_METRICS], "replicate metrics")
    require_columns(
        coverage,
        ["level", "matching_keys", "rows_assigned", "fraction_of_randomizable_rows", "rows_remaining_after_level"],
        "matching coverage",
    )
    require_columns(
        composition,
        [
            "analysis_rows",
            "randomizable_rows",
            "zero_innovation_rows",
            "response_increment_present_fraction",
            "exposure_increment_present_fraction",
            "support_present_fraction",
            "idle_present_fraction",
            "mean_response_active_mass",
            "mean_support_active_mass",
            "mean_idle_mass",
        ],
        "opportunity composition audit",
    )
    require_columns(observed_ratio, ["field", *GEOMETRY_METRICS], "observed/ratio/excess metrics")
    require_columns(
        field,
        [
            "x_bin",
            "y_bin",
            "M_center",
            "Psi_center",
            "occupancy_probability",
            "drift_M",
            "drift_Psi",
            "drift_supported",
            "null_mean_drift_M",
            "null_mean_drift_Psi",
            "null_sd_drift_M",
            "null_sd_drift_Psi",
            "excess_drift_M",
            "excess_drift_Psi",
            "pure_ratio_drift_M",
            "pure_ratio_drift_Psi",
        ],
        "field comparison",
    )

    expected_metric_rows = {FULL_FIELD_METRIC, *(metric for metric, _ in EXPECTED_PRIMARY_METRICS)}
    actual_metric_rows = set(summary["metric"].astype(str))
    gates.add(
        scope,
        "summary contains exactly the prespecified full-field and three basin tests",
        actual_metric_rows == expected_metric_rows and len(summary) == len(expected_metric_rows),
        sorted(actual_metric_rows),
    )
    summary = summary.set_index(summary["metric"].astype(str), drop=False)
    for metric, direction in EXPECTED_PRIMARY_METRICS:
        actual_direction = str(summary.loc[metric, "direction_supporting_excess_structure"])
        gates.add(
            scope,
            f"{metric} uses the prespecified one-sided direction",
            actual_direction == direction,
            actual_direction,
        )
    gates.add(
        scope,
        "full-field test uses the prespecified greater direction",
        str(summary.loc[FULL_FIELD_METRIC, "direction_supporting_excess_structure"]) == "greater",
        summary.loc[FULL_FIELD_METRIC, "direction_supporting_excess_structure"],
    )

    replicate_ids = pd.to_numeric(replicates["replicate"], errors="coerce").to_numpy(dtype=float)
    replicate_seeds = pd.to_numeric(replicates["seed"], errors="coerce").to_numpy(dtype=float)
    gates.add(scope, "replicate table row count matches the manifest", len(replicates) == declared_replicates, len(replicates))
    gates.add(
        scope,
        "replicate identifiers are the complete zero-based sequence",
        np.array_equal(replicate_ids, np.arange(len(replicates), dtype=float)),
        f"first={replicate_ids[0] if len(replicate_ids) else 'NA'}; last={replicate_ids[-1] if len(replicate_ids) else 'NA'}",
    )
    gates.add(scope, "replicate seeds are finite and unique", np.isfinite(replicate_seeds).all() and len(np.unique(replicate_seeds)) == len(replicate_seeds), len(np.unique(replicate_seeds)))

    # Array identities and archived table agreement.
    null_u = np.asarray(arrays["null_u"], dtype=float)
    null_v = np.asarray(arrays["null_v"], dtype=float)
    observed_u = np.asarray(arrays["observed_u"], dtype=float)
    observed_v = np.asarray(arrays["observed_v"], dtype=float)
    null_mean_u = np.asarray(arrays["null_mean_u"], dtype=float)
    null_mean_v = np.asarray(arrays["null_mean_v"], dtype=float)
    null_sd_u = np.asarray(arrays["null_sd_u"], dtype=float)
    null_sd_v = np.asarray(arrays["null_sd_v"], dtype=float)
    excess_u = np.asarray(arrays["excess_u"], dtype=float)
    excess_v = np.asarray(arrays["excess_v"], dtype=float)
    ratio_u = np.asarray(arrays["pure_ratio_u"], dtype=float)
    ratio_v = np.asarray(arrays["pure_ratio_v"], dtype=float)
    drift_mask = np.asarray(arrays["drift_mask"], dtype=bool)
    state_mask = np.asarray(arrays["state_mask"], dtype=bool)
    core_mask = np.asarray(arrays["core_mask"], dtype=bool)
    t_null_saved = np.asarray(arrays["t_null"], dtype=float)

    shape = observed_u.shape
    shape_ok = (
        observed_v.shape == shape
        and null_mean_u.shape == shape
        and null_mean_v.shape == shape
        and null_sd_u.shape == shape
        and null_sd_v.shape == shape
        and excess_u.shape == shape
        and excess_v.shape == shape
        and ratio_u.shape == shape
        and ratio_v.shape == shape
        and drift_mask.shape == shape
        and state_mask.shape == shape
        and core_mask.shape == shape
        and null_u.shape == (declared_replicates, *shape)
        and null_v.shape == (declared_replicates, *shape)
        and t_null_saved.shape == (declared_replicates,)
    )
    gates.add(scope, "all field arrays share the declared grid and replicate shape", shape_ok, {key: list(value.shape) for key, value in arrays.items()})

    null_mean_u_recomputed = np.mean(null_u, axis=0)
    null_mean_v_recomputed = np.mean(null_v, axis=0)
    null_sd_u_recomputed = np.std(null_u, axis=0, ddof=1) if declared_replicates > 1 else np.zeros_like(null_mean_u)
    null_sd_v_recomputed = np.std(null_v, axis=0, ddof=1) if declared_replicates > 1 else np.zeros_like(null_mean_v)
    array_identity_differences = {
        "null_mean_M": max_abs_difference(null_mean_u, null_mean_u_recomputed),
        "null_mean_Psi": max_abs_difference(null_mean_v, null_mean_v_recomputed),
        "null_sd_M": max_abs_difference(null_sd_u, null_sd_u_recomputed),
        "null_sd_Psi": max_abs_difference(null_sd_v, null_sd_v_recomputed),
        "excess_M": max_abs_difference(excess_u, observed_u - null_mean_u),
        "excess_Psi": max_abs_difference(excess_v, observed_v - null_mean_v),
    }
    gates.add(
        scope,
        "saved null means, standard deviations and excess fields are exact array identities",
        all(np.isfinite(value) and value <= 1e-12 for value in array_identity_differences.values()),
        array_identity_differences,
    )

    field_shape = (len(np.asarray(arrays["xcenters"])), len(np.asarray(arrays["ycenters"])))
    gates.add(scope, "field table row count matches the grid", len(field) == int(np.prod(field_shape)) and field_shape == shape, f"rows={len(field)}; shape={shape}")
    field_differences = {
        "observed_M": max_abs_difference(reshape_field_column(field, "drift_M", shape), observed_u),
        "observed_Psi": max_abs_difference(reshape_field_column(field, "drift_Psi", shape), observed_v),
        "null_mean_M": max_abs_difference(reshape_field_column(field, "null_mean_drift_M", shape), null_mean_u),
        "null_mean_Psi": max_abs_difference(reshape_field_column(field, "null_mean_drift_Psi", shape), null_mean_v),
        "null_sd_M": max_abs_difference(reshape_field_column(field, "null_sd_drift_M", shape), null_sd_u),
        "null_sd_Psi": max_abs_difference(reshape_field_column(field, "null_sd_drift_Psi", shape), null_sd_v),
        "excess_M": max_abs_difference(reshape_field_column(field, "excess_drift_M", shape), excess_u),
        "excess_Psi": max_abs_difference(reshape_field_column(field, "excess_drift_Psi", shape), excess_v),
        "pure_ratio_M": max_abs_difference(reshape_field_column(field, "pure_ratio_drift_M", shape), ratio_u),
        "pure_ratio_Psi": max_abs_difference(reshape_field_column(field, "pure_ratio_drift_Psi", shape), ratio_v),
    }
    table_drift_mask = bool_array(field["drift_supported"]).reshape(shape)
    table_state_mask = bool_array(field["state_supported"]).reshape(shape)
    drift_mask_exact = np.array_equal(table_drift_mask, drift_mask)
    state_mask_exact = np.array_equal(table_state_mask, state_mask)
    gates.add(
        scope,
        "field comparison table exactly reproduces the saved arrays and support masks",
        (
            drift_mask_exact
            and state_mask_exact
            and all(
                np.isfinite(value) and value <= 1e-12
                for value in field_differences.values()
            )
        ),
        {
            "drift_mask_exact": drift_mask_exact,
            "state_mask_exact": state_mask_exact,
            **field_differences,
        },
    )

    occupancy = reshape_field_column(field, "occupancy_probability", shape)
    gates.add(scope, "user-balanced occupancy is normalized", abs(float(np.nansum(occupancy)) - 1.0) <= 1e-10, float(np.nansum(occupancy)))

    # Stage-1 defines state and drift support independently: state support uses
    # occupancy and user-count thresholds, whereas drift support uses the count
    # of valid one-step transitions. Neither mask is required to contain the
    # other. The necessary consistency condition is that both supports are
    # non-empty and every drift-supported cell has positive state occupancy.
    state_cells = int(np.sum(state_mask))
    drift_cells = int(np.sum(drift_mask))
    common_cells = int(np.sum(state_mask & drift_mask))
    drift_has_positive_occupancy = bool(
        drift_cells > 0
        and np.all(np.isfinite(occupancy[drift_mask]))
        and np.all(occupancy[drift_mask] > 0.0)
    )
    gates.add(
        scope,
        "state and drift supports satisfy their independent Stage-1 contracts",
        state_cells > 0 and drift_has_positive_occupancy,
        (
            f"drift_cells={drift_cells}; state_cells={state_cells}; "
            f"common_cells={common_cells}; "
            f"drift_cells_with_nonpositive_occupancy="
            f"{int(np.sum(drift_mask & (~np.isfinite(occupancy) | (occupancy <= 0.0))))}"
        ),
    )
    gates.add(scope, "frozen core has the field-grid shape and non-zero support", core_mask.shape == shape and int(np.sum(core_mask)) > 0, int(np.sum(core_mask)))

    # Recompute the full-field primary statistic and its leave-one-out null.
    t_observed = weighted_field_distance(observed_u, observed_v, null_mean_u, null_mean_v, occupancy, drift_mask)
    null_sum_u = np.sum(null_u, axis=0)
    null_sum_v = np.sum(null_v, axis=0)
    t_null_recomputed = np.empty(declared_replicates, dtype=float)
    for index in range(declared_replicates):
        if declared_replicates > 1:
            leave_one_out_u = (null_sum_u - null_u[index]) / (declared_replicates - 1)
            leave_one_out_v = (null_sum_v - null_v[index]) / (declared_replicates - 1)
        else:
            leave_one_out_u = null_mean_u
            leave_one_out_v = null_mean_v
        t_null_recomputed[index] = weighted_field_distance(
            null_u[index], null_v[index], leave_one_out_u, leave_one_out_v, occupancy, drift_mask
        )
    gates.add(
        scope,
        "leave-one-out full-field null statistics are reproducible",
        max_abs_difference(t_null_saved, t_null_recomputed) <= 1e-12,
        max_abs_difference(t_null_saved, t_null_recomputed),
    )

    # Recompute all summary statistics and formal p/q values.
    audit_rows: List[Dict[str, Any]] = []
    full_row = summary.loc[FULL_FIELD_METRIC]
    full_p = monte_carlo_p(t_observed, t_null_recomputed, "greater")
    full_expected = {
        "observed": t_observed,
        "null_mean": float(np.mean(t_null_recomputed)),
        "null_sd": float(np.std(t_null_recomputed, ddof=1)) if declared_replicates > 1 else 0.0,
        "null_2p5": float(np.quantile(t_null_recomputed, 0.025)),
        "null_50": float(np.quantile(t_null_recomputed, 0.50)),
        "null_97p5": float(np.quantile(t_null_recomputed, 0.975)),
        "monte_carlo_p": full_p,
        "pure_ratio_contraction": weighted_field_distance(
            ratio_u, ratio_v, null_mean_u, null_mean_v, occupancy, drift_mask
        ),
    }
    full_matches = {key: values_close(full_row[key], value, atol=2e-12, rtol=2e-10) for key, value in full_expected.items()}
    gates.add(scope, "full-field summary and Monte Carlo p-value reproduce from arrays", all(full_matches.values()), full_matches)
    audit_rows.append({"metric": FULL_FIELD_METRIC, **full_expected})

    basin_p_values: List[float] = []
    basin_expected_rows: Dict[str, Dict[str, float]] = {}
    for metric, direction in EXPECTED_PRIMARY_METRICS:
        values = pd.to_numeric(replicates[metric], errors="coerce").to_numpy(dtype=float)
        observed = finite_float(summary.loc[metric, "observed"])
        p_value = monte_carlo_p(observed, values, direction)
        expected = {
            "observed": observed,
            "null_mean": float(np.nanmean(values)),
            "null_sd": float(np.nanstd(values, ddof=1)),
            "null_2p5": float(np.nanquantile(values, 0.025)),
            "null_50": float(np.nanquantile(values, 0.50)),
            "null_97p5": float(np.nanquantile(values, 0.975)),
            "monte_carlo_p": p_value,
        }
        basin_expected_rows[metric] = expected
        basin_p_values.append(p_value)
    q_values = bh_qvalues(basin_p_values)
    basin_matches: Dict[str, Dict[str, bool]] = {}
    for (metric, _), q_value in zip(EXPECTED_PRIMARY_METRICS, q_values):
        row = summary.loc[metric]
        expected = dict(basin_expected_rows[metric])
        expected["BH_q_across_three_basin_metrics"] = float(q_value)
        matches = {key: values_close(row[key], value, atol=2e-12, rtol=2e-10) for key, value in expected.items()}
        basin_matches[metric] = matches
        audit_rows.append({"metric": metric, **expected})
    gates.add(
        scope,
        "three basin summaries, one-sided Monte Carlo p-values and BH q-values reproduce",
        all(all(values.values()) for values in basin_matches.values()),
        basin_matches,
    )

    # Observed / pure-ratio / excess geometry table must agree with the summary.
    observed_ratio = observed_ratio.set_index(observed_ratio["field"].astype(str), drop=False)
    expected_fields = {"observed", "pure_ratio_contraction", "excess_observed_minus_null_mean"}
    gates.add(
        scope,
        "observed, pure-ratio and excess-field metric rows are present",
        set(observed_ratio.index) == expected_fields,
        sorted(set(observed_ratio.index)),
    )
    metric_table_matches: Dict[str, Dict[str, bool]] = {}
    for metric in GEOMETRY_METRICS:
        metric_table_matches[metric] = {
            "observed": values_close(observed_ratio.loc["observed", metric], summary.loc[metric, "observed"])
            if metric in summary.index
            else True,
            "pure_ratio": values_close(observed_ratio.loc["pure_ratio_contraction", metric], summary.loc[metric, "pure_ratio_contraction"])
            if metric in summary.index
            else True,
            "excess": values_close(observed_ratio.loc["excess_observed_minus_null_mean", metric], summary.loc[metric, "excess_field_value_descriptive"])
            if metric in summary.index
            else True,
        }
    gates.add(
        scope,
        "observed/ratio/excess geometry table agrees with the formal summary",
        all(all(values.values()) for values in metric_table_matches.values()),
        metric_table_matches,
    )

    # Matching coverage and composition accounting.
    if len(composition) != 1:
        raise RuntimeError(f"Opportunity composition audit must contain one row; found {len(composition)}.")
    composition_row = composition.iloc[0]
    analysis_rows = int(finite_float(composition_row["analysis_rows"], -1))
    randomizable_rows = int(finite_float(composition_row["randomizable_rows"], -1))
    zero_rows = int(finite_float(composition_row["zero_innovation_rows"], -1))
    coverage["rows_assigned"] = pd.to_numeric(coverage["rows_assigned"], errors="coerce").fillna(0).astype(np.int64)
    coverage_by_level = coverage.groupby(coverage["level"].astype(str), sort=False)["rows_assigned"].sum().to_dict()
    value = lambda name: int(coverage_by_level.get(name, 0))
    within_user_rows = value("within_user_fine") + value("within_user_coarse")
    across_user_rows = value("across_user_fine") + value("across_user_coarse")
    global_opportunity_rows = value("global_opportunity")
    last_resort_rows = value("global_last_resort")
    unmatched_rows = value("unmatched_singleton_self_exempt")
    deterministic_zero_rows = value("deterministic_zero_innovation")
    randomized_rows = within_user_rows + across_user_rows + global_opportunity_rows + last_resort_rows
    weak_fallback_fraction = (last_resort_rows + unmatched_rows) / max(randomizable_rows, 1)
    matching_summary = pd.DataFrame(
        [
            {
                "split": split,
                "analysis_rows": analysis_rows,
                "randomizable_rows_before_singleton_exemption": randomizable_rows,
                "randomized_rows": randomized_rows,
                "zero_innovation_rows": zero_rows,
                "within_user_matched_rows": within_user_rows,
                "across_user_matched_rows": across_user_rows,
                "global_opportunity_rows": global_opportunity_rows,
                "global_last_resort_rows": last_resort_rows,
                "unmatched_singleton_self_exempt_rows": unmatched_rows,
                "within_user_fraction_of_randomizable": within_user_rows / max(randomizable_rows, 1),
                "across_user_fraction_of_randomizable": across_user_rows / max(randomizable_rows, 1),
                "global_opportunity_fraction_of_randomizable": global_opportunity_rows / max(randomizable_rows, 1),
                "weak_fallback_fraction_of_randomizable": weak_fallback_fraction,
                "randomized_fraction_of_analysis_rows": randomized_rows / max(analysis_rows, 1),
                "zero_innovation_fraction_of_analysis_rows": zero_rows / max(analysis_rows, 1),
            }
        ]
    )
    gates.add(scope, "opportunity composition rows sum to the analysis rows", randomizable_rows + zero_rows == analysis_rows, f"{randomizable_rows}+{zero_rows} vs {analysis_rows}")
    gates.add(scope, "coverage levels account for every non-zero innovation row", randomized_rows + unmatched_rows == randomizable_rows, f"randomized={randomized_rows}; unmatched={unmatched_rows}; declared={randomizable_rows}")
    gates.add(scope, "coverage zero-innovation rows match the opportunity audit", deterministic_zero_rows == zero_rows, f"coverage={deterministic_zero_rows}; audit={zero_rows}")
    gates.add(
        scope,
        "weakly matched fallback share is within the reporting threshold",
        weak_fallback_fraction <= float(maximum_weak_fallback_fraction) + 1e-15,
        f"{weak_fallback_fraction:.6%} <= {float(maximum_weak_fallback_fraction):.6%}",
    )

    first_permutation = bundle.permutation_audit
    gates.add(scope, "first permutation randomizes exactly the assigned rows", int(first_permutation.get("randomized_rows", -1)) == randomized_rows, first_permutation.get("randomized_rows"))
    gates.add(scope, "first permutation has no fixed points among randomized rows", int(first_permutation.get("fixed_points_among_randomized_rows", -1)) == 0, first_permutation.get("fixed_points_among_randomized_rows"))
    gates.add(scope, "joint M/Psi innovations are moved together", first_permutation.get("joint_pair_moved_together") is True, first_permutation.get("joint_pair_moved_together"))
    gates.add(scope, "donor mapping is bijective within disjoint groups", first_permutation.get("overall_mapping_bijective_by_disjoint_group_permutations") is True, first_permutation.get("overall_mapping_bijective_by_disjoint_group_permutations"))
    permutation_differences = {
        "mean_Z_M": abs(finite_float(first_permutation.get("mean_Z_M_before")) - finite_float(first_permutation.get("mean_Z_M_after"))),
        "mean_Z_Psi": abs(finite_float(first_permutation.get("mean_Z_Psi_before")) - finite_float(first_permutation.get("mean_Z_Psi_after"))),
        "mean_Z_product": abs(finite_float(first_permutation.get("mean_product_Z_before")) - finite_float(first_permutation.get("mean_product_Z_after"))),
    }
    gates.add(scope, "joint innovation marginal moments are preserved to numerical precision", all(np.isfinite(value) and value <= 1e-12 for value in permutation_differences.values()), permutation_differences)

    # Reconstruction and archived Stage-1 field audit.
    reconstruction = bundle.reconstruction_audit
    reconstruction_keys = (
        "max_abs_next_M_reconstruction_error",
        "max_abs_next_Psi_reconstruction_error",
        "max_abs_delta_M_reconstruction_error",
        "max_abs_delta_Psi_reconstruction_error",
        "max_response_Z_bound_excess_before_clipping",
        "max_exposure_Z_bound_excess_before_clipping",
        "max_abs_next_response_mass_decay_error",
        "max_rel_next_response_mass_decay_error",
        "max_abs_next_exposure_denominator_decay_error",
        "max_rel_next_exposure_denominator_decay_error",
        "max_abs_next_exposure_numerator_decay_error",
        "max_rel_next_exposure_numerator_decay_error",
    )
    gates.add(
        scope,
        "all required reconstruction diagnostics are finite",
        all(np.isfinite(finite_float(reconstruction.get(key))) for key in reconstruction_keys),
        {key: reconstruction.get(key) for key in reconstruction_keys},
    )
    coordinate_reconstruction_values = {
        key: finite_float(reconstruction.get(key))
        for key in (
            "max_abs_next_M_reconstruction_error",
            "max_abs_next_Psi_reconstruction_error",
            "max_abs_delta_M_reconstruction_error",
            "max_abs_delta_Psi_reconstruction_error",
        )
    }
    gates.add(
        scope,
        "same-row coordinate reconstruction errors are within the formal tolerance",
        all(
            np.isfinite(value) and value <= RECONSTRUCTION_ATOL
            for value in coordinate_reconstruction_values.values()
        ),
        {
            "tolerance": RECONSTRUCTION_ATOL,
            **coordinate_reconstruction_values,
        },
    )
    innovation_bound_values = {
        key: finite_float(reconstruction.get(key))
        for key in (
            "max_response_Z_bound_excess_before_clipping",
            "max_exposure_Z_bound_excess_before_clipping",
        )
    }
    gates.add(
        scope,
        "normalized-innovation bound excess is within the formal tolerance",
        all(
            np.isfinite(value) and value <= RECONSTRUCTION_ATOL
            for value in innovation_bound_values.values()
        ),
        {
            "tolerance": RECONSTRUCTION_ATOL,
            **innovation_bound_values,
        },
    )
    mass_decay_pairs = {
        "response_evidence_mass": (
            finite_float(reconstruction.get("max_abs_next_response_mass_decay_error")),
            finite_float(reconstruction.get("max_rel_next_response_mass_decay_error")),
        ),
        "exposure_denominator": (
            finite_float(reconstruction.get("max_abs_next_exposure_denominator_decay_error")),
            finite_float(reconstruction.get("max_rel_next_exposure_denominator_decay_error")),
        ),
        "exposure_numerator": (
            finite_float(reconstruction.get("max_abs_next_exposure_numerator_decay_error")),
            finite_float(reconstruction.get("max_rel_next_exposure_numerator_decay_error")),
        ),
    }
    mass_decay_pass = {
        name: (
            np.isfinite(abs_error)
            and np.isfinite(rel_error)
            and (abs_error <= RECONSTRUCTION_ATOL or rel_error <= RECONSTRUCTION_RTOL)
        )
        for name, (abs_error, rel_error) in mass_decay_pairs.items()
    }
    gates.add(
        scope,
        "post-state masses decay to the archived next pre-state under the formal paired absolute/relative gate",
        all(mass_decay_pass.values()),
        {
            "absolute_tolerance": RECONSTRUCTION_ATOL,
            "relative_tolerance": RECONSTRUCTION_RTOL,
            "pairs": {
                name: {"max_abs": values[0], "max_rel": values[1], "passed": mass_decay_pass[name]}
                for name, values in mass_decay_pairs.items()
            },
        },
    )
    gates.add(scope, "formal optimized field estimator is reproduced", reconstruction.get("formal_field_estimator_reproduced_to_1e-12") is True, reconstruction.get("formal_field_estimator_reproduced_to_1e-12"))
    archived = dict(reconstruction.get("archived_field_audit", {}))
    if allow_smoke_test and archived.get("skipped") is True:
        gates.add(scope, "archived Stage-1 publication field equality", True, archived, required=False)
    else:
        gates.add(scope, "archived Stage-1 publication field equality was not skipped", archived.get("skipped") is False, archived.get("skipped"))
        gates.add(scope, "archived Stage-1 drift-support mask is exact", archived.get("drift_mask_exact_match") is True, archived.get("drift_mask_exact_match"))
        archive_differences = {
            "drift_M": finite_float(archived.get("max_abs_drift_M_difference")),
            "drift_Psi": finite_float(archived.get("max_abs_drift_Psi_difference")),
            "occupancy": finite_float(archived.get("max_abs_occupancy_probability_difference")),
        }
        gates.add(scope, f"archived Stage-1 field values match within {ARCHIVED_FIELD_ATOL:.0e}", all(np.isfinite(value) and value <= ARCHIVED_FIELD_ATOL for value in archive_differences.values()), archive_differences)

    # Formal test table with effect-size descriptors. These do not create new tests.
    formal_rows: List[Dict[str, Any]] = []
    for metric in [FULL_FIELD_METRIC, *(name for name, _ in EXPECTED_PRIMARY_METRICS)]:
        row = summary.loc[metric]
        direction = str(row["direction_supporting_excess_structure"])
        observed = finite_float(row["observed"])
        null_mean = finite_float(row["null_mean"])
        null_sd = finite_float(row["null_sd"])
        if direction == "greater":
            supportive_difference = observed - null_mean
        elif direction == "less":
            supportive_difference = null_mean - observed
        else:
            supportive_difference = float("nan")
        standardized = supportive_difference / null_sd if np.isfinite(null_sd) and null_sd > EPS else float("nan")
        if metric == FULL_FIELD_METRIC:
            null_values = t_null_recomputed
        else:
            null_values = pd.to_numeric(replicates[metric], errors="coerce").to_numpy(dtype=float)
        if direction == "greater":
            supportive_percentile = float(np.mean(null_values < observed))
        else:
            supportive_percentile = float(np.mean(null_values > observed))
        p_value = finite_float(row["monte_carlo_p"])
        q_value = finite_float(row["BH_q_across_three_basin_metrics"])
        formal_rows.append(
            {
                "split": split,
                "metric": metric,
                "metric_label": METRIC_LABELS.get(metric, metric),
                "direction_supporting_excess_structure": direction,
                "observed": observed,
                "pure_ratio_contraction": finite_float(row["pure_ratio_contraction"]),
                "null_mean": null_mean,
                "null_sd": null_sd,
                "null_2p5": finite_float(row["null_2p5"]),
                "null_50": finite_float(row["null_50"]),
                "null_97p5": finite_float(row["null_97p5"]),
                "observed_minus_null_mean": observed - null_mean,
                "supportive_direction_difference": supportive_difference,
                "descriptive_standardized_separation": standardized,
                "supportive_tail_percentile": supportive_percentile,
                "monte_carlo_p": p_value,
                "BH_q_across_three_basin_metrics": q_value,
                "formal_alpha": float(alpha),
                "formal_test_passed": bool(p_value < alpha) if metric == FULL_FIELD_METRIC else bool(q_value < alpha),
                "excess_field_value_descriptive": finite_float(row["excess_field_value_descriptive"]),
                "minimum_attainable_monte_carlo_p": 1.0 / (declared_replicates + 1),
            }
        )
    formal_tests = pd.DataFrame(formal_rows)

    # Replicate-distribution table for every saved field-geometry diagnostic.
    distribution_rows: List[Dict[str, Any]] = []
    for metric in GEOMETRY_METRICS:
        values = pd.to_numeric(replicates[metric], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        distribution_rows.append(
            {
                "split": split,
                "metric": metric,
                "metric_label": METRIC_LABELS.get(metric, metric),
                "finite_replicates": int(len(values)),
                "minimum": float(np.min(values)),
                "q2p5": float(np.quantile(values, 0.025)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.quantile(values, 0.50)),
                "q75": float(np.quantile(values, 0.75)),
                "q97p5": float(np.quantile(values, 0.975)),
                "maximum": float(np.max(values)),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    t_values = t_null_recomputed[np.isfinite(t_null_recomputed)]
    distribution_rows.append(
        {
            "split": split,
            "metric": FULL_FIELD_METRIC,
            "metric_label": METRIC_LABELS[FULL_FIELD_METRIC],
            "finite_replicates": int(len(t_values)),
            "minimum": float(np.min(t_values)),
            "q2p5": float(np.quantile(t_values, 0.025)),
            "q25": float(np.quantile(t_values, 0.25)),
            "median": float(np.quantile(t_values, 0.50)),
            "q75": float(np.quantile(t_values, 0.75)),
            "q97p5": float(np.quantile(t_values, 0.975)),
            "maximum": float(np.max(t_values)),
            "mean": float(np.mean(t_values)),
            "sd": float(np.std(t_values, ddof=1)) if len(t_values) > 1 else 0.0,
        }
    )
    null_distributions = pd.DataFrame(distribution_rows)

    # Field-pair and cellwise excess diagnostics are descriptive only.
    field_pairs = pd.DataFrame(
        [
            field_pair_diagnostics(
                "observed_vs_matched_null_mean",
                observed_u,
                observed_v,
                null_mean_u,
                null_mean_v,
                drift_mask,
                occupancy,
            ),
            field_pair_diagnostics(
                "observed_vs_pure_ratio_contraction",
                observed_u,
                observed_v,
                ratio_u,
                ratio_v,
                drift_mask,
                occupancy,
            ),
            field_pair_diagnostics(
                "matched_null_mean_vs_pure_ratio_contraction",
                null_mean_u,
                null_mean_v,
                ratio_u,
                ratio_v,
                drift_mask,
                occupancy,
            ),
        ]
    )
    field_pairs.insert(0, "split", split)

    supported = drift_mask & np.isfinite(occupancy)
    supported_weight = occupancy[supported]
    supported_weight_normalized = supported_weight / max(float(np.sum(supported_weight)), EPS)
    observed_speed = np.sqrt(observed_u * observed_u + observed_v * observed_v)
    null_speed = np.sqrt(null_mean_u * null_mean_u + null_mean_v * null_mean_v)
    ratio_speed = np.sqrt(ratio_u * ratio_u + ratio_v * ratio_v)
    excess_speed = np.sqrt(excess_u * excess_u + excess_v * excess_v)
    null_sd_magnitude = np.sqrt(null_sd_u * null_sd_u + null_sd_v * null_sd_v)
    valid_null_sd = supported & np.isfinite(null_sd_magnitude) & (null_sd_magnitude > EPS)
    excess_to_sd = np.divide(
        excess_speed,
        null_sd_magnitude,
        out=np.full_like(excess_speed, np.nan, dtype=float),
        where=valid_null_sd,
    )
    null_radius = np.sqrt((null_u - null_mean_u[None, :, :]) ** 2 + (null_v - null_mean_v[None, :, :]) ** 2)
    cellwise_null_radius_q97p5 = np.quantile(null_radius, 0.975, axis=0)
    observed_above_cellwise_q97p5 = supported & (excess_speed > cellwise_null_radius_q97p5)
    observed_vector = np.stack([observed_u, observed_v], axis=-1)
    excess_vector = np.stack([excess_u, excess_v], axis=-1)
    observed_excess_dot = np.sum(observed_vector * excess_vector, axis=-1)
    core_supported = supported & core_mask
    outside_core_supported = supported & ~core_mask

    def weighted_mean_map(values: np.ndarray, mask: np.ndarray) -> float:
        valid = mask & np.isfinite(values) & np.isfinite(occupancy) & (occupancy >= 0)
        if not np.any(valid):
            return float("nan")
        weights = occupancy[valid]
        return float(np.sum(weights * values[valid]) / max(float(np.sum(weights)), EPS))

    def occupancy_fraction(mask: np.ndarray, base_mask: Optional[np.ndarray] = None) -> float:
        base = supported if base_mask is None else (supported & np.asarray(base_mask, dtype=bool))
        valid = base & np.asarray(mask, dtype=bool)
        denominator = float(np.sum(occupancy[base]))
        return float(np.sum(occupancy[valid]) / max(denominator, EPS))

    cellwise_excess = pd.DataFrame(
        [
            {
                "split": split,
                "grid_shape": f"{shape[0]}x{shape[1]}",
                "grid_cells": int(np.prod(shape)),
                "state_supported_cells": int(np.sum(state_mask)),
                "drift_supported_cells": int(np.sum(drift_mask)),
                "frozen_core_cells": int(np.sum(core_mask)),
                "supported_core_cells": int(np.sum(core_supported)),
                "supported_occupancy_mass": float(np.sum(occupancy[supported])),
                "core_occupancy_fraction_within_supported_field": occupancy_fraction(core_mask),
                "occupancy_weighted_observed_speed": weighted_mean_map(observed_speed, supported),
                "occupancy_weighted_matched_null_mean_speed": weighted_mean_map(null_speed, supported),
                "occupancy_weighted_pure_ratio_speed": weighted_mean_map(ratio_speed, supported),
                "occupancy_weighted_excess_speed": weighted_mean_map(excess_speed, supported),
                "occupancy_weighted_null_vector_sd_magnitude": weighted_mean_map(null_sd_magnitude, supported),
                "occupancy_fraction_with_nonzero_null_sd": occupancy_fraction(valid_null_sd),
                "occupancy_weighted_median_excess_to_null_sd": weighted_quantile(
                    excess_to_sd[valid_null_sd], occupancy[valid_null_sd], 0.50
                ),
                "occupancy_fraction_excess_to_null_sd_gt_1_within_nonzero_sd": occupancy_fraction(
                    excess_to_sd > 1.0, valid_null_sd
                ),
                "occupancy_fraction_excess_to_null_sd_gt_2_within_nonzero_sd": occupancy_fraction(
                    excess_to_sd > 2.0, valid_null_sd
                ),
                "occupancy_fraction_observed_distance_above_cellwise_null_q97p5_descriptive": occupancy_fraction(
                    observed_above_cellwise_q97p5
                ),
                "occupancy_fraction_excess_aligned_with_observed": occupancy_fraction(observed_excess_dot > 0),
                "occupancy_weighted_excess_speed_in_core": weighted_mean_map(excess_speed, core_supported),
                "occupancy_weighted_excess_speed_outside_core": weighted_mean_map(excess_speed, outside_core_supported),
                "cellwise_inference_boundary": (
                    "cellwise null-radius exceedance is descriptive and carries no multiplicity-adjusted cellwise claim"
                ),
            }
        ]
    )

    # Matching cutpoint and audit summary.
    cutpoint_payload = bundle.cutpoint_metadata
    cutpoint_audit = dict(cutpoint_payload.get("audit", {}))
    cutpoints = dict(cutpoint_payload.get("cutpoints", {}))
    cutpoint_rows: List[Dict[str, Any]] = []
    for name in ("log_a_m", "log_a_psi", "support_share", "idle_share", "sequence_length"):
        value = cutpoints.get(name, [])
        cutpoint_rows.append(
            {
                "split": split,
                "matching_variable": name,
                "cutpoints": ";".join(format_value(item) for item in (value if isinstance(value, list) else [])),
                "fit_split": cutpoints.get("fit_split", cutpoint_audit.get("fit_split")),
                "fit_rows_sampled": cutpoints.get("fit_rows_sampled", cutpoint_audit.get("priority_sample_rows")),
                "fit_users": cutpoints.get("fit_users", cutpoint_audit.get("users_counted")),
                "rows_scanned": cutpoint_audit.get("rows_scanned"),
                "eligible_transition_rows": cutpoint_audit.get("eligible_transition_rows"),
                "sample_policy": cutpoint_audit.get("sample_policy"),
            }
        )
    cutpoint_table = pd.DataFrame(cutpoint_rows)
    gates.add(scope, "matching cutpoints were fitted on A_train", str(cutpoint_audit.get("fit_split")) == "A_train", cutpoint_audit.get("fit_split"))
    gates.add(scope, "A_train matching sample contains at least 10,000 rows", allow_smoke_test or int(cutpoint_audit.get("priority_sample_rows", 0)) >= 10000, cutpoint_audit.get("priority_sample_rows"))

    # Formal significance is a result, not a data-quality gate. Record it in a
    # separate claim-support table without failing report generation.
    claim_rows: List[Dict[str, Any]] = []
    for _, row in formal_tests.iterrows():
        metric = str(row["metric"])
        claim_rows.append(
            {
                "split": split,
                "claim_component": metric,
                "formal_role": "primary full-field test" if metric == FULL_FIELD_METRIC else "prespecified basin metric",
                "direction": row["direction_supporting_excess_structure"],
                "formal_value": row["monte_carlo_p"] if metric == FULL_FIELD_METRIC else row["BH_q_across_three_basin_metrics"],
                "criterion": f"p < {alpha}" if metric == FULL_FIELD_METRIC else f"BH q < {alpha}",
                "criterion_met": bool(row["formal_test_passed"]),
                "interpretation": (
                    "tests whether the observed field departs from state-independent joint signed innovations under matched accounting; "
                    "it does not prove uniqueness of the coordinates or a causal learner intervention"
                ),
            }
        )
    claim_support = pd.DataFrame(claim_rows)

    # Analysis contract table.
    contract = pd.DataFrame(
        [
            {
                "split": split,
                "analysis_role": "formal primary test" if split == PRIMARY_SPLIT else "frozen output-only replication",
                "rows_in_analysis_panel": int(manifest.get("rows_in_analysis_panel", -1)),
                "users_in_analysis_panel": int(manifest.get("users_in_analysis_panel", -1)),
                "valid_drift_rows": int(manifest.get("valid_drift_rows", -1)),
                "replicates": declared_replicates,
                "base_seed": int(manifest.get("base_seed", -1)),
                "minimum_monte_carlo_p": 1.0 / (declared_replicates + 1),
                "primary_coordinates": ";".join(manifest.get("primary_coordinates", [])),
                "shell_radius": finite_float(manifest.get("shell_radius")),
                "frozen_core_path": manifest.get("frozen_core_path"),
                "frozen_core_sha256": manifest.get("frozen_core_sha256"),
                "frozen_thresholds_path": manifest.get("frozen_thresholds_path"),
                "frozen_thresholds_sha256": manifest.get("frozen_thresholds_sha256"),
                "formal_stage1_script": manifest.get("formal_stage1_script"),
                "formal_stage1_script_sha256": manifest.get("formal_stage1_script_sha256"),
                "runtime_seconds": finite_float(manifest.get("runtime_seconds")),
            }
        ]
    )
    gates.add(scope, "manifest analysis rows match opportunity composition", int(manifest.get("valid_drift_rows", -1)) == analysis_rows, f"manifest={manifest.get('valid_drift_rows')}; composition={analysis_rows}")
    gates.add(scope, "manifest panel rows are not fewer than valid drift rows", int(manifest.get("rows_in_analysis_panel", -1)) >= analysis_rows, f"panel={manifest.get('rows_in_analysis_panel')}; valid={analysis_rows}")

    # Reconstruction audit flattened for reporting.
    reconstruction_rows: List[Dict[str, Any]] = []
    for key, value in reconstruction.items():
        if key in {"archived_field_audit", "resolved_activity_off_target_aliases"}:
            continue
        reconstruction_rows.append(
            {
                "split": split,
                "audit_group": "reconstruction",
                "metric": key,
                "value": value,
            }
        )
    for key, value in archived.items():
        reconstruction_rows.append(
            {
                "split": split,
                "audit_group": "archived_stage1_field",
                "metric": key,
                "value": value,
            }
        )
    for key, value in first_permutation.items():
        reconstruction_rows.append(
            {
                "split": split,
                "audit_group": "first_permutation",
                "metric": key,
                "value": value,
            }
        )
    numerical_audit = pd.DataFrame(reconstruction_rows)

    return {
        "formal_tests": formal_tests,
        "null_distributions": null_distributions,
        "matching_summary": matching_summary,
        "matching_coverage": coverage.reset_index(drop=True),
        "opportunity_composition": composition.reset_index(drop=True),
        "matching_cutpoints": cutpoint_table,
        "field_pairs": field_pairs,
        "cellwise_excess": cellwise_excess,
        "claim_support": claim_support,
        "analysis_contract": contract,
        "numerical_audit": numerical_audit,
        "summary": summary.reset_index(drop=True),
        "replicates": replicates.reset_index(drop=True),
        "field": field,
        "input_audit": bundle.input_audit,
        "arrays": arrays,
        "manifest": manifest,
    }


# -----------------------------------------------------------------------------
# Cross-split descriptive replication
# -----------------------------------------------------------------------------
def cross_split_report(
    validation: Dict[str, Any],
    confirmation: Dict[str, Any],
    gates: GateLedger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    val_arrays = validation["arrays"]
    con_arrays = confirmation["arrays"]
    val_field = validation["field"]
    con_field = confirmation["field"]
    val_manifest = validation["manifest"]
    con_manifest = confirmation["manifest"]

    same_shape = np.asarray(val_arrays["observed_u"]).shape == np.asarray(con_arrays["observed_u"]).shape
    same_x = np.array_equal(np.asarray(val_arrays["xcenters"]), np.asarray(con_arrays["xcenters"]))
    same_y = np.array_equal(np.asarray(val_arrays["ycenters"]), np.asarray(con_arrays["ycenters"]))
    same_core = np.array_equal(np.asarray(val_arrays["core_mask"], dtype=bool), np.asarray(con_arrays["core_mask"], dtype=bool))
    gates.add("A_val_vs_B_confirm", "grid shape and centers are identical", same_shape and same_x and same_y, {"shape": same_shape, "x": same_x, "y": same_y})
    gates.add("A_val_vs_B_confirm", "frozen A_train core mask is identical", same_core, same_core)
    gates.add(
        "A_val_vs_B_confirm",
        "formal Stage-1 script hash is identical",
        val_manifest.get("formal_stage1_script_sha256") == con_manifest.get("formal_stage1_script_sha256"),
        f"A_val={val_manifest.get('formal_stage1_script_sha256')}; B_confirm={con_manifest.get('formal_stage1_script_sha256')}",
    )
    gates.add(
        "A_val_vs_B_confirm",
        "frozen core and threshold hashes are identical",
        val_manifest.get("frozen_core_sha256") == con_manifest.get("frozen_core_sha256")
        and val_manifest.get("frozen_thresholds_sha256") == con_manifest.get("frozen_thresholds_sha256"),
        {
            "core_A_val": val_manifest.get("frozen_core_sha256"),
            "core_B_confirm": con_manifest.get("frozen_core_sha256"),
            "threshold_A_val": val_manifest.get("frozen_thresholds_sha256"),
            "threshold_B_confirm": con_manifest.get("frozen_thresholds_sha256"),
        },
    )
    gates.add(
        "A_val_vs_B_confirm",
        "shell radius, replicate count and base seed are identical",
        values_close(val_manifest.get("shell_radius"), con_manifest.get("shell_radius"))
        and int(val_manifest.get("replicates", -1)) == int(con_manifest.get("replicates", -2))
        and int(val_manifest.get("base_seed", -1)) == int(con_manifest.get("base_seed", -2)),
        {
            "shell_radius_A_val": val_manifest.get("shell_radius"),
            "shell_radius_B_confirm": con_manifest.get("shell_radius"),
            "replicates_A_val": val_manifest.get("replicates"),
            "replicates_B_confirm": con_manifest.get("replicates"),
            "base_seed_A_val": val_manifest.get("base_seed"),
            "base_seed_B_confirm": con_manifest.get("base_seed"),
        },
    )
    gates.add(
        "A_val_vs_B_confirm",
        "null definition is identical",
        json.dumps(json_safe(val_manifest.get("null_definition")), sort_keys=True)
        == json.dumps(json_safe(con_manifest.get("null_definition")), sort_keys=True),
        {
            "A_val": val_manifest.get("null_definition"),
            "B_confirm": con_manifest.get("null_definition"),
        },
    )
    gates.add(
        "A_val_vs_B_confirm",
        "A_train matching cutpoints and fit audit are identical",
        json.dumps(json_safe(val_manifest.get("matching_cutpoints")), sort_keys=True)
        == json.dumps(json_safe(con_manifest.get("matching_cutpoints")), sort_keys=True)
        and json.dumps(json_safe(val_manifest.get("matching_cutpoint_audit")), sort_keys=True)
        == json.dumps(json_safe(con_manifest.get("matching_cutpoint_audit")), sort_keys=True),
        {
            "matching_cutpoints_A_val": val_manifest.get("matching_cutpoints"),
            "matching_cutpoints_B_confirm": con_manifest.get("matching_cutpoints"),
            "matching_audit_A_val": val_manifest.get("matching_cutpoint_audit"),
            "matching_audit_B_confirm": con_manifest.get("matching_cutpoint_audit"),
        },
    )

    val_occ = reshape_field_column(val_field, "occupancy_probability", np.asarray(val_arrays["observed_u"]).shape)
    con_occ = reshape_field_column(con_field, "occupancy_probability", np.asarray(con_arrays["observed_u"]).shape)
    common_mask = np.asarray(val_arrays["drift_mask"], dtype=bool) & np.asarray(con_arrays["drift_mask"], dtype=bool)
    pooled_weight = 0.5 * (val_occ + con_occ)

    rows: List[Dict[str, Any]] = []
    for label, first_u, first_v, second_u, second_v in (
        (
            "observed_fields",
            val_arrays["observed_u"],
            val_arrays["observed_v"],
            con_arrays["observed_u"],
            con_arrays["observed_v"],
        ),
        (
            "matched_null_mean_fields",
            val_arrays["null_mean_u"],
            val_arrays["null_mean_v"],
            con_arrays["null_mean_u"],
            con_arrays["null_mean_v"],
        ),
        (
            "excess_fields",
            val_arrays["excess_u"],
            val_arrays["excess_v"],
            con_arrays["excess_u"],
            con_arrays["excess_v"],
        ),
        (
            "pure_ratio_fields",
            val_arrays["pure_ratio_u"],
            val_arrays["pure_ratio_v"],
            con_arrays["pure_ratio_u"],
            con_arrays["pure_ratio_v"],
        ),
    ):
        row = field_pair_diagnostics(
            label,
            np.asarray(first_u, dtype=float),
            np.asarray(first_v, dtype=float),
            np.asarray(second_u, dtype=float),
            np.asarray(second_v, dtype=float),
            common_mask,
            pooled_weight,
        )
        row["first_split"] = PRIMARY_SPLIT
        row["second_split"] = CONFIRMATION_SPLIT
        rows.append(row)
    field_replication = pd.DataFrame(rows)

    val_tests = validation["formal_tests"].set_index("metric")
    con_tests = confirmation["formal_tests"].set_index("metric")
    metric_rows: List[Dict[str, Any]] = []
    for metric in [FULL_FIELD_METRIC, *(name for name, _ in EXPECTED_PRIMARY_METRICS)]:
        metric_rows.append(
            {
                "metric": metric,
                "metric_label": METRIC_LABELS.get(metric, metric),
                "A_val_observed": finite_float(val_tests.loc[metric, "observed"]),
                "B_confirm_observed": finite_float(con_tests.loc[metric, "observed"]),
                "absolute_observed_difference": abs(
                    finite_float(val_tests.loc[metric, "observed"])
                    - finite_float(con_tests.loc[metric, "observed"])
                ),
                "A_val_null_mean": finite_float(val_tests.loc[metric, "null_mean"]),
                "B_confirm_null_mean": finite_float(con_tests.loc[metric, "null_mean"]),
                "absolute_null_mean_difference": abs(
                    finite_float(val_tests.loc[metric, "null_mean"])
                    - finite_float(con_tests.loc[metric, "null_mean"])
                ),
                "A_val_supportive_direction_difference": finite_float(
                    val_tests.loc[metric, "supportive_direction_difference"]
                ),
                "B_confirm_supportive_direction_difference": finite_float(
                    con_tests.loc[metric, "supportive_direction_difference"]
                ),
                "A_val_monte_carlo_p": finite_float(val_tests.loc[metric, "monte_carlo_p"]),
                "B_confirm_monte_carlo_p": finite_float(con_tests.loc[metric, "monte_carlo_p"]),
                "A_val_BH_q": finite_float(val_tests.loc[metric, "BH_q_across_three_basin_metrics"]),
                "B_confirm_BH_q": finite_float(con_tests.loc[metric, "BH_q_across_three_basin_metrics"]),
                "inference_policy": "reported separately; no cross-split p-value pooling",
            }
        )
    metric_replication = pd.DataFrame(metric_rows)
    return field_replication, metric_replication


# -----------------------------------------------------------------------------
# Publication ledgers and Markdown report
# -----------------------------------------------------------------------------
def ledger_row(
    priority: str,
    category: str,
    split: str,
    metric: str,
    value: Any,
    source: str,
    estimator: str,
    weighting: str,
    support: str,
    uncertainty_status: str,
    interpretation: str,
    manuscript_use: str,
) -> Dict[str, Any]:
    return {
        "priority": priority,
        "category": category,
        "split": split,
        "metric": metric,
        "value": value,
        "formatted_value": format_value(value),
        "source": source,
        "estimator": estimator,
        "weighting": weighting,
        "support": support,
        "uncertainty_status": uncertainty_status,
        "interpretation": interpretation,
        "manuscript_use": manuscript_use,
    }


def build_publication_ledger(split_results: Mapping[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split, result in split_results.items():
        contract = result["analysis_contract"].iloc[0]
        formal = result["formal_tests"]
        matching = result["matching_summary"].iloc[0]
        cellwise = result["cellwise_excess"].iloc[0]
        composition = result["opportunity_composition"].iloc[0]
        audit = result["numerical_audit"]
        role = "formal A_val inference" if split == PRIMARY_SPLIT else "output-only confirmation replication"
        source_prefix = f"{split} construction-null outputs"

        for metric, value, use in (
            ("analysis panel rows", contract["rows_in_analysis_panel"], "State in Additional Methods or the analysis-contract table."),
            ("analysis panel users", contract["users_in_analysis_panel"], "State in Additional Methods or the analysis-contract table."),
            ("valid drift rows", contract["valid_drift_rows"], "State with the field estimator support contract."),
            ("null replicates", contract["replicates"], "State with Monte Carlo p-value resolution."),
            ("minimum attainable Monte Carlo p", contract["minimum_monte_carlo_p"], "State when reporting exact permutation p-values."),
            ("frozen shell radius", contract["shell_radius"], "Additional Methods; inherited from A_train Stage-1 thresholds."),
        ):
            rows.append(
                ledger_row(
                    "additional_required",
                    "analysis contract",
                    split,
                    metric,
                    value,
                    source_prefix,
                    "frozen output audit",
                    "not applicable",
                    role,
                    "not an inferential estimate",
                    "Defines the analysis contract and prevents post-null refitting.",
                    use,
                )
            )

        for _, row in formal.iterrows():
            metric = str(row["metric"])
            label = str(row["metric_label"])
            uncertainty = (
                f"Monte Carlo permutation distribution with {int(contract['replicates'])} replicates; "
                f"minimum p={format_value(contract['minimum_monte_carlo_p'])}"
            )
            rows.extend(
                [
                    ledger_row(
                        "additional_required",
                        "formal construction-null test",
                        split,
                        f"{label}: observed",
                        row["observed"],
                        source_prefix,
                        METRIC_ESTIMATORS.get(metric, metric),
                        "user-balanced field / prespecified frozen geometry",
                        f"{int(cellwise['drift_supported_cells'])} supported grid cells; {role}",
                        uncertainty,
                        "Observed field statistic under the frozen Stage-1 contract.",
                        "Report with the matched-null interval and exact p or BH q.",
                    ),
                    ledger_row(
                        "additional_required",
                        "formal construction-null test",
                        split,
                        f"{label}: matched-null mean",
                        row["null_mean"],
                        source_prefix,
                        "mean over matched joint signed-innovation permutations",
                        "same anchors, denominator increments and user-balanced weights as observed",
                        f"{int(contract['replicates'])} null replicates",
                        uncertainty,
                        "Expected statistic when joint signed innovations are exchangeable within the matched accounting design.",
                        "Report with the 2.5–97.5% null interval.",
                    ),
                    ledger_row(
                        "additional_required",
                        "formal construction-null test",
                        split,
                        f"{label}: null 2.5%",
                        row["null_2p5"],
                        source_prefix,
                        "empirical 2.5th percentile of the null ensemble",
                        "same matched-null contract",
                        f"{int(contract['replicates'])} null replicates",
                        uncertainty,
                        "Lower endpoint of the descriptive 95% permutation interval.",
                        "Report as the first endpoint of the null interval.",
                    ),
                    ledger_row(
                        "additional_required",
                        "formal construction-null test",
                        split,
                        f"{label}: null 97.5%",
                        row["null_97p5"],
                        source_prefix,
                        "empirical 97.5th percentile of the null ensemble",
                        "same matched-null contract",
                        f"{int(contract['replicates'])} null replicates",
                        uncertainty,
                        "Upper endpoint of the descriptive 95% permutation interval.",
                        "Report as the second endpoint of the null interval.",
                    ),
                    ledger_row(
                        "additional_required",
                        "formal construction-null test",
                        split,
                        f"{label}: Monte Carlo p",
                        row["monte_carlo_p"],
                        source_prefix,
                        "one-sided exact Monte Carlo rank with +1 correction",
                        "same matched-null contract",
                        f"{int(contract['replicates'])} null replicates",
                        uncertainty,
                        "Formal full-field p-value or unadjusted basin-metric p-value in the prespecified direction.",
                        "For basin metrics, accompany this value with the BH-adjusted q-value.",
                    ),
                ]
            )
            if metric != FULL_FIELD_METRIC:
                rows.append(
                    ledger_row(
                        "additional_required",
                        "formal construction-null test",
                        split,
                        f"{label}: BH q across three basin metrics",
                        row["BH_q_across_three_basin_metrics"],
                        source_prefix,
                        "Benjamini–Hochberg adjustment across the three prespecified basin metrics",
                        "three one-sided Monte Carlo tests",
                        role,
                        uncertainty,
                        "Multiplicity-adjusted basin-metric evidence.",
                        "Use this, not the unadjusted p-value alone, for the basin-metric significance statement.",
                    )
                )
            rows.extend(
                [
                    ledger_row(
                        "additional_recommended",
                        "effect description",
                        split,
                        f"{label}: pure ratio-contraction baseline",
                        row["pure_ratio_contraction"],
                        source_prefix,
                        "denominator-growth-only zero-signed-innovation baseline",
                        "same observed anchors and denominator increments",
                        role,
                        "descriptive only; not the formal null",
                        "Shows the relaxation generated by denominator growth without signed innovations.",
                        "Label explicitly as a descriptive accounting baseline.",
                    ),
                    ledger_row(
                        "additional_recommended",
                        "effect description",
                        split,
                        f"{label}: descriptive standardized separation",
                        row["descriptive_standardized_separation"],
                        source_prefix,
                        "supportive-direction observed-minus-null-mean divided by null SD",
                        "matched-null distribution",
                        role,
                        "descriptive; no Gaussian assumption or additional p-value",
                        "Scale-free separation from the null ensemble in the prespecified direction.",
                        "Optional table value; do not substitute it for the Monte Carlo test.",
                    ),
                    ledger_row(
                        "additional_recommended",
                        "effect description",
                        split,
                        f"{label}: excess-field diagnostic",
                        row["excess_field_value_descriptive"],
                        source_prefix,
                        "field geometry recomputed on observed drift minus matched-null mean drift",
                        "user-balanced field with frozen core and support",
                        role,
                        "descriptive; not a second formal test",
                        "Describes the geometry retained after subtracting the matched-null mean field.",
                        "Use to interpret the excess-field panel, not as an independent inferential claim.",
                    ),
                ]
            )

        for metric, value, interpretation, use in (
            (
                "randomizable fraction of analysis rows",
                matching["randomizable_rows_before_singleton_exemption"] / max(matching["analysis_rows"], 1),
                "Fraction of valid field rows carrying at least one non-zero denominator increment.",
                "Additional Methods matching audit.",
            ),
            (
                "within-user matched fraction of randomizable rows",
                matching["within_user_fraction_of_randomizable"],
                "Share assigned before any cross-user fallback.",
                "Report to show that the strongest matching level dominates when it does.",
            ),
            (
                "across-user matched fraction of randomizable rows",
                matching["across_user_fraction_of_randomizable"],
                "Share assigned by matched cross-user opportunity strata.",
                "Additional Methods matching audit.",
            ),
            (
                "weak fallback fraction of randomizable rows",
                matching["weak_fallback_fraction_of_randomizable"],
                "Global-last-resort plus unmatched-singleton share.",
                "Report to demonstrate that weak matching remains below the declared ceiling.",
            ),
            (
                "support-present fraction",
                composition["support_present_fraction"],
                "Opportunity composition of the formal field rows.",
                "Additional table describing the preserved opportunity structure.",
            ),
            (
                "idle-present fraction",
                composition["idle_present_fraction"],
                "Opportunity composition of the formal field rows.",
                "Additional table describing the preserved opportunity structure.",
            ),
        ):
            rows.append(
                ledger_row(
                    "audit_required",
                    "matching and opportunity audit",
                    split,
                    metric,
                    value,
                    source_prefix,
                    "hierarchical disjoint permutation coverage or opportunity composition",
                    "row count or row fraction",
                    role,
                    "descriptive audit",
                    interpretation,
                    use,
                )
            )

        for metric, value, interpretation in (
            (
                "supported field cells",
                cellwise["drift_supported_cells"],
                "Cells entering the formal full-field comparison under the frozen Stage-1 support rule.",
            ),
            (
                "supported occupancy mass",
                cellwise["supported_occupancy_mass"],
                "User-balanced occupancy represented by supported drift cells.",
            ),
            (
                "occupancy-weighted excess speed",
                cellwise["occupancy_weighted_excess_speed"],
                "Mean magnitude of observed-minus-null-mean drift on supported cells.",
            ),
            (
                "occupancy fraction above cellwise null-radius 97.5%",
                cellwise["occupancy_fraction_observed_distance_above_cellwise_null_q97p5_descriptive"],
                "Descriptive spatial concentration of excess; no cellwise multiplicity-adjusted claim.",
            ),
        ):
            rows.append(
                ledger_row(
                    "additional_recommended",
                    "cellwise excess diagnostics",
                    split,
                    metric,
                    value,
                    source_prefix,
                    "supported-cell descriptive field diagnostic",
                    "user-balanced occupancy",
                    role,
                    "descriptive; no cellwise multiple-testing inference",
                    interpretation,
                    "Use in the Additional Figure/Table caption or field-audit table.",
                )
            )

        # Essential numerical audit values.
        audit_lookup = {
            (str(row["audit_group"]), str(row["metric"])): row["value"]
            for _, row in audit.iterrows()
        }
        for key, label in (
            (("reconstruction", "max_abs_next_M_reconstruction_error"), "maximum next-M reconstruction error"),
            (("reconstruction", "max_abs_next_Psi_reconstruction_error"), "maximum next-Psi reconstruction error"),
            (("archived_stage1_field", "max_abs_drift_M_difference"), "maximum archived-field M-drift difference"),
            (("archived_stage1_field", "max_abs_drift_Psi_difference"), "maximum archived-field Psi-drift difference"),
            (("archived_stage1_field", "max_abs_occupancy_probability_difference"), "maximum archived occupancy difference"),
            (("first_permutation", "fixed_points_among_randomized_rows"), "fixed points among randomized rows"),
        ):
            if key not in audit_lookup:
                continue
            rows.append(
                ledger_row(
                    "audit_required",
                    "numerical identity audit",
                    split,
                    label,
                    audit_lookup[key],
                    source_prefix,
                    "direct numerical audit",
                    "not applicable",
                    role,
                    "quality-control value",
                    "Verifies that the null uses the archived Stage-1 field and exact accounting reconstruction.",
                    "Retain in the Additional audit table; no scientific effect interpretation is attached.",
                )
            )

    return pd.DataFrame(rows)


def build_value_map(ledger: pd.DataFrame) -> pd.DataFrame:
    selected = ledger[ledger["priority"].isin(["additional_required", "audit_required", "additional_recommended"])].copy()
    selected = selected.rename(
        columns={
            "metric": "Manuscript quantity",
            "value": "Value",
            "formatted_value": "Formatted value",
            "source": "Source",
            "interpretation": "Interpretation",
            "manuscript_use": "Additional-information use",
        }
    )
    return selected[
        [
            "split",
            "category",
            "Manuscript quantity",
            "Value",
            "Formatted value",
            "Source",
            "Interpretation",
            "Additional-information use",
        ]
    ].reset_index(drop=True)


def build_markdown_report(
    split_results: Mapping[str, Dict[str, Any]],
    quality_gates: pd.DataFrame,
    publication_ledger: pd.DataFrame,
    input_audit: pd.DataFrame,
    cross_field: Optional[pd.DataFrame],
    cross_metrics: Optional[pd.DataFrame],
    generated_at: str,
) -> str:
    lines: List[str] = [
        "# Construction-matched null numerical report",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "## Scope and interpretation boundary",
        "",
        "This report is generated from frozen construction-null outputs. It does not read the learner panel, rebuild the semantic coordinates, redefine the convergence core, rerun permutations, refit a mesostate partition, or retrain either downstream model.",
        "",
        "The formal null preserves current `M/Psi` anchors, response/exposure denominator increments, user-balanced weights, the archived grid and support, and the A_train-defined convergence geometry. It reassigns the joint normalized signed response/exposure innovation pair within disjoint opportunity-matched permutation groups. Rejection therefore identifies state-conditioned signed-innovation structure beyond matched normalized-memory accounting; it does not establish causal intervention effects, uniqueness of the coordinates, or complete removal of every possible measurement artifact.",
        "",
        "`A_val` is the formal manuscript split. Any `B_confirm` section is a frozen output-only replication and is reported separately without pooling p-values. The pure ratio-contraction field and all cellwise excess summaries are descriptive; the inferential claims come only from the prespecified full-field Monte Carlo test and the three BH-adjusted basin metrics.",
    ]

    append_section(lines, "## Input audit", dataframe_to_markdown(input_audit))
    append_section(lines, "## Scientific quality gates", dataframe_to_markdown(quality_gates))

    for split, result in split_results.items():
        role = "formal primary analysis" if split == PRIMARY_SPLIT else "frozen output-only replication"
        append_section(
            lines,
            f"## {split}: analysis contract ({role})",
            dataframe_to_markdown(result["analysis_contract"]),
        )
        formal_display = result["formal_tests"].copy()
        formal_display["formal_test_value"] = np.where(
            formal_display["metric"].eq(FULL_FIELD_METRIC),
            formal_display["monte_carlo_p"],
            formal_display["BH_q_across_three_basin_metrics"],
        )
        formal_display = formal_display[
            [
                "metric_label",
                "direction_supporting_excess_structure",
                "observed",
                "pure_ratio_contraction",
                "null_mean",
                "null_2p5",
                "null_97p5",
                "monte_carlo_p",
                "BH_q_across_three_basin_metrics",
                "descriptive_standardized_separation",
                "formal_test_passed",
            ]
        ]
        append_section(lines, f"## {split}: prespecified formal tests", dataframe_to_markdown(formal_display))
        append_section(lines, f"### {split}: claim-support ledger", dataframe_to_markdown(result["claim_support"]))

        geometry = result["summary"].merge(
            result["formal_tests"][
                [
                    "metric",
                    "observed_minus_null_mean",
                    "supportive_direction_difference",
                    "descriptive_standardized_separation",
                    "supportive_tail_percentile",
                ]
            ],
            on="metric",
            how="left",
        )
        append_section(lines, f"### {split}: observed, matched-null, pure-ratio and excess values", dataframe_to_markdown(geometry))
        append_section(lines, f"### {split}: null-replicate distributions", dataframe_to_markdown(result["null_distributions"]))
        append_section(lines, f"### {split}: matching summary", dataframe_to_markdown(result["matching_summary"]))
        append_section(lines, f"### {split}: hierarchical matching coverage", dataframe_to_markdown(result["matching_coverage"]))
        append_section(lines, f"### {split}: preserved opportunity composition", dataframe_to_markdown(result["opportunity_composition"]))
        append_section(lines, f"### {split}: A_train matching cutpoints", dataframe_to_markdown(result["matching_cutpoints"]))
        append_section(lines, f"### {split}: field-pair diagnostics", dataframe_to_markdown(result["field_pairs"]))
        append_section(lines, f"### {split}: cellwise excess-field diagnostics", dataframe_to_markdown(result["cellwise_excess"]))
        append_section(lines, f"### {split}: reconstruction, archived-field and permutation audit", dataframe_to_markdown(result["numerical_audit"]))

    if cross_field is not None and cross_metrics is not None:
        append_section(
            lines,
            "## Validation–confirmation descriptive replication",
            "The following comparisons use the same frozen grid and A_train core. They quantify replication but do not combine the two permutation distributions or create a new formal test.",
        )
        append_section(lines, "### Cross-split field agreement", dataframe_to_markdown(cross_field))
        append_section(lines, "### Cross-split formal-metric comparison", dataframe_to_markdown(cross_metrics))

    append_section(lines, "## Publication metric ledger", dataframe_to_markdown(publication_ledger))
    append_section(
        lines,
        "## Additional Information writing rules encoded by this report",
        "\n".join(
            [
                "1. Describe the coordinates as prespecified semantic candidate order variables and the null as a test of construction-implied relaxation, not as a test of whether deliberate coarse-graining is permissible.",
                "2. State exactly what was preserved and randomized. Do not shorten the intervention to an unrestricted shuffle.",
                "3. Report the full-field Monte Carlo p-value as the primary test and the three basin metrics with BH-adjusted q-values.",
                "4. Identify the denominator-growth-only pure-ratio field as descriptive; it is not the formal permutation null.",
                "5. Treat observed-minus-null-mean field geometry and cellwise null-radius exceedance as descriptive localization of excess structure, with no cellwise multiplicity-adjusted significance claim.",
                "6. Report fallback coverage, zero-innovation rows, reconstruction errors, archived-field equality and the no-fixed-point joint-permutation audit.",
                "7. If B_confirm is included, label it output-only and report its values separately; do not pool A_val and B_confirm p-values.",
                "8. The strongest supported conclusion is state-conditioned signed-innovation structure beyond matched accounting. The experiment does not prove coordinate uniqueness, autonomous Markov closure, or causal effects on learners.",
            ]
        ),
    )
    return "\n".join(lines).rstrip() + "\n"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and audit the construction-matched-null numerical report."
    )
    parser.add_argument(
        "--validation-root",
        type=Path,
        required=True,
        help="Output root produced by the formal A_val construction-null run.",
    )
    parser.add_argument(
        "--confirmation-root",
        type=Path,
        default=None,
        help="Optional output root produced by the frozen B_confirm output-only run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Report directory. Default: <validation-root>/numeric_report.",
    )
    parser.add_argument("--minimum-replicates", type=int, default=100)
    parser.add_argument("--maximum-weak-fallback-fraction", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--allow-smoke-test",
        action="store_true",
        help="Allow user-subsampled or <minimum-replicates outputs. Never use this flag for publication tables.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.minimum_replicates < 20:
        raise ValueError("--minimum-replicates must be at least 20; use 100 for the publication report.")
    if not (0.0 <= args.maximum_weak_fallback_fraction <= 1.0):
        raise ValueError("--maximum-weak-fallback-fraction must lie in [0,1].")
    if not (0.0 < args.alpha < 1.0):
        raise ValueError("--alpha must lie in (0,1).")

    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else args.validation_root.resolve() / "numeric_report"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    bundles: Dict[str, SplitBundle] = {
        PRIMARY_SPLIT: load_split_bundle(args.validation_root, PRIMARY_SPLIT)
    }
    if args.confirmation_root is not None:
        bundles[CONFIRMATION_SPLIT] = load_split_bundle(args.confirmation_root, CONFIRMATION_SPLIT)

    gates = GateLedger()
    split_results: Dict[str, Dict[str, Any]] = {}
    for split, bundle in bundles.items():
        split_results[split] = audit_split_bundle(
            bundle=bundle,
            gates=gates,
            minimum_replicates=int(args.minimum_replicates),
            maximum_weak_fallback_fraction=float(args.maximum_weak_fallback_fraction),
            allow_smoke_test=bool(args.allow_smoke_test),
            alpha=float(args.alpha),
        )

    cross_field: Optional[pd.DataFrame] = None
    cross_metrics: Optional[pd.DataFrame] = None
    if CONFIRMATION_SPLIT in split_results:
        cross_field, cross_metrics = cross_split_report(
            split_results[PRIMARY_SPLIT], split_results[CONFIRMATION_SPLIT], gates
        )

    quality_gates = gates.frame()
    # Write the gate ledger before raising so a failed report remains diagnosable.
    write_csv(quality_gates, output_root / "construction_matched_null_quality_gates.csv")
    gates.raise_if_failed()

    combined_input_audit = pd.concat(
        [
            result["input_audit"].assign(split=split)
            for split, result in split_results.items()
        ],
        ignore_index=True,
        sort=False,
    )
    combined_formal = pd.concat(
        [result["formal_tests"] for result in split_results.values()],
        ignore_index=True,
        sort=False,
    )
    combined_distributions = pd.concat(
        [result["null_distributions"] for result in split_results.values()],
        ignore_index=True,
        sort=False,
    )
    combined_matching = pd.concat(
        [result["matching_summary"] for result in split_results.values()],
        ignore_index=True,
        sort=False,
    )
    combined_coverage = pd.concat(
        [result["matching_coverage"].assign(split=split) for split, result in split_results.items()],
        ignore_index=True,
        sort=False,
    )
    combined_composition = pd.concat(
        [result["opportunity_composition"].assign(split=split) for split, result in split_results.items()],
        ignore_index=True,
        sort=False,
    )
    combined_cutpoints = pd.concat(
        [result["matching_cutpoints"] for result in split_results.values()],
        ignore_index=True,
        sort=False,
    )
    combined_field_pairs = pd.concat(
        [result["field_pairs"] for result in split_results.values()],
        ignore_index=True,
        sort=False,
    )
    combined_cellwise = pd.concat(
        [result["cellwise_excess"] for result in split_results.values()],
        ignore_index=True,
        sort=False,
    )
    combined_claim_support = pd.concat(
        [result["claim_support"] for result in split_results.values()],
        ignore_index=True,
        sort=False,
    )
    combined_audit = pd.concat(
        [result["numerical_audit"] for result in split_results.values()],
        ignore_index=True,
        sort=False,
    )
    publication_ledger = build_publication_ledger(split_results)
    value_map = build_value_map(publication_ledger)

    table_outputs = {
        "input_audit": write_csv(combined_input_audit, output_root / "construction_matched_null_input_audit.csv"),
        "quality_gates": output_root / "construction_matched_null_quality_gates.csv",
        "formal_tests": write_csv(combined_formal, output_root / "construction_matched_null_formal_tests.csv"),
        "null_distributions": write_csv(combined_distributions, output_root / "construction_matched_null_null_distributions.csv"),
        "matching_summary": write_csv(combined_matching, output_root / "construction_matched_null_matching_summary.csv"),
        "matching_coverage": write_csv(combined_coverage, output_root / "construction_matched_null_matching_coverage.csv"),
        "opportunity_composition": write_csv(combined_composition, output_root / "construction_matched_null_opportunity_composition.csv"),
        "matching_cutpoints": write_csv(combined_cutpoints, output_root / "construction_matched_null_matching_cutpoints.csv"),
        "field_pair_diagnostics": write_csv(combined_field_pairs, output_root / "construction_matched_null_field_pair_diagnostics.csv"),
        "cellwise_excess_diagnostics": write_csv(combined_cellwise, output_root / "construction_matched_null_cellwise_excess_diagnostics.csv"),
        "claim_support": write_csv(combined_claim_support, output_root / "construction_matched_null_claim_support.csv"),
        "numerical_audit": write_csv(combined_audit, output_root / "construction_matched_null_numerical_audit.csv"),
        "publication_ledger": write_csv(publication_ledger, output_root / "construction_matched_null_publication_ledger.csv"),
        "value_map": write_csv(value_map, output_root / "construction_matched_null_value_map.csv"),
    }
    if cross_field is not None and cross_metrics is not None:
        table_outputs["cross_split_field_replication"] = write_csv(
            cross_field, output_root / "construction_matched_null_cross_split_field_replication.csv"
        )
        table_outputs["cross_split_metric_replication"] = write_csv(
            cross_metrics, output_root / "construction_matched_null_cross_split_metric_replication.csv"
        )

    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    report_text = build_markdown_report(
        split_results=split_results,
        quality_gates=quality_gates,
        publication_ledger=publication_ledger,
        input_audit=combined_input_audit,
        cross_field=cross_field,
        cross_metrics=cross_metrics,
        generated_at=generated_at,
    )
    report_path = output_root / "construction_matched_null_numeric_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    source_hashes: Dict[str, str] = {}
    for split, bundle in bundles.items():
        source_hashes[f"{split}_manifest"] = file_sha256(
            bundle.root / "metadata" / f"{split}_construction_null_manifest.json"
        )
        source_hashes[f"{split}_array_archive"] = file_sha256(
            bundle.root / "arrays" / f"{split}_construction_null_fields.npz"
        )
    manifest = {
        "script": Path(__file__).name,
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "created_at": generated_at,
        "output_root": str(output_root),
        "validation_root": str(args.validation_root.resolve()),
        "confirmation_root": str(args.confirmation_root.resolve()) if args.confirmation_root is not None else None,
        "splits": list(split_results.keys()),
        "primary_split": PRIMARY_SPLIT,
        "confirmation_policy": (
            "B_confirm is reported only as a separate frozen output-only replication; no p-values are pooled"
        ),
        "minimum_replicates_required": int(args.minimum_replicates),
        "maximum_weak_fallback_fraction": float(args.maximum_weak_fallback_fraction),
        "formal_alpha": float(args.alpha),
        "allow_smoke_test": bool(args.allow_smoke_test),
        "report_path": str(report_path),
        "table_outputs": {name: str(path) for name, path in table_outputs.items()},
        "source_hashes": source_hashes,
        "quality_gate_count": int(len(quality_gates)),
        "quality_gate_failures_required": int(
            np.sum(quality_gates["required"].astype(bool) & ~quality_gates["passed"].astype(bool))
        ),
        "quality_gate_unavailable_optional_checks": int(
            np.sum(~quality_gates["required"].astype(bool) & ~quality_gates["passed"].astype(bool))
        ),
        "interpretation_boundary": {
            "formal_inference": [FULL_FIELD_METRIC, *(name for name, _ in EXPECTED_PRIMARY_METRICS)],
            "pure_ratio_baseline": "descriptive only",
            "excess_field_geometry": "descriptive only",
            "cellwise_null_radius": "descriptive only; no multiplicity-adjusted cellwise claim",
            "causal_claim": False,
            "coordinate_uniqueness_claim": False,
            "complete_Markov_closure_claim": False,
        },
    }
    manifest_path = output_root / "construction_matched_null_numeric_report_manifest.json"
    save_json(manifest, manifest_path)

    print(f"[construction-null report] wrote: {report_path}")
    print(f"[construction-null report] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
