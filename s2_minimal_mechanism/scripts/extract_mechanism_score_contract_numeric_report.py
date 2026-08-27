#!/usr/bin/env python3
"""Extract a complete numerical report for mechanism score-contract and Pareto robustness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_SCORE_ROOT = Path(
    os.environ.get(
        "MECH_SCORE_CONTRACT_ROOT",
        "/data/datasets/KT4/outputs_KT4/stage2_phase1_score_contract_robustness",
    )
)
PRIMARY_COMPONENTS = (
    "one_step_mse_main_norm",
    "occupancy_js_MR_PsiA",
    "drift_local_rmse_loss_MR_PsiA",
    "drift_direction_loss_MR_PsiA",
    "drift_magnitude_loss_MR_PsiA",
)
COMPONENT_LABELS = {
    "one_step_mse_main_norm": "One-step closure loss",
    "occupancy_js_MR_PsiA": "Next-state landscape JS",
    "drift_local_rmse_loss_MR_PsiA": "Local drift loss",
    "drift_direction_loss_MR_PsiA": "Drift-direction loss",
    "drift_magnitude_loss_MR_PsiA": "Drift-magnitude loss",
}
EXPECTED_CONTRACTS = (
    "formal",
    "equal_primary",
    "omit_step",
    "omit_js",
    "omit_local",
    "omit_direction",
    "omit_magnitude",
)
CONTRACT_LABELS = {
    "formal": "Formal weights",
    "equal_primary": "Equal primary-component weights",
    "omit_step": "Omit one-step closure",
    "omit_js": "Omit landscape JS",
    "omit_local": "Omit local drift",
    "omit_direction": "Omit drift direction",
    "omit_magnitude": "Omit drift magnitude",
}
PARAMETER_COLUMNS = (
    "theta0",
    "thetaM",
    "thetaPsi",
    "thetaMPsi",
    "phi0",
    "deltaS",
    "phiPsi",
    "lambdaR",
    "lambdaA",
    "lambdaI",
)
STRUCTURAL_METRIC_COLUMNS = (
    "Training structural loss",
    "Validation structural loss",
    "Validation primary score",
    "One-step closure discrepancy",
    "Landscape divergence",
    "Local drift discrepancy",
    "Drift-direction discrepancy",
    "Drift-speed discrepancy",
    "Signed response next-state RMSE",
    "Exposure-alignment next-state RMSE",
    "Bootstrap mean primary score",
    "Bootstrap standard error",
    "Bootstrap 95% CI lower",
    "Bootstrap 95% CI upper",
    "difference_to_best_mean",
    "difference_to_best_ci95_lower",
    "difference_to_best_ci95_upper",
)
WEIGHT_COLUMNS = tuple(f"weight_{name}" for name in PRIMARY_COMPONENTS)


def now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int)):
        return obj
    if pd.isna(obj):
        return None
    return str(obj)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(obj: Any) -> str:
    payload = json.dumps(
        json_safe(obj),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(obj), handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    compression = "gzip" if path.name.endswith(".gz") else None
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required table not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def require_path(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Required file not found: {resolved}")
    return resolved


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing required columns: {missing}")


def coerce_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return False
        return bool(int(value))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return False


def certify_unchanged_inputs(
    manifest: Mapping[str, Any],
    label: str,
) -> Tuple[bool, Dict[str, Any]]:
    direct = manifest.get("formal_inputs_unchanged")
    if direct is not None:
        return coerce_bool(direct), {
            "source": f"{label}.formal_inputs_unchanged",
            "observed": direct,
        }

    before = manifest.get("formal_inputs_snapshot_before")
    after = manifest.get("formal_inputs_snapshot_after")
    if (
        isinstance(before, Mapping)
        and isinstance(after, Mapping)
        and bool(before)
        and bool(after)
    ):
        before_hash = stable_json_hash(before)
        after_hash = stable_json_hash(after)
        return before_hash == after_hash, {
            "source": f"{label}.formal_inputs_snapshots",
            "before_sha256": before_hash,
            "after_sha256": after_hash,
        }

    guardrails = manifest.get("guardrails")
    if (
        isinstance(guardrails, Mapping)
        and "formal_outputs_modified" in guardrails
    ):
        modified = guardrails.get("formal_outputs_modified")
        return not coerce_bool(modified), {
            "source": f"{label}.guardrails.formal_outputs_modified",
            "observed": modified,
        }

    for key in ("formal_outputs_modified", "formal_phase1_outputs_modified"):
        if key in manifest and manifest.get(key) is not None:
            modified = manifest.get(key)
            return not coerce_bool(modified), {
                "source": f"{label}.{key}",
                "observed": modified,
            }

    return False, {
        "source": "missing",
        "required_evidence": [
            "formal_inputs_unchanged",
            "matching formal input snapshots",
            "formal_outputs_modified=False",
            "formal_phase1_outputs_modified=False",
        ],
    }


def bool_series(series: pd.Series) -> pd.Series:
    return series.map(coerce_bool).astype(bool)


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite numerical value for {label}: {value!r}")
    return result


def integer_value(value: Any, label: str) -> int:
    result = int(value)
    return result


def verify_manifest_checksum(manifest_path: Path) -> Dict[str, Any]:
    sidecar = manifest_path.with_name(manifest_path.stem + ".sha256.json")
    sidecar = require_path(sidecar)
    payload = load_json(sidecar)
    expected = str(payload.get("manifest_sha256", "") or "")
    actual = sha256_file(manifest_path)
    if not expected or expected != actual:
        raise RuntimeError(
            f"Manifest checksum verification failed for {manifest_path}: "
            f"expected={expected!r}, actual={actual!r}"
        )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": actual,
        "sidecar": str(sidecar),
        "verified": True,
    }


def snapshot_files(paths: Mapping[str, Path]) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for name, path in paths.items():
        resolved = require_path(path)
        snapshot[name] = {
            "path": str(resolved),
            "size_bytes": int(resolved.stat().st_size),
            "sha256": sha256_file(resolved),
        }
    return snapshot


def assert_snapshots_equal(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> None:
    if stable_json_hash(before) != stable_json_hash(after):
        raise RuntimeError("Source robustness outputs changed during report extraction.")


def prepare_output_root(output_root: Path, protected_roots: Sequence[Path], overwrite: bool) -> None:
    resolved = output_root.resolve()
    for protected in protected_roots:
        protected_resolved = protected.resolve()
        if resolved == protected_resolved or resolved in protected_resolved.parents:
            raise RuntimeError(
                f"Report output root cannot replace or contain an input root: {resolved}"
            )
        if protected_resolved in resolved.parents and protected_resolved.name in {
            "formal_audit",
            "equal_primary_rerun",
        }:
            raise RuntimeError(
                f"Report output root cannot be nested inside a computational input root: {resolved}"
            )
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Report output directory is not empty: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def add_check(
    rows: List[Dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
) -> None:
    rows.append(
        {
            "check": name,
            "passed": bool(passed),
            "observed": json.dumps(json_safe(observed), ensure_ascii=False, sort_keys=True),
            "expected": json.dumps(json_safe(expected), ensure_ascii=False, sort_keys=True),
        }
    )
    if not passed:
        raise RuntimeError(
            f"Report integrity check failed: {name}; observed={observed!r}; expected={expected!r}"
        )


def format_number(value: Any, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return "--"
    if number == 0:
        return "0"
    if abs(number) < 1e-4 or abs(number) >= 1e5:
        return f"{number:.4e}"
    text = f"{number:.{digits}f}"
    return text.rstrip("0").rstrip(".")


def format_bool(value: Any) -> str:
    return "Yes" if coerce_bool(value) else "No"


def markdown_escape(value: Any) -> str:
    if value is None:
        return "--"
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text if text else "--"


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], headers: Optional[Sequence[str]] = None) -> str:
    if frame.empty:
        return "No rows."
    headers = list(headers) if headers is not None else list(columns)
    lines = [
        "| " + " | ".join(markdown_escape(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            + " | ".join(markdown_escape(row.get(column, "--")) for column in columns)
            + " |"
        )
    return "\n".join(lines)


def semicolon_values(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def component_bootstrap_summary(component_boot: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for family_key, family_frame in component_boot.groupby("family_key", sort=True):
        for component in PRIMARY_COMPONENTS:
            values = pd.to_numeric(family_frame[component], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                raise RuntimeError(f"No finite bootstrap values for {family_key}/{component}.")
            records.append(
                {
                    "family_key": str(family_key),
                    "component": component,
                    "component_label": COMPONENT_LABELS[component],
                    "bootstrap_reps": int(values.size),
                    "mean": float(np.mean(values)),
                    "standard_deviation": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "ci95_lower": float(np.quantile(values, 0.025)),
                    "median": float(np.quantile(values, 0.5)),
                    "ci95_upper": float(np.quantile(values, 0.975)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
    return pd.DataFrame(records)


def supported_drift_cell_summary(component_boot: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for family_key, family_frame in component_boot.groupby("family_key", sort=True):
        values = pd.to_numeric(
            family_frame["supported_drift_cells"], errors="coerce"
        ).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise RuntimeError(f"No finite supported-cell values for {family_key}.")
        records.append(
            {
                "family_key": str(family_key),
                "bootstrap_reps": int(values.size),
                "mean_supported_drift_cells": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "ci95_lower": float(np.quantile(values, 0.025)),
                "median": float(np.quantile(values, 0.5)),
                "ci95_upper": float(np.quantile(values, 0.975)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        )
    return pd.DataFrame(records)


def selected_contract_table(
    contract_summary: pd.DataFrame,
    contract_details: pd.DataFrame,
    formal_family: str,
    formal_parameter_count: int,
) -> pd.DataFrame:
    details = contract_details.copy()
    require_columns(details, ["contract", "family_key", "selected"], "Contract family details")
    details["selected"] = bool_series(details["selected"])
    selected = details.loc[details["selected"]].copy()
    if selected.groupby("contract").size().ne(1).any():
        raise RuntimeError("Each frozen-fit score contract must have exactly one selected family.")
    selected_columns = [
        "contract",
        "family_key",
        "bootstrap_mean",
        "bootstrap_sd_used_as_one_se",
        "bootstrap_ci95_lower",
        "bootstrap_ci95_upper",
        "difference_to_best_mean",
        "difference_to_best_ci95_lower",
        "difference_to_best_ci95_upper",
        "within_one_standard_error_of_best",
        "practically_equivalent_to_best",
        "eligible",
    ]
    selected = selected[[column for column in selected_columns if column in selected.columns]].rename(
        columns={
            "family_key": "selected_family_key_from_details",
            "bootstrap_mean": "selected_bootstrap_mean",
            "bootstrap_sd_used_as_one_se": "selected_bootstrap_sd_used_as_one_se",
            "bootstrap_ci95_lower": "selected_bootstrap_ci95_lower",
            "bootstrap_ci95_upper": "selected_bootstrap_ci95_upper",
            "difference_to_best_mean": "selected_difference_to_best_mean_from_details",
            "difference_to_best_ci95_lower": "selected_difference_to_best_ci95_lower",
            "difference_to_best_ci95_upper": "selected_difference_to_best_ci95_upper_from_details",
            "within_one_standard_error_of_best": "selected_within_one_se_from_details",
            "practically_equivalent_to_best": "selected_practically_equivalent_from_details",
            "eligible": "selected_eligible_from_details",
        }
    )
    merged = contract_summary.merge(selected, on="contract", how="left", validate="one_to_one")
    if not (
        merged["selected_family_key"].astype(str)
        == merged["selected_family_key_from_details"].astype(str)
    ).all():
        raise RuntimeError("Contract summary and family-detail selections disagree.")
    merged["contract_label"] = merged["contract"].map(CONTRACT_LABELS).fillna(merged["contract"])
    merged["same_as_formal_family"] = merged["selected_family_key"].astype(str) == formal_family
    merged["parameter_count_difference_from_formal"] = (
        pd.to_numeric(merged["selected_parameter_count"], errors="raise").astype(int)
        - int(formal_parameter_count)
    )
    order = {name: index for index, name in enumerate(EXPECTED_CONTRACTS)}
    merged["contract_order"] = merged["contract"].map(order)
    merged = merged.sort_values("contract_order", kind="mergesort").drop(columns="contract_order")
    return merged


def pareto_dominator_frequency(pareto_bootstrap: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    total = int(pareto_bootstrap["bootstrap_rep"].nunique())
    for kind, column in (
        ("not_more_complex", "not_more_complex_dominators"),
        ("strictly_simpler", "strictly_simpler_dominators"),
    ):
        counts: Dict[str, int] = {}
        for value in pareto_bootstrap[column].tolist():
            for family in set(semicolon_values(value)):
                counts[family] = counts.get(family, 0) + 1
        for family, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            records.append(
                {
                    "dominator_type": kind,
                    "family_key": family,
                    "bootstrap_count": int(count),
                    "bootstrap_reps": total,
                    "bootstrap_frequency": float(count / max(total, 1)),
                }
            )
    return pd.DataFrame(
        records,
        columns=[
            "dominator_type",
            "family_key",
            "bootstrap_count",
            "bootstrap_reps",
            "bootstrap_frequency",
        ],
    )


def parameter_comparison(
    formal_results: pd.DataFrame,
    formal_family: str,
    equal_summary: Mapping[str, Any],
) -> pd.DataFrame:
    formal_rows = formal_results.loc[formal_results["family_key"].astype(str) == formal_family]
    if len(formal_rows) != 1:
        raise RuntimeError("Formal selected family must appear exactly once in the formal results table.")
    formal_row = formal_rows.iloc[0]
    equal_family = str(equal_summary.get("final_selected_family", ""))
    equal_values = dict(equal_summary.get("final_selected_parameters", {}))
    free_parameters = set(str(value) for value in equal_summary.get("final_free_mechanism_parameters", []))
    records: List[Dict[str, Any]] = []
    for parameter in PARAMETER_COLUMNS:
        formal_value = formal_row.get(parameter, np.nan)
        equal_value = equal_values.get(parameter, np.nan)
        formal_finite = pd.notna(formal_value) and math.isfinite(float(formal_value))
        equal_finite = pd.notna(equal_value) and math.isfinite(float(equal_value))
        records.append(
            {
                "parameter": parameter,
                "formal_family": formal_family,
                "equal_primary_family": equal_family,
                "same_family": formal_family == equal_family,
                "equal_primary_parameter_role": "free" if parameter in free_parameters else "fixed_or_structural_zero",
                "formal_value": float(formal_value) if formal_finite else np.nan,
                "equal_primary_value": float(equal_value) if equal_finite else np.nan,
                "equal_minus_formal": (
                    float(equal_value) - float(formal_value)
                    if formal_finite and equal_finite and formal_family == equal_family
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def deletion_round_summary(deletions: pd.DataFrame) -> pd.DataFrame:
    if deletions.empty or "round" not in deletions.columns:
        return pd.DataFrame(
            columns=[
                "round",
                "current_family",
                "tests",
                "globally_required",
                "globally_removable",
                "parameters_tested",
            ]
        )
    frame = deletions.copy()
    for column in (
        "globally_required",
        "globally_eligible_under_selection_rule",
    ):
        if column in frame.columns:
            frame[column] = bool_series(frame[column])
    records: List[Dict[str, Any]] = []
    for round_value, group in frame.groupby("round", sort=True):
        records.append(
            {
                "round": int(round_value),
                "current_family": ";".join(sorted(group["current_family"].astype(str).unique())),
                "tests": int(len(group)),
                "globally_required": int(group.get("globally_required", pd.Series(False, index=group.index)).sum()),
                "globally_removable": int(group.get("globally_eligible_under_selection_rule", pd.Series(False, index=group.index)).sum()),
                "parameters_tested": ";".join(group["tested_removed_parameter"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(records)


def score_bootstrap_summary(frame: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for family_key, group in frame.groupby("family_key", sort=True):
        values = pd.to_numeric(group["primary_score"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise RuntimeError(f"No finite equal-primary bootstrap scores for {family_key}.")
        records.append(
            {
                "family_key": str(family_key),
                "bootstrap_reps": int(values.size),
                "bootstrap_mean_primary_score": float(np.mean(values)),
                "bootstrap_standard_deviation_used_as_one_se": (
                    float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                ),
                "bootstrap_ci95_lower": float(np.quantile(values, 0.025)),
                "bootstrap_median": float(np.quantile(values, 0.5)),
                "bootstrap_ci95_upper": float(np.quantile(values, 0.975)),
                "bootstrap_minimum": float(np.min(values)),
                "bootstrap_maximum": float(np.max(values)),
            }
        )
    return pd.DataFrame(records)


def selected_structural_metrics(
    formal_results: pd.DataFrame,
    formal_family: str,
    equal_results: pd.DataFrame,
    equal_family: str,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for analysis, frame, family in (
        ("formal", formal_results, formal_family),
        ("equal_primary_full_reoptimisation", equal_results, equal_family),
    ):
        rows = frame.loc[frame["family_key"].astype(str) == family]
        if len(rows) != 1:
            raise RuntimeError(f"Selected family {family!r} is not unique for {analysis}.")
        row = rows.iloc[0]
        for metric in STRUCTURAL_METRIC_COLUMNS:
            if metric not in frame.columns:
                continue
            value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
            records.append(
                {
                    "analysis": analysis,
                    "family_key": family,
                    "metric": metric,
                    "value": float(value) if pd.notna(value) else np.nan,
                }
            )
    return pd.DataFrame(records)


def bootstrap_verification_summary(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty:
        return {
            "available": False,
            "rows": 0,
            "difference_columns": [],
            "maximum_absolute_difference": None,
        }
    candidates = [
        column
        for column in frame.columns
        if any(token in column.lower() for token in ("difference", "error", "absolute"))
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    maxima: Dict[str, float] = {}
    for column in candidates:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            maxima[column] = float(np.max(np.abs(values)))
    return {
        "available": True,
        "rows": int(len(frame)),
        "difference_columns": candidates,
        "maximum_absolute_difference_by_column": maxima,
        "maximum_absolute_difference": max(maxima.values()) if maxima else None,
    }


def markdown_numeric_frame(frame: pd.DataFrame, numeric_columns: Iterable[str], bool_columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in numeric_columns:
        if column in out.columns:
            out[column] = out[column].map(format_number)
    for column in bool_columns:
        if column in out.columns:
            out[column] = out[column].map(format_bool)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-root", type=Path, default=DEFAULT_SCORE_ROOT)
    parser.add_argument("--formal-audit-root", type=Path, default=None)
    parser.add_argument("--equal-rerun-root", type=Path, default=None)
    parser.add_argument("--formal-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    score_root = require_path(args.score_root)
    formal_audit_root = require_path(args.formal_audit_root or score_root / "formal_audit")
    equal_root = require_path(args.equal_rerun_root or score_root / "equal_primary_rerun")
    output_root = (args.output_root or score_root / "numeric_report").resolve()
    prepare_output_root(output_root, [formal_audit_root, equal_root], args.overwrite)
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    table_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {
        "combined_summary_table": score_root / "tables" / "score_contract_robustness_summary.csv",
        "combined_summary_json": score_root / "metadata" / "score_contract_robustness_summary.json",
        "combined_manifest": score_root / "metadata" / "score_contract_robustness_manifest.json",
        "formal_audit_manifest": formal_audit_root / "metadata" / "formal_score_contract_audit_manifest.json",
        "pareto_summary": formal_audit_root / "metadata" / "selected_family_pareto_summary.json",
        "component_bootstrap": formal_audit_root / "tables" / "component_bootstrap_losses.csv.gz",
        "formal_score_reconstruction": formal_audit_root / "tables" / "formal_score_reconstruction_audit.csv.gz",
        "contract_selection": formal_audit_root / "tables" / "frozen_fit_score_contract_selection_summary.csv",
        "contract_family_details": formal_audit_root / "tables" / "frozen_fit_score_contract_family_details.csv.gz",
        "pareto_mean": formal_audit_root / "tables" / "frozen_fit_pareto_mean_components.csv",
        "pareto_bootstrap": formal_audit_root / "tables" / "selected_family_pareto_bootstrap.csv.gz",
        "equal_manifest": equal_root / "metadata" / "score_contract_rerun_manifest.json",
        "equal_summary": equal_root / "metadata" / "score_contract_sensitivity_summary.json",
        "equal_contract_summary": equal_root / "tables" / "objective_contract_rerun_summary.csv",
        "equal_family_results": equal_root / "tables" / "model_family_results.csv",
        "equal_bootstrap_scores": equal_root / "tables" / "model_family_bootstrap_scores.csv.gz",
        "equal_deletions": equal_root / "tables" / "selected_model_parameter_deletions.csv",
        "equal_final_deletion_audit": equal_root / "tables" / "global_scalar_deletion_audit.csv",
        "equal_boundaries": equal_root / "tables" / "parameter_grid_boundaries.csv",
        "equal_next_required_tests": equal_root / "tables" / "next_required_tests.csv",
    }
    configuration_candidates = (
        score_root / "metadata" / "equal_primary_configuration_audit.json",
        score_root / "metadata" / "equal_weight_configuration_audit.json",
    )
    configuration_path = next((path for path in configuration_candidates if path.exists()), None)
    if configuration_path is None:
        raise FileNotFoundError("Equal-primary configuration audit was not found.")
    paths["configuration_audit"] = configuration_path
    optional_verification = equal_root / "tables" / "optimized_bootstrap_equivalence_checks.csv"
    if optional_verification.exists():
        paths["equal_bootstrap_verification"] = optional_verification

    before = snapshot_files(paths)
    combined_manifest_checksum = verify_manifest_checksum(require_path(paths["combined_manifest"]))
    formal_manifest_checksum = verify_manifest_checksum(require_path(paths["formal_audit_manifest"]))
    equal_manifest_checksum = verify_manifest_checksum(require_path(paths["equal_manifest"]))

    combined_summary_table = read_table(paths["combined_summary_table"])
    combined_summary = load_json(paths["combined_summary_json"])
    combined_manifest = load_json(paths["combined_manifest"])
    formal_manifest = load_json(paths["formal_audit_manifest"])
    pareto_summary = load_json(paths["pareto_summary"])
    component_boot = read_table(paths["component_bootstrap"])
    reconstruction = read_table(paths["formal_score_reconstruction"])
    contract_summary = read_table(paths["contract_selection"])
    contract_details = read_table(paths["contract_family_details"])
    pareto_mean = read_table(paths["pareto_mean"])
    pareto_bootstrap = read_table(paths["pareto_bootstrap"])
    equal_manifest = load_json(paths["equal_manifest"])
    equal_summary = load_json(paths["equal_summary"])
    equal_contract_summary = read_table(paths["equal_contract_summary"])
    equal_family_results = read_table(paths["equal_family_results"])
    equal_bootstrap_scores = read_table(paths["equal_bootstrap_scores"])
    equal_deletions = read_table(paths["equal_deletions"])
    equal_final_deletion = read_table(paths["equal_final_deletion_audit"])
    equal_boundaries = read_table(paths["equal_boundaries"])
    equal_next_tests = read_table(paths["equal_next_required_tests"])
    configuration_audit = load_json(paths["configuration_audit"])
    verification = (
        read_table(paths["equal_bootstrap_verification"])
        if "equal_bootstrap_verification" in paths
        else pd.DataFrame()
    )

    formal_root = (
        require_path(args.formal_root)
        if args.formal_root is not None
        else require_path(Path(str(formal_manifest.get("formal_output_root", ""))))
    )
    formal_results_path = require_path(formal_root / "tables" / "model_family_results.csv")
    formal_results = read_table(formal_results_path)
    paths["formal_family_results"] = formal_results_path
    before["formal_family_results"] = {
        "path": str(formal_results_path),
        "size_bytes": int(formal_results_path.stat().st_size),
        "sha256": sha256_file(formal_results_path),
    }

    formal_inputs_unchanged, formal_input_evidence = certify_unchanged_inputs(
        formal_manifest,
        "formal_manifest",
    )
    equal_inputs_unchanged, equal_input_evidence = certify_unchanged_inputs(
        equal_manifest,
        "equal_manifest",
    )

    checks: List[Dict[str, Any]] = []
    add_check(checks, "combined_manifest_checksum", combined_manifest_checksum["verified"], True, True)
    add_check(checks, "formal_audit_manifest_checksum", formal_manifest_checksum["verified"], True, True)
    add_check(checks, "equal_rerun_manifest_checksum", equal_manifest_checksum["verified"], True, True)
    add_check(checks, "combined_B_confirm_not_read", not coerce_bool(combined_manifest.get("B_confirm_read", True)), combined_manifest.get("B_confirm_read"), False)
    add_check(checks, "summary_B_confirm_not_read", not coerce_bool(combined_summary.get("B_confirm_read", True)), combined_summary.get("B_confirm_read"), False)
    add_check(checks, "formal_audit_B_confirm_not_read", not coerce_bool(formal_manifest.get("B_confirm_read", True)), formal_manifest.get("B_confirm_read"), False)
    add_check(checks, "equal_rerun_B_confirm_not_read", not coerce_bool(equal_manifest.get("B_confirm_read", True)), equal_manifest.get("B_confirm_read"), False)
    add_check(checks, "equal_summary_B_confirm_not_read", not coerce_bool(equal_summary.get("B_confirm_read", True)), equal_summary.get("B_confirm_read"), False)
    add_check(checks, "formal_outputs_unmodified", not coerce_bool(combined_manifest.get("formal_outputs_modified", True)), combined_manifest.get("formal_outputs_modified"), False)
    add_check(checks, "formal_family_or_parameters_unmodified", not coerce_bool(combined_manifest.get("formal_family_or_parameters_modified", True)), combined_manifest.get("formal_family_or_parameters_modified"), False)
    add_check(checks, "combined_not_phase2_eligible", not coerce_bool(combined_manifest.get("eligible_for_phase2_freeze", True)), combined_manifest.get("eligible_for_phase2_freeze"), False)
    add_check(checks, "equal_rerun_sensitivity_only", coerce_bool(equal_manifest.get("sensitivity_only", False)), equal_manifest.get("sensitivity_only"), True)
    add_check(checks, "equal_rerun_not_phase2_eligible", not coerce_bool(equal_manifest.get("eligible_for_phase2_freeze", True)), equal_manifest.get("eligible_for_phase2_freeze"), False)
    add_check(checks, "configuration_audit_passed", coerce_bool(configuration_audit.get("passed", False)), configuration_audit.get("passed"), True)
    add_check(checks, "configuration_audit_has_no_mismatches", len(configuration_audit.get("mismatches", [])) == 0, configuration_audit.get("mismatches", []), [])
    add_check(
        checks,
        "formal_inputs_unchanged",
        formal_inputs_unchanged,
        formal_input_evidence,
        True,
    )
    add_check(
        checks,
        "equal_formal_inputs_unchanged",
        equal_inputs_unchanged,
        equal_input_evidence,
        True,
    )
    add_check(checks, "formal_script_checksum_consistent", str(formal_manifest.get("formal_script_sha256")) == str(equal_manifest.get("formal_script_sha256")), equal_manifest.get("formal_script_sha256"), formal_manifest.get("formal_script_sha256"))

    require_columns(component_boot, ["bootstrap_rep", "family_key", "supported_drift_cells", *PRIMARY_COMPONENTS], "Component bootstrap table")
    require_columns(reconstruction, ["bootstrap_rep", "family_key", "formal_score_reconstructed", "formal_score_archived", "absolute_reconstruction_difference"], "Formal score reconstruction table")
    require_columns(contract_summary, ["contract", "best_family_key", "selected_family_key", "selected_parameter_count", *WEIGHT_COLUMNS], "Frozen-fit contract summary")
    require_columns(contract_details, ["contract", "family_key", "selected", "eligible"], "Frozen-fit contract family details")
    require_columns(pareto_mean, ["family_key", "Free mechanism parameters", *PRIMARY_COMPONENTS, "performance_pareto_front", "complexity_pareto_front"], "Pareto mean table")
    require_columns(pareto_bootstrap, ["bootstrap_rep", "selected_family_key", "any_not_more_complex_family_dominates", "any_strictly_simpler_family_dominates", "not_more_complex_dominators", "strictly_simpler_dominators"], "Pareto bootstrap table")
    require_columns(equal_contract_summary, ["objective_contract", "final_best_family", "final_selected_family", "final_parameter_count", "would_pass_selection_gates_under_contract", "would_satisfy_current_phase2_family_contract"], "Equal-primary contract summary")
    require_columns(
        combined_summary_table,
        [
            "analysis_type",
            "contract",
            "selected_family_key",
            "selected_parameter_count",
            "same_as_formal_final_family",
        ],
        "Combined score-contract summary",
    )
    require_columns(equal_family_results, ["family_key", "Model family", "Free mechanism parameters", "Bootstrap mean primary score", "Within one standard error of best", "Practically equivalent to best", "Final scalar-minimal family"], "Equal-primary family results")
    require_columns(equal_bootstrap_scores, ["bootstrap_rep", "family_key", "primary_score"], "Equal-primary bootstrap scores")
    require_columns(formal_results, ["family_key", "Model family", "Free mechanism parameters", *PARAMETER_COLUMNS], "Formal family results")

    component_family_set = set(component_boot["family_key"].astype(str))
    contract_family_set = set(contract_details["family_key"].astype(str))
    pareto_family_set = set(pareto_mean["family_key"].astype(str))
    formal_family_set = set(formal_results["family_key"].astype(str))
    add_check(checks, "formal_family_sets_align", component_family_set == contract_family_set == pareto_family_set == formal_family_set, sorted(component_family_set), sorted(formal_family_set))

    expected_reps = integer_value(formal_manifest.get("bootstrap_reps"), "formal bootstrap reps")
    actual_component_reps = int(component_boot["bootstrap_rep"].nunique())
    add_check(checks, "component_bootstrap_rep_count", actual_component_reps == expected_reps, actual_component_reps, expected_reps)
    per_family_reps = component_boot.groupby("family_key")["bootstrap_rep"].nunique()
    add_check(checks, "component_bootstrap_complete_by_family", bool((per_family_reps == expected_reps).all()), per_family_reps.to_dict(), expected_reps)
    for component in PRIMARY_COMPONENTS:
        values = pd.to_numeric(component_boot[component], errors="coerce").to_numpy(dtype=float)
        add_check(checks, f"finite_component_{component}", bool(np.isfinite(values).all()), int(np.isfinite(values).sum()), int(values.size))
        add_check(checks, f"bounded_component_{component}", bool(((values >= -1e-12) & (values <= 1.0 + 1e-12)).all()), [float(np.min(values)), float(np.max(values))], [0.0, 1.0])

    reconstruction_values = pd.to_numeric(reconstruction["absolute_reconstruction_difference"], errors="coerce").to_numpy(dtype=float)
    reconstruction_max = float(np.max(reconstruction_values))
    reconstruction_tolerance = finite_float(formal_manifest["formal_score_reconstruction"]["tolerance"], "reconstruction tolerance")
    add_check(checks, "formal_score_reconstruction_passed", coerce_bool(formal_manifest["formal_score_reconstruction"].get("passed", False)), formal_manifest["formal_score_reconstruction"].get("passed"), True)
    add_check(checks, "formal_score_reconstruction_within_tolerance", reconstruction_max <= reconstruction_tolerance, reconstruction_max, reconstruction_tolerance)

    observed_contracts = tuple(contract_summary["contract"].astype(str).tolist())
    add_check(checks, "complete_frozen_fit_contract_set", set(observed_contracts) == set(EXPECTED_CONTRACTS) and len(observed_contracts) == len(EXPECTED_CONTRACTS), sorted(observed_contracts), sorted(EXPECTED_CONTRACTS))
    weight_sums = contract_summary[list(WEIGHT_COLUMNS)].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    add_check(checks, "score_contract_weights_sum_to_one", bool(np.allclose(weight_sums.to_numpy(dtype=float), 1.0, atol=1e-9, rtol=0.0)), weight_sums.tolist(), 1.0)

    formal_family = str(formal_manifest.get("formal_final_family", ""))
    if not formal_family:
        raise RuntimeError("Formal audit manifest does not identify the formal final family.")
    formal_rows = formal_results.loc[formal_results["family_key"].astype(str) == formal_family]
    if len(formal_rows) != 1:
        raise RuntimeError("Formal final family is not unique in formal family results.")
    formal_parameter_count = int(formal_rows.iloc[0]["Free mechanism parameters"])
    formal_contract_row = contract_summary.loc[contract_summary["contract"].astype(str) == "formal"]
    add_check(checks, "formal_contract_replays_formal_family", len(formal_contract_row) == 1 and str(formal_contract_row.iloc[0]["selected_family_key"]) == formal_family, formal_contract_row["selected_family_key"].tolist(), formal_family)

    selected_contracts = selected_contract_table(
        contract_summary,
        contract_details,
        formal_family,
        formal_parameter_count,
    )
    component_stats = component_bootstrap_summary(component_boot)
    component_stats = component_stats.merge(
        formal_results[["family_key", "Model family", "Free mechanism parameters"]],
        on="family_key",
        how="left",
        validate="many_to_one",
    )
    support_stats = supported_drift_cell_summary(component_boot).merge(
        formal_results[["family_key", "Model family", "Free mechanism parameters"]],
        on="family_key",
        how="left",
        validate="one_to_one",
    )

    pareto_mean = pareto_mean.copy()
    pareto_mean["performance_pareto_front"] = bool_series(pareto_mean["performance_pareto_front"])
    pareto_mean["complexity_pareto_front"] = bool_series(pareto_mean["complexity_pareto_front"])
    pareto_bootstrap["any_not_more_complex_family_dominates"] = bool_series(pareto_bootstrap["any_not_more_complex_family_dominates"])
    pareto_bootstrap["any_strictly_simpler_family_dominates"] = bool_series(pareto_bootstrap["any_strictly_simpler_family_dominates"])
    pareto_reps = int(pareto_bootstrap["bootstrap_rep"].nunique())
    add_check(checks, "pareto_bootstrap_rep_count", pareto_reps == expected_reps, pareto_reps, expected_reps)
    computed_not_more_frequency = float(pareto_bootstrap["any_not_more_complex_family_dominates"].mean())
    computed_simpler_frequency = float(pareto_bootstrap["any_strictly_simpler_family_dominates"].mean())
    add_check(checks, "pareto_not_more_complex_frequency_matches_summary", abs(computed_not_more_frequency - finite_float(pareto_summary["bootstrap_frequency_any_not_more_complex_dominator"], "Pareto not-more-complex frequency")) <= 1e-12, computed_not_more_frequency, pareto_summary["bootstrap_frequency_any_not_more_complex_dominator"])
    add_check(checks, "pareto_simpler_frequency_matches_summary", abs(computed_simpler_frequency - finite_float(pareto_summary["bootstrap_frequency_any_strictly_simpler_dominator"], "Pareto simpler frequency")) <= 1e-12, computed_simpler_frequency, pareto_summary["bootstrap_frequency_any_strictly_simpler_dominator"])
    dominator_frequency = pareto_dominator_frequency(pareto_bootstrap)

    add_check(
        checks,
        "equal_contract_summary_single_row",
        len(equal_contract_summary) == 1,
        len(equal_contract_summary),
        1,
    )
    equal_contract_row = equal_contract_summary.iloc[0]
    add_check(checks, "equal_contract_is_equal_primary", str(equal_contract_row["objective_contract"]) == "equal_primary", equal_contract_row["objective_contract"], "equal_primary")
    add_check(checks, "equal_summary_matches_contract_table", str(equal_summary.get("final_selected_family")) == str(equal_contract_row["final_selected_family"]), equal_summary.get("final_selected_family"), equal_contract_row["final_selected_family"])
    add_check(checks, "equal_manifest_matches_summary", str(equal_manifest.get("results", {}).get("final_selected_family")) == str(equal_summary.get("final_selected_family")), equal_manifest.get("results", {}).get("final_selected_family"), equal_summary.get("final_selected_family"))
    equal_reps = int(equal_bootstrap_scores["bootstrap_rep"].nunique())
    add_check(checks, "equal_bootstrap_rep_count", equal_reps == expected_reps, equal_reps, expected_reps)
    equal_family_set = set(equal_family_results["family_key"].astype(str))
    equal_boot_family_set = set(equal_bootstrap_scores["family_key"].astype(str))
    add_check(checks, "equal_family_and_bootstrap_sets_align", equal_family_set == equal_boot_family_set, sorted(equal_family_set), sorted(equal_boot_family_set))
    equal_family_results = equal_family_results.copy()
    for column in (
        "Within one standard error of best",
        "Practically equivalent to best",
        "Parsimonious family selected",
        "Final scalar-minimal family",
        "Sensitivity only",
    ):
        if column in equal_family_results.columns:
            equal_family_results[column] = bool_series(equal_family_results[column])
    equal_bootstrap_stats = score_bootstrap_summary(equal_bootstrap_scores)
    equal_bootstrap_comparison = equal_bootstrap_stats.merge(
        equal_family_results[[
            "family_key",
            "Bootstrap mean primary score",
            "Bootstrap standard error",
            "Bootstrap 95% CI lower",
            "Bootstrap 95% CI upper",
        ]],
        on="family_key",
        how="left",
        validate="one_to_one",
    )
    equal_bootstrap_differences = {
        "mean": float(np.max(np.abs(
            pd.to_numeric(equal_bootstrap_comparison["bootstrap_mean_primary_score"], errors="coerce").to_numpy(dtype=float)
            - pd.to_numeric(equal_bootstrap_comparison["Bootstrap mean primary score"], errors="coerce").to_numpy(dtype=float)
        ))),
        "standard_deviation": float(np.max(np.abs(
            pd.to_numeric(equal_bootstrap_comparison["bootstrap_standard_deviation_used_as_one_se"], errors="coerce").to_numpy(dtype=float)
            - pd.to_numeric(equal_bootstrap_comparison["Bootstrap standard error"], errors="coerce").to_numpy(dtype=float)
        ))),
        "ci95_lower": float(np.max(np.abs(
            pd.to_numeric(equal_bootstrap_comparison["bootstrap_ci95_lower"], errors="coerce").to_numpy(dtype=float)
            - pd.to_numeric(equal_bootstrap_comparison["Bootstrap 95% CI lower"], errors="coerce").to_numpy(dtype=float)
        ))),
        "ci95_upper": float(np.max(np.abs(
            pd.to_numeric(equal_bootstrap_comparison["bootstrap_ci95_upper"], errors="coerce").to_numpy(dtype=float)
            - pd.to_numeric(equal_bootstrap_comparison["Bootstrap 95% CI upper"], errors="coerce").to_numpy(dtype=float)
        ))),
    }
    add_check(
        checks,
        "equal_bootstrap_summary_matches_family_table",
        max(equal_bootstrap_differences.values()) <= 1e-10,
        equal_bootstrap_differences,
        "maximum absolute difference <= 1e-10",
    )
    equal_final_rows = equal_family_results.loc[equal_family_results["family_key"].astype(str) == str(equal_summary.get("final_selected_family"))]
    add_check(checks, "equal_final_family_unique", len(equal_final_rows) == 1, len(equal_final_rows), 1)
    add_check(checks, "equal_final_family_flagged", coerce_bool(equal_final_rows.iloc[0]["Final scalar-minimal family"]), equal_final_rows.iloc[0]["Final scalar-minimal family"], True)

    equal_final_family = str(equal_summary.get("final_selected_family", "")).strip()
    equal_final_parameter_count = int(equal_contract_row["final_parameter_count"])
    add_check(
        checks,
        "combined_manifest_embeds_combined_summary",
        isinstance(combined_manifest.get("combined_summary"), Mapping)
        and stable_json_hash(combined_manifest.get("combined_summary"))
        == stable_json_hash(combined_summary),
        combined_manifest.get("combined_summary"),
        combined_summary,
    )
    add_check(
        checks,
        "combined_summary_formal_family_matches",
        str(combined_summary.get("formal_final_family", "")).strip() == formal_family,
        combined_summary.get("formal_final_family"),
        formal_family,
    )
    add_check(
        checks,
        "combined_summary_equal_family_matches",
        str(
            combined_summary.get(
                "full_equal_primary_reoptimisation_selected_family",
                "",
            )
        ).strip()
        == equal_final_family,
        combined_summary.get(
            "full_equal_primary_reoptimisation_selected_family"
        ),
        equal_final_family,
    )
    add_check(
        checks,
        "combined_summary_equal_parameter_count_matches",
        int(
            combined_summary.get(
                "full_equal_primary_reoptimisation_parameter_count",
                -1,
            )
        )
        == equal_final_parameter_count,
        combined_summary.get(
            "full_equal_primary_reoptimisation_parameter_count"
        ),
        equal_final_parameter_count,
    )
    add_check(
        checks,
        "combined_summary_equal_gate_status_matches",
        coerce_bool(
            combined_summary.get(
                "full_equal_primary_reoptimisation_would_pass_selection_gates",
                False,
            )
        )
        == coerce_bool(
            equal_summary.get(
                "would_pass_selection_gates_under_contract",
                False,
            )
        ),
        combined_summary.get(
            "full_equal_primary_reoptimisation_would_pass_selection_gates"
        ),
        equal_summary.get("would_pass_selection_gates_under_contract"),
    )
    add_check(
        checks,
        "combined_summary_equal_phase2_contract_status_matches",
        coerce_bool(
            combined_summary.get(
                "full_equal_primary_reoptimisation_would_satisfy_current_phase2_family_contract",
                False,
            )
        )
        == coerce_bool(
            equal_summary.get(
                "would_satisfy_current_phase2_family_contract",
                False,
            )
        ),
        combined_summary.get(
            "full_equal_primary_reoptimisation_would_satisfy_current_phase2_family_contract"
        ),
        equal_summary.get(
            "would_satisfy_current_phase2_family_contract"
        ),
    )

    parameter_table = parameter_comparison(formal_results, formal_family, equal_summary)
    selected_metrics = selected_structural_metrics(
        formal_results,
        formal_family,
        equal_family_results,
        equal_final_family,
    )
    deletion_rounds = deletion_round_summary(equal_deletions)
    verification_summary = bootstrap_verification_summary(verification)

    boundary_blockers = list(equal_summary.get("boundary_blockers", []))
    next_required_tests = list(equal_summary.get("next_required_tests", []))
    add_check(checks, "equal_boundary_blockers_match_next_tests", len(next_required_tests) >= len(boundary_blockers), len(next_required_tests), f">={len(boundary_blockers)}")

    combined_summary_table = combined_summary_table.copy()
    combined_summary_table["same_as_formal_final_family"] = bool_series(combined_summary_table["same_as_formal_final_family"])
    if "full_parameter_reoptimisation" in combined_summary_table.columns:
        combined_summary_table["full_parameter_reoptimisation"] = bool_series(combined_summary_table["full_parameter_reoptimisation"])
    add_check(checks, "combined_summary_contains_frozen_and_full_equal_rows", set(combined_summary_table.get("analysis_type", pd.Series(dtype=str)).astype(str)) == {"frozen_formal_fit_reweighting", "full_family_parameter_reoptimisation"}, sorted(set(combined_summary_table.get("analysis_type", pd.Series(dtype=str)).astype(str))), ["frozen_formal_fit_reweighting", "full_family_parameter_reoptimisation"])
    combined_equal_rows = combined_summary_table.loc[
        combined_summary_table["analysis_type"].astype(str)
        == "full_family_parameter_reoptimisation"
    ]
    add_check(
        checks,
        "combined_summary_full_equal_row_unique",
        len(combined_equal_rows) == 1,
        len(combined_equal_rows),
        1,
    )
    combined_equal_row = combined_equal_rows.iloc[0]
    add_check(
        checks,
        "combined_summary_table_equal_family_matches",
        str(combined_equal_row["selected_family_key"]).strip()
        == equal_final_family,
        combined_equal_row["selected_family_key"],
        equal_final_family,
    )
    add_check(
        checks,
        "combined_summary_table_equal_parameter_count_matches",
        int(combined_equal_row["selected_parameter_count"])
        == equal_final_parameter_count,
        combined_equal_row["selected_parameter_count"],
        equal_final_parameter_count,
    )

    formal_best_family = str(
        selected_contracts.loc[selected_contracts["contract"] == "formal", "best_family_key"].iloc[0]
    )
    selected_component_stats = component_stats.loc[
        component_stats["family_key"].isin({formal_family, formal_best_family})
    ].copy()
    selected_component_stats["family_role"] = np.where(
        selected_component_stats["family_key"] == formal_family,
        "formal_selected",
        "formal_best",
    )
    if formal_best_family == formal_family:
        selected_component_stats["family_role"] = "formal_selected_and_best"

    component_mean_wide = component_stats.pivot_table(
        index=["family_key", "Model family", "Free mechanism parameters"],
        columns="component",
        values="mean",
        aggfunc="first",
    ).reset_index()
    component_mean_wide.columns.name = None
    component_mean_wide = component_mean_wide.merge(
        pareto_mean[[
            "family_key",
            "performance_pareto_front",
            "complexity_pareto_front",
            "performance_dominators",
            "complexity_dominators",
        ]],
        on="family_key",
        how="left",
        validate="one_to_one",
    )

    frozen_stress = selected_contracts.loc[selected_contracts["contract"] != "formal"]
    report_flags = {
        "all_frozen_fit_stress_contracts_select_formal_family": bool(
            frozen_stress["same_as_formal_family"].all()
        ),
        "no_frozen_fit_stress_contract_selects_a_simpler_family": bool(
            (pd.to_numeric(frozen_stress["selected_parameter_count"], errors="raise") >= formal_parameter_count).all()
        ),
        "full_equal_primary_reoptimisation_selects_formal_family": equal_final_family == formal_family,
        "full_equal_primary_reoptimisation_retains_formal_parameter_count": equal_final_parameter_count == formal_parameter_count,
        "full_equal_primary_reoptimisation_passes_selection_gates": coerce_bool(equal_contract_row["would_pass_selection_gates_under_contract"]),
        "full_equal_primary_reoptimisation_satisfies_current_phase2_family_contract": coerce_bool(equal_contract_row["would_satisfy_current_phase2_family_contract"]),
        "formal_selected_family_on_performance_pareto_front": coerce_bool(pareto_summary.get("selected_on_performance_pareto_front", False)),
        "formal_selected_family_on_complexity_pareto_front": coerce_bool(pareto_summary.get("selected_on_complexity_pareto_front", False)),
        "no_strictly_simpler_bootstrap_dominator": computed_simpler_frequency == 0.0,
        "configuration_audit_passed": coerce_bool(configuration_audit.get("passed", False)),
        "formal_score_reconstruction_passed": reconstruction_max <= reconstruction_tolerance,
    }
    report_flags["exact_family_robust_under_reported_contracts"] = bool(
        report_flags["all_frozen_fit_stress_contracts_select_formal_family"]
        and report_flags["full_equal_primary_reoptimisation_selects_formal_family"]
        and report_flags["full_equal_primary_reoptimisation_passes_selection_gates"]
    )
    report_flags["four_parameter_complexity_robust_under_reported_contracts"] = bool(
        report_flags["no_frozen_fit_stress_contract_selects_a_simpler_family"]
        and report_flags["full_equal_primary_reoptimisation_retains_formal_parameter_count"]
        and report_flags["formal_selected_family_on_complexity_pareto_front"]
    )

    reconstruction_summary = {
        "rows": int(len(reconstruction)),
        "bootstrap_reps": int(reconstruction["bootstrap_rep"].nunique()),
        "families": int(reconstruction["family_key"].nunique()),
        "maximum_absolute_difference": reconstruction_max,
        "mean_absolute_difference": float(np.mean(reconstruction_values)),
        "p95_absolute_difference": float(np.quantile(reconstruction_values, 0.95)),
        "p99_absolute_difference": float(np.quantile(reconstruction_values, 0.99)),
        "tolerance": reconstruction_tolerance,
        "passed": reconstruction_max <= reconstruction_tolerance,
    }

    development_manifest = dict(formal_manifest.get("development_data_load_manifest", {}))
    formal_runtime = dict(formal_manifest.get("formal_runtime_contract", {}))
    analysis_contract = {
        "formal_family": formal_family,
        "formal_parameter_count": formal_parameter_count,
        "formal_best_family": formal_best_family,
        "bootstrap_reps": expected_reps,
        "decision_bootstrap_seed": formal_manifest.get("decision_bootstrap_seed"),
        "practical_equivalence_margin": formal_manifest.get("practical_equivalence_margin"),
        "one_se_implementation": formal_manifest.get("one_se_implementation"),
        "pareto_tolerance": pareto_summary.get("pareto_tolerance"),
        "objective_components": list(PRIMARY_COMPONENTS),
        "frozen_fit_contracts": list(EXPECTED_CONTRACTS),
        "full_parameter_reoptimisation_contract": "equal_primary",
        "B_confirm_read": False,
        "post_hoc": True,
    }

    integrity_frame = pd.DataFrame(checks)
    write_csv(integrity_frame, table_root / "report_integrity_checks.csv")
    write_csv(selected_contracts, table_root / "score_contract_selection.csv")
    write_csv(contract_details, table_root / "score_contract_family_details.csv.gz")
    write_csv(component_stats, table_root / "formal_component_bootstrap_summary.csv")
    write_csv(support_stats, table_root / "formal_supported_drift_cells_summary.csv")
    write_csv(formal_results, table_root / "formal_family_parameters_and_metrics.csv")
    write_csv(component_mean_wide, table_root / "formal_family_component_means_and_pareto.csv")
    write_csv(pareto_mean, table_root / "pareto_family_summary.csv")
    write_csv(dominator_frequency, table_root / "pareto_dominator_frequencies.csv")
    write_csv(equal_family_results, table_root / "equal_primary_family_results.csv")
    write_csv(equal_bootstrap_stats, table_root / "equal_primary_bootstrap_summary.csv")
    write_csv(selected_metrics, table_root / "selected_family_structural_metrics.csv")
    write_csv(parameter_table, table_root / "selected_parameter_comparison.csv")
    write_csv(equal_deletions, table_root / "equal_primary_deletion_results.csv")
    write_csv(equal_final_deletion, table_root / "equal_primary_final_deletion_audit.csv")
    write_csv(deletion_rounds, table_root / "equal_primary_deletion_round_summary.csv")
    write_csv(equal_boundaries, table_root / "equal_primary_boundary_audit.csv")
    write_csv(equal_next_tests, table_root / "equal_primary_next_required_tests.csv")
    if not verification.empty:
        write_csv(verification, table_root / "equal_primary_bootstrap_verification.csv")

    contract_display = selected_contracts.copy()
    contract_display["weights"] = contract_display.apply(
        lambda row: "/".join(format_number(row[column], 4) for column in WEIGHT_COLUMNS),
        axis=1,
    )
    contract_display["selected_score_ci"] = contract_display.apply(
        lambda row: (
            f"{format_number(row['selected_bootstrap_mean'])} "
            f"[{format_number(row['selected_bootstrap_ci95_lower'])}, "
            f"{format_number(row['selected_bootstrap_ci95_upper'])}]"
        ),
        axis=1,
    )
    contract_display["difference_to_best_ci"] = contract_display.apply(
        lambda row: (
            f"{format_number(row['selected_difference_to_best_mean_from_details'])} "
            f"[{format_number(row['selected_difference_to_best_ci95_lower'])}, "
            f"{format_number(row['selected_difference_to_best_ci95_upper_from_details'])}]"
        ),
        axis=1,
    )
    contract_display = markdown_numeric_frame(
        contract_display,
        ["selected_parameter_count"],
        ["selected_within_one_se", "selected_practically_equivalent", "same_as_formal_family"],
    )

    selected_component_display = markdown_numeric_frame(
        selected_component_stats[[
            "family_role",
            "family_key",
            "component_label",
            "mean",
            "ci95_lower",
            "ci95_upper",
        ]],
        ["mean", "ci95_lower", "ci95_upper"],
        [],
    )

    component_mean_display = component_mean_wide.copy()
    component_mean_display = markdown_numeric_frame(
        component_mean_display,
        ["Free mechanism parameters", *PRIMARY_COMPONENTS],
        ["performance_pareto_front", "complexity_pareto_front"],
    )
    support_display = markdown_numeric_frame(
        support_stats[[
            "family_key",
            "Model family",
            "Free mechanism parameters",
            "mean_supported_drift_cells",
            "ci95_lower",
            "ci95_upper",
            "minimum",
            "maximum",
        ]],
        [
            "Free mechanism parameters",
            "mean_supported_drift_cells",
            "ci95_lower",
            "ci95_upper",
            "minimum",
            "maximum",
        ],
        [],
    )

    pareto_selected_row = pareto_mean.loc[pareto_mean["family_key"].astype(str) == formal_family].iloc[0]
    pareto_overview = pd.DataFrame(
        [
            {"Quantity": "Formal selected family", "Value": formal_family},
            {"Quantity": "Substantive-parameter count", "Value": formal_parameter_count},
            {"Quantity": "Performance Pareto front", "Value": format_bool(pareto_summary.get("selected_on_performance_pareto_front"))},
            {"Quantity": "Complexity Pareto front", "Value": format_bool(pareto_summary.get("selected_on_complexity_pareto_front"))},
            {"Quantity": "Mean-level performance dominators", "Value": pareto_summary.get("selected_performance_dominators") or "None"},
            {"Quantity": "Mean-level complexity dominators", "Value": pareto_summary.get("selected_complexity_dominators") or "None"},
            {"Quantity": "Bootstrap frequency: any not-more-complex dominator", "Value": format_number(computed_not_more_frequency)},
            {"Quantity": "Bootstrap frequency: any strictly simpler dominator", "Value": format_number(computed_simpler_frequency)},
            {"Quantity": "Pareto tolerance", "Value": format_number(pareto_summary.get("pareto_tolerance"), 12)},
        ]
    )

    equal_overview = pd.DataFrame(
        [
            {"Quantity": "Initial best family", "Value": equal_contract_row.get("initial_best_family")},
            {"Quantity": "Initial parsimony-selected family", "Value": equal_contract_row.get("initial_selected_family")},
            {"Quantity": "Final best family", "Value": equal_contract_row.get("final_best_family")},
            {"Quantity": "Final scalar-minimal family", "Value": equal_contract_row.get("final_selected_family")},
            {"Quantity": "Final substantive-parameter count", "Value": int(equal_contract_row.get("final_parameter_count"))},
            {"Quantity": "One-SE threshold", "Value": format_number(equal_contract_row.get("one_se_threshold"))},
            {"Quantity": "Scalar minimality confirmed", "Value": format_bool(equal_contract_row.get("scalar_minimality_confirmed"))},
            {"Quantity": "Search adequacy confirmed", "Value": format_bool(equal_contract_row.get("search_adequacy_confirmed"))},
            {"Quantity": "Persistence baseline excluded", "Value": format_bool(equal_contract_row.get("persistence_baseline_excluded"))},
            {"Quantity": "Final family within one SE", "Value": format_bool(equal_contract_row.get("final_family_within_one_se"))},
            {"Quantity": "Final family practically equivalent", "Value": format_bool(equal_contract_row.get("final_family_practically_equivalent"))},
            {"Quantity": "All selection gates passed", "Value": format_bool(equal_contract_row.get("would_pass_selection_gates_under_contract"))},
            {"Quantity": "Current Phase-2 family contract satisfied", "Value": format_bool(equal_contract_row.get("would_satisfy_current_phase2_family_contract"))},
            {"Quantity": "Boundary blockers", "Value": len(boundary_blockers)},
            {"Quantity": "Next required tests", "Value": len(next_required_tests)},
        ]
    )

    equal_family_display_columns = [
        "family_key",
        "Model family",
        "Free mechanism parameters",
        "Bootstrap mean primary score",
        "Bootstrap 95% CI lower",
        "Bootstrap 95% CI upper",
        "difference_to_best_mean",
        "difference_to_best_ci95_lower",
        "difference_to_best_ci95_upper",
        "Within one standard error of best",
        "Practically equivalent to best",
        "Final scalar-minimal family",
    ]
    equal_family_display_columns = [column for column in equal_family_display_columns if column in equal_family_results.columns]
    equal_family_display = markdown_numeric_frame(
        equal_family_results[equal_family_display_columns],
        [
            "Free mechanism parameters",
            "Bootstrap mean primary score",
            "Bootstrap 95% CI lower",
            "Bootstrap 95% CI upper",
            "difference_to_best_mean",
            "difference_to_best_ci95_lower",
            "difference_to_best_ci95_upper",
        ],
        [
            "Within one standard error of best",
            "Practically equivalent to best",
            "Final scalar-minimal family",
        ],
    )

    selected_metrics_display = markdown_numeric_frame(
        selected_metrics[["analysis", "family_key", "metric", "value"]],
        ["value"],
        [],
    )

    parameter_display = markdown_numeric_frame(
        parameter_table[[
            "parameter",
            "equal_primary_parameter_role",
            "formal_value",
            "equal_primary_value",
            "equal_minus_formal",
        ]],
        ["formal_value", "equal_primary_value", "equal_minus_formal"],
        [],
    )

    deletion_display_columns = [
        "round",
        "current_family",
        "tested_removed_parameter",
        "deletion_family",
        "bootstrap_mean_primary_score",
        "difference_to_best_mean",
        "difference_to_best_ci95_lower",
        "difference_to_best_ci95_upper",
        "within_one_standard_error_of_best",
        "practically_equivalent_to_best",
        "globally_eligible_under_selection_rule",
        "conclusion",
    ]
    deletion_display_columns = [column for column in deletion_display_columns if column in equal_deletions.columns]
    deletion_display = markdown_numeric_frame(
        equal_deletions[deletion_display_columns],
        [
            "round",
            "bootstrap_mean_primary_score",
            "difference_to_best_mean",
            "difference_to_best_ci95_lower",
            "difference_to_best_ci95_upper",
        ],
        [
            "within_one_standard_error_of_best",
            "practically_equivalent_to_best",
            "globally_eligible_under_selection_rule",
        ],
    )

    boundary_display_columns = [
        column
        for column in (
            "family_key",
            "parameter",
            "value",
            "boundary",
            "blocking_for_freeze",
            "resolution",
            "next_test_required",
        )
        if column in equal_boundaries.columns
    ]
    boundary_display = markdown_numeric_frame(
        equal_boundaries[boundary_display_columns] if boundary_display_columns else pd.DataFrame(),
        ["value"],
        ["blocking_for_freeze", "next_test_required"],
    )

    flags_frame = pd.DataFrame(
        [{"Diagnostic flag": key, "Value": format_bool(value)} for key, value in report_flags.items()]
    )

    report_lines: List[str] = []
    report_lines.append("# Mechanism score-contract and complexity-Pareto robustness")
    report_lines.append("")
    report_lines.append("## Analysis scope and integrity")
    report_lines.append("")
    scope_frame = pd.DataFrame(
        [
            {"Quantity": "Formal selected family", "Value": formal_family},
            {"Quantity": "Formal substantive-parameter count", "Value": formal_parameter_count},
            {"Quantity": "Development splits", "Value": "A_train and A_val"},
            {"Quantity": "Confirmation split read", "Value": "No"},
            {"Quantity": "Analysis status", "Value": "Post hoc, sensitivity only"},
            {"Quantity": "Formal outputs modified", "Value": "No"},
            {"Quantity": "Eligible for Phase-2 freeze", "Value": "No"},
            {"Quantity": "Paired-user bootstrap replicates", "Value": expected_reps},
            {"Quantity": "Decision bootstrap seed", "Value": formal_manifest.get("decision_bootstrap_seed")},
            {"Quantity": "Practical-equivalence margin", "Value": format_number(formal_manifest.get("practical_equivalence_margin"))},
            {"Quantity": "Configuration audit", "Value": "Passed"},
        ]
    )
    report_lines.append(markdown_table(scope_frame, ["Quantity", "Value"]))
    report_lines.append("")
    report_lines.append("All integrity checks passed. The extraction read only the completed development-only robustness outputs and the formal Phase-1 family table.")
    report_lines.append("")
    nonduplication = dict(combined_manifest.get("nonduplication", {}))
    nonduplication_frame = pd.DataFrame(
        [
            {"Quantity": "Equivalence-margin sensitivity repeated", "Value": format_bool(nonduplication.get("existing_equivalence_margin_sensitivity_repeated", False))},
            {"Quantity": "Formal-weight deletion result repeated", "Value": format_bool(nonduplication.get("formal_weight_parameter_deletion_result_repeated", False))},
            {"Quantity": "Equal-primary deletion rerun as part of new contract", "Value": format_bool(nonduplication.get("equal_primary_deletion_procedure_rerun_as_required_part_of_new_contract", False))},
            {"Quantity": "Formal deletion experiment replaced", "Value": format_bool(nonduplication.get("existing_parameter_deletion_experiment_replaced", False))},
            {"Quantity": "New analysis question", "Value": nonduplication.get("new_question")},
        ]
    )
    report_lines.append("### Non-duplication audit")
    report_lines.append("")
    report_lines.append(markdown_table(nonduplication_frame, ["Quantity", "Value"]))
    report_lines.append("")
    report_lines.append("## Formal score reconstruction")
    report_lines.append("")
    reconstruction_frame = pd.DataFrame(
        [
            {"Quantity": "Rows", "Value": reconstruction_summary["rows"]},
            {"Quantity": "Families", "Value": reconstruction_summary["families"]},
            {"Quantity": "Bootstrap replicates", "Value": reconstruction_summary["bootstrap_reps"]},
            {"Quantity": "Maximum absolute difference", "Value": format_number(reconstruction_summary["maximum_absolute_difference"], 12)},
            {"Quantity": "Mean absolute difference", "Value": format_number(reconstruction_summary["mean_absolute_difference"], 12)},
            {"Quantity": "99th percentile absolute difference", "Value": format_number(reconstruction_summary["p99_absolute_difference"], 12)},
            {"Quantity": "Acceptance tolerance", "Value": format_number(reconstruction_summary["tolerance"], 12)},
            {"Quantity": "Passed", "Value": format_bool(reconstruction_summary["passed"])},
        ]
    )
    report_lines.append(markdown_table(reconstruction_frame, ["Quantity", "Value"]))
    report_lines.append("")
    report_lines.append("## Frozen-fit score-contract replay")
    report_lines.append("")
    report_lines.append("Weight order: one-step closure / landscape JS / local drift / drift direction / drift magnitude.")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            contract_display,
            [
                "contract_label",
                "weights",
                "best_family_key",
                "selected_family_key",
                "selected_parameter_count",
                "selected_score_ci",
                "difference_to_best_ci",
                "selected_within_one_se",
                "selected_practically_equivalent",
                "same_as_formal_family",
            ],
            [
                "Contract",
                "Weights",
                "Best family",
                "Selected family",
                "k",
                "Selected score [95%]",
                "Difference to best [95%]",
                "Within 1-SE",
                "PE",
                "Formal family retained",
            ],
        )
    )
    report_lines.append("")
    report_lines.append("## Paired-bootstrap component losses")
    report_lines.append("")
    report_lines.append("### Formal selected and formal best families")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            selected_component_display,
            ["family_role", "family_key", "component_label", "mean", "ci95_lower", "ci95_upper"],
            ["Role", "Family", "Component", "Mean", "2.5%", "97.5%"],
        )
    )
    report_lines.append("")
    report_lines.append("### Supported drift cells across paired bootstrap replicates")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            support_display,
            [
                "family_key",
                "Model family",
                "Free mechanism parameters",
                "mean_supported_drift_cells",
                "ci95_lower",
                "ci95_upper",
                "minimum",
                "maximum",
            ],
            ["Family", "Model family", "k", "Mean", "2.5%", "97.5%", "Min", "Max"],
        )
    )
    report_lines.append("")
    report_lines.append("### Family mean components and Pareto status")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            component_mean_display,
            [
                "family_key",
                "Free mechanism parameters",
                *PRIMARY_COMPONENTS,
                "performance_pareto_front",
                "complexity_pareto_front",
            ],
            [
                "Family",
                "k",
                "Step",
                "JS",
                "Local",
                "Direction",
                "Magnitude",
                "Performance front",
                "Complexity front",
            ],
        )
    )
    report_lines.append("")
    report_lines.append("## Complexity-Pareto audit")
    report_lines.append("")
    report_lines.append(markdown_table(pareto_overview, ["Quantity", "Value"]))
    report_lines.append("")
    if dominator_frequency.empty:
        report_lines.append("No family appeared as a bootstrap dominator of the formal selected family.")
    else:
        dominator_display = markdown_numeric_frame(
            dominator_frequency,
            ["bootstrap_count", "bootstrap_reps", "bootstrap_frequency"],
            [],
        )
        report_lines.append(
            markdown_table(
                dominator_display,
                ["dominator_type", "family_key", "bootstrap_count", "bootstrap_reps", "bootstrap_frequency"],
                ["Dominator type", "Family", "Count", "Replicates", "Frequency"],
            )
        )
    report_lines.append("")
    report_lines.append("## Full equal-primary-component re-optimisation")
    report_lines.append("")
    report_lines.append(markdown_table(equal_overview, ["Quantity", "Value"]))
    report_lines.append("")
    report_lines.append("### Family comparison")
    report_lines.append("")
    report_lines.append(markdown_table(equal_family_display, equal_family_display_columns))
    report_lines.append("")
    report_lines.append("### Selected-family structural metrics")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            selected_metrics_display,
            ["analysis", "family_key", "metric", "value"],
            ["Analysis", "Family", "Metric", "Value"],
        )
    )
    report_lines.append("")
    report_lines.append("### Selected-parameter comparison")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            parameter_display,
            ["parameter", "equal_primary_parameter_role", "formal_value", "equal_primary_value", "equal_minus_formal"],
            ["Parameter", "Equal-primary role", "Formal", "Equal-primary", "Difference"],
        )
    )
    report_lines.append("")
    report_lines.append("## Scalar-deletion and search-adequacy audit")
    report_lines.append("")
    report_lines.append("### Direct deletion tests")
    report_lines.append("")
    report_lines.append(markdown_table(deletion_display, deletion_display_columns))
    report_lines.append("")
    report_lines.append("### Parameter-boundary audit")
    report_lines.append("")
    report_lines.append(markdown_table(boundary_display, boundary_display_columns) if boundary_display_columns else "No boundary rows.")
    report_lines.append("")
    if next_required_tests:
        report_lines.append("### Next required tests recorded by the equal-primary run")
        report_lines.append("")
        next_frame = pd.DataFrame(next_required_tests)
        report_lines.append(markdown_table(next_frame, list(next_frame.columns)))
        report_lines.append("")
    report_lines.append("## Integrated diagnostic flags")
    report_lines.append("")
    report_lines.append(markdown_table(flags_frame, ["Diagnostic flag", "Value"]))
    report_lines.append("")
    report_lines.append("## Computational and data contract")
    report_lines.append("")
    contract_frame = pd.DataFrame(
        [
            {"Quantity": "A_train valid rows", "Value": development_manifest.get("train_rows_valid")},
            {"Quantity": "A_train valid users", "Value": development_manifest.get("train_users_valid")},
            {"Quantity": "A_val valid rows", "Value": development_manifest.get("val_rows_valid")},
            {"Quantity": "A_val valid users", "Value": development_manifest.get("val_users_valid")},
            {"Quantity": "Grid profile", "Value": formal_runtime.get("grid_profile")},
            {"Quantity": "Bootstrap engine", "Value": formal_runtime.get("bootstrap_engine")},
            {"Quantity": "Search margin sensitivity", "Value": str(equal_manifest.get("margin_sensitivity_values"))},
            {"Quantity": "Formal script SHA-256", "Value": formal_manifest.get("formal_script_sha256")},
            {"Quantity": "Equal-run configuration difference", "Value": configuration_audit.get("intended_difference")},
            {"Quantity": "Bootstrap verification available", "Value": format_bool(verification_summary["available"])},
            {"Quantity": "Bootstrap verification maximum difference", "Value": format_number(verification_summary.get("maximum_absolute_difference"), 12)},
        ]
    )
    report_lines.append(markdown_table(contract_frame, ["Quantity", "Value"]))
    report_lines.append("")

    report_path = output_root / "mechanism_score_contract_numeric_report.md"
    report_json_path = output_root / "mechanism_score_contract_numeric_report.json"
    report_payload = {
        "created_at": now_string(),
        "analysis_contract": analysis_contract,
        "development_data": development_manifest,
        "integrity_checks": integrity_frame.to_dict(orient="records"),
        "input_immutability_certification": {
            "formal_audit": formal_input_evidence,
            "equal_primary_rerun": equal_input_evidence,
        },
        "formal_score_reconstruction": reconstruction_summary,
        "frozen_fit_score_contract_selection": selected_contracts.to_dict(orient="records"),
        "formal_component_bootstrap_summary": component_stats.to_dict(orient="records"),
        "formal_supported_drift_cells_summary": support_stats.to_dict(orient="records"),
        "formal_family_parameters_and_metrics": formal_results.to_dict(orient="records"),
        "pareto": {
            "summary": pareto_summary,
            "performance_front_families": pareto_mean.loc[
                pareto_mean["performance_pareto_front"], "family_key"
            ].astype(str).tolist(),
            "complexity_front_families": pareto_mean.loc[
                pareto_mean["complexity_pareto_front"], "family_key"
            ].astype(str).tolist(),
            "dominator_frequencies": dominator_frequency.to_dict(orient="records"),
        },
        "equal_primary_reoptimisation": {
            "summary": equal_summary,
            "contract_summary": equal_contract_summary.to_dict(orient="records"),
            "bootstrap_summary": equal_bootstrap_stats.to_dict(orient="records"),
            "bootstrap_summary_table_consistency": equal_bootstrap_differences,
            "selected_structural_metrics": selected_metrics.to_dict(orient="records"),
            "selected_parameter_comparison": parameter_table.to_dict(orient="records"),
            "deletion_round_summary": deletion_rounds.to_dict(orient="records"),
            "boundary_blockers": boundary_blockers,
            "next_required_tests": next_required_tests,
            "bootstrap_verification": verification_summary,
        },
        "configuration_audit": configuration_audit,
        "nonduplication": combined_manifest.get("nonduplication", {}),
        "diagnostic_flags": report_flags,
        "source_checksums": before,
        "interpretation_boundary": (
            "Frozen-fit stress contracts reuse parameters selected under the formal objective. "
            "Only the equal-primary-component analysis performs complete family and parameter re-optimisation. "
            "All analyses are post hoc, development-only and ineligible for Phase-2 freezing."
        ),
    }
    write_text("\n".join(report_lines).rstrip() + "\n", report_path)
    save_json(report_payload, report_json_path)

    output_paths = {
        "markdown_report": report_path,
        "json_report": report_json_path,
        "integrity_checks": table_root / "report_integrity_checks.csv",
        "score_contract_selection": table_root / "score_contract_selection.csv",
        "score_contract_family_details": table_root / "score_contract_family_details.csv.gz",
        "formal_component_bootstrap_summary": table_root / "formal_component_bootstrap_summary.csv",
        "formal_supported_drift_cells_summary": table_root / "formal_supported_drift_cells_summary.csv",
        "formal_family_parameters_and_metrics": table_root / "formal_family_parameters_and_metrics.csv",
        "formal_family_component_means_and_pareto": table_root / "formal_family_component_means_and_pareto.csv",
        "pareto_family_summary": table_root / "pareto_family_summary.csv",
        "pareto_dominator_frequencies": table_root / "pareto_dominator_frequencies.csv",
        "equal_primary_family_results": table_root / "equal_primary_family_results.csv",
        "equal_primary_bootstrap_summary": table_root / "equal_primary_bootstrap_summary.csv",
        "selected_family_structural_metrics": table_root / "selected_family_structural_metrics.csv",
        "selected_parameter_comparison": table_root / "selected_parameter_comparison.csv",
        "equal_primary_deletion_results": table_root / "equal_primary_deletion_results.csv",
        "equal_primary_final_deletion_audit": table_root / "equal_primary_final_deletion_audit.csv",
        "equal_primary_deletion_round_summary": table_root / "equal_primary_deletion_round_summary.csv",
        "equal_primary_boundary_audit": table_root / "equal_primary_boundary_audit.csv",
        "equal_primary_next_required_tests": table_root / "equal_primary_next_required_tests.csv",
    }
    if not verification.empty:
        output_paths["equal_primary_bootstrap_verification"] = table_root / "equal_primary_bootstrap_verification.csv"

    after = snapshot_files(paths)
    assert_snapshots_equal(before, after)
    output_inventory = {
        name: {
            "path": str(path.resolve()),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for name, path in output_paths.items()
    }
    manifest = {
        "created_at": now_string(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "score_root": str(score_root),
        "formal_audit_root": str(formal_audit_root),
        "equal_primary_rerun_root": str(equal_root),
        "formal_phase1_root": str(formal_root),
        "output_root": str(output_root),
        "B_confirm_read": False,
        "formal_outputs_modified": False,
        "source_files_before": before,
        "source_files_after": after,
        "source_files_unchanged": True,
        "input_immutability_certification": {
            "formal_audit": formal_input_evidence,
            "equal_primary_rerun": equal_input_evidence,
        },
        "input_manifest_checksums": {
            "combined": combined_manifest_checksum,
            "formal_audit": formal_manifest_checksum,
            "equal_rerun": equal_manifest_checksum,
        },
        "report_outputs": output_inventory,
        "diagnostic_flags": report_flags,
        "elapsed_seconds": float(time.time() - started),
    }
    manifest_path = metadata_root / "mechanism_score_contract_numeric_report_manifest.json"
    save_json(manifest, manifest_path)
    save_json(
        {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
        },
        metadata_root / "mechanism_score_contract_numeric_report_manifest.sha256.json",
    )

    print(f"Numerical report: {report_path}")
    print(f"Machine-readable report: {report_json_path}")
    print(f"Report manifest: {manifest_path}")


if __name__ == "__main__":
    main()
