#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-12
DEFAULT_SEEDS = (42, 2026, 666, 606, 37, 4669)
PRIMARY_SPLIT = "B_confirm"


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


def save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def parse_seeds(text: str) -> Tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique integer values.")
    return seeds


def label_for_seed(seed: int) -> str:
    return f"event_ssl_seed{int(seed)}"


def seed_from_label(label: str) -> Optional[int]:
    match = re.search(r"seed[_-]?(\d+)$", str(label))
    return int(match.group(1)) if match else None


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(valid)) < 3:
        return float("nan")
    aa = a[valid] - float(np.mean(a[valid]))
    bb = b[valid] - float(np.mean(b[valid]))
    denominator = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denominator <= EPS:
        return float("nan")
    return float(np.clip(np.dot(aa, bb) / denominator, -1.0, 1.0))


def vectorize(u: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(u) & np.isfinite(v)
    return np.column_stack([np.asarray(u)[valid], np.asarray(v)[valid]]).ravel()


def vector_correlation(
    first_u: np.ndarray,
    first_v: np.ndarray,
    second_u: np.ndarray,
    second_v: np.ndarray,
    mask: np.ndarray,
) -> float:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(first_u)
        & np.isfinite(first_v)
        & np.isfinite(second_u)
        & np.isfinite(second_v)
    )
    return pearson(vectorize(first_u, first_v, valid), vectorize(second_u, second_v, valid))


def speed_correlation(
    first_u: np.ndarray,
    first_v: np.ndarray,
    second_u: np.ndarray,
    second_v: np.ndarray,
    mask: np.ndarray,
) -> float:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(first_u)
        & np.isfinite(first_v)
        & np.isfinite(second_u)
        & np.isfinite(second_v)
    )
    return pearson(
        np.hypot(np.asarray(first_u)[valid], np.asarray(first_v)[valid]),
        np.hypot(np.asarray(second_u)[valid], np.asarray(second_v)[valid]),
    )


def weighted_local_cosine(
    first_u: np.ndarray,
    first_v: np.ndarray,
    second_u: np.ndarray,
    second_v: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float]:
    first_u = np.asarray(first_u, dtype=np.float64)
    first_v = np.asarray(first_v, dtype=np.float64)
    second_u = np.asarray(second_u, dtype=np.float64)
    second_v = np.asarray(second_v, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(first_u)
        & np.isfinite(first_v)
        & np.isfinite(second_u)
        & np.isfinite(second_v)
        & np.isfinite(weight)
        & (weight >= 0.0)
    )
    first_speed = np.hypot(first_u, first_v)
    second_speed = np.hypot(second_u, second_v)
    valid &= (first_speed > EPS) & (second_speed > EPS)
    total_weight = float(np.sum(np.where(np.asarray(mask, dtype=bool), np.maximum(weight, 0.0), 0.0)))
    valid_weight = float(np.sum(weight[valid])) if np.any(valid) else 0.0
    coverage = valid_weight / max(total_weight, EPS)
    if not np.any(valid):
        return float("nan"), float(coverage)
    cosine = (
        first_u[valid] * second_u[valid] + first_v[valid] * second_v[valid]
    ) / (first_speed[valid] * second_speed[valid])
    normalized = weight[valid] / max(float(np.sum(weight[valid])), EPS)
    return float(np.sum(normalized * np.clip(cosine, -1.0, 1.0))), float(coverage)


def residualize_on_target(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64).ravel()
    x = np.asarray(target, dtype=np.float64).ravel()
    valid = np.isfinite(y) & np.isfinite(x)
    if int(np.sum(valid)) < 3:
        return np.full_like(y, np.nan)
    design = np.column_stack([np.ones(int(np.sum(valid))), x[valid]])
    coefficients, _, _, _ = np.linalg.lstsq(design, y[valid], rcond=None)
    residual = np.full_like(y, np.nan)
    residual[valid] = y[valid] - design @ coefficients
    return residual


def partial_correlation_from_pairwise(r_ms: float, r_me: float, r_se: float) -> float:
    denominator = math.sqrt(max((1.0 - r_me * r_me) * (1.0 - r_se * r_se), 0.0))
    if denominator <= EPS:
        return float("nan")
    return float(np.clip((r_ms - r_me * r_se) / denominator, -1.0, 1.0))


def common_target_metrics(
    empirical_u: np.ndarray,
    empirical_v: np.ndarray,
    mechanism_u: np.ndarray,
    mechanism_v: np.ndarray,
    event_u: np.ndarray,
    event_v: np.ndarray,
    null_u: np.ndarray,
    null_v: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(empirical_u)
        & np.isfinite(empirical_v)
        & np.isfinite(mechanism_u)
        & np.isfinite(mechanism_v)
        & np.isfinite(event_u)
        & np.isfinite(event_v)
        & np.isfinite(null_u)
        & np.isfinite(null_v)
        & np.isfinite(weight)
    )
    if int(np.sum(valid)) < 3:
        raise RuntimeError("Too few cells for common-target-conditioned metrics.")
    empirical = vectorize(empirical_u, empirical_v, valid)
    mechanism = vectorize(mechanism_u, mechanism_v, valid)
    event = vectorize(event_u, event_v, valid)
    r_me = pearson(mechanism, empirical)
    r_se = pearson(event, empirical)
    r_ms = pearson(mechanism, event)
    target_benchmark = r_me * r_se
    partial_formula = partial_correlation_from_pairwise(r_ms, r_me, r_se)
    mechanism_linear_residual = residualize_on_target(mechanism, empirical)
    event_linear_residual = residualize_on_target(event, empirical)
    partial_direct = pearson(mechanism_linear_residual, event_linear_residual)

    mechanism_error_u = np.asarray(mechanism_u) - np.asarray(empirical_u)
    mechanism_error_v = np.asarray(mechanism_v) - np.asarray(empirical_v)
    event_error_u = np.asarray(event_u) - np.asarray(empirical_u)
    event_error_v = np.asarray(event_v) - np.asarray(empirical_v)
    direct_error_cosine, direct_error_coverage = weighted_local_cosine(
        mechanism_error_u,
        mechanism_error_v,
        event_error_u,
        event_error_v,
        weight,
        valid,
    )

    mechanism_correction_u = np.asarray(mechanism_u) - np.asarray(null_u)
    mechanism_correction_v = np.asarray(mechanism_v) - np.asarray(null_v)
    event_correction_u = np.asarray(event_u) - np.asarray(null_u)
    event_correction_v = np.asarray(event_v) - np.asarray(null_v)
    shared_null_cosine, shared_null_coverage = weighted_local_cosine(
        mechanism_correction_u,
        mechanism_correction_v,
        event_correction_u,
        event_correction_v,
        weight,
        valid,
    )
    return {
        "supported_cells": int(np.sum(valid)),
        "supported_occupancy_mass": float(np.sum(np.asarray(weight)[valid])),
        "mechanism_vs_empirical_vector_corr": r_me,
        "event_ssl_vs_empirical_vector_corr": r_se,
        "mechanism_vs_event_ssl_raw_vector_corr": r_ms,
        "linear_common_target_product_benchmark": float(target_benchmark),
        "raw_minus_target_product": float(r_ms - target_benchmark),
        "linear_partial_corr_given_empirical": partial_formula,
        "direct_ols_residual_corr_given_empirical": partial_direct,
        "partial_formula_direct_difference": float(partial_formula - partial_direct),
        "direct_empirical_error_vector_corr": vector_correlation(
            mechanism_error_u,
            mechanism_error_v,
            event_error_u,
            event_error_v,
            valid,
        ),
        "direct_empirical_error_speed_corr": speed_correlation(
            mechanism_error_u,
            mechanism_error_v,
            event_error_u,
            event_error_v,
            valid,
        ),
        "direct_empirical_error_weighted_local_cosine": direct_error_cosine,
        "direct_empirical_error_local_cosine_coverage": direct_error_coverage,
        "shared_null_subtracted_correction_vector_corr_descriptive": vector_correlation(
            mechanism_correction_u,
            mechanism_correction_v,
            event_correction_u,
            event_correction_v,
            valid,
        ),
        "shared_null_subtracted_correction_weighted_local_cosine_descriptive": shared_null_cosine,
        "shared_null_subtracted_local_cosine_coverage": shared_null_coverage,
        "interpretation_boundary": (
            "The target-product and partial-correlation quantities are descriptive linear common-target calibrations, "
            "not exchangeability nulls or cell-level inferential tests. Direct errors use the estimated empirical field "
            "as reference and therefore remain conditional on that field estimate. Shared-null-subtracted correction "
            "agreement is descriptive because both models subtract the same scaffold."
        ),
    }


def t_summary(values: np.ndarray, confidence: float = 0.95) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    n = int(len(array))
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if n > 1 else float("nan")
    if n > 1 and np.isfinite(sd):
        critical = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
        half = critical * sd / math.sqrt(n)
        low, high = mean - half, mean + half
    else:
        low = high = float("nan")
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_low": float(low),
        "ci_high": float(high),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summarize_seed_frame(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    group_label: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for metric in metrics:
        if metric not in frame.columns:
            continue
        values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=np.float64)
        summary = t_summary(values)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "group": group_label,
                "metric": metric,
                **summary,
                "positive_count": int(np.sum(finite > 0)),
                "negative_count": int(np.sum(finite < 0)),
                "nonpositive_count": int(np.sum(finite <= 0)),
                "nonnegative_count": int(np.sum(finite >= 0)),
            }
        )
    return pd.DataFrame(rows)


def load_checkpoint_seed(checkpoint_path: Path) -> Optional[int]:
    try:
        import torch

        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        config = dict(payload.get("config", {}))
        return int(config["seed"]) if "seed" in config else None
    except Exception:
        return None


def validate_event_root_provenance(
    root: Path,
    expected_seed: int,
    expected_label: str,
) -> Dict[str, Any]:
    manifest_path = root / "metadata" / "stage4_event_ssl_evaluation_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = load_json(manifest_path)
    if manifest.get("model_kind") != "predictive_state":
        raise RuntimeError(f"{expected_label} is not a predictive_state evaluation.")
    if manifest.get("primary_coordinates") != ["M", "Psi"]:
        raise RuntimeError(f"{expected_label} does not use exactly M and Psi.")
    checkpoint = Path(str(manifest.get("checkpoint", "")))
    if not checkpoint.exists():
        candidates = [
            root.parent / "models" / "predictive_state" / "best_model.pt",
            root.parent / "model" / "best_model.pt",
            root.parent / "models" / "predictive_state" / checkpoint.name,
        ]
        resolved = next((candidate for candidate in candidates if candidate.exists()), None)
        if resolved is None:
            raise FileNotFoundError(
                f"Checkpoint recorded by {expected_label} does not exist and no relocated checkpoint was found: {checkpoint}"
            )
        checkpoint = resolved
    actual_seed = load_checkpoint_seed(checkpoint)
    if actual_seed is None:
        raise RuntimeError(f"Could not verify the checkpoint seed for {expected_label}: {checkpoint}")
    if actual_seed != int(expected_seed):
        raise RuntimeError(
            f"{expected_label} checkpoint seed is {actual_seed}, expected {expected_seed}."
        )
    return {
        "label": expected_label,
        "expected_seed": int(expected_seed),
        "checkpoint_seed": actual_seed,
        "evaluation_root": str(root.resolve()),
        "evaluation_manifest": str(manifest_path.resolve()),
        "evaluation_manifest_sha256": file_sha256(manifest_path),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "train_script_sha256": manifest.get("train_script_sha256"),
        "input_root": manifest.get("input_root"),
    }


def _bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"})


def load_headline_bootstrap(
    root: Path,
    point_metrics: Mapping[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    table_path = find_table(root / "tables" / "field_cluster_bootstrap_replicates")
    point_path = find_table(root / "tables" / "formal_point_estimates")
    formal_point = read_table(point_path)
    required_point_columns = {"domain", "comparison", "metric", "point_estimate"}
    missing_point_columns = sorted(required_point_columns.difference(formal_point.columns))
    if missing_point_columns:
        raise RuntimeError(f"Headline formal-point table is missing: {missing_point_columns}")

    comparison_to_key = {
        "mechanism_vs_empirical": "mechanism_vs_empirical_vector_corr",
        "event_ssl_anchor_vs_empirical": "event_ssl_vs_empirical_vector_corr",
        "mechanism_vs_event_ssl_anchor": "mechanism_vs_event_ssl_raw_vector_corr",
    }
    archived_base: Dict[str, float] = {}
    current_base: Dict[str, float] = {}
    point_differences: Dict[str, float] = {}
    signed_point_differences: Dict[str, float] = {}
    for comparison, key in comparison_to_key.items():
        row = formal_point[
            (formal_point["domain"].astype(str) == "field")
            & (formal_point["comparison"].astype(str) == comparison)
            & (formal_point["metric"].astype(str) == "drift_vector_corr")
        ]
        if len(row) != 1:
            raise RuntimeError(f"Headline formal point is not unique for {comparison}.")
        archived_value = float(row.iloc[0]["point_estimate"])
        current_value = float(point_metrics[key])
        archived_base[key] = archived_value
        current_base[key] = current_value
        signed_point_differences[comparison] = current_value - archived_value
        point_differences[comparison] = abs(current_value - archived_value)

    def derived(values: Mapping[str, float]) -> Dict[str, float]:
        r_me = float(values["mechanism_vs_empirical_vector_corr"])
        r_se = float(values["event_ssl_vs_empirical_vector_corr"])
        r_ms = float(values["mechanism_vs_event_ssl_raw_vector_corr"])
        target_product = r_me * r_se
        return {
            "linear_common_target_product_benchmark": float(target_product),
            "raw_minus_target_product": float(r_ms - target_product),
            "linear_partial_corr_given_empirical": partial_correlation_from_pairwise(
                r_ms, r_me, r_se
            ),
        }

    archived_derived = derived(archived_base)
    current_derived = derived(current_base)
    maximum_point_difference = float(max(point_differences.values()))
    exact_point_match = bool(maximum_point_difference <= 1e-8)

    frame = read_table(table_path)
    required = {
        "replicate",
        "comparison",
        "support_contract",
        "fixed_support_complete",
        "supported_cells",
        "drift_vector_corr",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Headline bootstrap table is missing: {missing}")
    frame = frame[
        (frame["support_contract"].astype(str) == "fixed_formal_support")
        & _bool_series(frame["fixed_support_complete"])
        & frame["comparison"].astype(str).isin(
            [
                "mechanism_vs_empirical",
                "event_ssl_anchor_vs_empirical",
                "mechanism_vs_event_ssl_anchor",
            ]
        )
    ].copy()
    if frame.empty:
        raise RuntimeError("No complete fixed-support learner-cluster bootstrap rows were found.")
    if frame.duplicated(["replicate", "comparison"], keep=False).any():
        raise RuntimeError(
            "Headline bootstrap has duplicate replicate/comparison rows on fixed support."
        )

    metric_pivot = frame.pivot(
        index="replicate", columns="comparison", values="drift_vector_corr"
    )
    cell_pivot = frame.pivot(
        index="replicate", columns="comparison", values="supported_cells"
    )
    expected_columns = [
        "mechanism_vs_empirical",
        "event_ssl_anchor_vs_empirical",
        "mechanism_vs_event_ssl_anchor",
    ]
    if any(column not in metric_pivot.columns for column in expected_columns):
        raise RuntimeError("Headline bootstrap lacks one or more common-target comparisons.")
    metric_pivot = metric_pivot.dropna(subset=expected_columns)
    cell_pivot = cell_pivot.loc[metric_pivot.index]
    if len(metric_pivot) < 10:
        raise RuntimeError("Too few complete fixed-support learner-cluster bootstrap replicates.")
    cell_values = cell_pivot[expected_columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(cell_values)) or not np.all(
        cell_values == cell_values[:, [0]]
    ):
        raise RuntimeError(
            "Pairwise bootstrap correlations were not evaluated on identical fixed supports."
        )

    output = pd.DataFrame(
        {
            "replicate": metric_pivot.index.to_numpy(dtype=int),
            "mechanism_vs_empirical_vector_corr": metric_pivot[
                "mechanism_vs_empirical"
            ].to_numpy(dtype=float),
            "event_ssl_vs_empirical_vector_corr": metric_pivot[
                "event_ssl_anchor_vs_empirical"
            ].to_numpy(dtype=float),
            "mechanism_vs_event_ssl_raw_vector_corr": metric_pivot[
                "mechanism_vs_event_ssl_anchor"
            ].to_numpy(dtype=float),
            "supported_cells": cell_pivot[expected_columns[0]].to_numpy(dtype=int),
        }
    )
    output["linear_common_target_product_benchmark"] = (
        output["mechanism_vs_empirical_vector_corr"]
        * output["event_ssl_vs_empirical_vector_corr"]
    )
    output["raw_minus_target_product"] = (
        output["mechanism_vs_event_ssl_raw_vector_corr"]
        - output["linear_common_target_product_benchmark"]
    )
    output["linear_partial_corr_given_empirical"] = [
        partial_correlation_from_pairwise(r_ms, r_me, r_se)
        for r_ms, r_me, r_se in zip(
            output["mechanism_vs_event_ssl_raw_vector_corr"],
            output["mechanism_vs_empirical_vector_corr"],
            output["event_ssl_vs_empirical_vector_corr"],
        )
    ]
    output["point_estimand"] = "headline_joined_cohort_fixed_formal_support"

    summary_rows: List[Dict[str, Any]] = []
    for metric in (
        "linear_common_target_product_benchmark",
        "raw_minus_target_product",
        "linear_partial_corr_given_empirical",
    ):
        values = pd.to_numeric(output[metric], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        point = float(archived_derived[metric])
        current_point = float(current_derived[metric])
        if len(values) < 10:
            raise RuntimeError(f"Too few finite bootstrap replicates for {metric}.")
        if metric == "linear_partial_corr_given_empirical":
            transformed = np.arctanh(np.clip(values, -1 + 1e-12, 1 - 1e-12))
            point_transformed = float(
                np.arctanh(np.clip(point, -1 + 1e-12, 1 - 1e-12))
            )
            standard_error = float(np.std(transformed, ddof=1))
            low = math.tanh(point_transformed - 1.96 * standard_error)
            high = math.tanh(point_transformed + 1.96 * standard_error)
        else:
            standard_error = float(np.std(values, ddof=1))
            low = point - 1.96 * standard_error
            high = point + 1.96 * standard_error
        summary_rows.append(
            {
                "metric": metric,
                "point_estimate": point,
                "point_estimate_source": "headline_bootstrap_formal_point_estimates",
                "current_null_audit_point_estimate": current_point,
                "current_minus_headline_point": current_point - point,
                "point_estimands_exactly_matched": exact_point_match,
                "replicates": int(len(values)),
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_sd": float(np.std(values, ddof=1)),
                "bootstrap_2p5": float(np.quantile(values, 0.025)),
                "bootstrap_97p5": float(np.quantile(values, 0.975)),
                "point_centered_95_ci_low": float(low),
                "point_centered_95_ci_high": float(high),
                "interval_applies_to": "headline_joined_cohort_fixed_formal_support_estimand",
                "interval_note": (
                    "Derived from the existing ordinary multinomial learner-cluster bootstrap "
                    "on its archived fixed-support headline estimand. No additional resampling "
                    "was performed. When the current null-audit point differs, this interval is "
                    "not recentered or relabelled as uncertainty for the current point."
                ),
            }
        )

    audit = {
        "available": True,
        "point_estimands_exactly_matched": exact_point_match,
        "reuse_mode": (
            "same_estimand" if exact_point_match else "separate_headline_fixed_support_estimand"
        ),
        "source_table": str(table_path.resolve()),
        "source_sha256": file_sha256(table_path),
        "formal_point_table": str(point_path.resolve()),
        "formal_point_table_sha256": file_sha256(point_path),
        "archived_base_points": archived_base,
        "current_null_audit_base_points": current_base,
        "signed_current_minus_headline": signed_point_differences,
        "formal_point_max_abs_difference": maximum_point_difference,
        "compatibility_tolerance": 1e-8,
        "replicates_used": int(len(output)),
        "support_contract": "fixed_formal_support",
        "uncertainty_applies_to": "headline_joined_cohort_fixed_formal_support_estimand",
        "current_null_audit_points_not_recentered": not exact_point_match,
        "new_resampling_performed": False,
        "interpretation": (
            "The learner-cluster bootstrap remains valid for its archived joined-cohort "
            "fixed-support headline estimand. A nonzero point difference indicates a distinct "
            "aggregation contract; its interval is retained only for the archived headline "
            "point and is not recentered onto the null-audit point."
        ),
    }
    return output, pd.DataFrame(summary_rows), audit

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--null-recovery-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/six_seed_null_referenced_recovery"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/six_seed_null_common_target_audit"),
    )
    parser.add_argument(
        "--headline-bootstrap-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/frozen_headline_learner_cluster_uncertainty"),
    )
    parser.add_argument("--require-headline-bootstrap", action="store_true")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--splits", nargs="+", default=["A_val", "B_confirm"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def run_self_test() -> None:
    rng = np.random.default_rng(83)
    target = rng.normal(size=5000)
    mechanism = target + rng.normal(scale=0.6, size=5000)
    event = target + rng.normal(scale=0.8, size=5000)
    r_me = pearson(mechanism, target)
    r_se = pearson(event, target)
    r_ms = pearson(mechanism, event)
    partial_formula = partial_correlation_from_pairwise(r_ms, r_me, r_se)
    partial_direct = pearson(
        residualize_on_target(mechanism, target), residualize_on_target(event, target)
    )
    if abs(partial_formula - partial_direct) > 1e-10:
        raise AssertionError("Partial-correlation formula and direct residualization differ.")
    if abs(r_ms - r_me * r_se) > 0.05:
        raise AssertionError("Synthetic independent-error common-target benchmark failed.")

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        table_root = root / "tables"
        table_root.mkdir(parents=True, exist_ok=True)
        archived = {
            "mechanism_vs_empirical": 0.80,
            "event_ssl_anchor_vs_empirical": 0.70,
            "mechanism_vs_event_ssl_anchor": 0.60,
        }
        point_rows = [
            {
                "domain": "field",
                "comparison": comparison,
                "metric": "drift_vector_corr",
                "point_estimate": value,
            }
            for comparison, value in archived.items()
        ]
        write_table(pd.DataFrame(point_rows), table_root / "formal_point_estimates")
        replicate_rows = []
        for replicate in range(20):
            for comparison, value in archived.items():
                replicate_rows.append(
                    {
                        "replicate": replicate,
                        "comparison": comparison,
                        "support_contract": "fixed_formal_support",
                        "fixed_support_complete": True,
                        "supported_cells": 120,
                        "drift_vector_corr": value + 0.001 * (replicate - 9.5),
                    }
                )
        write_table(
            pd.DataFrame(replicate_rows),
            table_root / "field_cluster_bootstrap_replicates",
        )
        current = {
            "mechanism_vs_empirical_vector_corr": 0.791,
            "event_ssl_vs_empirical_vector_corr": 0.705,
            "mechanism_vs_event_ssl_raw_vector_corr": 0.596,
        }
        current_product = current["mechanism_vs_empirical_vector_corr"] * current[
            "event_ssl_vs_empirical_vector_corr"
        ]
        current.update(
            {
                "linear_common_target_product_benchmark": current_product,
                "raw_minus_target_product": current[
                    "mechanism_vs_event_ssl_raw_vector_corr"
                ]
                - current_product,
                "linear_partial_corr_given_empirical": partial_correlation_from_pairwise(
                    current["mechanism_vs_event_ssl_raw_vector_corr"],
                    current["mechanism_vs_empirical_vector_corr"],
                    current["event_ssl_vs_empirical_vector_corr"],
                ),
            }
        )
        output, summary, audit = load_headline_bootstrap(root, current)
        archived_product = archived["mechanism_vs_empirical"] * archived[
            "event_ssl_anchor_vs_empirical"
        ]
        reported_product = float(
            summary.loc[
                summary["metric"] == "linear_common_target_product_benchmark",
                "point_estimate",
            ].iloc[0]
        )
        if len(output) != 20 or len(summary) != 3:
            raise AssertionError("Headline-bootstrap self-test produced incomplete outputs.")
        if audit.get("point_estimands_exactly_matched") is not False:
            raise AssertionError("Point-estimand mismatch was not recorded.")
        if audit.get("reuse_mode") != "separate_headline_fixed_support_estimand":
            raise AssertionError("Point-estimand mismatch reuse mode is incorrect.")
        if abs(reported_product - archived_product) > 1e-12:
            raise AssertionError("Bootstrap interval was incorrectly recentered to the null-audit point.")

    print("state-independent common-target audit self-test passed")


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return

    seeds = parse_seeds(args.seeds)
    expected_labels = [label_for_seed(seed) for seed in seeds]
    null_root = args.null_recovery_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    table_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    null_manifest_path = null_root / "metadata" / "null_referenced_recovery_manifest.json"
    if not null_manifest_path.exists():
        raise FileNotFoundError(null_manifest_path)
    null_manifest = load_json(null_manifest_path)
    if str(null_manifest.get("primary_event_ssl_label", "")) != label_for_seed(42):
        raise RuntimeError("The primary Event-SSL label is not event_ssl_seed42.")
    supplied_labels = sorted(str(label) for label in dict(null_manifest.get("event_ssl_roots", {})))
    if sorted(expected_labels) != supplied_labels:
        raise RuntimeError(
            f"Null-recovery manifest labels differ: expected {expected_labels}, found {supplied_labels}."
        )
    guardrails = dict(null_manifest.get("guardrails", {}))
    required_false = (
        "model_training",
        "model_selection",
        "mechanism_refit",
        "coordinate_refit",
        "grid_or_mask_refit",
        "construction_null_protocol_changed",
        "learned_plane_used_for_null_subtraction",
        "B_confirm_used_for_update",
    )
    violations = [name for name in required_false if bool(guardrails.get(name, False))]
    if violations:
        raise RuntimeError(f"Null-recovery guardrails failed: {violations}")

    event_roots = {
        str(label): Path(str(root)).resolve()
        for label, root in dict(null_manifest["event_ssl_roots"]).items()
    }
    provenance_rows = []
    for seed in seeds:
        label = label_for_seed(seed)
        provenance_rows.append(validate_event_root_provenance(event_roots[label], seed, label))
    provenance = pd.DataFrame(provenance_rows)
    if provenance["train_script_sha256"].dropna().nunique() > 1:
        raise RuntimeError("Event-SSL seeds use different training-script SHA-256 values.")

    point_path = find_table(null_root / "tables" / "null_referenced_recovery_metrics_all_splits")
    point = read_table(point_path)
    required_point_columns = {
        "split",
        "model",
        "exact_null_sse_to_empirical",
        "model_sse_to_empirical",
        "primary_delta_sse_model_minus_exact_null",
        "null_relative_field_skill",
        "M_null_relative_field_skill",
        "Psi_null_relative_field_skill",
        "model_minus_diagonal_rescaled_scaffold_sse",
    }
    missing_point = sorted(required_point_columns.difference(point.columns))
    if missing_point:
        raise RuntimeError(f"Null-recovery point table is missing: {missing_point}")

    seed_rows: List[Dict[str, Any]] = []
    common_rows: List[Dict[str, Any]] = []
    quality_rows: List[Dict[str, Any]] = []
    common_reference_masks: Dict[str, np.ndarray] = {}
    common_seed42_points: Dict[str, float] = {}

    for split in args.splits:
        split_point = point[(point["split"] == split) & point["model"].isin(expected_labels)].copy()
        if len(split_point) != len(seeds):
            raise RuntimeError(f"Expected {len(seeds)} Event-SSL rows for {split}, found {len(split_point)}.")
        array_path = null_root / "arrays" / f"{split}_null_referenced_fields.npz"
        if not array_path.exists():
            raise FileNotFoundError(array_path)
        arrays = np.load(array_path)
        required_arrays = {
            "empirical_u",
            "empirical_v",
            "exact_null_u",
            "exact_null_v",
            "minimal_mechanism_u",
            "minimal_mechanism_v",
            "drift_mask",
            "occupancy_probability",
        }
        missing_arrays = sorted(required_arrays.difference(arrays.files))
        if missing_arrays:
            raise RuntimeError(f"Field array archive is missing: {missing_arrays}")
        empirical_u = np.asarray(arrays["empirical_u"], dtype=np.float64)
        empirical_v = np.asarray(arrays["empirical_v"], dtype=np.float64)
        null_u = np.asarray(arrays["exact_null_u"], dtype=np.float64)
        null_v = np.asarray(arrays["exact_null_v"], dtype=np.float64)
        mechanism_u = np.asarray(arrays["minimal_mechanism_u"], dtype=np.float64)
        mechanism_v = np.asarray(arrays["minimal_mechanism_v"], dtype=np.float64)
        formal_mask = np.asarray(arrays["drift_mask"], dtype=bool)
        weight = np.asarray(arrays["occupancy_probability"], dtype=np.float64)

        for seed in seeds:
            label = label_for_seed(seed)
            if f"{label}_u" not in arrays.files or f"{label}_v" not in arrays.files:
                raise RuntimeError(f"Field arrays lack {label}.")
            event_u = np.asarray(arrays[f"{label}_u"], dtype=np.float64)
            event_v = np.asarray(arrays[f"{label}_v"], dtype=np.float64)
            valid = (
                formal_mask
                & np.isfinite(empirical_u)
                & np.isfinite(empirical_v)
                & np.isfinite(null_u)
                & np.isfinite(null_v)
                & np.isfinite(mechanism_u)
                & np.isfinite(mechanism_v)
                & np.isfinite(event_u)
                & np.isfinite(event_v)
                & np.isfinite(weight)
            )
            if split not in common_reference_masks:
                common_reference_masks[split] = valid.copy()
            elif not np.array_equal(common_reference_masks[split], valid):
                raise RuntimeError(
                    f"Common field support differs across Event-SSL seeds on {split}."
                )
            metrics = common_target_metrics(
                empirical_u,
                empirical_v,
                mechanism_u,
                mechanism_v,
                event_u,
                event_v,
                null_u,
                null_v,
                weight,
                valid,
            )
            if abs(float(metrics["partial_formula_direct_difference"])) > 1e-10:
                raise RuntimeError(
                    f"Partial-correlation formula gate failed for {label} on {split}."
                )
            common_rows.append({"split": split, "seed": seed, "model": label, **metrics})
            if split == PRIMARY_SPLIT and seed == 42:
                common_seed42_points = {
                    key: float(metrics[key])
                    for key in (
                        "mechanism_vs_empirical_vector_corr",
                        "event_ssl_vs_empirical_vector_corr",
                        "mechanism_vs_event_ssl_raw_vector_corr",
                        "linear_common_target_product_benchmark",
                        "raw_minus_target_product",
                        "linear_partial_corr_given_empirical",
                    )
                }

            row = split_point[split_point["model"] == label]
            if len(row) != 1:
                raise RuntimeError(f"Point row is not unique for {split}/{label}.")
            values = row.iloc[0].to_dict()
            seed_rows.append(
                {
                    "split": split,
                    "seed": seed,
                    "model": label,
                    "exact_null_distance": math.sqrt(max(float(values["exact_null_sse_to_empirical"]), 0.0)),
                    "event_ssl_distance": math.sqrt(max(float(values["model_sse_to_empirical"]), 0.0)),
                    **{
                        key: values.get(key)
                        for key in point.columns
                        if key not in {"split", "model"}
                    },
                }
            )

        quality_rows.append(
            {
                "gate": f"{split}_common_support_identical_across_seeds",
                "passed": True,
                "value": int(np.sum(common_reference_masks[split])),
            }
        )

    seed_frame = pd.DataFrame(seed_rows)
    common_frame = pd.DataFrame(common_rows)
    if set(seed_frame["seed"].unique()) != set(seeds):
        raise RuntimeError("Six-seed null-calibration table is incomplete.")

    null_metrics = [
        "event_ssl_distance",
        "primary_delta_sse_model_minus_exact_null",
        "null_relative_field_skill",
        "M_null_relative_field_skill",
        "Psi_null_relative_field_skill",
        "model_minus_diagonal_rescaled_scaffold_sse",
        "model_vs_empirical_vector_corr",
        "null_referenced_correction_vs_empirical_excess_vector_corr",
        "null_referenced_correction_vs_empirical_excess_weighted_local_cosine",
        "null_referenced_excess_amplitude_slope",
    ]
    common_metrics = [
        "mechanism_vs_empirical_vector_corr",
        "event_ssl_vs_empirical_vector_corr",
        "mechanism_vs_event_ssl_raw_vector_corr",
        "linear_common_target_product_benchmark",
        "raw_minus_target_product",
        "linear_partial_corr_given_empirical",
        "direct_empirical_error_vector_corr",
        "direct_empirical_error_speed_corr",
        "direct_empirical_error_weighted_local_cosine",
        "shared_null_subtracted_correction_vector_corr_descriptive",
        "shared_null_subtracted_correction_weighted_local_cosine_descriptive",
    ]
    summary_parts = []
    common_summary_parts = []
    for split in args.splits:
        summary_parts.append(
            summarize_seed_frame(
                seed_frame[seed_frame["split"] == split], null_metrics, split
            )
        )
        common_summary_parts.append(
            summarize_seed_frame(
                common_frame[common_frame["split"] == split], common_metrics, split
            )
        )
    seed_summary = pd.concat(summary_parts, ignore_index=True)
    common_summary = pd.concat(common_summary_parts, ignore_index=True)

    sign_rows = []
    for split in args.splits:
        group = seed_frame[seed_frame["split"] == split]
        sign_rows.append(
            {
                "split": split,
                "seeds": int(len(group)),
                "delta_sse_below_zero": int(
                    np.sum(pd.to_numeric(group["primary_delta_sse_model_minus_exact_null"], errors="coerce") < 0)
                ),
                "overall_skill_above_zero": int(
                    np.sum(pd.to_numeric(group["null_relative_field_skill"], errors="coerce") > 0)
                ),
                "M_skill_above_zero": int(
                    np.sum(pd.to_numeric(group["M_null_relative_field_skill"], errors="coerce") > 0)
                ),
                "Psi_skill_above_zero": int(
                    np.sum(pd.to_numeric(group["Psi_null_relative_field_skill"], errors="coerce") > 0)
                ),
                "better_than_diagonal_rescaled_scaffold": int(
                    np.sum(pd.to_numeric(group["model_minus_diagonal_rescaled_scaffold_sse"], errors="coerce") < 0)
                ),
            }
        )
    sign_frame = pd.DataFrame(sign_rows)

    headline_bootstrap_replicates = pd.DataFrame()
    headline_bootstrap_summary = pd.DataFrame()
    headline_audit: Dict[str, Any] = {"available": False}
    headline_root = args.headline_bootstrap_root.resolve()
    try:
        headline_bootstrap_replicates, headline_bootstrap_summary, headline_audit = load_headline_bootstrap(
            headline_root, common_seed42_points
        )
    except FileNotFoundError:
        if args.require_headline_bootstrap:
            raise
        headline_audit = {
            "available": False,
            "reason": f"No existing learner-cluster bootstrap at {headline_root}",
        }

    paths = {
        "event_ssl_seed_null_metrics": write_table(
            seed_frame, table_root / "event_ssl_seed_null_calibration_metrics"
        ),
        "event_ssl_seed_null_summary": write_table(
            seed_summary, table_root / "event_ssl_seed_null_calibration_summary"
        ),
        "event_ssl_seed_sign_counts": write_table(
            sign_frame, table_root / "event_ssl_seed_null_calibration_sign_counts"
        ),
        "common_target_metrics": write_table(
            common_frame, table_root / "common_target_conditioned_metrics_by_seed"
        ),
        "common_target_summary": write_table(
            common_summary, table_root / "common_target_conditioned_seed_summary"
        ),
        "event_root_provenance": write_table(
            provenance, table_root / "event_ssl_seed_provenance"
        ),
    }
    if not headline_bootstrap_replicates.empty:
        paths["seed42_common_target_bootstrap_replicates"] = write_table(
            headline_bootstrap_replicates,
            table_root / "seed42_common_target_learner_bootstrap_replicates",
        )
        paths["seed42_common_target_bootstrap_summary"] = write_table(
            headline_bootstrap_summary,
            table_root / "seed42_common_target_learner_bootstrap_summary",
        )

    known_seed42 = seed_frame[
        (seed_frame["split"] == PRIMARY_SPLIT) & (seed_frame["seed"] == 42)
    ].iloc[0]
    known_checks = {
        "seed42_event_ssl_distance_close_to_0p2149": abs(float(known_seed42["event_ssl_distance"]) - 0.2149) <= 5e-4,
        "seed42_exact_null_distance_close_to_0p0907": abs(float(known_seed42["exact_null_distance"]) - 0.0907) <= 5e-4,
    }
    quality_rows.extend(
        {"gate": key, "passed": bool(value), "value": bool(value)}
        for key, value in known_checks.items()
    )
    quality_rows.extend(
        [
            {
                "gate": "all_six_seeds_present_on_each_split",
                "passed": bool(
                    all(
                        set(seed_frame[seed_frame["split"] == split]["seed"]) == set(seeds)
                        for split in args.splits
                    )
                ),
                "value": len(seeds),
            },
            {
                "gate": "partial_formula_matches_direct_residualization",
                "passed": bool(
                    np.nanmax(np.abs(common_frame["partial_formula_direct_difference"])) <= 1e-10
                ),
                "value": float(np.nanmax(np.abs(common_frame["partial_formula_direct_difference"]))),
            },
            {
                "gate": "single_training_script_sha_across_seeds",
                "passed": bool(provenance["train_script_sha256"].dropna().nunique() <= 1),
                "value": int(provenance["train_script_sha256"].dropna().nunique()),
            },
        ]
    )
    quality = pd.DataFrame(quality_rows)
    failed = quality.loc[~quality["passed"].astype(bool)]
    if not failed.empty:
        raise RuntimeError("Quality gates failed:\n" + failed.to_string(index=False))
    paths["quality_gates"] = write_table(quality, table_root / "quality_gates")

    source_files = {
        "analysis_script": Path(__file__).resolve(),
        "null_recovery_manifest": null_manifest_path,
        "null_recovery_point_table": point_path,
    }
    source_inventory = {
        name: {
            "path": str(path.resolve()),
            "sha256": file_sha256(path.resolve()),
            "size_bytes": int(path.stat().st_size),
        }
        for name, path in source_files.items()
    }
    manifest = {
        "analysis": "six-seed construction-null calibration and common-target-conditioned cross-model agreement",
        "status": "post hoc supplementary output-only audit",
        "created_at_unix": time.time(),
        "runtime_seconds": float(time.time() - started),
        "output_root": str(output_root),
        "null_recovery_root": str(null_root),
        "seeds": list(seeds),
        "splits": list(args.splits),
        "primary_split": PRIMARY_SPLIT,
        "source_inventory": source_inventory,
        "event_ssl_provenance": provenance_rows,
        "headline_bootstrap": headline_audit,
        "output_tables": {name: str(path) for name, path in paths.items()},
        "data_boundary": {
            "model_training": False,
            "model_selection": False,
            "checkpoint_selection": False,
            "mechanism_refit": False,
            "coordinate_refit": False,
            "grid_or_support_refit": False,
            "construction_null_rerun": False,
            "scaffold_rescaling_refit_per_seed": False,
            "B_confirm_used_for_update": False,
            "learned_plane_used_for_null_subtraction": False,
            "new_learner_resampling": False,
        },
        "common_target_boundary": (
            "The product benchmark and partial correlation are descriptive linear calibrations on the same "
            "cell-component vector and fixed formal support. They are not cell-level inferential nulls and do not "
            "establish a shared mechanism. Direct empirical errors remain conditional on the estimated empirical field."
        ),
        "uncertainty_boundary": (
            "Across-seed Student-t intervals summarize six frozen training pipelines. The previously completed "
            "seed-42 learner-cluster bootstrap remains attached to its archived joined-cohort fixed-support "
            "headline estimand. When that point differs from the Stage-1-weighted null-audit point, the bootstrap "
            "interval is not recentered and the reconciliation is recorded in the bootstrap summary and manifest. "
            "This analysis performs no additional resampling."
        ),
    }
    manifest_path = metadata_root / "six_seed_null_common_target_audit_manifest.json"
    save_json(manifest, manifest_path)
    save_json(
        {"manifest_sha256": file_sha256(manifest_path)},
        metadata_root / "six_seed_null_common_target_audit_manifest.sha256.json",
    )
    print(f"Audit completed: {output_root}", flush=True)


if __name__ == "__main__":
    main()
