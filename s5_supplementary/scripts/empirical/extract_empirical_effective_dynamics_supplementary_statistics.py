#!/usr/bin/env python3
"""Extract supplementary tables and numerical reports for empirical effective dynamics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
COORDINATE = "MR_PsiA"
MACROSTATE_K = 6
PRIMARY_SPLIT = "A_val"
SPLITS = ("A_train", "A_val", "B_confirm")
XCOL = "M_response_prebalanced_pre"
YCOL = "activity_alignment_order_Psi_pre"
NULL_METRIC_LABELS = {
    "occupancy_weighted_full_field_distance_from_null_mean": "Full-field distance",
    "negative_divergence_occupancy_fraction": "Negative-divergence occupancy",
    "flow_weighted_shell_fraction_inward": "Shell inward flow",
    "flow_core_to_shell_speed_ratio": "Central slowing",
}
NULL_SUPPORT_DIRECTIONS = {
    "occupancy_weighted_full_field_distance_from_null_mean": "greater",
    "negative_divergence_occupancy_fraction": "greater",
    "flow_weighted_shell_fraction_inward": "greater",
    "flow_core_to_shell_speed_ratio": "less",
}


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
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_path(base: Path) -> Path:
    for path in (base.with_suffix(".parquet"), base.with_suffix(".csv.gz"), base.with_suffix(".csv")):
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find table for {base}")


def read_table(base: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    path = table_path(base)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    return pd.read_csv(path, usecols=list(columns) if columns is not None else None, low_memory=False)


def available_columns(base: Path) -> Sequence[str]:
    path = table_path(base)
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            return tuple(pq.read_schema(path).names)
        except Exception:
            return tuple(pd.read_parquet(path).columns)
    return tuple(pd.read_csv(path, nrows=0).columns)


def write_table(frame: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = base.with_suffix(".parquet")
        frame.to_parquet(path, index=False)
    except Exception:
        path = base.with_suffix(".csv.gz")
        frame.to_csv(path, index=False, compression="gzip")
    return path


def resolve_stage1_root(path: Path) -> Path:
    for candidate in (path, path / "stage1"):
        if (candidate / "dynamics").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"No Stage-1 dynamics directory under {path}")


def import_stage1_module(path: Path):
    source = path.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Stage-1 script not found: {source}")
    spec = importlib.util.spec_from_file_location("formal_stage1_supplement", str(source))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Stage-1 script: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = {
        "coordinate_specs",
        "occupancy_drift_stats",
        "field_stats_from_dict",
        "global_field_contraction_summary",
        "evaluate_frozen_convergence_region",
    }
    missing = sorted(name for name in required if not hasattr(module, name))
    if missing:
        raise RuntimeError(f"Stage-1 script is missing required functions: {missing}")
    return module


def pearson_safe(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3:
        return np.nan
    a = a[valid] - float(np.mean(a[valid]))
    b = b[valid] - float(np.mean(b[valid]))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > EPS else np.nan


def js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    p = np.asarray(first, dtype=float).ravel()
    q = np.asarray(second, dtype=float).ravel()
    p = np.maximum(p, 0.0)
    q = np.maximum(q, 0.0)
    p = p / max(float(np.sum(p)), EPS)
    q = q / max(float(np.sum(q)), EPS)
    middle = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log((left[mask] + EPS) / (right[mask] + EPS))))

    return 0.5 * kl(p, middle) + 0.5 * kl(q, middle)


def validate_kmeans_contract(metadata: Mapping[str, Any], centres: pd.DataFrame) -> None:
    expected = {
        "coordinate": COORDINATE,
        "macrostate_k": MACROSTATE_K,
        "macrostate_k_rule": "fixed a priori",
        "fit_split": "A_train",
        "features": [XCOL, YCOL],
        "user_balanced_sampling": True,
        "user_balanced_kmeans_fit": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"Fixed-K contract mismatch for {key}: {metadata.get(key)!r} != {value!r}"
            )
    if int(metadata.get("fit_max_rows", -1)) != 500000:
        raise RuntimeError("Fixed-K fit_max_rows must be 500000")
    if int(metadata.get("kmeans_n_init", -1)) != 20:
        raise RuntimeError("Fixed-K n_init must be 20")
    if int(metadata.get("random_state", -1)) != 42:
        raise RuntimeError("Fixed-K random state must be 42")

    scaler_mean = np.asarray(metadata.get("scaler_mean", []), dtype=float)
    scaler_scale = np.asarray(metadata.get("scaler_scale", []), dtype=float)
    if scaler_mean.shape != (2,) or not np.isfinite(scaler_mean).all():
        raise RuntimeError("Fixed-K scaler_mean must contain two finite values")
    if scaler_scale.shape != (2,) or not np.isfinite(scaler_scale).all() or np.any(scaler_scale <= 0):
        raise RuntimeError("Fixed-K scaler_scale must contain two positive finite values")

    mapping = metadata.get("raw_to_ordered_label", {})
    if not isinstance(mapping, Mapping):
        raise RuntimeError("Fixed-K raw_to_ordered_label must be a mapping")
    try:
        mapping_keys = sorted(int(key) for key in mapping)
        mapping_values = sorted(int(value) for value in mapping.values())
    except Exception as exc:
        raise RuntimeError("Fixed-K raw-to-ordered labels are not integer-valued") from exc
    expected_labels = list(range(MACROSTATE_K))
    if mapping_keys != expected_labels or mapping_values != expected_labels:
        raise RuntimeError("Fixed-K raw-to-ordered mapping must be a permutation of states 0--5")

    required = {
        "macrostate",
        "center_M",
        "center_Psi",
        "center_M_standardized",
        "center_Psi_standardized",
    }
    missing = sorted(required.difference(centres.columns))
    if missing:
        raise RuntimeError(f"Fixed-K centres are missing columns: {missing}")
    states = pd.to_numeric(centres["macrostate"], errors="raise").astype(int).to_numpy()
    if len(centres) != MACROSTATE_K or not np.array_equal(np.sort(states), np.arange(MACROSTATE_K)):
        raise RuntimeError("Fixed-K centres must contain states 0--5 exactly once")
    centre_values = centres[[
        "center_M",
        "center_Psi",
        "center_M_standardized",
        "center_Psi_standardized",
    ]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(centre_values).all():
        raise RuntimeError("Fixed-K centre coordinates must be finite")


def load_null_split(root: Path, split: str, minimum_replicates: int) -> Dict[str, Any]:
    root = root.resolve()
    tables = root / "tables"
    metadata = root / "metadata"
    manifest_path = metadata / f"{split}_construction_null_manifest.json"
    manifest = load_json(manifest_path)
    if str(manifest.get("analysis_split")) != split:
        raise RuntimeError(f"Construction-null split mismatch in {manifest_path}")
    replicate_count = int(manifest.get("replicates", 0))
    if replicate_count < minimum_replicates:
        raise RuntimeError(
            f"Construction-null run for {split} has fewer than {minimum_replicates} replicates"
        )
    if split == "B_confirm" and manifest.get("confirmation_output_only") is not True:
        raise RuntimeError("B_confirm construction-null run is not marked output-only")

    quality = dict(manifest.get("quality_gates", {}))
    required_true = (
        "same_row_phase_reconstruction",
        "next_state_coordinate_reconstruction",
        "next_mass_decay_audit",
        "formal_field_estimator_reproduced",
        "archived_stage1_field_matched",
        "joint_innovation_pairs_permuted_together",
        "global_innovation_marginals_preserved",
        "frozen_A_train_core_reused",
    )
    required_false = (
        "coordinate_or_region_refit",
        "B_confirm_used_for_definition_or_selection",
    )
    failed_true = [key for key in required_true if quality.get(key) is not True]
    failed_false = [key for key in required_false if quality.get(key) is not False]
    if failed_true or failed_false:
        raise RuntimeError(
            f"Construction-null quality gates failed for {split}: "
            f"required_true={failed_true}, required_false={failed_false}"
        )

    source_bases = {
        "summary": tables / f"{split}_construction_null_summary",
        "replicates": tables / f"{split}_construction_null_replicate_metrics",
        "matching": tables / f"{split}_matching_fallback_coverage",
        "composition": tables / f"{split}_opportunity_composition_audit",
    }
    source_paths = {name: table_path(base) for name, base in source_bases.items()}
    summary = read_table(source_bases["summary"])
    replicates = read_table(source_bases["replicates"])
    matching = read_table(source_bases["matching"])
    composition = read_table(source_bases["composition"])

    if len(replicates) != replicate_count:
        raise RuntimeError(
            f"Construction-null replicate table for {split} has {len(replicates)} rows; "
            f"manifest reports {replicate_count}"
        )
    if "replicate" in replicates.columns:
        observed = pd.to_numeric(replicates["replicate"], errors="coerce")
        if observed.isna().any() or observed.nunique() != replicate_count:
            raise RuntimeError(f"Construction-null replicate identifiers are invalid for {split}")

    source_files = {
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        **{
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
    }
    return {
        "root": root,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "summary": summary,
        "replicates": replicates,
        "matching": matching,
        "composition": composition,
        "source_files": source_files,
    }


def null_test_table(bundle: Mapping[str, Any], split: str) -> pd.DataFrame:
    summary = bundle["summary"].copy()
    required = {
        "metric",
        "observed",
        "null_mean",
        "null_sd",
        "null_2p5",
        "null_97p5",
        "monte_carlo_p",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise RuntimeError(f"Construction-null summary is missing columns: {missing}")
    rows = []
    for _, row in summary.iterrows():
        metric = str(row["metric"])
        direction = str(row.get("direction_supporting_excess_structure", NULL_SUPPORT_DIRECTIONS.get(metric, "greater")))
        observed = float(row["observed"])
        null_mean = float(row["null_mean"])
        null_sd = float(row["null_sd"])
        sign = 1.0 if direction == "greater" else -1.0
        z = sign * (observed - null_mean) / null_sd if null_sd > 0 else np.nan
        lower_z = sign * (float(row["null_2p5"]) - null_mean) / null_sd if null_sd > 0 else np.nan
        upper_z = sign * (float(row["null_97p5"]) - null_mean) / null_sd if null_sd > 0 else np.nan
        rows.append(
            {
                "split": split,
                "metric": metric,
                "display_metric": NULL_METRIC_LABELS.get(metric, metric),
                "supportive_direction": direction,
                "observed": observed,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "null_2p5": float(row["null_2p5"]),
                "null_97p5": float(row["null_97p5"]),
                "supportive_standardized_departure": z,
                "null_2p5_standardized": min(lower_z, upper_z),
                "null_97p5_standardized": max(lower_z, upper_z),
                "monte_carlo_p": float(row["monte_carlo_p"]),
                "BH_q_across_three_basin_metrics": pd.to_numeric(
                    pd.Series([row.get("BH_q_across_three_basin_metrics", np.nan)]), errors="coerce"
                ).iloc[0],
                "excess_field_value_descriptive": pd.to_numeric(
                    pd.Series([row.get("excess_field_value_descriptive", np.nan)]), errors="coerce"
                ).iloc[0],
            }
        )
    return pd.DataFrame(rows)



def compact_construction_null_table(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["null_95pct_interval"] = data.apply(
        lambda row: f"[{float(row['null_2p5']):.4g}, {float(row['null_97p5']):.4g}]",
        axis=1,
    )
    output = data[
        [
            "split",
            "display_metric",
            "supportive_direction",
            "observed",
            "null_mean",
            "null_95pct_interval",
            "monte_carlo_p",
            "BH_q_across_three_basin_metrics",
        ]
    ].copy()
    return output.rename(
        columns={
            "display_metric": "metric",
            "supportive_direction": "direction",
            "null_mean": "matched_null_mean",
            "monte_carlo_p": "p_MC",
            "BH_q_across_three_basin_metrics": "q_BH",
        }
    )


def construction_matching_audit(bundle: Mapping[str, Any], split: str) -> pd.DataFrame:
    matching = bundle["matching"].copy()
    composition = bundle["composition"].copy()
    required_matching = {"level", "rows_assigned", "fraction_of_randomizable_rows"}
    missing_matching = sorted(required_matching.difference(matching.columns))
    if missing_matching:
        raise RuntimeError(f"Matching audit is missing columns: {missing_matching}")
    if composition.empty:
        raise RuntimeError(f"Opportunity-composition audit is empty for {split}")

    comp = composition.iloc[0]
    analysis_rows = int(comp.get("analysis_rows", 0))
    randomizable_rows = int(comp.get("randomizable_rows", 0))
    assigned = pd.to_numeric(matching["rows_assigned"], errors="coerce").fillna(0.0)
    levels = matching["level"].astype(str)

    def total_for(prefixes: Tuple[str, ...]) -> int:
        mask = np.zeros(len(matching), dtype=bool)
        for prefix in prefixes:
            mask |= levels.str.startswith(prefix).to_numpy()
        return int(assigned[mask].sum())

    within_user = total_for(("within_user",))
    across_user = total_for(("across_user",))
    opportunity = total_for(("global_opportunity",))
    last_resort = total_for(("global_last_resort",))
    unmatched = total_for(("unmatched_singleton_self_exempt",))
    denominator = max(randomizable_rows, 1)

    return pd.DataFrame(
        [
            {
                "split": split,
                "replicates": int(bundle["manifest"].get("replicates", 0)),
                "analysis_rows": analysis_rows,
                "randomizable_rows": randomizable_rows,
                "randomizable_fraction": randomizable_rows / max(analysis_rows, 1),
                "within_user_matched_fraction": within_user / denominator,
                "across_user_matched_fraction": across_user / denominator,
                "global_opportunity_fraction": opportunity / denominator,
                "weak_fallback_fraction": (last_resort + unmatched) / denominator,
                "response_increment_present_fraction": float(
                    comp.get("response_increment_present_fraction", np.nan)
                ),
                "exposure_increment_present_fraction": float(
                    comp.get("exposure_increment_present_fraction", np.nan)
                ),
                "support_present_fraction": float(comp.get("support_present_fraction", np.nan)),
                "idle_present_fraction": float(comp.get("idle_present_fraction", np.nan)),
            }
        ]
    )


def user_balanced_state_occupancy(assignments: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "macrostate"}
    missing = sorted(required.difference(assignments.columns))
    if missing:
        raise RuntimeError(f"Assignment table is missing columns: {missing}")
    frame = assignments.copy()
    frame["user_id"] = pd.to_numeric(frame["user_id"], errors="coerce")
    frame["macrostate"] = pd.to_numeric(frame["macrostate"], errors="coerce")
    frame = frame[frame["user_id"].notna()].copy()
    counts = frame.groupby("user_id")["user_id"].transform("count").to_numpy(dtype=float)
    frame["row_weight"] = 1.0 / np.maximum(counts, 1.0)
    valid = frame[frame["macrostate"].notna()].copy()
    valid["macrostate"] = valid["macrostate"].astype(int)
    grouped = valid.groupby("macrostate", sort=True).agg(
        observed_state_rows=("macrostate", "size"),
        observed_state_users=("user_id", "nunique"),
        user_balanced_mass=("row_weight", "sum"),
    )
    total = float(grouped["user_balanced_mass"].sum())
    grouped["user_balanced_occupancy_fraction"] = grouped["user_balanced_mass"] / max(total, EPS)
    return grouped.reset_index()


def load_mesostate_summary(stage1_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    root = stage1_root / "dynamics" / "fixed_k6_mesostates"
    metadata_path = root / "fixed_k6_model_metadata.json"
    metadata = load_json(metadata_path)
    centres = read_table(root / "fixed_k6_centers")
    validate_kmeans_contract(metadata, centres)
    centres = centres.sort_values("macrostate", kind="mergesort").copy()
    rows = []
    for split in SPLITS:
        try:
            assignments = read_table(root / f"{split}_fixed_k6_assignments")
            transition = read_table(root / f"{split}_fixed_k6_transition_matrix").to_numpy(dtype=float)
            counts = read_table(root / f"{split}_fixed_k6_transition_counts").to_numpy(dtype=float)
            residence = read_table(root / f"{split}_fixed_k6_residence_summary")
        except FileNotFoundError:
            continue
        if transition.shape != (MACROSTATE_K, MACROSTATE_K) or counts.shape != (MACROSTATE_K, MACROSTATE_K):
            raise RuntimeError(f"Transition matrices for {split} are not 6 by 6")
        occupancy = user_balanced_state_occupancy(assignments).set_index("macrostate")
        residence = residence.copy()
        residence["macrostate"] = pd.to_numeric(residence["macrostate"], errors="raise").astype(int)
        residence = residence.set_index("macrostate")
        for state in range(MACROSTATE_K):
            centre = centres[centres["macrostate"].astype(int) == state].iloc[0]
            occ = occupancy.loc[state] if state in occupancy.index else pd.Series(dtype=float)
            res = residence.loc[state] if state in residence.index else pd.Series(dtype=float)
            row = transition[state]
            dominant = int(np.nanargmax(row)) if np.isfinite(row).any() else -1
            rows.append(
                {
                    "split": split,
                    "macrostate": state,
                    "state_label": f"S{state}",
                    "center_M": float(centre["center_M"]),
                    "center_Psi": float(centre["center_Psi"]),
                    "user_balanced_occupancy_fraction": float(occ.get("user_balanced_occupancy_fraction", np.nan)),
                    "observed_state_rows": int(occ.get("observed_state_rows", 0)),
                    "observed_state_users": int(occ.get("observed_state_users", 0)),
                    "outgoing_transition_count": int(np.sum(counts[state])),
                    "self_transition_probability": float(transition[state, state]),
                    "dominant_next_state": dominant,
                    "diagonal_dominant": bool(dominant == state),
                    "n_residence_runs": int(res.get("n_runs", 0)),
                    "n_completed_exits": int(res.get("n_completed_exits", 0)),
                    "n_right_censored": int(res.get("n_right_censored", 0)),
                    "right_censoring_fraction": float(res.get("right_censoring_fraction", np.nan)),
                    "rmst_tau": float(res.get("rmst_tau", np.nan)),
                    "observed_restricted_mean_residence": float(res.get("obs_restricted_mean_residence", np.nan)),
                    "geometric_restricted_mean_residence": float(res.get("geo_null_restricted_mean", np.nan)),
                    "restricted_mean_residence_lift": float(res.get("restricted_mean_residence_lift", np.nan)),
                    "reference_length": int(res.get("reference_length", 0)),
                    "reference_at_risk": int(res.get("reference_at_risk", 0)),
                    "observed_tail_probability_at_reference": float(res.get("observed_tail_probability_at_reference", np.nan)),
                    "geometric_tail_probability_at_reference": float(res.get("geometric_tail_probability_at_reference", np.nan)),
                    "tail_ratio_at_reference": float(res.get("tail_ratio_at_reference", np.nan)),
                    "tail_excess_pvalue_greenwood": float(res.get("tail_excess_pvalue_greenwood", np.nan)),
                    "tail_excess_qvalue_bh": float(res.get("tail_excess_qvalue_bh", np.nan)),
                    "tail_excess_significant_bh": bool(res.get("tail_excess_significant_bh", False)),
                }
            )
    output = pd.DataFrame(rows)
    validation = output[output["split"].astype(str) == PRIMARY_SPLIT] if not output.empty else output
    if len(validation) != MACROSTATE_K or set(validation["macrostate"].astype(int)) != set(range(MACROSTATE_K)):
        raise RuntimeError(
            "Supplementary Table 1 requires exactly six A_val mesostate rows from the frozen partition"
        )
    audit = {
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "centres_path": str(table_path(root / "fixed_k6_centers").resolve()),
        "centres_sha256": sha256_file(table_path(root / "fixed_k6_centers")),
        "macrostate_k": MACROSTATE_K,
        "fit_split": metadata["fit_split"],
        "fit_rows": int(metadata.get("fit_rows", 0)),
    }
    return output, audit


def field_similarity(reference: Any, comparison: Any) -> Dict[str, float]:
    common = np.asarray(reference.drift_mask, dtype=bool) & np.asarray(comparison.drift_mask, dtype=bool)
    if not np.any(common):
        return {
            "common_supported_drift_cells": 0,
            "drift_vector_correlation_to_all_rows": np.nan,
            "mean_local_drift_cosine_to_all_rows": np.nan,
            "drift_component_rmse_to_all_rows": np.nan,
            "drift_speed_correlation_to_all_rows": np.nan,
            "occupancy_js_to_all_rows": js_divergence(reference.occupancy_probability, comparison.occupancy_probability),
        }
    ref_u = np.asarray(reference.drift_u, dtype=float)[common]
    ref_v = np.asarray(reference.drift_v, dtype=float)[common]
    cmp_u = np.asarray(comparison.drift_u, dtype=float)[common]
    cmp_v = np.asarray(comparison.drift_v, dtype=float)[common]
    ref_speed = np.sqrt(ref_u * ref_u + ref_v * ref_v)
    cmp_speed = np.sqrt(cmp_u * cmp_u + cmp_v * cmp_v)
    nonzero = (ref_speed > EPS) & (cmp_speed > EPS)
    cosine = (
        float(np.mean((ref_u[nonzero] * cmp_u[nonzero] + ref_v[nonzero] * cmp_v[nonzero]) / (ref_speed[nonzero] * cmp_speed[nonzero])))
        if np.any(nonzero)
        else np.nan
    )
    return {
        "common_supported_drift_cells": int(np.sum(common)),
        "drift_vector_correlation_to_all_rows": pearson_safe(
            np.concatenate([ref_u, ref_v]), np.concatenate([cmp_u, cmp_v])
        ),
        "mean_local_drift_cosine_to_all_rows": cosine,
        "drift_component_rmse_to_all_rows": float(
            np.sqrt(np.mean(np.concatenate([ref_u - cmp_u, ref_v - cmp_v]) ** 2))
        ),
        "drift_speed_correlation_to_all_rows": pearson_safe(ref_speed, cmp_speed),
        "occupancy_js_to_all_rows": js_divergence(reference.occupancy_probability, comparison.occupancy_probability),
    }


def observation_sensitivity(
    stage1_root: Path,
    stage1_script: Path,
    splits: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    module = import_stage1_module(stage1_script)
    spec = module.coordinate_specs()[0]
    if spec.name != COORDINATE:
        raise RuntimeError(f"Unexpected primary coordinate: {spec.name}")
    region_root = stage1_root / "dynamics" / "candidate_regions" / COORDINATE
    core_path = region_root / "A_train_primary_convergence_core_mask.npy"
    thresholds_path = region_root / "training_convergence_thresholds.json"
    core = np.load(core_path)
    thresholds = load_json(thresholds_path)
    shell_radius = float(thresholds["shell_radius"])
    rows = []
    for split in splits:
        columns = [
            "user_id",
            spec.xcol,
            spec.ycol,
            spec.dxcol,
            spec.dycol,
            "has_next_submitted_bundle",
            "long_gap_or_no_next",
        ]
        base = stage1_root / "dynamics" / f"student_dynamics_panel_core_{split}"
        try:
            present = set(available_columns(base))
            frame = read_table(base, columns=[column for column in columns if column in present])
        except FileNotFoundError:
            continue
        variants = {"all_rows": frame}
        if "has_next_submitted_bundle" in frame.columns:
            variants["next_observed_only"] = frame[frame["has_next_submitted_bundle"].astype(bool)].copy()
        if "long_gap_or_no_next" in frame.columns:
            variants["long_gap_excluded"] = frame[~frame["long_gap_or_no_next"].astype(bool)].copy()
        fields: Dict[str, Any] = {}
        for name, subset in variants.items():
            stats = module.occupancy_drift_stats(subset, spec)
            fields[name] = module.field_stats_from_dict(stats)
        reference = fields["all_rows"]
        for name, subset in variants.items():
            field = fields[name]
            contraction = module.global_field_contraction_summary(field, split)
            region, _ = module.evaluate_frozen_convergence_region(
                field,
                core,
                f"{split}_{name}",
                shell_radius,
            )
            region_row = region.iloc[0].to_dict()
            rows.append(
                {
                    "split": split,
                    "variant": name,
                    "rows": int(len(subset)),
                    "users": int(subset["user_id"].nunique()) if "user_id" in subset.columns else np.nan,
                    "valid_state_rows": int(field.valid_state_rows),
                    "valid_drift_rows": int(field.valid_drift_rows),
                    "supported_drift_cells": int(np.sum(field.drift_mask)),
                    **field_similarity(reference, field),
                    "negative_divergence_occupancy_fraction": float(
                        contraction["weighted_negative_divergence_fraction_interior_only"]
                    ),
                    "weighted_mean_divergence": float(
                        contraction["weighted_mean_local_divergence_interior_only"]
                    ),
                    "frozen_core_occupancy_fraction": float(region_row.get("occupancy_mass_fraction", np.nan)),
                    "frozen_core_shell_inward_fraction": float(
                        region_row.get("flow_weighted_shell_fraction_inward", np.nan)
                    ),
                    "frozen_core_to_shell_speed_ratio": float(
                        region_row.get("flow_core_to_shell_speed_ratio", np.nan)
                    ),
                    "regions_reselected": False,
                }
            )
    audit = {
        "stage1_script": str(stage1_script.resolve()),
        "stage1_script_sha256": sha256_file(stage1_script.resolve()),
        "core_path": str(core_path.resolve()),
        "core_sha256": sha256_file(core_path),
        "thresholds_path": str(thresholds_path.resolve()),
        "thresholds_sha256": sha256_file(thresholds_path),
        "shell_radius": shell_radius,
    }
    return pd.DataFrame(rows), audit


def compact_validation_mesostate_table(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame[frame["split"].astype(str) == PRIMARY_SPLIT].copy()
    rows = []
    for _, row in data.iterrows():
        center = f"({float(row['center_M']):.3f}, {float(row['center_Psi']):.3f})"
        runs = int(row["n_residence_runs"])
        censor = float(row["right_censoring_fraction"])
        rmst_lift = float(row["restricted_mean_residence_lift"])
        tau = int(row["rmst_tau"])
        risk = int(row["reference_at_risk"])
        tail_ratio = float(row["tail_ratio_at_reference"])
        q_value = float(row["tail_excess_qvalue_bh"])
        rows.append(
            {
                "state": str(row["state_label"]),
                "training_center_M_Psi": center,
                "validation_user_balanced_occupancy": float(row["user_balanced_occupancy_fraction"]),
                "validation_self_transition": float(row["self_transition_probability"]),
                "residence_runs_and_censoring": f"{runs:,} ({100.0*censor:.1f}% censored)",
                "RMST_lift_and_horizon": f"{rmst_lift:.3f} (tau={tau:,})",
                "tail_at_10": f"ratio={tail_ratio:.3g}; risk={risk:,}; q={q_value:.3g}",
            }
        )
    return pd.DataFrame(rows)



def compact_observation_sensitivity_table(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame[frame["split"].isin(["A_val", "B_confirm"])].copy()
    columns = [
        "split",
        "variant",
        "valid_drift_rows",
        "supported_drift_cells",
        "drift_vector_correlation_to_all_rows",
        "mean_local_drift_cosine_to_all_rows",
        "drift_speed_correlation_to_all_rows",
        "negative_divergence_occupancy_fraction",
        "frozen_core_shell_inward_fraction",
        "frozen_core_to_shell_speed_ratio",
    ]
    missing = sorted(set(columns).difference(data.columns))
    if missing:
        raise RuntimeError(f"Observation-sensitivity table is missing columns: {missing}")
    return data[columns].reset_index(drop=True)


def split_independence_audit(stage1_root: Path) -> pd.DataFrame:
    users: Dict[str, set[int]] = {}
    for split in SPLITS:
        try:
            frame = read_table(stage1_root / "splits" / f"{split}_users")
        except FileNotFoundError:
            continue
        if "user_id" not in frame.columns:
            continue
        users[split] = set(pd.to_numeric(frame["user_id"], errors="coerce").dropna().astype(int).tolist())
    rows = []
    names = list(users)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = users[first].intersection(users[second])
            rows.append(
                {
                    "split_a": first,
                    "split_b": second,
                    "users_a": len(users[first]),
                    "users_b": len(users[second]),
                    "overlap_users": len(overlap),
                    "user_disjoint": len(overlap) == 0,
                }
            )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 4) -> str:
    selected = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    for column in selected.columns:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(
                lambda value: "" if not np.isfinite(value) else f"{value:.{digits}g}"
            )
    try:
        return selected.to_markdown(index=False)
    except Exception:
        return selected.to_csv(index=False)


def build_markdown(
    null_tests: pd.DataFrame,
    matching_audit: pd.DataFrame,
    mesostates: pd.DataFrame,
    sensitivity: pd.DataFrame,
    quality: pd.DataFrame,
) -> str:
    lines = [
        "# Empirical effective-dynamics supplementary report",
        "",
        "## Recommended Supplementary Information package",
        "",
        "1. **Supplementary Note 1:** construction-matched accounting null, reported as a compact numerical table with matching-quality audit.",
        "2. **Supplementary Table 1:** frozen six-state centres with statewise occupancy, persistence and censor-aware residence diagnostics.",
        "3. **Supplementary Table 2:** observation-gap sensitivity under the frozen coordinate, grid, convergence core and mesostate contract.",
        "",
        "No supplementary figure is required for the empirical stage. Large cellwise, row-level and replicate-level outputs remain in the public repository rather than the typeset supplementary file.",
        "",
        "## Supplementary Note 1: construction-matched accounting null",
        "",
        markdown_table(
            null_tests,
            [
                "split",
                "display_metric",
                "observed",
                "null_mean",
                "null_2p5",
                "null_97p5",
                "supportive_standardized_departure",
                "monte_carlo_p",
                "BH_q_across_three_basin_metrics",
            ],
        ),
        "",
        "### Matching audit",
        "",
        markdown_table(
            matching_audit,
            [
                "split",
                "replicates",
                "analysis_rows",
                "randomizable_fraction",
                "within_user_matched_fraction",
                "across_user_matched_fraction",
                "global_opportunity_fraction",
                "weak_fallback_fraction",
            ],
        ),
        "",
        "## Supplementary Table 1: frozen six-state kinetic coarse-graining",
        "",
        markdown_table(
            mesostates[mesostates["split"] == PRIMARY_SPLIT],
            [
                "state_label",
                "center_M",
                "center_Psi",
                "user_balanced_occupancy_fraction",
                "self_transition_probability",
                "n_residence_runs",
                "right_censoring_fraction",
                "rmst_tau",
                "restricted_mean_residence_lift",
                "reference_at_risk",
                "tail_ratio_at_reference",
                "tail_excess_qvalue_bh",
            ],
        ),
        "",
        "## Supplementary Table 2: observation-gap sensitivity",
        "",
        markdown_table(
            sensitivity[sensitivity["split"].isin(["A_val", "B_confirm"])],
            [
                "split",
                "variant",
                "valid_drift_rows",
                "supported_drift_cells",
                "drift_vector_correlation_to_all_rows",
                "mean_local_drift_cosine_to_all_rows",
                "drift_speed_correlation_to_all_rows",
                "negative_divergence_occupancy_fraction",
                "frozen_core_shell_inward_fraction",
                "frozen_core_to_shell_speed_ratio",
            ],
        ),
        "",
        "## Quality gates",
        "",
        markdown_table(quality, list(quality.columns), digits=6),
        "",
        "## Suggested captions",
        "",
        "**Supplementary Note 1 | Construction-matched accounting null.** The observed validation field and three prespecified basin diagnostics were compared with matched joint signed-innovation permutations that preserved current states, denominator increments, user-balanced weights, field support and the training-defined convergence core. Validation supplied formal Monte Carlo inference; confirmation was evaluated independently under the frozen protocol. Full-field and basin statistics are reported with matched-null means, 95% intervals and the corresponding Monte Carlo or Benjamini--Hochberg-adjusted values.",
        "",
        "**Supplementary Table 1 | Frozen six-state kinetic coarse-graining.** Training-defined centres and validation statewise occupancy, self-transition, right-censoring, restricted-mean residence lift and the prespecified 10-step Greenwood--Benjamini--Hochberg tail diagnostic. The six-state partition was fitted only on training coordinates and applied unchanged.",
        "",
        "**Supplementary Table 2 | Observation-gap sensitivity.** Field and frozen-core diagnostics after restricting to observed successors or excluding terminal rows and gaps longer than seven days. Coordinates, grid support, convergence criteria and the six-state partition were not reselected.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage1-script", type=Path, required=True)
    parser.add_argument("--null-validation-root", type=Path, required=True)
    parser.add_argument("--null-confirmation-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minimum-null-replicates", type=int, default=100)
    parser.add_argument(
        "--sensitivity-splits",
        nargs="+",
        default=["A_val", "B_confirm"],
        choices=list(SPLITS),
    )
    parser.add_argument("--skip-observation-sensitivity", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stage1_root = resolve_stage1_root(args.stage1_root)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    report_root = output_root / "reports"

    null_bundles: Dict[str, Dict[str, Any]] = {
        "A_val": load_null_split(
            args.null_validation_root,
            "A_val",
            args.minimum_null_replicates,
        )
    }
    confirmation_root = args.null_confirmation_root or args.null_validation_root
    try:
        null_bundles["B_confirm"] = load_null_split(
            confirmation_root,
            "B_confirm",
            args.minimum_null_replicates,
        )
    except FileNotFoundError:
        pass

    null_tests = pd.concat(
        [null_test_table(bundle, split) for split, bundle in null_bundles.items()],
        ignore_index=True,
    )
    null_tests_compact = compact_construction_null_table(null_tests)
    matching_audit = pd.concat(
        [construction_matching_audit(bundle, split) for split, bundle in null_bundles.items()],
        ignore_index=True,
    )

    mesostates, kmeans_audit = load_mesostate_summary(stage1_root)
    mesostates_compact = compact_validation_mesostate_table(mesostates)
    if args.skip_observation_sensitivity:
        sensitivity = read_table(
            stage1_root
            / "dynamics"
            / "coordinate_analysis"
            / COORDINATE
            / "drift_observation_sensitivity_all_splits"
        )
        sensitivity_audit = {
            "source": "stored Stage-1 observation-sensitivity summary",
            "enhanced_field_comparison": False,
        }
    else:
        sensitivity, sensitivity_audit = observation_sensitivity(
            stage1_root,
            args.stage1_script,
            args.sensitivity_splits,
        )
    sensitivity_compact = compact_observation_sensitivity_table(sensitivity)
    split_audit = split_independence_audit(stage1_root)

    quality_rows = [
        {
            "scope": "fixed_k6_partition",
            "gate": "verified",
            "passed": True,
            "detail": f"K={MACROSTATE_K}; fit_split=A_train; fit_rows={kmeans_audit['fit_rows']}",
        },
        {
            "scope": "construction_null/A_val",
            "gate": "minimum_replicates",
            "passed": int(null_bundles["A_val"]["manifest"]["replicates"])
            >= args.minimum_null_replicates,
            "detail": int(null_bundles["A_val"]["manifest"]["replicates"]),
        },
        {
            "scope": "construction_null/A_val",
            "gate": "quality_gates",
            "passed": True,
            "detail": "reconstruction, matched permutation and no-refit audits passed",
        },
    ]
    if "B_confirm" in null_bundles:
        quality_rows.extend(
            [
                {
                    "scope": "construction_null/B_confirm",
                    "gate": "output_only",
                    "passed": null_bundles["B_confirm"]["manifest"].get(
                        "confirmation_output_only"
                    )
                    is True,
                    "detail": "confirmation protocol frozen before evaluation",
                },
                {
                    "scope": "construction_null/B_confirm",
                    "gate": "quality_gates",
                    "passed": True,
                    "detail": "reconstruction, matched permutation and no-refit audits passed",
                },
            ]
        )
    for _, row in split_audit.iterrows():
        quality_rows.append(
            {
                "scope": f"split_independence/{row['split_a']}-{row['split_b']}",
                "gate": "zero_user_overlap",
                "passed": bool(row["user_disjoint"]),
                "detail": int(row["overlap_users"]),
            }
        )
    quality = pd.DataFrame(quality_rows)
    if not bool(quality["passed"].all()):
        failed = quality[~quality["passed"]]
        raise RuntimeError(
            f"Supplementary quality gates failed:\n{failed.to_string(index=False)}"
        )

    output_paths = {
        "construction_null_tests": write_table(
            null_tests,
            table_root / "supplementary_construction_matched_null_tests",
        ),
        "construction_null_compact": write_table(
            null_tests_compact,
            table_root / "supplementary_construction_matched_null_compact",
        ),
        "construction_null_matching_audit": write_table(
            matching_audit,
            table_root / "supplementary_construction_matching_audit",
        ),
        "mesostate_all_splits": write_table(
            mesostates,
            table_root / "supplementary_empirical_mesostate_kinetics_all_splits",
        ),
        "mesostate_validation": write_table(
            mesostates[mesostates["split"] == PRIMARY_SPLIT].copy(),
            table_root / "supplementary_table1_validation_mesostate_kinetics",
        ),
        "mesostate_validation_compact": write_table(
            mesostates_compact,
            table_root / "supplementary_table1_validation_mesostate_kinetics_compact",
        ),
        "observation_sensitivity": write_table(
            sensitivity,
            table_root / "supplementary_table2_observation_gap_sensitivity",
        ),
        "observation_sensitivity_compact": write_table(
            sensitivity_compact,
            table_root / "supplementary_table2_observation_gap_sensitivity_compact",
        ),
        "split_independence": write_table(
            split_audit,
            table_root / "supplementary_split_independence_audit",
        ),
        "quality_gates": write_table(
            quality,
            table_root / "supplementary_empirical_quality_gates",
        ),
    }

    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / "empirical_effective_dynamics_supplementary_report.md"
    report_path.write_text(
        build_markdown(
            null_tests,
            matching_audit,
            mesostates,
            sensitivity,
            quality,
        ),
        encoding="utf-8",
    )

    manifest = {
        "script": Path(__file__).name,
        "stage1_root": str(stage1_root),
        "stage1_script": str(args.stage1_script.resolve()),
        "stage1_script_sha256": sha256_file(args.stage1_script.resolve()),
        "primary_coordinates": ["M", "Psi"],
        "fixed_k6_contract": kmeans_audit,
        "construction_null_roots": {
            split: str(bundle["root"]) for split, bundle in null_bundles.items()
        },
        "construction_null_sources": {
            split: bundle["source_files"] for split, bundle in null_bundles.items()
        },
        "construction_null_replicates": {
            split: int(bundle["manifest"]["replicates"])
            for split, bundle in null_bundles.items()
        },
        "observation_sensitivity_audit": sensitivity_audit,
        "outputs": {name: str(path.resolve()) for name, path in output_paths.items()},
        "report": str(report_path.resolve()),
        "figures_generated": False,
    }
    save_json(manifest, metadata_root / "empirical_supplementary_statistics_manifest.json")
    print(f"Supplementary empirical tables and report written to {output_root}")



if __name__ == "__main__":
    main()
