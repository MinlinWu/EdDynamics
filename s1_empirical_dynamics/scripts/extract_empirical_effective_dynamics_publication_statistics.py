#!/usr/bin/env python3
"""Extract Figure 2 and empirical-dynamics manuscript statistics from Stage 1 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
COORDINATE = "MR_PsiA"
MACROSTATE_K = 6
XCOL = "M_response_prebalanced_pre"
YCOL = "activity_alignment_order_Psi_pre"

VALUE_MAP_ROWS: Tuple[Tuple[str, str], ...] = (
    ("Training-defined compact flow-localization M interval (50% evidence)", "training_primary_convergence_flow_localization_M_shortest_interval_50_low / training_primary_convergence_flow_localization_M_shortest_interval_50_high"),
    ("Training-defined compact flow-localization Psi interval (50% evidence)", "training_primary_convergence_flow_localization_Psi_shortest_interval_50_low / training_primary_convergence_flow_localization_Psi_shortest_interval_50_high"),
    ("Training-defined flow-weighted convergence centre", "training_primary_convergence_convergence_center_M / training_primary_convergence_convergence_center_Psi"),
    ("Training local-affine fixed-point estimate", "training_primary_convergence_local_fixed_point_M / training_primary_convergence_local_fixed_point_Psi"),
    ("Validation supported drift cells", "validation_global_drift_supported_cells"),
    ("Validation interior-only negative-divergence occupancy fraction", "validation_global_weighted_negative_divergence_fraction_interior_only"),
    ("Validation interior-only weighted mean divergence", "validation_global_weighted_mean_local_divergence_interior_only"),
    ("Validation occupancy mass in frozen convergence core", "validation_frozen_primary_convergence_occupancy_mass_fraction"),
    ("Validation flow-weighted inward shell fraction", "validation_frozen_primary_convergence_flow_weighted_shell_fraction_inward"),
    ("Validation flow-weighted core-to-shell drift-speed ratio", "validation_frozen_primary_convergence_flow_core_to_shell_speed_ratio"),
    ("Training-validation convergence-mask Jaccard", "train_validation_convergence_mask_jaccard"),
    ("Training-validation convergence-centre distance", "train_validation_convergence_convergence_center_distance"),
    ("Validation first-shell flow-weighted inward fraction", "validation_convergence_radial_first_shell_flow_weighted_fraction_inward"),
    ("Validation first-shell flow-weighted inward cosine", "validation_convergence_radial_first_shell_flow_weighted_mean_inward_cosine"),
    ("Validation Psi 50% shortest occupancy interval", "validation_occupancy_Psi_shortest_interval_50_low / validation_occupancy_Psi_shortest_interval_50_high"),
    ("Validation M outermost-bin occupancy mass", "validation_occupancy_M_outermost_one_bin_mass_fraction"),
    ("Macrostate count", "macrostate_k"),
    ("Validation transitions", "validation_transition_count"),
    ("Diagonal-dominant states", "validation_diagonal_dominant_states"),
    ("Mean self-transition probability", "validation_mean_diagonal_probability"),
    ("Self-transition range", "validation_diagonal_probability_range"),
    ("Validation residence runs", "validation_residence_runs_total"),
    ("Completed exits / right-censored runs", "validation_residence_completed_exits_total / validation_residence_right_censored_total"),
    ("Overall residence right-censoring fraction", "validation_residence_right_censoring_fraction"),
    ("State-specific RMST horizon range", "validation_residence_rmst_tau_range"),
    ("Censor-aware restricted-mean residence-ratio range", "validation_restricted_mean_residence_lift_range"),
    ("Reference residence length", "residence_reference_length"),
    ("At-risk range at the reference length", "validation_reference_at_risk_range"),
    ("Kaplan--Meier tail-ratio range at the reference length", "validation_tail_ratio_at_reference_range"),
    ("States with descriptive tail excess at the reference length", "validation_tail_excess_states_descriptive_at_reference"),
    ("States with BH-significant tail excess at the reference length", "validation_tail_excess_states_significant_bh"),
    ("States with descriptive KM tail intervals", "validation_km_tail_excess_states_descriptive"),
    ("Reliable KM tail horizon maximum", "validation_km_tail_max_reliable_length"),
    ("Occupancy JS divergence", "train_validation_occupancy_js_divergence"),
    ("Common supported drift cells", "train_validation_common_supported_drift_cells"),
    ("Mean local drift cosine", "train_validation_mean_local_drift_cosine"),
    ("Drift component RMSE", "train_validation_drift_component_rmse"),
    ("Drift-speed correlation", "train_validation_drift_speed_pearson"),
    ("Residence RMST-ratio mean absolute log difference", "train_validation_residence_rmst_mean_abs_log_ratio"),
    ("Residence fixed-reference tail-ratio mean absolute log difference", "train_validation_residence_tail_ratio_mean_abs_log_ratio"),
    ("Residence right-censoring-fraction mean absolute difference", "train_validation_residence_right_censoring_fraction_mean_abs_difference"),
    ("Transition mean row TV", "train_validation_transition_mean_row_total_variation"),
    ("Transition maximum row TV", "train_validation_transition_max_row_total_variation"),
)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
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


def save_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_table(base: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    for path in (base.with_suffix(".parquet"), base.with_suffix(".csv.gz"), base.with_suffix(".csv")):
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
        return pd.read_csv(path, usecols=list(columns) if columns is not None else None, low_memory=False)
    raise FileNotFoundError(f"Could not find table for {base}")


def write_table(frame: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = base.with_suffix(".parquet")
        frame.to_parquet(path, index=False)
    except Exception:
        path = base.with_suffix(".csv.gz")
        frame.to_csv(path, index=False, compression="gzip")
    return path


def table_path(base: Path) -> Path:
    for path in (base.with_suffix(".parquet"), base.with_suffix(".csv.gz"), base.with_suffix(".csv")):
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find table for {base}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_stage1_root(path: Path) -> Path:
    for candidate in (path, path / "stage1"):
        if (candidate / "dynamics").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"No Stage-1 dynamics directory found under {path}")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")


def first_row(frame: pd.DataFrame, label: str) -> pd.Series:
    if frame.empty:
        raise RuntimeError(f"{label} is empty")
    return frame.iloc[0]


def numeric_range(values: pd.Series, decimals: int = 3) -> str:
    numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    if numbers.size == 0:
        return ""
    return f"{numbers.min():.{decimals}f}--{numbers.max():.{decimals}f}"


def integer_range(values: pd.Series) -> str:
    numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    if numbers.size == 0:
        return ""
    return f"{int(numbers.min())}--{int(numbers.max())}"


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
            raise RuntimeError(f"KMeans contract mismatch for {key}: {metadata.get(key)!r} != {value!r}")
    if int(metadata.get("kmeans_n_init", -1)) != 20:
        raise RuntimeError("KMeans n_init must be 20")
    if int(metadata.get("fit_max_rows", -1)) != 500000:
        raise RuntimeError("KMeans fit_max_rows must be 500000")
    if int(metadata.get("random_state", -1)) != 42:
        raise RuntimeError("KMeans random_state must be 42")
    require_columns(centres, ["macrostate", "center_M", "center_Psi"], "fixed K=6 centres")
    states = pd.to_numeric(centres["macrostate"], errors="raise").astype(int).to_numpy()
    if len(centres) != MACROSTATE_K or not np.array_equal(np.sort(states), np.arange(MACROSTATE_K)):
        raise RuntimeError("Fixed K=6 centres do not contain states 0--5 exactly once")


def load_sources(stage1_root: Path, include_confirmation: bool) -> Dict[str, Any]:
    coordinate_root = stage1_root / "dynamics" / "coordinate_analysis" / COORDINATE
    region_root = stage1_root / "dynamics" / "candidate_regions" / COORDINATE
    mesostate_root = stage1_root / "dynamics" / "fixed_k6_mesostates"
    manifest_path = stage1_root / "metadata" / "stage1_empirical_v3_manifest.json"
    metadata_path = mesostate_root / "fixed_k6_model_metadata.json"
    if not manifest_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Stage-1 manifest or fixed-K metadata is missing")

    sources: Dict[str, Any] = {
        "stage1_manifest": load_json(manifest_path),
        "kmeans_metadata": load_json(metadata_path),
        "field_train": read_table(coordinate_root / "A_train_publication_field_grid"),
        "field_val": read_table(coordinate_root / "A_val_publication_field_grid"),
        "occupancy": read_table(coordinate_root / "occupancy_geometry_summaries"),
        "global_contraction": read_table(coordinate_root / "global_field_contraction_summaries"),
        "training_regions": read_table(region_root / "training_flow_defined_convergence_regions"),
        "validation_frozen_region": read_table(region_root / "validation_frozen_training_convergence_region"),
        "region_reproducibility": read_table(region_root / "training_validation_convergence_region_reproducibility"),
        "radial_profiles": read_table(region_root / "convergence_radial_profiles"),
        "centres": read_table(mesostate_root / "fixed_k6_centers"),
        "fit_table": read_table(mesostate_root / "fixed_k6_fit_table"),
        "transition_counts_train": read_table(mesostate_root / "A_train_fixed_k6_transition_counts"),
        "transition_counts_val": read_table(mesostate_root / "A_val_fixed_k6_transition_counts"),
        "transition_train": read_table(mesostate_root / "A_train_fixed_k6_transition_matrix"),
        "transition_val": read_table(mesostate_root / "A_val_fixed_k6_transition_matrix"),
        "residence_runs_train": read_table(mesostate_root / "A_train_fixed_k6_residence_runs"),
        "residence_runs_val": read_table(mesostate_root / "A_val_fixed_k6_residence_runs"),
        "residence_summary_train": read_table(mesostate_root / "A_train_fixed_k6_residence_summary"),
        "residence_summary_val": read_table(mesostate_root / "A_val_fixed_k6_residence_summary"),
        "residence_curves_train": read_table(mesostate_root / "A_train_fixed_k6_residence_curves"),
        "residence_curves_val": read_table(mesostate_root / "A_val_fixed_k6_residence_curves"),
        "reproducibility": read_table(mesostate_root / "A_train_A_val_reproducibility_summary"),
    }
    if include_confirmation:
        sources.update({
            "field_confirm": read_table(coordinate_root / "B_confirm_publication_field_grid_output_only"),
            "confirmation_frozen_region": read_table(region_root / "confirmation_frozen_training_convergence_region_output_only"),
            "transition_counts_confirm": read_table(mesostate_root / "B_confirm_fixed_k6_transition_counts"),
            "transition_confirm": read_table(mesostate_root / "B_confirm_fixed_k6_transition_matrix"),
            "residence_runs_confirm": read_table(mesostate_root / "B_confirm_fixed_k6_residence_runs"),
            "residence_summary_confirm": read_table(mesostate_root / "B_confirm_fixed_k6_residence_summary"),
            "residence_curves_confirm": read_table(mesostate_root / "B_confirm_fixed_k6_residence_curves"),
        })
    validate_kmeans_contract(sources["kmeans_metadata"], sources["centres"])
    return sources


def longest_true_span(mask: np.ndarray, times: np.ndarray) -> int:
    best = 0
    current = 0
    previous: Optional[int] = None
    for flag, time_value in zip(mask.astype(bool), times.astype(int)):
        if flag and (previous is None or time_value == previous + 1):
            current += 1
        elif flag:
            current = 1
        else:
            current = 0
        best = max(best, current)
        previous = time_value if flag else None
    return int(best)


def km_tail_summary(curves: pd.DataFrame, residence: pd.DataFrame, minimum_consecutive: int = 3) -> Tuple[int, int]:
    descriptive = 0
    maximum_reliable = 0
    tau_by_state = {
        int(row.macrostate): int(row.rmst_tau)
        for row in residence[["macrostate", "rmst_tau"]].dropna().itertuples(index=False)
    }
    for state, group in curves.groupby("macrostate", sort=True):
        ordered = group.sort_values("residence_length")
        times = pd.to_numeric(ordered["residence_length"], errors="coerce").to_numpy(dtype=int)
        observed = pd.to_numeric(ordered["km_ccdf"], errors="coerce").to_numpy(dtype=float)
        geometric = pd.to_numeric(ordered["geometric_ccdf"], errors="coerce").to_numpy(dtype=float)
        tau = int(tau_by_state.get(int(state), 0))
        valid = (times >= 2) & (times <= tau) & np.isfinite(observed) & np.isfinite(geometric) & (geometric > 0)
        if np.any(valid):
            maximum_reliable = max(maximum_reliable, int(np.max(times[valid])))
        log_excess = np.log(np.maximum(observed, EPS) / np.maximum(geometric, EPS))
        positive = valid & (log_excess > 0)
        integrated = float(np.mean(np.maximum(log_excess[valid], 0.0))) if np.any(valid) else np.nan
        if longest_true_span(positive, times) >= minimum_consecutive and np.isfinite(integrated) and integrated > 0:
            descriptive += 1
    return int(descriptive), int(maximum_reliable)


def truthy_count(values: pd.Series) -> int:
    if pd.api.types.is_bool_dtype(values):
        return int(values.fillna(False).sum())
    normalized = values.astype(str).str.strip().str.lower()
    return int(normalized.isin({"1", "true", "yes", "y"}).sum())


def build_text_values(sources: Mapping[str, Any]) -> Dict[str, Any]:
    train_region = first_row(sources["training_regions"], "training convergence regions")
    val_region = first_row(sources["validation_frozen_region"], "validation frozen region")
    region_repro = first_row(sources["region_reproducibility"], "region reproducibility")
    occupancy = sources["occupancy"].set_index("split")
    contraction = sources["global_contraction"].set_index("split")
    radial = sources["radial_profiles"]
    first_shell = radial[(radial["split"] == "A_val") & (pd.to_numeric(radial["radial_bin"], errors="coerce") == 1)]
    first_shell_row = first_row(first_shell, "validation first radial shell")
    transition_counts_val = sources["transition_counts_val"].to_numpy(dtype=float)
    transition_val = sources["transition_val"].to_numpy(dtype=float)
    residence_val = sources["residence_summary_val"].copy()
    reproducibility = first_row(sources["reproducibility"], "training-validation reproducibility")

    diagonal = np.diag(transition_val)
    diagonal_dominant = int(np.sum(np.isclose(diagonal, np.max(transition_val, axis=1), atol=1e-12, rtol=0.0)))
    n_runs = pd.to_numeric(residence_val["n_runs"], errors="coerce")
    n_completed = pd.to_numeric(residence_val["n_completed_exits"], errors="coerce")
    n_censored = pd.to_numeric(residence_val["n_right_censored"], errors="coerce")
    total_runs = int(n_runs.sum())
    total_completed = int(n_completed.sum())
    total_censored = int(n_censored.sum())
    descriptive_tail = (
        pd.to_numeric(residence_val["observed_tail_probability_at_reference"], errors="coerce")
        > pd.to_numeric(residence_val["geometric_tail_probability_at_reference"], errors="coerce")
    )
    km_descriptive_count, km_max_reliable = km_tail_summary(sources["residence_curves_val"], residence_val)
    train_field = sources["field_train"].sort_values(["x_bin", "y_bin"], kind="mergesort")
    val_field = sources["field_val"].sort_values(["x_bin", "y_bin"], kind="mergesort")
    common_drift = train_field["drift_supported"].astype(bool).to_numpy() & val_field["drift_supported"].astype(bool).to_numpy()
    delta_m = pd.to_numeric(train_field["drift_M"], errors="coerce").to_numpy(dtype=float)[common_drift] - pd.to_numeric(val_field["drift_M"], errors="coerce").to_numpy(dtype=float)[common_drift]
    delta_psi = pd.to_numeric(train_field["drift_Psi"], errors="coerce").to_numpy(dtype=float)[common_drift] - pd.to_numeric(val_field["drift_Psi"], errors="coerce").to_numpy(dtype=float)[common_drift]
    drift_vector_rmse = float(np.sqrt(np.mean(delta_m * delta_m + delta_psi * delta_psi))) if delta_m.size else np.nan

    values: Dict[str, Any] = {
        "training_primary_convergence_flow_localization_M_shortest_interval_50_low": float(train_region["flow_localization_M_shortest_interval_50_low"]),
        "training_primary_convergence_flow_localization_M_shortest_interval_50_high": float(train_region["flow_localization_M_shortest_interval_50_high"]),
        "training_primary_convergence_flow_localization_Psi_shortest_interval_50_low": float(train_region["flow_localization_Psi_shortest_interval_50_low"]),
        "training_primary_convergence_flow_localization_Psi_shortest_interval_50_high": float(train_region["flow_localization_Psi_shortest_interval_50_high"]),
        "training_primary_convergence_convergence_center_M": float(train_region["convergence_center_M"]),
        "training_primary_convergence_convergence_center_Psi": float(train_region["convergence_center_Psi"]),
        "training_primary_convergence_local_fixed_point_M": float(train_region["local_fixed_point_M"]),
        "training_primary_convergence_local_fixed_point_Psi": float(train_region["local_fixed_point_Psi"]),
        "validation_global_drift_supported_cells": int(contraction.loc["A_val", "drift_supported_cells"]),
        "validation_global_weighted_negative_divergence_fraction_interior_only": float(contraction.loc["A_val", "weighted_negative_divergence_fraction_interior_only"]),
        "validation_global_weighted_mean_local_divergence_interior_only": float(contraction.loc["A_val", "weighted_mean_local_divergence_interior_only"]),
        "validation_frozen_primary_convergence_occupancy_mass_fraction": float(val_region["occupancy_mass_fraction"]),
        "validation_frozen_primary_convergence_flow_weighted_shell_fraction_inward": float(val_region["flow_weighted_shell_fraction_inward"]),
        "validation_frozen_primary_convergence_flow_core_to_shell_speed_ratio": float(val_region["flow_core_to_shell_speed_ratio"]),
        "train_validation_convergence_mask_jaccard": float(region_repro["mask_jaccard"]),
        "train_validation_convergence_convergence_center_distance": float(region_repro["convergence_center_distance"]),
        "validation_convergence_radial_first_shell_flow_weighted_fraction_inward": float(first_shell_row["flow_weighted_fraction_inward"]),
        "validation_convergence_radial_first_shell_flow_weighted_mean_inward_cosine": float(first_shell_row["flow_weighted_mean_inward_cosine"]),
        "validation_occupancy_Psi_shortest_interval_50_low": float(occupancy.loc["A_val", "Psi_shortest_interval_50_low"]),
        "validation_occupancy_Psi_shortest_interval_50_high": float(occupancy.loc["A_val", "Psi_shortest_interval_50_high"]),
        "validation_occupancy_M_outermost_one_bin_mass_fraction": float(occupancy.loc["A_val", "M_outermost_one_bin_mass_fraction"]),
        "macrostate_k": MACROSTATE_K,
        "validation_transition_count": int(transition_counts_val.sum()),
        "validation_diagonal_dominant_states": diagonal_dominant,
        "validation_mean_diagonal_probability": float(np.mean(diagonal)),
        "validation_diagonal_probability_range": f"{diagonal.min():.3f}--{diagonal.max():.3f}",
        "validation_residence_runs_total": total_runs,
        "validation_residence_completed_exits_total": total_completed,
        "validation_residence_right_censored_total": total_censored,
        "validation_residence_right_censoring_fraction": float(total_censored / max(total_runs, 1)),
        "validation_residence_rmst_tau_range": integer_range(residence_val["rmst_tau"]),
        "validation_restricted_mean_residence_lift_range": numeric_range(residence_val["restricted_mean_residence_lift"], 3),
        "residence_reference_length": int(pd.to_numeric(residence_val["reference_length"], errors="coerce").dropna().iloc[0]),
        "validation_reference_at_risk_range": integer_range(residence_val["reference_at_risk"]),
        "validation_tail_ratio_at_reference_range": numeric_range(residence_val["tail_ratio_at_reference"], 3),
        "validation_tail_excess_states_descriptive_at_reference": int(descriptive_tail.sum()),
        "validation_tail_excess_states_significant_bh": truthy_count(residence_val["tail_excess_significant_bh"]),
        "validation_km_tail_excess_states_descriptive": km_descriptive_count,
        "validation_km_tail_max_reliable_length": km_max_reliable,
        "train_validation_occupancy_js_divergence": float(reproducibility["occupancy_js_divergence"]),
        "train_validation_common_supported_drift_cells": int(reproducibility["common_supported_drift_cells"]),
        "train_validation_mean_local_drift_cosine": float(reproducibility["mean_local_drift_cosine"]),
        "train_validation_drift_component_rmse": drift_vector_rmse,
        "train_validation_drift_speed_pearson": float(reproducibility["drift_speed_pearson"]),
        "train_validation_residence_rmst_mean_abs_log_ratio": float(reproducibility["residence_rmst_mean_abs_log_ratio"]),
        "train_validation_residence_tail_ratio_mean_abs_log_ratio": float(reproducibility["residence_tail_ratio_mean_abs_log_ratio"]),
        "train_validation_residence_right_censoring_fraction_mean_abs_difference": float(reproducibility["residence_right_censoring_fraction_mean_abs_difference"]),
        "train_validation_transition_mean_row_total_variation": float(reproducibility["transition_mean_row_total_variation"]),
        "train_validation_transition_max_row_total_variation": float(reproducibility["transition_max_row_total_variation"]),
    }
    return values


def map_value(values: Mapping[str, Any], key_expression: str) -> str:
    keys = [key.strip() for key in key_expression.split("/")]
    rendered = [str(values.get(key, "")) for key in keys]
    return ", ".join(rendered)


def write_fill_map(values: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Results subsection value map",
        "",
        "The primary spatial quantities are flow-defined convergence regions. Occupancy marginal peaks and boundary mass are reported separately and must not be described as dynamical centres.",
        "",
        "Residence quantities are right-censoring-aware. Restricted means are Kaplan--Meier RMST values over state-specific reliable horizons; fixed-reference tail ratios use the Kaplan--Meier CCDF. Any BH count refers only to the Greenwood test at the fixed reference length.",
        "",
        "| Manuscript quantity | Output key | Value |",
        "|---|---|---:|",
    ]
    for label, key_expression in VALUE_MAP_ROWS:
        lines.append(f"| {label} | `{key_expression}` | {map_value(values, key_expression)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_inventory(stage1_root: Path, sources: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    table_bases = {
        "field_train": stage1_root / "dynamics" / "coordinate_analysis" / COORDINATE / "A_train_publication_field_grid",
        "field_val": stage1_root / "dynamics" / "coordinate_analysis" / COORDINATE / "A_val_publication_field_grid",
        "occupancy": stage1_root / "dynamics" / "coordinate_analysis" / COORDINATE / "occupancy_geometry_summaries",
        "global_contraction": stage1_root / "dynamics" / "coordinate_analysis" / COORDINATE / "global_field_contraction_summaries",
        "training_regions": stage1_root / "dynamics" / "candidate_regions" / COORDINATE / "training_flow_defined_convergence_regions",
        "validation_frozen_region": stage1_root / "dynamics" / "candidate_regions" / COORDINATE / "validation_frozen_training_convergence_region",
        "region_reproducibility": stage1_root / "dynamics" / "candidate_regions" / COORDINATE / "training_validation_convergence_region_reproducibility",
        "radial_profiles": stage1_root / "dynamics" / "candidate_regions" / COORDINATE / "convergence_radial_profiles",
        "centres": stage1_root / "dynamics" / "fixed_k6_mesostates" / "fixed_k6_centers",
        "fit_table": stage1_root / "dynamics" / "fixed_k6_mesostates" / "fixed_k6_fit_table",
        "transition_counts_train": stage1_root / "dynamics" / "fixed_k6_mesostates" / "A_train_fixed_k6_transition_counts",
        "transition_counts_val": stage1_root / "dynamics" / "fixed_k6_mesostates" / "A_val_fixed_k6_transition_counts",
        "transition_train": stage1_root / "dynamics" / "fixed_k6_mesostates" / "A_train_fixed_k6_transition_matrix",
        "transition_val": stage1_root / "dynamics" / "fixed_k6_mesostates" / "A_val_fixed_k6_transition_matrix",
        "residence_runs_train": stage1_root / "dynamics" / "fixed_k6_mesostates" / "A_train_fixed_k6_residence_runs",
        "residence_runs_val": stage1_root / "dynamics" / "fixed_k6_mesostates" / "A_val_fixed_k6_residence_runs",
        "residence_summary_train": stage1_root / "dynamics" / "fixed_k6_mesostates" / "A_train_fixed_k6_residence_summary",
        "residence_summary_val": stage1_root / "dynamics" / "fixed_k6_mesostates" / "A_val_fixed_k6_residence_summary",
        "residence_curves_train": stage1_root / "dynamics" / "fixed_k6_mesostates" / "A_train_fixed_k6_residence_curves",
        "residence_curves_val": stage1_root / "dynamics" / "fixed_k6_mesostates" / "A_val_fixed_k6_residence_curves",
        "reproducibility": stage1_root / "dynamics" / "fixed_k6_mesostates" / "A_train_A_val_reproducibility_summary",
    }
    for name, base in table_bases.items():
        path = table_path(base)
        frame = sources[name]
        rows.append({
            "name": name,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
        })
    return pd.DataFrame(rows)


def copy_publication_tables(sources: Mapping[str, Any], out_dir: Path) -> None:
    mapping = {
        "field_train": "field_grid_A_train",
        "field_val": "field_grid_A_val",
        "occupancy": "occupancy_geometry_summaries",
        "global_contraction": "global_field_contraction_summaries",
        "training_regions": "training_flow_defined_convergence_regions",
        "validation_frozen_region": "validation_frozen_training_convergence_region",
        "region_reproducibility": "training_validation_convergence_region_reproducibility",
        "radial_profiles": "convergence_radial_profiles",
        "centres": "macrostate_centers_fixed_k6",
        "fit_table": "macrostate_fit_table_fixed_k6",
        "transition_counts_train": "transition_counts_A_train_figure_consistent",
        "transition_counts_val": "transition_counts_A_val_figure_consistent",
        "transition_train": "transition_matrix_A_train_figure_consistent",
        "transition_val": "transition_matrix_A_val_figure_consistent",
        "residence_runs_train": "residence_runs_A_train_censoring_aware",
        "residence_runs_val": "residence_runs_A_val_censoring_aware",
        "residence_summary_train": "residence_summary_A_train_censoring_aware",
        "residence_summary_val": "residence_summary_A_val_censoring_aware",
        "residence_curves_train": "residence_kaplan_meier_curves_A_train",
        "residence_curves_val": "residence_kaplan_meier_curves_A_val",
        "reproducibility": "training_validation_reproducibility_metrics_figure_consistent",
    }
    for source_name, output_name in mapping.items():
        write_table(sources[source_name], out_dir / output_name)


def reference_audit(values: Mapping[str, Any], reference_path: Optional[Path], atol: float, rtol: float) -> pd.DataFrame:
    if reference_path is None:
        return pd.DataFrame(columns=["key", "observed", "reference", "absolute_difference", "matched"])
    reference = load_json(reference_path)
    rows = []
    for key, expected in reference.items():
        observed = values.get(key)
        if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
            difference = abs(float(observed) - float(expected))
            matched = bool(np.isclose(float(observed), float(expected), atol=atol, rtol=rtol))
        else:
            difference = np.nan
            matched = observed == expected
        rows.append({
            "key": key,
            "observed": observed,
            "reference": expected,
            "absolute_difference": difference,
            "matched": matched,
        })
    return pd.DataFrame(rows)


def quality_gates(sources: Mapping[str, Any], reference: pd.DataFrame, enforce_reference: bool) -> pd.DataFrame:
    metadata = sources["kmeans_metadata"]
    transition_val = sources["transition_val"].to_numpy(dtype=float)
    residence_val = sources["residence_summary_val"]
    gates = [
        ("primary_coordinate_is_M_Psi", sources["stage1_manifest"].get("coordinate_summary", {}).get("coordinate") == COORDINATE),
        ("macrostate_k_fixed_at_6", int(metadata.get("macrostate_k", -1)) == MACROSTATE_K and metadata.get("macrostate_k_rule") == "fixed a priori"),
        ("kmeans_fit_on_A_train", metadata.get("fit_split") == "A_train"),
        ("transition_matrix_is_6_by_6", transition_val.shape == (MACROSTATE_K, MACROSTATE_K)),
        ("transition_rows_normalized", bool(np.allclose(transition_val.sum(axis=1), 1.0, atol=1e-10, rtol=0.0))),
        ("right_censoring_fields_present", {"n_completed_exits", "n_right_censored", "rmst_tau", "tail_ratio_at_reference"}.issubset(residence_val.columns)),
        ("reference_values_match", bool((not enforce_reference) or reference.empty or reference["matched"].all())),
    ]
    return pd.DataFrame([{"gate": name, "passed": bool(passed)} for name, passed in gates])


def parse_args() -> argparse.Namespace:
    default_root = Path(os.environ.get("EDNET_KT4_OUTPUT_ROOT", "/data/datasets/KT4/outputs_KT4")) / "stage1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, default=Path(os.environ.get("EDNET_STAGE1_ROOT", str(default_root))))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--include-confirmation", action="store_true")
    parser.add_argument("--reference-values", type=Path, default=None)
    parser.add_argument("--reference-atol", type=float, default=1e-9)
    parser.add_argument("--reference-rtol", type=float, default=1e-9)
    parser.add_argument("--strict-reference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage1_root = resolve_stage1_root(args.stage1_root)
    out_dir = (args.out_dir or (stage1_root / "publication_empirical_statistics")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = load_sources(stage1_root, args.include_confirmation)
    values = build_text_values(sources)
    copy_publication_tables(sources, out_dir)

    reference_path = args.reference_values
    if reference_path is None:
        bundled = Path(__file__).with_name("archived_figure2_reference_values.json")
        reference_path = bundled if bundled.exists() else None
    reference = reference_audit(values, reference_path, args.reference_atol, args.reference_rtol)
    write_table(reference, out_dir / "archived_reference_value_match")
    if args.strict_reference and not reference.empty and not bool(reference["matched"].all()):
        failed = reference.loc[~reference["matched"], "key"].tolist()
        raise RuntimeError(f"Archived Figure 2 values do not match: {failed}")

    gates = quality_gates(sources, reference, args.strict_reference)
    write_table(gates, out_dir / "scientific_quality_gates")
    if not bool(gates["passed"].all()):
        failed = gates.loc[~gates["passed"], "gate"].tolist()
        raise RuntimeError(f"Scientific quality gates failed: {failed}")

    save_json(values, out_dir / "publication_results_text_values.json")
    write_table(pd.DataFrame([values]), out_dir / "publication_results_text_values")
    write_fill_map(values, out_dir / "publication_results_fill_map.md")

    inventory = source_inventory(stage1_root, sources)
    write_table(inventory, out_dir / "stage1_source_inventory")
    audit = {
        "script": Path(__file__).name,
        "stage1_root": stage1_root,
        "output_directory": out_dir,
        "primary_coordinate": COORDINATE,
        "macrostate_policy": {
            "k": MACROSTATE_K,
            "source": "Stage-1 fixed_k6_mesostates outputs",
            "refit_performed": False,
            "candidate_k_search_performed": False,
            "metadata": sources["kmeans_metadata"],
        },
        "field_policy": "Read authoritative Stage-1 field, convergence, occupancy, transition, and residence outputs without recomputation.",
        "reference_values": str(reference_path.resolve()) if reference_path is not None else None,
        "reference_match": bool(reference.empty or reference["matched"].all()),
        "confirmation_split_read": bool(args.include_confirmation),
        "source_files": inventory.to_dict(orient="records"),
    }
    save_json(audit, out_dir / "publication_empirical_statistics_audit.json")

    output_rows = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            output_rows.append({"file": path.name, "bytes": path.stat().st_size})
    write_table(pd.DataFrame(output_rows), out_dir / "output_inventory")
    print(f"Output directory: {out_dir}")
    print(f"Primary values: {out_dir / 'publication_results_text_values.json'}")
    print(f"Fill map: {out_dir / 'publication_results_fill_map.md'}")


if __name__ == "__main__":
    main()
