#!/usr/bin/env python3
"""Audit mechanism-family selection across predefined score contracts and a
complexity-aware Pareto criterion.

The formal-audit subcommand reconstructs the five paired-user bootstrap loss
components for every final Phase-1 family fit, verifies exact recovery of the
archived formal bootstrap score, replays the unchanged 1-SE + practical-
equivalence selection rule under predefined frozen-fit score contracts, and
computes performance- and complexity-Pareto diagnostics.

The finalize subcommand combines that frozen-fit audit with the independent
full equal-primary-component re-optimisation output.  Neither subcommand reads B_confirm.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

AUDIT_VERSION = "1.1.0"
EXPECTED_FORMAL_SCRIPT_SHA256 = "e5c023d85dc00c5ef8e89c12162b5dea8e0dfc205651f37ccacffd8e6e3a612e"
EXPECTED_FORMAL_PUBLIC_RELEASE_VERSION = "4.2.0"
EXPECTED_FORMAL_RUNTIME_DEFAULTS: Dict[str, Any] = {
    "random_state": 42,
    "grid_profile": "publication",
    "screening_train_users": 20000,
    "screening_max_candidates": 96,
    "full_train_top_k": 16,
    "validation_shortlist_k": 8,
    "local_refine_max_evals": 48,
    "deletion_exhaustive_max_combinations": 5000,
    "deletion_full_train_top_k": 32,
    "deletion_validation_shortlist_k": 16,
    "deletion_local_refine_max_evals": 96,
    "deletion_refine_starts": 5,
    "bootstrap_reps": 300,
    "equivalence_margin": 0.02,
    "bootstrap_engine": "optimized",
    "verify_optimized_bootstrap": True,
    "verify_bootstrap_reps": 2,
    "decision_bootstrap_seed_offset": 777,
    "sanity_penalty_weight": 0.25,
    "identity_regularization_weight": 0.05,
    "distribution_loss_max_rows": 200000,
    "signed_gain_quantile": 0.75,
}
DEFAULT_FORMAL_ROOT = Path(
    "/data/datasets/KT4/outputs_KT4/stage2_phase1_unified_minimality"
)
DEFAULT_SCORE_ROOT = Path(
    "/data/datasets/KT4/outputs_KT4/stage2_phase1_score_contract_robustness"
)

PRIMARY_COMPONENTS = (
    "one_step_mse_main_norm",
    "occupancy_js_MR_PsiA",
    "drift_local_rmse_loss_MR_PsiA",
    "drift_direction_loss_MR_PsiA",
    "drift_magnitude_loss_MR_PsiA",
)
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
FORMAL_WEIGHTS: Dict[str, float] = {
    "one_step_mse_main_norm": 0.10,
    "occupancy_js_MR_PsiA": 0.20,
    "drift_local_rmse_loss_MR_PsiA": 0.30,
    "drift_direction_loss_MR_PsiA": 0.20,
    "drift_magnitude_loss_MR_PsiA": 0.20,
}


def renormalise(weights: Mapping[str, float]) -> Dict[str, float]:
    out = {name: float(weights.get(name, 0.0)) for name in PRIMARY_COMPONENTS}
    if any((not math.isfinite(value) or value < 0.0) for value in out.values()):
        raise ValueError("Objective weights must be finite and non-negative.")
    total = float(sum(out.values()))
    if total <= 0.0:
        raise ValueError("At least one objective weight must be positive.")
    return {name: value / total for name, value in out.items()}


CONTRACTS: Dict[str, Dict[str, float]] = {
    "formal": renormalise(FORMAL_WEIGHTS),
    "equal_primary": renormalise({name: 1.0 for name in PRIMARY_COMPONENTS}),
    "omit_step": renormalise({**FORMAL_WEIGHTS, "one_step_mse_main_norm": 0.0}),
    "omit_js": renormalise({**FORMAL_WEIGHTS, "occupancy_js_MR_PsiA": 0.0}),
    "omit_local": renormalise({**FORMAL_WEIGHTS, "drift_local_rmse_loss_MR_PsiA": 0.0}),
    "omit_direction": renormalise({**FORMAL_WEIGHTS, "drift_direction_loss_MR_PsiA": 0.0}),
    "omit_magnitude": renormalise({**FORMAL_WEIGHTS, "drift_magnitude_loss_MR_PsiA": 0.0}),
}


def now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return json_safe(dataclasses.asdict(obj))
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
    return str(obj)


def stable_json_hash(obj: Any) -> str:
    payload = json.dumps(
        json_safe(obj),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(obj), handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    compression = "gzip" if path.name.endswith(".gz") else None
    frame.to_csv(tmp, index=False, compression=compression)
    os.replace(tmp, path)
    return path


def read_table(base_or_path: Path) -> pd.DataFrame:
    path = Path(base_or_path)
    candidates: List[Path]
    if path.suffix in {".csv", ".gz", ".parquet"} and path.exists():
        candidates = [path]
    else:
        candidates = [
            path.with_suffix(".parquet"),
            path.with_suffix(".csv.gz"),
            path.with_suffix(".csv"),
        ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".parquet":
            return pd.read_parquet(candidate)
        return pd.read_csv(candidate, low_memory=False)
    raise FileNotFoundError(f"Could not find table: {path}")


def import_formal_module(path: Path) -> Any:
    source = path.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Formal Phase-1 script not found: {source}")
    cache_dir = Path(tempfile.gettempdir()) / "ednet_kt4_numba_cache" / sha256_file(source)[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
    spec = importlib.util.spec_from_file_location(
        "formal_mechanism_family_ablation_for_component_audit",
        str(source),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import formal Phase-1 script: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def validate_formal_module(p1: Any) -> None:
    required = (
        "FamilySpec",
        "PredictionArrays",
        "FAMILY_BY_KEY",
        "ALL_PARAMS",
        "load_phase1_panels",
        "calibrate_from_A_train",
        "make_metric_cache",
        "BootstrapScorer",
        "simulate_for_family",
        "light_prediction",
        "objective_component_values",
        "js_divergence",
        "vector_corr",
        "drift_local_rmse_loss",
        "drift_magnitude_loss",
    )
    missing = [name for name in required if not hasattr(p1, name)]
    if missing:
        raise RuntimeError(f"Formal Phase-1 implementation is missing required names: {missing}")
    if set(p1.OBJECTIVE_WEIGHTS) != set(PRIMARY_COMPONENTS):
        raise RuntimeError("Formal Phase-1 primary component names do not match the audit contract.")


def reviewed_runtime_defaults(p1: Any) -> Dict[str, Any]:
    actual = {
        "random_state": int(p1.CONFIG_RANDOM_STATE),
        "grid_profile": str(p1.GRID_PROFILE),
        "screening_train_users": int(p1.SCREENING_TRAIN_USERS),
        "screening_max_candidates": int(p1.SCREENING_MAX_CANDIDATES_PER_FAMILY),
        "full_train_top_k": int(p1.FULL_TRAIN_TOP_K),
        "validation_shortlist_k": int(p1.VAL_SHORTLIST_K),
        "local_refine_max_evals": int(p1.LOCAL_REFINE_MAX_EVALS),
        "deletion_exhaustive_max_combinations": int(p1.DELETION_EXHAUSTIVE_MAX_COMBINATIONS),
        "deletion_full_train_top_k": int(p1.DELETION_FULL_TRAIN_TOP_K),
        "deletion_validation_shortlist_k": int(p1.DELETION_VAL_SHORTLIST_K),
        "deletion_local_refine_max_evals": int(p1.DELETION_LOCAL_REFINE_MAX_EVALS),
        "deletion_refine_starts": int(p1.DELETION_REFINE_STARTS),
        "bootstrap_reps": int(p1.BOOTSTRAP_REPS),
        "equivalence_margin": float(p1.PRACTICAL_EQ_MARGIN),
        "bootstrap_engine": str(p1.BOOTSTRAP_ENGINE),
        "verify_optimized_bootstrap": bool(p1.VERIFY_OPTIMIZED_BOOTSTRAP),
        "verify_bootstrap_reps": int(p1.VERIFY_BOOTSTRAP_REPS),
        "decision_bootstrap_seed_offset": int(p1.DECISION_BOOTSTRAP_SEED_OFFSET),
        "sanity_penalty_weight": float(p1.CONFIG_SANITY_PENALTY_WEIGHT),
        "identity_regularization_weight": float(p1.CONFIG_IDENTITY_REG_WEIGHT),
        "distribution_loss_max_rows": int(p1.CONFIG_DISTRIBUTION_LOSS_MAX_ROWS),
        "signed_gain_quantile": float(p1.CONFIG_SIGNED_GAIN_QUANTILE),
    }
    mismatches = {
        key: {"expected": EXPECTED_FORMAL_RUNTIME_DEFAULTS[key], "actual": actual[key]}
        for key in EXPECTED_FORMAL_RUNTIME_DEFAULTS
        if json_safe(actual[key]) != json_safe(EXPECTED_FORMAL_RUNTIME_DEFAULTS[key])
    }
    if mismatches:
        raise RuntimeError(
            "Runtime environment overrides the reviewed formal Phase-1 defaults: "
            f"{mismatches}. Set the formal defaults explicitly before running the audit."
        )
    if str(getattr(p1, "PUBLIC_RELEASE_VERSION", "")) != EXPECTED_FORMAL_PUBLIC_RELEASE_VERSION:
        raise RuntimeError(
            "Unexpected formal Phase-1 public release version: "
            f"{getattr(p1, 'PUBLIC_RELEASE_VERSION', None)!r}."
        )
    return actual


def _formal_paths(root: Path) -> Dict[str, Path]:
    return {
        "manifest": root / "metadata" / "minimality_experiment_manifest.json",
        "handoff": root / "metadata" / "phase1_minimal_mechanism_handoff.json",
        "results": root / "tables" / "model_family_results.csv",
        "bootstrap": root / "tables" / "model_family_bootstrap_scores.csv.gz",
        "margin": root / "tables" / "equivalence_margin_sensitivity.csv",
    }


def validate_formal_inputs(
    formal_root: Path,
    formal_script: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame, pd.DataFrame, Dict[str, Path]]:
    paths = _formal_paths(formal_root)
    for name in ("manifest", "handoff", "results", "bootstrap"):
        if not paths[name].exists():
            raise FileNotFoundError(f"Formal {name} output not found: {paths[name]}")
    manifest = load_json(paths["manifest"])
    handoff = load_json(paths["handoff"])
    results = pd.read_csv(paths["results"], low_memory=False)
    archived_boot = pd.read_csv(paths["bootstrap"], low_memory=False)

    actual_script_sha = sha256_file(formal_script)
    if actual_script_sha != EXPECTED_FORMAL_SCRIPT_SHA256:
        raise RuntimeError(
            "The supplied formal Phase-1 script differs from the reviewed implementation: "
            f"expected {EXPECTED_FORMAL_SCRIPT_SHA256}, found {actual_script_sha}."
        )
    recorded_script_sha = str(manifest.get("script_sha256", "") or "")
    if recorded_script_sha and recorded_script_sha != actual_script_sha:
        raise RuntimeError(
            "Formal script checksum does not match the archived minimality manifest: "
            f"{recorded_script_sha} != {actual_script_sha}"
        )
    if "B_confirm not read" not in str(manifest.get("experiment_scope", "")):
        raise RuntimeError("Formal manifest does not preserve the A_train/A_val-only boundary.")
    formal_weights = renormalise(manifest.get("objective_weights", FORMAL_WEIGHTS))
    if any(abs(formal_weights[name] - CONTRACTS["formal"][name]) > 1e-12 for name in PRIMARY_COMPONENTS):
        raise RuntimeError(f"Unexpected formal objective weights: {formal_weights}")
    required_result_columns = {
        "family_key",
        "Model family",
        "Free mechanism parameters",
        "Free parameter names",
        *PARAMETER_COLUMNS,
    }
    missing = required_result_columns.difference(results.columns)
    if missing:
        raise RuntimeError(f"Formal family-results table is missing columns: {sorted(missing)}")
    required_boot_columns = {"bootstrap_rep", "family_key", "primary_score"}
    missing_boot = required_boot_columns.difference(archived_boot.columns)
    if missing_boot:
        raise RuntimeError(f"Formal bootstrap table is missing columns: {sorted(missing_boot)}")
    if set(results["family_key"].astype(str)) != set(archived_boot["family_key"].astype(str)):
        raise RuntimeError("Formal results and bootstrap tables contain different family sets.")
    if results["family_key"].duplicated().any():
        raise RuntimeError("Formal results contain duplicate family keys.")
    return manifest, handoff, results, archived_boot, paths


def _parse_free_params(value: Any) -> Tuple[str, ...]:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return tuple()
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return tuple()
    return tuple(part.strip() for part in text.split(";") if part.strip())


def reconstruct_predictions(
    p1: Any,
    results: pd.DataFrame,
    val_cache: Any,
    calibration: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    predictions: Dict[str, Any] = {}
    specs: Dict[str, Any] = {}
    for _, row in results.iterrows():
        key = str(row["family_key"])
        free_params = _parse_free_params(row.get("Free parameter names", ""))
        if key in p1.FAMILY_BY_KEY and tuple(p1.FAMILY_BY_KEY[key].free_params) == free_params:
            spec = p1.FAMILY_BY_KEY[key]
        else:
            role = str(row.get("Role", "sensitivity_candidate"))
            spec = p1.FamilySpec(
                key=key,
                label=str(row.get("Model family", key)),
                free_params=free_params,
                description="Reconstructed from the final formal Phase-1 results table.",
                role=role,
            )
        params = {name: float(row[name]) for name in PARAMETER_COLUMNS}
        p1.assert_candidate_on_family_grid(spec, params)
        simulation = p1.simulate_for_family(spec, params, val_cache, calibration)
        predictions[key] = p1.light_prediction(simulation)
        specs[key] = spec
    return predictions, specs


def component_scores_optimized(
    p1: Any,
    scorer: Any,
    predictions: Mapping[str, Any],
    reps: int,
    seed: int,
) -> pd.DataFrame:
    base = scorer.base
    bank = scorer.bank_for(reps, seed)
    observed = scorer.observed_for(reps, seed, bank)
    counts = bank.multiplicities
    shape = (len(p1.GRID_BINS_SIGNED) - 1, len(p1.GRID_BINS_SIGNED) - 1)
    rows: List[Dict[str, Any]] = []

    for family_key, prediction in predictions.items():
        stats = scorer.stats_for(family_key, prediction)
        mse_num = np.asarray(counts @ stats.error_by_user, dtype=np.float64)
        mse = np.minimum(mse_num / np.maximum(observed.denominator, p1.EPS), 1.0)

        model_occ_raw = p1._aggregate_sparse(counts, stats.occupancy_by_user_cell)
        model_occ = model_occ_raw / np.maximum(model_occ_raw.sum(axis=1, keepdims=True), p1.EPS)
        model_sx = p1._aggregate_sparse(counts, stats.drift_sx_by_user_cell)
        model_sy = p1._aggregate_sparse(counts, stats.drift_sy_by_user_cell)
        model_u = model_sx / np.maximum(observed.drift_weight, p1.EPS)
        model_v = model_sy / np.maximum(observed.drift_weight, p1.EPS)
        mask_flat = base.observed_drift_mask & stats.drift_mask
        mask = mask_flat.reshape(shape)

        for rep in range(reps):
            obs_field = {
                "u": observed.drift_u[rep].reshape(shape),
                "v": observed.drift_v[rep].reshape(shape),
                "weight": observed.drift_weight[rep].reshape(shape),
                "mask": mask,
            }
            model_field = {
                "u": model_u[rep].reshape(shape),
                "v": model_v[rep].reshape(shape),
                "weight": observed.drift_weight[rep].reshape(shape),
                "mask": mask,
            }
            corr = p1.vector_corr(
                obs_field["u"],
                obs_field["v"],
                model_field["u"],
                model_field["v"],
                mask,
            )
            raw_metrics = {
                "one_step_mse_main_norm": float(mse[rep]),
                "occupancy_js_MR_PsiA": float(p1.js_divergence(
                    observed.occupancy[rep].reshape(shape) + p1.EPS,
                    model_occ[rep].reshape(shape) + p1.EPS,
                )),
                "drift_local_rmse_loss_MR_PsiA": float(
                    p1.drift_local_rmse_loss(obs_field, model_field, mask)
                ),
                "drift_direction_loss_MR_PsiA": float(
                    1.0 if not np.isfinite(corr) else 0.5 * (1.0 - corr)
                ),
                "drift_magnitude_loss_MR_PsiA": float(
                    p1.drift_magnitude_loss(obs_field, model_field, mask)
                ),
                "phase_loss_max_qdist": 0.0,
                "coverage_loss_max_qdist": 0.0,
            }
            bounded = p1.objective_component_values(raw_metrics)
            row: Dict[str, Any] = {
                "bootstrap_rep": int(rep),
                "family_key": family_key,
                "supported_drift_cells": int(mask.sum()),
            }
            row.update({name: float(bounded[name]) for name in PRIMARY_COMPONENTS})
            rows.append(row)
    return pd.DataFrame(rows)


def weighted_score(frame: pd.DataFrame, weights: Mapping[str, float]) -> np.ndarray:
    w = renormalise(weights)
    score = np.zeros(len(frame), dtype=float)
    for name in PRIMARY_COMPONENTS:
        score += w[name] * pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    return np.clip(score, 0.0, 1.0)


def replay_selection(
    component_boot: pd.DataFrame,
    family_meta: pd.DataFrame,
    contract_name: str,
    weights: Mapping[str, float],
    margin: float,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    data = component_boot[["bootstrap_rep", "family_key", *PRIMARY_COMPONENTS]].copy()
    data["score"] = weighted_score(data, weights)
    summary = (
        data.groupby("family_key", sort=False)["score"]
        .agg([("bootstrap_mean", "mean"), ("bootstrap_sd_used_as_one_se", "std")])
        .reset_index()
    )
    summary["bootstrap_ci95_lower"] = summary["family_key"].map(
        data.groupby("family_key")["score"].quantile(0.025)
    )
    summary["bootstrap_ci95_upper"] = summary["family_key"].map(
        data.groupby("family_key")["score"].quantile(0.975)
    )
    best_row = summary.sort_values(["bootstrap_mean", "family_key"], kind="mergesort").iloc[0]
    best_key = str(best_row["family_key"])
    best_sd = float(best_row["bootstrap_sd_used_as_one_se"])
    one_se_threshold = float(best_row["bootstrap_mean"] + best_sd)
    best_scores = data.loc[data["family_key"] == best_key, ["bootstrap_rep", "score"]].rename(
        columns={"score": "best_score"}
    )

    difference_rows: List[Dict[str, Any]] = []
    for family_key, group in data.groupby("family_key", sort=False):
        merged = group[["bootstrap_rep", "score"]].merge(best_scores, on="bootstrap_rep", how="inner")
        diff = merged["score"].to_numpy(dtype=float) - merged["best_score"].to_numpy(dtype=float)
        difference_rows.append({
            "family_key": str(family_key),
            "difference_to_best_mean": float(np.mean(diff)),
            "difference_to_best_ci95_lower": float(np.quantile(diff, 0.025)),
            "difference_to_best_ci95_upper": float(np.quantile(diff, 0.975)),
            "practically_equivalent_to_best": bool(np.quantile(diff, 0.975) <= margin),
        })
    details = summary.merge(pd.DataFrame(difference_rows), on="family_key", how="left")
    details = details.merge(
        family_meta[["family_key", "Model family", "Free mechanism parameters"]],
        on="family_key",
        how="left",
    )
    details["within_one_standard_error_of_best"] = details["bootstrap_mean"] <= one_se_threshold
    details["eligible"] = (
        details["within_one_standard_error_of_best"]
        & details["practically_equivalent_to_best"]
    )
    eligible = details[details["eligible"]].copy()
    if eligible.empty:
        selected_key = best_key
    else:
        selected_key = str(
            eligible.sort_values(
                ["Free mechanism parameters", "bootstrap_mean", "family_key"],
                kind="mergesort",
            ).iloc[0]["family_key"]
        )
    selected_row = details.loc[details["family_key"] == selected_key].iloc[0]
    details["contract"] = contract_name
    details["selected"] = details["family_key"] == selected_key
    result = {
        "contract": contract_name,
        **{f"weight_{name}": renormalise(weights)[name] for name in PRIMARY_COMPONENTS},
        "best_family_key": best_key,
        "best_bootstrap_mean": float(best_row["bootstrap_mean"]),
        "best_bootstrap_sd_used_as_one_se": best_sd,
        "one_se_threshold": one_se_threshold,
        "selected_family_key": selected_key,
        "selected_family_label": str(selected_row["Model family"]),
        "selected_parameter_count": int(selected_row["Free mechanism parameters"]),
        "selected_difference_to_best_mean": float(selected_row["difference_to_best_mean"]),
        "selected_difference_to_best_ci95_upper": float(selected_row["difference_to_best_ci95_upper"]),
        "selected_within_one_se": bool(selected_row["within_one_standard_error_of_best"]),
        "selected_practically_equivalent": bool(selected_row["practically_equivalent_to_best"]),
        "practical_equivalence_margin": float(margin),
        "fit_status": "formal-weight-fitted family representatives; no parameter re-optimisation",
    }
    return result, details


def dominates(
    candidate_losses: np.ndarray,
    target_losses: np.ndarray,
    candidate_k: int,
    target_k: int,
    *,
    include_complexity: bool,
    tolerance: float,
) -> bool:
    losses_no_worse = bool(np.all(candidate_losses <= target_losses + tolerance))
    if not losses_no_worse:
        return False
    losses_strict = bool(np.any(candidate_losses < target_losses - tolerance))
    if not include_complexity:
        return losses_strict
    complexity_no_worse = candidate_k <= target_k
    complexity_strict = candidate_k < target_k
    # Complexity is the sixth minimisation objective.  A lower-complexity family
    # with numerically equal component losses therefore dominates.
    return bool(complexity_no_worse and (losses_strict or complexity_strict))


def pareto_tables(
    component_boot: pd.DataFrame,
    family_meta: pd.DataFrame,
    selected_family: str,
    tolerance: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    means = component_boot.groupby("family_key", sort=False)[list(PRIMARY_COMPONENTS)].mean().reset_index()
    means = means.merge(
        family_meta[["family_key", "Model family", "Free mechanism parameters"]],
        on="family_key",
        how="left",
    )
    rows: List[Dict[str, Any]] = []
    for _, target in means.iterrows():
        target_key = str(target["family_key"])
        target_losses = target[list(PRIMARY_COMPONENTS)].to_numpy(dtype=float)
        target_k = int(target["Free mechanism parameters"])
        performance_dominators: List[str] = []
        complexity_dominators: List[str] = []
        for _, candidate in means.iterrows():
            candidate_key = str(candidate["family_key"])
            if candidate_key == target_key:
                continue
            candidate_losses = candidate[list(PRIMARY_COMPONENTS)].to_numpy(dtype=float)
            candidate_k = int(candidate["Free mechanism parameters"])
            if dominates(
                candidate_losses,
                target_losses,
                candidate_k,
                target_k,
                include_complexity=False,
                tolerance=tolerance,
            ):
                performance_dominators.append(candidate_key)
            if dominates(
                candidate_losses,
                target_losses,
                candidate_k,
                target_k,
                include_complexity=True,
                tolerance=tolerance,
            ):
                complexity_dominators.append(candidate_key)
        row = target.to_dict()
        row.update({
            "performance_pareto_front": not performance_dominators,
            "performance_dominators": ";".join(sorted(performance_dominators)),
            "complexity_pareto_front": not complexity_dominators,
            "complexity_dominators": ";".join(sorted(complexity_dominators)),
            "pareto_tolerance": float(tolerance),
        })
        rows.append(row)
    mean_table = pd.DataFrame(rows)

    selected_meta = family_meta.loc[family_meta["family_key"] == selected_family]
    if selected_meta.empty:
        raise RuntimeError(f"Selected formal family missing from metadata: {selected_family}")
    selected_k = int(selected_meta.iloc[0]["Free mechanism parameters"])
    reps = sorted(component_boot["bootstrap_rep"].unique().tolist())
    rep_rows: List[Dict[str, Any]] = []
    for rep in reps:
        rep_data = component_boot[component_boot["bootstrap_rep"] == rep].set_index("family_key")
        target_losses = rep_data.loc[selected_family, list(PRIMARY_COMPONENTS)].to_numpy(dtype=float)
        any_not_more_complex = False
        any_strictly_simpler = False
        dominators_not_more_complex: List[str] = []
        dominators_strictly_simpler: List[str] = []
        for family_key, meta_row in family_meta.set_index("family_key").iterrows():
            family_key = str(family_key)
            if family_key == selected_family or family_key not in rep_data.index:
                continue
            candidate_k = int(meta_row["Free mechanism parameters"])
            candidate_losses = rep_data.loc[family_key, list(PRIMARY_COMPONENTS)].to_numpy(dtype=float)
            is_dom = dominates(
                candidate_losses,
                target_losses,
                candidate_k,
                selected_k,
                include_complexity=True,
                tolerance=tolerance,
            )
            if is_dom and candidate_k <= selected_k:
                any_not_more_complex = True
                dominators_not_more_complex.append(family_key)
            if is_dom and candidate_k < selected_k:
                any_strictly_simpler = True
                dominators_strictly_simpler.append(family_key)
        rep_rows.append({
            "bootstrap_rep": int(rep),
            "selected_family_key": selected_family,
            "selected_parameter_count": selected_k,
            "any_not_more_complex_family_dominates": any_not_more_complex,
            "any_strictly_simpler_family_dominates": any_strictly_simpler,
            "not_more_complex_dominators": ";".join(sorted(dominators_not_more_complex)),
            "strictly_simpler_dominators": ";".join(sorted(dominators_strictly_simpler)),
        })
    rep_table = pd.DataFrame(rep_rows)
    selected_mean = mean_table.loc[mean_table["family_key"] == selected_family].iloc[0]
    summary = {
        "selected_family_key": selected_family,
        "selected_parameter_count": selected_k,
        "selected_on_performance_pareto_front": bool(selected_mean["performance_pareto_front"]),
        "selected_on_complexity_pareto_front": bool(selected_mean["complexity_pareto_front"]),
        "selected_performance_dominators": str(selected_mean["performance_dominators"]),
        "selected_complexity_dominators": str(selected_mean["complexity_dominators"]),
        "bootstrap_frequency_any_not_more_complex_dominator": float(
            rep_table["any_not_more_complex_family_dominates"].mean()
        ),
        "bootstrap_frequency_any_strictly_simpler_dominator": float(
            rep_table["any_strictly_simpler_family_dominates"].mean()
        ),
        "pareto_definition": (
            "five loss components for performance Pareto; parameter count plus the five "
            "loss components for complexity Pareto; all objectives are minimised"
        ),
        "pareto_tolerance": float(tolerance),
        "fit_status": "formal-weight-fitted family representatives",
    }
    return mean_table, rep_table, summary


def run_formal_audit(args: argparse.Namespace) -> None:
    started = time.time()
    formal_root = args.formal_root.resolve()
    formal_script = args.formal_script.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if args.overwrite:
            import shutil
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"Audit output directory is not empty: {output_root}")
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    table_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    manifest, handoff, results, archived_boot, paths = validate_formal_inputs(
        formal_root,
        formal_script,
    )
    p1 = import_formal_module(formal_script)
    validate_formal_module(p1)
    runtime_defaults = reviewed_runtime_defaults(p1)

    # Keep the formal contract active throughout reconstruction.
    p1.OBJECTIVE_WEIGHTS.clear()
    p1.OBJECTIVE_WEIGHTS.update(CONTRACTS["formal"])
    # Reconstruct the exact formal search-domain contract.  This matters for
    # archived compact smoke runs and guards against silently validating a
    # parameter vector against a different grid profile.
    recorded_grid = manifest.get("search_grid") or manifest.get("grids")
    if recorded_grid:
        p1.GRID = {name: [float(value) for value in values] for name, values in recorded_grid.items()}
    p1.GRID_PROFILE = str(manifest.get("grid_profile", getattr(p1, "GRID_PROFILE", "publication")))
    p1.BOOTSTRAP_REPS = int(manifest.get("bootstrap_reps", getattr(p1, "BOOTSTRAP_REPS", 300)))
    p1.PRACTICAL_EQ_MARGIN = float(manifest.get("practical_equivalence_margin", getattr(p1, "PRACTICAL_EQ_MARGIN", 0.02)))
    if hasattr(p1, "_EVALUATION_CACHE"):
        p1._EVALUATION_CACHE.clear()

    stage1_root = Path(args.stage1_root).resolve() if args.stage1_root else Path(manifest["stage1_root"]).resolve()
    train, val, eta, load_manifest = p1.load_phase1_panels(stage1_root)
    panel_contract = p1.development_panel_contract(train, val)
    recorded_panel_contract = manifest.get("development_panel_contract", {})
    if recorded_panel_contract and stable_json_hash(panel_contract) != stable_json_hash(recorded_panel_contract):
        raise RuntimeError(
            "Reconstructed A_train/A_val panel fingerprints differ from the formal manifest."
        )
    kmeans_contract = p1.audit_stage1_kmeans_contract(stage1_root, allow_missing=False)
    recorded_kmeans_contract = manifest.get("stage1_fixed_k6_contract", {})
    if recorded_kmeans_contract and stable_json_hash(kmeans_contract) != stable_json_hash(recorded_kmeans_contract):
        raise RuntimeError(
            "Current Stage-1 fixed-K contract differs from the formal minimality manifest."
        )
    calibration = p1.calibrate_from_A_train(
        train,
        eta,
        p1.TAU_RESPONSE_DAYS,
        p1.TAU_ACTIVITY_DAYS,
    )
    recorded_calibration = dict(manifest.get("calibration", {}))
    actual_calibration = dataclasses.asdict(calibration)
    for name, value in recorded_calibration.items():
        if name in actual_calibration and value is not None:
            if not np.isclose(float(actual_calibration[name]), float(value), atol=1e-12, rtol=0.0):
                raise RuntimeError(
                    f"Reconstructed A_train calibration differs from the formal manifest for {name}."
                )
    val_cache = p1.make_metric_cache(val, "A_val_full")
    scorer = p1.BootstrapScorer(val_cache)
    del train, val

    predictions, specs = reconstruct_predictions(p1, results, val_cache, calibration)
    reps = int(manifest.get("bootstrap_reps", archived_boot["bootstrap_rep"].nunique()))
    seed = int(manifest.get("decision_bootstrap_seed", 42 + 777))
    component_boot = component_scores_optimized(p1, scorer, predictions, reps, seed)
    component_boot["formal_score_reconstructed"] = weighted_score(
        component_boot,
        CONTRACTS["formal"],
    )
    merged = component_boot.merge(
        archived_boot.rename(columns={"primary_score": "formal_score_archived"}),
        on=["bootstrap_rep", "family_key"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise RuntimeError("Reconstructed component bootstrap and archived score table do not align one-to-one.")
    merged["absolute_reconstruction_difference"] = (
        merged["formal_score_reconstructed"] - merged["formal_score_archived"]
    ).abs()
    max_difference = float(merged["absolute_reconstruction_difference"].max())
    if max_difference > args.reconstruction_tolerance:
        raise RuntimeError(
            "Formal score reconstruction failed: "
            f"maximum absolute difference {max_difference:.3e} exceeds "
            f"{args.reconstruction_tolerance:.3e}."
        )

    family_meta = results[["family_key", "Model family", "Free mechanism parameters"]].copy()
    contract_rows: List[Dict[str, Any]] = []
    detail_frames: List[pd.DataFrame] = []
    for contract_name, weights in CONTRACTS.items():
        summary, details = replay_selection(
            component_boot,
            family_meta,
            contract_name,
            weights,
            float(manifest.get("practical_equivalence_margin", 0.02)),
        )
        contract_rows.append(summary)
        detail_frames.append(details)
    contract_summary = pd.DataFrame(contract_rows)
    contract_details = pd.concat(detail_frames, ignore_index=True, sort=False)

    formal_final_family = str(handoff.get("final_family_key", ""))
    formal_contract_selected = str(
        contract_summary.loc[contract_summary["contract"] == "formal", "selected_family_key"].iloc[0]
    )
    if formal_contract_selected != formal_final_family:
        raise RuntimeError(
            "Formal contract replay did not reproduce the archived final family: "
            f"{formal_contract_selected} != {formal_final_family}"
        )

    pareto_mean, pareto_reps, pareto_summary = pareto_tables(
        component_boot,
        family_meta,
        formal_final_family,
        args.pareto_tolerance,
    )

    write_csv(component_boot, table_root / "component_bootstrap_losses.csv.gz")
    write_csv(
        merged[[
            "bootstrap_rep",
            "family_key",
            "formal_score_reconstructed",
            "formal_score_archived",
            "absolute_reconstruction_difference",
        ]],
        table_root / "formal_score_reconstruction_audit.csv.gz",
    )
    write_csv(contract_summary, table_root / "frozen_fit_score_contract_selection_summary.csv")
    write_csv(contract_details, table_root / "frozen_fit_score_contract_family_details.csv.gz")
    write_csv(pareto_mean, table_root / "frozen_fit_pareto_mean_components.csv")
    write_csv(pareto_reps, table_root / "selected_family_pareto_bootstrap.csv.gz")
    save_json(pareto_summary, metadata_root / "selected_family_pareto_summary.json")

    formal_runtime_contract = {
        **runtime_defaults,
        "public_release_version": str(p1.PUBLIC_RELEASE_VERSION),
        "stage1_root": str(stage1_root),
        "stage1_fixed_k6_contract": kmeans_contract,
        "development_panel_contract": panel_contract,
        "grid_contract": p1.validate_grid_contract(),
        "search_grid": p1.GRID,
        "pilot_anchors": p1.PILOT_ANCHORS,
        "families": [dataclasses.asdict(family) for family in p1.FAMILIES],
        "fixed_nuisance_scales": dict(p1.FIXED_NUISANCE),
        "objective_sanity_limits": dict(p1.OBJECTIVE_SANITY_LIMITS),
        "calibration": actual_calibration,
        "decision_bootstrap_seed": int(seed),
        "margin_sensitivity_values": [float(p1.PRACTICAL_EQ_MARGIN)],
    }

    formal_audit_manifest = {
        "created_at": now_string(),
        "audit_version": AUDIT_VERSION,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "formal_script": str(formal_script),
        "formal_script_sha256": sha256_file(formal_script),
        "formal_runtime_contract": formal_runtime_contract,
        "formal_output_root": str(formal_root),
        "formal_manifest": str(paths["manifest"]),
        "formal_manifest_sha256": sha256_file(paths["manifest"]),
        "formal_handoff": str(paths["handoff"]),
        "formal_handoff_sha256": sha256_file(paths["handoff"]),
        "stage1_root": str(stage1_root),
        "development_data_load_manifest": load_manifest,
        "B_confirm_read": False,
        "post_hoc": True,
        "analysis_scope": (
            "frozen-fit component reweighting and complexity-Pareto audit; "
            "family parameters remain those selected under the formal objective"
        ),
        "contracts": CONTRACTS,
        "practical_equivalence_margin": float(manifest.get("practical_equivalence_margin", 0.02)),
        "bootstrap_reps": reps,
        "decision_bootstrap_seed": seed,
        "one_se_implementation": (
            "best-family standard deviation across paired-user bootstrap scores, "
            "matching the formal implementation"
        ),
        "formal_score_reconstruction": {
            "maximum_absolute_difference": max_difference,
            "tolerance": args.reconstruction_tolerance,
            "passed": True,
        },
        "formal_final_family": formal_final_family,
        "frozen_fit_contract_results": contract_summary.to_dict(orient="records"),
        "pareto_summary": pareto_summary,
        "existing_analysis_nonduplication": {
            "equivalence_margin_sensitivity_recomputed": False,
            "formal_weight_parameter_deletion_result_recomputed": False,
            "new_elements": [
                "primary-component paired bootstrap",
                "objective-component reweighting",
                "six-objective complexity-Pareto audit",
            ],
        },
        "guardrails": {
            "formal_outputs_modified": False,
            "formal_family_or_parameters_replaced": False,
            "phase2_or_phase3_invoked": False,
            "B_confirm_read": False,
        },
        "elapsed_seconds": float(time.time() - started),
    }
    audit_manifest_path = metadata_root / "formal_score_contract_audit_manifest.json"
    save_json(formal_audit_manifest, audit_manifest_path)
    save_json(
        {
            "manifest_path": str(audit_manifest_path),
            "manifest_sha256": sha256_file(audit_manifest_path),
        },
        metadata_root / "formal_score_contract_audit_manifest.sha256.json",
    )
    print("[score-contract-audit] formal audit completed")
    print(f"[score-contract-audit] exact reconstruction max diff: {max_difference:.3e}")
    print(f"[score-contract-audit] formal final family: {formal_final_family}")
    print(f"[score-contract-audit] outputs: {output_root}")


def compare_equal_rerun_contract(
    formal_runtime: Mapping[str, Any],
    equal_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    mismatches: List[Dict[str, Any]] = []

    def check_scalar(formal_key: str, equal_key: Optional[str] = None) -> None:
        key = equal_key or formal_key
        formal_value = formal_runtime.get(formal_key)
        equal_value = equal_manifest.get(key)
        if json_safe(formal_value) != json_safe(equal_value):
            mismatches.append({
                "field": key,
                "formal": formal_value,
                "equal": equal_value,
            })

    for key in (
        "public_release_version",
        "stage1_root",
        "random_state",
        "grid_profile",
        "bootstrap_reps",
        "decision_bootstrap_seed_offset",
        "decision_bootstrap_seed",
        "equivalence_margin",
        "bootstrap_engine",
        "verify_optimized_bootstrap",
        "verify_bootstrap_reps",
        "sanity_penalty_weight",
        "identity_regularization_weight",
        "distribution_loss_max_rows",
        "signed_gain_quantile",
    ):
        check_scalar(key, "formal_public_release_version" if key == "public_release_version" else None)

    formal_budgets = {
        "screening_train_users": formal_runtime.get("screening_train_users"),
        "screening_max_candidates": formal_runtime.get("screening_max_candidates"),
        "full_train_top_k": formal_runtime.get("full_train_top_k"),
        "validation_shortlist_k": formal_runtime.get("validation_shortlist_k"),
        "local_refine_max_evals": formal_runtime.get("local_refine_max_evals"),
        "deletion_exhaustive_max_combinations": formal_runtime.get("deletion_exhaustive_max_combinations"),
        "deletion_full_train_top_k": formal_runtime.get("deletion_full_train_top_k"),
        "deletion_validation_shortlist_k": formal_runtime.get("deletion_validation_shortlist_k"),
        "deletion_local_refine_max_evals": formal_runtime.get("deletion_local_refine_max_evals"),
        "deletion_refine_starts": formal_runtime.get("deletion_refine_starts"),
    }
    equal_budgets = dict(equal_manifest.get("search_budgets", {}))
    if stable_json_hash(formal_budgets) != stable_json_hash(equal_budgets):
        mismatches.append({
            "field": "search_budgets",
            "formal": formal_budgets,
            "equal": equal_budgets,
        })

    for key in (
        "stage1_fixed_k6_contract",
        "development_panel_contract",
        "grid_contract",
        "search_grid",
        "pilot_anchors",
        "families",
        "fixed_nuisance_scales",
        "objective_sanity_limits",
        "calibration",
    ):
        formal_value = formal_runtime.get(key)
        equal_value = equal_manifest.get(key)
        if stable_json_hash(formal_value) != stable_json_hash(equal_value):
            mismatches.append({
                "field": key,
                "formal_hash": stable_json_hash(formal_value),
                "equal_hash": stable_json_hash(equal_value),
            })

    equal_primarys = renormalise(equal_manifest.get("objective_weights", {}))
    if any(abs(equal_primarys[name] - CONTRACTS["equal_primary"][name]) > 1e-12 for name in PRIMARY_COMPONENTS):
        mismatches.append({
            "field": "objective_weights",
            "expected": CONTRACTS["equal_primary"],
            "equal": equal_primarys,
        })
    if str(equal_manifest.get("objective_contract", "")) != "equal_primary":
        mismatches.append({
            "field": "objective_contract",
            "expected": "equal_primary",
            "equal": equal_manifest.get("objective_contract"),
        })
    if list(equal_manifest.get("margin_sensitivity_values", [])) != [float(formal_runtime["equivalence_margin"])]:
        mismatches.append({
            "field": "margin_sensitivity_values",
            "expected": [float(formal_runtime["equivalence_margin"])],
            "equal": equal_manifest.get("margin_sensitivity_values"),
        })
    if str(equal_manifest.get("formal_script_sha256", "")) != EXPECTED_FORMAL_SCRIPT_SHA256:
        mismatches.append({
            "field": "formal_script_sha256",
            "expected": EXPECTED_FORMAL_SCRIPT_SHA256,
            "equal": equal_manifest.get("formal_script_sha256"),
        })
    if not bool(equal_manifest.get("sensitivity_only", False)):
        mismatches.append({"field": "sensitivity_only", "expected": True})
    if bool(equal_manifest.get("eligible_for_phase2_freeze", True)):
        mismatches.append({"field": "eligible_for_phase2_freeze", "expected": False})
    if bool(equal_manifest.get("B_confirm_read", True)):
        mismatches.append({"field": "B_confirm_read", "expected": False})
    guardrails = dict(equal_manifest.get("guardrails", {}))
    for key, expected in {
        "B_confirm_read": False,
        "eligible_for_phase2_freeze": False,
        "formal_handoff_written": False,
        "formal_family_replaced": False,
        "formal_parameters_replaced": False,
        "sensitivity_output_root_isolated": True,
    }.items():
        if bool(guardrails.get(key, not expected)) != expected:
            mismatches.append({
                "field": f"guardrails.{key}",
                "expected": expected,
                "equal": guardrails.get(key),
            })

    return {
        "passed": not mismatches,
        "mismatches": mismatches,
        "intended_difference": "primary objective weights only",
        "equal_primary_weights": equal_primarys,
    }


def run_finalize(args: argparse.Namespace) -> None:
    score_root = args.score_root.resolve()
    formal_audit_root = args.formal_audit_root.resolve()
    equal_root = args.equal_rerun_root.resolve()
    table_root = score_root / "tables"
    metadata_root = score_root / "metadata"
    table_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    audit_manifest_path = formal_audit_root / "metadata" / "formal_score_contract_audit_manifest.json"
    contract_path = formal_audit_root / "tables" / "frozen_fit_score_contract_selection_summary.csv"
    pareto_path = formal_audit_root / "metadata" / "selected_family_pareto_summary.json"
    equal_manifest_path = equal_root / "metadata" / "score_contract_rerun_manifest.json"
    equal_summary_path = equal_root / "metadata" / "score_contract_sensitivity_summary.json"
    equal_table_path = equal_root / "tables" / "objective_contract_rerun_summary.csv"
    forbidden_handoff = equal_root / "metadata" / "phase1_minimal_mechanism_handoff.json"
    if forbidden_handoff.exists():
        raise RuntimeError(
            "Equal-weight sensitivity root contains the formal Phase-2 handoff filename."
        )
    for path in (
        audit_manifest_path,
        contract_path,
        pareto_path,
        equal_manifest_path,
        equal_summary_path,
        equal_table_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required robustness output not found: {path}")

    audit_manifest = load_json(audit_manifest_path)
    frozen_contracts = pd.read_csv(contract_path, low_memory=False)
    pareto_summary = load_json(pareto_path)
    equal_manifest = load_json(equal_manifest_path)
    equal_summary = load_json(equal_summary_path)
    equal_table = pd.read_csv(equal_table_path, low_memory=False)

    formal_runtime = dict(audit_manifest.get("formal_runtime_contract", {}))
    if not formal_runtime:
        raise RuntimeError("Formal audit manifest lacks the reviewed runtime contract.")
    configuration_audit = compare_equal_rerun_contract(formal_runtime, equal_manifest)
    if not configuration_audit["passed"]:
        raise RuntimeError(
            "Equal-weight rerun changed settings beyond the primary objective weights: "
            f"{configuration_audit['mismatches']}"
        )

    manifest_contract = str(
        equal_manifest.get("objective_contract", "")
    ).strip()
    summary_contract_value = equal_summary.get("objective_contract")
    summary_contract = (
        manifest_contract
        if summary_contract_value is None
        or not str(summary_contract_value).strip()
        else str(summary_contract_value).strip()
    )
    if summary_contract != manifest_contract:
        raise RuntimeError(
            "Equal-primary manifest and sensitivity summary disagree on the "
            f"objective contract: manifest={manifest_contract!r}, "
            f"summary={summary_contract!r}."
        )
    if bool(equal_summary.get("B_confirm_read", True)):
        raise RuntimeError("Equal-weight rerun reports B_confirm access.")
    if bool(equal_summary.get("eligible_for_phase2_freeze", True)):
        raise RuntimeError("Equal-weight rerun is incorrectly marked eligible for Phase 2.")
    if str(audit_manifest.get("formal_script_sha256")) != str(equal_manifest.get("formal_script_sha256")):
        raise RuntimeError("Formal script checksum differs between the frozen-fit and full-rerun audits.")
    equal_primarys = renormalise(equal_manifest.get("objective_weights", {}))
    if any(abs(equal_primarys[name] - CONTRACTS["equal_primary"][name]) > 1e-12 for name in PRIMARY_COMPONENTS):
        raise RuntimeError(f"Equal-weight rerun has unexpected weights: {equal_primarys}")

    formal_final = str(audit_manifest["formal_final_family"])
    frozen_contracts = frozen_contracts.copy()
    frozen_contracts["analysis_type"] = "frozen_formal_fit_reweighting"
    frozen_contracts["full_parameter_reoptimisation"] = False
    frozen_contracts["same_as_formal_final_family"] = (
        frozen_contracts["selected_family_key"].astype(str) == formal_final
    )

    save_json(
        configuration_audit,
        metadata_root / "equal_primary_configuration_audit.json",
    )

    if len(equal_table) != 1:
        raise RuntimeError(
            f"Expected one equal-primary summary row, found {len(equal_table)}."
        )

    equal_row = equal_table.iloc[0].to_dict()
    table_contract_value = equal_row.get("objective_contract")
    table_contract = (
        ""
        if pd.isna(table_contract_value)
        else str(table_contract_value).strip()
    )
    if table_contract != manifest_contract:
        raise RuntimeError(
            "Equal-primary manifest and rerun table disagree on the "
            f"objective contract: manifest={manifest_contract!r}, "
            f"table={table_contract!r}."
        )

    equal_selected_family = str(
        equal_summary.get("final_selected_family", "")
    ).strip()
    if not equal_selected_family:
        raise RuntimeError(
            "Equal-primary sensitivity summary lacks the final selected family."
        )

    table_selected_value = equal_row.get("final_selected_family")
    table_selected_family = (
        ""
        if pd.isna(table_selected_value)
        else str(table_selected_value).strip()
    )
    if table_selected_family != equal_selected_family:
        raise RuntimeError(
            "Equal-primary table and sensitivity summary disagree on the "
            f"final selected family: table={table_selected_family!r}, "
            f"summary={equal_selected_family!r}."
        )

    equal_best_family = str(
        equal_summary.get("final_best_family", "")
    ).strip()
    if not equal_best_family:
        raise RuntimeError(
            "Equal-primary sensitivity summary lacks the final best family."
        )

    table_best_value = equal_row.get("final_best_family")
    table_best_family = (
        ""
        if pd.isna(table_best_value)
        else str(table_best_value).strip()
    )
    if table_best_family != equal_best_family:
        raise RuntimeError(
            "Equal-primary table and sensitivity summary disagree on the "
            f"final best family: table={table_best_family!r}, "
            f"summary={equal_best_family!r}."
        )

    parameter_count = int(equal_row["final_parameter_count"])

    equal_combined = {
        "contract": "equal_primary",
        **{
            f"weight_{name}": CONTRACTS["equal_primary"][name]
            for name in PRIMARY_COMPONENTS
        },
        "best_family_key": equal_best_family,
        "selected_family_key": equal_selected_family,
        "selected_parameter_count": parameter_count,
        "analysis_type": "full_family_parameter_reoptimisation",
        "full_parameter_reoptimisation": True,
        "same_as_formal_final_family": (
            equal_selected_family == formal_final
        ),
        "selected_within_one_se": equal_row.get(
            "final_family_within_one_se"
        ),
        "selected_practically_equivalent": equal_row.get(
            "final_family_practically_equivalent"
        ),
        "would_pass_selection_gates_under_contract": equal_row.get(
            "would_pass_selection_gates_under_contract"
        ),
        "would_satisfy_current_phase2_family_contract": equal_row.get(
            "would_satisfy_current_phase2_family_contract"
        ),
        "sensitivity_only": True,
        "eligible_for_phase2_freeze": False,
        "fit_status": (
            "all families and direct deletions re-optimised under "
            "equal primary-component weights"
        ),
    }
    combined = pd.concat(
        [frozen_contracts, pd.DataFrame([equal_combined])],
        ignore_index=True,
        sort=False,
    )
    write_csv(combined, table_root / "score_contract_robustness_summary.csv")

    equal_frozen_row = frozen_contracts.loc[frozen_contracts["contract"] == "equal_primary"].iloc[0]
    final_summary = {
        "created_at": now_string(),
        "formal_final_family": formal_final,
        "frozen_fit_equal_selected_family": str(equal_frozen_row["selected_family_key"]),
        "full_equal_primary_reoptimisation_selected_family": equal_selected_family,
        "full_equal_primary_reoptimisation_parameter_count": parameter_count,
        "full_equal_primary_reoptimisation_would_pass_selection_gates": bool(
            equal_summary.get(
                "would_pass_selection_gates_under_contract",
                False,
            )
        ),
        "full_equal_primary_reoptimisation_would_satisfy_current_phase2_family_contract": bool(
            equal_summary.get(
                "would_satisfy_current_phase2_family_contract",
                False,
            )
        ),
        "all_frozen_fit_stress_contracts_select_formal_family": bool(
            frozen_contracts.loc[
                frozen_contracts["contract"] != "formal",
                "same_as_formal_final_family",
            ].all()
        ),
        "no_frozen_fit_stress_contract_selects_a_simpler_family": bool(
            (pd.to_numeric(
                frozen_contracts.loc[
                    frozen_contracts["contract"] != "formal",
                    "selected_parameter_count",
                ],
                errors="coerce",
            ) >= int(
                frozen_contracts.loc[
                    frozen_contracts["contract"] == "formal",
                    "selected_parameter_count",
                ].iloc[0]
            )).all()
        ),
        "pareto_summary": pareto_summary,
        "equal_primary_configuration_audit": configuration_audit,
        "interpretation_boundary": (
            "The frozen-fit stress contracts do not re-optimise family parameters; "
            "only the equal-primary-component contract performs a complete Phase-1 re-optimisation. "
            "All analyses are post hoc and development-only."
        ),
        "eligible_for_phase2_freeze": False,
        "formal_family_or_parameters_modified": False,
        "B_confirm_read": False,
    }
    save_json(final_summary, metadata_root / "score_contract_robustness_summary.json")

    final_manifest = {
        "created_at": now_string(),
        "audit_version": AUDIT_VERSION,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "formal_audit_manifest": str(audit_manifest_path),
        "formal_audit_manifest_sha256": sha256_file(audit_manifest_path),
        "equal_rerun_manifest": str(equal_manifest_path),
        "equal_rerun_manifest_sha256": sha256_file(equal_manifest_path),
        "formal_script_sha256": audit_manifest["formal_script_sha256"],
        "post_hoc": True,
        "B_confirm_read": False,
        "formal_outputs_modified": False,
        "formal_family_or_parameters_modified": False,
        "eligible_for_phase2_freeze": False,
        "nonduplication": {
            "existing_equivalence_margin_sensitivity_repeated": False,
            "formal_weight_parameter_deletion_result_repeated": False,
            "equal_primary_deletion_procedure_rerun_as_required_part_of_new_contract": True,
            "existing_parameter_deletion_experiment_replaced": False,
            "new_question": "robustness to primary-score weights and six-objective complexity-Pareto dominance",
        },
        "equal_primary_configuration_audit": configuration_audit,
        "combined_summary": final_summary,
    }
    manifest_path = metadata_root / "score_contract_robustness_manifest.json"
    save_json(final_manifest, manifest_path)
    save_json(
        {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        metadata_root / "score_contract_robustness_manifest.sha256.json",
    )
    print("[score-contract-audit] finalization completed")
    print(f"[score-contract-audit] combined outputs: {score_root}")


def run_self_test() -> None:
    rows: List[Dict[str, Any]] = []
    for rep in range(20):
        for key, k, base in (
            ("simple", 2, 0.20),
            ("selected", 4, 0.15),
            ("rich", 7, 0.14),
        ):
            rows.append({
                "bootstrap_rep": rep,
                "family_key": key,
                **{name: base + 0.001 * rep for name in PRIMARY_COMPONENTS},
            })
    components = pd.DataFrame(rows)
    meta = pd.DataFrame({
        "family_key": ["simple", "selected", "rich"],
        "Model family": ["Simple", "Selected", "Rich"],
        "Free mechanism parameters": [2, 4, 7],
    })
    summary, details = replay_selection(components, meta, "equal", CONTRACTS["equal_primary"], 0.02)
    if summary["best_family_key"] != "rich":
        raise AssertionError("Selection self-test failed to identify the best family.")
    mean_table, rep_table, pareto = pareto_tables(components, meta, "selected", 1e-12)
    if mean_table.empty or rep_table.empty or "selected_on_complexity_pareto_front" not in pareto:
        raise AssertionError("Pareto self-test failed.")
    # A lower-complexity family with identical losses must complexity-dominate.
    equal_losses = np.array([0.1] * 5)
    if not dominates(equal_losses, equal_losses, 2, 4, include_complexity=True, tolerance=1e-12):
        raise AssertionError("Complexity dominance self-test failed.")
    print("score-contract audit self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=False)

    formal = sub.add_parser("formal-audit", help="Reconstruct formal component bootstraps and run frozen-fit audits.")
    formal.add_argument("--formal-script", type=Path, required=True)
    formal.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    formal.add_argument("--stage1-root", type=Path, default=None)
    formal.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_SCORE_ROOT / "formal_audit",
    )
    formal.add_argument("--reconstruction-tolerance", type=float, default=1e-10)
    formal.add_argument("--pareto-tolerance", type=float, default=1e-12)
    formal.add_argument("--overwrite", action="store_true")

    finalize = sub.add_parser("finalize", help="Combine frozen-fit audit with the full equal-primary-component rerun.")
    finalize.add_argument("--score-root", type=Path, default=DEFAULT_SCORE_ROOT)
    finalize.add_argument(
        "--formal-audit-root",
        type=Path,
        default=DEFAULT_SCORE_ROOT / "formal_audit",
    )
    finalize.add_argument(
        "--equal-rerun-root",
        type=Path,
        default=DEFAULT_SCORE_ROOT / "equal_primary_rerun",
    )

    sub.add_parser("self-test", help="Run deterministic selection and Pareto unit tests.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    command = args.command or "self-test"
    if command == "formal-audit":
        run_formal_audit(args)
    elif command == "finalize":
        run_finalize(args)
    elif command == "self-test":
        run_self_test()
    else:
        parser.error(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
