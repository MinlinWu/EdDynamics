#!/usr/bin/env python3
"""Extract compact Supplementary Information statistics for the frozen minimal mechanism."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EXPECTED_FAMILY = "offset_dual_channel"
EXPECTED_PRIMARY_MACROSTATE = ("M", "Psi")
EXPECTED_FREE_PARAMETERS = ("theta0", "thetaM", "phi0", "deltaS")
EXPECTED_ZERO_PARAMETERS = ("thetaPsi", "thetaMPsi", "phiPsi")
EXPECTED_NUISANCE_PARAMETERS = ("lambdaR", "lambdaA", "lambdaI")
EXPECTED_MACROSTATE_K = 6
EXPECTED_KMEANS_FEATURES = (
    "M_response_prebalanced_pre",
    "activity_alignment_order_Psi_pre",
)
EPS = 1e-12

PARAMETER_LABELS = {
    "theta0": r"$\theta_0$",
    "thetaM": r"$\theta_M$",
    "thetaPsi": r"$\theta_\Psi$",
    "thetaMPsi": r"$\theta_{M\Psi}$",
    "phi0": r"$\phi_0$",
    "deltaS": r"$\delta_S$",
    "phiPsi": r"$\phi_\Psi$",
    "lambdaR": r"$\lambda_R$",
    "lambdaA": r"$\lambda_A$",
    "lambdaI": r"$\lambda_I$",
    "eta": r"$\eta$",
    "tau_response_days": r"$\tau_R$",
    "tau_activity_days": r"$\tau_A$",
    "residual_mass_per_answer": r"$r$",
    "response_signed_gain": r"$\gamma_R$",
    "alignment_signed_gain": r"$\gamma_A$",
    "sigma_U0": r"$\sigma_U$",
    "sigma_Psi0": r"$\sigma_\Psi$",
}

PARAMETER_ROLES = {
    "theta0": "response offset",
    "thetaM": "response-restoring coefficient",
    "phi0": "alignment baseline",
    "deltaS": "response--support alignment contrast",
    "thetaPsi": "direct exposure-coordinate forcing",
    "thetaMPsi": "response--exposure interaction",
    "phiPsi": "state-dependent alignment feedback",
    "lambdaR": "fixed response-mass scale",
    "lambdaA": "fixed active-exposure scale",
    "lambdaI": "fixed idle-exposure scale",
    "eta": "evidence-maturity scale",
    "tau_response_days": "response-memory time constant (days)",
    "tau_activity_days": "exposure-memory time constant (days)",
    "residual_mass_per_answer": "residual evidence per answered question",
    "response_signed_gain": "signed-response calibration gain",
    "alignment_signed_gain": "signed-alignment calibration gain",
    "sigma_U0": "response residual scale (diagnostic only)",
    "sigma_Psi0": "exposure residual scale (diagnostic only)",
}

FAMILY_LABELS = {
    "persistence": "State persistence",
    "response_only": "Response-only mechanism",
    "alignment_only": "Alignment-only mechanism",
    "response_offset_core": "Response-offset core",
    "dual_channel_core": "Dual-channel core",
    "offset_dual_channel": "Offset dual-channel",
    "full_reference": "Seven-term reference",
}


@dataclass(frozen=True)
class SourceRecord:
    name: str
    path: Path
    sha256: str
    rows: Optional[int] = None
    columns: Optional[int] = None


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


def save_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_path(base: Path) -> Path:
    if base.exists() and base.is_file():
        return base
    for suffix in (".parquet", ".csv.gz", ".csv", ".tsv"):
        path = base.with_suffix(suffix)
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find table for {base}")


def read_table(base: Path) -> Tuple[pd.DataFrame, Path]:
    path = table_path(base)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".tsv":
        frame = pd.read_csv(path, sep="\t", low_memory=False)
    else:
        frame = pd.read_csv(path, low_memory=False)
    return frame, path


def write_table(frame: pd.DataFrame, base: Path) -> Dict[str, str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base.with_suffix(".csv")
    frame.to_csv(csv_path, index=False)
    outputs = {"csv": str(csv_path.resolve())}
    try:
        parquet_path = base.with_suffix(".parquet")
        frame.to_parquet(parquet_path, index=False)
        outputs["parquet"] = str(parquet_path.resolve())
    except Exception:
        pass
    return outputs


def coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return bool(int(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "t"}:
            return True
        if text in {"0", "false", "no", "n", "f"}:
            return False
    return None


def finite_float(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return float("nan")
    return number if np.isfinite(number) else float("nan")

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_keys(mapping: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise RuntimeError(f"{label} is missing keys: {missing}")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} is missing columns: {missing}")

def source_record(name: str, path: Path, frame: Optional[pd.DataFrame] = None) -> SourceRecord:
    return SourceRecord(
        name=name,
        path=path.resolve(),
        sha256=sha256_file(path),
        rows=None if frame is None else int(len(frame)),
        columns=None if frame is None else int(len(frame.columns)),
    )


def resolve_roots(args: argparse.Namespace) -> Dict[str, Path]:
    base = args.output_base.resolve()
    stage1 = (args.stage1_root or base / "stage1").resolve()
    minimality = (args.minimality_root or base / "stage2_phase1_unified_minimality").resolve()
    tuning = (args.tuning_root or base / "stage2_phase1").resolve()
    frozen = (args.frozen_root or base / "stage2_phase2_freeze").resolve()
    confirmation = (args.confirmation_root or base / "stage2_phase3_confirm").resolve()
    figure = (args.figure_root or confirmation / "figures_publication_minimal_mechanism").resolve()
    output = (args.output_root or confirmation / "supplementary_minimal_mechanism").resolve()
    return {
        "base": base,
        "stage1": stage1,
        "minimality": minimality,
        "tuning": tuning,
        "frozen": frozen,
        "confirmation": confirmation,
        "figure": figure,
        "output": output,
    }


def load_sources(roots: Mapping[str, Path], confirm_split: str) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame], List[SourceRecord]]:
    json_paths = {
        "minimality_manifest": roots["minimality"] / "metadata" / "minimality_experiment_manifest.json",
        "minimality_selection": roots["minimality"] / "metadata" / "phase1_minimal_mechanism_handoff.json",
        "tuned_parameters": roots["tuning"] / "metadata" / "phase1_selected_parameters.json",
        "tuning_manifest": roots["tuning"] / "metadata" / "phase1_manifest.json",
        "frozen_manifest": roots["frozen"] / "metadata" / "phase2_frozen_model_manifest.json",
        "frozen_calibration": roots["frozen"] / "metadata" / "phase2_pooled_development_calibration.json",
        "confirmation_manifest": roots["confirmation"] / "metadata" / "phase3_confirmation_manifest.json",
        "confirmation_audit": roots["confirmation"] / "metadata" / "phase3_no_update_audit.json",
        "figure_manifest": roots["figure"] / "publication_minimal_mechanism_figure_manifest.json",
        "fixed_k6_metadata": roots["stage1"] / "dynamics" / "fixed_k6_mesostates" / "fixed_k6_model_metadata.json",
    }
    jsons: Dict[str, Any] = {}
    records: List[SourceRecord] = []
    for name, path in json_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required source not found: {path}")
        jsons[name] = load_json(path)
        records.append(source_record(name, path))

    table_bases = {
        "family_results": roots["minimality"] / "tables" / "model_family_results",
        "family_summary": roots["minimality"] / "tables" / "manuscript_results_summary",
        "scalar_deletions": roots["minimality"] / "tables" / "global_scalar_deletion_audit",
        "margin_sensitivity": roots["minimality"] / "tables" / "equivalence_margin_sensitivity",
        "parameter_boundaries": roots["minimality"] / "tables" / "parameter_grid_boundaries",
        "next_required_tests": roots["minimality"] / "tables" / "next_required_tests",
        "development_confirmation_stability": roots["confirmation"] / "tables" / "phase3_development_vs_confirmation_metric_stability",
        "figure3_metrics": roots["figure"] / "tables" / "figure3_kinetic_recovery_metrics",
        "fixed_k6_centers": roots["stage1"] / "dynamics" / "fixed_k6_mesostates" / "fixed_k6_centers",
        "empirical_transition_matrix": roots["figure"] / "tables" / f"{confirm_split}_empirical_mesostate_transition_matrix",
        "mechanism_transition_matrix": roots["figure"] / "tables" / f"{confirm_split}_mechanism_mesostate_transition_matrix",
        "empirical_residence_curves": roots["figure"] / "tables" / f"{confirm_split}_empirical_residence_curves_fixed_k6",
        "mechanism_residence_references": roots["figure"] / "tables" / f"{confirm_split}_mechanism_residence_reference_curves",
        "empirical_residence_summary": roots["figure"] / "tables" / f"{confirm_split}_empirical_residence_summary_fixed_k6",
    }
    tables: Dict[str, pd.DataFrame] = {}
    for name, base in table_bases.items():
        frame, path = read_table(base)
        tables[name] = frame
        records.append(source_record(name, path, frame))
    return jsons, tables, records


def validate_contracts(jsons: Mapping[str, Any], tables: Mapping[str, pd.DataFrame], confirm_split: str) -> None:
    selection = jsons["minimality_selection"]
    required_true = (
        "ready_for_phase2_freeze",
        "scalar_parameter_minimality_confirmed",
        "search_adequacy_confirmed",
        "baseline_not_practically_equivalent_to_best",
        "final_model_selected_by_parsimony_rule",
    )
    failed = [name for name in required_true if coerce_bool(selection.get(name)) is not True]
    require(not failed, f"Minimality selection failed required gates: {failed}")
    require(selection.get("final_family_key") == EXPECTED_FAMILY, "Unexpected selected mechanism family")
    require(set(selection.get("final_free_mechanism_parameters", [])) == set(EXPECTED_FREE_PARAMETERS), "Unexpected free mechanism parameters")
    require(coerce_bool(selection.get("pairwise_parameter_necessity_not_claimed")) is True, "Pairwise-necessity boundary is not recorded")

    require(tuple(jsons["tuning_manifest"].get("primary_macrostate", [])) == EXPECTED_PRIMARY_MACROSTATE, "Tuning did not use the M/Psi primary state")
    require(tuple(jsons["frozen_manifest"].get("primary_macrostate", [])) == EXPECTED_PRIMARY_MACROSTATE, "Frozen specification did not use the M/Psi primary state")
    require(tuple(jsons["confirmation_manifest"].get("primary_macrostate", [])) == EXPECTED_PRIMARY_MACROSTATE, "Confirmation did not use the M/Psi primary state")
    require(str(jsons["confirmation_manifest"].get("confirm_split", "")) == confirm_split, "Unexpected confirmation split")

    phase2_guardrails = dict(jsons["frozen_manifest"].get("guardrails", {}))
    for key in (
        "B_confirm_read",
        "parameter_search_opened",
        "mechanism_family_reselected",
        "mechanism_parameters_refit",
        "kmeans_refit",
        "macrostate_k_selected",
        "region_redefinition",
    ):
        require(coerce_bool(phase2_guardrails.get(key, False)) is False, f"Frozen-development guardrail failed: {key}")
    require(str(phase2_guardrails.get("calibration_refit_scope", "")) == "A_train_plus_A_val_only", "Unexpected calibration scope")

    phase3_guardrails = dict(jsons["confirmation_manifest"].get("guardrails", {}))
    for key in (
        "parameter_search_opened",
        "calibration_reestimated",
        "mechanism_family_reselected",
        "mechanism_parameters_refit",
        "kmeans_refit",
        "macrostate_k_selected",
        "region_redefinition",
        "B_confirm_used_for_update",
    ):
        require(coerce_bool(phase3_guardrails.get(key, False)) is False, f"Confirmation guardrail failed: {key}")
    require(str(phase3_guardrails.get("confirmation_mode", "")) == "output_only", "Confirmation was not output-only")

    audit = jsons["confirmation_audit"]
    for key in (
        "parameter_search_opened",
        "calibration_reestimated",
        "mechanism_family_reselected",
        "mechanism_parameters_refit",
        "kmeans_refit",
        "macrostate_k_selected",
        "region_redefinition",
        "B_confirm_used_for_model_update",
    ):
        require(coerce_bool(audit.get(key, False)) is False, f"No-update audit failed: {key}")
    require(audit.get("frozen_parameter_hash_before_confirmation") == audit.get("frozen_parameter_hash_after_confirmation"), "Frozen parameter hash changed during confirmation")
    require(audit.get("frozen_calibration_hash_before_confirmation") == audit.get("frozen_calibration_hash_after_confirmation"), "Frozen calibration hash changed during confirmation")

    frozen = dict(jsons["frozen_manifest"].get("frozen_parameters", {}))
    require(frozen.get("family_key") == EXPECTED_FAMILY, "Frozen family differs from the selected family")
    require(set(frozen.get("free_mechanism_parameters", [])) == set(EXPECTED_FREE_PARAMETERS), "Frozen free parameters differ from the selected family")
    vector = dict(frozen.get("full_parameter_vector", {}))
    require_keys(vector, EXPECTED_FREE_PARAMETERS + EXPECTED_ZERO_PARAMETERS + EXPECTED_NUISANCE_PARAMETERS, "Frozen parameter vector")
    for name in EXPECTED_ZERO_PARAMETERS:
        require(abs(finite_float(vector[name])) <= 1e-12, f"Structural-zero parameter is non-zero: {name}")

    tuned = dict(jsons["tuned_parameters"].get("selected_parameters", {}))
    require_keys(tuned, EXPECTED_FREE_PARAMETERS + EXPECTED_ZERO_PARAMETERS + EXPECTED_NUISANCE_PARAMETERS, "Tuned parameter vector")
    for name in EXPECTED_FREE_PARAMETERS + EXPECTED_ZERO_PARAMETERS + EXPECTED_NUISANCE_PARAMETERS:
        require(np.isclose(finite_float(tuned[name]), finite_float(vector[name]), atol=1e-12, rtol=0.0), f"Tuned and frozen values differ for {name}")

    kmeans = jsons["fixed_k6_metadata"]
    expected = {
        "coordinate": "MR_PsiA",
        "macrostate_k": EXPECTED_MACROSTATE_K,
        "macrostate_k_rule": "fixed a priori",
        "fit_split": "A_train",
        "features": list(EXPECTED_KMEANS_FEATURES),
        "user_balanced_sampling": True,
        "user_balanced_kmeans_fit": True,
        "fit_max_rows": 500000,
        "kmeans_n_init": 20,
        "random_state": 42,
    }
    for key, value in expected.items():
        require(kmeans.get(key) == value, f"Fixed-K contract mismatch for {key}: {kmeans.get(key)!r}")

    figure_guardrails = dict(jsons["figure_manifest"].get("guardrails", {}))
    require(coerce_bool(figure_guardrails.get("KMeans_not_refit")) is True, "Publication evaluation refitted KMeans")
    require(coerce_bool(figure_guardrails.get("macrostate_k_fixed_at_6")) is True, "Publication evaluation changed K")
    require(coerce_bool(figure_guardrails.get("macrostate_k_selected_in_figure_code")) is False, "Publication evaluation selected K")
    require(coerce_bool(figure_guardrails.get("transition_and_residence_not_phase1_targets")) is True, "Post-selection kinetic boundary is missing")

    next_tests = tables["next_required_tests"]
    require(next_tests.empty, "Minimality experiment still lists required follow-up tests")
    centers = tables["fixed_k6_centers"]
    state_col = "macrostate" if "macrostate" in centers.columns else "state" if "state" in centers.columns else None
    require(state_col is not None, "Fixed-K centers lack macrostate labels")
    states = pd.to_numeric(centers[state_col], errors="coerce").dropna().astype(int).to_numpy()
    require(len(centers) == EXPECTED_MACROSTATE_K and np.array_equal(np.sort(states), np.arange(EXPECTED_MACROSTATE_K)), "Fixed-K centers do not contain states 0--5 exactly once")


def format_interval(lower: Any, upper: Any, digits: int = 4) -> str:
    lo = finite_float(lower)
    hi = finite_float(upper)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return ""
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def format_value(value: Any, digits: int = 4) -> str:
    number = finite_float(value)
    if not np.isfinite(number):
        return ""
    absolute = abs(number)
    if absolute != 0 and (absolute < 1e-3 or absolute >= 1e4):
        return f"{number:.3e}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def yes_no(value: Any) -> str:
    flag = coerce_bool(value)
    return "Yes" if flag is True else "No" if flag is False else ""


def family_selection_table(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    results = tables["family_results"].copy()
    required = (
        "family_key",
        "Model family",
        "Free mechanism parameters",
        "Free parameter names",
        "Bootstrap mean primary score",
        "Bootstrap 95% CI lower",
        "Bootstrap 95% CI upper",
        "difference_to_best_mean",
        "difference_to_best_ci95_lower",
        "difference_to_best_ci95_upper",
        "Within one standard error of best",
        "Practically equivalent to best",
        "Parsimonious family selected",
        "Final scalar-minimal family",
    )
    require_columns(results, required, "Model-family results")
    output = pd.DataFrame({
        "family_key": results["family_key"].astype(str),
        "family": results["Model family"].astype(str),
        "free_parameters": pd.to_numeric(results["Free mechanism parameters"], errors="coerce").astype("Int64"),
        "parameter_names": results["Free parameter names"].fillna("").astype(str),
        "bootstrap_mean_score": pd.to_numeric(results["Bootstrap mean primary score"], errors="coerce"),
        "bootstrap_95_interval": [
            format_interval(lo, hi)
            for lo, hi in zip(results["Bootstrap 95% CI lower"], results["Bootstrap 95% CI upper"])
        ],
        "difference_to_best_mean": pd.to_numeric(results["difference_to_best_mean"], errors="coerce"),
        "difference_to_best_95_interval": [
            format_interval(lo, hi)
            for lo, hi in zip(results["difference_to_best_ci95_lower"], results["difference_to_best_ci95_upper"])
        ],
        "within_one_standard_error": results["Within one standard error of best"].map(yes_no),
        "practically_equivalent_to_best": results["Practically equivalent to best"].map(yes_no),
        "parsimony_selected": results["Parsimonious family selected"].map(yes_no),
        "final_family": results["Final scalar-minimal family"].map(yes_no),
    })
    output["parameter_names"] = output["parameter_names"].replace({"nan": "", "NA": ""})
    return output.sort_values(["free_parameters", "bootstrap_mean_score", "family_key"], kind="mergesort").reset_index(drop=True)


def scalar_deletion_summary(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    deletion = tables["scalar_deletions"].copy()
    required = (
        "tested_removed_parameter",
        "deletion_family",
        "deletion_free_parameters",
        "difference_to_best_mean",
        "difference_to_best_ci95_lower",
        "difference_to_best_ci95_upper",
        "within_one_standard_error_of_best",
        "practically_equivalent_to_best",
        "globally_eligible_under_selection_rule",
        "globally_required",
        "pairwise_conclusion_descriptive",
    )
    require_columns(deletion, required, "Scalar-deletion audit")
    output = pd.DataFrame({
        "removed_parameter": deletion["tested_removed_parameter"].astype(str),
        "removed_parameter_label": deletion["tested_removed_parameter"].astype(str).map(PARAMETER_LABELS).fillna(deletion["tested_removed_parameter"].astype(str)),
        "reduced_family": deletion["deletion_family"].astype(str),
        "remaining_parameters": deletion["deletion_free_parameters"].fillna("").astype(str),
        "difference_to_best_mean": pd.to_numeric(deletion["difference_to_best_mean"], errors="coerce"),
        "difference_to_best_95_interval": [
            format_interval(lo, hi)
            for lo, hi in zip(deletion["difference_to_best_ci95_lower"], deletion["difference_to_best_ci95_upper"])
        ],
        "within_one_standard_error": deletion["within_one_standard_error_of_best"].map(yes_no),
        "practically_equivalent_to_best": deletion["practically_equivalent_to_best"].map(yes_no),
        "eligible_under_global_rule": deletion["globally_eligible_under_selection_rule"].map(yes_no),
        "retained_by_global_rule": deletion["globally_required"].map(yes_no),
        "pairwise_description": deletion["pairwise_conclusion_descriptive"].fillna("").astype(str),
    })
    order = {name: index for index, name in enumerate(EXPECTED_FREE_PARAMETERS)}
    output["_order"] = output["removed_parameter"].map(order).fillna(99)
    return output.sort_values(["_order", "removed_parameter"], kind="mergesort").drop(columns="_order").reset_index(drop=True)


def margin_sensitivity_table(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    margin = tables["margin_sensitivity"].copy()
    require_columns(margin, ("equivalence_margin", "selected_family_key", "selected_family_label"), "Margin-sensitivity table")
    output = margin[["equivalence_margin", "selected_family_key", "selected_family_label"]].copy()
    output["equivalence_margin"] = pd.to_numeric(output["equivalence_margin"], errors="coerce")
    return output.sort_values("equivalence_margin", kind="mergesort").reset_index(drop=True)


def boundary_audit_table(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    boundary = tables["parameter_boundaries"].copy()
    if boundary.empty:
        return boundary
    columns = [
        column for column in (
            "family_key",
            "parameter",
            "value",
            "boundary",
            "saturation_plateau",
            "blocking_for_freeze",
            "plateau_confirmed",
        )
        if column in boundary.columns
    ]
    output = boundary[columns].copy()
    for column in ("saturation_plateau", "blocking_for_freeze", "plateau_confirmed"):
        if column in output.columns:
            output[column] = output[column].map(yes_no)
    return output


def frozen_specification_summary(jsons: Mapping[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frozen = dict(jsons["frozen_manifest"].get("frozen_parameters", {}))
    vector = dict(frozen.get("full_parameter_vector", {}))
    calibration = dict(jsons["frozen_manifest"].get("frozen_calibration", {}))
    rows: List[Dict[str, Any]] = []
    for group, names in (
        ("Active mechanism term", EXPECTED_FREE_PARAMETERS),
        ("Structural zero", EXPECTED_ZERO_PARAMETERS),
        ("Fixed scale", EXPECTED_NUISANCE_PARAMETERS),
    ):
        for name in names:
            rows.append({
                "category": group,
                "symbol": PARAMETER_LABELS[name],
                "parameter": name,
                "value": finite_float(vector.get(name)),
                "role": PARAMETER_ROLES[name],
                "estimation_scope": "pooled development" if group == "Active mechanism term" else "fixed before family comparison",
            })
    calibration_names = (
        "eta",
        "tau_response_days",
        "tau_activity_days",
        "residual_mass_per_answer",
        "response_signed_gain",
        "alignment_signed_gain",
    )
    for name in calibration_names:
        if name in calibration:
            rows.append({
                "category": "Accounting calibration",
                "symbol": PARAMETER_LABELS[name],
                "parameter": name,
                "value": finite_float(calibration.get(name)),
                "role": PARAMETER_ROLES[name],
                "estimation_scope": "pooled development",
            })
    display = pd.DataFrame(rows)

    all_calibration_rows = []
    for name, value in calibration.items():
        all_calibration_rows.append({
            "parameter": str(name),
            "symbol": PARAMETER_LABELS.get(str(name), str(name)),
            "value": finite_float(value),
            "role": PARAMETER_ROLES.get(str(name), "frozen calibration or diagnostic quantity"),
            "estimation_scope": "pooled development",
            "affects_deterministic_mean_update": str(name) not in {"sigma_U0", "sigma_Psi0"},
        })
    complete_calibration = pd.DataFrame(all_calibration_rows)
    return display, complete_calibration


def confirmation_audit_table(jsons: Mapping[str, Any], tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    selection = jsons["minimality_selection"]
    phase2 = dict(jsons["frozen_manifest"].get("guardrails", {}))
    phase3 = dict(jsons["confirmation_manifest"].get("guardrails", {}))
    audit = jsons["confirmation_audit"]
    parameter_hash_changed = audit.get("frozen_parameter_hash_before_confirmation") != audit.get("frozen_parameter_hash_after_confirmation")
    calibration_hash_changed = audit.get("frozen_calibration_hash_before_confirmation") != audit.get("frozen_calibration_hash_after_confirmation")
    rows = [
        ("Final family selected by the prespecified parsimony rule", selection.get("final_model_selected_by_parsimony_rule"), "Yes"),
        ("Scalar-deletion fixed point reached", selection.get("scalar_parameter_minimality_confirmed"), "Yes"),
        ("Unresolved search-boundary case remained", not coerce_bool(selection.get("search_adequacy_confirmed")), "No"),
        ("Pairwise necessity of every coefficient claimed", not coerce_bool(selection.get("pairwise_parameter_necessity_not_claimed")), "No"),
        ("Confirmation data accessed before freezing", phase2.get("B_confirm_read"), "No"),
        ("Family selection reopened after development", phase2.get("mechanism_family_reselected"), "No"),
        ("Mechanism parameters refitted after development", phase2.get("mechanism_parameters_refit"), "No"),
        ("Parameter search opened during confirmation", phase3.get("parameter_search_opened"), "No"),
        ("Accounting calibration re-estimated during confirmation", phase3.get("calibration_reestimated"), "No"),
        ("Mesostate partition refitted during confirmation", phase3.get("kmeans_refit"), "No"),
        ("Mesostate number reselected during confirmation", phase3.get("macrostate_k_selected"), "No"),
        ("Spatial evaluation regions redefined during confirmation", phase3.get("region_redefinition"), "No"),
        ("Confirmation data used to update the mechanism", phase3.get("B_confirm_used_for_update"), "No"),
        ("Frozen parameter hash changed during confirmation", parameter_hash_changed, "No"),
        ("Frozen calibration hash changed during confirmation", calibration_hash_changed, "No"),
    ]
    output = pd.DataFrame(rows, columns=["audit_item", "observed", "required"])
    output["observed"] = output["observed"].map(yes_no)
    output["passed"] = output["observed"] == output["required"]
    output.loc[len(output)] = {
        "audit_item": "Confirmation mode",
        "observed": str(phase3.get("confirmation_mode", "")),
        "required": "output_only",
        "passed": str(phase3.get("confirmation_mode", "")) == "output_only",
    }
    output.loc[len(output)] = {
        "audit_item": "Outstanding required tests",
        "observed": str(int(len(tables["next_required_tests"]))),
        "required": "0",
        "passed": tables["next_required_tests"].empty,
    }
    return output


def matrix_from_frame(frame: pd.DataFrame, label: str) -> np.ndarray:
    columns = [column for column in frame.columns if not str(column).lower().startswith("unnamed")]
    matrix = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    require(matrix.shape == (EXPECTED_MACROSTATE_K, EXPECTED_MACROSTATE_K), f"{label} must be a 6 by 6 matrix")
    require(np.isfinite(matrix).all(), f"{label} contains non-finite values")
    return matrix


def ordered_centers(frame: pd.DataFrame) -> pd.DataFrame:
    state_col = "macrostate" if "macrostate" in frame.columns else "state"
    m_candidates = ("center_M", "M_center", "M")
    psi_candidates = ("center_Psi", "Psi_center", "Psi")
    m_col = next((name for name in m_candidates if name in frame.columns), None)
    psi_col = next((name for name in psi_candidates if name in frame.columns), None)
    require(m_col is not None and psi_col is not None, "Fixed-K center coordinates are unavailable")
    output = pd.DataFrame({
        "macrostate": pd.to_numeric(frame[state_col], errors="coerce").astype(int),
        "center_M": pd.to_numeric(frame[m_col], errors="coerce"),
        "center_Psi": pd.to_numeric(frame[psi_col], errors="coerce"),
    })
    return output.sort_values("macrostate", kind="mergesort").reset_index(drop=True)


def statewise_kinetic_table(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    empirical = matrix_from_frame(tables["empirical_transition_matrix"], "Empirical transition matrix")
    mechanism = matrix_from_frame(tables["mechanism_transition_matrix"], "Mechanism transition matrix")
    centers = ordered_centers(tables["fixed_k6_centers"])
    rows: List[Dict[str, Any]] = []
    for state in range(EXPECTED_MACROSTATE_K):
        pe = float(empirical[state, state])
        pm = float(mechanism[state, state])
        row_tv = 0.5 * float(np.sum(np.abs(mechanism[state] - empirical[state])))
        empirical_top = int(np.argmax(empirical[state]))
        mechanism_top = int(np.argmax(mechanism[state]))
        empirical_mean = 1.0 / max(1.0 - min(pe, 1.0 - 1e-9), EPS)
        mechanism_mean = 1.0 / max(1.0 - min(pm, 1.0 - 1e-9), EPS)
        center = centers.loc[centers["macrostate"] == state].iloc[0]
        rows.append({
            "macrostate": f"S{state}",
            "training_center_M": float(center["center_M"]),
            "training_center_Psi": float(center["center_Psi"]),
            "empirical_self_transition": pe,
            "mechanism_self_transition": pm,
            "self_transition_difference": pm - pe,
            "row_total_variation": row_tv,
            "empirical_diagonal_dominant": pe >= float(np.max(empirical[state])) - 1e-12,
            "mechanism_diagonal_dominant": pm >= float(np.max(mechanism[state])) - 1e-12,
            "empirical_top_next_state": f"S{empirical_top}",
            "mechanism_top_next_state": f"S{mechanism_top}",
            "top_outgoing_edge_match": empirical_top == mechanism_top,
            "empirical_geometric_mean_reference": empirical_mean,
            "mechanism_geometric_mean_reference": mechanism_mean,
            "geometric_mean_reference_ratio": mechanism_mean / empirical_mean,
        })
    return pd.DataFrame(rows)


def quality_gate_table(
    jsons: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    family: pd.DataFrame,
    deletion: pd.DataFrame,
    margin: pd.DataFrame,
    specification: pd.DataFrame,
    audit: pd.DataFrame,
    kinetics: pd.DataFrame,
) -> pd.DataFrame:
    selected = family[family["final_family"].eq("Yes")]
    expected_margin_family = set(margin["selected_family_key"].astype(str))
    figure3 = tables["figure3_metrics"]
    metrics_row = figure3.iloc[0] if not figure3.empty else pd.Series(dtype=object)
    gates = [
        ("final_family_unique", len(selected) == 1 and selected.iloc[0]["family_key"] == EXPECTED_FAMILY, str(selected["family_key"].tolist())),
        ("all_active_terms_tested_by_deletion", set(deletion["removed_parameter"].astype(str)) == set(EXPECTED_FREE_PARAMETERS), ";".join(sorted(deletion["removed_parameter"].astype(str)))),
        ("no_globally_eligible_one_term_deletion", not deletion["eligible_under_global_rule"].eq("Yes").any(), str(deletion.loc[deletion["eligible_under_global_rule"].eq("Yes"), "removed_parameter"].tolist())),
        ("margin_selection_stable", expected_margin_family == {EXPECTED_FAMILY}, ";".join(sorted(expected_margin_family))),
        ("frozen_specification_complete", set(EXPECTED_FREE_PARAMETERS + EXPECTED_ZERO_PARAMETERS + EXPECTED_NUISANCE_PARAMETERS).issubset(set(specification["parameter"])), str(len(specification))),
        ("confirmation_audit_passed", bool(audit["passed"].all()), str(int(audit["passed"].sum())) + "/" + str(len(audit))),
        ("fixed_six_state_kinetics_complete", len(kinetics) == EXPECTED_MACROSTATE_K, str(len(kinetics))),
        ("transition_and_residence_excluded_from_selection", bool(jsons["figure_manifest"].get("guardrails", {}).get("transition_and_residence_not_phase1_targets")), "verified"),
        ("residence_reference_is_transition_implied", "transition-implied" in str(jsons["figure_manifest"].get("guardrails", {}).get("mechanism_residence_curve", "")), str(jsons["figure_manifest"].get("guardrails", {}).get("mechanism_residence_curve", ""))),
        ("figure3_transition_count_finite", np.isfinite(finite_float(metrics_row.get("transition_count"))), format_value(metrics_row.get("transition_count"))),
    ]
    return pd.DataFrame(gates, columns=["gate", "passed", "detail"])


def escape_tex(text: Any) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
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
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value


def tex_value(value: Any, digits: int = 4) -> str:
    if isinstance(value, str):
        return escape_tex(value)
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"
    number = finite_float(value)
    if not np.isfinite(number):
        return "--"
    if abs(number) != 0 and (abs(number) < 1e-3 or abs(number) >= 1e4):
        return f"{number:.2e}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def write_tex_table(path: Path, caption: str, label: str, columns: Sequence[Tuple[str, str]], rows: Sequence[Sequence[Any]], width: str = r"\textwidth") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    alignment = "l" + "c" * (len(columns) - 1)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\resizebox{{{width}}}{{!}}{{%",
        f"\\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(header for _, header in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        rendered = []
        for index, value in enumerate(row):
            if index == 0 and isinstance(value, str) and value.startswith("$") and value.endswith("$"):
                rendered.append(value)
            elif isinstance(value, str) and ("$" in value or "\\" in value):
                rendered.append(value)
            else:
                rendered.append(tex_value(value))
        lines.append(" & ".join(rendered) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_fragments(
    output_root: Path,
    family: pd.DataFrame,
    kinetics: pd.DataFrame,
) -> Dict[str, str]:
    latex_root = output_root / "latex"
    family_rows = []
    for row in family.itertuples(index=False):
        family_rows.append([
            row.family,
            row.free_parameters,
            format_value(row.bootstrap_mean_score),
            row.bootstrap_95_interval,
            format_value(row.difference_to_best_mean),
            row.difference_to_best_95_interval,
            row.within_one_standard_error,
            row.practically_equivalent_to_best,
            row.final_family,
        ])
    family_path = latex_root / "supplementary_table3_family_selection.tex"
    write_tex_table(
        family_path,
        "Complete family-bounded comparison. Scores and intervals use paired validation-user bootstrap resamples; lower scores indicate closer structural recovery.",
        "tab:supp_mechanism_family_selection",
        (
            ("family", "Family"),
            ("p", "$p$"),
            ("mean", "Mean score"),
            ("ci", r"95\% interval"),
            ("delta", r"$\Delta$ to best"),
            ("delta_ci", r"$\Delta$ 95\% interval"),
            ("one_se", "One-SE"),
            ("equiv", "Equivalent"),
            ("final", "Final"),
        ),
        family_rows,
    )

    kinetic_rows = []
    for row in kinetics.itertuples(index=False):
        kinetic_rows.append([
            row.macrostate,
            f"({row.training_center_M:.3f}, {row.training_center_Psi:.3f})",
            row.empirical_self_transition,
            row.mechanism_self_transition,
            row.row_total_variation,
            "Yes" if row.top_outgoing_edge_match else "No",
            row.empirical_geometric_mean_reference,
            row.mechanism_geometric_mean_reference,
            row.geometric_mean_reference_ratio,
        ])
    kinetic_path = latex_root / "supplementary_table4_statewise_kinetics.tex"
    write_tex_table(
        kinetic_path,
        "Statewise post-selection kinetic recovery under the fixed six-state empirical partition. Geometric residence quantities are transition-implied one-step references, not free-running mechanism residence distributions.",
        "tab:supp_mechanism_statewise_kinetics",
        (
            ("state", "State"),
            ("center", r"Training centre $(M,\Psi)$"),
            ("emp_p", r"$P_{ii}^{\mathrm{emp}}$"),
            ("mech_p", r"$P_{ii}^{\mathrm{mech}}$"),
            ("tv", "Row TV"),
            ("edge", "Top edge match"),
            ("emp_geo", "Emp. geometric mean"),
            ("mech_geo", "Mech. geometric mean"),
            ("ratio", "Ratio"),
        ),
        kinetic_rows,
    )
    return {
        "family_selection": str(family_path.resolve()),
        "statewise_kinetics": str(kinetic_path.resolve()),
    }



def markdown_table(frame: pd.DataFrame, columns: Optional[Sequence[str]] = None, digits: int = 4) -> str:
    selected = frame.copy() if columns is None else frame[[column for column in columns if column in frame.columns]].copy()
    for column in selected.columns:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(lambda value: "" if not np.isfinite(value) else format_value(value, digits))
    try:
        return selected.to_markdown(index=False)
    except Exception:
        return selected.to_csv(index=False)


def build_report(
    family: pd.DataFrame,
    deletion: pd.DataFrame,
    margin: pd.DataFrame,
    specification: pd.DataFrame,
    audit: pd.DataFrame,
    kinetics: pd.DataFrame,
    figure3: pd.DataFrame,
    quality: pd.DataFrame,
) -> str:
    margin_values = pd.to_numeric(margin["equivalence_margin"], errors="coerce").dropna()
    margin_range = "unavailable"
    if not margin_values.empty:
        margin_range = f"{float(margin_values.min()):.3f}--{float(margin_values.max()):.3f}"

    deletion_lines = []
    for row in deletion.itertuples(index=False):
        reduced = FAMILY_LABELS.get(row.reduced_family, row.reduced_family.replace("_", " "))
        deletion_lines.append(
            f"- {row.removed_parameter_label}: reduced family {reduced}; "
            f"difference to best {format_value(row.difference_to_best_mean)} "
            f"({row.difference_to_best_95_interval}); one-SE {row.within_one_standard_error}; "
            f"practical equivalence {row.practically_equivalent_to_best}; "
            f"globally eligible {row.eligible_under_global_rule}; retained {row.retained_by_global_rule}."
        )

    specification_lines = []
    for category in ("Active mechanism term", "Structural zero", "Fixed scale", "Accounting calibration"):
        subset = specification[specification["category"].astype(str) == category]
        if subset.empty:
            continue
        values = ", ".join(
            f"{row.symbol}={format_value(row.value)}"
            for row in subset.itertuples(index=False)
        )
        specification_lines.append(f"- {category}: {values}.")

    lines = [
        "# Minimal-mechanism Supplementary Information report",
        "",
        "## Recommended typeset package",
        "",
        "Use one short result-level note and two compact supplementary tables. The scalar-deletion and frozen-specification values remain in this numerical report rather than being typeset as separate tables. No additional mechanism figure is required because the main figures already show family selection, field recovery, transition residuals and development-to-confirmation stability.",
        "",
        "1. **Supplementary Note 2 — Family-bounded mechanism selection and frozen confirmation.** Summarise the bounded hierarchy, paired-user bootstrap rule, scalar-deletion boundary, fixed specification and output-only confirmation in approximately 250--350 words.",
        "2. **Supplementary Table 3 — Complete family comparison.** Report exact paired-bootstrap scores, intervals and parsimony eligibility for every tested family.",
        "3. **Supplementary Table 4 — Statewise post-selection kinetics.** Report the six empirical/mechanism transition rows and transition-implied geometric residence references.",
        "",
        "Expected mechanism-specific footprint: approximately 1.5--2 supplementary pages. Scalar-deletion values, the frozen specification, confirmation audit, margin sensitivity, boundary diagnostics, residence curves and development-to-confirmation metrics remain in the numerical report or machine-readable Supplementary Data rather than occupying separate typeset tables.",
        "",
        "## Interpretation boundaries",
        "",
        "- Minimality is family-bounded within the declared hierarchy and finite search domain.",
        "- The global one-standard-error plus practical-equivalence rule determines scalar retention; pairwise necessity of every coefficient is not claimed.",
        "- Transition and residence diagnostics were excluded from selection.",
        "- Mechanism residence quantities are geometric references implied by the one-step transition matrix, not autonomous residence predictions.",
        "- Confirmation values are output-only point estimates from the frozen specification.",
        "",
        "## Supplementary Table 3 — Complete family comparison",
        "",
        markdown_table(family),
        "",
        "## Core numerical summary — direct scalar deletions",
        "",
        *deletion_lines,
        "",
        f"The selected family was unchanged across practical-equivalence margins {margin_range}.",
        "",
        "## Core numerical summary — frozen specification",
        "",
        *specification_lines,
        "",
        "## Supplementary Data — practical-equivalence margin sensitivity",
        "",
        markdown_table(margin),
        "",
        "## Supplementary Data — confirmation audit",
        "",
        markdown_table(audit),
        "",
        "## Supplementary Table 4 — Statewise post-selection kinetics",
        "",
        markdown_table(kinetics),
        "",
        "## Aggregate kinetic summary",
        "",
        markdown_table(figure3),
        "",
        "## Scientific quality gates",
        "",
        markdown_table(quality),
        "",
    ]
    return "\n".join(lines)



def source_audit_table(records: Sequence[SourceRecord]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "source": record.name,
            "path": str(record.path),
            "sha256": record.sha256,
            "rows": record.rows,
            "columns": record.columns,
        }
        for record in records
    ])


def build_arg_parser() -> argparse.ArgumentParser:
    default_base = Path(os.environ.get("EDNET_KT4_OUTPUT_ROOT", "/data/datasets/KT4/outputs_KT4"))
    parser = argparse.ArgumentParser(description="Extract compact supplementary statistics for the frozen minimal mechanism.")
    parser.add_argument("--output-base", type=Path, default=default_base)
    parser.add_argument("--stage1-root", type=Path, default=None)
    parser.add_argument("--minimality-root", type=Path, default=None)
    parser.add_argument("--tuning-root", type=Path, default=None)
    parser.add_argument("--frozen-root", type=Path, default=None)
    parser.add_argument("--confirmation-root", type=Path, default=None)
    parser.add_argument("--figure-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--confirm-split", type=str, default="B_confirm")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    require(args.confirm_split == "B_confirm", "Formal supplementary extraction requires confirm_split='B_confirm'")
    roots = resolve_roots(args)
    output_root = roots["output"]
    output_root.mkdir(parents=True, exist_ok=True)

    jsons, tables, records = load_sources(roots, args.confirm_split)
    if args.strict:
        validate_contracts(jsons, tables, args.confirm_split)

    family = family_selection_table(tables)
    deletion = scalar_deletion_summary(tables)
    margin = margin_sensitivity_table(tables)
    boundary = boundary_audit_table(tables)
    specification, _ = frozen_specification_summary(jsons)
    confirmation_audit = confirmation_audit_table(jsons, tables)
    kinetics = statewise_kinetic_table(tables)
    quality = quality_gate_table(
        jsons,
        tables,
        family,
        deletion,
        margin,
        specification,
        confirmation_audit,
        kinetics,
    )
    require(bool(quality["passed"].all()), "One or more supplementary quality gates failed")

    table_root = output_root / "tables"
    outputs: Dict[str, Any] = {}
    for name, frame in (
        ("supplementary_table3_family_selection", family),
        ("supplementary_margin_sensitivity", margin),
        ("supplementary_parameter_boundary_audit", boundary),
        ("supplementary_confirmation_audit", confirmation_audit),
        ("supplementary_table4_statewise_kinetics", kinetics),
        ("supplementary_development_confirmation_stability", tables["development_confirmation_stability"]),
        ("supplementary_aggregate_kinetic_metrics", tables["figure3_metrics"]),
        ("supplementary_empirical_residence_curves", tables["empirical_residence_curves"]),
        ("supplementary_mechanism_residence_reference_curves", tables["mechanism_residence_references"]),
        ("supplementary_source_audit", source_audit_table(records)),
        ("supplementary_quality_gates", quality),
    ):
        outputs[name] = write_table(frame, table_root / name)

    latex_outputs = write_latex_fragments(output_root, family, kinetics)

    report = build_report(
        family,
        deletion,
        margin,
        specification,
        confirmation_audit,
        kinetics,
        tables["figure3_metrics"],
        quality,
    )
    report_path = output_root / "reports" / "minimal_mechanism_supplementary_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    manifest = {
        "analysis": "minimal mechanism supplementary statistics",
        "primary_macrostate": list(EXPECTED_PRIMARY_MACROSTATE),
        "selected_family": EXPECTED_FAMILY,
        "free_mechanism_parameters": list(EXPECTED_FREE_PARAMETERS),
        "fixed_structural_zeros": list(EXPECTED_ZERO_PARAMETERS),
        "fixed_macrostate_k": EXPECTED_MACROSTATE_K,
        "confirm_split": args.confirm_split,
        "figures_generated": False,
        "typeset_recommendation": {
            "supplementary_note": "Family-bounded mechanism selection and frozen confirmation",
            "tables": [
                "Supplementary Table 3: complete family comparison",
                "Supplementary Table 4: statewise post-selection kinetics",
            ],
            "core_numeric_report_sections": [
                "direct scalar deletions",
                "frozen mechanism specification",
            ],
            "estimated_pages": "1.5--2",
        },
        "interpretation_boundaries": {
            "minimality_scope": "declared family hierarchy and finite search domain",
            "pairwise_parameter_necessity_claimed": False,
            "transition_and_residence_used_for_selection": False,
            "residence_reference": "transition-implied geometric reference, not an autonomous residence prediction",
            "confirmation_mode": "output_only",
        },
        "roots": {name: str(path) for name, path in roots.items()},
        "sources": [json_safe(record.__dict__) for record in records],
        "outputs": outputs,
        "latex_outputs": latex_outputs,
        "report": str(report_path.resolve()),
        "quality_gates_passed": True,
    }
    manifest_path = output_root / "metadata" / "minimal_mechanism_supplementary_manifest.json"
    save_json(manifest, manifest_path)
    print(f"Supplementary statistics written to {output_root}")


if __name__ == "__main__":
    main()
