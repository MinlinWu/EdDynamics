#!/usr/bin/env python3
"""Post hoc development-only rerun of the mechanism-family selection under a
predefined alternative primary-score contract.

This script imports the formal Phase-1 family-ablation implementation and
reuses its model family hierarchy, finite search, paired-user bootstrap,
one-standard-error/practical-equivalence rule, direct scalar-deletion audit,
and boundary checks.  It reads only A_train and A_val.  It never writes the
formal Phase-2 handoff and is never eligible to replace the frozen model.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

SENSITIVITY_VERSION = "1.1.0"
EXPECTED_FORMAL_SCRIPT_SHA256 = "e5c023d85dc00c5ef8e89c12162b5dea8e0dfc205651f37ccacffd8e6e3a612e"
EXPECTED_FORMAL_PUBLIC_RELEASE_VERSION = "4.2.0"
DEFAULT_STAGE1_ROOT = Path("/data/datasets/KT4/outputs_KT4/stage1")
DEFAULT_OUTPUT_ROOT = Path(
    "/data/datasets/KT4/outputs_KT4/"
    "stage2_phase1_score_contract_robustness/equal_primary_rerun"
)

PRIMARY_COMPONENTS = (
    "one_step_mse_main_norm",
    "occupancy_js_MR_PsiA",
    "drift_local_rmse_loss_MR_PsiA",
    "drift_direction_loss_MR_PsiA",
    "drift_magnitude_loss_MR_PsiA",
)

FORMAL_WEIGHTS: Dict[str, float] = {
    "one_step_mse_main_norm": 0.10,
    "occupancy_js_MR_PsiA": 0.20,
    "drift_local_rmse_loss_MR_PsiA": 0.30,
    "drift_direction_loss_MR_PsiA": 0.20,
    "drift_magnitude_loss_MR_PsiA": 0.20,
}


def _renormalise(weights: Mapping[str, float]) -> Dict[str, float]:
    out = {name: float(weights.get(name, 0.0)) for name in PRIMARY_COMPONENTS}
    if any((not math.isfinite(value) or value < 0.0) for value in out.values()):
        raise ValueError("Objective weights must be finite and non-negative.")
    total = float(sum(out.values()))
    if total <= 0.0:
        raise ValueError("At least one objective weight must be positive.")
    return {name: value / total for name, value in out.items()}


OBJECTIVE_CONTRACTS: Dict[str, Dict[str, float]] = {
    "formal": _renormalise(FORMAL_WEIGHTS),
    "equal_primary": _renormalise({name: 1.0 for name in PRIMARY_COMPONENTS}),
    "omit_step": _renormalise({**FORMAL_WEIGHTS, "one_step_mse_main_norm": 0.0}),
    "omit_js": _renormalise({**FORMAL_WEIGHTS, "occupancy_js_MR_PsiA": 0.0}),
    "omit_local": _renormalise({**FORMAL_WEIGHTS, "drift_local_rmse_loss_MR_PsiA": 0.0}),
    "omit_direction": _renormalise({**FORMAL_WEIGHTS, "drift_direction_loss_MR_PsiA": 0.0}),
    "omit_magnitude": _renormalise({**FORMAL_WEIGHTS, "drift_magnitude_loss_MR_PsiA": 0.0}),
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


def import_formal_module(path: Path) -> Any:
    source = path.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Formal Phase-1 script not found: {source}")
    cache_dir = Path(tempfile.gettempdir()) / "ednet_kt4_numba_cache" / sha256_file(source)[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
    spec = importlib.util.spec_from_file_location(
        "formal_mechanism_family_ablation_for_score_contract",
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
        "FAMILIES",
        "FAMILY_BY_KEY",
        "ALL_PARAMS",
        "FIXED_NUISANCE",
        "load_phase1_panels",
        "development_panel_contract",
        "calibrate_from_A_train",
        "sample_users",
        "make_metric_cache",
        "BootstrapScorer",
        "fit_or_resume_family",
        "bootstrap_family_scores",
        "build_results_table",
        "resolve_scalar_minimality_one_run",
        "boundary_adequacy",
        "final_model_validation",
        "audit_stage1_kmeans_contract",
        "validate_grid_contract",
        "prepare_output_root",
        "configure_from_args",
        "write_csv",
        "save_json",
        "timed_stage",
    )
    missing = [name for name in required if not hasattr(p1, name)]
    if missing:
        raise RuntimeError(f"Formal Phase-1 implementation is missing required names: {missing}")
    formal_keys = tuple(p1.OBJECTIVE_WEIGHTS.keys())
    if set(formal_keys) != set(PRIMARY_COMPONENTS):
        raise RuntimeError(
            "Formal primary-component contract changed: "
            f"expected {PRIMARY_COMPONENTS}, found {formal_keys}."
        )


def configure_formal_module(p1: Any, args: argparse.Namespace, weights: Mapping[str, float]) -> None:
    namespace = SimpleNamespace(
        random_state=args.random_state,
        grid_profile=args.grid_profile,
        no_numba=args.no_numba,
        screening_train_users=args.screening_train_users,
        screening_max_candidates=args.screening_max_candidates,
        full_train_top_k=args.full_train_top_k,
        val_shortlist_k=args.val_shortlist_k,
        local_refine_max_evals=args.local_refine_max_evals,
        deletion_exhaustive_max_combinations=args.deletion_exhaustive_max_combinations,
        deletion_full_train_top_k=args.deletion_full_train_top_k,
        deletion_val_shortlist_k=args.deletion_val_shortlist_k,
        deletion_local_refine_max_evals=args.deletion_local_refine_max_evals,
        deletion_refine_starts=args.deletion_refine_starts,
        bootstrap_reps=args.bootstrap_reps,
        equivalence_margin=args.equivalence_margin,
        # The supplement fixes the formal PE margin and does not repeat the
        # manuscript's margin-sensitivity experiment.
        margin_sensitivity=str(args.equivalence_margin),
        bootstrap_engine=args.bootstrap_engine,
        verify_optimized_bootstrap=args.verify_optimized_bootstrap,
        verify_bootstrap_reps=args.verify_bootstrap_reps,
        output_root=args.output_root,
        numba_threads=args.numba_threads,
    )
    p1.configure_from_args(namespace, create_output_dirs=True)
    p1.OBJECTIVE_WEIGHTS.clear()
    p1.OBJECTIVE_WEIGHTS.update(_renormalise(weights))
    # The formal cache key contains parameters and data label but not objective
    # weights.  Clear all module-level caches before any alternative-contract
    # evaluation to prevent formal-score metrics from leaking into this rerun.
    if hasattr(p1, "_EVALUATION_CACHE"):
        p1._EVALUATION_CACHE.clear()
    if hasattr(p1, "_BOOTSTRAP_VERIFICATION_ROWS"):
        p1._BOOTSTRAP_VERIFICATION_ROWS.clear()
    if hasattr(p1, "RUNTIME_PROFILE"):
        p1.RUNTIME_PROFILE.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a post hoc, development-only mechanism-family rerun under a "
            "predefined alternative score contract."
        )
    )
    parser.add_argument("--formal-script", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--objective-contract",
        choices=("equal_primary",),
        default="equal_primary",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--grid-profile", choices=("publication", "compact"), default="publication")
    parser.add_argument("--screening-train-users", type=int, default=20000)
    parser.add_argument("--screening-max-candidates", type=int, default=96)
    parser.add_argument("--full-train-top-k", type=int, default=16)
    parser.add_argument("--val-shortlist-k", type=int, default=8)
    parser.add_argument("--local-refine-max-evals", type=int, default=48)
    parser.add_argument("--deletion-exhaustive-max-combinations", type=int, default=5000)
    parser.add_argument("--deletion-full-train-top-k", type=int, default=32)
    parser.add_argument("--deletion-val-shortlist-k", type=int, default=16)
    parser.add_argument("--deletion-local-refine-max-evals", type=int, default=96)
    parser.add_argument("--deletion-refine-starts", type=int, default=5)
    parser.add_argument("--bootstrap-reps", type=int, default=300)
    parser.add_argument("--equivalence-margin", type=float, default=0.02)
    parser.add_argument("--bootstrap-engine", choices=("optimized", "reference"), default="optimized")
    parser.add_argument(
        "--verify-optimized-bootstrap",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--verify-bootstrap-reps", type=int, default=2)
    parser.add_argument("--numba-threads", type=int, default=0)
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-users-per-split", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-missing-kmeans-contract", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _write_empty_or_frame(p1: Any, frame: pd.DataFrame, path: Path, columns: Iterable[str]) -> None:
    if frame.empty:
        frame = pd.DataFrame(columns=list(columns))
    p1.write_csv(frame, path)


def main() -> None:
    args = parse_args()
    formal_script = args.formal_script.resolve()
    p1 = import_formal_module(formal_script)
    validate_formal_module(p1)
    formal_script_sha = sha256_file(formal_script)
    if formal_script_sha != EXPECTED_FORMAL_SCRIPT_SHA256:
        raise RuntimeError(
            "The supplied formal Phase-1 script differs from the reviewed implementation: "
            f"expected {EXPECTED_FORMAL_SCRIPT_SHA256}, found {formal_script_sha}."
        )
    if str(getattr(p1, "PUBLIC_RELEASE_VERSION", "")) != EXPECTED_FORMAL_PUBLIC_RELEASE_VERSION:
        raise RuntimeError(
            "Unexpected formal Phase-1 public release version: "
            f"{getattr(p1, 'PUBLIC_RELEASE_VERSION', None)!r}."
        )
    weights = OBJECTIVE_CONTRACTS[args.objective_contract]

    if args.self_test:
        # The formal synthetic test also verifies optimized versus row-level
        # paired-user bootstrap equivalence under the currently active weights.
        args.output_root = Path(tempfile.mkdtemp(prefix="mech_score_contract_selftest_"))
        configure_formal_module(p1, args, weights)
        p1.run_self_test()
        print("score-contract rerun self-test passed")
        return

    p1.prepare_output_root(args.output_root, resume=args.resume, overwrite=args.overwrite)
    configure_formal_module(p1, args, weights)
    grid_contract = p1.validate_grid_contract()
    start = time.time()
    wrapper_path = Path(__file__).resolve()

    print(f"[score-contract-rerun] started {now_string()}", flush=True)
    print(f"[score-contract-rerun] formal script: {formal_script}", flush=True)
    print(f"[score-contract-rerun] objective contract: {args.objective_contract}", flush=True)
    print(f"[score-contract-rerun] weights: {weights}", flush=True)
    print("[score-contract-rerun] sensitivity-only; B_confirm is not read", flush=True)

    with p1.timed_stage("load_and_prepare_development_panels"):
        kmeans_contract = p1.audit_stage1_kmeans_contract(
            args.stage1_root.resolve(),
            allow_missing=args.allow_missing_kmeans_contract,
        )
        train, val, eta, load_manifest = p1.load_phase1_panels(args.stage1_root.resolve())
        if args.smoke_test:
            train = p1.sample_users(train, args.smoke_users_per_split, args.random_state + 1)
            val = p1.sample_users(val, args.smoke_users_per_split, args.random_state + 2)
        if train.empty or val.empty:
            raise RuntimeError("A_train or A_val is empty after Phase-1 filtering.")
        panel_contract = p1.development_panel_contract(train, val)
        calibration = p1.calibrate_from_A_train(
            train,
            eta,
            p1.TAU_RESPONSE_DAYS,
            p1.TAU_ACTIVITY_DAYS,
        )
        screen_train = p1.sample_users(train, p1.SCREENING_TRAIN_USERS, args.random_state + 11)
        train_cache = p1.make_metric_cache(train, "A_train_full")
        val_cache = p1.make_metric_cache(val, "A_val_full")
        screen_cache = p1.make_metric_cache(screen_train, "A_train_screen")
        scorer = p1.BootstrapScorer(val_cache)
        del train, val, screen_train
        gc.collect()

    decision_bootstrap_seed = args.random_state + p1.DECISION_BOOTSTRAP_SEED_OFFSET

    run_contract = {
        "sensitivity_version": SENSITIVITY_VERSION,
        "formal_script": str(formal_script),
        "formal_script_sha256": formal_script_sha,
        "formal_public_release_version": str(p1.PUBLIC_RELEASE_VERSION),
        "wrapper_script": str(wrapper_path),
        "wrapper_script_sha256": sha256_file(wrapper_path),
        "experiment_scope": "post hoc score-contract sensitivity on A_train/A_val only",
        "sensitivity_only": True,
        "eligible_for_phase2_freeze": False,
        "B_confirm_read": False,
        "objective_contract": args.objective_contract,
        "objective_weights": weights,
        "objective_sanity_limits": p1.OBJECTIVE_SANITY_LIMITS,
        "fixed_nuisance_scales": dict(p1.FIXED_NUISANCE),
        "sanity_penalty_weight": p1.CONFIG_SANITY_PENALTY_WEIGHT,
        "identity_regularization_weight": p1.CONFIG_IDENTITY_REG_WEIGHT,
        "distribution_loss_max_rows": int(p1.CONFIG_DISTRIBUTION_LOSS_MAX_ROWS),
        "signed_gain_quantile": float(p1.CONFIG_SIGNED_GAIN_QUANTILE),
        "stage1_root": str(args.stage1_root.resolve()),
        "stage1_fixed_k6_contract": kmeans_contract,
        "development_panel_contract": panel_contract,
        "random_state": args.random_state,
        "grid_profile": p1.GRID_PROFILE,
        "grid_contract": grid_contract,
        "search_grid": p1.GRID,
        "pilot_anchors": p1.PILOT_ANCHORS,
        "families": [dataclasses.asdict(family) for family in p1.FAMILIES],
        "search_budgets": {
            "screening_train_users": p1.SCREENING_TRAIN_USERS,
            "screening_max_candidates": p1.SCREENING_MAX_CANDIDATES_PER_FAMILY,
            "full_train_top_k": p1.FULL_TRAIN_TOP_K,
            "validation_shortlist_k": p1.VAL_SHORTLIST_K,
            "local_refine_max_evals": p1.LOCAL_REFINE_MAX_EVALS,
            "deletion_exhaustive_max_combinations": p1.DELETION_EXHAUSTIVE_MAX_COMBINATIONS,
            "deletion_full_train_top_k": p1.DELETION_FULL_TRAIN_TOP_K,
            "deletion_validation_shortlist_k": p1.DELETION_VAL_SHORTLIST_K,
            "deletion_local_refine_max_evals": p1.DELETION_LOCAL_REFINE_MAX_EVALS,
            "deletion_refine_starts": p1.DELETION_REFINE_STARTS,
        },
        "bootstrap_reps": p1.BOOTSTRAP_REPS,
        "decision_bootstrap_seed_offset": int(p1.DECISION_BOOTSTRAP_SEED_OFFSET),
        "decision_bootstrap_seed": int(decision_bootstrap_seed),
        "equivalence_margin": p1.PRACTICAL_EQ_MARGIN,
        "margin_sensitivity_values": [float(p1.PRACTICAL_EQ_MARGIN)],
        "bootstrap_engine": p1.BOOTSTRAP_ENGINE,
        "verify_optimized_bootstrap": bool(p1.VERIFY_OPTIMIZED_BOOTSTRAP),
        "verify_bootstrap_reps": int(p1.VERIFY_BOOTSTRAP_REPS),
        "calibration": dataclasses.asdict(calibration),
    }
    run_hash = stable_json_hash(run_contract)

    fits: Dict[str, Any] = {}
    with p1.timed_stage("fit_prespecified_families"):
        for index, spec in enumerate(p1.FAMILIES):
            fit = p1.fit_or_resume_family(
                spec,
                screen_cache,
                train_cache,
                val_cache,
                calibration,
                args.random_state + 100 * (index + 1),
                run_hash,
                args.resume,
            )
            fits[spec.key] = fit
            print(
                f"[score-contract-rerun] {spec.key}: "
                f"validation primary={fit.val_metrics.get('objective_primary_score', np.nan):.6f}",
                flush=True,
            )

    with p1.timed_stage("initial_paired_user_bootstrap"):
        initial_predictions = {key: fit.val_prediction for key, fit in fits.items()}
        initial_boot = p1.bootstrap_family_scores(
            scorer,
            initial_predictions,
            p1.BOOTSTRAP_REPS,
            decision_bootstrap_seed,
        )
        initial_results, initial_best, _, initial_selected = p1.build_results_table(
            fits,
            initial_boot,
            p1.PRACTICAL_EQ_MARGIN,
        )

    with p1.timed_stage("global_scalar_minimality_fixed_point"):
        (
            final_fit,
            fits,
            deletion_records,
            final_scalar_audit,
            deletion_summary,
            final_boot,
            final_results,
            best_key,
            one_se_threshold,
            selected_after_deletion,
        ) = p1.resolve_scalar_minimality_one_run(
            fits,
            initial_selected,
            screen_cache,
            train_cache,
            val_cache,
            calibration,
            scorer,
            args.random_state + 4040,
            decision_bootstrap_seed,
        )

    final_key = final_fit.spec.key
    scalar_ok = bool(deletion_summary.get("scalar_parameter_minimality_confirmed", False))
    search_ok, boundary_df, boundary_blockers, boundary_next_tests = p1.boundary_adequacy(
        fits,
        final_results,
        final_key,
        p1.PRACTICAL_EQ_MARGIN,
        val_cache,
        calibration,
    )
    final_validation = p1.final_model_validation(final_results, final_key, best_key)
    final_equiv_ok = bool(
        final_validation.get("within_one_standard_error", False)
        and final_validation.get("practically_equivalent_to_best", False)
    )
    final_selected_by_parsimony = final_key == selected_after_deletion
    persistence_rows = final_results.loc[
        final_results["family_key"] == "persistence",
        "Practically equivalent to best",
    ]
    baselines_beaten = bool(persistence_rows.empty or not bool(persistence_rows.iloc[0]))
    would_pass_selection_gates = bool(
        search_ok
        and scalar_ok
        and baselines_beaten
        and final_equiv_ok
        and final_selected_by_parsimony
    )

    next_required_tests = list(boundary_next_tests)
    if not scalar_ok:
        next_required_tests.append({
            "reason": "scalar_minimality_fixed_point_not_confirmed",
            "family_key": final_key,
        })
    if not final_equiv_ok:
        next_required_tests.append({
            "reason": "final_family_not_eligible_against_contract_best",
            "family_key": final_key,
            "best_family_key": best_key,
        })
    if not baselines_beaten:
        next_required_tests.append({
            "reason": "persistence_baseline_practically_equivalent_to_best",
            "family_key": "persistence",
        })

    final_results = final_results.copy()
    final_results["Final scalar-minimal family"] = final_results["family_key"] == final_key
    final_results["Objective contract"] = args.objective_contract
    final_results["Sensitivity only"] = True

    tables = p1.TABLE_DIR
    metadata = p1.META_DIR
    with p1.timed_stage("write_sensitivity_outputs"):
        p1.write_csv(final_results, tables / "model_family_results.csv")
        p1.write_csv(final_boot, tables / "model_family_bootstrap_scores.csv.gz")
        _write_empty_or_frame(
            p1,
            deletion_records,
            tables / "selected_model_parameter_deletions.csv",
            ("round", "current_family", "tested_removed_parameter", "deletion_family"),
        )
        _write_empty_or_frame(
            p1,
            final_scalar_audit,
            tables / "global_scalar_deletion_audit.csv",
            ("round", "current_family", "tested_removed_parameter", "deletion_family"),
        )
        _write_empty_or_frame(
            p1,
            boundary_df,
            tables / "parameter_grid_boundaries.csv",
            ("family_key", "parameter", "value", "boundary", "blocking_for_freeze"),
        )
        p1.write_csv(
            pd.DataFrame(next_required_tests),
            tables / "next_required_tests.csv",
        )
        verification_df = pd.DataFrame(getattr(p1, "_BOOTSTRAP_VERIFICATION_ROWS", []))
        if not verification_df.empty:
            p1.write_csv(
                verification_df,
                tables / "optimized_bootstrap_equivalence_checks.csv",
            )

        summary_row = {
            "objective_contract": args.objective_contract,
            **{f"weight_{name}": weights[name] for name in PRIMARY_COMPONENTS},
            "initial_best_family": initial_best,
            "initial_selected_family": initial_selected,
            "final_best_family": best_key,
            "final_selected_family": final_key,
            "final_parameter_count": final_fit.spec.parameter_count,
            "one_se_threshold": one_se_threshold,
            "scalar_minimality_confirmed": scalar_ok,
            "search_adequacy_confirmed": search_ok,
            "persistence_baseline_excluded": baselines_beaten,
            "final_family_within_one_se": bool(
                final_validation.get("within_one_standard_error", False)
            ),
            "final_family_practically_equivalent": bool(
                final_validation.get("practically_equivalent_to_best", False)
            ),
            "final_family_eligible_against_best": final_equiv_ok,
            "final_family_selected_by_parsimony": final_selected_by_parsimony,
            "would_pass_selection_gates_under_contract": would_pass_selection_gates,
            "would_satisfy_current_phase2_family_contract": bool(
                would_pass_selection_gates
                and final_key == "offset_dual_channel"
                and set(final_fit.spec.free_params) == {"theta0", "thetaM", "phi0", "deltaS"}
            ),
            "sensitivity_only": True,
            "eligible_for_phase2_freeze": False,
            "B_confirm_read": False,
        }
        p1.write_csv(
            pd.DataFrame([summary_row]),
            tables / "objective_contract_rerun_summary.csv",
        )

        sensitivity_summary = {
            "created_at": now_string(),
            "objective_contract": args.objective_contract,
            "objective_weights": weights,
            "initial_best_family": initial_best,
            "initial_selected_family": initial_selected,
            "final_best_family": best_key,
            "final_selected_family": final_key,
            "final_selected_family_label": final_fit.spec.label,
            "final_free_mechanism_parameters": list(final_fit.spec.free_params),
            "final_selected_parameters": final_fit.selected_params,
            "one_se_threshold": one_se_threshold,
            "scalar_parameter_minimality_confirmed": scalar_ok,
            "search_adequacy_confirmed": search_ok,
            "baseline_not_practically_equivalent_to_best": baselines_beaten,
            "final_model_validation_against_best": final_validation,
            "final_model_selected_by_parsimony_rule": final_selected_by_parsimony,
            "would_pass_selection_gates_under_contract": would_pass_selection_gates,
            "would_satisfy_current_phase2_family_contract": bool(
                would_pass_selection_gates
                and final_key == "offset_dual_channel"
                and set(final_fit.spec.free_params) == {"theta0", "thetaM", "phi0", "deltaS"}
            ),
            "ready_for_phase2_freeze": False,
            "eligible_for_phase2_freeze": False,
            "sensitivity_only": True,
            "post_hoc": True,
            "B_confirm_read": False,
            "B_confirm_policy": "not read or used",
            "boundary_blockers": boundary_blockers,
            "next_required_tests": next_required_tests,
            "formal_phase1_outputs_modified": False,
            "formal_frozen_model_modified": False,
        }
        save_json(
            sensitivity_summary,
            metadata / "score_contract_sensitivity_summary.json",
        )

        manifest = {
            **run_contract,
            "created_at": now_string(),
            "output_root": str(args.output_root.resolve()),
            "run_contract_sha256": run_hash,
            "formal_selection_rule_reused": (
                "same paired-user bootstrap, one-standard-error and practical-equivalence "
                "rule, scalar deletion, and boundary audit"
            ),
            "formal_phase1_outputs_modified": False,
            "formal_phase2_or_phase3_invoked": False,
            "guardrails": {
                "B_confirm_read": False,
                "eligible_for_phase2_freeze": False,
                "formal_handoff_written": False,
                "formal_family_replaced": False,
                "formal_parameters_replaced": False,
                "sensitivity_output_root_isolated": True,
            },
            "results": sensitivity_summary,
            "runtime_profile": list(getattr(p1, "RUNTIME_PROFILE", [])),
            "elapsed_seconds": float(time.time() - start),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
        manifest_path = metadata / "score_contract_rerun_manifest.json"
        save_json(manifest, manifest_path)
        save_json(
            {
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "formal_script_sha256": sha256_file(formal_script),
                "wrapper_script_sha256": sha256_file(wrapper_path),
            },
            metadata / "score_contract_rerun_manifest.sha256.json",
        )

    forbidden_handoff = p1.META_DIR / "phase1_minimal_mechanism_handoff.json"
    if forbidden_handoff.exists():
        raise RuntimeError(
            "Sensitivity rerun wrote the formal Phase-2 handoff filename; "
            "this violates the isolation guardrail."
        )

    print("[score-contract-rerun] completed", flush=True)
    print(f"[score-contract-rerun] final family: {final_key}", flush=True)
    print(
        "[score-contract-rerun] would_pass_selection_gates_under_contract="
        f"{would_pass_selection_gates}",
        flush=True,
    )
    print("[score-contract-rerun] eligible_for_phase2_freeze=False", flush=True)
    print(f"[score-contract-rerun] outputs: {args.output_root.resolve()}", flush=True)
    print(f"[score-contract-rerun] elapsed seconds: {time.time() - start:.1f}", flush=True)


if __name__ == "__main__":
    main()
