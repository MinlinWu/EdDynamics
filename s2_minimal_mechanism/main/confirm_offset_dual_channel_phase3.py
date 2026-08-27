#!/usr/bin/env python3
"""Evaluate the frozen offset dual-channel mechanism on B_confirm without updates."""

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
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

EXPECTED_FAMILY = "offset_dual_channel"
EXPECTED_FREE_PARAMS = ("theta0", "thetaM", "phi0", "deltaS")
EXPECTED_ZERO_PARAMS = ("thetaPsi", "thetaMPsi", "phiPsi")
EXPECTED_NUISANCE_PARAMS = ("lambdaR", "lambdaA", "lambdaI")
EXPECTED_PRIMARY_MACROSTATE = ("M", "Psi")
DEFAULT_FROZEN_MANIFEST = (
    "/data/datasets/KT4/outputs_KT4/stage2_phase2_freeze/"
    "metadata/phase2_frozen_model_manifest.json"
)
DEFAULT_OUTPUT_ROOT = "/data/datasets/KT4/outputs_KT4/stage2_phase3_confirm"
DEFAULT_CONFIRM_SPLIT = "B_confirm"

STABILITY_METRICS = (
    "objective_primary_score",
    "objective_loss",
    "one_step_mse_main_norm",
    "one_step_rmse_M",
    "one_step_rmse_Psi",
    "occupancy_js_MR_PsiA",
    "drift_vector_corr_MR_PsiA",
    "drift_local_rmse_loss_MR_PsiA",
    "drift_direction_loss_MR_PsiA",
    "drift_magnitude_loss_MR_PsiA",
)
HIGHER_IS_BETTER = {"drift_vector_corr_MR_PsiA"}


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
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(obj), handle, indent=2, ensure_ascii=False, allow_nan=False)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_runtime_cache(phase1_script: Path) -> None:
    version = sha256_file(phase1_script.resolve())[:16]
    cache_dir = Path(tempfile.gettempdir()) / "ednet_kt4_numba_cache" / version
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)


def import_phase1_module(path: Path) -> Any:
    source = path.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Frozen Phase-1 implementation not found: {source}")
    spec = importlib.util.spec_from_file_location(
        "tune_offset_dual_channel_phase1",
        str(source),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import frozen Phase-1 implementation: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def validate_phase1_module_contract(p1: Any) -> None:
    required = (
        "read_core_panel",
        "stage1_dynamics_root",
        "prepare_panel",
        "Calibration",
        "make_metric_cache",
        "simulate_arrays",
        "structure_metrics_fast_no_regions",
        "objective_from_metrics",
        "objective_diagnostics",
        "primary_objective_score",
        "prediction_frame_from_cache",
        "field_grid_table",
        "audit_stage1_kmeans_contract",
        "write_table",
        "PARAM_NAMES",
        "PARAM_GRID_VALUES",
        "PARAM_DEFAULTS",
        "PARAM_BOUNDS",
        "TERM_SWITCHES",
    )
    missing = [name for name in required if not hasattr(p1, name)]
    if missing:
        raise RuntimeError(f"Frozen Phase-1 implementation is missing required names: {missing}")


def verify_manifest_checksum(path: Path, require_checksum: bool) -> Dict[str, Any]:
    actual = sha256_file(path)
    candidates = (
        path.with_suffix(".sha256.json"),
        path.parent / "phase2_frozen_model_manifest.sha256.json",
    )
    expected = None
    source = None
    for candidate in candidates:
        if candidate.exists():
            payload = load_json(candidate)
            expected = str(payload.get("manifest_sha256", "") or "")
            source = str(candidate.resolve())
            break
    if expected:
        if expected != actual:
            raise RuntimeError(
                f"Frozen manifest checksum mismatch: expected {expected}, found {actual}."
            )
        return {
            "manifest_sha256": actual,
            "checksum_source": source,
            "checksum_verified": True,
        }
    if require_checksum:
        raise RuntimeError(f"No checksum sidecar found for {path}")
    return {
        "manifest_sha256": actual,
        "checksum_source": None,
        "checksum_verified": False,
    }


def validate_frozen_manifest(
    manifest: Mapping[str, Any],
) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, Any]]:
    if tuple(manifest.get("primary_macrostate", [])) != EXPECTED_PRIMARY_MACROSTATE:
        raise RuntimeError("Frozen manifest primary macrostate is not ['M', 'Psi'].")
    guardrails = dict(manifest.get("guardrails", {}))
    required_false = (
        "B_confirm_read",
        "parameter_search_opened",
        "mechanism_family_reselected",
        "mechanism_parameters_refit",
        "kmeans_refit",
        "macrostate_k_selected",
        "region_redefinition",
    )
    failed = [name for name in required_false if bool(guardrails.get(name, False))]
    if failed:
        raise RuntimeError(f"Frozen manifest violates Phase-2 guardrails: {failed}")
    frozen = dict(manifest.get("frozen_parameters", {}))
    if str(frozen.get("family_key", "")) != EXPECTED_FAMILY:
        raise RuntimeError(f"Unexpected frozen family: {frozen.get('family_key')!r}")
    free = tuple(str(name) for name in frozen.get("free_mechanism_parameters", []))
    if set(free) != set(EXPECTED_FREE_PARAMS):
        raise RuntimeError(f"Unexpected frozen free parameters: {free}")
    vector = dict(frozen.get("full_parameter_vector", {}))
    if not vector:
        vector = {
            **dict(frozen.get("mechanism_parameters", {})),
            **dict(frozen.get("structural_zero_values", {})),
            **dict(frozen.get("fixed_nuisance_scales", {})),
        }
    required = EXPECTED_FREE_PARAMS + EXPECTED_ZERO_PARAMS + EXPECTED_NUISANCE_PARAMS
    missing = [name for name in required if name not in vector]
    if missing:
        raise RuntimeError(f"Frozen parameter vector is missing: {missing}")
    params = {name: float(vector[name]) for name in required}
    for name in EXPECTED_ZERO_PARAMS:
        if abs(params[name]) > 1e-12:
            raise RuntimeError(f"Frozen structural-zero parameter is non-zero: {name}")
    expected_hash = str(frozen.get("parameter_hash", "") or "")
    actual_hash = stable_json_hash({name: params[name] for name in required})
    if expected_hash and expected_hash != actual_hash:
        raise RuntimeError("Frozen parameter hash does not match the parameter vector.")
    calibration = dict(manifest.get("frozen_calibration", {}))
    required_calibration = (
        "eta",
        "tau_response_days",
        "tau_activity_days",
        "residual_mass_per_answer",
        "lambda_E",
        "response_signed_gain",
        "alignment_signed_gain",
        "sigma_U0",
        "sigma_Psi0",
    )
    missing_calibration = [name for name in required_calibration if name not in calibration]
    if missing_calibration:
        raise RuntimeError(f"Frozen calibration is missing: {missing_calibration}")
    expected_calibration_hash = str(manifest.get("calibration_hash", "") or "")
    actual_calibration_hash = stable_json_hash(calibration)
    if expected_calibration_hash and expected_calibration_hash != actual_calibration_hash:
        raise RuntimeError("Frozen calibration hash does not match the calibration payload.")
    return params, calibration, frozen


def resolve_phase1_script(
    override: Optional[Path],
    manifest: Mapping[str, Any],
) -> Tuple[Path, Dict[str, Any]]:
    implementation = dict(manifest.get("phase1_implementation", {}))
    expected_sha = str(implementation.get("frozen_copy_sha256", "") or implementation.get("source_sha256", ""))
    if override is not None:
        path = override.resolve()
        source = "command_line"
    else:
        raw = str(implementation.get("frozen_copy_path", "") or implementation.get("source_path", ""))
        if not raw:
            raise RuntimeError("Frozen manifest does not identify the Phase-1 implementation.")
        path = Path(raw).resolve()
        source = "phase2_frozen_manifest"
    if not path.exists():
        raise FileNotFoundError(f"Phase-1 implementation does not exist: {path}")
    actual_sha = sha256_file(path)
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError(
            f"Phase-1 implementation checksum mismatch: expected {expected_sha}, found {actual_sha}."
        )
    return path, {
        "path": str(path),
        "source": source,
        "sha256": actual_sha,
        "matches_phase2_freeze": True,
    }


def configure_frozen_phase1_module(p1: Any, params: Mapping[str, float]) -> None:
    p1.TERM_SWITCHES.clear()
    p1.TERM_SWITCHES.update({
        "theta0": True,
        "thetaPsi": False,
        "thetaMPsi": False,
        "deltaS": True,
        "phiPsi": False,
    })
    for name in p1.PARAM_NAMES:
        value = float(params[name])
        p1.PARAM_DEFAULTS[name] = value
        p1.PARAM_GRID_VALUES[name] = [value]
    p1.PARAM_BOUNDS = np.asarray(
        [[float(params[name]), float(params[name])] for name in p1.PARAM_NAMES],
        dtype=float,
    )


def calibration_from_manifest(p1: Any, payload: Mapping[str, Any]) -> Any:
    fields = {field.name for field in dataclasses.fields(p1.Calibration)}
    kwargs = {
        name: float(payload[name])
        for name in fields
        if name in payload and payload[name] is not None
    }
    return p1.Calibration(**kwargs)


def compare_kmeans_contracts(
    phase2_contract: Mapping[str, Any],
    current_contract: Mapping[str, Any],
) -> None:
    for name in ("metadata_sha256", "centers_sha256"):
        if str(phase2_contract.get(name, "")) != str(current_contract.get(name, "")):
            raise RuntimeError(f"Stage-1 fixed-K contract changed after Phase 2: {name}")


def load_confirm_panel(
    p1: Any,
    stage1_root: Path,
    split: str,
    eta: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if split != DEFAULT_CONFIRM_SPLIT:
        raise RuntimeError(f"Formal Phase 3 requires split {DEFAULT_CONFIRM_SPLIT!r}, found {split!r}.")
    dynamics = p1.stage1_dynamics_root(stage1_root)
    raw = p1.read_core_panel(dynamics, split)
    if raw.empty:
        raise RuntimeError(f"{split} Stage-1 panel is empty.")
    panel = p1.prepare_panel(raw, split, eta)
    if panel.empty:
        raise RuntimeError(f"{split} is empty after confirmation validity filtering.")
    return panel, {
        "stage1_root": str(stage1_root.resolve()),
        "stage1_dynamics_root": str(Path(dynamics).resolve()),
        "confirm_split": split,
        "raw_rows": int(len(raw)),
        "valid_rows": int(len(panel)),
        "valid_users": int(panel["user_id"].nunique()),
        "eta_source": "Phase-2 frozen calibration",
        "calibration_reestimated": False,
    }


def panel_fingerprint(panel: pd.DataFrame, label: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "label": label,
        "rows": int(len(panel)),
        "users": int(panel["user_id"].nunique()) if "user_id" in panel.columns else None,
    }
    if panel.empty:
        return out
    if "bundle_step_index" in panel.columns:
        out["bundle_step_min"] = int(panel["bundle_step_index"].min())
        out["bundle_step_max"] = int(panel["bundle_step_index"].max())
    columns = [name for name in ("user_id", "bundle_step_index") if name in panel.columns]
    if columns:
        sample = (
            panel.sort_values(columns, kind="mergesort")[columns]
            .head(100000)
            .to_numpy(dtype=np.int64, copy=True)
        )
        out["head_identifier_sample_sha256"] = hashlib.sha256(sample.tobytes()).hexdigest()
    return out


def evaluate_panel(
    p1: Any,
    panel: pd.DataFrame,
    params: Mapping[str, float],
    calibration: Any,
    label: str,
) -> Tuple[Dict[str, Any], Any, Any]:
    cache = p1.make_metric_cache(panel, [], label)
    simulation = p1.simulate_arrays(cache, dict(params), calibration)
    metrics = p1.structure_metrics_fast_no_regions(cache, simulation, label)
    metrics["objective_loss"] = float(p1.objective_from_metrics(metrics))
    metrics.update(p1.objective_diagnostics(metrics))
    metrics["objective_primary_score"] = float(p1.primary_objective_score(metrics))
    return metrics, cache, simulation


def build_metric_stability_table(
    confirmation: Mapping[str, Any],
    development_reference: Mapping[str, Any],
) -> pd.DataFrame:
    rows = []
    for reference_label in ("A_val", "A_train_plus_A_val", "A_train"):
        reference = development_reference.get(reference_label, {})
        if not isinstance(reference, Mapping) or not reference:
            continue
        for metric in STABILITY_METRICS:
            try:
                confirmation_value = float(confirmation.get(metric, np.nan))
                development_value = float(reference.get(metric, np.nan))
            except Exception:
                confirmation_value = np.nan
                development_value = np.nan
            if np.isfinite(confirmation_value) and np.isfinite(development_value):
                delta = confirmation_value - development_value
                relative_delta = delta / max(abs(development_value), 1e-12)
                degradation = (
                    development_value - confirmation_value
                    if metric in HIGHER_IS_BETTER
                    else confirmation_value - development_value
                )
            else:
                delta = np.nan
                relative_delta = np.nan
                degradation = np.nan
            rows.append({
                "reference_label": reference_label,
                "confirm_label": str(confirmation.get("label", DEFAULT_CONFIRM_SPLIT)),
                "metric": metric,
                "metric_direction": "higher_is_better" if metric in HIGHER_IS_BETTER else "lower_is_better",
                "development_value": development_value,
                "confirmation_value": confirmation_value,
                "confirmation_minus_development": delta,
                "relative_delta_vs_development": relative_delta,
                "diagnostic_degradation_amount": degradation,
            })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run output-only B_confirm evaluation for the frozen offset dual-channel mechanism."
    )
    parser.add_argument("--frozen-manifest", type=Path, default=Path(os.environ.get("MECH_PHASE3_FROZEN_MANIFEST", DEFAULT_FROZEN_MANIFEST)))
    parser.add_argument("--phase1-script", type=Path, default=None)
    parser.add_argument("--stage1-root", type=Path, default=None)
    parser.add_argument("--confirm-split", type=str, default=os.environ.get("MECH_PHASE3_CONFIRM_SPLIT", DEFAULT_CONFIRM_SPLIT))
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("MECH_PHASE3_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)))
    parser.add_argument("--audit-sample-rows", type=int, default=int(os.environ.get("MECH_PHASE3_AUDIT_SAMPLE_ROWS", "200000")))
    parser.add_argument("--write-full-predictions", action="store_true", default=bool(int(os.environ.get("MECH_PHASE3_WRITE_FULL_PREDICTIONS", "0"))))
    parser.add_argument("--require-manifest-checksum", action="store_true", default=bool(int(os.environ.get("MECH_PHASE3_REQUIRE_MANIFEST_CHECKSUM", "1"))))
    parser.add_argument("--random-state", type=int, default=int(os.environ.get("MECH_PHASE3_RANDOM_STATE", "42")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    output_root = args.output_root.resolve()
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    table_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    frozen_manifest_path = args.frozen_manifest.resolve()
    if not frozen_manifest_path.exists():
        raise FileNotFoundError(f"Phase-2 frozen manifest not found: {frozen_manifest_path}")
    checksum = verify_manifest_checksum(
        frozen_manifest_path,
        args.require_manifest_checksum,
    )
    frozen_manifest = load_json(frozen_manifest_path)
    params, calibration_payload, frozen_parameters = validate_frozen_manifest(frozen_manifest)
    phase1_script, phase1_implementation_audit = resolve_phase1_script(
        args.phase1_script,
        frozen_manifest,
    )

    prepare_runtime_cache(phase1_script)
    p1 = import_phase1_module(phase1_script)
    validate_phase1_module_contract(p1)
    configure_frozen_phase1_module(p1, params)
    calibration = calibration_from_manifest(p1, calibration_payload)

    if args.stage1_root is not None:
        stage1_root = args.stage1_root.resolve()
        stage1_root_source = "command_line"
    else:
        raw = str(frozen_manifest.get("stage1_root", "") or "")
        if not raw:
            raise RuntimeError("Phase-2 manifest does not identify the Stage-1 root.")
        stage1_root = Path(raw).resolve()
        stage1_root_source = "phase2_frozen_manifest"
    if not stage1_root.exists():
        raise FileNotFoundError(f"Stage-1 root does not exist: {stage1_root}")

    current_kmeans_audit = p1.audit_stage1_kmeans_contract(stage1_root)
    compare_kmeans_contracts(
        dict(frozen_manifest.get("stage1_fixed_k6_contract", {})),
        current_kmeans_audit,
    )
    save_json(current_kmeans_audit, metadata_root / "stage1_fixed_k6_contract.json")

    print(f"[phase3] started {now_string()}")
    panel, load_manifest = load_confirm_panel(
        p1,
        stage1_root,
        args.confirm_split,
        float(calibration.eta),
    )

    parameter_hash_before = stable_json_hash(params)
    calibration_hash_before = stable_json_hash(dataclasses.asdict(calibration))
    metrics, cache, simulation = evaluate_panel(
        p1,
        panel,
        params,
        calibration,
        args.confirm_split,
    )
    parameter_hash_after = stable_json_hash(params)
    calibration_hash_after = stable_json_hash(dataclasses.asdict(calibration))
    if parameter_hash_before != parameter_hash_after:
        raise RuntimeError("Frozen parameters changed during confirmation.")
    if calibration_hash_before != calibration_hash_after:
        raise RuntimeError("Frozen calibration changed during confirmation.")

    metrics_path = p1.write_table(
        pd.DataFrame([metrics]),
        table_root / f"phase3_{args.confirm_split}_structural_alignment_metrics",
    )
    stability = build_metric_stability_table(
        metrics,
        frozen_manifest.get("development_metric_reference", {}),
    )
    stability_path = p1.write_table(
        stability,
        table_root / "phase3_development_vs_confirmation_metric_stability",
    )
    field_grid_path = p1.write_table(
        p1.field_grid_table(cache, simulation, args.confirm_split),
        table_root / f"phase3_{args.confirm_split}_field_grid",
    )
    prediction_audit = p1.prediction_frame_from_cache(
        cache,
        simulation,
        args.confirm_split,
        args.audit_sample_rows,
        args.random_state + 300,
    )
    audit_path = p1.write_table(
        prediction_audit,
        table_root / f"phase3_{args.confirm_split}_prediction_audit_sample",
    )

    full_prediction_path = None
    if args.write_full_predictions:
        full_predictions = p1.prediction_frame_from_cache(
            cache,
            simulation,
            args.confirm_split,
            0,
            args.random_state,
        )
        full_prediction_path = str(
            p1.write_table(
                full_predictions,
                table_root / f"phase3_{args.confirm_split}_full_predictions",
            )
        )

    no_update_audit = {
        "parameter_search_opened": False,
        "calibration_reestimated": False,
        "mechanism_family_reselected": False,
        "mechanism_parameters_refit": False,
        "kmeans_refit": False,
        "macrostate_k_selected": False,
        "region_redefinition": False,
        "B_confirm_used_for_model_update": False,
        "frozen_parameter_hash_before_confirmation": parameter_hash_before,
        "frozen_parameter_hash_after_confirmation": parameter_hash_after,
        "frozen_calibration_hash_before_confirmation": calibration_hash_before,
        "frozen_calibration_hash_after_confirmation": calibration_hash_after,
        "frozen_manifest_sha256": checksum["manifest_sha256"],
    }
    save_json(no_update_audit, metadata_root / "phase3_no_update_audit.json")

    script_path = Path(__file__).resolve()
    confirmation_manifest = {
        "script": script_path.name,
        "phase": "Phase 3 B_confirm output-only confirmation",
        "created_at": now_string(),
        "output_root": str(output_root),
        "primary_macrostate": list(EXPECTED_PRIMARY_MACROSTATE),
        "auxiliary_accounting": ["E", "B", "G"],
        "maturity_diagnostic": "V is retained for evidence accounting and noise diagnostics only",
        "frozen_manifest": str(frozen_manifest_path),
        "frozen_manifest_checksum": checksum,
        "phase1_implementation": phase1_implementation_audit,
        "phase3_script_sha256": sha256_file(script_path),
        "stage1_root": str(stage1_root),
        "stage1_root_source": stage1_root_source,
        "stage1_fixed_k6_contract": current_kmeans_audit,
        "confirm_split": args.confirm_split,
        "guardrails": {
            "parameter_search_opened": False,
            "calibration_reestimated": False,
            "mechanism_family_reselected": False,
            "mechanism_parameters_refit": False,
            "kmeans_refit": False,
            "macrostate_k_selected": False,
            "region_redefinition": False,
            "B_confirm_read": True,
            "B_confirm_used_for_update": False,
            "confirmation_mode": "output_only",
        },
        "frozen_parameters": frozen_parameters,
        "frozen_parameter_hash": parameter_hash_before,
        "frozen_calibration": dataclasses.asdict(calibration),
        "frozen_calibration_hash": calibration_hash_before,
        "confirm_data_load_manifest": load_manifest,
        "confirm_panel_fingerprint": panel_fingerprint(panel, args.confirm_split),
        "confirmation_metrics": metrics,
        "development_metric_reference_source": "Phase-2 frozen manifest",
        "visualization_policy": "No images are generated; the confirmation field table is written for separate publication scripts.",
        "outputs": {
            "confirmation_structural_alignment_metrics": str(metrics_path),
            "development_vs_confirmation_metric_stability": str(stability_path),
            "confirmation_field_grid": str(field_grid_path),
            "prediction_audit_sample": str(audit_path),
            "full_prediction_table": full_prediction_path,
            "no_update_audit": str(metadata_root / "phase3_no_update_audit.json"),
        },
        "confirmation_status": "completed_output_only",
        "interpretation_boundary": (
            "Confirmation outputs must not be used to modify the selected family, "
            "frozen parameters, pooled calibration, or Stage-1 coarse graining."
        ),
        "elapsed_seconds": float(time.time() - started),
    }

    manifest_path = metadata_root / "phase3_confirmation_manifest.json"
    save_json(confirmation_manifest, manifest_path)
    manifest_sha = sha256_file(manifest_path)
    save_json(
        {
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
        },
        metadata_root / "phase3_confirmation_manifest.sha256.json",
    )
    (metadata_root / "phase3_confirmation_manifest.sha256").write_text(
        f"{manifest_sha}  {manifest_path.name}\n",
        encoding="utf-8",
    )

    print("[phase3] completed")
    print(f"[phase3] confirmation manifest: {manifest_path}")
    print(f"[phase3] elapsed seconds: {time.time() - started:.1f}")


if __name__ == "__main__":
    main()
