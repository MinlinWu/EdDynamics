#!/usr/bin/env python3
"""Freeze the selected offset dual-channel mechanism after pooled development calibration."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

EXPECTED_FAMILY = "offset_dual_channel"
EXPECTED_FREE_PARAMS = ("theta0", "thetaM", "phi0", "deltaS")
EXPECTED_ZERO_PARAMS = ("thetaPsi", "thetaMPsi", "phiPsi")
EXPECTED_NUISANCE_PARAMS = ("lambdaR", "lambdaA", "lambdaI")
EXPECTED_NUISANCE = {"lambdaR": 0.46, "lambdaA": 1.10, "lambdaI": 0.85}
EXPECTED_PRIMARY_MACROSTATE = ("M", "Psi")
PHASE1_SCRIPT_BASENAME = "tune_offset_dual_channel_phase1.py"

DEFAULT_STAGE1_ROOT = "/data/datasets/KT4/outputs_KT4/stage1"
DEFAULT_HANDOFF_PATH = (
    "/data/datasets/KT4/outputs_KT4/stage2_phase1_unified_minimality/"
    "metadata/phase1_minimal_mechanism_handoff.json"
)
DEFAULT_PHASE1_OUTPUT_ROOT = "/data/datasets/KT4/outputs_KT4/stage2_phase1"
DEFAULT_OUTPUT_ROOT = "/data/datasets/KT4/outputs_KT4/stage2_phase2_freeze"

PRIMARY_METRIC_COLUMNS = (
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
    "phase_loss_max_qdist",
    "coverage_loss_max_qdist",
)


def now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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


def save_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(obj), handle, indent=2, ensure_ascii=False, allow_nan=False)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_default_phase1_script() -> Path:
    env_path = os.environ.get("MECH_PHASE2_PHASE1_SCRIPT", "").strip()
    if env_path:
        return Path(env_path)
    here = Path(__file__).resolve().parent
    return here / PHASE1_SCRIPT_BASENAME


def prepare_runtime_cache(phase1_script: Path) -> None:
    version = sha256_file(phase1_script.resolve())[:16]
    cache_dir = Path(tempfile.gettempdir()) / "ednet_kt4_numba_cache" / version
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)


def import_phase1_module(path: Path) -> Any:
    source = path.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Phase-1 implementation not found: {source}")
    spec = importlib.util.spec_from_file_location(
        "tune_offset_dual_channel_phase1",
        str(source),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Phase-1 implementation: {source}")
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
        "infer_eta_from_stage1",
        "calibrate_from_A_train",
        "Calibration",
        "make_metric_cache",
        "simulate_arrays",
        "structure_metrics_fast_no_regions",
        "objective_from_metrics",
        "objective_diagnostics",
        "primary_objective_score",
        "prediction_frame_from_cache",
        "field_grid_table",
        "selected_law_tables",
        "estimate_noise_from_cache",
        "audit_stage1_kmeans_contract",
        "write_table",
        "PARAM_NAMES",
        "MECHANISM_PARAM_NAMES",
        "NUISANCE_PARAM_NAMES",
        "PARAM_GRID_VALUES",
        "PARAM_DEFAULTS",
        "PARAM_BOUNDS",
        "TERM_SWITCHES",
        "TAU_RESPONSE_DAYS",
        "TAU_ACTIVITY_DAYS",
    )
    missing = [name for name in required if not hasattr(p1, name)]
    if missing:
        raise RuntimeError(f"Phase-1 implementation is missing required names: {missing}")


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def validate_handoff(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Minimality handoff not found: {path}")
    handoff = load_json(path)
    required_true = (
        "ready_for_phase2_freeze",
        "scalar_parameter_minimality_confirmed",
        "search_adequacy_confirmed",
        "baseline_not_practically_equivalent_to_best",
        "final_model_selected_by_parsimony_rule",
    )
    failed = [name for name in required_true if not coerce_bool(handoff.get(name, False))]
    if failed:
        raise RuntimeError(f"Minimality handoff failed required gates: {failed}")
    if str(handoff.get("final_family_key", "")) != EXPECTED_FAMILY:
        raise RuntimeError(f"Unexpected final family: {handoff.get('final_family_key')!r}")
    free = tuple(str(name) for name in handoff.get("final_free_mechanism_parameters", []))
    if set(free) != set(EXPECTED_FREE_PARAMS):
        raise RuntimeError(f"Unexpected free mechanism parameters: {free}")
    selected = dict(handoff.get("final_selected_parameters", {}))
    for name in EXPECTED_ZERO_PARAMS:
        if abs(float(selected.get(name, 0.0))) > 1e-12:
            raise RuntimeError(f"Handoff structural-zero parameter is non-zero: {name}")
    nuisance = dict(handoff.get("fixed_nuisance_scales", {}))
    for name, expected in EXPECTED_NUISANCE.items():
        value = float(nuisance.get(name, selected.get(name, expected)))
        if not np.isclose(value, expected, atol=1e-12, rtol=0.0):
            raise RuntimeError(f"Unexpected handoff nuisance scale {name}={value}")
    return handoff


def validate_phase1_selected_parameters(
    path: Path,
    handoff: Mapping[str, Any],
    p1: Any,
) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Phase-1 selected-parameter file not found: {path}")
    payload = load_json(path)
    selected = dict(payload.get("selected_parameters", {}))
    if not selected:
        raise RuntimeError("Phase-1 selected-parameter payload is empty.")
    if str(payload.get("post_ablation_selected_family", "")) != EXPECTED_FAMILY:
        raise RuntimeError("Phase-1 selected-parameter family does not match the minimality handoff.")
    free = tuple(str(name) for name in payload.get("free_mechanism_parameters", []))
    if set(free) != set(EXPECTED_FREE_PARAMS):
        raise RuntimeError(f"Unexpected Phase-1 free mechanism parameters: {free}")
    if tuple(payload.get("primary_macrostate", [])) != EXPECTED_PRIMARY_MACROSTATE:
        raise RuntimeError("Phase-1 primary macrostate must be exactly ['M', 'Psi'].")
    if not coerce_bool(payload.get("minimality_handoff_ready", False)):
        raise RuntimeError("Phase-1 output does not record a ready minimality handoff.")
    if not coerce_bool(payload.get("minimality_handoff_scalar_minimality_confirmed", False)):
        raise RuntimeError("Phase-1 output does not record confirmed scalar minimality.")

    params: Dict[str, float] = {}
    for name in EXPECTED_FREE_PARAMS:
        if name not in selected:
            raise RuntimeError(f"Phase-1 selected parameters are missing {name!r}.")
        value = float(selected[name])
        if not math.isfinite(value):
            raise RuntimeError(f"Phase-1 selected parameter is non-finite: {name}={value}")
        params[name] = value
    for name in EXPECTED_ZERO_PARAMS:
        value = float(selected.get(name, 0.0))
        if abs(value) > 1e-12:
            raise RuntimeError(f"Phase-1 structural-zero parameter is non-zero: {name}={value}")
        params[name] = 0.0
    fixed_nuisance = dict(payload.get("fixed_nuisance_scales", {}))
    for name, expected in EXPECTED_NUISANCE.items():
        value = float(selected.get(name, fixed_nuisance.get(name, expected)))
        if not np.isclose(value, expected, atol=1e-12, rtol=0.0):
            raise RuntimeError(f"Unexpected Phase-1 nuisance scale {name}={value}")
        params[name] = value

    for name in EXPECTED_FREE_PARAMS:
        grid = np.asarray(p1.PARAM_GRID_VALUES[name], dtype=float)
        if not np.any(np.isclose(params[name], grid, atol=1e-12, rtol=0.0)):
            raise RuntimeError(f"Selected Phase-1 value is off the formal grid: {name}={params[name]}")

    handoff_values = dict(handoff.get("final_selected_parameters", {}))
    comparison = {}
    for name in EXPECTED_FREE_PARAMS:
        if name in handoff_values:
            comparison[name] = {
                "phase1_value_used_for_freeze": params[name],
                "minimality_handoff_value_not_used_for_freeze": float(handoff_values[name]),
                "difference": params[name] - float(handoff_values[name]),
            }

    return {
        "payload": payload,
        "parameters": params,
        "source_path": str(path.resolve()),
        "source_sha256": sha256_file(path.resolve()),
        "freeze_parameter_value_source": "metadata/phase1_selected_parameters.json:selected_parameters",
        "minimality_handoff_parameter_values_used_for_freeze": False,
        "differences_vs_minimality_handoff": comparison,
    }


def validate_phase1_manifest(
    path: Path,
    selected_parameters: Mapping[str, float],
    phase1_script: Path,
    stage1_root: Path,
    current_kmeans_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Phase-1 manifest not found: {path}")
    manifest = load_json(path)
    if str(manifest.get("family", "")) != EXPECTED_FAMILY:
        raise RuntimeError("Phase-1 manifest family does not match the selected mechanism.")
    if tuple(manifest.get("primary_macrostate", [])) != EXPECTED_PRIMARY_MACROSTATE:
        raise RuntimeError("Phase-1 manifest primary macrostate is not ['M', 'Psi'].")
    if str(manifest.get("B_confirm_policy", "")) != "not read or used":
        raise RuntimeError("Phase-1 manifest does not preserve the B_confirm boundary.")
    manifest_selected = dict(manifest.get("selected_parameters", {}))
    for name, expected in selected_parameters.items():
        if name not in manifest_selected or not np.isclose(
            float(manifest_selected[name]),
            float(expected),
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(f"Phase-1 manifest parameter mismatch for {name}.")
    manifest_kmeans = dict(manifest.get("stage1_fixed_k6_contract", {}))
    for name in ("metadata_sha256", "centers_sha256"):
        if str(manifest_kmeans.get(name, "")) != str(current_kmeans_audit.get(name, "")):
            raise RuntimeError(f"Phase-1 manifest fixed-K contract mismatch: {name}")
    manifest_stage1 = Path(str(manifest.get("stage1_load_manifest", {}).get("stage1_root", stage1_root))).resolve()
    if manifest_stage1 != stage1_root.resolve():
        raise RuntimeError("Phase-1 manifest Stage-1 root differs from the Phase-2 input root.")
    script_name = str(manifest.get("script", ""))
    if script_name and script_name != phase1_script.name:
        raise RuntimeError(
            f"Phase-1 manifest names {script_name!r}, but Phase 2 imported {phase1_script.name!r}."
        )
    manifest["manifest_path"] = str(path.resolve())
    manifest["manifest_sha256"] = sha256_file(path.resolve())
    return manifest


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


def infer_pooled_eta(p1: Any, train_raw: pd.DataFrame, val_raw: pd.DataFrame) -> float:
    required = ["response_evidence_mass_pre", "response_evidence_maturity_V_pre"]
    pooled = pd.concat(
        [train_raw[required], val_raw[required]],
        ignore_index=True,
        copy=False,
    )
    return float(p1.infer_eta_from_stage1(pooled))


def load_development_panels(
    p1: Any,
    stage1_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, Dict[str, Any]]:
    dynamics = p1.stage1_dynamics_root(stage1_root)
    train_raw = p1.read_core_panel(dynamics, "A_train")
    val_raw = p1.read_core_panel(dynamics, "A_val")
    if train_raw.empty or val_raw.empty:
        raise RuntimeError("A_train or A_val Stage-1 panel is empty.")
    eta = infer_pooled_eta(p1, train_raw, val_raw)
    train = p1.prepare_panel(train_raw, "A_train", eta)
    val = p1.prepare_panel(val_raw, "A_val", eta)
    if train.empty or val.empty:
        raise RuntimeError("A_train or A_val is empty after Phase-2 validity filtering.")
    overlap = np.intersect1d(
        np.asarray(sorted(train["user_id"].unique()), dtype=np.int64),
        np.asarray(sorted(val["user_id"].unique()), dtype=np.int64),
        assume_unique=True,
    )
    if overlap.size:
        raise RuntimeError(f"A_train and A_val share {overlap.size} users.")
    return train, val, eta, {
        "stage1_root": str(stage1_root.resolve()),
        "stage1_dynamics_root": str(Path(dynamics).resolve()),
        "train_rows_raw": int(len(train_raw)),
        "val_rows_raw": int(len(val_raw)),
        "train_rows_valid": int(len(train)),
        "val_rows_valid": int(len(val)),
        "train_users_valid": int(train["user_id"].nunique()),
        "val_users_valid": int(val["user_id"].nunique()),
        "eta_refit_scope": "A_train_plus_A_val",
        "eta": eta,
        "user_disjoint": True,
        "B_confirm_policy": "not read or used in Phase 2",
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


def slim_metric_reference(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        label = str(row.get("label", ""))
        if not label:
            continue
        out[label] = {name: row.get(name) for name in PRIMARY_METRIC_COLUMNS if name in row}
        out[label]["n_rows"] = row.get("n_rows")
        out[label]["n_users"] = row.get("n_users")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the Phase-1 offset dual-channel mechanism after pooled development calibration."
    )
    parser.add_argument("--phase1-script", type=Path, default=resolve_default_phase1_script())
    parser.add_argument("--stage1-root", type=Path, default=Path(os.environ.get("MECH_PHASE2_STAGE1_ROOT", DEFAULT_STAGE1_ROOT)))
    parser.add_argument("--handoff", type=Path, default=Path(os.environ.get("MECH_PHASE2_HANDOFF_PATH", DEFAULT_HANDOFF_PATH)))
    parser.add_argument("--phase1-output-root", type=Path, default=Path(os.environ.get("MECH_PHASE2_PHASE1_OUTPUT_ROOT", DEFAULT_PHASE1_OUTPUT_ROOT)))
    parser.add_argument("--phase1-selected-parameters", type=Path, default=Path(os.environ["MECH_PHASE2_PHASE1_SELECTED_PARAMETERS"]) if "MECH_PHASE2_PHASE1_SELECTED_PARAMETERS" in os.environ else None)
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("MECH_PHASE2_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)))
    parser.add_argument("--audit-sample-rows", type=int, default=int(os.environ.get("MECH_PHASE2_AUDIT_SAMPLE_ROWS", "200000")))
    parser.add_argument("--write-full-predictions", action="store_true", default=bool(int(os.environ.get("MECH_PHASE2_WRITE_FULL_PREDICTIONS", "0"))))
    parser.add_argument("--random-state", type=int, default=int(os.environ.get("MECH_PHASE2_RANDOM_STATE", "42")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    output_root = args.output_root.resolve()
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    table_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    phase1_script = args.phase1_script.resolve()
    stage1_root = args.stage1_root.resolve()
    handoff_path = args.handoff.resolve()
    phase1_selected_path = (
        args.phase1_selected_parameters.resolve()
        if args.phase1_selected_parameters is not None
        else (args.phase1_output_root / "metadata" / "phase1_selected_parameters.json").resolve()
    )
    phase1_manifest_path = phase1_selected_path.with_name("phase1_manifest.json")

    print(f"[phase2] started {now_string()}")
    prepare_runtime_cache(phase1_script)
    p1 = import_phase1_module(phase1_script)
    validate_phase1_module_contract(p1)
    handoff = validate_handoff(handoff_path)
    kmeans_audit = p1.audit_stage1_kmeans_contract(stage1_root)
    parameter_source = validate_phase1_selected_parameters(
        phase1_selected_path,
        handoff,
        p1,
    )
    params = dict(parameter_source["parameters"])
    phase1_manifest = validate_phase1_manifest(
        phase1_manifest_path,
        params,
        phase1_script,
        stage1_root,
        kmeans_audit,
    )
    configure_frozen_phase1_module(p1, params)

    save_json(handoff, metadata_root / "phase1_minimality_handoff_snapshot.json")
    save_json(parameter_source["payload"], metadata_root / "phase1_selected_parameters_snapshot.json")
    save_json(phase1_manifest, metadata_root / "phase1_manifest_snapshot.json")
    save_json(kmeans_audit, metadata_root / "stage1_fixed_k6_contract.json")
    parameter_source_audit = {
        "freeze_parameter_value_source": parameter_source["freeze_parameter_value_source"],
        "minimality_handoff_parameter_values_used_for_freeze": False,
        "differences_vs_minimality_handoff": parameter_source["differences_vs_minimality_handoff"],
        "phase1_selected_parameters": parameter_source["source_path"],
        "phase1_selected_parameters_sha256": parameter_source["source_sha256"],
    }
    save_json(parameter_source_audit, metadata_root / "phase2_parameter_source_audit.json")

    frozen_phase1_copy = metadata_root / "frozen_phase1_implementation.py"
    shutil.copy2(phase1_script, frozen_phase1_copy)
    if sha256_file(frozen_phase1_copy) != sha256_file(phase1_script):
        raise RuntimeError("Frozen Phase-1 implementation copy failed checksum verification.")

    print("[phase2] loading development panels")
    train, val, eta, load_manifest = load_development_panels(p1, stage1_root)
    pooled = pd.concat([train, val], ignore_index=True, sort=False, copy=False)

    print("[phase2] refitting pooled development calibration")
    calibration = p1.calibrate_from_A_train(
        pooled,
        eta=eta,
        tau_response_days=p1.TAU_RESPONSE_DAYS,
        tau_activity_days=p1.TAU_ACTIVITY_DAYS,
    )

    panels = {
        "A_train": train,
        "A_val": val,
        "A_train_plus_A_val": pooled,
    }
    metric_rows = []
    caches: Dict[str, Any] = {}
    simulations: Dict[str, Any] = {}
    field_paths: Dict[str, str] = {}
    for label, panel in panels.items():
        metrics, cache, simulation = evaluate_panel(p1, panel, params, calibration, label)
        metric_rows.append(metrics)
        caches[label] = cache
        simulations[label] = simulation
        field_paths[label] = str(
            p1.write_table(
                p1.field_grid_table(cache, simulation, label),
                table_root / f"phase2_{label}_field_grid",
            )
        )

    noise = p1.estimate_noise_from_cache(
        caches["A_train_plus_A_val"],
        simulations["A_train_plus_A_val"],
        calibration,
    )
    calibration.sigma_U0 = float(noise["sigma_U0_maturity_scaled_residual"])
    calibration.sigma_Psi0 = float(noise["sigma_Psi0_maturity_scaled_residual"])

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = p1.write_table(
        metrics_df,
        table_root / "phase2_development_structural_alignment_metrics",
    )
    metric_reference = slim_metric_reference(metric_rows)
    save_json(metric_reference, metadata_root / "phase2_development_metric_reference.json")
    save_json(dataclasses.asdict(calibration), metadata_root / "phase2_pooled_development_calibration.json")

    prediction_audit = p1.prediction_frame_from_cache(
        caches["A_train_plus_A_val"],
        simulations["A_train_plus_A_val"],
        "A_train_plus_A_val",
        args.audit_sample_rows,
        args.random_state + 200,
    )
    audit_path = p1.write_table(
        prediction_audit,
        table_root / "phase2_prediction_audit_sample",
    )

    full_prediction_paths: Dict[str, str] = {}
    if args.write_full_predictions:
        for label in panels:
            frame = p1.prediction_frame_from_cache(
                caches[label],
                simulations[label],
                label,
                0,
                args.random_state,
            )
            full_prediction_paths[label] = str(
                p1.write_table(frame, table_root / f"phase2_{label}_full_predictions")
            )

    response_law, alignment_law = p1.selected_law_tables(params)
    response_law_path = p1.write_table(response_law, table_root / "phase2_frozen_response_law")
    alignment_law_path = p1.write_table(alignment_law, table_root / "phase2_frozen_alignment_law")

    phase1_script_sha = sha256_file(phase1_script)
    handoff_sha = sha256_file(handoff_path)
    selected_sha = parameter_source["source_sha256"]
    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)
    full_parameter_vector = {name: float(params[name]) for name in p1.PARAM_NAMES}
    frozen_parameters = {
        "family_key": EXPECTED_FAMILY,
        "family_label": "Offset dual-channel",
        "free_mechanism_parameters": list(EXPECTED_FREE_PARAMS),
        "fixed_zero_mechanism_parameters": list(EXPECTED_ZERO_PARAMS),
        "mechanism_parameters": {name: full_parameter_vector[name] for name in EXPECTED_FREE_PARAMS},
        "structural_zero_values": {name: full_parameter_vector[name] for name in EXPECTED_ZERO_PARAMS},
        "fixed_nuisance_scales": {name: full_parameter_vector[name] for name in EXPECTED_NUISANCE_PARAMS},
        "full_parameter_vector": full_parameter_vector,
        "parameter_hash": stable_json_hash(full_parameter_vector),
    }

    manifest = {
        "script": script_path.name,
        "phase": "Phase 2 pooled-development calibration and frozen mechanism manifest",
        "created_at": now_string(),
        "output_root": str(output_root),
        "primary_macrostate": list(EXPECTED_PRIMARY_MACROSTATE),
        "auxiliary_accounting": ["E", "B", "G"],
        "maturity_diagnostic": "V is retained for evidence accounting and noise diagnostics only",
        "stage1_root": str(stage1_root),
        "stage1_fixed_k6_contract": kmeans_audit,
        "phase1_implementation": {
            "source_path": str(phase1_script),
            "source_sha256": phase1_script_sha,
            "frozen_copy_path": str(frozen_phase1_copy.resolve()),
            "frozen_copy_sha256": sha256_file(frozen_phase1_copy),
        },
        "phase1_minimality_handoff": str(handoff_path),
        "phase1_minimality_handoff_sha256": handoff_sha,
        "phase1_selected_parameters": parameter_source["source_path"],
        "phase1_selected_parameters_sha256": selected_sha,
        "phase1_manifest": phase1_manifest["manifest_path"],
        "phase1_manifest_sha256": phase1_manifest["manifest_sha256"],
        "phase2_script_sha256": script_sha,
        "guardrails": {
            "ready_handoff_required": True,
            "ready_for_phase2_freeze_from_handoff": True,
            "scalar_parameter_minimality_confirmed": True,
            "search_adequacy_confirmed": True,
            "B_confirm_read": False,
            "parameter_search_opened": False,
            "mechanism_family_reselected": False,
            "mechanism_parameters_refit": False,
            "calibration_refit_scope": "A_train_plus_A_val_only",
            "kmeans_refit": False,
            "macrostate_k_selected": False,
            "region_redefinition": False,
            "phase3_contract": "Phase 3 must use this manifest without parameter or calibration updates.",
        },
        "parameter_source_audit": {
            "freeze_parameter_value_source": parameter_source["freeze_parameter_value_source"],
            "minimality_handoff_parameter_values_used_for_freeze": False,
            "differences_vs_minimality_handoff": parameter_source["differences_vs_minimality_handoff"],
            "audit_path": str(metadata_root / "phase2_parameter_source_audit.json"),
        },
        "frozen_parameters": frozen_parameters,
        "frozen_calibration": dataclasses.asdict(calibration),
        "calibration_hash": stable_json_hash(dataclasses.asdict(calibration)),
        "development_data_load_manifest": load_manifest,
        "development_panel_fingerprints": {
            label: panel_fingerprint(panel, label)
            for label, panel in panels.items()
        },
        "development_metric_reference": metric_reference,
        "visualization_policy": "No images are generated; field and law tables are written for separate publication scripts.",
        "outputs": {
            "development_structural_alignment_metrics": str(metrics_path),
            "prediction_audit_sample": str(audit_path),
            "field_grid_tables": field_paths,
            "frozen_response_law": str(response_law_path),
            "frozen_alignment_law": str(alignment_law_path),
            "pooled_development_calibration": str(metadata_root / "phase2_pooled_development_calibration.json"),
            "development_metric_reference": str(metadata_root / "phase2_development_metric_reference.json"),
            "parameter_source_audit": str(metadata_root / "phase2_parameter_source_audit.json"),
            "full_prediction_tables": full_prediction_paths,
        },
        "elapsed_seconds": float(time.time() - started),
    }

    manifest_path = metadata_root / "phase2_frozen_model_manifest.json"
    save_json(manifest, manifest_path)
    manifest_sha = sha256_file(manifest_path)
    checksum = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "phase1_script_sha256": phase1_script_sha,
        "phase1_minimality_handoff_sha256": handoff_sha,
        "phase1_selected_parameters_sha256": selected_sha,
        "phase2_script_sha256": script_sha,
        "stage1_kmeans_metadata_sha256": kmeans_audit["metadata_sha256"],
        "stage1_kmeans_centers_sha256": kmeans_audit["centers_sha256"],
    }
    save_json(checksum, metadata_root / "phase2_frozen_model_manifest.sha256.json")
    (metadata_root / "phase2_frozen_model_manifest.sha256").write_text(
        f"{manifest_sha}  {manifest_path.name}\n",
        encoding="utf-8",
    )

    print("[phase2] completed")
    print(f"[phase2] frozen manifest: {manifest_path}")
    print(f"[phase2] elapsed seconds: {time.time() - started:.1f}")


if __name__ == "__main__":
    main()
