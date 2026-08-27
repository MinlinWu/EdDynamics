#!/usr/bin/env python3
"""Tune the frozen offset dual-channel mechanism on A_train and A_val."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    njit = None
    NUMBA_AVAILABLE = False

CONFIG_STAGE1_ROOT = Path(os.environ.get(
    "MECH_PHASE1_STAGE1_ROOT",
    "/data/datasets/KT4/outputs_KT4/stage1",
))
CONFIG_HANDOFF_PATH = Path(os.environ.get(
    "MECH_PHASE1_HANDOFF_PATH",
    "/data/datasets/KT4/outputs_KT4/stage2_phase1_unified_minimality/metadata/phase1_minimal_mechanism_handoff.json",
))
CONFIG_OUTPUT_ROOT = Path(os.environ.get(
    "MECH_PHASE1_OUTPUT_ROOT",
    "/data/datasets/KT4/outputs_KT4/stage2_phase1",
))
CONFIG_SEARCH_TOTAL_USERS = int(os.environ.get("MECH_PHASE1_SEARCH_TOTAL_USERS", "20000"))
CONFIG_BLOCK_TOP_K = int(os.environ.get("MECH_PHASE1_BLOCK_TOP_K", "10"))
CONFIG_FINE_STARTS = int(os.environ.get("MECH_PHASE1_FINE_STARTS", "3"))
CONFIG_FINE_REFINE_PASSES = int(os.environ.get("MECH_PHASE1_FINE_REFINE_PASSES", "2"))
CONFIG_FULL_SHORTLIST_K = int(os.environ.get("MECH_PHASE1_FULL_SHORTLIST_K", "16"))
CONFIG_FULL_REFINE_STARTS = int(os.environ.get("MECH_PHASE1_FULL_REFINE_STARTS", "2"))
CONFIG_FULL_REFINE_PASSES = int(os.environ.get("MECH_PHASE1_FULL_REFINE_PASSES", "2"))
CONFIG_PREDICTION_AUDIT_ROWS = int(os.environ.get("MECH_PHASE1_PREDICTION_AUDIT_ROWS", "200000"))
CONFIG_WRITE_FULL_PREDICTIONS = bool(int(os.environ.get("MECH_PHASE1_WRITE_FULL_PREDICTIONS", "0")))
CONFIG_SMOKE_TEST_MODE = bool(int(os.environ.get("MECH_PHASE1_SMOKE_TEST_MODE", "0")))
CONFIG_SMOKE_TEST_MAX_USERS_PER_SPLIT = int(os.environ.get("MECH_PHASE1_SMOKE_TEST_MAX_USERS_PER_SPLIT", "300"))
CONFIG_RANDOM_STATE = int(os.environ.get("MECH_PHASE1_RANDOM_STATE", os.environ.get("EDNET_STAGE1_RANDOM_STATE", "42")))
CONFIG_PROGRESS = bool(int(os.environ.get("MECH_PHASE1_PROGRESS", "1")))
CONFIG_PROGRESS_SNAPSHOT_SECONDS = float(os.environ.get("MECH_PHASE1_PROGRESS_SNAPSHOT_SECONDS", "10.0"))
CONFIG_USE_NUMBA = bool(int(os.environ.get("MECH_PHASE1_USE_NUMBA", "1"))) and NUMBA_AVAILABLE
CONFIG_SIGNED_GAIN_QUANTILE = float(os.environ.get("MECH_PHASE1_SIGNED_GAIN_QUANTILE", "0.75"))
CONFIG_DISTRIBUTION_LOSS_MAX_ROWS = int(os.environ.get("MECH_PHASE1_DISTRIBUTION_LOSS_MAX_ROWS", "200000"))
CONFIG_SANITY_PENALTY_WEIGHT = float(os.environ.get("MECH_PHASE1_SANITY_PENALTY_WEIGHT", "0.25"))
CONFIG_IDENTITY_REG_WEIGHT = float(os.environ.get("MECH_PHASE1_IDENTITY_REG_WEIGHT", "0.05"))
CONFIG_DELTA_S_OBJECTIVE_TOL = float(os.environ.get("MECH_PHASE1_DELTAS_OBJECTIVE_TOL", "1e-5"))
CONFIG_DELTA_S_SATURATION_TOL = float(os.environ.get("MECH_PHASE1_DELTAS_SATURATION_TOL", "2e-4"))

EPS = 1e-12
TAU_RESPONSE_DAYS = float(os.environ.get("EDNET_STAGE1_TAU_RESPONSE_DAYS", "10.0"))
TAU_ACTIVITY_DAYS = float(os.environ.get("EDNET_STAGE1_TAU_ACTIVITY_DAYS", "10.0"))
EVIDENCE_MATURITY_SCALE_DEFAULT = float(os.environ.get("EDNET_STAGE1_EVIDENCE_MATURITY_SCALE", "20.0"))
GRID_BINS_SIGNED = np.linspace(-1.0, 1.0, int(os.environ.get("EDNET_STAGE1_SIGNED_GRID_N", "41")))
MIN_DRIFT_BIN_COUNT = int(os.environ.get("EDNET_STAGE1_MIN_DRIFT_BIN_COUNT", "30"))
EXPECTED_STAGE1_MACROSTATE_K = 6
EXPECTED_STAGE1_KMEANS_N_INIT = int(os.environ.get("EDNET_STAGE1_KMEANS_N_INIT", "20"))
EXPECTED_STAGE1_KMEANS_FIT_MAX_ROWS = int(os.environ.get("EDNET_STAGE1_KMEANS_FIT_MAX_ROWS", "500000"))
EXPECTED_STAGE1_RANDOM_STATE = int(os.environ.get("EDNET_STAGE1_RANDOM_STATE", "42"))

MECHANISM_PARAM_NAMES = ["theta0", "thetaM", "thetaPsi", "thetaMPsi", "phi0", "deltaS", "phiPsi"]
NUISANCE_PARAM_NAMES = ["lambdaR", "lambdaA", "lambdaI"]
PARAM_NAMES = MECHANISM_PARAM_NAMES + NUISANCE_PARAM_NAMES
EXPECTED_FREE_PARAMS = ("theta0", "thetaM", "phi0", "deltaS")
EXPECTED_ZERO_PARAMS = ("thetaPsi", "thetaMPsi", "phiPsi")
EXPECTED_FAMILY = "offset_dual_channel"
FIXED_NUISANCE = {"lambdaR": 0.46, "lambdaA": 1.10, "lambdaI": 0.85}

@dataclass(frozen=True)
class SearchSpec:
    lower: float
    upper: float
    precision: float
    coarse_step: float
    subset_radius: float
    full_radius: float

SEARCH_SPECS = {
    "theta0": SearchSpec(-0.40, -0.05, 0.01, 0.05, 0.06, 0.03),
    "thetaM": SearchSpec(0.30, 1.00, 0.01, 0.10, 0.10, 0.03),
    "phi0": SearchSpec(-2.10, -0.80, 0.01, 0.10, 0.15, 0.03),
    "deltaS": SearchSpec(2.0, 8.0, 1.0, 1.0, 2.0, 1.0),
}

ARCHIVED_PHASE1_BEST = {
    "theta0": -0.22,
    "thetaM": 0.58,
    "phi0": -1.61,
    "deltaS": 8.0,
}

PHASE1_PILOT_REFERENCE = {
    "theta0": -0.22,
    "thetaM": 0.58,
    "thetaPsi": 0.0,
    "thetaMPsi": 0.0,
    "phi0": -1.61,
    "deltaS": 8.0,
    "phiPsi": 0.0,
    **FIXED_NUISANCE,
}

SEARCH_RANGE_RATIONALE = {
    "theta0": "negative response offset; positive values are excluded and the range covers weak to moderately strong negative baseline drive",
    "thetaM": "positive restoring coefficient; the range covers weak-to-strong contraction without entering the previously over-ordered regime",
    "phi0": "negative response-channel alignment baseline; the range spans a clearly negative channel while excluding near-zero and fully saturated tails",
    "deltaS": "positive response-support contrast; the range spans the near-neutral crossover through the verified support-channel saturation plateau",
}


def _decimal_places(step: float) -> int:
    text = f"{float(step):.12f}".rstrip("0")
    return len(text.split(".")[1]) if "." in text else 0


def _inclusive_grid(lower: float, upper: float, step: float) -> List[float]:
    if not np.isfinite([lower, upper, step]).all():
        raise ValueError("Grid bounds and step must be finite.")
    if step <= 0 or upper < lower:
        raise ValueError(f"Invalid grid: lower={lower}, upper={upper}, step={step}")
    n = int(math.floor((upper - lower) / step + 1e-10))
    values = lower + step * np.arange(n + 1, dtype=float)
    if values.size == 0 or values[-1] < upper - step * 1e-8:
        values = np.append(values, upper)
    values = np.clip(values, lower, upper)
    digits = max(
        _decimal_places(lower),
        _decimal_places(upper),
        _decimal_places(step),
    )
    return sorted(set(
        float(round(value, digits))
        for value in values
        if lower - 1e-12 <= value <= upper + 1e-12
    ))


PARAM_GRID_VALUES: Dict[str, List[float]] = {
    name: _inclusive_grid(spec.lower, spec.upper, spec.precision)
    for name, spec in SEARCH_SPECS.items()
}
PARAM_GRID_VALUES["theta0"] = [x for x in PARAM_GRID_VALUES["theta0"] if not np.isclose(x, 0.0)]
PARAM_GRID_VALUES.update({name: [0.0] for name in EXPECTED_ZERO_PARAMS})
PARAM_GRID_VALUES.update({name: [value] for name, value in FIXED_NUISANCE.items()})

PARAM_DEFAULTS = {
    "theta0": -0.18,
    "thetaM": 0.55,
    "thetaPsi": 0.0,
    "thetaMPsi": 0.0,
    "phi0": -1.45,
    "deltaS": 8.0,
    "phiPsi": 0.0,
    **FIXED_NUISANCE,
}
PARAM_BOUNDS = np.asarray([
    [min(PARAM_GRID_VALUES[name]), max(PARAM_GRID_VALUES[name])]
    for name in PARAM_NAMES
], dtype=float)
TERM_SWITCHES = {
    "theta0": True,
    "thetaPsi": False,
    "thetaMPsi": False,
    "deltaS": True,
    "phiPsi": False,
}

OBJECTIVE_WEIGHTS = {
    "one_step_mse_main_norm": 0.10,
    "occupancy_js_MR_PsiA": 0.20,
    "drift_local_rmse_loss_MR_PsiA": 0.30,
    "drift_direction_loss_MR_PsiA": 0.20,
    "drift_magnitude_loss_MR_PsiA": 0.20,
}
OBJECTIVE_SANITY_LIMITS = {
    "phase_loss_max_qdist": 0.85,
    "coverage_loss_max_qdist": 0.85,
}

PHASE1_PANEL_COLUMNS = [
    "split", "user_id", "bundle_step_index",
    "M_response_prebalanced_pre", "activity_alignment_order_Psi_pre",
    "response_evidence_maturity_V_pre", "response_evidence_mass_pre",
    "next_M_response_prebalanced", "next_activity_alignment_order_Psi",
    "next_response_evidence_maturity_V", "next_response_evidence_mass",
    "next_gap_days", "has_next_submitted_bundle", "bundle_n_questions",
    "answered_fraction_interval", "total_response_count_diagnostic",
    "response_active_mass_interval", "support_active_total_interval",
    "idle_mass_interval", "activity_active_mass_pre", "activity_idle_mass_pre",
    "activity_aligned_mass_pre", "activity_off_target_mass_pre",
    "activity_non_aligned_mass_pre", "response_aligned_mass_interval",
    "response_off_target_mass_interval", "response_neutral_mass_interval",
    "support_aligned_mass_interval", "support_off_target_mass_interval",
    "support_neutral_mass_interval", "next_activity_active_mass",
    "next_activity_idle_mass", "M_response_prebalanced_resp",
    "response_evidence_maturity_V_resp", "activity_alignment_order_Psi_support",
    "activity_alignment_order_Psi_post",
]


def json_safe(obj):
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
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
    return obj


def save_json(obj: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(obj), handle, indent=2, ensure_ascii=False, allow_nan=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_nuisance_scales() -> Dict[str, float]:
    return dict(FIXED_NUISANCE)


def stage1_dynamics_root(stage1_root: Path) -> Path:
    dynamics = Path(stage1_root).resolve() / "dynamics"
    if not dynamics.is_dir():
        raise FileNotFoundError(f"Stage-1 dynamics directory not found: {dynamics}")
    return dynamics


def read_core_panel(dyn_root: Path, split: str) -> pd.DataFrame:
    base = dyn_root / f"student_dynamics_panel_core_{split}"
    parquet = base.with_suffix(".parquet")
    if parquet.exists():
        try:
            import pyarrow.parquet as pq
            names = set(pq.read_schema(parquet).names)
            columns = [name for name in PHASE1_PANEL_COLUMNS if name in names]
            return pd.read_parquet(parquet, columns=columns)
        except Exception:
            pass
    try:
        return read_table(base, columns=PHASE1_PANEL_COLUMNS)
    except Exception:
        return read_table(base)


def audit_stage1_kmeans_contract(stage1_root: Path) -> Dict[str, object]:
    root = stage1_dynamics_root(stage1_root) / "fixed_k6_mesostates"
    metadata_path = root / "fixed_k6_model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Stage-1 fixed-K metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_features = [
        "M_response_prebalanced_pre",
        "activity_alignment_order_Psi_pre",
    ]
    mapping = metadata.get("raw_to_ordered_label", {})
    labels = list(range(EXPECTED_STAGE1_MACROSTATE_K))
    scaler_mean = np.asarray(metadata.get("scaler_mean", []), dtype=float)
    scaler_scale = np.asarray(metadata.get("scaler_scale", []), dtype=float)
    checks = {
        "coordinate": metadata.get("coordinate") == "MR_PsiA",
        "macrostate_k": int(metadata.get("macrostate_k", -1)) == EXPECTED_STAGE1_MACROSTATE_K,
        "macrostate_k_rule": metadata.get("macrostate_k_rule") == "fixed a priori",
        "features": list(metadata.get("features", [])) == expected_features,
        "fit_split": metadata.get("fit_split") == "A_train",
        "user_balanced_sampling": metadata.get("user_balanced_sampling") is True,
        "user_balanced_kmeans_fit": metadata.get("user_balanced_kmeans_fit") is True,
        "kmeans_n_init": int(metadata.get("kmeans_n_init", -1)) == EXPECTED_STAGE1_KMEANS_N_INIT,
        "fit_max_rows": int(metadata.get("fit_max_rows", -1)) == EXPECTED_STAGE1_KMEANS_FIT_MAX_ROWS,
        "random_state": int(metadata.get("random_state", -1)) == EXPECTED_STAGE1_RANDOM_STATE,
        "scaler": scaler_mean.shape == (2,) and scaler_scale.shape == (2,) and np.isfinite(scaler_mean).all() and np.isfinite(scaler_scale).all() and np.all(scaler_scale > 0),
        "label_mapping": sorted(int(k) for k in mapping) == labels and sorted(int(v) for v in mapping.values()) == labels,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Stage-1 fixed-K contract failed: " + ", ".join(failed))
    centers = read_table(root / "fixed_k6_centers")
    required = {"macrostate", "center_M", "center_Psi"}
    if not required.issubset(centers.columns):
        raise RuntimeError(f"Stage-1 fixed-K centers are missing: {sorted(required.difference(centers.columns))}")
    ids = pd.to_numeric(centers["macrostate"], errors="coerce").to_numpy(dtype=float)
    values = centers[["center_M", "center_Psi"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    expected = np.arange(EXPECTED_STAGE1_MACROSTATE_K, dtype=float)
    order = np.lexsort((values[:, 1], values[:, 0]))
    if len(centers) != EXPECTED_STAGE1_MACROSTATE_K or not np.array_equal(ids, expected) or not np.isfinite(values).all() or not np.array_equal(order, expected.astype(int)):
        raise RuntimeError("Stage-1 fixed-K centers are not the expected six ordered states.")
    centers_path = next(path for path in [
        (root / "fixed_k6_centers").with_suffix(".parquet"),
        (root / "fixed_k6_centers").with_suffix(".csv.gz"),
        (root / "fixed_k6_centers").with_suffix(".csv"),
    ] if path.exists())
    return {
        "status": "verified",
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": file_sha256(metadata_path),
        "centers_path": str(centers_path.resolve()),
        "centers_sha256": file_sha256(centers_path),
        "used_in_phase1_parameter_selection": False,
        "checks": checks,
        "metadata": metadata,
    }


def load_ready_handoff(path: Path) -> Tuple[Dict[str, object], Dict[str, float], Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Minimality handoff not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("ready_for_phase2_freeze") is not True:
        raise RuntimeError("The minimality handoff is not ready for Phase 2.")
    if data.get("scalar_parameter_minimality_confirmed") is not True:
        raise RuntimeError("Scalar parameter minimality is not confirmed.")
    if data.get("search_adequacy_confirmed") is not True:
        raise RuntimeError("Minimality search adequacy is not confirmed.")
    if data.get("final_family_key") != EXPECTED_FAMILY:
        raise RuntimeError(f"Expected family {EXPECTED_FAMILY!r}, found {data.get('final_family_key')!r}.")
    free = tuple(str(x) for x in data.get("final_free_mechanism_parameters", []))
    if set(free) != set(EXPECTED_FREE_PARAMS):
        raise RuntimeError(f"Unexpected free parameters: {free}")
    source = dict(data.get("final_selected_parameters", {}))
    seed = {name: float(source[name]) for name in EXPECTED_FREE_PARAMS}
    adjustments: Dict[str, object] = {}
    for name in ("theta0", "thetaM", "phi0"):
        spec = SEARCH_SPECS[name]
        if not spec.lower <= seed[name] <= spec.upper:
            raise RuntimeError(f"Handoff value {name}={seed[name]} is outside the Phase-1 range.")
    if seed["deltaS"] > SEARCH_SPECS["deltaS"].upper:
        adjustments["deltaS"] = {
            "handoff_value": seed["deltaS"],
            "search_seed": SEARCH_SPECS["deltaS"].upper,
            "reason": "values above the verified tanh plateau are not numerically identified",
        }
        seed["deltaS"] = SEARCH_SPECS["deltaS"].upper
    for name in EXPECTED_ZERO_PARAMS:
        if abs(float(source.get(name, 0.0))) > 1e-12:
            raise RuntimeError(f"Structural-zero parameter {name} is non-zero in the handoff.")
    nuisance = dict(data.get("fixed_nuisance_scales", {}))
    for name, expected in FIXED_NUISANCE.items():
        value = float(nuisance.get(name, source.get(name, expected)))
        if not np.isclose(value, expected, atol=1e-12, rtol=0.0):
            raise RuntimeError(f"Unexpected nuisance scale {name}={value}; expected {expected}.")
    return data, seed, {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "seed_adjustments": adjustments,
    }


def load_phase1_panels(stage1_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, float, Dict[str, object]]:
    dynamics = stage1_dynamics_root(stage1_root)
    train_raw = read_core_panel(dynamics, "A_train")
    val_raw = read_core_panel(dynamics, "A_val")
    eta = infer_eta_from_stage1(train_raw)
    train = prepare_panel(train_raw, "A_train", eta)
    val = prepare_panel(val_raw, "A_val", eta)
    train_users = np.asarray(sorted(train["user_id"].unique()), dtype=np.int64)
    val_users = np.asarray(sorted(val["user_id"].unique()), dtype=np.int64)
    overlap = np.intersect1d(train_users, val_users, assume_unique=True)
    if overlap.size:
        raise RuntimeError(f"A_train and A_val share {overlap.size} users.")
    return train, val, eta, {
        "stage1_root": str(Path(stage1_root).resolve()),
        "stage1_dynamics_root": str(dynamics),
        "train_rows_raw": int(len(train_raw)),
        "val_rows_raw": int(len(val_raw)),
        "train_rows_valid": int(len(train)),
        "val_rows_valid": int(len(val)),
        "train_users_valid": int(train_users.size),
        "val_users_valid": int(val_users.size),
        "eta_inferred_from_A_train": float(eta),
        "user_disjoint": True,
        "B_confirm_policy": "not read or used in Phase 1",
    }


def apply_term_switches(params: Mapping[str, float]) -> Dict[str, float]:
    out = {name: float(params.get(name, PARAM_DEFAULTS[name])) for name in PARAM_NAMES}
    for name in EXPECTED_ZERO_PARAMS:
        out[name] = 0.0
    for name, value in FIXED_NUISANCE.items():
        out[name] = float(value)
    return out


def dict_to_params(params: Mapping[str, float]) -> np.ndarray:
    values = apply_term_switches(params)
    return np.asarray([values[name] for name in PARAM_NAMES], dtype=float)


def snap_value_to_grid(name: str, value: float) -> float:
    values = np.asarray(PARAM_GRID_VALUES[name], dtype=float)
    return float(values[int(np.argmin(np.abs(values - float(value))))])


def assert_param_grids_valid() -> None:
    for name in EXPECTED_FREE_PARAMS:
        values = np.asarray(PARAM_GRID_VALUES[name], dtype=float)
        if values.size == 0 or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid grid for {name}.")
    sign_contract = {
        "theta0": np.all(np.asarray(PARAM_GRID_VALUES["theta0"], dtype=float) < 0.0),
        "thetaM": np.all(np.asarray(PARAM_GRID_VALUES["thetaM"], dtype=float) > 0.0),
        "phi0": np.all(np.asarray(PARAM_GRID_VALUES["phi0"], dtype=float) < 0.0),
        "deltaS": np.all(np.asarray(PARAM_GRID_VALUES["deltaS"], dtype=float) > 0.0),
    }
    failed = [name for name, valid in sign_contract.items() if not valid]
    if failed:
        raise RuntimeError(f"Parameter grids violate the selected-family sign contract: {failed}")
    worst_support_argument = SEARCH_SPECS["phi0"].lower + SEARCH_SPECS["deltaS"].upper
    residual = 1.0 - math.tanh(worst_support_argument)
    if residual > CONFIG_DELTA_S_SATURATION_TOL:
        raise RuntimeError("deltaS upper bound does not reach the declared tanh plateau.")


class PhaseProgress:

    def __init__(self, total_steps: int, meta_root: Path, enabled: bool=True) -> None:
        self.total_steps = int(max(total_steps, 1))
        self.meta_root = meta_root
        self.enabled = bool(enabled)
        self.step_idx = 0
        self.start_time = time.time()
        self.last_snapshot = 0.0
        self.snapshot_path = self.meta_root / 'phase1_progress.json'

    @staticmethod
    def _fmt_seconds(seconds: float) -> str:
        if not np.isfinite(seconds) or seconds < 0:
            return 'unknown'
        seconds_i = int(round(seconds))
        hours, rem = divmod(seconds_i, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f'{hours:02d}:{minutes:02d}:{secs:02d}'
        return f'{minutes:02d}:{secs:02d}'

    def _payload(self, message: str) -> dict:
        elapsed = time.time() - self.start_time
        fraction = min(self.step_idx / max(self.total_steps, 1), 1.0)
        eta = elapsed * (1.0 - fraction) / fraction if fraction > 0 else float('inf')
        return {'current_step': int(self.step_idx), 'total_steps': int(self.total_steps), 'fraction_complete': float(fraction), 'message': str(message), 'elapsed_seconds': float(elapsed), 'eta_seconds': float(eta)}

    def update(self, message: str, advance: int=1, force: bool=False) -> None:
        if not self.enabled:
            return
        self.step_idx = min(self.total_steps, self.step_idx + int(max(advance, 0)))
        payload = self._payload(message)
        print(f"[Phase1 progress] {payload['current_step']}/{payload['total_steps']} ({100.0 * payload['fraction_complete']:.1f}%) - {message} | elapsed={self._fmt_seconds(payload['elapsed_seconds'])}, eta={self._fmt_seconds(payload['eta_seconds'])}", flush=True)
        now = time.time()
        if force or now - self.last_snapshot >= CONFIG_PROGRESS_SNAPSHOT_SECONDS:
            save_json(payload, self.snapshot_path)
            self.last_snapshot = now

    def finish(self, message: str='completed') -> None:
        if not self.enabled:
            return
        self.step_idx = self.total_steps
        self.update(message, advance=0, force=True)

def iter_progress(iterable, total: Optional[int]=None, desc: str='', unit: str='item'):
    if CONFIG_PROGRESS and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)
    return iterable

def read_table(base: Path, columns: Optional[Sequence[str]]=None) -> pd.DataFrame:
    for ext in ('.parquet', '.csv.gz', '.csv'):
        p = base.with_suffix(ext)
        if p.exists():
            if ext == '.parquet':
                return pd.read_parquet(p, columns=list(columns) if columns is not None else None)
            return pd.read_csv(p, usecols=list(columns) if columns is not None else None, low_memory=False)
    raise FileNotFoundError(f'Could not find table for {base}')

def write_table(df: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        p = base.with_suffix('.parquet')
        df.to_parquet(p, index=False)
        return p
    except Exception:
        p = base.with_suffix('.csv.gz')
        df.to_csv(p, index=False, compression='gzip')
        return p

def digitize_closed_right(vals: np.ndarray, bins: np.ndarray) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    edges = np.asarray(bins, dtype=float)
    if edges.size == 0:
        return np.full(arr.shape, -1, dtype=np.int64)
    adjusted = np.where(arr == edges[-1], np.nextafter(edges[-1], edges[0]), arr)
    return np.digitize(adjusted, edges) - 1

def js_divergence(p: np.ndarray, q: np.ndarray, eps: float=1e-12) -> float:
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    p = p / max(float(p.sum()), eps)
    q = q / max(float(q.sum()), eps)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + eps) / (b[mask] + eps))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def vector_corr(a_u: np.ndarray, a_v: np.ndarray, b_u: np.ndarray, b_v: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.nan
    a = np.column_stack([np.asarray(a_u)[mask].ravel(), np.asarray(a_v)[mask].ravel()])
    b = np.column_stack([np.asarray(b_u)[mask].ravel(), np.asarray(b_v)[mask].ravel()])
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if ok.sum() < 3:
        return np.nan
    aa = a[ok].ravel()
    bb = b[ok].ravel()
    aa = aa - np.mean(aa)
    bb = bb - np.mean(bb)
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= EPS:
        return np.nan
    return float(np.clip(np.dot(aa, bb) / den, -1.0, 1.0))

def user_balanced_weights(df: pd.DataFrame) -> np.ndarray:
    if df.empty or 'user_id' not in df.columns:
        return np.ones(len(df), dtype=float)
    c = df.groupby('user_id')['user_id'].transform('count').to_numpy(dtype=float)
    return 1.0 / np.maximum(c, 1.0)

def sort_panel(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not {'user_id', 'bundle_step_index'}.issubset(df.columns):
        return df.reset_index(drop=True)
    return df.sort_values(['user_id', 'bundle_step_index'], kind='mergesort').reset_index(drop=True)

def numeric_series(df: pd.DataFrame, col: str, default: float=np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce')
    return pd.Series(default, index=df.index, dtype=float)

def infer_eta_from_stage1(train: pd.DataFrame) -> float:
    e = numeric_series(train, 'response_evidence_mass_pre').to_numpy(dtype=float)
    v = numeric_series(train, 'response_evidence_maturity_V_pre').to_numpy(dtype=float)
    ok = np.isfinite(e) & np.isfinite(v) & (e > 0) & (v > 1e-06) & (v < 1.0 - 1e-06)
    if ok.sum() < 100:
        return float(EVIDENCE_MATURITY_SCALE_DEFAULT)
    eta = e[ok] / np.maximum(-np.log1p(-v[ok]), EPS)
    eta = eta[np.isfinite(eta) & (eta > 0)]
    if eta.size == 0:
        return float(EVIDENCE_MATURITY_SCALE_DEFAULT)
    return float(np.median(eta))

def prepare_panel(df: pd.DataFrame, split_name: str, eta: float) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)
    d['split'] = split_name
    d['user_id'] = numeric_series(df, 'user_id').astype('Int64')
    d['bundle_step_index'] = numeric_series(df, 'bundle_step_index').astype('Int64')
    d['M'] = numeric_series(df, 'M_response_prebalanced_pre')
    d['Psi'] = numeric_series(df, 'activity_alignment_order_Psi_pre')
    d['V'] = numeric_series(df, 'response_evidence_maturity_V_pre')
    d['E'] = numeric_series(df, 'response_evidence_mass_pre')
    inv_e = -float(eta) * np.log1p(-np.clip(d['V'].to_numpy(dtype=float), 0.0, 1.0 - 1e-09))
    e_arr = d['E'].to_numpy(dtype=float)
    bad_e = ~np.isfinite(e_arr) | (e_arr < 0)
    d.loc[bad_e, 'E'] = inv_e[bad_e]
    d['target_M_next'] = numeric_series(df, 'next_M_response_prebalanced')
    d['target_Psi_next'] = numeric_series(df, 'next_activity_alignment_order_Psi')
    d['target_V_next'] = numeric_series(df, 'next_response_evidence_maturity_V')
    d['target_E_next'] = numeric_series(df, 'next_response_evidence_mass')
    target_e_inv = -float(eta) * np.log1p(-np.clip(d['target_V_next'].to_numpy(dtype=float), 0.0, 1.0 - 1e-09))
    te = d['target_E_next'].to_numpy(dtype=float)
    mask_te = ~np.isfinite(te) | (te < 0)
    d.loc[mask_te, 'target_E_next'] = target_e_inv[mask_te]
    d['next_gap_days'] = numeric_series(df, 'next_gap_days')
    d['has_next'] = numeric_series(df, 'has_next_submitted_bundle', default=0).fillna(0).astype(bool)
    bundle_n = numeric_series(df, 'bundle_n_questions')
    answered_fraction = numeric_series(df, 'answered_fraction_interval')
    total_response_count = numeric_series(df, 'total_response_count_diagnostic')
    answered_count = bundle_n * answered_fraction
    answered_count = answered_count.where(np.isfinite(answered_count) & (answered_count > 0), total_response_count)
    answered_count = answered_count.where(np.isfinite(answered_count) & (answered_count > 0), numeric_series(df, 'response_active_mass_interval') * bundle_n)
    d['answered_count_proxy'] = answered_count.clip(lower=0).fillna(0.0)
    response_aligned = numeric_series(df, 'response_aligned_mass_interval').clip(lower=0).fillna(0.0)
    response_off = numeric_series(df, 'response_off_target_mass_interval').clip(lower=0).fillna(0.0)
    response_neutral = numeric_series(df, 'response_neutral_mass_interval').clip(lower=0).fillna(0.0)
    support_aligned = numeric_series(df, 'support_aligned_mass_interval').clip(lower=0).fillna(0.0)
    support_off = numeric_series(df, 'support_off_target_mass_interval').clip(lower=0).fillna(0.0)
    support_neutral = numeric_series(df, 'support_neutral_mass_interval').clip(lower=0).fillna(0.0)
    response_alignable = response_aligned + response_off
    support_alignable = support_aligned + support_off
    response_active = numeric_series(df, 'response_active_mass_interval').clip(lower=0).fillna(0.0)
    support_active = numeric_series(df, 'support_active_total_interval').clip(lower=0).fillna(0.0)
    response_remainder = (response_active - response_alignable - response_neutral).clip(lower=0).fillna(0.0)
    support_remainder = (support_active - support_alignable - support_neutral).clip(lower=0).fillna(0.0)
    response_neutral = response_neutral + response_remainder
    support_neutral = support_neutral + support_remainder
    d['response_alignable_interval'] = response_alignable.clip(lower=0).fillna(0.0)
    d['support_alignable_interval'] = support_alignable.clip(lower=0).fillna(0.0)
    d['response_neutral_interval'] = response_neutral.clip(lower=0).fillna(0.0)
    d['support_neutral_interval'] = support_neutral.clip(lower=0).fillna(0.0)
    d['active_alignable_interval'] = d['response_alignable_interval'] + d['support_alignable_interval']
    d['active_neutral_interval'] = d['response_neutral_interval'] + d['support_neutral_interval']
    d['active_mass_interval'] = d['active_alignable_interval'] + d['active_neutral_interval']
    d['idle_mass_interval'] = numeric_series(df, 'idle_mass_interval').clip(lower=0).fillna(0.0)
    active_pre = numeric_series(df, 'activity_active_mass_pre').clip(lower=0)
    idle_pre = numeric_series(df, 'activity_idle_mass_pre').clip(lower=0)
    B = active_pre + idle_pre
    aligned_pre = numeric_series(df, 'activity_aligned_mass_pre')
    if 'activity_off_target_mass_pre' in df.columns:
        off_pre = numeric_series(df, 'activity_off_target_mass_pre')
    else:
        off_pre = numeric_series(df, 'activity_non_aligned_mass_pre')
    G = aligned_pre - off_pre
    fallback_B = np.maximum(np.abs(G.to_numpy(dtype=float)) / np.maximum(np.abs(d['Psi'].to_numpy(dtype=float)), 1e-06), 1.0)
    B_arr = B.to_numpy(dtype=float)
    B_arr = np.where(np.isfinite(B_arr) & (B_arr > 0), B_arr, fallback_B)
    G_arr = G.to_numpy(dtype=float)
    G_arr = np.where(np.isfinite(G_arr), G_arr, d['Psi'].to_numpy(dtype=float) * B_arr)
    d['B'] = np.maximum(B_arr, EPS)
    d['G'] = np.clip(G_arr, -d['B'].to_numpy(dtype=float), d['B'].to_numpy(dtype=float))
    next_active = numeric_series(df, 'next_activity_active_mass').clip(lower=0)
    next_idle = numeric_series(df, 'next_activity_idle_mass').clip(lower=0)
    target_B_direct = (next_active + next_idle).to_numpy(dtype=float)
    rho_A_for_target = np.exp(-np.maximum(d['next_gap_days'].to_numpy(dtype=float), 0.0) / max(float(TAU_ACTIVITY_DAYS), EPS))
    target_B_fallback = rho_A_for_target * np.maximum(d['B'].to_numpy(dtype=float) + d['active_alignable_interval'].to_numpy(dtype=float) + d['active_neutral_interval'].to_numpy(dtype=float) + d['idle_mass_interval'].to_numpy(dtype=float), EPS)
    use_direct_B = np.isfinite(target_B_direct) & (target_B_direct > 0)
    d['target_B_next'] = np.where(use_direct_B, target_B_direct, target_B_fallback)
    M_resp = numeric_series(df, 'M_response_prebalanced_resp')
    V_resp = numeric_series(df, 'response_evidence_maturity_V_resp')
    Psi_support = numeric_series(df, 'activity_alignment_order_Psi_support')
    Psi_post = numeric_series(df, 'activity_alignment_order_Psi_post')
    d['emp_delta_M_response'] = M_resp - d['M']
    d['emp_delta_V_response'] = V_resp - d['V']
    d['emp_delta_Psi_active'] = Psi_support - d['Psi']
    d['emp_delta_Psi_idle'] = Psi_post - Psi_support
    d['phase_columns_available'] = np.isfinite(d['emp_delta_M_response']) & np.isfinite(d['emp_delta_Psi_active']) & np.isfinite(d['emp_delta_Psi_idle'])
    valid = d['user_id'].notna() & d['bundle_step_index'].notna() & d['has_next'] & np.isfinite(d['M']) & np.isfinite(d['Psi']) & np.isfinite(d['V']) & np.isfinite(d['E']) & np.isfinite(d['target_M_next']) & np.isfinite(d['target_Psi_next']) & np.isfinite(d['target_V_next']) & np.isfinite(d['next_gap_days'])
    d = d.loc[valid].copy()
    d['user_id'] = d['user_id'].astype(int)
    d['bundle_step_index'] = d['bundle_step_index'].astype(int)
    d['M'] = d['M'].clip(-1.0, 1.0)
    d['Psi'] = d['Psi'].clip(-1.0, 1.0)
    d['V'] = d['V'].clip(0.0, 1.0)
    d['E'] = d['E'].clip(lower=0.0)
    d['target_M_next'] = d['target_M_next'].clip(-1.0, 1.0)
    d['target_Psi_next'] = d['target_Psi_next'].clip(-1.0, 1.0)
    d['target_V_next'] = d['target_V_next'].clip(0.0, 1.0)
    d['target_E_next'] = d['target_E_next'].clip(lower=0.0)
    d['target_B_next'] = d['target_B_next'].clip(lower=EPS)
    d['next_gap_days'] = d['next_gap_days'].clip(lower=0.0)
    d = d.sort_values(['user_id', 'bundle_step_index'], kind='mergesort').reset_index(drop=True)
    return d

@dataclass
class Calibration:
    eta: float
    tau_response_days: float
    tau_activity_days: float
    residual_mass_per_answer: float
    lambda_E: float
    response_signed_gain: float = 1.0
    alignment_signed_gain: float = 1.0
    sigma_U0: float = float('nan')
    sigma_Psi0: float = float('nan')

def _robust_fraction_gain(values: np.ndarray, default: float=1.0) -> float:
    vals = np.asarray(values, dtype=float)
    vals = np.abs(vals[np.isfinite(vals)])
    vals = vals[(vals > 1e-06) & (vals <= 5.0)]
    if vals.size < 100:
        return float(default)
    q = float(np.clip(CONFIG_SIGNED_GAIN_QUANTILE, 0.1, 0.95))
    return float(np.clip(np.nanquantile(vals, q), 0.1, 1.0))

def calibrate_from_A_train(train: pd.DataFrame, eta: float, tau_response_days: float, tau_activity_days: float) -> Calibration:
    gap = train['next_gap_days'].to_numpy(dtype=float)
    rho = np.exp(-np.maximum(gap, 0.0) / max(float(tau_response_days), EPS))
    E = train['E'].to_numpy(dtype=float)
    En = train['target_E_next'].to_numpy(dtype=float)
    answered = train['answered_count_proxy'].to_numpy(dtype=float)
    R_eff = En / np.maximum(rho, EPS) - E
    ok = np.isfinite(R_eff) & np.isfinite(answered) & (R_eff > 0) & (answered > 0)
    if ok.sum() >= 100:
        unit_vals = R_eff[ok] / np.maximum(answered[ok], EPS)
        unit_vals = unit_vals[np.isfinite(unit_vals) & (unit_vals > 0)]
        r_unit = float(np.median(unit_vals)) if unit_vals.size else 0.45
    else:
        r_unit = 0.45
    r_unit = float(np.clip(r_unit, 0.02, 2.0))
    M = train['M'].to_numpy(dtype=float)
    Psi = train['Psi'].to_numpy(dtype=float)
    B = train['B'].to_numpy(dtype=float)
    G = train['G'].to_numpy(dtype=float)
    emp_dM_resp = train.get('emp_delta_M_response', pd.Series(np.nan, index=train.index)).to_numpy(dtype=float)
    emp_dV_resp = train.get('emp_delta_V_response', pd.Series(np.nan, index=train.index)).to_numpy(dtype=float)
    V_resp = np.clip(train['V'].to_numpy(dtype=float) + emp_dV_resp, 0.0, 1.0 - 1e-09)
    E_resp = -float(eta) * np.log1p(-V_resp)
    M_resp = np.clip(M + emp_dM_resp, -1.0, 1.0)
    S = M * E
    S_resp = M_resp * E_resp
    R_resp = E_resp - E
    z_response = (S_resp - S) / np.maximum(R_resp, EPS)
    response_signed_gain = _robust_fraction_gain(z_response[np.isfinite(R_resp) & (R_resp > 1e-06)], default=1.0)
    emp_dPsi_active = train.get('emp_delta_Psi_active', pd.Series(np.nan, index=train.index)).to_numpy(dtype=float)
    Psi_active = np.clip(Psi + emp_dPsi_active, -1.0, 1.0)
    Amap = train['active_alignable_interval'].to_numpy(dtype=float)
    A0 = train['active_neutral_interval'].to_numpy(dtype=float)
    B_active = np.maximum(B + Amap + A0, EPS)
    G_active = Psi_active * B_active
    z_alignment = (G_active - G) / np.maximum(Amap, EPS)
    alignment_signed_gain = _robust_fraction_gain(z_alignment[np.isfinite(Amap) & (Amap > 1e-06)], default=1.0)
    e_pos = E[np.isfinite(E) & (E > 0)]
    lambda_E = float(np.median(e_pos)) if e_pos.size else float(eta)
    lambda_E = float(max(lambda_E, 1.0))
    return Calibration(eta=float(eta), tau_response_days=float(tau_response_days), tau_activity_days=float(tau_activity_days), residual_mass_per_answer=r_unit, lambda_E=lambda_E, response_signed_gain=response_signed_gain, alignment_signed_gain=alignment_signed_gain)

def _simulate_core_python(uid: np.ndarray, steps: np.ndarray, M_obs: np.ndarray, Psi_obs: np.ndarray, E_obs: np.ndarray, B_obs: np.ndarray, G_obs: np.ndarray, gap: np.ndarray, answered: np.ndarray, response_alignable: np.ndarray, support_alignable: np.ndarray, response_neutral: np.ndarray, support_neutral: np.ndarray, idle: np.ndarray, params_v: np.ndarray, eta: float, tau_response_days: float, tau_activity_days: float, residual_mass_per_answer: float, response_signed_gain: float, alignment_signed_gain: float) -> Tuple[np.ndarray, ...]:
    n = len(uid)
    sim_M = np.zeros(n, dtype=np.float64)
    sim_Psi = np.zeros(n, dtype=np.float64)
    sim_V = np.zeros(n, dtype=np.float64)
    sim_E = np.zeros(n, dtype=np.float64)
    sim_B = np.zeros(n, dtype=np.float64)
    sim_G = np.zeros(n, dtype=np.float64)
    pred_Mn = np.zeros(n, dtype=np.float64)
    pred_Psin = np.zeros(n, dtype=np.float64)
    pred_Vn = np.zeros(n, dtype=np.float64)
    pred_En = np.zeros(n, dtype=np.float64)
    pred_Bn = np.zeros(n, dtype=np.float64)
    pred_Gn = np.zeros(n, dtype=np.float64)
    pred_delta_M_resp = np.zeros(n, dtype=np.float64)
    pred_delta_V_resp = np.zeros(n, dtype=np.float64)
    pred_delta_Psi_active = np.zeros(n, dtype=np.float64)
    pred_delta_Psi_idle = np.zeros(n, dtype=np.float64)
    scaled_Amap = np.zeros(n, dtype=np.float64)
    scaled_A0 = np.zeros(n, dtype=np.float64)
    scaled_I = np.zeros(n, dtype=np.float64)
    theta0 = float(params_v[0])
    thetaM = float(params_v[1])
    thetaPsi = float(params_v[2])
    thetaMPsi = float(params_v[3])
    phi0 = float(params_v[4])
    deltaS = float(params_v[5])
    phiPsi = float(params_v[6])
    lambdaR = float(params_v[7]) if len(params_v) > 7 else 1.0
    lambdaA = float(params_v[8]) if len(params_v) > 8 else 1.0
    lambdaI = float(params_v[9]) if len(params_v) > 9 else 1.0
    for i in range(n):
        M = float(M_obs[i])
        if M < -1.0:
            M = -1.0
        elif M > 1.0:
            M = 1.0
        Psi = float(Psi_obs[i])
        if Psi < -1.0:
            Psi = -1.0
        elif Psi > 1.0:
            Psi = 1.0
        E = float(max(E_obs[i], 0.0))
        B = float(max(B_obs[i], EPS))
        G = float(G_obs[i])
        if G < -B:
            G = -B
        elif G > B:
            G = B
        sim_M[i] = M
        sim_Psi[i] = Psi
        sim_E[i] = E
        sim_B[i] = B
        sim_G[i] = G
        sim_V[i] = 1.0 - math.exp(-max(E, 0.0) / max(eta, EPS)) if E > 0 else 0.0
        rho_R = math.exp(-max(float(gap[i]), 0.0) / max(tau_response_days, EPS))
        rho_A = math.exp(-max(float(gap[i]), 0.0) / max(tau_activity_days, EPS))
        R = max(lambdaR * float(residual_mass_per_answer) * max(float(answered[i]), 0.0), 0.0)
        S = M * E
        signed_drive = math.tanh(theta0 - thetaM * M + thetaPsi * Psi + thetaMPsi * M * Psi)
        U = max(float(response_signed_gain), 0.0) * R * signed_drive
        E_resp = max(E + R, 0.0)
        S_resp = S + U
        if E_resp > EPS:
            M_resp = float(S_resp / max(E_resp, EPS))
            if M_resp < -1.0:
                M_resp = -1.0
            elif M_resp > 1.0:
                M_resp = 1.0
        else:
            M_resp = 0.0
        V_resp = 1.0 - math.exp(-E_resp / max(eta, EPS)) if E_resp > 0 else 0.0
        E_next = rho_R * E_resp
        S_next = rho_R * S_resp
        if E_next > EPS:
            M_next = float(S_next / max(E_next, EPS))
            if M_next < -1.0:
                M_next = -1.0
            elif M_next > 1.0:
                M_next = 1.0
        else:
            M_next = 0.0
        V_next = 1.0 - math.exp(-E_next / max(eta, EPS)) if E_next > 0 else 0.0
        Aresp = max(lambdaA * float(response_alignable[i]), 0.0)
        Asupp = max(lambdaA * float(support_alignable[i]), 0.0)
        A0resp = max(lambdaA * float(response_neutral[i]), 0.0)
        A0supp = max(lambdaA * float(support_neutral[i]), 0.0)
        Amap = Aresp + Asupp
        A0 = A0resp + A0supp
        I = max(lambdaI * float(idle[i]), 0.0)
        response_align_drive = math.tanh(phi0 - phiPsi * Psi)
        support_align_drive = math.tanh(phi0 + deltaS - phiPsi * Psi)
        H = max(float(alignment_signed_gain), 0.0) * (Aresp * response_align_drive + Asupp * support_align_drive)
        B_active = max(B + Amap + A0, EPS)
        G_active = G + H
        if G_active < -B_active:
            G_active = -B_active
        elif G_active > B_active:
            G_active = B_active
        Psi_active = float(G_active / max(B_active, EPS))
        if Psi_active < -1.0:
            Psi_active = -1.0
        elif Psi_active > 1.0:
            Psi_active = 1.0
        B_idle = max(B_active + I, EPS)
        G_idle = G_active
        Psi_idle = float(G_idle / max(B_idle, EPS))
        if Psi_idle < -1.0:
            Psi_idle = -1.0
        elif Psi_idle > 1.0:
            Psi_idle = 1.0
        B_next = rho_A * B_idle
        G_next = rho_A * G_idle
        if G_next < -B_next:
            G_next = -B_next
        elif G_next > B_next:
            G_next = B_next
        Psi_next = float(G_next / max(B_next, EPS))
        if Psi_next < -1.0:
            Psi_next = -1.0
        elif Psi_next > 1.0:
            Psi_next = 1.0
        pred_Mn[i] = M_next
        pred_Psin[i] = Psi_next
        pred_Vn[i] = V_next
        pred_En[i] = E_next
        pred_Bn[i] = B_next
        pred_Gn[i] = G_next
        pred_delta_M_resp[i] = M_resp - M
        pred_delta_V_resp[i] = V_resp - sim_V[i]
        pred_delta_Psi_active[i] = Psi_active - Psi
        pred_delta_Psi_idle[i] = Psi_idle - Psi_active
        scaled_Amap[i] = Amap
        scaled_A0[i] = A0
        scaled_I[i] = I
    return (sim_M, sim_Psi, sim_V, sim_E, sim_B, sim_G, pred_Mn, pred_Psin, pred_Vn, pred_En, pred_Bn, pred_Gn, pred_delta_M_resp, pred_delta_V_resp, pred_delta_Psi_active, pred_delta_Psi_idle, scaled_Amap, scaled_A0, scaled_I)

if NUMBA_AVAILABLE:
    _simulate_core_numba = njit(cache=True)(_simulate_core_python)
else:
    _simulate_core_numba = None


def _simulate_core_dispatch(uid: np.ndarray, steps: np.ndarray, M_obs: np.ndarray, Psi_obs: np.ndarray, E_obs: np.ndarray, B_obs: np.ndarray, G_obs: np.ndarray, gap: np.ndarray, answered: np.ndarray, response_alignable: np.ndarray, support_alignable: np.ndarray, response_neutral: np.ndarray, support_neutral: np.ndarray, idle: np.ndarray, params_v: np.ndarray, calib: Calibration) -> Tuple[np.ndarray, ...]:
    fn = _simulate_core_numba if CONFIG_USE_NUMBA and _simulate_core_numba is not None else _simulate_core_python
    return fn(uid.astype(np.int64), steps.astype(np.int64), M_obs.astype(np.float64), Psi_obs.astype(np.float64), E_obs.astype(np.float64), B_obs.astype(np.float64), G_obs.astype(np.float64), gap.astype(np.float64), answered.astype(np.float64), response_alignable.astype(np.float64), support_alignable.astype(np.float64), response_neutral.astype(np.float64), support_neutral.astype(np.float64), idle.astype(np.float64), np.asarray(params_v, dtype=np.float64), float(calib.eta), float(calib.tau_response_days), float(calib.tau_activity_days), float(calib.residual_mass_per_answer), float(calib.response_signed_gain), float(calib.alignment_signed_gain))

def simulate_panel(panel: pd.DataFrame, params: Dict[str, float], calib: Calibration) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    d = panel.sort_values(['user_id', 'bundle_step_index'], kind='mergesort').reset_index(drop=True)
    params_v = dict_to_params(params)
    arrays = _simulate_core_dispatch(d['user_id'].to_numpy(dtype=np.int64), d['bundle_step_index'].to_numpy(dtype=np.int64), d['M'].to_numpy(dtype=float), d['Psi'].to_numpy(dtype=float), d['E'].to_numpy(dtype=float), d['B'].to_numpy(dtype=float), d['G'].to_numpy(dtype=float), d['next_gap_days'].to_numpy(dtype=float), d['answered_count_proxy'].to_numpy(dtype=float), d['response_alignable_interval'].to_numpy(dtype=float), d['support_alignable_interval'].to_numpy(dtype=float), d['response_neutral_interval'].to_numpy(dtype=float), d['support_neutral_interval'].to_numpy(dtype=float), d['idle_mass_interval'].to_numpy(dtype=float), params_v, calib)
    sim_M, sim_Psi, sim_V, sim_E, sim_B, sim_G, pred_Mn, pred_Psin, pred_Vn, pred_En, pred_Bn, pred_Gn, pred_delta_M_resp, pred_delta_V_resp, pred_delta_Psi_active, pred_delta_Psi_idle, scaled_Amap, scaled_A0, scaled_I = arrays
    out = d[['split', 'user_id', 'bundle_step_index']].copy()
    out['sim_M'] = sim_M
    out['sim_Psi'] = sim_Psi
    out['sim_V'] = sim_V
    out['sim_E'] = sim_E
    out['sim_B'] = sim_B
    out['sim_G'] = sim_G
    out['pred_next_M'] = pred_Mn
    out['pred_next_Psi'] = pred_Psin
    out['pred_next_V'] = pred_Vn
    out['pred_next_E'] = pred_En
    out['pred_next_B'] = pred_Bn
    out['pred_next_G'] = pred_Gn
    out['pred_delta_M'] = pred_Mn - sim_M
    out['pred_delta_Psi'] = pred_Psin - sim_Psi
    out['pred_delta_V'] = pred_Vn - sim_V
    out['pred_delta_M_response'] = pred_delta_M_resp
    out['pred_delta_V_response'] = pred_delta_V_resp
    out['pred_delta_Psi_active'] = pred_delta_Psi_active
    out['pred_delta_Psi_idle'] = pred_delta_Psi_idle
    out['scaled_active_alignable_interval'] = scaled_Amap
    out['scaled_active_neutral_interval'] = scaled_A0
    out['scaled_idle_mass_interval'] = scaled_I
    return out

def identity_regularization_from_params(params: Dict[str, float]) -> float:
    vals = []
    for key in NUISANCE_PARAM_NAMES:
        lam = float(params.get(key, 1.0))
        lam = max(lam, EPS)
        vals.append(math.log(lam) ** 2)
    return float(math.sqrt(sum(vals)))

def objective_component_values(metrics: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in list(OBJECTIVE_WEIGHTS.keys()) + list(OBJECTIVE_SANITY_LIMITS.keys()):
        val = metrics.get(key, np.nan)
        if not np.isfinite(val):
            val = 1.0
        out[key] = float(np.clip(val, 0.0, 1.0))
    return out

def primary_objective_score(metrics: Dict[str, float]) -> float:
    comps = objective_component_values(metrics)
    total_w = float(sum((max(float(w), 0.0) for w in OBJECTIVE_WEIGHTS.values())))
    if total_w <= EPS:
        return 1.0
    score = 0.0
    for key, w in OBJECTIVE_WEIGHTS.items():
        score += max(float(w), 0.0) * float(comps.get(key, 1.0))
    return float(np.clip(score / max(total_w, EPS), 0.0, 1.0))

def sanity_constraint_penalty(metrics: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    comps = objective_component_values(metrics)
    excess_details: Dict[str, float] = {}
    vals: List[float] = []
    for key, limit in OBJECTIVE_SANITY_LIMITS.items():
        excess = max(0.0, float(comps.get(key, 1.0)) - float(limit))
        excess_details[f'objective_{key}_excess'] = float(excess)
        vals.append(excess * excess)
    penalty = float(math.sqrt(sum(vals))) if vals else 0.0
    return (penalty, excess_details)

def objective_diagnostics(metrics: Dict[str, float]) -> Dict[str, object]:
    comps = objective_component_values(metrics)
    if comps:
        worst_name = max(comps, key=lambda k: comps[k])
        worst_value = float(comps[worst_name])
        saturated_count = int(sum((v >= 1.0 - 1e-12 for v in comps.values())))
    else:
        worst_name = 'none'
        worst_value = 1.0
        saturated_count = 0
    primary_score = primary_objective_score(metrics)
    sanity_penalty, sanity_excess = sanity_constraint_penalty(metrics)
    id_penalty = float(metrics.get('identity_regularization', 0.0))
    if not np.isfinite(id_penalty):
        id_penalty = 0.0
    diag: Dict[str, object] = {'objective_selection_rule': 'primary_MR_PsiA_weighted_score_with_soft_phase_coverage_sanity_constraints', 'objective_primary_score': float(primary_score), 'objective_sanity_penalty': float(sanity_penalty), 'objective_worst_component': worst_name, 'objective_worst_component_value': worst_value, 'objective_saturated_component_count': saturated_count, 'objective_identity_penalty': id_penalty, 'objective_sanity_penalty_weight': float(CONFIG_SANITY_PENALTY_WEIGHT), 'objective_identity_reg_weight': float(CONFIG_IDENTITY_REG_WEIGHT)}
    diag.update(sanity_excess)
    return diag

def objective_from_metrics(metrics: Dict[str, float]) -> float:
    diag = objective_diagnostics(metrics)
    primary_score = float(diag['objective_primary_score'])
    sanity_penalty = float(diag['objective_sanity_penalty'])
    id_penalty = float(diag['objective_identity_penalty'])
    return float(primary_score + CONFIG_SANITY_PENALTY_WEIGHT * sanity_penalty + CONFIG_IDENTITY_REG_WEIGHT * id_penalty)

@dataclass
class SimArrays:
    sim_M: np.ndarray
    sim_Psi: np.ndarray
    sim_V: np.ndarray
    sim_E: np.ndarray
    sim_B: np.ndarray
    sim_G: np.ndarray
    pred_next_M: np.ndarray
    pred_next_Psi: np.ndarray
    pred_next_V: np.ndarray
    pred_next_E: np.ndarray
    pred_next_B: np.ndarray
    pred_next_G: np.ndarray
    pred_delta_M_response: np.ndarray
    pred_delta_V_response: np.ndarray
    pred_delta_Psi_active: np.ndarray
    pred_delta_Psi_idle: np.ndarray
    scaled_active_alignable: np.ndarray
    scaled_active_neutral: np.ndarray
    scaled_idle: np.ndarray

@dataclass
class MetricCache:
    label: str
    n_rows: int
    n_users: int
    uid: np.ndarray
    steps: np.ndarray
    weights: np.ndarray
    M: np.ndarray
    Psi: np.ndarray
    V: np.ndarray
    E: np.ndarray
    B: np.ndarray
    G: np.ndarray
    gap: np.ndarray
    answered: np.ndarray
    response_alignable: np.ndarray
    support_alignable: np.ndarray
    response_neutral: np.ndarray
    support_neutral: np.ndarray
    active_alignable: np.ndarray
    active_neutral: np.ndarray
    idle: np.ndarray
    target_M_next: np.ndarray
    target_Psi_next: np.ndarray
    target_V_next: np.ndarray
    target_E_next: np.ndarray
    target_B_next: np.ndarray
    emp_delta_M_response: np.ndarray
    emp_delta_V_response: np.ndarray
    emp_delta_Psi_active: np.ndarray
    emp_delta_Psi_idle: np.ndarray
    phase_available: np.ndarray
    scale_M: float
    scale_Psi: float
    scale_V: float
    H_obs: np.ndarray
    field_obs: Dict[str, np.ndarray]
    loss_sample_idx: np.ndarray

def occupancy_grid_weighted(x: np.ndarray, y: np.ndarray, weights: np.ndarray, xbins: np.ndarray=GRID_BINS_SIGNED, ybins: np.ndarray=GRID_BINS_SIGNED) -> np.ndarray:
    nx = len(xbins) - 1
    ny = len(ybins) - 1
    ix = digitize_closed_right(x, xbins)
    iy = digitize_closed_right(y, ybins)
    valid = np.isfinite(x) & np.isfinite(y) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not np.any(valid):
        return np.zeros((nx, ny), dtype=float)
    flat = ix[valid] * ny + iy[valid]
    H = np.bincount(flat, weights=np.asarray(weights, dtype=float)[valid], minlength=nx * ny).reshape(nx, ny).astype(float)
    return H / max(float(H.sum()), EPS)

def field_stats_from_arrays_weighted(x: np.ndarray, y: np.ndarray, dx: np.ndarray, dy: np.ndarray, weights: np.ndarray) -> Dict[str, np.ndarray]:
    nx = len(GRID_BINS_SIGNED) - 1
    ny = len(GRID_BINS_SIGNED) - 1
    ix = digitize_closed_right(x, GRID_BINS_SIGNED)
    iy = digitize_closed_right(y, GRID_BINS_SIGNED)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(dx) & np.isfinite(dy) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = ix[valid] * ny + iy[valid]
    w = np.asarray(weights, dtype=float)[valid]
    count = np.bincount(flat, minlength=nx * ny).reshape(nx, ny).astype(float) if np.any(valid) else np.zeros((nx, ny), dtype=float)
    weight = np.bincount(flat, weights=w, minlength=nx * ny).reshape(nx, ny).astype(float) if np.any(valid) else np.zeros((nx, ny), dtype=float)
    sx = np.bincount(flat, weights=w * dx[valid], minlength=nx * ny).reshape(nx, ny).astype(float) if np.any(valid) else np.zeros((nx, ny), dtype=float)
    sy = np.bincount(flat, weights=w * dy[valid], minlength=nx * ny).reshape(nx, ny).astype(float) if np.any(valid) else np.zeros((nx, ny), dtype=float)
    sxx = np.bincount(flat, weights=w * dx[valid] * dx[valid], minlength=nx * ny).reshape(nx, ny).astype(float) if np.any(valid) else np.zeros((nx, ny), dtype=float)
    syy = np.bincount(flat, weights=w * dy[valid] * dy[valid], minlength=nx * ny).reshape(nx, ny).astype(float) if np.any(valid) else np.zeros((nx, ny), dtype=float)
    u = sx / np.maximum(weight, EPS)
    v = sy / np.maximum(weight, EPS)
    diff = np.sqrt(np.maximum(sxx / np.maximum(weight, EPS) - u * u, 0.0) + np.maximum(syy / np.maximum(weight, EPS) - v * v, 0.0))
    mask = count >= MIN_DRIFT_BIN_COUNT
    return {'count': count, 'weight': weight, 'u': u, 'v': v, 'diff': diff, 'mask': mask}

def make_metric_cache(panel: pd.DataFrame, masks: List[np.ndarray], label: str) -> MetricCache:
    d = sort_panel(panel)
    uid = d['user_id'].to_numpy(dtype=np.int64)
    steps = d['bundle_step_index'].to_numpy(dtype=np.int64)
    weights = user_balanced_weights(d)
    M = d['M'].to_numpy(dtype=float)
    Psi = d['Psi'].to_numpy(dtype=float)
    V = d['V'].to_numpy(dtype=float)
    E = d['E'].to_numpy(dtype=float)
    B = d['B'].to_numpy(dtype=float)
    G = d['G'].to_numpy(dtype=float)
    gap = d['next_gap_days'].to_numpy(dtype=float)
    answered = d['answered_count_proxy'].to_numpy(dtype=float)
    response_alignable = d['response_alignable_interval'].to_numpy(dtype=float)
    support_alignable = d['support_alignable_interval'].to_numpy(dtype=float)
    response_neutral = d['response_neutral_interval'].to_numpy(dtype=float)
    support_neutral = d['support_neutral_interval'].to_numpy(dtype=float)
    active_alignable = response_alignable + support_alignable
    active_neutral = response_neutral + support_neutral
    idle = d['idle_mass_interval'].to_numpy(dtype=float)
    target_M = d['target_M_next'].to_numpy(dtype=float)
    target_Psi = d['target_Psi_next'].to_numpy(dtype=float)
    target_V = d['target_V_next'].to_numpy(dtype=float)
    target_E = d['target_E_next'].to_numpy(dtype=float)
    target_B = d['target_B_next'].to_numpy(dtype=float)
    emp_delta_M_response = d['emp_delta_M_response'].to_numpy(dtype=float) if 'emp_delta_M_response' in d.columns else np.full(len(d), np.nan)
    emp_delta_V_response = d['emp_delta_V_response'].to_numpy(dtype=float) if 'emp_delta_V_response' in d.columns else np.full(len(d), np.nan)
    emp_delta_Psi_active = d['emp_delta_Psi_active'].to_numpy(dtype=float) if 'emp_delta_Psi_active' in d.columns else np.full(len(d), np.nan)
    emp_delta_Psi_idle = d['emp_delta_Psi_idle'].to_numpy(dtype=float) if 'emp_delta_Psi_idle' in d.columns else np.full(len(d), np.nan)
    phase_available = d['phase_columns_available'].to_numpy(dtype=bool) if 'phase_columns_available' in d.columns else np.zeros(len(d), dtype=bool)
    H_obs = occupancy_grid_weighted(target_M, target_Psi, weights)
    field_obs = field_stats_from_arrays_weighted(M, Psi, target_M - M, target_Psi - Psi, weights)
    scale_M = max(float(np.nanstd(target_M)), 0.05)
    scale_Psi = max(float(np.nanstd(target_Psi)), 0.05)
    scale_V = max(float(np.nanstd(target_V)), 0.05)
    if len(d) > CONFIG_DISTRIBUTION_LOSS_MAX_ROWS > 0:
        rng = np.random.default_rng(CONFIG_RANDOM_STATE + 4301)
        loss_sample_idx = np.sort(rng.choice(np.arange(len(d)), size=CONFIG_DISTRIBUTION_LOSS_MAX_ROWS, replace=False)).astype(np.int64)
    else:
        loss_sample_idx = np.arange(len(d), dtype=np.int64)
    return MetricCache(label=label, n_rows=int(len(d)), n_users=int(d['user_id'].nunique()), uid=uid, steps=steps, weights=weights, M=M, Psi=Psi, V=V, E=E, B=B, G=G, gap=gap, answered=answered, response_alignable=response_alignable, support_alignable=support_alignable, response_neutral=response_neutral, support_neutral=support_neutral, active_alignable=active_alignable, active_neutral=active_neutral, idle=idle, target_M_next=target_M, target_Psi_next=target_Psi, target_V_next=target_V, target_E_next=target_E, target_B_next=target_B, emp_delta_M_response=emp_delta_M_response, emp_delta_V_response=emp_delta_V_response, emp_delta_Psi_active=emp_delta_Psi_active, emp_delta_Psi_idle=emp_delta_Psi_idle, phase_available=phase_available, scale_M=scale_M, scale_Psi=scale_Psi, scale_V=scale_V, H_obs=H_obs, field_obs=field_obs, loss_sample_idx=loss_sample_idx)

def simulate_arrays(cache: MetricCache, params: Dict[str, float], calib: Calibration) -> SimArrays:
    params_v = dict_to_params(params)
    arrays = _simulate_core_dispatch(cache.uid, cache.steps, cache.M, cache.Psi, cache.E, cache.B, cache.G, cache.gap, cache.answered, cache.response_alignable, cache.support_alignable, cache.response_neutral, cache.support_neutral, cache.idle, params_v, calib)
    sim_M, sim_Psi, sim_V, sim_E, sim_B, sim_G, pred_Mn, pred_Psin, pred_Vn, pred_En, pred_Bn, pred_Gn, pred_delta_M_resp, pred_delta_V_resp, pred_delta_Psi_active, pred_delta_Psi_idle, scaled_Amap, scaled_A0, scaled_I = arrays
    return SimArrays(sim_M=sim_M, sim_Psi=sim_Psi, sim_V=sim_V, sim_E=sim_E, sim_B=sim_B, sim_G=sim_G, pred_next_M=pred_Mn, pred_next_Psi=pred_Psin, pred_next_V=pred_Vn, pred_next_E=pred_En, pred_next_B=pred_Bn, pred_next_G=pred_Gn, pred_delta_M_response=pred_delta_M_resp, pred_delta_V_response=pred_delta_V_resp, pred_delta_Psi_active=pred_delta_Psi_active, pred_delta_Psi_idle=pred_delta_Psi_idle, scaled_active_alignable=scaled_Amap, scaled_active_neutral=scaled_A0, scaled_idle=scaled_I)

def quantile_vector(x: np.ndarray, probs: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=float)
    xx = xx[np.isfinite(xx)]
    if xx.size == 0:
        return np.full(len(probs), np.nan, dtype=float)
    return np.quantile(xx, probs)

def quantile_distance(a: np.ndarray, b: np.ndarray) -> float:
    probs = np.asarray([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], dtype=float)
    qa = quantile_vector(a, probs)
    qb = quantile_vector(b, probs)
    ok = np.isfinite(qa) & np.isfinite(qb)
    if ok.sum() < 3:
        return np.nan
    aa = np.asarray(a, dtype=float)
    aa = aa[np.isfinite(aa)]
    scale = float(np.nanpercentile(aa, 75) - np.nanpercentile(aa, 25)) if aa.size else np.nan
    scale = max(scale, 0.05)
    return float(min(np.linalg.norm(qa[ok] - qb[ok]) / (math.sqrt(ok.sum()) * scale + EPS), 1.0))

def drift_magnitude_loss(field_obs: Dict[str, np.ndarray], field_sim: Dict[str, np.ndarray], mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return 1.0
    speed_obs = np.sqrt(field_obs['u'] * field_obs['u'] + field_obs['v'] * field_obs['v'])
    speed_sim = np.sqrt(field_sim['u'] * field_sim['u'] + field_sim['v'] * field_sim['v'])
    ok = mask & np.isfinite(speed_obs) & np.isfinite(speed_sim)
    if ok.sum() < 3:
        return 1.0
    w = np.asarray(field_obs.get('weight', np.ones_like(speed_obs)), dtype=float)[ok]
    if not np.isfinite(w).any() or float(np.nansum(w)) <= EPS:
        w = np.ones(ok.sum(), dtype=float)
    diff = speed_sim[ok] - speed_obs[ok]
    rmse = math.sqrt(float(np.nansum(w * diff * diff) / max(float(np.nansum(w)), EPS)))
    denom = max(float(np.nanmedian(speed_obs[ok])), 0.0001)
    return float(min(rmse / denom, 1.0))

def drift_local_rmse_loss(field_obs: Dict[str, np.ndarray], field_sim: Dict[str, np.ndarray], mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    ok = mask & np.isfinite(field_obs['u']) & np.isfinite(field_obs['v']) & np.isfinite(field_sim['u']) & np.isfinite(field_sim['v'])
    if ok.sum() < 3:
        return 1.0
    du = field_sim['u'][ok] - field_obs['u'][ok]
    dv = field_sim['v'][ok] - field_obs['v'][ok]
    rmse = math.sqrt(float(np.nanmean(du * du + dv * dv)))
    speed_obs = np.sqrt(field_obs['u'][ok] * field_obs['u'][ok] + field_obs['v'][ok] * field_obs['v'][ok])
    denom = max(float(np.nanpercentile(speed_obs, 75)), 0.0001)
    return float(min(rmse / denom, 1.0))

def phase_loss(cache: MetricCache, sim: SimArrays) -> Tuple[float, Dict[str, float]]:
    idx = cache.loss_sample_idx
    avail = cache.phase_available[idx]
    details: Dict[str, float] = {'phase_available_rows': float(avail.sum())}
    if avail.sum() < 100:
        return (0.0, {**details, 'phase_loss_status': 0.0})
    pairs = {'phase_M_response_qdist': (cache.emp_delta_M_response[idx][avail], sim.pred_delta_M_response[idx][avail]), 'phase_Psi_active_qdist': (cache.emp_delta_Psi_active[idx][avail], sim.pred_delta_Psi_active[idx][avail]), 'phase_Psi_idle_qdist': (cache.emp_delta_Psi_idle[idx][avail], sim.pred_delta_Psi_idle[idx][avail])}
    vals = []
    for key, (a, b) in pairs.items():
        val = quantile_distance(a, b)
        details[key] = float(val) if np.isfinite(val) else np.nan
        if np.isfinite(val):
            vals.append(float(val))
    if not vals:
        return (0.0, details)
    return (float(max(vals)), details)

def coverage_loss(cache: MetricCache, sim: SimArrays) -> Tuple[float, Dict[str, float]]:
    idx = cache.loss_sample_idx
    pairs = {'coverage_B_next_qdist': (cache.target_B_next[idx], sim.pred_next_B[idx]), 'coverage_Amap_qdist': (cache.active_alignable[idx], sim.scaled_active_alignable[idx]), 'coverage_A0_qdist': (cache.active_neutral[idx], sim.scaled_active_neutral[idx]), 'coverage_idle_qdist': (cache.idle[idx], sim.scaled_idle[idx])}
    vals = []
    details: Dict[str, float] = {}
    for key, (a, b) in pairs.items():
        val = quantile_distance(a, b)
        details[key] = float(val) if np.isfinite(val) else np.nan
        if np.isfinite(val):
            vals.append(float(val))
    if not vals:
        return (0.0, details)
    return (float(max(vals)), details)

def structure_metrics_fast_no_regions(cache: MetricCache, sim: SimArrays, label: str) -> Dict[str, float]:
    if cache.n_rows == 0:
        return {'label': label, 'status': 'empty'}
    eM = sim.pred_next_M - cache.target_M_next
    eP = sim.pred_next_Psi - cache.target_Psi_next
    eV = sim.pred_next_V - cache.target_V_next
    mse_main = float(np.nanmean((eM / cache.scale_M) ** 2 + (eP / cache.scale_Psi) ** 2) / 2.0)
    H_sim = occupancy_grid_weighted(sim.pred_next_M, sim.pred_next_Psi, cache.weights)
    occ_js = js_divergence(cache.H_obs + EPS, H_sim + EPS)
    f_sim = field_stats_from_arrays_weighted(cache.M, cache.Psi, sim.pred_next_M - cache.M, sim.pred_next_Psi - cache.Psi, cache.weights)
    mask = cache.field_obs['mask'] & f_sim['mask']
    drift_corr = vector_corr(cache.field_obs['u'], cache.field_obs['v'], f_sim['u'], f_sim['v'], mask)
    drift_dir_loss = float(1.0 if not np.isfinite(drift_corr) else 0.5 * (1.0 - drift_corr))
    drift_mag_loss = drift_magnitude_loss(cache.field_obs, f_sim, mask)
    drift_local_loss = drift_local_rmse_loss(cache.field_obs, f_sim, mask)
    ph_loss, ph_details = phase_loss(cache, sim)
    cov_loss, cov_details = coverage_loss(cache, sim)
    out = {'label': label, 'n_rows': float(cache.n_rows), 'n_users': float(cache.n_users), 'one_step_mse_main_norm': float(min(mse_main, 1.0)), 'one_step_rmse_M': float(math.sqrt(np.nanmean(eM ** 2))), 'one_step_rmse_Psi': float(math.sqrt(np.nanmean(eP ** 2))), 'one_step_rmse_V_diagnostic_only': float(math.sqrt(np.nanmean(eV ** 2))), 'occupancy_js_MR_PsiA': float(occ_js), 'drift_vector_corr_MR_PsiA': float(drift_corr) if np.isfinite(drift_corr) else np.nan, 'drift_direction_loss_MR_PsiA': float(drift_dir_loss), 'drift_magnitude_loss_MR_PsiA': float(drift_mag_loss), 'drift_local_rmse_loss_MR_PsiA': float(drift_local_loss), 'phase_loss_max_qdist': float(ph_loss), 'coverage_loss_max_qdist': float(cov_loss)}
    out.update(ph_details)
    out.update(cov_details)
    return out

def sample_users(panel: pd.DataFrame, max_users: int, seed: int) -> pd.DataFrame:
    if max_users <= 0 or panel.empty:
        return panel.copy()
    users = panel['user_id'].drop_duplicates().to_numpy(dtype=np.int64)
    if len(users) <= max_users:
        return panel.copy()
    rng = np.random.default_rng(seed)
    chosen = set((int(u) for u in rng.choice(users, size=max_users, replace=False).tolist()))
    return panel[panel['user_id'].isin(chosen)].copy()

@dataclass
class CandidateResult:
    params: Dict[str, float]
    metrics: Dict[str, object]
    objective_loss: float


class CandidateEvaluator:
    def __init__(self, cache: MetricCache, calibration: Calibration, scope: str) -> None:
        self.cache = cache
        self.calibration = calibration
        self.scope = scope
        self.results: Dict[Tuple[float, ...], CandidateResult] = {}
        self.rows: List[Dict[str, object]] = []

    def evaluate(self, params: Mapping[str, float], stage: str) -> CandidateResult:
        normalized = apply_term_switches(params)
        key = tuple(round(float(normalized[name]), 8) for name in PARAM_NAMES)
        if key in self.results:
            return self.results[key]
        sim = simulate_arrays(self.cache, normalized, self.calibration)
        metrics = structure_metrics_fast_no_regions(self.cache, sim, self.scope)
        metrics["identity_regularization"] = identity_regularization_from_params(normalized)
        diagnostics = objective_diagnostics(metrics)
        loss = float(objective_from_metrics(metrics))
        row: Dict[str, object] = {
            "evaluation_scope": self.scope,
            "search_stage": stage,
            "objective_loss": loss,
        }
        row.update(normalized)
        row.update(metrics)
        row.update(diagnostics)
        self.rows.append(row)
        result = CandidateResult(normalized, {**metrics, **diagnostics}, loss)
        self.results[key] = result
        return result


def _coarse_values(name: str, seed_value: float) -> List[float]:
    spec = SEARCH_SPECS[name]
    values = _inclusive_grid(spec.lower, spec.upper, spec.coarse_step)
    values.extend([
        snap_value_to_grid(name, seed_value),
        snap_value_to_grid(name, PHASE1_PILOT_REFERENCE[name]),
    ])
    if name == "theta0":
        values = [value for value in values if not np.isclose(value, 0.0)]
    return sorted(set(snap_value_to_grid(name, value) for value in values))


def _local_values(name: str, center: float, radius: float) -> List[float]:
    spec = SEARCH_SPECS[name]
    lower = max(spec.lower, center - radius)
    upper = min(spec.upper, center + radius)
    values = [value for value in PARAM_GRID_VALUES[name] if lower - 1e-12 <= value <= upper + 1e-12]
    values.append(snap_value_to_grid(name, center))
    return sorted(set(values))


def _parameter_key(params: Mapping[str, float]) -> Tuple[float, ...]:
    normalized = apply_term_switches(params)
    return tuple(round(float(normalized[name]), 8) for name in PARAM_NAMES)


def _rank_candidates(evaluator: CandidateEvaluator, candidates: Sequence[Mapping[str, float]], stage: str) -> List[CandidateResult]:
    seen = set()
    results: List[CandidateResult] = []
    for params in iter_progress(candidates, total=len(candidates), desc=stage, unit="candidate"):
        key = _parameter_key(params)
        if key in seen:
            continue
        seen.add(key)
        results.append(evaluator.evaluate(params, stage))
    return sorted(results, key=lambda result: result.objective_loss)


def _coordinate_refine(
    evaluator: CandidateEvaluator,
    start: Mapping[str, float],
    radii: Mapping[str, float],
    passes: int,
    stage: str,
    names: Sequence[str] = EXPECTED_FREE_PARAMS,
) -> CandidateResult:
    current = evaluator.evaluate(start, stage)
    for _ in range(max(1, int(passes))):
        improved = False
        for name in names:
            candidates = []
            for value in _local_values(name, current.params[name], radii[name]):
                params = dict(current.params)
                params[name] = value
                candidates.append(params)
            best = min((evaluator.evaluate(params, stage) for params in candidates), key=lambda result: result.objective_loss)
            if best.objective_loss + 1e-12 < current.objective_loss:
                current = best
                improved = True
        if not improved:
            break
    return current


def _delta_s_plateau_select(evaluator: CandidateEvaluator, start: CandidateResult) -> Tuple[CandidateResult, Dict[str, object]]:
    profile: List[CandidateResult] = []
    for value in PARAM_GRID_VALUES["deltaS"]:
        params = dict(start.params)
        params["deltaS"] = value
        profile.append(evaluator.evaluate(params, "deltaS_profile"))
    minimum = min(result.objective_loss for result in profile)
    eligible = []
    for result in profile:
        residual = 1.0 - math.tanh(result.params["phi0"] + result.params["deltaS"])
        if result.objective_loss <= minimum + CONFIG_DELTA_S_OBJECTIVE_TOL and residual <= CONFIG_DELTA_S_SATURATION_TOL:
            eligible.append((result.params["deltaS"], result, residual))
    if eligible:
        _, selected, residual = min(eligible, key=lambda item: item[0])
        rule = "smallest finite plateau representative within objective tolerance"
    else:
        selected = min(profile, key=lambda result: result.objective_loss)
        residual = 1.0 - math.tanh(selected.params["phi0"] + selected.params["deltaS"])
        rule = "minimum objective because no plateau-equivalent candidate passed"
    return selected, {
        "selection_rule": rule,
        "profile_minimum_objective": float(minimum),
        "selected_deltaS": float(selected.params["deltaS"]),
        "selected_support_channel_residual_to_one": float(residual),
        "objective_tolerance": float(CONFIG_DELTA_S_OBJECTIVE_TOL),
        "saturation_tolerance": float(CONFIG_DELTA_S_SATURATION_TOL),
        "profile": [
            {
                "deltaS": float(result.params["deltaS"]),
                "objective_loss": float(result.objective_loss),
                "support_channel_residual_to_one": float(1.0 - math.tanh(result.params["phi0"] + result.params["deltaS"])),
            }
            for result in profile
        ],
    }


def hierarchical_grid_search(
    search_cache: MetricCache,
    full_cache: MetricCache,
    calibration: Calibration,
    seed: Mapping[str, float],
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, object]]:
    subset = CandidateEvaluator(search_cache, calibration, "search_subset")
    full = CandidateEvaluator(full_cache, calibration, "full_development")
    anchor = apply_term_switches(seed)
    pilot = apply_term_switches(PHASE1_PILOT_REFERENCE)

    response_candidates = []
    for theta0, thetaM in product(_coarse_values("theta0", anchor["theta0"]), _coarse_values("thetaM", anchor["thetaM"])):
        params = dict(anchor)
        params.update({"theta0": theta0, "thetaM": thetaM})
        response_candidates.append(params)
    response_top = _rank_candidates(subset, response_candidates, "coarse_response_block")[:CONFIG_BLOCK_TOP_K]

    alignment_candidates = []
    for phi0, delta_s in product(_coarse_values("phi0", anchor["phi0"]), _coarse_values("deltaS", anchor["deltaS"])):
        params = dict(anchor)
        params.update({"phi0": phi0, "deltaS": delta_s})
        alignment_candidates.append(params)
    alignment_top = _rank_candidates(subset, alignment_candidates, "coarse_alignment_block")[:CONFIG_BLOCK_TOP_K]

    joint_candidates = [anchor, pilot]
    for response, alignment in product(response_top, alignment_top):
        params = dict(anchor)
        params.update({
            "theta0": response.params["theta0"],
            "thetaM": response.params["thetaM"],
            "phi0": alignment.params["phi0"],
            "deltaS": alignment.params["deltaS"],
        })
        joint_candidates.append(params)
    joint_top = _rank_candidates(subset, joint_candidates, "coarse_joint_block")[:CONFIG_FINE_STARTS]

    subset_radii = {name: SEARCH_SPECS[name].subset_radius for name in EXPECTED_FREE_PARAMS}
    refined = [
        _coordinate_refine(subset, result.params, subset_radii, CONFIG_FINE_REFINE_PASSES, "fine_subset")
        for result in joint_top
    ]
    subset_ranked = sorted(subset.results.values(), key=lambda result: result.objective_loss)
    shortlist = []
    seen = set()
    for params in [anchor, pilot] + [result.params for result in refined + subset_ranked]:
        key = _parameter_key(params)
        if key in seen:
            continue
        seen.add(key)
        shortlist.append(apply_term_switches(params))
        if len(shortlist) >= CONFIG_FULL_SHORTLIST_K:
            break

    full_ranked = _rank_candidates(full, shortlist, "full_shortlist")
    full_radii = {name: SEARCH_SPECS[name].full_radius for name in EXPECTED_FREE_PARAMS}
    full_refined = [
        _coordinate_refine(full, result.params, full_radii, CONFIG_FULL_REFINE_PASSES, "fine_full")
        for result in full_ranked[:CONFIG_FULL_REFINE_STARTS]
    ]
    best = min(full_refined + full_ranked, key=lambda result: result.objective_loss)
    best, plateau_audit = _delta_s_plateau_select(full, best)
    continuous_names = ("theta0", "thetaM", "phi0")
    best = _coordinate_refine(full, best.params, full_radii, 1, "post_plateau_continuous_refine", continuous_names)
    best, plateau_audit_final = _delta_s_plateau_select(full, best)

    rows = pd.DataFrame(subset.rows + full.rows)
    if not rows.empty:
        rows = rows.sort_values(["evaluation_scope", "objective_loss"], kind="mergesort").reset_index(drop=True)
    audit = {
        "method": "deterministic hierarchical range grid",
        "final_precision": {name: spec.precision for name, spec in SEARCH_SPECS.items()},
        "ranges": {name: {"lower": spec.lower, "upper": spec.upper, "coarse_step": spec.coarse_step, "fine_step": spec.precision} for name, spec in SEARCH_SPECS.items()},
        "range_rationale": SEARCH_RANGE_RATIONALE,
        "handoff_seed": {name: float(anchor[name]) for name in PARAM_NAMES},
        "archived_phase1_best": ARCHIVED_PHASE1_BEST,
        "pilot_reference_after_deltaS_plateau_mapping": {name: float(pilot[name]) for name in PARAM_NAMES},
        "search_subset_rows": int(search_cache.n_rows),
        "search_subset_users": int(search_cache.n_users),
        "full_development_rows": int(full_cache.n_rows),
        "full_development_users": int(full_cache.n_users),
        "coarse_response_candidates": int(len(response_candidates)),
        "coarse_alignment_candidates": int(len(alignment_candidates)),
        "coarse_joint_candidates": int(len(joint_candidates)),
        "subset_unique_evaluations": int(len(subset.results)),
        "full_unique_evaluations": int(len(full.results)),
        "deltaS_plateau_initial": plateau_audit,
        "deltaS_plateau_final": plateau_audit_final,
        "selected_parameters": {name: float(best.params[name]) for name in PARAM_NAMES},
        "selected_objective_loss": float(best.objective_loss),
    }
    return best.params, rows, audit


def prediction_frame_from_cache(cache: MetricCache, sim: SimArrays, label: str, max_rows: int, seed: int) -> pd.DataFrame:
    n = int(cache.n_rows)
    if max_rows <= 0 or n <= max_rows:
        index = np.arange(n, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        index = np.sort(rng.choice(np.arange(n, dtype=np.int64), size=max_rows, replace=False))
    return pd.DataFrame({
        "label": label,
        "user_id": cache.uid[index],
        "bundle_step_index": cache.steps[index],
        "M": cache.M[index],
        "Psi": cache.Psi[index],
        "V_diagnostic": cache.V[index],
        "E_accounting": cache.E[index],
        "B_accounting": cache.B[index],
        "G_accounting": cache.G[index],
        "target_M_next": cache.target_M_next[index],
        "target_Psi_next": cache.target_Psi_next[index],
        "pred_next_M": sim.pred_next_M[index],
        "pred_next_Psi": sim.pred_next_Psi[index],
        "pred_delta_M": sim.pred_next_M[index] - cache.M[index],
        "pred_delta_Psi": sim.pred_next_Psi[index] - cache.Psi[index],
        "pred_delta_M_response": sim.pred_delta_M_response[index],
        "pred_delta_Psi_active": sim.pred_delta_Psi_active[index],
        "pred_delta_Psi_idle": sim.pred_delta_Psi_idle[index],
        "scaled_active_alignable_interval": sim.scaled_active_alignable[index],
        "scaled_active_neutral_interval": sim.scaled_active_neutral[index],
        "scaled_idle_mass_interval": sim.scaled_idle[index],
    })


def field_grid_table(cache: MetricCache, sim: SimArrays, label: str) -> pd.DataFrame:
    model_occupancy = occupancy_grid_weighted(sim.pred_next_M, sim.pred_next_Psi, cache.weights)
    model_field = field_stats_from_arrays_weighted(
        cache.M,
        cache.Psi,
        sim.pred_next_M - cache.M,
        sim.pred_next_Psi - cache.Psi,
        cache.weights,
    )
    centers = 0.5 * (GRID_BINS_SIGNED[:-1] + GRID_BINS_SIGNED[1:])
    rows = []
    for i, m_value in enumerate(centers):
        for j, psi_value in enumerate(centers):
            empirical_u = float(cache.field_obs["u"][i, j])
            empirical_v = float(cache.field_obs["v"][i, j])
            model_u = float(model_field["u"][i, j])
            model_v = float(model_field["v"][i, j])
            rows.append({
                "label": label,
                "M_center": float(m_value),
                "Psi_center": float(psi_value),
                "empirical_next_occupancy": float(cache.H_obs[i, j]),
                "mechanism_next_occupancy": float(model_occupancy[i, j]),
                "empirical_drift_M": empirical_u,
                "empirical_drift_Psi": empirical_v,
                "mechanism_drift_M": model_u,
                "mechanism_drift_Psi": model_v,
                "drift_residual_magnitude": float(math.hypot(model_u - empirical_u, model_v - empirical_v)),
                "empirical_supported": bool(cache.field_obs["mask"][i, j]),
                "mechanism_supported": bool(model_field["mask"][i, j]),
                "common_supported": bool(cache.field_obs["mask"][i, j] and model_field["mask"][i, j]),
            })
    return pd.DataFrame(rows)


def selected_law_tables(params: Mapping[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    values = apply_term_switches(params)
    m_grid = np.round(np.linspace(-1.0, 1.0, 201), 2)
    response = pd.DataFrame({
        "M": m_grid,
        "signed_response_drive": np.tanh(values["theta0"] - values["thetaM"] * m_grid),
    })
    support_share = np.linspace(0.0, 1.0, 101)
    response_drive = math.tanh(values["phi0"])
    support_drive = math.tanh(values["phi0"] + values["deltaS"])
    alignment = pd.DataFrame({
        "support_share": support_share,
        "response_channel_drive": response_drive,
        "support_channel_drive": support_drive,
        "mixed_alignment_drive": (1.0 - support_share) * response_drive + support_share * support_drive,
    })
    return response, alignment


def estimate_noise_from_cache(cache: MetricCache, sim: SimArrays, calibration: Calibration) -> Dict[str, float]:
    scale = np.sqrt(np.maximum(sim.sim_E, 0.0) + max(calibration.lambda_E, 1.0))
    error_m = sim.pred_next_M - cache.target_M_next
    error_psi = sim.pred_next_Psi - cache.target_Psi_next
    return {
        "sigma_U0_maturity_scaled_residual": float(np.nanstd(error_m * scale)),
        "sigma_Psi0_maturity_scaled_residual": float(np.nanstd(error_psi * scale)),
    }


def generate_figures(panel: pd.DataFrame, pred: pd.DataFrame, masks: List[np.ndarray], label: str, fig_dir: Path) -> Dict[str, object]:
    del masks
    data_dir = Path(fig_dir) / "tables"
    data_dir.mkdir(parents=True, exist_ok=True)
    d = sort_panel(panel)
    p = sort_panel(pred)
    weights = user_balanced_weights(d)
    empirical_occupancy = occupancy_grid_weighted(d["target_M_next"].to_numpy(dtype=float), d["target_Psi_next"].to_numpy(dtype=float), weights)
    mechanism_occupancy = occupancy_grid_weighted(p["pred_next_M"].to_numpy(dtype=float), p["pred_next_Psi"].to_numpy(dtype=float), weights)
    empirical_field = field_stats_from_arrays_weighted(d["M"].to_numpy(dtype=float), d["Psi"].to_numpy(dtype=float), d["target_M_next"].to_numpy(dtype=float) - d["M"].to_numpy(dtype=float), d["target_Psi_next"].to_numpy(dtype=float) - d["Psi"].to_numpy(dtype=float), weights)
    mechanism_field = field_stats_from_arrays_weighted(d["M"].to_numpy(dtype=float), d["Psi"].to_numpy(dtype=float), p["pred_next_M"].to_numpy(dtype=float) - d["M"].to_numpy(dtype=float), p["pred_next_Psi"].to_numpy(dtype=float) - d["Psi"].to_numpy(dtype=float), weights)
    centers = 0.5 * (GRID_BINS_SIGNED[:-1] + GRID_BINS_SIGNED[1:])
    rows = []
    for i, m_value in enumerate(centers):
        for j, psi_value in enumerate(centers):
            rows.append({
                "label": label,
                "M_center": float(m_value),
                "Psi_center": float(psi_value),
                "empirical_next_occupancy": float(empirical_occupancy[i, j]),
                "mechanism_next_occupancy": float(mechanism_occupancy[i, j]),
                "empirical_drift_M": float(empirical_field["u"][i, j]),
                "empirical_drift_Psi": float(empirical_field["v"][i, j]),
                "mechanism_drift_M": float(mechanism_field["u"][i, j]),
                "mechanism_drift_Psi": float(mechanism_field["v"][i, j]),
                "common_supported": bool(empirical_field["mask"][i, j] and mechanism_field["mask"][i, j]),
            })
    output = write_table(pd.DataFrame(rows), data_dir / f"{label}_mechanism_figure_data")
    return {
        "images_written": False,
        "figure_data_path": str(output),
        "policy": "visualization is generated by a separate publication script",
    }


def main() -> None:
    start_time = time.time()
    table_root = CONFIG_OUTPUT_ROOT / "tables"
    meta_root = CONFIG_OUTPUT_ROOT / "metadata"
    table_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)
    progress = PhaseProgress(12, meta_root, CONFIG_PROGRESS)

    progress.update("validating minimality handoff", force=True)
    handoff, seed, handoff_audit = load_ready_handoff(CONFIG_HANDOFF_PATH)
    save_json(handoff, meta_root / "phase1_minimality_handoff_snapshot.json")

    progress.update("checking fixed K=6 Stage-1 contract")
    kmeans_audit = audit_stage1_kmeans_contract(CONFIG_STAGE1_ROOT)
    save_json(kmeans_audit, meta_root / "stage1_fixed_k6_contract.json")

    progress.update("loading A_train and A_val")
    train, val, eta, load_manifest = load_phase1_panels(CONFIG_STAGE1_ROOT)
    if CONFIG_SMOKE_TEST_MODE:
        train = sample_users(train, CONFIG_SMOKE_TEST_MAX_USERS_PER_SPLIT, CONFIG_RANDOM_STATE + 1)
        val = sample_users(val, CONFIG_SMOKE_TEST_MAX_USERS_PER_SPLIT, CONFIG_RANDOM_STATE + 2)
    if train.empty or val.empty:
        raise RuntimeError("A_train or A_val is empty after filtering.")

    progress.update("calibrating A_train accounting")
    calibration = calibrate_from_A_train(train, eta, TAU_RESPONSE_DAYS, TAU_ACTIVITY_DAYS)
    save_json(asdict(calibration), meta_root / "A_train_only_skeleton_calibration.json")
    assert_param_grids_valid()

    progress.update("building search and full-development caches")
    total_users = int(train["user_id"].nunique() + val["user_id"].nunique())
    search_total = min(max(CONFIG_SEARCH_TOTAL_USERS, 2), total_users)
    train_share = train["user_id"].nunique() / max(total_users, 1)
    search_train_users = max(1, int(round(search_total * train_share)))
    search_val_users = max(1, search_total - search_train_users)
    search_train = sample_users(train, search_train_users, CONFIG_RANDOM_STATE + 11)
    search_val = sample_users(val, search_val_users, CONFIG_RANDOM_STATE + 12)
    search_panel = sort_panel(pd.concat([search_train, search_val], ignore_index=True, sort=False))
    full_panel = sort_panel(pd.concat([train, val], ignore_index=True, sort=False))
    search_cache = make_metric_cache(search_panel, [], "search_subset")
    full_cache = make_metric_cache(full_panel, [], "A_train_plus_A_val")

    progress.update("running hierarchical range-grid search")
    selected, search_results, search_audit = hierarchical_grid_search(search_cache, full_cache, calibration, seed)
    search_path = write_table(search_results, table_root / "phase1_parameter_search_results")
    save_json(search_audit, meta_root / "phase1_parameter_search_audit.json")

    progress.update("evaluating selected mechanism on pooled development data")
    pooled_sim = simulate_arrays(full_cache, selected, calibration)
    pooled_metrics = structure_metrics_fast_no_regions(full_cache, pooled_sim, "A_train_plus_A_val")
    pooled_metrics["identity_regularization"] = identity_regularization_from_params(selected)
    pooled_metrics["objective_loss"] = objective_from_metrics(pooled_metrics)
    pooled_metrics.update(objective_diagnostics(pooled_metrics))
    noise = estimate_noise_from_cache(full_cache, pooled_sim, calibration)
    calibration.sigma_U0 = float(noise["sigma_U0_maturity_scaled_residual"])
    calibration.sigma_Psi0 = float(noise["sigma_Psi0_maturity_scaled_residual"])
    save_json(asdict(calibration), meta_root / "phase1_calibration_with_postfit_noise_estimates.json")

    progress.update("evaluating selected mechanism by split")
    metrics_rows = [pooled_metrics]
    audit_frames = []
    output_paths: Dict[str, object] = {}
    for index, (label, panel) in enumerate((("A_train", train), ("A_val", val))):
        cache = make_metric_cache(panel, [], label)
        sim = simulate_arrays(cache, selected, calibration)
        metrics = structure_metrics_fast_no_regions(cache, sim, label)
        metrics["identity_regularization"] = identity_regularization_from_params(selected)
        metrics["objective_loss"] = objective_from_metrics(metrics)
        metrics.update(objective_diagnostics(metrics))
        metrics_rows.append(metrics)
        output_paths[f"{label}_field_grid"] = str(write_table(field_grid_table(cache, sim, label), table_root / f"phase1_{label}_field_grid"))
        sample_rows = max(1, CONFIG_PREDICTION_AUDIT_ROWS // 2)
        audit_frames.append(prediction_frame_from_cache(cache, sim, label, sample_rows, CONFIG_RANDOM_STATE + 30 + index))
        if CONFIG_WRITE_FULL_PREDICTIONS:
            output_paths[f"{label}_full_predictions"] = str(write_table(prediction_frame_from_cache(cache, sim, label, 0, CONFIG_RANDOM_STATE), table_root / f"phase1_{label}_full_predictions"))

    progress.update("writing metrics and figure-data tables")
    metrics_path = write_table(pd.DataFrame(metrics_rows), table_root / "phase1_structural_alignment_metrics")
    audit_path = write_table(pd.concat(audit_frames, ignore_index=True), table_root / "phase1_prediction_audit_sample")
    response_law, alignment_law = selected_law_tables(selected)
    response_law_path = write_table(response_law, table_root / "phase1_selected_response_law")
    alignment_law_path = write_table(alignment_law, table_root / "phase1_selected_alignment_law")

    progress.update("writing selected-parameter contract")
    selected_payload = {
        "selected_parameters": {name: float(selected[name]) for name in PARAM_NAMES},
        "post_ablation_selected_family": EXPECTED_FAMILY,
        "post_ablation_selected_family_label": "Offset dual-channel",
        "minimality_handoff_ready": True,
        "minimality_handoff_scalar_minimality_confirmed": True,
        "free_mechanism_parameters": list(EXPECTED_FREE_PARAMS),
        "fixed_zero_mechanism_parameters": list(EXPECTED_ZERO_PARAMS),
        "fixed_nuisance_scales": fixed_nuisance_scales(),
        "primary_macrostate": ["M", "Psi"],
        "auxiliary_accounting": ["E", "B", "G"],
        "maturity_diagnostic_role": "V is retained only for evidence-mass reconstruction, signed-gain calibration, and downstream noise diagnostics",
        "parameter_search_ranges": {name: asdict(spec) for name, spec in SEARCH_SPECS.items()},
        "parameter_search_range_rationale": SEARCH_RANGE_RATIONALE,
        "archived_phase1_best": ARCHIVED_PHASE1_BEST,
        "pilot_reference_after_deltaS_plateau_mapping": {
            name: float(PHASE1_PILOT_REFERENCE[name]) for name in PARAM_NAMES
        },
        "parameter_grid_values": PARAM_GRID_VALUES,
        "deltaS_plateau_audit": search_audit["deltaS_plateau_final"],
        "handoff_audit": handoff_audit,
    }
    save_json(selected_payload, meta_root / "phase1_selected_parameters.json")

    progress.update("writing manifest")
    manifest = {
        "script": Path(__file__).name,
        "phase": "Phase 1 fixed-family tuning after minimality selection",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage1_load_manifest": load_manifest,
        "stage1_fixed_k6_contract": kmeans_audit,
        "minimality_handoff": handoff_audit,
        "family": EXPECTED_FAMILY,
        "primary_macrostate": ["M", "Psi"],
        "auxiliary_accounting": ["E", "B", "G"],
        "maturity_diagnostic": "V is not a selection coordinate or objective term",
        "B_confirm_policy": "not read or used",
        "objective_weights": OBJECTIVE_WEIGHTS,
        "objective_sanity_limits": OBJECTIVE_SANITY_LIMITS,
        "selected_parameters": selected_payload["selected_parameters"],
        "search_protocol": search_audit,
        "visualization_policy": "no image output; numerical figure data are written as tables",
        "runtime_configuration": {
            "search_total_users": int(CONFIG_SEARCH_TOTAL_USERS),
            "search_train_users": int(search_train["user_id"].nunique()),
            "search_val_users": int(search_val["user_id"].nunique()),
            "block_top_k": int(CONFIG_BLOCK_TOP_K),
            "fine_starts": int(CONFIG_FINE_STARTS),
            "fine_refine_passes": int(CONFIG_FINE_REFINE_PASSES),
            "full_shortlist_k": int(CONFIG_FULL_SHORTLIST_K),
            "full_refine_starts": int(CONFIG_FULL_REFINE_STARTS),
            "full_refine_passes": int(CONFIG_FULL_REFINE_PASSES),
            "use_numba": bool(CONFIG_USE_NUMBA),
            "numba_available": bool(NUMBA_AVAILABLE),
            "write_full_predictions": bool(CONFIG_WRITE_FULL_PREDICTIONS),
            "random_state": int(CONFIG_RANDOM_STATE),
        },
        "outputs": {
            "parameter_search_results": str(search_path),
            "structural_alignment_metrics": str(metrics_path),
            "prediction_audit_sample": str(audit_path),
            "response_law_table": str(response_law_path),
            "alignment_law_table": str(alignment_law_path),
            **output_paths,
        },
        "elapsed_seconds": float(time.time() - start_time),
    }
    save_json(manifest, meta_root / "phase1_manifest.json")
    progress.finish("Phase 1 complete")
    print(f"Selected parameters: {selected_payload['selected_parameters']}")
    print(f"Output root: {CONFIG_OUTPUT_ROOT}")


if __name__ == "__main__":
    main()

