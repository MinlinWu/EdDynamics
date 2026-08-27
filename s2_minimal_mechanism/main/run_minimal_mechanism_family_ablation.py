#!/usr/bin/env python3
from __future__ import annotations

"""Run the Phase-1 seven-term mechanism-family minimality experiment.

The script reads frozen A_train/A_val panels, evaluates structural-zero
families on the M-Psi plane, performs paired-user bootstrap comparison and
scalar deletion, and writes the Phase-2 handoff. B_confirm is not read. The
Stage-1 fixed K=6 partition is audited but is not a selection target.
"""

PUBLIC_RELEASE_VERSION = "4.2.0"

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from numba import njit, prange, set_num_threads
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    njit = None
    prange = range
    set_num_threads = None
    NUMBA_AVAILABLE = False

# Runtime defaults.
DEFAULT_STAGE1_ROOT = Path(os.environ.get(
    "MECH_MINIMALITY_STAGE1_ROOT",
    "/data/datasets/KT4/outputs_KT4/stage1",
))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get(
    "MECH_MINIMALITY_OUTPUT_ROOT",
    "/data/datasets/KT4/outputs_KT4/stage2_phase1_unified_minimality",
))

CONFIG_RANDOM_STATE = int(os.environ.get("MECH_MINIMALITY_RANDOM_STATE", "42"))
CONFIG_USE_NUMBA = bool(int(os.environ.get("MECH_MINIMALITY_USE_NUMBA", "1"))) and NUMBA_AVAILABLE
CONFIG_DISTRIBUTION_LOSS_MAX_ROWS = int(os.environ.get("MECH_PHASE1_DISTRIBUTION_LOSS_MAX_ROWS", "200000"))
CONFIG_SIGNED_GAIN_QUANTILE = float(os.environ.get("MECH_PHASE1_SIGNED_GAIN_QUANTILE", "0.75"))
CONFIG_SANITY_PENALTY_WEIGHT = float(os.environ.get("MECH_PHASE1_SANITY_PENALTY_WEIGHT", "0.25"))
CONFIG_IDENTITY_REG_WEIGHT = float(os.environ.get("MECH_PHASE1_IDENTITY_REG_WEIGHT", "0.05"))

SCREENING_TRAIN_USERS = int(os.environ.get("MECH_MINIMALITY_SCREENING_TRAIN_USERS", "20000"))
SCREENING_MAX_CANDIDATES_PER_FAMILY = int(os.environ.get("MECH_MINIMALITY_SCREENING_MAX_CANDIDATES", "96"))
FULL_TRAIN_TOP_K = int(os.environ.get("MECH_MINIMALITY_FULL_TRAIN_TOP_K", "16"))
VAL_SHORTLIST_K = int(os.environ.get("MECH_MINIMALITY_VAL_SHORTLIST_K", "8"))
LOCAL_REFINE_MAX_EVALS = int(os.environ.get("MECH_MINIMALITY_LOCAL_REFINE_MAX_EVALS", "48"))
DELETION_EXHAUSTIVE_MAX_COMBINATIONS = int(os.environ.get(
    "MECH_MINIMALITY_DELETION_EXHAUSTIVE_MAX_COMBINATIONS", "5000"
))
DELETION_FULL_TRAIN_TOP_K = int(os.environ.get(
    "MECH_MINIMALITY_DELETION_FULL_TRAIN_TOP_K", "32"
))
DELETION_VAL_SHORTLIST_K = int(os.environ.get(
    "MECH_MINIMALITY_DELETION_VAL_SHORTLIST_K", "16"
))
DELETION_LOCAL_REFINE_MAX_EVALS = int(os.environ.get(
    "MECH_MINIMALITY_DELETION_LOCAL_REFINE_MAX_EVALS", "96"
))
DELETION_REFINE_STARTS = int(os.environ.get(
    "MECH_MINIMALITY_DELETION_REFINE_STARTS", "5"
))
DECISION_BOOTSTRAP_SEED_OFFSET = int(os.environ.get(
    "MECH_MINIMALITY_DECISION_BOOTSTRAP_SEED_OFFSET", "777"
))
BOOTSTRAP_REPS = int(os.environ.get("MECH_MINIMALITY_BOOTSTRAP_REPS", "300"))
PRACTICAL_EQ_MARGIN = float(os.environ.get("MECH_MINIMALITY_EQ_MARGIN", "0.02"))
MARGIN_SENSITIVITY_VALUES = [
    float(x) for x in os.environ.get(
        "MECH_MINIMALITY_MARGIN_SENSITIVITY",
        "0.010,0.015,0.020,0.025,0.030",
    ).split(",") if x.strip()
]
BOOTSTRAP_ENGINE = os.environ.get("MECH_MINIMALITY_BOOTSTRAP_ENGINE", "optimized").strip().lower()
VERIFY_OPTIMIZED_BOOTSTRAP = bool(int(os.environ.get("MECH_MINIMALITY_VERIFY_BOOTSTRAP", "1")))
VERIFY_BOOTSTRAP_REPS = int(os.environ.get("MECH_MINIMALITY_VERIFY_BOOTSTRAP_REPS", "2"))

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
MECH_PARAMS = MECHANISM_PARAM_NAMES
ALL_PARAMS = PARAM_NAMES

FIXED_NUISANCE = {
    "lambdaR": float(os.environ.get("MECH_MINIMALITY_LAMBDAR", "0.46")),
    "lambdaA": float(os.environ.get("MECH_MINIMALITY_LAMBDAA", "1.10")),
    "lambdaI": float(os.environ.get("MECH_MINIMALITY_LAMBDAI", "0.85")),
}

# Finite grids for exact reproduction and compact reruns.
PUBLICATION_GRID: Dict[str, List[float]] = {
    "theta0": [-0.72, -0.60, -0.48, -0.36, -0.30, -0.24, -0.18, -0.12, -0.06, -0.03, 0.03, 0.06, 0.12, 0.18, 0.30],
    "thetaM": [0.15, 0.20, 0.35, 0.55, 0.70, 0.77, 0.90, 1.10, 1.30, 1.60, 2.00, 2.60, 3.20, 3.80],
    "thetaPsi": [-0.45, -0.30, -0.15, -0.07, 0.07, 0.15, 0.30, 0.45],
    "thetaMPsi": [-0.45, -0.30, -0.15, -0.07, 0.07, 0.15, 0.30, 0.45],
    "phi0": [-3.20, -2.80, -2.40, -2.10, -1.85, -1.65, -1.45, -1.20, -1.00, -0.80, -0.60, -0.40, -0.25],
    "deltaS": [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.80, 1.20, 1.60, 2.20, 3.00, 4.50, 6.00, 9.00, 12.00, 16.00],
    "phiPsi": [-1.10, -0.80, -0.60, -0.30, -0.15, -0.07, 0.07, 0.15, 0.30, 0.60, 0.80, 1.10],
}
COMPACT_GRID: Dict[str, List[float]] = {
    "theta0": [-0.72, -0.48, -0.30, -0.24, -0.18, -0.12, -0.06, 0.06, 0.18, 0.30],
    "thetaM": [0.15, 0.35, 0.55, 0.70, 0.77, 1.10, 1.60, 2.60, 3.80],
    "thetaPsi": [-0.45, -0.15, -0.07, 0.07, 0.15, 0.45],
    "thetaMPsi": [-0.45, -0.15, -0.07, 0.07, 0.15, 0.45],
    "phi0": [-3.20, -2.40, -1.85, -1.65, -1.45, -1.20, -0.80, -0.40],
    "deltaS": [0.10, 0.50, 1.20, 2.20, 3.00, 4.50, 6.00, 8.00],
    "phiPsi": [-1.10, -0.60, -0.15, -0.07, 0.07, 0.15, 0.60, 1.10],
}
GRID_PROFILE = os.environ.get("MECH_MINIMALITY_GRID_PROFILE", "publication").strip().lower()
if GRID_PROFILE not in {"publication", "compact"}:
    raise ValueError("MECH_MINIMALITY_GRID_PROFILE must be publication or compact.")
GRID: Dict[str, List[float]] = {
    name: list(values)
    for name, values in (PUBLICATION_GRID if GRID_PROFILE == "publication" else COMPACT_GRID).items()
}
DELTA_S_SATURATION_TOL = float(os.environ.get(
    "MECH_MINIMALITY_DELTA_S_SATURATION_TOL", "0.002"
))
DELTA_S_PLATEAU_MAX_NEXT_PSI = float(os.environ.get(
    "MECH_MINIMALITY_DELTA_S_PLATEAU_MAX_NEXT_PSI", "0.0001"
))
DELTA_S_PLATEAU_MAX_SCORE_DIFF = float(os.environ.get(
    "MECH_MINIMALITY_DELTA_S_PLATEAU_MAX_SCORE_DIFF", "0.00001"
))
DELTA_S_PLATEAU_MAX_OBJECTIVE_DIFF = float(os.environ.get(
    "MECH_MINIMALITY_DELTA_S_PLATEAU_MAX_OBJECTIVE_DIFF", "0.00001"
))

ANCHOR: Dict[str, float] = {
    "theta0": -0.28,
    "thetaM": 0.10,
    "thetaPsi": 0.0,
    "thetaMPsi": 0.0,
    "phi0": -1.80,
    "deltaS": 6.0,
    "phiPsi": 0.0,
    **FIXED_NUISANCE,
}
PILOT_ANCHORS: List[Dict[str, float]] = [
    ANCHOR,
    {"theta0": 0.0, "thetaM": 0.77, "thetaPsi": 0.0, "thetaMPsi": 0.0, "phi0": -1.20, "deltaS": 6.0, "phiPsi": 0.0, **FIXED_NUISANCE},
    {"theta0": -0.25, "thetaM": 0.77, "thetaPsi": -0.15, "thetaMPsi": 0.10, "phi0": -1.65, "deltaS": 12.0, "phiPsi": -0.30, **FIXED_NUISANCE},
]

# Family candidates carry complete parameter vectors; disabled terms are zero.

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
QUANTILE_PROBS = np.asarray([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=float)

# Configured after CLI parsing.
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
TABLE_DIR = OUTPUT_ROOT / "tables"
META_DIR = OUTPUT_ROOT / "metadata"
SEARCH_DIR = TABLE_DIR / "family_searches"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
RUNTIME_PROFILE: List[Dict[str, Any]] = []


def now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def json_safe(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return json_safe(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int)):
        return obj
    return str(obj)


def stable_json_hash(obj: Any) -> str:
    payload = json.dumps(json_safe(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(obj), handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    compression = "gzip" if path.name.endswith(".gz") else None
    df.to_csv(tmp, index=False, compression=compression)
    os.replace(tmp, path)
    return path


def write_table(df: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    parquet = base.with_suffix(".parquet")
    tmp = parquet.with_name(parquet.name + ".tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, parquet)
        return parquet
    except Exception:
        if tmp.exists():
            tmp.unlink()
        csv = base.with_suffix(".csv.gz")
        return write_csv(df, csv)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_grid_contract() -> Dict[str, object]:
    for name, values in GRID.items():
        array = np.asarray(values, dtype=float)
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"Invalid grid for {name}.")
        if np.unique(array).size != array.size or np.any(np.diff(array) <= 0):
            raise ValueError(f"Grid values must be unique and increasing: {name}.")
        if np.any(np.isclose(array, 0.0, atol=1e-14, rtol=0.0)):
            raise ValueError(f"Enabled-term grid contains structural zero: {name}.")

    minimum_support_argument = (
        min(GRID["phi0"])
        + max(GRID["deltaS"])
        - max(abs(value) for value in GRID["phiPsi"])
    )
    residual = float(max(0.0, 1.0 - math.tanh(minimum_support_argument)))
    if residual > DELTA_S_SATURATION_TOL:
        raise ValueError(
            "The upper deltaS grid point does not reach the declared support-channel plateau."
        )
    return {
        "grid_profile": GRID_PROFILE,
        "minimum_support_argument_at_upper_deltaS": float(minimum_support_argument),
        "maximum_support_drive_residual_at_upper_deltaS": residual,
        "deltaS_saturation_tolerance": float(DELTA_S_SATURATION_TOL),
    }


@contextlib.contextmanager
def timed_stage(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        RUNTIME_PROFILE.append({"stage": name, "elapsed_seconds": time.perf_counter() - start})


def digitize_closed_right(vals: np.ndarray, bins: np.ndarray) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    edges = np.asarray(bins, dtype=float)
    if edges.size == 0:
        return np.full(arr.shape, -1, dtype=np.int64)
    adjusted = np.where(arr == edges[-1], np.nextafter(edges[-1], edges[0]), arr)
    return np.digitize(adjusted, edges) - 1


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
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
    if df.empty or "user_id" not in df.columns:
        return np.ones(len(df), dtype=float)
    c = df.groupby("user_id")["user_id"].transform("count").to_numpy(dtype=float)
    return 1.0 / np.maximum(c, 1.0)


def sort_panel(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not {"user_id", "bundle_step_index"}.issubset(df.columns):
        return df.reset_index(drop=True)
    return df.sort_values(["user_id", "bundle_step_index"], kind="mergesort").reset_index(drop=True)


def numeric_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def infer_eta_from_stage1(train: pd.DataFrame) -> float:
    e = numeric_series(train, "response_evidence_mass_pre").to_numpy(dtype=float)
    v = numeric_series(train, "response_evidence_maturity_V_pre").to_numpy(dtype=float)
    ok = np.isfinite(e) & np.isfinite(v) & (e > 0) & (v > 1e-6) & (v < 1.0 - 1e-6)
    if ok.sum() < 100:
        return float(EVIDENCE_MATURITY_SCALE_DEFAULT)
    eta = e[ok] / np.maximum(-np.log1p(-v[ok]), EPS)
    eta = eta[np.isfinite(eta) & (eta > 0)]
    if eta.size == 0:
        return float(EVIDENCE_MATURITY_SCALE_DEFAULT)
    return float(np.median(eta))


def prepare_panel(df: pd.DataFrame, eta: float) -> pd.DataFrame:
    """Build the numeric mechanism panel."""
    d = pd.DataFrame(index=df.index)
    d["user_id"] = numeric_series(df, "user_id").astype("Int64")
    d["bundle_step_index"] = numeric_series(df, "bundle_step_index").astype("Int64")

    d["M"] = numeric_series(df, "M_response_prebalanced_pre")
    d["Psi"] = numeric_series(df, "activity_alignment_order_Psi_pre")
    d["V"] = numeric_series(df, "response_evidence_maturity_V_pre")
    d["E"] = numeric_series(df, "response_evidence_mass_pre")

    # Use maturity only when direct evidence mass is unavailable.
    inv_e = -float(eta) * np.log1p(-np.clip(d["V"].to_numpy(dtype=float), 0.0, 1.0 - 1e-9))
    e_arr = d["E"].to_numpy(dtype=float)
    bad_e = ~np.isfinite(e_arr) | (e_arr < 0)
    d.loc[bad_e, "E"] = inv_e[bad_e]

    d["target_M_next"] = numeric_series(df, "next_M_response_prebalanced")
    d["target_Psi_next"] = numeric_series(df, "next_activity_alignment_order_Psi")
    d["target_V_next"] = numeric_series(df, "next_response_evidence_maturity_V")
    d["target_E_next"] = numeric_series(df, "next_response_evidence_mass")
    target_e_inv = -float(eta) * np.log1p(-np.clip(d["target_V_next"].to_numpy(dtype=float), 0.0, 1.0 - 1e-9))
    te = d["target_E_next"].to_numpy(dtype=float)
    mask_te = ~np.isfinite(te) | (te < 0)
    d.loc[mask_te, "target_E_next"] = target_e_inv[mask_te]

    d["next_gap_days"] = numeric_series(df, "next_gap_days")
    d["has_next"] = numeric_series(df, "has_next_submitted_bundle", default=0).fillna(0).astype(bool)

    bundle_n = numeric_series(df, "bundle_n_questions")
    answered_fraction = numeric_series(df, "answered_fraction_interval")
    total_response_count = numeric_series(df, "total_response_count_diagnostic")
    answered_count = bundle_n * answered_fraction
    answered_count = answered_count.where(np.isfinite(answered_count) & (answered_count > 0), total_response_count)
    answered_count = answered_count.where(np.isfinite(answered_count) & (answered_count > 0), numeric_series(df, "response_active_mass_interval") * bundle_n)
    d["answered_count_proxy"] = answered_count.clip(lower=0).fillna(0.0)

    # Keep signed and neutral activity masses separate.
    response_aligned = numeric_series(df, "response_aligned_mass_interval").clip(lower=0).fillna(0.0)
    response_off = numeric_series(df, "response_off_target_mass_interval").clip(lower=0).fillna(0.0)
    response_neutral = numeric_series(df, "response_neutral_mass_interval").clip(lower=0).fillna(0.0)
    support_aligned = numeric_series(df, "support_aligned_mass_interval").clip(lower=0).fillna(0.0)
    support_off = numeric_series(df, "support_off_target_mass_interval").clip(lower=0).fillna(0.0)
    support_neutral = numeric_series(df, "support_neutral_mass_interval").clip(lower=0).fillna(0.0)

    response_alignable = response_aligned + response_off
    support_alignable = support_aligned + support_off

    # Treat unexplained channel mass as neutral.
    response_active = numeric_series(df, "response_active_mass_interval").clip(lower=0).fillna(0.0)
    support_active = numeric_series(df, "support_active_total_interval").clip(lower=0).fillna(0.0)
    response_remainder = (response_active - response_alignable - response_neutral).clip(lower=0).fillna(0.0)
    support_remainder = (support_active - support_alignable - support_neutral).clip(lower=0).fillna(0.0)
    response_neutral = response_neutral + response_remainder
    support_neutral = support_neutral + support_remainder

    d["response_alignable_interval"] = response_alignable.clip(lower=0).fillna(0.0)
    d["support_alignable_interval"] = support_alignable.clip(lower=0).fillna(0.0)
    d["response_neutral_interval"] = response_neutral.clip(lower=0).fillna(0.0)
    d["support_neutral_interval"] = support_neutral.clip(lower=0).fillna(0.0)
    d["active_alignable_interval"] = d["response_alignable_interval"] + d["support_alignable_interval"]
    d["active_neutral_interval"] = d["response_neutral_interval"] + d["support_neutral_interval"]
    d["idle_mass_interval"] = numeric_series(df, "idle_mass_interval").clip(lower=0).fillna(0.0)

    active_pre = numeric_series(df, "activity_active_mass_pre").clip(lower=0)
    idle_pre = numeric_series(df, "activity_idle_mass_pre").clip(lower=0)
    B = active_pre + idle_pre
    aligned_pre = numeric_series(df, "activity_aligned_mass_pre")
    if "activity_off_target_mass_pre" in df.columns:
        off_pre = numeric_series(df, "activity_off_target_mass_pre")
    else:
        off_pre = numeric_series(df, "activity_non_aligned_mass_pre")
    G = aligned_pre - off_pre
    fallback_B = np.maximum(np.abs(G.to_numpy(dtype=float)) / np.maximum(np.abs(d["Psi"].to_numpy(dtype=float)), 1e-6), 1.0)
    B_arr = B.to_numpy(dtype=float)
    B_arr = np.where(np.isfinite(B_arr) & (B_arr > 0), B_arr, fallback_B)
    G_arr = G.to_numpy(dtype=float)
    G_arr = np.where(np.isfinite(G_arr), G_arr, d["Psi"].to_numpy(dtype=float) * B_arr)
    d["B"] = np.maximum(B_arr, EPS)
    d["G"] = np.clip(G_arr, -d["B"].to_numpy(dtype=float), d["B"].to_numpy(dtype=float))

    # Use the Stage-1 next activity denominator when available.
    next_active = numeric_series(df, "next_activity_active_mass").clip(lower=0)
    next_idle = numeric_series(df, "next_activity_idle_mass").clip(lower=0)
    target_B_direct = (next_active + next_idle).to_numpy(dtype=float)
    rho_A_for_target = np.exp(-np.maximum(d["next_gap_days"].to_numpy(dtype=float), 0.0) / max(float(TAU_ACTIVITY_DAYS), EPS))
    target_B_fallback = rho_A_for_target * np.maximum(
        d["B"].to_numpy(dtype=float)
        + d["active_alignable_interval"].to_numpy(dtype=float)
        + d["active_neutral_interval"].to_numpy(dtype=float)
        + d["idle_mass_interval"].to_numpy(dtype=float),
        EPS,
    )
    use_direct_B = np.isfinite(target_B_direct) & (target_B_direct > 0)
    d["target_B_next"] = np.where(use_direct_B, target_B_direct, target_B_fallback)

    # Optional empirical phase deltas, used only when Stage-1 core columns exist.
    M_resp = numeric_series(df, "M_response_prebalanced_resp")
    V_resp = numeric_series(df, "response_evidence_maturity_V_resp")
    Psi_support = numeric_series(df, "activity_alignment_order_Psi_support")
    Psi_post = numeric_series(df, "activity_alignment_order_Psi_post")
    d["emp_delta_M_response"] = M_resp - d["M"]
    d["emp_delta_V_response"] = V_resp - d["V"]
    d["emp_delta_Psi_active"] = Psi_support - d["Psi"]
    d["emp_delta_Psi_idle"] = Psi_post - Psi_support
    d["phase_columns_available"] = (
        np.isfinite(d["emp_delta_M_response"])
        & np.isfinite(d["emp_delta_Psi_active"])
        & np.isfinite(d["emp_delta_Psi_idle"])
    )

    valid = (
        d["user_id"].notna()
        & d["bundle_step_index"].notna()
        & d["has_next"]
        & np.isfinite(d["M"])
        & np.isfinite(d["Psi"])
        & np.isfinite(d["V"])
        & np.isfinite(d["E"])
        & np.isfinite(d["target_M_next"])
        & np.isfinite(d["target_Psi_next"])
        & np.isfinite(d["target_V_next"])
        & np.isfinite(d["next_gap_days"])
    )
    d = d.loc[valid].copy()
    d["user_id"] = d["user_id"].astype(int)
    d["bundle_step_index"] = d["bundle_step_index"].astype(int)
    d["M"] = d["M"].clip(-1.0, 1.0)
    d["Psi"] = d["Psi"].clip(-1.0, 1.0)
    d["V"] = d["V"].clip(0.0, 1.0)
    d["E"] = d["E"].clip(lower=0.0)
    d["target_M_next"] = d["target_M_next"].clip(-1.0, 1.0)
    d["target_Psi_next"] = d["target_Psi_next"].clip(-1.0, 1.0)
    d["target_V_next"] = d["target_V_next"].clip(0.0, 1.0)
    d["target_E_next"] = d["target_E_next"].clip(lower=0.0)
    d["target_B_next"] = d["target_B_next"].clip(lower=EPS)
    d["next_gap_days"] = d["next_gap_days"].clip(lower=0.0)
    d = d.sort_values(["user_id", "bundle_step_index"], kind="mergesort").reset_index(drop=True)
    return d


PHASE1_PANEL_COLUMNS = [
    "user_id", "bundle_step_index",
    "M_response_prebalanced_pre", "activity_alignment_order_Psi_pre", "response_evidence_maturity_V_pre",
    "response_evidence_mass_pre", "next_M_response_prebalanced", "next_activity_alignment_order_Psi",
    "next_response_evidence_maturity_V", "next_response_evidence_mass", "next_gap_days", "has_next_submitted_bundle",
    "bundle_n_questions", "answered_fraction_interval", "total_response_count_diagnostic", "response_active_mass_interval",
    "support_active_total_interval", "idle_mass_interval", "activity_active_mass_pre", "activity_idle_mass_pre",
    "activity_aligned_mass_pre", "activity_off_target_mass_pre", "activity_non_aligned_mass_pre",
    "response_aligned_mass_interval", "response_off_target_mass_interval",
    "response_neutral_mass_interval", "support_aligned_mass_interval", "support_off_target_mass_interval",
    "support_neutral_mass_interval",
    # Optional next-pre activity denominator.
    "next_activity_active_mass", "next_activity_idle_mass",
    # Optional phase-decomposition columns.
    "M_response_prebalanced_resp", "response_evidence_maturity_V_resp",
    "activity_alignment_order_Psi_support", "activity_alignment_order_Psi_post",
]


def stage1_dynamics_root(stage1_root: Path) -> Path:
    """Return the current Stage-1 dynamics directory."""
    dynamics = Path(stage1_root).resolve() / "dynamics"
    if not dynamics.is_dir():
        raise FileNotFoundError(
            f"Stage-1 dynamics directory not found: {dynamics}. "
            "Run the current Stage-1 script or pass its stage1 output root."
        )
    return dynamics


def audit_stage1_kmeans_contract(
    stage1_root: Path,
    allow_missing: bool,
) -> Dict[str, object]:
    """Check the fixed K=6 Stage-1 mesostate metadata."""
    metadata_path = (
        stage1_dynamics_root(stage1_root)
        / "fixed_k6_mesostates"
        / "fixed_k6_model_metadata.json"
    )
    if not metadata_path.exists():
        if allow_missing:
            return {
                "status": "not_found",
                "metadata_path": str(metadata_path),
                "used_in_minimality_selection": False,
            }
        raise FileNotFoundError(
            f"Stage-1 fixed-K metadata not found: {metadata_path}. "
            "Use --allow-missing-kmeans-contract only for panel-only archival inputs."
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected_features = [
        "M_response_prebalanced_pre",
        "activity_alignment_order_Psi_pre",
    ]
    scaler_mean = np.asarray(metadata.get("scaler_mean", []), dtype=float)
    scaler_scale = np.asarray(metadata.get("scaler_scale", []), dtype=float)
    raw_mapping = metadata.get("raw_to_ordered_label", {})
    mapping_keys = sorted(int(key) for key in raw_mapping) if isinstance(raw_mapping, dict) else []
    mapping_values = sorted(int(value) for value in raw_mapping.values()) if isinstance(raw_mapping, dict) else []
    expected_labels = list(range(EXPECTED_STAGE1_MACROSTATE_K))
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
        "scaler": bool(
            scaler_mean.shape == (2,)
            and scaler_scale.shape == (2,)
            and np.isfinite(scaler_mean).all()
            and np.isfinite(scaler_scale).all()
            and np.all(scaler_scale > 0)
        ),
        "label_mapping": mapping_keys == expected_labels and mapping_values == expected_labels,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "Stage-1 fixed-K metadata does not match the public analysis contract: "
            + ", ".join(failed)
        )

    centers_base = metadata_path.parent / "fixed_k6_centers"
    centers_path = next(
        (centers_base.with_suffix(ext) for ext in (".parquet", ".csv.gz", ".csv")
         if centers_base.with_suffix(ext).exists()),
        None,
    )
    if centers_path is None:
        if allow_missing:
            return {
                "status": "metadata_verified_centers_not_found",
                "metadata_path": str(metadata_path.resolve()),
                "metadata_sha256": file_sha256(metadata_path),
                "centers_path": None,
                "used_in_minimality_selection": False,
                "checks": checks,
                "metadata": metadata,
            }
        raise FileNotFoundError(f"Stage-1 fixed-K centers not found: {centers_base}")
    if centers_path.suffix == ".parquet":
        centers = pd.read_parquet(centers_path)
    else:
        centers = pd.read_csv(centers_path, low_memory=False)
    required_center_columns = {"macrostate", "center_M", "center_Psi"}
    if not required_center_columns.issubset(centers.columns):
        missing = sorted(required_center_columns.difference(centers.columns))
        raise RuntimeError(f"Stage-1 fixed-K centers are missing columns: {missing}")
    center_ids = pd.to_numeric(centers["macrostate"], errors="coerce").to_numpy(dtype=float)
    center_values = centers[["center_M", "center_Psi"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    expected_ids = np.arange(EXPECTED_STAGE1_MACROSTATE_K, dtype=float)
    lexicographic_order = np.lexsort((center_values[:, 1], center_values[:, 0]))
    if (
        len(centers) != EXPECTED_STAGE1_MACROSTATE_K
        or not np.array_equal(center_ids, expected_ids)
        or not np.isfinite(center_values).all()
        or not np.array_equal(lexicographic_order, expected_ids.astype(int))
    ):
        raise RuntimeError("Stage-1 fixed-K centers do not define the expected six ordered states.")
    return {
        "status": "verified",
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": file_sha256(metadata_path),
        "centers_path": str(centers_path.resolve()),
        "centers_sha256": file_sha256(centers_path),
        "used_in_minimality_selection": False,
        "checks": checks,
        "metadata": metadata,
    }


def read_core_panel(dyn_root: Path, split: str) -> pd.DataFrame:
    """Read only mechanism columns from a Stage-1 split panel."""
    base = dyn_root / f"student_dynamics_panel_core_{split}"
    for path in (
        base.with_suffix(".parquet"),
        base.with_suffix(".csv.gz"),
        base.with_suffix(".csv"),
    ):
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
                available = set(pq.read_schema(path).names)
            except Exception:
                available = set(pd.read_parquet(path).columns)
            columns = [name for name in PHASE1_PANEL_COLUMNS if name in available]
            return pd.read_parquet(path, columns=columns)
        available = set(pd.read_csv(path, nrows=0).columns)
        columns = [name for name in PHASE1_PANEL_COLUMNS if name in available]
        return pd.read_csv(path, usecols=columns, low_memory=False)
    raise FileNotFoundError(f"Could not find Stage-1 core panel for {split}: {base}")


def load_phase1_panels(stage1_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, float, Dict[str, object]]:
    dyn_root = stage1_dynamics_root(stage1_root)
    train_raw = read_core_panel(dyn_root, "A_train")
    val_raw = read_core_panel(dyn_root, "A_val")
    eta = infer_eta_from_stage1(train_raw)
    train = prepare_panel(train_raw, eta)
    val = prepare_panel(val_raw, eta)
    manifest = {
        "stage1_root": str(stage1_root),
        "stage1_dynamics_root": str(dyn_root),
        "train_rows_raw": int(len(train_raw)),
        "val_rows_raw": int(len(val_raw)),
        "train_rows_phase1_valid": int(len(train)),
        "val_rows_phase1_valid": int(len(val)),
        "eta_inferred_from_A_train_stage1_columns": float(eta),
        "column_projected_read_attempted": True,
        "B_confirm_policy": "not read or used in phase 1",
    }
    return train, val, eta, manifest


def development_panel_contract(train: pd.DataFrame, val: pd.DataFrame) -> Dict[str, object]:
    """Validate split integrity and return fingerprints."""
    required = {"user_id", "bundle_step_index", "M", "Psi", "target_M_next", "target_Psi_next"}
    for label, panel in (("A_train", train), ("A_val", val)):
        missing = sorted(required.difference(panel.columns))
        if missing:
            raise KeyError(f"{label} is missing required prepared-panel columns: {missing}")
        if panel.empty:
            raise RuntimeError(f"{label} is empty after validity filtering.")
        uid = panel["user_id"].to_numpy(dtype=np.int64, copy=False)
        step = panel["bundle_step_index"].to_numpy(dtype=np.int64, copy=False)
        if uid.size > 1 and np.any((uid[1:] == uid[:-1]) & (step[1:] == step[:-1])):
            raise RuntimeError(f"{label} contains duplicate (user_id, bundle_step_index) rows.")

    train_users = np.asarray(sorted(train["user_id"].unique()), dtype=np.int64)
    val_users = np.asarray(sorted(val["user_id"].unique()), dtype=np.int64)
    overlap = np.intersect1d(train_users, val_users, assume_unique=True)
    if overlap.size:
        raise RuntimeError(
            f"A_train and A_val are not user-disjoint; found {overlap.size} overlapping users."
        )

    def summarize(label: str, panel: pd.DataFrame, users: np.ndarray) -> Dict[str, object]:
        uid = panel["user_id"].to_numpy(dtype=np.int64, copy=False)
        step = panel["bundle_step_index"].to_numpy(dtype=np.int64, copy=False)
        sample_n = min(len(panel), 100_000)
        identifiers = np.column_stack([uid[:sample_n], step[:sample_n]])
        return {
            "label": label,
            "rows": int(len(panel)),
            "users": int(users.size),
            "user_ids_sha256": hashlib.sha256(users.tobytes()).hexdigest(),
            "head_identifier_rows_hashed": int(sample_n),
            "head_identifier_sha256": hashlib.sha256(identifiers.tobytes()).hexdigest(),
            "bundle_step_min": int(step.min()) if step.size else None,
            "bundle_step_max": int(step.max()) if step.size else None,
        }

    return {
        "user_disjoint": True,
        "A_train": summarize("A_train", train, train_users),
        "A_val": summarize("A_val", val, val_users),
    }


@dataclass
class Calibration:
    eta: float
    tau_response_days: float
    tau_activity_days: float
    residual_mass_per_answer: float
    response_signed_gain: float = 1.0
    alignment_signed_gain: float = 1.0


def _robust_fraction_gain(values: np.ndarray, default: float = 1.0) -> float:
    vals = np.asarray(values, dtype=float)
    vals = np.abs(vals[np.isfinite(vals)])
    vals = vals[(vals > 1e-6) & (vals <= 5.0)]
    if vals.size < 100:
        return float(default)
    q = float(np.clip(CONFIG_SIGNED_GAIN_QUANTILE, 0.10, 0.95))
    return float(np.clip(np.nanquantile(vals, q), 0.10, 1.0))


def calibrate_from_A_train(train: pd.DataFrame, eta: float, tau_response_days: float, tau_activity_days: float) -> Calibration:
    gap = train["next_gap_days"].to_numpy(dtype=float)
    rho = np.exp(-np.maximum(gap, 0.0) / max(float(tau_response_days), EPS))
    E = train["E"].to_numpy(dtype=float)
    En = train["target_E_next"].to_numpy(dtype=float)
    answered = train["answered_count_proxy"].to_numpy(dtype=float)
    R_eff = En / np.maximum(rho, EPS) - E
    ok = np.isfinite(R_eff) & np.isfinite(answered) & (R_eff > 0) & (answered > 0)
    if ok.sum() >= 100:
        unit_vals = R_eff[ok] / np.maximum(answered[ok], EPS)
        unit_vals = unit_vals[np.isfinite(unit_vals) & (unit_vals > 0)]
        r_unit = float(np.median(unit_vals)) if unit_vals.size else 0.45
    else:
        r_unit = 0.45
    r_unit = float(np.clip(r_unit, 0.02, 2.0))

    # Calibrate signed gains from A_train phase deltas.
    M = train["M"].to_numpy(dtype=float)
    Psi = train["Psi"].to_numpy(dtype=float)
    B = train["B"].to_numpy(dtype=float)
    G = train["G"].to_numpy(dtype=float)
    emp_dM_resp = train.get("emp_delta_M_response", pd.Series(np.nan, index=train.index)).to_numpy(dtype=float)
    emp_dV_resp = train.get("emp_delta_V_response", pd.Series(np.nan, index=train.index)).to_numpy(dtype=float)
    V_resp = np.clip(train["V"].to_numpy(dtype=float) + emp_dV_resp, 0.0, 1.0 - 1e-9)
    E_resp = -float(eta) * np.log1p(-V_resp)
    M_resp = np.clip(M + emp_dM_resp, -1.0, 1.0)
    S = M * E
    S_resp = M_resp * E_resp
    R_resp = E_resp - E
    z_response = (S_resp - S) / np.maximum(R_resp, EPS)
    response_signed_gain = _robust_fraction_gain(z_response[np.isfinite(R_resp) & (R_resp > 1e-6)], default=1.0)

    emp_dPsi_active = train.get("emp_delta_Psi_active", pd.Series(np.nan, index=train.index)).to_numpy(dtype=float)
    Psi_active = np.clip(Psi + emp_dPsi_active, -1.0, 1.0)
    Amap = train["active_alignable_interval"].to_numpy(dtype=float)
    A0 = train["active_neutral_interval"].to_numpy(dtype=float)
    B_active = np.maximum(B + Amap + A0, EPS)
    G_active = Psi_active * B_active
    z_alignment = (G_active - G) / np.maximum(Amap, EPS)
    alignment_signed_gain = _robust_fraction_gain(z_alignment[np.isfinite(Amap) & (Amap > 1e-6)], default=1.0)

    return Calibration(
        eta=float(eta),
        tau_response_days=float(tau_response_days),
        tau_activity_days=float(tau_activity_days),
        residual_mass_per_answer=r_unit,
        response_signed_gain=response_signed_gain,
        alignment_signed_gain=alignment_signed_gain,
    )


def params_to_vector(params: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(params[name]) for name in PARAM_NAMES], dtype=np.float64)


def _simulate_core_python(
    M_obs: np.ndarray,
    Psi_obs: np.ndarray,
    E_obs: np.ndarray,
    B_obs: np.ndarray,
    G_obs: np.ndarray,
    gap: np.ndarray,
    answered: np.ndarray,
    response_alignable: np.ndarray,
    support_alignable: np.ndarray,
    response_neutral: np.ndarray,
    support_neutral: np.ndarray,
    idle: np.ndarray,
    params_v: np.ndarray,
    eta: float,
    tau_response_days: float,
    tau_activity_days: float,
    residual_mass_per_answer: float,
    response_signed_gain: float,
    alignment_signed_gain: float,
) -> Tuple[np.ndarray, ...]:
    """Apply the state-reset one-step mechanism to independent rows."""
    n = len(M_obs)
    pred_Mn = np.zeros(n, dtype=np.float64)
    pred_Psin = np.zeros(n, dtype=np.float64)
    pred_Vn = np.zeros(n, dtype=np.float64)
    pred_Bn = np.zeros(n, dtype=np.float64)
    pred_delta_M_resp = np.zeros(n, dtype=np.float64)
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

    for i in prange(n):
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
        H = max(float(alignment_signed_gain), 0.0) * (
            Aresp * response_align_drive + Asupp * support_align_drive
        )
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
        pred_Bn[i] = B_next
        pred_delta_M_resp[i] = M_resp - M
        pred_delta_Psi_active[i] = Psi_active - Psi
        pred_delta_Psi_idle[i] = Psi_idle - Psi_active
        scaled_Amap[i] = Amap
        scaled_A0[i] = A0
        scaled_I[i] = I

    return (
        pred_Mn, pred_Psin, pred_Vn, pred_Bn,
        pred_delta_M_resp, pred_delta_Psi_active, pred_delta_Psi_idle,
        scaled_Amap, scaled_A0, scaled_I,
    )



if NUMBA_AVAILABLE:
    _simulate_core_numba = njit(cache=True, parallel=True)(_simulate_core_python)
else:
    _simulate_core_numba = None



def _simulate_core_dispatch(
    M_obs: np.ndarray,
    Psi_obs: np.ndarray,
    E_obs: np.ndarray,
    B_obs: np.ndarray,
    G_obs: np.ndarray,
    gap: np.ndarray,
    answered: np.ndarray,
    response_alignable: np.ndarray,
    support_alignable: np.ndarray,
    response_neutral: np.ndarray,
    support_neutral: np.ndarray,
    idle: np.ndarray,
    params_v: np.ndarray,
    calib: Calibration,
) -> Tuple[np.ndarray, ...]:
    fn = _simulate_core_numba if CONFIG_USE_NUMBA and _simulate_core_numba is not None else _simulate_core_python
    return fn(
        np.asarray(M_obs, dtype=np.float64),
        np.asarray(Psi_obs, dtype=np.float64),
        np.asarray(E_obs, dtype=np.float64),
        np.asarray(B_obs, dtype=np.float64),
        np.asarray(G_obs, dtype=np.float64),
        np.asarray(gap, dtype=np.float64),
        np.asarray(answered, dtype=np.float64),
        np.asarray(response_alignable, dtype=np.float64),
        np.asarray(support_alignable, dtype=np.float64),
        np.asarray(response_neutral, dtype=np.float64),
        np.asarray(support_neutral, dtype=np.float64),
        np.asarray(idle, dtype=np.float64),
        np.asarray(params_v, dtype=np.float64),
        float(calib.eta),
        float(calib.tau_response_days),
        float(calib.tau_activity_days),
        float(calib.residual_mass_per_answer),
        float(calib.response_signed_gain),
        float(calib.alignment_signed_gain),
    )


def objective_component_values(metrics: Dict[str, float]) -> Dict[str, float]:
    """Return bounded objective components; missing values receive loss one."""
    out: Dict[str, float] = {}
    for key in list(OBJECTIVE_WEIGHTS.keys()) + list(OBJECTIVE_SANITY_LIMITS.keys()):
        val = metrics.get(key, np.nan)
        if not np.isfinite(val):
            val = 1.0
        out[key] = float(np.clip(val, 0.0, 1.0))
    return out


def primary_objective_score(metrics: Dict[str, float]) -> float:
    comps = objective_component_values(metrics)
    total_w = float(sum(max(float(w), 0.0) for w in OBJECTIVE_WEIGHTS.values()))
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
        excess_details[f"objective_{key}_excess"] = float(excess)
        vals.append(excess * excess)
    penalty = float(math.sqrt(sum(vals))) if vals else 0.0
    return penalty, excess_details


def objective_diagnostics(metrics: Dict[str, float]) -> Dict[str, object]:
    comps = objective_component_values(metrics)
    if comps:
        worst_name = max(comps, key=lambda k: comps[k])
        worst_value = float(comps[worst_name])
        saturated_count = int(sum(v >= 1.0 - 1e-12 for v in comps.values()))
    else:
        worst_name = "none"
        worst_value = 1.0
        saturated_count = 0
    primary_score = primary_objective_score(metrics)
    sanity_penalty, sanity_excess = sanity_constraint_penalty(metrics)
    id_penalty = float(metrics.get("identity_regularization", 0.0))
    if not np.isfinite(id_penalty):
        id_penalty = 0.0
    diag: Dict[str, object] = {
        "objective_selection_rule": "primary_MR_PsiA_weighted_score_with_soft_phase_coverage_sanity_constraints",
        "objective_primary_score": float(primary_score),
        "objective_sanity_penalty": float(sanity_penalty),
        "objective_worst_component": worst_name,
        "objective_worst_component_value": worst_value,
        "objective_saturated_component_count": saturated_count,
        "objective_identity_penalty": id_penalty,
        "objective_sanity_penalty_weight": float(CONFIG_SANITY_PENALTY_WEIGHT),
        "objective_identity_reg_weight": float(CONFIG_IDENTITY_REG_WEIGHT),
    }
    diag.update(sanity_excess)
    return diag


def objective_from_metrics(metrics: Dict[str, float]) -> float:
    diag = objective_diagnostics(metrics)
    primary_score = float(diag["objective_primary_score"])
    sanity_penalty = float(diag["objective_sanity_penalty"])
    id_penalty = float(diag["objective_identity_penalty"])
    return float(
        primary_score
        + CONFIG_SANITY_PENALTY_WEIGHT * sanity_penalty
        + CONFIG_IDENTITY_REG_WEIGHT * id_penalty
    )


@dataclass
class SimArrays:
    pred_next_M: np.ndarray
    pred_next_Psi: np.ndarray
    pred_next_V: np.ndarray
    pred_next_B: np.ndarray
    pred_delta_M_response: np.ndarray
    pred_delta_Psi_active: np.ndarray
    pred_delta_Psi_idle: np.ndarray
    scaled_active_alignable: np.ndarray
    scaled_active_neutral: np.ndarray
    scaled_idle: np.ndarray


@dataclass(frozen=True)
class QuantileReference:
    quantiles: np.ndarray
    scale: float


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
    target_B_next: np.ndarray
    emp_delta_M_response: np.ndarray
    emp_delta_Psi_active: np.ndarray
    emp_delta_Psi_idle: np.ndarray
    phase_available: np.ndarray
    scale_M: float
    scale_Psi: float
    H_obs: np.ndarray
    field_obs: Dict[str, np.ndarray]
    current_cell: np.ndarray
    current_cell_valid: np.ndarray
    loss_sample_idx: np.ndarray
    phase_quantile_references: Dict[str, QuantileReference]
    coverage_quantile_references: Dict[str, QuantileReference]


def occupancy_grid_weighted(x: np.ndarray, y: np.ndarray, weights: np.ndarray, xbins: np.ndarray = GRID_BINS_SIGNED, ybins: np.ndarray = GRID_BINS_SIGNED) -> np.ndarray:
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


def field_stats_from_arrays_weighted(
    x: np.ndarray,
    y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    weights: np.ndarray,
    cells: Optional[np.ndarray] = None,
    state_valid: Optional[np.ndarray] = None,
    fixed_support: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Estimate the mean drift on the fixed M-Psi grid."""
    nx = len(GRID_BINS_SIGNED) - 1
    ny = nx
    if cells is None or state_valid is None:
        cells, state_valid = _cells(x, y)
    valid = np.asarray(state_valid, dtype=bool) & np.isfinite(dx) & np.isfinite(dy)
    flat = np.asarray(cells, dtype=np.int32)[valid]
    if not np.any(valid):
        zeros = np.zeros((nx, ny), dtype=float)
        return {
            "count": zeros.copy(),
            "weight": zeros.copy(),
            "u": zeros.copy(),
            "v": zeros.copy(),
            "mask": zeros.astype(bool),
        }
    w = np.asarray(weights, dtype=float)[valid]
    use_fixed_support = bool(
        fixed_support is not None
        and np.array_equal(valid, np.asarray(state_valid, dtype=bool))
    )
    if use_fixed_support:
        count = np.asarray(fixed_support["count"], dtype=float)
        weight = np.asarray(fixed_support["weight"], dtype=float)
        mask = np.asarray(fixed_support["mask"], dtype=bool)
    else:
        count = np.bincount(flat, minlength=nx * ny).reshape(nx, ny).astype(float)
        weight = np.bincount(flat, weights=w, minlength=nx * ny).reshape(nx, ny).astype(float)
        mask = count >= MIN_DRIFT_BIN_COUNT
    sx = np.bincount(flat, weights=w * dx[valid], minlength=nx * ny).reshape(nx, ny).astype(float)
    sy = np.bincount(flat, weights=w * dy[valid], minlength=nx * ny).reshape(nx, ny).astype(float)
    return {
        "count": count,
        "weight": weight,
        "u": sx / np.maximum(weight, EPS),
        "v": sy / np.maximum(weight, EPS),
        "mask": mask,
    }


def make_metric_cache(panel: pd.DataFrame, label: str) -> MetricCache:
    d = sort_panel(panel)
    uid = d["user_id"].to_numpy(dtype=np.int64)
    steps = d["bundle_step_index"].to_numpy(dtype=np.int64)
    weights = user_balanced_weights(d)
    M = d["M"].to_numpy(dtype=float)
    Psi = d["Psi"].to_numpy(dtype=float)
    V = d["V"].to_numpy(dtype=float)
    E = d["E"].to_numpy(dtype=float)
    B = d["B"].to_numpy(dtype=float)
    G = d["G"].to_numpy(dtype=float)
    gap = d["next_gap_days"].to_numpy(dtype=float)
    answered = d["answered_count_proxy"].to_numpy(dtype=float)
    response_alignable = d["response_alignable_interval"].to_numpy(dtype=float)
    support_alignable = d["support_alignable_interval"].to_numpy(dtype=float)
    response_neutral = d["response_neutral_interval"].to_numpy(dtype=float)
    support_neutral = d["support_neutral_interval"].to_numpy(dtype=float)
    active_alignable = response_alignable + support_alignable
    active_neutral = response_neutral + support_neutral
    idle = d["idle_mass_interval"].to_numpy(dtype=float)
    target_M = d["target_M_next"].to_numpy(dtype=float)
    target_Psi = d["target_Psi_next"].to_numpy(dtype=float)
    target_V = d["target_V_next"].to_numpy(dtype=float)
    target_B = d["target_B_next"].to_numpy(dtype=float)
    emp_delta_M_response = d["emp_delta_M_response"].to_numpy(dtype=float) if "emp_delta_M_response" in d.columns else np.full(len(d), np.nan)
    emp_delta_Psi_active = d["emp_delta_Psi_active"].to_numpy(dtype=float) if "emp_delta_Psi_active" in d.columns else np.full(len(d), np.nan)
    emp_delta_Psi_idle = d["emp_delta_Psi_idle"].to_numpy(dtype=float) if "emp_delta_Psi_idle" in d.columns else np.full(len(d), np.nan)
    phase_available = d["phase_columns_available"].to_numpy(dtype=bool) if "phase_columns_available" in d.columns else np.zeros(len(d), dtype=bool)

    # Estimate occupancy and drift on the primary M-Psi plane.
    H_obs = occupancy_grid_weighted(target_M, target_Psi, weights)
    current_cell, current_cell_valid = _cells(M, Psi)
    field_obs = field_stats_from_arrays_weighted(
        M,
        Psi,
        target_M - M,
        target_Psi - Psi,
        weights,
        cells=current_cell,
        state_valid=current_cell_valid,
    )
    scale_M = max(float(np.nanstd(target_M)), 0.05)
    scale_Psi = max(float(np.nanstd(target_Psi)), 0.05)

    if len(d) > CONFIG_DISTRIBUTION_LOSS_MAX_ROWS > 0:
        rng = np.random.default_rng(CONFIG_RANDOM_STATE + 4301)
        loss_sample_idx = np.sort(rng.choice(np.arange(len(d)), size=CONFIG_DISTRIBUTION_LOSS_MAX_ROWS, replace=False)).astype(np.int64)
    else:
        loss_sample_idx = np.arange(len(d), dtype=np.int64)

    phase_rows = loss_sample_idx[phase_available[loss_sample_idx]]
    phase_quantile_references = {
        "phase_M_response_qdist": make_quantile_reference(
            emp_delta_M_response[phase_rows]
        ),
        "phase_Psi_active_qdist": make_quantile_reference(
            emp_delta_Psi_active[phase_rows]
        ),
        "phase_Psi_idle_qdist": make_quantile_reference(
            emp_delta_Psi_idle[phase_rows]
        ),
    }
    coverage_quantile_references = {
        "coverage_B_next_qdist": make_quantile_reference(target_B[loss_sample_idx]),
        "coverage_Amap_qdist": make_quantile_reference(active_alignable[loss_sample_idx]),
        "coverage_A0_qdist": make_quantile_reference(active_neutral[loss_sample_idx]),
        "coverage_idle_qdist": make_quantile_reference(idle[loss_sample_idx]),
    }

    return MetricCache(
        label=label,
        n_rows=int(len(d)),
        n_users=int(d["user_id"].nunique()),
        uid=uid,
        steps=steps,
        weights=weights,
        M=M,
        Psi=Psi,
        V=V,
        E=E,
        B=B,
        G=G,
        gap=gap,
        answered=answered,
        response_alignable=response_alignable,
        support_alignable=support_alignable,
        response_neutral=response_neutral,
        support_neutral=support_neutral,
        active_alignable=active_alignable,
        active_neutral=active_neutral,
        idle=idle,
        target_M_next=target_M,
        target_Psi_next=target_Psi,
        target_V_next=target_V,
        target_B_next=target_B,
        emp_delta_M_response=emp_delta_M_response,
        emp_delta_Psi_active=emp_delta_Psi_active,
        emp_delta_Psi_idle=emp_delta_Psi_idle,
        phase_available=phase_available,
        scale_M=scale_M,
        scale_Psi=scale_Psi,
        H_obs=H_obs,
        field_obs=field_obs,
        current_cell=current_cell,
        current_cell_valid=current_cell_valid,
        loss_sample_idx=loss_sample_idx,
        phase_quantile_references=phase_quantile_references,
        coverage_quantile_references=coverage_quantile_references,
    )


def simulate_arrays(cache: MetricCache, params: Dict[str, float], calib: Calibration) -> SimArrays:
    params_v = params_to_vector(params)
    arrays = _simulate_core_dispatch(
        cache.M,
        cache.Psi,
        cache.E,
        cache.B,
        cache.G,
        cache.gap,
        cache.answered,
        cache.response_alignable,
        cache.support_alignable,
        cache.response_neutral,
        cache.support_neutral,
        cache.idle,
        params_v,
        calib,
    )
    (
        pred_Mn, pred_Psin, pred_Vn, pred_Bn,
        pred_delta_M_resp, pred_delta_Psi_active, pred_delta_Psi_idle,
        scaled_Amap, scaled_A0, scaled_I,
    ) = arrays
    return SimArrays(
        pred_next_M=pred_Mn,
        pred_next_Psi=pred_Psin,
        pred_next_V=pred_Vn,
        pred_next_B=pred_Bn,
        pred_delta_M_response=pred_delta_M_resp,
        pred_delta_Psi_active=pred_delta_Psi_active,
        pred_delta_Psi_idle=pred_delta_Psi_idle,
        scaled_active_alignable=scaled_Amap,
        scaled_active_neutral=scaled_A0,
        scaled_idle=scaled_I,
    )


def quantile_vector(x: np.ndarray, probs: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=float)
    xx = xx[np.isfinite(xx)]
    if xx.size == 0:
        return np.full(len(probs), np.nan, dtype=float)
    return np.quantile(xx, probs)


def make_quantile_reference(x: np.ndarray) -> QuantileReference:
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    quantiles = quantile_vector(values, QUANTILE_PROBS)
    scale = (
        float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))
        if values.size else np.nan
    )
    return QuantileReference(quantiles=quantiles, scale=max(scale, 0.05))


def quantile_distance_from_reference(reference: QuantileReference, values: np.ndarray) -> float:
    observed = np.asarray(reference.quantiles, dtype=float)
    predicted = quantile_vector(values, QUANTILE_PROBS)
    ok = np.isfinite(observed) & np.isfinite(predicted)
    if ok.sum() < 3:
        return np.nan
    return float(min(
        np.linalg.norm(observed[ok] - predicted[ok])
        / (math.sqrt(ok.sum()) * reference.scale + EPS),
        1.0,
    ))


def drift_magnitude_loss(field_obs: Dict[str, np.ndarray], field_sim: Dict[str, np.ndarray], mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return 1.0
    speed_obs = np.sqrt(field_obs["u"] * field_obs["u"] + field_obs["v"] * field_obs["v"])
    speed_sim = np.sqrt(field_sim["u"] * field_sim["u"] + field_sim["v"] * field_sim["v"])
    ok = mask & np.isfinite(speed_obs) & np.isfinite(speed_sim)
    if ok.sum() < 3:
        return 1.0
    w = np.asarray(field_obs.get("weight", np.ones_like(speed_obs)), dtype=float)[ok]
    if not np.isfinite(w).any() or float(np.nansum(w)) <= EPS:
        w = np.ones(ok.sum(), dtype=float)
    diff = speed_sim[ok] - speed_obs[ok]
    rmse = math.sqrt(float(np.nansum(w * diff * diff) / max(float(np.nansum(w)), EPS)))
    denom = max(float(np.nanmedian(speed_obs[ok])), 1e-4)
    return float(min(rmse / denom, 1.0))


def drift_local_rmse_loss(field_obs: Dict[str, np.ndarray], field_sim: Dict[str, np.ndarray], mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    ok = (
        mask
        & np.isfinite(field_obs["u"]) & np.isfinite(field_obs["v"])
        & np.isfinite(field_sim["u"]) & np.isfinite(field_sim["v"])
    )
    if ok.sum() < 3:
        return 1.0
    du = field_sim["u"][ok] - field_obs["u"][ok]
    dv = field_sim["v"][ok] - field_obs["v"][ok]
    rmse = math.sqrt(float(np.nanmean(du * du + dv * dv)))
    speed_obs = np.sqrt(field_obs["u"][ok] * field_obs["u"][ok] + field_obs["v"][ok] * field_obs["v"][ok])
    denom = max(float(np.nanpercentile(speed_obs, 75)), 1e-4)
    return float(min(rmse / denom, 1.0))


def phase_loss(cache: MetricCache, sim: SimArrays) -> Tuple[float, Dict[str, float]]:
    idx = cache.loss_sample_idx
    avail = cache.phase_available[idx]
    details: Dict[str, float] = {"phase_available_rows": float(avail.sum())}
    if avail.sum() < 100:
        return 0.0, {**details, "phase_loss_status": 0.0}
    pairs = {
        "phase_M_response_qdist": sim.pred_delta_M_response[idx][avail],
        "phase_Psi_active_qdist": sim.pred_delta_Psi_active[idx][avail],
        "phase_Psi_idle_qdist": sim.pred_delta_Psi_idle[idx][avail],
    }
    vals = []
    for key, values in pairs.items():
        val = quantile_distance_from_reference(
            cache.phase_quantile_references[key], values
        )
        details[key] = float(val) if np.isfinite(val) else np.nan
        if np.isfinite(val):
            vals.append(float(val))
    if not vals:
        return 0.0, details
    return float(max(vals)), details


def coverage_loss(cache: MetricCache, sim: SimArrays) -> Tuple[float, Dict[str, float]]:
    idx = cache.loss_sample_idx
    pairs = {
        "coverage_B_next_qdist": sim.pred_next_B[idx],
        "coverage_Amap_qdist": sim.scaled_active_alignable[idx],
        "coverage_A0_qdist": sim.scaled_active_neutral[idx],
        "coverage_idle_qdist": sim.scaled_idle[idx],
    }
    vals = []
    details: Dict[str, float] = {}
    for key, values in pairs.items():
        val = quantile_distance_from_reference(
            cache.coverage_quantile_references[key], values
        )
        details[key] = float(val) if np.isfinite(val) else np.nan
        if np.isfinite(val):
            vals.append(float(val))
    if not vals:
        return 0.0, details
    return float(max(vals)), details


def structure_metrics_fast_no_regions(cache: MetricCache, sim: SimArrays, label: str) -> Dict[str, float]:
    if cache.n_rows == 0:
        return {"label": label, "status": "empty"}
    eM = sim.pred_next_M - cache.target_M_next
    eP = sim.pred_next_Psi - cache.target_Psi_next
    eV = sim.pred_next_V - cache.target_V_next
    mse_main = float(np.nanmean((eM / cache.scale_M) ** 2 + (eP / cache.scale_Psi) ** 2) / 2.0)

    H_sim = occupancy_grid_weighted(sim.pred_next_M, sim.pred_next_Psi, cache.weights)
    occ_js = js_divergence(cache.H_obs + EPS, H_sim + EPS)

    f_sim = field_stats_from_arrays_weighted(
        cache.M,
        cache.Psi,
        sim.pred_next_M - cache.M,
        sim.pred_next_Psi - cache.Psi,
        cache.weights,
        cells=cache.current_cell,
        state_valid=cache.current_cell_valid,
        fixed_support=cache.field_obs,
    )
    mask = cache.field_obs["mask"] & f_sim["mask"]
    drift_corr = vector_corr(cache.field_obs["u"], cache.field_obs["v"], f_sim["u"], f_sim["v"], mask)
    drift_dir_loss = float(1.0 if not np.isfinite(drift_corr) else 0.5 * (1.0 - drift_corr))
    drift_mag_loss = drift_magnitude_loss(cache.field_obs, f_sim, mask)
    drift_local_loss = drift_local_rmse_loss(cache.field_obs, f_sim, mask)
    ph_loss, ph_details = phase_loss(cache, sim)
    cov_loss, cov_details = coverage_loss(cache, sim)

    out = {
        "label": label,
        "n_rows": float(cache.n_rows),
        "n_users": float(cache.n_users),
        "one_step_mse_main_norm": float(min(mse_main, 1.0)),
        "one_step_rmse_M": float(math.sqrt(np.nanmean(eM ** 2))),
        "one_step_rmse_Psi": float(math.sqrt(np.nanmean(eP ** 2))),
        "one_step_rmse_V_diagnostic_only": float(math.sqrt(np.nanmean(eV ** 2))),
        "occupancy_js_MR_PsiA": float(occ_js),
        "drift_vector_corr_MR_PsiA": float(drift_corr) if np.isfinite(drift_corr) else np.nan,
        "drift_direction_loss_MR_PsiA": float(drift_dir_loss),
        "drift_magnitude_loss_MR_PsiA": float(drift_mag_loss),
        "drift_local_rmse_loss_MR_PsiA": float(drift_local_loss),
        "phase_loss_max_qdist": float(ph_loss),
        "coverage_loss_max_qdist": float(cov_loss),
    }
    out.update(ph_details)
    out.update(cov_details)
    return out

@dataclass(frozen=True)
class FamilySpec:
    key: str
    label: str
    free_params: Tuple[str, ...]
    description: str
    role: str = "candidate"

    @property
    def parameter_count(self) -> int:
        return len(self.free_params)


FAMILIES: List[FamilySpec] = [
    FamilySpec(
        "persistence",
        "State persistence",
        tuple(),
        "No signed mechanism update; the next state equals the current state.",
        role="baseline",
    ),
    FamilySpec(
        "response_only",
        "Response-only signed mechanism",
        ("theta0", "thetaM"),
        "Response signed drive is active; alignment signed drive is disabled while fixed activity accounting and dilution remain.",
    ),
    FamilySpec(
        "alignment_only",
        "Alignment-only signed mechanism",
        ("phi0",),
        "Alignment signed drive is active; response signed drive is disabled while fixed response-evidence accounting and dilution remain.",
    ),
    FamilySpec("core_two_parameter", "Two-coordinate core", ("thetaM", "phi0"), "Response restoring plus alignment baseline."),
    FamilySpec("response_offset_core", "Response-offset core", ("theta0", "thetaM", "phi0"), "Response baseline/restoring plus alignment baseline."),
    FamilySpec("dual_channel_core", "Dual-channel core", ("thetaM", "phi0", "deltaS"), "Response restoring plus response/support alignment-channel contrast."),
    FamilySpec("offset_dual_channel", "Offset dual-channel", ("theta0", "thetaM", "phi0", "deltaS"), "Response baseline/restoring plus dual-channel alignment contrast."),
    FamilySpec("dual_plus_linear_coupling", "Dual-channel with linear coupling", ("thetaM", "thetaPsi", "phi0", "deltaS"), "Dual-channel core plus direct exposure-alignment effect on response residuals."),
    FamilySpec("dual_plus_interaction", "Dual-channel with interaction", ("thetaM", "thetaMPsi", "phi0", "deltaS"), "Dual-channel core plus M-by-alignment interaction."),
    FamilySpec(
        "full_reference",
        "Full seven-term reference mechanism",
        ("theta0", "thetaM", "thetaPsi", "thetaMPsi", "phi0", "deltaS", "phiPsi"),
        "Generic seven-term reference family.",
        role="reference",
    ),
]
FAMILY_BY_KEY = {family.key: family for family in FAMILIES}


@dataclass
class PredictionArrays:
    pred_next_M: np.ndarray
    pred_next_Psi: np.ndarray


@dataclass
class FamilyFit:
    spec: FamilySpec
    selected_params: Dict[str, float]
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    search_table: pd.DataFrame
    val_prediction: PredictionArrays


@dataclass
class ModelBootstrapStats:
    error_by_user: np.ndarray
    occupancy_by_user_cell: sparse.csr_matrix
    drift_sx_by_user_cell: sparse.csr_matrix
    drift_sy_by_user_cell: sparse.csr_matrix
    drift_mask: np.ndarray


@dataclass
class BootstrapBase:
    users: np.ndarray
    row_user_index: np.ndarray
    user_total_weight: np.ndarray
    target_occupancy_by_user_cell: sparse.csr_matrix
    current_weight_by_user_cell: sparse.csr_matrix
    observed_sx_by_user_cell: sparse.csr_matrix
    observed_sy_by_user_cell: sparse.csr_matrix
    observed_drift_mask: np.ndarray
    current_cell: np.ndarray
    current_cell_valid: np.ndarray
    n_cells: int


@dataclass
class BootstrapBank:
    multiplicities: np.ndarray


@dataclass
class BootstrapObservedAggregation:
    denominator: np.ndarray
    occupancy: np.ndarray
    drift_weight: np.ndarray
    drift_u: np.ndarray
    drift_v: np.ndarray


_EVALUATION_CACHE: Dict[Tuple[str, Tuple[float, ...]], Dict[str, float]] = {}
_BOOTSTRAP_VERIFICATION_ROWS: List[Dict[str, Any]] = []


def sample_users(panel: pd.DataFrame, max_users: int, seed: int) -> pd.DataFrame:
    if max_users <= 0 or panel.empty:
        return panel.copy()
    users = np.asarray(sorted(panel["user_id"].unique()), dtype=np.int64)
    if users.size <= max_users:
        return panel.copy()
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(users, size=max_users, replace=False).tolist())
    return panel[panel["user_id"].isin(keep)].copy()


def snap_to_grid(name: str, value: float, enabled: bool = True) -> float:
    if not enabled:
        return 0.0
    values = np.asarray(GRID[name], dtype=float)
    return float(values[np.argmin(np.abs(values - float(value)))])


def full_params_for_family(spec: FamilySpec, partial: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    params = {name: 0.0 for name in MECH_PARAMS}
    source = partial or {}
    for name in spec.free_params:
        params[name] = snap_to_grid(name, source.get(name, ANCHOR.get(name, GRID[name][0])), enabled=True)
    params.update(FIXED_NUISANCE)
    return params


def assert_candidate_on_family_grid(spec: FamilySpec, params: Dict[str, float]) -> None:
    for name in MECH_PARAMS:
        value = float(params.get(name, 0.0))
        if name in spec.free_params:
            allowed = np.asarray(GRID[name], dtype=float)
            if not np.any(np.isclose(value, allowed, atol=1e-12, rtol=0.0)):
                raise RuntimeError(f"Off-grid value for {spec.key}.{name}: {value}")
            if abs(value) < 1e-14:
                raise RuntimeError(f"Enabled mechanism parameter is zero: {spec.key}.{name}")
        elif abs(value) > 1e-12:
            raise RuntimeError(f"Disabled mechanism parameter is non-zero: {spec.key}.{name}={value}")
    for name, expected in FIXED_NUISANCE.items():
        value = float(params.get(name, np.nan))
        if not np.isclose(value, expected, atol=1e-12, rtol=0.0):
            raise RuntimeError(f"Nuisance scale changed for {spec.key}.{name}: {value} != {expected}")


def space_filling_candidates(
    spec: FamilySpec,
    max_candidates: int,
    seed: int,
    extra_anchors: Optional[Iterable[Mapping[str, float]]] = None,
) -> List[Dict[str, float]]:
    """Generate a deterministic finite candidate bank."""
    if spec.parameter_count == 0:
        return [full_params_for_family(spec)]
    rng = np.random.default_rng(seed)
    candidates: List[Dict[str, float]] = []
    anchors = list(PILOT_ANCHORS)
    if extra_anchors is not None:
        anchors.extend(dict(anchor) for anchor in extra_anchors)
    for anchor in anchors:
        candidates.append(full_params_for_family(spec, anchor))
    for name in spec.free_params:
        for value in GRID[name]:
            candidate = dict(ANCHOR)
            candidate[name] = float(value)
            candidates.append(full_params_for_family(spec, candidate))
    while len(candidates) < max_candidates:
        candidate = dict(ANCHOR)
        for name in spec.free_params:
            values = GRID[name]
            candidate[name] = float(values[int(rng.integers(0, len(values)))])
        candidates.append(full_params_for_family(spec, candidate))

    output: List[Dict[str, float]] = []
    seen: set[Tuple[Tuple[str, float], ...]] = set()
    for params in candidates:
        key = tuple((name, float(params[name])) for name in ALL_PARAMS)
        if key in seen:
            continue
        seen.add(key)
        assert_candidate_on_family_grid(spec, params)
        output.append(params)
        if len(output) >= max_candidates:
            break
    return output


def exhaustive_grid_candidates(
    spec: FamilySpec,
    max_combinations: int,
) -> Optional[List[Dict[str, float]]]:
    """Enumerate a deletion family's finite grid when it is tractable."""
    if spec.parameter_count == 0:
        return [full_params_for_family(spec)]
    total = math.prod(len(GRID[name]) for name in spec.free_params)
    if total > max_combinations:
        return None
    output: List[Dict[str, float]] = []
    values = [GRID[name] for name in spec.free_params]
    for combination in itertools.product(*values):
        params = {name: 0.0 for name in MECH_PARAMS}
        for name, value in zip(spec.free_params, combination):
            params[name] = float(value)
        params.update(FIXED_NUISANCE)
        assert_candidate_on_family_grid(spec, params)
        output.append(params)
    return output

def params_cache_key(params: Mapping[str, float], cache: MetricCache) -> Tuple[str, Tuple[float, ...]]:
    return cache.label, tuple(float(params[name]) for name in ALL_PARAMS)


def simulate_for_family(spec: FamilySpec, params: Dict[str, float], cache: MetricCache, calib: Calibration) -> SimArrays:
    if spec.key == "persistence":
        return persistence_sim_arrays(cache)
    assert_candidate_on_family_grid(spec, params)
    return simulate_arrays(cache, params, calib)


def persistence_sim_arrays(cache: MetricCache) -> SimArrays:
    zeros = np.zeros(cache.n_rows, dtype=float)
    return SimArrays(
        pred_next_M=cache.M.copy(),
        pred_next_Psi=cache.Psi.copy(),
        pred_next_V=cache.V.copy(),
        pred_next_B=cache.B.copy(),
        pred_delta_M_response=zeros.copy(),
        pred_delta_Psi_active=zeros.copy(),
        pred_delta_Psi_idle=zeros.copy(),
        scaled_active_alignable=zeros.copy(),
        scaled_active_neutral=zeros.copy(),
        scaled_idle=zeros.copy(),
    )


def light_prediction(sim: SimArrays) -> PredictionArrays:
    return PredictionArrays(
        pred_next_M=np.asarray(sim.pred_next_M, dtype=np.float64).copy(),
        pred_next_Psi=np.asarray(sim.pred_next_Psi, dtype=np.float64).copy(),
    )


def evaluate_params(spec: FamilySpec, params: Dict[str, float], cache: MetricCache, calib: Calibration) -> Dict[str, float]:
    key = params_cache_key(params, cache)
    cached = _EVALUATION_CACHE.get(key)
    if cached is not None:
        metrics = dict(cached)
        metrics["label"] = spec.key
        return metrics
    sim = simulate_for_family(spec, params, cache, calib)
    metrics = structure_metrics_fast_no_regions(cache, sim, spec.key)
    metrics.update(objective_diagnostics(metrics))
    metrics["objective_loss"] = float(objective_from_metrics(metrics))
    metrics["primary_score"] = float(primary_objective_score(metrics))
    cached_metrics = dict(metrics)
    cached_metrics.pop("label", None)
    _EVALUATION_CACHE[key] = cached_metrics
    return metrics


def candidate_record(
    spec: FamilySpec,
    stage: str,
    params: Dict[str, float],
    metrics: Dict[str, float],
    rank: Optional[int] = None,
) -> Dict[str, object]:
    record: Dict[str, object] = {
        "family_key": spec.key,
        "family_label": spec.label,
        "family_role": spec.role,
        "stage": stage,
        "free_mechanism_parameters": spec.parameter_count,
        "rank": rank if rank is not None else np.nan,
    }
    for name in ALL_PARAMS:
        record[name] = float(params.get(name, 0.0))
    record.update({
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating)) or pd.isna(value)
    })
    return record


def local_coordinate_refine(
    spec: FamilySpec,
    start: Dict[str, float],
    cache: MetricCache,
    calib: Calibration,
    max_evals: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    best = dict(start)
    best_metrics = evaluate_params(spec, best, cache, calib)
    best_loss = float(best_metrics["objective_loss"])
    evaluations = 1
    improved = True
    while improved and evaluations < max_evals:
        improved = False
        for name in spec.free_params:
            if evaluations >= max_evals:
                break
            for value in GRID[name]:
                if evaluations >= max_evals:
                    break
                if np.isclose(best[name], value):
                    continue
                candidate = dict(best)
                candidate[name] = float(value)
                metrics = evaluate_params(spec, candidate, cache, calib)
                evaluations += 1
                loss = float(metrics["objective_loss"])
                if loss < best_loss:
                    best = candidate
                    best_metrics = metrics
                    best_loss = loss
                    improved = True
                    break
    best_metrics = dict(best_metrics)
    best_metrics["local_refine_evals"] = float(evaluations)
    return best, best_metrics


def fit_family(
    spec: FamilySpec,
    screen_cache: MetricCache,
    train_cache: MetricCache,
    val_cache: MetricCache,
    calib: Calibration,
    seed: int,
    *,
    candidate_mode: str = "standard",
    max_candidates_override: Optional[int] = None,
    full_train_top_k_override: Optional[int] = None,
    val_shortlist_k_override: Optional[int] = None,
    local_refine_max_evals_override: Optional[int] = None,
    refine_starts: int = 3,
    extra_anchors: Optional[Iterable[Mapping[str, float]]] = None,
) -> FamilyFit:
    if spec.parameter_count == 0:
        params = full_params_for_family(spec)
        train_metrics = evaluate_params(spec, params, train_cache, calib)
        val_metrics = evaluate_params(spec, params, val_cache, calib)
        prediction = light_prediction(simulate_for_family(spec, params, val_cache, calib))
        search = pd.DataFrame([candidate_record(spec, "baseline", params, val_metrics, 1)])
        search["search_mode"] = candidate_mode
        return FamilyFit(spec, params, train_metrics, val_metrics, search, prediction)

    default_candidates = min(
        max(SCREENING_MAX_CANDIDATES_PER_FAMILY, 8),
        max(8, 12 * spec.parameter_count),
    )
    max_candidates = int(max_candidates_override or default_candidates)
    if candidate_mode == "exhaustive":
        candidates = exhaustive_grid_candidates(
            spec, max(DELETION_EXHAUSTIVE_MAX_COMBINATIONS, 1)
        )
        if candidates is None:
            candidates = space_filling_candidates(
                spec, max_candidates, seed, extra_anchors=extra_anchors
            )
            effective_mode = "expanded_finite"
        else:
            effective_mode = "exhaustive_finite"
    elif candidate_mode == "standard":
        candidates = space_filling_candidates(
            spec, max_candidates, seed, extra_anchors=extra_anchors
        )
        effective_mode = "standard_finite"
    else:
        raise ValueError(f"Unknown candidate mode: {candidate_mode!r}")

    screen_records: List[Dict[str, object]] = []
    for candidate in candidates:
        metrics = evaluate_params(spec, candidate, screen_cache, calib)
        screen_records.append(candidate_record(spec, "A_train_screen", candidate, metrics))
    screen_df = (
        pd.DataFrame(screen_records)
        .sort_values("objective_loss", kind="mergesort")
        .reset_index(drop=True)
    )

    refine_limit = int(local_refine_max_evals_override or LOCAL_REFINE_MAX_EVALS)
    refined_records: List[Dict[str, object]] = []
    for _, row in screen_df.head(min(max(refine_starts, 1), len(screen_df))).iterrows():
        start = {name: float(row[name]) for name in ALL_PARAMS}
        params_refined, metrics_refined = local_coordinate_refine(
            spec, start, screen_cache, calib, refine_limit
        )
        refined_records.append(candidate_record(
            spec, "A_train_screen_refined", params_refined, metrics_refined
        ))
    if refined_records:
        screen_df = pd.concat([screen_df, pd.DataFrame(refined_records)], ignore_index=True)
        screen_df = (
            screen_df
            .drop_duplicates(subset=ALL_PARAMS, keep="first")
            .sort_values("objective_loss", kind="mergesort")
            .reset_index(drop=True)
        )

    full_top_k = int(full_train_top_k_override or FULL_TRAIN_TOP_K)
    val_top_k = int(val_shortlist_k_override or VAL_SHORTLIST_K)
    train_records: List[Dict[str, object]] = []
    for rank, (_, row) in enumerate(screen_df.head(full_top_k).iterrows(), start=1):
        candidate = {name: float(row[name]) for name in ALL_PARAMS}
        metrics = evaluate_params(spec, candidate, train_cache, calib)
        train_records.append(candidate_record(spec, "A_train_full", candidate, metrics, rank))
    train_df = (
        pd.DataFrame(train_records)
        .sort_values("objective_loss", kind="mergesort")
        .reset_index(drop=True)
    )

    val_records: List[Dict[str, object]] = []
    for rank, (_, row) in enumerate(train_df.head(val_top_k).iterrows(), start=1):
        candidate = {name: float(row[name]) for name in ALL_PARAMS}
        metrics = evaluate_params(spec, candidate, val_cache, calib)
        val_records.append(candidate_record(spec, "A_val_selection", candidate, metrics, rank))
    val_df = (
        pd.DataFrame(val_records)
        .sort_values("objective_loss", kind="mergesort")
        .reset_index(drop=True)
    )
    best_row = val_df.iloc[0]
    selected = {name: float(best_row[name]) for name in ALL_PARAMS}
    train_metrics = evaluate_params(spec, selected, train_cache, calib)
    val_metrics = evaluate_params(spec, selected, val_cache, calib)
    prediction = light_prediction(simulate_for_family(spec, selected, val_cache, calib))
    search = pd.concat([screen_df, train_df, val_df], ignore_index=True, sort=False)
    search["search_mode"] = effective_mode
    return FamilyFit(spec, selected, train_metrics, val_metrics, search, prediction)

def family_checkpoint_path(spec: FamilySpec) -> Path:
    return CHECKPOINT_DIR / f"{spec.key}.json"


def save_family_checkpoint(fit: FamilyFit, run_hash: str) -> None:
    payload = {
        "run_hash": run_hash,
        "family": asdict(fit.spec),
        "selected_params": fit.selected_params,
        "train_metrics": fit.train_metrics,
        "val_metrics": fit.val_metrics,
        "search_table": str((SEARCH_DIR / f"{fit.spec.key}_search.csv").resolve()),
    }
    save_json(payload, family_checkpoint_path(fit.spec))


def load_family_checkpoint(
    spec: FamilySpec,
    run_hash: str,
    val_cache: MetricCache,
    calib: Calibration,
) -> Optional[FamilyFit]:
    path = family_checkpoint_path(spec)
    search_path = SEARCH_DIR / f"{spec.key}_search.csv"
    if not path.exists() or not search_path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("run_hash", "")) != run_hash:
        return None
    params = {name: float(value) for name, value in payload["selected_params"].items()}
    assert_candidate_on_family_grid(spec, params)
    sim = simulate_for_family(spec, params, val_cache, calib)
    return FamilyFit(
        spec=spec,
        selected_params=params,
        train_metrics=dict(payload["train_metrics"]),
        val_metrics=dict(payload["val_metrics"]),
        search_table=pd.read_csv(search_path),
        val_prediction=light_prediction(sim),
    )


def fit_or_resume_family(
    spec: FamilySpec,
    screen_cache: MetricCache,
    train_cache: MetricCache,
    val_cache: MetricCache,
    calib: Calibration,
    seed: int,
    run_hash: str,
    resume: bool,
) -> FamilyFit:
    if resume:
        fit = load_family_checkpoint(spec, run_hash, val_cache, calib)
        if fit is not None:
            print(f"[minimality] resumed family {spec.key}", flush=True)
            return fit
    fit = fit_family(spec, screen_cache, train_cache, val_cache, calib, seed)
    write_csv(fit.search_table, SEARCH_DIR / f"{spec.key}_search.csv")
    save_family_checkpoint(fit, run_hash)
    return fit


# -----------------------------------------------------------------------------
# Paired user bootstrap
# -----------------------------------------------------------------------------
def _cells(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    nx = len(GRID_BINS_SIGNED) - 1
    ny = nx
    ix = digitize_closed_right(x, GRID_BINS_SIGNED)
    iy = digitize_closed_right(y, GRID_BINS_SIGNED)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (ix >= 0)
        & (ix < nx)
        & (iy >= 0)
        & (iy < ny)
    )
    flat = np.full(len(x), -1, dtype=np.int32)
    flat[valid] = (ix[valid] * ny + iy[valid]).astype(np.int32)
    return flat, valid


def _user_cell_matrix(
    row_user_index: np.ndarray,
    cell: np.ndarray,
    valid: np.ndarray,
    values: np.ndarray,
    n_users: int,
    n_cells: int,
) -> sparse.csr_matrix:
    if not np.any(valid):
        return sparse.csr_matrix((n_users, n_cells), dtype=np.float64)
    matrix = sparse.coo_matrix(
        (
            np.asarray(values, dtype=np.float64)[valid],
            (row_user_index[valid], cell[valid]),
        ),
        shape=(n_users, n_cells),
        dtype=np.float64,
    )
    return matrix.tocsr()


def build_bootstrap_base(cache: MetricCache) -> BootstrapBase:
    users, row_user_index = np.unique(cache.uid, return_inverse=True)
    row_user_index = row_user_index.astype(np.int32, copy=False)
    n_users = len(users)
    n_cells = (len(GRID_BINS_SIGNED) - 1) ** 2
    current_cell = cache.current_cell
    current_valid = cache.current_cell_valid
    target_cell, target_valid = _cells(cache.target_M_next, cache.target_Psi_next)
    observed_dx = cache.target_M_next - cache.M
    observed_dy = cache.target_Psi_next - cache.Psi
    observed_valid = current_valid & np.isfinite(observed_dx) & np.isfinite(observed_dy)
    observed_count = np.bincount(
        current_cell[observed_valid], minlength=n_cells
    ).astype(np.float64)
    user_total_weight = np.bincount(
        row_user_index,
        weights=cache.weights,
        minlength=n_users,
    ).astype(np.float64)
    return BootstrapBase(
        users=users.astype(np.int64, copy=False),
        row_user_index=row_user_index,
        user_total_weight=user_total_weight,
        target_occupancy_by_user_cell=_user_cell_matrix(
            row_user_index, target_cell, target_valid, cache.weights, n_users, n_cells
        ),
        current_weight_by_user_cell=_user_cell_matrix(
            row_user_index, current_cell, observed_valid, cache.weights, n_users, n_cells
        ),
        observed_sx_by_user_cell=_user_cell_matrix(
            row_user_index,
            current_cell,
            observed_valid,
            cache.weights * observed_dx,
            n_users,
            n_cells,
        ),
        observed_sy_by_user_cell=_user_cell_matrix(
            row_user_index,
            current_cell,
            observed_valid,
            cache.weights * observed_dy,
            n_users,
            n_cells,
        ),
        observed_drift_mask=observed_count >= MIN_DRIFT_BIN_COUNT,
        current_cell=current_cell,
        current_cell_valid=current_valid,
        n_cells=n_cells,
    )


def build_model_bootstrap_stats(
    cache: MetricCache,
    prediction: PredictionArrays,
    base: BootstrapBase,
) -> ModelBootstrapStats:
    pred_M = np.asarray(prediction.pred_next_M, dtype=np.float64)
    pred_Psi = np.asarray(prediction.pred_next_Psi, dtype=np.float64)
    if len(pred_M) != cache.n_rows or len(pred_Psi) != cache.n_rows:
        raise ValueError("Prediction length does not match validation cache.")
    z = (
        ((pred_M - cache.target_M_next) / cache.scale_M) ** 2
        + ((pred_Psi - cache.target_Psi_next) / cache.scale_Psi) ** 2
    ) / 2.0
    z = np.where(np.isfinite(z), z, 0.0)
    error_by_user = np.bincount(
        base.row_user_index,
        weights=cache.weights * z,
        minlength=len(base.users),
    ).astype(np.float64)

    pred_cell, pred_valid = _cells(pred_M, pred_Psi)
    model_dx = pred_M - cache.M
    model_dy = pred_Psi - cache.Psi
    drift_valid = base.current_cell_valid & np.isfinite(model_dx) & np.isfinite(model_dy)
    drift_count = np.bincount(
        base.current_cell[drift_valid], minlength=base.n_cells
    ).astype(np.float64)
    return ModelBootstrapStats(
        error_by_user=error_by_user,
        occupancy_by_user_cell=_user_cell_matrix(
            base.row_user_index,
            pred_cell,
            pred_valid,
            cache.weights,
            len(base.users),
            base.n_cells,
        ),
        drift_sx_by_user_cell=_user_cell_matrix(
            base.row_user_index,
            base.current_cell,
            drift_valid,
            cache.weights * model_dx,
            len(base.users),
            base.n_cells,
        ),
        drift_sy_by_user_cell=_user_cell_matrix(
            base.row_user_index,
            base.current_cell,
            drift_valid,
            cache.weights * model_dy,
            len(base.users),
            base.n_cells,
        ),
        drift_mask=drift_count >= MIN_DRIFT_BIN_COUNT,
    )


def build_bootstrap_bank(users: np.ndarray, reps: int, seed: int) -> BootstrapBank:
    rng = np.random.default_rng(seed)
    counts = np.zeros((reps, len(users)), dtype=np.int32)
    for rep in range(reps):
        sample = rng.choice(users, size=len(users), replace=True)
        sampled_index = np.searchsorted(users, sample)
        counts[rep] = np.bincount(sampled_index, minlength=len(users)).astype(np.int32)
    return BootstrapBank(multiplicities=counts)


def _aggregate_sparse(multiplicities: np.ndarray, matrix: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(matrix.T.dot(np.asarray(multiplicities, dtype=np.float64).T).T)


def aggregate_observed(base: BootstrapBase, bank: BootstrapBank) -> BootstrapObservedAggregation:
    counts = bank.multiplicities
    denominator = np.asarray(counts @ base.user_total_weight, dtype=np.float64)
    occupancy_raw = _aggregate_sparse(counts, base.target_occupancy_by_user_cell)
    occupancy = occupancy_raw / np.maximum(occupancy_raw.sum(axis=1, keepdims=True), EPS)
    drift_weight = _aggregate_sparse(counts, base.current_weight_by_user_cell)
    sx = _aggregate_sparse(counts, base.observed_sx_by_user_cell)
    sy = _aggregate_sparse(counts, base.observed_sy_by_user_cell)
    drift_u = sx / np.maximum(drift_weight, EPS)
    drift_v = sy / np.maximum(drift_weight, EPS)
    return BootstrapObservedAggregation(
        denominator=denominator,
        occupancy=occupancy,
        drift_weight=drift_weight,
        drift_u=drift_u,
        drift_v=drift_v,
    )


def score_model_optimized(
    stats: ModelBootstrapStats,
    base: BootstrapBase,
    bank: BootstrapBank,
    observed: BootstrapObservedAggregation,
) -> np.ndarray:
    counts = bank.multiplicities
    mse_num = np.asarray(counts @ stats.error_by_user, dtype=np.float64)
    mse = np.minimum(mse_num / np.maximum(observed.denominator, EPS), 1.0)

    model_occ_raw = _aggregate_sparse(counts, stats.occupancy_by_user_cell)
    model_occ = model_occ_raw / np.maximum(model_occ_raw.sum(axis=1, keepdims=True), EPS)
    model_sx = _aggregate_sparse(counts, stats.drift_sx_by_user_cell)
    model_sy = _aggregate_sparse(counts, stats.drift_sy_by_user_cell)
    model_u = model_sx / np.maximum(observed.drift_weight, EPS)
    model_v = model_sy / np.maximum(observed.drift_weight, EPS)
    mask_flat = base.observed_drift_mask & stats.drift_mask
    mask = mask_flat.reshape(len(GRID_BINS_SIGNED) - 1, len(GRID_BINS_SIGNED) - 1)

    scores = np.empty(len(counts), dtype=np.float64)
    shape = mask.shape
    for rep in range(len(counts)):
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
        corr = vector_corr(
            obs_field["u"], obs_field["v"], model_field["u"], model_field["v"], mask
        )
        metrics = {
            "one_step_mse_main_norm": float(mse[rep]),
            "occupancy_js_MR_PsiA": float(js_divergence(
                observed.occupancy[rep].reshape(shape) + EPS,
                model_occ[rep].reshape(shape) + EPS,
            )),
            "drift_local_rmse_loss_MR_PsiA": float(drift_local_rmse_loss(
                obs_field, model_field, mask
            )),
            "drift_direction_loss_MR_PsiA": float(
                1.0 if not np.isfinite(corr) else 0.5 * (1.0 - corr)
            ),
            "drift_magnitude_loss_MR_PsiA": float(drift_magnitude_loss(
                obs_field, model_field, mask
            )),
            "phase_loss_max_qdist": 0.0,
            "coverage_loss_max_qdist": 0.0,
        }
        scores[rep] = primary_objective_score(metrics)
    return scores


def weighted_primary_score_reference(
    cache: MetricCache,
    prediction: PredictionArrays,
    base: BootstrapBase,
    multiplicities: np.ndarray,
) -> float:
    row_mult = np.asarray(multiplicities, dtype=np.float64)[base.row_user_index]
    if float(row_mult.sum()) <= 0:
        return np.nan
    weights = cache.weights * row_mult
    eM = prediction.pred_next_M - cache.target_M_next
    eP = prediction.pred_next_Psi - cache.target_Psi_next
    mse_main = float(
        np.nansum(weights * ((eM / cache.scale_M) ** 2 + (eP / cache.scale_Psi) ** 2) / 2.0)
        / max(np.nansum(weights), EPS)
    )
    H_obs = occupancy_grid_weighted(cache.target_M_next, cache.target_Psi_next, weights)
    H_sim = occupancy_grid_weighted(prediction.pred_next_M, prediction.pred_next_Psi, weights)
    occ_js = js_divergence(H_obs + EPS, H_sim + EPS)
    f_obs = field_stats_from_arrays_weighted(
        cache.M,
        cache.Psi,
        cache.target_M_next - cache.M,
        cache.target_Psi_next - cache.Psi,
        weights,
    )
    f_sim = field_stats_from_arrays_weighted(
        cache.M,
        cache.Psi,
        prediction.pred_next_M - cache.M,
        prediction.pred_next_Psi - cache.Psi,
        weights,
    )
    mask = f_obs["mask"] & f_sim["mask"]
    corr = vector_corr(f_obs["u"], f_obs["v"], f_sim["u"], f_sim["v"], mask)
    metrics = {
        "one_step_mse_main_norm": float(min(mse_main, 1.0)),
        "occupancy_js_MR_PsiA": float(occ_js),
        "drift_local_rmse_loss_MR_PsiA": float(drift_local_rmse_loss(f_obs, f_sim, mask)),
        "drift_direction_loss_MR_PsiA": float(
            1.0 if not np.isfinite(corr) else 0.5 * (1.0 - corr)
        ),
        "drift_magnitude_loss_MR_PsiA": float(drift_magnitude_loss(f_obs, f_sim, mask)),
        "phase_loss_max_qdist": 0.0,
        "coverage_loss_max_qdist": 0.0,
    }
    return float(primary_objective_score(metrics))


class BootstrapScorer:
    def __init__(self, cache: MetricCache):
        self.cache = cache
        self.base = build_bootstrap_base(cache)
        self.model_stats: Dict[Tuple[str, int, int], ModelBootstrapStats] = {}
        self.bank_cache: Dict[Tuple[int, int], BootstrapBank] = {}
        self.observed_cache: Dict[Tuple[int, int], BootstrapObservedAggregation] = {}

    def stats_for(self, key: str, prediction: PredictionArrays) -> ModelBootstrapStats:
        cache_key = (key, id(prediction.pred_next_M), id(prediction.pred_next_Psi))
        if cache_key not in self.model_stats:
            self.model_stats[cache_key] = build_model_bootstrap_stats(
                self.cache, prediction, self.base
            )
        return self.model_stats[cache_key]

    def bank_for(self, reps: int, seed: int) -> BootstrapBank:
        key = (int(reps), int(seed))
        if key not in self.bank_cache:
            self.bank_cache[key] = build_bootstrap_bank(self.base.users, reps, seed)
        return self.bank_cache[key]

    def observed_for(
        self, reps: int, seed: int, bank: BootstrapBank
    ) -> BootstrapObservedAggregation:
        key = (int(reps), int(seed))
        if key not in self.observed_cache:
            self.observed_cache[key] = aggregate_observed(self.base, bank)
        return self.observed_cache[key]

    def score(
        self,
        predictions: Mapping[str, PredictionArrays],
        reps: int,
        seed: int,
        engine: str,
        verify: bool,
    ) -> pd.DataFrame:
        bank = self.bank_for(reps, seed)
        keys = list(predictions.keys())
        if engine == "reference":
            score_map = {
                key: np.asarray([
                    weighted_primary_score_reference(
                        self.cache,
                        predictions[key],
                        self.base,
                        bank.multiplicities[rep],
                    )
                    for rep in range(reps)
                ], dtype=np.float64)
                for key in keys
            }
        elif engine == "optimized":
            observed = self.observed_for(reps, seed, bank)
            score_map = {
                key: score_model_optimized(
                    self.stats_for(key, predictions[key]),
                    self.base,
                    bank,
                    observed,
                )
                for key in keys
            }
            if verify and reps > 0 and keys:
                n_verify = min(max(VERIFY_BOOTSTRAP_REPS, 1), reps)
                verify_indices = sorted({0, len(keys) // 2, len(keys) - 1})
                for index in verify_indices:
                    key = keys[index]
                    for rep in range(n_verify):
                        reference = weighted_primary_score_reference(
                            self.cache,
                            predictions[key],
                            self.base,
                            bank.multiplicities[rep],
                        )
                        optimized = float(score_map[key][rep])
                        difference = abs(reference - optimized)
                        _BOOTSTRAP_VERIFICATION_ROWS.append({
                            "seed": seed,
                            "family_key": key,
                            "bootstrap_rep": rep,
                            "reference_score": reference,
                            "optimized_score": optimized,
                            "absolute_difference": difference,
                        })
                        if difference > 1e-10:
                            raise RuntimeError(
                                f"Optimized bootstrap mismatch for {key}, rep {rep}: "
                                f"reference={reference}, optimized={optimized}"
                            )
        else:
            raise ValueError(f"Unknown bootstrap engine: {engine!r}")

        rows: List[Dict[str, object]] = []
        for rep in range(reps):
            for key in keys:
                rows.append({
                    "bootstrap_rep": int(rep),
                    "family_key": key,
                    "primary_score": float(score_map[key][rep]),
                })
        return pd.DataFrame(rows)

def bootstrap_family_scores(
    scorer: BootstrapScorer,
    family_predictions: Mapping[str, PredictionArrays],
    reps: int,
    seed: int,
) -> pd.DataFrame:
    return scorer.score(
        family_predictions,
        reps,
        seed,
        engine=BOOTSTRAP_ENGINE,
        verify=VERIFY_OPTIMIZED_BOOTSTRAP,
    )


def summarize_bootstrap(boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in boot.groupby("family_key"):
        values = group["primary_score"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        rows.append({
            "family_key": key,
            "bootstrap_mean_primary_score": float(np.mean(values)) if values.size else np.nan,
            "bootstrap_standard_error": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
            "bootstrap_ci95_lower": float(np.quantile(values, 0.025)) if values.size else np.nan,
            "bootstrap_ci95_upper": float(np.quantile(values, 0.975)) if values.size else np.nan,
        })
    return pd.DataFrame(rows)


def paired_difference(boot: pd.DataFrame, simpler: str, richer: str, margin: float) -> Dict[str, object]:
    simpler_df = boot[boot.family_key == simpler][["bootstrap_rep", "primary_score"]].rename(
        columns={"primary_score": "simpler_score"}
    )
    richer_df = boot[boot.family_key == richer][["bootstrap_rep", "primary_score"]].rename(
        columns={"primary_score": "richer_score"}
    )
    merged = simpler_df.merge(richer_df, on="bootstrap_rep", how="inner")
    difference = merged["simpler_score"].to_numpy(dtype=float) - merged["richer_score"].to_numpy(dtype=float)
    difference = difference[np.isfinite(difference)]
    if difference.size == 0:
        return {"simpler_family": simpler, "richer_family": richer, "status": "no_bootstrap_overlap"}
    lower = float(np.quantile(difference, 0.025))
    upper = float(np.quantile(difference, 0.975))
    if lower > margin:
        conclusion = "richer_required"
    elif upper <= margin:
        conclusion = "simpler_practically_equivalent"
    else:
        conclusion = "inconclusive"
    return {
        "simpler_family": simpler,
        "richer_family": richer,
        "practical_equivalence_margin": float(margin),
        "paired_difference_mean": float(np.mean(difference)),
        "paired_difference_median": float(np.median(difference)),
        "paired_difference_ci95_lower": lower,
        "paired_difference_ci95_upper": upper,
        "probability_richer_improves": float(np.mean(difference > 0.0)),
        "probability_improvement_exceeds_margin": float(np.mean(difference > margin)),
        "conclusion": conclusion,
    }


def difference_to_best(boot: pd.DataFrame, candidate: str, best: str, margin: float) -> Dict[str, object]:
    if candidate == best:
        return {
            "family_key": candidate,
            "difference_to_best_mean": 0.0,
            "difference_to_best_ci95_lower": 0.0,
            "difference_to_best_ci95_upper": 0.0,
            "practically_equivalent_to_best": True,
        }
    candidate_df = boot[boot.family_key == candidate][["bootstrap_rep", "primary_score"]].rename(
        columns={"primary_score": "candidate_score"}
    )
    best_df = boot[boot.family_key == best][["bootstrap_rep", "primary_score"]].rename(
        columns={"primary_score": "best_score"}
    )
    merged = candidate_df.merge(best_df, on="bootstrap_rep", how="inner")
    difference = merged["candidate_score"].to_numpy(dtype=float) - merged["best_score"].to_numpy(dtype=float)
    difference = difference[np.isfinite(difference)]
    if difference.size == 0:
        return {
            "family_key": candidate,
            "difference_to_best_mean": np.nan,
            "difference_to_best_ci95_lower": np.nan,
            "difference_to_best_ci95_upper": np.nan,
            "practically_equivalent_to_best": False,
        }
    upper = float(np.quantile(difference, 0.975))
    return {
        "family_key": candidate,
        "difference_to_best_mean": float(np.mean(difference)),
        "difference_to_best_ci95_lower": float(np.quantile(difference, 0.025)),
        "difference_to_best_ci95_upper": upper,
        "practically_equivalent_to_best": bool(upper <= margin),
    }


def build_results_table(
    fits: Dict[str, FamilyFit],
    boot: pd.DataFrame,
    margin: float,
) -> Tuple[pd.DataFrame, str, float, str]:
    bootstrap_summary = summarize_bootstrap(boot)
    rows = []
    for key, fit in fits.items():
        row = {
            "family_key": key,
            "Model family": fit.spec.label,
            "Role": fit.spec.role,
            "Free mechanism parameters": fit.spec.parameter_count,
            "Free parameter names": ";".join(fit.spec.free_params),
            "Fixed nuisance scales": len(FIXED_NUISANCE) if fit.spec.parameter_count > 0 else 0,
            "Training structural loss": fit.train_metrics.get("objective_loss", np.nan),
            "Validation structural loss": fit.val_metrics.get("objective_loss", np.nan),
            "Validation primary score": fit.val_metrics.get("objective_primary_score", fit.val_metrics.get("primary_score", np.nan)),
            "One-step closure discrepancy": fit.val_metrics.get("one_step_mse_main_norm", np.nan),
            "Landscape divergence": fit.val_metrics.get("occupancy_js_MR_PsiA", np.nan),
            "Local drift discrepancy": fit.val_metrics.get("drift_local_rmse_loss_MR_PsiA", np.nan),
            "Drift-direction discrepancy": fit.val_metrics.get("drift_direction_loss_MR_PsiA", np.nan),
            "Drift-speed discrepancy": fit.val_metrics.get("drift_magnitude_loss_MR_PsiA", np.nan),
            "Signed response next-state RMSE": fit.val_metrics.get("one_step_rmse_M", np.nan),
            "Exposure-alignment next-state RMSE": fit.val_metrics.get("one_step_rmse_Psi", np.nan),
        }
        for name in ALL_PARAMS:
            row[name] = fit.selected_params.get(name, 0.0)
        rows.append(row)
    results = pd.DataFrame(rows).merge(bootstrap_summary, on="family_key", how="left")
    results = results.rename(columns={
        "bootstrap_mean_primary_score": "Bootstrap mean primary score",
        "bootstrap_standard_error": "Bootstrap standard error",
        "bootstrap_ci95_lower": "Bootstrap 95% CI lower",
        "bootstrap_ci95_upper": "Bootstrap 95% CI upper",
    })
    best_key = str(results.sort_values("Bootstrap mean primary score", kind="mergesort").iloc[0]["family_key"])
    best_mean = float(results.loc[results.family_key == best_key, "Bootstrap mean primary score"].iloc[0])
    best_se = float(results.loc[results.family_key == best_key, "Bootstrap standard error"].iloc[0])
    one_se_threshold = best_mean + best_se
    differences = pd.DataFrame([
        difference_to_best(boot, key, best_key, margin)
        for key in results.family_key.tolist()
    ])
    results = results.merge(differences, on="family_key", how="left")
    results["Within one standard error of best"] = results["Bootstrap mean primary score"] <= one_se_threshold
    results["Practically equivalent to best"] = results["practically_equivalent_to_best"].astype(bool)
    eligible = results[
        results["Within one standard error of best"]
        & results["Practically equivalent to best"]
    ]
    if eligible.empty:
        selected_key = best_key
    else:
        selected_key = str(
            eligible.sort_values(
                ["Free mechanism parameters", "Bootstrap mean primary score"],
                kind="mergesort",
            ).iloc[0]["family_key"]
        )
    results["Parsimonious family selected"] = results["family_key"] == selected_key
    return results, best_key, float(one_se_threshold), selected_key


def final_model_validation(results: pd.DataFrame, final_key: str, best_key: str) -> Dict[str, object]:
    if final_key not in set(results["family_key"].astype(str)):
        return {
            "final_family_key": final_key,
            "present_in_results": False,
            "within_one_standard_error": False,
            "practically_equivalent_to_best": False,
            "difference_to_best_ci95_upper": np.nan,
        }
    row = results[results.family_key == final_key].iloc[0]
    return {
        "final_family_key": final_key,
        "best_family_key": best_key,
        "present_in_results": True,
        "within_one_standard_error": bool(row["Within one standard error of best"]),
        "practically_equivalent_to_best": bool(row["Practically equivalent to best"]),
        "difference_to_best_mean": float(row.get("difference_to_best_mean", np.nan)),
        "difference_to_best_ci95_lower": float(row.get("difference_to_best_ci95_lower", np.nan)),
        "difference_to_best_ci95_upper": float(row.get("difference_to_best_ci95_upper", np.nan)),
        "bootstrap_mean_primary_score": float(row.get("Bootstrap mean primary score", np.nan)),
        "free_mechanism_parameters": int(row.get("Free mechanism parameters", -1)),
    }


def canonical_family_spec(
    free_params: Tuple[str, ...],
    parent: FamilyFit,
    removed_parameter: str,
) -> FamilySpec:
    for family in FAMILIES:
        if tuple(family.free_params) == tuple(free_params):
            return family
    return FamilySpec(
        f"{parent.spec.key}_minus_{removed_parameter}",
        f"{parent.spec.label} without {removed_parameter}",
        tuple(free_params),
        f"One-term deletion of {removed_parameter} from {parent.spec.key}.",
    )


def fit_confirmatory_deletion(
    parent: FamilyFit,
    parameter: str,
    fits: Mapping[str, FamilyFit],
    screen_cache: MetricCache,
    train_cache: MetricCache,
    val_cache: MetricCache,
    calib: Calibration,
    seed: int,
) -> FamilyFit:
    """Refit one direct deletion with a conservative finite search."""
    free_params = tuple(name for name in parent.spec.free_params if name != parameter)
    spec = canonical_family_spec(free_params, parent, parameter)
    existing = [
        fit for fit in fits.values()
        if tuple(fit.spec.free_params) == tuple(free_params)
    ]
    existing_fit = min(
        existing,
        key=lambda fit: float(fit.val_metrics.get("objective_loss", np.inf)),
        default=None,
    )
    fallback_candidates = max(
        SCREENING_MAX_CANDIDATES_PER_FAMILY,
        24 * max(spec.parameter_count, 1),
    )
    confirmatory = fit_family(
        spec,
        screen_cache,
        train_cache,
        val_cache,
        calib,
        seed,
        candidate_mode="exhaustive",
        max_candidates_override=fallback_candidates,
        full_train_top_k_override=max(FULL_TRAIN_TOP_K, DELETION_FULL_TRAIN_TOP_K),
        val_shortlist_k_override=max(VAL_SHORTLIST_K, DELETION_VAL_SHORTLIST_K),
        local_refine_max_evals_override=max(
            LOCAL_REFINE_MAX_EVALS, DELETION_LOCAL_REFINE_MAX_EVALS
        ),
        refine_starts=max(DELETION_REFINE_STARTS, 1),
        extra_anchors=[parent.selected_params],
    )
    confirmatory.search_table = confirmatory.search_table.copy()
    confirmatory.search_table["search_pass"] = "confirmatory_deletion"
    if existing_fit is None:
        return confirmatory

    existing_search = existing_fit.search_table.copy()
    existing_search["search_pass"] = "prespecified_family"
    combined_search = pd.concat(
        [existing_search, confirmatory.search_table],
        ignore_index=True,
        sort=False,
    )
    existing_loss = float(existing_fit.val_metrics.get("objective_loss", np.inf))
    confirmatory_loss = float(confirmatory.val_metrics.get("objective_loss", np.inf))
    chosen = confirmatory if confirmatory_loss < existing_loss else existing_fit
    return FamilyFit(
        spec=spec,
        selected_params=dict(chosen.selected_params),
        train_metrics=dict(chosen.train_metrics),
        val_metrics=dict(chosen.val_metrics),
        search_table=combined_search,
        val_prediction=chosen.val_prediction,
    )


def resolve_scalar_minimality_one_run(
    fits: Dict[str, FamilyFit],
    selected_key: str,
    screen_cache: MetricCache,
    train_cache: MetricCache,
    val_cache: MetricCache,
    calib: Calibration,
    scorer: BootstrapScorer,
    fit_seed: int,
    bootstrap_seed: int,
) -> Tuple[
    FamilyFit,
    Dict[str, FamilyFit],
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    str,
    float,
    str,
]:
    """Reach the globally eligible one-term-deletion fixed point in one run."""
    all_fits = dict(fits)
    current = all_fits[selected_key]
    records: List[Dict[str, object]] = []
    removed_parameters: List[str] = []
    visited: List[str] = []
    final_audit = pd.DataFrame()
    final_boot = pd.DataFrame()
    final_results = pd.DataFrame()
    best_key = selected_key
    one_se_threshold = np.nan
    selected_global = selected_key
    scalar_ok = False

    for round_index in range(1, len(MECH_PARAMS) + len(FAMILIES) + 3):
        if current.spec.key in visited:
            raise RuntimeError(
                f"Scalar-deletion selection cycle detected at {current.spec.key}."
            )
        visited.append(current.spec.key)
        free = list(current.spec.free_params)
        if not free:
            predictions = {key: fit.val_prediction for key, fit in all_fits.items()}
            final_boot = bootstrap_family_scores(
                scorer, predictions, BOOTSTRAP_REPS, bootstrap_seed
            )
            final_results, best_key, one_se_threshold, selected_global = build_results_table(
                all_fits, final_boot, PRACTICAL_EQ_MARGIN
            )
            if selected_global != current.spec.key:
                current = all_fits[selected_global]
                continue
            scalar_ok = True
            final_audit = pd.DataFrame(columns=[
                "round", "current_family", "tested_removed_parameter",
                "deletion_family", "globally_eligible_under_selection_rule",
                "globally_required",
            ])
            break

        deletion_fits: Dict[str, FamilyFit] = {}
        parameter_by_key: Dict[str, str] = {}
        for index, parameter in enumerate(free):
            fit = fit_confirmatory_deletion(
                current,
                parameter,
                all_fits,
                screen_cache,
                train_cache,
                val_cache,
                calib,
                fit_seed + 1000 * round_index + index,
            )
            all_fits[fit.spec.key] = fit
            deletion_fits[parameter] = fit
            parameter_by_key[fit.spec.key] = parameter
            write_csv(fit.search_table, SEARCH_DIR / f"{fit.spec.key}_search.csv")

        predictions = {key: fit.val_prediction for key, fit in all_fits.items()}
        final_boot = bootstrap_family_scores(
            scorer, predictions, BOOTSTRAP_REPS, bootstrap_seed
        )
        final_results, best_key, one_se_threshold, selected_global = build_results_table(
            all_fits, final_boot, PRACTICAL_EQ_MARGIN
        )
        indexed = final_results.set_index("family_key", drop=False)
        round_rows: List[Dict[str, object]] = []
        for parameter, fit in deletion_fits.items():
            row = indexed.loc[fit.spec.key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            pairwise = paired_difference(
                final_boot, fit.spec.key, current.spec.key, PRACTICAL_EQ_MARGIN
            )
            within_one_se = bool(row["Within one standard error of best"])
            equivalent_to_best = bool(row["Practically equivalent to best"])
            globally_eligible = bool(within_one_se and equivalent_to_best)
            record: Dict[str, object] = {
                "round": round_index,
                "current_family": current.spec.key,
                "tested_removed_parameter": parameter,
                "deletion_family": fit.spec.key,
                "deletion_free_parameters": ";".join(fit.spec.free_params),
                "deletion_validation_primary_score": fit.val_metrics.get(
                    "objective_primary_score", fit.val_metrics.get("primary_score", np.nan)
                ),
                "current_validation_primary_score": current.val_metrics.get(
                    "objective_primary_score", current.val_metrics.get("primary_score", np.nan)
                ),
                "bootstrap_mean_primary_score": float(row["Bootstrap mean primary score"]),
                "within_one_standard_error_of_best": within_one_se,
                "difference_to_best_mean": float(row.get("difference_to_best_mean", np.nan)),
                "difference_to_best_ci95_lower": float(row.get("difference_to_best_ci95_lower", np.nan)),
                "difference_to_best_ci95_upper": float(row.get("difference_to_best_ci95_upper", np.nan)),
                "practically_equivalent_to_best": equivalent_to_best,
                "globally_eligible_under_selection_rule": globally_eligible,
                "globally_required": not globally_eligible,
                "parameter_required": not globally_eligible,
                "parameter_deletion_practically_equivalent": pairwise.get("conclusion") == "simpler_practically_equivalent",
                "parameter_deletion_inconclusive": pairwise.get("conclusion") == "inconclusive",
                "pairwise_conclusion_descriptive": pairwise.get("conclusion"),
                "conclusion": "globally_removable" if globally_eligible else "globally_required",
                "decision_rule": "within_one_standard_error_and_practically_equivalent_to_global_best",
                "bootstrap_seed": int(bootstrap_seed),
                "confirmatory_search_completed": True,
            }
            record.update(pairwise)
            record["conclusion"] = (
                "globally_removable" if globally_eligible else "globally_required"
            )
            round_rows.append(record)
            records.append(record)
        final_audit = pd.DataFrame(round_rows)

        eligible = final_audit[
            final_audit["globally_eligible_under_selection_rule"].astype(bool)
        ].copy()
        if not eligible.empty:
            direct_keys = set(eligible["deletion_family"].astype(str))
            if selected_global in direct_keys:
                next_key = selected_global
            else:
                eligible = eligible.sort_values(
                    ["bootstrap_mean_primary_score", "deletion_family"],
                    kind="mergesort",
                )
                next_key = str(eligible.iloc[0]["deletion_family"])
            removed_parameters.append(parameter_by_key[next_key])
            current = all_fits[next_key]
            continue

        if selected_global != current.spec.key:
            selected_fit = all_fits[selected_global]
            if selected_fit.spec.parameter_count > current.spec.parameter_count:
                raise RuntimeError(
                    "Global parsimony selection moved to a more complex family during deletion."
                )
            current = selected_fit
            continue

        scalar_ok = bool(
            len(final_audit) == current.spec.parameter_count
            and final_audit["confirmatory_search_completed"].astype(bool).all()
            and final_audit["globally_required"].astype(bool).all()
            and not final_audit["globally_eligible_under_selection_rule"].astype(bool).any()
        )
        break
    else:
        raise RuntimeError("Scalar-deletion fixed point was not reached.")

    all_records = pd.DataFrame(records)
    pairwise_inconclusive = (
        sorted(
            all_records.loc[
                all_records["parameter_deletion_inconclusive"].astype(bool),
                "tested_removed_parameter",
            ].astype(str).unique().tolist()
        )
        if not all_records.empty else []
    )
    final_required = (
        sorted(final_audit.loc[
            final_audit["globally_required"].astype(bool),
            "tested_removed_parameter",
        ].astype(str).tolist())
        if not final_audit.empty else []
    )
    final_removable = (
        sorted(final_audit.loc[
            final_audit["globally_eligible_under_selection_rule"].astype(bool),
            "tested_removed_parameter",
        ].astype(str).tolist())
        if not final_audit.empty else []
    )
    summary: Dict[str, object] = {
        "backward_deletion_reached_fixed_point": bool(scalar_ok),
        "backward_deletion_stopped": not bool(scalar_ok),
        "removed_parameters": removed_parameters,
        "final_family_key": current.spec.key,
        "final_free_mechanism_parameters": list(current.spec.free_params),
        "final_round_index": int(final_audit["round"].iloc[0]) if not final_audit.empty else len(visited),
        "final_round_required_parameters": final_required,
        "final_round_globally_removable_parameters": final_removable,
        "historical_pairwise_inconclusive_deletions_descriptive": pairwise_inconclusive,
        "scalar_parameter_minimality_confirmed": bool(scalar_ok),
        "decision_rule": "no direct one-term deletion remains within one standard error and practically equivalent to the global best",
        "single_bootstrap_bank_used_for_all_selection_decisions": True,
        "decision_bootstrap_seed": int(bootstrap_seed),
        "confirmatory_deletion_search": {
            "finite_grid_exhaustive_when_combinations_at_most": int(DELETION_EXHAUSTIVE_MAX_COMBINATIONS),
            "full_train_top_k": int(max(FULL_TRAIN_TOP_K, DELETION_FULL_TRAIN_TOP_K)),
            "validation_shortlist_k": int(max(VAL_SHORTLIST_K, DELETION_VAL_SHORTLIST_K)),
            "local_refine_max_evals": int(max(LOCAL_REFINE_MAX_EVALS, DELETION_LOCAL_REFINE_MAX_EVALS)),
            "refine_starts": int(max(DELETION_REFINE_STARTS, 1)),
        },
        "visited_families": visited,
    }
    return (
        current,
        all_fits,
        all_records,
        final_audit,
        summary,
        final_boot,
        final_results,
        best_key,
        float(one_se_threshold),
        selected_global,
    )

def boundary_hits_for_fit(fit: FamilyFit) -> List[Dict[str, object]]:
    hits = []
    for parameter in fit.spec.free_params:
        values = np.asarray(GRID[parameter], dtype=float)
        value = float(fit.selected_params[parameter])
        if values.size <= 1:
            hits.append({"family_key": fit.spec.key, "parameter": parameter, "value": value, "boundary": "single_value_grid"})
        elif np.isclose(value, values.min()):
            hits.append({"family_key": fit.spec.key, "parameter": parameter, "value": value, "boundary": "lower"})
        elif np.isclose(value, values.max()):
            hits.append({"family_key": fit.spec.key, "parameter": parameter, "value": value, "boundary": "upper"})
    return hits



def delta_s_saturation_residual(params: Mapping[str, float]) -> float:
    """Return the largest remaining support-drive increase over Psi in [-1, 1]."""
    minimum_argument = (
        float(params["phi0"])
        + float(params["deltaS"])
        - abs(float(params.get("phiPsi", 0.0)))
    )
    return float(max(0.0, 1.0 - math.tanh(minimum_argument)))


def delta_s_plateau_check(
    fit: FamilyFit,
    val_cache: MetricCache,
    calib: Calibration,
) -> Dict[str, object]:
    """Compare the selected deltaS with the support-channel asymptote."""
    probe_params = dict(fit.selected_params)
    probe_params["deltaS"] = math.inf
    probe_sim = simulate_arrays(val_cache, probe_params, calib)
    probe_metrics = structure_metrics_fast_no_regions(
        val_cache,
        probe_sim,
        f"{fit.spec.key}_deltaS_asymptote",
    )
    probe_metrics.update(objective_diagnostics(probe_metrics))
    probe_metrics["objective_loss"] = float(objective_from_metrics(probe_metrics))

    selected_score = float(primary_objective_score(fit.val_metrics))
    selected_objective = float(fit.val_metrics.get(
        "objective_loss",
        objective_from_metrics(fit.val_metrics),
    ))
    probe_score = float(primary_objective_score(probe_metrics))
    probe_objective = float(probe_metrics["objective_loss"])
    max_next_psi = float(np.max(np.abs(
        probe_sim.pred_next_Psi - fit.val_prediction.pred_next_Psi
    )))
    score_difference = abs(probe_score - selected_score)
    objective_difference = abs(probe_objective - selected_objective)
    confirmed = bool(
        max_next_psi <= DELTA_S_PLATEAU_MAX_NEXT_PSI
        and score_difference <= DELTA_S_PLATEAU_MAX_SCORE_DIFF
        and objective_difference <= DELTA_S_PLATEAU_MAX_OBJECTIVE_DIFF
    )
    return {
        "plateau_probe": "deltaS_to_positive_infinity_support_drive_equals_one",
        "plateau_probe_run": True,
        "plateau_max_abs_next_Psi_difference": max_next_psi,
        "plateau_primary_score_difference": score_difference,
        "plateau_objective_difference": objective_difference,
        "plateau_max_abs_next_Psi_tolerance": DELTA_S_PLATEAU_MAX_NEXT_PSI,
        "plateau_primary_score_tolerance": DELTA_S_PLATEAU_MAX_SCORE_DIFF,
        "plateau_objective_tolerance": DELTA_S_PLATEAU_MAX_OBJECTIVE_DIFF,
        "plateau_confirmed": confirmed,
    }


def boundary_adequacy(
    fits: Dict[str, FamilyFit],
    results: pd.DataFrame,
    final_key: str,
    margin: float,
    val_cache: MetricCache,
    calib: Calibration,
) -> Tuple[bool, pd.DataFrame, List[str], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    blockers: List[str] = []
    next_tests: List[Dict[str, object]] = []
    final_count = fits[final_key].spec.parameter_count
    indexed = results.set_index("family_key", drop=False)
    for key, fit in fits.items():
        hits = boundary_hits_for_fit(fit)
        if not hits:
            continue
        if key in indexed.index:
            result_row = indexed.loc[key]
            if isinstance(result_row, pd.DataFrame):
                result_row = result_row.iloc[0]
            ci_lower = float(result_row.get("difference_to_best_ci95_lower", np.nan))
            clearly_worse = bool(np.isfinite(ci_lower) and ci_lower > margin)
        else:
            clearly_worse = False
        potential_challenger = bool(
            fit.spec.parameter_count <= final_count and not clearly_worse
        )
        for hit in hits:
            plateau = {}
            saturated_delta_s = False
            if hit["parameter"] == "deltaS" and hit["boundary"] == "upper":
                plateau = delta_s_plateau_check(fit, val_cache, calib)
                saturated_delta_s = bool(
                    delta_s_saturation_residual(fit.selected_params) <= DELTA_S_SATURATION_TOL
                    and plateau.get("plateau_confirmed", False)
                )
            blocking = bool(potential_challenger and not saturated_delta_s)
            row = dict(hit)
            row.update({
                "free_mechanism_parameters": fit.spec.parameter_count,
                "clearly_worse_than_best": clearly_worse,
                "potential_minimality_challenger": potential_challenger,
                "deltaS_saturation_residual": (
                    delta_s_saturation_residual(fit.selected_params)
                    if hit["parameter"] == "deltaS" else np.nan
                ),
                "deltaS_saturation_tolerance": DELTA_S_SATURATION_TOL,
                "saturation_plateau": saturated_delta_s,
                "blocking_for_freeze": blocking,
                **plateau,
            })
            rows.append(row)
            if blocking:
                blockers.append(f"{key}:{hit['parameter']}:{hit['boundary']}")
                next_tests.append({
                    "reason": "boundary_grid_hit",
                    "family_key": key,
                    "family_label": fit.spec.label,
                    "parameter": hit["parameter"],
                    "current_value": hit["value"],
                    "boundary": hit["boundary"],
                    "free_mechanism_parameters": fit.spec.parameter_count,
                    "suggested_action": "expand_grid_and_refit_family",
                    "suggested_direction": "lower" if hit["boundary"] == "lower" else "upper" if hit["boundary"] == "upper" else "both",
                    "why": "A lower- or equal-complexity challenger remains unresolved at a non-saturated boundary.",
                })
    return not blockers, pd.DataFrame(rows), blockers, next_tests


def build_synthetic_panel(seed: int = 7, users: int = 80, rows_per_user: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = users * rows_per_user
    uid = np.repeat(np.arange(users, dtype=np.int64), rows_per_user)
    step = np.tile(np.arange(rows_per_user, dtype=np.int64), users)
    M = rng.uniform(-0.9, 0.9, n)
    Psi = rng.uniform(-0.9, 0.9, n)
    E = rng.uniform(0.5, 20.0, n)
    B = rng.uniform(0.5, 20.0, n)
    V = 1.0 - np.exp(-E / 20.0)
    target_M = np.clip(M + rng.normal(0.0, 0.08, n), -1.0, 1.0)
    target_Psi = np.clip(Psi + rng.normal(0.0, 0.05, n), -1.0, 1.0)
    target_E = np.maximum(E + rng.uniform(0.0, 1.0, n), 0.0)
    target_V = 1.0 - np.exp(-target_E / 20.0)
    response_alignable = rng.uniform(0.0, 1.0, n)
    support_alignable = rng.uniform(0.0, 1.0, n)
    response_neutral = rng.uniform(0.0, 0.4, n)
    support_neutral = rng.uniform(0.0, 0.4, n)
    idle = rng.uniform(0.0, 0.6, n)
    return pd.DataFrame({
        "user_id": uid,
        "bundle_step_index": step,
        "M": M,
        "Psi": Psi,
        "V": V,
        "E": E,
        "B": B,
        "G": Psi * B,
        "next_gap_days": rng.uniform(0.0, 3.0, n),
        "answered_count_proxy": rng.uniform(0.0, 5.0, n),
        "response_alignable_interval": response_alignable,
        "support_alignable_interval": support_alignable,
        "response_neutral_interval": response_neutral,
        "support_neutral_interval": support_neutral,
        "active_alignable_interval": response_alignable + support_alignable,
        "active_neutral_interval": response_neutral + support_neutral,
        "idle_mass_interval": idle,
        "target_M_next": target_M,
        "target_Psi_next": target_Psi,
        "target_V_next": target_V,
        "target_E_next": target_E,
        "target_B_next": np.maximum(B + response_alignable + support_alignable + response_neutral + support_neutral + idle, EPS),
        "emp_delta_M_response": rng.normal(0.0, 0.03, n),
        "emp_delta_V_response": rng.normal(0.0, 0.01, n),
        "emp_delta_Psi_active": rng.normal(0.0, 0.03, n),
        "emp_delta_Psi_idle": rng.normal(0.0, 0.02, n),
        "phase_columns_available": True,
    })


def run_self_test() -> None:
    global MIN_DRIFT_BIN_COUNT
    original_min_drift = MIN_DRIFT_BIN_COUNT
    MIN_DRIFT_BIN_COUNT = 2
    try:
        panel = build_synthetic_panel()
        cache = make_metric_cache(panel, "synthetic")
        calib = Calibration(
            eta=20.0,
            tau_response_days=10.0,
            tau_activity_days=10.0,
            residual_mass_per_answer=0.45,
            response_signed_gain=0.8,
            alignment_signed_gain=0.7,
        )
        predictions: Dict[str, PredictionArrays] = {}
        for key in ["persistence", "offset_dual_channel", "full_reference"]:
            spec = FAMILY_BY_KEY[key]
            params = full_params_for_family(spec, PILOT_ANCHORS[-1])
            sim = simulate_for_family(spec, params, cache, calib)
            predictions[key] = light_prediction(sim)
            metrics = structure_metrics_fast_no_regions(cache, sim, key)
            if not np.isfinite(primary_objective_score(metrics)):
                raise RuntimeError(f"Non-finite self-test objective for {key}")
        scorer = BootstrapScorer(cache)
        reference = scorer.score(predictions, reps=8, seed=1234, engine="reference", verify=False)
        optimized = scorer.score(predictions, reps=8, seed=1234, engine="optimized", verify=True)
        merged = reference.merge(
            optimized,
            on=["bootstrap_rep", "family_key"],
            suffixes=("_reference", "_optimized"),
        )
        max_difference = float(np.max(np.abs(
            merged["primary_score_reference"] - merged["primary_score_optimized"]
        )))
        if max_difference > 1e-10:
            raise RuntimeError(f"Bootstrap self-test failed: max difference={max_difference}")
        print(f"Self-test passed. Maximum bootstrap score difference: {max_difference:.3e}")
    finally:
        MIN_DRIFT_BIN_COUNT = original_min_drift



def prepare_output_root(path: Path, *, resume: bool, overwrite: bool) -> None:
    """Prepare an empty or resumable output directory."""
    if resume and overwrite:
        raise ValueError("Use only one of --resume or --overwrite.")
    root = path.resolve()
    if root.exists() and any(root.iterdir()):
        if overwrite:
            shutil.rmtree(root)
        elif not resume:
            raise FileExistsError(
                f"Output directory is not empty: {root}. Use --resume or --overwrite explicitly."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone EdNet-KT4 seven-term mechanism-family ablation and bounded-minimality experiment."
    )
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--random-state", type=int, default=CONFIG_RANDOM_STATE)
    parser.add_argument(
        "--grid-profile",
        choices=("publication", "compact"),
        default=GRID_PROFILE,
        help="Use the exact reported grid or the reduced rerun grid.",
    )
    parser.add_argument("--screening-train-users", type=int, default=SCREENING_TRAIN_USERS)
    parser.add_argument("--screening-max-candidates", type=int, default=SCREENING_MAX_CANDIDATES_PER_FAMILY)
    parser.add_argument("--full-train-top-k", type=int, default=FULL_TRAIN_TOP_K)
    parser.add_argument("--val-shortlist-k", type=int, default=VAL_SHORTLIST_K)
    parser.add_argument("--local-refine-max-evals", type=int, default=LOCAL_REFINE_MAX_EVALS)
    parser.add_argument(
        "--deletion-exhaustive-max-combinations",
        type=int,
        default=DELETION_EXHAUSTIVE_MAX_COMBINATIONS,
    )
    parser.add_argument("--deletion-full-train-top-k", type=int, default=DELETION_FULL_TRAIN_TOP_K)
    parser.add_argument("--deletion-val-shortlist-k", type=int, default=DELETION_VAL_SHORTLIST_K)
    parser.add_argument(
        "--deletion-local-refine-max-evals",
        type=int,
        default=DELETION_LOCAL_REFINE_MAX_EVALS,
    )
    parser.add_argument("--deletion-refine-starts", type=int, default=DELETION_REFINE_STARTS)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--equivalence-margin", type=float, default=PRACTICAL_EQ_MARGIN)
    parser.add_argument("--margin-sensitivity", type=str, default=",".join(str(x) for x in MARGIN_SENSITIVITY_VALUES))
    parser.add_argument("--bootstrap-engine", choices=["optimized", "reference"], default=BOOTSTRAP_ENGINE)
    parser.add_argument("--verify-optimized-bootstrap", action=argparse.BooleanOptionalAction, default=VERIFY_OPTIMIZED_BOOTSTRAP)
    parser.add_argument("--verify-bootstrap-reps", type=int, default=VERIFY_BOOTSTRAP_REPS)
    parser.add_argument("--numba-threads", type=int, default=int(os.environ.get("MECH_MINIMALITY_NUMBA_THREADS", "0")))
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-users-per-split", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {PUBLIC_RELEASE_VERSION}")
    parser.add_argument(
        "--allow-missing-kmeans-contract",
        action="store_true",
        help="Allow panel-only archival inputs without Stage-1 fixed-K metadata.",
    )
    parser.add_argument("--write-selected-predictions", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def configure_from_args(args: argparse.Namespace, *, create_output_dirs: bool = True) -> None:
    global CONFIG_RANDOM_STATE, CONFIG_USE_NUMBA, GRID_PROFILE, GRID
    global SCREENING_TRAIN_USERS, SCREENING_MAX_CANDIDATES_PER_FAMILY
    global FULL_TRAIN_TOP_K, VAL_SHORTLIST_K, LOCAL_REFINE_MAX_EVALS
    global DELETION_EXHAUSTIVE_MAX_COMBINATIONS, DELETION_FULL_TRAIN_TOP_K
    global DELETION_VAL_SHORTLIST_K, DELETION_LOCAL_REFINE_MAX_EVALS
    global DELETION_REFINE_STARTS
    global BOOTSTRAP_REPS, PRACTICAL_EQ_MARGIN, MARGIN_SENSITIVITY_VALUES
    global BOOTSTRAP_ENGINE, VERIFY_OPTIMIZED_BOOTSTRAP, VERIFY_BOOTSTRAP_REPS
    global OUTPUT_ROOT, TABLE_DIR, META_DIR, SEARCH_DIR, CHECKPOINT_DIR

    CONFIG_RANDOM_STATE = int(args.random_state)
    GRID_PROFILE = str(args.grid_profile)
    source_grid = PUBLICATION_GRID if GRID_PROFILE == "publication" else COMPACT_GRID
    GRID = {name: list(values) for name, values in source_grid.items()}
    CONFIG_USE_NUMBA = bool(NUMBA_AVAILABLE and not args.no_numba)
    SCREENING_TRAIN_USERS = int(args.screening_train_users)
    SCREENING_MAX_CANDIDATES_PER_FAMILY = int(args.screening_max_candidates)
    FULL_TRAIN_TOP_K = int(args.full_train_top_k)
    VAL_SHORTLIST_K = int(args.val_shortlist_k)
    LOCAL_REFINE_MAX_EVALS = int(args.local_refine_max_evals)
    DELETION_EXHAUSTIVE_MAX_COMBINATIONS = int(args.deletion_exhaustive_max_combinations)
    DELETION_FULL_TRAIN_TOP_K = int(args.deletion_full_train_top_k)
    DELETION_VAL_SHORTLIST_K = int(args.deletion_val_shortlist_k)
    DELETION_LOCAL_REFINE_MAX_EVALS = int(args.deletion_local_refine_max_evals)
    DELETION_REFINE_STARTS = int(args.deletion_refine_starts)
    BOOTSTRAP_REPS = int(args.bootstrap_reps)
    PRACTICAL_EQ_MARGIN = float(args.equivalence_margin)
    MARGIN_SENSITIVITY_VALUES = [float(x) for x in args.margin_sensitivity.split(",") if x.strip()]
    BOOTSTRAP_ENGINE = str(args.bootstrap_engine)
    VERIFY_OPTIMIZED_BOOTSTRAP = bool(args.verify_optimized_bootstrap)
    VERIFY_BOOTSTRAP_REPS = int(args.verify_bootstrap_reps)
    if SCREENING_MAX_CANDIDATES_PER_FAMILY < 1:
        raise ValueError("--screening-max-candidates must be at least 1.")
    if FULL_TRAIN_TOP_K < 1 or VAL_SHORTLIST_K < 1:
        raise ValueError("--full-train-top-k and --val-shortlist-k must be at least 1.")
    if LOCAL_REFINE_MAX_EVALS < 1 or BOOTSTRAP_REPS < 1:
        raise ValueError("--local-refine-max-evals and --bootstrap-reps must be at least 1.")
    if (
        DELETION_EXHAUSTIVE_MAX_COMBINATIONS < 1
        or DELETION_FULL_TRAIN_TOP_K < 1
        or DELETION_VAL_SHORTLIST_K < 1
        or DELETION_LOCAL_REFINE_MAX_EVALS < 1
        or DELETION_REFINE_STARTS < 1
    ):
        raise ValueError("Deletion-audit search settings must be positive.")
    if PRACTICAL_EQ_MARGIN < 0:
        raise ValueError("--equivalence-margin must be non-negative.")

    OUTPUT_ROOT = args.output_root.resolve()
    TABLE_DIR = OUTPUT_ROOT / "tables"
    META_DIR = OUTPUT_ROOT / "metadata"
    SEARCH_DIR = TABLE_DIR / "family_searches"
    CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
    if create_output_dirs:
        for path in [OUTPUT_ROOT, TABLE_DIR, META_DIR, SEARCH_DIR, CHECKPOINT_DIR]:
            path.mkdir(parents=True, exist_ok=True)

    if args.numba_threads > 0 and NUMBA_AVAILABLE and set_num_threads is not None:
        set_num_threads(int(args.numba_threads))


def main() -> None:
    args = parse_args()
    if not args.self_test:
        prepare_output_root(args.output_root, resume=args.resume, overwrite=args.overwrite)
    configure_from_args(args, create_output_dirs=not args.self_test)
    grid_contract = validate_grid_contract()
    if args.self_test:
        run_self_test()
        return

    start = time.time()
    script_path = Path(__file__).resolve()
    print(f"[minimality] started {now_string()}", flush=True)
    print("[minimality] mechanism engine: built-in generic seven-term supermodel", flush=True)
    print(f"[minimality] Stage-1 root: {args.stage1_root.resolve()}", flush=True)
    print(f"[minimality] bootstrap engine: {BOOTSTRAP_ENGINE}", flush=True)
    print(f"[minimality] grid profile: {GRID_PROFILE}", flush=True)
    if GRID_PROFILE == "compact":
        print(
            "[minimality] compact grid changes the search domain; regenerate all reported family results before manuscript use.",
            flush=True,
        )

    with timed_stage("load_and_prepare_panels"):
        kmeans_contract = audit_stage1_kmeans_contract(
            args.stage1_root.resolve(),
            allow_missing=args.allow_missing_kmeans_contract,
        )
        train, val, eta, load_manifest = load_phase1_panels(args.stage1_root.resolve())
        if args.smoke_test:
            train = sample_users(train, args.smoke_users_per_split, CONFIG_RANDOM_STATE + 1)
            val = sample_users(val, args.smoke_users_per_split, CONFIG_RANDOM_STATE + 2)
            print(f"[minimality] smoke panels: train={len(train)}, val={len(val)}", flush=True)
        if train.empty or val.empty:
            raise RuntimeError("A_train or A_val is empty after Phase-1 validity filtering.")
        panel_contract = development_panel_contract(train, val)
        calib = calibrate_from_A_train(train, eta, TAU_RESPONSE_DAYS, TAU_ACTIVITY_DAYS)
        screen_train = sample_users(train, SCREENING_TRAIN_USERS, CONFIG_RANDOM_STATE + 11)
        train_cache = make_metric_cache(train, "A_train_full")
        val_cache = make_metric_cache(val, "A_val_full")
        screen_cache = make_metric_cache(screen_train, "A_train_screen")
        scorer = BootstrapScorer(val_cache)
        del train, val, screen_train
        gc.collect()

    script_sha256 = file_sha256(script_path)
    run_contract = {
        "public_release_version": PUBLIC_RELEASE_VERSION,
        "script_sha256": script_sha256,
        "standalone_generic_seven_term_engine": True,
        "external_mechanism_script_required": False,
        "stage1_root": str(args.stage1_root.resolve()),
        "stage1_fixed_k6_contract": kmeans_contract,
        "development_panel_contract": panel_contract,
        "smoke_test": bool(args.smoke_test),
        "smoke_users_per_split": int(args.smoke_users_per_split),
        "bootstrap_engine": BOOTSTRAP_ENGINE,
        "parameter_search_weighting": {
            "one_step_term": "interval-weighted",
            "occupancy_and_drift_terms": "user-balanced",
        },
        "family_bootstrap_weighting": "paired user-balanced",
        "random_state": CONFIG_RANDOM_STATE,
        "screening_train_users": SCREENING_TRAIN_USERS,
        "screening_max_candidates_per_family": SCREENING_MAX_CANDIDATES_PER_FAMILY,
        "full_train_top_k": FULL_TRAIN_TOP_K,
        "val_shortlist_k": VAL_SHORTLIST_K,
        "local_refine_max_evals": LOCAL_REFINE_MAX_EVALS,
        "confirmatory_deletion_search": {
            "exhaustive_max_combinations": DELETION_EXHAUSTIVE_MAX_COMBINATIONS,
            "full_train_top_k": DELETION_FULL_TRAIN_TOP_K,
            "validation_shortlist_k": DELETION_VAL_SHORTLIST_K,
            "local_refine_max_evals": DELETION_LOCAL_REFINE_MAX_EVALS,
            "refine_starts": DELETION_REFINE_STARTS,
        },
        "bootstrap_reps": BOOTSTRAP_REPS,
        "decision_bootstrap_seed": CONFIG_RANDOM_STATE + DECISION_BOOTSTRAP_SEED_OFFSET,
        "single_bootstrap_bank_for_all_selection_decisions": True,
        "scalar_minimality_rule": "no direct one-term deletion remains within one standard error and practically equivalent to the global best",
        "practical_equivalence_margin": PRACTICAL_EQ_MARGIN,
        "fixed_nuisance": FIXED_NUISANCE,
        "grid_profile": GRID_PROFILE,
        "grid_contract": grid_contract,
        "grids": GRID,
        "deltaS_saturation_tolerance": DELTA_S_SATURATION_TOL,
        "deltaS_plateau_probe": "positive_infinity_support_drive_asymptote",
        "deltaS_plateau_max_next_Psi": DELTA_S_PLATEAU_MAX_NEXT_PSI,
        "deltaS_plateau_max_score_difference": DELTA_S_PLATEAU_MAX_SCORE_DIFF,
        "deltaS_plateau_max_objective_difference": DELTA_S_PLATEAU_MAX_OBJECTIVE_DIFF,
        "anchors": PILOT_ANCHORS,
        "families": [asdict(family) for family in FAMILIES],
        "calibration": asdict(calib),
    }
    run_hash = stable_json_hash(run_contract)

    fits: Dict[str, FamilyFit] = {}
    with timed_stage("fit_prespecified_families"):
        iterator: Iterable[FamilySpec] = FAMILIES
        if tqdm is not None:
            iterator = tqdm(FAMILIES, desc="Fitting model families", unit="family")
        for index, spec in enumerate(iterator):
            fit = fit_or_resume_family(
                spec,
                screen_cache,
                train_cache,
                val_cache,
                calib,
                CONFIG_RANDOM_STATE + 100 * (index + 1),
                run_hash,
                args.resume,
            )
            fits[spec.key] = fit
            print(
                f"[minimality] {spec.label}: "
                f"val primary={fit.val_metrics.get('objective_primary_score', np.nan):.6f}",
                flush=True,
            )

    decision_bootstrap_seed = CONFIG_RANDOM_STATE + DECISION_BOOTSTRAP_SEED_OFFSET
    with timed_stage("initial_paired_user_bootstrap"):
        initial_predictions = {key: fit.val_prediction for key, fit in fits.items()}
        boot = bootstrap_family_scores(
            scorer, initial_predictions, BOOTSTRAP_REPS, decision_bootstrap_seed
        )
        results, best_key, one_se_threshold, selected_key = build_results_table(
            fits, boot, PRACTICAL_EQ_MARGIN
        )

    comparisons = [
        ("persistence", "response_only", "response signed block"),
        ("persistence", "alignment_only", "alignment signed block"),
        ("response_only", "core_two_parameter", "add alignment baseline"),
        ("alignment_only", "core_two_parameter", "add response restoring"),
        ("core_two_parameter", "response_offset_core", "add response offset"),
        ("core_two_parameter", "dual_channel_core", "add dual-channel contrast"),
        ("response_offset_core", "offset_dual_channel", "add dual-channel contrast to response-offset core"),
        ("dual_channel_core", "offset_dual_channel", "add response offset to dual-channel core"),
        ("dual_channel_core", "dual_plus_linear_coupling", "add linear cross-coordinate coupling"),
        ("dual_channel_core", "dual_plus_interaction", "add M-by-alignment interaction"),
        ("offset_dual_channel", "full_reference", "full seven-term reference extension"),
    ]
    with timed_stage("single_run_global_scalar_minimality"):
        (
            final_fit,
            fits,
            deletion_df,
            global_scalar_df,
            deletion_summary,
            boot,
            results,
            best_key,
            one_se_threshold,
            selected_after_deletion_key,
        ) = resolve_scalar_minimality_one_run(
            fits,
            selected_key,
            screen_cache,
            train_cache,
            val_cache,
            calib,
            scorer,
            CONFIG_RANDOM_STATE + 4040,
            decision_bootstrap_seed,
        )
        final_key = final_fit.spec.key
        scalar_ok = bool(
            deletion_summary.get("scalar_parameter_minimality_confirmed", False)
        )

    contrast_rows = []
    for simpler, richer, label in comparisons:
        if simpler in fits and richer in fits:
            row = paired_difference(boot, simpler, richer, PRACTICAL_EQ_MARGIN)
            row.update({
                "comparison": label,
                "simpler_label": fits[simpler].spec.label,
                "richer_label": fits[richer].spec.label,
            })
            contrast_rows.append(row)
    contrasts = pd.DataFrame(contrast_rows)

    search_ok, boundary_df, boundary_blockers, boundary_next_tests = boundary_adequacy(
        fits, results, final_key, PRACTICAL_EQ_MARGIN, val_cache, calib
    )
    final_validation = final_model_validation(results, final_key, best_key)
    final_equiv_ok = bool(
        final_validation.get("within_one_standard_error", False)
        and final_validation.get("practically_equivalent_to_best", False)
    )
    final_selected_by_parsimony = final_key == selected_after_deletion_key
    deletion_summary["scalar_parameter_minimality_confirmed_global_safe"] = scalar_ok
    deletion_summary["global_scalar_deletion_audit_rows"] = int(len(global_scalar_df))
    deletion_summary["global_scalar_reaudit_required"] = False
    deletion_summary["global_scalar_reaudit_completed"] = True
    persistence_rows = results.loc[
        results.family_key == "persistence", "Practically equivalent to best"
    ]
    baselines_beaten = bool(persistence_rows.empty or not bool(persistence_rows.iloc[0]))

    next_required_tests: List[Dict[str, object]] = list(boundary_next_tests)
    if not scalar_ok:
        next_required_tests.append({
            "reason": "scalar_minimality_fixed_point_not_confirmed",
            "family_key": final_key,
            "suggested_action": "inspect_single_run_global_scalar_deletion_audit",
            "why": "At least one direct one-term deletion remains globally eligible or the final audit is incomplete.",
        })
    if not final_equiv_ok:
        next_required_tests.append({
            "reason": "final_model_not_practically_equivalent_to_best",
            "family_key": final_key,
            "best_family_key": best_key,
            "suggested_action": "rerun_unified_minimality_with_expanded_grid_or_select_best_equivalent_family",
            "why": "The scalar-deleted final model must be validated against the best family before freezing.",
            "final_model_validation_against_best": final_validation,
        })
    if not baselines_beaten:
        next_required_tests.append({
            "reason": "baseline_practically_equivalent_to_best",
            "family_key": "persistence",
            "suggested_action": "do_not_freeze_mechanism_or_reassess_primary_objective",
            "why": "A zero-signed-mechanism baseline remains practically equivalent to the best family.",
        })

    ready = bool(
        search_ok
        and scalar_ok
        and baselines_beaten
        and final_equiv_ok
        and final_selected_by_parsimony
    )
    results["Final scalar-minimal family"] = results["family_key"] == final_key

    margin_rows = []
    for margin in MARGIN_SENSITIVITY_VALUES:
        differences = pd.DataFrame([
            difference_to_best(boot, key, best_key, margin)
            for key in results.family_key.tolist()
        ])
        table = results[[
            "family_key", "Model family", "Free mechanism parameters", "Bootstrap mean primary score"
        ]].merge(differences, on="family_key", how="left")
        table["within_one_se"] = table["Bootstrap mean primary score"] <= one_se_threshold
        eligible = table[table["within_one_se"] & table["practically_equivalent_to_best"]]
        selected = best_key if eligible.empty else str(
            eligible.sort_values(
                ["Free mechanism parameters", "Bootstrap mean primary score"],
                kind="mergesort",
            ).iloc[0]["family_key"]
        )
        label = fits[selected].spec.label if selected in fits else selected
        margin_rows.append({
            "equivalence_margin": margin,
            "selected_family_key": selected,
            "selected_family_label": label,
        })
    margin_df = pd.DataFrame(margin_rows)

    manuscript_columns = [
        "family_key", "Model family", "Free mechanism parameters", "Free parameter names",
        "Bootstrap mean primary score", "Bootstrap 95% CI lower", "Bootstrap 95% CI upper",
        "Landscape divergence", "Local drift discrepancy", "Drift-direction discrepancy", "Drift-speed discrepancy",
        "Within one standard error of best", "Practically equivalent to best",
        "Parsimonious family selected", "Final scalar-minimal family",
    ]
    manuscript = results[manuscript_columns].copy()

    deletion_columns = [
        "round", "current_family", "tested_removed_parameter", "deletion_family",
        "deletion_free_parameters", "deletion_validation_primary_score",
        "current_validation_primary_score", "bootstrap_mean_primary_score",
        "within_one_standard_error_of_best", "difference_to_best_mean",
        "difference_to_best_ci95_lower", "difference_to_best_ci95_upper",
        "practically_equivalent_to_best",
        "globally_eligible_under_selection_rule", "globally_required",
        "parameter_required", "parameter_deletion_practically_equivalent",
        "parameter_deletion_inconclusive", "pairwise_conclusion_descriptive",
        "simpler_family", "richer_family", "practical_equivalence_margin",
        "paired_difference_mean", "paired_difference_median",
        "paired_difference_ci95_lower", "paired_difference_ci95_upper",
        "probability_richer_improves", "probability_improvement_exceeds_margin",
        "decision_rule", "bootstrap_seed", "confirmatory_search_completed",
        "conclusion",
    ]
    deletion_df = deletion_df.reindex(columns=deletion_columns)
    boundary_columns = [
        "family_key", "parameter", "value", "boundary",
        "free_mechanism_parameters", "clearly_worse_than_best",
        "potential_minimality_challenger", "deltaS_saturation_residual",
        "deltaS_saturation_tolerance", "saturation_plateau",
        "blocking_for_freeze", "plateau_probe", "plateau_probe_run",
        "plateau_max_abs_next_Psi_difference", "plateau_primary_score_difference",
        "plateau_objective_difference", "plateau_max_abs_next_Psi_tolerance",
        "plateau_primary_score_tolerance", "plateau_objective_tolerance",
        "plateau_confirmed",
    ]
    boundary_df = boundary_df.reindex(columns=boundary_columns)

    with timed_stage("write_outputs"):
        write_csv(results, TABLE_DIR / "model_family_results.csv")
        write_csv(contrasts, TABLE_DIR / "nested_mechanism_contrasts.csv")
        write_csv(deletion_df, TABLE_DIR / "selected_model_parameter_deletions.csv")
        write_csv(global_scalar_df, TABLE_DIR / "global_scalar_deletion_audit.csv")
        write_csv(boundary_df, TABLE_DIR / "parameter_grid_boundaries.csv")
        write_csv(margin_df, TABLE_DIR / "equivalence_margin_sensitivity.csv")
        write_csv(manuscript, TABLE_DIR / "manuscript_results_summary.csv")
        write_csv(boot, TABLE_DIR / "model_family_bootstrap_scores.csv.gz")
        verification_df = pd.DataFrame(_BOOTSTRAP_VERIFICATION_ROWS)
        if not verification_df.empty:
            write_csv(verification_df, TABLE_DIR / "optimized_bootstrap_equivalence_checks.csv")

        next_required_df = pd.DataFrame(next_required_tests)
        if next_required_df.empty:
            next_required_df = pd.DataFrame(columns=[
                "reason", "family_key", "parameter", "suggested_action", "why"
            ])
        write_csv(next_required_df, TABLE_DIR / "next_required_tests.csv")

        scalar_audit_payload = {
            "family_key": final_key,
            "tested_parameters": sorted(
                global_scalar_df["tested_removed_parameter"].astype(str).tolist()
            ) if not global_scalar_df.empty else [],
            "required_parameters": sorted(
                global_scalar_df.loc[
                    global_scalar_df["globally_required"].astype(bool),
                    "tested_removed_parameter",
                ].astype(str).tolist()
            ) if not global_scalar_df.empty else [],
            "globally_removable_deletions": sorted(
                global_scalar_df.loc[
                    global_scalar_df["globally_eligible_under_selection_rule"].astype(bool),
                    "tested_removed_parameter",
                ].astype(str).tolist()
            ) if not global_scalar_df.empty else [],
            "pairwise_inconclusive_deletions_descriptive": sorted(
                global_scalar_df.loc[
                    global_scalar_df["parameter_deletion_inconclusive"].astype(bool),
                    "tested_removed_parameter",
                ].astype(str).tolist()
            ) if not global_scalar_df.empty else [],
            "untested_parameters": sorted(
                set(final_fit.spec.free_params)
                - set(global_scalar_df["tested_removed_parameter"].astype(str).tolist())
            ) if not global_scalar_df.empty else list(final_fit.spec.free_params),
            "decision_rule": "no direct one-term deletion remains within one standard error and practically equivalent to the global best",
            "single_bootstrap_bank": True,
            "bootstrap_seed": int(decision_bootstrap_seed),
            "confirmed": bool(scalar_ok),
        }

        selected_payload = {
            "selected_family_key_before_scalar_deletion": selected_key,
            "selected_family_label_before_scalar_deletion": fits[selected_key].spec.label,
            "final_family_key": final_key,
            "final_family_label": final_fit.spec.label,
            "final_free_mechanism_parameters": list(final_fit.spec.free_params),
            "final_selected_parameters": final_fit.selected_params,
            "fixed_nuisance_scales": FIXED_NUISANCE,
            "grid_profile": GRID_PROFILE,
            "paper_number_reproduction_profile": GRID_PROFILE == "publication",
            "parameter_interpretation": {
                "deltaS": "finite representative of a saturated support-channel contrast; values beyond the verified plateau are not interpreted numerically"
            },
            "ready_for_phase2_freeze": ready,
            "scalar_parameter_minimality_confirmed": scalar_ok,
            "scalar_parameter_minimality_scope": "family-bounded under the global one-SE plus practical-equivalence selection rule",
            "pairwise_parameter_necessity_not_claimed": True,
            "scalar_parameter_minimality_audit": scalar_audit_payload,
            "global_scalar_deletion_audit": global_scalar_df.to_dict(orient="records"),
            "single_run_minimality_resolution": True,
            "single_bootstrap_bank_used_for_all_selection_decisions": True,
            "decision_bootstrap_seed": int(decision_bootstrap_seed),
            "search_adequacy_confirmed": search_ok,
            "baseline_not_practically_equivalent_to_best": baselines_beaten,
            "boundary_blockers": boundary_blockers,
            "deletion_summary": deletion_summary,
            "final_model_validation_against_best": final_validation,
            "final_model_selected_by_parsimony_rule": final_selected_by_parsimony,
            "selected_family_after_final_retest": selected_after_deletion_key,
            "next_required_tests": next_required_tests,
        }
        expected_ready = bool(
            search_ok
            and scalar_ok
            and baselines_beaten
            and final_equiv_ok
            and final_selected_by_parsimony
        )
        if ready != expected_ready:
            raise AssertionError("Readiness gate is internally inconsistent.")
        save_json(selected_payload, META_DIR / "phase1_minimal_mechanism_handoff.json")

        if args.write_selected_predictions:
            prediction = final_fit.val_prediction
            write_table(pd.DataFrame({
                "user_id": val_cache.uid,
                "bundle_step_index": val_cache.steps,
                "M": val_cache.M,
                "Psi": val_cache.Psi,
                "target_M_next": val_cache.target_M_next,
                "target_Psi_next": val_cache.target_Psi_next,
                "pred_next_M": prediction.pred_next_M,
                "pred_next_Psi": prediction.pred_next_Psi,
            }), TABLE_DIR / "selected_family_A_val_predictions")

        manifest = {
            "created_at": now_string(),
            "public_release_version": PUBLIC_RELEASE_VERSION,
            "script": str(script_path),
            "script_sha256": script_sha256,
            "experiment_scope": "standalone Phase-1 family-bounded minimality on A_train/A_val; B_confirm not read",
            "standalone_generic_seven_term_engine": True,
            "external_seven_parameter_script_required": False,
            "external_post_ablation_script_required": False,
            "full_reference_definition": list(FAMILY_BY_KEY["full_reference"].free_params),
            "primary_macrostate": ["M", "Psi"],
            "auxiliary_accounting": ["E", "B", "G", "V"],
            "kmeans_used_in_minimality_selection": False,
            "stage1_mesostate_contract": "fixed K=6; post-selection kinetics only",
            "deltaS_saturation_tolerance": DELTA_S_SATURATION_TOL,
            "deltaS_plateau_probe": "positive_infinity_support_drive_asymptote",
            "deltaS_plateau_max_next_Psi": DELTA_S_PLATEAU_MAX_NEXT_PSI,
            "deltaS_plateau_max_score_difference": DELTA_S_PLATEAU_MAX_SCORE_DIFF,
            "deltaS_plateau_max_objective_difference": DELTA_S_PLATEAU_MAX_OBJECTIVE_DIFF,
            "deltaS_parameter_interpretation": "finite representative of a saturated support-channel contrast; only plateau membership is interpreted",
            "grid_profile": GRID_PROFILE,
            "grid_contract": grid_contract,
            "paper_number_reproduction_profile": "publication",
            "compact_profile_requires_full_data_regeneration": GRID_PROFILE == "compact",
            "candidate_prediction_cache_enabled": False,
            "prediction_retention_policy": "retain only selected A_val M/Psi predictions for bootstrap and handoff outputs",
            "stage1_root": str(args.stage1_root.resolve()),
            "stage1_fixed_k6_contract": kmeans_contract,
            "development_panel_contract": panel_contract,
            "output_root": str(OUTPUT_ROOT),
            "smoke_test": bool(args.smoke_test),
            "n_train_rows": int(panel_contract["A_train"]["rows"]),
            "n_val_rows": int(panel_contract["A_val"]["rows"]),
            "n_train_users": int(panel_contract["A_train"]["users"]),
            "n_val_users": int(panel_contract["A_val"]["users"]),
            "bootstrap_engine": BOOTSTRAP_ENGINE,
            "single_run_minimality_resolution": True,
            "single_bootstrap_bank_used_for_all_selection_decisions": True,
            "decision_bootstrap_seed": int(decision_bootstrap_seed),
            "scalar_minimality_rule": "no direct one-term deletion remains within one standard error and practically equivalent to the global best",
            "scalar_minimality_scope": "family-bounded; pairwise statistical necessity is not claimed",
            "confirmatory_deletion_search": deletion_summary.get("confirmatory_deletion_search", {}),
            "parameter_search_weighting": {
                "one_step_term": "interval-weighted",
                "occupancy_and_drift_terms": "user-balanced",
            },
            "family_bootstrap_weighting": {
                "one_step_occupancy_and_drift_terms": "paired user-balanced",
                "support_grid": "fixed full-A_val support mask",
            },
            "optimized_bootstrap_definition": "paired user multiplicities applied to pre-aggregated user-by-grid sufficient statistics",
            "bootstrap_support_grid_policy": "fixed full-A_val support mask, matching the reported implementation",
            "optimized_bootstrap_verified": bool(_BOOTSTRAP_VERIFICATION_ROWS) if BOOTSTRAP_ENGINE == "optimized" else None,
            "max_optimized_bootstrap_absolute_difference": float(max(
                [row["absolute_difference"] for row in _BOOTSTRAP_VERIFICATION_ROWS],
                default=0.0,
            )),
            "numba_available": NUMBA_AVAILABLE,
            "numba_enabled": CONFIG_USE_NUMBA,
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "random_state": CONFIG_RANDOM_STATE,
            "bootstrap_reps": BOOTSTRAP_REPS,
            "practical_equivalence_margin": PRACTICAL_EQ_MARGIN,
            "fixed_nuisance_scales": FIXED_NUISANCE,
            "families": [asdict(family) for family in FAMILIES],
            "search_grid": GRID,
            "pilot_anchors": PILOT_ANCHORS,
            "objective_weights": OBJECTIVE_WEIGHTS,
            "objective_sanity_limits": OBJECTIVE_SANITY_LIMITS,
            "readiness_rule": "final_model_PE_to_best_and_within_1SE + scalar_minimality_at_final_fixed_point + search_adequacy_for_potential_challengers + baseline_not_PE",
            "load_manifest": load_manifest,
            "calibration": asdict(calib),
            "run_contract_sha256": run_hash,
            "runtime_profile": RUNTIME_PROFILE,
        }
        save_json(manifest, META_DIR / "minimality_experiment_manifest.json")
        save_json({
            "elapsed_seconds": time.time() - start,
            "stages": RUNTIME_PROFILE,
        }, META_DIR / "runtime_profile.json")

    print("[minimality] completed", flush=True)
    print(f"[minimality] final family: {final_key}", flush=True)
    print(f"[minimality] ready_for_phase2_freeze={ready}", flush=True)
    print(f"[minimality] outputs: {OUTPUT_ROOT}", flush=True)
    print(f"[minimality] elapsed seconds: {time.time() - start:.1f}", flush=True)


if __name__ == "__main__":
    main()
