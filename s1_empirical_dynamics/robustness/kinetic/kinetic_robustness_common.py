#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg.lapack import get_lapack_funcs

try:
    from numba import njit
    NUMBA_IMPORT_ERROR = None
except Exception as exc:
    njit = None
    NUMBA_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

NUMBA_ENABLED = njit is not None and os.environ.get("KINETIC_ROBUSTNESS_USE_NUMBA", "1") == "1"
EPS = 1e-12
_BACKEND_AUDIT: Dict[str, Any] = {
    "numba_requested": bool(os.environ.get("KINETIC_ROBUSTNESS_USE_NUMBA", "1") == "1"),
    "numba_import_available": bool(njit is not None),
    "numba_import_error": NUMBA_IMPORT_ERROR,
    "recursive_labels_backend": None,
    "run_histograms_backend": None,
    "numba_runtime_error": None,
}


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
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
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_module(path: Path, module_name: str) -> Any:
    source = path.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Required Python module not found: {source}")
    specification = importlib.util.spec_from_file_location(module_name, str(source))
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not import module: {source}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def table_path(base_or_path: Path) -> Path:
    source = Path(base_or_path)
    if source.exists() and source.is_file():
        return source
    for suffix in (".parquet", ".csv.gz", ".csv"):
        candidate = source.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find table for {source}.[parquet|csv.gz|csv]")


def read_table(base_or_path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    path = table_path(base_or_path)
    selected = list(columns) if columns is not None else None
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=selected)
    return pd.read_csv(path, usecols=selected, low_memory=False)


def write_table(frame: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = base.with_suffix(".parquet")
        frame.to_parquet(path, index=False)
        return path
    except Exception:
        path = base.with_suffix(".csv.gz")
        frame.to_csv(path, index=False, compression="gzip")
        return path


def frame_identifier_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame[list(columns)].copy()
    for column in columns:
        selected[column] = pd.to_numeric(selected[column], errors="raise").astype(np.int64)
    values = selected.to_numpy(dtype=np.int64, copy=True)
    return hashlib.sha256(values.tobytes()).hexdigest()


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite = np.where(np.isfinite(values))[0]
    if finite.size == 0:
        return adjusted
    order = finite[np.argsort(values[finite])]
    ranked = values[order]
    corrected = ranked * finite.size / np.arange(1, finite.size + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    adjusted[order] = np.clip(corrected, 0.0, 1.0)
    return adjusted


def normal_survival(z_value: float) -> float:
    if not np.isfinite(z_value):
        return float("nan")
    return float(0.5 * math.erfc(float(z_value) / math.sqrt(2.0)))


def normalize_transition(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    row_totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, row_totals, out=np.zeros_like(values), where=row_totals > 0)


def transition_counts_from_assignments(assignments: pd.DataFrame, k: int) -> np.ndarray:
    data = assignments[["user_id", "bundle_step_index", "macrostate"]].copy()
    data["user_id"] = pd.to_numeric(data["user_id"], errors="coerce")
    data["bundle_step_index"] = pd.to_numeric(data["bundle_step_index"], errors="coerce")
    data["macrostate"] = pd.to_numeric(data["macrostate"], errors="coerce")
    data = data.dropna(subset=["user_id", "bundle_step_index"]).sort_values(
        ["user_id", "bundle_step_index"], kind="mergesort"
    )
    users = data["user_id"].to_numpy(dtype=np.int64)
    steps = data["bundle_step_index"].to_numpy(dtype=np.int64)
    states = data["macrostate"].to_numpy(dtype=float)
    counts = np.zeros((int(k), int(k)), dtype=float)
    if len(data) < 2:
        return counts
    adjacent = (
        (users[1:] == users[:-1])
        & (steps[1:] == steps[:-1] + 1)
        & np.isfinite(states[:-1])
        & np.isfinite(states[1:])
    )
    current = states[:-1][adjacent].astype(np.int64)
    next_state = states[1:][adjacent].astype(np.int64)
    valid = (
        (current >= 0)
        & (current < int(k))
        & (next_state >= 0)
        & (next_state < int(k))
    )
    if np.any(valid):
        encoded = current[valid] * int(k) + next_state[valid]
        counts += np.bincount(encoded, minlength=int(k) * int(k)).reshape(int(k), int(k))
    return counts


def transition_rows(assignments: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = assignments[["user_id", "bundle_step_index", "macrostate"]].copy()
    data["user_id"] = pd.to_numeric(data["user_id"], errors="coerce")
    data["bundle_step_index"] = pd.to_numeric(data["bundle_step_index"], errors="coerce")
    data["macrostate"] = pd.to_numeric(data["macrostate"], errors="coerce")
    data = data.dropna(subset=["user_id", "bundle_step_index"]).sort_values(
        ["user_id", "bundle_step_index"], kind="mergesort"
    )
    users = data["user_id"].to_numpy(dtype=np.int64)
    steps = data["bundle_step_index"].to_numpy(dtype=np.int64)
    states = data["macrostate"].to_numpy(dtype=float)
    if len(data) < 2:
        return (
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
        )
    adjacent = (
        (users[1:] == users[:-1])
        & (steps[1:] == steps[:-1] + 1)
        & np.isfinite(states[:-1])
        & np.isfinite(states[1:])
    )
    return (
        users[:-1][adjacent],
        states[:-1][adjacent].astype(np.int64),
        states[1:][adjacent].astype(np.int64),
    )


def user_balanced_state_occupancy(assignments: pd.DataFrame, k: int) -> np.ndarray:
    data = assignments[["user_id", "macrostate"]].copy()
    users = pd.to_numeric(data["user_id"], errors="coerce")
    states = pd.to_numeric(data["macrostate"], errors="coerce")
    valid_user = users.notna()
    data = data[valid_user].copy()
    data["user_id"] = users[valid_user].astype(np.int64)
    data["macrostate"] = states[valid_user]
    user_counts = data.groupby("user_id")["user_id"].transform("count").to_numpy(dtype=float)
    weights = 1.0 / np.maximum(user_counts, 1.0)
    state_values = data["macrostate"].to_numpy(dtype=float)
    valid_state = np.isfinite(state_values) & (state_values >= 0) & (state_values < int(k))
    totals = np.bincount(
        state_values[valid_state].astype(np.int64),
        weights=weights[valid_state],
        minlength=int(k),
    ).astype(float)
    return totals / max(float(totals.sum()), EPS)


def fixed_horizon_kinetics_from_histograms(
    transition: np.ndarray,
    total_histogram: np.ndarray,
    event_histogram: np.ndarray,
    horizon: int,
) -> Dict[str, np.ndarray | float | int]:
    matrix = np.asarray(transition, dtype=float)
    totals = np.asarray(total_histogram, dtype=float)
    events = np.asarray(event_histogram, dtype=float)
    k = matrix.shape[0]
    reference = max(int(horizon), 1)
    pii = np.clip(np.diag(matrix), 1e-6, 1.0 - 1e-6)
    rmst = np.full(k, np.nan, dtype=float)
    tail = np.full(k, np.nan, dtype=float)
    at_risk_reference = np.zeros(k, dtype=float)
    run_count = totals.sum(axis=1)
    completed = events.sum(axis=1)
    for state in range(k):
        state_totals = totals[state]
        state_events = events[state]
        if float(state_totals.sum()) <= 0:
            continue
        risk = np.cumsum(state_totals[::-1])[::-1]
        survival = 1.0
        area = 0.0
        for index in range(len(state_totals)):
            length = index + 1
            if length <= reference:
                area += survival
            if length == reference:
                tail[state] = survival
                at_risk_reference[state] = risk[index]
            if risk[index] > 0 and state_events[index] > 0:
                survival *= max(1.0 - state_events[index] / risk[index], 0.0)
        rmst[state] = area
        if at_risk_reference[state] <= 0:
            tail[state] = np.nan
    lengths = np.arange(1, reference + 1, dtype=float)[:, None]
    geometric_rmst = np.sum(pii[None, :] ** (lengths - 1.0), axis=0)
    geometric_tail = pii ** (reference - 1)
    lift = np.divide(rmst, geometric_rmst, out=np.full(k, np.nan), where=geometric_rmst > 0)
    tail_excess = tail - geometric_tail
    finite_lift = np.isfinite(lift) & (lift > 0)
    aggregate_log_lift = (
        float(np.mean(np.log(lift)))
        if int(np.sum(finite_lift)) == int(k)
        else float("nan")
    )
    diagonal_margin_values = []
    for state in range(k):
        off_diagonal = np.delete(matrix[state], state)
        if off_diagonal.size:
            diagonal_margin_values.append(matrix[state, state] - float(np.max(off_diagonal)))
    diagonal_margin = float(np.mean(diagonal_margin_values)) if diagonal_margin_values else float("nan")
    return {
        "self_transition": pii,
        "rmst_fixed": rmst,
        "geometric_rmst_fixed": geometric_rmst,
        "rmst_lift_fixed": lift,
        "tail_fixed": tail,
        "geometric_tail_fixed": geometric_tail,
        "tail_excess_fixed": tail_excess,
        "reference_at_risk": at_risk_reference,
        "run_count": run_count,
        "completed_exit_count": completed,
        "aggregate_mean_log_rmst_lift_fixed": aggregate_log_lift,
        "diagonal_margin": diagonal_margin,
        "diagonal_dominant_rows": int(
            np.sum(np.diag(matrix) >= np.max(matrix, axis=1) - 1e-15)
        ),
        "mean_self_transition": float(np.mean(pii)),
    }


def run_histograms_from_dataframe(
    runs: pd.DataFrame,
    k: int,
    maximum_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    data = runs.copy()
    data["macrostate"] = pd.to_numeric(data["macrostate"], errors="coerce")
    data["length"] = pd.to_numeric(data["length"], errors="coerce")
    data = data.dropna(subset=["macrostate", "length"])
    total = np.zeros((int(k), int(maximum_length)), dtype=float)
    events = np.zeros_like(total)
    if data.empty:
        return total, events
    states = data["macrostate"].astype(np.int64).to_numpy()
    lengths = np.minimum(data["length"].astype(np.int64).clip(lower=1).to_numpy(), int(maximum_length))
    observed_values = data["event_observed"]
    if pd.api.types.is_bool_dtype(observed_values) or pd.api.types.is_numeric_dtype(observed_values):
        observed = pd.to_numeric(observed_values, errors="coerce").fillna(0).astype(bool).to_numpy()
    else:
        observed = observed_values.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"}).to_numpy()
    valid = (states >= 0) & (states < int(k))
    encoded = states[valid] * int(maximum_length) + (lengths[valid] - 1)
    total += np.bincount(encoded, minlength=int(k) * int(maximum_length)).reshape(int(k), int(maximum_length))
    event_valid = valid & observed
    encoded_event = states[event_valid] * int(maximum_length) + (lengths[event_valid] - 1)
    events += np.bincount(encoded_event, minlength=int(k) * int(maximum_length)).reshape(int(k), int(maximum_length))
    return total, events


class ClusterKineticAccumulator:
    def __init__(
        self,
        assignments: pd.DataFrame,
        runs: pd.DataFrame,
        formal_summary: pd.DataFrame,
        k: int,
        maximum_length: int,
    ) -> None:
        self.k = int(k)
        users = np.asarray(sorted(pd.to_numeric(assignments["user_id"], errors="raise").astype(np.int64).unique()), dtype=np.int64)
        self.user_values = users
        self.n_users = len(users)
        user_position = pd.Series(np.arange(self.n_users, dtype=np.int64), index=users)

        transition_user, current, next_state = transition_rows(assignments)
        mapped_transition_user = user_position.reindex(transition_user).to_numpy(dtype=float)
        if not np.isfinite(mapped_transition_user).all():
            raise RuntimeError("Transition rows contain users outside the assignment user set.")
        valid_transition = (
            (current >= 0)
            & (current < self.k)
            & (next_state >= 0)
            & (next_state < self.k)
        )
        pair = current[valid_transition] * self.k + next_state[valid_transition]
        self.transition_by_user = sparse.coo_matrix(
            (
                np.ones(int(np.sum(valid_transition)), dtype=float),
                (pair, mapped_transition_user[valid_transition].astype(np.int64)),
            ),
            shape=(self.k * self.k, self.n_users),
        ).tocsr()

        summary = formal_summary.copy()
        summary["macrostate"] = pd.to_numeric(summary["macrostate"], errors="raise").astype(np.int64)
        summary_index = summary.set_index("macrostate")
        run_data = runs.copy()
        run_data["user_id"] = pd.to_numeric(run_data["user_id"], errors="coerce")
        run_data["macrostate"] = pd.to_numeric(run_data["macrostate"], errors="coerce")
        run_data["length"] = pd.to_numeric(run_data["length"], errors="coerce")
        run_data = run_data.dropna(subset=["user_id", "macrostate", "length"])
        run_data["user_index"] = user_position.reindex(run_data["user_id"].astype(np.int64)).to_numpy(dtype=float)
        run_data = run_data[np.isfinite(run_data["user_index"])].copy()
        run_data["user_index"] = run_data["user_index"].astype(np.int64)
        run_data["macrostate"] = run_data["macrostate"].astype(np.int64)
        run_data["length"] = run_data["length"].astype(np.int64).clip(lower=1, upper=int(maximum_length))
        observed_values = run_data["event_observed"]
        if pd.api.types.is_bool_dtype(observed_values) or pd.api.types.is_numeric_dtype(observed_values):
            run_data["observed"] = pd.to_numeric(observed_values, errors="coerce").fillna(0).astype(bool)
        else:
            run_data["observed"] = observed_values.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})

        self.states: Dict[int, Dict[str, Any]] = {}
        for state in range(self.k):
            if state not in summary_index.index:
                raise RuntimeError(f"Formal residence summary is missing macrostate {state}.")
            state_runs = run_data[run_data["macrostate"] == state]
            tau = int(float(summary_index.loc[state, "rmst_tau"]))
            reference = int(float(summary_index.loc[state, "reference_length"]))
            maximum = int(max(maximum_length, tau, reference, 1))
            lengths = np.minimum(state_runs["length"].to_numpy(dtype=np.int64), maximum)
            run_users = state_runs["user_index"].to_numpy(dtype=np.int64)
            observed = state_runs["observed"].to_numpy(dtype=bool)
            total_matrix = sparse.coo_matrix(
                (np.ones(len(state_runs), dtype=float), (lengths - 1, run_users)),
                shape=(maximum, self.n_users),
            ).tocsr()
            event_matrix = sparse.coo_matrix(
                (
                    np.ones(int(np.sum(observed)), dtype=float),
                    (lengths[observed] - 1, run_users[observed]),
                ),
                shape=(maximum, self.n_users),
            ).tocsr()
            self.states[state] = {
                "tau": max(tau, 1),
                "reference": max(reference, 1),
                "maximum": maximum,
                "total_matrix": total_matrix,
                "event_matrix": event_matrix,
            }

    def evaluate(self, multiplicities: np.ndarray) -> Dict[str, np.ndarray]:
        weights = np.asarray(multiplicities, dtype=float)
        if weights.ndim == 1:
            weights = weights[:, None]
        if weights.shape[0] != self.n_users:
            raise ValueError("Cluster multiplicities do not match the frozen user set.")
        batch = weights.shape[1]
        transition_flat = np.asarray(self.transition_by_user @ weights, dtype=float)
        transition_counts = transition_flat.reshape(self.k, self.k, batch)
        row_totals = transition_counts.sum(axis=1, keepdims=True)
        transition = np.divide(
            transition_counts,
            row_totals,
            out=np.zeros_like(transition_counts),
            where=row_totals > 0,
        )
        self_transition = np.asarray(
            [transition[state, state] for state in range(self.k)], dtype=float
        )
        row_present = row_totals[:, 0, :] > 0
        self_transition[~row_present] = np.nan
        diagonal_dominant = np.full((self.k, batch), np.nan, dtype=float)
        for replicate in range(batch):
            matrix = transition[:, :, replicate]
            present = row_present[:, replicate]
            diagonal_dominant[present, replicate] = (
                np.diag(matrix)[present]
                >= np.max(matrix, axis=1)[present] - 1e-15
            ).astype(float)

        rmst_lift = np.full((self.k, batch), np.nan, dtype=float)
        tail_excess = np.full((self.k, batch), np.nan, dtype=float)
        reference_tail = np.full((self.k, batch), np.nan, dtype=float)
        for state, payload in self.states.items():
            totals = np.asarray(payload["total_matrix"] @ weights, dtype=float)
            events = np.asarray(payload["event_matrix"] @ weights, dtype=float)
            risk = np.cumsum(totals[::-1], axis=0)[::-1]
            survival = np.ones(batch, dtype=float)
            rmst = np.zeros(batch, dtype=float)
            state_reference_tail = np.full(batch, np.nan, dtype=float)
            tau = int(payload["tau"])
            reference = int(payload["reference"])
            maximum = int(payload["maximum"])
            for index in range(maximum):
                length = index + 1
                if length <= tau:
                    rmst += survival
                if length == reference:
                    state_reference_tail = survival.copy()
                valid = risk[index] > 0
                fraction = np.zeros(batch, dtype=float)
                fraction[valid] = events[index, valid] / risk[index, valid]
                survival *= np.maximum(1.0 - fraction, 0.0)
            p = np.clip(self_transition[state], 1e-6, 1.0 - 1e-6)
            lengths = np.arange(1, tau + 1, dtype=float)[:, None]
            geometric_rmst = np.sum(p[None, :] ** (lengths - 1.0), axis=0)
            geometric_tail = p ** (reference - 1)
            tau_estimable = risk[tau - 1] > 0 if tau <= maximum else np.zeros(batch, dtype=bool)
            reference_estimable = (
                risk[reference - 1] > 0
                if reference <= maximum
                else np.zeros(batch, dtype=bool)
            )
            state_lift = np.divide(
                rmst,
                geometric_rmst,
                out=np.full(batch, np.nan),
                where=geometric_rmst > 0,
            )
            state_lift[~tau_estimable] = np.nan
            state_tail_excess = state_reference_tail - geometric_tail
            state_tail_excess[~reference_estimable] = np.nan
            state_reference_tail[~reference_estimable] = np.nan
            rmst_lift[state] = state_lift
            tail_excess[state] = state_tail_excess
            reference_tail[state] = state_reference_tail
        return {
            "transition": transition,
            "self_transition": self_transition,
            "diagonal_dominant": diagonal_dominant,
            "rmst_lift": rmst_lift,
            "tail_excess": tail_excess,
            "reference_tail": reference_tail,
        }


def coerce_matching_cutpoints(payload: Mapping[str, Any]) -> Any:
    converted: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            converted[key] = np.asarray(value, dtype=float)
        else:
            converted[key] = value
    return SimpleNamespace(**converted)


def compare_coverage_tables(current: pd.DataFrame, archived: pd.DataFrame) -> pd.DataFrame:
    required = ["level", "rows_assigned", "rows_remaining_after_level"]
    for label, frame in (("current", current), ("archived", archived)):
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise RuntimeError(f"{label} matching coverage is missing columns: {missing}")
    first = current[required].copy()
    second = archived[required].copy()
    first["level"] = first["level"].astype(str)
    second["level"] = second["level"].astype(str)
    merged = first.merge(second, on="level", how="outer", suffixes=("_current", "_archived"), indicator=True)
    merged["passed"] = (
        (merged["_merge"] == "both")
        & (pd.to_numeric(merged["rows_assigned_current"], errors="coerce") == pd.to_numeric(merged["rows_assigned_archived"], errors="coerce"))
        & (
            pd.to_numeric(merged["rows_remaining_after_level_current"], errors="coerce")
            == pd.to_numeric(merged["rows_remaining_after_level_archived"], errors="coerce")
        )
    )
    return merged


def _nearest_labels_python(
    values_m: np.ndarray,
    values_psi: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    standardized_centers: np.ndarray,
) -> np.ndarray:
    matrix = np.column_stack([values_m, values_psi])
    output = np.full(len(matrix), -1, dtype=np.int16)
    valid = np.isfinite(matrix).all(axis=1)
    if np.any(valid):
        standardized = (matrix[valid] - scaler_mean[None, :]) / scaler_scale[None, :]
        distance = np.sum(
            (standardized[:, None, :] - standardized_centers[None, :, :]) ** 2,
            axis=2,
        )
        output[valid] = np.argmin(distance, axis=1).astype(np.int16)
    return output


if NUMBA_ENABLED:
    @njit(cache=True)
    def _recursive_labels_numba(
        observed_m,
        observed_psi,
        finite_observed,
        edge_for_row,
        target_for_edge,
        e_pre,
        b_pre,
        a_m,
        a_psi,
        donor_z_m,
        donor_z_psi,
        scaler_mean,
        scaler_scale,
        standardized_centers,
    ):
        n = len(observed_m)
        k = standardized_centers.shape[0]
        state_m = np.empty(n, dtype=np.float64)
        state_psi = np.empty(n, dtype=np.float64)
        initialized = np.zeros(n, dtype=np.uint8)
        labels = np.full(n, -1, dtype=np.int16)
        bound_excess = 0.0
        invalid_denominator = 0
        for row in range(n):
            if not finite_observed[row]:
                continue
            if initialized[row] == 0:
                state_m[row] = observed_m[row]
                state_psi[row] = observed_psi[row]
                initialized[row] = 1
            standardized_m = (state_m[row] - scaler_mean[0]) / scaler_scale[0]
            standardized_psi = (state_psi[row] - scaler_mean[1]) / scaler_scale[1]
            best = 0
            best_distance = 1e300
            for state in range(k):
                dm = standardized_m - standardized_centers[state, 0]
                dp = standardized_psi - standardized_centers[state, 1]
                distance = dm * dm + dp * dp
                if distance < best_distance:
                    best_distance = distance
                    best = state
            labels[row] = best
            edge = edge_for_row[row]
            if edge < 0:
                continue
            target = target_for_edge[edge]
            denominator_m = e_pre[edge] + a_m[edge]
            denominator_psi = b_pre[edge] + a_psi[edge]
            if denominator_m <= 0.0 or denominator_psi <= 0.0:
                invalid_denominator += 1
                continue
            next_m = (state_m[row] * e_pre[edge] + a_m[edge] * donor_z_m[edge]) / denominator_m
            next_psi = (state_psi[row] * b_pre[edge] + a_psi[edge] * donor_z_psi[edge]) / denominator_psi
            excess_m = abs(next_m) - 1.0
            excess_psi = abs(next_psi) - 1.0
            if excess_m > bound_excess:
                bound_excess = excess_m
            if excess_psi > bound_excess:
                bound_excess = excess_psi
            if next_m > 1.0:
                next_m = 1.0
            elif next_m < -1.0:
                next_m = -1.0
            if next_psi > 1.0:
                next_psi = 1.0
            elif next_psi < -1.0:
                next_psi = -1.0
            state_m[target] = next_m
            state_psi[target] = next_psi
            initialized[target] = 1
        return labels, max(bound_excess, 0.0), invalid_denominator

    @njit(cache=True)
    def _run_histograms_numba(user_id, step, labels, k, maximum_length):
        total = np.zeros((k, maximum_length), dtype=np.float64)
        events = np.zeros((k, maximum_length), dtype=np.float64)
        n = len(labels)
        if n == 0:
            return total, events
        current_user = user_id[0]
        current_state = -1
        previous_step = -1
        run_length = 0
        for index in range(n):
            user = user_id[index]
            current_step = step[index]
            state = labels[index]
            if user != current_user:
                if current_state >= 0 and run_length > 0:
                    length_index = min(run_length, maximum_length) - 1
                    total[current_state, length_index] += 1.0
                current_user = user
                current_state = -1
                previous_step = -1
                run_length = 0
            if current_state >= 0 and previous_step >= 0 and current_step != previous_step + 1:
                length_index = min(run_length, maximum_length) - 1
                total[current_state, length_index] += 1.0
                current_state = -1
                run_length = 0
            if state < 0:
                if current_state >= 0 and run_length > 0:
                    length_index = min(run_length, maximum_length) - 1
                    total[current_state, length_index] += 1.0
                    current_state = -1
                    run_length = 0
                previous_step = current_step
                continue
            if current_state < 0:
                current_state = state
                run_length = 1
            elif state == current_state:
                run_length += 1
            else:
                length_index = min(run_length, maximum_length) - 1
                total[current_state, length_index] += 1.0
                events[current_state, length_index] += 1.0
                current_state = state
                run_length = 1
            previous_step = current_step
        if current_state >= 0 and run_length > 0:
            length_index = min(run_length, maximum_length) - 1
            total[current_state, length_index] += 1.0
        return total, events
else:
    _recursive_labels_numba = None
    _run_histograms_numba = None


def _recursive_labels_python_exact(
    observed_m: np.ndarray,
    observed_psi: np.ndarray,
    finite_observed: np.ndarray,
    edge_for_row: np.ndarray,
    target_for_edge: np.ndarray,
    e_pre: np.ndarray,
    b_pre: np.ndarray,
    a_m: np.ndarray,
    a_psi: np.ndarray,
    donor_z_m: np.ndarray,
    donor_z_psi: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    standardized_centers: np.ndarray,
) -> Tuple[np.ndarray, float, int]:
    n = int(len(observed_m))
    state_m = np.where(finite_observed, observed_m, np.nan).astype(np.float64, copy=True)
    state_psi = np.where(finite_observed, observed_psi, np.nan).astype(np.float64, copy=True)
    bound_excess = 0.0
    invalid_denominator = 0
    for row in range(n):
        if not bool(finite_observed[row]):
            continue
        edge = int(edge_for_row[row])
        if edge < 0:
            continue
        target = int(target_for_edge[edge])
        denominator_m = float(e_pre[edge] + a_m[edge])
        denominator_psi = float(b_pre[edge] + a_psi[edge])
        if denominator_m <= 0.0 or denominator_psi <= 0.0:
            invalid_denominator += 1
            continue
        next_m = (
            float(state_m[row]) * float(e_pre[edge])
            + float(a_m[edge]) * float(donor_z_m[edge])
        ) / denominator_m
        next_psi = (
            float(state_psi[row]) * float(b_pre[edge])
            + float(a_psi[edge]) * float(donor_z_psi[edge])
        ) / denominator_psi
        bound_excess = max(bound_excess, abs(next_m) - 1.0, abs(next_psi) - 1.0)
        state_m[target] = min(1.0, max(-1.0, next_m))
        state_psi[target] = min(1.0, max(-1.0, next_psi))
    labels = _nearest_labels_python(
        state_m,
        state_psi,
        scaler_mean,
        scaler_scale,
        standardized_centers,
    )
    labels[~finite_observed] = -1
    return labels, max(float(bound_excess), 0.0), int(invalid_denominator)


def _solve_recursive_coordinate_banded(
    observed: np.ndarray,
    finite_observed: np.ndarray,
    source_rows: np.ndarray,
    edge_indices: np.ndarray,
    target_rows: np.ndarray,
    active: np.ndarray,
    pre: np.ndarray,
    increment: np.ndarray,
    donor: np.ndarray,
) -> np.ndarray:
    n = int(len(observed))
    rhs = np.where(finite_observed, observed, 0.0).astype(np.float64, copy=True)
    band = np.zeros((2, n), dtype=np.float64, order="F")
    band[0, :] = 1.0
    if np.any(active):
        source = source_rows[active]
        edge = edge_indices[active]
        target = target_rows[active]
        denominator = pre[edge] + increment[edge]
        band[1, source] = -pre[edge] / denominator
        rhs[target] = increment[edge] * donor[edge] / denominator
    rhs_matrix = np.asfortranarray(rhs.reshape(n, 1))
    solver = get_lapack_funcs("tbtrs", (band, rhs_matrix))
    solution, info = solver(
        band,
        rhs_matrix,
        uplo=b"L",
        trans=b"N",
        diag=b"N",
        overwrite_b=1,
    )
    if info != 0:
        raise RuntimeError(f"LAPACK triangular-band recursion failed with info={info}.")
    state = np.asarray(solution[:, 0], dtype=np.float64)
    if np.any(finite_observed & ~np.isfinite(state)):
        raise RuntimeError("The SciPy recursive backend produced non-finite states.")
    return state


def _recursive_labels_banded(
    observed_m: np.ndarray,
    observed_psi: np.ndarray,
    finite_observed: np.ndarray,
    edge_for_row: np.ndarray,
    target_for_edge: np.ndarray,
    e_pre: np.ndarray,
    b_pre: np.ndarray,
    a_m: np.ndarray,
    a_psi: np.ndarray,
    donor_z_m: np.ndarray,
    donor_z_psi: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    standardized_centers: np.ndarray,
) -> Tuple[np.ndarray, float, int]:
    n = int(len(observed_m))
    source_rows = np.flatnonzero(edge_for_row >= 0).astype(np.int64, copy=False)
    if source_rows.size == 0:
        labels = _nearest_labels_python(
            observed_m,
            observed_psi,
            scaler_mean,
            scaler_scale,
            standardized_centers,
        )
        labels[~finite_observed] = -1
        return labels, 0.0, 0
    edge_indices = edge_for_row[source_rows].astype(np.int64, copy=False)
    if np.any(edge_indices < 0) or np.any(edge_indices >= len(target_for_edge)):
        return _recursive_labels_python_exact(
            observed_m,
            observed_psi,
            finite_observed,
            edge_for_row,
            target_for_edge,
            e_pre,
            b_pre,
            a_m,
            a_psi,
            donor_z_m,
            donor_z_psi,
            scaler_mean,
            scaler_scale,
            standardized_centers,
        )
    target_rows = target_for_edge[edge_indices].astype(np.int64, copy=False)
    if (
        np.any(target_rows < 0)
        or np.any(target_rows >= n)
        or np.any(target_rows != source_rows + 1)
        or np.unique(target_rows).size != target_rows.size
    ):
        return _recursive_labels_python_exact(
            observed_m,
            observed_psi,
            finite_observed,
            edge_for_row,
            target_for_edge,
            e_pre,
            b_pre,
            a_m,
            a_psi,
            donor_z_m,
            donor_z_psi,
            scaler_mean,
            scaler_scale,
            standardized_centers,
        )
    denominator_m = e_pre[edge_indices] + a_m[edge_indices]
    denominator_psi = b_pre[edge_indices] + a_psi[edge_indices]
    invalid = finite_observed[source_rows] & (
        ~np.isfinite(denominator_m)
        | ~np.isfinite(denominator_psi)
        | (denominator_m <= 0.0)
        | (denominator_psi <= 0.0)
    )
    active = finite_observed[source_rows] & finite_observed[target_rows] & ~invalid
    state_m = _solve_recursive_coordinate_banded(
        observed_m,
        finite_observed,
        source_rows,
        edge_indices,
        target_rows,
        active,
        e_pre,
        a_m,
        donor_z_m,
    )
    state_psi = _solve_recursive_coordinate_banded(
        observed_psi,
        finite_observed,
        source_rows,
        edge_indices,
        target_rows,
        active,
        b_pre,
        a_psi,
        donor_z_psi,
    )
    active_targets = target_rows[active]
    if active_targets.size:
        bound_excess = max(
            0.0,
            float(np.max(np.abs(state_m[active_targets]) - 1.0)),
            float(np.max(np.abs(state_psi[active_targets]) - 1.0)),
        )
    else:
        bound_excess = 0.0
    if bound_excess > 1e-10:
        return _recursive_labels_python_exact(
            observed_m,
            observed_psi,
            finite_observed,
            edge_for_row,
            target_for_edge,
            e_pre,
            b_pre,
            a_m,
            a_psi,
            donor_z_m,
            donor_z_psi,
            scaler_mean,
            scaler_scale,
            standardized_centers,
        )
    np.clip(state_m, -1.0, 1.0, out=state_m)
    np.clip(state_psi, -1.0, 1.0, out=state_psi)
    labels = _nearest_labels_python(
        state_m,
        state_psi,
        scaler_mean,
        scaler_scale,
        standardized_centers,
    )
    labels[~finite_observed] = -1
    return labels, float(bound_excess), int(np.sum(invalid))


def _run_histograms_python_exact(
    user_id: np.ndarray,
    step: np.ndarray,
    labels: np.ndarray,
    k: int,
    maximum_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    total = np.zeros((int(k), int(maximum_length)), dtype=np.float64)
    events = np.zeros((int(k), int(maximum_length)), dtype=np.float64)
    n = int(len(labels))
    if n == 0:
        return total, events
    current_user = int(user_id[0])
    current_state = -1
    previous_step = -1
    run_length = 0
    for index in range(n):
        user = int(user_id[index])
        current_step = int(step[index])
        state = int(labels[index])
        if user != current_user:
            if current_state >= 0 and run_length > 0:
                total[current_state, min(run_length, int(maximum_length)) - 1] += 1.0
            current_user = user
            current_state = -1
            previous_step = -1
            run_length = 0
        if current_state >= 0 and previous_step >= 0 and current_step != previous_step + 1:
            total[current_state, min(run_length, int(maximum_length)) - 1] += 1.0
            current_state = -1
            run_length = 0
        if state < 0:
            if current_state >= 0 and run_length > 0:
                total[current_state, min(run_length, int(maximum_length)) - 1] += 1.0
                current_state = -1
                run_length = 0
            previous_step = current_step
            continue
        if current_state < 0:
            current_state = state
            run_length = 1
        elif state == current_state:
            run_length += 1
        else:
            length_index = min(run_length, int(maximum_length)) - 1
            total[current_state, length_index] += 1.0
            events[current_state, length_index] += 1.0
            current_state = state
            run_length = 1
        previous_step = current_step
    if current_state >= 0 and run_length > 0:
        total[current_state, min(run_length, int(maximum_length)) - 1] += 1.0
    return total, events


def _run_histograms_numpy(
    user_id: np.ndarray,
    step: np.ndarray,
    labels: np.ndarray,
    k: int,
    maximum_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    total = np.zeros((int(k), int(maximum_length)), dtype=np.float64)
    events = np.zeros_like(total)
    n = int(len(labels))
    if n == 0:
        return total, events
    valid = (labels >= 0) & (labels < int(k))
    previous_contiguous = np.zeros(n, dtype=bool)
    previous_contiguous[1:] = (
        (user_id[1:] == user_id[:-1])
        & (step[1:] == step[:-1] + 1)
    )
    next_contiguous = np.zeros(n, dtype=bool)
    next_contiguous[:-1] = previous_contiguous[1:]
    starts = valid.copy()
    starts[1:] &= (
        ~previous_contiguous[1:]
        | ~valid[:-1]
        | (labels[1:] != labels[:-1])
    )
    ends = valid.copy()
    ends[:-1] &= (
        ~next_contiguous[:-1]
        | ~valid[1:]
        | (labels[:-1] != labels[1:])
    )
    start_rows = np.flatnonzero(starts)
    end_rows = np.flatnonzero(ends)
    if start_rows.size != end_rows.size:
        return _run_histograms_python_exact(user_id, step, labels, int(k), int(maximum_length))
    if start_rows.size == 0:
        return total, events
    lengths = end_rows - start_rows + 1
    if np.any(lengths <= 0):
        return _run_histograms_python_exact(user_id, step, labels, int(k), int(maximum_length))
    states = labels[start_rows].astype(np.int64, copy=False)
    clipped_lengths = np.minimum(lengths, int(maximum_length)).astype(np.int64, copy=False)
    encoded = states * int(maximum_length) + clipped_lengths - 1
    total += np.bincount(
        encoded,
        minlength=int(k) * int(maximum_length),
    ).reshape(int(k), int(maximum_length))
    observed_exit = np.zeros(start_rows.size, dtype=bool)
    eligible = end_rows < n - 1
    next_rows = end_rows[eligible] + 1
    observed_exit[eligible] = (
        (user_id[next_rows] == user_id[end_rows[eligible]])
        & (step[next_rows] == step[end_rows[eligible]] + 1)
        & valid[next_rows]
        & (labels[next_rows] != labels[end_rows[eligible]])
    )
    if np.any(observed_exit):
        events += np.bincount(
            encoded[observed_exit],
            minlength=int(k) * int(maximum_length),
        ).reshape(int(k), int(maximum_length))
    return total, events


def recursive_labels(
    observed_m: np.ndarray,
    observed_psi: np.ndarray,
    finite_observed: np.ndarray,
    edge_for_row: np.ndarray,
    target_for_edge: np.ndarray,
    e_pre: np.ndarray,
    b_pre: np.ndarray,
    a_m: np.ndarray,
    a_psi: np.ndarray,
    donor_z_m: np.ndarray,
    donor_z_psi: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    standardized_centers: np.ndarray,
) -> Tuple[np.ndarray, float, int]:
    arrays = (
        np.asarray(observed_m, dtype=np.float64),
        np.asarray(observed_psi, dtype=np.float64),
        np.asarray(finite_observed, dtype=np.bool_),
        np.asarray(edge_for_row, dtype=np.int64),
        np.asarray(target_for_edge, dtype=np.int64),
        np.asarray(e_pre, dtype=np.float64),
        np.asarray(b_pre, dtype=np.float64),
        np.asarray(a_m, dtype=np.float64),
        np.asarray(a_psi, dtype=np.float64),
        np.asarray(donor_z_m, dtype=np.float64),
        np.asarray(donor_z_psi, dtype=np.float64),
        np.asarray(scaler_mean, dtype=np.float64),
        np.asarray(scaler_scale, dtype=np.float64),
        np.asarray(standardized_centers, dtype=np.float64),
    )
    if _recursive_labels_numba is not None:
        try:
            result = _recursive_labels_numba(*arrays)
            _BACKEND_AUDIT["recursive_labels_backend"] = "numba"
            return result
        except Exception as exc:
            _BACKEND_AUDIT["numba_runtime_error"] = f"{type(exc).__name__}: {exc}"
    result = _recursive_labels_banded(*arrays)
    _BACKEND_AUDIT["recursive_labels_backend"] = "scipy_lapack_banded"
    return result


def run_histograms(
    user_id: np.ndarray,
    step: np.ndarray,
    labels: np.ndarray,
    k: int,
    maximum_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    arrays = (
        np.asarray(user_id, dtype=np.int64),
        np.asarray(step, dtype=np.int64),
        np.asarray(labels, dtype=np.int16),
        int(k),
        int(maximum_length),
    )
    if _run_histograms_numba is not None:
        try:
            result = _run_histograms_numba(*arrays)
            _BACKEND_AUDIT["run_histograms_backend"] = "numba"
            return result
        except Exception as exc:
            _BACKEND_AUDIT["numba_runtime_error"] = f"{type(exc).__name__}: {exc}"
    result = _run_histograms_numpy(*arrays)
    _BACKEND_AUDIT["run_histograms_backend"] = "numpy_vectorized"
    return result


def recursive_kernel_backend() -> str:
    recursive = _BACKEND_AUDIT.get("recursive_labels_backend")
    histogram = _BACKEND_AUDIT.get("run_histograms_backend")
    if recursive or histogram:
        return "+".join(value for value in (recursive, histogram) if value)
    return "numba_available" if _recursive_labels_numba is not None else "scipy_lapack_banded+numpy_vectorized"


def recursive_backend_audit() -> Dict[str, Any]:
    return dict(_BACKEND_AUDIT)

