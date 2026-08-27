#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

EPS = 1e-12


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
        x = float(value)
        return x if math.isfinite(x) else None
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_module(path: Path, module_name: str):
    source = path.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location(module_name, str(source))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_table(base: Path) -> Path:
    path = Path(base)
    if path.exists() and path.is_file():
        return path
    for suffix in (".parquet", ".csv.gz", ".csv"):
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find table for {base}")


def read_table(base: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    path = resolve_table(base)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    return pd.read_csv(path, usecols=list(columns) if columns is not None else None, low_memory=False)


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


def digitize_closed_right(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    edges = np.asarray(bins, dtype=float)
    adjusted = np.where(array == edges[-1], np.nextafter(edges[-1], edges[0]), array)
    return np.digitize(adjusted, edges) - 1


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    if int(valid.sum()) < 3:
        return float("nan")
    x = first[valid] - float(np.mean(first[valid]))
    y = second[valid] - float(np.mean(second[valid]))
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0)) if denominator > EPS else float("nan")


def weighted_pearson(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second) & np.isfinite(w) & (w >= 0)
    if int(valid.sum()) < 3 or float(w[valid].sum()) <= 0:
        return float("nan")
    x = first[valid]
    y = second[valid]
    ww = w[valid]
    total = float(ww.sum())
    mx = float(np.sum(ww * x) / total)
    my = float(np.sum(ww * y) / total)
    covariance = float(np.sum(ww * (x - mx) * (y - my)))
    variance_x = float(np.sum(ww * (x - mx) ** 2))
    variance_y = float(np.sum(ww * (y - my) ** 2))
    denominator = math.sqrt(max(variance_x * variance_y, 0.0))
    return float(np.clip(covariance / denominator, -1.0, 1.0)) if denominator > EPS else float("nan")


def weighted_rmse(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=float)
    true = np.asarray(target, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(pred) & np.isfinite(true) & np.isfinite(w) & (w >= 0)
    if not np.any(valid) or float(w[valid].sum()) <= 0:
        return float("nan")
    return float(np.sqrt(np.sum(w[valid] * (pred[valid] - true[valid]) ** 2) / np.sum(w[valid])))


def user_equal_row_weights(user_id: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    users = np.asarray(user_id, dtype=np.int64)
    mask = np.ones(len(users), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    output = np.zeros(len(users), dtype=float)
    if not np.any(mask):
        return output
    selected = pd.Series(users[mask])
    counts = selected.groupby(selected).transform("count").to_numpy(dtype=float)
    output[mask] = 1.0 / np.maximum(counts, 1.0)
    return output


@dataclass
class FieldGrid:
    bins: np.ndarray
    centers: np.ndarray
    occupancy_weight: np.ndarray
    occupancy_probability: np.ndarray
    occupancy_count: np.ndarray
    user_count: np.ndarray
    drift_weight: np.ndarray
    drift_count: np.ndarray
    u: np.ndarray
    v: np.ndarray
    diff_x: np.ndarray
    diff_y: np.ndarray
    diff_xy: np.ndarray
    state_mask: np.ndarray
    drift_mask: np.ndarray


class SparseFieldAccumulator:
    def __init__(
        self,
        user_id: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        bins: np.ndarray,
        state_weights: np.ndarray,
        drift_weights: np.ndarray,
        user_values: Optional[np.ndarray] = None,
    ) -> None:
        self.user_id = np.asarray(user_id, dtype=np.int64)
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.dx = np.asarray(dx, dtype=float)
        self.dy = np.asarray(dy, dtype=float)
        self.bins = np.asarray(bins, dtype=float)
        self.centers = 0.5 * (self.bins[:-1] + self.bins[1:])
        self.n_axis = len(self.centers)
        self.n_cells = self.n_axis * self.n_axis
        if user_values is None:
            self.user_values, inverse = np.unique(self.user_id, return_inverse=True)
        else:
            self.user_values = np.asarray(user_values, dtype=np.int64)
            position = pd.Series(np.arange(len(self.user_values), dtype=np.int64), index=self.user_values)
            mapped = position.reindex(self.user_id).to_numpy(dtype=float)
            if not np.isfinite(mapped).all():
                raise RuntimeError("Field rows contain users outside the frozen user set.")
            inverse = mapped.astype(np.int64)
        self.user_index = inverse
        self.n_users = len(self.user_values)
        ix = digitize_closed_right(self.x, self.bins)
        iy = digitize_closed_right(self.y, self.bins)
        state_valid = (
            np.isfinite(self.x)
            & np.isfinite(self.y)
            & (ix >= 0)
            & (ix < self.n_axis)
            & (iy >= 0)
            & (iy < self.n_axis)
        )
        drift_valid = state_valid & np.isfinite(self.dx) & np.isfinite(self.dy)
        self.state_valid = state_valid
        self.drift_valid = drift_valid
        state_cell = ix[state_valid] * self.n_axis + iy[state_valid]
        drift_cell = ix[drift_valid] * self.n_axis + iy[drift_valid]
        self.occupancy_count = np.bincount(state_cell, minlength=self.n_cells).astype(float)
        self.drift_count = np.bincount(drift_cell, minlength=self.n_cells).astype(float)
        if np.any(state_valid):
            encoded = self.user_index[state_valid].astype(np.int64) * self.n_cells + state_cell.astype(np.int64)
            unique_encoded = np.unique(encoded)
            unique_cells = unique_encoded % self.n_cells
            self.user_count = np.bincount(unique_cells, minlength=self.n_cells).astype(float)
        else:
            self.user_count = np.zeros(self.n_cells, dtype=float)
        state_w = np.asarray(state_weights, dtype=float)[state_valid]
        drift_w = np.asarray(drift_weights, dtype=float)[drift_valid]
        matrices = [
            self._component(state_cell, self.user_index[state_valid], state_w),
            self._component(drift_cell, self.user_index[drift_valid], drift_w),
            self._component(drift_cell, self.user_index[drift_valid], drift_w * self.dx[drift_valid]),
            self._component(drift_cell, self.user_index[drift_valid], drift_w * self.dy[drift_valid]),
            self._component(drift_cell, self.user_index[drift_valid], drift_w * self.dx[drift_valid] ** 2),
            self._component(drift_cell, self.user_index[drift_valid], drift_w * self.dy[drift_valid] ** 2),
            self._component(drift_cell, self.user_index[drift_valid], drift_w * self.dx[drift_valid] * self.dy[drift_valid]),
        ]
        self.block = sparse.vstack(matrices, format="csr")

    def _component(self, cells: np.ndarray, users: np.ndarray, data: np.ndarray) -> sparse.csr_matrix:
        if len(cells) == 0:
            return sparse.csr_matrix((self.n_cells, self.n_users), dtype=float)
        return sparse.coo_matrix(
            (np.asarray(data, dtype=float), (np.asarray(cells, dtype=np.int64), np.asarray(users, dtype=np.int64))),
            shape=(self.n_cells, self.n_users),
        ).tocsr()

    def totals(self, multipliers: np.ndarray) -> np.ndarray:
        values = np.asarray(multipliers, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.shape[0] != self.n_users:
            raise ValueError("Multiplier length does not match the frozen user set.")
        return np.asarray(self.block @ values, dtype=float)

    def field_from_totals(
        self,
        totals: np.ndarray,
        column: int = 0,
        min_state_count: int = 0,
        min_cell_users: int = 0,
        min_drift_count: int = 30,
    ) -> FieldGrid:
        array = np.asarray(totals, dtype=float)
        if array.ndim == 1:
            vector = array
        else:
            vector = array[:, column]
        parts = vector.reshape(7, self.n_cells)
        occupancy = parts[0]
        drift_weight = parts[1]
        sx = parts[2]
        sy = parts[3]
        sxx = parts[4]
        syy = parts[5]
        sxy = parts[6]
        denominator = np.maximum(drift_weight, EPS)
        u = sx / denominator
        v = sy / denominator
        diff_x = np.maximum(sxx / denominator - u * u, 0.0)
        diff_y = np.maximum(syy / denominator - v * v, 0.0)
        diff_xy = sxy / denominator - u * v
        probability = occupancy / max(float(np.sum(occupancy)), EPS)
        shape = (self.n_axis, self.n_axis)
        state_mask = (self.occupancy_count >= int(min_state_count)) & (self.user_count >= int(min_cell_users))
        drift_mask = self.drift_count >= int(min_drift_count)
        return FieldGrid(
            bins=self.bins,
            centers=self.centers,
            occupancy_weight=occupancy.reshape(shape),
            occupancy_probability=probability.reshape(shape),
            occupancy_count=self.occupancy_count.reshape(shape),
            user_count=self.user_count.reshape(shape),
            drift_weight=drift_weight.reshape(shape),
            drift_count=self.drift_count.reshape(shape),
            u=u.reshape(shape),
            v=v.reshape(shape),
            diff_x=diff_x.reshape(shape),
            diff_y=diff_y.reshape(shape),
            diff_xy=diff_xy.reshape(shape),
            state_mask=state_mask.reshape(shape),
            drift_mask=drift_mask.reshape(shape),
        )

    def point_field(self, **kwargs: Any) -> FieldGrid:
        return self.field_from_totals(self.totals(np.ones(self.n_users, dtype=float)), **kwargs)



class SparseDriftAccumulator:
    def __init__(
        self,
        user_id: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        bins: np.ndarray,
        weights: np.ndarray,
        user_values: Optional[np.ndarray] = None,
    ) -> None:
        users = np.asarray(user_id, dtype=np.int64)
        state_x = np.asarray(x, dtype=float)
        state_y = np.asarray(y, dtype=float)
        delta_x = np.asarray(dx, dtype=float)
        delta_y = np.asarray(dy, dtype=float)
        row_weights = np.asarray(weights, dtype=float)
        self.bins = np.asarray(bins, dtype=float)
        self.centers = 0.5 * (self.bins[:-1] + self.bins[1:])
        self.n_axis = len(self.centers)
        self.n_cells = self.n_axis * self.n_axis
        if user_values is None:
            self.user_values, inverse = np.unique(users, return_inverse=True)
        else:
            self.user_values = np.asarray(user_values, dtype=np.int64)
            position = pd.Series(np.arange(len(self.user_values), dtype=np.int64), index=self.user_values)
            mapped = position.reindex(users).to_numpy(dtype=float)
            if not np.isfinite(mapped).all():
                raise RuntimeError("Drift rows contain users outside the frozen user set.")
            inverse = mapped.astype(np.int64)
        self.n_users = len(self.user_values)
        ix = digitize_closed_right(state_x, self.bins)
        iy = digitize_closed_right(state_y, self.bins)
        valid = (
            np.isfinite(state_x)
            & np.isfinite(state_y)
            & np.isfinite(delta_x)
            & np.isfinite(delta_y)
            & np.isfinite(row_weights)
            & (row_weights >= 0)
            & (ix >= 0)
            & (ix < self.n_axis)
            & (iy >= 0)
            & (iy < self.n_axis)
        )
        cell = ix[valid] * self.n_axis + iy[valid]
        user_index = inverse[valid]
        weight = row_weights[valid]
        self.drift_count = np.bincount(cell, minlength=self.n_cells).astype(float)
        matrices = [
            sparse.coo_matrix(
                (data, (cell, user_index)),
                shape=(self.n_cells, self.n_users),
            ).tocsr()
            for data in (weight, weight * delta_x[valid], weight * delta_y[valid])
        ]
        self.block = sparse.vstack(matrices, format="csr")

    def totals(self, multipliers: np.ndarray) -> np.ndarray:
        values = np.asarray(multipliers, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.shape[0] != self.n_users:
            raise ValueError("Multiplier length does not match the frozen user set.")
        return np.asarray(self.block @ values, dtype=float)

    def field_from_totals(
        self,
        totals: np.ndarray,
        column: int = 0,
        min_drift_count: int = 30,
    ) -> FieldGrid:
        array = np.asarray(totals, dtype=float)
        vector = array if array.ndim == 1 else array[:, column]
        parts = vector.reshape(3, self.n_cells)
        drift_weight = parts[0]
        denominator = np.maximum(drift_weight, EPS)
        u = parts[1] / denominator
        v = parts[2] / denominator
        shape = (self.n_axis, self.n_axis)
        zeros = np.zeros(shape, dtype=float)
        return FieldGrid(
            bins=self.bins,
            centers=self.centers,
            occupancy_weight=zeros.copy(),
            occupancy_probability=zeros.copy(),
            occupancy_count=zeros.copy(),
            user_count=zeros.copy(),
            drift_weight=drift_weight.reshape(shape),
            drift_count=self.drift_count.reshape(shape),
            u=u.reshape(shape),
            v=v.reshape(shape),
            diff_x=zeros.copy(),
            diff_y=zeros.copy(),
            diff_xy=zeros.copy(),
            state_mask=np.zeros(shape, dtype=bool),
            drift_mask=(self.drift_count >= int(min_drift_count)).reshape(shape),
        )

    def point_field(self, min_drift_count: int = 30) -> FieldGrid:
        return self.field_from_totals(
            self.totals(np.ones(self.n_users, dtype=float)),
            min_drift_count=min_drift_count,
        )

def direct_field(
    user_id: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    bins: np.ndarray,
    state_weights: np.ndarray,
    drift_weights: np.ndarray,
    min_state_count: int = 0,
    min_cell_users: int = 0,
    min_drift_count: int = 30,
) -> FieldGrid:
    users = np.asarray(user_id, dtype=np.int64)
    state_x = np.asarray(x, dtype=float)
    state_y = np.asarray(y, dtype=float)
    delta_x = np.asarray(dx, dtype=float)
    delta_y = np.asarray(dy, dtype=float)
    edges = np.asarray(bins, dtype=float)
    state_w = np.asarray(state_weights, dtype=float)
    drift_w = np.asarray(drift_weights, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_axis = len(centers)
    n_cells = n_axis * n_axis
    ix = digitize_closed_right(state_x, edges)
    iy = digitize_closed_right(state_y, edges)
    state_valid = (
        np.isfinite(state_x)
        & np.isfinite(state_y)
        & np.isfinite(state_w)
        & (state_w >= 0)
        & (ix >= 0)
        & (ix < n_axis)
        & (iy >= 0)
        & (iy < n_axis)
    )
    drift_valid = (
        state_valid
        & np.isfinite(delta_x)
        & np.isfinite(delta_y)
        & np.isfinite(drift_w)
        & (drift_w >= 0)
    )
    state_cell = ix[state_valid] * n_axis + iy[state_valid]
    drift_cell = ix[drift_valid] * n_axis + iy[drift_valid]
    occupancy_count = np.bincount(state_cell, minlength=n_cells).astype(float)
    occupancy_weight = np.bincount(
        state_cell, weights=state_w[state_valid], minlength=n_cells
    ).astype(float)
    user_count = np.zeros(n_cells, dtype=float)
    if np.any(state_valid):
        user_values, user_inverse = np.unique(users[state_valid], return_inverse=True)
        encoded = user_inverse.astype(np.int64) * n_cells + state_cell.astype(np.int64)
        unique_encoded = np.unique(encoded)
        user_count = np.bincount(unique_encoded % n_cells, minlength=n_cells).astype(float)
    drift_count = np.bincount(drift_cell, minlength=n_cells).astype(float)
    drift_weight = np.bincount(
        drift_cell, weights=drift_w[drift_valid], minlength=n_cells
    ).astype(float)
    sx = np.bincount(
        drift_cell, weights=drift_w[drift_valid] * delta_x[drift_valid], minlength=n_cells
    ).astype(float)
    sy = np.bincount(
        drift_cell, weights=drift_w[drift_valid] * delta_y[drift_valid], minlength=n_cells
    ).astype(float)
    sxx = np.bincount(
        drift_cell, weights=drift_w[drift_valid] * delta_x[drift_valid] ** 2, minlength=n_cells
    ).astype(float)
    syy = np.bincount(
        drift_cell, weights=drift_w[drift_valid] * delta_y[drift_valid] ** 2, minlength=n_cells
    ).astype(float)
    sxy = np.bincount(
        drift_cell, weights=drift_w[drift_valid] * delta_x[drift_valid] * delta_y[drift_valid], minlength=n_cells
    ).astype(float)
    denominator = np.maximum(drift_weight, EPS)
    u = sx / denominator
    v = sy / denominator
    diff_x = np.maximum(sxx / denominator - u * u, 0.0)
    diff_y = np.maximum(syy / denominator - v * v, 0.0)
    diff_xy = sxy / denominator - u * v
    probability = occupancy_weight / max(float(occupancy_weight.sum()), EPS)
    shape = (n_axis, n_axis)
    return FieldGrid(
        bins=edges,
        centers=centers,
        occupancy_weight=occupancy_weight.reshape(shape),
        occupancy_probability=probability.reshape(shape),
        occupancy_count=occupancy_count.reshape(shape),
        user_count=user_count.reshape(shape),
        drift_weight=drift_weight.reshape(shape),
        drift_count=drift_count.reshape(shape),
        u=u.reshape(shape),
        v=v.reshape(shape),
        diff_x=diff_x.reshape(shape),
        diff_y=diff_y.reshape(shape),
        diff_xy=diff_xy.reshape(shape),
        state_mask=((occupancy_count >= int(min_state_count)) & (user_count >= int(min_cell_users))).reshape(shape),
        drift_mask=(drift_count >= int(min_drift_count)).reshape(shape),
    )



def direct_drift_field(
    x: np.ndarray,
    y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    bins: np.ndarray,
    weights: np.ndarray,
    min_drift_count: int = 30,
) -> FieldGrid:
    state_x = np.asarray(x, dtype=float)
    state_y = np.asarray(y, dtype=float)
    delta_x = np.asarray(dx, dtype=float)
    delta_y = np.asarray(dy, dtype=float)
    row_weights = np.asarray(weights, dtype=float)
    edges = np.asarray(bins, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_axis = len(centers)
    n_cells = n_axis * n_axis
    ix = digitize_closed_right(state_x, edges)
    iy = digitize_closed_right(state_y, edges)
    valid = (
        np.isfinite(state_x)
        & np.isfinite(state_y)
        & np.isfinite(delta_x)
        & np.isfinite(delta_y)
        & np.isfinite(row_weights)
        & (row_weights >= 0)
        & (ix >= 0)
        & (ix < n_axis)
        & (iy >= 0)
        & (iy < n_axis)
    )
    cell = ix[valid] * n_axis + iy[valid]
    weight = row_weights[valid]
    drift_count = np.bincount(cell, minlength=n_cells).astype(float)
    drift_weight = np.bincount(cell, weights=weight, minlength=n_cells).astype(float)
    sx = np.bincount(cell, weights=weight * delta_x[valid], minlength=n_cells).astype(float)
    sy = np.bincount(cell, weights=weight * delta_y[valid], minlength=n_cells).astype(float)
    denominator = np.maximum(drift_weight, EPS)
    shape = (n_axis, n_axis)
    zeros = np.zeros(shape, dtype=float)
    return FieldGrid(
        bins=edges,
        centers=centers,
        occupancy_weight=zeros.copy(),
        occupancy_probability=zeros.copy(),
        occupancy_count=zeros.copy(),
        user_count=zeros.copy(),
        drift_weight=drift_weight.reshape(shape),
        drift_count=drift_count.reshape(shape),
        u=(sx / denominator).reshape(shape),
        v=(sy / denominator).reshape(shape),
        diff_x=zeros.copy(),
        diff_y=zeros.copy(),
        diff_xy=zeros.copy(),
        state_mask=np.zeros(shape, dtype=bool),
        drift_mask=(drift_count >= int(min_drift_count)).reshape(shape),
    )

def drift_comparison(first: FieldGrid, second: FieldGrid, extra_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    mask = np.asarray(first.drift_mask, dtype=bool) & np.asarray(second.drift_mask, dtype=bool)
    if extra_mask is not None:
        mask &= np.asarray(extra_mask, dtype=bool)
    if int(mask.sum()) < 3:
        return {
            "common_supported_cells": int(mask.sum()),
            "drift_vector_corr": float("nan"),
            "mean_local_drift_cosine": float("nan"),
            "occupancy_weighted_local_drift_cosine": float("nan"),
            "drift_speed_corr": float("nan"),
            "drift_component_rmse": float("nan"),
        }
    a_u = first.u[mask]
    a_v = first.v[mask]
    b_u = second.u[mask]
    b_v = second.v[mask]
    speed_a = np.sqrt(a_u * a_u + a_v * a_v)
    speed_b = np.sqrt(b_u * b_u + b_v * b_v)
    valid_cos = (speed_a > EPS) & (speed_b > EPS)
    cosine = np.full(len(a_u), np.nan, dtype=float)
    cosine[valid_cos] = (a_u[valid_cos] * b_u[valid_cos] + a_v[valid_cos] * b_v[valid_cos]) / (
        speed_a[valid_cos] * speed_b[valid_cos]
    )
    weights = np.asarray(first.drift_weight[mask], dtype=float)
    valid_weighted = np.isfinite(cosine) & np.isfinite(weights) & (weights >= 0)
    weighted_cosine = (
        float(np.sum(weights[valid_weighted] * cosine[valid_weighted]) / np.sum(weights[valid_weighted]))
        if np.any(valid_weighted) and float(np.sum(weights[valid_weighted])) > 0
        else float("nan")
    )
    vector_a = np.column_stack([a_u, a_v]).ravel()
    vector_b = np.column_stack([b_u, b_v]).ravel()
    residual = np.concatenate([b_u - a_u, b_v - a_v])
    return {
        "common_supported_cells": int(mask.sum()),
        "drift_vector_corr": pearson(vector_a, vector_b),
        "mean_local_drift_cosine": float(np.nanmean(cosine)),
        "occupancy_weighted_local_drift_cosine": weighted_cosine,
        "drift_speed_corr": pearson(speed_a, speed_b),
        "drift_component_rmse": float(np.sqrt(np.nanmean(residual * residual))),
    }


def interior_divergence(field: FieldGrid, require_state_support: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    u = np.asarray(field.u, dtype=float)
    v = np.asarray(field.v, dtype=float)
    support = np.asarray(field.drift_mask, dtype=bool) & np.isfinite(u) & np.isfinite(v)
    if require_state_support:
        support &= np.asarray(field.state_mask, dtype=bool)
    divergence = np.full_like(u, np.nan, dtype=float)
    interior = np.zeros_like(support, dtype=bool)
    if u.shape[0] < 3 or u.shape[1] < 3:
        return divergence, interior
    interior[1:-1, 1:-1] = (
        support[1:-1, 1:-1]
        & support[:-2, 1:-1]
        & support[2:, 1:-1]
        & support[1:-1, :-2]
        & support[1:-1, 2:]
    )
    denominator_x = (field.centers[2:] - field.centers[:-2])[:, None]
    denominator_y = (field.centers[2:] - field.centers[:-2])[None, :]
    local = (u[2:, 1:-1] - u[:-2, 1:-1]) / np.maximum(denominator_x, EPS)
    local += (v[1:-1, 2:] - v[1:-1, :-2]) / np.maximum(denominator_y, EPS)
    target = divergence[1:-1, 1:-1]
    target[interior[1:-1, 1:-1]] = local[interior[1:-1, 1:-1]]
    divergence[1:-1, 1:-1] = target
    return divergence, interior


def contraction_metrics(field: FieldGrid) -> Dict[str, float]:
    divergence, interior = interior_divergence(field, require_state_support=True)
    weights = np.asarray(field.occupancy_weight, dtype=float)
    valid = interior & np.isfinite(divergence) & np.isfinite(weights) & (weights >= 0)
    if not np.any(valid) or float(np.sum(weights[valid])) <= 0:
        return {
            "interior_divergence_cells": int(np.sum(interior)),
            "weighted_negative_divergence_fraction": float("nan"),
            "weighted_mean_divergence": float("nan"),
        }
    total = float(np.sum(weights[valid]))
    return {
        "interior_divergence_cells": int(np.sum(valid)),
        "weighted_negative_divergence_fraction": float(np.sum(weights[valid] * (divergence[valid] < 0)) / total),
        "weighted_mean_divergence": float(np.sum(weights[valid] * divergence[valid]) / total),
    }


def inward_fraction(field: FieldGrid, reference: Tuple[float, float], extra_mask: Optional[np.ndarray] = None) -> float:
    mask = np.asarray(field.drift_mask, dtype=bool) & np.isfinite(field.u) & np.isfinite(field.v)
    if extra_mask is not None:
        mask &= np.asarray(extra_mask, dtype=bool)
    if not np.any(mask):
        return float("nan")
    x_grid, y_grid = np.meshgrid(field.centers, field.centers, indexing="ij")
    to_x = float(reference[0]) - x_grid[mask]
    to_y = float(reference[1]) - y_grid[mask]
    distance = np.sqrt(to_x * to_x + to_y * to_y)
    component = np.full(int(mask.sum()), np.nan, dtype=float)
    valid = distance > EPS
    component[valid] = (field.u[mask][valid] * to_x[valid] + field.v[mask][valid] * to_y[valid]) / distance[valid]
    weights = np.asarray(field.drift_weight[mask], dtype=float)
    good = np.isfinite(component) & np.isfinite(weights) & (weights >= 0)
    if not np.any(good) or float(np.sum(weights[good])) <= 0:
        return float("nan")
    return float(np.sum(weights[good] * (component[good] > 0)) / np.sum(weights[good]))


@dataclass(frozen=True)
class FrozenPartition:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    standardized_centers: np.ndarray
    k: int

    def labels(self, values: np.ndarray) -> np.ndarray:
        xy = np.asarray(values, dtype=float)
        output = np.full(len(xy), -1, dtype=np.int64)
        valid = np.isfinite(xy).all(axis=1)
        if np.any(valid):
            standardized = (xy[valid] - self.scaler_mean[None, :]) / self.scaler_scale[None, :]
            distance = np.sum((standardized[:, None, :] - self.standardized_centers[None, :, :]) ** 2, axis=2)
            output[valid] = np.argmin(distance, axis=1).astype(np.int64)
        return output


def load_frozen_partition(stage1_root: Path) -> FrozenPartition:
    root = Path(stage1_root).resolve() / "dynamics" / "fixed_k6_mesostates"
    metadata = load_json(root / "fixed_k6_model_metadata.json")
    centers = read_table(root / "fixed_k6_centers").sort_values("macrostate", kind="mergesort")
    k = int(metadata.get("macrostate_k", -1))
    if k != 6 or metadata.get("fit_split") != "A_train" or metadata.get("macrostate_k_rule") != "fixed a priori":
        raise RuntimeError("The Stage-1 fixed K=6 partition contract is not valid.")
    standardized = centers[["center_M_standardized", "center_Psi_standardized"]].to_numpy(dtype=float)
    return FrozenPartition(
        scaler_mean=np.asarray(metadata["scaler_mean"], dtype=float),
        scaler_scale=np.asarray(metadata["scaler_scale"], dtype=float),
        standardized_centers=standardized,
        k=k,
    )


class TransitionAccumulator:
    def __init__(
        self,
        user_id: np.ndarray,
        current: np.ndarray,
        next_state: np.ndarray,
        k: int,
        user_values: Optional[np.ndarray] = None,
    ) -> None:
        users = np.asarray(user_id, dtype=np.int64)
        cur = np.asarray(current, dtype=np.int64)
        nxt = np.asarray(next_state, dtype=np.int64)
        self.k = int(k)
        if user_values is None:
            self.user_values, inverse = np.unique(users, return_inverse=True)
        else:
            self.user_values = np.asarray(user_values, dtype=np.int64)
            position = pd.Series(np.arange(len(self.user_values), dtype=np.int64), index=self.user_values)
            mapped = position.reindex(users).to_numpy(dtype=float)
            if not np.isfinite(mapped).all():
                raise RuntimeError("Transition rows contain users outside the frozen user set.")
            inverse = mapped.astype(np.int64)
        self.n_users = len(self.user_values)
        valid = (cur >= 0) & (cur < self.k) & (nxt >= 0) & (nxt < self.k)
        pair = cur[valid] * self.k + nxt[valid]
        self.matrix = sparse.coo_matrix(
            (np.ones(int(valid.sum()), dtype=float), (pair, inverse[valid])),
            shape=(self.k * self.k, self.n_users),
        ).tocsr()

    def counts(self, multipliers: np.ndarray) -> np.ndarray:
        values = np.asarray(multipliers, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        return np.asarray(self.matrix @ values, dtype=float)

    def point_matrix(self) -> np.ndarray:
        counts = self.counts(np.ones(self.n_users, dtype=float))[:, 0].reshape(self.k, self.k)
        return normalize_transition(counts)

    def strict_user_equal_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        user_counts = np.asarray(self.matrix.T.toarray(), dtype=float).reshape(self.n_users, self.k, self.k)
        row_totals = user_counts.sum(axis=2, keepdims=True)
        valid = row_totals[:, :, 0] > 0
        probabilities = np.divide(user_counts, row_totals, out=np.zeros_like(user_counts), where=row_totals > 0)
        output = np.zeros((self.k, self.k), dtype=float)
        contributing = valid.sum(axis=0).astype(int)
        for state in range(self.k):
            if contributing[state] > 0:
                output[state] = probabilities[valid[:, state], state, :].mean(axis=0)
        return output, contributing


def normalize_transition(counts: np.ndarray) -> np.ndarray:
    array = np.asarray(counts, dtype=float)
    row_sum = array.sum(axis=1, keepdims=True)
    return np.divide(array, row_sum, out=np.zeros_like(array), where=row_sum > 0)


def transition_metrics(empirical: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    first = np.asarray(empirical, dtype=float)
    second = np.asarray(predicted, dtype=float)
    row_tv = 0.5 * np.sum(np.abs(first - second), axis=1)
    diagonal_first = np.diag(first)
    diagonal_second = np.diag(second)
    k = first.shape[0]
    return {
        "transition_mean_row_tv": float(np.mean(row_tv)),
        "transition_max_row_tv": float(np.max(row_tv)),
        "self_transition_corr": pearson(diagonal_first, diagonal_second),
        "self_transition_rmse": float(np.sqrt(np.mean((diagonal_second - diagonal_first) ** 2))),
        "diagonal_dominance_match_fraction": float(np.mean((np.argmax(first, axis=1) == np.arange(k)) == (np.argmax(second, axis=1) == np.arange(k)))),
        "top_transition_edge_overlap": float(np.mean(np.argmax(first, axis=1) == np.argmax(second, axis=1))),
    }


class UserPairMoments:
    def __init__(self, user_id: np.ndarray, first: np.ndarray, second: np.ndarray, user_values: Optional[np.ndarray] = None) -> None:
        users = np.asarray(user_id, dtype=np.int64)
        x = np.asarray(first, dtype=float)
        y = np.asarray(second, dtype=float)
        if user_values is None:
            self.user_values, inverse = np.unique(users, return_inverse=True)
        else:
            self.user_values = np.asarray(user_values, dtype=np.int64)
            position = pd.Series(np.arange(len(self.user_values), dtype=np.int64), index=self.user_values)
            mapped = position.reindex(users).to_numpy(dtype=float)
            if not np.isfinite(mapped).all():
                raise RuntimeError("Moment rows contain users outside the frozen user set.")
            inverse = mapped.astype(np.int64)
        self.n_users = len(self.user_values)
        valid = np.isfinite(x) & np.isfinite(y)
        idx = inverse[valid]
        self.n = np.bincount(idx, minlength=self.n_users).astype(float)
        self.sx = np.bincount(idx, weights=x[valid], minlength=self.n_users).astype(float)
        self.sy = np.bincount(idx, weights=y[valid], minlength=self.n_users).astype(float)
        self.sxx = np.bincount(idx, weights=x[valid] ** 2, minlength=self.n_users).astype(float)
        self.syy = np.bincount(idx, weights=y[valid] ** 2, minlength=self.n_users).astype(float)
        self.sxy = np.bincount(idx, weights=x[valid] * y[valid], minlength=self.n_users).astype(float)
        self.sse = np.bincount(idx, weights=(x[valid] - y[valid]) ** 2, minlength=self.n_users).astype(float)

    def evaluate(self, multipliers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g = np.asarray(multipliers, dtype=float)
        if g.ndim == 1:
            g = g[:, None]
        n = self.n @ g
        sx = self.sx @ g
        sy = self.sy @ g
        sxx = self.sxx @ g
        syy = self.syy @ g
        sxy = self.sxy @ g
        sse = self.sse @ g
        mean_x = sx / np.maximum(n, EPS)
        mean_y = sy / np.maximum(n, EPS)
        covariance = sxy - n * mean_x * mean_y
        variance_x = sxx - n * mean_x * mean_x
        variance_y = syy - n * mean_y * mean_y
        denominator = np.sqrt(np.maximum(variance_x * variance_y, 0.0))
        correlation = np.divide(covariance, denominator, out=np.full_like(covariance, np.nan), where=denominator > EPS)
        rmse = np.sqrt(np.divide(sse, n, out=np.full_like(sse, np.nan), where=n > 0))
        return np.clip(correlation, -1.0, 1.0), rmse


class WeightedResidenceAccumulator:
    def __init__(
        self,
        runs: pd.DataFrame,
        summary: pd.DataFrame,
        transition_accumulator: TransitionAccumulator,
        user_values: np.ndarray,
    ) -> None:
        self.user_values = np.asarray(user_values, dtype=np.int64)
        self.transition_accumulator = transition_accumulator
        self.k = transition_accumulator.k
        position = pd.Series(np.arange(len(self.user_values), dtype=np.int64), index=self.user_values)
        data = runs.copy()
        data["user_id"] = pd.to_numeric(data["user_id"], errors="coerce")
        data["macrostate"] = pd.to_numeric(data["macrostate"], errors="coerce")
        data["length"] = pd.to_numeric(data["length"], errors="coerce")
        data = data.dropna(subset=["user_id", "macrostate", "length"])
        data["user_index"] = position.reindex(data["user_id"].astype(np.int64)).to_numpy(dtype=float)
        data = data[np.isfinite(data["user_index"])].copy()
        data["user_index"] = data["user_index"].astype(np.int64)
        data["macrostate"] = data["macrostate"].astype(np.int64)
        data["length"] = data["length"].astype(np.int64).clip(lower=1)
        summary_table = summary.copy()
        summary_table["macrostate"] = pd.to_numeric(summary_table["macrostate"], errors="coerce")
        summary_table = summary_table.dropna(subset=["macrostate"]).copy()
        summary_table["macrostate"] = summary_table["macrostate"].astype(np.int64)
        summary_index = summary_table.set_index("macrostate")
        self.states: Dict[int, Dict[str, Any]] = {}
        for state in range(self.k):
            if state not in summary_index.index:
                raise RuntimeError(f"Residence summary is missing macrostate {state}.")
            state_runs = data[data["macrostate"] == state]
            tau_value = pd.to_numeric(pd.Series([summary_index.loc[state, "rmst_tau"]]), errors="coerce").iloc[0]
            reference_value = pd.to_numeric(pd.Series([summary_index.loc[state, "reference_length"]]), errors="coerce").iloc[0]
            if not np.isfinite(tau_value) or not np.isfinite(reference_value):
                raise RuntimeError(f"Residence summary has non-finite horizons for macrostate {state}.")
            tau = max(int(tau_value), 1)
            reference = max(int(reference_value), 1)
            observed_tail_value = pd.to_numeric(
                pd.Series([summary_index.loc[state, "observed_tail_probability_at_reference"]]),
                errors="coerce",
            ).iloc[0] if "observed_tail_probability_at_reference" in summary_index.columns else np.nan
            reference_available = bool(np.isfinite(observed_tail_value))
            observed_maximum = int(state_runs["length"].max()) if not state_runs.empty else 1
            maximum = max(tau, observed_maximum, 1)
            lengths = np.minimum(state_runs["length"].to_numpy(dtype=np.int64), maximum)
            users = state_runs["user_index"].to_numpy(dtype=np.int64)
            event_values = state_runs["event_observed"]
            if pd.api.types.is_bool_dtype(event_values) or pd.api.types.is_numeric_dtype(event_values):
                observed = pd.to_numeric(event_values, errors="coerce").fillna(0).astype(bool).to_numpy()
            else:
                observed = (
                    event_values.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
                ).to_numpy()
            total_matrix = sparse.coo_matrix(
                (np.ones(len(state_runs), dtype=float), (lengths - 1, users)),
                shape=(maximum, len(self.user_values)),
            ).tocsr()
            event_matrix = sparse.coo_matrix(
                (np.ones(int(observed.sum()), dtype=float), (lengths[observed] - 1, users[observed])),
                shape=(maximum, len(self.user_values)),
            ).tocsr()
            self.states[state] = {
                "tau": tau,
                "reference": reference,
                "reference_available": reference_available,
                "maximum": maximum,
                "total_matrix": total_matrix,
                "event_matrix": event_matrix,
            }

    def evaluate_chunk(self, multipliers: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
        g = np.asarray(multipliers, dtype=float)
        if g.ndim == 1:
            g = g[:, None]
        transition_counts = self.transition_accumulator.counts(g)
        batch = g.shape[1]
        output: Dict[int, Dict[str, np.ndarray]] = {}
        for state, payload in self.states.items():
            totals = np.asarray(payload["total_matrix"] @ g, dtype=float)
            events = np.asarray(payload["event_matrix"] @ g, dtype=float)
            risk = np.cumsum(totals[::-1], axis=0)[::-1]
            survival = np.ones(batch, dtype=float)
            rmst = np.zeros(batch, dtype=float)
            reference_tail = np.full(batch, np.nan, dtype=float)
            tau = int(payload["tau"])
            reference = int(payload["reference"])
            maximum = int(payload["maximum"])
            for index in range(maximum):
                length = index + 1
                if length <= tau:
                    rmst += survival
                if length == reference:
                    reference_tail = survival.copy()
                valid = risk[index] > 0
                fraction = np.zeros(batch, dtype=float)
                fraction[valid] = events[index, valid] / risk[index, valid]
                survival *= np.maximum(1.0 - fraction, 0.0)
            if (not bool(payload["reference_available"])) or reference > maximum:
                reference_tail = np.full(batch, np.nan, dtype=float)
            p_ii = np.full(batch, np.nan, dtype=float)
            for column in range(batch):
                counts = transition_counts[:, column].reshape(self.k, self.k)
                row_sum = float(np.sum(counts[state]))
                if row_sum > 0:
                    p_ii[column] = counts[state, state] / row_sum
            lengths = np.arange(1, tau + 1, dtype=float)[:, None]
            geometric_rmst = np.nansum(np.power(np.clip(p_ii[None, :], 1e-6, 1.0 - 1e-6), lengths - 1), axis=0)
            geometric_tail = np.power(np.clip(p_ii, 1e-6, 1.0 - 1e-6), reference - 1)
            output[state] = {
                "rmst_lift": np.divide(rmst, geometric_rmst, out=np.full(batch, np.nan), where=geometric_rmst > 0),
                "tail_excess": reference_tail - geometric_tail,
                "self_transition": p_ii,
            }
        return output


def percentile_summary(frame: pd.DataFrame, group_columns: Sequence[str], value_column: str = "value") -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(list(group_columns), dropna=False, sort=False):
        keys = key if isinstance(key, tuple) else (key,)
        values = pd.to_numeric(group[value_column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        row = {column: value for column, value in zip(group_columns, keys)}
        row.update({
            "replicates_finite": int(values.size),
            "bootstrap_mean": float(np.mean(values)) if values.size else np.nan,
            "bootstrap_median": float(np.median(values)) if values.size else np.nan,
            "ci_2p5": float(np.quantile(values, 0.025)) if values.size else np.nan,
            "ci_97p5": float(np.quantile(values, 0.975)) if values.size else np.nan,
            "minimum": float(np.min(values)) if values.size else np.nan,
            "maximum": float(np.max(values)) if values.size else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def stage5_corr_score(value: float) -> float:
    return float(np.clip((float(value) + 1.0) / 2.0, 0.0, 1.0)) if np.isfinite(value) else float("nan")


def stage5_rmse_score(value: float, scale: float) -> float:
    return float(1.0 / (1.0 + max(float(value), 0.0) / float(scale))) if np.isfinite(value) else float("nan")


def stage5_one_minus_score(value: float) -> float:
    return float(np.clip(1.0 - float(value), 0.0, 1.0)) if np.isfinite(value) else float("nan")


def stage5_direct_score(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0)) if np.isfinite(value) else float("nan")


def mean_finite(values: Iterable[float]) -> float:
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(np.mean(array)) if array.size else float("nan")


def stage5_domain_scores(metrics: Mapping[str, float], rmse_scale: float = 0.15) -> Dict[str, float]:
    coordinate = mean_finite([
        stage5_corr_score(float(metrics.get("coordinate_corr_M", np.nan))),
        stage5_corr_score(float(metrics.get("coordinate_corr_Psi", np.nan))),
    ])
    closure = mean_finite([
        stage5_rmse_score(float(metrics.get("one_step_rmse_M", np.nan)), rmse_scale),
        stage5_rmse_score(float(metrics.get("one_step_rmse_Psi", np.nan)), rmse_scale),
    ])
    drift = mean_finite([
        stage5_corr_score(float(metrics.get("learned_plane_drift_vector_corr", np.nan))),
        stage5_corr_score(float(metrics.get("learned_plane_occupancy_weighted_local_drift_cosine", np.nan))),
    ])
    transition = mean_finite([
        stage5_one_minus_score(float(metrics.get("learned_plane_transition_mean_row_tv", np.nan))),
        stage5_corr_score(float(metrics.get("learned_plane_self_transition_corr", np.nan))),
        stage5_direct_score(float(metrics.get("learned_plane_diagonal_dominance_match_fraction", np.nan))),
        stage5_direct_score(float(metrics.get("learned_plane_top_transition_edge_overlap", np.nan))),
    ])
    return {
        "coordinate_score": coordinate,
        "closure_score": closure,
        "drift_score": drift,
        "transition_score": transition,
        "macrostructure_composite_descriptive": mean_finite([coordinate, closure, drift, transition]),
    }


def interior_cell_mask(n_axis: int) -> np.ndarray:
    mask = np.ones((n_axis, n_axis), dtype=bool)
    if n_axis > 2:
        mask[0, :] = False
        mask[-1, :] = False
        mask[:, 0] = False
        mask[:, -1] = False
    return mask


def user_slices(user_id: np.ndarray) -> Sequence[Tuple[int, int]]:
    users = np.asarray(user_id, dtype=np.int64)
    if len(users) == 0:
        return []
    changes = np.flatnonzero(users[1:] != users[:-1]) + 1
    boundaries = np.concatenate([[0], changes, [len(users)]])
    return [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(len(boundaries) - 1)]
