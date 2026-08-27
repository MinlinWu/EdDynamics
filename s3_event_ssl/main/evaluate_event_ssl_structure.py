#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate Event-SSL recovery of the frozen M-Psi effective dynamics.

The script writes predictions, structural metrics and numerical arrays only.
The mesostate partition is the fixed K=6 Stage-1 partition fitted on A_train.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

EPS = 1e-12
GRID_BINS_SIGNED = np.linspace(-1.0, 1.0, 41)
MIN_DRIFT_BIN_COUNT = 30
EXPECTED_MACROSTATE_K = 6
EXPECTED_KMEANS_N_INIT = 20
EXPECTED_KMEANS_FIT_MAX_ROWS = 500000
EXPECTED_RANDOM_STATE = 42
TRAIN_SCRIPT_BASENAME = "train_event_ssl.py"


@dataclass(frozen=True)

class MacroPartition:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    centers: np.ndarray
    standardized_centers: np.ndarray
    k: int
    audit: Mapping[str, Any]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_train_module(path: Path):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Event-SSL training module not found: {path}")
    spec = importlib.util.spec_from_file_location("event_ssl_train", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Event-SSL training module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_train_script(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return Path(__file__).resolve().with_name(TRAIN_SCRIPT_BASENAME)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2, ensure_ascii=False)


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if np.isfinite(x) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    aa = a[ok] - np.mean(a[ok]); bb = b[ok] - np.mean(b[ok])
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / den) if den > EPS else float("nan")


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float).ravel(); q = np.asarray(q, dtype=float).ravel()
    p = p / max(float(np.nansum(p)), EPS); q = q / max(float(np.nansum(q)), EPS)
    m = 0.5 * (p + q)
    def kl(a, b):
        ok = a > 0
        return float(np.sum(a[ok] * np.log((a[ok] + EPS) / (b[ok] + EPS))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def user_balanced_weights(user_id: np.ndarray) -> np.ndarray:
    s = pd.Series(user_id)
    counts = s.groupby(s).transform("count").to_numpy(dtype=float)
    return 1.0 / np.maximum(counts, 1.0)


def digitize(vals: np.ndarray, bins: np.ndarray) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    adjusted = np.where(arr == bins[-1], np.nextafter(bins[-1], bins[0]), arr)
    return np.digitize(adjusted, bins) - 1


def occupancy_grid(x: np.ndarray, y: np.ndarray, user_id: np.ndarray) -> np.ndarray:
    nx = len(GRID_BINS_SIGNED) - 1; ny = nx
    ix = digitize(x, GRID_BINS_SIGNED); iy = digitize(y, GRID_BINS_SIGNED)
    valid = np.isfinite(x) & np.isfinite(y) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    w = user_balanced_weights(user_id)
    flat = ix[valid] * ny + iy[valid]
    H = np.bincount(flat, weights=w[valid], minlength=nx * ny).reshape(nx, ny).astype(float)
    return H / max(float(H.sum()), EPS)


class FieldStats:
    def __init__(self, u, v, mask, count, weight):
        self.u = u; self.v = v; self.mask = mask; self.count = count; self.weight = weight


def field_stats(x: np.ndarray, y: np.ndarray, dx: np.ndarray, dy: np.ndarray, user_id: np.ndarray) -> FieldStats:
    nx = len(GRID_BINS_SIGNED) - 1; ny = nx
    ix = digitize(x, GRID_BINS_SIGNED); iy = digitize(y, GRID_BINS_SIGNED)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(dx) & np.isfinite(dy) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    w = user_balanced_weights(user_id)
    flat = ix[valid] * ny + iy[valid]
    count = np.bincount(flat, minlength=nx * ny).reshape(nx, ny).astype(float)
    weight = np.bincount(flat, weights=w[valid], minlength=nx * ny).reshape(nx, ny).astype(float)
    sx = np.bincount(flat, weights=w[valid] * dx[valid], minlength=nx * ny).reshape(nx, ny).astype(float)
    sy = np.bincount(flat, weights=w[valid] * dy[valid], minlength=nx * ny).reshape(nx, ny).astype(float)
    u = sx / np.maximum(weight, EPS); v = sy / np.maximum(weight, EPS)
    mask = count >= MIN_DRIFT_BIN_COUNT
    return FieldStats(u, v, mask, count, weight)


def vector_corr(f_emp: FieldStats, f_pred: FieldStats) -> float:
    mask = f_emp.mask & f_pred.mask
    if mask.sum() < 3:
        return float("nan")
    a = np.column_stack([f_emp.u[mask].ravel(), f_emp.v[mask].ravel()]).ravel()
    b = np.column_stack([f_pred.u[mask].ravel(), f_pred.v[mask].ravel()]).ravel()
    return pearson(a, b)


def local_cosine(f_emp: FieldStats, f_pred: FieldStats) -> np.ndarray:
    dot = f_emp.u * f_pred.u + f_emp.v * f_pred.v
    se = np.sqrt(f_emp.u * f_emp.u + f_emp.v * f_emp.v)
    sp = np.sqrt(f_pred.u * f_pred.u + f_pred.v * f_pred.v)
    C = dot / np.maximum(se * sp, EPS)
    mask = f_emp.mask & f_pred.mask
    C[~mask] = np.nan
    return np.clip(C, -1.0, 1.0)


def drift_metrics(prefix: str, f_emp: FieldStats, f_pred: FieldStats) -> Dict[str, float]:
    mask = f_emp.mask & f_pred.mask
    if not np.any(mask):
        return {f"{prefix}_common_drift_cells": 0}
    du = f_pred.u[mask] - f_emp.u[mask]
    dv = f_pred.v[mask] - f_emp.v[mask]
    speed_emp = np.sqrt(f_emp.u[mask] ** 2 + f_emp.v[mask] ** 2)
    speed_pred = np.sqrt(f_pred.u[mask] ** 2 + f_pred.v[mask] ** 2)
    cos_map = local_cosine(f_emp, f_pred)
    cos = cos_map[mask]
    w = f_emp.weight[mask].astype(float)
    w = w / max(float(w.sum()), EPS)
    finite_w = f_emp.weight[mask][np.isfinite(f_emp.weight[mask])]
    q75 = np.nanquantile(finite_w, 0.75) if finite_w.size else np.nan
    high = f_emp.weight[mask] >= q75 if np.isfinite(q75) else np.zeros(mask.sum(), dtype=bool)
    resid = np.sqrt(du * du + dv * dv)
    return {
        f"{prefix}_common_drift_cells": int(mask.sum()),
        f"{prefix}_drift_vector_corr": vector_corr(f_emp, f_pred),
        f"{prefix}_drift_local_rmse": float(np.sqrt(np.nanmean(du * du + dv * dv))),
        f"{prefix}_drift_speed_corr": pearson(speed_emp, speed_pred),
        f"{prefix}_mean_local_drift_cosine": float(np.nanmean(cos)),
        f"{prefix}_occupancy_weighted_local_drift_cosine": float(np.nansum(w * cos)),
        f"{prefix}_fraction_cells_cosine_gt_0p8": float(np.nanmean(cos > 0.8)),
        f"{prefix}_high_support_residual_mean": float(np.nanmean(resid[high])) if np.any(high) else float("nan"),
        f"{prefix}_low_support_residual_mean": float(np.nanmean(resid[~high])) if np.any(~high) else float("nan"),
    }


def local_divergence(f: FieldStats) -> np.ndarray:
    """Compute central-difference divergence on fully supported stencils."""
    u = np.asarray(f.u, dtype=float)
    v = np.asarray(f.v, dtype=float)
    mask = np.asarray(f.mask, dtype=bool) & np.isfinite(u) & np.isfinite(v)
    out = np.full_like(u, np.nan, dtype=float)
    if u.shape[0] < 3 or u.shape[1] < 3:
        return out
    interior = np.zeros_like(mask, dtype=bool)
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    centers = 0.5 * (GRID_BINS_SIGNED[:-1] + GRID_BINS_SIGNED[1:])
    dx_den = (centers[2:] - centers[:-2])[:, None]
    dy_den = (centers[2:] - centers[:-2])[None, :]
    d_u_dx = (u[2:, 1:-1] - u[:-2, 1:-1]) / np.maximum(dx_den, EPS)
    d_v_dy = (v[1:-1, 2:] - v[1:-1, :-2]) / np.maximum(dy_den, EPS)
    local = d_u_dx + d_v_dy
    target = out[1:-1, 1:-1]
    supported = interior[1:-1, 1:-1]
    target[supported] = local[supported]
    out[1:-1, 1:-1] = target
    return out


def convergence_metrics(prefix: str, f: FieldStats, reference_center: Tuple[float, float]) -> Dict[str, float]:
    mask = np.asarray(f.mask, dtype=bool) & np.isfinite(f.u) & np.isfinite(f.v)
    if not np.any(mask):
        return {f"{prefix}_negative_divergence_weighted_fraction": float("nan"), f"{prefix}_inward_fraction_to_reference": float("nan")}
    H = f.weight.astype(float)
    div = local_divergence(f)
    div_mask = mask & np.isfinite(div)
    if np.any(div_mask):
        w_div = H[div_mask]
        w_div = w_div / max(float(w_div.sum()), EPS)
        neg = float(np.sum(w_div * (div[div_mask] < 0).astype(float)))
        mean_div = float(np.sum(w_div * div[div_mask]))
    else:
        neg = float("nan")
        mean_div = float("nan")
    w = H[mask]
    w = w / max(float(w.sum()), EPS)
    xc = 0.5 * (GRID_BINS_SIGNED[:-1] + GRID_BINS_SIGNED[1:])
    yc = xc
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    to_x = float(reference_center[0]) - X[mask]
    to_y = float(reference_center[1]) - Y[mask]
    drift_x = f.u[mask]
    drift_y = f.v[mask]
    dist = np.sqrt(to_x * to_x + to_y * to_y)
    speed = np.sqrt(drift_x * drift_x + drift_y * drift_y)
    inward = np.full(mask.sum(), np.nan, dtype=float)
    ok = (dist > EPS) & (speed > EPS)
    inward[ok] = (drift_x[ok] * to_x[ok] + drift_y[ok] * to_y[ok]) / dist[ok]
    frac_in = float(np.nansum(w * (inward > 0).astype(float))) if np.isfinite(inward).any() else float("nan")
    return {
        f"{prefix}_negative_divergence_weighted_fraction": neg,
        f"{prefix}_weighted_mean_local_divergence": mean_div,
        f"{prefix}_inward_fraction_to_reference": frac_in,
        f"{prefix}_weighted_inward_component_mean": float(np.nansum(w * np.where(np.isfinite(inward), inward, 0.0))),
    }


def row_tv(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return 0.5 * np.nansum(np.abs(P - Q), axis=1)


def normalize_transition(C: np.ndarray) -> np.ndarray:
    row = C.sum(axis=1, keepdims=True)
    P = np.zeros_like(C, dtype=float)
    ok = row[:, 0] > 0
    P[ok] = C[ok] / row[ok]
    return P


def transition_counts(cur: np.ndarray, nxt: np.ndarray, k: int) -> np.ndarray:
    C = np.zeros((k, k), dtype=float)
    ok = (cur >= 0) & (cur < k) & (nxt >= 0) & (nxt < k)
    if np.any(ok):
        C += np.bincount(cur[ok] * k + nxt[ok], minlength=k * k).reshape(k, k)
    return C


def transition_metrics(prefix: str, P_emp: np.ndarray, P_pred: np.ndarray) -> Dict[str, float]:
    tv = row_tv(P_emp, P_pred)
    self_emp = np.diag(P_emp); self_pred = np.diag(P_pred)
    k = P_emp.shape[0]
    emp_diag = np.argmax(P_emp, axis=1) == np.arange(k)
    pred_diag = np.argmax(P_pred, axis=1) == np.arange(k)
    return {
        f"{prefix}_transition_mean_row_tv": float(np.nanmean(tv)),
        f"{prefix}_transition_max_row_tv": float(np.nanmax(tv)),
        f"{prefix}_self_transition_rmse": float(np.sqrt(np.nanmean((self_pred - self_emp) ** 2))),
        f"{prefix}_self_transition_mae": float(np.nanmean(np.abs(self_pred - self_emp))),
        f"{prefix}_self_transition_corr": pearson(self_emp, self_pred),
        f"{prefix}_diagonal_dominance_empirical_states": int(np.sum(emp_diag)),
        f"{prefix}_diagonal_dominance_predicted_states": int(np.sum(pred_diag)),
        f"{prefix}_diagonal_dominance_match_fraction": float(np.mean(emp_diag == pred_diag)),
        f"{prefix}_top_transition_edge_overlap": float(np.mean(np.argmax(P_emp, axis=1) == np.argmax(P_pred, axis=1))),
    }


def read_arrays(input_root: Path, split: str) -> dict:
    manifest = load_json(input_root / "metadata" / "stage4_input_manifest.json")
    summary = manifest["split_summaries"][split]
    n = int(summary["rows"]); n_num = int(summary["numeric_shape"][1]); n_cat = int(summary["categorical_shape"][1])
    d = input_root / split
    return {
        "x_num": np.memmap(d / "x_num.float32.mmap", mode="r", dtype=np.float32, shape=(n, n_num)),
        "x_cat": np.memmap(d / "x_cat.int64.mmap", mode="r", dtype=np.int64, shape=(n, n_cat)),
        "y": np.memmap(d / "y_current.float32.mmap", mode="r", dtype=np.float32, shape=(n, 2)),
        "y_next": np.memmap(d / "y_next.float32.mmap", mode="r", dtype=np.float32, shape=(n, 2)),
        "user_id": np.memmap(d / "user_id.int64.mmap", mode="r", dtype=np.int64, shape=(n,)),
        "step": np.memmap(d / "bundle_step_index.int64.mmap", mode="r", dtype=np.int64, shape=(n,)),
        "offsets": np.load(d / "sequence_offsets.npy"),
        "n": n,
    }


def load_model(checkpoint: Path, device: torch.device, train_module):
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    shapes = ckpt["model_shapes"]
    model = train_module.PredictiveStateEventSSL(
        n_num=int(shapes["n_num"]),
        n_cat=int(shapes["n_cat"]),
        hash_buckets=int(shapes["hash_buckets"]),
        hidden_dim=int(cfg["hidden_dim"]),
        input_dim=int(cfg["input_dim"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
        categorical_emb_dim=int(cfg["categorical_emb_dim"]),
        future_steps=tuple(int(value) for value in cfg["future_steps"]),
        delta_scale=float(cfg["delta_scale"]),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, cfg


def table_path(base: Path) -> Path:
    for ext in (".parquet", ".csv.gz", ".csv"):
        path = base.with_suffix(ext)
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find table for {base}")


def load_fixed_k6_partition(stage1_root: Path, input_manifest: Mapping[str, Any]) -> MacroPartition:
    root = Path(stage1_root).resolve() / "dynamics" / "fixed_k6_mesostates"
    metadata_path = root / "fixed_k6_model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Stage-1 fixed-K metadata not found: {metadata_path}")
    metadata = load_json(metadata_path)
    centers_path = table_path(root / "fixed_k6_centers")
    centers_table = read_table(root / "fixed_k6_centers")

    expected_features = ["M_response_prebalanced_pre", "activity_alignment_order_Psi_pre"]
    mapping = metadata.get("raw_to_ordered_label", {})
    labels_expected = list(range(EXPECTED_MACROSTATE_K))
    scaler_mean = np.asarray(metadata.get("scaler_mean", []), dtype=float)
    scaler_scale = np.asarray(metadata.get("scaler_scale", []), dtype=float)
    checks = {
        "coordinate": metadata.get("coordinate") == "MR_PsiA",
        "macrostate_k": int(metadata.get("macrostate_k", -1)) == EXPECTED_MACROSTATE_K,
        "macrostate_k_rule": metadata.get("macrostate_k_rule") == "fixed a priori",
        "features": list(metadata.get("features", [])) == expected_features,
        "fit_split": metadata.get("fit_split") == "A_train",
        "user_balanced_sampling": metadata.get("user_balanced_sampling") is True,
        "user_balanced_kmeans_fit": metadata.get("user_balanced_kmeans_fit") is True,
        "kmeans_n_init": int(metadata.get("kmeans_n_init", -1)) == EXPECTED_KMEANS_N_INIT,
        "fit_max_rows": int(metadata.get("fit_max_rows", -1)) == EXPECTED_KMEANS_FIT_MAX_ROWS,
        "random_state": int(metadata.get("random_state", -1)) == EXPECTED_RANDOM_STATE,
        "scaler": scaler_mean.shape == (2,) and scaler_scale.shape == (2,) and np.isfinite(scaler_mean).all() and np.isfinite(scaler_scale).all() and np.all(scaler_scale > 0),
        "label_mapping": sorted(int(key) for key in mapping) == labels_expected and sorted(int(value) for value in mapping.values()) == labels_expected,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Stage-1 fixed-K contract failed: " + ", ".join(failed))

    required = {"macrostate", "center_M", "center_Psi", "center_M_standardized", "center_Psi_standardized"}
    if not required.issubset(centers_table.columns):
        raise RuntimeError(f"Stage-1 fixed-K centers are missing: {sorted(required.difference(centers_table.columns))}")
    centers_table = centers_table.sort_values("macrostate", kind="mergesort").reset_index(drop=True)
    ids = pd.to_numeric(centers_table["macrostate"], errors="coerce").to_numpy(dtype=float)
    centers = centers_table[["center_M", "center_Psi"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    standardized = centers_table[["center_M_standardized", "center_Psi_standardized"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    if len(centers_table) != EXPECTED_MACROSTATE_K or not np.array_equal(ids, np.arange(EXPECTED_MACROSTATE_K, dtype=float)) or not np.isfinite(centers).all() or not np.isfinite(standardized).all() or not np.array_equal(order, np.arange(EXPECTED_MACROSTATE_K)):
        raise RuntimeError("Stage-1 fixed-K centers are not the expected six ordered states.")

    metadata_sha = file_sha256(metadata_path)
    centers_sha = file_sha256(centers_path)
    prepared_contract = input_manifest.get("stage1_fixed_k6_contract", {})
    if prepared_contract:
        if prepared_contract.get("verified") is not True or int(prepared_contract.get("macrostate_k", -1)) != EXPECTED_MACROSTATE_K:
            raise RuntimeError("Prepared-input fixed-K contract is invalid.")
        if str(prepared_contract.get("metadata_sha256", "")) != metadata_sha:
            raise RuntimeError("Stage-1 fixed-K metadata changed after input preparation.")
        if str(prepared_contract.get("centers_sha256", "")) != centers_sha:
            raise RuntimeError("Stage-1 fixed-K centers changed after input preparation.")

    audit = {
        "source": "frozen Stage-1 fixed K=6 scaler and ordered centers",
        "verified": True,
        "coordinate": "MR_PsiA",
        "macrostate_k": EXPECTED_MACROSTATE_K,
        "macrostate_k_rule": "fixed a priori",
        "fit_split": "A_train",
        "features": expected_features,
        "fit_rows": int(metadata.get("fit_rows", 0)),
        "fit_max_rows": EXPECTED_KMEANS_FIT_MAX_ROWS,
        "user_balanced_sampling": True,
        "user_balanced_kmeans_fit": True,
        "kmeans_n_init": EXPECTED_KMEANS_N_INIT,
        "random_state": EXPECTED_RANDOM_STATE,
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": metadata_sha,
        "centers_path": str(centers_path.resolve()),
        "centers_sha256": centers_sha,
        "kmeans_refit": False,
        "macrostate_k_selected": False,
        "confirmation_data_used_for_partition": False,
    }
    return MacroPartition(
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        centers=centers,
        standardized_centers=standardized,
        k=EXPECTED_MACROSTATE_K,
        audit=audit,
    )


def labels(partition: MacroPartition, xy: np.ndarray) -> np.ndarray:
    values = np.asarray(xy, dtype=float)
    valid = np.isfinite(values).all(axis=1)
    output = np.full(len(values), -1, dtype=np.int64)
    if np.any(valid):
        standardized = (values[valid] - partition.scaler_mean[None, :]) / partition.scaler_scale[None, :]
        distance = np.sum((standardized[:, None, :] - partition.standardized_centers[None, :, :]) ** 2, axis=2)
        output[valid] = np.argmin(distance, axis=1).astype(np.int64)
    return output


@torch.inference_mode()

def predict_split(model, arrays: dict, device: torch.device, chunk_len: int, probe_current=None, probe_next=None, hidden_sample_max: int = 0) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    n = arrays["n"]
    pred_cur = np.zeros((n, 2), dtype=np.float32)
    pred_next = np.zeros((n, 2), dtype=np.float32)
    hidden_current_samples: List[np.ndarray] = []
    hidden_next_samples: List[np.ndarray] = []
    collected = 0
    autocast_enabled = device.type == "cuda"
    for si in range(len(arrays["offsets"]) - 1):
        s = int(arrays["offsets"][si]); e = int(arrays["offsets"][si + 1])
        h_state = None
        prev_hidden = None
        pos = s
        while pos < e:
            end = min(pos + chunk_len, e)
            x_num = torch.from_numpy(np.asarray(arrays["x_num"][pos:end])).unsqueeze(0).to(device, non_blocking=True)
            x_cat = torch.from_numpy(np.asarray(arrays["x_cat"][pos:end])).unsqueeze(0).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                z = model.embed_inputs(x_num, x_cat)
                h_after, h_state = model.rnn(z, h_state)
                first_before = torch.zeros_like(h_after[:, :1, :]) if prev_hidden is None else prev_hidden
                h_before = torch.cat([first_before, h_after[:, :-1, :]], dim=1)
                if probe_current is None:
                    state = model.state_head(h_before)
                    delta = model.delta_scale * model.delta_head(torch.cat([h_before, z], dim=-1))
                    nxt = torch.tanh(state + delta)
                    pred_cur[pos:end] = state.squeeze(0).float().cpu().numpy().astype(np.float32)
                    pred_next[pos:end] = nxt.squeeze(0).float().cpu().numpy().astype(np.float32)
                else:
                    hb = h_before.squeeze(0).float().cpu().numpy().astype(np.float32)
                    ha = h_after.squeeze(0).float().cpu().numpy().astype(np.float32)
                    pred_cur[pos:end] = np.clip(probe_current.predict(hb), -1.0, 1.0).astype(np.float32)
                    pred_next[pos:end] = np.clip(probe_next.predict(ha), -1.0, 1.0).astype(np.float32)
            if hidden_sample_max > 0 and collected < hidden_sample_max:
                hb_np = h_before.squeeze(0).float().cpu().numpy().astype(np.float32)
                ha_np = h_after.squeeze(0).float().cpu().numpy().astype(np.float32)
                take = min(hidden_sample_max - collected, hb_np.shape[0])
                hidden_current_samples.append(hb_np[:take])
                hidden_next_samples.append(ha_np[:take])
                collected += take
            prev_hidden = h_after[:, -1:, :].detach()
            pos = end
    hc = np.concatenate(hidden_current_samples, axis=0)[:hidden_sample_max] if hidden_current_samples else None
    hn = np.concatenate(hidden_next_samples, axis=0)[:hidden_sample_max] if hidden_next_samples else None
    return pred_cur, pred_next, hc, hn


def fit_probes(model, input_root: Path, device: torch.device, max_rows: int, chunk_len: int) -> Tuple[Ridge, Ridge]:
    arrays = read_arrays(input_root, "A_train")
    _, _, hc, hn = predict_split(model, arrays, device, chunk_len=chunk_len, hidden_sample_max=max_rows)
    if hc is None or hn is None:
        raise RuntimeError("Could not collect hidden states for probe fitting.")
    y = np.asarray(arrays["y"][: len(hc)], dtype=np.float32)
    y_next = np.asarray(arrays["y_next"][: len(hn)], dtype=np.float32)
    probe_cur = Ridge(alpha=1.0, fit_intercept=True)
    probe_next = Ridge(alpha=1.0, fit_intercept=True)
    probe_cur.fit(hc, y)
    probe_next.fit(hn, y_next)
    return probe_cur, probe_next


def read_table(base_or_path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    p = Path(base_or_path)
    if p.suffix in {".parquet", ".csv", ".gz"}:
        candidates = [p]
    else:
        candidates = [p.with_suffix(".parquet"), p.with_suffix(".csv.gz"), p.with_suffix(".csv")]
    for cand in candidates:
        if not cand.exists():
            continue
        if cand.suffix == ".parquet":
            return pd.read_parquet(cand, columns=list(columns) if columns is not None else None)
        return pd.read_csv(cand, usecols=list(columns) if columns is not None else None, low_memory=False)
    raise FileNotFoundError(f"Could not find table for {p} with .parquet/.csv.gz/.csv extensions.")


def resolve_stage1_root(input_manifest: Mapping[str, Any], explicit: Optional[Path]) -> Path:
    if explicit is not None:
        root = explicit.resolve()
    else:
        raw = str(input_manifest.get("stage1_root", "") or "").strip()
        if not raw:
            raise RuntimeError(
                "Stage-1 root is absent from the Stage-4 input manifest. "
                "Pass --stage1-root explicitly."
            )
        root = Path(raw).resolve()
    if not (root / "dynamics").exists():
        raise FileNotFoundError(f"Stage-1 root does not contain a dynamics directory: {root}")
    return root


def convergence_reference(input_root: Path, sample_max: int, seed: int) -> Tuple[float, float, Dict[str, float]]:
    arr = read_arrays(input_root, "A_train")
    y = np.asarray(arr["y"], dtype=np.float32)
    y_next = np.asarray(arr["y_next"], dtype=np.float32)
    uid = np.asarray(arr["user_id"], dtype=np.int64)
    if sample_max > 0 and len(y) > sample_max:
        rng = np.random.default_rng(seed + 17)
        idx = np.sort(rng.choice(len(y), size=sample_max, replace=False))
        y = y[idx]; y_next = y_next[idx]; uid = uid[idx]
    H = occupancy_grid(y[:, 0], y[:, 1], uid)
    f = field_stats(y[:, 0], y[:, 1], y_next[:, 0] - y[:, 0], y_next[:, 1] - y[:, 1], uid)
    candidate = f.mask & np.isfinite(f.u) & np.isfinite(f.v)
    if np.any(candidate):
        occ_vals = H[candidate]
        q = np.nanquantile(occ_vals, 0.60) if occ_vals.size else 0.0
        candidate &= H >= q
    if not np.any(candidate):
        candidate = H > 0
    score = np.where(candidate, -np.log(H + EPS), np.inf)
    ix, iy = np.unravel_index(int(np.nanargmin(score)), score.shape)
    centers = 0.5 * (GRID_BINS_SIGNED[:-1] + GRID_BINS_SIGNED[1:])
    ref = (float(centers[ix]), float(centers[iy]))
    meta = {"reference_M": ref[0], "reference_Psi": ref[1], "selection": "A_train high-occupancy supported quasi-potential minimum"}
    return ref[0], ref[1], meta


def metrics_for_predictions(arrays: dict, pred_cur: np.ndarray, pred_next: np.ndarray, partition: MacroPartition, conv_ref: Tuple[float, float]) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    y = np.asarray(arrays["y"], dtype=np.float32)
    y_next = np.asarray(arrays["y_next"], dtype=np.float32)
    uid = np.asarray(arrays["user_id"], dtype=np.int64)
    cur_m, cur_p = y[:, 0], y[:, 1]
    true_next_m, true_next_p = y_next[:, 0], y_next[:, 1]
    pred_m, pred_p = pred_cur[:, 0], pred_cur[:, 1]
    pred_next_m, pred_next_p = pred_next[:, 0], pred_next[:, 1]
    metrics: Dict[str, float] = {}
    for j, name in enumerate(["M", "Psi"]):
        diff = pred_cur[:, j] - y[:, j]
        ndiff = pred_next[:, j] - y_next[:, j]
        metrics[f"coordinate_rmse_{name}"] = float(np.sqrt(np.nanmean(diff * diff)))
        metrics[f"coordinate_mae_{name}"] = float(np.nanmean(np.abs(diff)))
        metrics[f"coordinate_corr_{name}"] = pearson(pred_cur[:, j], y[:, j])
        metrics[f"one_step_rmse_{name}"] = float(np.sqrt(np.nanmean(ndiff * ndiff)))
        metrics[f"one_step_corr_{name}"] = pearson(pred_next[:, j], y_next[:, j])
    H_emp_cur = occupancy_grid(cur_m, cur_p, uid)
    H_pred_cur = occupancy_grid(pred_m, pred_p, uid)
    H_emp_next = occupancy_grid(true_next_m, true_next_p, uid)
    H_pred_next = occupancy_grid(pred_next_m, pred_next_p, uid)
    metrics["current_state_occupancy_js"] = js_divergence(H_emp_cur, H_pred_cur)
    metrics["next_state_occupancy_js"] = js_divergence(H_emp_next, H_pred_next)
    metrics["current_state_occupancy_overlap"] = float(np.minimum(H_emp_cur, H_pred_cur).sum())
    metrics["next_state_occupancy_overlap"] = float(np.minimum(H_emp_next, H_pred_next).sum())

    f_emp = field_stats(cur_m, cur_p, true_next_m - cur_m, true_next_p - cur_p, uid)
    f_anchor = field_stats(cur_m, cur_p, pred_next_m - cur_m, pred_next_p - cur_p, uid)
    f_learned = field_stats(pred_m, pred_p, pred_next_m - pred_m, pred_next_p - pred_p, uid)
    metrics.update(drift_metrics("anchor", f_emp, f_anchor))
    metrics.update(drift_metrics("learned_plane", f_emp, f_learned))
    metrics.update(convergence_metrics("empirical", f_emp, conv_ref))
    metrics.update(convergence_metrics("anchor", f_anchor, conv_ref))
    metrics.update(convergence_metrics("learned_plane", f_learned, conv_ref))
    metrics["convergence_reference_M"] = float(conv_ref[0])
    metrics["convergence_reference_Psi"] = float(conv_ref[1])

    cur_lab_emp = labels(partition, y)
    true_next_lab = labels(partition, y_next)
    pred_cur_lab = labels(partition, pred_cur)
    pred_next_lab = labels(partition, pred_next)
    k = int(partition.k)
    C_emp = transition_counts(cur_lab_emp, true_next_lab, k)
    C_anchor = transition_counts(cur_lab_emp, pred_next_lab, k)
    C_learned = transition_counts(pred_cur_lab, pred_next_lab, k)
    P_emp = normalize_transition(C_emp)
    P_anchor = normalize_transition(C_anchor)
    P_learned = normalize_transition(C_learned)
    metrics["macrostate_k"] = k
    metrics["transition_count"] = int(C_emp.sum())
    metrics.update(transition_metrics("anchor", P_emp, P_anchor))
    metrics.update(transition_metrics("learned_plane", P_emp, P_learned))
    matrices = {
        "P_emp": P_emp, "P_anchor": P_anchor, "P_learned": P_learned,
        "C_emp": C_emp, "C_anchor": C_anchor, "C_learned": C_learned,
        "H_emp_cur": H_emp_cur, "H_pred_cur": H_pred_cur, "H_emp_next": H_emp_next, "H_pred_next": H_pred_next,
        "field_emp_u": f_emp.u, "field_emp_v": f_emp.v, "field_anchor_u": f_anchor.u, "field_anchor_v": f_anchor.v,
        "field_learned_u": f_learned.u, "field_learned_v": f_learned.v,
        "field_emp_mask": f_emp.mask, "field_anchor_mask": f_anchor.mask, "field_learned_mask": f_learned.mask,
        "macrostate_centers": np.asarray(partition.centers, dtype=float),
    }
    return metrics, matrices


def write_table(df: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = base.with_suffix(".parquet")
        df.to_parquet(out, index=False)
        return out
    except Exception:
        out = base.with_suffix(".csv.gz")
        df.to_csv(out, index=False, compression="gzip")
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Event-SSL structural recovery.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-script", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["A_val", "B_confirm"])
    parser.add_argument("--chunk-len", type=int, default=512)
    parser.add_argument("--stage1-root", type=Path, default=None)
    parser.add_argument("--fit-probe-for-pure-ssl", action="store_true")
    parser.add_argument("--probe-max-rows", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-num-threads", type=int, default=0)
    args = parser.parse_args()

    if args.torch_num_threads > 0:
        torch.set_num_threads(int(args.torch_num_threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_script = resolve_train_script(args.train_script)
    train_module = import_train_module(train_script)
    model, cfg = load_model(args.checkpoint, device, train_module)

    args.output_root.mkdir(parents=True, exist_ok=True)
    table_dir = args.output_root / "tables"
    pred_dir = args.output_root / "predictions"
    meta_dir = args.output_root / "metadata"
    for directory in (table_dir, pred_dir, meta_dir):
        directory.mkdir(parents=True, exist_ok=True)

    input_manifest = load_json(args.input_root / "metadata" / "stage4_input_manifest.json")
    if input_manifest.get("primary_coordinates") != ["M", "Psi"]:
        raise RuntimeError("Input manifest primary coordinates are not exactly ['M', 'Psi'].")
    stage1_root = resolve_stage1_root(input_manifest, args.stage1_root)
    partition = load_fixed_k6_partition(stage1_root, input_manifest)
    save_json(dict(partition.audit), meta_dir / "stage4_event_ssl_fixed_k6_partition_audit.json")
    pd.DataFrame({
        "macrostate": np.arange(partition.k, dtype=int),
        "center_M": partition.centers[:, 0],
        "center_Psi": partition.centers[:, 1],
        "center_M_standardized": partition.standardized_centers[:, 0],
        "center_Psi_standardized": partition.standardized_centers[:, 1],
    }).to_csv(table_dir / "stage4_event_ssl_fixed_k6_macrostate_centers.csv", index=False)

    convergence_sample_max = int(partition.audit.get("fit_max_rows", EXPECTED_KMEANS_FIT_MAX_ROWS))
    conv_M, conv_Psi, conv_meta = convergence_reference(args.input_root, convergence_sample_max, args.seed)
    conv_ref = (conv_M, conv_Psi)

    probe_current = probe_next = None
    if args.fit_probe_for_pure_ssl or cfg.get("model_kind") == "pure_ssl":
        print("[Stage4 eval] fitting A_train probes for pure SSL", flush=True)
        probe_current, probe_next = fit_probes(model, args.input_root, device, args.probe_max_rows, args.chunk_len)

    all_metrics: List[Dict[str, Any]] = []
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint.resolve()),
        "train_script": str(train_script),
        "train_script_sha256": file_sha256(train_script),
        "input_root": str(args.input_root.resolve()),
        "model_kind": cfg.get("model_kind"),
        "primary_coordinates": ["M", "Psi"],
        "excluded_coordinate_policy": input_manifest.get("excluded_coordinate_policy"),
        "stage1_root": str(stage1_root),
        "macro_partition": dict(partition.audit),
        "convergence_reference": conv_meta,
        "evaluation_views": {
            "empirical_anchor": "current cell is empirical (M,Psi); predicted drift is pred_next minus empirical current",
            "learned_plane": "current cell is predicted (M,Psi); predicted drift is pred_next minus pred_current",
        },
        "guardrails": {
            "kmeans_refit": False,
            "macrostate_k_selected": False,
            "macrostate_k": EXPECTED_MACROSTATE_K,
            "B_confirm_used_for_update": False,
        },
        "splits": {},
    }

    for split in args.splits:
        print(f"[Stage4 eval] predicting split={split}", flush=True)
        arrays = read_arrays(args.input_root, split)
        pred_current, pred_next, _, _ = predict_split(
            model,
            arrays,
            device,
            args.chunk_len,
            probe_current,
            probe_next,
        )
        metrics, matrices = metrics_for_predictions(arrays, pred_current, pred_next, partition, conv_ref)
        metrics["macrostate_partition_verified_against_stage1_fixed_k6"] = 1.0
        metrics["macrostate_k_fixed_a_priori"] = 1.0
        metrics["split"] = split
        all_metrics.append(metrics)

        predictions = pd.DataFrame({
            "split": split,
            "user_id": np.asarray(arrays["user_id"], dtype=np.int64),
            "bundle_step_index": np.asarray(arrays["step"], dtype=np.int64),
            "M": np.asarray(arrays["y"][:, 0], dtype=np.float32),
            "Psi": np.asarray(arrays["y"][:, 1], dtype=np.float32),
            "target_M_next": np.asarray(arrays["y_next"][:, 0], dtype=np.float32),
            "target_Psi_next": np.asarray(arrays["y_next"][:, 1], dtype=np.float32),
            "pred_M": pred_current[:, 0],
            "pred_Psi": pred_current[:, 1],
            "pred_next_M": pred_next[:, 0],
            "pred_next_Psi": pred_next[:, 1],
        })
        prediction_path = write_table(predictions, pred_dir / f"stage4_event_ssl_predictions_{split}")
        matrix_path = table_dir / f"stage4_event_ssl_transition_matrices_{split}.npz"
        np.savez_compressed(matrix_path, **matrices)
        metrics_path = table_dir / f"stage4_event_ssl_structural_metrics_{split}.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        manifest["splits"][split] = {
            "rows": int(len(predictions)),
            "users": int(predictions["user_id"].nunique()),
            "prediction_path": str(prediction_path),
            "matrix_path": str(matrix_path),
            "metrics_path": str(metrics_path),
            "metrics": metrics,
        }
        print(f"[Stage4 eval] {split} metrics: {metrics}", flush=True)

    pd.DataFrame(all_metrics).to_csv(table_dir / "stage4_event_ssl_structural_metrics_all_splits.csv", index=False)
    manifest["evaluation_boundary"] = "Only M and Psi are loaded as macrostate targets or evaluated coordinates."
    save_json(manifest, meta_dir / "stage4_event_ssl_evaluation_manifest.json")
    print(f"[Stage4 eval] wrote {meta_dir / 'stage4_event_ssl_evaluation_manifest.json'}")


if __name__ == "__main__":
    main()
