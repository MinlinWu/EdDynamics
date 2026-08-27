#!/usr/bin/env python3
"""Render Figure 2 from the frozen Stage-1 empirical-dynamics outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

EPS = 1e-12
COORDINATE = "MR_PsiA"
MACROSTATE_K = 6
XCOL = "M_response_prebalanced_pre"
YCOL = "activity_alignment_order_Psi_pre"
POTENTIAL_CLIP_Q = float(os.environ.get("EDNET_PUB_POTENTIAL_CLIP_Q", "0.98"))
DRIFT_ARROW_GAIN = float(os.environ.get("EDNET_PUB_DRIFT_ARROW_GAIN", "1.0"))
MAX_RESIDENCE_LEN_FOR_PLOT = int(os.environ.get("EDNET_PUB_MAX_RESIDENCE_LEN", "10000"))
MAX_REPRO_POINTS_PER_PANEL = int(os.environ.get("EDNET_PUB_REPRO_POINTS_PER_PANEL", "2500"))

BLUE_CMAP = LinearSegmentedColormap.from_list(
    "ednet_light_to_deep_blue",
    ["#f7fbff", "#deebf7", "#9ecae1", "#4292c6", "#08519c", "#08306b"],
)
BLUE_CMAP_REVERSED = LinearSegmentedColormap.from_list(
    "ednet_deep_to_light_blue",
    ["#08306b", "#08519c", "#4292c6", "#9ecae1", "#deebf7", "#f7fbff"],
)


@dataclass(frozen=True)
class FieldGrid:
    xbins: np.ndarray
    ybins: np.ndarray
    xcenters: np.ndarray
    ycenters: np.ndarray
    occupancy_probability: np.ndarray
    potential: np.ndarray
    drift_u: np.ndarray
    drift_v: np.ndarray
    drift_mask: np.ndarray


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
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False)


def read_table(base: Path) -> pd.DataFrame:
    for path in (base.with_suffix(".parquet"), base.with_suffix(".csv.gz"), base.with_suffix(".csv")):
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path, low_memory=False)
    raise FileNotFoundError(f"Could not find table for {base}")


def write_table(frame: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = base.with_suffix(".parquet")
        frame.to_parquet(path, index=False)
    except Exception:
        path = base.with_suffix(".csv.gz")
        frame.to_csv(path, index=False, compression="gzip")
    return path


def resolve_stage1_root(path: Path) -> Path:
    for candidate in (path, path / "stage1"):
        if (candidate / "dynamics").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"No Stage-1 dynamics directory found under {path}")


def require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")


def edges_from_centers(centers: np.ndarray) -> np.ndarray:
    values = np.asarray(centers, dtype=float)
    if values.size < 2:
        raise ValueError("At least two grid centers are required")
    middle = 0.5 * (values[:-1] + values[1:])
    first = values[0] - 0.5 * (values[1] - values[0])
    last = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate([[first], middle, [last]])


def load_field_grid(base: Path) -> FieldGrid:
    frame = read_table(base)
    required = [
        "x_bin", "y_bin", "M_center", "Psi_center", "occupancy_probability",
        "empirical_quasi_potential", "drift_M", "drift_Psi", "drift_supported",
    ]
    require_columns(frame, required, base.name)
    xcenters = np.sort(pd.to_numeric(frame["M_center"], errors="raise").unique())
    ycenters = np.sort(pd.to_numeric(frame["Psi_center"], errors="raise").unique())
    nx, ny = len(xcenters), len(ycenters)
    if len(frame) != nx * ny:
        raise RuntimeError(f"{base.name} is not a complete rectangular grid")
    ordered = frame.sort_values(["x_bin", "y_bin"], kind="mergesort")

    def matrix(column: str, dtype=float) -> np.ndarray:
        return ordered[column].to_numpy(dtype=dtype).reshape(nx, ny)

    return FieldGrid(
        xbins=edges_from_centers(xcenters),
        ybins=edges_from_centers(ycenters),
        xcenters=xcenters,
        ycenters=ycenters,
        occupancy_probability=matrix("occupancy_probability"),
        potential=matrix("empirical_quasi_potential"),
        drift_u=matrix("drift_M"),
        drift_v=matrix("drift_Psi"),
        drift_mask=matrix("drift_supported", dtype=bool),
    )


def load_matrix(base: Path) -> np.ndarray:
    frame = read_table(base)
    matrix = frame.to_numpy(dtype=float)
    if matrix.shape != (MACROSTATE_K, MACROSTATE_K):
        raise RuntimeError(f"{base.name} must be a 6 by 6 matrix")
    return matrix


def validate_kmeans_contract(metadata: Mapping[str, Any], centres: pd.DataFrame) -> None:
    expected = {
        "coordinate": COORDINATE,
        "macrostate_k": MACROSTATE_K,
        "macrostate_k_rule": "fixed a priori",
        "fit_split": "A_train",
        "features": [XCOL, YCOL],
        "user_balanced_sampling": True,
        "user_balanced_kmeans_fit": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"KMeans contract mismatch for {key}: {metadata.get(key)!r} != {value!r}")
    if int(metadata.get("kmeans_n_init", -1)) != 20:
        raise RuntimeError("KMeans n_init must be 20")
    if int(metadata.get("fit_max_rows", -1)) != 500000:
        raise RuntimeError("KMeans fit_max_rows must be 500000")
    if int(metadata.get("random_state", -1)) != 42:
        raise RuntimeError("KMeans random_state must be 42")
    require_columns(centres, ["macrostate", "center_M", "center_Psi"], "fixed K=6 centres")
    states = pd.to_numeric(centres["macrostate"], errors="raise").astype(int).to_numpy()
    if len(centres) != MACROSTATE_K or not np.array_equal(np.sort(states), np.arange(MACROSTATE_K)):
        raise RuntimeError("Fixed K=6 centres do not contain states 0--5 exactly once")


def clip_for_potential(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    finite = output[np.isfinite(output)]
    if finite.size:
        cap = float(np.nanquantile(finite, min(max(POTENTIAL_CLIP_Q, 0.50), 1.0)))
        output = np.minimum(output, cap)
    return output


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return np.nan
    left = left[mask] - float(np.mean(left[mask]))
    right = right[mask] - float(np.mean(right[mask]))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > EPS else np.nan


def cosine_safe(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 2:
        return np.nan
    denominator = float(np.linalg.norm(left[mask]) * np.linalg.norm(right[mask]))
    return float(np.dot(left[mask], right[mask]) / denominator) if denominator > EPS else np.nan


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_potential_ax(ax, field: FieldGrid, title: str, cmap=BLUE_CMAP, add_colorbar: bool = True) -> None:
    mesh = ax.pcolormesh(field.xbins, field.ybins, clip_for_potential(field.potential).T, shading="auto", cmap=cmap)
    if add_colorbar:
        colorbar = ax.figure.colorbar(mesh, ax=ax, pad=0.02)
        colorbar.set_label(r"empirical quasi-potential $U=-\log p(x)$")
    ax.set_xlabel("Response evidence order $M$")
    ax.set_ylabel(r"Exposure-alignment order $\Psi$")
    ax.set_title(title)
    ax.set_xlim(field.xbins[0], field.xbins[-1])
    ax.set_ylim(field.ybins[0], field.ybins[-1])
    ax.tick_params(direction="out")


def draw_drift_field_ax(
    ax,
    field: FieldGrid,
    title: str,
    cmap=BLUE_CMAP,
    add_colorbar: bool = True,
    invert_colorbar: bool = False,
) -> None:
    mesh = ax.pcolormesh(
        field.xbins,
        field.ybins,
        clip_for_potential(field.potential).T,
        shading="auto",
        cmap=cmap,
        alpha=0.62,
    )
    if add_colorbar:
        colorbar = ax.figure.colorbar(mesh, ax=ax, pad=0.02)
        colorbar.set_label(r"$U=-\log p(x)$")
        if invert_colorbar:
            colorbar.ax.invert_yaxis()
    if np.any(field.drift_mask):
        xgrid, ygrid = np.meshgrid(field.xcenters, field.ycenters, indexing="ij")
        ax.quiver(
            xgrid[field.drift_mask],
            ygrid[field.drift_mask],
            DRIFT_ARROW_GAIN * field.drift_u[field.drift_mask],
            DRIFT_ARROW_GAIN * field.drift_v[field.drift_mask],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0026,
            color="#08306b",
            alpha=0.92,
        )
    else:
        ax.text(0.5, 0.5, "No supported drift bins", ha="center", va="center", transform=ax.transAxes)
        warnings.warn(f"No supported drift bins for {title}")
    ax.set_xlabel("Response evidence order $M$")
    ax.set_ylabel(r"Exposure-alignment order $\Psi$")
    ax.set_title(title)
    ax.set_xlim(field.xbins[0], field.xbins[-1])
    ax.set_ylim(field.ybins[0], field.ybins[-1])
    ax.tick_params(direction="out")


def draw_transition_matrix_ax(ax, matrix: np.ndarray, title: str, cmap=BLUE_CMAP, add_colorbar: bool = True) -> None:
    maximum = max(1.0, float(np.nanmax(matrix))) if np.isfinite(matrix).any() else 1.0
    mesh = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=0.0, vmax=maximum)
    if add_colorbar:
        colorbar = ax.figure.colorbar(mesh, ax=ax, pad=0.02)
        colorbar.set_label("transition probability")
    labels = [f"S{state}" for state in range(matrix.shape[0])]
    ax.set_xlabel("Next macrostate")
    ax.set_ylabel("Current macrostate")
    ax.set_xticks(np.arange(matrix.shape[0]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    threshold = 0.55 * float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 0.5
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] >= threshold else "#08306b"
            ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=7, color=color)


def draw_residence_vs_geometric_ax(ax, curves: pd.DataFrame, title: str, max_len: int) -> None:
    if curves.empty:
        ax.text(0.5, 0.5, "No residence statistics", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf"]
    positive = []
    plotted = 0
    for state, group in curves.groupby("macrostate", sort=True):
        ordered = group.sort_values("residence_length")
        x = pd.to_numeric(ordered["residence_length"], errors="coerce").to_numpy(dtype=int)
        km = pd.to_numeric(ordered["km_ccdf"], errors="coerce").to_numpy(dtype=float)
        geometric = pd.to_numeric(ordered["geometric_ccdf"], errors="coerce").to_numpy(dtype=float)
        keep = x <= int(max_len)
        x = x[keep]
        km = km[keep]
        geometric = geometric[keep]
        if x.size == 0:
            continue
        color = colors[int(state) % len(colors)]
        ax.step(x, np.where(km > 0, km, np.nan), where="post", linewidth=1.8, color=color, label=f"S{int(state)} KM")
        ax.plot(x, np.where(geometric > 0, geometric, np.nan), linestyle="--", linewidth=1.55, color=color, label=f"S{int(state)} geom")
        positive.extend(km[km > 0].tolist())
        positive.extend(geometric[geometric > 0].tolist())
        plotted += 1
    if plotted == 0:
        ax.text(0.5, 0.5, "No residence statistics", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, max_len)
    if positive:
        ax.set_ylim(max(1e-5, min(positive) * 0.75), 1.05)
    ticks = [tick for tick in [1, 3, 10, 30, 100, 300, 1000, 3000, 10000] if tick <= max_len]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks])
    ax.set_xlabel("Residence length in submitted-bundle steps")
    ax.set_ylabel("Kaplan–Meier CCDF")
    ax.set_title(title)
    ax.grid(True, which="major", alpha=0.20)
    ax.legend(fontsize=7, ncol=2, frameon=False)


def ranked_delta_points(
    reference: np.ndarray,
    delta: np.ndarray,
    *,
    x_label: str,
    x_scale: str = "linear",
    xlim: Tuple[float, float] = (0.0, 1.0),
    use_reference_as_x: bool = False,
) -> pd.DataFrame:
    ref = np.asarray(reference, dtype=float).ravel()
    diff = np.asarray(delta, dtype=float).ravel()
    columns = ["x_rank_train", "x_value", "delta_val_minus_train", "train_reference", "x_axis_label", "x_scale", "xlim_left", "xlim_right"]
    if ref.size != diff.size or ref.size == 0:
        return pd.DataFrame(columns=columns)
    keep = np.isfinite(ref) & np.isfinite(diff)
    ref = ref[keep]
    diff = diff[keep]
    if ref.size == 0:
        return pd.DataFrame(columns=columns)
    order = np.argsort(ref, kind="mergesort")
    ref = ref[order]
    diff = diff[order]
    if ref.size > MAX_REPRO_POINTS_PER_PANEL:
        indices = np.unique(np.linspace(0, ref.size - 1, MAX_REPRO_POINTS_PER_PANEL, dtype=int))
        ref = ref[indices]
        diff = diff[indices]
    rank = np.linspace(0.0, 1.0, ref.size, dtype=float) if ref.size > 1 else np.asarray([0.5])
    return pd.DataFrame({
        "x_rank_train": rank,
        "x_value": ref if use_reference_as_x else rank,
        "delta_val_minus_train": diff,
        "train_reference": ref,
        "x_axis_label": x_label,
        "x_scale": x_scale,
        "xlim_left": float(xlim[0]),
        "xlim_right": float(xlim[1]),
    })


def transition_delta_points(train: np.ndarray, validation: np.ndarray) -> pd.DataFrame:
    keep = np.isfinite(train) & np.isfinite(validation)
    return ranked_delta_points(
        train[keep],
        validation[keep] - train[keep],
        x_label="Transition probability",
        xlim=(0.0, 1.0),
        use_reference_as_x=True,
    )


def residence_delta_points(train: pd.DataFrame, validation: pd.DataFrame, max_len: int) -> pd.DataFrame:
    columns = ["x_rank_train", "x_value", "delta_val_minus_train", "train_reference", "x_axis_label", "x_scale", "xlim_left", "xlim_right"]
    states = sorted(set(pd.to_numeric(train["macrostate"], errors="coerce").dropna().astype(int)) & set(pd.to_numeric(validation["macrostate"], errors="coerce").dropna().astype(int)))
    x_grid = np.unique(np.rint(np.geomspace(1, max_len, num=90)).astype(int))
    x_values, references, deltas = [], [], []
    for state in states:
        train_state = train[pd.to_numeric(train["macrostate"], errors="coerce") == state].set_index("residence_length")
        val_state = validation[pd.to_numeric(validation["macrostate"], errors="coerce") == state].set_index("residence_length")
        common = x_grid[np.isin(x_grid, train_state.index) & np.isin(x_grid, val_state.index)]
        if common.size == 0:
            continue
        train_ccdf = pd.to_numeric(train_state.loc[common, "km_ccdf"], errors="coerce").to_numpy(dtype=float)
        val_ccdf = pd.to_numeric(val_state.loc[common, "km_ccdf"], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(train_ccdf) & np.isfinite(val_ccdf) & ((train_ccdf > 0) | (val_ccdf > 0))
        x_values.extend(common[keep].astype(float).tolist())
        references.extend(train_ccdf[keep].tolist())
        deltas.extend((val_ccdf[keep] - train_ccdf[keep]).tolist())
    if not x_values:
        return pd.DataFrame(columns=columns)
    x_array = np.asarray(x_values, dtype=float)
    reference_array = np.asarray(references, dtype=float)
    delta_array = np.asarray(deltas, dtype=float)
    order = np.lexsort((reference_array, x_array))
    rank = np.linspace(0.0, 1.0, len(order), dtype=float) if len(order) > 1 else np.asarray([0.5])
    return pd.DataFrame({
        "x_rank_train": rank,
        "x_value": x_array[order],
        "delta_val_minus_train": delta_array[order],
        "train_reference": reference_array[order],
        "x_axis_label": "Residence length (steps)",
        "x_scale": "log",
        "xlim_left": 1.0,
        "xlim_right": float(max_len),
    })


def build_reproducibility_points(
    train_field: FieldGrid,
    val_field: FieldGrid,
    train_transition: np.ndarray,
    val_transition: np.ndarray,
    train_curves: pd.DataFrame,
    val_curves: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    keep_occupancy = np.isfinite(train_field.occupancy_probability) & np.isfinite(val_field.occupancy_probability) & ((train_field.occupancy_probability > EPS) | (val_field.occupancy_probability > EPS))
    occupancy = ranked_delta_points(
        train_field.occupancy_probability[keep_occupancy],
        val_field.occupancy_probability[keep_occupancy] - train_field.occupancy_probability[keep_occupancy],
        x_label="Occupancy quantile",
    )
    common = train_field.drift_mask & val_field.drift_mask
    if np.any(common):
        train_u = train_field.drift_u[common]
        train_v = train_field.drift_v[common]
        val_u = val_field.drift_u[common]
        val_v = val_field.drift_v[common]
        train_speed = np.sqrt(train_u * train_u + train_v * train_v)
        val_speed = np.sqrt(val_u * val_u + val_v * val_v)
        local_cosine = np.clip((train_u * val_u + train_v * val_v) / np.maximum(train_speed * val_speed, EPS), -1.0, 1.0)
        keep = np.isfinite(train_speed) & np.isfinite(local_cosine) & (train_speed > EPS) & (val_speed > EPS)
        drift = ranked_delta_points(train_speed[keep], 0.5 * (local_cosine[keep] + 1.0) - 1.0, x_label="Drift-magnitude quantile")
    else:
        drift = pd.DataFrame()
    return {
        "Occupancy landscape": occupancy,
        "Drift-direction agreement": drift,
        "Transition matrix": transition_delta_points(train_transition, val_transition),
        "Residence-time agreement": residence_delta_points(train_curves, val_curves, MAX_RESIDENCE_LEN_FOR_PLOT),
    }


def build_reproducibility_summary(
    train_field: FieldGrid,
    val_field: FieldGrid,
    train_transition: np.ndarray,
    val_transition: np.ndarray,
    train_summary: pd.DataFrame,
    val_summary: pd.DataFrame,
    stage1_summary: pd.Series,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    js = float(stage1_summary["occupancy_js_divergence"])
    local_cosine = float(stage1_summary["mean_local_drift_cosine"])
    row_tv = float(stage1_summary["transition_mean_row_total_variation"])
    residence_log = float(stage1_summary["residence_tail_ratio_mean_abs_log_ratio"])
    common = train_field.drift_mask & val_field.drift_mask
    global_cosine = cosine_safe(
        np.concatenate([train_field.drift_u[common], train_field.drift_v[common]]),
        np.concatenate([val_field.drift_u[common], val_field.drift_v[common]]),
    ) if np.any(common) else np.nan
    summary = {
        "A_train_A_val_occupancy_js": js,
        "A_train_A_val_log_occupancy_pearson": pearson_safe(np.log10(train_field.occupancy_probability.ravel() + EPS), np.log10(val_field.occupancy_probability.ravel() + EPS)),
        "A_train_A_val_occupancy_similarity": float(np.clip(1.0 - js / math.log(2.0), 0.0, 1.0)),
        "A_train_A_val_drift_cosine_common_bins": global_cosine,
        "A_train_A_val_mean_local_drift_cosine": local_cosine,
        "A_train_A_val_drift_agreement_score": float(np.clip(0.5 * (global_cosine + 1.0), 0.0, 1.0)) if np.isfinite(global_cosine) else np.nan,
        "common_drift_bins": int(stage1_summary["common_supported_drift_cells"]),
        "A_train_A_val_transition_diag_pearson": pearson_safe(np.diag(train_transition), np.diag(val_transition)),
        "A_train_A_val_transition_mean_row_total_variation": row_tv,
        "A_train_A_val_transition_similarity": float(np.clip(1.0 - row_tv, 0.0, 1.0)),
        "A_train_A_val_residence_tail_ratio_pearson": pearson_safe(pd.to_numeric(train_summary["tail_ratio_at_reference"], errors="coerce"), pd.to_numeric(val_summary["tail_ratio_at_reference"], errors="coerce")),
        "A_train_A_val_residence_mean_abs_log_tail_ratio": residence_log,
        "A_train_A_val_residence_reference_length": int(pd.to_numeric(val_summary["reference_length"], errors="coerce").dropna().iloc[0]),
        "A_train_A_val_residence_lift_pearson": pearson_safe(pd.to_numeric(train_summary["tail_ratio_at_reference"], errors="coerce"), pd.to_numeric(val_summary["tail_ratio_at_reference"], errors="coerce")),
        "A_train_A_val_residence_mean_abs_log_lift_ratio": residence_log,
        "A_train_A_val_residence_concordance": float(np.clip(math.exp(-residence_log), 0.0, 1.0)),
    }
    rows = [
        {
            "metric": "Occupancy landscape",
            "A_train_reference": 1.0,
            "A_val_relative_score": summary["A_train_A_val_occupancy_similarity"],
            "raw_metric": js,
            "raw_metric_name": "Jensen-Shannon divergence; score = 1 - JS/log(2)",
            "detail": rf"$D_{{\mathrm{{JS}}}}(p_{{\mathrm{{tr}}}},p_{{\mathrm{{val}}}})={js:.3f}$",
        },
        {
            "metric": "Drift-direction agreement",
            "A_train_reference": 1.0,
            "A_val_relative_score": summary["A_train_A_val_drift_agreement_score"],
            "raw_metric": local_cosine,
            "raw_metric_name": "mean local cosine similarity over common supported drift bins",
            "detail": rf"$\langle\cos(\Delta x_{{\mathrm{{tr}}}},\Delta x_{{\mathrm{{val}}}})\rangle={local_cosine:.2f}$",
        },
        {
            "metric": "Transition matrix",
            "A_train_reference": 1.0,
            "A_val_relative_score": summary["A_train_A_val_transition_similarity"],
            "raw_metric": row_tv,
            "raw_metric_name": "mean row total variation; score = 1 - mean row TV",
            "detail": rf"$K^{{-1}}\sum_i\mathrm{{TV}}(P^{{\mathrm{{tr}}}}_i,P^{{\mathrm{{val}}}}_i)={row_tv:.3f}$",
        },
        {
            "metric": "Residence-time agreement",
            "A_train_reference": 1.0,
            "A_val_relative_score": summary["A_train_A_val_residence_concordance"],
            "raw_metric": residence_log,
            "raw_metric_name": "mean |log(A_val tail ratio / A_train tail ratio)| at the fixed reference length",
            "detail": rf"$\langle|\log(L^{{\mathrm{{val}}}}_i/L^{{\mathrm{{tr}}}}_i)|\rangle={residence_log:.3f}$",
        },
    ]
    return summary, pd.DataFrame(rows)


def axis_metadata(points: pd.DataFrame, metric: str) -> Tuple[str, str, Tuple[float, float]]:
    if points is not None and not points.empty:
        return (
            str(points["x_axis_label"].iloc[0]),
            str(points["x_scale"].iloc[0]),
            (float(points["xlim_left"].iloc[0]), float(points["xlim_right"].iloc[0])),
        )
    if metric == "Residence-time agreement":
        return "Residence length (steps)", "log", (1.0, float(MAX_RESIDENCE_LEN_FOR_PLOT))
    if metric == "Transition matrix":
        return "Transition probability", "linear", (0.0, 1.0)
    if metric == "Drift-direction agreement":
        return "Drift-speed quantile", "linear", (0.0, 1.0)
    return "Occupancy quantile", "linear", (0.0, 1.0)


def draw_reproducibility_panel_ax(ax, metric: str, detail: str, points: Optional[pd.DataFrame], show_legend: bool) -> None:
    data = points.copy() if points is not None else pd.DataFrame()
    x_label, x_scale, xlim = axis_metadata(data, metric)
    x_column = "x_value" if "x_value" in data.columns else "x_rank_train"
    if not data.empty:
        data[x_column] = pd.to_numeric(data[x_column], errors="coerce")
        data["delta_val_minus_train"] = pd.to_numeric(data["delta_val_minus_train"], errors="coerce")
        data = data[np.isfinite(data[x_column]) & np.isfinite(data["delta_val_minus_train"])]
        if x_scale == "log":
            data = data[data[x_column] > 0]
    yvalues = data["delta_val_minus_train"].to_numpy(dtype=float) if not data.empty else np.asarray([])
    if metric == "Drift-direction agreement":
        limit = 0.50
    elif metric == "Transition matrix":
        limit = 0.05
    else:
        maximum = max(float(np.nanquantile(np.abs(yvalues), 0.98)), float(np.nanmax(np.abs(yvalues))) * 0.35, 1e-3) if yvalues.size else 0.05
        limit = max(0.05, min(1.0, maximum * 1.35))
    ax.set_xscale(x_scale)
    ax.set_xlim(*xlim)
    ax.set_ylim(-limit, limit)
    ax.set_yticks([-limit, 0.0, limit])
    ax.set_yticklabels([f"{-limit:.2f}", "0.00", f"{limit:.2f}"])
    if x_scale == "log":
        ticks = [tick for tick in [1, 3, 10, 30, 100, 300, 1000, 3000, 10000] if xlim[0] <= tick <= xlim[1]]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(tick) for tick in ticks])
    else:
        ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_xlabel(x_label, fontsize=8.2)
    ax.grid(axis="x", alpha=0.15)
    ax.grid(axis="y", alpha=0.10)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.95)
        spine.set_edgecolor("#6b7280")
    ax.axhspan(-0.12 * limit, 0.12 * limit, color="#deebf7", alpha=0.52, zorder=0)
    ax.axhline(0.0, color="#08306b", linestyle="--", linewidth=1.15, alpha=0.88, zorder=2)
    if not data.empty:
        xvalues = data[x_column].to_numpy(dtype=float)
        yplot = data["delta_val_minus_train"].to_numpy(dtype=float)
        ax.scatter(xvalues, yplot, s=12, color="#08306b", alpha=0.38, edgecolors="none", rasterized=True, zorder=3)
        med_x, med_y = [], []
        bins = np.geomspace(max(xlim[0], 1e-9), max(xlim[1], max(xlim[0], 1e-9) * 1.01), 21) if x_scale == "log" else np.linspace(xlim[0], xlim[1], 21)
        for left, right in zip(bins[:-1], bins[1:]):
            mask = (xvalues >= left) & (xvalues < right if right < xlim[1] else xvalues <= right)
            if np.any(mask):
                med_x.append(float(np.sqrt(left * right)) if x_scale == "log" else 0.5 * (left + right))
                med_y.append(float(np.nanmedian(yplot[mask])))
        if len(med_x) >= 2:
            ax.plot(med_x, med_y, color="#08306b", linewidth=1.05, alpha=0.82, zorder=4)
    else:
        ax.text(0.50, 0.0, "No matched validation fluctuations", ha="center", va="center", fontsize=8.5, color="0.35", transform=ax.get_yaxis_transform())
    ax.text(0.01, 0.86, metric, transform=ax.transAxes, ha="left", va="center", fontsize=10.2, fontweight="semibold", color="#0f172a")
    if detail and detail != "nan":
        ax.text(0.01, 0.08, detail, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, color="0.30")
    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#08306b", markeredgecolor="none", markersize=5.5, alpha=0.7, label="Validation set"),
            Line2D([0], [0], color="#08306b", linestyle="--", linewidth=1.15, label="Training set"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7.8, handlelength=1.6, borderaxespad=0.25, ncol=2, columnspacing=0.9, handletextpad=0.4)


def render_reproducibility_stack(axes: Sequence, metric_rows: pd.DataFrame, points: Mapping[str, pd.DataFrame], title: str) -> None:
    order = ["Occupancy landscape", "Drift-direction agreement", "Transition matrix", "Residence-time agreement"]
    rows = {str(row.metric): row for row in metric_rows.itertuples(index=False)}
    for index, (axis, metric) in enumerate(zip(axes, order)):
        row = rows.get(metric)
        draw_reproducibility_panel_ax(axis, metric, str(row.detail) if row is not None else "", points.get(metric), index == 0)
    axes[0].set_title(title, fontsize=12)
    axes[1].set_ylabel("Validation − training", fontsize=9.0)


def centers_for_publication(centres: pd.DataFrame) -> pd.DataFrame:
    output = centres.rename(columns={"center_M": "center_x", "center_Psi": "center_y"}).copy()
    output["x_coordinate"] = XCOL
    output["y_coordinate"] = YCOL
    output["coordinate"] = COORDINATE
    xvalues = pd.to_numeric(output["center_x"], errors="coerce").to_numpy(dtype=float)
    yvalues = pd.to_numeric(output["center_y"], errors="coerce").to_numpy(dtype=float)
    xcuts = np.quantile(xvalues, [1 / 3, 2 / 3])
    ycuts = np.quantile(yvalues, [1 / 3, 2 / 3])

    def band(value: float, cuts: np.ndarray) -> str:
        return "low" if value <= cuts[0] else "middle" if value <= cuts[1] else "high"

    output["center_M_R_signed_band"] = ["negative M_R" if value < -0.15 else "positive M_R" if value > 0.15 else "near-zero M_R" for value in xvalues]
    output["center_Psi_relative_band"] = [band(value, ycuts) for value in yvalues]
    output["center_M_R_relative_band"] = [band(value, xcuts) for value in xvalues]
    output["state_label"] = [f"S{int(state)}" for state in output["macrostate"]]
    output["state_description"] = [
        f"S{int(state)}: {band(x, xcuts)} M_R / {band(y, ycuts)} Psi macrostate center at ({x:.3f}, {y:.3f}); coordinate-defined, not a fixed learning-stage label"
        for state, x, y in zip(output["macrostate"], xvalues, yvalues)
    ]
    columns = [
        "macrostate", "center_x", "center_y", "x_coordinate", "y_coordinate",
        "coordinate", "center_M_R_signed_band", "center_Psi_relative_band",
        "center_M_R_relative_band", "state_label", "state_description",
    ]
    return output[columns]


def parse_args() -> argparse.Namespace:
    default_root = Path(os.environ.get("EDNET_KT4_OUTPUT_ROOT", "/data/datasets/KT4/outputs_KT4")) / "stage1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, default=Path(os.environ.get("EDNET_STAGE1_ROOT", str(default_root))))
    parser.add_argument("--out-subdir", type=str, default="figures_publication_empirical_evidence")
    parser.add_argument("--make-b-confirm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage1_root = resolve_stage1_root(args.stage1_root)
    coordinate_root = stage1_root / "dynamics" / "coordinate_analysis" / COORDINATE
    mesostate_root = stage1_root / "dynamics" / "fixed_k6_mesostates"
    figure_root = stage1_root / args.out_subdir
    table_root = figure_root / "tables"
    figure_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)

    metadata = load_json(mesostate_root / "fixed_k6_model_metadata.json")
    centres = read_table(mesostate_root / "fixed_k6_centers")
    validate_kmeans_contract(metadata, centres)

    train_field = load_field_grid(coordinate_root / "A_train_publication_field_grid")
    val_field = load_field_grid(coordinate_root / "A_val_publication_field_grid")
    train_transition = load_matrix(mesostate_root / "A_train_fixed_k6_transition_matrix")
    val_transition = load_matrix(mesostate_root / "A_val_fixed_k6_transition_matrix")
    train_curves = read_table(mesostate_root / "A_train_fixed_k6_residence_curves")
    val_curves = read_table(mesostate_root / "A_val_fixed_k6_residence_curves")
    train_residence = read_table(mesostate_root / "A_train_fixed_k6_residence_summary")
    val_residence = read_table(mesostate_root / "A_val_fixed_k6_residence_summary")
    stage1_repro = read_table(mesostate_root / "A_train_A_val_reproducibility_summary").iloc[0]

    summary, metric_rows = build_reproducibility_summary(
        train_field,
        val_field,
        train_transition,
        val_transition,
        train_residence,
        val_residence,
        stage1_repro,
    )
    points = build_reproducibility_points(
        train_field,
        val_field,
        train_transition,
        val_transition,
        train_curves,
        val_curves,
    )

    for split, field, cmap, invert in (
        ("A_train", train_field, BLUE_CMAP, False),
        ("A_val", val_field, BLUE_CMAP_REVERSED, True),
    ):
        label = "Training set" if split == "A_train" else "Validation set"
        fig, axis = plt.subplots(figsize=(7.4, 5.8), constrained_layout=True)
        draw_potential_ax(axis, field, f"{label} empirical quasi-potential", cmap=cmap)
        savefig(fig, figure_root / f"evidence_{split}_quasipotential_{COORDINATE}_no_markers.png")
        fig, axis = plt.subplots(figsize=(7.4, 5.8), constrained_layout=True)
        draw_drift_field_ax(axis, field, f"{label} empirical drift field", cmap=cmap, invert_colorbar=invert)
        savefig(fig, figure_root / f"evidence_{split}_empirical_drift_{COORDINATE}_no_markers.png")

    fig, axis = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    draw_transition_matrix_ax(axis, val_transition, "Validation set macrostate transition matrix")
    savefig(fig, figure_root / "evidence_Aval_macrostate_transition_matrix_v3coords.png")

    fig, axis = plt.subplots(figsize=(7.6, 5.6), constrained_layout=True)
    draw_residence_vs_geometric_ax(axis, val_curves, "Validation set residence-time tails versus geometric null", MAX_RESIDENCE_LEN_FOR_PLOT)
    savefig(fig, figure_root / "evidence_Aval_basin_residence_vs_geometric_null_v3coords.png")

    fig, axes = plt.subplots(4, 1, figsize=(9.6, 7.8), constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.08, hspace=0.15)
    render_reproducibility_stack(axes, metric_rows, points, "Training–validation reproducibility of the empirical effective dynamics")
    savefig(fig, figure_root / "evidence_Atrain_Aval_reproducibility_summary.png")

    fig = plt.figure(figsize=(15.6, 10.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    axis_a = fig.add_subplot(grid[0, 0])
    axis_b = fig.add_subplot(grid[0, 1])
    axis_c = fig.add_subplot(grid[1, 0])
    subgrid = grid[1, 1].subgridspec(4, 1, hspace=0.06)
    axes_d = [fig.add_subplot(subgrid[index, 0]) for index in range(4)]
    draw_drift_field_ax(axis_a, val_field, "(a) Validation-set occupancy landscape and empirical drift field", cmap=BLUE_CMAP_REVERSED, invert_colorbar=True)
    draw_transition_matrix_ax(axis_b, val_transition, "(b) Validation-set macrostate transition matrix")
    draw_residence_vs_geometric_ax(axis_c, val_curves, "(c) Validation-set residence-time tails versus geometric null", MAX_RESIDENCE_LEN_FOR_PLOT)
    render_reproducibility_stack(axes_d, metric_rows, points, "(d) Training–validation reproducibility")
    savefig(fig, figure_root / "evidence_empirical_effective_dynamics_evidence_chain_overview.png")

    if args.make_b_confirm:
        confirm_transition = load_matrix(mesostate_root / "B_confirm_fixed_k6_transition_matrix")
        confirm_curves = read_table(mesostate_root / "B_confirm_fixed_k6_residence_curves")
        fig, axis = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
        draw_transition_matrix_ax(axis, confirm_transition, "Confirmation set output-only macrostate transition matrix")
        savefig(fig, figure_root / "evidence_Bconfirm_macrostate_transition_matrix_output_only_v3coords.png")
        fig, axis = plt.subplots(figsize=(7.6, 5.6), constrained_layout=True)
        draw_residence_vs_geometric_ax(axis, confirm_curves, "Confirmation set output-only residence-time tails versus geometric null", MAX_RESIDENCE_LEN_FOR_PLOT)
        savefig(fig, figure_root / "evidence_Bconfirm_basin_residence_vs_geometric_null_output_only_v3coords.png")

    write_table(centers_for_publication(centres), table_root / "publication_macrostate_centers_v3coords")
    write_table(read_table(mesostate_root / "fixed_k6_fit_table"), table_root / "publication_macrostate_k_selection_Atrain_Aval")
    for split in ("A_train", "A_val"):
        write_table(read_table(mesostate_root / f"{split}_fixed_k6_transition_counts"), table_root / f"{split}_publication_macrostate_transition_counts")
        write_table(read_table(mesostate_root / f"{split}_fixed_k6_transition_matrix"), table_root / f"{split}_publication_macrostate_transition_matrix")
        write_table(read_table(mesostate_root / f"{split}_fixed_k6_residence_runs"), table_root / f"{split}_publication_macrostate_residence_runs")
        write_table(read_table(mesostate_root / f"{split}_fixed_k6_residence_curves"), table_root / f"{split}_publication_macrostate_residence_kaplan_meier_curves")
        write_table(read_table(mesostate_root / f"{split}_fixed_k6_residence_summary"), table_root / f"{split}_publication_macrostate_residence_significance")
    write_table(pd.DataFrame([summary]), table_root / "publication_Atrain_Aval_reproducibility_summary")
    write_table(metric_rows, table_root / "publication_Atrain_Aval_reproducibility_bar_metrics")
    for metric, frame in points.items():
        slug = metric.lower().replace("-", "_").replace(" ", "_")
        write_table(frame, table_root / f"publication_Atrain_Aval_reproducibility_points_{slug}")

    manifest = {
        "script": Path(__file__).name,
        "stage1_root": stage1_root,
        "figure_root": figure_root,
        "table_root": table_root,
        "coordinate": COORDINATE,
        "state_coordinates": {"x": XCOL, "y": YCOL},
        "macrostate_policy": {
            "k": MACROSTATE_K,
            "source": "Stage-1 fixed_k6_mesostates outputs",
            "refit_performed": False,
            "candidate_k_search_performed": False,
            "metadata": metadata,
        },
        "visualization_policy": {
            "potential_clip_quantile": POTENTIAL_CLIP_Q,
            "drift_and_potential_region_markers": "disabled",
            "A_train_colormap": "light-to-deep blue",
            "A_val_colormap": "deep-to-light blue with inverted drift colorbar",
            "transition_style": "row-normalized six-state matrix with cell labels",
            "residence_style": "Kaplan-Meier CCDF and state-matched geometric reference on log-log axes",
            "reproducibility_style": "four stacked validation-minus-training fluctuation panels",
        },
        "outputs": {
            "potential_A_train": figure_root / f"evidence_A_train_quasipotential_{COORDINATE}_no_markers.png",
            "potential_A_val": figure_root / f"evidence_A_val_quasipotential_{COORDINATE}_no_markers.png",
            "drift_A_train": figure_root / f"evidence_A_train_empirical_drift_{COORDINATE}_no_markers.png",
            "drift_A_val": figure_root / f"evidence_A_val_empirical_drift_{COORDINATE}_no_markers.png",
            "transition_A_val": figure_root / "evidence_Aval_macrostate_transition_matrix_v3coords.png",
            "residence_A_val": figure_root / "evidence_Aval_basin_residence_vs_geometric_null_v3coords.png",
            "train_validation_reproducibility": figure_root / "evidence_Atrain_Aval_reproducibility_summary.png",
            "combined_figure2": figure_root / "evidence_empirical_effective_dynamics_evidence_chain_overview.png",
        },
        "reproducibility_summary": summary,
        "B_confirm_rendered": bool(args.make_b_confirm),
    }
    save_json(manifest, figure_root / "publication_empirical_effective_dynamics_figure_manifest.json")
    print(f"Figure 2: {figure_root / 'evidence_empirical_effective_dynamics_evidence_chain_overview.png'}")


if __name__ == "__main__":
    main()
