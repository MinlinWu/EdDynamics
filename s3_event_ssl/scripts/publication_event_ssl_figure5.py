#!/usr/bin/env python3
from __future__ import annotations

"""Render Event-SSL macrostate-recovery publication Figure 5 from frozen Stage-4 outputs."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

EPS = 1e-12
GRID_BINS_SIGNED = np.linspace(-1.0, 1.0, 41)

BLUE_CMAP = LinearSegmentedColormap.from_list(
    "ednet_light_to_deep_blue",
    ["#f7fbff", "#deebf7", "#9ecae1", "#4292c6", "#08519c", "#08306b"],
)
RESIDUAL_CMAP = LinearSegmentedColormap.from_list(
    "ednet_blue_orange_residual",
    ["#8c2d04", "#fdd0a2", "#f7fbff", "#bdd7e7", "#08519c"],
)

plt.rcParams.update({
    "font.size": 9.2,
    "axes.titlesize": 10.8,
    "axes.labelsize": 9.2,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 8.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.85,
})

DEFAULT_BASE = Path(os.environ.get("EDNET_KT4_OUTPUT_ROOT", "/data/datasets/KT4/outputs_KT4"))
DEFAULT_EVAL_ROOT = DEFAULT_BASE / "stage4_event_ssl" / "evaluation_predictive_state"
DEFAULT_OUT_ROOT = DEFAULT_BASE / "stage4_event_ssl" / "figures_publication_event_ssl"
EXPECTED_MACROSTATE_K = 6


def publication_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#334155")
    ax.tick_params(length=3, width=0.7)


def read_table(base_or_path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    p = Path(base_or_path)
    candidates: List[Path]
    if p.suffix:
        candidates = [p]
    else:
        candidates = [p.with_suffix(".parquet"), p.with_suffix(".csv.gz"), p.with_suffix(".csv")]
    for cand in candidates:
        if not cand.exists():
            continue
        if cand.suffix == ".parquet":
            return pd.read_parquet(cand, columns=list(columns) if columns is not None else None)
        return pd.read_csv(cand, usecols=list(columns) if columns is not None else None, low_memory=False)
    raise FileNotFoundError(f"Could not find table at {p} with .parquet/.csv.gz/.csv extensions")


def load_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def audit_evaluation_root(eval_root: Path, splits: Sequence[str]) -> dict:
    manifest_path = eval_root / "metadata" / "stage4_event_ssl_evaluation_manifest.json"
    audit_path = eval_root / "metadata" / "stage4_event_ssl_fixed_k6_partition_audit.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Evaluation manifest not found: {manifest_path}")
    if not audit_path.exists():
        raise FileNotFoundError(f"Fixed-K audit not found: {audit_path}")
    manifest = load_json(manifest_path)
    partition = load_json(audit_path)
    if manifest.get("primary_coordinates") != ["M", "Psi"]:
        raise RuntimeError("Figure 5 requires the M-Psi primary state.")
    if str(manifest.get("model_kind", "")) != "predictive_state":
        raise RuntimeError("Figure 5 requires the predictive-state Event-SSL evaluation.")
    guardrails = dict(manifest.get("guardrails", {}))
    if guardrails.get("kmeans_refit") is not False:
        raise RuntimeError("The Event-SSL evaluation refit KMeans.")
    if guardrails.get("macrostate_k_selected") is not False:
        raise RuntimeError("The Event-SSL evaluation selected K.")
    if int(guardrails.get("macrostate_k", -1)) != EXPECTED_MACROSTATE_K:
        raise RuntimeError("The Event-SSL evaluation does not use fixed K=6.")
    if guardrails.get("B_confirm_used_for_update") is not False:
        raise RuntimeError("B_confirm was used for model update.")
    if partition.get("verified") is not True:
        raise RuntimeError("The Stage-1 fixed-K partition is not verified.")
    if int(partition.get("macrostate_k", -1)) != EXPECTED_MACROSTATE_K:
        raise RuntimeError("The Stage-1 partition is not K=6.")
    if partition.get("kmeans_refit") is not False or partition.get("macrostate_k_selected") is not False:
        raise RuntimeError("The Stage-1 partition contract was changed during evaluation.")
    metrics = load_metrics(eval_root)
    required = {"split", "macrostate_partition_verified_against_stage1_fixed_k6", "macrostate_k_fixed_a_priori"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise RuntimeError(f"Metric table is missing fixed-K audit fields: {missing}")
    selected = metrics[metrics["split"].astype(str).isin([str(x) for x in splits])]
    if len(selected) != len(set(str(x) for x in splits)):
        raise RuntimeError("Metric table does not contain all requested splits.")
    if not bool((pd.to_numeric(selected["macrostate_partition_verified_against_stage1_fixed_k6"], errors="coerce") == 1.0).all()):
        raise RuntimeError("A requested split failed the fixed-K partition audit.")
    if not bool((pd.to_numeric(selected["macrostate_k_fixed_a_priori"], errors="coerce") == 1.0).all()):
        raise RuntimeError("A requested split did not use fixed K=6.")
    return {
        "evaluation_manifest": str(manifest_path.resolve()),
        "fixed_k6_audit": str(audit_path.resolve()),
        "macrostate_k": EXPECTED_MACROSTATE_K,
    }

def json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_sanitize(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_sanitize(obj), f, indent=2, ensure_ascii=False)


def write_table(df: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = base.with_suffix(".parquet")
        df.to_parquet(out, index=False)
        return out
    except Exception:
        out = base.with_suffix(".csv")
        df.to_csv(out, index=False)
        return out


def savefig(fig: plt.Figure, path: Path, formats: Sequence[str] = ("png", "pdf", "svg")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_predictions(eval_root: Path, split: str, max_points: int, seed: int) -> pd.DataFrame:
    columns = ["M", "Psi", "target_M_next", "target_Psi_next", "pred_M", "pred_Psi", "pred_next_M", "pred_next_Psi"]
    df = read_table(eval_root / "predictions" / f"stage4_event_ssl_predictions_{split}", columns=columns)
    for c in columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isfinite(df[columns]).all(axis=1)].reset_index(drop=True)
    if max_points > 0 and len(df) > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(df), size=int(max_points), replace=False))
        df = df.iloc[idx].reset_index(drop=True)
    return df


def load_metrics(eval_root: Path) -> pd.DataFrame:
    path = eval_root / "tables" / "stage4_event_ssl_structural_metrics_all_splits.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for split in ("A_val", "B_confirm"):
        p = eval_root / "tables" / f"stage4_event_ssl_structural_metrics_{split}.csv"
        if p.exists():
            rows.append(pd.read_csv(p))
    if not rows:
        raise FileNotFoundError(f"Could not find Stage-4 metric tables under {eval_root / 'tables'}")
    return pd.concat(rows, ignore_index=True)


def metric_row(metrics: pd.DataFrame, split: str) -> Dict[str, float]:
    sub = metrics[metrics["split"].astype(str) == split]
    if sub.empty:
        raise RuntimeError(f"Metric table contains no row for split={split!r}.")
    out: Dict[str, float] = {}
    for k, v in sub.iloc[0].to_dict().items():
        try:
            out[str(k)] = float(np.real(v))
        except Exception:
            pass
    return out


def load_matrices(eval_root: Path, split: str) -> Dict[str, np.ndarray]:
    path = eval_root / "tables" / f"stage4_event_ssl_transition_matrices_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Could not find Stage-4 matrices: {path}")
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def publication_split_label(split: str) -> str:
    return {"A_train": "Training set", "A_val": "Validation set", "B_confirm": "Test set"}.get(split, str(split))


def fmt(x: float, digits: int = 3) -> str:
    try:
        v = float(np.real(x))
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def add_panel_label(ax: plt.Axes, label: str) -> None:
    panel = str(label)
    if not panel.startswith("("):
        panel = f"({panel})"
    ax.text(-0.115, 1.065, panel, transform=ax.transAxes, fontsize=12.0, fontweight="normal", va="top", ha="left")


def aligned_colorbar(fig: plt.Figure, ax: plt.Axes, mappable, label: str, *, width: str = "3.2%", pad: float = 0.06, labelpad: float = 2.0, shrink_ticks: bool = False):
    """Colorbar whose height is aligned to the parent axes."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=width, pad=pad)
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label, labelpad=labelpad)
    if shrink_ticks:
        cb.ax.tick_params(labelsize=7.6)
    return cb


def potential_from_H(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=float)
    H = H / max(float(np.nansum(H)), EPS)
    return -np.log(H + EPS)


def weighted_prediction_hist(obs: np.ndarray, pred: np.ndarray, bins: int = 85):
    ok = np.isfinite(obs) & np.isfinite(pred)
    H, xedges, yedges = np.histogram2d(obs[ok], pred[ok], bins=bins, range=[[-1, 1], [-1, 1]])
    return H.astype(float), xedges, yedges


def draw_prediction_density(fig: plt.Figure, ax: plt.Axes, obs: np.ndarray, pred: np.ndarray, title: str, xlab: str, ylab: str,
                            panel_label: str, norm: LogNorm) -> None:
    H, xedges, yedges = weighted_prediction_hist(np.asarray(obs, dtype=float), np.asarray(pred, dtype=float), bins=85)
    Hplot = np.where(H > 0, H, np.nan)
    cmap = BLUE_CMAP.copy(); cmap.set_bad("#f1f5f9")
    ax.set_facecolor("#f1f5f9")
    mesh = ax.pcolormesh(xedges, yedges, Hplot.T, shading="auto", cmap=cmap, norm=norm)
    ax.plot([-1, 1], [-1, 1], linestyle="--", color="#08306b", linewidth=1.0, alpha=0.88)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title, pad=6)
    publication_axes(ax)
    cb = aligned_colorbar(fig, ax, mesh, "sample count", width="3.0%", pad=0.055, labelpad=1.6, shrink_ticks=True)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    add_panel_label(ax, panel_label)


def plot_potential_residual(fig: plt.Figure, ax: plt.Axes, matrices: Mapping[str, np.ndarray], metrics: Mapping[str, float], vlim: float = 25.0) -> None:
    H_emp = np.asarray(matrices["H_emp_next"], dtype=float)
    H_pred = np.asarray(matrices["H_pred_next"], dtype=float)
    D = potential_from_H(H_pred) - potential_from_H(H_emp)
    positive = np.concatenate([H_emp[H_emp > 0].ravel(), H_pred[H_pred > 0].ravel()])
    if positive.size:
        min_mass = max(float(np.nanquantile(positive, 0.01)), EPS)
        support = (H_emp >= min_mass) | (H_pred >= min_mass)
    else:
        support = np.zeros_like(H_emp, dtype=bool)
    D = np.where(support, D, np.nan)
    lim = float(vlim) if np.isfinite(vlim) and float(vlim) > 0 else 25.0
    cmap = RESIDUAL_CMAP.copy(); cmap.set_bad("#f1f5f9")
    ax.set_facecolor("#f1f5f9")
    mesh = ax.pcolormesh(GRID_BINS_SIGNED, GRID_BINS_SIGNED, D.T, shading="auto", cmap=cmap, vmin=-lim, vmax=lim)
    cb = aligned_colorbar(fig, ax, mesh, r"predicted $-$ empirical $U_{\mathrm{next}}$", width="3.0%", pad=0.055, labelpad=1.6)
    cb.set_ticks([-lim, 0.0, lim])
    ax.set_title("Next-state quasi-potential residual", pad=6)
    ax.set_xlabel(r"Response order $M$")
    ax.set_ylabel(r"Exposure alignment $\Psi$")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect("equal", adjustable="box")
    publication_axes(ax)
    add_panel_label(ax, "c")


def plot_drift_residual(fig: plt.Figure, ax: plt.Axes, matrices: Mapping[str, np.ndarray], metrics: Mapping[str, float], vmax: float = 0.30) -> None:
    du = np.asarray(matrices["field_learned_u"], dtype=float) - np.asarray(matrices["field_emp_u"], dtype=float)
    dv = np.asarray(matrices["field_learned_v"], dtype=float) - np.asarray(matrices["field_emp_v"], dtype=float)
    mask = np.asarray(matrices["field_learned_mask"], dtype=bool) & np.asarray(matrices["field_emp_mask"], dtype=bool)
    residual = np.sqrt(du * du + dv * dv)
    residual = np.where(mask, residual, np.nan)
    vmax = float(vmax) if np.isfinite(vmax) and float(vmax) > 0 else 0.30
    cmap = BLUE_CMAP.copy(); cmap.set_bad("#f1f5f9")
    ax.set_facecolor("#f1f5f9")
    mesh = ax.pcolormesh(GRID_BINS_SIGNED, GRID_BINS_SIGNED, residual.T, shading="auto", cmap=cmap, vmin=0.0, vmax=vmax)
    cb = aligned_colorbar(fig, ax, mesh, r"drift residual magnitude $\|\hat b-b\|$", width="3.0%", pad=0.055, labelpad=1.6)
    cb.set_ticks([0.0, 0.5 * vmax, vmax])
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_title("Learned-plane drift residual", pad=6)
    ax.set_xlabel(r"Response order $M$")
    ax.set_ylabel(r"Exposure alignment $\Psi$")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect("equal", adjustable="box")
    publication_axes(ax)
    add_panel_label(ax, "d")


def plot_transition_residual(fig: plt.Figure, ax: plt.Axes, matrices: Mapping[str, np.ndarray], metrics: Mapping[str, float], vlim: float) -> None:
    P_emp = np.asarray(matrices["P_emp"], dtype=float)
    P_pred = np.asarray(matrices["P_learned"], dtype=float)
    diff = P_pred - P_emp
    finite = diff[np.isfinite(diff)]
    if not (np.isfinite(vlim) and vlim > 0):
        vlim = max(float(np.nanquantile(np.abs(finite), 0.98)) if finite.size else 0.10, 1e-6)
    im = ax.imshow(diff, origin="lower", vmin=-vlim, vmax=vlim, cmap=RESIDUAL_CMAP, aspect="auto")
    k = diff.shape[0]
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels([f"S{i}" for i in range(k)])
    ax.set_yticklabels([f"S{i}" for i in range(k)])
    ax.set_xlabel("Next macrostate")
    ax.set_ylabel("Current macrostate")
    ax.set_title("Transition residual", pad=6)
    threshold = 0.55 * (float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 0.5)
    for i in range(k):
        for j in range(k):
            val = float(diff[i, j])
            txt_color = "white" if abs(val) >= threshold else "#08306b"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=txt_color)
    publication_axes(ax)
    cb = aligned_colorbar(fig, ax, im, "predicted - empirical", width="3.0%", pad=0.045, labelpad=0.8, shrink_ticks=True)
    cb.set_ticks([-vlim, 0.0, vlim])
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    add_panel_label(ax, "e")


def score_metric(row: Mapping[str, float], key: str, transform: str = "identity") -> float:
    v = float(np.real(row.get(key, np.nan)))
    if not np.isfinite(v):
        return np.nan
    if transform == "one_minus":
        return 1.0 - v
    return v


def plot_stability(ax: plt.Axes, metrics_df: pd.DataFrame, test_split: str, val_split: str) -> None:
    rows = {val_split: metric_row(metrics_df, val_split), test_split: metric_row(metrics_df, test_split)}
    labels = ["$M$ corr", r"$\Psi$ corr", "1$-$JS", "drift r", "drift cos", "1$-$TV"]
    specs = [
        ("coordinate_corr_M", "identity"),
        ("coordinate_corr_Psi", "identity"),
        ("next_state_occupancy_js", "one_minus"),
        ("learned_plane_drift_vector_corr", "identity"),
        ("learned_plane_occupancy_weighted_local_drift_cosine", "identity"),
        ("learned_plane_transition_mean_row_tv", "one_minus"),
    ]
    x = np.arange(len(labels), dtype=float)
    width = 0.38
    vals_val = np.asarray([score_metric(rows[val_split], k, t) for k, t in specs], dtype=float)
    vals_test = np.asarray([score_metric(rows[test_split], k, t) for k, t in specs], dtype=float)
    vals_val = np.real_if_close(vals_val).astype(float)
    vals_test = np.real_if_close(vals_test).astype(float)
    vals_val = np.clip(vals_val, 0.0, 1.0, out=np.full_like(vals_val, np.nan), where=np.isfinite(vals_val))
    vals_test = np.clip(vals_test, 0.0, 1.0, out=np.full_like(vals_test, np.nan), where=np.isfinite(vals_test))
    ax.bar(x - width / 2, vals_val, width=width, color="#c6dbef", edgecolor="#08519c", linewidth=0.65, label="Validation set")
    ax.bar(x + width / 2, vals_test, width=width, color="#4292c6", edgecolor="#08306b", linewidth=0.65, label="Confirmation set")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("agreement score")
    ax.set_title("Confirmation stability", pad=6)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, loc="upper right", ncol=1, handlelength=1.4)
    publication_axes(ax)
    add_panel_label(ax, "f")


def build_summary(metrics: Mapping[str, float], split: str, out_root: Path) -> Path:
    keys = [
        "coordinate_corr_M", "coordinate_corr_Psi", "coordinate_rmse_M", "coordinate_rmse_Psi",
        "one_step_rmse_M", "one_step_rmse_Psi", "next_state_occupancy_js",
        "learned_plane_drift_vector_corr", "learned_plane_occupancy_weighted_local_drift_cosine",
        "learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr",
        "learned_plane_diagonal_dominance_match_fraction", "learned_plane_top_transition_edge_overlap",
    ]
    out = pd.DataFrame([{"split": split, "metric": k, "value": metrics.get(k, np.nan)} for k in keys])
    return write_table(out, out_root / "tables" / "figure5_event_ssl_key_metrics")


def make_figure(args: argparse.Namespace) -> None:
    eval_root = Path(args.eval_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    audit = audit_evaluation_root(eval_root, [args.validation_split, args.split])
    metrics_df = load_metrics(eval_root)
    metrics = metric_row(metrics_df, args.split)
    matrices = load_matrices(eval_root, args.split)
    pred_df = load_predictions(eval_root, args.split, max_points=args.max_plot_points, seed=args.seed)

    HM, xM, yM = weighted_prediction_hist(pred_df["M"].to_numpy(), pred_df["pred_M"].to_numpy())
    HP, xP, yP = weighted_prediction_hist(pred_df["Psi"].to_numpy(), pred_df["pred_Psi"].to_numpy())
    max_count = np.nanmax([np.nanmax(HM) if np.isfinite(HM).any() else 1.0, np.nanmax(HP) if np.isfinite(HP).any() else 1.0])
    norm = LogNorm(vmin=1.0, vmax=max(max_count, 1.0))

    fig = plt.figure(figsize=(16.9, 9.5), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.03, 1.03, 1.08], height_ratios=[1.0, 1.0], left=0.055, right=0.975, bottom=0.078, top=0.965, wspace=0.32, hspace=0.24)

    ax_a = fig.add_subplot(gs[0, 0])
    draw_prediction_density(
        fig, ax_a,
        pred_df["M"].to_numpy(), pred_df["pred_M"].to_numpy(),
        "Response order", r"Empirical $M$", r"Predicted $M$",
        "a", norm,
    )

    ax_b = fig.add_subplot(gs[0, 1])
    draw_prediction_density(
        fig, ax_b,
        pred_df["Psi"].to_numpy(), pred_df["pred_Psi"].to_numpy(),
        "Exposure alignment", r"Empirical $\Psi$", r"Predicted $\Psi$",
        "b", norm,
    )

    ax_c = fig.add_subplot(gs[0, 2])
    plot_potential_residual(fig, ax_c, matrices, metrics, vlim=float(args.potential_residual_vlim))

    ax_d = fig.add_subplot(gs[1, 0])
    plot_drift_residual(fig, ax_d, matrices, metrics, vmax=float(args.drift_residual_vmax))

    ax_e = fig.add_subplot(gs[1, 1])
    plot_transition_residual(fig, ax_e, matrices, metrics, vlim=float(args.transition_residual_vlim))

    ax_f = fig.add_subplot(gs[1, 2])
    plot_stability(ax_f, metrics_df, args.split, args.validation_split)

    formats = tuple(s.strip().lower() for s in args.formats.split(",") if s.strip())
    savefig(fig, out_root / args.figure_stem, formats=formats)

    summary_path = build_summary(metrics, args.split, out_root)
    manifest = {
        "script": Path(__file__).name,
        "eval_root": str(eval_root.resolve()),
        "output_root": str(out_root.resolve()),
        "split_internal": args.split,
        "split_publication_label": publication_split_label(args.split),
        "validation_split_internal": args.validation_split,
        "primary_macrostate": ["M", "Psi"],
        "panels": {
            "a": "Response-order recovery on the test set.",
            "b": "Exposure-alignment recovery on the test set.",
            "c": "Next-state quasi-potential residual map.",
            "d": "Learned-plane drift residual magnitude map.",
            "e": "Transition residual matrix under the fixed six-state coarse-graining.",
            "f": "Validation-test stability for primary structural scores.",
        },
        "key_metric_table": str(summary_path),
        "formal_input_audit": audit,
    }
    save_json(manifest, out_root / "metadata" / f"{args.figure_stem}_manifest.json")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render publication Figure 5 for Event-SSL macrostate recovery.")
    ap.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--split", type=str, default="B_confirm", help="Primary split to display as the test set.")
    ap.add_argument("--validation-split", type=str, default="A_val", help="Validation split used in the stability panel.")
    ap.add_argument("--figure-stem", type=str, default="figure5_event_ssl_macrostate_recovery")
    ap.add_argument("--formats", type=str, default="png,pdf,svg")
    ap.add_argument("--max-plot-points", type=int, default=200000)
    ap.add_argument("--potential-residual-vlim", type=float, default=25.0)
    ap.add_argument("--drift-residual-vmax", type=float, default=0.30)
    ap.add_argument("--transition-residual-vlim", type=float, default=0.50)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


if __name__ == "__main__":
    make_figure(parse_args())
