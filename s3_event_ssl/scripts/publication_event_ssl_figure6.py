#!/usr/bin/env python3
from __future__ import annotations

"""Render the Event-SSL control-experiment publication Figure 6 from frozen evaluations."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

DEFAULT_BASE = Path(os.environ.get("EDNET_KT4_OUTPUT_ROOT", "/data/datasets/KT4/outputs_KT4"))
DEFAULT_COMPARISON_ROOT = DEFAULT_BASE / "stage4_event_ssl" / "all_experiment_comparison"
DEFAULT_OUTPUT_ROOT = DEFAULT_BASE / "stage4_event_ssl" / "figures_publication_event_ssl"
EXPECTED_MACROSTATE_K = 6

DEFAULT_ROOTS = {
    "predictive_state_event_ssl": DEFAULT_BASE / "stage4_event_ssl" / "evaluation_predictive_state",
    "pure_event_ssl_probe": DEFAULT_BASE / "stage4_event_ssl" / "evaluation_pure_ssl_probe",
    "task_only": DEFAULT_BASE / "stage4_event_ssl" / "controls" / "task_only" / "evaluation",
    "time_shuffle_control": DEFAULT_BASE / "stage4_event_ssl_time_shuffle_control" / "evaluation_on_ordered_inputs",
    "tag_support_randomized": DEFAULT_BASE / "stage4_event_ssl_tag_support_randomized_control" / "evaluation",
}

MODEL_ORDER = [
    "predictive_state_event_ssl",
    "pure_event_ssl_probe",
    "task_only",
    "time_shuffle_control",
    "tag_support_randomized",
]
MODEL_LABEL = {
    "predictive_state_event_ssl": "Event-SSL",
    "pure_event_ssl_probe": "Pure SSL",
    "task_only": "Task-only",
    "time_shuffle_control": "Time-shuffle",
    "tag_support_randomized": "Support-alignment\nrandomised",
}
SHORT_LABEL = {
    "predictive_state_event_ssl": "Event-SSL",
    "pure_event_ssl_probe": "Pure SSL",
    "task_only": "Task-only",
    "time_shuffle_control": "Time-shuffle",
    "tag_support_randomized": "Support-alignment rand.",
}

DOMAIN_LABEL = {
    "coordinate": "Coordinate",
    "landscape": "Landscape",
    "learned_plane_drift": "Drift",
    "convergence": "Convergence",
    "transition": "Transition",
    "macrostructure_composite": "Composite",
}
DOMAIN_ORDER = ["coordinate", "landscape", "learned_plane_drift", "convergence", "transition", "macrostructure_composite"]

BLUE_CMAP = LinearSegmentedColormap.from_list(
    "ednet_light_to_deep_blue",
    ["#f7fbff", "#deebf7", "#9ecae1", "#4292c6", "#08519c", "#08306b"],
)

plt.rcParams.update({
    "font.size": 9.2,
    "axes.titlesize": 10.8,
    "axes.labelsize": 9.2,
    "xtick.labelsize": 8.1,
    "ytick.labelsize": 8.1,
    "legend.fontsize": 8.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.85,
})


def read_table(base_or_path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    p = Path(base_or_path)
    candidates: List[Path]
    if p.suffix:
        candidates = [p]
    else:
        candidates = [p.with_suffix(".parquet"), p.with_suffix(".csv.gz"), p.with_suffix(".csv")]
    last_exc: Optional[Exception] = None
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            if cand.suffix == ".parquet":
                return pd.read_parquet(cand, columns=list(columns) if columns is not None else None)
            return pd.read_csv(cand, usecols=list(columns) if columns is not None else None, low_memory=False)
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise FileNotFoundError(f"Could not find table at {p} with .parquet/.csv.gz/.csv extensions")


def load_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def audit_evaluation_root(model: str, root: Path, splits: Sequence[str]) -> Dict[str, object]:
    task_only = model == "task_only"
    manifest_name = "stage4_task_only_evaluation_manifest.json" if task_only else "stage4_event_ssl_evaluation_manifest.json"
    audit_name = "stage4_task_only_fixed_k6_partition_audit.json" if task_only else "stage4_event_ssl_fixed_k6_partition_audit.json"
    manifest_path = root / "metadata" / manifest_name
    audit_path = root / "metadata" / audit_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Evaluation manifest not found: {manifest_path}")
    if not audit_path.exists():
        raise FileNotFoundError(f"Fixed-K audit not found: {audit_path}")
    manifest = load_json(manifest_path)
    partition = load_json(audit_path)
    if manifest.get("primary_coordinates") != ["M", "Psi"]:
        raise RuntimeError(f"{model} does not use the M-Psi primary state.")
    expected_kind = {
        "predictive_state_event_ssl": "predictive_state",
        "pure_event_ssl_probe": "pure_ssl",
        "time_shuffle_control": "predictive_state",
        "tag_support_randomized": "predictive_state",
    }.get(model)
    if expected_kind is not None and str(manifest.get("model_kind", "")) != expected_kind:
        raise RuntimeError(f"{model} has model_kind={manifest.get('model_kind')!r}; expected {expected_kind!r}.")
    guardrails = dict(manifest.get("guardrails", {}))
    if guardrails.get("kmeans_refit") is not False:
        raise RuntimeError(f"{model} refit KMeans.")
    if guardrails.get("macrostate_k_selected") is not False:
        raise RuntimeError(f"{model} selected K.")
    if int(guardrails.get("macrostate_k", -1)) != EXPECTED_MACROSTATE_K:
        raise RuntimeError(f"{model} does not use fixed K=6.")
    if guardrails.get("B_confirm_used_for_update") is not False:
        raise RuntimeError(f"{model} used B_confirm for an update.")
    if partition.get("verified") is not True or int(partition.get("macrostate_k", -1)) != EXPECTED_MACROSTATE_K:
        raise RuntimeError(f"{model} has an invalid Stage-1 fixed-K audit.")
    if partition.get("kmeans_refit") is not False or partition.get("macrostate_k_selected") is not False:
        raise RuntimeError(f"{model} changed the frozen mesostate contract.")
    if task_only:
        metrics = read_table(root / "tables" / "stage4_task_only_structural_metrics_all_splits")
    else:
        metrics = read_table(root / "tables" / "stage4_event_ssl_structural_metrics_all_splits")
    required = {"split", "macrostate_partition_verified_against_stage1_fixed_k6", "macrostate_k_fixed_a_priori"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise RuntimeError(f"{model} metrics are missing fixed-K audit fields: {missing}")
    selected = metrics[metrics["split"].astype(str).isin([str(x) for x in splits])]
    if len(selected) != len(set(str(x) for x in splits)):
        raise RuntimeError(f"{model} does not contain all requested splits.")
    if not bool((pd.to_numeric(selected["macrostate_partition_verified_against_stage1_fixed_k6"], errors="coerce") == 1.0).all()):
        raise RuntimeError(f"{model} failed the fixed-K partition audit.")
    if not bool((pd.to_numeric(selected["macrostate_k_fixed_a_priori"], errors="coerce") == 1.0).all()):
        raise RuntimeError(f"{model} did not use fixed K=6.")
    return {
        "root": str(root.resolve()),
        "evaluation_manifest": str(manifest_path.resolve()),
        "fixed_k6_audit": str(audit_path.resolve()),
    }


def audit_roots(roots: Mapping[str, Path], splits: Sequence[str]) -> Dict[str, object]:
    return {model: audit_evaluation_root(model, Path(root), splits) for model, root in roots.items()}


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


def save_json(obj: Mapping, path: Path) -> None:
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating, float)):
            y = float(x)
            return y if np.isfinite(y) else None
        if isinstance(x, (np.bool_, bool)):
            return bool(x)
        if isinstance(x, Path):
            return str(x)
        return x
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(clean(dict(obj)), f, indent=2, ensure_ascii=False)


def savefig(fig: plt.Figure, path: Path, formats: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def publication_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#334155")
    ax.tick_params(length=3, width=0.7)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.115, 1.065, label, transform=ax.transAxes, fontsize=13, fontweight="normal", va="top", ha="left")


def load_wide_from_roots(roots: Mapping[str, Path]) -> pd.DataFrame:
    frames = []
    for model, root in roots.items():
        root = Path(root)
        if model == "task_only":
            base = root / "tables" / "stage4_task_only_structural_metrics_all_splits"
            task_base = root / "tables" / "stage4_task_only_task_metrics_all_splits"
            df = read_table(base)
            try:
                task = read_table(task_base)
                if "split" in task.columns:
                    df = pd.merge(df, task, on="split", how="left", suffixes=("", "_task"))
            except FileNotFoundError:
                pass
        else:
            df = read_table(root / "tables" / "stage4_event_ssl_structural_metrics_all_splits")
        df = df.copy()
        df.insert(0, "model_label", model)
        df.insert(1, "experiment_root", str(root.resolve()))
        frames.append(df)
    if not frames:
        raise RuntimeError("No metric tables were loaded.")
    return pd.concat(frames, ignore_index=True, sort=False)


def metric_value(wide: pd.DataFrame, model: str, split: str, metric: str) -> float:
    sub = wide[(wide["model_label"].astype(str) == model) & (wide["split"].astype(str) == split)]
    if sub.empty or metric not in sub.columns:
        return np.nan
    try:
        v = float(sub.iloc[0][metric])
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def normalize_metric(v: float, metric: str) -> float:
    if not np.isfinite(v):
        return np.nan
    if metric.endswith("_js") or "_rmse" in metric or metric.endswith("_bce") or metric.endswith("_row_tv"):
        return float(np.clip(1.0 - v, 0.0, 1.0))
    if "drift_vector_corr" in metric or "self_transition_corr" in metric or metric.startswith("coordinate_corr"):
        return float(np.clip(v, -1.0, 1.0))
    return float(np.clip(v, 0.0, 1.0))


def validate_comparison_wide(comparison_root: Path, direct: pd.DataFrame, splits: Sequence[str]) -> Optional[Path]:
    base = comparison_root / "tables" / "stage4_all_experiments_wide_metrics_all_splits"
    try:
        archived = read_table(base)
    except FileNotFoundError:
        return None
    keys = ["model_label", "split"]
    if not set(keys).issubset(archived.columns) or not set(keys).issubset(direct.columns):
        raise RuntimeError("Comparison-wide table lacks model and split identifiers.")
    requested = {str(x) for x in splits}
    archived = archived[archived["split"].astype(str).isin(requested)].copy()
    current = direct[direct["split"].astype(str).isin(requested)].copy()
    common = sorted(
        column for column in set(archived.columns).intersection(current.columns)
        if column not in {"experiment_root"} and column not in keys
    )
    merged = pd.merge(
        current[keys + common],
        archived[keys + common],
        on=keys,
        how="outer",
        suffixes=("_current", "_comparison"),
        indicator=True,
    )
    if not bool((merged["_merge"] == "both").all()):
        raise RuntimeError("Comparison-wide table does not match the formal evaluation rows.")
    for column in common:
        left = pd.to_numeric(merged[f"{column}_current"], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(merged[f"{column}_comparison"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(left) | np.isfinite(right)
        equal_nan = ~np.isfinite(left) & ~np.isfinite(right)
        close = np.isclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True)
        if not bool(np.all(close | equal_nan | ~finite)):
            raise RuntimeError(f"Comparison-wide metric differs from formal evaluations: {column}")
    candidates = [base.with_suffix(".parquet"), base.with_suffix(".csv.gz"), base.with_suffix(".csv")]
    return next(path.resolve() for path in candidates if path.exists())


def fallback_domain_scores(wide: pd.DataFrame, splits: Sequence[str]) -> pd.DataFrame:
    rows = []
    domain_specs = {
        "coordinate": ["coordinate_corr_M", "coordinate_corr_Psi"],
        "landscape": ["next_state_occupancy_js"],
        "learned_plane_drift": ["learned_plane_drift_vector_corr", "learned_plane_occupancy_weighted_local_drift_cosine"],
        "convergence": ["learned_plane_inward_fraction_to_reference", "learned_plane_negative_divergence_weighted_fraction"],
        "transition": ["learned_plane_transition_mean_row_tv", "learned_plane_self_transition_corr", "learned_plane_diagonal_dominance_match_fraction", "learned_plane_top_transition_edge_overlap"],
    }
    for split in splits:
        for model in MODEL_ORDER:
            for domain, metrics in domain_specs.items():
                vals = []
                used = []
                for m in metrics:
                    v = metric_value(wide, model, split, m)
                    s = normalize_metric(v, m)
                    if np.isfinite(s):
                        vals.append(s); used.append(m)
                if vals:
                    rows.append({"split": split, "model_label": model, "domain": domain, "domain_score": float(np.mean(vals)), "metrics_used": ";".join(used), "n_metrics_used": len(used)})
            comp_domains = ["coordinate", "landscape", "learned_plane_drift", "convergence", "transition"]
            sub_vals = [r["domain_score"] for r in rows if r["split"] == split and r["model_label"] == model and r["domain"] in comp_domains]
            if sub_vals:
                rows.append({"split": split, "model_label": model, "domain": "macrostructure_composite", "domain_score": float(np.mean(sub_vals)), "metrics_used": "domain_average_without_task", "n_metrics_used": len(sub_vals)})
    return pd.DataFrame(rows)


def load_inputs(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    roots = {
        "predictive_state_event_ssl": Path(args.main_root),
        "pure_event_ssl_probe": Path(args.pure_root),
        "task_only": Path(args.task_root),
        "time_shuffle_control": Path(args.time_shuffle_root),
        "tag_support_randomized": Path(args.tag_support_root),
    }
    root_audit = audit_roots(roots, [args.validation_split, args.split])
    wide = load_wide_from_roots(roots)
    comparison_root = Path(args.comparison_root)
    comparison_wide_path = validate_comparison_wide(
        comparison_root,
        wide,
        [args.validation_split, args.split],
    )
    score_base = comparison_root / "tables" / "stage4_all_experiments_domain_scores"
    if comparison_wide_path is not None:
        try:
            scores = read_table(score_base)
        except FileNotFoundError:
            scores = fallback_domain_scores(wide, [args.validation_split, args.split])
            score_source = "computed from formal evaluation metrics"
        else:
            required_models = set(MODEL_ORDER)
            required_splits = {str(args.validation_split), str(args.split)}
            if not {"model_label", "split", "domain", "domain_score"}.issubset(scores.columns):
                raise RuntimeError("Comparison-domain table has an unexpected schema.")
            if not required_models.issubset(set(scores["model_label"].astype(str))):
                raise RuntimeError("Comparison-domain table omits a required model.")
            if not required_splits.issubset(set(scores["split"].astype(str))):
                raise RuntimeError("Comparison-domain table omits a requested split.")
            score_candidates = [score_base.with_suffix(".parquet"), score_base.with_suffix(".csv.gz"), score_base.with_suffix(".csv")]
            score_path = next(path for path in score_candidates if path.exists())
            score_source = str(score_path.resolve())
    else:
        scores = fallback_domain_scores(wide, [args.validation_split, args.split])
        score_source = "computed from formal evaluation metrics"
    root_audit["comparison_wide_source"] = str(comparison_wide_path) if comparison_wide_path is not None else None
    root_audit["domain_score_source"] = score_source
    return wide, scores, root_audit

def domain_score(scores: pd.DataFrame, model: str, split: str, domain: str) -> float:
    sub = scores[(scores["model_label"].astype(str) == model) & (scores["split"].astype(str) == split) & (scores["domain"].astype(str) == domain)]
    if sub.empty:
        return np.nan
    try:
        return float(sub.iloc[0]["domain_score"])
    except Exception:
        return np.nan


def plot_domain_heatmap(fig: plt.Figure, ax: plt.Axes, scores: pd.DataFrame, split: str) -> None:
    data = np.full((len(MODEL_ORDER), len(DOMAIN_ORDER)), np.nan, dtype=float)
    for i, model in enumerate(MODEL_ORDER):
        for j, domain in enumerate(DOMAIN_ORDER):
            data[i, j] = domain_score(scores, model, split, domain)
    im = ax.imshow(data, vmin=0.0, vmax=1.0, cmap=BLUE_CMAP, aspect="auto")
    ax.set_box_aspect(0.83)
    ax.set_xticks(np.arange(len(DOMAIN_ORDER)))
    ax.set_xticklabels([DOMAIN_LABEL[d] for d in DOMAIN_ORDER], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_LABEL[m] for m in MODEL_ORDER])
    ax.set_title("Structural recovery across controls")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=7.1, color=("white" if data[i, j] > 0.62 else "#08306b"))
    publication_axes(ax)
    cax = inset_axes(
        ax,
        width="3.0%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.035, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("score")
    cb.set_ticks([0.0, 0.5, 1.0])
    add_panel_label(ax, "(a)")


def plot_empirical_vs_learned(ax: plt.Axes, wide: pd.DataFrame, split: str) -> None:
    models = MODEL_ORDER
    x = np.arange(len(models), dtype=float)
    anchor = [metric_value(wide, m, split, "anchor_drift_vector_corr") for m in models]
    learned = [metric_value(wide, m, split, "learned_plane_drift_vector_corr") for m in models]
    ax.plot(x, anchor, marker="o", linewidth=1.5, color="#6baed6", label="Empirical-plane closure")
    ax.plot(x, learned, marker="o", linewidth=1.5, color="#08519c", label="Learned-plane recovery")
    for xi, a, l in zip(x, anchor, learned):
        if np.isfinite(a) and np.isfinite(l):
            ax.plot([xi, xi], [a, l], color="#94a3b8", linewidth=0.8, alpha=0.8)
    ax.axhline(0.0, color="#334155", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABEL[m] for m in models], rotation=23, ha="right")
    ax.set_ylim(-0.5, 1.0)
    ax.set_ylabel("drift vector correlation")
    ax.set_title("Empirical-plane versus learned-plane flow")
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height - 0.03367])
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, loc="lower left", ncol=1, handlelength=1.4)
    publication_axes(ax)
    add_panel_label(ax, "(b)")


def plot_control_comparison_stack(fig: plt.Figure, parent, wide: pd.DataFrame, split: str) -> None:
    controls = [
        ("pure_event_ssl_probe", "Pure SSL"),
        ("task_only", "Task-only"),
        ("time_shuffle_control", "Time-shuffle"),
        ("tag_support_randomized", "Support-alignment\nrandomised"),
    ]
    metrics = [
        (r"$M$ corr", "coordinate_corr_M", "identity"),
        (r"$\Psi$ corr", "coordinate_corr_Psi", "identity"),
        ("Drift r", "learned_plane_drift_vector_corr", "identity"),
        ("1-TV", "learned_plane_transition_mean_row_tv", "one_minus"),
        ("Inward flow", "learned_plane_inward_fraction_to_reference", "identity"),
    ]

    outer = parent.subgridspec(
        5, 1,
        height_ratios=[0.14, 1.0, 1.0, 1.16, 1.0],
        hspace=0.075,
    )
    title_ax = fig.add_subplot(outer[0, 0])
    title_ax.set_axis_off()
    title_ax.text(0.5, 0.42, "Control comparisons with Event-SSL", ha="center", va="center", fontsize=10.8)
    add_panel_label(title_ax, "(c)")

    event_color = "#08519c"
    ctrl_color = "#c6dbef"
    x = np.arange(len(metrics), dtype=float)
    width = 0.38

    for idx, (model, control_label) in enumerate(controls):
        ax = fig.add_subplot(outer[idx + 1, 0])
        vals_main = []
        vals_ctrl = []
        for _, key, transform in metrics:
            vm = metric_value(wide, "predictive_state_event_ssl", split, key)
            vc = metric_value(wide, model, split, key)
            if transform == "one_minus":
                if np.isfinite(vm):
                    vm = 1.0 - vm
                if np.isfinite(vc):
                    vc = 1.0 - vc
            vals_main.append(vm if np.isfinite(vm) else np.nan)
            vals_ctrl.append(vc if np.isfinite(vc) else np.nan)

        ax.bar(
            x - width / 2, vals_main, width=width,
            color=event_color, edgecolor="#08306b", linewidth=0.55, label="Event-SSL",
        )
        ax.bar(
            x + width / 2, vals_ctrl, width=width,
            color=ctrl_color, edgecolor="#08306b", linewidth=0.55, label=control_label,
        )
        ax.axhline(0.0, color="#334155", linewidth=0.7)
        if model == "time_shuffle_control":
            ax.set_ylim(-0.45, 1.0)
            ax.set_yticks([-0.4, 0.0, 0.5, 1.0])
        else:
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([0.0, 0.5, 1.0])
        ax.grid(axis="y", alpha=0.18)
        ax.legend(frameon=False, loc="upper right", ncol=1, handlelength=1.35, labelspacing=0.25)
        if idx < len(controls) - 1:
            ax.set_xticks(x)
            ax.set_xticklabels([])
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([m[0] for m in metrics], rotation=18, ha="right")
        if idx == 1:
            ax.set_ylabel("metric value")
        else:
            ax.set_ylabel("")
        publication_axes(ax)


def build_key_table(wide: pd.DataFrame, scores: pd.DataFrame, split: str, out_root: Path) -> Path:
    rows: List[Dict[str, object]] = []
    for model in MODEL_ORDER:
        for metric in [
            "coordinate_corr_M", "coordinate_corr_Psi", "learned_plane_drift_vector_corr",
            "learned_plane_occupancy_weighted_local_drift_cosine", "learned_plane_transition_mean_row_tv",
            "learned_plane_self_transition_corr", "learned_plane_inward_fraction_to_reference", "task_auc_binary_rows",
        ]:
            rows.append({"split": split, "model_label": model, "quantity": metric, "value": metric_value(wide, model, split, metric), "source": "wide_metrics"})
        for domain in DOMAIN_ORDER:
            rows.append({"split": split, "model_label": model, "quantity": f"domain_{domain}", "value": domain_score(scores, model, split, domain), "source": "domain_scores"})
    return write_table(pd.DataFrame(rows), out_root / "tables" / "figure6_event_ssl_control_key_metrics")


def make_figure(args: argparse.Namespace) -> None:
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    wide, scores, input_audit = load_inputs(args)
    split = str(args.split)

    fig = plt.figure(figsize=(17.7, 5.5), constrained_layout=False)
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.02, 0.86, 1.42],
        left=0.055, right=0.985, bottom=0.135, top=0.925,
        wspace=0.27, hspace=0.0,
    )

    plot_domain_heatmap(fig, fig.add_subplot(gs[0, 0]), scores, split)
    plot_empirical_vs_learned(fig.add_subplot(gs[0, 1]), wide, split)
    plot_control_comparison_stack(fig, gs[0, 2], wide, split)

    formats = tuple(s.strip().lower() for s in str(args.formats).split(",") if s.strip())
    savefig(fig, out_root / args.figure_stem, formats)
    summary_path = build_key_table(wide, scores, split, out_root)
    manifest = {
        "script": Path(__file__).name,
        "comparison_root": str(Path(args.comparison_root).resolve()),
        "output_root": str(out_root.resolve()),
        "split_internal": split,
        "split_publication_label": "test set" if split == "B_confirm" else split,
        "primary_macrostate": ["M", "Psi"],
        "figure_stem": args.figure_stem,
        "key_metric_table": str(summary_path),
        "formal_input_audit": input_audit,
        "panels": {
            "a": "Domain-score overview across five Event-SSL and control models.",
            "b": "Empirical-plane closure versus learned-plane drift recovery.",
            "c": "Four stacked Event-SSL-versus-control comparisons across shared macrostructure metrics.",
        },
    }
    save_json(manifest, out_root / "metadata" / f"{args.figure_stem}_manifest.json")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render publication Figure 6 for Event-SSL control experiments.")
    ap.add_argument("--comparison-root", type=Path, default=DEFAULT_COMPARISON_ROOT)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--split", type=str, default="B_confirm")
    ap.add_argument("--validation-split", type=str, default="A_val")
    ap.add_argument("--figure-stem", type=str, default="figure6_event_ssl_controls")
    ap.add_argument("--formats", type=str, default="png,pdf,svg")
    ap.add_argument("--main-root", type=Path, default=DEFAULT_ROOTS["predictive_state_event_ssl"])
    ap.add_argument("--pure-root", type=Path, default=DEFAULT_ROOTS["pure_event_ssl_probe"])
    ap.add_argument("--task-root", type=Path, default=DEFAULT_ROOTS["task_only"])
    ap.add_argument("--time-shuffle-root", type=Path, default=DEFAULT_ROOTS["time_shuffle_control"])
    ap.add_argument("--tag-support-root", type=Path, default=DEFAULT_ROOTS["tag_support_randomized"])
    return ap.parse_args()


if __name__ == "__main__":
    make_figure(parse_args())
