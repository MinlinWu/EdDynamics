#!/usr/bin/env python3
from __future__ import annotations

"""Render Figure 7 from frozen Stage-5 and mechanism-Event-SSL comparison outputs."""

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

EPS = 1e-12

DEFAULT_BASE = Path(os.environ.get("EDNET_KT4_OUTPUT_ROOT", "/data/datasets/KT4/outputs_KT4"))
DEFAULT_MACRO_ROOT = DEFAULT_BASE / "stage5_macro_sufficiency" / "evaluation"
DEFAULT_GEOM_ROOT = DEFAULT_BASE / "stage5_representation_geometry" / "evaluation"
DEFAULT_CROSS_ROOT = DEFAULT_BASE / "cross_stage_mechanism_event_ssl_comparison"
DEFAULT_OUTPUT_ROOT = DEFAULT_BASE / "stage4_event_ssl" / "figures_publication_event_ssl"
EXPECTED_MACROSTATE_K = 6

BLUE = "#08519c"
MID_BLUE = "#4292c6"
LIGHT_BLUE = "#c6dbef"
PALE_BLUE = "#deebf7"
DARK = "#0f172a"
GRAY = "#64748b"
LIGHT_GRAY = "#e2e8f0"

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


HIGHER_IS_BETTER = {
    "coordinate_corr_M", "coordinate_corr_Psi", "one_step_corr_M", "one_step_corr_Psi",
    "anchor_drift_vector_corr", "learned_plane_drift_vector_corr",
    "anchor_occupancy_weighted_local_drift_cosine", "learned_plane_occupancy_weighted_local_drift_cosine",
    "learned_plane_self_transition_corr", "learned_plane_diagonal_dominance_match_fraction",
    "learned_plane_top_transition_edge_overlap", "task_auc", "task_accuracy_0p5",
    "representation_nmi_with_empirical_macrostate", "representation_ari_with_empirical_macrostate",
}
LOWER_IS_BETTER = {
    "coordinate_rmse_M", "coordinate_rmse_Psi", "one_step_rmse_M", "one_step_rmse_Psi",
    "current_state_occupancy_js", "next_state_occupancy_js",
    "learned_plane_transition_mean_row_tv", "anchor_transition_mean_row_tv",
    "task_bce", "task_rmse",
}


def publication_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#334155")
    ax.tick_params(length=3, width=0.7)


def savefig(fig: plt.Figure, path: Path, formats: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required table not found: {path}")
        warnings.warn(f"Optional table not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def read_table(base: Path) -> pd.DataFrame:
    for path in (base.with_suffix(".parquet"), base.with_suffix(".csv.gz"), base.with_suffix(".csv")):
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path, low_memory=False)
    raise FileNotFoundError(f"Required table not found: {base}.[parquet|csv.gz|csv]")


def load_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def audit_stage5_root(root: Path, analysis: str, split: str) -> Dict[str, object]:
    if analysis == "macro":
        manifest_name = "stage5_macro_sufficiency_evaluation_manifest.json"
        audit_name = "stage5_macro_sufficiency_fixed_k6_partition_audit.json"
        metrics_name = "stage5_macro_sufficiency_metrics_all_splits.csv"
    elif analysis == "geometry":
        manifest_name = "stage5_representation_geometry_evaluation_manifest.json"
        audit_name = "stage5_representation_geometry_fixed_k6_partition_audit.json"
        metrics_name = "stage5_representation_geometry_metrics_all_splits.csv"
    else:
        raise ValueError(analysis)
    manifest_path = root / "metadata" / manifest_name
    audit_path = root / "metadata" / audit_name
    metrics_path = root / "tables" / metrics_name
    for path in (manifest_path, audit_path, metrics_path):
        if not path.exists():
            raise FileNotFoundError(f"Required Stage-5 output not found: {path}")
    manifest = load_json(manifest_path)
    partition = load_json(audit_path)
    if manifest.get("primary_coordinates") != ["M", "Psi"]:
        raise RuntimeError(f"Stage-5 {analysis} output does not use the M-Psi primary state.")
    fixed = dict(manifest.get("fixed_k6_partition", {}))
    if int(fixed.get("macrostate_k", -1)) != EXPECTED_MACROSTATE_K:
        raise RuntimeError(f"Stage-5 {analysis} output does not use fixed K=6.")
    if fixed.get("kmeans_refit") is not False or fixed.get("macrostate_k_selected") is not False:
        raise RuntimeError(f"Stage-5 {analysis} changed the frozen mesostate contract.")
    if partition.get("verified") is not True or int(partition.get("macrostate_k", -1)) != EXPECTED_MACROSTATE_K:
        raise RuntimeError(f"Stage-5 {analysis} fixed-K audit is invalid.")
    if partition.get("kmeans_refit") is not False or partition.get("macrostate_k_selected") is not False:
        raise RuntimeError(f"Stage-5 {analysis} partition was refit or reselected.")
    metrics = pd.read_csv(metrics_path)
    required = {"split", "macrostate_partition_verified_against_stage1_fixed_k6"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise RuntimeError(f"Stage-5 {analysis} metrics are missing fixed-K audit fields: {missing}")
    selected = metrics[metrics["split"].astype(str) == str(split)]
    if selected.empty:
        raise RuntimeError(f"Stage-5 {analysis} metrics do not contain split={split!r}.")
    verified = pd.to_numeric(selected["macrostate_partition_verified_against_stage1_fixed_k6"], errors="coerce")
    if not bool((verified == 1.0).all()):
        raise RuntimeError(f"Stage-5 {analysis} split failed the fixed-K audit.")
    return {
        "root": str(root.resolve()),
        "evaluation_manifest": str(manifest_path.resolve()),
        "fixed_k6_audit": str(audit_path.resolve()),
        "metrics_table": str(metrics_path.resolve()),
    }


def audit_cross_model_root(root: Path, split: str) -> Dict[str, object]:
    manifest_path = root / "metadata" / "mechanism_event_ssl_comparison_manifest.json"
    table_bases = {
        "row_level": root / "tables" / "mechanism_event_ssl_row_level",
        "field": root / "tables" / "mechanism_event_ssl_field",
        "transition": root / "tables" / "mechanism_event_ssl_transition",
    }
    if not manifest_path.exists():
        raise FileNotFoundError(f"Required cross-model manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    primary = manifest.get("primary_macrostate", manifest.get("primary_coordinates"))
    if primary != ["M", "Psi"]:
        raise RuntimeError("Cross-model output does not use the M-Psi primary state.")
    source_split = str(
        manifest.get(
            "split_internal",
            manifest.get("split", split),
        )
    )
    if source_split != str(split):
        raise RuntimeError(
            f"Cross-model output uses split={source_split!r}, expected {split!r}."
        )

    partition = dict(
        manifest.get(
            "macro_partition_audit",
            manifest.get("fixed_k6_partition", {}),
        )
    )
    if (
        partition
        and int(partition.get("macrostate_k", EXPECTED_MACROSTATE_K))
        != EXPECTED_MACROSTATE_K
    ):
        raise RuntimeError("Cross-model output does not use fixed K=6.")
    if bool(partition.get("fit_on_test", False)):
        raise RuntimeError(
            "Cross-model partition was fitted on confirmation data."
        )

    guardrails = dict(manifest.get("guardrails", {}))
    if guardrails:
        guardrail_aliases = {
            "cross_model_fitting": (
                "cross_model_fitting",
                "cross_model_fit",
            ),
            "mechanism_refit": (
                "mechanism_refit",
                "mechanism_refit_to_event_ssl",
            ),
            "event_ssl_retrained": (
                "event_ssl_retrained",
                "event_ssl_refit_to_mechanism",
            ),
            "kmeans_refit": (
                "kmeans_refit",
            ),
            "macrostate_k_selected": (
                "macrostate_k_selected",
            ),
        }
        for label, names in guardrail_aliases.items():
            if any(bool(guardrails.get(name, False)) for name in names):
                raise RuntimeError(
                    f"Cross-model output violates guardrail {label!r}."
                )
    else:
        boundary = manifest.get("analysis_boundary", {})

        if isinstance(boundary, dict):
            required_false = (
                "model_retraining",
                "cross_model_fitting",
                "mechanism_refit_to_event_ssl",
                "event_ssl_refit_to_mechanism",
                "macrostate_redefinition",
                "kmeans_refit",
                "macrostate_k_selected",
                "confirmation_data_used_for_update",
            )
            missing = [
                name
                for name in required_false
                if name not in boundary
            ]
            if missing:
                raise RuntimeError(
                    "Cross-model manifest is missing analysis-boundary fields: "
                    f"{missing}."
                )

            failed = [
                name
                for name in required_false
                if bool(boundary.get(name))
            ]
            if failed:
                raise RuntimeError(
                    "Cross-model output violates analysis boundaries: "
                    f"{failed}."
                )
        else:
            boundary_text = " ".join(
                str(value).lower()
                for value in boundary
            )
            required_phrases = (
                "no retraining",
                "no refitting",
                "no macro-coordinate redefinition",
            )
            if not all(
                phrase in boundary_text
                for phrase in required_phrases
            ):
                raise RuntimeError(
                    "Cross-model manifest does not verify the "
                    "no-fitting analysis boundary."
                )
    join = dict(manifest.get("join_audit", {}))
    for name in ("join_fraction_of_mechanism", "join_fraction_of_event_ssl"):
        value = finite_float(join.get(name, np.nan))
        if np.isfinite(value) and value < 0.999999:
            raise RuntimeError(f"Cross-model join coverage is incomplete: {name}={value}.")
    joined_rows = finite_float(join.get("joined_rows", np.nan))
    analysed_rows = finite_float(join.get("analysis_rows_after_subsample", np.nan))
    if np.isfinite(joined_rows) and np.isfinite(analysed_rows) and int(joined_rows) != int(analysed_rows):
        raise RuntimeError("Cross-model publication output was generated from a row subsample.")
    for base in table_bases.values():
        read_table(base)
    return {
        "root": str(root.resolve()),
        "comparison_manifest": str(manifest_path.resolve()),
        "row_level_table": str(table_bases["row_level"]),
        "field_table": str(table_bases["field"]),
        "transition_table": str(table_bases["transition"]),
        "macrostate_k": EXPECTED_MACROSTATE_K,
        "cross_model_fitting": False,
    }


def comparison_metric(df: pd.DataFrame, comparisons: Sequence[str], metric: str) -> float:
    if not {"comparison", "metric", "value"}.issubset(df.columns):
        raise RuntimeError("Cross-model table is missing comparison, metric or value columns.")
    for comparison in comparisons:
        row = df[(df["comparison"].astype(str) == comparison) & (df["metric"].astype(str) == metric)]
        if not row.empty:
            value = finite_float(row.iloc[0]["value"])
            if np.isfinite(value):
                return value
    raise RuntimeError(f"Cross-model metric not found: comparisons={list(comparisons)}, metric={metric!r}.")


def load_cross_model_metrics(root: Path) -> Dict[str, float]:
    row = read_table(root / "tables" / "mechanism_event_ssl_row_level")
    field = read_table(root / "tables" / "mechanism_event_ssl_field")
    transition = read_table(root / "tables" / "mechanism_event_ssl_transition")
    values = {
        "next_M_corr": comparison_metric(row, ("mechanism_vs_event_ssl",), "next_M_corr"),
        "next_Psi_corr": comparison_metric(row, ("mechanism_vs_event_ssl",), "next_Psi_corr"),
        "interval_displacement_corr": comparison_metric(row, ("mechanism_vs_event_ssl",), "displacement_vector_corr"),
        "mean_interval_cosine": comparison_metric(row, ("mechanism_vs_event_ssl",), "mean_displacement_cosine"),
        "population_drift_corr": comparison_metric(field, ("mechanism_vs_event_ssl_anchor",), "drift_vector_corr"),
        "drift_speed_corr": comparison_metric(field, ("mechanism_vs_event_ssl_anchor",), "drift_speed_corr"),
        "local_drift_cosine": comparison_metric(field, ("mechanism_vs_event_ssl_anchor",), "occupancy_weighted_local_drift_cosine"),
        "transition_mean_row_tv": comparison_metric(transition, ("event_ssl_vs_mechanism", "mechanism_vs_event_ssl"), "mean_row_tv"),
        "statewise_persistence_corr": comparison_metric(transition, ("event_ssl_vs_mechanism", "mechanism_vs_event_ssl"), "self_transition_corr"),
    }
    for name in ("next_M_corr", "next_Psi_corr", "interval_displacement_corr", "mean_interval_cosine", "population_drift_corr", "drift_speed_corr", "local_drift_cosine", "statewise_persistence_corr"):
        if not -1.0 - 1e-9 <= values[name] <= 1.0 + 1e-9:
            raise RuntimeError(f"Cross-model correlation/cosine is outside [-1, 1]: {name}={values[name]}.")
    if not 0.0 <= values["transition_mean_row_tv"] <= 1.0 + 1e-9:
        raise RuntimeError("Cross-model transition row-TV is outside [0, 1].")
    return values


def finite_float(x: Any) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else np.nan
    except Exception:
        return np.nan


def val(row: Optional[pd.Series], name: str) -> float:
    if row is None:
        return np.nan
    return finite_float(row.get(name, np.nan))


def row_for(df: pd.DataFrame, split: str, representation: str) -> Optional[pd.Series]:
    if df.empty:
        return None
    sub = df[(df["split"].astype(str) == split) & (df["representation"].astype(str) == representation)]
    if sub.empty:
        return None
    return sub.iloc[0]


def bounded_corr_score(x: float) -> float:
    if not np.isfinite(x):
        return np.nan
    return float(np.clip((x + 1.0) / 2.0, 0.0, 1.0))


def positive_score_from_loss(x: float, scale: float = 1.0) -> float:
    if not np.isfinite(x):
        return np.nan
    return float(1.0 / (1.0 + max(x, 0.0) / max(scale, EPS)))


def score_from_metric(metric: str, value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    if metric in HIGHER_IS_BETTER:
        if "corr" in metric or "cosine" in metric or metric.endswith("ari_with_empirical_macrostate"):
            return bounded_corr_score(value)
        return float(np.clip(value, 0.0, 1.0))
    if metric in LOWER_IS_BETTER:
        if "row_tv" in metric or "js" in metric:
            return float(np.clip(1.0 - value, 0.0, 1.0))
        if "bce" in metric:
            return positive_score_from_loss(value, scale=1.0)
        return positive_score_from_loss(value, scale=0.15)
    return value


def compute_domain_scores(row: Optional[pd.Series]) -> Dict[str, float]:
    if row is None:
        return {k: np.nan for k in [
            "coordinate_score", "closure_score", "drift_score", "transition_score",
            "task_score", "macro_label_score", "macrostructure_composite_descriptive"
        ]}
    coord = np.nanmean([
        score_from_metric("coordinate_corr_M", val(row, "coordinate_corr_M")),
        score_from_metric("coordinate_corr_Psi", val(row, "coordinate_corr_Psi")),
    ])
    closure = np.nanmean([
        score_from_metric("one_step_rmse_M", val(row, "one_step_rmse_M")),
        score_from_metric("one_step_rmse_Psi", val(row, "one_step_rmse_Psi")),
    ])
    drift = np.nanmean([
        score_from_metric("learned_plane_drift_vector_corr", val(row, "learned_plane_drift_vector_corr")),
        score_from_metric("learned_plane_occupancy_weighted_local_drift_cosine", val(row, "learned_plane_occupancy_weighted_local_drift_cosine")),
    ])
    transition = np.nanmean([
        score_from_metric("learned_plane_transition_mean_row_tv", val(row, "learned_plane_transition_mean_row_tv")),
        score_from_metric("learned_plane_self_transition_corr", val(row, "learned_plane_self_transition_corr")),
        score_from_metric("learned_plane_diagonal_dominance_match_fraction", val(row, "learned_plane_diagonal_dominance_match_fraction")),
        score_from_metric("learned_plane_top_transition_edge_overlap", val(row, "learned_plane_top_transition_edge_overlap")),
    ])
    task = np.nan
    if np.isfinite(val(row, "task_auc")):
        task = score_from_metric("task_auc", val(row, "task_auc"))
    elif np.isfinite(val(row, "task_bce")):
        task = score_from_metric("task_bce", val(row, "task_bce"))
    macro_label_values = np.asarray([
        score_from_metric("representation_nmi_with_empirical_macrostate", val(row, "representation_nmi_with_empirical_macrostate")),
        score_from_metric("representation_ari_with_empirical_macrostate", val(row, "representation_ari_with_empirical_macrostate")),
    ], dtype=float)
    macro_label = float(np.nanmean(macro_label_values)) if np.isfinite(macro_label_values).any() else np.nan
    composite = np.nanmean([coord, closure, drift, transition])
    return {
        "coordinate_score": float(coord),
        "closure_score": float(closure),
        "drift_score": float(drift),
        "transition_score": float(transition),
        "task_score": float(task) if np.isfinite(task) else np.nan,
        "macro_label_score": float(macro_label) if np.isfinite(macro_label) else np.nan,
        "macrostructure_composite_descriptive": float(composite),
    }


def build_macro_retention(macro_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    domains = ["coordinate_score", "closure_score", "drift_score", "transition_score", "task_score", "macro_label_score", "macrostructure_composite_descriptive"]
    for split in sorted(macro_df["split"].astype(str).unique()):
        full = compute_domain_scores(row_for(macro_df, split, "full_hidden"))
        macro = compute_domain_scores(row_for(macro_df, split, "macro_only"))
        resid = compute_domain_scores(row_for(macro_df, split, "residual_hidden"))
        for d in domains:
            f, m, r = full[d], macro[d], resid[d]
            rows.append({
                "split": split,
                "domain": d,
                "full_hidden_score": f,
                "macro_only_score": m,
                "residual_hidden_score": r,
                "macro_retention_vs_full": m / f if np.isfinite(m) and np.isfinite(f) and abs(f) > EPS else np.nan,
                "residual_retention_vs_full": r / f if np.isfinite(r) and np.isfinite(f) and abs(f) > EPS else np.nan,
                "macro_minus_residual": m - r if np.isfinite(m) and np.isfinite(r) else np.nan,
            })
    return pd.DataFrame(rows)


def build_geometry_retention(geom_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    domains = ["coordinate_score", "closure_score", "drift_score", "transition_score", "macrostructure_composite_descriptive"]
    for split in sorted(geom_df["split"].astype(str).unique()):
        model = compute_domain_scores(row_for(geom_df, split, "model_readout"))
        linear = compute_domain_scores(row_for(geom_df, split, "linear_hidden"))
        resid = compute_domain_scores(row_for(geom_df, split, "residual_hidden"))
        for d in domains:
            mo, li, re = model[d], linear[d], resid[d]
            rows.append({
                "split": split,
                "domain": d,
                "model_readout_score": mo,
                "linear_hidden_score": li,
                "residual_hidden_score": re,
                "linear_retention_vs_model": li / mo if np.isfinite(li) and np.isfinite(mo) and abs(mo) > EPS else np.nan,
                "residual_retention_vs_model": re / mo if np.isfinite(re) and np.isfinite(mo) and abs(mo) > EPS else np.nan,
                "linear_minus_residual": li - re if np.isfinite(li) and np.isfinite(re) else np.nan,
            })
    return pd.DataFrame(rows)


def load_inputs(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    macro_root = Path(args.macro_root)
    geometry_root = Path(args.geometry_root)
    macro_df = read_csv(macro_root / "tables" / "stage5_macro_sufficiency_metrics_all_splits.csv")
    geom_df = read_csv(geometry_root / "tables" / "stage5_representation_geometry_metrics_all_splits.csv")
    pc_df = read_csv(geometry_root / "tables" / "stage5_representation_geometry_pc_macro_correlations.csv")
    gain_df = read_csv(geometry_root / "tables" / "stage5_representation_geometry_nonlinear_probe_gain.csv")
    macro_ret = build_macro_retention(macro_df)
    geom_ret = build_geometry_retention(geom_df)
    return macro_df, geom_df, pc_df, gain_df, macro_ret, geom_ret

def split_public_label(split: str) -> str:
    return {"A_val": "validation set", "B_confirm": "test set"}.get(split, split)


def domain_label(d: str) -> str:
    return {
        "coordinate_score": "Coordinate",
        "closure_score": "Closure",
        "drift_score": "Drift",
        "transition_score": "Transition",
        "task_score": "Task",
        "macro_label_score": "Macro label",
        "macrostructure_composite_descriptive": "Macrostructure",
    }.get(d, d)


def metric_from_retention(df: pd.DataFrame, split: str, domain: str, column: str) -> float:
    d = df[(df["split"].astype(str) == split) & (df["domain"].astype(str) == domain)]
    if d.empty or column not in d.columns:
        return np.nan
    return finite_float(d.iloc[0][column])


def plot_sufficiency_and_linear_panel(fig: plt.Figure, parent, macro_ret: pd.DataFrame, geom_ret: pd.DataFrame, split: str) -> None:
    """Draw macro-sufficiency and linear-accessibility subpanels."""
    sub = parent.subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.22)

    domains = ["coordinate_score", "closure_score", "drift_score", "transition_score", "macrostructure_composite_descriptive"]
    labels = [domain_label(d) for d in domains]
    x = np.arange(len(domains), dtype=float)

    ax_top = fig.add_subplot(sub[0, 0])
    d = macro_ret[(macro_ret["split"].astype(str) == split) & (macro_ret["domain"].isin(domains))].copy()
    d["order"] = d["domain"].map({xv: i for i, xv in enumerate(domains)})
    d = d.sort_values("order")
    full = d["full_hidden_score"].to_numpy(dtype=float)
    macro = d["macro_only_score"].to_numpy(dtype=float)
    resid = d["residual_hidden_score"].to_numpy(dtype=float)
    for xi, f, m, r in zip(x, full, macro, resid):
        vals = [f, m, r]
        if np.all(np.isfinite(vals)):
            ax_top.plot([xi, xi], [min(vals), max(vals)], color=LIGHT_GRAY, linewidth=1.8, zorder=1)
    ax_top.scatter(x, full, s=30, color=GRAY, edgecolor=DARK, linewidth=0.4, label="Full hidden", zorder=3)
    ax_top.scatter(x, macro, s=40, color=BLUE, edgecolor=DARK, linewidth=0.4, label="Macrostate only", zorder=4)
    ax_top.scatter(x, resid, s=40, color=LIGHT_BLUE, edgecolor=DARK, linewidth=0.4, label="Residual hidden", zorder=4)
    ax_top.set_ylim(0.0, 1.0)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([])
    ax_top.set_ylabel("recovery score")
    ax_top.set_title("(a) Macrostate bottleneck retention", pad=5)
    ax_top.grid(axis="y", alpha=0.18)
    ax_top.legend(frameon=False, loc="lower left", handlelength=1.2, labelspacing=0.22)
    publication_axes(ax_top)

    ax_bot = fig.add_subplot(sub[1, 0])
    g = geom_ret[(geom_ret["split"].astype(str) == split) & (geom_ret["domain"].isin(domains))].copy()
    g["order"] = g["domain"].map({xv: i for i, xv in enumerate(domains)})
    g = g.sort_values("order")
    model = g["model_readout_score"].to_numpy(dtype=float)
    linear = g["linear_hidden_score"].to_numpy(dtype=float)
    resg = g["residual_hidden_score"].to_numpy(dtype=float)
    for xi, mo, li, re in zip(x, model, linear, resg):
        vals = [mo, li, re]
        if np.all(np.isfinite(vals)):
            ax_bot.plot([xi, xi], [min(vals), max(vals)], color=LIGHT_GRAY, linewidth=1.8, zorder=1)
    ax_bot.scatter(x, model, s=30, color=GRAY, edgecolor=DARK, linewidth=0.4, label="Model readout", zorder=3)
    ax_bot.scatter(x, linear, s=40, color=BLUE, edgecolor=DARK, linewidth=0.4, label="Linear hidden", zorder=4)
    ax_bot.scatter(x, resg, s=40, color=LIGHT_BLUE, edgecolor=DARK, linewidth=0.4, label="Residual hidden", zorder=4)
    ax_bot.set_ylim(0.0, 1.0)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(labels, rotation=20, ha="right")
    ax_bot.set_ylabel("recovery score")
    ax_bot.set_title("Linear accessibility of macrostructure", pad=5)
    ax_bot.grid(axis="y", alpha=0.18)
    ax_bot.legend(frameon=False, loc="lower left", handlelength=1.2, labelspacing=0.22)
    publication_axes(ax_bot)


def plot_task_structure(ax: plt.Axes, macro_ret: pd.DataFrame, split: str) -> None:
    d = macro_ret[macro_ret["split"].astype(str) == split]
    comp = d[d["domain"] == "macrostructure_composite_descriptive"].iloc[0]
    task = d[d["domain"] == "task_score"].iloc[0]

    reps = [
        ("Full hidden", comp["full_hidden_score"], task["full_hidden_score"], GRAY, "o"),
        ("Macrostate only", comp["macro_only_score"], task["macro_only_score"], BLUE, "o"),
        ("Residual hidden", comp["residual_hidden_score"], task["residual_hidden_score"], LIGHT_BLUE, "o"),
    ]
    for label, y, x, color, marker in reps:
        if np.isfinite(x) and np.isfinite(y):
            ax.scatter(x, y, s=96, color=color, edgecolor=DARK, linewidth=0.6, marker=marker, label=label, zorder=5)
    full_x = finite_float(task["full_hidden_score"])
    full_y = finite_float(comp["full_hidden_score"])
    ax.axhline(full_y, color=GRAY, linewidth=0.75, linestyle="--", alpha=0.45)
    ax.axvline(full_x, color=GRAY, linewidth=0.75, linestyle="--", alpha=0.45)
    ax.set_xlim(0.575, 0.675)
    ax.set_ylim(0.43, 0.98)
    ax.set_xlabel("task score")
    ax.set_ylabel("macrostructure score")
    ax.set_title("(b) Task information separates from structure", pad=5)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="lower right", handlelength=1.2, labelspacing=0.25)
    publication_axes(ax)


def plot_geometry_panel(fig: plt.Figure, parent, geom_df: pd.DataFrame, pc_df: pd.DataFrame, gain_df: pd.DataFrame, split: str) -> None:
    sg = parent.subgridspec(
        2, 3,
        width_ratios=[1.16, 1.0, 0.88],
        height_ratios=[1.0, 1.0],
        wspace=0.82,
        hspace=0.46,
    )
    ax_pc = fig.add_subplot(sg[:, :2])
    p = pc_df[pc_df["split"].astype(str) == split].copy()
    if not p.empty:
        def pc_num(s: Any) -> int:
            try:
                return int(str(s).replace("PC", ""))
            except Exception:
                return 999
        p["pc_num"] = p["component"].map(pc_num)
        p = p.sort_values("pc_num").head(8)
        x = np.arange(len(p), dtype=float)
        width = 0.32
        ax_pc.bar(x - width / 2, p["abs_corr_M"].to_numpy(dtype=float), width=width, color=BLUE, edgecolor="#08306b", linewidth=0.6, label=r"$M$")
        ax_pc.bar(x + width / 2, p["abs_corr_Psi"].to_numpy(dtype=float), width=width, color=LIGHT_BLUE, edgecolor="#08306b", linewidth=0.6, label=r"$\Psi$")
        if "explained_variance_ratio_train" in p.columns:
            ax_var = ax_pc.twinx()
            ax_var.plot(x, p["explained_variance_ratio_train"].to_numpy(dtype=float), color=GRAY, marker="o", linewidth=1.0, markersize=3.2, label="variance")
            ax_var.set_ylim(0.0, max(0.20, float(np.nanmax(p["explained_variance_ratio_train"].to_numpy(dtype=float))) * 1.25))
            ax_var.set_ylabel("variance explained", labelpad=9)
            ax_var.tick_params(length=3, width=0.7, labelsize=7.8)
            for spine in ax_var.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("#334155")
            ax_var.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            h1, l1 = ax_pc.get_legend_handles_labels()
            h2, l2 = ax_var.get_legend_handles_labels()
            ax_pc.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", handlelength=1.2)
        else:
            ax_pc.legend(frameon=False, loc="upper right", handlelength=1.2)
        ax_pc.set_xticks(x)
        ax_pc.set_xticklabels(p["component"].astype(str).tolist())
    ax_pc.set_ylim(0.0, 0.85)
    ax_pc.set_ylabel("absolute correlation")
    ax_pc.set_title("(c) Leading hidden directions align with macrostates", pad=5)
    ax_pc.grid(axis="y", alpha=0.18)
    publication_axes(ax_pc)

    ax_cca = fig.add_subplot(sg[0, 2])
    row = row_for(geom_df, split, "model_readout")
    if row is None:
        row = row_for(geom_df, split, "linear_hidden")
    cca = np.array([val(row, "cca_corr_1"), val(row, "cca_corr_2")], dtype=float)
    ax_cca.bar(np.arange(2), cca, color=[BLUE, MID_BLUE], edgecolor="#08306b", linewidth=0.65, width=0.62)
    ax_cca.set_ylim(0.0, 1.0)
    ax_cca.set_xticks(np.arange(2))
    ax_cca.set_xticklabels(["CCA1", "CCA2"])
    ax_cca.set_title("Canonical", pad=5)
    ax_cca.set_ylabel("r")
    ax_cca.grid(axis="y", alpha=0.18)
    publication_axes(ax_cca)

    ax_gain = fig.add_subplot(sg[1, 2])
    g = gain_df[gain_df["split"].astype(str) == split]
    if not g.empty:
        gr = g.iloc[0]
        vals = np.array([finite_float(gr.get("nonlinear_gain_corr_M", np.nan)), finite_float(gr.get("nonlinear_gain_corr_Psi", np.nan))], dtype=float)
    else:
        vals = np.array([np.nan, np.nan], dtype=float)
    ax_gain.bar(np.arange(2), vals, color=[PALE_BLUE, LIGHT_BLUE], edgecolor="#08306b", linewidth=0.65, width=0.62)
    ax_gain.axhline(0.0, color="#334155", linewidth=0.75)
    ymax = max(0.01, float(np.nanmax(vals)) * 1.6 if np.isfinite(vals).any() else 0.01)
    ax_gain.set_ylim(0.0, ymax)
    ax_gain.set_xticks(np.arange(2))
    ax_gain.set_xticklabels([r"$M$", r"$\Psi$"])
    ax_gain.set_title("Nonlinear gain", pad=5)
    ax_gain.set_ylabel(r"$\Delta r$")
    ax_gain.grid(axis="y", alpha=0.18)
    publication_axes(ax_gain)


def plot_cross_model_panel(ax: plt.Axes, metrics: Dict[str, float]) -> None:
    labels = [
        "Statewise\npersistence $r$",
        "Drift-speed $r$",
        "Population\ndrift $r$",
        "Local drift\ncosine",
        "Interval\n$\\Delta X$ $r$",
    ]
    values = np.array([
        metrics["statewise_persistence_corr"],
        metrics["drift_speed_corr"],
        metrics["population_drift_corr"],
        metrics["local_drift_cosine"],
        metrics["interval_displacement_corr"],
    ], dtype=float)
    y = np.arange(len(labels), 0, -1, dtype=float)
    colors = [BLUE, BLUE, BLUE, MID_BLUE, LIGHT_BLUE]
    ax.hlines(y, 0.0, values, color=LIGHT_GRAY, linewidth=2.0, zorder=1)
    ax.scatter(values, y, s=[54, 46, 54, 46, 54], color=colors, edgecolor=DARK, linewidth=0.55, zorder=3)
    for value, yi in zip(values, y):
        if value >= 0.93:
            ax.text(value - 0.025, yi, f"{value:.3f}", ha="right", va="center", fontsize=7.7, color=DARK)
        else:
            ax.text(value + 0.025, yi, f"{value:.3f}", ha="left", va="center", fontsize=7.7, color=DARK)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.10, len(labels) + 0.72)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("cross-model agreement")
    ax.set_title("(d) Independent closures converge\non leading dynamics", pad=5)
    ax.grid(axis="x", alpha=0.18)
    ax.text(
        0.03, 0.025,
        f"mean interval cosine = {metrics['mean_interval_cosine']:.3f}\ntransition row-TV = {metrics['transition_mean_row_tv']:.3f}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.4, color=GRAY,
    )
    publication_axes(ax)


def key_metric_table(macro_ret: pd.DataFrame, geom_ret: pd.DataFrame, geom_df: pd.DataFrame, pc_df: pd.DataFrame, gain_df: pd.DataFrame, cross_metrics: Dict[str, float], split: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    def add(category: str, metric: str, value: float, interpretation: str):
        rows.append({"split": split, "category": category, "metric": metric, "value": value, "interpretation": interpretation})
    for dom in ["macrostructure_composite_descriptive", "coordinate_score", "transition_score", "drift_score", "task_score"]:
        add("macro-sufficiency", f"{dom}: macro-only/full-hidden", metric_from_retention(macro_ret, split, dom, "macro_retention_vs_full"), "macrostate bottleneck retention")
        add("macro-sufficiency", f"{dom}: residual/full-hidden", metric_from_retention(macro_ret, split, dom, "residual_retention_vs_full"), "residual-hidden retention")
    for dom in ["macrostructure_composite_descriptive", "coordinate_score", "transition_score", "drift_score"]:
        add("representation geometry", f"{dom}: model-readout score", metric_from_retention(geom_ret, split, dom, "model_readout_score"), "model state-head recovery")
        add("representation geometry", f"{dom}: linear-hidden score", metric_from_retention(geom_ret, split, dom, "linear_hidden_score"), "linear probe recovery")
        add("representation geometry", f"{dom}: residual-hidden score", metric_from_retention(geom_ret, split, dom, "residual_hidden_score"), "macrostate-removed residual recovery")
        add("representation geometry", f"{dom}: linear/model-readout", metric_from_retention(geom_ret, split, dom, "linear_retention_vs_model"), "linear accessibility ratio")
    row = row_for(geom_df, split, "model_readout")
    add("canonical geometry", "CCA1", val(row, "cca_corr_1"), "first canonical macrostate direction")
    add("canonical geometry", "CCA2", val(row, "cca_corr_2"), "second canonical macrostate direction")
    g = gain_df[gain_df["split"].astype(str) == split]
    if not g.empty:
        add("nonlinear probe", "nonlinear gain M", finite_float(g.iloc[0].get("nonlinear_gain_corr_M", np.nan)), "additional correlation over linear probe")
        add("nonlinear probe", "nonlinear gain Psi", finite_float(g.iloc[0].get("nonlinear_gain_corr_Psi", np.nan)), "additional correlation over linear probe")
    psub = pc_df[pc_df["split"].astype(str) == split]
    if not psub.empty:
        add("PC alignment", "max |corr(PC, M)|", finite_float(psub["abs_corr_M"].max()), "strongest leading-PC response-order alignment")
        add("PC alignment", "max |corr(PC, Psi)|", finite_float(psub["abs_corr_Psi"].max()), "strongest leading-PC exposure-alignment alignment")
    for metric, interpretation in [
        ("next_M_corr", "cross-model next-state response-order agreement"),
        ("next_Psi_corr", "cross-model next-state exposure-alignment agreement"),
        ("interval_displacement_corr", "cross-model interval displacement-vector agreement"),
        ("mean_interval_cosine", "mean cross-model interval displacement direction cosine"),
        ("population_drift_corr", "cross-model population drift-field agreement"),
        ("drift_speed_corr", "cross-model drift-speed pattern agreement"),
        ("local_drift_cosine", "occupancy-weighted cross-model local drift cosine"),
        ("transition_mean_row_tv", "cross-model fixed-K transition route difference"),
        ("statewise_persistence_corr", "cross-model statewise self-transition agreement"),
    ]:
        add("mechanism-Event-SSL", metric, cross_metrics[metric], interpretation)
    return pd.DataFrame(rows)


def make_figure(args: argparse.Namespace) -> None:
    split = args.split
    input_audit = {
        "macro_sufficiency": audit_stage5_root(Path(args.macro_root), "macro", split),
        "representation_geometry": audit_stage5_root(Path(args.geometry_root), "geometry", split),
        "mechanism_event_ssl": audit_cross_model_root(Path(args.cross_root), split),
    }
    macro_df, geom_df, pc_df, gain_df, macro_ret, geom_ret = load_inputs(args)
    cross_metrics = load_cross_model_metrics(Path(args.cross_root))
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20.8, 5.5), constrained_layout=False)
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1.42, 0.84, 1.60, 1.05],
        left=0.040, right=0.992, bottom=0.16, top=0.90,
        wspace=0.28, hspace=0.0,
    )

    plot_sufficiency_and_linear_panel(fig, gs[0, 0], macro_ret, geom_ret, split)
    plot_task_structure(fig.add_subplot(gs[0, 1]), macro_ret, split)
    plot_geometry_panel(fig, gs[0, 2], geom_df, pc_df, gain_df, split)
    plot_cross_model_panel(fig.add_subplot(gs[0, 3]), cross_metrics)

    formats = tuple(s.strip().lower() for s in str(args.formats).split(",") if s.strip())
    savefig(fig, out_root / args.figure_stem, formats=formats)

    table = key_metric_table(macro_ret, geom_ret, geom_df, pc_df, gain_df, cross_metrics, split)
    table_dir = out_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    try:
        table.to_parquet(table_dir / f"{args.figure_stem}_key_metrics.parquet", index=False)
    except Exception:
        pass
    table.to_csv(table_dir / f"{args.figure_stem}_key_metrics.csv", index=False)

    meta = {
        "script": Path(__file__).name,
        "split_internal": split,
        "split_public_label": split_public_label(split),
        "macro_root": str(Path(args.macro_root).resolve()),
        "geometry_root": str(Path(args.geometry_root).resolve()),
        "cross_root": str(Path(args.cross_root).resolve()),
        "primary_macrostate": ["M", "Psi"],
        "formal_input_audit": input_audit,
        "figure_design": "four aligned columns: macro-sufficiency and linear accessibility, task-structure dissociation, hidden geometry, and frozen mechanism-Event-SSL agreement",
        "panels": {
            "a": "Stacked macrostate-bottleneck sufficiency and linear-accessibility recovery scores.",
            "b": "Task score versus macrostructure score for full-hidden, macro-only and residual-hidden views.",
            "c": "Leading PC macrostate alignment, canonical alignment, and nonlinear probe gain.",
            "d": "Frozen mechanism-Event-SSL agreement in population drift, drift speed, local direction, statewise persistence and interval displacement, with interval cosine and transition row-TV shown as interpretation boundaries.",
        },
    }
    meta_dir = out_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with (meta_dir / f"{args.figure_stem}_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render publication Figure 7 for Event-SSL representation organization and frozen mechanism agreement.")
    ap.add_argument("--macro-root", type=Path, default=DEFAULT_MACRO_ROOT, help="Stage-5 macro-sufficiency evaluation root.")
    ap.add_argument("--geometry-root", type=Path, default=DEFAULT_GEOM_ROOT, help="Stage-5 representation-geometry evaluation root.")
    ap.add_argument("--cross-root", type=Path, default=DEFAULT_CROSS_ROOT, help="Frozen minimal-mechanism versus predictive-state Event-SSL comparison root.")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--split", type=str, default="B_confirm")
    ap.add_argument("--figure-stem", type=str, default="figure7_event_ssl_macrostate_organized_representations")
    ap.add_argument("--formats", type=str, default="png,pdf,svg")
    return ap.parse_args()


if __name__ == "__main__":
    make_figure(parse_args())
