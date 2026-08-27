#!/usr/bin/env python3
from __future__ import annotations

"""Render a supplementary spatial comparison of frozen mechanism and Event-SSL closures."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FormatStrFormatter

EPS = 1e-12
PRIMARY_COORDINATES = ["M", "Psi"]
MACROSTATE_K = 6
GRID_EDGES = np.linspace(-1.0, 1.0, 41)

BLUE_CMAP = LinearSegmentedColormap.from_list(
    "ednet_density",
    ["#f7fbff", "#deebf7", "#9ecae1", "#4292c6", "#08519c", "#08306b"],
)
RESIDUAL_CMAP = LinearSegmentedColormap.from_list(
    "ednet_residual",
    ["#8c2d04", "#fdd0a2", "#f7fbff", "#bdd7e7", "#08519c"],
)

plt.rcParams.update(
    {
        "font.size": 9.0,
        "axes.titlesize": 10.4,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.85,
    }
)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def save_json(value: Mapping[str, Any], path: Path) -> None:
    atomic_write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        path,
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_path(base: Path) -> Path:
    candidates = [base] if base.suffix else [
        base.with_suffix(".parquet"),
        base.with_suffix(".csv.gz"),
        base.with_suffix(".csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find table for {base}")


def read_table(base: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    path = table_path(base)
    selected = list(columns) if columns is not None else None
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=selected)
    return pd.read_csv(path, usecols=selected, low_memory=False)


def write_table(frame: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    temporary = base.with_suffix(".parquet.tmp")
    try:
        parquet = base.with_suffix(".parquet")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, parquet)
        frame.to_csv(base.with_suffix(".csv"), index=False)
        return parquet
    except Exception:
        if temporary.exists():
            temporary.unlink()
        compressed = base.with_suffix(".csv.gz")
        temporary = compressed.with_name(compressed.name + ".tmp")
        frame.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, compressed)
        return compressed


def publication_axes(axis: plt.Axes) -> None:
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#334155")
    axis.tick_params(length=3, width=0.7)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.16,
        1.07,
        label,
        transform=axis.transAxes,
        fontsize=13,
        va="top",
        ha="left",
    )


def validate_comparison_manifest(manifest: Mapping[str, Any], split: str) -> None:
    primary = manifest.get("primary_macrostate")
    if primary != PRIMARY_COORDINATES:
        raise RuntimeError("Cross-model output does not use the M-Psi state")
    source_split = str(manifest.get("split", ""))
    if source_split != split:
        raise RuntimeError(f"Cross-model output uses split={source_split!r}, expected {split!r}")
    if split != "B_confirm":
        raise RuntimeError("Publication supplementary output must use B_confirm")
    if manifest.get("minimal_mechanism_family") != "offset_dual_channel":
        raise RuntimeError("Unexpected minimal-mechanism family")
    if manifest.get("event_ssl_model_kind") != "predictive_state":
        raise RuntimeError("Unexpected Event-SSL model kind")

    partition = dict(manifest.get("fixed_k6_partition", {}))
    if int(partition.get("macrostate_k", -1)) != MACROSTATE_K:
        raise RuntimeError("Cross-model output does not use fixed K=6")
    if bool(partition.get("kmeans_refit", False)) or bool(partition.get("macrostate_k_selected", False)):
        raise RuntimeError("Cross-model output changed the frozen mesostate partition")
    fit_split = str(partition.get("fit_split", ""))
    if fit_split and fit_split != "A_train":
        raise RuntimeError("Cross-model partition was not fitted on A_train")

    boundary = manifest.get("analysis_boundary", {})
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
    if not isinstance(boundary, Mapping):
        raise RuntimeError("Cross-model manifest has no frozen analysis boundary")
    missing = [name for name in required_false if name not in boundary]
    failed = [name for name in required_false if bool(boundary.get(name, False))]
    if missing or failed:
        raise RuntimeError(f"Cross-model analysis-boundary failure: missing={missing}, failed={failed}")


def resolve_output_path(value: Any, default_path: Path) -> Path:
    if value:
        candidate = Path(str(value))
        if candidate.exists():
            return candidate.resolve()
    return default_path.resolve()


def verify_prediction_hashes(
    manifest: Mapping[str, Any],
    comparison_root: Path,
    mechanism_path: Path,
    event_path: Path,
) -> dict[str, str]:
    source_path = resolve_output_path(
        manifest.get("source_audit"),
        table_path(comparison_root / "tables" / "mechanism_event_ssl_source_audit"),
    )
    sources = read_table(source_path)
    required = {"name", "sha256"}
    missing = sorted(required.difference(sources.columns))
    if missing:
        raise RuntimeError(f"Source audit is missing columns: {missing}")
    expected_names = {
        "minimal_mechanism_full_predictions": mechanism_path,
        "event_ssl_predictions": event_path,
    }
    verified: dict[str, str] = {}
    for name, path in expected_names.items():
        selected = sources[sources["name"].astype(str) == name]
        if len(selected) != 1:
            raise RuntimeError(f"Source audit does not contain exactly one row for {name}")
        expected = str(selected.iloc[0]["sha256"])
        actual = sha256_file(path)
        if expected != actual:
            raise RuntimeError(f"Prediction checksum mismatch for {name}")
        verified[name] = actual
    verified["source_audit"] = sha256_file(source_path)
    return verified


def clean_prediction_table(
    frame: pd.DataFrame,
    prediction_columns: Sequence[str],
    label: str,
) -> pd.DataFrame:
    output = frame.copy()
    required = ["user_id", "bundle_step_index", *prediction_columns]
    missing = [column for column in required if column not in output.columns]
    if missing:
        raise RuntimeError(f"{label} prediction table is missing columns: {missing}")
    for column in required:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    valid = output[required].notna().all(axis=1)
    for column in prediction_columns:
        valid &= np.isfinite(output[column].to_numpy(dtype=float))
    output = output.loc[valid, required].copy()
    output["user_id"] = output["user_id"].astype(np.int64)
    output["bundle_step_index"] = output["bundle_step_index"].astype(np.int64)
    for column in prediction_columns:
        output[column] = output[column].astype(np.float32)
    if output.duplicated(["user_id", "bundle_step_index"]).any():
        raise RuntimeError(f"{label} prediction keys are not unique")
    return output.sort_values(["user_id", "bundle_step_index"], kind="mergesort").reset_index(drop=True)


def load_joined_predictions(
    phase3_root: Path,
    event_ssl_root: Path,
    split: str,
    join_contract: Mapping[str, Any],
    max_rows: int,
    sample_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], Path, Path]:
    mechanism_path = table_path(phase3_root / "tables" / f"phase3_{split}_full_predictions")
    event_path = table_path(event_ssl_root / "predictions" / f"stage4_event_ssl_predictions_{split}")

    mechanism = clean_prediction_table(
        read_table(
            mechanism_path,
            ["user_id", "bundle_step_index", "pred_next_M", "pred_next_Psi"],
        ).rename(
            columns={"pred_next_M": "mechanism_next_M", "pred_next_Psi": "mechanism_next_Psi"}
        ),
        ("mechanism_next_M", "mechanism_next_Psi"),
        "Minimal mechanism",
    )
    event_ssl = clean_prediction_table(
        read_table(
            event_path,
            ["user_id", "bundle_step_index", "pred_next_M", "pred_next_Psi"],
        ).rename(
            columns={"pred_next_M": "event_ssl_next_M", "pred_next_Psi": "event_ssl_next_Psi"}
        ),
        ("event_ssl_next_M", "event_ssl_next_Psi"),
        "Event-SSL",
    )
    joined = pd.merge(
        mechanism,
        event_ssl,
        on=["user_id", "bundle_step_index"],
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        raise RuntimeError("No common prediction rows were found")

    join_audit = {
        "mechanism_rows": int(len(mechanism)),
        "event_ssl_rows": int(len(event_ssl)),
        "joined_rows": int(len(joined)),
        "joined_users": int(joined["user_id"].nunique()),
        "join_fraction_of_mechanism": float(len(joined) / max(len(mechanism), 1)),
        "join_fraction_of_event_ssl": float(len(joined) / max(len(event_ssl), 1)),
    }
    minimum_fraction = float(join_contract.get("minimum_required_join_fraction", 0.999999))
    if join_audit["join_fraction_of_mechanism"] < minimum_fraction:
        raise RuntimeError("Mechanism prediction join coverage is below the required threshold")
    if join_audit["join_fraction_of_event_ssl"] < minimum_fraction:
        raise RuntimeError("Event-SSL prediction join coverage is below the required threshold")
    expected_rows = int(join_contract.get("joined_rows", len(joined)))
    if len(joined) != expected_rows:
        raise RuntimeError(f"Joined row count {len(joined)} differs from the frozen comparison {expected_rows}")

    if max_rows > 0 and len(joined) > max_rows:
        random = np.random.default_rng(sample_seed)
        indices = np.sort(random.choice(len(joined), size=int(max_rows), replace=False))
        joined = joined.iloc[indices].reset_index(drop=True)
    join_audit["density_rows"] = int(len(joined))
    join_audit["density_row_subsample"] = int(max_rows)
    return joined, join_audit, mechanism_path, event_path


def load_matrices(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        matrices = {name: np.asarray(data[name]) for name in data.files}
    required = (
        "H_mechanism",
        "H_event_ssl",
        "field_mech_u",
        "field_mech_v",
        "field_mech_mask",
        "field_ssl_anchor_u",
        "field_ssl_anchor_v",
        "field_ssl_anchor_mask",
    )
    missing = [name for name in required if name not in matrices]
    if missing:
        raise RuntimeError(f"Frozen comparison matrices are missing: {missing}")
    for name in required:
        if matrices[name].shape != (40, 40):
            raise RuntimeError(f"Matrix {name} has shape {matrices[name].shape}, expected (40, 40)")
    return matrices


def joint_histogram(frame: pd.DataFrame, x_column: str, y_column: str) -> np.ndarray:
    histogram, _, _ = np.histogram2d(
        frame[x_column].to_numpy(dtype=float),
        frame[y_column].to_numpy(dtype=float),
        bins=70,
        range=[[-1.0, 1.0], [-1.0, 1.0]],
    )
    return histogram


def potential(occupancy: np.ndarray) -> np.ndarray:
    values = np.asarray(occupancy, dtype=float)
    total = float(np.nansum(values))
    if total <= 0:
        raise RuntimeError("Occupancy matrix has zero total mass")
    return -np.log(values / total + EPS)


def add_density_panel(
    figure: plt.Figure,
    parent,
    joined: pd.DataFrame,
) -> tuple[list[plt.Axes], plt.Axes]:
    subgrid = parent.subgridspec(
        1,
        5,
        width_ratios=[1.0, 0.14, 1.0, 0.06, 0.04],
        wspace=0.0,
    )
    axes = [figure.add_subplot(subgrid[0, 0]), figure.add_subplot(subgrid[0, 2])]
    color_axis = figure.add_subplot(subgrid[0, 4])

    histograms = [
        joint_histogram(joined, "event_ssl_next_M", "mechanism_next_M"),
        joint_histogram(joined, "event_ssl_next_Psi", "mechanism_next_Psi"),
    ]
    maximum = max(float(np.nanmax(histogram)) for histogram in histograms)
    normalization = LogNorm(vmin=1.0, vmax=max(maximum, 1.0))
    mappable = None
    for index, (axis, histogram, title) in enumerate(
        zip(
            axes,
            histograms,
            ("Response order", "Exposure alignment"),
        )
    ):
        plotted = np.where(histogram > 0, histogram, np.nan)
        cmap = BLUE_CMAP.copy()
        cmap.set_bad("#f1f5f9")
        axis.set_facecolor("#f1f5f9")
        mappable = axis.pcolormesh(
            np.linspace(-1.0, 1.0, 71),
            np.linspace(-1.0, 1.0, 71),
            plotted.T,
            shading="auto",
            cmap=cmap,
            norm=normalization,
        )
        axis.plot([-1, 1], [-1, 1], color="#334155", linestyle="--", linewidth=0.9)
        axis.set_xlim(-1, 1)
        axis.set_ylim(-1, 1)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Event-SSL next state")
        axis.set_title(title)
        if index == 0:
            axis.set_ylabel("Mechanism next state")
        else:
            axis.tick_params(labelleft=False)
            axis.yaxis.set_label_position("right")
            axis.yaxis.set_label_coords(1.03, 0.5)
        publication_axes(axis)
    if mappable is None:
        raise RuntimeError("Density panel did not create a mappable")
    colorbar = figure.colorbar(mappable, cax=color_axis)
    colorbar.set_label("Interval count", rotation=270, labelpad=8.0, fontsize=8.0)
    add_panel_label(axes[0], "(a)")
    return axes, color_axis


def add_drift_panel(
    figure: plt.Figure,
    parent,
    matrices: Mapping[str, np.ndarray],
    vmax: float,
) -> tuple[plt.Axes, plt.Axes, int]:
    subgrid = parent.subgridspec(1, 2, width_ratios=[1.0, 0.060], wspace=0.10)
    axis = figure.add_subplot(subgrid[0, 0])
    color_axis = figure.add_subplot(subgrid[0, 1])
    common = matrices["field_mech_mask"].astype(bool) & matrices["field_ssl_anchor_mask"].astype(bool)
    difference_m = matrices["field_ssl_anchor_u"] - matrices["field_mech_u"]
    difference_psi = matrices["field_ssl_anchor_v"] - matrices["field_mech_v"]
    residual = np.where(common, np.hypot(difference_m, difference_psi), np.nan)
    cmap = BLUE_CMAP.copy()
    cmap.set_bad("#f1f5f9")
    axis.set_facecolor("#f1f5f9")
    image = axis.pcolormesh(
        GRID_EDGES,
        GRID_EDGES,
        residual.T,
        shading="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
    )
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"Response order $M$")
    axis.set_ylabel(r"Exposure alignment $\Psi$")
    axis.set_title("Empirical-anchor drift-field difference")
    publication_axes(axis)
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label(
        r"$\|\Delta \mathbf{b}\|$",
        rotation=270,
        labelpad=9.0,
        fontsize=7.8,
    )
    colorbar.set_ticks([0.0, vmax / 2.0, vmax])
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    add_panel_label(axis, "(b)")
    return axis, color_axis, int(common.sum())


def add_landscape_panel(
    figure: plt.Figure,
    parent,
    matrices: Mapping[str, np.ndarray],
    limit: float,
) -> tuple[plt.Axes, plt.Axes, int]:
    subgrid = parent.subgridspec(1, 2, width_ratios=[1.0, 0.060], wspace=0.10)
    axis = figure.add_subplot(subgrid[0, 0])
    color_axis = figure.add_subplot(subgrid[0, 1])
    mechanism = np.asarray(matrices["H_mechanism"], dtype=float)
    event_ssl = np.asarray(matrices["H_event_ssl"], dtype=float)
    common = (mechanism > 0) & (event_ssl > 0)
    difference = np.where(common, potential(event_ssl) - potential(mechanism), np.nan)
    cmap = RESIDUAL_CMAP.copy()
    cmap.set_bad("#f1f5f9")
    axis.set_facecolor("#f1f5f9")
    image = axis.pcolormesh(
        GRID_EDGES,
        GRID_EDGES,
        difference.T,
        shading="auto",
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
    )
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"Response order $M$")
    axis.set_ylabel(r"Exposure alignment $\Psi$")
    axis.set_title("Next-state quasi-potential difference")
    publication_axes(axis)
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label(
        r"$\Delta U_{\mathrm{next}}$",
        rotation=270,
        labelpad=8.0,
        fontsize=7.8,
    )
    colorbar.set_ticks([-limit, 0.0, limit])
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    add_panel_label(axis, "(c)")
    return axis, color_axis, int(common.sum())


def validate_layout(
    figure: plt.Figure,
    density_axes: Sequence[plt.Axes],
    main_axes: Sequence[plt.Axes],
    color_axes: Sequence[plt.Axes],
) -> dict[str, float]:
    positions = [axis.get_position() for axis in main_axes]
    heights = [position.height for position in positions]
    bottoms = [position.y0 for position in positions]
    height_range = max(heights) - min(heights)
    bottom_range = max(bottoms) - min(bottoms)
    if height_range > 0.025 or bottom_range > 0.025:
        raise RuntimeError(
            f"Main-panel alignment failed: height_range={height_range}, bottom_range={bottom_range}"
        )
    ordered = sorted(positions, key=lambda position: position.x0)
    for first, second in zip(ordered[:-1], ordered[1:]):
        if first.x1 > second.x0 + 1e-6:
            raise RuntimeError("Main plotting axes overlap")
    for main_axis, color_axis in zip((main_axes[1], main_axes[2], main_axes[3]), color_axes):
        main_position = main_axis.get_position()
        color_position = color_axis.get_position()
        if abs(main_position.y0 - color_position.y0) > 0.008:
            raise RuntimeError("A colorbar is vertically misaligned")
        if abs(main_position.height - color_position.height) > 0.008:
            raise RuntimeError("A colorbar height does not match its panel")

    renderer = figure.canvas.get_renderer()
    left_position = density_axes[0].get_position()
    right_position = density_axes[1].get_position()
    right_label_box = density_axes[1].yaxis.label.get_window_extent(renderer=renderer)
    right_label_box = right_label_box.transformed(figure.transFigure.inverted())

    density_color_position = color_axes[0].get_position()
    density_colorbar_gap = density_color_position.x0 - right_position.x1
    if density_colorbar_gap < 0.003:
        raise RuntimeError("The density colorbar is too close to the right density panel")
    if density_colorbar_gap > 0.015:
        raise RuntimeError("The density colorbar is too far from the right density panel")

    return {
        "main_axis_height_range": float(height_range),
        "main_axis_bottom_range": float(bottom_range),
        "right_density_ylabel_gap_from_left_panel": float(right_label_box.x0 - left_position.x1),
        "density_colorbar_gap": float(density_colorbar_gap),
    }


def write_caption(path: Path) -> None:
    caption = (
        r"\textbf{Interval- and state-resolved comparison of frozen neural and mechanistic closures.} "
        r"\textbf{a}, Joint densities of mechanism- and Event-SSL-implied next-state response order $M$ and exposure alignment $\Psi$ on common confirmation intervals; dashed lines show identity and colour denotes interval count on a logarithmic scale. "
        r"\textbf{b}, Magnitude of the difference between their empirical-anchor drift fields. "
        r"\textbf{c}, Event-SSL-minus-mechanism next-state occupancy-derived quasi-potential on common occupancy support. "
        r"All panels use frozen outputs on the same confirmation intervals; neither closure was fitted to the other, and pale cells lack common support."
    )
    atomic_write_text(caption + "\n", path)


def save_figure(figure: plt.Figure, base: Path, formats: Sequence[str]) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for extension in formats:
        output = base.with_suffix(f".{extension}")
        figure.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.05)
        outputs.append(str(output.resolve()))
    plt.close(figure)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a supplementary spatial comparison of frozen mechanism and Event-SSL closures."
    )
    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/cross_stage_mechanism_event_ssl_comparison"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="B_confirm")
    parser.add_argument("--max-density-rows", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--drift-vmax", type=float, default=0.30)
    parser.add_argument("--potential-vlim", type=float, default=25.0)
    parser.add_argument("--verify-prediction-hashes", action="store_true", default=True)
    parser.add_argument(
        "--skip-prediction-hash-check",
        action="store_false",
        dest="verify_prediction_hashes",
    )
    parser.add_argument("--formats", type=str, default="png,pdf,svg")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    comparison_root = args.comparison_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = comparison_root / "metadata" / "mechanism_event_ssl_comparison_manifest.json"
    manifest = load_json(manifest_path)
    validate_comparison_manifest(manifest, args.split)

    phase3_value = str(manifest.get("phase3_root", "")).strip()
    event_ssl_value = str(manifest.get("event_ssl_eval_root", "")).strip()
    if not phase3_value or not event_ssl_value:
        raise RuntimeError("The comparison manifest does not identify both prediction roots")
    phase3_root = Path(phase3_value).resolve()
    event_ssl_root = Path(event_ssl_value).resolve()
    if not phase3_root.exists() or not event_ssl_root.exists():
        raise FileNotFoundError("The comparison manifest refers to unavailable prediction roots")

    outputs = dict(manifest.get("outputs", {}))
    matrices_path = resolve_output_path(
        outputs.get("matrices"),
        comparison_root / "tables" / "mechanism_event_ssl_matrices.npz",
    )
    matrices = load_matrices(matrices_path)
    joined, join_audit, mechanism_path, event_path = load_joined_predictions(
        phase3_root,
        event_ssl_root,
        args.split,
        dict(manifest.get("join_audit", {})),
        args.max_density_rows,
        args.sample_seed,
    )
    prediction_hashes = (
        verify_prediction_hashes(
            manifest,
            comparison_root,
            mechanism_path,
            event_path,
        )
        if args.verify_prediction_hashes
        else {}
    )

    figure = plt.figure(figsize=(16.2, 4.65), constrained_layout=False)
    outer = figure.add_gridspec(
        1,
        3,
        width_ratios=[2.70, 1.30, 1.30],
        left=0.050,
        right=0.987,
        bottom=0.165,
        top=0.900,
        wspace=0.40,
    )
    density_axes, density_color_axis = add_density_panel(figure, outer[0, 0], joined)
    drift_axis, drift_color_axis, common_drift_cells = add_drift_panel(
        figure,
        outer[0, 1],
        matrices,
        args.drift_vmax,
    )
    landscape_axis, landscape_color_axis, common_landscape_cells = add_landscape_panel(
        figure,
        outer[0, 2],
        matrices,
        args.potential_vlim,
    )
    figure.canvas.draw()
    for main_axis, color_axis in (
        (density_axes[1], density_color_axis),
        (drift_axis, drift_color_axis),
        (landscape_axis, landscape_color_axis),
    ):
        main_position = main_axis.get_position()
        color_position = color_axis.get_position()
        color_axis.set_position(
            [color_position.x0, main_position.y0, color_position.width, main_position.height]
        )
    figure.canvas.draw()
    layout_audit = validate_layout(
        figure,
        density_axes,
        [density_axes[0], density_axes[1], drift_axis, landscape_axis],
        [density_color_axis, drift_color_axis, landscape_color_axis],
    )

    figure_root = output_root / "figures"
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    latex_root = output_root / "latex"
    for directory in (figure_root, table_root, metadata_root, latex_root):
        directory.mkdir(parents=True, exist_ok=True)

    formats = tuple(
        extension.strip().lower()
        for extension in args.formats.split(",")
        if extension.strip()
    )
    allowed = {"png", "pdf", "svg"}
    if not formats or any(extension not in allowed for extension in formats):
        raise ValueError(f"Formats must be drawn from {sorted(allowed)}")
    figure_outputs = save_figure(
        figure,
        figure_root / "supplementary_mechanism_event_ssl_spatial_comparison",
        formats,
    )

    summary = pd.DataFrame(
        [
            {
                **join_audit,
                "common_drift_cells": common_drift_cells,
                "common_landscape_cells": common_landscape_cells,
                "drift_vmax": float(args.drift_vmax),
                "potential_vlim": float(args.potential_vlim),
                **layout_audit,
            }
        ]
    )
    summary_path = write_table(
        summary,
        table_root / "supplementary_mechanism_event_ssl_spatial_comparison_audit",
    )
    caption_path = latex_root / "supplementary_mechanism_event_ssl_spatial_comparison_caption.tex"
    write_caption(caption_path)

    output_manifest = {
        "script": Path(__file__).name,
        "primary_coordinates": PRIMARY_COORDINATES,
        "split": args.split,
        "comparison_manifest": str(manifest_path.resolve()),
        "comparison_manifest_sha256": sha256_file(manifest_path),
        "comparison_matrices": str(matrices_path.resolve()),
        "comparison_matrices_sha256": sha256_file(matrices_path),
        "mechanism_predictions": str(mechanism_path.resolve()),
        "event_ssl_predictions": str(event_path.resolve()),
        "prediction_hashes_verified": bool(args.verify_prediction_hashes),
        "prediction_hashes": prediction_hashes,
        "cross_model_fitting": False,
        "density_rows_subsampled": bool(args.max_density_rows > 0),
        "figure_outputs": figure_outputs,
        "caption": str(caption_path.resolve()),
        "audit_table": str(summary_path.resolve()),
        "join_audit": join_audit,
        "layout_audit": layout_audit,
    }
    save_json(
        output_manifest,
        metadata_root / "supplementary_mechanism_event_ssl_spatial_comparison_manifest.json",
    )
    print(f"Supplementary mechanism-Event-SSL figure written to {output_root}")


if __name__ == "__main__":
    main()
