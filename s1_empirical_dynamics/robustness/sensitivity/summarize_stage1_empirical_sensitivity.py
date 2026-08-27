#!/usr/bin/env python3
"""Create the compact supplementary table for Stage-1 sensitivity runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


VARIANT_ORDER = [
    "formal_10d",
    "memory_5d",
    "memory_20d",
    "activity_fast",
    "activity_slow",
]

FORMAL_PARAMETERS = {
    "tau_response_days": 10.0,
    "tau_activity_days": 10.0,
    "response_half_sat_min": 3.0,
    "explanation_half_sat_min": 2.5,
    "lecture_half_sat_min": 4.0,
    "idle_half_sat_days": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--sensitivity-root", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    return parser.parse_args()


def read_table(base: Path) -> pd.DataFrame:
    if base.suffix in {".parquet", ".csv", ".gz"} and base.exists():
        if base.suffix == ".parquet":
            return pd.read_parquet(base)
        return pd.read_csv(base)
    for suffix in (".parquet", ".csv.gz", ".csv"):
        path = base.with_suffix(suffix)
        if path.exists():
            if suffix == ".parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path)
    raise FileNotFoundError(f"Could not find table for {base}")



def bool_scalar(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", "", "nan", "none"}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value!r}")


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.map(bool_scalar).astype(bool)


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(mask.sum()) < 3:
        return np.nan
    aa = aa[mask] - float(np.mean(aa[mask]))
    bb = bb[mask] - float(np.mean(bb[mask]))
    denominator = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denominator) if denominator > 1e-12 else np.nan


def field_agreement(left: pd.DataFrame, right: pd.DataFrame) -> Dict[str, float]:
    required = {"x_bin", "y_bin", "drift_M", "drift_Psi", "drift_supported"}
    for name, frame in (("left", left), ("right", right)):
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{name} field table is missing columns: {sorted(missing)}")
    merged = left[list(required)].merge(
        right[list(required)],
        on=["x_bin", "y_bin"],
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    common = (
        bool_series(merged["drift_supported_left"])
        & bool_series(merged["drift_supported_right"])
        & np.isfinite(pd.to_numeric(merged["drift_M_left"], errors="coerce"))
        & np.isfinite(pd.to_numeric(merged["drift_Psi_left"], errors="coerce"))
        & np.isfinite(pd.to_numeric(merged["drift_M_right"], errors="coerce"))
        & np.isfinite(pd.to_numeric(merged["drift_Psi_right"], errors="coerce"))
    )
    data = merged.loc[common]
    if data.empty:
        return {
            "common_supported_cells": 0,
            "vector_correlation": np.nan,
            "mean_local_cosine": np.nan,
            "speed_correlation": np.nan,
            "component_rmse": np.nan,
        }
    lu = data["drift_M_left"].to_numpy(dtype=float)
    lv = data["drift_Psi_left"].to_numpy(dtype=float)
    ru = data["drift_M_right"].to_numpy(dtype=float)
    rv = data["drift_Psi_right"].to_numpy(dtype=float)
    ls = np.sqrt(lu * lu + lv * lv)
    rs = np.sqrt(ru * ru + rv * rv)
    cosine_mask = (ls > 1e-12) & (rs > 1e-12)
    local_cosine = np.nan
    if np.any(cosine_mask):
        local_cosine = float(
            np.mean(
                (lu[cosine_mask] * ru[cosine_mask] + lv[cosine_mask] * rv[cosine_mask])
                / (ls[cosine_mask] * rs[cosine_mask])
            )
        )
    return {
        "common_supported_cells": int(len(data)),
        "vector_correlation": pearson_safe(
            np.concatenate([lu, lv]),
            np.concatenate([ru, rv]),
        ),
        "mean_local_cosine": local_cosine,
        "speed_correlation": pearson_safe(ls, rs),
        "component_rmse": float(
            math.sqrt(np.mean(np.concatenate([lu - ru, lv - rv]) ** 2))
        ),
    }


def coordinate_root(output_root: Path) -> Path:
    return output_root / "stage1" / "dynamics" / "coordinate_analysis" / "MR_PsiA"


def region_root(output_root: Path) -> Path:
    return output_root / "stage1" / "dynamics" / "candidate_regions" / "MR_PsiA"


def load_summary(output_root: Path) -> Dict[str, object]:
    path = coordinate_root(output_root) / "coordinate_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing coordinate summary: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_field(output_root: Path, split: str) -> pd.DataFrame:
    suffix = "_output_only" if split == "B_confirm" else ""
    return read_table(coordinate_root(output_root) / f"{split}_publication_field_grid{suffix}")


def load_primary_region(output_root: Path) -> Dict[str, object]:
    table = read_table(region_root(output_root) / "training_flow_defined_convergence_regions")
    if table.empty:
        return {}
    if "primary_convergence_region" in table.columns:
        primary = table[bool_series(table["primary_convergence_region"])]
        if not primary.empty:
            return primary.iloc[0].to_dict()
    return table.iloc[0].to_dict()


def variant_parameters(output_root: Path, formal: bool) -> Dict[str, float]:
    if formal:
        return dict(FORMAL_PARAMETERS)
    path = output_root / "stage1" / "metadata" / "stage1_sensitivity_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing sensitivity manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    values = manifest.get("variant_parameters")
    if not isinstance(values, dict):
        raise ValueError(f"Missing variant_parameters in {path}")
    return {key: float(value) for key, value in values.items()}


def row_for_variant(
    variant: str,
    output_root: Path,
    formal_val_field: pd.DataFrame,
    formal: bool,
) -> Dict[str, object]:
    summary = load_summary(output_root)
    train_field = load_field(output_root, "A_train")
    val_field = load_field(output_root, "A_val")
    train_val = field_agreement(train_field, val_field)
    val_formal = field_agreement(val_field, formal_val_field)
    primary = load_primary_region(output_root)
    fallback = bool_scalar(primary.get("flow_defined_fallback_used", False)) if primary else True
    qualified = bool_scalar(primary.get("dynamically_qualified", False)) if primary else False
    parameters = variant_parameters(output_root, formal=formal)
    return {
        "variant": variant,
        **parameters,
        "validation_drift_supported_cells": summary.get("validation_global_drift_supported_cells"),
        "train_validation_common_supported_cells": train_val["common_supported_cells"],
        "train_validation_drift_vector_correlation": train_val["vector_correlation"],
        "train_validation_mean_local_drift_cosine": train_val["mean_local_cosine"],
        "train_validation_drift_speed_correlation": train_val["speed_correlation"],
        "validation_vs_formal_common_supported_cells": val_formal["common_supported_cells"],
        "validation_vs_formal_drift_vector_correlation": val_formal["vector_correlation"],
        "validation_vs_formal_mean_local_drift_cosine": val_formal["mean_local_cosine"],
        "validation_vs_formal_drift_speed_correlation": val_formal["speed_correlation"],
        "validation_vs_formal_drift_component_rmse": val_formal["component_rmse"],
        "validation_negative_divergence_occupancy_fraction": summary.get(
            "validation_global_weighted_negative_divergence_fraction_interior_only"
        ),
        "validation_weighted_mean_local_divergence": summary.get(
            "validation_global_weighted_mean_local_divergence_interior_only"
        ),
        "validation_frozen_core_occupancy_mass_fraction": summary.get(
            "validation_frozen_primary_convergence_occupancy_mass_fraction"
        ),
        "validation_frozen_shell_inward_fraction": summary.get(
            "validation_frozen_primary_convergence_flow_weighted_shell_fraction_inward"
        ),
        "validation_frozen_core_to_shell_speed_ratio": summary.get(
            "validation_frozen_primary_convergence_flow_core_to_shell_speed_ratio"
        ),
        "training_core_center_M": primary.get("convergence_center_M", np.nan),
        "training_core_center_Psi": primary.get("convergence_center_Psi", np.nan),
        "training_core_dynamically_qualified": qualified,
        "training_core_fallback_used": fallback,
        "interpretation_status": (
            "formal setting"
            if formal
            else "formal core criteria retained"
            if (primary and qualified and not fallback)
            else "do not count as positive core-robustness evidence"
        ),
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    baseline_root = args.baseline_root.resolve()
    sensitivity_root = args.sensitivity_root.resolve()
    formal_val_field = load_field(baseline_root, "A_val")

    roots = {
        "formal_10d": baseline_root,
        "memory_5d": sensitivity_root / "memory_5d",
        "memory_20d": sensitivity_root / "memory_20d",
        "activity_fast": sensitivity_root / "activity_fast",
        "activity_slow": sensitivity_root / "activity_slow",
    }
    rows = [
        row_for_variant(
            variant,
            roots[variant],
            formal_val_field,
            formal=(variant == "formal_10d"),
        )
        for variant in VARIANT_ORDER
    ]
    output = pd.DataFrame(rows)
    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_base.with_suffix(".csv")
    json_path = args.output_base.with_suffix(".json")
    output.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(rows), handle, indent=2)
    print(f"Wrote sensitivity summary: {csv_path}")
    print(f"Wrote sensitivity summary: {json_path}")


if __name__ == "__main__":
    main()
