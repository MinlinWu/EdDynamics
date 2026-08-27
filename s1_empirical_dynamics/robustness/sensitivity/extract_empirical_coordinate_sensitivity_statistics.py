#!/usr/bin/env python3
"""Extract the Stage-1 memory and activity-quality sensitivity report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

EPS = 1e-12
DEFAULT_BASELINE_ROOT = Path(
    os.environ.get("EDNET_KT4_OUTPUT_ROOT", "/data/datasets/KT4/outputs_KT4")
)
DEFAULT_SENSITIVITY_ROOT = Path(
    os.environ.get(
        "EDNET_STAGE1_SENSITIVITY_ROOT",
        "/data/datasets/KT4/outputs_KT4/stage1_sensitivity",
    )
)
COORDINATE = "MR_PsiA"
EXPECTED_SPLITS = {"A_train": 178749, "A_val": 59583, "B_confirm": 59583}
FIXED_PARAMETERS = {
    "EDNET_STAGE1_A_TRAIN_USERS": "178749",
    "EDNET_STAGE1_A_VAL_USERS": "59583",
    "EDNET_STAGE1_B_CONFIRM_USERS": "59583",
    "EDNET_STAGE1_ALLOW_SMALL_DEV_SPLIT": "0",
    "EDNET_STAGE1_RANDOM_STATE": "42",
    "EDNET_STAGE1_EVIDENCE_MATURITY_SCALE": "20.0",
    "EDNET_STAGE1_TAG_PRIOR_KAPPA": "20.0",
    "EDNET_STAGE1_ITEM_PRIOR_KAPPA": "50.0",
    "EDNET_STAGE1_OBSERVATION_HORIZON_DAYS": "7.0",
    "EDNET_STAGE1_LONG_GAP_DAYS": "7.0",
    "EDNET_STAGE1_MAX_SUPPORT_EPISODE_ACTIVE": "1.0",
    "EDNET_STAGE1_SIGNED_GRID_N": "41",
    "EDNET_STAGE1_MIN_STATE_BIN_COUNT": "50",
    "EDNET_STAGE1_MIN_DRIFT_BIN_COUNT": "30",
    "EDNET_STAGE1_MIN_CELL_USERS": "5",
    "EDNET_STAGE1_CONVERGENCE_SPEED_QUANTILE": "0.60",
    "EDNET_STAGE1_CONVERGENCE_NEGATIVE_DIVERGENCE_QUANTILE": "0.80",
    "EDNET_STAGE1_CONVERGENCE_RATIO_QUANTILE": "0.60",
    "EDNET_STAGE1_CONVERGENCE_MIN_CELLS": "4",
    "EDNET_STAGE1_CONVERGENCE_SHELL_RADIUS": "0.35",
}
FIELD_COLUMNS = [
    "x_bin",
    "y_bin",
    "M_center",
    "Psi_center",
    "occupancy_probability",
    "drift_M",
    "drift_Psi",
    "drift_supported",
]


@dataclass(frozen=True)
class Setting:
    setting_id: str
    label: str
    analysis: str
    tau_response_days: float
    tau_activity_days: float
    response_half_sat_min: float
    explanation_half_sat_min: float
    lecture_half_sat_min: float
    idle_half_sat_days: float

    def parameters(self) -> Dict[str, float]:
        return {
            "tau_response_days": self.tau_response_days,
            "tau_activity_days": self.tau_activity_days,
            "response_half_sat_min": self.response_half_sat_min,
            "explanation_half_sat_min": self.explanation_half_sat_min,
            "lecture_half_sat_min": self.lecture_half_sat_min,
            "idle_half_sat_days": self.idle_half_sat_days,
        }


SETTINGS = [
    Setting("formal_10d", "Formal", "formal", 10.0, 10.0, 3.0, 2.5, 4.0, 1.0),
    Setting("memory_5d", "Memory 5 d", "memory sensitivity", 5.0, 5.0, 3.0, 2.5, 4.0, 1.0),
    Setting("memory_20d", "Memory 20 d", "memory sensitivity", 20.0, 20.0, 3.0, 2.5, 4.0, 1.0),
    Setting("activity_fast", "Activity fast", "activity-quality sensitivity", 10.0, 10.0, 2.0, 2.0, 3.0, 0.5),
    Setting("activity_slow", "Activity slow", "activity-quality sensitivity", 10.0, 10.0, 4.0, 4.0, 6.0, 2.0),
]


class SourceRegistry:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self.names: set[str] = set()

    def add(
        self,
        name: str,
        path: Path,
        rows: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> None:
        if name in self.names:
            return
        resolved = path.resolve()
        self.records.append(
            {
                "source": name,
                "path": str(resolved),
                "sha256": sha256_file(resolved),
                "bytes": int(resolved.stat().st_size),
                "rows_loaded": rows,
                "columns_loaded": ",".join(columns or []),
            }
        )
        self.names.add(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--sensitivity-root", type=Path, default=DEFAULT_SENSITIVITY_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def resolve_table(base: Path) -> Path:
    if base.exists() and base.is_file():
        return base
    for suffix in (".parquet", ".csv.gz", ".csv"):
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find table for {base}")


def read_table(base: Path, name: str, registry: SourceRegistry) -> pd.DataFrame:
    path = resolve_table(base)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    registry.add(name, path, len(frame), list(frame.columns))
    return frame


def read_json(path: Path, name: str, registry: SourceRegistry) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON source: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    registry.add(name, path)
    return value


def formal_manifest_path(root: Path) -> Path:
    candidates = sorted((root / "stage1" / "metadata").glob("stage1_empirical*_manifest.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one formal empirical manifest; found {len(candidates)}.")
    return candidates[0]


def coordinate_root(root: Path) -> Path:
    return root / "stage1" / "dynamics" / "coordinate_analysis" / COORDINATE


def region_root(root: Path) -> Path:
    return root / "stage1" / "dynamics" / "candidate_regions" / COORDINATE


def setting_root(setting: Setting, baseline_root: Path, sensitivity_root: Path) -> Path:
    return baseline_root if setting.setting_id == "formal_10d" else sensitivity_root / setting.setting_id


def bool_scalar(value: Any) -> bool:
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


def scalar(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row.get(key, np.nan))
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def pearson_safe(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return np.nan
    a = a[mask] - float(np.mean(a[mask]))
    b = b[mask] - float(np.mean(b[mask]))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > EPS else np.nan


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    p = np.asarray(left, dtype=float)
    q = np.asarray(right, dtype=float)
    p = np.where(np.isfinite(p) & (p >= 0), p, 0.0)
    q = np.where(np.isfinite(q) & (q >= 0), q, 0.0)
    p = p / max(float(p.sum()), EPS)
    q = q / max(float(q.sum()), EPS)
    midpoint = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + EPS) / (b[mask] + EPS))))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def validate_field(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = set(FIELD_COLUMNS).difference(frame.columns)
    if missing:
        raise KeyError(f"{name} is missing columns: {sorted(missing)}")
    field = frame[FIELD_COLUMNS].copy()
    field["x_bin"] = pd.to_numeric(field["x_bin"], errors="raise").astype(int)
    field["y_bin"] = pd.to_numeric(field["y_bin"], errors="raise").astype(int)
    if field.duplicated(["x_bin", "y_bin"]).any():
        raise RuntimeError(f"{name} contains duplicate grid cells.")
    field = field.sort_values(["x_bin", "y_bin"], kind="mergesort").reset_index(drop=True)
    if len(field) != 1600 or field["x_bin"].nunique() != 40 or field["y_bin"].nunique() != 40:
        raise RuntimeError(f"{name} does not use the fixed 40 x 40 grid.")
    return field


def field_agreement(left: pd.DataFrame, right: pd.DataFrame) -> Dict[str, float]:
    merged = left.merge(
        right,
        on=["x_bin", "y_bin"],
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    for coordinate in ("M_center", "Psi_center"):
        a = pd.to_numeric(merged[f"{coordinate}_left"], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(merged[f"{coordinate}_right"], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(a, b, atol=1e-12, rtol=0.0, equal_nan=True):
            raise RuntimeError("Compared fields do not share identical physical cell centres.")
    occupancy_js = js_divergence(
        pd.to_numeric(merged["occupancy_probability_left"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(merged["occupancy_probability_right"], errors="coerce").to_numpy(dtype=float),
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
            "occupancy_js": occupancy_js,
            "common_cells": 0,
            "vector_correlation": np.nan,
            "local_cosine": np.nan,
            "weighted_local_cosine": np.nan,
            "speed_correlation": np.nan,
            "component_rmse": np.nan,
        }
    lm = data["drift_M_left"].to_numpy(dtype=float)
    lp = data["drift_Psi_left"].to_numpy(dtype=float)
    rm = data["drift_M_right"].to_numpy(dtype=float)
    rp = data["drift_Psi_right"].to_numpy(dtype=float)
    ls = np.sqrt(lm * lm + lp * lp)
    rs = np.sqrt(rm * rm + rp * rp)
    valid_cosine = (ls > EPS) & (rs > EPS)
    local = np.full(len(data), np.nan, dtype=float)
    local[valid_cosine] = (
        lm[valid_cosine] * rm[valid_cosine] + lp[valid_cosine] * rp[valid_cosine]
    ) / (ls[valid_cosine] * rs[valid_cosine])
    local_mask = np.isfinite(local)
    local_cosine = float(np.mean(local[local_mask])) if np.any(local_mask) else np.nan
    weights = pd.to_numeric(data["occupancy_probability_right"], errors="coerce").to_numpy(dtype=float)
    weighted_mask = local_mask & np.isfinite(weights) & (weights >= 0)
    weighted_local = np.nan
    if np.any(weighted_mask) and float(weights[weighted_mask].sum()) > 0:
        weighted_local = float(
            np.sum(weights[weighted_mask] * local[weighted_mask])
            / np.sum(weights[weighted_mask])
        )
    residual = np.concatenate([lm - rm, lp - rp])
    return {
        "occupancy_js": occupancy_js,
        "common_cells": int(len(data)),
        "vector_correlation": pearson_safe(np.concatenate([lm, lp]), np.concatenate([rm, rp])),
        "local_cosine": local_cosine,
        "weighted_local_cosine": weighted_local,
        "speed_correlation": pearson_safe(ls, rs),
        "component_rmse": float(math.sqrt(np.mean(residual * residual))),
    }


def select_split(frame: pd.DataFrame, split: str, name: str) -> Dict[str, Any]:
    if "split" not in frame.columns:
        raise KeyError(f"{name} is missing split.")
    selected = frame[frame["split"].astype(str) == split]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one {split} row in {name}; found {len(selected)}.")
    return selected.iloc[0].to_dict()


def select_primary(frame: pd.DataFrame, name: str) -> Dict[str, Any]:
    if frame.empty:
        raise RuntimeError(f"{name} is empty.")
    if "primary_convergence_region" in frame.columns:
        selected = frame[bool_series(frame["primary_convergence_region"])]
        if len(selected) != 1:
            raise RuntimeError(f"Expected one primary convergence region in {name}.")
        return selected.iloc[0].to_dict()
    return frame.iloc[0].to_dict()


def validate_formal_manifest(manifest: Mapping[str, Any]) -> str:
    split_manifest = manifest.get("split_manifest")
    sizes = split_manifest.get("sizes") if isinstance(split_manifest, dict) else None
    if not isinstance(sizes, dict):
        raise RuntimeError("Formal manifest is missing split sizes.")
    for split, expected in EXPECTED_SPLITS.items():
        if int(sizes.get(split, -1)) != expected:
            raise RuntimeError(f"Formal {split} size is invalid.")
    primary = manifest.get("primary_state")
    if not isinstance(primary, dict) or primary.get("coordinates") != [
        "M_response_prebalanced",
        "activity_alignment_order_Psi",
    ]:
        raise RuntimeError("Formal manifest does not identify the expected primary state.")
    contract = manifest.get("convergence_core_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("Formal manifest is missing the convergence-core contract.")
    if contract.get("fit_split") != "A_train":
        raise RuntimeError("The formal convergence core was not defined on A_train.")
    if bool(contract.get("occupancy_used_for_selection", True)):
        raise RuntimeError("The formal convergence core used occupancy for selection.")
    if not bool(contract.get("validation_thresholds_frozen", False)):
        raise RuntimeError("The formal convergence thresholds were not frozen.")
    preprocess = manifest.get("input_preprocess_manifest")
    if not isinstance(preprocess, dict) or not preprocess.get("output_root"):
        raise RuntimeError("Formal manifest is missing the preprocessing source.")
    return str(preprocess["output_root"])


def validate_variant_manifest(
    manifest: Mapping[str, Any],
    setting: Setting,
    preprocess_root: str,
) -> None:
    if manifest.get("variant") != setting.setting_id:
        raise RuntimeError(f"Manifest does not match {setting.setting_id}.")
    parameters = manifest.get("variant_parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError(f"{setting.setting_id} is missing variant parameters.")
    for key, expected in setting.parameters().items():
        value = float(parameters.get(key, np.nan))
        if not np.isclose(value, expected, atol=1e-12, rtol=0.0):
            raise RuntimeError(f"{setting.setting_id} has an invalid {key} value.")
    fixed = manifest.get("fixed_publication_parameters")
    if not isinstance(fixed, dict):
        raise RuntimeError(f"{setting.setting_id} is missing fixed parameters.")
    for key, expected in FIXED_PARAMETERS.items():
        if str(fixed.get(key)) != expected:
            raise RuntimeError(f"{setting.setting_id} changed fixed parameter {key}.")
    preprocess = manifest.get("formal_input_preprocess_manifest")
    if not isinstance(preprocess, dict) or str(preprocess.get("output_root")) != preprocess_root:
        raise RuntimeError(f"{setting.setting_id} does not reuse the formal preprocessing output.")
    split_manifest = manifest.get("split_manifest")
    sizes = split_manifest.get("sizes") if isinstance(split_manifest, dict) else None
    if not isinstance(sizes, dict):
        raise RuntimeError(f"{setting.setting_id} is missing split sizes.")
    for split, expected in EXPECTED_SPLITS.items():
        if int(sizes.get(split, -1)) != expected:
            raise RuntimeError(f"{setting.setting_id} has an invalid {split} size.")
    scope = manifest.get("analysis_scope")
    expected_scope = {
        "B_confirm": "not processed or accessed",
        "construction_matched_null": "not rerun",
        "fixed_k6_mesostates": "not rerun",
        "minimal_mechanism": "not rerun",
        "Event_SSL": "not rerun",
    }
    if not isinstance(scope, dict):
        raise RuntimeError(f"{setting.setting_id} is missing analysis scope.")
    for key, expected in expected_scope.items():
        if str(scope.get(key)) != expected:
            raise RuntimeError(f"{setting.setting_id} violates the scope for {key}.")


def load_setting(
    setting: Setting,
    root: Path,
    registry: SourceRegistry,
) -> Dict[str, Any]:
    coordinate = coordinate_root(root)
    regions = region_root(root)
    train_field = validate_field(
        read_table(
            coordinate / "A_train_publication_field_grid",
            f"{setting.setting_id}_A_train_field",
            registry,
        ),
        f"{setting.setting_id} A_train field",
    )
    val_field = validate_field(
        read_table(
            coordinate / "A_val_publication_field_grid",
            f"{setting.setting_id}_A_val_field",
            registry,
        ),
        f"{setting.setting_id} A_val field",
    )
    contraction = select_split(
        read_table(
            coordinate / "global_field_contraction_summaries",
            f"{setting.setting_id}_global_contraction",
            registry,
        ),
        "A_val",
        f"{setting.setting_id} global contraction",
    )
    training_region = select_primary(
        read_table(
            regions / "training_flow_defined_convergence_regions",
            f"{setting.setting_id}_training_regions",
            registry,
        ),
        f"{setting.setting_id} training regions",
    )
    validation_region_table = read_table(
        regions / "validation_frozen_training_convergence_region",
        f"{setting.setting_id}_validation_frozen_region",
        registry,
    )
    if len(validation_region_table) != 1:
        raise RuntimeError(
            f"Expected one frozen A_val convergence-region row for {setting.setting_id}."
        )
    return {
        "train_field": train_field,
        "val_field": val_field,
        "contraction": contraction,
        "training_region": training_region,
        "validation_region": validation_region_table.iloc[0].to_dict(),
    }


def add_gate(
    gates: List[Dict[str, Any]],
    setting: str,
    category: str,
    gate: str,
    passed: bool,
    detail: str,
) -> None:
    gates.append(
        {
            "setting": setting,
            "category": category,
            "gate": gate,
            "passed": bool(passed),
            "detail": detail,
        }
    )


def build_row(
    setting: Setting,
    data: Mapping[str, Any],
    formal_val_field: pd.DataFrame,
    gates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    train_val = field_agreement(data["train_field"], data["val_field"])
    val_formal = field_agreement(data["val_field"], formal_val_field)
    contraction = data["contraction"]
    training = data["training_region"]
    validation = data["validation_region"]
    qualified = bool_scalar(training.get("dynamically_qualified", False))
    fallback = bool_scalar(training.get("flow_defined_fallback_used", False))
    negative_fraction = scalar(
        contraction, "weighted_negative_divergence_fraction_interior_only"
    )
    inward_fraction = scalar(validation, "flow_weighted_shell_fraction_inward")
    speed_ratio = scalar(validation, "flow_core_to_shell_speed_ratio")
    directional = bool(
        qualified
        and not fallback
        and np.isfinite(negative_fraction)
        and negative_fraction > 0.5
        and np.isfinite(inward_fraction)
        and inward_fraction > 0.5
        and np.isfinite(speed_ratio)
        and speed_ratio < 1.0
    )
    add_gate(
        gates,
        setting.setting_id,
        "scientific",
        "training_core_dynamically_qualified",
        qualified,
        "The A_train flow-defined core satisfies the formal qualification criteria.",
    )
    add_gate(
        gates,
        setting.setting_id,
        "scientific",
        "training_core_without_fallback",
        not fallback,
        "The A_train core was identified without the fallback branch.",
    )
    add_gate(
        gates,
        setting.setting_id,
        "scientific",
        "contractive_inward_slowing_directions_retained",
        directional,
        "A_val retains negative-divergence occupancy above one half, frozen-shell inward flow above one half, and a core-to-shell speed ratio below one.",
    )
    return {
        "setting_id": setting.setting_id,
        "setting": setting.label,
        "analysis": setting.analysis,
        **setting.parameters(),
        "A_val_users": int(scalar(contraction, "users")),
        "A_val_valid_state_rows": int(scalar(contraction, "valid_state_rows")),
        "A_val_valid_drift_rows": int(scalar(contraction, "valid_drift_rows")),
        "A_val_state_supported_cells": int(scalar(contraction, "state_supported_cells")),
        "A_val_drift_supported_cells": int(scalar(contraction, "drift_supported_cells")),
        "A_val_interior_divergence_cells": int(scalar(contraction, "interior_divergence_cells")),
        "A_train_A_val_occupancy_js": train_val["occupancy_js"],
        "A_train_A_val_common_drift_cells": train_val["common_cells"],
        "A_train_A_val_field_vector_correlation": train_val["vector_correlation"],
        "A_train_A_val_mean_local_drift_cosine": train_val["local_cosine"],
        "A_train_A_val_occupancy_weighted_local_drift_cosine": train_val[
            "weighted_local_cosine"
        ],
        "A_train_A_val_drift_speed_correlation": train_val["speed_correlation"],
        "A_train_A_val_drift_component_rmse": train_val["component_rmse"],
        "A_val_occupancy_js_vs_formal": val_formal["occupancy_js"],
        "A_val_common_drift_cells_vs_formal": val_formal["common_cells"],
        "A_val_field_vector_correlation_vs_formal": val_formal["vector_correlation"],
        "A_val_mean_local_drift_cosine_vs_formal": val_formal["local_cosine"],
        "A_val_occupancy_weighted_local_drift_cosine_vs_formal": val_formal[
            "weighted_local_cosine"
        ],
        "A_val_drift_speed_correlation_vs_formal": val_formal["speed_correlation"],
        "A_val_drift_component_rmse_vs_formal": val_formal["component_rmse"],
        "A_val_negative_divergence_occupancy_fraction": negative_fraction,
        "A_val_weighted_mean_local_divergence": scalar(
            contraction, "weighted_mean_local_divergence_interior_only"
        ),
        "A_val_frozen_core_occupancy_mass_fraction": scalar(
            validation, "occupancy_mass_fraction"
        ),
        "A_val_frozen_shell_inward_fraction": inward_fraction,
        "A_val_frozen_shell_inward_cosine": scalar(
            validation, "flow_weighted_shell_inward_cosine"
        ),
        "A_val_frozen_core_to_shell_speed_ratio": speed_ratio,
        "A_train_core_center_M": scalar(training, "convergence_center_M"),
        "A_train_core_center_Psi": scalar(training, "convergence_center_Psi"),
        "A_train_core_cells": int(scalar(training, "region_cells_total")),
        "A_train_core_dynamically_qualified": qualified,
        "A_train_core_fallback_used": fallback,
        "directional_result_retained": directional,
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(number):
        return "NA"
    if number == 0:
        return "0"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.{digits}g}"


def yes_no(value: Any) -> str:
    return "yes" if bool_scalar(value) else "no"


def compact_table(statistics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for row in statistics.to_dict(orient="records"):
        rows.append(
            {
                "Setting": str(row["setting"]),
                "Memory τR/τA (d)": f"{fmt(row['tau_response_days'])}/{fmt(row['tau_activity_days'])}",
                "Activity half-saturation": (
                    f"resp/exp/lec={fmt(row['response_half_sat_min'])}/"
                    f"{fmt(row['explanation_half_sat_min'])}/"
                    f"{fmt(row['lecture_half_sat_min'])} min; "
                    f"idle={fmt(row['idle_half_sat_days'])} d"
                ),
                "A_val coverage": (
                    f"users={fmt(row['A_val_users'])}; "
                    f"state/drift rows={fmt(row['A_val_valid_state_rows'])}/"
                    f"{fmt(row['A_val_valid_drift_rows'])}; "
                    f"state/drift cells={fmt(row['A_val_state_supported_cells'])}/"
                    f"{fmt(row['A_val_drift_supported_cells'])}; "
                    f"interior={fmt(row['A_val_interior_divergence_cells'])}"
                ),
                "A_train-A_val replication": (
                    f"JS={fmt(row['A_train_A_val_occupancy_js'])}; "
                    f"r_b={fmt(row['A_train_A_val_field_vector_correlation'])}; "
                    f"c_local={fmt(row['A_train_A_val_mean_local_drift_cosine'])}; "
                    f"c_w={fmt(row['A_train_A_val_occupancy_weighted_local_drift_cosine'])}; "
                    f"r_speed={fmt(row['A_train_A_val_drift_speed_correlation'])}; "
                    f"RMSE={fmt(row['A_train_A_val_drift_component_rmse'])}; "
                    f"n={fmt(row['A_train_A_val_common_drift_cells'])}"
                ),
                "A_val vs formal": (
                    f"JS={fmt(row['A_val_occupancy_js_vs_formal'])}; "
                    f"r_b={fmt(row['A_val_field_vector_correlation_vs_formal'])}; "
                    f"c_local={fmt(row['A_val_mean_local_drift_cosine_vs_formal'])}; "
                    f"c_w={fmt(row['A_val_occupancy_weighted_local_drift_cosine_vs_formal'])}; "
                    f"r_speed={fmt(row['A_val_drift_speed_correlation_vs_formal'])}; "
                    f"RMSE={fmt(row['A_val_drift_component_rmse_vs_formal'])}; "
                    f"n={fmt(row['A_val_common_drift_cells_vs_formal'])}"
                ),
                "Global contraction": (
                    f"f_neg={fmt(row['A_val_negative_divergence_occupancy_fraction'])}; "
                    f"mean_div={fmt(row['A_val_weighted_mean_local_divergence'])}"
                ),
                "Frozen basin": (
                    f"mass={fmt(row['A_val_frozen_core_occupancy_mass_fraction'])}; "
                    f"f_in={fmt(row['A_val_frozen_shell_inward_fraction'])}; "
                    f"c_in={fmt(row['A_val_frozen_shell_inward_cosine'])}; "
                    f"R_cs={fmt(row['A_val_frozen_core_to_shell_speed_ratio'])}"
                ),
                "A_train core": (
                    f"center=({fmt(row['A_train_core_center_M'])},"
                    f"{fmt(row['A_train_core_center_Psi'])}); "
                    f"cells={fmt(row['A_train_core_cells'])}; "
                    f"qualified={yes_no(row['A_train_core_dynamically_qualified'])}; "
                    f"fallback={yes_no(row['A_train_core_fallback_used'])}"
                ),
                "Directional result": (
                    "retained" if bool_scalar(row["directional_result_retained"]) else "not retained"
                ),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    def escape(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(escape(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(escape(value) for value in values) + " |")
    return "\n".join(lines)


def build_report(
    table: pd.DataFrame,
    statistics: pd.DataFrame,
    gates: pd.DataFrame,
) -> str:
    sensitivity = statistics[statistics["analysis"] != "formal"]
    retained = int(sensitivity["directional_result_retained"].astype(bool).sum())
    fallback = sensitivity[sensitivity["A_train_core_fallback_used"].astype(bool)][
        "setting"
    ].astype(str).tolist()
    unqualified = sensitivity[
        ~sensitivity["A_train_core_dynamically_qualified"].astype(bool)
    ]["setting"].astype(str).tolist()
    contract = gates[gates["category"] == "contract"]
    contract_passed = bool(contract["passed"].astype(bool).all()) if not contract.empty else True
    return "\n".join(
        [
            "# Empirical coordinate-sensitivity numerical report",
            "",
            "This report reads the completed formal Stage-1 output and the four frozen A_train/A_val sensitivity outputs. It does not rebuild coordinates, re-estimate priors, access B_confirm for a sensitivity result, rerun the construction-matched null, refit the fixed K=6 partition, or rerun the minimal mechanism or Event-SSL.",
            "",
            "JS denotes Jensen-Shannon divergence; r_b is flattened drift-vector correlation; c_local is mean local drift cosine; c_w is A_val-occupancy-weighted local cosine; r_speed is drift-speed correlation; f_neg is negative-divergence occupancy; f_in and c_in are flow-weighted shell inward fraction and cosine; R_cs is the core-to-shell speed ratio.",
            "",
            "## Supplementary table: memory and activity-quality sensitivity",
            "",
            markdown_table(table),
            "",
            f"All contract checks passed: {'yes' if contract_passed else 'no'}. The contractive, inward and slowing directions were retained in {retained}/{len(sensitivity)} sensitivity settings. Fallback-defined cores: {', '.join(fallback) if fallback else 'none'}. Non-qualified training cores: {', '.join(unqualified) if unqualified else 'none'}.",
            "",
            "The directional summary is descriptive and introduces no new significance test. The table evaluates only the declared memory and activity-mass mappings; it does not establish coordinate uniqueness or parameter robustness of the construction-matched null or downstream models.",
            "",
        ]
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or pd.isna(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    baseline_root = args.baseline_root.resolve()
    sensitivity_root = args.sensitivity_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (sensitivity_root / "summary").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = SourceRegistry()
    gates: List[Dict[str, Any]] = []

    formal_manifest = read_json(
        formal_manifest_path(baseline_root), "formal_stage1_manifest", registry
    )
    preprocess_root = validate_formal_manifest(formal_manifest)
    add_gate(
        gates,
        "all",
        "contract",
        "formal_stage1_contract",
        True,
        "The formal split, primary-state, preprocessing and A_train-to-A_val convergence contracts are valid.",
    )

    formal_data = load_setting(SETTINGS[0], baseline_root, registry)
    formal_val_field = formal_data["val_field"]
    rows: List[Dict[str, Any]] = []

    for setting in SETTINGS:
        root = setting_root(setting, baseline_root, sensitivity_root)
        if setting.setting_id == "formal_10d":
            data = formal_data
        else:
            manifest = read_json(
                root / "stage1" / "metadata" / "stage1_sensitivity_manifest.json",
                f"{setting.setting_id}_manifest",
                registry,
            )
            validate_variant_manifest(manifest, setting, preprocess_root)
            add_gate(
                gates,
                setting.setting_id,
                "contract",
                "frozen_input_and_scope_contract",
                True,
                "The formal preprocessing, priors, user split, grid, support thresholds and A_train/A_val analysis scope are retained.",
            )
            data = load_setting(setting, root, registry)
        rows.append(build_row(setting, data, formal_val_field, gates))
        add_gate(
            gates,
            setting.setting_id,
            "contract",
            "fixed_field_grid",
            True,
            "A_train and A_val use the fixed 40 x 40 physical grid with unique cells.",
        )

    statistics = pd.DataFrame(rows)
    compact = compact_table(statistics)
    quality = pd.DataFrame(gates)
    sources = pd.DataFrame(registry.records)

    report_path = output_dir / "empirical_coordinate_sensitivity_report.md"
    statistics_path = output_dir / "empirical_coordinate_sensitivity_statistics.csv"
    table_path = output_dir / "empirical_coordinate_sensitivity_table.csv"
    quality_path = output_dir / "empirical_coordinate_sensitivity_quality_gates.csv"
    sources_path = output_dir / "empirical_coordinate_sensitivity_source_audit.csv"
    manifest_path = output_dir / "empirical_coordinate_sensitivity_manifest.json"

    report_path.write_text(build_report(compact, statistics, quality), encoding="utf-8")
    statistics.to_csv(statistics_path, index=False)
    compact.to_csv(table_path, index=False)
    quality.to_csv(quality_path, index=False)
    sources.to_csv(sources_path, index=False)

    manifest = {
        "script": Path(__file__).name,
        "baseline_root": str(baseline_root),
        "sensitivity_root": str(sensitivity_root),
        "analysis_scope": {
            "A_train": "defines each setting's field criteria and convergence core",
            "A_val": "held-out sensitivity evaluation",
            "B_confirm": "not accessed for sensitivity statistics",
            "construction_matched_null": "not rerun",
            "fixed_k6_mesostates": "not rerun",
            "minimal_mechanism": "not rerun",
            "Event_SSL": "not rerun",
        },
        "settings": [
            {
                "setting_id": setting.setting_id,
                "label": setting.label,
                "analysis": setting.analysis,
                **setting.parameters(),
            }
            for setting in SETTINGS
        ],
        "outputs": {
            "report": str(report_path),
            "statistics": str(statistics_path),
            "supplementary_table": str(table_path),
            "quality_gates": str(quality_path),
            "source_audit": str(sources_path),
        },
        "quality_gates": json_safe(gates),
        "source_audit": json_safe(registry.records),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(f"Wrote report: {report_path}")
    print(f"Wrote statistics: {statistics_path}")
    print(f"Wrote supplementary table: {table_path}")
    print(f"Wrote quality gates: {quality_path}")
    print(f"Wrote source audit: {sources_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
