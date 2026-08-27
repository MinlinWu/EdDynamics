#!/usr/bin/env python3
from __future__ import annotations

"""Audit nonlinear state-only closure of the frozen Event-SSL macrostate."""

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import pickle
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from numpy.polynomial.hermite import hermgauss
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

EPS = 1e-12
PRIMARY_SEED = 42
EXPECTED_K = 6
FORMAL_SAMPLE_ROWS = 600000
FORMAL_SAMPLE_SEED = 42
FORMAL_RIDGE_ALPHA = 1.0
DEFAULT_KNOTS = (4, 5, 6)
DEFAULT_QUADRATIC_ALPHAS = (0.1, 1.0, 10.0)
DEFAULT_MEAN_ALPHAS = (0.1, 1.0, 10.0)
DEFAULT_VARIANCE_ALPHAS = (0.1, 1.0, 10.0)
DEFAULT_PERMUTATIONS = 50
DEFAULT_PERMUTATION_SEED = 20260805
DEFAULT_GH_ORDER = 5
DEFAULT_GH_AUDIT_ORDERS = (5, 7, 9, 11)
DEFAULT_GH_MAX_ORDER = 31
DEFAULT_GH_ORDER_STEP = 2
DEFAULT_GH_AUDIT_ROWS = 200000
DEFAULT_GH_MATRIX_TOL = 0.01
DEFAULT_GH_METRIC_TOL = 0.01
DEFAULT_GH_MIN_ORIGIN_ROWS = 100
DEFAULT_VARIANCE_CROSSFIT_FOLDS = 2
DEFAULT_VARIANCE_CROSSFIT_SEED = 20260806
VARIANCE_EPS = 1e-4
VARIANCE_MAX = 4.0
RHO_LIMIT = 0.95
RECONSTRUCTION_ATOL = 2e-5
METRIC_ATOL = 2e-6
TIE_TOL = 1e-6


@dataclass(frozen=True)
class SeedRoot:
    seed: int
    stage5_root: Path


@dataclass
class MeanClosure:
    n_knots: int
    degree: int
    alpha: float
    spline_m: SplineTransformer
    spline_psi: SplineTransformer
    scaler: StandardScaler
    regressor: Ridge

    def basis(self, state: np.ndarray) -> np.ndarray:
        values = np.asarray(state, dtype=np.float64)
        first = self.spline_m.transform(values[:, [0]])
        second = self.spline_psi.transform(values[:, [1]])
        tensor = np.einsum("ij,ik->ijk", first, second, optimize=True)
        return tensor.reshape(len(values), -1).astype(np.float32, copy=False)

    def features(self, state: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self.basis(state)).astype(np.float32, copy=False)

    def predict_raw(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(self.regressor.predict(self.features(state)), dtype=np.float64)

    def predict(self, state: np.ndarray) -> np.ndarray:
        return np.clip(self.predict_raw(state), -1.0, 1.0).astype(np.float32)


@dataclass
class GaussianClosure:
    mean: MeanClosure
    variance_alpha: float
    variance_regressor: Ridge
    log_variance_offset: np.ndarray
    rho: float
    variance_eps: float
    variance_max: float
    crossfit_folds: int
    crossfit_seed: int

    def predict_parameters(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = self.mean.predict(state).astype(np.float64)
        log_variance = np.asarray(
            self.variance_regressor.predict(self.mean.features(state)), dtype=np.float64
        ) + np.asarray(self.log_variance_offset, dtype=np.float64)[None, :]
        variance = np.clip(np.exp(log_variance), self.variance_eps, self.variance_max)
        return mean, variance, np.full(len(mean), float(self.rho), dtype=np.float64)


@dataclass(frozen=True)
class FixedPartition:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    standardized_centers: np.ndarray
    centers: np.ndarray
    k: int
    audit: Mapping[str, Any]


@dataclass
class SplitData:
    split: str
    user_id: np.ndarray
    step: np.ndarray
    target_current: np.ndarray
    target_next: np.ndarray
    predicted_current: np.ndarray
    quadratic_next: np.ndarray


@dataclass
class SeedContext:
    spec: SeedRoot
    supplied_stage5_root: Path
    stage5_training_root: Path
    stage5_root: Path
    input_root: Path
    checkpoint: Path
    artifacts_path: Path
    input_manifest: Dict[str, Any]
    evaluation_manifest: Dict[str, Any]
    training_manifest: Dict[str, Any]
    artifacts: Dict[str, Any]
    source_hashes: Dict[str, str]


def json_safe(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return json_safe(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(key): json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(obj), handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray, chunk_rows: int = 1000000) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    for start in range(0, len(array), int(chunk_rows)):
        digest.update(np.ascontiguousarray(array[start:start + int(chunk_rows)]).tobytes())
    return digest.hexdigest()


def splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint64) ^ np.uint64(int(seed) & ((1 << 64) - 1))
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def user_crossfit_fold(user_id: np.ndarray, folds: int, seed: int) -> np.ndarray:
    if int(folds) < 2:
        raise ValueError("Variance cross-fitting requires at least two folds.")
    return (splitmix64(np.asarray(user_id, dtype=np.uint64), int(seed)) % np.uint64(int(folds))).astype(np.int64)


def import_module(path: Path, name: str) -> Any:
    source = path.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location(name, str(source))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def find_table(base_or_path: Path) -> Path:
    path = Path(base_or_path)
    if path.exists() and path.is_file():
        return path
    for extension in (".parquet", ".csv.gz", ".csv"):
        candidate = path.with_suffix(extension)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find table for {base_or_path}")


def read_table(base_or_path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    path = find_table(base_or_path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    return pd.read_csv(path, usecols=list(columns) if columns is not None else None, low_memory=False)


def write_table(frame: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    parquet = base.with_suffix(".parquet")
    temporary = parquet.with_name(parquet.name + ".tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, parquet)
        return parquet
    except Exception:
        if temporary.exists():
            temporary.unlink()
        csv_path = base.with_suffix(".csv.gz")
        temporary_csv = csv_path.with_name(csv_path.name + ".tmp")
        frame.to_csv(temporary_csv, index=False, compression="gzip")
        os.replace(temporary_csv, csv_path)
        return csv_path


def parse_number_list(value: str, cast=float) -> Tuple[Any, ...]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("A non-empty comma-separated list is required.")
    return tuple(cast(item) for item in items)


def parse_seed_root(value: str) -> SeedRoot:
    if "=" not in value:
        raise ValueError("Each --seed-root must be SEED=/path/to/stage5_training_root")
    seed_raw, root_raw = value.split("=", 1)
    return SeedRoot(seed=int(seed_raw), stage5_root=Path(root_raw).expanduser().resolve())


def unique_paths(paths: Sequence[Path]) -> List[Path]:
    output: List[Path] = []
    seen = set()
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            output.append(resolved)
    return output


def discover_stage5_layout(supplied_root: Path) -> Tuple[Path, Path, Path, Path]:
    supplied = Path(supplied_root).expanduser().resolve()
    if not supplied.is_dir():
        raise FileNotFoundError(f"Stage-5 root does not exist: {supplied}")

    training_name = "stage5_macro_sufficiency_training_manifest.json"
    evaluation_name = "stage5_macro_sufficiency_evaluation_manifest.json"
    pairs: List[Tuple[Path, Path]] = [
        (supplied, supplied / "evaluation"),
        (supplied, supplied),
    ]
    if supplied.name in {"evaluation", "eval"}:
        pairs.insert(0, (supplied.parent, supplied))

    checked: List[str] = []
    matches: List[Tuple[Path, Path, Path, Path]] = []
    for training_root, evaluation_root in pairs:
        training_root = training_root.resolve()
        evaluation_root = evaluation_root.resolve()
        training_path = training_root / "metadata" / training_name
        evaluation_path = evaluation_root / "metadata" / evaluation_name
        checked.append(f"training={training_path}; evaluation={evaluation_path}")
        if training_path.is_file() and evaluation_path.is_file():
            matches.append((training_root, evaluation_root, training_path, evaluation_path))

    if not matches:
        message = "\n  ".join(checked)
        raise FileNotFoundError(
            f"Could not resolve the Stage-5 training/evaluation layout from {supplied}. "
            f"Checked:\n  {message}"
        )

    unique_matches: List[Tuple[Path, Path, Path, Path]] = []
    seen = set()
    for match in matches:
        key = tuple(str(path) for path in match[:2])
        if key not in seen:
            seen.add(key)
            unique_matches.append(match)
    if len(unique_matches) > 1:
        exact_child = [
            match for match in unique_matches
            if match[0] == supplied and match[1] == supplied / "evaluation"
        ]
        if len(exact_child) == 1:
            return exact_child[0]
        same_root = [match for match in unique_matches if match[0] == supplied and match[1] == supplied]
        if len(same_root) == 1:
            return same_root[0]
        layouts = "\n  ".join(
            f"training={match[0]}; evaluation={match[1]}" for match in unique_matches
        )
        raise RuntimeError(f"Ambiguous Stage-5 layout under {supplied}:\n  {layouts}")
    return unique_matches[0]


def resolve_recorded_path(
    value: Optional[str],
    local_candidates: Sequence[Path],
    label: str,
    relative_bases: Sequence[Path] = (),
) -> Path:
    candidates: List[Path] = []
    raw = str(value or "").strip()
    if raw:
        recorded = Path(raw).expanduser()
        if recorded.is_absolute():
            candidates.append(recorded)
        else:
            for base in relative_bases:
                candidates.append(Path(base) / recorded)
    candidates.extend(local_candidates)
    for candidate in unique_paths(candidates):
        if candidate.exists():
            return candidate
    listed = "\n  ".join(str(candidate) for candidate in unique_paths(candidates))
    raise FileNotFoundError(f"Could not resolve {label}. Checked:\n  {listed}")


def load_seed_context(spec: SeedRoot) -> SeedContext:
    supplied_root = spec.stage5_root.resolve()
    training_root, evaluation_root, training_path, evaluation_path = discover_stage5_layout(supplied_root)
    evaluation = load_json(evaluation_path)
    training = load_json(training_path)
    input_root = resolve_recorded_path(
        str(evaluation.get("input_root", training.get("input_root", "")) or ""),
        [],
        f"prepared input root for seed {spec.seed}",
        [training_root, evaluation_root],
    )
    checkpoint = resolve_recorded_path(
        str(evaluation.get("checkpoint", training.get("checkpoint", "")) or ""),
        [],
        f"Event-SSL checkpoint for seed {spec.seed}",
        [training_root, evaluation_root],
    )
    artifacts_path = resolve_recorded_path(
        str(evaluation.get("artifacts", "") or ""),
        [
            training_root / "metadata" / "stage5_macro_sufficiency_artifacts.pkl",
            evaluation_root / "metadata" / "stage5_macro_sufficiency_artifacts.pkl",
        ],
        f"Stage-5 artifacts for seed {spec.seed}",
        [training_root, evaluation_root],
    )
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    input_manifest_path = input_root / "metadata" / "stage4_input_manifest.json"
    input_manifest = load_json(input_manifest_path)
    with artifacts_path.open("rb") as handle:
        artifacts = pickle.load(handle)
    artifact_meta = dict(artifacts.get("meta", {}))
    checks = {
        "primary_coordinates": artifact_meta.get("primary_coordinates") == ["M", "Psi"],
        "evaluation_primary_coordinates": evaluation.get("primary_coordinates") == ["M", "Psi"],
        "artifact_input_root": Path(str(artifact_meta.get("input_root", input_root))).resolve() == input_root,
        "artifact_checkpoint": Path(str(artifact_meta.get("checkpoint", checkpoint))).resolve() == checkpoint,
        "evaluation_input_root": Path(str(evaluation.get("input_root", input_root))).resolve() == input_root,
        "evaluation_checkpoint": Path(str(evaluation.get("checkpoint", checkpoint))).resolve() == checkpoint,
        "fixed_k_verified": bool(input_manifest.get("stage1_fixed_k6_contract", {}).get("verified", False)),
        "fixed_k": int(input_manifest.get("stage1_fixed_k6_contract", {}).get("macrostate_k", -1)) == EXPECTED_K,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Stage-5 seed {spec.seed} contract failed: {failed}")
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    checkpoint_seed = int(checkpoint_data.get("config", {}).get("seed", -1))
    if checkpoint_seed != int(spec.seed):
        raise RuntimeError(f"Checkpoint seed {checkpoint_seed} does not match seed-root label {spec.seed}")
    source_hashes = {
        "evaluation_manifest": file_sha256(evaluation_path),
        "training_manifest": file_sha256(training_path),
        "input_manifest": file_sha256(input_manifest_path),
        "checkpoint": file_sha256(checkpoint),
        "artifacts": file_sha256(artifacts_path),
    }
    for split in ("A_val", "B_confirm"):
        prediction_path = find_table(evaluation_root / "predictions" / f"stage5_macro_sufficiency_predictions_macro_only_{split}")
        metric_path = evaluation_root / "tables" / f"stage5_macro_sufficiency_metrics_{split}.csv"
        if not metric_path.is_file():
            raise FileNotFoundError(metric_path)
        source_hashes[f"macro_only_predictions_{split}"] = file_sha256(prediction_path)
        source_hashes[f"metrics_{split}"] = file_sha256(metric_path)
    return SeedContext(
        spec=spec,
        supplied_stage5_root=supplied_root,
        stage5_training_root=training_root,
        stage5_root=evaluation_root,
        input_root=input_root,
        checkpoint=checkpoint,
        artifacts_path=artifacts_path,
        input_manifest=input_manifest,
        evaluation_manifest=evaluation,
        training_manifest=training,
        artifacts=artifacts,
        source_hashes=source_hashes,
    )


def validate_common_inputs(
    contexts: Sequence[SeedContext],
    evaluate_module: Any,
) -> Dict[int, Dict[str, Any]]:
    if not contexts:
        raise RuntimeError("No seed contexts were supplied.")
    reference = contexts[0]
    reference_contract = reference.input_manifest.get("stage1_fixed_k6_contract", {})
    reference_rows = reference.input_manifest.get("split_summaries", {})
    fingerprints: Dict[int, Dict[str, Any]] = {}
    reference_fingerprint: Optional[Dict[str, Any]] = None
    for context in contexts:
        if context.input_manifest.get("stage1_fixed_k6_contract", {}) != reference_contract:
            raise RuntimeError("Fixed K=6 input contract differs across seeds.")
        if context.input_manifest.get("split_summaries", {}) != reference_rows:
            raise RuntimeError("Prepared split summaries differ across seeds.")
        seed_fingerprint: Dict[str, Any] = {}
        for split in ("A_train", "A_val", "B_confirm"):
            arrays = evaluate_module.read_arrays(context.input_root, split)
            seed_fingerprint[split] = {
                "rows": int(arrays["n"]),
                "user_id_sha256": array_sha256(np.asarray(arrays["user_id"], dtype=np.int64)),
                "bundle_step_index_sha256": array_sha256(np.asarray(arrays["step"], dtype=np.int64)),
                "sequence_offsets_sha256": array_sha256(np.asarray(arrays["offsets"], dtype=np.int64)),
                "target_current_sha256": array_sha256(np.asarray(arrays["y"], dtype=np.float32)),
                "target_next_sha256": array_sha256(np.asarray(arrays["y_next"], dtype=np.float32)),
            }
        if reference_fingerprint is None:
            reference_fingerprint = seed_fingerprint
        elif seed_fingerprint != reference_fingerprint:
            raise RuntimeError(
                f"Prepared row identity or sequence order differs for seed {context.spec.seed}."
            )
        fingerprints[int(context.spec.seed)] = seed_fingerprint
    return fingerprints


def load_partition(evaluate_module: Any, context: SeedContext, stage1_root: Optional[Path]) -> FixedPartition:
    resolved = evaluate_module.resolve_stage1_root(context.input_manifest, stage1_root)
    partition = evaluate_module.load_fixed_k6_partition(resolved, context.input_manifest)
    return FixedPartition(
        scaler_mean=np.asarray(partition.scaler_mean, dtype=np.float64),
        scaler_scale=np.asarray(partition.scaler_scale, dtype=np.float64),
        standardized_centers=np.asarray(partition.standardized_centers, dtype=np.float64),
        centers=np.asarray(partition.centers, dtype=np.float64),
        k=int(partition.k),
        audit=dict(partition.audit),
    )


def partition_labels(partition: FixedPartition, state: np.ndarray) -> np.ndarray:
    values = np.asarray(state, dtype=np.float64)
    valid = np.isfinite(values).all(axis=1)
    output = np.full(len(values), -1, dtype=np.int64)
    if np.any(valid):
        standardized = (values[valid] - partition.scaler_mean[None, :]) / partition.scaler_scale[None, :]
        distance = np.sum(
            (standardized[:, None, :] - partition.standardized_centers[None, :, :]) ** 2,
            axis=2,
        )
        output[valid] = np.argmin(distance, axis=1).astype(np.int64)
    return output


def make_fixed_spline(n_knots: int, degree: int) -> Tuple[SplineTransformer, SplineTransformer]:
    knots = np.linspace(-1.0, 1.0, int(n_knots), dtype=np.float64)[:, None]
    first = SplineTransformer(
        degree=int(degree),
        knots=knots,
        extrapolation="constant",
        include_bias=True,
        order="C",
    )
    second = SplineTransformer(
        degree=int(degree),
        knots=knots,
        extrapolation="constant",
        include_bias=True,
        order="C",
    )
    dummy = np.asarray([[-1.0], [0.0], [1.0]], dtype=np.float64)
    first.fit(dummy)
    second.fit(dummy)
    return first, second


def tensor_basis(first: SplineTransformer, second: SplineTransformer, state: np.ndarray) -> np.ndarray:
    values = np.asarray(state, dtype=np.float64)
    b0 = first.transform(values[:, [0]])
    b1 = second.transform(values[:, [1]])
    output = np.einsum("ij,ik->ijk", b0, b1, optimize=True).reshape(len(values), -1)
    return output.astype(np.float32, copy=False)


def fit_mean_closure(
    state: np.ndarray,
    target_next: np.ndarray,
    n_knots: int,
    degree: int,
    alpha: float,
) -> MeanClosure:
    spline_m, spline_psi = make_fixed_spline(n_knots, degree)
    basis = tensor_basis(spline_m, spline_psi, state)
    scaler = StandardScaler(copy=True)
    features = scaler.fit_transform(basis).astype(np.float32, copy=False)
    regressor = Ridge(alpha=float(alpha), fit_intercept=True)
    regressor.fit(features, np.asarray(target_next, dtype=np.float32))
    return MeanClosure(
        n_knots=int(n_knots),
        degree=int(degree),
        alpha=float(alpha),
        spline_m=spline_m,
        spline_psi=spline_psi,
        scaler=scaler,
        regressor=regressor,
    )


def average_coordinate_mse(target: np.ndarray, prediction: np.ndarray) -> Tuple[float, float, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    mse = np.mean(error * error, axis=0)
    return float(np.mean(mse)), float(mse[0]), float(mse[1])


def gaussian_nll(
    target: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    rho: float,
) -> np.ndarray:
    residual = np.asarray(target, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    var = np.clip(np.asarray(variance, dtype=np.float64), VARIANCE_EPS, VARIANCE_MAX)
    sigma = np.sqrt(var)
    z0 = residual[:, 0] / sigma[:, 0]
    z1 = residual[:, 1] / sigma[:, 1]
    correlation = float(np.clip(rho, -RHO_LIMIT, RHO_LIMIT))
    one_minus = max(1.0 - correlation * correlation, 1e-6)
    quadratic = (z0 * z0 - 2.0 * correlation * z0 * z1 + z1 * z1) / one_minus
    return (
        math.log(2.0 * math.pi)
        + np.log(sigma[:, 0])
        + np.log(sigma[:, 1])
        + 0.5 * math.log(one_minus)
        + 0.5 * quadratic
    )


def fit_gaussian_closure(
    mean_model: MeanClosure,
    state: np.ndarray,
    target_next: np.ndarray,
    user_id: np.ndarray,
    variance_alpha: float,
    crossfit_folds: int,
    crossfit_seed: int,
) -> GaussianClosure:
    values = np.asarray(state, dtype=np.float32)
    target = np.asarray(target_next, dtype=np.float32)
    users = np.asarray(user_id, dtype=np.int64)
    if len(values) != len(target) or len(values) != len(users):
        raise ValueError("State, target and user arrays differ in length.")
    fold_id = user_crossfit_fold(users, int(crossfit_folds), int(crossfit_seed))
    oof_mean = np.full_like(target, np.nan, dtype=np.float32)
    fold_rows: List[Dict[str, Any]] = []
    for fold in range(int(crossfit_folds)):
        test = fold_id == fold
        train = ~test
        if int(np.sum(test)) == 0 or int(np.sum(train)) == 0:
            raise RuntimeError(f"Empty variance cross-fit fold: {fold}")
        fold_model = fit_mean_closure(
            values[train],
            target[train],
            int(mean_model.n_knots),
            int(mean_model.degree),
            float(mean_model.alpha),
        )
        oof_mean[test] = fold_model.predict(values[test])
        fold_rows.append({
            "fold": int(fold),
            "training_rows": int(np.sum(train)),
            "held_out_rows": int(np.sum(test)),
            "training_users": int(np.unique(users[train]).size),
            "held_out_users": int(np.unique(users[test]).size),
        })
    if not np.isfinite(oof_mean).all():
        raise RuntimeError("Variance cross-fitting left non-finite mean predictions.")
    residual = target.astype(np.float64) - oof_mean.astype(np.float64)
    log_squared = np.log(residual * residual + VARIANCE_EPS)
    features = mean_model.features(values)
    variance_regressor = Ridge(alpha=float(variance_alpha), fit_intercept=True)
    variance_regressor.fit(features, log_squared)
    raw_log_variance = np.asarray(variance_regressor.predict(features), dtype=np.float64)
    raw_variance = np.exp(raw_log_variance)
    target_second_moment = np.mean(residual * residual, axis=0)
    predicted_second_moment = np.mean(raw_variance, axis=0)
    log_variance_offset = np.log(
        np.maximum(target_second_moment, VARIANCE_EPS)
        / np.maximum(predicted_second_moment, VARIANCE_EPS)
    )
    variance = np.clip(
        np.exp(raw_log_variance + log_variance_offset[None, :]),
        VARIANCE_EPS,
        VARIANCE_MAX,
    )
    standardized = residual / np.sqrt(variance)
    valid = np.isfinite(standardized).all(axis=1)
    if valid.sum() < 10:
        rho = 0.0
    else:
        rho = float(np.corrcoef(standardized[valid, 0], standardized[valid, 1])[0, 1])
        if not np.isfinite(rho):
            rho = 0.0
    rho = float(np.clip(rho, -RHO_LIMIT, RHO_LIMIT))
    closure = GaussianClosure(
        mean=mean_model,
        variance_alpha=float(variance_alpha),
        variance_regressor=variance_regressor,
        log_variance_offset=np.asarray(log_variance_offset, dtype=np.float64),
        rho=rho,
        variance_eps=VARIANCE_EPS,
        variance_max=VARIANCE_MAX,
        crossfit_folds=int(crossfit_folds),
        crossfit_seed=int(crossfit_seed),
    )
    setattr(closure, "crossfit_audit", fold_rows)
    setattr(closure, "crossfit_residual_mse", np.mean(residual * residual, axis=0).tolist())
    setattr(closure, "calibrated_mean_variance", np.mean(variance, axis=0).tolist())
    return closure


def load_split_data(context: SeedContext, split: str, evaluate_module: Any) -> SplitData:
    base = context.stage5_root / "predictions" / f"stage5_macro_sufficiency_predictions_macro_only_{split}"
    columns = [
        "user_id",
        "bundle_step_index",
        "M",
        "Psi",
        "target_M_next",
        "target_Psi_next",
        "pred_M",
        "pred_Psi",
        "pred_next_M",
        "pred_next_Psi",
    ]
    table = read_table(base, columns=columns)
    arrays = evaluate_module.read_arrays(context.input_root, split)
    user_id = pd.to_numeric(table["user_id"], errors="raise").to_numpy(dtype=np.int64)
    step = pd.to_numeric(table["bundle_step_index"], errors="raise").to_numpy(dtype=np.int64)
    if len(table) != int(arrays["n"]):
        raise RuntimeError(f"{split} prediction row count differs from prepared arrays.")
    if not np.array_equal(user_id, np.asarray(arrays["user_id"], dtype=np.int64)):
        raise RuntimeError(f"{split} prediction user identifiers differ from prepared arrays.")
    if not np.array_equal(step, np.asarray(arrays["step"], dtype=np.int64)):
        raise RuntimeError(f"{split} prediction step identifiers differ from prepared arrays.")
    target_current = table[["M", "Psi"]].to_numpy(dtype=np.float32)
    target_next = table[["target_M_next", "target_Psi_next"]].to_numpy(dtype=np.float32)
    if not np.allclose(target_current, np.asarray(arrays["y"], dtype=np.float32), atol=1e-7, rtol=0.0):
        raise RuntimeError(f"{split} current targets differ from prepared arrays.")
    if not np.allclose(target_next, np.asarray(arrays["y_next"], dtype=np.float32), atol=1e-7, rtol=0.0):
        raise RuntimeError(f"{split} next targets differ from prepared arrays.")
    return SplitData(
        split=split,
        user_id=user_id,
        step=step,
        target_current=target_current,
        target_next=target_next,
        predicted_current=table[["pred_M", "pred_Psi"]].to_numpy(dtype=np.float32),
        quadratic_next=table[["pred_next_M", "pred_next_Psi"]].to_numpy(dtype=np.float32),
    )


def load_model(context: SeedContext, train_module: Any, device: torch.device) -> Any:
    checkpoint = torch.load(context.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    shapes = checkpoint["model_shapes"]
    if config.get("model_kind") != "predictive_state":
        raise RuntimeError("State-only closure audit requires the predictive_state model.")
    model = train_module.PredictiveStateEventSSL(
        n_num=int(shapes["n_num"]),
        n_cat=int(shapes["n_cat"]),
        hash_buckets=int(shapes["hash_buckets"]),
        hidden_dim=int(config["hidden_dim"]),
        input_dim=int(config["input_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        categorical_emb_dim=int(config["categorical_emb_dim"]),
        future_steps=tuple(int(value) for value in config["future_steps"]),
        delta_scale=float(config["delta_scale"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


@torch.inference_mode()
def collect_state_sample(
    model: Any,
    arrays: Mapping[str, Any],
    device: torch.device,
    chunk_len: int,
    max_rows: int,
    sample_seed: int,
) -> Dict[str, np.ndarray]:
    n_rows = int(arrays["n"])
    take_n = n_rows if max_rows <= 0 or max_rows >= n_rows else int(max_rows)
    rng = np.random.default_rng(int(sample_seed))
    selected = np.zeros(n_rows, dtype=bool)
    if take_n < n_rows:
        selected[rng.choice(n_rows, size=take_n, replace=False)] = True
    else:
        selected[:] = True
    state_rows: List[np.ndarray] = []
    target_rows: List[np.ndarray] = []
    user_rows: List[np.ndarray] = []
    index_rows: List[np.ndarray] = []
    autocast_enabled = device.type == "cuda"
    for sequence_index in range(len(arrays["offsets"]) - 1):
        start = int(arrays["offsets"][sequence_index])
        stop = int(arrays["offsets"][sequence_index + 1])
        if not selected[start:stop].any():
            continue
        hidden_state = None
        previous_hidden = None
        position = start
        while position < stop:
            end = min(position + int(chunk_len), stop)
            take = selected[position:end]
            x_num_np = np.array(arrays["x_num"][position:end], copy=True)
            x_cat_np = np.array(arrays["x_cat"][position:end], copy=True)
            x_num = torch.from_numpy(x_num_np).unsqueeze(0).to(device, non_blocking=True)
            x_cat = torch.from_numpy(x_cat_np).unsqueeze(0).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                embedded = model.embed_inputs(x_num, x_cat)
                h_after, hidden_state = model.rnn(embedded, hidden_state)
                first_before = torch.zeros_like(h_after[:, :1, :]) if previous_hidden is None else previous_hidden
                h_before = torch.cat([first_before, h_after[:, :-1, :]], dim=1)
                state = model.state_head(h_before)
            if take.any():
                state_rows.append(state.squeeze(0).float().cpu().numpy()[take].astype(np.float32))
                target_rows.append(np.asarray(arrays["y_next"][position:end], dtype=np.float32)[take])
                user_rows.append(np.asarray(arrays["user_id"][position:end], dtype=np.int64)[take])
                local_indices = np.arange(position, end, dtype=np.int64)[take]
                index_rows.append(local_indices)
            previous_hidden = h_after[:, -1:, :].detach()
            position = end
    if not state_rows:
        raise RuntimeError("No A_train state-only closure sample was collected.")
    return {
        "state": np.concatenate(state_rows, axis=0),
        "target_next": np.concatenate(target_rows, axis=0),
        "user_id": np.concatenate(user_rows, axis=0),
        "row_index": np.concatenate(index_rows, axis=0),
    }


def macro_features(state: np.ndarray) -> np.ndarray:
    values = np.asarray(state, dtype=np.float32)
    m = values[:, 0]
    psi = values[:, 1]
    return np.column_stack([m, psi, m * m, psi * psi, m * psi]).astype(np.float32)


def fit_quadratic_closure(
    state: np.ndarray,
    target_next: np.ndarray,
    alpha: float,
):
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=float(alpha), fit_intercept=True),
    )
    model.fit(macro_features(state), np.asarray(target_next, dtype=np.float32))
    return model


def predict_quadratic_closure(model: Any, state: np.ndarray) -> np.ndarray:
    return np.clip(
        model.predict(macro_features(np.asarray(state, dtype=np.float32))),
        -1.0,
        1.0,
    ).astype(np.float32)


def choose_quadratic_candidate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (float(row["validation_mean_mse"]), -float(row["alpha"])),
    )
    best_loss = float(ordered[0]["validation_mean_mse"])
    tied = [
        row for row in ordered
        if float(row["validation_mean_mse"]) <= best_loss + TIE_TOL
    ]
    return sorted(tied, key=lambda row: -float(row["alpha"]))[0]


def validate_quadratic_sample_contract(
    context: SeedContext,
    sample: Mapping[str, np.ndarray],
    split_data: SplitData,
    requested_max_rows: int,
    sample_seed: int,
    ridge_alpha: float,
) -> Dict[str, Any]:
    formal_probe = context.artifacts["macro_only"]["closure_probe_next"]
    formal_prediction = np.clip(
        formal_probe.predict(macro_features(split_data.predicted_current)), -1.0, 1.0
    ).astype(np.float32)
    archived_difference = float(np.max(np.abs(formal_prediction - split_data.quadratic_next)))
    reconstructed = fit_quadratic_closure(
        sample["state"], sample["target_next"], float(ridge_alpha)
    )
    reconstructed_prediction = np.clip(
        reconstructed.predict(macro_features(split_data.predicted_current)), -1.0, 1.0
    ).astype(np.float32)
    refit_difference = float(np.max(np.abs(reconstructed_prediction - formal_prediction)))
    return {
        "formal_probe_vs_archived_prediction_max_abs": archived_difference,
        "refitted_quadratic_vs_formal_probe_max_abs": refit_difference,
        "actual_sample_rows": int(len(sample["state"])),
        "requested_sample_max_rows": int(requested_max_rows),
        "sample_row_index_sha256": hashlib.sha256(np.asarray(sample["row_index"], dtype=np.int64).tobytes()).hexdigest(),
        "sample_seed": int(sample_seed),
        "ridge_alpha": float(ridge_alpha),
        "passed": bool(archived_difference <= RECONSTRUCTION_ATOL and refit_difference <= RECONSTRUCTION_ATOL),
    }


def metric_row_from_formal_table(context: SeedContext, split: str, representation: str) -> Dict[str, Any]:
    path = context.stage5_root / "tables" / f"stage5_macro_sufficiency_metrics_{split}.csv"
    frame = pd.read_csv(path)
    selected = frame[frame["representation"].astype(str) == representation]
    if len(selected) != 1:
        raise RuntimeError(f"Could not locate {representation} metrics for {split}")
    return selected.iloc[0].to_dict()


def compare_metrics(
    recomputed: Mapping[str, Any],
    archived: Mapping[str, Any],
    keys: Sequence[str],
) -> Dict[str, Any]:
    differences: Dict[str, float] = {}
    for key in keys:
        first = float(recomputed.get(key, np.nan))
        second = float(archived.get(key, np.nan))
        if np.isfinite(first) and np.isfinite(second):
            differences[key] = abs(first - second)
    maximum = max(differences.values()) if differences else float("nan")
    return {
        "differences": differences,
        "maximum_absolute_difference": maximum,
        "passed": bool(np.isfinite(maximum) and maximum <= METRIC_ATOL),
    }


def evaluate_mean_closure(
    evaluate_module: Any,
    context: SeedContext,
    split_data: SplitData,
    prediction: np.ndarray,
    partition: FixedPartition,
    convergence_reference: Tuple[float, float],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    arrays = evaluate_module.read_arrays(context.input_root, split_data.split)
    module_partition = evaluate_module.MacroPartition(
        scaler_mean=partition.scaler_mean,
        scaler_scale=partition.scaler_scale,
        centers=partition.centers,
        standardized_centers=partition.standardized_centers,
        k=partition.k,
        audit=partition.audit,
    )
    return evaluate_module.metrics_for_predictions(
        arrays,
        split_data.predicted_current,
        np.asarray(prediction, dtype=np.float32),
        module_partition,
        convergence_reference,
    )


def matched_origin_mean_metrics(
    evaluate_module: Any,
    split_data: SplitData,
    prediction: np.ndarray,
    partition: FixedPartition,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    current = np.asarray(split_data.predicted_current, dtype=np.float32)
    target_next = np.asarray(split_data.target_next, dtype=np.float32)
    predicted_next = np.asarray(prediction, dtype=np.float32)
    user_id = np.asarray(split_data.user_id, dtype=np.int64)
    target_field = evaluate_module.field_stats(
        current[:, 0],
        current[:, 1],
        target_next[:, 0] - current[:, 0],
        target_next[:, 1] - current[:, 1],
        user_id,
    )
    predicted_field = evaluate_module.field_stats(
        current[:, 0],
        current[:, 1],
        predicted_next[:, 0] - current[:, 0],
        predicted_next[:, 1] - current[:, 1],
        user_id,
    )
    metrics: Dict[str, Any] = {}
    metrics.update(
        evaluate_module.drift_metrics(
            "matched_origin", target_field, predicted_field
        )
    )
    current_labels = partition_labels(partition, current)
    target_labels = partition_labels(partition, target_next)
    predicted_labels = partition_labels(partition, predicted_next)
    target_matrix = normalise_transition(
        transition_counts(current_labels, target_labels, partition.k)
    )
    predicted_matrix = normalise_transition(
        transition_counts(current_labels, predicted_labels, partition.k)
    )
    for key, value in transition_metrics(target_matrix, predicted_matrix).items():
        metrics[f"matched_origin_{key}"] = value
    return metrics, {
        "P_matched_target": target_matrix,
        "P_matched_predicted": predicted_matrix,
        "field_matched_target_u": np.asarray(target_field.u, dtype=float),
        "field_matched_target_v": np.asarray(target_field.v, dtype=float),
        "field_matched_predicted_u": np.asarray(predicted_field.u, dtype=float),
        "field_matched_predicted_v": np.asarray(predicted_field.v, dtype=float),
    }


def matched_origin_empirical_transition(
    split_data: SplitData,
    partition: FixedPartition,
) -> np.ndarray:
    current = partition_labels(partition, split_data.predicted_current)
    target_next = partition_labels(partition, split_data.target_next)
    return normalise_transition(
        transition_counts(current, target_next, partition.k)
    )


def normalise_transition(counts: np.ndarray) -> np.ndarray:
    row_sum = counts.sum(axis=1, keepdims=True)
    output = np.zeros_like(counts, dtype=np.float64)
    valid = row_sum[:, 0] > 0
    output[valid] = counts[valid] / row_sum[valid]
    return output


def transition_counts(current: np.ndarray, next_state: np.ndarray, k: int) -> np.ndarray:
    output = np.zeros((k, k), dtype=np.float64)
    valid = (current >= 0) & (current < k) & (next_state >= 0) & (next_state < k)
    if np.any(valid):
        output += np.bincount(current[valid] * k + next_state[valid], minlength=k * k).reshape(k, k)
    return output


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan")
    aa = a[valid] - np.mean(a[valid])
    bb = b[valid] - np.mean(b[valid])
    denominator = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denominator) if denominator > EPS else float("nan")


def transition_metrics(empirical: np.ndarray, predicted: np.ndarray) -> Dict[str, Any]:
    tv = 0.5 * np.sum(np.abs(empirical - predicted), axis=1)
    empirical_self = np.diag(empirical)
    predicted_self = np.diag(predicted)
    k = empirical.shape[0]
    empirical_diag = np.argmax(empirical, axis=1) == np.arange(k)
    predicted_diag = np.argmax(predicted, axis=1) == np.arange(k)
    return {
        "transition_mean_row_tv": float(np.mean(tv)),
        "transition_max_row_tv": float(np.max(tv)),
        "self_transition_corr": pearson(empirical_self, predicted_self),
        "self_transition_rmse": float(np.sqrt(np.mean((empirical_self - predicted_self) ** 2))),
        "self_transition_mae": float(np.mean(np.abs(empirical_self - predicted_self))),
        "diagonal_dominance_predicted_states": int(np.sum(predicted_diag)),
        "diagonal_dominance_match_fraction": float(np.mean(empirical_diag == predicted_diag)),
        "top_transition_edge_overlap": float(np.mean(np.argmax(empirical, axis=1) == np.argmax(predicted, axis=1))),
    }


def quadrature_nodes(order: int) -> Tuple[np.ndarray, np.ndarray]:
    nodes, weights = hermgauss(int(order))
    standard_nodes = np.sqrt(2.0) * nodes
    standard_weights = weights / np.sqrt(math.pi)
    pairs = np.asarray([(first, second) for first in standard_nodes for second in standard_nodes], dtype=np.float64)
    pair_weights = np.asarray([first * second for first in standard_weights for second in standard_weights], dtype=np.float64)
    pair_weights = pair_weights / np.sum(pair_weights)
    return pairs, pair_weights


def gaussian_mesostate_probabilities(
    closure: GaussianClosure,
    state: np.ndarray,
    partition: FixedPartition,
    order: int,
    chunk_rows: int,
    return_audit: bool = False,
):
    values = np.asarray(state, dtype=np.float32)
    output = np.zeros((len(values), partition.k), dtype=np.float32)
    nodes, weights = quadrature_nodes(order)
    clipped_any_weight = 0.0
    clipped_m_weight = 0.0
    clipped_psi_weight = 0.0
    total_weighted_rows = 0.0
    for start in range(0, len(values), int(chunk_rows)):
        stop = min(start + int(chunk_rows), len(values))
        mean, variance, rho = closure.predict_parameters(values[start:stop])
        sigma = np.sqrt(variance)
        correlation = np.clip(rho, -RHO_LIMIT, RHO_LIMIT)
        remaining = np.sqrt(np.maximum(1.0 - correlation * correlation, 1e-8))
        probabilities = np.zeros((stop - start, partition.k), dtype=np.float64)
        for node, weight in zip(nodes, weights):
            sample = np.empty_like(mean)
            sample[:, 0] = mean[:, 0] + sigma[:, 0] * node[0]
            sample[:, 1] = mean[:, 1] + sigma[:, 1] * (
                correlation * node[0] + remaining * node[1]
            )
            outside_m = (sample[:, 0] < -1.0) | (sample[:, 0] > 1.0)
            outside_psi = (sample[:, 1] < -1.0) | (sample[:, 1] > 1.0)
            clipped_m_weight += float(weight) * float(np.sum(outside_m))
            clipped_psi_weight += float(weight) * float(np.sum(outside_psi))
            clipped_any_weight += float(weight) * float(np.sum(outside_m | outside_psi))
            total_weighted_rows += float(weight) * float(stop - start)
            sample = np.clip(sample, -1.0, 1.0)
            labels = partition_labels(partition, sample)
            for label in range(partition.k):
                probabilities[:, label] += float(weight) * (labels == label)
        probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), EPS)
        output[start:stop] = probabilities.astype(np.float32)
    if not return_audit:
        return output
    denominator = max(total_weighted_rows, EPS)
    audit = {
        "gauss_hermite_order": int(order),
        "rows": int(len(values)),
        "boundary_censoring_mass_any": float(clipped_any_weight / denominator),
        "boundary_censoring_mass_M": float(clipped_m_weight / denominator),
        "boundary_censoring_mass_Psi": float(clipped_psi_weight / denominator),
    }
    return output, audit


def transition_from_probabilities(
    current_state: np.ndarray,
    probabilities: np.ndarray,
    partition: FixedPartition,
) -> Tuple[np.ndarray, np.ndarray]:
    current_labels = partition_labels(partition, current_state)
    sums = np.zeros((partition.k, partition.k), dtype=np.float64)
    counts = np.zeros(partition.k, dtype=np.int64)
    for origin in range(partition.k):
        mask = current_labels == origin
        counts[origin] = int(np.sum(mask))
        if np.any(mask):
            sums[origin] = np.sum(np.asarray(probabilities[mask], dtype=np.float64), axis=0)
    matrix = np.zeros_like(sums)
    valid = counts > 0
    matrix[valid] = sums[valid] / counts[valid, None]
    return matrix, counts


def empirical_transition(split_data: SplitData, partition: FixedPartition) -> np.ndarray:
    current = partition_labels(partition, split_data.target_current)
    nxt = partition_labels(partition, split_data.target_next)
    return normalise_transition(transition_counts(current, nxt, partition.k))


def quadrature_order_audit(
    closure: GaussianClosure,
    split_data: SplitData,
    partition: FixedPartition,
    orders: Sequence[int],
    requested_order: int,
    maximum_order: int,
    order_step: int,
    sample_rows: int,
    seed: int,
    chunk_rows: int,
    matrix_tolerance: float,
    metric_tolerance: float,
    minimum_origin_rows: int,
) -> Tuple[int, pd.DataFrame, Dict[str, Any]]:
    n = len(split_data.predicted_current)
    take = n if sample_rows <= 0 or sample_rows >= n else int(sample_rows)
    rng = np.random.default_rng(int(seed))
    indices = (
        np.arange(n, dtype=np.int64)
        if take == n
        else np.sort(rng.choice(n, size=take, replace=False))
    )
    state = split_data.predicted_current[indices]
    subset = SplitData(
        split=split_data.split,
        user_id=split_data.user_id[indices],
        step=split_data.step[indices],
        target_current=split_data.target_current[indices],
        target_next=split_data.target_next[indices],
        predicted_current=state,
        quadratic_next=split_data.quadratic_next[indices],
    )
    empirical = empirical_transition(subset, partition)

    requested = int(requested_order)
    maximum = int(maximum_order)
    step = int(order_step)

    if requested < 3:
        raise ValueError("Requested Gauss--Hermite order must be at least 3.")
    if maximum < requested:
        raise ValueError(
            "--quadrature-max-order must be at least --gauss-hermite-order."
        )
    if step <= 0:
        raise ValueError("--quadrature-order-step must be positive.")

    declared_orders = sorted(
        set([requested] + [int(value) for value in orders])
    )
    if any(value < 3 for value in declared_orders):
        raise ValueError("Gauss--Hermite audit orders must be at least 3.")
    if max(declared_orders) > maximum:
        raise ValueError(
            "--quadrature-max-order must not be below a declared audit order."
        )

    matrices: Dict[int, np.ndarray] = {}
    metrics: Dict[int, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    automatically_added_orders: List[int] = []

    def evaluate_order(order: int) -> None:
        order = int(order)
        if order in matrices:
            return

        probability = gaussian_mesostate_probabilities(
            closure,
            state,
            partition,
            order,
            int(chunk_rows),
        )
        matrix, counts = transition_from_probabilities(
            state,
            probability,
            partition,
        )
        if int(np.min(counts)) < int(minimum_origin_rows):
            raise RuntimeError(
                "Gauss--Hermite audit subset does not cover every predicted "
                "origin state with at least "
                f"{int(minimum_origin_rows)} rows."
            )

        matrices[order] = matrix
        metrics[order] = transition_metrics(empirical, matrix)
        rows.append({
            "order": order,
            "sample_rows": int(take),
            "sample_indices_sha256": array_sha256(indices),
            "minimum_origin_intervals": int(np.min(counts)),
            **metrics[order],
        })

    for order in declared_orders:
        evaluate_order(order)

    comparisons: List[Dict[str, Any]] = []
    resolved: Optional[int] = None

    def compare_orders(lower: int, higher: int) -> bool:
        matrix_difference = float(
            np.max(np.abs(matrices[lower] - matrices[higher]))
        )
        row_tv_between_orders = 0.5 * np.sum(
            np.abs(matrices[lower] - matrices[higher]),
            axis=1,
        )
        mean_row_tv_between_orders = float(
            np.mean(row_tv_between_orders)
        )
        maximum_row_tv_between_orders = float(
            np.max(row_tv_between_orders)
        )
        passed = bool(
            matrix_difference <= float(matrix_tolerance)
            and maximum_row_tv_between_orders <= float(metric_tolerance)
        )
        comparisons.append({
            "lower_order": int(lower),
            "higher_order": int(higher),
            "maximum_transition_probability_difference": matrix_difference,
            "mean_row_tv_between_orders": mean_row_tv_between_orders,
            "maximum_row_tv_between_orders": maximum_row_tv_between_orders,
            "matrix_tolerance": float(matrix_tolerance),
            "row_tv_tolerance": float(metric_tolerance),
            "passed": passed,
        })
        return passed

    ordered = sorted(matrices)
    candidates = [value for value in ordered if value >= requested]

    for position, order in enumerate(candidates[:-1]):
        higher = candidates[position + 1]
        if compare_orders(order, higher):
            resolved = int(order)
            break

    while resolved is None and ordered[-1] < maximum:
        lower = int(ordered[-1])
        higher = min(lower + step, maximum)
        if higher <= lower:
            break

        evaluate_order(higher)
        automatically_added_orders.append(int(higher))
        ordered = sorted(matrices)

        if compare_orders(lower, higher):
            resolved = lower
            break

    if resolved is None:
        last = comparisons[-1] if comparisons else {}
        raise RuntimeError(
            "Gauss--Hermite transition probabilities did not converge by "
            f"the maximum audit order {maximum}; last comparison "
            f"{last.get('lower_order')}->{last.get('higher_order')}, "
            "maximum transition-probability difference="
            f"{last.get('maximum_transition_probability_difference')}, "
            "maximum row TV="
            f"{last.get('maximum_row_tv_between_orders')}."
        )

    rows.sort(key=lambda row: int(row["order"]))
    convergence_pair = next(
        (
            row
            for row in comparisons
            if bool(row["passed"])
            and int(row["lower_order"]) == int(resolved)
        ),
        None,
    )

    audit = {
        "requested_order": requested,
        "resolved_order": int(resolved),
        "declared_orders": declared_orders,
        "evaluated_orders": ordered,
        "automatically_added_orders": automatically_added_orders,
        "maximum_order": maximum,
        "order_step": step,
        "sample_rows": int(take),
        "sample_indices_sha256": array_sha256(indices),
        "minimum_origin_rows_required": int(minimum_origin_rows),
        "order_resolution_contract": (
            "predicted transition-matrix convergence only; empirical "
            "transition metrics are descriptive and do not select the order"
        ),
        "convergence_pair": convergence_pair,
        "comparisons": comparisons,
        "passed": True,
    }
    return int(resolved), pd.DataFrame(rows), audit


def validation_mean_loss(model: MeanClosure, split_data: SplitData, chunk_rows: int) -> Tuple[float, float, float, float]:
    squared = np.zeros(2, dtype=np.float64)
    total = 0
    clipped = 0
    for start in range(0, len(split_data.predicted_current), int(chunk_rows)):
        stop = min(start + int(chunk_rows), len(split_data.predicted_current))
        state = split_data.predicted_current[start:stop]
        raw = model.predict_raw(state)
        clipped += int(np.sum(np.any((raw < -1.0) | (raw > 1.0), axis=1)))
        prediction = np.clip(raw, -1.0, 1.0)
        error = prediction - split_data.target_next[start:stop]
        squared += np.sum(error * error, axis=0)
        total += len(error)
    mse = squared / max(total, 1)
    return float(np.mean(mse)), float(mse[0]), float(mse[1]), float(clipped / max(total, 1))


def validation_gaussian_nll(model: GaussianClosure, split_data: SplitData, chunk_rows: int) -> float:
    total_nll = 0.0
    total_rows = 0
    for start in range(0, len(split_data.predicted_current), int(chunk_rows)):
        stop = min(start + int(chunk_rows), len(split_data.predicted_current))
        mean, variance, rho = model.predict_parameters(split_data.predicted_current[start:stop])
        nll = gaussian_nll(split_data.target_next[start:stop], mean, variance, float(rho[0]))
        total_nll += float(np.sum(nll))
        total_rows += len(nll)
    return total_nll / max(total_rows, 1)


def model_clip_fraction(model: MeanClosure, state: np.ndarray, chunk_rows: int) -> float:
    clipped = 0
    total = 0
    for start in range(0, len(state), int(chunk_rows)):
        stop = min(start + int(chunk_rows), len(state))
        raw = model.predict_raw(state[start:stop])
        clipped += int(np.sum(np.any((raw < -1.0) | (raw > 1.0), axis=1)))
        total += len(raw)
    return float(clipped / max(total, 1))


def choose_mean_candidate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["validation_mean_mse"]),
            int(row["n_knots"]),
            -float(row["alpha"]),
        ),
    )
    best_loss = float(ordered[0]["validation_mean_mse"])
    tied = [row for row in ordered if float(row["validation_mean_mse"]) <= best_loss + TIE_TOL]
    return sorted(tied, key=lambda row: (int(row["n_knots"]), -float(row["alpha"])))[0]


def choose_variance_candidate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    ordered = sorted(rows, key=lambda row: (float(row["validation_gaussian_nll"]), -float(row["variance_alpha"])))
    best_loss = float(ordered[0]["validation_gaussian_nll"])
    tied = [row for row in ordered if float(row["validation_gaussian_nll"]) <= best_loss + TIE_TOL]
    return sorted(tied, key=lambda row: -float(row["variance_alpha"]))[0]


def within_user_groups(user_id: np.ndarray) -> List[Tuple[int, int]]:
    users = np.asarray(user_id, dtype=np.int64)
    if len(users) == 0:
        return []
    starts = np.concatenate([[0], np.flatnonzero(users[1:] != users[:-1]) + 1])
    stops = np.concatenate([starts[1:], [len(users)]])
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def within_user_permutation(groups: Sequence[Tuple[int, int]], n: int, rng: np.random.Generator) -> np.ndarray:
    indices = np.arange(n, dtype=np.int64)
    for start, stop in groups:
        if stop - start > 1:
            indices[start:stop] = start + rng.permutation(stop - start)
    return indices


def floor_metrics_for_mean(
    evaluate_module: Any,
    context: SeedContext,
    split_data: SplitData,
    predicted_next: np.ndarray,
    partition: FixedPartition,
    convergence_reference: Tuple[float, float],
    permutations: int,
    seed: int,
    label: str,
) -> pd.DataFrame:
    arrays = evaluate_module.read_arrays(context.input_root, split_data.split)
    module_partition = evaluate_module.MacroPartition(
        scaler_mean=partition.scaler_mean,
        scaler_scale=partition.scaler_scale,
        centers=partition.centers,
        standardized_centers=partition.standardized_centers,
        k=partition.k,
        audit=partition.audit,
    )
    groups = within_user_groups(split_data.user_id)
    rng = np.random.default_rng(int(seed))
    rows: List[Dict[str, Any]] = []
    for replicate in range(int(permutations)):
        current_index = within_user_permutation(groups, len(split_data.user_id), rng)
        next_index = within_user_permutation(groups, len(split_data.user_id), rng)
        permuted_current = split_data.predicted_current[current_index]
        permuted_next = np.asarray(predicted_next, dtype=np.float32)[next_index]
        metrics, _ = evaluate_module.metrics_for_predictions(
            arrays,
            permuted_current,
            permuted_next,
            module_partition,
            convergence_reference,
        )
        permuted_split = SplitData(
            split=split_data.split,
            user_id=split_data.user_id,
            step=split_data.step,
            target_current=split_data.target_current,
            target_next=split_data.target_next,
            predicted_current=permuted_current,
            quadratic_next=permuted_next,
        )
        matched_metrics, _ = matched_origin_mean_metrics(
            evaluate_module, permuted_split, permuted_next, partition
        )
        metrics.update(matched_metrics)
        row = {"model": label, "replicate": replicate}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def floor_metrics_for_gaussian(
    split_data: SplitData,
    probabilities: np.ndarray,
    partition: FixedPartition,
    empirical: np.ndarray,
    permutations: int,
    seed: int,
    label: str,
) -> pd.DataFrame:
    groups = within_user_groups(split_data.user_id)
    rng = np.random.default_rng(int(seed))
    rows: List[Dict[str, Any]] = []
    for replicate in range(int(permutations)):
        current_index = within_user_permutation(groups, len(split_data.user_id), rng)
        probability_index = within_user_permutation(groups, len(split_data.user_id), rng)
        permuted_current = split_data.predicted_current[current_index]
        predicted, _ = transition_from_probabilities(
            permuted_current,
            np.asarray(probabilities, dtype=np.float32)[probability_index],
            partition,
        )
        row = {"model": label, "replicate": replicate}
        row.update(transition_metrics(empirical, predicted))
        matched_target = normalise_transition(
            transition_counts(
                partition_labels(partition, permuted_current),
                partition_labels(partition, split_data.target_next),
                partition.k,
            )
        )
        for key, value in transition_metrics(matched_target, predicted).items():
            row[f"matched_origin_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_floor(observed_rows: pd.DataFrame, floor_rows: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for model in sorted(floor_rows["model"].unique()):
        subset = floor_rows[floor_rows["model"] == model]
        observed_subset = observed_rows[observed_rows["model"] == model]
        if len(observed_subset) != 1:
            continue
        observed = observed_subset.iloc[0]
        metric_names = [
            name for name in subset.columns
            if name not in {"model", "replicate"} and pd.api.types.is_numeric_dtype(subset[name])
        ]
        for metric in metric_names:
            values = pd.to_numeric(subset[metric], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            observed_value = float(observed.get(metric, np.nan))
            lower_is_better = any(token in metric for token in ("rmse", "mae", "row_tv", "js", "error"))
            floor_median = float(np.median(values))
            oriented = floor_median - observed_value if lower_is_better else observed_value - floor_median
            records.append({
                "model": model,
                "metric": metric,
                "observed": observed_value,
                "floor_median": floor_median,
                "floor_5p": float(np.quantile(values, 0.05)),
                "floor_95p": float(np.quantile(values, 0.95)),
                "oriented_improvement": oriented,
                "favourable_side_of_floor": bool(oriented > 0),
                "permutations": int(len(values)),
            })
    return pd.DataFrame(records)




def metric_difference(first: Mapping[str, Any], second: Mapping[str, Any], key: str) -> float:
    first_value = float(first.get(key, np.nan))
    second_value = float(second.get(key, np.nan))
    return float(first_value - second_value) if np.isfinite(first_value) and np.isfinite(second_value) else float("nan")

def statewise_transition_rows(
    seed: int,
    split: str,
    model: str,
    empirical: np.ndarray,
    predicted: np.ndarray,
    origin_counts: Optional[np.ndarray] = None,
    gauge: str = "formal",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tv = 0.5 * np.sum(np.abs(empirical - predicted), axis=1)
    for state in range(empirical.shape[0]):
        rows.append({
            "seed": int(seed),
            "split": split,
            "model": model,
            "gauge": gauge,
            "state": state,
            "empirical_self_transition": float(empirical[state, state]),
            "predicted_self_transition": float(predicted[state, state]),
            "self_transition_difference": float(predicted[state, state] - empirical[state, state]),
            "row_tv": float(tv[state]),
            "empirical_top_destination": int(np.argmax(empirical[state])),
            "predicted_top_destination": int(np.argmax(predicted[state])),
            "origin_intervals": int(origin_counts[state]) if origin_counts is not None else None,
        })
    return rows


def self_test() -> None:
    rng = np.random.default_rng(7)
    state = rng.uniform(-1.0, 1.0, size=(2000, 2)).astype(np.float32)
    target = np.column_stack([
        np.tanh(0.5 * state[:, 0] - 0.2 * state[:, 1] ** 2),
        np.tanh(-0.3 + 0.6 * state[:, 1] + 0.1 * state[:, 0] * state[:, 1]),
    ]).astype(np.float32)
    model = fit_mean_closure(state[:1500], target[:1500], 4, 3, 1.0)
    prediction = model.predict(state[1500:])
    loss, _, _ = average_coordinate_mse(target[1500:], prediction)
    if not np.isfinite(loss) or loss > 0.05:
        raise RuntimeError("Mean-closure self-test failed.")
    gaussian = fit_gaussian_closure(model, state[:1500], target[:1500], np.repeat(np.arange(150), 10), 1.0, 2, 13)
    mean, variance, rho = gaussian.predict_parameters(state[1500:])
    nll = gaussian_nll(target[1500:], mean, variance, float(rho[0]))
    if not np.isfinite(nll).all() or np.any(variance <= 0):
        raise RuntimeError("Gaussian-closure self-test failed.")
    print("state-only closure self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the post hoc state-only closure adequacy audit.")
    parser.add_argument(
        "--seed-root",
        action="append",
        default=[],
        help=(
            "SEED=/path/to/stage5_macro_sufficiency. The path may be the training root, "
            "a merged root, or its evaluation subdirectory; the layout is resolved automatically."
        ),
    )
    parser.add_argument("--primary-seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--output-root", type=Path, required=False)
    parser.add_argument("--train-script", type=Path, required=False)
    parser.add_argument("--evaluate-script", type=Path, required=False)
    parser.add_argument("--stage5-train-script", type=Path, required=False)
    parser.add_argument("--stage1-root", type=Path, default=None)
    parser.add_argument("--sample-max-rows", type=int, default=FORMAL_SAMPLE_ROWS)
    parser.add_argument("--sample-seed", type=int, default=FORMAL_SAMPLE_SEED)
    parser.add_argument("--chunk-len", type=int, default=512)
    parser.add_argument("--metric-chunk-rows", type=int, default=200000)
    parser.add_argument("--probability-chunk-rows", type=int, default=100000)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--knots", default="4,5,6")
    parser.add_argument("--quadratic-alphas", default="0.1,1,10")
    parser.add_argument("--mean-alphas", default="0.1,1,10")
    parser.add_argument("--variance-alphas", default="0.1,1,10")
    parser.add_argument("--variance-crossfit-folds", type=int, default=DEFAULT_VARIANCE_CROSSFIT_FOLDS)
    parser.add_argument("--variance-crossfit-seed", type=int, default=DEFAULT_VARIANCE_CROSSFIT_SEED)
    parser.add_argument(
        "--gauss-hermite-order",
        type=int,
        default=DEFAULT_GH_ORDER,
    )
    parser.add_argument(
        "--quadrature-audit-orders",
        default=",".join(str(value) for value in DEFAULT_GH_AUDIT_ORDERS),
    )
    parser.add_argument(
        "--quadrature-max-order",
        type=int,
        default=DEFAULT_GH_MAX_ORDER,
    )
    parser.add_argument(
        "--quadrature-order-step",
        type=int,
        default=DEFAULT_GH_ORDER_STEP,
    )
    parser.add_argument(
        "--quadrature-audit-rows",
        type=int,
        default=DEFAULT_GH_AUDIT_ROWS,
    )
    parser.add_argument(
        "--quadrature-matrix-tolerance",
        type=float,
        default=DEFAULT_GH_MATRIX_TOL,
    )
    parser.add_argument(
        "--quadrature-metric-tolerance",
        type=float,
        default=DEFAULT_GH_METRIC_TOL,
    )
    parser.add_argument(
        "--quadrature-min-origin-rows",
        type=int,
        default=DEFAULT_GH_MIN_ORIGIN_ROWS,
    )
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--permutation-seed", type=int, default=DEFAULT_PERMUTATION_SEED)
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--write-primary-predictions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.seed_root or args.output_root is None:
        raise ValueError("--seed-root and --output-root are required.")
    if args.train_script is None or args.evaluate_script is None or args.stage5_train_script is None:
        raise ValueError("--train-script, --evaluate-script and --stage5-train-script are required.")
    if args.torch_num_threads > 0:
        torch.set_num_threads(int(args.torch_num_threads))

    start_time = time.time()
    seed_specs = [parse_seed_root(value) for value in args.seed_root]
    if len({spec.seed for spec in seed_specs}) != len(seed_specs):
        raise ValueError("Duplicate seed-root labels were supplied.")
    contexts = [load_seed_context(spec) for spec in seed_specs]
    primary_contexts = [context for context in contexts if context.spec.seed == int(args.primary_seed)]
    if len(primary_contexts) != 1:
        raise RuntimeError("Exactly one primary seed context is required.")
    primary = primary_contexts[0]

    train_script = args.train_script.resolve()
    evaluate_script = args.evaluate_script.resolve()
    stage5_train_script = args.stage5_train_script.resolve()
    train_module = import_module(train_script, "state_only_event_ssl_train")
    evaluate_module = import_module(evaluate_script, "state_only_event_ssl_evaluate")
    import_module(stage5_train_script, "state_only_stage5_train")
    input_fingerprints = validate_common_inputs(contexts, evaluate_module)
    partition = load_partition(evaluate_module, primary, args.stage1_root)
    convergence_sample_max = int(partition.audit.get("fit_max_rows", 500000))
    convergence_m, convergence_psi, convergence_meta = evaluate_module.convergence_reference(
        primary.input_root, convergence_sample_max, int(args.primary_seed)
    )
    convergence_reference = (float(convergence_m), float(convergence_psi))

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_root}; pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    artifact_root = output_root / "artifacts"
    prediction_root = output_root / "predictions"
    for directory in (table_root, metadata_root, artifact_root, prediction_root):
        directory.mkdir(parents=True, exist_ok=True)

    knots = tuple(int(value) for value in parse_number_list(args.knots, int))
    quadratic_alphas = tuple(float(value) for value in parse_number_list(args.quadratic_alphas, float))
    mean_alphas = tuple(float(value) for value in parse_number_list(args.mean_alphas, float))
    variance_alphas = tuple(float(value) for value in parse_number_list(args.variance_alphas, float))
    if any(value <= 0 for value in quadratic_alphas + mean_alphas + variance_alphas):
        raise ValueError("All Ridge regularisation candidates must be positive.")
    if not any(np.isclose(value, FORMAL_RIDGE_ALPHA) for value in quadratic_alphas):
        raise ValueError("--quadratic-alphas must include the formal alpha 1.0 control.")
    if int(args.degree) != 3:
        raise ValueError("The public audit contract fixes cubic splines with degree 3.")
    if any(value < 4 for value in knots):
        raise ValueError("Each spline candidate requires at least four knots.")
    if int(args.variance_crossfit_folds) < 2:
        raise ValueError("--variance-crossfit-folds must be at least 2.")
    quadrature_orders = tuple(
        sorted(
            set(
                int(value)
                for value in parse_number_list(
                    args.quadrature_audit_orders,
                    int,
                )
            )
        )
    )
    if int(args.gauss_hermite_order) not in quadrature_orders:
        quadrature_orders = tuple(
            sorted(
                set(
                    quadrature_orders
                    + (int(args.gauss_hermite_order),)
                )
            )
        )
    if any(value < 3 for value in quadrature_orders):
        raise ValueError("Gauss--Hermite audit orders must be at least 3.")
    if int(args.quadrature_max_order) < max(quadrature_orders):
        raise ValueError(
            "--quadrature-max-order must be at least the largest declared "
            "Gauss--Hermite audit order."
        )
    if int(args.quadrature_order_step) <= 0:
        raise ValueError("--quadrature-order-step must be positive.")

    source_hashes = {
        "analysis_script": file_sha256(Path(__file__).resolve()),
        "train_script": file_sha256(train_script),
        "evaluate_script": file_sha256(evaluate_script),
        "stage5_train_script": file_sha256(stage5_train_script),
    }
    formal_scripts = {
        "train_script": str(train_script),
        "evaluate_script": str(evaluate_script),
        "stage5_train_script": str(stage5_train_script),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(primary, train_module, device)
    train_arrays = evaluate_module.read_arrays(primary.input_root, "A_train")
    sample = collect_state_sample(
        model,
        train_arrays,
        device,
        int(args.chunk_len),
        int(args.sample_max_rows),
        int(args.sample_seed),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    primary_val = load_split_data(primary, "A_val", evaluate_module)
    quadratic_contract = validate_quadratic_sample_contract(
        primary,
        sample,
        primary_val,
        int(args.sample_max_rows),
        int(args.sample_seed),
        FORMAL_RIDGE_ALPHA,
    )
    if not quadratic_contract["passed"]:
        raise RuntimeError(
            "The reconstructed A_train sample does not reproduce the formal quadratic closure. "
            "Check --sample-max-rows, --sample-seed and the formal Stage-5 inputs."
        )

    baseline_gate_rows: List[Dict[str, Any]] = []
    metric_keys = (
        "coordinate_rmse_M",
        "coordinate_rmse_Psi",
        "one_step_rmse_M",
        "one_step_rmse_Psi",
        "learned_plane_drift_vector_corr",
        "learned_plane_occupancy_weighted_local_drift_cosine",
        "learned_plane_transition_mean_row_tv",
        "learned_plane_self_transition_corr",
    )
    for split_data in (primary_val,):
        metrics, matrices = evaluate_mean_closure(
            evaluate_module,
            primary,
            split_data,
            split_data.quadratic_next,
            partition,
            convergence_reference,
        )
        archived = metric_row_from_formal_table(primary, split_data.split, "macro_only")
        audit = compare_metrics(metrics, archived, metric_keys)
        baseline_gate_rows.append({
            "seed": primary.spec.seed,
            "split": split_data.split,
            "maximum_absolute_metric_difference": audit["maximum_absolute_difference"],
            "passed": audit["passed"],
        })
        if not audit["passed"]:
            raise RuntimeError(f"Formal quadratic metric reconstruction failed for {split_data.split}")

    quadratic_candidate_rows: List[Dict[str, Any]] = []
    quadratic_models: Dict[float, Any] = {}
    for alpha in quadratic_alphas:
        candidate = fit_quadratic_closure(
            sample["state"], sample["target_next"], float(alpha)
        )
        prediction = predict_quadratic_closure(
            candidate, primary_val.predicted_current
        )
        mean_loss, mse_m, mse_psi = average_coordinate_mse(
            primary_val.target_next, prediction
        )
        quadratic_models[float(alpha)] = candidate
        quadratic_candidate_rows.append({
            "alpha": float(alpha),
            "basis_terms": 5,
            "validation_mean_mse": mean_loss,
            "validation_mse_M": mse_m,
            "validation_mse_Psi": mse_psi,
            "formal_alpha": bool(np.isclose(float(alpha), FORMAL_RIDGE_ALPHA)),
        })
    selected_quadratic_row = choose_quadratic_candidate(quadratic_candidate_rows)
    selected_quadratic = quadratic_models[float(selected_quadratic_row["alpha"])]
    for row in quadratic_candidate_rows:
        row["selected"] = bool(
            np.isclose(
                float(row["alpha"]),
                float(selected_quadratic_row["alpha"]),
                atol=0.0,
                rtol=0.0,
            )
        )

    mean_candidate_rows: List[Dict[str, Any]] = []
    mean_models: Dict[Tuple[int, float], MeanClosure] = {}
    for n_knots in knots:
        for alpha in mean_alphas:
            candidate = fit_mean_closure(
                sample["state"], sample["target_next"], n_knots, int(args.degree), alpha
            )
            mean_loss, mse_m, mse_psi, clip_fraction = validation_mean_loss(
                candidate, primary_val, int(args.metric_chunk_rows)
            )
            key = (int(n_knots), float(alpha))
            mean_models[key] = candidate
            mean_candidate_rows.append({
                "n_knots": int(n_knots),
                "degree": int(args.degree),
                "basis_terms": int((n_knots + args.degree - 1) ** 2),
                "alpha": float(alpha),
                "validation_mean_mse": mean_loss,
                "validation_mse_M": mse_m,
                "validation_mse_Psi": mse_psi,
                "validation_clip_fraction": clip_fraction,
            })
    selected_mean_row = choose_mean_candidate(mean_candidate_rows)
    selected_mean = mean_models[(int(selected_mean_row["n_knots"]), float(selected_mean_row["alpha"]))]
    for row in mean_candidate_rows:
        row["selected"] = bool(
            int(row["n_knots"]) == int(selected_mean_row["n_knots"])
            and np.isclose(float(row["alpha"]), float(selected_mean_row["alpha"]), atol=0.0, rtol=0.0)
        )

    variance_candidate_rows: List[Dict[str, Any]] = []
    variance_models: Dict[float, GaussianClosure] = {}
    for variance_alpha in variance_alphas:
        candidate = fit_gaussian_closure(
            selected_mean,
            sample["state"],
            sample["target_next"],
            sample["user_id"],
            variance_alpha,
            int(args.variance_crossfit_folds),
            int(args.variance_crossfit_seed),
        )
        nll = validation_gaussian_nll(candidate, primary_val, int(args.metric_chunk_rows))
        variance_models[float(variance_alpha)] = candidate
        variance_candidate_rows.append({
            "n_knots": int(selected_mean.n_knots),
            "degree": int(selected_mean.degree),
            "mean_alpha": float(selected_mean.alpha),
            "variance_alpha": float(variance_alpha),
            "rho": float(candidate.rho),
            "log_variance_offset_M": float(candidate.log_variance_offset[0]),
            "log_variance_offset_Psi": float(candidate.log_variance_offset[1]),
            "crossfit_folds": int(candidate.crossfit_folds),
            "crossfit_residual_mse_M": float(candidate.crossfit_residual_mse[0]),
            "crossfit_residual_mse_Psi": float(candidate.crossfit_residual_mse[1]),
            "calibrated_mean_variance_M": float(candidate.calibrated_mean_variance[0]),
            "calibrated_mean_variance_Psi": float(candidate.calibrated_mean_variance[1]),
            "validation_gaussian_nll": float(nll),
        })
    selected_variance_row = choose_variance_candidate(variance_candidate_rows)
    selected_gaussian = variance_models[float(selected_variance_row["variance_alpha"])]
    for row in variance_candidate_rows:
        row["selected"] = bool(
            np.isclose(
                float(row["variance_alpha"]),
                float(selected_variance_row["variance_alpha"]),
                atol=0.0,
                rtol=0.0,
            )
        )

    resolved_quadrature_order, quadrature_rows, quadrature_audit = quadrature_order_audit(
        selected_gaussian,
        primary_val,
        partition,
        quadrature_orders,
        int(args.gauss_hermite_order),
        int(args.quadrature_max_order),
        int(args.quadrature_order_step),
        int(args.quadrature_audit_rows),
        int(args.permutation_seed) + 41,
        int(args.probability_chunk_rows),
        float(args.quadrature_matrix_tolerance),
        float(args.quadrature_metric_tolerance),
        int(args.quadrature_min_origin_rows),
    )
    write_table(quadrature_rows, table_root / "gauss_hermite_order_audit")
    save_json(quadrature_audit, metadata_root / "gauss_hermite_order_audit.json")

    selected_contract = {
        "primary_seed": int(args.primary_seed),
        "matched_quadratic_control": {
            "alpha": float(selected_quadratic_row["alpha"]),
            "basis": ["M", "Psi", "M^2", "Psi^2", "M*Psi"],
            "selection_metric": "A_val average coordinate MSE after output clipping",
            "role": "regularisation-matched control separating spline basis effects from alpha selection",
        },
        "mean_closure": {
            "n_knots": int(selected_mean.n_knots),
            "degree": int(selected_mean.degree),
            "alpha": float(selected_mean.alpha),
            "selection_metric": "A_val average coordinate MSE after output clipping",
        },
        "distributional_closure": {
            "variance_alpha": float(selected_gaussian.variance_alpha),
            "rho": float(selected_gaussian.rho),
            "variance_eps": VARIANCE_EPS,
            "variance_max": VARIANCE_MAX,
            "selection_metric": "A_val continuous bivariate Gaussian negative log likelihood",
            "mesostate_labels_used_for_fitting_or_hyperparameter_selection": False,
            "empirical_transition_labels_used_for_quadrature_order_resolution": False,
            "variance_crossfit_folds": int(args.variance_crossfit_folds),
            "variance_crossfit_seed": int(args.variance_crossfit_seed),
            "log_variance_offset": np.asarray(selected_gaussian.log_variance_offset, dtype=float).tolist(),
            "gauss_hermite_order": int(resolved_quadrature_order),
            "boundary_handling": "Gaussian quadrature samples are censored to [-1,1] before frozen K=6 assignment; boundary-censoring mass is reported for every evaluated split.",
        },
        "B_confirm_used_for_selection": False,
    }
    save_json(selected_contract, metadata_root / "selected_state_only_closure_contract.json")
    primary_confirm = load_split_data(primary, "B_confirm", evaluate_module)

    split_metric_rows: List[Dict[str, Any]] = []
    transition_rows: List[Dict[str, Any]] = []
    seed_delta_rows: List[Dict[str, Any]] = []
    primary_predictions: Dict[str, Dict[str, np.ndarray]] = {}
    primary_gaussian_probabilities: Dict[str, np.ndarray] = {}

    for context in sorted(contexts, key=lambda item: item.spec.seed):
        model = load_model(context, train_module, device)
        arrays = evaluate_module.read_arrays(context.input_root, "A_train")
        seed_sample = collect_state_sample(
            model,
            arrays,
            device,
            int(args.chunk_len),
            int(args.sample_max_rows),
            int(args.sample_seed),
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if not np.array_equal(seed_sample["row_index"], sample["row_index"]):
            raise RuntimeError(f"A_train sample row indices differ for seed {context.spec.seed}")
        seed_tuned_quadratic = fit_quadratic_closure(
            seed_sample["state"],
            seed_sample["target_next"],
            float(selected_quadratic_row["alpha"]),
        )
        seed_mean = fit_mean_closure(
            seed_sample["state"],
            seed_sample["target_next"],
            int(selected_mean.n_knots),
            int(selected_mean.degree),
            float(selected_mean.alpha),
        )
        seed_gaussian = fit_gaussian_closure(
            seed_mean,
            seed_sample["state"],
            seed_sample["target_next"],
            seed_sample["user_id"],
            float(selected_gaussian.variance_alpha),
            int(args.variance_crossfit_folds),
            int(args.variance_crossfit_seed),
        )
        seed_reference_meta = dict(context.evaluation_manifest.get("convergence_reference", {}))
        if "reference_M" in seed_reference_meta and "reference_Psi" in seed_reference_meta:
            seed_convergence_reference = (
                float(seed_reference_meta["reference_M"]),
                float(seed_reference_meta["reference_Psi"]),
            )
        else:
            seed_m, seed_psi, _ = evaluate_module.convergence_reference(
                context.input_root, convergence_sample_max, int(context.spec.seed)
            )
            seed_convergence_reference = (float(seed_m), float(seed_psi))
        evaluation_splits = (
            ("A_val", "B_confirm")
            if context.spec.seed == int(args.primary_seed)
            else ("B_confirm",)
        )
        for split in evaluation_splits:
            split_data = load_split_data(context, split, evaluate_module)
            quadratic_metrics, quadratic_matrices = evaluate_mean_closure(
                evaluate_module,
                context,
                split_data,
                split_data.quadratic_next,
                partition,
                seed_convergence_reference,
            )
            archived = metric_row_from_formal_table(context, split, "macro_only")
            gate = compare_metrics(quadratic_metrics, archived, metric_keys)
            baseline_gate_rows.append({
                "seed": context.spec.seed,
                "split": split,
                "maximum_absolute_metric_difference": gate["maximum_absolute_difference"],
                "passed": gate["passed"],
            })
            if not gate["passed"]:
                raise RuntimeError(f"Formal quadratic metric reconstruction failed for seed {context.spec.seed}, {split}")
            quadratic_matched_metrics, quadratic_matched_matrices = matched_origin_mean_metrics(
                evaluate_module, split_data, split_data.quadratic_next, partition
            )
            quadratic_metrics.update(quadratic_matched_metrics)

            tuned_quadratic_next = predict_quadratic_closure(
                seed_tuned_quadratic, split_data.predicted_current
            )
            tuned_quadratic_metrics, tuned_quadratic_matrices = evaluate_mean_closure(
                evaluate_module,
                context,
                split_data,
                tuned_quadratic_next,
                partition,
                seed_convergence_reference,
            )
            tuned_matched_metrics, tuned_matched_matrices = matched_origin_mean_metrics(
                evaluate_module, split_data, tuned_quadratic_next, partition
            )
            tuned_quadratic_metrics.update(tuned_matched_metrics)

            spline_next = seed_mean.predict(split_data.predicted_current)
            spline_metrics, spline_matrices = evaluate_mean_closure(
                evaluate_module,
                context,
                split_data,
                spline_next,
                partition,
                seed_convergence_reference,
            )
            spline_matched_metrics, spline_matched_matrices = matched_origin_mean_metrics(
                evaluate_module, split_data, spline_next, partition
            )
            spline_metrics.update(spline_matched_metrics)

            probabilities, probability_audit = gaussian_mesostate_probabilities(
                seed_gaussian,
                split_data.predicted_current,
                partition,
                int(resolved_quadrature_order),
                int(args.probability_chunk_rows),
                return_audit=True,
            )
            kernel_matrix, origin_counts = transition_from_probabilities(
                split_data.predicted_current, probabilities, partition
            )
            empirical_matrix = empirical_transition(split_data, partition)
            matched_empirical_matrix = matched_origin_empirical_transition(
                split_data, partition
            )
            kernel_metrics = transition_metrics(empirical_matrix, kernel_matrix)
            matched_kernel_metrics = transition_metrics(
                matched_empirical_matrix, kernel_matrix
            )
            for key, value in matched_kernel_metrics.items():
                kernel_metrics[f"matched_origin_{key}"] = value

            for model_name, metrics in (
                ("quadratic_mean", quadratic_metrics),
                ("quadratic_tuned", tuned_quadratic_metrics),
                ("spline_mean", spline_metrics),
            ):
                row = {"seed": context.spec.seed, "split": split, "model": model_name}
                row.update(metrics)
                if model_name == "spline_mean":
                    row["output_clip_fraction"] = model_clip_fraction(
                        seed_mean, split_data.predicted_current, int(args.metric_chunk_rows)
                    )
                split_metric_rows.append(row)
            kernel_row = {
                "seed": context.spec.seed,
                "split": split,
                "model": "gaussian_distribution",
            }
            kernel_row.update(kernel_metrics)
            kernel_row["gaussian_nll"] = validation_gaussian_nll(
                seed_gaussian, split_data, int(args.metric_chunk_rows)
            )
            kernel_row["rho"] = float(seed_gaussian.rho)
            kernel_row["log_variance_offset_M"] = float(seed_gaussian.log_variance_offset[0])
            kernel_row["log_variance_offset_Psi"] = float(seed_gaussian.log_variance_offset[1])
            kernel_row.update(probability_audit)
            split_metric_rows.append(kernel_row)

            for model_name, formal_matrices, matched_matrices in (
                ("quadratic_mean", quadratic_matrices, quadratic_matched_matrices),
                ("quadratic_tuned", tuned_quadratic_matrices, tuned_matched_matrices),
                ("spline_mean", spline_matrices, spline_matched_matrices),
            ):
                transition_rows.extend(
                    statewise_transition_rows(
                        context.spec.seed,
                        split,
                        model_name,
                        formal_matrices["P_emp"],
                        formal_matrices["P_learned"],
                        gauge="formal",
                    )
                )
                transition_rows.extend(
                    statewise_transition_rows(
                        context.spec.seed,
                        split,
                        model_name,
                        matched_matrices["P_matched_target"],
                        matched_matrices["P_matched_predicted"],
                        gauge="matched_origin",
                    )
                )
            transition_rows.extend(
                statewise_transition_rows(
                    context.spec.seed,
                    split,
                    "gaussian_distribution",
                    empirical_matrix,
                    kernel_matrix,
                    origin_counts,
                    gauge="formal",
                )
            )
            transition_rows.extend(
                statewise_transition_rows(
                    context.spec.seed,
                    split,
                    "gaussian_distribution",
                    matched_empirical_matrix,
                    kernel_matrix,
                    origin_counts,
                    gauge="matched_origin",
                )
            )

            seed_delta_rows.append({
                "seed": context.spec.seed,
                "split": split,
                "tuned_minus_formal_quadratic_drift_vector_corr": metric_difference(
                    tuned_quadratic_metrics, quadratic_metrics, "learned_plane_drift_vector_corr"
                ),
                "spline_minus_tuned_quadratic_drift_vector_corr": metric_difference(
                    spline_metrics, tuned_quadratic_metrics, "learned_plane_drift_vector_corr"
                ),
                "spline_minus_tuned_quadratic_matched_drift_vector_corr": metric_difference(
                    spline_metrics, tuned_quadratic_metrics, "matched_origin_drift_vector_corr"
                ),
                "spline_minus_tuned_quadratic_local_cosine": metric_difference(
                    spline_metrics, tuned_quadratic_metrics,
                    "learned_plane_occupancy_weighted_local_drift_cosine",
                ),
                "spline_minus_tuned_quadratic_transition_tv": metric_difference(
                    spline_metrics, tuned_quadratic_metrics,
                    "learned_plane_transition_mean_row_tv",
                ),
                "spline_minus_tuned_quadratic_self_transition_corr": metric_difference(
                    spline_metrics, tuned_quadratic_metrics,
                    "learned_plane_self_transition_corr",
                ),
                "gaussian_minus_tuned_quadratic_transition_tv": float(
                    kernel_metrics.get("transition_mean_row_tv", np.nan)
                    - tuned_quadratic_metrics.get("learned_plane_transition_mean_row_tv", np.nan)
                ),
                "gaussian_minus_tuned_quadratic_self_transition_corr": float(
                    kernel_metrics.get("self_transition_corr", np.nan)
                    - tuned_quadratic_metrics.get("learned_plane_self_transition_corr", np.nan)
                ),
                "gaussian_minus_tuned_quadratic_matched_transition_tv": float(
                    kernel_metrics.get("matched_origin_transition_mean_row_tv", np.nan)
                    - tuned_quadratic_metrics.get("matched_origin_transition_mean_row_tv", np.nan)
                ),
                "gaussian_minus_tuned_quadratic_matched_self_transition_corr": float(
                    kernel_metrics.get("matched_origin_self_transition_corr", np.nan)
                    - tuned_quadratic_metrics.get("matched_origin_self_transition_corr", np.nan)
                ),
            })
            if context.spec.seed == int(args.primary_seed):
                primary_predictions[split] = {
                    "quadratic_formal": split_data.quadratic_next,
                    "quadratic_tuned": tuned_quadratic_next,
                    "spline": spline_next,
                    "current": split_data.predicted_current,
                }
                primary_gaussian_probabilities[split] = probabilities
                if args.write_primary_predictions:
                    frame = pd.DataFrame({
                        "split": split,
                        "user_id": split_data.user_id,
                        "bundle_step_index": split_data.step,
                        "pred_M": split_data.predicted_current[:, 0],
                        "pred_Psi": split_data.predicted_current[:, 1],
                        "target_M_next": split_data.target_next[:, 0],
                        "target_Psi_next": split_data.target_next[:, 1],
                        "quadratic_tuned_next_M": tuned_quadratic_next[:, 0],
                        "quadratic_tuned_next_Psi": tuned_quadratic_next[:, 1],
                        "spline_next_M": spline_next[:, 0],
                        "spline_next_Psi": spline_next[:, 1],
                    })
                    for label in range(partition.k):
                        frame[f"gaussian_next_state_probability_{label}"] = probabilities[:, label]
                    write_table(frame, prediction_root / f"state_only_closure_predictions_seed{context.spec.seed}_{split}")

    split_metrics_frame = pd.DataFrame(split_metric_rows)
    statewise_frame = pd.DataFrame(transition_rows)
    seed_delta_frame = pd.DataFrame(seed_delta_rows)
    baseline_gate_frame = pd.DataFrame(baseline_gate_rows).drop_duplicates(subset=["seed", "split"], keep="last")
    if not baseline_gate_frame["passed"].all():
        raise RuntimeError("At least one seed failed formal quadratic metric reconstruction.")
    confirm_data = primary_confirm
    quadratic_confirm_metrics = split_metrics_frame[
        (split_metrics_frame["seed"] == int(args.primary_seed))
        & (split_metrics_frame["split"] == "B_confirm")
        & (split_metrics_frame["model"] == "quadratic_mean")
    ].iloc[0].to_dict()
    tuned_quadratic_confirm_metrics = split_metrics_frame[
        (split_metrics_frame["seed"] == int(args.primary_seed))
        & (split_metrics_frame["split"] == "B_confirm")
        & (split_metrics_frame["model"] == "quadratic_tuned")
    ].iloc[0].to_dict()
    spline_confirm_metrics = split_metrics_frame[
        (split_metrics_frame["seed"] == int(args.primary_seed))
        & (split_metrics_frame["split"] == "B_confirm")
        & (split_metrics_frame["model"] == "spline_mean")
    ].iloc[0].to_dict()
    gaussian_confirm_metrics = split_metrics_frame[
        (split_metrics_frame["seed"] == int(args.primary_seed))
        & (split_metrics_frame["split"] == "B_confirm")
        & (split_metrics_frame["model"] == "gaussian_distribution")
    ].iloc[0].to_dict()
    empirical_confirm_matrix = empirical_transition(confirm_data, partition)

    observed_floor_rows = pd.DataFrame([
        {"model": "quadratic_mean", **quadratic_confirm_metrics},
        {"model": "quadratic_tuned", **tuned_quadratic_confirm_metrics},
        {"model": "spline_mean", **spline_confirm_metrics},
        {"model": "gaussian_distribution", **gaussian_confirm_metrics},
    ])
    quadratic_floor = floor_metrics_for_mean(
        evaluate_module,
        primary,
        confirm_data,
        primary_predictions["B_confirm"]["quadratic_formal"],
        partition,
        convergence_reference,
        int(args.permutations),
        int(args.permutation_seed),
        "quadratic_mean",
    )
    tuned_quadratic_floor = floor_metrics_for_mean(
        evaluate_module,
        primary,
        confirm_data,
        primary_predictions["B_confirm"]["quadratic_tuned"],
        partition,
        convergence_reference,
        int(args.permutations),
        int(args.permutation_seed),
        "quadratic_tuned",
    )
    spline_floor = floor_metrics_for_mean(
        evaluate_module,
        primary,
        confirm_data,
        primary_predictions["B_confirm"]["spline"],
        partition,
        convergence_reference,
        int(args.permutations),
        int(args.permutation_seed),
        "spline_mean",
    )
    gaussian_floor = floor_metrics_for_gaussian(
        confirm_data,
        primary_gaussian_probabilities["B_confirm"],
        partition,
        empirical_confirm_matrix,
        int(args.permutations),
        int(args.permutation_seed),
        "gaussian_distribution",
    )
    floor_replicates = pd.concat(
        [quadratic_floor, tuned_quadratic_floor, spline_floor, gaussian_floor],
        ignore_index=True,
        sort=False,
    )
    floor_summary = summarize_floor(observed_floor_rows, floor_replicates)

    with (artifact_root / "selected_state_only_closure_models.pkl").open("wb") as handle:
        pickle.dump(
            {
                "matched_quadratic_control": selected_quadratic,
                "mean_closure": selected_mean,
                "gaussian_closure": selected_gaussian,
                "selected_contract": selected_contract,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    write_table(
        pd.DataFrame(quadratic_candidate_rows),
        table_root / "quadratic_alpha_control_selection",
    )
    write_table(pd.DataFrame(mean_candidate_rows), table_root / "mean_closure_candidate_selection")
    write_table(pd.DataFrame(variance_candidate_rows), table_root / "distributional_closure_candidate_selection")
    write_table(
        pd.DataFrame(getattr(selected_gaussian, "crossfit_audit", [])),
        table_root / "distributional_variance_crossfit_audit",
    )
    write_table(split_metrics_frame, table_root / "state_only_closure_split_metrics")
    write_table(statewise_frame, table_root / "state_only_closure_statewise_transitions")
    write_table(seed_delta_frame, table_root / "state_only_closure_seed_differences")
    write_table(baseline_gate_frame, table_root / "quadratic_reconstruction_audit")
    write_table(floor_replicates, table_root / "state_only_closure_permutation_floor_replicates")
    write_table(floor_summary, table_root / "state_only_closure_permutation_floor_summary")

    input_inventory: List[Dict[str, Any]] = []
    for context in contexts:
        input_inventory.append({
            "seed": context.spec.seed,
            "supplied_stage5_root": str(context.supplied_stage5_root),
            "stage5_training_root": str(context.stage5_training_root),
            "stage5_evaluation_root": str(context.stage5_root),
            "stage5_root": str(context.stage5_root),
            "input_root": str(context.input_root),
            "checkpoint": str(context.checkpoint),
            "artifacts": str(context.artifacts_path),
            "source_hashes": context.source_hashes,
        })
    contract = {
        "analysis": "nonlinear state-only closure adequacy audit",
        "status": "post hoc supplementary analysis",
        "primary_seed": int(args.primary_seed),
        "seeds": [context.spec.seed for context in contexts],
        "input_features": ["pred_M", "pred_Psi"],
        "continuous_targets": ["target_M_next", "target_Psi_next"],
        "mesostate_labels_used_for_fitting_or_hyperparameter_selection": False,
        "empirical_transition_labels_used_for_quadrature_order_resolution": False,
        "matched_quadratic_control": {
            "alphas": list(quadratic_alphas),
            "selection_split": "A_val",
            "selection_metric": "average coordinate MSE after clipping",
            "formal_alpha": FORMAL_RIDGE_ALPHA,
            "role": "separates regularisation selection from nonlinear basis effects",
        },
        "mean_candidates": {
            "degree": int(args.degree),
            "knots": list(knots),
            "alphas": list(mean_alphas),
            "selection_split": "A_val",
            "selection_metric": "average coordinate MSE after clipping",
        },
        "distributional_candidates": {
            "variance_alphas": list(variance_alphas),
            "selection_split": "A_val",
            "selection_metric": "continuous bivariate Gaussian negative log likelihood",
            "working_distribution": "bivariate Gaussian with a clipped conditional mean, state-dependent marginal variances and one global residual correlation; quadrature samples are censored to [-1,1] before fixed-K assignment",
            "global_residual_correlation": True,
            "variance_crossfit_folds": int(args.variance_crossfit_folds),
            "variance_crossfit_seed": int(args.variance_crossfit_seed),
            "requested_gauss_hermite_order": int(args.gauss_hermite_order),
            "resolved_gauss_hermite_order": int(resolved_quadrature_order),
            "quadrature_audit_orders_declared": list(quadrature_orders),
            "quadrature_audit_orders": list(
                quadrature_audit["evaluated_orders"]
            ),
            "quadrature_automatically_added_orders": list(
                quadrature_audit["automatically_added_orders"]
            ),
            "quadrature_max_order": int(args.quadrature_max_order),
            "quadrature_order_step": int(args.quadrature_order_step),
            "quadrature_audit_rows": int(args.quadrature_audit_rows),
            "quadrature_matrix_tolerance": float(
                args.quadrature_matrix_tolerance
            ),
            "quadrature_metric_tolerance": float(
                args.quadrature_metric_tolerance
            ),
            "quadrature_min_origin_rows": int(
                args.quadrature_min_origin_rows
            ),
        },
        "permutation_floor": {
            "split": "B_confirm",
            "replicates": int(args.permutations),
            "base_seed": int(args.permutation_seed),
            "contract": "one shared deterministic replicate bank; independent within-user permutations of current and next closure outputs; no P value",
            "relationship_to_existing_table8_floor": "new model-specific descriptive floor; the independent robustness implementation used for the archived Table 8 floor was not supplied and is not claimed to be reconstructed",
        },
        "sample_contract": {
            "split": "A_train",
            "requested_max_rows": int(args.sample_max_rows),
            "actual_rows": int(quadratic_contract["actual_sample_rows"]),
            "seed": int(args.sample_seed),
            "ridge_alpha": FORMAL_RIDGE_ALPHA,
            "row_index_sha256": quadratic_contract["sample_row_index_sha256"],
        },
        "evaluation_gauges": {
            "formal": "empirical field and transitions use empirical current states; closure field and transitions use predicted current states, matching the formal Stage-5 contract",
            "matched_origin": "both empirical targets and closure outputs are conditioned on the frozen predicted current state; diagnostic only",
        },
        "fixed_contracts": {
            "event_ssl_weights_updated": False,
            "state_head_updated": False,
            "stage1_partition_updated": False,
            "field_grid_updated": False,
            "B_confirm_used_for_selection": False,
            "formal_stage5_outputs_modified": False,
        },
        "interpretation_boundary": (
            "The audit tests smooth population conditional closure from the frozen two-coordinate readout. "
            "It does not establish autonomous, Markov, causal or complete two-dimensional sufficiency."
        ),
    }
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": time.time() - start_time,
        "completed": True,
        "contract": contract,
        "selected_contract": selected_contract,
        "quadratic_sample_reconstruction": quadratic_contract,
        "quadratic_metric_reconstruction_passed": bool(baseline_gate_frame["passed"].all()),
        "partition": dict(partition.audit),
        "convergence_reference": convergence_meta,
        "formal_scripts": formal_scripts,
        "source_hashes": source_hashes,
        "input_inventory": input_inventory,
        "input_fingerprints": input_fingerprints,
        "outputs": {
            "quadratic_alpha_control_selection": str(find_table(table_root / "quadratic_alpha_control_selection")),
            "mean_candidate_selection": str(find_table(table_root / "mean_closure_candidate_selection")),
            "distributional_candidate_selection": str(find_table(table_root / "distributional_closure_candidate_selection")),
            "distributional_variance_crossfit_audit": str(find_table(table_root / "distributional_variance_crossfit_audit")),
            "split_metrics": str(find_table(table_root / "state_only_closure_split_metrics")),
            "statewise_transitions": str(find_table(table_root / "state_only_closure_statewise_transitions")),
            "seed_differences": str(find_table(table_root / "state_only_closure_seed_differences")),
            "quadratic_reconstruction_audit": str(find_table(table_root / "quadratic_reconstruction_audit")),
            "gauss_hermite_order_audit": str(find_table(table_root / "gauss_hermite_order_audit")),
            "gauss_hermite_order_audit_metadata": str((metadata_root / "gauss_hermite_order_audit.json").resolve()),
            "permutation_floor_replicates": str(find_table(table_root / "state_only_closure_permutation_floor_replicates")),
            "permutation_floor_summary": str(find_table(table_root / "state_only_closure_permutation_floor_summary")),
            "selected_models": str((artifact_root / "selected_state_only_closure_models.pkl").resolve()),
        },
    }
    manifest_path = metadata_root / "state_only_closure_audit_manifest.json"
    save_json(manifest, manifest_path)
    save_json(
        {"manifest_sha256": file_sha256(manifest_path)},
        metadata_root / "state_only_closure_audit_manifest.sha256.json",
    )
    print(f"[state-only closure audit] wrote {manifest_path}")


if __name__ == "__main__":
    main()
