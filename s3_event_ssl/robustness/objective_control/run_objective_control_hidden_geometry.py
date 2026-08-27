#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib.util
import json
import math
import os
import pickle
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-12
DEFAULT_SEEDS = (42, 2026, 666, 606, 37, 4669)
DEFAULT_SPLITS = ("A_train", "A_val", "B_confirm")
MODEL_ORDER = ("predictive_state", "pure_ssl", "task_only")
MODEL_LABELS = {
    "predictive_state": "Predictive-state Event-SSL",
    "pure_ssl": "Pure SSL",
    "task_only": "Task-only",
}
COORDINATES = ("M", "Psi")
HIDDEN_REFERENCE = "pre-interval recurrent hidden state h_before"


@dataclass(frozen=True)
class ResolvedSeedPaths:
    seed: int
    experiment_root: Path
    input_root: Path
    predictive_checkpoint: Path
    pure_checkpoint: Path
    task_checkpoint: Path
    predictive_training_manifest: Path
    pure_training_manifest: Path
    task_training_manifest: Path
    formal_geometry_train_root: Optional[Path]
    formal_geometry_eval_root: Optional[Path]


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
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


def save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        json_safe(payload),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_numeric_arrays(arrays: Sequence[np.ndarray], chunk_rows: int = 1_000_000) -> str:
    digest = hashlib.sha256()
    lengths = [len(np.asarray(array)) for array in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"Array lengths differ: {lengths}")
    n_rows = lengths[0]
    for array in arrays:
        view = np.asarray(array)
        digest.update(str(view.dtype).encode("utf-8"))
        digest.update(np.asarray(view.shape, dtype=np.int64).tobytes())
        for start in range(0, n_rows, int(chunk_rows)):
            stop = min(start + int(chunk_rows), n_rows)
            block = np.ascontiguousarray(view[start:stop])
            digest.update(block.tobytes())
    return digest.hexdigest()


def import_module(path: Path, module_name: str) -> Any:
    source = path.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location(module_name, str(source))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


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


def locate_table(base: Path) -> Path:
    if base.exists() and base.is_file():
        return base
    for suffix in (".parquet", ".csv.gz", ".csv"):
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Table not found: {base}.[parquet|csv.gz|csv]")


def read_table(base: Path) -> pd.DataFrame:
    path = locate_table(base)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3:
        return float("nan")
    aa = a[valid] - float(np.mean(a[valid]))
    bb = b[valid] - float(np.mean(b[valid]))
    denominator = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denominator) if denominator > EPS else float("nan")


def coordinate_metrics(prediction: np.ndarray, target: np.ndarray, prefix: str) -> Dict[str, float]:
    predicted = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(target, dtype=np.float64)
    if predicted.shape != observed.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError(f"Unexpected coordinate shapes: prediction={predicted.shape}, target={observed.shape}")
    output: Dict[str, float] = {}
    for index, coordinate in enumerate(COORDINATES):
        difference = predicted[:, index] - observed[:, index]
        output[f"{prefix}_corr_{coordinate}"] = pearson(predicted[:, index], observed[:, index])
        output[f"{prefix}_rmse_{coordinate}"] = float(np.sqrt(np.nanmean(difference * difference)))
        output[f"{prefix}_mae_{coordinate}"] = float(np.nanmean(np.abs(difference)))
        output[f"{prefix}_clip_fraction_{coordinate}"] = float(
            np.mean((predicted[:, index] <= -1.0 + 1e-7) | (predicted[:, index] >= 1.0 - 1e-7))
        )
    return output


def make_sample_indices(n_rows: int, max_rows: int, seed: int) -> np.ndarray:
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if max_rows <= 0 or max_rows >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    generator = np.random.default_rng(int(seed))
    return np.sort(generator.choice(n_rows, size=int(max_rows), replace=False).astype(np.int64))


def input_contract_subset(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    split_summaries = dict(manifest.get("split_summaries", {}))
    compact_splits = {
        split: {
            "rows": split_summaries.get(split, {}).get("rows"),
            "users": split_summaries.get(split, {}).get("users"),
            "numeric_shape": split_summaries.get(split, {}).get("numeric_shape"),
            "categorical_shape": split_summaries.get(split, {}).get("categorical_shape"),
            "target_shape": split_summaries.get(split, {}).get("target_shape"),
            "sequence_count": split_summaries.get(split, {}).get("sequence_count"),
            "sequence_break_policy": split_summaries.get(split, {}).get("sequence_break_policy"),
        }
        for split in DEFAULT_SPLITS
    }
    fixed_k = dict(manifest.get("stage1_fixed_k6_contract", {}))
    return {
        "primary_coordinates": manifest.get("primary_coordinates"),
        "targets": manifest.get("targets"),
        "numeric_input_source_columns": manifest.get("numeric_input_source_columns"),
        "numeric_feature_names_after_expansion": manifest.get("numeric_feature_names_after_expansion"),
        "categorical_input_source_columns": manifest.get("categorical_input_source_columns"),
        "categorical_hash_buckets": manifest.get("categorical_hash_buckets"),
        "forbidden_feature_tokens": manifest.get("forbidden_feature_tokens"),
        "normalization_fit_scope": manifest.get("normalization_fit_scope"),
        "sequence_boundary_policy": manifest.get("sequence_boundary_policy"),
        "fixed_k6_metadata_sha256": fixed_k.get("metadata_sha256"),
        "fixed_k6_centers_sha256": fixed_k.get("centers_sha256"),
        "splits": compact_splits,
    }


def validate_no_coordinate_input_leakage(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    required_forbidden = {
        "M_response",
        "activity_alignment_order_Psi",
        "delta_M",
        "delta_activity_alignment_order_Psi",
        "next_M_response",
        "next_activity_alignment_order_Psi",
        "macrostate",
        "candidate_region",
        "response_evidence_maturity",
        "maturity",
        "MR_",
        "PsiA",
    }
    declared_forbidden = {str(value) for value in manifest.get("forbidden_feature_tokens", [])}
    missing_required = sorted(required_forbidden.difference(declared_forbidden))
    if missing_required:
        raise RuntimeError(
            "Prepared-input manifest omits required leakage tokens: " + ", ".join(missing_required)
        )
    source_columns = [str(value) for value in manifest.get("numeric_input_source_columns", [])]
    source_columns.extend(str(value) for value in manifest.get("categorical_input_source_columns", []))
    violations = [
        column
        for column in source_columns
        if any(token in column for token in required_forbidden.union(declared_forbidden))
    ]
    if violations:
        raise RuntimeError(f"Prepared inputs contain forbidden macrostate-derived features: {violations}")
    if manifest.get("primary_coordinates") != ["M", "Psi"]:
        raise RuntimeError("Prepared inputs do not use the M-Psi target contract.")
    expected_targets = {
        "current": ["M_response_prebalanced_pre", "activity_alignment_order_Psi_pre"],
        "next": ["next_M_response_prebalanced", "next_activity_alignment_order_Psi"],
    }
    if manifest.get("targets") != expected_targets:
        raise RuntimeError(f"Prepared-input target contract changed: {manifest.get('targets')!r}")
    return {
        "verified": True,
        "input_columns_checked": len(source_columns),
        "required_forbidden_tokens": sorted(required_forbidden),
        "declared_forbidden_tokens": sorted(declared_forbidden),
        "violations": violations,
        "target_contract": expected_targets,
    }


def resolve_formal_geometry_roots(candidate: Path) -> Tuple[Optional[Path], Optional[Path]]:
    root = candidate.resolve()
    candidates: List[Tuple[Path, Path]] = [
        (root, root / "evaluation"),
        (root.parent, root),
        (root, root),
    ]
    for train_root, eval_root in candidates:
        train_manifest = train_root / "metadata" / "stage5_representation_geometry_training_manifest.json"
        eval_manifest = eval_root / "metadata" / "stage5_representation_geometry_evaluation_manifest.json"
        if train_manifest.exists() and eval_manifest.exists():
            return train_root, eval_root
    return None, None


def resolve_seed_paths(
    seed: int,
    experiment_root: Path,
    input_root: Optional[Path] = None,
    predictive_checkpoint: Optional[Path] = None,
    pure_checkpoint: Optional[Path] = None,
    task_checkpoint: Optional[Path] = None,
    formal_geometry_root: Optional[Path] = None,
) -> ResolvedSeedPaths:
    root = experiment_root.resolve()
    stage4 = root / "stage4_event_ssl"
    resolved_input = (input_root or (stage4 / "prepared_inputs")).resolve()
    predictive = (predictive_checkpoint or (stage4 / "models" / "predictive_state" / "best_model.pt")).resolve()
    pure = (pure_checkpoint or (stage4 / "models" / "pure_ssl" / "best_model.pt")).resolve()
    task = (task_checkpoint or (stage4 / "controls" / "task_only" / "model" / "best_model.pt")).resolve()
    predictive_manifest = predictive.parent / "training_manifest.json"
    pure_manifest = pure.parent / "training_manifest.json"
    task_manifest = task.parent / "training_manifest.json"
    required = [
        resolved_input / "metadata" / "stage4_input_manifest.json",
        predictive,
        pure,
        task,
        predictive_manifest,
        pure_manifest,
        task_manifest,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing seed inputs:\n" + "\n".join(missing))
    geometry_candidate = (formal_geometry_root or (root / "stage5_representation_geometry")).resolve()
    geometry_train, geometry_eval = resolve_formal_geometry_roots(geometry_candidate)
    return ResolvedSeedPaths(
        seed=int(seed),
        experiment_root=root,
        input_root=resolved_input,
        predictive_checkpoint=predictive,
        pure_checkpoint=pure,
        task_checkpoint=task,
        predictive_training_manifest=predictive_manifest.resolve(),
        pure_training_manifest=pure_manifest.resolve(),
        task_training_manifest=task_manifest.resolve(),
        formal_geometry_train_root=geometry_train,
        formal_geometry_eval_root=geometry_eval,
    )


def checkpoint_payload(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Checkpoint is not a mapping: {path}")
    return dict(payload)


def architecture_contract_from_checkpoint(model_label: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    config = dict(payload.get("config", {}))
    shapes = dict(payload.get("model_shapes", {}))
    return {
        "model_label": model_label,
        "n_num": int(shapes.get("n_num", -1)),
        "n_cat": int(shapes.get("n_cat", -1)),
        "hash_buckets": int(shapes.get("hash_buckets", -1)),
        "hidden_dim": int(config.get("hidden_dim", -1)),
        "input_dim": int(config.get("input_dim", -1)),
        "num_layers": int(config.get("num_layers", -1)),
        "dropout": float(config.get("dropout", np.nan)),
        "categorical_emb_dim": int(config.get("categorical_emb_dim", -1)),
        "future_steps": [int(value) for value in config.get("future_steps", [])],
        "delta_scale": float(config.get("delta_scale", np.nan)),
        "model_kind": str(config.get("model_kind", "")),
        "input_root_recorded": str(config.get("input_root", "")),
    }


def validate_objective_and_architecture_contracts(
    paths: ResolvedSeedPaths,
    train_script: Path,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    payloads = {
        "predictive_state": checkpoint_payload(paths.predictive_checkpoint),
        "pure_ssl": checkpoint_payload(paths.pure_checkpoint),
        "task_only": checkpoint_payload(paths.task_checkpoint),
    }
    architectures = {
        label: architecture_contract_from_checkpoint(label, payload)
        for label, payload in payloads.items()
    }
    architecture_fields = (
        "n_num",
        "n_cat",
        "hash_buckets",
        "hidden_dim",
        "input_dim",
        "num_layers",
        "dropout",
        "categorical_emb_dim",
    )
    reference = architectures["predictive_state"]
    mismatches: List[str] = []
    for label in ("pure_ssl", "task_only"):
        for field in architecture_fields:
            first = reference[field]
            second = architectures[label][field]
            if isinstance(first, float):
                equal = np.isclose(first, second, atol=1e-12, rtol=0.0)
            else:
                equal = first == second
            if not equal:
                mismatches.append(f"{label}:{field}:{second}!={first}")
    if mismatches:
        raise RuntimeError("Encoder architectures are not matched: " + "; ".join(mismatches))

    full_config = dict(payloads["predictive_state"].get("config", {}))
    pure_config = dict(payloads["pure_ssl"].get("config", {}))
    task_config = dict(payloads["task_only"].get("config", {}))
    if str(full_config.get("model_kind")) != "predictive_state":
        raise RuntimeError("Full checkpoint is not predictive_state.")
    if float(full_config.get("lambda_state", 0.0)) <= 0 or float(full_config.get("lambda_closure", 0.0)) <= 0:
        raise RuntimeError("Full checkpoint does not use positive state and closure weights.")
    if str(pure_config.get("model_kind")) != "pure_ssl":
        raise RuntimeError("Pure checkpoint is not pure_ssl.")
    if abs(float(pure_config.get("lambda_state", np.nan))) > 1e-12 or abs(float(pure_config.get("lambda_closure", np.nan))) > 1e-12:
        raise RuntimeError("Pure SSL checkpoint has non-zero state or closure weight.")
    if float(pure_config.get("lambda_future", 0.0)) <= 0:
        raise RuntimeError("Pure SSL checkpoint has no future-prediction objective.")
    if str(task_config.get("model_kind")) != "task_only_control":
        raise RuntimeError("Task-only checkpoint is not task_only_control.")

    training_manifests = {
        "predictive_state": load_json(paths.predictive_training_manifest),
        "pure_ssl": load_json(paths.pure_training_manifest),
        "task_only": load_json(paths.task_training_manifest),
    }
    expected_input_root = paths.input_root.resolve()
    expected_output_relatives = {
        "predictive_state": Path("stage4_event_ssl/models/predictive_state"),
        "pure_ssl": Path("stage4_event_ssl/models/pure_ssl"),
        "task_only": Path("stage4_event_ssl/controls/task_only/model"),
    }
    input_root_audit: Dict[str, Any] = {}
    for label, config in (
        ("predictive_state", full_config),
        ("pure_ssl", pure_config),
        ("task_only", task_config),
    ):
        try:
            recorded_seed = int(config.get("seed"))
        except Exception as exc:
            raise RuntimeError(f"{label} checkpoint does not record a valid seed.") from exc
        if recorded_seed != int(paths.seed):
            raise RuntimeError(
                f"{label} checkpoint seed is {recorded_seed}, expected {int(paths.seed)}."
            )
        recorded_input = str(config.get("input_root", "") or "").strip()
        recorded_output = str(config.get("output_root", "") or "").strip()
        if not recorded_input or not recorded_output:
            raise RuntimeError(f"{label} checkpoint does not record its input and output roots.")
        recorded_input_path = Path(recorded_input).expanduser().resolve()
        recorded_output_path = Path(recorded_output).expanduser().resolve()
        current_output_path = {
            "predictive_state": paths.predictive_checkpoint.parent,
            "pure_ssl": paths.pure_checkpoint.parent,
            "task_only": paths.task_checkpoint.parent,
        }[label].resolve()
        direct_match = recorded_input_path == expected_input_root
        coherent_relocation = False
        recorded_experiment_root: Optional[Path] = None
        if not direct_match:
            expected_suffix = Path("stage4_event_ssl/prepared_inputs")
            input_parts = recorded_input_path.parts
            suffix_parts = expected_suffix.parts
            if len(input_parts) >= len(suffix_parts) and tuple(input_parts[-len(suffix_parts):]) == suffix_parts:
                recorded_experiment_root = Path(*input_parts[:-len(suffix_parts)])
                coherent_relocation = (
                    recorded_output_path == (recorded_experiment_root / expected_output_relatives[label]).resolve()
                    and current_output_path == (paths.experiment_root / expected_output_relatives[label]).resolve()
                    and expected_input_root == (paths.experiment_root / expected_suffix).resolve()
                )
        if not direct_match and not coherent_relocation:
            raise RuntimeError(
                f"{label} checkpoint input/output roots do not match the supplied experiment root and "
                "do not form a coherent whole-experiment relocation."
            )
        input_root_audit[label] = {
            "recorded_input_root": str(recorded_input_path),
            "supplied_input_root": str(expected_input_root),
            "recorded_output_root": str(recorded_output_path),
            "supplied_output_root": str(current_output_path),
            "direct_path_match": bool(direct_match),
            "coherent_whole_experiment_relocation": bool(coherent_relocation),
            "recorded_experiment_root": str(recorded_experiment_root) if recorded_experiment_root else None,
        }
        manifest_config = dict(training_manifests[label].get("config", {}))
        for field in (
            "model_kind", "seed", "hidden_dim", "input_dim", "num_layers", "dropout",
            "categorical_emb_dim", "future_steps", "delta_scale",
        ):
            if json_safe(manifest_config.get(field)) != json_safe(config.get(field)):
                raise RuntimeError(
                    f"{label} checkpoint/training-manifest mismatch for {field}: "
                    f"{config.get(field)!r} versus {manifest_config.get(field)!r}."
                )
        manifest_input = str(manifest_config.get("input_root", "") or "").strip()
        manifest_output = str(manifest_config.get("output_root", "") or "").strip()
        if not manifest_input or not manifest_output:
            raise RuntimeError(f"{label} training manifest omits its input or output root.")
        if Path(manifest_input).expanduser().resolve() != recorded_input_path:
            raise RuntimeError(f"{label} checkpoint and training manifest disagree on the recorded input root.")
        if Path(manifest_output).expanduser().resolve() != recorded_output_path:
            raise RuntimeError(f"{label} checkpoint and training manifest disagree on the recorded output root.")

    task_boundary = dict(training_manifests["task_only"].get("control_boundary", {}))
    trained_losses = [str(value) for value in task_boundary.get("trained_losses", [])]
    excluded_losses = [str(value) for value in task_boundary.get("excluded_losses", [])]
    if trained_losses != ["task_bce_only"]:
        raise RuntimeError(f"Unexpected task-only trained losses: {trained_losses}")
    required_excluded = {"M/Psi_state_loss", "M/Psi_closure_loss", "future_ssl_loss"}
    if not required_excluded.issubset(set(excluded_losses)):
        raise RuntimeError("Task-only manifest does not exclude state, closure and future objectives.")
    if str(task_boundary.get("task_head_input")) != "pre-interval hidden state":
        raise RuntimeError("Task-only task head does not use the pre-interval hidden state.")

    current_train_sha = file_sha256(train_script.resolve())
    task_recorded_sha = str(payloads["task_only"].get("main_train_script_sha256", "") or "")
    task_manifest_sha = str(training_manifests["task_only"].get("main_train_script_sha256", "") or "")
    if task_recorded_sha and task_recorded_sha != current_train_sha:
        raise RuntimeError(
            "Task-only checkpoint was built with a different Event-SSL training implementation: "
            f"checkpoint={task_recorded_sha}, current={current_train_sha}."
        )
    if task_manifest_sha and task_manifest_sha != current_train_sha:
        raise RuntimeError(
            "Task-only training manifest was built with a different Event-SSL training implementation: "
            f"manifest={task_manifest_sha}, current={current_train_sha}."
        )
    if task_recorded_sha and task_manifest_sha and task_recorded_sha != task_manifest_sha:
        raise RuntimeError("Task-only checkpoint and training manifest disagree on the Event-SSL implementation SHA.")
    source_audit = {
        "current_train_script": str(train_script.resolve()),
        "current_train_script_sha256": current_train_sha,
        "task_checkpoint_recorded_train_script_sha256": task_recorded_sha or None,
        "task_manifest_recorded_train_script_sha256": task_manifest_sha or None,
        "task_checkpoint_train_script_matches_current": (
            task_recorded_sha == current_train_sha if task_recorded_sha else None
        ),
        "task_manifest_train_script_matches_current": (
            task_manifest_sha == current_train_sha if task_manifest_sha else None
        ),
        "all_checkpoint_seeds_match_requested_seed": True,
        "input_root_audit": input_root_audit,
        "all_checkpoint_inputs_direct_or_coherently_relocated": True,
        "checkpoint_and_training_manifest_configs_match": True,
        "architecture_fields_compared": list(architecture_fields),
        "architecture_matched": True,
        "objective_controls_are_not_single_factor_causal_ablations": True,
        "task_supervision_window_contract": task_config.get("supervise_truncated_windows"),
        "full_allow_truncated_supervision": full_config.get("allow_truncated_supervision"),
        "pure_allow_truncated_supervision": pure_config.get("allow_truncated_supervision"),
    }
    return source_audit, architectures, training_manifests


@torch.inference_mode()
def collect_hidden_sample(
    encoder: Any,
    arrays: Mapping[str, Any],
    sample_indices: np.ndarray,
    device: torch.device,
    chunk_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rows = int(arrays["n"])
    indices = np.asarray(sample_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("sample_indices must be a non-empty one-dimensional array")
    if np.any(indices < 0) or np.any(indices >= n_rows) or np.any(indices[1:] <= indices[:-1]):
        raise ValueError("sample_indices must be sorted unique in-range indices")
    selected = np.zeros(n_rows, dtype=bool)
    selected[indices] = True
    hidden_parts: List[np.ndarray] = []
    target_parts: List[np.ndarray] = []
    user_parts: List[np.ndarray] = []
    step_parts: List[np.ndarray] = []
    collected_indices: List[np.ndarray] = []
    autocast_enabled = device.type == "cuda"

    for sequence_index in range(len(arrays["offsets"]) - 1):
        start = int(arrays["offsets"][sequence_index])
        stop = int(arrays["offsets"][sequence_index + 1])
        if not selected[start:stop].any():
            continue
        hidden_state = None
        previous_hidden = None
        position = start
        last_required = int(np.flatnonzero(selected[start:stop])[-1]) + start + 1
        while position < min(stop, last_required):
            end = min(position + int(chunk_len), stop, last_required)
            take = selected[position:end]
            x_num = torch.from_numpy(np.array(arrays["x_num"][position:end], copy=True)).unsqueeze(0).to(device, non_blocking=True)
            x_cat = torch.from_numpy(np.array(arrays["x_cat"][position:end], copy=True)).unsqueeze(0).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                embedded = encoder.embed_inputs(x_num, x_cat)
                hidden_after, hidden_state = encoder.rnn(embedded, hidden_state)
                first_before = torch.zeros_like(hidden_after[:, :1, :]) if previous_hidden is None else previous_hidden
                hidden_before = torch.cat([first_before, hidden_after[:, :-1, :]], dim=1)
            if np.any(take):
                rows = np.arange(position, end, dtype=np.int64)[take]
                hidden_parts.append(hidden_before.squeeze(0).float().cpu().numpy().astype(np.float32)[take])
                target_parts.append(np.asarray(arrays["y"][position:end], dtype=np.float32)[take])
                user_parts.append(np.asarray(arrays["user_id"][position:end], dtype=np.int64)[take])
                step_parts.append(np.asarray(arrays["step"][position:end], dtype=np.int64)[take])
                collected_indices.append(rows)
            previous_hidden = hidden_after[:, -1:, :].detach()
            position = end

    if not hidden_parts:
        raise RuntimeError("No hidden states were collected.")
    collected = np.concatenate(collected_indices)
    hidden = np.concatenate(hidden_parts, axis=0)
    target = np.concatenate(target_parts, axis=0)
    users = np.concatenate(user_parts)
    steps = np.concatenate(step_parts)
    if not np.array_equal(collected, indices):
        raise RuntimeError("Collected hidden-state indices do not match the frozen sample bank.")
    if hidden.shape[0] != len(indices) or target.shape != (len(indices), 2):
        raise RuntimeError("Collected hidden-state sample has an unexpected shape.")
    return hidden, target, users, steps


def fit_hgb_regressors(scores: np.ndarray, targets: np.ndarray, seed: int) -> List[HistGradientBoostingRegressor]:
    models: List[HistGradientBoostingRegressor] = []
    for coordinate in range(2):
        regressor = HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.06,
            l2_regularization=1e-3,
            random_state=int(seed) + coordinate,
        )
        regressor.fit(scores, targets[:, coordinate])
        models.append(regressor)
    return models


def predict_hgb(models: Sequence[HistGradientBoostingRegressor], scores: np.ndarray) -> np.ndarray:
    return np.column_stack([model.predict(scores) for model in models]).astype(np.float32)


def participation_ratio(eigenvalues: np.ndarray) -> float:
    values = np.asarray(eigenvalues, dtype=np.float64)
    denominator = float(np.sum(values * values))
    return float(np.sum(values) ** 2 / denominator) if denominator > EPS else float("nan")


def effective_rank(eigenvalues: np.ndarray) -> float:
    values = np.asarray(eigenvalues, dtype=np.float64)
    total = float(np.sum(values))
    if total <= EPS:
        return float("nan")
    probabilities = values / total
    positive = probabilities > 0
    return float(np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive]))))


def pc_correlations(scores: np.ndarray, targets: np.ndarray, max_components: int) -> np.ndarray:
    count = min(int(max_components), scores.shape[1])
    output = np.full((count, 2), np.nan, dtype=np.float64)
    for component in range(count):
        for coordinate in range(2):
            output[component, coordinate] = pearson(scores[:, component], targets[:, coordinate])
    return output


def distinct_pc_assignment(correlations: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(correlations, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 2:
        raise ValueError("At least two PC components are required for a distinct assignment.")
    best: Optional[Tuple[float, int, int]] = None
    for index_m in range(values.shape[0]):
        for index_psi in range(values.shape[0]):
            if index_m == index_psi:
                continue
            score = abs(values[index_m, 0]) + abs(values[index_psi, 1])
            candidate = (float(score), int(index_m), int(index_psi))
            if best is None or candidate[0] > best[0] + 1e-15:
                best = candidate
            elif best is not None and abs(candidate[0] - best[0]) <= 1e-15:
                if (max(candidate[1], candidate[2]), candidate[1], candidate[2]) < (
                    max(best[1], best[2]), best[1], best[2]
                ):
                    best = candidate
    if best is None:
        raise RuntimeError("No distinct PC assignment was found.")
    _, index_m, index_psi = best
    return {
        "pc_index_M_zero_based": index_m,
        "pc_index_Psi_zero_based": index_psi,
        "pc_index_M": index_m + 1,
        "pc_index_Psi": index_psi + 1,
        "train_corr_M": float(values[index_m, 0]),
        "train_corr_Psi": float(values[index_psi, 1]),
        "train_abs_corr_M": float(abs(values[index_m, 0])),
        "train_abs_corr_Psi": float(abs(values[index_psi, 1])),
        "selection": "distinct pair maximizing the sum of absolute A_train coordinate correlations among the first components",
    }


def fit_geometry_artifacts(
    hidden: np.ndarray,
    targets: np.ndarray,
    pca_components: int,
    pc_alignment_components: int,
    ridge_alpha: float,
    seed: int,
    run_nonlinear: bool,
) -> Dict[str, Any]:
    component_count = int(max(2, min(int(pca_components), hidden.shape[1], hidden.shape[0] - 1)))
    hidden_scaler = StandardScaler().fit(hidden)
    standardized_hidden = hidden_scaler.transform(hidden)
    pca = PCA(n_components=component_count, random_state=int(seed)).fit(standardized_hidden)
    scores = pca.transform(standardized_hidden).astype(np.float32)
    eigenvalues = np.asarray(pca.explained_variance_, dtype=np.float64)
    macro_scaler = StandardScaler().fit(targets)
    cca = CCA(n_components=2, max_iter=1000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cca.fit(standardized_hidden, macro_scaler.transform(targets))
    hidden_canonical, macro_canonical = cca.transform(
        standardized_hidden,
        macro_scaler.transform(targets),
    )
    train_cca_raw = np.asarray(
        [pearson(hidden_canonical[:, index], macro_canonical[:, index]) for index in range(2)],
        dtype=np.float64,
    )
    cca_order = np.argsort(-np.abs(train_cca_raw)).astype(np.int64)
    cca_signs = np.asarray(
        [1.0 if train_cca_raw[index] >= 0 else -1.0 for index in cca_order],
        dtype=np.float64,
    )

    raw_ridge = make_pipeline(
        StandardScaler(),
        Ridge(alpha=float(ridge_alpha), fit_intercept=True),
    ).fit(hidden, targets)
    pca_ridge = make_pipeline(
        StandardScaler(),
        Ridge(alpha=float(ridge_alpha), fit_intercept=True),
    ).fit(scores, targets)
    hgb = fit_hgb_regressors(scores, targets, seed) if run_nonlinear else None
    train_pc = pc_correlations(scores, targets, int(pc_alignment_components))
    assignment = distinct_pc_assignment(train_pc)
    explained = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    summary = {
        "sample_rows": int(hidden.shape[0]),
        "hidden_dim": int(hidden.shape[1]),
        "pca_components": int(component_count),
        "pc_alignment_components": int(min(pc_alignment_components, component_count)),
        "pca_pc1_pc2_explained_variance": float(np.sum(explained[:2])),
        "pca_pc1_pc10_explained_variance": float(np.sum(explained[:10])),
        "pca_all_fitted_explained_variance": float(np.sum(explained)),
        "participation_ratio": participation_ratio(eigenvalues),
        "effective_rank": effective_rank(eigenvalues),
        "cca_train_raw": train_cca_raw.tolist(),
        "cca_order": cca_order.tolist(),
        "cca_signs": cca_signs.tolist(),
        "cca_train_sorted_abs": [float(abs(train_cca_raw[index])) for index in cca_order],
        "cca_n_iter": json_safe(getattr(cca, "n_iter_", None)),
        "cca_warnings": [str(item.message) for item in caught],
        "pc_assignment": assignment,
        "nonlinear_probe_fitted": bool(run_nonlinear),
        "nonlinear_probe_contract": {
            "features": "the same 64 A_train-fitted PCA scores as the matched linear probe",
            "max_iter": 180,
            "learning_rate": 0.06,
            "l2_regularization": 1e-3,
        },
    }
    return {
        "hidden_scaler": hidden_scaler,
        "pca": pca,
        "macro_scaler": macro_scaler,
        "cca": cca,
        "cca_order": cca_order,
        "cca_signs": cca_signs,
        "raw_ridge": raw_ridge,
        "pca_ridge": pca_ridge,
        "hgb": hgb,
        "train_pc_correlations": train_pc,
        "summary": summary,
    }


def evaluate_geometry_artifacts(
    artifacts: Mapping[str, Any],
    hidden: np.ndarray,
    targets: np.ndarray,
    split: str,
    model_label: str,
    seed: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    standardized_hidden = artifacts["hidden_scaler"].transform(hidden)
    scores = artifacts["pca"].transform(standardized_hidden).astype(np.float32)
    raw_prediction = np.clip(artifacts["raw_ridge"].predict(hidden), -1.0, 1.0)
    pca_prediction = np.clip(artifacts["pca_ridge"].predict(scores), -1.0, 1.0)
    output: Dict[str, Any] = {
        "seed": int(seed),
        "model": model_label,
        "model_display": MODEL_LABELS[model_label],
        "split": split,
        "sample_rows": int(len(hidden)),
        **coordinate_metrics(raw_prediction, targets, "raw_ridge"),
        **coordinate_metrics(pca_prediction, targets, "pca64_ridge"),
    }
    hgb_models = artifacts.get("hgb")
    if hgb_models is not None:
        nonlinear_prediction = np.clip(predict_hgb(hgb_models, scores), -1.0, 1.0)
        output.update(coordinate_metrics(nonlinear_prediction, targets, "pca64_hgb"))
        for coordinate in COORDINATES:
            output[f"matched_nonlinear_gain_corr_{coordinate}"] = (
                output[f"pca64_hgb_corr_{coordinate}"] - output[f"pca64_ridge_corr_{coordinate}"]
            )
            output[f"matched_nonlinear_gain_rmse_reduction_{coordinate}"] = (
                output[f"pca64_ridge_rmse_{coordinate}"] - output[f"pca64_hgb_rmse_{coordinate}"]
            )

    hidden_canonical, macro_canonical = artifacts["cca"].transform(
        standardized_hidden,
        artifacts["macro_scaler"].transform(targets),
    )
    order = np.asarray(artifacts["cca_order"], dtype=np.int64)
    signs = np.asarray(artifacts["cca_signs"], dtype=np.float64)
    for rank, original_index in enumerate(order):
        signed_correlation = pearson(
            hidden_canonical[:, original_index] * signs[rank],
            macro_canonical[:, original_index],
        )
        output[f"cca_corr_{rank + 1}"] = float(abs(signed_correlation))
        output[f"cca_oriented_corr_{rank + 1}"] = float(signed_correlation)
        output[f"cca_original_component_{rank + 1}"] = int(original_index + 1)

    first_count = int(artifacts["summary"]["pc_alignment_components"])
    correlations = pc_correlations(scores, targets, first_count)
    assignment = dict(artifacts["summary"]["pc_assignment"])
    index_m = int(assignment["pc_index_M_zero_based"])
    index_psi = int(assignment["pc_index_Psi_zero_based"])
    output["assigned_pc_M"] = index_m + 1
    output["assigned_pc_Psi"] = index_psi + 1
    output["assigned_pc_corr_M"] = float(correlations[index_m, 0])
    output["assigned_pc_corr_Psi"] = float(correlations[index_psi, 1])
    output["assigned_pc_abs_corr_M"] = float(abs(correlations[index_m, 0]))
    output["assigned_pc_abs_corr_Psi"] = float(abs(correlations[index_psi, 1]))
    output["descriptive_split_max_abs_pc_corr_M"] = float(np.nanmax(np.abs(correlations[:, 0])))
    output["descriptive_split_max_abs_pc_corr_Psi"] = float(np.nanmax(np.abs(correlations[:, 1])))
    output["descriptive_split_max_pc_M"] = int(np.nanargmax(np.abs(correlations[:, 0])) + 1)
    output["descriptive_split_max_pc_Psi"] = int(np.nanargmax(np.abs(correlations[:, 1])) + 1)
    for key in (
        "pca_pc1_pc2_explained_variance",
        "pca_pc1_pc10_explained_variance",
        "pca_all_fitted_explained_variance",
        "participation_ratio",
        "effective_rank",
    ):
        output[key] = float(artifacts["summary"][key])

    pc_rows: List[Dict[str, Any]] = []
    explained = np.asarray(artifacts["pca"].explained_variance_ratio_, dtype=np.float64)
    for component in range(first_count):
        pc_rows.append({
            "seed": int(seed),
            "model": model_label,
            "model_display": MODEL_LABELS[model_label],
            "split": split,
            "component": f"PC{component + 1}",
            "component_index": component + 1,
            "corr_M": float(correlations[component, 0]),
            "corr_Psi": float(correlations[component, 1]),
            "abs_corr_M": float(abs(correlations[component, 0])),
            "abs_corr_Psi": float(abs(correlations[component, 1])),
            "explained_variance_ratio_train": float(explained[component]),
            "selected_for_M_on_A_train": bool(component == index_m),
            "selected_for_Psi_on_A_train": bool(component == index_psi),
        })
    return output, pd.DataFrame(pc_rows)


def load_single_encoder_model(
    model_label: str,
    paths: ResolvedSeedPaths,
    device: torch.device,
    train_module: Any,
    evaluate_module: Any,
    task_module: Any,
) -> Tuple[Any, Any, Dict[str, Any]]:
    if model_label == "predictive_state":
        model, config = evaluate_module.load_model(paths.predictive_checkpoint, device, train_module)
        return model, model, dict(config)
    if model_label == "pure_ssl":
        model, config = evaluate_module.load_model(paths.pure_checkpoint, device, train_module)
        return model, model, dict(config)
    if model_label == "task_only":
        model, config = task_module.load_task_model(paths.task_checkpoint, train_module, device)
        return model, model.base, dict(config)
    raise ValueError(f"Unknown model label: {model_label}")


def compare_value(
    rows: List[Dict[str, Any]],
    category: str,
    metric: str,
    observed: Any,
    archived: Any,
    tolerance: float,
    hard: bool,
) -> None:
    try:
        first = float(observed)
        second = float(archived)
    except Exception:
        first = float("nan")
        second = float("nan")
    difference = abs(first - second) if np.isfinite(first) and np.isfinite(second) else float("nan")
    passed = bool(np.isfinite(difference) and difference <= float(tolerance))
    rows.append({
        "category": category,
        "metric": metric,
        "observed": first,
        "archived": second,
        "absolute_difference": difference,
        "tolerance": float(tolerance),
        "hard_gate": bool(hard),
        "passed": passed,
    })


def formal_seed42_reconstruction_audit(
    paths: ResolvedSeedPaths,
    metrics: pd.DataFrame,
    pc_table: pd.DataFrame,
    training_summaries: Mapping[str, Any],
    tolerance: float,
    require_reference: bool,
) -> pd.DataFrame:
    if paths.formal_geometry_train_root is None or paths.formal_geometry_eval_root is None:
        if require_reference:
            raise FileNotFoundError("Formal seed-42 representation-geometry roots were not found.")
        return pd.DataFrame([{
            "category": "availability",
            "metric": "formal_geometry_reference_available",
            "observed": 0,
            "archived": 1,
            "absolute_difference": 1,
            "tolerance": 0,
            "hard_gate": False,
            "passed": False,
        }])
    train_root = paths.formal_geometry_train_root
    eval_root = paths.formal_geometry_eval_root
    train_manifest = load_json(train_root / "metadata" / "stage5_representation_geometry_training_manifest.json")
    eval_metrics = read_table(eval_root / "tables" / "stage5_representation_geometry_metrics_all_splits")
    train_pc = read_table(train_root / "tables" / "stage5_pc_macro_correlations_train")
    eval_pc = read_table(eval_root / "tables" / "stage5_representation_geometry_pc_macro_correlations")
    audit_rows: List[Dict[str, Any]] = []
    summary = dict(training_summaries["predictive_state"])
    archived_cca = list(train_manifest.get("cca_correlations") or [])
    observed_cca = list(summary.get("cca_train_sorted_abs") or [])
    for index in range(min(2, len(archived_cca), len(observed_cca))):
        compare_value(audit_rows, "A_train", f"cca_corr_{index + 1}", observed_cca[index], abs(float(archived_cca[index])), tolerance, True)
    compare_value(audit_rows, "A_train", "participation_ratio", summary.get("participation_ratio"), train_manifest.get("participation_ratio"), tolerance, True)
    compare_value(audit_rows, "A_train", "effective_rank", summary.get("effective_rank"), train_manifest.get("effective_rank"), tolerance, True)

    observed_train_pc = pc_table[
        (pc_table["model"] == "predictive_state") & (pc_table["split"] == "A_train")
    ].copy()
    for component in range(1, 17):
        current = observed_train_pc[observed_train_pc["component_index"] == component]
        archived = train_pc[train_pc["component"].astype(str) == f"PC{component}"]
        if current.empty or archived.empty:
            continue
        for coordinate in COORDINATES:
            compare_value(
                audit_rows,
                "A_train_pc",
                f"PC{component}_corr_{coordinate}",
                current.iloc[0][f"corr_{coordinate}"],
                archived.iloc[0][f"corr_{coordinate}"],
                tolerance,
                True,
            )

    for split in ("A_val", "B_confirm"):
        observed_row = metrics[
            (metrics["model"] == "predictive_state") & (metrics["split"] == split)
        ]
        archived_linear = eval_metrics[
            (eval_metrics["split"].astype(str) == split)
            & (eval_metrics["representation"].astype(str) == "linear_hidden")
        ]
        archived_reference = eval_metrics[
            (eval_metrics["split"].astype(str) == split)
            & (eval_metrics["representation"].astype(str) == "model_readout")
        ]
        if observed_row.empty or archived_linear.empty:
            continue
        observed_series = observed_row.iloc[0]
        archived_series = archived_linear.iloc[0]
        for coordinate in COORDINATES:
            compare_value(
                audit_rows,
                split,
                f"raw_ridge_corr_{coordinate}",
                observed_series[f"raw_ridge_corr_{coordinate}"],
                archived_series[f"coordinate_corr_{coordinate}"],
                tolerance,
                True,
            )
            compare_value(
                audit_rows,
                split,
                f"raw_ridge_rmse_{coordinate}",
                observed_series[f"raw_ridge_rmse_{coordinate}"],
                archived_series[f"coordinate_rmse_{coordinate}"],
                tolerance,
                True,
            )
        if not archived_reference.empty:
            for index in (1, 2):
                compare_value(
                    audit_rows,
                    split,
                    f"cca_corr_{index}",
                    observed_series[f"cca_corr_{index}"],
                    archived_reference.iloc[0][f"cca_corr_{index}"],
                    tolerance,
                    True,
                )
        observed_pc = pc_table[
            (pc_table["model"] == "predictive_state") & (pc_table["split"] == split)
        ]
        archived_pc_split = eval_pc[eval_pc["split"].astype(str) == split]
        for component in range(1, 17):
            current = observed_pc[observed_pc["component_index"] == component]
            archived = archived_pc_split[archived_pc_split["component"].astype(str) == f"PC{component}"]
            if current.empty or archived.empty:
                continue
            for coordinate in COORDINATES:
                compare_value(
                    audit_rows,
                    f"{split}_pc",
                    f"PC{component}_corr_{coordinate}",
                    current.iloc[0][f"corr_{coordinate}"],
                    archived.iloc[0][f"corr_{coordinate}"],
                    tolerance,
                    True,
                )
    audit = pd.DataFrame(audit_rows)
    hard_failures = audit[(audit["hard_gate"] == True) & (audit["passed"] != True)]
    if not hard_failures.empty:
        raise RuntimeError(
            "Formal seed-42 geometry reconstruction failed: "
            + ", ".join(hard_failures["metric"].astype(str).tolist()[:10])
        )
    return audit


def run_seed(args: argparse.Namespace) -> None:
    seed = int(args.seed)
    result_seed_root = args.result_root.resolve() / "seeds" / f"seed{seed}"
    if result_seed_root.exists():
        if not args.overwrite:
            manifest_path = result_seed_root / "metadata" / "seed_geometry_manifest.json"
            if manifest_path.exists():
                print(f"[objective geometry] seed {seed} already complete: {manifest_path}")
                return
            raise FileExistsError(f"Incomplete output exists; rerun with --overwrite: {result_seed_root}")
        shutil.rmtree(result_seed_root)
    table_root = result_seed_root / "tables"
    metadata_root = result_seed_root / "metadata"
    artifact_root = result_seed_root / "artifacts"
    for directory in (table_root, metadata_root, artifact_root):
        directory.mkdir(parents=True, exist_ok=True)

    paths = resolve_seed_paths(
        seed=seed,
        experiment_root=args.experiment_root,
        input_root=args.input_root,
        predictive_checkpoint=args.predictive_checkpoint,
        pure_checkpoint=args.pure_checkpoint,
        task_checkpoint=args.task_checkpoint,
        formal_geometry_root=args.formal_geometry_root,
    )
    train_script = args.train_script.resolve()
    evaluate_script = args.evaluate_script.resolve()
    task_script = args.task_script.resolve()
    for source in (train_script, evaluate_script, task_script):
        if not source.exists():
            raise FileNotFoundError(source)

    input_manifest_path = paths.input_root / "metadata" / "stage4_input_manifest.json"
    normalizer_path = paths.input_root / "metadata" / "normalizer.json"
    if not normalizer_path.exists():
        raise FileNotFoundError(normalizer_path)
    input_manifest = load_json(input_manifest_path)
    leakage_audit = validate_no_coordinate_input_leakage(input_manifest)
    schema_contract = input_contract_subset(input_manifest)
    schema_contract["normalizer_sha256"] = file_sha256(normalizer_path)
    schema_signature = stable_json_hash(schema_contract)

    train_module = import_module(train_script, f"objective_geometry_train_{seed}")
    evaluate_module = import_module(evaluate_script, f"objective_geometry_evaluate_{seed}")
    task_module = import_module(task_script, f"objective_geometry_task_{seed}")
    required_evaluate = {"read_arrays", "load_model"}
    required_task = {"load_task_model"}
    if not required_evaluate.issubset(set(dir(evaluate_module))):
        raise RuntimeError("Formal evaluator is missing read_arrays or load_model.")
    if not required_task.issubset(set(dir(task_module))):
        raise RuntimeError("Task-only module is missing load_task_model.")

    source_audit, architectures, training_manifests = validate_objective_and_architecture_contracts(
        paths,
        train_script,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrays_by_split: Dict[str, Dict[str, Any]] = {
        split: evaluate_module.read_arrays(paths.input_root, split)
        for split in DEFAULT_SPLITS
    }
    identity_rows: List[Dict[str, Any]] = []
    sample_indices_by_split: Dict[str, np.ndarray] = {}
    sample_max_by_split = {
        "A_train": int(args.train_sample_max_rows),
        "A_val": int(args.heldout_sample_max_rows),
        "B_confirm": int(args.heldout_sample_max_rows),
    }
    for split, arrays in arrays_by_split.items():
        n_rows = int(arrays["n"])
        split_root = paths.input_root / split
        prepared_files = {
            "x_num": split_root / "x_num.float32.mmap",
            "x_cat": split_root / "x_cat.int64.mmap",
            "y_current": split_root / "y_current.float32.mmap",
            "y_next": split_root / "y_next.float32.mmap",
            "user_id": split_root / "user_id.int64.mmap",
            "bundle_step_index": split_root / "bundle_step_index.int64.mmap",
            "sequence_offsets": split_root / "sequence_offsets.npy",
        }
        missing_prepared = [str(path) for path in prepared_files.values() if not path.exists()]
        if missing_prepared:
            raise FileNotFoundError("Missing prepared-input files:\n" + "\n".join(missing_prepared))
        prepared_file_hashes = {
            name: file_sha256(path)
            for name, path in prepared_files.items()
        }
        prepared_content_signature = stable_json_hash(prepared_file_hashes)
        identifiers_hash = hash_numeric_arrays([arrays["user_id"], arrays["step"]])
        targets_hash = hash_numeric_arrays([arrays["y"]])
        next_targets_hash = hash_numeric_arrays([arrays["y_next"]])
        indices = make_sample_indices(n_rows, sample_max_by_split[split], int(args.sample_seed))
        sample_indices_by_split[split] = indices
        sample_key_hash = hash_numeric_arrays([
            np.asarray(arrays["user_id"])[indices],
            np.asarray(arrays["step"])[indices],
        ])
        sample_target_hash = hash_numeric_arrays([np.asarray(arrays["y"])[indices]])
        np.save(metadata_root / f"sample_indices_{split}.npy", indices)
        identity_rows.append({
            "seed": seed,
            "split": split,
            "rows": n_rows,
            "users": int(len(np.unique(np.asarray(arrays["user_id"])))),
            "identifier_hash": identifiers_hash,
            "current_target_hash": targets_hash,
            "next_target_hash": next_targets_hash,
            "prepared_content_signature": prepared_content_signature,
            "x_num_file_sha256": prepared_file_hashes["x_num"],
            "x_cat_file_sha256": prepared_file_hashes["x_cat"],
            "sequence_offsets_file_sha256": prepared_file_hashes["sequence_offsets"],
            "sample_rows": int(len(indices)),
            "sample_seed": int(args.sample_seed),
            "sample_index_hash": hashlib.sha256(indices.tobytes()).hexdigest(),
            "sample_key_hash": sample_key_hash,
            "sample_current_target_hash": sample_target_hash,
        })
    sample_audit = pd.DataFrame(identity_rows)
    write_table(sample_audit, table_root / "shared_sample_audit")

    metric_rows: List[Dict[str, Any]] = []
    pc_frames: List[pd.DataFrame] = []
    training_summary: Dict[str, Any] = {}
    model_contract_rows: List[Dict[str, Any]] = []
    for model_label in MODEL_ORDER:
        model, encoder, config = load_single_encoder_model(
            model_label,
            paths,
            device,
            train_module,
            evaluate_module,
            task_module,
        )
        run_nonlinear = bool(args.run_nonlinear and (seed == int(args.nonlinear_seed)))
        hidden_train, target_train, user_train, step_train = collect_hidden_sample(
            encoder,
            arrays_by_split["A_train"],
            sample_indices_by_split["A_train"],
            device,
            args.chunk_len,
        )
        train_sample_hash = hash_numeric_arrays([user_train, step_train, target_train])
        artifacts = fit_geometry_artifacts(
            hidden=hidden_train,
            targets=target_train,
            pca_components=args.pca_components,
            pc_alignment_components=args.pc_alignment_components,
            ridge_alpha=args.ridge_alpha,
            seed=seed,
            run_nonlinear=run_nonlinear,
        )
        training_summary[model_label] = dict(artifacts["summary"])
        training_summary[model_label]["train_sample_hash"] = train_sample_hash
        training_metrics, train_pc = evaluate_geometry_artifacts(
            artifacts,
            hidden_train,
            target_train,
            "A_train",
            model_label,
            seed,
        )
        metric_rows.append(training_metrics)
        pc_frames.append(train_pc)
        for split in ("A_val", "B_confirm"):
            hidden, targets, users, steps = collect_hidden_sample(
                encoder,
                arrays_by_split[split],
                sample_indices_by_split[split],
                device,
                args.chunk_len,
            )
            expected_hash = sample_audit.loc[
                sample_audit["split"] == split,
                "sample_key_hash",
            ].iloc[0]
            actual_hash = hash_numeric_arrays([users, steps])
            if actual_hash != expected_hash:
                raise RuntimeError(f"Sample-key mismatch for seed={seed}, model={model_label}, split={split}")
            metrics, pc_table = evaluate_geometry_artifacts(
                artifacts,
                hidden,
                targets,
                split,
                model_label,
                seed,
            )
            metric_rows.append(metrics)
            pc_frames.append(pc_table)
            del hidden, targets, users, steps
        if bool(args.write_seed_artifacts and seed == int(args.artifact_seed)):
            artifact_path = artifact_root / f"geometry_artifacts_{model_label}.pkl"
            with artifact_path.open("wb") as handle:
                pickle.dump(artifacts, handle, protocol=pickle.HIGHEST_PROTOCOL)
        checkpoint_path = {
            "predictive_state": paths.predictive_checkpoint,
            "pure_ssl": paths.pure_checkpoint,
            "task_only": paths.task_checkpoint,
        }[model_label]
        model_contract_rows.append({
            "seed": seed,
            "model": model_label,
            "model_display": MODEL_LABELS[model_label],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "training_manifest": str({
                "predictive_state": paths.predictive_training_manifest,
                "pure_ssl": paths.pure_training_manifest,
                "task_only": paths.task_training_manifest,
            }[model_label]),
            **architectures[model_label],
            "hidden_reference": HIDDEN_REFERENCE,
            "nonlinear_probe_fitted": run_nonlinear,
        })
        del artifacts, hidden_train, target_train, user_train, step_train, encoder, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_frame = pd.DataFrame(metric_rows)
    pc_frame = pd.concat(pc_frames, ignore_index=True)
    write_table(metrics_frame, table_root / "geometry_metrics")
    write_table(pc_frame, table_root / "pc_macro_correlations")
    write_table(pd.DataFrame(model_contract_rows), table_root / "model_objective_contracts")

    reconstruction = pd.DataFrame()
    if seed == int(args.reference_seed):
        reconstruction = formal_seed42_reconstruction_audit(
            paths=paths,
            metrics=metrics_frame,
            pc_table=pc_frame,
            training_summaries=training_summary,
            tolerance=float(args.reconstruction_tolerance),
            require_reference=bool(args.require_formal_reconstruction),
        )
        write_table(reconstruction, table_root / "formal_geometry_reconstruction_audit")

    model_root_sources = {
        "predictive_state": paths.predictive_training_manifest,
        "pure_ssl": paths.pure_training_manifest,
        "task_only": paths.task_training_manifest,
    }
    source_files = [train_script, evaluate_script, task_script, input_manifest_path, normalizer_path]
    source_files.extend(model_root_sources.values())
    source_files.extend([paths.predictive_checkpoint, paths.pure_checkpoint, paths.task_checkpoint])
    source_inventory = [
        {
            "path": str(path.resolve()),
            "sha256": file_sha256(path.resolve()),
            "bytes": int(path.stat().st_size),
        }
        for path in source_files
    ]
    save_json({"sources": source_inventory}, metadata_root / "source_hashes.json")

    manifest = {
        "analysis": "hidden-state geometry in objective-control representations",
        "status": "post hoc supplementary audit",
        "seed": seed,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_root": str(paths.experiment_root),
        "input_root": str(paths.input_root),
        "input_manifest": str(input_manifest_path),
        "input_manifest_sha256": file_sha256(input_manifest_path),
        "input_schema_contract": schema_contract,
        "input_schema_signature": schema_signature,
        "input_leakage_audit": leakage_audit,
        "source_audit": source_audit,
        "architectures": architectures,
        "training_manifest_objective_contracts": {
            label: {
                "path": str(model_root_sources[label]),
                "sha256": file_sha256(model_root_sources[label]),
                "config": training_manifests[label].get("config"),
                "control_boundary": training_manifests[label].get("control_boundary"),
            }
            for label in MODEL_ORDER
        },
        "sample_contract": {
            "sample_seed": int(args.sample_seed),
            "train_sample_max_rows": int(args.train_sample_max_rows),
            "heldout_sample_max_rows": int(args.heldout_sample_max_rows),
            "same_row_indices_for_all_models_within_seed": True,
            "same_sample_seed_for_all_seeds": True,
            "full_sequence_context_preserved": True,
            "hidden_reference": HIDDEN_REFERENCE,
        },
        "geometry_contract": {
            "pca_components": int(args.pca_components),
            "pc_alignment_components": int(args.pc_alignment_components),
            "ridge_alpha": float(args.ridge_alpha),
            "cca_components": 2,
            "pc_indices_selected_on": "A_train only",
            "heldout_pc_selection": False,
            "nonlinear_comparison": "PCA64 Ridge versus PCA64 HGB on identical features",
            "nonlinear_seed": int(args.nonlinear_seed),
            "nonlinear_all_seeds": False,
            "macro_bottleneck_or_residualisation_run": False,
            "drift_or_transition_metrics_run": False,
            "clustering_run": False,
        },
        "analysis_boundaries": {
            "post_hoc": True,
            "preregistered": False,
            "checkpoints_frozen": True,
            "B_confirm_used_for_fitting": False,
            "B_confirm_used_for_selection": False,
            "formal_control_outputs_modified": False,
            "causal_loss_ablation_claim": False,
            "objective_controls_differ_in_checkpoint_selection_and_supervision_contract": True,
            "CCA_is_supervised_diagnostic": True,
            "cross_model_basis_vectors_compared": False,
        },
        "training_geometry_summaries": training_summary,
        "formal_reconstruction": {
            "required": bool(args.require_formal_reconstruction),
            "available": bool(paths.formal_geometry_train_root is not None),
            "train_root": str(paths.formal_geometry_train_root) if paths.formal_geometry_train_root else None,
            "eval_root": str(paths.formal_geometry_eval_root) if paths.formal_geometry_eval_root else None,
            "rows": int(len(reconstruction)),
            "all_hard_gates_passed": bool(
                reconstruction.empty
                or reconstruction.loc[reconstruction["hard_gate"] == True, "passed"].all()
            ),
        },
        "outputs": {
            "geometry_metrics": str(locate_table(table_root / "geometry_metrics")),
            "pc_macro_correlations": str(locate_table(table_root / "pc_macro_correlations")),
            "shared_sample_audit": str(locate_table(table_root / "shared_sample_audit")),
            "model_objective_contracts": str(locate_table(table_root / "model_objective_contracts")),
        },
    }
    save_json(manifest, metadata_root / "seed_geometry_manifest.json")
    print(f"[objective geometry] completed seed {seed}: {metadata_root / 'seed_geometry_manifest.json'}")


def summarize_metric_group(values: pd.Series, confidence: float) -> Dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    count = int(len(numeric))
    if count == 0:
        return {
            "n_seeds": 0,
            "mean": np.nan,
            "sample_sd": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
        }
    mean = float(numeric.mean())
    sample_sd = float(numeric.std(ddof=1)) if count > 1 else np.nan
    if count > 1 and np.isfinite(sample_sd):
        critical = float(student_t.ppf(0.5 + confidence / 2.0, df=count - 1))
        half = critical * sample_sd / math.sqrt(count)
        lower = mean - half
        upper = mean + half
    else:
        lower = np.nan
        upper = np.nan
    return {
        "n_seeds": count,
        "mean": mean,
        "sample_sd": sample_sd,
        "ci_lower": lower,
        "ci_upper": upper,
        "minimum": float(numeric.min()),
        "maximum": float(numeric.max()),
    }


def metric_long(frame: pd.DataFrame) -> pd.DataFrame:
    identity = ["seed", "model", "model_display", "split"]
    excluded = set(identity + [
        "sample_rows",
        "assigned_pc_M",
        "assigned_pc_Psi",
        "descriptive_split_max_pc_M",
        "descriptive_split_max_pc_Psi",
        "cca_original_component_1",
        "cca_original_component_2",
    ])
    metric_columns = [
        column for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    rows: List[Dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        for metric in metric_columns:
            value = record.get(metric)
            try:
                number = float(value)
            except Exception:
                continue
            if not np.isfinite(number):
                continue
            rows.append({
                "seed": int(record["seed"]),
                "model": str(record["model"]),
                "model_display": str(record["model_display"]),
                "split": str(record["split"]),
                "metric": metric,
                "value": number,
            })
    return pd.DataFrame(rows)


def metric_direction(metric: str) -> str:
    if "rmse" in metric or "mae" in metric:
        return "lower_is_better"
    if "clip_fraction" in metric:
        return "descriptive"
    if metric.startswith("matched_nonlinear_gain"):
        return "descriptive"
    return "higher_is_better"


def build_paired_contrasts(long_frame: pd.DataFrame, confidence: float) -> pd.DataFrame:
    primary_metrics = {
        "cca_corr_1",
        "cca_corr_2",
        "raw_ridge_corr_M",
        "raw_ridge_corr_Psi",
        "raw_ridge_rmse_M",
        "raw_ridge_rmse_Psi",
        "assigned_pc_abs_corr_M",
        "assigned_pc_abs_corr_Psi",
        "pca_pc1_pc2_explained_variance",
        "matched_nonlinear_gain_corr_M",
        "matched_nonlinear_gain_corr_Psi",
        "matched_nonlinear_gain_rmse_reduction_M",
        "matched_nonlinear_gain_rmse_reduction_Psi",
    }
    selected = long_frame[long_frame["metric"].isin(primary_metrics)].copy()
    rows: List[Dict[str, Any]] = []
    for split in sorted(selected["split"].unique()):
        for metric in sorted(selected["metric"].unique()):
            subset = selected[(selected["split"] == split) & (selected["metric"] == metric)]
            pivot = subset.pivot_table(index="seed", columns="model", values="value", aggfunc="first")
            if "predictive_state" not in pivot.columns:
                continue
            direction = metric_direction(metric)
            for control in ("pure_ssl", "task_only"):
                if control not in pivot.columns:
                    continue
                paired = pivot[["predictive_state", control]].dropna()
                if direction == "lower_is_better":
                    differences = paired[control] - paired["predictive_state"]
                    definition = f"{control} minus predictive_state; positive favours predictive_state"
                else:
                    differences = paired["predictive_state"] - paired[control]
                    definition = f"predictive_state minus {control}; positive favours predictive_state"
                summary = summarize_metric_group(differences, confidence)
                rows.append({
                    "split": split,
                    "metric": metric,
                    "metric_direction": direction,
                    "control": control,
                    "control_display": MODEL_LABELS[control],
                    "difference_definition": definition,
                    "positive_fraction": float(np.mean(differences > 0)) if len(differences) else np.nan,
                    **summary,
                })
    return pd.DataFrame(rows)


def finalize(args: argparse.Namespace) -> None:
    result_root = args.result_root.resolve()
    seeds = [int(value) for value in args.seeds]
    manifests: Dict[int, Dict[str, Any]] = {}
    metrics_frames: List[pd.DataFrame] = []
    pc_frames: List[pd.DataFrame] = []
    sample_frames: List[pd.DataFrame] = []
    contract_frames: List[pd.DataFrame] = []
    reconstruction_frames: List[pd.DataFrame] = []
    for seed in seeds:
        seed_root = result_root / "seeds" / f"seed{seed}"
        manifest_path = seed_root / "metadata" / "seed_geometry_manifest.json"
        if not manifest_path.exists():
            if args.require_all_seeds:
                raise FileNotFoundError(manifest_path)
            continue
        manifest = load_json(manifest_path)
        manifests[seed] = manifest
        metrics_frames.append(read_table(seed_root / "tables" / "geometry_metrics"))
        pc_frames.append(read_table(seed_root / "tables" / "pc_macro_correlations"))
        sample_frames.append(read_table(seed_root / "tables" / "shared_sample_audit"))
        contract_frames.append(read_table(seed_root / "tables" / "model_objective_contracts"))
        reconstruction_base = seed_root / "tables" / "formal_geometry_reconstruction_audit"
        try:
            reconstruction = read_table(reconstruction_base)
            reconstruction["seed"] = seed
            reconstruction_frames.append(reconstruction)
        except FileNotFoundError:
            pass
    if args.require_all_seeds and len(manifests) != len(seeds):
        raise RuntimeError(f"Expected {len(seeds)} seeds, found {len(manifests)}")
    if not manifests:
        raise RuntimeError("No completed seed outputs were found.")

    schema_signatures = {str(manifest.get("input_schema_signature")) for manifest in manifests.values()}
    if len(schema_signatures) != 1:
        raise RuntimeError("Seed runs do not share one prepared-input schema contract.")
    sample_all = pd.concat(sample_frames, ignore_index=True)
    identity_gates: List[Dict[str, Any]] = []
    for split in DEFAULT_SPLITS:
        subset = sample_all[sample_all["split"] == split]
        gates = {
            "identifier_hash": subset["identifier_hash"].nunique(dropna=False) == 1,
            "current_target_hash": subset["current_target_hash"].nunique(dropna=False) == 1,
            "next_target_hash": subset["next_target_hash"].nunique(dropna=False) == 1,
            "prepared_content_signature": subset["prepared_content_signature"].nunique(dropna=False) == 1,
            "sample_index_hash": subset["sample_index_hash"].nunique(dropna=False) == 1,
            "sample_key_hash": subset["sample_key_hash"].nunique(dropna=False) == 1,
            "sample_current_target_hash": subset["sample_current_target_hash"].nunique(dropna=False) == 1,
        }
        for gate, passed in gates.items():
            identity_gates.append({"split": split, "gate": gate, "passed": bool(passed)})
        if not all(gates.values()):
            failed = [name for name, passed in gates.items() if not passed]
            raise RuntimeError(f"Cross-seed input/sample identity failed for {split}: {failed}")

    metrics_all = pd.concat(metrics_frames, ignore_index=True)
    pc_all = pd.concat(pc_frames, ignore_index=True)
    contracts_all = pd.concat(contract_frames, ignore_index=True)
    long_frame = metric_long(metrics_all)
    summary_rows: List[Dict[str, Any]] = []
    for (model, model_display, split, metric), group in long_frame.groupby(
        ["model", "model_display", "split", "metric"],
        sort=True,
    ):
        summary_rows.append({
            "model": model,
            "model_display": model_display,
            "split": split,
            "metric": metric,
            "metric_direction": metric_direction(metric),
            **summarize_metric_group(group["value"], float(args.confidence)),
        })
    summary = pd.DataFrame(summary_rows)
    paired = build_paired_contrasts(long_frame, float(args.confidence))

    table_root = result_root / "tables"
    metadata_root = result_root / "metadata"
    table_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "geometry_metrics_all_seeds": write_table(metrics_all, table_root / "geometry_metrics_all_seeds"),
        "geometry_metrics_long": write_table(long_frame, table_root / "geometry_metrics_long"),
        "six_seed_geometry_summary": write_table(summary, table_root / "six_seed_geometry_summary"),
        "same_seed_objective_contrasts": write_table(paired, table_root / "same_seed_objective_contrasts"),
        "pc_macro_correlations_all_seeds": write_table(pc_all, table_root / "pc_macro_correlations_all_seeds"),
        "shared_sample_audit_all_seeds": write_table(sample_all, table_root / "shared_sample_audit_all_seeds"),
        "model_objective_contracts_all_seeds": write_table(contracts_all, table_root / "model_objective_contracts_all_seeds"),
        "cross_seed_identity_gates": write_table(pd.DataFrame(identity_gates), table_root / "cross_seed_identity_gates"),
    }
    if reconstruction_frames:
        reconstruction_all = pd.concat(reconstruction_frames, ignore_index=True)
        outputs["formal_geometry_reconstruction_audit"] = write_table(
            reconstruction_all,
            table_root / "formal_geometry_reconstruction_audit",
        )
    manifest = {
        "analysis": "hidden-state geometry in objective-control representations",
        "status": "post hoc supplementary audit",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "requested_seeds": seeds,
        "completed_seeds": sorted(manifests),
        "reference_seed": int(args.reference_seed),
        "confidence_level": float(args.confidence),
        "input_schema_signature": next(iter(schema_signatures)),
        "cross_seed_identity_gates_passed": bool(pd.DataFrame(identity_gates)["passed"].all()),
        "analysis_boundaries": {
            "checkpoints_frozen": True,
            "A_train_only_fitting": True,
            "A_val_and_B_confirm_output_only": True,
            "common_sample_bank_across_models_and_seeds": True,
            "direct_state_or_closure_supervision_absent_in_pure_and_task_controls": True,
            "objective_controls_are_not_single_factor_causal_ablations": True,
            "macro_bottleneck_analysis_repeated": False,
            "residualisation_repeated": False,
            "drift_transition_or_clustering_repeated": False,
            "nonlinear_capacity_audit_seed": int(args.reference_seed),
        },
        "seed_manifests": {
            str(seed): str((result_root / "seeds" / f"seed{seed}" / "metadata" / "seed_geometry_manifest.json").resolve())
            for seed in sorted(manifests)
        },
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
    }
    save_json(manifest, metadata_root / "objective_control_hidden_geometry_manifest.json")
    print(f"[objective geometry] finalized: {metadata_root / 'objective_control_hidden_geometry_manifest.json'}")


def self_test() -> None:
    first = make_sample_indices(1000, 100, 42)
    second = make_sample_indices(1000, 100, 42)
    if not np.array_equal(first, second) or len(np.unique(first)) != 100:
        raise RuntimeError("Sample-index self-test failed.")
    generator = np.random.default_rng(7)
    scores = generator.normal(size=(2000, 16))
    targets = np.column_stack([
        0.8 * scores[:, 2] + 0.1 * generator.normal(size=2000),
        -0.7 * scores[:, 5] + 0.1 * generator.normal(size=2000),
    ])
    correlations = pc_correlations(scores, targets, 16)
    assignment = distinct_pc_assignment(correlations)
    if assignment["pc_index_M"] != 3 or assignment["pc_index_Psi"] != 6:
        raise RuntimeError("Distinct-PC assignment self-test failed.")
    hidden = generator.normal(size=(2000, 32)).astype(np.float32)
    targets = np.column_stack([
        hidden[:, 0] + 0.2 * hidden[:, 1],
        hidden[:, 2] - 0.1 * hidden[:, 3],
    ]).astype(np.float32)
    artifacts = fit_geometry_artifacts(hidden, targets, 16, 8, 1.0, 42, True)
    metrics, pc = evaluate_geometry_artifacts(artifacts, hidden, targets, "A_train", "predictive_state", 42)
    if metrics["cca_corr_2"] < 0.9 or len(pc) != 8:
        raise RuntimeError("Geometry self-test failed.")
    print("objective-control hidden-geometry self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit hidden-state geometry across frozen Event-SSL objective controls.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run-seed")
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--experiment-root", type=Path, required=True)
    run_parser.add_argument("--result-root", type=Path, required=True)
    run_parser.add_argument("--input-root", type=Path, default=None)
    run_parser.add_argument("--predictive-checkpoint", type=Path, default=None)
    run_parser.add_argument("--pure-checkpoint", type=Path, default=None)
    run_parser.add_argument("--task-checkpoint", type=Path, default=None)
    run_parser.add_argument("--formal-geometry-root", type=Path, default=None)
    run_parser.add_argument("--train-script", type=Path, required=True)
    run_parser.add_argument("--evaluate-script", type=Path, required=True)
    run_parser.add_argument("--task-script", type=Path, required=True)
    run_parser.add_argument("--train-sample-max-rows", type=int, default=300000)
    run_parser.add_argument("--heldout-sample-max-rows", type=int, default=250000)
    run_parser.add_argument("--sample-seed", type=int, default=42)
    run_parser.add_argument("--chunk-len", type=int, default=512)
    run_parser.add_argument("--pca-components", type=int, default=64)
    run_parser.add_argument("--pc-alignment-components", type=int, default=16)
    run_parser.add_argument("--ridge-alpha", type=float, default=1.0)
    run_parser.add_argument("--run-nonlinear", action="store_true")
    run_parser.add_argument("--nonlinear-seed", type=int, default=42)
    run_parser.add_argument("--write-seed-artifacts", action="store_true")
    run_parser.add_argument("--artifact-seed", type=int, default=42)
    run_parser.add_argument("--reference-seed", type=int, default=42)
    run_parser.add_argument("--require-formal-reconstruction", action="store_true")
    run_parser.add_argument("--reconstruction-tolerance", type=float, default=2e-3)
    run_parser.add_argument("--overwrite", action="store_true")

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--result-root", type=Path, required=True)
    finalize_parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    finalize_parser.add_argument("--reference-seed", type=int, default=42)
    finalize_parser.add_argument("--confidence", type=float, default=0.95)
    finalize_parser.add_argument("--require-all-seeds", action="store_true")

    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.command == "run-seed":
        run_seed(args)
    elif args.command == "finalize":
        finalize(args)
    else:
        parser.error("Specify run-seed, finalize, or --self-test")


if __name__ == "__main__":
    main()
