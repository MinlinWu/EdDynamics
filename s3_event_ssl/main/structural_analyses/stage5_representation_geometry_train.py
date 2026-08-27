#!/usr/bin/env python3
from __future__ import annotations

"""Fit Stage-5 representation-geometry probes on a frozen predictive-state Event-SSL model."""

import argparse
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-12
EXPECTED_MACROSTATE_K = 6
TRAIN_SCRIPT_BASENAME = "train_event_ssl.py"
EVALUATE_SCRIPT_BASENAME = "evaluate_event_ssl_structure.py"


def import_module(path: Path, module_name: str):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_script(explicit: Optional[Path], basename: str) -> Path:
    return explicit.resolve() if explicit is not None else Path(__file__).resolve().with_name(basename)


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
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(obj), handle, indent=2, ensure_ascii=False, allow_nan=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def validate_input_contract(input_root: Path) -> dict:
    manifest = load_json(input_root / "metadata" / "stage4_input_manifest.json")
    if manifest.get("primary_coordinates") != ["M", "Psi"]:
        raise RuntimeError("Stage-5 requires primary_coordinates exactly ['M', 'Psi'].")
    contract = dict(manifest.get("stage1_fixed_k6_contract", {}))
    if contract.get("verified") is not True or int(contract.get("macrostate_k", -1)) != EXPECTED_MACROSTATE_K:
        raise RuntimeError("Stage-4 inputs do not contain the verified fixed K=6 contract.")
    if contract.get("macrostate_k_rule") != "fixed a priori":
        raise RuntimeError("Stage-1 mesostate K was not fixed a priori.")
    return manifest


def load_model(checkpoint: Path, device: torch.device, train_module):
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    config = checkpoint_data["config"]
    shapes = checkpoint_data["model_shapes"]
    if config.get("model_kind") != "predictive_state":
        raise RuntimeError(
            "Representation geometry requires the predictive_state Event-SSL checkpoint; "
            f"got {config.get('model_kind')!r}."
        )
    model = train_module.PredictiveStateEventSSL(
        n_num=int(shapes["n_num"]),
        n_cat=int(shapes["n_cat"]),
        hash_buckets=int(shapes["hash_buckets"]),
        hidden_dim=int(config["hidden_dim"]),
        input_dim=int(config["input_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        categorical_emb_dim=int(config["categorical_emb_dim"]),
        future_steps=tuple(int(x) for x in config["future_steps"]),
        delta_scale=float(config["delta_scale"]),
    )
    model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


def macro_features(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    m = values[:, 0]
    psi = values[:, 1]
    return np.column_stack([m, psi, m * m, psi * psi, m * psi]).astype(np.float32)


@torch.inference_mode()
def collect_sample(
    model,
    arrays: dict,
    device: torch.device,
    chunk_len: int,
    max_rows: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    n_rows = int(arrays["n"])
    take_n = n_rows if max_rows <= 0 or max_rows >= n_rows else int(max_rows)
    rng = np.random.default_rng(seed)
    selected = np.zeros(n_rows, dtype=bool)
    if take_n < n_rows:
        selected[rng.choice(n_rows, size=take_n, replace=False)] = True
    else:
        selected[:] = True

    hidden_before: List[np.ndarray] = []
    hidden_after: List[np.ndarray] = []
    predicted_state: List[np.ndarray] = []
    predicted_next_state: List[np.ndarray] = []
    target_state: List[np.ndarray] = []
    target_next_state: List[np.ndarray] = []
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
            end = min(position + chunk_len, stop)
            take = selected[position:end]
            x_num = torch.from_numpy(np.asarray(arrays["x_num"][position:end])).unsqueeze(0).to(device, non_blocking=True)
            x_cat = torch.from_numpy(np.asarray(arrays["x_cat"][position:end])).unsqueeze(0).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                embedded = model.embed_inputs(x_num, x_cat)
                h_after, hidden_state = model.rnn(embedded, hidden_state)
                first_before = torch.zeros_like(h_after[:, :1, :]) if previous_hidden is None else previous_hidden
                h_before = torch.cat([first_before, h_after[:, :-1, :]], dim=1)
                state = model.state_head(h_before)
                delta = model.delta_scale * model.delta_head(torch.cat([h_before, embedded], dim=-1))
                next_state = torch.tanh(state + delta)
            if take.any():
                hidden_before.append(h_before.squeeze(0).float().cpu().numpy()[take].astype(np.float32))
                hidden_after.append(h_after.squeeze(0).float().cpu().numpy()[take].astype(np.float32))
                predicted_state.append(state.squeeze(0).float().cpu().numpy()[take].astype(np.float32))
                predicted_next_state.append(next_state.squeeze(0).float().cpu().numpy()[take].astype(np.float32))
                target_state.append(np.asarray(arrays["y"][position:end], dtype=np.float32)[take])
                target_next_state.append(np.asarray(arrays["y_next"][position:end], dtype=np.float32)[take])
            previous_hidden = h_after[:, -1:, :].detach()
            position = end

    if not hidden_before:
        raise RuntimeError("No hidden-state sample was collected.")

    return {
        "h_before": np.concatenate(hidden_before, axis=0),
        "h_after": np.concatenate(hidden_after, axis=0),
        "pred_state": np.concatenate(predicted_state, axis=0),
        "pred_next_state": np.concatenate(predicted_next_state, axis=0),
        "y": np.concatenate(target_state, axis=0),
        "y_next": np.concatenate(target_next_state, axis=0),
    }


def participation_ratio(eigenvalues: np.ndarray) -> float:
    values = np.asarray(eigenvalues, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return float("nan")
    return float((values.sum() ** 2) / np.sum(values * values))


def effective_rank(eigenvalues: np.ndarray) -> float:
    values = np.asarray(eigenvalues, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return float("nan")
    probability = values / values.sum()
    return float(np.exp(-np.sum(probability * np.log(probability + EPS))))


def twonn_dimension(x: np.ndarray, max_rows: int = 10000, seed: int = 42) -> float:
    values = np.asarray(x, dtype=np.float32)
    if len(values) < 50:
        return float("nan")
    if max_rows > 0 and len(values) > max_rows:
        rng = np.random.default_rng(seed)
        values = values[np.sort(rng.choice(len(values), size=max_rows, replace=False))]
    standardized = StandardScaler().fit_transform(values)
    neighbors = NearestNeighbors(n_neighbors=3, algorithm="auto", metric="euclidean")
    neighbors.fit(standardized)
    distances, _ = neighbors.kneighbors(standardized, n_neighbors=3)
    first = np.maximum(distances[:, 1], EPS)
    second = np.maximum(distances[:, 2], EPS)
    ratio = second / first
    log_ratio = np.log(ratio[np.isfinite(ratio) & (ratio > 1)])
    if log_ratio.size < 10:
        return float("nan")
    return float(1.0 / np.mean(log_ratio))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 3:
        return float("nan")
    centered_first = first[valid] - first[valid].mean()
    centered_second = second[valid] - second[valid].mean()
    denominator = np.linalg.norm(centered_first) * np.linalg.norm(centered_second)
    return float(np.dot(centered_first, centered_second) / denominator) if denominator > EPS else float("nan")


def fit_hgb_regressors(x: np.ndarray, y: np.ndarray, seed: int) -> List[HistGradientBoostingRegressor]:
    regressors: List[HistGradientBoostingRegressor] = []
    for index in range(y.shape[1]):
        regressor = HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.06,
            l2_regularization=1e-3,
            random_state=seed + index,
        )
        regressor.fit(x, y[:, index])
        regressors.append(regressor)
    return regressors


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=float(alpha), fit_intercept=True)).fit(x, y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Stage-5 representation-geometry artifacts.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-script", type=Path, default=None)
    parser.add_argument("--evaluate-script", type=Path, default=None)
    parser.add_argument("--train-split", default="A_train")
    parser.add_argument("--sample-max-rows", type=int, default=300000)
    parser.add_argument("--chunk-len", type=int, default=512)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-num-threads", type=int, default=0)
    args = parser.parse_args()

    if args.torch_num_threads > 0:
        torch.set_num_threads(int(args.torch_num_threads))

    input_manifest = validate_input_contract(args.input_root)
    train_script = resolve_script(args.train_script, TRAIN_SCRIPT_BASENAME)
    evaluate_script = resolve_script(args.evaluate_script, EVALUATE_SCRIPT_BASENAME)
    train_module = import_module(train_script, "stage5_geometry_train_event_ssl")
    evaluate_module = import_module(evaluate_script, "stage5_geometry_evaluate_event_ssl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, train_module)
    arrays = evaluate_module.read_arrays(args.input_root, args.train_split)
    sample = collect_sample(model, arrays, device, args.chunk_len, args.sample_max_rows, args.seed)

    hidden_before = sample["h_before"]
    hidden_after = sample["h_after"]
    target_state = sample["y"]
    target_next_state = sample["y_next"]
    predicted_state = np.clip(sample["pred_state"], -1.0, 1.0)
    predicted_next_state = np.clip(sample["pred_next_state"], -1.0, 1.0)

    component_count = int(max(2, min(args.pca_components, hidden_before.shape[1], hidden_before.shape[0] - 1)))
    hidden_scaler = StandardScaler().fit(hidden_before)
    standardized_hidden = hidden_scaler.transform(hidden_before)
    pca = PCA(n_components=component_count, random_state=args.seed).fit(standardized_hidden)
    scores = pca.transform(standardized_hidden).astype(np.float32)
    eigenvalues = pca.explained_variance_.astype(float)

    macro_scaler = StandardScaler().fit(target_state)
    cca = CCA(n_components=2, max_iter=1000)
    cca.fit(standardized_hidden, macro_scaler.transform(target_state))

    linear_current = fit_ridge(hidden_before, target_state, args.ridge_alpha)
    linear_next = fit_ridge(hidden_after, target_next_state, args.ridge_alpha)
    nonlinear_current = fit_hgb_regressors(scores, target_state, args.seed)

    macro = macro_features(predicted_state)
    macro_next = macro_features(predicted_next_state)
    macro_to_hidden_before = fit_ridge(macro, hidden_before, args.ridge_alpha)
    macro_to_hidden_after = fit_ridge(macro_next, hidden_after, args.ridge_alpha)
    residual_before = hidden_before - macro_to_hidden_before.predict(macro).astype(np.float32)
    residual_after = hidden_after - macro_to_hidden_after.predict(macro_next).astype(np.float32)
    residual_current = fit_ridge(residual_before, target_state, args.ridge_alpha)
    residual_next = fit_ridge(residual_after, target_next_state, args.ridge_alpha)

    pc_rows: List[Dict[str, float | str]] = []
    for index in range(min(16, scores.shape[1])):
        component = scores[:, index]
        pc_rows.append({
            "component": f"PC{index + 1}",
            "corr_M": correlation(component, target_state[:, 0]),
            "corr_Psi": correlation(component, target_state[:, 1]),
        })

    hidden_canonical, macro_canonical = cca.transform(
        standardized_hidden,
        macro_scaler.transform(target_state),
    )
    canonical_correlations = [
        correlation(hidden_canonical[:, index], macro_canonical[:, index])
        for index in range(min(2, hidden_canonical.shape[1]))
    ]

    summary = {
        "train_split": args.train_split,
        "sample_rows": int(len(hidden_before)),
        "hidden_dim": int(hidden_before.shape[1]),
        "pca_components": int(component_count),
        "pca_explained_variance_ratio_first_10": pca.explained_variance_ratio_[:10].tolist(),
        "pca_explained_variance_ratio_cumulative_10": float(np.sum(pca.explained_variance_ratio_[:10])),
        "pca_explained_variance_ratio_cumulative_all_fit_components": float(np.sum(pca.explained_variance_ratio_)),
        "participation_ratio": participation_ratio(eigenvalues),
        "effective_rank": effective_rank(eigenvalues),
        "twonn_intrinsic_dimension_estimate": twonn_dimension(hidden_before, seed=args.seed),
        "cca_correlations": canonical_correlations,
        "primary_coordinates": ["M", "Psi"],
        "stage1_fixed_k6_contract": input_manifest["stage1_fixed_k6_contract"],
    }

    artifacts = {
        "meta": {
            "script": Path(__file__).name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_root": str(args.input_root.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "train_script": str(train_script),
            "evaluate_script": str(evaluate_script),
            "train_split": args.train_split,
            "primary_coordinates": ["M", "Psi"],
            "stage1_fixed_k6_contract": input_manifest["stage1_fixed_k6_contract"],
            "macro_feature_basis": ["M", "Psi", "M^2", "Psi^2", "M*Psi"],
        },
        "hidden_scaler": hidden_scaler,
        "pca": pca,
        "macro_scaler": macro_scaler,
        "cca": cca,
        "linear_cur": linear_current,
        "linear_next": linear_next,
        "nonlinear_cur": nonlinear_current,
        "macro_to_hidden_before": macro_to_hidden_before,
        "macro_to_hidden_after": macro_to_hidden_after,
        "residual_cur": residual_current,
        "residual_next": residual_next,
        "summary": summary,
    }

    output_root = args.output_root.resolve()
    metadata_root = output_root / "metadata"
    table_root = output_root / "tables"
    metadata_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)
    dump_pickle(artifacts, metadata_root / "stage5_representation_geometry_artifacts.pkl")
    save_json(summary, metadata_root / "stage5_representation_geometry_training_manifest.json")
    pd.DataFrame(pc_rows).to_csv(table_root / "stage5_pc_macro_correlations_train.csv", index=False)
    print(f"[Stage5 representation geometry train] wrote {metadata_root / 'stage5_representation_geometry_artifacts.pkl'}")


if __name__ == "__main__":
    main()
