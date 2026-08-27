#!/usr/bin/env python3
from __future__ import annotations

"""Fit Stage-5 macro-sufficiency probes on a frozen predictive-state Event-SSL model."""

import argparse
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TASK_FEATURE_NAME = "current_accuracy_diagnostic_only"
REPRESENTATIONS = ("full_hidden", "macro_only", "residual_hidden")
REPRESENTATION_CLUSTER_K = 6
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
    if contract.get("verified") is not True:
        raise RuntimeError("Stage-4 inputs do not contain a verified Stage-1 fixed-K contract.")
    if int(contract.get("macrostate_k", -1)) != REPRESENTATION_CLUSTER_K:
        raise RuntimeError("Stage-5 requires the fixed Stage-1 K=6 contract.")
    if contract.get("macrostate_k_rule") != "fixed a priori":
        raise RuntimeError("Stage-1 mesostate K was not fixed a priori.")
    return manifest


def load_model(checkpoint: Path, device: torch.device, train_module):
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    config = checkpoint_data["config"]
    shapes = checkpoint_data["model_shapes"]
    if config.get("model_kind") != "predictive_state":
        raise RuntimeError(
            "Macro-sufficiency requires the predictive_state Event-SSL checkpoint; "
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


def task_feature_index(input_root: Path, task_feature_name: str) -> Tuple[Optional[int], float, float, str]:
    normalizer_path = input_root / "metadata" / "normalizer.json"
    if not normalizer_path.exists():
        return None, float("nan"), float("nan"), "normalizer_missing"
    normalizer = load_json(normalizer_path)
    names = list(normalizer.get("numeric_feature_names", []))
    if task_feature_name not in names:
        return None, float("nan"), float("nan"), "task_feature_missing"
    index = int(names.index(task_feature_name))
    return index, float(normalizer["mean"][index]), float(normalizer["std"][index]), "ok"


def recover_task_target(
    x_num: np.ndarray,
    task_index: Optional[int],
    mean: float,
    std: float,
) -> Optional[np.ndarray]:
    if task_index is None:
        return None
    target = np.asarray(x_num[:, task_index], dtype=np.float32) * np.float32(std) + np.float32(mean)
    return np.clip(target, 0.0, 1.0).astype(np.float32)


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
    task_index: Optional[int],
    task_mean: float,
    task_std: float,
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
    task_targets: List[np.ndarray] = []
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
            x_num_np = np.asarray(arrays["x_num"][position:end])
            x_cat_np = np.asarray(arrays["x_cat"][position:end])
            x_num = torch.from_numpy(x_num_np).unsqueeze(0).to(device, non_blocking=True)
            x_cat = torch.from_numpy(x_cat_np).unsqueeze(0).to(device, non_blocking=True)
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
                task = recover_task_target(x_num_np, task_index, task_mean, task_std)
                if task is not None:
                    task_targets.append(task[take])
            previous_hidden = h_after[:, -1:, :].detach()
            position = end

    if not hidden_before:
        raise RuntimeError("No hidden-state sample was collected.")

    output = {
        "h_before": np.concatenate(hidden_before, axis=0),
        "h_after": np.concatenate(hidden_after, axis=0),
        "pred_state": np.concatenate(predicted_state, axis=0),
        "pred_next_state": np.concatenate(predicted_next_state, axis=0),
        "y": np.concatenate(target_state, axis=0),
        "y_next": np.concatenate(target_next_state, axis=0),
    }
    if task_targets:
        output["task_y"] = np.concatenate(task_targets, axis=0)
    return output


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=float(alpha), fit_intercept=True)).fit(x, y)


def fit_task_classifier(x: np.ndarray, task_y: Optional[np.ndarray], seed: int):
    if task_y is None:
        return None
    binary = (np.asarray(task_y) >= 0.5).astype(np.int32)
    if np.unique(binary).size < 2:
        return None
    classifier = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=1000,
            tol=1e-3,
            random_state=seed,
            n_jobs=-1,
        ),
    )
    classifier.fit(x, binary)
    return classifier


def predict_task(classifier, x: np.ndarray) -> Optional[np.ndarray]:
    if classifier is None:
        return None
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(x)[:, 1].astype(np.float32)
    score = classifier.decision_function(x)
    return (1.0 / (1.0 + np.exp(-score))).astype(np.float32)


def task_metrics(task_y: Optional[np.ndarray], probability: Optional[np.ndarray], prefix: str) -> Dict[str, float]:
    if task_y is None or probability is None:
        return {f"{prefix}_task_available": 0.0}
    target = np.asarray(task_y, dtype=float)
    prob = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    valid = np.isfinite(target) & np.isfinite(prob)
    output: Dict[str, float] = {
        f"{prefix}_task_available": 1.0,
        f"{prefix}_task_rows": float(valid.sum()),
    }
    if valid.sum() < 10:
        return output
    output[f"{prefix}_task_rmse"] = float(np.sqrt(np.mean((prob[valid] - target[valid]) ** 2)))
    output[f"{prefix}_task_bce"] = float(
        -np.mean(target[valid] * np.log(prob[valid]) + (1.0 - target[valid]) * np.log(1.0 - prob[valid]))
    )
    binary = (target[valid] >= 0.5).astype(int)
    if np.unique(binary).size == 2:
        output[f"{prefix}_task_auc"] = float(roc_auc_score(binary, prob[valid]))
        output[f"{prefix}_task_accuracy_0p5"] = float(np.mean((prob[valid] >= 0.5).astype(int) == binary))
    return output


def fit_representation_cluster(x: np.ndarray, seed: int):
    pipeline = make_pipeline(
        StandardScaler(),
        MiniBatchKMeans(
            n_clusters=REPRESENTATION_CLUSTER_K,
            random_state=int(seed),
            n_init=10,
            batch_size=8192,
        ),
    )
    pipeline.fit(x)
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Stage-5 macro-sufficiency probes.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-script", type=Path, default=None)
    parser.add_argument("--evaluate-script", type=Path, default=None)
    parser.add_argument("--train-split", default="A_train")
    parser.add_argument("--sample-max-rows", type=int, default=600000)
    parser.add_argument("--chunk-len", type=int, default=512)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-feature-name", default=TASK_FEATURE_NAME)
    parser.add_argument("--torch-num-threads", type=int, default=0)
    args = parser.parse_args()

    if args.torch_num_threads > 0:
        torch.set_num_threads(int(args.torch_num_threads))

    input_manifest = validate_input_contract(args.input_root)
    train_script = resolve_script(args.train_script, TRAIN_SCRIPT_BASENAME)
    evaluate_script = resolve_script(args.evaluate_script, EVALUATE_SCRIPT_BASENAME)
    train_module = import_module(train_script, "stage5_macro_train_event_ssl")
    evaluate_module = import_module(evaluate_script, "stage5_macro_evaluate_event_ssl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, train_module)
    arrays = evaluate_module.read_arrays(args.input_root, args.train_split)
    task_index, task_mean, task_std, task_status = task_feature_index(args.input_root, args.task_feature_name)
    sample = collect_sample(
        model,
        arrays,
        device,
        args.chunk_len,
        args.sample_max_rows,
        args.seed,
        task_index,
        task_mean,
        task_std,
    )

    hidden_before = sample["h_before"]
    hidden_after = sample["h_after"]
    predicted_state = np.clip(sample["pred_state"], -1.0, 1.0)
    predicted_next_state = np.clip(sample["pred_next_state"], -1.0, 1.0)
    target_state = sample["y"]
    target_next_state = sample["y_next"]
    task_y = sample.get("task_y")
    macro = macro_features(predicted_state)
    macro_next = macro_features(predicted_next_state)

    full_current_probe = fit_ridge(hidden_before, target_state, args.ridge_alpha)
    full_next_probe = fit_ridge(hidden_after, target_next_state, args.ridge_alpha)
    full_task = fit_task_classifier(hidden_before, task_y, args.seed)
    full_cluster = fit_representation_cluster(hidden_before, args.seed)

    macro_closure_probe = fit_ridge(macro, target_next_state, args.ridge_alpha)
    macro_task = fit_task_classifier(macro, task_y, args.seed + 1)
    macro_cluster = fit_representation_cluster(predicted_state, args.seed + 1)

    macro_to_hidden_before = fit_ridge(macro, hidden_before, args.ridge_alpha)
    macro_to_hidden_after = fit_ridge(macro_next, hidden_after, args.ridge_alpha)
    residual_before = hidden_before - macro_to_hidden_before.predict(macro).astype(np.float32)
    residual_after = hidden_after - macro_to_hidden_after.predict(macro_next).astype(np.float32)
    residual_current_probe = fit_ridge(residual_before, target_state, args.ridge_alpha)
    residual_next_probe = fit_ridge(residual_after, target_next_state, args.ridge_alpha)
    residual_task = fit_task_classifier(residual_before, task_y, args.seed + 2)
    residual_cluster = fit_representation_cluster(residual_before, args.seed + 2)

    artifacts: Dict[str, object] = {
        "meta": {
            "script": Path(__file__).name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_root": str(args.input_root.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "train_script": str(train_script),
            "evaluate_script": str(evaluate_script),
            "train_split": args.train_split,
            "sample_rows": int(len(target_state)),
            "primary_coordinates": ["M", "Psi"],
            "stage1_fixed_k6_contract": input_manifest["stage1_fixed_k6_contract"],
            "task_feature_name": args.task_feature_name,
            "task_feature_status": task_status,
            "macro_feature_basis": ["M", "Psi", "M^2", "Psi^2", "M*Psi"],
            "representation_sources": list(REPRESENTATIONS),
            "representation_cluster_k": REPRESENTATION_CLUSTER_K,
            "representation_cluster_role": "diagnostic comparison with frozen empirical K=6 labels",
        },
        "full_hidden": {
            "probe_cur": full_current_probe,
            "probe_next": full_next_probe,
            "task_classifier": full_task,
            "rep_cluster": full_cluster,
        },
        "macro_only": {
            "closure_probe_next": macro_closure_probe,
            "task_classifier": macro_task,
            "rep_cluster": macro_cluster,
        },
        "residual_hidden": {
            "macro_to_hidden_before": macro_to_hidden_before,
            "macro_to_hidden_after": macro_to_hidden_after,
            "probe_cur": residual_current_probe,
            "probe_next": residual_next_probe,
            "task_classifier": residual_task,
            "rep_cluster": residual_cluster,
        },
    }

    rows: List[Dict[str, float | str]] = []

    def add_training_metrics(
        representation: str,
        predicted_current: np.ndarray,
        predicted_next: np.ndarray,
        task_probability: Optional[np.ndarray],
    ) -> None:
        row: Dict[str, float | str] = {"split": args.train_split, "representation": representation}
        for index, name in enumerate(("M", "Psi")):
            current_error = predicted_current[:, index] - target_state[:, index]
            next_error = predicted_next[:, index] - target_next_state[:, index]
            row[f"coordinate_rmse_{name}"] = float(np.sqrt(np.mean(current_error * current_error)))
            row[f"coordinate_corr_{name}"] = float(np.corrcoef(predicted_current[:, index], target_state[:, index])[0, 1])
            row[f"one_step_rmse_{name}"] = float(np.sqrt(np.mean(next_error * next_error)))
            row[f"one_step_corr_{name}"] = float(np.corrcoef(predicted_next[:, index], target_next_state[:, index])[0, 1])
        row.update(task_metrics(task_y, task_probability, representation))
        rows.append(row)

    add_training_metrics(
        "full_hidden",
        np.clip(full_current_probe.predict(hidden_before), -1.0, 1.0),
        np.clip(full_next_probe.predict(hidden_after), -1.0, 1.0),
        predict_task(full_task, hidden_before),
    )
    add_training_metrics(
        "macro_only",
        predicted_state,
        np.clip(macro_closure_probe.predict(macro), -1.0, 1.0),
        predict_task(macro_task, macro),
    )
    add_training_metrics(
        "residual_hidden",
        np.clip(residual_current_probe.predict(residual_before), -1.0, 1.0),
        np.clip(residual_next_probe.predict(residual_after), -1.0, 1.0),
        predict_task(residual_task, residual_before),
    )

    output_root = args.output_root.resolve()
    table_root = output_root / "tables"
    metadata_root = output_root / "metadata"
    table_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    dump_pickle(artifacts, metadata_root / "stage5_macro_sufficiency_artifacts.pkl")
    pd.DataFrame(rows).to_csv(table_root / "stage5_macro_sufficiency_training_probe_metrics.csv", index=False)
    save_json(artifacts["meta"], metadata_root / "stage5_macro_sufficiency_training_manifest.json")
    print(f"[Stage5 macro-sufficiency train] wrote {metadata_root / 'stage5_macro_sufficiency_artifacts.pkl'}")


if __name__ == "__main__":
    main()
