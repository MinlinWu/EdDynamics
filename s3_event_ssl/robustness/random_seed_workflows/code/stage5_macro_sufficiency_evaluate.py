#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate Stage-5 macro-sufficiency probes on frozen Event-SSL representations."""

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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score

REPRESENTATIONS = ("full_hidden", "macro_only", "residual_hidden")
TASK_FEATURE_NAME = "current_accuracy_diagnostic_only"
TRAIN_SCRIPT_BASENAME = "train_event_ssl.py"
EVALUATE_SCRIPT_BASENAME = "evaluate_event_ssl_structure.py"
EXPECTED_MACROSTATE_K = 6


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


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_table(df: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        output = base.with_suffix(".parquet")
        df.to_parquet(output, index=False)
        return output
    except Exception:
        output = base.with_suffix(".csv.gz")
        df.to_csv(output, index=False, compression="gzip")
        return output


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


def predict_task(classifier, x: np.ndarray) -> Optional[np.ndarray]:
    if classifier is None:
        return None
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(x)[:, 1].astype(np.float32)
    score = classifier.decision_function(x)
    return (1.0 / (1.0 + np.exp(-score))).astype(np.float32)


def task_metrics(task_y: Optional[np.ndarray], probability: Optional[np.ndarray], prefix: str = "task") -> Dict[str, float]:
    if task_y is None or probability is None:
        return {f"{prefix}_available": 0.0}
    target = np.asarray(task_y, dtype=float)
    prob = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    valid = np.isfinite(target) & np.isfinite(prob)
    output: Dict[str, float] = {
        f"{prefix}_available": 1.0,
        f"{prefix}_rows": float(valid.sum()),
    }
    if valid.sum() < 10:
        return output
    output[f"{prefix}_rmse"] = float(np.sqrt(np.mean((prob[valid] - target[valid]) ** 2)))
    output[f"{prefix}_bce"] = float(
        -np.mean(target[valid] * np.log(prob[valid]) + (1.0 - target[valid]) * np.log(1.0 - prob[valid]))
    )
    binary = (target[valid] >= 0.5).astype(int)
    if np.unique(binary).size == 2:
        output[f"{prefix}_auc"] = float(roc_auc_score(binary, prob[valid]))
        output[f"{prefix}_accuracy_0p5"] = float(np.mean((prob[valid] >= 0.5).astype(int) == binary))
    return output


@torch.inference_mode()
def evaluate_split_representations(
    model,
    artifacts: dict,
    arrays: dict,
    device: torch.device,
    chunk_len: int,
    task_info: Tuple[Optional[int], float, float, str],
) -> Dict[str, dict]:
    n_rows = int(arrays["n"])
    output: Dict[str, dict] = {}
    for representation in REPRESENTATIONS:
        output[representation] = {
            "pred_cur": np.zeros((n_rows, 2), dtype=np.float32),
            "pred_next": np.zeros((n_rows, 2), dtype=np.float32),
            "task_prob": np.full(n_rows, np.nan, dtype=np.float32),
            "rep_cluster": np.full(n_rows, -1, dtype=np.int32),
        }

    task_index, task_mean, task_std = task_info[:3]
    task_y = np.full(n_rows, np.nan, dtype=np.float32)
    autocast_enabled = device.type == "cuda"

    for sequence_index in range(len(arrays["offsets"]) - 1):
        start = int(arrays["offsets"][sequence_index])
        stop = int(arrays["offsets"][sequence_index + 1])
        hidden_state = None
        previous_hidden = None
        position = start
        while position < stop:
            end = min(position + chunk_len, stop)
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

            hidden_before = h_before.squeeze(0).float().cpu().numpy().astype(np.float32)
            hidden_after = h_after.squeeze(0).float().cpu().numpy().astype(np.float32)
            predicted_state = np.clip(state.squeeze(0).float().cpu().numpy().astype(np.float32), -1.0, 1.0)
            predicted_next_state = np.clip(next_state.squeeze(0).float().cpu().numpy().astype(np.float32), -1.0, 1.0)
            macro = macro_features(predicted_state)
            macro_next = macro_features(predicted_next_state)
            recovered_task = recover_task_target(x_num_np, task_index, task_mean, task_std)
            if recovered_task is not None:
                task_y[position:end] = recovered_task

            full = artifacts["full_hidden"]
            output["full_hidden"]["pred_cur"][position:end] = np.clip(
                full["probe_cur"].predict(hidden_before), -1.0, 1.0
            ).astype(np.float32)
            output["full_hidden"]["pred_next"][position:end] = np.clip(
                full["probe_next"].predict(hidden_after), -1.0, 1.0
            ).astype(np.float32)
            probability = predict_task(full.get("task_classifier"), hidden_before)
            if probability is not None:
                output["full_hidden"]["task_prob"][position:end] = probability
            output["full_hidden"]["rep_cluster"][position:end] = full["rep_cluster"].predict(hidden_before).astype(np.int32)

            macro_only = artifacts["macro_only"]
            output["macro_only"]["pred_cur"][position:end] = predicted_state
            output["macro_only"]["pred_next"][position:end] = np.clip(
                macro_only["closure_probe_next"].predict(macro), -1.0, 1.0
            ).astype(np.float32)
            probability = predict_task(macro_only.get("task_classifier"), macro)
            if probability is not None:
                output["macro_only"]["task_prob"][position:end] = probability
            output["macro_only"]["rep_cluster"][position:end] = macro_only["rep_cluster"].predict(predicted_state).astype(np.int32)

            residual = artifacts["residual_hidden"]
            residual_before = hidden_before - residual["macro_to_hidden_before"].predict(macro).astype(np.float32)
            residual_after = hidden_after - residual["macro_to_hidden_after"].predict(macro_next).astype(np.float32)
            output["residual_hidden"]["pred_cur"][position:end] = np.clip(
                residual["probe_cur"].predict(residual_before), -1.0, 1.0
            ).astype(np.float32)
            output["residual_hidden"]["pred_next"][position:end] = np.clip(
                residual["probe_next"].predict(residual_after), -1.0, 1.0
            ).astype(np.float32)
            probability = predict_task(residual.get("task_classifier"), residual_before)
            if probability is not None:
                output["residual_hidden"]["task_prob"][position:end] = probability
            output["residual_hidden"]["rep_cluster"][position:end] = residual["rep_cluster"].predict(residual_before).astype(np.int32)

            previous_hidden = h_after[:, -1:, :].detach()
            position = end

    output["task_y"] = task_y
    return output


def summarize_metrics(
    raw_metrics: Dict[str, float],
    task: Dict[str, float],
    clustering: Dict[str, float],
    representation: str,
    split: str,
) -> Dict[str, float | str]:
    row: Dict[str, float | str] = {"split": split, "representation": representation}
    row.update(raw_metrics)
    row.update(task)
    row.update(clustering)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage-5 macro-sufficiency probes.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-script", type=Path, default=None)
    parser.add_argument("--evaluate-script", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["A_val", "B_confirm"])
    parser.add_argument("--chunk-len", type=int, default=512)
    parser.add_argument("--stage1-root", type=Path, default=None)
    parser.add_argument("--task-feature-name", default=TASK_FEATURE_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-num-threads", type=int, default=0)
    args = parser.parse_args()

    if args.torch_num_threads > 0:
        torch.set_num_threads(int(args.torch_num_threads))

    input_manifest = validate_input_contract(args.input_root)
    train_script = resolve_script(args.train_script, TRAIN_SCRIPT_BASENAME)
    evaluate_script = resolve_script(args.evaluate_script, EVALUATE_SCRIPT_BASENAME)
    train_module = import_module(train_script, "stage5_macro_train_event_ssl_eval")
    evaluate_module = import_module(evaluate_script, "stage5_macro_evaluate_event_ssl_eval")

    artifacts = load_pickle(args.artifacts)
    artifact_meta = dict(artifacts.get("meta", {}))
    if artifact_meta.get("primary_coordinates") != ["M", "Psi"]:
        raise RuntimeError("Macro-sufficiency artifacts do not use the M-Psi primary state.")
    if int(artifact_meta.get("representation_cluster_k", EXPECTED_MACROSTATE_K)) != EXPECTED_MACROSTATE_K:
        raise RuntimeError("Representation clustering must use six clusters.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, train_module)
    stage1_root = evaluate_module.resolve_stage1_root(input_manifest, args.stage1_root)
    partition = evaluate_module.load_fixed_k6_partition(stage1_root, input_manifest)
    convergence_sample_max = int(partition.audit.get("fit_max_rows", 500000))
    convergence_m, convergence_psi, convergence_meta = evaluate_module.convergence_reference(
        args.input_root, convergence_sample_max, args.seed
    )
    task_info = task_feature_index(args.input_root, args.task_feature_name)

    output_root = args.output_root.resolve()
    table_root = output_root / "tables"
    prediction_root = output_root / "predictions"
    metadata_root = output_root / "metadata"
    for directory in (table_root, prediction_root, metadata_root):
        directory.mkdir(parents=True, exist_ok=True)

    save_json(dict(partition.audit), metadata_root / "stage5_macro_sufficiency_fixed_k6_partition_audit.json")
    pd.DataFrame({
        "macrostate": np.arange(partition.k, dtype=int),
        "center_M": partition.centers[:, 0],
        "center_Psi": partition.centers[:, 1],
    }).to_csv(table_root / "stage5_macro_sufficiency_fixed_k6_macrostate_centers.csv", index=False)

    rows: List[Dict[str, float | str]] = []
    manifest = {
        "script": Path(__file__).name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_root": str(args.input_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "artifacts": str(args.artifacts.resolve()),
        "train_script": str(train_script),
        "evaluate_script": str(evaluate_script),
        "primary_coordinates": ["M", "Psi"],
        "stage1_root": str(stage1_root),
        "fixed_k6_partition": dict(partition.audit),
        "representation_clustering": {
            "k": EXPECTED_MACROSTATE_K,
            "role": "diagnostic comparison with frozen empirical mesostate labels",
            "used_for_transition_metrics": False,
        },
        "comparison": {
            "full_hidden": "Linear, task and cluster probes from the full recurrent hidden state.",
            "macro_only": "Current M-Psi readout with task and closure probes from M-Psi features only.",
            "residual_hidden": "Hidden state after removing its fitted dependence on the predicted M-Psi state.",
        },
        "convergence_reference": convergence_meta,
        "task_feature_status": task_info[3],
        "splits": {},
    }

    for split in args.splits:
        arrays = evaluate_module.read_arrays(args.input_root, split)
        representation_outputs = evaluate_split_representations(
            model, artifacts, arrays, device, args.chunk_len, task_info
        )
        target_state = np.asarray(arrays["y"], dtype=np.float32)
        empirical_labels = evaluate_module.labels(partition, target_state)
        task_y = representation_outputs["task_y"]

        for representation in REPRESENTATIONS:
            predicted_current = representation_outputs[representation]["pred_cur"]
            predicted_next = representation_outputs[representation]["pred_next"]
            metrics, matrices = evaluate_module.metrics_for_predictions(
                arrays,
                predicted_current,
                predicted_next,
                partition,
                (convergence_m, convergence_psi),
            )
            metrics["macrostate_partition_verified_against_stage1_fixed_k6"] = 1.0
            cluster = representation_outputs[representation]["rep_cluster"]
            valid = (empirical_labels >= 0) & (cluster >= 0)
            clustering = {
                "representation_nmi_with_empirical_macrostate": float(
                    normalized_mutual_info_score(empirical_labels[valid], cluster[valid])
                ) if valid.sum() > 10 else np.nan,
                "representation_ari_with_empirical_macrostate": float(
                    adjusted_rand_score(empirical_labels[valid], cluster[valid])
                ) if valid.sum() > 10 else np.nan,
            }
            task = task_metrics(task_y, representation_outputs[representation]["task_prob"], prefix="task")
            rows.append(summarize_metrics(metrics, task, clustering, representation, split))

            prediction_table = pd.DataFrame({
                "split": split,
                "representation": representation,
                "user_id": np.asarray(arrays["user_id"], dtype=np.int64),
                "bundle_step_index": np.asarray(arrays["step"], dtype=np.int64),
                "M": target_state[:, 0],
                "Psi": target_state[:, 1],
                "target_M_next": np.asarray(arrays["y_next"][:, 0], dtype=np.float32),
                "target_Psi_next": np.asarray(arrays["y_next"][:, 1], dtype=np.float32),
                "pred_M": predicted_current[:, 0],
                "pred_Psi": predicted_current[:, 1],
                "pred_next_M": predicted_next[:, 0],
                "pred_next_Psi": predicted_next[:, 1],
                "task_y": task_y,
                "task_prob": representation_outputs[representation]["task_prob"],
                "rep_cluster": cluster,
                "empirical_macro_label": empirical_labels,
            })
            write_table(
                prediction_table,
                prediction_root / f"stage5_macro_sufficiency_predictions_{representation}_{split}",
            )
            np.savez_compressed(
                table_root / f"stage5_macro_sufficiency_transition_matrices_{representation}_{split}.npz",
                **matrices,
            )

        split_table = pd.DataFrame([row for row in rows if row["split"] == split])
        split_table.to_csv(table_root / f"stage5_macro_sufficiency_metrics_{split}.csv", index=False)
        manifest["splits"][split] = {
            "rows": int(arrays["n"]),
            "users": int(pd.Series(np.asarray(arrays["user_id"])).nunique()),
        }

    pd.DataFrame(rows).to_csv(table_root / "stage5_macro_sufficiency_metrics_all_splits.csv", index=False)
    save_json(manifest, metadata_root / "stage5_macro_sufficiency_evaluation_manifest.json")
    print(f"[Stage5 macro-sufficiency eval] wrote {table_root / 'stage5_macro_sufficiency_metrics_all_splits.csv'}")


if __name__ == "__main__":
    main()
