#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate Stage-5 geometry of frozen predictive-state Event-SSL representations."""

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
from sklearn.neighbors import NearestNeighbors

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


def predict_hgb(regressors, x: np.ndarray) -> np.ndarray:
    return np.column_stack([regressor.predict(x) for regressor in regressors]).astype(np.float32)


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


def twonn_dimension(x: np.ndarray, max_rows: int = 10000, seed: int = 42) -> float:
    values = np.asarray(x, dtype=np.float32)
    if len(values) < 50:
        return float("nan")
    if max_rows > 0 and len(values) > max_rows:
        rng = np.random.default_rng(seed)
        values = values[np.sort(rng.choice(len(values), size=max_rows, replace=False))]
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True) + 1e-6
    standardized = (values - mean) / std
    neighbors = NearestNeighbors(n_neighbors=3, metric="euclidean")
    neighbors.fit(standardized)
    distances, _ = neighbors.kneighbors(standardized, n_neighbors=3)
    first = np.maximum(distances[:, 1], EPS)
    second = np.maximum(distances[:, 2], EPS)
    ratio = second / first
    log_ratio = np.log(ratio[np.isfinite(ratio) & (ratio > 1)])
    if log_ratio.size < 10:
        return float("nan")
    return float(1.0 / np.mean(log_ratio))


@torch.inference_mode()
def evaluate_representations(
    model,
    artifacts: dict,
    arrays: dict,
    device: torch.device,
    chunk_len: int,
    sample_max_rows: int,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    n_rows = int(arrays["n"])
    predictions = {
        "model_readout_cur": np.zeros((n_rows, 2), dtype=np.float32),
        "model_readout_next": np.zeros((n_rows, 2), dtype=np.float32),
        "linear_hidden_cur": np.zeros((n_rows, 2), dtype=np.float32),
        "linear_hidden_next": np.zeros((n_rows, 2), dtype=np.float32),
        "residual_hidden_cur": np.zeros((n_rows, 2), dtype=np.float32),
        "residual_hidden_next": np.zeros((n_rows, 2), dtype=np.float32),
        "nonlinear_hidden_cur": np.zeros((n_rows, 2), dtype=np.float32),
    }

    rng = np.random.default_rng(seed)
    take_n = n_rows if sample_max_rows <= 0 or sample_max_rows >= n_rows else int(sample_max_rows)
    selected = np.zeros(n_rows, dtype=bool)
    if take_n < n_rows:
        selected[rng.choice(n_rows, size=take_n, replace=False)] = True
    else:
        selected[:] = True

    sample = {"h_before": [], "y": [], "pca_scores": []}
    autocast_enabled = device.type == "cuda"

    for sequence_index in range(len(arrays["offsets"]) - 1):
        start = int(arrays["offsets"][sequence_index])
        stop = int(arrays["offsets"][sequence_index + 1])
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

            hidden_before = h_before.squeeze(0).float().cpu().numpy().astype(np.float32)
            hidden_after = h_after.squeeze(0).float().cpu().numpy().astype(np.float32)
            predicted_state = np.clip(state.squeeze(0).float().cpu().numpy().astype(np.float32), -1.0, 1.0)
            predicted_next_state = np.clip(next_state.squeeze(0).float().cpu().numpy().astype(np.float32), -1.0, 1.0)

            predictions["model_readout_cur"][position:end] = predicted_state
            predictions["model_readout_next"][position:end] = predicted_next_state
            predictions["linear_hidden_cur"][position:end] = np.clip(
                artifacts["linear_cur"].predict(hidden_before), -1.0, 1.0
            ).astype(np.float32)
            predictions["linear_hidden_next"][position:end] = np.clip(
                artifacts["linear_next"].predict(hidden_after), -1.0, 1.0
            ).astype(np.float32)

            macro = macro_features(predicted_state)
            macro_next = macro_features(predicted_next_state)
            residual_before = hidden_before - artifacts["macro_to_hidden_before"].predict(macro).astype(np.float32)
            residual_after = hidden_after - artifacts["macro_to_hidden_after"].predict(macro_next).astype(np.float32)
            predictions["residual_hidden_cur"][position:end] = np.clip(
                artifacts["residual_cur"].predict(residual_before), -1.0, 1.0
            ).astype(np.float32)
            predictions["residual_hidden_next"][position:end] = np.clip(
                artifacts["residual_next"].predict(residual_after), -1.0, 1.0
            ).astype(np.float32)

            standardized_hidden = artifacts["hidden_scaler"].transform(hidden_before)
            scores = artifacts["pca"].transform(standardized_hidden).astype(np.float32)
            predictions["nonlinear_hidden_cur"][position:end] = np.clip(
                predict_hgb(artifacts["nonlinear_cur"], scores), -1.0, 1.0
            ).astype(np.float32)

            if take.any():
                sample["h_before"].append(hidden_before[take])
                sample["y"].append(np.asarray(arrays["y"][position:end], dtype=np.float32)[take])
                sample["pca_scores"].append(scores[take])

            previous_hidden = h_after[:, -1:, :].detach()
            position = end

    if not sample["h_before"]:
        raise RuntimeError("No geometry sample was collected.")

    sample_output = {
        "h_before": np.concatenate(sample["h_before"], axis=0),
        "y": np.concatenate(sample["y"], axis=0),
        "pca_scores": np.concatenate(sample["pca_scores"], axis=0),
    }
    return predictions, sample_output


def coordinate_metrics(predicted: np.ndarray, target: np.ndarray, representation: str) -> Dict[str, float | str]:
    output: Dict[str, float | str] = {"representation": representation}
    for index, name in enumerate(("M", "Psi")):
        error = predicted[:, index] - target[:, index]
        output[f"coordinate_rmse_{name}"] = float(np.sqrt(np.nanmean(error * error)))
        output[f"coordinate_corr_{name}"] = correlation(predicted[:, index], target[:, index])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage-5 representation geometry.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-script", type=Path, default=None)
    parser.add_argument("--evaluate-script", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["A_val", "B_confirm"])
    parser.add_argument("--chunk-len", type=int, default=512)
    parser.add_argument("--sample-max-rows", type=int, default=250000)
    parser.add_argument("--stage1-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-num-threads", type=int, default=0)
    args = parser.parse_args()

    if args.torch_num_threads > 0:
        torch.set_num_threads(int(args.torch_num_threads))

    input_manifest = validate_input_contract(args.input_root)
    train_script = resolve_script(args.train_script, TRAIN_SCRIPT_BASENAME)
    evaluate_script = resolve_script(args.evaluate_script, EVALUATE_SCRIPT_BASENAME)
    train_module = import_module(train_script, "stage5_geometry_train_event_ssl_eval")
    evaluate_module = import_module(evaluate_script, "stage5_geometry_evaluate_event_ssl_eval")

    artifacts = load_pickle(args.artifacts)
    artifact_meta = dict(artifacts.get("meta", {}))
    if artifact_meta.get("primary_coordinates") != ["M", "Psi"]:
        raise RuntimeError("Representation-geometry artifacts do not use the M-Psi primary state.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, train_module)
    stage1_root = evaluate_module.resolve_stage1_root(input_manifest, args.stage1_root)
    partition = evaluate_module.load_fixed_k6_partition(stage1_root, input_manifest)
    convergence_sample_max = int(partition.audit.get("fit_max_rows", 500000))
    convergence_m, convergence_psi, convergence_meta = evaluate_module.convergence_reference(
        args.input_root, convergence_sample_max, args.seed
    )

    output_root = args.output_root.resolve()
    table_root = output_root / "tables"
    prediction_root = output_root / "predictions"
    metadata_root = output_root / "metadata"
    for directory in (table_root, prediction_root, metadata_root):
        directory.mkdir(parents=True, exist_ok=True)

    save_json(dict(partition.audit), metadata_root / "stage5_representation_geometry_fixed_k6_partition_audit.json")
    pd.DataFrame({
        "macrostate": np.arange(partition.k, dtype=int),
        "center_M": partition.centers[:, 0],
        "center_Psi": partition.centers[:, 1],
    }).to_csv(table_root / "stage5_representation_geometry_fixed_k6_macrostate_centers.csv", index=False)

    all_rows: List[Dict[str, float | str]] = []
    pc_rows: List[Dict[str, float | str]] = []
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
        "convergence_reference": convergence_meta,
        "splits": {},
    }

    for split in args.splits:
        arrays = evaluate_module.read_arrays(args.input_root, split)
        predictions, sample = evaluate_representations(
            model,
            artifacts,
            arrays,
            device,
            args.chunk_len,
            args.sample_max_rows,
            args.seed,
        )
        target_state = np.asarray(arrays["y"], dtype=np.float32)
        target_next_state = np.asarray(arrays["y_next"], dtype=np.float32)

        structural_sources = {
            "model_readout": (predictions["model_readout_cur"], predictions["model_readout_next"]),
            "linear_hidden": (predictions["linear_hidden_cur"], predictions["linear_hidden_next"]),
            "residual_hidden": (predictions["residual_hidden_cur"], predictions["residual_hidden_next"]),
        }
        split_rows: List[Dict[str, float | str]] = []
        for representation, (predicted_current, predicted_next) in structural_sources.items():
            metrics, matrices = evaluate_module.metrics_for_predictions(
                arrays,
                predicted_current,
                predicted_next,
                partition,
                (convergence_m, convergence_psi),
            )
            metrics["macrostate_partition_verified_against_stage1_fixed_k6"] = 1.0
            row: Dict[str, float | str] = {"split": split, "representation": representation}
            row.update(metrics)
            all_rows.append(row)
            split_rows.append(row)

            prediction_table = pd.DataFrame({
                "split": split,
                "representation": representation,
                "user_id": np.asarray(arrays["user_id"], dtype=np.int64),
                "bundle_step_index": np.asarray(arrays["step"], dtype=np.int64),
                "M": target_state[:, 0],
                "Psi": target_state[:, 1],
                "target_M_next": target_next_state[:, 0],
                "target_Psi_next": target_next_state[:, 1],
                "pred_M": predicted_current[:, 0],
                "pred_Psi": predicted_current[:, 1],
                "pred_next_M": predicted_next[:, 0],
                "pred_next_Psi": predicted_next[:, 1],
            })
            write_table(
                prediction_table,
                prediction_root / f"stage5_representation_geometry_predictions_{representation}_{split}",
            )
            np.savez_compressed(
                table_root / f"stage5_representation_geometry_transition_matrices_{representation}_{split}.npz",
                **matrices,
            )

        nonlinear_row: Dict[str, float | str] = {"split": split}
        nonlinear_row.update(coordinate_metrics(predictions["nonlinear_hidden_cur"], target_state, "nonlinear_hidden"))
        nonlinear_row["macrostate_partition_verified_against_stage1_fixed_k6"] = 1.0
        all_rows.append(nonlinear_row)
        split_rows.append(nonlinear_row)

        scores = sample["pca_scores"]
        sample_targets = sample["y"]
        for index in range(min(16, scores.shape[1])):
            corr_m = correlation(scores[:, index], sample_targets[:, 0])
            corr_psi = correlation(scores[:, index], sample_targets[:, 1])
            pc_rows.append({
                "split": split,
                "component": f"PC{index + 1}",
                "corr_M": corr_m,
                "corr_Psi": corr_psi,
                "abs_corr_M": abs(corr_m),
                "abs_corr_Psi": abs(corr_psi),
                "explained_variance_ratio_train": float(artifacts["pca"].explained_variance_ratio_[index]),
            })

        standardized_hidden = artifacts["hidden_scaler"].transform(sample["h_before"])
        hidden_canonical, macro_canonical = artifacts["cca"].transform(
            standardized_hidden,
            artifacts["macro_scaler"].transform(sample_targets),
        )
        canonical_correlations = [
            correlation(hidden_canonical[:, index], macro_canonical[:, index])
            for index in range(min(2, hidden_canonical.shape[1]))
        ]
        twonn = twonn_dimension(sample["h_before"], seed=args.seed)

        for row in split_rows:
            row["sample_rows_geometry"] = int(len(sample_targets))
            row["twonn_dimension"] = twonn
            row["participation_ratio_train"] = float(artifacts["summary"].get("participation_ratio", np.nan))
            row["effective_rank_train"] = float(artifacts["summary"].get("effective_rank", np.nan))
            row["cca_corr_1"] = float(canonical_correlations[0]) if canonical_correlations else np.nan
            row["cca_corr_2"] = float(canonical_correlations[1]) if len(canonical_correlations) > 1 else np.nan

        manifest["splits"][split] = {
            "rows": int(arrays["n"]),
            "geometry_sample_rows": int(len(sample_targets)),
        }

    metrics_table = pd.DataFrame(all_rows)
    pc_table = pd.DataFrame(pc_rows)
    gain_rows: List[Dict[str, float | str]] = []
    for split in args.splits:
        linear = metrics_table[
            (metrics_table["split"] == split) & (metrics_table["representation"] == "linear_hidden")
        ]
        nonlinear = metrics_table[
            (metrics_table["split"] == split) & (metrics_table["representation"] == "nonlinear_hidden")
        ]
        if linear.empty or nonlinear.empty:
            continue
        linear_row = linear.iloc[0]
        nonlinear_row = nonlinear.iloc[0]
        row: Dict[str, float | str] = {"split": split}
        for coordinate in ("M", "Psi"):
            row[f"nonlinear_gain_corr_{coordinate}"] = float(
                nonlinear_row.get(f"coordinate_corr_{coordinate}", np.nan)
            ) - float(linear_row.get(f"coordinate_corr_{coordinate}", np.nan))
            row[f"nonlinear_gain_rmse_reduction_{coordinate}"] = float(
                linear_row.get(f"coordinate_rmse_{coordinate}", np.nan)
            ) - float(nonlinear_row.get(f"coordinate_rmse_{coordinate}", np.nan))
        gain_rows.append(row)

    metrics_table.to_csv(table_root / "stage5_representation_geometry_metrics_all_splits.csv", index=False)
    pc_table.to_csv(table_root / "stage5_representation_geometry_pc_macro_correlations.csv", index=False)
    pd.DataFrame(gain_rows).to_csv(table_root / "stage5_representation_geometry_nonlinear_probe_gain.csv", index=False)
    save_json(manifest, metadata_root / "stage5_representation_geometry_evaluation_manifest.json")
    print(f"[Stage5 representation geometry eval] wrote {table_root / 'stage5_representation_geometry_metrics_all_splits.csv'}")


if __name__ == "__main__":
    main()
