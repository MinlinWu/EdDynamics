#!/usr/bin/env python3
from __future__ import annotations

"""Prepare, train and evaluate the within-user time-shuffle control."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

PREPARE_SCRIPT_BASENAME = "prepare_event_ssl_inputs.py"
TRAIN_SCRIPT_BASENAME = "train_event_ssl.py"
EVALUATE_SCRIPT_BASENAME = "evaluate_event_ssl_structure.py"
EXPECTED_MACROSTATE_K = 6


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_module(path: Path, module_name: str):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_script(explicit: Optional[Path], basename: str) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return Path(__file__).resolve().with_name(basename)


def json_safe(obj: Any) -> Any:
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
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(obj), handle, indent=2, ensure_ascii=False, allow_nan=False)


def read_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation metrics not found: {path}")
    return pd.read_csv(path, low_memory=False)


def assert_fixed_k6_outputs(output_root: Path, splits: Sequence[str]) -> Dict[str, object]:
    metrics_path = output_root / "tables" / "stage4_event_ssl_structural_metrics_all_splits.csv"
    audit_path = output_root / "metadata" / "stage4_event_ssl_fixed_k6_partition_audit.json"
    metrics = read_metrics(metrics_path)
    required = {
        "split",
        "macrostate_partition_verified_against_stage1_fixed_k6",
        "macrostate_k_fixed_a_priori",
        "macrostate_k",
    }
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise RuntimeError(f"Evaluation metrics are missing fixed-K fields: {missing}")
    requested = {str(split) for split in splits}
    selected = metrics[metrics["split"].astype(str).isin(requested)].copy()
    if set(selected["split"].astype(str)) != requested:
        raise RuntimeError("Evaluation omitted one or more requested splits.")
    verified = pd.to_numeric(
        selected["macrostate_partition_verified_against_stage1_fixed_k6"],
        errors="coerce",
    )
    fixed = pd.to_numeric(selected["macrostate_k_fixed_a_priori"], errors="coerce")
    macro_k = pd.to_numeric(selected["macrostate_k"], errors="coerce")
    if not bool((verified == 1.0).all()):
        raise RuntimeError("At least one evaluation row failed the fixed-K partition audit.")
    if not bool((fixed == 1.0).all()) or not bool((macro_k == EXPECTED_MACROSTATE_K).all()):
        raise RuntimeError("At least one evaluation row changed the fixed K=6 contract.")
    if not audit_path.exists():
        raise FileNotFoundError(f"Fixed-K partition audit not found: {audit_path}")
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    if audit.get("verified") is not True:
        raise RuntimeError("Partition audit is not verified.")
    if int(audit.get("macrostate_k", -1)) != EXPECTED_MACROSTATE_K:
        raise RuntimeError("Partition audit does not use K=6.")
    if audit.get("kmeans_refit") is not False or audit.get("macrostate_k_selected") is not False:
        raise RuntimeError("Evaluation refit KMeans or selected K.")
    return {
        "verified": True,
        "macrostate_k": EXPECTED_MACROSTATE_K,
        "metrics_path": str(metrics_path.resolve()),
        "audit_path": str(audit_path.resolve()),
        "splits": sorted(requested),
    }


def time_shuffle_frame(df: pd.DataFrame, prepare_module: Any, seed: int, split: str) -> pd.DataFrame:
    offset = {"A_train": 101, "A_val": 202, "B_confirm": 303}.get(str(split), 404)
    rng = np.random.default_rng(int(seed) + offset)
    pieces = []
    dropped_last = 0
    for _, group in df.groupby("user_id", sort=False):
        group = group.sort_values("bundle_step_index", kind="mergesort").reset_index(drop=True)
        length = len(group)
        if length < 2:
            continue
        shuffled = group.iloc[rng.permutation(length)].copy().reset_index(drop=True)
        shuffled["bundle_step_index"] = np.arange(length, dtype=np.int64)
        shuffled[prepare_module.COL_M_NEXT] = shuffled[prepare_module.COL_M].shift(-1)
        shuffled[prepare_module.COL_PSI_NEXT] = shuffled[prepare_module.COL_PSI].shift(-1)
        pieces.append(shuffled.iloc[:-1].copy())
        dropped_last += 1
    if not pieces:
        raise RuntimeError(f"Time shuffle produced no rows for split={split}.")
    output = pd.concat(pieces, ignore_index=True, sort=False)
    output = output.sort_values(["user_id", "bundle_step_index"], kind="mergesort").reset_index(drop=True)
    output.attrs["dropped_last_rows"] = int(dropped_last)
    return output


def prepare_main(args: argparse.Namespace) -> None:
    prepare_script = resolve_script(args.prepare_script, PREPARE_SCRIPT_BASENAME)
    prepare_module = import_module(prepare_script, "event_ssl_prepare_for_time_shuffle")
    required = {
        "fixed_k6_contract",
        "select_existing_columns",
        "load_split_frame",
        "build_numeric_matrix",
        "robust_mean_std",
        "build_split_arrays",
        "SPLITS",
        "COL_M",
        "COL_PSI",
        "COL_M_NEXT",
        "COL_PSI_NEXT",
    }
    missing = sorted(name for name in required if not hasattr(prepare_module, name))
    if missing:
        raise RuntimeError(f"Input-preparation module is missing: {missing}")

    data_root = args.output_root / "prepared_inputs"
    metadata_root = data_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    kmeans_contract = prepare_module.fixed_k6_contract(args.stage1_root)
    selected = prepare_module.select_existing_columns(args.stage1_root)
    numeric_columns = list(selected["numeric"])
    categorical_columns = list(selected["categorical"])
    print(f"[time-shuffle prepare] numeric primitive columns: {len(numeric_columns)}")
    print(f"[time-shuffle prepare] categorical primitive columns: {len(categorical_columns)}")

    train_original = prepare_module.load_split_frame(
        args.stage1_root,
        "A_train",
        selected["read_columns"],
        args.max_users_per_split,
        args.seed,
    )
    train_numeric, numeric_feature_names = prepare_module.build_numeric_matrix(
        train_original,
        numeric_columns,
    )
    mean, std, count = prepare_module.robust_mean_std(train_numeric)
    save_json(
        {
            "fitted_on": "original ordered A_train before within-user time shuffle",
            "numeric_feature_names": list(numeric_feature_names),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "finite_count": count.tolist(),
        },
        metadata_root / "normalizer.json",
    )

    split_summaries: Dict[str, object] = {}
    shuffle_summaries: Dict[str, object] = {}
    for split in prepare_module.SPLITS:
        original = train_original if split == "A_train" else prepare_module.load_split_frame(
            args.stage1_root,
            split,
            selected["read_columns"],
            args.max_users_per_split,
            args.seed,
        )
        shuffled = time_shuffle_frame(original, prepare_module, args.seed, split)
        shuffle_summaries[split] = {
            "original_rows": int(len(original)),
            "shuffled_rows_written": int(len(shuffled)),
            "dropped_final_interval_per_user_sequence": int(shuffled.attrs.get("dropped_last_rows", 0)),
            "step_policy": "synthetic contiguous bundle_step_index in shuffled within-user order",
            "next_target_policy": "next M/Psi equals the next current M/Psi under shuffled order",
        }
        split_summaries[split] = prepare_module.build_split_arrays(
            df=shuffled,
            split_dir=data_root / split,
            numeric_cols=numeric_columns,
            numeric_feature_names=numeric_feature_names,
            categorical_cols=categorical_columns,
            hash_buckets=args.hash_buckets,
            mean=mean,
            std=std,
        )
        print(f"[time-shuffle prepare] {split}: {split_summaries[split]}")
        if split != "A_train":
            del original
        del shuffled
    del train_original

    save_json(
        {
            "script": Path(__file__).name,
            "control_type": "within_user_time_shuffle",
            "main_prepare_script": str(prepare_script),
            "main_prepare_script_sha256": file_sha256(prepare_script),
            "stage1_root": str(args.stage1_root.resolve()),
            "stage1_fixed_k6_contract": kmeans_contract,
            "output_root": str(args.output_root.resolve()),
            "data_root": str(data_root.resolve()),
            "primary_coordinates": ["M", "Psi"],
            "excluded_coordinate_policy": "V/maturity/noise and all coarse-state fields are excluded from features, targets and outputs",
            "targets": {
                "current": [prepare_module.COL_M, prepare_module.COL_PSI],
                "next": [prepare_module.COL_M_NEXT, prepare_module.COL_PSI_NEXT],
            },
            "numeric_input_source_columns": numeric_columns,
            "numeric_feature_names_after_expansion": list(numeric_feature_names),
            "categorical_input_source_columns": categorical_columns,
            "categorical_hash_buckets": int(args.hash_buckets),
            "forbidden_feature_tokens": prepare_module.FORBIDDEN_FEATURE_TOKENS,
            "normalization_fit_scope": "original ordered A_train before shuffle",
            "sequence_boundary_policy": "user_id change under synthetic contiguous shuffled step index",
            "control_transformation": "within-user random permutation of submitted-bundle intervals",
            "intended_evaluation": "train on shuffled inputs and evaluate the checkpoint on original ordered inputs",
            "confirmation_policy": "B_confirm shuffled arrays are not used for model selection; the main control evaluates ordered B_confirm output-only",
            "split_summaries": split_summaries,
            "shuffle_summaries": shuffle_summaries,
            "smoke_test_max_users_per_split": int(args.max_users_per_split),
        },
        metadata_root / "stage4_input_manifest.json",
    )
    print(f"[time-shuffle prepare] wrote manifest: {metadata_root / 'stage4_input_manifest.json'}")


def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-script", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--min-seq-len", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=320)
    parser.add_argument("--input-dim", type=int, default=224)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--future-steps", type=str, default="1,2,4")
    parser.add_argument("--lambda-future", type=float, default=1.0)
    parser.add_argument("--lambda-state", type=float, default=0.5)
    parser.add_argument("--lambda-closure", type=float, default=0.5)
    parser.add_argument("--delta-scale", type=float, default=0.50)
    parser.add_argument("--categorical-emb-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--allow-truncated-supervision", action="store_true")


def train_main(args: argparse.Namespace) -> None:
    train_script = resolve_script(args.train_script, TRAIN_SCRIPT_BASENAME)
    command = [
        sys.executable,
        str(train_script),
        "--input-root",
        str(args.input_root),
        "--output-root",
        str(args.output_root),
        "--model-kind",
        "predictive_state",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seq-len",
        str(args.seq_len),
        "--stride",
        str(args.stride),
        "--min-seq-len",
        str(args.min_seq_len),
        "--warmup-steps",
        str(args.warmup_steps),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--hidden-dim",
        str(args.hidden_dim),
        "--input-dim",
        str(args.input_dim),
        "--num-layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--num-workers",
        str(args.num_workers),
        "--future-steps",
        str(args.future_steps),
        "--lambda-future",
        str(args.lambda_future),
        "--lambda-state",
        str(args.lambda_state),
        "--lambda-closure",
        str(args.lambda_closure),
        "--delta-scale",
        str(args.delta_scale),
        "--categorical-emb-dim",
        str(args.categorical_emb_dim),
        "--seed",
        str(args.seed),
        "--amp-dtype",
        args.amp_dtype,
    ]
    if args.compile:
        command.append("--compile")
    if args.torch_num_threads > 0:
        command.extend(["--torch-num-threads", str(args.torch_num_threads)])
    if args.allow_truncated_supervision:
        command.append("--allow-truncated-supervision")
    print("[time-shuffle train] running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    save_json(
        {
            "control_type": "within_user_time_shuffle",
            "main_train_script": str(train_script),
            "main_train_script_sha256": file_sha256(train_script),
            "input_root": str(args.input_root.resolve()),
            "output_root": str(args.output_root.resolve()),
            "model_kind": "predictive_state",
            "control_change": "training order and next-M/Psi targets follow the within-user shuffled order",
        },
        args.output_root / "time_shuffle_control_training_wrapper_manifest.json",
    )


def evaluate_main(args: argparse.Namespace) -> None:
    evaluate_script = resolve_script(args.evaluate_script, EVALUATE_SCRIPT_BASENAME)
    train_script = resolve_script(args.train_script, TRAIN_SCRIPT_BASENAME)
    command = [
        sys.executable,
        str(evaluate_script),
        "--input-root",
        str(args.input_root),
        "--checkpoint",
        str(args.checkpoint),
        "--output-root",
        str(args.output_root),
        "--train-script",
        str(train_script),
        "--splits",
        *args.splits,
        "--chunk-len",
        str(args.chunk_len),
        "--seed",
        str(args.seed),
    ]
    if args.stage1_root is not None:
        command.extend(["--stage1-root", str(args.stage1_root)])
    if args.torch_num_threads > 0:
        command.extend(["--torch-num-threads", str(args.torch_num_threads)])
    print("[time-shuffle evaluate] running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    partition_audit = assert_fixed_k6_outputs(args.output_root, args.splits)
    save_json(
        {
            "control_type": "within_user_time_shuffle",
            "main_evaluate_script": str(evaluate_script),
            "main_evaluate_script_sha256": file_sha256(evaluate_script),
            "main_train_script": str(train_script),
            "main_train_script_sha256": file_sha256(train_script),
            "checkpoint": str(args.checkpoint.resolve()),
            "evaluation_input_root": str(args.input_root.resolve()),
            "training_input_root": str(args.train_input_root.resolve()) if args.train_input_root else None,
            "primary_coordinates": ["M", "Psi"],
            "main_comparison": "checkpoint trained on shuffled sequences and evaluated on ordered empirical sequences",
            "transition_partition_contract": partition_audit,
            "guardrails": {
                "kmeans_refit": False,
                "macrostate_k_selected": False,
                "macrostate_k": EXPECTED_MACROSTATE_K,
                "B_confirm_used_for_update": False,
            },
        },
        args.output_root / "metadata" / "stage4_time_shuffle_control_manifest.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Within-user time-shuffle Event-SSL control.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--prepare-script", type=Path, default=None)
    prepare_parser.add_argument(
        "--stage1-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/stage1"),
    )
    prepare_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/datasets/KT4/outputs_KT4/stage4_event_ssl_time_shuffle_control"),
    )
    prepare_parser.add_argument("--hash-buckets", type=int, default=32768)
    prepare_parser.add_argument("--max-users-per-split", type=int, default=0)
    prepare_parser.add_argument("--seed", type=int, default=42)

    train_parser = subparsers.add_parser("train")
    add_train_arguments(train_parser)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--input-root", type=Path, required=True)
    evaluate_parser.add_argument("--train-input-root", type=Path, default=None)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--output-root", type=Path, required=True)
    evaluate_parser.add_argument("--train-script", type=Path, default=None)
    evaluate_parser.add_argument("--evaluate-script", type=Path, default=None)
    evaluate_parser.add_argument("--splits", nargs="+", default=["A_val", "B_confirm"])
    evaluate_parser.add_argument("--chunk-len", type=int, default=512)
    evaluate_parser.add_argument("--stage1-root", type=Path, default=None)
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.add_argument("--torch-num-threads", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare_main(args)
    elif args.command == "train":
        train_main(args)
    elif args.command == "evaluate":
        evaluate_main(args)
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
