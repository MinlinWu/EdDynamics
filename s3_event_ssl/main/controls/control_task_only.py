#!/usr/bin/env python3
from __future__ import annotations

"""Train and evaluate the task-only Event-SSL control."""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

EPS = 1e-12
TASK_FEATURE_NAME = "current_accuracy_diagnostic_only"
TRAIN_SCRIPT_BASENAME = "train_event_ssl.py"
EVALUATE_SCRIPT_BASENAME = "evaluate_event_ssl_structure.py"
EXPECTED_MACROSTATE_K = 6


@dataclass
class TaskOnlyConfig:
    input_root: str
    output_root: str
    model_kind: str
    seed: int
    seq_len: int
    stride: int
    min_seq_len: int
    warmup_steps: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    hidden_dim: int
    input_dim: int
    num_layers: int
    dropout: float
    num_workers: int
    categorical_emb_dim: int
    future_steps: List[int]
    delta_scale: float
    amp_dtype: str
    use_compile: bool
    supervise_truncated_windows: bool
    task_feature_name: str


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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def write_table(df: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = base.with_suffix(".parquet")
        df.to_parquet(path, index=False)
        return path
    except Exception:
        path = base.with_suffix(".csv.gz")
        df.to_csv(path, index=False, compression="gzip")
        return path


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    valid = np.isfinite(aa) & np.isfinite(bb)
    if valid.sum() < 3:
        return float("nan")
    aa = aa[valid] - float(np.mean(aa[valid]))
    bb = bb[valid] - float(np.mean(bb[valid]))
    denominator = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denominator) if denominator > EPS else float("nan")


def validate_main_modules(train_module: Any, eval_module: Optional[Any] = None) -> None:
    train_required = {"PredictiveStateEventSSL", "validate_input_contract"}
    missing_train = sorted(name for name in train_required if not hasattr(train_module, name))
    if missing_train:
        raise RuntimeError(f"Training module is missing: {missing_train}")
    if eval_module is not None:
        eval_required = {
            "read_arrays",
            "resolve_stage1_root",
            "load_fixed_k6_partition",
            "convergence_reference",
            "metrics_for_predictions",
        }
        missing_eval = sorted(name for name in eval_required if not hasattr(eval_module, name))
        if missing_eval:
            raise RuntimeError(f"Evaluation module is missing: {missing_eval}")


def locate_task_feature(input_root: Path, task_feature_name: str) -> Tuple[int, float, float]:
    manifest = load_json(input_root / "metadata" / "stage4_input_manifest.json")
    normalizer = load_json(input_root / "metadata" / "normalizer.json")
    if manifest.get("primary_coordinates") != ["M", "Psi"]:
        raise RuntimeError("Input manifest primary coordinates are not ['M', 'Psi'].")
    names = list(normalizer.get("numeric_feature_names", []))
    if task_feature_name not in names:
        raise RuntimeError(f"Task feature {task_feature_name!r} is absent from the prepared inputs.")
    index = int(names.index(task_feature_name))
    return index, float(normalizer["mean"][index]), float(normalizer["std"][index])


class TaskOnlySequenceDataset(Dataset):
    def __init__(
        self,
        input_root: Path,
        split: str,
        seq_len: int,
        stride: int,
        min_seq_len: int,
        task_feature_name: str,
        supervise_truncated_windows: bool,
    ) -> None:
        self.input_root = Path(input_root)
        self.split_dir = self.input_root / split
        manifest = load_json(self.input_root / "metadata" / "stage4_input_manifest.json")
        summary = manifest["split_summaries"][split]
        n_rows = int(summary["rows"])
        self.n_num = int(summary["numeric_shape"][1])
        self.n_cat = int(summary["categorical_shape"][1])
        self.x_num = np.memmap(
            self.split_dir / "x_num.float32.mmap",
            mode="r",
            dtype=np.float32,
            shape=(n_rows, self.n_num),
        )
        self.x_cat = np.memmap(
            self.split_dir / "x_cat.int64.mmap",
            mode="r",
            dtype=np.int64,
            shape=(n_rows, self.n_cat),
        )
        self.offsets = np.load(self.split_dir / "sequence_offsets.npy")
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.min_seq_len = int(min_seq_len)
        self.supervise_truncated_windows = bool(supervise_truncated_windows)
        self.task_idx, self.task_mean, self.task_std = locate_task_feature(
            self.input_root,
            task_feature_name,
        )
        self.windows: List[Tuple[int, int, int]] = []
        for index in range(len(self.offsets) - 1):
            start = int(self.offsets[index])
            stop = int(self.offsets[index + 1])
            length = stop - start
            if length < self.min_seq_len:
                continue
            first_stop = min(start + self.seq_len, stop)
            self.windows.append((start, first_stop, 1))
            if first_stop >= stop:
                continue
            position = start + max(1, self.stride)
            while position + self.min_seq_len <= stop:
                window_stop = min(position + self.seq_len, stop)
                supervise = 1 if self.supervise_truncated_windows else 0
                self.windows.append((position, window_stop, supervise))
                if window_stop == stop:
                    break
                position += max(1, self.stride)
        if not self.windows:
            raise RuntimeError(f"No sequence windows for split={split}.")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        start, stop, supervise = self.windows[index]
        length = stop - start
        x_num = np.zeros((self.seq_len, self.n_num), dtype=np.float32)
        x_cat = np.zeros((self.seq_len, self.n_cat), dtype=np.int64)
        task_y = np.zeros((self.seq_len,), dtype=np.float32)
        supervised_mask = np.zeros((self.seq_len,), dtype=np.float32)
        x_num[:length] = self.x_num[start:stop]
        if self.n_cat:
            x_cat[:length] = self.x_cat[start:stop]
        raw = x_num[:length, self.task_idx] * np.float32(self.task_std) + np.float32(self.task_mean)
        task_y[:length] = np.clip(raw, 0.0, 1.0)
        if supervise:
            supervised_mask[:length] = 1.0
        return {
            "x_num": torch.from_numpy(x_num),
            "x_cat": torch.from_numpy(x_cat),
            "task_y": torch.from_numpy(task_y),
            "supervised_mask": torch.from_numpy(supervised_mask),
        }


class TaskOnlyEventModel(nn.Module):
    def __init__(self, base_model: nn.Module, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.base = base_model
        head_dim = max(hidden_dim // 2, 8)
        self.task_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedded = self.base.embed_inputs(x_num, x_cat)
        hidden_after, _ = self.base.rnn(embedded)
        zero = torch.zeros_like(hidden_after[:, :1, :])
        hidden_before = torch.cat([zero, hidden_after[:, :-1, :]], dim=1)
        logits = self.task_head(hidden_before).squeeze(-1)
        return {"logits": logits, "h_before": hidden_before, "h_after": hidden_after}


def masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    config: TaskOnlyConfig,
    device: torch.device,
    train: bool,
) -> Dict[str, float]:
    model.train(train)
    total_loss = 0.0
    batches = 0
    effective_rows = 0
    autocast_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16
    started = time.time()
    for batch in loader:
        batch = move_batch(batch, device)
        supervised_mask = batch["supervised_mask"].clone()
        if config.warmup_steps > 0:
            supervised_mask[:, : config.warmup_steps] = 0.0
        if float(supervised_mask.sum().detach().cpu()) <= 0.0:
            continue
        if train:
            if optimizer is None:
                raise RuntimeError("Training requires an optimizer.")
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=autocast_enabled,
            ):
                output = model(batch["x_num"], batch["x_cat"])
                loss = masked_bce_with_logits(
                    output["logits"],
                    batch["task_y"],
                    supervised_mask,
                )
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
        effective_rows += int(supervised_mask.sum().detach().cpu().item())
    return {
        "task_bce": total_loss / max(batches, 1),
        "batches": float(batches),
        "effective_rows": float(effective_rows),
        "seconds": time.time() - started,
    }


def build_task_model(
    train_module: Any,
    config: TaskOnlyConfig,
    n_num: int,
    n_cat: int,
    hash_buckets: int,
    device: torch.device,
) -> nn.Module:
    base = train_module.PredictiveStateEventSSL(
        n_num=n_num,
        n_cat=n_cat,
        hash_buckets=hash_buckets,
        hidden_dim=config.hidden_dim,
        input_dim=config.input_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        categorical_emb_dim=config.categorical_emb_dim,
        future_steps=tuple(config.future_steps),
        delta_scale=config.delta_scale,
    )
    return TaskOnlyEventModel(base, config.hidden_dim, config.dropout).to(device)


def train_main(args: argparse.Namespace) -> None:
    train_script = resolve_script(args.train_script, TRAIN_SCRIPT_BASENAME)
    train_module = import_module(train_script, "event_ssl_train_for_task_only")
    validate_main_modules(train_module)
    if args.torch_num_threads > 0:
        torch.set_num_threads(int(args.torch_num_threads))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    input_manifest = load_json(args.input_root / "metadata" / "stage4_input_manifest.json")
    train_module.validate_input_contract(input_manifest)
    hash_buckets = int(input_manifest["categorical_hash_buckets"])
    supervise_truncated = not args.no_truncated_supervision
    train_dataset = TaskOnlySequenceDataset(
        args.input_root,
        "A_train",
        args.seq_len,
        args.stride,
        args.min_seq_len,
        args.task_feature_name,
        supervise_truncated,
    )
    val_dataset = TaskOnlySequenceDataset(
        args.input_root,
        "A_val",
        args.seq_len,
        args.stride,
        args.min_seq_len,
        args.task_feature_name,
        supervise_truncated,
    )

    config = TaskOnlyConfig(
        input_root=str(args.input_root.resolve()),
        output_root=str(args.output_root.resolve()),
        model_kind="task_only_control",
        seed=args.seed,
        seq_len=args.seq_len,
        stride=args.stride,
        min_seq_len=args.min_seq_len,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        input_dim=args.input_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_workers=args.num_workers,
        categorical_emb_dim=args.categorical_emb_dim,
        future_steps=[int(value) for value in args.future_steps.split(",") if value.strip()],
        delta_scale=args.delta_scale,
        amp_dtype=args.amp_dtype,
        use_compile=bool(args.compile),
        supervise_truncated_windows=bool(supervise_truncated),
        task_feature_name=args.task_feature_name,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "config": asdict(config),
            "main_train_script": str(train_script),
            "main_train_script_sha256": file_sha256(train_script),
            "input_manifest_subset": {
                "primary_coordinates": input_manifest.get("primary_coordinates"),
                "excluded_coordinate_policy": input_manifest.get("excluded_coordinate_policy"),
                "sequence_boundary_policy": input_manifest.get("sequence_boundary_policy"),
                "stage1_fixed_k6_contract": input_manifest.get("stage1_fixed_k6_contract"),
            },
            "control_boundary": {
                "trained_losses": ["task_bce_only"],
                "excluded_losses": [
                    "M/Psi_state_loss",
                    "M/Psi_closure_loss",
                    "future_ssl_loss",
                    "transition_loss",
                    "residence_loss",
                    "convergence_loss",
                ],
                "task_head_input": "pre-interval hidden state",
                "primary_coordinates": ["M", "Psi"],
            },
        },
        args.output_root / "training_manifest.json",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_task_model(
        train_module,
        config,
        train_dataset.n_num,
        train_dataset.n_cat,
        hash_buckets,
        device,
    )
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[dict] = []
    best_val = float("inf")
    best_path = args.output_root / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, config, device, train=True)
        val_metrics = run_epoch(model, val_loader, None, config, device, train=False)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(f"[task-only] epoch={epoch} train={train_metrics} val={val_metrics}", flush=True)
        if val_metrics["task_bce"] < best_val:
            best_val = float(val_metrics["task_bce"])
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(
                {
                    "config": asdict(config),
                    "model_shapes": {
                        "n_num": train_dataset.n_num,
                        "n_cat": train_dataset.n_cat,
                        "hash_buckets": hash_buckets,
                    },
                    "base_model_state_dict": raw_model.base.state_dict(),
                    "task_head_state_dict": raw_model.task_head.state_dict(),
                    "best_val_task_bce": best_val,
                    "control_name": "task_only",
                    "main_train_script_sha256": file_sha256(train_script),
                },
                best_path,
            )
        save_json(
            {"history": history, "best_val_task_bce": best_val, "best_model": str(best_path)},
            args.output_root / "training_history.json",
        )
    print(f"[task-only] wrote best checkpoint: {best_path}")


def task_target_from_arrays(input_root: Path, arrays: dict, task_feature_name: str) -> np.ndarray:
    index, mean, std = locate_task_feature(input_root, task_feature_name)
    raw = np.asarray(arrays["x_num"][:, index], dtype=np.float32) * np.float32(std) + np.float32(mean)
    return np.clip(raw, 0.0, 1.0).astype(np.float32)


def load_task_model(checkpoint: Path, train_module: Any, device: torch.device) -> Tuple[nn.Module, dict]:
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    config_dict = checkpoint_data["config"]
    shapes = checkpoint_data["model_shapes"]
    config = TaskOnlyConfig(**config_dict)
    model = build_task_model(
        train_module,
        config,
        int(shapes["n_num"]),
        int(shapes["n_cat"]),
        int(shapes["hash_buckets"]),
        device,
    )
    model.base.load_state_dict(checkpoint_data["base_model_state_dict"], strict=True)
    model.task_head.load_state_dict(checkpoint_data["task_head_state_dict"], strict=True)
    model.to(device).eval()
    return model, config_dict


@torch.inference_mode()
def task_forward_split(
    model: TaskOnlyEventModel,
    arrays: dict,
    device: torch.device,
    chunk_len: int,
    sample_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    n_rows = int(arrays["n"])
    logits = np.zeros((n_rows,), dtype=np.float32)
    collect_samples = sample_indices is not None and len(sample_indices) > 0
    requested = set(sample_indices.tolist()) if collect_samples else set()
    before_parts: List[np.ndarray] = []
    after_parts: List[np.ndarray] = []
    collected_indices: List[int] = []
    autocast_enabled = device.type == "cuda"
    for sequence_index in range(len(arrays["offsets"]) - 1):
        start = int(arrays["offsets"][sequence_index])
        stop = int(arrays["offsets"][sequence_index + 1])
        hidden_state = None
        previous_hidden = None
        position = start
        while position < stop:
            end = min(position + chunk_len, stop)
            x_num = torch.from_numpy(np.asarray(arrays["x_num"][position:end])).unsqueeze(0).to(device, non_blocking=True)
            x_cat = torch.from_numpy(np.asarray(arrays["x_cat"][position:end])).unsqueeze(0).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                embedded = model.base.embed_inputs(x_num, x_cat)
                hidden_after, hidden_state = model.base.rnn(embedded, hidden_state)
                first_before = torch.zeros_like(hidden_after[:, :1, :]) if previous_hidden is None else previous_hidden
                hidden_before = torch.cat([first_before, hidden_after[:, :-1, :]], dim=1)
                output_logits = model.task_head(hidden_before).squeeze(-1)
            logits[position:end] = output_logits.squeeze(0).float().cpu().numpy().astype(np.float32)
            if collect_samples:
                rows = np.arange(position, end, dtype=np.int64)
                take = np.fromiter((int(row) in requested for row in rows), dtype=bool, count=len(rows))
                if np.any(take):
                    before_parts.append(hidden_before.squeeze(0).float().cpu().numpy().astype(np.float32)[take])
                    after_parts.append(hidden_after.squeeze(0).float().cpu().numpy().astype(np.float32)[take])
                    collected_indices.extend(rows[take].tolist())
            previous_hidden = hidden_after[:, -1:, :].detach()
            position = end
    if collect_samples:
        if not before_parts:
            return logits, np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
        order = np.argsort(np.asarray(collected_indices, dtype=np.int64))
        return (
            logits,
            np.concatenate(before_parts, axis=0)[order],
            np.concatenate(after_parts, axis=0)[order],
        )
    return logits, None, None


def sample_indices(n_rows: int, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or n_rows <= max_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(n_rows, dtype=np.int64), size=max_rows, replace=False))


def fit_macro_probes(
    model: TaskOnlyEventModel,
    eval_module: Any,
    input_root: Path,
    device: torch.device,
    max_rows: int,
    chunk_len: int,
    seed: int,
) -> Tuple[Ridge, Ridge]:
    arrays = eval_module.read_arrays(input_root, "A_train")
    indices = sample_indices(int(arrays["n"]), max_rows, seed)
    _, hidden_before, hidden_after = task_forward_split(
        model,
        arrays,
        device,
        chunk_len,
        sample_indices=indices,
    )
    if hidden_before is None or hidden_after is None or hidden_before.shape[0] == 0:
        raise RuntimeError("Could not collect hidden states for task-only macro probes.")
    current_target = np.asarray(arrays["y"][indices], dtype=np.float32)
    next_target = np.asarray(arrays["y_next"][indices], dtype=np.float32)
    current_probe = Ridge(alpha=1.0, fit_intercept=True).fit(hidden_before, current_target)
    next_probe = Ridge(alpha=1.0, fit_intercept=True).fit(hidden_after, next_target)
    return current_probe, next_probe


@torch.inference_mode()
def predict_with_probes(
    model: TaskOnlyEventModel,
    arrays: dict,
    device: torch.device,
    chunk_len: int,
    current_probe: Ridge,
    next_probe: Ridge,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = int(arrays["n"])
    pred_current = np.zeros((n_rows, 2), dtype=np.float32)
    pred_next = np.zeros((n_rows, 2), dtype=np.float32)
    logits = np.zeros((n_rows,), dtype=np.float32)
    autocast_enabled = device.type == "cuda"
    for sequence_index in range(len(arrays["offsets"]) - 1):
        start = int(arrays["offsets"][sequence_index])
        stop = int(arrays["offsets"][sequence_index + 1])
        hidden_state = None
        previous_hidden = None
        position = start
        while position < stop:
            end = min(position + chunk_len, stop)
            x_num = torch.from_numpy(np.asarray(arrays["x_num"][position:end])).unsqueeze(0).to(device, non_blocking=True)
            x_cat = torch.from_numpy(np.asarray(arrays["x_cat"][position:end])).unsqueeze(0).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                embedded = model.base.embed_inputs(x_num, x_cat)
                hidden_after, hidden_state = model.base.rnn(embedded, hidden_state)
                first_before = torch.zeros_like(hidden_after[:, :1, :]) if previous_hidden is None else previous_hidden
                hidden_before = torch.cat([first_before, hidden_after[:, :-1, :]], dim=1)
                output_logits = model.task_head(hidden_before).squeeze(-1)
            before_np = hidden_before.squeeze(0).float().cpu().numpy().astype(np.float32)
            after_np = hidden_after.squeeze(0).float().cpu().numpy().astype(np.float32)
            pred_current[position:end] = np.clip(current_probe.predict(before_np), -1.0, 1.0).astype(np.float32)
            pred_next[position:end] = np.clip(next_probe.predict(after_np), -1.0, 1.0).astype(np.float32)
            logits[position:end] = output_logits.squeeze(0).float().cpu().numpy().astype(np.float32)
            previous_hidden = hidden_after[:, -1:, :].detach()
            position = end
    return pred_current, pred_next, logits


def task_metrics(target: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    target = np.asarray(target, dtype=np.float32)
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)))
    epsilon = 1e-7
    bce = -np.nanmean(
        target * np.log(np.clip(probabilities, epsilon, 1.0 - epsilon))
        + (1.0 - target) * np.log(np.clip(1.0 - probabilities, epsilon, 1.0))
    )
    output = {
        "task_bce": float(bce),
        "task_rmse": float(np.sqrt(np.nanmean((probabilities - target) ** 2))),
        "task_mae": float(np.nanmean(np.abs(probabilities - target))),
        "task_prob_mean": float(np.nanmean(probabilities)),
        "task_target_mean": float(np.nanmean(target)),
        "task_prob_target_corr": pearson(probabilities, target),
    }
    binary = (target <= 1e-6) | (target >= 1.0 - 1e-6)
    if int(np.sum(binary)) >= 10 and len(np.unique((target[binary] > 0.5).astype(int))) == 2:
        binary_target = (target[binary] > 0.5).astype(int)
        binary_probability = probabilities[binary]
        output["task_auc_binary_rows"] = float(roc_auc_score(binary_target, binary_probability))
        output["task_accuracy_at_0p5_binary_rows"] = float(
            np.mean((binary_probability >= 0.5).astype(int) == binary_target)
        )
        output["task_binary_rows_used"] = int(np.sum(binary))
    else:
        thresholded_target = (target > 0.5).astype(int)
        if len(np.unique(thresholded_target)) == 2:
            output["task_auc_thresholded_all_rows"] = float(
                roc_auc_score(thresholded_target, probabilities)
            )
            output["task_accuracy_at_0p5_thresholded_all_rows"] = float(
                np.mean((probabilities >= 0.5).astype(int) == thresholded_target)
            )
        output["task_binary_rows_used"] = int(np.sum(binary))
    return output


def evaluate_main(args: argparse.Namespace) -> None:
    train_script = resolve_script(args.train_script, TRAIN_SCRIPT_BASENAME)
    evaluate_script = resolve_script(args.evaluate_script, EVALUATE_SCRIPT_BASENAME)
    train_module = import_module(train_script, "event_ssl_train_for_task_only_eval")
    eval_module = import_module(evaluate_script, "event_ssl_eval_for_task_only")
    validate_main_modules(train_module, eval_module)
    if args.torch_num_threads > 0:
        torch.set_num_threads(int(args.torch_num_threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_task_model(args.checkpoint, train_module, device)

    input_manifest = load_json(args.input_root / "metadata" / "stage4_input_manifest.json")
    train_module.validate_input_contract(input_manifest)
    stage1_root = eval_module.resolve_stage1_root(input_manifest, args.stage1_root)
    partition = eval_module.load_fixed_k6_partition(stage1_root, input_manifest)
    convergence_sample_max = int(partition.audit.get("fit_max_rows", 500000))
    conv_M, conv_Psi, convergence_meta = eval_module.convergence_reference(
        args.input_root,
        convergence_sample_max,
        args.seed,
    )
    current_probe, next_probe = fit_macro_probes(
        model,
        eval_module,
        args.input_root,
        device,
        args.probe_max_rows,
        args.chunk_len,
        args.seed,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    table_dir = args.output_root / "tables"
    prediction_dir = args.output_root / "predictions"
    metadata_dir = args.output_root / "metadata"
    for directory in (table_dir, prediction_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    save_json(
        dict(partition.audit),
        metadata_dir / "stage4_task_only_fixed_k6_partition_audit.json",
    )
    pd.DataFrame(
        {
            "macrostate": np.arange(partition.k, dtype=int),
            "center_M": partition.centers[:, 0],
            "center_Psi": partition.centers[:, 1],
            "center_M_standardized": partition.standardized_centers[:, 0],
            "center_Psi_standardized": partition.standardized_centers[:, 1],
        }
    ).to_csv(table_dir / "stage4_task_only_fixed_k6_macrostate_centers.csv", index=False)

    all_metrics: List[dict] = []
    all_task: List[dict] = []
    split_outputs: Dict[str, Any] = {}
    for split in args.splits:
        print(f"[task-only eval] split={split}", flush=True)
        arrays = eval_module.read_arrays(args.input_root, split)
        pred_current, pred_next, logits = predict_with_probes(
            model,
            arrays,
            device,
            args.chunk_len,
            current_probe,
            next_probe,
        )
        metrics, matrices = eval_module.metrics_for_predictions(
            arrays,
            pred_current,
            pred_next,
            partition,
            (conv_M, conv_Psi),
        )
        metrics["macrostate_partition_verified_against_stage1_fixed_k6"] = 1.0
        metrics["macrostate_k_fixed_a_priori"] = 1.0
        task_target = task_target_from_arrays(
            args.input_root,
            arrays,
            config.get("task_feature_name", TASK_FEATURE_NAME),
        )
        task_result = task_metrics(task_target, logits)
        metrics.update(task_result)
        metrics["split"] = split
        all_metrics.append(metrics)
        all_task.append({"split": split, **task_result})

        predictions = pd.DataFrame(
            {
                "split": split,
                "user_id": np.asarray(arrays["user_id"], dtype=np.int64),
                "bundle_step_index": np.asarray(arrays["step"], dtype=np.int64),
                "M": np.asarray(arrays["y"][:, 0], dtype=np.float32),
                "Psi": np.asarray(arrays["y"][:, 1], dtype=np.float32),
                "target_M_next": np.asarray(arrays["y_next"][:, 0], dtype=np.float32),
                "target_Psi_next": np.asarray(arrays["y_next"][:, 1], dtype=np.float32),
                "pred_M": pred_current[:, 0],
                "pred_Psi": pred_current[:, 1],
                "pred_next_M": pred_next[:, 0],
                "pred_next_Psi": pred_next[:, 1],
                "task_logit": logits,
                "task_probability": 1.0 / (1.0 + np.exp(-logits)),
                "task_target": task_target,
            }
        )
        prediction_path = write_table(
            predictions,
            prediction_dir / f"stage4_task_only_predictions_{split}",
        )
        metrics_path = write_table(
            pd.DataFrame([metrics]),
            table_dir / f"stage4_task_only_structural_metrics_{split}",
        )
        matrix_path = table_dir / f"stage4_task_only_transition_matrices_{split}.npz"
        np.savez_compressed(matrix_path, **matrices)
        split_outputs[split] = {
            "rows": int(len(predictions)),
            "users": int(predictions["user_id"].nunique()),
            "prediction_path": str(prediction_path),
            "metrics_path": str(metrics_path),
            "matrix_path": str(matrix_path),
        }

    metrics_df = pd.DataFrame(all_metrics)
    required_flags = {
        "macrostate_partition_verified_against_stage1_fixed_k6",
        "macrostate_k_fixed_a_priori",
    }
    missing_flags = sorted(required_flags.difference(metrics_df.columns))
    if missing_flags:
        raise RuntimeError(f"Task-only metrics are missing fixed-K audit fields: {missing_flags}")
    if not bool((pd.to_numeric(metrics_df["macrostate_partition_verified_against_stage1_fixed_k6"], errors="coerce") == 1.0).all()):
        raise RuntimeError("At least one task-only metric row failed the fixed-K audit.")
    if not bool((pd.to_numeric(metrics_df["macrostate_k_fixed_a_priori"], errors="coerce") == 1.0).all()):
        raise RuntimeError("At least one task-only metric row changed the fixed K=6 contract.")

    metrics_all_path = write_table(
        metrics_df,
        table_dir / "stage4_task_only_structural_metrics_all_splits",
    )
    task_all_path = write_table(
        pd.DataFrame(all_task),
        table_dir / "stage4_task_only_task_metrics_all_splits",
    )
    save_json(
        {
            "control_name": "task_only",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(args.checkpoint.resolve()),
            "input_root": str(args.input_root.resolve()),
            "stage1_root": str(stage1_root),
            "train_script": str(train_script),
            "train_script_sha256": file_sha256(train_script),
            "evaluate_script": str(evaluate_script),
            "evaluate_script_sha256": file_sha256(evaluate_script),
            "primary_coordinates": ["M", "Psi"],
            "probe_policy": "Ridge probes fitted on A_train hidden states only",
            "macro_partition": dict(partition.audit),
            "convergence_reference": convergence_meta,
            "guardrails": {
                "kmeans_refit": False,
                "macrostate_k_selected": False,
                "macrostate_k": EXPECTED_MACROSTATE_K,
                "B_confirm_used_for_update": False,
            },
            "splits": split_outputs,
            "outputs": {
                "structural_metrics_all_splits": str(metrics_all_path),
                "task_metrics_all_splits": str(task_all_path),
            },
        },
        metadata_dir / "stage4_task_only_evaluation_manifest.json",
    )


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
    parser.add_argument("--categorical-emb-dim", type=int, default=16)
    parser.add_argument("--future-steps", type=str, default="1,2,4")
    parser.add_argument("--delta-scale", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--task-feature-name", type=str, default=TASK_FEATURE_NAME)
    parser.add_argument("--no-truncated-supervision", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task-only Event-SSL control.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    add_train_arguments(train_parser)
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--input-root", type=Path, required=True)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--output-root", type=Path, required=True)
    eval_parser.add_argument("--train-script", type=Path, default=None)
    eval_parser.add_argument("--evaluate-script", type=Path, default=None)
    eval_parser.add_argument("--splits", nargs="+", default=["A_val", "B_confirm"])
    eval_parser.add_argument("--chunk-len", type=int, default=512)
    eval_parser.add_argument("--stage1-root", type=Path, default=None)
    eval_parser.add_argument("--probe-max-rows", type=int, default=300000)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--torch-num-threads", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        train_main(args)
    elif args.command == "evaluate":
        evaluate_main(args)
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
