#!/usr/bin/env python3
from __future__ import annotations

"""Train predictive-state or pure-SSL Event-SSL on prepared interval sequences.

The supervised model predicts only M and Psi. State and closure losses are
applied to windows that begin at a true sequence boundary unless explicitly
overridden.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


FUTURE_TARGET_CONTRACT = {
    "target": "numeric primitive vector shifted by each configured future horizon",
    "validity": "both source position t and shifted target position t+k must be observed",
    "padding_policy": "right-padding is excluded from future targets by the shifted observation mask",
}


@dataclass
class TrainConfig:
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
    lambda_future: float
    lambda_state: float
    lambda_closure: float
    future_steps: List[int]
    delta_scale: float
    categorical_emb_dim: int
    use_compile: bool
    amp_dtype: str
    allow_truncated_supervision: bool


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class SequenceMemmapDataset(Dataset):
    def __init__(
        self,
        input_root: Path,
        split: str,
        seq_len: int,
        stride: int,
        min_seq_len: int = 3,
        allow_truncated_supervision: bool = False,
    ) -> None:
        self.input_root = Path(input_root)
        self.split = split
        self.split_dir = self.input_root / split
        manifest = load_json(self.input_root / "metadata" / "stage4_input_manifest.json")
        summary = manifest["split_summaries"][split]
        n = int(summary["rows"])
        self.n_num = int(summary["numeric_shape"][1])
        self.n_cat = int(summary["categorical_shape"][1])
        self.x_num = np.memmap(self.split_dir / "x_num.float32.mmap", mode="r", dtype=np.float32, shape=(n, self.n_num))
        self.x_cat = np.memmap(self.split_dir / "x_cat.int64.mmap", mode="r", dtype=np.int64, shape=(n, self.n_cat))
        self.y = np.memmap(self.split_dir / "y_current.float32.mmap", mode="r", dtype=np.float32, shape=(n, 2))
        self.y_next = np.memmap(self.split_dir / "y_next.float32.mmap", mode="r", dtype=np.float32, shape=(n, 2))
        self.offsets = np.load(self.split_dir / "sequence_offsets.npy")
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.min_seq_len = int(min_seq_len)
        self.allow_truncated_supervision = bool(allow_truncated_supervision)
        self.windows: List[Tuple[int, int, int]] = []  # start, end, supervision_flag
        for i in range(len(self.offsets) - 1):
            s = int(self.offsets[i]); e = int(self.offsets[i + 1]); L = e - s
            if L < self.min_seq_len:
                continue
            first_end = min(s + self.seq_len, e)
            self.windows.append((s, first_end, 1))
            if first_end >= e:
                continue
            pos = s + max(1, self.stride)
            while pos + self.min_seq_len <= e:
                end = min(pos + self.seq_len, e)
                self.windows.append((pos, end, 1 if self.allow_truncated_supervision else 0))
                if end == e:
                    break
                pos += max(1, self.stride)
        if not self.windows:
            raise RuntimeError(f"No sequence windows for split={split}.")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s, e, supervise = self.windows[idx]
        L = e - s
        x_num = np.zeros((self.seq_len, self.n_num), dtype=np.float32)
        x_cat = np.zeros((self.seq_len, self.n_cat), dtype=np.int64)
        y = np.zeros((self.seq_len, 2), dtype=np.float32)
        y_next = np.zeros((self.seq_len, 2), dtype=np.float32)
        mask = np.zeros((self.seq_len,), dtype=np.float32)
        supervised_mask = np.zeros((self.seq_len,), dtype=np.float32)
        x_num[:L] = self.x_num[s:e]
        if self.n_cat:
            x_cat[:L] = self.x_cat[s:e]
        y[:L] = self.y[s:e]
        y_next[:L] = self.y_next[s:e]
        mask[:L] = 1.0
        if supervise:
            supervised_mask[:L] = 1.0
        return {
            "x_num": torch.from_numpy(x_num),
            "x_cat": torch.from_numpy(x_cat),
            "y": torch.from_numpy(y),
            "y_next": torch.from_numpy(y_next),
            "mask": torch.from_numpy(mask),
            "supervised_mask": torch.from_numpy(supervised_mask),
        }


class PredictiveStateEventSSL(nn.Module):
    def __init__(
        self,
        n_num: int,
        n_cat: int,
        hash_buckets: int,
        hidden_dim: int = 320,
        input_dim: int = 224,
        num_layers: int = 2,
        dropout: float = 0.10,
        categorical_emb_dim: int = 16,
        future_steps: Tuple[int, ...] = (1, 2, 4),
        delta_scale: float = 0.50,
    ) -> None:
        super().__init__()
        self.n_num = int(n_num)
        self.n_cat = int(n_cat)
        self.hash_buckets = int(hash_buckets)
        self.future_steps = tuple(int(k) for k in future_steps)
        self.delta_scale = float(delta_scale)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(self.hash_buckets, categorical_emb_dim, padding_idx=0) for _ in range(self.n_cat)
        ])
        cat_dim = self.n_cat * categorical_emb_dim
        self.input_proj = nn.Sequential(
            nn.Linear(self.n_num + cat_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
        )
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2), nn.Tanh()
        )
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2), nn.Tanh()
        )
        self.future_heads = nn.ModuleDict({str(k): nn.Linear(hidden_dim, self.n_num) for k in self.future_steps})

    def embed_inputs(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if self.n_cat > 0:
            embs = [emb(x_cat[..., j].clamp(min=0, max=self.hash_buckets - 1)) for j, emb in enumerate(self.cat_embeddings)]
            cat = torch.cat(embs, dim=-1)
            x = torch.cat([x_num, cat], dim=-1)
        else:
            x = x_num
        return self.input_proj(x)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.embed_inputs(x_num, x_cat)
        h_after, _ = self.rnn(z)
        zero = torch.zeros_like(h_after[:, :1, :])
        h_before = torch.cat([zero, h_after[:, :-1, :]], dim=1)
        state = self.state_head(h_before)
        delta = self.delta_scale * self.delta_head(torch.cat([h_before, z], dim=-1))
        next_state = torch.tanh(state + delta)
        future = {int(k): head(h_after) for k, head in self.future_heads.items()}
        return {"state": state, "next_state": next_state, "h_after": h_after, "h_before": h_before, "future": future}


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(-1)
    err = (pred - target) ** 2
    return (err * mask).sum() / mask.sum().clamp_min(1.0) / pred.shape[-1]


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float = 0.05) -> torch.Tensor:
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(-1)
    diff = pred - target
    abs_diff = diff.abs()
    loss = torch.where(abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0) / pred.shape[-1]


def shifted_future_target(
    x_num: torch.Tensor,
    mask: torch.Tensor,
    horizon: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shift numeric targets and exclude padded source or target positions."""
    h = int(horizon)
    if h <= 0:
        raise ValueError(f"Future horizon must be positive; got {h}.")
    if x_num.ndim != 3:
        raise ValueError(f"x_num must have shape [batch, time, features]; got {tuple(x_num.shape)}.")
    if mask.ndim != 2 or mask.shape[:2] != x_num.shape[:2]:
        raise ValueError(
            "mask must have shape [batch, time] matching x_num; "
            f"got mask={tuple(mask.shape)}, x_num={tuple(x_num.shape)}."
        )

    target = torch.zeros_like(x_num)
    valid = torch.zeros_like(mask)
    if x_num.shape[1] <= h:
        return target, valid

    target[:, :-h, :] = x_num[:, h:, :]
    # A future pair contributes only if both source t and target t+h are real.
    valid[:, :-h] = mask[:, :-h] * mask[:, h:]
    return target, valid


def compute_losses(model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: TrainConfig) -> Dict[str, torch.Tensor]:
    x_num = batch["x_num"]
    mask = batch["mask"]
    supervised_mask = batch["supervised_mask"].clone()
    if cfg.warmup_steps > 0:
        supervised_mask[:, : cfg.warmup_steps] = 0.0
    state_loss = masked_huber(model_out["state"], batch["y"], supervised_mask)
    closure_loss = masked_huber(model_out["next_state"], batch["y_next"], supervised_mask)
    future_loss = torch.zeros((), device=x_num.device, dtype=x_num.dtype)
    valid_horizons = 0
    for k, pred in model_out["future"].items():
        target, valid = shifted_future_target(x_num, mask, int(k))
        if not bool(torch.any(valid > 0).item()):
            continue
        future_loss = future_loss + masked_mse(pred, target, valid)
        valid_horizons += 1
    if valid_horizons > 0:
        future_loss = future_loss / valid_horizons
    total = cfg.lambda_future * future_loss + cfg.lambda_state * state_loss + cfg.lambda_closure * closure_loss
    return {"total": total, "future": future_loss, "state": state_loss, "closure": closure_loss}


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, cfg: TrainConfig, device: torch.device, train: bool) -> Dict[str, float]:
    model.train(train)
    sums = {"total": 0.0, "future": 0.0, "state": 0.0, "closure": 0.0}
    n_batches = 0
    autocast_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    t0 = time.time()
    for batch in loader:
        batch = move_batch(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=autocast_enabled):
                out = model(batch["x_num"], batch["x_cat"])
                losses = compute_losses(out, batch, cfg)
            if train:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        for k in sums:
            sums[k] += float(losses[k].detach().cpu())
        n_batches += 1
    out = {k: v / max(n_batches, 1) for k, v in sums.items()}
    out["batches"] = float(n_batches)
    out["seconds"] = time.time() - t0
    return out


def validate_input_contract(manifest: dict) -> None:
    if manifest.get("primary_coordinates") != ["M", "Psi"]:
        raise RuntimeError("Input manifest primary coordinates are not exactly ['M', 'Psi'].")
    contract = manifest.get("stage1_fixed_k6_contract", {})
    if contract.get("verified") is not True:
        raise RuntimeError("Input manifest does not contain a verified Stage-1 fixed-K contract.")
    if int(contract.get("macrostate_k", -1)) != 6 or contract.get("macrostate_k_rule") != "fixed a priori":
        raise RuntimeError("Stage-1 mesostate contract is not fixed K=6.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Stage-4 predictive-state Event-SSL model.")
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--model-kind", choices=["predictive_state", "pure_ssl"], default="predictive_state")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--min-seq-len", type=int, default=3)
    ap.add_argument("--warmup-steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=320)
    ap.add_argument("--input-dim", type=int, default=224)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--future-steps", type=str, default="1,2,4")
    ap.add_argument("--lambda-future", type=float, default=1.0)
    ap.add_argument("--lambda-state", type=float, default=None)
    ap.add_argument("--lambda-closure", type=float, default=None)
    ap.add_argument("--delta-scale", type=float, default=0.50)
    ap.add_argument("--categorical-emb-dim", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--torch-num-threads", type=int, default=0, help="Optional CPU thread cap; 0 leaves PyTorch default. Useful for CPU smoke tests.")
    ap.add_argument("--allow-truncated-supervision", action="store_true", help="Allow supervised state/closure loss on sliding windows that do not start at a sequence boundary. Off by default for scientific correctness.")
    args = ap.parse_args()

    if args.torch_num_threads and args.torch_num_threads > 0:
        torch.set_num_threads(int(args.torch_num_threads))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    input_manifest = load_json(args.input_root / "metadata" / "stage4_input_manifest.json")
    validate_input_contract(input_manifest)
    hash_buckets = int(input_manifest["categorical_hash_buckets"])
    train_ds = SequenceMemmapDataset(args.input_root, "A_train", args.seq_len, args.stride, args.min_seq_len, args.allow_truncated_supervision)
    val_ds = SequenceMemmapDataset(args.input_root, "A_val", args.seq_len, args.stride, args.min_seq_len, args.allow_truncated_supervision)

    lambda_state = (0.5 if args.model_kind == "predictive_state" else 0.0) if args.lambda_state is None else float(args.lambda_state)
    lambda_closure = (0.5 if args.model_kind == "predictive_state" else 0.0) if args.lambda_closure is None else float(args.lambda_closure)

    cfg = TrainConfig(
        input_root=str(args.input_root.resolve()), output_root=str(args.output_root.resolve()), model_kind=args.model_kind,
        seed=args.seed, seq_len=args.seq_len, stride=args.stride, min_seq_len=args.min_seq_len, warmup_steps=args.warmup_steps,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim, input_dim=args.input_dim, num_layers=args.num_layers, dropout=args.dropout,
        num_workers=args.num_workers, lambda_future=args.lambda_future, lambda_state=lambda_state, lambda_closure=lambda_closure,
        future_steps=[int(x) for x in args.future_steps.split(",") if x.strip()], delta_scale=args.delta_scale,
        categorical_emb_dim=args.categorical_emb_dim, use_compile=bool(args.compile), amp_dtype=args.amp_dtype,
        allow_truncated_supervision=bool(args.allow_truncated_supervision),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_json({
        "config": asdict(cfg),
        "input_manifest_subset": {
            "primary_coordinates": input_manifest.get("primary_coordinates"),
            "numeric_feature_count": len(input_manifest.get("numeric_feature_names_after_expansion", [])),
            "categorical_feature_count": len(input_manifest.get("categorical_input_source_columns", [])),
            "excluded_coordinate_policy": input_manifest.get("excluded_coordinate_policy"),
            "sequence_boundary_policy": input_manifest.get("sequence_boundary_policy"),
            "stage1_fixed_k6_contract": input_manifest.get("stage1_fixed_k6_contract"),
        },
        "future_target_contract": FUTURE_TARGET_CONTRACT,
    }, args.output_root / "training_manifest.json")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PredictiveStateEventSSL(
        n_num=train_ds.n_num, n_cat=train_ds.n_cat, hash_buckets=hash_buckets,
        hidden_dim=args.hidden_dim, input_dim=args.input_dim, num_layers=args.num_layers, dropout=args.dropout,
        categorical_emb_dim=args.categorical_emb_dim, future_steps=tuple(cfg.future_steps), delta_scale=args.delta_scale,
    ).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[dict] = []
    best_metric = float("inf")
    best_path = args.output_root / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, cfg, device, train=True)
        va = run_epoch(model, val_loader, optimizer, cfg, device, train=False)
        select_metric = va["future"] if args.model_kind == "pure_ssl" else (va["state"] + va["closure"])
        row = {"epoch": epoch, "train": tr, "val": va, "selection_metric": float(select_metric)}
        history.append(row)
        print(f"[Stage4 train] epoch={epoch} train_total={tr['total']:.5f} val_total={va['total']:.5f} selection={select_metric:.5f} val_state={va['state']:.5f} val_closure={va['closure']:.5f} val_future={va['future']:.5f}", flush=True)
        if select_metric < best_metric:
            best_metric = float(select_metric)
            ckpt = {
                "model_state_dict": model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict(),
                "config": asdict(cfg),
                "model_shapes": {"n_num": train_ds.n_num, "n_cat": train_ds.n_cat, "hash_buckets": hash_buckets},
                "best_epoch": epoch,
                "best_selection_metric": best_metric,
                "future_target_contract": FUTURE_TARGET_CONTRACT,
            }
            torch.save(ckpt, best_path)
        save_json({"history": history}, args.output_root / "training_history.json")

    print(f"[Stage4 train] best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
