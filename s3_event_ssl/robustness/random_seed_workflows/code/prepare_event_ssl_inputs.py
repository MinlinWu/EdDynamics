#!/usr/bin/env python3
from __future__ import annotations

"""Prepare interval-level inputs for predictive-state Event-SSL.

Only M and Psi are targets. Inputs contain interval primitives and context,
while macrostate coordinates, deltas, regions, mesostates and maturity fields
are excluded. Normalization is fitted on A_train only.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SPLITS = ("A_train", "A_val", "B_confirm")
EXPECTED_MACROSTATE_K = 6
EXPECTED_KMEANS_N_INIT = 20
EXPECTED_KMEANS_FIT_MAX_ROWS = 500000
EXPECTED_RANDOM_STATE = 42

COL_M = "M_response_prebalanced_pre"
COL_PSI = "activity_alignment_order_Psi_pre"
COL_M_NEXT = "next_M_response_prebalanced"
COL_PSI_NEXT = "next_activity_alignment_order_Psi"

REQUIRED_COLUMNS = ["user_id", "bundle_step_index", COL_M, COL_PSI, COL_M_NEXT, COL_PSI_NEXT]

# Interval primitives and context variables.
NUMERIC_PRIMITIVE_CANDIDATES = [
    "next_gap_days",
    "bundle_n_questions",
    "answered_fraction_interval",
    "current_accuracy_diagnostic_only",
    "response_active_mass_interval",
    "response_alignment_to_pre_demand",
    "response_aligned_mass_interval",
    "response_off_target_mass_interval",
    "response_neutral_mass_interval",
    "support_alignment_to_pre_demand_or_current_bundle",
    "support_active_total_interval",
    "support_active_mapped_interval",
    "support_active_unmapped_interval",
    "support_aligned_mass_interval",
    "support_off_target_mass_interval",
    "support_neutral_mass_interval",
    "support_exposure_increment_mass",
    "support_episode_count_interval",
    "support_media_event_count_interval",
    "support_media_pair_anomaly_count_interval",
    "idle_uncovered_ms_interval",
    "idle_mass_interval",
    "response_duration_active_proxy",
    "total_response_count_diagnostic",
    "response_change_count_diagnostic",
    "kt4_choice_process_count_diagnostic",
    "kt4_total_erase_count_diagnostic",
    "kt4_total_undo_erase_count_diagnostic",
    "kt4_total_text_enter_count_diagnostic",
    "access_status_aux_context_only",
    "payment_coupon_window_aux_context_only",
    "access_pay_event_count_context_only",
    "access_refund_event_count_context_only",
    "access_coupon_event_count_context_only",
]

CATEGORICAL_CANDIDATES = ["bundle_id", "part", "source_code", "platform_code"]

# Exclude macrostate, maturity and coarse-state fields from inputs.
FORBIDDEN_FEATURE_TOKENS = [
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
    "signed_graph_coherence",
    "MR_",
    "PsiA",
]

LOG1P_NUMERIC_TOKENS = (
    "count", "mass", "duration", "elapsed", "ms", "gap", "episode", "erase", "refund", "coupon", "pay", "bundle_n_questions",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_k6_contract(stage1_root: Path) -> Dict[str, object]:
    root = Path(stage1_root).resolve() / "dynamics" / "fixed_k6_mesostates"
    metadata_path = root / "fixed_k6_model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Stage-1 fixed-K metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    expected_features = [COL_M, COL_PSI]
    mapping = metadata.get("raw_to_ordered_label", {})
    labels = list(range(EXPECTED_MACROSTATE_K))
    scaler_mean = np.asarray(metadata.get("scaler_mean", []), dtype=float)
    scaler_scale = np.asarray(metadata.get("scaler_scale", []), dtype=float)
    checks = {
        "coordinate": metadata.get("coordinate") == "MR_PsiA",
        "macrostate_k": int(metadata.get("macrostate_k", -1)) == EXPECTED_MACROSTATE_K,
        "macrostate_k_rule": metadata.get("macrostate_k_rule") == "fixed a priori",
        "features": list(metadata.get("features", [])) == expected_features,
        "fit_split": metadata.get("fit_split") == "A_train",
        "user_balanced_sampling": metadata.get("user_balanced_sampling") is True,
        "user_balanced_kmeans_fit": metadata.get("user_balanced_kmeans_fit") is True,
        "kmeans_n_init": int(metadata.get("kmeans_n_init", -1)) == EXPECTED_KMEANS_N_INIT,
        "fit_max_rows": int(metadata.get("fit_max_rows", -1)) == EXPECTED_KMEANS_FIT_MAX_ROWS,
        "random_state": int(metadata.get("random_state", -1)) == EXPECTED_RANDOM_STATE,
        "scaler": scaler_mean.shape == (2,) and scaler_scale.shape == (2,) and np.isfinite(scaler_mean).all() and np.isfinite(scaler_scale).all() and np.all(scaler_scale > 0),
        "label_mapping": sorted(int(key) for key in mapping) == labels and sorted(int(value) for value in mapping.values()) == labels,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Stage-1 fixed-K contract failed: " + ", ".join(failed))

    centers_path = table_path(root / "fixed_k6_centers")
    centers = read_table(root / "fixed_k6_centers")
    required = {"macrostate", "center_M", "center_Psi", "center_M_standardized", "center_Psi_standardized"}
    if not required.issubset(centers.columns):
        raise RuntimeError(f"Stage-1 fixed-K centers are missing: {sorted(required.difference(centers.columns))}")
    centers = centers.sort_values("macrostate", kind="mergesort").reset_index(drop=True)
    ids = pd.to_numeric(centers["macrostate"], errors="coerce").to_numpy(dtype=float)
    values = centers[["center_M", "center_Psi", "center_M_standardized", "center_Psi_standardized"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    order = np.lexsort((values[:, 1], values[:, 0]))
    if len(centers) != EXPECTED_MACROSTATE_K or not np.array_equal(ids, np.arange(EXPECTED_MACROSTATE_K, dtype=float)) or not np.isfinite(values).all() or not np.array_equal(order, np.arange(EXPECTED_MACROSTATE_K)):
        raise RuntimeError("Stage-1 fixed-K centers are not the expected six ordered states.")

    return {
        "verified": True,
        "coordinate": "MR_PsiA",
        "macrostate_k": EXPECTED_MACROSTATE_K,
        "macrostate_k_rule": "fixed a priori",
        "fit_split": "A_train",
        "features": expected_features,
        "user_balanced_sampling": True,
        "user_balanced_kmeans_fit": True,
        "fit_max_rows": EXPECTED_KMEANS_FIT_MAX_ROWS,
        "kmeans_n_init": EXPECTED_KMEANS_N_INIT,
        "random_state": EXPECTED_RANDOM_STATE,
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": file_sha256(metadata_path),
        "centers_path": str(centers_path.resolve()),
        "centers_sha256": file_sha256(centers_path),
        "used_in_input_feature_construction": False,
    }


def table_path(base: Path) -> Path:
    for ext in (".parquet", ".csv.gz", ".csv"):
        p = base.with_suffix(ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find table for {base}.[parquet|csv.gz|csv]")


def read_table(base: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    p = table_path(base)
    if p.suffix == ".parquet":
        return pd.read_parquet(p, columns=list(columns) if columns is not None else None)
    return pd.read_csv(p, usecols=list(columns) if columns is not None else None, low_memory=False)


def available_columns(base: Path) -> List[str]:
    p = table_path(base)
    if p.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            return list(pq.read_schema(p).names)
        except Exception:
            return list(pd.read_parquet(p).columns)
    return list(pd.read_csv(p, nrows=5).columns)


def is_forbidden_feature(col: str) -> bool:
    return any(tok in str(col) for tok in FORBIDDEN_FEATURE_TOKENS)


def select_existing_columns(stage1_root: Path) -> Dict[str, object]:
    base = Path(stage1_root) / "dynamics" / "student_dynamics_panel_core_A_train"
    cols = set(available_columns(base))
    missing_required = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing_required:
        raise RuntimeError(f"Stage-1 A_train core panel is missing required columns: {missing_required}")

    numeric = [c for c in NUMERIC_PRIMITIVE_CANDIDATES if c in cols and not is_forbidden_feature(c)]
    categorical = [c for c in CATEGORICAL_CANDIDATES if c in cols and not is_forbidden_feature(c)]
    if not numeric and not categorical:
        raise RuntimeError("No interval primitive feature columns were found. Check Stage-1 core panel schema.")
    read_cols = sorted(set(REQUIRED_COLUMNS + numeric + categorical))
    return {"numeric": numeric, "categorical": categorical, "read_columns": read_cols}


def load_split_frame(stage1_root: Path, split: str, read_cols: Sequence[str], max_users: int = 0, seed: int = 42) -> pd.DataFrame:
    base = Path(stage1_root) / "dynamics" / f"student_dynamics_panel_core_{split}"
    df = read_table(base, columns=read_cols)
    for c in REQUIRED_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    valid = (
        df["user_id"].notna()
        & df["bundle_step_index"].notna()
        & np.isfinite(df[COL_M])
        & np.isfinite(df[COL_PSI])
        & np.isfinite(df[COL_M_NEXT])
        & np.isfinite(df[COL_PSI_NEXT])
    )
    df = df.loc[valid].copy()
    df["user_id"] = df["user_id"].astype(np.int64)
    df["bundle_step_index"] = df["bundle_step_index"].astype(np.int64)
    if max_users and max_users > 0:
        users = np.asarray(sorted(df["user_id"].unique()), dtype=np.int64)
        if len(users) > max_users:
            rng = np.random.default_rng(seed)
            keep = set(rng.choice(users, size=max_users, replace=False).tolist())
            df = df[df["user_id"].isin(keep)].copy()
    return df.sort_values(["user_id", "bundle_step_index"], kind="mergesort").reset_index(drop=True)


def build_numeric_matrix(df: pd.DataFrame, numeric_cols: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    arrays: List[np.ndarray] = []
    names: List[str] = []
    for col in numeric_cols:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        x = np.where(np.isfinite(x), x, np.nan)
        arrays.append(x)
        names.append(col)
        if any(tok in col for tok in LOG1P_NUMERIC_TOKENS):
            lx = np.log1p(np.clip(np.where(np.isfinite(x), x, 0.0), 0.0, None))
            arrays.append(lx)
            names.append(f"log1p_{col}")
    if not arrays:
        return np.zeros((len(df), 0), dtype=np.float32), []
    return np.vstack(arrays).T.astype(np.float32, copy=False), names


def robust_mean_std(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X64 = X.astype(np.float64, copy=False)
    mask = np.isfinite(X64)
    count = mask.sum(axis=0).astype(np.float64)
    filled = np.where(mask, X64, 0.0)
    mean = filled.sum(axis=0) / np.maximum(count, 1.0)
    centered = np.where(mask, X64 - mean[None, :], 0.0)
    var = (centered * centered).sum(axis=0) / np.maximum(count - 1.0, 1.0)
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32), count.astype(np.float32)


def normalize_numeric(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    X = np.where(np.isfinite(X), X, mean[None, :])
    return ((X - mean[None, :]) / np.maximum(std[None, :], 1e-6)).astype(np.float32, copy=False)


def stable_hash_series_to_bucket(series: pd.Series, buckets: int) -> np.ndarray:
    if buckets <= 1:
        return np.zeros(len(series), dtype=np.int64)
    s = series.astype("string").fillna("<NA>")
    missing = s.eq("<NA>") | s.eq("") | s.str.lower().isin(["nan", "none", "<na>"])
    hashed = pd.util.hash_pandas_object(s, index=False).to_numpy(dtype=np.uint64, copy=False)
    out = (hashed % np.uint64(buckets - 1)).astype(np.int64) + 1
    out[np.asarray(missing, dtype=bool)] = 0
    return out


def build_categorical_matrix(df: pd.DataFrame, cat_cols: Sequence[str], hash_buckets: int) -> np.ndarray:
    if not cat_cols:
        return np.zeros((len(df), 0), dtype=np.int64)
    out = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    for j, col in enumerate(cat_cols):
        out[:, j] = stable_hash_series_to_bucket(df[col], hash_buckets)
    return out


def sequence_offsets(user_ids: np.ndarray, steps: np.ndarray) -> np.ndarray:
    if user_ids.size == 0:
        return np.asarray([0], dtype=np.int64)
    user_break = user_ids[1:] != user_ids[:-1]
    step_break = steps[1:] != (steps[:-1] + 1)
    change = np.flatnonzero(user_break | step_break) + 1
    return np.concatenate([[0], change, [len(user_ids)]]).astype(np.int64)


def write_memmap(path: Path, arr: np.ndarray, dtype: np.dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(path, mode="w+", dtype=dtype, shape=arr.shape)
    mm[:] = arr.astype(dtype, copy=False)
    mm.flush()
    del mm


def build_split_arrays(
    df: pd.DataFrame,
    split_dir: Path,
    numeric_cols: Sequence[str],
    numeric_feature_names: Sequence[str],
    categorical_cols: Sequence[str],
    hash_buckets: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> Dict[str, object]:
    split_dir.mkdir(parents=True, exist_ok=True)
    X_raw, names_check = build_numeric_matrix(df, numeric_cols)
    if list(names_check) != list(numeric_feature_names):
        raise RuntimeError("Numeric feature expansion mismatch between A_train normalizer and split transform.")
    X_num = normalize_numeric(X_raw, mean, std)
    X_cat = build_categorical_matrix(df, categorical_cols, hash_buckets)
    y = df[[COL_M, COL_PSI]].to_numpy(dtype=np.float32)
    y_next = df[[COL_M_NEXT, COL_PSI_NEXT]].to_numpy(dtype=np.float32)
    uid = df["user_id"].to_numpy(dtype=np.int64)
    step = df["bundle_step_index"].to_numpy(dtype=np.int64)
    offsets = sequence_offsets(uid, step)
    lengths = np.diff(offsets)

    write_memmap(split_dir / "x_num.float32.mmap", X_num, np.float32)
    write_memmap(split_dir / "x_cat.int64.mmap", X_cat, np.int64)
    write_memmap(split_dir / "y_current.float32.mmap", y, np.float32)
    write_memmap(split_dir / "y_next.float32.mmap", y_next, np.float32)
    write_memmap(split_dir / "user_id.int64.mmap", uid, np.int64)
    write_memmap(split_dir / "bundle_step_index.int64.mmap", step, np.int64)
    np.save(split_dir / "sequence_offsets.npy", offsets)

    return {
        "rows": int(len(df)),
        "users": int(df["user_id"].nunique()),
        "numeric_shape": list(X_num.shape),
        "categorical_shape": list(X_cat.shape),
        "target_shape": list(y.shape),
        "sequence_count": int(len(offsets) - 1),
        "sequence_break_policy": "break on user_id change or non-contiguous bundle_step_index",
        "min_sequence_len": int(np.min(lengths)) if lengths.size else 0,
        "median_sequence_len": float(np.median(lengths)) if lengths.size else 0.0,
        "max_sequence_len": int(np.max(lengths)) if lengths.size else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage-4 Event-SSL interval primitive arrays.")
    ap.add_argument("--stage1-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/stage1"))
    ap.add_argument("--output-root", type=Path, default=Path("/data/datasets/KT4/outputs_KT4/stage4_event_ssl"))
    ap.add_argument("--hash-buckets", type=int, default=32768)
    ap.add_argument("--max-users-per-split", type=int, default=0, help="0 means full split; positive values are for smoke tests only.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = args.output_root / "prepared_inputs"
    meta_root = data_root / "metadata"
    meta_root.mkdir(parents=True, exist_ok=True)

    kmeans_contract = fixed_k6_contract(args.stage1_root)
    selected = select_existing_columns(args.stage1_root)
    numeric_cols = list(selected["numeric"])
    categorical_cols = list(selected["categorical"])
    print(f"[Stage4 input] numeric primitive columns: {len(numeric_cols)}")
    print(f"[Stage4 input] categorical primitive columns: {len(categorical_cols)}")

    train_df = load_split_frame(args.stage1_root, "A_train", selected["read_columns"], args.max_users_per_split, args.seed)
    X_train_raw, numeric_feature_names = build_numeric_matrix(train_df, numeric_cols)
    mean, std, count = robust_mean_std(X_train_raw)
    normalizer = {
        "fitted_on": "A_train only",
        "numeric_feature_names": list(numeric_feature_names),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "finite_count": count.tolist(),
    }
    with (meta_root / "normalizer.json").open("w", encoding="utf-8") as f:
        json.dump(normalizer, f, indent=2)

    split_summaries: Dict[str, object] = {}
    for split in SPLITS:
        df = train_df if split == "A_train" else load_split_frame(args.stage1_root, split, selected["read_columns"], args.max_users_per_split, args.seed)
        split_summaries[split] = build_split_arrays(
            df=df,
            split_dir=data_root / split,
            numeric_cols=numeric_cols,
            numeric_feature_names=numeric_feature_names,
            categorical_cols=categorical_cols,
            hash_buckets=args.hash_buckets,
            mean=mean,
            std=std,
        )
        print(f"[Stage4 input] {split}: {split_summaries[split]}")
        if split == "A_train":
            del train_df
        del df

    manifest = {
        "script": Path(__file__).name,
        "stage1_root": str(args.stage1_root.resolve()),
        "stage1_fixed_k6_contract": kmeans_contract,
        "output_root": str(args.output_root.resolve()),
        "data_root": str(data_root.resolve()),
        "primary_coordinates": ["M", "Psi"],
        "excluded_coordinate_policy": "V/maturity/noise coordinate and all maturity-related columns are excluded from features, targets and training outputs",
        "targets": {"current": [COL_M, COL_PSI], "next": [COL_M_NEXT, COL_PSI_NEXT]},
        "numeric_input_source_columns": numeric_cols,
        "numeric_feature_names_after_expansion": list(numeric_feature_names),
        "categorical_input_source_columns": categorical_cols,
        "categorical_hash_buckets": int(args.hash_buckets),
        "forbidden_feature_tokens": FORBIDDEN_FEATURE_TOKENS,
        "normalization_fit_scope": "A_train only",
        "sequence_boundary_policy": "user_id change or non-contiguous bundle_step_index",
        "confirmation_policy": "B_confirm arrays are written for output-only evaluation; no normalizer or feature selection is fitted on B_confirm.",
        "split_summaries": split_summaries,
        "smoke_test_max_users_per_split": int(args.max_users_per_split),
    }
    with (meta_root / "stage4_input_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Stage4 input] wrote manifest: {meta_root / 'stage4_input_manifest.json'}")


if __name__ == "__main__":
    main()
