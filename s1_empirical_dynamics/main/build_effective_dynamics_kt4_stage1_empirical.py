#!/usr/bin/env python3
"""Build the EdNet-KT4 empirical effective-dynamics panels and numerical outputs."""

from __future__ import annotations

import gzip
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label as ndimage_label
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

DATA_ROOT = Path(os.environ.get("EDNET_KT4_DATA_ROOT", "/data/datasets/KT4/data_297915"))
KT4_ROOT = Path(os.environ.get("EDNET_KT4_PREPROCESSED_ROOT", str(DATA_ROOT / "kt4")))
CONTENTS_ROOT = Path(os.environ.get("EDNET_CONTENTS_ROOT", str(DATA_ROOT / "contents")))
META_ROOT = Path(os.environ.get("EDNET_KT4_METADATA_ROOT", str(DATA_ROOT / "metadata")))
OUTPUT_ROOT = Path(os.environ.get("EDNET_KT4_OUTPUT_ROOT", "/data/datasets/KT4/outputs_KT4"))

STAGE1_ROOT = OUTPUT_ROOT / "stage1"
DYN_ROOT = STAGE1_ROOT / "dynamics"
META_OUT_ROOT = STAGE1_ROOT / "metadata"
SPLIT_ROOT = STAGE1_ROOT / "splits"
RAW_ROOT = DYN_ROOT / "raw_panels"
COORD_ROOT = DYN_ROOT / "coordinate_analysis"
REGION_ROOT = DYN_ROOT / "candidate_regions"
MESOSTATE_ROOT = DYN_ROOT / "fixed_k6_mesostates"

A_TRAIN_USERS = int(os.environ.get("EDNET_STAGE1_A_TRAIN_USERS", "178749"))
A_VAL_USERS = int(os.environ.get("EDNET_STAGE1_A_VAL_USERS", "59583"))
B_CONFIRM_USERS = int(os.environ.get("EDNET_STAGE1_B_CONFIRM_USERS", "59583"))
ALLOW_SMALL_DEV_SPLIT = bool(int(os.environ.get("EDNET_STAGE1_ALLOW_SMALL_DEV_SPLIT", "0")))
RANDOM_STATE = int(os.environ.get("EDNET_STAGE1_RANDOM_STATE", "42"))

DAY_MS = 86_400_000.0
TAU_RESPONSE_DAYS = float(os.environ.get("EDNET_STAGE1_TAU_RESPONSE_DAYS", "10.0"))
TAU_ACTIVITY_DAYS = float(os.environ.get("EDNET_STAGE1_TAU_ACTIVITY_DAYS", "10.0"))
EVIDENCE_MATURITY_SCALE = float(os.environ.get("EDNET_STAGE1_EVIDENCE_MATURITY_SCALE", "20.0"))
TAG_PRIOR_KAPPA = float(os.environ.get("EDNET_STAGE1_TAG_PRIOR_KAPPA", "20.0"))
ITEM_PRIOR_KAPPA = float(os.environ.get("EDNET_STAGE1_ITEM_PRIOR_KAPPA", "50.0"))

OBSERVATION_HORIZON_DAYS = float(os.environ.get("EDNET_STAGE1_OBSERVATION_HORIZON_DAYS", "7.0"))
LONG_GAP_DAYS = float(os.environ.get("EDNET_STAGE1_LONG_GAP_DAYS", "7.0"))

RESPONSE_DURATION_HALF_SAT_MIN = float(os.environ.get("EDNET_STAGE1_RESPONSE_DURATION_HALF_SAT_MIN", "3.0"))
EXPLANATION_HALF_SAT_MIN = float(os.environ.get("EDNET_STAGE1_EXPLANATION_HALF_SAT_MIN", "2.5"))
LECTURE_HALF_SAT_MIN = float(os.environ.get("EDNET_STAGE1_LECTURE_HALF_SAT_MIN", "4.0"))
IDLE_HALF_SAT_DAYS = float(os.environ.get("EDNET_STAGE1_IDLE_HALF_SAT_DAYS", "1.0"))
MAX_SUPPORT_EPISODE_ACTIVE = float(os.environ.get("EDNET_STAGE1_MAX_SUPPORT_EPISODE_ACTIVE", "1.0"))

GRID_BINS_SIGNED = np.linspace(-1.0, 1.0, int(os.environ.get("EDNET_STAGE1_SIGNED_GRID_N", "41")))
MIN_STATE_BIN_COUNT = int(os.environ.get("EDNET_STAGE1_MIN_STATE_BIN_COUNT", "50"))
MIN_DRIFT_BIN_COUNT = int(os.environ.get("EDNET_STAGE1_MIN_DRIFT_BIN_COUNT", "30"))
MIN_CELL_USERS = int(os.environ.get("EDNET_STAGE1_MIN_CELL_USERS", "5"))
CONVERGENCE_SPEED_QUANTILE = float(os.environ.get("EDNET_STAGE1_CONVERGENCE_SPEED_QUANTILE", "0.60"))
CONVERGENCE_NEGATIVE_DIVERGENCE_QUANTILE = float(os.environ.get("EDNET_STAGE1_CONVERGENCE_NEGATIVE_DIVERGENCE_QUANTILE", "0.80"))
CONVERGENCE_RATIO_QUANTILE = float(os.environ.get("EDNET_STAGE1_CONVERGENCE_RATIO_QUANTILE", "0.60"))
CONVERGENCE_MIN_CELLS = int(os.environ.get("EDNET_STAGE1_CONVERGENCE_MIN_CELLS", "4"))
CONVERGENCE_SHELL_RADIUS = float(os.environ.get("EDNET_STAGE1_CONVERGENCE_SHELL_RADIUS", "0.35"))

MACROSTATE_K = 6
KMEANS_N_INIT = int(os.environ.get("EDNET_STAGE1_KMEANS_N_INIT", "20"))
KMEANS_FIT_MAX_ROWS = int(os.environ.get("EDNET_STAGE1_KMEANS_FIT_MAX_ROWS", "500000"))
RESIDENCE_REFERENCE_LENGTH = int(os.environ.get("EDNET_STAGE1_RESIDENCE_REFERENCE_LENGTH", "10"))
MIN_RESIDENCE_AT_RISK = int(os.environ.get("EDNET_STAGE1_MIN_RESIDENCE_AT_RISK", "20"))
MAX_RESIDENCE_LENGTH = int(os.environ.get("EDNET_STAGE1_MAX_RESIDENCE_LENGTH", "10000"))
RESIDENCE_SELF_TRANSITION_THRESHOLD = float(os.environ.get("EDNET_STAGE1_RESIDENCE_MIN_SELF_TRANSITION", "0.55"))
RESIDENCE_P_THRESHOLD = float(os.environ.get("EDNET_STAGE1_RESIDENCE_P", "0.05"))

EPS = 1e-12

# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def ensure_dirs() -> None:
    for path in [
        OUTPUT_ROOT,
        STAGE1_ROOT,
        DYN_ROOT,
        META_OUT_ROOT,
        SPLIT_ROOT,
        RAW_ROOT,
        COORD_ROOT,
        REGION_ROOT,
        MESOSTATE_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def read_table(base: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    for ext in (".parquet", ".csv.gz", ".csv"):
        p = base.with_suffix(ext)
        if p.exists():
            if ext == ".parquet":
                return pd.read_parquet(p, columns=list(columns) if columns is not None else None)
            return pd.read_csv(p, usecols=list(columns) if columns is not None else None)
    raise FileNotFoundError(f"Could not find table for {base}")


def read_path_table(path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    return pd.read_csv(path, usecols=list(columns) if columns is not None else None)


def read_path_table_with_optional(path: Path, required: Sequence[str], optional: Sequence[str]) -> pd.DataFrame:
    cols = list(required) + list(optional)
    try:
        return read_path_table(path, columns=cols)
    except Exception:
        return read_path_table(path, columns=required)


def write_table(df: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        p = base.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        return p
    except Exception:
        p = base.with_suffix(".csv.gz")
        df.to_csv(p, index=False, compression="gzip")
        return p


def remove_table_outputs(base: Path) -> None:
    for ext in (".parquet", ".csv.gz", ".csv"):
        p = base.with_suffix(ext)
        if p.exists():
            p.unlink()


def append_table_csv_gz(df: pd.DataFrame, base: Path, first: bool, columns: Optional[List[str]] = None) -> Tuple[Path, List[str]]:
    base.parent.mkdir(parents=True, exist_ok=True)
    p = base.with_suffix(".csv.gz")
    if columns is None:
        columns = list(df.columns)
    aligned = df.reindex(columns=columns)
    with gzip.open(p, "at", encoding="utf-8", newline="") as fh:
        aligned.to_csv(fh, index=False, header=first)
    return p, columns


def sorted_chunk_paths(root: Path, prefix: str) -> List[Path]:
    paths = list(root.glob(f"{prefix}_chunk_*.parquet")) + list(root.glob(f"{prefix}_chunk_*.csv.gz")) + list(root.glob(f"{prefix}_chunk_*.csv"))
    rx = re.compile(r"chunk_(\d+)")
    return sorted(paths, key=lambda p: int(rx.search(p.name).group(1)) if rx.search(p.name) else -1)


def kt4_subdir(subdir: str) -> Path:
    return KT4_ROOT / subdir


def sorted_chunk_paths_kt4(subdir: str, prefix: str) -> List[Path]:
    return sorted_chunk_paths(kt4_subdir(subdir), prefix)


def chunk_index_from_path(path: Path) -> int:
    m = re.search(r"chunk_(\d+)", path.name)
    if not m:
        raise ValueError(f"Could not parse chunk index from {path}")
    return int(m.group(1))


def clip01(x: float) -> float:
    if x is None or pd.isna(x):
        return np.nan
    return float(min(1.0, max(0.0, float(x))))


def safe_float(x, default: float = 0.0) -> float:
    if x is None or pd.isna(x):
        return float(default)
    try:
        y = float(x)
    except Exception:
        return float(default)
    return y if np.isfinite(y) else float(default)


def safe_int(x, default: int = 0) -> int:
    if x is None or pd.isna(x):
        return int(default)
    try:
        return int(x)
    except Exception:
        return int(default)


def safe_ts(x) -> Optional[int]:
    if x is None or pd.isna(x):
        return None
    try:
        return int(x)
    except Exception:
        return None


def parse_tags(value) -> Tuple[int, ...]:
    if value is None or pd.isna(value):
        return tuple()
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = value
    else:
        txt = str(value).strip()
        if not txt:
            return tuple()
        vals = re.split(r"[;,\s]+", txt)
    out: List[int] = []
    for v in vals:
        s = str(v).strip()
        if not s or s == "-1":
            continue
        try:
            out.append(int(float(s)))
        except Exception:
            continue
    return tuple(sorted(set(out)))


def saturating_minutes(ms: Optional[float], half_sat_min: float) -> float:
    if ms is None or pd.isna(ms) or float(ms) <= 0:
        return 0.0
    minutes = float(ms) / 60000.0
    return float(minutes / (minutes + max(float(half_sat_min), EPS)))


def idle_units_from_gap_ms(ms: Optional[float]) -> float:
    if ms is None or pd.isna(ms) or float(ms) <= 0:
        return 0.0
    half = max(float(IDLE_HALF_SAT_DAYS) * DAY_MS, EPS)
    x = float(ms)
    return float(x / (x + half))


def cosine_dense(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return np.nan
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na <= EPS or nb <= EPS:
        return np.nan
    val = float(np.dot(aa, bb) / (na * nb))
    return float(np.clip(val, -1.0, 1.0))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(float(p.sum()), eps)
    q = q / max(float(q.sum()), eps)
    m = 0.5 * (p + q)
    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + eps) / (b[mask] + eps))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def row_value(row, name: str, default=np.nan):
    return getattr(row, name, default)


# -----------------------------------------------------------------------------
# Content lookups
# -----------------------------------------------------------------------------
@dataclass
class ContentLookups:
    tag_to_idx: Dict[int, int]
    idx_to_tag: Dict[int, int]
    question_tag_idx: Dict[str, np.ndarray]
    bundle_tag_weights: Dict[str, Dict[int, float]]
    bundle_tag_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]]
    bundle_n_questions: Dict[str, int]
    bundle_part: Dict[str, Optional[int]]
    bundle_explanation: Dict[str, Optional[str]]
    explanation_tags: Dict[str, Tuple[int, ...]]
    lecture_tags: Dict[str, Tuple[int, ...]]


class ContentsBuilder:
    def __init__(self) -> None:
        self.questions = read_table(CONTENTS_ROOT / "questions_clean")
        self.bundles = read_table(CONTENTS_ROOT / "bundles_clean")
        self.lectures = read_table(CONTENTS_ROOT / "lectures_clean")
        self.explanations = read_table(CONTENTS_ROOT / "explanations_clean")

    def build(self) -> ContentLookups:
        question_tags = {
            str(row.question_id): parse_tags(getattr(row, "tags_raw", None))
            for row in self.questions.itertuples(index=False)
        }
        vocabulary = sorted({tag for tags in question_tags.values() for tag in tags})
        tag_to_idx = {int(tag): index for index, tag in enumerate(vocabulary)}
        idx_to_tag = {index: int(tag) for tag, index in tag_to_idx.items()}

        question_tag_idx: Dict[str, np.ndarray] = {}
        bundle_tag_weights: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        bundle_n_questions: Dict[str, int] = defaultdict(int)
        bundle_part: Dict[str, Optional[int]] = {}
        bundle_explanation: Dict[str, Optional[str]] = {}

        for row in self.questions.itertuples(index=False):
            question_id = str(row.question_id)
            bundle_id = str(row.bundle_id)
            tags = question_tags.get(question_id, tuple())
            indices = np.asarray([tag_to_idx[tag] for tag in tags if tag in tag_to_idx], dtype=np.int32)
            question_tag_idx[question_id] = indices
            bundle_n_questions[bundle_id] += 1
            if bundle_id not in bundle_part:
                bundle_part[bundle_id] = int(row.part) if hasattr(row, "part") and pd.notna(row.part) else None
            if bundle_id not in bundle_explanation:
                value = getattr(row, "explanation_id", np.nan)
                bundle_explanation[bundle_id] = str(value) if pd.notna(value) else None
            if indices.size:
                weight = 1.0 / float(indices.size)
                for tag in tags:
                    if tag in tag_to_idx:
                        bundle_tag_weights[bundle_id][int(tag)] += weight

        bundle_tag_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for bundle_id, weights in list(bundle_tag_weights.items()):
            total = float(sum(weights.values()))
            normalized = {int(tag): float(value / total) for tag, value in weights.items()} if total > 0 else {}
            bundle_tag_weights[bundle_id] = normalized
            indices = np.asarray([tag_to_idx[tag] for tag in normalized if tag in tag_to_idx], dtype=np.int32)
            values = np.asarray([normalized[tag] for tag in normalized if tag in tag_to_idx], dtype=float)
            bundle_tag_arrays[bundle_id] = (indices, values)

        if "bundle_id" in self.bundles.columns:
            for bundle_id in self.bundles["bundle_id"].astype(str):
                bundle_tag_weights.setdefault(bundle_id, {})
                bundle_tag_arrays.setdefault(
                    bundle_id,
                    (np.empty(0, dtype=np.int32), np.empty(0, dtype=float)),
                )
                bundle_n_questions.setdefault(bundle_id, 0)
                bundle_part.setdefault(bundle_id, None)
                bundle_explanation.setdefault(bundle_id, None)

        explanation_tags: Dict[str, Tuple[int, ...]] = {}
        if {"explanation_id", "tags_raw"}.issubset(self.explanations.columns):
            for row in self.explanations.itertuples(index=False):
                explanation_tags[str(row.explanation_id)] = tuple(
                    tag for tag in parse_tags(row.tags_raw) if tag in tag_to_idx
                )

        lecture_tags: Dict[str, Tuple[int, ...]] = {}
        for row in self.lectures.itertuples(index=False):
            lecture_id = str(row.lecture_id)
            lecture_tags[lecture_id] = tuple(
                tag for tag in parse_tags(getattr(row, "tags_raw", None)) if tag in tag_to_idx
            )

        return ContentLookups(
            tag_to_idx=tag_to_idx,
            idx_to_tag=idx_to_tag,
            question_tag_idx=question_tag_idx,
            bundle_tag_weights={key: dict(value) for key, value in bundle_tag_weights.items()},
            bundle_tag_arrays=bundle_tag_arrays,
            bundle_n_questions=dict(bundle_n_questions),
            bundle_part=bundle_part,
            bundle_explanation=bundle_explanation,
            explanation_tags=explanation_tags,
            lecture_tags=lecture_tags,
        )


def dense_bundle_vector(content: ContentLookups, bundle_id: str) -> np.ndarray:
    x = np.zeros(len(content.tag_to_idx), dtype=float)
    idxs, vals = content.bundle_tag_arrays.get(str(bundle_id), (np.empty(0, dtype=np.int32), np.empty(0, dtype=float)))
    if idxs.size:
        x[idxs] = vals
    return x


def dense_from_tag_indices(K: int, idxs: np.ndarray, mass: float = 1.0) -> np.ndarray:
    x = np.zeros(K, dtype=float)
    if idxs is None or len(idxs) == 0:
        return x
    w = float(mass) / max(float(len(idxs)), 1.0)
    x[np.asarray(idxs, dtype=np.int32)] += w
    return x


# -----------------------------------------------------------------------------
# Splits and priors
# -----------------------------------------------------------------------------
def load_user_ids() -> np.ndarray:
    paths = sorted_chunk_paths_kt4("user_summary", "user_summary")
    ids = set()
    if paths:
        for p in paths:
            df = read_path_table(p, columns=["user_id"])
            ids.update(df["user_id"].dropna().astype(np.int64).tolist())
    else:
        for p in sorted_chunk_paths_kt4("bundle_attempts", "bundle_attempts"):
            df = read_path_table(p, columns=["user_id"])
            ids.update(df["user_id"].dropna().astype(np.int64).tolist())
    return np.asarray(sorted(ids), dtype=np.int64)


def create_splits(all_user_ids: np.ndarray) -> Dict[str, np.ndarray]:
    total_needed = A_TRAIN_USERS + A_VAL_USERS + B_CONFIRM_USERS
    rng = np.random.default_rng(RANDOM_STATE)
    perm = all_user_ids.copy()
    rng.shuffle(perm)
    if len(perm) < total_needed:
        if not ALLOW_SMALL_DEV_SPLIT:
            raise ValueError(
                f"Need {total_needed} users for the publication split; found {len(perm)}. "
                "Set EDNET_STAGE1_ALLOW_SMALL_DEV_SPLIT=1 for smoke tests only."
            )
        n = len(perm)
        n_train = max(1, int(round(0.60 * n)))
        n_val = max(1, int(round(0.20 * n))) if n - n_train > 1 else max(0, n - n_train)
        n_confirm = max(0, n - n_train - n_val)
        return {
            "A_train": np.sort(perm[:n_train]),
            "A_val": np.sort(perm[n_train:n_train+n_val]),
            "B_confirm": np.sort(perm[n_train+n_val:n_train+n_val+n_confirm]),
        }
    if len(perm) != total_needed:
        raise ValueError(
            f"Configured split sizes cover {total_needed} users, but preprocessing produced {len(perm)}. "
            "Set the three split-size environment variables so every user is assigned."
        )
    return {
        "A_train": np.sort(perm[:A_TRAIN_USERS]),
        "A_val": np.sort(perm[A_TRAIN_USERS:A_TRAIN_USERS + A_VAL_USERS]),
        "B_confirm": np.sort(perm[A_TRAIN_USERS + A_VAL_USERS:A_TRAIN_USERS + A_VAL_USERS + B_CONFIRM_USERS]),
    }


def verify_user_complete_chunks(max_examples: int = 50) -> Dict[str, object]:
    paths = sorted_chunk_paths_kt4("bundle_attempts", "bundle_attempts")
    seen: Dict[int, int] = {}
    overlaps: List[dict] = []
    for p in paths:
        ci = chunk_index_from_path(p)
        df = read_path_table(p, columns=["user_id"])
        for uid in df["user_id"].dropna().astype(np.int64).unique().tolist():
            u = int(uid)
            if u in seen:
                if len(overlaps) < max_examples:
                    overlaps.append({"user_id": u, "first_chunk": int(seen[u]), "later_chunk": int(ci)})
            else:
                seen[u] = int(ci)
    audit = {
        "bundle_chunk_count": int(len(paths)),
        "unique_users_seen": int(len(seen)),
        "overlap_example_count": int(len(overlaps)),
        "overlap_examples": overlaps,
        "gate_passed": bool(len(overlaps) == 0),
        "reason": "The panel builder maintains user state inside chunk; chunks must be user-complete.",
    }
    with open(META_OUT_ROOT / "bundle_chunk_user_completeness_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    if overlaps:
        raise RuntimeError("bundle_attempts chunks are not user-complete. Re-preprocess with user-complete chunks or add cross-chunk state carryover.")
    return audit


@dataclass
class TagPriorEstimate:
    priors: np.ndarray
    table: pd.DataFrame
    global_prior: float
    question_priors: Dict[str, float]
    question_table: pd.DataFrame


def estimate_tag_priors(content: ContentLookups, train_users: np.ndarray) -> TagPriorEstimate:
    """Estimate A_train-only tag and item empirical-Bayes baselines."""
    train_set = set(int(u) for u in train_users.tolist())
    K = len(content.tag_to_idx)
    tag_correct = np.zeros(K, dtype=float)
    tag_count = np.zeros(K, dtype=float)
    question_correct: Dict[str, float] = defaultdict(float)
    question_count: Dict[str, float] = defaultdict(float)
    q_paths = sorted_chunk_paths_kt4("question_attempts", "question_attempts")
    cols = ["user_id", "question_id", "is_correct"]
    for pth in tqdm(q_paths, desc="Estimating A_train itemEB/tag correctness priors", unit="chunk"):
        df = read_path_table(pth, columns=cols)
        df = df[df["user_id"].isin(train_set)]
        if df.empty:
            continue
        for r in df.itertuples(index=False):
            if pd.isna(r.is_correct):
                continue
            qid = str(r.question_id)
            c = float(r.is_correct)
            question_correct[qid] += c
            question_count[qid] += 1.0
            idxs = content.question_tag_idx.get(qid)
            if idxs is None or idxs.size == 0:
                continue
            w = 1.0 / float(idxs.size)
            tag_correct[idxs] += w * c
            tag_count[idxs] += w

    global_prior = float(tag_correct.sum() / max(tag_count.sum(), EPS)) if tag_count.sum() > 0 else 0.5
    global_prior = float(np.clip(global_prior, 1e-3, 1 - 1e-3))
    tag_priors = (TAG_PRIOR_KAPPA * global_prior + tag_correct) / np.maximum(TAG_PRIOR_KAPPA + tag_count, EPS)
    tag_priors = np.clip(tag_priors, 1e-3, 1 - 1e-3)

    def parent_prior_for_question(qid: str) -> Tuple[float, str]:
        idxs = content.question_tag_idx.get(str(qid))
        if idxs is not None and idxs.size > 0:
            return float(np.mean(tag_priors[idxs])), "mean_Atrain_tag_prior"
        return float(global_prior), "global_prior"

    question_priors: Dict[str, float] = {}
    q_rows: List[dict] = []
    # Cover all known questions so raw panel building has deterministic fallback.
    all_qids = sorted(set(content.question_tag_idx.keys()) | set(question_count.keys()))
    for qid in all_qids:
        parent, parent_scope = parent_prior_for_question(qid)
        cnt = float(question_count.get(qid, 0.0))
        corr = float(question_correct.get(qid, 0.0))
        q_prior = (ITEM_PRIOR_KAPPA * parent + corr) / max(ITEM_PRIOR_KAPPA + cnt, EPS)
        q_prior = float(np.clip(q_prior, 1e-3, 1 - 1e-3))
        question_priors[str(qid)] = q_prior
        q_rows.append({
            "question_id": str(qid),
            "prior_correct_itemEB_Atrain": q_prior,
            "response_correct_count_Atrain": corr,
            "response_count_Atrain": cnt,
            "item_eb_parent_prior": parent,
            "item_eb_parent_scope": parent_scope,
            "item_eb_kappa": float(ITEM_PRIOR_KAPPA),
            "semantic_note": "A_train-only item/question baseline for signed response residual Y-q; not estimated from A_val or B_confirm.",
        })

    tag_table = pd.DataFrame({
        "tag": [content.idx_to_tag[i] for i in range(K)],
        "prior_correct_Atrain_shrunk": tag_priors,
        "response_correct_mass_Atrain": tag_correct,
        "response_count_mass_Atrain": tag_count,
        "global_prior_correct_Atrain": global_prior,
        "prior_kappa": TAG_PRIOR_KAPPA,
        "semantic_note": "A_train-only tag baseline retained as robustness/control; primary M_R uses itemEB question priors.",
    })
    question_table = pd.DataFrame(q_rows)
    return TagPriorEstimate(
        priors=tag_priors,
        table=tag_table,
        global_prior=global_prior,
        question_priors=question_priors,
        question_table=question_table,
    )

# -----------------------------------------------------------------------------
# User-level empirical state construction
# -----------------------------------------------------------------------------
@dataclass
class ActivitySnapshot:
    active_mass: float
    aligned_mass: float
    off_target_mass: float
    neutral_active_mass: float
    idle_mass: float
    alignment_order_Psi: float


class ActivityMemory:
    """Store decayed aligned, off-target, neutral, and idle activity mass."""

    def __init__(self, tau_days: float = TAU_ACTIVITY_DAYS) -> None:
        self.tau_days = float(tau_days)
        self.aligned = 0.0
        self.off = 0.0
        self.neutral = 0.0
        self.idle = 0.0

    def decay(self, delta_days: float) -> None:
        if self.tau_days <= 0:
            return
        f = math.exp(-max(float(delta_days), 0.0) / self.tau_days)
        self.aligned *= f
        self.off *= f
        self.neutral *= f
        self.idle *= f

    def add_active_by_alignment(self, active_mass: float, alignment: float) -> Tuple[float, float, float]:
        a = max(float(active_mass), 0.0)
        if a <= 0:
            return 0.0, 0.0, 0.0
        if np.isfinite(alignment):
            q = clip01(float(alignment))
            al = a * q
            off = a * (1.0 - q)
            self.aligned += al
            self.off += off
            return al, off, 0.0
        self.neutral += a
        return 0.0, 0.0, a

    def add_idle(self, idle_mass: float) -> None:
        if idle_mass is None or pd.isna(idle_mass):
            return
        self.idle += max(float(idle_mass), 0.0)

    def snapshot(self) -> ActivitySnapshot:
        active = float(self.aligned + self.off + self.neutral)
        total = active + self.idle
        psi = float((self.aligned - self.off) / total) if total > EPS else np.nan
        return ActivitySnapshot(
            active_mass=active,
            aligned_mass=float(self.aligned),
            off_target_mass=float(self.off),
            neutral_active_mass=float(self.neutral),
            idle_mass=float(self.idle),
            alignment_order_Psi=psi,
        )


@dataclass
class EmpiricalState:
    metrics: Dict[str, float]
    demand_vector: np.ndarray


class UserEmpiricalDynamicsBuilder:
    """Build submitted-bundle intervals and the M/Psi state for one user."""

    def __init__(self, content: ContentLookups, priors: TagPriorEstimate) -> None:
        self.content = content
        self.tag_priors = np.asarray(priors.priors, dtype=float)
        self.question_priors = dict(priors.question_priors)
        self.global_prior = float(priors.global_prior)
        self.K = len(self.tag_priors)

    def _question_prior(self, qid: str, idxs: np.ndarray) -> float:
        q = self.question_priors.get(str(qid))
        if q is not None and np.isfinite(float(q)):
            return float(q)
        if idxs is not None and idxs.size > 0:
            return float(np.clip(np.mean(self.tag_priors[idxs]), 1e-3, 1.0 - 1e-3))
        return float(self.global_prior)

    def _state(
        self,
        signed_item: np.ndarray,
        abs_item: np.ndarray,
        activity: ActivityMemory,
    ) -> EmpiricalState:
        evidence_mass = float(abs_item.sum())
        response_order = float(signed_item.sum() / evidence_mass) if evidence_mass > EPS else np.nan
        maturity = (
            float(1.0 - math.exp(-evidence_mass / max(EVIDENCE_MATURITY_SCALE, EPS)))
            if evidence_mass > EPS
            else 0.0
        )
        demand = np.maximum(-signed_item, 0.0)
        snapshot = activity.snapshot()
        metrics = {
            "M_response_prebalanced": response_order,
            "response_evidence_maturity_V": maturity,
            "response_evidence_mass": evidence_mass,
            "unresolved_response_demand_mass": float(demand.sum()),
            "activity_alignment_order_Psi": snapshot.alignment_order_Psi,
            "activity_active_mass": snapshot.active_mass,
            "activity_aligned_mass": snapshot.aligned_mass,
            "activity_off_target_mass": snapshot.off_target_mass,
            "activity_non_aligned_mass": snapshot.off_target_mass,
            "activity_neutral_mass": snapshot.neutral_active_mass,
            "activity_idle_mass": snapshot.idle_mass,
        }
        return EmpiricalState(metrics=metrics, demand_vector=demand)

    def _phase_payload(self, suffix: str, state: EmpiricalState) -> Dict[str, float]:
        return {f"{k}_{suffix}": v for k, v in state.metrics.items()}

    def _agg_questions(self, question_rows: pd.DataFrame) -> Dict[int, dict]:
        output: Dict[int, dict] = {}
        if question_rows is None or question_rows.empty:
            return output
        for row in question_rows.itertuples(index=False):
            attempt_index = safe_int(row_value(row, "bundle_attempt_index"), default=-1)
            question_id = str(row_value(row, "question_id"))
            indices = self.content.question_tag_idx.get(question_id)
            if indices is None or indices.size == 0:
                continue
            correct = row_value(row, "is_correct", np.nan)
            if pd.isna(correct):
                continue
            accumulator = output.setdefault(
                attempt_index,
                {
                    "signed_item": np.zeros(self.K, dtype=float),
                    "abs_item": np.zeros(self.K, dtype=float),
                    "n_questions": 0,
                },
            )
            correct_value = float(correct)
            weight = 1.0 / float(indices.size)
            residual = correct_value - self._question_prior(question_id, indices)
            accumulator["signed_item"][indices] += weight * residual
            accumulator["abs_item"][indices] += weight * abs(residual)
            accumulator["n_questions"] += 1
        return output

    def _index_study(self, s_user: pd.DataFrame) -> Tuple[np.ndarray, List[dict]]:
        if s_user is None or s_user.empty:
            return np.asarray([], dtype=np.int64), []
        s = s_user.copy()
        if "enter_ts" not in s.columns:
            return np.asarray([], dtype=np.int64), []
        anchor = s["enter_ts"].where(pd.notna(s["enter_ts"]), s.get("quit_ts", np.nan))
        s["anchor_ts"] = anchor
        s = s[pd.notna(s["anchor_ts"])].sort_values(["anchor_ts", "study_episode_index" if "study_episode_index" in s.columns else "anchor_ts"], kind="mergesort")
        rows: List[dict] = []
        times: List[int] = []
        for r in s.itertuples(index=False):
            code = safe_int(row_value(r, "item_type_code"), default=-1)
            if code in {2, 3}:
                item_type = "explanation"
            elif code in {1, 4}:
                item_type = "lecture"
            else:
                item_type = "other"
            anchor_ts = int(row_value(r, "anchor_ts"))
            times.append(anchor_ts)
            rows.append({
                "item_id": str(row_value(r, "item_id")),
                "item_type": item_type,
                "enter_ts": safe_ts(row_value(r, "enter_ts", np.nan)),
                "quit_ts": safe_ts(row_value(r, "quit_ts", np.nan)),
                "dwell_ms": safe_float(row_value(r, "dwell_ms", np.nan), default=np.nan),
                "dwell_ratio": safe_float(row_value(r, "dwell_ratio", np.nan), default=np.nan),
                "video_length_ms": safe_float(row_value(r, "video_length_ms", np.nan), default=np.nan),
                "related_bundle_id": str(row_value(r, "related_bundle_id")) if pd.notna(row_value(r, "related_bundle_id", np.nan)) else None,
                "media_elapsed_ms": safe_float(row_value(r, "media_elapsed_ms", np.nan), default=np.nan),
                "media_event_count": safe_int(row_value(r, "media_event_count", np.nan), default=0),
                "media_pair_anomaly_count": safe_int(row_value(r, "media_pair_anomaly_count", np.nan), default=0),
                "anchor_ts": anchor_ts,
            })
        return np.asarray(times, dtype=np.int64), rows

    @staticmethod
    def _episode_end_ts(ev: dict) -> Optional[int]:
        enter = ev.get("enter_ts") if ev.get("enter_ts") is not None else ev.get("anchor_ts")
        quit_ts = ev.get("quit_ts")
        if quit_ts is not None:
            return int(quit_ts)
        if enter is None:
            return None
        dwell = safe_float(ev.get("dwell_ms"), default=np.nan)
        if np.isfinite(dwell) and dwell > 0:
            return int(enter + dwell)
        media = safe_float(ev.get("media_elapsed_ms"), default=np.nan)
        if np.isfinite(media) and media > 0:
            return int(enter + media)
        return int(enter)

    def _support_episode_strength_and_vector(self, ev: dict, current_bundle_id: str) -> Tuple[float, np.ndarray, float, int]:
        """Return active strength, tag vector, raw active milliseconds, and unmapped flag."""
        item_type = ev.get("item_type")
        if item_type not in {"explanation", "lecture"}:
            return 0.0, np.zeros(self.K, dtype=float), 0.0, 1
        dwell_ms = safe_float(ev.get("dwell_ms"), default=np.nan)
        media_ms = safe_float(ev.get("media_elapsed_ms"), default=np.nan)
        raw_ms = 0.0
        if np.isfinite(dwell_ms) and dwell_ms > 0:
            raw_ms = max(raw_ms, float(dwell_ms))
        if np.isfinite(media_ms) and media_ms > 0:
            raw_ms = max(raw_ms, float(media_ms))
        if item_type == "explanation":
            strength = max(saturating_minutes(dwell_ms, EXPLANATION_HALF_SAT_MIN), saturating_minutes(media_ms, EXPLANATION_HALF_SAT_MIN))
            related = ev.get("related_bundle_id")
            item_id = ev.get("item_id")
            tags: Tuple[int, ...]
            if related == current_bundle_id or self.content.bundle_explanation.get(current_bundle_id) == item_id:
                tags = tuple(self.content.bundle_tag_weights.get(current_bundle_id, {}).keys())
            else:
                tags = self.content.explanation_tags.get(str(item_id), tuple())
        else:
            dwell_ratio = safe_float(ev.get("dwell_ratio"), default=np.nan)
            if np.isfinite(dwell_ratio):
                dwell_strength = clip01(dwell_ratio)
            else:
                dwell_strength = saturating_minutes(dwell_ms, LECTURE_HALF_SAT_MIN)
            length_ms = safe_float(ev.get("video_length_ms"), default=np.nan)
            if np.isfinite(media_ms) and media_ms > 0 and np.isfinite(length_ms) and length_ms > 0:
                media_strength = clip01(media_ms / max(length_ms, EPS))
            else:
                media_strength = saturating_minutes(media_ms, LECTURE_HALF_SAT_MIN)
            strength = max(dwell_strength, media_strength)
            tags = self.content.lecture_tags.get(str(ev.get("item_id")), tuple())
        strength = float(min(max(strength, 0.0), MAX_SUPPORT_EPISODE_ACTIVE))
        if strength <= 0:
            return 0.0, np.zeros(self.K, dtype=float), raw_ms, 1
        idxs = np.asarray([self.content.tag_to_idx[t] for t in tags if t in self.content.tag_to_idx], dtype=np.int32)
        if idxs.size == 0:
            return strength, np.zeros(self.K, dtype=float), raw_ms, 1
        return strength, dense_from_tag_indices(self.K, idxs, mass=strength), raw_ms, 0

    def _support_window(self, bundle_id: str, study_slice: List[dict]) -> Dict[str, object]:
        z = np.zeros(self.K, dtype=float)
        active_total = 0.0
        active_mapped = 0.0
        active_unmapped = 0.0
        raw_ms_total = 0.0
        n_ep = 0
        media_events = 0
        pair_anom = 0
        for ev in study_slice:
            strength, vec, raw_ms, unmapped = self._support_episode_strength_and_vector(ev, bundle_id)
            if strength <= 0:
                continue
            n_ep += 1
            media_events += safe_int(ev.get("media_event_count"), default=0)
            pair_anom += safe_int(ev.get("media_pair_anomaly_count"), default=0)
            raw_ms_total += max(raw_ms, 0.0)
            active_total += strength
            if unmapped:
                active_unmapped += strength
            else:
                active_mapped += strength
                z += vec
        return {
            "support_vector": z,
            "support_active_total": float(active_total),
            "support_active_mapped": float(active_mapped),
            "support_active_unmapped": float(active_unmapped),
            "support_raw_active_ms": float(raw_ms_total),
            "support_episode_count": int(n_ep),
            "support_media_event_count": int(media_events),
            "support_media_pair_anomaly_count": int(pair_anom),
        }

    def _study_slice_between(self, study_times: np.ndarray, study_rows: List[dict], start_ts: Optional[int], end_ts: Optional[int]) -> List[dict]:
        if start_ts is None or end_ts is None or study_times.size == 0:
            return []
        lo = int(np.searchsorted(study_times, int(start_ts), side="left"))
        hi = int(np.searchsorted(study_times, int(end_ts), side="left"))
        return study_rows[lo:hi]

    def build_user(self, user_id: int, b_user: pd.DataFrame, q_user: pd.DataFrame, s_user: pd.DataFrame, split_name: str) -> List[dict]:
        if b_user is None or b_user.empty:
            return []
        b = b_user.copy()
        b["anchor_pre_ts"] = b["enter_ts"].where(pd.notna(b["enter_ts"]), b["submit_ts"])
        b = b[pd.notna(b["submit_ts"])].sort_values(["anchor_pre_ts", "bundle_attempt_index"], kind="mergesort").reset_index(drop=True)
        if b.empty:
            return []
        b["next_anchor_pre_ts"] = b["anchor_pre_ts"].shift(-1)
        q_agg = self._agg_questions(q_user)
        study_times, study_rows = self._index_study(s_user)

        signed_item = np.zeros(self.K, dtype=float)
        abs_item = np.zeros(self.K, dtype=float)
        activity = ActivityMemory(tau_days=TAU_ACTIVITY_DAYS)
        prev_pre_ts: Optional[int] = None
        rows: List[dict] = []

        for step_idx, row in enumerate(b.itertuples(index=False), start=1):
            bundle_id = str(row.bundle_id)
            ba = safe_int(row_value(row, "bundle_attempt_index"), default=-1)
            pre_ts = safe_ts(row_value(row, "anchor_pre_ts"))
            submit_ts = safe_ts(row_value(row, "submit_ts"))
            next_pre_ts = safe_ts(row_value(row, "next_anchor_pre_ts"))
            if prev_pre_ts is not None and pre_ts is not None:
                delta_days = max(0.0, (pre_ts - prev_pre_ts) / DAY_MS)
                if TAU_RESPONSE_DAYS > 0:
                    f_r = math.exp(-delta_days / TAU_RESPONSE_DAYS)
                    signed_item *= f_r
                    abs_item *= f_r
                activity.decay(delta_days)

            state_pre = self._state(signed_item, abs_item, activity)
            demand_pre = state_pre.demand_vector
            bundle_vec = dense_bundle_vector(self.content, bundle_id)
            response_alignment = cosine_dense(bundle_vec, demand_pre)
            qrow = q_agg.get(ba)
            n_questions_meta = max(int(self.content.bundle_n_questions.get(bundle_id, 0)), 1)
            answered = safe_float(row_value(row, "answered_question_count", np.nan), default=np.nan)
            total_response_count = safe_float(row_value(row, "total_response_count", np.nan), default=np.nan)
            if not np.isfinite(answered):
                answered = total_response_count if np.isfinite(total_response_count) else float(qrow["n_questions"] if qrow else 0)
            answered_fraction = clip01(answered / max(float(n_questions_meta), 1.0)) if np.isfinite(answered) else 0.0
            duration_ms = safe_float(row_value(row, "duration_ms", np.nan), default=np.nan)
            duration_active = saturating_minutes(duration_ms, RESPONSE_DURATION_HALF_SAT_MIN)
            response_active = max(answered_fraction, 0.5 * duration_active if duration_active > 0 else 0.0)
            r_aligned, r_off, r_neutral = activity.add_active_by_alignment(response_active, response_alignment)

            if qrow is not None:
                signed_item += qrow["signed_item"]
                abs_item += qrow["abs_item"]
            state_resp = self._state(signed_item, abs_item, activity)

            study_slice = self._study_slice_between(study_times, study_rows, submit_ts, next_pre_ts)
            sw = self._support_window(bundle_id, study_slice)
            support_vec = sw["support_vector"]
            support_target = demand_pre + bundle_vec
            support_alignment = cosine_dense(support_vec, support_target)
            s_aligned = s_off = s_neutral = 0.0
            if sw["support_active_total"] > 0:
                if np.isfinite(support_alignment):
                    s_aligned, s_off, s_neutral = activity.add_active_by_alignment(float(sw["support_active_mapped"]), support_alignment)
                    if float(sw["support_active_unmapped"]) > 0:
                        _, _, neu = activity.add_active_by_alignment(float(sw["support_active_unmapped"]), np.nan)
                        s_neutral += neu
                else:
                    _, _, s_neutral = activity.add_active_by_alignment(float(sw["support_active_total"]), np.nan)
            state_support = self._state(signed_item, abs_item, activity)

            uncovered_idle_ms = 0.0
            if submit_ts is not None and next_pre_ts is not None and next_pre_ts > submit_ts:
                raw_support_ms = min(float(sw["support_raw_active_ms"]), float(next_pre_ts - submit_ts))
                uncovered_idle_ms = max(0.0, float(next_pre_ts - submit_ts) - raw_support_ms)
                activity.add_idle(idle_units_from_gap_ms(uncovered_idle_ms))
            state_post = self._state(signed_item, abs_item, activity)

            row_out: Dict[str, object] = {
                "split": split_name,
                "user_id": int(user_id),
                "bundle_step_index": int(step_idx),
                "bundle_attempt_index": int(ba),
                "bundle_id": bundle_id,
                "part": int(self.content.bundle_part.get(bundle_id)) if self.content.bundle_part.get(bundle_id) is not None else -1,
                "source_code": safe_int(row_value(row, "source_code", np.nan), default=-1),
                "platform_code": safe_int(row_value(row, "platform_code", np.nan), default=-1),
                "pre_ts": pre_ts,
                "submit_ts": submit_ts,
                "current_accuracy_diagnostic_only": safe_float(row_value(row, "accuracy", np.nan), default=np.nan),
                "response_active_mass_interval": float(response_active),
                "response_alignment_to_pre_demand": float(response_alignment) if np.isfinite(response_alignment) else np.nan,
                "response_aligned_mass_interval": float(r_aligned),
                "response_off_target_mass_interval": float(r_off),
                "response_neutral_mass_interval": float(r_neutral),
                "support_alignment_to_pre_demand_or_current_bundle": float(support_alignment) if np.isfinite(support_alignment) else np.nan,
                "support_active_total_interval": float(sw["support_active_total"]),
                "support_active_mapped_interval": float(sw["support_active_mapped"]),
                "support_active_unmapped_interval": float(sw["support_active_unmapped"]),
                "support_aligned_mass_interval": float(s_aligned),
                "support_off_target_mass_interval": float(s_off),
                "support_neutral_mass_interval": float(s_neutral),
                "support_exposure_increment_mass": float(np.sum(support_vec)),
                "support_episode_count_interval": int(sw["support_episode_count"]),
                "support_media_event_count_interval": int(sw["support_media_event_count"]),
                "support_media_pair_anomaly_count_interval": int(sw["support_media_pair_anomaly_count"]),
                "idle_uncovered_ms_interval": float(uncovered_idle_ms),
                "idle_mass_interval": float(idle_units_from_gap_ms(uncovered_idle_ms)),
                "bundle_n_questions": int(n_questions_meta),
                "answered_fraction_interval": float(answered_fraction) if np.isfinite(answered_fraction) else np.nan,
                "response_duration_active_proxy": float(duration_active),
                "total_response_count_diagnostic": float(total_response_count) if np.isfinite(total_response_count) else np.nan,
                "response_change_count_diagnostic": safe_float(row_value(row, "total_response_change_count", np.nan), default=np.nan),
                "kt4_choice_process_count_diagnostic": safe_int(row_value(row, "total_choice_process_count", np.nan), default=0),
                "kt4_total_erase_count_diagnostic": safe_int(row_value(row, "total_erase_count", np.nan), default=0),
                "kt4_total_undo_erase_count_diagnostic": safe_int(row_value(row, "total_undo_erase_count", np.nan), default=0),
                "kt4_total_text_enter_count_diagnostic": safe_int(row_value(row, "total_text_enter_count", np.nan), default=0),
                "access_status_aux_context_only": safe_float(row_value(row, "access_status_aux", np.nan), default=np.nan),
                "payment_coupon_window_aux_context_only": safe_float(row_value(row, "payment_coupon_window_aux", np.nan), default=np.nan),
                "access_pay_event_count_context_only": safe_int(row_value(row, "access_pay_event_count", np.nan), default=0),
                "access_refund_event_count_context_only": safe_int(row_value(row, "access_refund_event_count", np.nan), default=0),
                "access_coupon_event_count_context_only": safe_int(row_value(row, "access_coupon_event_count", np.nan), default=0),
                "semantic_guardrail": "primary signed evidence is itemEB response residual Y-q_question estimated on A_train; support/access/process variables do not update signed response evidence; current/future accuracy changes are not used in activity coordinates",
            }
            row_out.update(self._phase_payload("pre", state_pre))
            row_out.update(self._phase_payload("resp", state_resp))
            row_out.update(self._phase_payload("support", state_support))
            row_out.update(self._phase_payload("post", state_post))
            rows.append(row_out)
            prev_pre_ts = pre_ts
        return rows


# -----------------------------------------------------------------------------
# Raw and finalized panels
# -----------------------------------------------------------------------------
def build_raw_panels(content: ContentLookups, priors: TagPriorEstimate, split_map: Dict[int, str]) -> List[Path]:
    builder = UserEmpiricalDynamicsBuilder(content, priors)
    out_paths: List[Path] = []
    bundle_paths = sorted_chunk_paths_kt4("bundle_attempts", "bundle_attempts")
    q_paths = {chunk_index_from_path(p): p for p in sorted_chunk_paths_kt4("question_attempts", "question_attempts")}
    s_paths = {chunk_index_from_path(p): p for p in sorted_chunk_paths_kt4("study_episodes", "study_episodes")}

    b_required = [
        "user_id", "bundle_attempt_index", "bundle_id", "enter_ts", "submit_ts", "source_code", "platform_code",
        "accuracy", "total_response_count", "total_response_change_count", "duration_ms",
    ]
    b_optional = [
        "answered_question_count", "total_erase_count", "total_undo_erase_count", "total_choice_process_count",
        "total_text_enter_count", "access_status_aux", "payment_coupon_window_aux",
        "access_pay_event_count", "access_refund_event_count", "access_coupon_event_count",
    ]
    q_cols = ["user_id", "bundle_attempt_index", "question_id", "is_correct"]
    s_required = ["user_id", "study_episode_index", "item_id", "item_type_code", "enter_ts", "quit_ts", "dwell_ms"]
    s_optional = [
        "dwell_ratio", "video_length_ms", "related_bundle_id", "media_event_count", "media_elapsed_ms",
        "media_pair_anomaly_count",
    ]

    for b_path in tqdm(bundle_paths, desc="Building empirical v3 raw panels", unit="chunk"):
        ci = chunk_index_from_path(b_path)
        b_df = read_path_table_with_optional(b_path, required=b_required, optional=b_optional)
        b_df = b_df[b_df["user_id"].map(split_map).notna()]
        if b_df.empty:
            continue
        users = b_df["user_id"].dropna().astype(np.int64).unique()
        q_df = read_path_table(q_paths[ci], columns=q_cols) if ci in q_paths else pd.DataFrame(columns=q_cols)
        if not q_df.empty:
            q_df = q_df[q_df["user_id"].isin(users)]
        s_df = read_path_table_with_optional(s_paths[ci], required=s_required, optional=s_optional) if ci in s_paths else pd.DataFrame(columns=s_required + s_optional)
        if not s_df.empty:
            s_df = s_df[s_df["user_id"].isin(users)]
        q_groups = {int(uid): g for uid, g in q_df.groupby("user_id", sort=False)} if not q_df.empty else {}
        s_groups = {int(uid): g for uid, g in s_df.groupby("user_id", sort=False)} if not s_df.empty else {}

        rows_all: List[dict] = []
        for uid, b_user in b_df.groupby("user_id", sort=False):
            u = int(uid)
            split_name = split_map.get(u)
            if split_name is None:
                continue
            rows = builder.build_user(u, b_user, q_groups.get(u, pd.DataFrame()), s_groups.get(u, pd.DataFrame()), split_name)
            rows_all.extend(rows)
        if rows_all:
            out_paths.append(write_table(pd.DataFrame(rows_all), RAW_ROOT / f"raw_panel_chunk_{ci:03d}"))
    return out_paths


def finalize_panel_chunk(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["split", "user_id", "bundle_step_index"], kind="mergesort").reset_index(drop=True)
    group = df.groupby("user_id", sort=False)
    state_fields = [
        "M_response_prebalanced",
        "response_evidence_maturity_V",
        "response_evidence_mass",
        "unresolved_response_demand_mass",
        "activity_alignment_order_Psi",
        "activity_active_mass",
        "activity_aligned_mass",
        "activity_off_target_mass",
        "activity_non_aligned_mass",
        "activity_neutral_mass",
        "activity_idle_mass",
    ]
    for field in state_fields:
        pre_col = f"{field}_pre"
        if pre_col not in df.columns:
            continue
        df[f"next_{field}"] = group[pre_col].shift(-1)
        df[f"delta_{field}_next"] = df[f"next_{field}"] - df[pre_col]
        for phase in ("resp", "support", "post"):
            phase_col = f"{field}_{phase}"
            if phase_col in df.columns:
                df[f"delta_{field}_{phase}_from_pre"] = df[phase_col] - df[pre_col]
    df["next_pre_ts"] = group["pre_ts"].shift(-1)
    df["next_gap_days"] = (
        pd.to_numeric(df["next_pre_ts"], errors="coerce")
        - pd.to_numeric(df["pre_ts"], errors="coerce")
    ) / DAY_MS
    df["has_next_submitted_bundle"] = df["next_pre_ts"].notna()
    df["has_next_within_observation_horizon"] = (
        df["has_next_submitted_bundle"]
        & (pd.to_numeric(df["next_gap_days"], errors="coerce") <= OBSERVATION_HORIZON_DAYS)
    )
    df["long_gap_or_no_next"] = (
        ~df["has_next_submitted_bundle"]
        | (pd.to_numeric(df["next_gap_days"], errors="coerce") > LONG_GAP_DAYS)
    )
    df["state_observed_M_Psi_pre"] = (
        np.isfinite(df.get("M_response_prebalanced_pre", np.nan))
        & np.isfinite(df.get("activity_alignment_order_Psi_pre", np.nan))
    )
    df["transition_observed_M_Psi"] = (
        df["state_observed_M_Psi_pre"]
        & np.isfinite(df.get("next_M_response_prebalanced", np.nan))
        & np.isfinite(df.get("next_activity_alignment_order_Psi", np.nan))
    )
    return df


def core_panel_columns(df: pd.DataFrame) -> List[str]:
    keep_exact = {
        "split",
        "user_id",
        "bundle_step_index",
        "bundle_attempt_index",
        "bundle_id",
        "part",
        "source_code",
        "platform_code",
        "pre_ts",
        "submit_ts",
        "next_pre_ts",
        "next_gap_days",
        "has_next_submitted_bundle",
        "has_next_within_observation_horizon",
        "long_gap_or_no_next",
        "current_accuracy_diagnostic_only",
        "total_response_count_diagnostic",
        "access_status_aux_context_only",
        "payment_coupon_window_aux_context_only",
        "semantic_guardrail",
    }
    prefixes = (
        "M_response_",
        "response_evidence_",
        "activity_",
        "support_active_",
        "support_alignment_",
        "response_alignment_",
        "response_active_",
        "response_aligned_",
        "response_off_",
        "response_neutral_",
        "support_aligned_",
        "support_off_",
        "support_neutral_",
        "idle_",
        "unresolved_response_",
        "delta_",
        "next_",
        "state_observed_",
        "transition_observed_",
        "has_next_",
        "long_gap_",
        "bundle_n_questions",
        "answered_fraction_",
        "kt4_",
        "access_",
        "payment_",
    )
    return [column for column in df.columns if column in keep_exact or column.startswith(prefixes)]


def downcast_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in {"user_id", "bundle_step_index", "bundle_attempt_index", "part", "source_code", "platform_code"}:
            out[c] = pd.to_numeric(out[c], errors="coerce", downcast="integer")
        elif out[c].dtype.kind in "fc":
            out[c] = pd.to_numeric(out[c], errors="coerce", downcast="float")
        elif out[c].dtype == object and c in {"split", "bundle_id"}:
            out[c] = out[c].astype("category")
    return out


def finalize_stage1_panels(raw_paths: List[Path]) -> Dict[str, object]:
    outputs = [
        DYN_ROOT / "student_dynamics_panel",
        DYN_ROOT / "student_dynamics_panel_core",
        DYN_ROOT / "student_dynamics_panel_core_A_train",
        DYN_ROOT / "student_dynamics_panel_core_A_val",
        DYN_ROOT / "student_dynamics_panel_core_B_confirm",
        DYN_ROOT / "B_confirm_panel_output_only",
    ]
    for base in outputs:
        remove_table_outputs(base)
    first = True
    full_cols: Optional[List[str]] = None
    core_cols: Optional[List[str]] = None
    total_rows = 0
    for p in tqdm(raw_paths, desc="Finalizing empirical v3 panels", unit="chunk"):
        raw = read_path_table(p)
        fin = finalize_panel_chunk(raw)
        if fin.empty:
            continue
        _, full_cols = append_table_csv_gz(fin, DYN_ROOT / "student_dynamics_panel", first=first, columns=full_cols)
        core = fin[core_panel_columns(fin)].copy()
        _, core_cols = append_table_csv_gz(core, DYN_ROOT / "student_dynamics_panel_core", first=first, columns=core_cols)
        for split in ("A_train", "A_val", "B_confirm"):
            sub = core[core["split"] == split]
            if first or not sub.empty:
                append_table_csv_gz(sub, DYN_ROOT / f"student_dynamics_panel_core_{split}", first=first, columns=core_cols)
        b = fin[fin["split"] == "B_confirm"]
        if first or not b.empty:
            append_table_csv_gz(b, DYN_ROOT / "B_confirm_panel_output_only", first=first, columns=full_cols)
        total_rows += int(len(fin))
        first = False
    manifest = {
        "finalized_rows": int(total_rows),
        "raw_chunk_count": int(len(raw_paths)),
        "full_panel_path": str((DYN_ROOT / "student_dynamics_panel.csv.gz").resolve()),
        "core_panel_path": str((DYN_ROOT / "student_dynamics_panel_core.csv.gz").resolve()),
        "split_core_paths": {s: str((DYN_ROOT / f"student_dynamics_panel_core_{s}.csv.gz").resolve()) for s in ("A_train", "A_val", "B_confirm")},
        "B_confirm_output_only_path": str((DYN_ROOT / "B_confirm_panel_output_only.csv.gz").resolve()),
    }
    with open(META_OUT_ROOT / "finalize_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def read_core_split(split: str) -> pd.DataFrame:
    df = read_table(DYN_ROOT / f"student_dynamics_panel_core_{split}")
    return downcast_frame(df)

# -----------------------------------------------------------------------------
# Coordinate statistics, regions, transitions and controls
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CoordinateSpec:
    name: str
    xcol: str
    ycol: str
    dxcol: str
    dycol: str
    xbins: np.ndarray
    ybins: np.ndarray
    y_short: str
    role: str


def coordinate_specs() -> List[CoordinateSpec]:
    return [
        CoordinateSpec(
            name="MR_PsiA",
            xcol="M_response_prebalanced_pre",
            ycol="activity_alignment_order_Psi_pre",
            dxcol="delta_M_response_prebalanced_next",
            dycol="delta_activity_alignment_order_Psi_next",
            xbins=GRID_BINS_SIGNED,
            ybins=GRID_BINS_SIGNED,
            y_short="Psi",
            role="primary semantic phase plane",
        )
    ]


def user_balanced_weights(df: pd.DataFrame) -> np.ndarray:
    if df.empty or "user_id" not in df.columns:
        return np.ones(len(df), dtype=float)
    c = df.groupby("user_id")["user_id"].transform("count").to_numpy(dtype=float)
    return 1.0 / np.maximum(c, 1.0)


def spec_arrays(df: pd.DataFrame, spec: CoordinateSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = pd.to_numeric(df[spec.xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[spec.ycol], errors="coerce").to_numpy(dtype=float)
    dx = pd.to_numeric(df[spec.dxcol], errors="coerce").to_numpy(dtype=float)
    dy = pd.to_numeric(df[spec.dycol], errors="coerce").to_numpy(dtype=float)
    state_valid = np.isfinite(x) & np.isfinite(y)
    drift_valid = state_valid & np.isfinite(dx) & np.isfinite(dy)
    return x, y, dx, dy, state_valid, drift_valid


def digitize_closed_right(vals: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Digitize values while including the final right edge."""
    arr = np.asarray(vals, dtype=float)
    edges = np.asarray(bins, dtype=float)
    if edges.size == 0:
        return np.full(arr.shape, -1, dtype=np.int64)
    adjusted = np.where(arr == edges[-1], np.nextafter(edges[-1], edges[0]), arr)
    return np.digitize(adjusted, edges) - 1


def occupancy_drift_stats(df: pd.DataFrame, spec: CoordinateSpec) -> Dict[str, object]:
    x, y, dx, dy, state_valid, drift_valid = spec_arrays(df, spec)
    weights = user_balanced_weights(df)
    nx = len(spec.xbins) - 1
    ny = len(spec.ybins) - 1
    shape = (nx, ny)
    xcenters = 0.5 * (spec.xbins[:-1] + spec.xbins[1:])
    ycenters = 0.5 * (spec.ybins[:-1] + spec.ybins[1:])

    ix = digitize_closed_right(x, spec.xbins)
    iy = digitize_closed_right(y, spec.ybins)
    state_in = state_valid & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    drift_in = drift_valid & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    state_cell = ix[state_in] * ny + iy[state_in]
    occupancy_weighted = np.bincount(
        state_cell,
        weights=weights[state_in],
        minlength=nx * ny,
    ).reshape(shape).astype(float)
    occupancy_count = np.bincount(
        state_cell,
        minlength=nx * ny,
    ).reshape(shape).astype(float)

    user_count = np.zeros(nx * ny, dtype=float)
    if np.any(state_in):
        user_ids = pd.to_numeric(df["user_id"], errors="coerce").to_numpy(dtype=np.int64)
        unique_user_cell = pd.DataFrame({"cell": state_cell, "user_id": user_ids[state_in]}).drop_duplicates()
        grouped = unique_user_cell.groupby("cell")["user_id"].size()
        user_count[grouped.index.to_numpy(dtype=int)] = grouped.to_numpy(dtype=float)
    user_count = user_count.reshape(shape)

    drift_count = np.zeros(shape, dtype=float)
    drift_weight = np.zeros(shape, dtype=float)
    drift_weight_sq = np.zeros(shape, dtype=float)
    sx = np.zeros(shape, dtype=float)
    sy = np.zeros(shape, dtype=float)
    sxx = np.zeros(shape, dtype=float)
    syy = np.zeros(shape, dtype=float)
    sxy = np.zeros(shape, dtype=float)
    if np.any(drift_in):
        drift_cell = ix[drift_in] * ny + iy[drift_in]
        wt = weights[drift_in]
        du = dx[drift_in]
        dv = dy[drift_in]
        drift_count = np.bincount(drift_cell, minlength=nx * ny).reshape(shape).astype(float)
        drift_weight = np.bincount(drift_cell, weights=wt, minlength=nx * ny).reshape(shape).astype(float)
        drift_weight_sq = np.bincount(drift_cell, weights=wt * wt, minlength=nx * ny).reshape(shape).astype(float)
        sx = np.bincount(drift_cell, weights=wt * du, minlength=nx * ny).reshape(shape)
        sy = np.bincount(drift_cell, weights=wt * dv, minlength=nx * ny).reshape(shape)
        sxx = np.bincount(drift_cell, weights=wt * du * du, minlength=nx * ny).reshape(shape)
        syy = np.bincount(drift_cell, weights=wt * dv * dv, minlength=nx * ny).reshape(shape)
        sxy = np.bincount(drift_cell, weights=wt * du * dv, minlength=nx * ny).reshape(shape)

    denominator = np.maximum(drift_weight, EPS)
    drift_u = sx / denominator
    drift_v = sy / denominator
    diff_x = np.maximum(sxx / denominator - drift_u * drift_u, 0.0)
    diff_y = np.maximum(syy / denominator - drift_v * drift_v, 0.0)
    diff_xy = sxy / denominator - drift_u * drift_v
    effective_n = drift_weight * drift_weight / np.maximum(drift_weight_sq, EPS)
    drift_se_u = np.sqrt(diff_x / np.maximum(effective_n, 1.0))
    drift_se_v = np.sqrt(diff_y / np.maximum(effective_n, 1.0))
    state_mask = (occupancy_count >= MIN_STATE_BIN_COUNT) & (user_count >= MIN_CELL_USERS)
    drift_mask = drift_count >= MIN_DRIFT_BIN_COUNT
    occupancy_probability = occupancy_weighted / max(float(occupancy_weighted.sum()), EPS)
    potential = -np.log(occupancy_probability + EPS)

    return {
        "coordinate": spec.name,
        "xbins": np.asarray(spec.xbins, dtype=float),
        "ybins": np.asarray(spec.ybins, dtype=float),
        "xcenters": xcenters,
        "ycenters": ycenters,
        "occupancy_weighted": occupancy_weighted,
        "occupancy_count": occupancy_count,
        "user_count": user_count,
        "occupancy_probability": occupancy_probability,
        "potential": potential,
        "drift_u": drift_u,
        "drift_v": drift_v,
        "drift_count": drift_count,
        "drift_weight": drift_weight,
        "drift_weight_sq": drift_weight_sq,
        "drift_effective_sample_size": effective_n,
        "drift_se_u": drift_se_u,
        "drift_se_v": drift_se_v,
        "diff_x": diff_x,
        "diff_y": diff_y,
        "diff_xy": diff_xy,
        "state_mask": state_mask,
        "drift_mask": drift_mask,
        "valid_state_rows": int(np.sum(state_in)),
        "valid_drift_rows": int(np.sum(drift_in)),
        "rows": int(len(df)),
        "users": int(df["user_id"].nunique()) if "user_id" in df.columns else int(len(df)),
        "occupied_bins": int(np.sum(state_mask)),
        "drift_bins": int(np.sum(drift_mask)),
        "state_valid_rate": float(np.mean(state_in)) if len(state_in) else 0.0,
        "drift_valid_rate": float(np.mean(drift_in)) if len(drift_in) else 0.0,
    }


@dataclass
class FieldStats:
    xbins: np.ndarray
    ybins: np.ndarray
    xcenters: np.ndarray
    ycenters: np.ndarray
    occupancy_weighted: np.ndarray
    occupancy_count: np.ndarray
    user_count: np.ndarray
    occupancy_probability: np.ndarray
    potential: np.ndarray
    drift_u: np.ndarray
    drift_v: np.ndarray
    drift_count: np.ndarray
    drift_weight: np.ndarray
    drift_weight_sq: np.ndarray
    drift_effective_sample_size: np.ndarray
    drift_se_u: np.ndarray
    drift_se_v: np.ndarray
    diff_x: np.ndarray
    diff_y: np.ndarray
    diff_xy: np.ndarray
    state_mask: np.ndarray
    drift_mask: np.ndarray
    valid_state_rows: int
    valid_drift_rows: int
    users: int


def field_stats_from_dict(stats: Dict[str, object]) -> FieldStats:
    return FieldStats(
        xbins=np.asarray(stats["xbins"], dtype=float),
        ybins=np.asarray(stats["ybins"], dtype=float),
        xcenters=np.asarray(stats["xcenters"], dtype=float),
        ycenters=np.asarray(stats["ycenters"], dtype=float),
        occupancy_weighted=np.asarray(stats["occupancy_weighted"], dtype=float),
        occupancy_count=np.asarray(stats["occupancy_count"], dtype=float),
        user_count=np.asarray(stats["user_count"], dtype=float),
        occupancy_probability=np.asarray(stats["occupancy_probability"], dtype=float),
        potential=np.asarray(stats["potential"], dtype=float),
        drift_u=np.asarray(stats["drift_u"], dtype=float),
        drift_v=np.asarray(stats["drift_v"], dtype=float),
        drift_count=np.asarray(stats["drift_count"], dtype=float),
        drift_weight=np.asarray(stats["drift_weight"], dtype=float),
        drift_weight_sq=np.asarray(stats["drift_weight_sq"], dtype=float),
        drift_effective_sample_size=np.asarray(
            stats["drift_effective_sample_size"], dtype=float
        ),
        drift_se_u=np.asarray(stats["drift_se_u"], dtype=float),
        drift_se_v=np.asarray(stats["drift_se_v"], dtype=float),
        diff_x=np.asarray(stats["diff_x"], dtype=float),
        diff_y=np.asarray(stats["diff_y"], dtype=float),
        diff_xy=np.asarray(stats["diff_xy"], dtype=float),
        state_mask=np.asarray(stats["state_mask"], dtype=bool),
        drift_mask=np.asarray(stats["drift_mask"], dtype=bool),
        valid_state_rows=int(stats["valid_state_rows"]),
        valid_drift_rows=int(stats["valid_drift_rows"]),
        users=int(stats["users"]),
    )


def field_grid_table(stats: FieldStats, split: str) -> pd.DataFrame:
    rows: List[dict] = []
    diffusion_trace = np.maximum(np.asarray(stats.diff_x) + np.asarray(stats.diff_y), 0.0)
    diffusion_det = np.maximum(np.asarray(stats.diff_x) * np.asarray(stats.diff_y) - np.asarray(stats.diff_xy) ** 2, 0.0)
    trace = diffusion_trace
    disc = np.sqrt(np.maximum((np.asarray(stats.diff_x) - np.asarray(stats.diff_y)) ** 2 + 4.0 * np.asarray(stats.diff_xy) ** 2, 0.0))
    diffusion_eig_max = 0.5 * (trace + disc)
    diffusion_eig_min = np.maximum(0.5 * (trace - disc), 0.0)
    diffusion_anisotropy = (diffusion_eig_max - diffusion_eig_min) / np.maximum(diffusion_eig_max + diffusion_eig_min, EPS)
    speed = np.sqrt(np.asarray(stats.drift_u) ** 2 + np.asarray(stats.drift_v) ** 2)
    drift_to_diffusion = speed / np.maximum(np.sqrt(diffusion_trace), EPS)
    flux_u = np.asarray(stats.occupancy_probability) * np.asarray(stats.drift_u)
    flux_v = np.asarray(stats.occupancy_probability) * np.asarray(stats.drift_v)
    flux_speed = np.sqrt(flux_u ** 2 + flux_v ** 2)
    drift_se_speed = np.sqrt(np.asarray(stats.drift_se_u) ** 2 + np.asarray(stats.drift_se_v) ** 2)
    drift_signal_to_se = speed / np.maximum(drift_se_speed, EPS)
    for i, x in enumerate(stats.xcenters):
        for j, y in enumerate(stats.ycenters):
            rows.append(
                {
                    "split": split,
                    "x_bin": i,
                    "y_bin": j,
                    "M_center": float(x),
                    "Psi_center": float(y),
                    "occupancy_weighted": float(stats.occupancy_weighted[i, j]),
                    "occupancy_count": int(stats.occupancy_count[i, j]),
                    "user_count": int(stats.user_count[i, j]),
                    "occupancy_probability": float(stats.occupancy_probability[i, j]),
                    "empirical_quasi_potential": float(stats.potential[i, j]),
                    "drift_M": float(stats.drift_u[i, j]),
                    "drift_Psi": float(stats.drift_v[i, j]),
                    "drift_speed": float(math.hypot(stats.drift_u[i, j], stats.drift_v[i, j])),
                    "drift_count": int(stats.drift_count[i, j]),
                    "drift_user_balanced_weight": float(stats.drift_weight[i, j]),
                    "drift_effective_sample_size_descriptive": float(stats.drift_effective_sample_size[i, j]),
                    "drift_standard_error_M_descriptive": float(stats.drift_se_u[i, j]),
                    "drift_standard_error_Psi_descriptive": float(stats.drift_se_v[i, j]),
                    "drift_signal_to_standard_error_descriptive": float(drift_signal_to_se[i, j]),
                    "state_supported": bool(stats.state_mask[i, j]),
                    "drift_supported": bool(stats.drift_mask[i, j]),
                    "diffusion_M": float(stats.diff_x[i, j]),
                    "diffusion_Psi": float(stats.diff_y[i, j]),
                    "diffusion_cross": float(stats.diff_xy[i, j]),
                    "diffusion_trace": float(diffusion_trace[i, j]),
                    "diffusion_determinant": float(diffusion_det[i, j]),
                    "diffusion_eigenvalue_max": float(diffusion_eig_max[i, j]),
                    "diffusion_eigenvalue_min": float(diffusion_eig_min[i, j]),
                    "diffusion_anisotropy": float(diffusion_anisotropy[i, j]),
                    "drift_to_diffusion_ratio": float(drift_to_diffusion[i, j]),
                    "probability_flux_M": float(flux_u[i, j]),
                    "probability_flux_Psi": float(flux_v[i, j]),
                    "probability_flux_magnitude": float(flux_speed[i, j]),
                }
            )
    return pd.DataFrame(rows)


def interior_divergence(stats: FieldStats) -> Tuple[np.ndarray, np.ndarray]:
    """Compute central-difference divergence on fully supported stencils."""
    u = np.asarray(stats.drift_u, dtype=float)
    v = np.asarray(stats.drift_v, dtype=float)
    mask = (
        np.asarray(stats.state_mask, dtype=bool)
        & np.asarray(stats.drift_mask, dtype=bool)
        & np.isfinite(u)
        & np.isfinite(v)
    )
    div = np.full_like(u, np.nan, dtype=float)
    interior = np.zeros_like(mask, dtype=bool)
    if u.shape[0] < 3 or u.shape[1] < 3:
        return div, interior
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    dx_den = (stats.xcenters[2:] - stats.xcenters[:-2])[:, None]
    dy_den = (stats.ycenters[2:] - stats.ycenters[:-2])[None, :]
    d_u_dx = (u[2:, 1:-1] - u[:-2, 1:-1]) / np.maximum(dx_den, EPS)
    d_v_dy = (v[1:-1, 2:] - v[1:-1, :-2]) / np.maximum(dy_den, EPS)
    local = d_u_dx + d_v_dy
    target = div[1:-1, 1:-1]
    supported = interior[1:-1, 1:-1]
    target[supported] = local[supported]
    div[1:-1, 1:-1] = target
    return div, interior


def weighted_shortest_interval(centres: np.ndarray, weights: np.ndarray, target_mass: float) -> Tuple[float, float, float]:
    """Return the shortest contiguous interval containing the requested mass."""
    x = np.asarray(centres, dtype=float)
    w = np.asarray(weights, dtype=float)
    good = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x = x[good]
    w = w[good]
    if x.size == 0 or w.sum() <= 0:
        return np.nan, np.nan, np.nan
    order = np.argsort(x)
    x = x[order]
    w = w[order] / w.sum()
    target = float(np.clip(target_mass, EPS, 1.0))
    best: Optional[Tuple[float, int, int, float]] = None
    right = 0
    running = 0.0
    for left in range(len(x)):
        while right < len(x) and running < target:
            running += float(w[right])
            right += 1
        if running >= target and right > left:
            width = float(x[right - 1] - x[left])
            item = (width, left, right - 1, running)
            if best is None or item[0] < best[0]:
                best = item
        running -= float(w[left])
        if right < left + 1:
            right = left + 1
    if best is None:
        return float(x.min()), float(x.max()), 1.0
    _, left, right, mass = best
    return float(x[left]), float(x[right]), float(mass)


def occupancy_geometry_diagnostics(stats: FieldStats, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize occupancy separately from flow-defined convergence."""
    p = np.asarray(stats.occupancy_probability, dtype=float)
    pm = p.sum(axis=1)
    pp = p.sum(axis=0)
    m50 = weighted_shortest_interval(stats.xcenters, pm, 0.50)
    m80 = weighted_shortest_interval(stats.xcenters, pm, 0.80)
    p50 = weighted_shortest_interval(stats.ycenters, pp, 0.50)
    p80 = weighted_shortest_interval(stats.ycenters, pp, 0.80)
    m_mean = float(np.sum(stats.xcenters * pm)) if pm.sum() > 0 else np.nan
    p_mean = float(np.sum(stats.ycenters * pp)) if pp.sum() > 0 else np.nan
    m_sd = float(np.sqrt(np.sum(pm * (stats.xcenters - m_mean) ** 2))) if pm.sum() > 0 else np.nan
    p_sd = float(np.sqrt(np.sum(pp * (stats.ycenters - p_mean) ** 2))) if pp.sum() > 0 else np.nan
    summary = pd.DataFrame([{
        "split": split,
        "M_marginal_peak": float(stats.xcenters[int(np.argmax(pm))]) if pm.size else np.nan,
        "Psi_marginal_peak": float(stats.ycenters[int(np.argmax(pp))]) if pp.size else np.nan,
        "M_weighted_mean": m_mean,
        "Psi_weighted_mean": p_mean,
        "M_weighted_sd": m_sd,
        "Psi_weighted_sd": p_sd,
        "M_shortest_interval_50_low": m50[0],
        "M_shortest_interval_50_high": m50[1],
        "M_shortest_interval_50_mass": m50[2],
        "M_shortest_interval_80_low": m80[0],
        "M_shortest_interval_80_high": m80[1],
        "M_shortest_interval_80_mass": m80[2],
        "Psi_shortest_interval_50_low": p50[0],
        "Psi_shortest_interval_50_high": p50[1],
        "Psi_shortest_interval_50_mass": p50[2],
        "Psi_shortest_interval_80_low": p80[0],
        "Psi_shortest_interval_80_high": p80[1],
        "Psi_shortest_interval_80_mass": p80[2],
        "M_outermost_one_bin_mass_fraction": float(pm[0] + pm[-1]) if len(pm) >= 2 else np.nan,
        "M_outermost_two_bins_mass_fraction": float(pm[:2].sum() + pm[-2:].sum()) if len(pm) >= 4 else np.nan,
        "Psi_outermost_one_bin_mass_fraction": float(pp[0] + pp[-1]) if len(pp) >= 2 else np.nan,
        "Psi_outermost_two_bins_mass_fraction": float(pp[:2].sum() + pp[-2:].sum()) if len(pp) >= 4 else np.nan,
        "interpretation": "occupancy geometry only; marginal peaks and boundary mass are not drift-defined convergence centres",
    }])
    ridge_rows: List[dict] = []
    for i, m in enumerate(stats.xcenters):
        row = p[i]
        if row.sum() <= 0:
            continue
        j = int(np.argmax(row))
        ridge_rows.append({
            "split": split,
            "M_center": float(m),
            "Psi_ridge_center": float(stats.ycenters[j]),
            "ridge_cell_probability": float(row[j]),
            "M_marginal_probability": float(pm[i]),
            "ridge_conditional_probability_given_M_bin": float(row[j] / max(row.sum(), EPS)),
        })
    return summary, pd.DataFrame(ridge_rows)


@dataclass(frozen=True)
class ConvergenceThresholds:
    speed_threshold: float
    negative_divergence_threshold: float
    drift_to_diffusion_threshold: float
    speed_quantile: float
    negative_divergence_quantile: float
    drift_to_diffusion_quantile: float


def _weighted_grid_mean(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)[mask]
    w = np.asarray(weights, dtype=float)[mask]
    good = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    if not good.any() or w[good].sum() <= 0:
        return np.nan
    return float(np.sum(v[good] * w[good]) / np.sum(w[good]))


def _weighted_grid_fraction(condition: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    c = np.asarray(condition, dtype=bool)[mask]
    w = np.asarray(weights, dtype=float)[mask]
    good = np.isfinite(w) & (w >= 0)
    if not good.any() or w[good].sum() <= 0:
        return np.nan
    return float(np.sum(w[good] * c[good].astype(float)) / np.sum(w[good]))


def _local_affine_drift_fit(stats: FieldStats, fit_mask: np.ndarray) -> dict:
    """Fit a local affine drift and estimate its stationary point."""
    X, Y = np.meshgrid(stats.xcenters, stats.ycenters, indexing="ij")
    u = np.asarray(stats.drift_u, dtype=float)
    v = np.asarray(stats.drift_v, dtype=float)
    support_weight = np.asarray(stats.drift_weight, dtype=float)
    mask = np.asarray(fit_mask, dtype=bool) & np.isfinite(u) & np.isfinite(v) & (support_weight > 0)
    if np.sum(mask) < 6:
        return {"local_linear_fit_cells": int(np.sum(mask)), "local_fixed_point_available": False}
    x = X[mask]
    y = Y[mask]
    du = u[mask]
    dv = v[mask]
    w = np.maximum(support_weight[mask], EPS)
    x0 = float(np.average(x, weights=w))
    y0 = float(np.average(y, weights=w))
    design = np.column_stack([np.ones(len(x)), x - x0, y - y0])
    sw = np.sqrt(w / max(float(np.mean(w)), EPS))
    dw = design * sw[:, None]
    try:
        beta_u, *_ = np.linalg.lstsq(dw, du * sw, rcond=None)
        beta_v, *_ = np.linalg.lstsq(dw, dv * sw, rcond=None)
    except np.linalg.LinAlgError:
        return {"local_linear_fit_cells": int(np.sum(mask)), "local_fixed_point_available": False}
    A = np.asarray([[beta_u[1], beta_u[2]], [beta_v[1], beta_v[2]]], dtype=float)
    c0 = np.asarray([beta_u[0], beta_v[0]], dtype=float)
    condition = float(np.linalg.cond(A)) if np.isfinite(A).all() else np.inf
    fixed = np.asarray([np.nan, np.nan])
    available = bool(np.isfinite(condition) and condition < 1e10)
    if available:
        try:
            delta = np.linalg.solve(A, -c0)
            fixed = np.asarray([x0 + delta[0], y0 + delta[1]], dtype=float)
        except np.linalg.LinAlgError:
            available = False
    eig = np.linalg.eigvals(A) if np.isfinite(A).all() else np.asarray([np.nan, np.nan])
    pred_u = design @ beta_u
    pred_v = design @ beta_v
    def r2(obs: np.ndarray, pred: np.ndarray) -> float:
        den = float(np.sum(w * (obs - np.average(obs, weights=w)) ** 2))
        num = float(np.sum(w * (obs - pred) ** 2))
        return float(1.0 - num / den) if den > EPS else np.nan
    stable = bool(np.all(np.real(eig) < 0)) if np.isfinite(eig).all() else False
    bbox = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    inside_bbox = bool(available and bbox[0] <= fixed[0] <= bbox[1] and bbox[2] <= fixed[1] <= bbox[3])
    return {
        "local_linear_fit_cells": int(np.sum(mask)),
        "local_fixed_point_available": available,
        "local_fixed_point_M": float(fixed[0]) if available else np.nan,
        "local_fixed_point_Psi": float(fixed[1]) if available else np.nan,
        "local_fixed_point_inside_fit_bbox": inside_bbox,
        "local_jacobian_trace": float(np.trace(A)),
        "local_jacobian_determinant": float(np.linalg.det(A)),
        "local_jacobian_condition_number": condition,
        "local_jacobian_eigenvalue_1_real": float(np.real(eig[0])) if eig.size else np.nan,
        "local_jacobian_eigenvalue_1_imag": float(np.imag(eig[0])) if eig.size else np.nan,
        "local_jacobian_eigenvalue_2_real": float(np.real(eig[1])) if eig.size > 1 else np.nan,
        "local_jacobian_eigenvalue_2_imag": float(np.imag(eig[1])) if eig.size > 1 else np.nan,
        "local_linearization_stable": stable,
        "local_linear_fit_r2_M_drift": r2(du, pred_u),
        "local_linear_fit_r2_Psi_drift": r2(dv, pred_v),
    }


def _region_metrics(
    stats: FieldStats,
    divergence: np.ndarray,
    interior: np.ndarray,
    core_mask: np.ndarray,
    shell_radius: float,
    split: str,
    region_id: int,
) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Summarize a flow-defined core and its surrounding shell."""
    h = np.asarray(stats.occupancy_weighted, dtype=float)
    p = np.asarray(stats.occupancy_probability, dtype=float)
    u = np.asarray(stats.drift_u, dtype=float)
    v = np.asarray(stats.drift_v, dtype=float)
    speed = np.sqrt(u * u + v * v)
    diffusion_trace = np.maximum(np.asarray(stats.diff_x) + np.asarray(stats.diff_y), 0.0)
    diffusion_scale = np.sqrt(diffusion_trace)
    ratio = speed / np.maximum(diffusion_scale, EPS)
    drift_mask = np.asarray(stats.drift_mask, dtype=bool) & np.isfinite(u) & np.isfinite(v)
    supported_core = np.asarray(core_mask, dtype=bool) & drift_mask
    dx = float(np.mean(np.diff(stats.xcenters))) if len(stats.xcenters) > 1 else 1.0
    dy = float(np.mean(np.diff(stats.ycenters))) if len(stats.ycenters) > 1 else 1.0
    distance, nearest = distance_transform_edt(~np.asarray(core_mask, dtype=bool), sampling=(dx, dy), return_indices=True)
    shell = drift_mask & ~np.asarray(core_mask, dtype=bool) & (distance > 0) & (distance <= float(shell_radius))
    X, Y = np.meshgrid(stats.xcenters, stats.ycenters, indexing="ij")
    near_x = X[nearest[0], nearest[1]]
    near_y = Y[nearest[0], nearest[1]]
    to_x = near_x - X
    to_y = near_y - Y
    inward_cosine = np.full_like(u, np.nan, dtype=float)
    ok = shell & (speed > EPS) & (distance > EPS)
    inward_cosine[ok] = (u[ok] * to_x[ok] + v[ok] * to_y[ok]) / (speed[ok] * distance[ok])
    inward_component = np.full_like(u, np.nan, dtype=float)
    inward_component[ok] = (u[ok] * to_x[ok] + v[ok] * to_y[ok]) / distance[ok]

    cells = np.argwhere(np.asarray(core_mask, dtype=bool))
    if cells.size:
        i_min, i_max = int(cells[:, 0].min()), int(cells[:, 0].max())
        j_min, j_max = int(cells[:, 1].min()), int(cells[:, 1].max())
        m_min = float(stats.xcenters[i_min])
        m_max = float(stats.xcenters[i_max])
        psi_min = float(stats.ycenters[j_min])
        psi_max = float(stats.ycenters[j_max])
        m_edge_min = float(stats.xbins[i_min])
        m_edge_max = float(stats.xbins[i_max + 1])
        psi_edge_min = float(stats.ybins[j_min])
        psi_edge_max = float(stats.ybins[j_max + 1])
        bounding_box_cells = int((i_max - i_min + 1) * (j_max - j_min + 1))
        bounding_box_fill_fraction = float(len(cells) / max(bounding_box_cells, 1))
    else:
        m_min = m_max = psi_min = psi_max = np.nan
        m_edge_min = m_edge_max = psi_edge_min = psi_edge_max = np.nan
        bounding_box_cells = 0
        bounding_box_fill_fraction = np.nan
    valid_core = supported_core & np.isfinite(divergence)
    contraction = np.maximum(-divergence, 0.0)
                                                                                  
                                                                            
    flow_weight = np.sqrt(np.maximum(np.asarray(stats.drift_weight, dtype=float), 0.0))
    median_speed = float(np.nanmedian(speed[drift_mask])) if np.any(drift_mask) else 0.0
    centre_weight = flow_weight * contraction / np.maximum(speed + 0.05 * median_speed, EPS)
    if np.isfinite(centre_weight[supported_core]).any() and np.nansum(centre_weight[supported_core]) > 0:
        center_m = _weighted_grid_mean(X, centre_weight, supported_core)
        center_psi = _weighted_grid_mean(Y, centre_weight, supported_core)
        localization_weight = np.where(supported_core & np.isfinite(centre_weight), np.maximum(centre_weight, 0.0), 0.0)
    else:
        center_m = _weighted_grid_mean(X, flow_weight, supported_core)
        center_psi = _weighted_grid_mean(Y, flow_weight, supported_core)
        localization_weight = np.where(supported_core & np.isfinite(flow_weight), np.maximum(flow_weight, 0.0), 0.0)

                                                                               
    localization_marginal_M = np.sum(localization_weight, axis=1)
    localization_marginal_Psi = np.sum(localization_weight, axis=0)
    loc_m50 = weighted_shortest_interval(stats.xcenters, localization_marginal_M, 0.50)
    loc_m80 = weighted_shortest_interval(stats.xcenters, localization_marginal_M, 0.80)
    loc_p50 = weighted_shortest_interval(stats.ycenters, localization_marginal_Psi, 0.50)
    loc_p80 = weighted_shortest_interval(stats.ycenters, localization_marginal_Psi, 0.80)

                                                                          
    core_speed = _weighted_grid_mean(speed, h, supported_core)
    shell_speed = _weighted_grid_mean(speed, h, shell)
    core_to_shell = core_speed / shell_speed if np.isfinite(core_speed) and np.isfinite(shell_speed) and shell_speed > 0 else np.nan
    inward_cos = _weighted_grid_mean(inward_cosine, h, shell)
    inward_frac = _weighted_grid_fraction(inward_cosine > 0, h, shell)
    inward_comp = _weighted_grid_mean(inward_component, h, shell)
    mean_div = _weighted_grid_mean(divergence, h, valid_core)
    neg_div_frac = _weighted_grid_fraction(divergence < 0, h, valid_core)
    occ_mass = float(p[np.asarray(core_mask, dtype=bool)].sum())
    mean_diff = _weighted_grid_mean(diffusion_trace, h, supported_core)
    mean_ratio = _weighted_grid_mean(ratio, h, supported_core)

                                                                            
    flow_core_speed = _weighted_grid_mean(speed, flow_weight, supported_core)
    flow_shell_speed = _weighted_grid_mean(speed, flow_weight, shell)
    flow_core_to_shell = flow_core_speed / flow_shell_speed if np.isfinite(flow_core_speed) and np.isfinite(flow_shell_speed) and flow_shell_speed > 0 else np.nan
    flow_inward_cos = _weighted_grid_mean(inward_cosine, flow_weight, shell)
    flow_inward_frac = _weighted_grid_fraction(inward_cosine > 0, flow_weight, shell)
    flow_inward_comp = _weighted_grid_mean(inward_component, flow_weight, shell)
    flow_mean_div = _weighted_grid_mean(divergence, flow_weight, valid_core)
    flow_neg_div_frac = _weighted_grid_fraction(divergence < 0, flow_weight, valid_core)
    flow_mean_diff = _weighted_grid_mean(diffusion_trace, flow_weight, supported_core)
    flow_mean_ratio = _weighted_grid_mean(ratio, flow_weight, supported_core)
    flow_support_mass = float(np.nansum(np.asarray(stats.drift_weight, dtype=float)[supported_core]))
    local_fit = _local_affine_drift_fit(stats, supported_core | shell)

    contraction_scale = float(np.nanmedian(contraction[interior & (contraction > 0)])) if np.any(interior & (contraction > 0)) else 1.0
    contraction_strength = max(-float(flow_mean_div), 0.0) / max(contraction_scale, EPS) if np.isfinite(flow_mean_div) else 0.0
    speed_drop = max(1.0 - float(flow_core_to_shell), 0.0) if np.isfinite(flow_core_to_shell) else 0.0
    inward_strength = max(float(flow_inward_cos), 0.0) if np.isfinite(flow_inward_cos) else 0.0
    inward_coverage = max(float(flow_inward_frac), 0.0) if np.isfinite(flow_inward_frac) else 0.0
    score = contraction_strength * (0.5 + inward_strength) * (0.5 + inward_coverage) * (0.5 + speed_drop) * math.log1p(max(flow_support_mass, EPS))

    summary = {
        "split": split,
        "region_id": int(region_id),
        "region_cells_total": int(np.sum(core_mask)),
        "region_cells_with_supported_drift": int(np.sum(supported_core)),
        "region_cells_with_supported_divergence": int(np.sum(valid_core)),
        "shell_supported_cells": int(np.sum(shell)),
        "M_min": m_min,
        "M_max": m_max,
        "Psi_min": psi_min,
        "Psi_max": psi_max,
        "M_edge_min": m_edge_min,
        "M_edge_max": m_edge_max,
        "Psi_edge_min": psi_edge_min,
        "Psi_edge_max": psi_edge_max,
        "bounding_box_cells": bounding_box_cells,
        "bounding_box_fill_fraction": bounding_box_fill_fraction,
        "convergence_center_M": center_m,
        "convergence_center_Psi": center_psi,
        "flow_localization_M_shortest_interval_50_low": loc_m50[0],
        "flow_localization_M_shortest_interval_50_high": loc_m50[1],
        "flow_localization_M_shortest_interval_50_mass": loc_m50[2],
        "flow_localization_M_shortest_interval_80_low": loc_m80[0],
        "flow_localization_M_shortest_interval_80_high": loc_m80[1],
        "flow_localization_M_shortest_interval_80_mass": loc_m80[2],
        "flow_localization_Psi_shortest_interval_50_low": loc_p50[0],
        "flow_localization_Psi_shortest_interval_50_high": loc_p50[1],
        "flow_localization_Psi_shortest_interval_50_mass": loc_p50[2],
        "flow_localization_Psi_shortest_interval_80_low": loc_p80[0],
        "flow_localization_Psi_shortest_interval_80_high": loc_p80[1],
        "flow_localization_Psi_shortest_interval_80_mass": loc_p80[2],
        "occupancy_mass_fraction": occ_mass,
        "raw_state_count": int(np.sum(stats.occupancy_count[np.asarray(core_mask, dtype=bool)])),
        "cell_user_count_sum": int(np.sum(stats.user_count[np.asarray(core_mask, dtype=bool)])),
        "occupancy_weighted_mean_divergence": mean_div,
        "occupancy_weighted_negative_divergence_fraction": neg_div_frac,
        "occupancy_weighted_mean_core_speed": core_speed,
        "occupancy_weighted_mean_shell_speed": shell_speed,
        "core_to_shell_speed_ratio": core_to_shell,
        "occupancy_weighted_shell_inward_cosine": inward_cos,
        "occupancy_weighted_shell_fraction_inward": inward_frac,
        "occupancy_weighted_shell_inward_component": inward_comp,
        "occupancy_weighted_mean_diffusion_trace": mean_diff,
        "occupancy_weighted_mean_drift_to_diffusion_ratio": mean_ratio,
        "flow_support_mass_user_balanced": flow_support_mass,
        "flow_weighted_mean_divergence": flow_mean_div,
        "flow_weighted_negative_divergence_fraction": flow_neg_div_frac,
        "flow_weighted_mean_core_speed": flow_core_speed,
        "flow_weighted_mean_shell_speed": flow_shell_speed,
        "flow_core_to_shell_speed_ratio": flow_core_to_shell,
        "flow_weighted_shell_inward_cosine": flow_inward_cos,
        "flow_weighted_shell_fraction_inward": flow_inward_frac,
        "flow_weighted_shell_inward_component": flow_inward_comp,
        "flow_weighted_mean_diffusion_trace": flow_mean_diff,
        "flow_weighted_mean_drift_to_diffusion_ratio": flow_mean_ratio,
        "shell_radius": float(shell_radius),
        "convergence_score": float(score),
        **local_fit,
    }
    return summary, shell, inward_cosine, inward_component


def identify_convergence_regions(
    stats: FieldStats,
    split: str,
    speed_quantile: float,
    negative_divergence_quantile: float,
    drift_to_diffusion_quantile: float,
    min_region_cells: int,
    shell_radius: float,
    thresholds: Optional[ConvergenceThresholds] = None,
    allow_fallback: bool = True,
) -> Tuple[pd.DataFrame, List[np.ndarray], pd.DataFrame, ConvergenceThresholds]:
    """Identify connected flow-convergence cores from training-defined criteria."""
    div, interior = interior_divergence(stats)
    u = np.asarray(stats.drift_u, dtype=float)
    v = np.asarray(stats.drift_v, dtype=float)
    speed = np.sqrt(u * u + v * v)
    diffusion = np.sqrt(np.maximum(np.asarray(stats.diff_x) + np.asarray(stats.diff_y), 0.0))
    ratio = speed / np.maximum(diffusion, EPS)
    valid = (
        interior
        & np.asarray(stats.state_mask, dtype=bool)
        & np.asarray(stats.drift_mask, dtype=bool)
        & np.isfinite(div)
        & np.isfinite(speed)
        & np.isfinite(ratio)
    )
    if not np.any(valid):
        raise RuntimeError(f"No fully supported interior drift cells for convergence analysis in {split}.")
    if thresholds is None:
        neg = div[valid & (div < 0)]
        if neg.size == 0:
            raise RuntimeError(f"No negative-divergence cells for convergence analysis in {split}.")
        thresholds = ConvergenceThresholds(
            speed_threshold=float(np.quantile(speed[valid], np.clip(speed_quantile, 0.05, 0.95))),
            negative_divergence_threshold=float(np.quantile(neg, np.clip(negative_divergence_quantile, 0.05, 0.95))),
            drift_to_diffusion_threshold=float(np.quantile(ratio[valid], np.clip(drift_to_diffusion_quantile, 0.05, 0.95))),
            speed_quantile=float(speed_quantile),
            negative_divergence_quantile=float(negative_divergence_quantile),
            drift_to_diffusion_quantile=float(drift_to_diffusion_quantile),
        )
    candidate = (
        valid
        & (div < 0)
        & (div <= thresholds.negative_divergence_threshold)
        & (speed <= thresholds.speed_threshold)
        & (ratio <= thresholds.drift_to_diffusion_threshold)
    )
    labels, n_labels = ndimage_label(candidate, structure=np.ones((3, 3), dtype=int))
    rows: List[dict] = []
    masks: List[np.ndarray] = []
    for lab in range(1, n_labels + 1):
        mask = labels == lab
        if int(mask.sum()) < int(min_region_cells):
            continue
        summary, _, _, _ = _region_metrics(stats, div, interior, mask, shell_radius, split, len(rows))
        rows.append(summary)
        masks.append(mask)

    fallback_used = False
    if not rows and allow_fallback:
                                                                                 
                                                                   
        fallback_used = True
        relaxed = valid & (div < 0) & (speed <= np.quantile(speed[valid], 0.75)) & (ratio <= np.quantile(ratio[valid], 0.75))
        labels, n_labels = ndimage_label(relaxed, structure=np.ones((3, 3), dtype=int))
        for lab in range(1, n_labels + 1):
            mask = labels == lab
            if int(mask.sum()) < max(2, int(min_region_cells) // 2):
                continue
            summary, _, _, _ = _region_metrics(stats, div, interior, mask, shell_radius, split, len(rows))
            rows.append(summary)
            masks.append(mask)
    if not rows and allow_fallback:
        valid_idx = np.argwhere(valid & (div < 0))
        if valid_idx.size == 0:
            raise RuntimeError(f"Unable to construct a flow-defined convergence core in {split}.")
        score_grid = np.where(valid & (div < 0), (-div) / np.maximum(speed + 0.10 * np.nanmedian(speed[valid]), EPS), -np.inf)
        seed = np.unravel_index(int(np.nanargmax(score_grid)), score_grid.shape)
        mask = np.zeros_like(valid, dtype=bool)
        i0, j0 = seed
        mask[max(0, i0 - 1):min(mask.shape[0], i0 + 2), max(0, j0 - 1):min(mask.shape[1], j0 + 2)] = valid[max(0, i0 - 1):min(mask.shape[0], i0 + 2), max(0, j0 - 1):min(mask.shape[1], j0 + 2)]
        summary, _, _, _ = _region_metrics(stats, div, interior, mask, shell_radius, split, 0)
        rows = [summary]
        masks = [mask]
        fallback_used = True

    if not rows:
        grid = field_grid_table(stats, split)
        grid["interior_divergence_supported"] = interior.ravel(order="C")
        grid["interior_local_divergence"] = div.ravel(order="C")
        grid["drift_to_diffusion_ratio"] = ratio.ravel(order="C")
        grid["flow_convergence_candidate"] = candidate.ravel(order="C")
        grid["convergence_region_id"] = -1
        grid["primary_convergence_core"] = False
        grid["primary_convergence_shell"] = False
        grid["inward_cosine_to_primary_core"] = np.nan
        grid["inward_component_to_primary_core"] = np.nan
        return pd.DataFrame(), [], grid, thresholds

                                                                           
    for row in rows:
        row["dynamically_qualified"] = bool(
            np.isfinite(row.get("flow_weighted_shell_fraction_inward", np.nan))
            and float(row.get("flow_weighted_shell_fraction_inward", np.nan)) >= 0.50
            and np.isfinite(row.get("flow_core_to_shell_speed_ratio", np.nan))
            and float(row.get("flow_core_to_shell_speed_ratio", np.nan)) < 1.0
            and np.isfinite(row.get("flow_weighted_mean_divergence", np.nan))
            and float(row.get("flow_weighted_mean_divergence", np.nan)) < 0.0
        )
    order = sorted(
        range(len(rows)),
        key=lambda i: (
            bool(rows[i].get("dynamically_qualified", False)),
            float(rows[i].get("convergence_score", -np.inf)),
            float(rows[i].get("flow_support_mass_user_balanced", 0.0)),
            int(rows[i].get("region_cells_with_supported_drift", 0)),
            -abs(float(rows[i].get("convergence_center_Psi", 0.0))),
            -abs(float(rows[i].get("convergence_center_M", 0.0))),
        ),
        reverse=True,
    )
    rows = [rows[i] for i in order]
    masks = [masks[i] for i in order]
    for rid, row in enumerate(rows):
        row["region_id"] = rid
        row["primary_convergence_region"] = rid == 0
        row["flow_defined_fallback_used"] = bool(fallback_used)
        row["speed_threshold"] = thresholds.speed_threshold
        row["negative_divergence_threshold"] = thresholds.negative_divergence_threshold
        row["drift_to_diffusion_threshold"] = thresholds.drift_to_diffusion_threshold
        row["definition"] = "connected core of supported negative divergence, low drift speed, and low drift-to-diffusion ratio; occupancy maxima are never used as dynamical centres"
        row["primary_selection_rule"] = "dynamical qualification, flow-only convergence score, user-balanced drift support and supported-cell count; deterministic coordinate ordering only for exact ties; training data only"

    primary = masks[0]
    _, primary_shell, inward_cos, inward_comp = _region_metrics(stats, div, interior, primary, shell_radius, split, 0)
    grid = field_grid_table(stats, split)
    region_id_grid = np.full(primary.shape, -1, dtype=int)
    for rid, mask in enumerate(masks):
        region_id_grid[mask] = rid
    grid["interior_divergence_supported"] = interior.ravel(order="C")
    grid["interior_local_divergence"] = div.ravel(order="C")
    grid["drift_to_diffusion_ratio"] = ratio.ravel(order="C")
    grid["flow_convergence_candidate"] = candidate.ravel(order="C")
    grid["convergence_region_id"] = region_id_grid.ravel(order="C")
    grid["primary_convergence_core"] = primary.ravel(order="C")
    grid["primary_convergence_shell"] = primary_shell.ravel(order="C")
    grid["inward_cosine_to_primary_core"] = inward_cos.ravel(order="C")
    grid["inward_component_to_primary_core"] = inward_comp.ravel(order="C")
    return pd.DataFrame(rows), masks, grid, thresholds


def evaluate_frozen_convergence_region(
    stats: FieldStats,
    core_mask: np.ndarray,
    split: str,
    shell_radius: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    div, interior = interior_divergence(stats)
    summary, shell, inward_cos, inward_comp = _region_metrics(stats, div, interior, core_mask, shell_radius, split, 0)
    summary["region_definition_source"] = "training-defined frozen flow-convergence mask"
    grid = field_grid_table(stats, split)
    grid["interior_divergence_supported"] = interior.ravel(order="C")
    grid["interior_local_divergence"] = div.ravel(order="C")
    grid["frozen_primary_convergence_core"] = np.asarray(core_mask, dtype=bool).ravel(order="C")
    grid["frozen_primary_convergence_shell"] = shell.ravel(order="C")
    grid["inward_cosine_to_frozen_primary_core"] = inward_cos.ravel(order="C")
    grid["inward_component_to_frozen_primary_core"] = inward_comp.ravel(order="C")
    return pd.DataFrame([summary]), grid


def convergence_region_reproducibility(
    train_mask: np.ndarray,
    train_regions: pd.DataFrame,
    val_masks: Sequence[np.ndarray],
    val_regions: pd.DataFrame,
) -> pd.DataFrame:
    if train_regions.empty:
        return pd.DataFrame()
    tr = train_regions.iloc[0]
    best: Optional[dict] = None
    for rid, mask in enumerate(val_masks):
        inter = int(np.sum(train_mask & mask))
        union = int(np.sum(train_mask | mask))
        smaller = int(min(np.sum(train_mask), np.sum(mask)))
        jaccard = inter / union if union > 0 else np.nan
        overlap = inter / smaller if smaller > 0 else np.nan
        vr = val_regions.iloc[rid]
        distance = float(math.hypot(float(vr["convergence_center_M"]) - float(tr["convergence_center_M"]), float(vr["convergence_center_Psi"]) - float(tr["convergence_center_Psi"])))
        candidate = {
            "validation_region_id": rid,
            "mask_intersection_cells": inter,
            "mask_union_cells": union,
            "mask_jaccard": jaccard,
            "mask_overlap_coefficient": overlap,
            "convergence_center_distance": distance,
        }
        key = (jaccard if np.isfinite(jaccard) else -1.0, -distance)
        if best is None or key > best["_key"]:
            candidate["_key"] = key
            best = candidate
    if best is None:
        return pd.DataFrame([{
            "training_primary_region_cells": int(np.sum(train_mask)),
            "validation_matching_region_available": False,
        }])
    best.pop("_key", None)
    return pd.DataFrame([{
        "training_primary_region_cells": int(np.sum(train_mask)),
        "validation_matching_region_available": True,
        **best,
    }])


def convergence_radial_profile(
    stats: FieldStats,
    core_mask: np.ndarray,
    split: str,
    radial_bin_width: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize flow by distance from a frozen convergence core."""
    core = np.asarray(core_mask, dtype=bool)
    u = np.asarray(stats.drift_u, dtype=float)
    v = np.asarray(stats.drift_v, dtype=float)
    speed = np.sqrt(u * u + v * v)
    diffusion_trace = np.maximum(np.asarray(stats.diff_x) + np.asarray(stats.diff_y), 0.0)
    ratio = speed / np.maximum(np.sqrt(diffusion_trace), EPS)
    divergence, divergence_mask = interior_divergence(stats)
    support = (
        np.asarray(stats.state_mask, dtype=bool)
        & np.asarray(stats.drift_mask, dtype=bool)
        & np.isfinite(u)
        & np.isfinite(v)
    )
    dx = abs(float(np.mean(np.diff(stats.xcenters)))) if len(stats.xcenters) > 1 else 1.0
    dy = abs(float(np.mean(np.diff(stats.ycenters)))) if len(stats.ycenters) > 1 else 1.0
    distance, nearest = distance_transform_edt(~core, sampling=(dx, dy), return_indices=True)
    X, Y = np.meshgrid(stats.xcenters, stats.ycenters, indexing="ij")
    near_x = X[nearest[0], nearest[1]]
    near_y = Y[nearest[0], nearest[1]]
    to_x = near_x - X
    to_y = near_y - Y
    inward_cos = np.full_like(u, np.nan, dtype=float)
    inward_component = np.full_like(u, np.nan, dtype=float)
    outside = support & (~core) & (distance > EPS) & (speed > EPS)
    inward_cos[outside] = (u[outside] * to_x[outside] + v[outside] * to_y[outside]) / (
        speed[outside] * distance[outside]
    )
    inward_component[outside] = (u[outside] * to_x[outside] + v[outside] * to_y[outside]) / distance[outside]
    inward_cos[core & support] = np.nan
    inward_component[core & support] = np.nan

    width = float(radial_bin_width) if radial_bin_width is not None else float(max(min(dx, dy), EPS))
    radial_bin = np.floor(distance / max(width, EPS) + 1e-12).astype(int)
    h = np.asarray(stats.occupancy_weighted, dtype=float)
    flow_w = np.sqrt(np.maximum(np.asarray(stats.drift_weight, dtype=float), 0.0))
    rows: List[dict] = []
    for rb in sorted(np.unique(radial_bin[support]).tolist()):
        mask = support & (radial_bin == int(rb))
        div_mask = mask & divergence_mask
        rows.append({
            "split": split,
            "radial_bin": int(rb),
            "distance_lower": float(rb * width),
            "distance_upper": float((rb + 1) * width),
            "supported_cells": int(np.sum(mask)),
            "occupancy_probability_mass": float(np.sum(stats.occupancy_probability[mask])),
            "user_balanced_drift_support_mass": float(np.sum(stats.drift_weight[mask])),
            "occupancy_weighted_mean_speed": _weighted_grid_mean(speed, h, mask),
            "flow_weighted_mean_speed": _weighted_grid_mean(speed, flow_w, mask),
            "occupancy_weighted_mean_diffusion_trace": _weighted_grid_mean(diffusion_trace, h, mask),
            "occupancy_weighted_mean_drift_to_diffusion_ratio": _weighted_grid_mean(ratio, h, mask),
            "occupancy_weighted_mean_divergence": _weighted_grid_mean(divergence, h, div_mask),
            "occupancy_weighted_fraction_negative_divergence": _weighted_grid_fraction(divergence < 0, h, div_mask),
            "occupancy_weighted_mean_inward_cosine": _weighted_grid_mean(inward_cos, h, mask),
            "occupancy_weighted_fraction_inward": _weighted_grid_fraction(inward_cos > 0, h, mask & np.isfinite(inward_cos)),
            "occupancy_weighted_mean_inward_component": _weighted_grid_mean(inward_component, h, mask),
            "flow_weighted_mean_inward_cosine": _weighted_grid_mean(inward_cos, flow_w, mask),
            "flow_weighted_fraction_inward": _weighted_grid_fraction(inward_cos > 0, flow_w, mask & np.isfinite(inward_cos)),
            "flow_weighted_mean_inward_component": _weighted_grid_mean(inward_component, flow_w, mask),
            "radial_bin_width": width,
        })
    grid_rows: List[dict] = []
    for i, m in enumerate(stats.xcenters):
        for j, psi in enumerate(stats.ycenters):
            grid_rows.append({
                "split": split,
                "x_bin": i,
                "y_bin": j,
                "M_center": float(m),
                "Psi_center": float(psi),
                "inside_primary_convergence_core": bool(core[i, j]),
                "distance_to_primary_convergence_core": float(distance[i, j]),
                "radial_bin": int(radial_bin[i, j]),
                "supported_drift_cell": bool(support[i, j]),
                "inward_cosine_to_nearest_core_cell": float(inward_cos[i, j]),
                "inward_component_to_nearest_core_cell": float(inward_component[i, j]),
                "drift_speed": float(speed[i, j]),
                "diffusion_trace": float(diffusion_trace[i, j]),
                "drift_to_diffusion_ratio": float(ratio[i, j]),
                "occupancy_probability": float(stats.occupancy_probability[i, j]),
            })
    return pd.DataFrame(rows), pd.DataFrame(grid_rows)


def global_field_contraction_summary(stats: FieldStats, split: str) -> dict:
    divergence, interior = interior_divergence(stats)
    occupancy = np.asarray(stats.occupancy_weighted, dtype=float)
    return {
        "split": split,
        "valid_state_rows": int(stats.valid_state_rows),
        "valid_drift_rows": int(stats.valid_drift_rows),
        "users": int(stats.users),
        "state_supported_cells": int(np.sum(stats.state_mask)),
        "drift_supported_cells": int(np.sum(stats.drift_mask)),
        "interior_divergence_cells": int(np.sum(interior)),
        "weighted_mean_local_divergence_interior_only": _weighted_grid_mean(
            divergence, occupancy, interior
        ),
        "weighted_negative_divergence_fraction_interior_only": _weighted_grid_fraction(
            divergence < 0, occupancy, interior
        ),
    }


def assign_field_regions(
    df: pd.DataFrame,
    spec: CoordinateSpec,
    stats: FieldStats,
    masks: Sequence[np.ndarray],
) -> np.ndarray:
    x = pd.to_numeric(df[spec.xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[spec.ycol], errors="coerce").to_numpy(dtype=float)
    ix = digitize_closed_right(x, stats.xbins)
    iy = digitize_closed_right(y, stats.ybins)
    labels = np.full(len(df), -1, dtype=int)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (ix >= 0)
        & (ix < len(stats.xcenters))
        & (iy >= 0)
        & (iy < len(stats.ycenters))
    )
    for row in np.where(valid)[0]:
        for region_id, mask in enumerate(masks):
            if mask[ix[row], iy[row]]:
                labels[row] = region_id
                break
    return labels


def attach_region_panel_support(
    regions: pd.DataFrame,
    masks: Sequence[np.ndarray],
    df: pd.DataFrame,
    spec: CoordinateSpec,
    stats: FieldStats,
) -> pd.DataFrame:
    if regions is None or regions.empty or not masks or df is None or df.empty:
        return regions
    labels = assign_field_regions(df, spec, stats, masks)
    output = regions.copy()
    for region_id in range(len(masks)):
        inside = labels == region_id
        output.loc[
            output["region_id"] == region_id, "assigned_state_rows"
        ] = int(np.sum(inside))
        output.loc[
            output["region_id"] == region_id, "assigned_unique_users"
        ] = int(df.loc[inside, "user_id"].nunique())
    return output


def transition_counts(index_df: pd.DataFrame, states: np.ndarray, n_states: int) -> np.ndarray:
    K = int(n_states)
    counts = np.zeros((K, K), dtype=float)
    if K <= 0 or len(states) == 0:
        return counts
    d = index_df.copy()
    d["_state"] = states
    d = d.dropna(subset=["_state"]).sort_values(["user_id", "bundle_step_index"], kind="mergesort")
    if d.empty:
        return counts
    uid = d["user_id"].to_numpy(dtype=np.int64)
    step = d["bundle_step_index"].to_numpy(dtype=np.int64)
    st = d["_state"].astype(int).to_numpy()
    if len(st) < 2:
        return counts
    adj = (uid[1:] == uid[:-1]) & (step[1:] == step[:-1] + 1)
    cur = st[:-1][adj]
    nxt = st[1:][adj]
    ok = (cur >= 0) & (cur < K) & (nxt >= 0) & (nxt < K)
    if ok.any():
        flat = cur[ok] * K + nxt[ok]
        counts += np.bincount(flat, minlength=K * K).reshape(K, K)
    return counts


def observation_censoring_diagnostics(df: pd.DataFrame, spec: CoordinateSpec, label: str) -> pd.DataFrame:
    """Summarize continued-observation rates on the frozen grid."""
    x, y, _, _, state_valid, _ = spec_arrays(df, spec)
    nx = len(spec.xbins) - 1
    ny = len(spec.ybins) - 1
    ix = digitize_closed_right(x, spec.xbins)
    iy = digitize_closed_right(y, spec.ybins)
    in_range = state_valid & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not in_range.any():
        return pd.DataFrame()
    has_next = df.get("has_next_submitted_bundle", pd.Series(False, index=df.index)).astype(bool).to_numpy()
    within_h = df.get("has_next_within_observation_horizon", pd.Series(False, index=df.index)).astype(bool).to_numpy()
    long_or_none = df.get("long_gap_or_no_next", pd.Series(True, index=df.index)).astype(bool).to_numpy()
    gap = pd.to_numeric(df.get("next_gap_days", pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(dtype=float)
    uid = pd.to_numeric(df.get("user_id", pd.Series(np.arange(len(df)))), errors="coerce").to_numpy(dtype=np.int64)
    cell = ix[in_range] * ny + iy[in_range]
    rows: List[dict] = []
    tmp = pd.DataFrame({
        "cell": cell,
        "user_id": uid[in_range],
        "has_next": has_next[in_range].astype(float),
        "within_horizon": within_h[in_range].astype(float),
        "long_gap_or_no_next": long_or_none[in_range].astype(float),
        "gap_days": gap[in_range],
    })
    grouped = tmp.groupby("cell", sort=False)
    user_counts = tmp.drop_duplicates(["cell", "user_id"]).groupby("cell")["user_id"].size()
    for cid, g in grouped:
        a = int(cid) // ny
        b = int(cid) % ny
        rows.append({
            "split": label,
            "coordinate": spec.name,
            "ix": int(a),
            "iy": int(b),
            "x": float(0.5 * (spec.xbins[a] + spec.xbins[a + 1])),
            "y": float(0.5 * (spec.ybins[b] + spec.ybins[b + 1])),
            "state_rows": int(len(g)),
            "state_users": int(user_counts.get(cid, 0)),
            "has_next_submitted_rate": float(g["has_next"].mean()),
            "has_next_within_horizon_rate": float(g["within_horizon"].mean()),
            "long_gap_or_no_next_rate": float(g["long_gap_or_no_next"].mean()),
            "mean_next_gap_days_observed": float(g.loc[np.isfinite(g["gap_days"]), "gap_days"].mean()) if np.isfinite(g["gap_days"]).any() else np.nan,
            "observation_horizon_days": float(OBSERVATION_HORIZON_DAYS),
            "long_gap_days": float(LONG_GAP_DAYS),
            "interpretation": "state-dependent observation/censoring audit; not a coordinate or region-selection criterion",
        })
    return pd.DataFrame(rows)


def drift_observation_sensitivity_table(df: pd.DataFrame, spec: CoordinateSpec, label: str) -> pd.DataFrame:
    rows: List[dict] = []
    variants = [("all_rows", df)]
    if "has_next_submitted_bundle" in df.columns:
        variants.append(("next_observed_only", df[df["has_next_submitted_bundle"].astype(bool)]))
    if "long_gap_or_no_next" in df.columns:
        variants.append(("long_gap_excluded", df[~df["long_gap_or_no_next"].astype(bool)]))
    for name, sub in variants:
        if sub is None or sub.empty:
            rows.append({"split": label, "coordinate": spec.name, "variant": name, "rows": 0})
            continue
        st = occupancy_drift_stats(sub, spec)
        speed = np.sqrt(np.asarray(st["drift_u"]) ** 2 + np.asarray(st["drift_v"]) ** 2)
        mask = np.asarray(st["drift_mask"], dtype=bool)
        rows.append({
            "split": label,
            "coordinate": spec.name,
            "variant": name,
            "rows": int(len(sub)),
            "users": int(sub["user_id"].nunique()) if "user_id" in sub.columns else int(len(sub)),
            "valid_drift_rows": int(st["valid_drift_rows"]),
            "drift_bins": int(st["drift_bins"]),
            "mean_abs_drift": float(np.nanmean(speed[mask])) if np.any(mask) else np.nan,
            "observation_sensitivity_scope": "diagnostic only; regions and thresholds are not reselected",
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Coordinate analysis orchestration
# -----------------------------------------------------------------------------


def occupancy_drift_grid_table(
    stats: Dict[str, object],
    spec: CoordinateSpec,
    split: str,
) -> pd.DataFrame:
    rows: List[dict] = []
    diffusion_trace = np.maximum(
        np.asarray(stats["diff_x"], dtype=float) + np.asarray(stats["diff_y"], dtype=float),
        0.0,
    )
    speed = np.sqrt(
        np.asarray(stats["drift_u"], dtype=float) ** 2
        + np.asarray(stats["drift_v"], dtype=float) ** 2
    )
    ratio = speed / np.maximum(np.sqrt(diffusion_trace), EPS)
    for ix, x in enumerate(np.asarray(stats["xcenters"], dtype=float)):
        for iy, y in enumerate(np.asarray(stats["ycenters"], dtype=float)):
            rows.append({
                "split": split,
                "coordinate": spec.name,
                "ix": int(ix),
                "iy": int(iy),
                "x": float(x),
                "y": float(y),
                "occupancy_weighted": float(stats["occupancy_weighted"][ix, iy]),
                "occupancy_probability": float(stats["occupancy_probability"][ix, iy]),
                "empirical_quasi_potential": float(stats["potential"][ix, iy]),
                "occupancy_count": int(stats["occupancy_count"][ix, iy]),
                "user_count": int(stats["user_count"][ix, iy]),
                "drift_count": int(stats["drift_count"][ix, iy]),
                "drift_weight": float(stats["drift_weight"][ix, iy]),
                "drift_u": float(stats["drift_u"][ix, iy]),
                "drift_v": float(stats["drift_v"][ix, iy]),
                "drift_speed": float(speed[ix, iy]),
                "diff_x": float(stats["diff_x"][ix, iy]),
                "diff_y": float(stats["diff_y"][ix, iy]),
                "diff_xy": float(stats["diff_xy"][ix, iy]),
                "diffusion_trace": float(diffusion_trace[ix, iy]),
                "drift_to_diffusion_ratio": float(ratio[ix, iy]),
                "state_supported": bool(stats["state_mask"][ix, iy]),
                "drift_supported": bool(stats["drift_mask"][ix, iy]),
            })
    return pd.DataFrame(rows)


def analyze_coordinate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    confirm_df: pd.DataFrame,
    spec: CoordinateSpec,
) -> Dict[str, object]:
    spec_root = COORD_ROOT / spec.name
    region_root = REGION_ROOT / spec.name
    spec_root.mkdir(parents=True, exist_ok=True)
    region_root.mkdir(parents=True, exist_ok=True)

    split_frames = {
        "A_train": train_df,
        "A_val": val_df,
        "B_confirm": confirm_df,
    }
    stats_by_split: Dict[str, Dict[str, object]] = {}
    fields_by_split: Dict[str, FieldStats] = {}
    occupancy_summaries: List[pd.DataFrame] = []
    occupancy_ridges: List[pd.DataFrame] = []
    contraction_summaries: List[dict] = []

    for split, frame in split_frames.items():
        if frame is None or frame.empty:
            continue
        stats = occupancy_drift_stats(frame, spec)
        field = field_stats_from_dict(stats)
        stats_by_split[split] = stats
        fields_by_split[split] = field
        suffix = "_output_only" if split == "B_confirm" else ""
        write_table(
            occupancy_drift_grid_table(stats, spec, split),
            spec_root / f"{split}_occupancy_drift_grid{suffix}",
        )
        write_table(
            field_grid_table(field, split),
            spec_root / f"{split}_publication_field_grid{suffix}",
        )
        occupancy_summary, occupancy_ridge = occupancy_geometry_diagnostics(field, split)
        occupancy_summaries.append(occupancy_summary)
        occupancy_ridges.append(occupancy_ridge)
        contraction_summaries.append(global_field_contraction_summary(field, split))

    write_table(
        pd.concat(occupancy_summaries, ignore_index=True),
        spec_root / "occupancy_geometry_summaries",
    )
    write_table(
        pd.concat(occupancy_ridges, ignore_index=True),
        spec_root / "occupancy_ridge_by_M",
    )
    write_table(
        pd.DataFrame(contraction_summaries),
        spec_root / "global_field_contraction_summaries",
    )

    observation_tables: List[pd.DataFrame] = []
    sensitivity_tables: List[pd.DataFrame] = []
    for split, frame in split_frames.items():
        if frame is None or frame.empty:
            continue
        observation = observation_censoring_diagnostics(frame, spec, split)
        if not observation.empty:
            suffix = "_output_only" if split == "B_confirm" else ""
            write_table(observation, spec_root / f"{split}_observation_censoring_grid{suffix}")
            observation_tables.append(observation)
        sensitivity = drift_observation_sensitivity_table(frame, spec, split)
        if not sensitivity.empty:
            sensitivity_tables.append(sensitivity)
    if observation_tables:
        write_table(
            pd.concat(observation_tables, ignore_index=True),
            spec_root / "observation_censoring_grid_all_splits",
        )
    if sensitivity_tables:
        write_table(
            pd.concat(sensitivity_tables, ignore_index=True),
            spec_root / "drift_observation_sensitivity_all_splits",
        )

    train_field = fields_by_split["A_train"]
    val_field = fields_by_split["A_val"]
    train_regions, train_masks, train_grid, thresholds = identify_convergence_regions(
        train_field,
        "A_train",
        CONVERGENCE_SPEED_QUANTILE,
        CONVERGENCE_NEGATIVE_DIVERGENCE_QUANTILE,
        CONVERGENCE_RATIO_QUANTILE,
        CONVERGENCE_MIN_CELLS,
        CONVERGENCE_SHELL_RADIUS,
        thresholds=None,
    )
    if not train_masks:
        raise RuntimeError("No training flow-convergence core was identified.")
    train_regions = attach_region_panel_support(
        train_regions,
        train_masks,
        train_df,
        spec,
        train_field,
    )
    primary_core_mask = train_masks[0]
    np.save(region_root / "A_train_primary_convergence_core_mask.npy", primary_core_mask)

    val_frozen_region, val_frozen_grid = evaluate_frozen_convergence_region(
        val_field,
        primary_core_mask,
        "A_val",
        CONVERGENCE_SHELL_RADIUS,
    )
    val_frozen_region = attach_region_panel_support(
        val_frozen_region,
        [primary_core_mask],
        val_df,
        spec,
        val_field,
    )
    val_detected_regions, val_detected_masks, val_detected_grid, _ = identify_convergence_regions(
        val_field,
        "A_val",
        CONVERGENCE_SPEED_QUANTILE,
        CONVERGENCE_NEGATIVE_DIVERGENCE_QUANTILE,
        CONVERGENCE_RATIO_QUANTILE,
        CONVERGENCE_MIN_CELLS,
        CONVERGENCE_SHELL_RADIUS,
        thresholds=thresholds,
        allow_fallback=False,
    )
    val_detected_regions = attach_region_panel_support(
        val_detected_regions,
        val_detected_masks,
        val_df,
        spec,
        val_field,
    )
    reproducibility = convergence_region_reproducibility(
        primary_core_mask,
        train_regions,
        val_detected_masks,
        val_detected_regions,
    )

    write_table(train_regions, region_root / "training_flow_defined_convergence_regions")
    write_table(val_frozen_region, region_root / "validation_frozen_training_convergence_region")
    write_table(
        val_detected_regions,
        region_root / "validation_flow_defined_convergence_regions_fixed_thresholds",
    )
    write_table(
        reproducibility,
        region_root / "training_validation_convergence_region_reproducibility",
    )
    write_table(train_grid, region_root / "training_flow_convergence_grid")
    write_table(val_frozen_grid, region_root / "validation_frozen_convergence_grid")
    write_table(val_detected_grid, region_root / "validation_detected_convergence_grid")
    with (region_root / "training_convergence_thresholds.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "speed_threshold": float(thresholds.speed_threshold),
            "negative_divergence_threshold": float(thresholds.negative_divergence_threshold),
            "drift_to_diffusion_threshold": float(thresholds.drift_to_diffusion_threshold),
            "speed_quantile": float(thresholds.speed_quantile),
            "negative_divergence_quantile": float(thresholds.negative_divergence_quantile),
            "drift_to_diffusion_quantile": float(thresholds.drift_to_diffusion_quantile),
            "min_region_cells": int(CONVERGENCE_MIN_CELLS),
            "shell_radius": float(CONVERGENCE_SHELL_RADIUS),
            "fit_split": "A_train",
        }, handle, indent=2)

    train_radial, train_radial_grid = convergence_radial_profile(
        train_field,
        primary_core_mask,
        "A_train",
    )
    val_radial, val_radial_grid = convergence_radial_profile(
        val_field,
        primary_core_mask,
        "A_val",
    )
    write_table(
        pd.concat([train_radial, val_radial], ignore_index=True),
        region_root / "convergence_radial_profiles",
    )
    write_table(train_radial_grid, region_root / "training_convergence_radial_grid")
    write_table(val_radial_grid, region_root / "validation_convergence_radial_grid")

    confirm_frozen_region = pd.DataFrame()
    if "B_confirm" in fields_by_split:
        confirm_field = fields_by_split["B_confirm"]
        confirm_frozen_region, confirm_grid = evaluate_frozen_convergence_region(
            confirm_field,
            primary_core_mask,
            "B_confirm",
            CONVERGENCE_SHELL_RADIUS,
        )
        confirm_frozen_region = attach_region_panel_support(
            confirm_frozen_region,
            [primary_core_mask],
            confirm_df,
            spec,
            confirm_field,
        )
        write_table(
            confirm_frozen_region,
            region_root / "confirmation_frozen_training_convergence_region_output_only",
        )
        write_table(
            confirm_grid,
            region_root / "confirmation_frozen_convergence_grid_output_only",
        )
        confirm_radial, confirm_radial_grid = convergence_radial_profile(
            confirm_field,
            primary_core_mask,
            "B_confirm",
        )
        write_table(
            confirm_radial,
            region_root / "confirmation_convergence_radial_profile_output_only",
        )
        write_table(
            confirm_radial_grid,
            region_root / "confirmation_convergence_radial_grid_output_only",
        )

    train_stats = stats_by_split["A_train"]
    val_stats = stats_by_split["A_val"]
    train_primary = train_regions.iloc[0].to_dict()
    val_primary = val_frozen_region.iloc[0].to_dict()
    occupancy_all = pd.concat(occupancy_summaries, ignore_index=True).set_index("split")
    contraction_all = pd.DataFrame(contraction_summaries).set_index("split")
    radial_first = val_radial[val_radial["radial_bin"] == 1]
    match_row = reproducibility.iloc[0].to_dict() if not reproducibility.empty else {}

    output: Dict[str, object] = {
        "coordinate": spec.name,
        "role": spec.role,
        "A_train_valid_state_rows": int(train_stats["valid_state_rows"]),
        "A_val_valid_state_rows": int(val_stats["valid_state_rows"]),
        "A_train_occupied_bins": int(train_stats["occupied_bins"]),
        "A_val_occupied_bins": int(val_stats["occupied_bins"]),
        "A_train_A_val_occupancy_js": float(js_divergence(
            np.asarray(train_stats["occupancy_weighted"]).ravel() + EPS,
            np.asarray(val_stats["occupancy_weighted"]).ravel() + EPS,
        )),
        "training_primary_convergence_center_M": float(train_primary.get("convergence_center_M", np.nan)),
        "training_primary_convergence_center_Psi": float(train_primary.get("convergence_center_Psi", np.nan)),
        "training_primary_convergence_flow_localization_M_shortest_interval_50_low": float(
            train_primary.get("flow_localization_M_shortest_interval_50_low", np.nan)
        ),
        "training_primary_convergence_flow_localization_M_shortest_interval_50_high": float(
            train_primary.get("flow_localization_M_shortest_interval_50_high", np.nan)
        ),
        "training_primary_convergence_flow_localization_Psi_shortest_interval_50_low": float(
            train_primary.get("flow_localization_Psi_shortest_interval_50_low", np.nan)
        ),
        "training_primary_convergence_flow_localization_Psi_shortest_interval_50_high": float(
            train_primary.get("flow_localization_Psi_shortest_interval_50_high", np.nan)
        ),
        "training_primary_local_fixed_point_M": float(train_primary.get("local_fixed_point_M", np.nan)),
        "training_primary_local_fixed_point_Psi": float(train_primary.get("local_fixed_point_Psi", np.nan)),
        "validation_global_drift_supported_cells": int(contraction_all.loc["A_val", "drift_supported_cells"]),
        "validation_global_weighted_negative_divergence_fraction_interior_only": float(contraction_all.loc["A_val", "weighted_negative_divergence_fraction_interior_only"]),
        "validation_global_weighted_mean_local_divergence_interior_only": float(contraction_all.loc["A_val", "weighted_mean_local_divergence_interior_only"]),
        "validation_frozen_primary_convergence_occupancy_mass_fraction": float(val_primary.get("occupancy_mass_fraction", np.nan)),
        "validation_frozen_primary_convergence_flow_weighted_shell_fraction_inward": float(val_primary.get("flow_weighted_shell_fraction_inward", np.nan)),
        "validation_frozen_primary_convergence_flow_core_to_shell_speed_ratio": float(val_primary.get("flow_core_to_shell_speed_ratio", np.nan)),
        "train_validation_convergence_mask_jaccard": float(match_row.get("mask_jaccard", np.nan)),
        "train_validation_convergence_center_distance": float(match_row.get("convergence_center_distance", np.nan)),
        "validation_occupancy_Psi_shortest_interval_50_low": float(occupancy_all.loc["A_val", "Psi_shortest_interval_50_low"]),
        "validation_occupancy_Psi_shortest_interval_50_high": float(occupancy_all.loc["A_val", "Psi_shortest_interval_50_high"]),
        "validation_occupancy_M_outermost_one_bin_mass_fraction": float(occupancy_all.loc["A_val", "M_outermost_one_bin_mass_fraction"]),
        "convergence_definition": "supported negative divergence, low drift speed, and low drift-to-diffusion ratio; thresholds and primary core defined on A_train",
        "occupancy_role": "reported after core definition and not used to select, rank, or centre convergence regions",
        "B_confirm_policy": "frozen A_train core evaluated output-only; no confirmation-region detection or threshold update",
    }
    if not radial_first.empty:
        output["validation_convergence_radial_first_shell_flow_weighted_fraction_inward"] = float(radial_first.iloc[0]["flow_weighted_fraction_inward"])
        output["validation_convergence_radial_first_shell_flow_weighted_mean_inward_cosine"] = float(radial_first.iloc[0]["flow_weighted_mean_inward_cosine"])
    if not confirm_frozen_region.empty:
        confirm_primary = confirm_frozen_region.iloc[0].to_dict()
        output["confirmation_frozen_primary_convergence_occupancy_mass_fraction"] = float(confirm_primary.get("occupancy_mass_fraction", np.nan))
        output["confirmation_frozen_primary_convergence_flow_weighted_shell_fraction_inward"] = float(confirm_primary.get("flow_weighted_shell_fraction_inward", np.nan))
        output["confirmation_frozen_primary_convergence_flow_core_to_shell_speed_ratio"] = float(confirm_primary.get("flow_core_to_shell_speed_ratio", np.nan))

    with (spec_root / "coordinate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    return output


# -----------------------------------------------------------------------------
# Fixed K=6 operational mesostates
# -----------------------------------------------------------------------------
def valid_state_frame(df: pd.DataFrame, spec: CoordinateSpec) -> pd.DataFrame:
    required = ["user_id", "bundle_step_index", spec.xcol, spec.ycol]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing macrostate columns: {missing}")
    output = df.dropna(subset=[spec.xcol, spec.ycol]).copy()
    output[spec.xcol] = pd.to_numeric(output[spec.xcol], errors="coerce")
    output[spec.ycol] = pd.to_numeric(output[spec.ycol], errors="coerce")
    return output[np.isfinite(output[spec.xcol]) & np.isfinite(output[spec.ycol])]


def state_matrix(df: pd.DataFrame, spec: CoordinateSpec) -> np.ndarray:
    return df[[spec.xcol, spec.ycol]].to_numpy(dtype=float)


def relabel_by_centers(
    labels: np.ndarray,
    centers: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    mapping = {int(old): int(new) for new, old in enumerate(order.tolist())}
    remapped = np.asarray([mapping[int(label)] for label in labels], dtype=int)
    return remapped, centers[order], mapping


def fit_fixed_kmeans(
    train_df: pd.DataFrame,
    spec: CoordinateSpec,
) -> Tuple[StandardScaler, KMeans, Dict[int, int], pd.DataFrame, Dict[str, object]]:
    train_valid = valid_state_frame(train_df, spec)
    if len(train_valid) < MACROSTATE_K:
        raise RuntimeError("Not enough training states for fixed K=6 KMeans.")
    if KMEANS_FIT_MAX_ROWS > 0 and len(train_valid) > KMEANS_FIT_MAX_ROWS:
        train_fit = train_valid.sample(
            n=KMEANS_FIT_MAX_ROWS,
            random_state=RANDOM_STATE,
            weights=user_balanced_weights(train_valid),
            replace=False,
        ).copy()
    else:
        train_fit = train_valid.copy()

    scaler = StandardScaler().fit(state_matrix(train_fit, spec))
    fit_matrix = scaler.transform(state_matrix(train_fit, spec))
    fit_weights = user_balanced_weights(train_fit)
    model = KMeans(
        n_clusters=MACROSTATE_K,
        n_init=KMEANS_N_INIT,
        random_state=RANDOM_STATE,
    )
    try:
        model.fit(fit_matrix, sample_weight=fit_weights)
    except TypeError:
        model.fit(fit_matrix)

    train_labels_raw = model.predict(scaler.transform(state_matrix(train_valid, spec))).astype(int)
    centers_raw = scaler.inverse_transform(model.cluster_centers_)
    _, centers_ordered, mapping = relabel_by_centers(train_labels_raw, centers_raw)
    raw_to_ordered = np.asarray([mapping[index] for index in range(MACROSTATE_K)], dtype=int)
    standardized_centers_ordered = model.cluster_centers_[np.argsort(raw_to_ordered)]

    centers = pd.DataFrame({
        "macrostate": np.arange(MACROSTATE_K, dtype=int),
        "center_M": centers_ordered[:, 0],
        "center_Psi": centers_ordered[:, 1],
        "center_M_standardized": standardized_centers_ordered[:, 0],
        "center_Psi_standardized": standardized_centers_ordered[:, 1],
    })
    metadata = {
        "coordinate": spec.name,
        "macrostate_k": MACROSTATE_K,
        "macrostate_k_rule": "fixed a priori",
        "features": [spec.xcol, spec.ycol],
        "fit_split": "A_train",
        "fit_rows": int(len(train_fit)),
        "fit_rows_available": int(len(train_valid)),
        "fit_max_rows": int(KMEANS_FIT_MAX_ROWS),
        "user_balanced_sampling": True,
        "user_balanced_kmeans_fit": True,
        "kmeans_n_init": int(KMEANS_N_INIT),
        "random_state": int(RANDOM_STATE),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "raw_to_ordered_label": {str(key): int(value) for key, value in mapping.items()},
    }
    return scaler, model, mapping, centers, metadata


def assign_fixed_kmeans(
    df: pd.DataFrame,
    spec: CoordinateSpec,
    scaler: StandardScaler,
    model: KMeans,
    mapping: Dict[int, int],
) -> pd.DataFrame:
    valid = valid_state_frame(df, spec)
    output = df[["user_id", "bundle_step_index"]].copy()
    output["macrostate"] = np.nan
    if not valid.empty:
        raw_labels = model.predict(scaler.transform(state_matrix(valid, spec))).astype(int)
        labels = np.asarray([mapping[int(label)] for label in raw_labels], dtype=int)
        output.loc[valid.index, "macrostate"] = labels.astype(float)
    output["macrostate_observed"] = output["macrostate"].notna()
    return output


def transition_matrix_observed(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    output = np.zeros_like(counts, dtype=float)
    valid = row_sums[:, 0] > 0
    output[valid] = counts[valid] / row_sums[valid]
    return output


def empty_censored_run_table() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "user_id",
        "macrostate",
        "length",
        "start_step",
        "end_step",
        "event_observed",
        "right_censored",
        "termination_reason",
    ])


def censored_residence_runs(
    df: pd.DataFrame,
    state_col: str = "macrostate",
) -> pd.DataFrame:
    if df is None or df.empty or state_col not in df.columns:
        return empty_censored_run_table()
    data = df[["user_id", "bundle_step_index", state_col]].copy()
    data["user_id"] = pd.to_numeric(data["user_id"], errors="coerce")
    data["bundle_step_index"] = pd.to_numeric(data["bundle_step_index"], errors="coerce")
    data[state_col] = pd.to_numeric(data[state_col], errors="coerce")
    data = data.dropna(subset=["user_id", "bundle_step_index"])
    if data.empty:
        return empty_censored_run_table()
    data["user_id"] = data["user_id"].astype(np.int64)
    data["bundle_step_index"] = data["bundle_step_index"].astype(np.int64)
    data = data.sort_values(["user_id", "bundle_step_index"], kind="mergesort")

    rows: List[dict] = []

    def append_run(
        user_id: int,
        state: int,
        length: int,
        start_step: int,
        end_step: int,
        event_observed: bool,
        reason: str,
    ) -> None:
        rows.append({
            "user_id": int(user_id),
            "macrostate": int(state),
            "length": int(length),
            "start_step": int(start_step),
            "end_step": int(end_step),
            "event_observed": bool(event_observed),
            "right_censored": bool(not event_observed),
            "termination_reason": reason,
        })

    for user_id, user_rows in data.groupby("user_id", sort=False):
        current_state: Optional[int] = None
        start_step: Optional[int] = None
        previous_step: Optional[int] = None
        run_length = 0
        steps = user_rows["bundle_step_index"].to_numpy(dtype=np.int64)
        states = user_rows[state_col].to_numpy(dtype=float)
        for step_value, state_value in zip(steps, states):
            step = int(step_value)
            if current_state is not None and previous_step is not None and step != previous_step + 1:
                append_run(
                    int(user_id),
                    current_state,
                    run_length,
                    int(start_step),
                    int(previous_step),
                    False,
                    "observation_gap",
                )
                current_state = None
                start_step = None
                run_length = 0

            if not np.isfinite(state_value):
                if current_state is not None and previous_step is not None:
                    append_run(
                        int(user_id),
                        current_state,
                        run_length,
                        int(start_step),
                        int(previous_step),
                        False,
                        "macrostate_unobserved",
                    )
                    current_state = None
                    start_step = None
                    run_length = 0
                previous_step = step
                continue

            state = int(state_value)
            if current_state is None:
                current_state = state
                start_step = step
                run_length = 1
            elif state == current_state:
                run_length += 1
            else:
                append_run(
                    int(user_id),
                    current_state,
                    run_length,
                    int(start_step),
                    int(previous_step),
                    True,
                    "adjacent_state_exit",
                )
                current_state = state
                start_step = step
                run_length = 1
            previous_step = step

        if current_state is not None and previous_step is not None:
            append_run(
                int(user_id),
                current_state,
                run_length,
                int(start_step),
                int(previous_step),
                False,
                "user_sequence_end",
            )

    return pd.DataFrame(rows) if rows else empty_censored_run_table()


def kaplan_meier_ccdf(
    run_df: pd.DataFrame,
    max_length: int = MAX_RESIDENCE_LENGTH,
) -> pd.DataFrame:
    columns = [
        "residence_length",
        "km_ccdf",
        "at_risk",
        "events",
        "censored",
        "greenwood_variance",
    ]
    if run_df is None or run_df.empty:
        return pd.DataFrame(columns=columns)
    data = run_df.copy()
    data["length"] = pd.to_numeric(data["length"], errors="coerce")
    data = data[np.isfinite(data["length"]) & (data["length"] >= 1)]
    if data.empty:
        return pd.DataFrame(columns=columns)

    lengths = data["length"].astype(int).to_numpy()
    observed = data["event_observed"].astype(bool).to_numpy()
    maximum = int(min(max(int(lengths.max()), 1), max(int(max_length), 1)))
    clipped = np.minimum(lengths, maximum)
    totals = np.bincount(clipped, minlength=maximum + 1)[1:].astype(int)
    events = np.bincount(clipped[observed], minlength=maximum + 1)[1:].astype(int)
    censored = np.bincount(clipped[~observed], minlength=maximum + 1)[1:].astype(int)
    at_risk = np.cumsum(totals[::-1])[::-1].astype(int)

    survival = 1.0
    greenwood_sum = 0.0
    ccdf = np.ones(maximum, dtype=float)
    variance = np.zeros(maximum, dtype=float)
    for index in range(maximum):
        ccdf[index] = survival
        variance[index] = survival * survival * greenwood_sum
        risk = int(at_risk[index])
        exits = int(events[index])
        if risk > 0 and exits > 0:
            survival *= max(1.0 - exits / risk, 0.0)
            if risk > exits:
                greenwood_sum += exits / (risk * (risk - exits))

    return pd.DataFrame({
        "residence_length": np.arange(1, maximum + 1, dtype=int),
        "km_ccdf": ccdf,
        "at_risk": at_risk,
        "events": events,
        "censored": censored,
        "greenwood_variance": variance,
    })


def restricted_mean_from_km(km: pd.DataFrame, tau: int) -> Tuple[float, float]:
    if km is None or km.empty or tau < 1:
        return np.nan, np.nan
    data = km[km["residence_length"] <= int(tau)]
    if data.empty:
        return np.nan, np.nan
    survival = data["km_ccdf"].to_numpy(dtype=float)
    times = data["residence_length"].to_numpy(dtype=int)
    risk = data["at_risk"].to_numpy(dtype=float)
    events = data["events"].to_numpy(dtype=float)
    variance = 0.0
    for time_value, risk_value, event_value in zip(times, risk, events):
        if event_value <= 0 or risk_value <= event_value:
            continue
        tail_area = float(np.sum(survival[times > time_value]))
        variance += tail_area * tail_area * event_value / (risk_value * (risk_value - event_value))
    return float(np.sum(survival)), float(math.sqrt(max(variance, 0.0)))


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite = np.where(np.isfinite(values))[0]
    if finite.size == 0:
        return adjusted
    order = finite[np.argsort(values[finite])]
    ranked = values[order]
    corrected = ranked * finite.size / np.arange(1, finite.size + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    adjusted[order] = np.clip(corrected, 0.0, 1.0)
    return adjusted


def residence_significance(
    transition: np.ndarray,
    run_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []
    for state in range(int(transition.shape[0])):
        state_runs = run_df[run_df["macrostate"].astype(int) == state] if not run_df.empty else empty_censored_run_table()
        if state_runs.empty:
            rows.append({
                "macrostate": state,
                "n_runs": 0,
                "n_completed_exits": 0,
                "n_right_censored": 0,
                "right_censoring_fraction": np.nan,
                "self_transition": np.nan,
                "rmst_tau": np.nan,
                "obs_restricted_mean_residence": np.nan,
                "obs_restricted_mean_residence_se": np.nan,
                "geo_null_restricted_mean": np.nan,
                "restricted_mean_residence_lift": np.nan,
                "reference_length": int(RESIDENCE_REFERENCE_LENGTH),
                "reference_at_risk": 0,
                "observed_tail_probability_at_reference": np.nan,
                "geometric_tail_probability_at_reference": np.nan,
                "tail_ratio_at_reference": np.nan,
                "tail_excess_pvalue_greenwood": np.nan,
            })
            continue

        km = kaplan_meier_ccdf(state_runs)
        self_transition = float(np.clip(transition[state, state], 1e-6, 1.0 - 1e-6))
        reliable = km[km["at_risk"] >= max(MIN_RESIDENCE_AT_RISK, 1)]
        tau = int(reliable["residence_length"].max()) if not reliable.empty else int(km["residence_length"].max())
        observed_rmst, observed_rmst_se = restricted_mean_from_km(km, tau)
        lengths = np.arange(1, tau + 1, dtype=int)
        geometric_rmst = float(np.sum(self_transition ** (lengths - 1)))
        lift = float(observed_rmst / geometric_rmst) if geometric_rmst > 0 else np.nan

        reference = max(1, RESIDENCE_REFERENCE_LENGTH)
        reference_row = km[km["residence_length"] == reference]
        if reference_row.empty:
            observed_tail = np.nan
            at_risk = 0
            variance = np.nan
        else:
            observed_tail = float(reference_row["km_ccdf"].iloc[0])
            at_risk = int(reference_row["at_risk"].iloc[0])
            variance = float(reference_row["greenwood_variance"].iloc[0])
        geometric_tail = float(self_transition ** (reference - 1))
        tail_ratio = float(observed_tail / geometric_tail) if np.isfinite(observed_tail) and geometric_tail > 0 else np.nan
        standard_error = math.sqrt(max(variance, 0.0)) if np.isfinite(variance) else np.nan
        if np.isfinite(observed_tail) and np.isfinite(standard_error) and standard_error > 0 and at_risk >= MIN_RESIDENCE_AT_RISK:
            z_score = (observed_tail - geometric_tail) / standard_error
            pvalue = float(0.5 * math.erfc(z_score / math.sqrt(2.0)))
        else:
            pvalue = np.nan

        n_runs = int(len(state_runs))
        n_completed = int(state_runs["event_observed"].astype(bool).sum())
        n_censored = int(state_runs["right_censored"].astype(bool).sum())
        rows.append({
            "macrostate": state,
            "n_runs": n_runs,
            "n_completed_exits": n_completed,
            "n_right_censored": n_censored,
            "right_censoring_fraction": float(n_censored / n_runs),
            "self_transition": self_transition,
            "rmst_tau": tau,
            "obs_restricted_mean_residence": observed_rmst,
            "obs_restricted_mean_residence_se": observed_rmst_se,
            "geo_null_restricted_mean": geometric_rmst,
            "restricted_mean_residence_lift": lift,
            "reference_length": reference,
            "reference_at_risk": at_risk,
            "observed_tail_probability_at_reference": observed_tail,
            "geometric_tail_probability_at_reference": geometric_tail,
            "tail_ratio_at_reference": tail_ratio,
            "tail_excess_pvalue_greenwood": pvalue,
        })

    output = pd.DataFrame(rows)
    output["tail_excess_qvalue_bh"] = benjamini_hochberg(
        pd.to_numeric(output["tail_excess_pvalue_greenwood"], errors="coerce").to_numpy(dtype=float)
    )
    output["tail_excess_significant_bh"] = (
        np.isfinite(output["tail_excess_qvalue_bh"])
        & (output["tail_excess_qvalue_bh"] < RESIDENCE_P_THRESHOLD)
        & (output["observed_tail_probability_at_reference"] > output["geometric_tail_probability_at_reference"])
    )
    output["candidate_basin"] = (
        (output["n_completed_exits"] >= 5)
        & (output["reference_at_risk"] >= MIN_RESIDENCE_AT_RISK)
        & (output["tail_ratio_at_reference"] > 1.0)
        & output["tail_excess_significant_bh"]
        & (output["self_transition"] >= RESIDENCE_SELF_TRANSITION_THRESHOLD)
    )
    return output


def residence_curve_table(
    transition: np.ndarray,
    run_df: pd.DataFrame,
) -> pd.DataFrame:
    tables: List[pd.DataFrame] = []
    if run_df is None or run_df.empty:
        return pd.DataFrame()
    for state in sorted(run_df["macrostate"].astype(int).unique()):
        km = kaplan_meier_ccdf(run_df[run_df["macrostate"].astype(int) == state])
        if km.empty:
            continue
        self_transition = float(np.clip(transition[state, state], 1e-6, 1.0 - 1e-6))
        km.insert(0, "macrostate", int(state))
        km["self_transition"] = self_transition
        km["geometric_ccdf"] = self_transition ** (km["residence_length"].to_numpy(dtype=int) - 1)
        tables.append(km)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    aa = a[mask] - float(np.mean(a[mask]))
    bb = b[mask] - float(np.mean(b[mask]))
    denominator = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denominator) if denominator > EPS else np.nan


def split_reproducibility_scores(
    train_stats: Dict[str, object],
    val_stats: Dict[str, object],
    train_transition: np.ndarray,
    val_transition: np.ndarray,
    train_residence: pd.DataFrame,
    val_residence: pd.DataFrame,
) -> Dict[str, object]:
    train_occupancy = np.asarray(train_stats["occupancy_probability"], dtype=float).ravel() + EPS
    val_occupancy = np.asarray(val_stats["occupancy_probability"], dtype=float).ravel() + EPS
    common = np.asarray(train_stats["drift_mask"], dtype=bool) & np.asarray(val_stats["drift_mask"], dtype=bool)
    if common.any():
        train_u = np.asarray(train_stats["drift_u"], dtype=float)[common]
        train_v = np.asarray(train_stats["drift_v"], dtype=float)[common]
        val_u = np.asarray(val_stats["drift_u"], dtype=float)[common]
        val_v = np.asarray(val_stats["drift_v"], dtype=float)[common]
        train_speed = np.sqrt(train_u * train_u + train_v * train_v)
        val_speed = np.sqrt(val_u * val_u + val_v * val_v)
        valid_cosine = (train_speed > EPS) & (val_speed > EPS)
        local_cosine = (
            float(np.mean((train_u[valid_cosine] * val_u[valid_cosine] + train_v[valid_cosine] * val_v[valid_cosine]) / (train_speed[valid_cosine] * val_speed[valid_cosine])))
            if valid_cosine.any()
            else np.nan
        )
        component_rmse = float(np.sqrt(np.mean(np.concatenate([train_u - val_u, train_v - val_v]) ** 2)))
        speed_correlation = pearson_safe(train_speed, val_speed)
    else:
        local_cosine = np.nan
        component_rmse = np.nan
        speed_correlation = np.nan

    row_tv = 0.5 * np.sum(np.abs(train_transition - val_transition), axis=1)
    train_residence_index = train_residence.set_index("macrostate")
    val_residence_index = val_residence.set_index("macrostate")
    common_states = sorted(set(train_residence_index.index) & set(val_residence_index.index))

    def mean_abs_log_difference(column: str) -> float:
        train_values = np.asarray([train_residence_index.loc[state, column] for state in common_states], dtype=float)
        val_values = np.asarray([val_residence_index.loc[state, column] for state in common_states], dtype=float)
        mask = np.isfinite(train_values) & np.isfinite(val_values) & (train_values > 0) & (val_values > 0)
        return float(np.mean(np.abs(np.log(val_values[mask] / train_values[mask])))) if mask.any() else np.nan

    censoring_difference = np.asarray([
        abs(
            float(val_residence_index.loc[state, "right_censoring_fraction"])
            - float(train_residence_index.loc[state, "right_censoring_fraction"])
        )
        for state in common_states
    ], dtype=float)
    return {
        "occupancy_js_divergence": float(js_divergence(train_occupancy, val_occupancy)),
        "common_supported_drift_cells": int(common.sum()),
        "mean_local_drift_cosine": local_cosine,
        "drift_component_rmse": component_rmse,
        "drift_speed_pearson": speed_correlation,
        "transition_mean_row_total_variation": float(np.mean(row_tv)),
        "transition_max_row_total_variation": float(np.max(row_tv)),
        "residence_rmst_mean_abs_log_ratio": mean_abs_log_difference("restricted_mean_residence_lift"),
        "residence_tail_ratio_mean_abs_log_ratio": mean_abs_log_difference("tail_ratio_at_reference"),
        "residence_right_censoring_fraction_mean_abs_difference": float(np.nanmean(censoring_difference)) if censoring_difference.size else np.nan,
    }


def run_fixed_kmeans_analysis(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    confirm_df: pd.DataFrame,
    spec: CoordinateSpec,
) -> Dict[str, object]:
    scaler, model, mapping, centers, metadata = fit_fixed_kmeans(train_df, spec)
    write_table(centers, MESOSTATE_ROOT / "fixed_k6_centers")
    with (MESOSTATE_ROOT / "fixed_k6_model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    frames = {"A_train": train_df, "A_val": val_df, "B_confirm": confirm_df}
    transition_counts_by_split: Dict[str, np.ndarray] = {}
    transitions: Dict[str, np.ndarray] = {}
    residence_summaries: Dict[str, pd.DataFrame] = {}
    assignment_paths: Dict[str, str] = {}
    for split, frame in frames.items():
        if frame is None or frame.empty:
            continue
        assigned = assign_fixed_kmeans(frame, spec, scaler, model, mapping)
        assignment_path = write_table(assigned, MESOSTATE_ROOT / f"{split}_fixed_k6_assignments")
        assignment_paths[split] = str(assignment_path)
        counts = transition_counts(
            assigned[["user_id", "bundle_step_index"]],
            assigned["macrostate"].to_numpy(),
            n_states=MACROSTATE_K,
        )
        transition = transition_matrix_observed(counts)
        transition_counts_by_split[split] = counts
        transitions[split] = transition
        write_table(pd.DataFrame(counts), MESOSTATE_ROOT / f"{split}_fixed_k6_transition_counts")
        write_table(pd.DataFrame(transition), MESOSTATE_ROOT / f"{split}_fixed_k6_transition_matrix")
        runs = censored_residence_runs(assigned)
        write_table(runs, MESOSTATE_ROOT / f"{split}_fixed_k6_residence_runs")
        residence = residence_significance(transition, runs)
        residence_summaries[split] = residence
        write_table(residence, MESOSTATE_ROOT / f"{split}_fixed_k6_residence_summary")
        write_table(
            residence_curve_table(transition, runs),
            MESOSTATE_ROOT / f"{split}_fixed_k6_residence_curves",
        )

    fit_table = pd.DataFrame([{
        "k": MACROSTATE_K,
        "fit_split": "A_train",
        "fit_rows": metadata["fit_rows"],
        "A_train_trace": float(np.trace(transitions["A_train"])),
        "A_val_trace": float(np.trace(transitions["A_val"])),
        "A_train_transition_count": int(transition_counts_by_split["A_train"].sum()),
        "A_val_transition_count": int(transition_counts_by_split["A_val"].sum()),
        "selected": True,
        "selection_rule": "K fixed at 6 before transition and residence analysis",
    }])
    write_table(fit_table, MESOSTATE_ROOT / "fixed_k6_fit_table")

    train_stats = occupancy_drift_stats(train_df, spec)
    val_stats = occupancy_drift_stats(val_df, spec)
    reproducibility = split_reproducibility_scores(
        train_stats,
        val_stats,
        transitions["A_train"],
        transitions["A_val"],
        residence_summaries["A_train"],
        residence_summaries["A_val"],
    )
    write_table(pd.DataFrame([reproducibility]), MESOSTATE_ROOT / "A_train_A_val_reproducibility_summary")
    output = {
        **metadata,
        "assignment_paths": assignment_paths,
        "reproducibility": reproducibility,
        "B_confirm_policy": "assigned with the frozen A_train scaler, centers, and label mapping",
    }
    with (MESOSTATE_ROOT / "fixed_k6_analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    return output


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ensure_dirs()
    preprocess_manifest: Dict[str, object] = {}
    preprocess_manifest_path = META_ROOT / "preprocess_manifest.json"
    if preprocess_manifest_path.exists():
        with preprocess_manifest_path.open("r", encoding="utf-8") as handle:
            preprocess_manifest = json.load(handle)

    all_user_ids = load_user_ids()
    splits = create_splits(all_user_ids)
    split_map = {
        int(user_id): split
        for split, user_ids in splits.items()
        for user_id in user_ids.tolist()
    }
    split_manifest = {
        "random_state": RANDOM_STATE,
        "source_user_count": int(len(all_user_ids)),
        "sizes": {name: int(len(user_ids)) for name, user_ids in splits.items()},
        "B_confirm_policy": "output-only after training definitions and fixed K=6 centres are frozen",
    }
    with (SPLIT_ROOT / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2)
    for split, user_ids in splits.items():
        write_table(pd.DataFrame({"user_id": user_ids}), SPLIT_ROOT / f"{split}_users")

    chunk_audit = verify_user_complete_chunks()
    content = ContentsBuilder().build()
    priors = estimate_tag_priors(content, splits["A_train"])
    write_table(priors.table, META_OUT_ROOT / "A_train_tag_correctness_priors")
    write_table(priors.question_table, META_OUT_ROOT / "A_train_question_itemEB_correctness_priors")

    raw_paths = build_raw_panels(content, priors, split_map)
    finalize_manifest = finalize_stage1_panels(raw_paths)

    train_df = read_core_split("A_train")
    val_df = read_core_split("A_val")
    confirm_df = read_core_split("B_confirm")
    spec = coordinate_specs()[0]
    required = [spec.xcol, spec.ycol, spec.dxcol, spec.dycol]
    for split, frame in {"A_train": train_df, "A_val": val_df}.items():
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise KeyError(f"{split} is missing primary state columns: {missing}")

    coordinate_summary = analyze_coordinate(train_df, val_df, confirm_df, spec)
    mesostate_summary = run_fixed_kmeans_analysis(train_df, val_df, confirm_df, spec)

    manifest = {
        "script": Path(__file__).name,
        "stage": "empirical effective dynamics",
        "input_preprocess_manifest": preprocess_manifest,
        "split_manifest": split_manifest,
        "bundle_chunk_user_completeness_audit": chunk_audit,
        "finalize_manifest": finalize_manifest,
        "primary_state": {
            "coordinates": ["M_response_prebalanced", "activity_alignment_order_Psi"],
            "M": "decayed item-conditioned signed response residual divided by absolute residual evidence mass",
            "Psi": "decayed aligned-minus-off-target activity divided by active-plus-idle activity mass",
            "drift": "next submitted-bundle pre-state minus current pre-state",
        },
        "auxiliary_accounting": {
            "response_evidence_mass": "retained for the frozen mechanism update",
            "response_evidence_maturity_V": "derived from evidence mass for accounting and diagnostics",
            "activity_mass_components": "retained to reconstruct the Psi numerator and denominator",
            "access_and_process_fields": "context-only outputs from preprocessing",
        },
        "coordinate_summary": coordinate_summary,
        "convergence_core_contract": {
            "source_algorithm": "publication convergence statistics v5",
            "fit_split": "A_train",
            "speed_quantile": float(CONVERGENCE_SPEED_QUANTILE),
            "negative_divergence_quantile": float(
                CONVERGENCE_NEGATIVE_DIVERGENCE_QUANTILE
            ),
            "drift_to_diffusion_quantile": float(CONVERGENCE_RATIO_QUANTILE),
            "minimum_connected_cells": int(CONVERGENCE_MIN_CELLS),
            "shell_radius": float(CONVERGENCE_SHELL_RADIUS),
            "occupancy_used_for_selection": False,
            "validation_thresholds_frozen": True,
            "confirmation_region_redefined": False,
        },
        "fixed_k6_mesostate_summary": mesostate_summary,
        "visualization_outputs": "none; figures are generated by a separate script from the saved panels and tables",
        "guardrails": [
            "response correctness is the only signed response evidence",
            "support and study events update exposure alignment only",
            "access events remain context only",
            "B_confirm does not define priors, coordinates, convergence thresholds, cores, centres, or labels",
            "the convergence core is selected from A_train flow only; occupancy is descriptive",
            "K is fixed at 6 and is not selected from candidate values",
        ],
    }
    with (META_OUT_ROOT / "stage1_empirical_v3_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print("EdNet-KT4 Stage-1 empirical effective dynamics completed.")


if __name__ == "__main__":
    main()
