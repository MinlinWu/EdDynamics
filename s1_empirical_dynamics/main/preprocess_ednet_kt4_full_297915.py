#!/usr/bin/env python3
"""Preprocess EdNet-KT4 into canonical event, attempt, study, and audit tables."""

from __future__ import annotations

import gc
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths and schema constants.

KT4_INPUT = Path(os.environ.get("EDNET_KT4_INPUT", "/data/datasets/KT4/KT4"))
CONTENTS_INPUT = Path(os.environ.get("EDNET_CONTENTS_INPUT", "/data/datasets/KT4/contents"))
OUTPUT_ROOT = Path(os.environ.get("EDNET_OUTPUT_ROOT", "/data/datasets/KT4/data_297915"))
EXPECTED_KT4_USERS = 297915

BASE_USER_COLUMNS = [
    "timestamp",
    "action_type",
    "item_id",
    "source",
    "user_answer",
    "platform",
]
OPTIONAL_USER_COLUMNS = ["cursor_time"]

# Canonical action names.
ACTION_ALIASES = {
    "eliminate_choice": "erase_choice",
    "erase": "erase_choice",
    "undo_eliminate_choice": "undo_erase_choice",
}

CORE_ACTIONS = {"enter", "respond", "submit", "quit"}
CHOICE_PROCESS_ACTIONS = {"erase_choice", "undo_erase_choice"}
MEDIA_PROCESS_ACTIONS = {"play_audio", "pause_audio", "play_video", "pause_video"}
TEXT_PROCESS_ACTIONS = {"text_enter"}
ACCESS_CONTEXT_ACTIONS = {"pay", "refund", "enroll_coupon"}
KNOWN_ACTIONS = CORE_ACTIONS | CHOICE_PROCESS_ACTIONS | MEDIA_PROCESS_ACTIONS | TEXT_PROCESS_ACTIONS | ACCESS_CONTEXT_ACTIONS

KNOWN_ITEM_PREFIX = {"b", "q", "e", "l", "p", "c"}
ANSWER_CODE = {"": 0, "a": 1, "b": 2, "c": 3, "d": 4}
ACTION_CODE = {
    "enter": 1,
    "respond": 2,
    "submit": 3,
    "quit": 4,
    "erase_choice": 5,
    "undo_erase_choice": 6,
    "play_audio": 7,
    "pause_audio": 8,
    "play_video": 9,
    "pause_video": 10,
    "text_enter": 11,
    "pay": 12,
    "refund": 13,
    "enroll_coupon": 14,
}
ACTION_GROUP_CODE = {
    "core": 1,
    "choice_process": 2,
    "media_process": 3,
    "text_process": 4,
    "access_context": 5,
}
ITEM_TYPE_CODE = {
    "bundle": 1,
    "question": 2,
    "explanation": 3,
    "lecture": 4,
    "payment_item": 5,
    "coupon": 6,
}
PLATFORM_CODE = {"unknown": 0, "mobile": 1, "web": 2}
SOURCE_CODE = {
    "unknown": 0,
    "sprint": 1,
    "todays_recommendation::sprint": 2,
    "todays_recommendation::review_quiz": 3,
    "adaptive_offer": 4,
    "tutor": 5,
    "in_review": 6,
    "after_sprint": 7,
    "after_review": 8,
    "my_note": 9,
    "archive": 10,
    "todays_recommendation::lecture": 11,
    "diagnosis": 12,
}
SOURCE_STANDARDIZATION = {
    "todays_recommendatin::lecture": "todays_recommendation::lecture",
    "todays recommendation::lecture": "todays_recommendation::lecture",
}
ACTION_SORT_PRIORITY = {
    "pay": -1,
    "refund": -1,
    "enroll_coupon": -1,
    "enter": 0,
    "text_enter": 1,
    "play_audio": 2,
    "play_video": 2,
    "pause_audio": 3,
    "pause_video": 3,
    "erase_choice": 4,
    "undo_erase_choice": 4,
    "respond": 5,
    "submit": 6,
    "quit": 7,
}

ACTION_NAME_FROM_CODE = {v: k for k, v in ACTION_CODE.items()}
ITEM_TYPE_NAME_FROM_CODE = {v: k for k, v in ITEM_TYPE_CODE.items()}
ITEM_PREFIX_FROM_TYPE_CODE = {
    ITEM_TYPE_CODE["bundle"]: "b",
    ITEM_TYPE_CODE["question"]: "q",
    ITEM_TYPE_CODE["explanation"]: "e",
    ITEM_TYPE_CODE["lecture"]: "l",
    ITEM_TYPE_CODE["payment_item"]: "p",
    ITEM_TYPE_CODE["coupon"]: "c",
}

REPORT_EVERY_USERS = int(os.environ.get("REPORT_EVERY_USERS", "200"))
REPORT_EVERY_SHARDS = int(os.environ.get("REPORT_EVERY_SHARDS", "5"))

FLUSH_EVENTS_ROWS = int(os.environ.get("FLUSH_EVENTS_ROWS", "2000000"))
FLUSH_EVENTS_LIGHT_ROWS = int(os.environ.get("FLUSH_EVENTS_LIGHT_ROWS", "2000000"))
FLUSH_SKIPPED_USERS_ROWS = int(os.environ.get("FLUSH_SKIPPED_USERS_ROWS", "20000"))
FLUSH_QUESTION_ROWS = int(os.environ.get("FLUSH_QUESTION_ROWS", "600000"))
FLUSH_BUNDLE_ROWS = int(os.environ.get("FLUSH_BUNDLE_ROWS", "300000"))
FLUSH_STUDY_ROWS = int(os.environ.get("FLUSH_STUDY_ROWS", "300000"))
FLUSH_USER_SUMMARY_ROWS = int(os.environ.get("FLUSH_USER_SUMMARY_ROWS", "50000"))
FLUSH_CHOICE_PROCESS_ROWS = int(os.environ.get("FLUSH_CHOICE_PROCESS_ROWS", "300000"))
FLUSH_MEDIA_PROCESS_ROWS = int(os.environ.get("FLUSH_MEDIA_PROCESS_ROWS", "300000"))

CLEAN_OUTPUT_DIRS = os.environ.get("CLEAN_OUTPUT_DIRS", "1") == "1"

USER_READ_DTYPES = {
    "action_type": "string",
    "item_id": "string",
    "source": "string",
    "user_answer": "string",
    "platform": "string",
    "cursor_time": "string",
}


def ensure_parquet_support() -> None:
    try:
        import pyarrow  # noqa: F401
        return
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return
        except Exception as exc:
            raise RuntimeError(
                "Parquet engine not available. Install pyarrow or fastparquet before running this script. "
                "csv.gz fallback is intentionally disabled."
            ) from exc


class ProgressVisualizer:
    def __init__(self, total_units: int, unit_name: str, metadata_dir: Path, stage_name: str, report_every: int) -> None:
        self.total_units = total_units
        self.unit_name = unit_name
        self.metadata_dir = metadata_dir
        self.stage_name = stage_name
        self.report_every = max(1, report_every)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.processed_units = 0
        self.valid_units = 0
        self.skipped_units = 0
        self.flush_count = 0
        self.start_time = time.time()
        self.last_snapshot_time = 0.0
        self.snapshot_path = self.metadata_dir / f"progress_{stage_name}.json"

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        if not math.isfinite(seconds) or seconds < 0:
            return "unknown"
        seconds_int = int(seconds)
        hours, rem = divmod(seconds_int, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _snapshot_payload(self) -> dict:
        elapsed = time.time() - self.start_time
        rate = self.processed_units / elapsed if elapsed > 0 else 0.0
        remaining = max(self.total_units - self.processed_units, 0)
        eta_seconds = remaining / rate if rate > 0 else float("inf")
        return {
            "stage_name": self.stage_name,
            "unit_name": self.unit_name,
            "total_units": self.total_units,
            "processed_units": self.processed_units,
            "remaining_units": remaining,
            "valid_units": self.valid_units,
            "skipped_units": self.skipped_units,
            "flush_count": self.flush_count,
            "elapsed_seconds": elapsed,
            "units_per_second": rate,
            "eta_seconds": eta_seconds,
        }

    def write_snapshot(self) -> None:
        with self.snapshot_path.open("w", encoding="utf-8") as f:
            json.dump(self._snapshot_payload(), f, indent=2)

    def update(self, was_valid: bool = True, force_print: bool = False) -> None:
        self.processed_units += 1
        if was_valid:
            self.valid_units += 1
        else:
            self.skipped_units += 1
        should_print = force_print or (self.processed_units % self.report_every == 0)
        if should_print:
            payload = self._snapshot_payload()
            print(
                f"[Progress:{self.stage_name}] processed={payload['processed_units']}/{payload['total_units']} "
                f"{self.unit_name}, remaining={payload['remaining_units']}, valid={payload['valid_units']}, "
                f"skipped={payload['skipped_units']}, flushes={payload['flush_count']}, "
                f"elapsed={self._format_seconds(payload['elapsed_seconds'])}, "
                f"eta={self._format_seconds(payload['eta_seconds'])}"
            )
        now = time.time()
        if force_print or (now - self.last_snapshot_time >= 10.0):
            self.write_snapshot()
            self.last_snapshot_time = now

    def record_flush(self, reason: str) -> None:
        self.flush_count += 1
        payload = self._snapshot_payload()
        print(
            f"[Flush:{self.stage_name}] reason={reason}, "
            f"processed={payload['processed_units']}/{payload['total_units']}, "
            f"remaining={payload['remaining_units']}, flush_count={self.flush_count}"
        )
        self.write_snapshot()
        self.last_snapshot_time = time.time()

    def finalize(self) -> None:
        self.write_snapshot()
        payload = self._snapshot_payload()
        print(
            f"[Done:{self.stage_name}] processed={payload['processed_units']}/{payload['total_units']} "
            f"{self.unit_name}, remaining={payload['remaining_units']}, valid={payload['valid_units']}, "
            f"skipped={payload['skipped_units']}, flushes={payload['flush_count']}"
        )


class ColumnarTableBuffer:
    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.data = {column: [] for column in columns}
        self.row_count = 0

    def append_row(self, row: Dict[str, object]) -> None:
        for column in self.columns:
            self.data[column].append(row.get(column))
        self.row_count += 1

    def extend_from_dataframe(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        for column in self.columns:
            self.data[column].extend(df[column].tolist())
        self.row_count += len(df)

    def to_dataframe(self) -> pd.DataFrame:
        if self.row_count == 0:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(self.data, columns=self.columns)

    def clear(self) -> None:
        self.data = {column: [] for column in self.columns}
        self.row_count = 0

    def __len__(self) -> int:
        return self.row_count


class SaveManager:
    def __init__(self, output_root: Path, metadata_dir: Path) -> None:
        ensure_parquet_support()
        self.output_root = output_root
        self.metadata_dir = metadata_dir
        self.counters = defaultdict(int)
        self.registry_records: List[dict] = []

    def write(self, df: pd.DataFrame, out_dir: Path, prefix: str, label: str, stage: str) -> Optional[Path]:
        if df is None or df.empty:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = self.counters[str(out_dir / prefix)]
        self.counters[str(out_dir / prefix)] += 1
        out_path = out_dir / f"{prefix}_{idx:05d}.parquet"
        df.to_parquet(out_path, index=False)
        record = {
            "stage": stage,
            "label": label,
            "path": str(out_path),
            "filename": out_path.name,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "shard_index": int(idx),
            "user_id_min": None,
            "user_id_max": None,
        }
        if "user_id" in df.columns and not df["user_id"].empty:
            user_numeric = pd.to_numeric(df["user_id"], errors="coerce")
            if user_numeric.notna().any():
                record["user_id_min"] = int(user_numeric.min())
                record["user_id_max"] = int(user_numeric.max())
        self.registry_records.append(record)
        return out_path

    def write_single(self, df: pd.DataFrame, out_path: Path, label: str, stage: str) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix != ".parquet":
            out_path = out_path.with_suffix(".parquet")
        df.to_parquet(out_path, index=False)
        self.registry_records.append(
            {
                "stage": stage,
                "label": label,
                "path": str(out_path),
                "filename": out_path.name,
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "shard_index": -1,
                "user_id_min": None,
                "user_id_max": None,
            }
        )
        return out_path

    def write_registry(self) -> Optional[Path]:
        if not self.registry_records:
            return None
        df = pd.DataFrame(self.registry_records)
        out_path = self.metadata_dir / "shard_registry.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        return out_path


def normalize_text_series(series: pd.Series, fill_value: str = "") -> pd.Series:
    return (
        series.fillna(fill_value)
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({"<na>": fill_value, "nan": fill_value, "none": fill_value})
    )


def normalize_source_value(value: str) -> str:
    if value is None:
        return "unknown"
    value = str(value).strip().lower()
    if value in {"", "nan", "none", "<na>"}:
        return "unknown"
    return SOURCE_STANDARDIZATION.get(value, value)


def normalize_action_value(value: str) -> str:
    if value is None:
        return ""
    value = str(value).strip().lower()
    if value in {"", "nan", "none", "<na>"}:
        return ""
    return ACTION_ALIASES.get(value, value)


def action_group(action: str) -> str:
    if action in CORE_ACTIONS:
        return "core"
    if action in CHOICE_PROCESS_ACTIONS:
        return "choice_process"
    if action in MEDIA_PROCESS_ACTIONS:
        return "media_process"
    if action in TEXT_PROCESS_ACTIONS:
        return "text_process"
    if action in ACCESS_CONTEXT_ACTIONS:
        return "access_context"
    return "unknown"


def answer_to_code(value: str) -> int:
    return ANSWER_CODE.get(value, 0)


def item_prefix_to_type(prefix: str) -> Optional[str]:
    if prefix == "b":
        return "bundle"
    if prefix == "q":
        return "question"
    if prefix == "e":
        return "explanation"
    if prefix == "l":
        return "lecture"
    if prefix == "p":
        return "payment_item"
    if prefix == "c":
        return "coupon"
    return None


def parse_numeric_id(text: object) -> Optional[int]:
    if text is None or pd.isna(text):
        return None
    text = str(text).strip()
    if len(text) < 2:
        return None
    suffix = text[1:]
    if suffix.isdigit():
        return int(suffix)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return int(digits)
    return None


def extract_numeric_suffix_series(series: pd.Series) -> pd.Series:
    normalized = normalize_text_series(series)
    suffix = normalized.str[1:]
    direct = pd.to_numeric(suffix.where(suffix.str.isdigit(), pd.NA), errors="coerce")
    missing_mask = direct.isna() & normalized.str.len().gt(0)
    if missing_mask.any():
        fallback = pd.to_numeric(normalized[missing_mask].str.extract(r"(\d+)", expand=False), errors="coerce")
        direct.loc[missing_mask] = fallback
    return direct.astype("Int32")


def parse_tags_string(value: object) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if value in {"", "-1", "nan", "None", "<NA>"}:
        return ""
    parts = [p.strip() for p in value.split(";") if p.strip()]
    ints = sorted({int(p) for p in parts if p.lstrip("-").isdigit() and int(p) >= 0})
    return ";".join(str(x) for x in ints)


def explode_tag_edges(df: pd.DataFrame, id_col: str, tags_col: str) -> pd.DataFrame:
    rows: List[Tuple[str, int]] = []
    for item_id, tags_raw in df[[id_col, tags_col]].itertuples(index=False):
        if not tags_raw:
            continue
        for token in str(tags_raw).split(";"):
            if token:
                rows.append((item_id, int(token)))
    return pd.DataFrame(rows, columns=[id_col, "tag"])


def mode_or_first(values: pd.Series) -> Optional[int]:
    values = values.dropna()
    if values.empty:
        return None
    modes = values.mode(dropna=True)
    if modes.empty:
        return int(values.iloc[0])
    return int(modes.iloc[0])


def union_tags(values: pd.Series) -> str:
    tags = set()
    for value in values.dropna():
        text = str(value)
        if not text:
            continue
        for token in text.split(";"):
            if token:
                tags.add(int(token))
    if not tags:
        return ""
    return ";".join(str(x) for x in sorted(tags))


def make_item_id(prefix: str, item_num: Optional[int]) -> Optional[str]:
    if item_num is None or pd.isna(item_num):
        return None
    return f"{prefix}{int(item_num)}"


def contiguous_user_slices(user_ids: np.ndarray) -> Iterator[Tuple[int, int]]:
    if user_ids.size == 0:
        return
    start = 0
    for idx in range(1, user_ids.size):
        if user_ids[idx] != user_ids[idx - 1]:
            yield start, idx
            start = idx
    yield start, user_ids.size


def finite_nonnegative_ms(value: object) -> object:
    if value is None or pd.isna(value):
        return np.nan
    try:
        x = float(value)
    except Exception:
        return np.nan
    if not math.isfinite(x) or x < 0:
        return np.nan
    return int(x)


def compute_media_metrics(events: List[dict], video_length_ms: object = np.nan) -> dict:
    """Summarize observed play, pause, and cursor events."""
    if not events:
        return {
            "n_play_audio": 0,
            "n_pause_audio": 0,
            "n_play_video": 0,
            "n_pause_video": 0,
            "media_event_count": 0,
            "media_elapsed_ms": np.nan,
            "cursor_min_ms": np.nan,
            "cursor_max_ms": np.nan,
            "cursor_span_ms": np.nan,
            "media_completion_proxy": np.nan,
            "media_observed": 0,
            "media_pair_anomaly_count": 0,
        }

    sorted_events = sorted(events, key=lambda e: (int(e["ts"]), int(e.get("seq_idx", 0))))
    n_play_audio = sum(1 for e in sorted_events if e["action"] == "play_audio")
    n_pause_audio = sum(1 for e in sorted_events if e["action"] == "pause_audio")
    n_play_video = sum(1 for e in sorted_events if e["action"] == "play_video")
    n_pause_video = sum(1 for e in sorted_events if e["action"] == "pause_video")

    cursor_values = [finite_nonnegative_ms(e.get("cursor_time_ms")) for e in sorted_events]
    cursor_values = [int(x) for x in cursor_values if not pd.isna(x)]
    cursor_min = int(min(cursor_values)) if cursor_values else np.nan
    cursor_max = int(max(cursor_values)) if cursor_values else np.nan
    cursor_span = int(cursor_max - cursor_min) if cursor_values else np.nan

    elapsed_total = 0
    anomaly_count = 0
    open_start_by_kind: Dict[str, Optional[int]] = {"audio": None, "video": None}
    for event in sorted_events:
        action = event["action"]
        ts = int(event["ts"])
        if action == "play_audio":
            if open_start_by_kind["audio"] is not None:
                anomaly_count += 1
            open_start_by_kind["audio"] = ts
        elif action == "pause_audio":
            start = open_start_by_kind.get("audio")
            if start is not None and ts >= start:
                elapsed_total += ts - start
                open_start_by_kind["audio"] = None
            else:
                anomaly_count += 1
        elif action == "play_video":
            if open_start_by_kind["video"] is not None:
                anomaly_count += 1
            open_start_by_kind["video"] = ts
        elif action == "pause_video":
            start = open_start_by_kind.get("video")
            if start is not None and ts >= start:
                elapsed_total += ts - start
                open_start_by_kind["video"] = None
            else:
                anomaly_count += 1

    # Count unmatched play events without imputing duration.
    if open_start_by_kind["audio"] is not None:
        anomaly_count += 1
    if open_start_by_kind["video"] is not None:
        anomaly_count += 1

    video_len = finite_nonnegative_ms(video_length_ms)
    completion = np.nan
    if not pd.isna(video_len) and video_len > 0 and cursor_values:
        completion = min(float(cursor_max) / float(video_len), 3.0)

    return {
        "n_play_audio": int(n_play_audio),
        "n_pause_audio": int(n_pause_audio),
        "n_play_video": int(n_play_video),
        "n_pause_video": int(n_pause_video),
        "media_event_count": int(len(sorted_events)),
        "media_elapsed_ms": int(elapsed_total) if elapsed_total > 0 else np.nan,
        "cursor_min_ms": cursor_min,
        "cursor_max_ms": cursor_max,
        "cursor_span_ms": cursor_span,
        "media_completion_proxy": completion,
        "media_observed": 1,
        "media_pair_anomaly_count": int(anomaly_count),
    }


class ContentLookups:
    def __init__(
        self,
        questions: pd.DataFrame,
        lectures: pd.DataFrame,
        bundles: pd.DataFrame,
        explanations: pd.DataFrame,
        payments: pd.DataFrame,
        coupons: pd.DataFrame,
    ) -> None:
        q_index = questions.set_index("question_id", drop=False)
        b_index = bundles.set_index("bundle_id", drop=False)
        e_index = explanations.set_index("explanation_id", drop=False)
        l_index = lectures.set_index("lecture_id", drop=False)

        self.q_to_bundle = q_index["bundle_id"].to_dict()
        self.q_to_correct_code = q_index["correct_answer_code"].to_dict()
        self.q_to_part = q_index["part"].to_dict()
        self.bundle_to_expected_questions = b_index["n_questions"].to_dict()
        self.bundle_to_part = b_index["part"].to_dict()
        self.bundle_to_explanation = b_index["explanation_id"].to_dict()
        self.explanation_to_part = e_index["part"].to_dict()
        self.explanation_to_bundle = e_index["bundle_id"].to_dict()
        self.lecture_to_part = l_index["part"].to_dict()
        self.lecture_to_video_length = l_index["video_length"].to_dict()

        questions_numeric = questions.dropna(subset=["question_n"]).copy()
        bundles_numeric = bundles.dropna(subset=["bundle_n"]).copy()
        explanations_numeric = explanations.dropna(subset=["explanation_n"]).copy()
        lectures_numeric = lectures.dropna(subset=["lecture_n"]).copy()
        payments_numeric = payments.dropna(subset=["payment_n"]).copy()
        coupons_numeric = coupons.dropna(subset=["coupon_n"]).copy()

        self.qnum_to_bundle_num = questions_numeric.set_index("question_n")["bundle_n"].dropna().astype(int).to_dict()
        self.qnum_to_correct_code = questions_numeric.set_index("question_n")["correct_answer_code"].fillna(0).astype(int).to_dict()
        self.qnum_to_part = questions_numeric.set_index("question_n")["part"].to_dict()
        self.bundle_num_to_expected_questions = bundles_numeric.set_index("bundle_n")["n_questions"].dropna().astype(int).to_dict()
        self.bundle_num_to_part = bundles_numeric.set_index("bundle_n")["part"].to_dict()
        self.bundle_num_to_explanation_num = bundles_numeric.set_index("bundle_n")["explanation_n"].to_dict()
        self.explanation_num_to_part = explanations_numeric.set_index("explanation_n")["part"].to_dict()
        self.explanation_num_to_bundle_num = explanations_numeric.set_index("explanation_n")["bundle_n"].to_dict()
        self.lecture_num_to_part = lectures_numeric.set_index("lecture_n")["part"].to_dict()
        self.lecture_num_to_video_length = lectures_numeric.set_index("lecture_n")["video_length"].to_dict()
        self.payment_num_to_type = payments_numeric.set_index("payment_n")["type"].astype(str).to_dict()
        self.payment_num_to_duration = payments_numeric.set_index("payment_n")["duration"].to_dict()
        self.payment_num_to_bundle_allowance = payments_numeric.set_index("payment_n")["number_of_bundles"].to_dict()
        self.coupon_num_to_duration = coupons_numeric.set_index("coupon_n")["duration"].to_dict()


class AccessContextState:
    def __init__(self, lookups: ContentLookups) -> None:
        self.lookups = lookups
        self.observed = False
        self.pay_event_count = 0
        self.refund_event_count = 0
        self.coupon_event_count = 0
        self.payment_grants: List[dict] = []
        self.coupon_grants: List[dict] = []

    @staticmethod
    def _positive_int(value: object) -> Optional[int]:
        if value is None or pd.isna(value):
            return None
        try:
            x = int(value)
        except Exception:
            return None
        return x if x > 0 else None

    def apply_event(self, action: str, item_type_code: int, item_num: int, ts: int) -> None:
        self.observed = True
        if action == "pay":
            self.pay_event_count += 1
            if item_type_code != ITEM_TYPE_CODE["payment_item"]:
                return
            type = str(self.lookups.payment_num_to_type.get(item_num, "")).strip().lower()
            duration_ms = self._positive_int(self.lookups.payment_num_to_duration.get(item_num))
            bundle_allowance = self._positive_int(self.lookups.payment_num_to_bundle_allowance.get(item_num))
            if type == "pass" and duration_ms is not None:
                self.payment_grants.append(
                    {"item_num": item_num, "kind": "pass", "start_ts": ts, "end_ts": ts + duration_ms, "active": True}
                )
            elif type == "paygo" and bundle_allowance is not None:
                self.payment_grants.append(
                    {"item_num": item_num, "kind": "paygo", "start_ts": ts, "end_ts": None, "active": True}
                )
        elif action == "refund":
            self.refund_event_count += 1
            if item_type_code != ITEM_TYPE_CODE["payment_item"]:
                return
            for grant in reversed(self.payment_grants):
                if grant["active"] and int(grant["item_num"]) == int(item_num):
                    grant["active"] = False
                    break
        elif action == "enroll_coupon":
            self.coupon_event_count += 1
            if item_type_code != ITEM_TYPE_CODE["coupon"]:
                return
            duration_ms = self._positive_int(self.lookups.coupon_num_to_duration.get(item_num))
            if duration_ms is not None:
                self.coupon_grants.append(
                    {"item_num": item_num, "start_ts": ts, "end_ts": ts + duration_ms, "active": True}
                )

    def snapshot(self, ts: int) -> Dict[str, object]:
        if not self.observed:
            access_status = np.nan
            duration_window = np.nan
        else:
            active_payment_window = any(
                grant["active"]
                and grant["kind"] == "pass"
                and int(grant["start_ts"]) <= ts < int(grant["end_ts"])
                for grant in self.payment_grants
            )
            active_coupon_window = any(
                grant["active"]
                and int(grant["start_ts"]) <= ts < int(grant["end_ts"])
                for grant in self.coupon_grants
            )
            active_paygo = any(
                grant["active"] and grant["kind"] == "paygo" and int(grant["start_ts"]) <= ts
                for grant in self.payment_grants
            )
            duration_window = 1.0 if (active_payment_window or active_coupon_window) else 0.0
            access_status = 1.0 if (duration_window > 0 or active_paygo) else 0.0
        return {
            "access_status_aux": access_status,
            "payment_coupon_window_aux": duration_window,
            "access_aux_observed": 1 if self.observed else 0,
            "access_pay_event_count": int(self.pay_event_count),
            "access_refund_event_count": int(self.refund_event_count),
            "access_coupon_event_count": int(self.coupon_event_count),
        }


class SessionState:
    def __init__(
        self,
        enter_ts: Optional[int],
        source_code: int,
        platform_code: int,
        enter_seq_idx: Optional[int],
        implicit_enter: int = 0,
        access_snapshot: Optional[Dict[str, object]] = None,
    ) -> None:
        self.enter_ts = enter_ts
        self.source_code = source_code
        self.platform_code = platform_code
        self.enter_seq_idx = enter_seq_idx
        self.implicit_enter = implicit_enter
        self.access_snapshot = access_snapshot or {
            "access_status_aux": np.nan,
            "payment_coupon_window_aux": np.nan,
            "access_aux_observed": 0,
            "access_pay_event_count": 0,
            "access_refund_event_count": 0,
            "access_coupon_event_count": 0,
        }
        # Per-question process events for the open bundle.
        self.responses: Dict[int, Dict[str, list]] = {}


class StudyState:
    def __init__(
        self,
        item_type_code: int,
        item_num: int,
        enter_ts: Optional[int],
        source_code: int,
        platform_code: int,
        enter_seq_idx: Optional[int],
        implicit_enter: int = 0,
        truncated_reenter: int = 0,
    ) -> None:
        self.item_type_code = item_type_code
        self.item_num = item_num
        self.enter_ts = enter_ts
        self.source_code = source_code
        self.platform_code = platform_code
        self.enter_seq_idx = enter_seq_idx
        self.implicit_enter = implicit_enter
        self.truncated_reenter = truncated_reenter
        self.media_events: List[dict] = []


class UserStats:
    def __init__(self) -> None:
        self.total_events = 0
        self.total_core_events = 0
        self.total_choice_process_events = 0
        self.total_media_process_events = 0
        self.total_text_process_events = 0
        self.total_bundles_submitted = 0
        self.total_bundles_unsubmitted = 0
        self.total_questions_answered = 0
        self.total_correct = 0
        self.total_incorrect = 0
        self.total_explanation_episodes = 0
        self.total_lecture_episodes = 0
        self.total_explanation_dwell_ms = 0
        self.total_lecture_dwell_ms = 0
        self.total_media_elapsed_ms = 0
        self.bundle_durations: List[int] = []
        self.bundle_accuracies: List[float] = []
        self.unique_sources = set()
        self.unique_platforms = set()


class CanonicalBuffers:
    def __init__(self) -> None:
        self.events = ColumnarTableBuffer(
            [
                "user_id", "seq_idx", "timestamp", "t_rel_ms", "delta_ms",
                "action_code", "action_group_code", "item_type_code", "item_num",
                "source_code", "platform_code", "answer_code", "cursor_time_ms",
            ]
        )
        self.events_light = ColumnarTableBuffer(
            ["user_id", "seq_idx", "timestamp", "t_rel_ms", "delta_ms", "action_code", "item_type_code", "item_num"]
        )
        self.skipped_users = ColumnarTableBuffer(["user_id", "file_name", "reason", "raw_rows", "kept_rows"])

    def should_flush(self) -> Tuple[bool, str]:
        if len(self.events) >= FLUSH_EVENTS_ROWS:
            return True, f"events_rows>={FLUSH_EVENTS_ROWS}"
        if len(self.events_light) >= FLUSH_EVENTS_LIGHT_ROWS:
            return True, f"events_light_rows>={FLUSH_EVENTS_LIGHT_ROWS}"
        if len(self.skipped_users) >= FLUSH_SKIPPED_USERS_ROWS:
            return True, f"skipped_users_rows>={FLUSH_SKIPPED_USERS_ROWS}"
        return False, ""

    def clear(self) -> None:
        self.events.clear()
        self.events_light.clear()
        self.skipped_users.clear()


class DerivedBuffers:
    def __init__(self) -> None:
        self.question_attempts = ColumnarTableBuffer(
            [
                "user_id", "bundle_attempt_index", "question_attempt_index", "question_attempt_index_global",
                "question_position_within_bundle_attempt", "bundle_id", "question_id", "part", "source_code", "platform_code",
                "bundle_enter_ts", "first_response_ts", "final_response_ts", "submit_ts", "bundle_duration_ms",
                "latency_enter_to_first_ms", "latency_enter_to_final_ms", "latency_final_to_submit_ms",
                "response_count", "response_change_count", "final_answer_code", "correct_answer_code", "is_correct",
                "implicit_bundle_enter",
                "erase_count", "undo_erase_count", "choice_process_count", "text_enter_count",
                "first_choice_process_ts", "last_choice_process_ts", "last_change_latency_ms",
                "question_audio_play_count", "question_audio_pause_count", "question_video_play_count", "question_video_pause_count",
                "question_media_event_count", "question_media_elapsed_ms", "question_cursor_max_ms", "question_media_observed",
            ]
        )
        self.bundle_attempts = ColumnarTableBuffer(
            [
                "user_id", "bundle_attempt_index", "bundle_id", "explanation_id", "part", "source_code", "platform_code",
                "enter_ts", "submit_ts", "duration_ms", "expected_question_count", "answered_question_count",
                "missing_question_count", "total_response_count", "total_response_change_count", "n_correct", "accuracy",
                "implicit_enter", "missing_submit",
                "total_erase_count", "total_undo_erase_count", "total_choice_process_count", "total_text_enter_count",
                "choice_process_mass_raw", "choice_process_observed", "first_choice_process_ts", "last_choice_process_ts",
                "total_question_media_event_count", "question_media_elapsed_ms", "question_cursor_max_ms", "question_media_observed",
                "access_status_aux", "payment_coupon_window_aux", "access_aux_observed",
                "access_pay_event_count", "access_refund_event_count", "access_coupon_event_count",
            ]
        )
        self.study_episodes = ColumnarTableBuffer(
            [
                "user_id", "study_episode_index", "item_type_code", "item_id", "part", "source_code", "platform_code",
                "enter_ts", "quit_ts", "dwell_ms", "dwell_ratio", "video_length_ms", "related_bundle_id",
                "implicit_enter", "matched_quit", "truncated_reenter",
                "n_play_audio", "n_pause_audio", "n_play_video", "n_pause_video", "media_event_count",
                "media_elapsed_ms", "cursor_min_ms", "cursor_max_ms", "cursor_span_ms",
                "media_completion_proxy", "media_observed", "media_pair_anomaly_count",
            ]
        )
        self.choice_process_features = ColumnarTableBuffer(
            [
                "user_id", "bundle_attempt_index", "bundle_id", "enter_ts", "submit_ts",
                "total_response_count", "total_response_change_count", "total_erase_count", "total_undo_erase_count",
                "total_text_enter_count", "choice_process_count", "choice_process_mass_raw",
                "answered_question_count", "choice_process_observed", "source_code", "platform_code",
            ]
        )
        self.media_process_features = ColumnarTableBuffer(
            [
                "user_id", "process_episode_index", "process_scope", "item_type_code", "item_id", "related_bundle_id",
                "part", "source_code", "platform_code", "enter_ts", "exit_ts", "dwell_ms", "video_length_ms",
                "n_play_audio", "n_pause_audio", "n_play_video", "n_pause_video", "media_event_count",
                "media_elapsed_ms", "cursor_min_ms", "cursor_max_ms", "cursor_span_ms", "media_completion_proxy",
                "media_observed", "media_pair_anomaly_count",
            ]
        )
        self.user_summary = ColumnarTableBuffer(
            [
                "user_id", "first_ts", "last_ts", "active_span_ms", "total_events", "total_core_events",
                "total_choice_process_events", "total_media_process_events", "total_text_process_events",
                "total_bundles_submitted", "total_bundles_unsubmitted", "total_questions_answered", "total_correct",
                "total_incorrect", "accuracy", "total_explanation_episodes", "total_lecture_episodes",
                "total_explanation_dwell_ms", "total_lecture_dwell_ms", "total_media_elapsed_ms",
                "n_unique_sources", "n_unique_platforms", "mean_delta_ms", "std_delta_ms", "p95_delta_ms",
                "mean_bundle_duration_ms", "mean_bundle_accuracy",
            ]
        )

    def should_flush(self) -> Tuple[bool, str]:
        if len(self.question_attempts) >= FLUSH_QUESTION_ROWS:
            return True, f"question_rows>={FLUSH_QUESTION_ROWS}"
        if len(self.bundle_attempts) >= FLUSH_BUNDLE_ROWS:
            return True, f"bundle_rows>={FLUSH_BUNDLE_ROWS}"
        if len(self.study_episodes) >= FLUSH_STUDY_ROWS:
            return True, f"study_rows>={FLUSH_STUDY_ROWS}"
        if len(self.choice_process_features) >= FLUSH_CHOICE_PROCESS_ROWS:
            return True, f"choice_process_rows>={FLUSH_CHOICE_PROCESS_ROWS}"
        if len(self.media_process_features) >= FLUSH_MEDIA_PROCESS_ROWS:
            return True, f"media_process_rows>={FLUSH_MEDIA_PROCESS_ROWS}"
        if len(self.user_summary) >= FLUSH_USER_SUMMARY_ROWS:
            return True, f"user_summary_rows>={FLUSH_USER_SUMMARY_ROWS}"
        return False, ""

    def clear(self) -> None:
        self.question_attempts.clear()
        self.bundle_attempts.clear()
        self.study_episodes.clear()
        self.choice_process_features.clear()
        self.media_process_features.clear()
        self.user_summary.clear()


class PreprocessorKT4:
    def __init__(self, kt4_input: Path, contents_input: Path, output_root: Path) -> None:
        self.kt4_input = kt4_input
        self.contents_input = contents_input
        self.output_root = output_root

        self.content_dir = self.output_root / "contents"
        self.kt4_dir = self.output_root / "kt4"
        self.events_dir = self.kt4_dir / "events"
        self.events_light_dir = self.kt4_dir / "events_light"
        self.question_dir = self.kt4_dir / "question_attempts"
        self.bundle_dir = self.kt4_dir / "bundle_attempts"
        self.study_dir = self.kt4_dir / "study_episodes"
        self.choice_process_dir = self.kt4_dir / "choice_process_features"
        self.media_process_dir = self.kt4_dir / "media_process_features"
        self.user_summary_dir = self.kt4_dir / "user_summary"
        self.audit_dir = self.kt4_dir / "audit"
        self.metadata_dir = self.output_root / "metadata"

        self.save_manager = SaveManager(output_root=self.output_root, metadata_dir=self.metadata_dir)
        self.lookups: Optional[ContentLookups] = None
        self.canonical_buffers = CanonicalBuffers()
        self.derived_buffers = DerivedBuffers()
        self.raw_action_counter: Counter = Counter()
        self.kept_action_counter: Counter = Counter()
        self.filtered_action_counter: Counter = Counter()
        self.raw_item_prefix_counter: Counter = Counter()

        self.run_stats = {
            "discovered_user_files": 0,
            "processed_valid_users": 0,
            "skipped_users": 0,
            "canonical_event_shards": 0,
            "canonical_event_light_shards": 0,
            "derived_event_shards_read": 0,
        }
        self.output_counts = defaultdict(int)

    def run(self) -> None:
        self._prepare_output_dirs()
        self._process_contents()
        self._build_canonical_events()
        self._derive_from_canonical_events()
        self._write_manifest()

    def _prepare_output_dirs(self) -> None:
        if CLEAN_OUTPUT_DIRS:
            for directory in [self.content_dir, self.kt4_dir, self.metadata_dir]:
                if directory.exists():
                    shutil.rmtree(directory)
        for directory in [
            self.output_root, self.content_dir, self.kt4_dir, self.events_dir, self.events_light_dir,
            self.question_dir, self.bundle_dir, self.study_dir, self.choice_process_dir,
            self.media_process_dir, self.user_summary_dir, self.audit_dir, self.metadata_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def _record_output(self, label: str) -> None:
        self.output_counts[label] += 1

    def _process_contents(self) -> None:
        progress = tqdm(total=4, desc="Processing contents", unit="file")
        questions_path = self.contents_input / "questions.csv"
        lectures_path = self.contents_input / "lectures.csv"
        payments_path = self.contents_input / "payments.csv"
        coupons_path = self.contents_input / "coupons.csv"
        if not questions_path.exists():
            raise FileNotFoundError(f"Missing file: {questions_path}")
        if not lectures_path.exists():
            raise FileNotFoundError(f"Missing file: {lectures_path}")
        if not payments_path.exists():
            raise FileNotFoundError(f"Missing file: {payments_path}")
        if not coupons_path.exists():
            raise FileNotFoundError(f"Missing file: {coupons_path}")

        questions = pd.read_csv(questions_path)
        questions.columns = [c.strip().lower() for c in questions.columns]
        questions = questions[["question_id", "bundle_id", "explanation_id", "correct_answer", "part", "tags", "deployed_at"]].copy()
        questions["question_id"] = normalize_text_series(questions["question_id"])
        questions["bundle_id"] = normalize_text_series(questions["bundle_id"])
        questions["explanation_id"] = normalize_text_series(questions["explanation_id"])
        questions["correct_answer"] = normalize_text_series(questions["correct_answer"])
        questions["question_n"] = extract_numeric_suffix_series(questions["question_id"])
        questions["bundle_n"] = extract_numeric_suffix_series(questions["bundle_id"])
        questions["explanation_n"] = extract_numeric_suffix_series(questions["explanation_id"])
        questions["correct_answer_code"] = questions["correct_answer"].map(answer_to_code).astype("Int8")
        questions["part"] = pd.to_numeric(questions["part"], errors="coerce").astype("Int8")
        questions["deployed_at"] = pd.to_numeric(questions["deployed_at"], errors="coerce").astype("Int64")
        questions["tags_raw"] = questions["tags"].map(parse_tags_string)
        questions["n_tags"] = questions["tags_raw"].map(lambda x: 0 if x == "" else len(x.split(";"))).astype("Int16")
        questions = questions.drop(columns=["tags"])
        questions = questions.drop_duplicates(subset=["question_id"], keep="last").reset_index(drop=True)
        progress.update(1)

        bundles = (
            questions.groupby("bundle_id", as_index=False)
            .agg(
                explanation_id=("explanation_id", "first"),
                bundle_n=("bundle_n", "first"),
                n_questions=("question_id", "nunique"),
                part=("part", mode_or_first),
                tags_raw=("tags_raw", union_tags),
                deployed_at_min=("deployed_at", "min"),
                deployed_at_max=("deployed_at", "max"),
            )
            .reset_index(drop=True)
        )
        bundles["explanation_n"] = extract_numeric_suffix_series(bundles["explanation_id"])
        bundles["n_tags"] = bundles["tags_raw"].map(lambda x: 0 if x == "" else len(x.split(";"))).astype("Int16")
        bundles["part"] = pd.to_numeric(bundles["part"], errors="coerce").astype("Int8")
        bundles["n_questions"] = pd.to_numeric(bundles["n_questions"], errors="coerce").astype("Int16")
        bundles["deployed_at_min"] = pd.to_numeric(bundles["deployed_at_min"], errors="coerce").astype("Int64")
        bundles["deployed_at_max"] = pd.to_numeric(bundles["deployed_at_max"], errors="coerce").astype("Int64")

        explanations = bundles[["bundle_id", "bundle_n", "explanation_id", "explanation_n", "part", "tags_raw", "n_questions"]].copy()
        explanations["n_tags"] = explanations["tags_raw"].map(lambda x: 0 if x == "" else len(x.split(";"))).astype("Int16")

        lectures = pd.read_csv(lectures_path)
        lectures.columns = [c.strip().lower() for c in lectures.columns]
        lectures = lectures[["lecture_id", "part", "tags", "video_length", "deployed_at"]].copy()
        lectures["lecture_id"] = normalize_text_series(lectures["lecture_id"])
        lectures["lecture_n"] = extract_numeric_suffix_series(lectures["lecture_id"])
        lectures["part"] = pd.to_numeric(lectures["part"], errors="coerce")
        lectures.loc[lectures["part"] < 0, "part"] = np.nan
        lectures["part"] = lectures["part"].astype("Int8")
        lectures["video_length"] = pd.to_numeric(lectures["video_length"], errors="coerce")
        lectures.loc[lectures["video_length"] < 0, "video_length"] = np.nan
        lectures["video_length"] = lectures["video_length"].astype("Int64")
        lectures["deployed_at"] = pd.to_numeric(lectures["deployed_at"], errors="coerce")
        lectures.loc[lectures["deployed_at"] < 0, "deployed_at"] = np.nan
        lectures["deployed_at"] = lectures["deployed_at"].astype("Int64")
        lectures["tags_raw"] = lectures["tags"].map(parse_tags_string)
        lectures["n_tags"] = lectures["tags_raw"].map(lambda x: 0 if x == "" else len(x.split(";"))).astype("Int16")
        lectures = lectures.drop(columns=["tags"])
        lectures = lectures.drop_duplicates(subset=["lecture_id"], keep="last").reset_index(drop=True)
        progress.update(1)

        payments = pd.read_csv(payments_path)
        payments.columns = [c.strip().lower() for c in payments.columns]
        if "number_of_bundles" in payments.columns:
            allowance_column = "number_of_bundles"
        elif "number_of_questions" in payments.columns:
            allowance_column = "number_of_questions"
        else:
            raise ValueError(
                f"{payments_path} must contain number_of_bundles or number_of_questions; "
                f"found columns={payments.columns.tolist()}"
            )
        payments = payments[["payment_item_id", "type", "duration", allowance_column]].copy()
        payments = payments.rename(columns={allowance_column: "number_of_bundles"})
        payments["payment_item_id"] = normalize_text_series(payments["payment_item_id"])
        payments["payment_n"] = extract_numeric_suffix_series(payments["payment_item_id"])
        payments["type"] = normalize_text_series(payments["type"])
        payments["duration"] = pd.to_numeric(payments["duration"], errors="coerce")
        payments.loc[payments["duration"] < 0, "duration"] = np.nan
        payments["duration"] = payments["duration"].astype("Int64")
        payments["number_of_bundles"] = pd.to_numeric(payments["number_of_bundles"], errors="coerce")
        payments.loc[payments["number_of_bundles"] < 0, "number_of_bundles"] = np.nan
        payments["number_of_bundles"] = payments["number_of_bundles"].astype("Int64")
        payments = payments.drop_duplicates(subset=["payment_item_id"], keep="last").reset_index(drop=True)
        progress.update(1)

        coupons = pd.read_csv(coupons_path)
        coupons.columns = [c.strip().lower() for c in coupons.columns]
        coupons = coupons[["coupon_id", "coupon_type", "duration"]].copy()
        coupons["coupon_id"] = normalize_text_series(coupons["coupon_id"])
        coupons["coupon_n"] = extract_numeric_suffix_series(coupons["coupon_id"])
        coupons["coupon_type"] = normalize_text_series(coupons["coupon_type"])
        coupons["duration"] = pd.to_numeric(coupons["duration"], errors="coerce")
        coupons.loc[coupons["duration"] < 0, "duration"] = np.nan
        coupons["duration"] = coupons["duration"].astype("Int64")
        coupons = coupons.drop_duplicates(subset=["coupon_id"], keep="last").reset_index(drop=True)
        progress.update(1)
        progress.close()

        question_tag_edges = explode_tag_edges(questions, "question_id", "tags_raw")
        lecture_tag_edges = explode_tag_edges(lectures, "lecture_id", "tags_raw")
        bundle_question_edges = questions[["bundle_id", "question_id"]].drop_duplicates().reset_index(drop=True)
        explanation_bundle_edges = bundles[["explanation_id", "bundle_id"]].drop_duplicates().reset_index(drop=True)
        question_part_edges = questions[["question_id", "part"]].drop_duplicates().reset_index(drop=True)

        self.save_manager.write_single(questions, self.content_dir / "questions_clean.parquet", label="questions_clean", stage="contents")
        self._record_output("questions_clean")
        self.save_manager.write_single(question_tag_edges, self.content_dir / "question_tag_edges.parquet", label="question_tag_edges", stage="contents")
        self._record_output("question_tag_edges")
        self.save_manager.write_single(bundles, self.content_dir / "bundles_clean.parquet", label="bundles_clean", stage="contents")
        self._record_output("bundles_clean")
        self.save_manager.write_single(explanations, self.content_dir / "explanations_clean.parquet", label="explanations_clean", stage="contents")
        self._record_output("explanations_clean")
        self.save_manager.write_single(lectures, self.content_dir / "lectures_clean.parquet", label="lectures_clean", stage="contents")
        self._record_output("lectures_clean")
        self.save_manager.write_single(payments, self.content_dir / "payments_clean.parquet", label="payments_clean", stage="contents")
        self._record_output("payments_clean")
        self.save_manager.write_single(coupons, self.content_dir / "coupons_clean.parquet", label="coupons_clean", stage="contents")
        self._record_output("coupons_clean")
        self.save_manager.write_single(lecture_tag_edges, self.content_dir / "lecture_tag_edges.parquet", label="lecture_tag_edges", stage="contents")
        self._record_output("lecture_tag_edges")
        self.save_manager.write_single(bundle_question_edges, self.content_dir / "bundle_question_edges.parquet", label="bundle_question_edges", stage="contents")
        self._record_output("bundle_question_edges")
        self.save_manager.write_single(explanation_bundle_edges, self.content_dir / "explanation_bundle_edges.parquet", label="explanation_bundle_edges", stage="contents")
        self._record_output("explanation_bundle_edges")
        self.save_manager.write_single(question_part_edges, self.content_dir / "question_part_edges.parquet", label="question_part_edges", stage="contents")
        self._record_output("question_part_edges")

        source_map_df = pd.DataFrame([{"source": source, "source_code": code} for source, code in SOURCE_CODE.items()]).sort_values("source_code")
        platform_map_df = pd.DataFrame([{"platform": platform, "platform_code": code} for platform, code in PLATFORM_CODE.items()]).sort_values("platform_code")
        action_map_df = pd.DataFrame([{"action_type": action, "action_code": code, "action_group": action_group(action), "action_group_code": ACTION_GROUP_CODE[action_group(action)]} for action, code in ACTION_CODE.items()]).sort_values("action_code")
        item_map_df = pd.DataFrame([{"item_type": item_type, "item_type_code": code} for item_type, code in ITEM_TYPE_CODE.items()]).sort_values("item_type_code")
        answer_map_df = pd.DataFrame([{"answer": answer, "answer_code": code} for answer, code in ANSWER_CODE.items()]).sort_values("answer_code")
        action_group_map_df = pd.DataFrame([{"action_group": group, "action_group_code": code} for group, code in ACTION_GROUP_CODE.items()]).sort_values("action_group_code")

        self.save_manager.write_single(source_map_df, self.metadata_dir / "source_codes.parquet", label="source_codes", stage="metadata")
        self._record_output("source_codes")
        self.save_manager.write_single(platform_map_df, self.metadata_dir / "platform_codes.parquet", label="platform_codes", stage="metadata")
        self._record_output("platform_codes")
        self.save_manager.write_single(action_map_df, self.metadata_dir / "action_codes.parquet", label="action_codes", stage="metadata")
        self._record_output("action_codes")
        self.save_manager.write_single(action_group_map_df, self.metadata_dir / "action_group_codes.parquet", label="action_group_codes", stage="metadata")
        self._record_output("action_group_codes")
        self.save_manager.write_single(item_map_df, self.metadata_dir / "item_type_codes.parquet", label="item_type_codes", stage="metadata")
        self._record_output("item_type_codes")
        self.save_manager.write_single(answer_map_df, self.metadata_dir / "answer_codes.parquet", label="answer_codes", stage="metadata")
        self._record_output("answer_codes")

        self.lookups = ContentLookups(
            questions=questions,
            lectures=lectures,
            bundles=bundles,
            explanations=explanations,
            payments=payments,
            coupons=coupons,
        )

    def _load_user_events(self, path: Path) -> Tuple[Optional[pd.DataFrame], str, int, int]:
        raw_rows = 0
        try:
            df = pd.read_csv(path, low_memory=False, dtype=USER_READ_DTYPES)
        except Exception as exc:
            return None, f"read_error:{type(exc).__name__}", raw_rows, 0

        df.columns = [c.strip().lower() for c in df.columns]
        missing = [c for c in BASE_USER_COLUMNS if c not in df.columns]
        if missing:
            return None, f"missing_required_columns:{','.join(missing)}", raw_rows, 0
        if "cursor_time" not in df.columns:
            df["cursor_time"] = pd.NA
        df = df[BASE_USER_COLUMNS + OPTIONAL_USER_COLUMNS].copy()
        raw_rows = len(df)
        if df.empty:
            return None, "empty_raw_file", raw_rows, 0

        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["action_type_raw"] = normalize_text_series(df["action_type"])
        df["action_type"] = df["action_type_raw"].map(normalize_action_value)
        df["item_id"] = normalize_text_series(df["item_id"])
        df["source"] = df["source"].map(normalize_source_value)
        df["user_answer"] = normalize_text_series(df["user_answer"])
        df["platform"] = normalize_text_series(df["platform"], fill_value="unknown")
        df.loc[~df["platform"].isin({"mobile", "web"}), "platform"] = "unknown"
        df["cursor_time_ms"] = pd.to_numeric(df["cursor_time"], errors="coerce")
        df.loc[df["cursor_time_ms"] < 0, "cursor_time_ms"] = np.nan

        self.raw_action_counter.update(df["action_type"].fillna("").astype(str).tolist())

        df = df.dropna(subset=["timestamp"])
        if df.empty:
            return None, "empty_after_timestamp_filter", raw_rows, 0

        before_action_filter = len(df)
        df = df[df["action_type"].isin(KNOWN_ACTIONS)].copy()
        if len(df) < before_action_filter:
            filtered = before_action_filter - len(df)
            self.filtered_action_counter.update({"unknown_or_out_of_scope_action": filtered})
        if df.empty:
            return None, "empty_after_action_filter", raw_rows, 0

        df = df[df["item_id"].str.len() > 0].copy()
        if df.empty:
            return None, "empty_after_item_id_filter", raw_rows, 0

        df["item_prefix"] = df["item_id"].str[0]
        self.raw_item_prefix_counter.update(df["item_prefix"].fillna("").astype(str).tolist())
        before_prefix_filter = len(df)
        df = df[df["item_prefix"].isin(KNOWN_ITEM_PREFIX)].copy()
        if len(df) < before_prefix_filter:
            self.filtered_action_counter.update({"non_learning_item_prefix": before_prefix_filter - len(df)})
        if df.empty:
            return None, "empty_after_item_prefix_filter", raw_rows, 0

        df["item_type"] = df["item_prefix"].map(item_prefix_to_type)
        df["item_num"] = extract_numeric_suffix_series(df["item_id"])
        df = df.dropna(subset=["item_num"]).copy()
        if df.empty:
            return None, "empty_after_item_num_filter", raw_rows, 0

        df["action_sort_key"] = df["action_type"].map(ACTION_SORT_PRIORITY).fillna(99).astype("int8")
        df["item_num"] = df["item_num"].astype("int32")
        df["timestamp"] = df["timestamp"].astype("int64")
        df["cursor_time_ms"] = df["cursor_time_ms"].astype("float64")
        df = df.drop_duplicates(subset=["timestamp", "action_type", "item_id", "source", "user_answer", "platform", "cursor_time_ms"], keep="first")
        df = df.sort_values(["timestamp", "action_sort_key", "item_id"], kind="mergesort").reset_index(drop=True)
        df = df.drop(columns=["action_sort_key", "cursor_time"])
        self.kept_action_counter.update(df["action_type"].fillna("").astype(str).tolist())
        return df, "ok", raw_rows, len(df)

    def _build_canonical_events(self) -> None:
        if self.lookups is None:
            raise RuntimeError("Contents must be processed before canonical events.")

        def file_key(path: Path) -> int:
            name = path.stem
            return int(name[1:]) if len(name) > 1 and name[1:].isdigit() else math.inf

        user_files = sorted(self.kt4_input.glob("u*.csv"), key=file_key)
        if not user_files:
            raise FileNotFoundError(f"No user csv files found in {self.kt4_input}")
        if len(user_files) != EXPECTED_KT4_USERS:
            raise ValueError(
                f"Expected exactly {EXPECTED_KT4_USERS} EdNet-KT4 user files; "
                f"found {len(user_files)} in {self.kt4_input}."
            )

        self.run_stats["discovered_user_files"] = len(user_files)
        progress_bar = tqdm(user_files, desc="Building canonical events", unit="user")
        visualizer = ProgressVisualizer(len(user_files), "students", self.metadata_dir, "canonical", REPORT_EVERY_USERS)
        visualizer.write_snapshot()

        for path in progress_bar:
            user_id = parse_numeric_id(path.stem)
            df, reason, raw_rows, kept_rows = self._load_user_events(path)
            was_valid = df is not None and not df.empty and user_id is not None
            if not was_valid:
                self.canonical_buffers.skipped_users.append_row({"user_id": user_id, "file_name": path.name, "reason": reason, "raw_rows": raw_rows, "kept_rows": kept_rows})
                self.run_stats["skipped_users"] += 1
            else:
                canonical_full_df, canonical_light_df = self._build_canonical_frames_for_user(user_id=user_id, df=df)
                self.canonical_buffers.events.extend_from_dataframe(canonical_full_df)
                self.canonical_buffers.events_light.extend_from_dataframe(canonical_light_df)
                self.run_stats["processed_valid_users"] += 1

            processed_total = self.run_stats["processed_valid_users"] + self.run_stats["skipped_users"]
            progress_bar.set_postfix_str(f"processed={processed_total} remaining={len(user_files) - processed_total}")
            visualizer.update(was_valid=was_valid)
            should_flush, reason_flush = self.canonical_buffers.should_flush()
            if should_flush:
                self._flush_canonical_buffers(reason=reason_flush, visualizer=visualizer)

        progress_bar.close()
        self._flush_canonical_buffers(reason="final_flush", visualizer=visualizer)
        visualizer.finalize()

    def _build_canonical_frames_for_user(self, user_id: int, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        first_ts = int(df["timestamp"].iloc[0])
        timestamps = df["timestamp"].to_numpy(dtype=np.int64)
        rel_array = timestamps - first_ts
        delta_array = np.empty(len(df), dtype=np.int64)
        delta_array[0] = 0
        if len(df) > 1:
            delta_array[1:] = timestamps[1:] - timestamps[:-1]

        action_codes = df["action_type"].map(ACTION_CODE).astype("int8")
        action_group_codes = df["action_type"].map(lambda x: ACTION_GROUP_CODE[action_group(str(x))]).astype("int8")
        item_type_codes = df["item_type"].map(ITEM_TYPE_CODE).astype("int8")
        source_codes = df["source"].map(lambda x: SOURCE_CODE.get(str(x), 0)).astype("int8")
        platform_codes = df["platform"].map(lambda x: PLATFORM_CODE.get(str(x), 0)).astype("int8")
        answer_codes = df["user_answer"].map(lambda x: ANSWER_CODE.get(str(x), 0)).astype("int8")
        cursor_time = pd.to_numeric(df["cursor_time_ms"], errors="coerce").to_numpy(dtype=np.float64)

        canonical_full_df = pd.DataFrame(
            {
                "user_id": np.full(len(df), user_id, dtype=np.int64),
                "seq_idx": np.arange(len(df), dtype=np.int64),
                "timestamp": timestamps,
                "t_rel_ms": rel_array,
                "delta_ms": delta_array,
                "action_code": action_codes.to_numpy(dtype=np.int8),
                "action_group_code": action_group_codes.to_numpy(dtype=np.int8),
                "item_type_code": item_type_codes.to_numpy(dtype=np.int8),
                "item_num": df["item_num"].to_numpy(dtype=np.int32),
                "source_code": source_codes.to_numpy(dtype=np.int8),
                "platform_code": platform_codes.to_numpy(dtype=np.int8),
                "answer_code": answer_codes.to_numpy(dtype=np.int8),
                "cursor_time_ms": cursor_time,
            }
        )
        canonical_light_df = canonical_full_df[["user_id", "seq_idx", "timestamp", "t_rel_ms", "delta_ms", "action_code", "item_type_code", "item_num"]].copy()
        return canonical_full_df, canonical_light_df

    def _flush_canonical_buffers(self, reason: str, visualizer: ProgressVisualizer) -> None:
        if not any([len(self.canonical_buffers.events), len(self.canonical_buffers.events_light), len(self.canonical_buffers.skipped_users)]):
            return
        events_df = self.canonical_buffers.events.to_dataframe()
        events_light_df = self.canonical_buffers.events_light.to_dataframe()
        skipped_df = self.canonical_buffers.skipped_users.to_dataframe()
        path = self.save_manager.write(events_df, self.events_dir, "events_chunk", label="events", stage="canonical")
        if path is not None:
            self.run_stats["canonical_event_shards"] += 1
            self._record_output("events")
        path = self.save_manager.write(events_light_df, self.events_light_dir, "events_light_chunk", label="events_light", stage="canonical")
        if path is not None:
            self.run_stats["canonical_event_light_shards"] += 1
            self._record_output("events_light")
        path = self.save_manager.write(skipped_df, self.audit_dir, "skipped_users_chunk", label="skipped_users", stage="audit")
        if path is not None:
            self._record_output("skipped_users")
        self.canonical_buffers.clear()
        visualizer.record_flush(reason=reason)
        gc.collect()

    def _derive_from_canonical_events(self) -> None:
        if self.lookups is None:
            raise RuntimeError("Contents must be processed before deriving outputs.")
        event_shards = sorted(self.events_dir.glob("events_chunk_*.parquet"))
        if not event_shards:
            raise RuntimeError("No canonical event shards found. Canonical stage must complete first.")

        progress_bar = tqdm(event_shards, desc="Deriving attempts and process features", unit="shard")
        visualizer = ProgressVisualizer(len(event_shards), "shards", self.metadata_dir, "derived", REPORT_EVERY_SHARDS)
        visualizer.write_snapshot()

        for shard_path in progress_bar:
            shard_df = pd.read_parquet(shard_path)
            if shard_df.empty:
                self.run_stats["derived_event_shards_read"] += 1
                visualizer.update(was_valid=False)
                continue
            shard_df = shard_df.sort_values(["user_id", "seq_idx"], kind="mergesort").reset_index(drop=True)
            user_ids = shard_df["user_id"].to_numpy()
            for start, end in contiguous_user_slices(user_ids):
                user_df = shard_df.iloc[start:end].reset_index(drop=True)
                self._derive_single_user_from_events(user_df)
                should_flush, reason = self.derived_buffers.should_flush()
                if should_flush:
                    self._flush_derived_buffers(reason=reason, visualizer=visualizer)
            self.run_stats["derived_event_shards_read"] += 1
            visualizer.update(was_valid=True)
            progress_bar.set_postfix_str(f"processed={self.run_stats['derived_event_shards_read']} remaining={len(event_shards) - self.run_stats['derived_event_shards_read']}")
            gc.collect()

        progress_bar.close()
        self._flush_derived_buffers(reason="final_flush", visualizer=visualizer)
        visualizer.finalize()

    @staticmethod
    def _question_state(session: SessionState, question_num: int) -> dict:
        return session.responses.setdefault(
            question_num,
            {
                "answers": [], "response_times": [],
                "erase_answers": [], "erase_times": [],
                "undo_answers": [], "undo_times": [],
                "text_enter_times": [],
                "media_events": [],
            },
        )

    def _derive_single_user_from_events(self, df: pd.DataFrame) -> None:
        lookups = self.lookups
        assert lookups is not None
        user_id = int(df["user_id"].iloc[0])
        open_bundles: Dict[int, SessionState] = {}
        open_studies: Dict[Tuple[int, int], StudyState] = {}
        stats = UserStats()
        question_attempt_counter = defaultdict(int)
        bundle_attempt_index = 0
        study_episode_index = 0
        process_episode_index = 0
        access_state = AccessContextState(lookups)

        for row in df.itertuples(index=False):
            ts = int(row.timestamp)
            action_code = int(row.action_code)
            action = ACTION_NAME_FROM_CODE.get(action_code, "unknown")
            item_type_code = int(row.item_type_code)
            item_num = int(row.item_num)
            source_code = int(row.source_code)
            platform_code = int(row.platform_code)
            answer_code = int(row.answer_code)
            seq_idx = int(row.seq_idx)
            cursor_time_ms = getattr(row, "cursor_time_ms", np.nan)

            stats.total_events += 1
            group = action_group(action)
            if group == "core":
                stats.total_core_events += 1
            elif group == "choice_process":
                stats.total_choice_process_events += 1
            elif group == "media_process":
                stats.total_media_process_events += 1
            elif group == "text_process":
                stats.total_text_process_events += 1
            stats.unique_sources.add(source_code)
            stats.unique_platforms.add(platform_code)

            if action in ACCESS_CONTEXT_ACTIONS:
                access_state.apply_event(action=action, item_type_code=item_type_code, item_num=item_num, ts=ts)
                continue

            if item_type_code == ITEM_TYPE_CODE["bundle"]:
                if action_code == ACTION_CODE["enter"]:
                    if item_num in open_bundles:
                        bundle_attempt_index += 1
                        process_episode_index = self._finalize_bundle(
                            user_id, item_num, open_bundles.pop(item_num), None, bundle_attempt_index,
                            question_attempt_counter, stats, process_episode_index,
                        )
                    open_bundles[item_num] = SessionState(
                        ts, source_code, platform_code, seq_idx, implicit_enter=0,
                        access_snapshot=access_state.snapshot(ts),
                    )
                elif action_code == ACTION_CODE["submit"]:
                    session = open_bundles.pop(item_num, None)
                    if session is None:
                        session = SessionState(
                            None, source_code, platform_code, None, implicit_enter=1,
                            access_snapshot=access_state.snapshot(ts),
                        )
                    bundle_attempt_index += 1
                    process_episode_index = self._finalize_bundle(
                        user_id, item_num, session, ts, bundle_attempt_index, question_attempt_counter, stats, process_episode_index,
                    )
                continue

            if item_type_code == ITEM_TYPE_CODE["question"]:
                bundle_num = lookups.qnum_to_bundle_num.get(item_num)
                if bundle_num is None:
                    continue
                session = open_bundles.get(bundle_num)
                if session is None:
                    session = SessionState(
                        None, source_code, platform_code, None, implicit_enter=1,
                        access_snapshot=access_state.snapshot(ts),
                    )
                    open_bundles[bundle_num] = session
                q_state = self._question_state(session, item_num)
                if action_code == ACTION_CODE["respond"]:
                    q_state["answers"].append(answer_code)
                    q_state["response_times"].append(ts)
                elif action_code == ACTION_CODE["erase_choice"]:
                    q_state["erase_answers"].append(answer_code)
                    q_state["erase_times"].append(ts)
                elif action_code == ACTION_CODE["undo_erase_choice"]:
                    q_state["undo_answers"].append(answer_code)
                    q_state["undo_times"].append(ts)
                elif action_code == ACTION_CODE["text_enter"]:
                    q_state["text_enter_times"].append(ts)
                elif action in MEDIA_PROCESS_ACTIONS:
                    q_state["media_events"].append({"action": action, "ts": ts, "seq_idx": seq_idx, "cursor_time_ms": cursor_time_ms})
                continue

            if item_type_code in {ITEM_TYPE_CODE["explanation"], ITEM_TYPE_CODE["lecture"]}:
                study_key = (item_type_code, item_num)
                if action_code == ACTION_CODE["enter"]:
                    if study_key in open_studies:
                        previous_state = open_studies.pop(study_key)
                        previous_state.truncated_reenter = 1
                        study_episode_index += 1
                        process_episode_index += 1
                        self._finalize_study_episode(user_id, item_num, previous_state, ts, study_episode_index, process_episode_index, matched_quit=0, stats=stats)
                    open_studies[study_key] = StudyState(item_type_code, item_num, ts, source_code, platform_code, seq_idx, implicit_enter=0, truncated_reenter=0)
                elif action_code == ACTION_CODE["quit"]:
                    state = open_studies.pop(study_key, None)
                    if state is None:
                        state = StudyState(item_type_code, item_num, None, source_code, platform_code, None, implicit_enter=1, truncated_reenter=0)
                    study_episode_index += 1
                    process_episode_index += 1
                    self._finalize_study_episode(user_id, item_num, state, ts, study_episode_index, process_episode_index, matched_quit=1 if state.enter_ts is not None else 0, stats=stats)
                elif action in MEDIA_PROCESS_ACTIONS:
                    state = open_studies.get(study_key)
                    if state is None:
                        state = StudyState(item_type_code, item_num, None, source_code, platform_code, None, implicit_enter=1, truncated_reenter=0)
                        open_studies[study_key] = state
                    state.media_events.append({"action": action, "ts": ts, "seq_idx": seq_idx, "cursor_time_ms": cursor_time_ms})
                continue

        for bundle_num, session in list(open_bundles.items()):
            bundle_attempt_index += 1
            process_episode_index = self._finalize_bundle(
                user_id, bundle_num, session, None, bundle_attempt_index, question_attempt_counter, stats, process_episode_index,
            )
        for (_, item_num), state in list(open_studies.items()):
            study_episode_index += 1
            process_episode_index += 1
            self._finalize_study_episode(user_id, item_num, state, None, study_episode_index, process_episode_index, matched_quit=0, stats=stats)

        deltas = df["delta_ms"].to_numpy(dtype=np.int64)
        nonzero_deltas = deltas[deltas > 0]
        accuracy = stats.total_correct / stats.total_questions_answered if stats.total_questions_answered > 0 else np.nan
        mean_bundle_duration = float(np.mean(stats.bundle_durations)) if stats.bundle_durations else np.nan
        mean_bundle_accuracy = float(np.mean(stats.bundle_accuracies)) if stats.bundle_accuracies else np.nan

        self.derived_buffers.user_summary.append_row(
            {
                "user_id": user_id,
                "first_ts": int(df["timestamp"].iloc[0]),
                "last_ts": int(df["timestamp"].iloc[-1]),
                "active_span_ms": int(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]),
                "total_events": stats.total_events,
                "total_core_events": stats.total_core_events,
                "total_choice_process_events": stats.total_choice_process_events,
                "total_media_process_events": stats.total_media_process_events,
                "total_text_process_events": stats.total_text_process_events,
                "total_bundles_submitted": stats.total_bundles_submitted,
                "total_bundles_unsubmitted": stats.total_bundles_unsubmitted,
                "total_questions_answered": stats.total_questions_answered,
                "total_correct": stats.total_correct,
                "total_incorrect": stats.total_incorrect,
                "accuracy": accuracy,
                "total_explanation_episodes": stats.total_explanation_episodes,
                "total_lecture_episodes": stats.total_lecture_episodes,
                "total_explanation_dwell_ms": stats.total_explanation_dwell_ms,
                "total_lecture_dwell_ms": stats.total_lecture_dwell_ms,
                "total_media_elapsed_ms": stats.total_media_elapsed_ms,
                "n_unique_sources": len(stats.unique_sources),
                "n_unique_platforms": len(stats.unique_platforms),
                "mean_delta_ms": float(nonzero_deltas.mean()) if nonzero_deltas.size else np.nan,
                "std_delta_ms": float(nonzero_deltas.std(ddof=0)) if nonzero_deltas.size else np.nan,
                "p95_delta_ms": float(np.percentile(nonzero_deltas, 95)) if nonzero_deltas.size else np.nan,
                "mean_bundle_duration_ms": mean_bundle_duration,
                "mean_bundle_accuracy": mean_bundle_accuracy,
            }
        )

    def _finalize_bundle(
        self,
        user_id: int,
        bundle_num: int,
        session: SessionState,
        submit_ts: Optional[int],
        bundle_attempt_index: int,
        question_attempt_counter: defaultdict,
        stats: UserStats,
        process_episode_index: int,
    ) -> int:
        assert self.lookups is not None
        expected_questions = self.lookups.bundle_num_to_expected_questions.get(bundle_num)
        part = self.lookups.bundle_num_to_part.get(bundle_num)
        explanation_num = self.lookups.bundle_num_to_explanation_num.get(bundle_num)
        bundle_id = make_item_id("b", bundle_num)
        explanation_id = make_item_id("e", explanation_num)

        answered_question_count = 0
        total_response_count = 0
        total_response_change_count = 0
        total_erase_count = 0
        total_undo_erase_count = 0
        total_text_enter_count = 0
        total_question_media_event_count = 0
        total_question_media_elapsed = 0
        question_cursor_max_values: List[int] = []
        n_correct = 0
        correctness_values: List[int] = []
        all_choice_times: List[int] = []

        ordered_questions = sorted(
            session.responses.items(),
            key=lambda kv: (
                min(kv[1].get("response_times", []) + kv[1].get("erase_times", []) + kv[1].get("undo_times", []) + kv[1].get("text_enter_times", []) or [math.inf]),
                kv[0],
            ),
        )

        for question_position, (question_num, q_state) in enumerate(ordered_questions, start=1):
            answers: List[int] = [int(x) for x in q_state.get("answers", []) if int(x) > 0]
            response_times: List[int] = [int(x) for x in q_state.get("response_times", [])]
            erase_times: List[int] = [int(x) for x in q_state.get("erase_times", [])]
            undo_times: List[int] = [int(x) for x in q_state.get("undo_times", [])]
            text_enter_times: List[int] = [int(x) for x in q_state.get("text_enter_times", [])]
            media_metrics = compute_media_metrics(q_state.get("media_events", []), video_length_ms=np.nan)
            choice_times = response_times + erase_times + undo_times + text_enter_times
            all_choice_times.extend(choice_times)

            erase_count = len(erase_times)
            undo_erase_count = len(undo_times)
            text_enter_count = len(text_enter_times)
            choice_process_count = erase_count + undo_erase_count + text_enter_count
            total_erase_count += erase_count
            total_undo_erase_count += undo_erase_count
            total_text_enter_count += text_enter_count
            total_question_media_event_count += int(media_metrics["media_event_count"])
            if not pd.isna(media_metrics["media_elapsed_ms"]):
                total_question_media_elapsed += int(media_metrics["media_elapsed_ms"])
            if not pd.isna(media_metrics["cursor_max_ms"]):
                question_cursor_max_values.append(int(media_metrics["cursor_max_ms"]))

            if not answers or not response_times:
                continue
            answered_question_count += 1
            response_count = len(answers)
            response_change_count = int(sum(a != b for a, b in zip(answers[1:], answers[:-1])))
            total_response_count += response_count
            total_response_change_count += response_change_count

            final_answer_code = int(answers[-1])
            correct_answer_code = int(self.lookups.qnum_to_correct_code.get(question_num, 0) or 0)
            is_correct = int(final_answer_code == correct_answer_code) if correct_answer_code > 0 else np.nan
            if is_correct == 1:
                n_correct += 1
            if not pd.isna(is_correct):
                correctness_values.append(int(is_correct))

            first_response_ts = int(response_times[0])
            final_response_ts = int(response_times[-1])
            enter_to_first = first_response_ts - session.enter_ts if session.enter_ts is not None else np.nan
            enter_to_final = final_response_ts - session.enter_ts if session.enter_ts is not None else np.nan
            final_to_submit = submit_ts - final_response_ts if submit_ts is not None else np.nan
            bundle_duration_ms = submit_ts - session.enter_ts if (submit_ts is not None and session.enter_ts is not None) else np.nan
            last_process_ts = max(choice_times) if choice_times else final_response_ts
            first_process_ts = min(choice_times) if choice_times else first_response_ts
            last_change_latency = submit_ts - last_process_ts if submit_ts is not None else np.nan

            question_attempt_counter[question_num] += 1
            question_attempt_index_global = int(question_attempt_counter[question_num])
            question_id = make_item_id("q", question_num)
            self.derived_buffers.question_attempts.append_row(
                {
                    "user_id": user_id,
                    "bundle_attempt_index": bundle_attempt_index,
                    "question_attempt_index": question_attempt_index_global,
                    "question_attempt_index_global": question_attempt_index_global,
                    "question_position_within_bundle_attempt": question_position,
                    "bundle_id": bundle_id,
                    "question_id": question_id,
                    "part": self.lookups.qnum_to_part.get(question_num),
                    "source_code": session.source_code,
                    "platform_code": session.platform_code,
                    "bundle_enter_ts": session.enter_ts,
                    "first_response_ts": first_response_ts,
                    "final_response_ts": final_response_ts,
                    "submit_ts": submit_ts,
                    "bundle_duration_ms": bundle_duration_ms,
                    "latency_enter_to_first_ms": enter_to_first,
                    "latency_enter_to_final_ms": enter_to_final,
                    "latency_final_to_submit_ms": final_to_submit,
                    "response_count": response_count,
                    "response_change_count": response_change_count,
                    "final_answer_code": final_answer_code,
                    "correct_answer_code": correct_answer_code,
                    "is_correct": is_correct,
                    "implicit_bundle_enter": session.implicit_enter,
                    "erase_count": erase_count,
                    "undo_erase_count": undo_erase_count,
                    "choice_process_count": choice_process_count,
                    "text_enter_count": text_enter_count,
                    "first_choice_process_ts": first_process_ts,
                    "last_choice_process_ts": last_process_ts,
                    "last_change_latency_ms": last_change_latency,
                    "question_audio_play_count": media_metrics["n_play_audio"],
                    "question_audio_pause_count": media_metrics["n_pause_audio"],
                    "question_video_play_count": media_metrics["n_play_video"],
                    "question_video_pause_count": media_metrics["n_pause_video"],
                    "question_media_event_count": media_metrics["media_event_count"],
                    "question_media_elapsed_ms": media_metrics["media_elapsed_ms"],
                    "question_cursor_max_ms": media_metrics["cursor_max_ms"],
                    "question_media_observed": media_metrics["media_observed"],
                }
            )

        missing_question_count = int(expected_questions - answered_question_count) if expected_questions is not None and not pd.isna(expected_questions) else np.nan
        duration_ms = submit_ts - session.enter_ts if (submit_ts is not None and session.enter_ts is not None) else np.nan
        accuracy = float(np.mean(correctness_values)) if correctness_values else np.nan
        total_choice_process_count = total_erase_count + total_undo_erase_count + total_text_enter_count
        choice_process_mass_raw = total_response_change_count + total_erase_count + total_undo_erase_count + total_text_enter_count
        first_choice_process_ts = min(all_choice_times) if all_choice_times else np.nan
        last_choice_process_ts = max(all_choice_times) if all_choice_times else np.nan
        question_media_elapsed_ms = int(total_question_media_elapsed) if total_question_media_elapsed > 0 else np.nan
        question_cursor_max_ms = int(max(question_cursor_max_values)) if question_cursor_max_values else np.nan
        question_media_observed = 1 if total_question_media_event_count > 0 else 0

        bundle_row = {
            "user_id": user_id,
            "bundle_attempt_index": bundle_attempt_index,
            "bundle_id": bundle_id,
            "explanation_id": explanation_id,
            "part": part,
            "source_code": session.source_code,
            "platform_code": session.platform_code,
            "enter_ts": session.enter_ts,
            "submit_ts": submit_ts,
            "duration_ms": duration_ms,
            "expected_question_count": expected_questions,
            "answered_question_count": answered_question_count,
            "missing_question_count": missing_question_count,
            "total_response_count": total_response_count,
            "total_response_change_count": total_response_change_count,
            "n_correct": n_correct,
            "accuracy": accuracy,
            "implicit_enter": session.implicit_enter,
            "missing_submit": 0 if submit_ts is not None else 1,
            "total_erase_count": total_erase_count,
            "total_undo_erase_count": total_undo_erase_count,
            "total_choice_process_count": total_choice_process_count,
            "total_text_enter_count": total_text_enter_count,
            "choice_process_mass_raw": choice_process_mass_raw,
            "choice_process_observed": 1 if total_choice_process_count > 0 else 0,
            "first_choice_process_ts": first_choice_process_ts,
            "last_choice_process_ts": last_choice_process_ts,
            "total_question_media_event_count": total_question_media_event_count,
            "question_media_elapsed_ms": question_media_elapsed_ms,
            "question_cursor_max_ms": question_cursor_max_ms,
            "question_media_observed": question_media_observed,
            **session.access_snapshot,
        }
        self.derived_buffers.bundle_attempts.append_row(bundle_row)
        self.derived_buffers.choice_process_features.append_row(
            {
                "user_id": user_id,
                "bundle_attempt_index": bundle_attempt_index,
                "bundle_id": bundle_id,
                "enter_ts": session.enter_ts,
                "submit_ts": submit_ts,
                "total_response_count": total_response_count,
                "total_response_change_count": total_response_change_count,
                "total_erase_count": total_erase_count,
                "total_undo_erase_count": total_undo_erase_count,
                "total_text_enter_count": total_text_enter_count,
                "choice_process_count": total_choice_process_count,
                "choice_process_mass_raw": choice_process_mass_raw,
                "answered_question_count": answered_question_count,
                "choice_process_observed": 1 if total_choice_process_count > 0 else 0,
                "source_code": session.source_code,
                "platform_code": session.platform_code,
            }
        )

        # Store question media events at bundle scope.
        if total_question_media_event_count > 0:
            process_episode_index += 1
            self.derived_buffers.media_process_features.append_row(
                {
                    "user_id": user_id,
                    "process_episode_index": process_episode_index,
                    "process_scope": "bundle_question_media",
                    "item_type_code": ITEM_TYPE_CODE["bundle"],
                    "item_id": bundle_id,
                    "related_bundle_id": bundle_id,
                    "part": part,
                    "source_code": session.source_code,
                    "platform_code": session.platform_code,
                    "enter_ts": session.enter_ts,
                    "exit_ts": submit_ts,
                    "dwell_ms": duration_ms,
                    "video_length_ms": np.nan,
                    "n_play_audio": np.nan,
                    "n_pause_audio": np.nan,
                    "n_play_video": np.nan,
                    "n_pause_video": np.nan,
                    "media_event_count": total_question_media_event_count,
                    "media_elapsed_ms": question_media_elapsed_ms,
                    "cursor_min_ms": np.nan,
                    "cursor_max_ms": question_cursor_max_ms,
                    "cursor_span_ms": np.nan,
                    "media_completion_proxy": np.nan,
                    "media_observed": question_media_observed,
                    "media_pair_anomaly_count": np.nan,
                }
            )

        if submit_ts is not None:
            stats.total_bundles_submitted += 1
            if not pd.isna(duration_ms):
                stats.bundle_durations.append(int(duration_ms))
            if not pd.isna(accuracy):
                stats.bundle_accuracies.append(float(accuracy))
        else:
            stats.total_bundles_unsubmitted += 1
        stats.total_questions_answered += answered_question_count
        stats.total_correct += n_correct
        if answered_question_count >= n_correct:
            stats.total_incorrect += answered_question_count - n_correct
        if not pd.isna(question_media_elapsed_ms):
            stats.total_media_elapsed_ms += int(question_media_elapsed_ms)
        return process_episode_index

    def _finalize_study_episode(
        self,
        user_id: int,
        item_num: int,
        state: StudyState,
        quit_ts: Optional[int],
        study_episode_index: int,
        process_episode_index: int,
        matched_quit: int,
        stats: UserStats,
    ) -> None:
        assert self.lookups is not None
        item_type_code = state.item_type_code
        item_type = ITEM_TYPE_NAME_FROM_CODE[item_type_code]
        item_id = make_item_id(ITEM_PREFIX_FROM_TYPE_CODE[item_type_code], item_num)
        dwell_ms = quit_ts - state.enter_ts if (quit_ts is not None and state.enter_ts is not None) else np.nan
        part = np.nan
        related_bundle_id = None
        video_length_ms = np.nan
        dwell_ratio = np.nan

        if item_type == "explanation":
            part = self.lookups.explanation_num_to_part.get(item_num)
            related_bundle_id = make_item_id("b", self.lookups.explanation_num_to_bundle_num.get(item_num))
            stats.total_explanation_episodes += 1
            if not pd.isna(dwell_ms):
                stats.total_explanation_dwell_ms += int(dwell_ms)
        elif item_type == "lecture":
            part = self.lookups.lecture_num_to_part.get(item_num)
            video_length_ms = self.lookups.lecture_num_to_video_length.get(item_num)
            if video_length_ms is not None and not pd.isna(video_length_ms) and video_length_ms > 0 and not pd.isna(dwell_ms):
                dwell_ratio = float(dwell_ms) / float(video_length_ms)
            stats.total_lecture_episodes += 1
            if not pd.isna(dwell_ms):
                stats.total_lecture_dwell_ms += int(dwell_ms)

        media_metrics = compute_media_metrics(state.media_events, video_length_ms=video_length_ms)
        if not pd.isna(media_metrics["media_elapsed_ms"]):
            stats.total_media_elapsed_ms += int(media_metrics["media_elapsed_ms"])

        study_row = {
            "user_id": user_id,
            "study_episode_index": study_episode_index,
            "item_type_code": item_type_code,
            "item_id": item_id,
            "part": part,
            "source_code": state.source_code,
            "platform_code": state.platform_code,
            "enter_ts": state.enter_ts,
            "quit_ts": quit_ts,
            "dwell_ms": dwell_ms,
            "dwell_ratio": dwell_ratio,
            "video_length_ms": video_length_ms,
            "related_bundle_id": related_bundle_id,
            "implicit_enter": state.implicit_enter,
            "matched_quit": matched_quit,
            "truncated_reenter": state.truncated_reenter,
            "n_play_audio": media_metrics["n_play_audio"],
            "n_pause_audio": media_metrics["n_pause_audio"],
            "n_play_video": media_metrics["n_play_video"],
            "n_pause_video": media_metrics["n_pause_video"],
            "media_event_count": media_metrics["media_event_count"],
            "media_elapsed_ms": media_metrics["media_elapsed_ms"],
            "cursor_min_ms": media_metrics["cursor_min_ms"],
            "cursor_max_ms": media_metrics["cursor_max_ms"],
            "cursor_span_ms": media_metrics["cursor_span_ms"],
            "media_completion_proxy": media_metrics["media_completion_proxy"],
            "media_observed": media_metrics["media_observed"],
            "media_pair_anomaly_count": media_metrics["media_pair_anomaly_count"],
        }
        self.derived_buffers.study_episodes.append_row(study_row)
        self.derived_buffers.media_process_features.append_row(
            {
                "user_id": user_id,
                "process_episode_index": process_episode_index,
                "process_scope": item_type,
                "item_type_code": item_type_code,
                "item_id": item_id,
                "related_bundle_id": related_bundle_id,
                "part": part,
                "source_code": state.source_code,
                "platform_code": state.platform_code,
                "enter_ts": state.enter_ts,
                "exit_ts": quit_ts,
                "dwell_ms": dwell_ms,
                "video_length_ms": video_length_ms,
                "n_play_audio": media_metrics["n_play_audio"],
                "n_pause_audio": media_metrics["n_pause_audio"],
                "n_play_video": media_metrics["n_play_video"],
                "n_pause_video": media_metrics["n_pause_video"],
                "media_event_count": media_metrics["media_event_count"],
                "media_elapsed_ms": media_metrics["media_elapsed_ms"],
                "cursor_min_ms": media_metrics["cursor_min_ms"],
                "cursor_max_ms": media_metrics["cursor_max_ms"],
                "cursor_span_ms": media_metrics["cursor_span_ms"],
                "media_completion_proxy": media_metrics["media_completion_proxy"],
                "media_observed": media_metrics["media_observed"],
                "media_pair_anomaly_count": media_metrics["media_pair_anomaly_count"],
            }
        )

    def _flush_derived_buffers(self, reason: str, visualizer: ProgressVisualizer) -> None:
        if not any([
            len(self.derived_buffers.question_attempts), len(self.derived_buffers.bundle_attempts),
            len(self.derived_buffers.study_episodes), len(self.derived_buffers.choice_process_features),
            len(self.derived_buffers.media_process_features), len(self.derived_buffers.user_summary),
        ]):
            return
        question_df = self.derived_buffers.question_attempts.to_dataframe()
        bundle_df = self.derived_buffers.bundle_attempts.to_dataframe()
        study_df = self.derived_buffers.study_episodes.to_dataframe()
        choice_df = self.derived_buffers.choice_process_features.to_dataframe()
        media_df = self.derived_buffers.media_process_features.to_dataframe()
        user_df = self.derived_buffers.user_summary.to_dataframe()

        path = self.save_manager.write(question_df, self.question_dir, "question_attempts_chunk", label="question_attempts", stage="derived")
        if path is not None:
            self._record_output("question_attempts")
        path = self.save_manager.write(bundle_df, self.bundle_dir, "bundle_attempts_chunk", label="bundle_attempts", stage="derived")
        if path is not None:
            self._record_output("bundle_attempts")
        path = self.save_manager.write(study_df, self.study_dir, "study_episodes_chunk", label="study_episodes", stage="derived")
        if path is not None:
            self._record_output("study_episodes")
        path = self.save_manager.write(choice_df, self.choice_process_dir, "choice_process_features_chunk", label="choice_process_features", stage="derived_process")
        if path is not None:
            self._record_output("choice_process_features")
        path = self.save_manager.write(media_df, self.media_process_dir, "media_process_features_chunk", label="media_process_features", stage="derived_process")
        if path is not None:
            self._record_output("media_process_features")
        path = self.save_manager.write(user_df, self.user_summary_dir, "user_summary_chunk", label="user_summary", stage="derived")
        if path is not None:
            self._record_output("user_summary")
        self.derived_buffers.clear()
        visualizer.record_flush(reason=reason)
        gc.collect()

    def _write_manifest(self) -> None:
        registry_path = self.save_manager.write_registry()
        if registry_path is not None:
            self._record_output("shard_registry")

        manifest = {
            "dataset": "EdNet-KT4",
            "kt4_input": str(self.kt4_input),
            "contents_input": str(self.contents_input),
            "output_root": str(self.output_root),
            "parquet_only": True,
            "clean_output_dirs": CLEAN_OUTPUT_DIRS,
            "known_actions": sorted(KNOWN_ACTIONS),
            "core_actions": sorted(CORE_ACTIONS),
            "choice_process_actions": sorted(CHOICE_PROCESS_ACTIONS),
            "media_process_actions": sorted(MEDIA_PROCESS_ACTIONS),
            "text_process_actions": sorted(TEXT_PROCESS_ACTIONS),
            "action_aliases": ACTION_ALIASES,
            "known_item_prefix": sorted(KNOWN_ITEM_PREFIX),
            "action_sort_priority": ACTION_SORT_PRIORITY,
            "flush_thresholds": {
                "events": FLUSH_EVENTS_ROWS,
                "events_light": FLUSH_EVENTS_LIGHT_ROWS,
                "skipped_users": FLUSH_SKIPPED_USERS_ROWS,
                "question_attempts": FLUSH_QUESTION_ROWS,
                "bundle_attempts": FLUSH_BUNDLE_ROWS,
                "study_episodes": FLUSH_STUDY_ROWS,
                "choice_process_features": FLUSH_CHOICE_PROCESS_ROWS,
                "media_process_features": FLUSH_MEDIA_PROCESS_ROWS,
                "user_summary": FLUSH_USER_SUMMARY_ROWS,
            },
            "pipeline_stages": [
                "contents_canonicalization_reused_from_kt3_v3",
                "kt4_canonical_events_generation_with_process_actions",
                "kt4_derived_attempts_episodes_process_features_generation",
            ],
            "source_code": SOURCE_CODE,
            "platform_code": PLATFORM_CODE,
            "action_code": ACTION_CODE,
            "action_group_code": ACTION_GROUP_CODE,
            "item_type_code": ITEM_TYPE_CODE,
            "answer_code": ANSWER_CODE,
            "raw_action_counts": dict(self.raw_action_counter),
            "kept_action_counts": dict(self.kept_action_counter),
            "filtered_action_counts": dict(self.filtered_action_counter),
            "raw_item_prefix_counts_after_action_filter": dict(self.raw_item_prefix_counter),
            "run_stats": self.run_stats,
            "output_counts": dict(self.output_counts),
            "registry_path": str(registry_path) if registry_path is not None else None,
            "output_contract": {
                "kt3_compatible_core_tables": [
                    "contents/questions_clean.parquet",
                    "contents/lectures_clean.parquet",
                    "contents/bundles_clean.parquet",
                    "contents/explanations_clean.parquet",
                    "contents/question_tag_edges.parquet",
                    "contents/lecture_tag_edges.parquet",
                    "kt4/events/events_chunk_*.parquet",
                    "kt4/events_light/events_light_chunk_*.parquet",
                    "kt4/question_attempts/question_attempts_chunk_*.parquet",
                    "kt4/bundle_attempts/bundle_attempts_chunk_*.parquet",
                    "kt4/study_episodes/study_episodes_chunk_*.parquet",
                    "kt4/user_summary/user_summary_chunk_*.parquet",
                ],
                "kt4_process_tables": [
                    "kt4/choice_process_features/choice_process_features_chunk_*.parquet",
                    "kt4/media_process_features/media_process_features_chunk_*.parquet",
                ],
                "kt4_access_content_tables": [
                    "contents/payments_clean.parquet",
                    "contents/coupons_clean.parquet",
                ],
                "kt4_access_context_columns_in_bundle_attempts": [
                    "access_status_aux",
                    "payment_coupon_window_aux",
                    "access_aux_observed",
                    "access_pay_event_count",
                    "access_refund_event_count",
                    "access_coupon_event_count",
                ],
            },
        }
        manifest_path = self.metadata_dir / "preprocess_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)


def main() -> None:
    processor = PreprocessorKT4(kt4_input=KT4_INPUT, contents_input=CONTENTS_INPUT, output_root=OUTPUT_ROOT)
    processor.run()


if __name__ == "__main__":
    main()
