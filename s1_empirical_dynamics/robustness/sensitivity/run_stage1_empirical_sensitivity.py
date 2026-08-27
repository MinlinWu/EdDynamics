#!/usr/bin/env python3
"""Run one minimal Stage-1 coordinate-sensitivity variant.

The publication Stage-1 implementation is imported unchanged. This runner
reuses its preprocessing, coordinate construction, field estimation, and
training-to-validation convergence protocol, while intentionally omitting
B_confirm and the fixed-K mesostate analysis because the supplementary
experiment concerns only coordinate-construction sensitivity.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict

import pandas as pd


VARIANTS: Dict[str, Dict[str, float]] = {
    "memory_5d": {
        "tau_response_days": 5.0,
        "tau_activity_days": 5.0,
        "response_half_sat_min": 3.0,
        "explanation_half_sat_min": 2.5,
        "lecture_half_sat_min": 4.0,
        "idle_half_sat_days": 1.0,
    },
    "memory_20d": {
        "tau_response_days": 20.0,
        "tau_activity_days": 20.0,
        "response_half_sat_min": 3.0,
        "explanation_half_sat_min": 2.5,
        "lecture_half_sat_min": 4.0,
        "idle_half_sat_days": 1.0,
    },
    "activity_fast": {
        "tau_response_days": 10.0,
        "tau_activity_days": 10.0,
        "response_half_sat_min": 2.0,
        "explanation_half_sat_min": 2.0,
        "lecture_half_sat_min": 3.0,
        "idle_half_sat_days": 0.5,
    },
    "activity_slow": {
        "tau_response_days": 10.0,
        "tau_activity_days": 10.0,
        "response_half_sat_min": 4.0,
        "explanation_half_sat_min": 4.0,
        "lecture_half_sat_min": 6.0,
        "idle_half_sat_days": 2.0,
    },
}


PUBLICATION_FIXED_ENV = {
    "EDNET_STAGE1_A_TRAIN_USERS": "178749",
    "EDNET_STAGE1_A_VAL_USERS": "59583",
    "EDNET_STAGE1_B_CONFIRM_USERS": "59583",
    "EDNET_STAGE1_ALLOW_SMALL_DEV_SPLIT": "0",
    "EDNET_STAGE1_RANDOM_STATE": "42",
    "EDNET_STAGE1_EVIDENCE_MATURITY_SCALE": "20.0",
    "EDNET_STAGE1_TAG_PRIOR_KAPPA": "20.0",
    "EDNET_STAGE1_ITEM_PRIOR_KAPPA": "50.0",
    "EDNET_STAGE1_OBSERVATION_HORIZON_DAYS": "7.0",
    "EDNET_STAGE1_LONG_GAP_DAYS": "7.0",
    "EDNET_STAGE1_MAX_SUPPORT_EPISODE_ACTIVE": "1.0",
    "EDNET_STAGE1_SIGNED_GRID_N": "41",
    "EDNET_STAGE1_MIN_STATE_BIN_COUNT": "50",
    "EDNET_STAGE1_MIN_DRIFT_BIN_COUNT": "30",
    "EDNET_STAGE1_MIN_CELL_USERS": "5",
    "EDNET_STAGE1_CONVERGENCE_SPEED_QUANTILE": "0.60",
    "EDNET_STAGE1_CONVERGENCE_NEGATIVE_DIVERGENCE_QUANTILE": "0.80",
    "EDNET_STAGE1_CONVERGENCE_RATIO_QUANTILE": "0.60",
    "EDNET_STAGE1_CONVERGENCE_MIN_CELLS": "4",
    "EDNET_STAGE1_CONVERGENCE_SHELL_RADIUS": "0.35",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-script", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_environment(variant: str, output_root: Path) -> Dict[str, float]:
    values = dict(VARIANTS[variant])
    for key, value in PUBLICATION_FIXED_ENV.items():
        os.environ[key] = value
    os.environ["EDNET_KT4_OUTPUT_ROOT"] = str(output_root.resolve())
    os.environ["EDNET_STAGE1_TAU_RESPONSE_DAYS"] = str(values["tau_response_days"])
    os.environ["EDNET_STAGE1_TAU_ACTIVITY_DAYS"] = str(values["tau_activity_days"])
    os.environ["EDNET_STAGE1_RESPONSE_DURATION_HALF_SAT_MIN"] = str(values["response_half_sat_min"])
    os.environ["EDNET_STAGE1_EXPLANATION_HALF_SAT_MIN"] = str(values["explanation_half_sat_min"])
    os.environ["EDNET_STAGE1_LECTURE_HALF_SAT_MIN"] = str(values["lecture_half_sat_min"])
    os.environ["EDNET_STAGE1_IDLE_HALF_SAT_DAYS"] = str(values["idle_half_sat_days"])
    return values


def load_source_module(path: Path) -> ModuleType:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Empirical Stage-1 source script not found: {source}")
    spec = importlib.util.spec_from_file_location("ednet_stage1_empirical_source", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Stage-1 source script: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_output_root(output_root: Path, variant: str, overwrite: bool) -> bool:
    output_root = output_root.resolve()
    done = output_root / "stage1" / "metadata" / "stage1_sensitivity_manifest.json"
    if done.exists() and not overwrite:
        with done.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        existing_variant = manifest.get("variant")
        if existing_variant != variant:
            raise RuntimeError(
                f"Existing completed output is for {existing_variant!r}, not {variant!r}: {output_root}"
            )
        print(f"Sensitivity variant already complete; skipping: {variant}")
        return False
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output root is non-empty and has no matching completion manifest: {output_root}. "
                "Use --overwrite only after verifying the path."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    return True



def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required archived formal output is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def validate_baseline_input_contract(module: ModuleType, baseline_root: Path) -> Dict[str, object]:
    manifest_path = baseline_root / "stage1" / "metadata" / "stage1_empirical_v3_manifest.json"
    manifest = load_json(manifest_path)
    preprocess = manifest.get("input_preprocess_manifest")
    if not isinstance(preprocess, dict):
        raise ValueError(f"Missing input_preprocess_manifest in {manifest_path}")
    preprocess_root_value = preprocess.get("output_root")
    if not preprocess_root_value:
        raise ValueError(f"Missing preprocessing output_root in {manifest_path}")
    preprocess_root = Path(str(preprocess_root_value)).resolve()
    expected_kt4 = (preprocess_root / "kt4").resolve()
    expected_contents = (preprocess_root / "contents").resolve()
    if module.KT4_ROOT.resolve() != expected_kt4:
        raise RuntimeError(
            "Current EDNET_KT4_PREPROCESSED_ROOT does not match the formal Stage-1 input: "
            f"current={module.KT4_ROOT.resolve()}, formal={expected_kt4}"
        )
    if module.CONTENTS_ROOT.resolve() != expected_contents:
        raise RuntimeError(
            "Current EDNET_CONTENTS_ROOT does not match the formal Stage-1 input: "
            f"current={module.CONTENTS_ROOT.resolve()}, formal={expected_contents}"
        )
    return manifest


def load_frozen_splits(module: ModuleType, baseline_root: Path) -> Dict[str, object]:
    splits = {}
    expected = {
        "A_train": int(module.A_TRAIN_USERS),
        "A_val": int(module.A_VAL_USERS),
        "B_confirm": int(module.B_CONFIRM_USERS),
    }
    all_ids = []
    for split, expected_count in expected.items():
        table = module.read_table(baseline_root / "stage1" / "splits" / f"{split}_users")
        if "user_id" not in table.columns:
            raise KeyError(f"Formal {split} split table is missing user_id")
        ids = pd.to_numeric(table["user_id"], errors="raise").to_numpy(dtype="int64")
        ids = module.np.sort(module.np.unique(ids))
        if len(ids) != expected_count:
            raise RuntimeError(
                f"Formal {split} split has {len(ids)} users; expected {expected_count}."
            )
        splits[split] = ids
        all_ids.append(ids)
    combined = module.np.concatenate(all_ids)
    if len(module.np.unique(combined)) != len(combined):
        raise RuntimeError("Formal user split tables are not mutually disjoint.")
    return splits


def load_frozen_priors(
    module: ModuleType,
    content,
    baseline_root: Path,
):
    metadata_root = baseline_root / "stage1" / "metadata"
    tag_table = module.read_table(metadata_root / "A_train_tag_correctness_priors")
    question_table = module.read_table(metadata_root / "A_train_question_itemEB_correctness_priors")
    tag_required = {
        "tag",
        "prior_correct_Atrain_shrunk",
        "global_prior_correct_Atrain",
        "prior_kappa",
    }
    question_required = {
        "question_id",
        "prior_correct_itemEB_Atrain",
        "item_eb_kappa",
    }
    if not tag_required.issubset(tag_table.columns):
        raise KeyError(f"Archived tag-prior table is missing: {sorted(tag_required - set(tag_table.columns))}")
    if not question_required.issubset(question_table.columns):
        raise KeyError(
            f"Archived item-prior table is missing: {sorted(question_required - set(question_table.columns))}"
        )
    tag_kappa = pd.to_numeric(tag_table["prior_kappa"], errors="raise").dropna().unique()
    item_kappa = pd.to_numeric(question_table["item_eb_kappa"], errors="raise").dropna().unique()
    if len(tag_kappa) != 1 or not module.np.isclose(float(tag_kappa[0]), module.TAG_PRIOR_KAPPA):
        raise RuntimeError("Archived tag-prior kappa does not match the fixed publication value.")
    if len(item_kappa) != 1 or not module.np.isclose(float(item_kappa[0]), module.ITEM_PRIOR_KAPPA):
        raise RuntimeError("Archived item-prior kappa does not match the fixed publication value.")
    tag_map = {
        int(tag): float(prior)
        for tag, prior in tag_table[["tag", "prior_correct_Atrain_shrunk"]].itertuples(index=False)
    }
    missing_tags = [
        int(content.idx_to_tag[index])
        for index in range(len(content.idx_to_tag))
        if int(content.idx_to_tag[index]) not in tag_map
    ]
    if missing_tags:
        raise RuntimeError(f"Archived tag priors do not cover {len(missing_tags)} current tags.")
    tag_priors = module.np.asarray(
        [tag_map[int(content.idx_to_tag[index])] for index in range(len(content.idx_to_tag))],
        dtype=float,
    )
    global_values = pd.to_numeric(
        tag_table["global_prior_correct_Atrain"], errors="raise"
    ).dropna().unique()
    if len(global_values) != 1:
        raise RuntimeError("Archived tag-prior table does not contain one frozen global prior.")
    question_priors = {
        str(question_id): float(prior)
        for question_id, prior in question_table[
            ["question_id", "prior_correct_itemEB_Atrain"]
        ].itertuples(index=False)
    }
    known_questions = set(content.question_tag_idx)
    if not known_questions.issubset(question_priors):
        missing = len(known_questions.difference(question_priors))
        raise RuntimeError(f"Archived item priors do not cover {missing} current questions.")
    return module.TagPriorEstimate(
        priors=tag_priors,
        table=tag_table,
        global_prior=float(global_values[0]),
        question_priors=question_priors,
        question_table=question_table,
    )


def load_frozen_chunk_audit(baseline_root: Path) -> Dict[str, object]:
    path = baseline_root / "stage1" / "metadata" / "bundle_chunk_user_completeness_audit.json"
    audit = load_json(path)
    if not bool(audit.get("gate_passed", False)):
        raise RuntimeError("The archived formal bundle-chunk completeness audit did not pass.")
    return audit

def run_variant(
    module: ModuleType,
    source_script: Path,
    baseline_root: Path,
    variant: str,
    values: Dict[str, float],
) -> None:
    module.ensure_dirs()
    baseline_root = baseline_root.resolve()
    formal_manifest = validate_baseline_input_contract(module, baseline_root)
    splits = load_frozen_splits(module, baseline_root)
    analysis_splits = ("A_train", "A_val")
    split_map = {
        int(user_id): split
        for split in analysis_splits
        for user_id in splits[split].tolist()
    }

    split_manifest = {
        "random_state": int(module.RANDOM_STATE),
        "source_user_count": int(sum(len(user_ids) for user_ids in splits.values())),
        "source": str((baseline_root / "stage1" / "splits").resolve()),
        "sizes": {name: int(len(user_ids)) for name, user_ids in splits.items()},
        "sensitivity_analysis_splits": list(analysis_splits),
        "B_confirm_policy": "not processed or accessed in coordinate-sensitivity runs",
    }
    with (module.SPLIT_ROOT / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2)
    for split, user_ids in splits.items():
        module.write_table(pd.DataFrame({"user_id": user_ids}), module.SPLIT_ROOT / f"{split}_users")

    chunk_audit = load_frozen_chunk_audit(baseline_root)
    with (module.META_OUT_ROOT / "bundle_chunk_user_completeness_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(chunk_audit, handle, indent=2)
    content = module.ContentsBuilder().build()
    priors = load_frozen_priors(module, content, baseline_root)
    module.write_table(priors.table, module.META_OUT_ROOT / "A_train_tag_correctness_priors")
    module.write_table(
        priors.question_table,
        module.META_OUT_ROOT / "A_train_question_itemEB_correctness_priors",
    )

    raw_paths = module.build_raw_panels(content, priors, split_map)
    finalize_manifest = module.finalize_stage1_panels(raw_paths)

    train_df = module.read_core_split("A_train")
    val_df = module.read_core_split("A_val")
    spec = module.coordinate_specs()[0]
    required = [spec.xcol, spec.ycol, spec.dxcol, spec.dycol]
    for split, frame in {"A_train": train_df, "A_val": val_df}.items():
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise KeyError(f"{split} is missing primary state columns: {missing}")

    coordinate_summary = module.analyze_coordinate(
        train_df,
        val_df,
        pd.DataFrame(),
        spec,
    )

    training_regions = module.read_table(
        module.REGION_ROOT / spec.name / "training_flow_defined_convergence_regions"
    )
    fallback_used = False
    if not training_regions.empty and "flow_defined_fallback_used" in training_regions.columns:
        fallback_used = bool(training_regions.iloc[0]["flow_defined_fallback_used"])

    manifest = {
        "script": Path(__file__).name,
        "source_empirical_script": str(source_script.resolve()),
        "variant": variant,
        "variant_parameters": values,
        "fixed_publication_parameters": {
            key: os.environ[key] for key in sorted(PUBLICATION_FIXED_ENV)
        },
        "formal_stage1_manifest_source": str(
            (baseline_root / "stage1" / "metadata" / "stage1_empirical_v3_manifest.json").resolve()
        ),
        "formal_input_preprocess_manifest": formal_manifest.get("input_preprocess_manifest"),
        "split_manifest": split_manifest,
        "bundle_chunk_user_completeness_audit": chunk_audit,
        "finalize_manifest": finalize_manifest,
        "coordinate_summary": coordinate_summary,
        "training_core_fallback_used": fallback_used,
        "analysis_scope": {
            "A_train": "construct priors, field criteria, and convergence core",
            "A_val": "held-out coordinate and field sensitivity evaluation",
            "B_confirm": "not processed or accessed",
            "construction_matched_null": "not rerun",
            "fixed_k6_mesostates": "not rerun",
            "minimal_mechanism": "not rerun",
            "Event_SSL": "not rerun",
        },
        "guardrails": [
            "the publication preprocessing tables are reused unchanged",
            "the archived publication user split is reused exactly",
            "the archived A_train tag and item priors are reused exactly",
            "only the declared memory or activity-saturation constants differ from the publication setting",
            "the A_train convergence algorithm is reapplied and frozen before A_val evaluation",
            "no validation result selects or modifies a variant",
            "a fallback-defined training core is recorded explicitly and must not be presented as formal robustness evidence",
        ],
    }
    manifest_path = module.META_OUT_ROOT / "stage1_sensitivity_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Completed Stage-1 sensitivity variant: {variant}")


def main() -> None:
    args = parse_args()
    if args.output_root.resolve() == args.baseline_root.resolve():
        raise ValueError("Sensitivity output root must differ from the formal baseline output root.")
    should_run = prepare_output_root(args.output_root, args.variant, args.overwrite)
    if not should_run:
        return
    values = configure_environment(args.variant, args.output_root)
    module = load_source_module(args.source_script)
    run_variant(module, args.source_script, args.baseline_root, args.variant, values)


if __name__ == "__main__":
    main()
