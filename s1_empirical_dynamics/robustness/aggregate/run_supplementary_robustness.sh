#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/datasets/KT4/outputs_KT4}"
ROBUSTNESS_ROOT="${ROBUSTNESS_ROOT:-${OUTPUT_BASE}/supplementary_robustness}"
STAGE1_ROOT="${STAGE1_ROOT:-${OUTPUT_BASE}/stage1}"
PHASE3_ROOT="${PHASE3_ROOT:-${OUTPUT_BASE}/stage2_phase3_confirm}"
FROZEN_MANIFEST="${FROZEN_MANIFEST:-${OUTPUT_BASE}/stage2_phase2_freeze/metadata/phase2_frozen_model_manifest.json}"
MAIN_ROOT="${MAIN_ROOT:-${OUTPUT_BASE}/stage4_event_ssl/evaluation_predictive_state}"
TIME_SHUFFLE_ROOT="${TIME_SHUFFLE_ROOT:-${OUTPUT_BASE}/stage4_event_ssl_time_shuffle_control/evaluation_on_ordered_inputs}"
TAG_SUPPORT_ROOT="${TAG_SUPPORT_ROOT:-${OUTPUT_BASE}/stage4_event_ssl_tag_support_randomized_control/evaluation}"
COORDINATE_SUMMARY_ROOT="${COORDINATE_SUMMARY_ROOT:-/data/datasets/KT4/outputs_KT4_stage1_sensitivity/summary}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-1000}"
BOOTSTRAP_CHUNK="${BOOTSTRAP_CHUNK:-25}"
PERMUTATION_REPLICATES="${PERMUTATION_REPLICATES:-1000}"
RANDOM_SEED="${RANDOM_SEED:-42}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_COLLECTION="${RUN_COLLECTION:-1}"
RUN_NUMERICAL_REPORT="${RUN_NUMERICAL_REPORT:-1}"

if [[ -z "${MACRO_ROOT:-}" ]]; then
    for candidate in \
        "${OUTPUT_BASE}/stage5_event_ssl_macro_sufficiency/evaluation" \
        "${OUTPUT_BASE}/stage5_macro_sufficiency/evaluation" \
        "${OUTPUT_BASE}/stage5_event_ssl/macro_sufficiency/evaluation"; do
        if [[ -d "$candidate" ]]; then
            MACRO_ROOT="$candidate"
            break
        fi
    done
    MACRO_ROOT="${MACRO_ROOT:-${OUTPUT_BASE}/stage5_event_ssl_macro_sufficiency/evaluation}"
fi

if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]]; then
    echo "MAX_PARALLEL must be a non-negative integer." >&2
    exit 1
fi
resolve_script() {
    local explicit="$1"
    shift
    if [[ -n "$explicit" ]]; then
        printf '%s\n' "$explicit"
        return
    fi
    local candidate
    for candidate in "$@"; do
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    return 1
}

STAGE1_SCRIPT="$(resolve_script "${STAGE1_SCRIPT:-}" \
    "${SCRIPT_DIR}/build_effective_dynamics_kt4_stage1_empirical.py")" || {
    echo "Set STAGE1_SCRIPT to the formal empirical Stage-1 script." >&2
    exit 1
}
PHASE3_SCRIPT="$(resolve_script "${PHASE3_SCRIPT:-}" \
    "${SCRIPT_DIR}/confirm_offset_dual_channel_phase3.py")" || {
    echo "Set PHASE3_SCRIPT to the formal mechanism Phase-3 script." >&2
    exit 1
}
SUMMARY_SCRIPT="$(resolve_script "${SUMMARY_SCRIPT:-}" \
    "${SCRIPT_DIR}/summarize_supplementary_robustness.py")" || {
    echo "Summary script not found." >&2
    exit 1
}
REPORT_EXTRACTOR="$(resolve_script "${REPORT_EXTRACTOR:-}" \
    "${SCRIPT_DIR}/extract_supplementary_robustness_statistics.py")" || {
    echo "Numerical report extractor not found." >&2
    exit 1
}

for path in \
    "$STAGE1_ROOT" \
    "$FROZEN_MANIFEST" \
    "$MAIN_ROOT" \
    "$TIME_SHUFFLE_ROOT" \
    "$TAG_SUPPORT_ROOT" \
    "$MACRO_ROOT" \
    "$COORDINATE_SUMMARY_ROOT/empirical_coordinate_sensitivity_statistics.csv" \
    "$COORDINATE_SUMMARY_ROOT/empirical_coordinate_sensitivity_manifest.json"; do
    if [[ ! -e "$path" ]]; then
        echo "Required input not found: $path" >&2
        exit 1
    fi
done

mkdir -p "$ROBUSTNESS_ROOT/logs"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

EMPIRICAL_MANIFEST="$ROBUSTNESS_ROOT/empirical/metadata/empirical_robustness_manifest.json"
MODEL_MANIFEST="$ROBUSTNESS_ROOT/models/metadata/model_robustness_manifest.json"
REPRESENTATION_MANIFEST="$ROBUSTNESS_ROOT/representations/metadata/representation_robustness_manifest.json"

EMPIRICAL_COMMAND=(
    "$PYTHON_BIN" "$SCRIPT_DIR/run_empirical_user_robustness.py"
    --stage1-root "$STAGE1_ROOT"
    --stage1-script "$STAGE1_SCRIPT"
    --output-root "$ROBUSTNESS_ROOT/empirical"
    --bootstrap-replicates "$BOOTSTRAP_REPLICATES"
    --bootstrap-chunk "$BOOTSTRAP_CHUNK"
    --seed "$RANDOM_SEED"
)

MODEL_COMMAND=(
    "$PYTHON_BIN" "$SCRIPT_DIR/run_model_user_robustness.py"
    --stage1-root "$STAGE1_ROOT"
    --phase3-root "$PHASE3_ROOT"
    --frozen-manifest "$FROZEN_MANIFEST"
    --phase3-script "$PHASE3_SCRIPT"
    --main-root "$MAIN_ROOT"
    --time-shuffle-root "$TIME_SHUFFLE_ROOT"
    --tag-support-root "$TAG_SUPPORT_ROOT"
    --output-root "$ROBUSTNESS_ROOT/models"
    --bootstrap-replicates "$BOOTSTRAP_REPLICATES"
    --bootstrap-chunk "$BOOTSTRAP_CHUNK"
    --seed "$RANDOM_SEED"
)
if [[ -n "${PHASE1_SCRIPT:-}" ]]; then
    MODEL_COMMAND+=(--phase1-script "$PHASE1_SCRIPT")
fi

REPRESENTATION_COMMAND=(
    "$PYTHON_BIN" "$SCRIPT_DIR/run_representation_robustness.py"
    --stage1-root "$STAGE1_ROOT"
    --macro-root "$MACRO_ROOT"
    --output-root "$ROBUSTNESS_ROOT/representations"
    --bootstrap-replicates "$BOOTSTRAP_REPLICATES"
    --bootstrap-chunk "$BOOTSTRAP_CHUNK"
    --permutation-replicates "$PERMUTATION_REPLICATES"
    --seed "$RANDOM_SEED"
)

validate_json() {
    "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(1)
with path.open("r", encoding="utf-8") as handle:
    value = json.load(handle)
if not isinstance(value, dict):
    raise SystemExit(1)
PY
}

run_branch() {
    local name="$1"
    local manifest="$2"
    local log="$3"
    shift 3
    local command=("$@")
    mkdir -p "$(dirname "$manifest")" "$(dirname "$log")"
    if [[ "$FORCE_RERUN" != "1" ]] && validate_json "$manifest" 2>/dev/null; then
        echo "[$name] completed manifest found; skipping."
        return 0
    fi
    rm -f "$manifest"
    echo "[$name] started. Log: $log"
    if ! "${command[@]}" >"$log" 2>&1; then
        echo "[$name] failed. Last log lines:" >&2
        tail -n 120 "$log" >&2 || true
        return 1
    fi
    if ! validate_json "$manifest"; then
        echo "[$name] finished without a valid completion manifest: $manifest" >&2
        tail -n 120 "$log" >&2 || true
        return 1
    fi
    echo "[$name] completed. Manifest: $manifest"
}

run_empirical() {
    run_branch \
        "empirical" \
        "$EMPIRICAL_MANIFEST" \
        "$ROBUSTNESS_ROOT/logs/empirical.log" \
        "${EMPIRICAL_COMMAND[@]}"
}

run_models() {
    run_branch \
        "models" \
        "$MODEL_MANIFEST" \
        "$ROBUSTNESS_ROOT/logs/models.log" \
        "${MODEL_COMMAND[@]}"
}

run_representations() {
    run_branch \
        "representations" \
        "$REPRESENTATION_MANIFEST" \
        "$ROBUSTNESS_ROOT/logs/representations.log" \
        "${REPRESENTATION_COMMAND[@]}"
}

JOB_FUNCTIONS=(run_empirical run_models run_representations)
JOB_COUNT="${#JOB_FUNCTIONS[@]}"
if [[ "$MAX_PARALLEL" -eq 0 ]] || [[ "$MAX_PARALLEL" -gt "$JOB_COUNT" ]]; then
    EFFECTIVE_PARALLEL="$JOB_COUNT"
else
    EFFECTIVE_PARALLEL="$MAX_PARALLEL"
fi

active=0
failed=0
for job_function in "${JOB_FUNCTIONS[@]}"; do
    while [[ "$active" -ge "$EFFECTIVE_PARALLEL" ]]; do
        if ! wait -n; then
            failed=1
        fi
        active=$((active - 1))
    done
    "$job_function" &
    active=$((active + 1))
done
while [[ "$active" -gt 0 ]]; do
    if ! wait -n; then
        failed=1
    fi
    active=$((active - 1))
done
if [[ "$failed" -ne 0 ]]; then
    echo "One or more supplementary robustness branches failed." >&2
    exit 1
fi

for manifest in "$EMPIRICAL_MANIFEST" "$MODEL_MANIFEST" "$REPRESENTATION_MANIFEST"; do
    if ! validate_json "$manifest"; then
        echo "Required completion manifest is missing or invalid: $manifest" >&2
        exit 1
    fi
done

if [[ "$RUN_COLLECTION" == "1" ]]; then
    "$PYTHON_BIN" "$SUMMARY_SCRIPT" \
        --empirical-root "$ROBUSTNESS_ROOT/empirical" \
        --model-root "$ROBUSTNESS_ROOT/models" \
        --representation-root "$ROBUSTNESS_ROOT/representations" \
        --coordinate-summary-root "$COORDINATE_SUMMARY_ROOT" \
        --output-root "$ROBUSTNESS_ROOT/summary" \
        >"$ROBUSTNESS_ROOT/logs/summary.log" 2>&1 || {
        echo "Collection failed. Last log lines:" >&2
        tail -n 120 "$ROBUSTNESS_ROOT/logs/summary.log" >&2 || true
        exit 1
    }
fi

if [[ "$RUN_NUMERICAL_REPORT" == "1" ]]; then
    "$PYTHON_BIN" "$REPORT_EXTRACTOR" \
        --robustness-root "$ROBUSTNESS_ROOT" \
        --coordinate-summary-root "$COORDINATE_SUMMARY_ROOT" \
        --output-dir "$ROBUSTNESS_ROOT/summary" \
        >"$ROBUSTNESS_ROOT/logs/numerical_report.log" 2>&1 || {
        echo "Numerical report extraction failed. Last log lines:" >&2
        tail -n 120 "$ROBUSTNESS_ROOT/logs/numerical_report.log" >&2 || true
        exit 1
    }
fi

echo "Completed supplementary robustness analyses."
if [[ "$RUN_COLLECTION" == "1" ]]; then
    echo "Collection report: $ROBUSTNESS_ROOT/summary/reports/supplementary_robustness_report.md"
fi
if [[ "$RUN_NUMERICAL_REPORT" == "1" ]]; then
    echo "Numerical report: $ROBUSTNESS_ROOT/summary/supplementary_robustness_numerical_report.md"
fi
