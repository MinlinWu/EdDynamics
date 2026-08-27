#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-/data/datasets/KT4/outputs_KT4}"
STAGE1_ROOT="${STAGE1_ROOT:-${OUTPUTS_ROOT}/stage1}"
PHASE3_ROOT="${PHASE3_ROOT:-${OUTPUTS_ROOT}/stage2_phase3_confirm}"
FROZEN_MANIFEST="${FROZEN_MANIFEST:-${OUTPUTS_ROOT}/stage2_phase2_freeze/metadata/phase2_frozen_model_manifest.json}"
EVENT_SSL_ROOT="${EVENT_SSL_ROOT:-${OUTPUTS_ROOT}/stage4_event_ssl/evaluation_predictive_state}"
RESULT_ROOT="${RESULT_ROOT:-${OUTPUTS_ROOT}/frozen_headline_learner_cluster_uncertainty}"
MECHANISM_EXPORT_ROOT="${MECHANISM_EXPORT_ROOT:-${OUTPUTS_ROOT}/frozen_headline_learner_cluster_uncertainty_mechanism_export}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-2000}"
BOOTSTRAP_BATCH_SIZE="${BOOTSTRAP_BATCH_SIZE:-20}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260806}"
CI_LEVEL="${CI_LEVEL:-0.95}"
MINIMUM_JOIN_FRACTION="${MINIMUM_JOIN_FRACTION:-0.999}"
MINIMUM_FINITE_FRACTION="${MINIMUM_FINITE_FRACTION:-0.99}"
EXPECTED_COMMON_ROWS="${EXPECTED_COMMON_ROWS:-3233208}"
EXPECTED_COMMON_USERS="${EXPECTED_COMMON_USERS:-56195}"
OVERWRITE="${OVERWRITE:-0}"

resolve_script() {
    local explicit="$1"
    shift
    if [[ -n "$explicit" ]]; then
        if [[ ! -f "$explicit" ]]; then
            echo "Required script not found: $explicit" >&2
            exit 1
        fi
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

ANALYSIS_SCRIPT="$(resolve_script "${ANALYSIS_SCRIPT:-}" \
    "${SCRIPT_DIR}/run_frozen_headline_learner_cluster_uncertainty.py")" || {
    echo "Analysis script is missing." >&2
    exit 1
}

EXTRACT_SCRIPT="$(resolve_script "${EXTRACT_SCRIPT:-}" \
    "${SCRIPT_DIR}/extract_frozen_headline_learner_cluster_uncertainty_report.py")" || {
    echo "Numerical-report extractor is missing." >&2
    exit 1
}

STAGE1_SCRIPT="$(resolve_script "${STAGE1_SCRIPT:-}" \
    "${SCRIPT_DIR}/build_effective_dynamics_kt4_stage1_empirical.py" \
    "${SCRIPT_DIR}/build_effective_dynamics_kt4_stage1_empirical(20260806-074719).py" \
    "${SCRIPT_DIR}/build_effective_dynamics_kt4_stage1_empirical(17).py")" || {
    echo "Set STAGE1_SCRIPT to the formal Stage-1 implementation." >&2
    exit 1
}

PHASE3_SCRIPT="$(resolve_script "${PHASE3_SCRIPT:-}" \
    "${SCRIPT_DIR}/confirm_offset_dual_channel_phase3.py" \
    "${SCRIPT_DIR}/confirm_offset_dual_channel_phase3(8).py" \
    "${SCRIPT_DIR}/confirm_offset_dual_channel_phase3(7).py")" || {
    echo "Set PHASE3_SCRIPT to the formal Phase-3 implementation." >&2
    exit 1
}

PHASE1_SCRIPT="$(resolve_script "${PHASE1_SCRIPT:-}" \
    "${SCRIPT_DIR}/tune_offset_dual_channel_phase1.py" \
    "${SCRIPT_DIR}/tune_offset_dual_channel_phase1(20260806-074735).py" \
    "${SCRIPT_DIR}/tune_offset_dual_channel_phase1(11).py")" || {
    echo "Set PHASE1_SCRIPT to the frozen Phase-1 implementation used by Phase 3." >&2
    exit 1
}

EVENT_EVALUATE_SCRIPT="$(resolve_script "${EVENT_EVALUATE_SCRIPT:-}" \
    "${SCRIPT_DIR}/evaluate_event_ssl_structure.py" \
    "${SCRIPT_DIR}/evaluate_event_ssl_structure(20260806-074712).py" \
    "${SCRIPT_DIR}/evaluate_event_ssl_structure(16).py")" || {
    echo "Set EVENT_EVALUATE_SCRIPT to the formal Event-SSL evaluator." >&2
    exit 1
}

for integer_value in "$BOOTSTRAP_REPLICATES" "$BOOTSTRAP_BATCH_SIZE" "$BOOTSTRAP_SEED" "$EXPECTED_COMMON_ROWS" "$EXPECTED_COMMON_USERS"; do
    if ! [[ "$integer_value" =~ ^[0-9]+$ ]]; then
        echo "Integer-valued settings must contain only digits: $integer_value" >&2
        exit 1
    fi
done
if (( BOOTSTRAP_REPLICATES < 1000 )); then
    echo "BOOTSTRAP_REPLICATES must be at least 1000; the publication default is 2000." >&2
    exit 1
fi
if (( BOOTSTRAP_BATCH_SIZE < 1 )); then
    echo "BOOTSTRAP_BATCH_SIZE must be positive." >&2
    exit 1
fi
if [[ "$OVERWRITE" != "0" && "$OVERWRITE" != "1" ]]; then
    echo "OVERWRITE must be 0 or 1." >&2
    exit 1
fi

for required in \
    "$STAGE1_ROOT" \
    "$FROZEN_MANIFEST" \
    "$EVENT_SSL_ROOT/metadata/stage4_event_ssl_evaluation_manifest.json" \
    "$STAGE1_SCRIPT" \
    "$PHASE3_SCRIPT" \
    "$PHASE1_SCRIPT" \
    "$EVENT_EVALUATE_SCRIPT" \
    "$ANALYSIS_SCRIPT" \
    "$EXTRACT_SCRIPT"; do
    if [[ ! -e "$required" ]]; then
        echo "Required input not found: $required" >&2
        exit 1
    fi
done

find_prediction_table() {
    local root="$1"
    local base="${root}/tables/phase3_B_confirm_full_predictions"
    local suffix
    for suffix in parquet csv.gz csv; do
        if [[ -f "${base}.${suffix}" ]]; then
            printf '%s\n' "${base}.${suffix}"
            return 0
        fi
    done
    return 1
}

manifest_declares_prediction() {
    local manifest="$1"
    local prediction="$2"
    local phase3_script="$3"
    "$PYTHON_BIN" - "$manifest" "$prediction" "$phase3_script" <<'PY'
import hashlib
import json
import pathlib
import sys
manifest_path = pathlib.Path(sys.argv[1])
prediction_path = pathlib.Path(sys.argv[2]).resolve()
phase3_script = pathlib.Path(sys.argv[3]).resolve()
if not manifest_path.exists() or not prediction_path.exists():
    raise SystemExit(1)
with manifest_path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
if payload.get("confirmation_status") != "completed_output_only":
    raise SystemExit(1)
if payload.get("confirm_split") != "B_confirm":
    raise SystemExit(1)
if payload.get("guardrails", {}).get("B_confirm_used_for_update", False):
    raise SystemExit(1)
if not phase3_script.exists():
    raise SystemExit(1)
digest = hashlib.sha256()
with phase3_script.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
if payload.get("phase3_script_sha256") != digest.hexdigest():
    raise SystemExit(1)
declared = payload.get("outputs", {}).get("full_prediction_table")
if not declared:
    raise SystemExit(1)
declared_path = pathlib.Path(str(declared))
if declared_path.resolve() == prediction_path or declared_path.name == prediction_path.name:
    print(str(prediction_path))
    raise SystemExit(0)
raise SystemExit(1)
PY
}

MECHANISM_PREDICTIONS=""
MECHANISM_MANIFEST=""
FORMAL_PHASE3_MANIFEST="${PHASE3_ROOT}/metadata/phase3_confirmation_manifest.json"
if candidate="$(find_prediction_table "$PHASE3_ROOT" 2>/dev/null)"; then
    if manifest_declares_prediction "$FORMAL_PHASE3_MANIFEST" "$candidate" "$PHASE3_SCRIPT" >/dev/null 2>&1; then
        MECHANISM_PREDICTIONS="$candidate"
        MECHANISM_MANIFEST="$FORMAL_PHASE3_MANIFEST"
    fi
fi

if [[ -z "$MECHANISM_PREDICTIONS" ]]; then
    EXPORT_MANIFEST="${MECHANISM_EXPORT_ROOT}/metadata/phase3_confirmation_manifest.json"
    if candidate="$(find_prediction_table "$MECHANISM_EXPORT_ROOT" 2>/dev/null)"; then
        if manifest_declares_prediction "$EXPORT_MANIFEST" "$candidate" "$PHASE3_SCRIPT" >/dev/null 2>&1; then
            MECHANISM_PREDICTIONS="$candidate"
            MECHANISM_MANIFEST="$EXPORT_MANIFEST"
        fi
    fi
fi

if [[ -z "$MECHANISM_PREDICTIONS" ]]; then
    if [[ -d "$MECHANISM_EXPORT_ROOT" ]] && [[ -n "$(find "$MECHANISM_EXPORT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        if [[ "$OVERWRITE" == "1" ]]; then
            rm -rf "$MECHANISM_EXPORT_ROOT"
        else
            echo "Mechanism export root is incomplete or inconsistent: $MECHANISM_EXPORT_ROOT" >&2
            echo "Set OVERWRITE=1 to rebuild the isolated frozen-output export." >&2
            exit 1
        fi
    fi
    mkdir -p "$MECHANISM_EXPORT_ROOT"
    "$PYTHON_BIN" "$PHASE3_SCRIPT" \
        --frozen-manifest "$FROZEN_MANIFEST" \
        --phase1-script "$PHASE1_SCRIPT" \
        --stage1-root "$STAGE1_ROOT" \
        --confirm-split B_confirm \
        --output-root "$MECHANISM_EXPORT_ROOT" \
        --write-full-predictions \
        --require-manifest-checksum
    MECHANISM_PREDICTIONS="$(find_prediction_table "$MECHANISM_EXPORT_ROOT")"
    MECHANISM_MANIFEST="${MECHANISM_EXPORT_ROOT}/metadata/phase3_confirmation_manifest.json"
    manifest_declares_prediction "$MECHANISM_MANIFEST" "$MECHANISM_PREDICTIONS" "$PHASE3_SCRIPT" >/dev/null
fi

"$PYTHON_BIN" -m py_compile "$ANALYSIS_SCRIPT" "$EXTRACT_SCRIPT"
"$PYTHON_BIN" "$ANALYSIS_SCRIPT" --self-test
"$PYTHON_BIN" "$EXTRACT_SCRIPT" --self-test

ANALYSIS_ARGS=(
    --stage1-root "$STAGE1_ROOT"
    --stage1-script "$STAGE1_SCRIPT"
    --mechanism-predictions "$MECHANISM_PREDICTIONS"
    --mechanism-manifest "$MECHANISM_MANIFEST"
    --mechanism-confirm-script "$PHASE3_SCRIPT"
    --event-ssl-root "$EVENT_SSL_ROOT"
    --event-ssl-evaluate-script "$EVENT_EVALUATE_SCRIPT"
    --output-root "$RESULT_ROOT"
    --bootstrap-replicates "$BOOTSTRAP_REPLICATES"
    --bootstrap-batch-size "$BOOTSTRAP_BATCH_SIZE"
    --bootstrap-seed "$BOOTSTRAP_SEED"
    --ci-level "$CI_LEVEL"
    --minimum-join-fraction "$MINIMUM_JOIN_FRACTION"
    --minimum-finite-fraction "$MINIMUM_FINITE_FRACTION"
    --expected-common-rows "$EXPECTED_COMMON_ROWS"
    --expected-common-users "$EXPECTED_COMMON_USERS"
)
if [[ "$OVERWRITE" == "1" ]]; then
    ANALYSIS_ARGS+=(--overwrite)
fi

"$PYTHON_BIN" "$ANALYSIS_SCRIPT" "${ANALYSIS_ARGS[@]}"

REPORT_ARGS=(--result-root "$RESULT_ROOT")
if [[ "$OVERWRITE" == "1" ]]; then
    REPORT_ARGS+=(--overwrite)
fi
"$PYTHON_BIN" "$EXTRACT_SCRIPT" "${REPORT_ARGS[@]}"

echo "Analysis completed: $RESULT_ROOT"
echo "Numerical report: $RESULT_ROOT/numeric_report/frozen_headline_learner_cluster_uncertainty_report.md"
