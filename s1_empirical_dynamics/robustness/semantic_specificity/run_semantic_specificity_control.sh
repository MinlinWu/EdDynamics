#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/datasets/KT4/outputs_KT4}"
STAGE1_ROOT="${STAGE1_ROOT:-${OUTPUT_BASE}/stage1}"
FORMAL_NULL_ROOT="${FORMAL_NULL_ROOT:-${OUTPUT_BASE}/stage1_construction_matched_null}"
FORMAL_CONFIRM_NULL_ROOT="${FORMAL_CONFIRM_NULL_ROOT:-${OUTPUT_BASE}/stage1_construction_matched_null_confirm}"
RESULT_ROOT="${RESULT_ROOT:-${OUTPUT_BASE}/semantic_specificity_control}"
COORDINATE_ROOT="${COORDINATE_ROOT:-${RESULT_ROOT}/coordinate}"
NULL_ROOT="${NULL_ROOT:-${RESULT_ROOT}/null}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${RESULT_ROOT}/protocol}"
PROTOCOL_FREEZE="${PROTOCOL_FREEZE:-${PROTOCOL_ROOT}/semantic_specificity_protocol_freeze.json}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${RESULT_ROOT}/summary}"
LOG_ROOT="${LOG_ROOT:-${RESULT_ROOT}/logs}"
REPLICATES="${REPLICATES:-100}"
RUN_CONFIRMATION="${RUN_CONFIRMATION:-1}"
OVERWRITE="${OVERWRITE:-0}"

resolve_script() {
    local explicit="$1"
    shift
    if [[ -n "$explicit" ]]; then
        [[ -f "$explicit" ]] || { echo "Required script not found: $explicit" >&2; exit 1; }
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

require_table() {
    local base="$1"
    if [[ ! -f "${base}.parquet" && ! -f "${base}.csv.gz" && ! -f "${base}.csv" ]]; then
        echo "Required table not found: ${base}.[parquet|csv.gz|csv]" >&2
        exit 1
    fi
}

run_logged() {
    local label="$1"
    local log_path="$2"
    shift 2
    echo "[$label] starting"
    if ! "$@" >"$log_path" 2>&1; then
        echo "[$label] failed. Last 160 log lines:" >&2
        tail -n 160 "$log_path" >&2 || true
        exit 1
    fi
    echo "[$label] complete"
}

STAGE1_SCRIPT="$(resolve_script "${STAGE1_SCRIPT:-}" \
    "${SCRIPT_DIR}/build_effective_dynamics_kt4_stage1_empirical.py")" || {
    echo "Set STAGE1_SCRIPT to the formal empirical Stage-1 implementation." >&2
    exit 1
}

CONSTRUCTION_NULL_SCRIPT="$(resolve_script "${CONSTRUCTION_NULL_SCRIPT:-}" \
    "${SCRIPT_DIR}/run_construction_matched_null.py")" || {
    echo "Set CONSTRUCTION_NULL_SCRIPT to the formal construction-matched-null implementation." >&2
    exit 1
}

COMMON_SCRIPT="${COMMON_SCRIPT:-${SCRIPT_DIR}/semantic_specificity_common.py}"
COORDINATE_SCRIPT="${COORDINATE_SCRIPT:-${SCRIPT_DIR}/run_nonsemantic_coordinate_control.py}"
NULL_SCRIPT="${NULL_SCRIPT:-${SCRIPT_DIR}/run_nonsemantic_construction_null.py}"
FREEZE_SCRIPT="${FREEZE_SCRIPT:-${SCRIPT_DIR}/freeze_semantic_specificity_protocol.py}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-${SCRIPT_DIR}/summarize_semantic_specificity_control.py}"

for path in \
    "$STAGE1_ROOT" \
    "$FORMAL_NULL_ROOT" \
    "$STAGE1_SCRIPT" \
    "$CONSTRUCTION_NULL_SCRIPT" \
    "$COMMON_SCRIPT" \
    "$COORDINATE_SCRIPT" \
    "$NULL_SCRIPT" \
    "$FREEZE_SCRIPT" \
    "$SUMMARY_SCRIPT" \
    "$FORMAL_NULL_ROOT/metadata/A_val_construction_null_manifest.json"; do
    [[ -e "$path" ]] || { echo "Required input not found: $path" >&2; exit 1; }
done

for split in A_train A_val B_confirm; do
    require_table "$STAGE1_ROOT/dynamics/student_dynamics_panel_core_${split}"
done

if [[ "$RUN_CONFIRMATION" == "1" ]]; then
    [[ -f "$FORMAL_CONFIRM_NULL_ROOT/metadata/B_confirm_construction_null_manifest.json" ]] || {
        echo "Missing formal B_confirm construction-null manifest: $FORMAL_CONFIRM_NULL_ROOT/metadata/B_confirm_construction_null_manifest.json" >&2
        exit 1
    }
fi

if ! [[ "$REPLICATES" =~ ^[0-9]+$ ]] || (( REPLICATES < 20 )); then
    echo "REPLICATES must be an integer of at least 20; 100 matches the formal null." >&2
    exit 1
fi
if [[ "$RUN_CONFIRMATION" != "0" && "$RUN_CONFIRMATION" != "1" ]]; then
    echo "RUN_CONFIRMATION must be 0 or 1." >&2
    exit 1
fi
if [[ "$OVERWRITE" != "0" && "$OVERWRITE" != "1" ]]; then
    echo "OVERWRITE must be 0 or 1." >&2
    exit 1
fi

if [[ "$OVERWRITE" == "1" ]]; then
    rm -rf "$RESULT_ROOT"
fi
mkdir -p "$LOG_ROOT" "$NULL_ROOT" "$PROTOCOL_ROOT" "$SUMMARY_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export EDNET_STAGE1_TAU_RESPONSE_DAYS=10.0
export EDNET_STAGE1_TAU_ACTIVITY_DAYS=10.0
export EDNET_STAGE1_RESPONSE_DURATION_HALF_SAT_MIN=3.0
export EDNET_STAGE1_EXPLANATION_HALF_SAT_MIN=2.5
export EDNET_STAGE1_LECTURE_HALF_SAT_MIN=4.0
export EDNET_STAGE1_IDLE_HALF_SAT_DAYS=1.0
export EDNET_STAGE1_SIGNED_GRID_N=41
export EDNET_STAGE1_MIN_STATE_BIN_COUNT=50
export EDNET_STAGE1_MIN_DRIFT_BIN_COUNT=30
export EDNET_STAGE1_MIN_CELL_USERS=5
export EDNET_STAGE1_CONVERGENCE_SPEED_QUANTILE=0.60
export EDNET_STAGE1_CONVERGENCE_NEGATIVE_DIVERGENCE_QUANTILE=0.80
export EDNET_STAGE1_CONVERGENCE_RATIO_QUANTILE=0.60
export EDNET_STAGE1_CONVERGENCE_MIN_CELLS=4
export EDNET_STAGE1_CONVERGENCE_SHELL_RADIUS=0.35

"$PYTHON_BIN" -m py_compile \
    "$COMMON_SCRIPT" \
    "$COORDINATE_SCRIPT" \
    "$NULL_SCRIPT" \
    "$FREEZE_SCRIPT" \
    "$SUMMARY_SCRIPT" \
    "$STAGE1_SCRIPT" \
    "$CONSTRUCTION_NULL_SCRIPT"

"$PYTHON_BIN" "$COORDINATE_SCRIPT" --stage1-script "$STAGE1_SCRIPT" --self-test
"$PYTHON_BIN" "$NULL_SCRIPT" --self-test
"$PYTHON_BIN" "$FREEZE_SCRIPT" --self-test
"$PYTHON_BIN" "$SUMMARY_SCRIPT" --self-test

run_logged "alignment-free coordinate A_train/A_val" "$LOG_ROOT/alignment_free_coordinate_preconfirmation.log" \
    "$PYTHON_BIN" "$COORDINATE_SCRIPT" \
    --stage1-root "$STAGE1_ROOT" \
    --stage1-script "$STAGE1_SCRIPT" \
    --construction-null-root "$FORMAL_NULL_ROOT" \
    --output-root "$COORDINATE_ROOT" \
    --skip-confirmation

test -f "$COORDINATE_ROOT/metadata/semantic_specificity_coordinate_manifest.json"

run_logged "alignment-free construction null A_val" "$LOG_ROOT/alignment_free_null_A_val.log" \
    "$PYTHON_BIN" "$NULL_SCRIPT" \
    --stage1-root "$STAGE1_ROOT" \
    --stage1-script "$STAGE1_SCRIPT" \
    --construction-null-script "$CONSTRUCTION_NULL_SCRIPT" \
    --construction-null-root "$FORMAL_NULL_ROOT" \
    --coordinate-output-root "$COORDINATE_ROOT" \
    --output-root "$NULL_ROOT" \
    --analysis-split A_val \
    --replicates "$REPLICATES" \
    --seed 42

test -f "$NULL_ROOT/metadata/A_val_nonsemantic_construction_null_manifest.json"

run_logged "protocol freeze" "$LOG_ROOT/alignment_specificity_protocol_freeze.log" \
    "$PYTHON_BIN" "$FREEZE_SCRIPT" \
    --coordinate-output-root "$COORDINATE_ROOT" \
    --nonsemantic-null-root "$NULL_ROOT" \
    --formal-null-root "$FORMAL_NULL_ROOT" \
    --stage1-script "$STAGE1_SCRIPT" \
    --construction-null-script "$CONSTRUCTION_NULL_SCRIPT" \
    --common-script "$COMMON_SCRIPT" \
    --coordinate-script "$COORDINATE_SCRIPT" \
    --null-script "$NULL_SCRIPT" \
    --summary-script "$SUMMARY_SCRIPT" \
    --output-path "$PROTOCOL_FREEZE"

test -f "$PROTOCOL_FREEZE"

if [[ "$RUN_CONFIRMATION" == "1" ]]; then
    run_logged "alignment-free coordinate B_confirm" "$LOG_ROOT/alignment_free_coordinate_confirmation.log" \
        "$PYTHON_BIN" "$COORDINATE_SCRIPT" \
        --stage1-root "$STAGE1_ROOT" \
        --stage1-script "$STAGE1_SCRIPT" \
        --construction-null-root "$FORMAL_NULL_ROOT" \
        --output-root "$COORDINATE_ROOT" \
        --append-confirmation \
        --protocol-freeze "$PROTOCOL_FREEZE"

    require_table "$COORDINATE_ROOT/stage1/dynamics/coordinate_analysis/MR_PhiAI/B_confirm_publication_field_grid_output_only"

    run_logged "alignment-free construction null B_confirm" "$LOG_ROOT/alignment_free_null_B_confirm.log" \
        "$PYTHON_BIN" "$NULL_SCRIPT" \
        --stage1-root "$STAGE1_ROOT" \
        --stage1-script "$STAGE1_SCRIPT" \
        --construction-null-script "$CONSTRUCTION_NULL_SCRIPT" \
        --construction-null-root "$FORMAL_CONFIRM_NULL_ROOT" \
        --validation-construction-null-root "$FORMAL_NULL_ROOT" \
        --coordinate-output-root "$COORDINATE_ROOT" \
        --output-root "$NULL_ROOT" \
        --analysis-split B_confirm \
        --confirmation-output-only \
        --protocol-freeze "$PROTOCOL_FREEZE" \
        --replicates "$REPLICATES" \
        --seed 42

    test -f "$NULL_ROOT/metadata/B_confirm_nonsemantic_construction_null_manifest.json"

    run_logged "alignment-specificity summary" "$LOG_ROOT/alignment_specificity_summary.log" \
        "$PYTHON_BIN" "$SUMMARY_SCRIPT" \
        --stage1-root "$STAGE1_ROOT" \
        --formal-null-root "$FORMAL_NULL_ROOT" \
        --formal-confirm-null-root "$FORMAL_CONFIRM_NULL_ROOT" \
        --coordinate-output-root "$COORDINATE_ROOT" \
        --nonsemantic-null-root "$NULL_ROOT" \
        --protocol-freeze "$PROTOCOL_FREEZE" \
        --output-root "$SUMMARY_ROOT"

    test -f "$SUMMARY_ROOT/semantic_specificity_summary_manifest.json"
fi

echo "Alignment-specificity comparison completed."
echo "Protocol freeze: $PROTOCOL_FREEZE"
echo "Coordinate manifest: $COORDINATE_ROOT/metadata/semantic_specificity_coordinate_manifest.json"
echo "Null manifests: $NULL_ROOT/metadata"
if [[ "$RUN_CONFIRMATION" == "1" ]]; then
    echo "Numerical report: $SUMMARY_ROOT/semantic_specificity_control_report.md"
else
    echo "Confirmation was not run; A_val remains the only new-control result."
fi
