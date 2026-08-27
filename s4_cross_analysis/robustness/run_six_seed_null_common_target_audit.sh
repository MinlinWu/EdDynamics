#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-/data/datasets/KT4/outputs_KT4}"
STAGE1_ROOT="${STAGE1_ROOT:-${OUTPUTS_ROOT}/stage1}"
CONSTRUCTION_NULL_ROOT="${CONSTRUCTION_NULL_ROOT:-${OUTPUTS_ROOT}/stage1_construction_matched_null}"
FROZEN_MECHANISM_MANIFEST="${FROZEN_MECHANISM_MANIFEST:-${OUTPUTS_ROOT}/stage2_phase2_freeze/metadata/phase2_frozen_model_manifest.json}"
NULL_RECOVERY_ROOT="${NULL_RECOVERY_ROOT:-${OUTPUTS_ROOT}/six_seed_null_referenced_recovery}"
RESULT_ROOT="${RESULT_ROOT:-${OUTPUTS_ROOT}/six_seed_null_common_target_audit}"
HEADLINE_BOOTSTRAP_ROOT="${HEADLINE_BOOTSTRAP_ROOT:-${OUTPUTS_ROOT}/frozen_headline_learner_cluster_uncertainty}"
REQUIRE_HEADLINE_BOOTSTRAP="${REQUIRE_HEADLINE_BOOTSTRAP:-0}"
OVERWRITE="${OVERWRITE:-0}"
OVERWRITE_NULL_RECOVERY="${OVERWRITE_NULL_RECOVERY:-0}"

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

find_table() {
    local base="$1"
    local extension
    for extension in .parquet .csv.gz .csv; do
        if [[ -f "${base}${extension}" ]]; then
            printf '%s\n' "${base}${extension}"
            return
        fi
    done
    return 1
}

resolve_event_root() {
    local seed="$1"
    local variable="EVENT_SSL_SEED${seed}_ROOT"
    local explicit="${!variable:-}"
    if [[ -n "$explicit" ]]; then
        printf '%s\n' "${explicit%/}"
        return
    fi
    local candidates=()
    if [[ "$seed" == "42" ]]; then
        candidates+=(
            "${OUTPUTS_ROOT}/stage4_event_ssl/evaluation_predictive_state"
            "${OUTPUTS_ROOT}/random_seed_experiments/seed_42/stage4_event_ssl/evaluation_predictive_state"
            "${OUTPUTS_ROOT}/random_seed_experiments/seed42/stage4_event_ssl/evaluation_predictive_state"
        )
    else
        candidates+=(
            "${OUTPUTS_ROOT}/random_seed_experiments/seed_${seed}/stage4_event_ssl/evaluation_predictive_state"
            "${OUTPUTS_ROOT}/random_seed_experiments/seed${seed}/stage4_event_ssl/evaluation_predictive_state"
        )
    fi
    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -d "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    echo "Missing Event-SSL evaluation root for seed ${seed}. Set ${variable}." >&2
    exit 1
}

NULL_EVALUATOR="$(resolve_script "${NULL_EVALUATOR:-}" \
    "${SCRIPT_DIR}/evaluate_null_referenced_downstream_recovery.py")" || {
    echo "Set NULL_EVALUATOR to evaluate_null_referenced_downstream_recovery.py" >&2
    exit 1
}
AUDIT_SCRIPT="$(resolve_script "${AUDIT_SCRIPT:-}" "${SCRIPT_DIR}/run_six_seed_null_common_target_audit.py")" || exit 1
REPORT_SCRIPT="$(resolve_script "${REPORT_SCRIPT:-}" "${SCRIPT_DIR}/extract_six_seed_null_common_target_audit_report.py")" || exit 1
STAGE1_SCRIPT="$(resolve_script "${STAGE1_SCRIPT:-}" \
    "${SCRIPT_DIR}/build_effective_dynamics_kt4_stage1_empirical.py")" || exit 1
CONSTRUCTION_NULL_SCRIPT="$(resolve_script "${CONSTRUCTION_NULL_SCRIPT:-}" \
    "${SCRIPT_DIR}/run_construction_matched_null.py")" || exit 1
MECHANISM_CONFIRM_SCRIPT="$(resolve_script "${MECHANISM_CONFIRM_SCRIPT:-}" \
    "${SCRIPT_DIR}/confirm_offset_dual_channel_phase3.py")" || exit 1
MECHANISM_PHASE1_SCRIPT="$(resolve_script "${MECHANISM_PHASE1_SCRIPT:-}" \
    "${SCRIPT_DIR}/tune_offset_dual_channel_phase1.py")" || exit 1

SEEDS=(42 2026 666 606 37 4669)
EVENT_ARGS=()
for seed in "${SEEDS[@]}"; do
    root="$(resolve_event_root "$seed")"
    [[ -f "${root}/metadata/stage4_event_ssl_evaluation_manifest.json" ]] || {
        echo "Missing evaluation manifest: ${root}/metadata/stage4_event_ssl_evaluation_manifest.json" >&2
        exit 1
    }
    find_table "${root}/predictions/stage4_event_ssl_predictions_A_val" >/dev/null || {
        echo "Missing A_val prediction table in ${root}" >&2
        exit 1
    }
    find_table "${root}/predictions/stage4_event_ssl_predictions_B_confirm" >/dev/null || {
        echo "Missing B_confirm prediction table in ${root}" >&2
        exit 1
    }
    EVENT_ARGS+=(--event-ssl-root "event_ssl_seed${seed}=${root}")
done

"${PYTHON_BIN}" -m py_compile "$NULL_EVALUATOR" "$AUDIT_SCRIPT" "$REPORT_SCRIPT"
"${PYTHON_BIN}" "$NULL_EVALUATOR" --construction-null-script "$CONSTRUCTION_NULL_SCRIPT" --self-test
"${PYTHON_BIN}" "$AUDIT_SCRIPT" --self-test
"${PYTHON_BIN}" "$REPORT_SCRIPT" --self-test

if [[ "$OVERWRITE_NULL_RECOVERY" == "1" ]]; then
    rm -rf "$NULL_RECOVERY_ROOT"
fi
if [[ "$OVERWRITE" == "1" ]]; then
    rm -rf "$RESULT_ROOT"
fi

if [[ ! -f "${NULL_RECOVERY_ROOT}/metadata/null_referenced_recovery_manifest.json" ]]; then
    "${PYTHON_BIN}" "$NULL_EVALUATOR" \
        --stage1-root "$STAGE1_ROOT" \
        --stage1-script "$STAGE1_SCRIPT" \
        --construction-null-script "$CONSTRUCTION_NULL_SCRIPT" \
        --construction-null-output-root "$CONSTRUCTION_NULL_ROOT" \
        --require-existing-construction-null-output \
        --mechanism-confirm-script "$MECHANISM_CONFIRM_SCRIPT" \
        --mechanism-phase1-script "$MECHANISM_PHASE1_SCRIPT" \
        --frozen-mechanism-manifest "$FROZEN_MECHANISM_MANIFEST" \
        --require-mechanism-manifest-checksum \
        "${EVENT_ARGS[@]}" \
        --require-event-ssl-manifest \
        --primary-event-ssl-label event_ssl_seed42 \
        --splits A_val B_confirm \
        --confirmation-output-only \
        --output-root "$NULL_RECOVERY_ROOT" \
        --bootstrap-replicates 0
else
    echo "Using existing six-seed null-recovery output: $NULL_RECOVERY_ROOT"
fi

AUDIT_ARGS=(
    --null-recovery-root "$NULL_RECOVERY_ROOT"
    --output-root "$RESULT_ROOT"
    --headline-bootstrap-root "$HEADLINE_BOOTSTRAP_ROOT"
)
if [[ "$REQUIRE_HEADLINE_BOOTSTRAP" == "1" ]]; then
    AUDIT_ARGS+=(--require-headline-bootstrap)
fi

"${PYTHON_BIN}" "$AUDIT_SCRIPT" "${AUDIT_ARGS[@]}"
"${PYTHON_BIN}" "$REPORT_SCRIPT" --result-root "$RESULT_ROOT"

echo "Audit completed: $RESULT_ROOT"
echo "Numerical report: $RESULT_ROOT/numeric_report/six_seed_null_common_target_audit_report.md"
