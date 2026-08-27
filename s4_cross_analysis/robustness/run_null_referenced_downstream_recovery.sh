#!/usr/bin/env bash
set -euo pipefail

# Override these environment variables when the formal publication paths differ.
CODE_ROOT="${CODE_ROOT:-$(pwd)}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-/data/datasets/KT4/outputs_KT4}"
STAGE1_ROOT="${STAGE1_ROOT:-${OUTPUTS_ROOT}/stage1}"
CONSTRUCTION_NULL_ROOT="${CONSTRUCTION_NULL_ROOT:-${OUTPUTS_ROOT}/stage1_construction_matched_null}"
FROZEN_MECHANISM_MANIFEST="${FROZEN_MECHANISM_MANIFEST:-${OUTPUTS_ROOT}/stage2_phase2_freeze/metadata/phase2_frozen_model_manifest.json}"
RESULT_ROOT="${RESULT_ROOT:-${OUTPUTS_ROOT}/null_referenced_downstream_recovery}"

resolve_code_file() {
  local formal_name="$1"
  local uploaded_name="$2"
  if [[ -f "${CODE_ROOT}/${formal_name}" ]]; then
    printf '%s\n' "${CODE_ROOT}/${formal_name}"
  elif [[ -f "${CODE_ROOT}/${uploaded_name}" ]]; then
    printf '%s\n' "${CODE_ROOT}/${uploaded_name}"
  else
    printf '%s\n' "${CODE_ROOT}/${formal_name}"
  fi
}

ANALYSIS_SCRIPT="${ANALYSIS_SCRIPT:-$(resolve_code_file evaluate_null_referenced_downstream_recovery.py evaluate_null_referenced_downstream_recovery.py)}"
STAGE1_SCRIPT="${STAGE1_SCRIPT:-$(resolve_code_file build_effective_dynamics_kt4_stage1_empirical.py 'build_effective_dynamics_kt4_stage1_empirical.py')}"
CONSTRUCTION_NULL_SCRIPT="${CONSTRUCTION_NULL_SCRIPT:-$(resolve_code_file run_construction_matched_null.py 'run_construction_matched_null.py')}"
MECHANISM_CONFIRM_SCRIPT="${MECHANISM_CONFIRM_SCRIPT:-$(resolve_code_file confirm_offset_dual_channel_phase3.py 'confirm_offset_dual_channel_phase3.py')}"
MECHANISM_PHASE1_SCRIPT="${MECHANISM_PHASE1_SCRIPT:-$(resolve_code_file tune_offset_dual_channel_phase1.py 'tune_offset_dual_channel_phase1.py')}"

# Each root must contain:
#   predictions/stage4_event_ssl_predictions_A_val.[parquet|csv.gz|csv]
#   predictions/stage4_event_ssl_predictions_B_confirm.[parquet|csv.gz|csv]
#   metadata/stage4_event_ssl_evaluation_manifest.json
EVENT_SSL_ARGS=(
  --event-ssl-root "event_ssl_seed42=${EVENT_SSL_SEED42_ROOT:-${OUTPUTS_ROOT}/stage4_event_ssl/evaluation_predictive_state}"
)

# Add the five extra frozen-seed evaluation roots when available. Point metrics
# will be written for every supplied seed; the positive user-level multiplier bootstrap remains on
# the declared primary seed and the frozen mechanism.
# EVENT_SSL_ARGS+=(--event-ssl-root "event_ssl_seed2026=${OUTPUTS_ROOT}/...")
# EVENT_SSL_ARGS+=(--event-ssl-root "event_ssl_seed666=${OUTPUTS_ROOT}/...")
# EVENT_SSL_ARGS+=(--event-ssl-root "event_ssl_seed606=${OUTPUTS_ROOT}/...")
# EVENT_SSL_ARGS+=(--event-ssl-root "event_ssl_seed37=${OUTPUTS_ROOT}/...")
# EVENT_SSL_ARGS+=(--event-ssl-root "event_ssl_seed4669=${OUTPUTS_ROOT}/...")

python "$ANALYSIS_SCRIPT" \
  --stage1-root "$STAGE1_ROOT" \
  --stage1-script "$STAGE1_SCRIPT" \
  --construction-null-script "$CONSTRUCTION_NULL_SCRIPT" \
  --construction-null-output-root "$CONSTRUCTION_NULL_ROOT" \
  --require-existing-construction-null-output \
  --mechanism-confirm-script "$MECHANISM_CONFIRM_SCRIPT" \
  --mechanism-phase1-script "$MECHANISM_PHASE1_SCRIPT" \
  --frozen-mechanism-manifest "$FROZEN_MECHANISM_MANIFEST" \
  --require-mechanism-manifest-checksum \
  "${EVENT_SSL_ARGS[@]}" \
  --require-event-ssl-manifest \
  --primary-event-ssl-label event_ssl_seed42 \
  --splits A_val B_confirm \
  --confirmation-output-only \
  --output-root "$RESULT_ROOT" \
  --bootstrap-splits B_confirm \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-1000}" \
  --bootstrap-batch-size "${BOOTSTRAP_BATCH_SIZE:-20}" \
  --bootstrap-seed "${BOOTSTRAP_SEED:-20260731}"
