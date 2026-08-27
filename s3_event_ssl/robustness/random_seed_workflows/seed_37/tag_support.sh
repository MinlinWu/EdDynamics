#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
acquire_lock tag_support
ensure_code_snapshot
select_gpu "${1:-${GPU_ID:-0}}"
if [[ "${FORCE}" == "1" || ! -f "${TAG_INPUT_ROOT}/metadata/stage4_input_manifest.json" ]]; then
  run_logged tag_support_prepare \
    "${PYTHON_BIN}" "${CODE_ROOT}/control_tag_support_randomization.py" prepare \
    --prepare-script "${CODE_ROOT}/prepare_event_ssl_inputs.py" \
    --stage1-root "${STAGE1_ROOT}" \
    --output-root "${TAG_ROOT}" \
    --hash-buckets "${HASH_BUCKETS}" \
    --max-users-per-split 0 \
    --seed "${CONTROL_SEED}"
fi
require_file "${TAG_INPUT_ROOT}/metadata/stage4_input_manifest.json"
if [[ "${FORCE}" == "1" || ! -f "${TAG_MODEL_ROOT}/best_model.pt" ]]; then
  run_logged tag_support_train \
    "${PYTHON_BIN}" "${CODE_ROOT}/control_tag_support_randomization.py" train \
    --input-root "${TAG_INPUT_ROOT}" \
    --output-root "${TAG_MODEL_ROOT}" \
    --train-script "${CODE_ROOT}/train_event_ssl.py" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --seq-len "${SEQ_LEN}" \
    --stride "${STRIDE}" \
    --min-seq-len "${MIN_SEQ_LEN}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --lr "${LEARNING_RATE}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --input-dim "${INPUT_DIM}" \
    --num-layers "${NUM_LAYERS}" \
    --dropout "${DROPOUT}" \
    --num-workers "${NUM_WORKERS}" \
    --future-steps "${FUTURE_STEPS}" \
    --lambda-future "${LAMBDA_FUTURE}" \
    --lambda-state "${LAMBDA_STATE}" \
    --lambda-closure "${LAMBDA_CLOSURE}" \
    --delta-scale "${DELTA_SCALE}" \
    --categorical-emb-dim "${CATEGORICAL_EMB_DIM}" \
    --seed "${MODEL_SEED}" \
    --amp-dtype "${AMP_DTYPE}" \
    --torch-num-threads "${TORCH_NUM_THREADS}"
fi
require_file "${TAG_MODEL_ROOT}/best_model.pt"
if [[ "${FORCE}" == "1" || ! -f "${TAG_EVAL_ROOT}/metadata/stage4_tag_support_randomization_control_manifest.json" ]]; then
  run_logged tag_support_evaluate \
    "${PYTHON_BIN}" "${CODE_ROOT}/control_tag_support_randomization.py" evaluate \
    --input-root "${TAG_INPUT_ROOT}" \
    --checkpoint "${TAG_MODEL_ROOT}/best_model.pt" \
    --output-root "${TAG_EVAL_ROOT}" \
    --train-script "${CODE_ROOT}/train_event_ssl.py" \
    --evaluate-script "${CODE_ROOT}/evaluate_event_ssl_structure.py" \
    --splits A_val B_confirm \
    --chunk-len "${CHUNK_LEN}" \
    --stage1-root "${STAGE1_ROOT}" \
    --seed "${EVAL_SEED}" \
    --torch-num-threads "${TORCH_NUM_THREADS}"
fi
require_file "${TAG_EVAL_ROOT}/metadata/stage4_tag_support_randomization_control_manifest.json"
mark_done tag_support
