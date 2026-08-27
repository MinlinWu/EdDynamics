#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
acquire_lock time_shuffle
ensure_code_snapshot
select_gpu "${1:-${GPU_ID:-0}}"
require_file "${MAIN_INPUT_ROOT}/metadata/stage4_input_manifest.json"
if [[ "${FORCE}" == "1" || ! -f "${TIME_INPUT_ROOT}/metadata/stage4_input_manifest.json" ]]; then
  run_logged time_shuffle_prepare \
    "${PYTHON_BIN}" "${CODE_ROOT}/control_time_shuffle.py" prepare \
    --prepare-script "${CODE_ROOT}/prepare_event_ssl_inputs.py" \
    --stage1-root "${STAGE1_ROOT}" \
    --output-root "${TIME_ROOT}" \
    --hash-buckets "${HASH_BUCKETS}" \
    --max-users-per-split 0 \
    --seed "${CONTROL_SEED}"
fi
require_file "${TIME_INPUT_ROOT}/metadata/stage4_input_manifest.json"
if [[ "${FORCE}" == "1" || ! -f "${TIME_MODEL_ROOT}/best_model.pt" ]]; then
  run_logged time_shuffle_train \
    "${PYTHON_BIN}" "${CODE_ROOT}/control_time_shuffle.py" train \
    --input-root "${TIME_INPUT_ROOT}" \
    --output-root "${TIME_MODEL_ROOT}" \
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
require_file "${TIME_MODEL_ROOT}/best_model.pt"
if [[ "${FORCE}" == "1" || ! -f "${TIME_EVAL_ROOT}/metadata/stage4_time_shuffle_control_manifest.json" ]]; then
  run_logged time_shuffle_evaluate \
    "${PYTHON_BIN}" "${CODE_ROOT}/control_time_shuffle.py" evaluate \
    --input-root "${MAIN_INPUT_ROOT}" \
    --train-input-root "${TIME_INPUT_ROOT}" \
    --checkpoint "${TIME_MODEL_ROOT}/best_model.pt" \
    --output-root "${TIME_EVAL_ROOT}" \
    --train-script "${CODE_ROOT}/train_event_ssl.py" \
    --evaluate-script "${CODE_ROOT}/evaluate_event_ssl_structure.py" \
    --splits A_val B_confirm \
    --chunk-len "${CHUNK_LEN}" \
    --stage1-root "${STAGE1_ROOT}" \
    --seed "${EVAL_SEED}" \
    --torch-num-threads "${TORCH_NUM_THREADS}"
fi
require_file "${TIME_EVAL_ROOT}/metadata/stage4_time_shuffle_control_manifest.json"
mark_done time_shuffle
