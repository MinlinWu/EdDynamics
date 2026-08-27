#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
acquire_lock task_only
ensure_code_snapshot
select_gpu "${1:-${GPU_ID:-0}}"
require_file "${MAIN_INPUT_ROOT}/metadata/stage4_input_manifest.json"
if [[ "${FORCE}" == "1" || ! -f "${TASK_MODEL_ROOT}/best_model.pt" ]]; then
  run_logged task_only_train \
    "${PYTHON_BIN}" "${CODE_ROOT}/control_task_only.py" train \
    --input-root "${MAIN_INPUT_ROOT}" \
    --output-root "${TASK_MODEL_ROOT}" \
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
    --categorical-emb-dim "${CATEGORICAL_EMB_DIM}" \
    --future-steps "${FUTURE_STEPS}" \
    --delta-scale "${DELTA_SCALE}" \
    --seed "${MODEL_SEED}" \
    --amp-dtype "${AMP_DTYPE}" \
    --torch-num-threads "${TORCH_NUM_THREADS}"
fi
require_file "${TASK_MODEL_ROOT}/best_model.pt"
if [[ "${FORCE}" == "1" || ! -f "${TASK_EVAL_ROOT}/metadata/stage4_task_only_evaluation_manifest.json" ]]; then
  run_logged task_only_evaluate \
    "${PYTHON_BIN}" "${CODE_ROOT}/control_task_only.py" evaluate \
    --input-root "${MAIN_INPUT_ROOT}" \
    --checkpoint "${TASK_MODEL_ROOT}/best_model.pt" \
    --output-root "${TASK_EVAL_ROOT}" \
    --train-script "${CODE_ROOT}/train_event_ssl.py" \
    --evaluate-script "${CODE_ROOT}/evaluate_event_ssl_structure.py" \
    --splits A_val B_confirm \
    --chunk-len "${CHUNK_LEN}" \
    --stage1-root "${STAGE1_ROOT}" \
    --probe-max-rows "${PURE_PROBE_MAX_ROWS}" \
    --seed "${PROBE_SEED}" \
    --torch-num-threads "${TORCH_NUM_THREADS}"
fi
require_file "${TASK_EVAL_ROOT}/metadata/stage4_task_only_evaluation_manifest.json"
mark_done task_only
