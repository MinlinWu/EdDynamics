#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
acquire_lock predictive
ensure_code_snapshot
select_gpu "${1:-${GPU_ID:-0}}"
require_file "${MAIN_INPUT_ROOT}/metadata/stage4_input_manifest.json"
if step_done predictive && [[ -f "${PRED_MODEL_ROOT}/best_model.pt" ]]; then
  echo "predictive training already completed for seed ${SEED}."
  exit 0
fi
run_logged predictive \
  "${PYTHON_BIN}" "${CODE_ROOT}/train_event_ssl.py" \
  --input-root "${MAIN_INPUT_ROOT}" \
  --output-root "${PRED_MODEL_ROOT}" \
  --model-kind predictive_state \
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
  --delta-scale "${DELTA_SCALE}" \
  --categorical-emb-dim "${CATEGORICAL_EMB_DIM}" \
  --seed "${MODEL_SEED}" \
  --amp-dtype "${AMP_DTYPE}" \
  --torch-num-threads "${TORCH_NUM_THREADS}" \
  --lambda-state "${LAMBDA_STATE}" \
  --lambda-closure "${LAMBDA_CLOSURE}"
require_file "${PRED_MODEL_ROOT}/best_model.pt"
mark_done predictive
