#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
acquire_lock evaluate_pure
ensure_code_snapshot
select_gpu "${1:-${GPU_ID:-0}}"
require_file "${PURE_MODEL_ROOT}/best_model.pt"
if step_done evaluate_pure && [[ -f "${PURE_EVAL_ROOT}/metadata/stage4_event_ssl_evaluation_manifest.json" ]]; then
  echo "pure-SSL evaluation already completed for seed ${SEED}."
  exit 0
fi
run_logged evaluate_pure \
  "${PYTHON_BIN}" "${CODE_ROOT}/evaluate_event_ssl_structure.py" \
  --input-root "${MAIN_INPUT_ROOT}" \
  --checkpoint "${PURE_MODEL_ROOT}/best_model.pt" \
  --output-root "${PURE_EVAL_ROOT}" \
  --train-script "${CODE_ROOT}/train_event_ssl.py" \
  --splits A_val B_confirm \
  --chunk-len "${CHUNK_LEN}" \
  --stage1-root "${STAGE1_ROOT}" \
  --fit-probe-for-pure-ssl \
  --probe-max-rows "${PURE_PROBE_MAX_ROWS}" \
  --seed "${PROBE_SEED}" \
  --torch-num-threads "${TORCH_NUM_THREADS}"
require_file "${PURE_EVAL_ROOT}/metadata/stage4_event_ssl_evaluation_manifest.json"
mark_done evaluate_pure
