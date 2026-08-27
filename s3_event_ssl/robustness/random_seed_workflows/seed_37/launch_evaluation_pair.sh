#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
IFS=',' read -r -a GPUS <<< "${EVAL_GPU_LIST:-0}"
if [[ "${#GPUS[@]}" -ge 2 ]]; then
  bash "${DIR}/evaluate_pred.sh" "${GPUS[0]}" & p1=$!
  bash "${DIR}/evaluate_pure.sh" "${GPUS[1]}" & p2=$!
  wait "$p1"; wait "$p2"
else
  bash "${DIR}/evaluate_pred.sh" "${GPUS[0]}"
  bash "${DIR}/evaluate_pure.sh" "${GPUS[0]}"
fi
