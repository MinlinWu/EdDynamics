#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
IFS=',' read -r -a GPUS <<< "${TRAIN_GPU_LIST:-0}"
if [[ "${#GPUS[@]}" -ge 2 ]]; then
  bash "${DIR}/predictive.sh" "${GPUS[0]}" & p1=$!
  bash "${DIR}/pure_ssl.sh" "${GPUS[1]}" & p2=$!
  wait "$p1"; wait "$p2"
else
  bash "${DIR}/predictive.sh" "${GPUS[0]}"
  bash "${DIR}/pure_ssl.sh" "${GPUS[0]}"
fi
