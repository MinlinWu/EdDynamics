#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
JOBS=(time_shuffle.sh task_only.sh tag_support.sh macro_suff.sh representation_geo.sh)
IFS=',' read -r -a GPUS <<< "${POST_GPU_LIST:-0}"
GPU_COUNT="${#GPUS[@]}"
for ((offset=0; offset<${#JOBS[@]}; offset+=GPU_COUNT)); do
  pids=()
  for ((slot=0; slot<GPU_COUNT; slot++)); do
    index=$((offset + slot))
    (( index < ${#JOBS[@]} )) || break
    bash "${DIR}/${JOBS[$index]}" "${GPUS[$slot]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done
