#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
acquire_lock representation_geo
ensure_code_snapshot
select_gpu "${1:-${GPU_ID:-0}}"
require_file "${PRED_MODEL_ROOT}/best_model.pt"
if [[ "${FORCE}" == "1" || ! -f "${GEOMETRY_ROOT}/metadata/stage5_representation_geometry_artifacts.pkl" ]]; then
  run_logged representation_geo_train \
    "${PYTHON_BIN}" "${CODE_ROOT}/stage5_representation_geometry_train.py" \
    --input-root "${MAIN_INPUT_ROOT}" \
    --checkpoint "${PRED_MODEL_ROOT}/best_model.pt" \
    --output-root "${GEOMETRY_ROOT}" \
    --train-script "${CODE_ROOT}/train_event_ssl.py" \
    --evaluate-script "${CODE_ROOT}/evaluate_event_ssl_structure.py" \
    --train-split A_train \
    --sample-max-rows "${GEOMETRY_TRAIN_SAMPLE_MAX_ROWS}" \
    --chunk-len "${CHUNK_LEN}" \
    --pca-components "${PCA_COMPONENTS}" \
    --ridge-alpha "${RIDGE_ALPHA}" \
    --seed "${PROBE_SEED}" \
    --torch-num-threads "${TORCH_NUM_THREADS}"
fi
require_file "${GEOMETRY_ROOT}/metadata/stage5_representation_geometry_artifacts.pkl"
if [[ "${FORCE}" == "1" || ! -f "${GEOMETRY_EVAL_ROOT}/metadata/stage5_representation_geometry_evaluation_manifest.json" ]]; then
  run_logged representation_geo_evaluate \
    "${PYTHON_BIN}" "${CODE_ROOT}/stage5_representation_geometry_evaluate.py" \
    --input-root "${MAIN_INPUT_ROOT}" \
    --checkpoint "${PRED_MODEL_ROOT}/best_model.pt" \
    --artifacts "${GEOMETRY_ROOT}/metadata/stage5_representation_geometry_artifacts.pkl" \
    --output-root "${GEOMETRY_EVAL_ROOT}" \
    --train-script "${CODE_ROOT}/train_event_ssl.py" \
    --evaluate-script "${CODE_ROOT}/evaluate_event_ssl_structure.py" \
    --stage1-root "${STAGE1_ROOT}" \
    --splits A_val B_confirm \
    --chunk-len "${CHUNK_LEN}" \
    --sample-max-rows "${GEOMETRY_EVAL_SAMPLE_MAX_ROWS}" \
    --seed "${EVAL_SEED}" \
    --torch-num-threads "${TORCH_NUM_THREADS}"
fi
require_file "${GEOMETRY_EVAL_ROOT}/metadata/stage5_representation_geometry_evaluation_manifest.json"
mark_done representation_geo
