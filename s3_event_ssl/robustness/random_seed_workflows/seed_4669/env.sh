#!/usr/bin/env bash
set -euo pipefail

SEED=4669
MODEL_SEED="${MODEL_SEED:-${SEED}}"
CONTROL_SEED="${CONTROL_SEED:-${SEED}}"
PROBE_SEED="${PROBE_SEED:-${SEED}}"
EVAL_SEED="${EVAL_SEED:-${SEED}}"
PREP_SEED="${PREP_SEED:-${SEED}}"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${THIS_DIR}/.." && pwd)"
SOURCE_CODE_ROOT="${SOURCE_CODE_ROOT:-${BUNDLE_ROOT}/code}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE1_ROOT="${STAGE1_ROOT:-/data/datasets/KT4/outputs_KT4/stage1}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/datasets/KT4/outputs_KT4/random_seed_experiments}"
RUN_ROOT="${OUTPUT_BASE}/seed_${SEED}"
CODE_ROOT="${RUN_ROOT}/code_snapshot"
LOG_ROOT="${RUN_ROOT}/logs"
LOCK_ROOT="${RUN_ROOT}/locks"
MARKER_ROOT="${RUN_ROOT}/markers"
ENV_ROOT="${RUN_ROOT}/environment"

MAIN_ROOT="${RUN_ROOT}/stage4_event_ssl"
MAIN_INPUT_ROOT="${MAIN_ROOT}/prepared_inputs"
PRED_MODEL_ROOT="${MAIN_ROOT}/models/predictive_state"
PURE_MODEL_ROOT="${MAIN_ROOT}/models/pure_ssl"
PRED_EVAL_ROOT="${MAIN_ROOT}/evaluation_predictive_state"
PURE_EVAL_ROOT="${MAIN_ROOT}/evaluation_pure_ssl_probe"

TIME_ROOT="${RUN_ROOT}/stage4_event_ssl_time_shuffle_control"
TIME_INPUT_ROOT="${TIME_ROOT}/prepared_inputs"
TIME_MODEL_ROOT="${TIME_ROOT}/model"
TIME_EVAL_ROOT="${TIME_ROOT}/evaluation_on_ordered_inputs"

TASK_ROOT="${MAIN_ROOT}/controls/task_only"
TASK_MODEL_ROOT="${TASK_ROOT}/model"
TASK_EVAL_ROOT="${TASK_ROOT}/evaluation"

TAG_ROOT="${RUN_ROOT}/stage4_event_ssl_tag_support_randomized_control"
TAG_INPUT_ROOT="${TAG_ROOT}/prepared_inputs"
TAG_MODEL_ROOT="${TAG_ROOT}/model"
TAG_EVAL_ROOT="${TAG_ROOT}/evaluation"

MACRO_ROOT="${RUN_ROOT}/stage5_macro_sufficiency"
MACRO_EVAL_ROOT="${MACRO_ROOT}/evaluation"
GEOMETRY_ROOT="${RUN_ROOT}/stage5_representation_geometry"
GEOMETRY_EVAL_ROOT="${GEOMETRY_ROOT}/evaluation"

HASH_BUCKETS="${HASH_BUCKETS:-32768}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-192}"
SEQ_LEN="${SEQ_LEN:-256}"
STRIDE="${STRIDE:-128}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-3}"
WARMUP_STEPS="${WARMUP_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-320}"
INPUT_DIM="${INPUT_DIM:-224}"
NUM_LAYERS="${NUM_LAYERS:-2}"
DROPOUT="${DROPOUT:-0.10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FUTURE_STEPS="${FUTURE_STEPS:-1,2,4}"
LAMBDA_FUTURE="${LAMBDA_FUTURE:-1.0}"
LAMBDA_STATE="${LAMBDA_STATE:-0.5}"
LAMBDA_CLOSURE="${LAMBDA_CLOSURE:-0.5}"
DELTA_SCALE="${DELTA_SCALE:-0.50}"
CATEGORICAL_EMB_DIM="${CATEGORICAL_EMB_DIM:-16}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
CHUNK_LEN="${CHUNK_LEN:-512}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-0}"
PURE_PROBE_MAX_ROWS="${PURE_PROBE_MAX_ROWS:-300000}"
MACRO_SAMPLE_MAX_ROWS="${MACRO_SAMPLE_MAX_ROWS:-600000}"
GEOMETRY_TRAIN_SAMPLE_MAX_ROWS="${GEOMETRY_TRAIN_SAMPLE_MAX_ROWS:-300000}"
GEOMETRY_EVAL_SAMPLE_MAX_ROWS="${GEOMETRY_EVAL_SAMPLE_MAX_ROWS:-250000}"
PCA_COMPONENTS="${PCA_COMPONENTS:-64}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
FORCE="${FORCE:-0}"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${SEED}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

CODE_FILES=(
  prepare_event_ssl_inputs.py
  train_event_ssl.py
  evaluate_event_ssl_structure.py
  control_time_shuffle.py
  control_task_only.py
  control_tag_support_randomization.py
  stage5_macro_sufficiency_train.py
  stage5_macro_sufficiency_evaluate.py
  stage5_representation_geometry_train.py
  stage5_representation_geometry_evaluate.py
)

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${LOCK_ROOT}" "${MARKER_ROOT}" "${ENV_ROOT}"

ensure_code_snapshot() {
  mkdir -p "${CODE_ROOT}"
  if [[ -f "${CODE_ROOT}/SHA256SUMS.txt" && "${REFRESH_CODE_SNAPSHOT:-0}" != "1" ]]; then
    (cd "${CODE_ROOT}" && sha256sum -c SHA256SUMS.txt >/dev/null)
    return
  fi
  rm -f "${CODE_ROOT}/SHA256SUMS.txt"
  for file in "${CODE_FILES[@]}"; do
    [[ -f "${SOURCE_CODE_ROOT}/${file}" ]] || { echo "Missing source code: ${SOURCE_CODE_ROOT}/${file}" >&2; exit 2; }
    cp -p "${SOURCE_CODE_ROOT}/${file}" "${CODE_ROOT}/${file}"
  done
  (cd "${CODE_ROOT}" && sha256sum "${CODE_FILES[@]}" > SHA256SUMS.txt)
}

record_environment() {
  [[ -f "${ENV_ROOT}/environment_complete.txt" && "${FORCE}" != "1" ]] && return
  {
    date -Is
    uname -a
    "${PYTHON_BIN}" --version
    echo "seed=${SEED}"
    echo "model_seed=${MODEL_SEED}"
    echo "control_seed=${CONTROL_SEED}"
    echo "probe_seed=${PROBE_SEED}"
    echo "eval_seed=${EVAL_SEED}"
    echo "prep_seed=${PREP_SEED}"
    echo "stage1_root=${STAGE1_ROOT}"
    echo "run_root=${RUN_ROOT}"
  } > "${ENV_ROOT}/runtime.txt"
  "${PYTHON_BIN}" -m pip freeze > "${ENV_ROOT}/pip_freeze.txt" 2>/dev/null || true
  nvidia-smi -L > "${ENV_ROOT}/nvidia_smi_devices.txt" 2>/dev/null || true
  nvidia-smi > "${ENV_ROOT}/nvidia_smi_snapshot.txt" 2>/dev/null || true
  cp "${CODE_ROOT}/SHA256SUMS.txt" "${ENV_ROOT}/code_SHA256SUMS.txt"
  date -Is > "${ENV_ROOT}/environment_complete.txt"
}

require_file() {
  [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 3; }
}

acquire_lock() {
  local name="$1"
  exec 9>"${LOCK_ROOT}/${name}.lock"
  if command -v flock >/dev/null 2>&1; then
    flock -n 9 || { echo "Step ${name} is already running for seed ${SEED}." >&2; exit 4; }
  fi
}

step_done() {
  [[ -f "${MARKER_ROOT}/$1.done" && "${FORCE}" != "1" ]]
}

mark_done() {
  date -Is > "${MARKER_ROOT}/$1.done"
}

run_logged() {
  local name="$1"
  shift
  local log="${LOG_ROOT}/${name}.log"
  echo "[$(date -Is)] seed=${SEED} step=${name}" | tee -a "${log}"
  echo "command: $*" | tee -a "${log}"
  if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -v "$@" 2>&1 | tee -a "${log}"
  else
    "$@" 2>&1 | tee -a "${log}"
  fi
  echo "[$(date -Is)] completed seed=${SEED} step=${name}" | tee -a "${log}"
}

select_gpu() {
  local requested="${1:-${GPU_ID:-0}}"
  export CUDA_VISIBLE_DEVICES="${requested}"
  echo "seed=${SEED} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
}
