#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNNER="${RUNNER:-${SCRIPT_DIR}/run_stage1_empirical_sensitivity.py}"
SUMMARIZER="${SUMMARIZER:-${SCRIPT_DIR}/summarize_stage1_empirical_sensitivity.py}"

SOURCE_SCRIPT="${SOURCE_SCRIPT:-${SCRIPT_DIR}/build_effective_dynamics_kt4_stage1_empirical.py}"
if [[ ! -f "${SOURCE_SCRIPT}" && -f "${SCRIPT_DIR}/build_effective_dynamics_kt4_stage1_empirical.py" ]]; then
    SOURCE_SCRIPT="${SCRIPT_DIR}/build_effective_dynamics_kt4_stage1_empirical.py"
fi

BASELINE_OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-/data/datasets/KT4/outputs_KT4}"
SENSITIVITY_ROOT="${SENSITIVITY_ROOT:-/data/datasets/KT4/outputs_KT4/stage1_sensitivity}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
OVERWRITE="${OVERWRITE:-0}"

if [[ ! -f "${SOURCE_SCRIPT}" ]]; then
    echo "Missing empirical Stage-1 source script: ${SOURCE_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${RUNNER}" ]]; then
    echo "Missing sensitivity runner: ${RUNNER}" >&2
    exit 1
fi
if [[ ! -f "${SUMMARIZER}" ]]; then
    echo "Missing sensitivity summarizer: ${SUMMARIZER}" >&2
    exit 1
fi
if [[ ! -f "${BASELINE_OUTPUT_ROOT}/stage1/dynamics/coordinate_analysis/MR_PsiA/coordinate_summary.json" ]]; then
    echo "Missing formal Stage-1 baseline outputs under: ${BASELINE_OUTPUT_ROOT}" >&2
    exit 1
fi
if ! [[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_PARALLEL must be a positive integer." >&2
    exit 1
fi

mkdir -p "${SENSITIVITY_ROOT}/logs" "${SENSITIVITY_ROOT}/summary"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

export PYTHON_BIN RUNNER SOURCE_SCRIPT BASELINE_OUTPUT_ROOT SENSITIVITY_ROOT OVERWRITE

run_one_variant() {
    local variant="$1"
    local output_root="${SENSITIVITY_ROOT}/${variant}"
    local log_path="${SENSITIVITY_ROOT}/logs/${variant}.log"
    local args=(
        "${RUNNER}"
        --source-script "${SOURCE_SCRIPT}"
        --baseline-root "${BASELINE_OUTPUT_ROOT}"
        --variant "${variant}"
        --output-root "${output_root}"
    )
    if [[ "${OVERWRITE}" == "1" ]]; then
        args+=(--overwrite)
    fi
    echo "[$(date '+%F %T')] starting ${variant}; log=${log_path}"
    "${PYTHON_BIN}" "${args[@]}" >"${log_path}" 2>&1
    echo "[$(date '+%F %T')] completed ${variant}"
}
export -f run_one_variant

variants=(memory_5d memory_20d activity_fast activity_slow)
printf '%s\0' "${variants[@]}" \
    | xargs -0 -n 1 -P "${MAX_PARALLEL}" bash -c 'run_one_variant "$1"' _

"${PYTHON_BIN}" "${SUMMARIZER}" \
    --baseline-root "${BASELINE_OUTPUT_ROOT}" \
    --sensitivity-root "${SENSITIVITY_ROOT}" \
    --output-base "${SENSITIVITY_ROOT}/summary/stage1_coordinate_sensitivity"

echo "All Stage-1 sensitivity runs completed."
echo "Summary: ${SENSITIVITY_ROOT}/summary/stage1_coordinate_sensitivity.csv"
