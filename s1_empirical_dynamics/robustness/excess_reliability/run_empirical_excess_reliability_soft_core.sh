#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${CODE_ROOT:-${SCRIPT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-/data/datasets/KT4/outputs_KT4}"
STAGE1_ROOT="${STAGE1_ROOT:-${OUTPUTS_ROOT}/stage1}"
CONSTRUCTION_NULL_ROOT="${CONSTRUCTION_NULL_ROOT:-${OUTPUTS_ROOT}/stage1_construction_matched_null}"
CONSTRUCTION_NULL_CONFIRM_ROOT="${CONSTRUCTION_NULL_CONFIRM_ROOT:-${OUTPUTS_ROOT}/stage1_construction_matched_null_confirm}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${OUTPUTS_ROOT}/empirical_excess_reliability_soft_core}"
OVERWRITE="${OVERWRITE:-0}"

resolve_code_file() {
  local formal_name="$1"
  local uploaded_name="$2"
  if [[ -f "${CODE_ROOT}/${formal_name}" ]]; then
    printf '%s\n' "${CODE_ROOT}/${formal_name}"
  elif [[ -f "${CODE_ROOT}/${uploaded_name}" ]]; then
    printf '%s\n' "${CODE_ROOT}/${uploaded_name}"
  else
    printf '%s\n' "${CODE_ROOT}/${formal_name}"
  fi
}

require_table() {
  local base="$1"
  if [[ ! -f "${base}.parquet" && ! -f "${base}.csv.gz" && ! -f "${base}.csv" ]]; then
    echo "Required table not found: ${base}.[parquet|csv.gz|csv]" >&2
    exit 1
  fi
}

ANALYSIS_SCRIPT="${ANALYSIS_SCRIPT:-$(resolve_code_file run_empirical_excess_reliability_soft_core.py run_empirical_excess_reliability_soft_core.py)}"
REPORT_SCRIPT="${REPORT_SCRIPT:-$(resolve_code_file extract_empirical_excess_reliability_soft_core_report.py extract_empirical_excess_reliability_soft_core_report.py)}"
STAGE1_SCRIPT="${STAGE1_SCRIPT:-$(resolve_code_file build_effective_dynamics_kt4_stage1_empirical.py 'build_effective_dynamics_kt4_stage1_empirical.py')}"
CONSTRUCTION_NULL_SCRIPT="${CONSTRUCTION_NULL_SCRIPT:-$(resolve_code_file run_construction_matched_null.py 'run_construction_matched_null.py')}"

for path in \
  "$ANALYSIS_SCRIPT" \
  "$REPORT_SCRIPT" \
  "$STAGE1_SCRIPT" \
  "$CONSTRUCTION_NULL_SCRIPT" \
  "$CONSTRUCTION_NULL_ROOT/metadata/A_val_construction_null_manifest.json" \
  "$CONSTRUCTION_NULL_CONFIRM_ROOT/metadata/B_confirm_construction_null_manifest.json" \
  "$CONSTRUCTION_NULL_ROOT/arrays/A_val_construction_null_fields.npz" \
  "$CONSTRUCTION_NULL_CONFIRM_ROOT/arrays/B_confirm_construction_null_fields.npz"; do
  [[ -f "$path" ]] || { echo "Required input not found: $path" >&2; exit 1; }
done
for split in A_train A_val B_confirm; do
  require_table "$STAGE1_ROOT/dynamics/student_dynamics_panel_core_${split}"
done

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

"$PYTHON_BIN" -m py_compile \
  "$ANALYSIS_SCRIPT" \
  "$REPORT_SCRIPT" \
  "$STAGE1_SCRIPT" \
  "$CONSTRUCTION_NULL_SCRIPT"
"$PYTHON_BIN" "$ANALYSIS_SCRIPT" --self-test
"$PYTHON_BIN" "$REPORT_SCRIPT" --self-test

analysis_overwrite=()
report_overwrite=()
if [[ "$OVERWRITE" == "1" ]]; then
  analysis_overwrite+=(--overwrite)
  report_overwrite+=(--overwrite)
fi

"$PYTHON_BIN" "$ANALYSIS_SCRIPT" \
  --stage1-root "$STAGE1_ROOT" \
  --stage1-script "$STAGE1_SCRIPT" \
  --construction-null-script "$CONSTRUCTION_NULL_SCRIPT" \
  --expected-construction-null-script-sha256 "${EXPECTED_CONSTRUCTION_NULL_SCRIPT_SHA256:-c0b4149a65a7ba155950914ef0936d31b1812d30d4ce9f8dad2ec5d02636d0f9}" \
  --construction-null-output-root "$CONSTRUCTION_NULL_ROOT" \
  --construction-null-confirm-output-root "$CONSTRUCTION_NULL_CONFIRM_ROOT" \
  --output-root "$ANALYSIS_ROOT" \
  --split-half-partitions "${SPLIT_HALF_PARTITIONS:-32}" \
  --split-half-seed "${SPLIT_HALF_SEED:-20260804}" \
  --minimum-half-cell-coverage "${MINIMUM_HALF_CELL_COVERAGE:-0.95}" \
  --minimum-half-occupancy-coverage "${MINIMUM_HALF_OCCUPANCY_COVERAGE:-0.98}" \
  --minimum-valid-partition-fraction "${MINIMUM_VALID_PARTITION_FRACTION:-0.90}" \
  --minimum-defined-benchmark-fraction "${MINIMUM_DEFINED_BENCHMARK_FRACTION:-0.90}" \
  --minimum-half-drift-count "${MINIMUM_HALF_DRIFT_COUNT:-0}" \
  --minimum-half-cell-users "${MINIMUM_HALF_CELL_USERS:-0}" \
  --soft-core-replicates "${SOFT_CORE_REPLICATES:-1000}" \
  --soft-core-batch-size "${SOFT_CORE_BATCH_SIZE:-10}" \
  --soft-core-seed "${SOFT_CORE_SEED:-20260731}" \
  --max-last-resort-fraction "${MAX_LAST_RESORT_FRACTION:-0.01}" \
  --progress-every "${PROGRESS_EVERY:-5}" \
  --confirmation-output-only \
  "${analysis_overwrite[@]}"

"$PYTHON_BIN" "$REPORT_SCRIPT" \
  --analysis-root "$ANALYSIS_ROOT" \
  "${report_overwrite[@]}"
