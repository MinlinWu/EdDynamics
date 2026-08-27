#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${CODE_ROOT:-${SCRIPT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-/data/datasets/KT4/outputs_KT4}"
STAGE1_ROOT="${STAGE1_ROOT:-${OUTPUTS_ROOT}/stage1}"
RESULT_ROOT="${RESULT_ROOT:-${OUTPUTS_ROOT}/state_only_closure_audit}"
REQUIRE_ALL_SEEDS="${REQUIRE_ALL_SEEDS:-1}"
OVERWRITE="${OVERWRITE:-0}"
WRITE_PRIMARY_PREDICTIONS="${WRITE_PRIMARY_PREDICTIONS:-0}"

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

ANALYSIS_SCRIPT="${ANALYSIS_SCRIPT:-$(resolve_code_file run_state_only_closure_audit.py run_state_only_closure_audit.py)}"
REPORT_SCRIPT="${REPORT_SCRIPT:-$(resolve_code_file extract_state_only_closure_audit_report.py extract_state_only_closure_audit_report.py)}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$(resolve_code_file train_event_ssl.py 'train_event_ssl.py')}"
EVALUATE_SCRIPT="${EVALUATE_SCRIPT:-$(resolve_code_file evaluate_event_ssl_structure.py 'evaluate_event_ssl_structure.py')}"
STAGE5_TRAIN_SCRIPT="${STAGE5_TRAIN_SCRIPT:-$(resolve_code_file stage5_macro_sufficiency_train.py 'stage5_macro_sufficiency_train.py')}"

STAGE5_SEED42_ROOT="${STAGE5_SEED42_ROOT:-${OUTPUTS_ROOT}/stage5_macro_sufficiency}"
STAGE5_SEED2026_ROOT="${STAGE5_SEED2026_ROOT:-${OUTPUTS_ROOT}/random_seed_experiments/seed_2026/stage5_macro_sufficiency}"
STAGE5_SEED666_ROOT="${STAGE5_SEED666_ROOT:-${OUTPUTS_ROOT}/random_seed_experiments/seed_666/stage5_macro_sufficiency}"
STAGE5_SEED606_ROOT="${STAGE5_SEED606_ROOT:-${OUTPUTS_ROOT}/random_seed_experiments/seed_606/stage5_macro_sufficiency}"
STAGE5_SEED37_ROOT="${STAGE5_SEED37_ROOT:-${OUTPUTS_ROOT}/random_seed_experiments/seed_37/stage5_macro_sufficiency}"
STAGE5_SEED4669_ROOT="${STAGE5_SEED4669_ROOT:-${OUTPUTS_ROOT}/random_seed_experiments/seed_4669/stage5_macro_sufficiency}"

for path in "$ANALYSIS_SCRIPT" "$REPORT_SCRIPT" "$TRAIN_SCRIPT" "$EVALUATE_SCRIPT" "$STAGE5_TRAIN_SCRIPT"; do
  [[ -f "$path" ]] || { echo "Missing required code file: $path" >&2; exit 1; }
done
[[ -d "$STAGE1_ROOT" ]] || { echo "Missing Stage-1 root: $STAGE1_ROOT" >&2; exit 1; }
[[ -d "$STAGE5_SEED42_ROOT" ]] || { echo "Missing primary Stage-5 root: $STAGE5_SEED42_ROOT" >&2; exit 1; }

SEED_ARGS=(--seed-root "42=${STAGE5_SEED42_ROOT}")
for seed in 2026 666 606 37 4669; do
  variable="STAGE5_SEED${seed}_ROOT"
  root="${!variable}"
  if [[ -d "$root" ]]; then
    SEED_ARGS+=(--seed-root "${seed}=${root}")
  elif [[ "$REQUIRE_ALL_SEEDS" == "1" ]]; then
    echo "Missing Stage-5 root for seed ${seed}: ${root}" >&2
    exit 1
  fi
done

"$PYTHON_BIN" -m py_compile \
  "$ANALYSIS_SCRIPT" \
  "$REPORT_SCRIPT" \
  "$TRAIN_SCRIPT" \
  "$EVALUATE_SCRIPT" \
  "$STAGE5_TRAIN_SCRIPT"
"$PYTHON_BIN" "$ANALYSIS_SCRIPT" --self-test

RUN_ARGS=(
  --output-root "$RESULT_ROOT"
  --primary-seed 42
  --train-script "$TRAIN_SCRIPT"
  --evaluate-script "$EVALUATE_SCRIPT"
  --stage5-train-script "$STAGE5_TRAIN_SCRIPT"
  --stage1-root "$STAGE1_ROOT"
  --sample-max-rows "${SAMPLE_MAX_ROWS:-600000}"
  --sample-seed "${SAMPLE_SEED:-42}"
  --chunk-len "${CHUNK_LEN:-512}"
  --metric-chunk-rows "${METRIC_CHUNK_ROWS:-200000}"
  --probability-chunk-rows "${PROBABILITY_CHUNK_ROWS:-100000}"
  --degree 3
  --knots "${SPLINE_KNOTS:-4,5,6}"
  --quadratic-alphas "${QUADRATIC_ALPHAS:-0.1,1,10}"
  --mean-alphas "${MEAN_ALPHAS:-0.1,1,10}"
  --variance-alphas "${VARIANCE_ALPHAS:-0.1,1,10}"
  --variance-crossfit-folds "${VARIANCE_CROSSFIT_FOLDS:-2}"
  --variance-crossfit-seed "${VARIANCE_CROSSFIT_SEED:-20260806}"
  --gauss-hermite-order "${GAUSS_HERMITE_ORDER:-5}"
  --quadrature-audit-orders "${QUADRATURE_AUDIT_ORDERS:-5,7,9,11}"
  --quadrature-max-order "${QUADRATURE_MAX_ORDER:-31}"
  --quadrature-order-step "${QUADRATURE_ORDER_STEP:-2}"
  --quadrature-audit-rows "${QUADRATURE_AUDIT_ROWS:-200000}"
  --quadrature-matrix-tolerance "${QUADRATURE_MATRIX_TOLERANCE:-0.01}"
  --quadrature-metric-tolerance "${QUADRATURE_METRIC_TOLERANCE:-0.01}"
  --quadrature-min-origin-rows "${QUADRATURE_MIN_ORIGIN_ROWS:-100}"
  --permutations "${PERMUTATIONS:-50}"
  --permutation-seed "${PERMUTATION_SEED:-20260805}"
  --torch-num-threads "${TORCH_NUM_THREADS:-0}"
)
RUN_ARGS+=("${SEED_ARGS[@]}")
if [[ "$OVERWRITE" == "1" ]]; then
  RUN_ARGS+=(--overwrite)
fi
if [[ "$WRITE_PRIMARY_PREDICTIONS" == "1" ]]; then
  RUN_ARGS+=(--write-primary-predictions)
fi

"$PYTHON_BIN" "$ANALYSIS_SCRIPT" "${RUN_ARGS[@]}"

REPORT_ARGS=(--analysis-root "$RESULT_ROOT")
if [[ "$OVERWRITE" == "1" ]]; then
  REPORT_ARGS+=(--overwrite)
fi
"$PYTHON_BIN" "$REPORT_SCRIPT" "${REPORT_ARGS[@]}"
