#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-/data/datasets/KT4/outputs_KT4}"
RESULT_ROOT="${RESULT_ROOT:-${OUTPUTS_ROOT}/stage5_objective_control_hidden_geometry}"
SEEDS="${SEEDS:-42,2026,666,606,37,4669}"
GEOMETRY_GPU_LIST="${GEOMETRY_GPU_LIST:-0}"
REQUIRE_ALL_SEEDS="${REQUIRE_ALL_SEEDS:-1}"
REQUIRE_FORMAL_RECONSTRUCTION="${REQUIRE_FORMAL_RECONSTRUCTION:-1}"
OVERWRITE="${OVERWRITE:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
TRAIN_SAMPLE_MAX_ROWS="${TRAIN_SAMPLE_MAX_ROWS:-300000}"
HELDOUT_SAMPLE_MAX_ROWS="${HELDOUT_SAMPLE_MAX_ROWS:-250000}"
CHUNK_LEN="${CHUNK_LEN:-512}"
PCA_COMPONENTS="${PCA_COMPONENTS:-64}"
PC_ALIGNMENT_COMPONENTS="${PC_ALIGNMENT_COMPONENTS:-16}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
RECONSTRUCTION_TOLERANCE="${RECONSTRUCTION_TOLERANCE:-0.002}"
REFERENCE_SEED="${REFERENCE_SEED:-42}"
RUN_NONLINEAR_REFERENCE="${RUN_NONLINEAR_REFERENCE:-1}"
WRITE_REFERENCE_ARTIFACTS="${WRITE_REFERENCE_ARTIFACTS:-1}"
CONFIDENCE_LEVEL="${CONFIDENCE_LEVEL:-0.95}"

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

ANALYSIS_SCRIPT="${ANALYSIS_SCRIPT:-$(resolve_code_file run_objective_control_hidden_geometry.py run_objective_control_hidden_geometry.py)}"
REPORT_SCRIPT="${REPORT_SCRIPT:-$(resolve_code_file extract_objective_control_hidden_geometry_report.py extract_objective_control_hidden_geometry_report.py)}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$(resolve_code_file train_event_ssl.py 'train_event_ssl.py')}"
EVALUATE_SCRIPT="${EVALUATE_SCRIPT:-$(resolve_code_file evaluate_event_ssl_structure.py 'evaluate_event_ssl_structure.py')}"
TASK_SCRIPT="${TASK_SCRIPT:-$(resolve_code_file control_task_only.py 'control_task_only.py')}"

for path in "$ANALYSIS_SCRIPT" "$REPORT_SCRIPT" "$TRAIN_SCRIPT" "$EVALUATE_SCRIPT" "$TASK_SCRIPT"; do
  [[ -f "$path" ]] || { echo "Missing required code file: $path" >&2; exit 1; }
done

resolve_additional_seed_root() {
  local seed="$1"
  local explicit_var="SEED${seed}_ROOT"
  local explicit="${!explicit_var:-}"
  if [[ -n "$explicit" ]]; then
    printf '%s\n' "$explicit"
    return
  fi
  local compact="${OUTPUTS_ROOT}/random_seed_experiments/seed${seed}"
  local underscored="${OUTPUTS_ROOT}/random_seed_experiments/seed_${seed}"
  if [[ -d "$compact" ]]; then
    printf '%s\n' "$compact"
  elif [[ -d "$underscored" ]]; then
    printf '%s\n' "$underscored"
  else
    printf '%s\n' "$compact"
  fi
}

seed_root() {
  local seed="$1"
  if [[ "$seed" == "42" ]]; then
    printf '%s\n' "${SEED42_ROOT:-${OUTPUTS_ROOT}}"
  else
    resolve_additional_seed_root "$seed"
  fi
}

IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
IFS=',' read -r -a GPU_ARRAY <<< "$GEOMETRY_GPU_LIST"
[[ ${#SEED_ARRAY[@]} -gt 0 ]] || { echo "No seeds supplied." >&2; exit 1; }
[[ ${#GPU_ARRAY[@]} -gt 0 ]] || { echo "No GPU slots supplied." >&2; exit 1; }

for seed in "${SEED_ARRAY[@]}"; do
  root="$(seed_root "$seed")"
  [[ -d "$root" ]] || { echo "Missing experiment root for seed ${seed}: ${root}" >&2; exit 1; }
  for required in \
    "${root}/stage4_event_ssl/prepared_inputs/metadata/stage4_input_manifest.json" \
    "${root}/stage4_event_ssl/models/predictive_state/best_model.pt" \
    "${root}/stage4_event_ssl/models/pure_ssl/best_model.pt" \
    "${root}/stage4_event_ssl/controls/task_only/model/best_model.pt"; do
    [[ -f "$required" ]] || { echo "Missing seed ${seed} input: ${required}" >&2; exit 1; }
  done
done

mkdir -p "${RESULT_ROOT}/logs"
python -m py_compile "$ANALYSIS_SCRIPT" "$REPORT_SCRIPT" "$TRAIN_SCRIPT" "$EVALUATE_SCRIPT" "$TASK_SCRIPT"
python "$ANALYSIS_SCRIPT" --self-test

run_seed_job() {
  local seed="$1"
  local gpu="$2"
  local root
  root="$(seed_root "$seed")"
  local args=(
    run-seed
    --seed "$seed"
    --experiment-root "$root"
    --result-root "$RESULT_ROOT"
    --train-script "$TRAIN_SCRIPT"
    --evaluate-script "$EVALUATE_SCRIPT"
    --task-script "$TASK_SCRIPT"
    --train-sample-max-rows "$TRAIN_SAMPLE_MAX_ROWS"
    --heldout-sample-max-rows "$HELDOUT_SAMPLE_MAX_ROWS"
    --sample-seed "$SAMPLE_SEED"
    --chunk-len "$CHUNK_LEN"
    --pca-components "$PCA_COMPONENTS"
    --pc-alignment-components "$PC_ALIGNMENT_COMPONENTS"
    --ridge-alpha "$RIDGE_ALPHA"
    --nonlinear-seed "$REFERENCE_SEED"
    --artifact-seed "$REFERENCE_SEED"
    --reference-seed "$REFERENCE_SEED"
    --reconstruction-tolerance "$RECONSTRUCTION_TOLERANCE"
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    args+=(--overwrite)
  fi
  if [[ "$seed" == "$REFERENCE_SEED" && "$RUN_NONLINEAR_REFERENCE" == "1" ]]; then
    args+=(--run-nonlinear)
  fi
  if [[ "$seed" == "$REFERENCE_SEED" && "$WRITE_REFERENCE_ARTIFACTS" == "1" ]]; then
    args+=(--write-seed-artifacts)
  fi
  if [[ "$seed" == "$REFERENCE_SEED" && "$REQUIRE_FORMAL_RECONSTRUCTION" == "1" ]]; then
    args+=(--require-formal-reconstruction)
  fi
  echo "[objective geometry] seed=${seed} gpu=${gpu} root=${root}" >&2
  if [[ "$gpu" == "cpu" ]]; then
    CUDA_VISIBLE_DEVICES="" python "$ANALYSIS_SCRIPT" "${args[@]}" \
      >"${RESULT_ROOT}/logs/seed${seed}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$gpu" python "$ANALYSIS_SCRIPT" "${args[@]}" \
      >"${RESULT_ROOT}/logs/seed${seed}.log" 2>&1
  fi
}

for ((start=0; start<${#SEED_ARRAY[@]}; start+=${#GPU_ARRAY[@]})); do
  pids=()
  batch_seeds=()
  for ((slot=0; slot<${#GPU_ARRAY[@]}; slot++)); do
    index=$((start + slot))
    if (( index >= ${#SEED_ARRAY[@]} )); then
      break
    fi
    seed="${SEED_ARRAY[$index]}"
    gpu="${GPU_ARRAY[$slot]}"
    run_seed_job "$seed" "$gpu" &
    pids+=("$!")
    batch_seeds+=("$seed")
  done
  status=0
  for ((i=0; i<${#pids[@]}; i++)); do
    if ! wait "${pids[$i]}"; then
      echo "Seed ${batch_seeds[$i]} failed. See ${RESULT_ROOT}/logs/seed${batch_seeds[$i]}.log" >&2
      status=1
    fi
  done
  (( status == 0 )) || exit 1
done

finalize_args=(
  finalize
  --result-root "$RESULT_ROOT"
  --reference-seed "$REFERENCE_SEED"
  --confidence "$CONFIDENCE_LEVEL"
  --seeds "${SEED_ARRAY[@]}"
)
if [[ "$REQUIRE_ALL_SEEDS" == "1" ]]; then
  finalize_args+=(--require-all-seeds)
fi
python "$ANALYSIS_SCRIPT" "${finalize_args[@]}"
python "$REPORT_SCRIPT" --result-root "$RESULT_ROOT" --overwrite

echo "[objective geometry] complete: ${RESULT_ROOT}"
