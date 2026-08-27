#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/datasets/KT4/outputs_KT4}"
STAGE1_ROOT="${STAGE1_ROOT:-${OUTPUT_BASE}/stage1}"
CONSTRUCTION_NULL_ROOT="${CONSTRUCTION_NULL_ROOT:-${OUTPUT_BASE}/stage1_construction_matched_null}"
RESULT_ROOT="${RESULT_ROOT:-${OUTPUT_BASE}/stage1_kinetic_robustness}"
PARTITION_K_VALUES="${PARTITION_K_VALUES:-4,5,6,7,8}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-1000}"
BOOTSTRAP_BATCH="${BOOTSTRAP_BATCH:-25}"
NULL_REPLICATES="${NULL_REPLICATES:-100}"
RANDOM_SEED="${RANDOM_SEED:-42}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

resolve_script() {
    local explicit="$1"
    shift
    if [[ -n "$explicit" ]]; then
        if [[ ! -f "$explicit" ]]; then
            echo "Required script not found: $explicit" >&2
            exit 1
        fi
        printf '%s\n' "$explicit"
        return
    fi
    local candidate
    for candidate in "$@"; do
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    return 1
}

STAGE1_SCRIPT="$(resolve_script "${STAGE1_SCRIPT:-}" \
    "${SCRIPT_DIR}/build_effective_dynamics_kt4_stage1_empirical.py")" || {
    echo "Set STAGE1_SCRIPT to the formal empirical Stage-1 implementation." >&2
    exit 1
}

CONSTRUCTION_NULL_SCRIPT="$(resolve_script "${CONSTRUCTION_NULL_SCRIPT:-}" \
    "${SCRIPT_DIR}/run_construction_matched_null.py")" || {
    echo "Set CONSTRUCTION_NULL_SCRIPT to the formal construction-matched null implementation." >&2
    exit 1
}

PARTITION_SCRIPT="${PARTITION_SCRIPT:-${SCRIPT_DIR}/run_partition_cluster_kinetic_robustness.py}"
RECURSIVE_SCRIPT="${RECURSIVE_SCRIPT:-${SCRIPT_DIR}/run_recursive_construction_inertia_null.py}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-${SCRIPT_DIR}/summarize_kinetic_robustness.py}"
COMMON_SCRIPT="${COMMON_SCRIPT:-${SCRIPT_DIR}/kinetic_robustness_common.py}"

for path in \
    "$STAGE1_ROOT" \
    "$STAGE1_SCRIPT" \
    "$CONSTRUCTION_NULL_SCRIPT" \
    "$CONSTRUCTION_NULL_ROOT/metadata/A_val_construction_null_manifest.json" \
    "$PARTITION_SCRIPT" \
    "$RECURSIVE_SCRIPT" \
    "$SUMMARY_SCRIPT" \
    "$COMMON_SCRIPT"; do
    if [[ ! -e "$path" ]]; then
        echo "Required input not found: $path" >&2
        exit 1
    fi
done

if ! [[ "$BOOTSTRAP_REPLICATES" =~ ^[0-9]+$ ]] || (( BOOTSTRAP_REPLICATES < 200 )); then
    echo "BOOTSTRAP_REPLICATES must be an integer of at least 200; the publication default is 1000." >&2
    exit 1
fi
if ! [[ "$BOOTSTRAP_BATCH" =~ ^[0-9]+$ ]] || (( BOOTSTRAP_BATCH < 1 )); then
    echo "BOOTSTRAP_BATCH must be a positive integer." >&2
    exit 1
fi
if ! [[ "$NULL_REPLICATES" =~ ^[0-9]+$ ]] || (( NULL_REPLICATES < 100 )); then
    echo "NULL_REPLICATES must be an integer of at least 100." >&2
    exit 1
fi
if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]]; then
    echo "MAX_PARALLEL must be a non-negative integer." >&2
    exit 1
fi
if [[ "$FORCE_RERUN" != "0" && "$FORCE_RERUN" != "1" ]]; then
    echo "FORCE_RERUN must be 0 or 1." >&2
    exit 1
fi

PARTITION_ROOT="${RESULT_ROOT}/partition_cluster"
RECURSIVE_ROOT="${RESULT_ROOT}/recursive_null"
SUMMARY_ROOT="${RESULT_ROOT}/summary"
LOG_ROOT="${RESULT_ROOT}/logs"
mkdir -p "$LOG_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export KINETIC_ROBUSTNESS_USE_NUMBA="${KINETIC_ROBUSTNESS_USE_NUMBA:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${RESULT_ROOT}/numba_cache}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "$NUMBA_CACHE_DIR"

BACKEND_NAME="$("$PYTHON_BIN" -c 'from kinetic_robustness_common import recursive_kernel_backend; print(recursive_kernel_backend())')"
echo "[workflow] recursive numerical backend: ${BACKEND_NAME}"

partition_manifest="${PARTITION_ROOT}/metadata/partition_cluster_kinetic_manifest.json"
recursive_manifest="${RECURSIVE_ROOT}/metadata/recursive_construction_inertia_null_manifest.json"

if [[ "$FORCE_RERUN" == "1" ]]; then
    rm -rf "$PARTITION_ROOT" "$RECURSIVE_ROOT" "$SUMMARY_ROOT"
fi

run_partition_branch() {
    "$PYTHON_BIN" "$PARTITION_SCRIPT" \
        --stage1-root "$STAGE1_ROOT" \
        --stage1-script "$STAGE1_SCRIPT" \
        --output-root "$PARTITION_ROOT" \
        --partition-k-values "$PARTITION_K_VALUES" \
        --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
        --bootstrap-batch "$BOOTSTRAP_BATCH" \
        --seed "$RANDOM_SEED"
}

run_recursive_branch() {
    "$PYTHON_BIN" "$RECURSIVE_SCRIPT" \
        --stage1-root "$STAGE1_ROOT" \
        --stage1-script "$STAGE1_SCRIPT" \
        --construction-null-script "$CONSTRUCTION_NULL_SCRIPT" \
        --construction-null-root "$CONSTRUCTION_NULL_ROOT" \
        --output-root "$RECURSIVE_ROOT" \
        --replicates "$NULL_REPLICATES" \
        --seed "$RANDOM_SEED"
}

TASK_LABELS=()
TASK_COMMANDS=()
if [[ ! -f "$partition_manifest" ]]; then
    TASK_LABELS+=("partition_cluster")
    TASK_COMMANDS+=("run_partition_branch")
else
    echo "[workflow] partition/cluster branch already complete: $partition_manifest"
fi
if [[ ! -f "$recursive_manifest" ]]; then
    TASK_LABELS+=("recursive_null")
    TASK_COMMANDS+=("run_recursive_branch")
else
    echo "[workflow] recursive-null branch already complete: $recursive_manifest"
fi

run_task() {
    local label="$1"
    local command_name="$2"
    local log="${LOG_ROOT}/${label}.log"
    echo "[workflow] starting ${label}; log=${log}"
    if "$command_name" >"$log" 2>&1; then
        echo "[workflow] completed ${label}"
    else
        local status=$?
        echo "[workflow] failed ${label}; last 120 log lines:" >&2
        tail -n 120 "$log" >&2 || true
        return "$status"
    fi
}

if (( ${#TASK_LABELS[@]} > 0 )); then
    if (( MAX_PARALLEL == 1 || ${#TASK_LABELS[@]} == 1 )); then
        for index in "${!TASK_LABELS[@]}"; do
            run_task "${TASK_LABELS[$index]}" "${TASK_COMMANDS[$index]}"
        done
    else
        active=0
        limit="$MAX_PARALLEL"
        if (( limit == 0 || limit > ${#TASK_LABELS[@]} )); then
            limit="${#TASK_LABELS[@]}"
        fi
        failures=0
        for index in "${!TASK_LABELS[@]}"; do
            while (( active >= limit )); do
                if wait -n; then
                    :
                else
                    failures=1
                fi
                active=$((active - 1))
            done
            run_task "${TASK_LABELS[$index]}" "${TASK_COMMANDS[$index]}" &
            active=$((active + 1))
        done
        while (( active > 0 )); do
            if wait -n; then
                :
            else
                failures=1
            fi
            active=$((active - 1))
        done
        if (( failures != 0 )); then
            echo "One or more kinetic robustness branches failed." >&2
            exit 1
        fi
    fi
fi

for manifest in "$partition_manifest" "$recursive_manifest"; do
    if [[ ! -f "$manifest" ]]; then
        echo "Expected completed-branch manifest not found: $manifest" >&2
        exit 1
    fi
done

mkdir -p "$SUMMARY_ROOT"
summary_log="${LOG_ROOT}/summary.log"
if ! "$PYTHON_BIN" "$SUMMARY_SCRIPT" \
    --partition-root "$PARTITION_ROOT" \
    --recursive-null-root "$RECURSIVE_ROOT" \
    --output-root "$SUMMARY_ROOT" >"$summary_log" 2>&1; then
    echo "[workflow] summary failed; last 120 log lines:" >&2
    tail -n 120 "$summary_log" >&2 || true
    exit 1
fi

cat <<EOF
[workflow] complete
[workflow] report: ${SUMMARY_ROOT}/kinetic_robustness_report.md
[workflow] table 1a: ${SUMMARY_ROOT}/kinetic_table1a_recursive_null.csv.gz or .parquet
[workflow] table 1b: ${SUMMARY_ROOT}/kinetic_table1b_partition_sensitivity.csv.gz or .parquet
[workflow] table 2: ${SUMMARY_ROOT}/kinetic_table2_cluster_inference.csv.gz or .parquet
EOF
