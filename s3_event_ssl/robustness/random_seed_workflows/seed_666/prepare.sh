#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
acquire_lock prepare
ensure_code_snapshot
record_environment
if step_done prepare && [[ -f "${MAIN_INPUT_ROOT}/metadata/stage4_input_manifest.json" ]]; then
  echo "prepare already completed for seed ${SEED}."
  exit 0
fi
run_logged prepare \
  "${PYTHON_BIN}" "${CODE_ROOT}/prepare_event_ssl_inputs.py" \
  --stage1-root "${STAGE1_ROOT}" \
  --output-root "${MAIN_ROOT}" \
  --hash-buckets "${HASH_BUCKETS}" \
  --max-users-per-split 0 \
  --seed "${PREP_SEED}"
require_file "${MAIN_INPUT_ROOT}/metadata/stage4_input_manifest.json"
mark_done prepare
