#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON_BIN:-python}"
MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
case "$MODE" in
  both)
    "$PYTHON" "$HERE/run_ssl.py" smoke "$@"
    "$PYTHON" "$HERE/run_ssl.py" cpu "$@"
    ;;
  check|smoke|cpu|all|s1|s2|s3|report|status|selftest)
    "$PYTHON" "$HERE/run_ssl.py" "$MODE" "$@"
    ;;
  *) echo "Usage: $0 {both|check|smoke|cpu|all|s1|s2|s3|report|status|selftest} [options]" >&2; exit 2;;
esac
