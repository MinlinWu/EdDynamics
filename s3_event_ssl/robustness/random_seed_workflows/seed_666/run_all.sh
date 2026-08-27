#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${DIR}/prepare.sh"
bash "${DIR}/launch_train_pair.sh"
bash "${DIR}/launch_evaluation_pair.sh"
bash "${DIR}/launch_post_stage.sh"
