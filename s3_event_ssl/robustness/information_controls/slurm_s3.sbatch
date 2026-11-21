#!/bin/bash
#SBATCH --job-name=ssl-information-controls
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:5
#SBATCH --mem=320G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=ssl_information_%j.log
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
CODE_ROOT="${SSL_CODE_ROOT:-${SLURM_SUBMIT_DIR:?}}"
PYTHON="${PYTHON_BIN:-python}"
test -f "$CODE_ROOT/slurm_s3.py"
exec srun --nodes=1 --ntasks=1 --cpus-per-task=40 --gres=gpu:5 --kill-on-bad-exit=1 "$PYTHON" -u -B "$CODE_ROOT/slurm_s3.py" --code-root "$CODE_ROOT" "$@"
