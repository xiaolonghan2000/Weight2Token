#!/bin/bash
#SBATCH --job-name=w2t-goemotions
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if [ -n "${VENV_ACTIVATE:-}" ]; then
  source "${VENV_ACTIVATE}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-./classification/outputs/goemotions_loras}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

mkdir -p "${OUTPUT_DIR}" logs

python ./classification/collect_goemotions_loras.py \
  --output_dir "${OUTPUT_DIR}" \
  --gpu_ids "${GPU_IDS}"
