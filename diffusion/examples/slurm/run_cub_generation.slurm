#!/bin/bash
#SBATCH --job-name=cub-lora
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-24

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${VENV_ACTIVATE:-}" ]; then
  source "$VENV_ACTIVATE"
fi

python data_generation/run_plan.py \
  --plan_csv ./data_generation/plans/cub_plan.csv \
  --runs_root ./data_generation/outputs/cub/models_rank8_full \
  --results_csv ./data_generation/outputs/cub/results.csv
