#!/bin/bash
#SBATCH --job-name=w2t-regression
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if [ -n "${VENV_ACTIVATE:-}" ]; then
  source "${VENV_ACTIVATE}"
fi

PLAN_DIR="${PLAN_DIR:-./regression/plans}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-./regression/collect/train_lora_llm.py}"
RUNS_ROOT="${RUNS_ROOT:-./outputs/regression/runs}"
RESULTS_CSV="${RESULTS_CSV:-./outputs/regression/results_arc_easy.csv}"

mkdir -p "$(dirname "${RESULTS_CSV}")" "${RUNS_ROOT}" logs

python ./regression/collect/run_plan.py \
  --plan_dir "${PLAN_DIR}" \
  --train_script "${TRAIN_SCRIPT}" \
  --runs_root "${RUNS_ROOT}" \
  --results_csv "${RESULTS_CSV}"
