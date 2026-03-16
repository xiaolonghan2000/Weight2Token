#!/bin/bash
#SBATCH --job-name=w2t-retrieval
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

PLAN_DIR="${PLAN_DIR:-./retrieval/plans}"
PLAN_FILES="${PLAN_FILES:-arc_challenge_gallery_plan.csv,boolq_gallery_plan.csv,gsm8k_gallery_plan.csv,mbpp_gallery_plan.csv}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-./regression/collect/train_lora_llm.py}"
RUNS_ROOT="${RUNS_ROOT:-./outputs/retrieval/gallery_runs}"
RESULTS_DIR="${RESULTS_DIR:-./outputs/retrieval/gallery_results}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}"

mkdir -p "${RUNS_ROOT}" "${RESULTS_DIR}" logs

python ./retrieval/collect/run_plan_ood.py \
  --plan_dir "${PLAN_DIR}" \
  --plan_files "${PLAN_FILES}" \
  --train_script "${TRAIN_SCRIPT}" \
  --runs_root "${RUNS_ROOT}" \
  --results_dir "${RESULTS_DIR}" \
  --num_shards "${NUM_SHARDS}" \
  --shard_id "${SHARD_ID}"
