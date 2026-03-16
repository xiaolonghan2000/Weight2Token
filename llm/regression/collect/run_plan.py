import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
if "SLURM_ARRAY_TASK_COUNT" in os.environ:
    num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
else:
    task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "0"))
    task_max = int(os.environ.get("SLURM_ARRAY_TASK_MAX", "0"))
    num_tasks = (task_max - task_min + 1) if task_max >= task_min else 1


print(f"[SLURM] task_id={task_id} / num_tasks={num_tasks}")


def belongs(run_id: int) -> bool:
    return (run_id % num_tasks) == task_id


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the ARC-Easy full-train LoRA plan with optional SLURM sharding."
    )
    ap.add_argument("--plan_dir", type=str, required=True)
    ap.add_argument(
        "--train_script",
        type=str,
        default=str(SCRIPT_DIR / "train_lora_llm.py"),
    )
    ap.add_argument("--runs_root", type=str, required=True)
    ap.add_argument("--results_csv", type=str, required=True)
    ap.add_argument("--max_trials", type=int, default=-1)
    ap.add_argument("--sleep_on_fail", type=int, default=5)
    args = ap.parse_args()

    plan_path = Path(args.plan_dir) / "plan.csv"
    df = pd.read_csv(plan_path)

    done = set()
    if os.path.exists(args.results_csv):
        prev = pd.read_csv(args.results_csv)
        if "run_id" in prev.columns:
            done = set(prev["run_id"].dropna().astype(int).tolist())
    print(f"[INFO] plan={len(df)} done={len(done)} remaining={len(df) - len(done)}")

    runs_completed = 0
    for _, row in df.iterrows():
        run_id = int(row["run_id"])
        if run_id in done or not belongs(run_id):
            continue

        run_name = f"trial_{run_id:06d}"
        os.makedirs(os.path.join(args.runs_root, run_name), exist_ok=True)
        cmd = [
            sys.executable,
            args.train_script,
            "--base_model",
            str(row["base_model"]),
            "--dataset",
            str(row["dataset"]),
            "--out_dir",
            args.runs_root,
            "--run_name",
            run_name,
            "--results_csv",
            args.results_csv,
            "--lr",
            str(row["lr"]),
            "--epochs",
            str(int(row["epochs"])),
            "--lora_r",
            str(int(row["lora_r"])),
            "--lora_alpha",
            str(int(row["lora_alpha"])),
            "--lora_dropout",
            str(float(row["lora_dropout"])),
            "--max_len",
            str(int(row["max_len"])),
            "--batch_size",
            str(int(row["batch_size"])),
            "--grad_accum",
            str(int(row["grad_accum"])),
            "--seed",
            str(int(row["seed"])),
            "--target_modules",
            str(row.get("target_modules", "q_proj,v_proj")),
            "--weight_decay",
            str(row["weight_decay"]),
        ]

        print(f"\n[RUN] {run_id} -> {run_name}")
        print(" ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            print(
                f"[WARN] run_id={run_id} failed rc={rc}, retrying next run after {args.sleep_on_fail}s"
            )
            time.sleep(args.sleep_on_fail)
            continue

        runs_completed += 1
        if args.max_trials > 0 and runs_completed >= args.max_trials:
            break

    print("[DONE]")


if __name__ == "__main__":
    main()
