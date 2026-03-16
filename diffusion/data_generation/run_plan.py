import argparse
import csv
import os
import subprocess
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


def belongs(run_id: int) -> bool:
    return (run_id % num_tasks) == task_id


def load_done_ids(results_csv: Path) -> set[int]:
    if not results_csv.exists():
        return set()
    done = set()
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "run_id" not in reader.fieldnames:
            return set()
        for row in reader:
            raw = row.get("run_id", "")
            if raw == "":
                continue
            done.add(int(float(raw)))
    return done


def append_result(results_csv: Path, row: dict) -> None:
    write_header = not results_csv.exists()
    with results_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run a diffusion LoRA plan with optional SLURM sharding."
    )
    ap.add_argument("--plan_csv", type=str, required=True)
    ap.add_argument("--runs_root", type=str, required=True)
    ap.add_argument("--results_csv", type=str, required=True)
    ap.add_argument("--accelerate_cmd", type=str, default="accelerate")
    ap.add_argument(
        "--train_script",
        type=str,
        default=str(SCRIPT_DIR / "train_dreambooth.py"),
    )
    ap.add_argument("--mixed_precision", type=str, default="fp16")
    ap.add_argument("--num_dataloader_workers", type=int, default=1)
    ap.add_argument("--max_trials", type=int, default=-1)
    ap.add_argument("--sleep_on_fail", type=int, default=5)
    args = ap.parse_args()

    plan_df = pd.read_csv(args.plan_csv)
    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    results_csv = Path(args.results_csv)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(results_csv)

    runs_completed = 0
    for _, row in plan_df.iterrows():
        run_id = int(row["run_id"])
        if run_id in done or not belongs(run_id):
            continue

        run_name = str(row["run_name"])
        output_dir = runs_root / run_name
        adapter_model = output_dir / "unet" / "adapter_model.safetensors"
        if adapter_model.exists():
            append_result(
                results_csv,
                {
                    **row.to_dict(),
                    "output_dir": str(output_dir),
                    "adapter_model_path": str(adapter_model),
                    "status": "skipped_existing",
                },
            )
            continue

        cmd = [
            args.accelerate_cmd,
            "launch",
            args.train_script,
            "--pretrained_model_name_or_path",
            str(row["pretrained_model_name_or_path"]),
            "--instance_data_dir",
            str(row["instance_data_dir"]),
            "--output_dir",
            str(output_dir),
            "--num_dataloader_workers",
            str(args.num_dataloader_workers),
            "--instance_prompt",
            str(row["instance_prompt"]),
            "--resolution",
            str(int(row["resolution"])),
            "--train_batch_size",
            str(int(row["train_batch_size"])),
            "--lr_scheduler",
            "constant",
            "--mixed_precision",
            args.mixed_precision,
            "--lr_warmup_steps",
            "0",
            "--report_to",
            "none",
            "--use_lora",
            "--lora_r",
            str(int(row["lora_r"])),
            "--lora_alpha",
            str(int(row["lora_alpha"])),
            "--lora_dropout",
            str(float(row.get("lora_dropout", 0.0))),
            "--learning_rate",
            str(row["learning_rate"]),
            "--gradient_accumulation_steps",
            str(int(row["gradient_accumulation_steps"])),
            "--max_train_steps",
            str(int(row["max_train_steps"])),
            "--seed",
            str(int(row["seed"])),
            "--no_tracemalloc",
        ]

        print(f"[RUN] {run_id} -> {run_name}")
        print(" ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[WARN] run_id={run_id} failed rc={rc}")
            time.sleep(args.sleep_on_fail)
            continue

        append_result(
            results_csv,
            {
                **row.to_dict(),
                "output_dir": str(output_dir),
                "adapter_model_path": str(adapter_model),
                "status": "completed",
            },
        )
        runs_completed += 1
        if args.max_trials > 0 and runs_completed >= args.max_trials:
            break

    print("[DONE]")


if __name__ == "__main__":
    main()
