import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def sanitize_name(s: str) -> str:
    x = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s).strip())
    return x.strip("_") or "unknown"


def resolve_plan_paths(plan_dir: str, plan_files: str) -> list[str]:
    tokens = [x.strip() for x in (plan_files or "").split(",") if x.strip()]
    if not tokens:
        tokens = ["plan.csv"]

    out = []
    for token in tokens:
        pattern = token if os.path.isabs(token) else os.path.join(plan_dir, token)
        matches = sorted(glob.glob(pattern))
        if matches:
            out.extend(matches)
            continue
        if os.path.exists(pattern):
            out.append(pattern)
            continue
        raise FileNotFoundError(f"plan file/pattern not found: {token} (resolved: {pattern})")

    uniq = []
    seen = set()
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def row_get(row: pd.Series, key: str, default):
    if key in row and pd.notna(row[key]):
        return row[key]
    return default


def load_done_ids(results_csv: str) -> set[int]:
    if not os.path.exists(results_csv):
        return set()
    done_ids = set()
    try:
        with open(results_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "run_id" not in reader.fieldnames:
                return set()
            for row in reader:
                raw = row.get("run_id", "")
                if raw is None:
                    continue
                s = str(raw).strip()
                if not s:
                    continue
                try:
                    done_ids.add(int(float(s)))
                except (TypeError, ValueError):
                    continue
    except (OSError, csv.Error):
        # Fallback for partially corrupted CSV rows: scan line by line and extract leading run_id.
        try:
            with open(results_csv, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i == 0:
                        continue
                    m = re.match(r"\s*([0-9]+)\s*,", line)
                    if m:
                        done_ids.add(int(m.group(1)))
        except OSError:
            return set()
    return done_ids


def infer_plan_dataset(df: pd.DataFrame, plan_path: str) -> str:
    if "dataset" not in df.columns:
        raise ValueError(f"{plan_path} missing column: dataset")
    vals = sorted({str(x) for x in df["dataset"].dropna().unique().tolist()})
    if not vals:
        raise ValueError(f"{plan_path} has empty dataset column")
    if len(vals) > 1:
        raise ValueError(f"{plan_path} contains multiple datasets: {vals}")
    return vals[0]


def main():
    ap = argparse.ArgumentParser(
        description=(
            "OOD plan runner with deterministic trial sharding. "
            "Use run_id %% num_shards == shard_id to parallelize one dataset across many GPUs."
        )
    )
    ap.add_argument("--plan_dir", type=str, required=True)
    ap.add_argument(
        "--plan_files",
        type=str,
        default="plan.csv",
        help="comma-separated names or glob patterns under --plan_dir",
    )
    ap.add_argument(
        "--train_script",
        type=str,
        default=str(SCRIPT_DIR.parent.parent / "regression" / "collect" / "train_lora_llm.py"),
    )
    ap.add_argument("--runs_root", type=str, required=True)
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--hf_cache_dir", type=str, default="")
    ap.add_argument("--data_cache_dir", type=str, default="")
    ap.add_argument("--map_num_proc", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--eval_acc_mode", type=str, choices=["full", "subset", "none"], default="full")
    ap.add_argument("--eval_acc_samples", type=int, default=512)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--max_trials", type=int, default=-1)
    ap.add_argument("--sleep_on_fail", type=int, default=5)
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"--shard_id must be in [0, {args.num_shards - 1}]")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.runs_root, exist_ok=True)

    plan_paths = resolve_plan_paths(args.plan_dir, args.plan_files)
    print(f"[INFO] plans={len(plan_paths)} files={plan_paths}")

    required_cols = {
        "run_id",
        "base_model",
        "dataset",
        "lr",
        "epochs",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "max_len",
        "batch_size",
        "grad_accum",
        "seed",
        "weight_decay",
    }

    total_launched = 0
    total_skipped = 0
    total_failed = 0

    for plan_path in plan_paths:
        df = pd.read_csv(plan_path)
        missing = sorted(required_cols - set(df.columns))
        if missing:
            raise ValueError(f"{plan_path} missing required columns: {missing}")
        if df["run_id"].duplicated().any():
            dup = df[df["run_id"].duplicated()]["run_id"].head(5).tolist()
            raise ValueError(f"{plan_path} has duplicated run_id(s), e.g. {dup}")

        dataset = infer_plan_dataset(df, plan_path)
        dataset_safe = sanitize_name(dataset)
        runs_root = os.path.join(args.runs_root, dataset_safe)
        global_results_csv = os.path.join(args.results_dir, f"results_{dataset_safe}.csv")
        if args.num_shards > 1:
            results_csv = os.path.join(
                args.results_dir,
                f"results_{dataset_safe}_shard{args.shard_id:02d}of{args.num_shards:02d}.csv",
            )
        else:
            results_csv = global_results_csv

        if not args.dry_run:
            os.makedirs(runs_root, exist_ok=True)

        if args.no_resume:
            done_ids = set()
        else:
            done_ids = load_done_ids(results_csv)
            # Allow smooth resume after switching from single-shard to multi-shard mode.
            if args.num_shards > 1:
                done_ids |= load_done_ids(global_results_csv)
        print(
            f"[PLAN] file={os.path.basename(plan_path)} dataset={dataset} rows={len(df)} "
            f"shard={args.shard_id}/{args.num_shards} done={len(done_ids)} results={results_csv}"
        )

        n_plan = 0
        n_filtered = 0
        for _, row in df.iterrows():
            run_id = int(row["run_id"])
            if run_id % args.num_shards != args.shard_id:
                n_filtered += 1
                continue
            if run_id in done_ids:
                total_skipped += 1
                continue

            run_name = f"trial_{run_id:06d}"
            cmd = [
                sys.executable,
                args.train_script,
                "--base_model",
                str(row["base_model"]),
                "--dataset",
                dataset,
                "--out_dir",
                runs_root,
                "--run_name",
                run_name,
                "--results_csv",
                results_csv,
                "--lr",
                str(float(row["lr"])),
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
                str(row_get(row, "target_modules", "q_proj,v_proj")),
                "--weight_decay",
                str(float(row_get(row, "weight_decay", 0.0))),
                "--map_num_proc",
                str(int(args.map_num_proc)),
                "--num_workers",
                str(int(args.num_workers)),
                "--prefetch_factor",
                str(int(args.prefetch_factor)),
                "--eval_acc_mode",
                str(args.eval_acc_mode),
                "--eval_acc_samples",
                str(int(args.eval_acc_samples)),
            ]
            if args.hf_cache_dir:
                cmd.extend(["--hf_cache_dir", args.hf_cache_dir])
            if args.data_cache_dir:
                cmd.extend(["--data_cache_dir", args.data_cache_dir])
            if "train_samples" in df.columns and pd.notna(row_get(row, "train_samples", None)):
                cmd.extend(["--train-samples", str(int(row_get(row, "train_samples", -1)))])
            if "subset_seed" in df.columns and pd.notna(row_get(row, "subset_seed", None)):
                cmd.extend(["--subset-seed", str(int(row_get(row, "subset_seed", 42)))])

            print(
                f"[RUN] dataset={dataset} run_id={run_id} "
                f"train_samples={row_get(row, 'train_samples', 'full')} "
                f"subset_seed={row_get(row, 'subset_seed', 'na')}"
            )
            if args.dry_run:
                total_launched += 1
                n_plan += 1
            else:
                rc = subprocess.call(cmd)
                if rc != 0:
                    total_failed += 1
                    print(f"[WARN] run_id={run_id} failed rc={rc}, sleep {args.sleep_on_fail}s")
                    time.sleep(args.sleep_on_fail)
                    continue
                done_ids.add(run_id)
                total_launched += 1
                n_plan += 1

            if args.max_trials > 0 and n_plan >= args.max_trials:
                print(f"[INFO] reach max_trials={args.max_trials} for dataset={dataset}")
                break
        print(
            f"[PLAN DONE] dataset={dataset} shard={args.shard_id}/{args.num_shards} "
            f"launched={n_plan} filtered={n_filtered}"
        )

    print(
        f"[DONE] shard={args.shard_id}/{args.num_shards} launched={total_launched} "
        f"skipped={total_skipped} failed={total_failed} dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
