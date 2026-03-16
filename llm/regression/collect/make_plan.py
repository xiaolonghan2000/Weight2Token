import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create an ARC-Easy full-train LoRA plan."
    )
    ap.add_argument("--out-dir", type=str, default="./regression/plans")
    ap.add_argument("--out-name", type=str, default="plan.csv")
    ap.add_argument("--num-trials", type=int, default=10000)
    ap.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--dataset", type=str, default="arc-easy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr-min", type=float, default=5e-6)
    ap.add_argument("--lr-max", type=float, default=5e-4)
    ap.add_argument("--epochs-min", type=int, default=1)
    ap.add_argument("--epochs-max", type=int, default=4)
    ap.add_argument("--r-choices", type=str, default="8")
    ap.add_argument("--alpha-choices", type=str, default="16")
    ap.add_argument("--dropout-min", type=float, default=0.0)
    ap.add_argument("--dropout-max", type=float, default=0.1)
    ap.add_argument("--weight-decay-min", type=float, default=0.0)
    ap.add_argument("--weight-decay-max", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--target-modules", type=str, default="q_proj,v_proj")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / args.out_name

    rng = np.random.default_rng(args.seed)
    r_choices = parse_int_csv(args.r_choices)
    alpha_choices = parse_int_csv(args.alpha_choices)

    rows: list[dict] = []
    for run_id in range(int(args.num_trials)):
        rows.append(
            {
                "run_id": run_id,
                "base_model": args.base_model,
                "dataset": args.dataset,
                "lr": log_uniform(rng, args.lr_min, args.lr_max),
                "epochs": int(rng.integers(args.epochs_min, args.epochs_max + 1)),
                "lora_r": int(rng.choice(r_choices)),
                "lora_alpha": int(rng.choice(alpha_choices)),
                "lora_dropout": float(rng.uniform(args.dropout_min, args.dropout_max)),
                "max_len": int(args.max_len),
                "batch_size": int(args.batch_size),
                "grad_accum": int(args.grad_accum),
                "seed": int(rng.integers(0, 2**31 - 1)),
                "target_modules": str(args.target_modules),
                "weight_decay": float(
                    rng.uniform(args.weight_decay_min, args.weight_decay_max)
                ),
            }
        )

    pd.DataFrame(rows).to_csv(plan_path, index=False)
    config_path = out_dir / "plan_config.json"
    config_path.write_text(
        json.dumps(
            {
                "out_name": args.out_name,
                "num_trials": int(args.num_trials),
                "base_model": args.base_model,
                "dataset": args.dataset,
                "seed": int(args.seed),
                "lr_min": float(args.lr_min),
                "lr_max": float(args.lr_max),
                "epochs_min": int(args.epochs_min),
                "epochs_max": int(args.epochs_max),
                "r_choices": r_choices,
                "alpha_choices": alpha_choices,
                "dropout_min": float(args.dropout_min),
                "dropout_max": float(args.dropout_max),
                "weight_decay_min": float(args.weight_decay_min),
                "weight_decay_max": float(args.weight_decay_max),
                "max_len": int(args.max_len),
                "batch_size": int(args.batch_size),
                "grad_accum": int(args.grad_accum),
                "target_modules": str(args.target_modules),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] wrote {plan_path} rows={len(rows)}")
    print(f"[OK] wrote {config_path}")


if __name__ == "__main__":
    main()
