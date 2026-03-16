import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Create fixed-hyperparameter few-shot LoRA training plans. "
            "Only the sampled train subset changes across runs."
        )
    )
    ap.add_argument("--out-dir", type=str, default="./retrieval/plans/fewshot_queries")
    ap.add_argument("--datasets", type=str, default="arc-challenge,boolq,gsm8k,mbpp")
    ap.add_argument("--shots", type=str, default="1,8,16,64,128,256")
    ap.add_argument("--runs-per-shot", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for subset-seed generation.")

    ap.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--target-modules", type=str, default="q_proj,v_proj")
    ap.add_argument(
        "--train-seed",
        type=int,
        default=42,
        help="Training seed kept fixed across runs so variation comes from sampled data only.",
    )
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    datasets = parse_csv_list(args.datasets)
    shots = parse_int_list(args.shots)
    rng = np.random.default_rng(args.seed)

    summary_rows: list[dict] = []
    for dataset in datasets:
        rows: list[dict] = []
        run_id = 0
        for shot in shots:
            for sample_id in range(int(args.runs_per_shot)):
                subset_seed = int(rng.integers(0, 2**31 - 1))
                rows.append(
                    {
                        "run_id": run_id,
                        "base_model": args.base_model,
                        "dataset": dataset,
                        "lr": float(args.lr),
                        "epochs": int(args.epochs),
                        "lora_r": int(args.lora_r),
                        "lora_alpha": int(args.lora_alpha),
                        "lora_dropout": float(args.lora_dropout),
                        "max_len": int(args.max_len),
                        "batch_size": int(args.batch_size),
                        "grad_accum": int(args.grad_accum),
                        "seed": int(args.train_seed),
                        "target_modules": str(args.target_modules),
                        "weight_decay": float(args.weight_decay),
                        "train_samples": int(shot),
                        "subset_seed": subset_seed,
                        "sample_id": int(sample_id),
                    }
                )
                run_id += 1

        df = pd.DataFrame(rows)
        plan_path = out_dir / f"plan_fewshot_{dataset.replace('-', '_')}.csv"
        df.to_csv(plan_path, index=False)
        summary_rows.append(
            {
                "dataset": dataset,
                "num_rows": int(len(df)),
                "shots": ",".join(str(x) for x in shots),
                "runs_per_shot": int(args.runs_per_shot),
                "path": str(plan_path),
            }
        )
        print(f"[OK] wrote {plan_path} rows={len(df)}")

    summary_path = out_dir / "fewshot_plan_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    config_path = out_dir / "fewshot_plan_config.json"
    config_path.write_text(
        json.dumps(
            {
                "datasets": datasets,
                "shots": shots,
                "runs_per_shot": int(args.runs_per_shot),
                "base_model": args.base_model,
                "lr": float(args.lr),
                "epochs": int(args.epochs),
                "lora_r": int(args.lora_r),
                "lora_alpha": int(args.lora_alpha),
                "lora_dropout": float(args.lora_dropout),
                "max_len": int(args.max_len),
                "batch_size": int(args.batch_size),
                "grad_accum": int(args.grad_accum),
                "weight_decay": float(args.weight_decay),
                "target_modules": str(args.target_modules),
                "train_seed": int(args.train_seed),
                "seed": int(args.seed),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote {config_path}")


if __name__ == "__main__":
    main()
