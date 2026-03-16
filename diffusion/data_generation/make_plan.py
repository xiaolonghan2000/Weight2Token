import argparse
import json
import random
from pathlib import Path

import pandas as pd


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def iter_celeb_dirs(celeb_root: Path) -> list[Path]:
    paths = [p for p in celeb_root.iterdir() if p.is_dir() and p.name.startswith("celeb_")]
    paths.sort(key=lambda p: int(p.name.split("_")[-1]))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create a CelebA diffusion LoRA plan."
    )
    ap.add_argument("--celeb_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./data_generation/plans")
    ap.add_argument("--out_name", type=str, default="plan.csv")
    ap.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=27)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--train_batch_size", type=int, default=1)
    ap.add_argument("--learning_rates", type=str, default="1e-4,3e-4,1e-3,3e-3")
    ap.add_argument("--gradient_accumulation_choices", type=str, default="1,2,3,4")
    ap.add_argument("--max_train_steps_choices", type=str, default="100,133,167,200")
    ap.add_argument("--prompt_choices", type=str, default="celebrity,person,human,face")
    args = ap.parse_args()

    celeb_root = Path(args.celeb_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / args.out_name

    lrs = [float(x) for x in parse_csv(args.learning_rates)]
    grad_accum_choices = [int(x) for x in parse_csv(args.gradient_accumulation_choices)]
    max_train_steps_choices = [int(x) for x in parse_csv(args.max_train_steps_choices)]
    prompt_choices = parse_csv(args.prompt_choices)

    rng = random.Random(args.seed)
    rows: list[dict] = []
    for run_id, celeb_dir in enumerate(iter_celeb_dirs(celeb_root)):
        celeb_num = int(celeb_dir.name.split("_")[-1])
        seeds = rng.sample(range(10000), 4)
        lr_idx = rng.randint(0, len(lrs) - 1)
        ga_idx = rng.randint(0, len(grad_accum_choices) - 1)
        step_idx = rng.randint(0, len(max_train_steps_choices) - 1)
        prompt_idx = rng.randint(0, len(prompt_choices) - 1)
        seed_idx = rng.randint(0, len(seeds) - 1)
        run_seed = seeds[seed_idx]

        rows.append(
            {
                "run_id": run_id,
                "celeb_folder": celeb_dir.name,
                "instance_data_dir": str(celeb_dir),
                "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
                "instance_prompt": prompt_choices[prompt_idx],
                "resolution": args.resolution,
                "train_batch_size": args.train_batch_size,
                "learning_rate": lrs[lr_idx],
                "gradient_accumulation_steps": grad_accum_choices[ga_idx],
                "max_train_steps": max_train_steps_choices[step_idx],
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "seed": run_seed,
                "run_name": f"model_{run_seed}_{lr_idx}_{step_idx}_{ga_idx}_{prompt_idx}_{celeb_num}",
            }
        )

    pd.DataFrame(rows).to_csv(plan_path, index=False)
    (out_dir / "plan_config.json").write_text(
        json.dumps(
            {
                "celeb_root": str(celeb_root),
                "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
                "seed": args.seed,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "resolution": args.resolution,
                "train_batch_size": args.train_batch_size,
                "learning_rates": lrs,
                "gradient_accumulation_choices": grad_accum_choices,
                "max_train_steps_choices": max_train_steps_choices,
                "prompt_choices": prompt_choices,
                "num_rows": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] wrote {plan_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
