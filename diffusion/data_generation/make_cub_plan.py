import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


CUB_DIR_PATTERN = re.compile(r"^cub_(\d+)$")


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def iter_cub_dirs(instance_root: Path) -> list[tuple[int, Path]]:
    items: list[tuple[int, Path]] = []
    for path in instance_root.iterdir():
        if not path.is_dir():
            continue
        match = CUB_DIR_PATTERN.match(path.name)
        if match:
            items.append((int(match.group(1)), path))
    items.sort(key=lambda item: item[0])
    return items


def first_image_name(instance_dir: Path) -> str:
    files = sorted(path.name for path in instance_dir.iterdir() if path.is_file())
    return files[0] if files else ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a CUB diffusion LoRA plan.")
    ap.add_argument("--instance_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./data_generation/plans")
    ap.add_argument("--out_name", type=str, default="cub_plan.csv")
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
    ap.add_argument("--max_train_steps_choices", type=str, default="100,150,200,250")
    ap.add_argument(
        "--prompt_choices",
        type=str,
        default="a photo of a sks bird,a sks bird,a photo of a bird,a close-up of a sks bird",
    )
    args = ap.parse_args()

    instance_root = Path(args.instance_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / args.out_name

    lrs = [float(x) for x in parse_csv(args.learning_rates)]
    grad_accum_choices = [int(x) for x in parse_csv(args.gradient_accumulation_choices)]
    max_train_steps_choices = [int(x) for x in parse_csv(args.max_train_steps_choices)]
    prompt_choices = parse_csv(args.prompt_choices)

    rng = random.Random(args.seed)
    rows: list[dict[str, object]] = []
    for run_id, (image_id, instance_dir) in enumerate(iter_cub_dirs(instance_root)):
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
                "cub_folder": instance_dir.name,
                "image_id": image_id,
                "image_name": first_image_name(instance_dir),
                "instance_data_dir": str(instance_dir),
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
                "run_name": f"model_{run_seed}_{lr_idx}_{step_idx}_{ga_idx}_{prompt_idx}_{image_id}",
            }
        )

    pd.DataFrame(rows).to_csv(plan_path, index=False)
    config_path = out_dir / f"{Path(args.out_name).stem}_config.json"
    config_path.write_text(
        json.dumps(
            {
                "instance_root": str(instance_root),
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
    print(f"[OK] wrote {config_path}")


if __name__ == "__main__":
    main()
