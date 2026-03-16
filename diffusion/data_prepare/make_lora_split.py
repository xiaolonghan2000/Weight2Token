import argparse
import json
import random
from pathlib import Path

import torch


def iter_model_names(lora_root: Path) -> list[str]:
    names = []
    for path in sorted(lora_root.iterdir()):
        if not path.is_dir():
            continue
        adapter_path = path / "unet" / "adapter_model.safetensors"
        if adapter_path.exists():
            names.append(path.name)
    return names


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create checkpoint-level train/valid/test splits for diffusion LoRAs."
    )
    ap.add_argument("--lora_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    args = ap.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train_ratio must be between 0 and 1.")
    if not 0.0 <= args.valid_ratio < 1.0:
        raise ValueError("--valid_ratio must be between 0 and 1.")
    if args.train_ratio + args.valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1.")

    lora_root = Path(args.lora_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = iter_model_names(lora_root)
    rng = random.Random(args.seed)
    rng.shuffle(names)

    n_total = len(names)
    n_train = int(round(n_total * args.train_ratio))
    n_valid = int(round(n_total * args.valid_ratio))
    n_train = min(n_train, n_total)
    n_valid = min(n_valid, max(0, n_total - n_train))

    train_names = names[:n_train]
    valid_names = names[n_train:n_train + n_valid]
    test_names = names[n_train + n_valid:]

    torch.save(train_names, out_dir / "train.pt")
    torch.save(valid_names, out_dir / "valid.pt")
    torch.save(test_names, out_dir / "test.pt")
    (out_dir / "split_summary.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "num_total": n_total,
                "num_train": len(train_names),
                "num_valid": len(valid_names),
                "num_test": len(test_names),
                "train_ratio": args.train_ratio,
                "valid_ratio": args.valid_ratio,
                "test_ratio": 1.0 - args.train_ratio - args.valid_ratio,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] wrote split files under {out_dir}")


if __name__ == "__main__":
    main()
