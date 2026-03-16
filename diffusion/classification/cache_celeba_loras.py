import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from w2t_utils import get_canonical_data_with_meta


MODEL_ID_PATTERN = re.compile(r"_(\d+)$")


def celeb_key_from_model_name(model_name: str) -> str:
    match = MODEL_ID_PATTERN.search(model_name)
    if not match:
        raise ValueError(f"Could not infer celeb id from model directory name: {model_name}")
    return f"celeb_{int(match.group(1))}"


def load_label_map(labels_csv: Path) -> dict[str, torch.Tensor]:
    df = pd.read_csv(labels_csv)
    meta_cols = {"celeb_folder", "num_images_in_folder", "num_unique_source_images"}
    attr_cols = [c for c in df.columns if c not in meta_cols]
    label_map = {}
    for _, row in df.iterrows():
        values = row[attr_cols].to_numpy()
        if set(np.unique(values)).issubset({0, 1}):
            label = torch.tensor(values.astype(np.int64), dtype=torch.bool)
        else:
            label = torch.tensor(values, dtype=torch.float32) >= 0.0
        label_map[str(row["celeb_folder"])] = label
    return label_map


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a canonical CelebA LoRA cache for W2T.")
    ap.add_argument("--lora_root", type=str, required=True)
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    args = ap.parse_args()

    lora_root = Path(args.lora_root)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    label_map = load_label_map(Path(args.labels_csv))

    items = []
    for model_dir in tqdm(sorted(lora_root.iterdir()), desc="Caching LoRAs"):
        if not model_dir.is_dir():
            continue
        unet_dir = model_dir / "unet"
        if not (unet_dir / "adapter_model.safetensors").exists():
            continue
        celeb_key = celeb_key_from_model_name(model_dir.name)
        if celeb_key not in label_map:
            continue
        features, meta = get_canonical_data_with_meta(str(unet_dir))
        if features is None:
            continue
        items.append(
            {
                "name": model_dir.name,
                "features": features,
                "meta": meta,
                "label": label_map[celeb_key],
            }
        )

    torch.save(items, out_path)
    print(f"[OK] wrote {out_path} samples={len(items)}")


if __name__ == "__main__":
    main()
