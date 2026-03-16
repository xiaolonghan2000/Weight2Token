import argparse
import gc
import glob
import json
import os

import torch
from tqdm import tqdm


def main(cache_dir: str, pattern: str = "cache_part_*.pt", out_name: str = "manifest.json") -> None:
    paths = sorted(glob.glob(os.path.join(cache_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No .pt files found under {cache_dir} with pattern {pattern}")

    manifest = {}
    total = 0
    for path in tqdm(paths, desc="Scanning shards"):
        data = torch.load(path, map_location="cpu", weights_only=False)
        count = len(data)
        manifest[os.path.basename(path)] = count
        total += count
        del data
        gc.collect()

    out_path = os.path.join(cache_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] wrote {out_path}, total items={total}, files={len(paths)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build a cache manifest for GoEmotions shards.")
    parser.add_argument("--cache_dir", type=str, default="./classification/cache")
    parser.add_argument("--pattern", type=str, default="cache_part_*.pt")
    parser.add_argument("--out_name", type=str, default="manifest.json")
    args = parser.parse_args()
    main(args.cache_dir, args.pattern, args.out_name)
