import argparse
import json
import os

import pandas as pd
import torch
from tqdm import tqdm

from w2t_utils import get_canonical_data_with_meta


def make_items(csv_path: str):
    df = pd.read_csv(csv_path)
    df["safetensors_path"] = df["safetensors_path"].astype(str).str.strip()
    if "safetensors_filename" not in df.columns:
        df["safetensors_filename"] = ""
    df["safetensors_filename"] = df["safetensors_filename"].astype(str).str.strip()
    df["name"] = df["run_id"].apply(lambda x: f"run_{int(x)}")

    def fullpath(row):
        path = row["safetensors_path"]
        filename = row["safetensors_filename"]
        if path.endswith(".safetensors") or not filename:
            return path
        return os.path.join(path, filename)

    df["safetensors_fullpath"] = df.apply(fullpath, axis=1)
    df["label_list"] = df["label_vector"].apply(json.loads)

    items = []
    for _, row in df.iterrows():
        items.append({"name": row["name"], "path": row["safetensors_fullpath"], "label": row["label_list"]})
    return items


def slice_items(items, part_id: int, num_parts: int):
    n_items = len(items)
    start = (n_items * part_id) // num_parts
    end = (n_items * (part_id + 1)) // num_parts
    return items[start:end], start, end


def main():
    parser = argparse.ArgumentParser("GoEmotions cache builder.")
    parser.add_argument(
        "--csv",
        type=str,
        default="./classification/lora_label_info.csv",
        help="CSV with columns: run_id, safetensors_path, optional safetensors_filename, label_vector",
    )
    parser.add_argument("--out_path", type=str, default="./classification/cache/goemo_part.pt")
    parser.add_argument("--part_id", type=int, required=True)
    parser.add_argument("--num_parts", type=int, required=True)
    parser.add_argument("--resume", type=int, default=1)
    args = parser.parse_args()

    if args.resume == 1 and os.path.exists(args.out_path):
        try:
            old = torch.load(args.out_path, map_location="cpu")
            if isinstance(old, list) and len(old) > 0:
                print(f"[RESUME] {args.out_path} exists ({len(old)} items). Skip.")
                return
        except Exception:
            print(f"[RESUME] {args.out_path} exists but is unreadable. Rebuilding.")

    items = make_items(args.csv)
    part, start, end = slice_items(items, args.part_id, args.num_parts)
    print(f"[SPLIT] part {args.part_id}/{args.num_parts} idx [{start}:{end}) -> {len(part)} items")

    out_parent = os.path.dirname(args.out_path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    records = []
    fail = {"missing": 0, "parse_fail": 0, "empty": 0}

    for item in tqdm(part, desc=f"Caching part {args.part_id}"):
        path = item["path"]
        if not os.path.exists(path):
            fail["missing"] += 1
            continue

        try:
            feats, meta = get_canonical_data_with_meta(path)
        except Exception:
            fail["parse_fail"] += 1
            continue

        if feats is None or len(feats) == 0:
            fail["empty"] += 1
            continue

        feats_np = []
        for (u, v, s) in feats:
            feats_np.append((u.detach().cpu().numpy(), v.detach().cpu().numpy(), s.detach().cpu().numpy()))

        records.append({"name": item["name"], "features": feats_np, "meta": meta, "label": item["label"]})

    torch.save(records, args.out_path)
    print(f"[DONE] saved {len(records)} items -> {args.out_path}")
    print(f"[DONE] fail stats: {fail}")


if __name__ == "__main__":
    main()
