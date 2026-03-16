import os
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def parse_img_index(filename: str) -> int:
    """
    Convert '000123.jpg'/'123.jpg'/'123_r7.jpg' -> 0-based row index (122).
    CelebA images are 1-based in filenames.
    """
    stem = Path(filename).stem          # '123' or '123_r7' or '000123'
    stem = stem.split("_")[0]           # '123'
    img_id = int(stem)                  # 123
    return img_id - 1                   # to 0-based row


def iter_celeb_folders(root: Path):
    # celeb_0, celeb_1, ...
    pat = re.compile(r"^celeb_(\d+)$")
    items = []
    for p in root.iterdir():
        if p.is_dir():
            m = pat.match(p.name)
            if m:
                items.append((int(m.group(1)), p))
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]


def main():
    ap = argparse.ArgumentParser("Precompute CelebA celebrity labels from foldered images.")
    ap.add_argument("--celeb_root", type=str, required=True, help="Root dir containing celeb_*/ folders")
    ap.add_argument("--attr_csv", type=str, required=True, help="list_attr_celeba.csv path")
    ap.add_argument("--out_pt", type=str, default="celeb_labels.pt")
    ap.add_argument("--out_csv", type=str, default="celeb_labels.csv")
    ap.add_argument("--min_images", type=int, default=1, help="Skip celeb folders with < min_images images")
    args = ap.parse_args()

    celeb_root = Path(args.celeb_root)
    attr_csv = Path(args.attr_csv)
    out_pt = Path(args.out_pt)
    out_csv = Path(args.out_csv)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Read attributes: many versions of this file have first column = image_id or filename
    df = pd.read_csv(attr_csv)

    # If first column looks like a filename/id column, drop it
    # Common: first column is 'image_id'
    if df.columns[0].lower() in {"image_id", "img_id", "filename", "image"}:
        df_attr = df[df.columns[1:]]
    else:
        # your current training code uses df.columns[1:] anyway,
        # but this is safer if the CSV is already attribute-only
        df_attr = df

    attr_names = list(df_attr.columns)
    attr_mat = df_attr.to_numpy()  # values typically in {-1,1}

    folders = iter_celeb_folders(celeb_root)
    if len(folders) == 0:
        raise RuntimeError(f"No celeb_* folders found under: {celeb_root}")

    labels = {}
    rows_out = []

    for folder in tqdm(folders, desc="Precomputing labels"):
        imgs = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
        if len(imgs) < args.min_images:
            continue

        # Map images -> row indices in attr_mat
        idx = []
        for fn in imgs:
            try:
                idx.append(parse_img_index(fn))
            except Exception:
                # ignore weird filenames
                continue

        if len(idx) == 0:
            continue

        idx = np.array(idx, dtype=np.int64)

        # Clip to valid range (in case someone has out-of-range filename)
        idx = idx[(idx >= 0) & (idx < attr_mat.shape[0])]
        if len(idx) == 0:
            continue

        # mean over selected rows, then threshold >= 0 (matches your training logic)
        mean_attr = attr_mat[idx].mean(axis=0)
        label = torch.tensor(mean_attr >= 0.0, dtype=torch.bool)

        key = folder.name  # "celeb_123"
        labels[key] = label

        rows_out.append([key, len(imgs), len(np.unique(idx))] + label.to(torch.int64).tolist())

    # Save .pt
    torch.save(
        {"attr_names": attr_names, "labels": labels},
        out_pt
    )

    # Save .csv (0/1)
    out_cols = ["celeb_folder", "num_images_in_folder", "num_unique_source_images"] + attr_names
    out_df = pd.DataFrame(rows_out, columns=out_cols)
    out_df.to_csv(out_csv, index=False)

    print(f"[OK] wrote: {out_pt}")
    print(f"[OK] wrote: {out_csv}")
    print(f"[OK] num celebs labeled: {len(labels)}")


if __name__ == "__main__":
    main()
