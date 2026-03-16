import argparse
import csv
import os
import shutil
from pathlib import Path

import pandas as pd


def read_images_table(cub_root: Path) -> pd.DataFrame:
    images_txt = cub_root / "images.txt"
    images_dir = cub_root / "images"
    df = pd.read_csv(
        images_txt,
        sep=r"\s+",
        header=None,
        names=["image_id", "rel_path"],
        engine="python",
    )
    df["image_id"] = df["image_id"].astype(int)
    df["rel_path"] = df["rel_path"].astype(str)
    df["image_name"] = df["rel_path"].map(lambda p: Path(p).name)
    df["src_path"] = df["rel_path"].map(lambda p: str(images_dir / p))
    df["cub_folder"] = df["image_id"].map(lambda image_id: f"cub_{image_id}")
    return df.sort_values("image_id").reset_index(drop=True)


def materialize_image(src_path: Path, dst_path: Path, copy_mode: str, overwrite: bool) -> None:
    if dst_path.exists():
        if not overwrite:
            return
        dst_path.unlink()

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "copy":
        shutil.copy2(src_path, dst_path)
        return
    if copy_mode == "hardlink":
        os.link(src_path, dst_path)
        return
    if copy_mode == "symlink":
        dst_path.symlink_to(src_path.resolve())
        return
    raise ValueError(f"Unsupported copy mode: {copy_mode}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create one-folder-per-image CUB instances for diffusion LoRA training."
    )
    ap.add_argument("--cub_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument(
        "--copy_mode",
        type=str,
        choices=["copy", "hardlink", "symlink"],
        default="copy",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--manifest_name", type=str, default="cub_instances.csv")
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    cub_root = Path(args.cub_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    df = read_images_table(cub_root)
    if args.limit > 0:
        df = df.iloc[: args.limit].copy()

    manifest_rows: list[dict[str, str | int]] = []
    for row in df.itertuples(index=False):
        src_path = Path(row.src_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Missing image file: {src_path}")

        dst_dir = out_root / str(row.cub_folder)
        dst_path = dst_dir / str(row.image_name)
        materialize_image(src_path, dst_path, args.copy_mode, args.overwrite)
        manifest_rows.append(
            {
                "cub_folder": str(row.cub_folder),
                "image_id": int(row.image_id),
                "image_name": str(row.image_name),
                "rel_path": str(row.rel_path),
                "src_path": str(src_path),
                "instance_data_dir": str(dst_dir),
                "instance_image_path": str(dst_path),
            }
        )

    manifest_path = out_root / args.manifest_name
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [])
        if manifest_rows:
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"[OK] wrote {len(manifest_rows)} CUB instance folders under {out_root}")
    print(f"[OK] wrote {manifest_path}")


if __name__ == "__main__":
    main()
