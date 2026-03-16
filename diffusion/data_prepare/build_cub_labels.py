import argparse
from pathlib import Path

import pandas as pd


def load_images_table(cub_root: Path) -> pd.DataFrame:
    df = pd.read_csv(
        cub_root / "images.txt",
        sep=r"\s+",
        header=None,
        names=["image_id", "rel_path"],
        engine="python",
    )
    df["image_id"] = df["image_id"].astype(int)
    df["rel_path"] = df["rel_path"].astype(str)
    df["image_name"] = df["rel_path"].map(lambda p: Path(p).name)
    df["cub_folder"] = df["image_id"].map(lambda image_id: f"cub_{image_id}")
    return df.sort_values("image_id").reset_index(drop=True)


def load_attribute_names(cub_root: Path) -> pd.DataFrame:
    df = pd.read_csv(
        cub_root / "attributes" / "attributes.txt",
        sep=r"\s+",
        header=None,
        names=["attr_id", "attribute_name"],
        engine="python",
    )
    df["attr_id"] = df["attr_id"].astype(int)
    df["attribute_name"] = (
        df["attribute_name"].astype(str).str.replace(r"\s+", "_", regex=True)
    )
    return df.sort_values("attr_id").reset_index(drop=True)


def load_image_attributes(cub_root: Path) -> pd.DataFrame:
    df = pd.read_csv(
        cub_root / "attributes" / "image_attribute_labels.txt",
        sep=r"\s+",
        header=None,
        usecols=[0, 1, 2],
        names=["image_id", "attr_id", "is_present"],
        engine="python",
    )
    df["image_id"] = df["image_id"].astype(int)
    df["attr_id"] = df["attr_id"].astype(int)
    df["is_present"] = df["is_present"].astype(int)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build image-level CUB attribute labels.")
    ap.add_argument("--cub_root", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="./data_prepare/outputs/cub_labels.csv")
    ap.add_argument(
        "--instance_root",
        type=str,
        default="",
        help="Optional root with cub_*/ folders to keep only prepared instances.",
    )
    args = ap.parse_args()

    cub_root = Path(args.cub_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    images_df = load_images_table(cub_root)
    attr_names_df = load_attribute_names(cub_root)
    image_attrs_df = load_image_attributes(cub_root)

    wide_df = image_attrs_df.pivot_table(
        index="image_id",
        columns="attr_id",
        values="is_present",
        aggfunc="max",
        fill_value=0,
    )
    wide_df = wide_df.reindex(columns=attr_names_df["attr_id"].tolist(), fill_value=0)
    wide_df.columns = attr_names_df["attribute_name"].tolist()

    out_df = images_df.merge(wide_df, left_on="image_id", right_index=True, how="left")
    attr_cols = attr_names_df["attribute_name"].tolist()
    out_df[attr_cols] = out_df[attr_cols].fillna(0).astype(int)

    if args.instance_root:
        instance_root = Path(args.instance_root)
        keep_folders = {
            path.name
            for path in instance_root.iterdir()
            if path.is_dir() and path.name.startswith("cub_")
        }
        out_df = out_df[out_df["cub_folder"].isin(keep_folders)].copy()

    out_df.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv} rows={len(out_df)}")


if __name__ == "__main__":
    main()
