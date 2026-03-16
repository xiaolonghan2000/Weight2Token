import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser("Merge GoEmotions metadata CSV shards.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="metadata_gpu_*.csv")
    parser.add_argument("--output_csv", type=str, default="./classification/lora_label_info.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    csv_paths = sorted(input_dir.glob(args.pattern))
    if not csv_paths:
        raise FileNotFoundError(f"No files matched {args.pattern} under {input_dir}")

    frames = [pd.read_csv(path) for path in csv_paths]
    merged = pd.concat(frames, ignore_index=True)
    if "run_id" in merged.columns:
        merged = merged.sort_values("run_id").reset_index(drop=True)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"[OK] merged {len(csv_paths)} files into {output_path}")


if __name__ == "__main__":
    main()
