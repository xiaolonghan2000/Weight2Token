import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Group CelebA images by identity and sample a fixed number per identity."
    )
    ap.add_argument("--img_dir", type=str, required=True)
    ap.add_argument("--identity_file", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--images_per_identity", type=int, default=21)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of creating symlinks.",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    img_dir = Path(args.img_dir)
    identity_file = Path(args.identity_file)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    img_to_id: dict[str, int] = {}
    with identity_file.open("r", encoding="utf-8") as f:
        for line in f:
            name, identity_id = line.strip().split()
            img_to_id[name] = int(identity_id)

    id_to_imgs: dict[int, list[Path]] = defaultdict(list)
    missing = 0
    for name, identity_id in img_to_id.items():
        path = img_dir / name
        if path.exists():
            id_to_imgs[identity_id].append(path)
        else:
            missing += 1

    valid_ids = sorted(identity_id for identity_id, imgs in id_to_imgs.items() if imgs)
    mapping_path = out_root / "mapping.csv"
    with mapping_path.open("w", encoding="utf-8") as f:
        f.write("celeb_k,identity_id,num_images_available,num_images_chosen,unique_chosen\n")
        for celeb_index, identity_id in enumerate(valid_ids):
            images = id_to_imgs[identity_id]
            if len(images) >= args.images_per_identity:
                chosen = rng.sample(images, args.images_per_identity)
            else:
                chosen = rng.choices(images, k=args.images_per_identity)

            celeb_dir = out_root / f"celeb_{celeb_index}"
            celeb_dir.mkdir(exist_ok=True)
            seen: dict[int, int] = defaultdict(int)

            for src in chosen:
                image_num = int(src.stem)
                repeat_id = seen[image_num]
                seen[image_num] += 1
                filename = f"{image_num}.jpg" if repeat_id == 0 else f"{image_num}_r{repeat_id}.jpg"
                dst = celeb_dir / filename
                if dst.exists():
                    continue
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    os.symlink(src.resolve(), dst)

            f.write(
                f"{celeb_index},{identity_id},{len(images)},{args.images_per_identity},{len(set(chosen))}\n"
            )

    print(f"[OK] wrote {mapping_path}")
    print(f"[INFO] identities with images: {len(valid_ids)}")
    print(f"[INFO] missing source images: {missing}")


if __name__ == "__main__":
    main()
