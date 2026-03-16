import argparse
import json
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from w2t_models import FullTransformer
from w2t_utils import (
    CachedSVDataset,
    CachedSVDatasetDir,
    RepackedSFTDatasetDir,
    get_canonical_data_with_meta,
    print_model_stats,
    test,
    train,
)

try:
    import wandb
except Exception:
    wandb = None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_label_vector(value: str) -> torch.Tensor:
    arr = json.loads(value)
    return torch.tensor(arr, dtype=torch.float32)


def build_random_split(names, seed: int = 42, ratios=(0.8, 0.1, 0.1)):
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    idx = np.arange(len(names))
    rng.shuffle(idx)
    n = len(names)
    n_train = int(ratios[0] * n)
    n_valid = int(ratios[1] * n)
    train_names = [names[i] for i in idx[:n_train]]
    valid_names = [names[i] for i in idx[n_train : n_train + n_valid]]
    test_names = [names[i] for i in idx[n_train + n_valid :]]
    return train_names, valid_names, test_names


class GoEmoLoRADataset(torch.utils.data.Dataset):
    """Load a GoEmotions LoRA checkpoint and its multi-label target."""

    def __init__(self, df: pd.DataFrame, to_keep=None, verbose: bool = False):
        self.verbose = verbose
        if to_keep is not None:
            keep = set(to_keep)
            df = df[df["name"].isin(keep)].reset_index(drop=True)
        self.df = df
        if self.verbose:
            print(f"[Dataset] #samples={len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        st_path = row["safetensors_fullpath"]
        target = row["label_tensor"]

        feats, meta = get_canonical_data_with_meta(st_path)
        if feats is None:
            raise RuntimeError(f"Failed to parse LoRA file: {st_path}")

        return {"features": feats, "meta": meta}, target


def build_cache(cache_path: str, df: pd.DataFrame) -> None:
    parent = os.path.dirname(cache_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Caching -> {cache_path}"):
        feats, meta = get_canonical_data_with_meta(row["safetensors_fullpath"])
        if feats is None:
            continue
        data.append(
            {
                "name": row["name"],
                "features": feats,
                "meta": meta,
                "label": row["label_tensor"],
            }
        )

    torch.save(data, cache_path)
    print(f"[Cache] saved {len(data)} items to {cache_path}")


def init_wandb(args: argparse.Namespace) -> bool:
    if args.wandb_mode == "disabled":
        return False
    if wandb is None:
        logging.warning("wandb is not installed; continuing with --wandb-mode=disabled.")
        return False

    kwargs = {
        "project": args.wandb_project,
        "config": vars(args),
        "mode": args.wandb_mode,
    }
    if args.wandb_mode == "online":
        key = os.environ.get("WANDB_API_KEY")
        if key:
            wandb.login(key=key)
    wandb.init(**kwargs)
    return True


def main() -> None:
    parser = argparse.ArgumentParser("GoEmotions LoRA classification with W2T.")
    parser.add_argument("--labels_csv", type=str, default="./classification/lora_label_info.csv")
    parser.add_argument("--cache_path", type=str, default="./classification/cache/goemotions_array.pt")
    parser.add_argument(
        "--repacked_dir",
        type=str,
        default=None,
        help="If set, use repacked cache at repacked_dir/{train,valid,test}.",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="./classification/checkpoints")

    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--shard_cache_size", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--rank_encoder_layers", type=int, default=1)
    parser.add_argument("--num_layer_layers", type=int, default=2)
    parser.add_argument("--mlp_dim", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)

    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)
    parser.add_argument("--sign_aug_prob", type=float, default=0.5)
    parser.add_argument("--rank_perm_prob", type=float, default=0.5)
    parser.add_argument("--ratio", type=float, default=1.0)

    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["disabled", "offline", "online"],
        default="disabled",
    )
    parser.add_argument("--wandb_project", type=str, default="W2T-GoEmotions")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info("[INFO] args: %s", args)
    set_seed(args.seed)
    wandb_active = init_wandb(args)

    df = pd.read_csv(args.labels_csv)
    df["name"] = df["run_id"].apply(lambda x: f"run_{int(x)}")
    df["safetensors_fullpath"] = df["safetensors_path"].astype(str)
    df["label_tensor"] = df["label_vector"].apply(_parse_label_vector)

    num_pred = int(df["label_tensor"].iloc[0].numel())

    names = df["name"].tolist()
    test_ratio = 1.0 - args.train_ratio - args.valid_ratio
    assert test_ratio > 0, "train_ratio + valid_ratio must be < 1.0"
    train_names, valid_names, test_names = build_random_split(
        names,
        seed=args.split_seed,
        ratios=(args.train_ratio, args.valid_ratio, test_ratio),
    )

    if args.repacked_dir:
        train_set = RepackedSFTDatasetDir(
            os.path.join(args.repacked_dir, "train"),
            shard_cache_size=args.shard_cache_size,
        )
        valid_set = RepackedSFTDatasetDir(
            os.path.join(args.repacked_dir, "valid"),
            shard_cache_size=args.shard_cache_size,
        )
        test_set = RepackedSFTDatasetDir(
            os.path.join(args.repacked_dir, "test"),
            shard_cache_size=args.shard_cache_size,
        )
    elif args.cache_path and os.path.isdir(args.cache_path):
        print(f"[INFO] Using cached dataset: {args.cache_path}")
        train_set = CachedSVDatasetDir(args.cache_path, to_keep=train_names)
        valid_set = CachedSVDatasetDir(args.cache_path, to_keep=valid_names)
        test_set = CachedSVDatasetDir(args.cache_path, to_keep=test_names)
    elif args.cache_path and os.path.isfile(args.cache_path):
        print(f"[INFO] Using cached dataset file: {args.cache_path}")
        train_set = CachedSVDataset(args.cache_path, to_keep=train_names)
        valid_set = CachedSVDataset(args.cache_path, to_keep=valid_names)
        test_set = CachedSVDataset(args.cache_path, to_keep=test_names)
    else:
        if args.cache_path:
            print(f"[INFO] Building cache at: {args.cache_path}")
            build_cache(args.cache_path, df)
            train_set = CachedSVDataset(args.cache_path, to_keep=train_names)
            valid_set = CachedSVDataset(args.cache_path, to_keep=valid_names)
            test_set = CachedSVDataset(args.cache_path, to_keep=test_names)
        else:
            train_df = df[df["name"].isin(set(train_names))].reset_index(drop=True)
            valid_df = df[df["name"].isin(set(valid_names))].reset_index(drop=True)
            test_df = df[df["name"].isin(set(test_names))].reset_index(drop=True)
            train_set = GoEmoLoRADataset(train_df, verbose=True)
            valid_set = GoEmoLoRADataset(valid_df, verbose=True)
            test_set = GoEmoLoRADataset(test_df, verbose=True)

    print(len(train_set), len(valid_set), len(test_set))
    if args.ratio < 1.0:
        full_len = len(train_set)
        sub_len = int(args.ratio * full_len)
        train_set, _ = torch.utils.data.random_split(
            train_set,
            [sub_len, full_len - sub_len],
            generator=torch.Generator().manual_seed(args.seed),
        )

    sample = train_set[0][0]
    sample_feats = sample["features"]
    sample_meta = sample["meta"]
    assert sample_meta is not None, "meta is required for module-aware embeddings."

    input_dims = [(u.shape[1], v.shape[1]) for (u, v, s) in sample_feats]
    layer_ids = [m["layer_id"] for m in sample_meta]
    module_ids = [m["module_id"] for m in sample_meta]
    num_layers = max(layer_ids) + 1
    num_modules = max(module_ids) + 1

    model = FullTransformer(
        input_dims=input_dims,
        layer_ids=layer_ids,
        module_ids=module_ids,
        num_layers=num_layers,
        num_modules=num_modules,
        hidden_dim=args.hidden_dim,
        out_dim=num_pred,
        num_rank_layers=args.rank_encoder_layers,
        num_layer_layers=args.num_layer_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        mlp_dim=args.mlp_dim,
        sign_aug_prob=args.sign_aug_prob,
        rank_perm_prob=args.rank_perm_prob,
    ).to(device)

    print_model_stats(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_model = train(
        model=model,
        device=device,
        train_set=train_set,
        valid_set=valid_set,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_pred=num_pred,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        metric_for_best="macro_f1",
        thr=0.5,
    )

    model.load_state_dict(best_model, strict=True)
    test_metrics = test(model, device, test_set, num_pred=num_pred, thr=0.5)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[INFO] checkpoint saved to {ckpt_path}")

    if wandb_active and wandb is not None:
        wandb.log({f"test/{k}": v for k, v in test_metrics.items() if isinstance(v, (float, int))})
        wandb.finish()


if __name__ == "__main__":
    main()
