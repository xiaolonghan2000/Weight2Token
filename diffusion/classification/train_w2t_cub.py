import argparse
import logging
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import w2t_utils
from w2t_models import FullTransformer, SimpleLoRATransformer
from w2t_utils import CachedSVDataset, get_canonical_data_with_meta, print_model_stats, test, train


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID_PATTERN = re.compile(r"_(\d+)$")


def set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


def freeze_backbone_for_warmup(model: torch.nn.Module) -> None:
    set_requires_grad(model, False)
    set_requires_grad(model.projectors, True)
    set_requires_grad(model.layer_embedding, True)
    set_requires_grad(model.module_embedding, True)
    set_requires_grad(model.classifier, True)


def unfreeze_all(model: torch.nn.Module) -> None:
    set_requires_grad(model, True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def image_id_from_model_name(model_name: str) -> int:
    match = MODEL_ID_PATTERN.search(model_name)
    if not match:
        raise ValueError(f"Could not infer image id from model directory name: {model_name}")
    return int(match.group(1))


def load_label_dataframe(labels_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    labels_df = pd.read_csv(labels_csv)
    meta_cols = {"cub_folder", "image_id", "image_name", "rel_path"}
    attr_cols = [c for c in labels_df.columns if c not in meta_cols]
    return labels_df, attr_cols


def build_label_map(labels_df: pd.DataFrame, attr_cols: list[str]) -> dict[int, torch.Tensor]:
    label_map: dict[int, torch.Tensor] = {}
    for _, row in labels_df.iterrows():
        values = row[attr_cols].to_numpy()
        if set(np.unique(values)).issubset({0, 1}):
            label = torch.tensor(values.astype(np.int64), dtype=torch.bool)
        else:
            label = torch.tensor(values, dtype=torch.float32) >= 0.0
        label_map[int(row["image_id"])] = label
    return label_map


class CUBLoRADataset(torch.utils.data.Dataset):
    def __init__(self, lora_root: Path, label_map: dict[int, torch.Tensor], names: list[str]):
        self.lora_root = lora_root
        self.label_map = label_map
        self.names = list(names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        model_name = self.names[idx]
        image_id = image_id_from_model_name(model_name)
        features, meta = get_canonical_data_with_meta(str(self.lora_root / model_name / "unet"))
        if features is None or meta is None:
            raise FileNotFoundError(f"Missing adapter_model.safetensors under {self.lora_root / model_name / 'unet'}")
        return {"features": features, "meta": meta}, self.label_map[image_id]


def load_split_names(split_dir: Path) -> tuple[list[str], list[str], list[str]]:
    train_names = torch.load(split_dir / "train.pt")
    valid_names = torch.load(split_dir / "valid.pt")
    test_names = torch.load(split_dir / "test.pt")
    return list(train_names), list(valid_names), list(test_names)


def init_wandb(args: argparse.Namespace) -> bool:
    if args.wandb_mode == "disabled":
        w2t_utils.set_wandb_module(None)
        return False
    try:
        import wandb  # type: ignore
    except Exception:
        logging.warning("wandb is not installed; continuing with wandb disabled.")
        w2t_utils.set_wandb_module(None)
        return False
    wandb.init(project=args.wandb_project, config=vars(args), mode=args.wandb_mode)
    w2t_utils.set_wandb_module(wandb)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate W2T on CUB-LoRA.")
    ap.add_argument("--lora_root", type=str, default="./data_generation/outputs/cub/models_rank8_full")
    ap.add_argument("--labels_csv", type=str, default="./data_prepare/outputs/cub_labels.csv")
    ap.add_argument("--split_dir", type=str, default="./data_prepare/outputs/splits_cub_rank8_full")
    ap.add_argument("--cache_path", type=str, default="")
    ap.add_argument("--checkpoint_dir", type=str, default="./classification/checkpoints/cub")
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--rank_encoder_layers", type=int, default=1)
    ap.add_argument("--num_layer_layers", type=int, default=2)
    ap.add_argument("--mlp_dim", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--mixup_alpha", type=float, default=0.0)
    ap.add_argument("--sign_aug_prob", type=float, default=0.5)
    ap.add_argument("--rank_perm_prob", type=float, default=0.5)
    ap.add_argument("--train_ratio", type=float, default=1.0)
    ap.add_argument("--warmup_epochs", type=int, default=4)
    ap.add_argument("--warmup_lr", type=float, default=None)
    ap.add_argument("--warmup_start_factor", type=float, default=0.1)
    ap.add_argument("--simplemodel", action="store_true")
    ap.add_argument("--wandb_mode", type=str, choices=["disabled", "offline", "online"], default="disabled")
    ap.add_argument("--wandb_project", type=str, default="W2T-CUB")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("[INFO] args: %s", args)
    set_seed(args.seed)
    use_wandb = init_wandb(args)

    labels_df, attr_cols = load_label_dataframe(Path(args.labels_csv))
    label_map = build_label_map(labels_df, attr_cols)
    num_pred = len(attr_cols)
    train_names, valid_names, test_names = load_split_names(Path(args.split_dir))

    cache_path = Path(args.cache_path) if args.cache_path else None
    if cache_path and cache_path.exists():
        logging.info("Using cache file: %s", cache_path)
        train_set = CachedSVDataset(str(cache_path), to_keep=train_names)
        valid_set = CachedSVDataset(str(cache_path), to_keep=valid_names)
        test_set = CachedSVDataset(str(cache_path), to_keep=test_names)
    else:
        lora_root = Path(args.lora_root)
        train_set = CUBLoRADataset(lora_root, label_map, train_names)
        valid_set = CUBLoRADataset(lora_root, label_map, valid_names)
        test_set = CUBLoRADataset(lora_root, label_map, test_names)

    if not 0.0 < args.train_ratio <= 1.0:
        raise ValueError("--train_ratio must be in (0, 1].")
    if args.train_ratio < 1.0:
        full_len = len(train_set)
        sub_len = max(1, int(args.train_ratio * full_len))
        train_set, _ = torch.utils.data.random_split(
            train_set,
            [sub_len, full_len - sub_len],
            generator=torch.Generator().manual_seed(args.seed),
        )

    sample = train_set[0][0]
    sample_feats = sample["features"]
    sample_meta = sample["meta"]
    input_dims = [(u.shape[1], v.shape[1]) for (u, v, s) in sample_feats]
    layer_ids = [m["layer_id"] for m in sample_meta]
    module_ids = [m["module_id"] for m in sample_meta]
    num_layers = max(layer_ids) + 1
    num_modules = max(module_ids) + 1

    if args.simplemodel:
        model = SimpleLoRATransformer(
            input_dims=input_dims,
            layer_ids=layer_ids,
            module_ids=module_ids,
            num_layers=num_layers,
            num_modules=num_modules,
            hidden_dim=args.hidden_dim,
            out_dim=num_pred,
            num_layer_layers=args.num_layer_layers,
            nhead=args.nhead,
            dropout=args.dropout,
            mlp_dim=args.mlp_dim,
            num_queries=args.nhead,
            sigma_prior_alpha=0.5,
        ).to(device)
    else:
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

    warmup_epochs = max(0, int(args.warmup_epochs))
    main_epochs = int(args.epochs) - warmup_epochs
    if main_epochs < 0:
        raise ValueError("--epochs must be >= --warmup_epochs.")

    if warmup_epochs > 0:
        freeze_backbone_for_warmup(model)
        warmup_lr = args.lr if args.warmup_lr is None else float(args.warmup_lr)
        warmup_params = [p for p in model.parameters() if p.requires_grad]
        warmup_optim = torch.optim.AdamW(warmup_params, lr=warmup_lr, weight_decay=args.weight_decay)
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            warmup_optim,
            start_factor=float(args.warmup_start_factor),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        warmup_state = train(
            model,
            device,
            train_set,
            valid_set,
            warmup_optim,
            warmup_sched,
            epochs=warmup_epochs,
            batch_size=args.batch_size,
            num_pred=num_pred,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha,
        )
        model.load_state_dict(warmup_state)

    unfreeze_all(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, main_epochs),
        eta_min=1e-5,
    )
    best_state = train(
        model,
        device,
        train_set,
        valid_set,
        optimizer,
        scheduler,
        epochs=main_epochs,
        batch_size=args.batch_size,
        num_pred=num_pred,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        metric_for_best="macro_f1",
    )
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")

    test_metrics = test(model, device, test_set, num_pred=num_pred, thr=0.5)
    logging.info("[TEST] %s", test_metrics)
    if use_wandb:
        wandb.log(
            {
                "test_loss": test_metrics["loss"],
                "test_acc": test_metrics["acc"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_micro_f1": test_metrics["micro_f1"],
                "test_mean_auroc": test_metrics["mean_auroc"],
                "test_mean_auprc": test_metrics["mean_auprc"],
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
