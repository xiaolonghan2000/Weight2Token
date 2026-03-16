from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from safetensors.numpy import load_file as st_load_numpy
from torch.utils.data import Dataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_model_stats(model: torch.nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: total={total_params:,}, trainable={trainable_params:,}")


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


@torch.no_grad()
def multilabel_f1_scores(probs: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> Tuple[float, float]:
    preds = (probs >= thr).to(torch.int32)
    t = targets.to(torch.int32)

    tp = (preds & t).sum(dim=0).to(torch.float32)
    fp = (preds & (1 - t)).sum(dim=0).to(torch.float32)
    fn = ((1 - preds) & t).sum(dim=0).to(torch.float32)

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1_per_label = 2 * precision * recall / (precision + recall + 1e-12)
    macro_f1 = float(f1_per_label.mean().item())

    tp_sum = float(tp.sum().item())
    fp_sum = float(fp.sum().item())
    fn_sum = float(fn.sum().item())
    micro_precision = _safe_div(tp_sum, tp_sum + fp_sum)
    micro_recall = _safe_div(tp_sum, tp_sum + fn_sum)
    micro_f1 = _safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)
    return macro_f1, micro_f1


@torch.no_grad()
def _binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    y = labels.to(torch.int32)
    pos = (y == 1)
    neg = (y == 0)
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float32)

    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.numel():
        j = i
        while j + 1 < sorted_scores.numel() and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_ranks_pos = float(ranks[pos].sum().item())
    u_stat = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_stat / (n_pos * n_neg))


@torch.no_grad()
def _binary_auprc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    y = labels.to(torch.int32)
    n_pos = int((y == 1).sum().item())
    if n_pos == 0:
        return float("nan")

    order = torch.argsort(scores, descending=True)
    y_sorted = y[order]

    tp = torch.cumsum((y_sorted == 1).to(torch.float32), dim=0)
    fp = torch.cumsum((y_sorted == 0).to(torch.float32), dim=0)
    precision = tp / (tp + fp + 1e-12)
    return float(precision[y_sorted == 1].mean().item())


@torch.no_grad()
def multilabel_auc_metrics(probs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    num_labels = probs.shape[1]
    aurocs = []
    auprcs = []
    for k in range(num_labels):
        aurocs.append(_binary_auroc(probs[:, k], targets[:, k]))
        auprcs.append(_binary_auprc(probs[:, k], targets[:, k]))

    auroc_t = torch.tensor(aurocs, dtype=torch.float32)
    auprc_t = torch.tensor(auprcs, dtype=torch.float32)
    mean_auroc = float(torch.nanmean(auroc_t).item())
    mean_auprc = float(torch.nanmean(auprc_t).item())
    return mean_auroc, mean_auprc


@torch.no_grad()
def compute_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thr: float = 0.5,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()
    acc = float(preds.eq(targets).float().mean().item())
    macro_f1, micro_f1 = multilabel_f1_scores(probs, targets, thr=thr)
    auroc, auprc = multilabel_auc_metrics(probs, targets)
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "auroc": auroc,
        "auprc": auprc,
    }


def move_batch_to_device(
    batch: Tuple[Dict[str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]], torch.Tensor],
    device: torch.device,
) -> Tuple[Dict[str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]], torch.Tensor]:
    x, y = batch
    feats = []
    for u, v, s in x["features"]:
        feats.append((u.to(device), v.to(device), s.to(device)))
    return {"features": feats}, y.to(device)


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x, y = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    thr: float = 0.5,
) -> Dict[str, float]:
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    total_loss = 0.0
    all_logits = []
    all_targets = []

    for batch in loader:
        x, y = move_batch_to_device(batch, device)
        logits = model(x)
        total_loss += float(loss_fn(logits, y).item())
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    avg_loss = float(total_loss / (targets.shape[0] * targets.shape[1]))

    metrics = compute_metrics_from_logits(logits, targets, thr=thr)
    metrics["loss"] = avg_loss
    return metrics


def load_split_names(split_path: str) -> List[str]:
    names = torch.load(split_path)
    if isinstance(names, torch.Tensor):
        names = names.tolist()
    return [str(x) for x in names]


def build_labels_map(labels_csv: str) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    df = pd.read_csv(labels_csv)
    meta_cols = {"celeb_folder", "num_images_in_folder", "num_unique_source_images"}
    attr_cols = [c for c in df.columns if c not in meta_cols]
    if "celeb_folder" not in df.columns:
        raise ValueError("labels_csv must contain 'celeb_folder' column.")

    labels_map: Dict[str, torch.Tensor] = {}
    for _, row in df.iterrows():
        key = str(row["celeb_folder"])
        y = row[attr_cols].to_numpy()
        uniq = set(np.unique(y).tolist())
        if uniq.issubset({0, 1}):
            labels = torch.tensor(y.astype(np.float32), dtype=torch.float32)
        else:
            labels = (torch.tensor(y, dtype=torch.float32) >= 0.0).to(torch.float32)
        labels_map[key] = labels
    return labels_map, attr_cols


def model_name_to_celeb_key(model_name: str) -> str:
    m = re.search(r"(\d+)$", model_name)
    if m is None:
        raise ValueError(f"Cannot parse numeric id from model name: {model_name}")
    return f"celeb_{int(m.group(1))}"


def canonical_svd_features(B: torch.Tensor, A_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_b, r_b = torch.linalg.qr(B)
    q_a, r_a = torch.linalg.qr(A_t)

    m = r_b @ r_a.T
    u_small, s, vh_small = torch.linalg.svd(m, full_matrices=False)
    v_small = vh_small.mT

    u = q_b @ u_small
    v = q_a @ v_small

    max_idx = torch.argmax(torch.abs(u), dim=0)
    col_idx = torch.arange(u.shape[1], device=u.device)
    signs = torch.sign(u[max_idx, col_idx])
    signs[signs == 0] = 1.0

    u = u * signs.view(1, -1)
    v = v * signs.view(1, -1)
    return u.T.contiguous(), v.T.contiguous(), s.view(-1, 1).contiguous()


_MODULE_PATTERNS = [
    (re.compile(r"\.to_q\b"), "to_q"),
    (re.compile(r"\.to_k\b"), "to_k"),
    (re.compile(r"\.to_v\b"), "to_v"),
    (re.compile(r"\.to_out\b"), "to_out"),
    (re.compile(r"\.proj_in\b"), "proj_in"),
    (re.compile(r"\.proj_out\b"), "proj_out"),
    (re.compile(r"\.ff\b|\.feed_forward\b|\.mlp\b"), "mlp"),
]


def _infer_module_type(base_key: str) -> str:
    for rgx, name in _MODULE_PATTERNS:
        if rgx.search(base_key):
            return name
    parts = base_key.split(".")
    return parts[-1] if parts else "unknown"


def _infer_layer_group(base_key: str) -> str:
    s = base_key
    s = re.sub(r"\.to_(q|k|v|out)\b", "", s)
    s = re.sub(r"\.(proj_in|proj_out)\b", "", s)
    s = re.sub(r"\.+", ".", s).strip(".")
    return s


def get_canonical_data_with_meta(path: str) -> Tuple[Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]], Optional[List[dict]]]:
    st_path = os.path.join(path, "adapter_model.safetensors")
    if not os.path.exists(st_path):
        return None, None

    sd = st_load_numpy(st_path)
    groups: Dict[str, Dict[str, np.ndarray]] = {}
    for k, arr in sd.items():
        if ".lora_A.weight" in k:
            groups.setdefault(k.replace(".lora_A.weight", ""), {})["A"] = arr
        elif ".lora_B.weight" in k:
            groups.setdefault(k.replace(".lora_B.weight", ""), {})["B"] = arr

    entries = []
    for base, ab in groups.items():
        if "A" not in ab or "B" not in ab:
            continue
        entries.append((_infer_layer_group(base), _infer_module_type(base), base, ab["A"], ab["B"]))
    entries.sort(key=lambda x: (x[0], x[1], x[2]))

    if not entries:
        return None, None

    layer_keys = sorted(set(e[0] for e in entries))
    module_keys = sorted(set(e[1] for e in entries))
    layer2id = {k: i for i, k in enumerate(layer_keys)}
    module2id = {k: i for i, k in enumerate(module_keys)}

    features = []
    meta = []
    for layer_key, module_type, base, a_np, b_np in entries:
        a = torch.tensor(a_np, dtype=torch.float32)
        b = torch.tensor(b_np, dtype=torch.float32)
        u_t, v_t, s = canonical_svd_features(b, a.T)
        features.append((u_t, v_t, s))
        meta.append(
            {
                "layer_id": layer2id[layer_key],
                "module_id": module2id[module_type],
                "layer_key": layer_key,
                "module_type": module_type,
                "base_key": base,
            }
        )
    return features, meta


class CachedCanonicalDataset(Dataset):
    def __init__(self, cache_path: str, split_names: Optional[Sequence[str]] = None):
        data = torch.load(cache_path, map_location="cpu")
        keep = set(split_names) if split_names is not None else None

        self.records = []
        for item in data:
            name = str(item["name"])
            if keep is not None and name not in keep:
                continue
            features = item["features"]
            meta = item.get("meta", None)
            label = item["label"]
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, dtype=torch.float32)
            label = label.to(torch.float32)
            self.records.append((name, features, meta, label))

        if len(self.records) == 0:
            raise RuntimeError(f"No usable samples from cache: {cache_path}")
        self.sample_meta = self.records[0][2]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor]:
        _, features, _, label = self.records[idx]
        return features, label


class OnTheFlyCanonicalDataset(Dataset):
    def __init__(self, train_directory: str, labels_map: Dict[str, torch.Tensor], split_names: Sequence[str]):
        self.train_directory = Path(train_directory)
        self.labels_map = labels_map

        model_names: List[str] = []
        for name in split_names:
            if "model" not in name:
                continue
            st_path = self.train_directory / name / "unet" / "adapter_model.safetensors"
            if not st_path.exists():
                continue
            celeb_key = model_name_to_celeb_key(name)
            if celeb_key not in labels_map:
                continue
            model_names.append(name)

        if len(model_names) == 0:
            raise RuntimeError("No valid models found for split.")
        self.model_names = sorted(model_names)

        first_path = self.train_directory / self.model_names[0] / "unet"
        _, sample_meta = get_canonical_data_with_meta(str(first_path))
        self.sample_meta = sample_meta

    def __len__(self) -> int:
        return len(self.model_names)

    def __getitem__(self, idx: int) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor]:
        model_name = self.model_names[idx]
        celeb_key = model_name_to_celeb_key(model_name)
        label = self.labels_map[celeb_key].to(torch.float32)
        path = self.train_directory / model_name / "unet"
        features, _ = get_canonical_data_with_meta(str(path))
        if features is None:
            raise RuntimeError(f"Failed to load canonical features: {path}")
        return features, label


def collate_canonical_features(
    batch: Sequence[Tuple[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor]]
) -> Tuple[Dict[str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]], torch.Tensor]:
    feats_list, labels = zip(*batch)
    num_pos = len(feats_list[0])
    if not all(len(x) == num_pos for x in feats_list):
        raise RuntimeError("Inconsistent number of LoRA positions in batch.")

    batched_feats: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for p in range(num_pos):
        u = torch.stack([x[p][0] for x in feats_list], dim=0)
        v = torch.stack([x[p][1] for x in feats_list], dim=0)
        s = torch.stack([x[p][2] for x in feats_list], dim=0)
        batched_feats.append((u, v, s))

    y = torch.stack(labels, dim=0).to(torch.float32)
    return {"features": batched_feats}, y


def metrics_brief(metrics: Dict[str, float]) -> str:
    return (
        f"loss={metrics['loss']:.4f} "
        f"acc={metrics['acc']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"micro_f1={metrics['micro_f1']:.4f} "
        f"auroc={metrics['auroc']:.4f} "
        f"auprc={metrics['auprc']:.4f}"
    )
