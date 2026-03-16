from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.numpy import load_file as st_load_numpy


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


@torch.no_grad()
def multilabel_f1_scores(probs: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> tuple[float, float]:
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
def multilabel_auc_metrics(probs: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    k = probs.shape[1]
    aurocs = []
    auprcs = []
    for i in range(k):
        aurocs.append(_binary_auroc(probs[:, i], targets[:, i]))
        auprcs.append(_binary_auprc(probs[:, i], targets[:, i]))
    auroc_t = torch.tensor(aurocs, dtype=torch.float32)
    auprc_t = torch.tensor(auprcs, dtype=torch.float32)
    return float(torch.nanmean(auroc_t).item()), float(torch.nanmean(auprc_t).item())


@torch.no_grad()
def compute_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> Dict[str, float]:
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


def metrics_brief(metrics: Dict[str, float]) -> str:
    return (
        f"loss={metrics['loss']:.4f} "
        f"acc={metrics['acc']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"micro_f1={metrics['micro_f1']:.4f} "
        f"auroc={metrics['auroc']:.4f} "
        f"auprc={metrics['auprc']:.4f}"
    )


def to_device(x, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, tuple):
        return tuple(to_device(t, device) for t in x)
    if isinstance(x, list):
        return [to_device(t, device) for t in x]
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


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
    for data, target in loader:
        data = to_device(data, device)
        target = target.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        loss = loss_fn(logits, target)
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

    for data, target in loader:
        data = to_device(data, device)
        target = target.to(device).float()
        logits = model(data)
        total_loss += float(loss_fn(logits, target).item())
        all_logits.append(logits.detach().cpu())
        all_targets.append(target.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    avg_loss = float(total_loss / (targets.shape[0] * targets.shape[1]))
    metrics = compute_metrics_from_logits(logits, targets, thr=thr)
    metrics["loss"] = avg_loss
    return metrics


def get_uvs_from_file(path: str) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
    st_path = os.path.join(path, "adapter_model.safetensors")
    if not os.path.exists(st_path):
        return None
    tensors = list(st_load_numpy(st_path).values())
    uvs = []
    for i in range(len(tensors) // 2):
        b = torch.tensor(tensors[2 * i + 1], dtype=torch.float32)
        a = torch.tensor(tensors[2 * i], dtype=torch.float32)
        uvs.append((b, a.T))
    return uvs


def get_equiv_shapes(point: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[list[int], list[int]]:
    ns = []
    ms = []
    for u, v in point:
        ns.append(u.shape[0])
        ms.append(v.shape[0])
    return ns, ms


_LAYER_PATTERNS = [
    re.compile(r"(down_blocks)\.(\d+)"),
    re.compile(r"(up_blocks)\.(\d+)"),
    re.compile(r"(mid_block)"),
    re.compile(r"(transformer_blocks)\.(\d+)"),
    re.compile(r"(blocks)\.(\d+)"),
]


def infer_layer_id(name: str) -> str:
    for pat in _LAYER_PATTERNS:
        m = pat.search(name)
        if m:
            if m.group(1) == "mid_block":
                return "mid_block"
            if m.lastindex and m.lastindex >= 2:
                return f"{m.group(1)}.{m.group(2)}"
            return m.group(1)
    return "unknown"


def build_layerwise_flat_BA(tensors_by_key: dict) -> tuple[list[str], list[torch.Tensor]]:
    layer_buf = defaultdict(list)
    for k, w in tensors_by_key.items():
        kk = k.lower()
        if ("lora" not in kk) and ("adapter" not in kk):
            continue
        lid = infer_layer_id(k)
        layer_buf[lid].append(torch.tensor(w).reshape(-1).float())

    layer_keys = sorted(layer_buf.keys())
    x_layers = []
    for lid in layer_keys:
        if len(layer_buf[lid]) == 0:
            x_layers.append(torch.zeros(1))
        else:
            x_layers.append(torch.cat(layer_buf[lid], dim=0))
    return layer_keys, x_layers


def build_layerwise_tokenized_BA(tensors_by_key: dict, token_size: int = 2048) -> tuple[list[str], list[torch.Tensor]]:
    layer_buf = defaultdict(list)
    for k, w in tensors_by_key.items():
        kk = k.lower()
        if ("lora" not in kk) and ("adapter" not in kk):
            continue
        lid = infer_layer_id(k)
        layer_buf[lid].append(torch.tensor(w).reshape(-1).float())

    layer_keys = sorted(layer_buf.keys())
    x_layers = []
    for lid in layer_keys:
        if len(layer_buf[lid]) == 0:
            x_layers.append(torch.zeros(1, token_size))
            continue

        full_vec = torch.cat(layer_buf[lid], dim=0)
        d = full_vec.numel()
        n_tokens = (d + token_size - 1) // token_size
        pad_len = n_tokens * token_size - d
        if pad_len > 0:
            full_vec = F.pad(full_vec, (0, pad_len))
        x_layers.append(full_vec.view(n_tokens, token_size))
    return layer_keys, x_layers


def collate_glnet(batch):
    xs, ys = zip(*batch)
    num_layers = len(xs[0])
    batched = []
    for l in range(num_layers):
        u = torch.stack([x[l][0] for x in xs], dim=0)
        v = torch.stack([x[l][1] for x in xs], dim=0)
        batched.append((u, v))
    y = torch.stack(ys).float()
    return batched, y


def collate_layerwise_flat(batch):
    xs, ys = zip(*batch)
    num_layers = len(xs[0])
    out_layers = []
    for l in range(num_layers):
        layer_vecs = [x[l] for x in xs]
        max_d = max(v.numel() for v in layer_vecs)
        x_pad = torch.zeros(len(layer_vecs), max_d, dtype=layer_vecs[0].dtype)
        for i, v in enumerate(layer_vecs):
            x_pad[i, : v.numel()] = v
        out_layers.append(x_pad)
    y = torch.stack(ys).float()
    return out_layers, y


def collate_layerwise_tokenized(batch):
    xs, ys = zip(*batch)
    num_layers = len(xs[0])
    if not all(len(x) == num_layers for x in xs):
        raise RuntimeError("Inconsistent number of layers in tokenized batch.")

    token_size = xs[0][0].shape[1]
    seq_tokens = []
    seq_layer_ids = []
    seq_lens = []

    for x in xs:
        sample_tokens = []
        sample_layer_ids = []
        for lid in range(num_layers):
            t = x[lid]
            if t.dim() != 2 or t.shape[1] != token_size:
                raise RuntimeError("Inconsistent token shape in tokenized batch.")
            sample_tokens.append(t)
            sample_layer_ids.append(torch.full((t.shape[0],), lid, dtype=torch.long))
        t_all = torch.cat(sample_tokens, dim=0)
        lid_all = torch.cat(sample_layer_ids, dim=0)
        seq_tokens.append(t_all)
        seq_layer_ids.append(lid_all)
        seq_lens.append(t_all.shape[0])

    max_t = max(seq_lens)
    bsz = len(xs)
    tokens = torch.zeros(bsz, max_t, token_size, dtype=seq_tokens[0].dtype)
    layer_ids = torch.zeros(bsz, max_t, dtype=torch.long)
    padding_mask = torch.ones(bsz, max_t, dtype=torch.bool)

    for i, (t_all, lid_all) in enumerate(zip(seq_tokens, seq_layer_ids)):
        n = t_all.shape[0]
        tokens[i, :n, :] = t_all
        layer_ids[i, :n] = lid_all
        padding_mask[i, :n] = False

    y = torch.stack(ys).float()
    return {"tokens": tokens, "padding_mask": padding_mask, "layer_ids": layer_ids}, y
