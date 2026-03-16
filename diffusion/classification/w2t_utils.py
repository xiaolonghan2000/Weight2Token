import os
import re
import copy
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from safetensors.numpy import load_file
import math

wandb = None


def set_wandb_module(module) -> None:
    global wandb
    wandb = module


def _wandb_log(data: dict) -> None:
    if wandb is not None:
        wandb.log(data)

def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0

@torch.no_grad()
def multilabel_f1_scores(probs: torch.Tensor, targets: torch.Tensor, thr: float = 0.5):
    preds = (probs >= thr).to(torch.int32)
    t = targets.to(torch.int32)

    # per-label TP/FP/FN
    tp = (preds & t).sum(dim=0).to(torch.float32)            # [K]
    fp = (preds & (1 - t)).sum(dim=0).to(torch.float32)      # [K]
    fn = ((1 - preds) & t).sum(dim=0).to(torch.float32)      # [K]

    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1_per_label = 2 * precision * recall / (precision + recall + 1e-12)  # [K]

    macro_f1 = f1_per_label.mean().item()

    # micro: pool over labels
    TP = tp.sum().item()
    FP = fp.sum().item()
    FN = fn.sum().item()
    micro_precision = _safe_div(TP, TP + FP)
    micro_recall    = _safe_div(TP, TP + FN)
    micro_f1 = _safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    return macro_f1, micro_f1, f1_per_label.cpu()

@torch.no_grad()
def _binary_auroc(scores_1d: torch.Tensor, y_1d: torch.Tensor):
    """
    AUROC for binary labels using rank-based Mann-Whitney U statistic.
    Returns NaN if all labels are same.
    """
    y = y_1d.to(torch.int32)
    pos = (y == 1)
    neg = (y == 0)
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # ranks of scores (ascending)
    scores = scores_1d.to(torch.float32)
    order = torch.argsort(scores)  # ascending
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float32)

    # handle ties by average rank (simple stable way)
    # (optional) to be more exact: group ties; here we do a tie-aware pass
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.numel():
        j = i
        while j + 1 < sorted_scores.numel() and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = (i + 1 + j + 1) / 2.0
            ranks[order[i:j+1]] = avg_rank
        i = j + 1

    sum_ranks_pos = ranks[pos].sum().item()
    # U statistic
    U = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    auroc = U / (n_pos * n_neg)
    return float(auroc)

@torch.no_grad()
def _binary_average_precision(scores_1d: torch.Tensor, y_1d: torch.Tensor):
    """
    Average Precision (area under PR curve, step-wise).
    Returns NaN if no positive.
    """
    y = y_1d.to(torch.int32)
    n_pos = int((y == 1).sum().item())
    if n_pos == 0:
        return float("nan")

    scores = scores_1d.to(torch.float32)
    order = torch.argsort(scores, descending=True)
    y_sorted = y[order]

    tp = torch.cumsum((y_sorted == 1).to(torch.float32), dim=0)
    fp = torch.cumsum((y_sorted == 0).to(torch.float32), dim=0)

    precision = tp / (tp + fp + 1e-12)
    # AP = mean precision at each positive
    ap = precision[y_sorted == 1].mean().item()
    return float(ap)

@torch.no_grad()
def multilabel_auc_metrics(probs: torch.Tensor, targets: torch.Tensor):
    """
    Returns:
      mean_auroc, mean_auprc, auroc_per_label[K], auprc_per_label[K]
    """
    K = probs.shape[1]
    aurocs = []
    auprcs = []
    for k in range(K):
        s = probs[:, k]
        y = targets[:, k]
        aurocs.append(_binary_auroc(s, y))
        auprcs.append(_binary_average_precision(s, y))

    auroc_t = torch.tensor(aurocs, dtype=torch.float32)
    auprc_t = torch.tensor(auprcs, dtype=torch.float32)

    # mean over valid labels (exclude NaN)
    mean_auroc = float(torch.nanmean(auroc_t).item()) if torch.isnan(auroc_t).any() else float(auroc_t.mean().item())
    mean_auprc = float(torch.nanmean(auprc_t).item()) if torch.isnan(auprc_t).any() else float(auprc_t.mean().item())

    return mean_auroc, mean_auprc, auroc_t, auprc_t


def to_cuda(obj, device=None):
    """Move nested tensors (lists/tuples/dicts) to device while preserving structure."""
    if device is None:
        device = torch.device("cuda")
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).to(device)
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(to_cuda(x, device) for x in obj)
    if isinstance(obj, list):
        return [to_cuda(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: to_cuda(v, device) for k, v in obj.items()}
    return obj


def train(model, device, train_set, valid_set, optimizer, scheduler, epochs, batch_size=32, num_pred=40, label_smoothing=0.0, mixup_alpha=0.0,
          metric_for_best="macro_f1", thr=0.5):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    loss_fn = nn.BCEWithLogitsLoss()

    best_model_state = copy.deepcopy(model.state_dict())
    best_valid_score = -1.0

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        tot_loss = 0.0

        for data, target in train_loader:
            data = to_cuda(data, device)

            target = target.float().to(device)
            target = smooth_targets(target, label_smoothing)
            if mixup_alpha and mixup_alpha > 0 and model.training:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                index = torch.randperm(target.size(0), device=target.device)
                data = mixup_nested(data, lam, index)
                target = lam * target + (1 - lam) * target[index]

            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tot_loss += loss.item()

        scheduler.step()
        tqdm.write(f"EPOCH {epoch}: TrainLoss {tot_loss / max(1, len(train_loader)):.4f}")

        valid_metrics = valid(model, device, valid_set, num_pred=num_pred, thr=thr)

        _wandb_log({
            "train_loss": tot_loss / len(train_loader),
            "valid_loss": valid_metrics["loss"],
            "valid_acc": valid_metrics["acc"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "valid_micro_f1": valid_metrics["micro_f1"],
            "valid_mean_auroc": valid_metrics["mean_auroc"],
            "valid_mean_auprc": valid_metrics["mean_auprc"],
            "epoch": epoch
        })

        score = valid_metrics[metric_for_best]
        if score > best_valid_score:
            best_valid_score = score
            best_model_state = copy.deepcopy(model.state_dict())
            tqdm.write(f"Saving best model at epoch {epoch} | {metric_for_best}={score:.4f}")
            # torch.save(model.state_dict(), f"best_model_{epoch}.pth")

    return best_model_state


@torch.no_grad()
def valid(model, device, valid_set, num_pred=40, thr=0.5):
    model.eval()
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=False)

    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0

    all_logits = []
    all_targets = []

    for data, target in valid_loader:
        data = to_cuda(data, device)
        target = target.float().to(device)

        logits = model(data)  # [B, K]
        total_loss += loss_fn(logits, target).item()

        all_logits.append(logits.detach())
        all_targets.append(target.detach())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    probs = torch.sigmoid(logits)

    # keep acc as supplementary (same semantics as your old code)
    pred = (probs >= thr).float()
    correct = pred.eq(targets).sum().item()
    acc = correct / (len(valid_set) * num_pred)

    macro_f1, micro_f1, f1_per_label = multilabel_f1_scores(probs, targets, thr=thr)
    mean_auroc, mean_auprc, auroc_per_label, auprc_per_label = multilabel_auc_metrics(probs, targets)

    avg_loss = total_loss / (len(valid_loader.dataset) * num_pred)

    print(
        f"Valid: loss={avg_loss:.4f} "
        f"acc={acc:.4f} macroF1={macro_f1:.4f} microF1={micro_f1:.4f} "
        f"mAUROC={mean_auroc:.4f} mAUPRC={mean_auprc:.4f}"
    )

    metrics = {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "mean_auroc": mean_auroc,
        "mean_auprc": mean_auprc,
        # Optional: keep per-label vectors for later analysis without logging them every step.
        "f1_per_label": f1_per_label,
        "auroc_per_label": auroc_per_label.cpu(),
        "auprc_per_label": auprc_per_label.cpu(),
    }
    return metrics


@torch.no_grad()
def test(model, device, test_set, num_pred=40, thr=0.5):
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0

    all_logits = []
    all_targets = []

    for data, target in test_loader:
        data = to_cuda(data, device)
        target = target.float().to(device)

        logits = model(data)
        total_loss += loss_fn(logits, target).item()
        all_logits.append(logits.detach())
        all_targets.append(target.detach())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    probs = torch.sigmoid(logits)

    pred = (probs >= thr).float()
    correct = pred.eq(targets).sum().item()
    acc = correct / (len(test_set) * num_pred)

    macro_f1, micro_f1, f1_per_label = multilabel_f1_scores(probs, targets, thr=thr)
    mean_auroc, mean_auprc, auroc_per_label, auprc_per_label = multilabel_auc_metrics(probs, targets)

    avg_loss = total_loss / (len(test_loader.dataset) * num_pred)

    print(
        f"Test:  loss={avg_loss:.4f} "
        f"acc={acc:.4f} macroF1={macro_f1:.4f} microF1={micro_f1:.4f} "
        f"mAUROC={mean_auroc:.4f} mAUPRC={mean_auprc:.4f}"
    )

    metrics = {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "mean_auroc": mean_auroc,
        "mean_auprc": mean_auprc,
        "f1_per_label": f1_per_label,
        "auroc_per_label": auroc_per_label.cpu(),
        "auprc_per_label": auprc_per_label.cpu(),
    }
    return metrics


def canonical_svd_features(B: torch.Tensor, A_T: torch.Tensor):
    device = B.device

    Q_B, R_B = torch.linalg.qr(B)
    Q_A, R_A = torch.linalg.qr(A_T)

    M = R_B @ R_A.T
    U_small, S, Vh_small = torch.linalg.svd(M, full_matrices=False)
    V_small = Vh_small.mT

    U = Q_B @ U_small  # [d_out, r]
    V = Q_A @ V_small  # [d_in, r]

    # deterministic sign fixing using U
    max_idx = torch.argmax(torch.abs(U), dim=0)
    col_idx = torch.arange(U.shape[1], device=device)
    signs = torch.sign(U[max_idx, col_idx])
    signs[signs == 0] = 1.0

    U = U * signs.view(1, -1)
    V = V * signs.view(1, -1)

    return U.T.contiguous(), V.T.contiguous(), S.view(-1, 1).contiguous()

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
    """Infer a coarse module type from a LoRA key without .lora_[AB].weight suffix."""
    for rgx, name in _MODULE_PATTERNS:
        if rgx.search(base_key):
            return name
    # fallback: take last segment(s)
    parts = base_key.split(".")
    return parts[-1] if parts else "unknown"

def _infer_layer_group(base_key: str) -> str:
    """Infer a layer-group identifier from the key.

    We intentionally keep it model-agnostic: remove only the module-type token so
    that different modules (q/v/mlp/...) under the same block share the same group.
    """
    # strip common module markers so the remaining string groups by "block"
    s = base_key
    s = re.sub(r"\.to_(q|k|v|out)\b", "", s)
    s = re.sub(r"\.(proj_in|proj_out)\b", "", s)
    # collapse consecutive dots that can appear after removal
    s = re.sub(r"\.+", ".", s).strip(".")
    return s


def get_canonical_data_with_meta(path: str):
    """Return canonical SVD features + (layer_id, module_id) meta for each LoRA matrix.

    Output:
      features: List[(U_T, V_T, S)] aligned with meta
      meta:     List[{'layer_id': int, 'module_id': int, 'layer_key': str, 'module_type': str, 'base_key': str}]
    """
    new_path = os.path.join(path, "adapter_model.safetensors")
    if not os.path.exists(new_path):
        return None, None

    sd = load_file(new_path)  # dict: key -> np.ndarray
    # group A/B pairs by base key (strip lora_A/B.weight)
    groups = {}
    for k, arr in sd.items():
        if ".lora_A.weight" in k:
            base = k.replace(".lora_A.weight", "")
            groups.setdefault(base, {})["A"] = arr
        elif ".lora_B.weight" in k:
            base = k.replace(".lora_B.weight", "")
            groups.setdefault(base, {})["B"] = arr

    # build deterministic ordering
    entries = []
    for base, ab in groups.items():
        if "A" not in ab or "B" not in ab:
            continue
        module_type = _infer_module_type(base)
        layer_key = _infer_layer_group(base)
        entries.append((layer_key, module_type, base, ab["A"], ab["B"]))

    # sort by layer_key then module_type then base for determinism
    entries.sort(key=lambda x: (x[0], x[1], x[2]))

    # map to ids
    layer_keys = []
    module_types = []
    for layer_key, module_type, *_ in entries:
        layer_keys.append(layer_key)
        module_types.append(module_type)

    layer2id = {k: i for i, k in enumerate(sorted(set(layer_keys)))}
    module2id = {k: i for i, k in enumerate(sorted(set(module_types)))}

    features = []
    meta = []
    for layer_key, module_type, base, A_np, B_np in entries:
        B = torch.tensor(B_np, dtype=torch.float32)        # [d_out, r]
        A = torch.tensor(A_np, dtype=torch.float32)        # [r, d_in]
        U_T, V_T, S = canonical_svd_features(B, A.T)
        features.append((U_T, V_T, S))
        meta.append({
            "layer_id": layer2id[layer_key],
            "module_id": module2id[module_type],
            "layer_key": layer_key,
            "module_type": module_type,
            "base_key": base,
        })

    return features, meta


def get_canonical_data_list(path: str):
    """Backward-compatible wrapper.

    Returns only the feature list (ordered deterministically). Use
    get_canonical_data_with_meta() when layer/module ids are needed.
    """
    features, _ = get_canonical_data_with_meta(path)
    return features


class CachedSVDataset(torch.utils.data.Dataset):
    """Cache format: list of dict.

    New recommended cache item:
      {'name': str, 'features': List[(U_T,V_T,S)], 'meta': List[...], 'label': BoolTensor[num_pred]}
    Backward compatible with older caches that omit 'meta'.
    """
    def __init__(self, cache_path, to_keep=None):
        self.data = torch.load(cache_path, map_location="cpu")
        if to_keep is not None:
            keep = set(to_keep)
            self.data = [d for d in self.data if d["name"] in keep]

        num_pos = len(self.data[0]["features"])
        assert all(len(d["features"]) == num_pos for d in self.data), "Inconsistent number of positions across samples."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Return (features, meta) if available; otherwise features only
        if "meta" in item:
            return {"features": item["features"], "meta": item["meta"]}, item["label"]
        return {"features": item["features"], "meta": None}, item["label"]


def print_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: total={total_params:,} trainable={trainable:,}")

def smooth_targets(y: torch.Tensor, eps: float) -> torch.Tensor:
    # multi-label: move targets toward 0.5
    if eps <= 0:
        return y
    return y * (1.0 - eps) + 0.5 * eps

def mixup_nested(data: dict, lam: float, index: torch.Tensor) -> dict:
    # Mix only tensor fields under "features" list: (u,v,s)
    out = {}
    out["meta"] = data.get("meta", None)
    feats = data["features"]
    mixed = []
    for (u, v, s) in feats:
        u2 = u[index]
        v2 = v[index]
        s2 = s[index]
        mixed.append((lam * u + (1 - lam) * u2,
                      lam * v + (1 - lam) * v2,
                      lam * s + (1 - lam) * s2))
    out["features"] = mixed
    return out
