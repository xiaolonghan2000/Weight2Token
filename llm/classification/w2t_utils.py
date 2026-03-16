import os
import re
import glob
import copy
import json
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from safetensors.numpy import load_file

try:
    import wandb
except Exception:
    wandb = None


def _wandb_log(payload: dict) -> None:
    if wandb is None or getattr(wandb, "run", None) is None:
        return
    wandb.log(payload)

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
    AUROC for binary labels using rank-based MannWhitney U statistic.
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
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
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
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=False, num_workers=4)

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
        "f1_per_label": f1_per_label,
        "auroc_per_label": auroc_per_label.cpu(),
        "auprc_per_label": auprc_per_label.cpu(),
    }
    return metrics


@torch.no_grad()
def test(model, device, test_set, num_pred=40, thr=0.5):
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

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

def _resolve_safetensors_path(path: str) -> str | None:
    """
    Accept either:
      - a directory containing adapter_model.safetensors
      - a direct .safetensors file path
    """
    if path is None:
        return None
    if os.path.isfile(path) and path.endswith(".safetensors"):
        return path
    # old diffusion-style
    cand = os.path.join(path, "adapter_model.safetensors")
    if os.path.exists(cand):
        return cand
    return None


# ------------------------------------------------------------
# ------------------------------------------------------------
_LORA_A_PAT = re.compile(r"^(.*)\.lora_A(?:\.(?:default|adapter))?(?:\.weight)?$")
_LORA_B_PAT = re.compile(r"^(.*)\.lora_B(?:\.(?:default|adapter))?(?:\.weight)?$")

def _split_lora_ab_key(full_key: str):
    mA = _LORA_A_PAT.match(full_key)
    if mA:
        return mA.group(1), "A"
    mB = _LORA_B_PAT.match(full_key)
    if mB:
        return mB.group(1), "B"
    return None, None


# ------------------------------------------------------------
# ------------------------------------------------------------

_LLAMA_LAYER_PAT = re.compile(r"(?:^|\.)(model\.layers)\.(\d+)(?:\.|$)")

def _infer_layer_group(base_key: str) -> str:
    """
    Return a stable string id for "layer group".
    For LLaMA: 'model.layers.<idx>'
    Fallback to your old rules if not matched.
    """
    m = _LLAMA_LAYER_PAT.search(base_key)
    if m:
        return f"{m.group(1)}.{m.group(2)}"  # model.layers.12

    # ---- keep your old diffusion-ish fallbacks (example) ----
    # Minimal fallback:
    return "unknown"


def _infer_module_type(base_key: str) -> str:
    """
    LLaMA module types:
      self_attn.{q_proj,k_proj,v_proj,o_proj}
      mlp.{gate_proj,up_proj,down_proj}
    """
    # self attention
    if ".self_attn." in base_key:
        if ".q_proj" in base_key: return "self_attn.q_proj"
        if ".k_proj" in base_key: return "self_attn.k_proj"
        if ".v_proj" in base_key: return "self_attn.v_proj"
        if ".o_proj" in base_key: return "self_attn.o_proj"
        return "self_attn.other"

    # mlp
    if ".mlp." in base_key:
        if ".gate_proj" in base_key: return "mlp.gate_proj"
        if ".up_proj" in base_key:   return "mlp.up_proj"
        if ".down_proj" in base_key: return "mlp.down_proj"
        return "mlp.other"

    # fallback
    return "other"

def get_canonical_data_with_meta(path: str):
    """
    Returns:
      features: List[(U_T, V_T, S)]  where
        U_T: [d_out, r], V_T: [d_in, r], S: [r]
      meta: List[dict] with layer_id/module_id etc.
    """
    st_path = _resolve_safetensors_path(path)
    if st_path is None:
        return None, None

    try:
        tensors = load_file(st_path)  # dict[str, np.ndarray]
    except Exception as e:
        print(f"[WARN] failed to load safetensors: {st_path} err={e}")
        return None, None

    # group by base key
    groups = {}
    for k, v in tensors.items():
        base, ab = _split_lora_ab_key(k)
        if base is None:
            continue
        if base not in groups:
            groups[base] = {}
        groups[base][ab] = v

    # build deterministic ordering
    entries = []
    for base, ab in groups.items():
        if "A" not in ab or "B" not in ab:
            continue
        module_type = _infer_module_type(base)
        layer_key = _infer_layer_group(base)
        entries.append((layer_key, module_type, base, ab["A"], ab["B"]))

    # sort by (layer_key, module_type, base) for determinism
    entries.sort(key=lambda x: (x[0], x[1], x[2]))

    # map to ids
    layer_keys = [e[0] for e in entries]
    module_types = [e[1] for e in entries]
    layer2id = {k: i for i, k in enumerate(sorted(set(layer_keys)))}
    module2id = {k: i for i, k in enumerate(sorted(set(module_types)))}

    features = []
    meta = []
    for layer_key, module_type, base, A_np, B_np in entries:
        B = torch.tensor(B_np, dtype=torch.float32)  # [d_out, r]
        A = torch.tensor(A_np, dtype=torch.float32)  # [r, d_in]

        # canonical_svd_features expects (B, A_T)
        U_T, V_T, S = canonical_svd_features(B, A.T)

        features.append((U_T, V_T, S))
        meta.append({
            "layer_id": layer2id[layer_key],
            "module_id": module2id[module_type],
            "layer_key": layer_key,
            "module_type": module_type,
            "base_key": base,
        })

    if len(features) == 0:
        return None, None

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


class CachedSVDatasetDir(torch.utils.data.Dataset):
    """
    Lazy shard dataset with working `to_keep`.
    Requirements:
      - cache_dir contains manifest.json (recommended)
      - cache_dir contains cache_part_*.pt (or filenames listed in manifest)
      - each item is a dict with keys: name, features, meta, label
      - features stored as numpy arrays (u,v,s)
    """
    def __init__(self, cache_dir: str, to_keep=None, manifest_name="manifest.json",
                 index_name="name_index.pt", shard_cache_size=1):
        self.cache_dir = cache_dir
        self.shard_cache_size = max(1, int(shard_cache_size))

        # ---- load manifest (preferred) ----
        manifest_path = os.path.join(cache_dir, manifest_name)
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            # stable order
            basenames = sorted(manifest.keys())
            self.shard_paths = [os.path.join(cache_dir, b) for b in basenames]
            self.shard_counts = [int(manifest[b]) for b in basenames]
        else:
            # fallback: glob (not recommended)
            self.shard_paths = sorted(glob.glob(os.path.join(cache_dir, "cache_part_*.pt")))
            if not self.shard_paths:
                raise FileNotFoundError(f"No cache_part_*.pt under {cache_dir}")
            # compute counts by loading each shard once (costly)
            self.shard_counts = []
            for p in self.shard_paths:
                data = torch.load(p, map_location="cpu", weights_only=False)
                self.shard_counts.append(len(data))
                del data

        # prefix sums for global indexing
        self.cum = [0]
        for c in self.shard_counts:
            self.cum.append(self.cum[-1] + c)
        self.total = self.cum[-1]
        if self.total == 0:
            raise ValueError(f"Cache contains 0 items: {cache_dir}")

        # ---- LRU cache for shards ----
        self._cache = {}       # shard_idx -> loaded list
        self._cache_order = [] # LRU order

        # ---- to_keep handling: build/restore name->(shard_idx, local_idx) index ----
        self.mapping = None  # if not None: list[(shard_idx, local_idx)] for kept items
        if to_keep is not None:
            keep = list(to_keep)
            index_path = os.path.join(cache_dir, index_name)

            if os.path.exists(index_path):
                name2pos = torch.load(index_path, map_location="cpu", weights_only=False)
            else:
                name2pos = self._build_name_index(index_path)

            mapped = []
            missing = 0
            for n in keep:
                pos = name2pos.get(n, None)
                if pos is None:
                    missing += 1
                    continue
                mapped.append(tuple(pos))  # (shard_idx, local_idx)

            if len(mapped) == 0:
                raise ValueError(f"to_keep provided but 0 items matched. missing={missing}/{len(keep)}")

            self.mapping = mapped
            if missing > 0:
                print(f"[WARN] to_keep missing {missing}/{len(keep)} names (not found in cache index).")

    def _build_name_index(self, index_path: str):
        """
        Build name -> (shard_idx, local_idx) mapping by scanning shards once.
        Peak memory: one shard at a time.
        """
        print(f"[INDEX] building name index -> {index_path}")
        name2pos = {}
        for si, p in enumerate(self.shard_paths):
            shard = torch.load(p, map_location="cpu", weights_only=False)
            for li, item in enumerate(shard):
                n = item.get("name", None)
                if n is not None:
                    name2pos[n] = (si, li)
            del shard
        os.makedirs(os.path.dirname(index_path), exist_ok=True) if os.path.dirname(index_path) else None
        torch.save(name2pos, index_path)
        print(f"[INDEX] done. total names={len(name2pos)}")
        return name2pos

    def _load_shard(self, shard_idx: int):
        if shard_idx in self._cache:
            # refresh LRU
            self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
            return self._cache[shard_idx]

        shard = torch.load(self.shard_paths[shard_idx], map_location="cpu", weights_only=False)
        self._cache[shard_idx] = shard
        self._cache_order.append(shard_idx)

        # evict
        while len(self._cache_order) > self.shard_cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return shard

    def __len__(self):
        return len(self.mapping) if self.mapping is not None else self.total

    def __getitem__(self, idx):
        if self.mapping is not None:
            shard_idx, local_idx = self.mapping[idx]
        else:
            # global idx -> (shard_idx, local_idx) via prefix sums
            # binary search
            lo, hi = 0, len(self.cum) - 1
            while lo < hi - 1:
                mid = (lo + hi) // 2
                if idx < self.cum[mid]:
                    hi = mid
                else:
                    lo = mid
            shard_idx = lo
            local_idx = idx - self.cum[shard_idx]

        shard = self._load_shard(shard_idx)
        x = shard[local_idx]

        # numpy -> torch
        feats = [(torch.from_numpy(u).float(),
                  torch.from_numpy(v).float(),
                  torch.from_numpy(s).float()) for (u, v, s) in x["features"]]
        y = torch.tensor(x["label"], dtype=torch.float32)
        return {"features": feats, "meta": x["meta"]}, y

import json
from safetensors.torch import load_file as load_sft  # NEW

class RepackedSFTDatasetDir(torch.utils.data.Dataset):
    def __init__(self, split_dir: str, shard_cache_size: int = 2):
        self.split_dir = split_dir
        self.shard_cache_size = max(1, int(shard_cache_size))

        mani_path = os.path.join(split_dir, "manifest.json")
        meta_path = os.path.join(split_dir, "meta.json")
        if not os.path.exists(mani_path):
            raise FileNotFoundError(f"missing manifest.json: {mani_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"missing meta.json: {meta_path}")

        with open(mani_path, "r") as f:
            mani = json.load(f)

        self.shard_files = sorted(mani.keys())
        self.shard_counts = [int(mani[k]) for k in self.shard_files]
        self.shard_paths = [os.path.join(split_dir, k) for k in self.shard_files]

        # prefix sums: global idx -> (shard_idx, local_idx)
        self.cum = [0]
        for c in self.shard_counts:
            self.cum.append(self.cum[-1] + c)
        self.total = self.cum[-1]
        if self.total == 0:
            raise ValueError(f"empty repacked split dir: {split_dir}")

        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self.num_pos = len(self.meta)

        # LRU cache for loaded shards
        self._cache = {}       # shard_idx -> tensor_dict
        self._cache_order = [] # LRU

    def __len__(self):
        return self.total

    def _load_shard(self, shard_idx: int):
        if shard_idx in self._cache:
            self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
            return self._cache[shard_idx]

        td = load_sft(self.shard_paths[shard_idx])  # dict[str, Tensor] on CPU
        self._cache[shard_idx] = td
        self._cache_order.append(shard_idx)

        while len(self._cache_order) > self.shard_cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return td

    def _global_to_local(self, idx: int):
        # binary search in prefix sums
        lo, hi = 0, len(self.cum) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if idx < self.cum[mid]:
                hi = mid
            else:
                lo = mid
        shard_idx = lo
        local_idx = idx - self.cum[shard_idx]
        return shard_idx, local_idx

    def __getitem__(self, idx):
        shard_idx, local_idx = self._global_to_local(idx)
        td = self._load_shard(shard_idx)

        feats = []
        for p in range(self.num_pos):
            U = td[f"U_{p}"][local_idx].float()  # [d_out, r]
            V = td[f"V_{p}"][local_idx].float()  # [d_in,  r]
            S = td[f"S_{p}"][local_idx].float()
            # allow [r,1] or [r]
            if S.ndim == 2 and S.shape[1] == 1:
                S = S.view(-1)
            feats.append((U, V, S))

        y = td["Y"][local_idx].float()  # [C]
        return {"features": feats, "meta": self.meta}, y
