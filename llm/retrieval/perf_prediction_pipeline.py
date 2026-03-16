#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from collections import OrderedDict, defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as st_load_torch
from safetensors.torch import save_file as st_save_torch
from torch import nn
from torch.utils.data import Sampler

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compat.baseline_models import GLInvariantMLP, FlattenMLP_Layerwise, CNN1D_Layerwise, TokenViT_Layerwise
from compat.baseline_utils import (
    collate_layerwise_flat as baseline_collate_layerwise_flat,
    collate_layerwise_tokenized as baseline_collate_layerwise_tokenized,
    to_device as baseline_to_device,
)
from common.w2t_models import FullTransformer
from common.w2t_utils import canonical_svd_features as w2t_canonical_svd_features


REQUIRED_METADATA_COLUMNS = [
    "run_id",
    "time",
    "run_name",
    "base_model",
    "dataset",
    "lr",
    "epochs",
    "batch_size",
    "grad_accum",
    "max_len",
    "warmup_ratio",
    "weight_decay",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "seed",
    "test_loss",
    "test_ppl",
    "test_acc",
    "safetensors_path",
    "sidecar_config",
]
TARGET_COLUMNS = ["test_acc", "test_loss", "test_ppl"]
LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"
EPS = 1e-8

LORA_KEY_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Common PEFT LoRA keys:
    #   ...lora_A.weight
    #   ...lora_B.weight
    #   ...lora_A.default.weight
    #   ...lora_B.default.weight
    (re.compile(r"^(?P<base>.+?)\.(?:lora_A|lora_down)(?:\.[^.]+)?\.weight$", re.IGNORECASE), "A"),
    (re.compile(r"^(?P<base>.+?)\.(?:lora_B|lora_up)(?:\.[^.]+)?\.weight$", re.IGNORECASE), "B"),
]


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def natural_sort_key(text: str) -> List[Any]:
    parts = re.split(r"(\d+)", str(text))
    out: List[Any] = []
    for part in parts:
        if part.isdigit():
            out.append(int(part))
        else:
            out.append(part.lower())
    return out


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def upsert_csv(
    path: Path,
    df_new: pd.DataFrame,
    key_cols: Sequence[str],
    sort_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if df_new.empty:
        return df_new

    frames = []
    if path.exists():
        try:
            df_old = pd.read_csv(path)
            frames.append(df_old)
        except Exception:
            pass
    frames.append(df_new)
    merged = pd.concat(frames, ignore_index=True)

    valid_keys = [c for c in key_cols if c in merged.columns]
    if valid_keys:
        merged = merged.drop_duplicates(subset=valid_keys, keep="last")
    else:
        merged = merged.drop_duplicates(keep="last")

    valid_sort = [c for c in (sort_cols or []) if c in merged.columns]
    if valid_sort:
        merged = merged.sort_values(valid_sort).reset_index(drop=True)
    else:
        merged = merged.reset_index(drop=True)

    merged.to_csv(path, index=False)
    return merged


def _install_legacy_main_globals() -> None:
    """Make old checkpoints saved with argparse func references loadable across entry scripts."""
    import sys as _sys

    main_mod = _sys.modules.get("__main__")
    if main_mod is None:
        return

    def _noop(*args, **kwargs):
        return None

    for name in ["cmd_train", "cmd_predict", "cmd_benchmark", "main"]:
        if not hasattr(main_mod, name):
            setattr(main_mod, name, _noop)


def torch_load_compat(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    # Pass weights_only explicitly to avoid future default-change warnings.
    _install_legacy_main_globals()
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Backward compatibility for older torch versions.
        return torch.load(path, map_location=map_location)


def sanitize_args_for_checkpoint(args_ns: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in vars(args_ns).items():
        if callable(v):
            continue
        if v is None or isinstance(v, (bool, int, float, str)):
            out[str(k)] = v
            continue
        if isinstance(v, (list, tuple)):
            if all(x is None or isinstance(x, (bool, int, float, str)) for x in v):
                out[str(k)] = list(v)
            else:
                out[str(k)] = str(v)
            continue
        if isinstance(v, dict):
            simple = True
            for kk, vv in v.items():
                if not isinstance(kk, (str, int, float, bool)):
                    simple = False
                    break
                if not (vv is None or isinstance(vv, (bool, int, float, str))):
                    simple = False
                    break
            out[str(k)] = dict(v) if simple else str(v)
            continue
        out[str(k)] = str(v)
    return out


def _state_dict_cpu_contiguous(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        out[str(k)] = v.detach().cpu().contiguous()
    return out


def load_checkpoint_with_fallback(ckpt_path: Path) -> Dict[str, Any]:
    try:
        ckpt = torch_load_compat(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            return ckpt
        raise RuntimeError(f"Invalid checkpoint payload: {ckpt_path}")
    except Exception as primary_exc:
        fallback_pairs = [
            (ckpt_path.parent / "best_model_meta.json", ckpt_path.parent / "best_state.safetensors"),
            (ckpt_path.with_suffix(".meta.json"), ckpt_path.with_suffix(".safetensors")),
        ]
        for meta_path, state_path in fallback_pairs:
            if not (meta_path.exists() and state_path.exists()):
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if not isinstance(meta, dict):
                    raise RuntimeError(f"Invalid meta json: {meta_path}")
                meta["state_dict"] = st_load_torch(str(state_path))
                meta["_checkpoint_source"] = "safetensors_fallback"
                return meta
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Failed loading checkpoint {ckpt_path} and fallback bundle "
                    f"({meta_path}, {state_path}). primary={type(primary_exc).__name__}: {primary_exc}; "
                    f"fallback={type(fallback_exc).__name__}: {fallback_exc}"
                ) from fallback_exc
        raise RuntimeError(
            f"Failed loading checkpoint {ckpt_path}. "
            f"error={type(primary_exc).__name__}: {primary_exc}. "
            "Fallback bundle not found."
        ) from primary_exc


def _infer_mlp_flat_dims_from_state_dict(state_dict: Dict[str, Any]) -> Optional[List[int]]:
    # FlattenMLP encoders[i][0] is Linear(d -> hidden_dim), so in_features reveals per-layer dim.
    pat = re.compile(r"^base\.encoders\.(\d+)\.0\.weight$")
    dims: Dict[int, int] = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor) or v.ndim != 2:
            continue
        m = pat.match(str(k))
        if not m:
            continue
        idx = int(m.group(1))
        dims[idx] = int(v.shape[1])
    if not dims:
        return None
    return [dims[i] for i in sorted(dims.keys())]


def adapt_state_dict_for_model(model_type: str, state_dict: Dict[str, Any], model: nn.Module) -> Dict[str, Any]:
    if model_type != "glnet":
        return state_dict

    src_keys = [str(k) for k in state_dict.keys()]
    dst_keys = [str(k) for k in model.state_dict().keys()]
    src_has_underscore = any(".w_1" in k or ".w_2" in k for k in src_keys)
    src_has_plain = any(".w1" in k or ".w2" in k for k in src_keys)
    dst_wants_underscore = any(".w_1" in k or ".w_2" in k for k in dst_keys)
    dst_wants_plain = any(".w1" in k or ".w2" in k for k in dst_keys)

    if src_has_underscore and dst_wants_plain:
        return {
            str(k).replace(".w_1", ".w1").replace(".w_2", ".w2"): v
            for k, v in state_dict.items()
        }
    if src_has_plain and dst_wants_underscore:
        return {
            str(k).replace(".w1", ".w_1").replace(".w2", ".w_2"): v
            for k, v in state_dict.items()
        }
    return state_dict


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def to_device(obj: Any, device: torch.device) -> Any:
    # Keep device transfer behavior aligned with normalized baseline utils.
    return baseline_to_device(obj, device)


def canonical_svd_features(B: torch.Tensor, A_T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Keep canonicalization behavior aligned with normalized w2t utils.
    return w2t_canonical_svd_features(B, A_T)


def parse_path_maps(path_map_items: Sequence[str], path_map_json: Optional[str]) -> List[Tuple[str, str]]:
    mapping: Dict[str, str] = {}
    if path_map_json:
        payload = json.loads(Path(path_map_json).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("--path-map-json must be {source_prefix: target_prefix}.")
        for k, v in payload.items():
            mapping[str(k)] = str(v)
    for item in path_map_items:
        if "=" not in item:
            raise ValueError(f"Invalid --path-map item: {item}. Expected source=target.")
        src, dst = item.split("=", 1)
        mapping[src] = dst
    return sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)


def resolve_safetensors_path(raw_path: str, path_maps: Sequence[Tuple[str, str]]) -> Path:
    path_str = str(raw_path)
    for src, dst in path_maps:
        if path_str.startswith(src):
            path_str = dst + path_str[len(src):]
            break
    candidate = Path(path_str)
    if candidate.is_dir():
        candidate = candidate / "adapter_model.safetensors"
    return candidate


def _allocate_split_counts(n: int, ratios: Sequence[float]) -> Tuple[int, int, int]:
    raw = np.array(ratios, dtype=np.float64) * n
    base = np.floor(raw).astype(int)
    remain = int(n - base.sum())
    frac = raw - base
    order = np.argsort(-frac)
    for i in range(remain):
        base[order[i % len(order)]] += 1
    return int(base[0]), int(base[1]), int(base[2])


def _build_strata(df: pd.DataFrame, target_col: str, n_bins: int) -> pd.Series:
    strata = df["lora_r"].astype(str)
    if target_col not in df.columns:
        return strata
    nunique = int(df[target_col].nunique(dropna=True))
    q = min(max(2, n_bins), nunique)
    if q < 2:
        return strata
    try:
        bins = pd.qcut(df[target_col], q=q, duplicates="drop")
        strata = strata + "|" + bins.astype(str)
    except Exception:
        pass
    return strata


def _split_dataframe(
    df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_target: str,
    stratify_bins: int,
) -> pd.Series:
    if not math.isclose(train_ratio + valid_ratio + test_ratio, 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("train_ratio + valid_ratio + test_ratio must sum to 1.")

    strata = _build_strata(df, stratify_target, stratify_bins)
    by_strata: Dict[str, List[int]] = defaultdict(list)
    for idx, s in zip(df.index.tolist(), strata.tolist()):
        by_strata[str(s)].append(idx)

    rng = np.random.default_rng(seed)
    split = pd.Series(index=df.index, dtype="string")
    for _, idx_list in by_strata.items():
        idx_arr = np.array(idx_list, dtype=np.int64)
        rng.shuffle(idx_arr)
        n_train, n_valid, n_test = _allocate_split_counts(
            len(idx_arr), [train_ratio, valid_ratio, test_ratio]
        )
        split.loc[idx_arr[:n_train]] = "train"
        split.loc[idx_arr[n_train : n_train + n_valid]] = "valid"
        split.loc[idx_arr[n_train + n_valid : n_train + n_valid + n_test]] = "test"
    return split.astype(str)


def _summarize_splits(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"global_count": int(len(df)), "splits": {}}
    for split_name in ["train", "valid", "test"]:
        sub = df[df["split"] == split_name]
        stats: Dict[str, Any] = {"count": int(len(sub))}
        for col in TARGET_COLUMNS:
            desc = sub[col].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
            stats[col] = {k: float(v) for k, v in desc.items()}
        stats["lora_r_counts"] = {
            str(k): int(v) for k, v in sub["lora_r"].value_counts().sort_index().items()
        }
        out["splits"][split_name] = stats
    return out


def cmd_prepare(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    csv_files = sorted(input_dir.glob(args.glob), key=lambda p: natural_sort_key(p.name))
    if not csv_files:
        raise FileNotFoundError(f"No csv files matched: {input_dir / args.glob}")

    frames = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df["source_csv"] = csv_path.name
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)

    missing_cols = [c for c in REQUIRED_METADATA_COLUMNS if c not in merged.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    merged = merged.drop_duplicates(subset=["run_id"], keep="first")
    merged = merged.sort_values("run_id").reset_index(drop=True)
    merged["split"] = _split_dataframe(
        merged,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_target=args.stratify_target,
        stratify_bins=args.stratify_bins,
    )

    out_dir = ensure_dir(args.output_dir)
    merged.to_csv(out_dir / "all_metadata.csv", index=False)
    merged[merged["split"] == "train"].to_csv(out_dir / "train.csv", index=False)
    merged[merged["split"] == "valid"].to_csv(out_dir / "valid.csv", index=False)
    merged[merged["split"] == "test"].to_csv(out_dir / "test.csv", index=False)

    split_ids = {
        "train": merged.loc[merged["split"] == "train", "run_id"].astype(int).tolist(),
        "valid": merged.loc[merged["split"] == "valid", "run_id"].astype(int).tolist(),
        "test": merged.loc[merged["split"] == "test", "run_id"].astype(int).tolist(),
    }
    dump_json(out_dir / "split_run_ids.json", split_ids)
    dump_json(out_dir / "split_summary.json", _summarize_splits(merged))

    print(f"[prepare] merged rows: {len(merged)}")
    print("[prepare] split sizes:", {k: len(v) for k, v in split_ids.items()})
    print(f"[prepare] outputs: {out_dir}")


def _infer_module_type(base_key: str) -> str:
    patterns = [
        (re.compile(r"\.(q_proj|to_q)\b"), "q"),
        (re.compile(r"\.(k_proj|to_k)\b"), "k"),
        (re.compile(r"\.(v_proj|to_v)\b"), "v"),
        (re.compile(r"\.(o_proj|to_out)\b"), "o"),
        (re.compile(r"\.(up_proj)\b"), "up"),
        (re.compile(r"\.(down_proj)\b"), "down"),
        (re.compile(r"\.(gate_proj)\b"), "gate"),
        (re.compile(r"\.(proj_in)\b"), "proj_in"),
        (re.compile(r"\.(proj_out)\b"), "proj_out"),
        (re.compile(r"\.(mlp|ffn|feed_forward|ff)\b"), "mlp"),
    ]
    for rgx, name in patterns:
        if rgx.search(base_key):
            return name
    parts = base_key.split(".")
    return parts[-1] if parts else "unknown"


def _infer_layer_group(base_key: str) -> str:
    s = base_key
    s = re.sub(r"\.(q_proj|k_proj|v_proj|o_proj|to_q|to_k|to_v|to_out)\b", "", s)
    s = re.sub(r"\.(up_proj|down_proj|gate_proj|proj_in|proj_out)\b", "", s)
    s = re.sub(r"\.(mlp|ffn|feed_forward|ff)\b", "", s)
    s = re.sub(r"\.+", ".", s).strip(".")
    return s


_LAYER_PATTERNS = [
    re.compile(r"\.(?:layers|layer|h|blocks|block|transformer_blocks)\.(\d+)\b"),
    re.compile(r"\.(?:down_blocks|up_blocks)\.(\d+)\b"),
    re.compile(r"\.(\d+)\.(?:self_attn|attn|mlp|ffn)\b"),
    re.compile(r"\.(\d+)\."),
]


def _infer_layer_index(base_key: str) -> int:
    for pat in _LAYER_PATTERNS:
        m = pat.search(base_key)
        if m:
            return int(m.group(1))
    return 10**9


def _entry_sort_key(entry: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        _infer_layer_index(entry["base_key"]),
        natural_sort_key(entry["layer_key"]),
        entry["module_type"],
        entry["base_key"],
    )


def _parse_lora_key(key: str) -> Tuple[Optional[str], Optional[str]]:
    for pat, side in LORA_KEY_PATTERNS:
        m = pat.match(key)
        if m:
            return m.group("base"), side
    return None, None


def _extract_lora_entries(tensors_by_key: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    for key, tensor in tensors_by_key.items():
        base_key, side = _parse_lora_key(key)
        if base_key is not None and side is not None:
            groups[base_key][side] = tensor

    entries: List[Dict[str, Any]] = []
    for base_key, ab in groups.items():
        if "A" not in ab or "B" not in ab:
            continue
        A = ab["A"].detach().cpu().float()
        B = ab["B"].detach().cpu().float()
        if A.ndim != 2 or B.ndim != 2:
            continue
        if A.shape[0] != B.shape[1]:
            raise ValueError(f"Rank mismatch at {base_key}: A{tuple(A.shape)} vs B{tuple(B.shape)}.")
        layer_key = _infer_layer_group(base_key)
        module_type = _infer_module_type(base_key)
        entries.append(
            {
                "base_key": base_key,
                "layer_key": layer_key,
                "module_type": module_type,
                "A": A,
                "B": B,
            }
        )
    entries.sort(key=_entry_sort_key)
    return entries


def _init_schema(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    position_keys = [e["base_key"] for e in entries]
    position_layer_keys = [e["layer_key"] for e in entries]
    position_module_types = [e["module_type"] for e in entries]

    uniq_layers = sorted(set(position_layer_keys), key=natural_sort_key)
    uniq_modules = sorted(set(position_module_types), key=natural_sort_key)
    layer2id = {k: i for i, k in enumerate(uniq_layers)}
    module2id = {k: i for i, k in enumerate(uniq_modules)}

    return {
        "position_keys": position_keys,
        "position_layer_keys": position_layer_keys,
        "position_module_types": position_module_types,
        "flat_layer_keys": uniq_layers,
        "layer2id": layer2id,
        "module2id": module2id,
        "layer_ids": [int(layer2id[k]) for k in position_layer_keys],
        "module_ids": [int(module2id[k]) for k in position_module_types],
        "num_layers": len(uniq_layers),
        "num_modules": len(uniq_modules),
    }


def _align_entries_to_schema(entries: List[Dict[str, Any]], schema: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    current_map = {e["base_key"]: e for e in entries}
    aligned: List[Dict[str, Any]] = []
    for key in schema["position_keys"]:
        if key not in current_map:
            return None, f"missing_key:{key}"
        aligned.append(current_map[key])
    return aligned, ""


def _to_storage_dtype(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.float16:
        return t.half()
    return t.float()


def _build_w2t_rep(entries: List[Dict[str, Any]], schema: Dict[str, Any], out_dtype: torch.dtype) -> Dict[str, Any]:
    features: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for e in entries:
        U_T, V_T, S = canonical_svd_features(e["B"], e["A"].T)
        features.append(
            (
                _to_storage_dtype(U_T, out_dtype),
                _to_storage_dtype(V_T, out_dtype),
                _to_storage_dtype(S, out_dtype),
            )
        )
    return {
        "features": features,
        "layer_ids": list(schema["layer_ids"]),
        "module_ids": list(schema["module_ids"]),
    }


def _build_glnet_rep(entries: List[Dict[str, Any]], out_dtype: torch.dtype) -> Dict[str, Any]:
    uvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for e in entries:
        u = _to_storage_dtype(e["B"], out_dtype)     # [d_out, r]
        v = _to_storage_dtype(e["A"].T, out_dtype)   # [d_in, r]
        uvs.append((u, v))
    return {"uvs": uvs}


def _build_flat_rep(entries: List[Dict[str, Any]], schema: Dict[str, Any], out_dtype: torch.dtype) -> Dict[str, Any]:
    layer_buf: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for e in entries:
        layer_buf[e["layer_key"]].append(e["B"].reshape(-1))
        layer_buf[e["layer_key"]].append(e["A"].reshape(-1))

    layers: List[torch.Tensor] = []
    for layer_key in schema["flat_layer_keys"]:
        vecs = layer_buf.get(layer_key, [])
        if not vecs:
            layers.append(torch.zeros(1, dtype=out_dtype))
        else:
            layers.append(_to_storage_dtype(torch.cat(vecs, dim=0), out_dtype))
    return {"layer_keys": list(schema["flat_layer_keys"]), "layers": layers}


def _build_token_rep(flat_rep: Dict[str, Any], token_size: int, out_dtype: torch.dtype) -> Dict[str, Any]:
    token_layers: List[torch.Tensor] = []
    for vec in flat_rep["layers"]:
        x = vec.float()
        n_token = int(math.ceil(x.numel() / float(token_size)))
        total = n_token * token_size
        if total > x.numel():
            x = F.pad(x, (0, total - x.numel()))
        token_layers.append(_to_storage_dtype(x.view(n_token, token_size), out_dtype))
    return {
        "layer_keys": list(flat_rep["layer_keys"]),
        "layers": token_layers,
        "token_size": int(token_size),
    }


def _load_split_map(split_dir: Path) -> Dict[int, str]:
    split_map: Dict[int, str] = {}
    name_map = {
        "train": ["train.csv"],
        "valid": ["valid.csv", "val.csv"],
        "test": ["test.csv"],
    }
    for split, candidates in name_map.items():
        chosen = None
        for c in candidates:
            p = split_dir / c
            if p.exists():
                chosen = p
                break
        if chosen is None:
            raise FileNotFoundError(f"Missing split file for {split} under: {split_dir}")
        df = pd.read_csv(chosen)
        for rid in df["run_id"].tolist():
            split_map[int(rid)] = split
    return split_map


def cmd_cache(args: argparse.Namespace) -> None:
    metadata_csv = Path(args.metadata_csv)
    split_dir = Path(args.split_dir)
    out_dir = ensure_dir(args.output_dir)

    if not metadata_csv.exists():
        raise FileNotFoundError(metadata_csv)
    split_map = _load_split_map(split_dir)
    metadata = pd.read_csv(metadata_csv)
    metadata = metadata[metadata["run_id"].isin(split_map.keys())].copy()
    metadata = metadata.sort_values("run_id").reset_index(drop=True)
    if args.max_samples is not None:
        metadata = metadata.iloc[: args.max_samples].copy()

    req = [x.strip().lower() for x in args.representations.split(",") if x.strip()]
    core_reps = set()
    if "w2t" in req:
        core_reps.add("w2t")
    if "glnet" in req:
        core_reps.add("glnet")
    if "flat" in req or "mlp" in req or "cnn" in req or "vit" in req or "token" in req:
        core_reps.add("flat")
    if "token" in req or "vit" in req:
        core_reps.add("token")
    if not core_reps:
        raise ValueError("No valid representation requested. Use --representations with at least one of w2t,glnet,mlp,cnn,vit.")

    out_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    path_maps = parse_path_maps(args.path_map, args.path_map_json)

    schema: Optional[Dict[str, Any]] = None
    shard_items: List[Dict[str, Any]] = []
    shard_infos: List[Dict[str, Any]] = []
    index_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    mismatch_rows: List[Dict[str, Any]] = []
    shard_id = 0

    def flush_shard() -> None:
        nonlocal shard_items, shard_id
        if not shard_items:
            return
        shard_name = f"cache_shard_{shard_id:05d}.pt"
        torch.save({"items": shard_items}, out_dir / shard_name)
        shard_infos.append({"file": shard_name, "num_items": len(shard_items)})
        shard_items = []
        shard_id += 1

    iterator = tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Caching")
    for row in iterator:
        run_id = int(row.run_id)
        split = split_map.get(run_id)
        if split is None:
            continue

        resolved_path = resolve_safetensors_path(str(row.safetensors_path), path_maps)
        if not resolved_path.exists():
            miss = {
                "run_id": run_id,
                "split": split,
                "raw_path": str(row.safetensors_path),
                "resolved_path": str(resolved_path),
                "reason": "missing_file",
            }
            missing_rows.append(miss)
            if args.skip_missing:
                continue
            raise FileNotFoundError(str(miss))

        try:
            tensors = st_load_torch(str(resolved_path))
            entries = _extract_lora_entries(tensors)
        except Exception as exc:
            miss = {
                "run_id": run_id,
                "split": split,
                "raw_path": str(row.safetensors_path),
                "resolved_path": str(resolved_path),
                "reason": f"load_error:{type(exc).__name__}:{exc}",
            }
            missing_rows.append(miss)
            if args.skip_missing:
                continue
            raise

        if not entries:
            key_preview = [str(k) for k in list(tensors.keys())[:5]]
            miss = {
                "run_id": run_id,
                "split": split,
                "raw_path": str(row.safetensors_path),
                "resolved_path": str(resolved_path),
                "reason": "no_lora_pairs",
                "key_preview": " | ".join(key_preview),
            }
            missing_rows.append(miss)
            if args.skip_missing:
                continue
            raise RuntimeError(f"No LoRA A/B pairs found in {resolved_path}")

        if schema is None:
            schema = _init_schema(entries)
        else:
            aligned, reason = _align_entries_to_schema(entries, schema)
            if aligned is None:
                mismatch_rows.append(
                    {
                        "run_id": run_id,
                        "split": split,
                        "raw_path": str(row.safetensors_path),
                        "resolved_path": str(resolved_path),
                        "reason": reason,
                    }
                )
                if args.strict_schema:
                    continue
                else:
                    continue
            entries = aligned

        if schema is None:
            continue
        aligned_entries, reason = _align_entries_to_schema(entries, schema)
        if aligned_entries is None:
            mismatch_rows.append(
                {
                    "run_id": run_id,
                    "split": split,
                    "raw_path": str(row.safetensors_path),
                    "resolved_path": str(resolved_path),
                    "reason": reason,
                }
            )
            continue
        entries = aligned_entries

        sample: Dict[str, Any] = {
            "run_id": run_id,
            "split": split,
            "targets": {
                "test_acc": float(row.test_acc),
                "test_loss": float(row.test_loss),
                "test_ppl": float(row.test_ppl),
            },
            "meta": {
                "time": str(row.time),
                "run_name": str(row.run_name),
                "base_model": str(row.base_model),
                "dataset": str(row.dataset),
                "lr": float(row.lr),
                "epochs": int(row.epochs),
                "batch_size": int(row.batch_size),
                "grad_accum": int(row.grad_accum),
                "max_len": int(row.max_len),
                "warmup_ratio": float(row.warmup_ratio),
                "weight_decay": float(row.weight_decay),
                "lora_r": int(row.lora_r),
                "lora_alpha": int(row.lora_alpha),
                "lora_dropout": float(row.lora_dropout),
                "target_modules": str(row.target_modules),
                "seed": int(row.seed),
                "safetensors_path": str(row.safetensors_path),
                "sidecar_config": str(row.sidecar_config),
                "resolved_safetensors_path": str(resolved_path),
            },
        }

        flat_rep: Optional[Dict[str, Any]] = None
        if "w2t" in core_reps:
            sample["w2t"] = _build_w2t_rep(entries, schema, out_dtype)
        if "glnet" in core_reps:
            sample["glnet"] = _build_glnet_rep(entries, out_dtype)
        if "flat" in core_reps or "token" in core_reps:
            flat_rep = _build_flat_rep(entries, schema, out_dtype)
            if "flat" in core_reps:
                sample["flat"] = flat_rep
        if "token" in core_reps:
            if flat_rep is None:
                flat_rep = _build_flat_rep(entries, schema, out_dtype)
            sample["token"] = _build_token_rep(flat_rep, token_size=args.token_size, out_dtype=out_dtype)

        current_shard = f"cache_shard_{shard_id:05d}.pt"
        shard_items.append(sample)
        index_rows.append(
            {
                "run_id": run_id,
                "split": split,
                "shard": current_shard,
                "offset": len(shard_items) - 1,
                "targets": sample["targets"],
                "lora_r": int(row.lora_r),
            }
        )
        if len(shard_items) >= args.shard_size:
            flush_shard()

    flush_shard()

    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(out_dir / "cache_missing.csv", index=False)
    if mismatch_rows:
        pd.DataFrame(mismatch_rows).to_csv(out_dir / "cache_schema_mismatch.csv", index=False)

    manifest = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metadata_csv": str(metadata_csv.resolve()),
        "split_dir": str(split_dir.resolve()),
        "representations": sorted(core_reps),
        "requested_representations": req,
        "dtype": args.dtype,
        "token_size": int(args.token_size),
        "target_columns": TARGET_COLUMNS,
        "schema": schema,
        "index": index_rows,
        "shards": shard_infos,
        "stats": {
            "requested_rows": int(len(metadata)),
            "cached_rows": int(len(index_rows)),
            "missing_rows": int(len(missing_rows)),
            "schema_mismatch_rows": int(len(mismatch_rows)),
            "num_shards": int(len(shard_infos)),
        },
    }
    manifest_path = out_dir / "manifest.json"
    dump_json(manifest_path, manifest)

    print(f"[cache] cached rows: {len(index_rows)} / {len(metadata)}")
    print(f"[cache] shards: {len(shard_infos)}")
    print(f"[cache] missing rows: {len(missing_rows)}")
    print(f"[cache] schema mismatch rows: {len(mismatch_rows)}")
    print(f"[cache] manifest: {manifest_path}")


class ShardCacheReader:
    def __init__(self, manifest_path: str | Path, max_cached_shards: int = 2):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.max_cached_shards = max(1, int(max_cached_shards))
        self._cache: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()

    def _load_shard(self, shard_name: str) -> List[Dict[str, Any]]:
        if shard_name in self._cache:
            data = self._cache.pop(shard_name)
            self._cache[shard_name] = data
            return data
        payload = torch_load_compat(self.root / shard_name, map_location="cpu")
        if isinstance(payload, dict) and "items" in payload:
            items = payload["items"]
        else:
            items = payload
        self._cache[shard_name] = items
        while len(self._cache) > self.max_cached_shards:
            self._cache.popitem(last=False)
        return items

    def get_item(self, shard_name: str, offset: int) -> Dict[str, Any]:
        shard = self._load_shard(shard_name)
        return shard[offset]


def select_feature_from_item(item: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    if model_type == "w2t":
        feat = item.get("w2t")
    elif model_type == "glnet":
        feat = item.get("glnet")
    elif model_type in ("mlp", "cnn"):
        feat = item.get("flat")
    elif model_type == "vit":
        feat = item.get("token")
    else:
        raise ValueError(f"Unknown model_type={model_type}")
    if feat is None:
        raise KeyError(
            f"Feature for model_type={model_type} not found in cache. "
            "Rebuild cache with required representation."
        )
    return feat


class LoRACacheDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        reader: ShardCacheReader,
        split: str,
        model_type: str,
        target_col: str,
    ):
        self.reader = reader
        self.model_type = model_type
        self.target_col = target_col
        all_entries = self.reader.manifest["index"]
        self.entries = [e for e in all_entries if e["split"] == split]
        if not self.entries:
            raise ValueError(f"No cache entries for split={split}.")

    def __len__(self) -> int:
        return len(self.entries)

    def _select_feature(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return select_feature_from_item(item, self.model_type)

    def __getitem__(self, idx: int):
        ent = self.entries[idx]
        item = self.reader.get_item(ent["shard"], int(ent["offset"]))
        feat = self._select_feature(item)
        y = float(item["targets"][self.target_col])
        run_id = int(item["run_id"])
        return feat, y, run_id

    def target_values(self) -> List[float]:
        vals = []
        for e in self.entries:
            vals.append(float(e["targets"][self.target_col]))
        return vals

    def shard_name(self, idx: int) -> str:
        return str(self.entries[idx]["shard"])


class PackedLoRACacheDataset(torch.utils.data.Dataset):
    """Packed cache dataset for high-throughput training.

    Supports two formats:
    1) by-model file: items = [{'feature': ..., 'targets': ..., 'run_id': ...}], payload has model_type
    2) by-split file: items = [{'item': full_sample_dict}] or full sample dict directly
    """

    def __init__(self, packed_path: str | Path, target_col: str, model_type: str):
        payload = torch_load_compat(Path(packed_path), map_location="cpu")
        self.model_type = str(model_type)
        self.split = str(payload.get("split", "unknown"))
        self.items = payload["items"]
        self.target_col = target_col
        self._packed_mode = str(payload.get("packed_mode", "by_model" if "model_type" in payload else "by_split"))
        if not self.items:
            raise ValueError(f"Packed cache has no items: {packed_path}")

    def __len__(self) -> int:
        return len(self.items)

    def _decode(self, rec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
        if self._packed_mode == "by_model":
            feat = rec["feature"]
            targets = rec["targets"]
            run_id = int(rec["run_id"])
            return feat, targets, run_id

        # by_split: keep whole sample once, select feature by model at read time
        base = rec.get("item", rec)
        feat = select_feature_from_item(base, self.model_type)
        targets = base["targets"]
        run_id = int(base["run_id"])
        return feat, targets, run_id

    def __getitem__(self, idx: int):
        feat, targets, run_id = self._decode(self.items[idx])
        y = float(targets[self.target_col])
        return feat, y, run_id

    def target_values(self) -> List[float]:
        out = []
        for rec in self.items:
            _, targets, _ = self._decode(rec)
            out.append(float(targets[self.target_col]))
        return out


class ShardAwareSampler(Sampler[int]):
    """Shard-local sampler to reduce random shard thrashing on disk IO.

    It shuffles shard order and in-shard sample order, while yielding contiguous
    indices per shard so each shard can stay hot in cache.
    """

    def __init__(self, dataset: LoRACacheDataset, shuffle: bool = True, seed: int = 42):
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        groups: Dict[str, List[int]] = defaultdict(list)
        for i in range(len(dataset)):
            groups[dataset.shard_name(i)].append(i)
        self.groups = groups
        self.shards = sorted(groups.keys(), key=natural_sort_key)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        shards = list(self.shards)
        if self.shuffle:
            rng.shuffle(shards)
        for shard in shards:
            idxs = list(self.groups[shard])
            if self.shuffle:
                rng.shuffle(idxs)
            for idx in idxs:
                yield idx


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _resolve_model_types_arg(s: str) -> List[str]:
    if s == "all":
        return ["w2t", "glnet", "mlp", "cnn", "vit"]
    vals = _parse_csv_list(s)
    for v in vals:
        if v not in {"w2t", "glnet", "mlp", "cnn", "vit"}:
            raise ValueError(f"Unknown model type in list: {v}")
    return vals


def _resolve_splits_arg(s: str) -> List[str]:
    vals = _parse_csv_list(s)
    for v in vals:
        if v not in {"train", "valid", "test"}:
            raise ValueError(f"Unknown split: {v}")
    return vals


def _packed_file_path(out_dir: Path, model_type: str, split: str) -> Path:
    return out_dir / f"packed_{model_type}_{split}.pt"


def _packed_split_file_path(out_dir: Path, split: str) -> Path:
    return out_dir / f"packed_split_{split}.pt"


def _required_rep_keys(model_types: List[str]) -> List[str]:
    rep = set()
    for m in model_types:
        if m == "w2t":
            rep.add("w2t")
        elif m == "glnet":
            rep.add("glnet")
        elif m in ("mlp", "cnn"):
            rep.add("flat")
        elif m == "vit":
            rep.add("token")
    return sorted(rep)


def cmd_pack(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.output_dir)
    reader = ShardCacheReader(args.manifest, max_cached_shards=args.max_shards_in_mem)
    index = reader.manifest["index"]
    model_types = _resolve_model_types_arg(args.model_types)
    splits = _resolve_splits_arg(args.splits)
    required_rep_keys = _required_rep_keys(model_types)

    summary_rows: List[Dict[str, Any]] = []

    if args.pack_mode == "by_model":
        for model_type in model_types:
            for split in splits:
                out_path = _packed_file_path(out_dir, model_type, split)
                if out_path.exists() and (not args.overwrite):
                    print(f"[pack] skip existing: {out_path}", flush=True)
                    continue

                subset = [e for e in index if e["split"] == split]
                if not subset:
                    print(f"[pack] no rows for split={split}, skip", flush=True)
                    continue

                items: List[Dict[str, Any]] = []
                for ent in tqdm(subset, desc=f"Pack {model_type}/{split}"):
                    item = reader.get_item(ent["shard"], int(ent["offset"]))
                    feat = select_feature_from_item(item, model_type)
                    items.append(
                        {
                            "run_id": int(item["run_id"]),
                            "targets": {
                                "test_acc": float(item["targets"]["test_acc"]),
                                "test_loss": float(item["targets"]["test_loss"]),
                                "test_ppl": float(item["targets"]["test_ppl"]),
                            },
                            "feature": feat,
                        }
                    )

                payload = {
                    "version": 2,
                    "packed_mode": "by_model",
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model_type": model_type,
                    "split": split,
                    "num_items": len(items),
                    "source_manifest": str(Path(args.manifest).resolve()),
                    "items": items,
                }
                torch.save(payload, out_path)
                summary_rows.append(
                    {
                        "packed_mode": "by_model",
                        "model_type": model_type,
                        "split": split,
                        "num_items": len(items),
                        "path": str(out_path),
                    }
                )
                print(f"[pack] saved {model_type}/{split}: {len(items)} -> {out_path}", flush=True)
    else:
        # by_split: one file per split, shared across all models (no duplicated storage)
        for split in splits:
            out_path = _packed_split_file_path(out_dir, split)
            if out_path.exists() and (not args.overwrite):
                print(f"[pack] skip existing: {out_path}", flush=True)
                continue

            subset = [e for e in index if e["split"] == split]
            if not subset:
                print(f"[pack] no rows for split={split}, skip", flush=True)
                continue

            items: List[Dict[str, Any]] = []
            for ent in tqdm(subset, desc=f"Pack split/{split}"):
                src = reader.get_item(ent["shard"], int(ent["offset"]))
                packed_item: Dict[str, Any] = {
                    "run_id": int(src["run_id"]),
                    "targets": {
                        "test_acc": float(src["targets"]["test_acc"]),
                        "test_loss": float(src["targets"]["test_loss"]),
                        "test_ppl": float(src["targets"]["test_ppl"]),
                    },
                }
                for rk in required_rep_keys:
                    if rk in src:
                        packed_item[rk] = src[rk]
                items.append(packed_item)

            payload = {
                "version": 2,
                "packed_mode": "by_split",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "split": split,
                "num_items": len(items),
                "model_types": model_types,
                "required_rep_keys": required_rep_keys,
                "source_manifest": str(Path(args.manifest).resolve()),
                "items": items,
            }
            torch.save(payload, out_path)
            summary_rows.append(
                {
                    "packed_mode": "by_split",
                    "model_type": "shared",
                    "split": split,
                    "num_items": len(items),
                    "path": str(out_path),
                }
            )
            print(f"[pack] saved shared split/{split}: {len(items)} -> {out_path}", flush=True)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "packed_summary.csv", index=False)
        print(f"[pack] summary: {out_dir / 'packed_summary.csv'}", flush=True)
def collate_w2t(batch):
    feats, ys, run_ids = zip(*batch)
    B = len(feats)
    L = len(feats[0]["features"])
    out_features = []
    masks = []
    for p in range(L):
        u_list = [f["features"][p][0].float() for f in feats]  # [r, d_out]
        v_list = [f["features"][p][1].float() for f in feats]  # [r, d_in]
        s_list = [f["features"][p][2].float() for f in feats]  # [r, 1]

        d_out = int(u_list[0].shape[1])
        d_in = int(v_list[0].shape[1])
        max_r = max(int(u.shape[0]) for u in u_list)

        U = torch.zeros(B, max_r, d_out, dtype=torch.float32)
        V = torch.zeros(B, max_r, d_in, dtype=torch.float32)
        S = torch.zeros(B, max_r, 1, dtype=torch.float32)
        mask = torch.ones(B, max_r, dtype=torch.bool)

        for i, (u, v, s) in enumerate(zip(u_list, v_list, s_list)):
            r = int(u.shape[0])
            U[i, :r, :] = u
            V[i, :r, :] = v
            S[i, :r, :] = s
            mask[i, :r] = False

        out_features.append((U, V, S))
        masks.append(mask)

    data = {"features": out_features, "src_key_padding_masks": masks}
    y = torch.tensor(ys, dtype=torch.float32)
    return data, y, list(run_ids)


def collate_glnet(batch):
    feats, ys, run_ids = zip(*batch)
    B = len(feats)
    L = len(feats[0]["uvs"])
    out_uvs = []
    for p in range(L):
        u_list = [f["uvs"][p][0].float() for f in feats]  # [d_out, r]
        v_list = [f["uvs"][p][1].float() for f in feats]  # [d_in, r]
        d_out = int(u_list[0].shape[0])
        d_in = int(v_list[0].shape[0])
        max_r = max(int(u.shape[1]) for u in u_list)

        U = torch.zeros(B, d_out, max_r, dtype=torch.float32)
        V = torch.zeros(B, d_in, max_r, dtype=torch.float32)
        for i, (u, v) in enumerate(zip(u_list, v_list)):
            r = int(u.shape[1])
            U[i, :, :r] = u
            V[i, :, :r] = v
        out_uvs.append((U, V))
    y = torch.tensor(ys, dtype=torch.float32)
    return out_uvs, y, list(run_ids)


def collate_flat(batch, layer_dims: Optional[Sequence[int]] = None):
    feats, ys, run_ids = zip(*batch)
    if layer_dims is None:
        util_batch = [
            (f["layers"], torch.as_tensor(y, dtype=torch.float32))
            for f, y in zip(feats, ys)
        ]
        out_layers, y = baseline_collate_layerwise_flat(util_batch)
        out_layers = [x.float() for x in out_layers]
        return out_layers, y, list(run_ids)

    if not feats:
        raise RuntimeError("Empty batch for collate_flat.")
    fixed_dims = [int(d) for d in layer_dims]
    num_layers = len(feats[0]["layers"])
    if len(fixed_dims) != num_layers:
        raise RuntimeError(
            f"layer_dims length mismatch: got {len(fixed_dims)}, expected {num_layers}."
        )

    batch_size = len(feats)
    out_layers = []
    for p in range(num_layers):
        cap = fixed_dims[p]
        X = torch.zeros(batch_size, cap, dtype=torch.float32)
        for i, f in enumerate(feats):
            x = f["layers"][p].float().reshape(-1)
            d = int(x.numel())
            if d > cap:
                raise RuntimeError(
                    f"Layer {p} dim {d} exceeds configured dim {cap}. "
                    "Recompute fixed layer dims from all splits."
                )
            X[i, :d] = x
        out_layers.append(X)

    y = torch.tensor(ys, dtype=torch.float32)
    return out_layers, y, list(run_ids)


def collate_token(batch):
    feats, ys, run_ids = zip(*batch)
    util_batch = [
        (f["layers"], torch.as_tensor(y, dtype=torch.float32))
        for f, y in zip(feats, ys)
    ]
    token_dict, y = baseline_collate_layerwise_tokenized(util_batch)
    # Keep dtype consistent with model weights to avoid Half/Float matmul mismatch.
    token_dict["tokens"] = token_dict["tokens"].float()
    return token_dict, y, list(run_ids)


def get_collate_fn(model_type: str, flat_layer_dims: Optional[Sequence[int]] = None):
    if model_type == "w2t":
        return collate_w2t
    if model_type == "glnet":
        return collate_glnet
    if model_type in ("mlp", "cnn"):
        if flat_layer_dims is None:
            return collate_flat
        return partial(collate_flat, layer_dims=list(flat_layer_dims))
    if model_type == "vit":
        return collate_token
    raise ValueError(model_type)


def infer_flat_layer_dims(
    datasets: Sequence[torch.utils.data.Dataset],
    desc: str = "Infer fixed flat dims",
) -> List[int]:
    max_dims: Optional[List[int]] = None
    for ds in datasets:
        for i in tqdm(range(len(ds)), desc=desc):
            feat, _, _ = ds[i]
            layers = feat["layers"]
            if max_dims is None:
                max_dims = [0 for _ in layers]
            if len(layers) != len(max_dims):
                raise RuntimeError(
                    f"Inconsistent number of layers: got {len(layers)} vs expected {len(max_dims)}."
                )
            for p, x in enumerate(layers):
                d = int(x.numel())
                if d > max_dims[p]:
                    max_dims[p] = d
    if max_dims is None:
        raise RuntimeError("Failed to infer flat layer dims from empty datasets.")
    return max_dims


class W2TRegressionWrapper(nn.Module):
    def __init__(self, base: FullTransformer):
        super().__init__()
        self.base = base

    def forward(self, data):
        masks = data.get("src_key_padding_masks")
        out = self.base({"features": data["features"]}, src_key_padding_masks=masks)
        return out.squeeze(-1)


class SqueezeRegressionWrapper(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x):
        out = self.base(x)
        return out.squeeze(-1)


def build_model(
    model_type: str,
    sample_feature: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    flat_layer_dims: Optional[Sequence[int]] = None,
) -> nn.Module:
    if model_type == "w2t":
        feats = sample_feature["features"]
        input_dims = [(int(u.shape[1]), int(v.shape[1])) for (u, v, _) in feats]
        layer_ids = [int(x) for x in sample_feature["layer_ids"]]
        module_ids = [int(x) for x in sample_feature["module_ids"]]
        num_layers = int(max(layer_ids) + 1) if layer_ids else 0
        num_modules = int(max(module_ids) + 1) if module_ids else 0

        base = FullTransformer(
            input_dims=input_dims,
            layer_ids=layer_ids,
            module_ids=module_ids,
            num_layers=num_layers,
            num_modules=num_modules,
            hidden_dim=args.hidden_dim,
            out_dim=1,
            num_rank_layers=args.num_rank_layers,
            num_layer_layers=args.num_layer_layers,
            nhead=args.nhead,
            dropout=args.dropout,
            mlp_dim=args.mlp_dim,
            sign_aug_prob=0.0,
            rank_perm_prob=0.0,
        )
        return W2TRegressionWrapper(base).to(device)

    if model_type == "glnet":
        uvs = sample_feature["uvs"]
        ns = [int(u.shape[0]) for (u, _) in uvs]
        ms = [int(v.shape[0]) for (_, v) in uvs]
        base = GLInvariantMLP(
            ns=ns,
            ms=ms,
            n_input_layers=len(uvs),
            out_dim=1,
            hidden_dim_equiv=args.hidden_dim,
            n_layers=args.glnet_layers,
            hidden_dim_inv=args.mlp_dim,
            clip=False,
        )
        return SqueezeRegressionWrapper(base).to(device)

    if model_type == "mlp":
        layers = sample_feature["layers"]
        layer_dims = [int(x.numel()) for x in layers] if flat_layer_dims is None else [int(x) for x in flat_layer_dims]
        base = FlattenMLP_Layerwise(
            layer_dims=layer_dims,
            out_dim=1,
            hidden_dim=args.hidden_dim,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout,
        )
        return SqueezeRegressionWrapper(base).to(device)

    if model_type == "cnn":
        layers = sample_feature["layers"]
        layer_dims = [int(x.numel()) for x in layers] if flat_layer_dims is None else [int(x) for x in flat_layer_dims]
        base = CNN1D_Layerwise(
            layer_dims=layer_dims,
            out_dim=1,
            hidden_dim=args.hidden_dim,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout,
        )
        return SqueezeRegressionWrapper(base).to(device)

    if model_type == "vit":
        layers = sample_feature["layers"]
        sample_token_size = int(sample_feature.get("token_size", layers[0].shape[1] if layers else args.token_size))
        base = TokenViT_Layerwise(
            num_layers=len(layers),
            out_dim=1,
            token_size=sample_token_size,
            embed_dim=args.hidden_dim,
            depth=args.vit_depth,
            nhead=args.nhead,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout,
        )
        return SqueezeRegressionWrapper(base).to(device)

    raise ValueError(f"Unknown model_type={model_type}")


class TargetTransform:
    def __init__(self, mode: str):
        if mode not in {"none", "zscore", "log1p_zscore"}:
            raise ValueError(f"Unknown target transform mode: {mode}")
        self.mode = mode
        self.mean = 0.0
        self.std = 1.0

    def fit(self, values: Sequence[float]) -> None:
        x = np.asarray(values, dtype=np.float64)
        if self.mode == "none":
            self.mean = 0.0
            self.std = 1.0
            return
        if self.mode == "zscore":
            y = x
        else:
            y = np.log1p(np.clip(x, 0.0, None))
        self.mean = float(np.mean(y))
        self.std = float(np.std(y) + EPS)

    def transform_tensor(self, y: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return y
        if self.mode == "zscore":
            base = y
        else:
            base = torch.log1p(torch.clamp(y, min=0.0))
        return (base - self.mean) / self.std

    def inverse_tensor(self, y_hat: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return y_hat
        base = y_hat * self.std + self.mean
        if self.mode == "zscore":
            return base
        return torch.expm1(base).clamp_min(0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "mean": self.mean, "std": self.std}


def _average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.zeros_like(values, dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    den = math.sqrt(float((xc * xc).sum()) * float((yc * yc).sum()))
    if den < EPS:
        return float("nan")
    return float((xc * yc).sum() / den)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    rx = _average_rank(x)
    ry = _average_rank(y)
    return pearson_corr(rx, ry)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
        }
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return {
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson_corr(y_true, y_pred),
        "spearman": spearman_corr(y_true, y_pred),
    }


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    target_tf: TargetTransform,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    ys: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    run_ids: List[int] = []

    for data, y, rid in loader:
        data = to_device(data, device)
        y = y.to(device)
        pred_trans = model(data)
        pred = target_tf.inverse_tensor(pred_trans)
        ys.append(y.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
        run_ids.extend(rid)

    y_true = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.float32)
    y_pred = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)
    metrics = regression_metrics(y_true, y_pred) if y_true.size else {
        "mae": float("nan"),
        "rmse": float("nan"),
        "pearson": float("nan"),
        "spearman": float("nan"),
    }
    return {
        "run_ids": run_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "metrics": metrics,
    }


def _resolve_target_transform(mode: str, target_col: str) -> str:
    if mode != "auto":
        return mode
    if target_col == "test_ppl":
        return "log1p_zscore"
    return "zscore"


def train_one_model(
    model_type: str,
    reader: Optional[ShardCacheReader],
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> Dict[str, Any]:
    ensure_dir(out_dir)

    use_packed = bool(args.packed_cache_dir)
    if use_packed:
        pack_dir = Path(args.packed_cache_dir)
        split_train = _packed_split_file_path(pack_dir, "train")
        split_valid = _packed_split_file_path(pack_dir, "valid")
        split_test = _packed_split_file_path(pack_dir, "test")
        if split_train.exists() and split_valid.exists() and split_test.exists():
            train_set = PackedLoRACacheDataset(split_train, target_col=args.target_col, model_type=model_type)
            valid_set = PackedLoRACacheDataset(split_valid, target_col=args.target_col, model_type=model_type)
            test_set = PackedLoRACacheDataset(split_test, target_col=args.target_col, model_type=model_type)
        else:
            train_set = PackedLoRACacheDataset(
                _packed_file_path(pack_dir, model_type, "train"),
                target_col=args.target_col,
                model_type=model_type,
            )
            valid_set = PackedLoRACacheDataset(
                _packed_file_path(pack_dir, model_type, "valid"),
                target_col=args.target_col,
                model_type=model_type,
            )
            test_set = PackedLoRACacheDataset(
                _packed_file_path(pack_dir, model_type, "test"),
                target_col=args.target_col,
                model_type=model_type,
            )
        print(f"[train][{model_type}] use packed cache from {pack_dir}", flush=True)
    else:
        if reader is None:
            raise ValueError("reader is required when --packed-cache-dir is not provided.")
        train_set = LoRACacheDataset(reader=reader, split="train", model_type=model_type, target_col=args.target_col)
        valid_set = LoRACacheDataset(reader=reader, split="valid", model_type=model_type, target_col=args.target_col)
        test_set = LoRACacheDataset(reader=reader, split="test", model_type=model_type, target_col=args.target_col)

    flat_layer_dims = None
    if model_type in ("mlp", "cnn"):
        print(f"[train][{model_type}] inferring fixed flat layer dims from train/valid/test ...", flush=True)
        flat_layer_dims = infer_flat_layer_dims(
            [train_set, valid_set, test_set],
            desc=f"{model_type}-dims",
        )
        print(
            f"[train][{model_type}] fixed layer dims inferred. "
            f"num_layers={len(flat_layer_dims)} max_dim={max(flat_layer_dims)}",
            flush=True,
        )

    collate_fn = get_collate_fn(model_type, flat_layer_dims=flat_layer_dims)
    pin_mem = device.type == "cuda"

    train_sampler = None
    if isinstance(train_set, LoRACacheDataset):
        train_sampler = ShardAwareSampler(train_set, shuffle=args.train_shuffle, seed=args.seed)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(args.train_shuffle and train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        collate_fn=collate_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        collate_fn=collate_fn,
    )

    sample_feature, _, _ = train_set[0]
    model = build_model(model_type, sample_feature, args, device, flat_layer_dims=flat_layer_dims)

    tf_mode = _resolve_target_transform(args.target_transform, args.target_col)
    target_tf = TargetTransform(tf_mode)
    target_tf.fit(train_set.target_values())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_valid_mae = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    history: List[Dict[str, Any]] = []
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        seen = 0
        t_epoch = time.time()
        for step, (data, y, _) in enumerate(train_loader, start=1):
            data = to_device(data, device)
            y = y.to(device)
            y_t = target_tf.transform_tensor(y)

            optimizer.zero_grad(set_to_none=True)
            pred_t = model(data)
            loss = F.mse_loss(pred_t, y_t)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            bs = int(y.shape[0])
            running_loss += float(loss.item()) * bs
            seen += bs
            if args.log_interval > 0 and (step % args.log_interval == 0):
                avg_mse_t = running_loss / max(1, seen)
                print(
                    f"[train][{model_type}] epoch={epoch:03d} step={step:05d}/{len(train_loader):05d} "
                    f"mse_t={avg_mse_t:.6f} elapsed={time.time()-t_epoch:.1f}s",
                    flush=True,
                )

        scheduler.step()
        train_loss = running_loss / max(1, seen)
        valid_eval = evaluate_loader(model, valid_loader, target_tf, device)
        valid_mae = float(valid_eval["metrics"]["mae"])
        valid_rmse = float(valid_eval["metrics"]["rmse"])

        history.append(
            {
                "epoch": epoch,
                "train_mse_transformed": train_loss,
                "valid_mae": valid_mae,
                "valid_rmse": valid_rmse,
                "valid_pearson": float(valid_eval["metrics"]["pearson"]),
                "valid_spearman": float(valid_eval["metrics"]["spearman"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        improved = valid_mae < (best_valid_mae - 1e-12)
        if improved:
            best_valid_mae = valid_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"[train][{model_type}] epoch={epoch:03d} "
            f"train_mse_t={train_loss:.6f} valid_mae={valid_mae:.6f} valid_rmse={valid_rmse:.6f}",
            flush=True,
        )

        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"[train][{model_type}] early stop at epoch {epoch} (patience={args.patience})", flush=True)
            break

    model.load_state_dict(best_state)
    valid_eval = evaluate_loader(model, valid_loader, target_tf, device)
    test_eval = evaluate_loader(model, test_loader, target_tf, device)

    state_cpu = _state_dict_cpu_contiguous(model.state_dict())
    ckpt_payload = {
        "model_type": model_type,
        "state_dict": state_cpu,
        "target_col": args.target_col,
        "target_transform": target_tf.to_dict(),
        "flat_layer_dims": (None if flat_layer_dims is None else [int(x) for x in flat_layer_dims]),
        "args": sanitize_args_for_checkpoint(args),
    }
    torch.save(ckpt_payload, out_dir / "best_model.pth")

    # Portable fallback bundle: robust to .pth corruption during transfer.
    st_save_torch(state_cpu, str(out_dir / "best_state.safetensors"))
    dump_json(
        out_dir / "best_model_meta.json",
        {
            "model_type": model_type,
            "target_col": args.target_col,
            "target_transform": target_tf.to_dict(),
            "flat_layer_dims": (None if flat_layer_dims is None else [int(x) for x in flat_layer_dims]),
            "args": sanitize_args_for_checkpoint(args),
        },
    )

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    valid_pred_df = pd.DataFrame(
        {
            "run_id": valid_eval["run_ids"],
            "y_true": valid_eval["y_true"],
            "y_pred": valid_eval["y_pred"],
        }
    )
    valid_pred_df["abs_error"] = np.abs(valid_pred_df["y_pred"] - valid_pred_df["y_true"])
    valid_pred_df.to_csv(out_dir / "valid_predictions.csv", index=False)

    test_pred_df = pd.DataFrame(
        {
            "run_id": test_eval["run_ids"],
            "y_true": test_eval["y_true"],
            "y_pred": test_eval["y_pred"],
        }
    )
    test_pred_df["abs_error"] = np.abs(test_pred_df["y_pred"] - test_pred_df["y_true"])
    test_pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    metrics_payload = {
        "model_type": model_type,
        "target_col": args.target_col,
        "target_transform": target_tf.to_dict(),
        "train_size": len(train_set),
        "valid_size": len(valid_set),
        "test_size": len(test_set),
        "best_epoch": int(best_epoch),
        "best_valid_mae": float(best_valid_mae),
        "valid_metrics": {k: float(v) for k, v in valid_eval["metrics"].items()},
        "test_metrics": {k: float(v) for k, v in test_eval["metrics"].items()},
    }
    dump_json(out_dir / "metrics.json", metrics_payload)

    return {
        "model_type": model_type,
        "target_col": args.target_col,
        "best_epoch": int(best_epoch),
        "best_valid_mae": float(best_valid_mae),
        "valid_mae": float(valid_eval["metrics"]["mae"]),
        "valid_rmse": float(valid_eval["metrics"]["rmse"]),
        "valid_pearson": float(valid_eval["metrics"]["pearson"]),
        "valid_spearman": float(valid_eval["metrics"]["spearman"]),
        "test_mae": float(test_eval["metrics"]["mae"]),
        "test_rmse": float(test_eval["metrics"]["rmse"]),
        "test_pearson": float(test_eval["metrics"]["pearson"]),
        "test_spearman": float(test_eval["metrics"]["spearman"]),
    }


def cmd_train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(args.seed)

    reader = None
    if not args.packed_cache_dir:
        reader = ShardCacheReader(args.manifest, max_cached_shards=args.max_shards_in_mem)
    out_root = ensure_dir(args.output_dir)

    if args.model_type == "all":
        model_types = ["w2t", "glnet", "mlp", "cnn", "vit"]
    else:
        model_types = [args.model_type]

    summary_rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []

    for model_type in model_types:
        print(f"[train] start model={model_type} target={args.target_col} device={device}", flush=True)
        model_out = ensure_dir(out_root / model_type)
        t0 = time.time()
        try:
            row = train_one_model(model_type, reader, args, device, model_out)
            row["elapsed_sec"] = float(time.time() - t0)
            summary_rows.append(row)
            print(
                f"[train] done model={model_type} "
                f"test_mae={row['test_mae']:.6f} test_spearman={row['test_spearman']:.6f}",
                flush=True,
            )
        except Exception as exc:
            fail = {
                "model_type": model_type,
                "target_col": args.target_col,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failed_rows.append(fail)
            print(f"[train] failed model={model_type} error={fail['error']}", flush=True)
            if not args.continue_on_error:
                raise

    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        upsert_csv(
            out_root / "benchmark_summary.csv",
            df_sum,
            key_cols=["model_type", "target_col"],
            sort_cols=["target_col", "test_mae"],
        )
        print(f"[train] summary saved: {out_root / 'benchmark_summary.csv'}", flush=True)
    if failed_rows:
        upsert_csv(
            out_root / "benchmark_failures.csv",
            pd.DataFrame(failed_rows),
            key_cols=["model_type", "target_col", "error"],
            sort_cols=["target_col", "model_type"],
        )
        print(f"[train] failures saved: {out_root / 'benchmark_failures.csv'}", flush=True)


def _load_packed_dataset_for_split(
    pack_dir: Path,
    model_type: str,
    split: str,
    target_col: str,
) -> PackedLoRACacheDataset:
    shared_path = _packed_split_file_path(pack_dir, split)
    if shared_path.exists():
        return PackedLoRACacheDataset(shared_path, target_col=target_col, model_type=model_type)
    model_path = _packed_file_path(pack_dir, model_type, split)
    if model_path.exists():
        return PackedLoRACacheDataset(model_path, target_col=target_col, model_type=model_type)
    raise FileNotFoundError(
        f"Packed split not found for split={split}, model_type={model_type}. "
        f"Tried: {shared_path} and {model_path}"
    )


def _target_transform_from_checkpoint(ckpt: Dict[str, Any]) -> TargetTransform:
    cfg = ckpt.get("target_transform", {})
    mode = str(cfg.get("mode", "none"))
    if mode not in {"none", "zscore", "log1p_zscore"}:
        mode = "none"
    tf = TargetTransform(mode)
    tf.mean = float(cfg.get("mean", 0.0))
    std = float(cfg.get("std", 1.0))
    if (not np.isfinite(std)) or abs(std) < EPS:
        std = 1.0
    tf.std = std
    return tf


def _model_args_from_checkpoint(ckpt_args: Any, runtime_args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace()
    if isinstance(ckpt_args, dict):
        for k, v in ckpt_args.items():
            setattr(ns, str(k), v)

    defaults = {
        "hidden_dim": int(getattr(runtime_args, "hidden_dim", 128)),
        "mlp_dim": int(getattr(runtime_args, "mlp_dim", 128)),
        "dropout": float(getattr(runtime_args, "dropout", 0.0)),
        "nhead": int(getattr(runtime_args, "nhead", 4)),
        "num_rank_layers": int(getattr(runtime_args, "num_rank_layers", 1)),
        "num_layer_layers": int(getattr(runtime_args, "num_layer_layers", 2)),
        "vit_depth": int(getattr(runtime_args, "vit_depth", 4)),
        "token_size": int(getattr(runtime_args, "token_size", 2048)),
        "glnet_layers": int(getattr(runtime_args, "glnet_layers", 1)),
    }
    for k, v in defaults.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns


def cmd_predict(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(args.seed)

    pack_dir = Path(args.packed_cache_dir)
    if not pack_dir.exists():
        raise FileNotFoundError(pack_dir)
    trained_root = Path(args.trained_root)
    if not trained_root.exists():
        raise FileNotFoundError(trained_root)
    out_root = ensure_dir(args.output_dir)

    if args.model_type == "all":
        model_types = ["w2t", "glnet", "mlp", "cnn", "vit"]
    else:
        model_types = [args.model_type]
    splits = _resolve_splits_arg(args.splits)

    summary_rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []

    for model_type in model_types:
        print(f"[predict] start model={model_type} device={device}", flush=True)
        t0 = time.time()
        try:
            ckpt_path = trained_root / model_type / args.checkpoint_name
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            ckpt = load_checkpoint_with_fallback(ckpt_path)

            ckpt_target_col = ckpt.get("target_col")
            target_col = args.target_col or ckpt_target_col
            if target_col not in TARGET_COLUMNS:
                raise ValueError(
                    f"target_col invalid or missing. got={target_col}, "
                    f"checkpoint_target_col={ckpt_target_col}, cli_target_col={args.target_col}"
                )
            if (
                ckpt_target_col in TARGET_COLUMNS
                and args.target_col is not None
                and args.target_col != ckpt_target_col
                and (not args.allow_target_col_mismatch)
            ):
                raise ValueError(
                    f"target_col mismatch: checkpoint={ckpt_target_col} cli={args.target_col}. "
                    f"Use --allow-target-col-mismatch to force."
                )

            datasets: Dict[str, PackedLoRACacheDataset] = {}
            for split in splits:
                datasets[split] = _load_packed_dataset_for_split(pack_dir, model_type, split, target_col)

            sample_split = splits[0]
            sample_feature, _, _ = datasets[sample_split][0]

            flat_layer_dims = None
            if model_type == "mlp":
                ckpt_dims = ckpt.get("flat_layer_dims")
                if ckpt_dims is not None:
                    flat_layer_dims = [int(x) for x in ckpt_dims]
                else:
                    inferred_dims = _infer_mlp_flat_dims_from_state_dict(ckpt["state_dict"])
                    if inferred_dims is not None:
                        flat_layer_dims = inferred_dims
                    else:
                        dim_sets: List[torch.utils.data.Dataset] = []
                        for split in ["train", "valid", "test"]:
                            try:
                                dim_sets.append(_load_packed_dataset_for_split(pack_dir, model_type, split, target_col))
                            except Exception:
                                pass
                        if not dim_sets:
                            dim_sets = [datasets[s] for s in splits]
                        print(
                            f"[predict][{model_type}] inferring fixed flat layer dims from packed splits ...",
                            flush=True,
                        )
                        flat_layer_dims = infer_flat_layer_dims(dim_sets, desc=f"predict-{model_type}-dims")

            model_args = _model_args_from_checkpoint(ckpt.get("args"), args)
            model = build_model(
                model_type,
                sample_feature,
                model_args,
                device,
                flat_layer_dims=flat_layer_dims,
            )
            state_dict = adapt_state_dict_for_model(model_type, ckpt["state_dict"], model)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"State dict mismatch after compatibility mapping. "
                    f"missing={missing[:10]} unexpected={unexpected[:10]}"
                )

            target_tf = _target_transform_from_checkpoint(ckpt)
            collate_fn = get_collate_fn(model_type, flat_layer_dims=flat_layer_dims)
            pin_mem = device.type == "cuda"

            model_out = ensure_dir(out_root / model_type)
            split_metrics: Dict[str, Any] = {}
            for split in splits:
                loader = torch.utils.data.DataLoader(
                    datasets[split],
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=pin_mem,
                    collate_fn=collate_fn,
                )
                ev = evaluate_loader(model, loader, target_tf, device)
                pred_df = pd.DataFrame(
                    {
                        "run_id": ev["run_ids"],
                        "y_true": ev["y_true"],
                        "y_pred": ev["y_pred"],
                    }
                )
                pred_df["abs_error"] = np.abs(pred_df["y_pred"] - pred_df["y_true"])
                pred_df.to_csv(model_out / f"{split}_predictions.csv", index=False)

                metrics = {k: float(v) for k, v in ev["metrics"].items()}
                split_metrics[split] = metrics
                summary_rows.append(
                    {
                        "model_type": model_type,
                        "split": split,
                        "target_col": target_col,
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "pearson": metrics["pearson"],
                        "spearman": metrics["spearman"],
                        "n_samples": int(len(ev["run_ids"])),
                        "elapsed_sec": float(time.time() - t0),
                    }
                )
                print(
                    f"[predict] done model={model_type} split={split} "
                    f"mae={metrics['mae']:.6f} spearman={metrics['spearman']:.6f}",
                    flush=True,
                )

            dump_json(
                model_out / "predict_metrics.json",
                {
                    "model_type": model_type,
                    "target_col": target_col,
                    "checkpoint_path": str(ckpt_path.resolve()),
                    "checkpoint_target_col": ckpt_target_col,
                    "target_transform": target_tf.to_dict(),
                    "splits": split_metrics,
                },
            )

        except Exception as exc:
            fail = {
                "model_type": model_type,
                "target_col": (args.target_col if args.target_col is not None else ""),
                "error": f"{type(exc).__name__}: {exc}",
            }
            failed_rows.append(fail)
            print(f"[predict] failed model={model_type} error={fail['error']}", flush=True)
            if not args.continue_on_error:
                raise

    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        upsert_csv(
            out_root / "predict_summary.csv",
            df_sum,
            key_cols=["model_type", "split", "target_col"],
            sort_cols=["split", "model_type", "target_col"],
        )
        print(f"[predict] summary saved: {out_root / 'predict_summary.csv'}", flush=True)
    if failed_rows:
        upsert_csv(
            out_root / "predict_failures.csv",
            pd.DataFrame(failed_rows),
            key_cols=["model_type", "target_col", "error"],
            sort_cols=["target_col", "model_type"],
        )
        print(f"[predict] failures saved: {out_root / 'predict_failures.csv'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("LoRA ARC performance prediction pipeline.")
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("prepare", help="Merge regression CSVs and create train/valid/test split.")
    pp.add_argument("--input-dir", type=str, default="regression")
    pp.add_argument("--glob", type=str, default="results_arc_easy_llama_task*.csv")
    pp.add_argument("--output-dir", type=str, default="regression/prepared_arc")
    pp.add_argument("--train-ratio", type=float, default=0.8)
    pp.add_argument("--valid-ratio", type=float, default=0.1)
    pp.add_argument("--test-ratio", type=float, default=0.1)
    pp.add_argument("--stratify-target", type=str, default="test_acc")
    pp.add_argument("--stratify-bins", type=int, default=10)
    pp.add_argument("--seed", type=int, default=42)
    pp.set_defaults(func=cmd_prepare)

    pc = sub.add_parser("cache", help="Build sharded high-speed cache from LoRA safetensors.")
    pc.add_argument("--metadata-csv", type=str, default="regression/prepared_arc/all_metadata.csv")
    pc.add_argument("--split-dir", type=str, default="regression/prepared_arc")
    pc.add_argument("--output-dir", type=str, default="regression/cache_arc")
    pc.add_argument("--representations", type=str, default="w2t,glnet,mlp,cnn,vit")
    pc.add_argument("--token-size", type=int, default=2048)
    pc.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float16")
    pc.add_argument("--shard-size", type=int, default=128)
    pc.add_argument("--path-map", action="append", default=[], help="Path prefix map item: source_prefix=target_prefix")
    pc.add_argument("--path-map-json", type=str, default=None, help="JSON file containing path prefix mappings.")
    pc.add_argument("--max-samples", type=int, default=None)
    pc.add_argument("--strict-schema", dest="strict_schema", action="store_true")
    pc.add_argument("--no-strict-schema", dest="strict_schema", action="store_false")
    pc.set_defaults(strict_schema=True)
    pc.add_argument("--skip-missing", dest="skip_missing", action="store_true")
    pc.add_argument("--no-skip-missing", dest="skip_missing", action="store_false")
    pc.set_defaults(skip_missing=True)
    pc.set_defaults(func=cmd_cache)

    pk = sub.add_parser("pack", help="Repack shard cache into split/model single files for fast loading.")
    pk.add_argument("--manifest", type=str, default="regression/cache_arc/manifest.json")
    pk.add_argument("--output-dir", type=str, default="regression/cache_arc_packed")
    pk.add_argument("--pack-mode", type=str, choices=["by_split", "by_model"], default="by_split")
    pk.add_argument("--model-types", type=str, default="all", help="all or comma list: w2t,glnet,mlp,cnn,vit")
    pk.add_argument("--splits", type=str, default="train,valid,test")
    pk.add_argument("--max-shards-in-mem", type=int, default=8)
    pk.add_argument("--overwrite", action="store_true", default=False)
    pk.set_defaults(func=cmd_pack)

    pt = sub.add_parser("train", help="Train performance prediction model(s) on cached data.")
    pt.add_argument("--manifest", type=str, default="regression/cache_arc/manifest.json")
    pt.add_argument("--packed-cache-dir", type=str, default=None, help="Use packed cache dir from `pack` command.")
    pt.add_argument("--output-dir", type=str, default="regression/results_perf_pred")
    pt.add_argument("--model-type", type=str, choices=["w2t", "glnet", "mlp", "cnn", "vit", "all"], default="all")
    pt.add_argument("--target-col", type=str, choices=TARGET_COLUMNS, default="test_acc")
    pt.add_argument(
        "--target-transform",
        type=str,
        choices=["auto", "none", "zscore", "log1p_zscore"],
        default="auto",
    )
    pt.add_argument("--epochs", type=int, default=50)
    pt.add_argument("--batch-size", type=int, default=32)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--weight-decay", type=float, default=1e-4)
    pt.add_argument("--patience", type=int, default=10)
    pt.add_argument("--grad-clip", type=float, default=1.0)
    pt.add_argument("--hidden-dim", type=int, default=128)
    pt.add_argument("--mlp-dim", type=int, default=128)
    pt.add_argument("--dropout", type=float, default=0.0)
    pt.add_argument("--nhead", type=int, default=4)
    pt.add_argument("--num-rank-layers", type=int, default=1)
    pt.add_argument("--num-layer-layers", type=int, default=2)
    pt.add_argument("--vit-depth", type=int, default=4)
    pt.add_argument("--token-size", type=int, default=2048)
    pt.add_argument("--glnet-layers", type=int, default=1)
    pt.add_argument("--num-workers", type=int, default=0)
    pt.add_argument("--max-shards-in-mem", type=int, default=8)
    pt.add_argument("--log-interval", type=int, default=50, help="Print training progress every N steps.")
    pt.add_argument("--train-shuffle", dest="train_shuffle", action="store_true")
    pt.add_argument("--no-train-shuffle", dest="train_shuffle", action="store_false")
    pt.set_defaults(train_shuffle=True)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--device", type=str, default="auto")
    pt.add_argument("--continue-on-error", action="store_true", default=False)
    pt.set_defaults(func=cmd_train)

    ppd = sub.add_parser(
        "predict",
        help="Load pre-trained model checkpoints and run direct prediction on packed OOD cache.",
    )
    ppd.add_argument("--packed-cache-dir", type=str, required=True)
    ppd.add_argument(
        "--trained-root",
        type=str,
        required=True,
        help="Directory containing trained model subfolders, e.g. .../results_perf_pred_acc",
    )
    ppd.add_argument("--output-dir", type=str, default="regression/results_ood_pred")
    ppd.add_argument(
        "--model-type",
        type=str,
        choices=["w2t", "glnet", "mlp", "cnn", "vit", "all"],
        default="all",
    )
    ppd.add_argument("--checkpoint-name", type=str, default="best_model.pth")
    ppd.add_argument("--splits", type=str, default="test", help="Comma list, e.g. test or train,valid,test")
    ppd.add_argument("--target-col", type=str, choices=TARGET_COLUMNS, default=None)
    ppd.add_argument("--allow-target-col-mismatch", action="store_true", default=False)
    ppd.add_argument("--batch-size", type=int, default=128)
    ppd.add_argument("--num-workers", type=int, default=0)
    # Fallback architecture args when old checkpoints miss these fields.
    ppd.add_argument("--hidden-dim", type=int, default=128)
    ppd.add_argument("--mlp-dim", type=int, default=128)
    ppd.add_argument("--dropout", type=float, default=0.0)
    ppd.add_argument("--nhead", type=int, default=4)
    ppd.add_argument("--num-rank-layers", type=int, default=1)
    ppd.add_argument("--num-layer-layers", type=int, default=2)
    ppd.add_argument("--vit-depth", type=int, default=4)
    ppd.add_argument("--token-size", type=int, default=2048)
    ppd.add_argument("--glnet-layers", type=int, default=1)
    ppd.add_argument("--seed", type=int, default=42)
    ppd.add_argument("--device", type=str, default="auto")
    ppd.add_argument("--continue-on-error", action="store_true", default=False)
    ppd.set_defaults(func=cmd_predict)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

