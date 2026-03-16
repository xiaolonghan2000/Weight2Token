#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from perf_prediction_pipeline import (
    REQUIRED_METADATA_COLUMNS,
    TARGET_COLUMNS,
    LoRACacheDataset,
    ShardCacheReader,
    _align_entries_to_schema,
    _extract_lora_entries,
    _infer_mlp_flat_dims_from_state_dict,
    _init_schema,
    _model_args_from_checkpoint,
    _load_split_map,
    adapt_state_dict_for_model,
    build_model,
    dump_json,
    ensure_dir,
    get_collate_fn,
    load_checkpoint_with_fallback,
    resolve_device,
    resolve_safetensors_path,
    set_seed,
    st_load_torch,
    to_device,
)


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_csv_list(text: str) -> List[int]:
    vals = []
    for item in parse_csv_list(text):
        vals.append(int(item))
    return vals


def expand_patterns(patterns: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for token in patterns:
        matched = sorted(glob.glob(token))
        if matched:
            for item in matched:
                p = Path(item)
                if p not in seen:
                    out.append(p)
                    seen.add(p)
            continue
        p = Path(token)
        if not p.exists():
            raise FileNotFoundError(f"File or pattern not found: {token}")
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _require_metadata_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def _sample_gallery_per_dataset(df: pd.DataFrame, per_dataset: int, seed: int) -> pd.DataFrame:
    if per_dataset <= 0:
        return df.copy()
    parts = []
    for dataset, group in df.groupby("dataset", sort=True):
        if len(group) <= per_dataset:
            parts.append(group.copy())
            continue
        parts.append(group.sample(n=per_dataset, random_state=seed).copy())
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def cmd_prepare(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.output_dir)
    gallery_paths = expand_patterns(parse_csv_list(args.gallery_csvs))
    query_paths = expand_patterns(parse_csv_list(args.query_csvs))
    if not gallery_paths:
        raise ValueError("No gallery CSVs found.")
    if not query_paths:
        raise ValueError("No query CSVs found.")

    gallery_frames = []
    for path in gallery_paths:
        df = pd.read_csv(path)
        _require_metadata_columns(df, path)
        df["source_csv"] = str(path)
        df["retrieval_role"] = "gallery"
        gallery_frames.append(df)
    gallery_df = pd.concat(gallery_frames, ignore_index=True)
    gallery_df = gallery_df.drop_duplicates(subset=["safetensors_path"]).reset_index(drop=True)
    gallery_df = _sample_gallery_per_dataset(gallery_df, int(args.gallery_per_dataset), int(args.seed))
    gallery_df["train_samples"] = pd.to_numeric(gallery_df.get("train_samples", np.nan), errors="coerce")

    query_frames = []
    for path in query_paths:
        df = pd.read_csv(path)
        _require_metadata_columns(df, path)
        df["source_csv"] = str(path)
        df["retrieval_role"] = "query"
        query_frames.append(df)
    query_df = pd.concat(query_frames, ignore_index=True)
    query_df = query_df.drop_duplicates(subset=["safetensors_path"]).reset_index(drop=True)
    query_df["train_samples"] = pd.to_numeric(query_df.get("train_samples", np.nan), errors="coerce")
    query_df["subset_seed"] = pd.to_numeric(query_df.get("subset_seed", np.nan), errors="coerce")

    overlap = sorted(set(gallery_df["safetensors_path"].tolist()) & set(query_df["safetensors_path"].tolist()))
    if overlap:
        preview = overlap[:5]
        raise ValueError(
            "Gallery/query overlap detected on safetensors_path. "
            f"Example paths: {preview}"
        )

    if int(args.max_queries_per_dataset_shot) > 0 and "train_samples" in query_df.columns:
        limited_parts = []
        for _, group in query_df.groupby(["dataset", "train_samples"], dropna=False, sort=True):
            take = min(len(group), int(args.max_queries_per_dataset_shot))
            limited_parts.append(group.sample(n=take, random_state=int(args.seed)).copy())
        query_df = pd.concat(limited_parts, ignore_index=True) if limited_parts else query_df.iloc[0:0].copy()

    all_df = pd.concat([gallery_df, query_df], ignore_index=True)
    if all_df.empty:
        raise ValueError("Merged retrieval metadata is empty.")

    original_run_id = all_df["run_id"].tolist()
    all_df["source_run_id"] = original_run_id
    all_df["run_id"] = np.arange(len(all_df), dtype=np.int64)

    split_map = pd.DataFrame(
        {
            "run_id": all_df.loc[all_df["retrieval_role"] == "gallery", "run_id"].astype(int).tolist(),
        }
    )
    test_map = pd.DataFrame(
        {
            "run_id": all_df.loc[all_df["retrieval_role"] == "query", "run_id"].astype(int).tolist(),
        }
    )

    split_dir = ensure_dir(out_dir / "splits")
    split_map.to_csv(split_dir / "train.csv", index=False)
    pd.DataFrame({"run_id": []}).to_csv(split_dir / "valid.csv", index=False)
    test_map.to_csv(split_dir / "test.csv", index=False)

    metadata_path = out_dir / "all_metadata.csv"
    all_df.to_csv(metadata_path, index=False)

    gallery_index_path = out_dir / "gallery_index.csv"
    query_index_path = out_dir / "query_index.csv"
    all_df[all_df["retrieval_role"] == "gallery"].to_csv(gallery_index_path, index=False)
    all_df[all_df["retrieval_role"] == "query"].to_csv(query_index_path, index=False)

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metadata_csv": str(metadata_path.resolve()),
        "split_dir": str(split_dir.resolve()),
        "num_gallery": int((all_df["retrieval_role"] == "gallery").sum()),
        "num_query": int((all_df["retrieval_role"] == "query").sum()),
        "gallery_per_dataset": int(args.gallery_per_dataset),
        "gallery_datasets": sorted(str(x) for x in gallery_df["dataset"].dropna().unique().tolist()),
        "query_datasets": sorted(str(x) for x in query_df["dataset"].dropna().unique().tolist()),
        "query_shots": sorted(int(x) for x in query_df["train_samples"].dropna().astype(int).unique().tolist()),
        "gallery_csvs": [str(p) for p in gallery_paths],
        "query_csvs": [str(p) for p in query_paths],
    }
    dump_json(out_dir / "prepare_summary.json", summary)
    print(f"[prepare] metadata: {metadata_path}")
    print(f"[prepare] split_dir: {split_dir}")
    print(f"[prepare] gallery={summary['num_gallery']} query={summary['num_query']}")


def _w2t_encode_base(model: torch.nn.Module, data: Any, src_key_padding_masks=None) -> torch.Tensor:
    base = model.base
    encode_fn = getattr(base, "encode", None)
    if callable(encode_fn):
        return encode_fn({"features": data["features"]}, src_key_padding_masks=src_key_padding_masks)

    # Legacy FullTransformer versions do not expose encode(); reconstruct the
    # pooled embedding from the shared submodules before the classifier.
    x_list = data["features"] if isinstance(data, dict) else data
    pos_tokens = []
    for p, (u, v, s) in enumerate(x_list):
        rank_mask = None
        if src_key_padding_masks is not None:
            rank_mask = src_key_padding_masks[p].bool()

        h = base.projectors[p](u, v, s)
        if getattr(base, "rank_encoder", None) is not None:
            h = base.rank_encoder(h, src_key_padding_mask=rank_mask)

        t_p = base.rank_pool(h, sigma=s, mask=rank_mask)
        pos_tokens.append(t_p)

    h_pos = torch.stack(pos_tokens, dim=1)
    e = base.layer_embedding(base.pos_layer_ids) + base.module_embedding(base.pos_module_ids)
    h_pos = base.pos_token_norm(h_pos + e.unsqueeze(0))
    h_pos = base.layer_encoder(h_pos)

    pos_mask = data.get("pos_mask", None) if isinstance(data, dict) else None
    pos_mask = pos_mask.bool() if pos_mask is not None else None
    return base.pos_pool(h_pos, mask=pos_mask)


def _build_embedding_extractor(model_type: str):
    def w2t_embed(model, data):
        return _w2t_encode_base(model, data, src_key_padding_masks=data.get("src_key_padding_masks"))

    def glnet_embed(model, data):
        base = model.base
        uvs_1 = base.equiv_mlp(data)
        out = torch.cat(
            [f(u, v).flatten(start_dim=1) for (u, v), f in zip(uvs_1, base.invariant_head.invariant_outputs)],
            dim=1,
        )
        h = base.invariant_head.linear1(out)
        h = base.invariant_head.ln1(h)
        return F.relu(h)

    def mlp_embed(model, data):
        base = model.base
        hs = [base.encoders[i](x) for i, x in enumerate(data)]
        h = torch.stack(hs, dim=1)
        return base.pool(h)

    def cnn_embed(model, data):
        base = model.base
        hs = [base.encoders[i](x) for i, x in enumerate(data)]
        h = torch.stack(hs, dim=1)
        return base.pool(h)

    def vit_embed(model, data):
        base = model.base
        tokens = data["tokens"]
        padding_mask = data.get("padding_mask", None)
        layer_ids = data.get("layer_ids", None)
        bsz = int(tokens.shape[0])
        h = base.proj(tokens)
        if layer_ids is not None:
            layer_ids = layer_ids.clamp(min=0, max=base.num_layers - 1)
            h = h + base.layer_embed(layer_ids)
        cls = base.cls.expand(bsz, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + base._build_sinusoidal_pos(h.shape[1], base.embed_dim, h.device, h.dtype)
        src_key_padding_mask = None
        if padding_mask is not None:
            cls_mask = torch.zeros((bsz, 1), dtype=torch.bool, device=padding_mask.device)
            src_key_padding_mask = torch.cat([cls_mask, padding_mask.bool()], dim=1)
        h = base.encoder(h, src_key_padding_mask=src_key_padding_mask)
        return base.norm(h[:, 0])

    table = {
        "w2t": w2t_embed,
        "glnet": glnet_embed,
        "mlp": mlp_embed,
        "cnn": cnn_embed,
        "vit": vit_embed,
    }
    if model_type not in table:
        raise ValueError(f"Unsupported model_type={model_type}")
    return table[model_type]


@torch.no_grad()
def encode_dataset(
    model_type: str,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    collate_fn,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    embed_fn = _build_embedding_extractor(model_type)
    all_emb: List[torch.Tensor] = []
    all_run_ids: List[int] = []
    for data, _, run_ids in loader:
        data = to_device(data, device)
        z = embed_fn(model, data).detach().float()
        if normalize:
            z = F.normalize(z, dim=1)
        all_emb.append(z.cpu())
        all_run_ids.extend(int(x) for x in run_ids)
    if not all_emb:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    emb = torch.cat(all_emb, dim=0).numpy()
    return emb, np.asarray(all_run_ids, dtype=np.int64)


def _resolve_model_types_arg(text: str) -> List[str]:
    vals = parse_csv_list(text)
    if len(vals) == 1 and vals[0] == "all":
        return ["w2t", "glnet", "mlp", "cnn", "vit"]
    for val in vals:
        if val not in {"w2t", "glnet", "mlp", "cnn", "vit"}:
            raise ValueError(f"Unknown model type: {val}")
    return vals


def _dcg_at_k(rel: np.ndarray, k: int) -> float:
    if rel.size == 0 or k <= 0:
        return 0.0
    gains = rel[:k].astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, 2 + gains.shape[0], dtype=np.float64))
    return float(np.sum(gains * discounts))


def _compute_retrieval_rows(
    sim: np.ndarray,
    query_meta: pd.DataFrame,
    gallery_meta: pd.DataFrame,
    topk: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gallery_dataset = gallery_meta["dataset"].astype(str).to_numpy()
    gallery_run_ids = gallery_meta["run_id"].astype(np.int64).to_numpy()

    per_query_rows: List[Dict[str, Any]] = []
    ranking_rows: List[Dict[str, Any]] = []
    for i in range(sim.shape[0]):
        scores = sim[i]
        order = np.argsort(-scores)
        q_dataset = str(query_meta.iloc[i]["dataset"])
        rel = (gallery_dataset == q_dataset).astype(np.int32)
        first_pos_rank = int(np.where(rel[order] > 0)[0][0] + 1) if np.any(rel[order] > 0) else -1
        top_idx = order[:topk]
        top_rel = rel[top_idx]
        num_pos = int(rel.sum())
        ideal_rel = np.ones(min(num_pos, topk), dtype=np.int32)
        ndcg = _dcg_at_k(top_rel, topk) / max(_dcg_at_k(ideal_rel, topk), 1e-12)
        top1_idx = int(order[0])

        per_query_rows.append(
            {
                "query_run_id": int(query_meta.iloc[i]["run_id"]),
                "query_dataset": q_dataset,
                "query_shot": int(pd.to_numeric(query_meta.iloc[i].get("train_samples", -1), errors="coerce"))
                if not pd.isna(query_meta.iloc[i].get("train_samples", np.nan))
                else -1,
                "top1_gallery_run_id": int(gallery_run_ids[top1_idx]),
                "top1_gallery_dataset": str(gallery_dataset[top1_idx]),
                "top1_score": float(scores[top1_idx]),
                "first_positive_rank": int(first_pos_rank),
                "hit_at_1": float(top_rel[0] > 0),
                "hit_at_10": float(np.any(top_rel[: min(10, top_rel.shape[0])] > 0)),
                "precision_at_10": float(np.mean(top_rel[: min(10, top_rel.shape[0])])) if top_rel.size else 0.0,
                "recall_at_10": float(np.sum(top_rel[: min(10, top_rel.shape[0])]) / max(num_pos, 1)),
                "mrr": float(1.0 / first_pos_rank) if first_pos_rank > 0 else 0.0,
                "ndcg_at_10": float(ndcg),
                "num_gallery_positives": int(num_pos),
            }
        )

        for rank, g_idx in enumerate(top_idx, start=1):
            ranking_rows.append(
                {
                    "query_run_id": int(query_meta.iloc[i]["run_id"]),
                    "query_dataset": q_dataset,
                    "query_shot": int(pd.to_numeric(query_meta.iloc[i].get("train_samples", -1), errors="coerce"))
                    if not pd.isna(query_meta.iloc[i].get("train_samples", np.nan))
                    else -1,
                    "rank": int(rank),
                    "gallery_run_id": int(gallery_run_ids[g_idx]),
                    "gallery_dataset": str(gallery_dataset[g_idx]),
                    "score": float(scores[g_idx]),
                    "is_positive": int(rel[g_idx]),
                }
            )

    return pd.DataFrame(per_query_rows), pd.DataFrame(ranking_rows)


def _aggregate_summary(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    agg = (
        df.groupby(list(group_cols), dropna=False)
        .agg(
            n_queries=("query_run_id", "count"),
            hit_at_1=("hit_at_1", "mean"),
            hit_at_10=("hit_at_10", "mean"),
            precision_at_10=("precision_at_10", "mean"),
            recall_at_10=("recall_at_10", "mean"),
            mrr=("mrr", "mean"),
            ndcg_at_10=("ndcg_at_10", "mean"),
        )
        .reset_index()
    )
    return agg


def _save_embedding_exports(
    out_root: Path,
    model_type: str,
    gallery_emb: np.ndarray,
    gallery_run_ids: np.ndarray,
    gallery_meta: pd.DataFrame,
    query_emb: np.ndarray,
    query_run_ids: np.ndarray,
    query_meta: pd.DataFrame,
) -> Dict[str, str]:
    emb_dir = ensure_dir(out_root / "embeddings")
    gallery_emb_path = emb_dir / f"{model_type}_gallery_emb.npz"
    query_emb_path = emb_dir / f"{model_type}_query_emb.npz"
    gallery_meta_path = emb_dir / f"{model_type}_gallery_meta.csv"
    query_meta_path = emb_dir / f"{model_type}_query_meta.csv"

    np.savez_compressed(
        gallery_emb_path,
        embeddings=np.asarray(gallery_emb, dtype=np.float32),
        run_ids=np.asarray(gallery_run_ids, dtype=np.int64),
    )
    np.savez_compressed(
        query_emb_path,
        embeddings=np.asarray(query_emb, dtype=np.float32),
        run_ids=np.asarray(query_run_ids, dtype=np.int64),
    )
    gallery_meta.to_csv(gallery_meta_path, index=False)
    query_meta.to_csv(query_meta_path, index=False)

    return {
        "gallery_emb_npz": str(gallery_emb_path),
        "query_emb_npz": str(query_emb_path),
        "gallery_meta_csv": str(gallery_meta_path),
        "query_meta_csv": str(query_meta_path),
    }


def _parse_path_map_args(path_map_items: Sequence[str], path_map_json: str = "") -> List[Tuple[str, str]]:
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
        mapping[str(src)] = str(dst)
    return sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)


def _pad_lora_factor_entry(entry: Dict[str, Any], target_rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    b = entry["B"].detach().cpu().float()
    a = entry["A"].detach().cpu().float()
    rank = int(a.shape[0])
    if rank > target_rank:
        raise RuntimeError(f"Observed rank {rank} exceeds target_rank={target_rank}")
    if rank == target_rank:
        return b, a
    b_pad = F.pad(b, (0, target_rank - rank))
    a_pad = F.pad(a, (0, 0, 0, target_rank - rank))
    return b_pad, a_pad


def _build_raw_weight_vector(entries: List[Dict[str, Any]], schema: Dict[str, Any]) -> np.ndarray:
    aligned_entries, reason = _align_entries_to_schema(entries, schema)
    if aligned_entries is None:
        raise RuntimeError(f"Failed aligning LoRA entries to shared schema: {reason}")
    parts: List[torch.Tensor] = []
    target_rank = int(schema["max_rank"])
    for entry in aligned_entries:
        b_pad, a_pad = _pad_lora_factor_entry(entry, target_rank)
        parts.append(b_pad.reshape(-1))
        parts.append(a_pad.reshape(-1))
    if not parts:
        raise RuntimeError("No LoRA A/B tensors found after schema alignment.")
    return torch.cat(parts, dim=0).numpy().astype(np.float32, copy=False)


def _encode_raw_weight_rows(
    rows: pd.DataFrame,
    path_maps: Sequence[Tuple[str, str]],
    normalize: bool,
    schema: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    loaded_entries: List[List[Dict[str, Any]]] = []
    run_ids: List[int] = []
    shared_schema = dict(schema) if schema is not None else None
    global_max_rank = int(shared_schema.get("max_rank", 0)) if shared_schema is not None else 0

    for row in rows.itertuples(index=False):
        resolved_path = resolve_safetensors_path(str(row.safetensors_path), path_maps)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Missing safetensors for run_id={int(row.run_id)}: {resolved_path}")
        tensors = st_load_torch(str(resolved_path))
        entries = _extract_lora_entries(tensors)
        if not entries:
            raise RuntimeError(f"No LoRA A/B pairs found in {resolved_path}")
        if shared_schema is None:
            shared_schema = _init_schema(entries)
        global_max_rank = max(global_max_rank, max(int(e["A"].shape[0]) for e in entries))
        loaded_entries.append(entries)
        run_ids.append(int(row.run_id))

    if not loaded_entries:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), shared_schema or {}

    assert shared_schema is not None
    shared_schema["max_rank"] = int(global_max_rank)

    emb_list: List[np.ndarray] = []
    for entries in loaded_entries:
        vec = _build_raw_weight_vector(entries, shared_schema)
        emb_list.append(vec)

    emb = np.stack(emb_list, axis=0).astype(np.float32, copy=False)
    if normalize:
        denom = np.linalg.norm(emb, axis=1, keepdims=True)
        denom = np.maximum(denom, 1e-12)
        emb = emb / denom
    return emb, np.asarray(run_ids, dtype=np.int64), shared_schema or {}


def cmd_retrieve_rawcos(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    metadata_path = Path(args.metadata_csv)
    split_dir = Path(args.split_dir)
    out_root = ensure_dir(args.output_dir)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not split_dir.exists():
        raise FileNotFoundError(split_dir)

    meta_df = pd.read_csv(metadata_path)
    if "run_id" not in meta_df.columns or "dataset" not in meta_df.columns:
        raise ValueError("--metadata-csv must contain run_id and dataset columns.")
    if "safetensors_path" not in meta_df.columns:
        raise ValueError("--metadata-csv must contain safetensors_path for raw-weight retrieval.")
    if meta_df["run_id"].duplicated().any():
        raise ValueError("metadata_csv has duplicated run_id values; run_ids must be unique.")

    split_map = _load_split_map(split_dir)
    gallery_ids = [int(rid) for rid, split in split_map.items() if split == str(args.gallery_split)]
    query_ids = [int(rid) for rid, split in split_map.items() if split == str(args.query_split)]
    if not gallery_ids or not query_ids:
        raise RuntimeError(f"Empty raw retrieval split: gallery={len(gallery_ids)} query={len(query_ids)}")

    gallery_meta = meta_df.set_index("run_id").loc[gallery_ids].reset_index().sort_values("run_id").reset_index(drop=True)
    query_meta = meta_df.set_index("run_id").loc[query_ids].reset_index().sort_values("run_id").reset_index(drop=True)
    gallery_rank_filter = parse_int_csv_list(args.gallery_lora_ranks)
    query_rank_filter = parse_int_csv_list(args.query_lora_ranks)
    if gallery_rank_filter:
        if "lora_r" not in gallery_meta.columns:
            raise ValueError("--gallery-lora-ranks requires metadata_csv to contain lora_r.")
        gallery_meta = gallery_meta[gallery_meta["lora_r"].astype(int).isin(gallery_rank_filter)].copy()
    if query_rank_filter:
        if "lora_r" not in query_meta.columns:
            raise ValueError("--query-lora-ranks requires metadata_csv to contain lora_r.")
        query_meta = query_meta[query_meta["lora_r"].astype(int).isin(query_rank_filter)].copy()
    if gallery_meta.empty or query_meta.empty:
        raise RuntimeError(
            f"Empty raw retrieval split after rank filtering: gallery={len(gallery_meta)} query={len(query_meta)}"
        )
    print(
        f"[retrieve-rawcos] rank filter gallery={gallery_rank_filter or 'all'} "
        f"query={query_rank_filter or 'all'} -> gallery={len(gallery_meta)} query={len(query_meta)}",
        flush=True,
    )
    path_maps = _parse_path_map_args(args.path_map, args.path_map_json)

    print("[retrieve-rawcos] encoding gallery raw LoRA weights", flush=True)
    gallery_emb, gallery_run_ids, schema = _encode_raw_weight_rows(
        gallery_meta,
        path_maps=path_maps,
        normalize=bool(args.normalize),
        schema=None,
    )
    print("[retrieve-rawcos] encoding query raw LoRA weights", flush=True)
    query_emb, query_run_ids, _ = _encode_raw_weight_rows(
        query_meta,
        path_maps=path_maps,
        normalize=bool(args.normalize),
        schema=schema,
    )

    gallery_meta = meta_df.set_index("run_id").loc[gallery_run_ids].reset_index()
    query_meta = meta_df.set_index("run_id").loc[query_run_ids].reset_index()

    embedding_exports: Dict[str, str] = {}
    if bool(args.save_embeddings):
        embedding_exports = _save_embedding_exports(
            out_root=out_root,
            model_type="rawcos",
            gallery_emb=gallery_emb,
            gallery_run_ids=gallery_run_ids,
            gallery_meta=gallery_meta,
            query_emb=query_emb,
            query_run_ids=query_run_ids,
            query_meta=query_meta,
        )

    sim = np.matmul(query_emb.astype(np.float32), gallery_emb.astype(np.float32).T)
    per_query_df, ranking_df = _compute_retrieval_rows(sim, query_meta, gallery_meta, topk=int(args.topk))
    per_query_df["query_split"] = str(args.query_split)

    per_query_path = out_root / "rawcos_per_query.csv"
    ranking_path = out_root / "rawcos_topk.csv"
    per_query_df.to_csv(per_query_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)

    overall_df = _aggregate_summary(per_query_df, group_cols=["query_split"])
    by_dataset_df = _aggregate_summary(per_query_df, group_cols=["query_dataset"])
    by_shot_df = _aggregate_summary(per_query_df, group_cols=["query_shot"])
    by_dataset_shot_df = _aggregate_summary(per_query_df, group_cols=["query_dataset", "query_shot"])

    for df in [overall_df, by_dataset_df, by_shot_df, by_dataset_shot_df]:
        df.insert(0, "model_type", "rawcos")

    overall_df.to_csv(out_root / "rawcos_summary_overall.csv", index=False)
    by_dataset_df.to_csv(out_root / "rawcos_summary_by_dataset.csv", index=False)
    by_shot_df.to_csv(out_root / "rawcos_summary_by_shot.csv", index=False)
    by_dataset_shot_df.to_csv(out_root / "rawcos_summary_by_dataset_shot.csv", index=False)

    overall_row = overall_df.iloc[0].to_dict()
    summary_df = pd.DataFrame(
        [
            {
                "model_type": "rawcos",
                "n_gallery": int(len(gallery_run_ids)),
                "n_query": int(len(query_run_ids)),
                "hit_at_1": float(overall_row["hit_at_1"]),
                "hit_at_10": float(overall_row["hit_at_10"]),
                "precision_at_10": float(overall_row["precision_at_10"]),
                "recall_at_10": float(overall_row["recall_at_10"]),
                "mrr": float(overall_row["mrr"]),
                "ndcg_at_10": float(overall_row["ndcg_at_10"]),
                "per_query_csv": str(per_query_path),
                "topk_csv": str(ranking_path),
                **embedding_exports,
            }
        ]
    )
    summary_path = out_root / "retrieval_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    dump_json(
        out_root / "retrieval_config.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "raw_weight_cosine",
            "metadata_csv": str(metadata_path.resolve()),
            "split_dir": str(split_dir.resolve()),
            "gallery_split": str(args.gallery_split),
            "query_split": str(args.query_split),
            "normalize": bool(args.normalize),
            "topk": int(args.topk),
            "path_maps": [{"src": src, "dst": dst} for src, dst in path_maps],
            "summary_csv": str(summary_path.resolve()),
        },
    )
    print(
        f"[retrieve-rawcos] done hit@1={float(overall_row['hit_at_1']):.4f} "
        f"hit@10={float(overall_row['hit_at_10']):.4f} ndcg@10={float(overall_row['ndcg_at_10']):.4f}",
        flush=True,
    )
    print(f"[retrieve-rawcos] summary: {summary_path}", flush=True)


def cmd_retrieve(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(args.seed)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    metadata_path = Path(args.metadata_csv)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    trained_root = Path(args.trained_root)
    if not trained_root.exists():
        raise FileNotFoundError(trained_root)
    out_root = ensure_dir(args.output_dir)

    meta_df = pd.read_csv(metadata_path)
    if "run_id" not in meta_df.columns or "dataset" not in meta_df.columns:
        raise ValueError("--metadata-csv must contain run_id and dataset columns.")
    if meta_df["run_id"].duplicated().any():
        raise ValueError("metadata_csv has duplicated run_id values after prepare; run_ids must be unique.")

    model_types = _resolve_model_types_arg(args.model_type)
    summary_rows: List[Dict[str, Any]] = []

    for model_type in model_types:
        print(f"[retrieve] start model={model_type}", flush=True)
        reader = ShardCacheReader(manifest_path, max_cached_shards=int(args.max_cached_shards))
        gallery_ds = LoRACacheDataset(reader, split=args.gallery_split, model_type=model_type, target_col=args.target_col)
        query_ds = LoRACacheDataset(reader, split=args.query_split, model_type=model_type, target_col=args.target_col)
        if len(gallery_ds) == 0 or len(query_ds) == 0:
            raise RuntimeError(f"Empty retrieval split: gallery={len(gallery_ds)} query={len(query_ds)}")

        ckpt_path = trained_root / model_type / args.checkpoint_name
        ckpt = load_checkpoint_with_fallback(ckpt_path)
        sample_feature, _, _ = gallery_ds[0]

        flat_layer_dims = None
        if model_type == "mlp":
            ckpt_dims = ckpt.get("flat_layer_dims")
            if ckpt_dims is not None:
                flat_layer_dims = [int(x) for x in ckpt_dims]
            else:
                inferred = _infer_mlp_flat_dims_from_state_dict(ckpt["state_dict"])
                if inferred is not None:
                    flat_layer_dims = inferred
                else:
                    flat_layer_dims = []

        model_args = _model_args_from_checkpoint(ckpt.get("args"), args)
        model = build_model(
            model_type,
            sample_feature,
            model_args,
            device,
            flat_layer_dims=flat_layer_dims if flat_layer_dims else None,
        )
        state_dict = adapt_state_dict_for_model(model_type, ckpt["state_dict"], model)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"State dict mismatch after compatibility mapping. missing={missing[:10]} unexpected={unexpected[:10]}"
            )
        model.eval()
        collate_fn = get_collate_fn(model_type, flat_layer_dims=flat_layer_dims if flat_layer_dims else None)

        gallery_emb, gallery_run_ids = encode_dataset(
            model_type=model_type,
            model=model,
            dataset=gallery_ds,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
            collate_fn=collate_fn,
            normalize=bool(args.normalize),
        )
        query_emb, query_run_ids = encode_dataset(
            model_type=model_type,
            model=model,
            dataset=query_ds,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
            collate_fn=collate_fn,
            normalize=bool(args.normalize),
        )

        gallery_meta = meta_df.set_index("run_id").loc[gallery_run_ids].reset_index()
        query_meta = meta_df.set_index("run_id").loc[query_run_ids].reset_index()

        embedding_exports: Dict[str, str] = {}
        if bool(args.save_embeddings):
            embedding_exports = _save_embedding_exports(
                out_root=out_root,
                model_type=model_type,
                gallery_emb=gallery_emb,
                gallery_run_ids=gallery_run_ids,
                gallery_meta=gallery_meta,
                query_emb=query_emb,
                query_run_ids=query_run_ids,
                query_meta=query_meta,
            )

        sim = np.matmul(query_emb.astype(np.float32), gallery_emb.astype(np.float32).T)

        per_query_df, ranking_df = _compute_retrieval_rows(sim, query_meta, gallery_meta, topk=int(args.topk))
        per_query_df["query_split"] = str(args.query_split)
        per_query_path = out_root / f"{model_type}_per_query.csv"
        ranking_path = out_root / f"{model_type}_topk.csv"
        per_query_df.to_csv(per_query_path, index=False)
        ranking_df.to_csv(ranking_path, index=False)

        overall_df = _aggregate_summary(per_query_df, group_cols=["query_split"])
        if overall_df.empty:
            overall_df = pd.DataFrame(
                [
                    {
                        "query_split": args.query_split,
                        "n_queries": 0,
                        "hit_at_1": float("nan"),
                        "hit_at_10": float("nan"),
                        "precision_at_10": float("nan"),
                        "recall_at_10": float("nan"),
                        "mrr": float("nan"),
                        "ndcg_at_10": float("nan"),
                    }
                ]
            )
        by_dataset_df = _aggregate_summary(per_query_df, group_cols=["query_dataset"])
        by_shot_df = _aggregate_summary(per_query_df, group_cols=["query_shot"])
        by_dataset_shot_df = _aggregate_summary(per_query_df, group_cols=["query_dataset", "query_shot"])

        overall_df.insert(0, "model_type", model_type)
        by_dataset_df.insert(0, "model_type", model_type)
        by_shot_df.insert(0, "model_type", model_type)
        by_dataset_shot_df.insert(0, "model_type", model_type)

        overall_df.to_csv(out_root / f"{model_type}_summary_overall.csv", index=False)
        by_dataset_df.to_csv(out_root / f"{model_type}_summary_by_dataset.csv", index=False)
        by_shot_df.to_csv(out_root / f"{model_type}_summary_by_shot.csv", index=False)
        by_dataset_shot_df.to_csv(out_root / f"{model_type}_summary_by_dataset_shot.csv", index=False)

        overall_row = overall_df.iloc[0].to_dict()
        summary_rows.append(
            {
                "model_type": model_type,
                "n_gallery": int(len(gallery_run_ids)),
                "n_query": int(len(query_run_ids)),
                "hit_at_1": float(overall_row["hit_at_1"]),
                "hit_at_10": float(overall_row["hit_at_10"]),
                "precision_at_10": float(overall_row["precision_at_10"]),
                "recall_at_10": float(overall_row["recall_at_10"]),
                "mrr": float(overall_row["mrr"]),
                "ndcg_at_10": float(overall_row["ndcg_at_10"]),
                "per_query_csv": str(per_query_path),
                "topk_csv": str(ranking_path),
                **embedding_exports,
            }
        )
        print(
            f"[retrieve] done model={model_type} hit@1={float(overall_row['hit_at_1']):.4f} "
            f"hit@10={float(overall_row['hit_at_10']):.4f} ndcg@10={float(overall_row['ndcg_at_10']):.4f}",
            flush=True,
        )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("model_type").reset_index(drop=True)
        summary_path = out_root / "retrieval_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        dump_json(
            out_root / "retrieval_config.json",
            {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "manifest": str(manifest_path.resolve()),
                "metadata_csv": str(metadata_path.resolve()),
                "trained_root": str(trained_root.resolve()),
                "model_type": args.model_type,
                "checkpoint_name": args.checkpoint_name,
                "gallery_split": args.gallery_split,
                "query_split": args.query_split,
                "target_col": args.target_col,
                "batch_size": int(args.batch_size),
                "num_workers": int(args.num_workers),
                "normalize": bool(args.normalize),
                "save_embeddings": bool(args.save_embeddings),
                "topk": int(args.topk),
                "summary_csv": str(summary_path.resolve()),
            },
        )
        print(f"[retrieve] summary: {summary_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Few-shot LoRA retrieval workflow.")
    sub = p.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Merge gallery/query metadata and create cache split files.")
    prep.add_argument("--gallery-csvs", type=str, required=True, help="Comma-separated CSV paths or glob patterns.")
    prep.add_argument("--query-csvs", type=str, required=True, help="Comma-separated CSV paths or glob patterns.")
    prep.add_argument("--output-dir", type=str, required=True)
    prep.add_argument("--gallery-per-dataset", type=int, default=1000)
    prep.add_argument("--max-queries-per-dataset-shot", type=int, default=-1)
    prep.add_argument("--seed", type=int, default=42)
    prep.set_defaults(func=cmd_prepare)

    ret = sub.add_parser("retrieve", help="Run frozen-encoder retrieval from cache manifest.")
    ret.add_argument("--manifest", type=str, required=True)
    ret.add_argument("--metadata-csv", type=str, required=True)
    ret.add_argument("--trained-root", type=str, required=True)
    ret.add_argument("--output-dir", type=str, required=True)
    ret.add_argument("--model-type", type=str, default="w2t")
    ret.add_argument("--checkpoint-name", type=str, default="best_model.pth")
    ret.add_argument("--gallery-split", type=str, default="train", choices=["train", "valid", "test"])
    ret.add_argument("--query-split", type=str, default="test", choices=["train", "valid", "test"])
    ret.add_argument("--target-col", type=str, default="test_acc", choices=TARGET_COLUMNS)
    ret.add_argument("--batch-size", type=int, default=128)
    ret.add_argument("--num-workers", type=int, default=0)
    ret.add_argument("--max-cached-shards", type=int, default=4)
    ret.add_argument("--normalize", action="store_true", default=True)
    ret.add_argument("--no-normalize", dest="normalize", action="store_false")
    ret.add_argument("--save-embeddings", action="store_true", default=True)
    ret.add_argument("--no-save-embeddings", dest="save_embeddings", action="store_false")
    ret.add_argument("--topk", type=int, default=10)
    ret.add_argument("--hidden-dim", type=int, default=128)
    ret.add_argument("--mlp-dim", type=int, default=128)
    ret.add_argument("--dropout", type=float, default=0.0)
    ret.add_argument("--nhead", type=int, default=4)
    ret.add_argument("--num-rank-layers", type=int, default=1)
    ret.add_argument("--num-layer-layers", type=int, default=2)
    ret.add_argument("--vit-depth", type=int, default=4)
    ret.add_argument("--token-size", type=int, default=2048)
    ret.add_argument("--glnet-layers", type=int, default=1)
    ret.add_argument("--seed", type=int, default=42)
    ret.add_argument("--device", type=str, default="auto")
    ret.set_defaults(func=cmd_retrieve)

    raw = sub.add_parser("retrieve-rawcos", help="Run retrieval by cosine similarity on raw LoRA weights.")
    raw.add_argument("--metadata-csv", type=str, required=True)
    raw.add_argument("--split-dir", type=str, required=True)
    raw.add_argument("--output-dir", type=str, required=True)
    raw.add_argument("--gallery-split", type=str, default="train", choices=["train", "valid", "test"])
    raw.add_argument("--query-split", type=str, default="test", choices=["train", "valid", "test"])
    raw.add_argument("--topk", type=int, default=10)
    raw.add_argument("--seed", type=int, default=42)
    raw.add_argument("--normalize", action="store_true", default=True)
    raw.add_argument("--no-normalize", dest="normalize", action="store_false")
    raw.add_argument("--save-embeddings", action="store_true", default=True)
    raw.add_argument("--no-save-embeddings", dest="save_embeddings", action="store_false")
    raw.add_argument("--gallery-lora-ranks", type=str, default="", help="Comma-separated gallery lora_r values to keep.")
    raw.add_argument("--query-lora-ranks", type=str, default="", help="Comma-separated query lora_r values to keep.")
    raw.add_argument("--path-map", action="append", default=[], help="Path rewrite rule source=target.")
    raw.add_argument("--path-map-json", type=str, default="", help="JSON file with {source_prefix: target_prefix}.")
    raw.set_defaults(func=cmd_retrieve_rawcos)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
