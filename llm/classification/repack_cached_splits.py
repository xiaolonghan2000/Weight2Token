import os, glob, json, gc, random
import argparse
from collections import defaultdict
from tqdm import tqdm

import torch
from safetensors.torch import save_file


def make_split(names, seed=42, train_ratio=0.8, valid_ratio=0.1):
    rng = random.Random(seed)
    names = list(names)
    rng.shuffle(names)
    n = len(names)
    n_train = int(train_ratio * n)
    n_valid = int(valid_ratio * n)
    train = set(names[:n_train])
    valid = set(names[n_train:n_train + n_valid])
    test  = set(names[n_train + n_valid:])
    return train, valid, test


def flush_split(out_dir, split, shard_id, buf, dtype):
    if not buf:
        return shard_id, {}

    os.makedirs(out_dir, exist_ok=True)
    B = len(buf)
    meta = buf[0]["meta"]
    P = len(buf[0]["features"])

    # label
    y0 = buf[0]["label"]
    C = int(y0.numel()) if isinstance(y0, torch.Tensor) else int(len(y0))
    Y = torch.zeros((B, C), dtype=torch.float32)

    U_list = [[] for _ in range(P)]
    V_list = [[] for _ in range(P)]
    S_list = [[] for _ in range(P)]
    names = []

    for bi, it in enumerate(buf):
        names.append(it["name"])
        if isinstance(it["label"], torch.Tensor):
            Y[bi] = it["label"].float().view(-1)
        else:
            Y[bi] = torch.tensor(it["label"], dtype=torch.float32)

        feats = it["features"]
        for p in range(P):
            u, v, s = feats[p]
            if not isinstance(u, torch.Tensor): u = torch.from_numpy(u)
            if not isinstance(v, torch.Tensor): v = torch.from_numpy(v)
            if not isinstance(s, torch.Tensor): s = torch.from_numpy(s)
            U_list[p].append(u.to(dtype=dtype))
            V_list[p].append(v.to(dtype=dtype))
            S_list[p].append(s.to(dtype=dtype))

    tensors = {}
    for p in range(P):
        tensors[f"U_{p}"] = torch.stack(U_list[p], 0).contiguous()
        tensors[f"V_{p}"] = torch.stack(V_list[p], 0).contiguous()
        tensors[f"S_{p}"] = torch.stack(S_list[p], 0).contiguous()
    tensors["Y"] = Y.contiguous()

    st_path = os.path.join(out_dir, f"shard_{shard_id:05d}.safetensors")
    save_file(tensors, st_path)

    with open(os.path.join(out_dir, f"shard_{shard_id:05d}.names.json"), "w") as f:
        json.dump(names, f)

    meta_path = os.path.join(out_dir, "meta.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    return shard_id + 1, {os.path.basename(st_path): B}


def main():
    ap = argparse.ArgumentParser("Fast repack (single node): each input pt loaded once")
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="cache_part_*.pt")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--out_shard_size", type=int, default=1024)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    in_paths = sorted(glob.glob(os.path.join(args.in_dir, args.pattern)))
    if not in_paths:
        raise FileNotFoundError("No input pt files found.")

    # ---------- pass 1: collect names only ----------
    all_names = []
    for p in tqdm(in_paths, desc="Pass1: scan names"):
        shard = torch.load(p, map_location="cpu", weights_only=False)
        for it in shard:
            n = it.get("name", None)
            if n is not None:
                all_names.append(n)
        del shard
        gc.collect()

    train_set, valid_set, test_set = make_split(all_names, args.seed, args.train_ratio, args.valid_ratio)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "split_train.json"), "w") as f:
        json.dump(sorted(list(train_set)), f)
    with open(os.path.join(args.out_dir, "split_valid.json"), "w") as f:
        json.dump(sorted(list(valid_set)), f)
    with open(os.path.join(args.out_dir, "split_test.json"), "w") as f:
        json.dump(sorted(list(test_set)), f)

    # ---------- pass 2: repack by reading each input pt once ----------
    out_train = os.path.join(args.out_dir, "train")
    out_valid = os.path.join(args.out_dir, "valid")
    out_test  = os.path.join(args.out_dir, "test")

    buf_train, buf_valid, buf_test = [], [], []
    sid_train = sid_valid = sid_test = 0
    man_train, man_valid, man_test = {}, {}, {}

    for p in tqdm(in_paths, desc="Pass2: repack"):
        shard = torch.load(p, map_location="cpu", weights_only=False)
        for it in shard:
            n = it.get("name", None)
            if n is None:
                continue
            if n in train_set:
                buf_train.append(it)
                if len(buf_train) >= args.out_shard_size:
                    sid_train, m = flush_split(out_train, "train", sid_train, buf_train, dtype)
                    man_train.update(m)
                    buf_train = []
            elif n in valid_set:
                buf_valid.append(it)
                if len(buf_valid) >= args.out_shard_size:
                    sid_valid, m = flush_split(out_valid, "valid", sid_valid, buf_valid, dtype)
                    man_valid.update(m)
                    buf_valid = []
            else:
                buf_test.append(it)
                if len(buf_test) >= args.out_shard_size:
                    sid_test, m = flush_split(out_test, "test", sid_test, buf_test, dtype)
                    man_test.update(m)
                    buf_test = []
        del shard
        gc.collect()

    # flush remaining
    sid_train, m = flush_split(out_train, "train", sid_train, buf_train, dtype); man_train.update(m)
    sid_valid, m = flush_split(out_valid, "valid", sid_valid, buf_valid, dtype); man_valid.update(m)
    sid_test,  m = flush_split(out_test,  "test",  sid_test,  buf_test,  dtype); man_test.update(m)

    with open(os.path.join(out_train, "manifest.json"), "w") as f: json.dump(man_train, f)
    with open(os.path.join(out_valid, "manifest.json"), "w") as f: json.dump(man_valid, f)
    with open(os.path.join(out_test,  "manifest.json"), "w") as f: json.dump(man_test,  f)

    print("[DONE]")
    print("train items:", sum(man_train.values()), "shards:", len(man_train))
    print("valid items:", sum(man_valid.values()), "shards:", len(man_valid))
    print("test  items:", sum(man_test.values()),  "shards:", len(man_test))


if __name__ == "__main__":
    main()