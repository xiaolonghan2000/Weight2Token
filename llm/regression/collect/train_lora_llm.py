import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file as save_safetensors
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


@dataclass
class SFTDataCollator:
    tokenizer: Any
    pad_to_multiple_of: int | None = None

    def __call__(self, features):
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        max_len = batch["input_ids"].shape[1]

        padded = []
        for lab in labels:
            lab = list(lab)
            if len(lab) > max_len:
                lab = lab[:max_len]
            pad_len = max_len - len(lab)
            padded.append(lab + [-100] * pad_len)

        batch["labels"] = torch.tensor(padded, dtype=torch.long)
        return batch


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def now_str():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def ppl_from_loss(loss: float) -> float:
    if loss > 50:
        return float("inf")
    return float(math.exp(loss))


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_csv_row(path: str, row: Dict[str, object], field_order: Optional[List[str]] = None):
    import csv

    def _lock(path_lock: str):
        fd = os.open(path_lock, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        except Exception:
            pass
        return fd

    def _unlock(fd: int):
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    lock_fd = _lock(path + ".lock")
    try:
        exists = os.path.exists(path)
        header = None
        if exists:
            try:
                with open(path, "r", newline="", encoding="utf-8") as f:
                    r = csv.reader(f)
                    header = next(r, None)
            except OSError:
                header = None

        if header:
            # Keep schema stable when appending to existing results files from older runs.
            field_order = list(header)
        elif field_order is None:
            field_order = list(row.keys())

        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=field_order, extrasaction="ignore")
            if (not exists) or os.path.getsize(path) == 0:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in field_order})
            f.flush()
            os.fsync(f.fileno())
    finally:
        _unlock(lock_fd)


def sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "unknown"


def _tmp_path(path: str) -> str:
    return f"{path}.tmp_{os.getpid()}_{int(time.time())}"


def save_jsonl(path: str, rows: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def acquire_dir_lock(lock_dir: str, timeout_sec: int = 7200, poll_sec: float = 2.0):
    start = time.time()
    while True:
        try:
            os.mkdir(lock_dir)
            return
        except FileExistsError:
            if time.time() - start > timeout_sec:
                raise TimeoutError(f"timeout waiting lock: {lock_dir}")
            time.sleep(poll_sec)


def release_dir_lock(lock_dir: str):
    try:
        os.rmdir(lock_dir)
    except OSError:
        pass


def format_arc_example(ex: dict) -> Tuple[str, str, List[str]]:
    q = ex["question"]
    choices = ex["choices"]
    labels = choices["label"]
    texts = choices["text"]
    pairs = sorted(zip(labels, texts), key=lambda x: x[0])
    opt_texts = [p[1] for p in pairs]

    ans_key = ex["answerKey"]
    gold_text = None
    for lab, txt in pairs:
        if lab == ans_key:
            gold_text = txt
            break

    prompt = (
        "You are a helpful assistant. Answer the multiple-choice question by choosing the correct option.\n\n"
        f"Question: {q}\n"
        "Options:\n"
        + "\n".join([f"{lab}. {txt}" for lab, txt in pairs])
        + "\n\nAnswer:"
    )
    return prompt, gold_text, opt_texts


def format_boolq_example(ex: dict) -> Tuple[str, str, List[str]]:
    passage = ex.get("passage", "")
    question = ex.get("question", "")

    if "answer" in ex:
        ans_raw = ex["answer"]
    elif "label" in ex:
        ans_raw = ex["label"]
    else:
        raise KeyError(f"BoolQ sample has neither 'answer' nor 'label'. keys={list(ex.keys())}")

    if isinstance(ans_raw, str):
        norm = ans_raw.strip().lower()
        if norm in {"1", "true", "yes"}:
            ans_bool = True
        elif norm in {"0", "false", "no"}:
            ans_bool = False
        else:
            raise ValueError(f"Unsupported BoolQ label string: {ans_raw!r}")
    else:
        ans_bool = bool(ans_raw)

    label = "yes" if ans_bool else "no"
    options = ["yes", "no"]
    prompt = (
        "You are a helpful assistant. Answer with 'yes' or 'no'.\n\n"
        f"Passage: {passage}\n"
        f"Question: {question}\n\nAnswer:"
    )
    return prompt, label, options


def format_gsm8k_example(ex: dict) -> Tuple[str, str]:
    q = ex["question"]
    a = ex["answer"]
    prompt = (
        "Solve the math problem. Give the final numeric answer.\n\n"
        f"Question: {q}\n\nAnswer:"
    )
    return prompt, a


def format_mbpp_example(ex: dict) -> Tuple[str, str]:
    text = ex.get("text", "")
    code = ex.get("code", "")
    prompt = "Write a Python function that solves the task.\n\n" f"Task: {text}\n\nAnswer:\n"
    return prompt, code


def build_sft_dataset(ds, tokenizer, formatter, max_len: int, num_proc: int = 0, desc: str = ""):
    def _map(ex):
        if formatter.__name__.startswith("format_arc"):
            prompt, gold_text, _ = formatter(ex)
            target = " " + (gold_text if gold_text is not None else "")
        elif formatter.__name__.startswith("format_boolq"):
            prompt, gold_label, _ = formatter(ex)
            target = " " + gold_label
        elif formatter.__name__.startswith("format_gsm8k"):
            prompt, ans = formatter(ex)
            target = " " + ans
        else:
            prompt, ans = formatter(ex)
            target = "\n" + ans

        p = tokenizer(prompt, truncation=True, max_length=max_len)
        t = tokenizer(target, truncation=True, max_length=max_len)

        input_ids = (p["input_ids"] + t["input_ids"])[:max_len]
        attn = [1] * len(input_ids)
        labels = ([-100] * len(p["input_ids"]) + t["input_ids"])[:max_len]
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    map_kwargs = {"remove_columns": ds.column_names}
    if num_proc and num_proc > 1:
        map_kwargs["num_proc"] = int(num_proc)
    if desc:
        map_kwargs["desc"] = desc
    return ds.map(_map, **map_kwargs)


def load_raw_dataset_and_formatter(dataset_name: str, hf_cache_dir: str = ""):
    cache_dir = hf_cache_dir or None
    if dataset_name in ["arc-easy", "arc-challenge"]:
        arc_name = "ARC-Easy" if dataset_name == "arc-easy" else "ARC-Challenge"
        raw = load_dataset("ai2_arc", arc_name, cache_dir=cache_dir)
        formatter = format_arc_example
        train_raw = raw["train"]
        test_raw = raw["test"]
    elif dataset_name == "boolq":
        raw = load_dataset("super_glue", "boolq", cache_dir=cache_dir)
        formatter = format_boolq_example
        train_raw = raw["train"]
        test_raw = raw["validation"]
    elif dataset_name == "gsm8k":
        raw = load_dataset("gsm8k", "main", cache_dir=cache_dir)
        formatter = format_gsm8k_example
        train_raw = raw["train"]
        test_raw = raw["test"]
    else:
        raw = load_dataset("mbpp", cache_dir=cache_dir)
        formatter = format_mbpp_example
        train_raw = raw["train"]
        test_raw = raw["test"]
    return train_raw, test_raw, formatter


def build_eval_items(dataset_name: str, ds_test) -> List[dict]:
    items = []
    if dataset_name in ["arc-easy", "arc-challenge"]:
        for ex in ds_test:
            prompt, gold_text, options = format_arc_example(ex)
            if gold_text is None:
                continue
            items.append({"prompt": prompt, "gold": gold_text, "options": options})
    elif dataset_name == "boolq":
        for ex in ds_test:
            prompt, gold, options = format_boolq_example(ex)
            items.append({"prompt": prompt, "gold": gold, "options": options})
    return items


def cache_paths(cache_root: str, dataset_name: str, tok_name: str, max_len: int):
    key_obj = {
        "dataset": dataset_name,
        "tokenizer": tok_name,
        "max_len": int(max_len),
        "format_version": "v2",
    }
    key_hash = hashlib.md5(json.dumps(key_obj, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    key_name = f"{sanitize_name(dataset_name)}_{sanitize_name(tok_name.split('/')[-1])}_len{max_len}_{key_hash}"
    root = os.path.join(cache_root, key_name)
    return {
        "root": root,
        "lock": root + ".lock",
        "train": os.path.join(root, "train_tok"),
        "test": os.path.join(root, "test_tok"),
        "eval": os.path.join(root, "eval_items.jsonl"),
        "meta": os.path.join(root, "meta.json"),
    }


def maybe_select_train_subset(train_ds, train_samples: int, subset_seed: int) -> Tuple[Any, Dict[str, Any]]:
    total_size = int(len(train_ds))
    info: Dict[str, Any] = {
        "train_dataset_size": total_size,
        "train_samples_requested": int(train_samples),
        "train_samples_effective": total_size,
        "subset_seed": int(subset_seed),
        "subset_hash": "",
        "selected_indices": [],
    }
    if train_samples is None or int(train_samples) <= 0:
        return train_ds, info
    if int(train_samples) > total_size:
        raise ValueError(
            f"--train-samples={train_samples} exceeds dataset size {total_size} for the selected train split."
        )
    import numpy as np

    rng = np.random.default_rng(int(subset_seed))
    indices = sorted(int(x) for x in rng.choice(total_size, size=int(train_samples), replace=False).tolist())
    sig = hashlib.md5(",".join(str(x) for x in indices).encode("utf-8")).hexdigest()[:12]
    subset = train_ds.select(indices)
    info["train_samples_effective"] = int(len(subset))
    info["subset_hash"] = sig
    info["selected_indices"] = indices
    return subset, info


def load_or_build_prepared_data(args, tokenizer):
    need_eval_items = args.eval_acc_mode != "none" and args.dataset in ["arc-easy", "arc-challenge", "boolq"]

    if args.disable_data_cache:
        train_raw, test_raw, formatter = load_raw_dataset_and_formatter(args.dataset, args.hf_cache_dir)
        train_ds = build_sft_dataset(
            train_raw,
            tokenizer,
            formatter,
            args.max_len,
            num_proc=args.map_num_proc,
            desc=f"tokenize-{args.dataset}-train",
        )
        test_tok = build_sft_dataset(
            test_raw,
            tokenizer,
            formatter,
            args.max_len,
            num_proc=args.map_num_proc,
            desc=f"tokenize-{args.dataset}-test",
        )
        eval_items = build_eval_items(args.dataset, test_raw) if need_eval_items else None
        train_ds, subset_info = maybe_select_train_subset(train_ds, args.train_samples, args.subset_seed)
        return train_ds, test_tok, eval_items, subset_info

    ensure_dir(args.data_cache_dir)
    tok_name = getattr(tokenizer, "name_or_path", args.base_model)
    paths = cache_paths(args.data_cache_dir, args.dataset, tok_name, args.max_len)
    ensure_dir(paths["root"])

    have_train = os.path.isdir(paths["train"])
    have_test = os.path.isdir(paths["test"])
    have_eval = (not need_eval_items) or os.path.isfile(paths["eval"])
    if have_train and have_test and have_eval:
        print(f"[CACHE] hit dataset={args.dataset} root={paths['root']}")
        train_ds = load_from_disk(paths["train"])
        test_tok = load_from_disk(paths["test"])
        eval_items = load_jsonl(paths["eval"]) if need_eval_items else None
        train_ds, subset_info = maybe_select_train_subset(train_ds, args.train_samples, args.subset_seed)
        return train_ds, test_tok, eval_items, subset_info

    print(f"[CACHE] miss dataset={args.dataset} root={paths['root']}, building...")
    acquire_dir_lock(paths["lock"], timeout_sec=args.cache_lock_timeout, poll_sec=2.0)
    try:
        have_train = os.path.isdir(paths["train"])
        have_test = os.path.isdir(paths["test"])
        have_eval = (not need_eval_items) or os.path.isfile(paths["eval"])
        if not (have_train and have_test and have_eval):
            train_raw, test_raw, formatter = load_raw_dataset_and_formatter(args.dataset, args.hf_cache_dir)
            train_ds = build_sft_dataset(
                train_raw,
                tokenizer,
                formatter,
                args.max_len,
                num_proc=args.map_num_proc,
                desc=f"tokenize-{args.dataset}-train",
            )
            test_tok = build_sft_dataset(
                test_raw,
                tokenizer,
                formatter,
                args.max_len,
                num_proc=args.map_num_proc,
                desc=f"tokenize-{args.dataset}-test",
            )

            tmp_train = _tmp_path(paths["train"])
            tmp_test = _tmp_path(paths["test"])
            if os.path.isdir(tmp_train):
                shutil.rmtree(tmp_train)
            if os.path.isdir(tmp_test):
                shutil.rmtree(tmp_test)
            train_ds.save_to_disk(tmp_train)
            test_tok.save_to_disk(tmp_test)
            if os.path.isdir(paths["train"]):
                shutil.rmtree(paths["train"])
            if os.path.isdir(paths["test"]):
                shutil.rmtree(paths["test"])
            shutil.move(tmp_train, paths["train"])
            shutil.move(tmp_test, paths["test"])

            if need_eval_items:
                eval_items = build_eval_items(args.dataset, test_raw)
                tmp_eval = _tmp_path(paths["eval"])
                save_jsonl(tmp_eval, eval_items)
                os.replace(tmp_eval, paths["eval"])

            write_json(
                paths["meta"],
                {
                    "dataset": args.dataset,
                    "tokenizer": tok_name,
                    "max_len": int(args.max_len),
                    "created_at": now_str(),
                    "train_cache": paths["train"],
                    "test_cache": paths["test"],
                },
            )
    finally:
        release_dir_lock(paths["lock"])

    train_ds = load_from_disk(paths["train"])
    test_tok = load_from_disk(paths["test"])
    eval_items = load_jsonl(paths["eval"]) if need_eval_items else None
    train_ds, subset_info = maybe_select_train_subset(train_ds, args.train_samples, args.subset_seed)
    return train_ds, test_tok, eval_items, subset_info


@torch.no_grad()
def batch_option_loglikelihood(model, tokenizer, prompt: str, options: List[str], device: torch.device) -> List[float]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_attention_mask=False)["input_ids"]
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_token_id]

    seqs = []
    labels = []
    max_len = 0
    for opt in options:
        opt_ids = tokenizer(" " + opt, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        ids = prompt_ids + opt_ids
        lab = ([-100] * len(prompt_ids)) + opt_ids
        seqs.append(ids)
        labels.append(lab)
        max_len = max(max_len, len(ids))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    bsz = len(seqs)
    input_ids = torch.full((bsz, max_len), int(pad_id), dtype=torch.long, device=device)
    attn = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
    label_t = torch.full((bsz, max_len), -100, dtype=torch.long, device=device)

    for i in range(bsz):
        n = len(seqs[i])
        input_ids[i, :n] = torch.tensor(seqs[i], dtype=torch.long, device=device)
        attn[i, :n] = 1
        label_t[i, :n] = torch.tensor(labels[i], dtype=torch.long, device=device)

    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits[:, :-1, :]
    tgt = input_ids[:, 1:]
    ll_tok = torch.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = label_t[:, 1:] != -100
    ll = (ll_tok * mask).sum(dim=1)
    return [float(x) for x in ll.detach().cpu().tolist()]


@torch.no_grad()
def eval_mc_accuracy(
    model,
    tokenizer,
    eval_items: List[dict],
    device: torch.device,
    max_samples: int = -1,
) -> float:
    if max_samples is not None and max_samples > 0:
        items = eval_items[: max_samples]
    else:
        items = eval_items

    n = 0
    correct = 0
    for ex in items:
        scores = batch_option_loglikelihood(model, tokenizer, ex["prompt"], ex["options"], device)
        pred = ex["options"][int(torch.tensor(scores).argmax().item())]
        correct += int(pred == ex["gold"])
        n += 1
    return correct / max(n, 1)


@torch.no_grad()
def eval_lm_loss(model, loader, device) -> float:
    model.eval()
    losses = []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(**batch)
        losses.append(float(out.loss.item()))
    return sum(losses) / max(len(losses), 1)


def save_lora_sidecar(model, out_dir: str, run_name: str, lora_cfg: dict):
    ensure_dir(out_dir)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
    sft_path = os.path.join(out_dir, f"{run_name}.safetensors")
    save_safetensors(sd, sft_path)
    cfg_path = os.path.join(out_dir, f"{run_name}_config.json")
    write_json(cfg_path, lora_cfg)
    return sft_path, cfg_path


def build_loader(ds, batch_size: int, shuffle: bool, collator, num_workers: int, prefetch_factor: int):
    loader_kwargs = {
        "dataset": ds,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collator,
        "num_workers": max(int(num_workers), 0),
        "pin_memory": bool(torch.cuda.is_available()),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        if prefetch_factor and prefetch_factor > 0:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**loader_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument(
        "--dataset",
        type=str,
        default="arc-easy",
        choices=["arc-easy", "arc-challenge", "boolq", "gsm8k", "mbpp"],
    )
    parser.add_argument("--out_dir", type=str, default="./runs_lora")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")

    parser.add_argument("--eval_bs", type=int, default=2)
    parser.add_argument("--results_csv", type=str, default="./results.csv")

    parser.add_argument("--hf_cache_dir", type=str, default="")
    parser.add_argument("--data_cache_dir", type=str, default="./dataset_cache")
    parser.add_argument("--disable_data_cache", action="store_true")
    parser.add_argument("--cache_lock_timeout", type=int, default=7200)
    parser.add_argument("--map_num_proc", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--eval_acc_mode", type=str, choices=["full", "subset", "none"], default="full")
    parser.add_argument("--eval_acc_samples", type=int, default=512)
    parser.add_argument("--train-samples", type=int, default=-1, help="If > 0, train on a deterministic random subset of the train split.")
    parser.add_argument("--subset-seed", type=int, default=42, help="Seed used only for train subset selection.")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"lora_{args.dataset}_{args.base_model.split('/')[-1]}_{now_str()}"

    ensure_dir(args.out_dir)
    run_dir = os.path.join(args.out_dir, run_name)
    ensure_dir(run_dir)

    cache_dir = args.hf_cache_dir or None
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=model_dtype,
            cache_dir=cache_dir,
        ).to(device)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=model_dtype,
            cache_dir=cache_dir,
        ).to(device)
    base.train()

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    lora_cfg = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "base_model": args.base_model,
        "dataset": args.dataset,
    }
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(base, peft_cfg)
    model.train()

    train_ds, test_tok, eval_items, subset_info = load_or_build_prepared_data(args, tok)
    subset_meta_path = os.path.join(run_dir, "train_subset.json")
    write_json(
        subset_meta_path,
        {
            "dataset": args.dataset,
            "train_samples_requested": int(subset_info["train_samples_requested"]),
            "train_samples_effective": int(subset_info["train_samples_effective"]),
            "train_dataset_size": int(subset_info["train_dataset_size"]),
            "subset_seed": int(subset_info["subset_seed"]),
            "subset_hash": str(subset_info["subset_hash"]),
            "selected_indices": list(subset_info["selected_indices"]),
        },
    )
    print(
        "[SUBSET]",
        f"dataset={args.dataset}",
        f"train_samples={subset_info['train_samples_effective']}",
        f"train_dataset_size={subset_info['train_dataset_size']}",
        f"subset_seed={subset_info['subset_seed']}",
        f"subset_hash={subset_info['subset_hash']}",
        flush=True,
    )

    collator = SFTDataCollator(tok, pad_to_multiple_of=8 if torch.cuda.is_available() else None)
    train_loader = build_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collator=collator,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    test_loader = build_loader(
        test_tok,
        batch_size=args.eval_bs,
        shuffle=False,
        collator=collator,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * math.ceil(len(train_loader) / args.grad_accum)
    warmup_steps = int(total_steps * args.warmup_ratio)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)

    model.train()
    optim.zero_grad(set_to_none=True)
    optimizer_steps = 0
    num_train_batches = len(train_loader)
    if num_train_batches == 0:
        raise RuntimeError("Training loader is empty after subset selection.")

    for _ in range(args.epochs):
        accum_steps = 0
        for it, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            accum_steps += 1

            should_step = ((it + 1) % args.grad_accum == 0) or ((it + 1) == num_train_batches)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                optimizer_steps += 1
                accum_steps = 0

    if optimizer_steps == 0:
        raise RuntimeError(
            "No optimizer step was executed. "
            "This should not happen; check grad_accum and training subset size."
        )
    print(
        "[TRAIN]",
        f"epochs={args.epochs}",
        f"num_train_batches={num_train_batches}",
        f"grad_accum={args.grad_accum}",
        f"optimizer_steps={optimizer_steps}",
        flush=True,
    )

    model.eval()
    test_loss = eval_lm_loss(model, test_loader, device)
    test_ppl = ppl_from_loss(test_loss)

    test_acc = None
    if args.dataset in ["arc-easy", "arc-challenge", "boolq"] and args.eval_acc_mode != "none":
        max_samples = -1 if args.eval_acc_mode == "full" else int(args.eval_acc_samples)
        test_acc = eval_mc_accuracy(model, tok, eval_items, device, max_samples=max_samples)

    sft_path, cfg_path = save_lora_sidecar(model, run_dir, run_name, lora_cfg)

    metrics = {
        "run_name": run_name,
        "base_model": args.base_model,
        "dataset": args.dataset,
        "test_loss": float(test_loss),
        "test_ppl": float(test_ppl),
        "test_acc": (None if test_acc is None else float(test_acc)),
        "safetensors_path": sft_path,
        "sidecar_config": cfg_path,
        "time": now_str(),
        "eval_acc_mode": args.eval_acc_mode,
        "eval_acc_samples": int(args.eval_acc_samples),
    }
    metrics.update(
        {
            "lr": float(args.lr),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "grad_accum": int(args.grad_accum),
            "max_len": int(args.max_len),
            "warmup_ratio": float(args.warmup_ratio),
            "weight_decay": float(args.weight_decay),
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "target_modules": ",".join(target_modules),
            "seed": int(args.seed),
            "num_workers": int(args.num_workers),
            "map_num_proc": int(args.map_num_proc),
            "data_cache_dir": args.data_cache_dir,
            "train_samples": int(subset_info["train_samples_effective"]),
            "train_dataset_size": int(subset_info["train_dataset_size"]),
            "subset_seed": int(subset_info["subset_seed"]),
            "subset_hash": str(subset_info["subset_hash"]),
            "train_subset_meta": subset_meta_path,
            "optimizer_steps": int(optimizer_steps),
        }
    )
    m = re.match(r"trial_(\d+)", run_name)
    metrics["run_id"] = int(m.group(1)) if m else None
    write_json(os.path.join(run_dir, "metrics.json"), metrics)

    field_order = [
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
        "num_workers",
        "map_num_proc",
        "eval_acc_mode",
        "eval_acc_samples",
        "train_samples",
        "train_dataset_size",
        "subset_seed",
        "subset_hash",
        "optimizer_steps",
        "test_loss",
        "test_ppl",
        "test_acc",
        "safetensors_path",
        "sidecar_config",
    ]
    append_csv_row(args.results_csv, metrics, field_order=field_order)

    print("[DONE] saved:", run_dir)
    print("[TEST]", "loss=", test_loss, "ppl=", test_ppl, "acc=", test_acc)


if __name__ == "__main__":
    main()
