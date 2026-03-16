import os
import re
import json
import math
import argparse
import random
import shutil
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

import torch
import torch.multiprocessing as mp

from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)

from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import snapshot_download


# ======================================================================================
# NOTE
# This script is a multi-label (multi-hot) dataset generator for LoRA weights.
# Each *sample* becomes one LoRA adapter checkpoint.
#
# Output structure (resumable):
#   output_dir/
#     <dataset_name>/
#       <base_model_slug>/
#         labels.json
#         lora_<dataset>_<model>_<run_id>.safetensors
#         lora_<dataset>_<model>_<run_id>_config.json   (optional)
#         metadata_gpu_<gpu>.csv / .jsonl
#
# The resume logic matches the existing collect_lora_dataset.py style: if the
# target .safetensors exists, the run is skipped.
# ======================================================================================


_model_dir_cache = {}


def get_local_model_dir(model_id: str) -> str:
    if model_id in _model_dir_cache:
        return _model_dir_cache[model_id]
    local_dir = snapshot_download(
        repo_id=model_id,
        token=os.environ.get("HF_TOKEN"),
    )
    _model_dir_cache[model_id] = local_dir
    return local_dir


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def getenv(name: str, default=None):
    return os.environ.get(name, default)


def stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]


# ======================================================================================
# Hyperparameter sampling (narrower + stable for single-sample SFT)
# ======================================================================================


@dataclass
class HyperparamConfig:
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    learning_rate: float
    weight_decay: float
    num_epochs: int
    warmup_ratio: float
    lr_scheduler_type: str
    per_device_bs: int
    gradient_accumulation_steps: int
    max_length: int
    seed: int


def sample_hyperparams(
    seed: int,
    per_device_bs: int,
    target_bs: int,
    max_length: int,
    enable_sampling: bool = True,
    fixed_rank: int = 8,
) -> HyperparamConfig:
    rng = random.Random(seed)

    # Fixed rank per your spec
    lora_r = fixed_rank
    lora_alpha = 2 * fixed_rank

    if not enable_sampling:
        return HyperparamConfig(
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            learning_rate=2e-4,
            weight_decay=0.01,
            num_epochs=3,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            per_device_bs=per_device_bs,
            gradient_accumulation_steps=max(1, target_bs // per_device_bs),
            max_length=max_length,
            seed=seed,
        )

    # Single-sample training is extremely sensitive; keep ranges narrow.
    lora_dropout = rng.uniform(0.0, 0.15)
    target_modules = rng.choice([["q_proj", "v_proj"]])

    # lr in [3e-5, 3e-4]
    learning_rate = 10 ** rng.uniform(math.log10(3e-5), math.log10(3e-4))
    weight_decay = 10 ** rng.uniform(math.log10(1e-6), math.log10(1e-3))
    num_epochs = rng.choice([1, 2, 3, 4])
    warmup_ratio = rng.uniform(0.0, 0.2)
    lr_scheduler_type = rng.choice(["linear", "cosine", "constant"])
    grad_accum = max(1, target_bs // per_device_bs)

    return HyperparamConfig(
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        per_device_bs=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        seed=seed,
    )


# ======================================================================================
# Training metrics callback
# ======================================================================================


class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.learning_rates = []
        self.grad_norms = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
            if "learning_rate" in logs:
                self.learning_rates.append(logs["learning_rate"])
            if "grad_norm" in logs:
                self.grad_norms.append(logs["grad_norm"])


# ======================================================================================
# Prompt / label rendering
# ======================================================================================


def render_multilabel_target(label_names: List[str], y: List[int]) -> str:
    # Stable, explicit representation; easy to parse downstream.
    # One line per label.
    lines = [f"{name}={int(v)}" for name, v in zip(label_names, y)]
    return "\n".join(lines) + "\n"


def format_causal_pairs(tokenizer, inputs: List[str], targets: List[str], max_length: int):
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True)
    labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True)["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs


# ======================================================================================
# Dataset builders
# ======================================================================================


def build_goemotions(n_samples: int, seed: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    """GoEmotions multi-label dataset.

    We rely on a HF-hosted copy that exposes one-hot columns for each emotion.
    """
    ds = load_dataset("SetFit/go_emotions", split="train")

    # Identify label columns: dataset contains "text" plus many 0/1 columns.
    ignore = {"text", "id"}
    label_names = [c for c in ds.column_names if c not in ignore]
    # Keep deterministic ordering
    label_names = sorted(label_names)

    # Shuffle + take subset
    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(min(n_samples, len(ds))))

    records = []
    for ex in ds:
        y = [int(ex.get(lbl, 0)) for lbl in label_names]
        records.append({"text": ex["text"], "y": y})
    return records, label_names


def _extract_absa_labels_from_example(ex: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return list of (category, polarity) pairs from various ABSA dataset schemas."""
    pairs = []

    # Common keys seen on HF for SemEval ABSA variants
    # - "categories": list of dicts {"category":..., "polarity":...}
    # - "aspects": list of dicts {"term"/"category", "polarity"}
    # - "labels": list of dicts
    for key in ["categories", "aspects", "labels", "opinions"]:
        if key in ex and isinstance(ex[key], list):
            for item in ex[key]:
                if not isinstance(item, dict):
                    continue
                cat = item.get("category") or item.get("aspect") or item.get("target")
                pol = item.get("polarity") or item.get("sentiment")
                if cat is None:
                    continue
                if pol is None:
                    pol = "unknown"
                pairs.append((str(cat).strip(), str(pol).strip()))

    # Another common schema: "aspect_category" and "sentiment" per row
    if "aspect_category" in ex:
        cat = ex.get("aspect_category")
        pol = ex.get("sentiment") or ex.get("polarity") or "unknown"
        if cat is not None:
            pairs.append((str(cat).strip(), str(pol).strip()))

    # De-dup
    pairs = list({(c, p) for c, p in pairs if c})
    return pairs


def build_absa_english(n_samples: int, seed: int):
    """
    Robust ABSA builder:
    - Supports multiple HF ABSA schemas.
    - Extracts text from many possible keys.
    - Extracts (category, polarity) via _extract_absa_labels_from_example, and
      also parses common string label formats.
    - If still 0, prints schema debug and raises.
    """
    from datasets import load_dataset, DatasetDict, concatenate_datasets
    import random
    import re

    rng = random.Random(seed)
    records = []
    debug_infos = []

    # Prefer "data-file/parquet" style repos (avoid dataset scripts)
    sources = [
        ("jakartaresearch/semeval-absa", None),
        ("psimm/absa-semeval2014-alpaca", None),
    ]

    # helper: load split(s) robustly
    def load_any_splits(name: str, subset: str | None):
        try:
            if subset is None:
                obj = load_dataset(name)
            else:
                obj = load_dataset(name, subset)
        except Exception as e:
            debug_infos.append((name, subset, f"LOAD_FAIL: {type(e).__name__}: {e}"))
            return None

        if isinstance(obj, DatasetDict):
            # Prefer train if exists, else concatenate all splits
            if "train" in obj:
                return obj["train"]
            else:
                return concatenate_datasets([obj[k] for k in obj.keys()])
        else:
            # already a Dataset (rare)
            return obj

    # helper: pick text from many keys
    def get_text(ex):
        for k in ["text", "sentence", "review", "content", "document", "doc", "input", "prompt"]:
            v = ex.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # Alpaca-ish: instruction + input
        inst = ex.get("instruction")
        inp = ex.get("input")
        if isinstance(inst, str) and inst.strip():
            if isinstance(inp, str) and inp.strip():
                return (inst.strip() + "\n" + inp.strip()).strip()
            return inst.strip()
        return None

    # helper: parse labels from common string formats
    # e.g. "FOOD#QUALITY=positive", "SERVICE#GENERAL: negative", etc.
    label_pat = re.compile(r"([A-Za-z0-9_#/\- ]+)\s*(?:=|:)\s*(positive|negative|neutral|conflict|unknown)", re.I)

    for name, subset in sources:
        ds = load_any_splits(name, subset)
        if ds is None:
            continue

        # light debug info
        try:
            debug_infos.append((name, subset, f"cols={ds.column_names[:]} len={len(ds)} ex0={ {k: str(ds[0].get(k))[:80] for k in ds.column_names[:6]} }"))
        except Exception:
            pass

        for ex in ds:
            text = get_text(ex)
            if not text:
                continue

            # 1) structured extraction (most stable)
            pairs = _extract_absa_labels_from_example(ex)  # returns list[(cat, pol)]
            labels = []
            for cat, pol in pairs:
                pol_l = str(pol).strip().lower()
                if pol_l == "conflict":
                    continue
                labels.append(f"{cat}={pol_l}")

            # 2) common flat schema: aspect_category + sentiment/polarity/label
            if not labels:
                cat = ex.get("aspect_category") or ex.get("category") or ex.get("aspect")
                pol = ex.get("sentiment") or ex.get("polarity")
                if isinstance(cat, str) and cat.strip():
                    pol_l = str(pol).strip().lower() if pol is not None else "unknown"
                    if pol_l != "conflict":
                        labels.append(f"{cat.strip()}={pol_l}")

            # 3) string label field schema (many datasets use this)
            if not labels:
                for k in ["label", "labels", "output", "target"]:
                    v = ex.get(k)
                    if isinstance(v, str) and v.strip():
                        hits = label_pat.findall(v)
                        for cat, pol in hits:
                            pol_l = pol.strip().lower()
                            if pol_l == "conflict":
                                continue
                            labels.append(f"{cat.strip()}={pol_l}")
                        if labels:
                            break

            if labels:
                # de-dup per example
                labels = sorted(set(labels))
                records.append({"text": text, "labels": labels})

    rng.shuffle(records)
    if records:
        records = records[: min(n_samples, len(records))]

    if not records:
        # Print actionable debug so you instantly see which fields are present
        print("[ABSA DEBUG] No usable samples. Dataset schemas seen:")
        for info in debug_infos[:10]:
            print("  ", info)
        raise RuntimeError("ABSA builder produced 0 usable samples. Check above ABSA DEBUG output.")

    label_names = sorted({l for r in records for l in r["labels"]})

    # IMPORTANT: return in the same format as other builders expect: {"text":..., "y":[...]}
    out = []
    label_index = {name: i for i, name in enumerate(label_names)}
    L = len(label_names)
    for r in records:
        y = [0] * L
        for lab in r["labels"]:
            y[label_index[lab]] = 1
        out.append({"text": r["text"], "y": y})

    return out, label_names



def build_eurlex(n_samples: int, seed: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    """LexGLUE / EURLEX: real legal text with multi-label topics."""
    ds = load_dataset("lex_glue", "eurlex", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))

    # Determine label space (Sequence(ClassLabel)) if available
    label_names: Optional[List[str]] = None
    try:
        if "labels" in ds.features and hasattr(ds.features["labels"], "feature") and hasattr(ds.features["labels"].feature, "names"):
            label_names = list(ds.features["labels"].feature.names)
    except Exception:
        label_names = None

    # Fallback: infer max label id from subset
    if label_names is None:
        max_id = -1
        for ex in ds:
            labs = ex.get("labels")
            if isinstance(labs, (list, tuple)) and labs:
                if isinstance(labs[0], int):
                    max_id = max(max_id, max(labs))
                else:
                    max_id = max(max_id, len(labs) - 1)
        label_names = [f"label_{i}" for i in range(max_id + 1)]

    L = len(label_names)

    records: List[Dict[str, Any]] = []
    for ex in ds:
        text = ex.get("text") or ex.get("sentence") or ex.get("doc") or ex.get("content")
        labs = ex.get("labels")
        if not text or labs is None:
            continue

        # Normalize to positive indices
        if isinstance(labs, (list, tuple)):
            if len(labs) > 0 and isinstance(labs[0], int):
                pos = list(labs)
            else:
                pos = [i for i, v in enumerate(labs) if int(v) == 1]
        else:
            pos = [int(labs)]

        y = [0] * L
        for i in pos:
            ii = int(i)
            if 0 <= ii < L:
                y[ii] = 1

        records.append({"text": text, "y": y})

    if not records:
        raise RuntimeError("EURLEX builder produced 0 usable samples (unexpected).")

    return records, label_names

DATASET_BUILDERS = {
    "goemotions": build_goemotions,
    "absa_en": build_absa_english,
    "eurlex": build_eurlex,
    }


def build_records(dataset_name: str, n_samples: int, seed: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    if dataset_name not in DATASET_BUILDERS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_BUILDERS)}")
    return DATASET_BUILDERS[dataset_name](n_samples=n_samples, seed=seed)


def write_label_names(output_root: str, dataset_name: str, base_model: str, label_names: List[str]):
    base_slug = slugify(base_model.split("/")[-1])
    dest_dir = os.path.join(output_root, dataset_name, base_slug)
    ensure_dir(dest_dir)
    p = os.path.join(dest_dir, "labels.json")
    if os.path.exists(p):
        return
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset_name, "base_model": base_model, "label_names": label_names}, f, ensure_ascii=False, indent=2)


def build_splits_for_single_sample(
    tokenizer,
    record: Dict[str, Any],
    label_names: List[str],
    eval_pool: List[Dict[str, Any]],
    eval_size: int,
    max_length: int,
    seed: int,
) -> Dict[str, Dataset]:

    def make_pair(ex: Dict[str, Any]) -> Tuple[str, str]:
        text = ex["text"]
        y = ex["y"]
        inp = f"Text: {text}\n\nReturn a multi-label 0/1 assignment for each label.\nLabels:\n"
        tgt = render_multilabel_target(label_names, y)
        return inp, tgt

    # Train set: exactly one sample
    tr_inp, tr_tgt = make_pair(record)
    train_ds = Dataset.from_dict({"input": [tr_inp], "target": [tr_tgt]})

    # Validation/Test: deterministic subset from eval_pool
    rng = random.Random(seed)
    pool = list(eval_pool)
    rng.shuffle(pool)
    take = min(eval_size, len(pool))
    half = take // 2
    val_pool = pool[:half]
    test_pool = pool[half:take]

    def to_ds(pool_ex: List[Dict[str, Any]]) -> Dataset:
        inputs, targets = [], []
        for ex in pool_ex:
            i, t = make_pair(ex)
            inputs.append(i)
            targets.append(t)
        return Dataset.from_dict({"input": inputs, "target": targets})

    val_ds = to_ds(val_pool)
    test_ds = to_ds(test_pool)

    def preprocess(batch):
        return format_causal_pairs(tokenizer, batch["input"], batch["target"], max_length)

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)
    test_tok = test_ds.map(preprocess, batched=True, remove_columns=test_ds.column_names)

    return {"train": train_tok, "validation": val_tok, "test": test_tok}


# ======================================================================================
# Single LoRA training (resumable)
# ======================================================================================


def train_single_lora(
    base_model: str,
    dataset_name: str,
    output_root: str,
    run_id: int,
    hyperparams: HyperparamConfig,
    keep_config: bool,
    record: Dict[str, Any],
    label_names: List[str],
    eval_pool: List[Dict[str, Any]],
    eval_size: int,
) -> Optional[Dict[str, Any]]:

    hf_token = getenv("HF_TOKEN")

    base_slug = slugify(base_model.split("/")[-1])
    dest_dir = os.path.join(output_root, dataset_name, base_slug)
    dest_sft = os.path.join(dest_dir, f"lora_{dataset_name}_{base_slug}_{run_id}.safetensors")

    # === Resume ===
    if os.path.exists(dest_sft):
        print(f"[SKIP] Run {run_id} exists at {dest_sft}, skipping.")
        return None

    # Seed
    torch.manual_seed(hyperparams.seed)
    random.seed(hyperparams.seed)

    local_dir = get_local_model_dir(base_model)
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=hf_token,
    )

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=hyperparams.lora_r,
        lora_alpha=hyperparams.lora_alpha,
        lora_dropout=hyperparams.lora_dropout,
        target_modules=hyperparams.target_modules,
    )
    model = get_peft_model(model, peft_cfg)

    # Build datasets
    data_bundle = build_splits_for_single_sample(
        tokenizer=tokenizer,
        record=record,
        label_names=label_names,
        eval_pool=eval_pool,
        eval_size=eval_size,
        max_length=hyperparams.max_length,
        seed=hyperparams.seed,
    )

    tmp_run_dir = os.path.join(output_root, "_tmp_runs", f"{dataset_name}_{base_slug}", f"run_{run_id}_{hyperparams.seed}")
    ensure_dir(tmp_run_dir)

    args = TrainingArguments(
        output_dir=tmp_run_dir,
        learning_rate=hyperparams.learning_rate,
        weight_decay=hyperparams.weight_decay,
        num_train_epochs=hyperparams.num_epochs,
        per_device_train_batch_size=hyperparams.per_device_bs,
        per_device_eval_batch_size=hyperparams.per_device_bs,
        gradient_accumulation_steps=hyperparams.gradient_accumulation_steps,
        warmup_ratio=hyperparams.warmup_ratio,
        lr_scheduler_type=hyperparams.lr_scheduler_type,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=10,
        report_to="none",
        fp16=False,
        bf16=True,
        save_safetensors=True,
        disable_tqdm=True,
        seed=hyperparams.seed,
    )

    metrics_cb = MetricsCallback()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data_bundle["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[metrics_cb],
    )

    trainer.train()

    # Optional eval (still useful as scalar "quality" label if you need it)
    val_metrics = trainer.evaluate(eval_dataset=data_bundle["validation"])
    test_metrics = trainer.evaluate(eval_dataset=data_bundle["test"])
    val_loss = float(val_metrics.get("eval_loss", float("nan")))
    test_loss = float(test_metrics.get("eval_loss", float("nan")))

    adapter_dir = os.path.join(tmp_run_dir, "final_adapter")
    model.save_pretrained(adapter_dir, safe_serialization=True)

    src_sft = os.path.join(adapter_dir, "adapter_model.safetensors")
    ensure_dir(dest_dir)
    shutil.copy2(src_sft, dest_sft)

    if keep_config:
        cfg_src = os.path.join(adapter_dir, "adapter_config.json")
        if os.path.exists(cfg_src):
            cfg_dest = dest_sft.replace(".safetensors", "_config.json")
            shutil.copy2(cfg_src, cfg_dest)

    shutil.rmtree(tmp_run_dir, ignore_errors=True)

    # Metadata record
    text = record["text"]
    y = record["y"]
    rec = {
        "run_id": run_id,
        "dataset": dataset_name,
        "base_model": base_model,
        "safetensors_path": dest_sft,
        "text_sha1": stable_hash(text),
        "label_vector": y,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "final_train_loss": metrics_cb.train_losses[-1] if metrics_cb.train_losses else None,
        "avg_train_loss": sum(metrics_cb.train_losses) / len(metrics_cb.train_losses) if metrics_cb.train_losses else None,
        "final_lr": metrics_cb.learning_rates[-1] if metrics_cb.learning_rates else None,
        "avg_grad_norm": sum(metrics_cb.grad_norms) / len(metrics_cb.grad_norms) if metrics_cb.grad_norms else None,
        "max_grad_norm": max(metrics_cb.grad_norms) if metrics_cb.grad_norms else None,
        "optimizer_steps": trainer.state.global_step,
        "effective_batch_size": hyperparams.per_device_bs * hyperparams.gradient_accumulation_steps,
        "effective_tokens": trainer.state.global_step * hyperparams.per_device_bs * hyperparams.gradient_accumulation_steps * hyperparams.max_length,
        **asdict(hyperparams),
    }
    return rec


# ======================================================================================
# GPU worker
# ======================================================================================


def gpu_worker(
    gpu_id: int,
    task_queue: List[Tuple[int, str, str, HyperparamConfig, Dict[str, Any], List[str], List[Dict[str, Any]], int]],
    output_root: str,
    keep_config: bool,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("WANDB_MODE", "disabled")

    worker_csv = os.path.join(output_root, f"metadata_gpu_{gpu_id}.csv")
    worker_jsonl = os.path.join(output_root, f"metadata_gpu_{gpu_id}.jsonl")

    print(f"[GPU {gpu_id}] Assigned {len(task_queue)} tasks.")

    for run_id, base_model, dataset_name, hparams, record, label_names, eval_pool, eval_size in task_queue:
        try:
            print(f"[GPU {gpu_id}] Starting run {run_id}: {dataset_name} + {base_model}")

            write_label_names(output_root, dataset_name, base_model, label_names)

            rec = train_single_lora(
                base_model=base_model,
                dataset_name=dataset_name,
                output_root=output_root,
                run_id=run_id,
                hyperparams=hparams,
                keep_config=keep_config,
                record=record,
                label_names=label_names,
                eval_pool=eval_pool,
                eval_size=eval_size,
            )

            if rec is None:
                continue

            rec["gpu_id"] = gpu_id

            df = pd.DataFrame([rec])
            write_header = not os.path.exists(worker_csv)
            df.to_csv(worker_csv, mode="a", header=write_header, index=False)

            with open(worker_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            torch.cuda.empty_cache()
            print(f"[GPU {gpu_id}] Completed run {run_id}")

        except Exception as e:
            print(f"[GPU {gpu_id}] Run {run_id} failed: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"[GPU {gpu_id}] Finished all assigned tasks")


# ======================================================================================
# Task generation (resumable, per-sample)
# ======================================================================================


def generate_task_list(
    datasets: List[str],
    base_models: List[str],
    samples_per_dataset: Dict[str, int],
    samples_per_model: Dict[str, int],
    per_device_bs: int,
    target_bs: int,
    max_length: int,
    enable_sampling: bool,
    global_seed: int,
    output_dir: str,
    fixed_rank: int,
    eval_size: int,
) -> List[Tuple[int, str, str, HyperparamConfig, Dict[str, Any], List[str], List[Dict[str, Any]], int]]:

    tasks = []
    rng = random.Random(global_seed)

    # Preload records per dataset once (memory ok for 8k)
    dataset_cache: Dict[str, Tuple[List[Dict[str, Any]], List[str]]] = {}
    for ds_name in datasets:
        n = samples_per_dataset.get(ds_name, 0)
        recs, label_names = build_records(ds_name, n_samples=n, seed=global_seed)
        dataset_cache[ds_name] = (recs, label_names)

    for dataset_name in datasets:
        records, label_names = dataset_cache[dataset_name]

        for base_model in base_models:
            # how many from this dataset for this model
            n_samples = min(
                samples_per_dataset.get(dataset_name, float("inf")),
                samples_per_model.get(base_model, float("inf")),
                len(records),
            )

            base_slug = slugify(base_model.split("/")[-1])
            dest_dir = os.path.join(output_dir, dataset_name, base_slug)
            ensure_dir(dest_dir)

            print(f"Plan: {dataset_name} / {base_model} -> {n_samples} samples (per-sample LoRA)")

            skipped_count = 0
            for run_id in range(n_samples):
                seed = global_seed + run_id
                expected_file = os.path.join(dest_dir, f"lora_{dataset_name}_{base_slug}_{run_id}.safetensors")
                if os.path.exists(expected_file):
                    skipped_count += 1
                    continue

                hparams = sample_hyperparams(
                    seed=seed,
                    per_device_bs=per_device_bs,
                    target_bs=target_bs,
                    max_length=max_length,
                    enable_sampling=enable_sampling,
                    fixed_rank=fixed_rank,
                )

                # Eval pool excludes the training record itself
                eval_pool = records[:run_id] + records[run_id + 1 :]
                tasks.append((run_id, base_model, dataset_name, hparams, records[run_id], label_names, eval_pool, eval_size))

            if skipped_count:
                print(f"[SKIP] Skipped {skipped_count} existing files for {dataset_name}/{base_model}")

    rng.shuffle(tasks)
    return tasks


# ======================================================================================
# Main
# ======================================================================================


def parse_args():
    p = argparse.ArgumentParser("LoRA Multi-label (Per-sample) Dataset Generator")
    p.add_argument("--output_dir", type=str, default="./classification/outputs/goemotions_loras")
    p.add_argument("--datasets", type=str, default="goemotions")
    p.add_argument("--base_models", type=str, default="meta-llama/Llama-3.2-3B")

    # Sample counts
    p.add_argument("--samples_per_dataset", type=str, default=None)
    p.add_argument("--samples_per_model", type=str, default=None)
    p.add_argument("--total_samples", type=int, default=20000)

    # Hyperparameter sampling
    p.add_argument("--enable_sampling", action="store_true", default=True)
    p.add_argument("--global_seed", type=int, default=42)
    p.add_argument("--fixed_rank", type=int, default=8)

    # Training config
    p.add_argument("--per_device_bs", type=int, default=1)
    p.add_argument("--target_bs", type=int, default=8)
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--eval_size", type=int, default=256, help="#examples used for (val+test) loss label per LoRA")
    p.add_argument("--keep_config", action="store_true")

    # GPU config
    p.add_argument("--gpu_ids", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    os.environ.setdefault("WANDB_MODE", "disabled")

    datasets_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    base_models_list = [m.strip() for m in args.base_models.split(",") if m.strip()]

    # Parse per-dataset samples
    if args.samples_per_dataset:
        samples_per_dataset = {}
        for item in args.samples_per_dataset.split(","):
            k, v = item.split("=")
            samples_per_dataset[k.strip()] = int(v.strip())
    else:
        samples_per_dataset = {ds: args.total_samples for ds in datasets_list}

    # Parse per-model samples
    if args.samples_per_model:
        samples_per_model = {}
        for item in args.samples_per_model.split(","):
            k, v = item.split("=")
            samples_per_model[k.strip()] = int(v.strip())
    else:
        samples_per_model = {m: float("inf") for m in base_models_list}

    if args.gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]

    if not gpu_ids:
        raise RuntimeError("No GPUs available")

    print("[START] LoRA multi-label generator (per-sample, resumable)")
    print(f"[INFO] Datasets: {datasets_list}")
    print(f"[INFO] Base models: {base_models_list}")
    print(f"[INFO] Output: {args.output_dir}")
    print(f"[INFO] Fixed rank: {args.fixed_rank}")

    tasks = generate_task_list(
        datasets=datasets_list,
        base_models=base_models_list,
        samples_per_dataset=samples_per_dataset,
        samples_per_model=samples_per_model,
        per_device_bs=args.per_device_bs,
        target_bs=args.target_bs,
        max_length=args.max_length,
        enable_sampling=args.enable_sampling,
        global_seed=args.global_seed,
        output_dir=args.output_dir,
        fixed_rank=args.fixed_rank,
        eval_size=args.eval_size,
    )

    print(f"[INFO] Scheduled {len(tasks)} new tasks")

    # Distribute tasks
    num_gpus = len(gpu_ids)
    gpu_queues = [[] for _ in range(num_gpus)]
    for i, task in enumerate(tasks):
        gpu_queues[i % num_gpus].append(task)

    mp.set_start_method("spawn", force=True)
    procs = []
    for rank, gid in enumerate(gpu_ids):
        if not gpu_queues[rank]:
            continue
        p = mp.Process(target=gpu_worker, args=(gid, gpu_queues[rank], args.output_dir, args.keep_config))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("[DONE] All GPU workers completed.")


if __name__ == "__main__":
    main()
