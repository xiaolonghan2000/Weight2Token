# W2T LLM

This directory contains the language LoRA experiments in W2T.

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main workflows:

- `classification/`: GoEmotions attribute classification from per-sample LoRAs
- `regression/`: ARC-Easy performance prediction from full-train LoRAs
- `retrieval/`: gallery-query retrieval over ARC-Challenge, BoolQ, GSM8K, and MBPP LoRAs

Quick start:

```bash
python classification/collect_goemotions_loras.py --output_dir ./classification/outputs/goemotions_loras --gpu_ids 0
python classification/merge_metadata.py --input_dir ./classification/outputs/goemotions_loras --output_csv ./classification/lora_label_info.csv
python classification/train_w2t_classifier.py --labels_csv ./classification/lora_label_info.csv --checkpoint_dir ./classification/checkpoints
```

```bash
python regression/collect/make_plan.py --out-dir ./regression/plans
python regression/collect/run_plan.py --plan_dir ./regression/plans --runs_root ./outputs/regression/runs --results_csv ./outputs/regression/results_arc_easy.csv
python regression/perf_prediction_pipeline.py train --packed-cache-dir ./outputs/regression/cache_packed --output-dir ./outputs/regression/w2t_results --model-type w2t --target-col test_acc
```

```bash
python retrieval/collect/make_plan.py --out-dir ./retrieval/plans
python retrieval/collect/make_fewshot_plan.py --out-dir ./outputs/retrieval/fewshot_plans
python retrieval/fewshot_retrieval.py retrieve --manifest ./outputs/retrieval/cache/manifest.json --metadata-csv ./outputs/retrieval/prepared/all_metadata.csv --trained-root ./outputs/regression/w2t_results --output-dir ./outputs/retrieval/results --model-type w2t
```

See the repository root README for the full workflow.
