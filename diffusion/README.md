# W2T Vision

This directory contains the vision LoRA experiments in W2T.

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main workflows:

- `data_prepare/`: prepare CelebA and CUB instance folders and labels
- `data_generation/`: generate diffusion LoRAs
- `classification/`: train and evaluate W2T on diffusion LoRAs

CelebA:

```bash
python data_prepare/split_celeba_identities.py --img_dir /path/to/img_align_celeba --identity_file /path/to/identity_CelebA.txt --out_root ./data_prepare/outputs/celeba_identities
python data_prepare/build_celeba_labels.py --celeb_root ./data_prepare/outputs/celeba_identities --attr_csv /path/to/list_attr_celeba.csv --out_pt ./data_prepare/outputs/celeba_labels.pt --out_csv ./data_prepare/outputs/celeba_labels.csv
```

```bash
python data_generation/make_plan.py --celeb_root ./data_prepare/outputs/celeba_identities --out_dir ./data_generation/plans
python data_generation/run_plan.py --plan_csv ./data_generation/plans/plan.csv --runs_root ./data_generation/outputs/celeba/models_rank8_full --results_csv ./data_generation/outputs/celeba/results.csv
```

```bash
python data_prepare/make_lora_split.py --lora_root ./data_generation/outputs/celeba/models_rank8_full --out_dir ./data_prepare/outputs/splits_celeba_rank8_full
python classification/cache_celeba_loras.py --lora_root ./data_generation/outputs/celeba/models_rank8_full --labels_csv ./data_prepare/outputs/celeba_labels.csv --out_path ./classification/cache/celeba_rank8_full.pt
python classification/train_w2t_celeba.py --lora_root ./data_generation/outputs/celeba/models_rank8_full --labels_csv ./data_prepare/outputs/celeba_labels.csv --split_dir ./data_prepare/outputs/splits_celeba_rank8_full --cache_path ./classification/cache/celeba_rank8_full.pt --checkpoint_dir ./classification/checkpoints
```

CUB:

```bash
python data_prepare/split_cub_images.py --cub_root /path/to/CUB_200_2011 --out_root ./data_prepare/outputs/cub_instances
python data_prepare/build_cub_labels.py --cub_root /path/to/CUB_200_2011 --instance_root ./data_prepare/outputs/cub_instances --out_csv ./data_prepare/outputs/cub_labels.csv
```

```bash
python data_generation/make_cub_plan.py --instance_root ./data_prepare/outputs/cub_instances --out_dir ./data_generation/plans
python data_generation/run_plan.py --plan_csv ./data_generation/plans/cub_plan.csv --runs_root ./data_generation/outputs/cub/models_rank8_full --results_csv ./data_generation/outputs/cub/results.csv
```

```bash
python data_prepare/make_lora_split.py --lora_root ./data_generation/outputs/cub/models_rank8_full --out_dir ./data_prepare/outputs/splits_cub_rank8_full
python classification/cache_cub_loras.py --lora_root ./data_generation/outputs/cub/models_rank8_full --labels_csv ./data_prepare/outputs/cub_labels.csv --out_path ./classification/cache/cub_rank8_full.pt
python classification/train_w2t_cub.py --lora_root ./data_generation/outputs/cub/models_rank8_full --labels_csv ./data_prepare/outputs/cub_labels.csv --split_dir ./data_prepare/outputs/splits_cub_rank8_full --cache_path ./classification/cache/cub_rank8_full.pt --checkpoint_dir ./classification/checkpoints/cub
```

See the repository root README for the full workflow.
