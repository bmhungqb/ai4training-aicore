# DiffGEBD for Sewing Step Segmentation

This directory vendors [JaejunHwang/DiffGEBD](https://github.com/JaejunHwang/DiffGEBD)
(*"DiffGEBD: Generic Event Boundary Detection via Denoising Diffusion"*, ICCV 2025,
[arXiv:2508.12084](https://arxiv.org/pdf/2508.12084)) and adapts it to train on the
industrial sewing "Step Segmentation" dataset built from `data/**/step_segments.json`.

The vendored code (`train.py`, `solver/`, `modeling/`, `datasets/`, `utils/`) is kept as
close as possible to upstream (see `README_upstream.md` for the original README) so that
future updates from the source repo can be diffed/merged easily. The sewing-specific
additions are:

- `datasets/dataset.py` / `datasets/__init__.py` — added a `SEWING` dataset name (alongside
  upstream `GEBD`/`TAPOS`) that reads annotations from
  `data/diff_gebd_dataset/{train,val}_annotation.pkl` and frames named `frame<N>.jpg`.
- `train.py` — added the matching `SEWING` branch when resolving the GT annotation path for
  evaluation, and made the `wandb` import optional (not required unless `--wandb` is passed).
- `config/sewing_diffgebd_resnet50.yaml` — training config for the sewing dataset
  (ResNet-50 backbone, DiffFormer/Transformer diffusion head, CFG).
- `tools/run_train.sh` — entrypoint for training on the sewing dataset (1 or N GPUs).

## 1. Environment

```bash
conda create --name DiffGEBD python=3.10
conda activate DiffGEBD
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r src/step_segment/DiffGEBD/requirements.txt
```

## 2. Dataset preparation

Build the offline frames + Kinetics-GEBD-style pickle annotations from
`data/**/step_segments.json`:

```bash
python tools/prepare_diff_gebd_dataset.py --split-mode random --val-ratio 0.2 --seed 42
```

This produces:

```
data/diff_gebd_dataset/
  images/{train,val}/<video_id>/frame<N>.jpg   # ROI-cropped/masked if *.mask.png exists
  train_annotation.pkl
  val_annotation.pkl
```

For a quick sanity check on 1-2 videos without processing the whole dataset:

```bash
python tools/prepare_diff_gebd_dataset.py --max-videos 2 --overwrite
```

## 3. Training

```bash
cd src/step_segment/DiffGEBD
bash tools/run_train.sh 1          # 1 GPU
bash tools/run_train.sh 2          # 2 GPUs (torchrun DDP)
```

Override any yacs config key on the command line, e.g.:

```bash
bash tools/run_train.sh 1 SOLVER.BATCH_SIZE 4 SOLVER.MAX_EPOCHS 50
```

## 4. Evaluation

```bash
cd src/step_segment/DiffGEBD
torchrun --nproc_per_node=1 train.py \
  --config-file output/sewing_diffgebd_resnet50_ann1_dim512_len100/config.yaml \
  --test-only --all-thres \
  --resume output/sewing_diffgebd_resnet50_ann1_dim512_len100/model_best.pth
```

Reports F1/Recall/Precision at relative-distance thresholds `[0.05, ..., 0.5]`
(see `utils/eval.py`), matching the GEBD convention.

## 5. Export predictions & benchmark vs DDM-Net / EfficientGEBD

```bash
cd ../../..   # repo root
python tools/export_diffgebd_predictions.py \
  --pred-pkl src/step_segment/DiffGEBD/output/sewing_diffgebd_resnet50_ann1_dim512_len100/model_pred_dict_ep-1.pkl \
  --split val --out-dir experiments/diffgebd_preds

python tools/benchmark_step_segment_models.py \
  --pred-a experiments/diffgebd_preds/diffgebd_preds.json --name-a DiffGEBD \
  --pred-b efficient_gebd_preds.json --name-b EfficientGEBD
```

See `docs/step_segment_diffgebd_training_guidance.md` for the full walkthrough and a
DDM-Net vs EfficientGEBD vs DiffGEBD comparison table.
