# Step Segmentation Training Guidance (DiffGEBD)

This guide covers fine-tuning [DiffGEBD](https://github.com/JaejunHwang/DiffGEBD)
(*Generic Event Boundary Detection via Denoising Diffusion*, ICCV 2025,
[arXiv:2508.12084](https://arxiv.org/pdf/2508.12084)) to detect step/operation
boundaries in the sewing videos under `data/`.

## 1. What "step segmentation" means here

See `docs/step_segment_training_guidance.md` §1 — same ground-truth notion (boundaries
between consecutive **step segments**, produced by `tools/process_step_segments.py`).
DiffGEBD differs from DDM-Net / EfficientGEBD in *how* it predicts those boundaries: instead
of a deterministic per-frame binary classifier, it treats the 1D boundary sequence as the
target of a **denoising diffusion process** (DDPM/DDIM-style), conditioned on ResNet-50
spatio-temporal + self-similarity features, and uses **Classifier-Free Guidance (CFG)** at
sampling time to trade off fidelity vs. diversity of the predicted boundaries.

## 2. Environment setup (server)

```bash
conda create -n diffgebd python=3.10 -y
conda activate diffgebd
cd ai4training-aicore-poc
# install the correct torch/torchvision build for your CUDA version first, e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r src/step_segment/DiffGEBD/requirements.txt
```

## 3. Data processing

### Step 1 — merge consecutive same-operation clips into step segments

```bash
python tools/process_step_segments.py
```

(Skip if already run for DDM-Net/EfficientGEBD — writes the same `step_segments.json`.)

### Step 2 — build the DiffGEBD train/val dataset

```bash
# Random 80/20 split by video
python tools/prepare_diff_gebd_dataset.py --split-mode random --val-ratio 0.2 --seed 42

# Or hold out specific cd folders for validation
python tools/prepare_diff_gebd_dataset.py --split-mode by_folder --val-folders cd18,cd19,cd20

# Quick sanity check on 2 videos only
python tools/prepare_diff_gebd_dataset.py --max-videos 2 --overwrite
```

This writes, under `data/diff_gebd_dataset/`:
- `images/{train,val}/<video_id>/frame<N>.jpg` — offline-extracted frames (ROI-masked if a
  `*.mask.png` exists next to the source `.mp4`, same convention as `tools/mask_editor`).
- `train_annotation.pkl`, `val_annotation.pkl` — Kinetics-GEBD schema pickles (`fps`,
  `video_duration`, `substages_myframeidx`, `substages_timestamps`, single annotator).

Re-run Step 2 any time you change the split; pass `--overwrite` to re-extract frames.

## 4. Launch training

Default config: `src/step_segment/DiffGEBD/config/sewing_diffgebd_resnet50.yaml`
(ResNet-50 backbone, 224px input, 100-frame end-to-end sequences, DiffFormer/Transformer
diffusion head, `DIFFUSION.TIMESTEPS=1000`, `DIFFUSION.SAMPLING_TIMESTEPS=16`,
`DIFFUSION.CFG_SCALE=7.0`, batch size 2, AMP enabled). Run everything from the repo root or
`src/step_segment/DiffGEBD/` (paths in the config are relative to the latter).

```bash
cd src/step_segment/DiffGEBD

# Single GPU
bash tools/run_train.sh 1

# Multi-GPU (DDP) — also flip MODEL.SYNC_BN back to true when using >1 GPU
bash tools/run_train.sh 2 MODEL.SYNC_BN true

# Or call train.py directly to override any config value via CLI
torchrun --nproc_per_node=1 train.py \
  --config-file config/sewing_diffgebd_resnet50.yaml \
  SOLVER.BATCH_SIZE 4 SOLVER.MAX_EPOCHS 50
```

Run in the background on a server with `tmux`/`nohup`:

```bash
tmux new -s diffgebd_train
conda activate diffgebd
cd src/step_segment/DiffGEBD
bash tools/run_train.sh 1 2>&1 | tee train.log
# detach: Ctrl-b d, reattach later: tmux attach -t diffgebd_train
```

### Monitoring

- Console/`train.log`: per-step loss, per-epoch `Rec/Prec/F1@0.05` table (all relative
  distance thresholds with `--all-thres`).
- Checkpoints + per-epoch predictions are written under
  `output/sewing_diffgebd_resnet50_ann1_dim512_len100/`:
  `model_best.pth`, `metrics.txt`, `model_pred_dict_ep<N>.pkl`.

## 5. Validation / inference & benchmarking

```bash
cd src/step_segment/DiffGEBD
torchrun --nproc_per_node=1 train.py \
  --config-file output/sewing_diffgebd_resnet50_ann1_dim512_len100/config.yaml \
  --test-only --all-thres \
  --resume output/sewing_diffgebd_resnet50_ann1_dim512_len100/model_best.pth
```

Convert the resulting `model_pred_dict_ep*.pkl` into project-standard outputs:

```bash
cd ../../..   # repo root
python tools/export_diffgebd_predictions.py \
  --pred-pkl src/step_segment/DiffGEBD/output/sewing_diffgebd_resnet50_ann1_dim512_len100/model_pred_dict_ep-1.pkl \
  --split val \
  --out-dir experiments/diffgebd_preds
```

This writes `experiments/diffgebd_preds/diffgebd_preds.json` (`{video_id: [boundary_ts]}`)
and one `step_segments_pred.json` per video. Compare directly against DDM-Net /
EfficientGEBD predictions (same JSON format) with:

```bash
python tools/benchmark_step_segment_models.py \
  --pred-a experiments/diffgebd_preds/diffgebd_preds.json --name-a DiffGEBD \
  --pred-b efficient_gebd_preds.json --name-b EfficientGEBD \
  --thresholds 0.05 0.1 0.2 0.3 0.4 0.5 \
  --out experiments/stage1_boundary_recall_9cd/diffgebd_vs_efficientgebd.json
```

## 6. Comparison table: DDM-Net vs EfficientGEBD vs DiffGEBD

| | DDM-Net | EfficientGEBD | DiffGEBD |
|---|---|---|---|
| Paradigm | Deterministic binary classifier (per-frame) | Deterministic binary classifier + FPN/DiffFormer heads | Denoising diffusion generative model + Classifier-Free Guidance |
| Backbone | ResNet-50 / DINOv2 (frozen or fine-tuned) | ResNet-50 or CSN (R50/R152) | ResNet-50 |
| Sequence length | configurable window around candidate boundary | 100 frames end-to-end | 100 frames end-to-end |
| Sampling cost | single forward pass | single forward pass | `DIFFUSION.SAMPLING_TIMESTEPS` forward passes (16 by default) — slower inference |
| Extra knobs | `frames_per_side`, `min_change_dur`, EMA | FPN `HEAD_CHOICE`, similarity groups | `CFG_SCALE`, `TIMESTEPS`, `SAMPLING_TIMESTEPS`, `BETA_SCHEDULE` |
| Config | `src/step_segment/DDM-Net/config/ddm_train_config.yaml` | `src/step_segment/EfficientGEBD/config-files/sewing_csn.yaml` | `src/step_segment/DiffGEBD/config/sewing_diffgebd_resnet50.yaml` |
| Train entrypoint | `train_sop_lightning.py` | `train.py` | `train.py` |
| Eval metric | Boundary F1/Rec/Prec (project-defined thresholds) | GEBD Rel-distance F1/Rec/Prec | GEBD Rel-distance F1/Rec/Prec (0.05–0.5) |

Fill in the resource-usage columns (VRAM, train speed, inference fps) once each model has
been trained on this hardware, then feed them into `tools/benchmark_step_segment_models.py`
for a full side-by-side report alongside `experiments/stage1_boundary_recall_9cd`.

## 7. Key parameters to tune for this dataset

- `DIFFUSION.CFG_SCALE` — higher values sharpen boundary predictions (less diverse, more
  faithful to the training distribution); the paper defaults to 4.0–7.0. Start at 7.0 (our
  config default) and sweep down if boundaries look over-confident/spurious.
- `DIFFUSION.SAMPLING_TIMESTEPS` — DDIM steps at inference; higher = better quality but
  slower. 16 is a reasonable balance for a 33-video dataset; try 8 or 32 during ablation.
- `INPUT.SEQUENCE_LENGTH` — frames sampled per video (uniformly, padded if shorter). Sewing
  clips are longer (dozens of seconds) than Kinetics-GEBD's ~10s clips; 100 is the default,
  raise it if step boundaries are being missed between sampled frames.
- `SOLVER.BATCH_SIZE` / `SOLVER.AMPE` — kept small (2) + AMP on to fit end-to-end
  100-frame×224px×ResNet-50 training on a single consumer GPU.
- `MODEL.SYNC_BN` — must stay `false` for single-GPU (`torchrun --nproc_per_node=1`) runs,
  since `SyncBatchNorm` requires an initialized `torch.distributed` process group; flip to
  `true` only for multi-GPU DDP training.
