# EfficientGEBD for Sewing Step Segmentation

This directory vendors [Ziwei-Zheng/EfficientGEBD](https://github.com/Ziwei-Zheng/EfficientGEBD)
(*"Rethinking the Architecture Design for Efficient Generic Event Boundary
Detection"*, ACM MM 2024, [arXiv:2407.12622](https://arxiv.org/abs/2407.12622))
and adapts it to train on the industrial sewing "Step Segmentation" dataset
built from `data/**/step_segments.json`.

The vendored code (`train.py`, `post_process.py`, `solver/`, `modeling/`,
`datasets/`, `utils/`, `script/`) is kept as close as possible to upstream
(commit `7f6dd7b`, see `README_upstream.md` for the original README) so that
future updates from the source repo can be diffed/merged easily. The
sewing-specific additions are:

- `datasets/roi_mask.py` — vendored from `DDM-Net/datasets/roi_mask.py` (crop
  frames to a `*.mask.png` ROI bounding box, then multiply by the mask).
- `datasets/dataset.py` / `datasets/__init__.py` — added a `SEWING` dataset
  name (alongside upstream `GEBD`/`TAPOS`) that reads annotations from
  `data/efficient_gebd_dataset/{train,val}_annotation.pkl`.
- `train.py` — added the matching `SEWING` branch when resolving the GT
  annotation path for evaluation, and made the "save scripts to output dir"
  step tolerant of a missing `config-files/...` reference file.
- `config-files/sewing_csn.yaml` — training config for the sewing dataset
  (video-domain CSN backbone + FPN + DiffFormer/DiffMixer heads x2).
- `script/train/train_sewing_csn.sh`, `script/test/test_sewing_csn.sh` —
  entrypoints for training/evaluating on the sewing dataset.

## 1. Environment

```bash
conda create --name EfficientGEBD python=3.10
conda activate EfficientGEBD
pip install -r src/step_segment/EfficientGEBD/requirements.txt
```

## 2. Dataset preparation

Build the offline frames + Kinetics-GEBD-style pickle annotations from
`data/**/step_segments.json`:

```bash
python tools/prepare_efficient_gebd_dataset.py \
  --split-mode random --val-ratio 0.2 --seed 42
```

This produces:

```
data/efficient_gebd_dataset/
  images/{train,val}/<video_id>/frame<N>.jpg   # ROI-cropped/masked if *.mask.png exists
  train_annotation.pkl
  val_annotation.pkl
```

For a quick sanity check on 1-2 videos without processing the whole dataset:

```bash
python tools/prepare_efficient_gebd_dataset.py --max-videos 2 --overwrite
```

## 3. CSN pretrained backbone

Like upstream, the video-domain CSN backbone (`modeling/backbone.py`) loads an
mmaction2 config + ig65m-pretrained checkpoint from `CSN-pretrained/`:

```
CSN-pretrained/CSN-configs/R152/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.py
CSN-pretrained/CSN-ckpt/R152/ircsn_from_scratch_r152_ig65m_20200807-771c4135.pth
```

Download these from the links in the upstream README (`README_upstream.md`)
and place them under `src/step_segment/EfficientGEBD/CSN-pretrained/`.

## 4. Training

```bash
cd src/step_segment/EfficientGEBD
bash script/train/train_sewing_csn.sh          # 1 GPU
bash script/train/train_sewing_csn.sh 4         # 4 GPUs (torchrun DDP)
```

Override any yacs config key on the command line, e.g.:

```bash
bash script/train/train_sewing_csn.sh 1 SOLVER.BATCH_SIZE 4 SOLVER.MAX_EPOCHS 50
```

## 5. Evaluation

```bash
cd src/step_segment/EfficientGEBD
bash script/test/test_sewing_csn.sh output/sewing/sewing_csn_x2/model_best.pth
```

Reports F1/Recall/Precision at relative-distance thresholds
`[0.05, ..., 0.5]` (see `utils/eval.py`), matching the GEBD convention.

## 6. Benchmark vs DDM-Net

`tools/benchmark_step_segment_models.py` compares EfficientGEBD and DDM-Net
predictions against the same `step_segments.json` ground truth (boundary
Recall/Precision/F1 at multiple tolerance thresholds, plus checkpoint size /
inference speed if provided). See that script's `--help` for usage.
