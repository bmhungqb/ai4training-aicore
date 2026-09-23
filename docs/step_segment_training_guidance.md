# Step Segmentation Training Guidance (DDM-Net)

This guide covers fine-tuning [DDM-Net](https://github.com/MCG-NJU/DDM) (vendored from
NVIDIA's [SOP Monitoring Blueprints](https://github.com/NVIDIA/sop-monitoring-blueprints/tree/main/microservices/sop-training-bp/microservices/ddm-training-ms/ddm))
to detect step/operation boundaries in the sewing videos under `data/`.

## 1. What "step segmentation" means here

Each video has `action_segments_annotated.json`: a dense list of short, physically-segmented
action clips, each labeled with an `operation_name` (in Vietnamese) — e.g. many consecutive
clips can share the same operation. DDM-Net is trained instead to detect **generic event
boundaries**: the timestamps where the *operation* changes, regardless of what the operation
is called. So before training we merge consecutive same-operation clips into **step segments**
— the boundaries between them are the ground-truth events DDM-Net learns to detect.

Empty (`""`) and `"UNKNOWN"` operations are kept as their own step segments (they represent
idle/transition intervals, and a boundary into/out of them is still a real transition).

## 2. Environment setup (server)

```bash
conda create -n ddm python=3.10 -y
conda activate ddm
cd ai4training-aicore-poc
pip install -r src/step_segment/DDM-Net/requirements.txt
# only needed for the nvdinov2 backbone (CUDA extension from TAO); skip for resnet50
cd src/step_segment/DDM-Net && pip install -e . && cd -
```

Requires PyTorch + CUDA to already be installed for your GPU (not pinned in
`requirements.txt`) — install the correct `torch`/`torchvision` build for your CUDA version
first, e.g.:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 3. Data processing

### Step 1 — merge consecutive same-operation clips into step segments

```bash
python tools/process_step_segments.py
```

Writes `step_segments.json` next to every `data/<cd>/<chuyen>/action_segments_annotated.json`.
Use `--dry-run` to preview merge counts without writing files.

### Step 2 — build the DDM-Net train/val dataset

```bash
# Random 80/20 split by video
python tools/prepare_ddm_dataset.py --split-mode random --val-ratio 0.2 --seed 42

# Or hold out specific cd folders for validation
python tools/prepare_ddm_dataset.py --split-mode by_folder --val-folders cd18,cd19,cd20
```

This writes, under `data/ddm_dataset/`:
- `train_annotation.json`, `val_annotation.json` — DDM-Net format:
  `{"<cd>_<chuyen>": [{"event": ..., "description": ..., "start_timestamp": ..., "end_timestamp": ...}, ...]}`
- `videos/<cd>_<chuyen>.mp4` — symlinks to the original `.mp4` files (no duplication of
  video data).

Re-run Step 2 any time you change the split; it overwrites the symlinks/annotation files.

## 4. Launch training

Default config: `src/step_segment/DDM-Net/config/ddm_train_config.yaml` (resnet50 backbone,
224px input, batch size 8, AdamW, 30 epochs, EMA + F1-score checkpoint selection). Paths in
the config are relative to the repo root, so always run from `ai4training-aicore-poc/`.

```bash
# Single GPU
bash src/step_segment/DDM-Net/tools/run_train.sh 1

# Multi-GPU (DDP)
bash src/step_segment/DDM-Net/tools/run_train.sh 4

# Or call the training script directly to override any config value via CLI
python src/step_segment/DDM-Net/train_sop_lightning.py \
  --config src/step_segment/DDM-Net/config/ddm_train_config.yaml \
  --epochs 50 --batch-size 16 --num-gpus 2
```

Run in the background on a server with `tmux`/`nohup`:

```bash
tmux new -s ddm_train
conda activate ddm
bash src/step_segment/DDM-Net/tools/run_train.sh 1 2>&1 | tee train.log
# detach: Ctrl-b d, reattach later: tmux attach -t ddm_train
```

### Monitoring

- Console/`train.log`: per-step loss, per-epoch `val/f1_score`.
- TensorBoard: `tensorboard --logdir lightning_output` (checkpoints and logs are written to
  `training_config.output`, default `./output_lightning`).
- Checkpoints: best `checkpoint_top_k` models by `val/f1_score` are saved under
  `lightning_output/<exp_name>/checkpoints/`.

## 5. Validation / inference

Use the saved checkpoint to run validation-only, or point DDM-Net's inference tooling
(`microservices/sop-inference-bp` in the NVIDIA blueprint repo) at
`lightning_output/<exp_name>/checkpoints/<best>.ckpt` and
`data/ddm_dataset/val_annotation.json` to score boundary precision/recall/F1.

See `src/step_segment/DDM-Net/config/config_guide.md` and `VENDOR_README.md` for the full
list of config knobs (augmentation, EMA, schedulers, distributed strategy, etc.).

## 6. Key parameters to tune for this dataset

- `dataset_config.frames_per_side` — half-width of the temporal window DDM-Net compares
  around each candidate boundary. Sewing operations are short (seconds); start with the
  default (5) and reduce if boundaries are too coarse.
- `dataset_config.min_change_dur` — minimum gap (seconds) between two labeled boundaries;
  segments merged from very fast consecutive actions may need a smaller value.
- `training_config.model_ema` / `model_ema_start_epoch` — enabled by default; delays EMA
  averaging until the model has partially converged.
