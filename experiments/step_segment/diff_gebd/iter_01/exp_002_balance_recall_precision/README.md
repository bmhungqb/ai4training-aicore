# Experiment: exp_002_balance_recall_precision

- **Track**: `step_segment`
- **Model**: `DiffGEBD` (ResNet-50 + Denoising Diffusion Generative Model)
- **Iteration**: `iter_01`
- **Baseline**: `exp_001_baseline`

## Objective
`exp_001_baseline` prioritizes recall because its underlying config
(`config/sewing_diffgebd_resnet50_chunked.yaml`) sets `INPUT.GAUS_SIGMA: 1.5`,
tuned to fix a prior recall=0.19 sampling bug: it widens the gaussian
"positive" bump the model is trained to regress to around each true boundary,
so the model learns to fire high scores over a wider temporal window (more
coverage = more recall, less localization = more false positives = lower
precision).

(An earlier revision of this repo also added `MODEL.POS_LOSS_WEIGHT: 5.0`,
up-weighting the per-frame MSE loss 5x on boundary frames, doubling down on
the recall bias. That weighting has since been reverted — the loss is back
to the method's original, unweighted `F.mse_loss(logits, loss_targets)`.)

This experiment dials `GAUS_SIGMA` back down to move the operating point
back towards precision:

| Knob | exp_001_baseline | exp_002 (this) |
|---|---|---|
| `INPUT.GAUS_SIGMA` | 1.5 | 0.8 |

This is passed as a CLI config override on top of the same
`sewing_diffgebd_resnet50_chunked.yaml` base config (see
`experiments/templates/step_segment_runner.py::run_diff_gebd`), so no new
DiffGEBD config file is needed and both experiments stay diffable via
`training_params` in each `config.yaml`.

Note: at inference time, `TEST.THRESHOLD` (score cutoff, default 0.5) is a
second, independent lever for the same recall/precision trade-off — see
`tools/sweep_diffgebd_threshold.py` to tune it post-hoc on either checkpoint
without retraining.

## How to Run

```bash
# 1. Full Training & Inference
python experiments/step_segment/diff_gebd/iter_01/exp_002_balance_recall_precision/run.py --mode train

# 2. Inference Only
python experiments/step_segment/diff_gebd/iter_01/exp_002_balance_recall_precision/run.py --mode infer
```

## Expected Outputs
Outputs stored in `outputs/step_segment/diff_gebd/iter_01/exp_002_balance_recall_precision/`:
- `run.log`, `train.log`, `infer.log`
- `model_best.pth`
- `predictions.json`
- `metrics.json` — compare `macro_recall`/`macro_precision` against `exp_001_baseline`'s.
