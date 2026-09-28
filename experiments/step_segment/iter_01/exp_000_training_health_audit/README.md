# exp_000_training_health_audit

**Status**: BLOCKING GATE — must be run and PASS/WARN (not FAIL) before trusting `exp_002_val_overlap_context` and `exp_003_min_peak_distance_suppression`.

## What this does
Read-only, CPU-only audit of `train.log` for the iter_01 baseline
(`sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75`). Parses every epoch/F1
line, segments by restart boundaries (epoch number resets), and reconstructs
the true chronological Rel@0.05 F1 curve across all restart segments. Flags
whether `model_best.pth` plausibly corresponds to the global-best epoch.

No GPU, no re-training, no re-inference. Runtime: ~1-2 min.

## Run

```bash
python experiments/step_segment/iter_01/exp_000_training_health_audit/run.py
```

## Outputs
- `outputs/step_segment/iter_01/exp_000_training_health_audit/audit_report.json`
- `outputs/step_segment/iter_01/exp_000_training_health_audit/training_curve_audit.png`

## What to observe
- `status`: `PASS` / `WARN` / `FAIL`.
  - `FAIL` = NaN/divergence detected in the reconstructed curve → **do not trust** Exp 1/2 results against this baseline; baseline must be retrained cleanly first.
  - `WARN` = restart structure confirmed (matches the Debater's original red flag) but no divergence found → proceed to Exp 1/2, but Human should spot-check `model_best.pth`'s save timestamp against `global_best_epoch`/`global_best_segment` in the report.
  - `PASS` = single clean run, no restarts.
- `n_restart_segments`, `global_best_epoch`, `global_best_f1`: cross-check against `model_best.pth`.
