# exp_003_min_peak_distance_suppression

Targets **Hypothesis 2: Feature Representation Over-Sensitive to Fine Local Motion** (Confidence: 65%, highest-confidence hypothesis).
**Prerequisite**: `exp_000_training_health_audit` and `exp_000b_aggregate_error_histograms` must both PASS/confirm before trusting this experiment's results.

## What this does
Entirely **post-processing on the existing raw score pickle**
(`model_pred_dict_ep-1.pkl`) — **no re-inference, no GPU**:

1. **Step 0 — Simple-threshold-sweep control**: runs the already-existing
   `tools/sweep_diffgebd_threshold.py` across thresholds `[0.3 .. 0.8]` as a
   zero-new-code baseline (Occam's Razor — prefer this if it already achieves
   comparable precision gains).
2. **Main — LOVO min-peak-distance suppression**: adds a new greedy
   min-distance peak-suppression step
   (`tools/export_diffgebd_predictions.py::suppress_min_peak_distance`,
   candidates `{0.5s, 1.0s, 1.5s}`), tuned via **Leave-One-Video-Out (LOVO)**
   cross-validation across the 9 val videos (tune on 8, evaluate held-out on
   the 9th, aggregate across folds) instead of sweeping directly against the
   reported val set — avoids tuning-on-test leakage.

## Run

```bash
python experiments/step_segment/iter_01/exp_003_min_peak_distance_suppression/run.py
```

No GPU needed. Estimated total runtime: ~3-5 min.

## Outputs
`outputs/step_segment/iter_01/exp_003_min_peak_distance_suppression/`
- `step0_threshold_sweep_control/threshold_sweep_log.txt`
- `lovo_min_peak_distance/lovo_report.json`
- `lovo_min_peak_distance/predictions.json` (LOVO-aggregated held-out predictions)

## What to observe
1. **Occam's Razor check**: does `lovo_report.json`'s `lovo_aggregate_f1`
   Pareto-dominate the best threshold-only F1 from `threshold_sweep_log.txt`
   at matched recall? If not, adopt the simpler threshold-only fix instead.
2. **`cd12_chuyen1` over-prediction fix**: check `fold_results` for
   `cd12_chuyen1` — precision should improve toward 35-45% (from 22.5%
   baseline) with <5pt recall loss.
3. **Worker/station leakage (mandatory)**: `worker_station_leakage_check_cd12`
   field reports `cd12_chuyen1` AND `cd12_chuyen2` together. If
   `cd12_chuyen1` improves but `cd12_chuyen2` recall (baseline 46.2%)
   regresses, this is **overfitting to one session's noise profile**, not a
   general fix — do NOT adopt despite aggregate Macro F1 looking better.
4. LOVO caveat: results use LOVO in place of a true disjoint test set (must
   be stated explicitly in the final results report, per Debater's requirement).
