# Evaluation Report: iter_02

- **Track**: `step_segment` (DiffGEBD)
- **Scope**: Meta-evaluation of the 4 experiments approved and executed in `experiments/step_segment/diff_gebd/iter_01/04_debate_verdict.md` (`exp_000`, `exp_000b`, `exp_002`, `exp_003`), read against the iter_01 baseline (`sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75`).
- **Baseline reference (`01_eval_report.md` / `eval_report.json` of iter_01)**: Macro F1 (0.5s) = **0.4077**, Recall = 0.4067, Precision = 0.4086, F1@0.25s = 0.2275, F1@1.0s = 0.5856.
- **Machine-readable facts**: `experiments/step_segment/diff_gebd/iter_02/eval_report.json`

---

## Pillar 1 — Training Log Health & Dynamics (`exp_000_training_health_audit`)

- **Status**: `WARN`
- `train.log` contains **3 restart segments** (epoch numbers reset at line 3 and again at line 10), confirming the log is a concatenation of ≥2 resumed/restarted training runs rather than one continuous run.
- Reconstructed chronological curve (Rel@0.05 F1, 18 recorded epoch-lines total across 3 segments):

  | Segment | Epochs | F1 sequence |
  | :---: | :--- | :--- |
  | 0 | 0-1 | 0.2184, 0.2304 |
  | 1 | 0-7 | 0.0818, 0.1699, 0.2169, 0.2859, 0.2174, 0.2741, **0.2946**, 0.2301 |
  | 2 | 7-14 | 0.2745, 0.2684, 0.2533, 0.2784, 0.2536, 0.2697, 0.2920, 0.2834 |

- **Global-best epoch**: Segment 1, Epoch 6, F1 = **0.2946**. No epoch in segment 2 (the final 8 recorded epochs) exceeds this value — the segment-2 F1 values oscillate between 0.2533 and 0.2920 without a clear upward trend.
- `model_best.pth` exists on disk at `src/step_segment/DiffGEBD/output/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/model_best.pth`.
- No NaN/Inf/divergence detected in any of the 3 segments.
- Note: the Rel@0.05 F1 values logged in `train.log` (max 0.2946) use a different metric definition than the multi-tolerance `f1_at_0_5s` reported in `eval_report.json` (0.4077) — the two numbers are **not directly comparable**.

---

## Pillar 2 — Aggregate Error Distribution & Post-Hoc Interventions

### 2.1. `exp_000b_aggregate_error_histograms` — Dataset-wide hypothesis check

Built from all 9 val videos' existing baseline `predictions.json` + GT (no re-inference).

| Metric | Value |
| :--- | :---: |
| Total FN (missed GT) | 264 |
| FN near chunk-start (<1.5s from chunk boundary) | 81 (30.68%) |
| Expected fraction if uniformly distributed | 30.0% |
| Total FP (spurious detections) | 262 |
| FP within 1.0s of another prediction | 83 (31.68%) |

- **H1 (FN concentrated near chunk-start, motivating `exp_002`)**: `hypothesis_1_generalizes = false` — the observed 30.68% near-chunk-start FN fraction is statistically indistinguishable from the 30% uniform-distribution expectation across all 264 dataset-wide FN cases (vs. the n=2 anecdotal sample originally used to motivate the hypothesis).
- **H2 (FP clustered sub-1s from another prediction, motivating `exp_003`)**: `hypothesis_2_generalizes = false` — 31.68% of the 262 dataset-wide FP cases are within 1.0s of another prediction, not markedly different from a baseline expectation for a dataset with predictions spread across full-length videos.
- **Verdict recorded by the tool**: Both `exp_002_val_overlap_context` and `exp_003_min_peak_distance_suppression` are flagged `DE-PRIORITIZE` on dataset-wide evidence, contradicting the n=2/n=2 anecdotal pattern that originally motivated both experiments in `02_diagnosis.md`.

### 2.2. `exp_002_val_overlap_context` — Overlapping val chunks (no retraining)

**Step 0 — Seed-variance noise floor** (unmodified baseline inference, re-run 3x with different DDIM sampling seeds, `--val-overlap-seconds 0.0`):

| Seed | Macro F1 (0.5s) | Recall | Precision |
| :---: | :---: | :---: | :---: |
| 42 | 0.3790 | 0.3483 | 0.4155 |
| 123 | 0.3553 | 0.3146 | 0.4082 |
| 2024 | 0.3435 | 0.3034 | 0.3959 |

- Run-to-run Macro F1 range across 3 seeds: **0.3435 – 0.3790** (spread = 0.0355, i.e. ±3.55 pts).
- All 3 seed-noise-floor Macro F1 values are already **below** the iter_01 reported baseline of 0.4077 (single-seed run) recorded in `01_eval_report.md`.

**Step 0.5 — Shifted-window control** (single chunk window shifted, no overlapping/duplicate predictions):
- Macro F1 (0.5s) = 0.3594, Recall = 0.3303, Precision = 0.3941 — falls within the seed-noise-floor range above.

**Main overlap treatment** (`--val-overlap-seconds 2.5`, same checkpoint, `merge-eps 0.35`):
- Macro F1 (0.5s) = **0.2975**, Recall = 0.2427, Precision = 0.3843, `num_pred = 281` (vs. 373 predictions in the seed-noise-floor / shifted-window runs).
- This is **below** every value in the 3-seed noise floor (0.3435–0.3790) and below the shifted-window control (0.3594).

**Per-video comparison for worker-leakage check (`cd12_chuyen1` vs `cd12_chuyen2`, same worker/station, different session):**

| Run | `cd12_chuyen1` F1 (R/P) | `cd12_chuyen2` F1 (R/P) |
| :--- | :---: | :---: |
| seed42 | 0.3103 (0.3462/0.2812) | 0.4944 (0.4231/0.5946) |
| seed123 | 0.2881 (0.3269/0.2576) | 0.5128 (0.3846/0.7692) |
| seed2024 | 0.3077 (0.3462/0.2769) | 0.5000 (0.4038/0.6562) |
| shifted_window | 0.2931 (0.3269/0.2656) | 0.5843 (0.5000/0.7027) |
| main_overlap_run | 0.2885 (0.2885/0.2885) | 0.4110 (0.2885/0.7143), num_pred=21 |

- In the overlap treatment, both `cd12_chuyen1` and `cd12_chuyen2` recall drop to the identical value **0.2885**, and `cd12_chuyen2`'s `num_pred` falls to 21 (vs. 26-37 across all control runs).

### 2.3. `exp_003_min_peak_distance_suppression` — Post-processing peak suppression

**Step 0 — Simple threshold-sweep control** (`tools/sweep_diffgebd_threshold.py`, zero new code):

| Threshold | Recall | Precision | F1 |
| :---: | :---: | :---: | :---: |
| 0.3 | 34.16% | 40.86% | 37.21% |
| 0.4 | 34.61% | 41.29% | 37.65% |
| 0.5 | 34.83% | 41.22% | 37.76% |
| 0.6 | 34.61% | 41.18% | 37.61% |
| 0.7 | 34.61% | 41.18% | 37.61% |
| 0.8 | 34.83% | 41.33% | **37.80%** |

**LOVO cross-validation for `--min-peak-distance`** (candidates tested: 0.5s / 1.0s / 1.5s):

- `baseline_f1_no_suppression` (no min-peak-distance) = 0.3776
- `lovo_aggregate_f1` (with LOVO-tuned suppression) = 0.3807
- `improvement` = **+0.0031** (0.31 pts)
- Every fold's selected `min_peak_distance` = 0.5s (the smallest candidate tested).
- Held-out per-fold F1 ranged from 0.25 (`cd4_chuyen1`) to 0.5176 (`cd12_chuyen2`).

**Worker-leakage check (`cd12_chuyen1` vs `cd12_chuyen2` under the LOVO-selected suppression):**

| Video | F1 (0.5s) | Recall | Precision |
| :--- | :---: | :---: | :---: |
| `cd12_chuyen1` | 0.3091 | 0.3269 | 0.2931 |
| `cd12_chuyen2` | 0.5176 | 0.4231 | 0.6667 |

- Recorded tool caveat: LOVO cross-validation was used in place of a disjoint held-out test set (9 val videos deemed too few for a 3-way split); this substitution is a stated limitation in `lovo_report.json`.

---

## Summary of Raw Numbers (No Interpretation)

| Run | Macro F1 (0.5s) |
| :--- | :---: |
| iter_01 baseline (`01_eval_report.md`) | 0.4077 |
| exp_002 seed42 noise floor | 0.3790 |
| exp_002 seed123 noise floor | 0.3553 |
| exp_002 seed2024 noise floor | 0.3435 |
| exp_002 shifted-window control | 0.3594 |
| exp_002 main overlap treatment (2.5s) | 0.2975 |
| exp_003 threshold-sweep control (best, thr=0.8) | 0.3780 |
| exp_003 LOVO min-peak-distance (aggregate) | 0.3807 |

---

## Next Step

Evaluation complete with training log health and aggregate dataset-wide error-distribution facts, plus the full seed-variance noise floor and per-video worker-leakage numbers for `exp_002` and `exp_003`. Proceed to run `@research-diagnoser` to formulate root-cause hypotheses based on these facts (iter_02 diagnosis should specifically address: (a) why `exp_000b`'s dataset-wide histograms contradict the anecdotal H1/H2 that motivated `exp_002`/`exp_003`, and (b) why the overlap treatment in `exp_002` scores below its own noise floor).
