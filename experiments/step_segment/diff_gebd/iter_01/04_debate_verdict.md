# Debate & Final Approval Verdict: iter_01

- **Track**: `step_segment` (DiffGEBD)
- **Input Plan**: `03_research_plan.md`
- **Debaters**: Debate Agent (Devil's Advocate) & Human Researcher
- **Timestamp**: 2026-09-28
- **STATUS**: `APPROVED FOR IMPLEMENTATION` — Human reviewed `03_research_plan.md` (Revision 2) and signed off

---

## 1. Debate Agent Challenges & Critical Review

### 1.1. Cross-Cutting Issues (Apply to All 3 Experiments)

| Checkpoint | Status | Debate Critique | Proposed Counter-Measure / Refinement |
| :--- | :---: | :--- | :--- |
| **Evidence Sufficiency** | `WARN` | The entire diagnosis is built from **exactly 2 FN cases and 2 FP cases**, and the 2 FP cases come from the **same single video** (`cd12_chuyen1`). Confidence scores of 65% (H2) and 55% (H1) are asserted on this anecdotal n=2/n=2 sample. There is no statistical aggregation (e.g., "what fraction of ALL false positives across all 9 videos occur within Xs of another prediction?", "what fraction of ALL false negatives fall in the first 1.5s of their chunk?"). Val set is only **9 videos, 445 GT boundaries total** — high variance is guaranteed at this scale. | Before running any experiment, require a **cheap, code-only aggregate probe** (no GPU, just re-processing existing `predictions.json` + GT): (a) histogram of ALL FN discrepancy distances bucketed by chunk-relative position, (b) histogram of ALL FP-to-nearest-other-prediction distances. This costs ~zero compute and either confirms or kills H1/H2 *before* spending inference/training budget. |
| **Overfitting Risk (Worker/Station Confound)** | `FAIL` | Checked `train_annotation.pkl` vs `val_annotation.pkl`: **worker/station `cd12`, `cd4`, `cd5`, `cd6`, `cd8`, `cd10` all appear in BOTH train and val splits**, just under different `chuyenN` (session) suffixes (e.g. `cd12_chuyen3` is in train, `cd12_chuyen1`/`cd12_chuyen2` are in val). This means the split is **session-level, not worker/station-level**. Any solution that "fixes" `cd12_chuyen1`'s precision (worst in val) risks simply better-memorizing that specific worker's camera angle/sewing style seen during training on `cd12_chuyen3`, rather than fixing a genuine architectural flaw. Notably `cd12_chuyen2` (same worker/station, different session) is already the **best**-precision video (60%) in val — the within-worker variance (22.5% vs 60% precision) is larger than most of the effects the experiments are trying to produce. | Any claimed improvement must be checked **per-worker**, not just aggregate Macro F1. If `exp_003` (peak suppression) improves `cd12_chuyen1` precision but `cd12_chuyen2`'s recall (already low-ish at 46.2%) drops, that is evidence of overfitting to one video's noise profile, not a general fix. Require the verification step to explicitly report both `cd12_chuyen1` AND `cd12_chuyen2` together, not `cd12_chuyen1` in isolation as currently written in Exp 2's "Verification test." |
| **Non-Determinism Confound (Diffusion Sampling)** | `FAIL` | Config confirms `DIFFUSION.DETERMINISTIC: false`. DiffGEBD uses **DDIM stochastic sampling with CFG**. Experiment 1 (`exp_002_val_overlap_context`) requires **re-running inference** on the existing checkpoint — since sampling is non-deterministic, any F1 delta observed could be pure run-to-run sampling noise, NOT the effect of added chunk overlap. The plan does not mention seeding or running multiple inference passes to establish a noise floor. | **Mandatory pre-experiment control**: Before touching val-overlap, re-run the *unmodified* baseline inference 2-3 times with different seeds and report the Macro F1 variance across runs. If baseline Macro F1 already swings ±3-4 points run-to-run, any "+3-6 pt improvement" claimed by Exp 1 is statistically meaningless. Alternatively, force `DETERMINISTIC: true` for this specific diagnostic experiment (accepting the accuracy trade-off) so the overlap variable can be cleanly isolated. |
| **Threshold/Hyperparameter Leakage** | `FAIL` | Experiment 2 explicitly proposes to **sweep `--min-peak-distance` over {0.5s, 1.0s, 1.5s} directly on the val set** and pick whichever improves val Macro F1. This is the same val set used for the primary reported metric in `01_eval_report.md`. Sweeping any hyperparameter against the exact set you report final numbers on is **tuning-on-test / leakage** — the resulting F1 gain is optimistic and will not generalize. The existing `tools/sweep_diffgebd_threshold.py` tool has this same flaw baked in already (per its own docstring, it sweeps directly on `--split val`). | Either (a) hold out a further split of val as a tuning-only subset and report final numbers on a disjoint portion, or (b) explicitly flag in the final report that the chosen `min-peak-distance`/threshold is **selected**, not blind-tested, and must be re-validated once new data/videos become available. Given the tiny dataset (9 val videos), a full train/val/test 3-way split may not be feasible — at minimum, this caveat must be written into the eventual results report, not silently treated as a held-out win. |

### 1.2. Experiment-Specific Critiques

| Checkpoint | Status | Debate Critique | Proposed Counter-Measure / Refinement |
| :--- | :---: | :--- | :--- |
| **Exp 1 — Variable Isolation** | `WARN` | Adding overlap to val chunks doesn't just "restore context" — it also means each physical timestamp near a chunk boundary is now **covered by 2+ overlapping chunk inferences instead of 1**, and `merge_close_boundaries` (eps=0.35s by default) averages/collapses near-duplicate detections. This is closer to an **ensembling/multi-view voting effect** than a pure "more context" effect — the plan conflates these two distinct mechanisms and will not be able to tell which one caused any observed change. | Rename this what it is: an entangled (context + multi-view-averaging) intervention. To truly isolate "context alone," a cleaner control would be to shift the *single* chunk window (e.g., start chunks 2s earlier so the same GT boundary sits mid-chunk instead of near-edge) without introducing multiple overlapping predictions per timestamp. Recommend adding this as an explicit alternative control run alongside the overlap version. |
| **Exp 1 — Ablation Baseline** | `WARN` | The plan compares against `sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75` (iter_01) as baseline — correct in principle — but does NOT control for the `merge-eps` parameter change in effect. Baseline used non-overlapping chunks so `merge_close_boundaries` rarely had duplicate candidates to merge; the new run will exercise this merge logic heavily for the first time. If `merge-eps=0.35` is miscalibrated, the experiment could show a false "no improvement" or false "improvement" purely from bad merge behavior. | Add an explicit unit-level sanity check: manually inspect 3-5 chunk-overlap regions' raw (pre-merge) predicted boundary lists to confirm `merge_close_boundaries` is behaving as intended before trusting the final F1 comparison. |
| **Exp 2 — Complexity Test** | `WARN` | Greedy peak-suppression by "keep highest score within N seconds" is reasonable, but is essentially manual NMS — the plan doesn't justify why a **simpler, single-parameter fix** (e.g., just raising the global score threshold, which `tools/sweep_diffgebd_threshold.py` already supports with zero new code) wouldn't achieve a similar precision/recall trade-off with far less implementation risk. The research plan's own train.log shows peak training-time Rel@0.05 F1 recall (~0.44) far exceeding precision (~0.22) — a symptom a **simple threshold increase** directly addresses. | Before writing new suppression code, run the *already-existing* `tools/sweep_diffgebd_threshold.py` across a range of thresholds as a zero-cost control baseline. Only proceed with the new min-peak-distance code if it demonstrably outperforms the simple threshold-only sweep at matched recall — otherwise Occam's Razor favors the existing tool. |
| **Exp 3 — Ablation / Isolation** | `FAIL` | Correctly de-prioritized by the Planner (lowest confidence, 35%), but even as an optional experiment, widening `GAUS_SIGMA` **simultaneously changes the effective positive:negative frame ratio during training** (wider Gaussian = more frames labeled as "near-boundary positive"), which is a **second, uncontrolled variable** conflated with "boundary softness/ambiguity absorption." A Macro F1 change could just as easily be explained by a shift in effective class balance as by "the model now tolerates label ambiguity better." | If Exp 3 is ever run, it must be paired with a loss-reweighting control (hold effective positive:negative ratio constant, e.g. via `POS_WEIGHT`-style compensation) to cleanly isolate "soft target width" from "positive-class frequency." Given the low confidence and high cost (full retrain), recommend this experiment be **dropped from this iteration entirely** unless Exp 1 & 2 leave a large unexplained residual gap. |
| **Training Health Red Flag (Not Addressed by Any Experiment)** | `FAIL` | `train.log` for the evaluated baseline shows **duplicate/non-monotonic epoch numbers** (`Epoch 00, 01, 00, 01, ..., 07, 07`), consistent with the log being a concatenation of multiple restarted/resumed training runs, not one clean continuous run. None of the 3 proposed experiments investigate whether the baseline itself trained cleanly to convergence. If the baseline model is undertrained or was corrupted by a bad resume, then all 3 proposed "fixes" are being layered on top of an **already-confounded baseline**, and any improvement (or lack thereof) cannot be cleanly attributed. | Recommend adding a **zero-cost Experiment 0**: re-plot the full training curve from `train.log` (all restart segments), confirm final `model_best.pth` corresponds to the run's true best epoch (not an artifact of a resume overwriting a better earlier checkpoint), before trusting comparisons against this baseline in Exp 1-3. |

---

## 2. Human Feedback & Rebuttals

Human reviewed `03_research_plan.md` Revision 2 (which already incorporates all Rev-1 critiques from Section 1 above — 4 `FAIL`-rated items fully conceded/fixed, all `WARN`-rated items addressed with added controls) and **explicitly approves the full Revision-2 batch as written**, with `exp_004_wider_gaussian_sigma` remaining dropped from this iteration per the Planner's own concession.

- **Decision on Experiment 0 (`exp_000_training_health_audit`)**: `Approved` -> Note: Blocking gate, run first.
- **Decision on Experiment 0b (`exp_000b_aggregate_error_histograms`)**: `Approved` -> Note: Blocking gate, run first; Exp 1/2 are only trusted if the corresponding histogram confirms the anecdotal pattern.
- **Decision on Proposal A / Experiment 1 (`exp_002_val_overlap_context`)**: `Approved` -> Note: Including Step 0 (seed-variance noise floor) and Step 0.5 (shifted-window control) as specified in Revision 2.
- **Decision on Proposal B / Experiment 2 (`exp_003_min_peak_distance_suppression`)**: `Approved` -> Note: Including Step 0 (simple-threshold-sweep control) and LOVO cross-validation tuning as specified in Revision 2.
- **Decision on Proposal C / Experiment 3 (`exp_004_wider_gaussian_sigma`)**: `Rejected (for this iteration)` -> Note: Confirmed dropped, deferred to a future iteration per Planner's Section 5 rationale.

---

## 3. Final Approved Experiments for Implementation

List of experiments authorized to be generated by `Implementor`:

### Approved 0: `exp_000_training_health_audit`
- **Scope**: Read-only audit of `outputs/step_segment/diff_gebd/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/train.log` to verify `model_best.pth` corresponds to the true global-best epoch across all restart segments.
- **Fixed Variables**: No re-training, no re-inference — CPU-only log parsing.
- **Modified Variable**: None (diagnostic gate).
- **Target Output Directory**: `outputs/step_segment/diff_gebd/iter_01/exp_000_training_health_audit/`

### Approved 0b: `exp_000b_aggregate_error_histograms`
- **Scope**: Dataset-wide (all 9 val videos) FN chunk-relative-position histogram and FP nearest-neighbor-distance histogram, built from the existing baseline `predictions.json` + GT.
- **Fixed Variables**: No re-inference — reuses existing baseline predictions.
- **Modified Variable**: None (diagnostic gate).
- **Target Output Directory**: `outputs/step_segment/diff_gebd/iter_01/exp_000b_aggregate_error_histograms/`

### Approved 1: `exp_002_val_overlap_context`
- **Scope**: Restore pre-boundary temporal context at val inference time via overlapping val chunks, on the existing trained checkpoint (no retraining).
- **Fixed Variables**: Model weights (`model_best.pth`), train chunking, threshold (`0.5`), `merge-eps` (`0.35`).
- **Modified Variable**: `--val-overlap-seconds` (`0.0` baseline -> `2.5` treatment), plus Step 0 seed-variance control runs and Step 0.5 shifted-grid control run.
- **Target Output Directory**: `outputs/step_segment/diff_gebd/iter_01/exp_002_val_overlap_context/`

### Approved 2: `exp_003_min_peak_distance_suppression`
- **Scope**: Post-processing peak-selection suppression on the existing raw score pickle (`model_pred_dict_ep-1.pkl`), with LOVO-tuned `--min-peak-distance`.
- **Fixed Variables**: Model weights, training config, base threshold (`0.5`).
- **Modified Variable**: `--min-peak-distance` (new post-processing parameter on `tools/export_diffgebd_predictions.py`), tuned via Leave-One-Video-Out cross-validation; Step 0 threshold-sweep control run first.
- **Target Output Directory**: `outputs/step_segment/diff_gebd/iter_01/exp_003_min_peak_distance_suppression/`

---

## 4. Sign-off
- **Human Approval Status**: ✅ **APPROVED FOR IMPLEMENTATION**
- **Approved by**: Human (Project Owner)

---

> [!IMPORTANT]
> Handoff to `@research-implementor`: scaffold `exp_000_training_health_audit/`, `exp_000b_aggregate_error_histograms/`, `exp_002_val_overlap_context/`, `exp_003_min_peak_distance_suppression/` under `experiments/step_segment/diff_gebd/iter_01/`, plus `run_all.sh`, per Section 3 above. `exp_004_wider_gaussian_sigma` is NOT authorized for implementation this iteration.
