# Research & Experiment Plan: iter_01

- **Track**: `step_segment` (DiffGEBD)
- **Input Diagnosis**: `02_diagnosis.md`
- **Target Hypotheses Addressed**: `Hypothesis 1 (Chunk-Boundary Temporal Context Starvation)`, `Hypothesis 2 (Feature Representation Over-Sensitive to Fine Local Motion)` — `Hypothesis 3 (Ground-Truth Boundary Ambiguity)` dropped in Revision 2, see Section 5.
- **Timestamp**: 2026-09-28 (Revision 1) / 2026-09-28 (Revision 2, post-debate)
- **Revision**: **Rev 2** — responding to `@research-debater` critiques in `04_debate_verdict.md`

---

## 1. Literature & Technical Survey

| Solution Idea | Dimension (Data / Training / Arch / Inference) | References / Prior Art | Key Mechanism |
| :--- | :--- | :--- | :--- |
| Idea 1: Overlapping inference context window at val-time | `Data` | Sliding-window temporal action detection standard practice (e.g. BSN, BMN use overlapping proposal windows at inference to avoid boundary-truncation artifacts); DiffGEBD's own training-side chunking (`docs/step_segment/diff_gebd.md §2.2`) already uses this trick to fix long-video sampling starvation, but only for the *training* split. | Val chunking (`tools/chunk_diff_gebd_dataset.py`, `--val-overlap-seconds`) currently defaults to `0.0` (see code confirmed below), while train uses `--overlap-seconds 1.0`. Boundaries close to a chunk's start frame have zero preceding temporal context to compute self-similarity against, and are structurally under-represented at inference-time only, not training-time. |
| Idea 2: Minimum-Distance / Peak-Isolation Suppression on Score Curve | `Inference` | Standard 1D NMS in generic event boundary detection & DDM-Net's own `1D NMS filter` (already implemented for the DDM-Net track, per `step_segment_overview.md §4`); GEBD benchmark papers commonly apply peak-picking with minimum inter-peak distance rather than static thresholding alone. | Current `tools/export_diffgebd_predictions.py::get_boundary_frame_indices` only groups *consecutive* above-threshold frames — it has **no minimum-distance suppression** between separate above-threshold groups, so isolated noisy spikes anywhere in the sequence (even 0.3–0.5s apart) become independent predicted boundaries. Adding a minimum-spacing constraint (keep only the highest-scoring peak within a sliding window) directly targets spurious fine-motion-triggered predictions. |
| Idea 3: Wider Gaussian Boundary Target (`GAUS_SIGMA`) | `Training` | Kinetics-GEBD baseline / soft-label boundary regression literature — widening the Gaussian target kernel is the standard technique to accommodate annotator boundary-time disagreement (soft targets absorb temporal jitter in "true" boundary position). | Current config already applies `ADAPTIVE_GAUS_SIGMA: true` with `GAUS_SIGMA_MIN: 0.5` / `GAUS_SIGMA_MAX: 1.0`. Fixing/raising this ceiling (e.g. widening to a flat `GAUS_SIGMA: 1.5-2.0`) tests whether the model under-fits ambiguous/soft transitions because the current sigma range is too tight, directly probing Hypothesis 3 without needing new annotation data. |

---

## 2. Proposed Solutions Breakdown

### Solution Candidate A: Overlapping Validation Chunking (Restore Pre-Boundary Context)
- **Target Hypothesis**: Addresses `Hypothesis 1: Chunk-Boundary Temporal Context Starvation`
- **Component Affected**: `tools/chunk_diff_gebd_dataset.py` (`--val-overlap-seconds` argument), regenerating `val_annotation_chunked.pkl` + re-running `tools/infer_diffgebd.py` on the **existing trained checkpoint** (no retraining required — this is a pure inference-side data reslicing change).
- **Pros / Cons**:
  - *Pros*: Zero retraining cost — isolates purely whether inference-time context truncation (not model capacity) explains the FN cluster near clip-start timestamps (~3s). Cheap to test (~minutes, re-slicing + re-inference only).
  - *Risks / Cons*: If val chunks now overlap, `merge_close_boundaries` (`merge-eps=0.35`) must correctly de-duplicate boundaries detected redundantly in overlapping regions, or precision could *artificially drop* from duplicate near-identical predictions — must verify merge logic behaves correctly before attributing any F1 change to the context hypothesis alone.

### Solution Candidate B: Minimum-Distance Peak Suppression on Score Curve
- **Target Hypothesis**: Addresses `Hypothesis 2: Feature Representation Over-Sensitive to Fine Local Motion`
- **Component Affected**: `tools/export_diffgebd_predictions.py::get_boundary_frame_indices` (add a `--min-peak-distance` post-processing step: among any two predicted boundaries closer than N seconds, keep only the one with higher peak score).
- **Pros / Cons**:
  - *Pros*: Directly targets the observed failure mode in `cd12_chuyen1` (71 preds vs 52 GT) without touching the model or retraining; reuses the already-dumped `model_pred_dict_ep*.pkl` raw scores, so it can be swept cheaply like `tools/sweep_diffgebd_threshold.py` does for the threshold value.
  - *Risks / Cons*: If two GT boundaries are genuinely close together (e.g. rapid sub-steps), aggressive suppression could newly introduce False Negatives — must sweep the minimum-distance parameter against a range of values and check per-video recall doesn't regress on the higher-boundary-density videos (`cd8_chuyen2`, `cd19_chuyen3`, `cd12_chuyen2`).

### Solution Candidate C: Widened Gaussian Boundary Target (`GAUS_SIGMA`)
- **Target Hypothesis**: Addresses `Hypothesis 3: Ground-Truth Boundary Ambiguity`
- **Component Affected**: `src/step_segment/DiffGEBD/config/sewing_diffgebd_resnet50_chunk5s.yaml` (`INPUT.GAUS_SIGMA_MAX`, `INPUT.GAUS_SIGMA_MIN`), requires **full retraining**.
- **Pros / Cons**:
  - *Pros*: Directly probes whether the model is under-fitting soft/ambiguous transitions (like the two FN cases showing continuous fabric-feeding motion) due to an overly narrow target kernel; low implementation cost (single config change).
  - *Risks / Cons*: Most expensive experiment (full retrain, ~similar cost to original baseline run); widening sigma also inherently *lowers* achievable precision at the tightest tolerance window (±0.25s) by design, since positive-target mass spreads over more frames — must interpret Macro F1 change specifically at ±0.5s (primary window) rather than ±0.25s to avoid conflating "looser labels" with "genuinely better localization."

---

## 3. Batch Experiment Plan (Candidates for Debate) — Revision 2

> [!NOTE]
> **Revision 2 changes**: Added `Experiment 0` (zero-cost training-health audit, blocking gate) and `Experiment 0b` (zero-cost aggregate FN/FP histogram probe, blocking gate) that must both pass before Exp 1/2 are trusted. `Experiment 1` now includes a seed-variance noise-floor control and a shifted-single-window control alongside the overlap run. `Experiment 2` now runs the existing `tools/sweep_diffgebd_threshold.py` first as a simpler control, and uses leave-one-video-out (LOVO) tuning instead of sweeping directly on the full reported val set. `Experiment 3` (`exp_004_wider_gaussian_sigma`) is **dropped from this iteration** per Debater critique (uncontrolled class-balance confound + lowest confidence + highest cost) — see Section 5 for full rebuttal.

### Experiment 0: `exp_000_training_health_audit` (NEW — Blocking Gate, zero-cost)
- **Target Diagnoser Hypothesis**: Cross-cutting precondition for all 3 hypotheses — validates that the baseline being compared against in Exp 1/2 is itself trustworthy.
- **Missing Probe Implemented**: Debater flagged that `train.log` shows non-monotonic/duplicate epoch numbers (`00,01,00,01,...,07,07`), consistent with a concatenated restart/resume log rather than one clean run.
- **Variable to Isolate**: None (read-only audit, no re-run).
- **Baseline to Compare Against**: N/A.
- **Implementation Changes**:
  - File: none (analysis-only). Parse all epoch/F1 lines in `outputs/step_segment/iter_01/sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75/train.log`, segment by restart boundaries (epoch number resets), and plot the true chronological Rel@0.05 F1 curve across all segments.
  - Cross-check that `model_best.pth` actually corresponds to the highest-F1 epoch across the *entire* concatenated history (not just the final restart segment) by comparing its save timestamp / epoch tag against the parsed curve.
- **Expected Outcome**:
  - Primary metric expectation: N/A (diagnostic gate, not a performance experiment).
  - Verification test: **PASS** = `model_best.pth` matches the global-best epoch and training shows normal convergence (no divergence/NaN across restarts) → proceed to Exp 1/2 with current baseline. **FAIL** = best checkpoint is stale or a restart silently regressed → baseline must be retrained cleanly before Exp 1/2 results can be trusted.

### Experiment 0b: `exp_000b_aggregate_error_histograms` (NEW — Blocking Gate, zero-cost)
- **Target Diagnoser Hypothesis**: Strengthens evidence base for `Hypothesis 1` and `Hypothesis 2` beyond the current n=2/n=2 anecdotal cases, per Debater's Evidence Sufficiency critique.
- **Missing Probe Implemented**: (a) histogram of ALL false-negative discrepancy distances bucketed by chunk-relative position (first 1.5s of chunk vs. rest); (b) histogram of ALL false-positive-to-nearest-other-prediction distances, across all 9 val videos, not just `cd12_chuyen1`.
- **Variable to Isolate**: None (read-only analysis on existing `predictions.json` + GT).
- **Baseline to Compare Against**: N/A.
- **Implementation Changes**:
  - File: new lightweight script, e.g. `tools/analyze_diffgebd_error_distributions.py`, consuming `outputs/step_segment/iter_01/.../predictions.json` and `val_annotation_chunked.pkl` (for chunk boundaries) — no GPU needed.
- **Expected Outcome**:
  - Primary metric expectation: N/A (diagnostic gate).
  - Verification test: If FN rate is NOT materially higher near chunk-starts across the full dataset (not just the 2 anecdotal cases), **Hypothesis 1 / Experiment 1 is de-prioritized or cancelled**. If FP-to-nearest-prediction distances are NOT concentrated in the sub-1s range dataset-wide, **Hypothesis 2 / Experiment 2 is de-prioritized or cancelled**. Only proceed to the corresponding paid experiment if its histogram confirms the anecdotal pattern generalizes.

### Experiment 1: `exp_002_val_overlap_context` (Revised)
- **Target Diagnoser Hypothesis**: Addresses `Hypothesis 1: Chunk-Boundary Temporal Context Starvation` (Confidence: 55%)
- **Missing Probe Implemented**: Diagnoser's missing probe asked to "bucket FN timestamps by relative position within their inference chunk and compare recall." This experiment operationalizes that probe indirectly: instead of measuring position-bucketed recall on the *existing* zero-overlap chunks, it removes the chunk-start-truncation condition entirely by re-slicing val with overlap, then re-measures recall on the same FN cases. If recall recovers, the original probe's conclusion is confirmed without needing to build a separate position-bucketing analysis script.
- **Variable to Isolate**: `--val-overlap-seconds` in `tools/chunk_diff_gebd_dataset.py`: `0.0` (baseline, current iter_01) → `2.5` (new, ~50% overlap of the 5s chunk).
- **Baseline to Compare Against**: `sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75` (iter_01, this evaluated run).
- **Implementation Changes**:
  - File: `tools/chunk_diff_gebd_dataset.py` invocation (val split only; train chunking / model weights untouched).
  - **[NEW] Step 0 — Noise-floor control (addresses Debater's Non-Determinism Confound, `FAIL`)**: Before changing anything, re-run `tools/infer_diffgebd.py` on the **unmodified** baseline (zero-overlap val chunks) 3 times with different random seeds (`DIFFUSION.DETERMINISTIC: false` is confirmed in config — this is expected stochastic DDIM/CFG sampling variance). Record the Macro F1@0.5s spread across these 3 runs as the noise floor. Any effect claimed for the overlap change below must exceed this noise floor to be considered real; alternatively, set `DIFFUSION.DETERMINISTIC: true` for this diagnostic experiment specifically to remove sampling noise entirely (accepted trade-off: slightly less representative of production inference behavior, but cleanly isolates the variable under test).
  - **[NEW] Step 0.5 — Shifted single-window control (addresses Debater's Variable Isolation, `WARN`)**: In addition to the overlapping-chunks run, add a second control condition: re-slice val with a single (non-overlapping) grid shifted by `+2.0s` at the start offset (`_chunk_grid(..., grid_start=... )`), so previously edge-adjacent GT boundaries (e.g. @3.05s, @3.43s) now sit mid-chunk instead of near-edge, WITHOUT introducing multiple overlapping predictions per timestamp or invoking `merge_close_boundaries`. This isolates "more context" from "multi-view averaging" as two separate, independently measurable effects.
  - Main run: `python tools/chunk_diff_gebd_dataset.py --chunk-seconds 5.0 --overlap-seconds 1.0 --val-overlap-seconds 2.5` to regenerate `val_annotation_chunked.pkl` with overlapping val chunks. Then re-run `tools/infer_diffgebd.py --config-file config/sewing_diffgebd_resnet50_chunk5s.yaml --weights <existing model_best.pth>` (same trained checkpoint, no retraining) and `tools/export_diffgebd_predictions.py`.
  - **[NEW] Sanity check (addresses Debater's Ablation Baseline, `WARN`)**: Manually inspect the raw (pre-merge) predicted-boundary lists for 3-5 overlap regions to confirm `merge_close_boundaries` (`merge-eps=0.35`) is deduplicating as intended before trusting any F1 comparison — this parameter was rarely exercised in the non-overlapping baseline and is now load-bearing for the first time.
- **Expected Outcome**:
  - Primary metric expectation: Recall@0.5s improves specifically among currently-missed boundaries located in the first ~1.5–2s of their original chunk (including the two severe FN cases: `cd5_chuyen3` @3.05s, `cd4_chuyen1` @3.43s), **exceeding the seed-variance noise floor established in Step 0**, while overall Macro F1@0.5s increases modestly (target: +3-6 pts) without precision collapsing.
  - Verification test: Re-generate the same `error_cases/` visual report; confirm `cd5_chuyen3_FN_t3.05s` and `cd4_chuyen1_FN_t3.43s` (or their equivalent) either disappear from the severe FN list or shrink in discrepancy distance, in BOTH the overlap run and the shifted-window control (if only overlap helps and shifted-window doesn't, the effect is likely multi-view averaging, not context restoration). **[NEW — Overfitting Risk]**: Report `cd4_chuyen1` and `cd10_chuyen1` (worst-F1, short videos) per-video F1 improvement, but this experiment does not target the worker/station leakage issue (no `cd12` involvement expected here) so no additional cross-check is required beyond standard per-video reporting.

### Experiment 2: `exp_003_min_peak_distance_suppression` (Revised)
- **Target Diagnoser Hypothesis**: Addresses `Hypothesis 2: Feature Representation Over-Sensitive to Fine Local Motion` (Confidence: 65%, highest-confidence hypothesis)
- **Missing Probe Implemented**: Diagnoser's missing probe asked to inspect the raw score curve around the two FP timestamps and compare "peakiness" between `cd12_chuyen1` (worst precision) and `cd12_chuyen2` (best precision). This experiment goes one step further: it directly re-thresholds the *same* raw `model_pred_dict_ep-1.pkl` scores with an added minimum-peak-distance constraint and measures whether the resulting precision gain on `cd12_chuyen1` specifically confirms that closely-spaced/noisy peaks (not distinct semantic boundaries) were the source of the false positives.
- **Variable to Isolate**: Post-processing peak-selection logic only — `--min-peak-distance` parameter added to `tools/export_diffgebd_predictions.py::get_boundary_frame_indices` (new: keep highest-score peak per N-second window). Model weights, threshold value (`0.5`), and all training config remain identical to baseline.
- **Baseline to Compare Against**: `sewing_diffgebd_resnet50_chunk5s_ann1_dim512_len75` (iter_01, this evaluated run) — same raw prediction pickle, only re-thresholded.
- **Implementation Changes**:
  - **[NEW] Step 0 — Simple-baseline control first (addresses Debater's Complexity Test, `WARN`)**: Before writing any new suppression code, run the already-existing `tools/sweep_diffgebd_threshold.py --pred-pkl <same raw pickle> --split val` across a range of thresholds (e.g. 0.3-0.8) as a zero-new-code control. Plot the resulting precision/recall trade-off curve. Only proceed to build the min-peak-distance suppression code if it demonstrably Pareto-dominates the simple-threshold sweep at matched recall (Occam's Razor: prefer the existing single-parameter tool if it achieves comparable precision gains).
  - File: `tools/export_diffgebd_predictions.py`.
  - Modification: Add a post-grouping suppression step after `get_boundary_frame_indices`: sort candidate boundary groups by peak score descending; greedily keep a boundary only if it is more than `--min-peak-distance` seconds away from all already-kept (higher-scoring) boundaries.
  - **[NEW] Leave-One-Video-Out (LOVO) tuning (addresses Debater's Threshold/Hyperparameter Leakage, `FAIL`)**: Instead of sweeping `--min-peak-distance` over `{0.5s, 1.0s, 1.5s}` directly against the full 9-video val set and picking the best-scoring value (tuning-on-test), use LOVO cross-validation across the 9 val videos: for each held-out video, select the best `min-peak-distance` using the *other 8* videos, apply it to the held-out video, and aggregate the resulting held-out-only F1 across all 9 folds. This is the standard low-data-regime substitute for a disjoint tune/test split (full 3-way split infeasible given only 9 videos / 445 boundaries total) and prevents the reported final number from being directly optimized against itself.
- **Expected Outcome**:
  - Primary metric expectation: `cd12_chuyen1` precision improves substantially (target: from 22.5% toward 35-45%) with minimal recall loss (<5 pts) on that video; overall LOVO-aggregated Macro F1@0.5s improves (target: +2-5 pts) as long as high-density videos (`cd8_chuyen2`: 121 GT, `cd19_chuyen3`: 88 GT) don't show new recall regressions from over-aggressive suppression, AND the simple-threshold-sweep control from Step 0 is outperformed (otherwise adopt the simpler fix instead).
  - Verification test: Re-run `tools/eval_step_segment_predictions.py` per-video breakdown under LOVO; confirm `cd12_chuyen1`'s over-prediction ratio (71 pred / 52 GT) drops toward parity, and that `cd12_chuyen1_FP_t94.29s` / `cd12_chuyen1_FP_t66.25s` (or nearby duplicate peaks) are successfully suppressed while true boundaries in the same video remain detected. **[NEW — Overfitting/Worker-Leakage Risk]**: Debater confirmed `cd12_chuyen1` and `cd12_chuyen2` are the **same worker/station** (different sessions) — report both together explicitly; if `cd12_chuyen1` precision improves but `cd12_chuyen2` recall (currently 46.2%) regresses, this indicates overfitting to one session's noise profile rather than a general fix, and the change should NOT be adopted despite the aggregate Macro F1 looking better.

### Experiment 3: `exp_004_wider_gaussian_sigma` — **DROPPED in Revision 2**
- **Status**: `CANCELLED — not part of this iteration's batch.`
- **Reason**: Debater's `FAIL`-rated Ablation/Isolation critique is conceded without rebuttal: widening `GAUS_SIGMA` simultaneously changes the effective positive:negative frame ratio during training (wider Gaussian → more frames labeled near-boundary-positive), which is a genuine second uncontrolled variable conflated with "boundary softness absorption." Combined with (a) this being the Diagnoser's lowest-confidence hypothesis (35%), (b) the highest implementation cost (full retrain, several GPU-hours), and (c) Experiments 0/0b/1/2 already consuming the iteration's compute+review budget, this experiment is removed from the current batch entirely rather than patched.
- **Future Reconsideration**: If Experiments 1-2 leave a large unexplained residual gap (e.g. FN cases persist even after context/suppression fixes, still showing continuous ambiguous motion at GT timestamps), revisit this hypothesis in a **future iteration** with the isolation fix Debater implicitly demands: pair any `GAUS_SIGMA` widening with a compensating `POS_WEIGHT`-style loss reweighting to hold the effective positive:negative class ratio constant, cleanly isolating "soft target width" from "positive-class frequency."

---

## 4. Resource & Feasibility Estimation (Revision 2)
- **Experiment 0** (`exp_000_training_health_audit`): Log parsing + plotting only. **~2 min, CPU only, blocking gate — must PASS before Exp 1/2 results are trusted.**
- **Experiment 0b** (`exp_000b_aggregate_error_histograms`): Reprocessing existing `predictions.json` + GT across all 9 videos. **~5 min, CPU only, blocking gate — must confirm anecdotal FN/FP patterns generalize before funding Exp 1/2.**
- **Experiment 1** (`exp_002_val_overlap_context`): Seed-variance control (3 extra inference runs on unmodified baseline, ~15-30 min) + shifted-window control re-chunk+infer (~15 min) + main overlap run re-chunk+infer (~15 min) + eval/export (~2 min). **Total: ~45-65 min (up from ~15 min in Rev 1, due to added noise-floor and isolation controls), no GPU training needed, single GPU inference only.**
- **Experiment 2** (`exp_003_min_peak_distance_suppression`): Simple-threshold-sweep control (~1 min, existing tool) + LOVO cross-validation sweep of `--min-peak-distance` across 9 folds (~1-2 min, still pure post-processing on existing pickle, CPU only). **Total: ~3-5 min, no re-inference or GPU needed at all** (up slightly from Rev 1 due to LOVO looping, still negligible cost).
- **Experiment 3** (`exp_004_wider_gaussian_sigma`): **DROPPED in Revision 2** — no compute allocated this iteration. If reinstated in a future iteration with the `POS_WEIGHT` isolation fix, estimate unchanged from Rev 1: several hours on single GPU (~6-8GB VRAM).

---

## 5. Rebuttal & Plan Revision History (Debate Feedback Loop)

### Revision Round: `Rev 2` (responding to `04_debate_verdict.md`)

#### Cross-cutting critiques

- **Critique Addressed**: *Evidence Sufficiency* (`WARN`) — hypotheses built on n=2 FN / n=2 FP anecdotes, no dataset-wide aggregation.
  - **Concession**: Fully conceded. Added **Experiment 0b** (`exp_000b_aggregate_error_histograms`), a zero-cost blocking gate that computes dataset-wide FN-position and FP-nearest-distance histograms across all 9 val videos before Exp 1/2 are funded. If the anecdotal pattern doesn't generalize, the corresponding paid experiment is cancelled.

- **Critique Addressed**: *Overfitting Risk — Worker/Station Confound* (`FAIL`) — `cd12_chuyen1`/`cd12_chuyen2` (both val) share the same worker/station as `cd12_chuyen3` (train); split is session-level (`--split-mode random`), not worker-level.
  - **Defense**: This is a **pre-existing dataset construction decision** (`tools/prepare_diff_gebd_dataset.py --split-mode random --val-ratio 0.2 --seed 42`, documented in `docs/step_segment/diff_gebd.md`), not something introduced by this research plan — it predates iter_01 and affects the baseline itself, not specifically the proposed experiments. The tool already supports `--split-mode by_folder --val-folders <list>` for worker-stratified splitting, so a cleaner split is possible but is a **separate, larger data-engineering change** out of scope for a same-checkpoint diagnostic iteration.
  - **Concession**: Nonetheless, conceded for THIS iteration's verification protocol: Experiment 2's verification test now explicitly requires reporting `cd12_chuyen1` AND `cd12_chuyen2` together, and rejects any change that improves one at the other's expense as overfitting rather than a genuine fix. **Recommend opening a separate future-iteration ticket** to regenerate the dataset split with `--split-mode by_folder` stratified by worker/station, so this confound doesn't recur in later iterations.

- **Critique Addressed**: *Non-Determinism Confound* (`FAIL`) — `DIFFUSION.DETERMINISTIC: false`; Exp 1 requires re-inference, so any F1 delta could be sampling noise, not the overlap effect.
  - **Concession**: Fully conceded. Added **Step 0 (Noise-floor control)** to Experiment 1: 3 seeded re-runs of the *unmodified* baseline establish the noise floor; any claimed improvement must exceed it, or `DETERMINISTIC: true` is used for this diagnostic run specifically.

- **Critique Addressed**: *Threshold/Hyperparameter Leakage* (`FAIL`) — sweeping `--min-peak-distance` directly on the reported val set is tuning-on-test.
  - **Defense**: A full disjoint train/tune/test 3-way split is not feasible given only 9 val videos / 445 total GT boundaries — further splitting would leave individual folds with single-digit boundary counts, making per-fold metrics statistically meaningless.
  - **Concession**: Adopted the standard low-data-regime alternative: **Leave-One-Video-Out (LOVO) cross-validation** for tuning `--min-peak-distance` (tune on 8 videos, evaluate held-out on the 9th, aggregate across all 9 folds). This prevents the final reported number from being the direct argmax over the same set it's evaluated on, while remaining feasible at this data scale. This caveat (LOVO used in place of a true held-out test set) will be explicitly stated in the eventual results report per Debater's requirement.

#### Experiment-specific critiques

- **Critique Addressed**: *Exp 1 — Variable Isolation* (`WARN`) — overlap conflates "context restoration" with "multi-view averaging" via `merge_close_boundaries`.
  - **Concession**: Fully conceded. Added **Step 0.5 (Shifted single-window control)**: a non-overlapping grid shifted +2.0s so edge-adjacent GT boundaries sit mid-chunk, with no overlapping predictions or merge logic involved — isolates the "context" mechanism cleanly from the "averaging" mechanism.

- **Critique Addressed**: *Exp 1 — Ablation Baseline* (`WARN`) — `merge-eps=0.35` untested under heavy load (baseline rarely exercised it).
  - **Concession**: Fully conceded. Added a manual sanity-check step: inspect raw pre-merge boundary lists for 3-5 overlap regions before trusting any F1 comparison.

- **Critique Addressed**: *Exp 2 — Complexity Test* (`WARN`) — new min-peak-distance code proposed without first testing the existing simpler `tools/sweep_diffgebd_threshold.py`.
  - **Concession**: Fully conceded. Added **Step 0 (Simple-baseline control)**: run the existing threshold-sweep tool first; only proceed with new suppression code if it Pareto-dominates the simple-threshold precision/recall trade-off (Occam's Razor).

- **Critique Addressed**: *Exp 3 — Ablation/Isolation* (`FAIL`) — widening `GAUS_SIGMA` confounds "boundary softness" with "positive:negative class balance shift."
  - **Concession**: Fully conceded, no rebuttal. **Experiment 3 (`exp_004_wider_gaussian_sigma`) is dropped from this iteration's batch entirely** rather than patched, given it was already the lowest-confidence (35%) and highest-cost (full retrain) hypothesis. Deferred to a future iteration with a `POS_WEIGHT`-style compensating control if Exp 1-2 leave a large residual gap.

- **Critique Addressed**: *Training Health Red Flag* (`FAIL`) — `train.log` shows non-monotonic/duplicate epoch numbers, baseline validity unverified.
  - **Concession**: Fully conceded. Added **Experiment 0** (`exp_000_training_health_audit`) as a blocking gate: parse the full concatenated training history, verify `model_best.pth` corresponds to the true global-best epoch, before trusting any Exp 1/2 comparison against this baseline.

#### Modified Experiments Summary

- **`exp_000_training_health_audit`** (NEW): Blocking gate, zero-cost, CPU-only log audit.
- **`exp_000b_aggregate_error_histograms`** (NEW): Blocking gate, zero-cost, CPU-only dataset-wide error distribution analysis.
- **`exp_002_val_overlap_context`** (MODIFIED): Added seed-variance noise-floor control (Step 0) and shifted-single-window isolation control (Step 0.5); added merge-logic sanity check; verification test now requires exceeding the noise floor.
- **`exp_003_min_peak_distance_suppression`** (MODIFIED): Added simple-threshold-sweep control (Step 0) before new code; switched from direct val-set sweep to Leave-One-Video-Out (LOVO) cross-validation; verification test now requires joint `cd12_chuyen1` + `cd12_chuyen2` reporting to catch worker-level overfitting.
- **`exp_004_wider_gaussian_sigma`** (CANCELLED): Dropped from this iteration's batch; deferred to future iteration with an isolation fix.

---

> [!NOTE]
> **Planner Handoff**: Plan revised to **Revision 2**, incorporating all Debater critiques: 4 `FAIL`-rated issues fully conceded and fixed (worker/station leakage → joint reporting requirement + future re-split recommendation; non-determinism → seed-variance control; threshold leakage → LOVO cross-validation; unverified training health → new blocking audit gate), plus all `WARN`-rated issues addressed (evidence sufficiency → aggregate histogram gate; variable isolation → shifted-window control; ablation baseline → merge-logic sanity check; complexity test → simple-threshold control-first). `exp_004_wider_gaussian_sigma` dropped entirely rather than patched. Handing back to `@research-debater` and Human for next review round.
