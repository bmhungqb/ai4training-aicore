# Plan: Instance-style query matching for TQT (fix consecutive same-category action merging at the architecture level)

## Context / background

TQT (`src/TAS/TQT/`) is a Mask2Former-derived timestamp-supervised TAS method.
Its decoder (`decoder.py`, `TemporalMaskFormer_V1`..`V5`) uses a **fixed
number of queries, one per class** (`num_queries = num_classes`, set in
`model.py` lines 91/115). Each query has a **permanent 1:1 binding to one
category** — e.g. query index 1 is *always* "Sewing/Joining", for the
entire video, no matter how many separate sewing operations occur in it.

This architecture is "semantic-style" (like semantic segmentation: every
frame gets a class label, but same-class regions are indistinguishable from
each other), not "instance-style" (like instance/panoptic segmentation:
each individual object/occurrence gets its own slot/query, even if two
objects share the same class).

This document plans converting TQT's decoder usage from semantic-style to
instance-style, to solve — at the root, architecturally — the problem
that two consecutive real operations of the same coarse category
currently cannot be told apart by the model.

## Related prior work in this repo (context, not superseded)

Two other fixes were already implemented for the *other* 2 TAS methods
(MS-TCN) as part of this same investigation — this plan is a 3rd,
independent approach, specific to TQT's architecture:

- **Fix A** (`tools/prepare_tas_dataset.py`): emits
  `dataset_tas/boundaries/<video>.txt`, fine-grained instance-transition
  labels as a side-channel. Data-level only, doesn't change `groundTruth/`.
- **Fix B** (`src/TAS/ms-tcn/model.py`, `--use_boundary`): MS-TCN auxiliary
  ASRF-style boundary-detection head, trained on Fix A's labels, used to
  re-segment the classifier's output at inference. Measured recall on
  same-category transitions: **21.8%** — mechanism works (doesn't rely on
  category-change to find boundaries) but not yet accurate enough for
  production use.
- **Fix A+C** (`tools/prepare_tas_dataset.py --split-repeated`): rewrites
  `groundTruth/` labels directly, alternating `#A`/`#B` suffix on repeated
  same-category segments so the label string itself never repeats across
  adjacent same-category segments. Measured recall: **13.2%** (worse than
  Fix B) — doubling the label space (5→9 classes) made the learning
  problem harder without adding data, net negative so far.

Both prior fixes top out well under 25% recall on the specific "same
category, consecutive" case. This plan targets a different, arguably more
correct mechanism: make the **architecture itself** structurally capable of
representing multiple instances of the same class, instead of trying to
patch semantic-style output after the fact (Fix B) or force category
strings to differ (Fix A+C).

---

## 1. Issues (what's broken, confirmed by reading the code + measuring the data)

### Issue 1 — `num_queries == num_classes` is a hard structural bottleneck
`model.py`:
```python
self.decoder = TemporalMaskFormer_V1(num_queries=num_classes, ...)   # line 91
self.decoder = action_map[args.decoder](num_queries=num_classes, ...) # line 115
```
With `dataset_tas` having 5 classes, there are only 5 query slots total.
If a video has 3 separate "Sewing/Joining" operations, all 3 are forced to
share query index 1 (whatever index "Sewing/Joining" is assigned to) —
there is no query slot left to represent "the 2nd/3rd occurrence of
sewing" as something distinct. This is true regardless of training data
quality or training duration — it's a capacity limit of the architecture.

### Issue 2 — No Hungarian matching / set prediction; assignment is static
`get_q_losses()` (`model.py` line 887) does:
```python
mask_logits, class_logits = self.model.decoder(memory)   # [1, num_queries, T]
loss = self.ce(mask_logits, batch_target)                 # query i forced = class i, always
```
This is cross-entropy against a **fixed** query→class assignment — not the
bipartite matching DETR/Mask2Former instance-mode uses (match each
ground-truth instance to whichever query best predicts it, per-video,
dynamically). Confirmed by reading `get_q_losses`/`get_q_losses2`/
`get_q_losses3` (`model.py` lines 887-1030) — none of them implement a
matching step.

### Issue 3 — Existing timestamp annotations are built from *merged* groundTruth
`TQT/prepare_data.py`:
```python
def segments(labels):          # labels = dataset_tas/groundTruth/*.txt (coarse, 5-class)
    ...
def sample_timestamps(gt_path, seed=0):
    idx = [rng.randrange(s, e) for s, e in segments(labels)]   # 1 timestamp per COARSE segment
```
Measured directly: `cd9_chuyen3.txt` → only **20 timestamps** generated,
matching the 20 *coarse* (merged) segments, not the 62 real fine-grained
instances. This means even if the architecture is fixed (Issues 1+2), the
*training signal* currently available still can't teach the model about
instances that got merged away before timestamp sampling — this has to be
regenerated from the fine-grained source (`step_segments_clean.json`, via
Fix A's logic in `tools/prepare_tas_dataset.py`) BEFORE any instance-style
training can be meaningfully tested.

### Issue 4 — Dataset size is small for DETR-style matching to train stably
Measured directly (`data/cd*/chuyen*/step_segments_clean.json`, 44 videos):

| | Max/video | Mean | P90 |
|---|---|---|---|
| Fine-grained instances | 138 | 51.7 | 109 |
| Coarse-category instances (post Fix-A, unmerged) | 77 | 25.0 | 58 |

DETR/Mask2Former's original papers train matching on datasets with
hundreds of thousands of images; this is a known convergence-stability
risk, not just a capacity-sizing question. Called out explicitly as a risk
in section 4, not assumed away.

### Issue 5 — No evaluation harness exists for instance-level output
`eval.py`'s `f_score`/`levenstein` (lines 36-90) operate on frame-level
*category* sequences (the final post-matching, flattened output), which is
fine for the existing "coarse 5-class F1@IoU" metric but says nothing
about whether individual same-category instances were correctly separated.
Fix B's validation already needed a bespoke boundary-recall script (not
part of `eval.py`) for this reason — the same gap applies here and is
scoped into this plan (Phase 4).

---

## 2. Target (what "done" means)

### Primary target
Train a TQT variant (`decoder` with `num_queries >> num_classes`, trained
with Hungarian-matched set prediction loss) and measure, on the same 9-video
val split used throughout this investigation:

- **Same-category-transition recall** (identical metric used for Fix B:
  does the model detect the true instance boundary, within ±5 frames,
  when two consecutive ground-truth instances share a coarse category) —
  target: **meaningfully above Fix B's 21.8%** (stretch target: >50%,
  since this mechanism is structurally capable of it, unlike Fix B/Fix
  A+C's approaches).
- **Coarse 5-class F1@IoU / Acc / Edit** (standard metric, for
  comparability with MS-TCN/FACT/TQT-semantic baselines already measured)
  — target: not meaningfully worse than current TQT-semantic numbers
  (Acc 75.4%, F1@0.50 55.6% — though note that number benefits from using
  ground-truth timestamps at inference, see caveat in `TQT/RUN.md`; a fair
  comparison point is more subtle, see Phase 4).

### Secondary target
A reusable instance-level evaluation script (Phase 4) that can also be
pointed at Fix B's output, enabling a true apples-to-apples comparison of
all 3 approaches (Fix B / Fix A+C / this plan) on the same metric.

### Explicit non-goals
- Not attempting to also convert FACT or MS-TCN to instance-style in this
  plan (TQT's decoder already has the right shape for this; MS-TCN's
  architecture does not have a query mechanism to repurpose — a different
  plan would be needed there).
- Not changing TQT's timestamp-supervision premise (still trains from
  sparse per-instance timestamps, not dense masks) — matching cost will be
  computed from single timestamps + their known class, not full masks.

---

## 3. Plan (phased, each phase independently checkpointable)

### Phase 0 — Fix the timestamp-annotation generation gap (Issue 3)
Prerequisite for everything else; otherwise instance-style training would
be trained on the same merged/incomplete signal as today.

- Modify `TQT/prepare_data.py`'s `sample_timestamps()` to source segments
  from the **fine-grained** `step_segments_clean.json` (same source Fix A
  reads), not from merged `groundTruth/*.txt`. Each sampled timestamp
  keeps its *coarse* class label (for the existing CE loss term) but now
  one timestamp exists per real instance, including same-category
  neighbors.
- Verify: `cd9_chuyen3.txt` should get 62 timestamps (not 20).
- This alone does **not** fix the merging problem (Issues 1+2 are
  architectural), but is required before Phase 2 can be evaluated
  meaningfully.

### Phase 1 — Decoder/model capacity change (Issue 1)
- Add a new CLI-exposed hyperparameter `--num_queries` (default: fall back
  to `num_classes` to preserve current behavior exactly when unset).
- `model.py` lines 91/115: change
  `TemporalMaskFormer_V1(num_queries=num_classes, ...)` →
  `TemporalMaskFormer_V1(num_queries=args.num_queries, ...)`.
- `decoder.py` requires **no changes** — `TemporalMaskFormer_V1..V5`
  already accept arbitrary `num_queries` and produce
  `[B, num_queries, T]` mask_logits / `[B, num_queries, num_classes]`
  class_logits shaped outputs regardless of the query count (confirmed by
  reading the forward() methods — the only place `num_queries` is used is
  `nn.Embedding(num_queries, d_model)`).
- Set `num_queries` from the Issue-4 data measurement: **80** (P90 of 58
  rounded up with margin, capped below the max of 77 + slack for edge
  cases/longer videos in a larger dataset later).
- `class_head` output dimension changes from `num_queries` (current,
  accidentally-coupled-to-num_classes bug worth noting) to the *actual*
  `num_classes + 1` (the `+1` is the new "no-object" class — see Phase 2).

### Phase 2 — Hungarian matching + set-prediction loss (Issue 2)
This is the substantial new code (~150-200 lines estimated, no equivalent
exists in the repo today):

- New module `TQT/matcher.py`:
  - Input per video: predicted `(class_logits, mask_logits)` from the
    decoder for all `num_queries` queries, and ground-truth instances
    (from Phase 0's per-instance timestamps: each instance = 1 known
    class + 1 known representative frame index).
  - Cost matrix: for each (query, gt-instance) pair, cost =
    `-log P(class_logits[query] == gt_instance.class)` +
    `-mask_logits[query, gt_instance.timestamp]` (encourage the matched
    query's mask to peak at the known instance timestamp). Weights for
    the two cost terms are a tunable hyperparameter (start 1:1, matching
    DETR's original ratio choice process).
  - Solve via `scipy.optimize.linear_sum_assignment` (standard DETR
    approach, no need to hand-roll Hungarian algorithm).
  - Output: a list of (query_idx, gt_instance_idx) pairs; all unmatched
    queries get the implicit "no-object" class target.
- New loss function in `model.py` (new method, e.g. `get_q_losses_instance`,
  added alongside — not replacing — `get_q_losses`/`get_q_losses2`/
  `get_q_losses3`, consistent with this codebase's existing pattern of
  keeping prior variants intact):
  - Matched queries: CE(predicted_class, gt_class) + a mask-localization
    term (BCE or focal loss between `mask_logits[query]` and a target
    that peaks at the gt instance's timestamp — since we only have sparse
    timestamp supervision, not a dense mask, this cannot be full Dice
    loss like image Mask2Former; closer to the existing
    `confidence_loss`/boundary-style losses already in `model.py`).
  - Unmatched queries: CE(predicted_class, "no-object"), down-weighted
    (standard DETR practice — e.g. 0.1x weight — since most queries are
    unmatched by construction when `num_queries ≫ num_instances`).
- New training action mode in `main.py`/`main7_1.py` (e.g.
  `--action train_instance`), reusing the existing pretrain/data-loading
  scaffolding, swapping only the loss call — minimizes risk of breaking
  the already-verified existing action modes.

### Phase 3 — Inference / decoding for instance-style output
- At inference: take the queries whose predicted class ≠ "no-object"
  (above a confidence threshold, tunable), each contributes a predicted
  (class, mask) pair.
- Resolve overlapping/competing query masks into a single frame-level
  label sequence (needed to compute the standard frame-level F1@IoU for
  comparability) — e.g. per-frame argmax over all "active" queries'
  mask_logits at that frame, consistent with how Mask2Former flattens
  instance predictions into a semantic map for evaluation.
- This flattened sequence is what gets written to
  `results/tas/<split>/<video>` in the same format `export_best_predictions.py`
  already uses, so it plugs into the existing `visualize_predictions.py`
  pipeline with no changes needed there.

### Phase 4 — Evaluation harness (Issue 5)
- Extend the same boundary-recall methodology used for Fix B (script
  logic already proven in this investigation) into a reusable script
  `src/TAS/common/eval_instance_boundaries.py`:
  - Input: any method's frame-level prediction + `dataset_tas/boundaries/`
    (Fix A's fine-grained transition ground truth, already exists for all
    44 videos).
  - Output: same-category-transition recall (the key metric from Phase 0's
    target), plus the split-out "different-category" recall for context
    (as computed for Fix B: 21.8% same vs 25.0% different).
  - Must be runnable against Fix B's existing predictions too (regression
    check — reproduce the already-measured 21.8% as a sanity check that
    the new generic script matches the bespoke one-off script's logic).
- Run standard `eval.py` F1@IoU/Edit/Acc on the Phase 3 flattened output,
  for comparability with MS-TCN/FACT/TQT-semantic numbers already in
  `outputs/tas_viz/README.md`.

### Phase 5 — Train, measure, compare, document
- Train on the same 35/9 train/val split used throughout.
- Report both metrics from Phase 4 against: MS-TCN baseline, MS-TCN+Fix B,
  TQT-semantic (current), TQT-instance (this plan).
- Update `src/TAS/README.md` "Known issue" section with the 3-way (now
  4-way) comparison and a final recommendation.
- If recall target is met: write `TQT/RUN.md` instructions for the new
  `--action train_instance` mode, consistent with existing RUN.md style
  (exact copy-pasteable commands, documented quirks).
- If recall target is NOT met: document why (most likely candidate per
  Issue 4: insufficient data for matching to converge) and record the
  measured number + diagnosis rather than silently dropping the approach,
  consistent with how Fix B/Fix A+C's shortfalls were documented rather
  than hidden.

---

## 4. Risks / open questions (flagged up front, not discovered mid-implementation)

- **Data scale (Issue 4)**: 44 videos / 35 train is small for Hungarian
  matching to learn a stable query-to-instance assignment strategy. DETR
  literature generally needs large data + long training (500 epochs on
  COCO) for matching to stabilize. Mitigation to try if initial results
  are unstable: lower `num_queries` closer to the P90 (58) rather than
  the max, add matching-cost warmup (start with class-only cost, add mask
  cost after N epochs), and/or borrow DETR's auxiliary loss on every
  decoder layer's output (the `V3`/`V4` decoder variants already return
  per-layer logits — `mask_logits_list`/`class_logits_list` — which is a
  convenient existing hook for this without new code).
- **Mask supervision is sparse, not dense**: unlike image Mask2Former
  (dense pixel masks), only 1 timestamp per instance is known, so the
  mask loss term is weaker than typical DETR instance segmentation. This
  is inherent to timestamp-supervised TAS and not fixable by more
  engineering — expectations should be calibrated accordingly (see
  "stretch target" framing in section 2, not a hard requirement).
- **"No-object" class imbalance**: with `num_queries=80` and typically
  ~25 real instances/video, ~55 queries per video are "no-object" by
  construction — standard DETR down-weighting should handle this but is a
  known tuning-sensitive hyperparameter (DETR paper settled on 0.1 after
  experimentation; may differ for this smaller/different-domain setup).
- **Fair comparison caveat carries over**: as already noted in
  `TQT/RUN.md` and `outputs/tas_viz/README.md`, any TQT variant's
  F1@IoU numbers are not directly comparable to MS-TCN/FACT's, since TQT
  uses known timestamp labels at inference (a real, not synthetic,
  advantage baked into the paradigm) — this plan does not change that
  caveat, Phase 5's comparison table must keep repeating it.
