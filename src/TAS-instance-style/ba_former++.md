# BaFormer — Current Architecture & Pipeline

This document describes the **current logic** of the source code in `BaFormer/`, as actually implemented (not the aspirational design in `problem_definition.md`). It covers the end-to-end pipeline: config → dataset → model forward → loss → inference → metrics → training loop.

Reference entry points used while tracing the logic:
- `main.py` (training/validation loop, target construction, inference-time decoding)
- `action_segmentation/models/bk_fde_tde.py` (top-level `Network` module, registered under `model.name = bk_fde_tde`)
- `action_segmentation/models/frame_decoder/asformer_encoder.py` (frame-wise encoder, registered as `ASFormerEncoder`)
- `action_segmentation/models/backbone/asformer_model.py` (low-level ASFormer attention building blocks, shared by both backbone and frame_decoder registries)
- `action_segmentation/models/transformer_decoder/transformer_decoder_mask_bd_mulkv.py` (query-based mask+boundary decoder, registered as `TransformerDecoderMask_Boundary_MulKV`)
- `action_segmentation/models/matcher.py`, `action_segmentation/models/criterion_bd.py` (Hungarian matching + Mask2Former-style set loss + boundary BCE loss)
- `action_segmentation/datasets/datasets.py` (`InstanceDatasetFolder` for `tas_instance`, legacy `DatasetFolder` for `gtea/50salads/breakfast`)
- `action_segmentation/datasets/dataloader.py` (dataloader construction, batch_size=1 always)

---

## 1. High-Level Pipeline

```
Config (YAML + defaults.py)
        │
        ▼
Dataloader (batch_size=1, one video per sample)
        │   data: (1, C=2048, L)      frame_target: (1, L)     instances: (1, N, 3)  [tas_instance only]
        ▼
Network (bk_fde_tde.Network)
        │
        ├── frame_decoder = ASFormerEncoder   (frame_decoder/asformer_encoder.py)
        │       conv_1x1 → 10x dilated sliding-window self-attention layers (AttModule, from asformer_model.py)
        │       → outputs: class_logits, feature, mask_features, multi_features (list of per-layer features)
        │
        └── transformer_predictor = TransformerDecoderMask_Boundary_MulKV
                (query-based Mask2Former-style decoder; one query-feature cross-attn per ASFormer layer / "num_decode")
                → outputs: pred_logits (B,Q,C+1), pred_masks (B,Q,L), pred_boundarys (B,Q,L), aux_outputs[] (deep supervision)
        │
        ▼
Targets construction (main.py)
        │  tas_instance: prepare_target_from_instances() — ONE target mask/class PER GROUND-TRUTH INSTANCE
        │  legacy datasets: prepare_target() — re-derives segments by run-length splitting frame labels (cannot
        │                    separate same-class adjacent instances)
        ▼
SetCriterion_bd (criterion_bd.py)
        │  1. HungarianMatcher bipartite-matches predicted queries ↔ GT instances (cost = CE + focal + dice)
        │  2. loss_labels (CE with empty_weight for "no-object" queries + label smoothing)
        │  3. loss_masks  (sigmoid focal loss + dice loss on matched mask pairs)
        │  4. loss_boundarys (BCE with pos_weight, on ALL queries' boundary predictions, not just matched ones)
        │  5. Deep supervision: steps 1-4 repeated independently for every aux_outputs[i] (intermediate decoder layer)
        ▼
Weighted sum of losses → backward() → optimizer.step() (Adam, batch_size=1, no grad accumulation)
        │
        ▼
Inference decoding (inference_bd_peak, main.py)
        │  1. sigmoid(pred_boundarys) → threshold → find local peaks → segment the timeline into intervals
        │  2. for each interval, pick the query (among all Q queries) with the highest summed mask-sigmoid response
        │     as the "owner" of that interval → build a dense per-frame one-hot assignment (mask_ref)
        │  3. combine with softmax(pred_logits) class probabilities → per-frame class score map (seg_pred)
        │  4. optional _relabeling() smooths out short spurious segments (unused in the main val/train path shown,
        │     only used in `inference_bd_peak`'s own post-processing branch)
        ▼
Metrics (metrics.py, compute_metrics)
        │  frame-wise accuracy (argmax vs GT) + edit distance (Levenshtein over collapsed segments)
        │  + segmental F1 @ IoU 0.10 / 0.25 / 0.50 (f_score, with background class excluded)
        ▼
Logging / checkpointing (main.py train()/validate()/main())
        logs per-epoch to log.txt; tracks running max of val accuracy; saves `checkpoint_best` only when
        val accuracy improves (NOT when F1 improves)
```

---

## 2. Config System

- `action_segmentation/config/defaults.py` defines a `yacs`-based `ConfigNode` tree with sections: `dataset`, `model`, `train`, `optim`, `scheduler`, `validation`, `augmentation`, `test`.
- `main.load_config()`:
  1. Loads `get_default_config()`.
  2. Merges in a YAML file (`--config`, default `configs/framed_en_de.yaml`).
  3. Merges in CLI `options` overrides (`key value` pairs after the YAML path).
  4. If `--resume <dir>` is given, re-loads `<dir>/config.yaml` and forces `train.resume=True`.
  5. `update_config()` + `config.freeze()`.
- Key knobs relevant to the current architecture (`bk_fde_tde` / `final` experiments):
  - `dataset.name = tas_instance`, `dataset.n_classes`, `dataset.num_query` (number of decoder queries, e.g. 150), `dataset.pos_weight` (boundary BCE pos_weight), `dataset.threshold` (boundary peak threshold used at inference).
  - `model.name = bk_fde_tde` → selects which `Network` class is imported (`action_segmentation.models.bk_fde_tde`).
  - `model.ce_weight / mask_weight / dice_weight / bd_weight` → loss term weights.
  - `model.action_seg.frame_decoder.name = ASFormerEncoder` → selects the frame-wise encoder.
  - `model.action_seg.transformer_decoder.name = TransformerDecoderMask_Boundary_MulKV` → selects the query decoder.
  - `model.action_seg.transformer_decoder.deep_supervision = True` → auxiliary losses on every intermediate decoder layer.
  - `model.action_seg.backbone.name = None` → the separate "backbone" registry path is **unused** in this config; `ASFormerEncoder` under `frame_decoder` does all feature extraction directly from the raw (2048-d) I3D/feature input.

---

## 3. Dataset / Data Pipeline

Two dataset implementations exist; only `InstanceDatasetFolder` is used by the current `tas_instance` experiments.

### 3.1 `InstanceDatasetFolder` (`datasets.py`, used when `dataset.name == 'tas_instance'`)
- Reads `mapping.txt` (class name ↔ id), `splits/{train,val}.bundle` (video id lists), and per-video `features/<id>.npy` (C,T) + `annotations/<id>.json` (`num_frames`, `instances: [{start_frame,end_frame,class_id}, ...]`).
- `__getitem__`:
  - **train**: subsamples both feature and (derived) frame-wise target by `sample_rate`; also returns an `(N,3)` instance array `[start,end,class_id]` rescaled to the subsampled resolution; adds a random Gaussian `noise` tensor (same shape as features, used only if `dataset.noise_weight` is set).
  - **val**: feature is subsampled but `target`/`instances` are kept at **full resolution** — the model's output is later upsampled back (`main.validate()`, `repeat_interleave` / `F.interpolate`) to match before computing metrics.
- Note the instance target is built from per-instance JSON entries, NOT by re-deriving segments from a frame-wise label sequence — this is what allows (per `problem_definition.md`) two adjacent same-class instances to remain distinct ground-truth targets.

### 3.2 `DatasetFolder` (legacy, `gtea`/`50salads`/`breakfast`)
- Reads `splits/*.bundle`, `features/*.npy`, `groundTruth/*.txt` (frame-wise label text files) via `mapping.txt`.
- No instance JSON — segments are only available as frame-wise labels.

### 3.3 Dataloader (`dataloader.py`)
- `create_dataloader()` branches on `config.dataset.name == 'tas_instance'` to call `create_instance_dataset()`, otherwise the legacy `create_dataset()`/`create_dataset_all()`.
- **`batch_size` is hardcoded to 1** in all branches (both train and val) — each gradient step / each eval step processes exactly one full video sequence. `train.batch_size` in config is not actually used to batch multiple videos together.
- `shuffle=True` for train, `False` for val. Supports `DistributedSampler` if `torch.distributed` is initialized.

---

## 4. Model Forward Pass (`bk_fde_tde.Network`)

```python
class Network(nn.Module):
    def __init__(self, backbone, frame_decoder, transformer_predictor):
        self.backbone = backbone                  # None in current config
        self.frame_decoder = frame_decoder         # ASFormerEncoder
        self.predictor = transformer_predictor     # TransformerDecoderMask_Boundary_MulKV

    def forward(self, x):                          # x: (1, 2048, L)
        mask = torch.ones_like(x)                  # all-ones mask (no real padding handling for bs=1)
        frame_out = self.frame_decoder(x, mask)
        outputs = self.predictor(frame_out['multi_features'], frame_out['mask_features'], mask=None)
        return outputs
```

### 4.1 `ASFormerEncoder` (frame_decoder) — per-frame feature extraction
- `conv_1x1`: projects raw 2048-d input features down to `embed_dim` (e.g. 64).
- Optional channel dropout (`channel_masking_rate`, dataset-dependent: 0.5 for gtea else 0.3) applied before the conv.
- `num_layers=10` stacked `AttModule` blocks (from `backbone/asformer_model.py`), each with:
  - dilated `ConvFeedForward` (dilation = `2**i`, so receptive field grows exponentially across the 10 layers),
  - `InstanceNorm1d` → `AttLayer` self-attention with `att_type='sliding_att'` (a windowed/local self-attention variant that reshapes the sequence into overlapping blocks of size `bl = 2**i` and applies scaled dot-product attention within each window, using a precomputed `window_mask`),
  - residual connections and a final `1x1 conv` + dropout, gated by the (all-ones) mask.
- Every layer's output is collected into `multi_features` (list of `num_layers` tensors) — these become the per-layer memory/key-value source for each of the transformer decoder's `num_decode` cross-attention stages (see below).
- `mask_features = Conv1d(k=3)` applied to the **final** ASFormer layer output — this is the single shared feature map against which every query's temporal mask is computed (via dot product in the decoder).
- `class_logits = conv_out(feature)` is also produced for completeness but **not actually consumed anywhere downstream** in this registered `frame_decoder` version (the final classification happens entirely through the query-based decoder's `pred_logits`, not this frame-wise head).

### 4.2 `TransformerDecoderMask_Boundary_MulKV` (transformer_predictor) — query-based set prediction
This is a Mask2Former/DETR-style decoder adapted to 1-D temporal sequences, with an added boundary-prediction head:

- **Learned queries**: `self.query` is a single learned `(1, num_queries, hidden_dim)` parameter (e.g. 150 queries), with a sinusoidal-style `PositionalEncoding` added (`self.pe`).
- **Multi-KV cross-attention across ASFormer layers** ("MulKV" in the class name): `num_decode = dec_layers` (e.g. 10, matching the frame_decoder's layer count). For stage `i`, the query features cross-attend to `src[i] = input_proj[i](multi_features[i])` — i.e. **each successive decoder stage attends to a different ASFormer layer's features**, from shallow to deep, rather than only the final layer's output as in standard Mask2Former.
- Each `Transformer_decoder_layer` = cross-attention (query → ASFormer layer-i features, with a soft attention-mask gate from the previous stage's predicted mask) → self-attention (query ↔ query) → feed-forward (pre-norm residual blocks throughout, GELU activations).
- **`forward_prediction_heads`** (called once before the loop for stage-0 predictions, then once per decoder stage):
  - `class_embed`: Linear → `(B, Q, num_classes+1)` logits (the `+1` is the "no object"/empty class for unmatched queries).
  - `mask_embed`: 3-layer MLP → mask embedding vector per query, dotted (`einsum`) with the shared `mask_features` → `pred_masks (B,Q,L)` (raw logits, sigmoid applied later for actual probabilities).
  - **Boundary head** (the "Bd" in `TransformerDecoderMask_Boundary_MulKV`): an MLP-Mixer-like pooling across the query dimension (`vid_embed_before` MLP → `vid_embed: Linear(num_queries → 1)` applied along the query axis) collapses all Q queries into a single "video-level" embedding per spatial position, which is then dotted with `mask_features` to produce `pred_boundarys (B,1,L)` — i.e. **boundary prediction is a single shared curve over time, pooled across all queries, not per-query**.
  - `attn_mask`: the current stage's predicted mask (sigmoided, broadcast over attention heads) is fed into the **next** stage's cross-attention as a soft gating mask — this is the "masked attention" mechanism from Mask2Former, implementing iterative mask refinement across stages.
- **Deep supervision**: predictions from every intermediate stage (`predictions_class[:-1]`, etc.) are packed into `aux_outputs`, and the criterion computes the full loss independently on each of them in addition to the final stage.
- Final output dict: `{pred_logits, pred_masks, pred_boundarys, aux_outputs: [...]}`.

---

## 5. Target Construction (`main.py`)

Two parallel target-building functions exist, selected by `is_instance_dataset = (config.dataset.name == 'tas_instance')`:

- **`prepare_target_from_instances(instances, seq_len, sigma)`** (current/instance-aware path): builds one `(mask, class, boundary-contribution)` tuple **per ground-truth instance row** directly from the `(start,end,class_id)` array loaded from the dataset's JSON annotations. Same-class adjacent instances are naturally kept as separate target rows because they come from separate JSON entries, not from re-splitting a frame-wise label sequence. `frame_bd[start]=1` is set for every instance except the first (boundary = instance start, including same-class boundaries).
- **`prepare_target(frame_tgs, sigma)`** (legacy path, used for `gtea/50salads/breakfast`): walks the frame-wise label sequence and starts a new segment whenever `frame_tgs[j] != frame_tgs[j-1]` — this **cannot** separate two consecutive same-class instances (they'd be silently merged into one segment), which is exactly the limitation `problem_definition.md` identifies and that `InstanceDatasetFolder` + `prepare_target_from_instances` was built to fix.
- `create_heatmap(sequence, sigma)`: if `dataset.guassian_sigma` is set (currently `None` in the analyzed config), hard 0/1 masks are converted into soft Gaussian-bump heatmaps centered on the segment; otherwise masks stay binary.
- Both paths output a `targets = [target]` list with keys `labels` (class ids), `masks` (binary/heatmap masks per instance), `boundarys` (single dense 0/1 vector over time).

---

## 6. Matching & Loss (`matcher.py`, `criterion_bd.py`)

### 6.1 `HungarianMatcher`
For each sample (batch size always 1 here), builds a cost matrix `(num_queries × num_gt_instances)`:
```
C = cost_mask * batch_sigmoid_focal_loss(pred_mask, gt_mask)
  + cost_class * (-softmax(pred_logits)[:, gt_class])
  + cost_dice  * batch_dice_loss(pred_mask, gt_mask)
```
Solved via `scipy.optimize.linear_sum_assignment` → one-to-one assignment between predicted queries and GT instances (unmatched queries are implicitly assigned to the "no object" class).

### 6.2 `SetCriterion_bd.forward`
1. Matches on the **final-stage** outputs (`outputs_without_aux`).
2. Computes `num_masks` = total GT instance count (for loss normalization), clamped to ≥1.
3. For each loss in `['labels', 'masks', 'boundarys']`, computes:
   - **`loss_labels`**: cross-entropy over `(num_classes+1)` logits, matched queries get their GT class id, all unmatched queries get the "no object" class (`num_classes`), weighted by `empty_weight` (down-weights the no-object class via `eos_coef`), with label smoothing (`augmentation.label_smoothing_epsilon`).
   - **`loss_masks`**: on the matched (query,instance) pairs only — `sigmoid_focal_loss` (RetinaNet-style, γ=2, α=0.25) + `dice_loss`, both normalized by `num_masks`.
   - **`loss_boundarys`**: BCE-with-logits between the model's single pooled `pred_boundarys` curve and the dense GT `boundarys` vector, using a **per-batch dynamic `pos_weight`** = `target_boundarys * pos_weight * pos_weight_ratio` (if `pos_weight` is a list → `use_dynamic=True`, loads dataset-wide boundary-class imbalance ratio from a cached CSV) or a flat scalar `config.dataset.pos_weight` otherwise — this heavily up-weights the rare positive (boundary) frames.
4. **Deep supervision loop**: for every `aux_outputs[i]` (one per intermediate decoder stage), re-runs the Hungarian matcher **independently** (predictions can match differently at each stage) and recomputes all three losses, suffixed `_i`.
5. Returns a flat dict of all (main + auxiliary) loss terms; `main.py`'s `train()`/`validate()` then multiplies each by its `weight_dict` entry (`ce_weight`, `mask_weight`, `dice_weight`, `bd_weight`, replicated across aux indices when `deep_supervision=True`) and sums them into the single scalar optimized by `loss.backward()`.

Note: in `validate()`, the boundary loss term is excluded from `weight_dict` (`weight_dict` there only lists `loss_ce/loss_mask/loss_dice`), so `loss_bd*` terms computed by the criterion are silently dropped (`losses.pop(k)`) from the reported validation loss, even though `loss_boundarys` is still computed internally by `SetCriterion_bd` — the logged val loss is therefore not perfectly comparable to the train loss (which does include `loss_bd`).

---

## 7. Inference-Time Decoding

Used identically in both `train()` (for logging training-set running metrics) and `validate()`:

- **`inference_bd_peak(outputs, threshold)`** (the actual decoding function wired into the training/val loop — `inference()` and `inference_bd()` are older/unused alternatives left in the file):
  1. Sigmoid the pooled boundary curve, zero out values below `threshold` (`dataset.threshold`, e.g. 0.2).
  2. Find local peaks (`bd[i-1] < bd[i] > bd[i+1]`) in the thresholded curve → these peak positions, plus the sequence start/end, define a partition of the timeline into contiguous intervals.
  3. For each interval, among **all** `num_queries` queries' predicted masks, pick the query whose summed sigmoid-mask response is highest over that interval, and assign every frame in the interval to that query (`mask_ref`, a hard one-hot-per-query dense assignment) — this is how instance separation is actually realized at inference: **boundaries from the pooled boundary head cut the timeline, and per-interval the best-matching query decides which instance "owns" that interval**.
  4. Combine `mask_ref` with `softmax(pred_logits)[..., :-1]` (dropping the no-object class) via einsum to get a dense per-frame class-probability curve `seg_pred` — this is what frame-wise metrics are computed against.
  5. `_relabeling(outputs, theta_t=100)` is applied only inside `inference_bd_peak` (smooths out predicted segments shorter than `theta_t` frames by overwriting them with the preceding segment's prediction) — a hand-tuned temporal-smoothing post-process.
- In `train()`: `ins_seg_pred = inference_bd_peak(...)`, `seg_pred = ins_seg_pred.softmax(dim=-1)` is used purely for **running training-metric logging** (loss itself does not depend on this decoding — gradients flow only through the Hungarian-matched set loss).
- In `validate()`: same decoding, plus (if `dataset.sample_rate != 1`) the raw model outputs are first upsampled (`repeat_interleave` for masks, `F.interpolate` for the boundary curve) back to the full frame-rate length before decoding, to match the full-resolution validation targets.

---

## 8. Metrics (`utils/metrics.py`)

- **`accuracy(pred, targets)`**: plain per-frame `argmax == target` mean — a frame-wise accuracy metric, computed on the dense `seg_pred`/`ins_seg_pred` from the decoding above, not instance-aware.
- **`edit_score`**: run-length-collapses both prediction and GT (excluding background class id per-dataset) into ordered label sequences, then computes a normalized Levenshtein edit distance between them (`100 * (1 - dist/max_len)`).
- **`f_score(dataset, recognized, ground_truth, overlap)`**: segmental F1 — collapses pred/GT into `(label,start,end)` segments, computes IoU between every predicted segment and all GT segments of the same label, counts a hit (`tp`) if `IoU ≥ overlap` and that GT segment hasn't already been claimed by an earlier/better-matching prediction; leftover unmatched predictions are `fp`, unmatched GT are `fn`. Called at `overlap ∈ {0.10, 0.25, 0.50}` → logged as `f1@10/25/50`.
- **`compute_metrics`**: wraps all of the above plus `num_correct` (raw count, for `AverageMeter_acc` weighted averaging) into a single call; invoked once per sample for both the **categorical (`ins_seg_pred`)** decoding and, separately, the **semantic** decoding (variable naming in `Meter_dict.get_update_metric` — in the current code both `seg_pred` and `ins_seg_pred` actually originate from the same `inference_bd_peak` call, so `acc_meter` and `ins_acc_meter` end up numerically identical in this config, as seen in the logs where `acc% == ins_acc%` on every line).

---

## 9. Training / Validation Loop (`main.py`)

### 9.1 `train(epoch, ...)`
- One pass over the train dataloader (1 video per step, `step` counts batches not videos-per-batch since `batch_size=1`).
- Per step: build targets → forward → compute weighted loss dict → `loss.backward()` → `optimizer.step()` → decode (`inference_bd_peak`) → update running meters (loss/acc/edit/f1 via `AverageMeter*`) → log every `log_period` steps or at the last step of the epoch.
- `scheduler.step()` called **once per epoch** (not per step) — consistent with `scheduler.type = multistep` with epoch-indexed `milestones`.
- `config.augmentation.is_use` (default False in analyzed config) optionally applies `augment_crop` to the input before anything else; `config.dataset.noise_weight` (also `None` here) optionally adds the precomputed Gaussian noise tensor to the input features.

### 9.2 `validate(epoch, ...)`
- `@torch.no_grad()`, `model.eval()`.
- Same target-building + forward + loss-dict machinery as train, but gradients aren't computed and the boundary loss term is excluded from the reported total (see §6.2 caveat).
- Handles `sample_rate != 1` upsampling before decoding (see §7).
- `val_record.update_record(epoch, acc_avg, edit_avg, f1_avg)`: tracks a running best based **solely on `acc_avg`** (frame accuracy) — `edit_max`/`f1_max` are only updated as a side-effect of whichever epoch had the best accuracy, **not** independently tracked best-F1 epochs. This explains log lines where `f1@1_max` can appear "stuck" even while instantaneous `f1@10` moves — the "_max" trackers only update together, gated by accuracy improving.
- Logs a single consolidated line per epoch with both instantaneous and running-max values for acc/edit/f1.

### 9.3 `main()`
- Builds output dir `experiments/<dataset.name>/<model.name>/<model.note>/<dataset.split>/`.
- Saves `config.yaml`, `env.yaml`, and a config-diff file (`find_config_diff`) on first run (not on resume).
- Builds train/val dataloaders, model, optimizer (`create_optimizer`), scheduler (`create_scheduler`, steps_per_epoch-aware), and a `fvcore` `Checkpointer`.
- Resume logic: `--resume <dir>` restores model/optimizer/scheduler state + `global_step`/`start_epoch` from the checkpoint's saved config; otherwise can load a bare `train.checkpoint` state dict (weights only, no optimizer/epoch state).
- Main loop: for each epoch, `train()` then (if `epoch % val_period == 0`) `validate()`; saves `checkpoint_best` **only when validate() reports `if_update=True`** (i.e. only on new best-accuracy epochs) — note the checkpoint-saving code references `if_update` outside the `if val_period>0` guard, so if validation is skipped on a given epoch, it reuses the previous epoch's `if_update` value (a pre-existing latent bug: a checkpoint could be (re-)saved on a non-validated epoch if the prior epoch happened to have set `if_update=True`).

---

## 10. Summary: What Makes This "BaFormer" (vs. plain Mask2Former/MaskFormer-style TAS)

1. **Multi-KV decoder** — instead of cross-attending only to the backbone's final feature map, each of the `dec_layers` decoder stages attends to a *different* ASFormer encoder layer's intermediate features (shallow→deep), giving the query decoder access to multiple temporal receptive fields across its stages.
2. **Explicit boundary head** — a pooled (query-mixed) single boundary curve is predicted and supervised with a heavily pos-weighted BCE loss, independent of the per-query mask/class heads, and used at inference time to literally cut the sequence into pieces before query-mask-based instance assignment (`inference_bd_peak`).
3. **Instance-aware targets** (`tas_instance` dataset path only) — ground truth is built directly from per-instance JSON annotations (`prepare_target_from_instances`) rather than by re-deriving segments from a frame-wise label sequence, which is what in principle allows the Hungarian matcher to keep two same-class adjacent instances as separate matchable targets (per the requirement in `problem_definition.md`).
4. **Deep supervision with independent re-matching** at every decoder stage (not just loss on the final layer), following Mask2Former's recipe.
5. At inference, the actual "instance separation" is **not** done by the Hungarian-matched queries directly (that only governs training loss) — it is reconstructed post-hoc by (a) thresholding+peak-finding on the single boundary curve to get candidate cut points, then (b) greedily assigning the best-scoring query to each resulting interval. This decoupling between how instances are *trained* (per-query set loss) and how they are *decoded* (boundary-cut + per-interval query argmax) is a notable asymmetry in the current pipeline worth flagging for future review.
