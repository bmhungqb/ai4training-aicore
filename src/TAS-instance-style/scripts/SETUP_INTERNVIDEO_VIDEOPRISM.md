# Setting up InternVideo and VideoPrism as chunk-feature backbones

`extract_chunk_features.py` already supports `videomae`, `mvit_v1_b`, `s3d`,
`i3d_r50` out of the box (no extra setup — all installed / auto-downloaded).
InternVideo and VideoPrism are **not** plug-and-play: neither has a simple
`pip install && AutoModel.from_pretrained(...)` path. This doc walks through
what's actually required for each, and the minimal integration code to wire
a working checkpoint into `build_<name>()` in `extract_chunk_features.py`.

---

## 1. InternVideo

There are two generations; both require manual repo setup (not just `pip install`):

- **InternVideo (v1)** — `OpenGVLab/InternVideo1.0` on HuggingFace. Gated repo
  (`gated: auto` — need to accept terms, ~instant approval). Checkpoints are
  raw `.ckpt`/`.pth` state dicts, not something `AutoModel` can load directly
  — you need the model class definitions from the official GitHub repo.
- **InternVideo2** — `OpenGVLab/InternVideo2-Stage1-1B-224p-K400` etc. Same
  story: gated, raw checkpoints, needs repo code. Also **1B+ parameters**
  (multi-GB download, needs a real GPU — not CPU-friendly).

### Step-by-step (InternVideo2-Stage1, video-only encoder, K400-finetuned)

1. **Request access** (one-time, usually auto-approved):
   - Go to https://huggingface.co/OpenGVLab/InternVideo2-Stage1-1B-224p-K400
   - Click "Agree and access repository", fill the gated-access form.
   - Log in locally: `huggingface-cli login` (paste a token from
     https://huggingface.co/settings/tokens).

2. **Clone the official repo** (for the model class — the checkpoint alone
   is just a state_dict, it has no architecture definition attached):
   ```bash
   cd /home/hungbm/ai4training/ai4training-aicore
   git clone https://github.com/OpenGVLab/InternVideo.git tmp_repos/InternVideo
   cd tmp_repos/InternVideo/InternVideo2/multi_modality
   pip install -r requirements.txt   # heavy: flash-attn, decord, deepspeed, etc.
   ```
   Note: `flash-attn` typically needs a matching CUDA toolkit + `ninja`; on a
   CPU-only / mismatched-CUDA machine this step is the most likely to fail.
   If it does, check the repo's issues for a `flash-attn`-free fallback
   config (some configs allow disabling flash-attn for the vision tower).

3. **Download the checkpoint**:
   ```bash
   huggingface-cli download OpenGVLab/InternVideo2-Stage1-1B-224p-K400 \
     1B_ft_k710_ft_k400_f8.pth --local-dir checkpoints/internvideo2_stage1
   ```

4. **Load the vision encoder only** (we don't need the text/retrieval heads,
   just the video tower, pooled to one vector per clip):
   ```python
   import sys
   sys.path.insert(0, "tmp_repos/InternVideo/InternVideo2/multi_modality")
   from models.backbones.internvideo2 import InternVideo2  # exact module path
                                                              # varies by repo
                                                              # version -- check
                                                              # the repo's own
                                                              # demo/config scripts
   # Build with the repo's own config loader (see demo.py / config.py in the
   # repo), load the .pth state_dict, then call the vision tower's
   # `forward_features` (or equivalent) method on an (B, C, T, H, W) clip of
   # f8 (8 frames) at 224x224 to get the pooled embedding (dim depends on
   # the specific checkpoint — f8 1B models are typically 1408-dim before
   # any projection head).
   ```
   The repo doesn't expose a clean one-line "give me embeddings" API — you
   have to follow their `demo.py` / `scripts/evaluation/*.py` to see exactly
   how they build the model and call `.encode_vision()` / similar.

5. **Wire it into `extract_chunk_features.py`**: add
   ```python
   def build_internvideo2(device):
       ...  # steps 2-4 above, return (infer_clip, feat_dim, clip_len=8, resize=224)
   ```
   and register it in `BACKBONES`.

**Reality check**: this is a multi-GB download + a non-trivial custom
dependency stack (flash-attn, deepspeed) typically meant for A100-class GPUs.
For a CPU-only or modest-GPU box, InternVideo2-1B is likely impractical —
consider it only if you have a real GPU (≥16GB VRAM) available.

**Lighter alternative**: `OpenGVLab/InternVideo1.0`'s `internvideomae_classification/vit_b_hybrid_pt_800e_k400_ft.pth`
is a plain VideoMAE-ViT-B checkpoint (not the CLIP/retrieval branch) — much
closer in size/complexity to the `videomae` backbone already in the script.
If the goal is "try something InternVideo-branded without the full repo",
that's the more realistic entry point: download the `.pth`, load it into a
standard `transformers.VideoMAEModel` architecture (if the state_dict keys
match — may need a small key-remapping script) rather than the full
InternVideo2 pipeline.

---

## 2. VideoPrism

VideoPrism (Google, 2024) has **no official PyTorch release** — the
[official repo](https://github.com/google-deepmind/videoprism) is
JAX/Flax-only. Two realistic paths:

### Option A — run it in JAX (official, more faithful, extra runtime)

1. ```bash
   cd /home/hungbm/ai4training/ai4training-aicore
   git clone https://github.com/google-deepmind/videoprism.git tmp_repos/videoprism
   cd tmp_repos/videoprream  # (sic: tmp_repos/videoprism)
   pip install -e .
   ```
   This pulls in `jax`, `flax`, `scenic` — a second deep-learning stack
   alongside PyTorch. On CPU-only JAX works but is slow; for GPU you need a
   JAX build matching your CUDA version (`pip install jax[cuda12]` family —
   check exact command on https://github.com/jax-ml/jax#installation,
   must match the CUDA toolkit already used by `torch==2.14.0+cu132`, i.e.
   CUDA 13.2 — JAX's CUDA wheels may lag behind this; verify compatibility
   before installing, or run VideoPrism on CPU only to avoid a conflicting
   second CUDA stack).

2. Download weights per the repo's `README.md` (typically a `gsutil` /
   direct-URL checkpoint, e.g. `videoprism_public_v1_base`), and follow its
   `videoprism/models.py` example to get an embedding for a `(T, H, W, 3)`
   clip (VideoPrism-Base uses 16 frames @ 288x288 IIRC — check the repo's
   model config for the exact clip_len/resolution).

3. In `extract_chunk_features.py`, this would be a **separate script**
   rather than a `build_<name>()` function, since it's a different
   framework (JAX, not torch) — `infer_clip()` would call into the JAX
   model and `.block_until_ready()` / `np.asarray()` the output before
   handing back to the shared `extract_chunks()` loop. Minimal shape:
   ```python
   import jax.numpy as jnp
   def build_videoprism_jax():
       from videoprism import models as vp
       flax_model, params = vp.get_model("videoprism_public_v1_base")
       apply_fn = vp.get_apply_fn(flax_model)  # check exact API in repo
       clip_len, resize, feat_dim = 16, 288, 768  # verify against repo config

       def infer_clip(clip_frames_rgb: list) -> "np.ndarray":
           x = jnp.asarray(np.stack(clip_frames_rgb))[None] / 255.0  # (1,T,H,W,3)
           embeddings, _ = apply_fn(params, x, train=False)
           return np.asarray(embeddings.mean(axis=1)[0])  # pool tokens -> (feat_dim,)

       return infer_clip, feat_dim, clip_len, resize
   ```
   (treat this as a starting skeleton — confirm exact function names/output
   shapes against whatever `videoprism/models.py` version you clone, APIs
   have changed between releases.)

### Option B — unofficial PyTorch ports (faster to integrate, less vetted)

Search GitHub for `videoprism pytorch` — as of now there are a few
community re-implementations/weight-port attempts (none official/merged
upstream). If you find one with a working weight-conversion script:
1. Clone it, convert the official JAX checkpoint to a `.pt` state_dict per
   its instructions.
2. Load the ported `nn.Module`, same pattern as the other `build_<name>()`
   functions in `extract_chunk_features.py` (resolve clip_len/resize from
   the port's config, pool to one vector per clip).
Caveat: numerical fidelity to the official JAX model is not guaranteed —
spot-check a few outputs against the JAX reference if correctness matters.

**Recommendation**: unless VideoPrism's specific pretraining (large-scale
web video, strong temporal modeling) is something you specifically need for
this garment-sewing dataset, I'd skip it — InternVideo2's 1B-scale download
and JAX/VideoPrism's separate framework stack are both heavy asks relative
to the 4 backbones already working (`videomae`, `mvit_v1_b`, `s3d`,
`i3d_r50`), which already span 2D-ViT-temporal, MViT, 3D-CNN (S3D), and
classic I3D architecture families.
