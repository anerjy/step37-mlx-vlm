---
library_name: mlx
pipeline_tag: image-text-to-text
license: apache-2.0
base_model:
- stepfun-ai/Step-3.7-Flash
tags:
- mlx
- mlx-vlm
- mlx-lm
- step3p7
- vision-language
- multimodal
- apple-silicon
---

# Step 3.7 Flash — MLX Adapter (text + vision)

First MLX-native runtime adapter for [`stepfun-ai/Step-3.7-Flash`](https://huggingface.co/stepfun-ai/Step-3.7-Flash) — Step Robotics' multimodal model based on the step3p5 MoE backbone (Qwen3.6-A3B family) with a 47-layer PerceptionEncoder vision tower.

Lets you run Step 3.7 Flash on **Apple Silicon** through [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm) (multimodal, recommended) or [`mlx_lm`](https://github.com/ml-explore/mlx-lm) (text-only, fallback).

This is the **adapter code only** — no model weights are shipped here. You download or quantize Step-3.7-Flash separately (instructions below).

## What this includes

| Path | Purpose |
|---|---|
| `mlx_vlm/models/step3p7/` | mlx-vlm adapter — 6 files, multimodal (text + vision) |
| `mlx_lm/models/step3p7.py` | mlx-lm adapter — text-only |
| `model_dir_patches/chat_template.jinja` | patched Jinja template (multi-crop image expansion + `reasoning_effort=none`) — drop into your model dir |
| `scripts/extract_vision_weights.py` | reproducer for the vision-shard pre-transpose step (needed for MLX NHWC convention) |

## Architecture

Step 3.7 Flash = step3p5 backbone (45-layer Qwen3.6-A3B MoE, 12 full-attn + 33 sliding-window=512 layers) + 47-layer **PerceptionEncoder** ViT + 2 stride-2 Conv2d downsamplers + Linear(6144→4096) projector.

Vision pipeline:
1. **ImagePatcher** (pure PIL/numpy port of upstream `ImagePatcher`): square-pad extreme aspect ratios, resize so `max(W, H) ≤ 3024`, then either single-view (`long ≤ 728`) or sliding-window 504² sub-patches.
2. **Vision encoder dual-pass**: global 728² → 169 tokens (52×52 grid → 2× downsample → 13² = 169), each 504² sub-patch → 81 tokens (36×36 grid → 9² = 81). Same ViT weights, two grid sizes via interpolated positional embeddings.
3. **Placeholder layout per image** (chat-template + processor): `[<patch_start>{81×<im_patch>}<patch_end> + opt <patch_newline>] × N + <im_start>{169×<im_patch>}<im_end>`. `encode_image` returns features in matching order; merge via `mx.cumsum` + `mx.where`.

## Bugs caught + fixed during the port

1. **MLX `nn.Upsample` applies 2-tuple `scale_factor` to dims 1+2 of the input** — not the last 2 spatial dims like PyTorch's `F.interpolate` does on NCHW. So 504²→36×36 positional embedding interpolation needs **NHWC layout natively** (no NCHW transpose). The naive port silently shrank the 1536 hidden dim instead of the spatial 52² dims, producing a tensor with 1,989,936 elements that couldn't reshape to `(1296, 1536)`.

2. **`mx.load + mx.save_safetensors` round-trip writes zeros for lazy values.** When extracting vision weights from upstream bf16 shards, you must `mx.eval(v)` each tensor before storing it in the output dict, or the safetensors file ends up shape-correct but full of zeros. The format=`mlx` metadata is required for `mlx-vlm` to load the file correctly.

3. **Vision feature cache must be ignored on multi-crop path.** Engines that pre-cache vision features via `model.encode_image(pixel_values)` only get the global 169 tokens; multi-crop prompts have `169 + N×81` placeholder positions, so the cached features no longer match. The naive zero-pad fallback then trashes every patch slot with zeros. Fix: detect `patch_pixel_values` in `kwargs` and recompute everything when present.

4. **Vision weight provenance matters more than vision architecture.** Step3-VL-10B and Step-3.7-Flash share the same `StepRoboticsVisionEncoder` architecture, but their projector weights were trained against different text-side LMs (Qwen3-8B vs step3p5 MoE) — using Step3-VL-10B's vision weights with Step-3.7-Flash's LM gives mathematically correct vision features that the LM can't decode (model collapses to "white" / "a man in a suit" for every image). Always pull `model-vit-00001/00002.safetensors` from the matching upstream model.

## Installation

Drop the adapter into your `mlx_vlm` (and optionally `mlx_lm`) install:

```bash
# Discover the install paths from Python — works for pip / uv / conda /
# bundled frameworks alike.
MLX_VLM_DIR=$(python -c "import os, mlx_vlm; print(os.path.dirname(mlx_vlm.__file__))")
MLX_LM_DIR=$(python  -c "import os, mlx_lm;  print(os.path.dirname(mlx_lm.__file__))")

cp -r mlx_vlm/models/step3p7  "$MLX_VLM_DIR/models/"
cp    mlx_lm/models/step3p7.py "$MLX_LM_DIR/models/"
```

You also need a 1-line patch to `mlx_vlm/prompt_utils.py`:
```python
# In MESSAGE_FORMATS dict, add:
"step3p7": MessageFormat.LIST_WITH_IMAGE_FIRST,
```

## Preparing the model weights

You will need quantized text shards + 1 vision shard with pre-transposed Conv2d kernels.

```bash
# 1. Get Step-3.7-Flash bf16 from upstream (very large)
hf download stepfun-ai/Step-3.7-Flash --local-dir Step-3.7-Flash

# 2. Quantize text shards to 4-bit (or use an existing community quant)
# e.g. https://huggingface.co/<some>/Step-3.7-Flash-4bit/

# 3. Extract vision weights with the NHWC Conv2d transpose applied
python scripts/extract_vision_weights.py \
  --src Step-3.7-Flash \
  --dst Step-3.7-Flash-4bit/model-00023-of-00023.safetensors

# 4. Patch chat_template.jinja into the model dir
cp model_dir_patches/chat_template.jinja Step-3.7-Flash-4bit/

# 5. Update model.safetensors.index.json to include the new vision shard keys.
# (Index update logic depends on your text quant — see `processing_step3.py` upstream for the canonical layout.)
```

## Usage (via mlx-vlm)

```python
from mlx_vlm import load, apply_chat_template, generate
from PIL import Image

model, processor = load("Step-3.7-Flash-4bit")
img = Image.open("photo.jpg")
prompt = apply_chat_template(processor, model.config, [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "What is in this image?"},
    ]}
], num_images=1)
output = generate(model, processor, prompt, image=img, max_tokens=200)
print(output)
```

Or call any OpenAI-compatible MLX engine with `model="Step-3.7-Flash-4bit"` and image content blocks.

## Tested on

- Mac Studio M3 Ultra 512 GB
- `mlx==0.30.x`, `mlx-vlm==0.5.0`
- Text decode: ~44 tok/s warm
- Vision: 4-corner color/shape recognition on 2000×1500 images, sun detection on 1920×1080 landscape — all correct
- Long-context: 182K-token needle-in-haystack retrieval — correct answer, ~11 min wall (Step 3.7 4-bit is **slow-but-correct** vs the same generation's Huihui 8-bit Opus-distill at ~4.5 min)

## Multi-Token Prediction (MTP) speculative decoding — experimental

Step 3.7 ships 3 MTP heads (vLLM `Step3p5MTP`) trained against the upstream
BF16 backbone. We ported the MTPModule + MTPLayer classes to MLX
(`mlx_vlm/models/step3p7/language.py`) and wired `mtp_forward` /
`make_mtp_cache` / `return_hidden` into both LanguageModel and the outer
`Model` so the oMLX MTP draft/verify cycle (qwen35 PR 990-derived
batch_generator patch) engages automatically when `mtp_enabled: True` is
set in the model settings.

**Status:** infrastructure works end-to-end — `MTP path activated for
uid=N (model has mtp_forward, batch=1)` confirmed; draft/verify cycle
executes; weights load cleanly. Draft quality on M3 Ultra with
Hikari07jp's BF16 extraction is too low for net speedup:

| Config | Overall TPS (warm, 200 tok) | Note |
|---|---|---|
| MTP off (`turboquant_kv: True`) | **45.6** | production default |
| MTP on, no norm shift | 32.9 | 0.0% accept |
| MTP on, +1 shift on 5 Gemma norms | 32.0 | 2.6-4.2% accept |
| MTP on, +1 shift on 4 Gemma norms (no `shared_head_norm`) | 30.9 | 2.1% accept |

Break-even for net speedup requires ≈ 43% accept rate; we observe 2-4%.

**Likely remaining issues (next investigation):**

1. **Embedding precision** — the MTP shard ships its own BF16
   `model.embed_tokens.weight`; we currently drop it and reuse the
   backbone's 4-bit quantized embedding via `self.model.embed_tokens`.
   vLLM's `Step3p5AMultiTokenPredictor` keeps a separate embedding.
   Add a BF16 `embed_tokens` to `MTPModule` + use it in `mtp_forward`.
2. **shared_head_norm convention** — mean ≈ 1.70 is between Gemma
   zero-centered (~0) and plain RMSNorm (~1-2.5). The current shift
   decision (don't shift it) is empirical; pairing this with the
   embedding fix above may reveal the correct convention.
3. **Hidden state precision** — backbone forward returns hidden at
   the dequantized 4-bit precision; MTP receives it via `return_hidden`
   and feeds bf16 weights. Mixed precision might compound on the
   `enorm/hnorm/eh_proj` fusion path.

To experiment, set `mtp_enabled: True` and `turboquant_kv_enabled: False`
(they're mutually exclusive per oMLX) in `your engine settings file`
for `Step-3.7-Flash-4bit`, restart oMLX, and check
`grep "MTP\[" your engine log` for accept rate. To run the
shard rewrite yourself:

```bash
hf download Hikari07jp/Step-3.7-Flash-MTP-draft --local-dir /tmp/mtp-src
python scripts/rewrite_mtp_shard.py \
  /tmp/mtp-src/model.safetensors \
  Step-3.7-Flash-4bit/model-00024-of-00024.safetensors
# Then update model.safetensors.index.json so the 48 new keys point
# at the new shard filename.
```

The `MLX_LM_MTP` patch from oMLX 0.3.10+ exposes its own
`_is_mtp_compatible` allowlist; ours patches that list to recognise
`step3p5` and `step3p7` model_types (see
`patches/mlx_lm_mtp_compat.diff` for the one-line addition).

## License

Adapter code: Apache-2.0 (see `LICENSE`). Upstream Step-3.7-Flash weights are licensed separately by Step Robotics — see [their model card](https://huggingface.co/stepfun-ai/Step-3.7-Flash) before redistributing.

## Credits

Architecture port + bug catches were done end-to-end across 7 sessions of MLX debugging on Apple Silicon, with [PyTorch reference parity tests](https://github.com/stepfun-ai/Step3-VL-10B) used to localize the four bugs above. References that helped (none of them validate end-to-end vision on Step-3.7-Flash specifically):

- [`monyschuk/Huihui-Step3-VL-10B-abliterated-mlx`](https://huggingface.co/monyschuk/Huihui-Step3-VL-10B-abliterated-mlx) — MLX port for Step3-VL-10B (omits LayerScale + positional embedding, incompatible with stock weights)
- [`AITRADER/Huihui-Step3-VL-10B-abliterated-mlx-fp16`](https://huggingface.co/AITRADER/Huihui-Step3-VL-10B-abliterated-mlx-fp16) — explicitly marks vision inference "out of scope"
- Upstream [`stepfun-ai/Step3-VL-10B`](https://github.com/stepfun-ai/Step3-VL-10B) — PyTorch reference for the vision encoder
