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

## Multi-Token Prediction (MTP) speculative decoding

Step 3.7 ships 3 MTP heads (vLLM `Step3p5MTP`) trained against the upstream
BF16 backbone. We port the MTPModule + MTPLayer classes to MLX
(`mlx_vlm/models/step3p7/language.py`) and wire `mtp_forward` /
`make_mtp_cache` / `return_hidden` into both LanguageModel and the outer
`Model` so the oMLX MTP draft/verify cycle (qwen35 PR 990-derived
batch_generator patch) engages automatically when `mtp_enabled: True` is
set in the model settings.

**Status: works. 73-80% accept rate, lossless at temp=0, ~15% net speedup.**

| Config (Step-3.7-Flash-4bit, M3 Ultra, 300 tok warm, 5 iters) | Overall TPS | Accept |
|---|---|---|
| MTP off (`turboquant_kv: True`) | 45.6 | n/a |
| **MTP on (per-MTP-layer lm_head + BF16 embed + 7 norm shifts)** | **52.6** | **73-80%** |

Per-prompt: short_code 60, physics 57, Chinese translation 49 tok/s.
Greedy output is byte-identical to MTP-off (verified on 4 prompts).

### What it took to get here

The hard part was identifying which weight-loading conventions Hikari07jp's
BF16 extraction expected. With the wrong conventions, accept rate stuck at
0-4% (slowdown vs baseline); with the right conventions, 75%.

1. **Per-MTP-layer `shared_head.output` is NOT tied to backbone `lm_head`**
   — biggest finding. Each of the 3 MTP layers in the upstream BF16 ships
   its own `shared_head.output` weight (different per-layer means
   0.000008/0.000029/0.000034). We were initially calling the backbone's
   4-bit quantized `lm_head` from `mtp_forward`, which produced ~0% useful
   draft tokens. Per-MTP-layer BF16 head jumped accept from 2-4% to 70-80%.
   Detected by inspecting `model.layers.{45,46,47}.transformer.shared_head.output`
   in the BF16 source and noticing the three rows aren't equal.
2. **MTP-side `embed_tokens` is also separate from backbone** — vLLM
   `Step3p5AMultiTokenPredictor` keeps its own `VocabParallelEmbedding`.
   We add an `nn.Embedding` directly on `MTPModule` and route the BF16
   `model.embed_tokens.weight` from the MTP shard there. (This alone
   didn't move the needle, but it removes the mixed-precision drift
   from `mtp_forward`'s input lookup.)
3. **+1 norm shift on extract** — vLLM `GemmaRMSNorm.forward_native`
   does `weight = stored + 1.0; rms_norm(x, weight, eps)`. MLX
   `ZeroCenteredRMSNorm` is plain `mx.fast.rms_norm(x, weight, eps)` (no
   +1 baked in despite the class name). We pre-add 1 to the 7 Gemma
   norms in the MTP shard (`enorm`, `hnorm`, `mtp_block.input_layernorm`,
   `mtp_block.post_attention_layernorm`, `mtp_block.self_attn.q_norm`,
   `mtp_block.self_attn.k_norm`, `shared_head_norm`) at offline
   extract time.
4. **Pre-norm hidden state passed to MTP** — `LanguageModel.__call__`
   returns the residual stream *before* `body.norm` when called with
   `return_hidden=True`. vLLM `Step3p5Model.forward()` does the same.
   Empirically pre-norm gives 73-80% accept; post-norm gave 1-2%.

The mismatched-precision hypothesis (4-bit backbone vs BF16 MTP) wasn't
the bottleneck — BF16 embed_tokens didn't move accept on its own. The
hidden state precision wasn't either. The bottleneck was using the wrong
lm_head.

### How to use

The MTP path activates automatically when these settings are set on
`Step-3.7-Flash-4bit` (or your equivalent model id) in
`your engine settings file`:

```json
{
  "mtp_enabled": true,
  "turboquant_kv_enabled": false
}
```

`mtp_enabled` and `turboquant_kv_enabled` are mutually exclusive per oMLX
(TurboQuant patches the attention path that MTP relies on). Bench results
above are with TurboQuant OFF. Restart oMLX after changing settings:
`bash restart your engine`. Verify activation in the log:

```
[engine] Native MTP patch applied for ... (model_type=step3p7, active)
[batch_generator] MTP path activated for uid=N (model has mtp_forward, batch=1)
[batch_generator] MTP[N] finish=length tokens=200 cycles=117 accept=82/117 (70.1%) ...
```

To reproduce the shard rewrite from scratch:

```bash
hf download Hikari07jp/Step-3.7-Flash-MTP-draft --local-dir /tmp/mtp-src
python scripts/rewrite_mtp_shard.py \
  /tmp/mtp-src/model.safetensors \
  Step-3.7-Flash-4bit/model-00024-of-00024.safetensors
# Then update model.safetensors.index.json so the 52 new keys point
# at the new shard filename (49 layer + 3 shared_head_output entries).
```

### Acknowledgements

- [Hikari07jp/Step-3.7-Flash-MTP-draft](https://huggingface.co/Hikari07jp/Step-3.7-Flash-MTP-draft)
  — extracted BF16 MTP-draft layers from upstream
  `stepfun-ai/Step-3.7-Flash`; the same weights run at ~80% accept in vLLM.
- [oMLX](https://omlx.ai/) — `patches/mlx_lm_mtp/` ports mlx-lm PR 990's
  MTP draft+verify batch_generator dispatch to oMLX's continuous-batching
  scheduler. We add `step3p5` / `step3p7` to its compatible list.
- [mlx-optiq/Gemma 4 spec dec post](https://mlx-optiq.com/blog/gemma-spec-decoding)
  — the structurally identical 0%→3%→33% accept-rate fix narrative for
  Gemma 4 spec dec on MLX was a useful template for diagnosing this one.

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
