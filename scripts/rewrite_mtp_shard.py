#!/usr/bin/env python3
"""Pre-rewrite Hikari07jp/Step-3.7-Flash-MTP-draft shard keys to the final
MLX-side names that Step3p7.Model's load_weights consumes directly.

Background — mlx_vlm.utils.load() skips Model.sanitize when the FIRST shard
in the model dir has metadata `format=mlx` (line 251 of mlx_vlm/utils.py:
`if not is_mlx_format: sanitize_weights(...)`). Our backbone 4-bit shards
all carry that metadata, so any new MTP shard appended to the model dir
must already be in the final key naming form — sanitize will not run.

This script does the rewrite once, offline:

    model.embed_tokens.weight                                       → drop (reuse backbone embed)
    model.layers.{45,46,47}.eh_proj.weight                          → language_model.mtp.layers.{0,1,2}.eh_proj.weight
    model.layers.{45,46,47}.enorm.weight                            → language_model.mtp.layers.{0,1,2}.enorm.weight
    model.layers.{45,46,47}.hnorm.weight                            → language_model.mtp.layers.{0,1,2}.hnorm.weight
    model.layers.{45,46,47}.transformer.shared_head.norm.weight     → language_model.mtp.layers.{0,1,2}.shared_head_norm.weight
    model.layers.{45,46,47}.transformer.shared_head.output.weight   → drop (reuse backbone lm_head)
    model.layers.{45,46,47}.<everything else>                       → language_model.mtp.layers.{0,1,2}.mtp_block.<everything else>

The result is saved with `metadata={"format": "mlx"}` so mlx-vlm's load path
treats it as a regular shard with no sanitization. Uses safetensors.torch
because mx.save_safetensors silently failed on the ~1.7 GB bf16 dict
during testing (mlx 0.30.x).

Usage:
    python scripts/rewrite_mtp_shard.py <src-shard> <dst-shard>

After running, copy the dst-shard to your model dir as the next
`model-XXXXX-of-XXXXX.safetensors` and update `model.safetensors.index.json`
to map all new keys → that shard.
"""
import argparse
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


MTP_DIRECT_NAMES = ("enorm", "hnorm", "eh_proj", "shared_head")
NUM_DECODER_LAYERS = 45  # Step 3.7 backbone — MTP layers start at index 45

# Hikari07jp's BF16 MTP-draft shard stores these norms in vLLM's
# **GemmaRMSNorm zero-centered convention** — vLLM computes
# `(1 + weight) * normalize(x)` so its stored weights are tiny offsets
# centered around 0 (e.g. enorm mean ≈ -0.26, input_layernorm mean ≈ 0.06).
# MLX-side `ZeroCenteredRMSNorm` (despite the name) is actually plain
# `mx.fast.rms_norm(x, weight, eps)` — no +1 baked in — so we must add 1
# to these weights at extract time. Otherwise activations through the MTP
# head are 10-20× too small and the head emits garbage (0% accept rate,
# observed in production 2026-05-30 first bench).
#
# Affected keys (suffixes), per vLLM Step3p5MTP source:
#   enorm.weight, hnorm.weight                     (Step3p5AMultiTokenPredictorLayer)
#   mtp_block.input_layernorm.weight               (Step3p5DecoderLayer)
#   mtp_block.post_attention_layernorm.weight      (Step3p5DecoderLayer)
#   shared_head_norm.weight                        (SharedHead)
# NOT applied to q_norm / k_norm (those are plain RMSNorm in vLLM too).
GEMMA_NORM_SUFFIXES = (
    ".enorm.weight",
    ".hnorm.weight",
    ".mtp_block.input_layernorm.weight",
    ".mtp_block.post_attention_layernorm.weight",
    # shared_head_norm intentionally OMITTED: its stored mean ≈ 1.70 (vs the
    # other Gemma norms' mean ≈ 0), and the backbone's `model.norm.weight`
    # mean ≈ 2.46 — same order of magnitude. Empirically, including it
    # double-shifts to ≈ 2.70 which over-scales activations. First-bench
    # accept rate caps at 4.2% with shift; experiment without shift to
    # match the plain-RMSNorm storage Hikari07jp appears to have used for
    # this one tensor.
)


def rewrite_key(src_key: str) -> str | None:
    """Return the MLX-side key name, or None if the tensor should be dropped."""
    if src_key == "model.embed_tokens.weight":
        return None
    if not src_key.startswith("model.layers."):
        return src_key
    parts = src_key.split(".")
    if len(parts) <= 2 or not parts[2].isdigit():
        return src_key
    layer_idx = int(parts[2])
    if layer_idx < NUM_DECODER_LAYERS:
        return src_key
    rel_idx = layer_idx - NUM_DECODER_LAYERS
    suffix = ".".join(parts[3:])
    if suffix.startswith("transformer."):
        suffix = suffix[len("transformer."):]
    if any(suffix.startswith(n) for n in MTP_DIRECT_NAMES):
        if suffix.startswith("shared_head."):
            if "shared_head.output" in suffix:
                return None
            suffix = suffix.replace("shared_head.norm", "shared_head_norm")
        return f"language_model.mtp.layers.{rel_idx}.{suffix}"
    return f"language_model.mtp.layers.{rel_idx}.mtp_block.{suffix}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="Path to Hikari07jp model.safetensors")
    ap.add_argument("dst", type=Path, help="Output safetensors path")
    args = ap.parse_args()

    new_tensors = {}
    n_kept = n_dropped = n_shifted = 0
    with safe_open(str(args.src), framework="pt") as f:
        for k in f.keys():
            new_k = rewrite_key(k)
            if new_k is None:
                n_dropped += 1
                continue
            t = f.get_tensor(k)
            if any(new_k.endswith(sfx) for sfx in GEMMA_NORM_SUFFIXES):
                # Add 1.0 to convert vLLM's Gemma zero-centered storage to MLX
                # plain-RMSNorm format. Done in fp32 to avoid bf16 rounding loss
                # near 1.0, then cast back to source dtype.
                src_dtype = t.dtype
                t = (t.to(dtype=__import__("torch").float32) + 1.0).to(src_dtype)
                n_shifted += 1
            new_tensors[new_k] = t
            n_kept += 1
    print(f"kept={n_kept} dropped={n_dropped} norm_shifted={n_shifted} total_out={len(new_tensors)}")
    save_file(new_tensors, str(args.dst), metadata={"format": "mlx"})
    print(f"wrote {args.dst} ({args.dst.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
