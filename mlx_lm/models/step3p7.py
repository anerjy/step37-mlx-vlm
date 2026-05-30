# Step 3.7 Flash MLX adapter — Phase 1 (text-only).
#
# Inherits step3p5.py's entire architecture (verified identical text_config:
# 4096 hidden, 45 layers, 64 attn heads, 8 KV groups, 128 head_dim, 288 experts,
# top-8, 1280 expert dim, partial rotary [0.5/1/1/1], sliding 512, etc.).
#
# Only delta vs step3p5:
#   1. Weights live under `language_model.*` prefix (Step3p7ForConditionalGeneration
#      multimodal wrapper). sanitize() strips the prefix.
#   2. Vision keys (vision_model.*, vit_large_projector, multi_modal_projector,
#      mm_*) are dropped in Phase 1. Phase 2 will add a real vision_tower.
#   3. mlx-community 4-bit weights happen to ship with vision stripped already
#      (same packaging bug as Huihui mlx-8bit), so step 2 is defensive in case
#      6-bit/8-bit packages or future re-uploads include vision.
#
# step3p5's existing MTP-skip + layer-index skip + .moe.→.mlp.switch_mlp.
# remappings are preserved unchanged via super().sanitize().

from typing import Optional, List, Any

from mlx_lm.models import step3p5

# transformers 5.7.0 doesn't know model_type "step3p7" — AutoTokenizer.from_pretrained
# falls back to an empty PreTrainedConfig() which has no max_position_embeddings,
# crashing tokenizer load. oMLX ships a generic AutoTokenizer wrapper for the
# DeepSeek V4 variant of the same bug; the wrapper's fallback fires whenever
# `max_position_embeddings` appears in the AttributeError message, so it works
# unchanged for us. We just need it active before mlx_lm.load() calls
# AutoTokenizer.from_pretrained — easiest path is a side-effect on import,
# since mlx_lm.utils.load_model imports this module before load_tokenizer.
try:
    from omlx.patches.deepseek_v4.tokenizer_patch import apply_tokenizer_patch
    apply_tokenizer_patch()
except Exception:  # standalone mlx_lm without oMLX — patch unavailable, leave it
    pass

# Reuse ModelArgs unchanged. Config hack: caller flattens text_config into
# top-level before mlx_lm.load() reads it.
ModelArgs = step3p5.ModelArgs


class Model(step3p5.Model):
    def sanitize(self, weights):
        unwrapped = {}
        for k, v in weights.items():
            if (
                k.startswith("vision_model.")
                or k.startswith("vit_large_projector")
                or k.startswith("multi_modal_projector")
                or k.startswith("mm_")
            ):
                continue

            if k.startswith("language_model."):
                k = k[len("language_model.") :]

            unwrapped[k] = v

        return super().sanitize(unwrapped)
