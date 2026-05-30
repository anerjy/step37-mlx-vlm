"""Top-level Step-3.7-Flash multimodal Model for mlx_vlm.

Wires:
  - language_model = step3p5 backbone (45 layers, Qwen3.6-A3B MoE)
  - vision_model = StepRoboticsVisionEncoder (47-layer custom ViT)
  - vit_large_projector = single Linear(width*4, hidden_size)

Pipeline matches modeling_step3p7.Step3p7Model:
  1. encode image through vision_model → (B, num_patches, width)
  2. _process_image_features: reshape (B, C, Gh, Gw) → vit_downsampler1 →
     vit_downsampler2 → flatten (B, 169, width*4) → vit_large_projector
     → (B, 169, hidden_size)
  3. get_input_embeddings: replace image_token_id (128001) positions in
     input_ids with the projected image embeddings.
"""
import logging
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .vision import StepRoboticsVisionEncoder

logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type

        self.language_model = LanguageModel(config.text_config)
        self.vision_model = StepRoboticsVisionEncoder(config.vision_config)
        self.vit_large_projector = nn.Linear(
            config.vision_config.width * 4,
            config.text_config.hidden_size,
            bias=config.projector_bias,
        )
        self.image_token_id = config.image_token_id

    # oMLX's VLMBatchedEngine looks for `model.vision_tower` (qwen/llava
    # convention). Expose it as a Python property so the alias doesn't
    # appear as a duplicate sub-module in mlx's parameter tree.
    @property
    def vision_tower(self):
        return self.vision_model

    def encode_image(
        self,
        pixel_values: mx.array,
        patch_pixel_values: Optional[mx.array] = None,
        num_patches: Optional[mx.array] = None,
    ) -> mx.array:
        """Full image-to-text-embedding pipeline (without merge).

        Single-view (no patches): returns (B_imgs, 169, hidden_size).
        Multi-crop (patches present): returns flat (total_tokens, hidden_size)
        where token order matches the chat-template-emitted placeholder
        sequence — per image: [patches_for_image_i_flat] + [global_i].

        oMLX VLMBatchedEngine strategy-1 calls this with just pixel_values
        to pre-compute features for caching. The cached features only
        cover the global view; get_input_embeddings detects the presence
        of patch_pixel_values via kwargs and falls back to a full
        recompute that includes patches.
        """
        if not isinstance(pixel_values, mx.array):
            pixel_values = mx.array(pixel_values)
        vit_out = self.vision_model(pixel_values)
        global_feats = self._process_image_features(vit_out)
        if patch_pixel_values is None:
            return global_feats
        if not isinstance(patch_pixel_values, mx.array):
            patch_pixel_values = mx.array(patch_pixel_values)
        if patch_pixel_values.shape[0] == 0:
            return global_feats

        patch_vit = self.vision_model(patch_pixel_values)
        patch_feats = self._process_image_features(patch_vit)  # (B_patches, 81, hidden)
        hidden = global_feats.shape[-1]

        if num_patches is None:
            return mx.concatenate(
                [patch_feats.reshape(-1, hidden), global_feats.reshape(-1, hidden)],
                axis=0,
            )

        if hasattr(num_patches, "tolist"):
            num_patches_list = num_patches.tolist()
        else:
            num_patches_list = list(num_patches)
        out_parts = []
        patch_offset = 0
        for img_idx, n in enumerate(num_patches_list):
            n = int(n)
            if n > 0:
                img_patches = patch_feats[patch_offset:patch_offset + n]  # (n, 81, hidden)
                out_parts.append(img_patches.reshape(-1, hidden))           # (n*81, hidden)
                patch_offset += n
            out_parts.append(global_feats[img_idx].reshape(-1, hidden))      # (169, hidden)
        return mx.concatenate(out_parts, axis=0)

    # --- mlx_vlm conventions ---------------------------------------------

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    # --- vision pipeline -------------------------------------------------

    def _process_image_features(self, image_features: mx.array) -> mx.array:
        """(B, num_patches, width) → (B, num_patches/4, hidden_size).

        Mirrors Step3p7Model._process_image_features in modeling_step3p7.py:
          - permute to (B, width, Gh, Gw)
          - 2× Conv2d stride-2 downsamplers (width → 2W → 4W)
          - flatten back to (B, Gh*Gw/16, 4W)
          - vit_large_projector to text hidden_size
        """
        B, P, _C = image_features.shape
        HW = int(P ** 0.5)  # 52 for a 728² image after patching at 14

        # PyTorch: (B, P, C) → permute → (B, C, P) → view (B, C, HW, HW)
        # MLX Conv2d expects NHWC: feed (B, HW, HW, C) directly.
        x = image_features.reshape(B, HW, HW, _C)

        x = self.vision_model.vit_downsampler1(x)  # (B, HW/2, HW/2, 2C)
        x = self.vision_model.vit_downsampler2(x)  # (B, HW/4, HW/4, 4C)

        # Flatten spatial dims
        new_HW = x.shape[1]
        x = x.reshape(B, new_HW * new_HW, -1)        # (B, 169, 4C)
        x = self.vit_large_projector(x)               # (B, 169, hidden_size)
        return x

    # --- mlx_vlm interface -----------------------------------------------

    def get_input_embeddings(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        cached_image_features: Optional[mx.array] = None,
        **kwargs,
    ):
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        wte = self.language_model.model.embed_tokens

        if pixel_values is None and cached_image_features is None:
            return InputEmbeddingsFeatures(inputs_embeds=wte(input_ids))

        # Multi-crop kwargs from processor (Step 3.7 specific)
        patch_pixel_values = kwargs.get("patch_pixel_values")
        num_patches = kwargs.get("num_patches")
        has_patches = (
            patch_pixel_values is not None
            and hasattr(patch_pixel_values, "shape")
            and patch_pixel_values.shape[0] > 0
        )

        if cached_image_features is not None and not has_patches:
            # Cache hit, no patches needed → use cached global features as-is.
            image_embeds = cached_image_features
        else:
            # pixel_values comes in as either (C, H, W) or (B, C, H, W) or
            # (B, T, C, H, W) where T is num crops/images per sample. Flatten
            # the leading dims into the batch axis for the vision tower.
            if pixel_values.ndim == 3:
                pixel_values = pixel_values[None, ...]
            if pixel_values.ndim == 5:
                B5, T5, C, H, W = pixel_values.shape
                pixel_values = pixel_values.reshape(B5 * T5, C, H, W)

            if has_patches:
                # When patches are present we ignore any pre-cached features
                # (engine cache stores global-only) and run a full encode_image
                # pass — this handles patches + global concat in the right
                # token order for the chat-template placeholder layout.
                image_embeds = self.encode_image(
                    pixel_values,
                    patch_pixel_values=patch_pixel_values,
                    num_patches=num_patches,
                )
            else:
                # Standard single-view path
                vit_features = self.vision_model(pixel_values)
                image_embeds = self._process_image_features(vit_features)

        # Step 3: merge image embeddings into the embedded input_ids stream.
        # Method: build a (seq_len, hidden) "fill" tensor where image-token
        # rows hold the next image embed and non-image rows hold zeros,
        # then mx.where the original text embeddings with the fill at the
        # is_image mask. This works without in-place assignment, which
        # MLX doesn't reliably propagate through reshape views.
        batch_size, seq_len = input_ids.shape
        inputs_embeds = wte(input_ids)
        is_image = input_ids == self.image_token_id

        n_image_tokens = int(mx.sum(is_image).item())
        if n_image_tokens == 0:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        flat_image = image_embeds.reshape(-1, image_embeds.shape[-1])  # (N_img, hidden)
        # Fail fast on placeholder/feature count mismatch — silent zero-pad
        # would produce a model that hallucinates ("I see white" / "a man in
        # a suit") instead of obviously erroring. Common causes: chat-template
        # not in sync with processor (P3c multi-crop layout contract — see
        # README bug #3), patch_pixel_values dropped by an engine cache layer,
        # or num_patches array out of sync with the patch tensor.
        if flat_image.shape[0] != n_image_tokens:
            raise ValueError(
                f"step3p7: vision feature count {flat_image.shape[0]} != "
                f"image-token placeholder count {n_image_tokens} in input_ids. "
                f"Likely chat-template ↔ processor placeholder-emission "
                f"contract drift (per-image layout: "
                f"[<patch_start>{{81×<im_patch>}}<patch_end>[<patch_newline>]?]×N "
                f"+ <im_start>{{169×<im_patch>}}<im_end>). Check that "
                f"chat_template.jinja emits ONE <im_patch> marker per image "
                f"and Step3p7Processor._per_image_repl expands it correctly."
            )

        # Build (batch, seq_len, hidden) fill tensor. For each image-token
        # position (in scan order across the flattened mask), place the
        # corresponding flat_image[k] row.
        # Approach: cumulative sum of the mask gives per-row image index.
        mask_int = is_image.astype(mx.int32)
        cum = mx.cumsum(mask_int.reshape(-1), axis=0) - 1  # (B*L,)
        cum = mx.maximum(cum, mx.zeros_like(cum))  # clamp the leading non-image rows
        gathered = flat_image[cum]                # (B*L, hidden)
        gathered = gathered.reshape(batch_size, seq_len, -1)

        # Where mask is True, take from gathered; else keep original embed.
        mask_3d = is_image.astype(inputs_embeds.dtype)[..., None]
        inputs_embeds = inputs_embeds * (1 - mask_3d) + gathered * mask_3d
        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        return_hidden: bool = False,
        **kwargs,
    ):
        embed = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return self.language_model(
            input_ids,
            inputs_embeds=embed.inputs_embeds,
            mask=mask,
            cache=cache,
            return_hidden=return_hidden,
        )

    # --- MTP pass-through ------------------------------------------------
    # oMLX's scheduler invokes MTP via the outer Model wrapper; forward to
    # LanguageModel so the scheduler doesn't need to know about the
    # vision/language split.

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache=None, spec_step_idx=0):
        return self.language_model.mtp_forward(
            hidden_states, next_token_ids, mtp_cache=mtp_cache, spec_step_idx=spec_step_idx
        )

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    @property
    def has_mtp(self):
        return getattr(self.language_model, "mtp", None) is not None

    # --- weight sanitize -------------------------------------------------

    def sanitize(self, weights):
        """Strip outer prefixes + apply per-component sanitize.

        Three buckets:
          - vision_model.* → keep (Conv2d weights need NCHW→NHWC transpose)
          - vit_large_projector.* → keep as-is
          - language_model.* → strip prefix, run step3p5 sanitize, re-add prefix
          - anything else → drop (defensive)
        """
        # Step 1: split buckets
        text_weights = {}
        new_weights = {}

        for k, v in weights.items():
            if k.startswith("language_model."):
                inner = k[len("language_model."):]
                text_weights[inner] = v
            elif k.startswith("vision_model."):
                # Conv2d weights need NCHW→NHWC transpose for MLX
                if k.endswith(".weight") and (
                    "conv1." in k or "vit_downsampler1." in k or "vit_downsampler2." in k
                ):
                    if v.ndim == 4:
                        v = v.transpose(0, 2, 3, 1)
                new_weights[k] = v
            elif k.startswith("vit_large_projector"):
                new_weights[k] = v
            elif k.startswith("model.layers.") or k == "model.embed_tokens.weight":
                # MTP shard weights ship WITHOUT a `language_model.` prefix
                # (Hikari07jp's BF16 extraction keeps the original stepfun-ai
                # key naming). Route them into the text bucket so
                # LanguageModel.sanitize can rewrite layer 45+ into
                # mtp.layers.{0..2}.* (vLLM Step3p5MTP convention).
                # model.embed_tokens.weight in the MTP shard is a BF16 copy
                # of the backbone embedding; drop it — the backbone's quantized
                # embed_tokens is already loaded from earlier shards.
                if k == "model.embed_tokens.weight":
                    continue
                text_weights[k] = v
            else:
                # multi_modal_projector / mm_* / image_newline / etc — drop
                continue

        # Step 2: apply language_model sanitize (step3p5 logic)
        text_sanitized = LanguageModel.sanitize(text_weights)
        for k, v in text_sanitized.items():
            new_weights[f"language_model.{k}"] = v

        return new_weights

    @property
    def cast_predicate(self):
        def predicate(k):
            return "router_bias" not in k
        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if "mlp.gate.gate" in path:
                return {"group_size": 64, "bits": 8}
            return True
        return predicate
