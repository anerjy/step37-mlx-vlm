"""Step 3.7 language backbone wrapper for mlx_vlm.

The underlying architecture is identical to step3p5 (45-layer Qwen3.6-style
MoE with shared experts). We reuse mlx_lm.models.step3p5 directly and add
the inputs_embeds plumbing that the VLM call path needs.
"""
from dataclasses import asdict
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.step3p5 import (
    Model as Step3p5Model,
    ModelArgs as Step3p5Args,
    Step3p5Model as Step3p5Body,
)
from mlx_lm.models.base import create_attention_mask

from .config import TextConfig


def _make_step3p5_args(text_cfg: TextConfig) -> Step3p5Args:
    """Project mlx_vlm-side TextConfig into mlx_lm-side ModelArgs."""
    # Step3p5Args is a dataclass; pick only fields it accepts.
    fields = {f.name for f in Step3p5Args.__dataclass_fields__.values()}
    src = asdict(text_cfg)
    return Step3p5Args(**{k: v for k, v in src.items() if k in fields})


class LanguageModel(nn.Module):
    """Wraps step3p5.Model with an inputs_embeds-aware __call__."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        args = _make_step3p5_args(config)
        inner = Step3p5Model(args)
        # Mirror checkpoint keys: language_model.model.* + language_model.lm_head.*
        self.model = inner.model
        self.lm_head = inner.lm_head
        self._args = args

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        # Reuse step3p5.Model's cache builder by reconstructing it via the
        # underlying layers.
        from mlx_lm.models.cache import KVCache, RotatingKVCache
        caches = []
        for layer in self.model.layers:
            if getattr(layer, "is_sliding", False):
                caches.append(RotatingKVCache(max_size=self._args.sliding_window))
            else:
                caches.append(KVCache())
        return caches

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
        **kwargs,
    ):
        # Forward through Step3p5Body, optionally bypassing embed_tokens.
        body = self.model

        if inputs_embeds is None:
            h = body.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * body.num_layers

        full_mask = None
        swa_mask = None
        if body._full_idx is not None:
            full_mask = create_attention_mask(h, cache[body._full_idx])
        if body._swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[body._swa_idx], window_size=body.args.sliding_window
            )

        for layer, c in zip(body.layers, cache):
            m = swa_mask if layer.is_sliding else full_mask
            h = layer(h, mask=m, cache=c)

        h = body.norm(h)
        return self.lm_head(h)

    @staticmethod
    def sanitize(weights):
        """Apply step3p5's MTP-skip + layer-index skip + moe→switch_mlp remaps.

        Caller is expected to have already stripped the outer
        `language_model.` prefix and pre-filtered keys that don't
        belong to the text model.
        """
        # Build a temporary Step3p5Model instance just to use its sanitize.
        # We don't need real ModelArgs — sanitize's only param dependency is
        # num_hidden_layers for the index check, which we can mimic by
        # reading from any layer key in the weights.
        max_layer_idx = -1
        for k in weights:
            if "model.layers." in k:
                parts = k.split("model.layers.")[1].split(".")
                if parts and parts[0].isdigit():
                    max_layer_idx = max(max_layer_idx, int(parts[0]))

        remappings = [
            (".moe.gate_proj.", ".mlp.switch_mlp.gate_proj."),
            (".moe.up_proj.", ".mlp.switch_mlp.up_proj."),
            (".moe.down_proj.", ".mlp.switch_mlp.down_proj."),
            (".moe.gate.", ".mlp.gate.gate."),
            (".moe.router_bias", ".mlp.gate.router_bias"),
            (".share_expert.", ".mlp.share_expert."),
        ]
        is_vanilla = any(
            src in k and dst not in k for k in weights for src, dst in remappings
        )

        # Determine "decoder layer count" so we know which layer indices to
        # drop (MTP heads sit after the decoder).
        # For Step 3.7 the text config has 45 decoder layers; any layer
        # >= 45 is an MTP head and must be skipped.
        num_decoder_layers = 45
        new_weights = {}
        for k, v in weights.items():
            if ".mtp" in k:
                continue
            if "model.layers." in k:
                parts = k.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    if int(parts[2]) >= num_decoder_layers:
                        continue
            for src, dst in remappings:
                if src in k and dst not in k:
                    k = k.replace(src, dst)
                    break
            if is_vanilla and k.endswith(".weight") and "norm" in k:
                v = v + 1
            new_weights[k] = v
        return new_weights
