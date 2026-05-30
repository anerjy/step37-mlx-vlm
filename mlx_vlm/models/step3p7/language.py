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
    Step3p5DecoderLayer,
    ZeroCenteredRMSNorm,
)
from mlx_lm.models.base import create_attention_mask

from .config import TextConfig


# ============================================================================
# Multi-Token Prediction (MTP / nextn) — mirror of vLLM Step3p5MTP, ported to MLX.
#
# Hikari07jp/Step-3.7-Flash-MTP-draft (5.92 GB BF16) extracts 3 MTP layers from
# the upstream Step-3.7-Flash BF16 checkpoint (originally trained by stepfun-ai;
# discarded by the NVFP4 release). Reference vLLM source:
#   vllm/model_executor/models/step3p5_mtp.py — class Step3p5MTP
#
# Per-layer architecture:
#   enorm(embedding_t+1) + hnorm(hidden_t) → eh_proj(2D→D) → mtp_block (full
#   decoder layer) → shared_head.norm → shared_head.head (= lm_head)
#
# Weight key contract (Hikari07jp original BF16 names → MLX-side names):
#   model.layers.{45,46,47}.enorm.weight                → mtp.layers.{45-47}.enorm
#   model.layers.{45,46,47}.hnorm.weight                → mtp.layers.{45-47}.hnorm
#   model.layers.{45,46,47}.eh_proj.weight              → mtp.layers.{45-47}.eh_proj
#   model.layers.{45,46,47}.input_layernorm.weight      → mtp.layers.X.mtp_block.input_layernorm
#   model.layers.{45,46,47}.self_attn.q_proj.weight     → mtp.layers.X.mtp_block.self_attn.q_proj
#   model.layers.{45,46,47}.mlp.gate_proj.weight        → mtp.layers.X.mtp_block.mlp.gate_proj
#   ... (all transformer-block weights get .mtp_block. injected after .X.)
#   model.layers.47.transformer.shared_head.norm.weight → mtp.layers.47.shared_head_norm
#   model.layers.47.transformer.shared_head.output      → (shared with backbone lm_head)
#
# Spec dec dispatch (vLLM cycles spec_step_idx % num_mtp_layers, picking
# layer 45 on step 0, 46 on step 1, 47 on step 2 for K=3; K=1 only uses 45):
#   logits = mtp.compute_logits(mtp_forward(hidden, next_token_ids, spec_step_idx))
#
# This class block is dormant until an oMLX patch (patches/mlx_lm_mtp/
# step3p5_model.py) wires `LanguageModel.mtp_forward` into the scheduler's
# verify/accept loop. Without that wiring, MTP weights are still loaded
# (so sanitize doesn't drop them) but the standard __call__ path ignores them.
# ============================================================================


class MTPLayer(nn.Module):
    """One MTP head layer — mirror of vLLM Step3p5AMultiTokenPredictorLayer.

    Combines the previous-step hidden state with the next-token embedding
    (via two RMSNorms + a fused Linear), then runs a full decoder layer
    on the fused representation.
    """

    def __init__(self, args: Step3p5Args, layer_idx: int):
        super().__init__()
        eps = args.rms_norm_eps
        H = args.hidden_size
        self.enorm = ZeroCenteredRMSNorm(H, eps=eps)
        self.hnorm = ZeroCenteredRMSNorm(H, eps=eps)
        self.eh_proj = nn.Linear(H * 2, H, bias=False)
        # mtp_block is a full Step3p5 decoder layer. The MTP layer_idx lives
        # at position `num_hidden_layers + offset` (45/46/47 for Step 3.7),
        # but we hand the underlying Step3p5DecoderLayer the original idx
        # so its layer-type lookup / rope theta indexing match upstream.
        self.mtp_block = Step3p5DecoderLayer(args, layer_idx)
        # shared_head.norm is per-MTP-layer; shared_head.output (lm_head) is
        # tied to backbone.lm_head — we don't allocate it here.
        self.shared_head_norm = ZeroCenteredRMSNorm(H, eps=eps)

    def __call__(
        self,
        inputs_embeds: mx.array,
        previous_hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Any = None,
    ) -> mx.array:
        e = self.enorm(inputs_embeds)
        h = self.hnorm(previous_hidden_states)
        fused = self.eh_proj(mx.concatenate([e, h], axis=-1))
        return self.mtp_block(fused, mask=mask, cache=cache)


class MTPModule(nn.Module):
    """Multi-Token Predictor module — mirror of vLLM Step3p5AMultiTokenPredictor.

    Holds `num_nextn_predict_layers` MTPLayer instances indexed by their
    absolute layer position (`num_hidden_layers .. num_hidden_layers +
    num_mtp - 1`). vLLM uses a ModuleDict keyed by str(layer_idx); we
    mirror that so weight key indices line up directly.
    """

    def __init__(self, args: Step3p5Args, mtp_start_layer_idx: int, num_mtp_layers: int):
        super().__init__()
        self.mtp_start_layer_idx = mtp_start_layer_idx
        self.num_mtp_layers = num_mtp_layers
        # MLX nn.Module doesn't have nn.ModuleDict; use a plain dict of
        # str-keyed children. nn.Module registers them via __setattr__
        # only if assigned as direct attrs OR appended to a list — store
        # in `self.layers` as a list, expose by absolute idx via a property.
        self.layers = [
            MTPLayer(args, mtp_start_layer_idx + k)
            for k in range(num_mtp_layers)
        ]

    def __call__(
        self,
        input_embeds: mx.array,
        previous_hidden_states: mx.array,
        spec_step_idx: int = 0,
        mask: Optional[mx.array] = None,
        cache: Any = None,
    ) -> mx.array:
        """Run one MTP step. spec_step_idx is wrapped mod num_mtp_layers."""
        current = spec_step_idx % self.num_mtp_layers
        layer = self.layers[current]
        h = layer(input_embeds, previous_hidden_states, mask=mask, cache=cache)
        return layer.shared_head_norm(h)


def _make_step3p5_args(text_cfg: TextConfig) -> Step3p5Args:
    """Project mlx_vlm-side TextConfig into mlx_lm-side ModelArgs."""
    # Step3p5Args is a dataclass; pick only fields it accepts.
    fields = {f.name for f in Step3p5Args.__dataclass_fields__.values()}
    src = asdict(text_cfg)
    return Step3p5Args(**{k: v for k, v in src.items() if k in fields})


class LanguageModel(nn.Module):
    """Wraps step3p5.Model with an inputs_embeds-aware __call__.

    Also holds an optional MTPModule head for speculative decoding (oMLX
    `mtp_enabled: True`). MTP module is built unconditionally when the
    config declares `num_nextn_predict_layers > 0` — sanitize then routes
    `model.layers.{45..47}.*` weights to `mtp.layers.{0..2}.*`. If MTP
    weights are absent from the checkpoint, sanitize ignores `mtp.*`
    routing and the head stays randomly initialized but unused — `__call__`
    is the only consumer of the backbone and doesn't touch `self.mtp`.
    """

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
        # MTP head — present iff config has num_nextn_predict_layers > 0.
        num_mtp = getattr(config, "num_nextn_predict_layers", 0) or 0
        if num_mtp > 0:
            self.mtp = MTPModule(
                args,
                mtp_start_layer_idx=args.num_hidden_layers,
                num_mtp_layers=num_mtp,
            )
        else:
            self.mtp = None

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

    def make_mtp_cache(self):
        """One KVCache per MTP layer (each MTP block is a full decoder layer).

        Step3p5DecoderLayer's attention type is determined by `layer_idx`
        — layers 45/46/47 land on `is_sliding` according to step3p5's
        layer_types config. We mirror that by asking each MTP block's
        underlying decoder layer whether it's sliding.
        """
        from mlx_lm.models.cache import KVCache, RotatingKVCache
        if self.mtp is None:
            return []
        caches = []
        for layer in self.mtp.layers:
            block = layer.mtp_block
            if getattr(block, "is_sliding", False):
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
        return_hidden: bool = False,
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

        # Capture pre-norm hidden state — MTP consumes the un-normed value
        # (vLLM Step3p5MTP takes the residual stream out of the last
        # backbone layer before final norm; the per-MTP-layer `hnorm`
        # supplies its own normalization on the way in).
        pre_norm_hidden = h
        h = body.norm(h)
        logits = self.lm_head(h)
        if return_hidden:
            return logits, pre_norm_hidden
        return logits

    def mtp_forward(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        mtp_cache: Optional[List[Any]] = None,
        spec_step_idx: int = 0,
    ) -> mx.array:
        """Run one MTP step → logits for next-token prediction.

        Args:
            hidden_states: pre-norm hidden state from the most recent
              backbone forward (shape `(B, T, H)` — typically `T=1` at
              decode time for K=1 spec).
            next_token_ids: ids whose embedding we condition on (the
              token the backbone just emitted). Shape `(B, T)`.
            mtp_cache: list returned by `make_mtp_cache` (or None).
            spec_step_idx: which MTP layer to use (cycled mod num_mtp_layers).

        Returns:
            logits over vocab — `(B, T, V)`. Caller picks argmax /
            samples to get the draft token for verification.
        """
        if self.mtp is None:
            raise RuntimeError(
                "mtp_forward called but LanguageModel.mtp is None — "
                "check num_nextn_predict_layers in config."
            )
        embeds = self.model.embed_tokens(next_token_ids)
        # Which MTP layer fires this step
        current = spec_step_idx % self.mtp.num_mtp_layers
        # Per-layer cache (KVCache built in make_mtp_cache)
        c = mtp_cache[current] if mtp_cache is not None else None
        fused = self.mtp(
            embeds,
            hidden_states,
            spec_step_idx=spec_step_idx,
            mask=None,  # decode-time single-token, causal trivially satisfied
            cache=c,
        )
        # `fused` is already shared_head_norm'd by MTPModule — apply lm_head.
        return self.lm_head(fused)

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

        # Layer index threshold — Step 3.7 has 45 backbone layers + up to 3 MTP
        # layers (45, 46, 47). vLLM Step3p5MTP._rewrite_spec_layer_name routes
        # these to `model.mtp_block.*`; we route to `mtp.layers.{X-45}.*` so
        # MTPModule.layers[k] indices match its 0-based list.
        num_decoder_layers = 45
        # Names that stay at the MTPLayer level (don't get .mtp_block. injected):
        mtp_direct_names = ("enorm", "hnorm", "eh_proj", "shared_head")
        new_weights = {}
        for k, v in weights.items():
            # Route layer 45+ to mtp.layers.{idx-45}.* — vLLM rewrite logic
            if "model.layers." in k:
                parts = k.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    layer_idx = int(parts[2])
                    if layer_idx >= num_decoder_layers:
                        rel_idx = layer_idx - num_decoder_layers
                        # `mlx.nn.Module` exposes a list as `attr.<idx>` keys
                        # when loaded via load_weights, so `mtp.layers.0.<...>`
                        # matches the MTPLayer at list index 0.
                        new_prefix = f"mtp.layers.{rel_idx}"
                        suffix = ".".join(parts[3:])
                        # Strip optional `transformer.` segment first (vLLM rewrite
                        # rule). Each MTP layer's shared_head ships under
                        # `model.layers.X.transformer.shared_head.*` in the
                        # Hikari07jp BF16 export — that's just a remnant of the
                        # original vLLM layer-block wrapping.
                        if suffix.startswith("transformer."):
                            suffix = suffix[len("transformer."):]
                        if any(suffix.startswith(n) for n in mtp_direct_names):
                            # enorm / hnorm / eh_proj / shared_head sit directly
                            # on the MTPLayer (not inside mtp_block).
                            if suffix.startswith("shared_head."):
                                # `shared_head.output` → tied to backbone.lm_head; drop
                                if "shared_head.output" in suffix:
                                    continue
                                # `shared_head.norm.weight` → `shared_head_norm.weight`
                                suffix = suffix.replace("shared_head.norm", "shared_head_norm")
                            new_k = f"{new_prefix}.{suffix}"
                        else:
                            # transformer-block weights go under .mtp_block.
                            new_k = f"{new_prefix}.mtp_block.{suffix}"
                        # Apply the same MoE→switch_mlp remap to the rewritten key
                        for src, dst in remappings:
                            if src in new_k and dst not in new_k:
                                new_k = new_k.replace(src, dst)
                                break
                        if is_vanilla and new_k.endswith(".weight") and "norm" in new_k:
                            v = v + 1
                        new_weights[new_k] = v
                        continue
            # Pre-2026-05-30 behaviour: drop bare `.mtp` keys that aren't from
            # the layer-45+ MTP weights (none expected; defensive).
            if ".mtp." in k or k.startswith("mtp."):
                # Unrouted MTP keys (e.g. legacy `mtp.head.*`) — drop, the new
                # MTP module reuses backbone embed/lm_head.
                continue
            # Standard backbone weight — apply moe→switch_mlp remap
            for src, dst in remappings:
                if src in k and dst not in k:
                    k = k.replace(src, dst)
                    break
            if is_vanilla and k.endswith(".weight") and "norm" in k:
                v = v + 1
            new_weights[k] = v
        return new_weights
