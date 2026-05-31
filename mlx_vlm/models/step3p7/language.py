"""Step 3.7 language backbone wrapper for mlx_vlm.

The underlying architecture is identical to step3p5 (45-layer Qwen3.6-style
MoE with shared experts). We reuse mlx_lm.models.step3p5 directly and add
the inputs_embeds plumbing that the VLM call path needs.
"""
import math
from dataclasses import asdict
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.step3p5 import (
    Model as Step3p5Model,
    ModelArgs as Step3p5Args,
    Step3p5Model as Step3p5Body,
    Step3p5DecoderLayer,
    ZeroCenteredRMSNorm,
)
from mlx_lm.models.cache import KVCache
from mlx_lm.models.base import create_attention_mask

from .config import TextConfig


# ============================================================================
# PyramidKV — Cai et al. 2024 (arxiv 2406.02069), MLX port.
#
# After prefill ends, full-attention layers' KV caches are compressed by
# scoring each cached token's importance via attention from the last α
# "instruction" tokens. Per-layer arithmetic-decay budget (bottom layers
# keep more, top layers less). Empirically: 12% cache retention matches
# full-cache quality (62.6 needle vs 65.0 full at 100K context).
#
# Reference implementation:
#   https://github.com/Zefan-Cai/KVCache-Factory/blob/main/pyramidkv/pyramidkv_utils.py
#
# Design choices for Step-3.7 hybrid architecture:
#   - Only compress the 12 full-attention layers. The 33 sliding-window=512
#     layers are already O(window), no compression benefit.
#   - GQA aggregation: scores summed across the n_heads-per-kv-head group
#     before top-K, so each kv-head gets a single coherent index list.
#   - Last `window_size` tokens always preserved verbatim (the standard
#     "recent" guarantee from the original paper).
#   - logical_offset decoupled from buffer size: cached K vectors retain
#     their original RoPE positions; new K post-compression continues at
#     the ORIGINAL position (e.g. 39061 after compressing a 39060-token
#     prefill to 5000-slot cache).
# ============================================================================


def _pyramid_avg_pool_1d(x: mx.array, kernel_size: int = 5) -> mx.array:
    """1D average pool along last axis with SAME zero padding."""
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    pads = [(0, 0)] * (x.ndim - 1) + [(pad, pad)]
    xp = mx.pad(x, pads)
    pieces = [xp[..., i : i + x.shape[-1]] for i in range(kernel_size)]
    return mx.stack(pieces, axis=0).sum(axis=0) / kernel_size


def _pyramid_compress_layer(
    keys: mx.array,
    values: mx.array,
    q_anchor: mx.array,
    budget: int,
    window_size: int,
    num_key_value_groups: int,
    kernel_size: int = 5,
) -> Tuple[mx.array, mx.array]:
    """Compress one layer's KV cache using PyramidKV scoring.

    See module-level docstring for design notes. Returns (K, V) where the
    sequence dim is min(N, budget). When N <= budget, returns inputs
    unchanged.
    """
    B, n_kv_heads, N, head_dim = keys.shape
    if N <= budget:
        return keys, values
    keep_from_past = budget - window_size
    assert keep_from_past > 0, "budget must exceed window_size"

    # GQA: expand kv to n_heads for score compute, then aggregate per kv-head
    n_heads = q_anchor.shape[1]
    if num_key_value_groups > 1:
        keys_for_score = mx.repeat(keys, num_key_value_groups, axis=1)
    else:
        keys_for_score = keys

    # Q · K^T / √d, scores from the last `window_size` queries
    scale = 1.0 / math.sqrt(head_dim)
    attn = mx.matmul(q_anchor, keys_for_score.swapaxes(-1, -2)) * scale

    # Causal mask on the tail × tail sub-block (window queries vs window keys)
    causal = mx.triu(mx.ones((window_size, window_size)) * -1e9, k=1)
    tail = attn[..., -window_size:] + causal
    attn = mx.concatenate([attn[..., :-window_size], tail], axis=-1)
    attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(keys.dtype)

    # Sum window-Q attention to each pre-window K position → score
    score = attn[..., :-window_size].sum(axis=-2)  # (B, n_heads, N-ws)
    score = _pyramid_avg_pool_1d(score, kernel_size=kernel_size)

    # GQA aggregation: sum scores across each kv-head's group, then per-kv-head
    # top-K. This gives each kv-head one coherent index list.
    if num_key_value_groups > 1:
        score = score.reshape(B, n_kv_heads, num_key_value_groups, -1).sum(axis=2)
    # score now (B, n_kv_heads, N-ws)

    neg = -score
    topk_unsorted = mx.argpartition(neg, kth=keep_from_past - 1, axis=-1)[..., :keep_from_past]
    topk_sorted = mx.sort(topk_unsorted, axis=-1)  # ascending temporal order

    # Gather K, V at selected indices
    indices = mx.broadcast_to(
        mx.expand_dims(topk_sorted, axis=-1),
        (B, n_kv_heads, keep_from_past, head_dim),
    )
    k_past = mx.take_along_axis(keys[..., :-window_size, :], indices, axis=2)
    v_past = mx.take_along_axis(values[..., :-window_size, :], indices, axis=2)
    k_window = keys[..., -window_size:, :]
    v_window = values[..., -window_size:, :]
    return (
        mx.concatenate([k_past, k_window], axis=2),
        mx.concatenate([v_past, v_window], axis=2),
    )


def _pyramid_budget(
    layer_idx: int,
    num_full_layers: int,
    max_capacity_prompt: int,
    window_size: int,
    seq_len: int,
    beta: int = 20,
) -> int:
    """Per-layer budget (arithmetic decay, bottom=most, top=least).

    Returns total slots to keep INCLUDING the always-preserved window.
    Returns seq_len when no compression needed.
    """
    if seq_len <= max_capacity_prompt:
        return seq_len
    min_num = (max_capacity_prompt - window_size) // beta
    max_num = (max_capacity_prompt - window_size) * 2 - min_num
    if max_num >= seq_len - window_size:
        max_num = seq_len - window_size
        min_num = (max_capacity_prompt - window_size) * 2 - max_num
    step = (max_num - min_num) // max(num_full_layers - 1, 1)
    keep_past = max_num - layer_idx * step
    keep_past = max(keep_past, 1)
    return keep_past + window_size


# Module-level capture flag. When non-zero, Step3p5Attention's patched
# __call__ saves Q[..., -window_size:, :] (post-RoPE) into
# `self._pyramid_last_q` on each call. Off by default — zero cost.
_PYRAMID_CAPTURE_WINDOW: int = 0


def _install_pyramid_q_capture() -> None:
    """One-shot monkey-patch: make Step3p5Attention save the tail Q.

    Called from LanguageModel.__init__ when pyramidkv is enabled. The
    patched __call__ saves Q[..., -_PYRAMID_CAPTURE_WINDOW:, :] (post-RoPE,
    post-q_norm, post-transpose) on each forward when capture is active.

    Idempotent: re-running is a no-op once the patch marker is set.
    """
    from mlx_lm.models.step3p5 import Step3p5Attention
    if getattr(Step3p5Attention, "_pyramid_capture_patched", False):
        return
    original_call = Step3p5Attention.__call__

    def patched_call(self, x, mask=None, cache=None):
        # Reuse the entire original forward; capture Q post-everything.
        ws = _PYRAMID_CAPTURE_WINDOW
        if ws > 0 and x.shape[1] >= ws:
            # Reproduce the Q computation path INLINE so we can save the
            # post-RoPE Q without re-running attention. Then call the
            # original __call__ for actual attention. Slightly wasteful
            # (Q computed twice during prefill chunk that fires capture)
            # but only matters for the chunks where capture runs (the
            # last chunk of prefill at most).
            B, L, _ = x.shape
            q_full = self.q_proj(x)
            q_full = self.q_norm(
                q_full.reshape(B, L, self.num_heads, -1)
            ).transpose(0, 2, 1, 3)
            if cache is not None:
                q_full = self.rope(q_full, offset=cache.offset)
            else:
                q_full = self.rope(q_full)
            self._pyramid_last_q = q_full[..., -ws:, :]
        return original_call(self, x, mask=mask, cache=cache)

    Step3p5Attention.__call__ = patched_call
    Step3p5Attention._pyramid_capture_patched = True


def _set_pyramid_capture_active(window_size: int) -> None:
    """Toggle capture; window_size=0 to disable."""
    global _PYRAMID_CAPTURE_WINDOW
    _PYRAMID_CAPTURE_WINDOW = int(window_size)


class PyramidKVCache(KVCache):
    """KVCache with a LOGICAL offset that survives PyramidKV compression.

    After compression, ``self.keys.shape[-2]`` shrinks (e.g. 39060 → 5000)
    but the next decode step should still RoPE-position-encode at the
    original 39061. Parent KVCache.update_and_fetch resets self.offset to
    buffer-size every call; we override to advance offset by S instead.

    Cached K vectors retain their original RoPE-positions (applied during
    prefill at insertion time), so Q·K dot products use correct relative
    positional angles even though the slot order may differ from absolute
    position order.
    """

    def __init__(self):
        super().__init__()
        self._logical_offset = 0
        self._compressed = False

    def update_and_fetch(self, keys, values):
        S = keys.shape[-2]
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=-2)
            self.values = mx.concatenate([self.values, values], axis=-2)
        # KEY DIFFERENCE: advance logical offset by S, NOT to buffer size.
        # When self._compressed is True (post-PyramidKV), buffer < logical.
        self._logical_offset += S
        return self.keys, self.values

    @property
    def offset(self):
        return self._logical_offset

    @offset.setter
    def offset(self, v):
        self._logical_offset = int(v)

    def install_compressed(self, new_keys: mx.array, new_values: mx.array) -> None:
        """Replace cached tensors with their compressed versions.

        Preserves logical offset — caller must NOT touch ``self.offset``
        after this call.
        """
        self.keys = new_keys
        self.values = new_values
        self._compressed = True

    def is_trimmable(self):
        # Post-compression trim semantics are subtle; conservatively disable.
        return not self._compressed and self._logical_offset < (
            self.keys.shape[-2] if self.keys is not None else 0
        )

    def trim(self, n):
        # Pre-compression trim: physically drop n tail slots.
        if self._compressed or self.keys is None:
            return 0
        n = min(n, self.keys.shape[-2])
        if n <= 0:
            return 0
        self.keys = self.keys[..., :-n, :]
        self.values = self.values[..., :-n, :]
        self._logical_offset -= n
        return n

    # ---- SSD persistence (oMLX paged_ssd_cache) -----------------------------
    #
    # 2026-05-31 second iteration:
    # Earlier attempt returned a 4-tuple from ``.state`` to ship
    # ``_logical_offset`` + ``_compressed`` alongside (keys, values). That
    # broke ``prefix_cache._extract_block_tensor_slice`` which does
    # ``keys, values = state`` and raised "too many values to unpack
    # (expected 2)" for every block save → no per-block-hash entries
    # written → next request's cache lookup misses.
    #
    # Final shape: ``.state`` stays as the 2-tuple expected by all of
    # oMLX's block tensor handling. ``_logical_offset`` and ``_compressed``
    # are exposed via ``meta_state`` (the parallel channel that
    # RotatingKVCache uses for ``(keep, max_size, offset, _idx)``).
    # ``meta_state.setter`` re-hydrates both fields on restore. SSD
    # snapshots save+load via ``cache.state`` AND ``cache.meta_state`` per
    # the ``_BaseCache`` contract.
    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, v):
        if v is None:
            return
        self.keys, self.values = v[0], v[1]
        # Parent KVCache.state setter would do
        # ``self.offset = self.keys.shape[-2]`` which routes through our
        # offset.setter and clobbers _logical_offset to the *physical*
        # buffer size. Skip that — meta_state.setter restores it
        # correctly from the saved (logical_offset, compressed) tuple.

    @property
    def meta_state(self) -> tuple:
        return (int(self._logical_offset), int(bool(self._compressed)))

    @meta_state.setter
    def meta_state(self, v) -> None:
        if v is None or v == "" or len(v) == 0:
            return
        if len(v) >= 2:
            try:
                self._logical_offset = int(v[0])
                self._compressed = bool(int(v[1]))
            except (TypeError, ValueError):
                # Legacy or malformed meta — fall back to assuming the
                # cache is uncompressed.
                self._logical_offset = (
                    self.keys.shape[-2] if self.keys is not None else 0
                )
                self._compressed = False


# Note: do NOT monkey-patch mlx_lm.generate._make_cache to pass PyramidKVCache
# through as a "batch cache". Tried 2026-05-30: it makes _make_cache succeed
# but a downstream scheduler `_extend_cache` then calls `.extend()` on
# PyramidKVCache (which has no such method), corrupting the cache and 500'ing.
# Original behavior — `_make_cache` raises ValueError → MTP reconcile drops
# gracefully → standard decode continues on the compressed cache — is correct.
# MTP loses speculative decoding after an extend, but the request still
# completes.


def _register_pyramidkv_cache_handler() -> None:
    """Register a non-sliceable handler for PyramidKVCache with oMLX's
    CacheTypeRegistry. Without this, oMLX defaults to KVCacheHandler which
    reports ``supports_block_slicing=True``, causing the prefix-cache save
    path to slice the post-PKV compressed buffer (e.g. 8K slots) at original
    prefill block positions (e.g. block 32 = tokens 16384–16896). The slice
    is out-of-bounds, the save produces empty / wrong block data, the SSD
    write either fails or stores garbage. On the next request, the trie
    matches the prefix hashes but ``load_block_with_metadata`` returns None
    for block 1+, the reconstruction loop short-circuits at the first miss,
    and the surviving block 0 then trips the RotatingKVCache placeholder
    reject because no boundary snapshot was captured at the 512-token mark.

    Net symptom: ``Partial cache reconstruction: 1/N blocks, 512 tokens``
    followed by ``RotatingKVCache layer 1: ... Rejecting cache``.

    A non-sliceable handler routes through oMLX's "last-block-only +
    boundary-snapshot" path (same lane as RotatingKVCache), which never
    asks the compressed buffer for an out-of-range slice. Same-prefix
    requests then restore the cache as a single full-state object (logical
    offset preserved via the 4-tuple state), exactly like RotatingKVCache.

    Verified 2026-05-31: PKV-off cold→warm gave 97.8% cache hit. With this
    handler PKV-on should match that pattern.

    Idempotent on import; safe to call multiple times.
    """
    try:
        from omlx.cache.type_handlers import (
            CacheType,
            CacheTypeHandler,
            CacheStateAxisInfo,
        )
        from omlx.cache.type_registry import CacheTypeRegistry
    except Exception:
        return  # Not running under oMLX (e.g. plain mlx-vlm import). No-op.

    if "PyramidKVCache" in getattr(CacheTypeRegistry, "_class_name_map", {}):
        return  # already registered

    class PyramidKVCacheHandler(CacheTypeHandler):
        """Mirror of RotatingKVCacheHandler with PyramidKVCache reconstruction.

        PyramidKVCache post-compression has a physical buffer (≤ pyramidkv_
        max_capacity) that does NOT match its logical offset (the original
        prompt length). The buffer's slot order is not position-monotone —
        slots are picked top-k by attention score — so per-block slicing
        along the sequence axis produces empty or wrong data.
        """

        @property
        def cache_type(self) -> CacheType:
            # No dedicated enum value; reuse KVCACHE for registry slot.
            # The lookup path uses class_name → handler directly, so this
            # tag only affects metadata serialization.
            return CacheType.KVCACHE

        @property
        def supports_block_slicing(self) -> bool:
            return False  # ← the load-bearing flag

        def get_state_axis_info(self) -> tuple:
            return (
                CacheStateAxisInfo(name="keys", sequence_axis=2, sliceable=False),
                CacheStateAxisInfo(name="values", sequence_axis=2, sliceable=False),
            )

        def extract_state(self, cache_obj):
            # PyramidKVCache.state is now a 2-tuple; _logical_offset /
            # _compressed live in meta_state.
            keys, values = cache_obj.state
            meta = cache_obj.meta_state
            return {
                "keys": keys,
                "values": values,
                "logical_offset": int(meta[0]) if meta and len(meta) >= 1 else int(
                    getattr(cache_obj, "_logical_offset", 0)
                ),
                "compressed": bool(int(meta[1])) if meta and len(meta) >= 2 else bool(
                    getattr(cache_obj, "_compressed", False)
                ),
                "meta_state": meta,
                "cache_type": "PyramidKVCache",
            }

        def get_seq_len(self, state: dict) -> int:
            # Effective sequence length is the LOGICAL offset (pre-
            # compression count), not the physical buffer size.
            return int(state.get("logical_offset", 0))

        def slice_state(self, state: dict, start_idx: int, end_idx: int):
            # Non-sliceable: return full state. Mirrors RotatingKVCacheHandler.
            return {
                "keys": state.get("keys"),
                "values": state.get("values"),
                "logical_offset": state.get("logical_offset", 0),
                "compressed": state.get("compressed", False),
                "meta_state": state.get("meta_state", ()),
                "is_full_state": True,
                "cache_type": "PyramidKVCache",
            }

        def concatenate_states(self, states: list) -> dict:
            if not states:
                return {}
            # Use the most recent state — same as RotatingKVCache.
            latest = states[-1]
            return {
                "keys": latest.get("keys"),
                "values": latest.get("values"),
                "logical_offset": latest.get("logical_offset", 0),
                "compressed": latest.get("compressed", False),
                "meta_state": latest.get("meta_state", ()),
                "is_full_state": True,
                "cache_type": "PyramidKVCache",
            }

        def reconstruct_cache(self, state: dict, meta_state=None):
            keys = state.get("keys")
            values = state.get("values")
            if keys is None or values is None:
                return None
            cache = PyramidKVCache()
            cache.keys = keys
            cache.values = values
            cache._logical_offset = int(state.get("logical_offset", 0))
            cache._compressed = bool(state.get("compressed", False))
            return cache

    handler = PyramidKVCacheHandler()

    # Register class-name → handler. Two integration points needed:
    # (1) ``_class_name_map`` (route ``get_handler_by_class_name`` for our class)
    # (2) Direct override of ``get_handler_by_class_name`` so that even though
    #     "PyramidKVCache" maps to ``CacheType.KVCACHE``, the lookup returns
    #     our non-sliceable handler instead of the registered KVCacheHandler.
    CacheTypeRegistry._class_name_map["PyramidKVCache"] = CacheType.KVCACHE

    original_get_by_name = CacheTypeRegistry.get_handler_by_class_name

    def _patched_get_handler_by_class_name(cls, class_name):
        if class_name == "PyramidKVCache":
            return handler
        # Preserve the original classmethod's binding by calling through __func__.
        try:
            return original_get_by_name.__func__(cls, class_name)
        except AttributeError:
            # Already an unbound function (unlikely but defensive).
            return original_get_by_name(class_name)

    CacheTypeRegistry.get_handler_by_class_name = classmethod(
        _patched_get_handler_by_class_name
    )


_register_pyramidkv_cache_handler()


def _patch_omlx_emit_prefill_boundary_snapshot() -> None:
    """Fix oMLX's chunked-prefill VLM path so boundary snapshots actually
    get persisted instead of being silently dropped on the floor.

    THE BUG (oMLX 0.3.12 ``omlx/scheduler.py`` at the time of writing):

    - Non-chunked ``_do_external_prefill`` sets up a ``request_id → temp_uid``
      mapping at scheduler.py:5035-5037 BEFORE calling itself, so
      ``_emit_prefill_boundary_snapshot``'s lookup
      ``uid = self.request_id_to_uid.get(request.request_id, -1)`` returns a
      real uid and the snapshot lands in ``_boundary_cache_snapshots``.
    - The chunked path (``_begin_prefill`` → ``_step_prefill_chunk``) is
      missing that setup. Any model that hits the chunked path — which
      includes every VLM model on text-only prompts longer than
      ``prefill_step_size`` (Step-3.7-Flash with a 20 K prompt is the
      canonical case) — emits boundary snapshots with ``uid = -1``.
    - ``_on_prefill_boundary_snapshot`` then does
      ``self.uid_to_request_id.get(-1) is None`` → ``return`` → snapshot
      thrown away.

    Symptom (with PKV ON): cross-request prefix cache reuse looks completely
    broken. Trie matches the prefix hashes, ``load_block_with_metadata``
    returns None for every block past block 0 (no per-block-hash entry was
    ever written), reconstruction reports ``1/N blocks, 512 tokens``,
    walk-back sees RotatingKVCache placeholder in block 0, ``Rejecting
    cache to prevent stale sliding-window state``, request re-prefills the
    full prompt. 2026-05-31 REDACTED-IP fan-out reported 130 such events in a
    single day, ~7 GB of prefix-cache reuse lost. With this patch +
    ``PyramidKVCacheHandler`` + 2-tuple ``.state`` (Invariants 11-12 below),
    PKV-on cache reuse is identical to PKV-off: 97.8% cached on a same-
    prompt resend, 20 K prefill cold→warm 60 s → 2.6 s (23× speedup).

    Fix: override ``_emit_prefill_boundary_snapshot`` to bypass the uid
    round-trip entirely. Build ``snapshot_cache`` the same way the original
    does (keep non-sliceable layers, None-out sliceable ones), then save
    directly under ``request.request_id`` — the actual storage key used by
    ``_get_boundary_store_override`` and ``_BoundarySnapshotProvider`` at
    restoration time.

    Idempotent. Safe to call on every model load.
    """
    try:
        from omlx.scheduler import Scheduler as _Scheduler  # type: ignore
        from omlx import scheduler as _sched_mod
    except Exception:
        return

    if getattr(_Scheduler, "_pkv_emit_bypass_patched", False):
        return
    _Scheduler._pkv_emit_bypass_patched = True

    _KNOWN_SLICEABLE = _sched_mod._KNOWN_SLICEABLE_CACHE_TYPES

    def _wrapped_emit(self, request, prompt_cache, total_tokens):
        # Build snapshot_cache the same way the original does:
        # keep non-sliceable layers (PyramidKVCache, RotatingKVCache, …),
        # None-out sliceable ones.
        try:
            snapshot_cache = [
                c if type(c).__name__ not in _KNOWN_SLICEABLE else None
                for c in prompt_cache
            ]
        except Exception:
            return

        # Mirror the original gating exactly.
        if not self._cache_list_needs_boundary_snapshot(snapshot_cache):
            return

        block_size = self.config.paged_cache_block_size
        if block_size <= 0 or total_tokens <= 0 or (total_tokens % block_size != 0):
            return

        request_id = getattr(request, "request_id", None)
        if request_id is None:
            return

        if request_id not in self._boundary_cache_snapshots:
            self._boundary_cache_snapshots[request_id] = {}

        # Skip duplicate at the same token_count.
        if total_tokens in self._boundary_cache_snapshots[request_id]:
            return

        # Offload to SSD when available; in-memory fallback otherwise.
        if self._boundary_snapshot_store is not None:
            try:
                with self._phase_timer("boundary_snapshot_save"):
                    saved = self._boundary_snapshot_store.save(
                        request_id,
                        total_tokens,
                        snapshot_cache,
                        self._extract_cache_states,
                    )
            except Exception:
                saved = False
            if saved:
                self._boundary_cache_snapshots[request_id][total_tokens] = None
            else:
                self._boundary_cache_snapshots[request_id][total_tokens] = snapshot_cache
        else:
            self._boundary_cache_snapshots[request_id][total_tokens] = snapshot_cache

        self._boundary_snapshot_required = True

    _Scheduler._emit_prefill_boundary_snapshot = _wrapped_emit


_patch_omlx_emit_prefill_boundary_snapshot()


# ============================================================================


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
        # vLLM Step3p5MTP's SharedHead contains BOTH a norm AND a per-MTP-layer
        # head (Linear projection to vocab). Verified 2026-05-30 by inspecting
        # Hikari07jp's shard: the three shared_head.output weights for layers
        # 45/46/47 are non-identical (per-layer means 0.000029/0.000008/0.000034
        # — slightly different patterns), so they CANNOT be tied to the
        # backbone lm_head. Each MTP layer has its own bf16 lm_head, trained
        # separately during MTP-head fine-tuning. Using the backbone's 4-bit
        # quantized lm_head for the MTP forward path was the missing piece
        # that capped accept rate at 2-4%.
        self.shared_head_norm = ZeroCenteredRMSNorm(H, eps=eps)
        self.shared_head_output = nn.Linear(H, args.vocab_size, bias=False)

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
        # MTP-side embed_tokens — bf16, separate from the backbone's 4-bit
        # quantized embedding. vLLM Step3p5AMultiTokenPredictor keeps its
        # own copy via VocabParallelEmbedding; Hikari07jp's shard ships
        # model.embed_tokens.weight which we route here. Empirically:
        # reusing backbone's 4-bit embedding for the MTP forward pass
        # introduces mixed-precision drift that may explain the low (<5%)
        # accept rate we observed in the first port — try sourcing the
        # MTP head's input embeddings from its own bf16 table.
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
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

        # PyramidKV: install Q-anchor capture monkey-patch and activate
        # capture iff config.pyramidkv_max_capacity > 0.
        self._pyramid_cap = int(getattr(config, "pyramidkv_max_capacity", 0) or 0)
        self._pyramid_ws = int(getattr(config, "pyramidkv_window_size", 32) or 32)
        self._pyramid_kernel = int(getattr(config, "pyramidkv_kernel_size", 5) or 5)
        self._pyramid_beta = int(getattr(config, "pyramidkv_beta", 20) or 20)
        if self._pyramid_cap > 0:
            _install_pyramid_q_capture()
            _set_pyramid_capture_active(self._pyramid_ws)
        # Per-cache `_compressed` flag (on PyramidKVCache) is the source of
        # truth for idempotency — no per-model flag needed.

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        # Reuse step3p5.Model's cache builder by reconstructing it via the
        # underlying layers.
        #
        # Cache type selection (priority order):
        #   1. pyramidkv_max_capacity > 0 → PyramidKVCache for full-attn
        #      layers, plain RotatingKVCache for sliding (PyramidKV needs
        #      unbounded prefill cache to score from). Sink ignored.
        #   2. attention_sink_size + attention_sink_window > 0 → sink
        #      RotatingKVCache for full-attn (StreamingLLM-style).
        #   3. Default → unbounded KVCache for full-attn, sliding
        #      RotatingKVCache for sliding.
        #
        # Attention-sink hook: when config declares both attention_sink_size
        # and attention_sink_window > 0, the 12 full-attention layers get
        # RotatingKVCache(max_size=window, keep=sink) instead of unbounded
        # KVCache. RotatingKVCache natively supports the StreamingLLM
        # "keep first K" pattern via its ``keep`` param — MLX engineers
        # shipped this already (cache.py:413, 424).
        #
        # Convention: ``attention_sink_window`` is the TOTAL cache slot count
        # (sink slots inclusive). Effective rotating recall = window - sink.
        # We use total-not-rotating-portion because oMLX's paged cache
        # alignment (scheduler.py:1291) requires ALL RotatingKVCache
        # max_sizes to be the same after divide-down to a common block size
        # (block_size must divide every max_size). Step3p5 sliding layers
        # use max_size=512; so sink+full-attn cache max_size must be a
        # multiple of 512. With window=8192 (=16×512), the constraint
        # is satisfied and effective recall is window-sink=8188 tokens.
        # Sliding-attention layers are unchanged; they're already O(512).
        from mlx_lm.models.cache import KVCache, RotatingKVCache
        cfg = self.config
        sink = int(getattr(cfg, "attention_sink_size", 0) or 0)
        window = int(getattr(cfg, "attention_sink_window", 0) or 0)
        pkv_cap = int(getattr(cfg, "pyramidkv_max_capacity", 0) or 0)
        use_sink = sink > 0 and window > sink
        use_pkv = pkv_cap > 0
        caches = []
        for layer in self.model.layers:
            if getattr(layer, "is_sliding", False):
                caches.append(RotatingKVCache(max_size=self._args.sliding_window))
            elif use_pkv:
                caches.append(PyramidKVCache())
            elif use_sink:
                caches.append(RotatingKVCache(max_size=window, keep=sink))
            else:
                caches.append(KVCache())
        return caches

    def pyramid_compress_caches(self, cache_list: List[Any]) -> int:
        """Apply PyramidKV compression to full-attention layer caches.

        Called once at the boundary between prefill and decode. Skips
        sliding-attention layers (already O(window)) and any non-
        PyramidKVCache instance. Reads each layer's captured Q-anchor
        (saved by the install_pyramid_q_capture monkey-patch during PP)
        and replaces (keys, values) in the cache with the compressed
        versions while preserving the logical offset.

        Returns the count of layers compressed (0 if disabled).
        """
        if self._pyramid_cap <= 0:
            return 0

        import logging as _logging
        _log = _logging.getLogger("PYRAMID_KV")

        body = self.model
        ws = self._pyramid_ws
        max_cap = self._pyramid_cap
        beta = self._pyramid_beta
        kernel = self._pyramid_kernel

        # Count full-attention layers (the only ones we compress).
        full_layer_idxs = [
            i for i, lyr in enumerate(body.layers) if not getattr(lyr, "is_sliding", False)
        ]
        num_full = len(full_layer_idxs)
        if num_full == 0:
            return 0

        n_compressed = 0
        # Lazy import — BatchKVCache lives in mlx_lm.models.cache; oMLX
        # swaps our PyramidKVCache instance for BatchKVCache during the
        # request-prep stage (it manages cache lifecycle for batched
        # serving). So we accept BOTH cache types — for BatchKVCache we
        # operate on the underlying tensors directly (cache.keys[..., :idx],
        # cache.values, cache._idx) and trust that cache.offset already
        # tracks the logical token position correctly.
        from mlx_lm.models.cache import BatchKVCache as _BatchKVCache

        for relative_full_idx, abs_idx in enumerate(full_layer_idxs):
            cache = cache_list[abs_idx]
            if cache.keys is None:
                continue
            # Already-compressed caches skip (idempotent). Layer 0's budget
            # can exceed max_cap (arithmetic decay puts ~2× there) so we
            # can't rely on shape-based detection alone.
            if getattr(cache, "_pyramid_done", False):
                continue
            # For BatchKVCache, the "valid" length is cache._idx; for our
            # PyramidKVCache, it's the buffer shape.
            if isinstance(cache, _BatchKVCache):
                seq_len = int(cache._idx)
            else:
                seq_len = cache.keys.shape[-2]
            if seq_len <= max_cap:
                continue
            layer = body.layers[abs_idx]
            q_anchor = getattr(layer.self_attn, "_pyramid_last_q", None)
            if q_anchor is None or q_anchor.shape[-2] < ws:
                # No anchor captured for this layer — skip safely.
                continue

            # Per-layer arithmetic-decay budget (bottom layers larger,
            # top layers smaller — captures the "pyramidal information
            # funneling" property from the paper). Enabled now that
            # LanguageModel.__call__ builds a per-layer mask above.
            budget = _pyramid_budget(
                layer_idx=relative_full_idx,
                num_full_layers=num_full,
                max_capacity_prompt=max_cap,
                window_size=ws,
                seq_len=seq_len,
                beta=beta,
            )
            num_kv_groups = layer.self_attn.num_heads // layer.self_attn.num_kv_heads
            # For BatchKVCache we slice keys/values to the valid range first
            if isinstance(cache, _BatchKVCache):
                cache_k_valid = cache.keys[..., :seq_len, :]
                cache_v_valid = cache.values[..., :seq_len, :]
            else:
                cache_k_valid = cache.keys
                cache_v_valid = cache.values
            new_k, new_v = _pyramid_compress_layer(
                cache_k_valid,
                cache_v_valid,
                q_anchor,
                budget=budget,
                window_size=ws,
                num_key_value_groups=num_kv_groups,
                kernel_size=kernel,
            )
            mx.eval(new_k, new_v)
            if isinstance(cache, _BatchKVCache):
                # Replace buffer + set _idx to new compressed length.
                # cache.offset (per-batch token count for RoPE) stays.
                cache.keys = new_k
                cache.values = new_v
                cache._idx = new_k.shape[-2]
                cache._pyramid_done = True
            else:
                cache.install_compressed(new_k, new_v)
                cache._pyramid_done = True
            # Free the captured Q now that compression is done
            layer.self_attn._pyramid_last_q = None
            n_compressed += 1

        if n_compressed > 0:
            _log.info(
                "[PyramidKV] compressed %d full-attn layers (budget=%d, window=%d)",
                n_compressed, self._pyramid_cap, self._pyramid_ws,
            )
        return n_compressed

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

        # PyramidKV trigger: compresses each full-attn layer's cache from
        # ~N tokens to ~max_capacity_prompt tokens after PP ends. Idempotent
        # (per-cache _pyramid_done flag).
        if self._pyramid_cap > 0 and h.shape[1] <= 16:
            self.pyramid_compress_caches(cache)

        # 2026-05-30 DEBUG: log cache types per layer when prompt is "long enough".
        # Triggered only when env var STEP37_CACHE_DEBUG=1. Uses h.shape (the
        # embedded tensor) so VLM-via-inputs_embeds path also triggers.
        import os as _os
        if _os.environ.get("STEP37_CACHE_DEBUG") == "1":
            try:
                seq_dim = h.shape[1] if hasattr(h, "shape") else 0
            except Exception:
                seq_dim = 0
            if seq_dim > 100:
                import logging as _logging
                _log = _logging.getLogger("STEP37_CACHE")
                _log.warning(
                    "[STEP37 cache] h.shape=%s inputs.shape=%s",
                    tuple(h.shape) if hasattr(h, "shape") else None,
                    tuple(inputs.shape) if inputs is not None and hasattr(inputs, "shape") else None,
                )
                for li, c in enumerate(cache[:6]):
                    ct = type(c).__name__ if c is not None else "None"
                    offset = getattr(c, "offset", "?") if c is not None else "?"
                    max_size = getattr(c, "max_size", "?") if c is not None else "?"
                    keep = getattr(c, "keep", "?") if c is not None else "?"
                    buf = getattr(c, "keys", None)
                    buf_shape = tuple(buf.shape) if buf is not None else None
                    _log.warning(
                        "  layer %d: %s offset=%s max_size=%s keep=%s buffer_shape=%s",
                        li, ct, offset, max_size, keep, buf_shape,
                    )

        # Mask construction. When PyramidKV is active with per-layer
        # pyramid budgets, each full-attention layer can have a DIFFERENT
        # cache size after compression (bottom layers larger budgets, top
        # layers smaller). So build a per-layer mask for full-attn layers.
        # When PyramidKV is off (or uniform budget), all full-attn caches
        # share size and we could fall back to a single mask — but the
        # per-layer build cost is negligible (a few μs per mask), so keep
        # it unconditional for simpler code.
        full_masks: List[Any] = [None] * body.num_layers
        swa_mask = None
        for li, (lyr, c) in enumerate(zip(body.layers, cache)):
            if not lyr.is_sliding:
                full_masks[li] = create_attention_mask(h, c)
        if body._swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[body._swa_idx], window_size=body.args.sliding_window
            )

        for li, (layer, c) in enumerate(zip(body.layers, cache)):
            m = swa_mask if layer.is_sliding else full_masks[li]
            h = layer(h, mask=m, cache=c)

        # MTP receives PRE-norm hidden (matches vLLM Step3p5Model.forward
        # which returns hidden BEFORE applying self.norm). Empirically
        # tested 2026-05-30: post-norm gives 1.0-2.1% accept, pre-norm
        # gives 2.6-4.2% — pre-norm is correct.
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
        # Use the MTP module's dedicated bf16 embedding (not backbone's
        # 4-bit quantized embedding) for next_token_ids lookup. vLLM's
        # Step3p5AMultiTokenPredictor uses its own VocabParallelEmbedding;
        # we mirror that here. Falls back to backbone embed if MTP-side
        # weights were not loaded (e.g. older shard format).
        if hasattr(self.mtp, "embed_tokens") and self.mtp.embed_tokens.weight.shape[0] == self.model.embed_tokens.weight.shape[0]:
            embeds = self.mtp.embed_tokens(next_token_ids)
        else:
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
        # `fused` is already shared_head_norm'd by MTPModule. Now apply the
        # per-MTP-layer lm_head (shared_head_output) — NOT the backbone
        # lm_head, which would be wrong (per-MTP-layer heads have
        # non-identical bf16 weights, verified from Hikari07jp's shard).
        active_layer = self.mtp.layers[spec_step_idx % self.mtp.num_mtp_layers]
        return active_layer.shared_head_output(fused)

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
