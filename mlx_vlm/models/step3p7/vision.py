"""MLX port of stepfun-ai Step-3.7-Flash vision tower.

Source: configuration_step3p7.py + vision_encoder.py from
mlx-community/Step-3.7-Flash-4bit (which copies the upstream stepfun-ai
reference verbatim).

Architecture: custom ViT with 2D RoPE + LayerScale.
  conv1(14×14 patch) → ln_pre → +abs_posemb → 47×EncoderVisionBlock →
  ln_post(Identity since use_ln_post=False)
Then in Step3p7Model._process_image_features (outside this file):
  reshape → vit_downsampler1(stride=2) → vit_downsampler2(stride=2) →
  flatten → vit_large_projector(Linear 6144→4096)

Key delta vs molmo's MLX vision:
  - Fused QKV projection (in_proj_weight + in_proj_bias) matching PyTorch
    nn.MultiheadAttention convention
  - LayerScale (gamma) after attn AND after MLP
  - 2D RoPE injected into Q/K
  - QuickGELU activation in MLP
  - vit_downsampler1/2 Conv2d layers at the module level (called externally)

Weight name preservation (verified vs Step3-VL-10B and Step-3.7-Flash bf16):
  vision_model.
    conv1.weight                                      (out, in, kH, kW) → MLX needs (out, kH, kW, in)
    ln_pre.{weight, bias}
    positional_embedding                              (n_pos, width)
    transformer.resblocks.{N}.
      attn.in_proj_weight                             (3*width, width)
      attn.in_proj_bias                               (3*width,)
      attn.out_proj.{weight, bias}                    (width, width), (width,)
      ln_1.{weight, bias}
      ln_2.{weight, bias}
      ls_1.gamma                                      (width,)
      ls_2.gamma                                      (width,)
      mlp.c_fc.{weight, bias}                         (intermediate, width), (intermediate,)
      mlp.c_proj.{weight, bias}                       (width, intermediate), (width,)
    vit_downsampler1.{weight, bias}                   (out, in, kH, kW) — Conv2d needs transpose
    vit_downsampler2.{weight, bias}                   (out, in, kH, kW) — Conv2d needs transpose
"""
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


# --- helpers --------------------------------------------------------------

def _quick_gelu(x: mx.array) -> mx.array:
    """OpenCLIP-style QuickGELU: x * sigmoid(1.702 * x)."""
    return x * mx.sigmoid(1.702 * x)


def _act(name: str):
    if name == "quick_gelu":
        return _quick_gelu
    if name == "gelu":
        return nn.gelu
    if name == "silu":
        return nn.silu
    raise ValueError(f"Unsupported vision hidden_act: {name}")


# --- LayerScale -----------------------------------------------------------

class EncoderLayerScale(nn.Module):
    """Per-channel residual scaling — y = x * gamma (broadcasts over batch+seq)."""

    def __init__(self, dim: int, init_values: float):
        super().__init__()
        # PyTorch checkpoint stores this as ls_X.gamma → we mirror the name.
        self.gamma = mx.full((dim,), init_values)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


# --- 2D RoPE --------------------------------------------------------------

def _rotate_half(x: mx.array) -> mx.array:
    # x: (..., D) where D is even. Split last dim into pairs and rotate each.
    s = list(x.shape)
    x = x.reshape(*s[:-1], -1, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    out = mx.stack([-x2, x1], axis=-1)
    return out.reshape(*s)


def _apply_rotary(freqs: mx.array, t: mx.array) -> mx.array:
    """Apply 2D RoPE to t (..., L, D). freqs is (1, 1, L, D) covering full head_dim."""
    # AITRADER/Huihui-Step3-VL port pattern: simpler, no t_left/t_right split
    # since rot_dim always equals head_dim.
    dtype = t.dtype
    t = (t * mx.cos(freqs)) + (_rotate_half(t) * mx.sin(freqs))
    return t.astype(dtype)


class EncoderRope2D(nn.Module):
    """Cacheable 2D rotary positional embedding.

    For each token at grid (h, w) we concatenate W-rotation freqs and
    H-rotation freqs over the head dimension. Cached eagerly at the
    default grid (52×52 for 728²/14) at construction.
    """

    def __init__(
        self,
        dim: int,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        theta: float = 10000.0,
        theta_rescale_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_grid_height = max_grid_height
        self.max_grid_width = max_grid_width
        self.use_cls_token = use_cls_token
        # Match PyTorch rescaling
        self.theta = theta * (theta_rescale_factor ** (dim / max(dim - 2, 1)))
        # Bypass nn.Module's __setattr__ tracking — freqs_cache is a
        # deterministic buffer (not a learned parameter), and the
        # checkpoint correctly omits it. If we stored it as a regular
        # attribute it would show up in mlx's parameters() iteration and
        # the strict loader would complain "missing 47 parameters".
        object.__setattr__(self, "_freqs_cache", self._compute_2d_freqs())

    def _compute_inv_freq(self, base: float, dim_half: int) -> mx.array:
        # Match PyTorch: 1 / base ** (arange(0, dim_half, 2) / dim_half)
        idx = mx.arange(0, dim_half, 2).astype(mx.float32)
        return 1.0 / (base ** (idx / dim_half))

    def _compute_freqs(self, t: mx.array, inv_freq: mx.array) -> mx.array:
        # einsum("..., f -> ... f")
        freqs = t.astype(inv_freq.dtype)[..., None] * inv_freq[None, :]
        # repeat_interleave(2, dim=-1)
        freqs = mx.repeat(freqs[..., None], 2, axis=-1)
        return freqs.reshape(*freqs.shape[:-2], -1)

    def _compute_2d_freqs(self) -> mx.array:
        H, W = self.max_grid_height, self.max_grid_width
        grid_h = mx.arange(H).astype(mx.float32)
        grid_w = mx.arange(W).astype(mx.float32)
        if self.use_cls_token:
            grid_h = grid_h + 1
            grid_w = grid_w + 1
        inv_freq = self._compute_inv_freq(self.theta, self.dim // 2)
        fh = self._compute_freqs(grid_h, inv_freq)        # (H, dim/2)
        fw = self._compute_freqs(grid_w, inv_freq)        # (W, dim/2)
        # Broadcast over (H, W)
        fh = mx.broadcast_to(fh[:, None, :], (H, W, fh.shape[-1]))
        fw = mx.broadcast_to(fw[None, :, :], (H, W, fw.shape[-1]))
        # Concat W first, H second along last dim — match PyTorch order
        freqs = mx.concatenate([fw, fh], axis=-1).reshape(H * W, -1)
        if self.use_cls_token:
            freqs = mx.concatenate([mx.zeros((1, freqs.shape[-1])), freqs], axis=0)
        # (1, 1, L, D) so we can broadcast with (B, num_heads, L, head_dim)
        return freqs[None, None, ...]

    def __call__(self, q: mx.array, k: mx.array, grid_hw: Tuple[int, int]):
        H, W = grid_hw
        cache = self._freqs_cache
        if H == self.max_grid_height and W == self.max_grid_width:
            freqs = cache
        else:
            rows = mx.arange(H)[:, None]
            cols = mx.arange(W)[None, :]
            positions = (rows * self.max_grid_width + cols).reshape(-1).astype(mx.int32)
            if self.use_cls_token:
                positions = mx.concatenate([mx.zeros((1,), dtype=mx.int32), positions + 1], axis=0)
            freqs = cache[:, :, positions, :]
        q = _apply_rotary(freqs, q)
        k = _apply_rotary(freqs, k)
        return q, k


# --- MLP ------------------------------------------------------------------

class EncoderMLP(nn.Module):
    """c_fc(width → intermediate) → quick_gelu → c_proj(intermediate → width)."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.c_proj = nn.Linear(intermediate_size, hidden_size, bias=True)
        self._act_fn = _act(hidden_act)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(self._act_fn(self.c_fc(x)))


# --- Attention with fused QKV + 2D RoPE -----------------------------------

class EncoderVisionAttention(nn.Module):
    """Fused-QKV self-attention with optional 2D RoPE on Q/K.

    Weights stored as `in_proj_weight` (3*D, D) + `in_proj_bias` (3*D,) +
    `out_proj` (Linear). Mirrors PyTorch nn.MultiheadAttention checkpoints.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        use_rope2d: bool = True,
        rope_theta: float = 10000.0,
        rope_theta_rescale_factor: float = 1.0,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.hidden_size = hidden_size

        # Fused QKV — match PyTorch nn.MultiheadAttention layout.
        self.in_proj_weight = mx.zeros((hidden_size * 3, hidden_size))
        self.in_proj_bias = mx.zeros((hidden_size * 3,))
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.use_rope2d = use_rope2d
        self.rope: Optional[EncoderRope2D] = None
        if use_rope2d:
            self.rope = EncoderRope2D(
                dim=self.head_dim,
                max_grid_height=max_grid_height,
                max_grid_width=max_grid_width,
                use_cls_token=use_cls_token,
                theta=rope_theta,
                theta_rescale_factor=rope_theta_rescale_factor,
            )

    def __call__(self, x: mx.array, grid_hw: Tuple[int, int]) -> mx.array:
        B, L, _ = x.shape
        # Equivalent to F.linear: y = x @ W^T + b
        qkv = x @ self.in_proj_weight.T + self.in_proj_bias
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.rope is not None:
            q, k = self.rope(q, k, grid_hw=grid_hw)

        # mx.fast.scaled_dot_product_attention restored 2026-05-29 after
        # PyTorch parity test confirmed mathematical correctness — the
        # manual matmul + softmax pattern we used in Phase 2 was for
        # parity debugging only. mx.fast.sdpa is materially faster on
        # 47-layer × 2704-token vision forward.
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = attn.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return self.out_proj(out)


# --- Block + Transformer --------------------------------------------------

class EncoderVisionBlock(nn.Module):
    """ViT block: LN → attn → LayerScale → +residual → LN → MLP → LayerScale → +residual."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        hidden_act: str,
        layer_norm_eps: float,
        ls_init_value: float,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        use_rope2d: bool = True,
        rope_theta: float = 10000.0,
        rope_theta_rescale_factor: float = 1.0,
    ):
        super().__init__()
        self.attn = EncoderVisionAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_grid_height=max_grid_height,
            max_grid_width=max_grid_width,
            use_cls_token=use_cls_token,
            use_rope2d=use_rope2d,
            rope_theta=rope_theta,
            rope_theta_rescale_factor=rope_theta_rescale_factor,
        )
        self.ln_1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        intermediate = int(hidden_size * mlp_ratio)
        self.mlp = EncoderMLP(hidden_size, intermediate, hidden_act)
        self.ls_1 = EncoderLayerScale(hidden_size, ls_init_value)
        self.ls_2 = EncoderLayerScale(hidden_size, ls_init_value)

    def __call__(self, x: mx.array, grid_hw: Tuple[int, int]) -> mx.array:
        x = x + self.ls_1(self.attn(self.ln_1(x), grid_hw=grid_hw))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class EncoderVisionTransformer(nn.Module):
    """Stack of N EncoderVisionBlock — checkpoint keys are `resblocks.N.*`."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        hidden_act: str,
        layer_norm_eps: float,
        ls_init_value: float,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        use_rope2d: bool = True,
        rope_theta: float = 10000.0,
        rope_theta_rescale_factor: float = 1.0,
    ):
        super().__init__()
        self.resblocks = [
            EncoderVisionBlock(
                hidden_size=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                hidden_act=hidden_act,
                layer_norm_eps=layer_norm_eps,
                ls_init_value=ls_init_value,
                max_grid_height=max_grid_height,
                max_grid_width=max_grid_width,
                use_cls_token=use_cls_token,
                use_rope2d=use_rope2d,
                rope_theta=rope_theta,
                rope_theta_rescale_factor=rope_theta_rescale_factor,
            )
            for _ in range(depth)
        ]

    def __call__(self, x: mx.array, grid_hw: Tuple[int, int]) -> mx.array:
        for block in self.resblocks:
            x = block(x, grid_hw=grid_hw)
        return x


# --- Top-level vision tower ----------------------------------------------

class StepRoboticsVisionEncoder(nn.Module):
    """Full vision tower as defined in vision_encoder.py.

    Forward returns post-transformer hidden states (B, num_patches, width).
    The vit_downsampler1/2 + projector are called externally by
    Step3p7Model._process_image_features.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.width
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.use_cls_token = config.use_cls_token
        self.use_rope2d = config.use_rope2d
        self.use_abs_posemb = config.use_abs_posemb
        self.use_ln_pre = config.use_ln_pre
        self.use_ln_post = config.use_ln_post

        grid_size = self.image_size // self.patch_size  # 728/14 = 52
        self.base_grid = (grid_size, grid_size)

        # Patch embedding — Conv2d in MLX uses NHWC. Weights loaded as
        # (out, kH, kW, in) after sanitize() transposes the PyTorch
        # (out, in, kH, kW) checkpoint format.
        self.conv1 = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        if self.use_ln_pre:
            self.ln_pre = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        if self.use_ln_post:
            self.ln_post = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

        if self.use_cls_token:
            # Sized to match PyTorch checkpoint
            self.class_embedding = mx.zeros((self.hidden_size,))

        if self.use_abs_posemb:
            n_pos = (1 if self.use_cls_token else 0) + grid_size * grid_size
            self.positional_embedding = mx.zeros((n_pos, self.hidden_size))

        self.transformer = EncoderVisionTransformer(
            embed_dim=self.hidden_size,
            depth=config.layers,
            num_heads=config.heads,
            mlp_ratio=config.mlp_ratio,
            hidden_act=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
            ls_init_value=config.ls_init_value,
            max_grid_height=grid_size,
            max_grid_width=grid_size,
            use_cls_token=self.use_cls_token,
            use_rope2d=self.use_rope2d,
            rope_theta=config.rope_theta,
            rope_theta_rescale_factor=config.rope_theta_rescale_factor,
        )

        # Spatial downsamplers — called externally by Step3p7Model.
        # Conv2d in MLX expects NHWC; weights (out, kH, kW, in).
        self.vit_downsampler1 = nn.Conv2d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
        )
        self.vit_downsampler2 = nn.Conv2d(
            in_channels=self.hidden_size * 2,
            out_channels=self.hidden_size * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
        )

    # --- positional embedding helpers ------------------------------------

    def _sample_abs_posemb(self, grid_h: int, grid_w: int) -> mx.array:
        """Return (1, H*W [+cls], width) positional embeddings, interpolated if needed."""
        base = self.image_size // self.patch_size
        if grid_h == base and grid_w == base:
            return self.positional_embedding[None, ...]

        pos = self.positional_embedding
        cls = None
        if self.use_cls_token:
            cls = pos[:1]
            pos = pos[1:]

        # MLX nn.Upsample with 2-tuple scale_factor applies to dims 1+2 of the
        # input — so we use NHWC layout natively (no NCHW transpose) and the
        # scale operates on (H, W). PyTorch reference uses F.interpolate on
        # NCHW with mode="bilinear" align_corners=False; MLX mode="linear" on
        # 4D input is N-D linear (= bilinear for 2D spatial), align_corners
        # is honoured.
        pos2d = pos.reshape(1, base, base, -1)  # NHWC: (1, base, base, hidden)
        scale = (grid_h / base, grid_w / base)
        upsample = nn.Upsample(scale_factor=scale, mode="linear", align_corners=False)
        pos2d = upsample(pos2d)  # (1, grid_h, grid_w, hidden)
        pos2d = pos2d.reshape(-1, self.hidden_size)

        if cls is not None:
            pos2d = mx.concatenate([cls, pos2d], axis=0)
        return pos2d[None, ...]

    # --- forward ---------------------------------------------------------

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """pixel_values: (B, C, H, W). Returns (B, num_patches, width)."""
        B, C, H, W = pixel_values.shape
        grid_h, grid_w = H // self.patch_size, W // self.patch_size

        # MLX Conv2d expects NHWC. PyTorch convention is NCHW.
        x = pixel_values.transpose(0, 2, 3, 1)              # (B, H, W, C)
        x = self.conv1(x)                                    # (B, Gh, Gw, D)
        x = x.reshape(B, grid_h * grid_w, self.hidden_size)  # (B, N, D)

        if self.use_cls_token:
            cls = mx.broadcast_to(self.class_embedding[None, None, :], (B, 1, self.hidden_size))
            x = mx.concatenate([cls, x], axis=1)

        if self.use_abs_posemb:
            x = x + self._sample_abs_posemb(grid_h, grid_w)

        if self.use_ln_pre:
            x = self.ln_pre(x)

        x = self.transformer(x, grid_hw=(grid_h, grid_w))

        if self.use_ln_post:
            x = self.ln_post(x)

        if self.use_cls_token:
            x = x[:, 1:, :]

        return x
