"""Config dataclasses for Step-3.7-Flash multimodal VLM adapter.

Mirrors stepfun-ai's configuration_step3p7.py:
- StepRoboticsVisionEncoderConfig → VisionConfig
- Step3p7TextConfig → TextConfig (reuses step3p5/qwen-style fields)
- Step3p7Config → ModelConfig (composite + projector/image-token wiring)
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    """Vision tower config = perception_encoder ViT (custom 47-layer 1536-width).

    Defaults match upstream StepRoboticsVisionEncoderConfig.
    """
    model_type: str = "perception_encoder"
    width: int = 1536
    layers: int = 47
    heads: int = 16
    num_channels: int = 3
    image_size: int = 728
    patch_size: int = 14
    mlp_ratio: float = 8960.0 / 1536.0  # → intermediate_size = 8960
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    use_cls_token: bool = False
    use_ln_pre: bool = True
    use_ln_post: bool = False
    use_abs_posemb: bool = True
    use_rope2d: bool = True
    ls_init_value: float = 0.1
    # RoPE-2D kwargs (defaults from EncoderRope2D)
    rope_theta: float = 10000.0
    rope_max_freq: int = 10
    rope_num_freqs: int = 1
    rope_theta_rescale_factor: float = 1.0


@dataclass
class TextConfig(BaseModelConfig):
    """Step 3.7 text backbone = identical to step3p5 (45-layer Qwen3.6-style MoE).

    Field names match the upstream Step3p7TextConfig + step3p5 ModelArgs.
    """
    model_type: str = "step3p5"
    hidden_size: int = 4096
    intermediate_size: int = 11264
    num_hidden_layers: int = 45
    vocab_size: int = 128896
    num_attention_heads: int = 64
    num_attention_groups: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-5
    rope_theta: Any = 10000.0  # may be float OR list[float] per-layer
    rope_scaling: Optional[Dict] = None
    max_position_embeddings: int = 262144
    sliding_window: int = 512
    layer_types: Optional[List[str]] = None
    yarn_only_types: Optional[List[str]] = None
    partial_rotary_factors: Optional[List[float]] = None
    attention_other_setting: Optional[Dict] = None
    use_head_wise_attn_gate: bool = True
    moe_num_experts: int = 288
    moe_top_k: int = 8
    moe_intermediate_size: int = 1280
    share_expert_dim: int = 1280
    moe_layers_enum: Optional[str] = None
    moe_router_scaling_factor: float = 3.0
    norm_expert_weight: bool = True
    swiglu_limits: Optional[List[float]] = None
    swiglu_limits_shared: Optional[List[float]] = None
    tie_word_embeddings: bool = False
    # MTP / nextn (multi-token prediction) — Step 3.7 declares 3 in upstream
    # BF16. NVFP4 / our 4-bit drop them; Hikari07jp re-extracted shard adds them
    # back. LanguageModel.__init__ builds MTPModule iff this is > 0.
    num_nextn_predict_layers: int = 0


@dataclass
class ModelConfig(BaseModelConfig):
    """Top-level Step3p7 multimodal config.

    Composes vision_config + text_config + projector + image-token wiring.
    """
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "step3p7"
    understand_projector_stride: int = 2
    projector_bias: bool = False
    image_token_id: int = 128001  # observed value in mlx-community/Step-3.7-Flash-4bit
    image_token_len: int = 169
    patch_token_len: int = 81
    vision_select_layer: int = -1
    eos_token_id: Optional[List[int]] = None
    tie_word_embeddings: bool = False
