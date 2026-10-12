# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Local config for cosmos3_edge checkpoints. Registered with AutoConfig only
# when the installed transformers does not provide the model type natively.

import os

import torch
from transformers.configuration_utils import PreTrainedConfig

_DTYPE_NAMES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _as_torch_dtype(value) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        key = value.replace("torch.", "").lower()
        if key not in _DTYPE_NAMES:
            raise ValueError(f"Unsupported dtype string: {value}")
        return _DTYPE_NAMES[key]
    return torch.bfloat16


class Cosmos3EdgeTextConfig(PreTrainedConfig):
    """Dense Edge Reasoner text config.

    The checkpoint stores 28 fused attention+MLP blocks. mRoPE is interleaved
    even when ``mrope_interleaved`` is omitted from ``rope_parameters``.
    """

    model_type = "cosmos3_edge_text"
    # ``mrope_section`` is not part of the default RoPE schema. Ignore it during
    # transformers' rope validation, then mark the layout as interleaved mRoPE.
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved", "type"}

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 2048,
        intermediate_size: int = 9216,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "relu2",
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-5,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        rope_theta: float = 1e8,
        rope_parameters: dict | None = None,
        dtype: str | torch.dtype | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        kwargs.pop("torch_dtype", None)
        kwargs.pop("model_type", None)
        resolved_dtype = _as_torch_dtype(dtype or kwargs.pop("dtype", None) or "bfloat16")
        rope_parameters = dict(rope_parameters or {})
        section = list(rope_parameters.get("mrope_section") or [24, 20, 20])
        theta = float(rope_parameters.get("rope_theta", rope_theta))
        rope_parameters = {
            "rope_type": rope_parameters.get("rope_type", "default"),
            "rope_theta": theta,
            "mrope_section": section,
        }
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            dtype=resolved_dtype,
            rope_parameters=rope_parameters,
            rope_theta=theta,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.rope_theta = theta
        self.dtype = resolved_dtype
        # ``rope_scaling`` is an alias of ``rope_parameters``. Keep the mRoPE
        # fields on that dict so Qwen3VLModelBase can set ``type`` in place.
        self.rope_parameters["mrope_section"] = section
        self.rope_parameters["mrope_interleaved"] = True
        self.rope_parameters["type"] = "mrope"
        self.rope_parameters["rope_theta"] = theta


class Cosmos3EdgeVisionConfig(PreTrainedConfig):
    """SigLIP2 vision tower used by Cosmos3-Edge."""

    model_type = "cosmos3_edge_vision"

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        patch_size: int = 16,
        num_patches: int = 256,
        hidden_act: str = "gelu_pytorch_tanh",
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        # Qwen vision attention reads ``num_heads``.
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout


class Cosmos3EdgeProjectorConfig(PreTrainedConfig):
    model_type = "cosmos3_edge_projector"

    def __init__(
        self,
        input_hidden_size: int = 1152,
        use_postshuffle_norm: bool = False,
        spatial_merge_size: int = 2,
        merger_intermediate_size: int = 11520,
        out_hidden_size: int = 2048,
        **kwargs,
    ):
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)
        self.input_hidden_size = input_hidden_size
        self.use_postshuffle_norm = use_postshuffle_norm
        self.spatial_merge_size = spatial_merge_size
        self.merger_intermediate_size = merger_intermediate_size
        self.out_hidden_size = out_hidden_size


class Cosmos3EdgeConfig(PreTrainedConfig):
    """Composite config for ``nvidia/Cosmos3-Edge`` Reasoner checkpoints."""

    model_type = "cosmos3_edge"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {
        "text_config": Cosmos3EdgeTextConfig,
        "vision_config": Cosmos3EdgeVisionConfig,
        "projector_config": Cosmos3EdgeProjectorConfig,
    }

    def __init__(
        self,
        text_config: dict | Cosmos3EdgeTextConfig | None = None,
        vision_config: dict | Cosmos3EdgeVisionConfig | None = None,
        projector_config: dict | Cosmos3EdgeProjectorConfig | None = None,
        image_token_id: int = 19,
        video_token_id: int = 18,
        vision_start_token_id: int = 20,
        vision_end_token_id: int = 21,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        kwargs.pop("model_type", None)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.text_config = (
            Cosmos3EdgeTextConfig(**text_config) if isinstance(text_config, dict) else text_config
        )
        if self.text_config is None:
            self.text_config = Cosmos3EdgeTextConfig()
        self.vision_config = (
            Cosmos3EdgeVisionConfig(**vision_config)
            if isinstance(vision_config, dict)
            else vision_config
        )
        if self.vision_config is None:
            self.vision_config = Cosmos3EdgeVisionConfig()
        self.projector_config = (
            Cosmos3EdgeProjectorConfig(**projector_config)
            if isinstance(projector_config, dict)
            else projector_config
        )
        if self.projector_config is None:
            self.projector_config = Cosmos3EdgeProjectorConfig()

        # Qwen3-VL processing and the patch merger read these from the vision
        # config. The Edge checkpoint stores the merge width on the projector.
        self.vision_config.spatial_merge_size = self.projector_config.spatial_merge_size
        self.vision_config.temporal_patch_size = 1
        self.vision_config.out_hidden_size = self.projector_config.out_hidden_size
        self.vision_config.num_heads = self.vision_config.num_attention_heads
        self.vision_config.num_key_value_heads = self.vision_config.num_attention_heads

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = tie_word_embeddings

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config = super().from_dict(config_dict, **kwargs)
        name_or_path = kwargs.get("_name_or_path") or kwargs.get("name_or_path")
        if name_or_path and (
            not getattr(config, "_name_or_path", None) or len(str(config._name_or_path)) < 2
        ):
            config._name_or_path = os.fspath(name_or_path)
        return config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not getattr(config, "_name_or_path", None) or len(str(config._name_or_path)) < 2:
            config._name_or_path = os.fspath(pretrained_model_name_or_path)
        return config
