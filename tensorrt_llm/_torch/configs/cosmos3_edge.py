# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Local config for cosmos3_edge checkpoints. Registered with AutoConfig only
# when the installed transformers does not provide the model type natively.

import torch
from transformers.configuration_utils import PreTrainedConfig

from .cosmos3 import _with_checkpoint_name

_DTYPE_NAMES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _as_torch_dtype(value) -> torch.dtype:
    """Resolve a checkpoint dtype. Missing or unknown values raise."""
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        key = value.replace("torch.", "").lower()
        if key not in _DTYPE_NAMES:
            raise ValueError(f"Unsupported dtype string: {value}")
        return _DTYPE_NAMES[key]
    raise TypeError(f"Unsupported dtype value: {value!r}")


_SUPPORTED_ROPE_TYPES = {"default", "linear", "dynamic", "longrope", "llama3", "yarn", "mrope"}


def _edge_rope_parameters(rope_parameters, rope_theta: float) -> dict:
    """Keep checkpoint RoPE keys and add the interleaved mRoPE fields."""
    incoming = dict(rope_parameters or {})
    rope_type = incoming.get("rope_type", "default")
    if rope_type not in _SUPPORTED_ROPE_TYPES:
        raise ValueError(
            f"Cosmos3-Edge text RoPE type {rope_type!r} is not supported. "
            f"Supported types: {sorted(_SUPPORTED_ROPE_TYPES)}."
        )
    position_type = incoming.get("type")
    if position_type not in (None, "mrope"):
        raise ValueError(
            "Cosmos3-Edge text position type "
            f"{position_type!r} is not supported. This Reasoner uses interleaved mRoPE."
        )
    if incoming.get("mrope_interleaved", True) is not True:
        raise ValueError("Cosmos3-Edge Reasoner requires mrope_interleaved=True.")
    incoming["rope_type"] = rope_type
    incoming["rope_theta"] = float(incoming.get("rope_theta", rope_theta))
    incoming["mrope_section"] = list(incoming.get("mrope_section") or [24, 20, 20])
    incoming["mrope_interleaved"] = True
    incoming["type"] = "mrope"
    return incoming


def _use_or_match(config, name: str, expected):
    """Keep a vision field when it matches, and fill it only when the checkpoint omits it."""
    current = getattr(config, name, None)
    if current is None:
        setattr(config, name, expected)
        return
    if int(current) != int(expected):
        raise ValueError(
            f"Cosmos3-Edge vision_config.{name}={current} does not match the projector value {expected}."
        )


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
        # Checkpoints may still say ``torch_dtype`` instead of ``dtype``.
        legacy_torch_dtype = kwargs.pop("torch_dtype", None)
        kwargs.pop("model_type", None)
        resolved_dtype = _as_torch_dtype(
            dtype or kwargs.pop("dtype", None) or legacy_torch_dtype or "bfloat16"
        )
        rope_parameters = _edge_rope_parameters(rope_parameters, rope_theta)
        theta = float(rope_parameters["rope_theta"])
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            dtype=resolved_dtype,
            max_position_embeddings=max_position_embeddings,
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
        # Validation can drop keys that are not in the default RoPE schema.
        # Put the checkpoint dict back, including scaling fields such as ``factor``.
        for key, value in rope_parameters.items():
            self.rope_parameters[key] = value


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
        spatial_merge_size: int | None = None,
        temporal_patch_size: int | None = None,
        out_hidden_size: int | None = None,
        dtype: str | torch.dtype | None = None,
        **kwargs,
    ):
        kwargs.pop("model_type", None)
        legacy_torch_dtype = kwargs.pop("torch_dtype", None)
        raw_dtype = dtype or kwargs.pop("dtype", None) or legacy_torch_dtype
        resolved_dtype = None if raw_dtype is None else _as_torch_dtype(raw_dtype)
        if resolved_dtype is not None:
            kwargs["dtype"] = resolved_dtype
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
        if spatial_merge_size is not None:
            self.spatial_merge_size = spatial_merge_size
        if temporal_patch_size is not None:
            self.temporal_patch_size = temporal_patch_size
        if out_hidden_size is not None:
            self.out_hidden_size = out_hidden_size
        if resolved_dtype is not None:
            self.dtype = resolved_dtype
            self.torch_dtype = resolved_dtype


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


def _build_edge_sub_config(config_cls, value, attn_implementation):
    """Build one Edge sub-config and keep the parent attention implementation."""
    if isinstance(value, dict):
        value = dict(value)
        if attn_implementation is not None and "attn_implementation" not in value:
            value["attn_implementation"] = attn_implementation
        return config_cls(**value)
    if value is None:
        extra = {}
        if attn_implementation is not None:
            extra["attn_implementation"] = attn_implementation
        return config_cls(**extra)
    if attn_implementation is not None:
        value._attn_implementation = attn_implementation
    return value


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
        # Capture before ``PreTrainedConfig`` consumes the kwarg. Sub-configs are
        # built after that call, so they do not see the propagation that happens
        # during base initialization.
        attn_implementation = kwargs.get("attn_implementation")
        kwargs.pop("model_type", None)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        if attn_implementation is None:
            attn_implementation = getattr(self, "_attn_implementation", None)
        self.text_config = _build_edge_sub_config(
            Cosmos3EdgeTextConfig, text_config, attn_implementation
        )
        self.vision_config = _build_edge_sub_config(
            Cosmos3EdgeVisionConfig, vision_config, attn_implementation
        )
        self.projector_config = _build_edge_sub_config(
            Cosmos3EdgeProjectorConfig, projector_config, attn_implementation
        )

        # Processing and the patch merger read these from the vision config.
        # The checkpoint stores the merge width on both sub-configs; keep a
        # vision value that matches, and fill one only when it is absent.
        _use_or_match(
            self.vision_config, "spatial_merge_size", self.projector_config.spatial_merge_size
        )
        _use_or_match(self.vision_config, "out_hidden_size", self.projector_config.out_hidden_size)
        temporal_patch_size = getattr(self.vision_config, "temporal_patch_size", None)
        if temporal_patch_size is None:
            self.vision_config.temporal_patch_size = 1
        elif int(temporal_patch_size) != 1:
            raise ValueError(
                f"Cosmos3-Edge vision temporal_patch_size must be 1, got {temporal_patch_size}."
            )
        vision_dtype = getattr(self.vision_config, "dtype", None)
        if not isinstance(vision_dtype, torch.dtype):
            vision_dtype = self.text_config.dtype
        self.vision_config.dtype = vision_dtype
        self.vision_config.torch_dtype = vision_dtype
        # Qwen vision attention reads this from the parent before the text config.
        self.max_position_embeddings = self.text_config.max_position_embeddings

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = tie_word_embeddings

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config = super().from_dict(config_dict, **kwargs)
        name_or_path = kwargs.get("_name_or_path") or kwargs.get("name_or_path")
        return _with_checkpoint_name(
            config, name_or_path, bool(kwargs.get("return_unused_kwargs", False))
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        return _with_checkpoint_name(
            config,
            pretrained_model_name_or_path,
            bool(kwargs.get("return_unused_kwargs", False)),
        )
