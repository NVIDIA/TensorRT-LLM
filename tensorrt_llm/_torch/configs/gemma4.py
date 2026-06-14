# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class Gemma4TextConfig(PretrainedConfig):
    """Gemma4 text sub-config for PyTorch backend config loading."""

    model_type = "gemma4_text"

    def __init__(
        self,
        vocab_size: int = 262_144,
        hidden_size: int = 2816,
        intermediate_size: int = 2112,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 256,
        global_head_dim: int = 512,
        num_global_key_value_heads: int = 2,
        hidden_activation: str = "gelu_pytorch_tanh",
        max_position_embeddings: int = 131_072,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        attention_k_eq_v: bool = True,
        sliding_window: int = 1024,
        layer_types: Optional[list] = None,
        rope_parameters: Optional[dict] = None,
        final_logit_softcapping: Optional[float] = 30.0,
        hidden_size_per_layer_input: int = 0,
        num_kv_shared_layers: int = 0,
        use_double_wide_mlp: bool = False,
        enable_moe_block: bool = True,
        num_experts: Optional[int] = 128,
        top_k_experts: Optional[int] = 8,
        expert_intermediate_size: Optional[int] = 704,
        pad_token_id: Optional[int] = 0,
        eos_token_id=1,
        bos_token_id: Optional[int] = 2,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.global_head_dim = global_head_dim
        self.num_global_key_value_heads = num_global_key_value_heads
        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attention_k_eq_v = attention_k_eq_v
        self.sliding_window = sliding_window
        self.layer_types = layer_types or (["sliding_attention"] * num_hidden_layers)
        self.rope_parameters = rope_parameters
        self.final_logit_softcapping = final_logit_softcapping
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.num_kv_shared_layers = num_kv_shared_layers
        self.use_double_wide_mlp = use_double_wide_mlp
        self.enable_moe_block = enable_moe_block
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.expert_intermediate_size = expert_intermediate_size
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Gemma4Config(PretrainedConfig):
    """Top-level Gemma4 multimodal config for PyTorch backend config loading.

    This enables loading gemma4 checkpoints whose config.json has
    model_type='gemma4' without requiring a transformers version that
    natively supports gemma4.
    """

    model_type = "gemma4"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        audio_config=None,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        self.initializer_range = initializer_range
        if text_config is None:
            self.text_config = Gemma4TextConfig()
        elif isinstance(text_config, dict):
            self.text_config = Gemma4TextConfig(**text_config)
        else:
            self.text_config = text_config

        self.vision_config = vision_config
        self.audio_config = audio_config
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
