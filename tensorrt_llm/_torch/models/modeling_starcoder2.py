# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
from torch import nn
from transformers import Starcoder2Config

from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.attention_backend.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import (
    DecoderModel,
    DecoderModelForCausalLM,
    _load_weights_impl,
    register_auto_model,
)
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.modules.mlp import MLP
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm.functional import PositionEmbeddingType


class Starcoder2Attention(Attention):
    """
    StarCoder2 Attention with Grouped Query Attention and Sliding Window support.
    """

    def __init__(
        self,
        model_config: ModelConfig[Starcoder2Config],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.use_bias,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            ),
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )

        # Configure sliding window attention (4096 tokens)
        self.attention_window_size = getattr(config, "sliding_window", 4096)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """
        Overrides parent to pass attention_window_size parameter.
        """
        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_window_size=self.attention_window_size,
            **kwargs,
        )


class Starcoder2DecoderLayer(DecoderLayer):
    """
    StarCoder2 Decoder Layer.

    Architecture:
        - Layer normalization before attention (with bias)
        - Self-attention with GQA and sliding window
        - Layer normalization before MLP (with bias)
        - MLP with GELU activation
    """

    def __init__(
        self,
        model_config: ModelConfig[Starcoder2Config],
        layer_idx: int,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx

        self.self_attn = Starcoder2Attention(
            model_config,
            layer_idx=layer_idx,
        )

        if config.mlp_type == "default":
            self.mlp = MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                bias=config.use_bias,
                activation=nn.GELU(),
                dtype=config.torch_dtype,
                config=model_config,
            )
        else:
            raise ValueError(
                f"Unsupported mlp_type: {config.mlp_type}. Only default (linear) MLP is supported."
            )

        norm_eps = getattr(config, "norm_epsilon", 1e-5)
        self.input_layernorm = LayerNorm(
            hidden_size=config.hidden_size,
            eps=norm_eps,
            dtype=config.torch_dtype,
            has_bias=True,  # StarCoder2 uses bias in layer norm
        )

        self.post_attention_layernorm = LayerNorm(
            hidden_size=config.hidden_size,
            eps=norm_eps,
            dtype=config.torch_dtype,
            has_bias=True,  # StarCoder2 uses bias in layer norm
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        # Fully Connected (MLP)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)

        return hidden_states, residual


class Starcoder2Model(DecoderModel):
    """
    StarCoder2 Transformer Model.
    """

    def __init__(self, model_config: ModelConfig[Starcoder2Config]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )

        self.layers = nn.ModuleList(
            [
                Starcoder2DecoderLayer(
                    model_config,
                    layer_idx,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # Use norm_epsilon (Starcoder2Config attribute name)
        norm_eps = getattr(config, "norm_epsilon", 1e-5)
        self.norm = LayerNorm(
            hidden_size=config.hidden_size,
            eps=norm_eps,
            dtype=config.torch_dtype,
            has_bias=True,  # StarCoder2 uses bias in layer norm
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params=None,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                lora_params=lora_params,
            )

        # Use LayerNorm's built-in residual connection support
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("Starcoder2ForCausalLM")
class Starcoder2ForCausalLM(DecoderModelForCausalLM[Starcoder2Model, Starcoder2Config]):
    def __init__(
        self,
        model_config: ModelConfig[Starcoder2Config],
    ):
        # Ensure torch_dtype is set on pretrained_config (StarCoder2 uses bfloat16).
        # For the 15B FP32 checkpoint, we cast it to bfloat16 for consistency.
        torch_dtype_to_check = model_config.pretrained_config.torch_dtype
        if torch_dtype_to_check is None or torch_dtype_to_check == torch.float32:
            model_config.pretrained_config.torch_dtype = torch.bfloat16

        super().__init__(
            Starcoder2Model(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    def load_weights(self, weights, weight_mapper=None, skip_modules=None):
        """
        Load weights with custom mapping for StarCoder2.

        StarCoder2 uses GPT-2 style MLP naming (c_fc, c_proj)
        while our MLP module expects (up_proj, down_proj).
        """
        if skip_modules is None:
            skip_modules = []

        # Map HuggingFace StarCoder2 weight names to TensorRT-LLM names
        params_map = {
            r"(.*?)\.mlp\.c_fc\.(.*)": r"\1.mlp.up_proj.\2",
            r"(.*?)\.mlp\.c_proj\.(.*)": r"\1.mlp.down_proj.\2",
        }
        preload_weight_modules = getattr(self, "preload_weight_modules", None)
        _load_weights_impl(
            self,
            weights,
            skip_modules,
            params_map=params_map,
            preload_weight_modules=preload_weight_modules,
        )
