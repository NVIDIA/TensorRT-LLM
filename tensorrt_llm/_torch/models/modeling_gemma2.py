# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
from typing import Dict, Optional

import torch
from torch import nn
from transformers import Gemma2Config

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (
    AttentionMask,
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
    RopeParams,
)
from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..utils import inference_mode_unless_compiling
from .modeling_utils import DecoderModel, DecoderModelForCausalLM, register_auto_model


class Gemma2ScaledWordEmbedding(Embedding):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dtype: Optional[torch.dtype] = None,
        mapping: Optional[Mapping] = None,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        gather_output: bool = False,
    ):
        super().__init__(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=tensor_parallel_mode,
            gather_output=gather_output,
        )
        self.embed_scale = math.sqrt(hidden_size)

    @inference_mode_unless_compiling
    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale


class Gemma2Attention(Attention):
    """Gemma2 attention: plain RoPE (no QK norm), with per-layer attn softcap."""

    def __init__(
        self,
        model_config: ModelConfig[Gemma2Config],
        layer_idx: Optional[int] = None,
        is_sliding: bool = False,
    ):
        self.is_sliding = is_sliding
        config = model_config.pretrained_config

        # Gemma2 uses the same RoPE for all layers (no separate local/global theta).
        # RopeParams.from_config handles the rope_type="default" convention.
        rope_params = RopeParams.from_config(config)
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=rope_params,
        )
        q_scaling = math.sqrt(config.query_pre_attn_scalar) / math.sqrt(config.head_dim)
        self.attention_window_size = config.sliding_window if is_sliding else None

        attn_logit_softcap = getattr(config, "attn_logit_softcapping", None)

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=False,
            config=model_config,
            q_scaling=q_scaling,
            logits_soft_cap=attn_logit_softcap,
        )

    @inference_mode_unless_compiling
    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=attention_mask,
            attention_window_size=self.attention_window_size,
            **kwargs,
        )


def gelu_tanh(gate_x: torch.Tensor) -> torch.Tensor:
    """gelu_pytorch_tanh activation used by Gemma2."""
    if IS_FLASHINFER_AVAILABLE:
        return torch.ops.trtllm.flashinfer_gelu_tanh_and_mul(gate_x)
    gate, x = gate_x.chunk(2, dim=-1)
    return nn.functional.gelu(gate, approximate="tanh") * x


class Gemma2DecoderLayer(DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig[Gemma2Config],
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config
        layer_types = getattr(config, "layer_types", [])
        is_sliding = len(layer_types) > layer_idx and layer_types[layer_idx] == "sliding_attention"
        self.self_attn = Gemma2Attention(
            model_config,
            layer_idx=layer_idx,
            is_sliding=is_sliding,
        )
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=False,
            activation=gelu_tanh,
            dtype=config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
        )
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.pre_feedforward_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.post_feedforward_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    @inference_mode_unless_compiling
    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, lora_params=kwargs.get("lora_params", None))
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Gemma2TextModel(DecoderModel):
    def __init__(self, model_config: ModelConfig[Gemma2Config]):
        super().__init__(model_config)
        config = self.model_config
        self.hidden_size = config.pretrained_config.hidden_size

        self.embed_tokens = Gemma2ScaledWordEmbedding(
            config.pretrained_config.vocab_size,
            config.pretrained_config.hidden_size,
            dtype=config.pretrained_config.torch_dtype,
            mapping=config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList(
            [
                Gemma2DecoderLayer(model_config, layer_idx)
                for layer_idx in range(config.pretrained_config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden_size=config.pretrained_config.hidden_size,
            eps=config.pretrained_config.rms_norm_eps,
            dtype=config.pretrained_config.torch_dtype,
        )

    @inference_mode_unless_compiling
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds.to(self.dtype)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


@register_auto_model("Gemma2ForCausalLM")
class Gemma2ForCausalLM(DecoderModelForCausalLM[Gemma2TextModel, Gemma2Config]):
    def __init__(self, model_config: ModelConfig[Gemma2Config]):
        super().__init__(
            Gemma2TextModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )
        self._final_logit_softcap = getattr(
            model_config.pretrained_config, "final_logit_softcapping", None
        )

    @inference_mode_unless_compiling
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        output = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )
        if self._final_logit_softcap is not None:
            cap = float(self._final_logit_softcap)
            if cap <= 0:
                raise ValueError(f"final_logit_softcapping must be > 0, got {cap}.")
            logits = logits * (1.0 / cap)
            logits = torch.tanh(logits) * cap
        return logits

    def load_weights(self, weights: Dict, weight_mapper: Optional[BaseWeightMapper] = None):
        super().load_weights(weights, weight_mapper)
