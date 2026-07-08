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
from typing import Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from .modeling_utils import DecoderModel, DecoderModelForCausalLM, register_auto_model


class ChatGLMAttention(Attention):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        # ChatGLM rotates only the first half of each head with GPT-J interleaving.
        rotary_dim = int(head_dim * getattr(config, "partial_rotary_factor", 0.5))
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gptj,
            rope=RopeParams(
                dim=rotary_dim,
                theta=getattr(config, "rope_theta", 10000.0),
                max_positions=config.max_position_embeddings,
            ),
            is_neox=False,
        )

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=True,
            pos_embd_params=pos_embd_params,
            rope_fusion=False,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=False,
            config=model_config,
        )


class ChatGLMDecoderLayer(DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config

        self.self_attn = ChatGLMAttention(model_config, layer_idx=layer_idx)

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=False,
            dtype=config.torch_dtype,
            config=model_config,
        )
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, **kwargs)

        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)
        return hidden_states, residual


class ChatGLMModel(DecoderModel):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
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
                ChatGLMDecoderLayer(model_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

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
                **kwargs,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("ChatGLMModel")
class ChatGLMForCausalLM(DecoderModelForCausalLM[ChatGLMModel, PretrainedConfig]):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(
            ChatGLMModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    def load_weights(self, weights: dict[str, torch.Tensor], weight_mapper=None):
        config = self.config
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        q_dim = config.num_attention_heads * head_dim
        kv_dim = config.num_key_value_heads * head_dim
        ffn = config.intermediate_size

        converted: dict[str, torch.Tensor] = {}

        def rename_layer(i: int) -> None:
            src = f"transformer.encoder.layers.{i}."
            dst = f"model.layers.{i}."
            # Split ChatGLM's fused QKV so the shared loader can shard it.
            qkv_w = weights[src + "self_attention.query_key_value.weight"]
            qkv_b = weights[src + "self_attention.query_key_value.bias"]
            q_w, k_w, v_w = qkv_w.split([q_dim, kv_dim, kv_dim], dim=0)
            q_b, k_b, v_b = qkv_b.split([q_dim, kv_dim, kv_dim], dim=0)
            converted[dst + "self_attn.q_proj.weight"] = q_w
            converted[dst + "self_attn.q_proj.bias"] = q_b
            converted[dst + "self_attn.k_proj.weight"] = k_w
            converted[dst + "self_attn.k_proj.bias"] = k_b
            converted[dst + "self_attn.v_proj.weight"] = v_w
            converted[dst + "self_attn.v_proj.bias"] = v_b
            converted[dst + "self_attn.o_proj.weight"] = weights[
                src + "self_attention.dense.weight"
            ]
            gate_up = weights[src + "mlp.dense_h_to_4h.weight"]
            gate_w, up_w = gate_up.split([ffn, ffn], dim=0)
            converted[dst + "mlp.gate_proj.weight"] = gate_w
            converted[dst + "mlp.up_proj.weight"] = up_w
            converted[dst + "mlp.down_proj.weight"] = weights[src + "mlp.dense_4h_to_h.weight"]
            converted[dst + "input_layernorm.weight"] = weights[src + "input_layernorm.weight"]
            converted[dst + "post_attention_layernorm.weight"] = weights[
                src + "post_attention_layernorm.weight"
            ]

        for i in range(config.num_hidden_layers):
            rename_layer(i)

        converted["model.embed_tokens.weight"] = weights[
            "transformer.embedding.word_embeddings.weight"
        ]
        converted["model.norm.weight"] = weights["transformer.encoder.final_layernorm.weight"]
        converted["lm_head.weight"] = weights["transformer.output_layer.weight"]

        super().load_weights(converted)
