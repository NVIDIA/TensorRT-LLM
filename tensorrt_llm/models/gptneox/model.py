# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tensorrt as trt

from ..._common import default_net
from ..._utils import pad_vocab_size
from ...functional import PositionEmbeddingType, Tensor
from ...layers import (MLP, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, KeyValueCacheParams, LayerNorm)
from ...module import Module
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              PretrainedConfig)


class GPTNeoXDecoderLayer(Module):

    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        hidden_size = config.hidden_size
        dtype = config.dtype
        tp_group = config.mapping.tp_group
        tp_size = config.mapping.tp_size

        self.input_layernorm = LayerNorm(normalized_shape=hidden_size,
                                         dtype=dtype)

        self.post_attention_layernorm = LayerNorm(normalized_shape=hidden_size,
                                                  dtype=dtype)

        self.attention = Attention(
            hidden_size=hidden_size,
            num_attention_heads=config.num_attention_heads,
            rotary_embedding_percentage=config.rotary_pct,
            rotary_embedding_base=config.rotary_emb_base,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            max_position_embeddings=config.max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=True,
            tp_group=tp_group,
            tp_size=tp_size)

        self.mlp = MLP(hidden_size=hidden_size,
                       ffn_hidden_size=hidden_size * 4,
                       hidden_act=config.hidden_act,
                       dtype=dtype,
                       tp_group=tp_group,
                       tp_size=tp_size)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):
        if not default_net(
        ).plugin_config.layernorm_plugin and trt.__version__[:3] == '8.6':
            raise AssertionError(
                "You need to enable the LayerNorm plugin for GPT-NeoX with TensorRT 8.6. Please set plugin_config.layernorm_plugin"
            )
        residual = hidden_states

        input_layernorm_output = self.input_layernorm(hidden_states)
        post_attention_layernorm_output = self.post_attention_layernorm(
            hidden_states)

        attention_output = self.attention(input_layernorm_output,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params,
                                          norm_before_bmm1=True)

        if use_cache:
            attention_output, presents = attention_output

        feed_forward_hidden_states = self.mlp(post_attention_layernorm_output)
        hidden_states = attention_output + feed_forward_hidden_states + residual
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class GPTNeoXModel(Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        mapping = config.mapping
        self.vocab_embedding = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
            tp_size=mapping.tp_size if config.use_parallel_embedding else 1,
            tp_group=mapping.tp_group
            if config.use_parallel_embedding else None,
            sharding_dim=config.embedding_sharding_dim,
            tp_rank=mapping.tp_rank)

        self.layers = DecoderLayerList(GPTNeoXDecoderLayer, config)

        self.ln_f = LayerNorm(normalized_shape=config.hidden_size,
                              dtype=config.dtype)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None):
        hidden_states = self.vocab_embedding(input_ids)

        kv_cache_params.fill_none_tensor_list(len(self.layers))

        if use_cache:
            presents = []

        for layer, past, max_attention_window_size in zip(
                self.layers, kv_cache_params.past_key_value,
                kv_cache_params.host_max_attention_window_sizes):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    host_max_attention_window_sizes=max_attention_window_size,
                    host_sink_token_length=kv_cache_params.
                    host_sink_token_length,
                    cache_indirection=kv_cache_params.cache_indirection),
                attention_params=attention_params)

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class GPTNeoXForCausalLM(DecoderModelForCausalLM):

    def __init__(self, config: PretrainedConfig):
        transformer = GPTNeoXModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)

        lm_head = ColumnLinear(config.hidden_size,
                               vocab_size_padded,
                               bias=False,
                               dtype=config.dtype,
                               tp_group=config.mapping.tp_group,
                               tp_size=config.mapping.tp_size,
                               gather_output=True)
        super().__init__(config, transformer, lm_head)
