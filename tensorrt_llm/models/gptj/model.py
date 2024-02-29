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

from ..._utils import pad_vocab_size
from ...functional import PositionEmbeddingType, Tensor, allreduce
from ...layers import (MLP, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, KeyValueCacheParams, LayerNorm)
from ...module import Module
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              PretrainedConfig)


class GPTJDecoderLayer(Module):

    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        rotary_dim = config.rotary_dim
        dtype = config.dtype
        tp_size = config.mapping.tp_size
        tp_rank = config.mapping.tp_rank
        layernorm_epsilon = config.norm_epsilon

        self.input_layernorm = LayerNorm(normalized_shape=hidden_size,
                                         eps=layernorm_epsilon,
                                         dtype=dtype)

        self.attention = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            rotary_embedding_percentage=rotary_dim /
            (hidden_size // num_attention_heads),
            max_position_embeddings=config.max_position_embeddings,
            attention_mask_type=AttentionMaskType.causal,
            dtype=dtype,
            tp_group=None,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=False,
            position_embedding_type=PositionEmbeddingType.rope_gptj,
            quant_mode=config.quant_mode)

        self.mlp = MLP(hidden_size=hidden_size,
                       ffn_hidden_size=hidden_size * 4,
                       hidden_act=config.hidden_act,
                       dtype=dtype,
                       bias=True,
                       tp_group=None,
                       tp_size=tp_size,
                       quant_mode=config.quant_mode)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):
        assert isinstance(hidden_states, Tensor)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(hidden_states,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params)

        if use_cache:
            attention_output, presents = attention_output
        attention_output = attention_output

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attention_output + feed_forward_hidden_states
        if self.config.mapping.tp_size > 1:
            hidden_states = allreduce(hidden_states,
                                      self.config.mapping.tp_group)
        hidden_states = hidden_states + residual

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class GPTJModel(Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config

        if config.mapping.is_first_pp_rank():
            if config.use_parallel_embedding:
                self.vocab_embedding = Embedding(
                    config.vocab_size,
                    config.hidden_size,
                    dtype=config.dtype,
                    tp_group=config.mapping.tp_group,
                    tp_size=config.mapping.tp_size,
                    sharding_dim=config.embedding_sharding_dim,
                    tp_rank=config.mapping.tp_rank)
            else:
                self.vocab_embedding = Embedding(config.vocab_size,
                                                 config.hidden_size,
                                                 dtype=config.dtype)
        self.layers = DecoderLayerList(GPTJDecoderLayer, config)
        if config.mapping.is_last_pp_rank():
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

        for layer, past, pointer, host_pointer, max_attention_window_size in zip(
                self.layers, kv_cache_params.past_key_value,
                kv_cache_params.kv_cache_block_pointers,
                kv_cache_params.host_kv_cache_block_pointers,
                kv_cache_params.host_max_attention_window_sizes):
            hidden_states = layer(
                hidden_states,
                use_cache=use_cache,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    host_max_attention_window_sizes=max_attention_window_size,
                    host_sink_token_length=kv_cache_params.
                    host_sink_token_length,
                    kv_cache_block_pointers=[pointer],
                    host_kv_cache_block_pointers=[host_pointer],
                    cache_indirection=kv_cache_params.cache_indirection),
                attention_params=attention_params)

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class GPTJForCausalLM(DecoderModelForCausalLM):

    def __init__(self, config: PretrainedConfig):
        self.check_config(config)
        transformer = GPTJModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)
        if config.mapping.is_last_pp_rank():
            lm_head = ColumnLinear(config.hidden_size,
                                   vocab_size_padded,
                                   bias=True,
                                   dtype=config.dtype,
                                   tp_group=config.mapping.tp_group,
                                   tp_size=config.mapping.tp_size,
                                   gather_output=True)
        else:
            lm_head = None
        super().__init__(config, transformer, lm_head)

    def check_config(self, config):
        config.set_if_not_exist('rotary_dim', 64)
