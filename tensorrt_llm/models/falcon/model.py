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
from ...functional import Tensor, allreduce, recv, send
from ...layers import (MLP, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, KeyValueCacheParams, LayerNorm)
from ...module import Module
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              PretrainedConfig)


class FalconDecoderLayer(Module):

    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        hidden_size = config.hidden_size
        dtype = config.dtype
        tp_group = config.mapping.tp_group
        tp_size = config.mapping.tp_size
        tp_rank = config.mapping.tp_rank
        layernorm_epsilon = config.norm_epsilon

        self.input_layernorm = LayerNorm(normalized_shape=hidden_size,
                                         eps=layernorm_epsilon,
                                         dtype=dtype)

        self.new_decoder_architecture = config.new_decoder_architecture
        self.parallel_attn = config.parallel_attention
        if self.is_parallel_attention:
            # Not to apply allreduce inside the Attention/MLP layers.
            # allreduce applies after those layer.
            tp_group = None
        self.attention = Attention(
            hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            attention_mask_type=AttentionMaskType.causal,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=config.bias,
            position_embedding_type=config.position_embedding_type,
            quant_mode=config.quant_mode,
        )

        mlp_hidden_size = hidden_size * 4 if config.intermediate_size is None else config.intermediate_size

        if self.new_decoder_architecture:
            # Layernorm before MLP.
            self.mlp_layernorm = LayerNorm(normalized_shape=hidden_size,
                                           eps=layernorm_epsilon,
                                           dtype=dtype)
        else:
            self.mlp_layernorm = None
        self.mlp = MLP(
            hidden_size=hidden_size,
            ffn_hidden_size=mlp_hidden_size,
            hidden_act=config.hidden_act,
            dtype=dtype,
            bias=config.bias,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=config.quant_mode,
        )
        if self.is_parallel_attention:
            self.post_layernorm = None
        else:
            self.post_layernorm = LayerNorm(normalized_shape=hidden_size,
                                            dtype=dtype)

    @property
    def is_parallel_attention(self):
        return self.new_decoder_architecture or self.parallel_attn

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):
        assert isinstance(hidden_states, Tensor)

        residual = hidden_states

        if self.new_decoder_architecture:
            mlp_ln_output = self.mlp_layernorm(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        input_ln_output = hidden_states
        attention_output = self.attention(hidden_states,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params)

        if use_cache:
            attention_output, presents = attention_output

        if not self.new_decoder_architecture:
            if self.parallel_attn:
                hidden_states = input_ln_output
            else:
                hidden_states = residual + attention_output
                residual = hidden_states
                hidden_states = self.post_layernorm(hidden_states)
        else:
            hidden_states = mlp_ln_output

        hidden_states = self.mlp(hidden_states)

        if self.is_parallel_attention:
            hidden_states = hidden_states + attention_output
            if self.config.mapping.tp_size > 1:
                hidden_states = allreduce(hidden_states,
                                          self.config.mapping.tp_group)

        hidden_states = residual + hidden_states
        if use_cache:
            return hidden_states, presents
        return hidden_states


class FalconModel(Module):

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

        self.layers = DecoderLayerList(FalconDecoderLayer, config)
        if config.mapping.is_last_pp_rank():
            self.ln_f = LayerNorm(normalized_shape=config.hidden_size,
                                  dtype=config.dtype)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None):

        kv_cache_params.fill_none_tensor_list(len(self.layers))

        if use_cache:
            presents = []

        if self.config.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids)
        else:
            hidden_states = recv(hidden_states,
                                 self.config.mapping.prev_pp_rank())

        for layer, past, pointer, host_pointer, max_attention_window_size in zip(
                self.layers, kv_cache_params.past_key_value,
                kv_cache_params.kv_cache_block_pointers,
                kv_cache_params.host_kv_cache_block_pointers,
                kv_cache_params.host_max_attention_window_sizes):
            hidden_states = layer(
                hidden_states,
                use_cache=use_cache,
                attention_mask=attention_mask,
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

        if self.config.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states,
                                 self.config.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class FalconForCausalLM(DecoderModelForCausalLM):

    def __init__(self, config: PretrainedConfig):
        self.check_config(config)
        transformer = FalconModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)
        if config.mapping.is_last_pp_rank():
            share_weight = None
            if config.share_embedding_table:
                share_weight = transformer.vocab_embedding.weight

            lm_head = ColumnLinear(config.hidden_size,
                                   vocab_size_padded,
                                   bias=False,
                                   dtype=config.dtype,
                                   tp_group=config.mapping.tp_group,
                                   tp_size=config.mapping.tp_size,
                                   gather_output=True,
                                   share_weight=share_weight)
        else:
            lm_head = None
        super().__init__(config, transformer, lm_head)

    def check_config(self, config):
        config.set_if_not_exist('bias', True)
        config.set_if_not_exist('new_decoder_architecture', False)
        config.set_if_not_exist('parallel_attention', False)
