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
from ...functional import Tensor, recv, send
from ...layers import (MOE, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, GatedMLP, LayerNorm)
from ...module import Module
from ..modeling_utils import DecoderLayerList, DecoderModelForCausalLM
from .config import DbrxConfig


class DbrxDecoderLayer(Module):

    def __init__(self, config: DbrxConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.input_layernorm = LayerNorm(normalized_shape=config.hidden_size,
                                         eps=config.norm_epsilon,
                                         bias=False,
                                         dtype=config.dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]
        self.attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=config.bias,
            position_embedding_type=config.position_embedding_type,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            quant_mode=config.quant_mode,
            clip_qkv=config.clip_qkv)

        ClsMLP = GatedMLP
        mlp_kwargs = {}
        if config.moe.has_moe():
            ClsMLP = MOE
            mlp_kwargs = {
                "moe_config": config.moe,
                "mapping": config.mapping,
            }

        self.mlp = ClsMLP(hidden_size=config.hidden_size,
                          ffn_hidden_size=config.intermediate_size,
                          hidden_act=config.hidden_act,
                          dtype=config.dtype,
                          bias=config.bias,
                          tp_group=config.mapping.tp_group,
                          tp_size=config.mapping.tp_size,
                          quant_mode=config.quant_mode,
                          **mlp_kwargs)
        self.post_layernorm = LayerNorm(normalized_shape=config.hidden_size,
                                        eps=config.norm_epsilon,
                                        bias=False,
                                        dtype=config.dtype)

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

        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class DbrxModel(Module):

    def __init__(self, config: DbrxConfig):
        super().__init__()
        self.config = config

        if config.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)

        self.layers = DecoderLayerList(DbrxDecoderLayer, config)

        if config.mapping.is_last_pp_rank():
            self.ln_f = LayerNorm(normalized_shape=config.hidden_size,
                                  eps=config.norm_epsilon,
                                  bias=False,
                                  dtype=config.dtype)

    def forward(self,
                input_ids,
                position_ids,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None):

        if self.config.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids)
        else:
            hidden_states = recv(hidden_states,
                                 self.config.mapping.prev_pp_rank())

        hidden_states = self.layers(hidden_states,
                                    use_cache=use_cache,
                                    attention_mask=attention_mask,
                                    kv_cache_params=kv_cache_params,
                                    attention_params=attention_params)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.config.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states,
                                 self.config.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class DbrxForCausalLM(DecoderModelForCausalLM):
    config_class = DbrxConfig

    def __init__(self, config: DbrxConfig):
        transformer = DbrxModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)
        if config.mapping.is_last_pp_rank():
            lm_head = ColumnLinear(config.hidden_size,
                                   vocab_size_padded,
                                   bias=config.bias,
                                   dtype=config.dtype,
                                   tp_group=config.mapping.tp_group,
                                   tp_size=config.mapping.tp_size,
                                   gather_output=True)
        else:
            lm_head = None
        self.quant_mode = config.quant_mode
        self.mapping = config.mapping
        super().__init__(config, transformer, lm_head)
