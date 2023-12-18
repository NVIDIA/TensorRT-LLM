# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from ...functional import Tensor
from ...layers import (MLP, Attention, AttentionMaskType, ColumnLinear,
                       LayerNorm, PositionEmbeddingType)
from ...module import Module
from ..gpt.model import GPTEmbedding
from ..modeling_utils import DecoderLayerList, DecoderModelForCausalLM


class OPTDecoderLayer(Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.do_layer_norm_before = self.config.do_layer_norm_before

        hidden_size = config.hidden_size
        dtype = config.dtype
        tp_group = config.mapping.tp_group
        tp_size = config.mapping.tp_size

        self.input_layernorm = LayerNorm(normalized_shape=hidden_size,
                                         dtype=dtype)

        self.attention = Attention(
            hidden_size,
            config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            attention_mask_type=AttentionMaskType.causal,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size)

        mlp_hidden_size = hidden_size * 4 if config.intermediate_size is None else config.intermediate_size

        self.mlp = MLP(hidden_size=hidden_size,
                       ffn_hidden_size=mlp_hidden_size,
                       hidden_act=config.hidden_act,
                       dtype=dtype,
                       tp_group=tp_group,
                       tp_size=tp_size)
        self.post_layernorm = LayerNorm(normalized_shape=hidden_size,
                                        dtype=dtype)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):
        residual = hidden_states

        attention_input = hidden_states
        if self.do_layer_norm_before:
            attention_input = self.input_layernorm(hidden_states)

        # At this point the hidden_states object must be a Tensor.
        assert isinstance(attention_input, Tensor)

        attention_output = self.attention(attention_input,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params)
        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output
        if not self.do_layer_norm_before:
            hidden_states = self.input_layernorm(hidden_states)

        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        if not self.do_layer_norm_before:
            hidden_states = self.post_layernorm(hidden_states)

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class OPTModel(Module):

    def __init__(self, config):
        super().__init__()
        self.do_layer_norm_before = config.do_layer_norm_before
        use_parallel_embedding = config.use_parallel_embedding
        embedding_sharding_dim = config.embedding_sharding_dim
        use_prompt_tuning = config.use_prompt_tuning
        mapping = config.mapping

        self.embedding = GPTEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.max_position_embeddings,
            position_embedding_type=PositionEmbeddingType.learned_absolute,
            dtype=config.dtype,
            use_prompt_tuning=use_prompt_tuning,
            tensor_parallel=mapping.tp_size if use_parallel_embedding else 1,
            tensor_parallel_group=mapping.tp_group
            if use_parallel_embedding else None,
            sharding_dim=embedding_sharding_dim,
            tp_rank=mapping.tp_rank)

        self.layers = DecoderLayerList(OPTDecoderLayer, config)

        if self.do_layer_norm_before:
            self.ln_f = LayerNorm(normalized_shape=config.hidden_size,
                                  dtype=config.dtype)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                prompt_embedding_table=None,
                prompt_tasks=None,
                prompt_vocab_size=None):

        hidden_states = self.embedding(input_ids, position_ids,
                                       prompt_embedding_table, prompt_tasks,
                                       prompt_vocab_size)

        hidden_states = self.layers(hidden_states,
                                    use_cache=use_cache,
                                    attention_mask=attention_mask,
                                    kv_cache_params=kv_cache_params,
                                    attention_params=attention_params)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.do_layer_norm_before:
            hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class OPTForCausalLM(DecoderModelForCausalLM):

    def __init__(self, config):
        use_parallel_embedding = config.use_parallel_embedding
        embedding_sharding_dim = config.embedding_sharding_dim
        share_embedding_table = config.share_embedding_table
        mapping = config.mapping

        if share_embedding_table and mapping.tp_size > 1:
            if (not use_parallel_embedding) or (use_parallel_embedding and
                                                embedding_sharding_dim == 1):
                raise NotImplementedError(
                    'For multiple-processes cases, sharing the embedding table must set use_parallel_embedding=True and embedding_sharding_dim=0'
                )

        transformer = OPTModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)

        share_weight = None
        if share_embedding_table:
            share_weight = transformer.embedding.vocab_embedding.weight

        lm_head = ColumnLinear(config.hidden_size,
                               vocab_size_padded,
                               bias=False,
                               dtype=config.dtype,
                               tp_group=mapping.tp_group,
                               tp_size=mapping.tp_size,
                               gather_output=True,
                               share_weight=share_weight)

        super().__init__(config, transformer, lm_head)

    def check_config(self):
        self.config.set_if_not_exist('use_parallel_embedding', False)
        self.config.set_if_not_exist('embedding_sharding_dim', 0)
        self.config.set_if_not_exist('share_embedding_table', False)
        self.config.set_if_not_exist('do_layer_norm_before', False)
