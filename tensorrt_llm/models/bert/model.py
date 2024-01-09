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
import math

import numpy as np

from ..._common import default_net
from ...functional import (bert_attention, concat, constant, expand, matmul,
                           shape, slice, softmax, split)
from ...layers import MLP, ColumnLinear, Embedding, LayerNorm, Linear, RowLinear
from ...mapping import Mapping
from ...module import Module, ModuleList


class BertEmbedding(Module):

    def __init__(self,
                 vocab_size,
                 hidden_size,
                 max_position_embeddings,
                 type_vocab_size,
                 dtype=None):
        super().__init__()
        self.vocab_embedding = Embedding(vocab_size, hidden_size, dtype=dtype)
        self.position_embedding = Embedding(max_position_embeddings,
                                            hidden_size,
                                            dtype=dtype)
        self.token_embedding = Embedding(type_vocab_size,
                                         hidden_size,
                                         dtype=dtype)
        self.max_position_embeddings = max_position_embeddings

        self.embedding_ln = LayerNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        position_ids_buffer = constant(
            np.expand_dims(
                np.arange(self.max_position_embeddings).astype(np.int32), 0))

        token_type_ids_buffer = constant(
            np.expand_dims(
                np.zeros(self.max_position_embeddings).astype(np.int32), 0))

        seq_len_2d = concat([1, shape(input_ids, 1)])

        if position_ids is None:
            # slice
            position_ids = slice(position_ids_buffer,
                                 starts=[0, 0],
                                 sizes=seq_len_2d)
            position_ids = expand(position_ids, shape(input_ids))

        if token_type_ids is None:
            # slice
            token_type_ids = slice(token_type_ids_buffer,
                                   starts=[0, 0],
                                   sizes=seq_len_2d)
            token_type_ids = expand(token_type_ids, shape(input_ids))

        x = self.vocab_embedding(input_ids)
        x = x + self.position_embedding(position_ids)
        x = x + self.token_embedding(token_type_ids)
        x = self.embedding_ln(x)
        return x


class BertAttention(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 dtype=None,
                 tp_group=None,
                 tp_size=1):
        super().__init__()

        self.attention_head_size = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads // tp_size
        self.hidden_size = hidden_size // tp_size
        self.max_position_embeddings = max_position_embeddings
        self.norm_factor = math.sqrt(self.attention_head_size)

        self.qkv = ColumnLinear(hidden_size,
                                hidden_size * 3,
                                dtype=dtype,
                                tp_group=tp_group,
                                tp_size=tp_size,
                                gather_output=False)
        self.dense = RowLinear(hidden_size,
                               hidden_size,
                               dtype=dtype,
                               tp_group=tp_group,
                               tp_size=tp_size)

    def forward(self, hidden_states, attention_mask=None, input_lengths=None):
        qkv = self.qkv(hidden_states)

        # attention
        if default_net().plugin_config.bert_attention_plugin:
            assert input_lengths is not None
            context = bert_attention(qkv, input_lengths,
                                     self.num_attention_heads,
                                     self.attention_head_size, 1.0)
        else:

            def transpose_for_scores(x):
                new_x_shape = concat([
                    shape(x, 0),
                    shape(x, 1), self.num_attention_heads,
                    self.attention_head_size
                ])
                return x.view(new_x_shape).permute([0, 2, 1, 3])

            query, key, value = split(qkv, self.hidden_size, dim=2)
            query = transpose_for_scores(query)
            key = transpose_for_scores(key)
            value = transpose_for_scores(value)

            key = key.permute([0, 1, 3, 2])
            attention_scores = matmul(query, key)
            attention_scores = attention_scores / self.norm_factor

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            attention_probs = softmax(attention_scores, dim=-1)

            context = matmul(attention_probs, value,
                             use_fp32_acc=False).permute([0, 2, 1, 3])
            context = context.view(
                concat([shape(context, 0),
                        shape(context, 1), self.hidden_size]))

        context = self.dense(context)

        return context


class BertEncoderLayer(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 hidden_act='relu',
                 tp_group=None,
                 tp_size=1,
                 dtype=None):
        super().__init__()
        self.input_layernorm = LayerNorm(normalized_shape=hidden_size,
                                         dtype=dtype)

        self.attention = BertAttention(hidden_size,
                                       num_attention_heads,
                                       max_position_embeddings,
                                       tp_group=tp_group,
                                       tp_size=tp_size,
                                       dtype=dtype)
        self.mlp = MLP(hidden_size=hidden_size,
                       ffn_hidden_size=hidden_size * 4,
                       hidden_act=hidden_act,
                       tp_group=tp_group,
                       tp_size=tp_size,
                       dtype=dtype)
        self.post_layernorm = LayerNorm(normalized_shape=hidden_size,
                                        dtype=dtype)

    def forward(self, hidden_states, attention_mask=None, input_lengths=None):
        residual = hidden_states

        attention_output = self.attention(hidden_states,
                                          attention_mask=attention_mask,
                                          input_lengths=input_lengths)

        hidden_states = residual + attention_output

        hidden_states = self.input_layernorm(hidden_states)

        residual = hidden_states

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        hidden_states = self.post_layernorm(hidden_states)

        return hidden_states


class BertModel(Module):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 type_vocab_size,
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()

        self.embedding = BertEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            dtype=dtype)

        self.layers = ModuleList([
            BertEncoderLayer(hidden_size=hidden_size,
                             num_attention_heads=num_heads,
                             max_position_embeddings=max_position_embeddings,
                             hidden_act=hidden_act,
                             tp_group=mapping.tp_group,
                             tp_size=mapping.tp_size,
                             dtype=dtype) for _ in range(num_layers)
        ])

    def forward(self,
                input_ids=None,
                input_lengths=None,
                token_type_ids=None,
                position_ids=None,
                hidden_states=None):
        hidden_states = self.embedding(input_ids, position_ids, token_type_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states,
                                  input_lengths=input_lengths)

        return hidden_states


class BertForQuestionAnswering(Module):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 type_vocab_size,
                 num_labels=2,
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()
        self.bert = BertModel(num_layers=num_layers,
                              num_heads=num_heads,
                              hidden_size=hidden_size,
                              vocab_size=vocab_size,
                              hidden_act=hidden_act,
                              max_position_embeddings=max_position_embeddings,
                              type_vocab_size=type_vocab_size,
                              mapping=mapping,
                              dtype=dtype)
        self.num_labels = num_labels
        self.qa_outputs = Linear(hidden_size, num_labels, dtype=dtype)

    def forward(self,
                input_ids=None,
                input_lengths=None,
                token_type_ids=None,
                position_ids=None,
                hidden_states=None):

        hidden_states = self.bert.forward(input_ids=input_ids,
                                          input_lengths=input_lengths,
                                          token_type_ids=token_type_ids,
                                          position_ids=position_ids,
                                          hidden_states=hidden_states)

        logits = self.qa_outputs(hidden_states)

        return logits
