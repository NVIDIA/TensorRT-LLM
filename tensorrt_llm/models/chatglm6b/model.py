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
import math
from collections import OrderedDict

import numpy as np
import tensorrt as trt

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import (PositionEmbeddingType, Tensor, assertion, concat,
                           constant, gather_last_token_logits, gpt_attention,
                           shape, split)
from ...layers import (MLP, AttentionMaskType, AttentionParams, ColumnLinear,
                       Embedding, KeyValueCacheParams, LayerNorm, RowLinear)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...parameter import Parameter
from ...quantization import QuantMode


class ChatGLMAttention(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 num_layers=1,
                 apply_query_key_layer_scaling=False,
                 bias=True,
                 dtype=None,
                 position_embedding_type:
                 PositionEmbeddingType = PositionEmbeddingType.learned_absolute,
                 use_int8_kv_cache=False,
                 tp_group=None,
                 tp_size=1,
                 multi_block_mode=False,
                 multi_query_mode=False):
        super().__init__()

        self.attention_mask_type = AttentionMaskType.bidirectional
        self.attention_head_size = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads // tp_size
        self.num_attention_kv_heads = 1 if multi_query_mode else self.num_attention_heads
        self.hidden_size = hidden_size // tp_size
        self.max_position_embeddings = max_position_embeddings

        self.num_layers = num_layers
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.q_scaling = 1
        if self.apply_query_key_layer_scaling:
            self.norm_factor *= self.num_layers
            self.q_scaling *= self.num_layers

        self.multi_block_mode = multi_block_mode
        self.multi_query_mode = multi_query_mode

        self.rotary_embedding_dim = 0
        self.position_embedding_type = position_embedding_type
        self.dtype = dtype

        self.use_int8_kv_cache = use_int8_kv_cache
        if self.use_int8_kv_cache:
            self.kv_orig_quant_scale = Parameter(shape=(1, ), dtype='float32')
            self.kv_quant_orig_scale = Parameter(shape=(1, ), dtype='float32')
        else:
            self.register_parameter('kv_orig_quant_scale', None)
            self.register_parameter('kv_quant_orig_scale', None)

        # Note: in multi_query_mode, only query heads are split between multiple GPUs,
        # while key/value head are not split as there is only one head per key/value.
        # The output feature size is therefore (h/tp + 2) * d, where h is num_heads,
        # d is head_size, and tp is tensor_parallel_size.
        # In ColumnLinear op, the output dim is calculated by (h + 2*tp) * d / tp,
        # which matches the desired output size (h/tp + 2) * d after splitting
        self.qkv = ColumnLinear(hidden_size,
                                hidden_size *
                                3 if not multi_query_mode else hidden_size +
                                2 * tp_size * self.attention_head_size,
                                bias=bias,
                                dtype=dtype,
                                tp_group=tp_group,
                                tp_size=tp_size,
                                gather_output=False)
        self.dense = RowLinear(hidden_size,
                               hidden_size,
                               bias=bias,
                               dtype=dtype,
                               tp_group=tp_group,
                               tp_size=tp_size)

    def forward(self,
                hidden_states: Tensor,
                position_embedding,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):

        if not default_net().plugin_config.gpt_attention_plugin:
            raise ValueError(
                'ChatGLM is only supported with GPTAttention plugin')

        assert isinstance(hidden_states, Tensor)
        qkv = self.qkv(hidden_states)

        # attention

        qkv = qkv.view(
            concat([
                shape(qkv, 0),
                shape(qkv, 1), self.num_attention_heads, 3,
                self.attention_head_size
            ]))
        query, key, value = split(qkv, 1, dim=3)
        query = query.view(
            concat([
                shape(qkv, 0),
                shape(qkv, 1), self.num_attention_heads,
                self.attention_head_size
            ]))
        key = key.view(
            concat([
                shape(qkv, 0),
                shape(qkv, 1), self.num_attention_heads,
                self.attention_head_size
            ]))
        value = value.view(
            concat([
                shape(qkv, 0),
                shape(qkv, 1), self.num_attention_heads,
                self.attention_head_size
            ]))
        zero = constant(
            np.ascontiguousarray(
                np.zeros([1, 1, 1, 1],
                         dtype=np.float16
                         if self.dtype == trt.float16 else np.float32)))

        def rotate(x64):
            x32_part0, x32_part1 = x64.split(32, dim=-1)

            x32_part1_negtive = zero - x32_part1

            y64 = concat([x32_part1_negtive, x32_part0], dim=3)
            return y64

        def rotate_embedding(x, position_embedding_value):
            cos0, cos1, sin0, sin1 = position_embedding_value

            x128 = x
            x64_part0, x64_part1 = x128.split(64, dim=-1)

            x64_part0_rotate = rotate(x64_part0)
            y64_part0 = x64_part0 * cos0 + x64_part0_rotate * sin0

            x64_part1_rotate = rotate(x64_part1)
            y64_part1 = x64_part1 * cos1 + x64_part1_rotate * sin1

            y128 = concat([y64_part0, y64_part1], dim=3)
            y128 = y128.view(shape(x))
            return y128

        query = rotate_embedding(query, position_embedding)
        key = rotate_embedding(key, position_embedding)

        kv_orig_quant_scale = self.kv_orig_quant_scale.value if self.use_int8_kv_cache else None
        kv_quant_orig_scale = self.kv_quant_orig_scale.value if self.use_int8_kv_cache else None

        qkv = concat([query, key, value], dim=2)
        qkv = qkv.view(
            concat([shape(qkv, 0),
                    shape(qkv, 1), self.hidden_size * 3]))
        context, past_key_value = gpt_attention(
            tensor=qkv,
            past_key_value=kv_cache_params.get_first_past_key_value(),
            sequence_length=attention_params.sequence_length,
            host_past_key_value_lengths=kv_cache_params.
            host_past_key_value_lengths,
            context_lengths=attention_params.context_lengths,
            cache_indirection=kv_cache_params.cache_indirection,
            host_request_types=attention_params.host_request_types,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_attention_kv_heads,
            hidden_size_per_head=self.attention_head_size,
            q_scaling=self.q_scaling,
            rotary_embedding_dim=self.rotary_embedding_dim,
            position_embedding_type=self.position_embedding_type,
            multi_block_mode=self.multi_block_mode,
            kv_orig_quant_scale=kv_orig_quant_scale,
            kv_quant_orig_scale=kv_quant_orig_scale,
            kv_cache_quant_mode=QuantMode.from_description(
                use_int8_kv_cache=self.use_int8_kv_cache),
            max_context_length=attention_params.max_context_length,
            mask_type=self.attention_mask_type.value,
            host_context_lengths=attention_params.host_context_lengths)

        context = self.dense(context)

        if use_cache:
            return (context, past_key_value)
        else:
            return context


class ChatGLM6BDecoderLayer(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 num_layers,
                 dtype=None,
                 apply_query_key_layer_scaling=False,
                 hidden_act='relu',
                 quant_mode=QuantMode(0),
                 inter_size=None,
                 bias=True,
                 tp_group=None,
                 tp_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.num_layers = num_layers
        self.dtype = dtype
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.hidden_act = hidden_act
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.input_layernorm = LayerNorm(normalized_shape=hidden_size,
                                         dtype=dtype)

        self.attention = ChatGLMAttention(
            hidden_size,
            num_attention_heads,
            max_position_embeddings,
            num_layers,
            apply_query_key_layer_scaling,
            dtype=dtype,
            position_embedding_type=PositionEmbeddingType.learned_absolute,
            bias=bias,
            tp_group=tp_group,
            tp_size=tp_size,
            use_int8_kv_cache=quant_mode.has_int8_kv_cache())

        if inter_size is None:
            inter_size = hidden_size * 4

        self.mlp = MLP(hidden_size=hidden_size,
                       ffn_hidden_size=inter_size,
                       hidden_act=hidden_act,
                       dtype=dtype,
                       bias=bias,
                       tp_group=tp_group,
                       tp_size=tp_size)
        self.post_layernorm = LayerNorm(normalized_shape=hidden_size,
                                        dtype=dtype)

    def forward(self,
                hidden_states: Tensor,
                position_embedding,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):

        assert isinstance(hidden_states, Tensor)
        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(hidden_states,
                                          position_embedding,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params)

        if use_cache:
            attention_output, presents = attention_output

        hidden_states = hidden_states * 7.484375 + attention_output

        hidden_states = self.post_layernorm(hidden_states)

        mlp_output = self.mlp(hidden_states)

        hidden_states = hidden_states * 7.484375 + mlp_output

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class ChatGLM6BModel(Module):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 dtype=None,
                 mapping=Mapping(),
                 apply_query_key_layer_scaling=False,
                 inter_size=None,
                 bias=True,
                 quant_mode=QuantMode(0)):
        super().__init__()

        self.half_head_size = hidden_size // num_heads // 2

        self.embedding = Embedding(vocab_size, hidden_size, dtype=dtype)
        self.position_embedding_cos = Embedding(max_position_embeddings,
                                                self.half_head_size,
                                                dtype=dtype)
        self.position_embedding_sin = Embedding(max_position_embeddings,
                                                self.half_head_size,
                                                dtype=dtype)

        self.layers = ModuleList([
            ChatGLM6BDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                max_position_embeddings=max_position_embeddings,
                num_layers=num_layers,
                dtype=dtype,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                hidden_act=hidden_act,
                tp_group=mapping.tp_group,
                tp_size=mapping.tp_size,
                inter_size=inter_size,
                bias=bias,
                quant_mode=quant_mode) for _ in range(num_layers)
        ])

        self.ln_f = LayerNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(self,
                input_ids=None,
                position_ids=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):

        batch_size = shape(input_ids, 0)
        input_len = shape(input_ids, 1)

        hidden_states = self.embedding(input_ids)

        position_embedding_cos = self.position_embedding_cos(position_ids)
        position_embedding_sin = self.position_embedding_sin(position_ids)

        position_embedding_cos0, position_embedding_cos1 = position_embedding_cos.split(
            1, dim=1)
        position_embedding_sin0, position_embedding_sin1 = position_embedding_sin.split(
            1, dim=1)

        position_embedding_cos0 = position_embedding_cos0.view(
            concat([batch_size, input_len, 1, self.half_head_size]))
        position_embedding_cos1 = position_embedding_cos1.view(
            concat([batch_size, input_len, 1, self.half_head_size]))
        position_embedding_sin0 = position_embedding_sin0.view(
            concat([batch_size, input_len, 1, self.half_head_size]))
        position_embedding_sin1 = position_embedding_sin1.view(
            concat([batch_size, input_len, 1, self.half_head_size]))

        position_embedding = [
            position_embedding_cos0, position_embedding_cos1,
            position_embedding_sin0, position_embedding_sin1
        ]

        if kv_cache_params.past_key_value is None:
            kv_cache_params.past_key_value = tuple([None] * len(self.layers))

        if use_cache:
            presents = []

        for layer, past in zip(self.layers, kv_cache_params.past_key_value):
            hidden_states = layer(
                hidden_states,
                position_embedding,
                use_cache=use_cache,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    cache_indirection=kv_cache_params.cache_indirection),
                attention_params=attention_params)

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class ChatGLM6BHeadModel(ChatGLM6BModel):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 dtype,
                 mapping=Mapping(),
                 apply_query_key_layer_scaling=False,
                 inter_size=None,
                 bias=True,
                 quant_mode=QuantMode(0)):
        if isinstance(dtype, str):
            self._kv_dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self._kv_dtype = dtype
        self._dtype = self._kv_dtype
        if quant_mode.has_int8_kv_cache():
            self._kv_dtype = str_dtype_to_trt('int8')
        elif quant_mode.has_fp8_kv_cache():
            self._kv_dtype = str_dtype_to_trt('fp8')

        self.quant_mode = quant_mode

        self._num_layers = num_layers
        self._num_heads = num_heads
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._tp_size = mapping.tp_size
        super().__init__(num_layers, num_heads, hidden_size, vocab_size,
                         hidden_act, max_position_embeddings, dtype, mapping,
                         apply_query_key_layer_scaling, inter_size, bias,
                         quant_mode)
        vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
        self.lm_head = ColumnLinear(hidden_size,
                                    vocab_size_padded,
                                    bias=False,
                                    dtype=dtype,
                                    tp_group=mapping.tp_group,
                                    tp_size=mapping.tp_size,
                                    gather_output=True)

    def forward(self,
                input_ids=None,
                position_ids=None,
                use_cache=False,
                last_token_ids=None,
                kv_cache_params=None,
                attention_params=None):

        hidden_states = super().forward(input_ids, position_ids, use_cache,
                                        kv_cache_params, attention_params)

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = gather_last_token_logits(
            hidden_states, last_token_ids,
            default_net().plugin_config.remove_input_padding)

        # [batch_size, hidden_size] -> [batch_size, vocab_size]
        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output('logits', self._dtype)
        # out_inter.mark_output('inter', str_dtype_to_trt('float32'))

        if use_cache and default_net().plugin_config.paged_kv_cache == False:
            for i, present in enumerate(presents):
                present.mark_output(f'present_key_value_{i}', self._kv_dtype)
            return (lm_logits, presents)

        return lm_logits

    def prepare_inputs(self,
                       max_batch_size,
                       max_input_len,
                       max_new_tokens,
                       use_cache,
                       max_beam_width: int = 1):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        head_size = self._hidden_size // self._num_heads
        num_heads = self._num_heads // self._tp_size
        num_heads_kv = num_heads
        max_len = max_input_len + max_new_tokens
        bb_range = [
            1, (max_batch_size * max_beam_width + 1) // 2,
            max_batch_size * max_beam_width
        ]
        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        beam_width_range = [1, (max_beam_width + 1) // 2, max_beam_width]
        inlen_range = [1, 1, max_input_len]
        max_len_range = [1, (max_len + 1) // 2 + 1, max_len + 1]

        past_key_value = []
        sequence_length = None
        host_past_key_value_lengths = None

        input_ids = Tensor(name='input_ids',
                           dtype=trt.int32,
                           shape=[-1, -1],
                           dim_range=OrderedDict([
                               ('batch_beam_size', [bb_range]),
                               ('input_len', [inlen_range]),
                           ]))

        position_ids = Tensor(name='position_ids',
                              dtype=trt.int32,
                              shape=[-1, 2, -1],
                              dim_range=OrderedDict([
                                  ('batch_beam_size', [bb_range]),
                                  ('2', [2]),
                                  ('input_len', [inlen_range]),
                              ]))

        for i in range(self._num_layers):
            kv_dim_range = OrderedDict([
                ('batch_beam_size', [bb_range]),
                ('kv', [2]),
                ('num_heads', [num_heads_kv]),
                ('past_key_len', [max_len_range]),
                ('head_size', [head_size]),
            ])
            kv = Tensor(name=f'past_key_value_{i}',
                        dtype=self._kv_dtype,
                        shape=[-1, 2, num_heads_kv, -1, head_size],
                        dim_range=kv_dim_range)
            past_key_value.append(kv)

            # TODO(kaiyu): Remove this when TRT fix the named dimension
            assertion(shape(input_ids, 0) == shape(kv, 0), 'batch size')

        sequence_length = Tensor(
            name='sequence_length',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([('batch_beam_size', [bb_range])]),
        )
        host_past_key_value_lengths = Tensor(
            name='host_past_key_value_lengths',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([('batch_beam_size', [bb_range])]),
        )
        context_lengths = Tensor(name='context_lengths',
                                 dtype=trt.int32,
                                 shape=[-1],
                                 dim_range=OrderedDict([('batch_beam_size',
                                                         [bb_range])]))

        host_context_lengths = None
        if default_net().plugin_config.remove_input_padding:
            host_context_lengths = Tensor(name='host_context_lengths',
                                          dtype=trt.int32,
                                          shape=[-1],
                                          dim_range=OrderedDict([
                                              ('batch_beam_size', [bb_range])
                                          ]))

        host_request_types = Tensor(name='host_request_types',
                                    dtype=trt.int32,
                                    shape=[-1],
                                    dim_range=OrderedDict([('batch_beam_size',
                                                            [bb_range])]))

        last_token_ids = Tensor(name='last_token_ids',
                                dtype=trt.int32,
                                shape=[-1],
                                dim_range=OrderedDict([
                                    ('batch_beam_size', [bb_range]),
                                ]))

        cache_indirection = Tensor(name='cache_indirection',
                                   dtype=trt.int32,
                                   shape=[-1, -1, -1],
                                   dim_range=OrderedDict([
                                       ('batch_size', [bs_range]),
                                       ('beam_width', [beam_width_range]),
                                       ('max_seq_len', [max_len_range]),
                                   ]))

        return (input_ids, position_ids, True, last_token_ids,
                KeyValueCacheParams(
                    past_key_value=past_key_value,
                    host_past_key_value_lengths=host_past_key_value_lengths,
                    cache_indirection=cache_indirection,
                ),
                AttentionParams(sequence_length=sequence_length,
                                context_lengths=context_lengths,
                                host_context_lengths=host_context_lengths,
                                max_context_length=max_input_len,
                                host_request_types=host_request_types))
