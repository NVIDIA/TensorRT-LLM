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

import tensorrt as trt

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import (PositionEmbeddingType, Tensor, assertion,
                           gather_last_token_logits, shape)
from ...layers import (MLP, Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, Embedding, KeyValueCacheParams, LayerNorm)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...quantization import QuantMode


class GPTJDecoderLayer(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 rotary_dim,
                 dtype=None,
                 hidden_act='relu',
                 tp_group=None,
                 tp_size=1,
                 quant_mode=QuantMode(0)):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rotary_dim = rotary_dim
        self.dtype = dtype
        self.hidden_act = hidden_act
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.quant_mode = quant_mode
        self.input_layernorm = LayerNorm(normalized_shape=hidden_size,
                                         dtype=dtype)

        self.attention = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            rotary_embedding_percentage=rotary_dim /
            (hidden_size // num_attention_heads),
            position_embedding_type=PositionEmbeddingType.rope_gptj,
            max_position_embeddings=max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
            use_int8_kv_cache=quant_mode.has_int8_kv_cache(),
            quant_mode=quant_mode)

        self.mlp = MLP(hidden_size=hidden_size,
                       ffn_hidden_size=hidden_size * 4,
                       hidden_act=hidden_act,
                       dtype=dtype,
                       tp_group=tp_group,
                       tp_size=tp_size,
                       quant_mode=quant_mode)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):
        if not default_net(
        ).plugin_config.layernorm_plugin and trt.__version__[:3] == '8.6':
            raise AssertionError(
                "You need to enable the LayerNorm plugin for GPT-J with TensorRT 8.6. Please set plugin_config.layernorm_plugin"
            )
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
        hidden_states = attention_output + feed_forward_hidden_states + residual
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class GPTJModel(Module):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 rotary_dim,
                 dtype=None,
                 mapping=Mapping(),
                 quant_mode=QuantMode(0)):
        super().__init__()
        self.embedding = Embedding(vocab_size, hidden_size, dtype=dtype)

        self.layers = ModuleList([
            GPTJDecoderLayer(hidden_size=hidden_size,
                             num_attention_heads=num_heads,
                             max_position_embeddings=max_position_embeddings,
                             rotary_dim=rotary_dim,
                             dtype=dtype,
                             hidden_act=hidden_act,
                             tp_group=mapping.tp_group,
                             tp_size=mapping.tp_size,
                             quant_mode=quant_mode) for _ in range(num_layers)
        ])

        self.ln_f = LayerNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(self,
                input_ids: Tensor,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):

        hidden_states = self.embedding(input_ids)

        if kv_cache_params.past_key_value is None:
            kv_cache_params.past_key_value = tuple([None] * len(self.layers))

        if use_cache:
            presents = []

        for layer, past, pointer in zip(
                self.layers, kv_cache_params.past_key_value,
                kv_cache_params.kv_cache_block_pointers):
            hidden_states = layer(
                hidden_states,
                use_cache=use_cache,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    kv_cache_block_pointers=[pointer],
                    cache_indirection=kv_cache_params.cache_indirection),
                attention_params=attention_params)

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class GPTJForCausalLM(GPTJModel):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 rotary_dim,
                 dtype,
                 logits_dtype='float32',
                 mapping=Mapping(),
                 quant_mode=QuantMode(0)):
        if isinstance(dtype, str):
            self._dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self._dtype = dtype
        self._kv_dtype = dtype
        self.quant_mode = quant_mode
        if quant_mode.has_int8_kv_cache():
            self._kv_dtype = str_dtype_to_trt('int8')
        elif quant_mode.has_fp8_kv_cache():
            self._kv_dtype = str_dtype_to_trt('fp8')

        if isinstance(logits_dtype, str):
            self._logits_dtype = str_dtype_to_trt(logits_dtype)
        else:
            assert isinstance(logits_dtype, trt.DataType)
            self._logits_dtype = logits_dtype

        self._num_layers = num_layers
        self._num_heads = num_heads
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._tp_size = mapping.tp_size
        super().__init__(num_layers, num_heads, hidden_size, vocab_size,
                         hidden_act, max_position_embeddings, rotary_dim, dtype,
                         mapping, quant_mode)
        self._vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
        self.lm_head = ColumnLinear(hidden_size,
                                    self._vocab_size_padded,
                                    bias=True,
                                    dtype=dtype,
                                    tp_group=mapping.tp_group,
                                    tp_size=mapping.tp_size,
                                    gather_output=True)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                last_token_ids=None,
                kv_cache_params=None,
                attention_params=None):
        hidden_states = super().forward(input_ids, use_cache, kv_cache_params,
                                        attention_params)

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = gather_last_token_logits(
            hidden_states, last_token_ids,
            default_net().plugin_config.remove_input_padding)

        # [batch_size, hidden_size] -> [batch_size, vocab_size]
        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output('logits', self._logits_dtype)

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
                       max_beam_width,
                       max_num_tokens: int = None,
                       enable_two_optimization_profiles: bool = False):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''
        # Prepare inputs
        head_size = self._hidden_size // self._num_heads
        num_heads = self._num_heads // self._tp_size
        max_len = max_input_len + max_new_tokens
        bb_range_gen = [
            1, (max_batch_size * max_beam_width + 1) // 2,
            max_batch_size * max_beam_width
        ]
        bb_range_cxt = [1, (max_batch_size + 1) // 2, max_batch_size]
        _bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        _beam_width_range = [1, (max_beam_width + 1) // 2, max_beam_width]
        inlen_range_cxt = [1, (max_input_len + 1) // 2, max_input_len]
        inlen_range_gen = [1, 1, 1]
        _max_len_range = [0, (max_len + 1) // 2, max_len]
        if enable_two_optimization_profiles:
            bb_range = [bb_range_cxt, bb_range_gen]
            bs_range = [_bs_range, _bs_range]
            beam_width_range = [_beam_width_range, _beam_width_range]
            inlen_range = [inlen_range_cxt, inlen_range_gen]
            max_len_range = [_max_len_range, _max_len_range]
        else:
            bb_range = [bb_range_gen]
            bs_range = [_bs_range]
            beam_width_range = [_beam_width_range]
            inlen_range = [inlen_range_cxt]
            max_len_range = [_max_len_range]
        if max_num_tokens is None:
            num_tokens_range = [
                1, max_batch_size * max_beam_width,
                max(max_input_len * max_batch_size,
                    max_beam_width * max_batch_size)
            ]
        else:
            num_tokens_range = [1, (max_num_tokens + 1) // 2, max_num_tokens]

        past_key_value = []
        sequence_length = None
        host_past_key_value_lengths = None
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        remove_input_padding = default_net().plugin_config.remove_input_padding
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        tokens_per_block = default_net().plugin_config.tokens_per_block

        if remove_input_padding:
            input_ids = Tensor(name='input_ids',
                               dtype=trt.int32,
                               shape=[1, -1],
                               dim_range=OrderedDict([
                                   ('batch_size_fake', [1]),
                                   ('num_tokens', [num_tokens_range]),
                               ]))
            position_ids = Tensor(name='position_ids',
                                  dtype=trt.int32,
                                  shape=[1, -1],
                                  dim_range=OrderedDict([
                                      ('batch_size_fake', [1]),
                                      ('num_tokens', [num_tokens_range]),
                                  ]))
        else:
            input_ids = Tensor(name='input_ids',
                               dtype=trt.int32,
                               shape=[-1, -1],
                               dim_range=OrderedDict([
                                   ('batch_size_input_ids', bb_range),
                                   ('input_len', inlen_range),
                               ]))
            position_ids = Tensor(name='position_ids',
                                  dtype=trt.int32,
                                  shape=[-1, -1],
                                  dim_range=OrderedDict([
                                      ('batch_size_position_ids', bb_range),
                                      ('input_len', inlen_range),
                                  ]))

        kv_cache_block_pointers_list = []
        if not paged_kv_cache:
            for i in range(self._num_layers):
                kv_dim_range = OrderedDict([
                    ('batch_size_kv', bb_range),
                    ('kv', [2, 2] if enable_two_optimization_profiles else [2]),
                    ('num_heads', [num_heads, num_heads]
                     if enable_two_optimization_profiles else [num_heads]),
                    ('past_key_len', max_len_range),
                    ('head_size', [head_size, head_size]
                     if enable_two_optimization_profiles else [head_size]),
                ])
                kv = Tensor(name=f'past_key_value_{i}',
                            dtype=self._kv_dtype,
                            shape=[-1, 2, num_heads, -1, head_size],
                            dim_range=kv_dim_range)
                past_key_value.append(kv)
                # TODO: Remove this when TRT fix the named dimension
                if not remove_input_padding:
                    assertion(shape(input_ids, 0) == shape(kv, 0), 'batch size')

                kv_cache_block_pointers_list.append(None)
        else:
            max_blocks_per_seq_range = [
                math.ceil(max_len_range[0][0] / tokens_per_block),
                math.ceil(max_len_range[0][1] / tokens_per_block),
                math.ceil(max_len_range[0][2] / tokens_per_block)
            ]

            max_blocks_per_seq_range = [x for x in max_blocks_per_seq_range]

            for i in range(self._num_layers):
                # (blocks, 2, kv_num_heads, tokens_per_block, head_size)

                kv_cache_block_pointers = Tensor(
                    name=f'kv_cache_block_pointers_{i}',
                    dtype=trt.int64,
                    shape=[-1, 2, -1],
                    dim_range=OrderedDict([
                        ('batch_size', bb_range),
                        ('kv', [2]),
                        ('max_blocks_per_seq', [max_blocks_per_seq_range]),
                    ]))
                kv_cache_block_pointers_list.append(kv_cache_block_pointers)
                past_key_value.append(None)

        if use_gpt_attention_plugin:
            dim_range = bb_range
            host_past_key_value_lengths = Tensor(
                name='host_past_key_value_lengths',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict(batch_size_kvl=dim_range))

        context_lengths = None
        host_context_lengths = None
        host_request_types = None
        if use_gpt_attention_plugin and remove_input_padding:
            host_context_lengths = Tensor(name='host_context_lengths',
                                          dtype=trt.int32,
                                          shape=[-1],
                                          dim_range=OrderedDict([('batch_size',
                                                                  bb_range)]))
        if use_gpt_attention_plugin:
            sequence_length = Tensor(
                name='sequence_length',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size', bb_range)]),
            )

            context_lengths = Tensor(name='context_lengths',
                                     dtype=trt.int32,
                                     shape=[-1],
                                     dim_range=OrderedDict([('batch_size',
                                                             bb_range)]))
            host_request_types = Tensor(name='host_request_types',
                                        dtype=trt.int32,
                                        shape=[-1],
                                        dim_range=OrderedDict([('batch_size',
                                                                bb_range)]))

        last_token_ids = Tensor(name='last_token_ids',
                                dtype=trt.int32,
                                shape=[-1],
                                dim_range=OrderedDict([
                                    ('batch_size', bb_range),
                                ]))

        cache_indirection = Tensor(name='cache_indirection',
                                   dtype=trt.int32,
                                   shape=[-1, -1, -1],
                                   dim_range=OrderedDict([
                                       ('batch_size_cache', bs_range),
                                       ('beam_width', beam_width_range),
                                       ('max_seq_len', max_len_range),
                                   ]))

        return (input_ids, position_ids, True, last_token_ids,
                KeyValueCacheParams(
                    past_key_value=past_key_value,
                    host_past_key_value_lengths=host_past_key_value_lengths,
                    kv_cache_block_pointers=kv_cache_block_pointers_list,
                    cache_indirection=cache_indirection,
                ),
                AttentionParams(sequence_length=sequence_length,
                                context_lengths=context_lengths,
                                host_context_lengths=host_context_lengths,
                                max_context_length=max_input_len,
                                host_request_types=host_request_types))
