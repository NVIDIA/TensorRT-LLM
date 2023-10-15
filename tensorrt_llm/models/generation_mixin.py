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

from ..functional import Tensor
from ..mapping import Mapping


class GenerationMixin:

    def get_transformer_layers(self, mapping, num_layers):
        layers_per_pipeline_stage = num_layers // mapping.pp_size
        layers_range = list(
            range(mapping.pp_rank * layers_per_pipeline_stage,
                  (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))
        return layers_range

    def prepare_basic_inputs(self,
                             max_batch_size,
                             max_beam_width,
                             max_input_len,
                             max_new_tokens,
                             num_kv_heads,
                             head_size,
                             num_layers,
                             kv_dtype,
                             remove_input_padding=False,
                             use_gpt_attention_plugin=False,
                             use_gemm_plugin=False,
                             use_custom_all_reduce=False,
                             paged_kv_cache=False,
                             tokens_per_block=64,
                             gather_all_token_logits=False,
                             dtype=None,
                             num_heads=None,
                             mapping=Mapping(),
                             max_num_tokens=None):

        max_len = max_input_len + max_new_tokens

        bb_range_cxt = [1, (max_batch_size + 1) // 2, max_batch_size]
        bb_range_gen = [
            1, (max_batch_size * max_beam_width + 1) // 2,
            max_batch_size * max_beam_width
        ]
        _bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        _beam_width_range = [1, (max_beam_width + 1) // 2, max_beam_width]
        inlen_range_cxt = [1, (max_input_len + 1) // 2, max_input_len]
        inlen_range_gen = [1, 1, 1]
        _mask_len_ctx = [1, (max_input_len + 1) // 2, max_input_len]
        _mask_len_gen = [2, (max_len + 1) // 2 + 1, max_len + 1]
        _kv_cache_range_ctx = [0, 0, 0]
        _kv_cache_range_gen = [1, (max_len + 1) // 2, max_len]
        _max_len_range = [0, (max_len + 1) // 2, max_len]

        if max_num_tokens is None:
            num_tokens_range_ctx = [
                1, (max_input_len * max_batch_size + 1) // 2,
                max_input_len * max_batch_size
            ]
            num_tokens_range_gen = [
                1, max_batch_size * max_beam_width,
                max_beam_width * max_batch_size
            ]
        else:
            num_tokens_range_ctx = [[
                1, (max_num_tokens + 1) // 2, max_num_tokens
            ]]
            num_tokens_range_gen = [[
                1, (max_num_tokens + 1) // 2, max_num_tokens
            ]]

        enable_two_optimization_profiles = False
        if use_gpt_attention_plugin == False or use_gemm_plugin == False:
            use_in_flight_batching = use_gpt_attention_plugin and remove_input_padding and paged_kv_cache
            enable_two_optimization_profiles = not use_in_flight_batching
        if enable_two_optimization_profiles:
            bb_range = [bb_range_cxt, bb_range_gen]
            bs_range = [_bs_range, _bs_range]
            beam_width_range = [_beam_width_range, _beam_width_range]
            inlen_range = [inlen_range_cxt, inlen_range_gen]
            mask_len_range = [_mask_len_ctx, _mask_len_gen]
            if use_gpt_attention_plugin:
                kv_cache_range = [_kv_cache_range_gen, _kv_cache_range_gen]
            else:
                kv_cache_range = [_kv_cache_range_ctx, _kv_cache_range_gen]
            max_len_range = [_max_len_range, _max_len_range]
            num_tokens_range = [num_tokens_range_ctx, num_tokens_range_gen]
        else:
            bb_range = [bb_range_gen]
            bs_range = [_bs_range]
            beam_width_range = [_beam_width_range]
            inlen_range = [[1, 1, max_input_len]]
            mask_len_range = [[1, (max_len + 1) // 2 + 1, max_len + 1]]
            kv_cache_range = [[0, (max_len + 1) // 2, max_len]]
            max_len_range = [_max_len_range]
            if max_num_tokens is None:
                num_tokens_range = [[
                    1, max_batch_size * max_beam_width,
                    max(max_input_len * max_batch_size,
                        max_beam_width * max_batch_size)
                ]]
            else:
                num_tokens_range = num_tokens_range_ctx

        input_ids = None
        position_ids = None
        hidden_states = None
        if remove_input_padding:
            if mapping.is_first_pp_rank():
                input_ids = Tensor(
                    name='input_ids',
                    dtype=trt.int32,
                    shape=[1, -1],
                    dim_range=OrderedDict([
                        ('batch_size_fake',
                         [1, 1] if enable_two_optimization_profiles else [1]),
                        ('num_tokens', num_tokens_range),
                    ]))
                position_ids = Tensor(
                    name='position_ids',
                    dtype=trt.int32,
                    shape=[1, -1],
                    dim_range=OrderedDict([
                        ('batch_size_fake',
                         [1, 1] if enable_two_optimization_profiles else [1]),
                        ('num_tokens', num_tokens_range),
                    ]))
            else:
                assert dtype is not None
                assert num_heads is not None
                hidden_states = Tensor(
                    name='hidden_states_input',
                    dtype=dtype,
                    shape=[1, -1, head_size * num_heads],
                    dim_range=OrderedDict([
                        ('batch_size_fake',
                         [1, 1] if enable_two_optimization_profiles else [1]),
                        ('num_tokens', num_tokens_range),
                        ('hidden_size',
                         [head_size * num_heads, head_size *
                          num_heads] if enable_two_optimization_profiles else
                         [head_size * num_heads]),
                    ]))

        else:
            if mapping.is_first_pp_rank():
                input_ids = Tensor(name='input_ids',
                                   dtype=trt.int32,
                                   shape=[-1, -1],
                                   dim_range=OrderedDict([
                                       ('batch_size_beam_width', bb_range),
                                       ('input_len', inlen_range),
                                   ]))
                position_ids = Tensor(name='position_ids',
                                      dtype=trt.int32,
                                      shape=[-1, -1],
                                      dim_range=OrderedDict([
                                          ('batch_size_beam_width', bb_range),
                                          ('input_len', inlen_range),
                                      ]))
            else:
                assert dtype is not None
                assert num_heads is not None
                hidden_states = Tensor(
                    name='hidden_states_input',
                    dtype=dtype,
                    shape=[-1, -1, head_size * num_heads],
                    dim_range=OrderedDict([
                        ('batch_size_beam_width', bb_range),
                        ('input_len', inlen_range),
                        ('hidden_size',
                         [head_size * num_heads, head_size *
                          num_heads] if enable_two_optimization_profiles else
                         [head_size * num_heads]),
                    ]))

        num_kv_heads = (num_kv_heads + mapping.tp_size - 1) // mapping.tp_size
        layers_range = self.get_transformer_layers(mapping, num_layers)
        past_key_value = []
        kv_cache_block_pointers_list = []
        if not paged_kv_cache:
            for i in layers_range:
                kv_dim_range = OrderedDict([
                    ('batch_size_beam_width', bb_range),
                    ('kv', [2, 2] if enable_two_optimization_profiles else [2]),
                    ('num_heads', [num_kv_heads, num_kv_heads]
                     if enable_two_optimization_profiles else [num_kv_heads]),
                    ('past_key_len', kv_cache_range),
                    ('head_size', [head_size, head_size]
                     if enable_two_optimization_profiles else [head_size]),
                ])
                kv = Tensor(name=f'past_key_value_{i}',
                            dtype=kv_dtype,
                            shape=[-1, 2, num_kv_heads, -1, head_size],
                            dim_range=kv_dim_range)
                past_key_value.append(kv)

                kv_cache_block_pointers_list.append(None)
        else:
            if enable_two_optimization_profiles:
                max_blocks_per_seq_range = [
                    [
                        math.ceil(kv_cache_range[0][0] / tokens_per_block),
                        math.ceil(kv_cache_range[0][1] / tokens_per_block),
                        math.ceil(kv_cache_range[0][2] / tokens_per_block)
                    ],
                    [
                        math.ceil(kv_cache_range[1][0] / tokens_per_block),
                        math.ceil(kv_cache_range[1][1] / tokens_per_block),
                        math.ceil(kv_cache_range[1][2] / tokens_per_block)
                    ]
                ]
                blocks_range = [
                    [
                        bb_range[0][0] * max_blocks_per_seq_range[0][0],
                        bb_range[0][1] * max_blocks_per_seq_range[0][1],
                        bb_range[0][2] * max_blocks_per_seq_range[0][2]
                    ],
                    [
                        bb_range[1][0] * max_blocks_per_seq_range[1][0],
                        bb_range[1][1] * max_blocks_per_seq_range[1][1],
                        bb_range[1][2] * max_blocks_per_seq_range[1][2]
                    ],
                ]

                max_blocks_per_seq_range = [[
                    x for x in max_blocks_per_seq_range[0]
                ], [x for x in max_blocks_per_seq_range[1]]]
            else:
                max_blocks_per_seq_range = [[
                    math.ceil(kv_cache_range[0][0] / tokens_per_block),
                    math.ceil(kv_cache_range[0][1] / tokens_per_block),
                    math.ceil(kv_cache_range[0][2] / tokens_per_block)
                ]]
                blocks_range = [[
                    bb_range[0][0] * max_blocks_per_seq_range[0][0],
                    bb_range[0][1] * max_blocks_per_seq_range[0][1],
                    bb_range[0][2] * max_blocks_per_seq_range[0][2]
                ]]

                max_blocks_per_seq_range = [[
                    x for x in max_blocks_per_seq_range[0]
                ]]

            kv_dim_range = OrderedDict([
                ('blocks', blocks_range),
                ('kv', [2, 2] if enable_two_optimization_profiles else [2]),
                ('num_heads', [num_kv_heads, num_kv_heads]
                 if enable_two_optimization_profiles else [num_kv_heads]),
                ('tokens_per_block', [tokens_per_block, tokens_per_block]
                 if enable_two_optimization_profiles else [tokens_per_block]),
                ('head_size', [head_size, head_size]
                 if enable_two_optimization_profiles else [head_size]),
            ])
            for i in layers_range:
                kv_cache_block_pointers = Tensor(
                    name=f'kv_cache_block_pointers_{i}',
                    dtype=trt.int64,
                    shape=[-1, 2, -1],
                    dim_range=OrderedDict([
                        ('batch_size_beam_width', bb_range),
                        ('kv',
                         [2, 2] if enable_two_optimization_profiles else [2]),
                        ('max_blocks_per_seq', max_blocks_per_seq_range),
                    ]))
                kv_cache_block_pointers_list.append(kv_cache_block_pointers)
                past_key_value.append(None)

        sequence_length = None
        context_lengths = None
        host_context_lengths = None
        host_past_key_value_lengths = None
        attention_mask = None
        cache_indirection = None
        host_request_types = None

        if use_gpt_attention_plugin:
            sequence_length = Tensor(
                name='sequence_length',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size_beam_width', bb_range)]),
            )

            host_request_types = Tensor(
                name='host_request_types',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size_beam_width', bb_range)]),
            )
            host_past_key_value_lengths = Tensor(
                name='host_past_key_value_lengths',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size_beam_width', bb_range)]),
            )
            context_lengths = Tensor(
                name='context_lengths',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size_beam_width', bb_range)]),
            )
        else:
            attention_mask = Tensor(
                name='attention_mask',
                dtype=trt.int32,
                shape=[-1, -1],
                dim_range=OrderedDict([
                    ('batch_size_beam_width', bb_range),
                    ('mask_len', mask_len_range),
                ]),
            )

        if use_gpt_attention_plugin and remove_input_padding:
            host_context_lengths = Tensor(
                name='host_context_lengths',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size_beam_width', bb_range)]),
            )

        last_token_ids = None
        if mapping.is_last_pp_rank() and not gather_all_token_logits:
            last_token_ids = Tensor(
                name='last_token_ids',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([
                    ('batch_size_last_token_ids', bb_range),
                ]),
            )

        cache_indirection = Tensor(
            name='cache_indirection',
            dtype=trt.int32,
            shape=[-1, -1, -1],
            dim_range=OrderedDict([
                ('batch_size_cache', bs_range),
                ('beam_width', beam_width_range),
                ('max_seq_len', max_len_range),
            ]),
        )

        all_reduce_workspace = None
        if use_custom_all_reduce and mapping.tp_size > 1:
            # 3 (= buffer + signals_in + signals_out)
            workspace_size = 3 * mapping.tp_size
            all_reduce_workspace = Tensor(
                name='all_reduce_workspace',
                dtype=trt.int64,
                shape=[workspace_size],
                dim_range=OrderedDict([
                    ('all_reduce_size', [workspace_size, workspace_size]
                     if enable_two_optimization_profiles else [workspace_size])
                ]))

        return {
            'input_ids': input_ids,
            'hidden_states_input': hidden_states,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'sequence_length': sequence_length,
            'host_past_key_value_lengths': host_past_key_value_lengths,
            'past_key_value': past_key_value,
            'last_token_ids': last_token_ids,
            'cache_indirection': cache_indirection,
            'kv_cache_block_pointers_list': kv_cache_block_pointers_list,
            'context_lengths': context_lengths,
            'host_context_lengths': host_context_lengths,
            'host_request_types': host_request_types,
            'all_reduce_workspace': all_reduce_workspace,
        }
