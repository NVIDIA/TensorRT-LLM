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

from ..functional import Tensor, assertion, shape


class GenerationMixin:

    def prepare_basic_inputs(self,
                             max_batch_size,
                             max_beam_width,
                             max_input_len,
                             max_new_tokens,
                             num_heads,
                             head_size,
                             num_layers,
                             kv_dtype,
                             remove_input_padding=False,
                             use_gpt_attention_plugin=False,
                             paged_kv_cache=False,
                             tokens_per_block=64):

        max_len = max_input_len + max_new_tokens
        bb_range = [
            1, (max_batch_size * max_beam_width + 1) // 2,
            max_batch_size * max_beam_width
        ]
        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        beam_width_range = [1, (max_beam_width + 1) // 2, max_beam_width]
        inlen_range = [1, 1, max_input_len]
        mask_len_range = [1, (max_len + 1) // 2 + 1, max_len + 1]
        max_len_range = [0, (max_len + 1) // 2, max_len]
        num_tokens_range = [
            1, max_batch_size * max_beam_width,
            max(max_input_len * max_batch_size, max_beam_width * max_batch_size)
        ]

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
                                   ('batch_size_beam_width', [bb_range]),
                                   ('input_len', [inlen_range]),
                               ]))
            position_ids = Tensor(name='position_ids',
                                  dtype=trt.int32,
                                  shape=[-1, -1],
                                  dim_range=OrderedDict([
                                      ('batch_size_beam_width', [bb_range]),
                                      ('input_len', [inlen_range]),
                                  ]))

        past_key_value = []
        kv_cache_block_pointers_list = []
        if not paged_kv_cache:
            for i in range(num_layers):
                kv_dim_range = OrderedDict([
                    ('kv_batch_size_beam_width', [bb_range]),
                    ('kv', [2]),
                    ('num_heads', [num_heads]),
                    ('past_key_len', [max_len_range]),
                    ('head_size', [head_size]),
                ])
                kv = Tensor(name=f'past_key_value_{i}',
                            dtype=kv_dtype,
                            shape=[-1, 2, num_heads, -1, head_size],
                            dim_range=kv_dim_range)
                past_key_value.append(kv)
                # TODO(kaiyu): Remove this when TRT fix the named dimension
                if not remove_input_padding:
                    assertion(shape(input_ids, 0) == shape(kv, 0), 'batch size')

                kv_cache_block_pointers_list.append(None)
        else:
            max_blocks_per_seq_range = [
                math.ceil(max_len_range[0] / tokens_per_block),
                math.ceil(max_len_range[1] / tokens_per_block),
                math.ceil(max_len_range[2] / tokens_per_block)
            ]
            blocks_range = [
                bb_range[0] * max_blocks_per_seq_range[0],
                bb_range[1] * max_blocks_per_seq_range[1],
                bb_range[2] * max_blocks_per_seq_range[2]
            ]
            # NOTE(nkorobov): we multiply max_blocks_per_seq by 2 because plugin expects pointers as int64,
            # but TRT does not support int64. Thus, we emulate int64 with doubled int32.
            max_blocks_per_seq_range = [2 * x for x in max_blocks_per_seq_range]

            kv_dim_range = OrderedDict([
                ('blocks', [blocks_range]),
                ('kv', [2]),
                ('num_heads', [num_heads]),
                ('tokens_per_block', [tokens_per_block]),
                ('head_size', [head_size]),
            ])
            for i in range(self._num_layers):
                # (blocks, 2, kv_num_heads, tokens_per_block, head_size)
                kv = Tensor(
                    name=f'past_key_value_{i}',
                    dtype=self._kv_dtype,
                    shape=[-1, 2, num_heads, tokens_per_block, head_size],
                    dim_range=kv_dim_range)
                past_key_value.append(kv)

                kv_cache_block_pointers = Tensor(
                    name=f'kv_cache_block_pointers_{i}',
                    dtype=trt.int32,
                    shape=[-1, 2, -1],
                    dim_range=OrderedDict([
                        ('batch_size_beam_width', [bb_range]),
                        ('kv', [2]),
                        ('max_blocks_per_seq', [max_blocks_per_seq_range]),
                    ]))
                kv_cache_block_pointers_list.append(kv_cache_block_pointers)

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
                dim_range=OrderedDict([('batch_size_beam_width', [bb_range])]),
            )

            host_request_types = Tensor(
                name='host_request_types',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size_beam_width', [bb_range])]),
            )
            host_past_key_value_lengths = Tensor(
                name='host_past_key_value_lengths',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size_beam_width', [bb_range])]),
            )
            context_lengths = Tensor(
                name='context_lengths',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size_beam_width', [bb_range])]),
            )
        else:
            attention_mask = Tensor(
                name='attention_mask',
                dtype=trt.int32,
                shape=[-1, -1],
                dim_range=OrderedDict([
                    ('batch_size_beam_width', [bb_range]),
                    ('mask_len', [mask_len_range]),
                ]),
            )

        if use_gpt_attention_plugin and remove_input_padding:
            host_context_lengths = Tensor(
                name='host_context_lengths',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size_beam_width', [bb_range])]),
            )

        last_token_ids = Tensor(
            name='last_token_ids',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([
                ('batch_size_last_token_ids', [bb_range]),
            ]),
        )

        cache_indirection = Tensor(
            name='cache_indirection',
            dtype=trt.int32,
            shape=[-1, -1, -1],
            dim_range=OrderedDict([
                ('batch_size_cache', [bs_range]),
                ('beam_width', [beam_width_range]),
                ('max_seq_len', [max_len_range]),
            ]),
        )

        return {
            'input_ids': input_ids,
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
            'host_request_types': host_request_types
        }
