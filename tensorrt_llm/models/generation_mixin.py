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
from collections import OrderedDict
from typing import List

import tensorrt as trt

from tensorrt_llm.plugin.plugin import current_all_reduce_helper

from ..functional import Tensor
from ..logger import logger
from ..mapping import Mapping


class GenerationMixin:

    @staticmethod
    def has_two_optimization_profiles(use_gpt_attention_plugin: bool,
                                      use_gemm_plugin: bool,
                                      remove_input_padding: bool,
                                      paged_kv_cache: bool) -> bool:
        res = False
        if use_gpt_attention_plugin == False or use_gemm_plugin == False:
            use_in_flight_batching = use_gpt_attention_plugin and remove_input_padding and paged_kv_cache
            res = not use_in_flight_batching
        return res

    @staticmethod
    def default_range(max_range, offset=0):
        result = [1, (max_range + 1) // 2, max_range]
        return [elem + offset for elem in result]

    def prepare_attention_inputs(self,
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
                                 paged_kv_cache=False,
                                 tokens_per_block=64,
                                 mapping=Mapping(),
                                 use_cache=True):

        max_len = max_input_len + max_new_tokens

        default_range = GenerationMixin.default_range
        bb_range_cxt = default_range(max_batch_size)
        bb_range_gen = default_range(max_batch_size * max_beam_width)
        _bs_range = default_range(max_batch_size)
        _beam_width_range = default_range(max_beam_width)
        _max_len_range = default_range(max_len)
        _mask_len_ctx = default_range(max_input_len)
        _mask_len_gen = default_range(max_len, 1)
        _kv_cache_range_ctx = [0, 0, 0]
        _kv_cache_range_gen = default_range(max_len)

        enable_two_optimization_profiles = GenerationMixin.has_two_optimization_profiles(
            use_gpt_attention_plugin, use_gemm_plugin, remove_input_padding,
            paged_kv_cache)
        if enable_two_optimization_profiles:
            bb_range = [bb_range_cxt, bb_range_gen]
            bs_range = [_bs_range, _bs_range]
            beam_width_range = [_beam_width_range, _beam_width_range]
            max_len_range = [_max_len_range, _max_len_range]
            mask_len_range = [_mask_len_ctx, _mask_len_gen]
            if use_gpt_attention_plugin:
                kv_cache_range = [_kv_cache_range_gen, _kv_cache_range_gen]
            else:
                kv_cache_range = [_kv_cache_range_ctx, _kv_cache_range_gen]
        else:
            bb_range = [bb_range_gen]
            bs_range = [_bs_range]
            beam_width_range = [_beam_width_range]
            max_len_range = [_max_len_range]
            mask_len_range = [_mask_len_gen]
            kv_cache_range = [_kv_cache_range_gen]

        num_kv_heads = (num_kv_heads + mapping.tp_size - 1) // mapping.tp_size
        layers_range = mapping.pp_layers(num_layers)
        past_key_value = []
        kv_cache_block_pointers_list = []
        host_kv_cache_block_pointers_list = []
        if use_cache:
            if not paged_kv_cache:
                for i in layers_range:
                    kv_dim_range = OrderedDict([
                        ('batch_size_beam_width', bb_range),
                        ('kv',
                         [2, 2] if enable_two_optimization_profiles else [2]),
                        ('num_heads', [num_kv_heads, num_kv_heads] if
                         enable_two_optimization_profiles else [num_kv_heads]),
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
                    host_kv_cache_block_pointers_list.append(None)
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
                    max_blocks_per_seq_range = [[
                        x for x in max_blocks_per_seq_range[0]
                    ], [x for x in max_blocks_per_seq_range[1]]]
                else:
                    max_blocks_per_seq_range = [[
                        math.ceil(kv_cache_range[0][0] / tokens_per_block),
                        math.ceil(kv_cache_range[0][1] / tokens_per_block),
                        math.ceil(kv_cache_range[0][2] / tokens_per_block)
                    ]]
                    max_blocks_per_seq_range = [[
                        x for x in max_blocks_per_seq_range[0]
                    ]]

                for i in layers_range:
                    kv_cache_block_pointers = Tensor(
                        name=f'kv_cache_block_pointers_{i}',
                        dtype=trt.int64,
                        shape=[-1, 2, -1],
                        dim_range=OrderedDict([
                            ('batch_size_beam_width', bb_range),
                            ('kv', [2, 2]
                             if enable_two_optimization_profiles else [2]),
                            ('max_blocks_per_seq', max_blocks_per_seq_range),
                        ]))
                    kv_cache_block_pointers_list.append(kv_cache_block_pointers)
                    host_kv_cache_block_pointers = Tensor(
                        name=f'host_kv_cache_block_pointers_{i}',
                        dtype=trt.int64,
                        shape=[-1, 2, -1],
                        dim_range=OrderedDict([
                            ('batch_size_beam_width', bb_range),
                            ('kv', [2, 2]
                             if enable_two_optimization_profiles else [2]),
                            ('max_blocks_per_seq', max_blocks_per_seq_range),
                        ]))
                    host_kv_cache_block_pointers_list.append(
                        host_kv_cache_block_pointers)
                    past_key_value.append(None)

        sequence_length = None
        context_lengths = None
        host_context_lengths = None
        host_past_key_value_lengths = None
        host_max_attention_window_sizes = None
        host_sink_token_length = None
        attention_mask = None
        cache_indirection = None
        host_request_types = None

        if use_gpt_attention_plugin:
            if use_cache:
                sequence_length = Tensor(
                    name='sequence_length',
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict([('batch_size_beam_width', bb_range)
                                           ]),
                )

            host_request_types = Tensor(
                name='host_request_types',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size_beam_width', bb_range)]),
            )
            if use_cache:
                host_past_key_value_lengths = Tensor(
                    name='host_past_key_value_lengths',
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict([('batch_size_beam_width', bb_range)
                                           ]),
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

        if use_gpt_attention_plugin:
            host_max_attention_window_sizes = []
            for i in layers_range:
                host_max_attention_window_tensor = Tensor(
                    name=f'host_max_attention_window_size_{i}',
                    dtype=trt.int32,
                    shape=[1],
                    dim_range=OrderedDict([
                        ('scalar',
                         [1, 1] if enable_two_optimization_profiles else [1])
                    ]))
                host_max_attention_window_sizes.append(
                    host_max_attention_window_tensor)

            host_sink_token_length = Tensor(
                name=f'host_sink_token_length',
                dtype=trt.int32,
                shape=[1],
                dim_range=OrderedDict([
                    ('scalar',
                     [1, 1] if enable_two_optimization_profiles else [1])
                ]))

        if use_cache:
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

        return {
            'attention_mask': attention_mask,
            'sequence_length': sequence_length,
            'host_past_key_value_lengths': host_past_key_value_lengths,
            'host_max_attention_window_sizes': host_max_attention_window_sizes,
            'host_sink_token_length': host_sink_token_length,
            'past_key_value': past_key_value,
            'cache_indirection': cache_indirection,
            'kv_cache_block_pointers_list': kv_cache_block_pointers_list,
            'host_kv_cache_block_pointers_list':
            host_kv_cache_block_pointers_list,
            'context_lengths': context_lengths,
            'host_context_lengths': host_context_lengths,
            'host_request_types': host_request_types,
        }

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
                             gather_context_logits=False,
                             gather_generation_logits=False,
                             dtype=None,
                             num_heads=None,
                             mapping=Mapping(),
                             max_num_tokens=None,
                             prompt_embedding_table_size: int = 0,
                             position_encoding_2d=False,
                             use_lora_plugin: bool = False,
                             lora_target_modules: List[str] = None,
                             max_draft_len=0):

        default_range = GenerationMixin.default_range
        last_token_range = [1, max_draft_len + 1, max_draft_len + 1]
        bb_range_cxt = default_range(max_batch_size)
        bb_range_gen = default_range(max_batch_size * max_beam_width)
        bbd_range_ctx = [
            bb_range_cxt[i] * ((max_draft_len + 1) if i != 0 else 1)
            for i in range(len(bb_range_cxt))
        ]
        bbd_range_gen = [
            bb_range_gen[i] * ((max_draft_len + 1) if i != 0 else 1)
            for i in range(len(bb_range_gen))
        ]
        inlen_range_cxt = default_range(max_input_len)
        inlen_range_gen = [1, 1, max_draft_len + 1]

        if max_num_tokens is None:
            default_max_num_tokens = 4096
            logger.warning(
                "max_num_tokens is not set, will choose a smaller "
                f"value between max_input_len * max_batch_size ({max_input_len * max_batch_size}) "
                f"and default_max_num_tokens ({default_max_num_tokens}).")
            max_num_tokens = min(max_input_len * max_batch_size,
                                 default_max_num_tokens)
        if max_num_tokens < max_input_len:
            logger.warning(
                f"max_num_tokens ({max_num_tokens}) should be greater than "
                f"max_input_len ({max_input_len}), specifying to "
                f"max_input_len ({max_input_len}).")
            max_num_tokens = max_input_len

        enable_two_optimization_profiles = GenerationMixin.has_two_optimization_profiles(
            use_gpt_attention_plugin, use_gemm_plugin, remove_input_padding,
            paged_kv_cache)
        if enable_two_optimization_profiles:
            bb_range = [bb_range_cxt, bb_range_gen]
            bbd_range = [bbd_range_ctx, bbd_range_gen]
            inlen_range = [inlen_range_cxt, inlen_range_gen]
            position_ids_inlen_range = [inlen_range_cxt, [1, 1, 1]]
            num_tokens_range_ctx = default_range(max_num_tokens)
            num_tokens_range_gen = default_range(max_batch_size *
                                                 max_beam_width)
            num_tokens_range = [num_tokens_range_ctx, num_tokens_range_gen]
            last_token_range = [last_token_range, last_token_range]
        else:
            bb_range = [bb_range_gen]
            bbd_range = [bbd_range_gen]
            last_token_range = [last_token_range]
            inlen_range = [[1, 1, max_input_len]]
            position_ids_inlen_range = [[1, 1, max_input_len]]
            num_tokens_range = [default_range(max_num_tokens)]

        input_ids = None
        position_ids = None
        hidden_states = None
        if remove_input_padding:
            if mapping.is_first_pp_rank():
                input_ids = Tensor(name='input_ids',
                                   dtype=trt.int32,
                                   shape=[-1],
                                   dim_range=OrderedDict([
                                       ('num_tokens', num_tokens_range),
                                   ]))
                if position_encoding_2d:
                    position_ids = Tensor(
                        name='position_ids',
                        dtype=trt.int32,
                        shape=[2, -1],
                        dim_range=OrderedDict([
                            ('2', [2, 2]
                             if enable_two_optimization_profiles else [2]),
                            ('num_tokens', num_tokens_range),
                        ]),
                    )
                else:
                    position_ids = Tensor(
                        name='position_ids',
                        dtype=trt.int32,
                        shape=[-1],
                        dim_range=OrderedDict([
                            ('num_tokens', num_tokens_range),
                        ]),
                    )
            else:
                assert dtype is not None
                assert num_heads is not None
                hidden_states = Tensor(
                    name='hidden_states_input',
                    dtype=dtype,
                    shape=[-1, head_size * num_heads],
                    dim_range=OrderedDict([
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
                if position_encoding_2d:
                    position_ids = Tensor(
                        name='position_ids',
                        dtype=trt.int32,
                        shape=[-1, 2, -1],
                        dim_range=OrderedDict([
                            ('batch_size_beam_width', bb_range),
                            ('2', [2, 2]
                             if enable_two_optimization_profiles else [2]),
                            ('position_ids_inlen_range',
                             position_ids_inlen_range),
                        ]),
                    )
                else:
                    position_ids = Tensor(
                        name='position_ids',
                        dtype=trt.int32,
                        shape=[-1, -1],
                        dim_range=OrderedDict([
                            ('batch_size_beam_width', bb_range),
                            ('position_ids_inlen_range',
                             position_ids_inlen_range),
                        ]),
                    )
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

        if use_custom_all_reduce and mapping.tp_size > 1:
            current_all_reduce_helper().set_workspace_tensor(
                mapping, enable_two_optimization_profiles)

        prompt_embedding_table = None
        tasks = None
        prompt_vocab_size = None
        if prompt_embedding_table_size > 0:
            assert num_heads is not None
            hidden_size = num_heads * head_size
            _p_embedding_range = [
                1, prompt_embedding_table_size // 2, prompt_embedding_table_size
            ]
            if enable_two_optimization_profiles:
                p_embedding_range = [_p_embedding_range, _p_embedding_range]
            else:
                p_embedding_range = [_p_embedding_range]

            prompt_embedding_table = Tensor(
                name='prompt_embedding_table',
                dtype=dtype,
                shape=[-1, hidden_size],
                dim_range=OrderedDict([
                    ('prompt_embedding_table_size', p_embedding_range),
                    ('hidden_size', [hidden_size, hidden_size]
                     if enable_two_optimization_profiles else [hidden_size]),
                ]))
            if remove_input_padding:
                tasks = Tensor(name='tasks',
                               dtype=trt.int32,
                               shape=[-1],
                               dim_range=OrderedDict([
                                   ('input_len_task', num_tokens_range),
                               ]))
            else:
                tasks = Tensor(
                    name='tasks',
                    dtype=trt.int32,
                    shape=[-1, 1],
                    dim_range=OrderedDict([
                        ('batch_size_beam_width', bb_range),
                        ('broadcast_dim',
                         [1, 1] if enable_two_optimization_profiles else [1]),
                    ]))
            prompt_vocab_size = Tensor(
                name='prompt_vocab_size',
                dtype=trt.int32,
                shape=[1],
                dim_range=OrderedDict([
                    ('size',
                     [1, 1] if enable_two_optimization_profiles else [1])
                ]))

        lora_weights_pointers = None
        lora_ranks = None
        if use_lora_plugin:
            lora_weights_pointers = []
            lora_ranks = []
            layers_range = mapping.pp_layers(num_layers)
            for i in layers_range:
                lora_weight_pointer_dict = {}
                lora_rank_dict = {}
                for lora_module in lora_target_modules:

                    lora_weight_pointer = Tensor(
                        name=f'{lora_module}_lora_weights_pointers_{i}',
                        dtype=trt.int64,
                        shape=[-1, 2],
                        dim_range=OrderedDict([
                            ('batch_size_beam_width', bb_range),
                            ('in_out', [2, 2]
                             if enable_two_optimization_profiles else [2]),
                        ]))
                    lora_weight_pointer_dict.update({
                        f"{lora_module}_lora_weights_pointers":
                        lora_weight_pointer
                    })

                    lora_rank = Tensor(
                        name=f'{lora_module}_lora_ranks_{i}',
                        dtype=trt.int32,
                        shape=[-1],
                        dim_range=OrderedDict([('batch_size_beam_width',
                                                bb_range)]),
                    )
                    lora_rank_dict.update(
                        {f"{lora_module}_lora_ranks": lora_rank})

                lora_weights_pointers.append(lora_weight_pointer_dict)
                lora_ranks.append(lora_rank_dict)

        last_token_ids = None
        if mapping.is_last_pp_rank() and not gather_context_logits:
            if not remove_input_padding and max_draft_len > 0:
                last_token_ids = Tensor(
                    name='last_token_ids',
                    dtype=trt.int32,
                    shape=[-1, -1],
                    dim_range=OrderedDict([
                        ('batch_size_beam_width', bb_range),
                        ('last_token_ids', last_token_range),
                    ]),
                )
            else:
                last_token_ids = Tensor(
                    name='last_token_ids',
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict([
                        ('batch_size_last_token_ids', bbd_range),
                    ]),
                )

        basic_inputs = {
            'input_ids': input_ids,
            'hidden_states_input': hidden_states,
            'position_ids': position_ids,
            'last_token_ids': last_token_ids,
            'prompt_embedding_table': prompt_embedding_table,
            'tasks': tasks,
            'prompt_vocab_size': prompt_vocab_size,
            'lora_ranks': lora_ranks,
            'lora_weights_pointers': lora_weights_pointers,
        }

        attention_inputs = self.prepare_attention_inputs(
            max_batch_size,
            max_beam_width,
            max_input_len,
            max_new_tokens,
            num_kv_heads,
            head_size,
            num_layers,
            kv_dtype,
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            mapping=mapping)

        for key, value in attention_inputs.items():
            basic_inputs[key] = value

        return basic_inputs
