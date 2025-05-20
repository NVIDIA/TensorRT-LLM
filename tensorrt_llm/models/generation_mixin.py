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
from typing import List, Optional

import tensorrt as trt

from ..bindings import KVCacheType
from ..functional import Tensor
from ..layers import MropeParams, SpecDecodingParams
from ..mapping import Mapping
from ..plugin import current_all_reduce_helper


class GenerationMixin:

    @staticmethod
    def has_ctx_gen_opt_profiles(
            use_gpt_attention_plugin: bool = False,
            use_gemm_plugin: bool = False,
            use_mamba_conv1d_plugin: bool = False,
            remove_input_padding: bool = False,
            paged_state: bool = False,
            kv_cache_type: KVCacheType = KVCacheType.CONTINUOUS) -> bool:
        res = False

        if not use_gpt_attention_plugin or not use_gemm_plugin:
            use_in_flight_batching = False
            # Refer to modelConfig.h: supportsInflightBatching(), this should be consistent its implementation.
            # We skip check transformer or rnn arch for simplification.
            if remove_input_padding and use_gpt_attention_plugin:
                use_in_flight_batching = kv_cache_type in [
                    KVCacheType.PAGED, KVCacheType.DISABLED
                ]
            elif remove_input_padding and use_mamba_conv1d_plugin:
                use_in_flight_batching = paged_state == True

            res = not use_in_flight_batching

        return res

    @staticmethod
    def default_range(max_range, offset=0, min_range=1, opt_offset=0):
        result = [
            min_range, (max_range + min_range + opt_offset) // 2, max_range
        ]
        return [elem + offset for elem in result]

    @staticmethod
    def split_num_tokens_range(max_num_tokens):
        split_point = [64, 128, 256, 512, 1024]
        num_tokens_ranges = []
        for i, p in enumerate(split_point):
            if i == 0 and max_num_tokens <= p:
                return [1, max_num_tokens, max_num_tokens]
            elif max_num_tokens <= p:
                num_tokens_ranges.append(
                    [split_point[i - 1], max_num_tokens, max_num_tokens])
                return num_tokens_ranges
            elif i == 0 and max_num_tokens > p:
                num_tokens_ranges = [[1, 64, 64]]
            else:
                num_tokens_ranges.append(
                    [split_point[i - 1], split_point[i], split_point[i]])
        num_tokens_ranges.append(
            [split_point[-1], max_num_tokens, max_num_tokens])
        return num_tokens_ranges

    @staticmethod
    def get_profiles_ranges(
            *,
            max_batch_size,
            max_beam_width,
            max_input_len,
            max_num_tokens,
            max_draft_len,
            opt_batch_size,
            opt_num_tokens,
            enable_ctx_gen_opt_profiles,
            multiple_profiles,
            kv_cache_type: KVCacheType = KVCacheType.CONTINUOUS):
        default_range = GenerationMixin.default_range
        if opt_batch_size:
            bb_range_cxt = [1, opt_batch_size, max_batch_size]
            bb_range_gen = [
                1, opt_batch_size * max_beam_width,
                max_batch_size * max_beam_width
            ]
        else:
            bb_range_cxt = default_range(max_batch_size)
            bb_range_gen = default_range(max_batch_size * max_beam_width)
        tokens_per_engine_step = max_draft_len + 1
        tokens_per_engine_step_range = [
            1, tokens_per_engine_step, tokens_per_engine_step
        ]
        bbd_range_ctx = [
            bb_range_cxt[i] * (tokens_per_engine_step if i != 0 else 1)
            for i in range(len(bb_range_cxt))
        ]
        bbd_range_gen = [
            bb_range_gen[i] * (tokens_per_engine_step if i != 0 else 1)
            for i in range(len(bb_range_gen))
        ]
        inlen_range_cxt = default_range(max_input_len)
        inlen_range_gen = [1, 1, tokens_per_engine_step]
        if enable_ctx_gen_opt_profiles:
            num_profiles = 2
            bb_range = [bb_range_cxt, bb_range_gen]
            bbd_range = [bbd_range_ctx, bbd_range_gen]
            inlen_range = [inlen_range_cxt, inlen_range_gen]
            position_ids_inlen_range = [inlen_range_cxt, [1, 1, 1]]
            num_tokens_range_ctx = default_range(max_batch_size * max_input_len)
            # Draft tokens cannot be combined with beam search
            num_tokens_range_gen = default_range(
                max_batch_size * max(tokens_per_engine_step, max_beam_width))
            num_tokens_range = [num_tokens_range_ctx, num_tokens_range_gen]

            # Only keep context range when kv cache is disabled.
            if kv_cache_type == KVCacheType.DISABLED:
                num_profiles = 1
                bb_range = [bb_range[0]]
                bbd_range = [bbd_range[0]]
                inlen_range = [inlen_range[0]]
                position_ids_inlen_range = [position_ids_inlen_range[0]]
                num_tokens_range_ctx = [num_tokens_range_ctx[0]]
                # Draft tokens cannot be combined with beam search
                num_tokens_range_gen = [num_tokens_range_gen[0]]
                num_tokens_range = [num_tokens_range[0]]

        else:
            if multiple_profiles:
                num_tokens_range = GenerationMixin.split_num_tokens_range(
                    max_num_tokens)
            else:
                if opt_num_tokens is None:
                    opt_num_tokens = min(max_num_tokens,
                                         max_batch_size * max_beam_width)
                num_tokens_range = [[1, opt_num_tokens, max_num_tokens]]
            num_profiles = len(num_tokens_range)
            bb_range = [bb_range_gen] * num_profiles
            bbd_range = [bbd_range_gen] * num_profiles
            inlen_range = [[1, 1, max_input_len]] * num_profiles
            position_ids_inlen_range = [[1, 1, max_input_len]] * num_profiles
        tokens_per_engine_step_range = [tokens_per_engine_step_range
                                        ] * num_profiles

        position_ids_num_tokens_range = num_tokens_range
        # If max_draft_len != 0, the input_ids may include draft tokens. And the length of position_ids may be not the same as input_ids.
        # In extreme cases, input_ids may contain (max_draft_token + 1) * N, and the actual position_ids value is only 1 * N.
        # Therefore, we need to adjust the min value in the ranges of position_ids.
        if max_draft_len != 0:
            position_ids_num_tokens_range = list(
                map(
                    lambda x:
                    [math.ceil(x[0] / (max_draft_len + 1)), x[1], x[2]],
                    num_tokens_range))

        ranges = {
            'bb_range': bb_range,
            'bbd_range': bbd_range,
            'inlen_range': inlen_range,
            'position_ids_inlen_range': position_ids_inlen_range,
            'num_tokens_range': num_tokens_range,
            'tokens_per_engine_step_range': tokens_per_engine_step_range,
            'position_ids_num_tokens_range': position_ids_num_tokens_range,
        }
        return num_profiles, ranges

    def prepare_attention_inputs(
            self,
            *,
            max_batch_size,
            max_beam_width,
            max_input_len,
            max_seq_len,
            num_kv_heads,
            head_size,
            num_layers,
            kv_dtype,
            kv_cache_type: KVCacheType,
            num_profiles=1,
            enable_ctx_gen_opt_profiles=False,
            remove_input_padding=False,
            use_gpt_attention_plugin=False,
            tokens_per_block=32,
            mapping=Mapping(),
            streamingllm=False,
            attn_layer_idx=None,
            opt_batch_size=None,
            num_kv_heads_per_layer: Optional[List[int]] = None):

        if attn_layer_idx is not None and num_kv_heads_per_layer is not None:
            assert len(attn_layer_idx) == len(num_kv_heads_per_layer), (
                f"Expected len(attn_layer_idx) ({len(attn_layer_idx)})"
                f" == len(num_kv_heads_per_layer) ({len(num_kv_heads_per_layer)})"
            )
        default_range = GenerationMixin.default_range

        if opt_batch_size:
            bb_range_cxt = [1, opt_batch_size, max_batch_size]
            bb_range_gen = [
                1, opt_batch_size * max_beam_width,
                max_batch_size * max_beam_width
            ]
        else:
            bb_range_cxt = default_range(max_batch_size)
            bb_range_gen = default_range(max_batch_size * max_beam_width)

        _bs_range = default_range(max_batch_size)
        _beam_width_range = default_range(max_beam_width)
        _max_len_range = default_range(max_seq_len)
        _mask_len_ctx = default_range(max_input_len)
        _kv_cache_range_ctx = [0, 0, 0]
        _kv_cache_range_gen = default_range(max_seq_len, -1)
        if kv_cache_type == KVCacheType.DISABLED:
            _kv_cache_range = default_range(max_seq_len)
        else:
            kv_max_seq_len = max_seq_len
            if streamingllm:
                # add the max bubble length
                kv_max_seq_len += tokens_per_block - 1
            if max_beam_width > 1:
                # support cyclic kv cache cases that use one more block
                kv_max_seq_len += tokens_per_block
            _kv_cache_range = default_range(kv_max_seq_len)

        if enable_ctx_gen_opt_profiles:
            if kv_cache_type != KVCacheType.DISABLED:
                assert num_profiles == 2
                bb_range = [bb_range_cxt, bb_range_gen]
                mask_len_range = [_mask_len_ctx, _max_len_range]
                if use_gpt_attention_plugin:
                    kv_cache_range = [_kv_cache_range, _kv_cache_range]
                else:
                    kv_cache_range = [_kv_cache_range_ctx, _kv_cache_range_gen]
            else:
                assert num_profiles == 1
                bb_range = [bb_range_cxt]
                mask_len_range = [_mask_len_ctx]
                if use_gpt_attention_plugin:
                    kv_cache_range = [_kv_cache_range]
                else:
                    kv_cache_range = [_kv_cache_range_ctx]

        else:
            bb_range = [bb_range_gen] * num_profiles
            mask_len_range = [_max_len_range] * num_profiles
            kv_cache_range = [_kv_cache_range] * num_profiles
        bs_range = [_bs_range] * num_profiles
        beam_width_range = [_beam_width_range] * num_profiles
        max_len_range = [_max_len_range] * num_profiles

        num_kv_heads = (num_kv_heads + mapping.tp_size - 1) // mapping.tp_size
        if num_kv_heads_per_layer is not None:
            num_kv_heads_per_layer = [
                (nheads + mapping.tp_size - 1) // mapping.tp_size
                for nheads in num_kv_heads_per_layer
            ]

        layers_range = mapping.pp_layers(num_layers)
        if attn_layer_idx is None:
            attn_layer_idx = [i for i in range(num_layers)]
        # layer indices of attention layers local to the current pp rank
        local_attn_layers = [i for i in layers_range if i in attn_layer_idx]
        # number of attention layers local to previous pp ranks
        num_attn_layers_lower_ranks = attn_layer_idx.index(local_attn_layers[0])
        num_attn_layers = len(local_attn_layers)
        num_layers_prev_rank = layers_range[
            0] // mapping.pp_rank if mapping.pp_rank != 0 else len(layers_range)
        past_key_value = []
        kv_cache_block_offsets = None
        host_kv_cache_block_offsets = None
        host_kv_cache_pool_pointers = None
        host_kv_cache_pool_mapping = None
        if kv_cache_type == KVCacheType.DISABLED:
            past_key_value = [None] * num_layers_prev_rank
        else:
            if kv_cache_type != KVCacheType.PAGED:
                for layer_idx in layers_range:
                    if layer_idx not in local_attn_layers:
                        # not an attention layer ==> give it None pkv input
                        past_key_value.append(None)
                        continue

                    attn_idx = local_attn_layers.index(layer_idx)
                    if num_kv_heads_per_layer is not None:
                        heads_dim_name = f"num_heads_{layer_idx}"
                        kv_heads = num_kv_heads_per_layer[
                            num_attn_layers_lower_ranks + attn_idx]
                    else:
                        heads_dim_name = "num_heads"
                        kv_heads = num_kv_heads

                    kv_dim_range = OrderedDict([
                        ('batch_size_beam_width', bb_range),
                        ('kv', [2] * num_profiles),
                        (heads_dim_name, [kv_heads] * num_profiles),
                        ('past_key_len', kv_cache_range),
                        ('head_size', [head_size] * num_profiles),
                    ])

                    kv = Tensor(name=f'past_key_value_{layer_idx}',
                                dtype=kv_dtype,
                                shape=[-1, 2, kv_heads, -1, head_size],
                                dim_range=kv_dim_range)
                    past_key_value.append(kv)
            else:
                if enable_ctx_gen_opt_profiles:
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
                else:
                    max_blocks_per_seq_range = [[
                        math.ceil(kv_cache_range[0][0] / tokens_per_block),
                        math.ceil(kv_cache_range[0][1] / tokens_per_block),
                        math.ceil(kv_cache_range[0][2] / tokens_per_block)
                    ]] * num_profiles

                NUM_KV_CACHE_POOLS = -1  # the number of unique variable window sizes, which is only known at runtime, affects the number of pools.
                # dim_range=(min=1, opt=1 (this is the usual case - non vgqa, non vsliding_window), max=num_layers,
                # TODO(nhaber): Benchmark if making NUM_KV_CACHE_POOLS dynamic has a significant performance hit?

                kv_pools_range = [[1, 1, len(local_attn_layers)]] * num_profiles
                kv_cache_block_offsets = Tensor(
                    name=f'kv_cache_block_offsets',
                    dtype=trt.int32,
                    shape=[NUM_KV_CACHE_POOLS, -1, 2, -1],
                    dim_range=OrderedDict([
                        ('num_kv_cache_pools', kv_pools_range),
                        ('batch_size_beam_width', bb_range),
                        ('kv', [2] * num_profiles),
                        ('max_blocks_per_seq', max_blocks_per_seq_range),
                    ]))
                host_kv_cache_block_offsets = Tensor(
                    name=f'host_kv_cache_block_offsets',
                    dtype=trt.int32,
                    shape=[NUM_KV_CACHE_POOLS, -1, 2, -1],
                    dim_range=OrderedDict([
                        ('num_kv_cache_pools', kv_pools_range),
                        ('batch_size_beam_width', bb_range),
                        ('kv', [2] * num_profiles),
                        ('max_blocks_per_seq', max_blocks_per_seq_range),
                    ]))
                host_kv_cache_pool_pointers = Tensor(
                    name=f'host_kv_cache_pool_pointers',
                    dtype=trt.int64,
                    shape=[NUM_KV_CACHE_POOLS, 2],
                    dim_range=OrderedDict([
                        ('num_pools_layers', kv_pools_range),
                        ('num_pools_kv', [2] * num_profiles),
                    ]))

                host_kv_cache_pool_mapping = Tensor(
                    name=f'host_kv_cache_pool_mapping',
                    dtype=trt.int32,
                    shape=[num_attn_layers,
                           2],  # 2: (Index of pool, Index of layer within pool)
                    dim_range=OrderedDict([
                        ('pools_mapping', [num_attn_layers] * num_profiles),
                        ('layer_cache_pool_locator', [2] * num_profiles)
                    ]))

                past_key_value = [None] * num_layers_prev_rank

        assert len(past_key_value) == num_layers_prev_rank
        sequence_length = None
        context_lengths = None
        host_context_lengths = None
        host_past_key_value_lengths = None
        host_max_attention_window_sizes = None
        host_sink_token_length = None
        attention_mask = None
        cache_indirection = None
        host_request_types = None
        runtime_perf_knobs = None
        context_progress = None

        if use_gpt_attention_plugin:
            if kv_cache_type != KVCacheType.DISABLED:
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
            if kv_cache_type != KVCacheType.DISABLED:
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
            runtime_perf_knobs = Tensor(name='host_runtime_perf_knobs',
                                        dtype=trt.int64,
                                        shape=[16],
                                        dim_range=OrderedDict([
                                            ('perf_knob_size',
                                             [16] * num_profiles)
                                        ]))
            context_progress = Tensor(name='host_context_progress',
                                      dtype=trt.int64,
                                      shape=[1],
                                      dim_range=OrderedDict([
                                          ('context_progress_size',
                                           [1] * num_profiles)
                                      ]))
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
            # TODO: change shape to [1]
            host_max_attention_window_sizes = Tensor(
                name=f'host_max_attention_window_sizes',
                dtype=trt.int32,
                shape=[num_attn_layers],
                dim_range=OrderedDict([('num_layers',
                                        [num_attn_layers] * num_profiles)]))

            host_sink_token_length = Tensor(name='host_sink_token_length',
                                            dtype=trt.int32,
                                            shape=[1],
                                            dim_range=OrderedDict([
                                                ('scalar', [1] * num_profiles)
                                            ]))

        if kv_cache_type != KVCacheType.DISABLED:
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
            'kv_cache_block_offsets': kv_cache_block_offsets,
            'host_kv_cache_block_offsets': host_kv_cache_block_offsets,
            'host_kv_cache_pool_pointers': host_kv_cache_pool_pointers,
            'host_kv_cache_pool_mapping': host_kv_cache_pool_mapping,
            'context_lengths': context_lengths,
            'host_context_lengths': host_context_lengths,
            'host_request_types': host_request_types,
            'host_runtime_perf_knobs': runtime_perf_knobs,
            'host_context_progress': context_progress,
        }

    def prepare_basic_inputs(
        self,
        *,
        max_batch_size,
        max_beam_width,
        max_input_len,
        max_seq_len,
        max_num_tokens,
        hidden_size,
        num_kv_heads,
        head_size,
        num_layers,
        kv_dtype,
        kv_cache_type: KVCacheType,
        remove_input_padding=False,
        use_gpt_attention_plugin=False,
        use_gemm_plugin=False,
        tokens_per_block=32,
        gather_context_logits=False,
        dtype=None,
        num_heads=None,
        mapping=Mapping(),
        opt_num_tokens=None,
        prompt_embedding_table_size: int = 0,
        position_encoding_2d=False,
        use_lora_plugin: bool = False,
        lora_target_modules: List[str] = None,
        speculative_decoding_draft_tokens_external: bool = False,
        spec_decoding_is_generation_length_variable: bool = False,
        max_draft_len=0,
        multiple_profiles: bool = False,
        streamingllm: bool = False,
        opt_batch_size=None,
        pp_reduce_scatter: bool = False,
        mrope_rotary_cos_sin_size: int = None,
    ):

        enable_ctx_gen_opt_profiles = GenerationMixin.has_ctx_gen_opt_profiles(
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            remove_input_padding=remove_input_padding,
            kv_cache_type=kv_cache_type)

        num_profiles, ranges = GenerationMixin.get_profiles_ranges(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_num_tokens=max_num_tokens,
            max_draft_len=max_draft_len,
            opt_batch_size=opt_batch_size,
            opt_num_tokens=opt_num_tokens,
            enable_ctx_gen_opt_profiles=enable_ctx_gen_opt_profiles,
            multiple_profiles=multiple_profiles,
            kv_cache_type=kv_cache_type)
        bb_range = ranges['bb_range']
        bbd_range = ranges['bbd_range']
        inlen_range = ranges['inlen_range']
        num_tokens_range = ranges['num_tokens_range']
        position_ids_inlen_range = ranges['position_ids_inlen_range']
        tokens_per_engine_step_range = ranges['tokens_per_engine_step_range']
        position_ids_num_tokens_range = ranges['position_ids_num_tokens_range']

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
                            ('2', [2] * num_profiles),
                            ('position_ids_num_tokens_range',
                             position_ids_num_tokens_range),
                        ]),
                    )
                else:
                    position_ids = Tensor(
                        name='position_ids',
                        dtype=trt.int32,
                        shape=[-1],
                        dim_range=OrderedDict([
                            ('position_ids_num_tokens_range',
                             position_ids_num_tokens_range),
                        ]),
                    )
            else:
                assert dtype is not None
                assert num_heads is not None
                pp_hidden_size = hidden_size // mapping.tp_size if pp_reduce_scatter else hidden_size
                hidden_states = Tensor(
                    name='hidden_states_input',
                    dtype=dtype,
                    shape=[-1, pp_hidden_size],
                    dim_range=OrderedDict([
                        ('num_tokens', num_tokens_range),
                        ('hidden_size', [pp_hidden_size] * num_profiles),
                    ]),
                )

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
                            ('2', [2] * num_profiles),
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
                    shape=[-1, -1, hidden_size],
                    dim_range=OrderedDict([
                        ('batch_size_beam_width', bb_range),
                        ('input_len', inlen_range),
                        ('hidden_size', [hidden_size] * num_profiles),
                    ]),
                )

        if mapping.tp_size > 1:
            current_all_reduce_helper().set_workspace_tensor(
                mapping, num_profiles)

        prompt_embedding_table = None
        tasks = None
        prompt_vocab_size = None
        if prompt_embedding_table_size > 0:
            _p_embedding_range = [
                1, prompt_embedding_table_size // 2, prompt_embedding_table_size
            ]
            p_embedding_range = [_p_embedding_range] * num_profiles

            prompt_embedding_table = Tensor(name='prompt_embedding_table',
                                            dtype=dtype,
                                            shape=[-1, hidden_size],
                                            dim_range=OrderedDict([
                                                ('prompt_embedding_table_size',
                                                 p_embedding_range),
                                                ('hidden_size',
                                                 [hidden_size] * num_profiles),
                                            ]))
            if remove_input_padding:
                tasks = Tensor(name='tasks',
                               dtype=trt.int32,
                               shape=[-1],
                               dim_range=OrderedDict([
                                   ('input_len_task', num_tokens_range),
                               ]))
            else:
                tasks = Tensor(name='tasks',
                               dtype=trt.int32,
                               shape=[-1, 1],
                               dim_range=OrderedDict([
                                   ('batch_size_beam_width', bb_range),
                                   ('broadcast_dim', [1] * num_profiles),
                               ]))
            prompt_vocab_size = Tensor(name='prompt_vocab_size',
                                       dtype=trt.int32,
                                       shape=[1],
                                       dim_range=OrderedDict([
                                           ('size', [1] * num_profiles)
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
                        shape=[-1, 3],
                        dim_range=OrderedDict([
                            ('batch_size_beam_width', bb_range),
                            ('in_out_scales', [3] * num_profiles),
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
                        ('last_token_ids', tokens_per_engine_step_range),
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

        spec_decoding_params = None
        # Use positional offsets and packed mask only when not in SpS spec decoding
        if speculative_decoding_draft_tokens_external == False and max_draft_len > 0:
            tokens_per_engine_step = max_draft_len + 1
            # 32 bits packed mask aligned.
            num_packed_masks = (tokens_per_engine_step + 32 - 1) // 32
            packed_mask_len_range = [[0, 1, num_packed_masks]] * num_profiles
            # total number of spec decoding tokens for all sequences (sequence length can be variable).
            num_gen_tokens_range = [
                GenerationMixin.default_range(
                    max_batch_size * max_beam_width * tokens_per_engine_step,
                    min_range=0)
            ] * num_profiles
            bb_range_0 = [[0] + bbr[1:] for bbr in bb_range]

            # support variable sequence lengths for medusa.
            spec_decoding_generation_lengths = Tensor(
                name='spec_decoding_generation_lengths',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size_beam_width_0', bb_range_0)
                                       ]),
            )

            # position offsets that are fixed during the whole session.
            # it will be shared among all sequences.
            spec_decoding_position_offsets = Tensor(
                name='spec_decoding_position_offsets',
                dtype=trt.int32,
                shape=[-1, -1],
                dim_range=OrderedDict([
                    ('batch_size_beam_width_0', bb_range_0),
                    ('spec_decoding_position_ids_dim0',
                     tokens_per_engine_step_range),
                ]),
            )

            spec_decoding_packed_mask = Tensor(
                name='spec_decoding_packed_mask',
                dtype=trt.int32,
                shape=[-1, -1],
                dim_range=OrderedDict([
                    ('spec_decoding_packed_mask_dim0', num_gen_tokens_range),
                    ('spec_decoding_packed_mask_dim1', packed_mask_len_range),
                ]),
            )
            spec_decoding_use = Tensor(name='spec_decoding_use',
                                       dtype=trt.int32,
                                       shape=[1],
                                       dim_range=OrderedDict([
                                           ('spec_decoding_use_dim',
                                            [1] * num_profiles),
                                       ]))
            spec_decoding_params = SpecDecodingParams(
                spec_decoding_is_generation_length_variable=
                spec_decoding_is_generation_length_variable,
                spec_decoding_max_generation_length=tokens_per_engine_step,
                spec_decoding_generation_lengths=
                spec_decoding_generation_lengths,
                spec_decoding_position_offsets=spec_decoding_position_offsets,
                spec_decoding_packed_mask=spec_decoding_packed_mask,
                spec_decoding_use=spec_decoding_use)

        mrope_params = None
        if mrope_rotary_cos_sin_size is not None:
            mrope_rotary_cos_sin = Tensor(
                name='mrope_rotary_cos_sin',
                dtype=trt.float32,
                shape=[-1, mrope_rotary_cos_sin_size],
                dim_range=OrderedDict([
                    ('batch_size_beam_width', bb_range),
                    ('mult_dim', [mrope_rotary_cos_sin_size] * num_profiles),
                ]),
            )

            mrope_position_deltas = Tensor(
                name='mrope_position_deltas',
                dtype=trt.int32,
                shape=[-1, 1],
                dim_range=OrderedDict([('batch_size_beam_width', bb_range),
                                       ('mult_dim_delta', [1] * num_profiles)]),
            )
            mrope_params = MropeParams(
                mrope_rotary_cos_sin=mrope_rotary_cos_sin,
                mrope_position_deltas=mrope_position_deltas,
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
            'spec_decoding_params': spec_decoding_params,
            'mrope_params': mrope_params,
        }

        attention_inputs = self.prepare_attention_inputs(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            num_layers=num_layers,
            kv_dtype=kv_dtype,
            num_profiles=num_profiles,
            enable_ctx_gen_opt_profiles=enable_ctx_gen_opt_profiles,
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            kv_cache_type=kv_cache_type,
            tokens_per_block=tokens_per_block,
            mapping=mapping,
            streamingllm=streamingllm,
            opt_batch_size=opt_batch_size)
        for key, value in attention_inputs.items():
            basic_inputs[key] = value

        return basic_inputs
