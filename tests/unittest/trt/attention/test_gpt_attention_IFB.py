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
import unittest
from itertools import product

import numpy as np
import pytest

# isort: off
import torch
import tensorrt as trt
# isort: on

from parameterized import parameterized
from transformers import GPT2Config, GPTBigCodeConfig, GPTJConfig, LlamaConfig
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import (AttentionMaskConverter,
                                                   _prepare_4d_attention_mask)
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.gpt_bigcode.modeling_gpt_bigcode import \
    GPTBigCodeAttention
from transformers.models.gptj.modeling_gptj import GPTJAttention
from transformers.models.llama.modeling_llama import (LlamaAttention,
                                                      LlamaRotaryEmbedding)
from utils.util import (skip_bf16_fp32_accum, skip_fp8_pre_ada,
                        unittest_name_func)

import tensorrt_llm
import tensorrt_llm.quantization.layers
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_np, torch_to_numpy
from tensorrt_llm.functional import (PositionEmbeddingType, RopeEmbeddingUtils,
                                     RotaryScalingType)
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import GenerationSequence
from tensorrt_llm.runtime.memory_pools.memory_pools_allocator import \
    MemoryPoolsAllocator
from tensorrt_llm.runtime.memory_pools.pools_kv_cache_manager import \
    PoolsKVCacheManager


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')

    def _build_trt_engine(self, trt_network, trt_builder, dtype, shape_dict,
                          use_int8):
        config = trt_builder.create_builder_config(opt_level=0)
        if dtype == 'float16':
            config.flags = 1 << (int)(trt.BuilderFlag.FP16)

        opt_profile = trt_builder.create_optimization_profile()
        # Set optimization profiles for the input bindings that need them
        for i in range(trt_network.num_inputs):
            inp_tensor = trt_network.get_input(i)
            name = inp_tensor.name
            # Set profiles for dynamic execution tensors
            if not inp_tensor.is_shape_tensor and -1 in inp_tensor.shape:
                dims = trt.Dims(shape_dict[name])
                opt_profile.set_shape(name, dims, dims, dims)
        config.add_optimization_profile(opt_profile)
        return trt_builder.build_engine(trt_network, config)

    def load_test_cases():
        test_cases = []
        test_cases += list(
            product(['gpt2_attention', 'llama_attention', 'gptj_attention'],
                    [ContextFMHAType.disabled], ['float16'], ['float16'], [2],
                    [128], [8], [4], [64], [0], [1], [True, False], [False]))

        # TODO: add more unit tests
        test_cases += list(
            product(['llama_attention'], [
                ContextFMHAType.disabled, ContextFMHAType.enabled,
                ContextFMHAType.enabled_with_fp32_acc
            ], ['float16'], ['float16'], [2], [90], [8], [4], [32], [0], [1],
                    [False], [False]))

        # Test cases for the multi-block MMHA.
        test_cases += list(
            product(['llama_attention'], [
                ContextFMHAType.enabled, ContextFMHAType.enabled_with_fp32_acc
            ], ['float16', 'float32'], [''], [2], [2048], [8], [4], [64], [0],
                    [1], [True, False], [False]))
        test_cases += list(
            product(['llama_attention'],
                    [ContextFMHAType.enabled_with_fp32_acc], ['float16'], [''],
                    [16], [2048], [32], [4], [64], [0], [1], [False], [False]))

        # Test cases for the int8/fp8 K/V cache.
        test_cases += list(
            product(['gpt2_attention'], [ContextFMHAType.disabled],
                    ['float16', 'float32'], ['int8', 'fp8'], [2], [128], [8],
                    [4], [64], [0], [1], [False], [False]))

        # test cases for multi-query attention
        test_cases += list(
            product(['gpt_bigcode_attention'], [
                ContextFMHAType.disabled, ContextFMHAType.enabled,
                ContextFMHAType.enabled_with_fp32_acc
            ], ['float16'], [''], [2], [128], [8], [4], [64], [1], [1], [False],
                    [False]))

        # test cases for beam search
        test_cases += list(
            product(['gpt2_attention'], [ContextFMHAType.disabled], ['float16'],
                    [''], [2], [128], [8], [4], [64], [0], [4], [False],
                    [False]))

        # test cases for grouped-query attention
        test_cases += list(
            product(['llama_attention'], [ContextFMHAType.disabled],
                    ['float16'], [''], [2], [128], [8], [8], [32], [2, 4], [1],
                    [False], [False]))

        # test cases for rotary scaling
        test_cases += list(
            product(['llama_attention'], [ContextFMHAType.disabled],
                    ['float32'], [''], [2], [128], [8], [8], [32], [2, 8], [1],
                    [False], [False], [10000.0, 1000000.0], [
                        {
                            "type": "linear",
                            "factor": 3.0
                        },
                        {
                            "type": "dynamic",
                            "factor": 2.0
                        },
                    ]))

        # test cases for StreamingLLM
        test_cases += list(
            product(['llama_attention'], [ContextFMHAType.disabled],
                    ['float16'], [''], [2], [128], [8], [4], [64], [0], [1],
                    [False], [False], [10000.0], [None], [4]))

        # test cases for fp8_context_fmha
        test_cases += list(
            product(['gpt2_attention'], [ContextFMHAType.enabled], ['float16'],
                    ['fp8'], [2], [128], [8], [4], [64], [0], [1], [True],
                    [True]))

        return test_cases

    @parameterized.expand(load_test_cases, name_func=unittest_name_func)
    def test_gpt_attention_IFB(self,
                               attention_type,
                               context_fmha_type,
                               dtype,
                               kv_cache_dtype,
                               batch_size,
                               in_len,
                               out_len,
                               num_heads,
                               head_size,
                               num_kv_heads,
                               beam_width,
                               fuse_bias,
                               use_fp8_context_fmha,
                               rope_base=10000.0,
                               rope_scaling=None,
                               sink_token_len=0):
        use_int8_kv_cache = True if kv_cache_dtype == 'int8' else False
        use_fp8_kv_cache = True if kv_cache_dtype == 'fp8' else False

        if not (use_int8_kv_cache or use_fp8_kv_cache):
            kv_cache_dtype = dtype

        if use_fp8_kv_cache == False or context_fmha_type == ContextFMHAType.disabled:
            use_fp8_context_fmha = False

        skip_fp8_pre_ada(use_fp8_kv_cache or use_fp8_context_fmha)
        skip_bf16_fp32_accum(dtype, context_fmha_type)

        tolerances = {
            'float32': 2e-3,
            'float16': 2e-3,
            'bfloat16': 2e-3,
            'fp8': 3e-3,
        }

        if num_kv_heads == 0:
            num_kv_heads = num_heads

        session = None
        if use_int8_kv_cache or use_fp8_kv_cache:
            # Fixing seed to avoid flakiness in tests with quantization
            torch.manual_seed(42)

        if beam_width != 1:
            pytest.skip("Beam search is not supported in this test yet")

        tokens_per_block = 128
        streamingllm = sink_token_len > 0

        if streamingllm:
            pytest.skip(
                "Waived for now because attention sink cannot work with the non-cyclic kv cache kernel & runtime changes."
            )

        remove_input_padding = True

        def _construct_execution(session,
                                 input_tensor,
                                 weight,
                                 bias,
                                 host_kv_cache_block_offsets,
                                 host_kv_cache_pool_pointers,
                                 host_kv_cache_pool_mapping,
                                 sequence_length,
                                 host_past_key_value_lengths,
                                 host_max_attention_window_sizes,
                                 host_sink_token_length,
                                 context_lengths,
                                 max_context_length,
                                 cache_indirection,
                                 num_heads,
                                 hidden_size,
                                 num_kv_heads,
                                 output,
                                 dtype,
                                 kv_quant_scale,
                                 kv_dequant_scale,
                                 host_context_lengths,
                                 host_request_types,
                                 host_runtime_perf_knobs,
                                 host_context_progress,
                                 use_fp8_context_fmha=False,
                                 atten_output_quant_scale=None):
            kv_cache_block_offsets = host_kv_cache_block_offsets.to('cuda')
            head_size = hidden_size // num_heads
            # construct trt network
            builder = tensorrt_llm.Builder()
            net = builder.create_network()
            net.plugin_config.gpt_attention_plugin = dtype
            net.plugin_config.set_context_fmha(context_fmha_type)
            net.plugin_config.remove_input_padding = True
            net.plugin_config.enable_paged_kv_cache(tokens_per_block)
            net.plugin_config.use_paged_context_fmha = False

            if not use_fp8_context_fmha:
                net.plugin_config.use_fp8_context_fmha = False

            with tensorrt_llm.net_guard(net):
                x_tensor = Tensor(name='input',
                                  shape=tuple(input_tensor.shape),
                                  dtype=tensorrt_llm.str_dtype_to_trt(dtype))
                sequence_length_tensor = Tensor(
                    name='sequence_length',
                    shape=tuple(sequence_length.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                host_past_key_value_lengths_tensor = Tensor(
                    name='host_past_key_value_lengths',
                    shape=tuple(host_past_key_value_lengths.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                host_max_attention_window_sizes_tensor = Tensor(
                    name='host_max_attention_window_sizes',
                    shape=tuple(host_max_attention_window_sizes.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                host_sink_token_length_tensor = Tensor(
                    name='host_sink_token_length',
                    shape=tuple(host_sink_token_length.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                input_lengths_tensor = Tensor(
                    name='context_lengths',
                    shape=tuple(context_lengths.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                cache_indirection_tensor = Tensor(
                    name='cache_indirection',
                    shape=tuple(cache_indirection.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                host_request_types_tensor = Tensor(
                    name='host_request_types',
                    shape=tuple(host_request_types.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                kv_cache_block_offsets_tensor = Tensor(
                    name='kv_cache_block_offsets',
                    shape=tuple(kv_cache_block_offsets.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                host_kv_cache_block_offsets_tensor = Tensor(
                    name='host_kv_cache_block_offsets',
                    shape=tuple(kv_cache_block_offsets.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                host_kv_cache_pool_pointers_tensor = Tensor(
                    name='host_kv_cache_pool_pointers',
                    shape=(
                        1,
                        1,
                    ),
                    dtype=tensorrt_llm.str_dtype_to_trt('int64'))
                host_kv_cache_pool_mapping_tensor = Tensor(
                    name='host_kv_cache_pool_mapping',
                    shape=(1, 1),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                host_runtime_perf_knobs_tensor = Tensor(
                    name='host_runtime_perf_knobs',
                    shape=[16],
                    dtype=tensorrt_llm.str_dtype_to_trt('int64'))
                host_context_progress_tensor = Tensor(
                    name='host_context_progress',
                    shape=[1],
                    dtype=tensorrt_llm.str_dtype_to_trt('int64'))
                kv_quant_scale_tensor = None
                kv_dequant_scale_tensor = None
                if use_int8_kv_cache or use_fp8_kv_cache:
                    kv_quant_scale_tensor = Tensor(
                        name='kv_quant_scale',
                        shape=(1, ),
                        dtype=tensorrt_llm.str_dtype_to_trt('float32'))
                    kv_dequant_scale_tensor = Tensor(
                        name='kv_dequant_scale',
                        shape=(1, ),
                        dtype=tensorrt_llm.str_dtype_to_trt('float32'))

                atten_output_quant_scale_tensor = None
                if use_fp8_context_fmha:
                    atten_output_quant_scale_tensor = Tensor(
                        name='atten_output_quant_scale',
                        shape=(1, ),
                        dtype=tensorrt_llm.str_dtype_to_trt('float32'))

                host_context_lengths_tensor = None
                if remove_input_padding:
                    host_context_lengths_tensor = Tensor(
                        name='host_context_lengths',
                        shape=tuple(host_context_lengths.shape),
                        dtype=tensorrt_llm.str_dtype_to_trt('int32'))

                linear = tensorrt_llm.layers.Linear(hidden_size,
                                                    weight.size()[-1],
                                                    bias=attention_type in [
                                                        'gpt2_attention',
                                                        'llama_attention',
                                                        'gpt_bigcode_attention'
                                                    ])
                linear.weight.value = np.ascontiguousarray(
                    torch_to_numpy(weight.T.cpu()))
                if attention_type in [
                        'gpt2_attention', 'llama_attention',
                        'gpt_bigcode_attention'
                ]:
                    linear.bias.value = torch_to_numpy(bias.cpu())
                if fuse_bias:
                    qkv = tensorrt_llm.functional.matmul(x_tensor,
                                                         linear.weight.value,
                                                         transb=True)
                    qkv_bias = tensorrt_llm.functional.constant(
                        np.zeros((linear.out_features, ),
                                 dtype=str_dtype_to_np(dtype))
                    ) if linear.bias is None else linear.bias.value
                else:
                    qkv = linear(x_tensor)
                    qkv_bias = None

                rotary_embedding_dim = head_size if attention_type in [
                    'llama_attention', 'gptj_attention'
                ] else 0
                if attention_type == 'llama_attention':
                    position_embedding_type = PositionEmbeddingType.rope_gpt_neox
                elif attention_type == 'gptj_attention':
                    position_embedding_type = PositionEmbeddingType.rope_gptj
                else:
                    position_embedding_type = PositionEmbeddingType.learned_absolute

                rope_base = 10000.0
                rope_scale_type = RotaryScalingType.none
                rope_scale = 1.0
                if attention_type == "llama_attention":
                    rope_base = configuration.rope_theta
                    if configuration.rope_scaling is not None:
                        rope_scale_type = {
                            "linear": RotaryScalingType.linear,
                            "dynamic": RotaryScalingType.dynamic
                        }[configuration.rope_scaling["type"]]
                        rope_scale = configuration.rope_scaling["factor"]
                rotary_inv_freq, embed_positions_for_gpt_attention = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
                    configuration.max_position_embeddings, rotary_embedding_dim,
                    rope_base, rope_scale)
                rotary_inv_freq_cache = tensorrt_llm.functional.constant(
                    rotary_inv_freq) if position_embedding_type.is_rope(
                    ) else None
                rotary_cos_sin = tensorrt_llm.functional.constant(
                    embed_positions_for_gpt_attention
                ) if position_embedding_type.is_rope() else None
                outputs = tensorrt_llm.functional.gpt_attention(
                    qkv=qkv,
                    past_key_value=None,
                    sequence_length=sequence_length_tensor,
                    host_past_key_value_lengths=
                    host_past_key_value_lengths_tensor,
                    host_max_attention_window_sizes=
                    host_max_attention_window_sizes_tensor,
                    host_sink_token_length=host_sink_token_length_tensor,
                    context_lengths=input_lengths_tensor,
                    cache_indirection=cache_indirection_tensor,
                    host_request_types=host_request_types_tensor,
                    layer_idx=0,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    hidden_size_per_head=head_size,
                    q_scaling=1.0,
                    rotary_embedding_dim=rotary_embedding_dim,
                    rotary_embedding_base=rope_base,
                    rotary_embedding_scale_type=rope_scale_type,
                    rotary_embedding_scale=rope_scale,
                    rotary_embedding_max_positions=configuration.
                    max_position_embeddings,
                    position_embedding_type=position_embedding_type,
                    rotary_inv_freq=rotary_inv_freq_cache,
                    rotary_cos_sin=rotary_cos_sin,
                    kv_orig_quant_scale=kv_quant_scale_tensor,
                    kv_quant_orig_scale=kv_dequant_scale_tensor,
                    kv_cache_quant_mode=QuantMode.from_description(
                        use_int8_kv_cache=use_int8_kv_cache,
                        use_fp8_kv_cache=use_fp8_kv_cache,
                        use_fp8_qdq=use_fp8_context_fmha),
                    attention_output_orig_quant_scale=
                    atten_output_quant_scale_tensor,
                    max_context_length=max_context_length,
                    kv_cache_block_offsets=kv_cache_block_offsets_tensor,
                    host_kv_cache_block_offsets=
                    host_kv_cache_block_offsets_tensor,
                    host_kv_cache_pool_pointers=
                    host_kv_cache_pool_pointers_tensor,
                    host_kv_cache_pool_mapping=host_kv_cache_pool_mapping_tensor,
                    host_context_lengths=host_context_lengths_tensor,
                    qkv_bias=qkv_bias,
                    host_runtime_perf_knobs=host_runtime_perf_knobs_tensor,
                    host_context_progress=host_context_progress_tensor)

                output_dtype = 'fp8' if use_fp8_context_fmha else dtype

                net._mark_output(
                    outputs[0],
                    'output',
                    dtype=tensorrt_llm.str_dtype_to_trt(output_dtype))

            inputs = {
                'input': input_tensor,
                'sequence_length': sequence_length,
                'host_past_key_value_lengths': host_past_key_value_lengths,
                'host_max_attention_window_sizes':
                host_max_attention_window_sizes,
                'host_sink_token_length': host_sink_token_length,
                'context_lengths': context_lengths,
                'cache_indirection': cache_indirection,
                'host_request_types': host_request_types,
                'kv_cache_block_offsets': kv_cache_block_offsets,
                'host_kv_cache_block_offsets': host_kv_cache_block_offsets,
                'host_kv_cache_pool_pointers': host_kv_cache_pool_pointers,
                'host_kv_cache_pool_mapping': host_kv_cache_pool_mapping,
                'host_runtime_perf_knobs': host_runtime_perf_knobs,
                'host_context_progress': host_context_progress
            }
            if use_int8_kv_cache or use_fp8_kv_cache:
                inputs['kv_quant_scale'] = kv_quant_scale
                inputs['kv_dequant_scale'] = kv_dequant_scale

            if use_fp8_context_fmha:
                inputs['atten_output_quant_scale'] = atten_output_quant_scale

            if remove_input_padding:
                inputs['host_context_lengths'] = host_context_lengths

            outputs = {
                'output': output,
            }

            stream = torch.cuda.Stream()
            # NOTE: since we use int8 only for paged_kv_cache no int8 tensors are visible to TRT
            int8_trt_flag = False
            builder_config = builder.create_builder_config(
                name=attention_type,
                precision=dtype,
                fp8=use_fp8_context_fmha,
                int8=int8_trt_flag)
            if session is None:
                engine = builder.build_engine(net, builder_config)
                session = tensorrt_llm.runtime.Session.from_serialized_engine(
                    engine)
            session.run(inputs=inputs,
                        outputs=outputs,
                        stream=stream.cuda_stream)

            torch.cuda.synchronize()

            return session, outputs['output']

        hidden_size = num_heads * head_size  # embed dimension
        # If MQA/GQA and that GPTBigCodeAttention/LlamaAttention is tested, use compacted IO shape.
        # If MQA/GQA but other attention types are tested, use regular IO shape.
        # This is because GPTBigCodeAttention/LlamaAttention requires exact number of KV heads for MQA/GQA.
        # Other attention types do not support MQA/GQA natively so we emulate the effect of MQA/GQA by repeating KV heads.
        plugin_kv_num_heads = num_kv_heads if attention_type == 'llama_attention' or attention_type == 'gpt_bigcode_attention' else num_heads
        kv_hidden_size = plugin_kv_num_heads * head_size
        qkv_hidden_size = hidden_size + 2 * kv_hidden_size
        max_seq_len = in_len + out_len * 3
        num_req = batch_size
        in_lens = torch.randint(1, in_len + 1, (num_req, ))
        sink_tokens_in_last_block = sink_token_len % tokens_per_block
        bubble_len = tokens_per_block - sink_tokens_in_last_block if sink_tokens_in_last_block > 0 else 0
        max_blocks_per_seq = math.ceil(
            (max_seq_len + bubble_len) / tokens_per_block)
        num_blocks = num_req * beam_width * max_blocks_per_seq
        shape_dict = {
            'weight': (hidden_size, qkv_hidden_size),
            'bias': (qkv_hidden_size, ),
            'cache_indirection': (num_req, beam_width, max_seq_len),
            'past_key_value':
            (num_blocks, 2, plugin_kv_num_heads, tokens_per_block, head_size)
        }
        shape_dict['present_key_value'] = shape_dict['past_key_value']

        ordered_key_value = torch.zeros(
            shape_dict['present_key_value'],
            dtype=tensorrt_llm._utils.str_dtype_to_torch(kv_cache_dtype),
            device='cuda')

        weight = torch.randn(
            shape_dict['weight'],
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device='cuda') * 1e-3
        # FIXME: test_gpt_attention_llama_attention_False_float16_2_90_4_64_False_False_False_True
        # fails with xavier_uniform_ initialization
        # torch.nn.init.xavier_uniform_(weight)

        bias = torch.randn(shape_dict['bias'],
                           dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                           device='cuda') * 1e-2

        cache_indirection = torch.zeros(shape_dict['cache_indirection'],
                                        dtype=torch.int32,
                                        device='cuda')
        for req_idx in range(num_req):
            in_len_req = in_lens[req_idx]
            for iteration in range(1, beam_width):
                cache_indirection[req_idx, iteration, in_len_req:] = iteration

        ConfigCls = None
        AttentionCls = None
        if attention_type == 'gpt2_attention':
            ConfigCls = GPT2Config
            AttentionCls = GPT2Attention
        elif attention_type == 'gptj_attention':
            ConfigCls = GPTJConfig
            AttentionCls = GPTJAttention
        elif attention_type == 'llama_attention':
            ConfigCls = LlamaConfig
            AttentionCls = LlamaAttention
        elif attention_type == 'gpt_bigcode_attention':
            ConfigCls = GPTBigCodeConfig
            AttentionCls = GPTBigCodeAttention

        configuration = ConfigCls(hidden_size=hidden_size,
                                  num_hidden_layers=1,
                                  num_attention_heads=num_heads,
                                  vocab_size=51200,
                                  use_cache=True,
                                  resid_pdrop=0,
                                  embd_pdrop=0,
                                  attn_pdrop=0,
                                  hidden_act='gelu',
                                  dtype=dtype,
                                  attn_implementation='eager')
        if attention_type == 'llama_attention':
            configuration.num_key_value_heads = num_kv_heads
            configuration.rope_theta = rope_base
            configuration.rope_scaling = rope_scaling
            if rope_scaling is not None:
                # scaling is typically used for supporting longer seq lens than max_position_embeddings
                # so we set the max_position_embeddings to be smaller than total seq len
                # the following will use default path (no scaling) when generating half of the outputs
                # the other half will use activate the scaling
                # NOTE: in_len is also halved because in case the other half is treated as padding.
                configuration.max_position_embeddings = (
                    in_len // 2) + out_len - (out_len // 2)
            rotary_emb = LlamaRotaryEmbedding(config=configuration,
                                              device='cuda')
        attention = AttentionCls(configuration, layer_idx=0).cuda().eval()
        if attention_type == 'gpt2_attention':
            attention.c_attn.weight = torch.nn.parameter.Parameter(
                data=weight.clone().detach(), requires_grad=False)
            attention.c_attn.bias = torch.nn.parameter.Parameter(
                data=bias.clone().detach(), requires_grad=False)
            attention.c_proj.weight = torch.nn.parameter.Parameter(
                data=torch.eye(
                    hidden_size,
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                    device='cuda'),
                requires_grad=False)
            attention.c_proj.bias = torch.nn.parameter.Parameter(
                data=torch.zeros(
                    (hidden_size, ),
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                    device='cuda'),
                requires_grad=False)
        elif attention_type == 'llama_attention':
            q_w, k_w, v_w = torch.split(weight.transpose(
                0, 1), [hidden_size, kv_hidden_size, kv_hidden_size],
                                        dim=0)
            q_b, k_b, v_b = torch.split(
                bias, [hidden_size, kv_hidden_size, kv_hidden_size])
            attention.q_proj.weight = torch.nn.parameter.Parameter(
                data=q_w.contiguous().clone().detach(), requires_grad=False)
            attention.k_proj.weight = torch.nn.parameter.Parameter(
                data=k_w.contiguous().clone().detach(), requires_grad=False)
            attention.v_proj.weight = torch.nn.parameter.Parameter(
                data=v_w.contiguous().clone().detach(), requires_grad=False)

            attention.q_proj.bias = torch.nn.parameter.Parameter(
                data=q_b.contiguous().clone().detach(), requires_grad=False)
            attention.k_proj.bias = torch.nn.parameter.Parameter(
                data=k_b.contiguous().clone().detach(), requires_grad=False)
            attention.v_proj.bias = torch.nn.parameter.Parameter(
                data=v_b.contiguous().clone().detach(), requires_grad=False)

            attention.o_proj.weight = torch.nn.parameter.Parameter(
                data=torch.eye(
                    hidden_size,
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                    device='cuda'),
                requires_grad=False)
            attention.o_proj.bias = torch.nn.parameter.Parameter(
                data=torch.zeros(
                    (hidden_size, ),
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                    device='cuda'),
                requires_grad=False)
            attention.layer_idx = 0
        elif attention_type == 'gptj_attention':
            q_w, k_w, v_w = torch.split(weight.transpose(
                0, 1), [hidden_size, kv_hidden_size, kv_hidden_size],
                                        dim=0)
            attention.q_proj.weight = torch.nn.parameter.Parameter(
                data=q_w.contiguous().clone().detach(), requires_grad=False)
            attention.k_proj.weight = torch.nn.parameter.Parameter(
                data=k_w.contiguous().clone().detach(), requires_grad=False)
            attention.v_proj.weight = torch.nn.parameter.Parameter(
                data=v_w.contiguous().clone().detach(), requires_grad=False)

            attention.out_proj.weight = torch.nn.parameter.Parameter(
                data=torch.eye(
                    hidden_size,
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                    device='cuda'),
                requires_grad=False)
            attention.layer_idx = 0
        elif attention_type == 'gpt_bigcode_attention':
            attention.c_attn.weight = torch.nn.parameter.Parameter(
                data=weight.transpose(0, 1).clone().detach(),
                requires_grad=False)
            attention.c_attn.bias = torch.nn.parameter.Parameter(
                data=bias.clone().detach(), requires_grad=False)
            attention.c_proj.weight = torch.nn.parameter.Parameter(
                data=torch.eye(
                    hidden_size,
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                    device='cuda'),
                requires_grad=False)
            attention.c_proj.bias = torch.nn.parameter.Parameter(
                data=torch.zeros(
                    (hidden_size, ),
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                    device='cuda'),
                requires_grad=False)
            attention.layer_idx = 0
        else:
            raise RuntimeError("attention_type not properly set")

        ctx_attention_mask_list = [None] * num_req

        # Setup weights/biases for MQA: key/value shares weights/biases across heads
        if attention_type != 'gpt_bigcode_attention' and attention_type != "llama_attention" and num_kv_heads == 1:
            q_w, k_w, v_w = torch.tensor_split(weight, 3, dim=-1)
            q_b, k_b, v_b = torch.tensor_split(bias, 3)
            k_w_head = k_w[:, :head_size]
            v_w_head = k_w[:, :head_size]
            k_w_repeat = k_w_head.repeat(1, num_heads)
            v_w_repeat = v_w_head.repeat(1, num_heads)
            k_b_head = k_b[:head_size]
            v_b_head = v_b[:head_size]
            k_b_repeat = k_b_head.repeat(num_heads)
            v_b_repeat = v_b_head.repeat(num_heads)

            # Setup MQA weights/biases for _construct_execution()
            weight = torch.cat([q_w, k_w_repeat, v_w_repeat], dim=-1)
            bias = torch.cat([q_b, k_b_repeat, v_b_repeat])

            # Plugin will always use compacted MQA format without repeating KV heads
            weight_plugin = torch.cat([q_w, k_w_head, v_w_head], dim=-1)
            bias_plugin = torch.cat([q_b, k_b_head, v_b_head])

            # Setup MQA weights/biases for torch
            if attention_type == 'gpt2_attention':
                attention.c_attn.weight = torch.nn.parameter.Parameter(
                    data=weight.clone().detach(), requires_grad=False)
                attention.c_attn.bias = torch.nn.parameter.Parameter(
                    data=bias.clone().detach(), requires_grad=False)
            elif attention_type == 'llama_attention':
                attention.k_proj.weight = torch.nn.parameter.Parameter(
                    data=k_w_repeat.contiguous().clone().detach(),
                    requires_grad=False)
                attention.v_proj.weight = torch.nn.parameter.Parameter(
                    data=v_w_repeat.contiguous().clone().detach(),
                    requires_grad=False)
                attention.k_proj.bias = torch.nn.parameter.Parameter(
                    data=k_b_repeat.contiguous().clone().detach(),
                    requires_grad=False)
                attention.v_proj.bias = torch.nn.parameter.Parameter(
                    data=v_b_repeat.contiguous().clone().detach(),
                    requires_grad=False)
            elif attention_type == 'gptj_attention':
                attention.k_proj.weight = torch.nn.parameter.Parameter(
                    data=k_w_repeat.contiguous().clone().detach(),
                    requires_grad=False)
                attention.v_proj.weight = torch.nn.parameter.Parameter(
                    data=v_w_repeat.contiguous().clone().detach(),
                    requires_grad=False)
            else:
                raise RuntimeError("attention_type not properly set")

        else:  # not MQA/GQA
            weight_plugin = weight
            bias_plugin = bias

        # torch execution for one sequence
        def torch_exec(step: int,
                       input: torch.Tensor,
                       ctx_attention_mask: torch.Tensor,
                       req_idx: int,
                       layer_past=None):
            assert layer_past != None or input.shape[0] == 1
            nonlocal attention
            nonlocal attention_type
            nonlocal in_lens
            in_len = in_lens[req_idx]
            position_ids = ctx_attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(ctx_attention_mask == 0, 1)
            if step != 0:
                position_ids = position_ids[:, -1].unsqueeze(-1)

            attention_mask = _prepare_4d_attention_mask(
                ctx_attention_mask,
                dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                tgt_len=(in_len if step == 0 else 1))
            if attention_type == 'gpt2_attention':
                torch_output = attention(input,
                                         past_key_value=layer_past,
                                         use_cache=True,
                                         attention_mask=attention_mask)[0]
                torch_present = layer_past
            elif attention_type == 'llama_attention':
                position_embeddings = rotary_emb(input, position_ids)
                attention_mask = attention_mask + AttentionMaskConverter._make_causal_mask(
                    input.shape[:2],
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                    device='cuda',
                    past_key_values_length=(0 if step == 0 else in_len + step -
                                            1))
                torch_output = attention(
                    input,
                    past_key_value=layer_past,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    use_cache=True)[0]
                torch_present = layer_past
            elif attention_type == 'gptj_attention':
                torch_output, torch_present = attention(
                    input,
                    layer_past=layer_past,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=True)
                torch_present = layer_past
            elif attention_type == 'gpt_bigcode_attention':
                # target shape = (b, h, s_query or 1, s_key)
                attention_mask = (attention_mask
                                  >= 0).expand(input.shape[0], num_heads,
                                               in_len if step == 0 else 1,
                                               in_len + step)
                torch_output, torch_present = attention(
                    input,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    use_cache=True)
                torch_present = layer_past
            else:
                raise RuntimeError("attention_type not properly set")

            torch.cuda.synchronize()
            return torch_output, torch_present

        # Init Pools KV cache manager
        memory_pools_allocator = MemoryPoolsAllocator(
            num_blocks=num_blocks,
            tokens_per_block=tokens_per_block,
            head_size=head_size)
        num_kv_heads_per_layer = MemoryPoolsAllocator.prepare_num_kv_heads_per_layer(
            plugin_kv_num_heads, 1)
        memory_pools_allocator.allocate(dtype, num_kv_heads_per_layer)
        pools_kv_cache_manager = PoolsKVCacheManager(
            memory_pools_allocator.pools_metadata,
            max_blocks_per_seq,
            num_blocks,
            tokens_per_block,
            head_size,
            max_attention_window_size=max_seq_len,
            beam_width=beam_width,
            sink_token_len=sink_token_len)

        host_kv_cache_pool_pointers = memory_pools_allocator.get_kv_cache_pool_pointers(
        )
        host_kv_cache_pool_mapping = memory_pools_allocator.pool_mapping
        print("pool ptr ", ordered_key_value.data_ptr())

        torch_cache_list = [None] * num_req
        cache_num_req = 0
        # We don't start all requests together. The i-th request starts from the i-th iteration.
        # Like below, each column is a iteration. c means context step, g means generation step and n means none
        # req0 |c|g|g|g|g|g|g|g|n|n|n|
        # req1 |n|c|g|g|g|g|g|g|g|n|n|
        # req2 |n|n|c|g|g|g|g|g|g|g|n|
        # req3 |n|n|n|c|g|g|g|g|g|g|g|
        for iteration in range(num_req + out_len - 1):
            get_step = lambda req_idx: iteration - req_idx
            is_valid_step = lambda step: step >= 0 and step < out_len

            num_context_req = 0
            num_generation_req = 0
            input_length_list = []
            request_type_list = []
            sequence_length_list = []
            host_past_key_value_length_list = []
            context_length_list = []
            sequence_selection = []
            batch_req_ids = []
            for req_idx in reversed(range(num_req)):
                step = get_step(req_idx)
                in_len_req = in_lens[req_idx]
                if is_valid_step(step):
                    batch_req_ids.append(req_idx)
                    if step == 0:
                        input_length_list.append([in_len_req])
                        context_length_list += [in_len_req]
                        request_type_list += [0]
                        host_past_key_value_length_list += [0]
                        sequence_selection += [req_idx * beam_width]
                        num_context_req += 1
                    else:
                        input_length_list.append([1] * beam_width)
                        context_length_list += [in_len_req] * beam_width
                        request_type_list += [1] * beam_width
                        host_past_key_value_length_list += [
                            in_len_req + step - 1
                        ] * beam_width
                        num_generation_req += 1
                        sequence_selection += list(
                            range(req_idx * beam_width,
                                  (req_idx + 1) * beam_width))
                    sequence_length_list += [in_len_req + step] * beam_width

            num_seq = num_context_req + num_generation_req * beam_width

            # Check if new sequence arrived
            if iteration < num_req:
                in_len_req = in_lens[iteration]
                # Add sequence to the manager
                sequence = GenerationSequence(seq_idx=iteration,
                                              batch_idx=iteration)
                pools_kv_cache_manager.add_sequence(sequence,
                                                    in_len_req.clone())

            # Get arrays of pointers to the "pages" of KV values
            offset_array = pools_kv_cache_manager.get_block_offsets(beam_width)
            assert offset_array.shape[
                0] == 1, f"test is suppose to use only one pool. sequence_selection is based on a single pool"
            # assume only one pool
            dense_offset_array = offset_array[0][sequence_selection]

            host_input_lengths = np.concatenate(input_length_list)
            host_input_lengths = torch.tensor(host_input_lengths,
                                              dtype=torch.int32,
                                              device='cpu').reshape(-1)
            host_context_lengths = torch.tensor(context_length_list,
                                                dtype=torch.int32,
                                                device='cpu').reshape(-1)
            host_past_key_value_lengths = torch.tensor(
                host_past_key_value_length_list,
                dtype=torch.int32,
                device='cpu').reshape(-1)
            host_max_attention_window_sizes = torch.tensor([max_seq_len],
                                                           dtype=torch.int32,
                                                           device='cpu')
            host_sink_token_length = torch.tensor([0],
                                                  dtype=torch.int32,
                                                  device='cpu')
            total_num_tokens = int(sum(host_input_lengths))
            max_context_length = in_len
            context_lengths = host_context_lengths.cuda()

            host_request_types = torch.tensor(request_type_list,
                                              dtype=torch.int32).reshape(-1)

            sequence_lengths = torch.tensor(sequence_length_list,
                                            dtype=torch.int32,
                                            device='cuda').reshape(-1)

            perf_knob_tensor_size = 16
            generation_host_runtime_perf_knobs = torch.tensor(
                [-1] * perf_knob_tensor_size, dtype=torch.int64, device='cpu')

            generation_host_runtime_perf_knobs[
                0] = 1  # multi_block_mode is default on
            if context_fmha_type == ContextFMHAType.enabled_with_fp32_acc:
                generation_host_runtime_perf_knobs[
                    1] = 1  # enable_context_fmha_fp32_acc

            host_context_progress = torch.tensor([0],
                                                 dtype=torch.int64,
                                                 device='cpu')

            local_shape_dict = {
                'input': (total_num_tokens, hidden_size),
                'output': (total_num_tokens, hidden_size),
                'past_key_value':
                (num_blocks, 2, num_kv_heads, tokens_per_block, head_size),
                'sequence_length': (num_seq, ),
                'host_context_lengths': (num_seq, ),
                'host_request_types': (num_seq, ),
                'context_lengths': (num_seq, ),
                'cache_indir': (num_req, beam_width, max_seq_len),
                'block_pointers': (num_req, beam_width, 2, max_blocks_per_seq),
                'host_request_type': (num_seq),
            }

            input_tensor = torch.randn(
                local_shape_dict['input'],
                dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                device='cuda') * 1e-3

            output_dtype = 'fp8' if use_fp8_context_fmha else dtype
            output = torch.zeros(
                local_shape_dict['output'],
                dtype=tensorrt_llm._utils.str_dtype_to_torch(output_dtype),
                device='cuda')

            output_ref = torch.zeros(
                local_shape_dict['output'],
                dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                device='cuda')

            for req_idx in batch_req_ids:
                step = get_step(req_idx)
                assert is_valid_step(step)
                in_len_req = in_lens[req_idx]
                if step == 0:
                    ctx_attention_mask_list[req_idx] = torch.ones(
                        (1, in_len_req), dtype=torch.int32, device='cuda')
                else:
                    if step == 1:
                        ctx_attention_mask_list[req_idx] = torch.ones(
                            (beam_width, in_len_req),
                            dtype=torch.int32,
                            device='cuda')
                    ctx_attention_mask_list[req_idx] = torch.cat(
                        (ctx_attention_mask_list[req_idx],
                         ctx_attention_mask_list[req_idx].new_ones(
                             (beam_width, 1))),
                        dim=-1).contiguous()

            offset = 0
            max_kv_cache = 0
            max_output = 0
            for i, req_idx in enumerate(batch_req_ids):
                step = get_step(req_idx)
                assert is_valid_step(step)
                if step == 1 and beam_width > 1:
                    if attention_type != "gpt_bigcode_attention":
                        assert torch_cache_list[req_idx][0].shape[0] == 1
                        torch_cache_list[req_idx] = [
                            x.repeat((beam_width, 1, 1, 1))
                            for x in torch_cache_list[req_idx]
                        ]
                    else:
                        torch_cache_list[req_idx] = torch_cache_list[
                            req_idx].repeat(beam_width, 1, 1)
                input_length = input_length_list[i][0]
                local_beam_width = beam_width if step != 0 else 1
                offset_next = offset + input_length * local_beam_width
                if remove_input_padding:
                    torch_in = input_tensor[offset:offset_next, :].reshape(
                        (local_beam_width, input_length, hidden_size))
                else:
                    torch_in = input_tensor[:, offset:offset_next, :].reshape(
                        (local_beam_width, input_length, hidden_size))

                # llama/gpt2 uses DynamicCache
                past_key_values = DynamicCache.from_legacy_cache(
                    torch_cache_list[req_idx])

                torch_out, past_key_values = torch_exec(
                    step, torch_in, ctx_attention_mask_list[req_idx], req_idx,
                    past_key_values)

                # llama/gpt2 uses DynamicCache
                torch_cache_list[req_idx] = past_key_values.to_legacy_cache()
                past_key_values = torch_cache_list[req_idx][0]

                if use_fp8_kv_cache or use_int8_kv_cache:
                    max_kv_cache = max(
                        max_kv_cache,
                        torch.concat((past_key_values[0],
                                      past_key_values[1])).abs().max().item())

                if use_fp8_context_fmha:
                    max_output = max(max_output, torch_out.abs().max().item())

                output_ref[offset:offset_next, :] = torch_out.reshape(
                    (offset_next - offset, -1))
                offset = offset_next

            kv_dequant_scale = None
            kv_quant_scale = None

            if use_fp8_kv_cache:
                kv_dequant_scale = torch.tensor([max_kv_cache],
                                                device='cuda').float() / 440.
                kv_quant_scale = 1.0 / kv_dequant_scale
            elif use_int8_kv_cache:
                kv_dequant_scale = torch.tensor([max_kv_cache],
                                                device='cuda').float() / 127.
                kv_quant_scale = 1.0 / kv_dequant_scale

            atten_output_dequant_scale = None
            atten_output_quant_scale = None

            if use_fp8_context_fmha:
                atten_output_dequant_scale = torch.tensor(
                    [max_output], device='cuda').float() / 440.
                atten_output_quant_scale = 1.0 / atten_output_dequant_scale

            session, output = _construct_execution(
                session, input_tensor, weight_plugin, bias_plugin,
                dense_offset_array, host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping, sequence_lengths,
                host_past_key_value_lengths, host_max_attention_window_sizes,
                host_sink_token_length, context_lengths, max_context_length,
                cache_indirection, num_heads, hidden_size, num_kv_heads, output,
                dtype, kv_quant_scale, kv_dequant_scale, host_context_lengths,
                host_request_types, generation_host_runtime_perf_knobs,
                host_context_progress, use_fp8_context_fmha,
                atten_output_quant_scale)

            del session
            session = None

            if use_fp8_context_fmha:
                output = output.to(
                    output_ref.dtype) * atten_output_dequant_scale
                output = output.to(output_ref.dtype)

            torch.testing.assert_close(
                output,
                output_ref,
                rtol=tolerances[output_dtype],
                atol=tolerances[output_dtype],
            )

            # Due to the design of the test we do not remove finished sequences, but keep them as "none" instead
            cache_num_req = min(iteration + 1, num_req)
            finished = [False for _ in range(cache_num_req)]
            # Iterate to the next step. Increase number of tokens for all unfinished sequences
            # And allocate new blocks if needed
            pools_kv_cache_manager.step(finished)


if __name__ == "__main__":
    unittest.main()
