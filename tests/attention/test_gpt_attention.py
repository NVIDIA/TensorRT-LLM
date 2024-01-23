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
import os
import sys
import unittest
from itertools import product

import numpy as np
import pytest
import torch
from parameterized import parameterized
from transformers import GPT2Config, GPTBigCodeConfig, GPTJConfig, LlamaConfig
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import (AttentionMaskConverter,
                                                   _prepare_4d_attention_mask)
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.gpt_bigcode.modeling_gpt_bigcode import \
    GPTBigCodeAttention
from transformers.models.gptj.modeling_gptj import GPTJAttention
from transformers.models.llama.modeling_llama import LlamaAttention

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm._utils import (str_dtype_to_np, str_dtype_to_torch,
                                 torch_to_numpy)
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import GenerationSequence, KVCacheManager

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import getSMVersion


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def load_test_cases():
        test_cases = []
        test_cases += list(
            product(['gpt2_attention', 'llama_attention', 'gptj_attention'],
                    [ContextFMHAType.disabled], ['float16', 'bfloat16'], [None],
                    [2], [128], [4], [64], [0], [False], [False], [1, 4],
                    [True, False], [True, False]))

        # Test cases for input padding
        test_cases += list(
            product(['llama_attention'], [
                ContextFMHAType.disabled,
                ContextFMHAType.enabled,
            ], ['float16', 'bfloat16'], [None], [2], [128], [4], [64], [False],
                    [False], [False, True], [1], [False], [False]))

        # Test cases for fused context MHAs
        test_cases += list(
            product(['llama_attention'], [
                ContextFMHAType.enabled, ContextFMHAType.enabled_with_fp32_acc
            ], ['float16', 'bfloat16'], [None], [2], [90, 1024], [4],
                    [32, 64, 80, 112, 128], [0], [False], [False, True], [1],
                    [False], [False]))

        # Test cases for gptj rotary embedding.
        # The previous test cases have covered the gptneox rotary embedding.
        test_cases += list(
            product(['gptj_attention'], [
                ContextFMHAType.enabled,
            ], ['float16', 'bfloat16', 'float32'], [None], [2], [128], [32],
                    [32, 64, 80, 96, 128], [0], [False], [False], [1],
                    [True, False], [False]))

        # Test cases of float32 d=256 case (for testing MMHA key loops).
        test_cases += list(
            product(['gptj_attention'], [
                ContextFMHAType.enabled,
            ], ['float32'], [None], [2], [128], [2], [256], [False], [False],
                    [True], [1], [False], [True, False]))

        # # Test cases for the multi-block MMHA.
        # # NOTE: With long in_len=2048, beam_width=4 runs into OOM issue.
        test_cases += list(
            product(['llama_attention'], [
                ContextFMHAType.enabled, ContextFMHAType.enabled_with_fp32_acc
            ], ['float16', 'bfloat16'], [None], [2], [2048], [4], [64], [0],
                    [True], [False], [1], [False], [False]))

        # Test cases for the multi-block MMHA (with large number of blocks per sequence).
        test_cases += list(
            product(['llama_attention'], [
                ContextFMHAType.enabled, ContextFMHAType.enabled_with_fp32_acc
            ], ['float16', 'bfloat16', 'float32'], [None], [1], [4096], [1],
                    [128], [0], [True], [False], [1], [False], [False]))

        # Test cases for the 8-bit K/V cache.
        test_cases += list(
            product(['gpt2_attention'], [ContextFMHAType.disabled],
                    ['float16', 'float32'], ['int8', 'fp8'], [2], [128], [4],
                    [64], [0], [False], [False], [1, 4], [False], [False]))

        # test cases for multi-query attention
        test_cases += list(
            product(['gpt_bigcode_attention'], [
                ContextFMHAType.disabled, ContextFMHAType.enabled,
                ContextFMHAType.enabled_with_fp32_acc
            ], ['float16', 'bfloat16'], [None], [2], [128], [4], [64], [1],
                    [False], [False], [1, 4], [False], [False]))

        # test cases for grouped-query attention
        test_cases += list(
            product(['llama_attention'], [ContextFMHAType.disabled],
                    ['bfloat16', 'float16'], [None], [2], [4], [8], [32],
                    [2, 4], [False], [False], [1], [False], [False]))
        test_cases += list(
            product(['llama_attention'], [ContextFMHAType.enabled], ['float32'],
                    [None], [1], [165], [32], [128], [4], [False], [False], [1],
                    [False], [False]))

        # test cases for RoPE base and scaling
        test_cases += list(
            product(
                ['llama_attention'],
                [ContextFMHAType.disabled],
                ['bfloat16', 'float32'],
                [None],
                [2],
                [4],
                [8],
                [32],
                [2, 4],
                [False],
                [False],
                [1],
                [False],
                [False],
                [10000.0, 1000000.0],  # rope base
                [  # rope scaling
                    {
                        "type": "linear",
                        "factor": 2.0
                    },
                    {
                        "type": "dynamic",
                        "factor": 3.0
                    },
                ]))
        test_cases += list(
            product(
                ['llama_attention'],
                [ContextFMHAType.enabled],
                ['float32'],
                [None],
                [1],
                [165],
                [32],
                [128],
                [4],
                [False],
                [False],
                [1],
                [False],
                [False],
                [10000.0, 1000000.0],  # rope base
                [  # rope scaling
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
                    ['float16'], [None], [2], [128], [4], [64], [0], [False],
                    [False], [1, 4], [True, False], [False], [10000.0], [None],
                    [4], [False, True]))

        # add gpu_arch_lists for testing (help reducing workload if there are duplicates).
        test_cases = [("all", ) + case for case in test_cases]

        # For XQA kernels
        # all arches use the same kernel traits, we can distribute the workloads.
        # for XQA int8/fp8 KVCache
        test_cases += list(
            product(
                [80],
                ['llama_attention'],
                [ContextFMHAType.enabled],
                ['bfloat16', 'float16'],
                ['int8'],
                [2],
                [165, 544, 2032],
                [16],
                [128],
                [2],
                [False, True],
                [False],
                [1],
                [False, True],
                [False],
                [10000.0],  # rope base
                [  # rope scaling
                    None,
                ]))

        test_cases += list(
            product(
                [89],
                ['llama_attention'],
                [ContextFMHAType.enabled],
                ['bfloat16', 'float16'],
                ['fp8'],
                [2],
                [165, 544, 2032],
                [16],
                [128],
                [2],
                [False, True],
                [False],
                [1],
                [False, True],
                [False],
                [10000.0],  # rope base
                [  # rope scaling
                    None,
                ]))

        # Test case for GPT-J beam_width=4, used in MLPerf.
        test_cases += list(
            product([80], ['gptj_attention'], [ContextFMHAType.disabled],
                    ['float16'], ['float16'], [2], [128], [4], [256], [0],
                    [False], [False], [4], [True], [False]))
        test_cases += list(
            product([90], ['gptj_attention'], [ContextFMHAType.disabled],
                    ['float16'], ['fp8'], [2], [128], [4], [256], [0], [False],
                    [False], [4], [True], [False]))

        # split test cases into two partitions
        test_cases = [("partition0", ) + case if i % 2 == 0 else
                      ("partition1", ) + case
                      for i, case in enumerate(test_cases)]

        # tests (h100 only)
        # For warp-specialized kernels on Hopper.
        test_cases += list(
            product(
                ['h100_only'],
                [90],
                ['llama_attention'],
                [
                    ContextFMHAType.enabled,
                    ContextFMHAType.enabled_with_fp32_acc
                ],
                ['float16', 'bfloat16'],
                [None],
                [4],
                [68, 2543],
                [16],
                [32, 64, 128, 256],
                [2],
                [False],
                [False],
                [1],
                [False],
                [False],
                [10000.0],  # rope base
                [  # rope scaling
                    None,
                ]))

        # For XQA kernels
        # all arches use the same kernel traits, we can assign some workloads to h100-only.
        test_cases += list(
            product(
                ['h100_only'],
                [90],
                ['llama_attention'],
                [ContextFMHAType.enabled],
                ['float16', 'bfloat16'],
                [None],
                [2],
                [165, 1025, 2543],
                [16],
                [128],
                [2],
                [False, True],
                [False],
                [1],
                [False, True],
                [False],
                [10000.0],  # rope base
                [  # rope scaling
                    None,
                ]))

        # for XQA multi batch split logic test
        test_cases += list(
            product(
                ['h100_only'],
                [90],
                ['llama_attention'],
                [ContextFMHAType.enabled],
                ['float16'],
                [None],
                [150],
                [500],
                [32],
                [128],
                [4],
                [False, True],
                [False],
                [1],
                [False],
                [False],
                [10000.0],  # rope base
                [  # rope scaling
                    None,
                ]))

        return test_cases

    def custom_name_func(testcase_func, param_num, param):
        return "%s_%s" % (
            testcase_func.__name__,
            parameterized.to_safe_name("_".join(str(x) for x in param.args)),
        )

    @parameterized.expand(load_test_cases, name_func=custom_name_func)
    def test_gpt_attention(self,
                           test_partition,
                           gpu_arch,
                           attention_type,
                           context_fmha_type,
                           dtype,
                           kv_cache_dtype,
                           batch_size,
                           in_len,
                           num_heads,
                           head_size,
                           num_kv_heads,
                           enable_multi_block_mmha,
                           enable_remove_input_padding,
                           beam_width,
                           paged_kv_cache,
                           fuse_bias,
                           rope_base=10000.0,
                           rope_scaling=None,
                           sink_token_len=0,
                           enable_pos_shift=False):
        # if attention_type != "gpt_bigcode_attention" and attention_type != "llama_attention":
        #     assert num_kv_heads == 0 # safe guard against bad test case configs

        os.environ['TRTLLM_ENABLE_XQA'] = '1'
        os.environ['TRTLLM_FORCE_XQA'] = '1'
        use_int8_kv_cache = True if kv_cache_dtype == 'int8' else False
        use_fp8_kv_cache = True if kv_cache_dtype == 'fp8' else False
        if kv_cache_dtype is None:
            kv_cache_dtype = dtype
        # skip tests based on the gpu_arch_lists
        if gpu_arch != 'all':
            assert gpu_arch in [70, 80, 86, 89, 90]
            if getSMVersion() != gpu_arch:
                pytest.skip(
                    "Skip the test as the target gpu arch doesn't match this gpu arch."
                )

        if num_kv_heads == 0:
            num_kv_heads = num_heads

        # Skip tests that are not supported in pre-ampere architecture
        if getSMVersion() < 80:
            if context_fmha_type == ContextFMHAType.enabled_with_fp32_acc:
                pytest.skip(
                    "ContextFMHAType with fp32 acc is not supported in pre-ampere architecture"
                )
            if dtype == 'bfloat16':
                pytest.skip(
                    "bfloat16 is not supported in pre-ampere architecture")

        if getSMVersion() < 89:
            if use_fp8_kv_cache:
                pytest.skip("FP8 is not supported on pre-Ada architectures")

        # Skip duplicated tests.
        if context_fmha_type == ContextFMHAType.enabled_with_fp32_acc and \
            dtype == 'bfloat16':
            pytest.skip("bfloat16 Context FMHA will always accumulate on FP32, \
                so it has been tested with ContextFMHAType.enabled")

        session = None
        if use_int8_kv_cache or use_fp8_kv_cache or True:
            # Fixing seed to avoid flakiness in tests with quantization
            torch.manual_seed(42)

        tokens_per_block = 128 if paged_kv_cache else -1

        def _construct_execution(
                session, input_tensor, weight, bias, past_key_value,
                host_pointer_array, sequence_length,
                host_past_key_value_lengths, host_max_attention_window_sizes,
                host_sink_token_length, context_lengths, host_context_lengths,
                cache_indirection, host_request_types, num_heads, hidden_size,
                num_kv_heads, output, dtype, max_context_length, shape_dict,
                kv_int8_quant_scale, kv_int8_dequant_scale, configuration):
            pointer_array = None
            if paged_kv_cache:
                pointer_array = host_pointer_array.to('cuda')
            head_size = hidden_size // num_heads
            # construct trt network
            builder = tensorrt_llm.Builder()
            net = builder.create_network()
            net.plugin_config.set_gpt_attention_plugin(dtype)
            net.plugin_config.set_context_fmha(context_fmha_type)
            if enable_remove_input_padding:
                net.plugin_config.enable_remove_input_padding()
            if paged_kv_cache:
                net.plugin_config.enable_paged_kv_cache(tokens_per_block)
            if enable_multi_block_mmha:
                net.plugin_config.enable_mmha_multi_block_mode()
            # always enable xqa kernels for test.
            net.plugin_config.enable_xqa_optimization()

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
                context_lengths_tensor = Tensor(
                    name='context_lengths',
                    shape=tuple(context_lengths.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                host_context_lengths_tensor = Tensor(
                    name='host_context_lengths',
                    shape=tuple(context_lengths.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt(
                        'int32')) if enable_remove_input_padding else None
                cache_indirection_tensor = Tensor(
                    name='cache_indirection',
                    shape=tuple(cache_indirection.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                host_request_types_tensor = Tensor(
                    name='host_request_types',
                    shape=tuple(host_request_types.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))

                past_key_value_tensor = None
                pointer_array_tensor = None
                host_pointer_array_tensor = None
                if paged_kv_cache:
                    pointer_array_tensor = Tensor(
                        name='kv_cache_block_pointers',
                        shape=tuple(pointer_array.shape),
                        dtype=tensorrt_llm.str_dtype_to_trt('int64'))
                    host_pointer_array_tensor = Tensor(
                        name='host_kv_cache_block_pointers',
                        shape=tuple(pointer_array.shape),
                        dtype=tensorrt_llm.str_dtype_to_trt('int64'))
                else:
                    past_key_value_tensor = Tensor(
                        name='past_key_value',
                        shape=tuple(past_key_value.shape),
                        dtype=tensorrt_llm.str_dtype_to_trt(kv_cache_dtype))

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
                outputs = tensorrt_llm.functional.gpt_attention(
                    qkv=qkv,
                    past_key_value=past_key_value_tensor,
                    sequence_length=sequence_length_tensor,
                    host_past_key_value_lengths=
                    host_past_key_value_lengths_tensor,
                    host_max_attention_window_sizes=
                    host_max_attention_window_sizes_tensor,
                    host_sink_token_length=host_sink_token_length_tensor,
                    context_lengths=context_lengths_tensor,
                    cache_indirection=cache_indirection_tensor,
                    host_request_types=host_request_types_tensor,
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
                    kv_orig_quant_scale=kv_quant_scale_tensor,
                    kv_quant_orig_scale=kv_dequant_scale_tensor,
                    host_context_lengths=host_context_lengths_tensor,
                    kv_cache_quant_mode=QuantMode.from_description(
                        use_int8_kv_cache=use_int8_kv_cache,
                        use_fp8_kv_cache=use_fp8_kv_cache),
                    kv_cache_block_pointers=pointer_array_tensor,
                    host_kv_cache_block_pointers=host_pointer_array_tensor,
                    max_context_length=max_context_length,
                    qkv_bias=qkv_bias,
                    enable_pos_shift=enable_pos_shift)

                net._mark_output(outputs[0],
                                 'output',
                                 dtype=tensorrt_llm.str_dtype_to_trt(dtype))
                if not paged_kv_cache:
                    net._mark_output(
                        outputs[1],
                        'present_key_value',
                        dtype=tensorrt_llm.str_dtype_to_trt(kv_cache_dtype))

            inputs = {
                'input': input_tensor,
                'sequence_length': sequence_length,
                'host_past_key_value_lengths': host_past_key_value_lengths,
                'host_max_attention_window_sizes':
                host_max_attention_window_sizes,
                'host_sink_token_length': host_sink_token_length,
                'context_lengths': context_lengths,
                'cache_indirection': cache_indirection,
                'host_request_types': host_request_types
            }
            if paged_kv_cache:
                inputs['kv_cache_block_pointers'] = pointer_array
                inputs['host_kv_cache_block_pointers'] = host_pointer_array
            else:
                inputs['past_key_value'] = past_key_value

            if use_int8_kv_cache or use_fp8_kv_cache:
                inputs['kv_quant_scale'] = kv_quant_scale
                inputs['kv_dequant_scale'] = kv_dequant_scale

            if enable_remove_input_padding:
                inputs['host_context_lengths'] = host_context_lengths

            outputs = {'output': output}
            if not paged_kv_cache:
                outputs['present_key_value'] = past_key_value

            stream = torch.cuda.current_stream()
            # NOTE: when 8-bit kv cache is used together with paged kv cache no 8-bit tensors are exposed to TRT
            int8_trt_flag = use_int8_kv_cache and not paged_kv_cache
            use_fp8_kv_cache and not paged_kv_cache
            quant_mode = QuantMode.from_description(
                use_fp8_kv_cache=use_fp8_kv_cache
            ) if use_fp8_kv_cache and not paged_kv_cache else QuantMode(0)
            builder_config = builder.create_builder_config(
                name=attention_type,
                precision=dtype,
                int8=int8_trt_flag,
                quant_mode=quant_mode)

            if session is None:
                engine = builder.build_engine(net, builder_config)
                session = tensorrt_llm.runtime.Session.from_serialized_engine(
                    engine)
            session.run(inputs=inputs,
                        outputs=outputs,
                        stream=stream.cuda_stream)

            torch.cuda.synchronize()
            return session, outputs['output'], past_key_value

        hidden_size = num_heads * head_size  # embed dimension
        # If MQA/GQA and that GPTBigCodeAttention/LlamaAttention is tested, use compacted IO shape.
        # If MQA/GQA but other attention types are tested, use regular IO shape.
        # This is because GPTBigCodeAttention/LlamaAttention requires exact number of KV heads for MQA/GQA.
        # Other attention types do not support MQA/GQA natively so we emulate the effect of MQA/GQA by repeating KV heads.
        plugin_kv_num_heads = num_kv_heads if attention_type == 'llama_attention' or attention_type == 'gpt_bigcode_attention' else num_heads
        kv_hidden_size = plugin_kv_num_heads * head_size
        qkv_hidden_size = hidden_size + 2 * kv_hidden_size
        out_len = 8
        max_seq_len = in_len + 24
        sink_tokens_in_last_block = sink_token_len % tokens_per_block
        bubble_len = tokens_per_block - sink_tokens_in_last_block if sink_tokens_in_last_block > 0 else 0
        max_blocks_per_seq = math.ceil(
            (max_seq_len + bubble_len) / tokens_per_block)
        blocks = batch_size * beam_width * max_blocks_per_seq
        shape_dict = {
            'weight': (hidden_size, qkv_hidden_size),
            'bias': (qkv_hidden_size, ),
            'host_past_key_value_lengths': (batch_size, ),
            'host_max_attention_window_sizes': (1, ),
            'host_sink_token_length': (1, ),
            'sequence_length': (batch_size, ),
            'context_lengths': (batch_size, ),
            'kv_quant_scale': (1, ),
            'kv_dequant_scale': (1, ),
            'cache_indirection': (batch_size, 1, max_seq_len),
            'host_request_types': (batch_size)
        }
        if paged_kv_cache:
            shape_dict['past_key_value'] = (blocks, 2, plugin_kv_num_heads,
                                            tokens_per_block, head_size)
        else:
            shape_dict['past_key_value'] = (batch_size, 2, plugin_kv_num_heads,
                                            max_seq_len, head_size)
        shape_dict['present_key_value'] = shape_dict['past_key_value']
        if enable_remove_input_padding:
            shape_dict['host_context_lengths'] = (batch_size, )

        # HACK: pytorch does not have fp8 dtype yet
        torch_kv_cache_dtype = tensorrt_llm._utils.str_dtype_to_torch(
            'int8'
        ) if kv_cache_dtype == 'fp8' else tensorrt_llm._utils.str_dtype_to_torch(
            kv_cache_dtype)
        present_key_value = torch.zeros(shape_dict['past_key_value'],
                                        dtype=torch_kv_cache_dtype,
                                        device='cuda')
        # Init KV cache block manager
        if paged_kv_cache:
            manager = KVCacheManager([present_key_value],
                                     blocks,
                                     tokens_per_block,
                                     max_blocks_per_seq,
                                     max_seq_len,
                                     sink_token_len,
                                     beam_width=beam_width)

            # Add sequences to the manager
            for bi in range(batch_size):
                manager.add_sequence(
                    GenerationSequence(seq_idx=bi, batch_idx=bi), in_len)

        weight = torch.randn(shape_dict['weight'],
                             dtype=str_dtype_to_torch(dtype),
                             device='cuda') * 1e-3
        # FIXME: test_gpt_attention_llama_attention_False_float16_2_90_4_64_False_False_False_True
        # fails with xavier_uniform_ initialization
        # torch.nn.init.xavier_uniform_(weight)

        bias = torch.randn(shape_dict['bias'],
                           dtype=str_dtype_to_torch(dtype),
                           device='cuda') * 1e-2
        torch_present = None

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

        configuration = ConfigCls(
            hidden_size=hidden_size,
            num_hidden_layers=1,
            num_attention_heads=num_heads,
            vocab_size=51200,
            use_cache=True,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            hidden_act='gelu',
            torch_dtype=dtype,
        )

        if attention_type in ['gptj_attention', 'llama_attention']:
            configuration.rotary_dim = head_size

        if attention_type == 'llama_attention':
            configuration.num_key_value_heads = num_kv_heads
            configuration.rope_theta = rope_base
            configuration.rope_scaling = rope_scaling
            if rope_scaling is not None:
                # scaling is typically used for supporting longer seq lens than max_position_embeddings
                # so we set the max_position_embeddings to be smaller than total seq len
                # the following will use default path (no scaling) when generating half of the outputs
                # the other half will use activate the scaling
                # NOTE: in_len is also halved because the other half is treated as padding.
                #       See input_lengths below.
                configuration.max_position_embeddings = (
                    in_len // 2) + out_len - (out_len // 2)
        attention = AttentionCls(configuration).cuda().eval()
        if attention_type == 'gpt2_attention':
            attention.c_attn.weight = torch.nn.parameter.Parameter(
                data=weight.clone().detach(), requires_grad=False)
            attention.c_attn.bias = torch.nn.parameter.Parameter(
                data=bias.clone().detach(), requires_grad=False)
            attention.c_proj.weight = torch.nn.parameter.Parameter(
                data=torch.eye(hidden_size,
                               dtype=str_dtype_to_torch(dtype),
                               device='cuda'),
                requires_grad=False)
            attention.c_proj.bias = torch.nn.parameter.Parameter(
                data=torch.zeros((hidden_size, ),
                                 dtype=str_dtype_to_torch(dtype),
                                 device='cuda'),
                requires_grad=False)
            attention.layer_idx = 0
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
                data=torch.eye(hidden_size,
                               dtype=str_dtype_to_torch(dtype),
                               device='cuda'),
                requires_grad=False)
            attention.o_proj.bias = torch.nn.parameter.Parameter(
                data=torch.zeros((hidden_size, ),
                                 dtype=str_dtype_to_torch(dtype),
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
                data=torch.eye(hidden_size,
                               dtype=str_dtype_to_torch(dtype),
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
                data=torch.eye(hidden_size,
                               dtype=str_dtype_to_torch(dtype),
                               device='cuda'),
                requires_grad=False)
            attention.c_proj.bias = torch.nn.parameter.Parameter(
                data=torch.zeros((hidden_size, ),
                                 dtype=str_dtype_to_torch(dtype),
                                 device='cuda'),
                requires_grad=False)
            attention.layer_idx = 0
        else:
            raise RuntimeError("attention_type not properly set")

        input_lengths = torch.ones(
            (batch_size, ), dtype=torch.int32, device='cuda') * (in_len // 2)
        host_context_lengths = input_lengths.cpu(
        ) if enable_remove_input_padding else None
        ctx_attention_mask = torch.ones((batch_size, in_len),
                                        dtype=torch.int32,
                                        device='cuda')
        for i in range(batch_size):
            ctx_attention_mask[i, input_lengths[i]:in_len] = 0

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

        def remove_input_padding(tensor):
            batch_size = tensor.shape[0]
            tmp = []
            for b in range(batch_size):
                tmp.append(tensor[b, :in_len // 2, :])
            return torch.cat(tmp,
                             dim=1).cuda().reshape(batch_size * (in_len // 2),
                                                   -1)

        cache_indirection = torch.full((
            batch_size,
            beam_width,
            max_seq_len,
        ),
                                       0,
                                       dtype=torch.int32,
                                       device='cuda')

        def get_kv_quant_scale(torch_present):

            torch_kv = torch.cat((torch_present[0], torch_present[1]))
            kv_dequant_scale = torch.tensor([torch.max(torch_kv).item() / 127],
                                            dtype=torch.float32,
                                            device='cuda').reshape(
                                                shape_dict['kv_dequant_scale'])

            # fp8 kv cache uses 1.0f scale.
            if not use_int8_kv_cache:
                kv_dequant_scale = torch.tensor(
                    [1.0], dtype=torch.float32,
                    device='cuda').reshape(shape_dict['kv_dequant_scale'])

            kv_quant_scale = 1.0 / kv_dequant_scale
            return kv_dequant_scale, kv_quant_scale

        def verify_kv_cache(torch_present):
            if not use_int8_kv_cache and not use_fp8_kv_cache and num_kv_heads == num_heads and beam_width == 1:
                if paged_kv_cache:
                    kv_cache_cont = manager.blocks_manager.get_continous_caches(
                        0)
                    kv_cache_cont = kv_cache_cont.permute(1, 0, 2)
                else:
                    kv_cache_cont = present_key_value
                    kv_cache_cont = kv_cache_cont.permute(1, 0, 2, 3, 4)

                key, value = kv_cache_cont.to(torch.float32).chunk(2)

                if paged_kv_cache:
                    # TRT-LLM after paged KV cache it comes with blocks
                    # K cache has shape: [batch_size, max_blocks_per_seq, num_kv_heads, tokens_per_block, head_size]
                    key = key.reshape(batch_size, max_blocks_per_seq,
                                      num_kv_heads, tokens_per_block, head_size)
                    key = key.permute(0, 2, 1, 3, 4).reshape(
                        batch_size, num_kv_heads,
                        max_blocks_per_seq * tokens_per_block, head_size)
                else:
                    key = key.reshape(batch_size, num_kv_heads, max_seq_len,
                                      head_size)

                # Note K and V shares the same layout now.
                if paged_kv_cache:
                    # TRT-LLM after paged KV cache it comes with blocks
                    # V cache has shape: [batch_size, max_blocks_per_seq, num_kv_heads, tokens_per_block, head_size]
                    value = value.reshape(batch_size, max_blocks_per_seq,
                                          num_kv_heads, tokens_per_block,
                                          head_size)
                    value = value.permute(0, 2, 1, 3, 4).reshape(
                        batch_size, num_kv_heads,
                        max_blocks_per_seq * tokens_per_block, head_size)
                else:
                    value = value.reshape(batch_size, num_kv_heads, max_seq_len,
                                          head_size)

                tols = {
                    "float32": 2e-04,
                    "float16": 2e-04,
                    "bfloat16": 2e-01,
                }

                np.testing.assert_allclose(
                    key.cpu().numpy()[:, :, :in_len // 2, :],
                    torch_present[0].to(
                        torch.float32).cpu().numpy()[:, :, :in_len // 2, :],
                    atol=tols[dtype],
                    rtol=tols[dtype])
                np.testing.assert_allclose(
                    value.cpu().numpy()[:, :, :in_len // 2, :],
                    torch_present[1].to(
                        torch.float32).cpu().numpy()[:, :, :in_len // 2, :],
                    atol=tols[dtype],
                    rtol=tols[dtype])

        max_context_length = in_len // 2 if enable_remove_input_padding else in_len
        for step in range(out_len):
            # The sequence_lengths = context_lengths + step for generation stage.
            sequence_length = torch.add(input_lengths, step)

            pointer_array = None
            if paged_kv_cache:
                # Get arrays of pointers to the "pages" of KV values
                pointer_array = manager.get_pointer_arrays(beam_width)[0]

            if step == 0:
                host_request_types = torch.tensor([0] * batch_size,
                                                  dtype=torch.int32)
                if paged_kv_cache:
                    # Reassemble pointer array to have KV cache for bs context invocations instead of batch_beam
                    pointer_array = pointer_array[:, 0, :, :]
                    pointer_array = pointer_array.reshape(
                        batch_size, 1, 2, max_blocks_per_seq)

                # Context stage
                shape_dict['input'] = (batch_size, in_len, hidden_size)
                shape_dict['output'] = shape_dict['input']
                host_past_key_value_lengths = torch.tensor([0] * batch_size,
                                                           dtype=torch.int32)
                host_max_attention_window_sizes = torch.tensor(
                    [max_seq_len], dtype=torch.int32)
                host_sink_token_length = torch.tensor([0], dtype=torch.int32)

                input_tensor = torch.randn(shape_dict['input'],
                                           dtype=str_dtype_to_torch(dtype),
                                           device='cuda') * 1e-3

                # torch execution
                position_ids = ctx_attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(ctx_attention_mask == 0, 1)

                attention_mask = _prepare_4d_attention_mask(
                    ctx_attention_mask,
                    dtype=str_dtype_to_torch(dtype),
                    tgt_len=in_len)
                if attention_type == 'gpt2_attention':
                    torch_output, torch_present = attention(
                        input_tensor,
                        layer_past=None,
                        use_cache=True,
                        attention_mask=attention_mask)
                elif attention_type == 'llama_attention':
                    attention_mask = attention_mask + AttentionMaskConverter._make_causal_mask(
                        input_tensor.shape[:2],
                        dtype=str_dtype_to_torch(dtype),
                        device='cuda',
                        past_key_values_length=0)
                    torch_output, _, torch_present = attention(
                        input_tensor,
                        past_key_value=DynamicCache(),
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        use_cache=True)
                    torch_present = torch_present.to_legacy_cache()
                elif attention_type == 'gptj_attention':
                    torch_output, torch_present = attention(
                        input_tensor,
                        layer_past=None,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        use_cache=True)
                elif attention_type == 'gpt_bigcode_attention':
                    attention_mask = _prepare_4d_attention_mask(
                        ctx_attention_mask,
                        dtype=str_dtype_to_torch(dtype),
                        tgt_len=in_len)
                    # source shape = (b, 1, s_query, s_key)
                    # target shape = (b, s_query, h, s_key)
                    attention_mask = (attention_mask >= 0).permute(
                        [0, 2, 1, 3]).expand(batch_size, in_len, num_heads,
                                             in_len)
                    torch_output, torch_present = attention(
                        input_tensor,
                        layer_past=None,
                        attention_mask=attention_mask,
                        use_cache=True)
                else:
                    raise RuntimeError("attention_type not properly set")

                torch.cuda.synchronize()

                if attention_type == 'llama_attention':
                    kv_dequant_scale, kv_quant_scale = get_kv_quant_scale(
                        torch_present[0])
                else:
                    kv_dequant_scale, kv_quant_scale = get_kv_quant_scale(
                        torch_present)

                if enable_remove_input_padding:
                    shape_dict['input'] = (batch_size * (in_len // 2),
                                           hidden_size)
                    input_tensor = remove_input_padding(input_tensor)

                shape_dict['output'] = shape_dict['input']
                output = torch.zeros(shape_dict['output'],
                                     dtype=str_dtype_to_torch(dtype),
                                     device='cuda')

                session, output, present_key_value = _construct_execution(
                    session, input_tensor, weight_plugin, bias_plugin,
                    present_key_value, pointer_array, sequence_length,
                    host_past_key_value_lengths,
                    host_max_attention_window_sizes, host_sink_token_length,
                    input_lengths, host_context_lengths, cache_indirection,
                    host_request_types, num_heads, hidden_size, num_kv_heads,
                    output, dtype, max_context_length, shape_dict,
                    kv_quant_scale, kv_dequant_scale, configuration)
                del session
                session = None

                # Note: Volta has larger errors.
                # We speculate it’s because Volta’s TC is smaller and more calculations are required,
                # which may lead to more error accumulation.
                if enable_remove_input_padding:
                    torch_output = remove_input_padding(torch_output)
                    np.testing.assert_allclose(
                        output.to(torch.float32).cpu().numpy(),
                        torch_output.to(torch.float32).cpu().numpy(),
                        atol=5e-3 if getSMVersion() > 70 else 5e-2)
                else:
                    np.testing.assert_allclose(
                        output[:, :in_len // 2, :].to(
                            torch.float32).cpu().numpy(),
                        torch_output[:, :in_len // 2, :].to(
                            torch.float32).cpu().numpy(),
                        atol=5e-3 if getSMVersion() > 70 else 5e-2)
                if attention_type == 'llama_attention':
                    verify_kv_cache(torch_present[0])
                else:
                    verify_kv_cache(torch_present)

            else:
                # Generation stage
                shape_dict['input'] = (batch_size, 1, hidden_size)
                host_past_key_value_lengths = sequence_length.cpu() - 1
                host_max_attention_window_sizes = torch.tensor(
                    [max_seq_len], dtype=torch.int32)
                host_sink_token_length = torch.tensor([0], dtype=torch.int32)
                input_tensor = torch.randn(shape_dict['input'],
                                           dtype=str_dtype_to_torch(dtype),
                                           device='cuda') * 1e-3

                host_request_types = torch.tensor([1] * batch_size,
                                                  dtype=torch.int32)

                ctx_attention_mask = torch.cat((ctx_attention_mask,
                                                ctx_attention_mask.new_ones(
                                                    (batch_size, 1))),
                                               dim=-1).contiguous()

                position_ids = ctx_attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(ctx_attention_mask == 0, 1)
                position_ids = position_ids[:, -1].unsqueeze(-1)

                attention_mask = _prepare_4d_attention_mask(
                    ctx_attention_mask,
                    dtype=str_dtype_to_torch(dtype),
                    tgt_len=1)

                # torch execution
                if attention_type == 'gpt2_attention':
                    torch_output, torch_present = attention(
                        input_tensor,
                        layer_past=torch_present,
                        use_cache=True,
                        attention_mask=attention_mask)
                elif attention_type == 'llama_attention':
                    attention_mask = attention_mask + AttentionMaskConverter._make_causal_mask(
                        input_tensor.shape[:2],
                        dtype=str_dtype_to_torch(dtype),
                        device='cuda',
                        past_key_values_length=in_len + step - 1)
                    # llama uses DynamicCache
                    torch_output, _, torch_present = attention(
                        input_tensor,
                        past_key_value=DynamicCache.from_legacy_cache(
                            torch_present),
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        use_cache=True)
                    torch_present = torch_present.to_legacy_cache()
                elif attention_type == 'gptj_attention':
                    torch_output, torch_present = attention(
                        input_tensor,
                        layer_past=torch_present,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        use_cache=True)
                elif attention_type == 'gpt_bigcode_attention':
                    # source shape = (b, 1, 1, s_key)
                    # target shape = (b, 1, h, s_key)
                    key_seqlen = in_len + step  # ctx_attention_mask.shape[1]
                    attention_mask = (attention_mask >= 0).permute(
                        [0, 2, 1, 3]).expand(batch_size, 1, num_heads,
                                             key_seqlen)
                    torch_output, torch_present = attention(
                        input_tensor,
                        layer_past=torch_present,
                        use_cache=True,
                        attention_mask=attention_mask)

                def tile_beam_width(tensor: torch.Tensor, num_beams: int):
                    if num_beams == 1:
                        return tensor
                    else:
                        new_shape = np.array(tensor.shape)
                        new_shape[0] = new_shape[0] * num_beams
                        tile_size = np.ones(new_shape.shape, dtype=np.int32)
                        tile_size = np.insert(tile_size, 1, num_beams)
                        new_tensor = torch.unsqueeze(tensor, 1)
                        new_tensor = new_tensor.tile(tile_size.tolist())
                        new_tensor = new_tensor.reshape(new_shape.tolist())
                        return new_tensor

                torch_output = tile_beam_width(torch_output, beam_width)
                torch_output = torch_output.reshape(
                    [batch_size, beam_width, -1])

                torch.cuda.synchronize()

                tiled_input_tensor = tile_beam_width(input_tensor, beam_width)
                tiled_attention_mask = tile_beam_width(attention_mask,
                                                       beam_width)
                tiled_input_lengths = tile_beam_width(input_lengths, beam_width)
                tiled_host_context_lengths = tiled_input_lengths.cpu(
                ) if enable_remove_input_padding else None
                tiled_host_past_key_value_lengths = tile_beam_width(
                    host_past_key_value_lengths, beam_width)
                tiled_host_request_types = tile_beam_width(
                    host_request_types, beam_width)
                tiled_present_key_value = tile_beam_width(
                    present_key_value,
                    beam_width) if not paged_kv_cache else present_key_value
                tiled_sequence_length = tile_beam_width(sequence_length,
                                                        beam_width)

                if enable_remove_input_padding:
                    shape_dict['input'] = (batch_size, hidden_size)
                    input_tensor = input_tensor.view(shape_dict['input'])

                # TRT LLM execution
                shape_dict['output'] = shape_dict['input']
                output = torch.zeros(shape_dict['output'],
                                     dtype=str_dtype_to_torch(dtype),
                                     device='cuda')

                input_tensor = input_tensor.reshape([batch_size, hidden_size])
                tiled_input_tensor = tile_beam_width(input_tensor, beam_width)
                tiled_input_tensor = tiled_input_tensor.reshape(
                    [batch_size * beam_width, 1, hidden_size])
                output = output.reshape([batch_size, hidden_size])
                tiled_output = tile_beam_width(output, beam_width)
                tiled_output = tiled_output.reshape(
                    [batch_size * beam_width, 1, hidden_size])

                session, tiled_output, present_key_value = _construct_execution(
                    session, tiled_input_tensor, weight_plugin, bias_plugin,
                    tiled_present_key_value, pointer_array,
                    tiled_sequence_length, tiled_host_past_key_value_lengths,
                    host_max_attention_window_sizes, host_sink_token_length,
                    tiled_input_lengths, tiled_host_context_lengths,
                    cache_indirection, tiled_host_request_types, num_heads,
                    hidden_size, num_kv_heads, tiled_output, dtype,
                    max_context_length, shape_dict, kv_quant_scale,
                    kv_dequant_scale, configuration)

                del session
                session = None

                # compare result
                np.testing.assert_allclose(
                    torch.flatten(tiled_output).to(torch.float32).cpu().numpy(),
                    torch.flatten(torch_output).to(torch.float32).cpu().numpy(),
                    atol=2e-3)

            if paged_kv_cache:
                # Iterate to the next step. Increase number of tokens for all unfinished sequences
                # And allocate new blocks if needed
                manager.step([False] * batch_size)
        # assert False, "Force fail"
        return


if __name__ == "__main__":
    unittest.main()
