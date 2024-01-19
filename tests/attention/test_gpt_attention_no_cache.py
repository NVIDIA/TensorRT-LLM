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
import unittest
from collections import OrderedDict

# isort: off
import torch
import tensorrt as trt
# isort: on
from parameterized import parameterized

import tensorrt_llm
from tensorrt_llm import Tensor, str_dtype_to_trt
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.functional import gpt_attention
from tensorrt_llm.models.generation_mixin import GenerationMixin
from tensorrt_llm.plugin.plugin import ContextFMHAType


class TestPluginNoCache(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('info')

    @staticmethod
    def build_engine(qkv_shape,
                     max_batch_size,
                     max_beam_width,
                     max_input_len,
                     max_new_tokens,
                     num_kv_heads,
                     head_size,
                     dtype,
                     num_layers,
                     remove_input_padding,
                     context_fmha_type,
                     use_cache=True):
        kv_dtype = str_dtype_to_trt(dtype)
        hidden_size = num_kv_heads * head_size
        num_tokens = max_batch_size * max_input_len

        builder = tensorrt_llm.Builder()
        builder_config = builder.create_builder_config(
            name="attention",
            precision=dtype,
        )
        net = builder.create_network()
        net.plugin_config.set_gpt_attention_plugin(dtype)
        net.plugin_config.set_context_fmha(context_fmha_type)
        net.plugin_config.remove_input_padding = remove_input_padding
        with tensorrt_llm.net_guard(net):
            inputs = GenerationMixin().prepare_attention_inputs(
                max_batch_size,
                max_beam_width,
                max_input_len,
                max_new_tokens,
                num_kv_heads,
                head_size,
                num_layers,
                kv_dtype,
                remove_input_padding,
                use_gpt_attention_plugin=True,
                use_gemm_plugin=
                True,  # because we don't want two optimization profiles
                use_cache=use_cache,
            )

            if remove_input_padding:
                qkv = Tensor(name="qkv",
                             shape=(-1, hidden_size * 3),
                             dtype=str_dtype_to_trt(dtype),
                             dim_range=OrderedDict([
                                 ('tokens', [(1, num_tokens // 2, num_tokens)]),
                                 ('hidden_size', [hidden_size * 3]),
                             ]))
            else:
                qkv = Tensor(name="qkv",
                             shape=(-1, -1, hidden_size * 3),
                             dtype=str_dtype_to_trt(dtype),
                             dim_range=OrderedDict([
                                 ('batch_size', [(1, max_batch_size // 2,
                                                  max_batch_size)]),
                                 ('tokens', [(1, max_input_len // 2,
                                              max_input_len)]),
                                 ('hidden_size', [hidden_size * 3]),
                             ]))

            sequence_length = inputs['sequence_length']
            host_context_lengths = inputs['host_context_lengths']
            host_max_attention_window_sizes = inputs[
                'host_max_attention_window_sizes'][0]
            host_sink_token_length = inputs['host_sink_token_length']
            context_lengths = inputs['context_lengths']
            host_request_types = inputs['host_request_types']

            host_past_key_value_lengths = inputs['host_past_key_value_lengths']
            past_key_value = inputs['past_key_value']
            if past_key_value:
                past_key_value = past_key_value[0]
            cache_indirection = inputs['cache_indirection']

            outputs = gpt_attention(
                qkv=qkv,
                past_key_value=past_key_value,
                sequence_length=sequence_length,
                host_past_key_value_lengths=host_past_key_value_lengths,
                host_max_attention_window_sizes=host_max_attention_window_sizes,
                host_sink_token_length=host_sink_token_length,
                context_lengths=context_lengths,
                cache_indirection=cache_indirection,
                host_request_types=host_request_types,
                num_heads=num_kv_heads,
                num_kv_heads=num_kv_heads,
                hidden_size_per_head=head_size,
                q_scaling=1.0,
                rotary_embedding_dim=0,
                max_context_length=max_input_len,
                host_context_lengths=host_context_lengths,
                use_cache=use_cache,
            )

            net._mark_output(outputs[0],
                             'output',
                             dtype=str_dtype_to_trt(dtype))
            if use_cache:
                net._mark_output(outputs[1],
                                 'present_key_value',
                                 dtype=str_dtype_to_trt(dtype))

        return builder.build_engine(net, builder_config)

    @parameterized.expand([("float16", True, ContextFMHAType.disabled),
                           ("float16", False, ContextFMHAType.enabled)])
    def test_plugin_no_cache(self, dtype: str, remove_input_padding: bool,
                             fmha_type: ContextFMHAType):

        torch_dtype_to_trt = {
            torch.float16: trt.float16,
            torch.float32: trt.float32,
            torch.int32: trt.int32
        }

        max_batch_size = 8
        max_beam_width = 1
        max_input_len = 128
        max_new_tokens = 0
        num_kv_heads = 16
        head_size = 32
        num_layers = 1
        hidden_size = num_kv_heads * head_size
        str_dtype_to_trt(dtype)

        if remove_input_padding:
            qkv_shape = (max_batch_size * max_input_len, hidden_size * 3)
            out_shape = (max_batch_size * max_input_len, hidden_size)
        else:
            qkv_shape = (max_batch_size, max_input_len, hidden_size * 3)
            out_shape = (max_batch_size, max_input_len, hidden_size)

        qkv = torch.randn(
            qkv_shape, dtype=str_dtype_to_torch(dtype), device="cuda") * 1e-3
        sequence_length = torch.full([max_batch_size],
                                     max_input_len,
                                     dtype=torch.int32).cuda()
        host_past_key_value_lengths = torch.zeros([max_batch_size],
                                                  dtype=torch.int32).cpu()
        host_max_attention_window_sizes = torch.tensor([max_input_len],
                                                       dtype=torch.int32).cpu()
        host_sink_token_length = torch.tensor([0], dtype=torch.int32).cpu()
        context_lengths = torch.full([max_batch_size],
                                     max_input_len,
                                     dtype=torch.int32).cuda()
        cache_indirection = torch.zeros(
            [max_batch_size, max_beam_width, max_input_len],
            dtype=torch.int32,
            device='cuda')
        host_request_types = torch.zeros([max_batch_size],
                                         dtype=torch.int32).cpu()
        host_context_lengths = torch.full([max_batch_size],
                                          max_input_len,
                                          dtype=torch.int32).cpu()

        present_key_value = torch.zeros(
            [max_batch_size, 2, num_kv_heads, max_input_len, head_size],
            dtype=str_dtype_to_torch(dtype),
            device='cuda')
        output = torch.zeros(out_shape,
                             dtype=str_dtype_to_torch(dtype),
                             device="cuda")
        output_nocache = torch.zeros(out_shape,
                                     dtype=str_dtype_to_torch(dtype),
                                     device="cuda")

        engine = TestPluginNoCache.build_engine(qkv_shape,
                                                max_batch_size,
                                                max_beam_width,
                                                max_input_len,
                                                max_new_tokens,
                                                num_kv_heads,
                                                head_size,
                                                dtype,
                                                num_layers,
                                                remove_input_padding,
                                                fmha_type,
                                                use_cache=False)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)
        inputs = {
            'qkv': qkv,
            'host_max_attention_window_size_0': host_max_attention_window_sizes,
            'host_sink_token_length': host_sink_token_length,
            'context_lengths': context_lengths,
            'host_request_types': host_request_types,
        }
        if remove_input_padding:
            inputs['host_context_lengths'] = host_context_lengths
        outputs = {
            'output': output_nocache,
        }
        inputs_info = [
            tensorrt_llm.runtime.TensorInfo(name,
                                            torch_dtype_to_trt[tensor.dtype],
                                            tensor.shape)
            for name, tensor in inputs.items()
        ]
        session.infer_shapes(inputs_info)
        stream = torch.cuda.current_stream()
        session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)

        engine = TestPluginNoCache.build_engine(qkv_shape, max_batch_size,
                                                max_beam_width, max_input_len,
                                                max_new_tokens, num_kv_heads,
                                                head_size, dtype, num_layers,
                                                remove_input_padding, fmha_type)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)

        inputs = {
            'qkv': qkv,
            'sequence_length': sequence_length,
            'host_past_key_value_lengths': host_past_key_value_lengths,
            'host_max_attention_window_size_0': host_max_attention_window_sizes,
            'host_sink_token_length': host_sink_token_length,
            'context_lengths': context_lengths,
            'cache_indirection': cache_indirection,
            'host_request_types': host_request_types,
            'past_key_value_0': present_key_value,
        }
        if remove_input_padding:
            inputs['host_context_lengths'] = host_context_lengths

        outputs = {
            'output': output,
            'present_key_value': present_key_value,
        }

        inputs_info = [
            tensorrt_llm.runtime.TensorInfo(name,
                                            torch_dtype_to_trt[tensor.dtype],
                                            tensor.shape)
            for name, tensor in inputs.items()
        ]
        session.infer_shapes(inputs_info)
        stream = torch.cuda.current_stream()
        session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)

        assert torch.equal(output, output_nocache)


if __name__ == "__main__":
    unittest.main()
