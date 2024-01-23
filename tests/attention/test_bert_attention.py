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
import os
import sys
import unittest
from itertools import product

import numpy as np
import torch
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.plugin.plugin import ContextFMHAType

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import getSMVersion


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def load_test_cases():
        test_cases = [(1, 128, 12, 64, False, 'float32')]
        test_cases += list(
            product([1, 8], [64, 256, 512, 1024], [16], [32, 64], [
                ContextFMHAType.disabled, ContextFMHAType.enabled,
                ContextFMHAType.enabled_with_fp32_acc
            ], ['float16']))
        return test_cases

    def custom_name_func(testcase_func, param_num, param):
        return "%s_%s" % (
            testcase_func.__name__,
            parameterized.to_safe_name("_".join(str(x) for x in param.args)),
        )

    @parameterized.expand(load_test_cases, name_func=custom_name_func)
    def test_bert_attention(self, batch_size, in_len, num_heads, head_size,
                            context_fmha_type, dtype):

        if getSMVersion() < 80:
            if context_fmha_type == ContextFMHAType.enabled_with_fp32_acc:
                self.skipTest(
                    "ContextFMHAType with fp32 acc is not supported in pre-ampere architecture"
                )

        def _construct_execution(input_tensor, weight, bias, input_lengths,
                                 num_heads, hidden_size, output, dtype,
                                 shape_dict):
            head_size = hidden_size // num_heads
            # construct trt network
            builder = tensorrt_llm.Builder()
            net = builder.create_network()
            net.plugin_config.set_bert_attention_plugin(dtype)
            net.plugin_config.set_context_fmha(context_fmha_type)
            with tensorrt_llm.net_guard(net):
                network = tensorrt_llm.default_trtnet()
                x_tensor = Tensor(name='input',
                                  shape=tuple(input_tensor.shape),
                                  dtype=tensorrt_llm.str_dtype_to_trt(dtype))
                input_lengths_tensor = Tensor(
                    name='input_lengths',
                    shape=tuple(input_lengths.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))

                # qkv projection
                linear = tensorrt_llm.layers.Linear(hidden_size,
                                                    hidden_size * 3,
                                                    bias=True)
                linear.weight.value = np.ascontiguousarray(
                    weight.cpu().numpy().transpose())
                linear.bias.value = bias.cpu().numpy()
                qkv = linear(x_tensor)

                # attention (padding mask)
                outputs = tensorrt_llm.functional.bert_attention(
                    qkv,
                    input_lengths_tensor,
                    num_heads=num_heads,
                    head_size=head_size,
                    q_scaling=1.0)

                network.mark_output(outputs.trt_tensor)
                outputs.trt_tensor.name = 'output'
                outputs.trt_tensor.dtype = tensorrt_llm.str_dtype_to_trt(dtype)

            engine = EngineFromNetwork(
                (builder.trt_builder, net.trt_network),
                config=CreateConfig(fp16=(dtype == 'float16')))

            with TrtRunner(engine) as runner:
                outputs = runner.infer(feed_dict={
                    'input': input_tensor,
                    'input_lengths': input_lengths
                })

            return outputs['output']

        hidden_size = num_heads * head_size
        shape_dict = {
            'weight': (hidden_size, hidden_size * 3),
            'bias': (hidden_size * 3, ),
            'input_lengths': (batch_size, ),
        }

        weight = torch.empty(
            shape_dict['weight'],
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device='cuda') * 1e-3
        torch.nn.init.xavier_uniform_(weight)
        bias = torch.randn(shape_dict['bias'],
                           dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                           device='cuda') * 1e-2

        ConfigCls = BertConfig
        AttentionCls = BertSelfAttention

        configuration = ConfigCls(
            hidden_size=hidden_size,
            num_hidden_layers=1,
            num_attention_heads=num_heads,
            vocab_size=30522,
            hidden_act='gelu',
            torch_dtype=dtype,
        )
        attention = AttentionCls(configuration).cuda().eval()

        query, key, value = torch.split(weight, hidden_size, dim=-1)
        q_bias, k_bias, v_bias = torch.split(bias, hidden_size, dim=0)
        attention.query.weight = torch.nn.parameter.Parameter(
            data=query.clone().detach(), requires_grad=False)
        attention.key.weight = torch.nn.parameter.Parameter(
            data=key.clone().detach(), requires_grad=False)
        attention.value.weight = torch.nn.parameter.Parameter(
            data=value.clone().detach(), requires_grad=False)
        attention.query.bias = torch.nn.parameter.Parameter(
            data=q_bias.clone().detach(), requires_grad=False)
        attention.key.bias = torch.nn.parameter.Parameter(
            data=k_bias.clone().detach(), requires_grad=False)
        attention.value.bias = torch.nn.parameter.Parameter(
            data=v_bias.clone().detach(), requires_grad=False)

        input_lengths = torch.ones(
            (batch_size, ), dtype=torch.int32, device='cuda') * in_len

        # Context stage
        shape_dict['input'] = (batch_size, in_len, hidden_size)
        shape_dict['output'] = shape_dict['input']

        input_tensor = torch.randn(
            shape_dict['input'],
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device='cuda') * 1e-3
        output = torch.zeros(
            shape_dict['output'],
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device='cuda')
        output = _construct_execution(input_tensor, weight, bias, input_lengths,
                                      num_heads, hidden_size, output, dtype,
                                      shape_dict)

        # torch execution
        torch_output = attention(input_tensor)[0]

        np.testing.assert_allclose(output.cpu().numpy(),
                                   torch_output.cpu().numpy(),
                                   atol=1e-3)


if __name__ == "__main__":
    unittest.main()
