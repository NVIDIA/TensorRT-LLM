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

import torch
from flash_attn.flash_attn_interface import flash_attn_func
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner
from utils.util import skip_non_hopper_unittest, unittest_name_func

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.plugin.plugin import ContextFMHAType


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def load_test_cases():
        test_cases = [(3, 9279, 24, 128, ContextFMHAType.enabled_with_fp32_acc,
                       "bfloat16", 64, 64, 256),
                      (3, 9279, 24, 80, ContextFMHAType.enabled_with_fp32_acc,
                       "bfloat16", 64, 64, 256),
                      (3, 9279, 24, 72, ContextFMHAType.enabled_with_fp32_acc,
                       "bfloat16", 64, 64, 256)]
        return test_cases

    @parameterized.expand(load_test_cases, name_func=unittest_name_func)
    @skip_non_hopper_unittest
    def test_sage_attention(self, batch_size, in_len, num_heads, head_size,
                            context_fmha_type, dtype, q_quant_block_size,
                            k_quant_block_size, v_quant_block_size):

        def _construct_execution(input_tensor, input_lengths, num_heads,
                                 hidden_size, dtype):
            head_size = hidden_size // num_heads
            # construct trt network
            builder = tensorrt_llm.Builder()
            builder.strongly_typed = False  # Test need to run in weekly typed mode
            net = builder.create_network()
            net.plugin_config.to_legacy_setting()
            net.plugin_config.bert_attention_plugin = dtype
            net.plugin_config.set_context_fmha(context_fmha_type)
            with tensorrt_llm.net_guard(net):
                network = tensorrt_llm.default_trtnet()
                qkv = Tensor(name='input',
                             shape=tuple(input_tensor.shape),
                             dtype=tensorrt_llm.str_dtype_to_trt(dtype))
                input_lengths_tensor = Tensor(
                    name='input_lengths',
                    shape=tuple(input_lengths.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))

                # sage attention
                outputs = tensorrt_llm.functional.bert_attention(
                    qkv,
                    input_lengths_tensor,
                    num_heads=num_heads,
                    head_size=head_size,
                    q_scaling=1.0,
                    sage_attn=True,
                    sage_attn_q_block_size=q_quant_block_size,
                    sage_attn_k_block_size=k_quant_block_size,
                    sage_attn_v_block_size=v_quant_block_size)

                network.mark_output(outputs.trt_tensor)
                outputs.trt_tensor.name = 'output'
                outputs.trt_tensor.dtype = tensorrt_llm.str_dtype_to_trt(dtype)

            engine = EngineFromNetwork(
                (builder.trt_builder, net.trt_network),
                config=CreateConfig(bf16=(dtype == 'bfloat16')))

            with TrtRunner(engine) as runner:
                outputs = runner.infer(feed_dict={
                    'input': input_tensor,
                    'input_lengths': input_lengths
                })

            return outputs['output']

        hidden_size = num_heads * head_size

        shape_dict = {
            'input_lengths': (batch_size, ),
        }

        input_lengths = torch.ones(
            (batch_size, ), dtype=torch.int32, device='cuda') * in_len

        shape_dict['input'] = (batch_size, in_len, 3, hidden_size)
        shape_dict['output'] = (batch_size, in_len, 1, hidden_size)

        input_tensor = torch.randn(
            shape_dict['input'],
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device='cuda')

        output = _construct_execution(input_tensor, input_lengths, num_heads,
                                      hidden_size, dtype)
        output = output.reshape(batch_size, in_len, num_heads, head_size)
        output_norm = torch.norm(output.reshape(-1).to(
            torch.float32)).to("cuda")
        print("trt_norm: ", output_norm)

        # flash attention output
        q = input_tensor[:, :, 0, :].squeeze().reshape(batch_size, in_len,
                                                       num_heads, head_size)
        k = input_tensor[:, :, 1, :].squeeze().reshape(batch_size, in_len,
                                                       num_heads, head_size)
        v = input_tensor[:, :, 2, :].squeeze().reshape(batch_size, in_len,
                                                       num_heads, head_size)
        flash_attn_output = flash_attn_func(q, k, v)
        flash_attn_norm = torch.norm(
            flash_attn_output.reshape(-1).to(torch.float32))
        print("flash_attn_norm: ", flash_attn_norm)

        cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_similarity = cos_sim(
            output.to("cuda").reshape(-1).to(torch.float32),
            flash_attn_output.reshape(-1).to(torch.float32))
        print("cos_similarity: ", cos_similarity)

        if cos_similarity < 0.98 or cos_similarity == 1.0:
            raise Exception(
                "cos_similarity lower than expected or equal to 1.0!")


if __name__ == "__main__":
    unittest.main()
