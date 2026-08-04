# # SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: Apache-2.0
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
import os
import unittest
from itertools import product

import numpy as np
import torch
from parameterized import parameterized
from utils.util import create_session, run_session

import tensorrt_llm
from tensorrt_llm import Tensor


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    # TODO rank 1 is not supported now
    @parameterized.expand(
        list(
            product([[1], [2], [4], [2, 4], [8], [16], [8, 16], [1, 2, 4],
                     [1, 2, 4, 8, 16]])))
    def test_ranks(self, lora_ranks_list):
        print(f"[INFO] test lora_ranks_list: {lora_ranks_list}")
        os.environ['LORA_USE_UNIFIED_GEMM'] = 'OFF'
        torch.random.manual_seed(0)

        dtype = 'float16'
        torch_dtype = torch.float16
        device = 'cuda'

        batch_size = len(lora_ranks_list)
        input_length = 32
        hidden_size = 4096
        input_data = [
            torch.randn(input_length, hidden_size,
                        device=device).to(torch_dtype) * 0.1
            for _ in range(batch_size)
        ]
        lora_weight_ins = [
            torch.randn(hidden_size, lora_rank, device=device).to(torch_dtype) *
            0.1 for lora_rank in lora_ranks_list
        ]
        lora_weight_outs = [
            torch.randn(lora_rank, hidden_size, device=device).to(torch_dtype) *
            0.1 for lora_rank in lora_ranks_list
        ]
        host_context_lengths = torch.Tensor(
            [input_length for _ in range(batch_size)]).to(torch.int32)
        lora_ranks = torch.Tensor(lora_ranks_list).to(torch.int32)

        ref_data = [
            torch.matmul(torch.matmul(input, in_weight),
                         out_weight) for input, in_weight, out_weight in zip(
                             input_data, lora_weight_ins, lora_weight_outs)
        ]

        lora_weight_ins = [
            tmp.transpose(1, 0).contiguous() for tmp in lora_weight_ins
        ]
        lora_weight_outs = [
            tmp.transpose(1, 0).contiguous() for tmp in lora_weight_outs
        ]

        lora_weights_pointers = []
        for in_ptr, out_ptr in zip(lora_weight_ins, lora_weight_outs):
            lora_weights_pointers.append(in_ptr.data_ptr())
            lora_weights_pointers.append(out_ptr.data_ptr())
            # null dora scale
            lora_weights_pointers.append(0)

        lora_weights_pointers = torch.LongTensor(lora_weights_pointers).to(
            torch.int64).reshape([batch_size, 3])
        host_request_types = torch.zeros_like(host_context_lengths,
                                              device='cpu').int()

        concat_input_data = torch.concat(input_data).contiguous().to(device)

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        network.plugin_config.set_lora_plugin(dtype)
        with tensorrt_llm.net_guard(network):

            input_tensor = Tensor(name='input_tensor',
                                  shape=concat_input_data.shape,
                                  dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            host_request_types_tensor = Tensor(
                name='host_request_types',
                shape=[batch_size],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            host_context_lengths_tensor = Tensor(
                name='host_context_lengths',
                shape=[batch_size],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            lora_ranks_tensor = Tensor(
                name='lora_ranks',
                shape=[batch_size],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            lora_weights_pointers_tensor = Tensor(
                name='lora_weights_pointers',
                shape=[batch_size, 3],
                dtype=tensorrt_llm.str_dtype_to_trt('int64'))

            output = tensorrt_llm.functional.lora_plugin(
                input_tensor,
                hidden_size,
                [hidden_size],
                host_request_types_tensor,
                False,
                True,
                host_context_lengths_tensor,
                max(max(lora_ranks_list), 8),
                [lora_ranks_tensor],
                [lora_weights_pointers_tensor],
                weight_index=0,
            )
            output.mark_output('output')

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {
            'input_tensor': concat_input_data,
            'host_request_types': host_request_types,
            'host_context_lengths': host_context_lengths,
            'lora_ranks': lora_ranks,
            'lora_weights_pointers': lora_weights_pointers,
        }
        outputs = run_session(session, inputs)

        # pytorch run
        ref_data = torch.concat(ref_data)
        # compare diff

        dtype_atol = {"float16": 1e-2, "float32": 2e-3, "bfloat16": 1e-1}

        np.testing.assert_allclose(ref_data.to(torch.float32).cpu().numpy(),
                                   outputs['output'].to(
                                       torch.float32).cpu().numpy(),
                                   atol=dtype_atol[dtype])
