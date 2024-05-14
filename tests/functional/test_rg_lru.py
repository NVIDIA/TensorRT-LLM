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
import pytest
import torch
from parameterized import parameterized
from torch_ref import rg_lru_batch_ref

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import skip_bf16_pre_ampere, unittest_name_func


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand(list(
        product([2560], ['context', 'generation'],
                ['float32', 'float16', 'bfloat16'], [3], [16], [False, True],
                [True, False], [True, False])),
                          name_func=unittest_name_func)
    def test_rg_lru(self, dim, req_type, dtype, batch_size, max_seq_len,
                    remove_padding, has_y, has_y_bias):
        # Skip tests that are not supported in pre-ampere architecture
        skip_bf16_pre_ampere(dtype)

        # Skip cases of has_y = False but has_y_bias = True
        if not has_y and has_y_bias:
            pytest.skip(
                "Skipping test cases with has_y == False and has_y_bias == True."
            )

        # configs
        device = "cuda"
        seq_len = max_seq_len if req_type == 'context' else 1
        torch_dtype = str_dtype_to_torch(dtype)

        # test data
        torch.random.manual_seed(10)
        if req_type == 'context':
            last_token_ids = torch.randint(1,
                                           seq_len + 1, (batch_size, ),
                                           dtype=torch.int32,
                                           device=device)
            last_token_ids[0] = seq_len
            segment_pos = torch.arange(0,
                                       seq_len,
                                       dtype=torch.int32,
                                       device=device).unsqueeze(0).repeat(
                                           batch_size, 1)
            segment_pos = segment_pos * (
                segment_pos < last_token_ids.unsqueeze(1)).type(torch.int32)
        else:
            last_token_ids = torch.ones(size=[batch_size],
                                        dtype=torch.int32,
                                        device=device)
            segment_pos = torch.randint(1,
                                        seq_len + 1,
                                        size=(batch_size, ),
                                        dtype=torch.int32,
                                        device=device)
            segment_pos[0] = seq_len
            segment_pos = segment_pos.unsqueeze(1)

        if remove_padding:
            last_token_ids = torch.cumsum(last_token_ids,
                                          dim=0,
                                          dtype=torch.int32).to(device)
            total_num_tokens = last_token_ids[batch_size - 1]
            x_shape = [total_num_tokens, dim]
        else:
            total_num_tokens = batch_size * seq_len
            x_shape = [batch_size, seq_len, dim]

        # init inputs
        x = torch.randn(size=x_shape, dtype=torch_dtype, device=device)
        gate_x = torch.randn(size=x_shape, dtype=torch_dtype, device=device)
        gate_a = torch.randn(size=x_shape, dtype=torch_dtype, device=device)
        if has_y:
            y = torch.randn(size=x_shape, dtype=torch_dtype, device=device)
        if has_y_bias:
            y_bias = torch.randn(size=[dim], dtype=torch_dtype, device=device)
        if req_type == 'context':
            lru_state = torch.empty(size=[batch_size, dim],
                                    dtype=torch.float32,
                                    device=device)
        else:
            lru_state = torch.randn(size=[batch_size, dim],
                                    dtype=torch.float32,
                                    device=device)
        host_request_types = torch.tensor([0 if req_type == 'context' else 1] *
                                          batch_size,
                                          dtype=torch.int32)

        # init A
        A = torch.randn(dim, device=device)
        min_rad, max_rad, eps = 0.9, 0.999, 1e-8
        A = torch.randn(dim, device=device, dtype=torch_dtype)
        A.uniform_(min_rad**2 + eps, max_rad**2 + eps)
        A.log_().mul_(0.5)
        A.neg_().exp_().sub_(1.0).log_()

        # output tensors
        output = torch.zeros(x.shape,
                             device=device,
                             dtype=str_dtype_to_torch(dtype))

        # ref tensors
        x_ref = x.detach().clone()
        gate_x_ref = gate_x.detach().clone()
        gate_a_ref = gate_a.detach().clone()
        lru_state_ref = lru_state.detach().clone()
        A_ref = A.detach().clone()
        y_ref = y.detach().clone() if has_y else None
        y_bias_ref = y_bias.view(1, 1,
                                 dim).detach().clone() if has_y_bias else None

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        if remove_padding:
            net.plugin_config.enable_remove_input_padding()
        else:
            net.plugin_config.remove_input_padding = False
        net.plugin_config.paged_state = False
        with tensorrt_llm.net_guard(net):
            x_tensor = Tensor(name='x',
                              shape=x.shape,
                              dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            gate_x_tensor = Tensor(name='gate_x',
                                   shape=gate_x.shape,
                                   dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            gate_a_tensor = Tensor(name='gate_a',
                                   shape=gate_a.shape,
                                   dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            if has_y:
                y_tensor = Tensor(name='y',
                                  shape=y.shape,
                                  dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            else:
                y_tensor = None
            if has_y_bias:
                y_bias_tensor = Tensor(
                    name='y_bias',
                    shape=y_bias.shape,
                    dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            else:
                y_bias_tensor = None
            state_tensor = Tensor(
                name='state',
                shape=lru_state.shape,
                dtype=tensorrt_llm.str_dtype_to_trt('float32'))
            A_tensor = Tensor(name='A',
                              shape=A.shape,
                              dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            host_request_types_tensor = Tensor(
                name='host_request_types',
                shape=host_request_types.shape,
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            last_token_ids_tensor = Tensor(
                name='last_token_ids',
                shape=last_token_ids.shape,
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            outputs = tensorrt_llm.functional.rg_lru(x_tensor,
                                                     gate_x_tensor,
                                                     gate_a_tensor,
                                                     A_tensor,
                                                     state_tensor,
                                                     host_request_types_tensor,
                                                     last_token_ids_tensor,
                                                     dim,
                                                     dtype,
                                                     y=y_tensor,
                                                     y_bias=y_bias_tensor)
            net._mark_output(outputs[0],
                             'output',
                             dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            net._mark_output(outputs[1],
                             'present_state',
                             dtype=tensorrt_llm.str_dtype_to_trt('float32'))

        # trt run
        inputs = {
            'x': x,
            'gate_x': gate_x,
            'gate_a': gate_a,
            'A': A,
            'state': lru_state,
            'host_request_types': host_request_types,
            'last_token_ids': last_token_ids,
        }
        if has_y:
            inputs['y'] = y
        if has_y_bias:
            inputs['y_bias'] = y_bias
        outputs = {'output': output, 'present_state': lru_state}
        stream = torch.cuda.current_stream()
        builder_config = builder.create_builder_config(precision=dtype, )
        engine = builder.build_engine(net, builder_config)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)
        session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)

        # pytorch run
        out_ref, present_state_ref = rg_lru_batch_ref(
            x_ref, gate_x_ref, gate_a_ref, y_ref, y_bias_ref, segment_pos,
            lru_state_ref, A_ref, batch_size, remove_padding, last_token_ids)
        dtype_atol = {"float16": 5e-2, "float32": 2e-3, "bfloat16": 5e-2}

        # get mask
        if not remove_padding and req_type == 'context':
            out_mask = torch.zeros(batch_size, seq_len, device=device)
            for i in range(batch_size):
                indices = torch.arange(last_token_ids[i])
                out_mask[i, indices] = 1
            out_mask = out_mask.unsqueeze(2).expand([batch_size, seq_len, dim])
        else:
            out_mask = torch.ones(size=x_shape, device=device)

        # compare results
        output_trtllm = (outputs['output'] * out_mask).to(torch.float32).cpu()
        output_torch = (out_ref * out_mask).to(torch.float32).cpu()
        present_state_trtllm = outputs['present_state'].to(torch.float32).cpu()
        present_state_torch = present_state_ref.to(torch.float32).cpu()
        np.testing.assert_allclose(output_torch.numpy(),
                                   output_trtllm.numpy(),
                                   atol=dtype_atol[dtype])
        np.testing.assert_allclose(present_state_torch.numpy(),
                                   present_state_trtllm.numpy(),
                                   atol=dtype_atol[dtype])
