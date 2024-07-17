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
from torch_ref import selective_scan_ref, selective_state_update_ref

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import skip_bf16_pre_ampere, unittest_name_func


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand(list(
        product([2048], [16], ['context', 'generation'],
                ['float16', 'float32', 'bfloat16'], [3], [16], [False, True])),
                          name_func=unittest_name_func)
    def test_selective_scan(self, dim, dstate, req_type, dtype, batch_size,
                            max_seq_len, remove_padding):

        # Skip tests that are not supported in pre-ampere architecture
        skip_bf16_pre_ampere(dtype)

        # configs
        device = "cuda"
        seq_len = max_seq_len if req_type == 'context' else 1
        dt_rank = 160
        is_variable_B = True
        is_variable_C = True
        delta_softplus = True

        # test data
        torch.random.manual_seed(0)
        if remove_padding:
            last_token_ids = torch.randint(1,
                                           seq_len + 1, (batch_size, ),
                                           dtype=torch.int32)
            last_token_ids = torch.cumsum(last_token_ids,
                                          dim=0,
                                          dtype=torch.int32).to(device)
            total_num_tokens = last_token_ids[batch_size - 1]
        else:
            last_token_ids = torch.ones(
                (batch_size, ), dtype=torch.int32, device=device) * seq_len
            total_num_tokens = batch_size * seq_len
        state = torch.randn(batch_size,
                            dstate,
                            dim,
                            device=device,
                            dtype=str_dtype_to_torch(dtype))
        x = torch.randn(total_num_tokens,
                        dim,
                        device=device,
                        dtype=str_dtype_to_torch(dtype))
        dt = torch.randn(total_num_tokens,
                         dim,
                         device=device,
                         dtype=str_dtype_to_torch(dtype))
        dt_bias = torch.rand(dim, device=device) - 4.0
        A = -torch.rand(dstate, dim, device=device) - 1.0
        BC = torch.randn(total_num_tokens,
                         dt_rank + dstate * 2,
                         device=device,
                         dtype=str_dtype_to_torch(dtype))
        D = torch.randn(dim, device=device)
        z = torch.randn_like(x)
        host_request_types = torch.tensor([0 if req_type == 'context' else 1] *
                                          batch_size,
                                          dtype=torch.int32)
        if not remove_padding or req_type == 'generation':
            x = x.view(-1, seq_len, dim)
            dt = dt.view(-1, seq_len, dim)
            BC = BC.view(-1, seq_len, dt_rank + dstate * 2)
            z = z.view(-1, seq_len, dim)
        output = torch.zeros(x.shape,
                             device=device,
                             dtype=str_dtype_to_torch(dtype))

        state_ref = state.detach().clone()
        x_ref = x.detach().clone()
        dt_ref = dt.detach().clone()
        dt_bias_ref = dt_bias.detach().clone()
        A_ref = A.detach().clone()
        B_ref = BC[..., dt_rank:dt_rank + dstate].detach().clone()
        C_ref = BC[..., dt_rank + dstate:].detach().clone()
        D_ref = D.detach().clone()
        z_ref = z.detach().clone()

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        if remove_padding:
            net.plugin_config.remove_input_padding = True
        else:
            net.plugin_config.remove_input_padding = False
        net.plugin_config.paged_state = False
        with tensorrt_llm.net_guard(net):
            x_tensor = Tensor(name='input',
                              shape=x.shape,
                              dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            state_tensor = Tensor(name='state',
                                  shape=state.shape,
                                  dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            dt_tensor = Tensor(name='delta',
                               shape=dt.shape,
                               dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            dt_bias_tensor = Tensor(
                name='delta_bias',
                shape=dt_bias.shape,
                dtype=tensorrt_llm.str_dtype_to_trt('float32'))
            A_tensor = Tensor(name='A',
                              shape=A.shape,
                              dtype=tensorrt_llm.str_dtype_to_trt('float32'))
            BC_tensor = Tensor(name='BC',
                               shape=BC.shape,
                               dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            D_tensor = Tensor(name='D',
                              shape=D.shape,
                              dtype=tensorrt_llm.str_dtype_to_trt('float32'))
            z_tensor = Tensor(name='z',
                              shape=z.shape,
                              dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            host_request_types_tensor = Tensor(
                name='host_request_types',
                shape=host_request_types.shape,
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            last_token_ids_tensor = Tensor(
                name='last_token_ids',
                shape=last_token_ids.shape,
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            outputs = tensorrt_llm.functional.selective_scan(
                x_tensor, state_tensor, dt_tensor, dt_bias_tensor, A_tensor,
                BC_tensor, D_tensor, z_tensor, host_request_types_tensor,
                last_token_ids_tensor, dim, dstate, dt_rank, is_variable_B,
                is_variable_C, delta_softplus, dtype)
            net._mark_output(outputs[0],
                             'output',
                             dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            net._mark_output(outputs[1],
                             'present_state',
                             dtype=tensorrt_llm.str_dtype_to_trt(dtype))

        # trt run
        inputs = {
            'input': x,
            'state': state,
            'delta': dt,
            'delta_bias': dt_bias,
            'A': A,
            'BC': BC,
            'D': D,
            'z': z,
            'host_request_types': host_request_types,
            'last_token_ids': last_token_ids
        }
        outputs = {'output': output, 'present_state': state}
        stream = torch.cuda.current_stream()
        builder_config = builder.create_builder_config(precision=dtype, )
        engine = builder.build_engine(net, builder_config)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)
        session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        out_ref = torch.zeros(output.shape,
                              device=device,
                              dtype=str_dtype_to_torch(dtype))
        if req_type == 'context':
            # pytorch run
            if remove_padding:
                for i in range(batch_size):
                    start_id = 0 if i == 0 else last_token_ids[i - 1]
                    end_id = last_token_ids[i]
                    part_out_ref, part_state_ref = selective_scan_ref(
                        x_ref[start_id:end_id].unsqueeze(0),
                        dt_ref[start_id:end_id].unsqueeze(0),
                        A_ref,
                        B_ref[start_id:end_id].unsqueeze(0),
                        C_ref[start_id:end_id].unsqueeze(0),
                        D=D_ref,
                        z=z_ref[start_id:end_id].unsqueeze(0),
                        delta_bias=dt_bias_ref,
                        delta_softplus=True)
                    out_ref[start_id:end_id][:] = part_out_ref.squeeze(0)
                    state_ref[i][:][:] = part_state_ref.squeeze(0)
            else:
                out_ref, state_ref = selective_scan_ref(x_ref,
                                                        dt_ref,
                                                        A_ref,
                                                        B_ref,
                                                        C_ref,
                                                        D=D_ref,
                                                        z=z_ref,
                                                        delta_bias=dt_bias_ref,
                                                        delta_softplus=True)
        elif req_type == 'generation':
            # pytorch run

            out_ref = selective_state_update_ref(state_ref,
                                                 x_ref.squeeze(1),
                                                 dt_ref.squeeze(1),
                                                 A_ref,
                                                 B_ref.squeeze(1),
                                                 C_ref.squeeze(1),
                                                 D=D_ref,
                                                 z=z_ref.squeeze(1),
                                                 dt_bias=dt_bias_ref,
                                                 dt_softplus=True)
            out_ref = out_ref.unsqueeze(1)

        dtype_atol = {"float16": 5e-3, "float32": 2e-3, "bfloat16": 5e-2}

        output_cpu = outputs['output'].to(torch.float32).cpu()
        present_state_cpu = outputs['present_state'].to(torch.float32).cpu()
        np.testing.assert_allclose(out_ref.to(torch.float32).cpu().numpy(),
                                   output_cpu.numpy(),
                                   atol=dtype_atol[dtype])
        np.testing.assert_allclose(state_ref.to(torch.float32).cpu().numpy(),
                                   present_state_cpu.numpy(),
                                   atol=dtype_atol[dtype])
