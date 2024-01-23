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

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from torch_ref import selective_scan_ref, selective_state_update_ref
from utils.util import getSMVersion


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand(
        list(
            product([2048], [16], ['context', 'generation'],
                    ['float16', 'float32', 'bfloat16'])))
    def test_selective_scan(self, dim, dstate, req_type, dtype):

        # Skip tests that are not supported in pre-ampere architecture
        if getSMVersion() < 80:
            if dtype == 'bfloat16':
                pytest.skip(
                    "bfloat16 is not supported in pre-ampere architecture")

        # configs
        batch_size = 1
        device = "cuda"
        seq_len = 16 if req_type == 'context' else 1
        is_variable_B = True
        is_variable_C = True
        delta_softplus = True

        # test data
        torch.random.manual_seed(0)
        state = torch.randn(batch_size, dim, dstate, device=device)
        x = torch.randn(batch_size,
                        dim,
                        seq_len,
                        device=device,
                        dtype=str_dtype_to_torch(dtype))
        dt = torch.randn(batch_size,
                         dim,
                         seq_len,
                         device=device,
                         dtype=str_dtype_to_torch(dtype))
        dt_bias = torch.rand(dim, device=device) - 4.0
        A = -torch.rand(dim, dstate, device=device) - 1.0
        B = torch.randn(batch_size,
                        dstate,
                        seq_len,
                        device=device,
                        dtype=str_dtype_to_torch(dtype))
        C = torch.randn(batch_size,
                        dstate,
                        seq_len,
                        device=device,
                        dtype=str_dtype_to_torch(dtype))
        D = torch.randn(dim, device=device)
        z = torch.randn_like(x)
        host_request_types = torch.tensor([0 if req_type == 'context' else 1] *
                                          batch_size,
                                          dtype=torch.int32)
        output = torch.zeros(x.shape,
                             device=device,
                             dtype=str_dtype_to_torch(dtype))

        state_ref = state.detach().clone()
        x_ref = x.detach().clone()
        dt_ref = dt.detach().clone()
        dt_bias_ref = dt_bias.detach().clone()
        A_ref = A.detach().clone()
        B_ref = B.detach().clone()
        C_ref = C.detach().clone()
        D_ref = D.detach().clone()
        z_ref = z.detach().clone()

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        net.plugin_config.set_selective_scan_plugin(dtype)
        with tensorrt_llm.net_guard(net):
            x_tensor = Tensor(name='input',
                              shape=x.shape,
                              dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            state_tensor = Tensor(
                name='state',
                shape=state.shape,
                dtype=tensorrt_llm.str_dtype_to_trt('float32'))
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
            B_tensor = Tensor(name='B',
                              shape=B.shape,
                              dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            C_tensor = Tensor(name='C',
                              shape=C.shape,
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
            outputs = tensorrt_llm.functional.selective_scan(
                x_tensor, state_tensor, dt_tensor, dt_bias_tensor, A_tensor,
                B_tensor, C_tensor, D_tensor, z_tensor,
                host_request_types_tensor, dim, dstate, is_variable_B,
                is_variable_C, delta_softplus)
            net._mark_output(outputs[0],
                             'output',
                             dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            net._mark_output(outputs[1],
                             'present_state',
                             dtype=tensorrt_llm.str_dtype_to_trt('float32'))

        # trt run
        inputs = {
            'input': x,
            'state': state,
            'delta': dt,
            'delta_bias': dt_bias,
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'z': z,
            'host_request_types': host_request_types
        }
        outputs = {'output': output, 'present_state': state}
        stream = torch.cuda.current_stream()
        builder_config = builder.create_builder_config(precision=dtype, )
        engine = builder.build_engine(net, builder_config)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)
        session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)

        if req_type == 'context':
            # pytorch run
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
                                                 x_ref.squeeze(2),
                                                 dt_ref.squeeze(2),
                                                 A_ref,
                                                 B_ref.squeeze(2),
                                                 C_ref.squeeze(2),
                                                 D=D_ref,
                                                 z=z_ref.squeeze(2),
                                                 dt_bias=dt_bias_ref,
                                                 dt_softplus=True)
            out_ref = out_ref.unsqueeze(2)

        dtype_atol = {"float16": 5e-3, "float32": 2e-3, "bfloat16": 5e-2}
        np.testing.assert_allclose(out_ref.to(torch.float32).cpu().numpy(),
                                   outputs['output'].to(
                                       torch.float32).cpu().numpy(),
                                   atol=dtype_atol[dtype])
        np.testing.assert_allclose(state_ref.to(torch.float32).cpu().numpy(),
                                   outputs['present_state'].to(
                                       torch.float32).cpu().numpy(),
                                   atol=dtype_atol[dtype])
