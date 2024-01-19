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
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Parameter, Tensor
from tensorrt_llm._utils import torch_to_numpy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import getSMVersion


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')
        torch.manual_seed(42)

    @parameterized.expand(
        list(product([True, False], ['float16', 'float32', 'bfloat16'])))
    def test_layer_norm_plugin(self, remove_batch_dim, dtype):

        # Skip tests that are not supported in pre-ampere architecture
        if getSMVersion() < 80:
            if dtype == 'bfloat16':
                pytest.skip(
                    "bfloat16 is not supported in pre-ampere architecture")

        # test data
        hidden_size = 1024
        x_data = torch.randn((8, 128, hidden_size) if not remove_batch_dim else
                             (8 * 128, hidden_size),
                             dtype=torch.float64,
                             device="cuda")
        weight = torch.randn((hidden_size), dtype=torch.float64, device="cuda")
        bias = torch.randn((hidden_size), dtype=torch.float64, device="cuda")
        eps = 1e-5

        m = torch.nn.LayerNorm(hidden_size,
                               eps=eps,
                               dtype=torch.float64,
                               device="cuda")
        m.weight = torch.nn.Parameter(weight)
        m.bias = torch.nn.Parameter(bias)

        # pytorch run
        with torch.no_grad():
            ref = m(x_data)

        m.to(tensorrt_llm._utils.str_dtype_to_torch(dtype))
        x_data = x_data.to(tensorrt_llm._utils.str_dtype_to_torch(dtype))

        gamma_data = m.weight.detach().cpu()
        beta_data = m.bias.detach().cpu()

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        net.plugin_config.set_layernorm_plugin(dtype)
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            weight = Parameter(torch_to_numpy(gamma_data.cpu())).value
            bias = Parameter(torch_to_numpy(beta_data.cpu())).value

            output = tensorrt_llm.functional.layer_norm(x, hidden_size, weight,
                                                        bias, eps).trt_tensor
            output.name = 'output'
            network.mark_output(output)
            output.dtype = tensorrt_llm.str_dtype_to_trt(dtype)

        # trt run
        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(fp16=(dtype == 'float16'),
                                bf16=(dtype == 'bfloat16')))
        assert build_engine is not None, "Build engine failed"
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.cpu()})

        # compare diff
        dtype_atol = {"float16": 2e-2, "float32": 2e-6, "bfloat16": 8e-2}
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'].to(torch.float32),
                                   atol=dtype_atol[dtype])
