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

import numpy as np
import torch
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([
        ('int32', (256, ), 0),
        ('int32', (256, ), -1),
        ('float32', (3, 16), 0),
        ('float32', (3, 16), 1),
        ('float32', (3, 16), -2),
        ('float16', (5, 6, 8), 1),
        ('float16', (5, 6, 8), 2),
        ('float16', (5, 6, 8), -3),
    ])
    def test_cumsum(self, dtype, x_shape, dim):
        if 'float' in dtype:
            x_data = torch.rand(
                x_shape, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        else:
            x_data = torch.randint(
                -100,
                100,
                x_shape,
                dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.cumsum(x, dim=dim).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(fp16=(dtype == 'float16')))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})

        ref = torch.cumsum(x_data.cuda(), dim=dim)
        tols = {
            "float32": {
                "rtol": 1e-05,
                "atol": 1e-05
            },
            "float16": {
                "rtol": 1e-02,
                "atol": 1e-02
            },
            "int32": {
                "rtol": 0,
                "atol": 0
            },
        }
        np.testing.assert_allclose(ref.cpu().numpy(), outputs['output'],
                                   **tols[dtype])
