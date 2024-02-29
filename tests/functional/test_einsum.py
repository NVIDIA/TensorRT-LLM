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
from polygraphy.backend.trt import (CreateConfig, EngineFromNetwork, Profile,
                                    TrtRunner)

import tensorrt_llm
from tensorrt_llm import Tensor


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([('float32', )])
    def test_einsum(self, dtype):
        # torch 1.13: "baddbmm_with_gemm" not implemented for 'Half'
        # test data
        x_shape = (12, 12, 96, 96)
        y_shape = (12, 12, 96, 64)
        x_data = torch.rand(x_shape,
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        y_data = torch.rand(y_shape,
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        equation = 'bnth,bnhs->bnts'

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            y = Tensor(name='y',
                       shape=y_shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.einsum(equation, [x, y]).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        profiles = [
            Profile().add('x', x_shape, x_shape,
                          x_shape).add('y', y_shape, y_shape, y_shape)
        ]
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network),
                                         config=CreateConfig(profiles=profiles))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'x': x_data.numpy(),
                'y': y_data.numpy()
            })

        # pytorch run
        ref = torch.functional.einsum(equation, [x_data, y_data])

        # compare diff
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-4)
