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
from polygraphy.backend.trt import EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')

    @parameterized.expand([
        (True, ),
        (False, ),
    ])
    def test_where_from_bool(self, condition=True):
        dtype = 'float32'
        t_data = torch.randn(2, 3)
        f_data = torch.randn(2, 3)

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            t = Tensor(name='t',
                       shape=t_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            f = Tensor(name='f',
                       shape=f_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.where(condition, t, f).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                't': t_data.numpy(),
                'f': f_data.numpy(),
            })

        ref = torch.where(torch.tensor(condition), t_data, f_data)
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-5)

    def test_where_from_tensor(self):
        dtype = 'float32'
        t_data = torch.randn(3, 4)
        f_data = torch.randn(3, 4)
        c_data = torch.randint(2, size=(3, 1), dtype=torch.bool)
        ref = torch.where(c_data, t_data, f_data)
        print(ref)

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            t = Tensor(name='t',
                       shape=t_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            f = Tensor(name='f',
                       shape=f_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            c = Tensor(name='c',
                       shape=c_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt('bool'))
            output = tensorrt_llm.functional.where(c, t, f).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                't': t_data.numpy(),
                'f': f_data.numpy(),
                'c': c_data.numpy(),
            })

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-5)
        print(t_data)
        print(f_data)
        print(c_data)
        print(outputs['output'])
        # assert False, "FORCED"
