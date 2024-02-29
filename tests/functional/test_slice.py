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

# isort: off
import torch
import tensorrt as trt
# isort: on
from parameterized import parameterized
from polygraphy.backend.trt import (CreateConfig, EngineFromNetwork, Profile,
                                    TrtRunner)

import tensorrt_llm
from tensorrt_llm import Tensor


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([('float32', ), ('float16')])
    def test_slice_1(self, dtype):
        # test data
        x_shape = (1, 256)
        x_data = torch.rand(x_shape,
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        starts_data = torch.tensor([0, 128]).int()
        sizes_data = torch.tensor([1, 1]).int()

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            starts = Tensor(name='starts', shape=(2, ), dtype=trt.int32)

            sizes = Tensor(name='sizes', shape=(2, ), dtype=trt.int32)

            output = tensorrt_llm.functional.slice(x, starts, sizes).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        profiles = [
            Profile().add('starts', (0, 0), (0, 128),
                          (0, 256)).add('sizes', (1, 1), (1, 1), (1, 256))
        ]
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network),
                                         config=CreateConfig(profiles=profiles))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(
                feed_dict={
                    'x': x_data.numpy(),
                    'starts': starts_data.numpy(),
                    'sizes': sizes_data.numpy(),
                })

        # pytorch run
        ref = x_data[0:1, 128:129]

        # compare diff
        np.testing.assert_allclose(ref.cpu().numpy(), outputs['output'])

    def test_slice_2(self):
        dtype = 'float32'
        x_shape = (256, )
        slice_length = 128
        x_data = torch.rand(x_shape,
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            starts = tensorrt_llm.functional.constant(
                np.array([0], dtype=np.int32))
            output_length = tensorrt_llm.functional.constant(
                np.array([0] * slice_length, dtype=np.int32))
            sizes = tensorrt_llm.functional.shape(output_length, 0)

            output = tensorrt_llm.functional.slice(x, starts,
                                                   sizes.view([1])).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})

        ref = x_data[0:slice_length]
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-5)
