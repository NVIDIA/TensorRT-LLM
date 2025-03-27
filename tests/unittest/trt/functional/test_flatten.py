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
# isort: on
from polygraphy.backend.trt import EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def test_flatten_1(self):
        # test data
        dtype = 'float32'
        input_shape = (2, 3, 4, 5)
        start_dim = 0
        end_dim = -1
        input_data = torch.rand(
            input_shape, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            input = Tensor(name='input',
                           shape=input_shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = input.flatten(start_dim, end_dim).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'input': input_data.numpy(),
            })

        # pytorch run
        ref = input_data.flatten(start_dim, end_dim)

        # compare diff
        np.testing.assert_allclose(ref.cpu().numpy(), outputs['output'])

    def test_flatten_2(self):
        # test data
        dtype = 'float32'
        input_shape = (2, 3, 4, 5)
        start_dim = 0
        end_dim = 3
        input_data = torch.rand(
            input_shape, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            input = Tensor(name='input',
                           shape=input_shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = input.flatten(start_dim, end_dim).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'input': input_data.numpy(),
            })

        # pytorch run
        ref = input_data.flatten(start_dim, end_dim)

        # compare diff
        np.testing.assert_allclose(ref.cpu().numpy(), outputs['output'])

    def test_flatten_3(self):
        # test data
        dtype = 'float32'
        input_shape = (2, 3, 4, 5)
        start_dim = 0
        end_dim = 2
        input_data = torch.rand(
            input_shape, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            input = Tensor(name='input',
                           shape=input_shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = input.flatten(start_dim, end_dim).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'input': input_data.numpy(),
            })

        # pytorch run
        ref = input_data.flatten(start_dim, end_dim)

        # compare diff
        np.testing.assert_allclose(ref.cpu().numpy(), outputs['output'])

    def test_flatten_4(self):
        # test data
        dtype = 'float32'
        input_shape = (2, 3, 4, 5)
        start_dim = 1
        end_dim = 3
        input_data = torch.rand(
            input_shape, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            input = Tensor(name='input',
                           shape=input_shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = input.flatten(start_dim, end_dim).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'input': input_data.numpy(),
            })

        # pytorch run
        ref = input_data.flatten(start_dim, end_dim)

        # compare diff
        np.testing.assert_allclose(ref.cpu().numpy(), outputs['output'])

    def test_flatten_5(self):
        # test data
        dtype = 'float32'
        input_shape = (2, 3, 4, 5)
        start_dim = 1
        end_dim = 2
        input_data = torch.rand(
            input_shape, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            input = Tensor(name='input',
                           shape=input_shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = input.flatten(start_dim, end_dim).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'input': input_data.numpy(),
            })

        # pytorch run
        ref = input_data.flatten(start_dim, end_dim)

        # compare diff
        np.testing.assert_allclose(ref.cpu().numpy(), outputs['output'])

    def test_flatten_6(self):
        # test data
        dtype = 'float32'
        input_shape = (2, 3, 4, 5)
        start_dim = 2
        end_dim = 2
        input_data = torch.rand(
            input_shape, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            input = Tensor(name='input',
                           shape=input_shape,
                           dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = input.flatten(start_dim, end_dim).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'input': input_data.numpy(),
            })

        # pytorch run
        ref = input_data.flatten(start_dim, end_dim)

        # compare diff
        np.testing.assert_allclose(ref.cpu().numpy(), outputs['output'])
