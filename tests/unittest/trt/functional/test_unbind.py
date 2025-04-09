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

    def test_unbind_1(self):
        # test data
        dtype = 'float32'
        input_shape = (1, 2, 3)
        unbind_dim = 0
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
            outputs = input.unbind(unbind_dim)
            for i in range(len(outputs)):
                outputs[i].name = f'output_{i}'
                network.mark_output(outputs[i].trt_tensor)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'input': input_data.numpy(),
            })

        # pytorch run
        refs = input_data.unbind(unbind_dim)

        # compare diff
        # compare diff
        for idx, ref in enumerate(refs):
            np.testing.assert_allclose(ref.cpu().numpy(),
                                       outputs[f'output_{idx}'])

    def test_unbind_1(self):
        # test data
        dtype = 'float32'
        input_shape = (1, 2, 3)
        unbind_dim = 1
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
            outputs = input.unbind(unbind_dim)
            for i in range(len(outputs)):
                outputs[i].name = f'output_{i}'
                network.mark_output(outputs[i].trt_tensor)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'input': input_data.numpy(),
            })

        # pytorch run
        refs = input_data.unbind(unbind_dim)

        # compare diff
        # compare diff
        for idx, ref in enumerate(refs):
            np.testing.assert_allclose(ref.cpu().numpy(),
                                       outputs[f'output_{idx}'])
