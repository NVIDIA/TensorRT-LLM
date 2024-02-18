# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([
        ((3, 5), 1, 1, True),
        ((3, 4, 6), 2, 0, True),
        ((3, 5), 1, 1, False),
        ((3, 4, 6), 2, 0, False),
    ])
    def test_topk(self, input_shape, k, d, largest):
        value_dtype = 'float32'
        indices_dtype = 'int32'
        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()

        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            input_data = np.random.rand(*input_shape).astype(np.float32)
            m = tensorrt_llm.functional.constant(input_data)
            topk_values, topk_indices = tensorrt_llm.functional.topk(
                m, k, d, largest=largest)
            topk_values = topk_values.trt_tensor
            topk_indices = topk_indices.trt_tensor
            topk_values.name = 'output_values'
            topk_indices.name = 'output_indices'
            network.mark_output(topk_values)
            network.mark_output(topk_indices)
            topk_values.dtype = tensorrt_llm.str_dtype_to_trt(value_dtype)
            topk_indices.dtype = tensorrt_llm.str_dtype_to_trt(indices_dtype)

            # trt run
            build_engine = EngineFromNetwork(
                (builder.trt_builder, net.trt_network))
            with TrtRunner(build_engine) as runner:
                outputs = runner.infer(feed_dict={})
            values, indices = torch.topk(torch.Tensor(input_data),
                                         k,
                                         dim=d,
                                         largest=largest)

            np.testing.assert_allclose(values.cpu().numpy(),
                                       outputs['output_values'],
                                       atol=1e-5)
            np.testing.assert_allclose(indices.cpu().numpy(),
                                       outputs['output_indices'])
