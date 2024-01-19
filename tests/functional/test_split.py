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
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([
        ('float32', 0, 4),
        ('float32', 1, 4),
        ('float32', -1, 4),
        ('float32', -2, 4),
        ('float16', 0, 4),
        ('float16', 1, 4),
        ('float16', -1, 4),
        ('float16', -2, 4),
        ('float32', 0, [2, 100, 26]),
        ('float32', 1, [2, 100, 100, 52, 2]),
        ('float32', -1, [2, 100, 100, 52, 2]),
        ('float32', -2, [2, 100, 26]),
        ('float16', 0, [2, 100, 26]),
        ('float16', 1, [2, 100, 100, 52, 2]),
        ('float16', -1, [2, 100, 100, 52, 2]),
        ('float16', -2, [2, 100, 26]),
    ])
    def test_split(self, dtype, dim, split_size_or_sections):
        # test data
        x_shape = (128, 256)
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

            outputs = tensorrt_llm.functional.split(x, split_size_or_sections,
                                                    dim)
            for i in range(len(outputs)):
                output = outputs[i].trt_tensor
                output.name = f'output_{i}'
                network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'x': x_data.numpy(),
            })

        # pytorch run
        ref_outputs = torch.split(x_data, split_size_or_sections, dim)

        # compare diff
        assert len(outputs.keys()) == len(ref_outputs)
        for i in range(len(ref_outputs)):
            np.testing.assert_allclose(ref_outputs[i].cpu().numpy(),
                                       outputs[f'output_{i}'])
