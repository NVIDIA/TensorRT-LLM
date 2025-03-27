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
from parameterized import parameterized
from polygraphy.backend.trt import (CreateConfig, EngineFromNetwork, Profile,
                                    TrtRunner)

import tensorrt_llm
from tensorrt_llm import Tensor


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([
        ((4, ), ),
        ((4, 2), ),
        ((0, 4, 2), ),
    ])
    def test_nonzero(self, x_shape):
        # test data
        # x_shape = (4, 4)
        x_shape_last = list(x_shape[1:])
        x_data = torch.randint(2, size=x_shape, dtype=torch.int32).bool()
        print(x_data)

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=[-1] + x_shape_last,
                       dtype=tensorrt_llm.torch_dtype_to_trt(x_data.dtype))

            output = tensorrt_llm.functional.nonzero(x).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        # needs profile for dynamic shape
        profiles = Profile().add('x', [0] + x_shape_last, [2] + x_shape_last,
                                 [32] + x_shape_last)
        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(profiles=[profiles]))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'x': x_data.numpy(),
            })

        print(outputs['output'].transpose())
        # pytorch run
        # print(x_data.nonzero())
        ref = x_data.nonzero().transpose(0, 1)

        # compare diff
        np.testing.assert_allclose(ref.cpu().numpy(), outputs['output'])
        return
