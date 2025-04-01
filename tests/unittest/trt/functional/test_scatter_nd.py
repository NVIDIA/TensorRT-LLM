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
from itertools import product

import torch
from parameterized import parameterized
from utils.util import create_session, run_session

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_torch


class TestScatter(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def scatter_nd_ref(self, input_data, indices, updates):
        input_data[tuple(indices.t())] = updates.view(-1)
        return input_data

    @parameterized.expand(
        list(
            product([
                ([[-3.0, -2.0, -1.0, 10.0, -25.0]
                  ], [[0.0, 1.0, 2.0, -2.0, -1.0]]),
            ], [([0, 0, 1], [1, 0, 3])], [([-1.0, 0])],
                    ['float16', 'float32', 'int32'])))
    def test_scatter(self, input_data, indices, updates, dtype):

        torch_dtype = str_dtype_to_torch(dtype)
        input_data = torch.tensor(input_data).cuda()
        indices = torch.tensor(indices).int().cuda()
        updates = torch.tensor(updates).cuda()

        input_data = input_data.to(torch_dtype)
        updates = updates.to(torch_dtype)

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            input_t = Tensor(name='input',
                             shape=input_data.shape,
                             dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            indices_t = Tensor(name='indices',
                               shape=indices.shape,
                               dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            updates_t = Tensor(name='updates',
                               shape=updates.shape,
                               dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            output = tensorrt_llm.functional.scatter_nd(input_t, indices_t,
                                                        updates_t)

            output.mark_output('output')

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {'input': input_data, 'indices': indices, 'updates': updates}
        outputs = run_session(session, inputs)

        # pytorch run
        ref = self.scatter_nd_ref(input_data, indices, updates)

        # compare diff
        torch.testing.assert_close(ref, outputs['output'])
