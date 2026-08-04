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

import torch
from parameterized import parameterized
from utils.util import create_session, run_session

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_torch


class TestScatter(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([
        (
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [1, 0, 2],
                [0, 2, 1],
            ],
            [[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]],
            0,
        ),
        ([[1, 2, 3], [4, 5, 6]], [[1, 2], [0, 1]], [[-1, -2], [-3, -4]], 1),
        (
            [[[-3.0, -2.0, -1.0, 10.0, -25.0]], [[0.0, 1.0, 2.0, -2.0, -1.0]]],
            [[[1, 2, 3, 0, 4]], [[4, 1, 2, 3, 0]]],
            [[[-1.0, 2.4, 3.2, 10.8, 8.9]], [[0, -11.2, 34.2, 223.9, -100]]],
            2,
        ),
    ])
    def test_scatter(self, input_data, indices, updates, dim):
        dtype = 'float32'
        torch_dtype = str_dtype_to_torch(dtype)
        input_data = input_data.cuda() if isinstance(
            input_data, torch.Tensor) else torch.tensor(input_data).cuda()
        indices = indices.cuda() if isinstance(
            indices, torch.Tensor) else torch.tensor(indices).int().cuda()
        updates = updates.cuda() if isinstance(
            updates, torch.Tensor) else torch.tensor(updates).cuda()
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

            output = tensorrt_llm.functional.scatter(input_t,
                                                     dim=dim,
                                                     indices=indices_t,
                                                     updates=updates_t)

            output.mark_output('output')

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {'input': input_data, 'indices': indices, 'updates': updates}
        outputs = run_session(session, inputs)

        # pytorch run
        ref = torch.scatter(input_data,
                            dim=dim,
                            index=indices.to(dtype=torch.int64),
                            src=updates)

        # compare diff
        torch.testing.assert_close(ref, outputs['output'])
