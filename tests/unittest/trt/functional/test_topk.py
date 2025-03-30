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

import torch
from parameterized import parameterized
from utils.util import create_session, run_session, unittest_name_func

import tensorrt_llm


class TestTopK(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand(
        [
            t + (p, ) for t in [
                ((3, 5), 1, 1, True),
                ((3, 4, 6), 2, 0, True),
                ((3, 5), 1, 1, False),
                ((3, 4, 6), 2, 0, False),
                ((3, 4, 5000), 10, 2, True),
            ] for p in [False, True]  # prefer_plugin
        ],
        name_func=unittest_name_func)
    def test_topk(self, input_shape, k, d, largest, prefer_plugin):
        value_dtype = 'float32'
        indices_dtype = 'int32'
        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        input_data = torch.rand(*input_shape,
                                dtype=torch.float32,
                                device="cuda")
        with tensorrt_llm.net_guard(network):

            m = tensorrt_llm.functional.constant(input_data.cpu().numpy())
            topk_values, topk_indices = tensorrt_llm.functional.topk(
                m, k, d, largest=largest, prefer_plugin=prefer_plugin)
            topk_values.mark_output('output_values', value_dtype)
            topk_indices.mark_output('topk_indices', indices_dtype)

        # trt run
        session = create_session(builder, network)
        inputs = {}
        outputs = run_session(session, inputs)

        # pytorch run
        values, indices = torch.topk(input_data, k, dim=d, largest=largest)

        # compare diff
        torch.testing.assert_close(values, outputs['output_values'])
        # dtype does not match
        torch.testing.assert_close(indices.int(), outputs['topk_indices'].int())
