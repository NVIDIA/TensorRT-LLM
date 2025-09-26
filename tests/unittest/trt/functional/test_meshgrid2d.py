# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from utils.util import create_session, run_session

import tensorrt_llm
from tensorrt_llm import Tensor


class TestMeshgrid2d(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def test_meshgrid2d(self):
        dtype = 'float32'
        x_data = torch.tensor(
            [1, 2, 3],
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device="cuda")
        y_data = torch.tensor(
            [4, 5, 6],
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device="cuda")

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):

            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            y = Tensor(name='y',
                       shape=y_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output_x, output_y = tensorrt_llm.functional.meshgrid2d(x, y)
            output_x.mark_output('output_x')
            output_y.mark_output('output_y')

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {
            'x': x_data,
            'y': y_data,
        }
        outputs = run_session(session, inputs)

        # pytorch run
        ref_x, ref_y = torch.meshgrid([x_data, y_data], indexing='ij')

        # compare diff
        torch.testing.assert_close(ref_x, outputs['output_x'])
        torch.testing.assert_close(ref_y, outputs['output_y'])
