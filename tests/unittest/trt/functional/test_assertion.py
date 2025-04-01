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
from utils.util import create_session, run_session

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.functional import shape


class TestAssertion(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def test_assertion(self):
        dtype = 'float32'
        torch_dtype = tensorrt_llm.str_dtype_to_torch(dtype)
        # test data
        x_shape = (2, 4, 8)
        y_shape = (4, 4, 4)
        x_data = torch.rand(x_shape, dtype=torch_dtype, device="cuda")
        y_data = torch.rand(y_shape, dtype=torch_dtype, device="cuda")

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            x = Tensor(name='x',
                       shape=x_shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            y = Tensor(name='y',
                       shape=y_shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            tensorrt_llm.functional.assertion(shape(x, 1) == shape(y, 1))
            output = tensorrt_llm.functional.identity(x)
            output.mark_output('output', dtype)

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {'x': x_data, 'y': y_data}
        run_session(session, inputs)
