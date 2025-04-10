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
import math
import unittest

import torch
from parameterized import parameterized
from utils.util import create_session, run_session, unittest_name_func

import tensorrt_llm
from tensorrt_llm import Tensor


class TestGelu(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @staticmethod
    def gelu(x, dtype):
        if dtype == 'float32':
            res = torch.nn.functional.gelu(x)
        else:
            res = 0.5 * x * (1 + torch.tanh(
                math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return res

    @parameterized.expand(('float32', 'float16', 'bfloat16'),
                          name_func=unittest_name_func)
    def test_gelu(self, dtype):
        torch_dtype = tensorrt_llm._utils.str_dtype_to_torch(dtype)
        x_shape = (12, 12, 96, 96)
        x_data = torch.rand(x_shape, dtype=torch_dtype, device="cuda")

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):

            x = Tensor(name='x',
                       shape=x_shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.gelu(x)
            output.mark_output('output', dtype)

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {
            'x': x_data,
        }
        outputs = run_session(session, inputs)

        # pytorch run
        ref = self.gelu(x_data, dtype).to(torch_dtype)

        # compare diff
        if dtype == 'bfloat16':
            atol, rtol = 1e-5, 2e-2
        else:
            atol, rtol = 1e-5, 2e-3
        torch.testing.assert_close(outputs['output'], ref, atol=atol, rtol=rtol)
