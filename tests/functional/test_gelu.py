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
import itertools
import math
import os
import sys
import unittest

import torch
from parameterized import parameterized
from polygraphy.backend.trt import EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import skip_bf16_pre_ampere, unittest_name_func


class TestFunctional(unittest.TestCase):

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

    @parameterized.expand(itertools.product(
        ('float32', 'float16', 'bfloat16'),
        (False, True),
    ),
                          name_func=unittest_name_func)
    def test_gelu(self, dtype, strongly_typed):
        # Skip tests that are not supported in pre-ampere architecture
        skip_bf16_pre_ampere(dtype)

        torch_dtype = tensorrt_llm._utils.str_dtype_to_torch(dtype)
        x_shape = (12, 12, 96, 96)
        x_data = torch.rand(x_shape, dtype=torch_dtype)

        # construct trt network
        builder = tensorrt_llm.Builder()
        builder.strongly_typed = strongly_typed
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.gelu(x).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data})
        out = outputs['output'].to(torch_dtype)

        # Reference
        ref = self.gelu(x_data, dtype)

        if dtype == 'bfloat16':
            atol, rtol = 1e-5, 2e-2
        else:
            atol, rtol = 1e-5, 2e-3
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


if __name__ == '__main__':
    unittest.main()
