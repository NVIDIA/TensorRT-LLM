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

# isort: off
import torch
# isort: on

from parameterized import parameterized
from utils.util import create_session, run_session, unittest_name_func

import tensorrt_llm
from tensorrt_llm import Tensor


class TestMatmul(unittest.TestCase):

    def setUp(self):
        torch.backends.cudnn.allow_tf32 = False
        tensorrt_llm.logger.set_level('error')

    def _matmul(self, m, n, k, dtype, ta, tb):
        shape1 = (k, m) if ta else (m, k)
        mat1 = torch.randn(shape1,
                           dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                           device="cuda") * 1e-1
        shape2 = (n, k) if tb else (k, n)
        mat2 = torch.randn(shape2,
                           dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                           device="cuda") * 1e-1
        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):

            x = Tensor(name='x',
                       shape=mat1.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            y = Tensor(name='y',
                       shape=mat2.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.matmul(x, y, transa=ta, transb=tb)
            output.mark_output('output', dtype)

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {'x': mat1, 'y': mat2}
        outputs = run_session(session, inputs)

        tols = {
            "float32": {
                "rtol": 4e-4,
                "atol": 1e-02
            },
            "float16": {
                "rtol": 1e-02,
                "atol": 1e-02
            },
            "bfloat16": {
                "rtol": 1e-02,
                "atol": 1e-02
            },
        }

        # pytorch run
        if ta:
            mat1 = mat1.transpose(0, 1)
        if tb:
            mat2 = mat2.transpose(0, 1)
        ref = torch.matmul(mat1, mat2)
        torch.testing.assert_close(ref, outputs['output'], **tols[dtype])

    @parameterized.expand([('float16', False, False), ('float16', False, True),
                           ('float16', True, False), ('float16', True, True),
                           ('bfloat16', True, False), ('bfloat16', True, True),
                           ('float32', False, False), ('float32', False, True),
                           ('float32', True, False), ('float32', True, True)],
                          name_func=unittest_name_func)
    def test_matmul(self, dtype, transa, transb):
        bs = 2
        inseq = 16
        hidden_size = 768
        tp = 1

        # qkv_gemm
        self._matmul(bs * inseq, 3 * hidden_size // tp, hidden_size, dtype,
                     transa, transb)

        # mlp_gemm_1
        self._matmul(bs * inseq, 4 * hidden_size // tp, hidden_size, dtype,
                     transa, transb)

    def test_matmul_broadcast(self):
        dtype = 'float32'
        x_data = torch.randn(16, 4, 4, 5, device="cuda")
        y_data = torch.randn(16, 1, 5, 4, device="cuda")

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
            output = tensorrt_llm.functional.matmul(x, y)
            output.mark_output('output', dtype)

        # trt run
        session = create_session(builder, network, precision=dtype)
        inputs = {'x': x_data, 'y': y_data}
        outputs = run_session(session, inputs)

        # pytorch run
        ref = torch.matmul(x_data, y_data)

        # compare diff
        torch.testing.assert_close(ref, outputs['output'])
