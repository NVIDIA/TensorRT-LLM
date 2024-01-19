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
import os
import sys
import unittest

import numpy as np
import pytest

# isort: off
import torch
import tensorrt as trt
# isort: on
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import getSMVersion


class TestMatmul(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def _matmul(self, m, n, k, dtype, ta, tb):
        shape1 = (k, m) if ta else (m, k)
        mat1 = torch.randn(
            shape1, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype)) * 1e-1
        shape2 = (n, k) if tb else (k, n)
        mat2 = torch.randn(
            shape2, dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype)) * 1e-1
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=mat1.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            y = Tensor(name='y',
                       shape=mat2.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.matmul(x, y, transa=ta,
                                                    transb=tb).trt_tensor
            output.name = 'output'
            network.mark_output(output)
            output.dtype = tensorrt_llm.str_dtype_to_trt(dtype)

        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(
                fp16=(dtype == 'float16'),
                bf16=(dtype == 'bfloat16'),
                precision_constraints='obey',
                memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 33554432}))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': mat1, 'y': mat2})

        if ta:
            mat1 = mat1.cuda().transpose(0, 1)
        if tb:
            mat2 = mat2.cuda().transpose(0, 1)

        tols = {
            "float32": {
                "rtol": 1e-05,
                "atol": 1e-05
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

        if dtype != "float32":
            mat1 = mat1.cuda()
            mat2 = mat2.cuda()
        else:
            mat1 = mat1.cpu()
            mat2 = mat2.cpu()

        ref = torch.matmul(mat1, mat2).to(torch.float32)
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'].to(torch.float32),
                                   **tols[dtype])

    @parameterized.expand([('float16', False, False), ('float16', False, True),
                           ('float16', True, False), ('float16', True, True),
                           ('bfloat16', True, False), ('bfloat16', True, True),
                           ('float32', False, False), ('float32', False, True),
                           ('float32', True, False), ('float32', True, True)])
    def test_matmul(self, dtype, transa, transb):
        bs = 2
        inseq = 16
        hidden_size = 768
        tp = 1

        # Skip tests that are not supported in pre-ampere architecture
        if getSMVersion() < 80:
            if dtype == 'bfloat16':
                pytest.skip(
                    "bfloat16 is not supported in pre-ampere architecture")

        # qkv_gemm
        self._matmul(bs * inseq, 3 * hidden_size // tp, hidden_size, dtype,
                     transa, transb)

        # mlp_gemm_1
        self._matmul(bs * inseq, 4 * hidden_size // tp, hidden_size, dtype,
                     transa, transb)

    def test_matmul_broadcast(self):
        dtype = 'float32'
        x_data = torch.randn(16, 4, 4, 5)
        y_data = torch.randn(16, 1, 5, 4)

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            y = Tensor(name='y',
                       shape=y_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            output = tensorrt_llm.functional.matmul(x, y).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'x': x_data.numpy(),
                'y': y_data.numpy(),
            })

        ref = torch.matmul(x_data, y_data)
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-5)
