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

import numpy as np
import tensorrt as trt
import torch
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, engine_from_network
from utils.util import run_session, skip_pre_hopper, unittest_name_func

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.quantization.functional import fp8_rowwise_gemm

from . import _utils


class TestFp8RowwiseGemm(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('verbose')

    def _fp8_rowwise_gemm(self, m, n, k, dtype, per_token_scaling,
                          per_channel_scaling):
        shape1 = (m, k)
        mat1 = (1 * torch.randn(shape1, device="cuda")).to(
            dtype=torch.float8_e4m3fn)
        shape2 = (n, k)
        mat2 = (1 * torch.randn(shape2, device="cuda")).to(
            dtype=torch.float8_e4m3fn)

        # Init scales in fp32
        shape_scale_a = (m, 1) if per_token_scaling else (1, 1)
        scale_a_torch = torch.ones(shape_scale_a,
                                   device="cuda",
                                   dtype=torch.float32)
        scale_a_torch *= 1e-2 * torch.randint(
            1, 10, shape_scale_a, device="cuda", dtype=torch.float32)
        shape_scale_b = (1, n) if per_channel_scaling else (1, 1)
        scale_b_torch = torch.ones(shape_scale_b,
                                   device="cuda",
                                   dtype=torch.float32)
        scale_b_torch *= 1e-2 * torch.randint(
            1, 10, shape_scale_b, device="cuda", dtype=torch.float32)

        # Create builder
        builder = tensorrt_llm.Builder()
        builder.strongly_typed = False  # Test need to run in weekly typed mode
        # Create empty network
        network = builder.create_network()
        # Allow fp8_rowwise_gemm_plugin of dtype type
        network.plugin_config.fp8_rowwise_gemm_plugin = dtype
        with tensorrt_llm.net_guard(network):
            # Init TensorRT LLM tensor for mat1
            x = Tensor(name='x',
                       shape=mat1.shape,
                       dtype=tensorrt_llm._utils.str_dtype_to_trt("fp8"))
            # Init TensorRT LLM tensor for mat2
            y = Tensor(name='y',
                       shape=mat2.shape,
                       dtype=tensorrt_llm._utils.str_dtype_to_trt("fp8"))
            # Init TensorRT LLM tensor for per token scaling
            scale_a = Tensor(
                name='scale_a',
                shape=scale_a_torch.shape,
                dtype=tensorrt_llm._utils.str_dtype_to_trt("float32"))
            # Init TensorRT LLM tensor for per channel scaling
            scale_b = Tensor(
                name='scale_b',
                shape=scale_b_torch.shape,
                dtype=tensorrt_llm._utils.str_dtype_to_trt("float32"))
            # Get output tensor for fp8_rowwise_gemm gemm
            output = fp8_rowwise_gemm(x, y, scale_a, scale_b, per_token_scaling,
                                      per_channel_scaling)
            output.mark_output('output', dtype)

        engine = engine_from_network(
            (builder.trt_builder, network.trt_network),
            config=CreateConfig(
                fp8=True,
                fp16=(dtype == "float16"),
                precision_constraints="obey",
                memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 33554432}))
        assert engine is not None, "Failed to build engine"

        # Create TensorRT LLM session
        session = tensorrt_llm.runtime.Session.from_serialized_engine(
            engine.serialize())

        inputs = {
            'x': mat1,
            'y': mat2,
            'scale_a': scale_a_torch,
            'scale_b': scale_b_torch
        }
        # Infer engine
        outputs = run_session(session, inputs)

        ref = _utils.gt_matmul_fp8_rowwise(mat1,
                                           mat2,
                                           scale_a_torch,
                                           scale_b_torch,
                                           dtype,
                                           bias=None)

        for i in range(10):
            outputs = run_session(session, inputs)

        dtype_atol = {"float16": 5e-3, "float32": 5e-3, "bfloat16": 5e-2}
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'].cpu().numpy(),
                                   atol=dtype_atol[dtype])

    @parameterized.expand(product(["float16"], [True], [True]),
                          name_func=unittest_name_func)
    @skip_pre_hopper  # fp8_rowwise_gemm is not supported in pre-Hopper
    def test_matmul(self, dtype, per_token_scaling, per_channel_scaling):
        bs = 2
        inseq = 64
        hidden_size = 512

        # qkv_gemm
        self._fp8_rowwise_gemm(bs * inseq, 3 * hidden_size, hidden_size, dtype,
                               per_token_scaling, per_channel_scaling)

        # mlp_gemm_1
        self._fp8_rowwise_gemm(bs * inseq, 4 * hidden_size, hidden_size, dtype,
                               per_channel_scaling, per_token_scaling)


if __name__ == '__main__':
    unittest.main()
