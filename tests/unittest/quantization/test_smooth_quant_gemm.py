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
from itertools import chain, product

import _utils
import numpy as np
import tensorrt as trt
import torch
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.quantization.functional import smooth_quant_gemm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import unittest_name_func


class TestSmoothQuantGemm(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def _sq_gemm(self, m, n, k, dtype, per_token_scaling, per_channel_scaling,
                 use_plugin):
        # Init operands for multiplication in int32
        shape1 = (m, k)
        mat1 = torch.randint(-128, 128, shape1, dtype=torch.int8)
        shape2 = (n, k)
        mat2 = torch.randint(-128, 128, shape2, dtype=torch.int8)

        # Init scales in fp32
        shape_scale_a = (m, 1) if per_token_scaling else (1, 1)
        scale_a_torch = torch.ones(shape_scale_a, dtype=torch.float32) * 1e-2
        scale_a_torch *= torch.randint(1,
                                       10,
                                       shape_scale_a,
                                       dtype=torch.float32)
        shape_scale_b = (1, n) if per_channel_scaling else (1, 1)
        scale_b_torch = torch.ones(shape_scale_b, dtype=torch.float32) * 1e-2
        scale_b_torch *= torch.randint(1,
                                       10,
                                       shape_scale_b,
                                       dtype=torch.float32)

        # Create builder
        builder = tensorrt_llm.Builder()
        # Create empty network
        network = builder.create_network()
        # Allow SQ plugin of dtype type
        if use_plugin:
            network.plugin_config.smooth_quant_gemm_plugin = dtype
        with tensorrt_llm.net_guard(network):
            # Init TensorRT-LLM tensor for mat1
            x = Tensor(name='x',
                       shape=mat1.shape,
                       dtype=tensorrt_llm._utils.str_dtype_to_trt("int8"))
            # Init TensorRT-LLM tensor for mat2
            y = Tensor(name='y',
                       shape=mat2.shape,
                       dtype=tensorrt_llm._utils.str_dtype_to_trt("int8"))
            # Init TensorRT-LLM tensor for per token scaling
            scale_a = tensorrt_llm.functional.constant(scale_a_torch.numpy())
            # Init TensorRT-LLM tensor for per channel scaling
            scale_b = tensorrt_llm.functional.constant(scale_b_torch.numpy())
            # Get output tensor for SQ gemm
            output = smooth_quant_gemm(x, y, scale_a, scale_b,
                                       per_token_scaling, per_channel_scaling,
                                       dtype)
            output.mark_output('output', dtype)

        # TODO: When dtype=int32, per_token_scaling=False, per_channel_scaling=False,
        # This test will break using new API on A30, only when running with all other unit tests.
        # This is a weird issue, so skip changing this file.
        engine = EngineFromNetwork(
            (builder.trt_builder, network.trt_network),
            config=CreateConfig(
                memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 33554432}))

        # Infer engine
        with TrtRunner(engine) as runner:
            outputs = runner.infer(feed_dict={
                'x': mat1.numpy(),
                'y': mat2.numpy(),
            })

        ref = _utils.gt_matmul_smooth_quant(mat1,
                                            mat2,
                                            scale_a_torch,
                                            scale_b_torch,
                                            dtype,
                                            bias=None)

        np.testing.assert_allclose(ref.cpu().numpy(), outputs['output'])

    @parameterized.expand(chain(
        product(["float16", "float32", "int32"], [True, False], [True, False],
                [True]),
        product(["float16", "float32"], [True, False], [True, False], [False])),
                          name_func=unittest_name_func)
    def test_matmul(self, dtype, per_token_scaling, per_channel_scaling,
                    use_plugin):
        bs = 2
        inseq = 16
        hidden_size = 768

        # qkv_gemm
        self._sq_gemm(bs * inseq, 3 * hidden_size, hidden_size, dtype,
                      per_token_scaling, per_channel_scaling, use_plugin)

        # mlp_gemm_1
        self._sq_gemm(bs * inseq, 4 * hidden_size, hidden_size, dtype,
                      per_channel_scaling, per_token_scaling, use_plugin)


if __name__ == '__main__':
    unittest.main()
