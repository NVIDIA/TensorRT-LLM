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
import tensorrt as trt
import torch
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_trt
from tensorrt_llm.functional import low_latency_gemm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Monkey Patching for torch.float8_e4m3fn support
from polygraphy.datatype import DataType
from utils.util import getSMVersion

original_to_dtype = DataType.to_dtype


def patched_to_dtype(dtype, target_module):
    if dtype == DataType.FLOAT8E4M3FN and target_module == 'torch':
        return torch.float8_e4m3fn
    else:
        return original_to_dtype(dtype, target_module)


DataType.to_dtype = patched_to_dtype


class TestLowLatencyGemm(unittest.TestCase):

    def setUp(self) -> None:
        tensorrt_llm.logger.set_level('error')

    def reference_gemm_fp8(self, x, w, dtype):
        w = w.transpose(0, 1).to(dtype=torch.float32)
        y = torch.matmul(x.to(torch.float32), w)
        return y.to(str_dtype_to_torch(dtype))

    # float32
    def run_low_latency_gemm_sm90(self, m, n, k, output_dtype):
        torch.random.manual_seed(42)
        shape_x = (m, k)
        shape_w = (n, k)
        x = torch.randint(-2, 2, shape_x).to(str_dtype_to_torch('fp8'))
        w = torch.randint(-2, 2, shape_w).to(str_dtype_to_torch('fp8'))
        # Create builder
        builder = tensorrt_llm.Builder()
        # Create empty network
        net = builder.create_network()
        net.plugin_config.low_latency_gemm_plugin = "fp8"
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            # Init TensorRT-LLM tensor for x
            x_tensor = Tensor(name='x',
                              shape=x.shape,
                              dtype=str_dtype_to_trt('fp8'))
            # Init TensorRT-LLM tensor for w
            w_tensor = Tensor(name='w',
                              shape=w.shape,
                              dtype=str_dtype_to_trt('fp8'))
            # Get output tensor
            output = low_latency_gemm(
                x_tensor, w_tensor,
                strict_dtype=str_dtype_to_trt(output_dtype)).trt_tensor

            output.name = 'output'
            network.mark_output(output)
            output.dtype = str_dtype_to_trt(output_dtype)

        engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(
                fp8=True,
                fp16=True,
                memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 33554432}))

        feed_dict = {'x': x, "w": w}
        with TrtRunner(engine) as runner:
            outputs = runner.infer(feed_dict=feed_dict, check_inputs=False)

        ref = self.reference_gemm_fp8(x, w, output_dtype)
        np.testing.assert_allclose(ref.float().cpu().numpy(),
                                   outputs['output'].float())

    @pytest.mark.skipif(getSMVersion() != 90,
                        reason="LowLatencyGemm is only supported in SM90"
                        )  # Skip tests that are not supported in SM90
    def test_low_latency_gemm(self):
        m = 64
        n = 128
        k = 128
        output_dtype = "float32"
        self.run_low_latency_gemm_sm90(m, n, k, output_dtype)
        output_dtype = "float16"
        self.run_low_latency_gemm_sm90(m, n, k, output_dtype)


if __name__ == '__main__':
    unittest.main()
