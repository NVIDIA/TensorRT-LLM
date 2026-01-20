# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pytest
import torch
from parameterized import parameterized
# Monkey Patching for torch.float8_e4m3fn support
from polygraphy.datatype import DataType
from utils.util import getSMVersion, unittest_name_func

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_trt
from tensorrt_llm.functional import gemm_swiglu, low_latency_gemm_swiglu

original_to_dtype = DataType.to_dtype


def patched_to_dtype(dtype, target_module):
    if dtype == DataType.FLOAT8E4M3FN and target_module == 'torch':
        return torch.float8_e4m3fn
    else:
        return original_to_dtype(dtype, target_module)


DataType.to_dtype = patched_to_dtype


class TestGemmSwiglu(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def reference_gemm_swiglu_sm90(self, x: torch.Tensor, w: torch.Tensor,
                                   scale_d0: float, scale_d1: float,
                                   scale_output: float, dtype):
        silu = torch.nn.SiLU()
        y = torch.matmul(x.to(torch.float32), w.to(torch.float32))
        split, split_gate = torch.split(y, y.size(1) // 2, dim=1)
        y_swiglu = (
            (scale_d0 * split) * silu(scale_d1 * split_gate)) * scale_output
        return y_swiglu.to(str_dtype_to_torch(dtype))

    def run_gemm_swiglu_sm90(self, m, n, k, scale_d0, scale_d1, scale_output,
                             dtype, is_low_latency):
        assert n % 32 == 0, "dim N must be a integer multiples of 32"
        assert k % 16 == 0, "dim K must be a integer multiples of 16"

        torch.random.manual_seed(42)

        shape_x = (m, k)
        x = torch.randint(-2, 2, shape_x,
                          device="cuda").to(str_dtype_to_torch(dtype))
        shape_w = (k, n)
        w = torch.randint(-2, 2, shape_w,
                          device="cuda").to(str_dtype_to_torch(dtype))

        output_dtype = "fp8"
        # Create builder
        builder = tensorrt_llm.Builder()
        # Create empty network
        net = builder.create_network()
        # Allow plugin of dtype type
        if is_low_latency:
            net.plugin_config.low_latency_gemm_swiglu_plugin = dtype
        else:
            net.plugin_config.gemm_swiglu_plugin = dtype

        with tensorrt_llm.net_guard(net):
            # Init TensorRT LLM tensor for x
            x_tensor = Tensor(name='x',
                              shape=x.shape,
                              dtype=str_dtype_to_trt(dtype))
            # Init TensorRT LLM tensor for w
            w_tensor = Tensor(name='w',
                              shape=w.shape,
                              dtype=str_dtype_to_trt(dtype))
            # Get output tensor
            if not is_low_latency:
                output = gemm_swiglu(x_tensor, w_tensor, None, scale_d0,
                                     scale_d1, scale_output)
            else:
                output = low_latency_gemm_swiglu(x_tensor, w_tensor, scale_d0,
                                                 scale_d1, scale_output)
            net._mark_output(output,
                             'output',
                             dtype=str_dtype_to_trt(output_dtype))

        feed_dict = {'x': x, "w": w.t().reshape(shape_w)}
        output_trt = torch.empty((m, n // 2),
                                 device="cuda",
                                 dtype=str_dtype_to_torch(output_dtype))
        outputs = {'output': output_trt}
        stream = torch.cuda.current_stream()
        builder_config = builder.create_builder_config(precision=output_dtype)
        engine = builder.build_engine(net, builder_config)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)
        session.run(inputs=feed_dict,
                    outputs=outputs,
                    stream=stream.cuda_stream)
        torch.cuda.synchronize()

        ref = self.reference_gemm_swiglu_sm90(x, w, scale_d0, scale_d1,
                                              scale_output, dtype)
        np.testing.assert_allclose(ref.float().cpu().numpy(),
                                   outputs['output'].cpu().float(),
                                   rtol=1e-3)

    @parameterized.expand(list(product([('fp8')], [False, True])),
                          name_func=unittest_name_func)
    @pytest.mark.skipif(getSMVersion() != 90,
                        reason="GemmSwigluSm90 is only supported in SM90"
                        )  # Skip tests that are not supported in SM90
    def test_gemm_swiglu_sm90(self, dtype, is_low_latency):
        bs = 2
        inseq = 13
        hidden_size = 256
        out_size = 32
        scale_d0 = 0.2
        scale_d1 = 1.3
        scale_output = 0.001

        self.run_gemm_swiglu_sm90(bs * inseq, out_size, hidden_size, scale_d0,
                                  scale_d1, scale_output, dtype, is_low_latency)


if __name__ == '__main__':
    unittest.main()
