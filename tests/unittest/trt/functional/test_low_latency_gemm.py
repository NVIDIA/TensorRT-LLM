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

import numpy as np
import pytest
import torch
# Monkey Patching for torch.float8_e4m3fn support
from polygraphy.datatype import DataType
from utils.util import getSMVersion

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_trt

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

    # float32
    def run_low_latency_gemm_sm90(self, m, n, k, output_dtype):
        torch.random.manual_seed(42)
        shape_x = (m, k)
        shape_w = (n, k)
        x = torch.rand(shape_x, device="cuda").to(str_dtype_to_torch('fp8'))
        w = torch.rand(shape_w, device="cuda").to(str_dtype_to_torch('fp8'))
        # Create builder
        builder = tensorrt_llm.Builder()
        # Create empty network
        net = builder.create_network()
        net.plugin_config.low_latency_gemm_plugin = "fp8"
        with tensorrt_llm.net_guard(net):
            # Init TensorRT LLM tensor for x
            x_tensor = Tensor(name='x',
                              shape=x.shape,
                              dtype=str_dtype_to_trt('fp8'))
            # Init TensorRT LLM tensor for w
            w_tensor = Tensor(name='w',
                              shape=w.shape,
                              dtype=str_dtype_to_trt('fp8'))
            # Get output tensor
            output = tensorrt_llm.functional.low_latency_gemm(
                x_tensor, w_tensor, strict_dtype=str_dtype_to_trt(output_dtype))
            net._mark_output(output,
                             'output',
                             dtype=str_dtype_to_trt(output_dtype))

        feed_dict = {'x': x, "w": w}
        output_trt = torch.empty((m, n),
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
        ref = torch._scaled_mm(
            x,
            w.t(),
            scale_a=torch.tensor(1.0).cuda(),
            scale_b=torch.tensor(1.0).cuda(),
            out_dtype=str_dtype_to_torch(output_dtype),
            use_fast_accum=True,
        )
        np.testing.assert_allclose(ref.float().cpu(),
                                   outputs['output'].float().cpu())

    @pytest.mark.skipif(getSMVersion() != 90,
                        reason="LowLatencyGemm is only supported in SM90"
                        )  # Skip tests that are not supported in SM90
    def test_low_latency_gemm(self):
        m = 64
        k = 8192
        n = 8192
        output_dtype = "float32"
        self.run_low_latency_gemm_sm90(m, n, k, output_dtype)
        output_dtype = "float16"
        self.run_low_latency_gemm_sm90(m, n, k, output_dtype)


if __name__ == '__main__':
    unittest.main()
