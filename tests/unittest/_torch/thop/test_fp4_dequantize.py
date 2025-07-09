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
from parameterized import parameterized
from utils.util import unittest_name_func

import tensorrt_llm


class TestFP4Dequantize(unittest.TestCase):
    """Test suite for FP4 quantization and dequantization accuracy validation."""

    def setUp(self):
        tensorrt_llm.logger.set_level("warning")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    @parameterized.expand(
        [(seq_len, hidden_dim, swizzled_layout, output_dtype)
         for seq_len in [1, 128, 512] for hidden_dim in [32, 128, 1024, 7168]
         for swizzled_layout in [True, False]
         for output_dtype in ["float16", "bfloat16", "float32"]],
        name_func=unittest_name_func,
    )
    def test_compare_cpu_gpu_implementations(self, seq_len, hidden_dim,
                                             swizzled_layout, output_dtype):
        """Compare GPU dequantization with CPU reference implementation."""
        # Create test tensor with parameterized dimensions and dtype
        test_tensor = torch.randn(seq_len,
                                  hidden_dim,
                                  dtype=torch.float16,
                                  device='cuda')

        # Create global scale factor
        global_scale = (448 * 6) / test_tensor.abs().max().float()

        sf_vec_size = 16
        sf_use_ue8m0 = False

        # Quantize using GPU
        fp4_tensor, scale_factors = torch.ops.trtllm.fp4_quantize(
            test_tensor, global_scale, sf_vec_size, sf_use_ue8m0,
            swizzled_layout)

        # Dequantize using GPU
        dequantized_gpu = torch.ops.trtllm.fp4_dequantize(
            fp4_tensor, scale_factors, 1.0 / global_scale, sf_vec_size,
            sf_use_ue8m0, swizzled_layout, output_dtype)

        # Dequantize using CPU reference
        dequantized_cpu = torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float_v2(
            fp4_tensor.cpu(),
            scale_factors.cpu(),
            (1.0 / global_scale).cpu(),
            sf_vec_size,
            1,  # sf_type (1 for UE4M3)
            swizzled_layout)

        # Compare results
        self.assertTrue(
            torch.allclose(dequantized_cpu,
                           dequantized_gpu.cpu().to(dequantized_cpu.dtype),
                           atol=1e-2,
                           rtol=1e-2))
