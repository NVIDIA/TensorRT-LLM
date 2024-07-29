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

import torch
from parameterized import parameterized

import tensorrt_llm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import skip_bf16_pre_ampere, unittest_name_func

FP8_E4M3_MAX = 448.0


class TestDynamicFP8QuantDequant(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([(torch.float32), (torch.float16), (torch.bfloat16)],
                          name_func=unittest_name_func)
    def test_quantization_activation_scales(self, dtype):
        # Skip tests that are not supported in pre-ampere architecture
        skip_bf16_pre_ampere(dtype)

        A = torch.tensor([[1, 2, 3], [2, 4, 6]], dtype=dtype)
        _, s = torch.ops.tensorrt_llm.quantize_e4m3_activation(A)
        s_ref = (torch.max(A, -1)[0].float() / FP8_E4M3_MAX)

        torch.testing.assert_close(s_ref.to(dtype), s.squeeze().to(dtype))

    @parameterized.expand([(torch.float32), (torch.float16), (torch.bfloat16)],
                          name_func=unittest_name_func)
    def test_quantization_weight_scales(self, dtype):
        # Skip tests that are not supported in pre-ampere architecture
        skip_bf16_pre_ampere(dtype)

        A = torch.tensor([[1, 2, 3], [2, 4, 6]], dtype=dtype)
        _, s = torch.ops.tensorrt_llm.quantize_e4m3_weight(A)
        s_ref = (torch.max(A, 0)[0].float() / FP8_E4M3_MAX)

        torch.testing.assert_close(s_ref.to(dtype), s.squeeze().to(dtype))

    @parameterized.expand([(torch.float32), (torch.float16), (torch.bfloat16)],
                          name_func=unittest_name_func)
    def test_quantization_per_tensor_scales(self, dtype):
        # Skip tests that are not supported in pre-ampere architecture
        skip_bf16_pre_ampere(dtype)

        A = torch.tensor([[1, 2, 3], [2, 4, 6]], dtype=dtype)
        _, s = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(A)
        s_ref = (A.flatten().max().float() / FP8_E4M3_MAX)

        torch.testing.assert_close(s_ref.to(dtype), s.squeeze().to(dtype))

    @parameterized.expand([(torch.float32), (torch.float16), (torch.bfloat16)],
                          name_func=unittest_name_func)
    def test_quantization_dequantization_activation(self, dtype):
        # Skip tests that are not supported in pre-ampere architecture
        skip_bf16_pre_ampere(dtype)

        n = 512
        m = 1024
        A = torch.randn((n, m), dtype=dtype)

        assert A.stride() == (m, 1)

        qA, s = torch.ops.tensorrt_llm.quantize_e4m3_activation(A)

        assert qA.shape == A.shape
        assert qA.shape[:-1] == s.shape[:-1]
        assert s.shape[-1] == 1
        assert s.dtype == A.dtype
        assert qA.dtype == torch.int8

        s_ref = (torch.max(A.float().abs(), -1)[0] / FP8_E4M3_MAX)
        torch.testing.assert_close(s_ref.to(dtype), s.squeeze().to(dtype))

        B = torch.ops.tensorrt_llm.dequantize_e4m3_activation(qA, s)

        assert B.shape == A.shape
        assert B.dtype == A.dtype

        torch.testing.assert_close(A, B, atol=0.2, rtol=0.01)

        # testing exact match
        A = torch.randint(0, 8, (n, m), dtype=dtype)

        qA, s = torch.ops.tensorrt_llm.quantize_e4m3_activation(A)
        B = torch.ops.tensorrt_llm.dequantize_e4m3_activation(qA, s)

        torch.testing.assert_close(A, B)

    @parameterized.expand([(torch.float32), (torch.float16), (torch.bfloat16)],
                          name_func=unittest_name_func)
    def test_quantization_dequantization_weight(self, dtype):
        # Skip tests that are not supported in pre-ampere architecture
        skip_bf16_pre_ampere(dtype)

        n = 512
        m = 1024
        A = torch.randn((n, m), dtype=dtype)

        assert A.stride() == (m, 1)

        qA, s = torch.ops.tensorrt_llm.quantize_e4m3_weight(A)

        assert qA.shape == A.shape
        assert qA.shape[1:] == s.shape[1:]
        assert s.shape[0] == 1

        s_ref = (torch.max(A.float().abs(), 0)[0] / FP8_E4M3_MAX)
        torch.testing.assert_close(s_ref.to(dtype), s.squeeze().to(dtype))

        B = torch.ops.tensorrt_llm.dequantize_e4m3_weight(qA, s)

        torch.testing.assert_close(A, B, atol=0.2, rtol=0)

        # testing exact match
        A = torch.randint(0, 8, (n, m), dtype=dtype)

        qA, s = torch.ops.tensorrt_llm.quantize_e4m3_weight(A)
        B = torch.ops.tensorrt_llm.dequantize_e4m3_weight(qA, s)

        torch.testing.assert_close(A, B)

    @parameterized.expand([(torch.float32), (torch.float16), (torch.bfloat16)],
                          name_func=unittest_name_func)
    def test_quantization_dequantization_per_tensor(self, dtype):
        # Skip tests that are not supported in pre-ampere architecture
        skip_bf16_pre_ampere(dtype)

        n = 512
        m = 1024
        A = torch.randn((n, m), dtype=dtype)

        qA, s = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(A)

        assert qA.shape == A.shape
        assert qA.dim() == s.dim()
        assert s.numel() == 1

        s_ref = (A.flatten().float().abs().max() / FP8_E4M3_MAX)
        torch.testing.assert_close(s_ref.to(dtype), s.squeeze().to(dtype))

        B = torch.ops.tensorrt_llm.dequantize_e4m3_per_tensor(qA, s)

        # per tensor is less accurate than others, so larger atol is used.
        torch.testing.assert_close(A, B, atol=0.25, rtol=0)

        # testing exact match
        A = torch.randint(0, 8, (n, m), dtype=dtype)

        qA, s = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(A)
        B = torch.ops.tensorrt_llm.dequantize_e4m3_per_tensor(qA, s)

        torch.testing.assert_close(A, B)


if __name__ == '__main__':
    unittest.main()
