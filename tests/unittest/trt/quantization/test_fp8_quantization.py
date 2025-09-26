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

FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


class TestDynamicFP8QuantDequant(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        tensorrt_llm.logger.set_level('error')

    def _ref_quant(self, x_, x_scale_):
        x_ = x_.float()
        finfo = torch.finfo(torch.float8_e4m3fn)
        inv_scale = x_scale_.float().reciprocal()
        x_fp8_ = (x_ * inv_scale).clamp(min=finfo.min, max=finfo.max)
        return x_fp8_.to(torch.float8_e4m3fn)

    @parameterized.expand([(torch.float32), (torch.float16), (torch.bfloat16)],
                          name_func=unittest_name_func)
    def test_quantization_activation_scales(self, dtype):
        m = 11
        n = 11
        A = torch.randn((m, n), dtype=dtype).cuda()
        B, s = torch.ops.tensorrt_llm.quantize_e4m3_activation(A)
        s_ref = (torch.max(A.float().abs(), -1)[0].view(m, 1) /
                 FP8_E4M3_MAX).to(dtype)
        B_ref = self._ref_quant(A, s_ref)

        torch.testing.assert_close(s_ref, s)
        torch.testing.assert_close(B.float(), B_ref.float())

        B_s, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_activation(
            A, s.float())

        torch.testing.assert_close(B_s.float(), B_ref.float())

    @parameterized.expand([(torch.float32), (torch.float16), (torch.bfloat16)],
                          name_func=unittest_name_func)
    def test_quantization_weight_scales(self, dtype):
        m = 11
        n = 11
        A = torch.randn((m, n), dtype=dtype).cuda()
        B, s = torch.ops.tensorrt_llm.quantize_e4m3_weight(A)
        s_ref = (torch.max(A.float().abs(), 0)[0].view(1, n) /
                 FP8_E4M3_MAX).to(dtype)
        B_ref = self._ref_quant(A, s_ref)

        torch.testing.assert_close(s_ref, s)
        torch.testing.assert_close(B.float(), B_ref.float())

        B_s, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_weight(
            A, s.float())

        torch.testing.assert_close(B_s.float(), B_ref.float())

    @parameterized.expand([(torch.float32), (torch.float16), (torch.bfloat16)],
                          name_func=unittest_name_func)
    def test_quantization_per_tensor_scales(self, dtype):
        m = 11
        n = 11
        A = torch.randn((m, n), dtype=dtype).cuda()
        B, s = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(A)
        s_ref = (A.flatten().float().abs().max().view(1, 1) /
                 FP8_E4M3_MAX).to(dtype)
        B_ref = self._ref_quant(A, s_ref)

        torch.testing.assert_close(s_ref, s)
        torch.testing.assert_close(B.float(), B_ref.float())

        B_s, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            A, s.float())

        torch.testing.assert_close(B_s.float(), B_ref.float())

    @parameterized.expand([(torch.float32), (torch.float16), (torch.bfloat16)],
                          name_func=unittest_name_func)
    def test_quantization_dequantization_activation(self, dtype):
        n = 512
        m = 1024
        A = torch.randn((n, m), dtype=dtype).cuda()

        assert A.stride() == (m, 1)

        qA, s = torch.ops.tensorrt_llm.quantize_e4m3_activation(A)

        assert qA.shape == A.shape
        assert qA.shape[:-1] == s.shape[:-1]
        assert s.shape[-1] == 1
        assert s.dtype == A.dtype
        assert qA.dtype == torch.float8_e4m3fn

        s_ref = (torch.max(A.float().abs(), -1)[0].view(n, 1) /
                 FP8_E4M3_MAX).to(dtype)
        torch.testing.assert_close(s_ref, s)

        B = torch.ops.tensorrt_llm.dequantize_e4m3_activation(qA, s)

        assert B.shape == A.shape
        assert B.dtype == A.dtype

        torch.testing.assert_close(A, B, atol=0.2, rtol=0.01)

        # testing exact match
        A = torch.randint(0, 8, (n, m), dtype=dtype).cuda()

        qA, s = torch.ops.tensorrt_llm.quantize_e4m3_activation(A)
        B = torch.ops.tensorrt_llm.dequantize_e4m3_activation(qA, s)

        torch.testing.assert_close(A, B)

    @parameterized.expand([(torch.float32), (torch.float16), (torch.bfloat16)],
                          name_func=unittest_name_func)
    def test_quantization_dequantization_weight(self, dtype):
        n = 512
        m = 1024
        A = torch.randn((n, m), dtype=dtype).cuda()

        assert A.stride() == (m, 1)

        qA, s = torch.ops.tensorrt_llm.quantize_e4m3_weight(A)

        assert qA.shape == A.shape
        assert qA.shape[1:] == s.shape[1:]
        assert s.shape[0] == 1

        s_ref = (torch.max(A.float().abs(), 0)[0].view(1, m) /
                 FP8_E4M3_MAX).to(dtype)
        torch.testing.assert_close(s_ref, s)

        B = torch.ops.tensorrt_llm.dequantize_e4m3_weight(qA, s)

        torch.testing.assert_close(A, B, atol=0.2, rtol=0)

        # testing exact match
        A = torch.randint(0, 8, (n, m), dtype=dtype).cuda()

        qA, s = torch.ops.tensorrt_llm.quantize_e4m3_weight(A)
        B = torch.ops.tensorrt_llm.dequantize_e4m3_weight(qA, s)

        torch.testing.assert_close(A, B)

    @parameterized.expand([(torch.float32), (torch.float16), (torch.bfloat16)],
                          name_func=unittest_name_func)
    def test_quantization_dequantization_per_tensor(self, dtype):
        n = 512
        m = 1024
        A = torch.randn((n, m), dtype=dtype).cuda()

        qA, s = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(A)

        assert qA.shape == A.shape
        assert qA.dim() == s.dim()
        assert s.numel() == 1

        s_ref = (A.flatten().float().abs().max().view(1, 1) /
                 FP8_E4M3_MAX).to(dtype)
        torch.testing.assert_close(s_ref, s)

        B = torch.ops.tensorrt_llm.dequantize_e4m3_per_tensor(qA, s)

        # per tensor is less accurate than others, so larger atol is used.
        torch.testing.assert_close(A, B, atol=0.25, rtol=0)

        # testing exact match
        A = torch.randint(0, 8, (n, m), dtype=dtype).cuda()

        qA, s = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(A)
        B = torch.ops.tensorrt_llm.dequantize_e4m3_per_tensor(qA, s)

        torch.testing.assert_close(A, B)


if __name__ == '__main__':
    unittest.main()
