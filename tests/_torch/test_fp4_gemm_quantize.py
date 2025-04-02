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

import pytest
import torch
from parameterized import parameterized

import tensorrt_llm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.util import skip_pre_blackwell_unittest, unittest_name_func


# Used by the (fp16 -> int4) quant layer + int4 gemm network.
def e2m1_and_ufp8_scale_to_float_tensor_v2(
    e2m1_tensor: torch.Tensor,
    ufp8_scale_tensor: torch.Tensor,
    global_scale_tensor: torch.Tensor,
    sf_vec_size,
    ufp8_type: int = 1,
):
    float_tensor = torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float_v2(
        e2m1_tensor, ufp8_scale_tensor, global_scale_tensor, sf_vec_size,
        ufp8_type)
    return float_tensor


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level("warning")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    @parameterized.expand(
        list([
            [1024, 1024, 1024],
            [7, 32, 32],
        ]),
        name_func=unittest_name_func,
    )
    @skip_pre_blackwell_unittest
    def test_fp4_quantize_gemm_torch(self, m, n, k):
        pytest.skip("https://nvbugs/5100633")
        a = torch.randn([m, k], dtype=torch.float32)
        b = torch.randn([n, k], dtype=torch.float32)
        a_global_sf = (448 * 6) / a.abs().max().float()
        b_global_sf = (448 * 6) / b.abs().max().float()
        ab_global_sf = 1 / (a_global_sf * b_global_sf)
        ab_global_sf = ab_global_sf.cuda()

        sf_vec_size = 16
        a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(a.half().cuda(),
                                                    a_global_sf.cuda(),
                                                    sf_vec_size, False)
        b_fp4, b_sf = torch.ops.trtllm.fp4_quantize(b.half().cuda(),
                                                    b_global_sf.cuda(),
                                                    sf_vec_size, False)

        a_pt = e2m1_and_ufp8_scale_to_float_tensor_v2(a_fp4.cpu(), a_sf.cpu(),
                                                      1 / a_global_sf,
                                                      sf_vec_size)
        b_pt = e2m1_and_ufp8_scale_to_float_tensor_v2(b_fp4.cpu(), b_sf.cpu(),
                                                      1 / b_global_sf,
                                                      sf_vec_size)

        c = (torch.ops.trtllm.fp4_gemm(a_fp4, b_fp4, a_sf, b_sf, ab_global_sf,
                                       False).float().cpu())

        torch.cuda.synchronize()
        c_pt = torch.nn.functional.linear(a_pt, b_pt)
        self.assertTrue(torch.allclose(c_pt, c, atol=1e-2, rtol=1e-2))

    @parameterized.expand(list([[1024, 1024, torch.half, False],
                                [2, 512, torch.bfloat16, False],
                                [13, 16, torch.half, True]]),
                          name_func=unittest_name_func)
    @skip_pre_blackwell_unittest
    def test_fp4_quantize_torch(self, m, k, dtype, use_ue8m0):
        a = torch.randn([m, k], dtype=torch.float32).to(dtype).float()
        a_global_sf = (448 * 6) / a.abs().max().float()
        sf_vec_size = 16

        a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(
            a.to(dtype).cuda(), a_global_sf.cuda(), sf_vec_size, use_ue8m0)

        a_pt = e2m1_and_ufp8_scale_to_float_tensor_v2(a_fp4.cpu(), a_sf.cpu(),
                                                      1 / a_global_sf,
                                                      sf_vec_size)

        torch.cuda.synchronize()
        if not use_ue8m0:
            # The gap is too large for ue8m0, so we just make sure that it runs
            self.assertTrue(torch.allclose(a_pt, a, atol=1, rtol=0))

    @parameterized.expand(list([[64, 64, torch.float8_e4m3fn, False],
                                [13, 16, torch.float8_e4m3fn, True]]),
                          name_func=unittest_name_func)
    @skip_pre_blackwell_unittest
    def test_fp4_quantize_torch_fp8(self, m, k, dtype, use_ue8m0):
        assert dtype == torch.float8_e4m3fn
        a = torch.randn([m, k], dtype=torch.float32)
        amax = a.abs().max().float()
        a_fp8 = (a / amax * 448).to(dtype)
        aq_fp32 = a_fp8.float() * amax / 448
        a_global_sf = (448 * 6) / amax
        sf_vec_size = 16

        a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(a_fp8.cuda(),
                                                    a_global_sf.cuda(),
                                                    sf_vec_size, use_ue8m0)

        a_pt = e2m1_and_ufp8_scale_to_float_tensor_v2(a_fp4.cpu(), a_sf.cpu(),
                                                      1 / a_global_sf,
                                                      sf_vec_size)

        torch.cuda.synchronize()

        if not use_ue8m0:
            # The gap is too large for ue8m0, so we just make sure that it runs
            self.assertTrue(torch.allclose(a_pt, aq_fp32, atol=1, rtol=0))


class TestProfiling(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level("warning")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    @parameterized.expand(
        list([
            [1024, 1024, 1024],
            [512, 32, 64],
            [7, 32, 32],
        ]),
        name_func=unittest_name_func,
    )
    @skip_pre_blackwell_unittest
    def test_fp4_quantize_gemm_torch_profiling(self, m: int, n: int, k: int):
        a = torch.randn([m, k], dtype=torch.float32)
        b = torch.randn([n, k], dtype=torch.float32)
        a_global_sf = (448 * 6) / a.abs().max().float()
        b_global_sf = (448 * 6) / b.abs().max().float()
        ab_global_sf = 1 / (a_global_sf * b_global_sf)
        ab_global_sf = ab_global_sf.cuda()

        profiler = torch.classes.trtllm.FP4GemmRunner.get_instance(torch.half)
        buckets = [1, 16, 32, 48, 64, 1024, 2048, 4096]
        profiler.run_profile(n, k, buckets)

        sf_vec_size = 16
        a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(a.half().cuda(),
                                                    a_global_sf.cuda(),
                                                    sf_vec_size, False)
        b_fp4, b_sf = torch.ops.trtllm.fp4_quantize(b.half().cuda(),
                                                    b_global_sf.cuda(),
                                                    sf_vec_size, False)

        a_pt = e2m1_and_ufp8_scale_to_float_tensor_v2(a_fp4.cpu(), a_sf.cpu(),
                                                      1 / a_global_sf,
                                                      sf_vec_size)
        torch.cuda.synchronize()

        b_pt = e2m1_and_ufp8_scale_to_float_tensor_v2(b_fp4.cpu(), b_sf.cpu(),
                                                      1 / b_global_sf,
                                                      sf_vec_size)

        c_ref = torch.ops.trtllm.fp4_gemm(a_fp4, b_fp4, a_sf, b_sf,
                                          ab_global_sf, False)

        best_config_idx = profiler.get_best_config_id(m, n, k)
        c_actual = profiler.run_gemm(a_fp4, b_fp4, a_sf, b_sf, ab_global_sf,
                                     False, best_config_idx)

        torch.cuda.synchronize()

        torch.testing.assert_close(c_actual, c_ref, atol=1e-2, rtol=0)
