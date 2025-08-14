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

import pytest
import torch
from parameterized import parameterized
from utils.util import (skip_blackwell_geforce, skip_pre_blackwell_unittest,
                        unittest_name_func)

import tensorrt_llm
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils


# Used by the (fp16 -> int4) quant layer + int4 gemm network.
def e2m1_and_ufp8_scale_to_float_tensor_v2(
    e2m1_tensor: torch.Tensor,
    ufp8_scale_tensor: torch.Tensor,
    global_scale_tensor: torch.Tensor,
    sf_vec_size,
    ufp8_type: int = 1,
    is_sf_swizzled_layout: bool = True,
):
    float_tensor = torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float_v2(
        e2m1_tensor, ufp8_scale_tensor, global_scale_tensor, sf_vec_size,
        ufp8_type, is_sf_swizzled_layout)
    return float_tensor


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level("warning")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    @parameterized.expand(
        list([
            [1024, 1024, 1024],
            [256, 128, 512],
        ]),
        name_func=unittest_name_func,
    )
    @skip_pre_blackwell_unittest
    # TODO: add GEMM test for linear SF layout when kernel is ready
    def test_fp4_quantize_gemm_torch(self, m, n, k):
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

        c = (torch.ops.trtllm.fp4_gemm(
            a_fp4, b_fp4, a_sf, b_sf, ab_global_sf,
            fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4).float().cpu())

        torch.cuda.synchronize()
        c_pt = torch.nn.functional.linear(a_pt, b_pt)
        self.assertTrue(torch.allclose(c_pt, c, atol=1e-2, rtol=1e-2))

    @parameterized.expand(
        list([
            [1024, 1024, 1024],
            [256, 128, 512],
        ]),
        name_func=unittest_name_func,
    )
    @skip_pre_blackwell_unittest
    @skip_blackwell_geforce
    def test_fp4_quantize_gemm_trtllmgen(self, m, n, k):
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

        c = (torch.ops.trtllm.fp4_gemm_trtllmgen(a_fp4, b_fp4, a_sf, b_sf,
                                                 ab_global_sf,
                                                 False).float().cpu())

        torch.cuda.synchronize()
        c_pt = torch.nn.functional.linear(a_pt, b_pt)
        self.assertTrue(torch.allclose(c_pt, c, atol=1e-2, rtol=1e-2))

    @parameterized.expand(
        list([
            [1024, 1024, 1024],
            [128, 8, 256],
        ]),
        name_func=unittest_name_func,
    )
    @skip_pre_blackwell_unittest
    @skip_blackwell_geforce
    def test_fp4_fp8_gemm_trtllmgen(self, m, n, k):
        a = torch.randn([m, k], dtype=torch.float32)
        b = torch.randn([n, k], dtype=torch.float32)
        b_fp8, b_global_sf = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
            b.cuda())
        b_fp8 = b_fp8.view(torch.float8_e4m3fn)
        a_global_sf = 448.0 / a.abs().max().float()

        # FIXME: this depends on the kernel internals
        epilogue_tile_m = 128
        sf_vec_size = 32

        a_fp4, a_sf, rep_float = torch.ops.tensorrt_llm.float_to_e2m1_and_ufp8sf_scale(
            a * a_global_sf, sf_vec_size, 1, False)
        a_pt = e2m1_and_ufp8_scale_to_float_tensor_v2(a_fp4, a_sf,
                                                      1.0 / a_global_sf,
                                                      sf_vec_size, 1, False)
        b_pt = (b_fp8.to(torch.float32) * b_global_sf).cpu()
        c_pt = torch.nn.functional.linear(b_pt, a_pt)

        a_fp4_shuffled = fp4_utils.shuffle_matrix_a(a_fp4, epilogue_tile_m)
        # sf is swizzled as well.
        a_sf_shuffled = fp4_utils.shuffle_matrix_sf_a(a_sf.reshape(
            (m, -1)), epilogue_tile_m, sf_vec_size)

        ab_global_sf = b_global_sf / a_global_sf
        c = torch.ops.trtllm.fp4_fp8_gemm_trtllmgen(
            b_fp8, a_fp4_shuffled.cuda(),
            a_sf_shuffled.view(dtype=torch.float8_e4m3fn).cuda(),
            ab_global_sf.cuda())
        torch.cuda.synchronize()
        c = c.float().cpu()
        self.assertTrue(torch.allclose(c_pt, c, atol=1e-2, rtol=1e-2))

    @parameterized.expand(list([[1024, 1024, torch.half, False, True],
                                [2, 512, torch.bfloat16, False, True],
                                [2, 512, torch.bfloat16, True, True],
                                [16, 512, torch.half, True, True],
                                [16, 512, torch.half, False, True],
                                [16, 512, torch.half, True, False],
                                [16, 512, torch.half, False, False]]),
                          name_func=unittest_name_func)
    @skip_pre_blackwell_unittest
    def test_fp4_quantize_torch(self, m, k, dtype, use_ue8m0,
                                is_sf_swizzled_layout):
        a = torch.randn([m, k], dtype=torch.float32).to(dtype).float()
        if use_ue8m0:
            # Expand the range of the input to cover more cases
            a = a * 16

        a_global_sf = (448 * 6) / a.abs().max().float()
        sf_vec_size = 32 if use_ue8m0 else 16

        a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(
            a.to(dtype).cuda(), a_global_sf.cuda(), sf_vec_size, use_ue8m0,
            is_sf_swizzled_layout)

        sf_type = 0 if use_ue8m0 else 1

        a_pt = e2m1_and_ufp8_scale_to_float_tensor_v2(a_fp4.cpu(), a_sf.cpu(),
                                                      1 / a_global_sf,
                                                      sf_vec_size, sf_type,
                                                      is_sf_swizzled_layout)

        torch.cuda.synchronize()
        atol = 8 if use_ue8m0 else 1
        rtol = 0
        self.assertTrue(torch.allclose(a_pt, a, atol=atol, rtol=rtol))

    @parameterized.expand(list([[2, 16, torch.half, False, True],
                                [2, 16, torch.half, False, False],
                                [128, 512, torch.half, True, False],
                                [256, 128, torch.bfloat16, False, True],
                                [128, 128, torch.bfloat16, False, False],
                                [1024, 512, torch.bfloat16, True, False]]),
                          name_func=unittest_name_func)
    @skip_pre_blackwell_unittest
    def test_fp4_quantize_torch_different_sf_layout(self, m, k, dtype,
                                                    use_ue8m0,
                                                    is_sf_swizzled_layout):
        a = torch.randn([m, k], dtype=torch.float32).to(dtype).float()
        a_global_sf = (448 * 6) / a.abs().max().float()
        sf_vec_size = 32 if use_ue8m0 else 16

        a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(
            a.to(dtype).cuda(), a_global_sf.cuda(), sf_vec_size, use_ue8m0,
            is_sf_swizzled_layout)

        a_pt = e2m1_and_ufp8_scale_to_float_tensor_v2(a_fp4.cpu(), a_sf.cpu(),
                                                      1 / a_global_sf,
                                                      sf_vec_size, 1,
                                                      is_sf_swizzled_layout)

        torch.cuda.synchronize()
        if not use_ue8m0:
            # The gap is too large for ue8m0, so we just make sure that it runs
            self.assertTrue(torch.allclose(a_pt, a, atol=1, rtol=0))

    @parameterized.expand(list([[64, 64, torch.float8_e4m3fn, False, True],
                                [13, 32, torch.float8_e4m3fn, True, True],
                                [3, 48, torch.float8_e4m3fn, False, False],
                                [1024, 1024, torch.float8_e4m3fn, True,
                                 False]]),
                          name_func=unittest_name_func)
    @skip_pre_blackwell_unittest
    def test_fp4_quantize_torch_fp8(self, m, k, dtype, use_ue8m0,
                                    is_sf_swizzled_layout):
        assert dtype == torch.float8_e4m3fn
        a = torch.randn([m, k], dtype=torch.float32)
        amax = a.abs().max().float()
        a_fp8 = (a / amax * 448).to(dtype)
        aq_fp32 = a_fp8.float() * amax / 448
        a_global_sf = (448 * 6) / amax
        sf_vec_size = 32 if use_ue8m0 else 16

        a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(a_fp8.cuda(),
                                                    a_global_sf.cuda(),
                                                    sf_vec_size, use_ue8m0,
                                                    is_sf_swizzled_layout)

        a_pt = e2m1_and_ufp8_scale_to_float_tensor_v2(a_fp4.cpu(), a_sf.cpu(),
                                                      1 / a_global_sf,
                                                      sf_vec_size, 1,
                                                      is_sf_swizzled_layout)

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
        pytest.skip("https://nvbugs/5100633")
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

        c_ref = torch.ops.trtllm.fp4_gemm(
            a_fp4, b_fp4, a_sf, b_sf, ab_global_sf,
            fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4)

        best_config_idx = profiler.get_best_config_id(m, n, k)
        c_actual = profiler.run_gemm(a_fp4, b_fp4, a_sf, b_sf, ab_global_sf,
                                     False, best_config_idx)

        torch.cuda.synchronize()

        torch.testing.assert_close(c_actual, c_ref, atol=1e-2, rtol=0)
