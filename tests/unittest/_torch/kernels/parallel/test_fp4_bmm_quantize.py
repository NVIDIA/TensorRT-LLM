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
from utils.util import (skip_pre_blackwell, skip_pre_blackwell_unittest,
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


def e2m1_and_ufp8_scale_batches(mat_fp4: torch.Tensor,
                                scale_tensor: torch.Tensor,
                                global_scale_tensor: torch.Tensor,
                                sf_vec_size: int,
                                ufp8_type: int = 1):
    num_batches = mat_fp4.size(0)

    tensors = [
        e2m1_and_ufp8_scale_to_float_tensor_v2(mat_fp4[b, :, :],
                                               scale_tensor[b, :],
                                               global_scale_tensor, sf_vec_size)
        for b in range(num_batches)
    ]

    result = torch.stack(tensors)

    return result


def prepare_ref_output(a_pt: torch.Tensor, b_pt: torch.Tensor):
    num_batches = a_pt.size(0)

    tensors = [
        torch.nn.functional.linear(a_pt[b, :, :], b_pt[b, :, :])
        for b in range(num_batches)
    ]

    result = torch.stack(tensors)

    return result


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level("warning")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    @parameterized.expand(
        list([
            [1, 7, 128, 64],
            [10, 7, 128, 64],
            [1, 1024, 1024, 1024],
            [10, 1024, 1024, 1024],
        ]),
        name_func=unittest_name_func,
    )
    @skip_pre_blackwell_unittest
    def test_fp4_bmm_torch(self, b: int, m: int, n: int, k: int):

        mat_a = torch.randn([b, m, k], dtype=torch.float32)
        mat_b = torch.randn([b, n, k], dtype=torch.float32)
        a_global_sf = (448 * 6) / mat_a.abs().max().float()
        b_global_sf = (448 * 6) / mat_b.abs().max().float()
        ab_global_sf = 1 / (a_global_sf * b_global_sf)
        ab_global_sf = ab_global_sf.cuda()

        sf_vec_size = 16

        a_fp4, a_sf = torch.ops.trtllm.fp4_batched_quantize(
            mat_a.to(torch.half).cuda(), a_global_sf.cuda(), sf_vec_size, False)
        b_fp4, b_sf = torch.ops.trtllm.fp4_batched_quantize(
            mat_b.to(torch.half).cuda(), b_global_sf.cuda(), sf_vec_size, False)

        a_pt_batched = e2m1_and_ufp8_scale_batches(a_fp4.cpu(), a_sf.cpu(),
                                                   1 / a_global_sf, sf_vec_size)

        b_pt_batched = e2m1_and_ufp8_scale_batches(b_fp4.cpu(), b_sf.cpu(),
                                                   1 / b_global_sf, sf_vec_size)

        c = (torch.ops.trtllm.fp4_bmm(
            a_fp4, b_fp4, a_sf, b_sf, ab_global_sf,
            fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4).float().cpu())

        torch.cuda.synchronize()

        c_ref = prepare_ref_output(a_pt_batched, b_pt_batched)
        self.assertTrue(torch.allclose(c_ref, c, atol=1e-2, rtol=1e-2))


@skip_pre_blackwell
@pytest.mark.parametrize(
    "b,m,k,dtype,use_ue8m0",
    [
        (1, 128, 16, torch.half, False),
        (2, 255, 32, torch.bfloat16, False),
        (5, 512, 48, torch.bfloat16, True),
        (15, 1023, 1040, torch.half, True),
        (129, 1023, 1040, torch.half, False),
    ],
)
def test_fp4_batched_quantize(b, m, k, dtype, use_ue8m0):
    a = torch.randn([b, m, k], dtype=torch.float32).to(dtype).float()
    a_global_sf = (448 * 6) / a.abs().max().float()
    sf_vec_size = 16

    a_fp4, a_sf = torch.ops.trtllm.fp4_batched_quantize(
        a.to(dtype).cuda(), a_global_sf.cuda(), sf_vec_size, use_ue8m0)

    assert a_fp4.dim() == 3, "batched fp4 tensor must have 3 dimensions!"
    assert a_sf.dim(
    ) == 2, "batched fp4 tensor scale factor must have 2 dimensions!"

    a_pt = e2m1_and_ufp8_scale_batches(a_fp4.cpu(), a_sf.cpu(), 1 / a_global_sf,
                                       sf_vec_size)

    torch.cuda.synchronize()
    if not use_ue8m0:
        # The gap is too large for ue8m0, so we just make sure that it runs
        torch.testing.assert_allclose(a_pt, a, atol=1, rtol=0)


@skip_pre_blackwell
@pytest.mark.parametrize(
    "b,m,k",
    [
        (1, 128, 4),
        (1, 128, 8),
        (1, 128, 128),
        (1, 256, 4),
        (1, 2048, 4),
        (15, 128, 48),
        (2, 255, 32),
    ],
)
def test_fp4_sf_interleave(b, m, k):
    shape = [m, k] if b is None else [b, m, k]
    w = torch.randint(0, 256, shape, dtype=torch.uint8)
    w_cuda = w.cuda()

    # The cpu and cuda kernels are different
    w_out_cpu = torch.ops.trtllm.block_scale_interleave(w)
    w_out_cuda = torch.ops.trtllm.block_scale_interleave(w_cuda)
    torch.cuda.synchronize()

    torch.testing.assert_allclose(w_out_cpu.cuda(), w_out_cuda)
