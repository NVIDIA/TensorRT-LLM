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

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.util import skip_pre_blackwell_unittest, unittest_name_func


# torch.ops.trtllm.fp4_quantize does not currently batch
# This calls it B times as a workaround for now
def fp4_quantize_batches(mat: torch.Tensor, global_sf: torch.Tensor,
                         sf_vec_size: int):
    num_batches = mat.size(0)

    q_mats = []
    q_sfs = []

    for b in range(num_batches):
        batch = mat[b, :, :]

        q_mat, q_sf = torch.ops.trtllm.fp4_quantize(batch.half().cuda(),
                                                    global_sf.cuda(),
                                                    sf_vec_size, False)

        q_mats.append(q_mat)
        q_sfs.append(q_sf)

    result = (torch.stack(q_mats), torch.stack(q_sfs))

    return result


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

        a_fp4, a_sf = fp4_quantize_batches(mat_a, a_global_sf, sf_vec_size)
        b_fp4, b_sf = fp4_quantize_batches(mat_b, b_global_sf, sf_vec_size)

        a_pt_batched = e2m1_and_ufp8_scale_batches(a_fp4.cpu(), a_sf.cpu(),
                                                   1 / a_global_sf, sf_vec_size)

        b_pt_batched = e2m1_and_ufp8_scale_batches(b_fp4.cpu(), b_sf.cpu(),
                                                   1 / b_global_sf, sf_vec_size)

        c = (torch.ops.trtllm.fp4_bmm_kmajor(a_fp4, b_fp4, a_sf, b_sf,
                                             ab_global_sf, False).float().cpu())

        torch.cuda.synchronize()

        c_ref = prepare_ref_output(a_pt_batched, b_pt_batched)
        self.assertTrue(torch.allclose(c_ref, c, atol=1e-2, rtol=1e-2))


class TestProfiling(unittest.TestCase):

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
    def test_fp4_bmm_torch_profiling(self, b: int, m: int, n: int, k: int):
        mat_a = torch.randn([b, m, k], dtype=torch.float32)
        mat_b = torch.randn([b, n, k], dtype=torch.float32)
        a_global_sf = (448 * 6) / mat_a.abs().max().float()
        b_global_sf = (448 * 6) / mat_b.abs().max().float()
        ab_global_sf = 1 / (a_global_sf * b_global_sf)
        ab_global_sf = ab_global_sf.cuda()
        sf_vec_size = 16

        a_fp4, a_sf = fp4_quantize_batches(mat_a, a_global_sf, sf_vec_size)
        b_fp4, b_sf = fp4_quantize_batches(mat_b, b_global_sf, sf_vec_size)

        profiler = torch.classes.trtllm.FP4BmmRunner.get_instance(torch.half)
        buckets = [1, 16, 32, 48, 64, 1024, 2048, 4096]
        profiler.run_profile(n, k, b, buckets)

        best_config_idx = profiler.get_best_config_id(m, n, k, b)

        c = profiler.run_bmm_kmajor(a_fp4, b_fp4, a_sf, b_sf, ab_global_sf,
                                    False, best_config_idx)

        c_ref = torch.ops.trtllm.fp4_bmm_kmajor(a_fp4, b_fp4, a_sf, b_sf,
                                                ab_global_sf, False)

        torch.cuda.synchronize()

        self.assertTrue(torch.allclose(c_ref, c, atol=1e-2, rtol=1e-2))
