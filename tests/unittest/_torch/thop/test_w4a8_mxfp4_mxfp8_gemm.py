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
from utils.util import skip_pre_blackwell_unittest, unittest_name_func

import tensorrt_llm


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level("warning")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    @parameterized.expand(
        list([
            [128, 128, 256],
            [234, 512, 512],
            [1024, 1024, 1024],
        ]),
        name_func=unittest_name_func,
    )
    @skip_pre_blackwell_unittest
    def test_w4a8_mxfp4_mxfp8_gemm_torch(self, m: int, n: int, k: int):
        mat_a = torch.randn([m, k],
                            dtype=torch.float32).cuda().to(torch.float8_e4m3fn)
        mat_a_ref = mat_a.to(torch.float32)

        mat_b = torch.ones([n, k // 2], dtype=torch.float32).to(
            torch.uint8).fill_(34).cuda()
        mat_b_ref = torch.ones([n, k], dtype=torch.float32).cuda()

        a_block_sf = torch.ones([m, k // 32], dtype=torch.float32).cuda()
        a_block_sf = a_block_sf.to(torch.uint8).fill_(127)

        b_block_sf = torch.ones([n, k // 32], dtype=torch.float32).cuda()
        b_block_sf = b_block_sf.to(torch.uint8).fill_(127)

        a_sf = torch.tensor([1.234], dtype=torch.float32).cuda()

        def random_noise(pos, mat_ref, mat):
            for _pos in pos:
                mat_ref[_pos[0], _pos[1] * 32:(_pos[1] + 1) * 32] *= 2
                mat[_pos[0], _pos[1]] = 128

        random_noise([[2, 1]], mat_a_ref, a_block_sf)
        random_noise([[61, 3]], mat_a_ref, a_block_sf)
        random_noise([[22, 4]], mat_a_ref, a_block_sf)

        random_noise([[61, 3]], mat_b_ref, b_block_sf)
        random_noise([[22, 4]], mat_b_ref, b_block_sf)

        mat_b[0][0] = 36
        mat_b_ref[0][0] = 2.0

        a_block_sf = torch.ops.trtllm.block_scale_interleave(a_block_sf)
        b_block_sf = torch.ops.trtllm.block_scale_interleave(b_block_sf)

        c = (torch.ops.trtllm.w4a8_mxfp4_fp8_gemm(mat_a, mat_b, a_block_sf,
                                                  b_block_sf, a_sf,
                                                  torch.bfloat16))
        c_ref = (mat_a_ref @ mat_b_ref.T * a_sf).to(torch.bfloat16)

        assert torch.allclose(c_ref, c, atol=1e-2, rtol=1e-2)
