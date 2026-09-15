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
import torch.nn.functional as F
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
            [234, 1024, 1024],
            [1024, 4096, 4096],
        ]),
        name_func=unittest_name_func,
    )
    @skip_pre_blackwell_unittest
    def test_w4a8_mxfp4_mxfp8_gemm_torch(self, m: int, n: int, k: int):
        torch.manual_seed(1234)

        mat_a = torch.randn([m, k], dtype=torch.bfloat16).cuda()
        mat_b = torch.randn([n, k], dtype=torch.bfloat16).cuda()

        fp8_a, a_block_sf = torch.ops.trtllm.mxfp8_quantize(mat_a, True)
        global_scale_b = (448 * 6) / mat_b.abs().max().float()
        fp4_b, b_block_sf = torch.ops.trtllm.fp4_quantize(
            mat_b, global_scale_b, 32, True)

        alpha = torch.tensor([1.234], dtype=torch.float32).cuda()

        c = (torch.ops.trtllm.w4a8_mxfp4_fp8_gemm(fp8_a, fp4_b, a_block_sf,
                                                  b_block_sf, alpha,
                                                  torch.bfloat16))
        c_ref = mat_a @ mat_b.T * alpha
        assert F.cosine_similarity(c.flatten(), c_ref.flatten(),
                                   dim=0).item() > 0.98
