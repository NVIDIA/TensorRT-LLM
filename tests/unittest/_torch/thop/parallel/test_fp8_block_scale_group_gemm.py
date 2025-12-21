# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import math
import random

import pytest
import torch
from _torch.helpers import calc_diff, per_block_cast_to_fp8
from utils.util import getSMVersion, isSM100Family

from tensorrt_llm._torch.autotuner import autotune


@pytest.mark.skipif(
    not isSM100Family(),
    reason="The test is for Blackwell only. Current SM is %d." % getSMVersion(),
)
@pytest.mark.parametrize("num_experts", [72])
@pytest.mark.parametrize("k", [1536])
@pytest.mark.parametrize("n", [2560])
@pytest.mark.parametrize("max_tokens_per_group", [10, 50, 100, 128, 256, 512, 1000, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_cute_dsl_fp8_block_scale_group_gemm(dtype, num_experts, k, n, max_tokens_per_group):
    random.seed(0)
    torch.random.manual_seed(0)

    group_m = []
    for i in range(num_experts):
        group_m.append(random.randint(0, max_tokens_per_group))
    group_m = torch.tensor(group_m, dtype=torch.int32, device="cuda")
    group_m_cum = torch.cumsum(group_m, dim=0)
    group_offset = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"), group_m_cum], dim=0)
    group_offset = group_offset.to(torch.int32)

    m = sum(group_m)
    a = torch.randn((m, k), device="cuda", dtype=dtype) / k
    b = torch.randn((num_experts, n, k), device="cuda", dtype=dtype) / k
    output_expected = torch.zeros((m, n), device="cuda", dtype=dtype)

    for i in range(num_experts):
        start = group_offset[i]
        end = group_offset[i + 1]
        output_expected[start:end, :] = torch.einsum("mk,nk->mn", a[start:end, :], b[i, :, :])

    a_fp8, a_scale = torch.ops.trtllm.fp8_quantize_1x128(a)
    b_fp8 = torch.empty(num_experts, n, k, dtype=torch.float8_e4m3fn, device="cuda")
    b_scale = torch.empty(
        num_experts, math.ceil(n / 128), math.ceil(k / 128), dtype=torch.float32, device="cuda"
    )
    for i in range(num_experts):
        cur_b, cur_b_scale = per_block_cast_to_fp8(b[i, :, :])
        b_fp8[i, :, :] = cur_b
        b_scale[i, :, :] = cur_b_scale

    with autotune():
        output = torch.ops.trtllm.cute_dsl_fp8_group_blockwise_gemm_blackwell(
            input=a_fp8,
            weight=b_fp8,
            input_scale=a_scale,
            weight_scale=b_scale,
            group_offset=group_offset,
        )
    output = torch.ops.trtllm.cute_dsl_fp8_group_blockwise_gemm_blackwell(
        input=a_fp8,
        weight=b_fp8,
        input_scale=a_scale,
        weight_scale=b_scale,
        group_offset=group_offset,
    )

    diff = calc_diff(output, output_expected)
    assert diff < 1e-3
    torch.testing.assert_close(output, output_expected, atol=1e-3, rtol=1e-3)
