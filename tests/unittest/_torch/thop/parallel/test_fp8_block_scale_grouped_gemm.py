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
from utils.util import getSMVersion

from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
    FP8BlockScalingGroupedGemmInputsHelper,
)


def _prepare_fp8_block_scale_grouped_gemm_inputs(dtype, num_tokens, num_experts, top_k, k, n):
    """Prepare inputs and reference output for fp8 block-scale grouped GEMM."""
    random.seed(0)
    torch.random.manual_seed(0)

    helper = FP8BlockScalingGroupedGemmInputsHelper(
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_experts,
        local_expert_offset=0,
    )
    num_tokens_per_expert = helper.generate_num_tokens_per_expert(num_tokens, approx_max_load=True)

    group_offset = torch.zeros(num_experts + 1, dtype=torch.int32, device="cuda")
    group_offset[1:] = torch.cumsum(
        torch.tensor(num_tokens_per_expert, dtype=torch.int32, device="cuda"), dim=0
    )

    m = int(group_offset[-1].item())
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

    return a_fp8, a_scale, b_fp8, b_scale, group_offset, output_expected


@pytest.mark.skipif(
    getSMVersion() not in (100, 103),
    reason="The test is for SM100/SM103. Current SM is %d." % getSMVersion(),
)
@pytest.mark.parametrize("num_tokens", [1024, 8192])
@pytest.mark.parametrize("num_experts, top_k", [(72, 6), (256, 8)])
@pytest.mark.parametrize("k, n", [(7168, 2048), (2048, 7168), (2560, 1536), (1536, 2560)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_cute_dsl_fp8_block_scale_grouped_gemm(dtype, num_tokens, num_experts, top_k, k, n):
    a_fp8, a_scale, b_fp8, b_scale, group_offset, output_expected = (
        _prepare_fp8_block_scale_grouped_gemm_inputs(dtype, num_tokens, num_experts, top_k, k, n)
    )

    with AutoTuner.get().capture() as tactics_capture:
        output = torch.ops.trtllm.cute_dsl_fp8_blockwise_grouped_gemm_blackwell(
            input=a_fp8,
            weight=b_fp8,
            input_scale=a_scale,
            weight_scale=b_scale,
            group_offset=group_offset,
            num_experts=num_experts,
            top_k=1,
            num_local_experts=num_experts,
            local_expert_offset=0,
        )

    tactics_list = list(tactics_capture)
    print(f"  Found {len(tactics_list)} tactics")
    assert len(tactics_list) > 0, "No valid tactics found"

    for tactic_idx, tactic_config in enumerate(tactics_list):
        runner, tactic_value = tactic_config[0]
        runner_name = runner.__class__.__name__

        with AutoTuner.get().replay(tactic_config):
            output = torch.ops.trtllm.cute_dsl_fp8_blockwise_grouped_gemm_blackwell(
                input=a_fp8,
                weight=b_fp8,
                input_scale=a_scale,
                weight_scale=b_scale,
                group_offset=group_offset,
                num_experts=num_experts,
                top_k=1,
                num_local_experts=num_experts,
                local_expert_offset=0,
            )

        diff = calc_diff(output, output_expected)
        assert diff < 1e-3, (
            f"Tactic {tactic_idx} ({runner_name}, tactic={tactic_value}) failed: diff={diff}"
        )
        torch.testing.assert_close(output, output_expected, atol=1e-3, rtol=1e-3)
