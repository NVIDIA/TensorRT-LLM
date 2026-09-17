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

import pytest
import torch
import torch.nn.functional as F
from utils.util import getSMVersion

import tensorrt_llm._torch.custom_ops.torch_custom_ops  # noqa: F401
from tensorrt_llm._torch.custom_ops.torch_custom_ops import _map_to_mxfp8_large_m_bucket


@pytest.mark.skipif(
    getSMVersion() not in (100, 103),
    reason="MXFP8 GEMM is supported on SM100 and SM103 only. Current SM is %d." % getSMVersion(),
)
@pytest.mark.parametrize(
    "m,n,k",
    [
        (6599, 9216, 6144),
        (6599, 6144, 3072),
        (14906, 6144, 8192),
        (14906, 6144, 6144),
        (29765, 24576, 6144),
        (29765, 6144, 12288),
        (8193, 9216, 6144),
    ],
)
def test_mxfp8_mxfp8_gemm_large_m(m: int, n: int, k: int):
    """The generic fallback agrees with BF16 GEMM for representative shapes."""
    torch.manual_seed(42)
    mat_a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    mat_b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    fp8_a, a_block_sf = torch.ops.trtllm.mxfp8_quantize(mat_a, True)
    fp8_b, b_block_sf = torch.ops.trtllm.mxfp8_quantize(mat_b, True)
    global_scale = torch.ones((1,), device="cuda", dtype=torch.float32)

    output = torch.ops.trtllm.mxfp8_mxfp8_gemm(
        fp8_a,
        a_block_sf,
        fp8_b,
        b_block_sf,
        global_scale,
        torch.bfloat16,
    )
    output_ref = mat_a @ mat_b.t()

    assert F.cosine_similarity(output.flatten(), output_ref.flatten(), dim=0).item() > 0.98


@pytest.mark.skipif(
    getSMVersion() not in (100, 103),
    reason="MXFP8 tactic runner requires SM100 or SM103. Current SM is %d." % getSMVersion(),
)
def test_mxfp8_mxfp8_runner_tactics():
    """Every exposed tactic and the generic fallback produce aligned output."""
    torch.manual_seed(42)
    m, n, k = 128, 256, 512
    mat_a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    mat_b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    fp8_a, a_block_sf = torch.ops.trtllm.mxfp8_quantize(mat_a, True)
    fp8_b, b_block_sf = torch.ops.trtllm.mxfp8_quantize(mat_b, True)
    global_scale = torch.ones((1,), device="cuda", dtype=torch.float32)
    output_ref = mat_a @ mat_b.t()

    runner = torch.classes.trtllm.MXFP8GemmRunner(torch.bfloat16)
    num_tactics = runner.get_num_configs()
    assert num_tactics > 0

    for tactic in [-1, *range(num_tactics)]:
        output = runner.run_gemm(
            fp8_a,
            a_block_sf,
            fp8_b,
            b_block_sf,
            global_scale,
            tactic,
        )
        similarity = F.cosine_similarity(output.flatten(), output_ref.flatten(), dim=0)
        assert similarity.item() > 0.98


@pytest.mark.skipif(
    getSMVersion() not in (100, 103),
    reason="MXFP8 tactic cache requires SM100 or SM103. Current SM is %d." % getSMVersion(),
)
def test_mxfp8_mxfp8_native_tactic_cache():
    """The direct op uses cached tactics and preserves the generic fallback."""
    torch.manual_seed(42)
    m, n, k = 128, 256, 512
    mat_a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    mat_b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    fp8_a, a_block_sf = torch.ops.trtllm.mxfp8_quantize(mat_a, True)
    fp8_b, b_block_sf = torch.ops.trtllm.mxfp8_quantize(mat_b, True)
    global_scale = torch.ones((1,), device="cuda", dtype=torch.float32)
    runner = torch.classes.trtllm.MXFP8GemmRunner(torch.bfloat16)

    try:
        runner.clear_tactic_cache()
        assert runner.get_cached_tactic(m, n, k) == -2

        runner.register_tactic(m, n, k, 0)
        assert runner.get_cached_tactic(m, n, k) == 0
        cached_output = torch.ops.trtllm.mxfp8_mxfp8_gemm(
            fp8_a,
            a_block_sf,
            fp8_b,
            b_block_sf,
            global_scale,
            torch.bfloat16,
        )
        explicit_output = runner.run_gemm(
            fp8_a,
            a_block_sf,
            fp8_b,
            b_block_sf,
            global_scale,
            0,
        )
        torch.testing.assert_close(cached_output, explicit_output, rtol=0, atol=0)

        runner.register_tactic(m, n, k, -1)
        assert runner.get_cached_tactic(m, n, k) == -1
        cached_fallback_output = torch.ops.trtllm.mxfp8_mxfp8_gemm(
            fp8_a,
            a_block_sf,
            fp8_b,
            b_block_sf,
            global_scale,
            torch.bfloat16,
        )
        explicit_fallback_output = runner.run_gemm(
            fp8_a,
            a_block_sf,
            fp8_b,
            b_block_sf,
            global_scale,
            -1,
        )
        torch.testing.assert_close(cached_fallback_output, explicit_fallback_output, rtol=0, atol=0)
    finally:
        runner.clear_tactic_cache()


@pytest.mark.skipif(
    getSMVersion() not in (100, 103),
    reason="MXFP8 tactic cache requires SM100 or SM103. Current SM is %d." % getSMVersion(),
)
@pytest.mark.parametrize(
    "lower_bound,upper_bound,bucket",
    [
        (6553, 8192, 8192),
        (13106, 16384, 16384),
        (19659, 32768, 32768),
    ],
)
def test_mxfp8_native_tactic_cache_large_m_bucket_boundaries(
    lower_bound: int, upper_bound: int, bucket: int
):
    """C++ cache bucketing stays aligned with the Python tuning profiles."""
    n, k = 9216, 6144
    runner = torch.classes.trtllm.MXFP8GemmRunner(torch.bfloat16)

    try:
        runner.clear_tactic_cache()
        runner.register_tactic(bucket, n, k, -1)

        assert _map_to_mxfp8_large_m_bucket(lower_bound) == bucket
        assert _map_to_mxfp8_large_m_bucket(upper_bound) == bucket
        assert runner.get_cached_tactic(lower_bound, n, k) == -1
        assert runner.get_cached_tactic(upper_bound, n, k) == -1
        assert _map_to_mxfp8_large_m_bucket(lower_bound - 1) == lower_bound - 1
        assert _map_to_mxfp8_large_m_bucket(upper_bound + 1) == upper_bound + 1
        assert runner.get_cached_tactic(lower_bound - 1, n, k) == -2
        assert runner.get_cached_tactic(upper_bound + 1, n, k) == -2
    finally:
        runner.clear_tactic_cache()
