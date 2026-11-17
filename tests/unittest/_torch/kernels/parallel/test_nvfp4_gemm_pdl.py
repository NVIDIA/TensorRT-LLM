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
"""PDL ordering of the CUTLASS NVFP4 GEMM against the kernel that produces its activation scale factors."""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._utils import is_sm_100f

FP8_MAX, E2M1_MAX, SF_VEC = 448.0, 6.0, 16
# dsv3_fused_a_gemm_op releases its dependents early only in its custom kernel, which it uses for 1-16 bf16 tokens
# with the 7168 -> 2112 or 6144 -> 2624 shapes. Other sizes fall back to cuBLAS, which leaves no race window, so the
# test would pass without the fix.
PRODUCER_TOKENS, PRODUCER_K, PRODUCER_N = 16, 7168, 2112

pytestmark = pytest.mark.skipif(
    not is_sm_100f() or not get_env_enable_pdl(),
    reason="Needs an SM10x GPU and PDL (TRTLLM_ENABLE_PDL=1)",
)


def _quantize(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = (FP8_MAX * E2M1_MAX / t.float().abs().amax()).view(1)
    fp4, sf = torch.ops.trtllm.fp4_quantize(t, scale, SF_VEC, False)
    return fp4, sf, scale


@pytest.mark.parametrize("m", [1, 4, 16])
def test_nvfp4_gemm_waits_for_activation_scale_factors(m: int) -> None:
    """Graph-captured producer -> NVFP4 GEMM must match running the GEMM after the producer finished.

    The producer, dsv3_fused_a_gemm_op, allows its dependents to launch before it computes anything, so a
    GEMM that loads its activation scale factors before griddepcontrol.wait reads their previous contents.
    """
    torch.manual_seed(0)
    k = 4096
    sf_bytes = 128 * (k // SF_VEC)  # swizzled activation scale factors for up to 128 rows
    producer_weight = (
        torch.randn(PRODUCER_N, PRODUCER_K, dtype=torch.bfloat16, device="cuda") * 0.05
    )
    act_fp4, _, act_scale = _quantize(torch.randn(m, k, dtype=torch.bfloat16, device="cuda"))
    weight_fp4, weight_sf, weight_scale = _quantize(
        torch.randn(256, k, dtype=torch.bfloat16, device="cuda")
    )
    alpha = 1.0 / (act_scale * weight_scale)

    def producer(x: torch.Tensor) -> torch.Tensor:
        return torch.ops.trtllm.dsv3_fused_a_gemm_op(x, producer_weight.t(), None, None)

    def consumer(sf_source: torch.Tensor) -> torch.Tensor:
        act_sf = sf_source.view(torch.uint8).reshape(-1)[:sf_bytes]
        return torch.ops.trtllm.nvfp4_gemm(
            act_fp4,
            weight_fp4,
            act_sf,
            weight_sf,
            alpha,
            torch.bfloat16,
            allowed_backends="cutlass",
        )

    producer_input = torch.randn(PRODUCER_TOKENS, PRODUCER_K, dtype=torch.bfloat16, device="cuda")
    consumer(producer(producer_input))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sf_source = producer(producer_input)
        graph_output = consumer(sf_source)

    stale_runs = 0
    num_runs = 100
    for _ in range(num_runs):
        producer_input.copy_(torch.randn_like(producer_input))
        graph.replay()
        torch.cuda.synchronize()
        # The scale factors are complete here, so this is the GEMM result the graph must reproduce.
        expected = consumer(sf_source)
        torch.cuda.synchronize()
        # Scale factors are raw producer bytes, so some products are NaN/Inf; compare bitwise after mapping them.
        stale_runs += int(not torch.equal(graph_output.nan_to_num(), expected.nan_to_num()))
    assert stale_runs == 0, (
        f"NVFP4 GEMM read stale activation scale factors in {stale_runs}/{num_runs} runs"
    )
