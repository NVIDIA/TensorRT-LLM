# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical coverage for the Marlin NVFP4 Triton MoE combine path."""

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.fused_moe_marlin import _sum_topk_expert_outputs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "num_tokens,top_k,hidden_size",
    [
        (1, 1, 256),
        (3, 2, 257),
        (4, 6, 2688),
        (4, 8, 2688),
        (16, 2, 4096),
    ],
)
def test_triton_sum_topk_matches_aten_scatter_reduce(
    num_tokens: int, top_k: int, hidden_size: int
) -> None:
    """Triton FP32-accumulate combine matches BF16 ATen index_add_ within BF16 tolerance.

    The Triton kernel accumulates in FP32 before storing BF16, which is more
    numerically stable than BF16 index_add_ but produces a slightly different
    result. The tolerance (rtol=2e-2, atol=0.125) covers the worst-case BF16
    rounding gap between the two accumulation orders.
    """
    torch.manual_seed(0)
    expert_outputs = torch.randn(
        (num_tokens * top_k, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    expected = torch.zeros((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
    token_indices = torch.arange(num_tokens * top_k, device="cuda") // top_k
    expected.index_add_(0, token_indices, expert_outputs)

    actual = _sum_topk_expert_outputs(
        expert_outputs, num_tokens, top_k, hidden_size, torch.bfloat16
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=0.125)
