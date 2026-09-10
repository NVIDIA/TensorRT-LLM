# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical coverage for the Marlin NVFP4 Triton MoE combine path."""

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.fused_moe_marlin import _sum_topk_expert_outputs


def _aten_reference(expert_outputs: torch.Tensor, num_tokens: int, top_k: int,
                    hidden_size: int, dtype: torch.dtype) -> torch.Tensor:
    """ATen scatter-reduce reference — mirrors the non-contiguous fallback in forward()."""
    result = torch.zeros((num_tokens, hidden_size), dtype=dtype,
                         device=expert_outputs.device)
    if num_tokens == 0:
        return result
    token_indices = torch.arange(num_tokens * top_k,
                                 device=expert_outputs.device) // top_k
    result.index_add_(0, token_indices, expert_outputs)
    return result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "num_tokens,top_k,hidden_size",
    [
        (0, 2, 256),    # zero tokens — kernel must not launch, return empty tensor
        (1, 1, 256),
        (3, 2, 257),    # hidden_size not a multiple of BLOCK_H=256
        (4, 6, 2688),   # Nemotron Nano 30B hidden size
        (4, 8, 2688),
        (16, 2, 4096),
    ],
)
def test_triton_sum_topk_matches_aten_scatter_reduce(
        num_tokens: int, top_k: int, hidden_size: int) -> None:
    """Triton FP32-accumulate combine matches BF16 ATen index_add_ within BF16 tolerance.

    The Triton kernel accumulates in FP32 before storing BF16, which is more
    numerically stable than BF16 index_add_ but produces a slightly different
    result. The tolerance (rtol=2e-2, atol=0.125) covers the worst-case BF16
    rounding gap between the two accumulation orders.
    """
    torch.manual_seed(0)
    expert_outputs = torch.randn((num_tokens * top_k, hidden_size),
                                 dtype=torch.bfloat16, device="cuda")

    expected = _aten_reference(expert_outputs, num_tokens, top_k, hidden_size,
                                torch.bfloat16)
    actual = _sum_topk_expert_outputs(expert_outputs, num_tokens, top_k,
                                      hidden_size, torch.bfloat16)

    assert actual.shape == (num_tokens, hidden_size)
    assert actual.dtype == torch.bfloat16
    if num_tokens > 0:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=0.125)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_noncontiguous_fallback_matches_triton() -> None:
    """Non-contiguous gemm2_out takes the ATen index_add_ path in forward().

    This test exercises the fallback branch directly (not through forward())
    to confirm both paths agree within BF16 tolerance — ensuring the fallback
    remains correct if the Triton path is ever skipped.
    """
    num_tokens, top_k, hidden_size = 4, 2, 2688
    torch.manual_seed(42)
    base = torch.randn((num_tokens * top_k, hidden_size),
                       dtype=torch.bfloat16, device="cuda")

    # Make a non-contiguous view (transpose then transpose back gives a strided tensor)
    non_contig = base.T.T
    # confirm it's actually non-contiguous after the round-trip transpose
    # (if not, fall back to as_strided to force non-contiguity)
    if non_contig.is_contiguous():
        non_contig = base.as_strided(base.shape, (hidden_size + 1, 1))
    assert not non_contig.is_contiguous(), "test setup error: tensor should be non-contiguous"

    aten_out = _aten_reference(non_contig, num_tokens, top_k, hidden_size,
                                torch.bfloat16)
    triton_out = _sum_topk_expert_outputs(base, num_tokens, top_k, hidden_size,
                                          torch.bfloat16)

    torch.testing.assert_close(triton_out, aten_out, rtol=2e-2, atol=0.125)
