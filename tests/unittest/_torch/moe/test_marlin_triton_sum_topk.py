# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical coverage for the Marlin NVFP4 Triton MoE combine path."""

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.fused_moe_marlin import sum_topk_expert_outputs

_MARLIN_SM_SKIP = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="Marlin NVFP4 requires SM89+ (Ada/Hopper)",
)


def _aten_reference(
    expert_outputs: torch.Tensor, num_tokens: int, top_k: int, hidden_size: int, dtype: torch.dtype
) -> torch.Tensor:
    """ATen scatter-reduce reference — mirrors the non-contiguous fallback in forward()."""
    result = torch.zeros((num_tokens, hidden_size), dtype=dtype, device=expert_outputs.device)
    if num_tokens == 0:
        return result
    token_indices = torch.arange(num_tokens * top_k, device=expert_outputs.device) // top_k
    result.index_add_(0, token_indices, expert_outputs)
    return result


def _fp32_reference(
    expert_outputs: torch.Tensor, num_tokens: int, top_k: int, hidden_size: int, dtype: torch.dtype
) -> torch.Tensor:
    """FP32 reduce-then-cast reference — enforces that the kernel accumulates in FP32."""
    if num_tokens == 0:
        return torch.empty((0, hidden_size), dtype=dtype, device=expert_outputs.device)
    return expert_outputs.float().reshape(num_tokens, top_k, hidden_size).sum(dim=1).to(dtype)


@_MARLIN_SM_SKIP
@pytest.mark.parametrize(
    "num_tokens,top_k,hidden_size",
    [
        (0, 2, 256),  # zero tokens — kernel must not launch, return empty tensor
        (1, 1, 256),
        (3, 2, 257),  # hidden_size not a multiple of BLOCK_H=256
        (4, 6, 2688),  # Nemotron Nano 30B hidden size
        (4, 8, 2688),
        (16, 2, 4096),
    ],
)
def test_triton_sum_topk_matches_aten_scatter_reduce(
    num_tokens: int, top_k: int, hidden_size: int
) -> None:
    """Triton kernel accumulates in FP32 and matches both a FP32 and a BF16 ATen reference.

    Two assertions:
    1. FP32 contract: actual must match the FP32-reduce-then-cast reference within
       a narrow BF16-level tolerance (rtol=1e-3, atol=0.01). A regression that
       accumulates in BF16 would diverge from this reference and fail.
    2. Fallback parity: actual must also match the BF16 ATen index_add_ reference
       within the wider BF16 rounding gap (rtol=2e-2, atol=0.125).
    """
    torch.manual_seed(0)
    expert_outputs = torch.randn(
        (num_tokens * top_k, hidden_size), dtype=torch.bfloat16, device="cuda"
    )

    actual = sum_topk_expert_outputs(expert_outputs, num_tokens, top_k, hidden_size, torch.bfloat16)

    assert actual.shape == (num_tokens, hidden_size)
    assert actual.dtype == torch.bfloat16
    if num_tokens > 0:
        fp32_ref = _fp32_reference(expert_outputs, num_tokens, top_k, hidden_size, torch.bfloat16)
        torch.testing.assert_close(actual, fp32_ref, rtol=1e-3, atol=0.01)

        aten_ref = _aten_reference(expert_outputs, num_tokens, top_k, hidden_size, torch.bfloat16)
        torch.testing.assert_close(actual, aten_ref, rtol=2e-2, atol=0.125)


@_MARLIN_SM_SKIP
def test_noncontiguous_fallback_matches_triton() -> None:
    """Non-contiguous gemm2_out takes the ATen index_add_ path in forward().

    This test exercises the fallback branch directly (not through forward())
    to confirm both paths agree within BF16 tolerance — ensuring the fallback
    remains correct if the Triton path is ever skipped.
    """
    num_tokens, top_k, hidden_size = 4, 2, 2688
    torch.manual_seed(42)
    base = torch.randn((num_tokens * top_k, hidden_size), dtype=torch.bfloat16, device="cuda")

    # Build a non-contiguous view via padded backing storage: stride along dim-0
    # is (hidden_size + 1) instead of hidden_size, so is_contiguous() returns False.
    padded = torch.empty((num_tokens * top_k, hidden_size + 1), dtype=torch.bfloat16, device="cuda")
    padded[:, :hidden_size].copy_(base)
    non_contig = padded[:, :hidden_size]
    assert not non_contig.is_contiguous(), "test setup error: tensor should be non-contiguous"

    aten_out = _aten_reference(non_contig, num_tokens, top_k, hidden_size, torch.bfloat16)
    triton_out = sum_topk_expert_outputs(base, num_tokens, top_k, hidden_size, torch.bfloat16)

    torch.testing.assert_close(triton_out, aten_out, rtol=2e-2, atol=0.125)
