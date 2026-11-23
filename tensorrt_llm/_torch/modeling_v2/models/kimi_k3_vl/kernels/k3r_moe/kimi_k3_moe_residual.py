# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve the two BF16 rounding boundaries in a MoE residual update."""

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_residual_add_kernel(
    routed, shared, residual, output, size: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    routed_value = tl.load(routed + offsets, mask, other=0).to(tl.float32)
    shared_value = tl.load(shared + offsets, mask, other=0).to(tl.float32)
    residual_value = tl.load(residual + offsets, mask, other=0).to(tl.float32)
    moe_output = (routed_value + shared_value).to(tl.bfloat16).to(tl.float32)
    tl.store(output + offsets, residual_value + moe_output, mask)


def _fused_moe_residual_add(
    routed: torch.Tensor, shared: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """Compute residual + BF16(routed + shared), for matching [tokens, hidden] inputs."""
    tensors = (routed, shared, residual)
    eligible = (
        routed.ndim == 2
        and 0 < routed.shape[0] <= 96
        and routed.shape[1] == 7168
        and all(
            x.shape == routed.shape
            and x.device == routed.device
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            for x in tensors
        )
        and not (torch.is_grad_enabled() and any(x.requires_grad for x in tensors))
    )
    if not eligible:
        return residual + (routed + shared)
    output = torch.empty_like(routed)
    block = 1024
    _moe_residual_add_kernel[(triton.cdiv(routed.numel(), block),)](
        routed, shared, residual, output, routed.numel(), block, num_warps=4
    )
    return output
