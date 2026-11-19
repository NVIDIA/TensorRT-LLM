# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve the two BF16 rounding boundaries in a MoE residual update."""

import torch
import triton
import triton.language as tl

# The element-count-specialized launch: the generation steps' row counts (CUDA graphs).
_FUSED_MAX_ROWS = 96
# The any-row launch (a context step) indexes elements in int32: up to 2**18 rows of 7168 values,
# its last block ends below 2**31.
_ANY_ROWS_MAX = 1 << 18


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


@triton.jit
def _moe_residual_add_rows_kernel(routed, shared, residual, output, size, BLOCK: tl.constexpr):
    # The body of _moe_residual_add_kernel with a runtime ``size``: one compiled kernel serves
    # every row count of a context step, where a constexpr size would compile one per count.
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    routed_value = tl.load(routed + offsets, mask, other=0).to(tl.float32)
    shared_value = tl.load(shared + offsets, mask, other=0).to(tl.float32)
    residual_value = tl.load(residual + offsets, mask, other=0).to(tl.float32)
    moe_output = (routed_value + shared_value).to(tl.bfloat16).to(tl.float32)
    tl.store(output + offsets, residual_value + moe_output, mask)


def _fused_moe_residual_add(
    routed: torch.Tensor,
    shared: torch.Tensor,
    residual: torch.Tensor,
    *,
    any_rows: bool = False,
) -> torch.Tensor:
    """Compute residual + BF16(routed + shared), for matching [tokens, hidden] inputs.

    One launch of the element-count-specialized kernel up to 96 rows; with ``any_rows`` (a
    context step), one launch of the runtime-size kernel at every row count from 1 to
    ``_ANY_ROWS_MAX``. Otherwise the same two additions in torch.
    """
    tensors = (routed, shared, residual)
    max_rows = _ANY_ROWS_MAX if any_rows else _FUSED_MAX_ROWS
    eligible = (
        routed.ndim == 2
        and 0 < routed.shape[0] <= max_rows
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
    kernel = _moe_residual_add_rows_kernel if any_rows else _moe_residual_add_kernel
    kernel[(triton.cdiv(routed.numel(), block),)](
        routed, shared, residual, output, routed.numel(), block, num_warps=4
    )
    return output
