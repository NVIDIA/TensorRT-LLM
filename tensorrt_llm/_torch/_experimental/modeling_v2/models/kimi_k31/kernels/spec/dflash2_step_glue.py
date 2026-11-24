# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels for the DFlash2 drafter step's small glue around the block decode, one launch each.

* :func:`noise_block`: the drafter's input block, every request's bonus-token embedding row followed
  by ``block - 1`` copies of the mask row.
* :func:`agreed_tokens`: the TP group's agreed draft tokens read back from the all-reduced FP32
  agreement row, and the next step's input tokens with those drafts.

Both are copies (and an FP32 -> integer conversion of exact integers): the values of the torch ops
they stand for, bit for bit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: Columns one program of :func:`noise_block` copies: 8 BF16 values (16 bytes) a thread at 4 warps.
_NOISE_BLOCK_H = 1024


@triton.jit
def _noise_block_kernel(
    table_ptr,
    stride_table,
    bonus_ptr,
    mask_ptr,
    out_ptr,
    hidden,
    BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Program (row, column block): row = request * BLOCK + slot.
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    cmask = cols < hidden
    slot = row % BLOCK
    if slot == 0:
        token = tl.load(bonus_ptr + row // BLOCK).to(tl.int64)
        values = tl.load(table_ptr + token * stride_table + cols, mask=cmask)
    else:
        values = tl.load(mask_ptr + cols, mask=cmask)
    tl.store(out_ptr + row.to(tl.int64) * hidden + cols, values, mask=cmask)


def noise_block(
    table: torch.Tensor, bonus: torch.Tensor, mask_row: torch.Tensor, block: int
) -> torch.Tensor:
    """``[len(bonus), block, hidden]`` in the table's dtype: slot 0 of request ``i`` is
    ``table[bonus[i]]``, every other slot ``mask_row``. ``table`` ``[vocab, hidden]`` with unit
    column stride, ``bonus`` integer ``[n]``, ``mask_row`` ``[hidden]`` contiguous."""
    vocab, hidden = table.shape
    if table.stride(-1) != 1 or not mask_row.is_contiguous() or mask_row.shape != (hidden,):
        raise ValueError("noise_block: table rows and the mask row must be contiguous [hidden]")
    if mask_row.dtype != table.dtype or bonus.dim() != 1 or block < 1:
        raise ValueError(
            "noise_block: the mask row in the table's dtype, 1-D bonus ids, block >= 1"
        )
    n = bonus.shape[0]
    out = torch.empty((n, block, hidden), dtype=table.dtype, device=table.device)
    if n == 0:
        return out
    grid = (n * block, triton.cdiv(hidden, _NOISE_BLOCK_H))
    _noise_block_kernel[grid](
        table,
        table.stride(0),
        bonus,
        mask_row,
        out,
        hidden,
        BLOCK=block,
        BLOCK_H=_NOISE_BLOCK_H,
        num_warps=4,
    )
    return out


@triton.jit
def _agreed_tokens_kernel(
    reduced_ptr,
    stride_reduced,
    new_ptr,
    stride_new,
    drafts_ptr,
    out_new_ptr,
    rows,
    DRAFT_LEN: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program: [rows, DRAFT_LEN] drafts from the FP32 row, then [rows, 1 + DRAFT_LEN] tokens.
    r = tl.arange(0, BLOCK_R)
    k = tl.arange(0, BLOCK_K)
    rmask = r < rows
    dmask = rmask[:, None] & (k[None, :] < DRAFT_LEN)
    agreed = tl.load(
        reduced_ptr + r[:, None].to(tl.int64) * stride_reduced + k[None, :], mask=dmask
    )
    tl.store(
        drafts_ptr + r[:, None] * DRAFT_LEN + k[None, :],
        agreed.to(drafts_ptr.dtype.element_ty),
        mask=dmask,
    )
    width = DRAFT_LEN + 1
    first = tl.load(new_ptr + r.to(tl.int64) * stride_new, mask=rmask)
    tl.store(out_new_ptr + r * width, first, mask=rmask)
    tl.store(
        out_new_ptr + r[:, None] * width + 1 + k[None, :],
        agreed.to(drafts_ptr.dtype.element_ty).to(out_new_ptr.dtype.element_ty),
        mask=dmask,
    )


def agreed_tokens(
    reduced: torch.Tensor,
    draft_len: int,
    draft_dtype: torch.dtype,
    new_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(drafts, next_new)``: ``drafts = reduced[:, :draft_len].to(draft_dtype)`` ``[rows,
    draft_len]`` and ``next_new = cat([new_tokens[:, :1], drafts.to(new_tokens.dtype)], 1)``
    ``[rows, 1 + draft_len]``, both new tensors. ``reduced`` FP32 ``[rows, >= draft_len]`` with
    unit column stride; ``new_tokens`` integer ``[rows, >= 1]``."""
    rows = reduced.shape[0]
    if reduced.dtype != torch.float32 or reduced.stride(-1) != 1 or reduced.shape[1] < draft_len:
        raise ValueError("agreed_tokens: FP32 [rows, >= draft_len] rows with unit column stride")
    if new_tokens.shape[0] != rows or new_tokens.dim() != 2 or new_tokens.stride(-1) != 1:
        raise ValueError("agreed_tokens: new_tokens [rows, >= 1] with unit column stride")
    device = reduced.device
    drafts = torch.empty((rows, draft_len), dtype=draft_dtype, device=device)
    out_new = torch.empty((rows, draft_len + 1), dtype=new_tokens.dtype, device=device)
    if rows == 0:
        return drafts, out_new
    _agreed_tokens_kernel[(1,)](
        reduced,
        reduced.stride(0),
        new_tokens,
        new_tokens.stride(0),
        drafts,
        out_new,
        rows,
        DRAFT_LEN=draft_len,
        BLOCK_R=triton.next_power_of_2(rows),
        BLOCK_K=triton.next_power_of_2(max(draft_len, 1)),
        num_warps=1,
    )
    return drafts, out_new


__all__ = ["agreed_tokens", "noise_block"]
