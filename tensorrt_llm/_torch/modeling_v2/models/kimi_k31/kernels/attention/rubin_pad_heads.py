# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Head padding of the TRTLLM-Gen sparse MLA calls at 128 heads (Triton).

TRTLLM-Gen's sparse MLA tiles a head group of more than 64 heads in Q64 tiles, which must divide the group, so K3.1's
96 heads per rank (attention DP, Helix CP) run padded to 128 heads with zero queries. ``_pad_heads`` fills the padded
buffers: it copies the real heads and zeroes the padding heads.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _pad_heads_kernel(
    src_ptr,  # [n, heads, D] head rows to copy (unused when HAS_SRC is False)
    dst_ptr,  # [n, PAD, D] contiguous
    n_rows,  # n * PAD
    heads,
    pad,
    src_tok_stride,
    src_head_stride,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_SRC: tl.constexpr,
):
    """With HAS_SRC: row r of the n * pad rows is dst[t, h] = src[t, h] (h < heads) or 0. Without: row r of the
    n * (pad - heads) padding rows is dst[t, heads + j] = 0."""
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    m = r < n_rows
    d = tl.arange(0, D)
    if HAS_SRC:
        t = (r // pad).to(tl.int64)
        h = (r % pad).to(tl.int64)
        x = tl.load(
            src_ptr + (t * src_tok_stride + h * src_head_stride)[:, None] + d[None, :],
            mask=(m & (h < heads))[:, None],
            other=0,
        )
        tl.store(dst_ptr + (t * pad + h)[:, None] * D + d[None, :], x, mask=m[:, None])
    else:
        t = (r // (pad - heads)).to(tl.int64)
        h = (heads + r % (pad - heads)).to(tl.int64)
        zeros = tl.zeros((BLOCK_R, D), dtype=dst_ptr.dtype.element_ty)
        tl.store(dst_ptr + (t * pad + h)[:, None] * D + d[None, :], zeros, mask=m[:, None])


def _pad_heads(dst: torch.Tensor, src: Optional[torch.Tensor], heads: int) -> None:
    """Fill a contiguous ``[n, PAD, D]`` buffer: heads ``< heads`` from ``src`` (if given, last dim contiguous),
    the padding heads with zeros. ``D`` in bytes of a uint8 view must be a power of two."""
    n, pad, d = dst.shape
    if n == 0:
        return
    if src is not None:
        assert src.dtype == dst.dtype and src.stride(-1) == 1
        src_strides = (src.stride(0), src.stride(1))
    else:
        src_strides = (0, 0)
    assert d & (d - 1) == 0, "head row length must be a power of two"
    block_r = 64
    rows = n * pad if src is not None else n * (pad - heads)
    if rows == 0:
        return
    _pad_heads_kernel[(triton.cdiv(rows, block_r),)](
        src if src is not None else dst,
        dst,
        rows,
        heads,
        pad,
        *src_strides,
        D=d,
        BLOCK_R=block_r,
        HAS_SRC=src is not None,
    )
