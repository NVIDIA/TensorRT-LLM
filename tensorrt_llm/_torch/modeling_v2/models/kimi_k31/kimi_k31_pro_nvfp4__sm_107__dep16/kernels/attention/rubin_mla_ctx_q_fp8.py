# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FP8 latent columns of the sparse MLA prefill query at 128 heads (Triton).

The FP8 context FMHA reads its query as E4M3 ``[tokens, heads, kv_lora + rope]``. Given a BF16 query, the attention
op quantizes all of it first (``quantizeCopyInputToFp8Kernel``, padding heads included). ``_ctx_q_fp8`` writes the
latent columns of that FP8 query straight from the absorbed ``q_nope @ W_UK`` rows: the real heads converted the way
the op converts them (unit scale, round to nearest even, saturated to +-448, NaN kept), the padding heads zero. The
op's RoPE kernel then writes the RoPE columns in FP8 itself, and the op skips its own quantization.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

E4M3_MAX = 448.0


@triton.jit
def _ctx_q_fp8_kernel(
    src_ptr,  # bf16 [n, >= heads, D] latent rows (last dim contiguous)
    dst_ptr,  # e4m3 [n, pad, D] latent columns of the FP8 query (last dim contiguous)
    n_rows,  # n * pad
    heads,
    pad,
    src_tok_stride,
    src_head_stride,
    dst_tok_stride,
    dst_head_stride,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Row r of the n * pad (token, head) rows: dst[t, h] = e4m3(src[t, h]) for h < heads, else zeros."""
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    m = r < n_rows
    t = (r // pad).to(tl.int64)
    h = (r % pad).to(tl.int64)
    d = tl.arange(0, D)
    x = tl.load(
        src_ptr + (t * src_tok_stride + h * src_head_stride)[:, None] + d[None, :],
        mask=(m & (h < heads))[:, None],
        other=0.0,
    )
    # The attention op's conversion of its BF16 query: unit scale, then the saturating E4M3 cast. The
    # clamp makes the saturation explicit (an infinity becomes +-448, a NaN stays a NaN).
    y = tl.clamp(x.to(tl.float32), -448.0, 448.0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(
        dst_ptr + (t * dst_tok_stride + h * dst_head_stride)[:, None] + d[None, :],
        y.to(dst_ptr.dtype.element_ty),
        mask=m[:, None],
    )


def _ctx_q_fp8(dst: torch.Tensor, src: torch.Tensor, heads: int) -> None:
    """Fill ``dst`` ``[n, pad, D]`` (E4M3): heads ``< heads`` from ``src`` ``[n, >= heads, D]`` (BF16), the padding
    heads with zeros. Both last dims contiguous; ``D`` a power of two."""
    n, pad, d = dst.shape
    if n == 0:
        return
    assert d & (d - 1) == 0, "the latent width must be a power of two"
    block_r = 16
    rows = n * pad
    _ctx_q_fp8_kernel[(triton.cdiv(rows, block_r),)](
        src,
        dst,
        rows,
        heads,
        pad,
        src.stride(0),
        src.stride(1),
        dst.stride(0),
        dst.stride(1),
        D=d,
        BLOCK_R=block_r,
        num_warps=4,
    )
