# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E4M3 query of the PrimTS sparse MLA (Triton).

The absorbed query of each token and head (the latent part ``q_nope @ W_UK``, 512 wide, and the RoPE part, 64 wide)
becomes one per-tensor E4M3 ``[latent 512 | rope 64]`` row, the query layout of the PrimTS kernels
(``kernels/attention/mla_decode_*``).
"""

import torch
import triton
import triton.language as tl

NOPE_DIM = 512
ROPE_DIM = 64
QUERY_DIM = NOPE_DIM + ROPE_DIM

_FP8_MAX = 448.0


@triton.jit
def _quant_query_kernel(
    lat_ptr,  # bf16 [n, H, 512] (absorbed q_nope), last dim contiguous
    pe_ptr,  # bf16 [n, H, 64] (rope'd q_pe), last dim contiguous
    out_ptr,  # fp8 [n, H, 576] contiguous
    scale_ptr,  # fp32 [1] quantization scale (latent part)
    pe_scale_ptr,  # fp32 [1] quantization scale (RoPE part)
    num_rows,  # n * H
    heads,
    lat_tok_stride,
    lat_head_stride,
    pe_tok_stride,
    pe_head_stride,
    FP8_MAX: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    m = r < num_rows
    t = (r // heads).to(tl.int64)
    h = (r % heads).to(tl.int64)
    s = tl.load(scale_ptr)
    d = tl.arange(0, 512)
    x = tl.load(
        lat_ptr + (t * lat_tok_stride + h * lat_head_stride)[:, None] + d[None, :],
        mask=m[:, None],
    )
    x = tl.minimum(tl.maximum(x.to(tl.float32) * s, -FP8_MAX), FP8_MAX)
    o = r.to(tl.int64) * 576
    tl.store(out_ptr + o[:, None] + d[None, :], x.to(tl.float8e4nv), mask=m[:, None])
    e = tl.arange(0, 64)
    y = tl.load(
        pe_ptr + (t * pe_tok_stride + h * pe_head_stride)[:, None] + e[None, :],
        mask=m[:, None],
    )
    y = tl.minimum(tl.maximum(y.to(tl.float32) * tl.load(pe_scale_ptr), -FP8_MAX), FP8_MAX)
    tl.store(out_ptr + o[:, None] + 512 + e[None, :], y.to(tl.float8e4nv), mask=m[:, None])


def quantize_query_e4m3(
    q_latent: torch.Tensor,
    q_pe: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """``out[n, H, 576] = e4m3(clamp([q_latent | q_pe] * scale, +-448))`` (saturating, round to nearest); the
    kernel's RoPE-part scale operand takes ``scale`` as well."""
    n, heads, _ = q_latent.shape
    rows = n * heads
    if rows == 0:
        return
    if q_latent.stride(-1) != 1 or q_pe.stride(-1) != 1 or not out.is_contiguous():
        raise ValueError("the query parts need a unit last stride and out must be contiguous")
    block_r = 16
    _quant_query_kernel[(triton.cdiv(rows, block_r),)](
        q_latent,
        q_pe,
        out,
        scale,
        scale,
        rows,
        heads,
        q_latent.stride(0),
        q_latent.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        FP8_MAX=_FP8_MAX,
        BLOCK_R=block_r,
    )


__all__ = [
    "NOPE_DIM",
    "QUERY_DIM",
    "ROPE_DIM",
    "quantize_query_e4m3",
]
