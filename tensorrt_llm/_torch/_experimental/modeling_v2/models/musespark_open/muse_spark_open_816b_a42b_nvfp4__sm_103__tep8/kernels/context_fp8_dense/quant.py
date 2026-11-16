# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 activation quantization for the Rubin MXFP8 GEMM, one Triton pass over the BF16 rows.

``mxfp8_quantize(x)`` returns what ``trtllm::mxfp8_quantize(x, True)`` returns for a 2-D BF16
``x`` whose K is a multiple of 128: E4M3 values ``[M, K]`` and one UE8M0 scale per 1x32
block in the 128x4-swizzled layout (``pad_up(M, 128) * K / 32`` bytes,
``get_sf_out_offset_128x4`` in ``cpp/tensorrt_llm/kernels/quantization.cuh``), the scale
rows past M zero. Same rounding: scale = amax / 448 rounded up to a power of two, values
``x / scale`` rounded to nearest E4M3. Each program reads a 32 x 256 tile once, so the pass
runs at the bandwidth of one BF16 read and one E4M3 write.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

#: Rows and columns of one program's tile; K must be a multiple of BLOCK_K.
BLOCK_M = 32
BLOCK_K = 256


@triton.jit
def _mxfp8_quantize_kernel(
    x_ptr,
    q_ptr,
    sf_ptr,
    M,
    stride_xm,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SWIZZLED: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    in_rows = rows < M
    x = tl.load(
        x_ptr + rows[:, None].to(tl.int64) * stride_xm + cols[None, :],
        mask=in_rows[:, None],
        other=0.0,
    ).to(tl.float32)
    GROUPS: tl.constexpr = BLOCK_K // 32
    xg = tl.reshape(x, (BLOCK_M, GROUPS, 32))
    amax = tl.max(tl.abs(xg), axis=2)
    bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    # UE8M0 rounded towards +inf: the exponent, plus one when any mantissa bit is set.
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    exponent = tl.minimum(exponent, 254)
    # 2^(127 - exponent), the reciprocal of the scale; 0 for an all-zero block.
    inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    inverse = tl.where(amax > 0.0, inverse, 0.0)
    q = tl.reshape((xg * inverse[:, :, None]).to(tl.float8e4nv), (BLOCK_M, BLOCK_K))
    tl.store(q_ptr + rows[:, None].to(tl.int64) * K + cols[None, :], q, mask=in_rows[:, None])
    blocks = pid_k * GROUPS + tl.arange(0, GROUPS)
    if SWIZZLED:
        K_TILES: tl.constexpr = K // 128
        offsets = (
            (rows // 128).to(tl.int64)[:, None] * (K_TILES * 512)
            + (blocks // 4)[None, :] * 512
            + (rows % 32)[:, None] * 16
            + ((rows % 128) // 32)[:, None] * 4
            + (blocks % 4)[None, :]
        )
        tl.store(sf_ptr + offsets, exponent.to(tl.uint8))
    else:
        offsets = rows.to(tl.int64)[:, None] * (K // 32) + blocks[None, :]
        tl.store(sf_ptr + offsets, exponent.to(tl.uint8), mask=in_rows[:, None])


def mxfp8_quantize(x: torch.Tensor, swizzled: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(E4M3 [M, K], UE8M0 scales)`` of a 2-D BF16 ``x`` (last dim contiguous).

    The scales are 128x4-swizzled (``pad_up(M, 128) * K / 32`` bytes, what the GEMM reads), or
    with ``swizzled=False`` row-major ``[M, K / 32]``, which row blocks of several calls can be
    concatenated in (``swizzle_scales`` then lays them out for the GEMM).
    """
    if x.dim() != 2 or x.dtype != torch.bfloat16 or x.stride(1) != 1:
        raise ValueError(
            f"needs a 2-D BF16 tensor with contiguous rows, got {x.dtype} {tuple(x.shape)}"
        )
    m, k = x.shape
    if k % BLOCK_K:
        raise ValueError(f"K={k} must be a multiple of {BLOCK_K}")
    padded_m = -(-m // 128) * 128
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    if swizzled:
        sf = torch.empty((padded_m * (k // 32),), dtype=torch.uint8, device=x.device)
        grid = (padded_m // BLOCK_M, k // BLOCK_K)
    else:
        sf = torch.empty((m, k // 32), dtype=torch.uint8, device=x.device)
        grid = (triton.cdiv(m, BLOCK_M), k // BLOCK_K)
    _mxfp8_quantize_kernel[grid](
        x,
        q,
        sf,
        m,
        x.stride(0),
        K=k,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        SWIZZLED=swizzled,
        num_warps=4,
    )
    return q, sf


def swizzle_scales(scales: torch.Tensor) -> torch.Tensor:
    """Row-major ``[M, K / 32]`` UE8M0 scales in the GEMM's 128x4-swizzled layout (rows past M zero)."""
    return torch.ops.trtllm.block_scale_interleave(scales.contiguous())
