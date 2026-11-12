# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton BF16 -> FP8 E4M3 per-tensor static quantization for the K3 FP8 context FMHA inputs.

:func:`static_quantize_e4m3` writes the bytes of
``torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor``: every element becomes
``cvt.rn.satfinite.e4m3(float(x) * (1 / scale))`` with an IEEE FP32 reciprocal, and each program
converts a large contiguous block of BF16 pairs. :func:`static_quantize_e4m3_rows` writes the same
bytes for a row-strided view (e.g. the RoPE columns of ``latent_cache``) without a contiguous copy.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3x2_from_bf16x2(pairs, inv_scale):
    """Convert int32-packed BF16 pairs to int16-packed E4M3 pairs, first element in the low byte."""
    lo = (pairs << 16).to(tl.float32, bitcast=True) * inv_scale
    hi = (pairs & -65536).to(tl.float32, bitcast=True) * inv_scale
    return tl.inline_asm_elementwise(
        "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;",
        "=h,f,f",
        [lo, hi],
        dtype=tl.int16,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _static_quant_e4m3_kernel(x_ptr, y_ptr, scale_ptr, num_pairs, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_pairs
    inv_scale = tl.math.div_rn(1.0, tl.load(scale_ptr).to(tl.float32))
    pairs = tl.load(x_ptr + offsets, mask=mask, other=0)
    tl.store(y_ptr + offsets, _e4m3x2_from_bf16x2(pairs, inv_scale), mask=mask)


@triton.jit
def _static_quant_e4m3_rows_kernel(
    x_ptr,
    y_ptr,
    scale_ptr,
    num_rows,
    x_row_stride,
    ROW_PAIRS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, ROW_PAIRS)
    mask = (rows < num_rows)[:, None]
    inv_scale = tl.math.div_rn(1.0, tl.load(scale_ptr).to(tl.float32))
    pairs = tl.load(x_ptr + rows[:, None] * x_row_stride + cols[None, :], mask=mask, other=0)
    tl.store(
        y_ptr + rows[:, None] * ROW_PAIRS + cols[None, :],
        _e4m3x2_from_bf16x2(pairs, inv_scale),
        mask=mask,
    )


def _check_scale(scale: torch.Tensor, device: torch.device) -> None:
    if scale.dtype != torch.float32 or scale.numel() != 1 or scale.device != device:
        raise ValueError(
            "FP8 static quantization requires a one-element FP32 scale on the input device."
        )


def static_quantize_e4m3(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize a contiguous BF16 tensor with an even element count to FP8 E4M3 (same shape)."""
    if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.numel() % 2 != 0:
        raise ValueError("FP8 static quantization requires a contiguous BF16 tensor of even size.")
    _check_scale(scale, x.device)
    y = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    num_pairs = x.numel() // 2
    if num_pairs == 0:
        return y
    block = 8192
    _static_quant_e4m3_kernel[(triton.cdiv(num_pairs, block),)](
        x.view(-1).view(torch.int32),
        y.view(-1).view(torch.int16),
        scale,
        num_pairs,
        BLOCK=block,
        num_warps=8,
    )
    return y


def supports_rows(x: torch.Tensor) -> bool:
    """Whether :func:`static_quantize_e4m3_rows` accepts ``x``: a 2-D BF16 view with unit column
    stride, even row stride and storage offset, and a power-of-two width of at least 2."""
    if x.dtype != torch.bfloat16 or x.dim() != 2:
        return False
    width = x.shape[1]
    return (
        x.stride(1) == 1
        and x.stride(0) % 2 == 0
        and x.storage_offset() % 2 == 0
        and width >= 2
        and width & (width - 1) == 0
    )


def static_quantize_e4m3_rows(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize a row-strided BF16 [rows, width] view (see :func:`supports_rows`) into a
    contiguous FP8 E4M3 [rows, width] tensor."""
    if not supports_rows(x):
        raise ValueError(
            "FP8 row quantization requires unit column stride, even row stride and offset, "
            f"and a power-of-two width; got shape {tuple(x.shape)}, stride {x.stride()}."
        )
    _check_scale(scale, x.device)
    rows, width = x.shape
    y = torch.empty((rows, width), dtype=torch.float8_e4m3fn, device=x.device)
    if rows == 0:
        return y
    pairs = x.view(torch.int32)
    block_rows = 64
    _static_quant_e4m3_rows_kernel[(triton.cdiv(rows, block_rows),)](
        pairs,
        y.view(torch.int16),
        scale,
        rows,
        pairs.stride(0),
        ROW_PAIRS=width // 2,
        BLOCK_ROWS=block_rows,
        num_warps=4,
    )
    return y
