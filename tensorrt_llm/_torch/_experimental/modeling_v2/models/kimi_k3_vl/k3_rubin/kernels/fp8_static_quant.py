# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton BF16 -> FP8 E4M3 per-tensor static quantization for the K3 FP8 context FMHA inputs.

:func:`static_quantize_e4m3` writes the bytes of
``torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor``: every element becomes
``cvt.rn.satfinite.e4m3(float(x) * (1 / scale))`` with an IEEE FP32 reciprocal, and each program
converts a large contiguous block of BF16 pairs. :func:`static_quantize_e4m3_rows` writes the same
bytes for a row-strided view (e.g. the RoPE columns of ``latent_cache``) without a contiguous copy.

:func:`mla_kv_to_fp8` builds the FP8 K and V of non-absorption MLA context attention straight
from the kv_b_proj output and k_pe, with the bytes the attention op's FP8 context copy
(``quantizeCopyInputToFp8Kernel``) writes for the concatenated BF16 K and V.
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


@triton.jit(do_not_specialize=["num_tokens"])
def _mla_kv_fp8_kernel(
    kv_ptr,
    pe_ptr,
    k_ptr,
    v_ptr,
    multiplier,
    num_tokens,
    kv_row_stride,
    pe_row_stride,
    NUM_HEADS: tl.constexpr,
    NOPE_PAIRS: tl.constexpr,
    ROPE_PAIRS: tl.constexpr,
    V_PAIRS: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    mask = (rows < num_tokens)[:, None]
    head0 = tl.program_id(1) * HEADS_PER_PROGRAM
    nope = tl.arange(0, NOPE_PAIRS)[None, :]
    rope = tl.arange(0, ROPE_PAIRS)[None, :]
    v_cols = tl.arange(0, V_PAIRS)[None, :]
    kv_rows = kv_ptr + rows[:, None] * kv_row_stride
    k_rows = k_ptr + rows[:, None] * (NUM_HEADS * (NOPE_PAIRS + ROPE_PAIRS))
    v_rows = v_ptr + rows[:, None] * (NUM_HEADS * V_PAIRS)
    pe = tl.load(pe_ptr + rows[:, None] * pe_row_stride + rope, mask=mask, other=0)
    pe = _e4m3x2_from_bf16x2(pe, multiplier)
    for i in tl.static_range(HEADS_PER_PROGRAM):
        head = head0 + i
        k_head = k_rows + head * (NOPE_PAIRS + ROPE_PAIRS)
        x = tl.load(kv_rows + head * NOPE_PAIRS + nope, mask=mask, other=0)
        tl.store(k_head + nope, _e4m3x2_from_bf16x2(x, multiplier), mask=mask)
        tl.store(k_head + NOPE_PAIRS + rope, pe, mask=mask)
        x = tl.load(kv_rows + NUM_HEADS * NOPE_PAIRS + head * V_PAIRS + v_cols, mask=mask, other=0)
        tl.store(v_rows + head * V_PAIRS + v_cols, _e4m3x2_from_bf16x2(x, multiplier), mask=mask)


def _pairs_view(x: torch.Tensor, name: str) -> torch.Tensor:
    """int32 view of a 2-D BF16 tensor with unit column stride and even row stride / offset."""
    if (
        x.dtype != torch.bfloat16
        or x.dim() != 2
        or x.stride(1) != 1
        or x.stride(0) % 2
        or x.storage_offset() % 2
        or x.shape[1] % 2
    ):
        raise ValueError(
            f"{name} must be a 2-D BF16 view with unit column stride and even width, row stride "
            f"and offset; got {x.dtype}, shape {tuple(x.shape)}, stride {x.stride()}."
        )
    return x.view(torch.int32)


def mla_kv_to_fp8(
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    *,
    block_tokens: int = 32,
    heads_per_program: int = 4,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 E4M3 K and V for the FP8 context FMHA of non-absorption MLA, in one pass.

    ``kv`` is the BF16 kv_b_proj output ``[T, H * (nope + v)]`` (the nope columns of all heads,
    then their V columns) and ``k_pe`` the BF16 RoPE key ``[T, rope]`` (unit column stride, any
    even row stride, e.g. a ``latent_cache`` view). Returns contiguous K ``[T, H * (nope + rope)]``
    (each head's nope columns followed by k_pe) and V ``[T, H * v]``. Every element becomes
    ``cvt.rn.satfinite.e4m3(float(x) * 1)``: the bytes the attention op's context copy writes
    for the concatenated BF16 K and V (its dense MLA path quantizes with a unit scale).
    """
    num_tokens, rope_head_dim = k_pe.shape
    nope_cols, v_cols = num_heads * qk_nope_head_dim, num_heads * v_head_dim
    if kv.dim() != 2 or tuple(kv.shape) != (num_tokens, nope_cols + v_cols):
        raise ValueError(f"kv must be [{num_tokens}, {nope_cols + v_cols}], got {tuple(kv.shape)}.")
    for name, dim in (
        ("qk_nope_head_dim", qk_nope_head_dim),
        ("qk_rope_head_dim", rope_head_dim),
        ("v_head_dim", v_head_dim),
    ):
        if dim < 2 or dim & (dim - 1):
            raise ValueError(f"{name} must be a power of two >= 2, got {dim}.")
    if num_heads % heads_per_program:
        raise ValueError(f"heads_per_program {heads_per_program} must divide {num_heads} heads.")
    kv_pairs = _pairs_view(kv, "kv")
    pe_pairs = _pairs_view(k_pe, "k_pe")
    k = torch.empty(
        (num_tokens, num_heads * (qk_nope_head_dim + rope_head_dim)),
        dtype=torch.float8_e4m3fn,
        device=kv.device,
    )
    v = torch.empty((num_tokens, v_cols), dtype=torch.float8_e4m3fn, device=kv.device)
    if num_tokens == 0:
        return k, v
    grid = (triton.cdiv(num_tokens, block_tokens), num_heads // heads_per_program)
    _mla_kv_fp8_kernel[grid](
        kv_pairs,
        pe_pairs,
        k.view(torch.int16),
        v.view(torch.int16),
        1.0,
        num_tokens,
        kv_pairs.stride(0),
        pe_pairs.stride(0),
        NUM_HEADS=num_heads,
        NOPE_PAIRS=qk_nope_head_dim // 2,
        ROPE_PAIRS=rope_head_dim // 2,
        V_PAIRS=v_head_dim // 2,
        HEADS_PER_PROGRAM=heads_per_program,
        BLOCK_TOKENS=block_tokens,
        num_warps=num_warps,
    )
    return k, v
