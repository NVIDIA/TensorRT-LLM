# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exchange MiniMax-H3 attention output as static per-tensor FP8 across four Ulysses ranks."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from tensorrt_llm._torch.modules.linear import FP8QDQLinearMethod

_ULYSSES_SIZE = 4
_LOCAL_HEADS = 14
_HEAD_DIM = 128
_LOCAL_WIDTH = _LOCAL_HEADS * _HEAD_DIM  # 1792 = 7 * 256
_FULL_WIDTH = _ULYSSES_SIZE * _LOCAL_WIDTH  # 7168
_CHUNK_COLS = 256
_NUM_CHUNKS = _LOCAL_WIDTH // _CHUNK_COLS  # 7
_QUANT_BLOCK_SIZE = 16384


@triton.jit
def _static_fp8_quant_kernel(x_ptr, scale_ptr, y_ptr, numel, BLOCK_SIZE: tl.constexpr):
    """Per-tensor static E4M3 quantization matching ``static_quantize_e4m3_per_tensor``."""
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    inv_scale = libdevice.rcp_rn(tl.load(scale_ptr).to(tl.float32))
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offsets, (x * inv_scale).to(y_ptr.dtype.element_ty), mask=mask)


def quantize_static_fp8(input_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize contiguous BF16 *input_tensor* to float8_e4m3fn using static FP32 *scale*."""
    if input_tensor.dtype != torch.bfloat16 or not input_tensor.is_contiguous():
        raise ValueError("quantize_static_fp8 expects a contiguous bfloat16 tensor")
    if scale.dtype != torch.float32 or scale.numel() != 1:
        raise ValueError("quantize_static_fp8 expects a one-element float32 scale")
    output = torch.empty_like(input_tensor, dtype=torch.float8_e4m3fn)
    numel = input_tensor.numel()
    if numel > 0:
        _static_fp8_quant_kernel[(triton.cdiv(numel, _QUANT_BLOCK_SIZE),)](
            input_tensor, scale, output, numel, BLOCK_SIZE=_QUANT_BLOCK_SIZE, num_warps=8
        )
    return output


@triton.jit
def _unpack_u4_output_kernel(
    recv_ptr,
    out_ptr,
    rows,
    BLOCK_R: tl.constexpr,
    ULYSSES_SIZE: tl.constexpr,
    LOCAL_WIDTH: tl.constexpr,
    CHUNK_COLS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    """Transpose contiguous ``[4, rows, 1792]`` into contiguous ``[rows, 4, 1792]``."""
    row = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
    u = tl.program_id(1).to(tl.int64)
    valid = row < rows
    cols = tl.arange(0, CHUNK_COLS)
    src_base = (u * rows + row) * LOCAL_WIDTH
    dst_base = (row * ULYSSES_SIZE + u) * LOCAL_WIDTH
    for chunk in tl.static_range(NUM_CHUNKS):
        col = chunk * CHUNK_COLS + cols
        vals = tl.load(recv_ptr + src_base[:, None] + col[None, :], mask=valid[:, None], other=0)
        tl.store(out_ptr + dst_base[:, None] + col[None, :], vals, mask=valid[:, None])


def static_fp8_output_scale(linear) -> torch.Tensor | None:
    """Return the validated static FP8 input scale if *linear* can consume prequantized E4M3."""
    quant_method = getattr(linear, "quant_method", None)
    if type(quant_method) is not FP8QDQLinearMethod:
        return None
    if getattr(linear, "dtype", None) != torch.bfloat16:
        return None
    if getattr(linear, "all_reduce", None) is not None:
        return None
    weight = getattr(linear, "weight", None)
    if weight is None or weight.ndim != 2 or weight.shape[1] != _FULL_WIDTH:
        return None
    scale = quant_method.get_static_input_scale(linear)
    if (
        scale is None
        or scale is not getattr(linear, "input_scale", None)
        or scale.dtype != torch.float32
        or scale.numel() != 1
        or not scale.is_cuda
    ):
        return None
    return scale


def unpack_u4_output(recv: torch.Tensor, block_r: int = 8) -> torch.Tensor:
    """Transpose ``[4, S_local, 1792]`` (uint8 or uint16/bf16) to ``[1, S_local, 7168]``."""
    if recv.ndim != 3 or recv.shape[0] != _ULYSSES_SIZE or recv.shape[2] != _LOCAL_WIDTH:
        raise ValueError(
            f"expected [{_ULYSSES_SIZE}, S_local, {_LOCAL_WIDTH}], got {tuple(recv.shape)}"
        )
    rows = recv.shape[1]
    if not recv.is_contiguous():
        recv = recv.contiguous()
    out = torch.empty((rows, _ULYSSES_SIZE, _LOCAL_WIDTH), dtype=recv.dtype, device=recv.device)
    if rows > 0:
        grid = (triton.cdiv(rows, block_r), _ULYSSES_SIZE)
        _unpack_u4_output_kernel[grid](
            recv,
            out,
            rows,
            BLOCK_R=block_r,
            ULYSSES_SIZE=_ULYSSES_SIZE,
            LOCAL_WIDTH=_LOCAL_WIDTH,
            CHUNK_COLS=_CHUNK_COLS,
            NUM_CHUNKS=_NUM_CHUNKS,
            num_warps=4,
        )
    return out.view(1, rows, _FULL_WIDTH)


def exchange_fp8_output(
    attn_out: torch.Tensor,
    input_scale: torch.Tensor,
    group,
    *,
    block_r: int = 8,
) -> torch.Tensor:
    """Quantize BF16 ``[1, 4 * S_local, 14, 128]`` with static *input_scale*, all-to-all as FP8,
    and assemble into contiguous ``[1, S_local, 7168]`` float8_e4m3fn for ``to_out``."""
    if (
        attn_out.ndim != 4
        or attn_out.shape[0] != 1
        or attn_out.shape[2] != _LOCAL_HEADS
        or attn_out.shape[3] != _HEAD_DIM
        or attn_out.shape[1] % _ULYSSES_SIZE != 0
        or attn_out.dtype != torch.bfloat16
    ):
        raise ValueError(
            f"expected BF16 [1, 4*S_local, {_LOCAL_HEADS}, {_HEAD_DIM}], "
            f"got {attn_out.dtype} {tuple(attn_out.shape)}"
        )
    rows = attn_out.shape[1] // _ULYSSES_SIZE
    flat = attn_out.contiguous().reshape(-1)
    send = quantize_static_fp8(flat, input_scale).view(torch.uint8).view(_ULYSSES_SIZE, rows, _LOCAL_WIDTH)
    recv = torch.empty_like(send)
    torch.distributed.all_to_all_single(recv.view(-1), send.view(-1), group=group)
    return unpack_u4_output(recv, block_r=block_r).view(torch.float8_e4m3fn)
