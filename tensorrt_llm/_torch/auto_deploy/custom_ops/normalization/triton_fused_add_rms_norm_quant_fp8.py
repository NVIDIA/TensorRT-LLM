# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton RMSNorm + FP8 quantization custom ops.

This module provides two related kernels:
1. RMSNorm + FP8 quantization:
   - norm_out = rms_norm(input, weight, eps)
   - fp8_out = quantize_fp8(norm_out, scale)
2. Fused Add + RMSNorm + FP8 quantization:
   - add_out = x + residual
   - norm_out = rms_norm(add_out, weight, eps)
   - fp8_out = quantize_fp8(norm_out, scale)

Both variants produce BF16/FP16 outputs (for non-quantized consumers) and FP8
outputs (for GEMM), while avoiding extra DRAM round-trips.
Merging both cases to a single kernel is inefficient for the no-bias case.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

# FP8 E4M3 value range
_FP8_MIN = float(torch.finfo(torch.float8_e4m3fn).min)  # -448.0
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0


@triton.jit
def rms_norm_quant_fp8_kernel(
    input_ptr,
    weight_ptr,
    output_bf16_ptr,
    output_fp8_ptr,
    scale_ptr,
    row_stride: tl.constexpr,
    eps: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused RMSNorm + FP8 per-tensor quantization kernel.

    Each program instance processes one row of the input matrix.
    Produces two outputs per row:
      1. BF16/FP16 normalized result (same dtype as input)
      2. FP8 E4M3 quantized result  (normalized, scaled, clamped, cast)
    """
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N_COLS

    # Load RMSNorm weight
    w = tl.load(weight_ptr + offsets, mask=mask)

    # Load input row
    x_ptr = input_ptr + prog_id * row_stride
    x = tl.load(x_ptr + offsets, mask=mask)
    xf = x.to(tl.float32)

    # RMSNorm: variance -> normalize -> scale by weight (all in FP32)
    var = tl.sum(xf * xf, 0) * float(1.0 / N_COLS)
    normed = xf / tl.sqrt(var + eps)
    out_f32 = w.to(tl.float32) * normed

    # Store BF16/FP16 output (cast from FP32 for consumers needing native dtype)
    out_bf16_row = output_bf16_ptr + prog_id * row_stride
    tl.store(out_bf16_row + offsets, out_f32.to(x.dtype), mask=mask)

    # FP8 quantization from full FP32 norm (avoids BF16 round-trip precision loss)
    scale = tl.load(scale_ptr)
    out_scaled = out_f32 / scale
    out_clamped = tl.maximum(tl.minimum(out_scaled, FP8_MAX), FP8_MIN)
    out_fp8 = out_clamped.to(tl.float8e4nv)

    # Store FP8 output
    out_fp8_row = output_fp8_ptr + prog_id * N_COLS
    tl.store(out_fp8_row + offsets, out_fp8, mask=mask)


def rms_norm_quant_fp8(
    hidden_states: Tensor, weight: Tensor, eps: float, scale: Tensor
) -> Tuple[Tensor, Tensor]:
    """Fused RMSNorm + FP8 quantization.

    Args:
        hidden_states: Input tensor of shape [..., hidden_size].
        weight: RMSNorm weight of shape [hidden_size].
        eps: Epsilon for numerical stability.
        scale: Per-tensor FP8 quantization scale (scalar tensor).

    Returns:
        Tuple of (bf16_output, fp8_output), both with shape [..., hidden_size].
    """
    orig_shape = hidden_states.shape
    feat_size = weight.shape[0]
    hidden_states_flat = hidden_states.reshape(-1, feat_size)
    seq_len = hidden_states_flat.shape[0]
    input_stride = hidden_states_flat.stride(-2)

    BLOCK_N = triton.next_power_of_2(feat_size)
    out_bf16 = torch.empty_like(hidden_states_flat)
    out_fp8 = torch.empty(
        hidden_states_flat.shape,
        dtype=torch.float8_e4m3fn,
        device=hidden_states.device,
    )

    grid = (seq_len,)
    rms_norm_quant_fp8_kernel[grid](
        hidden_states_flat,
        weight,
        out_bf16,
        out_fp8,
        scale,
        row_stride=input_stride,
        eps=eps,
        FP8_MIN=_FP8_MIN,
        FP8_MAX=_FP8_MAX,
        N_COLS=feat_size,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )

    return out_bf16.reshape(orig_shape), out_fp8.reshape(orig_shape)


@torch.library.custom_op("auto_deploy::triton_rms_norm_quant_fp8", mutates_args=())
def triton_rms_norm_quant_fp8(
    input: torch.Tensor, weight: torch.Tensor, eps: float, scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused RMSNorm + FP8 quantization custom op.

    Computes RMSNorm and FP8 per-tensor quantization in a single Triton kernel,
    producing both a BF16/FP16 output (for residual connections or other consumers)
    and an FP8 output (for subsequent GEMM operations).

    Args:
        input: Input tensor to normalize, shape [..., hidden_size].
        weight: RMSNorm scaling weights, shape [hidden_size].
        eps: Small constant for numerical stability.
        scale: Per-tensor FP8 quantization scale (scalar tensor, float32).

    Returns:
        Tuple of (bf16_output, fp8_output).
    """
    return rms_norm_quant_fp8(input, weight, eps, scale)


@triton_rms_norm_quant_fp8.register_fake
def _rms_norm_quant_fp8_fake(
    input: torch.Tensor, weight: torch.Tensor, eps: float, scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for tracing / shape inference."""
    bf16_out = torch.empty_like(input)
    fp8_out = torch.empty(input.shape, dtype=torch.float8_e4m3fn, device=input.device)
    return bf16_out, fp8_out


@triton.jit
def fused_add_rms_norm_quant_fp8_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_bf16_ptr,
    output_fp8_ptr,
    output_add_ptr,
    scale_ptr,
    row_stride_x: tl.constexpr,
    row_stride_residual: tl.constexpr,
    row_stride_out: tl.constexpr,
    eps: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused Add + RMSNorm + FP8 per-tensor quantization kernel.

    Each program instance processes one row.
    Produces three outputs per row:
      1. BF16/FP16 normalized result (same dtype as input)
      2. FP8 E4M3 quantized result (normalized, scaled, clamped, cast)
      3. Add result (x + residual, for residual connection)
    """
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N_COLS

    # Load RMSNorm weight
    w = tl.load(weight_ptr + offsets, mask=mask)

    # Load x and residual
    x_row = x_ptr + prog_id * row_stride_x
    residual_row = residual_ptr + prog_id * row_stride_residual
    x = tl.load(x_row + offsets, mask=mask)
    residual = tl.load(residual_row + offsets, mask=mask)

    # Fused add
    add_out = x + residual
    add_out_f32 = add_out.to(tl.float32)

    # RMSNorm: variance -> normalize -> scale by weight (all in FP32)
    var = tl.sum(add_out_f32 * add_out_f32, 0) * float(1.0 / N_COLS)
    normed = add_out_f32 / tl.sqrt(var + eps)
    norm_out_f32 = w.to(tl.float32) * normed

    # Store BF16/FP16 norm output (cast from FP32 for consumers needing native dtype)
    out_bf16_row = output_bf16_ptr + prog_id * row_stride_out
    tl.store(out_bf16_row + offsets, norm_out_f32.to(x.dtype), mask=mask)

    # FP8 quantization from full FP32 norm (avoids BF16 round-trip precision loss)
    scale = tl.load(scale_ptr)
    out_scaled = norm_out_f32 / scale
    out_clamped = tl.maximum(tl.minimum(out_scaled, FP8_MAX), FP8_MIN)
    out_fp8 = out_clamped.to(tl.float8e4nv)

    # Store FP8 output
    out_fp8_row = output_fp8_ptr + prog_id * N_COLS
    tl.store(out_fp8_row + offsets, out_fp8, mask=mask)

    # Store add output (for residual connection)
    out_add_row = output_add_ptr + prog_id * row_stride_out
    tl.store(out_add_row + offsets, add_out, mask=mask)


def fused_add_rms_norm_quant_fp8(
    x: Tensor, residual: Tensor, weight: Tensor, eps: float, scale: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Fused Add + RMSNorm + FP8 quantization.

    Args:
        x: First input tensor of shape [..., hidden_size].
        residual: Second input tensor (residual) of shape [..., hidden_size].
        weight: RMSNorm weight of shape [hidden_size].
        eps: Epsilon for numerical stability.
        scale: Per-tensor FP8 quantization scale (scalar tensor).

    Returns:
        Tuple of (bf16_norm_output, fp8_norm_output, add_output),
        all with shape [..., hidden_size].
    """
    orig_shape = x.shape
    feat_size = weight.shape[0]
    x_flat = x.reshape(-1, feat_size)
    residual_flat = residual.reshape(-1, feat_size)
    seq_len = x_flat.shape[0]

    BLOCK_N = triton.next_power_of_2(feat_size)
    out_bf16 = torch.empty_like(x_flat)
    out_fp8 = torch.empty(
        x_flat.shape,
        dtype=torch.float8_e4m3fn,
        device=x.device,
    )
    out_add = torch.empty_like(x_flat)

    grid = (seq_len,)
    fused_add_rms_norm_quant_fp8_kernel[grid](
        x_flat,
        residual_flat,
        weight,
        out_bf16,
        out_fp8,
        out_add,
        scale,
        row_stride_x=x_flat.stride(-2),
        row_stride_residual=residual_flat.stride(-2),
        row_stride_out=out_bf16.stride(-2),
        eps=eps,
        FP8_MIN=_FP8_MIN,
        FP8_MAX=_FP8_MAX,
        N_COLS=feat_size,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )

    return (
        out_bf16.reshape(orig_shape),
        out_fp8.reshape(orig_shape),
        out_add.reshape(orig_shape),
    )


@torch.library.custom_op("auto_deploy::triton_fused_add_rms_norm_quant_fp8", mutates_args=())
def triton_fused_add_rms_norm_quant_fp8(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Add + RMSNorm + FP8 quantization custom op.

    Computes add, RMSNorm, and FP8 quantization in a single Triton kernel:
        add_out = x + residual
        norm_out = rms_norm(add_out, weight, eps)
        fp8_out = quantize_fp8(norm_out, scale)

    Args:
        x: First input tensor, shape [..., hidden_size].
        residual: Second input tensor (residual), shape [..., hidden_size].
        weight: RMSNorm scaling weights, shape [hidden_size].
        eps: Small constant for numerical stability.
        scale: Per-tensor FP8 quantization scale (scalar tensor, float32).

    Returns:
        Tuple of (bf16_norm_output, fp8_norm_output, add_output).
    """
    return fused_add_rms_norm_quant_fp8(x, residual, weight, eps, scale)


@triton_fused_add_rms_norm_quant_fp8.register_fake
def _fused_add_rms_norm_quant_fp8_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake implementation for tracing / shape inference."""
    bf16_out = torch.empty_like(x)
    fp8_out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    add_out = torch.empty_like(x)
    return bf16_out, fp8_out, add_out
