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

"""Triton RMSNorm + FP8 quantization custom ops."""

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

_FP8_MIN = float(torch.finfo(torch.float8_e4m3fn).min)
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


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
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N_COLS

    w = tl.load(weight_ptr + offsets, mask=mask)
    x_ptr = input_ptr + prog_id * row_stride
    x = tl.load(x_ptr + offsets, mask=mask)
    xf = x.to(tl.float32)

    var = tl.sum(xf * xf, 0) * float(1.0 / N_COLS)
    normed = xf / tl.sqrt(var + eps)
    out_f32 = w.to(tl.float32) * normed

    out_bf16_row = output_bf16_ptr + prog_id * row_stride
    tl.store(out_bf16_row + offsets, out_f32.to(x.dtype), mask=mask)

    scale = tl.load(scale_ptr)
    out_scaled = out_f32 / scale
    out_clamped = tl.maximum(tl.minimum(out_scaled, FP8_MAX), FP8_MIN)
    out_fp8 = out_clamped.to(tl.float8e4nv)

    out_fp8_row = output_fp8_ptr + prog_id * N_COLS
    tl.store(out_fp8_row + offsets, out_fp8, mask=mask)


def rms_norm_quant_fp8(
    hidden_states: Tensor, weight: Tensor, eps: float, scale: Tensor
) -> Tuple[Tensor, Tensor]:
    assert hidden_states.shape[-1] == weight.numel(), "hidden size must match weight size"

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
    return rms_norm_quant_fp8(input, weight, eps, scale)


@triton_rms_norm_quant_fp8.register_fake
def _rms_norm_quant_fp8_fake(
    input: torch.Tensor, weight: torch.Tensor, eps: float, scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N_COLS

    w = tl.load(weight_ptr + offsets, mask=mask)
    x_row = x_ptr + prog_id * row_stride_x
    residual_row = residual_ptr + prog_id * row_stride_residual
    x = tl.load(x_row + offsets, mask=mask)
    residual = tl.load(residual_row + offsets, mask=mask)

    add_out = x + residual
    add_out_f32 = add_out.to(tl.float32)

    var = tl.sum(add_out_f32 * add_out_f32, 0) * float(1.0 / N_COLS)
    normed = add_out_f32 / tl.sqrt(var + eps)
    norm_out_f32 = w.to(tl.float32) * normed

    out_bf16_row = output_bf16_ptr + prog_id * row_stride_out
    tl.store(out_bf16_row + offsets, norm_out_f32.to(x.dtype), mask=mask)

    scale = tl.load(scale_ptr)
    out_scaled = norm_out_f32 / scale
    out_clamped = tl.maximum(tl.minimum(out_scaled, FP8_MAX), FP8_MIN)
    out_fp8 = out_clamped.to(tl.float8e4nv)

    out_fp8_row = output_fp8_ptr + prog_id * N_COLS
    tl.store(out_fp8_row + offsets, out_fp8, mask=mask)

    out_add_row = output_add_ptr + prog_id * row_stride_out
    tl.store(out_add_row + offsets, add_out, mask=mask)


def fused_add_rms_norm_quant_fp8(
    x: Tensor, residual: Tensor, weight: Tensor, eps: float, scale: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    assert x.shape == residual.shape, "x and residual must have identical shape"
    assert x.shape[-1] == weight.numel(), "x hidden size must match weight size"

    orig_shape = x.shape
    feat_size = weight.shape[0]
    x_flat = x.reshape(-1, feat_size)
    residual_flat = residual.reshape(-1, feat_size)
    seq_len = x_flat.shape[0]

    BLOCK_N = triton.next_power_of_2(feat_size)
    out_bf16 = torch.empty_like(x_flat)
    out_fp8 = torch.empty(x_flat.shape, dtype=torch.float8_e4m3fn, device=x.device)
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

    return out_bf16.reshape(orig_shape), out_fp8.reshape(orig_shape), out_add.reshape(orig_shape)


@torch.library.custom_op("auto_deploy::triton_fused_add_rms_norm_quant_fp8", mutates_args=())
def triton_fused_add_rms_norm_quant_fp8(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return fused_add_rms_norm_quant_fp8(x, residual, weight, eps, scale)


@triton_fused_add_rms_norm_quant_fp8.register_fake
def _fused_add_rms_norm_quant_fp8_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bf16_out = torch.empty_like(x)
    fp8_out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    add_out = torch.empty_like(x)
    return bf16_out, fp8_out, add_out
