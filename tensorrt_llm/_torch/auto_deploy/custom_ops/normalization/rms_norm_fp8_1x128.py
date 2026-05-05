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

"""Fused RMSNorm + per-1x128-block FP8 quantization Triton kernel.

Designed to feed trtllm_finegrained_fp8_linear_prequant, eliminating the
direct_copy + scale_1x128_kernel overhead that trtllm_finegrained_fp8_linear
pays internally when it quantizes its BF16 input on-the-fly.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0


@triton.jit
def _rms_norm_fp8_1x128_kernel(
    x_ptr,
    w_ptr,
    bf16_out_ptr,
    fp8_out_ptr,
    sf_out_ptr,
    M,
    K,
    eps: tl.constexpr,
    N_QUANT: tl.constexpr,  # K // 128 (K must be divisible by 128)
):
    """Fused RMSNorm + per-128-element FP8 block quantization.

    Grid: (M,) — one program per token row.

    Two-pass over N_QUANT blocks of 128 elements per row:
      Pass 1: accumulate sum-of-squares for RMSNorm variance.
      Pass 2: apply norm, store BF16, compute per-block scale, quantize to FP8.

    Scale output layout matches fp8_quantize_1x128: [N_QUANT, M] float32,
    where sf[blk, row] = max(|norm[row, blk*128:(blk+1)*128]|) / FP8_MAX.
    """
    row = tl.program_id(0)
    blk_offsets = tl.arange(0, 128)

    # Pass 1: accumulate sum of squares across all blocks
    _var = tl.zeros([128], dtype=tl.float32)
    for blk in range(N_QUANT):
        off = blk * 128 + blk_offsets
        mask = off < K
        x = tl.load(x_ptr + row * K + off, mask=mask, other=0.0).to(tl.float32)
        _var += x * x  # masked loads give 0.0 for out-of-bounds, so x*x=0 there
    sum_sq = tl.sum(_var)
    rstd = tl.rsqrt(sum_sq / K + eps)

    # Pass 2: normalize, store BF16, per-block FP8 quant
    FP8_MAX: tl.constexpr = 448.0

    for blk in range(N_QUANT):
        off = blk * 128 + blk_offsets
        mask = off < K

        x = tl.load(x_ptr + row * K + off, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + off, mask=mask, other=0.0).to(tl.float32)
        x_norm = x * rstd * w

        # BF16 norm output
        tl.store(bf16_out_ptr + row * K + off, x_norm.to(tl.bfloat16), mask=mask)

        # Per-block scale: max(|x_norm|) / FP8_MAX
        x_abs = tl.abs(x_norm)
        max_val = tl.max(tl.where(mask, x_abs, 0.0), axis=0)
        scale = tl.maximum(max_val / FP8_MAX, 1e-12)

        # Quantize: clamp then cast to FP8
        x_fp8 = tl.minimum(tl.maximum(x_norm / scale, -FP8_MAX), FP8_MAX).to(tl.float8e4nv)
        tl.store(fp8_out_ptr + row * K + off, x_fp8, mask=mask)

        # Scale at [blk, row] in [N_QUANT, M] layout (matches fp8_quantize_1x128 output)
        tl.store(sf_out_ptr + blk * M + row, scale)


def _rms_norm_fp8_1x128_impl(
    input: Tensor, weight: Tensor, eps: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """Fused RMSNorm + per-1x128-block FP8 quantization.

    Args:
        input:  [..., K] bfloat16, K must be divisible by 128.
        weight: [K] bfloat16 norm scale weights.
        eps:    small constant for numerical stability.

    Returns:
        bf16_out: same shape as input, bfloat16 — RMSNorm output.
        fp8_out:  same shape as input, float8_e4m3fn — quantized output.
        sf_out:   [K//128, M] float32 — per-block activation scale factors,
                  in the same layout as torch.ops.trtllm.fp8_quantize_1x128.
                  M = product of all dimensions of input except the last.
    """
    orig_shape = input.shape
    K = input.shape[-1]
    M = input.numel() // K

    assert K % 128 == 0, f"K={K} must be divisible by 128 for rms_norm_fp8_1x128"
    N_QUANT = K // 128

    x2 = input.reshape(M, K)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    bf16_out = torch.empty_like(x2)
    fp8_out = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x2.device)
    sf_out = torch.empty(N_QUANT, M, dtype=torch.float32, device=x2.device)

    grid = (M,)
    _rms_norm_fp8_1x128_kernel[grid](
        x2,
        weight,
        bf16_out,
        fp8_out,
        sf_out,
        M,
        K,
        eps=eps,
        N_QUANT=N_QUANT,
    )

    return bf16_out.reshape(orig_shape), fp8_out.reshape(orig_shape), sf_out


@torch.library.custom_op("auto_deploy::rms_norm_fp8_1x128", mutates_args=())
def rms_norm_fp8_1x128(input: Tensor, weight: Tensor, eps: float) -> Tuple[Tensor, Tensor, Tensor]:
    """Custom op: fused RMSNorm + per-1x128-block FP8 quantization.

    Intended to replace flashinfer_rms_norm when its output feeds only
    trtllm_finegrained_fp8_linear nodes, eliminating the per-linear
    direct_copy + scale_1x128_kernel overhead.

    Returns (bf16_norm, fp8_quant, act_scale_factors) where act_scale_factors
    can be passed directly to trtllm_finegrained_fp8_linear_prequant.
    """
    return _rms_norm_fp8_1x128_impl(input, weight, eps)


@rms_norm_fp8_1x128.register_fake
def _rms_norm_fp8_1x128_fake(
    input: Tensor, weight: Tensor, eps: float
) -> Tuple[Tensor, Tensor, Tensor]:
    K = input.shape[-1]
    M = input.numel() // K
    N_QUANT = (K + 127) // 128
    bf16_out = torch.empty_like(input)
    fp8_out = torch.empty(input.shape, dtype=torch.float8_e4m3fn, device=input.device)
    sf_out = torch.empty(N_QUANT, M, dtype=torch.float32, device=input.device)
    return bf16_out, fp8_out, sf_out
