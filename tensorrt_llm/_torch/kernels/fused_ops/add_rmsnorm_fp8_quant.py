# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Fused residual add + RMSNorm + static FP8 (E4M3) quantization.

One pass writes the bf16 normalized activation and its FP8 copy. The residual sum matches
flashinfer's fused add + RMSNorm exactly and the FP8 copy is the exact static quantization of
the bf16 output, which may differ from flashinfer's by one ulp (sum-of-squares order).
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


@triton.jit
def _add_rmsnorm_fp8_quant_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    scale_ptr,
    fp8_ptr,
    eps,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WEIGHT_OFFSET: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    offsets = tl.program_id(0).to(tl.int64) * N + cols

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(residual_ptr + offsets, x.to(residual_ptr.dtype.element_ty), mask=mask)

    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normed = (x * rstd * (weight + WEIGHT_OFFSET)).to(input_ptr.dtype.element_ty)
    tl.store(input_ptr + offsets, normed, mask=mask)

    inv_scale = tldevice.rcp_rn(tl.load(scale_ptr).to(tl.float32))
    fp8 = (normed.to(tl.float32) * inv_scale).to(fp8_ptr.dtype.element_ty)
    tl.store(fp8_ptr + offsets, fp8, mask=mask)


def add_rmsnorm_fp8_quant_supported(input: torch.Tensor, residual: torch.Tensor) -> bool:
    """Whether ``add_rmsnorm_fp8_quant`` handles this input/residual pair."""
    return (
        input.dtype == residual.dtype == torch.bfloat16
        and input.shape == residual.shape
        and input.is_contiguous()
        and residual.is_contiguous()
        and 0 < input.shape[-1] <= 16384
    )


def add_rmsnorm_fp8_quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    use_gemma: bool = False,
) -> torch.Tensor:
    """Fused ``residual += input``, ``input = rmsnorm(residual)`` and FP8 quantize.

    Args:
        input: Contiguous bf16 tensor [..., H]; overwritten with the normalized
            activation.
        residual: Contiguous bf16 tensor shaped like ``input``; overwritten
            with ``input + residual``.
        weight: [H] RMSNorm weight; Gemma-style norms scale by ``1 + weight``.
        scale: One-element fp32 static input scale of the consuming FP8 Linear.
        eps: RMSNorm epsilon.
        use_gemma: Whether the norm scales by ``1 + weight``.

    Returns:
        The float8_e4m3fn quantization of the normalized activation,
        ``fp8(input / scale)``, shaped like ``input``.
    """
    assert add_rmsnorm_fp8_quant_supported(input, residual)
    assert scale.dtype == torch.float32 and scale.numel() == 1
    hidden_size = input.shape[-1]
    assert weight.shape == (hidden_size,)

    fp8 = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    num_rows = input.numel() // hidden_size
    if num_rows == 0:
        return fp8
    block_n = triton.next_power_of_2(hidden_size)
    _add_rmsnorm_fp8_quant_kernel[(num_rows,)](
        input,
        residual,
        weight,
        scale,
        fp8,
        eps,
        N=hidden_size,
        BLOCK_N=block_n,
        WEIGHT_OFFSET=1.0 if use_gemma else 0.0,
        num_warps=min(max(block_n // 256, 4), 16),
    )
    return fp8
