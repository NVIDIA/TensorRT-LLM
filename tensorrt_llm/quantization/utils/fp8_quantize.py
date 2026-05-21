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
"""Triton FP8 quantization kernels.

Contains:
  - 1x128 block-scale quantization with optional UE8M0 scale rounding
    (alternative to CUDA ``fp8_quantize_1x128`` on SM100+)
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# 1x128 block-scale quantization
# ---------------------------------------------------------------------------


@triton.jit
def _fp8_1x128_quantize_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    output_ptr,
    stride_output_0,
    stride_output_1,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    m,
    k,
    fp8_max,
    fp8_min,
    M_BLOCK: tl.constexpr,
    K_BLOCK: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    k_block_idx = tl.program_id(0)
    m_block_idx = tl.program_id(1)

    offs_m = m_block_idx * M_BLOCK + tl.arange(0, M_BLOCK)
    offs_k = k_block_idx * K_BLOCK + tl.arange(0, K_BLOCK)

    in_ptrs = input_ptr + (offs_m[:, None] * stride_input_0 + offs_k[None, :] * stride_input_1)
    out_ptrs = output_ptr + (offs_m[:, None] * stride_output_0 + offs_k[None, :] * stride_output_1)

    valid = (offs_k[None, :] < k) & (offs_m[:, None] < m)
    act = tl.load(in_ptrs, mask=valid, other=0.0).to(tl.float32)

    absmax = tl.maximum(tl.max(tl.abs(act), axis=1), 1e-10)
    scale = absmax / fp8_max
    if SCALE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(tl.abs(scale))))

    qval = tl.clamp(act / scale.expand_dims(1), fp8_min, fp8_max).to(output_ptr.dtype.element_ty)

    tl.store(out_ptrs, qval, mask=valid)

    scale_ptrs = (
        output_scale_ptr
        + k_block_idx * stride_output_scale_0
        + (m_block_idx * M_BLOCK + tl.arange(0, M_BLOCK)) * stride_output_scale_1
    )
    tl.store(scale_ptrs, scale, mask=(offs_m < m))


@torch.compiler.disable()
def triton_fp8_quantize_1x128(
    input: torch.Tensor,
    quant_group_size: int = 128,
    use_ue8m0: bool = True,
) -> tuple:
    """FP8 E4M3 1x128 block-scale quantization via Triton.

    Drop-in replacement for ``torch.ops.trtllm.fp8_quantize_1x128`` on SM89+.
    Faster than the CUDA kernel when M is large (crossover ~2-4k rows on B200).

    Args:
        input: BF16 tensor of shape ``[m, k]`` (must be contiguous).
        quant_group_size: Block size along K for scale computation (default 128).
        use_ue8m0: If True, round scales to power-of-2 (UE8M0 format).

    Returns:
        ``(fp8_output, scale)`` where ``fp8_output`` is ``[m, k]`` float8_e4m3fn
        and ``scale`` is ``[scale_k, m]`` float32.
    """
    assert input.is_contiguous() and input.dim() == 2
    m, k = input.shape
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    output = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=input.device)
    scale_k = (k + quant_group_size - 1) // quant_group_size
    output_scale = torch.empty((scale_k, m), dtype=torch.float32, device=input.device)

    K_BLOCK = quant_group_size
    M_BLOCK = 128
    grid = (triton.cdiv(k, K_BLOCK), triton.cdiv(m, M_BLOCK), 1)

    _fp8_1x128_quantize_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        m,
        k,
        fp8_max,
        fp8_min,
        M_BLOCK=M_BLOCK,
        K_BLOCK=K_BLOCK,
        SCALE_UE8M0=use_ue8m0,
        num_warps=8,
    )
    return output, output_scale
