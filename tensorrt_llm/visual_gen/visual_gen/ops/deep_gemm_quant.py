# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch

from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import triton
    import triton.language as tl
except ImportError:
    logger.warning("Triton is not installed, deepgemm on sm100 requires triton")


@torch.compiler.disable()
def quant_and_transform_ue8m0(
    input: torch.Tensor,
    quant_group_size: int = 128,
    scale_ue8m0: bool = True,
):
    """Input shape [m, k]
    output shape [m, k // 2], dtype fp8
    output_scale shape[m, k // 128], dtype float32
    """
    assert input.is_contiguous()
    assert len(input.shape) == 2
    assert input.shape[-1] % 2 == 0

    # FP8 quantization parameters
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    m, k = input.shape

    # Create output
    output = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=input.device)

    # Create output scale
    scale_k = (k + quant_group_size - 1) // quant_group_size

    output_scale = torch.empty(
        (scale_k, m), dtype=torch.float32, device=input.device
    )  # output_scale shape [k // 128, m]

    k_block_size = quant_group_size  # BLOCK = 128
    k_block_num = triton.cdiv(k, k_block_size)
    m_block_size = 128
    m_block_num = triton.cdiv(m, m_block_size)

    num_warps = 8
    NUM_STAGES = 1

    grid = (
        k_block_num,
        m_block_num,
        1,
    )
    quant_and_transform_ue8m0_kernel[grid](
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
        m_block_size=m_block_size,
        k_block_size=k_block_size,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
        SCALE_UE8M0=scale_ue8m0,
    )
    output_scale = output_scale.transpose(0, 1)
    return output, output_scale


@triton.jit
def quant_and_transform_ue8m0_kernel(
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
    m_block_size: tl.constexpr,
    k_block_size: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    k_block_index = tl.program_id(0)
    m_block_index = tl.program_id(1)

    offs_m = m_block_index * m_block_size + tl.arange(0, m_block_size)
    offs_k = k_block_index * k_block_size + tl.arange(0, k_block_size)

    input_ptrs = input_ptr + (offs_m[:, None] * stride_input_0 + offs_k[None, :] * stride_input_1)
    output_ptrs = output_ptr + (
        offs_m[:, None] * stride_output_0 + offs_k[None, :] * stride_output_1
    )

    offs_scale_m = m_block_index * m_block_size + tl.arange(0, m_block_size)
    output_scale_ptrs = (
        output_scale_ptr
        + k_block_index * stride_output_scale_0
        + offs_scale_m * stride_output_scale_1
    )

    act = tl.load(input_ptrs, mask=(offs_k[None, :] < k) & (offs_m[:, None] < m), other=0.0).to(
        tl.float32
    )

    _absmax = tl.maximum(tl.max(tl.abs(act), axis=1), 1e-10)
    output_s = _absmax / fp8_max

    output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))
    output_s_1 = output_s.expand_dims(1)

    output_q = tl.clamp(act / output_s_1, fp8_min, fp8_max).to(output_ptr.dtype.element_ty)

    tl.store(
        output_ptrs,
        output_q,
        mask=(offs_k[None, :] < k) & (offs_m[:, None] < m),
    )
    tl.store(
        output_scale_ptrs,
        output_s,
        mask=(offs_scale_m < m),
    )
