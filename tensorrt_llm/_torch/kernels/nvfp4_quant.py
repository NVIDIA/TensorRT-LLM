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
"""NVFP4 activation quant (16-element blocks, E4M3 scale factors, linear SF layout) in Triton.

Matches the CUDA arithmetic in ``torch.ops.trtllm.fp4_quantize(x, global_scale, 16, False, False)``: it follows
``cvt_warp_fp16_to_fp4`` step for step, including the ``rcp.approx.ftz`` reciprocals and the ``cvt.rn.satfinite``
conversions.
"""

import torch
import triton
import triton.language as tl

_BLOCK_H = 1024


@triton.jit
def _rcp_approx_ftz(x: tl.tensor) -> tl.tensor:
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _e2m1x2(lo: tl.tensor, hi: tl.tensor) -> tl.tensor:
    """Pack two fp32 values as e2m1 nibbles (lo in bits 0-3), round to nearest even, saturate."""
    return tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t; }",
        "=r,f,f",
        [lo, hi],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _nvfp4_quant_kernel(
    x_ptr: tl.tensor,
    global_scale_ptr: tl.tensor,
    q_ptr: tl.tensor,
    sf_ptr: tl.tensor,
    num_tokens: tl.tensor,
    hidden: tl.constexpr,
    block_t: tl.constexpr,
    block_h: tl.constexpr,
) -> None:
    num_blocks: tl.constexpr = block_h // 16
    rows = tl.program_id(0).to(tl.int64) * block_t + tl.arange(0, block_t)
    cols = tl.program_id(1) * block_h + tl.arange(0, block_h)
    mask = (rows < num_tokens)[:, None]
    x = tl.load(x_ptr + rows[:, None] * hidden + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    global_scale = tl.load(global_scale_ptr)
    blocks = tl.reshape(x, (block_t, num_blocks, 16))
    vec_max = tl.max(tl.abs(blocks), axis=2)
    rcp_six = _rcp_approx_ftz(tl.full((), 6.0, tl.float32))
    rcp_global_scale = _rcp_approx_ftz(global_scale)
    sf = (global_scale * (vec_max * rcp_six)).to(tl.float8e4nv)
    sf_value = sf.to(tl.float32)
    output_scale = tl.where(vec_max != 0, _rcp_approx_ftz(sf_value * rcp_global_scale), 0.0)
    scaled = tl.reshape(blocks * output_scale[:, :, None], (block_t, block_h // 2, 2))
    lo, hi = tl.split(scaled)
    q_cols = tl.program_id(1) * (block_h // 2) + tl.arange(0, block_h // 2)
    tl.store(
        q_ptr + rows[:, None] * (hidden // 2) + q_cols[None, :],
        _e2m1x2(lo, hi).to(tl.uint8),
        mask=mask,
    )
    sf_cols = tl.program_id(1) * num_blocks + tl.arange(0, num_blocks)
    tl.store(
        sf_ptr + rows[:, None] * (hidden // 16) + sf_cols[None, :],
        sf.to(tl.uint8, bitcast=True),
        mask=mask,
    )


def can_use_nvfp4_quant(x: torch.Tensor, global_scale: torch.Tensor) -> bool:
    """Check the BF16, linear-layout contract; the caller must check the device ISA."""
    return (
        x.is_cuda
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.shape[1] > 0
        and x.shape[1] % _BLOCK_H == 0
        and x.shape[0] > 0
        and global_scale.device == x.device
        and global_scale.dtype == torch.float32
        and global_scale.numel() == 1
    )


def nvfp4_quant(x: torch.Tensor, global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 ``[T, H]`` with a float32 CUDA scalar global scale.

    Requires a Blackwell-or-later FP4 ISA and ``can_use_nvfp4_quant`` inputs.
    Returns packed FP4 ``[T, H/2]`` and linear E4M3 scales ``[T * H/16]``,
    both uint8, as ``fp4_quantize(x, global_scale, 16, False, False)``.
    """
    num_tokens, hidden = x.shape
    q = torch.empty((num_tokens, hidden // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty((num_tokens * (hidden // 16),), dtype=torch.uint8, device=x.device)
    _nvfp4_quant_kernel[(triton.cdiv(num_tokens, 4), hidden // _BLOCK_H)](
        x,
        global_scale,
        q,
        sf,
        num_tokens,
        hidden=hidden,
        block_t=4,
        block_h=_BLOCK_H,
        num_warps=2,
        enable_fp_fusion=False,
    )
    return q, sf
