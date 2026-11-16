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
conversions. ``nvfp4_fp8_dual_quant`` also writes the per-tensor E4M3 copy of the same read, as
``static_quantize_e4m3_per_tensor`` (``scaleMatrixPerTensorVec``: ``x * (1.0f / s)``, RN, saturating).
"""

from typing import Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from ..flashinfer_utils import get_env_enable_pdl

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
    fp8_scale_ptr,  # None unless write_fp8
    fp8_ptr,  # None unless write_fp8
    num_tokens: tl.tensor,
    hidden: tl.constexpr,
    block_t: tl.constexpr,
    block_h: tl.constexpr,
    write_fp8: tl.constexpr,
    launch_with_pdl: tl.constexpr,
) -> None:
    if launch_with_pdl:
        tl.extra.cuda.gdc_wait()
    num_blocks: tl.constexpr = block_h // 16
    rows = tl.program_id(0).to(tl.int64) * block_t + tl.arange(0, block_t)
    cols = tl.program_id(1) * block_h + tl.arange(0, block_h)
    mask = (rows < num_tokens)[:, None]
    x = tl.load(x_ptr + rows[:, None] * hidden + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    global_scale = tl.load(global_scale_ptr)
    blocks = tl.reshape(x, (block_t, num_blocks, 16))
    pairs = tl.reshape(tl.abs(blocks), (block_t, num_blocks, 8, 2))
    even, odd = tl.split(pairs)
    even_max = tl.max(even, axis=2)
    odd_max = tl.max(odd, axis=2)
    vec_max = tl.where(even_max > odd_max, even_max, odd_max)
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
    if write_fp8:
        fp8_factor = tldevice.rcp_rn(tl.load(fp8_scale_ptr))
        tl.store(
            fp8_ptr + rows[:, None] * hidden + cols[None, :],
            (x * fp8_factor).to(tl.float8e4nv),
            mask=mask,
        )

    if launch_with_pdl:
        # Publish both outputs before any warp signals completion to a consumer.
        tl.inline_asm_elementwise(
            "fence.acq_rel.gpu; // dummy $0",
            "=r,~{memory}",
            [],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        tl.debug_barrier()
        tl.extra.cuda.gdc_launch_dependents()


def can_use_nvfp4_quant(x: torch.Tensor, global_scale: torch.Tensor) -> bool:
    """Check input eligibility; the caller must separately check the device ISA.

    Args:
        x: Contiguous CUDA bfloat16 activations of shape ``[T, H]``, where
            ``T > 0`` and ``H`` is a positive multiple of 1024.
        global_scale: One-element float32 tensor on the same device as ``x``.

    Returns:
        Whether the tensors satisfy the quantizer's input requirements.
        This does not check the device's FP4 instruction support.
    """
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


def _launch(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    fp8_scale: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    num_tokens, hidden = x.shape
    q = torch.empty((num_tokens, hidden // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty((num_tokens * (hidden // 16),), dtype=torch.uint8, device=x.device)
    fp8 = None if fp8_scale is None else torch.empty_like(x, dtype=torch.float8_e4m3fn)
    launch_with_pdl = get_env_enable_pdl()
    _nvfp4_quant_kernel[(triton.cdiv(num_tokens, 4), hidden // _BLOCK_H)](
        x,
        global_scale,
        q,
        sf,
        fp8_scale,
        fp8,
        num_tokens,
        hidden=hidden,
        block_t=4,
        block_h=_BLOCK_H,
        write_fp8=fp8 is not None,
        launch_with_pdl=launch_with_pdl,
        launch_pdl=launch_with_pdl,
        num_warps=2,
        enable_fp_fusion=False,
    )
    return q, sf, fp8


def nvfp4_quant(x: torch.Tensor, global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 ``[T, H]`` with a float32 CUDA scalar global scale.

    Requires a Blackwell-or-later FP4 ISA and ``can_use_nvfp4_quant`` inputs.
    Returns packed FP4 ``[T, H/2]`` and linear E4M3 scales ``[T * H/16]``,
    both uint8, as ``fp4_quantize(x, global_scale, 16, False, False)``.
    """
    q, sf, _ = _launch(x, global_scale, None)
    return q, sf


def nvfp4_fp8_dual_quant(
    x: torch.Tensor, global_scale: torch.Tensor, fp8_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``nvfp4_quant`` plus the per-tensor E4M3 copy of the same input, in one read.

    Requires ``can_use_nvfp4_quant(x, global_scale)`` and a one-element float32
    ``fp8_scale`` on the same device. Returns ``(q, sf, fp8)``: ``q`` and ``sf``
    as ``nvfp4_quant``, and ``fp8`` ``[T, H]`` equal to
    ``static_quantize_e4m3_per_tensor(x, fp8_scale)``.
    """
    return _launch(x, global_scale, fp8_scale)
