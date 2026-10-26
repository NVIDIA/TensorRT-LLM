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
"""Triton NVFP4 activation quantizer, bit-identical to ``trtllm::fp4_quantize``.

``torch.ops.trtllm.fp4_quantize(x, global_scale, 16, False, swizzled)`` runs
``quantize_with_block_size`` (cpp/tensorrt_llm/kernels/quantization.cuh): one
512-thread block per row with a single 16-byte load in flight per thread.
This kernel loads a ``[BLOCK_M, BLOCK_K]`` tile per program and
replays ``cvt_warp_fp16_to_fp4`` op for op with the same PTX, so its output
bytes are identical:

* the exact max of the 16 ``|x|`` of a block,
* ``SF = e4m3_rn_satfinite(gs * (vmax * rcp.approx.ftz(6)))`` (``mul.rn``; the
  library is built without fast math),
* ``out_scale = vmax != 0 ? rcp.approx.ftz(float(SF) * rcp.approx.ftz(gs)) : 0``,
* values ``e2m1_rn_satfinite(x * out_scale)``, packed two per byte, odd element
  in the high nibble,
* the same scale-factor layout: 128x4 swizzled (``get_sf_out_offset_128x4``,
  padding rows zero-filled) or linear.

Supported: contiguous BF16 input, ``K % 64 == 0``, 16-element blocks with
E4M3 scale factors (NVFP4). Everything else (and torch.compile) keeps the C++
op. Enable with ``TRTLLM_NVFP4_QUANTIZE_TRITON=1`` (default off).
"""

import os
from typing import Optional, Tuple

import torch

from ...flashinfer_utils import get_env_enable_pdl

try:
    import triton  # type: ignore[import]
    import triton.language as tl  # type: ignore[import]
except ImportError:  # pragma: no cover
    triton = None

NVFP4_QUANTIZE_TRITON_ENV = "TRTLLM_NVFP4_QUANTIZE_TRITON"

# SF dtype / packed FP4 dtype, as tensorrt_llm/quantization/utils/fp4_utils.py.
_SF_DTYPE = torch.uint8
_FP4_DTYPE = torch.uint8

# Tile dimensions for the quantization kernel.
_BLOCK_K = 512
_NUM_WARPS = 4


def nvfp4_quantize_triton_enabled() -> bool:
    """Whether ``TRTLLM_NVFP4_QUANTIZE_TRITON=1`` (read per call: cheap)."""
    return triton is not None and os.environ.get(NVFP4_QUANTIZE_TRITON_ENV, "0") == "1"


if triton is not None:

    @triton.jit
    def _rcp_approx_ftz(x):
        return tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def _mul_rn(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def _f32_to_e4m3_bits(x):
        # One float -> its E4M3 byte (cvt.*x2 converts a pair; both halves get x).
        return (
            tl.inline_asm_elementwise(
                "{\n.reg .b16 t;\ncvt.rn.satfinite.e4m3x2.f32 t, $1, $1;\ncvt.u32.u16 $0, t;\n}",
                "=r,f",
                [x],
                dtype=tl.int32,
                is_pure=True,
                pack=1,
            )
            & 255
        )

    @triton.jit
    def _f32x2_to_e2m1x2(hi, lo):
        # One byte = e2m1(hi) << 4 | e2m1(lo), as fp32_vec_to_e2m1 packs it.
        return tl.inline_asm_elementwise(
            "{\n.reg .b8 t;\ncvt.rn.satfinite.e2m1x2.f32 t, $1, $2;\ncvt.u32.u8 $0, t;\n}",
            "=r,f,f",
            [hi, lo],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _e4m3_bits_to_f32(bits):
        # Exact decode of a non-negative E4M3 byte held in an int32
        # (static_cast<float>(__nv_fp8_e4m3)).
        e = (bits >> 3) & 15
        m = bits & 7
        normal = (((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
        subnormal = m.to(tl.float32) * 0.001953125
        nan = tl.full(bits.shape, 0x7FC00000, tl.int32).to(tl.float32, bitcast=True)
        return tl.where((bits & 127) == 127, nan, tl.where(e == 0, subnormal, normal))

    @triton.jit(do_not_specialize=["M"])
    def _nvfp4_quantize_kernel(
        x_ptr,
        gs_ptr,
        q_ptr,
        sf_ptr,
        M,
        K: tl.constexpr,
        SWIZZLED: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        USE_PDL: tl.constexpr,
    ):
        NV: tl.constexpr = BLOCK_K // 16
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        row_ok = rows < M
        cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + rows[:, None] * K + cols[None, :], mask=row_ok[:, None], other=0.0)
        xg = tl.reshape(x.to(tl.float32), (BLOCK_M, NV, 16))
        vmax = tl.max(tl.abs(xg), axis=2)
        gs = tl.load(gs_ptr) + tl.zeros((BLOCK_M, NV), tl.float32)
        rcp6 = _rcp_approx_ftz(tl.full((BLOCK_M, NV), 6.0, tl.float32))
        sf = _mul_rn(gs, _mul_rn(vmax, rcp6))
        sf_bits = _f32_to_e4m3_bits(sf)
        sf_val = _e4m3_bits_to_f32(sf_bits)
        out_scale = _rcp_approx_ftz(_mul_rn(sf_val, _rcp_approx_ftz(gs)))
        out_scale = tl.where(vmax != 0.0, out_scale, 0.0)
        y = _mul_rn(xg, tl.broadcast_to(out_scale[:, :, None], (BLOCK_M, NV, 16)))
        lo, hi = tl.split(tl.reshape(y, (BLOCK_M, BLOCK_K // 2, 2)))
        q = _f32x2_to_e2m1x2(hi, lo).to(tl.uint8)
        qcols = pid_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
        tl.store(q_ptr + rows[:, None] * (K // 2) + qcols[None, :], q, mask=row_ok[:, None])
        sf_bits = tl.where(row_ok[:, None], sf_bits, 0).to(tl.uint8)
        kv = pid_k * NV + tl.arange(0, NV)
        if SWIZZLED:
            # get_sf_out_offset_128x4: [mTile, kTile, 32, 4, 4]. The grid covers
            # exactly pad_up(M, 128) rows, so padding rows get a zero SF.
            K_TILES: tl.constexpr = (K // 16 + 3) // 4
            off = (
                (rows // 128)[:, None] * (K_TILES * 512)
                + (kv // 4)[None, :] * 512
                + (rows % 32)[:, None] * 16
                + ((rows % 128) // 32)[:, None] * 4
                + (kv % 4)[None, :]
            )
            tl.store(sf_ptr + off, sf_bits)
        else:
            tl.store(
                sf_ptr + rows[:, None] * (K // 16) + kv[None, :], sf_bits, mask=row_ok[:, None]
            )


def nvfp4_quantize_triton_supported(
    x: torch.Tensor,
    global_scale: Optional[torch.Tensor],
    sf_vec_size: int = 16,
    sf_use_ue8m0: bool = False,
) -> bool:
    """Inputs for which nvfp4_quantize_triton reproduces fp4_quantize bit for bit."""
    return (
        triton is not None
        and sf_vec_size == 16
        and not sf_use_ue8m0
        and isinstance(global_scale, torch.Tensor)
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() >= 2
        and x.is_contiguous()
        and x.numel() > 0
        and x.shape[-1] % 64 == 0
        and global_scale.is_cuda
        and global_scale.dtype == torch.float32
        and global_scale.numel() == 1
        and not torch.compiler.is_compiling()
    )


def _pad_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def nvfp4_quantize_triton(
    x: torch.Tensor, global_scale: torch.Tensor, is_sf_swizzled_layout: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``torch.ops.trtllm.fp4_quantize(x, global_scale, 16, False,
    is_sf_swizzled_layout)``, bit for bit (see nvfp4_quantize_triton_supported).

    Returns ``(fp4 [..., K/2] uint8, sf uint8)``; ``sf`` is 1-D with
    ``pad_up(M, 128) * K/16`` (swizzled) or ``M * K/16`` (linear) elements, as
    the C++ op returns it.
    """
    k = x.shape[-1]
    m = x.numel() // k
    block_m = 16 if k >= 16384 else 8
    block_k = _BLOCK_K
    while k % block_k:
        block_k //= 2
    q = torch.empty((*x.shape[:-1], k // 2), dtype=_FP4_DTYPE, device=x.device)
    if is_sf_swizzled_layout:
        grid_rows = _pad_up(m, 128)
        sf = torch.empty((grid_rows * _pad_up(k // 16, 4),), dtype=_SF_DTYPE, device=x.device)
    else:
        grid_rows = m
        sf = torch.empty((m * (k // 16),), dtype=_SF_DTYPE, device=x.device)
    use_pdl = get_env_enable_pdl()
    grid = (triton.cdiv(grid_rows, block_m), k // block_k)
    _nvfp4_quantize_kernel[grid](
        x,
        global_scale,
        q,
        sf,
        m,
        K=k,
        SWIZZLED=is_sf_swizzled_layout,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        USE_PDL=use_pdl,
        num_warps=_NUM_WARPS,
        launch_pdl=use_pdl,
    )
    return q, sf


def fp4_quantize(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    sf_vec_size: int = 16,
    sf_use_ue8m0: bool = False,
    is_sf_swizzled_layout: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for ``torch.ops.trtllm.fp4_quantize``: the Triton kernel when
    TRTLLM_NVFP4_QUANTIZE_TRITON=1 and the inputs are supported (same bytes),
    the C++ op otherwise."""
    if nvfp4_quantize_triton_enabled() and nvfp4_quantize_triton_supported(
        x, global_scale, sf_vec_size, sf_use_ue8m0
    ):
        return nvfp4_quantize_triton(x, global_scale, is_sf_swizzled_layout)
    return torch.ops.trtllm.fp4_quantize(
        x, global_scale, sf_vec_size, sf_use_ue8m0, is_sf_swizzled_layout
    )
