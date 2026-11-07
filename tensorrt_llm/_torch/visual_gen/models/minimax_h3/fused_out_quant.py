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
"""NVFP4 quantization of the MiniMax-H3 attention output for the static-NVFP4 ``to_out`` Linear.

The attention output has no producer kernel to fuse its quantization into (the attention backend
owns the epilogue), so the Linear re-reads the BF16 tensor through ``quantize_with_block_size``.
That kernel runs one row per thread block with one 16-byte load per thread in flight, which on a
Rubin-class part leaves it bound by memory-level parallelism at roughly a fifth of the HBM
bandwidth. The kernel here computes the same bytes with the access pattern of ``fused_quant.py``:
a tile of ``RB`` rows by ``CH`` columns per program, loaded as one contiguous block per row.

Rounding points are those of ``cvt_warp_fp16_to_fp4`` (cpp/tensorrt_llm/kernels/quantization.cuh):
block amax over 16 columns, ``sf = e4m3(gs * (amax * rcp(6)))`` with ``rcp.approx.ftz``,
``scale = rcp(float(sf) * rcp(gs))``, codes ``cvt.rn.satfinite.e2m1x2.f32`` (low nibble = even
column), scales in the swizzled 128x4 layout with zeroed padding rows. Every rounding point goes
through PTX so the compiler cannot fold it; the output equals
``fp4_quantize(x, input_scale, 16, sf_use_ue8m0=False)`` byte for byte.
"""

import os
from typing import Tuple

import torch
import triton
import triton.language as tl

# "trtllm" inside the package; the study tree imports this file as h3_fused_out_quant.
_NS = "h3study" if __name__.startswith("h3_") else "trtllm"
_ENV = os.environ.get("H3_OUT_QUANT", "1")
OUT_QUANT_ENABLED = _ENV not in ("0", "off", "false")


def _launch_config() -> Tuple[int, int, int]:
    """(rows per program, columns per program, warps) from the environment.

    Defaults are the R200 sweep best: 16 rows x 512 columns per program, 4 warps.
    """
    rb = int(os.environ.get("H3_OUT_QUANT_RB", "16"))
    ch = int(os.environ.get("H3_OUT_QUANT_CH", "512"))
    warps = int(os.environ.get("H3_OUT_QUANT_WARPS", "4"))
    if rb & (rb - 1) or ch & (ch - 1) or ch % 16 or rb < 1 or ch < 16:
        raise ValueError(
            f"H3_OUT_QUANT_RB / _CH must be powers of two, CH a multiple of 16: {rb}, {ch}"
        )
    return rb, ch, warps


def swizzled_sf_rows(m: int) -> int:
    return ((m + 127) // 128) * 128


def swizzled_sf_ktiles(k: int) -> int:
    return ((k // 16) + 3) // 4


def sf_numel(rows: int, k: int) -> int:
    return swizzled_sf_rows(rows) * swizzled_sf_ktiles(k) * 4


# -------------------------------------------------------------------------- PTX rounding points


@triton.jit
def _rcp_approx(x):
    """rcp.approx.ftz.f32, as TRT-LLM's reciprocal_approximate_ftz."""
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _e4m3_encode(x):
    """f32 -> e4m3 byte (int8), cvt.rn.satfinite.e4m3x2.f32: what __nv_fp8_e4m3(float) runs."""
    return tl.inline_asm_elementwise(
        "{ .reg .b16 lo, hi; "
        "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1; "
        "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3; "
        "mov.b32 $0, {lo, hi}; }",
        "=r,f,f,f,f",
        [x],
        dtype=tl.int8,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _e2m1_pack(lo, hi):
    """Two f32 planes -> one byte per pair: low nibble = lo (even column), high nibble = hi."""
    return tl.inline_asm_elementwise(
        "{ .reg .b8 b0, b1, b2, b3; "
        "cvt.rn.satfinite.e2m1x2.f32 b0, $5, $1; "
        "cvt.rn.satfinite.e2m1x2.f32 b1, $6, $2; "
        "cvt.rn.satfinite.e2m1x2.f32 b2, $7, $3; "
        "cvt.rn.satfinite.e2m1x2.f32 b3, $8, $4; "
        "mov.b32 $0, {b0, b1, b2, b3}; }",
        "=r,f,f,f,f,f,f,f,f",
        [lo, hi],
        dtype=tl.int8,
        is_pure=True,
        pack=4,
    )


# -------------------------------------------------------------------------- kernel


@triton.jit
def _sf_row_base(r64, SF_KTILES):
    """Swizzled 128x4 layout: [mTile, kTile, m % 32, (m % 128) // 32, k % 4]; k-tile term added per block."""
    kt = SF_KTILES.to(tl.int64)
    return (r64 // 128) * (kt * 512) + (r64 % 32) * 16 + ((r64 % 128) // 32) * 4


@triton.jit
def _h3_out_quant_kernel(
    x_ptr,
    xq_ptr,
    sf_ptr,
    gs_ptr,
    rows,
    rows_sf,
    K,
    x_stride,
    xq_stride,
    SF_KTILES,
    RB: tl.constexpr,
    CH: tl.constexpr,
):
    """Quantize the tile [row0, row0 + RB) x [col0, col0 + CH) of a BF16 [rows, K] matrix to NVFP4.

    Programs over rows in [rows, rows_sf) (the 128-row padding of the swizzled scale layout) only
    zero their scale bytes; the packed codes are sized [rows, K / 2] and are never written there.
    """
    KB: tl.constexpr = CH // 16
    row0 = tl.program_id(0) * RB
    col0 = tl.program_id(1) * CH
    r = row0 + tl.arange(0, RB)
    r64 = r.to(tl.int64)
    valid = r < rows
    c = col0 + tl.arange(0, CH)
    x = tl.load(
        x_ptr + r64[:, None] * x_stride + c[None, :],
        mask=valid[:, None] & (c < K)[None, :],
        other=0.0,
    ).to(tl.float32)
    t = tl.reshape(x, [RB, KB, 16])
    amax = tl.max(tl.abs(t), axis=2)  # [RB, KB] exact (bf16 values)
    gs = tl.load(gs_ptr).to(tl.float32)
    rcp_gs = _rcp_approx(gs)
    rcp6 = _rcp_approx(gs * 0.0 + 6.0)
    sfv = gs * (amax * rcp6)
    sf_b = _e4m3_encode(sfv)  # [RB, KB] int8
    sf_f = sf_b.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    oscale = tl.where(amax != 0.0, _rcp_approx(sf_f * rcp_gs), 0.0)
    scaled = t * oscale[:, :, None]
    lo, hi = tl.split(tl.reshape(scaled, [RB, KB, 8, 2]))
    packed = _e2m1_pack(lo, hi)  # [RB, KB, 8] int8
    kk = tl.arange(0, KB)
    kidx = col0 // 16 + kk  # [KB] block index along K
    kmask = kidx < K // 16
    qoff = (
        r64[:, None, None] * xq_stride
        + col0 // 2
        + kk[None, :, None] * 8
        + tl.arange(0, 8)[None, None, :]
    )
    tl.store(
        xq_ptr + qoff,
        packed.to(tl.uint8, bitcast=True),
        mask=valid[:, None, None] & kmask[None, :, None],
    )
    soff = (
        _sf_row_base(r64, SF_KTILES)[:, None]
        + (kidx // 4).to(tl.int64)[None, :] * 512
        + (kidx % 4)[None, :]
    )
    sf_out = tl.where(valid[:, None], sf_b, tl.zeros_like(sf_b))
    tl.store(
        sf_ptr + soff,
        sf_out.to(tl.uint8, bitcast=True),
        mask=(r < rows_sf)[:, None] & kmask[None, :],
    )


# -------------------------------------------------------------------------- launcher and op


def _launch(x: torch.Tensor, gs: torch.Tensor, rb=None, ch=None, num_warps=None):
    K = x.shape[-1]
    if K % 16:
        raise ValueError(f"width {K} is not a multiple of 16")
    x2 = x.reshape(-1, K)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    rows = x2.shape[0]
    rows_sf = swizzled_sf_rows(rows)
    cfg = _launch_config()
    rb = rb or cfg[0]
    ch = ch or cfg[1]
    num_warps = num_warps or cfg[2]
    xq = torch.empty((rows, K // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty((sf_numel(rows, K),), dtype=torch.uint8, device=x.device)
    grid = (triton.cdiv(rows_sf, rb), triton.cdiv(K, ch))
    _h3_out_quant_kernel[grid](
        x2,
        xq,
        sf,
        gs,
        rows,
        rows_sf,
        K,
        x2.stride(0),
        xq.stride(0),
        swizzled_sf_ktiles(K),
        RB=rb,
        CH=ch,
        num_warps=num_warps,
    )
    return xq, sf


@torch.library.custom_op(
    f"{_NS}::minimax_h3_out_quant_fp4",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def h3_out_quant_fp4(x: torch.Tensor, gs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [..., K] bf16 -> NVFP4 with the static global scale ``gs``: (codes [..., K/2] u8, swizzled e4m3 scales)."""
    xq, sf = _launch(x, gs)
    return xq.view(*x.shape[:-1], x.shape[-1] // 2), sf


@h3_out_quant_fp4.register_fake
def _(x, gs):
    rows = x.numel() // x.shape[-1]
    return (
        x.new_empty((*x.shape[:-1], x.shape[-1] // 2), dtype=torch.uint8),
        x.new_empty((sf_numel(rows, x.shape[-1]),), dtype=torch.uint8),
    )


def static_nvfp4_input(linear) -> bool:
    """Whether ``linear`` consumes NVFP4 activations quantized with a static, calibrated input scale.

    The kernel reproduces ``fp4_quantize(x, input_scale, 16)`` exactly, so it applies only to the
    plain W4A4 method with the checkpoint's ``input_scale`` (no dynamic quantization, no AWQ
    ``pre_quant_scale``, 16-element scale blocks).
    """
    from tensorrt_llm._torch.modules.linear import NVFP4LinearMethod

    method = getattr(linear, "quant_method", None)
    return (
        type(method) is NVFP4LinearMethod
        and getattr(linear, "input_scale", None) is not None
        and not getattr(linear, "force_dynamic_quantization", False)
        and getattr(linear, "pre_quant_scale", None) is None
        and getattr(linear, "scaling_vector_size", 16) == 16
    )
