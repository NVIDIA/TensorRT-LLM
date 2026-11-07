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
"""MiniMax-H3 producers that emit NVFP4 activations for the static-NVFP4 Linears.

Three producers in the NVFP4 block write a BF16 tensor that the next Linear re-reads only to
quantize it (``quantize_with_block_size``): the AdaLN RMSNorm feeding ``qkv_proj``, the gated
residual + AdaLN RMSNorm feeding ``gate_up_proj`` (both ``fused_adaln.py``) and SiLU·up feeding
``down_proj`` (``silu_and_mul``). The kernels here keep the producer math of those kernels,
rounding point for rounding point, and quantize the BF16 result in registers:

    out, sf = fp4_quantize(producer(x).to(bf16), input_scale, 16, sf_use_ue8m0=False)

with the same recipe as ``cvt_warp_fp16_to_fp4`` (cpp/tensorrt_llm/kernels/quantization.cuh):
block amax over 16 columns, ``sf = e4m3(gs * (amax * rcp(6)))`` with ``rcp.approx.ftz``,
``scale = rcp(float(sf) * rcp(gs))``, codes ``cvt.rn.satfinite.e2m1x2.f32`` (low nibble = even
column), scales in the swizzled 128x4 layout (``get_sf_out_offset_128x4``) with zeroed padding
rows. Every rounding point goes through PTX so the compiler cannot fold it. The consumer Linear
takes the pair as ``Fp4QuantizedTensor`` and applies its static ``alpha``.

Access pattern: one row per program, 1024-column register chunks loaded contiguously (the even /
odd split happens in registers), packed codes stored 8 per 32-bit word.
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl

# "trtllm" inside the package; the study tree imports this file as h3_fused_quant and registers
# the same ops under "h3study" so both can coexist in one process.
_NS = "h3study" if __name__.startswith("h3_") else "trtllm"
_ENV = os.environ.get("H3_FUSED_QUANT", "1")
FUSED_QUANT_ENABLED = _ENV not in ("0", "off", "false")
NUM_WARPS = int(os.environ.get("H3_FUSED_QUANT_WARPS", "4"))

_CH = 1024
CH = tl.constexpr(_CH)
KB = tl.constexpr(_CH // 16)


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
def _round_bf16(x):
    """f32 -> bf16 (round to nearest even) -> f32, through PTX so it is never folded away."""
    h = tl.inline_asm_elementwise(
        "cvt.rn.bf16.f32 $0, $1;", "=h,f", [x], dtype=tl.int16, is_pure=True, pack=1
    )
    return (h.to(tl.int32) << 16).to(tl.float32, bitcast=True)


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


# -------------------------------------------------------------------------- chunk helpers


@triton.jit
def _ld(ptr, off, D, CH: tl.constexpr):
    c = off + tl.arange(0, CH)
    return tl.load(ptr + c, mask=c < D, other=0.0).to(tl.float32)


@triton.jit
def _st(ptr, off, D, val, CH: tl.constexpr):
    c = off + tl.arange(0, CH)
    tl.store(ptr + c, val.to(tl.bfloat16), mask=c < D)


@triton.jit
def _sf_row_base(r64, SF_KTILES):
    """Swizzled 128x4 layout: [mTile, kTile, m % 32, (m % 128) // 32, k % 4]; k-tile term added per block."""
    kt = SF_KTILES.to(tl.int64)
    return (r64 // 128) * (kt * 512) + (r64 % 32) * 16 + ((r64 % 128) // 32) * 4


@triton.jit
def _quant_chunk(m, col0, D, gs, rcp_gs, rcp6, xq_row, sf_ptr, sf_base):
    """Quantize the bf16-valued f32 chunk ``m`` (columns [col0, col0 + CH), zero beyond D) and store it."""
    t = tl.reshape(m, [KB, 16])
    amax = tl.max(tl.abs(t), axis=1)  # [KB] exact (bf16 values)
    sfv = gs * (amax * rcp6)
    sf_b = _e4m3_encode(sfv)  # [KB] int8
    sf_f = sf_b.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    oscale = tl.where(amax != 0.0, _rcp_approx(sf_f * rcp_gs), 0.0)
    scaled = t * oscale[:, None]
    lo, hi = tl.split(tl.reshape(scaled, [KB, 8, 2]))
    packed = _e2m1_pack(lo, hi)  # [KB, 8] int8
    kk = tl.arange(0, KB)
    kidx = col0 // 16 + kk  # [KB] block index along K
    kmask = kidx < D // 16
    qoff = col0 // 2 + kk[:, None] * 8 + tl.arange(0, 8)[None, :]
    tl.store(xq_row + qoff, packed.to(tl.uint8, bitcast=True), mask=kmask[:, None])
    soff = sf_base + (kidx // 4).to(tl.int64) * 512 + (kidx % 4)
    tl.store(sf_ptr + soff, sf_b.to(tl.uint8, bitcast=True), mask=kmask)


@triton.jit
def _zero_sf_chunk(col0, D, sf_ptr, sf_base):
    kk = tl.arange(0, KB)
    kidx = col0 // 16 + kk
    soff = sf_base + (kidx // 4).to(tl.int64) * 512 + (kidx % 4)
    tl.store(sf_ptr + soff, tl.zeros([KB], dtype=tl.uint8), mask=kidx < D // 16)


# -------------------------------------------------------------------------- kernels


@triton.jit
def _minimax_h3_adaln_fp4_kernel(
    x_ptr,
    a_ptr,
    x1_ptr,
    xq_ptr,
    sf_ptr,
    w_ptr,
    mod_ptr,
    gmod_ptr,
    idx_ptr,
    gs_ptr,
    rows,
    D,
    x_stride,
    a_stride,
    x1_stride,
    xq_stride,
    mod_stride,
    gmod_stride,
    gate_off,
    scale_off,
    shift_off,
    eps,
    SEQ,
    SF_KTILES,
    HAS_RES: tl.constexpr,
    CH: tl.constexpr,
):
    """fused_adaln._minimax_h3_adaln_kernel with the modulated output quantized to NVFP4.

    The gate of the residual is read from ``gmod`` (the same table as ``mod``, or the previous
    block's table when a block folds its MLP residual into the next block's norm). The grid
    covers the 128-padded scale rows; programs past ``rows`` only zero their scales.
    """
    row = tl.program_id(0)
    r64 = row.to(tl.int64)
    sf_base = _sf_row_base(r64, SF_KTILES)
    if row >= rows:
        for i in tl.static_range(6):
            _zero_sf_chunk(i * CH, D, sf_ptr, sf_base)
    else:
        mrow = tl.load(idx_ptr + row % SEQ).to(tl.int64)
        mb = mod_ptr + mrow * mod_stride
        xr = x_ptr + r64 * x_stride
        x0 = _ld(xr, 0 * CH, D, CH)
        x1 = _ld(xr, 1 * CH, D, CH)
        x2 = _ld(xr, 2 * CH, D, CH)
        x3 = _ld(xr, 3 * CH, D, CH)
        x4 = _ld(xr, 4 * CH, D, CH)
        x5 = _ld(xr, 5 * CH, D, CH)
        if HAS_RES:
            ar = a_ptr + r64 * a_stride
            gb = gmod_ptr + mrow * gmod_stride + gate_off
            x0 = (
                (x0 + _ld(gb, 0 * CH, D, CH) * _ld(ar, 0 * CH, D, CH))
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            x1 = (
                (x1 + _ld(gb, 1 * CH, D, CH) * _ld(ar, 1 * CH, D, CH))
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            x2 = (
                (x2 + _ld(gb, 2 * CH, D, CH) * _ld(ar, 2 * CH, D, CH))
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            x3 = (
                (x3 + _ld(gb, 3 * CH, D, CH) * _ld(ar, 3 * CH, D, CH))
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            x4 = (
                (x4 + _ld(gb, 4 * CH, D, CH) * _ld(ar, 4 * CH, D, CH))
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            x5 = (
                (x5 + _ld(gb, 5 * CH, D, CH) * _ld(ar, 5 * CH, D, CH))
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            o1 = x1_ptr + r64 * x1_stride
            _st(o1, 0 * CH, D, x0, CH)
            _st(o1, 1 * CH, D, x1, CH)
            _st(o1, 2 * CH, D, x2, CH)
            _st(o1, 3 * CH, D, x3, CH)
            _st(o1, 4 * CH, D, x4, CH)
            _st(o1, 5 * CH, D, x5, CH)
        ss = (
            tl.sum(x0 * x0, 0)
            + tl.sum(x1 * x1, 0)
            + tl.sum(x2 * x2, 0)
            + tl.sum(x3 * x3, 0)
            + tl.sum(x4 * x4, 0)
            + tl.sum(x5 * x5, 0)
        )
        r = tl.math.rsqrt(ss / D + eps)
        gs = tl.load(gs_ptr).to(tl.float32)
        rcp_gs = _rcp_approx(gs)
        rcp6 = _rcp_approx(gs * 0.0 + 6.0)
        xq_row = xq_ptr + r64 * xq_stride
        sb = mb + scale_off
        hb = mb + shift_off
        for i in tl.static_range(6):
            if i == 0:
                xi = x0
            elif i == 1:
                xi = x1
            elif i == 2:
                xi = x2
            elif i == 3:
                xi = x3
            elif i == 4:
                xi = x4
            else:
                xi = x5
            n = (xi * r * _ld(w_ptr, i * CH, D, CH)).to(tl.bfloat16).to(tl.float32)
            m = _round_bf16(n * (1.0 + _ld(sb, i * CH, D, CH)) + _ld(hb, i * CH, D, CH))
            _quant_chunk(m, i * CH, D, gs, rcp_gs, rcp6, xq_row, sf_ptr, sf_base)


@triton.jit
def _minimax_h3_swiglu_fp4_kernel(
    h_ptr,
    xq_ptr,
    sf_ptr,
    gs_ptr,
    rows,
    inter,
    h_stride,
    xq_stride,
    SF_KTILES,
    swiglu_alpha: tl.constexpr,
    swiglu_beta: tl.constexpr,
    CH: tl.constexpr,
):
    """silu_and_mul_kernel (modules/swiglu.py) on h = [gate | up] with the result quantized to NVFP4."""
    row = tl.program_id(0)
    col0 = tl.program_id(1) * CH
    r64 = row.to(tl.int64)
    sf_base = _sf_row_base(r64, SF_KTILES)
    if row >= rows:
        _zero_sf_chunk(col0, inter, sf_ptr, sf_base)
    else:
        hr = h_ptr + r64 * h_stride
        a = _ld(hr, col0, inter, CH)
        b = _ld(hr + inter, col0, inter, CH)
        result = a * tl.sigmoid(swiglu_alpha * a) * (b + swiglu_beta)
        gs = tl.load(gs_ptr).to(tl.float32)
        rcp_gs = _rcp_approx(gs)
        rcp6 = _rcp_approx(gs * 0.0 + 6.0)
        _quant_chunk(
            _round_bf16(result),
            col0,
            inter,
            gs,
            rcp_gs,
            rcp6,
            xq_ptr + r64 * xq_stride,
            sf_ptr,
            sf_base,
        )


# -------------------------------------------------------------------------- launchers


def _alloc(rows, k, device):
    xq = torch.empty((rows, k // 2), dtype=torch.uint8, device=device)
    sf = torch.empty((sf_numel(rows, k),), dtype=torch.uint8, device=device)
    return xq, sf


def _launch_adaln(
    x, a, w, mod, idx, gs, eps, gate_col, scale_col, shift_col, num_warps=None, gate_mod=None
):
    D = x.shape[-1]
    if D > 6 * _CH or D % 16:
        raise ValueError(f"hidden size {D} is not supported by the fused NVFP4 AdaLN kernel")
    seq_len = x.shape[-2] if x.ndim >= 2 else x.shape[0]
    if idx.numel() != seq_len:
        raise ValueError(
            f"adaln_indices must have one entry per sequence position: {idx.numel()} vs {seq_len}"
        )
    x2 = x.reshape(-1, D)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    rows = x2.shape[0]
    has_res = a is not None
    a2 = a.reshape(-1, D) if has_res else x2
    if has_res and a2.stride(-1) != 1:
        a2 = a2.contiguous()
    x1 = torch.empty((rows, D), dtype=x.dtype, device=x.device) if has_res else x2
    if mod.stride(-1) != 1:
        mod = mod.contiguous()
    gmod = mod if gate_mod is None else gate_mod
    if gmod.stride(-1) != 1:
        gmod = gmod.contiguous()
    xq, sf = _alloc(rows, D, x.device)
    _minimax_h3_adaln_fp4_kernel[(swizzled_sf_rows(rows),)](
        x2,
        a2,
        x1,
        xq,
        sf,
        w,
        mod,
        gmod,
        idx,
        gs,
        rows,
        D,
        x2.stride(0),
        a2.stride(0),
        x1.stride(0),
        xq.stride(0),
        mod.stride(0),
        gmod.stride(0),
        gate_col * D,
        scale_col * D,
        shift_col * D,
        eps,
        seq_len,
        swizzled_sf_ktiles(D),
        HAS_RES=has_res,
        CH=_CH,
        num_warps=num_warps or NUM_WARPS,
    )
    return x1, xq, sf


def _launch_swiglu(h, gs, num_warps=None):
    inter = h.shape[-1] // 2
    if inter % 16:
        raise ValueError(f"intermediate size {inter} is not a multiple of 16")
    h2 = h.reshape(-1, 2 * inter)
    if h2.stride(-1) != 1:
        h2 = h2.contiguous()
    rows = h2.shape[0]
    xq, sf = _alloc(rows, inter, h.device)
    _minimax_h3_swiglu_fp4_kernel[(swizzled_sf_rows(rows), triton.cdiv(inter, _CH))](
        h2,
        xq,
        sf,
        gs,
        rows,
        inter,
        h2.stride(0),
        xq.stride(0),
        swizzled_sf_ktiles(inter),
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        CH=_CH,
        num_warps=num_warps or NUM_WARPS,
    )
    return xq, sf


# -------------------------------------------------------------------------- custom ops


@torch.library.custom_op(
    f"{_NS}::minimax_h3_norm_mod_fp4",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def h3_norm_mod_fp4(
    x: torch.Tensor,
    w: torch.Tensor,
    mod: torch.Tensor,
    idx: torch.Tensor,
    gs: torch.Tensor,
    eps: float,
    scale_col: int,
    shift_col: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 of rmsnorm(x) * (1 + mod[idx, scale]) + mod[idx, shift]: (codes [..., D/2] u8, swizzled e4m3 scales)."""
    _, xq, sf = _launch_adaln(x, None, w, mod, idx, gs, eps, 0, scale_col, shift_col)
    return xq.view(*x.shape[:-1], x.shape[-1] // 2), sf


@h3_norm_mod_fp4.register_fake
def _(x, w, mod, idx, gs, eps, scale_col, shift_col):
    rows = x.numel() // x.shape[-1]
    return (
        x.new_empty((*x.shape[:-1], x.shape[-1] // 2), dtype=torch.uint8),
        x.new_empty((sf_numel(rows, x.shape[-1]),), dtype=torch.uint8),
    )


@torch.library.custom_op(
    f"{_NS}::minimax_h3_gate_res_norm_mod_fp4",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def h3_gate_res_norm_mod_fp4(
    x: torch.Tensor,
    a: torch.Tensor,
    w: torch.Tensor,
    mod: torch.Tensor,
    idx: torch.Tensor,
    gs: torch.Tensor,
    eps: float,
    gate_col: int,
    scale_col: int,
    shift_col: int,
    gate_mod: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x1 = x + gate_mod[idx, gate] * a (bf16); NVFP4 of rmsnorm(x1) * (1 + mod[idx, scale]) + mod[idx, shift].

    ``gate_mod`` defaults to ``mod``; a block that folds its MLP residual into the next block's
    norm passes its own table here while ``mod`` is the next block's.
    """
    x1, xq, sf = _launch_adaln(
        x, a, w, mod, idx, gs, eps, gate_col, scale_col, shift_col, gate_mod=gate_mod
    )
    return x1.view(x.shape), xq.view(*x.shape[:-1], x.shape[-1] // 2), sf


@h3_gate_res_norm_mod_fp4.register_fake
def _(x, a, w, mod, idx, gs, eps, gate_col, scale_col, shift_col, gate_mod=None):
    rows = x.numel() // x.shape[-1]
    return (
        torch.empty_like(x),
        x.new_empty((*x.shape[:-1], x.shape[-1] // 2), dtype=torch.uint8),
        x.new_empty((sf_numel(rows, x.shape[-1]),), dtype=torch.uint8),
    )


@torch.library.custom_op(
    f"{_NS}::minimax_h3_swiglu_fp4",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def h3_swiglu_fp4(h: torch.Tensor, gs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """h [..., 2I] bf16 = [gate | up] -> NVFP4 of silu(gate) * up: (codes [..., inter/2] u8, swizzled e4m3 scales)."""
    inter = h.shape[-1] // 2
    xq, sf = _launch_swiglu(h, gs)
    return xq.view(*h.shape[:-1], inter // 2), sf


@h3_swiglu_fp4.register_fake
def _(h, gs):
    inter = h.shape[-1] // 2
    rows = h.numel() // h.shape[-1]
    return (
        h.new_empty((*h.shape[:-1], inter // 2), dtype=torch.uint8),
        h.new_empty((sf_numel(rows, inter),), dtype=torch.uint8),
    )


def static_nvfp4_input(linear) -> bool:
    """Whether ``linear`` consumes NVFP4 activations quantized with a static, calibrated input scale.

    The fused producers reproduce ``fp4_quantize(x, input_scale, 16)`` exactly, so they apply only
    to the plain W4A4 method with the checkpoint's ``input_scale`` (no dynamic quantization, no
    AWQ ``pre_quant_scale``, 16-element scale blocks).
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
