# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original exact norm tree with a precomputed RN reciprocal of frozen scale."""

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.visual_gen.models.minimax_h3.fused_adaln import _ld, _st


@triton.jit
def _st_fp8(ptr, off, D, factor, val, CH: tl.constexpr):
    c = off + tl.arange(0, CH)
    rounded = val.to(tl.bfloat16).to(tl.float32)
    code = tl.minimum(tl.maximum(rounded * factor, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(ptr + c, code, mask=c < D)


@triton.jit
def _adaln_fp8_kernel(
    x_ptr,
    a_ptr,
    x1_ptr,
    m_ptr,
    input_scale,
    w_ptr,
    mod_ptr,
    idx_ptr,
    D,
    x_stride,
    a_stride,
    x1_stride,
    m_stride,
    mod_stride,
    gate_off,
    scale_off,
    shift_off,
    eps,
    SEQ,
    HAS_RES: tl.constexpr,
    CH: tl.constexpr,
):
    row = tl.program_id(0)
    r64 = row.to(tl.int64)
    # idx has one entry per sequence position; rows are [batch * sequence].
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
        gb = mb + gate_off
        x0 = (x0 + _ld(gb, 0 * CH, D, CH) * _ld(ar, 0 * CH, D, CH)).to(tl.bfloat16).to(tl.float32)
        x1 = (x1 + _ld(gb, 1 * CH, D, CH) * _ld(ar, 1 * CH, D, CH)).to(tl.bfloat16).to(tl.float32)
        x2 = (x2 + _ld(gb, 2 * CH, D, CH) * _ld(ar, 2 * CH, D, CH)).to(tl.bfloat16).to(tl.float32)
        x3 = (x3 + _ld(gb, 3 * CH, D, CH) * _ld(ar, 3 * CH, D, CH)).to(tl.bfloat16).to(tl.float32)
        x4 = (x4 + _ld(gb, 4 * CH, D, CH) * _ld(ar, 4 * CH, D, CH)).to(tl.bfloat16).to(tl.float32)
        x5 = (x5 + _ld(gb, 5 * CH, D, CH) * _ld(ar, 5 * CH, D, CH)).to(tl.bfloat16).to(tl.float32)
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
    om = m_ptr + r64 * m_stride
    sb = mb + scale_off
    hb = mb + shift_off
    factor = tl.load(input_scale).to(tl.float32)
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
        _st_fp8(
            om,
            i * CH,
            D,
            factor,
            n * (1.0 + _ld(sb, i * CH, D, CH)) + _ld(hb, i * CH, D, CH),
            CH,
        )


def launch(x, a, w, mod, idx, scale, eps, gate_col, scale_col, shift_col):
    width = x.shape[-1]
    sequence = x.shape[-2]
    x2 = x.reshape(-1, width)
    has_res = a is not None
    a2 = a.reshape_as(x2) if has_res else x2
    if x2.stride(-1) != 1 or a2.stride(-1) != 1 or mod.stride(-1) != 1:
        raise ValueError("The FP8 AdaLN kernel needs unit-stride rows")
    idx = idx.contiguous()  # the kernel reads the indices at unit stride
    if idx.numel() != sequence:
        raise ValueError(
            f"The modulation index has {idx.numel()} entries for a sequence of {sequence}"
        )
    codes = torch.empty(x2.shape, dtype=torch.float8_e4m3fn, device=x.device)
    residual = torch.empty_like(x2) if has_res else x2
    _adaln_fp8_kernel[(x2.shape[0],)](
        x2,
        a2,
        residual,
        codes,
        scale,
        w,
        mod,
        idx,
        width,
        x2.stride(0),
        a2.stride(0),
        residual.stride(0),
        codes.stride(0),
        mod.stride(0),
        gate_col * width,
        scale_col * width,
        shift_col * width,
        eps,
        sequence,
        HAS_RES=has_res,
        CH=1024,
        num_warps=4,
    )
    return residual.view(x.shape), codes.view(x.shape)
