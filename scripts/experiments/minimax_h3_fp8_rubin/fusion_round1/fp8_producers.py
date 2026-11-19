# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static FP8 producer fusions preserving the existing BF16 rounding boundaries."""

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.swiglu import silu_and_mul_kernel
from tensorrt_llm._torch.visual_gen.models.minimax_h3 import fused_adaln as original
from tensorrt_llm._torch.visual_gen.models.minimax_h3.fused_adaln import _ld, _st

STATE = {"adaln": False, "swiglu": False, "tile": 4096, "warps": 4}


@triton.jit
def _st_fp8(ptr, off, D, scale, val, CH: tl.constexpr):
    c = off + tl.arange(0, CH)
    rounded = val.to(tl.bfloat16).to(tl.float32)
    factor = tl.div_rn(1.0, tl.load(scale).to(tl.float32))
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
            input_scale,
            n * (1.0 + _ld(sb, i * CH, D, CH)) + _ld(hb, i * CH, D, CH),
            CH,
        )


def _launch(x, a, w, mod, idx, scale, eps, gate_col, scale_col, shift_col):
    width = x.shape[-1]
    sequence = x.shape[-2]
    x2 = x.reshape(-1, width)
    has_res = a is not None
    a2 = a.reshape_as(x2) if has_res else x2
    assert x2.stride(-1) == 1 and a2.stride(-1) == 1 and mod.stride(-1) == 1
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


@torch.library.custom_op("h3_fp8_round1::norm", mutates_args=())
def norm(
    x: torch.Tensor,
    w: torch.Tensor,
    mod: torch.Tensor,
    idx: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if STATE["adaln"]:
        return _launch(x, None, w, mod, idx, scale, eps, 0, 1, 0)[1]
    value = original.h3_norm_mod(x, w, mod, idx, eps, 1, 0)
    return torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(value, scale)[0]


@norm.register_fake
def _(x, w, mod, idx, scale, eps):
    return torch.empty_like(x, dtype=torch.float8_e4m3fn)


@torch.library.custom_op("h3_fp8_round1::gate_norm", mutates_args=())
def gate_norm(
    x: torch.Tensor,
    a: torch.Tensor,
    w: torch.Tensor,
    mod: torch.Tensor,
    idx: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if STATE["adaln"]:
        return _launch(x, a, w, mod, idx, scale, eps, 2, 4, 3)
    residual, value = original.h3_gate_res_norm_mod(x, a, w, mod, idx, eps, 2, 4, 3)
    return residual, torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(value, scale)[0]


@gate_norm.register_fake
def _(x, a, w, mod, idx, scale, eps):
    return torch.empty_like(x), torch.empty_like(x, dtype=torch.float8_e4m3fn)


@torch.library.custom_op("h3_fp8_round1::swiglu", mutates_args=())
def swiglu(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if not STATE["swiglu"]:
        return torch.ops.trtllm.silu_and_mul(x, scale=scale, dtype=torch.float8_e4m3fn)
    rows, twice_width = x.shape
    width = twice_width // 2
    output = torch.empty((rows, width), dtype=torch.float8_e4m3fn, device=x.device)
    silu_and_mul_kernel[(rows, triton.cdiv(width, STATE["tile"]))](
        output,
        output.stride(0),
        scale,
        x,
        x.stride(0),
        width,
        swiglu_limit=0.0,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        BLOCK_SIZE=STATE["tile"],
        HAS_O_SCALE=True,
        HAS_SWIGLU_LIMIT=False,
        num_warps=STATE["warps"],
    )
    return output


@swiglu.register_fake
def _(x, scale):
    return torch.empty((x.shape[0], x.shape[1] // 2), dtype=torch.float8_e4m3fn, device=x.device)
