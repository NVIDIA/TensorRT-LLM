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
"""MiniMax-H3 AdaLN fusions: RMSNorm + per-token gathered shift/scale, optionally preceded by the gated residual.

    norm_mod:           m      = rmsnorm(x) * (1 + scale[idx]) + shift[idx]
    gate_res_norm_mod:  x1     = x + gate[idx] * a
                        m      = rmsnorm(x1) * (1 + scale[idx]) + shift[idx]        -> (x1, m)

``idx`` must hold one row index per sequence position, each below ``mod.shape[0]`` (the model validates
its layout once per request; the kernel does not bounds-check the gather).
``mod`` is the AdaLN projection output viewed as ``[n_t * 3, 6 * D]`` (one row per (timestep, modality) pair, the
six chunks shift/scale/gate for attention and MLP in order); ``idx`` maps each packed token to its row. The tables
are read in place (no per-token materialization). Rounding points follow the compiled reference: x1 is stored in
bf16 and normalized from those bf16 values, the RMSNorm result is rounded to bf16 (the reference norm is an opaque
custom op), the modulation is fp32 math with one final rounding.
"""

import torch
import triton
import triton.language as tl

_CH = 1024


@triton.jit
def _ld(ptr, off, D, CH: tl.constexpr):
    c = off + tl.arange(0, CH)
    return tl.load(ptr + c, mask=c < D, other=0.0).to(tl.float32)


@triton.jit
def _st(ptr, off, D, val, CH: tl.constexpr):
    c = off + tl.arange(0, CH)
    tl.store(ptr + c, val.to(tl.bfloat16), mask=c < D)


@triton.jit
def _minimax_h3_adaln_kernel(
    x_ptr,
    a_ptr,
    x1_ptr,
    m_ptr,
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
        _st(om, i * CH, D, n * (1.0 + _ld(sb, i * CH, D, CH)) + _ld(hb, i * CH, D, CH), CH)


def h3_adaln_supported(hidden_size: int, dtype: torch.dtype) -> bool:
    """BF16 on CUDA with a hidden size that fits the six register chunks."""
    return dtype == torch.bfloat16 and torch.cuda.is_available() and hidden_size <= 6 * _CH


def _launch(x, a, w, mod, idx, eps, gate_col, scale_col, shift_col):
    D = x.shape[-1]
    seq_len = x.shape[-2] if x.ndim >= 2 else x.shape[0]
    if idx.numel() != seq_len:
        raise ValueError(
            f"adaln_indices must have one entry per sequence position: {idx.numel()} vs {seq_len}"
        )
    x2 = x.reshape(-1, D)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    rows = x2.shape[0]
    m = torch.empty((rows, D), dtype=x.dtype, device=x.device)
    has_res = a is not None
    a2 = a.reshape(-1, D) if has_res else x2
    if has_res and a2.stride(-1) != 1:
        a2 = a2.contiguous()
    x1 = torch.empty((rows, D), dtype=x.dtype, device=x.device) if has_res else m
    if mod.stride(-1) != 1:
        mod = mod.contiguous()
    _minimax_h3_adaln_kernel[(rows,)](
        x2,
        a2,
        x1,
        m,
        w,
        mod,
        idx,
        D,
        x2.stride(0),
        a2.stride(0),
        x1.stride(0),
        m.stride(0),
        mod.stride(0),
        gate_col * D,
        scale_col * D,
        shift_col * D,
        eps,
        seq_len,
        HAS_RES=has_res,
        CH=_CH,
        num_warps=4,
    )
    return x1, m


@torch.library.custom_op(
    "trtllm::minimax_h3_norm_mod",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def h3_norm_mod(
    x: torch.Tensor,
    w: torch.Tensor,
    mod: torch.Tensor,
    idx: torch.Tensor,
    eps: float,
    scale_col: int,
    shift_col: int,
) -> torch.Tensor:
    """m = rmsnorm(x) * (1 + mod[idx, scale]) + mod[idx, shift]; x [..., D] bf16, mod [rows, 6*D], idx [prod(...)]."""
    return _launch(x, None, w, mod, idx, eps, 0, scale_col, shift_col)[1].view(x.shape)


@h3_norm_mod.register_fake
def _(x, w, mod, idx, eps, scale_col, shift_col):
    return torch.empty_like(x)


@torch.library.custom_op(
    "trtllm::minimax_h3_gate_res_norm_mod",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def h3_gate_res_norm_mod(
    x: torch.Tensor,
    a: torch.Tensor,
    w: torch.Tensor,
    mod: torch.Tensor,
    idx: torch.Tensor,
    eps: float,
    gate_col: int,
    scale_col: int,
    shift_col: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """x1 = x + mod[idx, gate] * a; m = rmsnorm(x1) * (1 + mod[idx, scale]) + mod[idx, shift]."""
    x1, m = _launch(x, a, w, mod, idx, eps, gate_col, scale_col, shift_col)
    return x1.view(x.shape), m.view(x.shape)


@h3_gate_res_norm_mod.register_fake
def _(x, a, w, mod, idx, eps, gate_col, scale_col, shift_col):
    return torch.empty_like(x), torch.empty_like(x)
