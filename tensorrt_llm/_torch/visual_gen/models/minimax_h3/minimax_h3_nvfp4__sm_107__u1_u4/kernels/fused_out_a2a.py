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
"""Ulysses output exchange of the MiniMax-H3 attention output as NVFP4 codes.

Under Ulysses each rank holds the attention output of the full sequence for its ``H / U`` heads,
``[1, S, H / U, D]``, and the output all-to-all turns it into ``[1, S / U, H, D]`` for the
``to_out`` Linear, which (static NVFP4) immediately quantizes it with ``fused_out_quant``. The
NVFP4 scale blocks are 16 consecutive elements of the ``H * D`` row and a head is 128 wide, so a
block never straddles heads: quantizing a rank's head slice before the exchange gives, element
for element, the same codes and the same block scales as quantizing the gathered row afterwards.
This module therefore quantizes first, moves ``0.5 + 1/16`` bytes per element instead of 2, and
assembles the received slices into the layout the Linear expects (packed codes ``[S / U, H * D / 2]``
and the swizzled 128x4 scale layout). The rounding points are those of ``fused_out_quant``.

Buffers: the send buffer is ``[S, K_L / 2 + K_L / 16]`` uint8 with ``K_L = H * D / U``; row ``t``
holds its codes then its ``K_L / 16`` linear e4m3 scales. Chunk ``u`` (rows ``[u * S / U, (u + 1) * S / U)``)
is what rank ``u`` receives, so the buffer is already in exchange order and one
``all_to_all_single`` moves it. The receive buffer, indexed by source rank, holds for each local
token the slice of heads rank ``u`` computed; the assemble kernel writes it into column block ``u``
of the gathered code row and scatters the scales into the swizzled layout.

``out_a2a_fp4=True`` (default) enables this path; it is effective only on the packed FP8
exchange path (``packed_a2a=True``) with a static-NVFP4 ``to_out``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from ..config import SCENARIO
from .fused_out_quant import (
    _e2m1_pack,
    _e4m3_encode,
    _rcp_approx,
    _sf_row_base,
    sf_numel,
    swizzled_sf_ktiles,
    swizzled_sf_rows,
)

OUT_A2A_FP4_ENABLED = SCENARIO.out_a2a_fp4


@triton.jit
def _h3_out_quant_send_kernel(
    x_ptr,
    send_ptr,
    gs_ptr,
    rows,
    K,
    x_stride,
    SEND_STRIDE,
    RB: tl.constexpr,
    CH: tl.constexpr,
):
    """Quantize the tile [row0, row0 + RB) x [col0, col0 + CH) of the BF16 [rows, K] head slice to
    NVFP4: packed codes at send[r, col / 2], linear e4m3 scales at send[r, K / 2 + col / 16].
    Same arithmetic as fused_out_quant._h3_out_quant_kernel."""
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
    amax = tl.max(tl.abs(t), axis=2)
    gs = tl.load(gs_ptr).to(tl.float32)
    rcp_gs = _rcp_approx(gs)
    rcp6 = _rcp_approx(gs * 0.0 + 6.0)
    sfv = gs * (amax * rcp6)
    sf_b = _e4m3_encode(sfv)
    sf_f = sf_b.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    oscale = tl.where(amax != 0.0, _rcp_approx(sf_f * rcp_gs), 0.0)
    scaled = t * oscale[:, :, None]
    lo, hi = tl.split(tl.reshape(scaled, [RB, KB, 8, 2]))
    packed = _e2m1_pack(lo, hi)
    kk = tl.arange(0, KB)
    kidx = col0 // 16 + kk
    kmask = kidx < K // 16
    qoff = (
        r64[:, None, None] * SEND_STRIDE
        + col0 // 2
        + kk[None, :, None] * 8
        + tl.arange(0, 8)[None, None, :]
    )
    tl.store(
        send_ptr + qoff,
        packed.to(tl.uint8, bitcast=True),
        mask=valid[:, None, None] & kmask[None, :, None],
    )
    soff = r64[:, None] * SEND_STRIDE + K // 2 + kidx[None, :]
    tl.store(
        send_ptr + soff,
        sf_b.to(tl.uint8, bitcast=True),
        mask=valid[:, None] & kmask[None, :],
    )


@triton.jit
def _h3_out_assemble_kernel(
    recv_ptr,
    xq_ptr,
    sf_ptr,
    s_local,
    rows_sf,
    RECV_STRIDE,
    SF_KTILES,
    U: tl.constexpr,
    K_LOCAL: tl.constexpr,
    RB: tl.constexpr,
    CB: tl.constexpr,
    SFB: tl.constexpr,
):
    """Program (row tile, source rank u): copy u's code slice of RB local tokens into column block u
    of the gathered [s_local, U * K_LOCAL / 2] code matrix and scatter its K_LOCAL / 16 scales into
    the swizzled 128x4 layout for K = U * K_LOCAL; rows in [s_local, rows_sf) get zero scales."""
    QW: tl.constexpr = K_LOCAL // 2
    SW: tl.constexpr = K_LOCAL // 16
    pid_r = tl.program_id(0)
    u = tl.program_id(1)
    r = pid_r * RB + tl.arange(0, RB)
    r64 = r.to(tl.int64)
    valid = r < s_local
    src_row = (u.to(tl.int64) * s_local + r64) * RECV_STRIDE
    for c0 in range(0, QW, CB):
        c = c0 + tl.arange(0, CB)
        m = valid[:, None] & (c < QW)[None, :]
        vals = tl.load(recv_ptr + src_row[:, None] + c[None, :], mask=m, other=0)
        tl.store(xq_ptr + r64[:, None] * (U * QW) + u * QW + c[None, :], vals, mask=m)
    j = tl.arange(0, SFB)
    jm = j < SW
    sfv = tl.load(
        recv_ptr + src_row[:, None] + QW + j[None, :], mask=valid[:, None] & jm[None, :], other=0
    )
    kidx = u * SW + j
    soff = (
        _sf_row_base(r64, SF_KTILES)[:, None]
        + (kidx // 4).to(tl.int64)[None, :] * 512
        + (kidx % 4)[None, :]
    )
    tl.store(sf_ptr + soff, sfv, mask=(r < rows_sf)[:, None] & jm[None, :])


def _pow2_at_least(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def quant_to_send(
    x: torch.Tensor, gs: torch.Tensor, rb: int = 16, ch: int = 512, num_warps: int = 4
) -> torch.Tensor:
    """BF16 ``[S, K_L]`` head slice -> send buffer ``[S, K_L / 2 + K_L / 16]`` uint8."""
    if x.dim() != 2 or x.stride(-1) != 1:
        raise ValueError("quant_to_send expects a 2-D row-major tensor")
    rows, K = x.shape
    if K % 16:
        raise ValueError(f"width {K} is not a multiple of 16")
    send_stride = K // 2 + K // 16
    send = torch.empty((rows, send_stride), dtype=torch.uint8, device=x.device)
    grid = (triton.cdiv(rows, rb), triton.cdiv(K, ch))
    _h3_out_quant_send_kernel[grid](
        x, send, gs, rows, K, x.stride(0), send_stride, RB=rb, CH=ch, num_warps=num_warps
    )
    return send


def assemble_from_recv(
    recv: torch.Tensor, ulysses: int, k_local: int, rb: int = 16, num_warps: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Receive buffer ``[U * S_local, K_L / 2 + K_L / 16]`` -> (codes ``[S_local, U * K_L / 2]``,
    swizzled scales as ``fused_out_quant`` lays them out for ``[S_local, U * K_L]``)."""
    total, recv_stride = recv.shape
    s_local = total // ulysses
    K = ulysses * k_local
    rows_sf = swizzled_sf_rows(s_local)
    codes = torch.empty((s_local, K // 2), dtype=torch.uint8, device=recv.device)
    sf = torch.empty((sf_numel(s_local, K),), dtype=torch.uint8, device=recv.device)
    grid = (triton.cdiv(rows_sf, rb), ulysses)
    _h3_out_assemble_kernel[grid](
        recv,
        codes,
        sf,
        s_local,
        rows_sf,
        recv_stride,
        swizzled_sf_ktiles(K),
        U=ulysses,
        K_LOCAL=k_local,
        RB=rb,
        CB=min(_pow2_at_least(k_local // 2), 1024),
        SFB=_pow2_at_least(k_local // 16),
        num_warps=num_warps,
    )
    return codes, sf


@torch.compiler.disable
def h3_out_a2a_fp4(
    out: torch.Tensor, gs: torch.Tensor, process_group
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention output ``[1, S, H / U, D]`` (this rank's heads, full sequence) -> the ``to_out``
    input of this rank's ``S / U`` tokens as NVFP4: (codes ``[1, S / U, H * D / 2]``, swizzled
    scales), exchanged as codes + scales instead of BF16."""
    if out.dim() != 4 or out.shape[0] != 1:
        raise ValueError(f"expected [1, S, H/U, D]; got {tuple(out.shape)}")
    ulysses = torch.distributed.get_world_size(group=process_group)
    _, s_full, h_local, d = out.shape
    if s_full % ulysses:
        raise ValueError(f"sequence {s_full} not divisible by U={ulysses}")
    k_local = h_local * d
    x2 = out.reshape(s_full, k_local)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    send = quant_to_send(x2, gs)
    recv = torch.empty_like(send)
    torch.distributed.all_to_all_single(recv.view(-1), send.view(-1), group=process_group)
    codes, sf = assemble_from_recv(recv, ulysses, k_local)
    return codes.view(1, s_full // ulysses, ulysses * k_local // 2), sf
