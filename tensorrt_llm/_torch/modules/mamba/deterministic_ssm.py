# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed Mamba2 recurrence with the same FP32 arithmetic at every step."""

import torch
import triton
import triton.language as tl

from .softplus import softplus


@triton.jit
def _ssm_scan(
    XBC,
    DT,
    A,
    D,
    DT_BIAS,
    STATE,
    CU,
    INDICES,
    HAS_INITIAL,
    OUT,
    SX: tl.constexpr,
    SDT: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    SS2: tl.constexpr,
    SS3: tl.constexpr,
    HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
    DIM: tl.constexpr,
    N: tl.constexpr,
    SOFTPLUS: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr,
):
    seq, head = tl.program_id(0), tl.program_id(1)
    dims = tl.program_id(2) * BM + tl.arange(0, BM)
    ns = tl.arange(0, BN)
    slot = tl.load(INDICES + seq).to(tl.int64)
    start = tl.load(CU + seq).to(tl.int64)
    end = tl.load(CU + seq + 1).to(tl.int64)
    initial = tl.load(HAS_INITIAL + seq)
    ptrs = STATE + slot * SS0 + head * SS1 + dims[:, None] * SS2 + ns[None, :] * SS3
    mask = (dims[:, None] < DIM) & (ns[None, :] < N) & (slot >= 0)
    state = tl.load(ptrs, mask & initial, other=0).to(tl.float32)
    decay = tl.load(A + head).to(tl.float32)
    skip = tl.load(D + head).to(tl.float32)
    bias = tl.load(DT_BIAS + head).to(tl.float32)
    group = head // (HEADS // GROUPS)
    for row in range(start, end):
        x = tl.load(XBC + row * SX + head * DIM + dims, dims < DIM, other=0).to(tl.float32)
        b = tl.load(XBC + row * SX + HEADS * DIM + group * N + ns, ns < N, other=0).to(tl.float32)
        c = tl.load(XBC + row * SX + HEADS * DIM + GROUPS * N + group * N + ns, ns < N, other=0).to(
            tl.float32
        )
        dt = tl.load(DT + row * SDT + head).to(tl.float32) + bias
        if SOFTPLUS:
            dt = softplus(dt)
        state = state * tl.exp(dt * decay) + (b * dt)[None, :] * x[:, None]
        output = tl.sum(state * c[None, :], 1) + x * skip
        output = tl.where(slot >= 0, output, 0)
        tl.store(OUT + row * HEADS * DIM + head * DIM + dims, output, dims < DIM)
    tl.store(ptrs, state, mask)


def deterministic_ssm(
    xbc: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    d: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    indices: torch.Tensor,
    has_initial: torch.Tensor,
    out: torch.Tensor,
    groups: int,
    dt_softplus: bool,
) -> None:
    """Consume packed sequences and update only their indexed FP32 state rows."""
    _, heads, dim, n = state.shape
    _ssm_scan[(indices.numel(), heads, triton.cdiv(dim, 16))](
        xbc,
        dt,
        a,
        d,
        dt_bias,
        state,
        cu_seqlens,
        indices,
        has_initial,
        out,
        xbc.stride(0),
        dt.stride(0),
        *state.stride(),
        heads,
        groups,
        dim,
        n,
        dt_softplus,
        triton.next_power_of_2(n),
        16,
        num_warps=4,
    )
