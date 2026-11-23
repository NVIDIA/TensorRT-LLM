# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton grouped MoE for Inkling 2's MXFP8 experts: one launch per GEMM for all local experts.

The routes `(token, slot)` that land on this rank's experts are grouped by expert into blocks of
`BLOCK_M` rows (`align_routes`, on the device, without a host synchronization; the block count is
bounded by `T * top_k / BLOCK_M + E`). Then:

* `gemm1`: per block and 128 intermediate columns, `x^ @ gate^T` and `x^ @ up^T` (gate rows `2j`,
  up rows `2j + 1` of `w13`), with the E4M3 weights and their UE8M0 scales expanded to bf16 in the
  kernel (exactly: an E4M3 value times a power of two is a bf16 value), fp32 accumulation, then
  SiTU and the MXFP8 rounding of the result per 32 columns; writes `h^` `[routes, I]` bf16.
* `gemm2`: per block and 128 hidden columns, `h^ @ down^T` in fp32, times the route's weight,
  added atomically into the fp32 output row of its token.

`x^`, the MXFP8-rounded input, is computed once per call by the caller.

Both activation roundings, `x^` (the caller's `mxfp8_quantize`) and `h^`, take the block scale
`2^ceil(log2(amax / 448))`, as SGLang's Inkling 2 MoE does (`_mxfp8_block` in
`sglang/kernels/ops/moe/inkling2_moe.py`), so that no value of a block saturates. The OCP MX rule
`2^(floor(log2(amax)) - 8)` would instead clip a block whose `amax` lies in
`(448 * 2^k, 512 * 2^k)`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK_M = 32
#: The K tile of both GEMMs: the hidden and intermediate sizes must be multiples of it.
K_TILE = 128


def align_routes(
    topk_ids: torch.Tensor, local_expert_offset: int, num_local: int, block_m: int = BLOCK_M
):
    """Group this rank's routes by expert into blocks of `block_m` rows, on the device and without
    a host synchronization.

    Returns `route_of_row` `[max_blocks * block_m]` int32 (the flat route `token * top_k + slot` of
    each row, -1 for padding), `expert_of_block` `[max_blocks]` int32 (-1 past the last used block)
    and `max_blocks`, a bound on the blocks any routing needs.
    """
    num_tokens, top_k = topk_ids.shape
    device = topk_ids.device
    flat = topk_ids.reshape(-1).long() - local_expert_offset
    is_local = (flat >= 0) & (flat < num_local)
    expert = torch.where(
        is_local, flat, torch.full_like(flat, num_local)
    )  # non-local routes sort last
    counts = torch.zeros(num_local + 1, dtype=torch.long, device=device)
    counts.scatter_add_(0, expert, torch.ones_like(expert))
    counts = counts[:num_local]
    blocks_per_expert = (counts + block_m - 1) // block_m
    block_end = torch.cumsum(blocks_per_expert, 0)
    block_start = block_end - blocks_per_expert
    max_blocks = num_tokens * top_k // block_m + num_local
    order = torch.argsort(expert, stable=True)
    sorted_expert = expert[order]
    sorted_local = sorted_expert < num_local
    e = sorted_expert.clamp(max=num_local - 1)
    rank_in_expert = (
        torch.arange(order.numel(), device=device) - (torch.cumsum(counts, 0) - counts)[e]
    )
    rows = max_blocks * block_m
    # Non-local routes write to one spare slot past the end.
    row = torch.where(
        sorted_local, block_start[e] * block_m + rank_in_expert, torch.full_like(e, rows)
    )
    route_of_row = torch.full((rows + 1,), -1, dtype=torch.int32, device=device)
    route_of_row.scatter_(0, row, order.to(torch.int32))
    owner = torch.searchsorted(block_end, torch.arange(max_blocks, device=device), right=True)
    expert_of_block = torch.where(owner < num_local, owner, torch.full_like(owner, -1)).to(
        torch.int32
    )
    return route_of_row[:rows], expert_of_block, max_blocks


@triton.jit
def _dequant_tile(w_ptrs, s_ptrs, mask):
    """E4M3 weights and their UE8M0 exponents -> bf16 (exact)."""
    w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)
    e = tl.load(s_ptrs, mask=mask, other=127).to(tl.float32)
    return (w * tl.exp2(e - 127.0)).to(tl.bfloat16)


@triton.jit
def _round_e4m3(x):
    """Round fp32 values in [-448, 448] to the nearest E4M3 value (ties to even), in fp32.

    The quantum is 2^(e - 3) for a value in [2^e, 2^(e + 1)), and 2^-9 below 2^-6 (subnormals);
    dividing by a power of two is exact, so `rint` gives the E4M3 rounding.
    """
    exponent = ((x.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127
    quantum = tl.exp2((tl.maximum(exponent, -6) - 3).to(tl.float32))
    return libdevice.rint(x / quantum) * quantum


@triton.jit
def _gemm1_kernel(
    X,
    W13,
    S13,
    ROUTE,
    EXPERT,
    H,
    top_k,
    hidden,
    inter,
    stride_x,
    stride_we,
    stride_wr,
    stride_se,
    stride_sr,
    stride_h,
    gate_softcap,
    up_softcap,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block = tl.program_id(0)
    expert = tl.load(EXPERT + block)
    if expert < 0:
        return
    rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
    route = tl.load(ROUTE + rows)
    row_ok = route >= 0
    token = tl.where(row_ok, route // top_k, 0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)  # intermediate columns j
    col_ok = cols < inter
    offs_k = tl.arange(0, BLOCK_K)
    w_base = W13 + expert.to(tl.int64) * stride_we
    s_base = S13 + expert.to(tl.int64) * stride_se
    acc_g = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_u = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, hidden, BLOCK_K):
        k = k0 + offs_k
        x = tl.load(X + token[:, None] * stride_x + k[None, :], mask=row_ok[:, None], other=0.0)
        gate_rows = (2 * cols).to(tl.int64)
        mask = col_ok[:, None]
        g = _dequant_tile(
            w_base + gate_rows[:, None] * stride_wr + k[None, :],
            s_base + gate_rows[:, None] * stride_sr + (k // 32)[None, :],
            mask,
        )
        u = _dequant_tile(
            w_base + (gate_rows + 1)[:, None] * stride_wr + k[None, :],
            s_base + (gate_rows + 1)[:, None] * stride_sr + (k // 32)[None, :],
            mask,
        )
        acc_g += tl.dot(x, tl.trans(g))
        acc_u += tl.dot(x, tl.trans(u))
    h = (
        gate_softcap
        * libdevice.tanh(acc_g / gate_softcap)
        * tl.sigmoid(acc_g)
        * up_softcap
        * libdevice.tanh(acc_u / up_softcap)
    )
    # MXFP8 rounding per 32 columns: scale 2^ceil(log2(amax / 448)), values to E4M3.
    groups = tl.reshape(h, [BLOCK_M, BLOCK_N // 32, 32])
    raw = tl.max(tl.abs(groups), axis=2) / 448.0
    bits = raw.to(tl.int32, bitcast=True)
    exponent = ((bits + 0x007FFFFF) & 0x7F800000) >> 23
    # 2^-127 (an fp32 subnormal) and 2^127, built from their bits: as Python literals Triton
    # would type the subnormal as fp64.
    min_scale = tl.full([1, 1], 1 << 22, tl.int32).to(tl.float32, bitcast=True)
    max_inverse = tl.full([1, 1], 254 << 23, tl.int32).to(tl.float32, bitcast=True)
    exponent = tl.where(raw <= min_scale, 0, tl.minimum(exponent, 254))
    scale = tl.maximum((exponent << 23).to(tl.float32, bitcast=True), min_scale)
    inverse = tl.where(exponent == 0, max_inverse, 1.0 / scale)
    q = _round_e4m3(tl.clamp(groups * inverse[:, :, None], -448.0, 448.0))
    h_hat = tl.reshape(q * scale[:, :, None], [BLOCK_M, BLOCK_N])
    tl.store(
        H + rows.to(tl.int64)[:, None] * stride_h + cols[None, :],
        h_hat.to(tl.bfloat16),
        mask=row_ok[:, None] & col_ok[None, :],
    )


@triton.jit
def _gemm2_kernel(
    H,
    W2,
    S2,
    ROUTE,
    EXPERT,
    WEIGHTS,
    OUT,
    top_k,
    hidden,
    inter,
    stride_h,
    stride_we,
    stride_wr,
    stride_se,
    stride_sr,
    stride_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block = tl.program_id(0)
    expert = tl.load(EXPERT + block)
    if expert < 0:
        return
    rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
    route = tl.load(ROUTE + rows)
    row_ok = route >= 0
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)  # hidden columns
    col_ok = cols < hidden
    offs_k = tl.arange(0, BLOCK_K)
    w_base = W2 + expert.to(tl.int64) * stride_we
    s_base = S2 + expert.to(tl.int64) * stride_se
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, inter, BLOCK_K):
        k = k0 + offs_k
        h = tl.load(
            H + rows.to(tl.int64)[:, None] * stride_h + k[None, :], mask=row_ok[:, None], other=0.0
        )
        w = _dequant_tile(
            w_base + cols.to(tl.int64)[:, None] * stride_wr + k[None, :],
            s_base + cols.to(tl.int64)[:, None] * stride_sr + (k // 32)[None, :],
            col_ok[:, None],
        )
        acc += tl.dot(h, tl.trans(w))
    weight = tl.load(WEIGHTS + tl.where(row_ok, route, 0), mask=row_ok, other=0.0)
    token = tl.where(row_ok, route // top_k, 0).to(tl.int64)
    tl.atomic_add(
        OUT + token[:, None] * stride_out + cols[None, :],
        acc * weight[:, None],
        mask=row_ok[:, None] & col_ok[None, :],
        sem="relaxed",
    )


def grouped_mxfp8_moe(
    x_hat: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w13_s: torch.Tensor,
    w2: torch.Tensor,
    w2_s: torch.Tensor,
    local_expert_offset: int,
    gate_softcap: float,
    up_softcap: float,
) -> torch.Tensor:
    """`x_hat` `[T, H]` bf16 (MXFP8-rounded) -> fp32 `[T, H]`, this rank's experts' weighted sum."""
    num_tokens, hidden = x_hat.shape
    local, rows, _ = w13.shape
    inter = rows // 2
    if hidden % K_TILE or inter % K_TILE:
        # Neither GEMM masks a partial K tile.
        raise ValueError(
            f"grouped MXFP8 MoE needs hidden ({hidden}) and intermediate ({inter}) sizes that are "
            f"multiples of {K_TILE}"
        )
    top_k = topk_ids.shape[1]
    out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x_hat.device)
    if num_tokens == 0:
        return out
    route_of_row, expert_of_block, max_blocks = align_routes(topk_ids, local_expert_offset, local)
    h = torch.empty(max_blocks * BLOCK_M, inter, dtype=torch.bfloat16, device=x_hat.device)
    block_n, block_k = 128, K_TILE
    _gemm1_kernel[(max_blocks, triton.cdiv(inter, block_n))](
        x_hat,
        w13,
        w13_s,
        route_of_row,
        expert_of_block,
        h,
        top_k,
        hidden,
        inter,
        x_hat.stride(0),
        w13.stride(0),
        w13.stride(1),
        w13_s.stride(0),
        w13_s.stride(1),
        h.stride(0),
        float(gate_softcap),
        float(up_softcap),
        BLOCK_M=BLOCK_M,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=8,
    )
    _gemm2_kernel[(max_blocks, triton.cdiv(hidden, block_n))](
        h,
        w2,
        w2_s,
        route_of_row,
        expert_of_block,
        topk_weights.reshape(-1).contiguous(),
        out,
        top_k,
        hidden,
        inter,
        h.stride(0),
        w2.stride(0),
        w2.stride(1),
        w2_s.stride(0),
        w2_s.stride(1),
        out.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return out
