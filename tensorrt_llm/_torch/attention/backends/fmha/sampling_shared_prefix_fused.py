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
"""Fused Triton attention for the n-sample shared-prefix path: two launches per layer.

Launch 1 (``_phase1_kernel``) does all the attention work of a layer:

* prefix programs: one per (group, KV head, split of the group's shared pages,
  chunk of the group's query rows). The queries of all rows of the group (times
  the GQA factor) form the M dimension of tensor-core tiles, so every shared
  page is read once for the whole group;
* tail programs: one per (row, KV head, split of the row's private pages). The
  row's current token never goes through the cache: its K/V come straight from
  the QKV tensor, and the first split of each (row, KV head) also persists them
  in the cache page.

Each program writes an un-normalized online-softmax state (running max in
base 2, running sum, accumulator) for its split. Launch 2 (``_reduce_kernel``)
combines the states of a (row, head) and writes the output in batch order.
All grids are fixed by constants, every size that changes between steps is read
from device tensors, so the pair can be captured in a CUDA graph.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

_NEG = tl.constexpr(-1.0e30)


@triton.jit
def _prefix_programs(
    pid,
    q_ptr,
    q_sb,
    q_sh,
    pool_ptr,
    pool_ps,
    pool_kvs,
    pool_hs,
    pool_ss,
    perm_ptr,
    qo_ptr,
    pi_ptr,
    pidx_ptr,
    plast_ptr,
    pempty_ptr,
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    ntok,
    sm_scale_log2,
    num_groups,
    HKV: tl.constexpr,
    GS: tl.constexpr,
    QL: tl.constexpr,
    D: tl.constexpr,
    TPB: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SP: tl.constexpr,
    MCH: tl.constexpr,
    GG: tl.constexpr,
    HAS_PERM: tl.constexpr,
    HAS_EMPTY: tl.constexpr,
):
    H: tl.constexpr = HKV * GS
    mc = pid % MCH
    t = pid // MCH
    s = t % SP
    t = t // SP
    h = t % HKV
    gg = t // HKV

    # A program serves the groups gg, gg + GG, ...: the grid holds ``GG`` group slots,
    # not one per padded group (a batch has a handful of sharing groups; the table
    # is padded to many for the FlashInfer calls, and empty groups cost a loop turn).
    for g in range(gg, num_groups, GG):
        # ``qo`` counts query tokens (``QL`` per row), as the FlashInfer prefix call takes it.
        row0 = tl.load(qo_ptr + g) // QL
        nrows = tl.load(qo_ptr + g + 1) // QL - row0
        # The M dimension of a tile enumerates (row of the group, query token of the
        # row, query head of the KV head's GQA group); every query of the group
        # attends to every shared token, so no mask is needed.
        for m0 in range(mc * BLOCK_M, nrows * QL * GS, MCH * BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            m_ok = offs_m < nrows * QL * GS
            gpos = row0 + offs_m // (QL * GS)
            tok_in_row = (offs_m // GS) % QL
            head = h * GS + offs_m % GS
            if HAS_PERM:
                b = tl.load(perm_ptr + gpos, mask=m_ok, other=0).to(tl.int64)
            else:
                b = gpos.to(tl.int64)
            token = b * QL + tok_in_row
            dims = tl.arange(0, D)
            q = tl.load(
                q_ptr + token[:, None] * q_sb + head[:, None] * q_sh + dims[None, :],
                mask=m_ok[:, None],
                other=0.0,
            )

            p0 = tl.load(pi_ptr + g)
            npg = tl.load(pi_ptr + g + 1) - p0
            last = tl.load(plast_ptr + g)
            total = tl.where(npg > 0, (npg - 1) * TPB + last, 0)
            per_split = (npg + SP - 1) // SP
            tok_lo = s * per_split * TPB
            tok_hi = tl.minimum((s + 1) * per_split * TPB, total)

            m_i = tl.full([BLOCK_M], _NEG, tl.float32)
            l_i = tl.zeros([BLOCK_M], tl.float32)
            acc = tl.zeros([BLOCK_M, D], tl.float32)
            for start in range(tok_lo, tok_hi, BLOCK_N):
                offs_n = start + tl.arange(0, BLOCK_N)
                n_ok = offs_n < tok_hi
                page = tl.load(pidx_ptr + p0 + offs_n // TPB, mask=n_ok, other=0).to(tl.int64)
                base = pool_ptr + page * pool_ps + h * pool_hs + (offs_n % TPB) * pool_ss
                k = tl.load(base[:, None] + dims[None, :], mask=n_ok[:, None], other=0.0,
                             eviction_policy="evict_last")
                v = tl.load(base[:, None] + pool_kvs + dims[None, :], mask=n_ok[:, None], other=0.0,
                             eviction_policy="evict_last")
                scores = tl.dot(q, tl.trans(k)) * sm_scale_log2
                scores = tl.where(n_ok[None, :], scores, _NEG)
                m_new = tl.maximum(m_i, tl.max(scores, 1))
                alpha = tl.exp2(m_i - m_new)
                p = tl.exp2(scores - m_new[:, None])
                l_i = l_i * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                m_i = m_new
            if HAS_EMPTY:
                empty = tl.load(pempty_ptr + gpos, mask=m_ok, other=0) != 0
                m_i = tl.where(empty, _NEG, m_i)
                l_i = tl.where(empty, 0.0, l_i)
                acc = tl.where(empty[:, None], 0.0, acc)
            idx = (s * ntok + token) * H + head
            tl.store(part_m_ptr + idx, m_i, mask=m_ok)
            tl.store(part_l_ptr + idx, l_i, mask=m_ok)
            tl.store(part_acc_ptr + idx[:, None] * D + dims[None, :], acc, mask=m_ok[:, None])


@triton.jit
def _tail_programs(
    pid,
    q_ptr,
    q_sb,
    q_sh,
    k_ptr,
    v_ptr,
    kv_sb,
    pool_ptr,
    pool_ps,
    pool_kvs,
    pool_hs,
    pool_ss,
    perm_ptr,
    ti_ptr,
    tidx_ptr,
    tlast_ptr,
    wpage_ptr,
    wslot_ptr,
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    ntok,
    sm_scale_log2,
    HKV: tl.constexpr,
    GS: tl.constexpr,
    QL: tl.constexpr,
    QP: tl.constexpr,
    MT: tl.constexpr,
    D: tl.constexpr,
    TPB: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SP: tl.constexpr,
    ST: tl.constexpr,
    HAS_PERM: tl.constexpr,
):
    H: tl.constexpr = HKV * GS
    st = pid % ST
    t = pid // ST
    h = t % HKV
    i = t // HKV
    if HAS_PERM:
        b = tl.load(perm_ptr + i).to(tl.int64)
    else:
        b = i.to(tl.int64)
    dims = tl.arange(0, D)
    # M dimension: (query token of the row, query head of the GQA group), padded to MT.
    offs_m = tl.arange(0, MT)
    m_ok = offs_m < QL * GS
    tok_in_row = offs_m // GS
    head = h * GS + offs_m % GS
    token = b * QL + tok_in_row
    q = tl.load(
        q_ptr + token[:, None] * q_sb + head[:, None] * q_sh + dims[None, :],
        mask=m_ok[:, None],
        other=0.0,
    )
    t0 = tl.load(ti_ptr + i)
    npg = tl.load(ti_ptr + i + 1) - t0
    last = tl.load(tlast_ptr + i)
    # Tokens of the row's private pages before this step's QL tokens. ``last`` is
    # derived from the device KV length, so it may be zero or negative when the
    # host page count runs ahead; the sum is the true length either way.
    cached = tl.maximum(tl.where(npg > 0, (npg - 1) * TPB + last, 0) - QL, 0)
    per_split = ((cached + ST - 1) // ST + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    tok_lo = st * per_split
    tok_hi = tl.minimum(tok_lo + per_split, cached)

    m_i = tl.full([MT], _NEG, tl.float32)
    l_i = tl.zeros([MT], tl.float32)
    acc = tl.zeros([MT, D], tl.float32)
    for start in range(tok_lo, tok_hi, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        n_ok = offs_n < tok_hi
        page = tl.load(tidx_ptr + t0 + offs_n // TPB, mask=n_ok, other=0).to(tl.int64)
        base = pool_ptr + page * pool_ps + h * pool_hs + (offs_n % TPB) * pool_ss
        k = tl.load(base[:, None] + dims[None, :], mask=n_ok[:, None], other=0.0)
        v = tl.load(base[:, None] + pool_kvs + dims[None, :], mask=n_ok[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(k)) * sm_scale_log2
        scores = tl.where(n_ok[None, :], scores, _NEG)
        m_new = tl.maximum(m_i, tl.max(scores, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    if st == 0:
        # This step's QL tokens: attended straight from the QKV tensor (causal
        # among themselves), then persisted in the row's private pages.
        offs_j = tl.arange(0, QP)
        j_ok = offs_j < QL
        new_tok = b * QL + offs_j
        k_new = tl.load(
            k_ptr + new_tok[:, None] * kv_sb + h * D + dims[None, :], mask=j_ok[:, None], other=0.0
        )
        v_new = tl.load(
            v_ptr + new_tok[:, None] * kv_sb + h * D + dims[None, :], mask=j_ok[:, None], other=0.0
        )
        scores = tl.dot(q, tl.trans(k_new)) * sm_scale_log2
        visible = (offs_j[None, :] <= tok_in_row[:, None]) & j_ok[None, :]
        scores = tl.where(visible, scores, _NEG)
        m_new = tl.maximum(m_i, tl.max(scores, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_new.dtype), v_new)
        m_i = m_new
        if QL == 1:
            page_w = tl.load(wpage_ptr + b).to(tl.int64)
            slot_w = tl.load(wslot_ptr + b).to(tl.int64)
            dst = pool_ptr + page_w * pool_ps + h * pool_hs + slot_w * pool_ss + dims
            tl.store(dst + 0 * offs_j[:, None], k_new, mask=offs_j[:, None] == 0)
            tl.store(dst + 0 * offs_j[:, None] + pool_kvs, v_new, mask=offs_j[:, None] == 0)
        else:
            pos = cached + offs_j
            pg = pos // TPB
            w_ok = j_ok & (pg < npg)
            page_w = tl.load(tidx_ptr + t0 + pg, mask=w_ok, other=0).to(tl.int64)
            dst = (
                pool_ptr
                + page_w[:, None] * pool_ps
                + h * pool_hs
                + (pos % TPB)[:, None] * pool_ss
                + dims[None, :]
            )
            tl.store(dst, k_new, mask=w_ok[:, None])
            tl.store(dst + pool_kvs, v_new, mask=w_ok[:, None])

    idx = ((SP + st) * ntok + token) * H + head
    tl.store(part_m_ptr + idx, m_i, mask=m_ok)
    tl.store(part_l_ptr + idx, l_i, mask=m_ok)
    tl.store(part_acc_ptr + idx[:, None] * D + dims[None, :], acc, mask=m_ok[:, None])


@triton.jit
def _phase1_kernel(
    q_ptr,
    q_sb,
    q_sh,
    k_ptr,
    v_ptr,
    kv_sb,
    pool_ptr,
    pool_ps,
    pool_kvs,
    pool_hs,
    pool_ss,
    perm_ptr,
    qo_ptr,
    pi_ptr,
    pidx_ptr,
    plast_ptr,
    pempty_ptr,
    ti_ptr,
    tidx_ptr,
    tlast_ptr,
    wpage_ptr,
    wslot_ptr,
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    ntok,
    sm_scale_log2,
    num_prefix_programs,
    num_groups,
    HKV: tl.constexpr,
    GS: tl.constexpr,
    QL: tl.constexpr,
    QP: tl.constexpr,
    MT: tl.constexpr,
    D: tl.constexpr,
    TPB: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_NT: tl.constexpr,
    SP: tl.constexpr,
    ST: tl.constexpr,
    MCH: tl.constexpr,
    GG: tl.constexpr,
    HAS_PERM: tl.constexpr,
    HAS_EMPTY: tl.constexpr,
    MODE: tl.constexpr,
):
    # MODE 0: prefix programs then tail programs in one grid; 1: prefix only; 2: tail only.
    # A single-purpose launch picks its body at compile time. With a runtime test both bodies
    # stay in the kernel and the launch pays the registers and shared memory of the heavier
    # one (the prefix tile), which throttles the streaming tail programs' occupancy.
    pid = tl.program_id(0)
    if MODE == 1:
        _prefix_programs(
            pid, q_ptr, q_sb, q_sh, pool_ptr, pool_ps, pool_kvs, pool_hs, pool_ss, perm_ptr,
            qo_ptr, pi_ptr, pidx_ptr, plast_ptr, pempty_ptr, part_m_ptr, part_l_ptr,
            part_acc_ptr, ntok, sm_scale_log2, num_groups, HKV, GS, QL, D, TPB, BLOCK_M,
            BLOCK_N, SP, MCH, GG, HAS_PERM, HAS_EMPTY,
        )  # fmt: skip
    elif MODE == 2:
        _tail_programs(
            pid, q_ptr, q_sb, q_sh, k_ptr, v_ptr, kv_sb, pool_ptr,
            pool_ps, pool_kvs, pool_hs, pool_ss, perm_ptr, ti_ptr, tidx_ptr, tlast_ptr,
            wpage_ptr, wslot_ptr, part_m_ptr, part_l_ptr, part_acc_ptr, ntok, sm_scale_log2,
            HKV, GS, QL, QP, MT, D, TPB, BLOCK_NT, SP, ST, HAS_PERM,
        )  # fmt: skip
    else:
        if pid < num_prefix_programs:
            _prefix_programs(
                pid, q_ptr, q_sb, q_sh, pool_ptr, pool_ps, pool_kvs, pool_hs, pool_ss, perm_ptr,
                qo_ptr, pi_ptr, pidx_ptr, plast_ptr, pempty_ptr, part_m_ptr, part_l_ptr,
                part_acc_ptr, ntok, sm_scale_log2, num_groups, HKV, GS, QL, D, TPB, BLOCK_M,
                BLOCK_N, SP, MCH, GG, HAS_PERM, HAS_EMPTY,
            )  # fmt: skip
        else:
            _tail_programs(
                pid - num_prefix_programs, q_ptr, q_sb, q_sh, k_ptr, v_ptr, kv_sb, pool_ptr,
                pool_ps, pool_kvs, pool_hs, pool_ss, perm_ptr, ti_ptr, tidx_ptr, tlast_ptr,
                wpage_ptr, wslot_ptr, part_m_ptr, part_l_ptr, part_acc_ptr, ntok, sm_scale_log2,
                HKV, GS, QL, QP, MT, D, TPB, BLOCK_NT, SP, ST, HAS_PERM,
            )  # fmt: skip


@triton.jit
def _reduce_kernel(
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    out_ptr,
    out_sb,
    ntok,
    H: tl.constexpr,
    HP: tl.constexpr,
    D: tl.constexpr,
    NSLOT: tl.constexpr,
):
    # One program per query token, all heads: few large programs instead of
    # tokens * heads tiny ones.
    b = tl.program_id(0)
    heads = tl.arange(0, HP)
    h_ok = heads < H
    dims = tl.arange(0, D)
    peak = tl.load(part_m_ptr + b * H + heads, mask=h_ok, other=0.0)
    for slot in tl.static_range(1, NSLOT):
        peak = tl.maximum(
            peak, tl.load(part_m_ptr + (slot * ntok + b) * H + heads, mask=h_ok, other=0.0)
        )
    l_tot = tl.zeros([HP], tl.float32)
    acc = tl.zeros([HP, D], tl.float32)
    for slot in tl.static_range(NSLOT):
        idx = (slot * ntok + b) * H + heads
        w = tl.exp2(tl.load(part_m_ptr + idx, mask=h_ok, other=0.0) - peak)
        l_tot += w * tl.load(part_l_ptr + idx, mask=h_ok, other=0.0)
        acc += w[:, None] * tl.load(
            part_acc_ptr + idx[:, None] * D + dims[None, :], mask=h_ok[:, None], other=0.0
        )
    tl.store(
        out_ptr + b.to(tl.int64) * out_sb + heads[:, None] * D + dims[None, :],
        (acc / l_tot[:, None]).to(out_ptr.dtype.element_ty),
        mask=h_ok[:, None],
    )


@triton.jit
def _gather_q_kernel(
    q_ptr,
    q_sb,
    q_sh,
    perm_ptr,
    out_ptr,
    H: tl.constexpr,
    HP: tl.constexpr,
    D: tl.constexpr,
    QL: tl.constexpr,
    HAS_PERM: tl.constexpr,
):
    # Grouped token t = position * QL + j reads the batch row perm[position].
    t = tl.program_id(0)
    pos = t // QL
    j = t % QL
    if HAS_PERM:
        row = tl.load(perm_ptr + pos).to(tl.int64)
    else:
        row = pos.to(tl.int64)
    heads = tl.arange(0, HP)
    dims = tl.arange(0, D)
    mask = (heads < H)[:, None]
    src = q_ptr + (row * QL + j) * q_sb + heads[:, None] * q_sh + dims[None, :]
    dst = out_ptr + t.to(tl.int64) * (H * D) + heads[:, None] * D + dims[None, :]
    tl.store(dst, tl.load(src, mask=mask, other=0.0), mask=mask)


@triton.jit
def _merge_ctx_kernel(
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    pre_out_ptr,
    pre_lse_ptr,
    perm_ptr,
    pempty_ptr,
    out_ptr,
    out_sb,
    ntok,
    H: tl.constexpr,
    HP: tl.constexpr,
    D: tl.constexpr,
    QL: tl.constexpr,
    NSLOT: tl.constexpr,
    HAS_PERM: tl.constexpr,
    HAS_EMPTY: tl.constexpr,
):
    # One program per grouped token: the prefix state (normalized output and base-2 log-sum-exp
    # of the context kernel, grouped order) joins the tail states (partial arena, batch order).
    t = tl.program_id(0)
    pos = t // QL
    j = t % QL
    if HAS_PERM:
        row = tl.load(perm_ptr + pos).to(tl.int64)
    else:
        row = pos.to(tl.int64)
    tok = row * QL + j
    heads = tl.arange(0, HP)
    h_ok = heads < H
    dims = tl.arange(0, D)
    lse = tl.load(pre_lse_ptr + t * H + heads, mask=h_ok, other=_NEG)
    if HAS_EMPTY:
        lse = tl.where(tl.load(pempty_ptr + pos) != 0, _NEG, lse)
    peak = lse
    for slot in tl.static_range(NSLOT):
        peak = tl.maximum(
            peak, tl.load(part_m_ptr + (slot * ntok + tok) * H + heads, mask=h_ok, other=_NEG)
        )
    w_pre = tl.exp2(lse - peak)
    l_tot = w_pre
    pre = tl.load(
        pre_out_ptr + (t.to(tl.int64) * H + heads[:, None]) * D + dims[None, :],
        mask=h_ok[:, None],
        other=0.0,
    ).to(tl.float32)
    acc = w_pre[:, None] * pre
    for slot in tl.static_range(NSLOT):
        idx = (slot * ntok + tok) * H + heads
        w = tl.exp2(tl.load(part_m_ptr + idx, mask=h_ok, other=_NEG) - peak)
        l_tot += w * tl.load(part_l_ptr + idx, mask=h_ok, other=0.0)
        acc += w[:, None] * tl.load(
            part_acc_ptr + idx[:, None] * D + dims[None, :], mask=h_ok[:, None], other=0.0
        )
    tl.store(
        out_ptr + tok * out_sb + heads[:, None] * D + dims[None, :],
        (acc / l_tot[:, None]).to(out_ptr.dtype.element_ty),
        mask=h_ok[:, None],
    )


class CtxPrefix:
    """Shared-prefix attention through the TRTLLM-Gen context kernel (SM100), non-causal.

    The queries of every group (grouped order, ``q_len`` tokens per row) attend the group's
    shared pages in one call: the group is a request whose KV is its page list. The kernel
    returns the normalized output and the base-2 log-sum-exp, which the merge kernel combines
    with the tail states. It reaches ~700 TFLOP/s at MHA 20/20 d128 where the Triton prefix
    tile gets 265-330, but its tile choice depends on ``max_q_len``: up to 512 queries per
    request it is twice as fast as above (3 groups x 40 rows x q_len 9: 63 vs 112 us), so a
    group is cut into chunks of at most ``MAX_CHUNK_QUERIES`` query tokens (each chunk is a
    request over the same pages). The tables are fixed-address device tensors, rewritten
    when the groups change (``set_tables``); ``max_q_len`` and ``max_kv_len`` are launch
    constants of the captured graph.
    """

    MAX_CHUNK_QUERIES = 504  # a multiple of every q_len up to 9 below the 512 tile threshold

    def __init__(
        self,
        tokens: int,
        num_heads: int,
        head_dim: int,
        q_len: int,
        num_requests: int,
        max_pages: int,
        max_kv_len: int,
        dtype: torch.dtype,
        device,
        tokens_per_block: int,
    ):
        self.num_heads = num_heads
        self.q_len = q_len
        self.tokens_per_block = tokens_per_block
        self.chunk_tokens = max(q_len, self.MAX_CHUNK_QUERIES // q_len * q_len)
        self.num_requests = num_requests
        self.max_kv_len = max_kv_len
        self.q = torch.empty((tokens, num_heads, head_dim), dtype=dtype, device=device)
        self.out = torch.empty_like(self.q)
        self.lse = torch.empty((tokens, num_heads), dtype=torch.float32, device=device)
        self.block_tables = torch.zeros((num_requests, max_pages), dtype=torch.int32, device=device)
        self.seq_lens = torch.zeros((num_requests,), dtype=torch.int32, device=device)
        self.cum_q = torch.zeros((num_requests + 1,), dtype=torch.int32, device=device)
        self.cum_kv = torch.zeros((num_requests + 1,), dtype=torch.int32, device=device)
        self.workspace = _ctx_workspace(device)

    @staticmethod
    def num_requests_for(num_groups: int, tokens: int, q_len: int) -> int:
        """Upper bound of the (group, chunk) requests of a batch of ``tokens`` query tokens."""
        chunk = max(q_len, CtxPrefix.MAX_CHUNK_QUERIES // q_len * q_len)
        return num_groups + tokens // chunk + 1

    def table_arrays(self):
        """Host copies' layout: (block_tables, seq_lens, cum_q, cum_kv)."""
        return self.block_tables, self.seq_lens, self.cum_q, self.cum_kv

    def run(self, pool: torch.Tensor, sm_scale: float) -> None:
        from flashinfer.prefill import trtllm_batch_context_with_kv_cache

        trtllm_batch_context_with_kv_cache(
            query=self.q,
            kv_cache=pool,
            workspace_buffer=self.workspace,
            block_tables=self.block_tables,
            seq_lens=self.seq_lens,
            max_q_len=self.chunk_tokens,
            max_kv_len=self.max_kv_len,
            bmm1_scale=sm_scale,
            bmm2_scale=1.0,
            batch_size=self.num_requests,
            cum_seq_lens_q=self.cum_q,
            cum_seq_lens_kv=self.cum_kv,
            out=self.out,
            kv_layout="HND",
            causal=False,
            lse=self.lse,
            return_lse=True,
        )


_ctx_workspaces: dict = {}


def _ctx_workspace(device) -> torch.Tensor:
    """Zero-initialized scratch of the context kernel (its counters must start at zero)."""
    buf = _ctx_workspaces.get(device)
    if buf is None:
        buf = _ctx_workspaces[device] = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
    return buf


def ctx_prefix_supported(head_dim: int, dtype: torch.dtype, device=None) -> bool:
    """Whether the TRTLLM-Gen context kernel can serve the prefix (SM100/SM103, bf16, d64/d128)."""
    if dtype != torch.bfloat16 or head_dim not in (64, 128):
        return False
    major = torch.cuda.get_device_capability(device)[0]
    if major != 10:
        return False
    try:
        from flashinfer.prefill import trtllm_batch_context_with_kv_cache  # noqa: F401
    except ImportError:
        return False
    return True


class FusedWorkspace:
    """Partial-state buffers for one (tokens, heads, head_dim, splits) configuration.

    ``tokens`` is the number of query tokens of the batch (rows times ``q_len``).
    """

    def __init__(self, tokens: int, num_heads: int, head_dim: int, slots: int, device):
        self.m = torch.empty((slots, tokens, num_heads), dtype=torch.float32, device=device)
        self.l = torch.empty_like(self.m)
        self.acc = torch.empty(
            (slots, tokens, num_heads, head_dim), dtype=torch.float32, device=device
        )

    @classmethod
    def shared(
        cls, tokens: int, num_heads: int, head_dim: int, slots: int, device
    ) -> "FusedWorkspace":
        """Views of one per-device arena that every captured batch size shares.

        The partial states live only between the two launches of a layer and CUDA
        graphs run one after the other on a stream, so batch sizes can share the
        storage instead of each keeping ``slots * tokens * heads * head_dim`` floats.
        A larger request replaces the arena; graphs captured earlier keep the old
        one alive through the views they hold.
        """
        floats = slots * tokens * num_heads * (head_dim + 2)
        arena = _arenas.get(device)
        if arena is None or arena.numel() < floats:
            arena = torch.empty(floats, dtype=torch.float32, device=device)
            _arenas[device] = arena
        self = cls.__new__(cls)
        n_ml = slots * tokens * num_heads
        self.m = arena[:n_ml].view(slots, tokens, num_heads)
        self.l = arena[n_ml : 2 * n_ml].view(slots, tokens, num_heads)
        self.acc = arena[2 * n_ml : floats].view(slots, tokens, num_heads, head_dim)
        return self


_arenas: dict = {}
_side_streams: dict = {}


def _side_stream(device) -> torch.cuda.Stream:
    stream = _side_streams.get(device)
    if stream is None:
        stream = _side_streams[device] = torch.cuda.Stream(device)
    return stream


DEFAULT_BLOCK_M = 128
"""Query rows per prefix tile (capped by the group's query rows). 128 rows fill a tcgen05 tile;
with the single-purpose launches (``MODE`` 1/2) the tail programs no longer inherit the tile's
footprint, and the per-layer time of the two-stream path at q_len 9 (GB200, MHA 20/20, d128,
3 groups x 40 rows, 1000-token tails) is 308 us at 4000 prompt tokens and 253 us at 1600 with
128 rows, against 364 / ~330 us with 64 (the earlier default, chosen when a 128-row tile slowed
the tail launch down by 10x)."""

DEFAULT_GROUP_GRID = 8
"""Group slots in the launch grid (see ``group_grid``)."""

MAX_QUERIES_PER_TILE = 64
"""Largest ``q_len * gqa_group`` the tail tile (and the K/V of the step's own
tokens) holds; larger shapes are served by the FlashInfer calls."""


def default_prefix_splits(
    num_groups: int, num_kv_heads: int, max_group_rows: int, q_len: int, gqa: int, num_sms: int
) -> int:
    """Split-K factor of the shared pages: enough programs to cover every SM once.

    A (group, KV head, query chunk) program streams the group's whole prefix. When
    those already number about the SM count (wide groups at q_len 9) extra splits
    only add partial-state traffic (measured: 1 beats 2 and 4 there); with q_len 1
    and groups of 20 rows there are a few dozen, and 2 splits are faster than 1.
    """
    chunks = triton.cdiv(max_group_rows * q_len * gqa, DEFAULT_BLOCK_M)
    programs = max(min(num_groups, DEFAULT_GROUP_GRID) * num_kv_heads * chunks, 1)
    return max(1, min(4, triton.cdiv(num_sms, programs)))


def fused_sampling_attention(
    pool: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    *,
    perm: Optional[torch.Tensor],
    group_qo_indptr: torch.Tensor,
    prefix_indptr: torch.Tensor,
    prefix_indices: torch.Tensor,
    prefix_last_page_len: torch.Tensor,
    prefix_empty: Optional[torch.Tensor],
    tail_indptr: torch.Tensor,
    tail_indices: torch.Tensor,
    tail_last_page_len: torch.Tensor,
    write_page: torch.Tensor,
    write_slot: torch.Tensor,
    sm_scale: float,
    num_groups: int,
    max_group_rows: int,
    workspace: FusedWorkspace,
    group_grid: Optional[int] = None,
    q_len: int = 1,
    prefix_splits: int = 8,
    tail_splits: int = 2,
    block_m: Optional[int] = None,
    block_n: int = 64,
    block_nt: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
    split_launch: bool = True,
    overlap: bool = True,
    tail_num_warps: int = 2,
    tail_num_stages: int = 2,
    sm90_tune: bool = True,  # apply SM90-specific tuning (evict_last already in kernel)
    ctx: Optional[CtxPrefix] = None,
) -> None:
    """One layer: persist the step's K/V, attend prefix + tail, write ``out`` (batch order).

    With ``q_len`` > 1 (speculative decoding) every row carries ``q_len`` query
    tokens at positions ``kv_len - q_len + j`` (``kv_len`` the device KV length
    including them): they all see the group's shared pages, the row's private
    pages, and the row's own tokens up to themselves.

    Args:
        pool: [pages, 2, kv_heads, tokens_per_block, head_dim] layer pool (HND).
        q: [rows * q_len, heads, head_dim]; k, v: [rows * q_len, kv_heads, head_dim],
            batch order (the tokens of a row consecutive), each token's heads contiguous.
        out: [rows * q_len, heads, head_dim], batch order.
        perm: [rows] batch row of each grouped position (None: identity).
        group_qo_indptr: [num_groups + 1] int32, group query-token ranges in grouped
            order (``q_len`` tokens per row).
        prefix_indptr, prefix_indices, prefix_last_page_len: the shared pages of each
            group (as the FlashInfer prefix call takes them).
        prefix_empty: [rows] bool, grouped order, rows that take the tail alone.
        tail_indptr, tail_indices, tail_last_page_len: private pages of each row,
            grouped order. The last page's length counts the step's ``q_len``
            tokens; it is derived from the device KV length, so the host page
            count may exceed what it needs (the length is then zero or negative).
        write_page, write_slot: [rows] cache position of the current token
            (``q_len`` == 1 only; larger ``q_len`` derives the positions of the
            tokens from ``tail_last_page_len`` and the tail pages).
        num_groups, max_group_rows: launch-grid bounds (padding groups are empty).
        workspace: partial-state buffers for ``rows * q_len`` tokens.
        ctx: serve the shared pages through the TRTLLM-Gen context kernel (see ``CtxPrefix``;
            its tables must describe the same groups). The Triton prefix launch is skipped and
            ``prefix_splits`` is ignored; the workspace needs ``tail_splits`` slots.
    """
    ntok, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    gs = num_heads // num_kv_heads
    tokens_per_block = pool.shape[3]
    # SM90 (H100) tuning: evict_last is already in the kernel; bump warps/stages.
    if sm90_tune and q.device.type == "cuda":
        import torch
        cc = torch.cuda.get_device_capability(q.device)
        if cc == (9, 0):  # SM90
            num_warps = max(num_warps, 8)
            num_stages = max(num_stages, 3)
    assert ntok % q_len == 0 and out.stride(2) == 1 and pool.stride(4) == 1
    assert q_len * gs <= MAX_QUERIES_PER_TILE and q_len <= 16
    for tensor in (q, k, v):
        assert tensor.stride(2) == 1 and tensor.stride(1) == head_dim
    assert k.stride(0) == v.stride(0)
    qp = 16
    mt = max(16, triton.next_power_of_2(q_len * gs))
    if block_m is None:
        block_m = min(DEFAULT_BLOCK_M, max(16, triton.next_power_of_2(max_group_rows * q_len * gs)))
    mch = triton.cdiv(max_group_rows * q_len * gs, block_m)
    gg = min(num_groups, group_grid or DEFAULT_GROUP_GRID)
    if ctx is not None:
        prefix_splits = 0
    num_prefix = gg * num_kv_heads * prefix_splits * mch
    num_tail = (ntok // q_len) * num_kv_heads * tail_splits
    slots = prefix_splits + tail_splits
    assert workspace.m.shape[0] >= slots and workspace.m.shape[1] >= ntok

    def launch(mode, grid, warps, stages, block_nt):
        _phase1_kernel[(grid,)](
            q, q.stride(0), q.stride(1), k, v, k.stride(0),
            pool, pool.stride(0), pool.stride(1), pool.stride(2), pool.stride(3),
            perm if perm is not None else write_page,
            group_qo_indptr, prefix_indptr, prefix_indices, prefix_last_page_len,
            prefix_empty if prefix_empty is not None else write_page,
            tail_indptr, tail_indices, tail_last_page_len, write_page, write_slot,
            workspace.m, workspace.l, workspace.acc,
            ntok, sm_scale * 1.4426950408889634, num_prefix, num_groups,
            HKV=num_kv_heads, GS=gs, QL=q_len, QP=qp, MT=mt, D=head_dim, TPB=tokens_per_block,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_NT=block_nt,
            SP=prefix_splits, ST=tail_splits, MCH=mch, GG=gg,
            HAS_PERM=perm is not None, HAS_EMPTY=prefix_empty is not None, MODE=mode,
            num_warps=warps, num_stages=stages,
        )  # fmt: skip

    if ctx is not None:
        _gather_q_kernel[(ntok,)](
            q, q.stride(0), q.stride(1), perm if perm is not None else write_page, ctx.q,
            H=num_heads, HP=triton.next_power_of_2(num_heads), D=head_dim, QL=q_len,
            HAS_PERM=perm is not None, num_warps=4,
        )  # fmt: skip
        ctx.run(pool, sm_scale)
        launch(2, num_tail, tail_num_warps, tail_num_stages, block_nt)
        _merge_ctx_kernel[(ntok,)](
            workspace.m, workspace.l, workspace.acc, ctx.out, ctx.lse,
            perm if perm is not None else write_page,
            prefix_empty if prefix_empty is not None else write_page, out, out.stride(0), ntok,
            H=num_heads, HP=triton.next_power_of_2(num_heads), D=head_dim, QL=q_len,
            NSLOT=slots, HAS_PERM=perm is not None, HAS_EMPTY=prefix_empty is not None,
            num_warps=4,
        )  # fmt: skip
        return
    if split_launch and overlap:
        # The prefix tiles are tensor-core work, the tail's programs stream private
        # pages; on two streams (forked and joined inside a CUDA graph capture too)
        # they can run side by side.
        main = torch.cuda.current_stream()
        side = _side_stream(q.device)
        side.wait_stream(main)
        launch(1, num_prefix, num_warps, num_stages, block_nt)
        with torch.cuda.stream(side):
            launch(2, num_tail, tail_num_warps, tail_num_stages, block_nt)
        main.wait_stream(side)
    elif split_launch:
        # Separate launches: the prefix tiles need many registers and shared memory
        # (a few CTAs per SM), the tail's streaming programs are best at high occupancy.
        launch(1, num_prefix, num_warps, num_stages, block_nt)
        launch(2, num_tail, tail_num_warps, tail_num_stages, block_nt)
    else:
        launch(0, num_prefix + num_tail, num_warps, num_stages, block_nt)
    _reduce_kernel[(ntok,)](
        workspace.m, workspace.l, workspace.acc, out, out.stride(0), ntok,
        H=num_heads, HP=triton.next_power_of_2(num_heads), D=head_dim, NSLOT=slots, num_warps=4,
    )  # fmt: skip
