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
"""Triton kernels of cross_layer_expert_prefetch.

``rank_and_stats``: one launch over the 256 experts of one MoE layer, with no
sequential loop over experts.

* Prediction (``NE / BE`` programs, one block of ``BE`` experts each): from
  approximate router logits of the next layer (``h_L`` times the next layer's
  gate), the score ``sigmoid(logit) + bias`` selects each live row's top-8
  experts; experts are ranked by how many rows select them, then by their
  maximum score over the rows, then by ID. Each program computes the rank of
  its own experts by pairwise comparison against all 256 and writes the IDs
  ranked below ``KMAX`` to their rank slot.
* Statistics (one more program): from this layer's router logits (the ones
  its routed MoE uses) and the ranking predicted for this layer by the
  previous layer, the program adds to this layer's counters: calls, distinct
  experts D, and the number of the D experts among the first K predicted, for
  each K of ``STAT_KS``.

``gemm_rank``: the prediction in one launch, without the router GEMM: each of
``NE / 16`` programs computes the logits of 16 experts (``tl.dot`` over the
hidden size) and their scores, publishes them, and the last program to arrive
ranks all experts in the same order as above, with one sort of a unique key.

``delay``: one warp that sleeps for a given time, to place the work queued after
it on the same stream later in the decode step.

``prefetch_experts``: fire-and-forget ``cp.async.bulk.prefetch.L2`` of the
local FC1 / FC2 weights and block scales of the first ``K`` predicted experts,
expert by expert in rank order, FC1 bytes before FC2 bytes.
"""

import triton
import triton.language as tl

NUM_EXPERTS = 256
TOP_K = 8
MAX_ROWS = 8
KMAX = 32
# Experts ranked per prediction program.
RANK_BLOCK = 32
# Experts per program of the fused GEMM + rank kernel.
GEMM_BLOCK = 16
# Columns of the per-layer counter rows (int64).
STAT_KS = (8, 12, 18, 24, 32)
C_CALLS = 0
C_DISTINCT = 1
C_HITS = 2  # C_HITS + i for STAT_KS[i]
NUM_COUNTERS = C_HITS + len(STAT_KS)


@triton.jit
def _scores_topk_union(s, live, NE: tl.constexpr, TOPK: tl.constexpr):
    """For scores ``s`` [BR, NE] (dead rows -inf), the experts in each live
    row's top-``TOPK``; returns (count over rows [NE], max score [NE])."""
    e = tl.arange(0, NE)
    # TOPK-th largest score of every row: TOPK passes, each removing one
    # maximum (the lowest index among equal maxima).
    work = s
    for _ in tl.static_range(TOPK):
        thr, idx = tl.max(work, axis=1, return_indices=True)
        work = tl.where(e[None, :] == idx[:, None], float("-inf"), work)
    sel = (s >= thr[:, None]) & live[:, None]
    cnt = tl.sum(sel.to(tl.int32), axis=0)
    mx = tl.max(s, axis=0)
    return cnt, mx


@triton.jit
def _row_topk_union(
    logits_ptr, bias_ptr, rows, NE: tl.constexpr, BR: tl.constexpr, TOPK: tl.constexpr
):
    """Per live row, the experts in that row's top-``TOPK`` of
    ``sigmoid(logit) + bias``; returns (count over rows [NE], max score [NE])."""
    r = tl.arange(0, BR)
    e = tl.arange(0, NE)
    live = r < rows
    x = tl.load(logits_ptr + r[:, None] * NE + e[None, :], mask=live[:, None], other=0.0).to(
        tl.float32
    )
    b = tl.load(bias_ptr + e).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-x)) + b[None, :]
    s = tl.where(live[:, None], s, float("-inf"))
    return _scores_topk_union(s, live, NE, TOPK)


@triton.jit
def _store_ranked(cnt, mx, eb, out_ids_ptr, NE: tl.constexpr, KM: tl.constexpr):
    """Rank the experts ``eb`` against all NE by (rows selecting, max score,
    ID) and store each one ranked below ``KM`` to its rank slot."""
    e = tl.arange(0, NE)
    same = e[None, :] == eb[:, None]
    cb = tl.sum(tl.where(same, cnt[None, :], 0), axis=1)
    mb = tl.max(tl.where(same, mx[None, :], float("-inf")), axis=1)
    cj = cnt[None, :]
    mj = mx[None, :]
    better = (cj > cb[:, None]) | (
        (cj == cb[:, None])
        & ((mj > mb[:, None]) | ((mj == mb[:, None]) & (e[None, :] < eb[:, None])))
    )
    rank = tl.sum(better.to(tl.int32), axis=1)
    tl.store(out_ids_ptr + rank, eb.to(tl.int32), mask=rank < KM)


@triton.jit
def _rank_and_stats_kernel(
    next_logits_ptr,
    next_bias_ptr,
    out_ids_ptr,
    cur_logits_ptr,
    cur_bias_ptr,
    cur_ids_ptr,
    counters_ptr,
    rows,
    NPRED: tl.constexpr,
    HAS_STATS: tl.constexpr,
    NE: tl.constexpr,
    BR: tl.constexpr,
    BE: tl.constexpr,
    TOPK: tl.constexpr,
    KM: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    K3: tl.constexpr,
    K4: tl.constexpr,
):
    pid = tl.program_id(0)
    e = tl.arange(0, NE)
    if pid < NPRED:
        cnt, mx = _row_topk_union(next_logits_ptr, next_bias_ptr, rows, NE, BR, TOPK)
        # this program's experts, ranked with the values of the full vectors
        _store_ranked(cnt, mx, pid * BE + tl.arange(0, BE), out_ids_ptr, NE, KM)
    elif HAS_STATS:
        cnt, _ = _row_topk_union(cur_logits_ptr, cur_bias_ptr, rows, NE, BR, TOPK)
        actual = cnt > 0
        # rank position of every expert in the ranking predicted for this
        # layer (KM = not among the first KM)
        j = tl.arange(0, KM)
        ids = tl.load(cur_ids_ptr + j)
        pos = tl.min(tl.where(ids[:, None] == e[None, :], j[:, None], KM), axis=0)
        tl.atomic_add(counters_ptr + 0, tl.full([], 1, tl.int64))
        tl.atomic_add(counters_ptr + 1, tl.sum(actual.to(tl.int64), axis=0))
        tl.atomic_add(counters_ptr + 2, tl.sum((actual & (pos < K0)).to(tl.int64), axis=0))
        tl.atomic_add(counters_ptr + 3, tl.sum((actual & (pos < K1)).to(tl.int64), axis=0))
        tl.atomic_add(counters_ptr + 4, tl.sum((actual & (pos < K2)).to(tl.int64), axis=0))
        tl.atomic_add(counters_ptr + 5, tl.sum((actual & (pos < K3)).to(tl.int64), axis=0))
        tl.atomic_add(counters_ptr + 6, tl.sum((actual & (pos < K4)).to(tl.int64), axis=0))


def rank_and_stats(
    next_logits,
    next_bias,
    out_ids,
    cur_logits,
    cur_bias,
    cur_ids,
    counters,
    rows: int,
    has_pred: bool,
    has_stats: bool,
) -> None:
    """Launch the prediction and / or statistics programs on the current
    stream. Unused pointer arguments may be any tensor of the right device."""
    npred = NUM_EXPERTS // RANK_BLOCK if has_pred else 0
    grid = npred + (1 if has_stats else 0)
    if grid == 0:
        return
    _rank_and_stats_kernel[(grid,)](
        next_logits,
        next_bias,
        out_ids,
        cur_logits,
        cur_bias,
        cur_ids,
        counters,
        rows,
        NPRED=npred,
        HAS_STATS=has_stats,
        NE=NUM_EXPERTS,
        BR=MAX_ROWS,
        BE=RANK_BLOCK,
        TOPK=TOP_K,
        KM=KMAX,
        K0=STAT_KS[0],
        K1=STAT_KS[1],
        K2=STAT_KS[2],
        K3=STAT_KS[3],
        K4=STAT_KS[4],
        num_warps=4,
    )


@triton.jit
def _gemm_rank_kernel(
    h_ptr,
    w_ptr,
    bias_ptr,
    scores_ptr,
    ticket_ptr,
    out_ids_ptr,
    rows,
    H: tl.constexpr,
    NE: tl.constexpr,
    BR: tl.constexpr,
    BM: tl.constexpr,
    BE: tl.constexpr,
    BK: tl.constexpr,
    TOPK: tl.constexpr,
    KM: tl.constexpr,
):
    # Logits and scores of this program's BE experts for the rows of h; the
    # gate rows are read contiguously along the hidden dimension.
    pid = tl.program_id(0)
    NPROG: tl.constexpr = NE // BE
    m = tl.arange(0, BM)
    eb = pid * BE + tl.arange(0, BE)
    kk = tl.arange(0, BK)
    acc = tl.zeros([BM, BE], dtype=tl.float32)
    for k0 in range(0, H, BK):
        a = tl.load(h_ptr + m[:, None] * H + k0 + kk[None, :], mask=(m < rows)[:, None], other=0.0)
        b = tl.load(w_ptr + eb[:, None] * H + k0 + kk[None, :])
        acc += tl.dot(a, tl.trans(b))
    bias = tl.load(bias_ptr + eb).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-acc)) + bias[None, :]
    # Publish the scores at L2 and take a ticket; the last program to arrive
    # ranks all NE experts and resets the ticket for the next launch.
    tl.atomic_xchg(
        scores_ptr + m[:, None] * NE + eb[None, :], s, mask=(m < BR)[:, None], sem="relaxed"
    )
    tl.debug_barrier()
    ticket = tl.atomic_add(ticket_ptr, 1, sem="acq_rel")
    if ticket == NPROG - 1:
        r = tl.arange(0, BR)
        e = tl.arange(0, NE)
        live = r < rows
        sc = tl.load(scores_ptr + r[:, None] * NE + e[None, :], cache_modifier=".cg")
        sc = tl.where(live[:, None], sc, float("-inf"))
        cnt, mx = _scores_topk_union(sc, live, NE, TOPK)
        # One sort of a unique key per expert: rows selecting, then the max
        # score (order-preserving bits), then the lower ID first.
        bits = mx.to(tl.int32, bitcast=True)
        ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64) + 2147483648
        key = (cnt.to(tl.int64) << 41) + (ordered << 8) + (NE - 1 - e).to(tl.int64)
        key = tl.sort(key, dim=0, descending=True)
        ids = (NE - 1 - (key & (NE - 1))).to(tl.int32)
        tl.store(out_ids_ptr + e, ids, mask=e < KM)
        tl.atomic_xchg(ticket_ptr, 0, sem="relaxed")


def gemm_rank(hidden, weight, bias, scores, ticket, out_ids, rows: int) -> None:
    """Prediction in one launch on the current stream: ``hidden`` [rows, H]
    BF16 times the next layer's gate ``weight`` [256, H] BF16, scores, per-row
    top-8 union and ranking into ``out_ids`` (as ``rank_and_stats``).
    ``scores`` is an FP32 [MAX_ROWS, 256] scratch and ``ticket`` an int32 [1]
    zero; both are reused by every launch on the same stream."""
    _gemm_rank_kernel[(NUM_EXPERTS // GEMM_BLOCK,)](
        hidden,
        weight,
        bias,
        scores,
        ticket,
        out_ids,
        rows,
        H=hidden.shape[1],
        NE=NUM_EXPERTS,
        BR=MAX_ROWS,
        BM=16,
        BE=GEMM_BLOCK,
        BK=512,
        TOPK=TOP_K,
        KM=KMAX,
        num_warps=4,
        num_stages=4,
    )


@triton.jit
def _globaltimer(z):
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer; // $1", "=l,r", [z], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _nanosleep(ns):
    return tl.inline_asm_elementwise(
        "nanosleep.u32 $1; mov.u32 $0, 0;", "=r,r", [ns], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _delay_kernel(delay_ns):
    # One warp sleeps until delay_ns of global time have passed.
    z = tl.zeros([32], dtype=tl.int32)
    t0 = tl.max(_globaltimer(z), axis=0)
    t = t0
    while t - t0 < delay_ns:
        _nanosleep(z + 256)
        t = tl.max(_globaltimer(z), axis=0)


def delay(delay_ns: int) -> None:
    """Hold the current stream for about ``delay_ns`` nanoseconds with one
    warp (the work queued after it starts that much later)."""
    _delay_kernel[(1,)](delay_ns, num_warps=1)


@triton.jit
def _bulk_prefetch_l2(addr, nbytes):
    tl.inline_asm_elementwise(
        "cp.async.bulk.prefetch.L2.global [$1], $2; // dummy $0",
        "=r,l,r",
        [addr, nbytes],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _prefetch_experts_kernel(
    ids_ptr,
    w13_addr,
    w13_sf_addr,
    w2_addr,
    w2_sf_addr,
    K: tl.constexpr,
    CHUNK: tl.constexpr,
    C13: tl.constexpr,
    C13SF: tl.constexpr,
    C2: tl.constexpr,
    C2SF: tl.constexpr,
    NPROG: tl.constexpr,
    NT: tl.constexpr,
):
    # Chunks of one expert, in the order the routed kernel reads them:
    # FC1 weight, FC1 scales, FC2 weight, FC2 scales. Addresses are int64
    # byte addresses (the four tensors have different element types).
    CPE: tl.constexpr = C13 + C13SF + C2 + C2SF
    TOTAL: tl.constexpr = K * CPE
    ITERS: tl.constexpr = (TOTAL + NPROG * NT - 1) // (NPROG * NT)
    pid = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, NT).to(tl.int64)
    for it in range(ITERS):
        c = (it * NPROG + pid) * NT + lanes
        live = c < TOTAL
        c = tl.where(live, c, 0)
        slot = c // CPE
        off = c - slot * CPE
        ex = tl.load(ids_ptr + slot, mask=live, other=0).to(tl.int64)
        a13 = w13_addr + ex * (C13 * CHUNK) + off * CHUNK
        o1 = off - C13
        a13sf = w13_sf_addr + ex * (C13SF * CHUNK) + o1 * CHUNK
        o2 = o1 - C13SF
        a2 = w2_addr + ex * (C2 * CHUNK) + o2 * CHUNK
        o3 = o2 - C2
        a2sf = w2_sf_addr + ex * (C2SF * CHUNK) + o3 * CHUNK
        addr = tl.where(off < C13, a13, tl.where(o1 < C13SF, a13sf, tl.where(o2 < C2, a2, a2sf)))
        # Lanes past the end re-issue the first chunk, which is harmless.
        _bulk_prefetch_l2(addr, tl.full([NT], CHUNK, tl.int32))


def prefetch_experts(
    ids, w13, w13_sf, w2, w2_sf, k: int, chunk: int, ctas: int, threads: int = 32
) -> None:
    """Prefetch the first ``k`` experts of ``ids`` into L2 on the current
    stream. Every expert slice of the four tensors must be a whole number of
    ``chunk`` bytes."""

    def per_expert_chunks(t):
        nbytes = t[0].numel() * t.element_size()
        assert nbytes % chunk == 0, (tuple(t.shape), chunk)
        return nbytes // chunk

    _prefetch_experts_kernel[(ctas,)](
        ids,
        w13.data_ptr(),
        w13_sf.data_ptr(),
        w2.data_ptr(),
        w2_sf.data_ptr(),
        K=k,
        CHUNK=chunk,
        C13=per_expert_chunks(w13),
        C13SF=per_expert_chunks(w13_sf),
        C2=per_expert_chunks(w2),
        C2SF=per_expert_chunks(w2_sf),
        NPROG=ctas,
        NT=threads,
        num_warps=max(1, threads // 32),
        num_stages=1,
    )
