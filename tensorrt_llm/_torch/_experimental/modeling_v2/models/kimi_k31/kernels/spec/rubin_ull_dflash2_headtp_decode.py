# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels of the head-parallel DFlash2 drafter window: one rank's heads over the whole window,
read from its replicated ring (``ops/spec/rubin_ull_dflash2_headtp_ring.py``).

Request ``g`` at block start ``n`` reads the ``W + block - 1`` keys at positions ``n - W + 1 + i``
(``i`` the key index; ring row ``slot * R + (position mod R)``, negative positions absent). Its
``block x heads`` queries (``m = j * heads + h``, block row ``j``) share them: query ``m`` sees keys
``i`` in ``[j, j + W - 1]``. ``_partial`` takes one ``BN``-key split of one request for every query
at once (one pass over the FP8 keys, no per-row copy of the window) and writes the split's
unnormalized ``P V`` (FP32) and its ``(max, sum)`` (base 2); ``_combine`` merges the splits per query
by log-sum-exp and stores the BF16 result. Keys are the 576-wide latents, values their first 512
columns.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _partial(
    q_ptr,
    ring_ptr,
    slot_ptr,
    start_ptr,
    o_ptr,
    ml_ptr,
    scale,
    ring_rows,
    W: tl.constexpr,
    NKEYS: tl.constexpr,
    HEADS: tl.constexpr,
    NQ: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    LAT: tl.constexpr,
    ROPE: tl.constexpr,
):
    s = tl.program_id(0)
    vb = tl.program_id(1)
    g = tl.program_id(2)
    splits = tl.num_programs(0)
    D: tl.constexpr = LAT + ROPE
    m = tl.arange(0, BM)
    row = m // HEADS
    mq = m < NQ
    n = s * BN + tl.arange(0, BN)
    slot = tl.load(slot_ptr + g).to(tl.int64)
    start = tl.load(start_ptr + g).to(tl.int64)
    pos = start - (W - 1) + n
    live = (n < NKEYS) & (pos >= 0)
    krow = slot * ring_rows + tl.where(pos >= 0, pos % ring_rows, 0)
    kbase = ring_ptr + krow[:, None] * D
    qbase = q_ptr + (g * NQ + m[:, None]).to(tl.int64) * D
    acc = tl.zeros([BM, BN], tl.float32)
    for k0 in tl.static_range(0, LAT, BK):
        kk = k0 + tl.arange(0, BK)
        q = tl.load(qbase + kk[None, :], mask=mq[:, None], other=0.0)
        k = tl.load(kbase + kk[None, :], mask=live[:, None], other=0.0).to(tl.bfloat16)
        acc += tl.dot(q, tl.trans(k))
    rr = LAT + tl.arange(0, ROPE)
    q = tl.load(qbase + rr[None, :], mask=mq[:, None], other=0.0)
    k = tl.load(kbase + rr[None, :], mask=live[:, None], other=0.0).to(tl.bfloat16)
    acc += tl.dot(q, tl.trans(k))
    acc = acc * (scale * 1.4426950408889634)
    vis = (n[None, :] >= row[:, None]) & (n[None, :] <= row[:, None] + (W - 1)) & live[None, :]
    acc = tl.where(vis, acc, float("-inf"))
    mx = tl.max(acc, 1)
    mxs = tl.where(mx == float("-inf"), 0.0, mx)
    pr = tl.exp2(acc - mxs[:, None])
    lsum = tl.sum(pr, 1)
    vv = vb * BV + tl.arange(0, BV)
    v = tl.load(kbase + vv[None, :], mask=live[:, None], other=0.0).to(tl.bfloat16)
    o = tl.dot(pr.to(tl.bfloat16), v)
    part = (g * splits + s) * BM
    tl.store(o_ptr + (part + m[:, None]).to(tl.int64) * LAT + vv[None, :], o)
    if vb == 0:
        tl.store(ml_ptr + (part + m) * 2, mx)
        tl.store(ml_ptr + (part + m) * 2 + 1, lsum)


@triton.jit
def _combine(
    o_ptr,
    ml_ptr,
    out_ptr,
    vscale,
    NQ: tl.constexpr,
    BM: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
    BV: tl.constexpr,
    LAT: tl.constexpr,
):
    m = tl.program_id(0)
    vb = tl.program_id(1)
    g = tl.program_id(2)
    sp = tl.arange(0, BS)
    on = sp < SPLITS
    part = (g * SPLITS + sp) * BM + m
    mx = tl.load(ml_ptr + part * 2, mask=on, other=float("-inf"))
    ls = tl.load(ml_ptr + part * 2 + 1, mask=on, other=0.0)
    peak = tl.max(mx, 0)
    wgt = tl.where(mx == float("-inf"), 0.0, tl.exp2(mx - peak))
    den = tl.sum(wgt * ls, 0)
    vv = vb * BV + tl.arange(0, BV)
    o = tl.load(o_ptr + part[:, None].to(tl.int64) * LAT + vv[None, :], mask=on[:, None], other=0.0)
    res = tl.sum(o * wgt[:, None], 0) / den * vscale
    tl.store(out_ptr + (g * NQ + m).to(tl.int64) * LAT + vv, res.to(tl.bfloat16))


@triton.jit
def _partial_ranked(
    q_ptr,
    ring_ptr,
    slot_ptr,
    start_ptr,
    w_ptr,
    part_ptr,
    lse_ptr,
    scale,
    vscale,
    ring_rows,
    groups,
    w_sh,
    w_sk,
    w_sv,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
    HEADS: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    LAT: tl.constexpr,
    ROPE: tl.constexpr,
    PAGE: tl.constexpr,
    CP: tl.constexpr,
    V: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """helix_mla's partition of the window and its per-rank W_UV: program ``(r, h, g * SPLITS + s)`` takes the
    keys a helix CP rank ``r`` owns (positions in global pages ``g % CP == r``; at most two pages of the
    window) for head ``h``'s block rows, a FP32 softmax per row over them, BF16 P and V, and the ``s``-th of
    ``SPLITS`` equal ranges of the 512 latent columns: the normalized partial rounded to BF16 (as the MLA
    decode's output) ``BC`` columns at a time, each chunk times W_UV's ``[BC, V]`` rows into one FP32
    ``[rows, V]``. Stores that FP32 sum at split ``s`` (``_ranked_merge`` adds the splits in order and rounds
    once to BF16, helix_mla's per-rank partial after W_UV) and, from split 0, the natural-log log-sum-exp
    (``-inf`` with no visible key). Every split recomputes the scores. Launched with PDL: it waits for its
    predecessor before any load and then releases its own successor (the merge)."""
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    r = tl.program_id(0)
    h = tl.program_id(1)
    g = tl.program_id(2) // SPLITS
    s = tl.program_id(2) % SPLITS
    NQ: tl.constexpr = BLOCK * HEADS
    j = tl.arange(0, BR)
    jm = j < BLOCK
    i = tl.arange(0, 2 * PAGE)
    slot = tl.load(slot_ptr + g).to(tl.int64)
    start = tl.load(start_ptr + g).to(tl.int64)
    lo = tl.maximum(start - (W - 1), 0)
    p0 = lo // PAGE
    first = p0 + (r - p0 % CP + CP) % CP
    pos = (first + (i // PAGE) * CP) * PAGE + i % PAGE
    live = (pos >= lo) & (pos <= start + (BLOCK - 1))
    krow = slot * ring_rows + tl.where(live, pos % ring_rows, 0)
    kbase = ring_ptr + krow[:, None] * (LAT + ROPE)
    m = j * HEADS + h
    qbase = q_ptr + (g * NQ + m[:, None]).to(tl.int64) * (LAT + ROPE)
    acc = tl.zeros([BR, 2 * PAGE], tl.float32)
    for k0 in tl.static_range(0, LAT, BK):
        kk = k0 + tl.arange(0, BK)
        q = tl.load(qbase + kk[None, :], mask=jm[:, None], other=0.0)
        k = tl.load(kbase + kk[None, :], mask=live[:, None], other=0.0).to(tl.bfloat16)
        acc += tl.dot(q, tl.trans(k))
    rr = LAT + tl.arange(0, ROPE)
    q = tl.load(qbase + rr[None, :], mask=jm[:, None], other=0.0)
    k = tl.load(kbase + rr[None, :], mask=live[:, None], other=0.0).to(tl.bfloat16)
    acc += tl.dot(q, tl.trans(k))
    acc = acc * (scale * 1.4426950408889634)
    first_pos = start + j - (W - 1)
    vis = (
        (pos[None, :] >= first_pos[:, None]) & (pos[None, :] <= start + j[:, None]) & live[None, :]
    )
    acc = tl.where(vis, acc, float("-inf"))
    mx = tl.max(acc, 1)
    mxs = tl.where(mx == float("-inf"), 0.0, mx)
    p = tl.exp2(acc - mxs[:, None])
    lsum = tl.sum(p, 1)
    pr = p.to(tl.bfloat16)
    inv = tl.where(lsum > 0, vscale / lsum, 0.0)
    v = tl.arange(0, V)
    COLS: tl.constexpr = LAT // SPLITS
    out = tl.zeros([BR, V], tl.float32)
    for c0 in tl.static_range(0, COLS, BC):
        cc = s * COLS + c0 + tl.arange(0, BC)
        vt = tl.load(kbase + cc[None, :], mask=live[:, None], other=0.0).to(tl.bfloat16)
        o = (tl.dot(pr, vt) * inv[:, None]).to(tl.bfloat16)
        wt = tl.load(w_ptr + h * w_sh + cc[:, None].to(tl.int64) * w_sk + v[None, :] * w_sv)
        out += tl.dot(o, wt)
    base = (r * groups + g) * NQ + m
    # A BF16 part (one split) takes the rounded partial, the merge's own first rounding.
    tl.store(
        part_ptr + (s * (CP * groups * NQ) + base)[:, None].to(tl.int64) * V + v[None, :],
        out.to(part_ptr.dtype.element_ty),
        mask=jm[:, None],
    )
    if s == 0:
        lse = tl.where(lsum > 0, (mx + tl.log2(lsum)) * 0.6931471805599453, float("-inf"))
        tl.store(lse_ptr + base, lse, mask=jm)


def launch(query, ring, slots, starts, out, o_part, ml, scale, kv_scale, ring_rows, window, tiles):
    """``query [G * block, heads, 576]`` BF16, ``ring`` the layer's ring rows ``[slots * R, 576]``
    (FP8 or BF16), ``slots`` / ``starts`` ``[G]``, ``out [G * block, heads, 512]`` BF16;
    ``o_part`` / ``ml`` the split buffers of ``tiles = (BN, BV, BK, warps)``."""
    rows, heads, dim = query.shape
    groups = slots.shape[0]
    block = rows // groups
    nq = block * heads
    bm = max(16, triton.next_power_of_2(nq))
    bn, bv, bk, warps = tiles
    lat = out.shape[-1]
    nkeys = window + block - 1
    splits = triton.cdiv(nkeys, bn)
    _partial[(splits, lat // bv, groups)](
        query,
        ring,
        slots,
        starts,
        o_part,
        ml,
        scale * kv_scale,
        ring_rows,
        W=window,
        NKEYS=nkeys,
        HEADS=heads,
        NQ=nq,
        BM=bm,
        BN=bn,
        BV=bv,
        BK=bk,
        LAT=lat,
        ROPE=dim - lat,
        num_warps=warps,
    )
    _combine[(nq, lat // 128, groups)](
        o_part,
        ml,
        out,
        kv_scale,
        NQ=nq,
        BM=bm,
        SPLITS=splits,
        BS=triton.next_power_of_2(splits),
        BV=128,
        LAT=lat,
        num_warps=4,
    )


def split_shapes(groups, block, heads, window, tiles, lat=512):
    """``(o_part, ml)`` shapes of :func:`launch` for ``groups`` requests."""
    bm = max(16, triton.next_power_of_2(block * heads))
    splits = triton.cdiv(window + block - 1, tiles[0])
    return (groups * splits * bm, lat), (groups * splits * bm, 2)


def launch_ranked(
    query,
    ring,
    slots,
    starts,
    v_b_t,
    part,
    lse,
    scale,
    kv_scale,
    ring_rows,
    window,
    page,
    cp,
    bc=128,
    bk=64,
    warps=8,
    splits=2,
    maxnreg=None,
):
    """:func:`_partial_ranked` over ``cp`` page groups, this rank's heads and ``splits`` ranges of the latent
    columns: ``v_b_t`` ``[heads, 512, V]`` BF16 (any strides: the drafter's is a transposed view); ``part``
    flat FP32 (or BF16 at one split: the partials rounded) ``[>= splits * cp * rows * heads, V]`` gets split
    ``s`` of group ``r``'s partials after W_UV at
    rows ``s * cp * rows * heads + [r * rows * heads, (r + 1) * rows * heads)`` (``m = j * heads + h`` within
    a request), ``lse`` flat FP32 ``[>= cp * rows * heads]`` their log-sum-exps. ``maxnreg``: ptxas's register
    cap a thread (None: the compiler's choice)."""
    rows, heads, dim = query.shape
    groups = slots.shape[0]
    block = rows // groups
    assert part.element_size() == 4 or splits == 1, "BF16 partials hold one split"
    lat = v_b_t.shape[1]
    assert lat % (splits * bc) == 0, f"{splits} splits of {lat} latent columns in chunks of {bc}"
    _partial_ranked[(cp, heads, groups * splits)](
        query,
        ring,
        slots,
        starts,
        v_b_t,
        part,
        lse,
        scale * kv_scale,
        kv_scale,
        ring_rows,
        groups,
        v_b_t.stride(0),
        v_b_t.stride(1),
        v_b_t.stride(2),
        W=window,
        BLOCK=block,
        HEADS=heads,
        BR=max(16, triton.next_power_of_2(block)),
        BK=bk,
        BC=bc,
        LAT=lat,
        ROPE=dim - lat,
        PAGE=page,
        CP=cp,
        V=v_b_t.shape[-1],
        SPLITS=splits,
        num_warps=warps,
        launch_pdl=True,
        **({} if maxnreg is None else {"maxnreg": maxnreg}),
    )


@triton.jit
def _ranked_merge(
    part_ptr,
    lse_ptr,
    out_ptr,
    groups,
    CP: tl.constexpr,
    BLOCK: tl.constexpr,
    HEADS: tl.constexpr,
    V: tl.constexpr,
    SPLITS: tl.constexpr,
    VB: tl.constexpr,
):
    """Block row ``q`` of request ``q // BLOCK``, head ``h``, value columns ``[VB * program_id(2), + VB)``: each
    page group's partial is its ``SPLITS`` FP32 column-range sums added in split order and rounded once to BF16;
    the groups' partials are merged by log-sum-exp in FP32, rounded once.
    :func:`..ops.spec.rubin_ull_dflash2_headtp_decode.headtp_ranked_merge`'s arithmetic step for step: ``__nv_expf`` weights, products, the sums over the groups in torch's order for a
    16-long outer reduction (four strided accumulators, then ((a0 + a1) + a2) + a3), a correctly rounded
    quotient. Launched with PDL: it waits for :func:`_partial_ranked` before any load and then releases its own
    successor (the o_proj GEMM)."""
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    q = tl.program_id(0)
    h = tl.program_id(1)
    NQ: tl.constexpr = BLOCK * HEADS
    g = q // BLOCK
    m = (q % BLOCK) * HEADS + h
    v = tl.program_id(2) * VB + tl.arange(0, VB)
    peak = float("-inf")
    for r in tl.static_range(CP):
        peak = tl.maximum(peak, tl.load(lse_ptr + (r * groups + g) * NQ + m))
    peak = tl.where(peak > float("-inf"), peak, 0.0)
    a0 = tl.zeros([VB], tl.float32)
    a1 = tl.zeros([VB], tl.float32)
    a2 = tl.zeros([VB], tl.float32)
    a3 = tl.zeros([VB], tl.float32)
    d0 = 0.0
    d1 = 0.0
    d2 = 0.0
    d3 = 0.0
    stride = CP * groups * NQ
    for r in tl.static_range(CP):
        row = (r * groups + g) * NQ + m
        lse = tl.load(lse_ptr + row)
        live = lse > float("-inf")
        wgt = tl.where(live, libdevice.exp(tl.where(live, lse, 0.0) - peak), 0.0)
        part = tl.load(part_ptr + row.to(tl.int64) * V + v)
        for s in tl.static_range(1, SPLITS):
            part += tl.load(part_ptr + (s * stride + row).to(tl.int64) * V + v)
        term = tl.where(live, wgt * part.to(tl.bfloat16).to(tl.float32), 0.0)
        if r % 4 == 0:
            a0 += term
            d0 += wgt
        elif r % 4 == 1:
            a1 += term
            d1 += wgt
        elif r % 4 == 2:
            a2 += term
            d2 += wgt
        else:
            a3 += term
            d3 += wgt
    den = tl.maximum(((d0 + d1) + d2) + d3, 1.1754943508222875e-38)
    merged = tl.math.div_rn(((a0 + a1) + a2) + a3, den)
    tl.store(out_ptr + (g * NQ + m).to(tl.int64) * V + v, merged.to(tl.bfloat16))


def launch_ranked_merge(part, lse, out, groups, block, heads, cp, warps=4, splits=2, vsplit=4):
    """:func:`_ranked_merge`: ``part`` / ``lse`` as :func:`launch_ranked` wrote them (``splits`` of them),
    ``out`` ``[groups * block, heads * V]`` BF16. ``vsplit`` programs a row and head share its V columns (``warps``
    between them); every output is computed alike."""
    _ranked_merge[(groups * block, heads, vsplit)](
        part,
        lse,
        out,
        groups,
        CP=cp,
        BLOCK=block,
        HEADS=heads,
        V=part.shape[-1],
        SPLITS=splits,
        VB=part.shape[-1] // vsplit,
        num_warps=max(1, warps // vsplit),
        launch_pdl=True,
    )
