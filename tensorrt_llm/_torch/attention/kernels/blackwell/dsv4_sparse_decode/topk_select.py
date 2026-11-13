# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selection constants and the bin scale of the threshold band."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, Uint64

from . import topk_select_lib as L

NB_SEL = L.NB_SEL
QUADC = L.QUADC
NBS = L.NBS


@cute.jit
def bin_scale(lo: Float32, hi: Float32):
    wv = (hi - lo) * Float32(1.0 / NB_SEL)
    sc = Float32(0.0)
    if wv > Float32(0.0):
        if wv < Float32(3e38):
            sc = cute.arch.rcp_approx(wv)
    return sc


@cute.jit
def _pst(prof, slot, tid):
    if cutlass.const_expr(prof is not None):
        if tid == Int32(0):
            cutlass.inttoptr(
                prof + cutlass.Int64(slot) * cutlass.Int64(8), mem_space=1, dtype=cutlass.Int64
            ).store(cutlass.Int64(L.gtimer()))


@cute.jit
def select_slots(
    lst,
    C,
    K,
    out_s,
    s_hist,
    s_ck,
    s_cs,
    s_res,
    s_red,
    s_wc,
    tid,
    nthr: cutlass.Constexpr,
    bar_id: cutlass.Constexpr,
    rng_lo,
    rng_hi,
    hist_ready,
    prof=None,
):
    NW = nthr // 32
    lane = tid & Int32(31)
    wid = tid >> Int32(5)
    lm = Int32(cute.arch.lanemask_lt())
    nout = C
    if nout > K:
        nout = K
    mret = Int32(0)
    info = Int32(0)
    Bret = Int32(-1)
    if C <= K:
        i = tid
        while i < C:
            out_s[i] = i
            i = i + Int32(nthr)
        if tid < Int32(NB_SEL):
            for z_ in cutlass.range_constexpr(NB_SEL // nthr if NB_SEL >= nthr else 1):
                s_hist[tid + Int32(z_ * nthr)] = Int32(0)
        L.bar(bar_id, nthr)
    else:
        lo = rng_lo
        hi = rng_hi
        sc = bin_scale(lo, hi)
        if hist_ready == Int32(0):
            for z_ in cutlass.range_constexpr(NB_SEL // nthr if NB_SEL >= nthr else 1):
                if tid + Int32(z_ * nthr) < Int32(NB_SEL):
                    s_hist[tid + Int32(z_ * nthr)] = Int32(0)
            L.bar(bar_id, nthr)
            i4 = (tid >> Int32(5)) * Int32(128) + lane
            while i4 < C:
                bq_ = [Int32(0)] * 4
                for q in cutlass.range_constexpr(4):
                    icl = i4 + Int32(32 * q)
                    if icl > C - Int32(1):
                        icl = C - Int32(1)
                    bq_[q] = L._vbin(Uint64(lst[icl]), lo, sc, NB_SEL)
                for q in cutlass.range_constexpr(4):
                    if i4 + Int32(32 * q) < C:
                        cute.arch.red(
                            s_hist.iterator + bq_[q],
                            Int32(1),
                            op="add",
                            dtype=Int32,
                            sem="relaxed",
                            scope="cta",
                        )
                i4 = i4 + Int32(nthr * 4)
        L.bar(bar_id, nthr)
        _pst(prof, 58, tid)
        L.find_cross(s_hist, K, tid, s_res, NB_SEL)
        L.bar(bar_id, nthr)
        _pst(prof, 59, tid)
        Bb = s_res[L.RES_B]
        above = s_res[L.RES_ABOVE]
        m = s_res[L.RES_M]
        Bret = Bb
        need = K - above
        mret = m
        whole = Int32(0)
        if need >= m:
            whole = Int32(1)
        keepc = Int32(0)
        if whole == Int32(0):
            if m <= Int32(QUADC):
                keepc = Int32(1)
        Bw = Bb
        if whole != Int32(0):
            Bw = Bb - Int32(1)
        for z_ in cutlass.range_constexpr(NB_SEL // nthr if NB_SEL >= nthr else 1):
            if tid + Int32(z_ * nthr) < Int32(NB_SEL):
                s_hist[tid + Int32(z_ * nthr)] = Int32(0)
        chunk = (C + Int32(nthr * 4) - Int32(1)) // Int32(nthr * 4) * Int32(128)
        c0 = wid * chunk
        c1 = c0 + chunk
        if c1 > C:
            c1 = C
        cw = Int32(0)
        cc = Int32(0)
        i4 = c0 + lane
        while i4 < c1:
            for q in cutlass.range_constexpr(4):
                icl = i4 + Int32(32 * q)
                if icl > C - Int32(1):
                    icl = C - Int32(1)
                bb = L._vbin(Uint64(lst[icl]), lo, sc, NB_SEL)
                if i4 + Int32(32 * q) < c1:
                    if bb > Bw:
                        cw = cw + Int32(1)
                    elif bb == Bb:
                        cc = cc + Int32(1)
            i4 = i4 + Int32(128)
        wcw = Int32(cute.arch.warp_redux_sync(cw, "add"))
        wcc = Int32(cute.arch.warp_redux_sync(cc, "add"))
        if lane == Int32(0):
            s_wc[wid] = wcw
            s_wc[Int32(NW) + wid] = wcc
        L.bar(bar_id, nthr)
        _pst(prof, 60, tid)
        pw = Int32(0)
        pc = Int32(0)
        for w_ in cutlass.range_constexpr(NW):
            if Int32(w_) < wid:
                pw = pw + Int32(s_wc[w_])
                pc = pc + Int32(s_wc[NW + w_])
        base0 = c0
        while base0 < c1:
            ed = []
            for q in cutlass.range_constexpr(4):
                icl = base0 + Int32(32 * q) + lane
                if icl > C - Int32(1):
                    icl = C - Int32(1)
                ed.append(Uint64(lst[icl]))
            for q in cutlass.range_constexpr(4):
                sl = base0 + Int32(32 * q) + lane
                fw = Int32(0)
                fc = Int32(0)
                if sl < c1:
                    bd_ = L._vbin(ed[q], lo, sc, NB_SEL)
                    if bd_ > Bw:
                        fw = Int32(1)
                    elif bd_ == Bb:
                        fc = Int32(1)
                mw = Int32(cute.arch.vote_ballot_sync(fw != Int32(0)))
                mc = Int32(cute.arch.vote_ballot_sync(fc != Int32(0)))
                if fw != Int32(0):
                    out_s[pw + Int32(cute.arch.popc(mw & lm))] = sl
                if fc != Int32(0):
                    if keepc != Int32(0):
                        pcs = pc + Int32(cute.arch.popc(mc & lm))
                        s_ck[pcs] = L.entry_key(ed[q])
                        s_cs[pcs] = sl
                pw = pw + Int32(cute.arch.popc(mw))
                pc = pc + Int32(cute.arch.popc(mc))
            base0 = base0 + Int32(128)
        L.bar(bar_id, nthr)
        _pst(prof, 61, tid)
        if whole == Int32(0):
            if m <= Int32(32):
                info = Int32(1)
                if wid == Int32(0):
                    kq = Uint64(0)
                    sq = Int32(0)
                    if lane < m:
                        kq = Uint64(s_ck[lane])
                        sq = s_cs[lane]
                    khi = Uint32(kq >> Uint64(32))
                    klo = Uint32(kq & Uint64(4294967295))
                    r_ = Int32(0)
                    for j in cutlass.range_constexpr(32):
                        ohi = cute.arch.shuffle_sync(khi, Int32(j))
                        olo = cute.arch.shuffle_sync(klo, Int32(j))
                        if Int32(j) < m:
                            if ohi > khi:
                                r_ = r_ + Int32(1)
                            elif ohi == khi:
                                if olo > klo:
                                    r_ = r_ + Int32(1)
                    if lane < m:
                        if r_ < need:
                            out_s[above + r_] = sq
                L.bar(bar_id, nthr)
            elif m <= Int32(QUADC):
                info = Int32(1)
                i = tid
                while i < m:
                    ki = Uint64(s_ck[i])
                    r_ = Int32(0)
                    jq = Int32(0)
                    while jq < m:
                        if Uint64(s_ck[jq]) > ki:
                            r_ = r_ + Int32(1)
                        jq = jq + Int32(1)
                    if r_ < need:
                        out_s[above + r_] = s_cs[i]
                    i = i + Int32(nthr)
                L.bar(bar_id, nthr)
            else:
                info = Int32(2)
                pref = Uint64(0)
                plen = Int32(0)
                needr = need
                done = Int32(0)
                while done == Int32(0):
                    shift = Int32(56) - plen
                    if tid < Int32(NBS):
                        s_hist[tid] = Int32(0)
                    if cutlass.const_expr(nthr < NBS):
                        for z_ in cutlass.range_constexpr(1, NBS // nthr):
                            s_hist[tid + Int32(z_ * nthr)] = Int32(0)
                    L.bar(bar_id, nthr)
                    i = tid
                    while i < C:
                        e2_ = Uint64(lst[i])
                        if L._vbin(e2_, lo, sc, NB_SEL) == Bb:
                            k2 = L.entry_key(e2_)
                            okp = Int32(1)
                            if plen > Int32(0):
                                if k2 >> Uint64(Int32(64) - plen) != pref:
                                    okp = Int32(0)
                            if okp != Int32(0):
                                dg = Int32(Uint32(k2 >> Uint64(shift) & Uint64(255)))
                                cute.arch.red(
                                    s_hist.iterator + dg,
                                    Int32(1),
                                    op="add",
                                    dtype=Int32,
                                    sem="relaxed",
                                    scope="cta",
                                )
                        i = i + Int32(nthr)
                    L.bar(bar_id, nthr)
                    L.scan_cross0(
                        s_hist, needr, tid, s_res, needr, needr, s_hist, nb=NBS, zero=True
                    )
                    L.bar(bar_id, nthr)
                    dB = s_res[L.RES_B]
                    dab = s_res[L.RES_ABOVE]
                    dm = s_res[L.RES_M]
                    L.bar(bar_id, nthr)
                    pref = pref << Uint64(8) | Uint64(dB)
                    plen = plen + Int32(8)
                    needr = needr - dab
                    if needr >= dm:
                        done = Int32(1)
                    if plen >= Int32(64):
                        done = Int32(1)
                thr = pref << Uint64(Int32(64) - plen)
                if plen >= Int32(64):
                    thr = pref
                nw2 = Int32(0)
                r0 = c0
                while r0 < c1:
                    ii = r0 + lane
                    f2_ = Int32(0)
                    if ii < c1:
                        e3_ = Uint64(lst[ii])
                        if L._vbin(e3_, lo, sc, NB_SEL) == Bb:
                            if L.entry_key(e3_) >= thr:
                                f2_ = Int32(1)
                    nw2 = nw2 + Int32(cute.arch.popc(cute.arch.vote_ballot_sync(f2_ != Int32(0))))
                    r0 = r0 + Int32(32)
                ex2, tot2 = L._warp_prefix(nw2, s_wc, tid, nthr, bar_id)
                p2 = above + ex2
                r0 = c0
                while r0 < c1:
                    ii = r0 + lane
                    f2_ = Int32(0)
                    if ii < c1:
                        e4 = Uint64(lst[ii])
                        if L._vbin(e4, lo, sc, NB_SEL) == Bb:
                            if L.entry_key(e4) >= thr:
                                f2_ = Int32(1)
                    m2 = Int32(cute.arch.vote_ballot_sync(f2_ != Int32(0)))
                    if f2_ != Int32(0):
                        out_s[p2 + Int32(cute.arch.popc(m2 & lm))] = ii
                    p2 = p2 + Int32(cute.arch.popc(m2))
                    r0 = r0 + Int32(32)
                L.bar(bar_id, nthr)
                if tid < Int32(NBS):
                    s_hist[tid] = Int32(0)
                if cutlass.const_expr(nthr < NBS):
                    for z_ in cutlass.range_constexpr(1, NBS // nthr):
                        s_hist[tid + Int32(z_ * nthr)] = Int32(0)
                L.bar(bar_id, nthr)
    _pst(prof, 62, tid)
    return (nout, mret, info, Bret)
