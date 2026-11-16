# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent indexer kernel: FP8 indexer logits, exact top-512 selection and KV-row mapping."""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as xcuda
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from . import attention_helpers as A
from . import cache_policy as CP
from . import selection_lib as L

_RND_RN = "rn"
NEG_INF = -math.inf
K_TOP = 512
RING = 256
RING_CHUNK = 64
MAIN_CAP = 2048
TRIG = 1280
PMAX_SMEM = 4
PMAX = 8
SEGW = 520
MARK_HI = 4294967295
SEL_BAR = 2
ROW_BAR = 4
END_BAR = 5
SEL_NTHR = 128
ATTQ = 8
DQ_CAP = 128
DEFER_LONG = 512
ST_SCHED = 1
ST_ROW = 2
ST_DQFULL = 4


@dsl_user_op
def _consume_f32(x, *, loc=None, ip=None):
    return Float32(
        _llvm.inline_asm(
            _T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _warp_atomic_i32(ptr, lane):
    # One atomic poll per warp; propagate the acquire edge to every reader.
    value = Int32(0)
    if lane == Int32(0):
        value = cute.arch.load(ptr, Int32, sem="acquire", scope="cta")
    value = cute.arch.shuffle_sync(value, 0)
    cute.arch.sync_warp()
    return value


@cute.jit
def _warp_atomic_u32(ptr, lane):
    # The packed tag and conservative floor are the complete atomic payload.
    value = Uint32(0)
    if lane == Int32(0):
        value = cute.arch.load(ptr, Uint32, sem="relaxed", scope="cta")
    return cute.arch.shuffle_sync(value, 0)


@cute.jit
def _pack_floor_key(key, seg):
    # At most 1024 segments per CTA in scope, so the epoch fits in ten bits.
    # Rounding an ordered key down can only admit additional candidates.
    return (Uint32(key) & Uint32(0xFFFFFC00)) | Uint32(seg)


@cute.jit
def _wait_chunk_empty(empty_ptr, stage, phase, stall, prof: cutlass.Constexpr = False):
    ready = prims.mbarrier_wait_parity(
        empty_ptr + stage, phase, prims.MBarrierWait.TEST, scope="cta", order="acquire"
    )
    ready = cute.arch.vote_all_sync(ready)
    if not ready:
        t_s = Int64(0)
        if cutlass.const_expr(prof):
            t_s = cutlass.Int64(cute.arch.globaltimer())
        ready = cutlass.Boolean(False)
        while not ready:
            ready = prims.mbarrier_try_wait_parity(
                empty_ptr + stage, phase, scope="cta", order="acquire"
            )
            ready = cute.arch.vote_all_sync(ready)
        if cutlass.const_expr(prof):
            stall = stall + (cutlass.Int64(cute.arch.globaltimer()) - t_s)
    return stall


@cute.jit
def _push_ring(
    hit,
    entry,
    head,
    s_ring,
    ring_off,
    empty_ptr,
    full_ptr,
    lane,
    lt_mask,
    stall,
    prof: cutlass.Constexpr = False,
):
    mask = Int32(cute.arch.vote_ballot_sync(hit))
    if mask != Int32(0):
        n = Int32(cute.arch.popc(mask))
        offset = head & Int32(RING_CHUNK - 1)
        stage = (head >> Int32(6)) & Int32(RING // RING_CHUNK - 1)
        if offset == Int32(0):
            phase = ((head >> Int32(8)) & Int32(1)) ^ Int32(1)
            stall = _wait_chunk_empty(empty_ptr, stage, phase, stall, prof)
        first = Int32(RING_CHUNK) - offset
        if first > n:
            first = n
        rank = Int32(cute.arch.popc(mask & lt_mask))
        if hit & (rank < first):
            slot = (head + rank) & Int32(RING - 1)
            s_ring[ring_off + slot] = entry
        head = head + first
        if offset + first == Int32(RING_CHUNK):
            # All 32 writers publish their own stores, including earlier pushes.
            prims.mbarrier_arrive(full_ptr + stage, scope="cta", relaxed=False)
        if first < n:
            # Publish the completed chunk before waiting on the spill chunk.
            stage = (head >> Int32(6)) & Int32(RING // RING_CHUNK - 1)
            phase = ((head >> Int32(8)) & Int32(1)) ^ Int32(1)
            stall = _wait_chunk_empty(empty_ptr, stage, phase, stall, prof)
            if hit & (rank >= first):
                slot = (head + rank - first) & Int32(RING - 1)
                s_ring[ring_off + slot] = entry
            head = head + n - first
    return (head, stall)


@cute.jit
def _push_marker(seg_ord, head, s_ring, ring_off, empty_ptr, full_ptr, lane):
    offset = head & Int32(RING_CHUNK - 1)
    stage = (head >> Int32(6)) & Int32(RING // RING_CHUNK - 1)
    if offset == Int32(0):
        phase = ((head >> Int32(8)) & Int32(1)) ^ Int32(1)
        _wait_chunk_empty(empty_ptr, stage, phase, Int64(0))
    chunk_base = head - offset
    # The consumer reads all 64 slots; initialize the complete marker suffix.
    if lane >= offset:
        s_ring[ring_off + ((chunk_base + lane) & Int32(RING - 1))] = Uint64(MARK_HI) << Uint64(
            32
        ) | Uint64(Uint32(seg_ord))
    if lane + Int32(32) >= offset:
        s_ring[ring_off + ((chunk_base + lane + Int32(32)) & Int32(RING - 1))] = Uint64(
            MARK_HI
        ) << Uint64(32) | Uint64(Uint32(seg_ord))
    prims.mbarrier_arrive(full_ptr + stage, scope="cta", relaxed=False)
    return chunk_base + Int32(RING_CHUNK)


@cute.jit
def _floor_of(tw_ptr, seg_ord, lane):
    tw = _warp_atomic_u32(tw_ptr, lane)
    t = Float32(NEG_INF)
    if (tw & Uint32(1023)) == Uint32(seg_ord):
        key = tw & Uint32(0xFFFFFC00)
        # Zero initialization and rounded -inf lie in the negative-NaN range.
        if key >= Uint32(0x007FFFFF):
            t = L.invkey(key)
    return t


@cute.jit
def _drain_ring(
    r,
    tail_r,
    done_r,
    kth_key,
    s_ring,
    full_ptr,
    empty_ptr,
    sMain,
    cnt_ptr,
    lane,
    lt_mask,
    kmin_ptr,
    kmax_ptr,
):
    used = Int32(0)
    prog = Int32(0)
    if done_r == Int32(0):
        stage = (tail_r >> Int32(6)) & Int32(RING // RING_CHUNK - 1)
        phase = (tail_r >> Int32(8)) & Int32(1)
        # A selector warp services two rings, so probe rather than park on one.
        ready = prims.mbarrier_wait_parity(
            full_ptr + stage, phase, prims.MBarrierWait.TEST, scope="cta", order="acquire"
        )
        ready = cute.arch.vote_all_sync(ready)
        if ready:
            e0 = Uint64(s_ring[r * Int32(RING) + ((tail_r + lane) & Int32(RING - 1))])
            e1 = Uint64(s_ring[r * Int32(RING) + ((tail_r + lane + Int32(32)) & Int32(RING - 1))])
            ism0 = Uint32(e0 >> Uint64(32)) == Uint32(MARK_HI)
            ism1 = Uint32(e1 >> Uint64(32)) == Uint32(MARK_HI)
            mm0 = Int32(cute.arch.vote_ballot_sync(ism0))
            mm1 = Int32(cute.arch.vote_ballot_sync(ism1))
            # Markers occupy a suffix and every slot in that suffix is initialized.
            nuse = Int32(RING_CHUNK) - Int32(cute.arch.popc(mm0)) - Int32(cute.arch.popc(mm1))
            if (mm0 | mm1) != Int32(0):
                done_r = Int32(1)
            fk0 = L.fkey_bits(L.entry_bits(e0))
            fk1 = L.fkey_bits(L.entry_bits(e1))
            # _select_role always passes a score-only floor (low 32 bits zero).
            floor = Uint32(kth_key >> Uint64(32))
            keep0 = (not ism0) & (fk0 >= floor)
            keep1 = (not ism1) & (fk1 >= floor)
            km0 = Int32(cute.arch.vote_ballot_sync(keep0))
            km1 = Int32(cute.arch.vote_ballot_sync(keep1))
            nk0 = Int32(cute.arch.popc(km0))
            nk = nk0 + Int32(cute.arch.popc(km1))
            fmn = Uint32(4294967295)
            fmx = Uint32(0)
            if keep0:
                fmn = fk0
                fmx = fk0
            if keep1:
                if fk1 < fmn:
                    fmn = fk1
                if fk1 > fmx:
                    fmx = fk1
            fmn = L.warp_min_u32(fmn)
            fmx = L.warp_max_u32(fmx)
            base = Int32(0)
            if lane == Int32(0):
                if nk > Int32(0):
                    base = Int32(cute.arch.atomic_add(cnt_ptr, nk, sem="relaxed", scope="cta"))
                    cute.arch.atomic_min(kmin_ptr, fmn, sem="relaxed", scope="cta")
                    cute.arch.atomic_max(kmax_ptr, fmx, sem="relaxed", scope="cta")
            base = cute.arch.shuffle_sync(base, 0)
            if keep0:
                sMain[base + Int32(cute.arch.popc(km0 & lt_mask))] = e0
            if keep1:
                sMain[base + nk0 + Int32(cute.arch.popc(km1 & lt_mask))] = e1
            used = nuse
            tail_r = tail_r + Int32(RING_CHUNK)
            # Every reader releases its own load before this chunk can be reused.
            prims.mbarrier_arrive(empty_ptr + stage, scope="cta", relaxed=False)
            prog = Int32(1)
    return (tail_r, done_r, used, prog)


EPT = MAIN_CAP // SEL_NTHR
KEEP_CAP = 1280


@cute.jit
def _compact_approx(
    sMain,
    C,
    sHist,
    sRes,
    cnt_ptr,
    kmin_ptr,
    kmax_ptr,
    kmin2_ptr,
    sScal,
    tid,
    nthr: cutlass.Constexpr,
    bar_id: cutlass.Constexpr,
    prof=None,
    keep_cap: cutlass.Constexpr = KEEP_CAP,
    nb: cutlass.Constexpr = 1024,
    tw_ptr=None,
    seg=None,
    tiebin: cutlass.Constexpr = False,
    sTb=None,
):
    lane = tid & Int32(31)
    tp0 = Int64(0)
    if cutlass.const_expr(prof is not None):
        tp0 = cutlass.Int64(cute.arch.globaltimer())
    lo = L.invkey(Uint32(cute.arch.load(kmin_ptr, Uint32, sem="relaxed", scope="cta")))
    hi = L.invkey(Uint32(cute.arch.load(kmax_ptr, Uint32, sem="relaxed", scope="cta")))
    wv = (hi - lo) * Float32(1.0 / nb)
    sc = Float32(0.0)
    if wv > Float32(0.0):
        if wv < Float32(3e38):
            sc = cute.arch.rcp_approx(wv)
    for z_ in cutlass.range_constexpr(max(1, nb // nthr)):
        if tid + Int32(z_ * nthr) < Int32(nb):
            sHist[tid + Int32(z_ * nthr)] = Int32(0)
    if tid == Int32(0):
        sScal[24] = Int32(0)
        cute.arch.store(kmin2_ptr, Uint32(4294967295), sem="relaxed", scope="cta")
    L.bar(bar_id, nthr)
    if cutlass.const_expr(prof is not None):
        if tid == Int32(0):
            tp1 = cutlass.Int64(cute.arch.globaltimer())
            prof[0] = prof[0] + (tp1 - tp0)
            tp0 = tp1
    ev = [Uint64(0)] * EPT
    for k in cutlass.range_constexpr(EPT):
        i = tid + Int32(k * nthr)
        if i < C:
            ev[k] = Uint64(sMain[i])
            cute.arch.atomic_add(
                sHist.iterator + L._vbin(ev[k], lo, sc, nb), Int32(1), sem="relaxed", scope="cta"
            )
    L.bar(bar_id, nthr)
    if cutlass.const_expr(prof is not None):
        if tid == Int32(0):
            tp1 = cutlass.Int64(cute.arch.globaltimer())
            prof[1] = prof[1] + (tp1 - tp0)
            tp0 = tp1
    L.find_cross(sHist, Int32(K_TOP), tid, sRes, nb)
    L.bar(bar_id, nthr)
    if cutlass.const_expr(prof is not None):
        if tid == Int32(0):
            tp1 = cutlass.Int64(cute.arch.globaltimer())
            prof[2] = prof[2] + (tp1 - tp0)
            tp0 = tp1
    B = sRes[L.RES_B]
    kept = sRes[L.RES_ABOVE] + sRes[L.RES_M]
    mode2 = Int32(0)
    idx_thr = Int32(2147483647)
    if cutlass.const_expr(tiebin):
        if kept > Int32(keep_cap):
            above = sRes[L.RES_ABOVE]
            if tid == Int32(0):
                sTb[0] = Uint32(4294967295)
                sTb[1] = Uint32(0)
                sTb[2] = Uint32(4294967295)
                sTb[3] = Uint32(0)
            for z_ in cutlass.range_constexpr(max(1, nb // nthr)):
                if tid + Int32(z_ * nthr) < Int32(nb):
                    sHist[tid + Int32(z_ * nthr)] = Int32(0)
            L.bar(bar_id, nthr)
            fmn_b = Uint32(4294967295)
            fmx_b = Uint32(0)
            imn_b = Uint32(4294967295)
            imx_b = Uint32(0)
            for k in cutlass.range(EPT, unroll=1):
                i = tid + k * Int32(nthr)
                if i < C:
                    e = Uint64(sMain[i])
                    if L._vbin(e, lo, sc, nb) == B:
                        fk = L.fkey_bits(L.entry_bits(e))
                        ix = Uint32(e >> Uint64(32))
                        if fk < fmn_b:
                            fmn_b = fk
                        if fk > fmx_b:
                            fmx_b = fk
                        if ix < imn_b:
                            imn_b = ix
                        if ix > imx_b:
                            imx_b = ix
            fmn_b = L.warp_min_u32(fmn_b)
            fmx_b = L.warp_max_u32(fmx_b)
            imn_b = L.warp_min_u32(imn_b)
            imx_b = L.warp_max_u32(imx_b)
            if lane == Int32(0):
                cute.arch.atomic_min(sTb.iterator, fmn_b, sem="relaxed", scope="cta")
                cute.arch.atomic_max(sTb.iterator + 1, fmx_b, sem="relaxed", scope="cta")
                cute.arch.atomic_min(sTb.iterator + 2, imn_b, sem="relaxed", scope="cta")
                cute.arch.atomic_max(sTb.iterator + 3, imx_b, sem="relaxed", scope="cta")
            L.bar(bar_id, nthr)
            fmn_b = Uint32(sTb[0])
            fmx_b = Uint32(sTb[1])
            imn_b = Uint32(sTb[2])
            imx_b = Uint32(sTb[3])
            if fmn_b == fmx_b:
                mode2 = Int32(1)
                span = imx_b - imn_b
                sh = Uint32(0)
                while span >> sh >= Uint32(nb):
                    sh = sh + Uint32(1)
                for k in cutlass.range(EPT, unroll=1):
                    i = tid + k * Int32(nthr)
                    if i < C:
                        e = Uint64(sMain[i])
                        if L._vbin(e, lo, sc, nb) == B:
                            b2 = Int32(imx_b - Uint32(e >> Uint64(32)) >> sh)
                            cute.arch.atomic_add(
                                sHist.iterator + b2, Int32(1), sem="relaxed", scope="cta"
                            )
                L.bar(bar_id, nthr)
                L.find_cross(sHist, Int32(K_TOP) - above, tid, sRes, nb)
                L.bar(bar_id, nthr)
                B2 = sRes[L.RES_B]
                kept = above + sRes[L.RES_ABOVE] + sRes[L.RES_M]
                idx_thr = Int32(imx_b - (Uint32(B2) << sh))
    if cutlass.const_expr(tw_ptr is not None):
        if tid < Int32(8):
            if (kept <= Int32(keep_cap)) & (mode2 == Int32(0)):
                if B >= Int32(2):
                    te = lo + Float32(B - Int32(2)) * wv
                    mag = cute.math.max(abs(lo), abs(hi))
                    if wv > mag * Float32(1e-05):
                        if te > lo:
                            cute.arch.atomic_exch(
                                tw_ptr + tid,
                                _pack_floor_key(L.fkey(te), seg),
                                sem="relaxed",
                                scope="cta",
                            )
    tf = Uint32(0)
    if kept <= Int32(keep_cap):
        lm = Int32(cute.arch.lanemask_lt())
        fmn = Uint32(4294967295)
        kmask = [Int32(0)] * EPT
        wtot = Int32(0)
        for k in cutlass.range_constexpr(EPT):
            i = tid + Int32(k * nthr)
            kp = False
            bb = Int32(-1)
            if i < C:
                bb = L._vbin(ev[k], lo, sc, nb)
                if bb >= B:
                    kp = True
                if cutlass.const_expr(tiebin):
                    if bb == B:
                        if Int32(Uint32(ev[k] >> Uint64(32))) > idx_thr:
                            kp = False
            if kp & (bb == B):
                fk = L.fkey_bits(L.entry_bits(ev[k]))
                if fk < fmn:
                    fmn = fk
            kmask[k] = Int32(cute.arch.vote_ballot_sync(kp))
            wtot = wtot + Int32(cute.arch.popc(kmask[k]))
        base = Int32(0)
        if lane == Int32(0):
            if wtot > Int32(0):
                base = Int32(
                    cute.arch.atomic_add(
                        sScal.iterator + Int32(24), wtot, sem="relaxed", scope="cta"
                    )
                )
        base = cute.arch.shuffle_sync(base, 0)
        for k in cutlass.range_constexpr(EPT):
            if kmask[k] >> lane & Int32(1) != Int32(0):
                sMain[base + Int32(cute.arch.popc(kmask[k] & lm))] = ev[k]
            base = base + Int32(cute.arch.popc(kmask[k]))
        fmn = L.warp_min_u32(fmn)
        if lane == Int32(0):
            cute.arch.atomic_min(kmin2_ptr, fmn, sem="relaxed", scope="cta")
        L.bar(bar_id, nthr)
        tf = Uint32(cute.arch.load(kmin2_ptr, Uint32, sem="relaxed", scope="cta"))
        if tid == Int32(0):
            cute.arch.store(cnt_ptr, kept, sem="relaxed", scope="cta")
            cute.arch.store(kmin_ptr, tf, sem="relaxed", scope="cta")
        L.bar(bar_id, nthr)
        if cutlass.const_expr(prof is not None):
            if tid == Int32(0):
                tp1 = cutlass.Int64(cute.arch.globaltimer())
                prof[3] = prof[3] + (tp1 - tp0)
    else:
        kept = Int32(-1)
    return (kept, tf)


@cute.jit
def _parts_of_row(q, cta, sched, num_sms, lane):
    P = Int32(0)
    p = Int32(0)
    nit = (num_sms + Int32(31)) // Int32(32)
    it = Int32(0)
    while it < nit:
        c = lane + it * Int32(32)
        v = False
        if c < num_sms:
            sq0 = sched[c, 0]
            skv0 = sched[c, 1]
            sq1 = sched[c + 1, 0]
            skv1 = sched[c + 1, 1]
            hi = sq1
            if skv1 == Int32(0):
                hi = sq1 - Int32(1)
            if (sq0 != sq1) | (skv0 != skv1):
                if sq0 <= q:
                    if q <= hi:
                        v = True
        bv = Int32(cute.arch.vote_ballot_sync(v))
        bl = Int32(cute.arch.vote_ballot_sync(v & (c < cta)))
        P = P + Int32(cute.arch.popc(bv))
        p = p + Int32(cute.arch.popc(bl))
        it = it + Int32(1)
    return (P, p)


@cute.jit
def _cover_ballot(q, cta, sched, num_sms, lane, it):
    c = lane + it * Int32(32)
    v = False
    if c < num_sms:
        if c != cta:
            sq0 = sched[c, 0]
            skv0 = sched[c, 1]
            sq1 = sched[c + 1, 0]
            skv1 = sched[c + 1, 1]
            hi = sq1
            if skv1 == Int32(0):
                hi = sq1 - Int32(1)
            if (sq0 != sq1) | (skv0 != skv1):
                if sq0 <= q:
                    if q <= hi:
                        v = True
    return Int32(cute.arch.vote_ballot_sync(v))


@cute.jit
def _write_row_outputs_regs(
    row, n, s_out, nout, short, topk_out, gidx, bt_swa, bt_c, pos, req, mbs, mbc, tid
):
    gv = []
    for q in cutlass.range_constexpr(5):
        col = tid + Int32(q * 128)
        g = Int32(-1)
        if cutlass.const_expr(q == 0):
            g = L.l2g_swa(col, pos, req, bt_swa, mbs, Int32(0))
        else:
            i = col - Int32(L.SWA)
            idx = Int32(-1)
            if short != Int32(0):
                if i < n:
                    idx = i
            elif i < nout:
                idx = Int32(Uint32(4294967295) - Uint32(Uint64(s_out[i]) & Uint64(4294967295)))
            topk_out[row, i] = idx
            g = L.l2g_compressed(idx, req, bt_c, mbc, Int32(0))
        gv.append(g)
    for q in cutlass.range_constexpr(5):
        gidx[row, tid + Int32(q * 128)] = gv[q]
    return (gv[0], gv[1], gv[2], gv[3], gv[4])


@cute.jit
def _final512_token32(sHand, nout, tid, bar_id: cutlass.Constexpr):
    # topk is a set and attention does not depend on row order: convert the completed row's entries to the
    # output writer's inverted-token low word in place instead of sorting them.
    base = tid * Int32(4)
    for i in cutlass.range_constexpr(4):
        if base + Int32(i) < nout:
            sHand[base + Int32(i)] = Uint64(
                Uint32(4294967295) - Uint32(Uint64(sHand[base + Int32(i)]) >> Uint64(32))
            )
    L.bar(bar_id, 128)


@cute.jit
def _pick_final32(a, b, first):
    r = cutlass.min(a, b)
    if not first:
        r = cutlass.max(a, b)
    return r


@cute.jit
def _sort_final512_token32(sHand, nout, sScratch, tid, bar_id: cutlass.Constexpr):
    # Sort only the completed row. Four keys per thread keep the small
    # exchanges in registers; three large exchanges use dead merge scratch.
    base = tid * Int32(4)
    e0 = Uint32(4294967295)
    e1 = Uint32(4294967295)
    e2 = Uint32(4294967295)
    e3 = Uint32(4294967295)
    if base < nout:
        e0 = Uint32(Uint64(sHand[base]) >> Uint64(32))
    if base + Int32(1) < nout:
        e1 = Uint32(Uint64(sHand[base + Int32(1)]) >> Uint64(32))
    if base + Int32(2) < nout:
        e2 = Uint32(Uint64(sHand[base + Int32(2)]) >> Uint64(32))
    if base + Int32(3) < nout:
        e3 = Uint32(Uint64(sHand[base + Int32(3)]) >> Uint64(32))
    sx = cute.make_tensor(cute.recast_ptr(sScratch.iterator, dtype=Uint32), cute.make_layout(512))
    kk = Int32(1)
    while kk < Int32(10):
        jj = kk - Int32(1)
        while jj >= Int32(0):
            if jj >= Int32(7):
                # Transposed exchange gives contiguous words to each warp.
                # Scratch is separate from sHand: initial raw loads cannot
                # race with the first exchange's stores from another warp.
                sx[tid] = e0
                sx[tid + Int32(128)] = e1
                sx[tid + Int32(256)] = e2
                sx[tid + Int32(384)] = e3
                L.bar(bar_id, 128)
                peer = tid ^ (Int32(1) << (jj - Int32(2)))
                p0 = Uint32(sx[peer])
                p1 = Uint32(sx[peer + Int32(128)])
                p2 = Uint32(sx[peer + Int32(256)])
                p3 = Uint32(sx[peer + Int32(384)])
                L.bar(bar_id, 128)
                first = (((base >> jj) ^ (base >> kk)) & Int32(1)) == Int32(0)
                e0 = _pick_final32(e0, p0, first)
                e1 = _pick_final32(e1, p1, first)
                e2 = _pick_final32(e2, p2, first)
                e3 = _pick_final32(e3, p3, first)
            elif jj >= Int32(2):
                delta = Int32(1) << (jj - Int32(2))
                p0 = cute.arch.shuffle_sync_bfly(e0, delta)
                p1 = cute.arch.shuffle_sync_bfly(e1, delta)
                p2 = cute.arch.shuffle_sync_bfly(e2, delta)
                p3 = cute.arch.shuffle_sync_bfly(e3, delta)
                first = (((base >> jj) ^ (base >> kk)) & Int32(1)) == Int32(0)
                e0 = _pick_final32(e0, p0, first)
                e1 = _pick_final32(e1, p1, first)
                e2 = _pick_final32(e2, p2, first)
                e3 = _pick_final32(e3, p3, first)
            elif jj == Int32(1):
                first = ((base >> kk) & Int32(1)) == Int32(0)
                a0 = e0
                b0 = e2
                a1 = e1
                b1 = e3
                e0 = _pick_final32(a0, b0, first)
                e2 = _pick_final32(a0, b0, not first)
                e1 = _pick_final32(a1, b1, first)
                e3 = _pick_final32(a1, b1, not first)
            else:
                first0 = ((base >> kk) & Int32(1)) == Int32(0)
                first2 = (((base + Int32(2)) >> kk) & Int32(1)) == Int32(0)
                a0 = e0
                b0 = e1
                a2 = e2
                b2 = e3
                e0 = _pick_final32(a0, b0, first0)
                e1 = _pick_final32(a0, b0, not first0)
                e2 = _pick_final32(a2, b2, first2)
                e3 = _pick_final32(a2, b2, not first2)
            jj = jj - Int32(1)
        kk = kk + Int32(1)
    # The existing output writer consumes only the inverted token index in
    # the low word. No score data is needed after the completed-row sort.
    sHand[base] = Uint64(Uint32(4294967295) - e0)
    sHand[base + Int32(1)] = Uint64(Uint32(4294967295) - e1)
    sHand[base + Int32(2)] = Uint64(Uint32(4294967295) - e2)
    sHand[base + Int32(3)] = Uint64(Uint32(4294967295) - e3)
    L.bar(bar_id, 128)


@cute.jit
def _finalize_seg(
    tid,
    bar_id: cutlass.Constexpr,
    cta,
    start_q,
    q,
    n,
    P,
    p,
    sMb,
    cbuf,
    rel_ptr,
    sHand,
    sAttB,
    sAttO,
    sHistS,
    sCkS,
    sResS,
    sRedS,
    sWcS,
    s_flag,
    mKvLens,
    mReq,
    mBtSwa,
    mBtC,
    mTopk,
    mGidx,
    mRowStat,
    mSched,
    ws_seg_base,
    ws_ctr_base,
    mbs,
    mbc,
    num_sms: cutlass.Constexpr,
    skew_ns: cutlass.Constexpr,
    skew_seed,
    salt,
    stamps: cutlass.Constexpr,
    mStamps,
    sbase,
    dly_t=None,
    dly_us: cutlass.Constexpr = 0,
    rel_keep=None,
    rsort: cutlass.Constexpr = False,
    rel_bar=None,
):
    NT: cutlass.Constexpr[int] = SEL_NTHR
    if cutlass.const_expr(stamps):
        if tid == Int32(0):
            mStamps[sbase] = cutlass.Int64(cute.arch.globaltimer())
    fast = Int32(0)
    if (n <= Int32(K_TOP)) & (P == Int32(1)):
        fast = Int32(1)
    nout = Int32(0)
    if fast == Int32(0):
        nout, kth_, m_, inf_ = L.select_topk_list(
            sMb, cbuf, Int32(K_TOP), sHand, sHistS, sCkS, sResS, sRedS, sWcS, tid, False, NT, bar_id
        )
    # Keep this parity slot until all raw segment unions have been selected.
    # The 2048-entry source buffer is also our bounded <=1024-entry merge list.
    if cutlass.const_expr(dly_us > 0):
        if dly_t >= Int32(0):
            if tid == Int32(0):
                t_n = Int32(Uint32(cutlass.Int64(cute.arch.globaltimer()) >> Int64(10)))
                while t_n - dly_t < Int32(dly_us):
                    L.nanosleep(Int32(1000))
                    t_n = Int32(Uint32(cutlass.Int64(cute.arch.globaltimer()) >> Int64(10)))
    L.bar(bar_id, NT)
    if cutlass.const_expr(stamps):
        if tid == Int32(0):
            mStamps[sbase + Int32(1)] = cutlass.Int64(cute.arch.globaltimer())
    if cutlass.const_expr(stamps):
        if tid == Int32(0):
            mStamps[sbase + Int32(2)] = cutlass.Int64(cute.arch.globaltimer())
    pos = Int32(mKvLens[q]) - Int32(1)
    req = Int32(mReq[q])
    short = Int32(0)
    if n <= Int32(K_TOP):
        short = Int32(1)
    done_row = Int32(0)
    gv0 = Int32(-1)
    gv1 = Int32(-1)
    gv2 = Int32(-1)
    gv3 = Int32(-1)
    gv4 = Int32(-1)
    if P == Int32(1):
        if short == Int32(0):
            _final512_token32(sHand, nout, tid, bar_id)
        gv0, gv1, gv2, gv3, gv4 = _write_row_outputs_regs(
            q, n, sHand, nout, short, mTopk, mGidx, mBtSwa, mBtC, pos, req, mbs, mbc, tid
        )
        done_row = Int32(1)
    else:
        pside = Int64(1)
        if q == start_q:
            pside = Int64(0)
        so = ws_seg_base + (Int64(cta) * Int64(2) + pside) * Int64(SEGW * 8)
        out_i = tid
        while out_i < nout:
            L.st_g_u64(so + Int64(out_i) * Int64(8), Uint64(sHand[out_i]))
            out_i = out_i + Int32(NT)
        if tid == Int32(0):
            L.st_g_u32(so + Int64(512 * 8), nout)
        L.bar(bar_id, NT)
        if tid == Int32(0):
            if cutlass.const_expr(skew_ns > 0):
                L.skew_sleep(skew_seed, salt, Int32(skew_ns))
            ca = ws_ctr_base + Int64(q) * Int64(128)
            old = L.gc_arrive(ca)
            lst = Int32(0)
            if Int32(Uint32(old & Uint64(65535))) == P - Int32(1):
                lst = Int32(1)
                L.gc_pad(ca, P)
                L.fence_acq_rel_gpu()
            s_flag[0] = lst
        L.bar(bar_id, NT)
        if cutlass.const_expr(stamps):
            if tid == Int32(0):
                mStamps[sbase + Int32(3)] = cutlass.Int64(cute.arch.globaltimer())
        if s_flag[0] != Int32(0):
            na = nout
            rlane = tid & Int32(31)
            nit = (Int32(num_sms) + Int32(31)) // Int32(32)
            it = Int32(0)
            while it < nit:
                bv = _cover_ballot(q, cta, mSched, Int32(num_sms), rlane, it)
                while bv != Int32(0):
                    low = bv & Int32(0) - bv
                    cs = it * Int32(32) + Int32(cute.arch.popc(low - Int32(1)))
                    sside = Int64(1)
                    if mSched[cs, 0] == q:
                        sside = Int64(0)
                    src = ws_seg_base + (Int64(cs) * Int64(2) + sside) * Int64(SEGW * 8)
                    nb = L.ld_cg_i32(src + Int64(512 * 8))
                    ia = tid
                    while ia < na:
                        sMb[ia] = Uint64(sHand[ia])
                        ia = ia + Int32(NT)
                    ib = tid
                    while ib < nb:
                        sMb[na + ib] = L.ld_cg_u64(src + Int64(ib) * Int64(8))
                        ib = ib + Int32(NT)
                    L.bar(bar_id, NT)
                    nm, kth_m, m_m, inf_m = L.select_topk_list(
                        sMb,
                        na + nb,
                        Int32(K_TOP),
                        sHand,
                        sHistS,
                        sCkS,
                        sResS,
                        sRedS,
                        sWcS,
                        tid,
                        False,
                        NT,
                        bar_id,
                    )
                    na = nm
                    bv = bv ^ low
                it = it + Int32(1)
            if cutlass.const_expr(stamps):
                if tid == Int32(0):
                    mStamps[sbase + Int32(4)] = cutlass.Int64(cute.arch.globaltimer())
            if short == Int32(0):
                _final512_token32(sHand, na, tid, bar_id)
            gv0, gv1, gv2, gv3, gv4 = _write_row_outputs_regs(
                q, n, sHand, na, short, mTopk, mGidx, mBtSwa, mBtC, pos, req, mbs, mbc, tid
            )
            done_row = Int32(1)
    L.bar(bar_id, NT)
    if done_row != Int32(0):
        if tid == Int32(0):
            L.publish_row(mRowStat.iterator.toint(), q)
    if cutlass.const_expr(rel_bar is not None):
        rk = Int32(0)
        if cutlass.const_expr(rel_keep is not None):
            rk = rel_keep
        if rk == Int32(0):
            # Release the parity slot only after every union selection read.
            prims.mbarrier_arrive(rel_bar)
    return (done_row, gv0, gv1, gv2, gv3, gv4)


class IndexerKernel:
    def __init__(
        self,
        num_sms: int = 148,
        block_kv: int = 128,
        phys_block_kv: int = 32,
        num_heads: int = 64,
        head_dim: int = 128,
        policy: int = CP.POL_NONE,
        debug_dump: bool = False,
        admit: bool = True,
        kv_stages: int = 3,
        q_stages: int = 3,
        math_regs: int = 136,
        sel_regs: int = 88,
        row_regs: int = 96,
        trig: int = TRIG,
        stamps: bool = False,
        num_w_in_reg: int = 64,
        attn: bool = False,
        dbg_dup: bool = False,
        row_noop: bool = False,
        dbg_hand: bool = False,
        skew_ns: int = 0,
        no_deep: bool = False,
        dbg_addr: bool = False,
        att_skip: int = 0,
        late_release: bool = True,
        tmem_fences: bool = False,
        keep_cap: int = KEEP_CAP,
        desc: bool = False,
        rot: bool = False,
        sel_fin: bool = False,
        gath: bool = False,
        cbins: int = 1024,
        efl: bool = False,
        ahint: int = 0,
        etile: bool = False,
        spin_ns: int = 0,
        pfence: bool = True,
        sel_idle_ns: int = 32,
        rdelay: int = 0,
        apace: int = 0,
        ring2: bool = False,
        rsort: bool = False,
        trig0: int = 0,
        kfloor: bool = False,
        pace_adapt: bool = True,
        defer: bool = True,
        dq_cap: int = DQ_CAP,
        validate: bool = True,
        tiebin: bool = True,
    ):
        self.block_kv = block_kv
        self.phys_block_kv = phys_block_kv
        self.num_blocks_per_mma = block_kv // phys_block_kv
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.next_n_atom = 1
        self.N = num_heads
        self.num_sms = num_sms
        self.epi_dtype = cutlass.Float32
        self.acc_dtype = cutlass.Float32
        self.output_dtype = cutlass.Float32
        self.epi_bytes = 4
        w_stage_bytes = self.N * self.epi_bytes
        self.w_stage_stride = (w_stage_bytes + 127) // 128 * 128 // self.epi_bytes
        self.num_math_threads = 256
        self.num_math_warps = 8
        self.threads_per_cta = 640
        self.num_q_stages = q_stages
        self.num_umma_stages = 2
        self.num_kv_stages = kv_stages
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_mn = (block_kv, self.N)
        if policy not in CP.POL_NAMES:
            raise ValueError(f"unknown cache policy {policy}")
        self.policy = policy
        self.debug_dump = bool(debug_dump)
        self.admit = bool(admit)
        if attn and q_stages < 3:
            raise ValueError(
                "the deep attention ring keeps Q slices 0-2 in the 3 indexer Q stages (q_stages >= 3)"
            )
        if (
            math_regs * 256 + (24 + sel_regs + row_regs) * 128 > 96 * 640
            or (math_regs | sel_regs | row_regs) % 8
        ):
            raise ValueError("register budget (CTA pool 61,440)")
        self.math_regs = math_regs
        self.sel_regs = sel_regs
        self.row_regs = row_regs
        if not K_TOP < trig <= MAIN_CAP - 512:
            raise ValueError("trig must be in (512, MAIN_CAP - 512]")
        self.trig = trig
        self.stamps = bool(stamps)
        if num_heads != 64 or num_w_in_reg != num_heads:
            raise ValueError("split epilogue requires 64 heads and all head weights in registers")
        self.num_w_in_reg = num_w_in_reg
        self.attn = bool(attn)
        self.dbg_dup = bool(dbg_dup)
        self.row_noop = bool(row_noop)
        self.dbg_hand = bool(dbg_hand)
        self.skew_ns = int(skew_ns)
        self.no_deep = bool(no_deep)
        self.dbg_addr = bool(dbg_addr)
        self.att_skip = int(att_skip)
        self.late_release = bool(late_release)
        self.keep_cap = int(keep_cap)
        self.tmem_fences = bool(tmem_fences)
        self.desc = bool(desc)
        self.rot = bool(rot)
        self.sel_fin = bool(sel_fin)
        self.gath = bool(gath) and self.sel_fin and self.attn and (int(att_skip) == 0)
        if cbins not in (128, 256, 512, 1024):
            raise ValueError("cbins")
        self.cbins = int(cbins)
        self.etile = bool(etile) and self.gath
        self.pfence = bool(pfence)
        self.sel_idle_ns = int(sel_idle_ns)
        self.spin_ns = int(spin_ns)
        self.rdelay = int(rdelay)
        self.apace = int(apace)
        self.ahint = int(ahint)
        self.efl = bool(efl)
        self.ring2 = bool(ring2) and self.attn
        self.rsort = bool(rsort)
        if trig0 and trig0 <= K_TOP:
            raise ValueError("trig0 must be > 512")
        self.trig0 = min(int(trig0), int(trig)) if trig0 else 0
        self.kfloor = bool(kfloor)
        self.pace_adapt = bool(pace_adapt)
        if defer and (etile or ring2 or rdelay):
            raise ValueError(
                "defer is built for the shipped tail (gath without etile / ring2 / rdelay)"
            )
        self.defer = bool(defer) and self.attn
        if not 1 <= int(dq_cap) <= 1024:
            raise ValueError("dq_cap")
        self.dq_cap = int(dq_cap)
        self.validate = bool(validate)
        self.tiebin = bool(tiebin)

    def _epi_half(self, tTR_rAcc, w_cache, head_start, s0x, s0y, s1x, s1y):
        # The same four FMA chains continue across the two 32-head TMEM loads.
        acc_vec = tTR_rAcc.load()
        for h in range(0, 32, 4):
            x0 = acc_vec[h]
            x1 = acc_vec[h + 1]
            x2 = acc_vec[h + 2]
            x3 = acc_vec[h + 3]
            a0, a1 = cute.arch.add_packed_f32x2((x0, x1), (abs(x0), abs(x1)), rnd=_RND_RN)
            a2, a3 = cute.arch.add_packed_f32x2((x2, x3), (abs(x2), abs(x3)), rnd=_RND_RN)
            w0 = w_cache[head_start + h]
            w1 = w_cache[head_start + h + 1]
            w2 = w_cache[head_start + h + 2]
            w3 = w_cache[head_start + h + 3]
            s0x, s0y = cute.arch.fma_packed_f32x2((a0, a1), (w0, w1), (s0x, s0y), rnd=_RND_RN)
            s1x, s1y = cute.arch.fma_packed_f32x2((a2, a3), (w2, w3), (s1x, s1y), rnd=_RND_RN)
        return (s0x, s0y, s1x, s1y)

    @cute.jit
    def _nseg(self, start_q, start_kv_half, end_q_idx, end_kv_half):
        qb = end_q_idx
        if end_kv_half == Int32(0):
            qb = end_q_idx - Int32(1)
        ns = qb - start_q + Int32(1)
        if (start_q == end_q_idx) & (start_kv_half == end_kv_half):
            ns = Int32(0)
        return ns

    @cute.jit
    def _tiles(self, q, mContextLens, batch_size):
        return (
            self._ctx_of(cutlass.min(q, batch_size - 1), mContextLens, batch_size)
            + Int32(self.block_kv - 1)
        ) // Int32(self.block_kv)

    @cute.jit
    def _walk_init(self, start_q, start_kv_half, end_q_idx, end_kv_half, mContextLens, batch_size):
        end_kv_idx = end_kv_half * Int32(2)
        nq = start_q
        nk = start_kv_half * Int32(2)
        ph = Int32(2)
        stop_kv = end_kv_idx
        rot_on = Int32(0)
        if cutlass.const_expr(self.rot):
            if (end_kv_half > Int32(0)) & (end_q_idx > start_q):
                rot_on = Int32(1)
                nq = end_q_idx
                nk = Int32(0)
                ph = Int32(1)
                stop_kv = Int32(0)
        nn = self._tiles(nq, mContextLens, batch_size)
        hw = (ph == Int32(1)) | ((nq != end_q_idx) | (nk != stop_kv))
        return (nq, nk, nn, ph, stop_kv, hw, rot_on)

    @cute.jit
    def _walk_next(
        self,
        q,
        nk,
        num_kv,
        nn,
        ph,
        stop_kv,
        start_q,
        start_kv_half,
        end_q_idx,
        end_kv_idx,
        mContextLens,
        batch_size,
        seg_end: cutlass.Constexpr = False,
    ):
        nq = q
        nk2 = nk
        nn2 = nn
        ph2 = ph
        if ph == Int32(1):
            if nk >= end_kv_idx:
                nq = start_q
                nk2 = start_kv_half * Int32(2)
                ph2 = Int32(2)
                nn2 = self._tiles(start_q, mContextLens, batch_size)
        elif nk >= num_kv:
            nq = q + Int32(1)
            nk2 = Int32(0)
            if nq < batch_size:
                nn2 = self._tiles(nq, mContextLens, batch_size)
        hw = (ph2 == Int32(1)) | ((nq != end_q_idx) | (nk2 != stop_kv))
        return (nq, nk2, nn2, ph2, hw)

    @cute.jit
    def _walk_qnext(self, q, ph, rot_on, start_q, end_q_idx, end_kv_idx):
        pq = q + Int32(1)
        ok = Int32(0)
        if ph == Int32(1):
            pq = start_q
            ok = Int32(1)
        elif pq < end_q_idx:
            ok = Int32(1)
        elif pq == end_q_idx:
            if end_kv_idx > Int32(0):
                if rot_on == Int32(0):
                    ok = Int32(1)
        return (pq, ok)

    @cute.jit
    def _seg_s(self, kv_idx, num_kv, q_idx, end_q_idx, end_kv_idx):
        b2 = num_kv + Int32(1) & Int32(-2)
        if q_idx == end_q_idx:
            b2 = end_kv_idx
        if b2 < kv_idx + Int32(2):
            b2 = kv_idx + Int32(2)
        return kv_idx + b2 - Int32(2)

    @cute.jit
    def _sched_verdict(self, mSched, mContextLens, batch_size, lane):
        G = self.num_sms
        bad = Int32(0)
        for it in cutlass.range_constexpr((G + 1 + 31) // 32):
            s = lane + Int32(it * 32)
            if s <= Int32(G):
                q = mSched[s, 0]
                k = mSched[s, 1]
                if (q < Int32(0)) | (q > batch_size) | (k < Int32(0)):
                    bad = Int32(1)
                elif q == batch_size:
                    if k != Int32(0):
                        bad = Int32(1)
                else:
                    u = (mContextLens[q] + Int32(255)) // Int32(256)
                    if (k > Int32(0)) & (k >= u):
                        bad = Int32(1)
                if s < Int32(G):
                    q1 = mSched[s + Int32(1), 0]
                    k1 = mSched[s + Int32(1), 1]
                    if (q1 < q) | (q1 == q) & (k1 < k):
                        bad = Int32(1)
                elif q != batch_size:
                    bad = Int32(1)
        q0 = cutlass.min(cutlass.max(mSched[0, 0], Int32(0)), batch_size)
        r = lane
        while r < q0:
            if mContextLens[r] != Int32(0):
                bad = Int32(1)
            r = r + Int32(32)
        if mSched[0, 1] != Int32(0):
            bad = Int32(1)
        v = Int32(0)
        if Int32(cute.arch.vote_ballot_sync(bad != Int32(0))) != Int32(0):
            v = Int32(ST_SCHED)
        return v

    @cute.jit
    def _ctx_of(self, q, mContextLens, batch_size):
        ctx = cutlass.Int32(0)
        if q < batch_size:
            ctx = mContextLens[q]
        return ctx

    def _setup_mma(self, a_dtype, b_dtype, a_major, b_major):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.mma_tiler = (*self.mma_tiler_mn, 1)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype, b_dtype, a_major, b_major, self.acc_dtype, self.cta_group, self.mma_tiler_mn
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.head_dim // mma_inst_shape_k
        full_k = mma_inst_shape_k * mma_inst_tile_k
        self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], full_k)
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.epi_tile = self.cta_tile_shape_mnk[:2]
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, a_dtype, self.num_kv_stages
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, b_dtype, self.num_q_stages
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_alloc_cols = cutlass.utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        self.num_tmem_alloc_cols_total = self.num_tmem_alloc_cols * 2 * self.num_umma_stages
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        kv_fused: cute.Tensor,
        b: cute.Tensor,
        weights: cute.Tensor,
        logits: cute.Tensor,
        block_table: cute.Tensor,
        context_lens: cute.Tensor,
        schedule_meta: cute.Tensor,
        kv_lens: cute.Tensor,
        req_id: cute.Tensor,
        bt_swa: cute.Tensor,
        bt_c: cute.Tensor,
        topk_out: cute.Tensor,
        gidx: cute.Tensor,
        ws_seg: cute.Tensor,
        ws_ctr: cute.Tensor,
        rowstat: cute.Tensor,
        stamps: cute.Tensor,
        q_att: cute.Tensor,
        swa_pool: cute.Tensor,
        c_pool: cute.Tensor,
        sinks: cute.Tensor,
        out: cute.Tensor,
        status: cute.Tensor,
        softmax_scale: cutlass.Float32,
        num_phys_blocks: cutlass.Int32,
        batch_size: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        phys_block_kv = self.phys_block_kv
        scale_offset_elems = phys_block_kv * self.head_dim
        kv_fp8 = cute.recast_tensor(kv_fused, cutlass.Float8E4M3FN)
        b = cute.recast_tensor(b, cutlass.Float8E4M3FN)
        kv_block_stride = kv_fused.layout.stride[0]
        kv_layout = cute.make_layout(
            (phys_block_kv, self.head_dim, num_phys_blocks),
            stride=(self.head_dim, 1, kv_block_stride),
        )
        a = cute.make_tensor(kv_fp8.iterator, kv_layout)
        scale_fp8_layout = cute.make_layout(
            (phys_block_kv * 4, num_phys_blocks), stride=(1, kv_block_stride)
        )
        scale_fp8 = cute.make_tensor(kv_fp8.iterator + scale_offset_elems, scale_fp8_layout)
        scales = cute.recast_tensor(scale_fp8, cutlass.Float32)
        a_dtype = a.element_type
        b_dtype = b.element_type
        a_major = cutlass.utils.LayoutEnum.from_tensor(a).mma_major_mode()
        b_major = cutlass.utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
        tiled_mma = self._setup_mma(a_dtype, b_dtype, a_major, b_major)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        self.a_tma_view_layout = sm100_utils.make_smem_layout(
            cute.nvgpu.OperandMajorMode.K,
            (self.block_kv, self.head_dim),
            a_dtype,
            self.num_kv_stages,
        )
        self.a_tma_view_layout = cute.tiled_divide(
            self.a_tma_view_layout, (self.phys_block_kv, self.head_dim)
        )
        self.a_tma_view_layout = cute.select(self.a_tma_view_layout, mode=[0, 1, 3])
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_load_op, a, self.a_tma_view_layout[0], (self.phys_block_kv, self.head_dim)
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        self.w_smem_layout_staged = cute.make_layout(
            (self.N, self.num_q_stages), stride=(1, self.w_stage_stride)
        )
        w_smem_per_stage = cute.select(self.w_smem_layout_staged, mode=[0])
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op, weights, w_smem_per_stage, self.w_smem_layout_staged.shape[:1]
        )
        self.s_smem_layout_staged = cute.make_layout((self.block_kv, self.num_kv_stages))
        s_smem_per_subblock = cute.make_layout((phys_block_kv,))
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_load_op, scales, s_smem_per_subblock, (phys_block_kv,)
        )
        b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
        w_copy_size = self.N * self.epi_bytes
        kv_tma_bytes_per_subblock = phys_block_kv * self.head_dim
        scale_tma_bytes_per_subblock = phys_block_kv * 4
        self.num_kv_scale_tma_bytes = self.num_blocks_per_mma * (
            kv_tma_bytes_per_subblock + scale_tma_bytes_per_subblock
        )
        self.num_q_tma_bytes = b_copy_size * atom_thr_size + w_copy_size

        @cute.struct
        class SharedStorage:
            kv_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            kv_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.num_q_stages * 2]
            umma_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            umma_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            tmem_holding_buf: cutlass.Int32

        tma_q = xcuda.create_tensor_map_tiled(
            global_address=q_att.iterator.toint(),
            dtype=cutlass.Float8E4M3FN,
            global_dims=[A.LATENT, A.HEADS, q_att.shape[0]],
            global_strides=[A.LATENT // 16, A.HEADS * A.LATENT // 16],
            box_dims=[A.SLICE, A.HEADS, 1],
            swizzle=xcuda.TensorMapSwizzle.s128b,
        )
        tma_swa = xcuda.create_tensor_map_tiled(
            global_address=swa_pool.iterator.toint(),
            dtype=cutlass.Float8E4M3FN,
            global_dims=[A.LATENT, swa_pool.shape[0]],
            global_strides=[A.LATENT // 16],
            box_dims=[A.SLICE, 1],
            swizzle=xcuda.TensorMapSwizzle.s128b,
        )
        tma_cmp = xcuda.create_tensor_map_tiled(
            global_address=c_pool.iterator.toint(),
            dtype=cutlass.Float8E4M3FN,
            global_dims=[A.LATENT, c_pool.shape[0]],
            global_strides=[A.LATENT // 16],
            box_dims=[A.SLICE, 1],
            swizzle=xcuda.TensorMapSwizzle.s128b,
        )
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_s,
            tma_tensor_s,
            tma_q,
            tma_swa,
            tma_cmp,
            sinks,
            out,
            status,
            softmax_scale,
            logits,
            block_table,
            context_lens,
            schedule_meta,
            kv_lens,
            req_id,
            bt_swa,
            bt_c,
            topk_out,
            gidx,
            ws_seg,
            ws_ctr,
            rowstat,
            stamps,
            batch_size,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.w_smem_layout_staged,
            self.s_smem_layout_staged,
            self.a_tma_view_layout,
            self.epi_tile,
            SharedStorage,
        ).launch(
            grid=(1, 1, self.num_sms),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_w: cute.CopyAtom,
        mW_tma: cute.Tensor,
        tma_atom_s: cute.CopyAtom,
        mS_tma: cute.Tensor,
        tma_q: cutlass.GridConstant[xcuda.TensorMap],
        tma_swa: cutlass.GridConstant[xcuda.TensorMap],
        tma_cmp: cutlass.GridConstant[xcuda.TensorMap],
        mSinks: cute.Tensor,
        mOut: cute.Tensor,
        mStatus: cute.Tensor,
        softmax_scale: cutlass.Float32,
        mLogits: cute.Tensor,
        mBlockTable: cute.Tensor,
        mContextLens: cute.Tensor,
        mScheduleMeta: cute.Tensor,
        mKvLens: cute.Tensor,
        mReq: cute.Tensor,
        mBtSwa: cute.Tensor,
        mBtC: cute.Tensor,
        mTopk: cute.Tensor,
        mGidx: cute.Tensor,
        mWsSeg: cute.Tensor,
        mWsCtr: cute.Tensor,
        mRowStat: cute.Tensor,
        mStamps: cute.Tensor,
        batch_size: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.Layout,
        s_smem_layout_staged: cute.Layout,
        a_tma_view_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
        SharedStorage: cutlass.Constexpr,
    ):
        # Let the attention grid start on the SMs this grid leaves free; it consumes rows as they are published.
        cute.arch.griddepcontrol_launch_dependents()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()
        warpgroup_idx = warp_idx // 4
        is_math_warp = warp_idx < 8
        is_tma_warp_0 = warp_idx == 8
        is_tma_warp_1 = warp_idx == 9
        is_tma_warp = is_tma_warp_0 | is_tma_warp_1
        is_umma_warp_0 = warp_idx == 10
        is_umma_warp_1 = warp_idx == 11
        is_sel_warp = (warp_idx >= 12) & (warp_idx < 16)
        is_row_warp = warp_idx >= 16
        if cutlass.const_expr(self.stamps):
            if tidx == 0:
                mStamps[bidz * 256 + 0] = cutlass.Int64(cute.arch.globaltimer())
                mStamps[bidz * 256 + 15] = cutlass.Int64(cute.arch.smid())
                mStamps[bidz * 256 + 13] = L.clk64()
        NUM_MATH_WG = 2
        NUM_BLOCKS_PER_MMA = self.num_blocks_per_mma
        sm_idx = bidz
        start_q = mScheduleMeta[sm_idx, 0]
        start_kv_half = mScheduleMeta[sm_idx, 1]
        end_q_idx = mScheduleMeta[sm_idx + 1, 0]
        end_kv_half = mScheduleMeta[sm_idx + 1, 1]
        refused = Int32(0)
        if cutlass.const_expr(self.validate):
            refused = self._sched_verdict(mScheduleMeta, mContextLens, batch_size, tidx % 32)
            if refused != Int32(0):
                start_q = batch_size
                start_kv_half = Int32(0)
                end_q_idx = batch_size
                end_kv_half = Int32(0)
        if is_tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_s)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        block_kv_val = self.block_kv
        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        q_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 8)
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar.data_ptr(),
            num_stages=self.num_q_stages,
            producer_group=prod_group,
            consumer_group=q_cons_group,
            tx_count=self.num_q_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        q_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_q_stages
        )
        q_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        q_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        q_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        math_warps_per_group = self.num_math_warps // 2
        kv_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, math_warps_per_group)
        kv_pipeline_0 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar_0.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        kv_pipeline_1 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar_1.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        kv_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        kv_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        kv_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_math_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_math_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        math_threads_per_group = self.num_math_threads // 2
        umma_pipeline_0 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar_0.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, math_threads_per_group),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        umma_pipeline_1 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar_1.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, math_threads_per_group),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        umma_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )
        umma_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )
        umma_cons_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_umma_stages
        )
        umma_cons_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_umma_stages
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=640)
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=False,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        sKV_0 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=a_smem_layout_staged.inner,
        )
        sKV_1 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=a_smem_layout_staged.inner,
        )
        sQ = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=b_smem_layout_staged.inner,
        )
        sW = smem.allocate_tensor(
            element_type=self.epi_dtype, layout=w_smem_layout_staged, byte_alignment=128
        )
        sScales_0 = smem.allocate_tensor(
            element_type=cutlass.Float32, layout=s_smem_layout_staged, byte_alignment=128
        )
        sScales_1 = smem.allocate_tensor(
            element_type=cutlass.Float32, layout=s_smem_layout_staged, byte_alignment=128
        )
        s_ring = smem.allocate_tensor(
            element_type=Uint64, layout=cute.make_layout(RING * 8), byte_alignment=1024
        )
        sMain = smem.allocate_tensor(
            element_type=Uint64, layout=cute.make_layout(2 * MAIN_CAP), byte_alignment=1024
        )
        sOut = smem.allocate_tensor(
            element_type=Uint64, layout=cute.make_layout(K_TOP), byte_alignment=128
        )
        sHand = smem.allocate_tensor(
            element_type=Uint64, layout=cute.make_layout(K_TOP), byte_alignment=128
        )
        sHB = smem.allocate_tensor(
            element_type=Uint64, layout=cute.make_layout(K_TOP), byte_alignment=128
        )
        sHist = cute.make_tensor(
            cute.recast_ptr(sHB.iterator, dtype=Int32), cute.make_layout(L.NB_SEL)
        )
        sAttB = smem.allocate_tensor(
            element_type=Uint64, layout=cute.make_layout(K_TOP), byte_alignment=128
        )
        sAttO = smem.allocate_tensor(
            element_type=Uint64, layout=cute.make_layout(K_TOP), byte_alignment=128
        )
        sHI = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(16), byte_alignment=16
        )
        # In the split path these arguments are unused: no attention role runs.
        # Alias existing storage instead of reserving the 24 KiB shallow ring.
        if cutlass.const_expr(self.attn):
            sQatt = smem.allocate_tensor(
                element_type=cutlass.Float8E4M3FN,
                layout=cute.make_layout(A.Q_SLICE_BYTES),
                byte_alignment=1024,
            )
            sKVatt = smem.allocate_tensor(
                element_type=cutlass.Float8E4M3FN,
                layout=cute.make_layout(A.SLOT_BYTES),
                byte_alignment=1024,
            )
        else:
            sQatt = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, dtype=cutlass.Float8E4M3FN),
                cute.make_layout(A.Q_SLICE_BYTES),
            )
            sKVatt = cute.make_tensor(
                cute.recast_ptr(sKV_0.iterator, dtype=cutlass.Float8E4M3FN),
                cute.make_layout(A.SLOT_BYTES),
            )
        sAbar = smem.allocate_tensor(
            element_type=Int64, layout=cute.make_layout(16), byte_alignment=16
        )
        sMsk = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(8), byte_alignment=16
        )
        sRS = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(16), byte_alignment=16
        )
        smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(ATTQ + 4), byte_alignment=16
        )
        sKr = smem.allocate_tensor(
            element_type=Uint32, layout=cute.make_layout(4), byte_alignment=16
        )
        sTb = smem.allocate_tensor(
            element_type=Uint32, layout=cute.make_layout(8), byte_alignment=16
        )
        sCk = smem.allocate_tensor(
            element_type=Uint64, layout=cute.make_layout(L.QUADC), byte_alignment=16
        )
        sHead = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(8), byte_alignment=128
        )
        sTail = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(8), byte_alignment=128
        )
        sTw = smem.allocate_tensor(
            element_type=Uint32, layout=cute.make_layout(8), byte_alignment=128
        )
        sCnt = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(4), byte_alignment=16
        )
        sRes = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(8), byte_alignment=16
        )
        sRed = smem.allocate_tensor(
            element_type=Uint32, layout=cute.make_layout(8), byte_alignment=16
        )
        sWc = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(8), byte_alignment=16
        )
        sScal = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(32), byte_alignment=16
        )
        sHQ = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(16), byte_alignment=16
        )
        sRres = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(8), byte_alignment=16
        )
        sRred = smem.allocate_tensor(
            element_type=Uint32, layout=cute.make_layout(8), byte_alignment=16
        )
        sRwc = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(8), byte_alignment=16
        )
        sGv = cute.make_tensor(
            cute.recast_ptr(sOut.iterator, dtype=Int32), cute.make_layout(2 * K_TOP)
        )
        sDQ = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout(self.dq_cap if self.defer else 4),
            byte_alignment=16,
        )
        sRingBar = smem.allocate_tensor(
            element_type=Int64,
            layout=cute.make_layout(2 * 8 * (RING // RING_CHUNK)),
            byte_alignment=16,
        )
        if tidx < 2 * 8 * (RING // RING_CHUNK):
            prims.mbarrier_init(sRingBar.iterator.raw_ptr() + tidx, 32)
        # full[2], empty[2], final row scratch ownership, TMEM-address ready.
        sRowBar = smem.allocate_tensor(
            element_type=Int64, layout=cute.make_layout(6), byte_alignment=16
        )
        if tidx == 0:
            for i in cutlass.range_constexpr(5):
                prims.mbarrier_init(sRowBar.iterator.raw_ptr() + i, 128)
            prims.mbarrier_init(sRowBar.iterator.raw_ptr() + 5, 1)
        prims.fence_mbarrier_init()
        if tidx < 8:
            sHead[tidx] = Int32(0)
            sTail[tidx] = Int32(0)
        if cutlass.const_expr(self.dbg_addr):
            if tidx == 0:
                bufs = [
                    sKV_0,
                    sKV_1,
                    sQ,
                    sW,
                    sScales_0,
                    sScales_1,
                    s_ring,
                    sMain,
                    sOut,
                    sHand,
                    sHB,
                    sAttB,
                    sAttO,
                    sHI,
                    sQatt,
                    sKVatt,
                    sAbar,
                    sMsk,
                ]
                for k in cutlass.range_constexpr(len(bufs)):
                    mStamps[bidz * 256 + 64 + k] = cutlass.Int64(bufs[k].iterator.toint())
        if tidx < 8:
            sTw[tidx] = Uint32(0)
        if tidx == 0:
            sCnt[0] = Int32(0)
        if tidx < 16:
            sHI[tidx] = Int32(0)
        if cutlass.const_expr(self.attn):
            if warp_idx == 16:
                if prims.elect_sync():
                    for i in cutlass.range_constexpr(15):
                        prims.mbarrier_init(sAbar.iterator.raw_ptr() + i, 1)
                prims.fence_mbarrier_init()
        a_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        sKV_0_for_tma = cute.make_tensor(sKV_0.iterator, a_tma_view_layout.outer)
        sKV_1_for_tma = cute.make_tensor(sKV_1.iterator, a_tma_view_layout.outer)
        gA = cute.local_tile(mA_mkl, (self.phys_block_kv, self.head_dim), coord=(None, None, None))
        tAsA_0, tAgA_0 = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1), sKV_0_for_tma, cute.group_modes(gA, 0, 2)
        )
        tAsA_1, tAgA_1 = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1), sKV_1_for_tma, cute.group_modes(gA, 0, 2)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        tCgB = thr_mma.partition_B(gB_nkl)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tBgB = tBgB[None, 0, None, None]
        w_cta_layout = cute.make_layout((1,))
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w, 0, w_cta_layout, cute.group_modes(sW, 0, 1), cute.group_modes(mW_tma, 0, 1)
        )
        s_tma_view_layout = cute.make_layout(
            (self.phys_block_kv, self.num_blocks_per_mma, self.num_kv_stages),
            stride=(1, self.phys_block_kv, self.block_kv),
        )
        sScales_0_for_tma = cute.make_tensor(sScales_0.iterator, s_tma_view_layout)
        sScales_1_for_tma = cute.make_tensor(sScales_1.iterator, s_tma_view_layout)
        gS = cute.local_tile(mS_tma, (self.phys_block_kv,), coord=(None, None))
        tSsS_0, tSgS_0 = cpasync.tma_partition(
            tma_atom_s, 0, cute.make_layout(1), sScales_0_for_tma, gS
        )
        tSsS_1, tSgS_1 = cpasync.tma_partition(
            tma_atom_s, 0, cute.make_layout(1), sScales_1_for_tma, gS
        )
        tCrA_0 = tiled_mma.make_fragment_A(sKV_0)
        tCrA_1 = tiled_mma.make_fragment_A(sKV_1)
        tCrB = tiled_mma.make_fragment_B(sQ)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        us = self.num_umma_stages
        cols = self.num_tmem_alloc_cols
        acc_shape_staged = cute.append(acc_shape, us)
        tCtAcc_fake_staged = tiled_mma.make_fragment_C(acc_shape_staged)
        cols_per_group = cols * us * (32 // self.acc_dtype.width)
        num_tmem_alloc_cols_total = self.num_tmem_alloc_cols_total
        c_layout = cutlass.utils.LayoutEnum.ROW_MAJOR
        epi_sub_mn = (epi_tile[0], 32)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            c_layout,
            self.acc_dtype,
            self.acc_dtype,
            epi_sub_mn,
            use_2cta_instrs,
        )
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        # All participants retrieve TMEM at the same static barrier site.
        # Separate waits inside the role branches violate aligned-barrier rules.
        if cutlass.const_expr(self.attn):
            tmem.allocate(512)
        else:
            tmem.allocate(num_tmem_alloc_cols_total)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        if warpgroup_idx == 2:
            cute.arch.warpgroup_reg_dealloc(24)
        if is_tma_warp_0:
            pol = None
            if cutlass.const_expr(self.policy != CP.POL_NONE):
                pol = CP.make_l2_policy(self.policy)
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx, next_kv_idx, next_num_kv, phase, stop_kv, has_work, rot_on = (
                self._walk_init(
                    start_q, start_kv_half, end_q_idx, end_kv_half, mContextLens, batch_size
                )
            )
            q_idx = batch_size
            lane_idx = tidx % 32
            cached_blks = [cutlass.Int32(0) for _ in range(NUM_BLOCKS_PER_MMA)]
            kv_blk_ptr = cutlass.Int32(32)
            bt_cols = cute.size(mBlockTable, mode=[1])
            if has_work:
                q_pipeline.producer_acquire(q_prod_state)
                q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                cute.copy(
                    tma_atom_b,
                    tBgB[None, 0, next_q_idx],
                    tBsB[None, q_prod_state.index],
                    tma_bar_ptr=q_bar,
                    mcast_mask=b_mcast_mask,
                )
                cute.copy(
                    tma_atom_w,
                    tWgW[None, next_q_idx],
                    tWsW[None, q_prod_state.index],
                    tma_bar_ptr=q_bar,
                )
                q_prod_state.advance()
            seg_s = Int32(0)
            while has_work:
                q_idx_old = q_idx
                q_idx = next_q_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv
                if q_idx != q_idx_old:
                    if cutlass.const_expr(self.desc):
                        seg_s = self._seg_s(kv_idx, num_kv, q_idx, end_q_idx, end_kv_idx)
                    kv_blk_ptr = cutlass.Int32(32)
                    prefetch_next, pf_ok = self._walk_qnext(
                        q_idx, phase, rot_on, start_q, end_q_idx, end_kv_idx
                    )
                    if pf_ok != Int32(0):
                        q_pipeline.producer_acquire(q_prod_state)
                        q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                        cute.copy(
                            tma_atom_b,
                            tBgB[None, 0, prefetch_next],
                            tBsB[None, q_prod_state.index],
                            tma_bar_ptr=q_bar,
                            mcast_mask=b_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_w,
                            tWgW[None, prefetch_next],
                            tWsW[None, q_prod_state.index],
                            tma_bar_ptr=q_bar,
                        )
                        q_prod_state.advance()
                # Keep row metadata and Q handoffs outside the per-tile loop.
                # One dummy pair is still issued for a zero-length row.
                kv_go = cutlass.Boolean(True)
                while kv_go:
                    kv_d = kv_idx
                    if cutlass.const_expr(self.desc):
                        kv_d = seg_s - kv_idx
                    if kv_blk_ptr == 32:
                        kv_blk_ptr = cutlass.Int32(0)
                        prefetch_kv = kv_idx + lane_idx * NUM_MATH_WG
                        if cutlass.const_expr(self.desc):
                            prefetch_kv = kv_d - lane_idx * NUM_MATH_WG
                        base_phys = prefetch_kv * NUM_BLOCKS_PER_MMA
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cached_blks[i] = mBlockTable[
                                q_idx, cutlass.min(cutlass.max(base_phys + i, 0), bt_cols - 1)
                            ]
                    tile_ok = kv_d + 0 < num_kv
                    phys_blks = [cutlass.Int32(0)] * NUM_BLOCKS_PER_MMA
                    for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                        pv = cute.arch.shuffle_sync(cached_blks[i], kv_blk_ptr)
                        phys_blks[i] = cutlass.select_(tile_ok, pv, cutlass.Int32(0))
                    kv_blk_ptr = kv_blk_ptr + 1
                    kv_pipeline_0.producer_acquire(kv_prod_state_0)
                    bar = kv_pipeline_0.producer_get_barrier(kv_prod_state_0)
                    stage = kv_prod_state_0.index
                    for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                        cute.copy(
                            tma_atom_a,
                            tAgA_0[None, 0, 0, phys_blks[i]],
                            tAsA_0[None, i, stage],
                            tma_bar_ptr=bar,
                            mcast_mask=a_mcast_mask,
                            cache_policy=pol,
                        )
                        cute.copy(
                            tma_atom_s,
                            tSgS_0[None, 0, phys_blks[i]],
                            tSsS_0[None, i, stage],
                            tma_bar_ptr=bar,
                            cache_policy=pol,
                        )
                    kv_prod_state_0.advance()
                    kv_idx = kv_idx + NUM_MATH_WG
                    kv_go = (kv_idx < num_kv) & ((q_idx != end_q_idx) | (kv_idx != end_kv_idx))
                next_q_idx, next_kv_idx, next_num_kv, phase, has_work = self._walk_next(
                    q_idx,
                    kv_idx,
                    num_kv,
                    next_num_kv,
                    phase,
                    stop_kv,
                    start_q,
                    start_kv_half,
                    end_q_idx,
                    end_kv_idx,
                    mContextLens,
                    batch_size,
                )
        elif is_tma_warp_1:
            pol = None
            if cutlass.const_expr(self.policy != CP.POL_NONE):
                pol = CP.make_l2_policy(self.policy)
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx, next_kv_idx, next_num_kv, phase, stop_kv, has_work, rot_on = (
                self._walk_init(
                    start_q, start_kv_half, end_q_idx, end_kv_half, mContextLens, batch_size
                )
            )
            q_idx = batch_size
            lane_idx = tidx % 32
            cached_blks = [cutlass.Int32(0) for _ in range(NUM_BLOCKS_PER_MMA)]
            kv_blk_ptr = cutlass.Int32(32)
            bt_cols = cute.size(mBlockTable, mode=[1])
            seg_s = Int32(0)
            while has_work:
                q_idx_old = q_idx
                q_idx = next_q_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv
                if q_idx != q_idx_old:
                    if cutlass.const_expr(self.desc):
                        seg_s = self._seg_s(kv_idx, num_kv, q_idx, end_q_idx, end_kv_idx)
                    kv_blk_ptr = cutlass.Int32(32)
                # Keep row metadata and Q handoffs outside the per-tile loop.
                # One dummy pair is still issued for a zero-length row.
                kv_go = cutlass.Boolean(True)
                while kv_go:
                    kv_d = kv_idx
                    if cutlass.const_expr(self.desc):
                        kv_d = seg_s - kv_idx
                    if kv_blk_ptr == 32:
                        kv_blk_ptr = cutlass.Int32(0)
                        prefetch_kv = kv_idx + 1 + lane_idx * NUM_MATH_WG
                        if cutlass.const_expr(self.desc):
                            prefetch_kv = kv_d + 1 - lane_idx * NUM_MATH_WG
                        base_phys = prefetch_kv * NUM_BLOCKS_PER_MMA
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cached_blks[i] = mBlockTable[
                                q_idx, cutlass.min(cutlass.max(base_phys + i, 0), bt_cols - 1)
                            ]
                    tile_ok = kv_d + 1 < num_kv
                    phys_blks = [cutlass.Int32(0)] * NUM_BLOCKS_PER_MMA
                    for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                        pv = cute.arch.shuffle_sync(cached_blks[i], kv_blk_ptr)
                        phys_blks[i] = cutlass.select_(tile_ok, pv, cutlass.Int32(0))
                    kv_blk_ptr = kv_blk_ptr + 1
                    kv_pipeline_1.producer_acquire(kv_prod_state_1)
                    bar = kv_pipeline_1.producer_get_barrier(kv_prod_state_1)
                    stage = kv_prod_state_1.index
                    for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                        cute.copy(
                            tma_atom_a,
                            tAgA_1[None, 0, 0, phys_blks[i]],
                            tAsA_1[None, i, stage],
                            tma_bar_ptr=bar,
                            mcast_mask=a_mcast_mask,
                            cache_policy=pol,
                        )
                        cute.copy(
                            tma_atom_s,
                            tSgS_1[None, 0, phys_blks[i]],
                            tSsS_1[None, i, stage],
                            tma_bar_ptr=bar,
                            cache_policy=pol,
                        )
                    kv_prod_state_1.advance()
                    kv_idx = kv_idx + NUM_MATH_WG
                    kv_go = (kv_idx < num_kv) & ((q_idx != end_q_idx) | (kv_idx != end_kv_idx))
                next_q_idx, next_kv_idx, next_num_kv, phase, has_work = self._walk_next(
                    q_idx,
                    kv_idx,
                    num_kv,
                    next_num_kv,
                    phase,
                    stop_kv,
                    start_q,
                    start_kv_half,
                    end_q_idx,
                    end_kv_idx,
                    mContextLens,
                    batch_size,
                )
        elif is_umma_warp_0:
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx, next_kv_idx, next_num_kv, phase, stop_kv, has_work, rot_on = (
                self._walk_init(
                    start_q, start_kv_half, end_q_idx, end_kv_half, mContextLens, batch_size
                )
            )
            q_idx = batch_size
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            if is_leader_cta:
                num_k_blocks = cute.size(tCrA_0.shape[2])
                q_stage_0 = cutlass.Int32(0)
                while has_work:
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            q_cons_state_umma_0.advance()
                        q_pipeline.consumer_wait(q_cons_state_umma_0)
                        q_stage_0 = q_cons_state_umma_0.index
                    # Keep row metadata and Q handoffs outside the per-tile loop.
                    # One dummy pair is still issued for a zero-length row.
                    kv_go = cutlass.Boolean(True)
                    while kv_go:
                        kv_pipeline_0.consumer_wait(kv_cons_state_umma_0)
                        umma_pipeline_0.producer_acquire(umma_prod_state_0)
                        if cutlass.const_expr(self.tmem_fences):
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        kv_stage = kv_cons_state_umma_0.index
                        tCtAcc_0 = tCtAcc_base_0[None, None, None, umma_prod_state_0.index]
                        for k_block in cutlass.range_constexpr(num_k_blocks):
                            cute.gemm(
                                tiled_mma,
                                tCtAcc_0,
                                tCrA_0[None, None, k_block, kv_stage],
                                tCrB[None, None, k_block, q_stage_0],
                                tCtAcc_0,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        kv_cons_state_umma_0.advance()
                        umma_pipeline_0.producer_commit(umma_prod_state_0)
                        umma_prod_state_0.advance()
                        kv_idx = kv_idx + NUM_MATH_WG
                        kv_go = (kv_idx < num_kv) & ((q_idx != end_q_idx) | (kv_idx != end_kv_idx))
                    next_q_idx, next_kv_idx, next_num_kv, phase, has_work = self._walk_next(
                        q_idx,
                        kv_idx,
                        num_kv,
                        next_num_kv,
                        phase,
                        stop_kv,
                        start_q,
                        start_kv_half,
                        end_q_idx,
                        end_kv_idx,
                        mContextLens,
                        batch_size,
                    )
        elif is_umma_warp_1:
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx, next_kv_idx, next_num_kv, phase, stop_kv, has_work, rot_on = (
                self._walk_init(
                    start_q, start_kv_half, end_q_idx, end_kv_half, mContextLens, batch_size
                )
            )
            q_idx = batch_size
            tCtAcc_base_1 = cute.make_tensor(tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout)
            if is_leader_cta:
                num_k_blocks_1 = cute.size(tCrA_1.shape[2])
                q_stage_1 = cutlass.Int32(0)
                while has_work:
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            q_cons_state_umma_1.advance()
                        q_pipeline.consumer_wait(q_cons_state_umma_1)
                        q_stage_1 = q_cons_state_umma_1.index
                    # Keep row metadata and Q handoffs outside the per-tile loop.
                    # One dummy pair is still issued for a zero-length row.
                    kv_go = cutlass.Boolean(True)
                    while kv_go:
                        kv_pipeline_1.consumer_wait(kv_cons_state_umma_1)
                        umma_pipeline_1.producer_acquire(umma_prod_state_1)
                        if cutlass.const_expr(self.tmem_fences):
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        kv_stage_1 = kv_cons_state_umma_1.index
                        tCtAcc_1 = tCtAcc_base_1[None, None, None, umma_prod_state_1.index]
                        for k_block in cutlass.range_constexpr(num_k_blocks_1):
                            cute.gemm(
                                tiled_mma,
                                tCtAcc_1,
                                tCrA_1[None, None, k_block, kv_stage_1],
                                tCrB[None, None, k_block, q_stage_1],
                                tCtAcc_1,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        kv_cons_state_umma_1.advance()
                        umma_pipeline_1.producer_commit(umma_prod_state_1)
                        umma_prod_state_1.advance()
                        kv_idx = kv_idx + NUM_MATH_WG
                        kv_go = (kv_idx < num_kv) & ((q_idx != end_q_idx) | (kv_idx != end_kv_idx))
                    next_q_idx, next_kv_idx, next_num_kv, phase, has_work = self._walk_next(
                        q_idx,
                        kv_idx,
                        num_kv,
                        next_num_kv,
                        phase,
                        stop_kv,
                        start_q,
                        start_kv_half,
                        end_q_idx,
                        end_kv_idx,
                        mContextLens,
                        batch_size,
                    )
        elif is_math_warp:
            cute.arch.warpgroup_reg_alloc(self.math_regs)
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx, next_kv_idx, next_num_kv, phase, stop_kv, has_work, rot_on = (
                self._walk_init(
                    start_q, start_kv_half, end_q_idx, end_kv_half, mContextLens, batch_size
                )
            )
            q_idx = batch_size
            if cutlass.const_expr(self.attn):
                if tidx == 0:
                    sHI[9] = cute.arch.load(
                        storage.tmem_holding_buf.ptr, Int32, sem="relaxed", scope="cta"
                    )
                    prims.mbarrier_arrive(sRowBar.iterator.raw_ptr() + Int32(5))
            local_tidx = tidx % 128
            m_lane = tidx % 32
            cC = cute.make_identity_tensor(epi_sub_mn)
            NUM_W_IN_REG = self.num_w_in_reg
            if cutlass.const_expr(self.debug_dump):
                logits_stride0 = cutlass.Int64(mLogits.layout.stride[0])
                mLogits_flat = cute.make_tensor(
                    mLogits.iterator,
                    cute.make_layout(cute.size(mLogits, mode=[0]) * logits_stride0),
                )
                out_row_off = cutlass.Int64(0)
            grp = warpgroup_idx
            tCtAcc_base_g = cute.make_tensor(
                tmem_ptr + grp * cols_per_group, tCtAcc_fake_staged.layout
            )
            tAcc_ref = tCtAcc_base_g[None, None, None, 0][(None, None), 0, 0]
            tAcc_ref_epi = cute.flat_divide(tAcc_ref, epi_sub_mn)
            tiled_copy_ref = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_ref_epi[None, None, 0, 0])
            thr_copy_ref = tiled_copy_ref.get_slice(local_tidx)
            tTR_cC = thr_copy_ref.partition_D(cC)
            m_coord = tTR_cC[0][0]
            tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)
            w_cache = cute.make_rmem_tensor((max(NUM_W_IN_REG, 1),), self.epi_dtype)
            q_stage_local = cutlass.Int32(0)
            ctx_q = cutlass.Int32(0)
            lt_mask = cutlass.Int32(cute.arch.lanemask_lt())
            ring_off = warp_idx * Int32(RING)
            head = Int32(0)
            seg_ord = Int32(0)
            tw_ptr = sTw.iterator + warp_idx
            full_ptr = sRingBar.iterator.raw_ptr() + warp_idx * Int32(RING // RING_CHUNK)
            empty_ptr = full_ptr + Int32(8 * (RING // RING_CHUNK))
            T = Float32(NEG_INF)
            stall_ns = Int64(0)
            skew_seed = Int32(0)
            if cutlass.const_expr(self.skew_ns > 0):
                skew_seed = Int32(
                    Uint32(cutlass.Int64(cute.arch.globaltimer()) >> Int64(5))
                ) + bidz * Int32(7919)
            seg_s = Int32(0)
            tcount = Int32(0)
            if warpgroup_idx == 0:
                while has_work:
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv
                    if cutlass.const_expr(self.desc):
                        seg_s = self._seg_s(kv_idx, num_kv, q_idx, end_q_idx, end_kv_idx)
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            if cutlass.const_expr(self.pfence):
                                cute.arch.fence_view_async_shared()
                            q_pipeline.consumer_release(q_cons_state)
                            q_cons_state.advance()
                        q_pipeline.consumer_wait(q_cons_state)
                        q_stage_local = q_cons_state.index
                        if cutlass.const_expr(self.debug_dump):
                            out_row_off = cutlass.Int64(q_idx) * logits_stride0
                        for w_j in cutlass.range_constexpr(NUM_W_IN_REG):
                            w_cache[w_j] = sW[w_j, q_stage_local]
                        ctx_q = self._ctx_of(q_idx, mContextLens, batch_size)
                        # A saved floor is valid only for this row segment.
                        T = Float32(NEG_INF)
                    kv_go = cutlass.Boolean(True)
                    while kv_go:
                        kv_pos = kv_idx * block_kv_val + m_coord
                        if cutlass.const_expr(self.desc):
                            kv_pos = (seg_s - kv_idx) * block_kv_val + m_coord
                        kv_ready_0 = kv_pipeline_0.consumer_try_wait(kv_cons_state_math_0)
                        umma_ready_0 = umma_pipeline_0.consumer_try_wait(umma_cons_state_0)
                        kv_pipeline_0.consumer_wait(kv_cons_state_math_0, kv_ready_0)
                        scale_val = sScales_0[m_coord, kv_cons_state_math_0.index]
                        umma_pipeline_0.consumer_wait(umma_cons_state_0, umma_ready_0)
                        if cutlass.const_expr(self.tmem_fences):
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        tCtAcc_c0 = tCtAcc_base_g[None, None, None, umma_cons_state_0.index]
                        tAcc_c0 = tCtAcc_c0[(None, None), 0, 0]
                        tAcc_c0_epi = cute.flat_divide(tAcc_c0, epi_sub_mn)
                        tTR_0 = thr_copy_ref.partition_S(tAcc_c0_epi)
                        if cutlass.const_expr(not self.late_release):
                            scale_val = _consume_f32(scale_val)
                            if cutlass.const_expr(self.pfence):
                                cute.arch.fence_view_async_shared()
                            cute.arch.sync_warp()
                            kv_pipeline_0.consumer_release(kv_cons_state_math_0)
                            kv_cons_state_math_0.advance()
                        cute.copy(tiled_copy_ref, tTR_0[None, None, None, 0, 0], tTR_rAcc)
                        cute.arch.fence_view_async_tmem_load()
                        s0x, s0y, s1x, s1y = self._epi_half(
                            tTR_rAcc,
                            w_cache,
                            0,
                            Float32(0.0),
                            Float32(0.0),
                            Float32(0.0),
                            Float32(0.0),
                        )
                        cute.copy(tiled_copy_ref, tTR_0[None, None, None, 0, 1], tTR_rAcc)
                        cute.arch.fence_view_async_tmem_load()
                        if cutlass.const_expr(self.tmem_fences):
                            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        umma_pipeline_0.consumer_release(umma_cons_state_0)
                        umma_cons_state_0.advance()
                        s0x, s0y, s1x, s1y = self._epi_half(
                            tTR_rAcc, w_cache, 32, s0x, s0y, s1x, s1y
                        )
                        result_t = s0x + s0y + s1x + s1y
                        scale_mul = scale_val * 0.5
                        val = self.output_dtype(result_t * scale_mul)
                        if cutlass.const_expr(self.late_release):
                            if cutlass.const_expr(self.pfence):
                                cute.arch.fence_view_async_shared()
                            cute.arch.sync_warp()
                            kv_pipeline_0.consumer_release(kv_cons_state_math_0)
                            kv_cons_state_math_0.advance()
                        if cutlass.const_expr(self.debug_dump):
                            mLogits_flat[out_row_off + kv_pos] = val
                        if cutlass.const_expr(self.admit):
                            # Stale same-segment floors remain conservative.
                            if (kv_idx & Int32(6)) == Int32(0):
                                T = _floor_of(tw_ptr, seg_ord, m_lane)
                            hit = (kv_pos < ctx_q) & (val >= T)
                            ent = L.pack_entry(L.u32_of_f32(val), kv_pos)
                            head, stall_ns = _push_ring(
                                hit,
                                ent,
                                head,
                                s_ring,
                                ring_off,
                                empty_ptr,
                                full_ptr,
                                m_lane,
                                lt_mask,
                                stall_ns,
                                self.stamps,
                            )
                        kv_idx = kv_idx + NUM_MATH_WG
                        if cutlass.const_expr(self.stamps):
                            tcount = tcount + Int32(1)
                            if tcount & Int32(63) == Int32(0):
                                if warp_idx == Int32(0):
                                    if m_lane == Int32(0):
                                        tslot = tcount >> Int32(6)
                                        if tslot < Int32(94):
                                            mStamps[bidz * 256 + 162 + tslot] = cutlass.Int64(
                                                cute.arch.globaltimer()
                                            )
                        kv_go = (kv_idx < num_kv) & ((q_idx != end_q_idx) | (kv_idx != end_kv_idx))
                    if cutlass.const_expr(self.admit):
                        if cutlass.const_expr(self.skew_ns > 0):
                            L.skew_sleep(
                                skew_seed + warp_idx,
                                seg_ord * Int32(8) + Int32(1),
                                Int32(self.skew_ns),
                            )
                        head = _push_marker(
                            seg_ord, head, s_ring, ring_off, empty_ptr, full_ptr, m_lane
                        )
                    if cutlass.const_expr(self.stamps):
                        if m_lane == Int32(0):
                            if seg_ord < Int32(8):
                                mStamps[bidz * 256 + 32 + warp_idx * 8 + seg_ord] = stall_ns
                                mStamps[bidz * 256 + 96 + seg_ord] = cutlass.Int64(
                                    cute.arch.globaltimer()
                                )
                    seg_ord = seg_ord + 1
                    next_q_idx, next_kv_idx, next_num_kv, phase, has_work = self._walk_next(
                        q_idx,
                        kv_idx,
                        num_kv,
                        next_num_kv,
                        phase,
                        stop_kv,
                        start_q,
                        start_kv_half,
                        end_q_idx,
                        end_kv_idx,
                        mContextLens,
                        batch_size,
                    )
                if q_idx < batch_size:
                    if cutlass.const_expr(self.pfence):
                        cute.arch.fence_view_async_shared()
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()
            else:
                while has_work:
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv
                    if cutlass.const_expr(self.desc):
                        seg_s = self._seg_s(kv_idx, num_kv, q_idx, end_q_idx, end_kv_idx)
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            if cutlass.const_expr(self.pfence):
                                cute.arch.fence_view_async_shared()
                            q_pipeline.consumer_release(q_cons_state)
                            q_cons_state.advance()
                        q_pipeline.consumer_wait(q_cons_state)
                        q_stage_local = q_cons_state.index
                        if cutlass.const_expr(self.debug_dump):
                            out_row_off = cutlass.Int64(q_idx) * logits_stride0
                        for w_j in cutlass.range_constexpr(NUM_W_IN_REG):
                            w_cache[w_j] = sW[w_j, q_stage_local]
                        ctx_q = self._ctx_of(q_idx, mContextLens, batch_size)
                        # A saved floor is valid only for this row segment.
                        T = Float32(NEG_INF)
                    kv_go = cutlass.Boolean(True)
                    while kv_go:
                        kv_idx_1 = kv_idx + 1
                        kv_pos = kv_idx_1 * block_kv_val + m_coord
                        if cutlass.const_expr(self.desc):
                            kv_pos = (seg_s - kv_idx + 1) * block_kv_val + m_coord
                        kv_ready_1 = kv_pipeline_1.consumer_try_wait(kv_cons_state_math_1)
                        umma_ready_1 = umma_pipeline_1.consumer_try_wait(umma_cons_state_1)
                        kv_pipeline_1.consumer_wait(kv_cons_state_math_1, kv_ready_1)
                        scale_val = sScales_1[m_coord, kv_cons_state_math_1.index]
                        umma_pipeline_1.consumer_wait(umma_cons_state_1, umma_ready_1)
                        if cutlass.const_expr(self.tmem_fences):
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        tCtAcc_c1 = tCtAcc_base_g[None, None, None, umma_cons_state_1.index]
                        tAcc_c1 = tCtAcc_c1[(None, None), 0, 0]
                        tAcc_c1_epi = cute.flat_divide(tAcc_c1, epi_sub_mn)
                        tTR_1 = thr_copy_ref.partition_S(tAcc_c1_epi)
                        if cutlass.const_expr(not self.late_release):
                            scale_val = _consume_f32(scale_val)
                            if cutlass.const_expr(self.pfence):
                                cute.arch.fence_view_async_shared()
                            cute.arch.sync_warp()
                            kv_pipeline_1.consumer_release(kv_cons_state_math_1)
                            kv_cons_state_math_1.advance()
                        cute.copy(tiled_copy_ref, tTR_1[None, None, None, 0, 0], tTR_rAcc)
                        cute.arch.fence_view_async_tmem_load()
                        s0x, s0y, s1x, s1y = self._epi_half(
                            tTR_rAcc,
                            w_cache,
                            0,
                            Float32(0.0),
                            Float32(0.0),
                            Float32(0.0),
                            Float32(0.0),
                        )
                        cute.copy(tiled_copy_ref, tTR_1[None, None, None, 0, 1], tTR_rAcc)
                        cute.arch.fence_view_async_tmem_load()
                        if cutlass.const_expr(self.tmem_fences):
                            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        umma_pipeline_1.consumer_release(umma_cons_state_1)
                        umma_cons_state_1.advance()
                        s0x, s0y, s1x, s1y = self._epi_half(
                            tTR_rAcc, w_cache, 32, s0x, s0y, s1x, s1y
                        )
                        result_t = s0x + s0y + s1x + s1y
                        scale_mul = scale_val * 0.5
                        val = self.output_dtype(result_t * scale_mul)
                        if cutlass.const_expr(self.late_release):
                            if cutlass.const_expr(self.pfence):
                                cute.arch.fence_view_async_shared()
                            cute.arch.sync_warp()
                            kv_pipeline_1.consumer_release(kv_cons_state_math_1)
                            kv_cons_state_math_1.advance()
                        if cutlass.const_expr(self.debug_dump):
                            mLogits_flat[out_row_off + kv_pos] = val
                        if cutlass.const_expr(self.admit):
                            # Stale same-segment floors remain conservative.
                            if (kv_idx & Int32(6)) == Int32(0):
                                T = _floor_of(tw_ptr, seg_ord, m_lane)
                            hit = (kv_pos < ctx_q) & (val >= T)
                            ent = L.pack_entry(L.u32_of_f32(val), kv_pos)
                            head, stall_ns = _push_ring(
                                hit,
                                ent,
                                head,
                                s_ring,
                                ring_off,
                                empty_ptr,
                                full_ptr,
                                m_lane,
                                lt_mask,
                                stall_ns,
                                self.stamps,
                            )
                        kv_idx = kv_idx + NUM_MATH_WG
                        if cutlass.const_expr(self.stamps):
                            tcount = tcount + Int32(1)
                            if tcount & Int32(63) == Int32(0):
                                if warp_idx == Int32(0):
                                    if m_lane == Int32(0):
                                        tslot = tcount >> Int32(6)
                                        if tslot < Int32(94):
                                            mStamps[bidz * 256 + 162 + tslot] = cutlass.Int64(
                                                cute.arch.globaltimer()
                                            )
                        kv_go = (kv_idx < num_kv) & ((q_idx != end_q_idx) | (kv_idx != end_kv_idx))
                    if cutlass.const_expr(self.admit):
                        if cutlass.const_expr(self.skew_ns > 0):
                            L.skew_sleep(
                                skew_seed + warp_idx,
                                seg_ord * Int32(8) + Int32(1),
                                Int32(self.skew_ns),
                            )
                        head = _push_marker(
                            seg_ord, head, s_ring, ring_off, empty_ptr, full_ptr, m_lane
                        )
                    if cutlass.const_expr(self.stamps):
                        if m_lane == Int32(0):
                            if seg_ord < Int32(8):
                                mStamps[bidz * 256 + 32 + warp_idx * 8 + seg_ord] = stall_ns
                                mStamps[bidz * 256 + 96 + seg_ord] = cutlass.Int64(
                                    cute.arch.globaltimer()
                                )
                    seg_ord = seg_ord + 1
                    next_q_idx, next_kv_idx, next_num_kv, phase, has_work = self._walk_next(
                        q_idx,
                        kv_idx,
                        num_kv,
                        next_num_kv,
                        phase,
                        stop_kv,
                        start_q,
                        start_kv_half,
                        end_q_idx,
                        end_kv_idx,
                        mContextLens,
                        batch_size,
                    )
                if q_idx < batch_size:
                    if cutlass.const_expr(self.pfence):
                        cute.arch.fence_view_async_shared()
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()
            if cutlass.const_expr(self.stamps):
                if tidx == 0:
                    mStamps[bidz * 256 + 1] = cutlass.Int64(cute.arch.globaltimer())
                if m_lane == Int32(0):
                    mStamps[bidz * 256 + 4 + warp_idx] = stall_ns
            cute.arch.fence_view_async_shared()
            cute.arch.fence_view_async_tmem_load()
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            cute.arch.barrier(barrier_id=3, number_of_threads=self.num_math_threads)
            if cutlass.const_expr(self.attn):
                if tidx == 0:
                    cute.arch.fence_acq_rel_cta()
                    cute.arch.atomic_exch(
                        sHI.iterator + Int32(11), Int32(1), sem="release", scope="cta"
                    )
                cute.arch.barrier(barrier_id=END_BAR, number_of_threads=self.num_math_threads + 128)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
        elif is_sel_warp:
            cute.arch.warpgroup_reg_dealloc(self.sel_regs)
            if cutlass.const_expr(self.admit):
                self._select_role(
                    tidx - 384,
                    bidz,
                    start_q,
                    start_kv_half,
                    end_q_idx,
                    end_kv_half,
                    batch_size,
                    mContextLens,
                    mScheduleMeta,
                    mRowStat,
                    mStamps,
                    s_ring,
                    sMain,
                    sOut,
                    sHist,
                    sCk,
                    sHead,
                    sTail,
                    sTw,
                    sCnt,
                    sRes,
                    sRed,
                    sWc,
                    sScal,
                    sKr,
                    sHI,
                    sHQ,
                    sHand,
                    sAttB,
                    sAttO,
                    sGv,
                    mKvLens,
                    mReq,
                    mBtSwa,
                    mBtC,
                    mTopk,
                    mGidx,
                    mWsSeg,
                    mWsCtr,
                    sKV_0,
                    sQ,
                    sQatt,
                    sKVatt,
                    sAbar,
                    tma_q,
                    tma_swa,
                    tma_cmp,
                    sTb,
                    sRowBar,
                    sRingBar,
                )
        elif is_row_warp:
            if cutlass.const_expr(self.row_regs >= 96):
                cute.arch.warpgroup_reg_alloc(self.row_regs)
            else:
                cute.arch.warpgroup_reg_dealloc(self.row_regs)
            if cutlass.const_expr(self.admit):
                self._row_role(
                    tidx - 512,
                    bidz,
                    mKvLens,
                    mReq,
                    mBtSwa,
                    mBtC,
                    mTopk,
                    mGidx,
                    mWsSeg,
                    mWsCtr,
                    mRowStat,
                    mStamps,
                    sHand,
                    sAttB,
                    sAttO,
                    sHI,
                    sRS,
                    tma_q,
                    tma_swa,
                    tma_cmp,
                    mSinks,
                    mOut,
                    softmax_scale,
                    sQatt,
                    sKVatt,
                    sAbar,
                    sMsk,
                    sKV_0,
                    sQ,
                    start_q,
                    batch_size,
                    mScheduleMeta,
                    sMain,
                    sHQ,
                    sRres,
                    sRred,
                    sRwc,
                    s_ring,
                    sGv,
                    self._nseg(start_q, start_kv_half, end_q_idx, end_kv_half),
                    sDQ,
                    refused,
                    mStatus,
                    sRowBar,
                )
            if cutlass.const_expr(self.attn):
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                cute.arch.barrier_arrive(
                    barrier_id=END_BAR, number_of_threads=self.num_math_threads + 128, aligned=False
                )

    @cute.jit
    def _select_role(
        self,
        sel_tid,
        cta,
        start_q,
        start_kv_half,
        end_q_idx,
        end_kv_half,
        batch_size,
        mContextLens,
        mSched,
        mRowStat,
        mStamps,
        s_ring,
        sMain,
        sOut,
        sHist,
        sCk,
        sHead,
        sTail,
        sTw,
        sCnt,
        sRes,
        sRed,
        sWc,
        sScal,
        sKr,
        sHI,
        sHQ,
        sHand,
        sAttB,
        sAttO,
        sGv,
        mKvLens,
        mReq,
        mBtSwa,
        mBtC,
        mTopk,
        mGidx,
        mWsSeg,
        mWsCtr,
        sKV_0,
        sQ,
        sQatt,
        sKVatt,
        sAbar,
        tma_q,
        tma_swa,
        tma_cmp,
        sTb,
        sRowBar,
        sRingBar,
    ):
        NT: cutlass.Constexpr[int] = SEL_NTHR
        lane = sel_tid & Int32(31)
        swarp = sel_tid >> Int32(5)
        lt_mask = Int32(cute.arch.lanemask_lt())
        r0 = swarp * Int32(2)
        r1 = r0 + Int32(1)
        empty = (start_q == end_q_idx) & (start_kv_half == end_kv_half)
        qb = end_q_idx
        if end_kv_half == Int32(0):
            qb = end_q_idx - Int32(1)
        nseg = qb - start_q + Int32(1)
        if empty:
            nseg = Int32(0)
        tail0 = Int32(0)
        tail1 = Int32(0)
        skew_seed = Int32(0)
        if cutlass.const_expr(self.skew_ns > 0):
            skew_seed = (
                Int32(Uint32(cutlass.Int64(cute.arch.globaltimer()) >> Int64(5)))
                + cta * Int32(7919)
                + Int32(13)
            )
        t_comp = Int64(0)
        prof_t = None
        if cutlass.const_expr(self.stamps):
            prof_t = cute.make_tensor(
                mStamps.iterator + cta * Int32(256) + Int32(16), cute.make_layout(4)
            )
        rot_on = Int32(0)
        if cutlass.const_expr(self.rot):
            if (end_kv_half > Int32(0)) & (end_q_idx > start_q):
                rot_on = Int32(1)
        seg = Int32(0)
        while seg < nseg:
            q = start_q + seg
            if rot_on != Int32(0):
                q = start_q + seg - Int32(1)
                if seg == Int32(0):
                    q = qb
            n = Int32(mContextLens[q])
            bsel = seg & Int32(1)
            sMb = cute.make_tensor(
                sMain.iterator + bsel * Int32(MAIN_CAP), cute.make_layout(MAIN_CAP)
            )
            if swarp == Int32(0):
                Pq = Int32(1)
                pq_ = Int32(0)
                if (q == start_q) | (q == qb):
                    Pq, pq_ = _parts_of_row(q, cta, mSched, Int32(self.num_sms), lane)
                if lane == Int32(0):
                    sScal[0] = Pq
                    sScal[1] = pq_
            # Empty barriers start at phase 0; generation 0 waits against 1.
            A._wait(
                sRowBar.iterator.raw_ptr() + Int32(2) + bsel,
                ((seg >> Int32(1)) & Int32(1)) ^ Int32(1),
            )
            if sel_tid == Int32(0):
                sCnt[0] = Int32(0)
                sKr[0] = Uint32(4294967295)
                sKr[1] = Uint32(0)
                if cutlass.const_expr(self.defer):
                    t_lo = Int32(0)
                    if q == start_q:
                        t_lo = start_kv_half * Int32(2)
                    t_hi = (n + Int32(self.block_kv - 1)) // Int32(self.block_kv) + Int32(
                        1
                    ) & Int32(-2)
                    if q == end_q_idx:
                        t_hi = end_kv_half * Int32(2)
                    cute.arch.atomic_exch(
                        sHI.iterator + Int32(5), t_hi - t_lo, sem="release", scope="cta"
                    )
            if sel_tid < Int32(8):
                cute.arch.atomic_exch(
                    sTw.iterator + sel_tid, Uint32(seg), sem="relaxed", scope="cta"
                )
            L.bar(SEL_BAR, NT)
            P = sScal[0]
            p = sScal[1]
            kth_key = Uint64(0)
            done0 = Int32(0)
            done1 = Int32(0)
            adm = Int32(0)
            ncomp = Int32(0)
            alldone = Int32(0)
            kf_done = Int32(0)
            while alldone == Int32(0):
                cnow = cute.arch.load(sCnt.iterator, Int32, sem="relaxed", scope="cta")
                mydone = done0 & done1
                trg = Int32(self.trig)
                if cutlass.const_expr(self.trig0 > 0):
                    if ncomp == Int32(0):
                        trg = Int32(self.trig0)
                lite = Int32(0)
                if cutlass.const_expr(self.kfloor):
                    if (kf_done == Int32(0)) & (ncomp == Int32(0)) & (cnow >= Int32(K_TOP)):
                        lite = Int32(1)
                if (cnow >= trg) | (mydone != Int32(0)) | (lite != Int32(0)):
                    if lane == Int32(0):
                        sScal[4 + swarp] = done0 + done1
                    L.bar(SEL_BAR, NT)
                    c = sCnt[0]
                    nd = sScal[4] + sScal[5] + sScal[6] + sScal[7]
                    L.bar(SEL_BAR, NT)
                    if c >= trg:
                        if cutlass.const_expr(self.skew_ns > 0):
                            if sel_tid == Int32(0):
                                L.skew_sleep(
                                    skew_seed, ncomp * Int32(16) + seg, Int32(self.skew_ns)
                                )
                        t_c0 = Int64(0)
                        if cutlass.const_expr(self.stamps):
                            t_c0 = cutlass.Int64(cute.arch.globaltimer())
                        tw_e = None
                        if cutlass.const_expr(self.efl):
                            tw_e = sTw.iterator
                        kept, tf = _compact_approx(
                            sMb,
                            c,
                            sHist,
                            sRes,
                            sCnt.iterator,
                            sKr.iterator,
                            sKr.iterator + 1,
                            sKr.iterator + 2,
                            sScal,
                            sel_tid,
                            NT,
                            SEL_BAR,
                            prof_t,
                            min(self.keep_cap, self.trig - 1),
                            self.cbins,
                            tw_e,
                            seg,
                            self.tiebin,
                            sTb,
                        )
                        if kept < Int32(0):
                            no_, kth, m_, inf_ = L.select_topk_list(
                                sMb,
                                c,
                                Int32(K_TOP),
                                sOut,
                                sHist,
                                sCk,
                                sRes,
                                sRed,
                                sWc,
                                sel_tid,
                                False,
                                NT,
                                SEL_BAR,
                                True,
                            )
                            ic = sel_tid
                            while ic < Int32(K_TOP):
                                sMb[ic] = Uint64(sOut[ic])
                                ic = ic + Int32(NT)
                            tf = Uint32(kth >> Uint64(32))
                            L.bar(SEL_BAR, NT)
                            if sel_tid == Int32(0):
                                sCnt[0] = Int32(K_TOP)
                                sKr[0] = tf
                            L.bar(SEL_BAR, NT)
                        kth_key = Uint64(tf) << Uint64(32)
                        if sel_tid < Int32(8):
                            cute.arch.atomic_exch(
                                sTw.iterator + sel_tid,
                                _pack_floor_key(tf, seg),
                                sem="relaxed",
                                scope="cta",
                            )
                        ncomp = ncomp + Int32(1)
                        if cutlass.const_expr(self.stamps):
                            t_comp = t_comp + (cutlass.Int64(cute.arch.globaltimer()) - t_c0)
                    elif nd == Int32(8):
                        alldone = Int32(1)
                    if cutlass.const_expr(self.kfloor):
                        if (
                            (kf_done == Int32(0))
                            & (ncomp == Int32(0))
                            & (alldone == Int32(0))
                            & (c >= Int32(K_TOP))
                        ):
                            kf_done = Int32(1)
                            tfk = Uint32(sKr[0])
                            kth_key = Uint64(tfk) << Uint64(32)
                            if sel_tid < Int32(8):
                                cute.arch.atomic_exch(
                                    sTw.iterator + sel_tid,
                                    _pack_floor_key(tfk, seg),
                                    sem="relaxed",
                                    scope="cta",
                                )
                            L.bar(SEL_BAR, NT)
                else:
                    tail0, done0, u0, g0 = _drain_ring(
                        r0,
                        tail0,
                        done0,
                        kth_key,
                        s_ring,
                        sRingBar.iterator.raw_ptr() + r0 * Int32(RING // RING_CHUNK),
                        sRingBar.iterator.raw_ptr()
                        + Int32(8 * (RING // RING_CHUNK))
                        + r0 * Int32(RING // RING_CHUNK),
                        sMb,
                        sCnt.iterator,
                        lane,
                        lt_mask,
                        sKr.iterator,
                        sKr.iterator + 1,
                    )
                    tail1, done1, u1, g1 = _drain_ring(
                        r1,
                        tail1,
                        done1,
                        kth_key,
                        s_ring,
                        sRingBar.iterator.raw_ptr() + r1 * Int32(RING // RING_CHUNK),
                        sRingBar.iterator.raw_ptr()
                        + Int32(8 * (RING // RING_CHUNK))
                        + r1 * Int32(RING // RING_CHUNK),
                        sMb,
                        sCnt.iterator,
                        lane,
                        lt_mask,
                        sKr.iterator,
                        sKr.iterator + 1,
                    )
                    adm = adm + u0 + u1
                    if g0 | g1 == Int32(0):
                        L.nanosleep(Int32(self.sel_idle_ns))
            if cutlass.const_expr(self.stamps):
                if sel_tid == Int32(0):
                    if seg == nseg - Int32(1):
                        mStamps[cta * 256 + 2] = cutlass.Int64(cute.arch.globaltimer())
                    if seg < Int32(8):
                        mStamps[cta * 256 + 104 + seg] = cutlass.Int64(cute.arch.globaltimer())
            cfin = sCnt[0]
            if lane == Int32(0):
                sScal[12 + swarp] = adm
            if seg == nseg - Int32(1):
                cute.arch.fence_view_async_shared()
            L.bar(SEL_BAR, NT)
            pref = Int32(0)
            sdone = Int32(1)
            if seg == nseg - Int32(1):
                if sel_tid == Int32(0):
                    cute.arch.fence_acq_rel_cta()
                    cute.arch.atomic_exch(
                        sHI.iterator + Int32(3), Int32(1), sem="release", scope="cta"
                    )
                if cutlass.const_expr(self.etile):
                    if sel_tid == Int32(0):
                        az = cute.arch.load(
                            sHI.iterator + Int32(15), Int32, sem="acquire", scope="cta"
                        )
                        og = cute.arch.load(
                            sHI.iterator + Int32(4), Int32, sem="acquire", scope="cta"
                        )
                        dk = A.deep_ok(sHI.iterator)
                        ear = Int32(0)
                        if (az >= seg) & (dk != Int32(0)) & (og != Int32(0)):
                            ear = Int32(1)
                        sScal[9] = ear
                    L.bar(SEL_BAR, NT)
                    if sScal[9] != Int32(0):
                        sGv[sel_tid] = L.l2g_swa(
                            sel_tid,
                            Int32(mKvLens[q]) - Int32(1),
                            Int32(mReq[q]),
                            mBtSwa,
                            Int32(cute.size(mBtSwa, mode=[1])),
                            Int32(0),
                        )
                        L.bar(SEL_BAR, NT)
                        abar_e = sAbar.iterator.raw_ptr()
                        A.issue_row_loads(
                            sel_tid,
                            q,
                            sGv,
                            cute.recast_ptr(sKV_0.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr(),
                            sKVatt.iterator.raw_ptr(),
                            cute.recast_ptr(s_ring.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr(),
                            cute.recast_ptr(sQ.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr(),
                            sQatt.iterator.raw_ptr(),
                            abar_e,
                            abar_e + Int32(4),
                            abar_e + Int32(13),
                            tma_q,
                            tma_swa,
                            tma_cmp,
                            SEL_BAR,
                            Int32(0),
                            Int32(1),
                        )
                        if sel_tid == Int32(0):
                            sHQ[14] = q
                            cute.arch.fence_acq_rel_cta()
                            cute.arch.atomic_exch(
                                sHI.iterator + Int32(1), Int32(1), sem="release", scope="cta"
                            )
                if cutlass.const_expr(self.sel_fin):
                    if seg > Int32(0):
                        # One-shot transfer after every earlier row finalizer.
                        A._wait(sRowBar.iterator.raw_ptr() + Int32(4), Int32(0))
                    L.bar(SEL_BAR, NT)
                    pref = Int32(1)
                    sdone, s0, s1, s2, s3, s4 = _finalize_seg(
                        sel_tid,
                        SEL_BAR,
                        cta,
                        start_q,
                        q,
                        n,
                        P,
                        p,
                        sMb,
                        cfin,
                        None,
                        sHand,
                        sAttB,
                        sAttO,
                        sHist,
                        sCk,
                        sRes,
                        sRed,
                        sWc,
                        cute.make_tensor(sScal.iterator + Int32(8), cute.make_layout(1)),
                        mKvLens,
                        mReq,
                        mBtSwa,
                        mBtC,
                        mTopk,
                        mGidx,
                        mRowStat,
                        mSched,
                        mWsSeg.iterator.toint(),
                        mWsCtr.iterator.toint(),
                        Int32(cute.size(mBtSwa, mode=[1])),
                        Int32(cute.size(mBtC, mode=[1])),
                        self.num_sms,
                        self.skew_ns,
                        skew_seed,
                        seg * Int32(16) + Int32(7),
                        self.stamps,
                        mStamps,
                        cta * Int32(256) + Int32(153),
                        None,
                        0,
                        None,
                        self.rsort,
                    )
                    if sdone != Int32(0):
                        sGv[sel_tid] = s0
                        sGv[sel_tid + Int32(128)] = s1
                        sGv[sel_tid + Int32(256)] = s2
                        sGv[sel_tid + Int32(384)] = s3
                        sGv[sel_tid + Int32(512)] = s4
                    L.bar(SEL_BAR, NT)
            if sel_tid == Int32(0):
                hs = bsel * Int32(8)
                sHQ[hs + Int32(0)] = q
                sHQ[hs + Int32(1)] = n
                sHQ[hs + Int32(2)] = P
                sHQ[hs + Int32(3)] = p
                sHQ[hs + Int32(4)] = cfin
                sHQ[hs + Int32(5)] = bsel
                sHQ[hs + Int32(6)] = pref
                sHQ[hs + Int32(7)] = Int32(
                    Uint32(cutlass.Int64(cute.arch.globaltimer()) >> Int64(10))
                )
                if cutlass.const_expr(self.skew_ns > 0):
                    L.skew_sleep(skew_seed, seg * Int32(16) + Int32(5), Int32(self.skew_ns))
                if sdone != Int32(0):
                    cute.arch.fence_acq_rel_cta()
                    cute.arch.atomic_exch(sHI.iterator, seg + Int32(1), sem="release", scope="cta")
                if cutlass.const_expr(self.stamps):
                    if seg < Int32(8):
                        mStamps[cta * 256 + 112 + seg] = cutlass.Int64(cute.arch.globaltimer())
            if sdone != Int32(0):
                # Every selector releases its own sMain/sGv writes; thread 0
                # also publishes sHQ before its arrival completes this phase.
                prims.mbarrier_arrive(sRowBar.iterator.raw_ptr() + bsel)
            if cutlass.const_expr(self.gath):
                if (pref != Int32(0)) & (sdone != Int32(0)):
                    if sel_tid == Int32(0):
                        az = cute.arch.load(
                            sHI.iterator + Int32(15), Int32, sem="acquire", scope="cta"
                        )
                        og = cute.arch.load(
                            sHI.iterator + Int32(4), Int32, sem="acquire", scope="cta"
                        )
                        dk = A.deep_ok(sHI.iterator)
                        while (az < seg) | (dk == Int32(0)) | (og == Int32(0)):
                            L.nanosleep(Int32(64))
                            az = cute.arch.load(
                                sHI.iterator + Int32(15), Int32, sem="acquire", scope="cta"
                            )
                            og = cute.arch.load(
                                sHI.iterator + Int32(4), Int32, sem="acquire", scope="cta"
                            )
                            dk = A.deep_ok(sHI.iterator)
                        cute.arch.fence_acq_rel_cta()
                        if cutlass.const_expr(self.stamps):
                            mStamps[cta * 256 + 158] = cutlass.Int64(cute.arch.globaltimer())
                    L.bar(SEL_BAR, NT)
                    abar_s = sAbar.iterator.raw_ptr()
                    jfirst = Int32(0)
                    if cutlass.const_expr(self.etile):
                        jfirst = sScal[9]
                    A.issue_row_loads(
                        sel_tid,
                        q,
                        sGv,
                        cute.recast_ptr(sKV_0.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr(),
                        sKVatt.iterator.raw_ptr(),
                        cute.recast_ptr(s_ring.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr(),
                        cute.recast_ptr(sQ.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr(),
                        sQatt.iterator.raw_ptr(),
                        abar_s,
                        abar_s + Int32(4),
                        abar_s + Int32(13),
                        tma_q,
                        tma_swa,
                        tma_cmp,
                        SEL_BAR,
                        jfirst,
                        Int32(A.NTILES),
                    )
                    if cutlass.const_expr(self.stamps):
                        if sel_tid == Int32(0):
                            mStamps[cta * 256 + 159] = cutlass.Int64(cute.arch.globaltimer())
            seg = seg + Int32(1)
        if sel_tid == Int32(0):
            cute.arch.fence_acq_rel_cta()
            cute.arch.atomic_exch(sHI.iterator + Int32(3), Int32(1), sem="release", scope="cta")
            cute.arch.atomic_exch(sHI.iterator + Int32(2), Int32(1), sem="release", scope="cta")
        if cutlass.const_expr(self.stamps):
            if sel_tid == Int32(0):
                mStamps[cta * 256 + 3] = cutlass.Int64(cute.arch.globaltimer())
                mStamps[cta * 256 + 12] = t_comp

    @cute.jit
    def _row_role(
        self,
        rtid,
        cta,
        mKvLens,
        mReq,
        mBtSwa,
        mBtC,
        mTopk,
        mGidx,
        mWsSeg,
        mWsCtr,
        mRowStat,
        mStamps,
        sHand,
        sAttB,
        sAttO,
        sHI,
        sRS,
        tma_q,
        tma_swa,
        tma_cmp,
        mSinks,
        mOut,
        softmax_scale,
        sQatt,
        sKVatt,
        sAbar,
        sMsk,
        sKV_0,
        sQ,
        start_q,
        batch_size,
        mSched,
        sMain,
        sHQ,
        sRres,
        sRred,
        sRwc,
        s_ring,
        sGv,
        nseg_r,
        sDQ,
        refused,
        mStatus,
        sRowBar,
    ):
        NT: cutlass.Constexpr[int] = SEL_NTHR
        mbs = Int32(cute.size(mBtSwa, mode=[1]))
        mbc = Int32(cute.size(mBtC, mode=[1]))
        ws_seg_base = mWsSeg.iterator.toint()
        ws_ctr_base = mWsCtr.iterator.toint()
        nh = Int32(0)
        live = Int32(1)
        aph = Int32(0)
        skew_seed = Int32(0)
        if cutlass.const_expr(self.skew_ns > 0):
            skew_seed = (
                Int32(Uint32(cutlass.Int64(cute.arch.globaltimer()) >> Int64(5)))
                + cta * Int32(7919)
                + Int32(29)
            )
        scale_log2 = softmax_scale * Float32(A.LOG2_E)
        tmem_att = Int32(0)
        q_slot0 = sQatt.iterator.raw_ptr()
        kv_slot0 = sKVatt.iterator.raw_ptr()
        deep_base = cute.recast_ptr(sKV_0.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr()
        deep_q = cute.recast_ptr(sQ.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr()
        abar = sAbar.iterator.raw_ptr()
        out_addr = mOut.iterator.toint()
        sink_base = mSinks.iterator.raw_ptr()
        kv_slot7 = cute.recast_ptr(s_ring.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr()
        main_b = cute.recast_ptr(sMain.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr()
        sHistR = cute.make_tensor(
            cute.recast_ptr(sAttB.iterator, dtype=Int32), cute.make_layout(L.NB_SEL)
        )
        sCkR = cute.make_tensor(sAttO.iterator, cute.make_layout(L.QUADC))
        oend = cutlass.min(mSched[0, 0], batch_size)
        if refused != Int32(0):
            oend = Int32(0)
            rb = cta
            while rb < batch_size:
                ic = rtid
                while ic < Int32(K_TOP):
                    mTopk[rb, ic] = Int32(-1)
                    ic = ic + Int32(NT)
                ic = rtid
                while ic < Int32(640):
                    mGidx[rb, ic] = Int32(-1)
                    ic = ic + Int32(NT)
                if cutlass.const_expr(self.attn):
                    ob = mOut.iterator.toint() + Int64(rb) * Int64(A.HEADS * A.LATENT * 2)
                    iw = rtid
                    while iw < Int32(A.HEADS * A.LATENT // 2):
                        cutlass.inttoptr(ob + Int64(iw) * Int64(4), mem_space=1, dtype=Int32).store(
                            Int32(2143322048)
                        )
                        iw = iw + Int32(NT)
                L.bar(ROW_BAR, NT)
                if rtid == Int32(0):
                    L.publish_row(mRowStat.iterator.toint(), rb)
                rb = rb + Int32(self.num_sms)
        if cta == Int32(0):
            if rtid == Int32(0):
                mStatus[0] = refused
        if cta < oend:
            oq = cta
            while oq < oend:
                opos = Int32(mKvLens[oq]) - Int32(1)
                oreq = Int32(mReq[oq])
                og0, og1, og2, og3, og4 = _write_row_outputs_regs(
                    oq,
                    Int32(0),
                    sHand,
                    Int32(0),
                    Int32(1),
                    mTopk,
                    mGidx,
                    mBtSwa,
                    mBtC,
                    opos,
                    oreq,
                    mbs,
                    mbc,
                    rtid,
                )
                L.bar(ROW_BAR, NT)
                if rtid == Int32(0):
                    L.publish_row(mRowStat.iterator.toint(), oq)
                if cutlass.const_expr(self.attn):
                    if rtid == Int32(0):
                        A._wait(sRowBar.iterator.raw_ptr() + Int32(5), Int32(0))
                        sRS[2] = sHI[9]
                        sRS[3] = A.deep_ok(sHI.iterator)
                    L.bar(ROW_BAR, NT)
                    tmem_att = sRS[2]
                    odeep = Int32(0)
                    if cutlass.const_expr(not self.no_deep):
                        if sRS[3] != Int32(0):
                            odeep = Int32(1)
                    if cutlass.const_expr(self.att_skip & 32 == 0):
                        aph = A.att_row(
                            rtid,
                            oq,
                            og0,
                            og1,
                            og2,
                            og3,
                            og4,
                            tmem_att,
                            odeep,
                            q_slot0,
                            kv_slot0,
                            deep_base,
                            deep_q,
                            abar,
                            abar + Int32(4),
                            abar + Int32(12),
                            aph,
                            tma_q,
                            tma_swa,
                            tma_cmp,
                            sink_base + oq * Int32(A.HEADS),
                            out_addr,
                            scale_log2,
                            sMsk,
                            sHI.iterator,
                            ROW_BAR,
                            self.att_skip,
                            not self.no_deep,
                            self.stamps,
                            mStamps,
                            cta * Int32(256) + Int32(128),
                            kv_slot7,
                            Int32(0),
                            abar + Int32(13),
                            self.ahint,
                        )
                oq = oq + Int32(self.num_sms)
            L.bar(ROW_BAR, NT)
        if rtid == Int32(0):
            cute.arch.fence_acq_rel_cta()
            cute.arch.atomic_exch(sHI.iterator + Int32(4), Int32(1), sem="release", scope="cta")
        adone = Int32(0)
        st_m = Float32(A.NEG_FLT_MAX)
        st_l = Float32(0.0)
        st_q = Int32(0)
        st_p = Int32(0)
        st_d = Int32(0)
        dq_n = Int32(0)
        while live != Int32(0):
            if rtid == Int32(0):
                prod = cute.arch.load(sHI.iterator, Int32, sem="acquire", scope="cta")
                fin = cute.arch.load(sHI.iterator + Int32(2), Int32, sem="acquire", scope="cta")
                ea = Int32(0)
                if cutlass.const_expr(self.etile):
                    ea = cute.arch.load(sHI.iterator + Int32(1), Int32, sem="acquire", scope="cta")
                dgo = Int32(0)
                if cutlass.const_expr(self.defer):
                    if dq_n > Int32(0):
                        dgo = A.deep_ok(sHI.iterator)
                        if cute.arch.load(
                            sHI.iterator + Int32(5), Int32, sem="acquire", scope="cta"
                        ) >= Int32(DEFER_LONG):
                            dgo = Int32(1)
                while (
                    (prod <= nh)
                    & (fin == Int32(0))
                    & ((ea == Int32(0)) | (adone != Int32(0)))
                    & (dgo == Int32(0))
                ):
                    L.nanosleep(Int32(128))
                    prod = cute.arch.load(sHI.iterator, Int32, sem="acquire", scope="cta")
                    fin = cute.arch.load(sHI.iterator + Int32(2), Int32, sem="acquire", scope="cta")
                    if cutlass.const_expr(self.etile):
                        ea = cute.arch.load(
                            sHI.iterator + Int32(1), Int32, sem="acquire", scope="cta"
                        )
                    if cutlass.const_expr(self.defer):
                        if dq_n > Int32(0):
                            dgo = A.deep_ok(sHI.iterator)
                            if cute.arch.load(
                                sHI.iterator + Int32(5), Int32, sem="acquire", scope="cta"
                            ) >= Int32(DEFER_LONG):
                                dgo = Int32(1)
                if prod <= nh:
                    cute.arch.fence_acq_rel_cta()
                    prod = cute.arch.load(sHI.iterator, Int32, sem="acquire", scope="cta")
                cute.arch.fence_acq_rel_cta()
                wk = Int32(0)
                if prod > nh:
                    # The leader consults pref before the row-wide work barrier.
                    A._wait(
                        sRowBar.iterator.raw_ptr() + (nh & Int32(1)), (nh >> Int32(1)) & Int32(1)
                    )
                    wk = Int32(1)
                    if cutlass.const_expr(self.defer):
                        if dq_n > Int32(0):
                            if sHQ[(nh & Int32(1)) * Int32(8) + Int32(6)] != Int32(0):
                                wk = Int32(3)
                elif dq_n > Int32(0):
                    wk = Int32(3)
                elif (ea != Int32(0)) & (adone == Int32(0)):
                    wk = Int32(2)
                sRS[0] = wk
            L.bar(ROW_BAR, NT)
            work = sRS[0]
            # Every reader captures work before the leader may publish again.
            L.bar(ROW_BAR, NT)
            if work == Int32(1):
                A._wait(sRowBar.iterator.raw_ptr() + (nh & Int32(1)), (nh >> Int32(1)) & Int32(1))
            if work == Int32(0):
                live = Int32(0)
            elif cutlass.const_expr(self.row_noop):
                L.bar(ROW_BAR, NT)
                prims.mbarrier_arrive(sRowBar.iterator.raw_ptr() + Int32(2) + (nh & Int32(1)))
                nh = nh + Int32(1)
            else:
                hs = (nh & Int32(1)) * Int32(8)
                q = Int32(0)
                n = Int32(0)
                P = Int32(1)
                p = Int32(0)
                cbuf = Int32(0)
                bsel = Int32(0)
                pref = Int32(0)
                # The selector may rewrite these slots while deferred or early
                # attention runs. Only a published segment consumes its metadata.
                if work == Int32(1):
                    q = sHQ[hs + Int32(0)]
                    n = sHQ[hs + Int32(1)]
                    P = sHQ[hs + Int32(2)]
                    p = sHQ[hs + Int32(3)]
                    cbuf = sHQ[hs + Int32(4)]
                    bsel = sHQ[hs + Int32(5)]
                    pref = sHQ[hs + Int32(6)]
                done_row = Int32(0)
                gv0 = Int32(-1)
                gv1 = Int32(-1)
                gv2 = Int32(-1)
                gv3 = Int32(-1)
                gv4 = Int32(-1)
                j0 = Int32(0)
                j1 = Int32(A.NTILES)
                keep = Int32(0)
                dly_t = Int32(-1)
                if cutlass.const_expr(self.rdelay > 0):
                    if work == Int32(1):
                        if nh + Int32(2) < nseg_r:
                            dly_t = sHQ[hs + Int32(7)]
                if work == Int32(2):
                    q = sHQ[14]
                    pref = Int32(1)
                    gv0 = sGv[rtid]
                    done_row = Int32(1)
                    j1 = Int32(1)
                elif work == Int32(3):
                    dq_n = dq_n - Int32(1)
                    q = sDQ[dq_n]
                    gv0 = Int32(mGidx[q, rtid])
                    gv1 = Int32(mGidx[q, rtid + Int32(128)])
                    gv2 = Int32(mGidx[q, rtid + Int32(256)])
                    gv3 = Int32(mGidx[q, rtid + Int32(384)])
                    gv4 = Int32(mGidx[q, rtid + Int32(512)])
                    pref = Int32(0)
                    done_row = Int32(1)
                elif pref != Int32(0):
                    gv0 = sGv[rtid]
                    gv1 = sGv[rtid + Int32(128)]
                    gv2 = sGv[rtid + Int32(256)]
                    gv3 = sGv[rtid + Int32(384)]
                    gv4 = sGv[rtid + Int32(512)]
                    done_row = Int32(1)
                    if adone != Int32(0):
                        j0 = Int32(1)
                else:
                    if cutlass.const_expr(self.ring2):
                        if P == Int32(1):
                            keep = Int32(1)
                    sMb = cute.make_tensor(
                        sMain.iterator + bsel * Int32(MAIN_CAP), cute.make_layout(MAIN_CAP)
                    )
                    done_row, gv0, gv1, gv2, gv3, gv4 = _finalize_seg(
                        rtid,
                        ROW_BAR,
                        cta,
                        start_q,
                        q,
                        n,
                        P,
                        p,
                        sMb,
                        cbuf,
                        sHI.iterator + Int32(12) + bsel,
                        sHand,
                        sAttB,
                        sAttO,
                        sHistR,
                        sCkR,
                        sRres,
                        sRred,
                        sRwc,
                        cute.make_tensor(sRS.iterator + Int32(1), cute.make_layout(1)),
                        mKvLens,
                        mReq,
                        mBtSwa,
                        mBtC,
                        mTopk,
                        mGidx,
                        mRowStat,
                        mSched,
                        ws_seg_base,
                        ws_ctr_base,
                        mbs,
                        mbc,
                        self.num_sms,
                        self.skew_ns,
                        skew_seed,
                        nh * Int32(16) + Int32(3),
                        self.stamps,
                        mStamps,
                        cta * Int32(256) + Int32(148),
                        dly_t,
                        self.rdelay,
                        keep,
                        self.rsort,
                        sRowBar.iterator.raw_ptr() + Int32(2) + bsel,
                    )
                if work == Int32(1):
                    if nh + Int32(2) == nseg_r:
                        # The selector takes sHand/sAttB/sAttO only for its last
                        # segment, after all 128 row threads finish the prior one.
                        prims.mbarrier_arrive(sRowBar.iterator.raw_ptr() + Int32(4))
                if cutlass.const_expr(self.attn):
                    if done_row != Int32(0):
                        if rtid == Int32(0):
                            A._wait(sRowBar.iterator.raw_ptr() + Int32(5), Int32(0))
                            sRS[2] = sHI[9]
                            sRS[3] = A.deep_ok(sHI.iterator)
                            if cutlass.const_expr(self.defer):
                                sRS[5] = cute.arch.load(
                                    sHI.iterator, Int32, sem="acquire", scope="cta"
                                )
                                sRS[6] = cute.arch.load(
                                    sHI.iterator + Int32(5), Int32, sem="acquire", scope="cta"
                                )
                        if cutlass.const_expr(self.skew_ns > 0):
                            L.skew_sleep(
                                skew_seed + rtid, nh * Int32(16) + Int32(9), Int32(self.skew_ns)
                            )
                        L.bar(ROW_BAR, NT)
                        tmem_att = sRS[2]
                        dfr = Int32(0)
                        if cutlass.const_expr(self.defer):
                            if (
                                (work == Int32(1))
                                & (pref == Int32(0))
                                & (sRS[3] == Int32(0))
                                & (dq_n < Int32(self.dq_cap))
                            ):
                                if sRS[5] > nh + Int32(1):
                                    dfr = Int32(1)
                                elif sRS[6] < Int32(DEFER_LONG):
                                    dfr = Int32(1)
                        if dfr != Int32(0):
                            if rtid == Int32(0):
                                sDQ[dq_n] = q
                            dq_n = dq_n + Int32(1)
                        else:
                            deep0 = Int32(0)
                            if cutlass.const_expr(not self.no_deep):
                                if sRS[3] != Int32(0):
                                    deep0 = Int32(1)
                            ext_on = Int32(0)
                            if cutlass.const_expr(self.gath):
                                if pref != Int32(0):
                                    ext_on = Int32(1)
                                    deep0 = Int32(1)
                            if cutlass.const_expr(self.stamps):
                                if rtid == Int32(0):
                                    if work == Int32(2):
                                        mStamps[cta * 256 + 160] = cutlass.Int64(
                                            cute.arch.globaltimer()
                                        )
                                    elif (nh < Int32(8)) & (work == Int32(1)):
                                        mStamps[cta * 256 + 24 + nh] = cutlass.Int64(
                                            cute.arch.globaltimer()
                                        )
                            kv_b = None
                            two_a = None
                            if cutlass.const_expr(self.ring2):
                                kv_b = main_b + bsel * Int32(MAIN_CAP * 8)
                                two_a = keep
                                cute.arch.fence_view_async_shared()
                            if cutlass.const_expr(self.att_skip & 32 == 0):
                                if cutlass.const_expr(self.etile):
                                    i_m = Float32(A.NEG_FLT_MAX)
                                    i_l = Float32(0.0)
                                    i_q = Int32(0)
                                    i_p = Int32(0)
                                    i_d = deep0
                                    if j0 != Int32(0):
                                        i_m = st_m
                                        i_l = st_l
                                        i_q = st_q
                                        i_p = st_p
                                        i_d = st_d
                                    aph, st_m, st_l, st_q, st_p, st_d = A.att_row(
                                        rtid,
                                        q,
                                        gv0,
                                        gv1,
                                        gv2,
                                        gv3,
                                        gv4,
                                        tmem_att,
                                        deep0,
                                        q_slot0,
                                        kv_slot0,
                                        deep_base,
                                        deep_q,
                                        abar,
                                        abar + Int32(4),
                                        abar + Int32(12),
                                        aph,
                                        tma_q,
                                        tma_swa,
                                        tma_cmp,
                                        sink_base + q * Int32(A.HEADS),
                                        out_addr,
                                        scale_log2,
                                        sMsk,
                                        sHI.iterator,
                                        ROW_BAR,
                                        self.att_skip,
                                        not self.no_deep,
                                        self.stamps,
                                        mStamps,
                                        cta * Int32(256) + Int32(128),
                                        kv_slot7,
                                        ext_on,
                                        abar + Int32(13),
                                        self.ahint,
                                        True,
                                        j0,
                                        j1,
                                        (i_m, i_l, i_q, i_p, i_d),
                                        self.apace,
                                        kv_b,
                                        two_a,
                                    )
                                else:
                                    pl = None
                                    if cutlass.const_expr(self.pace_adapt):
                                        pl = nh + Int32(1)
                                        if work == Int32(3):
                                            pl = nh
                                    aph = A.att_row(
                                        rtid,
                                        q,
                                        gv0,
                                        gv1,
                                        gv2,
                                        gv3,
                                        gv4,
                                        tmem_att,
                                        deep0,
                                        q_slot0,
                                        kv_slot0,
                                        deep_base,
                                        deep_q,
                                        abar,
                                        abar + Int32(4),
                                        abar + Int32(12),
                                        aph,
                                        tma_q,
                                        tma_swa,
                                        tma_cmp,
                                        sink_base + q * Int32(A.HEADS),
                                        out_addr,
                                        scale_log2,
                                        sMsk,
                                        sHI.iterator,
                                        ROW_BAR,
                                        self.att_skip,
                                        not self.no_deep,
                                        self.stamps,
                                        mStamps,
                                        cta * Int32(256) + Int32(128),
                                        kv_slot7,
                                        ext_on,
                                        abar + Int32(13),
                                        self.ahint,
                                        False,
                                        None,
                                        None,
                                        None,
                                        self.apace,
                                        kv_b,
                                        two_a,
                                        pl,
                                    )
                    if cutlass.const_expr(self.stamps):
                        if work == Int32(1):
                            if nh < Int32(8):
                                mStamps[cta * 256 + 120 + nh] = cutlass.Int64(
                                    cute.arch.globaltimer()
                                )
                        else:
                            mStamps[cta * 256 + 161] = cutlass.Int64(cute.arch.globaltimer())
                if cutlass.const_expr(self.ring2):
                    if keep != Int32(0):
                        if rtid == Int32(0):
                            cute.arch.fence_view_async_shared()
                        prims.mbarrier_arrive(sRowBar.iterator.raw_ptr() + Int32(2) + bsel)
                if work == Int32(2):
                    adone = Int32(1)
                elif work == Int32(3):
                    if dq_n == Int32(0):
                        if rtid == Int32(0):
                            cute.arch.fence_acq_rel_cta()
                            cute.arch.atomic_exch(
                                sHI.iterator + Int32(15), nh, sem="release", scope="cta"
                            )
                else:
                    if dq_n == Int32(0):
                        if rtid == Int32(0):
                            cute.arch.fence_acq_rel_cta()
                            cute.arch.atomic_exch(
                                sHI.iterator + Int32(15), nh + Int32(1), sem="release", scope="cta"
                            )
                    nh = nh + Int32(1)
        if cutlass.const_expr(self.stamps):
            if rtid == Int32(0):
                mStamps[cta * 256 + 20] = cutlass.Int64(cute.arch.globaltimer())
                mStamps[cta * 256 + 14] = L.clk64()


_COMPILED = {}


def compile_indexer(mk: IndexerKernel):
    key = tuple(
        ((k, v) for k, v in sorted(vars(mk).items()) if isinstance(v, (int, float, bool, str)))
    )
    if key in _COMPILED:
        return _COMPILED[key]
    H, D = (mk.num_heads, mk.head_dim)
    block_bytes = mk.phys_block_kv * (D + 4)
    rt = cute.runtime
    sb = cute.sym_int()
    kv_fake = rt.make_fake_tensor(
        cutlass.Uint8, (cute.sym_int(), block_bytes), stride=(cute.sym_int64(), 1)
    )
    q_fake = rt.make_fake_compact_tensor(
        cutlass.Uint8, (H, D, cute.sym_int()), stride_order=(1, 0, 2)
    )
    w_fake = rt.make_fake_compact_tensor(cutlass.Float32, (H, cute.sym_int()), stride_order=(0, 1))
    lg_fake = rt.make_fake_tensor(
        cutlass.Float32, (cute.sym_int(), cute.sym_int()), stride=(cute.sym_int64(), 1)
    )
    bt_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb, cute.sym_int()), stride_order=(1, 0))
    cl_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb,), stride_order=(0,))
    sm_fake = rt.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(), 2), stride_order=(1, 0))
    kvl_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb,), stride_order=(0,))
    req_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb,), stride_order=(0,))
    bts_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb, cute.sym_int()), stride_order=(1, 0))
    btc_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb, cute.sym_int()), stride_order=(1, 0))
    tk_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb, K_TOP), stride_order=(1, 0))
    gi_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb, 640), stride_order=(1, 0))
    wss_fake = rt.make_fake_compact_tensor(cutlass.Int64, (cute.sym_int(),), stride_order=(0,))
    wsc_fake = rt.make_fake_compact_tensor(cutlass.Int64, (cute.sym_int(),), stride_order=(0,))
    rs_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb, 16), stride_order=(1, 0))
    st_fake = rt.make_fake_compact_tensor(cutlass.Int64, (cute.sym_int(),), stride_order=(0,))
    qa_fake = rt.make_fake_compact_tensor(
        cutlass.Uint8, (cute.sym_int64(), 64, 512), stride_order=(2, 1, 0), assumed_align=16
    )
    sp_fake = rt.make_fake_compact_tensor(
        cutlass.Uint8, (cute.sym_int64(), 512), stride_order=(1, 0), assumed_align=16
    )
    cp_fake = rt.make_fake_compact_tensor(
        cutlass.Uint8, (cute.sym_int64(), 512), stride_order=(1, 0), assumed_align=16
    )
    sk_fake = rt.make_fake_compact_tensor(
        cutlass.Float32, (cute.sym_int(), 64), stride_order=(1, 0), assumed_align=16
    )
    ou_fake = rt.make_fake_compact_tensor(
        cutlass.BFloat16, (cute.sym_int(), 64, 512), stride_order=(2, 1, 0), assumed_align=16
    )
    stw_fake = rt.make_fake_compact_tensor(cutlass.Int32, (4,), stride_order=(0,))
    stream = rt.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        mk,
        kv_fake,
        q_fake,
        w_fake,
        lg_fake,
        bt_fake,
        cl_fake,
        sm_fake,
        kvl_fake,
        req_fake,
        bts_fake,
        btc_fake,
        tk_fake,
        gi_fake,
        wss_fake,
        wsc_fake,
        rs_fake,
        st_fake,
        qa_fake,
        sp_fake,
        cp_fake,
        sk_fake,
        ou_fake,
        stw_fake,
        cutlass.Float32(1.0),
        cutlass.Int32(1),
        cutlass.Int32(1),
        stream,
        options="--enable-tvm-ffi --generate-line-info",
    )
    _COMPILED[key] = compiled
    return compiled
