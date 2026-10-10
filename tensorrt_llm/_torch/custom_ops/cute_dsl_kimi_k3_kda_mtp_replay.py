# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# One-launch Conv4 + KDA MTP replay (CuTe DSL, SM100 family).
#
# One CTA owns one (decode row, head); thread `tid` is simultaneously the key
# index (q/k/G) and the value index (v/U/output), so every 128-wide vector of
# the problem maps exactly one element per thread.
#
# Replay never materialises the 128x128 recurrent state.  With
#   recurrent_t = checkpoint*exp(cumG_t) + sum_i u_i k_i exp(cumG_t - G_i)
# every token's state read collapses into two products against ONE shared
# operand  X_t * exp(cumG_t)   (X_t = [k_t; q_t]):
#   * checkpoint @ operand^T                     -> 128 x 128 x 2T
#   * U^T @ ((K_i exp(-G_i)) @ operand^T)        -> (W+T) x 128 x 2T  then 128 x W x 2T
# plus a small intra-window triangular correction carried forward in registers.

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import warp as cwarp
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils import SmemAllocator

KD = 128
LOG2E = 1.4426950408889634
QSCALE = 1.0 / math.sqrt(128.0)
LOWER_BOUND = -5.0

_CACHE = {}


@dsl_user_op
def _cp_async_shared_global(dst, src, size, cache_mode, *, loc=None, ip=None):
    """Issue the 16-byte cp.async form used by the generated kernel."""
    if size != 16 or cache_mode not in ("ca", "cg"):
        raise ValueError("KDA cp.async supports only 16-byte ca/cg copies")
    cute.arch.inline_ptx(
        f"cp.async.{cache_mode}.shared.global [{{$r0}}], [{{$r1}}], 16;",
        read_only_args=[dst, src],
        loc=loc,
        ip=ip,
    )


@cute.kernel
def _kda_kernel(
    pX,
    pCW,
    pCS,
    pRG,
    pBT,
    pST,
    pIdx,
    pDum,
    pHK,
    pHU,
    pHG,
    pHL,
    pAl,
    pDt,
    pOut,
    pCd,
    T: cutlass.Constexpr,
    H: cutlass.Constexpr,
    W: cutlass.Constexpr,
    CAP: cutlass.Constexpr,
    S: cutlass.Constexpr,
    SD: cutlass.Constexpr,
    XB: cutlass.Constexpr,
    XT: cutlass.Constexpr,
    CSS: cutlass.Constexpr,
    CSC: cutlass.Constexpr,
    SSS: cutlass.Constexpr,
    BTB: cutlass.Constexpr,
    BTT: cutlass.Constexpr,
    I64: cutlass.Constexpr,
):
    f32 = cutlass.Float32
    bf16 = cutlass.BFloat16

    C = 3 * H * KD
    NC = 2 * T
    NP = ((NC + 7) // 8) * 8
    NQ = NP // 4
    NACC = NP + 4
    APAD = KD + 8  # bf16 tile pitch (ldmatrix conflict free)
    CHUNK = 32
    CPAD = CHUNK + 4  # fp32 chunk pitch (LDS.128 conflict free)
    NCH = KD // CHUNK
    NBUF = 2  # fp32 checkpoint staging depth
    RROWS = 64
    NR = W + T
    NWR = (NR + 15) // 16
    HW = W // 16
    KB = KD

    tid, _, _ = cute.arch.thread_idx()
    h, b, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    widx = cute.arch.warp_idx()

    # ---------------- shared memory ----------------
    # One staging buffer per CTA, reused in sequence: checkpoint tile ->
    # accumulators / per-warp partials -> R operand -> GEMM2 accumulators.
    # The R operand is published only after the checkpoint product is finished,
    # which is what lets the whole kernel live in a single tile-sized arena.
    alloc = SmemAllocator()
    if cutlass.const_expr(SD == 0):
        pA = alloc.allocate_array(bf16, KD * APAD, byte_alignment=16)
        pBase = cute.recast_ptr(pA, dtype=f32)
        pAccM = pBase
        pOff = KD * NACC
    else:
        pF = alloc.allocate_array(f32, NBUF * KD * CPAD, byte_alignment=16)
        pOpT = alloc.allocate_array(f32, KD * NACC, byte_alignment=16)
        pBase = pF
        pOff = 0
    pR = cute.recast_ptr(pBase + pOff, dtype=bf16)
    pAccR = pBase + pOff + (RROWS * APAD) // 2
    pXb = alloc.allocate_array(bf16, NP * APAD, byte_alignment=16)
    pRed = alloc.allocate_array(f32, 4 * NP, byte_alignment=16)

    sR = cute.make_tensor(pR, cute.make_layout((RROWS, KD), stride=(APAD, 1)))
    sXb = cute.make_tensor(pXb, cute.make_layout((NP, KD), stride=(APAD, 1)))
    sRed = cute.make_tensor(pRed, cute.make_layout((4, NP), stride=(NP, 1)))
    sAccR = cute.make_tensor(pAccR, cute.make_layout((RROWS, NP), stride=(NACC, 1)))
    sAccR4 = cute.make_tensor(pAccR, cute.make_layout((RROWS, NQ, 4), stride=(NACC, 4, 1)))
    if cutlass.const_expr(SD == 0):
        sAccM = cute.make_tensor(pAccM, cute.make_layout((KD, NP), stride=(NACC, 1)))
        sAccM4 = cute.make_tensor(pAccM, cute.make_layout((KD, NQ, 4), stride=(NACC, 4, 1)))
    else:
        sOpT = cute.make_tensor(pOpT, cute.make_layout((KD, NP), stride=(NACC, 1)))
        sOpT4 = cute.make_tensor(pOpT, cute.make_layout((KD, NQ, 4), stride=(NACC, 4, 1)))

    # ---------------- slot indirection ----------------
    slot = cute.make_tensor(pIdx + b, cute.make_layout((1,), stride=(1,)))[0]
    dummy = cute.make_tensor(pDum + b, cute.make_layout((1,), stride=(1,)))[0]

    # Everything that does not need the slot is fetched here, ahead of the
    # dummy-row exit.  The exit is inline asm and therefore a scheduling
    # barrier: anything written below it cannot issue until `is_dummy` has
    # come back from memory, so leaving these loads below would bolt a full
    # HBM round trip onto the front of every live row.
    cq = h * KD + tid
    ck = H * KD + cq
    cv = 2 * H * KD + cq

    gX = cute.make_tensor(pX + b * XB, cute.make_layout((T, C), stride=(XT, 1)))
    gCW = cute.make_tensor(pCW, cute.make_layout((C, 4), stride=(4, 1)))
    gRG = cute.make_tensor(
        pRG + b * (T * H * KD) + h * KD, cute.make_layout((T, KD), stride=(H * KD, 1))
    )
    gBT = cute.make_tensor(pBT + b * BTB + h, cute.make_layout((T,), stride=(BTT,)))
    gAl = cute.make_tensor(pAl, cute.make_layout((H,), stride=(1,)))
    gDt = cute.make_tensor(pDt, cute.make_layout((H, KD), stride=(KD, 1)))

    wq = cute.make_rmem_tensor((4,), bf16)
    wk = cute.make_rmem_tensor((4,), bf16)
    wv = cute.make_rmem_tensor((4,), bf16)
    cute.autovec_copy(gCW[(cq, None)], wq)
    cute.autovec_copy(gCW[(ck, None)], wk)
    cute.autovec_copy(gCW[(cv, None)], wv)

    xq = cute.make_rmem_tensor((T,), bf16)
    xk = cute.make_rmem_tensor((T,), bf16)
    xv = cute.make_rmem_tensor((T,), bf16)
    for t in cutlass.range_constexpr(T):
        xq[t] = gX[t, cq]
        xk[t] = gX[t, ck]
        xv[t] = gX[t, cv]

    rgv = cute.make_rmem_tensor((T,), bf16)
    btv = cute.make_rmem_tensor((T,), bf16)
    for t in cutlass.range_constexpr(T):
        rgv[t] = gRG[t, tid]
        btv[t] = gBT[t]
    dtb = gDt[h, tid]
    alraw = gAl[h]

    # A rejected row produces no checked output and is forbidden from mutating
    # any cache, so the whole CTA leaves before it touches the checkpoint,
    # history or candidate traffic.  `dummy` is CTA-uniform, so the predicated
    # PTX `exit` retires every warp together and keeps the remainder of the
    # kernel out of a dynamic branch region.
    cute.arch.inline_ptx("exit;", predicate=(dummy.to(cutlass.Int32) != 0))

    if cutlass.const_expr(I64):
        ckp = pST + (cutlass.Int64(slot) * cutlass.Int64(SSS) + cutlass.Int64(h * KD * KD))
    else:
        ckp = pST + (slot * SSS + h * KD * KD)

    # ---------------- slot-dependent global views ----------------
    gCS = cute.make_tensor(pCS + slot * CSS, cute.make_layout((C, 3), stride=(CSC, 1)))
    gOut = cute.make_tensor(
        pOut + b * (T * H * KD) + h * KD, cute.make_layout((T, KD), stride=(H * KD, 1))
    )
    hbase = (slot * H + h) * CAP
    gHK = cute.make_tensor(pHK + hbase * KD, cute.make_layout((CAP, KD), stride=(KD, 1)))
    gHU = cute.make_tensor(pHU + hbase * KD, cute.make_layout((CAP, KD), stride=(KD, 1)))
    gHG = cute.make_tensor(pHG + hbase * KD, cute.make_layout((CAP, KD), stride=(KD, 1)))
    gHL = cute.make_tensor(pHL, cute.make_layout((S,), stride=(1,)))

    L = gHL[slot]

    # ---------------- stage the checkpoint tile ----------------
    # FP32 checkpoints are twice the bytes of BF16 ones, so they stream through
    # an NBUF-deep cp.async pipeline: every buffer is filled before the first
    # consumer wait, which leaves exactly one memory round trip exposed instead
    # of one per chunk.
    if cutlass.const_expr(SD == 0):
        for itc in cutlass.range_constexpr(KD * KD // (8 * 128)):
            eid = itc * 128 + tid
            vrw = eid // 16
            ocw = eid % 16
            _cp_async_shared_global(
                (pA + (vrw * APAD + ocw * 8)).align(16),
                (ckp + (vrw * KD + ocw * 8)).align(16),
                16,
                "cg",
            )
        cute.arch.cp_async_commit_group()
    else:
        for cc in cutlass.range_constexpr(NBUF if NBUF < NCH else NCH):
            for itc in cutlass.range_constexpr(KD * CHUNK // (4 * 128)):
                eid = itc * 128 + tid
                vrw = eid // (CHUNK // 4)
                qwd = eid % (CHUNK // 4)
                _cp_async_shared_global(
                    (pF + (cc * KD * CPAD + vrw * CPAD + qwd * 4)).align(16),
                    (ckp + (vrw * KD + cc * CHUNK + qwd * 4)).align(16),
                    16,
                    "cg",
                )
            cute.arch.cp_async_commit_group()

    # ---------------- depthwise conv4 + SiLU ----------------
    sq = cute.make_rmem_tensor((3,), bf16)
    sk = cute.make_rmem_tensor((3,), bf16)
    sv = cute.make_rmem_tensor((3,), bf16)
    for j in cutlass.range_constexpr(3):
        sq[j] = gCS[cq, j]
        sk[j] = gCS[ck, j]
        sv[j] = gCS[cv, j]

    # candidate_x is a verbatim copy of the consumed raw_x slice, so emit it
    # on a re-mapped thread->channel assignment with 128-bit accesses: 3 wide
    # loads + 3 wide stores replace 3*T narrow loads + 3*T narrow stores, and
    # the wide loads land on lines the scalar conv loads above already pulled.
    gX8 = cute.make_tensor(pX + b * XB, cute.make_layout((T, C // 8, 8), stride=(XT, 8, 1)))
    gCd8 = cute.make_tensor(pCd + b * (T * C), cute.make_layout((T, C // 8, 8), stride=(C, 8, 1)))
    cbuf = cute.make_rmem_tensor((8,), bf16)
    NGRP = 3 * T * (KD // 8)
    for itv in cutlass.range_constexpr((NGRP + 127) // 128):
        gidx = itv * 128 + tid
        if cutlass.const_expr((itv + 1) * 128 <= NGRP):
            cute.autovec_copy(
                gX8[
                    (gidx % (16 * T) // 16, gidx // (16 * T) * (H * 16) + h * 16 + gidx % 16, None)
                ],
                cbuf,
            )
            cute.autovec_copy(
                cbuf,
                gCd8[
                    (gidx % (16 * T) // 16, gidx // (16 * T) * (H * 16) + h * 16 + gidx % 16, None)
                ],
            )
        else:
            if gidx < NGRP:
                cute.autovec_copy(
                    gX8[
                        (
                            gidx % (16 * T) // 16,
                            gidx // (16 * T) * (H * 16) + h * 16 + gidx % 16,
                            None,
                        )
                    ],
                    cbuf,
                )
                cute.autovec_copy(
                    cbuf,
                    gCd8[
                        (
                            gidx % (16 * T) // 16,
                            gidx // (16 * T) * (H * 16) + h * 16 + gidx % 16,
                            None,
                        )
                    ],
                )

    qp = cute.make_rmem_tensor((T,), f32)
    kp = cute.make_rmem_tensor((T,), f32)
    vp = cute.make_rmem_tensor((T,), f32)
    for t in cutlass.range_constexpr(T):
        aq = f32(0.0)
        ak = f32(0.0)
        av = f32(0.0)
        for tap in cutlass.range_constexpr(4):
            j = tap + t
            if cutlass.const_expr(j < 3):
                aq = aq + sq[j].to(f32) * wq[tap].to(f32)
                ak = ak + sk[j].to(f32) * wk[tap].to(f32)
                av = av + sv[j].to(f32) * wv[tap].to(f32)
            else:
                aq = aq + xq[j - 3].to(f32) * wq[tap].to(f32)
                ak = ak + xk[j - 3].to(f32) * wk[tap].to(f32)
                av = av + xv[j - 3].to(f32) * wv[tap].to(f32)
        eq = cute.math.exp2(aq * f32(-LOG2E), approx=True, ftz=True)
        ek = cute.math.exp2(ak * f32(-LOG2E), approx=True, ftz=True)
        ev = cute.math.exp2(av * f32(-LOG2E), approx=True, ftz=True)
        qp[t] = (aq * cute.math.rcp(f32(1.0) + eq, approx=True, ftz=True)).to(bf16).to(f32)
        kp[t] = (ak * cute.math.rcp(f32(1.0) + ek, approx=True, ftz=True)).to(bf16).to(f32)
        vp[t] = (av * cute.math.rcp(f32(1.0) + ev, approx=True, ftz=True)).to(bf16).to(f32)

    # ---------------- q/k L2 norms ----------------
    rks = cute.make_rmem_tensor((T,), f32)
    rqs = cute.make_rmem_tensor((T,), f32)
    for t in cutlass.range_constexpr(T):
        rks[t] = cute.arch.warp_reduction_sum(kp[t] * kp[t])
        rqs[t] = cute.arch.warp_reduction_sum(qp[t] * qp[t])
    if lane == 0:
        for t in cutlass.range_constexpr(T):
            sRed[widx, t] = rks[t]
            sRed[widx, T + t] = rqs[t]

    # ---------------- gates / cumulative G ----------------
    decay_rate = cute.math.exp2(alraw * f32(LOG2E), approx=True, ftz=True)
    gate = cute.make_rmem_tensor((T,), f32)
    for t in cutlass.range_constexpr(T):
        gi = rgv[t].to(f32) + dtb
        ez = cute.math.exp2(decay_rate * gi * f32(-LOG2E), approx=True, ftz=True)
        gate[t] = f32(LOWER_BOUND) * cute.math.rcp(f32(1.0) + ez, approx=True, ftz=True)

    bet = cute.make_rmem_tensor((T,), f32)
    for t in cutlass.range_constexpr(T):
        eb = cute.math.exp2(btv[t].to(f32) * f32(-LOG2E), approx=True, ftz=True)
        bet[t] = cute.math.rcp(f32(1.0) + eb, approx=True, ftz=True)
    # ---------------- cached history rows (kept in registers) ----------------
    # Every row index is clamped rather than branched on: masked rows re-read
    # row 0 (an L1 hit, no extra HBM traffic) and are selected away afterwards,
    # which lets all of these loads be in flight at once instead of paying one
    # full memory round trip per window slot.
    prevG = f32(0.0)
    if L > 0:
        prevG = gHG[L - 1, tid]

    cum = cute.make_rmem_tensor((T,), f32)
    runG = prevG
    for t in cutlass.range_constexpr(T):
        runG = runG + gate[t]
        cum[t] = runG

    hk = cute.make_rmem_tensor((W,), bf16)
    hu = cute.make_rmem_tensor((W,), bf16)
    for i in cutlass.range_constexpr(W):
        rv = bf16(0.0)
        uv = bf16(0.0)
        if i < L:
            dcy = cute.math.exp2(gHG[i, tid] * f32(-LOG2E), approx=True, ftz=True)
            rv = (gHK[i, tid].to(f32) * dcy).to(bf16)
            uv = gHU[i, tid]
        hk[i] = rv
        hu[i] = uv

    cute.arch.barrier()

    # ---------------- normalise q/k, build the shared operand ----------------
    kbv = cute.make_rmem_tensor((T,), bf16)
    rkh = cute.make_rmem_tensor((T,), bf16)
    for t in cutlass.range_constexpr(T):
        tk = sRed[0, t] + sRed[1, t] + sRed[2, t] + sRed[3, t]
        tq = sRed[0, T + t] + sRed[1, T + t] + sRed[2, T + t] + sRed[3, T + t]
        ik = cute.math.rcp(
            cute.math.sqrt(tk + f32(1.0e-6), approx=True, ftz=True), approx=True, ftz=True
        )
        iq = cute.math.rcp(
            cute.math.sqrt(tq + f32(1.0e-6), approx=True, ftz=True), approx=True, ftz=True
        )
        kb = (kp[t] * ik).to(bf16)
        qb = (qp[t] * (f32(QSCALE) * iq)).to(bf16)
        kbv[t] = kb
        ec = cute.math.exp2(cum[t] * f32(LOG2E), approx=True, ftz=True)
        ei = cute.math.exp2(cum[t] * f32(-LOG2E), approx=True, ftz=True)
        opk = kb.to(f32) * ec
        opq = qb.to(f32) * ec
        if cutlass.const_expr(SD != 0):
            sOpT[tid, t] = opk
            sOpT[tid, T + t] = opq
        sXb[t, tid] = opk.to(bf16)
        sXb[T + t, tid] = opq.to(bf16)
        rkh[t] = (kb.to(f32) * ei).to(bf16)
    for n in cutlass.range_constexpr(NP - NC):
        sXb[NC + n, tid] = bf16(0.0)
        if cutlass.const_expr(SD != 0):
            sOpT[tid, NC + n] = f32(0.0)

    # Retire the K / cumulative-G appends now: addresses are disjoint from the
    # cached rows this CTA read, and it frees `cum` before the MMA section.
    for t in cutlass.range_constexpr(T):
        gHK[L + t, tid] = kbv[t]
        gHG[L + t, tid] = cum[t]

    if cutlass.const_expr(SD == 0):
        cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()

    # ---------------- MMA plumbing ----------------
    mma_op = cwarp.MmaF16BF16Op(bf16, f32, (16, 8, 16))
    tiled_mma = cute.make_tiled_mma(
        mma_op, cute.make_layout((4, 1, 1)), permutation_mnk=(64, 8, 16)
    )
    thr_mma = tiled_mma.get_slice(tid)
    ld_a = cute.make_copy_atom(cwarp.LdMatrix8x8x16bOp(False, 4), bf16)
    ld_b = cute.make_copy_atom(cwarp.LdMatrix8x8x16bOp(False, 2), bf16)
    tld_a = cute.make_tiled_copy_A(ld_a, tiled_mma)
    tld_b = cute.make_tiled_copy_B(ld_b, tiled_mma)
    thr_a = tld_a.get_slice(tid)
    thr_b = tld_b.get_slice(tid)

    # ---------------- GEMM1: checkpoint projection ----------------
    pk = cute.make_rmem_tensor((NC,), f32)
    e4 = cute.make_rmem_tensor((4,), f32)

    if cutlass.const_expr(SD == 0):
        tCsC = thr_mma.partition_C(sAccM)
        acc = cute.make_rmem_tensor(tCsC.shape, f32)
        acc.fill(0.0)
        if cutlass.const_expr(True):
            for kbi in cutlass.range_constexpr(KD // KB):
                sXa = cute.make_tensor(pXb + kbi * KB, cute.make_layout((NP, KB), stride=(APAD, 1)))
                sAa = cute.make_tensor(pA + kbi * KB, cute.make_layout((KD, KB), stride=(APAD, 1)))
                frBa = tiled_mma.make_fragment_B(thr_mma.partition_B(sXa))
                frAa = tiled_mma.make_fragment_A(thr_mma.partition_A(sAa))
                cute.copy(tld_b, thr_b.partition_S(sXa), thr_b.retile(frBa))
                cute.copy(tld_a, thr_a.partition_S(sAa), thr_a.retile(frAa))
                cute.gemm(tiled_mma, acc, frAa, frBa, acc)
        cute.arch.barrier()
        cute.autovec_copy(acc, tCsC)
        cute.arch.barrier()
        for cn in cutlass.range_constexpr(NQ):
            cute.autovec_copy(sAccM4[(tid, cn, None)], e4)
            for e in cutlass.range_constexpr(4):
                n = cn * 4 + e
                if cutlass.const_expr(n < NC):
                    pk[n] = e4[e]
    else:
        acc32 = cute.make_rmem_tensor((NC,), f32)
        for n in cutlass.range_constexpr(NC):
            acc32[n] = f32(0.0)
        c4 = cute.make_rmem_tensor((4,), f32)
        o4 = cute.make_rmem_tensor((4,), f32)
        NFIRST = NBUF if NBUF < NCH else NCH
        for c in cutlass.range_constexpr(NCH):
            nissued = NFIRST + (c if c < NCH - NFIRST else NCH - NFIRST)
            cute.arch.cp_async_wait_group(nissued - c - 1)
            cute.arch.barrier()
            sCF = cute.make_tensor(
                pF + (c % NBUF) * KD * CPAD,
                cute.make_layout((KD, CPAD // 4, 4), stride=(CPAD, 4, 1)),
            )
            for kk in cutlass.range_constexpr(CHUNK // 4):
                cute.autovec_copy(sCF[(tid, kk, None)], c4)
                for jj in cutlass.range_constexpr(4):
                    for cm in cutlass.range_constexpr(NQ):
                        cute.autovec_copy(sOpT4[(c * CHUNK + kk * 4 + jj, cm, None)], o4)
                        for em in cutlass.range_constexpr(2):
                            nm = cm * 4 + em * 2
                            if cutlass.const_expr(nm + 1 < NC):
                                acc32[nm], acc32[nm + 1] = cute.arch.fma_packed_f32x2(
                                    (o4[em * 2], o4[em * 2 + 1]),
                                    (c4[jj], c4[jj]),
                                    (acc32[nm], acc32[nm + 1]),
                                )
            if cutlass.const_expr(c + NBUF < NCH):
                cute.arch.barrier()
                for itf in cutlass.range_constexpr(KD * CHUNK // (4 * 128)):
                    eif = itf * 128 + tid
                    vrf = eif // (CHUNK // 4)
                    qwf = eif % (CHUNK // 4)
                    _cp_async_shared_global(
                        (pF + ((c % NBUF) * KD * CPAD + vrf * CPAD + qwf * 4)).align(16),
                        (ckp + (vrf * KD + (c + NBUF) * CHUNK + qwf * 4)).align(16),
                        16,
                        "cg",
                    )
                cute.arch.cp_async_commit_group()
        for n in cutlass.range_constexpr(NC):
            pk[n] = acc32[n]
        cute.arch.barrier()

    # ---------------- publish the R operand ----------------
    # No extra rendezvous: the arena was already drained after the checkpoint
    # product, and the R region is disjoint from the accumulators just read.
    for i in cutlass.range_constexpr(W):
        sR[i, tid] = hk[i]
    for t in cutlass.range_constexpr(T):
        sR[W + t, tid] = rkh[t]
    cute.arch.barrier()

    # ---------------- GEMM2: history / replay coefficients ----------------
    tCsCR = thr_mma.partition_C(sAccR)
    accR = cute.make_rmem_tensor(tCsCR.shape, f32)
    accR.fill(0.0)
    if (widx < NWR) and (L > 0 or widx >= HW):
        for kbj in cutlass.range_constexpr(KD // KB):
            sXr = cute.make_tensor(pXb + kbj * KB, cute.make_layout((NP, KB), stride=(APAD, 1)))
            sRr = cute.make_tensor(pR + kbj * KB, cute.make_layout((RROWS, KB), stride=(APAD, 1)))
            frBr = tiled_mma.make_fragment_B(thr_mma.partition_B(sXr))
            frAr = tiled_mma.make_fragment_A(thr_mma.partition_A(sRr))
            cute.copy(tld_b, thr_b.partition_S(sXr), thr_b.retile(frBr))
            cute.copy(tld_a, thr_a.partition_S(sRr), thr_a.retile(frAr))
            cute.gemm(tiled_mma, accR, frAr, frBr, accR)
    cute.arch.barrier()
    if (widx < NWR) and (L > 0 or widx >= HW):
        cute.autovec_copy(accR, tCsCR)
    cute.arch.barrier()

    # ---------------- epilogue ----------------
    for i in cutlass.range_constexpr(W):
        if i < L:
            uf = hu[i].to(f32)
            for cn in cutlass.range_constexpr(NQ):
                cute.autovec_copy(sAccR4[(i, cn, None)], e4)
                for e in cutlass.range_constexpr(2):
                    n = cn * 4 + e * 2
                    if cutlass.const_expr(n + 1 < NC):
                        pk[n], pk[n + 1] = cute.arch.fma_packed_f32x2(
                            (e4[e * 2], e4[e * 2 + 1]), (uf, uf), (pk[n], pk[n + 1])
                        )

    # Forward-propagating recurrence: once token t's update is known it is
    # pushed into every later column of `pk`, so each replay coefficient row is
    # read once with 128-bit loads instead of O(T^2) scalar broadcasts.
    uu = cute.make_rmem_tensor((T,), f32)
    for t in cutlass.range_constexpr(T):
        ut = bet[t] * (vp[t] - pk[t])
        uu[t] = ut
        ov = pk[T + t]
        for cn in cutlass.range_constexpr(NQ):
            cute.autovec_copy(sAccR4[(W + t, cn, None)], e4)
            for e in cutlass.range_constexpr(4):
                n = cn * 4 + e
                if cutlass.const_expr(n == T + t):
                    ov = ov + ut * e4[e]
                elif cutlass.const_expr(n < NC and ((n < T and n > t) or (n >= T and n - T > t))):
                    pk[n] = pk[n] + ut * e4[e]
        gOut[t, tid] = ov.to(bf16)

    for t in cutlass.range_constexpr(T):
        gHU[L + t, tid] = uu[t].to(bf16)


@cute.jit
def _launch(
    pX,
    pCW,
    pCS,
    pRG,
    pBT,
    pST,
    pIdx,
    pDum,
    pHK,
    pHU,
    pHG,
    pHL,
    pAl,
    pDt,
    pOut,
    pCd,
    B: cutlass.Int32,
    T: cutlass.Constexpr,
    H: cutlass.Constexpr,
    W: cutlass.Constexpr,
    CAP: cutlass.Constexpr,
    S: cutlass.Constexpr,
    SD: cutlass.Constexpr,
    XB: cutlass.Constexpr,
    XT: cutlass.Constexpr,
    CSS: cutlass.Constexpr,
    CSC: cutlass.Constexpr,
    SSS: cutlass.Constexpr,
    BTB: cutlass.Constexpr,
    BTT: cutlass.Constexpr,
    I64: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    # The wide-T FP32 arena fits 4 CTAs/SM but its register demand collapses
    # that to 2, starving the memory pipeline; an explicit floor buys the
    # occupancy back.  Every other configuration is already at its shared-memory
    # limit, and forcing a floor there only perturbs the L1/SMEM split (measured
    # +19..37% on FP32 T=3 and BF16 T=3), so those launch plain.
    kern = _kda_kernel(
        pX,
        pCW,
        pCS,
        pRG,
        pBT,
        pST,
        pIdx,
        pDum,
        pHK,
        pHU,
        pHG,
        pHL,
        pAl,
        pDt,
        pOut,
        pCd,
        T,
        H,
        W,
        CAP,
        S,
        SD,
        XB,
        XT,
        CSS,
        CSC,
        SSS,
        BTB,
        BTT,
        I64,
    )
    if cutlass.const_expr(SD != 0 and T >= 5):
        kern.launch(grid=(H, B, 1), block=(KD, 1, 1), stream=stream, min_blocks_per_mp=3)
    else:
        kern.launch(grid=(H, B, 1), block=(KD, 1, 1), stream=stream)


@torch.no_grad()
def kda_mtp_replay(
    raw_x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    checkpoint: torch.Tensor,
    state_indices: torch.Tensor,
    is_dummy: torch.Tensor,
    history_k: torch.Tensor,
    history_u: torch.Tensor,
    history_G: torch.Tensor,
    history_len: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    output: torch.Tensor,
    candidate_x: torch.Tensor,
) -> None:
    """Fuse Conv4 with one KDA MTP replay step.

    The recurrent and Conv checkpoints are read-only. New K/U/cumulative-G
    rows are appended to the single history buffer, and raw Conv candidates
    are retained for the all-layer post-sampling commit.
    """
    B = raw_x.shape[0]
    T = raw_x.shape[1]
    S, H, CAP, _ = history_k.shape
    W = CAP - T

    if B == 0:
        return
    C = 3 * H * KD
    if not 1 <= T <= 8:
        raise ValueError("KDA MTP replay requires 1 <= tokens_per_step <= 8")
    if W not in (16, 32):
        raise ValueError("KDA MTP replay requires a history window of 16 or 32")
    if raw_x.shape != (B, T, C) or raw_x.dtype != torch.bfloat16:
        raise ValueError("raw_x must be BF16 with shape [batch, tokens, 3 * heads * 128]")
    if raw_x.stride(2) != 1:
        raise ValueError("raw_x channels must be dense")
    if conv_weight.shape != (C, 4) or conv_weight.dtype != torch.bfloat16:
        raise ValueError("conv_weight must be BF16 with shape [channels, 4]")
    if not conv_weight.is_contiguous():
        raise ValueError("conv_weight must be contiguous")
    if conv_state.shape != (S, C, 3) or conv_state.dtype != torch.bfloat16:
        raise ValueError("conv_state must be BF16 with shape [slots, channels, 3]")
    if conv_state.stride()[1:] != (3, 1):
        raise ValueError("conv_state channel/window dimensions must be dense")
    if checkpoint.shape != (S, H, KD, KD):
        raise ValueError("checkpoint must have shape [slots, heads, 128, 128]")
    if checkpoint.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("KDA checkpoint must be BF16 or FP32")
    if checkpoint.stride()[1:] != (KD * KD, KD, 1):
        raise ValueError("checkpoint head/value/key dimensions must be dense")
    if raw_g.shape != (B, T, H, KD) or raw_g.dtype != torch.bfloat16:
        raise ValueError("raw_g must be BF16 with shape [batch, tokens, heads, 128]")
    if not raw_g.is_contiguous():
        raise ValueError("raw_g must be contiguous")
    if raw_beta.shape != (B, T, H) or raw_beta.dtype != torch.bfloat16:
        raise ValueError("raw_beta must be BF16 with shape [batch, tokens, heads]")
    if raw_beta.stride(2) != 1:
        raise ValueError("raw_beta heads must be dense")
    expected_history_shape = (S, H, CAP, KD)
    if history_u.shape != expected_history_shape or history_k.shape != expected_history_shape:
        raise ValueError("KDA K/U histories have an unexpected shape")
    if history_G.shape != expected_history_shape:
        raise ValueError("KDA cumulative-G history has an unexpected shape")
    if history_k.dtype != torch.bfloat16 or history_u.dtype != torch.bfloat16:
        raise TypeError("KDA K/U histories must be BF16")
    if history_G.dtype != torch.float32:
        raise TypeError("KDA cumulative-G history must be FP32")
    if (
        not history_k.is_contiguous()
        or not history_u.is_contiguous()
        or not history_G.is_contiguous()
    ):
        raise ValueError("KDA histories must be contiguous")
    if history_len.shape != (S,) or history_len.dtype != torch.int32:
        raise ValueError("history_len must be int32 with shape [slots]")
    if state_indices.shape != (B,) or state_indices.dtype != torch.int32:
        raise ValueError("state_indices must be int32 with shape [batch]")
    if is_dummy.shape != (B,) or is_dummy.dtype != torch.bool:
        raise ValueError("is_dummy must be bool with shape [batch]")
    if output.shape != (B, T, H, KD) or output.dtype != torch.bfloat16:
        raise ValueError("output must be BF16 with shape [batch, tokens, heads, 128]")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
    if candidate_x.ndim != 3 or candidate_x.shape[1:] != (T, C):
        raise ValueError("candidate_x must have shape [capacity, tokens, channels]")
    if candidate_x.shape[0] < B or candidate_x.dtype != torch.bfloat16:
        raise ValueError("candidate_x must be a sufficiently large BF16 capacity buffer")
    if not candidate_x.is_contiguous():
        raise ValueError("candidate_x must be contiguous")
    if A_log.shape != (H,) or A_log.dtype != torch.float32:
        raise ValueError("A_log must be FP32 with shape [heads]")
    if dt_bias.shape != (H * KD,) or dt_bias.dtype != torch.float32:
        raise ValueError("dt_bias must be FP32 with shape [heads * 128]")

    tensors = (
        raw_x,
        conv_weight,
        conv_state,
        raw_g,
        raw_beta,
        checkpoint,
        state_indices,
        is_dummy,
        history_k,
        history_u,
        history_G,
        history_len,
        A_log,
        dt_bias,
        output,
        candidate_x,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("KDA MTP replay tensors must be CUDA tensors")
    if any(tensor.device != output.device for tensor in tensors):
        raise ValueError("KDA MTP replay tensors must be on the same device")

    sd = int(checkpoint.dtype == torch.float32)
    sss = checkpoint.stride(0)

    xb, xt, _ = raw_x.stride()
    css, csc, _ = conv_state.stride()
    btb, btt, _ = raw_beta.stride()

    i64 = ((S - 1) * sss + H * KD * KD) >= 2**31
    device_index = output.device.index
    key = (device_index, T, H, W, CAP, S, sd, xb, xt, css, csc, sss, btb, btt, i64)
    fn = _CACHE.get(key)

    f32 = cutlass.Float32
    bf16 = cutlass.BFloat16
    i32 = cutlass.Int32
    u8 = cutlass.Uint8
    g = cutlass.AddressSpace.gmem

    args = (
        make_ptr(bf16, raw_x.data_ptr(), g, assumed_align=16),
        make_ptr(bf16, conv_weight.data_ptr(), g, assumed_align=16),
        make_ptr(bf16, conv_state.data_ptr(), g, assumed_align=16),
        make_ptr(bf16, raw_g.data_ptr(), g, assumed_align=16),
        make_ptr(bf16, raw_beta.data_ptr(), g, assumed_align=16),
        make_ptr(f32 if sd else bf16, checkpoint.data_ptr(), g, assumed_align=16),
        make_ptr(i32, state_indices.data_ptr(), g, assumed_align=4),
        make_ptr(u8, is_dummy.data_ptr(), g, assumed_align=1),
        make_ptr(bf16, history_k.data_ptr(), g, assumed_align=16),
        make_ptr(bf16, history_u.data_ptr(), g, assumed_align=16),
        make_ptr(f32, history_G.data_ptr(), g, assumed_align=16),
        make_ptr(i32, history_len.data_ptr(), g, assumed_align=4),
        make_ptr(f32, A_log.data_ptr(), g, assumed_align=4),
        make_ptr(f32, dt_bias.data_ptr(), g, assumed_align=16),
        make_ptr(bf16, output.data_ptr(), g, assumed_align=16),
        make_ptr(bf16, candidate_x.data_ptr(), g, assumed_align=16),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if fn is None:
        fn = cute.compile(
            _launch,
            *args,
            cutlass.Int32(B),
            T,
            H,
            W,
            CAP,
            S,
            sd,
            xb,
            xt,
            css,
            csc,
            sss,
            btb,
            btt,
            i64,
            stream,
        )
        _CACHE[key] = fn
    fn(*args, cutlass.Int32(B), stream)
