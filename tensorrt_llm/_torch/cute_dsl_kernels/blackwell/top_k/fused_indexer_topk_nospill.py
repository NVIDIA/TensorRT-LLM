# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused DSv4 decode indexer + exact top-K, single kernel, no logits in GMEM.

One launch computes relu-weighted 64-head MQA logits over the paged FP4
indexer K cache and selects each row's exact top-K, without ever writing
per-token scores to global memory: scores live as signed-monotone 16-bit
keys in (distributed) shared memory; selection is a two-level histogram
descent (2048 coarse / 32 fine bins) with warp-aggregated output claims.
Rows may be split across a thread-block cluster; cross-CTA state moves
through DSMEM only.

Contract
- kv_cache pages are the production planar layout: per page 2048 B of
  packed fp4 data (32 tokens x 64 B) followed by 128 B of scale words
  (one int32 = 4 ue8m0 exponents per token).
- weights are SIGNED fp32; scores may be negative (signed key order).
- Output values are fp16-rounded scores: the selected set is exact up to
  fp16 ordering at the K-th boundary (observed < 1e-4 relative on real
  captures; the score inputs are themselves fp4-derived).
- n_comp (block_table width x 32) must be one of the validated grid
  sizes {8192, 16384, 32768}; callers pad the block table (dummy page
  ids beyond context_lens are never dereferenced past the length mask).
- context_lens are per-row and arbitrary within n_comp.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import make_ptr

try:
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass.cutlass_dsl import T as _T

    _ASM = True
except Exception:  # pragma: no cover
    _ASM = False

PFD = 4  # default L2 prefetch distance, in 128-token tiles

FP4 = cutlass.Float4E2M1FN
SF = cutlass.Float8E8M0FNU
U8 = cutlass.Uint8
U16 = cutlass.Uint16
U32 = cutlass.Uint32
F16 = cutlass.Float16
F32 = cutlass.Float32
I32 = cutlass.Int32
GMEM = cute.AddressSpace.gmem

TOK = 128  # tokens per MMA tile (M)
HD = 64  # heads (N) -- exact, no zero padding
DIM = 128  # feature dims (K)
DIMB = 64  # bytes per token row
PAGE = 32  # tokens per KV page
PGB = 2176  # bytes per page
NBINS = 2048  # coarse bins = key >> 5 (signed-monotone 16-bit keys)
NFINE = 32  # fine bins  = key & 31
CAP = 2048
CROW = 32  # rows per gmem copy sub-tile

C_THREADS = 384
P_THREADS = 64  # 2 producer warps: same #cp.async instructions in flight
PPP = 16  # producer threads per KV page
M_WARP = 14
NTHREADS = 480  # 15 warps => 136 regs/thread, enough to keep all 64
# per-head weights resident in registers for the scan
NACC = 3


@cute.jit
def _pfl2(addr):
    """prefetch.global.L2 [addr] -- fire-and-forget L2 warm-up.

    cp.async.cg misses go all the way to DRAM (~900 cycles); the number of
    outstanding line fills an SM can track is what caps its streaming rate.
    Warming L2 a few tiles ahead turns those misses into ~250-cycle L2 hits,
    so the same number of in-flight requests carries several times the
    bandwidth.  Prefetches do not allocate an L1 fill buffer."""
    _llvm.inline_asm(
        _T.i32(),
        [cutlass.Int64(addr).ir_value()],
        "prefetch.global.L2 [$1];\n\tmov.u32 $0, 0;",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _pick(sHist, sPart, sCtl, tidx, warp_idx, target, out):
    """Find bin b where the descending cumulative count first reaches `target`;
    write (b, count_strictly_above) to sCtl[out], sCtl[out+1]."""
    if tidx < NBINS // 8:
        base = NBINS - 8 * (tidx + 1)
        acc = I32(0)
        for j in cutlass.range_constexpr(8):
            acc = acc + sHist[base + j]
        sPart[tidx] = acc
    cute.arch.barrier()
    if warp_idx == 0:
        lane = tidx
        ls = I32(0)
        for j in cutlass.range_constexpr(NBINS // 256):
            ls = ls + sPart[lane * (NBINS // 256) + j]
        incl = ls
        for d in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_up(incl, 1 << d, mask_and_clamp=0)
            if lane >= (1 << d):
                incl = incl + other
        excl = incl - ls
        if excl < target and incl >= target:
            acc = excl
            gsel = I32(0)
            done = I32(0)
            for j in cutlass.range_constexpr(NBINS // 256):
                nxt = acc + sPart[lane * (NBINS // 256) + j]
                if done == 0:
                    if nxt >= target:
                        gsel = I32(lane * (NBINS // 256) + j)
                        done = I32(1)
                    else:
                        acc = nxt
            base2 = NBINS - 8 * (gsel + 1)
            bsel = I32(0)
            done2 = I32(0)
            for j in cutlass.range_constexpr(8):
                bb = base2 + 7 - j
                nxt = acc + sHist[bb]
                if done2 == 0:
                    if nxt >= target:
                        bsel = bb
                        done2 = I32(1)
                    else:
                        acc = nxt
            sCtl[out] = bsel
            sCtl[out + 1] = acc


@cute.jit
def _pick32(sFine, sCtl, tidx, warp_idx, target, out):
    """Same descending-cumulative search over the 32 fine bins (one warp)."""
    if warp_idx == 0:
        lane = tidx
        c = sFine[NFINE - 1 - lane]
        incl = c
        for d in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_up(incl, 1 << d, mask_and_clamp=0)
            if lane >= (1 << d):
                incl = incl + other
        excl = incl - c
        if excl < target and incl >= target:
            sCtl[out] = I32(NFINE - 1) - lane
            sCtl[out + 1] = excl


@cute.kernel
def _dsv4_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_k: cute.CopyAtom,
    mKV: cute.Tensor,
    smem_layout_k_tma: cute.ComposedLayout,
    kv_ptr: cute.Pointer,
    q_ptr: cute.Pointer,
    sfq_ptr: cute.Pointer,
    w_ptr: cute.Pointer,
    clen_ptr: cute.Pointer,
    bt_ptr: cute.Pointer,
    oi_ptr: cute.Pointer,
    ov_ptr: cute.Pointer,
    LAY: cutlass.Constexpr,
    SWZ: cutlass.Constexpr,
    NCOMP: cutlass.Constexpr,
    KTOP: cutlass.Constexpr,
    MAXB: cutlass.Constexpr,
    STAGES: cutlass.Constexpr,
    CS: cutlass.Constexpr,
    NLOC: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    g2s = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), FP4, num_bits_per_copy=128
    )
    gcpq = cute.make_tiled_copy_tv(
        g2s,
        cute.make_ordered_layout((8, 4), order=(1, 0)),
        cute.make_ordered_layout((4, 32), order=(1, 0)),
    )
    qcp = cute.make_tiled_copy_tv(
        g2s, cute.make_ordered_layout((CROW, 4), order=(1, 0)), cute.make_layout((1, 32))
    )
    g2s_u32 = cute.make_copy_atom(cpasync.CopyG2SOp(), U32, num_bits_per_copy=32)

    smem = utils.SmemAllocator()
    sA_raw = smem.allocate_array(U32, STAGES * TOK * DIMB // 4, byte_alignment=1024)
    sB_raw = smem.allocate_array(U32, HD * DIMB // 4, byte_alignment=1024)
    sSFA_raw = smem.allocate_array(U32, STAGES * 128, byte_alignment=128)
    sSFB_raw = smem.allocate_array(U32, 128, byte_alignment=128)
    sKey_raw = smem.allocate_array(U16, NLOC * TOK, byte_alignment=128)
    sHist = smem.allocate_tensor(I32, cute.make_layout(NBINS), byte_alignment=128)
    sTot = smem.allocate_tensor(I32, cute.make_layout(NBINS), byte_alignment=128)
    sFTot = smem.allocate_tensor(I32, cute.make_layout(NFINE), byte_alignment=128)
    sW = smem.allocate_tensor(F32, cute.make_layout(64), byte_alignment=128)
    sPart = smem.allocate_tensor(I32, cute.make_layout(NBINS // 8), byte_alignment=128)
    sCtl = smem.allocate_tensor(I32, cute.make_layout(32), byte_alignment=128)
    sFine = smem.allocate_tensor(I32, cute.make_layout(NFINE), byte_alignment=128)
    sBT = smem.allocate_tensor(I32, cute.make_layout(MAXB), byte_alignment=128)
    sCand = smem.allocate_tensor(I32, cute.make_layout(CAP), byte_alignment=128)
    tmem_hold = smem.allocate_array(I32, 1, byte_alignment=16)
    mbar = smem.allocate_array(cutlass.Int64, 2 * STAGES + 2 * NACC, byte_alignment=16)

    sKey = cute.make_tensor(sKey_raw, cute.make_layout(NLOC * TOK))
    sVal = cute.make_tensor(cute.recast_ptr(sKey_raw, dtype=F16), cute.make_layout(NLOC * TOK))
    sKey32 = cute.make_tensor(
        cute.recast_ptr(sKey_raw, dtype=U32), cute.make_layout(NLOC * TOK // 2)
    )

    ab_full = mbar
    ab_empty = mbar + STAGES
    acc_full = mbar + 2 * STAGES
    acc_empty = mbar + 2 * STAGES + NACC

    if tidx == 0:
        for i in cutlass.range_constexpr(STAGES):
            cute.arch.mbarrier_init(ab_full + i, P_THREADS)
            cute.arch.mbarrier_init(ab_empty + i, 1)
        for i in cutlass.range_constexpr(NACC):
            cute.arch.mbarrier_init(acc_full + i, 1)
            cute.arch.mbarrier_init(acc_empty + i, 128)
        cute.arch.mbarrier_init_fence()

    sA_swz = cute.make_swizzle(SWZ[0], SWZ[1], SWZ[2])
    sA_ptr = cute.recast_ptr(sA_raw, sA_swz, dtype=FP4)
    sB_ptr = cute.recast_ptr(sB_raw, sA_swz, dtype=FP4)
    sA = cute.make_tensor(sA_ptr, cute.make_layout(LAY[0][0], stride=LAY[0][1]))
    sB = cute.make_tensor(sB_ptr, cute.make_layout(LAY[1][0], stride=LAY[1][1]))
    # Partition the staged UMMA A layout into four independent 32x128 TMA
    # destinations (TOK / PAGE = 4 page slots per MMA tile).  The GMEM tensor
    # keeps the physical page pool as its residual mode, so an arbitrary
    # block-table page ID is a legal runtime coordinate and does not require
    # rebuilding the TMA descriptor.
    sK_tma = cute.make_tensor(cute.recast_ptr(sA_raw, dtype=FP4), smem_layout_k_tma)
    sK_tma, gK_tma = cute.nvgpu.cpasync.tma_partition(
        tma_atom_k,
        0,
        cute.make_layout(1),
        smem_tensor=sK_tma,
        gmem_tensor=cute.group_modes(mKV, 0, 2),
    )
    sSFA = cute.make_tensor(
        cute.recast_ptr(sSFA_raw, dtype=SF), cute.make_layout(LAY[2][0], stride=LAY[2][1])
    )
    sSFB = cute.make_tensor(
        cute.recast_ptr(sSFB_raw, dtype=SF), cute.make_layout(LAY[3][0], stride=LAY[3][1])
    )
    acc_layout = cute.make_layout(LAY[4][0], stride=LAY[4][1])
    sfa_tmem_layout = cute.make_layout(LAY[5][0], stride=LAY[5][1])
    sfb_tmem_layout = cute.make_layout(LAY[6][0], stride=LAY[6][1])
    sSFB_u32 = cute.make_tensor(sSFB_raw, cute.make_layout(128))

    if tidx >= 128:
        ii = tidx - 128
        for i in cutlass.range(ii, NBINS, NTHREADS - 128, unroll=1):
            sHist[i] = I32(0)
            sTot[i] = I32(0)
    if tidx < 32:
        sCtl[tidx] = I32(0)
        sFine[tidx] = I32(0)
        sFTot[tidx] = I32(0)

    if warp_idx == 0:
        cute.arch.alloc_tmem(512, tmem_hold)
    if warp_idx == 11:
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)

    b = bidx // CS
    crk = bidx % CS
    L = cute.make_tensor(clen_ptr, cute.make_layout(1 << 20))[b]
    ntile = (L + TOK - 1) // TOK
    # this CTA owns the interleaved tile subsequence crk, crk+CS, crk+2CS, ...
    ntl = 0
    if ntile > crk:
        ntl = (ntile - crk + CS - 1) // CS

    gBT = cute.make_tensor(bt_ptr + cutlass.Int64(b) * MAXB, cute.make_layout(MAXB))
    if tidx >= 128:
        ii = tidx - 128
        for i in cutlass.range(ii, MAXB, NTHREADS - 128, unroll=1):
            sBT[i] = gBT[i]

    if tidx < 64:
        sW[tidx] = cute.make_tensor(w_ptr, cute.make_layout(1 << 20))[b * 64 + tidx]
    if tidx < 128:
        sfq = cute.make_tensor(sfq_ptr, cute.make_layout(1 << 20))
        sSFB_u32[(tidx % 32) * 4 + (tidx // 32)] = sfq[
            b * 64 + ((tidx % 32) + 32 * (tidx // 32)) % 64
        ]

    if tidx < 128:
        thr_q = qcp.get_slice(tidx)
        sB_c = cute.make_tensor(sB_ptr, cute.make_layout((HD, DIM), stride=(DIM, 1)))
        gQ = cute.make_tensor(
            cute.make_ptr(
                FP4, q_ptr.toint() + cutlass.Int64(b) * (64 * DIMB), GMEM, assumed_align=16
            ),
            cute.make_layout((64, DIM), stride=(DIM, 1)),
        )
        for r in cutlass.range_constexpr(2):
            src = cute.local_tile(gQ, (CROW, DIM), (r, 0))
            dst = cute.local_tile(sB_c, (CROW, DIM), (r, 0))
            cute.copy(gcpq, thr_q.partition_S(src), thr_q.partition_D(dst))
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

    cute.arch.barrier()

    tmem_ptr = cute.arch.retrieve_tmem_ptr(F32, 16, tmem_hold)
    # One scale-factor TMEM buffer PER accumulator.  With a single shared SFA
    # buffer the tcgen05.cp for tile i+1 cannot start until the MMA of tile i
    # has drained it, which serialises the whole MMA pipeline.
    tSFA0 = cute.make_tensor(cute.recast_ptr(tmem_ptr + 256, dtype=SF), sfa_tmem_layout)
    tSFA1 = cute.make_tensor(cute.recast_ptr(tmem_ptr + 288, dtype=SF), sfa_tmem_layout)
    tSFA2 = cute.make_tensor(cute.recast_ptr(tmem_ptr + 320, dtype=SF), sfa_tmem_layout)
    tSFB = cute.make_tensor(cute.recast_ptr(tmem_ptr + 352, dtype=SF), sfb_tmem_layout)

    tAcc0 = cute.make_tensor(tmem_ptr, acc_layout)
    tAcc1 = cute.make_tensor(tmem_ptr + HD, acc_layout)
    tAcc2 = cute.make_tensor(tmem_ptr + 2 * HD, acc_layout)

    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)

    ca = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), SF)
    tc_sfa0 = tcgen05.make_s2t_copy(ca, cute.filter_zeros(tSFA0))
    tc_sfa1 = tcgen05.make_s2t_copy(ca, cute.filter_zeros(tSFA1))
    tc_sfa2 = tcgen05.make_s2t_copy(ca, cute.filter_zeros(tSFA2))
    tc_sfb = tcgen05.make_s2t_copy(ca, cute.filter_zeros(tSFB))
    ta0 = tc_sfa0.get_slice(0)
    ta1 = tc_sfa1.get_slice(0)
    ta2 = tc_sfa2.get_slice(0)
    sfa_src0 = tcgen05.get_s2t_smem_desc_tensor(tc_sfa0, ta0.partition_S(cute.filter_zeros(sSFA)))
    sfa_src1 = tcgen05.get_s2t_smem_desc_tensor(tc_sfa1, ta1.partition_S(cute.filter_zeros(sSFA)))
    sfa_src2 = tcgen05.get_s2t_smem_desc_tensor(tc_sfa2, ta2.partition_S(cute.filter_zeros(sSFA)))
    sfa_dst0 = ta0.partition_D(cute.filter_zeros(tSFA0))
    sfa_dst1 = ta1.partition_D(cute.filter_zeros(tSFA1))
    sfa_dst2 = ta2.partition_D(cute.filter_zeros(tSFA2))
    tb = tc_sfb.get_slice(0)
    sfb_src = tcgen05.get_s2t_smem_desc_tensor(tc_sfb, tb.partition_S(cute.filter_zeros(sSFB)))
    sfb_dst = tb.partition_D(cute.filter_zeros(tSFB))

    # ---------------- producer warps 12..15 ----------------
    if warp_idx >= 12 and warp_idx < M_WARP:
        lt = tidx - C_THREADS
        pgt = lt // PPP
        tkt = lt % PPP
        # A TMA issue must be warp uniform, but this kernel's producer geometry
        # splits each of the two producer warps into two 16-lane page groups.
        # So the DATA plane is issued per warp: warp w owns the two page slots
        # 2*(w-12) and 2*(w-12)+1, i.e. exactly the pages its two half warps
        # would have fetched with cp.async.  The SCALE plane keeps the original
        # 16-lane-per-page decomposition and all 64 per-thread arrivals.
        pgw = 2 * (warp_idx - 12)
        for j in cutlass.range(ntl, unroll=1):
            i = crk + j * CS
            s = j % STAGES
            if j >= STAGES:
                cute.arch.mbarrier_wait(ab_empty + s, ((j // STAGES) - 1) & 1)
            if cutlass.const_expr(_ASM):
                jf = j + PFD
                if cutlass.const_expr(NCOMP == 32768):
                    jf = j + 2
                if jf < ntl:
                    pgf = sBT[(crk + jf * CS) * 4 + pgt]
                    base = kv_ptr.toint() + cutlass.Int64(pgf) * PGB
                    _pfl2(base + tkt * 128)
                    if tkt == 0:
                        _pfl2(base + 2048)
            # scale plane keeps the per-half-warp page; data plane is per warp
            pg0 = cute.arch.make_warp_uniform(sBT[i * 4 + pgw])
            pg1 = cute.arch.make_warp_uniform(sBT[i * 4 + pgw + 1])
            pg = pg0
            if pgt != pgw:
                pg = pg1
            # One elected expect_tx per producer warp covers both of that warp's
            # page transactions; two warps therefore account for the full
            # 4 * PAGE * DIMB = 8192 bytes of the tile on the same full barrier.
            with cute.arch.elect_one():
                cute.arch.mbarrier_expect_tx(ab_full + s, 2 * PAGE * DIMB)
            cute.copy(
                tma_atom_k,
                gK_tma[(None, pg0)],
                sK_tma[(None, pgw, s)],
                tma_bar_ptr=ab_full + s,
            )
            cute.copy(
                tma_atom_k,
                gK_tma[(None, pg1)],
                sK_tma[(None, pgw + 1, s)],
                tma_bar_ptr=ab_full + s,
            )
            for r in cutlass.range_constexpr(PAGE // PPP):
                tk = tkt + PPP * r
                gsf = cute.make_tensor(
                    cute.make_ptr(
                        U32,
                        kv_ptr.toint() + cutlass.Int64(pg) * PGB + 2048 + tk * 4,
                        GMEM,
                        assumed_align=4,
                    ),
                    cute.make_layout(1),
                )
                dsf = cute.make_tensor(sSFA_raw + (s * 128 + tk * 4 + pgt), cute.make_layout(1))
                cute.copy(g2s_u32, gsf, dsf)
            cute.arch.cp_async_mbarrier_arrive_noinc(ab_full + s)

    # ---------------- mma warp ----------------
    elif warp_idx == M_WARP:
        cute.copy(tc_sfb, sfb_src[(None, None, None, None, 0)], sfb_dst)
        for j in cutlass.range(ntl, unroll=1):
            s = j % STAGES
            a = j % NACC
            cute.arch.mbarrier_wait(ab_full + s, (j // STAGES) & 1)
            if j >= NACC:
                cute.arch.mbarrier_wait(acc_empty + a, ((j // NACC) - 1) & 1)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            if a == 0:
                cute.copy(tc_sfa0, sfa_src0[(None, None, None, None, s)], sfa_dst0)
                cute.gemm(
                    tiled_mma,
                    tAcc0,
                    [tCrA[(None, None, None, s)], tSFA0],
                    [tCrB[(None, None, None, 0)], tSFB],
                    tAcc0,
                )
            elif a == 1:
                cute.copy(tc_sfa1, sfa_src1[(None, None, None, None, s)], sfa_dst1)
                cute.gemm(
                    tiled_mma,
                    tAcc1,
                    [tCrA[(None, None, None, s)], tSFA1],
                    [tCrB[(None, None, None, 0)], tSFB],
                    tAcc1,
                )
            else:
                cute.copy(tc_sfa2, sfa_src2[(None, None, None, None, s)], sfa_dst2)
                cute.gemm(
                    tiled_mma,
                    tAcc2,
                    [tCrA[(None, None, None, s)], tSFA2],
                    [tCrB[(None, None, None, 0)], tSFB],
                    tAcc2,
                )
            with cute.arch.elect_one():
                tcgen05.commit(acc_full + a)
                tcgen05.commit(ab_empty + s)

    # ---------------- consumer warps 0..11 ----------------
    elif warp_idx < 12:
        op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE)
        atom_t2r = cute.make_copy_atom(op, F32)
        lane = tidx % 128
        gwg = tidx // 128
        lay32 = cute.make_layout(((TOK, 32), 1, 1), stride=((65536, 1), 0, 0))
        a0 = cute.make_tensor(tmem_ptr, lay32)
        tt = tcgen05.make_tmem_copy(atom_t2r, a0)
        thr = tt.get_slice(lane)
        s00 = thr.partition_S(cute.make_tensor(tmem_ptr, lay32))
        s01 = thr.partition_S(cute.make_tensor(tmem_ptr + 32, lay32))
        s10 = thr.partition_S(cute.make_tensor(tmem_ptr + HD, lay32))
        s11 = thr.partition_S(cute.make_tensor(tmem_ptr + HD + 32, lay32))
        s20 = thr.partition_S(cute.make_tensor(tmem_ptr + 2 * HD, lay32))
        s21 = thr.partition_S(cute.make_tensor(tmem_ptr + 2 * HD + 32, lay32))
        tD = thr.partition_D(cute.make_identity_tensor(a0.shape))
        frg = cute.make_rmem_tensor(tD.shape, F32)
        # All 64 per-head weights are loop invariant: hoist them into registers
        # once (15 warps => 136 regs/thread), which removes 16 LDS.128 per tile
        # per lane from the innermost loop entirely.
        wr = cute.make_rmem_tensor(cute.make_layout(64), F32)
        cute.autovec_copy(sW, wr)
        z = F32(0.0)
        # relu is scalar (there is no max.f32x2) but the weighted accumulation
        # runs on the Blackwell packed FP32 datapath: 64 fmax + 32 fma.f32x2
        # instead of 64 fmax + 64 fma per token.
        for i in cutlass.range(gwg, ntl, NACC, unroll=1):
            cute.arch.mbarrier_wait(acc_full + gwg, (i // NACC) & 1)
            if gwg == 0:
                cute.copy(tt, s00, frg)
            elif gwg == 1:
                cute.copy(tt, s10, frg)
            else:
                cute.copy(tt, s20, frg)
            cute.arch.fence_view_async_tmem_load()
            acc = [[z, z], [z, z], [z, z], [z, z]]
            for j in cutlass.range_constexpr(16):
                k = j & 3
                r0 = cute.arch.fmax(frg[2 * j], z)
                r1 = cute.arch.fmax(frg[2 * j + 1], z)
                acc[k][0], acc[k][1] = cute.arch.fma_packed_f32x2(
                    (wr[2 * j], wr[2 * j + 1]), (r0, r1), (acc[k][0], acc[k][1])
                )
            if gwg == 0:
                cute.copy(tt, s01, frg)
            elif gwg == 1:
                cute.copy(tt, s11, frg)
            else:
                cute.copy(tt, s21, frg)
            cute.arch.fence_view_async_tmem_load()
            cute.arch.mbarrier_arrive(acc_empty + gwg)
            for j in cutlass.range_constexpr(16):
                k = j & 3
                r0 = cute.arch.fmax(frg[2 * j], z)
                r1 = cute.arch.fmax(frg[2 * j + 1], z)
                acc[k][0], acc[k][1] = cute.arch.fma_packed_f32x2(
                    (wr[32 + 2 * j], wr[33 + 2 * j]), (r0, r1), (acc[k][0], acc[k][1])
                )
            q0 = cute.arch.add_packed_f32x2((acc[0][0], acc[0][1]), (acc[1][0], acc[1][1]))
            q1 = cute.arch.add_packed_f32x2((acc[2][0], acc[2][1]), (acc[3][0], acc[3][1]))
            q2 = cute.arch.add_packed_f32x2(q0, q1)
            sc = q2[0] + q2[1]
            # signed monotone key: k = u ^ (0x8000 | sign*0x7FFF). The
            # buffer stores KEYS, so every downstream ordering site (coarse
            # bin >> 5, fine bin & 31, comparisons) is unchanged; only the
            # value OUTPUT sites decode back to fp16 bits.
            hv0 = sc.to(F16)
            ui = I32(hv0.bitcast(U16)) & 0xFFFF
            ki = ui ^ (0x8000 + ((ui >> 15) & 1) * 0x7FFF)
            hv = U16(ki).bitcast(F16)
            pos = i * TOK + lane
            sVal[pos] = hv
            if (crk + i * CS) * TOK + lane < L:
                cute.arch.atomic_add(sHist.iterator + (ki >> 5), I32(1), scope="cta")

    # Each warp signals cluster arrival as soon as its own scan work is done;
    # the matching wait sits after the CTA barrier, so a CTA only pays the
    # cross-CTA skew instead of a full barrier round trip.
    if cutlass.const_expr(CS > 1):
        cute.arch.cluster_arrive()
    cute.arch.barrier()
    if warp_idx == 0:
        cute.arch.dealloc_tmem(tmem_ptr, 512)

    gI = cute.make_tensor(oi_ptr + cutlass.Int64(b) * KTOP, cute.make_layout(KTOP))
    gV = cute.make_tensor(ov_ptr + cutlass.Int64(b) * KTOP, cute.make_layout(KTOP))

    # counters live in cluster-rank-0's SMEM; every CTA of the row claims there
    cnt = sCtl.iterator + 8
    if cutlass.const_expr(CS > 1):
        cute.arch.cluster_wait()
        cnt = cute.arch.map_dsmem_ptr(sCtl.iterator + 8, 0)
        pts = [cute.arch.map_dsmem_ptr(sTot.iterator, c) for c in range(CS)]
        if cutlass.const_expr(CS >= 8):
            # Reduce-scatter to bin owners, then owners broadcast their final
            # segment: 2*NBINS remote adds per CTA instead of NBINS*CS. Below
            # CS=8 the all-to-all traffic is cheaper than the extra rendezvous.
            SEG = NBINS // CS
            for i in cutlass.range(tidx, NBINS, NTHREADS, unroll=1):
                v = sHist[i]
                if v != 0:
                    own = i // SEG
                    for c in cutlass.range_constexpr(CS):
                        if own == c:
                            cute.arch.atomic_add(pts[c] + i, v, scope="cluster")
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            for i in cutlass.range(tidx, SEG, NTHREADS, unroll=1):
                bi = crk * SEG + i
                v = sTot[bi]
                if v != 0:
                    for c in cutlass.range_constexpr(CS):
                        if c != crk:
                            cute.arch.atomic_add(pts[c] + bi, v, scope="cluster")
        else:
            for i in cutlass.range(tidx, NBINS, NTHREADS, unroll=1):
                v = sHist[i]
                if v != 0:
                    for c in cutlass.range_constexpr(CS):
                        cute.arch.atomic_add(pts[c] + i, v, scope="cluster")
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

    # ---------------- level 1: coarse descent on the online histogram --------
    if cutlass.const_expr(CS > 1):
        _pick(sTot, sPart, sCtl, tidx, warp_idx, I32(KTOP), 0)
    else:
        _pick(sHist, sPart, sCtl, tidx, warp_idx, I32(KTOP), 0)
    cute.arch.barrier()
    b1 = sCtl[0]
    cabove = sCtl[1]
    r1 = I32(KTOP) - cabove

    # ---------------- single streaming pass: claim + fine histogram ----------
    # Single-CTA rows stage winners in SMEM (sHist / sTot are dead after the
    # coarse descent; local slot == output slot) and write them out with full
    # lines after the pass, instead of per-warp partial-line stores.
    sWK = cute.make_tensor(cute.recast_ptr(sTot.iterator, dtype=U16), cute.make_layout(2 * NBINS))
    nw = ntl * (TOK // 2)
    niter = (nw + NTHREADS - 1) // NTHREADS
    lane = cute.arch.lane_idx()
    lmask = (I32(1) << lane) - I32(1)
    for it in cutlass.range(niter, unroll=1):
        w = it * NTHREADS + tidx
        live = w < nw
        xi = I32(0)
        if live:
            xi = I32(sKey32[w])
        kv0 = U16(xi & 0xFFFF)
        kv1 = U16((xi >> 16) & 0xFFFF)
        p0 = 2 * w
        # local slot -> global kv position
        t0 = (crk + (p0 >> 7) * CS) * TOK + (p0 & (TOK - 1))
        t1 = t0 + 1
        n0 = I32(kv0) >> 5
        n1 = I32(kv1) >> 5
        hi0 = live and (t0 < L) and (n0 > b1)
        hi1 = live and (t1 < L) and (n1 > b1)
        # warp-aggregated claim: lanes of a warp take consecutive output slots,
        # which turns 32 scattered 4B stores into one coalesced pair of stores.
        m0 = cute.arch.vote_ballot_sync(hi0)
        m1 = cute.arch.vote_ballot_sync(hi1)
        tot = cute.arch.popc(m0) + cute.arch.popc(m1)
        base = I32(0)
        if lane == 0:
            base = cute.arch.atomic_add(cnt, tot, scope="cluster")
        base = cute.arch.shuffle_sync(base, 0)
        n0lo = cute.arch.popc(m0 & lmask)
        if hi0:
            p = base + n0lo
            if cutlass.const_expr(CS == 1):
                sHist[p] = t0
                sWK[p] = kv0
            else:
                gI[p] = t0
                u0 = I32(kv0) ^ (0x8000 + ((((I32(kv0) >> 15) & 1) ^ 1) * 0x7FFF))
                gV[p] = U16(u0 & 0xFFFF).bitcast(F16).to(F32)
        if hi1:
            p = base + cute.arch.popc(m0) + cute.arch.popc(m1 & lmask)
            if cutlass.const_expr(CS == 1):
                sHist[p] = t1
                sWK[p] = kv1
            else:
                gI[p] = t1
                u1 = I32(kv1) ^ (0x8000 + ((((I32(kv1) >> 15) & 1) ^ 1) * 0x7FFF))
                gV[p] = U16(u1 & 0xFFFF).bitcast(F16).to(F32)
        if live and (t0 < L) and (n0 == b1):
            q = cute.arch.atomic_add(sCtl.iterator + 11, I32(1), scope="cta")
            if q < CAP:
                sCand[q] = p0
            cute.arch.atomic_add(sFine.iterator + (I32(kv0) & (NFINE - 1)), I32(1), scope="cta")
        if live and (t1 < L) and (n1 == b1):
            q = cute.arch.atomic_add(sCtl.iterator + 11, I32(1), scope="cta")
            if q < CAP:
                sCand[q] = p0 + 1
            cute.arch.atomic_add(sFine.iterator + (I32(kv1) & (NFINE - 1)), I32(1), scope="cta")
    cute.arch.barrier()

    if cutlass.const_expr(CS == 1):
        nwin = sCtl[8]
        for j in cutlass.range(tidx, nwin, NTHREADS, unroll=1):
            gI[j] = sHist[j]
            kw = I32(sWK[j])
            uw = kw ^ (0x8000 + ((((kw >> 15) & 1) ^ 1) * 0x7FFF))
            gV[j] = U16(uw & 0xFFFF).bitcast(F16).to(F32)
    # No rendezvous before the fine push: peer sFTot buffers were zeroed before
    # the scan-end arrive and nobody reads them until the wait below.
    if cutlass.const_expr(CS > 1):
        ptf = [cute.arch.map_dsmem_ptr(sFTot.iterator, c) for c in range(CS)]
        if tidx < NFINE:
            v = sFine[tidx]
            if v != 0:
                for c in cutlass.range_constexpr(CS):
                    cute.arch.atomic_add(ptf[c] + tidx, v, scope="cluster")
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

    # ---------------- level 2: fine descent (32 bins) ------------------------
    if cutlass.const_expr(CS > 1):
        _pick32(sFTot, sCtl, tidx, warp_idx, r1, 2)
    else:
        _pick32(sFine, sCtl, tidx, warp_idx, r1, 2)
    cute.arch.barrier()
    b2 = sCtl[2]
    c2above = sCtl[3]
    r2 = r1 - c2above
    base2 = cabove + c2above

    # ---------------- boundary claim + tie fill ------------------------------
    cnt9 = sCtl.iterator + 9
    cnt10 = sCtl.iterator + 10
    if cutlass.const_expr(CS > 1):
        cnt9 = cute.arch.map_dsmem_ptr(sCtl.iterator + 9, 0)
        cnt10 = cute.arch.map_dsmem_ptr(sCtl.iterator + 10, 0)
    ncand = sCtl[11]
    nscan = ncand
    if ncand > CAP:
        nscan = ntl * TOK
    for ci in cutlass.range(tidx, nscan, NTHREADS, unroll=2):
        pl = ci
        if ncand <= CAP:
            pl = sCand[ci]
        t = (crk + (pl >> 7) * CS) * TOK + (pl & (TOK - 1))
        kv = sKey[pl]
        kk = I32(kv)
        take = t < L
        if ncand <= CAP:
            take = True
        else:
            take = take and ((kk >> 5) == b1)
        if take:
            k2 = kk & (NFINE - 1)
            if k2 > b2:
                p = cute.arch.atomic_add(cnt9, I32(1), scope="cluster")
                ub = kk ^ (0x8000 + ((((kk >> 15) & 1) ^ 1) * 0x7FFF))
                gI[cabove + p] = t
                gV[cabove + p] = U16(ub & 0xFFFF).bitcast(F16).to(F32)
            elif k2 == b2:
                p = cute.arch.atomic_add(cnt10, I32(1), scope="cluster")
                if p < r2:
                    gI[base2 + p] = t
                    ub2 = kk ^ (0x8000 + ((((kk >> 15) & 1) ^ 1) * 0x7FFF))
                    gV[base2 + p] = U16(ub2 & 0xFFFF).bitcast(F16).to(F32)
    if cutlass.const_expr(CS > 1):
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()


@cute.jit
def _launch(
    kv_ptr: cute.Pointer,
    q_ptr: cute.Pointer,
    sfq_ptr: cute.Pointer,
    w_ptr: cute.Pointer,
    clen_ptr: cute.Pointer,
    bt_ptr: cute.Pointer,
    oi_ptr: cute.Pointer,
    ov_ptr: cute.Pointer,
    stream: cuda.CUstream,
    B: cutlass.Constexpr,
    NCOMP: cutlass.Constexpr,
    KTOP: cutlass.Constexpr,
    MAXB: cutlass.Constexpr,
    NPAGES: cutlass.Constexpr,
    STAGES: cutlass.Constexpr,
    CS: cutlass.Constexpr,
    NLOC: cutlass.Constexpr,
):
    tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
        FP4, FP4, OperandMajorMode.K, OperandMajorMode.K, SF, 32, tcgen05.CtaGroup.ONE, (TOK, HD)
    )
    mt = (TOK, HD, DIM)
    sA_layout = sm100_utils.make_smem_layout_a(tiled_mma, mt, FP4, STAGES)
    sB_layout = sm100_utils.make_smem_layout_b(tiled_mma, mt, FP4, 1)
    sSFA_layout = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mt, 32, STAGES)
    sSFB_layout = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mt, 32, 1)

    # Present the paged allocation as (token-within-page, dim, physical-page).
    # PGB is a byte stride, hence 2*PGB in the FP4 element domain.
    mKV = cute.make_tensor(
        cute.recast_ptr(kv_ptr, dtype=FP4),
        cute.make_layout(
            (PAGE, DIM, NPAGES),
            stride=(DIM, 1, PGB * 2),
        ),
    )
    sA_layout_mk = cute.composition(sA_layout, cute.make_layout((TOK, DIM, STAGES)))
    smem_layout_k_tma = cute.tiled_divide(sA_layout_mk, (PAGE, DIM))
    smem_layout_k_tma = cute.select(smem_layout_k_tma, [0, 1, 3])
    tma_atom_k, mKV = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        mKV,
        smem_layout_k_tma[0],
        (PAGE, DIM),
    )

    acc_layout = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((TOK, HD))).layout
    sfa_tmem_layout = blockscaled_utils.make_tmem_layout_sfa(
        tiled_mma, mt, 32, cute.slice_(sSFA_layout, (None, None, None, 0))
    )
    sfb_tmem_layout = blockscaled_utils.make_tmem_layout_sfb(
        tiled_mma, mt, 32, cute.slice_(sSFB_layout, (None, None, None, 0))
    )

    def _sl(lay):
        return (lay.shape, lay.stride)

    LAY = (
        _sl(sA_layout.outer),
        _sl(sB_layout.outer),
        _sl(sSFA_layout),
        _sl(sSFB_layout),
        _sl(acc_layout),
        _sl(sfa_tmem_layout),
        _sl(sfb_tmem_layout),
    )
    swz = sA_layout.inner
    SWZ = (swz.num_bits, swz.num_base, swz.num_shift)

    smem_bytes = (
        STAGES * (TOK * DIMB + 512)
        + HD * DIMB
        + 512
        + NLOC * TOK * 2
        + 2 * NBINS * 4
        + 64 * 4
        + (NBINS // 8) * 4
        + 32 * 4
        + 2 * NFINE * 4
        + 64
        + (2 * STAGES + 2 * NACC) * 8
        + MAXB * 4
        + CAP * 4
        + 2048
    )

    _dsv4_kernel(
        tiled_mma,
        tma_atom_k,
        mKV,
        smem_layout_k_tma,
        kv_ptr,
        q_ptr,
        sfq_ptr,
        w_ptr,
        clen_ptr,
        bt_ptr,
        oi_ptr,
        ov_ptr,
        LAY,
        SWZ,
        NCOMP,
        KTOP,
        MAXB,
        STAGES,
        CS,
        NLOC,
    ).launch(
        grid=[B * CS, 1, 1],
        block=[NTHREADS, 1, 1],
        cluster=[CS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


_cache = {}

SMEM_CAP = 231424


def _stages(NLOC, MAXB):
    fixed = (
        HD * DIMB
        + 512
        + NLOC * TOK * 2
        + 2 * NBINS * 4
        + 64 * 4
        + (NBINS // 8) * 4
        + 32 * 4
        + 2 * NFINE * 4
        + 64
        + MAXB * 4
        + CAP * 4
        + 2048
    )
    s = (SMEM_CAP - fixed) // (TOK * DIMB + 512 + 16)
    if s > 16:
        s = 16
    if s < 4:
        s = 4
    return int(s)


@torch.no_grad()
def run(q_fp4, sf_q, kv_cache, weights, context_lens, block_table, top_k_t, indices, values):
    B, MAXB = block_table.shape
    NCOMP = MAXB * 32
    NPAGES = kv_cache.numel() // PGB
    KTOP = indices.shape[1]
    # split one row across a cluster of CTAs when the batch cannot fill the GPU;
    # scores stay in each CTA's SMEM and the top-K is reduced through DSMEM.
    CS = 1
    while CS < 16 and (B * CS * 2) <= 148:
        CS *= 2
    NLOC = (NCOMP // TOK + CS - 1) // CS
    STAGES = _stages(NLOC, MAXB)
    key = (B, NCOMP, KTOP, MAXB, NPAGES, STAGES, CS, NLOC)
    kv_ptr = make_ptr(U8, kv_cache.data_ptr(), GMEM, assumed_align=16)
    q_ptr = make_ptr(U8, q_fp4.data_ptr(), GMEM, assumed_align=16)
    sfq_ptr = make_ptr(U32, sf_q.data_ptr(), GMEM, assumed_align=16)
    w_ptr = make_ptr(F32, weights.data_ptr(), GMEM, assumed_align=16)
    clen_ptr = make_ptr(I32, context_lens.data_ptr(), GMEM, assumed_align=16)
    bt_ptr = make_ptr(I32, block_table.data_ptr(), GMEM, assumed_align=16)
    oi_ptr = make_ptr(I32, indices.data_ptr(), GMEM, assumed_align=16)
    ov_ptr = make_ptr(F32, values.data_ptr(), GMEM, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = _cache.get(key)
    if fn is None:
        fn = cute.compile(
            _launch,
            kv_ptr,
            q_ptr,
            sfq_ptr,
            w_ptr,
            clen_ptr,
            bt_ptr,
            oi_ptr,
            ov_ptr,
            stream,
            B,
            NCOMP,
            KTOP,
            MAXB,
            NPAGES,
            STAGES,
            CS,
            NLOC,
        )
        _cache[key] = fn
    fn(kv_ptr, q_ptr, sfq_ptr, w_ptr, clen_ptr, bt_ptr, oi_ptr, ov_ptr, stream)
