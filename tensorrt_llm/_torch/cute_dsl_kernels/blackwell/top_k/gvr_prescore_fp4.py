# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GVR prescore for the FP4 (MXFP4) indexer K cache.

One launch per layer step re-scores the previous step's top-k indices, their
neighbours and the newest positions on the current query: records are 64 B of
packed e2m1 plus 4 UE8M0 block scales (one per 32 dims), the query is fp4 with
its own 4 UE8M0 scales per head, and the MMA is the block-scaled tcgen05 kind
with both scale sets fed through TMEM (UTCCP). Scores go into the row's
4096-bin histogram (sign, a 64-binade exponent window and 6 mantissa bits of
the fp32 order key; two 16-bit counters per word, global atomics); the FP4
scorer derives the seed lines from it and the top-k resets it. The kernel
itself is the ping-pong variant in gvr_prescore_fp4_pp.py; this module holds
the geometry, the tile address chain and the per-tile MMA/epilogue steps.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float8E8M0FNU, Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05

from ..paged_mqa_logits.fp4_paged_mqa_logits import utccp_required_smem_warp_transpose
from ..utils import TRTLLM_ENABLE_PDL

TS = 128  # samples per CTA (= UMMA M)
# histogram digit: 32 binades of magnitude (2^-16 .. 2^16) x 6 mantissa bits (bin width
# 1.5%) mirrored around zero: negatives fill bins 2047..0 (larger magnitude -> lower bin),
# positives 2048..4095. Magnitudes below the window clamp to the two middle bins, above
# it to the two outer bins (bin 0 is treated as unbounded below).
MAG_SHIFT = 17
MAG_BASE = 111 << 6  # exponent field of 2^-16, in >> 17 units
MAG_BINS = 2048
NBINS = 2 * MAG_BINS
HIST_WORDS = NBINS // 2  # two 16-bit counters per word
NSTAGE = 2  # K-tile ring depth
NUM_THREADS = 128  # compute warps (gather, epilogue)


def _u32_bits(f):
    return Uint32(llvm.bitcast(Uint32.mlir_type, f.ir_value()))


class _GvrPrescoreBase:
    """Sample geometry and the per-tile address chain shared by the FP4 prescore kernels."""

    def __init__(
        self,
        top_k: int,
        num_heads: int,
        head_dim: int,
        tokens_per_block: int,
        record_bytes: int,
        tpp: int,
        recent: int = 0,
    ):
        self.TPP = tpp
        assert num_heads in (32, 64, 128, 256) and head_dim == 128
        assert (3 * top_k) % TS == 0 and recent % TS == 0
        assert 3 * top_k + recent < 65536  # 16-bit counters
        self.K = top_k
        self.W = recent  # newest positions sampled in addition to prev top-k +-1
        self.H = num_heads
        self.D = head_dim
        self.TPB = tokens_per_block
        self.BPT = record_bytes
        self.S = 3 * top_k + recent
        self.TILES = self.S // TS
        self.mma_tiler = (TS, num_heads, head_dim)
        assert self.TILES % self.TPP == 0

    @cute.jit
    def _addr_a(self, b, tile, tidx, mPrev: cute.Tensor, mFlags: cute.Tensor):
        """first half of a tile's address chain: the two independent loads"""
        K = cutlass.const_expr(self.K)
        s = tile * Int32(TS) + tidx
        c = s // Int32(K)
        i = s - c * Int32(K)
        pv = Int32(0)
        fl = Int32(0)
        if cutlass.const_expr(self.W > 0):
            if c >= Int32(3):
                c = Int32(3)
                i = s - Int32(3 * K)
            else:
                pv = mPrev[(b, i)]
                fl = Int32(mFlags[(b, i)])
        else:
            pv = mPrev[(b, i)]
            fl = Int32(mFlags[(b, i)])
        return c, i, pv, fl

    @cute.jit
    def _addr_b(self, b, last, c, i, pv, fl, mBt: cute.Tensor):
        """second half: position, duplicate rule, block-table load"""
        TPB = cutlass.const_expr(self.TPB)
        p0 = pv
        if p0 < Int32(0):
            p0 = Int32(0)
        if p0 > last:
            p0 = last
        pos = p0
        dupi = Int32(0)
        if i > Int32(0):
            if pv <= Int32(0):
                dupi = Int32(1)
            if pv >= last:
                dupi = Int32(1)
        if c == Int32(1):
            pos = p0 - Int32(1)
            if (fl & Int32(1)) != Int32(0):
                dupi = Int32(1)
            if p0 == Int32(0):
                dupi = Int32(1)
        if c == Int32(2):
            pos = p0 + Int32(1)
            if (fl & Int32(6)) != Int32(0):
                dupi = Int32(1)
            if p0 == last:
                dupi = Int32(1)
        if cutlass.const_expr(self.W > 0):
            # the newest W positions are sampled directly; prev entries inside that
            # window are duplicates of it
            if c == Int32(3):
                pos = last - Int32(self.W - 1) + i
                dupi = Int32(0)
                if pos < Int32(0):
                    dupi = Int32(1)
                    pos = Int32(0)
            else:
                if pos > last - Int32(self.W):
                    dupi = Int32(1)
        pblk = pos // Int32(TPB)
        soff = pos - pblk * Int32(TPB)
        blk = Int32(0)
        if dupi == Int32(0):
            blk = mBt[(b, pblk)]
        return dupi, blk, soff


ROW_BYTES = 64  # 128 fp4 values
SF_BYTES = 4  # 4 UE8M0 per token (32-dim groups)
SF_ATOM = 128  # tokens per UTCCP scale atom (int32 each)


class GvrPrescoreFp4Kernel(_GvrPrescoreBase):
    def __init__(
        self,
        top_k: int,
        num_heads: int,
        head_dim: int,
        tokens_per_block: int,
        record_bytes: int,
        tpp: int,
        num_acc: int = 2,
        recent: int = 0,
        pdl: bool = False,
    ):
        super().__init__(top_k, num_heads, head_dim, tokens_per_block, record_bytes, tpp, recent)
        self.pdl = pdl
        assert num_heads == 64 and head_dim == 128 and record_bytes == ROW_BYTES + SF_BYTES
        # two accumulators overlap MMA(it) with epilogue(it-1) (256 TMEM columns, two CTAs
        # per SM); one accumulator serialises them but fits 128 columns (four CTAs per SM)
        assert num_acc in (1, 2)
        self.NACC = num_acc
        self.NSFS = NSTAGE  # K-scale smem regions
        self.NSFT = NSTAGE  # K-scale TMEM regions
        # UTCCP source buffers: one per tile of the CTA. Reusing a buffer whose previous
        # UTCCP read has completed still hands the copy the old contents (the CTA barrier
        # does not order this warp's generic rewrite against its earlier async-proxy
        # read), so the buffers are never reused within a CTA.
        self.NUT = tpp
        self.mma_tiler = (TS, num_heads, head_dim)
        # TMEM: NACC fp32 accumulators (one column per head) + K scales per stage + Q scales
        self.SFA_COLS = (TS // 32) * 2
        self.SFB_COLS = (SF_ATOM // 32) * 2
        raw = num_acc * num_heads + max(32, self.NSFT * self.SFA_COLS + self.SFB_COLS) + 32
        self.TMEM_COLS = 32
        while self.TMEM_COLS < raw:
            self.TMEM_COLS *= 2
        assert self.TMEM_COLS <= 512

    def smem_bytes(self) -> int:
        return (
            NSTAGE * TS * ROW_BYTES  # K ring (also the histogram staging)
            + self.H * ROW_BYTES  # Q
            + 1024  # alignment slack
            + self.NSFS * SF_ATOM * 4  # K scales per stage
            + SF_ATOM * 4  # Q scales (padded to one atom)
            + (self.NUT + 1) * SF_ATOM * 4  # MMA warp's UTCCP source buffers
            + self.H * 4
            + 32 * 4
            + 64
        )

    @cute.jit
    def _gather(
        self, dupi, blk, soff, tidx, stage_base, sSFK: cute.Tensor, sf_off, kd_base, g2s, g2s4
    ):
        """cp.async of the tile's 128 fp4 records (64 B each, SW64 layout: 16 B chunk ch of
        row r lands at chunk ch ^ ((r >> 1) & 3)); four lanes share a row so a warp
        instruction moves eight whole records. The thread's own record scales (4 B) go
        to the flat scale array (0 for a duplicate)."""
        TPB = cutlass.const_expr(self.TPB)
        BPT = cutlass.const_expr(self.BPT)
        lane = tidx & Int32(31)
        ch = lane & Int32(3)
        rq = lane >> Int32(2)
        wbase = tidx - lane
        for r in cutlass.range_constexpr(4):
            src_lane = Int32(8 * r) + rq
            d_r = cute.arch.shuffle_sync(dupi, src_lane)
            blk_r = cute.arch.shuffle_sync(blk, src_lane)
            soff_r = cute.arch.shuffle_sync(soff, src_lane)
            row = wbase + src_lane
            if d_r == Int32(0):
                src = (
                    kd_base
                    + Int64(blk_r) * Int64(TPB * BPT)
                    + Int64(soff_r) * Int64(ROW_BYTES)
                    + Int64(ch) * Int64(16)
                )
                dst = (
                    stage_base
                    + Int64(row) * Int64(ROW_BYTES)
                    + Int64((ch ^ ((row >> Int32(1)) & Int32(3))) * Int32(16))
                )
                src_p = cute.make_ptr(Uint8, src, cute.AddressSpace.gmem, assumed_align=16)
                dst_p = cute.make_ptr(Uint8, dst, cute.AddressSpace.smem, assumed_align=16)
                cute.copy_atom_call(
                    g2s,
                    cute.make_tensor(src_p, cute.make_layout((16,))),
                    cute.make_tensor(dst_p, cute.make_layout((16,))),
                )
        if dupi == Int32(0):
            sf_src = cute.make_ptr(
                Int32,
                kd_base
                + Int64(blk) * Int64(TPB * BPT)
                + Int64(TPB * ROW_BYTES)
                + Int64(soff) * Int64(SF_BYTES),
                cute.AddressSpace.gmem,
                assumed_align=4,
            )
            cute.copy_atom_call(
                g2s4,
                cute.make_tensor(sf_src, cute.make_layout((1,))),
                cute.make_tensor(sSFK.iterator + (sf_off + tidx), cute.make_layout((1,))),
            )
        else:
            sSFK[sf_off + tidx] = Int32(0)
        return Float32(1.0)

    @cute.jit
    def _sf_to_tmem(
        self,
        sSF: cute.Tensor,
        sf_off,
        sUT: cute.Tensor,
        ut_off,
        chunk_layout,
        tCtSF: cute.Tensor,
        s2t_atom,
    ):
        """flat 128 x int32 scales (written by the compute warps) -> this warp's private
        UTCCP source buffer in chunk layout -> TMEM. The flat region is free for the next
        tile as soon as this warp passes the iteration barrier."""
        lane = cute.arch.lane_idx()
        for i in cutlass.range_constexpr(4):
            v = sSF[sf_off + Int32(i * 32) + lane]
            sUT[ut_off + Int32(i * 32) + lane] = v
        cute.arch.sync_warp()
        utccp_required_smem_warp_transpose(sUT.iterator + ut_off)
        cute.arch.fence_view_async_shared()
        s_ue8 = cute.recast_tensor(sUT, Float8E8M0FNU)
        chunk = cute.make_tensor(s_ue8.iterator + ut_off * Int32(4), chunk_layout)
        src_c = cute.filter_zeros(chunk)
        dst_c = cute.filter_zeros(tCtSF)
        tc = tcgen05.make_s2t_copy(s2t_atom, dst_c)
        thr = tc.get_slice(0)
        s_part = thr.partition_S(src_c)
        s_desc = tcgen05.get_s2t_smem_desc_tensor(tc, s_part)
        d_part = thr.partition_D(dst_c)
        cute.copy(tc, s_desc, d_part)

    @cute.jit
    def _mma(
        self,
        tiled_mma: cute.TiledMma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        tCtAcc: cute.Tensor,
        tCtSFA: cute.Tensor,
        tCtSFB: cute.Tensor,
        stage: cutlass.Constexpr[int],
        mbar,
    ):
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
            sfa_k = cute.slice_(tCtSFA, (None, None, kb, None))
            sfb_k = cute.slice_(tCtSFB, (None, None, kb, None))
            tiled_mma.set(tcgen05.Field.SFA, sfa_k.iterator)
            tiled_mma.set(tcgen05.Field.SFB, sfb_k.iterator)
            cute.gemm(
                tiled_mma, tCtAcc, tCrA[None, None, kb, stage], tCrB[None, None, kb, 0], tCtAcc
            )
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        with cute.arch.elect_one():
            tcgen05.commit(mbar)

    @cute.jit
    def _epilogue(
        self,
        b,
        tidx,
        tCtAcc: cute.Tensor,
        t2r_atom: cute.CopyAtom,
        sW: cute.Tensor,
        dupi,
        scale,
        mHist: cute.Tensor,
    ):
        """thread = one sample row: relu, head-weighted sum (scales already in the
        accumulator), magnitude-window digit -> the row's histogram"""
        H = cutlass.const_expr(self.H)
        tAcc = tCtAcc[((None, None), 0, 0)]
        tAcc_epi = cute.flat_divide(tAcc, (TS, H))
        tc = tcgen05.make_tmem_copy(t2r_atom, tAcc_epi[(None, None, 0, 0)])
        tr = tc.get_slice(tidx)
        tTR = tr.partition_S(tAcc_epi)
        cC = cute.make_identity_tensor((TS, H))
        tTR_cC = tr.partition_D(cC)
        tTR_rAcc = cute.make_fragment_like(tTR_cC, Float32)
        cute.copy(tc, tTR[(None, None, None, 0, 0)], tTR_rAcc)
        cute.arch.fence_view_async_tmem_load()
        acc_vec = tTR_rAcc.load()
        ssum = Float32(0.0)
        for h in cutlass.range_constexpr(H):
            ssum = ssum + cute.arch.fmax(acc_vec[h], Float32(0.0)) * sW[h]
        if dupi == Int32(0):
            u = _u32_bits(ssum)
            m = Int32((u & Uint32(0x7FFFFFFF)) >> Uint32(MAG_SHIFT)) - Int32(MAG_BASE)
            if m < Int32(0):
                m = Int32(0)
            if m > Int32(MAG_BINS - 1):
                m = Int32(MAG_BINS - 1)
            dig = Int32(MAG_BINS) + m
            if (u >> Uint32(31)) != Uint32(0):
                dig = Int32(MAG_BINS - 1) - m
            cute.arch.atomic_add(
                mHist.iterator + (b * Int32(HIST_WORDS) + (dig >> Int32(1))),
                Int32(1) << ((dig & Int32(1)) * Int32(16)),
                sem="relaxed",
                scope="gpu",
            )


def prescore_cute_fp4(
    prev,
    lens,
    flags,
    block_table,
    kv_bytes,
    q_bytes,
    q_sf,
    weights,
    hist,
    seed_row,
    cand_ctl,
    cand_cur,
    *,
    top_k: int,
    num_heads: int,
    head_dim: int,
    tokens_per_block: int,
    record_bytes: int,
    recent: int = 0,
    tpp: int = 0,
    pdl: bool = TRTLLM_ENABLE_PDL,
) -> None:
    from .gvr_prescore_fp4_pp import prescore_cute_fp4_pp

    prescore_cute_fp4_pp(
        prev,
        lens,
        flags,
        block_table,
        kv_bytes,
        q_bytes,
        q_sf,
        weights,
        hist,
        seed_row,
        cand_ctl,
        cand_cur,
        top_k=top_k,
        num_heads=num_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        record_bytes=record_bytes,
        recent=recent,
        tpp=tpp,
        pdl=pdl,
    )


def recent_for(top_k: int, recent: int) -> int:
    """Round the recent window up so the row's tile count is a multiple of 8."""
    if recent <= 0:
        return 0
    tiles = -(-(3 * top_k + recent) // TS)
    while tiles % 8:
        tiles += 1
    return tiles * TS - 3 * top_k
