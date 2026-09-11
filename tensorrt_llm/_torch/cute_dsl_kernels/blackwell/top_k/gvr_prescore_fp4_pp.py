# SPDX-License-Identifier: Apache-2.0
"""Ping-pong FP4 prescore kernel: two compute warpgroups own alternate tiles, each with
a two-stage K ring, scale regions, an accumulator and an mbarrier; the MMA warp serves
them in tile order through per-group named barriers."""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Float4E2M1FN, Float8E8M0FNU, Float32, Int32, Int64, Uint8
from cutlass.cute.nvgpu import cpasync, tcgen05

from ..utils import griddepcontrol_launch_dependents, griddepcontrol_wait
from .gvr_prescore_fp4 import (
    HIST_WORDS,
    NSTAGE,
    NUM_THREADS,
    ROW_BYTES,
    SF_ATOM,
    TS,
    GvrPrescoreFp4Kernel,
)

PP_MMA_WARP = 8
PP_STAGES = 4  # two K stages per compute group
PP_THREADS = 2 * NUM_THREADS + 32
BAR_FULL0, BAR_FULL1, BAR_END = 4, 5, 6


class GvrPrescoreFp4PPKernel(GvrPrescoreFp4Kernel):
    def __init__(
        self,
        top_k,
        num_heads,
        head_dim,
        tokens_per_block,
        record_bytes,
        tpp,
        num_acc=2,
        recent=0,
        pdl=False,
    ):
        super().__init__(
            top_k, num_heads, head_dim, tokens_per_block, record_bytes, tpp, num_acc, recent, pdl
        )
        assert tpp % 2 == 0

    def smem_bytes(self) -> int:
        return super().smem_bytes() + (PP_STAGES - NSTAGE) * TS * ROW_BYTES + 2 * SF_ATOM * 4

    @cute.jit
    def __call__(
        self,
        mPrev,
        mLens,
        mFlags,
        mBt,
        mKdata,
        mQ,
        mQsf,
        mW,
        mHist,
        mSeed,
        mCtl,
        mCur,
        stream,
        min_blocks_per_mp: cutlass.Constexpr[int] = 1,
    ):
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            Float4E2M1FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float8E8M0FNU,
            32,
            tcgen05.CtaGroup.ONE,
            (TS, self.H),
        )
        a_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, Float4E2M1FN, PP_STAGES
        )
        b_layout = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, Float4E2M1FN, 1)
        sfa_chunk = blockscaled_utils.make_smem_layout_sfa(tiled_mma, self.mma_tiler, 32, 1)
        sfb_chunk = blockscaled_utils.make_smem_layout_sfb(tiled_mma, self.mma_tiler, 32, 1)
        tmem_sfa = cute.append(
            blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma, self.mma_tiler, 32, cute.slice_(sfa_chunk, (None, None, None, 0))
            ),
            cute.make_layout(1, stride=0),
        )
        tmem_sfb = cute.append(
            blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma, self.mma_tiler, 32, cute.slice_(sfb_chunk, (None, None, None, 0))
            ),
            cute.make_layout(1, stride=0),
        )
        t2r_atom = sm100_utils.get_tmem_load_op(
            self.mma_tiler, utils.LayoutEnum.ROW_MAJOR, Float32, Float32, (TS, self.H), False
        )
        s2t_atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), Float8E8M0FNU)
        self.kernel(
            tiled_mma,
            a_layout,
            b_layout,
            sfa_chunk,
            sfb_chunk,
            tmem_sfa,
            tmem_sfb,
            t2r_atom,
            s2t_atom,
            mPrev,
            mLens,
            mFlags,
            mBt,
            mKdata,
            mQ,
            mQsf,
            mW,
            mHist,
            mSeed,
            mCtl,
            mCur,
        ).launch(
            grid=[mPrev.shape[0] * (self.TILES // self.TPP), 1, 1],
            block=[PP_THREADS, 1, 1],
            smem=self.smem_bytes(),
            stream=stream,
            min_blocks_per_mp=min_blocks_per_mp,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_chunk: cute.Layout,
        sfb_chunk: cute.Layout,
        tmem_sfa: cute.Layout,
        tmem_sfb: cute.Layout,
        t2r_atom: cute.CopyAtom,
        s2t_atom: cute.CopyAtom,
        mPrev: cute.Tensor,
        mLens: cute.Tensor,
        mFlags: cute.Tensor,
        mBt: cute.Tensor,
        mKdata: cute.Tensor,
        mQ: cute.Tensor,
        mQsf: cute.Tensor,
        mW: cute.Tensor,
        mHist: cute.Tensor,
        mSeed: cute.Tensor,
        mCtl: cute.Tensor,
        mCur: cute.Tensor,
    ):
        H = cutlass.const_expr(self.H)
        TPP = cutlass.const_expr(self.TPP)
        NT = cutlass.const_expr(self.TPP // 2)  # tiles per compute group
        CPR = cutlass.const_expr(self.TILES // self.TPP)

        tidx = cute.arch.thread_idx()[0]
        bid = cute.arch.block_idx()[0]
        b = bid // Int32(CPR)
        t0 = (bid - b * Int32(CPR)) * Int32(TPP)
        wid = tidx // Int32(32)
        grp = wid // Int32(4)
        tl = tidx - grp * Int32(NUM_THREADS)

        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(
            Float4E2M1FN, a_layout.outer, byte_alignment=1024, swizzle=a_layout.inner
        )
        sQ = smem.allocate_tensor(
            Float4E2M1FN, b_layout.outer, byte_alignment=1024, swizzle=b_layout.inner
        )
        sSFK = smem.allocate_tensor(
            Int32, cute.make_layout((PP_STAGES * SF_ATOM,)), byte_alignment=512
        )
        sSFQ = smem.allocate_tensor(Int32, cute.make_layout((SF_ATOM,)), byte_alignment=512)
        NUT = cutlass.const_expr(self.NUT)
        sUT = smem.allocate_tensor(
            Int32, cute.make_layout(((NUT + 1) * SF_ATOM,)), byte_alignment=512
        )
        sW = smem.allocate_tensor(Float32, cute.make_layout((H,)), byte_alignment=16)
        mbar = smem.allocate_array(Int64, 2, byte_alignment=8)
        tmem_holding = smem.allocate_array(Int32, 1, byte_alignment=4)
        sK_base = cute.recast_ptr(sK.iterator, swizzle_=None).toint()
        sQ_base = cute.recast_ptr(sQ.iterator, swizzle_=None).toint()

        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=PP_THREADS)
        tmem = utils.TmemAllocator(
            tmem_holding,
            barrier_for_retrieve=tmem_bar,
            allocator_warp_id=PP_MMA_WARP,
            is_two_cta=False,
        )
        acc_shape = tiled_mma.partition_shape_C((TS, H))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        acc_cols = cutlass.const_expr(self.H)
        SFA_COLS = cutlass.const_expr(self.SFA_COLS)
        tmem.allocate(cutlass.const_expr(self.TMEM_COLS))
        if tidx == Int32(0):
            cute.arch.mbarrier_init(mbar, 1)
            cute.arch.mbarrier_init(mbar + 1, 1)
            cute.arch.mbarrier_init_fence()

        g2s = cute.make_copy_atom(cpasync.CopyG2SOp(), Uint8, num_bits_per_copy=128)
        g2s4 = cute.make_copy_atom(cpasync.CopyG2SOp(), Int32, num_bits_per_copy=32)
        kd_base = mKdata.iterator.toint()
        last = mLens[b] - Int32(1)

        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        sfa_base = 2 * acc_cols
        tCtSFB = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + sfa_base + 2 * SFA_COLS, dtype=Float8E8M0FNU), tmem_sfb
        )

        if wid == Int32(PP_MMA_WARP):
            tmem.relinquish_alloc_permit()
            tCrA = tiled_mma.make_fragment_A(sK)
            tCrB = tiled_mma.make_fragment_B(sQ)
            accs = [cute.make_tensor(tmem_ptr + i * acc_cols, tCtAcc_fake.layout) for i in range(2)]
            sfas = [
                cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + sfa_base + i * SFA_COLS, dtype=Float8E8M0FNU),
                    tmem_sfa,
                )
                for i in range(2)
            ]
            cute.arch.sync_threads()  # tiles 0/1, Q, weights and scales landed
            self._sf_to_tmem(sSFQ, Int32(0), sUT, Int32(NUT * SF_ATOM), sfb_chunk, tCtSFB, s2t_atom)
            for it in cutlass.range_constexpr(TPP):
                g = it % 2
                if it >= 2:
                    if g == 0:
                        cute.arch.barrier(barrier_id=BAR_FULL0, number_of_threads=NUM_THREADS + 32)
                    else:
                        cute.arch.barrier(barrier_id=BAR_FULL1, number_of_threads=NUM_THREADS + 32)
                st = g * 2 + (it // 2) % 2
                self._sf_to_tmem(
                    sSFK,
                    Int32(st * SF_ATOM),
                    sUT,
                    Int32(it * SF_ATOM),
                    sfa_chunk,
                    sfas[g],
                    s2t_atom,
                )
                self._mma(tiled_mma, tCrA, tCrB, accs[g], sfas[g], tCtSFB, st, mbar + g)
            cute.arch.barrier(barrier_id=BAR_END, number_of_threads=PP_THREADS)
            tmem.free(tmem_ptr)
        else:
            acc = cute.make_tensor(tmem_ptr + grp * Int32(acc_cols), tCtAcc_fake.layout)
            stage0 = sK_base + Int64(grp) * Int64(2 * TS * ROW_BYTES)
            sf0 = grp * Int32(2 * SF_ATOM)
            ca = [None] * NT
            dup = [None] * NT
            blk = [None] * NT
            soff = [None] * NT
            for k in cutlass.range_constexpr(NT):
                ca[k] = self._addr_a(b, t0 + Int32(2 * k) + grp, tl, mPrev, mFlags)
            for k in cutlass.range_constexpr(NT):
                ck, ik, pvk, flk = ca[k]
                dup[k], blk[k], soff[k] = self._addr_b(b, last, ck, ik, pvk, flk, mBt)
            if cutlass.const_expr(self.pdl):
                griddepcontrol_wait()
            # the tile-0 CTA resets the scorer's accumulation slots (seed-row
            # counts, list control, cursors): ordered after the previous consumer
            # (wait above) and before the scorer's first accumulate (its own
            # wait on this grid)
            if t0 == Int32(0) and grp == Int32(0):
                if tl < Int32(5):
                    mSeed[(b, Int32(3) + tl)] = Float32(0.0)
                if tl < Int32(4):
                    mCtl[(b, tl)] = Int32(0)
                    mCur[(b, tl)] = Int32(0)
            if grp == Int32(0):
                q_base = mQ.iterator.toint() + Int64(b) * Int64(H * ROW_BYTES)
                for j in cutlass.range_constexpr((H * ROW_BYTES // 16) // NUM_THREADS):
                    cid = tl + Int32(NUM_THREADS * j)
                    hh = cid // Int32(ROW_BYTES // 16)
                    ch = cid - hh * Int32(ROW_BYTES // 16)
                    src_p = cute.make_ptr(
                        Uint8,
                        q_base + Int64(hh) * Int64(ROW_BYTES) + Int64(ch) * Int64(16),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    dst_p = cute.make_ptr(
                        Uint8,
                        sQ_base
                        + Int64(hh) * Int64(ROW_BYTES)
                        + Int64((ch ^ ((hh >> Int32(1)) & Int32(3))) * Int32(16)),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    )
                    cute.copy_atom_call(
                        g2s,
                        cute.make_tensor(src_p, cute.make_layout((16,))),
                        cute.make_tensor(dst_p, cute.make_layout((16,))),
                    )
                if tl < Int32(H):
                    sW[tl] = mW[(b, tl)]
                    sSFQ[tl] = mQsf[(b, tl)]
                else:
                    sSFQ[tl] = Int32(0)
                cute.arch.cp_async_commit_group()
            self._gather(dup[0], blk[0], soff[0], tl, stage0, sSFK, sf0, kd_base, g2s, g2s4)
            cute.arch.cp_async_commit_group()
            if cutlass.const_expr(NT > 1):
                self._gather(
                    dup[1],
                    blk[1],
                    soff[1],
                    tl,
                    stage0 + Int64(TS * ROW_BYTES),
                    sSFK,
                    sf0 + Int32(SF_ATOM),
                    kd_base,
                    g2s,
                    g2s4,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(1)
            else:
                cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()
            for k in cutlass.range_constexpr(NT):
                cute.arch.mbarrier_wait(mbar + grp, k & 1)  # MMA(k) done: its stage and acc free
                if cutlass.const_expr(k + 2 < NT):
                    self._gather(
                        dup[k + 2],
                        blk[k + 2],
                        soff[k + 2],
                        tl,
                        stage0 + Int64((k % 2) * TS * ROW_BYTES),
                        sSFK,
                        sf0 + Int32((k % 2) * SF_ATOM),
                        kd_base,
                        g2s,
                        g2s4,
                    )
                    cute.arch.cp_async_commit_group()
                self._epilogue(b, tl, acc, t2r_atom, sW, dup[k], Float32(1.0), mHist)
                if cutlass.const_expr(k + 1 < NT):
                    if cutlass.const_expr(k + 2 < NT):
                        cute.arch.cp_async_wait_group(1)
                    else:
                        cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    if grp == Int32(0):
                        cute.arch.barrier_arrive(
                            barrier_id=BAR_FULL0, number_of_threads=NUM_THREADS + 32
                        )
                    else:
                        cute.arch.barrier_arrive(
                            barrier_id=BAR_FULL1, number_of_threads=NUM_THREADS + 32
                        )
            if cutlass.const_expr(self.pdl):
                griddepcontrol_launch_dependents()
            cute.arch.barrier_arrive(barrier_id=BAR_END, number_of_threads=PP_THREADS)


_COMPILE_CACHE = {}
_MAX_CTAS = {}


def _compile(
    top_k,
    num_heads,
    head_dim,
    tokens_per_block,
    record_bytes,
    tpp,
    recent,
    pdl,
):
    key = (top_k, num_heads, head_dim, tokens_per_block, record_bytes, tpp, recent, pdl)
    fn = _COMPILE_CACHE.get(key)
    if fn is not None:
        return fn
    rows = cute.sym_int()
    mk = cute.runtime.make_fake_compact_tensor
    fakes = (
        mk(Int32, (rows, top_k), stride_order=(1, 0), assumed_align=16),
        mk(Int32, (rows,), stride_order=(0,), assumed_align=4),
        mk(Uint8, (rows, top_k), stride_order=(1, 0), assumed_align=16),
        mk(Int32, (rows, cute.sym_int()), stride_order=(1, 0), assumed_align=4),
        mk(Uint8, (cute.sym_int(),), stride_order=(0,), assumed_align=16),
        mk(Uint8, (cute.sym_int(),), stride_order=(0,), assumed_align=16),
        mk(Int32, (rows, num_heads), stride_order=(1, 0), assumed_align=16),
        mk(Float32, (rows, num_heads), stride_order=(1, 0), assumed_align=16),
        mk(Int32, (rows, HIST_WORDS), stride_order=(1, 0), assumed_align=16),
        mk(Float32, (rows, 8), stride_order=(1, 0), assumed_align=16),
        mk(Int32, (rows, 4), stride_order=(1, 0), assumed_align=16),
        mk(Int32, (rows, 4), stride_order=(1, 0), assumed_align=16),
    )
    kern = GvrPrescoreFp4PPKernel(
        top_k,
        num_heads,
        head_dim,
        tokens_per_block,
        record_bytes,
        tpp,
        2,
        recent,
        pdl,
    )
    fn = cute.compile(
        kern,
        *fakes,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        min_blocks_per_mp=2,
        options="--enable-tvm-ffi",
    )
    _COMPILE_CACHE[key] = fn
    return fn


def prescore_cute_fp4_pp(
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
    top_k,
    num_heads,
    head_dim,
    tokens_per_block,
    record_bytes,
    recent=0,
    tpp=0,
    pdl=False,
):
    import torch

    tiles = (3 * top_k + recent) // TS
    rows = prev.shape[0]
    cap = _MAX_CTAS.get(prev.device)
    if cap is None:
        cap = 2 * torch.cuda.get_device_properties(prev.device).multi_processor_count
        _MAX_CTAS[prev.device] = cap
    if tpp <= 0:
        # one wave of CTAs (two per SM), at most eight tiles per compute group
        cands = [c for c in range(2, min(16, tiles) + 1, 2) if tiles % c == 0]
        fits = [c for c in cands if rows * tiles // c <= cap]
        tpp = fits[0] if fits else cands[-1]
    assert tiles % tpp == 0 and tpp % 2 == 0
    fn = _compile(
        top_k,
        num_heads,
        head_dim,
        tokens_per_block,
        record_bytes,
        tpp,
        recent,
        pdl,
    )
    fn(
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
    )
