# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The row-tile pass of `qsa_sparse_gqa_rowtile` on SM103 tensor cores (CuTe DSL, tcgen05).

Inputs and output: per group of `rows_per_tile` rows of one request, the plan's packed
slot list (`slots[group]`, `listed` entries), each row's bitmap re-indexed onto it
(`packed_bits`, as int32 halves) and `group_plan[group] = (tiles, listed)`; groups with
`tiles < 0` belong to the lone-row pass.

One CTA per (group, KV head), 128 MMA rows = `rows_per_tile` query rows x `group_size`
heads, 9 warps:

* warps 5-8 gather the list's K and V rows (64 a tile) with `cp.async` into one ring of
  `kv_slots` 32 KB SW128 shared-memory slots, filled in the MMA warp's order K0, K1, V0,
  K2, V1, ...; each slot completes on its mbarrier through `cp.async.mbarrier.arrive`;
* warp 4 issues the tcgen05 MMAs: S[j % 2] = Q K(j)^T (Q from tensor memory), then
  O += P(j - 1) V(j - 1) (P from tensor memory, overwriting the S it came from);
* warps 0-3, one thread per MMA row, first copy their Q row into tensor memory, then per
  tile load S with its row max (`tcgen05.ld.red`), rescale O when that max grew by more
  than `rescale_threshold` in log2 units (lazily; otherwise exponentiate against the stale
  max), write P = 2^(s - max) on the row's own bits and 0 elsewhere, and at the end
  normalize O and store it through the idle ring as whole 512-byte rows.

Tensor memory: O in columns 0-255, S/P buffers at 256 and 320, Q at 384 (512 columns).
The tile's row max covers every listed entry, an upper bound of the row's own: a valid
softmax offset. P is rounded to bf16 for the PV MMA and summed in fp32 unrounded.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass._mlir.dialects import nvvm as _nvvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync
from cutlass.cute.runtime import make_ptr
from cutlass.cute.typing import Float32, Int32, Int64
from cutlass.cutlass_dsl import dsl_user_op

try:
    from cutlass.memory import SmemAllocator
except ImportError:  # DSL releases before the memory namespace split
    from cutlass.utils import SmemAllocator

BF16 = cutlass.BFloat16
_LOG2_E = 1.4426950408889634
#: The head dimension the tensor-memory plan is laid out for.
HEAD_DIM = 256
#: MMA rows per CTA and tokens per tile (one 64-bit bitmap word).
BLOCK_M = 128
BLOCK_N = 64


@dsl_user_op
def tcgen05_fence_before(*, loc=None, ip=None):
    """tcgen05.fence::before_thread_sync: order this thread's prior tcgen05 ops before a thread sync."""
    _nvvm.tcgen05_fence(_nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC, loc=loc, ip=ip)


@dsl_user_op
def tcgen05_fence_after(*, loc=None, ip=None):
    """tcgen05.fence::after_thread_sync: order later tcgen05 ops after a thread sync."""
    _nvvm.tcgen05_fence(_nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC, loc=loc, ip=ip)


class _RowtileAttentionKernel:
    def __init__(
        self,
        group_size: int,
        rows_per_tile: int,
        head_dim: int = 256,
        tokens_per_block: int = 32,
        kv_slots: int = 6,
        rescale_threshold: float = 8.0,
        use_pdl: bool = True,
    ):
        assert head_dim == 256, (
            "the TMEM plan (O 256 + 2 x S 64 + Q 128 columns) is laid out for D = 256"
        )
        assert group_size * rows_per_tile <= 128
        self.group_size = group_size
        self.rows_per_tile = rows_per_tile
        self.d = head_dim
        self.tpb = tokens_per_block
        self.bm = 128
        self.bn = 64
        self.kv_slots = kv_slots
        self.tau = rescale_threshold
        self.use_pdl = use_pdl
        self.mma_warp = 4
        self.load_warp0 = 5
        self.threads = 32 * 9
        self.tmem_o = 0
        self.tmem_s0 = 256
        self.tmem_q = 384  # Q as the QK MMA's A operand: 256 bf16 a row, two a column
        self.tmem_cols = 512
        self.chunk_cols = 32  # O columns a TMEM load/store moves per thread

    @cute.jit
    def __call__(
        self,
        q: cute.Pointer,
        k: cute.Pointer,
        v: cute.Pointer,
        slots: cute.Pointer,
        bits: cute.Pointer,
        plan: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        num_groups: Int32,
        num_kv_heads: Int32,
        q_stride_row: Int32,
        k_stride_page: Int32,
        k_stride_head: Int32,
        k_stride_token: Int32,
        v_stride_page: Int32,
        v_stride_head: Int32,
        v_stride_token: Int32,
        slots_stride_group: Int32,
        bits_stride_row: Int32,
        out_stride_row: Int32,
        scale_log2: Float32,
        stream,
    ):
        def qk():
            return sm100_utils.make_trivial_tiled_mma(
                BF16,
                BF16,
                OperandMajorMode.K,
                OperandMajorMode.K,
                Float32,
                tcgen05.CtaGroup.ONE,
                (self.bm, self.bn),
                tcgen05.OperandSource.TMEM,
            )

        def pv():
            return sm100_utils.make_trivial_tiled_mma(
                BF16,
                BF16,
                OperandMajorMode.K,
                OperandMajorMode.MN,
                Float32,
                tcgen05.CtaGroup.ONE,
                (self.bm, self.d),
                tcgen05.OperandSource.TMEM,
            )

        # Overwrite / accumulate twins of each MMA: the ACCUMULATE field is set once at kernel
        # scope, so no loop or branch rebinds an MMA value.
        qk_mma, qk_mma_acc, pv_mma, pv_mma_acc = qk(), qk(), pv(), pv()
        q_layout = sm100_utils.make_smem_layout_a(qk_mma, (self.bm, self.bn, self.d), BF16, 1)
        k_layout = sm100_utils.make_smem_layout_b(
            qk_mma, (self.bm, self.bn, self.d), BF16, self.kv_slots
        )
        v_layout = sm100_utils.make_smem_layout_b(
            pv_mma, (self.bm, self.d, self.bn), BF16, self.kv_slots
        )
        p_layout = sm100_utils.make_smem_layout_a(pv_mma, (self.bm, self.d, self.bn), BF16, 1)

        @cute.struct
        class SharedStorage:
            q_full: cute.struct.MemRange[Int64, 1]
            kv_full: cute.struct.MemRange[Int64, self.kv_slots]
            kv_empty: cute.struct.MemRange[Int64, self.kv_slots]
            s_full: cute.struct.MemRange[Int64, 2]
            p_full: cute.struct.MemRange[Int64, 2]
            o_full: cute.struct.MemRange[Int64, 1]
            o_final: cute.struct.MemRange[Int64, 1]
            tmem_holding: Int32
            sKV: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(k_layout)], 1024]

        self.shared_storage = SharedStorage

        self.kernel(
            qk_mma,
            qk_mma_acc,
            pv_mma,
            pv_mma_acc,
            q_layout,
            k_layout,
            v_layout,
            p_layout,
            q,
            k,
            v,
            slots,
            bits,
            plan,
            out,
            rows,
            q_stride_row,
            k_stride_page,
            k_stride_head,
            k_stride_token,
            v_stride_page,
            v_stride_head,
            v_stride_token,
            slots_stride_group,
            bits_stride_row,
            out_stride_row,
            scale_log2,
        ).launch(
            grid=[num_groups, num_kv_heads, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        qk_mma: cute.TiledMma,
        qk_mma_acc: cute.TiledMma,
        pv_mma: cute.TiledMma,
        pv_mma_acc: cute.TiledMma,
        q_layout: cute.ComposedLayout,
        k_layout: cute.ComposedLayout,
        v_layout: cute.ComposedLayout,
        p_layout: cute.ComposedLayout,
        q: cute.Pointer,
        k: cute.Pointer,
        v: cute.Pointer,
        slots: cute.Pointer,
        bits: cute.Pointer,
        plan: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        q_stride_row: Int32,
        k_stride_page: Int32,
        k_stride_head: Int32,
        k_stride_token: Int32,
        v_stride_page: Int32,
        v_stride_head: Int32,
        v_stride_token: Int32,
        slots_stride_group: Int32,
        bits_stride_row: Int32,
        out_stride_row: Int32,
        scale_log2: Float32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        group, kv_head, _ = cute.arch.block_idx()
        qk_mma.set(tcgen05.Field.ACCUMULATE, False)
        qk_mma_acc.set(tcgen05.Field.ACCUMULATE, True)
        pv_mma.set(tcgen05.Field.ACCUMULATE, False)
        pv_mma_acc.set(tcgen05.Field.ACCUMULATE, True)

        mPlan = cute.make_tensor(plan, cute.make_layout((2 * 1048576,)))
        tiles = mPlan[group * 2]
        listed = mPlan[group * 2 + 1]

        if tiles >= 0:
            smem = SmemAllocator()
            storage = smem.allocate(self.shared_storage)
            q_full = storage.q_full.data_ptr()
            kv_full = storage.kv_full.data_ptr()
            kv_empty = storage.kv_empty.data_ptr()
            s_full = storage.s_full.data_ptr()
            p_full = storage.p_full.data_ptr()
            o_full = storage.o_full.data_ptr()
            o_final = storage.o_final.data_ptr()
            if tidx == 0:
                cute.arch.mbarrier_init(q_full, 128)
                for i in cutlass.range_constexpr(self.kv_slots):
                    cute.arch.mbarrier_init(kv_full + i, 128)
                    cute.arch.mbarrier_init(kv_empty + i, 1)
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(s_full + i, 1)
                    cute.arch.mbarrier_init(p_full + i, 128)
                cute.arch.mbarrier_init(o_full, 1)
                cute.arch.mbarrier_init(o_final, 1)
            cute.arch.mbarrier_init_fence()

            tmem_users = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * 5)
            if warp_idx == self.mma_warp:
                cute.arch.alloc_tmem(self.tmem_cols, storage.tmem_holding.ptr)
                cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.sync_threads()

            # Slices taken before the role branches: the MMA warp's `set(ACCUMULATE)` rebinds the
            # tiled MMAs inside its branch, which the other branches must not see.
            qk_thr = qk_mma.get_slice(0)
            pv_thr = pv_mma.get_slice(0)
            sK = storage.sKV.get_tensor(k_layout.outer, swizzle=k_layout.inner)
            sV = cute.make_tensor(cute.recast_ptr(sK.iterator, v_layout.inner), v_layout.outer)

            if warp_idx >= self.load_warp0:
                self.load_role(
                    tidx - 32 * self.load_warp0,
                    group,
                    kv_head,
                    tiles,
                    listed,
                    sK,
                    k,
                    v,
                    slots,
                    k_stride_page,
                    k_stride_head,
                    k_stride_token,
                    v_stride_page,
                    v_stride_head,
                    v_stride_token,
                    slots_stride_group,
                    kv_full,
                    kv_empty,
                )
            elif warp_idx == self.mma_warp:
                tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, storage.tmem_holding.ptr)
                self.mma_role(
                    qk_mma,
                    qk_mma_acc,
                    pv_mma,
                    pv_mma_acc,
                    qk_thr,
                    pv_thr,
                    q_layout,
                    p_layout,
                    sK,
                    sV,
                    tmem_ptr,
                    tiles,
                    q_full,
                    kv_full,
                    kv_empty,
                    s_full,
                    p_full,
                    o_full,
                    o_final,
                )
                tmem_users.arrive_and_wait()
                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_cols)
            else:
                tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, storage.tmem_holding.ptr)
                self.softmax_role(
                    qk_thr,
                    pv_thr,
                    tmem_ptr,
                    tidx,
                    group,
                    kv_head,
                    tiles,
                    rows,
                    bits,
                    bits_stride_row,
                    out,
                    out_stride_row,
                    scale_log2,
                    q_full,
                    s_full,
                    p_full,
                    o_full,
                    o_final,
                    storage.sKV.data_ptr(),
                    q,
                    q_stride_row,
                )
                tmem_users.arrive_and_wait()

        if cutlass.const_expr(self.use_pdl):
            # The lone-row pass may still run: finish only after it has, so a later launch sees every row.
            cute.arch.griddepcontrol_wait()

    # ------------------------------------------------------------------ loaders (warps 5-8)
    @cute.jit
    def load_role(
        self,
        lt: Int32,
        group: Int32,
        kv_head: Int32,
        tiles: Int32,
        listed: Int32,
        sK,
        k: cute.Pointer,
        v: cute.Pointer,
        slots: cute.Pointer,
        k_stride_page: Int32,
        k_stride_head: Int32,
        k_stride_token: Int32,
        v_stride_page: Int32,
        v_stride_head: Int32,
        v_stride_token: Int32,
        slots_stride_group: Int32,
        kv_full,
        kv_empty,
    ):
        """Fill the K/V ring in the MMA warp's order K0, K1, V0, K2, V1, ..., V(T-1). Thread `lt`
        gathers tokens 16 (lt // 32) + 4 t + (lt % 32) // 8 (t < 4), one 16-byte chunk (lt % 8) of
        each of their four 128-byte dim blocks; the slots of tile j + 1 are fetched while tile j's
        copies issue."""
        D = self.d
        atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BF16, num_bits_per_copy=128
        )
        chunk = lt % 8
        # Element-coordinate view of the swizzled ring: (8 elements, (chunk, 64-dim block), token, slot).
        sKV_c = cute.make_tensor(
            sK.iterator,
            cute.make_layout(
                (8, (8, D // 64), self.bn, self.kv_slots),
                stride=(1, (8, self.bn * 64), 64, self.bn * D),
            ),
        )
        if tiles > 0:
            mSlots = cute.make_tensor(
                slots + Int64(group) * slots_stride_group, cute.make_layout((1 << 30,))
            )
            first_token = 16 * (lt // 32) + (lt % 32) // 8
            slot_next = cute.make_rmem_tensor((4,), Int32)
            k_cur = cute.make_rmem_tensor((4,), Int64)
            v_cur = cute.make_rmem_tensor((4,), Int64)
            v_prev = cute.make_rmem_tensor((4,), Int64)
            for t in cutlass.range_constexpr(4):
                slot_next[t] = mSlots[cutlass.min(first_token + 4 * t, listed - 1)]
            ring = Int32(0)
            ring_phase = Int32(1)
            for j in cutlass.range(0, tiles, 1, unroll=1):
                for t in cutlass.range_constexpr(4):
                    slot = slot_next[t]
                    page = slot // self.tpb
                    tok = slot % self.tpb
                    k_cur[t] = (
                        Int64(page) * k_stride_page + kv_head * k_stride_head + tok * k_stride_token
                    )
                    v_cur[t] = (
                        Int64(page) * v_stride_page + kv_head * v_stride_head + tok * v_stride_token
                    )
                if j + 1 < tiles:
                    for t in cutlass.range_constexpr(4):
                        slot_next[t] = mSlots[
                            cutlass.min((j + 1) * self.bn + first_token + 4 * t, listed - 1)
                        ]
                # K(j)
                cute.arch.mbarrier_wait(kv_empty + ring, ring_phase)
                self.issue_tile(atom, k, k_cur, sKV_c, ring, chunk, first_token)
                cute.arch.cp_async_mbarrier_arrive_noinc(kv_full + ring)
                ring, ring_phase = self.advance(ring, ring_phase)
                if j > 0:
                    # V(j - 1)
                    cute.arch.mbarrier_wait(kv_empty + ring, ring_phase)
                    self.issue_tile(atom, v, v_prev, sKV_c, ring, chunk, first_token)
                    cute.arch.cp_async_mbarrier_arrive_noinc(kv_full + ring)
                    ring, ring_phase = self.advance(ring, ring_phase)
                for t in cutlass.range_constexpr(4):
                    v_prev[t] = v_cur[t]
            # V(tiles - 1)
            cute.arch.mbarrier_wait(kv_empty + ring, ring_phase)
            self.issue_tile(atom, v, v_prev, sKV_c, ring, chunk, first_token)
            cute.arch.cp_async_mbarrier_arrive_noinc(kv_full + ring)
            ring, ring_phase = self.advance(ring, ring_phase)
            # Tail: every slot released, so no load is in flight when the CTA retires and the ring
            # is free for the epilogue's staging.
            for i in cutlass.range_constexpr(self.kv_slots):
                cute.arch.mbarrier_wait(kv_empty + ring, ring_phase)
                ring, ring_phase = self.advance(ring, ring_phase)

    @cute.jit
    def issue_tile(
        self,
        atom,
        src_base: cute.Pointer,
        bases,
        sKV_c,
        ring: Int32,
        chunk: Int32,
        first_token: Int32,
    ):
        for t in cutlass.range_constexpr(4):
            n = first_token + 4 * t
            for b in cutlass.range_constexpr(self.d // 64):
                offset = cute.assume(bases[t] + b * 64 + chunk * 8, divby=8)
                src = cute.make_tensor(src_base + offset, cute.make_layout((8,)))
                dst = sKV_c[(None, (chunk, b), n, ring)]
                cute.copy(atom, src, dst)

    # ------------------------------------------------------------------ MMA (warp 4)
    @cute.jit
    def mma_role(
        self,
        qk_mma: cute.TiledMma,
        qk_mma_acc: cute.TiledMma,
        pv_mma: cute.TiledMma,
        pv_mma_acc: cute.TiledMma,
        qk_thr,
        pv_thr,
        q_layout,
        p_layout,
        sK,
        sV,
        tmem_ptr,
        tiles: Int32,
        q_full,
        kv_full,
        kv_empty,
        s_full,
        p_full,
        o_full,
        o_final,
    ):
        """QK(0), then QK(j) / PV(j - 1) pairs, then the last PV. S/P buffer b sits at TMEM columns
        tmem_s0 + 64 b and is picked by pointer offset; QK(j) overwrites S[j % 2] only after PV(j - 2)
        was issued (tcgen05 MMAs of one thread execute in order), and P(j - 2) was its last reader."""
        tQ = cute.make_tensor(tmem_ptr + self.tmem_q, q_layout.outer)
        tSrQ = qk_thr.make_fragment_A(tQ)[None, None, None, 0]
        tSrQ = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_q, dtype=BF16), tSrQ.layout)
        tSrK = qk_thr.make_fragment_B(sK)
        tOrV = pv_thr.make_fragment_B(sV)
        tStS = qk_thr.make_fragment_C(qk_thr.partition_shape_C((self.bm, self.bn)))
        tOtO_f = pv_thr.make_fragment_C(pv_thr.partition_shape_C((self.bm, self.d)))
        tOtO = cute.make_tensor(tmem_ptr + self.tmem_o, tOtO_f.layout)
        tP = cute.make_tensor(tmem_ptr + self.tmem_s0, p_layout.outer)
        tOrP = pv_thr.make_fragment_A(tP)[None, None, None, 0]
        if tiles > 0:
            cute.arch.mbarrier_wait(q_full, 0)
            tcgen05_fence_after()
            ring = Int32(0)
            ring_phase = Int32(0)
            # QK(0)
            cute.arch.mbarrier_wait(kv_full + ring, ring_phase)
            self.gemm_qk(
                qk_mma,
                qk_mma_acc,
                tSrQ,
                tSrK[None, None, None, ring],
                cute.make_tensor(tmem_ptr + self.tmem_s0, tStS.layout),
            )
            with cute.arch.elect_one():
                tcgen05.commit(s_full)
                tcgen05.commit(kv_empty + ring)
            ring, ring_phase = self.advance(ring, ring_phase)
            for j in cutlass.range(1, tiles + 1, 1, unroll=1):
                buf = j % 2
                if j < tiles:
                    # QK(j) into S[j % 2]
                    cute.arch.mbarrier_wait(kv_full + ring, ring_phase)
                    self.gemm_qk(
                        qk_mma,
                        qk_mma_acc,
                        tSrQ,
                        tSrK[None, None, None, ring],
                        cute.make_tensor(tmem_ptr + (self.tmem_s0 + buf * self.bn), tStS.layout),
                    )
                    with cute.arch.elect_one():
                        tcgen05.commit(s_full + buf)
                        tcgen05.commit(kv_empty + ring)
                    ring, ring_phase = self.advance(ring, ring_phase)
                # PV(j - 1)
                prev = j - 1
                pbuf = prev % 2
                cute.arch.mbarrier_wait(kv_full + ring, ring_phase)
                cute.arch.mbarrier_wait(p_full + pbuf, (prev // 2) % 2)
                tcgen05_fence_after()
                tOrP_b = cute.make_tensor(
                    cute.recast_ptr(
                        tmem_ptr + (self.tmem_s0 + pbuf * self.bn), dtype=tOrP.element_type
                    ),
                    tOrP.layout,
                )
                if prev == 0:
                    self.gemm_pv(pv_mma, pv_mma_acc, tOrP_b, tOrV[None, None, None, ring], tOtO)
                else:
                    self.gemm_pv(pv_mma_acc, pv_mma_acc, tOrP_b, tOrV[None, None, None, ring], tOtO)
                with cute.arch.elect_one():
                    tcgen05.commit(kv_empty + ring)
                    tcgen05.commit(o_full)
                ring, ring_phase = self.advance(ring, ring_phase)
            with cute.arch.elect_one():
                tcgen05.commit(o_final)

    @cute.jit
    def advance(self, ring: Int32, ring_phase: Int32):
        ring = ring + 1
        if ring == self.kv_slots:
            ring = Int32(0)
            ring_phase = ring_phase ^ 1
        return ring, ring_phase

    @cute.jit
    def gemm_qk(self, first: cute.TiledMma, rest: cute.TiledMma, tSrQ, tSrK, tStS):
        for kb in cutlass.range_constexpr(cute.size(tSrQ, mode=[2])):
            if cutlass.const_expr(kb == 0):
                cute.gemm(first, tStS, tSrQ[None, None, kb], tSrK[None, None, kb], tStS)
            else:
                cute.gemm(rest, tStS, tSrQ[None, None, kb], tSrK[None, None, kb], tStS)

    @cute.jit
    def gemm_pv(self, first: cute.TiledMma, rest: cute.TiledMma, tOrP, tOrV, tOtO):
        for kb in cutlass.range_constexpr(cute.size(tOrP, mode=[2])):
            if cutlass.const_expr(kb == 0):
                cute.gemm(first, tOtO, tOrP[None, None, kb], tOrV[None, None, kb], tOtO)
            else:
                cute.gemm(rest, tOtO, tOrP[None, None, kb], tOrV[None, None, kb], tOtO)

    # ------------------------------------------------------------------ softmax + epilogue (warps 0-3)
    @cute.jit
    def softmax_role(
        self,
        qk_thr,
        pv_thr,
        tmem_ptr,
        st: Int32,
        group: Int32,
        kv_head: Int32,
        tiles: Int32,
        rows: Int32,
        bits: cute.Pointer,
        bits_stride_row: Int32,
        out: cute.Pointer,
        out_stride_row: Int32,
        scale_log2: Float32,
        q_full,
        s_full,
        p_full,
        o_full,
        o_final,
        stage,
        q: cute.Pointer,
        q_stride_row: Int32,
    ):
        G = self.group_size
        R = self.rows_per_tile
        D = self.d
        tStS = qk_thr.make_fragment_C(qk_thr.partition_shape_C((self.bm, self.bn)))
        tStS0 = cute.make_tensor(tmem_ptr + self.tmem_s0, tStS.layout)
        tOtO_f = pv_thr.make_fragment_C(pv_thr.partition_shape_C((self.bm, self.d)))
        tOtO = cute.make_tensor(tmem_ptr + self.tmem_o, tOtO_f.layout)

        # S: one 64-column load a tile; P: 32 packed-bf16 columns stored over the S it came from.
        cS = cute.make_identity_tensor((self.bm, self.bn))
        tScS = qk_thr.partition_C(cS)
        # The load also reduces the row max over the tile's 64 columns (tcgen05.ld.red, SM103): an
        # upper bound of the row's masked max, which serves as the softmax offset just as well.
        load_s = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.copy.LdRed32x32bOp(
                    tcgen05.copy.Repetition(32), redOp=tcgen05.TmemLoadRedOp.MAX
                ),
                Float32,
            ),
            tStS0,
        )
        thr_load_s = load_s.get_slice(st)
        tLtS0 = thr_load_s.partition_S(tStS0)
        tLcS = thr_load_s.partition_D(tScS)
        p_cols = self.bn // 2
        tStP_layout = cute.composition(tStS0.layout, cute.make_layout((self.bm, p_cols)))
        tStP0 = cute.make_tensor(tStS0.iterator, tStP_layout)
        tScP = cute.make_tensor(
            tScS.iterator, cute.composition(tScS.layout, cute.make_layout((self.bm, p_cols)))
        )
        store_p = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(p_cols)), Float32),
            tStP0,
        )
        thr_store_p = store_p.get_slice(st)
        tPtP0 = thr_store_p.partition_D(tStP0)
        tPcP = thr_store_p.partition_S(tScP)

        # O in 32-column chunks (rescale and epilogue).
        cO = cute.make_identity_tensor((self.bm, self.d))
        tOcO = pv_thr.partition_C(cO)
        tOtO_c = cute.make_tensor(
            tOtO.iterator,
            cute.composition(tOtO.layout, cute.make_layout((self.bm, self.chunk_cols))),
        )
        tOcO_c = cute.make_tensor(
            tOcO.iterator,
            cute.composition(tOcO.layout, cute.make_layout((self.bm, self.chunk_cols))),
        )
        load_o = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.chunk_cols)), Float32
            ),
            tOtO_c,
        )
        store_o = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(self.chunk_cols)), Float32
            ),
            tOtO_c,
        )
        thr_load_o = load_o.get_slice(st)
        thr_store_o = store_o.get_slice(st)
        tLtO = thr_load_o.partition_S(tOtO_c)
        tLcO = thr_load_o.partition_D(tOcO_c)
        tStO = thr_store_o.partition_D(tOtO_c)
        n_chunks = self.d // self.chunk_cols

        m = st
        local_row = m // G
        head = m % G
        row = group * R + local_row
        live = (local_row < R) and (row < rows)
        safe_row = group * R
        if live:
            safe_row = row
        mBits = cute.make_tensor(
            bits + Int64(safe_row) * bits_stride_row, cute.make_layout((1 << 30,))
        )

        # Q row -> TMEM columns tmem_q.. (the QK MMA's A operand), 64 elements a store. Dead rows
        # store zeros; their S rows are masked out.
        if tiles > 0:
            q_row = cute.make_tensor(
                q + cute.assume(Int64(safe_row) * q_stride_row + (kv_head * G + head) * D, divby=8),
                cute.make_layout((D,)),
            )
            for c in cutlass.range_constexpr(D // (2 * p_cols)):
                rQ = cute.make_rmem_tensor(tPcP.shape, Float32)
                rQ16 = cute.make_tensor(
                    cute.recast_ptr(rQ.iterator, dtype=BF16), cute.make_layout((2 * p_cols,))
                )
                if live:
                    for i in cutlass.range_constexpr(2 * p_cols // 8):
                        cute.autovec_copy(
                            cute.make_tensor(
                                q_row.iterator + (c * 2 * p_cols + i * 8), cute.make_layout((8,))
                            ),
                            cute.make_tensor(rQ16.iterator + i * 8, cute.make_layout((8,))),
                        )
                else:
                    rQ.fill(0.0)
                cute.copy(
                    store_p,
                    rQ,
                    cute.make_tensor(
                        tPtP0.iterator + (self.tmem_q - self.tmem_s0 + c * p_cols), tPtP0.layout
                    ),
                )
            cute.arch.fence_view_async_tmem_store()
            tcgen05_fence_before()
            cute.arch.mbarrier_arrive(q_full)

        row_max = -Float32.inf
        row_sum = Float32(0.0)
        lo_next = Int32(0)
        hi_next = Int32(0)
        if live and tiles > 0:
            lo_next = mBits[0]
            hi_next = mBits[1]
        tau = Float32(self.tau)
        for j in cutlass.range(0, tiles, 1, unroll=1):
            lo = lo_next
            hi = hi_next
            if live and j + 1 < tiles:
                lo_next = mBits[2 * j + 2]
                hi_next = mBits[2 * j + 3]
            buf = j % 2
            cute.arch.mbarrier_wait(s_full + buf, (j // 2) % 2)
            tcgen05_fence_after()
            rS = cute.make_rmem_tensor(tLcS.shape, Float32)
            n_ld = cute.size(rS, mode=[1])
            rMax = cute.make_rmem_tensor(cute.make_layout((1, n_ld)), Float32)
            tLtS_b = cute.make_tensor(tLtS0.iterator + buf * self.bn, tLtS0.layout)
            for c in cutlass.range_constexpr(n_ld):
                cute.copy_atom_call(
                    load_s, tLtS_b[None, c, 0, 0], (rS[None, c, 0, 0], rMax[None, c])
                )
            cute.arch.fence_view_async_tmem_load()
            tile_max = rMax[0]
            for c in cutlass.range_constexpr(1, n_ld):
                tile_max = cute.arch.fmax(tile_max, rMax[c])
            new_max = cute.arch.fmax(row_max, tile_max)
            use_max = row_max
            factor = Float32(1.0)
            if row_max == -Float32.inf:
                use_max = new_max
            elif (new_max - row_max) * scale_log2 > tau:
                use_max = new_max
                factor = cute.math.exp2((row_max - new_max) * scale_log2, fastmath=True)
            need = factor != Float32(1.0)
            if cute.arch.vote_any_sync(need):
                # O holds PV(0..j-1): wait for PV(j-1), then rescale this warp's rows.
                cute.arch.mbarrier_wait(o_full, (j - 1) % 2)
                tcgen05_fence_after()
                for c in cutlass.range_constexpr(n_chunks):
                    rO = cute.make_rmem_tensor(tLcO.shape, Float32)
                    cute.copy(
                        load_o,
                        cute.make_tensor(tLtO.iterator + c * self.chunk_cols, tLtO.layout),
                        rO,
                    )
                    for i in cutlass.range_constexpr(cute.size(rO)):
                        rO[i] = rO[i] * factor
                    cute.copy(
                        store_o,
                        rO,
                        cute.make_tensor(tStO.iterator + c * self.chunk_cols, tStO.layout),
                    )
                cute.arch.fence_view_async_tmem_store()
            row_max = use_max
            neg = -use_max * scale_log2
            rP = cute.make_rmem_tensor(tPcP.shape, Float32)
            rP16 = cute.make_tensor(cute.recast_ptr(rP.iterator, dtype=BF16), rS.layout)
            # p = 2^(s * scale - max * scale) on the row's own entries, 0 elsewhere; four partial sums.
            acc0 = (Float32(0.0), Float32(0.0))
            acc1 = (Float32(0.0), Float32(0.0))
            for i in cutlass.range_constexpr(0, self.bn, 4):
                word = lo if i < 32 else hi
                e = cute.make_rmem_tensor((4,), Float32)
                e[0], e[1] = cute.arch.fma_packed_f32x2(
                    (rS[i], rS[i + 1]), (scale_log2, scale_log2), (neg, neg)
                )
                e[2], e[3] = cute.arch.fma_packed_f32x2(
                    (rS[i + 2], rS[i + 3]), (scale_log2, scale_log2), (neg, neg)
                )
                for u in cutlass.range_constexpr(4):
                    x = cute.math.exp2(e[u], fastmath=True)
                    rS[i + u] = x if (word & (1 << ((i + u) % 32))) != 0 else Float32(0.0)
                acc0 = cute.arch.add_packed_f32x2(acc0, (rS[i], rS[i + 1]))
                acc1 = cute.arch.add_packed_f32x2(acc1, (rS[i + 2], rS[i + 3]))
            row_sum = row_sum * factor + ((acc0[0] + acc0[1]) + (acc1[0] + acc1[1]))
            rP16.store(rS.load().to(BF16))
            cute.copy(store_p, rP, cute.make_tensor(tPtP0.iterator + buf * self.bn, tPtP0.layout))
            cute.arch.fence_view_async_tmem_store()
            tcgen05_fence_before()
            cute.arch.mbarrier_arrive(p_full + buf)

        # Epilogue: O / l in bf16, staged through the (now idle) K/V ring -- a row's 32 16-byte
        # chunks XOR-swizzled by the row -- so each warp then stores whole 512-byte rows.
        if tiles > 0:
            cute.arch.mbarrier_wait(o_final, 0)
            tcgen05_fence_after()
        inv = Float32(0.0)
        if row_sum > Float32(0.0):
            inv = Float32(1.0) / row_sum
        swz = m % 32
        for c in cutlass.range_constexpr(n_chunks):
            rO = cute.make_rmem_tensor(tLcO.shape, Float32)
            if tiles > 0:
                cute.copy(
                    load_o, cute.make_tensor(tLtO.iterator + c * self.chunk_cols, tLtO.layout), rO
                )
                for i in cutlass.range_constexpr(cute.size(rO)):
                    rO[i] = rO[i] * inv
            else:
                rO.fill(0.0)
            rB = cute.make_rmem_tensor((self.chunk_cols,), BF16)
            rB.store(rO.load().to(BF16))
            for i in cutlass.range_constexpr(self.chunk_cols // 8):
                logical = c * (self.chunk_cols // 8) + i
                offset = cute.assume(m * D + ((logical ^ swz) * 8), divby=8)
                dst = cute.make_tensor(stage + offset, cute.make_layout((8,)))
                cute.autovec_copy(
                    cute.make_tensor(rB.iterator + i * 8, cute.make_layout((8,))), dst
                )
        cute.arch.sync_warp()
        lane = st % 32
        warp = st // 32
        for r in cutlass.range_constexpr(32):
            mr = warp * 32 + r
            lr = mr // G
            row_r = group * R + lr
            if lr < R and row_r < rows:
                src_off = cute.assume(mr * D + ((lane ^ r) * 8), divby=8)
                rV = cute.make_rmem_tensor((8,), BF16)
                cute.autovec_copy(cute.make_tensor(stage + src_off, cute.make_layout((8,))), rV)
                dst_off = cute.assume(
                    Int64(row_r) * out_stride_row + (kv_head * G + mr % G) * D + lane * 8, divby=8
                )
                cute.autovec_copy(rV, cute.make_tensor(out + dst_off, cute.make_layout((8,))))


@functools.lru_cache(maxsize=None)
def _compiled(group_size: int, rows_per_tile: int, tokens_per_block: int, use_pdl: bool):
    kernel = _RowtileAttentionKernel(
        group_size, rows_per_tile, HEAD_DIM, tokens_per_block, use_pdl=use_pdl
    )
    gmem = cute.AddressSpace.gmem

    def fake(dtype, align):
        return make_ptr(dtype, 16 * 1024, gmem, assumed_align=align)

    args = (
        fake(BF16, 16),
        fake(BF16, 16),
        fake(BF16, 16),
        fake(Int32, 4),
        fake(Int32, 4),
        fake(Int32, 4),
        fake(BF16, 16),
        1,
        1,
        1,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        1,
        1,
        8,
        1.0,
        cuda_drv.CUstream(0),
    )
    return cute.compile(kernel, *args)


def _aligned(t: torch.Tensor, strides: int) -> bool:
    return t.data_ptr() % 16 == 0 and all(t.stride(i) % 8 == 0 for i in range(strides))


def eligible(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> bool:
    """Whether a launch fits this pass: SM103, bf16 q and K/V, D = 256, unit-stride head
    dimension and 16-byte-aligned rows."""
    if q.shape[2] != HEAD_DIM:
        return False
    if not (q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16):
        return False
    if torch.cuda.get_device_capability(q.device) != (10, 3):
        return False
    if (
        q.stride(2) != 1
        or q.stride(1) != HEAD_DIM
        or k_cache.stride(3) != 1
        or v_cache.stride(3) != 1
    ):
        return False
    return _aligned(q, 1) and _aligned(k_cache, 3) and _aligned(v_cache, 3)


class _Launcher:
    """`rowtile_attention_cute[grid](...)`: grid-indexed like the entry's Triton launches;
    the grid is `(groups, kv_heads)`."""

    def __getitem__(self, grid):
        return functools.partial(self._launch, tuple(grid))

    @staticmethod
    def _launch(
        grid,
        q,
        k_cache,
        v_cache,
        slots,
        packed_bits,
        group_plan,
        output,
        softmax_scale,
        rows,
        *,
        group_size,
        rows_per_tile,
        tokens_per_block,
        use_pdl,
    ):
        assert grid == (group_plan.shape[0], k_cache.shape[1]), grid
        assert (
            output.is_contiguous()
            and packed_bits.dtype == torch.int64
            and slots.dtype == torch.int32
        )
        fn = _compiled(group_size, rows_per_tile, tokens_per_block, bool(use_pdl))
        gmem = cute.AddressSpace.gmem

        def ptr(t, dtype, align):
            return make_ptr(dtype, t.data_ptr(), gmem, assumed_align=align)

        fn(
            ptr(q, BF16, 16),
            ptr(k_cache, BF16, 16),
            ptr(v_cache, BF16, 16),
            ptr(slots, Int32, 4),
            ptr(packed_bits, Int32, 4),
            ptr(group_plan, Int32, 4),
            ptr(output, BF16, 16),
            rows,
            grid[0],
            grid[1],
            q.stride(0),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            slots.stride(0),
            2 * packed_bits.stride(0),
            output.stride(0),
            softmax_scale * _LOG2_E,
            cuda_drv.CUstream(torch.cuda.current_stream(q.device).cuda_stream),
        )


rowtile_attention_cute = _Launcher()
