# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The row-tile pass of `qsa_sparse_gqa_rowtile` on SM103 tensor cores (CuTe DSL, tcgen05).

Inputs and output: per group of `rows_per_tile` rows of one request, the plan's packed
slot list (`slots[group]`, `listed` entries), each row's bitmap re-indexed onto it
(`packed_bits`, as int32 halves) and `group_plan[group] = (tiles, listed)`; groups with
`tiles < 0` belong to the lone-row pass.

A persistent kernel: one CTA per SM, each taking work items -- a (group, KV head) pair, 128
MMA rows = `rows_per_tile` query rows x `group_size` heads -- from a global ticket counter,
longest-first (the plan's last group first). 9 warps:

* warp 5 lane 0 draws the tickets and publishes the items through a shared-memory ring that
  every role reads; warps 5-8 gather each item's Q rows (into a 60 KB Q buffer) and its list's
  K and V rows (64 a tile) with `cp.async` into one ring of `kv_slots` 32 KB SW128 slots, filled
  in the MMA warp's order K0, K1, V0, K2, V1, ...; the next item's Q is fetched while the
  current item's last tiles load;
* warp 4 issues the tcgen05 MMAs: S[t % 2] = Q K(t)^T (Q from tensor memory), then
  O += P(t - 1) V(t - 1) (P from tensor memory, overwriting the S it came from), `t` counting
  the CTA's tiles across items;
* warps 0-3, one thread per MMA row, per tile load S with its row max (`tcgen05.ld.red`),
  rescale O when that max grew by more than `rescale_threshold` in log2 units (lazily;
  otherwise exponentiate against the stale max), write P = 2^(s - max) on the row's own bits
  and 0 elsewhere. After an item's last tile they copy the next item's Q from the Q buffer
  into tensor memory (so its first QK MMAs overlap this epilogue), then normalize O, stage it
  in the Q buffer as four 64-dim SW128 boxes of the item's G heads x R rows and store it with
  TMA (thread 0 hands the buffer back to the loaders once the store has read it).

The Q buffer is shared by Q and the staged O: a softmax thread's Q row and its O row occupy
the same bytes, so the next item's Q is read out before this item's O is staged over it. An
item followed by a lone (or empty) group stores O row by row from registers instead, as the
loaders may then fetch the following item's Q into the buffer at any time.

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
#: K/V ring slots (32 KB each); with the 60 KB Q buffer, about 220 KB of shared memory.
KV_SLOTS = 5
#: Work items in flight in the shared-memory schedule ring.
SCHED_SLOTS = 8
#: The next item's Q is fetched once this many tiles of the current item remain to load.
Q_LEAD = 3


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
        kv_slots: int = KV_SLOTS,
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
        self.sched_slots = SCHED_SLOTS
        self.q_lead = Q_LEAD
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
        work: cute.Pointer,
        rows: Int32,
        num_groups: Int32,
        num_kv_heads: Int32,
        num_ctas: Int32,
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
            qs_full: cute.struct.MemRange[Int64, 1]
            qs_empty: cute.struct.MemRange[Int64, 1]
            sch_full: cute.struct.MemRange[Int64, self.sched_slots]
            sch_empty: cute.struct.MemRange[Int64, self.sched_slots]
            sch_val: cute.struct.MemRange[Int32, self.sched_slots]
            tmem_holding: Int32
            sKV: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(k_layout)], 1024]
            # The Q buffer / O staging: four 64-dim boxes of the live G x R MMA rows.
            sQ: cute.struct.Align[
                cute.struct.MemRange[BF16, self.group_size * self.rows_per_tile * self.d], 1024
            ]

        self.shared_storage = SharedStorage

        # The epilogue's TMA store: out viewed as (D, heads, rows), one box = 64 dims of the
        # group's G heads x R rows, staged in the Q buffer as 128-byte SW128 rows (MMA row m =
        # local row x G + head), four boxes 64 G R elements apart.
        G = self.group_size
        R = self.rows_per_tile
        o_box = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3), 0, cute.make_layout((64, G, R), stride=(1, 64, 64 * G))
        )
        mO = cute.make_tensor(
            out,
            cute.make_layout((self.d, num_kv_heads * G, rows), stride=(1, self.d, out_stride_row)),
        )
        tma_atom_o, tma_tensor_o = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mO, o_box, (64, G, R)
        )

        self.kernel(
            qk_mma,
            qk_mma_acc,
            pv_mma,
            pv_mma_acc,
            q_layout,
            k_layout,
            v_layout,
            p_layout,
            tma_atom_o,
            tma_tensor_o,
            q,
            k,
            v,
            slots,
            bits,
            plan,
            out,
            work,
            rows,
            num_groups,
            num_kv_heads,
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
            grid=[num_ctas, 1, 1],
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
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        q: cute.Pointer,
        k: cute.Pointer,
        v: cute.Pointer,
        slots: cute.Pointer,
        bits: cute.Pointer,
        plan: cute.Pointer,
        out: cute.Pointer,
        work: cute.Pointer,
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
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        qk_mma.set(tcgen05.Field.ACCUMULATE, False)
        qk_mma_acc.set(tcgen05.Field.ACCUMULATE, True)
        pv_mma.set(tcgen05.Field.ACCUMULATE, False)
        pv_mma_acc.set(tcgen05.Field.ACCUMULATE, True)

        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        q_full = storage.q_full.data_ptr()
        kv_full = storage.kv_full.data_ptr()
        kv_empty = storage.kv_empty.data_ptr()
        s_full = storage.s_full.data_ptr()
        p_full = storage.p_full.data_ptr()
        o_full = storage.o_full.data_ptr()
        o_final = storage.o_final.data_ptr()
        qs_full = storage.qs_full.data_ptr()
        qs_empty = storage.qs_empty.data_ptr()
        sch_full = storage.sch_full.data_ptr()
        sch_empty = storage.sch_empty.data_ptr()
        sch_val = storage.sch_val.data_ptr()
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
            cute.arch.mbarrier_init(qs_full, 128)
            # One arrival a hand-back: softmax thread 0, after the softmax warps are done with it.
            cute.arch.mbarrier_init(qs_empty, 1)
            for i in cutlass.range_constexpr(self.sched_slots):
                cute.arch.mbarrier_init(sch_full + i, 1)
                # One elected lane of each of the 9 warps releases an item.
                cute.arch.mbarrier_init(sch_empty + i, 9)
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
        sQ = storage.sQ.data_ptr()
        sO = storage.sQ.get_tensor(
            cute.make_layout(
                (64, self.group_size, self.rows_per_tile, self.d // 64),
                stride=(1, 64, 64 * self.group_size, 64 * self.group_size * self.rows_per_tile),
            ),
            swizzle=cute.make_swizzle(3, 4, 3),
        )

        if warp_idx >= self.load_warp0:
            self.load_role(
                tidx - 32 * self.load_warp0,
                sK,
                sQ,
                q,
                k,
                v,
                slots,
                plan,
                work,
                rows,
                num_groups,
                num_kv_heads,
                q_stride_row,
                k_stride_page,
                k_stride_head,
                k_stride_token,
                v_stride_page,
                v_stride_head,
                v_stride_token,
                slots_stride_group,
                kv_full,
                kv_empty,
                qs_full,
                qs_empty,
                sch_full,
                sch_empty,
                sch_val,
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
                plan,
                num_kv_heads,
                q_full,
                kv_full,
                kv_empty,
                s_full,
                p_full,
                o_full,
                o_final,
                sch_full,
                sch_empty,
                sch_val,
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
                plan,
                num_kv_heads,
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
                qs_full,
                qs_empty,
                sch_full,
                sch_empty,
                sch_val,
                sQ,
                sO,
                tma_atom_o,
                tma_tensor_o,
            )
            tmem_users.arrive_and_wait()

        if cutlass.const_expr(self.use_pdl):
            # The lone-row pass may still run: finish only after it has, so a later launch sees every row.
            cute.arch.griddepcontrol_wait()

    # ------------------------------------------------------------------ schedule
    @cute.jit
    def read_item(self, n: Int32, sch_full, sch_val):
        """Item `n` of this CTA (group * kv_heads + head, or -1 past the last), once published."""
        slot = n % self.sched_slots
        cute.arch.mbarrier_wait(sch_full + slot, (n // self.sched_slots) % 2)
        return cute.make_tensor(sch_val, cute.make_layout((self.sched_slots,)))[slot]

    @cute.jit
    def release_item(self, n: Int32, sch_empty):
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(sch_empty + n % self.sched_slots)

    @cute.jit
    def item_info(self, item: Int32, plan: cute.Pointer, num_kv_heads: Int32):
        """(group, KV head, tiles, listed) of a published item; tiles = -1 past the last."""
        group = Int32(0)
        head = Int32(0)
        tiles = Int32(-1)
        listed = Int32(0)
        if item >= 0:
            group = item // num_kv_heads
            head = item - group * num_kv_heads
            mPlan = cute.make_tensor(plan, cute.make_layout((1 << 30,)))
            tiles = mPlan[group * 2]
            listed = mPlan[group * 2 + 1]
        return group, head, tiles, listed

    # ------------------------------------------------------------------ loaders (warps 5-8)
    @cute.jit
    def publish(self, n: Int32, ticket: Int32, num_items: Int32, num_groups: Int32,
                num_kv_heads: Int32, sch_full, sch_empty, sch_val):  # fmt: skip
        """Item `n` from a ticket: the plan's groups last to first, both heads of a group in turn."""
        item = Int32(-1)
        if ticket < num_items:
            back = ticket // num_kv_heads
            item = (num_groups - 1 - back) * num_kv_heads + (ticket - back * num_kv_heads)
        slot = n % self.sched_slots
        cute.arch.mbarrier_wait(sch_empty + slot, ((n // self.sched_slots) % 2) ^ 1)
        cute.make_tensor(sch_val, cute.make_layout((self.sched_slots,)))[slot] = item
        cute.arch.mbarrier_arrive(sch_full + slot)

    @cute.jit
    def load_role(
        self,
        lt: Int32,
        sK,
        sQ: cute.Pointer,
        q: cute.Pointer,
        k: cute.Pointer,
        v: cute.Pointer,
        slots: cute.Pointer,
        plan: cute.Pointer,
        work: cute.Pointer,
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
        kv_full,
        kv_empty,
        qs_full,
        qs_empty,
        sch_full,
        sch_empty,
        sch_val,
    ):
        """Per item: its Q rows into the Q buffer (unless fetched during the previous item), then
        the ring in the MMA warp's order K0, K1, V0, K2, V1, ..., V(T-1). Thread `lt` gathers
        tokens 16 (lt // 32) + 4 t + (lt % 32) // 8 (t < 4), one 16-byte chunk (lt % 8) of each of
        their four 128-byte dim blocks; the slots of tile j + 1 are fetched while tile j's copies
        issue. Lane 0 of warp 5 publishes item n + 1 when the warps start item n.
        """
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
        num_items = num_groups * num_kv_heads
        first_token = 16 * (lt // 32) + (lt % 32) // 8
        ticket = Int32(0)
        if lt == 0:
            ticket = cute.arch.atomic_add(work, Int32(1))
            self.publish(
                Int32(0), ticket, num_items, num_groups, num_kv_heads, sch_full, sch_empty, sch_val
            )
            ticket = cute.arch.atomic_add(work, Int32(1))
        n = Int32(0)
        ring = Int32(0)
        ring_phase = Int32(1)
        qs_phase = Int32(1)
        q_ready = Int32(0)
        item = self.read_item(n, sch_full, sch_val)
        while item >= 0:
            if lt == 0:
                self.publish(
                    n + 1, ticket, num_items, num_groups, num_kv_heads, sch_full, sch_empty, sch_val
                )
                ticket = cute.arch.atomic_add(work, Int32(1))
            group, kv_head, tiles, listed = self.item_info(item, plan, num_kv_heads)
            nitem = self.read_item(n + 1, sch_full, sch_val)
            ngroup, nhead, ntiles, _ = self.item_info(nitem, plan, num_kv_heads)
            if tiles > 0:
                if q_ready == 0:
                    cute.arch.mbarrier_wait(qs_empty, qs_phase)
                    qs_phase = qs_phase ^ 1
                    self.issue_q(lt, group, kv_head, rows, q, q_stride_row, sQ)
                    cute.arch.cp_async_mbarrier_arrive_noinc(qs_full)
                q_ready = Int32(0)
                q_at = cutlass.max(tiles - self.q_lead, Int32(0))
                mSlots = cute.make_tensor(
                    slots + Int64(group) * slots_stride_group, cute.make_layout((1 << 30,))
                )
                slot_next = cute.make_rmem_tensor((4,), Int32)
                k_cur = cute.make_rmem_tensor((4,), Int64)
                v_cur = cute.make_rmem_tensor((4,), Int64)
                v_prev = cute.make_rmem_tensor((4,), Int64)
                for t in cutlass.range_constexpr(4):
                    slot_next[t] = mSlots[cutlass.min(first_token + 4 * t, listed - 1)]
                for j in cutlass.range(0, tiles, 1, unroll=1):
                    for t in cutlass.range_constexpr(4):
                        slot = slot_next[t]
                        page = slot // self.tpb
                        tok = slot % self.tpb
                        k_cur[t] = (
                            Int64(page) * k_stride_page
                            + kv_head * k_stride_head
                            + tok * k_stride_token
                        )
                        v_cur[t] = (
                            Int64(page) * v_stride_page
                            + kv_head * v_stride_head
                            + tok * v_stride_token
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
                    if j == q_at and ntiles > 0:
                        # The next item's Q, once this item's epilogue no longer needs the buffer.
                        cute.arch.mbarrier_wait(qs_empty, qs_phase)
                        qs_phase = qs_phase ^ 1
                        self.issue_q(lt, ngroup, nhead, rows, q, q_stride_row, sQ)
                        cute.arch.cp_async_mbarrier_arrive_noinc(qs_full)
                        q_ready = Int32(1)
                # V(tiles - 1)
                cute.arch.mbarrier_wait(kv_empty + ring, ring_phase)
                self.issue_tile(atom, v, v_prev, sKV_c, ring, chunk, first_token)
                cute.arch.cp_async_mbarrier_arrive_noinc(kv_full + ring)
                ring, ring_phase = self.advance(ring, ring_phase)
            self.release_item(n, sch_empty)
            n = n + 1
            item = nitem
        # Tail: every slot released, so no load is in flight when the CTA retires.
        for i in cutlass.range_constexpr(self.kv_slots):
            cute.arch.mbarrier_wait(kv_empty + ring, ring_phase)
            ring, ring_phase = self.advance(ring, ring_phase)

    @cute.jit
    def issue_q(
        self,
        lt: Int32,
        group: Int32,
        kv_head: Int32,
        rows: Int32,
        q: cute.Pointer,
        q_stride_row: Int32,
        sQ: cute.Pointer,
    ):
        """An item's live Q rows into the Q buffer: MMA row m = local row x G + head at m x D,
        its 16-byte chunk c at chunk c ^ (m % 8). Consecutive threads take consecutive chunks of
        a row's G heads (contiguous in q).
        """
        G = self.group_size
        R = self.rows_per_tile
        D = self.d
        atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BF16, num_bits_per_copy=128
        )
        per_row = G * (D // 8)
        total = R * per_row
        for it in cutlass.range_constexpr((total + 127) // 128):
            x = lt + 128 * it
            local_row = x // per_row
            rem = x - local_row * per_row
            head = rem // (D // 8)
            c = rem - head * (D // 8)
            row = group * R + local_row
            if x < total and row < rows:
                m = local_row * G + head
                src = cute.make_tensor(
                    q
                    + cute.assume(
                        Int64(row) * q_stride_row + (kv_head * G + head) * D + c * 8, divby=8
                    ),
                    cute.make_layout((8,)),
                )
                dst = cute.make_tensor(
                    sQ + cute.assume(self.box_offset(m, c), divby=8), cute.make_layout((8,))
                )
                cute.copy(atom, src, dst)

    @cute.jit
    def box_offset(self, m, c):
        """Element offset of row m's 16-byte chunk c (of D / 8) in the Q buffer: the layout of the
        epilogue's TMA boxes -- box c // 8 (64 dims of G heads x R rows), its 128-byte row m,
        chunk (c % 8) ^ (m % 8) (SW128) -- so a softmax thread's Q row and its O row occupy the
        same bytes.
        """
        return (
            (c // 8) * (64 * self.group_size * self.rows_per_tile)
            + m * 64
            + (((c % 8) ^ (m % 8)) * 8)
        )

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
        plan: cute.Pointer,
        num_kv_heads: Int32,
        q_full,
        kv_full,
        kv_empty,
        s_full,
        p_full,
        o_full,
        o_final,
        sch_full,
        sch_empty,
        sch_val,
    ):
        """Per item: QK(0), then QK(j) / PV(j - 1) pairs, then the last PV. S/P buffer b sits at
        TMEM columns tmem_s0 + 64 b, b = the CTA's tile count mod 2; QK(t) overwrites S[t % 2] only
        after PV(t - 2) was issued (tcgen05 MMAs of one thread execute in order), and P(t - 2) was
        its last reader. An item's PV(0) overwrites O: it waits on P(0), which the softmax warps
        publish only after reading the previous item's O out.
        """
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
        n = Int32(0)
        g = Int32(0)
        nq = Int32(0)
        ring = Int32(0)
        ring_phase = Int32(0)
        item = self.read_item(n, sch_full, sch_val)
        while item >= 0:
            group, kv_head, tiles, listed = self.item_info(item, plan, num_kv_heads)
            if tiles > 0:
                cute.arch.mbarrier_wait(q_full, nq % 2)
                tcgen05_fence_after()
                # QK(0)
                buf0 = g % 2
                cute.arch.mbarrier_wait(kv_full + ring, ring_phase)
                self.gemm_qk(
                    qk_mma,
                    qk_mma_acc,
                    tSrQ,
                    tSrK[None, None, None, ring],
                    cute.make_tensor(tmem_ptr + (self.tmem_s0 + buf0 * self.bn), tStS.layout),
                )
                with cute.arch.elect_one():
                    tcgen05.commit(s_full + buf0)
                    tcgen05.commit(kv_empty + ring)
                ring, ring_phase = self.advance(ring, ring_phase)
                for j in cutlass.range(1, tiles + 1, 1, unroll=1):
                    t = g + j
                    if j < tiles:
                        # QK(j) into S[t % 2]
                        buf = t % 2
                        cute.arch.mbarrier_wait(kv_full + ring, ring_phase)
                        self.gemm_qk(
                            qk_mma,
                            qk_mma_acc,
                            tSrQ,
                            tSrK[None, None, None, ring],
                            cute.make_tensor(
                                tmem_ptr + (self.tmem_s0 + buf * self.bn), tStS.layout
                            ),
                        )
                        with cute.arch.elect_one():
                            tcgen05.commit(s_full + buf)
                            tcgen05.commit(kv_empty + ring)
                        ring, ring_phase = self.advance(ring, ring_phase)
                    # PV(j - 1)
                    prev = t - 1
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
                    if j == 1:
                        self.gemm_pv(pv_mma, pv_mma_acc, tOrP_b, tOrV[None, None, None, ring], tOtO)
                    else:
                        self.gemm_pv(
                            pv_mma_acc, pv_mma_acc, tOrP_b, tOrV[None, None, None, ring], tOtO
                        )
                    with cute.arch.elect_one():
                        tcgen05.commit(kv_empty + ring)
                        tcgen05.commit(o_full)
                    ring, ring_phase = self.advance(ring, ring_phase)
                with cute.arch.elect_one():
                    tcgen05.commit(o_final)
                g = g + tiles
                nq = nq + 1
            self.release_item(n, sch_empty)
            n = n + 1
            item = self.read_item(n, sch_full, sch_val)

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
    def copy_q(self, m: Int32, live, sQ: cute.Pointer, store_p, tPtP0, tPcP, p_cols: int, q_full):
        """Row m's Q from the Q buffer into TMEM columns tmem_q.. (the QK MMA's A operand), 64
        elements a store; dead rows store zeros (their S rows are masked out).
        """
        D = self.d
        for c in cutlass.range_constexpr(D // (2 * p_cols)):
            rQ = cute.make_rmem_tensor(tPcP.shape, Float32)
            rQ16 = cute.make_tensor(
                cute.recast_ptr(rQ.iterator, dtype=BF16), cute.make_layout((2 * p_cols,))
            )
            if live:
                for i in cutlass.range_constexpr(2 * p_cols // 8):
                    cidx = c * (2 * p_cols // 8) + i
                    off = cute.assume(self.box_offset(m, cidx), divby=8)
                    cute.autovec_copy(
                        cute.make_tensor(sQ + off, cute.make_layout((8,))),
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

    @cute.jit
    def softmax_role(
        self,
        qk_thr,
        pv_thr,
        tmem_ptr,
        st: Int32,
        plan: cute.Pointer,
        num_kv_heads: Int32,
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
        qs_full,
        qs_empty,
        sch_full,
        sch_empty,
        sch_val,
        sQ: cute.Pointer,
        sO,
        tma_atom_o,
        tma_tensor_o,
    ):
        G = self.group_size
        R = self.rows_per_tile
        D = self.d
        tStS = qk_thr.make_fragment_C(qk_thr.partition_shape_C((self.bm, self.bn)))
        tStS0 = cute.make_tensor(tmem_ptr + self.tmem_s0, tStS.layout)
        tOtO_f = pv_thr.make_fragment_C(pv_thr.partition_shape_C((self.bm, self.d)))
        tOtO = cute.make_tensor(tmem_ptr + self.tmem_o, tOtO_f.layout)
        # The softmax warps' own barrier (hand-backs of the Q buffer, the TMA store's staging).
        softmax_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=128)

        # S: one 64-column load a tile; P: 32 packed-bf16 columns stored over the S it came from
        # (or two 16-column halves).
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

        # The TMA store's view: per item, box b of (group, KV head).
        gO = cute.flat_divide(tma_tensor_o, (64, G, R))
        tOsO, tOgO = cpasync.tma_partition(
            tma_atom_o,
            0,
            cute.make_layout(1),
            cute.group_modes(sO, 0, 3),
            cute.group_modes(gO, 0, 3),
        )

        m = st
        local_row = m // G
        head = m % G
        tau = Float32(self.tau)
        n = Int32(0)
        g = Int32(0)
        nq = Int32(0)
        qs_phase = Int32(0)
        q_ready = Int32(0)
        store_pending = Int32(0)
        item = self.read_item(n, sch_full, sch_val)
        while item >= 0:
            group, kv_head, tiles, listed = self.item_info(item, plan, num_kv_heads)
            row = group * R + local_row
            live = (local_row < R) and (row < rows)
            if tiles > 0:
                if q_ready == 0:
                    cute.arch.mbarrier_wait(qs_full, qs_phase)
                    qs_phase = qs_phase ^ 1
                    self.copy_q(m, live, sQ, store_p, tPtP0, tPcP, p_cols, q_full)
                    # Every row read out: hand the buffer back.
                    if st < 32:
                        softmax_bar.arrive_and_wait()
                        if st == 0:
                            cute.arch.mbarrier_arrive(qs_empty)
                    else:
                        softmax_bar.arrive()
                q_ready = Int32(0)
                safe_row = group * R
                if live:
                    safe_row = row
                mBits = cute.make_tensor(
                    bits + Int64(safe_row) * bits_stride_row, cute.make_layout((1 << 30,))
                )
                row_max = -Float32.inf
                row_sum = Float32(0.0)
                lo_next = Int32(0)
                hi_next = Int32(0)
                if live:
                    lo_next = mBits[0]
                    hi_next = mBits[1]
                for j in cutlass.range(0, tiles, 1, unroll=1):
                    t = g + j
                    lo = lo_next
                    hi = hi_next
                    if live and j + 1 < tiles:
                        lo_next = mBits[2 * j + 2]
                        hi_next = mBits[2 * j + 3]
                    buf = t % 2
                    cute.arch.mbarrier_wait(s_full + buf, (t // 2) % 2)
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
                        cute.arch.mbarrier_wait(o_full, (t - 1) % 2)
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
                    # p = 2^(s * scale - max * scale) on the row's own entries, 0 elsewhere; four
                    # partial sums.
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
                    cute.copy(
                        store_p, rP, cute.make_tensor(tPtP0.iterator + buf * self.bn, tPtP0.layout)
                    )
                    cute.arch.fence_view_async_tmem_store()
                    tcgen05_fence_before()
                    cute.arch.mbarrier_arrive(p_full + buf)
                    if j == 0 and store_pending != 0:
                        # The previous item's TMA store has read the Q buffer: hand it back.
                        if st == 0:
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                            cute.arch.mbarrier_arrive(qs_empty)
                        store_pending = Int32(0)

                # The last QK of this item has completed (its S arrived): the next item's Q goes
                # into tensor memory now, so its first QK MMAs overlap this epilogue.
                nitem = self.read_item(n + 1, sch_full, sch_val)
                ngroup, nhead, ntiles, _ = self.item_info(nitem, plan, num_kv_heads)
                next_q = ntiles > 0
                if next_q:
                    nrow = ngroup * R + local_row
                    nlive = (local_row < R) and (nrow < rows)
                    cute.arch.mbarrier_wait(qs_full, qs_phase)
                    qs_phase = qs_phase ^ 1
                    self.copy_q(m, nlive, sQ, store_p, tPtP0, tPcP, p_cols, q_full)
                    q_ready = Int32(1)
                # The Q buffer is free for staging O unless a later item still has to load its Q
                # there (a following lone or empty group): then each thread stores its own row.
                staged = next_q or nitem < 0

                # Epilogue: O / l in bf16.
                cute.arch.mbarrier_wait(o_final, nq % 2)
                tcgen05_fence_after()
                inv = Float32(0.0)
                if row_sum > Float32(0.0):
                    inv = Float32(1.0) / row_sum
                if staged:
                    if nitem < 0:
                        # No next Q orders this staging after the previous item's TMA store read
                        # the buffer (thread 0 waited for it at this item's first tile): sync.
                        softmax_bar.arrive_and_wait()
                    for c in cutlass.range_constexpr(n_chunks):
                        rO = cute.make_rmem_tensor(tLcO.shape, Float32)
                        cute.copy(
                            load_o,
                            cute.make_tensor(tLtO.iterator + c * self.chunk_cols, tLtO.layout),
                            rO,
                        )
                        for i in cutlass.range_constexpr(cute.size(rO)):
                            rO[i] = rO[i] * inv
                        rB = cute.make_rmem_tensor((self.chunk_cols,), BF16)
                        rB.store(rO.load().to(BF16))
                        if m < G * R:
                            for i in cutlass.range_constexpr(self.chunk_cols // 8):
                                offset = cute.assume(
                                    self.box_offset(m, c * (self.chunk_cols // 8) + i), divby=8
                                )
                                cute.autovec_copy(
                                    cute.make_tensor(rB.iterator + i * 8, cute.make_layout((8,))),
                                    cute.make_tensor(sQ + offset, cute.make_layout((8,))),
                                )
                    cute.arch.fence_view_async_shared()
                    if st < 32:
                        softmax_bar.arrive_and_wait()
                        if st == 0:
                            for b in cutlass.range_constexpr(self.d // 64):
                                cute.copy(tma_atom_o, tOsO[None, b], tOgO[None, b, kv_head, group])
                            cute.arch.cp_async_bulk_commit_group()
                    else:
                        softmax_bar.arrive()
                    store_pending = Int32(1)
                    if nitem < 0:
                        if st == 0:
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                        store_pending = Int32(0)
                else:
                    for c in cutlass.range_constexpr(n_chunks):
                        rO = cute.make_rmem_tensor(tLcO.shape, Float32)
                        cute.copy(
                            load_o,
                            cute.make_tensor(tLtO.iterator + c * self.chunk_cols, tLtO.layout),
                            rO,
                        )
                        for i in cutlass.range_constexpr(cute.size(rO)):
                            rO[i] = rO[i] * inv
                        rB = cute.make_rmem_tensor((self.chunk_cols,), BF16)
                        rB.store(rO.load().to(BF16))
                        if live:
                            for i in cutlass.range_constexpr(self.chunk_cols // 8):
                                dst_off = cute.assume(
                                    Int64(row) * out_stride_row
                                    + (kv_head * G + head) * D
                                    + c * self.chunk_cols
                                    + i * 8,
                                    divby=8,
                                )
                                cute.autovec_copy(
                                    cute.make_tensor(rB.iterator + i * 8, cute.make_layout((8,))),
                                    cute.make_tensor(out + dst_off, cute.make_layout((8,))),
                                )
                g = g + tiles
                nq = nq + 1
            elif tiles == 0:
                # Nothing resident for this group: zeros.
                if live:
                    rZ = cute.make_rmem_tensor((8,), BF16)
                    rZ.fill(0.0)
                    for i in cutlass.range_constexpr(D // 8):
                        dst_off = cute.assume(
                            Int64(row) * out_stride_row + (kv_head * G + head) * D + i * 8,
                            divby=8,
                        )
                        cute.autovec_copy(
                            rZ, cute.make_tensor(out + dst_off, cute.make_layout((8,)))
                        )
            self.release_item(n, sch_empty)
            n = n + 1
            item = self.read_item(n, sch_full, sch_val)


@functools.lru_cache(maxsize=None)
def _compiled(group_size: int, rows_per_tile: int, tokens_per_block: int, use_pdl: bool):
    kernel = _RowtileAttentionKernel(
        group_size, rows_per_tile, HEAD_DIM, tokens_per_block, kv_slots=KV_SLOTS, use_pdl=use_pdl
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
        fake(Int32, 4),
        1,
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


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def num_ctas(num_items: int, device: torch.device) -> int:
    """The persistent grid: one CTA per SM, never more than there are items."""
    return max(1, min(num_items, _sm_count(device.index if device.index is not None else 0)))


def _aligned(t: torch.Tensor, strides: int) -> bool:
    return t.data_ptr() % 16 == 0 and all(t.stride(i) % 8 == 0 for i in range(strides))


def eligible(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> bool:
    """Whether a launch fits this pass: SM103, bf16 q and K/V, D = 256, unit-stride head
    dimension and 16-byte-aligned rows.
    """
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
    the grid is `(ctas,)`, the persistent CTAs (`num_ctas`).
    """

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
        work,
        softmax_scale,
        rows,
        *,
        group_size,
        rows_per_tile,
        tokens_per_block,
        use_pdl,
    ):
        assert len(grid) == 1, grid
        assert (
            output.is_contiguous()
            and packed_bits.dtype == torch.int64
            and slots.dtype == torch.int32
            and work.dtype == torch.int32
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
            ptr(work, Int32, 4),
            rows,
            group_plan.shape[0],
            k_cache.shape[1],
            grid[0],
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
