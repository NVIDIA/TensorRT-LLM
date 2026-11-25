# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-connection mixer first half at decode row counts (1..256 rows) as one
tcgen05 split-K GEMM: the first projection `normed @ [down; block_inject]^T`
with the SiLU and injection activations in its epilogue, so neither the fp32
logits nor a split-K reduce kernel nor `qwen4_exp_hc_lowrank_act` exists.

The hyper-connection is replicated on every rank, so its shapes are the checkpoint's at
every member (tp4, tep4, dep4); only the rows move.

Tier 6 (CuTe DSL). One CTA computes one
`[128 rows, BN columns]` partial over `K / KS` of the 10240-wide stream (grid z
is the 128-row M tile): a TMA warp streams the normed rows (rows past `T`
zero-filled by TMA, never read)
and the weight rows (only the `n_cols` real ones; the load-time zero padding
to 384 rows is never read), one MMA warp accumulates in tensor memory, four
epilogue warps move the fp32 partial into shared memory. The `KS` CTAs of one
column tile form a thread-block cluster; after a cluster barrier each CTA sums
a slice of the tile's `KS` partials out of its peers' shared memory (DSMEM) in
a fixed order (deterministic) and writes the activations.

Launched with programmatic dependent launch: the weight stages stream before
the grid-dependency wait (the producer, `qwen4_exp_hc_inject_norm`, releases
dependents at its start), and this kernel releases its own consumer
(`qwen4_exp_hc_up_gate_mean(_fp4)`) at once so that it prefetches its weight.

Two plans for an attention-DP rank's decode rows, both bitwise the 128-row tile's outputs
(the same K slices, accumulated and summed in the same order):
- `tm=64` (up to `TILE64_MAX_ROWS` rows, tile `_TILE64`): a 64-row M tile, half the normed
  bytes a CTA streams and half the partial it keeps;
- rows on N (`HcDecodeTcFirstRowsNKernel`, `hc_first_rowsn`; 65..96 rows): the operand
  roles swapped, the weight rows on a 64-row MMA M tile and the normed rows on the MMA N
  in 16-row steps, where tcgen05 has no M tile between 64 and 128 rows.

One plan of another K split (opt-in at the call site): `tm64x2=True` (65..128 rows,
`TILE64X2_ROWS`) runs two 64-row M tiles on an 8-way K split (`_TILE64X2`), the split the
default plan takes past 128 rows, so the logits are summed in another order than the
128-row tile's 16-way split at these rows.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
import torch.nn.functional as F
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.kernels.blackwell.custom_pipeline import PipelineTmaUmma, PipelineUmmaAsync
from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr
from tensorrt_llm._torch.modeling_v2.catalog._op import Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.archs import SERVED_ARCHS

#: Largest row count this entry is certified for: two 128-row MMA tiles.
MAX_ROWS = 256

#: Programmatic dependent launch (sm_90+), off with TRTLLM_ENABLE_PDL=0. Without it
#: the grid-dependency wait and the dependents trigger are no-ops.
_PDL = get_env_enable_pdl()


def _asm(ret, args, text, constraints, loc, ip, side_effects=False):
    return llvm.inline_asm(
        ret,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mapa(addr, rank, *, loc=None, ip=None):
    """This CTA's shared-memory address `addr` in cluster CTA `rank`'s window."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Int32(addr), cutlass.Int32(rank)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _ld_dsmem_f32(addr, *, loc=None, ip=None):
    """`ld.shared::cluster.f32`; volatile so it stays behind the cluster barrier."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Int32(addr)],
            "ld.shared::cluster.f32 $0, [$1];",
            "=f,r",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def _ld_dsmem_f32_if(addr, pred, *, loc=None, ip=None):
    """`ld.shared::cluster.f32` where `pred` is not 0, else 0.0 with nothing read; volatile
    so it stays behind the cluster barrier."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Int32(addr), cutlass.Int32(pred)],
            "{ .reg .pred p; setp.ne.b32 p, $2, 0; mov.b32 $0, 0; "
            "@p ld.shared::cluster.f32 $0, [$1]; }",
            "=f,r,r",
            loc,
            ip,
            side_effects=True,
        )
    )


class HcDecodeTcFirstKernel:
    """`low[t, n] = silu(L[t, n] / C)` for `n < R` and, with `with_inject`,
    `inject[t, j] = 2 * sigmoid(L[t, R + j] / C)`, where `L = A @ B^T`,
    `A = normed [T, K]`, `B = weight rows [0, n_cols)` of `[>= n_cols, K]`.

    Compile-time: `bn` (MMA N, a multiple of 16), `ks` (K split = cluster
    size), `hc`, `rank`, `n_cols`, `k`, `with_inject`, `tm` (the M tile, 128 or 64
    rows). Runtime: the row count (one grid-z M tile per `tm` rows) and the pointers.
    """

    def __init__(self, bn, ks, hc, rank, n_cols, k, with_inject, epi_cols=16, tm=128):
        assert tm in (64, 128)
        self.tm = tm
        assert bn % epi_cols == 0 and bn % 16 == 0 and 16 <= bn <= 256
        assert k % (64 * ks) == 0, "K must split into 64-wide tiles per CTA"
        self.bn, self.ks, self.hc, self.rank, self.n_cols, self.k = bn, ks, hc, rank, n_cols, k
        self.with_inject = with_inject
        self.n_tiles = (n_cols + bn - 1) // bn
        self.acc_dtype = cutlass.Float32
        self.io_dtype = cutlass.BFloat16
        self.mma_tiler = (tm, bn, 1)
        self.epi_cols = epi_cols
        self.cta_group = tcgen05.CtaGroup.ONE
        # The MMA itself is single-CTA: no multicast; the launch cluster is along K only.
        self.cluster_shape_mn = (1, 1)
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * 6
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.num_acc_stage = 1
        # Flat elements of one column tile each CTA reduces, at most, per epilogue thread.
        per_cta = (tm * bn + ks - 1) // ks
        self.red_iters = (per_cta + self.threads_per_cta - 1) // self.threads_per_cta

    def _setup(self, a_major, b_major):
        tiled_mma = utils.sm100.make_trivial_tiled_mma(
            self.io_dtype, a_major, b_major, self.acc_dtype, self.cta_group, self.mma_tiler[:2]
        )
        inst_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (self.mma_tiler[0], self.mma_tiler[1], inst_k * 4)
        self.cta_tile_shape_mnk = self.mma_tiler
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.epi_tile = (self.mma_tiler[0], self.epi_cols)
        a_one = utils.sm100.make_smem_layout_a(tiled_mma, self.mma_tiler, self.io_dtype, 1)
        b_one = utils.sm100.make_smem_layout_b(tiled_mma, self.mma_tiler, self.io_dtype, 1)
        ab_bytes = cute.size_in_bytes(self.io_dtype, a_one) + cute.size_in_bytes(
            self.io_dtype, b_one
        )
        smem_capacity = utils.get_smem_capacity_in_bytes()
        self.k_tiles = self.k // self.mma_tiler[2] // self.ks
        # The fp32 partial, one padded row per accumulator row (conflict-free row stores).
        self.p_stride = self.bn + 1
        p_bytes = self.tm * self.p_stride * 4
        self.num_ab_stage = max(2, min(self.k_tiles, (smem_capacity - 2048 - p_bytes) // ab_bytes))
        self.n_pre = min(self.num_ab_stage, self.k_tiles)
        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.io_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.io_dtype, self.num_ab_stage
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(acc_fake, arch="sm_100")
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        m: cutlass.Int32,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        low_t: cute.Tensor,
        inj_ptr: cute.Pointer,
        stream: cuda.CUstream,
    ):
        k, ks = self.k, self.ks
        a = cute.make_tensor(a_ptr, cute.make_layout((m, k, 1), stride=(k, 1, 0)))
        b = cute.make_tensor(b_ptr, cute.make_layout((self.n_cols, k, 1), stride=(k, 1, 0)))
        # `low` arrives as a tensor (the TVM-FFI entry takes its stream from a GPU tensor).
        low = cute.make_tensor(
            low_t.iterator, cute.make_layout((m, self.rank), stride=(self.rank, 1))
        )
        inj = cute.make_tensor(inj_ptr, cute.make_layout((m, self.hc), stride=(self.hc, 1)))
        k_major = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
        tiled_mma = self._setup(k_major, k_major)

        a_op = utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        self.num_tma_load_bytes = cute.size_in_bytes(self.io_dtype, a_smem) + cute.size_in_bytes(
            self.io_dtype, b_smem
        )
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_a,
            tma_atom_b,
            tma_b,
            low,
            inj,
            m,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_tile,
        ).launch(
            grid=[ks, self.n_tiles, (m + self.tm - 1) // self.tm],
            block=[self.threads_per_cta, 1, 1],
            cluster=[ks, 1, 1],
            stream=stream,
            use_pdl=_PDL,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mLow: cute.Tensor,
        mInj: cute.Tensor,
        m: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # The consumer (the up projection, launched with programmatic dependent launch)
        # may start at once: it streams its own static weight, then waits for this grid.
        cute.arch.griddepcontrol_launch_dependents()
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        kr = bidx  # K slice == rank in the (KS, 1, 1) cluster
        nt = bidy  # column tile
        mt = bidz  # M tile of `tm` rows

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        ab_producer, ab_consumer = PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        ).make_participants()
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilogue_warp_id)
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sP = smem.allocate_tensor(
            element_type=self.acc_dtype,
            layout=cute.make_layout((self.tm, self.bn), stride=(self.p_stride, 1)),
            byte_alignment=16,
        )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        k0 = kr * self.k_tiles

        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx == self.tma_warp_id:
            tAgA_slice = tAgA[(None, mt, None, 0)]
            tBgB_slice = tBgB[(None, nt, None, 0)]
            ab_producer.reset()
            # The weight does not depend on the previous kernel: the first stages' weight
            # tiles stream before the grid-dependency wait (programmatic dependent
            # launch), the normed rows -- the previous kernel's output -- after it.
            pre = []
            for kk in cutlass.range_constexpr(self.n_pre):
                handle = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_b,
                    tBgB_slice[(None, k0 + kk)],
                    tBsB[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
                pre.append(handle)
            cute.arch.griddepcontrol_wait()
            for kk in cutlass.range_constexpr(self.n_pre):
                handle = pre[kk]
                cute.copy(
                    tma_atom_a,
                    tAgA_slice[(None, k0 + kk)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
            for k_tile in cutlass.range(self.n_pre, self.k_tiles, 1, unroll=1):
                handle = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA_slice[(None, k0 + k_tile)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_slice[(None, k0 + k_tile)],
                    tBsB[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
            ab_producer.tail()

        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
            ab_consumer.reset()
            peek = ab_consumer.try_wait()
            acc_pipeline.producer_acquire(acc_producer_state)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_tile in cutlass.range(0, self.k_tiles, 1, unroll=1):
                handle = ab_consumer.wait_and_advance(peek)
                num_kblocks = cute.size(tCrA, mode=[2])
                for kb in cutlass.range(num_kblocks, unroll_full=True):
                    kcrd = (None, None, kb, handle.index)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[kcrd], tCrB[kcrd], tCtAcc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                handle.release()
                peek = cutlass.Boolean(1)
                if handle.count + 1 < self.k_tiles:
                    peek = ab_consumer.try_wait()
            acc_pipeline.producer_commit(acc_producer_state)
            acc_producer_state.advance()
            acc_pipeline.producer_tail(acc_producer_state)

        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            copy_atom_t2r = utils.sm100.get_tmem_load_op(
                self.cta_tile_shape_mnk,
                utils.LayoutEnum.ROW_MAJOR,
                self.acc_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            )
            tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
            tTR_sP = thr_copy_t2r.partition_D(cute.flat_divide(sP, epi_tile))
            frag_shape = tTR_sP[(None, None, None, 0, 0)].shape
            tTR_rAcc = cute.make_rmem_tensor(frag_shape, self.acc_dtype)
            tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
            acc_pipeline.consumer_wait(acc_consumer_state)
            for sub in cutlass.range_constexpr(self.bn // self.epi_cols):
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, sub)], tTR_rAcc)
                cute.autovec_copy(tTR_rAcc, tTR_sP[(None, None, None, 0, sub)])
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer_state)
            acc_consumer_state.advance()

            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id,
                number_of_threads=32 * len(self.epilogue_warp_id),
            )
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # Release / acquire at cluster scope, every thread of every CTA: after it, each
        # CTA reads all `KS` partials of its column tile out of its peers' shared memory.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        # This CTA's share of the tile's `rows * BN` outputs (`rows` of this M tile),
        # all 6 warps. Every load is issued before any sum (indices clamped in bounds,
        # so no branch separates them); the `KS` partials are summed in slice order
        # (deterministic).
        row0 = mt * self.tm
        total = cutlass.min(m - row0, self.tm) * self.bn
        per = (total + self.ks - 1) // self.ks
        base = kr * per
        nthr = self.threads_per_cta
        vals = cute.make_rmem_tensor((self.red_iters * self.ks,), self.acc_dtype)
        p_base = cutlass.Int32(sP.iterator.toint())
        peers = [_mapa(p_base, j) for j in range(self.ks)]
        for it in cutlass.range_constexpr(self.red_iters):
            fs = cutlass.min(base + tidx + it * nthr, total - 1)
            row = fs // self.bn
            col = fs - row * self.bn
            off = (row * self.p_stride + col) * 4
            for j in cutlass.range_constexpr(self.ks):
                vals[it * self.ks + j] = _ld_dsmem_f32(peers[j] + off)
        inv_hc = cutlass.Float32(1.0 / self.hc)
        for it in cutlass.range_constexpr(self.red_iters):
            local = tidx + it * nthr
            f = base + local
            if local < per and f < total:
                row = f // self.bn
                col = f - row * self.bn
                n = nt * self.bn + col
                acc = vals[it * self.ks]
                for j in cutlass.range_constexpr(1, self.ks):
                    acc = acc + vals[it * self.ks + j]
                z = acc * inv_hc
                sg = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp(-z))
                if n < self.rank:
                    mLow[(row0 + row, n)] = (z * sg).to(self.io_dtype)
                if cutlass.const_expr(self.with_inject):
                    if n >= self.rank and n < self.rank + self.hc:
                        mInj[(row0 + row, n - self.rank)] = (cutlass.Float32(2.0) * sg).to(
                            self.io_dtype
                        )

        # No CTA may exit (releasing its shared memory) while a peer still reads it.
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()


class HcDecodeTcFirstRowsNKernel:
    """`HcDecodeTcFirstKernel` with the operand roles swapped ("rows on N").
    The WEIGHT rows ride the MMA M tile (`tm` weight rows a CTA, one grid-z tile per `tm`
    of the `n_cols` real rows) and the normed ROWS ride the MMA N (`bn` >= the row count,
    a multiple of 16; TMA zero-fills the rows past `m`), so the tile follows the row count
    in 16-row steps where tcgen05 has no M between 64 and 128. The K slices, the order in
    which a slice's K tiles are accumulated and the order in which the `ks` partials are
    summed are the other kernel's: `low` / `inject` are bitwise its outputs (GB300 micro,
    12 checkpoint sites, 65..96 rows). The weight (static) streams before the
    grid-dependency wait, the normed rows after it, as there. Text below: the other
    kernel's, with `A` / `B` read as swapped.

    `low[t, n] = silu(L[t, n] / C)` for `n < R` and, with `with_inject`,
    `inject[t, j] = 2 * sigmoid(L[t, R + j] / C)`, where `L = A @ B^T`,
    `A = normed [T, K]`, `B = weight rows [0, n_cols)` of `[>= n_cols, K]`.

    Compile-time: `bn` (MMA N, a multiple of 16), `ks` (K split = cluster
    size), `hc`, `rank`, `n_cols`, `k`, `with_inject`, `tm` (the M tile, 128 or 64
    rows). Runtime: the row count (one grid-z M tile per `tm` rows) and the pointers.
    """

    def __init__(self, bn, ks, hc, rank, n_cols, k, with_inject, epi_cols=16, tm=128):
        assert tm in (64, 128)
        self.tm = tm
        assert bn % epi_cols == 0 and bn % 16 == 0 and 16 <= bn <= 256
        assert k % (64 * ks) == 0, "K must split into 64-wide tiles per CTA"
        self.bn, self.ks, self.hc, self.rank, self.n_cols, self.k = bn, ks, hc, rank, n_cols, k
        self.with_inject = with_inject
        self.n_tiles = 1  # rows-on-N: one N tile holds every normed row (bn >= rows)
        self.m_tiles = (n_cols + tm - 1) // tm  # M tiles over the weight rows
        self.acc_dtype = cutlass.Float32
        self.io_dtype = cutlass.BFloat16
        self.mma_tiler = (tm, bn, 1)
        self.epi_cols = epi_cols
        self.cta_group = tcgen05.CtaGroup.ONE
        # The MMA itself is single-CTA: no multicast; the launch cluster is along K only.
        self.cluster_shape_mn = (1, 1)
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * 6
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.num_acc_stage = 1
        # Flat elements of one column tile each CTA reduces, at most, per epilogue thread.
        per_cta = (tm * bn + ks - 1) // ks
        self.red_iters = (per_cta + self.threads_per_cta - 1) // self.threads_per_cta

    def _setup(self, a_major, b_major):
        tiled_mma = utils.sm100.make_trivial_tiled_mma(
            self.io_dtype, a_major, b_major, self.acc_dtype, self.cta_group, self.mma_tiler[:2]
        )
        inst_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (self.mma_tiler[0], self.mma_tiler[1], inst_k * 4)
        self.cta_tile_shape_mnk = self.mma_tiler
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.epi_tile = (self.mma_tiler[0], self.epi_cols)
        a_one = utils.sm100.make_smem_layout_a(tiled_mma, self.mma_tiler, self.io_dtype, 1)
        b_one = utils.sm100.make_smem_layout_b(tiled_mma, self.mma_tiler, self.io_dtype, 1)
        ab_bytes = cute.size_in_bytes(self.io_dtype, a_one) + cute.size_in_bytes(
            self.io_dtype, b_one
        )
        smem_capacity = utils.get_smem_capacity_in_bytes()
        self.k_tiles = self.k // self.mma_tiler[2] // self.ks
        # The fp32 partial, one padded row per accumulator row (conflict-free row stores).
        self.p_stride = self.bn + 1
        p_bytes = self.tm * self.p_stride * 4
        self.num_ab_stage = max(2, min(self.k_tiles, (smem_capacity - 2048 - p_bytes) // ab_bytes))
        self.n_pre = min(self.num_ab_stage, self.k_tiles)
        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.io_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.io_dtype, self.num_ab_stage
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(acc_fake, arch="sm_100")
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        m: cutlass.Int32,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        low_t: cute.Tensor,
        inj_ptr: cute.Pointer,
        stream: cuda.CUstream,
    ):
        k, ks = self.k, self.ks
        # rows-on-N: A (the MMA M operand) is the WEIGHT, B (the N operand) the normed rows.
        a = cute.make_tensor(b_ptr, cute.make_layout((self.n_cols, k, 1), stride=(k, 1, 0)))
        b = cute.make_tensor(a_ptr, cute.make_layout((m, k, 1), stride=(k, 1, 0)))
        # `low` arrives as a tensor (the TVM-FFI entry takes its stream from a GPU tensor).
        low = cute.make_tensor(
            low_t.iterator, cute.make_layout((m, self.rank), stride=(self.rank, 1))
        )
        inj = cute.make_tensor(inj_ptr, cute.make_layout((m, self.hc), stride=(self.hc, 1)))
        k_major = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
        tiled_mma = self._setup(k_major, k_major)

        a_op = utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        self.num_tma_load_bytes = cute.size_in_bytes(self.io_dtype, a_smem) + cute.size_in_bytes(
            self.io_dtype, b_smem
        )
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_a,
            tma_atom_b,
            tma_b,
            low,
            inj,
            m,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_tile,
        ).launch(
            grid=[ks, 1, self.m_tiles],
            block=[self.threads_per_cta, 1, 1],
            cluster=[ks, 1, 1],
            stream=stream,
            use_pdl=_PDL,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mLow: cute.Tensor,
        mInj: cute.Tensor,
        m: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # The consumer (the up projection, launched with programmatic dependent launch)
        # may start at once: it streams its own static weight, then waits for this grid.
        cute.arch.griddepcontrol_launch_dependents()
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        kr = bidx  # K slice == rank in the (KS, 1, 1) cluster
        nt = bidy  # column tile
        mt = bidz  # M tile of `tm` rows

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        ab_producer, ab_consumer = PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        ).make_participants()
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilogue_warp_id)
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sP = smem.allocate_tensor(
            element_type=self.acc_dtype,
            layout=cute.make_layout((self.tm, self.bn), stride=(self.p_stride, 1)),
            byte_alignment=16,
        )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        k0 = kr * self.k_tiles

        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx == self.tma_warp_id:
            tAgA_slice = tAgA[(None, mt, None, 0)]
            tBgB_slice = tBgB[(None, nt, None, 0)]
            ab_producer.reset()
            # The weight does not depend on the previous kernel: the first stages' weight
            # tiles stream before the grid-dependency wait (programmatic dependent
            # launch), the normed rows -- the previous kernel's output -- after it.
            pre = []
            for kk in cutlass.range_constexpr(self.n_pre):
                handle = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA_slice[(None, k0 + kk)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
                pre.append(handle)
            cute.arch.griddepcontrol_wait()
            for kk in cutlass.range_constexpr(self.n_pre):
                handle = pre[kk]
                cute.copy(
                    tma_atom_b,
                    tBgB_slice[(None, k0 + kk)],
                    tBsB[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
            for k_tile in cutlass.range(self.n_pre, self.k_tiles, 1, unroll=1):
                handle = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA_slice[(None, k0 + k_tile)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_slice[(None, k0 + k_tile)],
                    tBsB[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
            ab_producer.tail()

        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
            ab_consumer.reset()
            peek = ab_consumer.try_wait()
            acc_pipeline.producer_acquire(acc_producer_state)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_tile in cutlass.range(0, self.k_tiles, 1, unroll=1):
                handle = ab_consumer.wait_and_advance(peek)
                num_kblocks = cute.size(tCrA, mode=[2])
                for kb in cutlass.range(num_kblocks, unroll_full=True):
                    kcrd = (None, None, kb, handle.index)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[kcrd], tCrB[kcrd], tCtAcc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                handle.release()
                peek = cutlass.Boolean(1)
                if handle.count + 1 < self.k_tiles:
                    peek = ab_consumer.try_wait()
            acc_pipeline.producer_commit(acc_producer_state)
            acc_producer_state.advance()
            acc_pipeline.producer_tail(acc_producer_state)

        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            copy_atom_t2r = utils.sm100.get_tmem_load_op(
                self.cta_tile_shape_mnk,
                utils.LayoutEnum.ROW_MAJOR,
                self.acc_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            )
            tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
            tTR_sP = thr_copy_t2r.partition_D(cute.flat_divide(sP, epi_tile))
            frag_shape = tTR_sP[(None, None, None, 0, 0)].shape
            tTR_rAcc = cute.make_rmem_tensor(frag_shape, self.acc_dtype)
            tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
            acc_pipeline.consumer_wait(acc_consumer_state)
            for sub in cutlass.range_constexpr(self.bn // self.epi_cols):
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, sub)], tTR_rAcc)
                cute.autovec_copy(tTR_rAcc, tTR_sP[(None, None, None, 0, sub)])
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer_state)
            acc_consumer_state.advance()

            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id,
                number_of_threads=32 * len(self.epilogue_warp_id),
            )
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # Release / acquire at cluster scope, every thread of every CTA: after it, each
        # CTA reads all `KS` partials of its column tile out of its peers' shared memory.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        # This CTA's share of the tile's `rows * BN` outputs (`rows` of this M tile),
        # all 6 warps. Every load is issued before any sum (predicated on the thread's share,
        # indices clamped in bounds, so no branch separates them); the `KS` partials are summed
        # in slice order (deterministic).
        row0 = mt * self.tm
        total = cutlass.min(cutlass.Int32(self.n_cols) - row0, self.tm) * self.bn
        per = (total + self.ks - 1) // self.ks
        base = kr * per
        nthr = self.threads_per_cta
        vals = cute.make_rmem_tensor((self.red_iters * self.ks,), self.acc_dtype)
        p_base = cutlass.Int32(sP.iterator.toint())
        peers = [_mapa(p_base, j) for j in range(self.ks)]
        for it in cutlass.range_constexpr(self.red_iters):
            fs = cutlass.min(base + tidx + it * nthr, total - 1)
            row = fs // self.bn
            col = fs - row * self.bn
            off = (row * self.p_stride + col) * 4
            # Only the loads of a step this thread sums (the sum's own condition): every load is
            # consumed before the exit barrier, so none is in flight when the peers exit.
            live = cutlass.Int32(0)
            if tidx + it * nthr < per and base + tidx + it * nthr < total:
                live = cutlass.Int32(1)
            for j in cutlass.range_constexpr(self.ks):
                vals[it * self.ks + j] = _ld_dsmem_f32_if(peers[j] + off, live)
        inv_hc = cutlass.Float32(1.0 / self.hc)
        for it in cutlass.range_constexpr(self.red_iters):
            local = tidx + it * nthr
            f = base + local
            if local < per and f < total:
                row = f // self.bn
                col = f - row * self.bn
                n = row0 + row  # rows-on-N: the weight row; `col` is the normed row
                acc = vals[it * self.ks]
                for j in cutlass.range_constexpr(1, self.ks):
                    acc = acc + vals[it * self.ks + j]
                z = acc * inv_hc
                sg = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp(-z))
                if n < self.rank and col < m:
                    mLow[(col, n)] = (z * sg).to(self.io_dtype)
                if cutlass.const_expr(self.with_inject):
                    if n >= self.rank and n < self.rank + self.hc and col < m:
                        mInj[(col, n - self.rank)] = (cutlass.Float32(2.0) * sg).to(self.io_dtype)

        # No CTA may exit (releasing its shared memory) while a peer still reads it. A relaxed
        # arrive suffices because a thread arrives only after summing every load it issued; a
        # load issued but never summed would still be in flight here, and a peer could exit under
        # it (CUDBG_EXCEPTION_CLUSTER_BLOCK_NOT_PRESENT).
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()


_COMPILED: dict = {}


def _launcher(bn, ks, hc, rank, n_cols, k, with_inject, ptrs, tm=128):
    """Compile once per static configuration; the row count is a runtime argument
    and the TVM-FFI entry launches on the caller's current stream (graph-capturable).
    `ptrs` is `(normed, weight, low tensor, inject)`. `tm`: the M tile in rows."""
    key = (bn, ks, hc, rank, n_cols, k, with_inject, tm)
    fn = _COMPILED.get(key)
    if fn is None:
        kern = HcDecodeTcFirstKernel(bn, ks, hc, rank, n_cols, k, with_inject, tm=tm)
        gmem = cute.AddressSpace.gmem
        cptrs = [
            make_ptr(cutlass.BFloat16, ptrs[0], gmem, assumed_align=16),
            make_ptr(cutlass.BFloat16, ptrs[1], gmem, assumed_align=16),
            from_dlpack(ptrs[2], assumed_align=16).mark_layout_dynamic(leading_dim=1),
            make_ptr(cutlass.BFloat16, ptrs[3], gmem, assumed_align=2),
        ]
        fn = cute.compile(
            kern,
            MAX_ROWS,
            *cptrs,
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _COMPILED[key] = fn
    return fn


def hc_decode_tc_first(normed, first, rank, hc_count, with_inject, bn, ks, tm=128):
    """Launch with an explicit tile (`bn` columns, `ks`-way K split, `tm`-row M tile).
    `first` is the `[K, >= n_cols]` transposed view of the row-major weight, as
    `cublas_mm(normed, first)` takes it. Returns `(low, inject)`."""
    rows, k = normed.shape
    n_cols = rank + (hc_count if with_inject else 0)
    dev, dt = normed.device, normed.dtype
    low = torch.empty((rows, rank), dtype=dt, device=dev)
    inject = torch.empty((rows, hc_count), dtype=dt, device=dev) if with_inject else None
    if rows > 0:
        ptrs = (
            normed.data_ptr(),
            first.data_ptr(),
            low,
            (inject if with_inject else low).data_ptr(),
        )
        _launcher(bn, ks, hc_count, rank, n_cols, k, with_inject, ptrs, tm)(rows, *ptrs)
    return low, inject


#: Rows-on-N plan: `(KS, weight rows a CTA)`. 16-way K split (the 128-row tile's, so the
#: outputs are bitwise), 64 weight rows a CTA: 6 M tiles over the 324 real rows = 96 CTAs.
#: GB300 micro, us a site against the 128-row tile: -0.55 to -0.60 at 65..80 rows (MMA N
#: 80), -0.14 to -0.23 at 81..96 (N 96), slower from 97 rows.
ROWSN_PLAN = (16, 64)
#: Rows the rows-on-N kernel serves in place of the 128-row tile (`Qwen4ExpHcDecodeTcFirst(...,
#: rows_on_n=True)`). It takes any row count up to its last (MMA N from 16); the target also runs
#: it at 8..16 rows in place of the 64-row tile under an opt-in.
ROWSN_ROWS = (65, 96)

_COMPILED_ROWSN: dict = {}


def _launcher_rowsn(bn, ks, hc, rank, n_cols, k, with_inject, ptrs, tm=64):
    """`_launcher` for `HcDecodeTcFirstRowsNKernel` (same operands and pointer order)."""
    key = (bn, ks, hc, rank, n_cols, k, with_inject, tm)
    fn = _COMPILED_ROWSN.get(key)
    if fn is None:
        kern = HcDecodeTcFirstRowsNKernel(bn, ks, hc, rank, n_cols, k, with_inject, tm=tm)
        gmem = cute.AddressSpace.gmem
        cptrs = [
            make_ptr(cutlass.BFloat16, ptrs[0], gmem, assumed_align=16),
            make_ptr(cutlass.BFloat16, ptrs[1], gmem, assumed_align=16),
            from_dlpack(ptrs[2], assumed_align=16).mark_layout_dynamic(leading_dim=1),
            make_ptr(cutlass.BFloat16, ptrs[3], gmem, assumed_align=2),
        ]
        fn = cute.compile(
            kern,
            bn,
            *cptrs,
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _COMPILED_ROWSN[key] = fn
    return fn


def hc_first_rowsn(normed, first, rank, hc_count, with_inject, bn):
    """`hc_decode_tc_first`'s operands and results on the rows-on-N kernel: the normed rows on
    an MMA N of `bn` (a multiple of 16, >= the row count), the plan `ROWSN_PLAN`. Bitwise
    `hc_decode_tc_first(..., 48, 16)` (the 128-row tile's plan at 5..128 rows)."""
    rows, k = normed.shape
    assert 0 < rows <= bn <= 256 and bn % 16 == 0, (rows, bn)
    ks, tm = ROWSN_PLAN
    n_cols = rank + (hc_count if with_inject else 0)
    dev, dt = normed.device, normed.dtype
    low = torch.empty((rows, rank), dtype=dt, device=dev)
    inject = torch.empty((rows, hc_count), dtype=dt, device=dev) if with_inject else None
    ptrs = (
        normed.data_ptr(),
        first.data_ptr(),
        low,
        (inject if with_inject else low).data_ptr(),
    )
    _launcher_rowsn(bn, ks, hc_count, rank, n_cols, k, with_inject, ptrs, tm)(rows, *ptrs)
    return low, inject


def precompile_rowsn(k, rank, hc_count, device, bns, injects=(True,)) -> None:
    """Compile, without launching, the rows-on-N kernel for every MMA N in `bns`, for each
    `with_inject` in `injects` (placeholder operands, as `precompile`)."""
    normed = torch.empty((MAX_ROWS, k), dtype=torch.bfloat16, device=device)
    first = torch.empty((rank + hc_count, k), dtype=torch.bfloat16, device=device)
    low = torch.empty((MAX_ROWS, rank), dtype=torch.bfloat16, device=device)
    ptrs = (normed.data_ptr(), first.data_ptr(), low, low.data_ptr())
    ks, tm = ROWSN_PLAN
    for with_inject in injects:
        n_cols = rank + (hc_count if with_inject else 0)
        for bn in bns:
            _launcher_rowsn(bn, ks, hc_count, rank, n_cols, k, with_inject, ptrs, tm)


#: Tile `(BN, KS)` by row count: `BN` weight rows per CTA (48: 7 tiles cover the
#: 324 real rows of 336) and a `KS`-way K split, one cluster per column tile. One
#: M tile (<= 128 rows) runs 16-way (7 x 16 = 112 CTAs); two M tiles run 8-way, so
#: the grid (7 x 8 x 2 = 112 CTAs) stays one wave of one CTA an SM. Up to 4 rows the
#: split is 10-way (7 x 10 = 70 CTAs).
_TILES = ((4, (48, 10)), (128, (48, 16)), (MAX_ROWS, (48, 8)))
#: Every K split `_TILES` uses (16, 10, 8) divides K into 64-wide tiles per CTA when K
#: does by this.
_K_ALIGN = 64 * 80


#: The 64-row M tile (`tm=64`): `(BN, KS)` and the most rows it takes. One M tile, 7 x 16 =
#: 112 CTAs as the 128-row tile at the same rows, each with half the normed bytes (all 10 K
#: tiles of a slice fit the ring) and half the partial. GB300 micro, us a launch alone:
#: 5.1-5.6 -> 4.4-4.9 at 8-64 rows; two 64-row tiles above 64 rows are slower (224 CTAs, +1.0
#: to +1.5) and not offered.
_TILE64 = (48, 16)
TILE64_MAX_ROWS = 64

#: `tm64x2=True`: two 64-row M tiles on an 8-way K split (7 x 8 x 2 = 112 CTAs, one wave of one
#: CTA an SM), and the rows it serves. GB300 micro, us a launch with the injection against the
#: rows-on-N kernel: 5.41 / 5.46 / 5.47 -> 5.21 / 5.21 / 5.29 at 68 / 72 / 80 rows, 5.74 / 5.80
#: / 5.80 -> 5.29 at 84 / 88 / 96. Another K split than the 128-row tile's (8 slices of 1280
#: columns for 16 of 640): `low` within one bf16 step of the default plan's, `inject` measured
#: bitwise at 68..96 rows.
_TILE64X2 = (48, 8)
TILE64X2_ROWS = (TILE64_MAX_ROWS + 1, 128)


def _tile(rows: int) -> tuple[int, int]:
    return next(t for hi, t in _TILES if rows <= hi)


def precompile(
    k: int, rank: int, hc_count: int, device, tile64: bool = False, tile64x2: bool = False
) -> None:
    """Compile every `(tile, with_inject)` variant `Qwen4ExpHcDecodeTcFirst` can launch
    for a `[*, k]` stream, without launching: a variant first met by a serving step would
    stall it for the compile. Placeholder operands (the compile reads only their
    addresses and the `low` layout). `tile64`: also the two `tm=64` variants;
    `tile64x2`: the two `tm64x2` variants."""
    normed = torch.empty((MAX_ROWS, k), dtype=torch.bfloat16, device=device)
    first = torch.empty((rank + hc_count, k), dtype=torch.bfloat16, device=device)
    low = torch.empty((MAX_ROWS, rank), dtype=torch.bfloat16, device=device)
    ptrs = (normed.data_ptr(), first.data_ptr(), low, low.data_ptr())
    for _, (bn, ks) in _TILES:
        for with_inject in (True, False):
            n_cols = rank + (hc_count if with_inject else 0)
            _launcher(bn, ks, hc_count, rank, n_cols, k, with_inject, ptrs)
    if tile64:
        for with_inject in (True, False):
            n_cols = rank + (hc_count if with_inject else 0)
            _launcher(*_TILE64, hc_count, rank, n_cols, k, with_inject, ptrs, 64)
    if tile64x2:
        for with_inject in (True, False):
            n_cols = rank + (hc_count if with_inject else 0)
            _launcher(*_TILE64X2, hc_count, rank, n_cols, k, with_inject, ptrs, 64)


class Qwen4ExpHcDecodeTcFirst(OpWrapper):
    """With `R = rank`, `C = hc_count`, `normed` the `[T, K]` grouped-normed
    stream and `first` the `[K, >= R (+ C)]` transposed view of the load-time
    `[down; block_inject; 0]` weight, `L = normed @ first` in fp32::

        low = silu(L[:, :R] / C)  # [T, R], normed's dtype
        inject = 2 * sigmoid(L[:, R : R + C] / C)  # [T, C], only with_inject

    The projection accumulates in fp32 (tensor memory, then a fixed-order sum
    of the K-split partials); each activation is fp32 with one rounding.
    Returns fresh `(low, inject)`; `inject` is None exactly when `with_inject`
    is False. Fusion boundary: what `cublas_mm(normed, first, None, fp32)` +
    `qwen4_exp_hc_lowrank_act` compute (cuBLAS's split-K reduce included).
    """

    ARCHS = SERVED_ARCHS

    # spec: rows, inject (True). The checkpoint's geometry throughout: hc_count 4,
    # rank 320, hidden 2560, the padded [384, K] weight, bf16. HC is replicated, so the
    # rows are this rank's tokens at every member.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="1 row: a dep4 rank's smallest captured decode batch, 1 of a tile's 128 rows",
            spec=dict(rows=1),
        ),
        Cell(why="2 rows: one request's MTP-1 verify step at tep4", spec=dict(rows=2)),
        Cell(
            why="4 rows: one request's MTP-3 verify step (ll_tp4 / ll_tep4, concurrency 1)",
            spec=dict(rows=4),
        ),
        Cell(
            why="16 rows: a dep4 rank's padded decode batch at concurrency 128",
            spec=dict(rows=16),
        ),
        Cell(
            why="32 rows: dep4's per-rank graph cap; 4 requests x MTP-7 verify at tep4",
            spec=dict(rows=32),
        ),
        Cell(
            why="128 rows: one full M tile (a short prefill step's rows)",
            spec=dict(rows=128),
        ),
        Cell(why="129 rows: a 1-row second M tile", spec=dict(rows=129)),
        Cell(why="256 rows: MAX_ROWS, two full M tiles", spec=dict(rows=256)),
        Cell(
            why="stack-final mixer: 320 cols, no inject, 4 rows (MTP-3 verify)",
            spec=dict(rows=4, inject=False),
        ),
        Cell(
            why="stack-final mixer: 320 cols, no inject, 16 rows (dep4 decode)",
            spec=dict(rows=16, inject=False),
        ),
        Cell(
            why="stack-final mixer: 320 cols, no inject, 200 rows (two M tiles)",
            spec=dict(rows=200, inject=False),
        ),
        Cell(
            why="MTP draft head mixer: 320 cols, no inject, 1 row (ll draft step)",
            spec=dict(rows=1, inject=False),
        ),
        Cell(
            why="MTP draft head mixer: 320 cols, no inject, 32 rows (dep4 rank batch)",
            spec=dict(rows=32, inject=False),
        ),
    ) + tuple(
        # Attention-DP decode plans (dep4 / dep8 verify rows of 4 a request).
        Cell(why=why, spec=dict(rows=t, inject=inj, **plan))
        for t, inj, plan, why in (
            (5, True, dict(tm=64), "64-row tile, 5 rows: the smallest row count it takes"),
            (36, True, dict(tm=64), "64-row tile, 36 rows: dep8 at 9 requests a rank"),
            (64, True, dict(tm=64), "64-row tile, 64 rows: a full 64-row tile"),
            (36, False, dict(tm=64), "64-row tile, no inject: the stack-final mixer"),
            (65, True, dict(rows_on_n=True), "rows on N, 65 rows: MMA N 80"),
            (80, True, dict(rows_on_n=True), "rows on N, 80 rows: dep8 at 20 requests a rank"),
            (96, True, dict(rows_on_n=True), "rows on N, 96 rows: MMA N 96, the last it serves"),
            (8, True, dict(rows_on_n=True), "rows on N, 8 rows: MMA N 16, dep8 at 2 requests"),
            (65, True, dict(tm64x2=True), "two 64-row tiles, 8-way K, 65 rows: the first"),
            (80, True, dict(tm64x2=True), "two 64-row tiles, 80 rows: dep8 at 20 requests a rank"),
            (96, True, dict(tm64x2=True), "two 64-row tiles, 96 rows: dep8 at 24 requests"),
            (128, False, dict(tm64x2=True), "two 64-row tiles, no inject, 128 rows: the last"),
        )
    )
    note = """
    Tier 6 (CuTe DSL): tiers 1-5 have no GEMM whose epilogue applies an
    activation after a split-K reduction; cuBLAS at N = 384 runs split-K plus
    its own reduce kernel and writes fp32 logits that `qwen4_exp_hc_lowrank_act`
    reads back (three launches).
    Metadata consumed: none. Call sites: `_hc_mixer`, every trunk
    hyper-connection mixer of a step of at most `_HC_TC_FIRST_MAX_ROWS`
    (= MAX_ROWS) rows (every captured decode / MTP verify batch of tp4,
    tep4 and dep4, and short prefill steps), 97 per forward, and with MTP
    the draft layer's three sites per draft step (`_mtp_front`'s and
    `_mtp_tail`'s with the injection, `MTPLayer.shared_head`'s without:
    1 and 4 rows at ll, a rank's batch and its verify rows at dep4), each
    between `qwen4_exp_hc_inject_norm` (or the PLE tail) and
    `qwen4_exp_hc_up_gate_mean`. Preconditions (bf16): `normed`
    `[T, K]` contiguous, T <= 256; `first` the `.t()` view of a row-major
    `[N, K]` weight with N >= R (+ C); 16-byte aligned; K a multiple of
    64 * 80. Only the R (+ C) real weight rows are read (TMA zero-fills
    the tile past them), never the load-time padding.
    Launch: grid (KS, cdiv(R + C, BN), cdiv(T, 128)) CTAs of 6 warps,
    cluster (KS, 1, 1), one CTA per SM (~227 KB shared memory: the K tiles
    of the slice in flight, as many as fit, plus the padded fp32 partial);
    `(BN, KS)` = (48, 10) up to 4 rows (10 x 7 = 70 CTAs), (48, 16) up to 128
    rows (16 x 7 = 112 CTAs), (48, 8) above (8 x 7 x 2 = 112 CTAs). The
    partials are reduced through distributed shared memory between two
    cluster barriers; no global workspace.
    Deterministic (fixed K-slice order). Programmatic dependent launch
    (TRTLLM_ENABLE_PDL=0 turns it off): weight stages before
    `griddepcontrol.wait`, `normed` after it; dependents released at kernel
    start. Graph-capturable (current stream, no host sync, no persistent
    state). Compiles once per (BN, KS, C, R, N, K, inject); `precompile`
    builds every variant at load.
    Certified: bf16, 324 columns with the injection at 1, 2, 4, 16, 32,
    128, 129 and 256 rows, 320 columns without it at 1, 4, 16, 32 and 200
    rows;
    graph replay at 1, 4, 16 and 256 rows.
    `tm=64` (up to 64 rows) and `rows_on_n=True` (`ROWSN_ROWS`) are other plans of the same
    sums: bitwise the default plan's outputs at those rows; cells at 5 / 36 / 64 rows (with
    and without the injection) and at 65 / 80 / 96 rows (with it). `tm64x2=True`
    (`TILE64X2_ROWS`) sums in another order (an 8-way K split): cells at 65 / 80 / 96 rows
    with the injection and 128 without, under the same fp64 gate as the default plan.
    Not certified: hc_count other than 4, an unpadded weight (N = R + C), zero
    rows (no launch; accepted, no cell drives them), other row counts without
    the injection.
    """

    def __call__(
        self,
        normed: torch.Tensor,
        first: torch.Tensor,
        rank: int,
        hc_count: int,
        with_inject: bool,
        tm: int = 128,
        rows_on_n: bool = False,
        tm64x2: bool = False,
    ):
        if tm64x2:
            assert TILE64X2_ROWS[0] <= normed.shape[0] <= TILE64X2_ROWS[1], normed.shape
            return hc_decode_tc_first(normed, first, rank, hc_count, with_inject, *_TILE64X2, 64)
        if rows_on_n:
            bn = -(-normed.shape[0] // 16) * 16
            return hc_first_rowsn(normed, first, rank, hc_count, with_inject, bn)
        if tm == 64:
            assert normed.shape[0] <= TILE64_MAX_ROWS, normed.shape
            return hc_decode_tc_first(normed, first, rank, hc_count, with_inject, *_TILE64, 64)
        assert tm == 128, tm
        return hc_decode_tc_first(
            normed, first, rank, hc_count, with_inject, *_tile(normed.shape[0])
        )

    def reference(
        self,
        normed: torch.Tensor,
        first: torch.Tensor,
        rank: int,
        hc_count: int,
        with_inject: bool,
        tm: int = 128,
        rows_on_n: bool = False,
        tm64x2: bool = False,
    ):
        """float64 projection and activations, one rounding per output. `tm`, `rows_on_n` and
        `tm64x2` move no value: ignored."""
        n_cols = rank + (hc_count if with_inject else 0)
        lg = normed.double() @ first[:, :n_cols].double()
        low = F.silu(lg[:, :rank] / hc_count).to(normed.dtype)
        inject = None
        if with_inject:
            inject = (2.0 * torch.sigmoid(lg[:, rank:n_cols] / hc_count)).to(normed.dtype)
        return low, inject

    def is_valid(
        self,
        normed: torch.Tensor,
        first: torch.Tensor,
        rank: int,
        hc_count: int,
        with_inject: bool,
        tm: int = 128,
        rows_on_n: bool = False,
        tm64x2: bool = False,
    ) -> None:
        rows, k = normed.shape
        assert tm == 128 or (tm == 64 and rows <= TILE64_MAX_ROWS), "tm: 128, or 64 up to 64 rows"
        assert not rows_on_n or 1 <= rows <= ROWSN_ROWS[1], "rows on N: 1..96 rows"
        assert not tm64x2 or TILE64X2_ROWS[0] <= rows <= TILE64X2_ROWS[1], (
            "two 64-row tiles: 65..128 rows"
        )
        assert not (tm64x2 and (rows_on_n or tm != 128)), "one plan a call"
        n_cols = rank + (hc_count if with_inject else 0)
        assert normed.dtype == first.dtype == torch.bfloat16, "bf16 operands"
        assert rows <= MAX_ROWS, f"at most {MAX_ROWS} rows (two MMA tiles)"
        assert normed.stride() == (k, 1), "normed contiguous [T, K]"
        assert first.shape[0] == k and first.shape[1] >= n_cols, "first [K, >= n_cols]"
        assert first.stride() == (1, k), "first must be the .t() view of a row-major [N, K] weight"
        assert k % _K_ALIGN == 0, f"K must be a multiple of {_K_ALIGN}"
        for t in (normed, first):
            assert t.data_ptr() % 16 == 0, "16-byte aligned operands"


qwen4_exp_hc_decode_tc_first = Qwen4ExpHcDecodeTcFirst()
