# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-connection mixer first half at decode row counts (1..640) as one tcgen05 split-K
GEMM: `normed @ [down; block_inject]^T` with the SiLU and injection activations in its
epilogue. The NVFP4 default target's kernel; HC is replicated, so only the rows vary.

A CTA computes a `[128, BN]` partial over `K / KS` (grid z is the M tile): a TMA warp streams
the normed rows (past `T` zero-filled) and the `n_cols` real weight rows (the padding is never
read), one MMA warp accumulates in tensor memory, four epilogue warps push the fp32 partial
into the shared memory of the cluster CTA that reduces those rows; after one cluster barrier
each CTA sums its rows' `KS` partials in K-slice order (deterministic). PDL: the weight
streams before the grid-dependency wait, and the consumer is released at once.
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

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.kernels.blackwell.custom_pipeline import PipelineTmaUmma, PipelineUmmaAsync
from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

#: Largest row count this entry is certified for: five 128-row MMA tiles.
MAX_ROWS = 640

#: Programmatic dependent launch (sm_90+), off with TRTLLM_ENABLE_PDL=0.
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
def _sts_v4_f32(addr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [cutlass.Int32(addr)] + [cutlass.Float32(v) for v in (a, b, c, d)],
        "{ st.shared.v4.f32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0; }",
        "=r,r,f,f,f,f",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _bulk_to_peer(dst, src, nbytes, mbar, peer, *, loc=None, ip=None):
    """`nbytes` of this CTA's shared memory at `src` into cluster CTA `peer`'s shared
    memory at `dst`, completing on `peer`'s mbarrier at `mbar` (`dst` / `mbar` are offsets
    in this CTA's window: every CTA has the same layout)."""
    _asm(
        T.i32(),
        [cutlass.Int32(x) for x in (dst, src, nbytes, mbar, peer)],
        "{ .reg .b32 rd, rb; mapa.shared::cluster.u32 rd, $1, $5; "
        "mapa.shared::cluster.u32 rb, $4, $5; "
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes "
        "[rd], [$2], $3, [rb]; mov.u32 $0, 0; }",
        "=r,r,r,r,r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _bulk_commit_wait_read(*, loc=None, ip=None):
    """Commit this thread's bulk copies and wait until they have read their sources."""
    _asm(
        T.i32(),
        [],
        "{ cp.async.bulk.commit_group; cp.async.bulk.wait_group.read 0; mov.u32 $0, 0; }",
        "=r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _lds_v4_f32(addr, *, loc=None, ip=None):
    """16 bytes of this CTA's shared memory; side-effecting, so it stays behind the
    cluster barrier that makes them valid."""
    ty = T.f32()
    res = _asm(
        llvm.StructType.get_literal([ty] * 4),
        [cutlass.Int32(addr)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        loc,
        ip,
        side_effects=True,
    )
    return [cutlass.Float32(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _bf16x2(lo, hi, *, loc=None, ip=None):
    """Two fp32 values rounded (RN, as `.to(BFloat16)`) into one bf16x2 word, `lo` low."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Float32(hi), cutlass.Float32(lo)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _stg_v2(addr, a, b, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [cutlass.Int64(addr), cutlass.Int32(a), cutlass.Int32(b)],
        "{ st.global.v2.b32 [$1], {$2, $3}; mov.u32 $0, 0; }",
        "=r,l,r,r",
        loc,
        ip,
        side_effects=True,
    )


class HcDecodeTcFirstKernel:
    """`Qwen4ExpHcDecodeTcFirst`'s kernel. Compile-time: `bn` (MMA N, a multiple of 16),
    `ks` (K split = cluster size), `hc`, `rank`, `n_cols`, `k`, `with_inject`; runtime: the
    row count (one grid-z M tile per 128 rows) and the pointers."""

    def __init__(self, bn, ks, hc, rank, n_cols, k, with_inject, epi_cols=16):
        assert bn % epi_cols == 0 and bn % 16 == 0 and 16 <= bn <= 256
        assert k % (64 * ks) == 0, "K must split into 64-wide tiles per CTA"
        assert rank % 4 == 0 and hc % 4 == 0, "the reduction writes 4-column groups"
        self.bn, self.ks, self.hc, self.rank, self.n_cols, self.k = bn, ks, hc, rank, n_cols, k
        self.with_inject = with_inject
        self.n_tiles = (n_cols + bn - 1) // bn
        self.acc_dtype = cutlass.Float32
        self.io_dtype = cutlass.BFloat16
        self.mma_tiler = (128, bn, 1)
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
        # Accumulator rows of a column tile each cluster CTA reduces (the last may own fewer).
        self.rows_per = (128 + ks - 1) // ks
        # 4-column groups a CTA reduces, at most, per thread (all six warps).
        groups = self.rows_per * (bn // 4)
        self.red_iters = (groups + self.threads_per_cta - 1) // self.threads_per_cta

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
        # The receive buffer: for each K slice j (row block j) the `rows_per` accumulator rows
        # this CTA reduces, as its peer j pushed them, padded by 16 bytes a row.
        self.p_stride = self.bn + 4
        p_bytes = self.ks * self.rows_per * self.p_stride * 4
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
            grid=[ks, self.n_tiles, (m + 127) // 128],
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
        # The dependent may start now: it streams its own weight, then waits for this grid.
        cute.arch.griddepcontrol_launch_dependents()
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        kr = bidx  # K slice == rank in the (KS, 1, 1) cluster
        nt = bidy  # column tile
        mt = bidz  # 128-row M tile

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            recv_full: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # Completes once every K slice's partial of this CTA's rows has landed.
        if tidx == 0:
            cute.arch.mbarrier_init(storage.recv_full, 1)
        cute.arch.mbarrier_init_fence()
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
            layout=cute.make_layout(self.ks * self.rows_per * self.p_stride),
            byte_alignment=16,
        )
        p_base = cutlass.Int32(sP.iterator.toint())
        row0 = mt * 128
        nreal = cutlass.min(m - row0, 128)

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
        # Start-up cluster barrier (split): a peer pushes partials into this CTA only after
        # its wait, i.e. after this CTA initialized `recv_full`.
        cute.arch.cluster_arrive_relaxed()
        if tidx == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(
                storage.recv_full, self.ks * self._rows_of(kr, nreal) * self.p_stride * 4
            )

        if warp_idx == self.tma_warp_id:
            tAgA_slice = tAgA[(None, mt, None, 0)]
            tBgB_slice = tBgB[(None, nt, None, 0)]
            ab_producer.reset()
            # The first stages' weight tiles stream before the grid-dependency wait; the
            # normed rows, the previous kernel's output, after it.
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
            tTR_cP = thr_copy_t2r.partition_D(
                cute.flat_divide(cute.make_identity_tensor((128, self.bn)), epi_tile)
            )
            tTR_rAcc = cute.make_rmem_tensor(tTR_cP[(None, None, None, 0, 0)].shape, self.acc_dtype)
            tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
            # 32dp32b: warp w reads TMEM lanes [32w, 32w + 32), so this thread holds accumulator
            # row `tidx`. Rows are staged in row order in the operand ring (dead once the
            # accumulator is complete), so each owner's rows are one contiguous block.
            send_base = cutlass.Int32(sA.iterator.toint())
            acc_pipeline.consumer_wait(acc_consumer_state)
            for sub in cutlass.range_constexpr(self.bn // self.epi_cols):
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, sub)], tTR_rAcc)
                if tidx < nreal:
                    for q in cutlass.range_constexpr(self.epi_cols // 4):
                        _sts_v4_f32(
                            send_base + (tidx * self.p_stride + sub * self.epi_cols + 4 * q) * 4,
                            tTR_rAcc[4 * q],
                            tTR_rAcc[4 * q + 1],
                            tTR_rAcc[4 * q + 2],
                            tTR_rAcc[4 * q + 3],
                        )
            cute.arch.fence_proxy("async.shared", space="cta")
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer_state)
            acc_consumer_state.advance()

            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id,
                number_of_threads=32 * len(self.epilogue_warp_id),
            )
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
            # Peers' `recv_full` are initialized (the start-up barrier; warp-aligned wait).
            cute.arch.cluster_wait()
            if tidx == 0:
                # One bulk copy per owner (this CTA included): its rows, into its block `kr`.
                for o in cutlass.range_constexpr(self.ks):
                    rows_o = self._rows_of(o, nreal)
                    if rows_o > 0:
                        _bulk_to_peer(
                            p_base + kr * self.rows_per * self.p_stride * 4,
                            send_base + o * self.rows_per * self.p_stride * 4,
                            rows_o * self.p_stride * 4,
                            storage.recv_full.toint(),
                            o,
                        )
        else:
            cute.arch.cluster_wait()

        # Every K slice's partial of this CTA's rows is local once `recv_full` completes;
        # no DSMEM access follows (a CTA may exit once its own copies have read their source).
        cute.arch.mbarrier_wait(storage.recv_full, 0)

        # This CTA's rows [kr * rows_per, ...) of the tile's real rows, in 4-column groups,
        # all 6 warps; an element's `KS` partials summed in K-slice order (deterministic).
        r_lo = kr * self.rows_per
        mine = self._rows_of(kr, nreal)
        g4 = self.bn // 4
        groups = mine * g4
        inv_hc = cutlass.Float32(1.0 / self.hc)
        low_base = mLow.iterator.toint()
        inj_base = mInj.iterator.toint()
        for it in cutlass.range_constexpr(self.red_iters):
            g = tidx + it * self.threads_per_cta
            if g < groups:
                rl = g // g4
                c = (g - rl * g4) * 4
                src = p_base + (rl * self.p_stride + c) * 4
                parts = [
                    _lds_v4_f32(src + j * self.rows_per * self.p_stride * 4) for j in range(self.ks)
                ]
                n = nt * self.bn + c
                row = cutlass.Int64(row0 + r_lo + rl)
                lows, injs = [], []
                for e in cutlass.range_constexpr(4):
                    acc = parts[0][e]
                    for j in cutlass.range_constexpr(1, self.ks):
                        acc = acc + parts[j][e]
                    z = acc * inv_hc
                    sg = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp(-z))
                    lows.append(z * sg)
                    injs.append(cutlass.Float32(2.0) * sg)
                if n < self.rank:
                    _stg_v2(
                        low_base + (row * self.rank + n) * 2,
                        _bf16x2(lows[0], lows[1]),
                        _bf16x2(lows[2], lows[3]),
                    )
                if cutlass.const_expr(self.with_inject):
                    if n >= self.rank and n < self.rank + self.hc:
                        _stg_v2(
                            inj_base + (row * self.hc + n - self.rank) * 2,
                            _bf16x2(injs[0], injs[1]),
                            _bf16x2(injs[2], injs[3]),
                        )
        if tidx == 0:
            _bulk_commit_wait_read()

    def _rows_of(self, owner, nreal):
        """Real rows of the tile that cluster CTA `owner` reduces (0 past the real rows)."""
        lo = owner * self.rows_per
        hi = cutlass.min(lo + self.rows_per, 128)
        return cutlass.max(cutlass.min(nreal, hi) - lo, 0)


_COMPILED: dict = {}


def _launcher(bn, ks, hc, rank, n_cols, k, with_inject, ptrs):
    """Compile once per static configuration (the row count is a runtime argument); the
    entry launches on the caller's current stream. `ptrs`: `(normed, weight, low, inject)`."""
    key = (bn, ks, hc, rank, n_cols, k, with_inject)
    fn = _COMPILED.get(key)
    if fn is None:
        kern = HcDecodeTcFirstKernel(bn, ks, hc, rank, n_cols, k, with_inject)
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


def hc_decode_tc_first(normed, first, rank, hc_count, with_inject, bn, ks):
    """Launch with an explicit tile (`bn` columns, `ks`-way K split). `first` is
    the `[K, >= n_cols]` transposed view of the row-major weight, as
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
        _launcher(bn, ks, hc_count, rank, n_cols, k, with_inject, ptrs)(rows, *ptrs)
    return low, inject


#: Tile `(BN, KS)` by row count: 48 weight rows per CTA (7 tiles cover the 324 real rows)
#: and a `KS`-way K split, one cluster per column tile, so the grid stays about one wave of
#: one CTA an SM (70 to 140 CTAs).
_TILES = (
    (4, (48, 10)),
    (128, (48, 16)),
    (256, (48, 8)),
    (384, (48, 5)),
    (512, (48, 4)),
    (MAX_ROWS, (48, 4)),
)
#: Every `_TILES` K split divides K into 64-wide tiles per CTA when K divides by this.
_K_ALIGN = 64 * 80


def _tile(rows: int) -> tuple[int, int]:
    return next(t for hi, t in _TILES if rows <= hi)


def precompile(k: int, rank: int, hc_count: int, device) -> None:
    """Compile every `(tile, with_inject)` variant for a `[*, k]` stream without launching,
    so no serving step stalls on a compile. The placeholder operands' contents are unread."""
    normed = torch.empty((MAX_ROWS, k), dtype=torch.bfloat16, device=device)
    first = torch.empty((rank + hc_count, k), dtype=torch.bfloat16, device=device)
    low = torch.empty((MAX_ROWS, rank), dtype=torch.bfloat16, device=device)
    ptrs = (normed.data_ptr(), first.data_ptr(), low, low.data_ptr())
    for _, (bn, ks) in _TILES:
        for with_inject in (True, False):
            n_cols = rank + (hc_count if with_inject else 0)
            _launcher(bn, ks, hc_count, rank, n_cols, k, with_inject, ptrs)


class Qwen4ExpHcDecodeTcFirst(OpWrapper):
    """With `R = rank`, `C = hc_count`, `normed` the `[T, K]` grouped-normed
    stream and `first` the `[K, >= R (+ C)]` transposed view of the load-time
    `[down; block_inject; 0]` weight, `L = normed @ first` in fp32::

        low = silu(L[:, :R] / C)  # [T, R], normed's dtype
        inject = 2 * sigmoid(L[:, R : R + C] / C)  # [T, C], only with_inject

    fp32 accumulation (a fixed-order sum of the K-split partials), each activation rounded
    once. Returns fresh `(low, inject)`; `inject` is None exactly when `with_inject` is
    False. Fusion boundary: `cublas_mm(normed, first, None, fp32)` + `qwen4_exp_hc_lowrank_act`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, inject (True); the checkpoint's geometry (hc 4, rank 320, the padded
    # [384, K] weight, bf16). With inject one cell per `_TILES` bucket at partial and full
    # M tiles; without, three.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="1 row: a one-request decode step without MTP (tp1..dep4), a draft pass",
            spec=dict(rows=1),
        ),
        Cell(
            why="16 rows: ll_tep4's 4-request MTP-3 verify step; a dep4 rank's draft pass",
            spec=dict(rows=16),
        ),
        Cell(
            why="128 rows: ht_dep4's 32-request rank graph cap (MTP-3 verify), one M tile",
            spec=dict(rows=128),
        ),
        Cell(why="129 rows: a 1-row second M tile", spec=dict(rows=129)),
        Cell(
            why="288 rows: tep2 gen 96-request MTP-2 verify step, three M tiles",
            spec=dict(rows=288),
        ),
        Cell(
            why="480 rows: tep2 gen 160-request MTP-2 verify step, four M tiles",
            spec=dict(rows=480),
        ),
        Cell(why="640 rows: MAX_ROWS, five full M tiles", spec=dict(rows=640)),
        Cell(
            why="stack-final mixer: 320 cols, no inject, 200 rows (two M tiles)",
            spec=dict(rows=200, inject=False),
        ),
        Cell(
            why="stack-final mixer: 320 cols, no inject, 384 rows (tep2 gen b128 verify)",
            spec=dict(rows=384, inject=False),
        ),
        Cell(
            why="MTP draft head mixer: 320 cols, no inject, 1 row (ll draft step)",
            spec=dict(rows=1, inject=False),
        ),
    )
    note = """
    Source: `HcDecodeTcFirstKernel` above (CuTe DSL). Preconditions (bf16): `normed` `[T, K]`
    contiguous, T <= 640; `first` the `.t()` view of a row-major `[N, K]` weight with
    N >= R (+ C); 16-byte aligned; K a multiple of 64 * 80. Only the R (+ C) real weight rows
    are read. One CTA an SM (~227 KB shared memory); the partials meet in distributed shared
    memory, no global workspace. PDL off with TRTLLM_ENABLE_PDL=0. Graph-capturable (current
    stream, no host sync, no persistent state); compiles once per (BN, KS, C, R, N, K,
    inject), so call `precompile` before capture.
    Certified on GB300 at the cells' rows, graph replay bitwise the eager launch.
    Not certified: hc_count other than 4, an unpadded weight (N = R + C), zero rows.
    """

    def __call__(
        self,
        normed: torch.Tensor,
        first: torch.Tensor,
        rank: int,
        hc_count: int,
        with_inject: bool,
    ):
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
    ):
        """float64 projection and activations, one rounding per output."""
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
    ) -> None:
        rows, k = normed.shape
        n_cols = rank + (hc_count if with_inject else 0)
        assert normed.dtype == first.dtype == torch.bfloat16, "bf16 operands"
        assert rows <= MAX_ROWS, f"at most {MAX_ROWS} rows (five MMA tiles)"
        assert normed.stride() == (k, 1), "normed contiguous [T, K]"
        assert first.shape[0] == k and first.shape[1] >= n_cols, "first [K, >= n_cols]"
        assert first.stride() == (1, k), "first must be the .t() view of a row-major [N, K] weight"
        assert k % _K_ALIGN == 0, f"K must be a multiple of {_K_ALIGN}"
        for t in (normed, first):
            assert t.data_ptr() % 16 == 0, "16-byte aligned operands"


qwen4_exp_hc_decode_tc_first = Qwen4ExpHcDecodeTcFirst()
