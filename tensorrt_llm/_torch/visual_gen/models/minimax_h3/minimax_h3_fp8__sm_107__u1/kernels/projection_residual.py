# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 projection GEMM with a gated residual epilogue, for the two projections back to the hidden width.

``out = bf16(hidden + gate * bf16(scale_a * scale_w * (activation @ weight.T)))`` in one kernel: FP32
accumulation, the GEMM result rounded to BF16, then the gated residual as one FP32 FMA rounded to
BF16. ``gate`` is a column block of the AdaLN modulation table, selected per row by ``indices``;
``gate_column`` names the block (2 for the attention output, 5 for the MLP down projection).
"""

from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.rubin_helpers as sm107_helpers
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05 import CollectorOp
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout


@dsl_user_op
def _scale_to_bf16x2(a0, a1, s_ab, *, loc=None, ip=None):
    """(acc0, acc1) -> bf16x2 {lo: bf16_rn(acc0*s), hi: bf16_rn(acc1*s)} (the mandatory BF16 GEMM rounding)."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, s_ab)],
            "{\n.reg .f32 x0, x1;\n"
            "mul.rn.f32 x0, $1, $3;\n"
            "mul.rn.f32 x1, $2, $3;\n"
            "cvt.rn.bf16x2.f32 $0, x1, x0;\n}\n",
            "=r,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _st_shared_zero(ptr, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [ptr.toint(loc=loc, ip=ip).ir_value()],
        "st.shared.u32 [$0], 0;",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _ld_shared_volatile(ptr, *, loc=None, ip=None):
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [ptr.toint(loc=loc, ip=ip).ir_value()],
            "ld.volatile.shared.u32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class ProjectionBreuseKernel:
    def __init__(
        self,
        n_dim,
        k_dim,
        cluster_shape_mn=(4, 1),
        num_ab_stage=4,
        swizzle_size=8,
        fallback_cluster_shape_mn=(2, 1),
        mma_tile_k=128,
        tile_n=256,
        gate_column=2,
    ):
        self.n_dim = n_dim
        self.gate_column = gate_column
        self.k_dim = k_dim
        self.acc_dtype = cutlass.Float32
        self.cta_group = tcgen05.CtaGroup.TWO
        self.mma_inst_shape = (256, tile_n, 64)
        self.mma_tiler = (512, tile_n, mma_tile_k)  # 2 x inst M -> Bkeep/Breuse
        self.cluster_shape_mn = cluster_shape_mn
        self.fallback_cluster_shape_mn = fallback_cluster_shape_mn
        self.num_ab_stage = num_ab_stage
        self.num_tmem_alloc_cols = 2 * tile_n
        self.epi_tile = (128, 32)
        self.num_epi_warps = 8
        self.epilogue_warp_id = tuple(range(self.num_epi_warps))
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.sched_warp_id = 10
        self.threads_per_cta = 32 * 11
        self.num_clc_response_bytes = 16
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        self.arch = "sm_107"
        self.swizzle_size = swizzle_size

    def _make_tiled_mma(self, b_collector_op=CollectorOp.DISCARD):
        # No M permutation: the 512-row MMA tile is 2 atoms of 256 rows; within atom c the
        # pair's CTA v owns rows 256c + 128v + [0, 128) (keep acc c=0, reuse acc c=1).
        return sm107_helpers.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_inst_shape,
            permutation_mnk=(1, 1, 1),
            b_collector_op=b_collector_op,
        )

    @cute.jit
    def __call__(
        self,
        act: cute.Tensor,
        wgt: cute.Tensor,
        s_in: cute.Tensor,
        s_w: cute.Tensor,
        aux: Tuple[cute.Tensor, cute.Tensor, cute.Tensor],
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.a_dtype = act.element_type
        self.b_dtype = wgt.element_type

        tiled_mma = self._make_tiled_mma()
        tiled_mma_bkeep = self._make_tiled_mma(CollectorOp.FILL)
        tiled_mma_breuse = self._make_tiled_mma(CollectorOp.LASTUSE)

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.fb_cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.fallback_cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )

        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        a_smem_layout = cute.select(
            self.a_smem_layout_staged, mode=list(range(cute.rank(self.a_smem_layout_staged) - 1))
        )
        b_smem_layout = cute.select(
            self.b_smem_layout_staged, mode=list(range(cute.rank(self.b_smem_layout_staged) - 1))
        )

        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id),
            act,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id),
            wgt,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        fb_tma_atom_a, fb_tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            utils.sm100.cluster_shape_to_tma_atom_A(
                self.fallback_cluster_shape_mn, tiled_mma.thr_id
            ),
            act,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.fb_cluster_layout_vmnk.shape,
        )
        fb_tma_atom_b, fb_tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            utils.sm100.cluster_shape_to_tma_atom_B(
                self.fallback_cluster_shape_mn, tiled_mma.thr_id
            ),
            wgt,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.fb_cluster_layout_vmnk.shape,
        )
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * cute.size(tiled_mma.thr_id.shape)

        # CTA tiles: 256 rows (two 128-row A chunks) x 256 N; one 2-CTA pair per 512-row MMA tile.
        m = act.shape[0]
        num_ctas_m = (m + self.mma_tiler[0] - 1) // self.mma_tiler[0] * 2
        num_ctas_mnl = (num_ctas_m, self.n_dim // self.mma_tiler[1], 1)
        self.tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            num_ctas_mnl,
            (*self.cluster_shape_mn, 1),
            1,
            True,
            fallback_cluster_shape_mnk=(*self.fallback_cluster_shape_mn, 1),
        )
        grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(self.tile_sched_params)

        self.kernel(
            tiled_mma,
            tiled_mma_bkeep,
            tiled_mma_breuse,
            (tma_atom_a, fb_tma_atom_a),
            (tma_tensor_a, fb_tma_tensor_a),
            (tma_atom_b, fb_tma_atom_b),
            (tma_tensor_b, fb_tma_tensor_b),
            out,
            s_in,
            s_w,
            aux,
            (self.cluster_layout_vmnk, self.fb_cluster_layout_vmnk),
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            fallback_cluster=(*self.fallback_cluster_shape_mn, 1),
            stream=stream,
            smem_merge_branch_allocs=True,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_bkeep: cute.TiledMma,
        tiled_mma_breuse: cute.TiledMma,
        tma_atom_a: Tuple[cute.CopyAtom, cute.CopyAtom],
        mA_mk: Tuple[cute.Tensor, cute.Tensor],
        tma_atom_b: Tuple[cute.CopyAtom, cute.CopyAtom],
        mB_nk: Tuple[cute.Tensor, cute.Tensor],
        mOut: cute.Tensor,
        s_in: cute.Tensor,
        s_w: cute.Tensor,
        aux: Tuple[cute.Tensor, cute.Tensor, cute.Tensor],
        cluster_layout_vmnk: Tuple[cute.Layout, cute.Layout],
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
    ):
        cbdim_x, cbdim_y, cbdim_z = cute.arch.block_in_cluster_dim()
        is_preferred_cluster = (
            cbdim_x == self.cluster_shape_mn[0]
            and cbdim_y == self.cluster_shape_mn[1]
            and cbdim_z == 1
        )
        if is_preferred_cluster:
            self.kernel_impl(
                tiled_mma,
                tiled_mma_bkeep,
                tiled_mma_breuse,
                tma_atom_a[0],
                mA_mk[0],
                tma_atom_b[0],
                mB_nk[0],
                mOut,
                s_in,
                s_w,
                aux,
                cluster_layout_vmnk[0],
                a_smem_layout_staged,
                b_smem_layout_staged,
                tile_sched_params,
                self.cluster_shape_mn,
            )
        else:
            self.kernel_impl(
                tiled_mma,
                tiled_mma_bkeep,
                tiled_mma_breuse,
                tma_atom_a[1],
                mA_mk[1],
                tma_atom_b[1],
                mB_nk[1],
                mOut,
                s_in,
                s_w,
                aux,
                cluster_layout_vmnk[1],
                a_smem_layout_staged,
                b_smem_layout_staged,
                tile_sched_params,
                self.fallback_cluster_shape_mn,
            )

    @cute.jit
    def _remap_tile(self, tile_idx, n_px):
        """Map a CLC work id to a swizzled CTA tile.

        swizzled (CTA M tile, N tile): consecutive slots walk `swizzle_size` N clusters, then M.
        """
        cm_sz = self.cluster_shape_mn[0]
        cn_sz = self.cluster_shape_mn[1]
        tx = tile_idx[0]
        ty = tile_idx[1]
        px = tx // cm_sz
        py = ty // cn_sz
        slot = px + n_px * py
        cn_in = slot % self.swizzle_size
        t = slot // self.swizzle_size
        cm = t % n_px
        cn = (t // n_px) * self.swizzle_size + cn_in
        return cm * cm_sz + (tx - px * cm_sz), cn * cn_sz + (ty - py * cn_sz)

    @cute.jit
    def kernel_impl(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_bkeep: cute.TiledMma,
        tiled_mma_breuse: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nk: cute.Tensor,
        mOut: cute.Tensor,
        s_in: cute.Tensor,
        s_w: cute.Tensor,
        aux: Tuple[cute.Tensor, cute.Tensor, cute.Tensor],
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        cluster_shape_mn: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_first_cta_in_cluster = cta_rank_in_cluster == 0
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]
            tinfo_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 4]
            tinfo: cute.struct.MemRange[cutlass.Int32, 4 * 4]
            zero_word: cutlass.Int32

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        zero_ptr = storage.zero_word.ptr
        if warp_idx == self.tma_warp_id:
            with cute.arch.elect_one():
                _st_shared_zero(zero_ptr)

        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Warp),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            enable_multicast_signaling=True,
            defer_sync=True,
        ).make_participants()

        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epi_warps * 2),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * (1 + cluster_size * 2),
            ),
            tx_count=self.num_clc_response_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        clc_response_ptr = storage.clc_response.data_ptr()
        clc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

        # Decouple the eight epilogue warps from CLC.  The MMA warp publishes the
        # already-remapped coordinates of each accumulator through a four-deep CTA ring.
        tinfo_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tinfo_mbar_ptr.data_ptr(),
            num_stages=4,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epi_warps),
            defer_sync=True,
        )
        tinfo = cute.make_tensor(storage.tinfo.data_ptr(), cute.make_layout((4, 4), stride=(1, 4)))

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * (1 + self.num_epi_warps),
        )
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_dealloc_sync_bar_id,
            num_threads=32 * self.num_epi_warps,
        )
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch=self.arch,
        )

        pipeline_init_arrive(cluster_shape_mn=cluster_shape_mn, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        a_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

        # (bM, bK, RestM, RestK)
        gA_mk = cute.local_tile(mA_mk, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None))
        # (bN, bK, RestN, RestK)
        gB_nk = cute.local_tile(mB_nk, cute.slice_(self.mma_tiler, (0, None, None)), (None, None))
        k_tile_cnt = cute.size(gA_mk, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mk)
        tCgB = thr_mma.partition_B(gB_nk)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M=2, MMA_N=1): keep acc = [.., 0, 0], reuse acc = [.., 1, 0]
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        pipeline_init_wait(cluster_shape_mn=cluster_shape_mn)

        # TMEM: warp 0 allocates; a CTA-wide barrier (every warp, same program point) publishes
        # the address before the MMA / epilogue warps retrieve it.
        tmem.allocate(self.num_tmem_alloc_cols)  # executed by the allocator warp only
        cute.arch.barrier()

        tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim(), clc_response_ptr
        )
        work_tile = tile_sched.initial_work_tile_info()
        n_px = (
            (mOut.shape[0] + self.mma_tiler[0] - 1) // self.mma_tiler[0] * 2
            + self.cluster_shape_mn[0]
            - 1
        ) // self.cluster_shape_mn[0]

        # ---------------- TMA producer warp ----------------
        if warp_idx == self.tma_warp_id:
            while work_tile.is_valid_tile:
                m_cta, n_t = self._remap_tile(work_tile.tile_idx, n_px)
                tAgA_slice = tAgA[(None, m_cta // 2, None)]
                tBgB_slice = tBgB[(None, n_t, None)]

                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            ab_producer.tail()

        # ---------------- CLC scheduler warp (first CTA of each cluster) ----------------
        if warp_idx == self.sched_warp_id and is_first_cta_in_cluster:
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer, 1
            )
            while work_tile.is_valid_tile:
                clc_pipeline.producer_acquire(clc_producer_state)
                mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
                tile_sched.advance_to_next_work(mbarrier_addr)
                clc_producer_state.advance()
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            clc_pipeline.producer_tail(clc_producer_state)

        # ---------------- MMA warp (Bkeep / Breuse) ----------------
        if warp_idx == self.mma_warp_id:
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCtAcc_keep = tCtAcc[(None, 0, 0)]
            tCtAcc_reuse = tCtAcc[(None, 1, 0)]
            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            tinfo_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 4
            )
            while work_tile.is_valid_tile:
                m_cta, n_t = self._remap_tile(work_tile.tile_idx, n_px)
                tinfo_pipeline.producer_acquire(tinfo_producer_state)
                with cute.arch.elect_one():
                    tinfo[0, tinfo_producer_state.index] = m_cta
                    tinfo[1, tinfo_producer_state.index] = n_t
                    tinfo[2, tinfo_producer_state.index] = cutlass.Int32(1)
                    tinfo_pipeline.producer_commit(tinfo_producer_state)
                tinfo_producer_state.advance()
                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                        tCrA_stage = tCrA[(None, None, None, handle.index)]
                        tCrB_stage = tCrB[(None, None, None, handle.index)]
                        for k_phase in cutlass.range_constexpr(
                            self.mma_tiler[2] // self.mma_inst_shape[2]
                        ):
                            tCrB_slice = tCrB_stage[(None, 0, k_phase)]
                            tiled_mma_bkeep.set(
                                tcgen05.Field.ACCUMULATE, k_tile != 0 or k_phase != 0
                            )
                            cute.gemm(
                                tiled_mma_bkeep,
                                tCtAcc_keep,
                                tCrA_stage[(None, 0, k_phase)],
                                tCrB_slice,
                                tCtAcc_keep,
                            )
                            tiled_mma_breuse.set(
                                tcgen05.Field.ACCUMULATE, k_tile != 0 or k_phase != 0
                            )
                            cute.gemm(
                                tiled_mma_breuse,
                                tCtAcc_reuse,
                                tCrA_stage[(None, 1, k_phase)],
                                tCrB_slice,
                                tCtAcc_reuse,
                            )
                        handle.release()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            tinfo_pipeline.producer_acquire(tinfo_producer_state)
            with cute.arch.elect_one():
                tinfo[2, tinfo_producer_state.index] = cutlass.Int32(0)
                tinfo_pipeline.producer_commit(tinfo_producer_state)
            tinfo_producer_state.advance()
            acc_pipeline.producer_tail(acc_producer_state)

        # ---------------- Epilogue warps: TMEM -> BF16 regs -> release -> SG -> E4M3 -> global ----------------
        if warp_idx < self.mma_warp_id:
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            # warps 0-3 drain the keep accumulator (M atom 0), warps 4-7 the reuse accumulator
            # (M atom 1); each thread owns one TMEM lane = one output row
            chunk = warp_idx // 4
            etid = tidx % 128

            # ((128,2),(256,1)) -> (128, 32, 2, 8): [lane, col, chunk, gate 0..3 | up 4..7]
            tAcc = transform_partitioned_tensor_layout(tCtAcc)
            tAcc_epi = cute.flat_divide(tAcc, self.epi_tile)
            copy_atom_t2r = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition.x32), self.acc_dtype
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(etid)
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

            tTR_rAcc = cute.make_rmem_tensor(
                thr_copy_t2r.partition_D(cute.make_identity_tensor(self.epi_tile)).shape,
                self.acc_dtype,
            )
            n_val = cute.size(tTR_rAcc)  # 32 fp32 columns per subtile
            num_sub = self.mma_tiler[1] // self.epi_tile[1]  # all eight natural-N subtiles
            # BF16 staging: [gate subtiles 0..3 | up subtiles 0..3] x 16 packed bf16x2
            rStage = cute.make_rmem_tensor((num_sub * n_val // 2,), cutlass.Uint32)
            tTR_rO = cute.make_rmem_tensor((n_val,), cutlass.BFloat16)
            tTR_rO32 = cute.recast_tensor(tTR_rO, cutlass.Uint32)
            store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=256
            )

            aux_x, aux_mod, aux_idx = aux
            rX = cute.make_rmem_tensor((n_val,), aux_x.element_type)
            rG = cute.make_rmem_tensor((n_val,), aux_mod.element_type)
            load_x = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), aux_x.element_type, num_bits_per_copy=256
            )
            load_mod = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), aux_mod.element_type, num_bits_per_copy=256
            )
            s_ab = s_in[0] * s_w[0]
            m_total = mOut.shape[0]

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            tinfo_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 4
            )
            tinfo_pipeline.consumer_wait(tinfo_consumer_state)
            m_cta = tinfo[0, tinfo_consumer_state.index]
            n_t = tinfo[1, tinfo_consumer_state.index]
            valid = tinfo[2, tinfo_consumer_state.index] != 0
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                tinfo_pipeline.consumer_release(tinfo_consumer_state)
            tinfo_consumer_state.advance()
            while valid:
                # 512-row MMA tile (m_cta // 2): M atom `chunk`, pair CTA v = m_cta % 2, lane etid
                row = (m_cta // 2) * 512 + chunk * 256 + (m_cta % 2) * 128 + etid
                out_col = n_t * self.mma_tiler[1]

                # ---- drain: TMEM fp32 -> bf16_rn(acc * sa*sw) staged in registers ----
                acc_pipeline.consumer_wait(acc_consumer_state)
                cute.arch.sync_warp()
                for sub in cutlass.range_constexpr(num_sub):
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, chunk, sub)], tTR_rAcc)
                    for i in cutlass.range_constexpr(n_val // 2):
                        rStage[sub * (n_val // 2) + i] = _scale_to_bf16x2(
                            tTR_rAcc[2 * i], tTR_rAcc[2 * i + 1], s_ab
                        )
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()
                zero = _ld_shared_volatile(zero_ptr)

                # Volatile-zero dependence pins auxiliary loads after the full TMEM drain.
                aux_x, aux_mod, aux_idx = aux
                selected = cutlass.Int64(0)
                aux_row = row + cutlass.Int32(zero)
                if row < m_total:
                    selected = cutlass.Int64(aux_idx[aux_row])
                gRow = cute.local_tile(mOut[(row, None)], (n_val,), (None,))
                for sub in cutlass.range_constexpr(num_sub):
                    for i in cutlass.range_constexpr(n_val // 2):
                        tTR_rO32[i] = rStage[sub * (n_val // 2) + i] | zero
                    if row < m_total:
                        xRow = cute.local_tile(aux_x[(aux_row, None)], (n_val,), (None,))
                        gGate = cute.local_tile(aux_mod[(selected, None)], (n_val,), (None,))
                        cute.copy(load_x, xRow[(None, out_col // n_val + sub)], rX)
                        cute.copy(
                            load_mod,
                            gGate[(None, (self.gate_column * self.n_dim + out_col) // n_val + sub)],
                            rG,
                        )
                        for i in cutlass.range_constexpr(n_val // 2):
                            g0 = cutlass.Float32(rG[2 * i])
                            g1 = cutlass.Float32(rG[2 * i + 1])
                            x0 = cutlass.Float32(rX[2 * i])
                            x1 = cutlass.Float32(rX[2 * i + 1])
                            value = cute.arch.fma_packed_f32x2(
                                (g0, g1),
                                (
                                    cutlass.Float32(tTR_rO[2 * i]),
                                    cutlass.Float32(tTR_rO[2 * i + 1]),
                                ),
                                (x0, x1),
                            )
                            tTR_rO[2 * i] = cutlass.BFloat16(value[0])
                            tTR_rO[2 * i + 1] = cutlass.BFloat16(value[1])
                        cute.copy(store_atom, tTR_rO, gRow[(None, out_col // n_val + sub)])
                    cute.arch.sync_warp()

                tinfo_pipeline.consumer_wait(tinfo_consumer_state)
                m_cta = tinfo[0, tinfo_consumer_state.index]
                n_t = tinfo[1, tinfo_consumer_state.index]
                valid = tinfo[2, tinfo_consumer_state.index] != 0
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    tinfo_pipeline.consumer_release(tinfo_consumer_state)
                tinfo_consumer_state.advance()

            tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


# ----------------------------------------------------------------------------------------------
# Host side
# ----------------------------------------------------------------------------------------------
CONFIG = dict(
    tile_n=256,
    cluster_shape_mn=(4, 1),
    num_ab_stage=4,
    swizzle_size=7,
    fallback_cluster_shape_mn=(2, 1),
    mma_tile_k=128,
)
ATTENTION_OUTPUT_GATE, DOWN_GATE = 2, 5
_DTYPES = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
}
_compiled = {}


def block_shared_memory_bytes(device_index):
    """Shared memory one block of this device may opt in to, 0 when the driver does not report it."""
    status, capacity = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute(150), cuda.CUdevice(device_index)
    )
    return int(capacity) if int(status) == 0 else 0


def compile_kernel(device_index, n, k, gate_column, modulation_dtype, index_dtype):
    """The compiled kernel for one projection shape; row counts are symbolic, so one build serves
    every sequence length and every number of modulation rows."""
    key = (device_index, n, k, gate_column, modulation_dtype, index_dtype)
    if key not in _compiled:
        tile_n = CONFIG["tile_n"]
        cn = CONFIG["cluster_shape_mn"][1]
        tiles = n // tile_n
        if n % tile_n or k % 128 or tiles % cn or (tiles // cn) % CONFIG["swizzle_size"]:
            raise ValueError(f"the FP8 projection kernel does not tile a [{n}, {k}] weight")
        capacity = block_shared_memory_bytes(device_index)
        needed = (256 + tile_n // 2) * CONFIG["mma_tile_k"] * CONFIG["num_ab_stage"] + 4096
        if capacity < needed:
            raise RuntimeError(
                f"the FP8 projection kernel needs {needed} bytes of shared memory per block, "
                f"the device offers {capacity}"
            )
        kern = ProjectionBreuseKernel(n, k, gate_column=gate_column, **CONFIG)
        m_sym = cute.sym_int32()
        act = make_fake_compact_tensor(
            cutlass.Float8E4M3FN, (m_sym, k), stride_order=(1, 0), assumed_align=256
        )
        wgt = make_fake_compact_tensor(
            cutlass.Float8E4M3FN, (n, k), stride_order=(1, 0), assumed_align=256
        )
        scale = make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
        out = make_fake_compact_tensor(
            cutlass.BFloat16, (m_sym, n), stride_order=(1, 0), assumed_align=256
        )
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        x = make_fake_compact_tensor(
            cutlass.BFloat16, (m_sym, n), stride_order=(1, 0), assumed_align=256
        )
        gates = make_fake_compact_tensor(
            _DTYPES[modulation_dtype],
            (cute.sym_int32(), 6 * n),
            stride_order=(1, 0),
            assumed_align=32,
        )
        indices = make_fake_compact_tensor(_DTYPES[index_dtype], (m_sym,), assumed_align=16)
        _compiled[key] = cute.compile(
            kern,
            act,
            wgt,
            scale,
            scale,
            (x, gates, indices),
            out,
            stream,
            options="--enable-tvm-ffi",
        )
    return _compiled[key]


def supports(a, w, x, mod, idx, gate_column):
    """Whether a kernel compiled by ``compile_kernel`` serves these tensors as they are.

    Nothing is compiled here: a shape, gate or dtype without a compiled kernel, or a tensor
    outside the kernel's layout and alignment contract, takes the caller's unfused path."""
    if not (a.is_cuda and a.ndim == w.ndim == mod.ndim == 2):
        return False
    m, n = a.shape[0], w.shape[0]
    key = (a.device.index, n, a.shape[1], gate_column, mod.dtype, idx.dtype)
    return (
        key in _compiled
        and m > 0
        and a.shape[1] == w.shape[1]
        and a.dtype == w.dtype == torch.float8_e4m3fn
        and x.dtype == torch.bfloat16
        and x.numel() == m * n
        and mod.shape[1] == 6 * n
        and idx.numel() == m
        and all(t.device == a.device for t in (w, x, mod, idx))
        and all(t.is_contiguous() for t in (a, w, x, mod, idx))
        and all(t.data_ptr() % 256 == 0 for t in (a, w, x))
        and mod.data_ptr() % 32 == 0
        and mod.stride(0) * mod.element_size() % 32 == 0
        and idx.data_ptr() % 16 == 0
    )


def run(a, w, sa, sw, x, mod, idx, gate_column):
    """``bf16(x + gate * bf16(sa * sw * a @ w.T))`` as a new ``[M, N]`` BF16 tensor."""
    m, k = a.shape
    n = w.shape[0]
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    _compiled[(a.device.index, n, k, gate_column, mod.dtype, idx.dtype)](
        a, w, sa.reshape(1), sw.reshape(1), (x.view(m, n), mod, idx), out
    )
    return out
