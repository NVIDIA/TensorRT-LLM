# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A 2-CTA (``tcgen05`` ``cta_group::2``) BF16 GEMM with an in-kernel cluster split-K reduction.

For the TP8 QKV projection at 65..128 rows (9..16 requests at K = 7): ``out[R, N] = x[R, K] @ w[N, K]^T``
with N = 2304, K = 8192. The one-launch split-K tiles of ``splitk_gemm`` lose to cuBLAS there (W11:
8.5..10.1 us vs 7.2..8.6): every 64-column weight tile re-streams the whole activation tile, so each
CTA ingests 512..768 KiB. cuBLAS (nvjet 192x64 ``1x2_2cta`` split-K) halves the per-CTA operand
traffic with 2-CTA MMA but pays a separate ``splitKreduce`` launch (~2.4 us). This kernel takes both:

* Kernel M = the weight's N (output columns), kernel N = the activation rows (tokens), as
  ``splitk_gemm``. A CTA pair computes a 128-column x 128-token tile with one 2-CTA MMA per K block:
  each CTA stages its own 64 weight columns (A) and half of the 128 tokens (B); the MMA reads both
  CTAs' shared memory, so a CTA ingests 64 x K/split_k weights + 64 x K/split_k activations: the
  64-row ``splitk_gemm`` launch's bytes per CTA, for twice the rows.
* The cluster is ``(2, split_k, 1)``: x = the pair, y = the K slice. CTA (x, y) accumulates K slice y
  of its 64 x 128 tile in FP32 (TMEM); the K-slice CTAs of the same x reduce over DSMEM into the
  y = 0 CTA (``st.async`` + transaction bytes on one mbarrier), which sums its own partial, then
  y = 1, 2, ... in that order, rounds once to BF16 and stores. That is ``splitk_gemm``'s owner order:
  with the same split_k, a row's bits are the 64-row launch's whenever the 2-CTA MMA accumulates each
  element over its K slice in the same order as the 1-CTA MMA (W14 checks it per row).
* Grid ``(2 * ceil(N / 128), ceil(R / 128) * split_k, 1)``: 36 x 4 = 144 CTAs at split_k 4.
* PDL (``pdl``): only the TMA warp executes ``griddepcontrol.wait``; before it, with
  ``prefetch_a``, it prefetches its whole weight slice into L2 (TMA prefetch), so the weight
  stream overlaps the predecessor. It releases dependents after issuing its last load. Every
  output store follows the MMAs, which follow the activation loads, which follow the wait.

Not served yet (no lever): the 1-GPU W14 gate in ``test_modeling_v2_wide128_bits.py`` times it
against cuBLAS first. Host entry: ``run_splitk_2cta(x, w, out, pdl, split_k)``. Any compile or
launch failure raises.
"""

import functools
from typing import Tuple

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch as _torch
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.custom_pipeline import (
    PipelineTmaUmma,
    PipelineUmmaAsync,
)

#: The pair tile (kernel M x kernel N): 128 weight columns (64 per CTA) x 128 tokens.
PAIR_TILE_MN = (128, 128)
#: The most activation rows one launch takes (one 128-token tile).
MAX_ROWS = 128
#: K extents of one A/B stage.
K_TILES = (64, 128)
#: Split-K sizes this kernel serves (cluster (2, split_k) <= 8 CTAs: portable).
SPLIT_KS = (2, 4)
_OPTIN_SMEM_BYTES = utils.get_smem_capacity_in_bytes("sm_100")


@dsl_user_op
def _map_shared_rank(smem_ptr, peer_rank: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr.toint(loc=loc, ip=ip).ir_value(), peer_rank.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _store_remote_v4(v0, v1, v2, v3, smem_ptr, mbar_ptr, peer_rank: Int32, *, loc=None, ip=None):
    """Four FP32 values into a peer CTA's shared memory, crediting 16 transaction bytes."""
    llvm.inline_asm(
        None,
        [
            _map_shared_rank(smem_ptr, peer_rank, loc=loc, ip=ip).ir_value(),
            v0.bitcast(Int32).ir_value(loc=loc, ip=ip),
            v1.bitcast(Int32).ir_value(loc=loc, ip=ip),
            v2.bitcast(Int32).ir_value(loc=loc, ip=ip),
            v3.bitcast(Int32).ir_value(loc=loc, ip=ip),
            _map_shared_rank(mbar_ptr, peer_rank, loc=loc, ip=ip).ir_value(),
        ],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];",
        "r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class SplitK2CtaGemmKernel:
    """``C[M, N] = A[M, K] @ B[N, K]^T`` (A and B K-major, C M-major) on 2-CTA MMA with a cluster
    split-K reduction; see the module docstring."""

    def __init__(
        self,
        split_k: int,
        use_pdl: bool,
        ab_stages: int = 0,
        prefetch_a: bool = True,
        rs: bool = False,
        k_tile: int = 64,
        prestage: bool = False,
    ):
        if split_k not in SPLIT_KS:
            raise ValueError(f"split_k must be one of {SPLIT_KS}, got {split_k}")
        self.split_k = int(split_k)
        self.use_pdl = bool(use_pdl)
        self.prefetch_a = bool(prefetch_a) and self.use_pdl
        #: Reduce-scatter epilogue: epilogue thread group g (128 / split_k threads) of every K-slice
        #: CTA belongs to K slice g, which sums that group's values over the slices in slice order
        #: (slice 0's partial, then 1's, ...: the owner order, the same bits) and stores them. The
        #: mailbox is (split_k - 1) / split_k of a tile instead of split_k - 1 tiles, so more A/B
        #: stages fit (12 instead of 8 at split_k 4), and the K-slice CTAs share the reduction.
        self.rs = bool(rs)
        #: K extent of one A/B stage (64 or 128): 128 halves the stage count's barrier round trips
        #: for the same bytes in flight; the MMAs still run in ascending k (the same bits).
        if int(k_tile) not in K_TILES:
            raise ValueError(f"k_tile must be one of {K_TILES}, got {k_tile}")
        self.k_tile = int(k_tile)
        #: With PDL: the first ``num_ab_stage`` weight k-tiles are TMA-loaded into their (free)
        #: stages before ``griddepcontrol.wait``, their activation halves after it on the same full
        #: barriers, as the 1-CTA split-K kernel stages its weights before the wait.
        self.prestage = bool(prestage) and self.use_pdl
        self.requested_stages = int(ab_stages)
        self.acc_dtype = cutlass.Float32
        self.mma_tiler_mn = PAIR_TILE_MN
        self.cluster_shape_mn = (2, self.split_k)
        self.cta_group = tcgen05.CtaGroup.TWO
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * 6
        self.tmem_alloc_sync_bar_id = 2
        self.epilog_threads = 128

    def _setup(self, a_dtype, a_major, b_major, c_layout, c_dtype):
        self.a_dtype = a_dtype
        self.c_dtype = c_dtype
        self.c_layout = c_layout
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype, a_dtype, a_major, b_major, self.acc_dtype, self.cta_group, self.mma_tiler_mn
        )
        mma_inst_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        if self.k_tile % mma_inst_k:
            raise ValueError(
                f"k_tile {self.k_tile} is not whole MMA instructions of K {mma_inst_k}"
            )
        self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.k_tile)
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        # TMA and the pipelines see the whole cluster: V = the pair, M = 1, N = the K slices. A and B
        # are never multicast (the K-slice CTAs load different K ranges).
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.pair_layout_vmnk = cute.tiled_divide(
            cute.make_layout((2, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk, True, c_layout, c_dtype
        )
        a_one = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, a_dtype, 1)
        b_one = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, a_dtype, 1)
        stage_bytes = cute.size_in_bytes(a_dtype, a_one) + cute.size_in_bytes(a_dtype, b_one)
        tile_elems = self.cta_tile_shape_mnk[0] * self.cta_tile_shape_mnk[1]
        self.rs_group = self.epilog_threads // self.split_k
        if self.rs:
            self.mailbox_elems = (self.split_k - 1) * tile_elems // self.split_k
        else:
            self.mailbox_elems = (self.split_k - 1) * tile_elems
        fit = (_OPTIN_SMEM_BYTES - self.mailbox_elems * 4 - 2048) // stage_bytes
        stages = min(fit, 12)
        if self.requested_stages:
            stages = min(stages, self.requested_stages)
        if stages < 2:
            raise ValueError(f"2-CTA split-K: only {fit} A/B stages fit beside the mailbox")
        self.num_ab_stage = stages
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, a_dtype, stages
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, a_dtype, stages
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(
            tiled_mma.make_fragment_C(cute.append(acc_shape, 1)), arch="sm_100"
        )
        return tiled_mma

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: _cuda.CUstream):
        tiled_mma = self._setup(
            a.element_type,
            utils.LayoutEnum.from_tensor(a).mma_major_mode(),
            utils.LayoutEnum.from_tensor(b).mma_major_mode(),
            utils.LayoutEnum.from_tensor(c),
            c.element_type,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((2, 1), tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem_layout, self.mma_tiler, tiled_mma, self.pair_layout_vmnk.shape
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B((2, 1), tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_layout, self.mma_tiler, tiled_mma, self.pair_layout_vmnk.shape
        )
        a_copy = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy = cute.size_in_bytes(self.a_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy + b_copy) * atom_thr_size
        grid = (
            cute.ceil_div(c.layout.shape[0], self.cta_tile_shape_mnk[0]),
            cute.ceil_div(c.layout.shape[1], self.cta_tile_shape_mnk[1]) * self.split_k,
            1,
        )
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_tile,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        bidx, bidy, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        k_rank = bidy % self.split_k
        tile_m = bidx // 2
        tile_n = bidy // self.split_k
        tidx, _, _ = cute.arch.thread_idx()

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_dealloc_mbar: cutlass.Int64
            reduce_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline = PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            # B's "multicast" mode spans only the pair (M = 1): the MMA commit reaches both CTAs
            # of this pair and never another K slice.
            mcast_mode_mn=(0, 1),
        )
        ab_producer, ab_consumer = ab_pipeline.make_participants()
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilogue_warp_id) * 2
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        reduce_mbar = storage.reduce_mbar.ptr
        if warp_idx == self.epilogue_warp_id[0]:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(reduce_mbar, 1)
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        mailbox = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(self.mailbox_elems),
            byte_alignment=128,
        )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        cC_mnl = cute.local_tile(
            cute.make_identity_tensor(mC_mnl.shape),
            cute.slice_(self.mma_tiler, (None, None, 0)),
            (None, None, None),
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        k_tiles = k_tile_cnt // self.split_k
        k_begin = k_rank * k_tiles

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)
        tCcC = thr_mma.partition_C(cC_mnl)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )  # fmt: skip
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1), cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )  # fmt: skip
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # TMA warp: this CTA's 64 weight columns and its half of the tokens, K slice k_rank.
        if warp_idx == self.tma_warp_id:
            tAgA_slice = tAgA[(None, tile_m, None, 0)]
            tBgB_slice = tBgB[(None, tile_n, None, 0)]
            ab_producer.reset()
            pre = Int32(0)
            if cutlass.const_expr(self.prestage):
                # The first stages' weight k-tiles into shared memory before the wait.
                pre = cutlass.min(k_tiles, Int32(self.num_ab_stage))
                for kk in cutlass.range(0, pre, 1, unroll=1):
                    handle = ab_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, k_begin + handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                    )
            if cutlass.const_expr(self.prefetch_a):
                # The weight slice into L2 while the predecessor runs (weights never depend on it).
                for k_tile in cutlass.range(pre, k_tiles, 1, unroll=1):
                    cute.prefetch(tma_atom_a, tAgA_slice[(None, k_begin + k_tile)])
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.prestage):
                # Their activation halves, on the same stages' full barriers (stage kk = k-tile kk).
                for kk in cutlass.range(0, pre, 1, unroll=1):
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, k_begin + kk)],
                        tBsB[(None, kk)],
                        tma_bar_ptr=ab_pipeline.sync_object_full.get_barrier(kk),
                    )
            peek = ab_producer.try_acquire()
            for k_tile in cutlass.range(pre, k_tiles, 1, unroll=1):
                handle = ab_producer.acquire_and_advance(peek)
                cute.copy(
                    tma_atom_a,
                    tAgA_slice[(None, k_begin + handle.count)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_slice[(None, k_begin + handle.count)],
                    tBsB[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
                peek = cutlass.Boolean(1)
                if handle.count + 1 < k_tiles:
                    peek = ab_producer.try_acquire()
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()
            ab_producer.tail()

        # MMA warp: the pair leader issues the 2-CTA MMAs of the K slice.
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)[(None, None, None, 0)]
            acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            ab_consumer.reset()
            peek_full = cutlass.Boolean(1)
            if is_leader_cta:
                peek_full = ab_consumer.try_wait()
                acc_pipeline.producer_acquire(acc_state)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_tile in cutlass.range(0, k_tiles, 1, unroll=1):
                if is_leader_cta:
                    handle = ab_consumer.wait_and_advance(peek_full)
                    num_kblocks = cute.size(tCrA, mode=[2])
                    for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                        crd = (None, None, kblock_idx, handle.index)
                        cute.gemm(tiled_mma, tCtAcc, tCrA[crd], tCrB[crd], tCtAcc)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    handle.release()
                    peek_full = cutlass.Boolean(1)
                    if handle.count + 1 < k_tiles:
                        peek_full = ab_consumer.try_wait()
            if is_leader_cta:
                acc_pipeline.producer_commit(acc_state)
            acc_state.advance()
            acc_pipeline.producer_tail(acc_state)

        # Epilogue warps: TMEM -> registers, the K-slice reduction into k_rank 0, BF16 store.
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk, self.c_layout, self.c_dtype, self.acc_dtype, epi_tile, True
            )
            tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)[(None, None, None, None, None, 0)]
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
            gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
            tTR_gC = thr_copy_t2r.partition_D(gC_epi)[
                (None, None, None, None, None, tile_m, tile_n, 0)
            ]
            tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
            cC_epi = cute.flat_divide(tCcC[((None, None), 0, 0, None, None, None)], epi_tile)
            tTR_cC = thr_copy_t2r.partition_D(cC_epi)[
                (None, None, None, None, None, tile_m, tile_n, 0)
            ]
            tTR_cC = cute.group_modes(tTR_cC, 3, cute.rank(tTR_cC))
            tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0)].shape, self.acc_dtype)
            subtiles = cute.size(tTR_tAcc.shape, mode=[3])
            vals = cute.size(tTR_rAcc)
            groups = vals // 4

            acc_pipeline.consumer_wait(acc_state)
            if cutlass.const_expr(self.rs):
                # Reduce-scatter: thread group dest = tidx // rs_group of every K-slice CTA sends its
                # partials to slice dest's CTA (same pair position), whose group dest sums slice 0's
                # value, then slice 1's, ... (the owner order: the same bits), rounds once, stores.

                dest = tidx // self.rs_group
                lt = tidx % self.rs_group
                if dest == k_rank:
                    if lt == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(reduce_mbar, self.mailbox_elems * 4)
                    cute.arch.mbarrier_wait(reduce_mbar, 0)
                for s in cutlass.range(subtiles):
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rAcc)
                    if dest != k_rank:
                        # This sender's slot in slice dest's mailbox: slices below dest keep their index,
                        # slices above shift down by one.
                        j = k_rank - cutlass.select_(k_rank > dest, Int32(1), Int32(0))
                        base = (j * subtiles + s) * groups
                        for g in cutlass.range_constexpr(groups):
                            _store_remote_v4(
                                tTR_rAcc[4 * g],
                                tTR_rAcc[4 * g + 1],
                                tTR_rAcc[4 * g + 2],
                                tTR_rAcc[4 * g + 3],
                                mailbox.iterator + ((base + g) * self.rs_group + lt) * 4,
                                reduce_mbar,
                                cta_rank + 2 * (dest - k_rank),
                            )
                    else:
                        for g in cutlass.range_constexpr(groups):
                            for i in cutlass.range_constexpr(4):
                                own = tTR_rAcc[4 * g + i]
                                acc = own
                                for r in cutlass.range_constexpr(self.split_k):
                                    # slice r's partial: this CTA's own, or its mailbox slot (clamped
                                    # when unused).
                                    jr = cutlass.min(
                                        Int32(r)
                                        - cutlass.select_(Int32(r) > k_rank, Int32(1), Int32(0)),
                                        Int32(self.split_k - 2),
                                    )
                                    off = (
                                        ((jr * subtiles + s) * groups + g) * self.rs_group + lt
                                    ) * 4 + i
                                    term = cutlass.select_(Int32(r) == k_rank, own, mailbox[off])
                                    if cutlass.const_expr(r == 0):
                                        acc = term
                                    else:
                                        acc = acc + term
                                tTR_rAcc[4 * g + i] = acc
                        tTR_gC_s = tTR_gC[(None, None, None, s)]
                        tTR_cC_s = tTR_cC[(None, None, None, s)]
                        for v in cutlass.range_constexpr(vals):
                            if cute.elem_less(tTR_cC_s[v], mC_mnl.shape):
                                tTR_gC_s[v] = tTR_rAcc[v].to(self.c_dtype)
            else:
                # Owner: slice 0's CTA sums the slices' partials in slice order and stores the tile.

                if k_rank == 0:
                    if tidx == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(reduce_mbar, self.mailbox_elems * 4)
                    cute.arch.mbarrier_wait(reduce_mbar, 0)
                for s in cutlass.range(subtiles):
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rAcc)
                    if k_rank != 0:
                        # Slot (k_rank - 1, s, group, thread) of the k_rank-0 CTA of this pair position.
                        base = ((k_rank - 1) * subtiles + s) * groups
                        for g in cutlass.range_constexpr(groups):
                            _store_remote_v4(
                                tTR_rAcc[4 * g],
                                tTR_rAcc[4 * g + 1],
                                tTR_rAcc[4 * g + 2],
                                tTR_rAcc[4 * g + 3],
                                mailbox.iterator + ((base + g) * self.epilog_threads + tidx) * 4,
                                reduce_mbar,
                                cta_rank - 2 * k_rank,
                            )
                    else:
                        for j in cutlass.range_constexpr(self.split_k - 1):
                            base = (j * subtiles + s) * groups
                            for g in cutlass.range_constexpr(groups):
                                off = ((base + g) * self.epilog_threads + tidx) * 4
                                for i in cutlass.range_constexpr(4):
                                    tTR_rAcc[4 * g + i] = tTR_rAcc[4 * g + i] + mailbox[off + i]
                        tTR_gC_s = tTR_gC[(None, None, None, s)]
                        tTR_cC_s = tTR_cC[(None, None, None, s)]
                        for v in cutlass.range_constexpr(vals):
                            if cute.elem_less(tTR_cC_s[v], mC_mnl.shape):
                                tTR_gC_s[v] = tTR_rAcc[v].to(self.c_dtype)
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_state)
            acc_state.advance()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

    @cute.jit
    def wrapper(
        self,
        m: Int32,
        n: Int32,
        k: Int32,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        stream: _cuda.CUstream,
    ):
        """A = w [m, k] (K-major), B = x [n, k] (K-major), C = out viewed as [m, n] M-major."""
        a = cute.make_tensor(a_ptr, layout=cute.make_ordered_layout((m, k, 1), order=(1, 0, 2)))
        b = cute.make_tensor(b_ptr, layout=cute.make_ordered_layout((n, k, 1), order=(1, 0, 2)))
        c = cute.make_tensor(c_ptr, layout=cute.make_ordered_layout((m, n, 1), order=(0, 1, 2)))
        self(a, b, c, stream)


@functools.cache
def _compiled(
    split_k: int,
    use_pdl: bool,
    ab_stages: int,
    prefetch_a: bool = True,
    rs: bool = False,
    k_tile: int = 64,
    prestage: bool = False,
):
    kernel = SplitK2CtaGemmKernel(split_k, use_pdl, ab_stages, prefetch_a, rs, k_tile, prestage)
    dummy = _torch.empty(16, dtype=_torch.bfloat16, device="cuda")
    ptr = lambda: make_ptr(  # noqa: E731
        cutlass.BFloat16, dummy.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
    return cute.compile(kernel.wrapper, 2304, 128, 8192, ptr(), ptr(), ptr(), stream)


def check(
    x: _torch.Tensor, w: _torch.Tensor, out: _torch.Tensor, split_k: int, k_tile: int = 64
) -> Tuple[int, int, int]:
    """``(rows, n, k)``; raises ``ValueError`` for operands this kernel does not take."""
    if x.dtype != _torch.bfloat16 or w.dtype != _torch.bfloat16 or out.dtype != _torch.bfloat16:
        raise ValueError("2-CTA split-K takes BF16 operands")
    if x.dim() != 2 or w.dim() != 2 or out.dim() != 2:
        raise ValueError("2-CTA split-K takes 2-D x, w and out")
    if not (x.is_contiguous() and w.is_contiguous() and out.is_contiguous()):
        raise ValueError("2-CTA split-K takes contiguous x, w and out")
    rows, k = x.shape
    n = w.shape[0]
    if w.shape[1] != k or tuple(out.shape) != (rows, n):
        raise ValueError(
            f"2-CTA split-K shapes: x {tuple(x.shape)}, w {tuple(w.shape)}, out {tuple(out.shape)}"
        )
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"2-CTA split-K takes 1..{MAX_ROWS} rows, got {rows}")
    if n % PAIR_TILE_MN[0] or k % (k_tile * split_k):
        raise ValueError(f"2-CTA split-K: N {n} % 128 or K {k} % {k_tile * split_k} is not 0")
    if any(t.data_ptr() % 16 for t in (x, w, out)):
        raise ValueError("2-CTA split-K takes 16-byte aligned x, w and out")
    return rows, n, k


def run_splitk_2cta(
    x: _torch.Tensor,
    w: _torch.Tensor,
    out: _torch.Tensor,
    pdl: bool,
    split_k: int = 4,
    ab_stages: int = 0,
    prefetch_a: bool = True,
    rs: bool = False,
    k_tile: int = 64,
    prestage: bool = False,
) -> _torch.Tensor:
    """``out = x @ w.T`` in one launch (see the module docstring); ``rs`` the reduce-scatter
    epilogue (``SplitK2CtaGemmKernel``)."""
    rows, n, k = check(x, w, out, split_k, k_tile)
    compiled = _compiled(
        int(split_k),
        bool(pdl),
        int(ab_stages),
        bool(prefetch_a),
        bool(rs),
        int(k_tile),
        bool(prestage),
    )
    mk = lambda t: make_ptr(  # noqa: E731
        cutlass.BFloat16, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    stream = _cuda.CUstream(_torch.cuda.current_stream(x.device).cuda_stream)
    compiled(n, rows, k, mk(w), mk(x), mk(out), stream)
    return out
