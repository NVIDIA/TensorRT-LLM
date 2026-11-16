# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL kernel for the DSA indexer's ragged prefill scores on the SM100 family (tcgen05).

    logits[r, j] = scale[j] * sum_h w[r, h] * relu(q[r, h, :] . k[j, :])    ks[r] <= j < ke[r]
    logits[r, j] = -inf                                                       other j < n_k

One UMMA tile is 128 keys (M) by 4 query rows x 32 heads (N = 128), K = 128, FP8 operands and
an FP32 accumulator in tensor memory. The keys sit on the TMEM lanes, so a thread owns one key
and reduces the 32 heads of a row in registers: no cross-thread reduction. The epilogue loads
`rows_per_load` rows of the accumulator at a time and the row's head weights from shared memory.

Work: the query rows come in blocks of 4 and the keys in blocks of 128, grouped into splits of
`kv_per_split` key blocks. A work item is one (query block, split); the persistent CTAs take
items round robin. An item scores only the key blocks that meet the union of its rows' windows
and writes -inf over the rest of its split, so every logit of `[rows, n_k]` is written once and
the tiles outside every window cost a store, not an MMA.

Warp roles (384 threads), as the paged decode kernel (`fp8_paged_mqa_logits.py`):
  warps 0-7   two math warpgroups; group g reduces the key blocks c_lo + g, c_lo + g + 2, ...
  warp 8      TMA: the item's queries and head weights, then group 0's keys and key scales
  warp 9      TMA: group 1's keys and key scales
  warps 10-11 UMMA for groups 0 and 1
Each group has its own key pipeline and two accumulator stages (4 x 128 TMEM columns).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

HEADS = 32
HEAD_DIM = 128
#: Query rows per UMMA tile: 4 rows x 32 heads = N 128.
BLOCK_Q = 4
#: Keys per UMMA tile (M).
BLOCK_KV = 128

# CuTe DSL validates rounding modes as string literals.
_RND_RN = "rn"


class Fp8MqaLogitsPrefillKernel:
    """Ragged FP8 MQA logits over gathered keys, one persistent launch."""

    def __init__(self, num_kv_stages: int = 4, num_q_stages: int = 2, rows_per_load: int = 2):
        self.num_kv_stages = num_kv_stages
        self.num_q_stages = num_q_stages
        # Query rows per TMEM load, 32 accumulator columns each.
        if BLOCK_Q % rows_per_load:
            raise ValueError(f"rows_per_load must divide {BLOCK_Q}, got {rows_per_load}")
        self.rows_per_load = rows_per_load
        self.num_umma_stages = 2
        self.N = BLOCK_Q * HEADS
        self.acc_dtype = cutlass.Float32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_mn = (BLOCK_KV, self.N)
        self.threads_per_cta = 384

    def _setup_mma(self, ab_dtype, a_major, b_major):
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, a_major, b_major, self.acc_dtype, self.cta_group, self.mma_tiler_mn
        )
        self.mma_tiler = (BLOCK_KV, self.N, HEAD_DIM)
        self.cta_tile_shape_mnk = self.mma_tiler
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, ab_dtype, self.num_kv_stages
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, ab_dtype, self.num_q_stages
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        self.num_tmem_cols = utils.get_num_tmem_alloc_cols(tiled_mma.make_fragment_C(acc_shape))
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        k_u8: cute.Tensor,  # [n_k, 128] uint8 (e4m3 codes), rows contiguous
        q_u8: cute.Tensor,  # [rows * 32, 128] uint8 (e4m3 codes)
        weights: cute.Tensor,  # [rows * 32] fp32
        scales: cute.Tensor,  # [n_k] fp32
        ks: cute.Tensor,  # [rows] int32
        ke: cute.Tensor,  # [rows] int32
        logits: cute.Tensor,  # [rows, >= n_k] fp32, unit column stride
        num_rows: cutlass.Int32,
        num_keys: cutlass.Int32,
        kv_per_split: cutlass.Int32,
        num_splits: cutlass.Int32,
        num_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        k_fp8 = cute.recast_tensor(k_u8, cutlass.Float8E4M3FN)
        q_fp8 = cute.recast_tensor(q_u8, cutlass.Float8E4M3FN)
        a = cute.make_tensor(
            k_fp8.iterator,
            cute.make_layout((num_keys, HEAD_DIM, 1), stride=(HEAD_DIM, 1, num_keys * HEAD_DIM)),
        )
        b = cute.make_tensor(
            q_fp8.iterator,
            cute.make_layout(
                (num_rows * HEADS, HEAD_DIM, 1), stride=(HEAD_DIM, 1, num_rows * HEADS * HEAD_DIM)
            ),
        )
        ab_dtype = a.element_type
        self.ab_dtype = ab_dtype
        a_major = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        b_major = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        tiled_mma = self._setup_mma(ab_dtype, a_major, b_major)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        # The query block's 128 head weights and the key block's 128 scales: 1-D tiles whose
        # out-of-range tail (the last query or key block) the TMA fills with zeros.
        self.w_smem_layout_staged = cute.make_layout(
            (self.N, self.num_q_stages), stride=(1, self.N)
        )
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op, weights, cute.make_layout((self.N,)), (self.N,)
        )
        self.s_smem_layout_staged = cute.make_layout(
            (BLOCK_KV, self.num_kv_stages), stride=(1, BLOCK_KV)
        )
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_load_op, scales, cute.make_layout((BLOCK_KV,)), (BLOCK_KV,)
        )
        self.num_q_tma_bytes = cute.size_in_bytes(ab_dtype, b_smem_layout) + self.N * 4
        self.num_kv_tma_bytes = cute.size_in_bytes(ab_dtype, a_smem_layout) + BLOCK_KV * 4

        @cute.struct
        class SharedStorage:
            kv_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            kv_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.num_q_stages * 2]
            umma_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            umma_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            tmem_holding_buf: cutlass.Int32

        num_kb = (num_keys + BLOCK_KV - 1) // BLOCK_KV
        num_items = (num_rows + BLOCK_Q - 1) // BLOCK_Q * num_splits
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_s,
            tma_tensor_s,
            ks,
            ke,
            logits,
            num_rows,
            num_keys,
            num_kb,
            kv_per_split,
            num_splits,
            num_items,
            num_ctas,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.w_smem_layout_staged,
            self.s_smem_layout_staged,
            SharedStorage,
        ).launch(
            grid=(num_ctas, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.jit
    def _item(self, item, mStarts, mEnds, num_rows, num_keys, num_kb, kv_per_split, num_splits):
        """Work item `item`: its query block, its split's key blocks `[sp_lo, sp_hi)`, and the
        key blocks `[c_lo, c_hi)` of the split that meet its live rows' windows (empty when
        `c_hi <= c_lo`). A row past `num_rows` or with an empty window is not live."""
        qb = item // num_splits
        split = item - qb * num_splits
        sp_lo = split * kv_per_split
        sp_hi = cutlass.min(sp_lo + kv_per_split, num_kb)
        lo = num_keys
        hi = cutlass.Int32(0)
        for t in cutlass.range_constexpr(BLOCK_Q):
            r = qb * BLOCK_Q + t
            if r < num_rows:
                start = cutlass.max(mStarts[r], cutlass.Int32(0))
                end = cutlass.min(mEnds[r], num_keys)
                if end > start:
                    lo = cutlass.min(lo, start)
                    hi = cutlass.max(hi, end)
        c_lo = cutlass.max(sp_lo, lo // BLOCK_KV)
        c_hi = cutlass.min(sp_hi, (hi + BLOCK_KV - 1) // BLOCK_KV)
        return qb, sp_lo, sp_hi, c_lo, c_hi

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_w: cute.CopyAtom,
        mW: cute.Tensor,
        tma_atom_s: cute.CopyAtom,
        mS: cute.Tensor,
        mStarts: cute.Tensor,
        mEnds: cute.Tensor,
        mLogits: cute.Tensor,
        num_rows: cutlass.Int32,
        num_keys: cutlass.Int32,
        num_kb: cutlass.Int32,
        kv_per_split: cutlass.Int32,
        num_splits: cutlass.Int32,
        num_items: cutlass.Int32,
        num_ctas: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.Layout,
        s_smem_layout_staged: cute.Layout,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warpgroup_idx = warp_idx // 4
        is_math_warp = warp_idx < 8

        if warp_idx == 8:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_s)
        if warp_idx == 9:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_s)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # --- pipelines ---
        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # Queries + weights: both math warpgroups consume (and release) every staged block.
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar.data_ptr(),
            num_stages=self.num_q_stages,
            producer_group=prod_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
            tx_count=self.num_q_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        # Keys + scales, per group: the UMMA warp waits, the group's math warps release once
        # the accumulator (hence the MMA's reads) is done.
        kv_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
        kv_pipeline_0 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar_0.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        kv_pipeline_1 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar_1.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        umma_pipeline_0 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar_0.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        umma_pipeline_1 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar_1.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # TMEM: math warp 0 allocates and frees (the last reader); the UMMA warps retrieve.
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=320)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=False,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sKV_0 = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sKV_1 = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sQ = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sW = smem.allocate_tensor(
            element_type=cutlass.Float32, layout=w_smem_layout_staged, byte_alignment=128
        )
        sS_0 = smem.allocate_tensor(
            element_type=cutlass.Float32, layout=s_smem_layout_staged, byte_alignment=128
        )
        sS_1 = smem.allocate_tensor(
            element_type=cutlass.Float32, layout=s_smem_layout_staged, byte_alignment=128
        )

        # --- TMA partitions ---
        thr_mma = tiled_mma.get_slice(0)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        one_cta = cute.make_layout(1)
        tAsA_0, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, one_cta, cute.group_modes(sKV_0, 0, 3), cute.group_modes(tCgA, 0, 3)
        )
        tAsA_1, _ = cpasync.tma_partition(
            tma_atom_a, 0, one_cta, cute.group_modes(sKV_1, 0, 3), cute.group_modes(tCgA, 0, 3)
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, one_cta, cute.group_modes(sQ, 0, 3), cute.group_modes(tCgB, 0, 3)
        )
        gW = cute.local_tile(mW, (self.N,), (None,))
        tWsW, tWgW = cpasync.tma_partition(tma_atom_w, 0, one_cta, cute.group_modes(sW, 0, 1), gW)
        gS = cute.local_tile(mS, (BLOCK_KV,), (None,))
        tSsS_0, tSgS = cpasync.tma_partition(
            tma_atom_s, 0, one_cta, cute.group_modes(sS_0, 0, 1), gS
        )
        tSsS_1, _ = cpasync.tma_partition(tma_atom_s, 0, one_cta, cute.group_modes(sS_1, 0, 1), gS)

        # --- MMA fragments and the staged accumulators ---
        tCrA_0 = tiled_mma.make_fragment_A(sKV_0)
        tCrA_1 = tiled_mma.make_fragment_A(sKV_1)
        tCrB = tiled_mma.make_fragment_B(sQ)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_umma_stages))
        cols_per_group = self.num_tmem_cols * self.num_umma_stages
        num_k_blocks = cute.size(tCrA_0, mode=[2])

        # Epilogue: `rows_per_load` rows (32 heads each) of the accumulator per TMEM load.
        epi_sub = (BLOCK_KV, HEADS * self.rows_per_load)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            utils.LayoutEnum.ROW_MAJOR,
            self.acc_dtype,
            self.acc_dtype,
            epi_sub,
            False,
        )

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx == 8:
            cute.arch.warpgroup_reg_dealloc(24)
            q_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_q_stages
            )
            kv_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_kv_stages
            )
            item = cta
            while item < num_items:
                qb, sp_lo, sp_hi, c_lo, c_hi = self._item(
                    item, mStarts, mEnds, num_rows, num_keys, num_kb, kv_per_split, num_splits
                )
                if c_hi > c_lo:
                    q_pipeline.producer_acquire(q_prod)
                    bar = q_pipeline.producer_get_barrier(q_prod)
                    cute.copy(
                        tma_atom_b, tBgB[(None, qb, 0, 0)], tBsB[(None, q_prod.index)],
                        tma_bar_ptr=bar,
                    )  # fmt: skip
                    cute.copy(
                        tma_atom_w, tWgW[(None, qb)], tWsW[(None, q_prod.index)], tma_bar_ptr=bar
                    )
                    q_prod.advance()
                    kb = c_lo
                    while kb < c_hi:
                        kv_pipeline_0.producer_acquire(kv_prod)
                        bar = kv_pipeline_0.producer_get_barrier(kv_prod)
                        cute.copy(
                            tma_atom_a, tAgA[(None, kb, 0, 0)], tAsA_0[(None, kv_prod.index)],
                            tma_bar_ptr=bar,
                        )  # fmt: skip
                        cute.copy(
                            tma_atom_s, tSgS[(None, kb)], tSsS_0[(None, kv_prod.index)],
                            tma_bar_ptr=bar,
                        )  # fmt: skip
                        kv_prod.advance()
                        kb = kb + 2
                item = item + num_ctas

        elif warp_idx == 9:
            cute.arch.warpgroup_reg_dealloc(24)
            kv_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_kv_stages
            )
            item = cta
            while item < num_items:
                qb, sp_lo, sp_hi, c_lo, c_hi = self._item(
                    item, mStarts, mEnds, num_rows, num_keys, num_kb, kv_per_split, num_splits
                )
                kb = c_lo + 1
                while kb < c_hi:
                    kv_pipeline_1.producer_acquire(kv_prod)
                    bar = kv_pipeline_1.producer_get_barrier(kv_prod)
                    cute.copy(
                        tma_atom_a, tAgA[(None, kb, 0, 0)], tAsA_1[(None, kv_prod.index)],
                        tma_bar_ptr=bar,
                    )  # fmt: skip
                    cute.copy(
                        tma_atom_s, tSgS[(None, kb)], tSsS_1[(None, kv_prod.index)],
                        tma_bar_ptr=bar,
                    )  # fmt: skip
                    kv_prod.advance()
                    kb = kb + 2
                item = item + num_ctas

        elif warp_idx == 10 or warp_idx == 11:
            cute.arch.warpgroup_reg_dealloc(24)
            group = warp_idx - 10
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            q_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_q_stages
            )
            kv_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_kv_stages
            )
            umma_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_umma_stages
            )
            if group == 0:
                self._umma_loop(
                    tiled_mma, cute.make_tensor(tmem_ptr, tCtAcc_fake.layout), tCrA_0, tCrB,
                    num_k_blocks, q_pipeline, kv_pipeline_0, umma_pipeline_0, q_cons, kv_cons,
                    umma_prod, 0, cta, num_ctas, num_items, mStarts, mEnds, num_rows, num_keys, num_kb,
                    kv_per_split, num_splits,
                )  # fmt: skip
            else:
                self._umma_loop(
                    tiled_mma,
                    cute.make_tensor(tmem_ptr + cols_per_group, tCtAcc_fake.layout), tCrA_1,
                    tCrB, num_k_blocks, q_pipeline, kv_pipeline_1, umma_pipeline_1, q_cons,
                    kv_cons, umma_prod, 1, cta, num_ctas, num_items, mStarts, mEnds, num_rows,
                    num_keys, num_kb, kv_per_split, num_splits,
                )  # fmt: skip

        elif is_math_warp:
            cute.arch.warpgroup_reg_alloc(240)
            tmem.allocate(cols_per_group * 2)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            q_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_q_stages
            )
            kv_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_kv_stages
            )
            umma_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_umma_stages
            )
            if warpgroup_idx == 0:
                self._math_loop(
                    cute.make_tensor(tmem_ptr, tCtAcc_fake.layout), copy_atom_t2r, epi_sub, sW,
                    sS_0, q_pipeline, kv_pipeline_0, umma_pipeline_0, q_cons, kv_cons, umma_cons,
                    0, tidx % 128, cta, num_ctas, num_items, mStarts, mEnds, mLogits, num_rows,
                    num_keys, num_kb, kv_per_split, num_splits,
                )  # fmt: skip
            else:
                self._math_loop(
                    cute.make_tensor(tmem_ptr + cols_per_group, tCtAcc_fake.layout),
                    copy_atom_t2r, epi_sub, sW, sS_1, q_pipeline, kv_pipeline_1, umma_pipeline_1,
                    q_cons, kv_cons, umma_cons, 1, tidx % 128, cta, num_ctas, num_items, mStarts, mEnds,
                    mLogits, num_rows, num_keys, num_kb, kv_per_split, num_splits,
                )  # fmt: skip
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
        else:
            cute.arch.warpgroup_reg_dealloc(24)

    @cute.jit
    def _umma_loop(
        self,
        tiled_mma,
        tCtAcc,
        tCrA,
        tCrB,
        num_k_blocks,
        q_pipeline,
        kv_pipeline,
        umma_pipeline,
        q_cons,
        kv_cons,
        umma_prod,
        group: cutlass.Constexpr[int],
        cta,
        num_ctas,
        num_items,
        mStarts,
        mEnds,
        num_rows,
        num_keys,
        num_kb,
        kv_per_split,
        num_splits,
    ):
        """One group's MMAs. The queries' stage is waited on only by a group with key blocks
        in the item: the math warps release it, so a group with none could otherwise find the
        stage already refilled for a later item."""
        item = cta
        while item < num_items:
            qb, sp_lo, sp_hi, c_lo, c_hi = self._item(
                item, mStarts, mEnds, num_rows, num_keys, num_kb, kv_per_split, num_splits
            )
            if c_hi > c_lo:
                if c_hi > c_lo + group:
                    q_pipeline.consumer_wait(q_cons)
                q_stage = q_cons.index
                kb = c_lo + group
                while kb < c_hi:
                    kv_pipeline.consumer_wait(kv_cons)
                    umma_pipeline.producer_acquire(umma_prod)
                    acc = tCtAcc[(None, None, None, umma_prod.index)]
                    kv_stage = kv_cons.index
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k_block in cutlass.range_constexpr(num_k_blocks):
                        cute.gemm(
                            tiled_mma,
                            acc,
                            tCrA[None, None, k_block, kv_stage],
                            tCrB[None, None, k_block, q_stage],
                            acc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    kv_cons.advance()
                    umma_pipeline.producer_commit(umma_prod)
                    umma_prod.advance()
                    kb = kb + 2
                q_cons.advance()
            item = item + num_ctas

    @cute.jit
    def _math_loop(
        self,
        tCtAcc,
        copy_atom_t2r,
        epi_sub,
        sW,
        sS,
        q_pipeline,
        kv_pipeline,
        umma_pipeline,
        q_cons,
        kv_cons,
        umma_cons,
        group: cutlass.Constexpr[int],
        local_tidx,
        cta,
        num_ctas,
        num_items,
        mStarts,
        mEnds,
        mLogits,
        num_rows,
        num_keys,
        num_kb,
        kv_per_split,
        num_splits,
    ):
        """One group's epilogue: per key block, relu, the head weights' FMA and the key scale
        for each of the 4 rows, masked to the row's window; -inf over the group's share of the
        split's key blocks outside the scored range."""
        acc_ref = tCtAcc[(None, None, None, 0)][((None, None), 0, 0)]
        acc_ref_epi = cute.flat_divide(acc_ref, epi_sub)
        t2r_ref = tcgen05.make_tmem_copy(copy_atom_t2r, acc_ref_epi[(None, None, 0, 0)])
        tTR_cC = t2r_ref.get_slice(local_tidx).partition_D(cute.make_identity_tensor(epi_sub))
        m_coord = tTR_cC[0][0]
        tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)
        # One row's head weights, copied from shared memory per row: the item's 128 weights held
        # in registers beside the accumulator rows leave the epilogue short of registers.
        w_reg = cute.make_rmem_tensor(HEADS, cutlass.Float32)
        neg_inf = cutlass.Float32(-float("inf"))
        zero = cutlass.Float32(0.0)
        rows_per_load = self.rows_per_load

        item = cta
        while item < num_items:
            qb, sp_lo, sp_hi, c_lo, c_hi = self._item(
                item, mStarts, mEnds, num_rows, num_keys, num_kb, kv_per_split, num_splits
            )
            row0 = qb * BLOCK_Q
            # The rows' windows (a row past num_rows reads the last row's and stores nothing).
            starts = [cutlass.Int32(0)] * BLOCK_Q
            ends = [cutlass.Int32(0)] * BLOCK_Q
            for t in cutlass.range_constexpr(BLOCK_Q):
                r = cutlass.min(row0 + t, num_rows - 1)
                starts[t] = mStarts[r]
                ends[t] = mEnds[r]
            if c_hi > c_lo:
                q_pipeline.consumer_wait(q_cons)
                q_stage = q_cons.index
                kb = c_lo + group
                while kb < c_hi:
                    kv_pipeline.consumer_wait(kv_cons)
                    scale = sS[(m_coord, kv_cons.index)]
                    umma_pipeline.consumer_wait(umma_cons)
                    kv_pipeline.consumer_release(kv_cons)
                    kv_cons.advance()
                    acc = tCtAcc[(None, None, None, umma_cons.index)][((None, None), 0, 0)]
                    acc_epi = cute.flat_divide(acc, epi_sub)
                    t2r = tcgen05.make_tmem_copy(copy_atom_t2r, acc_epi[(None, None, 0, 0)])
                    tTR = t2r.get_slice(local_tidx).partition_S(acc_epi)
                    key = kb * BLOCK_KV + m_coord
                    for i in cutlass.range_constexpr(BLOCK_Q // rows_per_load):
                        cute.copy(t2r, tTR[(None, None, None, 0, i)], tTR_rAcc)
                        cute.arch.fence_view_async_tmem_load()
                        if i == BLOCK_Q // rows_per_load - 1:
                            umma_pipeline.consumer_release(umma_cons)
                            umma_cons.advance()
                        v = tTR_rAcc.load()
                        for u in cutlass.range_constexpr(rows_per_load):
                            t = i * rows_per_load + u
                            cute.autovec_copy(
                                cute.local_tile(sW[(None, q_stage)], (HEADS,), (t,)), w_reg
                            )
                            s0x = zero
                            s0y = zero
                            s1x = zero
                            s1y = zero
                            for h in cutlass.range_constexpr(0, HEADS, 4):
                                c = u * HEADS + h
                                a0 = cutlass.max(v[c], zero)
                                a1 = cutlass.max(v[c + 1], zero)
                                a2 = cutlass.max(v[c + 2], zero)
                                a3 = cutlass.max(v[c + 3], zero)
                                w = h
                                s0x, s0y = cute.arch.fma_packed_f32x2(
                                    (a0, a1), (w_reg[w], w_reg[w + 1]), (s0x, s0y), rnd=_RND_RN
                                )
                                s1x, s1y = cute.arch.fma_packed_f32x2(
                                    (a2, a3), (w_reg[w + 2], w_reg[w + 3]), (s1x, s1y), rnd=_RND_RN
                                )
                            score = ((s0x + s0y) + (s1x + s1y)) * scale
                            if (key < starts[t]) | (key >= ends[t]):
                                score = neg_inf
                            if (row0 + t < num_rows) & (key < num_keys):
                                mLogits[(row0 + t, key)] = score
                    kb = kb + 2
                q_pipeline.consumer_release(q_cons)
                q_cons.advance()
            # -inf over this group's share of the split's unscored key blocks.
            kb = sp_lo + group
            while kb < sp_hi:
                if (kb < c_lo) | (kb >= c_hi):
                    key = kb * BLOCK_KV + m_coord
                    if key < num_keys:
                        for t in cutlass.range_constexpr(BLOCK_Q):
                            if row0 + t < num_rows:
                                mLogits[(row0 + t, key)] = neg_inf
                kb = kb + 2
            item = item + num_ctas
