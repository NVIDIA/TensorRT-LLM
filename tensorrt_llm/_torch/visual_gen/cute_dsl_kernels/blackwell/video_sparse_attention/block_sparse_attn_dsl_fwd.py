# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Video Sparse Attention Forward (Blackwell)

This file implements the forward pass of the video sparse attention.
It will produce the output and the log-sum-exp of the attention scores.

This implementation requires zero-padding on the input tensor, to align with the block-size (64x64).
"""

import math
from functools import partial
from typing import Callable, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05

from . import ptx, scheduler


def make_thread_cooperative_group(size: int):
    """
    Create a thread cooperative group.
    """
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


def next_power_of_2(x: int) -> int:
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()


SM100_TMEM_CAPACITY_COLUMNS: int = 512


class VideoSparseAttentionForwardGroup2QInterleaveKV:
    """
    This class implements the forward pass of the video sparse attention.
    It doesn't require swap QK.
              V0
        K0 K1 V1
    Q0  S0 0  O0
    Q1  0  S1 O1
    """

    MAX_INDICES = 4 * 1024

    def __init__(
        self,
        block_m: int,
        block_n: int,
        headdim: int,
    ):
        self.block_m = block_m
        self.block_n = block_n
        assert block_m == 64 and block_n == 64, "Block size must be 64x64"
        self.headdim = headdim
        assert self.headdim == 128, "Head dimension must be 128"

        self.acc_dtype: cutlass.Numeric = cutlass.Float32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_qk = (self.block_m * 2, self.block_n * 2, 1)
        self.mma_tiler_pv = (self.block_m * 2, self.headdim, 1)

        self.occupancy = 1
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

        self.threads_per_warp: int = 32
        self.threads_per_wg: int = 128

        self.load_warp_id = 0
        self.mma_warp_id = 1
        self.epilogue_warp_id = 2
        self.empty_warp_ids = (3,)
        self.softmax_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)

        self.threads_per_cta: int = self.threads_per_warp * len(
            (
                *self.softmax_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.epilogue_warp_id,
                *self.empty_warp_ids,
            )
        )

        self.buffer_align_bytes: int = 1024

        self.scheduler_cls = scheduler.StaticPersistentScheduler

        self.rescale_threshold: float = 8.0

        self.mask_O: bool = False
        # either P is in SMEM or in TMEM
        self.p_in_smem: bool = False

        self.num_regs_load: int = 104
        self.num_regs_mma: int = 104
        self.num_regs_epi: int = 104
        self.num_regs_softmax: int = 224  # bigger as much as possible
        self.num_regs_correction: int = 176
        self.num_regs_empty: int = 24

    def _compute_grid(
        self,
        num_q_blocks: int,
        num_kv_blocks: int,
        num_heads: int,
        batchsize: int,
        headdim: int,
        headdim_v: int,
    ) -> Tuple[Tuple[int, int, int], scheduler.ParamsBase]:
        cluster_shape = (*self.cluster_shape_mn, 1)

        scheduler_params = scheduler.TileSchedulerParams(
            num_block=cute.ceil_div(num_q_blocks, 2),
            num_head=num_heads,
            num_batch=batchsize,
            headdim=headdim,
            headdim_v=headdim_v,
        )
        params = self.scheduler_cls.to_underlying_arguments(scheduler_params)

        grid = self.scheduler_cls.get_grid_shape(params)
        grid = cute.round_up(grid, cluster_shape)
        return grid, params

    def _compute_stages(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        input_dtype: cutlass.Numeric,
    ):
        self.q_stage = 1
        assert self.q_stage == 1, "Q will be preserving in SMEM"
        self.kv_stage = 2 if self.p_in_smem else 3
        self.s_stage = 2
        assert self.s_stage == 2, "S stage must be 2 for Interleaving KV blocks"
        self.o_stage = 1
        self.epi_stage = 1 if self.p_in_smem else 2

        self.scale_buffers = 2

        self.tmem_cols_S = self.mma_tiler_qk[1] * self.s_stage
        self.tmem_cols_O = self.mma_tiler_pv[1] * self.o_stage

        # P reuses S TMEM, when there is no space for P
        self.p_in_s: bool = (not self.p_in_smem) and (self.o_stage == 2)

        self.tmem_cols_P = (self.mma_tiler_pv[1] // 2) * self.s_stage * (not self.p_in_s)

        self.tmem_offset_S = 0
        self.tmem_offset_O = self.tmem_cols_S
        self.tmem_offset_P = self.tmem_offset_O + self.tmem_cols_O

        self.tmem_alloc_cols = self.tmem_cols_S + self.tmem_cols_O + self.tmem_cols_P
        self.tmem_alloc_cols = next_power_of_2(self.tmem_alloc_cols)
        assert self.tmem_alloc_cols <= SM100_TMEM_CAPACITY_COLUMNS
        self.do_tmem_alloc: bool = self.tmem_alloc_cols != SM100_TMEM_CAPACITY_COLUMNS

    def _setup_attributes(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        input_dtype: cutlass.Numeric,
    ):
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )

        qk_mma_inst_shape_k = cute.size(tiled_mma_qk.shape_mnk, mode=[2])
        qk_mma_inst_tile_k: int = self.headdim // qk_mma_inst_shape_k
        self.mma_tiler_qk = (
            self.mma_tiler_qk[0],
            self.mma_tiler_qk[1],
            qk_mma_inst_shape_k * qk_mma_inst_tile_k,
        )

        pv_mma_inst_shape_k = cute.size(tiled_mma_pv.shape_mnk, mode=[2])
        pv_mma_inst_tile_k: int = self.mma_tiler_qk[1] // pv_mma_inst_shape_k
        self.mma_tiler_pv = (
            self.mma_tiler_pv[0],
            self.mma_tiler_pv[1],
            pv_mma_inst_shape_k * pv_mma_inst_tile_k,
        )

        self._compute_stages(tiled_mma_qk, tiled_mma_pv, input_dtype)

    @cute.jit
    def __call__(
        self,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        sm_scale: cutlass.Float32,
        O: cute.Tensor,  # noqa: E741
        LSE: cute.Tensor,
        q2k_block_sparse_index: cute.Tensor,
        q2k_block_sparse_num: cute.Tensor,
        variable_block_sizes: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        input_dtype = Q.element_type
        if cutlass.const_expr(input_dtype not in [cutlass.Float16, cutlass.BFloat16]):
            raise RuntimeError("Input dtype must be Float16 or BFloat16")
        if cutlass.const_expr(
            not (input_dtype == K.element_type == V.element_type == O.element_type)
        ):
            raise RuntimeError("All input tensors must have the same data type")

        batch, heads, seqlen, dim = Q.layout.shape
        if cutlass.const_expr(dim != self.headdim):
            raise RuntimeError(f"Dimension mismatch: {dim} != {self.headdim}")

        _, _, _, dim_v = V.layout.shape
        if cutlass.const_expr(dim_v != self.headdim):
            raise RuntimeError(f"Dimension mismatch: {dim_v} != {self.headdim}")

        _, _, num_q_blocks, num_kv_blocks = q2k_block_sparse_index.layout.shape

        grid, params = self._compute_grid(
            num_q_blocks=num_q_blocks,
            num_kv_blocks=num_kv_blocks,
            num_heads=heads,
            batchsize=batch,
            headdim=dim,
            headdim_v=dim_v,
        )

        # [batch, heads, seqlen, dim] -> [seqlen, dim, heads, batch]
        Q_layout_transpose = [2, 3, 1, 0]
        KV_layout_transpose = [2, 3, 1, 0]
        # [batch, heads, seqlen, dim] -> [seqlen, dim, heads, batch]
        O_layout_transpose = [2, 3, 1, 0]
        # [seqlen, dim, heads, batch] -> [dim, seqlen, heads, batch]
        V_layout_transpose = [1, 0, 2, 3]
        # [batch, heads, seqlen] -> [seqlen, heads, batch]
        LSE_layout_transpose = [2, 1, 0]
        Q = cute.make_tensor(Q.iterator, cute.select(Q.layout, mode=Q_layout_transpose))
        K = cute.make_tensor(K.iterator, cute.select(K.layout, mode=KV_layout_transpose))
        V = cute.make_tensor(V.iterator, cute.select(V.layout, mode=KV_layout_transpose))
        # NOTE: need transpose here, make V be N-major
        V = cute.make_tensor(V.iterator, cute.select(V.layout, mode=V_layout_transpose))
        O = cute.make_tensor(O.iterator, cute.select(O.layout, mode=O_layout_transpose))  # noqa: E741
        LSE = cute.make_tensor(LSE.iterator, cute.select(LSE.layout, mode=LSE_layout_transpose))

        q_major_mode = utils.LayoutEnum.from_tensor(Q).mma_major_mode()
        k_major_mode = utils.LayoutEnum.from_tensor(K).mma_major_mode()
        v_major_mode = tcgen05.OperandMajorMode.MN

        tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(
            input_dtype,
            q_major_mode,
            k_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_qk[:2],
        )
        p_major_mode = tcgen05.OperandMajorMode.K
        p_source = tcgen05.OperandSource.SMEM if self.p_in_smem else tcgen05.OperandSource.TMEM
        tiled_mma_pv = sm100_utils.make_trivial_tiled_mma(
            input_dtype,
            p_major_mode,
            v_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_pv[:2],
            a_source=p_source,
        )
        self._setup_attributes(tiled_mma_qk, tiled_mma_pv, input_dtype)

        o_layout = utils.LayoutEnum.from_tensor(O)
        self.epi_tile = (self.mma_tiler_pv[0], self.mma_tiler_pv[1])

        sQ_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, input_dtype, self.q_stage
        )
        sK_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, input_dtype, self.kv_stage
        )

        sP_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, input_dtype, self.s_stage
        )

        sV_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, input_dtype, self.kv_stage
        )

        sO_layout = sm100_utils.make_smem_layout_epi(
            input_dtype, o_layout, self.epi_tile, self.epi_stage
        )
        fake_sO_layout = sm100_utils.make_smem_layout_epi(
            input_dtype, o_layout, (self.block_m, self.headdim), self.epi_stage
        )

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", Q, sQ_layout),
                ("K", K, sK_layout),
                ("V", V, sV_layout),
            ]
        }

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        # use fake tiled mma to get TMA atoms
        fake_tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(
            input_dtype,
            q_major_mode,
            k_major_mode,
            self.acc_dtype,
            self.cta_group,
            (self.block_m, self.block_n),
        )
        fake_sQ_layout = sm100_utils.make_smem_layout_a(
            fake_tiled_mma_qk,
            (self.block_m, self.block_n, self.mma_tiler_qk[2]),
            input_dtype,
            self.q_stage,
        )
        fake_sK_layout = sm100_utils.make_smem_layout_b(
            fake_tiled_mma_qk,
            (self.block_m, self.block_n, self.mma_tiler_qk[2]),
            input_dtype,
            self.kv_stage,
        )

        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            Q,
            cute.select(fake_sQ_layout, mode=[0, 1, 2]),
            (self.block_m, self.block_n, self.mma_tiler_qk[2]),
            fake_tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            K,
            cute.select(fake_sK_layout, mode=[0, 1, 2]),
            (self.block_m, self.block_n, self.mma_tiler_qk[2]),
            fake_tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )

        fake_tiled_mma_pv = sm100_utils.make_trivial_tiled_mma(
            input_dtype,
            p_major_mode,
            v_major_mode,
            self.acc_dtype,
            self.cta_group,
            (self.block_m, self.block_n),
        )
        fake_sV_layout = sm100_utils.make_smem_layout_b(
            fake_tiled_mma_pv,
            (self.block_m, self.block_n, self.block_n),
            input_dtype,
            self.kv_stage,
        )

        tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            V,
            cute.select(fake_sV_layout, mode=[0, 1, 2]),
            (self.block_m, self.block_n, self.block_n),
            fake_tiled_mma_pv,
            self.cluster_layout_vmnk.shape,
        )

        o_cta_v_layout = cute.composition(
            cute.make_identity_layout(O.shape), (self.block_m, self.headdim)
        )
        tma_atom_O, mO = cpasync.make_tiled_tma_atom(
            tma_store_op, O, cute.select(fake_sO_layout, mode=[0, 1]), o_cta_v_layout
        )

        sScale_layout = None
        if cutlass.const_expr(self.o_stage == 1):
            # [acc_scale]
            sScale_layout = cute.make_ordered_layout((self.mma_tiler_qk[0], self.s_stage), (0, 1))
        else:
            # [acc_scale, running_max]
            sScale_layout = cute.make_ordered_layout(
                (2, self.mma_tiler_qk[0], self.s_stage), (0, 1, 2)
            )

        # [running_sum, running_max] * scale_buffers
        # TODO: rearrange this to make to LDS.64 instead of LDS.32 x 2
        sFinal_layout = cute.make_ordered_layout(
            (2, self.mma_tiler_qk[0], self.scale_buffers), (0, 1, 2)
        )

        self.max_indices = self.MAX_INDICES

        @cute.struct
        class SharedStorage:
            load_KV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]
            qk_mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.s_stage * 2]
            p_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.s_stage * 2]
            o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.o_stage * 2]
            pv_mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.o_stage * 2]
            store_O_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.epi_stage * 2]
            corr_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.s_stage * 2]

            tmem_holding_buf: cutlass.Int32

            sScale: cute.struct.MemRange[cutlass.Float32, cute.cosize(sScale_layout)]

            sFinal: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(sFinal_layout)],
                8,  # for LDS.64 it shall be 8 bytes aligned.
            ]
            sFinal_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.scale_buffers * 2]

            O_final_guard: cutlass.Int64

            sQ: cute.struct.Align[
                cute.struct.MemRange[input_dtype, cute.cosize(sQ_layout)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[input_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]
            sP: cute.struct.Align[
                cute.struct.MemRange[input_dtype, cute.cosize(sP_layout) * (self.p_in_smem)],
                self.buffer_align_bytes,
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[input_dtype, cute.cosize(sO_layout)],
                self.buffer_align_bytes,
            ]
            sVariable_block_sizes: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.max_indices],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.LOG2_E: float = math.log2(math.e)
        self.LN2: float = math.log(2.0)
        sm_scale_log2 = sm_scale * self.LOG2_E

        self.kernel(
            mQ,
            mK,
            mV,
            sm_scale_log2,
            mO,
            LSE,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            tiled_mma_qk,
            tiled_mma_pv,
            fake_tiled_mma_qk,
            fake_tiled_mma_pv,
            sQ_layout,
            sK_layout,
            sV_layout,
            sP_layout,
            sScale_layout,
            sFinal_layout,
            sO_layout,
            self.cluster_layout_vmnk,
            params,
            q2k_block_sparse_index,
            q2k_block_sparse_num,
            variable_block_sizes,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # [seq, dim, head, batch]
        mK: cute.Tensor,  # [seq, dim, head, batch]
        mV: cute.Tensor,  # [dim, seq, head, batch]
        sm_scale_log2: cutlass.Float32,  # softmax scale in log2
        mO: cute.Tensor,  # [dim, seq, head, batch]
        mLSE: cute.Tensor,  # [seq, head, batch]
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        fake_tiled_mma_qk: cute.TiledMma,
        fake_tiled_mma_pv: cute.TiledMma,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout,
        sScale_layout: cute.Layout,
        sFinal_layout: cute.Layout,
        sO_layout: cute.ComposedLayout,
        cluster_layout_vmnk: cute.Layout,
        scheduler_params: scheduler.ParamsBase,
        q2k_block_sparse_index: cute.Tensor,
        q2k_block_sparse_num: cute.Tensor,
        variable_block_sizes: cute.Tensor,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0]

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_O)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        KV_mbar_ptr = storage.load_KV_mbar_ptr.data_ptr()
        O_final_guard = storage.O_final_guard
        if warp_idx == self.load_warp_id:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(
                    O_final_guard, self.threads_per_warp * len(self.correction_warp_ids)
                )
                for i in cutlass.range_constexpr(self.kv_stage, unroll_full=True):
                    # producer
                    cute.arch.mbarrier_init(KV_mbar_ptr + i * 2, len([self.load_warp_id]))
                    # consumer
                    cute.arch.mbarrier_init(KV_mbar_ptr + i * 2 + 1, len([self.mma_warp_id]))
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )

        qk_mma_pipeline = pipeline.PipelineUmmaAsync.create(
            num_stages=self.s_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax_warp_ids)
            ),
            barrier_storage=storage.qk_mma_mbar_ptr.data_ptr(),
        )
        qk_mma_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.s_stage
        )
        qk_mma_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.s_stage
        )

        p_pipeline = pipeline.PipelineAsyncUmma.create(
            num_stages=self.s_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.p_mbar_ptr.data_ptr(),
        )
        p_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.s_stage
        )
        p_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.s_stage
        )
        o_pipeline = pipeline.PipelineAsyncUmma.create(
            num_stages=self.o_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.o_mbar_ptr.data_ptr(),
        )
        o_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.o_stage
        )
        o_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.o_stage
        )

        pv_mma_pipeline = pipeline.PipelineUmmaAsync.create(
            num_stages=self.o_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.pv_mma_mbar_ptr.data_ptr(),
        )
        pv_mma_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.o_stage
        )
        pv_mma_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.o_stage
        )

        st_O_pipeline = pipeline.PipelineAsync.create(
            num_stages=self.epi_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len([self.epilogue_warp_id])
            ),
            barrier_storage=storage.store_O_mbar_ptr.data_ptr(),
        )
        st_O_producer_state = pipeline.PipelineState(
            self.epi_stage,
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
        )
        st_O_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.epi_stage
        )

        corr_pipeline = pipeline.PipelineAsync.create(
            num_stages=self.s_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.corr_mbar_ptr.data_ptr(),
        )

        correction_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.s_stage
        )

        sFinal_pipeline = pipeline.PipelineAsync.create(
            num_stages=self.scale_buffers,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.sFinal_mbar_ptr.data_ptr(),
        )
        sFinal_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.scale_buffers
        )
        sFinal_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.scale_buffers
        )

        cute.arch.mbarrier_init_fence()

        tmem_holding_buf = storage.tmem_holding_buf
        if cutlass.const_expr(self.do_tmem_alloc):
            if warp_idx == 0:
                cute.arch.alloc_tmem(
                    self.tmem_alloc_cols,
                    tmem_holding_buf,
                    is_two_cta=False,
                )
        cute.arch.sync_threads()

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # NOTE: V will reuse the same smem as K
        # stripe swizzle info to reuse smem
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)

        sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)

        sVariable_block_sizes_layout = cute.make_layout((self.max_indices,))
        sVariable_block_sizes = storage.sVariable_block_sizes.get_tensor(
            sVariable_block_sizes_layout
        )

        for i in cutlass.range(tidx, variable_block_sizes.shape[0], cute.arch.block_dim()[0]):
            sVariable_block_sizes[i] = variable_block_sizes[i]
        # cta sync
        cute.arch.sync_threads()

        sScale = storage.sScale.get_tensor(sScale_layout)
        sFinal = storage.sFinal.get_tensor(sFinal_layout)

        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_pv = tiled_mma_pv.get_slice(0)

        # ----- GMEM partition ----------- #
        # [block_m, dimk, block_cnt, loop_k, head, batch]
        gQ = cute.flat_divide(mQ, (self.block_m, self.headdim))
        gK = cute.flat_divide(mK, (self.block_n, self.headdim))
        gO = cute.flat_divide(mO, (self.block_m, self.headdim))
        # [dimK, block_n, loop_k, block_cnt, head, batch]
        gV = cute.flat_divide(mV, (self.headdim, self.block_n))

        # [block_m, block_cnt, head, batch]
        gLSE = cute.flat_divide(mLSE, (self.block_m,))

        # ----- TMEM partition ----------- #
        # NOTE: we already know each SM's occupancy is 1,
        # so tmem_ptr starting at zero
        tmem_ptr = cute.make_ptr(
            self.acc_dtype, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16
        )
        if cutlass.const_expr(self.do_tmem_alloc):
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )

        tS_shape = (self.mma_tiler_qk[0], self.tmem_cols_S)
        tCtS_shape = thr_mma_qk.partition_shape_C(tS_shape)
        tCtS_fake = thr_mma_qk.make_fragment_C(tCtS_shape)
        tCtS = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_offset_S, dtype=self.acc_dtype), tCtS_fake.layout
        )
        sP = None
        if cutlass.const_expr(self.p_in_smem):
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        else:
            if cutlass.const_expr(self.p_in_s):
                p_ptr = cute.recast_ptr(tmem_ptr + self.tmem_offset_S, dtype=sV.element_type)
                tP = cute.make_tensor(p_ptr, sP_layout.outer)
                tCtP = thr_mma_pv.make_fragment_A(tP)
                stride = tCtP.stride
                # since it will reuse the S TMEM, we need to make sure the stride is correct
                stride = (*(stride[i] for i in range(len(stride) - 1)), stride[len(stride) - 1] * 2)
                layout = cute.make_layout(tCtP.shape, stride=stride)
                tCtP = cute.make_tensor(tCtP.iterator, layout)
                sP = cute.make_tensor(p_ptr, tCtP.layout)
            else:
                # P has its own TMEM
                p_ptr = cute.recast_ptr(tmem_ptr + self.tmem_offset_P, dtype=sV.element_type)
                tP = cute.make_tensor(p_ptr, sP_layout.outer)
                tCtP = thr_mma_pv.make_fragment_A(tP)
                sP = cute.make_tensor(p_ptr, tCtP.layout)

        tO_shape = (self.mma_tiler_pv[0], self.tmem_cols_O)
        tCtO_shape = thr_mma_pv.partition_shape_C(tO_shape)
        tCtO_fake = thr_mma_pv.make_fragment_C(tCtO_shape)
        tCtO = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_offset_O, dtype=self.acc_dtype), tCtO_fake.layout
        )

        TileSchedulerCls = partial(self.scheduler_cls.create, scheduler_params)

        if warp_idx in self.empty_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_load)

            self.load(
                TileSchedulerCls,
                gQ,
                gK,
                gV,
                KV_mbar_ptr,
                kv_producer_state,
                q2k_block_sparse_num,
                q2k_block_sparse_index,
                sQ,
                sK,
                sV,
                fake_tiled_mma_qk,
                fake_tiled_mma_pv,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
            )

        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_mma)

            self.mma(
                TileSchedulerCls,
                KV_mbar_ptr,
                kv_consumer_state,
                q2k_block_sparse_num,
                qk_mma_pipeline,
                qk_mma_producer_state,
                tiled_mma_qk,
                p_pipeline,
                p_consumer_state,
                o_pipeline,
                o_consumer_state,
                pv_mma_pipeline,
                pv_mma_producer_state,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                sP,
                tCtS,
                tCtO,
            )

        if warp_idx == self.epilogue_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_epi)

            self.epilogue(
                TileSchedulerCls,
                st_O_pipeline,
                st_O_consumer_state,
                gO,
                sO,
                tma_atom_O,
            )

        if warp_idx in self.softmax_warp_ids:
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)

            num_q_blocks = cute.size(gQ, mode=[2])
            self.softmax(
                TileSchedulerCls,
                q2k_block_sparse_num,
                q2k_block_sparse_index,
                variable_block_sizes,
                sVariable_block_sizes,
                qk_mma_pipeline,
                qk_mma_consumer_state,
                p_pipeline,
                p_producer_state,
                num_q_blocks,
                sP,
                tCtS,
                thr_mma_qk,
                sm_scale_log2,
                sScale,
                corr_pipeline,
                sFinal,
                sFinal_pipeline,
                sFinal_producer_state,
                O_final_guard,
            )

        if warp_idx in self.correction_warp_ids:
            cute.arch.warpgroup_reg_alloc(self.num_regs_correction)

            num_q_blocks = cute.size(gQ, mode=[2])

            self.correction(
                TileSchedulerCls,
                q2k_block_sparse_num,
                p_producer_state,
                o_pipeline,
                o_producer_state,
                pv_mma_pipeline,
                pv_mma_consumer_state,
                st_O_pipeline,
                st_O_producer_state,
                tCtO,
                thr_mma_pv,
                sO,
                corr_pipeline,
                correction_consumer_state,
                sScale,
                sm_scale_log2,
                num_q_blocks,
                variable_block_sizes,
                sFinal,
                sFinal_pipeline,
                sFinal_consumer_state,
                O_final_guard,
                gLSE,
            )

        if cutlass.const_expr(self.do_tmem_alloc):
            cute.arch.sync_threads()
            if warp_idx == 0:
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.dealloc_tmem(
                    tmem_ptr,
                    self.tmem_alloc_cols,
                )

    @cute.jit
    def load(
        self,
        TileSchedulerCls: Callable,
        gQ: cute.Tensor,
        gK: cute.Tensor,
        gV: cute.Tensor,
        KV_mbar_ptr: cutlass.Pointer,
        kv_producer_state: pipeline.PipelineState,
        q2k_block_sparse_num: cute.Tensor,
        q2k_block_sparse_index: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        fake_tiled_mma_qk: cute.TiledMma,
        fake_tiled_mma_pv: cute.TiledMma,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
    ):
        fake_thr_mma_qk = fake_tiled_mma_qk.get_slice(0)
        fake_thr_mma_pv = fake_tiled_mma_pv.get_slice(0)

        def _get_sQ_cpy(
            sQ: cute.Tensor,
        ) -> cute.Tensor:
            _tmp = cute.flatten(sQ)
            mode0 = cute.get(_tmp.layout, mode=[0])
            mode0_split = cute.flat_divide(mode0, (self.block_m,))
            shape = (
                mode0_split.shape,
                cute.get(_tmp.layout, mode=[1]).shape,
                cute.get(_tmp.layout, mode=[2]).shape,
                cute.get(_tmp.layout, mode=[3]).shape,
                cute.get(_tmp.layout, mode=[4]).shape,
                cute.get(_tmp.layout, mode=[5]).shape,
            )
            stride = (
                mode0_split.stride,
                cute.get(_tmp.layout, mode=[1]).stride,
                cute.get(_tmp.layout, mode=[2]).stride,
                cute.get(_tmp.layout, mode=[3]).stride,
                cute.get(_tmp.layout, mode=[4]).stride,
                cute.get(_tmp.layout, mode=[5]).stride,
            )
            layout = cute.make_layout(shape, stride=stride)
            layout = cute.flatten(layout)
            layout = cute.select(layout, mode=[0, 2, 3, 4, 5, 1, 6])
            layout = cute.group_modes(layout, 0, 2)
            cpy_tensor = cute.make_tensor(sQ.iterator, layout)
            return cpy_tensor

        sQ_cpy = _get_sQ_cpy(sQ)

        def _get_gQ_cpy(
            tSgQ: cute.Tensor,
        ):
            layout = cute.select(tSgQ.layout, mode=[2, 0, 1, 3, 4, 5, 6])
            layout = cute.flat_divide(
                layout,
                (4,),  # it relates to 128B swizzle for 16-bit data
            )
            layout = cute.select(layout, mode=[2, 3, 0, 1, 4, 5, 6, 7])
            cpy_tensor = cute.make_tensor(tSgQ.iterator, layout)
            return cpy_tensor

        tSgQ = _get_gQ_cpy(fake_thr_mma_qk.partition_A(gQ))
        tTMAsQ, tTMAgQ = cpasync.tma_partition(
            tma_atom_Q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ_cpy, 0, 3),
            cute.group_modes(tSgQ, 0, 3),
        )

        def _load_Q(
            tTMAgQ: cute.Tensor,
            tTMAsQ: cute.Tensor,
            tma_atom_Q: cute.CopyAtom,
            producer_mbar: cutlass.Pointer,
            m_block_1st: cutlass.Int32,
            m_block_2nd: cutlass.Int32,
            head_idx: cutlass.Int32,
            batch_idx: cutlass.Int32,
        ):
            tTMAgQ_1st = tTMAgQ[None, None, m_block_1st, None, head_idx, batch_idx]
            tTMAgQ_2nd = tTMAgQ[None, None, m_block_2nd, None, head_idx, batch_idx]

            cute.copy(
                tma_atom_Q,
                tTMAgQ_1st[None, 0, 0],
                tTMAsQ[None, 0, 0, 0],
                tma_bar_ptr=producer_mbar,
            )
            cute.copy(
                tma_atom_Q,
                tTMAgQ_1st[None, 1, 0],
                tTMAsQ[None, 1, 0, 0],
                tma_bar_ptr=producer_mbar,
            )
            cute.copy(
                tma_atom_Q,
                tTMAgQ_2nd[None, 0, 0],
                tTMAsQ[None, 0, 1, 0],
                tma_bar_ptr=producer_mbar,
            )
            cute.copy(
                tma_atom_Q,
                tTMAgQ_2nd[None, 1, 0],
                tTMAsQ[None, 1, 1, 0],
                tma_bar_ptr=producer_mbar,
            )

        load_Q = partial(
            _load_Q,
            tTMAgQ=tTMAgQ,
            tTMAsQ=tTMAsQ,
            tma_atom_Q=tma_atom_Q,
        )

        def _get_sK_cpy(
            sK: cute.Tensor,
        ) -> cute.Tensor:
            return _get_sQ_cpy(sK)

        sK_cpy = _get_sK_cpy(sK)

        def _get_gK_cpy(
            tSgK: cute.Tensor,
        ):
            layout = cute.select(tSgK.layout, mode=[2, 0, 1, 3, 4, 5, 6])
            layout = cute.flat_divide(
                layout,
                (4,),  # it relates to 128B swizzle for 16-bit data
            )
            layout = cute.select(layout, mode=[2, 3, 0, 1, 4, 5, 6, 7])
            cpy_tensor = cute.make_tensor(tSgK.iterator, layout)
            return cpy_tensor

        tSgK = _get_gK_cpy(fake_thr_mma_qk.partition_B(gK))
        tTMAsK, tTMAgK = cpasync.tma_partition(
            tma_atom_K,
            0,
            cute.make_layout(1),
            cute.group_modes(sK_cpy, 0, 3),
            cute.group_modes(tSgK, 0, 3),
        )

        def _load_K(
            tTMAgK: cute.Tensor,
            tTMAsK: cute.Tensor,
            tma_atom_K: cute.CopyAtom,
            producer_mbar: cutlass.Pointer,
            buffer_idx: cutlass.Int32,
            n: cutlass.Int32,
            m_block_1st: cutlass.Int32,
            m_block_2nd: cutlass.Int32,
            valid_m_block_2nd: cutlass.Boolean,
            head_idx: cutlass.Int32,
            batch_idx: cutlass.Int32,
            q2k_block_sparse_index: cute.Tensor,
            num_k_blocks: cutlass.Int32,
        ):
            n_block_1st = q2k_block_sparse_index[batch_idx, head_idx, m_block_1st, n]
            n_block_2nd = (
                q2k_block_sparse_index[batch_idx, head_idx, m_block_2nd, n]
                if valid_m_block_2nd
                else num_k_blocks
            )

            tTMAgK_1st = tTMAgK[None, None, n_block_1st, None, head_idx, batch_idx]
            tTMAgK_2nd = tTMAgK[None, None, n_block_2nd, None, head_idx, batch_idx]

            cute.copy(
                tma_atom_K,
                tTMAgK_1st[None, 0, 0],
                tTMAsK[None, 0, 0, buffer_idx],
                tma_bar_ptr=producer_mbar,
            )
            cute.copy(
                tma_atom_K,
                tTMAgK_1st[None, 1, 0],
                tTMAsK[None, 1, 0, buffer_idx],
                tma_bar_ptr=producer_mbar,
            )
            cute.copy(
                tma_atom_K,
                tTMAgK_2nd[None, 0, 0],
                tTMAsK[None, 0, 1, buffer_idx],
                tma_bar_ptr=producer_mbar,
            )
            cute.copy(
                tma_atom_K,
                tTMAgK_2nd[None, 1, 0],
                tTMAsK[None, 1, 1, buffer_idx],
                tma_bar_ptr=producer_mbar,
            )

        load_K = partial(
            _load_K,
            tTMAgK=tTMAgK,
            tTMAsK=tTMAsK,
            tma_atom_K=tma_atom_K,
            q2k_block_sparse_index=q2k_block_sparse_index,
        )

        def _get_sV_cpy(
            sV: cute.Tensor,
        ) -> cute.Tensor:
            layout = cute.flatten(sV.layout)
            layout = cute.select(layout, mode=[4, 0, 1, 2, 3, 5])
            layout = cute.flat_divide(
                layout,
                (4,),  # it relates to 128B swizzle for 16-bit data
            )
            layout = cute.select(layout, mode=[2, 3, 4, 5, 0, 1, 6])
            layout = cute.select(layout, mode=[0, 2, 3, 4, 5, 1, 6])
            layout = cute.group_modes(layout, 0, 2)
            cpy_tensor = cute.make_tensor(sV.iterator, layout)
            return cpy_tensor

        sV_cpy = _get_sV_cpy(sV)

        def _get_gV_cpy(
            tOgV: cute.Tensor,
        ):
            layout = cute.append_ones(tOgV.layout)
            layout = cute.select(layout, mode=[0, 7, 2, 1, 3, 4, 5, 6])
            cpy_tensor = cute.make_tensor(tOgV.iterator, layout)
            return cpy_tensor

        tOgV = _get_gV_cpy(fake_thr_mma_pv.partition_B(gV))

        tTMAsV, tTMAgV = cpasync.tma_partition(
            tma_atom_V,
            0,
            cute.make_layout(1),
            cute.group_modes(sV_cpy, 0, 3),
            cute.group_modes(tOgV, 0, 3),
        )

        def _load_V(
            tTMAgV: cute.Tensor,
            tTMAsV: cute.Tensor,
            tma_atom_V: cute.CopyAtom,
            producer_mbar: cutlass.Pointer,
            buffer_idx: cutlass.Int32,
            n: cutlass.Int32,
            m_block_1st: cutlass.Int32,
            m_block_2nd: cutlass.Int32,
            valid_m_block_2nd: cutlass.Boolean,
            num_v_blocks: cutlass.Int32,
            head_idx: cutlass.Int32,
            batch_idx: cutlass.Int32,
            q2k_block_sparse_index: cute.Tensor,
        ):
            n_block_1st = q2k_block_sparse_index[batch_idx, head_idx, m_block_1st, n]
            n_block_2nd = (
                q2k_block_sparse_index[batch_idx, head_idx, m_block_2nd, n]
                if valid_m_block_2nd
                else num_v_blocks
            )

            tTMAgV_1st = tTMAgV[None, None, None, n_block_1st, head_idx, batch_idx]
            tTMAgV_2nd = tTMAgV[None, None, None, n_block_2nd, head_idx, batch_idx]

            cute.copy(
                tma_atom_V,
                tTMAgV_1st[None, 0, 0],
                tTMAsV[None, 0, 0, buffer_idx],
                tma_bar_ptr=producer_mbar,
            )
            cute.copy(
                tma_atom_V,
                tTMAgV_1st[None, 1, 0],
                tTMAsV[None, 0, 1, buffer_idx],
                tma_bar_ptr=producer_mbar,
            )
            cute.copy(
                tma_atom_V,
                tTMAgV_2nd[None, 0, 0],
                tTMAsV[None, 1, 0, buffer_idx],
                tma_bar_ptr=producer_mbar,
            )
            cute.copy(
                tma_atom_V,
                tTMAgV_2nd[None, 1, 0],
                tTMAsV[None, 1, 1, buffer_idx],
                tma_bar_ptr=producer_mbar,
            )

        load_V = partial(
            _load_V,
            tTMAgV=tTMAgV,
            tTMAsV=tTMAsV,
            tma_atom_V=tma_atom_V,
            q2k_block_sparse_index=q2k_block_sparse_index,
        )

        num_q_blocks = cute.size(gQ, mode=[2])
        num_k_blocks = cute.size(gK, mode=[2])
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m2_idx, head_idx, batch_idx = work_tile.tile_idx

            m_block_1st = m2_idx * 2
            m_block_2nd = m_block_1st + 1
            valid_m_block_2nd: cutlass.Boolean = m_block_2nd < num_q_blocks
            m_block_2nd = m_block_2nd if valid_m_block_2nd else num_q_blocks

            # NOTE: Assumption: different Q-tile has the same number of KV-blocks
            _num_n_blocks = q2k_block_sparse_num[batch_idx, head_idx, m_block_1st]

            cute.arch.mbarrier_wait(
                KV_mbar_ptr + kv_producer_state.index * 2 + 1, kv_producer_state.phase
            )
            # load Q
            load_Q(
                producer_mbar=KV_mbar_ptr + kv_producer_state.index * 2,
                m_block_1st=m_block_1st,
                m_block_2nd=m_block_2nd,
                head_idx=head_idx,
                batch_idx=batch_idx,
            )
            load_K(
                producer_mbar=KV_mbar_ptr + kv_producer_state.index * 2,
                buffer_idx=kv_producer_state.index,
                n=0,
                m_block_1st=m_block_1st,
                m_block_2nd=m_block_2nd,
                valid_m_block_2nd=valid_m_block_2nd,
                num_k_blocks=num_k_blocks,
                head_idx=head_idx,
                batch_idx=batch_idx,
            )

            # load K0
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    KV_mbar_ptr + kv_producer_state.index * 2,
                    self.tma_copy_bytes["Q"] + self.tma_copy_bytes["K"],
                )
            kv_producer_state.advance()

            for n in cutlass.range(_num_n_blocks - 1):
                # load Kn
                cute.arch.mbarrier_wait(
                    KV_mbar_ptr + kv_producer_state.index * 2 + 1, kv_producer_state.phase
                )

                load_K(
                    producer_mbar=KV_mbar_ptr + kv_producer_state.index * 2,
                    buffer_idx=kv_producer_state.index,
                    n=n + 1,
                    m_block_1st=m_block_1st,
                    m_block_2nd=m_block_2nd,
                    valid_m_block_2nd=valid_m_block_2nd,
                    num_k_blocks=num_k_blocks,
                    head_idx=head_idx,
                    batch_idx=batch_idx,
                )

                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        KV_mbar_ptr + kv_producer_state.index * 2, self.tma_copy_bytes["K"]
                    )
                kv_producer_state.advance()
                # load Vn-1
                cute.arch.mbarrier_wait(
                    KV_mbar_ptr + kv_producer_state.index * 2 + 1, kv_producer_state.phase
                )

                load_V(
                    producer_mbar=KV_mbar_ptr + kv_producer_state.index * 2,
                    buffer_idx=kv_producer_state.index,
                    n=n,
                    m_block_1st=m_block_1st,
                    m_block_2nd=m_block_2nd,
                    valid_m_block_2nd=valid_m_block_2nd,
                    num_v_blocks=num_k_blocks,
                    head_idx=head_idx,
                    batch_idx=batch_idx,
                )

                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        KV_mbar_ptr + kv_producer_state.index * 2, self.tma_copy_bytes["V"]
                    )
                kv_producer_state.advance()

            # load Vn
            cute.arch.mbarrier_wait(
                KV_mbar_ptr + kv_producer_state.index * 2 + 1, kv_producer_state.phase
            )

            load_V(
                producer_mbar=KV_mbar_ptr + kv_producer_state.index * 2,
                buffer_idx=kv_producer_state.index,
                n=_num_n_blocks - 1,
                m_block_1st=m_block_1st,
                m_block_2nd=m_block_2nd,
                valid_m_block_2nd=valid_m_block_2nd,
                num_v_blocks=num_k_blocks,
                head_idx=head_idx,
                batch_idx=batch_idx,
            )

            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    KV_mbar_ptr + kv_producer_state.index * 2, self.tma_copy_bytes["V"]
                )
            kv_producer_state.advance()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma(
        self,
        TileSchedulerCls: Callable,
        KV_mbar_ptr: cutlass.Pointer,
        kv_consumer_state: pipeline.PipelineState,
        q2k_block_sparse_num: cute.Tensor,
        qk_mma_pipeline: pipeline.PipelineUmmaAsync,
        qk_mma_producer_state: pipeline.PipelineState,
        tiled_mma_qk: cute.TiledMma,
        p_pipeline: pipeline.PipelineAsyncUmma,
        p_consumer_state: pipeline.PipelineState,
        o_pipeline: pipeline.PipelineAsyncUmma,
        o_consumer_state: pipeline.PipelineState,
        pv_mma_pipeline: pipeline.PipelineUmmaAsync,
        pv_mma_producer_state: pipeline.PipelineState,
        tiled_mma_pv: cute.TiledMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sP: cute.Tensor,
        tCtS: cute.Tensor,
        tCtO: cute.Tensor,
    ):
        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_pv = tiled_mma_pv.get_slice(0)

        tCsQ = thr_mma_qk.make_fragment_A(sQ)
        tCsK = thr_mma_qk.make_fragment_B(sK)

        tCsP = None
        if cutlass.const_expr(self.p_in_smem):
            tCsP = thr_mma_pv.make_fragment_A(sP)
        else:
            # it resides in TMEM
            tCsP = sP
        tCsV = thr_mma_pv.make_fragment_B(sV)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m2_idx, head_idx, batch_idx = work_tile.tile_idx

            m_block_1st = m2_idx * 2
            _num_n_blocks = q2k_block_sparse_num[batch_idx, head_idx, m_block_1st]

            # Q @ K0
            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
            cute.arch.mbarrier_wait(
                KV_mbar_ptr + kv_consumer_state.index * 2, kv_consumer_state.phase
            )
            qk_mma_pipeline.producer_acquire(qk_mma_producer_state)
            for kblock_idx in cutlass.range_constexpr(cute.size(sQ, mode=[2])):
                cute.gemm(
                    tiled_mma_qk,
                    cute.append_ones(tCtS[None, None, qk_mma_producer_state.index]),
                    tCsQ[None, None, kblock_idx, 0],
                    tCsK[None, None, kblock_idx, kv_consumer_state.index],
                    cute.append_ones(tCtS[None, None, qk_mma_producer_state.index]),
                )
                tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
            qk_mma_pipeline.producer_commit(qk_mma_producer_state)
            qk_mma_producer_state.advance()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    KV_mbar_ptr + kv_consumer_state.index * 2 + 1, 0
                )
            kv_consumer_state.advance()

            for n in cutlass.range(_num_n_blocks - 1):
                # Q @ Kn
                tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                cute.arch.mbarrier_wait(
                    KV_mbar_ptr + kv_consumer_state.index * 2, kv_consumer_state.phase
                )
                qk_mma_pipeline.producer_acquire(qk_mma_producer_state)
                for kblock_idx in cutlass.range_constexpr(cute.size(sQ, mode=[2])):
                    cute.gemm(
                        tiled_mma_qk,
                        cute.append_ones(tCtS[None, None, qk_mma_producer_state.index]),
                        tCsQ[None, None, kblock_idx, 0],
                        tCsK[None, None, kblock_idx, kv_consumer_state.index],
                        cute.append_ones(tCtS[None, None, qk_mma_producer_state.index]),
                    )
                    tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)

                qk_mma_pipeline.producer_commit(qk_mma_producer_state)
                qk_mma_producer_state.advance()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        KV_mbar_ptr + kv_consumer_state.index * 2 + 1, 0
                    )
                kv_consumer_state.advance()

                # P @ Vn-1
                tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, n >= self.o_stage)
                p_pipeline.consumer_wait(p_consumer_state)
                o_pipeline.consumer_wait(o_consumer_state)
                cute.arch.mbarrier_wait(
                    KV_mbar_ptr + kv_consumer_state.index * 2, kv_consumer_state.phase
                )
                pv_mma_pipeline.producer_acquire(pv_mma_producer_state)
                for kblock_idx in cutlass.range_constexpr(cute.size(sV, mode=[2])):
                    cute.gemm(
                        tiled_mma_pv,
                        cute.append_ones(tCtO[None, None, pv_mma_producer_state.index]),
                        tCsP[None, None, kblock_idx, p_consumer_state.index],
                        tCsV[None, None, kblock_idx, kv_consumer_state.index],
                        cute.append_ones(tCtO[None, None, pv_mma_producer_state.index]),
                    )
                    tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
                pv_mma_pipeline.producer_commit(pv_mma_producer_state)
                pv_mma_producer_state.advance()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        KV_mbar_ptr + kv_consumer_state.index * 2 + 1, 0
                    )
                kv_consumer_state.advance()
                p_pipeline.consumer_release(p_consumer_state)
                p_consumer_state.advance()
                o_pipeline.consumer_release(o_consumer_state)
                o_consumer_state.advance()

            # P @ Vn
            tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, (_num_n_blocks - 1) >= self.o_stage)
            p_pipeline.consumer_wait(p_consumer_state)
            o_pipeline.consumer_wait(o_consumer_state)
            cute.arch.mbarrier_wait(
                KV_mbar_ptr + kv_consumer_state.index * 2, kv_consumer_state.phase
            )
            pv_mma_pipeline.producer_acquire(pv_mma_producer_state)
            for kblock_idx in cutlass.range_constexpr(cute.size(sV, mode=[2])):
                cute.gemm(
                    tiled_mma_pv,
                    cute.append_ones(tCtO[None, None, pv_mma_producer_state.index]),
                    tCsP[None, None, kblock_idx, p_consumer_state.index],
                    tCsV[None, None, kblock_idx, kv_consumer_state.index],
                    cute.append_ones(tCtO[None, None, pv_mma_producer_state.index]),
                )
                tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
            pv_mma_pipeline.producer_commit(pv_mma_producer_state)
            pv_mma_producer_state.advance()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    KV_mbar_ptr + kv_consumer_state.index * 2 + 1, 0
                )
            kv_consumer_state.advance()
            p_pipeline.consumer_release(p_consumer_state)
            p_consumer_state.advance()
            o_pipeline.consumer_release(o_consumer_state)
            o_consumer_state.advance()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def exp2f(
        self,
        value: cutlass.Float32,
    ) -> cutlass.Float32:
        return cute.arch.exp2(value)

    @cute.jit
    def update_row_max(
        self,
        max_new: cutlass.Float32,
        max_old: cutlass.Float32,
        is_first: cutlass.Boolean,
        sm_scale_log2: cutlass.Float32,
    ):
        _max_safe = max_new if max_new != -cutlass.Float32.inf else 0.0
        acc_scale = 0.0
        if cutlass.const_expr(not is_first):
            acc_scale_ = (max_old - _max_safe) * sm_scale_log2
            acc_scale = self.exp2f(acc_scale_)
            if cutlass.const_expr(self.rescale_threshold > 0.0):
                if acc_scale_ >= -self.rescale_threshold:
                    _max_safe = max_old
                    acc_scale = 1.0
        return _max_safe, acc_scale

    @cute.jit
    def mask_then_cal_local_max(
        self,
        tensor: cute.Tensor,
        right_bound: cutlass.Int32,
    ) -> cutlass.Float32:
        def clamp(
            tensor: cute.Tensor,
            index: cutlass.Int32,
            right_bound: cutlass.Int32,
        ) -> cutlass.Float32:
            tensor[index] = tensor[index] if (index < right_bound) else -cutlass.Float32.inf
            return tensor[index]

        _clamp = partial(clamp, tensor=tensor, right_bound=right_bound)

        if cutlass.const_expr(cute.size(tensor, mode=[0]) < 8):
            _max = -cutlass.Float32.inf
            for i in cutlass.range_constexpr(0, cute.size(tensor, mode=[0]), 2):
                _max = ptx.max3f(_max, _clamp(index=i), _clamp(index=i + 1))
            return _max
        else:
            local_max = [
                ptx.max3f(_clamp(index=0), _clamp(index=1), -cutlass.Float32.inf),
                ptx.max3f(_clamp(index=2), _clamp(index=3), -cutlass.Float32.inf),
                ptx.max3f(_clamp(index=4), _clamp(index=5), -cutlass.Float32.inf),
                ptx.max3f(_clamp(index=6), _clamp(index=7), -cutlass.Float32.inf),
            ]
            for i in cutlass.range_constexpr(8, cute.size(tensor, mode=[0]), 8):
                local_max[0] = ptx.max3f(local_max[0], _clamp(index=i), _clamp(index=i + 1))
                local_max[1] = ptx.max3f(local_max[1], _clamp(index=i + 2), _clamp(index=i + 3))
                local_max[2] = ptx.max3f(local_max[2], _clamp(index=i + 4), _clamp(index=i + 5))
                local_max[3] = ptx.max3f(local_max[3], _clamp(index=i + 6), _clamp(index=i + 7))
            local_max[0] = cute.arch.fmax(local_max[0], local_max[1])
            return ptx.max3f(local_max[0], local_max[2], local_max[3])

    @cute.jit
    def cal_local_max(
        self,
        tensor: cute.Tensor,
    ) -> cutlass.Float32:
        # Full-block fast path: no column is masked, so read directly without
        # the per-element bound test / -inf write that mask_then_cal_local_max
        # performs. Mirrors the masked variant's reduction structure exactly.
        if cutlass.const_expr(cute.size(tensor, mode=[0]) < 8):
            _max = -cutlass.Float32.inf
            for i in cutlass.range_constexpr(0, cute.size(tensor, mode=[0]), 2):
                _max = ptx.max3f(_max, tensor[i], tensor[i + 1])
            return _max
        else:
            local_max = [
                ptx.max3f(tensor[0], tensor[1], -cutlass.Float32.inf),
                ptx.max3f(tensor[2], tensor[3], -cutlass.Float32.inf),
                ptx.max3f(tensor[4], tensor[5], -cutlass.Float32.inf),
                ptx.max3f(tensor[6], tensor[7], -cutlass.Float32.inf),
            ]
            for i in cutlass.range_constexpr(8, cute.size(tensor, mode=[0]), 8):
                local_max[0] = ptx.max3f(local_max[0], tensor[i], tensor[i + 1])
                local_max[1] = ptx.max3f(local_max[1], tensor[i + 2], tensor[i + 3])
                local_max[2] = ptx.max3f(local_max[2], tensor[i + 4], tensor[i + 5])
                local_max[3] = ptx.max3f(local_max[3], tensor[i + 6], tensor[i + 7])
            local_max[0] = cute.arch.fmax(local_max[0], local_max[1])
            return ptx.max3f(local_max[0], local_max[2], local_max[3])

    @cute.jit
    def cal_local_sum(
        self,
        tensor: cute.Tensor,
        init: cutlass.Float32,
    ) -> cutlass.Float32:
        if cutlass.const_expr(cute.size(tensor, mode=[0]) < 8):
            _sum = init
            for i in cutlass.range_constexpr(cute.size(tensor, mode=[0])):
                _sum += tensor[i]
            return _sum
        else:
            local_sum = [
                cute.arch.add_packed_f32x2((init, 0.0), (tensor[0], tensor[1])),
                (tensor[2], tensor[3]),
                (tensor[4], tensor[5]),
                (tensor[6], tensor[7]),
            ]
            for i in cutlass.range_constexpr(8, cute.size(tensor, mode=[0]), 8):
                local_sum[0] = cute.arch.add_packed_f32x2(
                    local_sum[0], (tensor[i + 0], tensor[i + 1])
                )
                local_sum[1] = cute.arch.add_packed_f32x2(
                    local_sum[1], (tensor[i + 2], tensor[i + 3])
                )
                local_sum[2] = cute.arch.add_packed_f32x2(
                    local_sum[2], (tensor[i + 4], tensor[i + 5])
                )
                local_sum[3] = cute.arch.add_packed_f32x2(
                    local_sum[3], (tensor[i + 6], tensor[i + 7])
                )
            local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[1])
            local_sum[2] = cute.arch.add_packed_f32x2(local_sum[2], local_sum[3])
            local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[2])
            return local_sum[0][0] + local_sum[0][1]

    @cute.jit
    def softmax_step(
        self,
        which: cutlass.Constexpr[cutlass.Int32],
        stage: cutlass.Constexpr[cutlass.Int32],
        n: cutlass.Int32,
        q2k_block_sparse_index: cute.Tensor,
        sVariable_block_sizes: cute.Tensor,
        p_producer_state: pipeline.PipelineState,
        sP: cute.Tensor,
        tCcS_ld: cute.Tensor,
        qk_mma_pipeline: pipeline.PipelineUmmaAsync,
        p_pipeline: pipeline.PipelineAsyncUmma,
        qk_mma_consumer_state: pipeline.PipelineState,
        tiled_copy_t2r: cute.TiledCopy,
        tiled_copy_r2t: Optional[cute.TiledCopy],
        tCrS_ld_half_zeros: Optional[cute.Tensor],
        tCtS_ld: cute.Tensor,
        tCrS_ld: cute.Tensor,
        tCrS_ld_half: cute.Tensor,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block_1st: cutlass.Int32,
        m_block_2nd: cutlass.Int32,
        tidx: cutlass.Int32,
        running_max: cute.Tensor,
        running_sum: cute.Tensor,
        sm_scale_log2: cutlass.Float32,
        sScale: cute.Tensor,
        correction_pipeline: pipeline.PipelineAsync,
        is_first: cutlass.Boolean,
    ):
        n_size = 0
        if cutlass.const_expr(which == 0):
            n_block_1st = q2k_block_sparse_index[
                batch_idx, head_idx, m_block_1st, n
            ]  # TODO: move to smem
            n_size = sVariable_block_sizes[n_block_1st]
        elif cutlass.const_expr(which == 1):
            n_block_2nd = q2k_block_sparse_index[batch_idx, head_idx, m_block_2nd, n]
            n_size = sVariable_block_sizes[n_block_2nd]

        qk_mma_pipeline.consumer_wait(qk_mma_consumer_state)
        correction_pipeline.producer_acquire(p_producer_state)
        cute.copy(
            tiled_copy_t2r,
            tCtS_ld[None, None, None, 0, which, qk_mma_consumer_state.index],
            tCrS_ld,
        )
        # calculate P
        # Full-block fast path: when the KV block is full (n_size == tile width,
        # i.e. BLOCK==64 at the focus shapes) the variable-block mask masks
        # nothing every iteration. Skip the per-element bound test / -inf write
        # and take the unmasked local-max path. The masked path is preserved for
        # the partial-block case (n_size < tile width) so correctness holds.
        # NOTE: predeclare _max before the dynamic branch — CuTe DSL does not
        # propagate variables first bound inside dynamic control flow.
        _max = -cutlass.Float32.inf
        if n_size >= cute.size(tCrS_ld, mode=[0]):
            _max = self.cal_local_max(tCrS_ld)
        else:
            _max = self.mask_then_cal_local_max(tCrS_ld, n_size)
        _max_safe, _acc_scale = self.update_row_max(
            max_new=_max,
            is_first=is_first,
            max_old=running_max[0],
            sm_scale_log2=sm_scale_log2,
        )
        if cutlass.const_expr(self.o_stage == 1):
            sScale[tidx, p_producer_state.index] = _acc_scale
        else:
            acc_scale = cute.arch.exp2(
                (sScale[1, tidx, p_producer_state.index] - _max_safe) * sm_scale_log2
            )
            sScale[0, tidx, p_producer_state.index] = acc_scale
            sScale[1, tidx, p_producer_state.index] = _max_safe
        correction_pipeline.producer_commit(p_producer_state)

        running_max[0] = _max_safe
        minus_coeff = -_max_safe * sm_scale_log2
        for i in cutlass.range(0, cute.size(tCrS_ld.shape), 2, unroll_full=True):
            tCrS_ld[i], tCrS_ld[i + 1] = cute.arch.fma_packed_f32x2(
                (tCrS_ld[i], tCrS_ld[i + 1]),
                (sm_scale_log2, sm_scale_log2),
                (minus_coeff, minus_coeff),
            )
            tCrS_ld[i] = self.exp2f(tCrS_ld[i])
            tCrS_ld[i + 1] = self.exp2f(tCrS_ld[i + 1])

        # type conversion
        tCrS_ld_half.store(tCrS_ld.load().to(tCrS_ld_half.element_type))

        # update running sum
        running_sum[0] *= _acc_scale
        running_sum[0] = self.cal_local_sum(tCrS_ld, running_sum[0])

        p_pipeline.producer_acquire(p_producer_state)
        if cutlass.const_expr(self.p_in_smem):
            # copy P to SMEM
            cute.autovec_copy(tCrS_ld_half, sP[None, which, p_producer_state.index])
        else:
            # copy P to TMEM
            cute.copy(
                tiled_copy_r2t, tCrS_ld_half, sP[None, None, None, 0, which, p_producer_state.index]
            )
            if cutlass.const_expr(self.p_in_s):
                cute.copy(
                    tiled_copy_r2t,
                    tCrS_ld_half_zeros,
                    sP[None, None, None, 0, which ^ 1, p_producer_state.index],
                )
        p_pipeline.producer_commit(p_producer_state)

        qk_mma_pipeline.consumer_release(qk_mma_consumer_state)

    @cute.jit
    def softmax_loop(
        self,
        TileSchedulerCls: Callable,
        q2k_block_sparse_num: cute.Tensor,
        p_producer_state: pipeline.PipelineState,
        softmax_step: Callable,
        qk_mma_consumer_state: pipeline.PipelineState,
        sFinal: cute.Tensor,
        sFinal_pipeline: pipeline.PipelineAsync,
        sFinal_producer_state: pipeline.PipelineState,
        num_q_blocks: cutlass.Int32,
        tidx: cutlass.Int32,
        which: cutlass.Constexpr[cutlass.Int32],
        corr_pipeline: pipeline.PipelineAsync,
        O_final_guard: cutlass.Pointer,
        sScale: cute.Tensor,
    ):
        running_states = cute.make_rmem_tensor(cute.make_layout((2, 1)), self.acc_dtype)
        running_sum = running_states[0, None]
        running_max = running_states[1, None]

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        O_final_phase = 1
        while work_tile.is_valid_tile:
            m2_idx, head_idx, batch_idx = work_tile.tile_idx

            m_block_1st = m2_idx * 2
            _num_n_blocks = q2k_block_sparse_num[batch_idx, head_idx, m_block_1st]
            m_block_2nd = m_block_1st + 1 if m_block_1st + 1 < num_q_blocks else m_block_1st

            cute.arch.mbarrier_wait(O_final_guard, O_final_phase)
            O_final_phase ^= 1

            running_max.fill(-cutlass.Float32.inf)
            running_sum.fill(0.0)
            if cutlass.const_expr(self.o_stage != 1):
                sScale[1, tidx, None].fill(-cutlass.Float32.inf)

            softmax_step(
                n=0,
                stage=0,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block_1st=m_block_1st,
                m_block_2nd=m_block_2nd,
                p_producer_state=p_producer_state,
                qk_mma_consumer_state=qk_mma_consumer_state,
                running_max=running_max,
                running_sum=running_sum,
                which=which,
                correction_pipeline=corr_pipeline,
                is_first=True,
            )
            p_producer_state.advance()
            qk_mma_consumer_state.advance()

            for n in cutlass.range(1, _num_n_blocks):
                softmax_step(
                    n=n,
                    stage=0,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    m_block_1st=m_block_1st,
                    m_block_2nd=m_block_2nd,
                    p_producer_state=p_producer_state,
                    qk_mma_consumer_state=qk_mma_consumer_state,
                    running_max=running_max,
                    running_sum=running_sum,
                    which=which,
                    correction_pipeline=corr_pipeline,
                    is_first=False,
                )
                p_producer_state.advance()
                qk_mma_consumer_state.advance()

            # put running stages to SMEM
            sFinal_pipeline.producer_acquire(sFinal_producer_state)
            cute.autovec_copy(
                running_states, cute.append_ones(sFinal[None, tidx, sFinal_producer_state.index])
            )
            sFinal_pipeline.producer_commit(sFinal_producer_state)
            sFinal_producer_state.advance()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def softmax(
        self,
        TileSchedulerCls: Callable,
        q2k_block_sparse_num: cute.Tensor,
        q2k_block_sparse_index: cute.Tensor,
        variable_block_sizes: cute.Tensor,
        sVariable_block_sizes: cute.Tensor,
        qk_mma_pipeline: pipeline.PipelineUmmaAsync,
        qk_mma_consumer_state: pipeline.PipelineState,
        p_pipeline: pipeline.PipelineAsyncUmma,
        p_producer_state: pipeline.PipelineState,
        num_q_blocks: cutlass.Int32,
        sP: cute.Tensor,
        tCtS: cute.Tensor,
        thr_mma_qk: cute.core.ThrMma,
        sm_scale_log2: cutlass.Float32,
        sScale: cute.Tensor,
        corr_pipeline: pipeline.PipelineAsync,
        sFinal: cute.Tensor,
        sFinal_pipeline: pipeline.PipelineAsync,
        sFinal_producer_state: pipeline.PipelineState,
        O_final_guard: cutlass.Pointer,
    ):
        tidx = cute.arch.thread_idx()[0] % (self.threads_per_warp * len(self.softmax_warp_ids))
        in_which = cute.arch.make_warp_uniform(tidx // self.block_m)

        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.block_n)),
            self.acc_dtype,
        )
        tS_load = cute.flat_divide(
            tCtS[(None, None), 0, None], (self.mma_tiler_qk[0], self.block_n)
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tS_load[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

        tCtS_ld = thr_copy_t2r.partition_S(tS_load)

        cS = cute.make_identity_tensor(self.mma_tiler_qk[:2])
        tCcS = thr_mma_qk.partition_C(cS)
        cS_load = cute.flat_divide(
            tCcS[(None, None), 0, None], (self.mma_tiler_qk[0], self.block_n)
        )
        tCcS_ld = thr_copy_t2r.partition_D(cS_load)
        tCrS_ld = cute.make_fragment(cute.select(tCcS_ld.shape, mode=[0, 1, 2]), self.acc_dtype)
        tCrS_ld_half = cute.make_fragment(tCrS_ld.layout, sP.element_type)

        sP_cpy_slice = None
        tiled_copy_r2t = None
        tCrS_ld_half_zeros = None
        if cutlass.const_expr(self.p_in_smem):
            tv_layout = cute.make_ordered_layout(
                (self.threads_per_wg, self.mma_tiler_qk[1], self.s_stage), (0, 1, 2)
            )
            sP_cpy = cute.composition(sP, tv_layout)
            sP_cpy_slice = cute.flatten(sP_cpy[tidx, None, None])
        else:
            copy_atom_r2t = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(self.block_n // 2)), sP.element_type
            )

            def _get_tP_store(tP: cute.Tensor):
                layout = cute.flatten(tP.layout)
                mode1 = cute.make_layout(
                    cute.get(layout, mode=[1]).shape * cute.get(layout, mode=[3]).shape,
                    stride=cute.get(layout, mode=[1]).stride,
                )
                shape = (
                    cute.get(layout, mode=[0]).shape,
                    mode1.shape,
                    cute.get(layout, mode=[2]).shape,
                    cute.get(layout, mode=[4]).shape,
                    cute.get(layout, mode=[5]).shape,
                )
                stride = (
                    cute.get(layout, mode=[0]).stride,
                    mode1.stride,
                    cute.get(layout, mode=[2]).stride,
                    cute.get(layout, mode=[4]).stride,
                    cute.get(layout, mode=[5]).stride,
                )
                layout = cute.make_layout(shape, stride=stride)
                return cute.make_tensor(tP.iterator, layout)

            tP_store = _get_tP_store(sP)

            tiled_copy_r2t = tcgen05.make_tmem_copy(copy_atom_r2t, tP_store[(None, None, 0, 0, 0)])
            thr_copy_r2t = tiled_copy_r2t.get_slice(tidx)

            tCtP_st = thr_copy_r2t.partition_D(tP_store)
            sP_cpy_slice = tCtP_st
            if cutlass.const_expr(self.p_in_s):
                tCrS_ld_half_zeros = cute.make_rmem_tensor(
                    tCrS_ld_half.layout, tCrS_ld_half.element_type
                )
                tCrS_ld_half_zeros.fill(0.0)
            else:
                # P has its own TMEM
                _zeros = cute.make_rmem_tensor(tCrS_ld_half.layout, tCrS_ld_half.element_type)
                _zeros.fill(0.0)
                for stage in cutlass.range_constexpr(self.s_stage):
                    cute.copy(
                        tiled_copy_r2t, _zeros, tCtP_st[None, None, None, 0, in_which ^ 1, stage]
                    )

        _softmax_step = partial(
            self.softmax_step,
            q2k_block_sparse_index=q2k_block_sparse_index,
            sVariable_block_sizes=sVariable_block_sizes,
            sP=sP_cpy_slice,
            tCcS_ld=tCcS_ld,
            qk_mma_pipeline=qk_mma_pipeline,
            p_pipeline=p_pipeline,
            tiled_copy_t2r=tiled_copy_t2r,
            tiled_copy_r2t=tiled_copy_r2t,
            tCrS_ld_half_zeros=tCrS_ld_half_zeros,
            tCtS_ld=tCtS_ld,
            tCrS_ld=tCrS_ld,
            tCrS_ld_half=tCrS_ld_half,
            tidx=tidx,
            sm_scale_log2=sm_scale_log2,
            sScale=sScale,
        )

        _softmax_loop = partial(
            self.softmax_loop,
            TileSchedulerCls=TileSchedulerCls,
            q2k_block_sparse_num=q2k_block_sparse_num,
            p_producer_state=p_producer_state,
            softmax_step=_softmax_step,
            qk_mma_consumer_state=qk_mma_consumer_state,
            sFinal=sFinal,
            sFinal_pipeline=sFinal_pipeline,
            sFinal_producer_state=sFinal_producer_state,
            num_q_blocks=num_q_blocks,
            tidx=tidx,
            O_final_guard=O_final_guard,
            sScale=sScale,
        )

        if cutlass.const_expr(self.p_in_smem):
            sP_cpy = cute.composition(
                sP,
                cute.make_ordered_layout(
                    (self.threads_per_wg, self.mma_tiler_qk[1] * self.s_stage), (0, 1)
                ),
            )
            sP_cpy_slice = cute.flatten(sP_cpy[tidx, None])
            _zeros = cute.make_rmem_tensor((self.block_n, 1), sP_cpy_slice.element_type)
            _zeros.fill(0.0)

            for stage in cutlass.range_constexpr(self.s_stage):
                cute.autovec_copy(_zeros, sP_cpy_slice[None, stage * 2 + (in_which ^ 1)])
        if in_which == 0:
            _softmax_loop(which=0, corr_pipeline=corr_pipeline)
        else:
            _softmax_loop(which=1, corr_pipeline=corr_pipeline)

    @cute.jit
    def correction_loop(
        self,
        TileSchedulerCls: Callable,
        variable_block_sizes: cute.Tensor,
        q2k_block_sparse_num: cute.Tensor,
        correction_pipeline: pipeline.PipelineAsync,
        p_producer_state: pipeline.PipelineState,
        o_pipeline: pipeline.PipelineAsyncUmma,
        o_producer_state: pipeline.PipelineState,
        correction_consumer_state: pipeline.PipelineState,
        corr_ld_repeat: cutlass.Constexpr[cutlass.Int32],
        corr_tiled_copy_t2r: cute.TiledCopy,
        corr_tCtO_ld: cute.Tensor,
        corr_tCrO_ld: cute.Tensor,
        corr_tiled_copy_r2t: cute.TiledCopy,
        pv_mma_pipeline: pipeline.PipelineUmmaAsync,
        pv_mma_consumer_state: pipeline.PipelineState,
        st_O_pipeline: pipeline.PipelineAsync,
        st_O_producer_state: pipeline.PipelineState,
        tv_layout: cute.Layout,
        sO: cute.Tensor,
        sScale: cute.Tensor,
        wb_tCcO_ld: cute.Tensor,
        wb_tCrO_ld: cute.Tensor,
        sFinal: cute.Tensor,
        sFinal_pipeline: pipeline.PipelineAsync,
        sFinal_consumer_state: pipeline.PipelineState,
        wb_ld_repeat: cutlass.Constexpr[cutlass.Int32],
        wb_tCrO_reduction: Optional[cute.Tensor],
        wb_tiled_copy_t2r: cute.TiledCopy,
        wb_tCtO_ld: cute.Tensor,
        wb_tCrO_ld_half: cute.Tensor,
        tidx: cutlass.Int32,
        sm_scale_log2: cutlass.Float32,
        num_q_blocks: cutlass.Int32,
        O_final_guard: cutlass.Pointer,
        gLSE: cute.Tensor,
        in_which: cutlass.Int32,
    ):
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m2_idx, head_idx, batch_idx = work_tile.tile_idx

            _num_n_blocks = q2k_block_sparse_num[batch_idx, head_idx, m2_idx * 2]

            # non-rescaling
            for n in cutlass.range_constexpr(self.o_stage):
                correction_pipeline.consumer_wait(correction_consumer_state)
                correction_pipeline.consumer_release(correction_consumer_state)
                p_producer_state.advance()
                correction_consumer_state.advance()

                o_pipeline.producer_acquire(o_producer_state)
                o_pipeline.producer_commit(o_producer_state)
                o_producer_state.advance()

                pv_mma_pipeline.consumer_wait(pv_mma_consumer_state)
                pv_mma_pipeline.consumer_release(pv_mma_consumer_state)
                pv_mma_consumer_state.advance()

            for n in cutlass.range(self.o_stage, _num_n_blocks):
                correction_pipeline.consumer_wait(correction_consumer_state)
                o_pipeline.producer_acquire(o_producer_state)

                acc_scale = 0.0
                if cutlass.const_expr(self.o_stage == 1):
                    acc_scale = sScale[tidx, p_producer_state.index]
                else:
                    acc_scale = sScale[0, tidx, p_producer_state.index]
                should_rescale = cute.arch.vote_ballot_sync(acc_scale < 1.0) != 0
                if should_rescale:
                    for repeat in cutlass.range(corr_ld_repeat):
                        cute.copy(
                            corr_tiled_copy_t2r,
                            corr_tCtO_ld[None, None, None, 0, repeat, pv_mma_consumer_state.index],
                            corr_tCrO_ld,
                        )

                        # apply correction
                        for i in cutlass.range_constexpr(0, cute.size(corr_tCrO_ld, mode=[0]), 2):
                            corr_tCrO_ld[i], corr_tCrO_ld[i + 1] = cute.arch.mul_packed_f32x2(
                                (corr_tCrO_ld[i], corr_tCrO_ld[i + 1]),
                                (acc_scale, acc_scale),
                            )

                        cute.copy(
                            corr_tiled_copy_r2t,
                            corr_tCrO_ld,
                            corr_tCtO_ld[None, None, None, 0, repeat, pv_mma_consumer_state.index],
                        )
                o_pipeline.producer_commit(o_producer_state)
                o_producer_state.advance()

                correction_pipeline.consumer_release(correction_consumer_state)
                p_producer_state.advance()
                correction_consumer_state.advance()

                # currently, we don't need pv_mma_result
                pv_mma_pipeline.consumer_wait(pv_mma_consumer_state)
                pv_mma_pipeline.consumer_release(pv_mma_consumer_state)
                pv_mma_consumer_state.advance()

            # ----- Correction Epilogue: Store O to SMEM ----- #
            st_O_pipeline.producer_acquire(st_O_producer_state)
            running_states = cute.make_rmem_tensor(cute.make_layout((2, 1)), self.acc_dtype)
            running_sum = running_states[0, None]
            running_max = running_states[1, None]

            sFinal_pipeline.consumer_wait(sFinal_consumer_state)
            cute.autovec_copy(
                cute.append_ones(sFinal[None, tidx, sFinal_consumer_state.index]), running_states
            )
            sFinal_pipeline.consumer_release(sFinal_consumer_state)
            sFinal_consumer_state.advance()

            # calculate LSE
            lse = (
                running_max[0] * sm_scale_log2 + cute.math.log2(running_sum[0], fastmath=True)
            ) * self.LN2

            gLSE_1st = gLSE[None, m2_idx * 2, head_idx, batch_idx]
            gLSE_2nd = gLSE[None, m2_idx * 2 + 1, head_idx, batch_idx]
            if in_which == 0:
                gLSE_1st[tidx] = lse
            elif m2_idx * 2 + 1 < num_q_blocks:
                gLSE_2nd[tidx - self.block_m] = lse

            # calculate scale
            scale = cute.arch.rcp_approx(running_sum[0])

            if cutlass.const_expr(self.o_stage == 1):
                for repeat in cutlass.range(wb_ld_repeat):
                    cute.copy(
                        wb_tiled_copy_t2r,
                        wb_tCtO_ld[None, None, None, 0, repeat, pv_mma_consumer_state.index],
                        wb_tCrO_ld,
                    )
                    # scale
                    for i in cutlass.range_constexpr(0, cute.size(wb_tCrO_ld, mode=[0]), 2):
                        wb_tCrO_ld[i], wb_tCrO_ld[i + 1] = cute.arch.mul_packed_f32x2(
                            (wb_tCrO_ld[i], wb_tCrO_ld[i + 1]),
                            (scale, scale),
                        )
                    # type conversion
                    wb_tCrO_ld_half.store(wb_tCrO_ld.load().to(wb_tCrO_ld_half.element_type))

                    # store to SMEM
                    cute.autovec_copy(wb_tCrO_ld_half, sO[None, repeat, st_O_producer_state.index])
            else:
                stage_in_use = pv_mma_consumer_state.index - 1
                stage_in_use = stage_in_use if stage_in_use != -1 else self.o_stage - 1
                for repeat in cutlass.range(wb_ld_repeat):
                    wb_tCrO_reduction.fill(0.0)
                    for stage in cutlass.range(cutlass.min(self.o_stage, _num_n_blocks)):
                        _stage_id = (stage_in_use + stage) % self.o_stage
                        cute.copy(
                            wb_tiled_copy_t2r,
                            wb_tCtO_ld[None, None, None, 0, repeat, _stage_id],
                            wb_tCrO_ld,
                        )

                        _scale = scale
                        if cutlass.const_expr(self.o_stage != 1):
                            acc_scale = cute.arch.exp2(
                                (sScale[1, tidx, _stage_id] - running_max[0]) * sm_scale_log2
                            )
                            _scale *= acc_scale

                        # scale and reduction
                        for i in cutlass.range_constexpr(0, cute.size(wb_tCrO_ld, mode=[0]), 2):
                            wb_tCrO_reduction[i], wb_tCrO_reduction[i + 1] = (
                                cute.arch.fma_packed_f32x2(
                                    (wb_tCrO_ld[i], wb_tCrO_ld[i + 1]),
                                    (_scale, _scale),
                                    (wb_tCrO_reduction[i], wb_tCrO_reduction[i + 1]),
                                )
                            )
                    # type conversion
                    wb_tCrO_ld_half.store(wb_tCrO_reduction.load().to(wb_tCrO_ld_half.element_type))

                    # store to SMEM
                    cute.autovec_copy(wb_tCrO_ld_half, sO[None, repeat, st_O_producer_state.index])

            # NOTE: It is safe now to do next wave
            cute.arch.mbarrier_arrive_and_expect_tx(O_final_guard, 0)

            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            st_O_pipeline.producer_commit(st_O_producer_state)
            st_O_producer_state.advance()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def correction(
        self,
        TileSchedulerCls: Callable,
        q2k_block_sparse_num: cute.Tensor,
        p_producer_state: pipeline.PipelineState,
        o_pipeline: pipeline.PipelineAsyncUmma,
        o_producer_state: pipeline.PipelineState,
        pv_mma_pipeline: pipeline.PipelineUmmaAsync,
        pv_mma_consumer_state: pipeline.PipelineState,
        st_O_pipeline: pipeline.PipelineAsync,
        st_O_producer_state: pipeline.PipelineState,
        tCtO: cute.Tensor,
        thr_mma_pv: cute.core.ThrMma,
        sO: cute.Tensor,
        corr_pipeline: pipeline.PipelineAsync,
        correction_consumer_state: pipeline.PipelineState,
        sScale: cute.Tensor,
        sm_scale_log2: cutlass.Float32,
        num_q_blocks: cutlass.Int32,
        variable_block_sizes: cute.Tensor,
        sFinal: cute.Tensor,
        sFinal_pipeline: pipeline.PipelineAsync,
        sFinal_consumer_state: pipeline.PipelineState,
        O_final_guard: cutlass.Pointer,
        gLSE: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0] % (self.threads_per_warp * len(self.correction_warp_ids))
        in_which = cute.arch.make_warp_uniform(tidx // self.block_m)
        # A-tmem: widen correction TMEM<->reg copy 32->64 cols (halve LDTM/STTM count)
        corr_ld_inst: int = 64
        corr_ld_repeat: int = cute.ceil_div(self.mma_tiler_pv[1], corr_ld_inst)

        corr_copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(
                tcgen05.copy.Repetition(corr_ld_inst),
            ),
            self.acc_dtype,
        )
        corr_tO_load = cute.flat_divide(
            tCtO[(None, None), 0, None], (self.mma_tiler_pv[0], corr_ld_inst)
        )
        corr_tiled_copy_t2r = tcgen05.make_tmem_copy(
            corr_copy_atom_t2r, corr_tO_load[None, None, 0, 0, 0]
        )
        corr_thr_copy_t2r = corr_tiled_copy_t2r.get_slice(tidx)
        corr_tCtO_ld = corr_thr_copy_t2r.partition_S(corr_tO_load)

        corr_copy_atom_r2t = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(
                tcgen05.copy.Repetition(corr_ld_inst),
            ),
            self.acc_dtype,
        )
        corr_tiled_copy_r2t = tcgen05.make_tmem_copy(
            corr_copy_atom_r2t, corr_tO_load[None, None, 0, 0, 0]
        )

        cO = cute.make_identity_tensor(self.mma_tiler_pv[:2])
        tCcO = thr_mma_pv.partition_C(cO)
        corr_cO_load = cute.flat_divide(
            tCcO[(None, None), 0, None], (self.mma_tiler_pv[0], corr_ld_inst)
        )
        corr_tCcO_ld = corr_thr_copy_t2r.partition_D(corr_cO_load)
        corr_tCrO_ld = cute.make_fragment(
            cute.select(corr_tCcO_ld.shape, mode=[0, 1, 2]), self.acc_dtype
        )

        # for writeback
        wb_ld_inst: int = 64
        wb_ld_repeat: int = cute.ceil_div(self.mma_tiler_pv[1], wb_ld_inst)

        wb_copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(
                tcgen05.copy.Repetition(wb_ld_inst),
            ),
            self.acc_dtype,
        )
        wb_tO_load = cute.flat_divide(
            tCtO[(None, None), 0, None], (self.mma_tiler_pv[0], wb_ld_inst)
        )
        wb_tiled_copy_t2r = tcgen05.make_tmem_copy(
            wb_copy_atom_t2r, wb_tO_load[(None, None, 0, 0, 0)]
        )
        wb_thr_copy_t2r = wb_tiled_copy_t2r.get_slice(tidx)
        wb_tCtO_ld = wb_thr_copy_t2r.partition_S(wb_tO_load)

        wb_cO_load = cute.flat_divide(
            tCcO[(None, None), 0, None], (self.mma_tiler_pv[0], wb_ld_inst)
        )
        wb_tCcO_ld = wb_thr_copy_t2r.partition_D(wb_cO_load)
        wb_tCrO_ld = cute.make_fragment(
            cute.select(wb_tCcO_ld.shape, mode=[0, 1, 2]), self.acc_dtype
        )
        wb_tCrO_ld_half = cute.make_fragment(wb_tCrO_ld.layout, sO.element_type)
        tv_layout = cute.make_ordered_layout(
            (self.threads_per_wg, self.mma_tiler_pv[1], self.s_stage), (0, 1, 2)
        )
        sO_cpy = cute.composition(sO, tv_layout)
        sO_cpy_slice = cute.flatten(sO_cpy[tidx, None, None])

        wb_tCrO_reduction = None
        if cutlass.const_expr(self.o_stage != 1):
            wb_tCrO_reduction = cute.make_fragment(wb_tCrO_ld.layout, self.acc_dtype)

        _correction_loop = partial(
            self.correction_loop,
            TileSchedulerCls=TileSchedulerCls,
            variable_block_sizes=variable_block_sizes,
            q2k_block_sparse_num=q2k_block_sparse_num,
            p_producer_state=p_producer_state,
            o_pipeline=o_pipeline,
            o_producer_state=o_producer_state,
            correction_consumer_state=correction_consumer_state,
            corr_ld_repeat=corr_ld_repeat,
            corr_tiled_copy_t2r=corr_tiled_copy_t2r,
            corr_tCtO_ld=corr_tCtO_ld,
            corr_tCrO_ld=corr_tCrO_ld,
            corr_tiled_copy_r2t=corr_tiled_copy_r2t,
            pv_mma_pipeline=pv_mma_pipeline,
            pv_mma_consumer_state=pv_mma_consumer_state,
            st_O_pipeline=st_O_pipeline,
            st_O_producer_state=st_O_producer_state,
            tv_layout=tv_layout,
            sO=sO_cpy_slice,
            sScale=sScale,
            wb_tCcO_ld=wb_tCcO_ld,
            wb_tCrO_ld=wb_tCrO_ld,
            sFinal=sFinal,
            sFinal_pipeline=sFinal_pipeline,
            sFinal_consumer_state=sFinal_consumer_state,
            wb_ld_repeat=wb_ld_repeat,
            wb_tCrO_reduction=wb_tCrO_reduction,
            wb_tiled_copy_t2r=wb_tiled_copy_t2r,
            wb_tCtO_ld=wb_tCtO_ld,
            wb_tCrO_ld_half=wb_tCrO_ld_half,
            tidx=tidx,
            sm_scale_log2=sm_scale_log2,
            num_q_blocks=num_q_blocks,
            O_final_guard=O_final_guard,
            gLSE=gLSE,
            in_which=in_which,
        )

        _correction_loop(correction_pipeline=corr_pipeline)

    @cute.jit
    def epilogue(
        self,
        TileSchedulerCls: Callable,
        st_O_pipeline: pipeline.PipelineAsync,
        st_O_consumer_state: pipeline.PipelineState,
        gO: cute.Tensor,
        sO: cute.Tensor,
        tma_atom_O: cute.CopyAtom,
    ):
        def _get_sO_cpy(
            sO: cute.Tensor,
        ) -> cute.Tensor:
            layout = cute.flatten(sO.layout)
            layout = cute.select(layout, mode=[1, 0, 2, 3, 4, 5])
            layout = cute.flat_divide(
                layout,
                (8,),  # it relates to 128B swizzle for 16-bit data
            )
            layout = cute.select(layout, mode=[2, 0, 3, 4, 1, 5, 6])
            layout = cute.group_modes(layout, 0, 2)
            layout = cute.group_modes(layout, 4, cute.rank(layout))
            cpy_tensor = cute.make_tensor(sO.iterator, layout)
            return cpy_tensor

        sO_cpy = _get_sO_cpy(sO)

        def _get_gO_cpy(
            gO: cute.Tensor,
        ) -> cute.Tensor:
            layout = cute.flat_divide(gO.layout, (64, 64))
            layout = cute.group_modes(layout, 0, 2)
            cpy_tensor = cute.make_tensor(gO.iterator, layout)
            return cpy_tensor

        _gO = _get_gO_cpy(gO)
        tTMAsO, tTMAgO = cute.nvgpu.cpasync.tma_partition(
            tma_atom_O,
            0,
            cute.make_layout(1),
            cute.group_modes(sO_cpy, 0, 2),
            cute.group_modes(_gO, 0, 2),
        )

        num_o_blocks = cute.size(gO, mode=[2])
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m2_idx, head_idx, batch_idx = work_tile.tile_idx

            m_block_1st = m2_idx * 2
            m_block_2nd = m_block_1st + 1
            m_block_2nd = m_block_2nd if m_block_2nd < num_o_blocks else num_o_blocks

            tTMAgO_1st = tTMAgO[None, None, m_block_1st, None, head_idx, batch_idx]
            tTMAgO_2nd = tTMAgO[None, None, m_block_2nd, None, head_idx, batch_idx]

            cute.arch.cp_async_bulk_wait_group(self.epi_stage - 1, read=True)
            st_O_pipeline.consumer_release(st_O_consumer_state)
            st_O_pipeline.consumer_wait(st_O_consumer_state)

            cute.copy(
                tma_atom_O, tTMAsO[None, 0, 0, st_O_consumer_state.index], tTMAgO_1st[None, 0, 0]
            )
            cute.copy(
                tma_atom_O, tTMAsO[None, 1, 0, st_O_consumer_state.index], tTMAgO_1st[None, 1, 0]
            )
            cute.copy(
                tma_atom_O, tTMAsO[None, 0, 1, st_O_consumer_state.index], tTMAgO_2nd[None, 0, 0]
            )
            cute.copy(
                tma_atom_O, tTMAsO[None, 1, 1, st_O_consumer_state.index], tTMAgO_2nd[None, 1, 0]
            )

            cute.arch.cp_async_bulk_commit_group()
            st_O_consumer_state.advance()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        cute.arch.cp_async_bulk_wait_group(0, read=True)
