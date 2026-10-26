# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file is copied and modified from cutlass example https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py
# and the CUTLASS pipeline implementation https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL/cutlass/pipeline

"""Engram's KV projection read directly from packed MXFP8 activation rows."""

from dataclasses import dataclass
from typing import Optional, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import if_generate as _if_generate
from cutlass.pipeline import Agent, CooperativeGroup, PipelineOp, PipelineState, agent_sync

from ..custom_pipeline import PipelineTmaUmma, PipelineUmmaAsync
from ..dense_blockscaled_gemm_persistent import Sm100BlockScaledPersistentDenseGemmKernel
from ..utils import TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait


@dataclass(frozen=True)
class _PackedProjectionPipeline(PipelineTmaUmma):
    """A/B stage barriers of a two-CTA cluster: a stage is full once the leader has its TMA bytes
    and every producer lane of both CTAs has arrived after copying A's scale codes."""

    tx_count: int

    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: int,
        barrier_storage: cute.Pointer,
        cta_layout_vmnk: cute.Layout,
    ) -> "_PackedProjectionPipeline":
        producer = (PipelineOp.AsyncLoad, producer_group)
        consumer = (PipelineOp.TCGen05Mma, consumer_group)
        sync_object_full = pipeline.PipelineTmaUmma._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer, tx_count
        )
        sync_object_empty = pipeline.PipelineTmaUmma._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages, consumer
        )
        # The leader's cluster rank: the V bit cleared.
        cta_rank_here = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        producer_mask = cta_rank_here & ~cutlass.Int32(1)
        consumer_mask = PipelineTmaUmma._compute_mcast_arrival_mask(cta_layout_vmnk, (1, 1))
        is_leader_cta = PipelineTmaUmma._compute_is_leader_cta(cta_layout_vmnk)
        cute.arch.mbarrier_init_fence()
        agent_sync(Agent.ThreadBlockCluster, is_relaxed=True)
        return _PackedProjectionPipeline(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            consumer_mask,
            is_leader_cta,
            cute.nvgpu.tcgen05.CtaGroup.TWO,
            tx_count,
        )

    def producer_acquire_tma(self, state: PipelineState, try_acquire_token: cutlass.Boolean):
        """Wait for an empty stage. Only the leader expects the stage's multicast TMA bytes."""
        _if_generate(
            try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase),
        )

        def _leader_arrive_and_expect_tx() -> None:
            self.sync_object_full.arrive_and_expect_tx(state.index, self.tx_count)

        _if_generate(
            self.is_leader_cta,
            _leader_arrive_and_expect_tx,
        )

    def producer_arrive_remote(self, stage_index: cutlass.Int32) -> None:
        bar = self.sync_object_full.get_barrier(stage_index)
        cute.arch.mbarrier_arrive(bar, self.producer_mask)


class EngramPackedProjectionKernel(Sm100BlockScaledPersistentDenseGemmKernel):
    """MXFP8 GEMM ``C[M, N] = A[M, K] * B[N, K]^T`` with A read from ``num_sources`` blocks of
    packed rows (E4M3 bytes, then their UE8M0 codes); B's scales use the native swizzle."""

    def __init__(self) -> None:
        super().__init__(32, (256, 256), (2, 1), swizzle_size=8, raster_along_m=False)
        # Leave shared memory for concurrent model kernels.
        self.smem_capacity = min(self.smem_capacity, 192 * 1024)

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)
        self.q_fp8_fusion = False
        self._setup_attributes()

        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, self.sf_vec_size)
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mnk[:2],
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mnk_sfb[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # The producer warp copies A's scales without TMA.
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size + sfb_copy_size) * atom_thr_size

        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor,
            epi_smem_layout,
            self.epi_tile,
        )

        self.tile_sched_params, grid = self._compute_grid(
            c_tensor,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            self.swizzle_size,
            self.raster_along_m,
        )

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            sfa_tensor,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            alpha,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        alpha: cute.Tensor,
    ):
        alpha_value = alpha[0].to(self.c_dtype)

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = storage.tmem_holding_buf

        ab_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 1 + 32 * cute.size(tiled_mma.thr_id.shape)
        )
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = _PackedProjectionPipeline.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len(self.epilog_warp_id) * 2
        )
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        if warp_idx == self.tma_warp_id:
            num_tmem_dealloc_threads = 32
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads)
        cute.arch.mbarrier_init_fence()

        cute.arch.cluster_arrive_relaxed(aligned=True)

        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        a_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )
        sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
        )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (M, K / 128, sources): each 32-bit word holds a row's four scale codes of one K tile
        gScale4 = cute.make_tensor(
            cute.recast_ptr(mSFA_mkl.iterator, dtype=cutlass.Uint32),
            cute.make_layout(
                (mSFA_mkl.shape[0], mSFA_mkl.shape[1] // 4, mSFA_mkl.shape[2]),
                stride=(mSFA_mkl.stride[0] // 4, 1, mSFA_mkl.stride[2] // 4),
            ),
        )
        # (CTA rows, STAGE) scale words
        sScale4 = cute.make_tensor(
            cute.recast_ptr(sSFA.iterator, dtype=cutlass.Uint32),
            cute.make_layout((128, self.num_ab_stage), stride=(1, 128)),
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        source_k_tiles = cutlass.Int32(cute.size(gA_mkl, mode=[3]))
        k_block_cnt = cutlass.Int32(cute.size(gB_nkl, mode=[3]))

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

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

        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # One accumulator stage: the epilogue releases it early and the next tile's MMA overlaps.
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 2))
        tCtAcc_fake = cute.make_tensor(
            tCtAcc_fake.iterator,
            cute.make_layout(
                tCtAcc_fake.shape,
                stride=(
                    tCtAcc_fake.stride[0],
                    tCtAcc_fake.stride[1],
                    tCtAcc_fake.stride[2],
                    (256 - self.num_sf_tmem_cols) * tCtAcc_fake.stride[0][1],
                ),
            ),
        )

        cute.arch.cluster_wait()

        griddepcontrol_wait()

        if warp_idx == self.tma_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
                tBgSFB_slice = tBgSFB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_block_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                for k_block in cutlass.range(0, k_block_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire_tma(ab_producer_state, peek_ab_empty_status)

                    source_rank = ab_producer_state.count // source_k_tiles
                    local_k_tile = ab_producer_state.count % source_k_tiles
                    # A uses a local-K descriptor with the source rank in its batch mode.
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, mma_tile_coord_mnl[0], local_k_tile, source_rank)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_producer_state.count)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # Each lane copies four rows' scale words of this K tile. A cp.async arrive
                    # would reach only this CTA's barrier, so the lane waits for its copies and
                    # arrives on the leader's barrier, one stage late to overlap the next stage.
                    scale_atom = cute.make_copy_atom(
                        cpasync.CopyG2SOp(), self.sf_dtype, num_bits_per_copy=32
                    )
                    scale_pred = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Boolean)
                    lane = tidx % 32
                    row_base = mma_tile_coord_mnl[0] * self.mma_tiler[0] + mma_tile_coord_v * 128
                    for row_group in cutlass.range_constexpr(4):
                        row = row_base + lane + row_group * 32
                        scale_pred[0] = row < mSFA_mkl.shape[0]
                        safe_row = min(row, mSFA_mkl.shape[0] - 1)
                        src_ptr = gScale4.iterator + gScale4.layout(
                            (safe_row, local_k_tile, source_rank)
                        )
                        dst_ptr = sScale4.iterator + sScale4.layout(
                            (lane * 4 + row_group, ab_producer_state.index)
                        )
                        src4 = cute.make_tensor(
                            cute.recast_ptr(src_ptr, dtype=self.sf_dtype), cute.make_layout((4,))
                        )
                        dst4 = cute.make_tensor(
                            cute.recast_ptr(dst_ptr, dtype=self.sf_dtype), cute.make_layout((4,))
                        )
                        cute.copy_atom_call(scale_atom, src4, dst4, pred=scale_pred)
                    cute.arch.cp_async_commit_group()
                    if k_block > 0:
                        cute.arch.cp_async_wait_group(1)
                        cute.arch.sync_warp()
                        cute.arch.fence_proxy("async.shared", space="cta")
                        cute.arch.sync_warp()
                        previous_stage = (
                            ab_producer_state.index + self.num_ab_stage - 1
                        ) % self.num_ab_stage
                        ab_pipeline.producer_arrive_remote(previous_stage)

                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_block_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_warp()
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_warp()
                last_stage = (ab_producer_state.index + self.num_ab_stage - 1) % self.num_ab_stage
                ab_pipeline.producer_arrive_remote(last_stage)

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            ab_pipeline.producer_tail(ab_producer_state)

        if warp_idx == self.mma_warp_id:
            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            )
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
            )

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.phase ^ 1)]

                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_block_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_block in range(k_block_cnt):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        num_kphases = cute.size(tCrA, mode=[2])
                        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                            kphase_coord = (
                                None,
                                None,
                                kphase_idx,
                                ab_consumer_state.index,
                            )

                            sf_kphase_coord = (None, None, kphase_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kphase_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB[sf_kphase_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kphase_coord],
                                tCrB[kphase_coord],
                                tCtAcc,
                            )

                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        ab_pipeline.consumer_release(ab_consumer_state)

                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_block_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            acc_pipeline.producer_tail(acc_producer_state)
        if warp_idx < self.mma_warp_id:
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(self.num_tmem_alloc_cols, tmem_holding_buf, is_two_cta=True)

            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, True
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            tma_atom_c, bSG_sC, bSG_gC_partitioned = self.epilog_gmem_copy_and_partition(
                epi_tidx, tma_atom_c, tCgC, epi_tile, sC
            )

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        *mma_tile_coord_mnl,
                    )
                ]

                acc_stage_index = acc_consumer_state.phase
                reverse_subtile = (
                    cutlass.Boolean(True) if acc_stage_index == 0 else cutlass.Boolean(False)
                )

                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt

                for subtile_idx in cutlass.range(subtile_cnt):
                    real_subtile_idx = subtile_idx
                    if reverse_subtile:
                        real_subtile_idx = (
                            self.cta_tile_shape_mnk[1] // self.epi_tile_n - 1 - subtile_idx
                        )
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    if subtile_idx == self.iter_acc_early_release_in_epilogue:
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    acc_vec = alpha_value * acc_vec.to(self.c_dtype)
                    tRS_rC.store(acc_vec)

                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    epilog_threads = 32 * len(self.epilog_warp_id)
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC[(None, real_subtile_idx)],
                        )
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=True)
            epilog_threads = 32 * len(self.epilog_warp_id)
            cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads)
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1)
                cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=True)
            c_pipeline.producer_tail()

        griddepcontrol_launch_dependents()

    @cute.jit
    def wrapper(
        self,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        source_stride: cutlass.Int64,
        row_stride: cutlass.Int64,
        local_k: cutlass.Constexpr,
        num_sources: cutlass.Constexpr,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        current_stream: cuda.CUstream,
    ):
        """Run the GEMM on packed rows of ``num_sources`` blocks of ``local_k`` features."""
        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout((m, local_k, num_sources), stride=(row_stride, 1, source_stride)),
        )
        b_tensor = cute.make_tensor(b_ptr, cute.make_ordered_layout((n, k, 1), order=(1, 0, 2)))
        c_tensor = cute.make_tensor(c_ptr, cute.make_ordered_layout((m, n, 1), order=(1, 0, 2)))
        sfa_tensor = cute.make_tensor(
            a_sf_ptr,
            cute.make_layout(
                (m, local_k // 32, num_sources), stride=(row_stride, 1, source_stride)
            ),
        )
        sfb_tensor = cute.make_tensor(b_sf_ptr, cute.make_layout((1,), stride=(1,)))
        self(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            alpha_tensor,
            max_active_clusters,
            current_stream,
        )


_compiled: dict = {}


def engram_packed_projection(
    packed: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Project 16-byte aligned packed rows ``[num_sources, M, local_k * 33 / 32]`` with an MXFP8
    weight (native swizzled scales) to BF16 ``[M, N]`` on the current stream."""
    num_sources, m, _ = packed.shape
    n, k = weight.shape
    local_k = k // num_sources
    output = torch.empty((m, n), dtype=torch.bfloat16, device=packed.device)
    if m == 0:
        return output
    sm_count = torch.cuda.get_device_properties(packed.device).multi_processor_count
    key = (packed.device.index, sm_count, local_k, num_sources)
    with torch.cuda.device(packed.device):
        if key not in _compiled:
            kernel = EngramPackedProjectionKernel()
            pointers = [
                make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=16)
                for address, dtype in (
                    (packed.data_ptr(), cutlass.Float8E4M3FN),
                    (weight.data_ptr(), cutlass.Float8E4M3FN),
                    (packed.data_ptr() + local_k, cutlass.Float8E8M0FNU),
                    (weight_scale.data_ptr(), cutlass.Float8E8M0FNU),
                    (output.data_ptr(), cutlass.BFloat16),
                )
            ]
            _compiled[key] = cute.compile(
                kernel.wrapper,
                m,
                n,
                k,
                packed.stride(0),
                packed.stride(1),
                local_k,
                num_sources,
                *pointers,
                cute.runtime.from_dlpack(alpha),
                sm_count // kernel.cluster_shape_mn[0],
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--opt-level 2 --enable-tvm-ffi",
            )
        _compiled[key](
            m,
            n,
            k,
            packed.stride(0),
            packed.stride(1),
            packed.data_ptr(),
            weight.data_ptr(),
            packed.data_ptr() + local_k,
            weight_scale.data_ptr(),
            output.data_ptr(),
            alpha,
        )
    return output
