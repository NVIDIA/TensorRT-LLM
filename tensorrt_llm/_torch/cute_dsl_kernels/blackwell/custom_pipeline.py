# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.

from dataclasses import dataclass
from typing import Optional

import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, if_generate
from cutlass.pipeline import (CooperativeGroup, PipelineAsync, PipelineOp,
                              PipelineState)


def pipeline_init_wait(cta_layout_vmnk: Optional[cute.Layout] = None):
    """
    Fences the mbarrier init and syncs the threadblock or cluster
    """
    cute.arch.mbarrier_init_fence()


##############################################################################
# Pipeline classes
##############################################################################


@dataclass(frozen=True)
class PipelineTmaUmma(PipelineAsync):
    """
    PipelineTmaUmma is used for TMA producers and UMMA consumers (e.g. Blackwell mainloops).
    """

    is_leader_cta: bool
    cta_group: cute.nvgpu.tcgen05.CtaGroup

    @staticmethod
    def _compute_mcast_arrival_mask(cta_layout_vmnk: cute.Layout,
                                    mcast_mode_mn: tuple[int, int]):
        """
        Computes a mask for signaling arrivals to multicasting threadblocks.
        """
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster())
        cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster)

        tma_mcast_mask_a = cute.nvgpu.cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2)
        tma_mcast_mask_b = cute.nvgpu.cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=1)

        block_in_cluster_coord_vmnk_peer = (
            cta_in_cluster_coord_vmnk[0] ^ 1,
            *cta_in_cluster_coord_vmnk[1:],
        )
        tma_mcast_mask_a_peer = cute.nvgpu.cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=2)
        tma_mcast_mask_b_peer = cute.nvgpu.cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=1)

        assert not (mcast_mode_mn[0] == 0 and mcast_mode_mn[1] == 0)
        if mcast_mode_mn[0] == 1 and mcast_mode_mn[1] == 1:
            return (tma_mcast_mask_a
                    | tma_mcast_mask_b
                    | tma_mcast_mask_a_peer
                    | tma_mcast_mask_b_peer)
        elif mcast_mode_mn[1] == 1:
            return tma_mcast_mask_b | tma_mcast_mask_b_peer
        assert mcast_mode_mn[0] == 1
        return tma_mcast_mask_a | tma_mcast_mask_a_peer

    @staticmethod
    def _compute_is_leader_cta(cta_layout_vmnk: cute.Layout):
        """
        Computes leader threadblocks for 2CTA kernels. For 1CTA, all threadblocks are leaders.
        """
        bidx, bidy, _ = cute.arch.block_idx()

        mma_coord_vmnk = (
            bidx % cute.size(cta_layout_vmnk, mode=[0]),
            bidx // cute.size(cta_layout_vmnk, mode=[0]),
            bidy,
            None,
        )
        return mma_coord_vmnk[0] == 0

    @staticmethod
    def create(
            *,
            num_stages: int,
            producer_group: CooperativeGroup,
            consumer_group: CooperativeGroup,
            tx_count: int,
            barrier_storage: cute.Pointer = None,
            cta_layout_vmnk: Optional[cute.Layout] = None,
            mcast_mode_mn: tuple[int, int] = (1, 1),
    ):
        """
        This helper function computes any necessary attributes and returns an instance of PipelineTmaUmma.
        :param barrier_storage: Pointer to the smem address for this pipeline's mbarriers
        :type barrier_storage: cute.Pointer
        :param num_stages: Number of buffer stages for this pipeline
        :type num_stages: Int32
        :param producer_group: `CooperativeGroup` for the producer agent
        :type producer_group: CooperativeGroup
        :param consumer_group: `CooperativeGroup` for the consumer agent
        :type consumer_group: CooperativeGroup
        :param tx_count: Number of bytes expected to be written to the transaction barrier for one stage
        :type tx_count: int
        :param cta_layout_vmnk: Layout of the cluster shape
        :type cta_layout_vmnk: cute.Layout | None
        :param mcast_mode_mn: Tuple of two integers, specifying whether mcast is enabled for the m and n modes. At least one of the two integers must be 1.
        :type mcast_mode_mn: tuple[int, int]
        """
        if not isinstance(barrier_storage, cute.Pointer):
            raise ValueError(
                f"Expected barrier_storage to be a cute.Pointer, but got {type(barrier_storage)}"
            )

        producer_type = PipelineOp.TmaLoad
        consumer_type = PipelineOp.TCGen05Mma

        producer = (producer_type, producer_group)
        consumer = (consumer_type, consumer_group)

        sync_object_full = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer, tx_count)
        sync_object_empty = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages,
            consumer)

        if cta_layout_vmnk is None or cute.size(cta_layout_vmnk) == 1:
            # No mcast mask if not using clusters
            producer_mask = None
            # All threadblocks are leaders if not using clusters
            is_leader_cta = True
        else:
            producer_mask = PipelineTmaUmma._compute_mcast_arrival_mask(
                cta_layout_vmnk, mcast_mode_mn)
            is_leader_cta = PipelineTmaUmma._compute_is_leader_cta(
                cta_layout_vmnk)

        cta_group = (cute.nvgpu.tcgen05.CtaGroup.ONE if cta_layout_vmnk is None
                     or cute.size(cta_layout_vmnk, mode=[0]) == 1 else
                     cute.nvgpu.tcgen05.CtaGroup.TWO)

        consumer_mask = producer_mask

        pipeline_init_wait(cta_layout_vmnk)

        return PipelineTmaUmma(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            consumer_mask,
            is_leader_cta,
            cta_group,
        )

    def consumer_release(self, state: PipelineState):
        """
        UMMA consumer release buffer empty, cta_group needs to be provided.
        """
        self.sync_object_empty.arrive(state.index, self.consumer_mask,
                                      self.cta_group)

    def producer_acquire(self,
                         state: PipelineState,
                         try_acquire_token: Optional[Boolean] = None):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase),
        )
        if_generate(
            self.is_leader_cta,
            lambda: self.sync_object_full.arrive(state.index, self.producer_mask
                                                 ),
        )

    def producer_commit(self, state: PipelineState):
        """
        TMA producer commit is a noop since TMA instruction itself updates the transaction count.
        """


@dataclass(frozen=True)
class PipelineUmmaAsync(PipelineAsync):
    """
    PipelineUmmaAsync is used for UMMA producers and AsyncThread consumers (e.g. Blackwell accumulator pipelines).
    """

    cta_group: cute.nvgpu.tcgen05.CtaGroup

    @staticmethod
    def _compute_tmem_sync_mask(cta_layout_vmnk: cute.Layout):
        """
        Computes a mask to signal completion of tmem buffers for 2CTA kernels.
        """
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster())
        cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster)
        return cute.make_layout_image_mask(cta_layout_vmnk,
                                           cta_in_cluster_coord_vmnk,
                                           mode=0)

    @staticmethod
    def _compute_peer_cta_rank():
        """
        Computes a mask to signal release of tmem buffers for 2CTA kernels.
        """
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster())
        return cta_rank_in_cluster // 2 * 2

    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        barrier_storage: cute.Pointer = None,
        cta_layout_vmnk: Optional[cute.Layout] = None,
    ):
        """
        This helper function computes any necessary attributes and returns an instance of PipelineUmmaAsync.
        :param barrier_storage: Pointer to the smem address for this pipeline's mbarriers
        :type barrier_storage: cute.Pointer
        :param num_stages: Number of buffer stages for this pipeline
        :type num_stages: Int32
        :param producer_group: `CooperativeGroup` for the producer agent
        :type producer_group: CooperativeGroup
        :param consumer_group: `CooperativeGroup` for the consumer agent
        :type consumer_group: CooperativeGroup
        :param cta_layout_vmnk: Layout of the cluster shape
        :type cta_layout_vmnk: cute.Layout | None
        """
        if not isinstance(barrier_storage, cute.Pointer):
            raise ValueError(
                f"Expected barrier_storage to be a cute.Pointer, but got {type(barrier_storage)}"
            )

        producer_type = PipelineOp.TCGen05Mma
        consumer_type = PipelineOp.AsyncThread

        producer = (producer_type, producer_group)
        consumer = (consumer_type, consumer_group)

        sync_object_full = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer)
        sync_object_empty = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages,
            consumer)

        if cta_layout_vmnk is None or cute.size(cta_layout_vmnk) == 1:
            # Set mask to None if not using clusters (i.e. 1CTA kernels)
            producer_mask = None
        else:
            producer_mask = PipelineUmmaAsync._compute_tmem_sync_mask(
                cta_layout_vmnk)

        if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, mode=[0]) == 1:
            # Set mask to None if not using 2CTA instructions
            consumer_mask = None
        else:
            consumer_mask = PipelineUmmaAsync._compute_peer_cta_rank()

        cta_group = (cute.nvgpu.tcgen05.CtaGroup.ONE if cta_layout_vmnk is None
                     or cute.size(cta_layout_vmnk, mode=[0]) == 1 else
                     cute.nvgpu.tcgen05.CtaGroup.TWO)

        pipeline_init_wait(cta_layout_vmnk)

        return PipelineUmmaAsync(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            consumer_mask,
            cta_group,
        )

    def producer_commit(self, state: PipelineState):
        """
        UMMA producer commit buffer full, cta_group needs to be provided.
        """
        self.sync_object_full.arrive(state.index, self.producer_mask,
                                     self.cta_group)

    def producer_tail(self, state: PipelineState):
        """
        Make sure the last used buffer empty signal is visible to producer.
        Producer tail is usually executed by producer before exit, to avoid dangling
        mbarrier arrive signals after kernel exit.

        :param state: The pipeline state that points to next useful buffer
        :type state: PipelineState
        """
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster())
        is_leader_cta = cta_rank_in_cluster % 2 == 0

        def then_body():
            # Assume state contains that next useful buffer
            # So we only need to advance to num_stages - 1 times to last used buffer
            for i in range(self.num_stages - 1):
                state.advance()
            self.producer_acquire(state)

        if_generate(is_leader_cta, then_body)
