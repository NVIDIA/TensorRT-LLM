# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This file is copied and modified from https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL/cutlass/pipeline

from dataclasses import dataclass
from typing import Optional

import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, if_generate
from cutlass.pipeline import (CooperativeGroup, PipelineAsync, PipelineOp,
                              PipelineState)


def pipeline_init_wait(cta_layout_vmnk: Optional[cute.Layout] = None):
    """Initializes the mbarrier and synchronizes the threadblock or cluster.

    This function places a fence on the mbarrier initialization to ensure
    proper synchronization across the threadblock or cluster.

    Args:
        cta_layout_vmnk (Optional[cute.Layout]): The CTA layout for VMNK. Defaults to None.
    """
    cute.arch.mbarrier_init_fence()


##############################################################################
# Pipeline classes
##############################################################################


@dataclass(frozen=True)
class PipelineTmaUmma(PipelineAsync):
    """PipelineTmaUmma is used for TMA producers and UMMA consumers.

    This class is typically used in scenarios such as Blackwell mainloops, where TMA (Tensor Memory Access) producers interact with UMMA (Universal Matrix Multiply Accumulate) consumers.

    Attributes:
        is_leader_cta (bool): Indicates if the current CTA is the leader.
        cta_group (cute.nvgpu.tcgen05.CtaGroup): The CTA group associated with the pipeline.
    """

    is_leader_cta: bool
    cta_group: cute.nvgpu.tcgen05.CtaGroup

    @staticmethod
    def _compute_mcast_arrival_mask(cta_layout_vmnk: cute.Layout,
                                    mcast_mode_mn: tuple[int, int]):
        """Computes a mask for signaling arrivals to multicasting threadblocks.

        Returns:
            The computed mask for multicasting threadblocks.
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

        Args:
            cta_layout_vmnk (cute.Layout): Layout of the cluster shape.

        Returns:
            bool: True if the current threadblock is a leader, False otherwise.
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

        Args:
            barrier_storage (cute.Pointer): Pointer to the smem address for this pipeline's mbarriers.
            num_stages (int): Number of buffer stages for this pipeline.
            producer_group (CooperativeGroup): `CooperativeGroup` for the producer agent.
            consumer_group (CooperativeGroup): `CooperativeGroup` for the consumer agent.
            tx_count (int): Number of bytes expected to be written to the transaction barrier for one stage.
            cta_layout_vmnk (cute.Layout | None): Layout of the cluster shape.
            mcast_mode_mn (tuple[int, int]): Tuple of two integers, specifying whether mcast is enabled for the m and n modes. At least one of the two integers must be 1.

        Returns:
            PipelineTmaUmma: An instance of PipelineTmaUmma with all necessary attributes computed.
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

        Google style:
        Args:
            state (PipelineState): The current pipeline state.

        Returns:
            None
        """
        self.sync_object_empty.arrive(state.index, self.consumer_mask,
                                      self.cta_group)

    def producer_acquire(self,
                         state: PipelineState,
                         try_acquire_token: Optional[Boolean] = None):
        """
        Conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.

        Google style:
        This method is used by the TMA producer to conditionally wait for the buffer to be empty and, for leader threadblocks, to set the transaction barrier.

        Args:
            state (PipelineState): The current pipeline state.
            try_acquire_token (Optional[Boolean]): Optional token to control conditional acquire.

        Returns:
            None
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

        Google style:
        This method does nothing because the TMA instruction automatically updates the transaction count.

        Args:
            state (PipelineState): The current pipeline state.

        Returns:
            None
        """


@dataclass(frozen=True)
class PipelineUmmaAsync(PipelineAsync):

    cta_group: cute.nvgpu.tcgen05.CtaGroup

    @staticmethod
    def _compute_tmem_sync_mask(cta_layout_vmnk: cute.Layout):
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster())
        cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster)
        return cute.make_layout_image_mask(cta_layout_vmnk,
                                           cta_in_cluster_coord_vmnk,
                                           mode=0)

    @staticmethod
    def _compute_peer_cta_rank():
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

        Args:
            barrier_storage (cute.Pointer): Pointer to the smem address for this pipeline's mbarriers.
            num_stages (int): Number of buffer stages for this pipeline.
            producer_group (CooperativeGroup): CooperativeGroup for the producer agent.
            consumer_group (CooperativeGroup): CooperativeGroup for the consumer agent.
            cta_layout_vmnk (cute.Layout or None): Layout of the cluster shape.

        Returns:
            PipelineUmmaAsync: An instance of PipelineUmmaAsync.
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
        self.sync_object_full.arrive(state.index, self.producer_mask,
                                     self.cta_group)

    def producer_tail(self, state: PipelineState):
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
