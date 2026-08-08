# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# nvidia-cutlass-dsl 4.4.2 split the sync-object factory: sm90's
# PipelineAsync._make_sync_object no longer accepts Blackwell ops like
# TCGen05Mma/ClcLoad. The sm100 PipelineTmaUmma provides the expanded variant
# that handles every op (TCGen05Mma) used by PipelineCpAsyncUmma below.
from cutlass.pipeline import (Agent, CooperativeGroup, PipelineAsync,
                              PipelineOp, PipelineState)
from cutlass.pipeline import PipelineTmaUmma as _Sm100PipelineFactory
from cutlass.pipeline import agent_sync

_make_sync_object = _Sm100PipelineFactory._make_sync_object


@dataclass(frozen=True)
class PipelineCpAsyncUmma(PipelineAsync):
    """
    PipelineCpAsyncUmma is used for LDGSTS (CpAsync) producers and UMMA consumers.

    This pipeline is specifically designed for scenarios where:
    - Producers use LDGSTS instructions (cp.async) to load data from global to shared memory
    - Consumers are UMMA warps that perform MMA operations using the loaded data

    Key differences from PipelineAsyncUmma:
    - Suitable for gather/permutation operations during load
    - Used in this kernel for A and SFA matrices with token-based gather addressing
    """

    cta_group: cute.nvgpu.tcgen05.CtaGroup

    @staticmethod
    def _compute_leading_cta_rank(cta_v_size):
        """
        Computes the leading CTA rank.
        """
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster())
        return cta_rank_in_cluster // cta_v_size * cta_v_size

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
    def _compute_peer_cta_mask(cta_layout_vmnk: cute.Layout):
        """
        Computes a mask for signaling arrivals to multicasting threadblocks.
        """
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster())
        cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster)
        mask_self = cute.nvgpu.cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=0)
        block_in_cluster_coord_vmnk_peer = (
            cta_in_cluster_coord_vmnk[0] ^ 1,
            *cta_in_cluster_coord_vmnk[1:],
        )
        mask_peer = cute.nvgpu.cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=0)
        return mask_self | mask_peer

    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        barrier_storage: cute.Pointer = None,
        cta_layout_vmnk: Optional[cute.Layout] = None,
        defer_sync: bool = False,
    ):
        """Creates and initializes a new PipelineCpAsyncUmma instance.

        :param num_stages: Number of buffer stages for this pipeline
        :type num_stages: int
        :param producer_group: CooperativeGroup for the producer agent
        :type producer_group: CooperativeGroup
        :param consumer_group: CooperativeGroup for the consumer agent
        :type consumer_group: CooperativeGroup
        :param barrier_storage: Pointer to the shared memory address for this pipeline's mbarriers
        :type barrier_storage: cute.Pointer, optional
        :param cta_layout_vmnk: Layout of the cluster shape
        :type cta_layout_vmnk: cute.Layout, optional
        :param defer_sync: Whether to defer the sync
        :type defer_sync: bool, optional
        :raises ValueError: If barrier_storage is not a cute.Pointer instance
        :return: A new PipelineCpAsyncUmma instance configured with the provided parameters
        :rtype: PipelineCpAsyncUmma
        """
        if not isinstance(barrier_storage, cute.Pointer):
            raise ValueError(
                f"Expected barrier_storage to be a cute.Pointer, but got {type(barrier_storage)}"
            )

        producer_type = PipelineOp.AsyncLoad
        consumer_type = PipelineOp.TCGen05Mma

        producer = (producer_type, producer_group)
        consumer = (consumer_type, consumer_group)

        sync_object_full = _make_sync_object(
            barrier_storage.align(min_align=8),
            num_stages,
            producer,
        )
        sync_object_empty = _make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages,
            consumer)

        cta_v_size = cute.size(cta_layout_vmnk,
                               mode=[0]) if cta_layout_vmnk is not None else 1
        cta_group = (cute.nvgpu.tcgen05.CtaGroup.ONE if cta_layout_vmnk is None
                     or cute.size(cta_layout_vmnk, mode=[0]) == 1 else
                     cute.nvgpu.tcgen05.CtaGroup.TWO)
        if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, mode=[0]) == 1:
            # No mcast mask if we're not using 2CTA tcgen05 MMA
            producer_mask = None
            consumer_mask = None
        else:
            # If we're using 2CTA UMMAs, producer will arrive the mbar on leading CTA
            # We need to get the target cta_rank
            producer_mask = PipelineCpAsyncUmma._compute_leading_cta_rank(
                cta_v_size)
            # consumer needs to get the mask to signal
            consumer_mask = PipelineCpAsyncUmma._compute_peer_cta_mask(
                cta_layout_vmnk)

        if not defer_sync:
            if cta_layout_vmnk is None or cute.size(cta_layout_vmnk) == 1:
                agent_sync(Agent.ThreadBlock)
            else:
                agent_sync(Agent.ThreadBlockCluster, is_relaxed=True)

        return PipelineCpAsyncUmma(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            consumer_mask,
            cta_group,
        )

    def consumer_release(self, state: PipelineState, *, loc=None, ip=None):
        """
        UMMA consumer release buffer empty, cta_group needs to be provided.
        """
        self.sync_object_empty.arrive(state.index,
                                      self.consumer_mask,
                                      self.cta_group,
                                      loc=loc,
                                      ip=ip)
