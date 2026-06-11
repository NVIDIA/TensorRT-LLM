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

from typing import Mapping, Protocol, TypeAlias


KvCacheIterationStatValue: TypeAlias = int | float
KvCacheIterationStatsDict: TypeAlias = dict[
    str, dict[str, KvCacheIterationStatValue]]


class KvCacheIterationStatsLike(Protocol):
    primary_max_num_blocks: int
    primary_free_num_blocks: int
    primary_used_num_blocks: int
    secondary_max_num_blocks: int
    secondary_free_num_blocks: int
    secondary_used_num_blocks: int
    iter_alloc_total_blocks: int
    iter_alloc_new_blocks: int
    iter_reused_blocks: int
    iter_full_reused_blocks: int
    iter_partial_reused_blocks: int
    iter_missed_blocks: int
    iter_cache_hit_rate: float
    iter_gen_alloc_blocks: int
    iter_onboard_blocks: int
    iter_onboard_bytes: int
    iter_offload_blocks: int
    iter_offload_bytes: int
    iter_intra_device_copy_blocks: int
    iter_intra_device_copy_bytes: int
    iter_transfer_pinned_blocks: int
    iter_transfer_already_primary_blocks: int
    iter_transfer_primary_block_reservations: int
    iter_transfer_onboarded_blocks: int
    iter_transfer_reservation_failures: int
    iter_transfer_lease_release_blocks: int


def kv_cache_iteration_stats_to_dict(
        kv_iter_stats: Mapping[int, KvCacheIterationStatsLike]
) -> KvCacheIterationStatsDict:
    return {
        str(window_size): {
            "primaryMaxNumBlocks": s.primary_max_num_blocks,
            "primaryFreeNumBlocks": s.primary_free_num_blocks,
            "primaryUsedNumBlocks": s.primary_used_num_blocks,
            "secondaryMaxNumBlocks": s.secondary_max_num_blocks,
            "secondaryFreeNumBlocks": s.secondary_free_num_blocks,
            "secondaryUsedNumBlocks": s.secondary_used_num_blocks,
            "iterAllocTotalBlocks": s.iter_alloc_total_blocks,
            "iterAllocNewBlocks": s.iter_alloc_new_blocks,
            "iterReusedBlocks": s.iter_reused_blocks,
            "iterFullReusedBlocks": s.iter_full_reused_blocks,
            "iterPartialReusedBlocks": s.iter_partial_reused_blocks,
            "iterMissedBlocks": s.iter_missed_blocks,
            "iterCacheHitRate": s.iter_cache_hit_rate,
            "iterGenAllocBlocks": s.iter_gen_alloc_blocks,
            "iterOnboardBlocks": s.iter_onboard_blocks,
            "iterOnboardBytes": s.iter_onboard_bytes,
            "iterOffloadBlocks": s.iter_offload_blocks,
            "iterOffloadBytes": s.iter_offload_bytes,
            "iterIntraDeviceCopyBlocks": s.iter_intra_device_copy_blocks,
            "iterIntraDeviceCopyBytes": s.iter_intra_device_copy_bytes,
            "iterTransferPinnedBlocks": s.iter_transfer_pinned_blocks,
            "iterTransferAlreadyPrimaryBlocks":
            s.iter_transfer_already_primary_blocks,
            "iterTransferPrimaryBlockReservations":
            s.iter_transfer_primary_block_reservations,
            "iterTransferOnboardedBlocks": s.iter_transfer_onboarded_blocks,
            "iterTransferReservationFailures":
            s.iter_transfer_reservation_failures,
            "iterTransferLeaseReleaseBlocks":
            s.iter_transfer_lease_release_blocks,
        }
        for window_size, s in kv_iter_stats.items()
    }
