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

from dataclasses import dataclass, field
from typing import Any, Literal

KV_CACHE_ITERATION_STATS_REUSE_KEYS = (
    "iterReusedBlocks",
    "iterFullReusedBlocks",
    "iterPartialReusedBlocks",
    "iterMissedBlocks",
    "iterCacheHitRate",
)

KV_CACHE_ITERATION_STATS_POOL_GROUP_KEYS = (
    "primaryMaxNumBlocks",
    "primaryFreeNumBlocks",
    "primaryUsedNumBlocks",
    "primaryEvictableNumBlocks",
    "primaryPeakFreeNumBlocks",
    "primaryPeakUsedNumBlocks",
    "primaryPeakEvictableNumBlocks",
    "secondaryMaxNumBlocks",
    "secondaryFreeNumBlocks",
    "secondaryUsedNumBlocks",
    "secondaryEvictableNumBlocks",
    "secondaryPeakFreeNumBlocks",
    "secondaryPeakUsedNumBlocks",
    "secondaryPeakEvictableNumBlocks",
    "iterAllocTotalBlocks",
    "iterAllocNewBlocks",
    "iterGenAllocBlocks",
    "iterOnboardBlocks",
    "iterOnboardBytes",
    "iterOffloadBlocks",
    "iterOffloadBytes",
    "iterIntraDeviceCopyBlocks",
    "iterIntraDeviceCopyBytes",
    "iterHostDroppedBlocks",
    "iterHostDroppedBytes",
)


@dataclass(slots=True)
class KVCacheV2PoolGroupIterationStats:
    pool_group_id: int
    slot_size: tuple[int, ...]
    window_sizes: tuple[int, ...]
    stats: Any


@dataclass(slots=True)
class KVCacheV2LifeCycleIterationStats:
    life_cycle_id: int
    pool_group_id: int
    window_size: int | None
    kind: str
    stats: Any


@dataclass(slots=True)
class KVCacheV2SsmSnapshotIterationStats:
    iter_snapshot_lookups: int
    iter_snapshot_hits: int
    iter_snapshot_misses: int
    iter_reused_tokens: int
    iter_unreused_tokens: int
    iter_aligned_snapshot_hits: int
    iter_unaligned_snapshot_hits: int

    @property
    def iter_snapshot_hit_rate(self) -> float:
        if self.iter_snapshot_hits == 0 or self.iter_snapshot_lookups == 0:
            return 0.0
        return self.iter_snapshot_hits / self.iter_snapshot_lookups


@dataclass(slots=True)
class KVCacheV2SsmLifeCycleIterationStats:
    life_cycle_id: int
    pool_group_id: int
    snapshot_stats: KVCacheV2SsmSnapshotIterationStats
    window_size: None = field(default=None, init=False)
    kind: Literal["ssm"] = field(default="ssm", init=False)


@dataclass(slots=True)
class KVCacheV2IterationStatsReport:
    by_window_size: dict[int, Any]
    by_pool_group: dict[int, KVCacheV2PoolGroupIterationStats]
    by_life_cycle: dict[
        int, KVCacheV2LifeCycleIterationStats | KVCacheV2SsmLifeCycleIterationStats
    ] = field(default_factory=dict)


def serialize_kv_cache_iteration_stats(stats, keys: tuple[str, ...] | None = None) -> dict:
    fields = {
        "primaryMaxNumBlocks": stats.primary_max_num_blocks,
        "primaryFreeNumBlocks": stats.primary_free_num_blocks,
        "primaryUsedNumBlocks": stats.primary_used_num_blocks,
        "primaryEvictableNumBlocks": stats.primary_evictable_num_blocks,
        "primaryPeakFreeNumBlocks": stats.primary_peak_free_num_blocks,
        "primaryPeakUsedNumBlocks": stats.primary_peak_used_num_blocks,
        "primaryPeakEvictableNumBlocks": stats.primary_peak_evictable_num_blocks,
        "secondaryMaxNumBlocks": stats.secondary_max_num_blocks,
        "secondaryFreeNumBlocks": stats.secondary_free_num_blocks,
        "secondaryUsedNumBlocks": stats.secondary_used_num_blocks,
        "secondaryEvictableNumBlocks": stats.secondary_evictable_num_blocks,
        "secondaryPeakFreeNumBlocks": stats.secondary_peak_free_num_blocks,
        "secondaryPeakUsedNumBlocks": stats.secondary_peak_used_num_blocks,
        "secondaryPeakEvictableNumBlocks": stats.secondary_peak_evictable_num_blocks,
        "iterAllocTotalBlocks": stats.iter_alloc_total_blocks,
        "iterAllocNewBlocks": stats.iter_alloc_new_blocks,
        "iterReusedBlocks": stats.iter_reused_blocks,
        "iterFullReusedBlocks": stats.iter_full_reused_blocks,
        "iterPartialReusedBlocks": stats.iter_partial_reused_blocks,
        "iterMissedBlocks": stats.iter_missed_blocks,
        "iterCacheHitRate": stats.iter_cache_hit_rate,
        "iterGenAllocBlocks": stats.iter_gen_alloc_blocks,
        "iterOnboardBlocks": stats.iter_onboard_blocks,
        "iterOnboardBytes": stats.iter_onboard_bytes,
        "iterOffloadBlocks": stats.iter_offload_blocks,
        "iterOffloadBytes": stats.iter_offload_bytes,
        "iterIntraDeviceCopyBlocks": stats.iter_intra_device_copy_blocks,
        "iterIntraDeviceCopyBytes": stats.iter_intra_device_copy_bytes,
        "iterHostDroppedBlocks": stats.iter_host_dropped_blocks,
        "iterHostDroppedBytes": stats.iter_host_dropped_bytes,
    }
    if keys is None:
        return fields
    return {key: fields[key] for key in keys}


def serialize_ssm_snapshot_iteration_stats(
    stats: KVCacheV2SsmSnapshotIterationStats,
) -> dict:
    return {
        "iterSnapshotLookups": stats.iter_snapshot_lookups,
        "iterSnapshotHits": stats.iter_snapshot_hits,
        "iterSnapshotMisses": stats.iter_snapshot_misses,
        "iterSnapshotHitRate": stats.iter_snapshot_hit_rate,
        "iterReusedTokens": stats.iter_reused_tokens,
        "iterUnreusedTokens": stats.iter_unreused_tokens,
        "iterAlignedSnapshotHits": stats.iter_aligned_snapshot_hits,
        "iterUnalignedSnapshotHits": stats.iter_unaligned_snapshot_hits,
    }


def append_kv_cache_iteration_stats(stats_dict: dict, kv_iter_stats) -> None:
    if kv_iter_stats is None:
        return
    if isinstance(kv_iter_stats, KVCacheV2IterationStatsReport):
        by_window_size = kv_iter_stats.by_window_size
        by_pool_group = kv_iter_stats.by_pool_group
    else:
        by_window_size = kv_iter_stats
        by_pool_group = None

    stats_dict["kvCacheIterationStats"] = {
        str(window_size): serialize_kv_cache_iteration_stats(stats)
        for window_size, stats in by_window_size.items()
    }
    if by_pool_group is None:
        return

    stats_dict["kvCacheIterationStatsByPoolGroup"] = {
        str(pool_group_id): {
            "poolGroupId": stats.pool_group_id,
            "slotSize": list(stats.slot_size),
            "windowSizes": list(stats.window_sizes),
            **serialize_kv_cache_iteration_stats(
                stats.stats, KV_CACHE_ITERATION_STATS_POOL_GROUP_KEYS
            ),
        }
        for pool_group_id, stats in by_pool_group.items()
    }

    if not kv_iter_stats.by_life_cycle:
        return

    stats_by_life_cycle = {}
    for life_cycle_id, stats in kv_iter_stats.by_life_cycle.items():
        serialized = {
            "lifeCycleId": stats.life_cycle_id,
            "poolGroupId": stats.pool_group_id,
            "windowSize": stats.window_size,
            "kind": stats.kind,
        }
        if isinstance(stats, KVCacheV2SsmLifeCycleIterationStats):
            serialized["snapshotStats"] = serialize_ssm_snapshot_iteration_stats(
                stats.snapshot_stats
            )
        else:
            serialized.update(
                serialize_kv_cache_iteration_stats(stats.stats, KV_CACHE_ITERATION_STATS_REUSE_KEYS)
            )
        stats_by_life_cycle[str(life_cycle_id)] = serialized
    stats_dict["kvCacheIterationStatsByLifecycle"] = stats_by_life_cycle
