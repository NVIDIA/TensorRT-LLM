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

import struct
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

KV_CACHE_ITERATION_STATS_REUSE_KEYS = (
    "iterReusedBlocks",
    "iterFullReusedBlocks",
    "iterPartialReusedBlocks",
    "iterMissedBlocks",
    "iterCacheHitRate",
)

# Hot pool groups hold no cold blocks, and cold levels group lifecycles independently of the hot
# level, so a hot pool-group id cannot index a cold level at all. Cold blocks are reported only by
# kvCacheIterationStatsByColdPoolGroup, in that view's own numbering.
KV_CACHE_ITERATION_STATS_POOL_GROUP_KEYS = (
    "primaryMaxNumBlocks",
    "primaryFreeNumBlocks",
    "primaryUsedNumBlocks",
    "primaryEvictableNumBlocks",
    "primaryPeakFreeNumBlocks",
    "primaryPeakUsedNumBlocks",
    "primaryPeakEvictableNumBlocks",
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

# Subset of KV_CACHE_ITERATION_STATS_POOL_GROUP_KEYS reported per cold pool group. The primary_* keys
# are omitted because a cold group holds no GPU blocks, and the iter* delta keys because they are not
# tracked per cold group -- emitting them would report an untracked quantity as a measured zero.
KV_CACHE_ITERATION_STATS_COLD_POOL_GROUP_KEYS = (
    "secondaryMaxNumBlocks",
    "secondaryFreeNumBlocks",
    "secondaryUsedNumBlocks",
    "secondaryEvictableNumBlocks",
    "secondaryPeakFreeNumBlocks",
    "secondaryPeakUsedNumBlocks",
    "secondaryPeakEvictableNumBlocks",
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
    # Reuse block counts split by the cache level the reused pages were resident on. Indexed by
    # CacheLevel, so entry i is the i-th configured tier -- a deployment with a hot and a cold GPU
    # level gets two distinct entries rather than one merged "gpu" bucket. Empty when the life
    # cycle recorded no reuse this iteration.
    full_reused_blocks_by_level: list[int] = field(default_factory=list)
    partial_reused_blocks_by_level: list[int] = field(default_factory=list)


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
    # Keyed by *cold* pool-group id, which is unrelated to the hot ids used by by_pool_group.
    by_cold_pool_group: dict[int, KVCacheV2PoolGroupIterationStats] = field(default_factory=dict)
    # Preemption counters for this iteration. resumed_requests counts recoveries
    # only -- a request's initial admission also drives a SUSPENDED->ACTIVE
    # transition internally, but it is not a preemption and is excluded.
    suspended_requests: int = 0
    resumed_requests: int = 0
    # Blocks the disk-prefetch mechanism actually migrated from disk to host during this
    # iteration. Counts prefetch movement only, not reuse hits served from disk.
    disk_prefetch_blocks: int = 0
    # Initial current-residency cached-token attribution for requests admitted during this
    # iteration, indexed by cache level so entry i is the i-th configured tier.
    cached_tokens_by_level: list[int] = field(default_factory=list)
    # Readable tier name per cache level, in level order. Lets consumers label the level-indexed
    # counters above without hard-coding a gpu/host/disk split.
    cache_level_tiers: list[str] = field(default_factory=list)


@dataclass(slots=True)
class KVCacheIterationStatsValues:
    primary_max_num_blocks: int = 0
    primary_free_num_blocks: int = 0
    primary_used_num_blocks: int = 0
    primary_evictable_num_blocks: int = 0
    primary_peak_free_num_blocks: int = 0
    primary_peak_used_num_blocks: int = 0
    primary_peak_evictable_num_blocks: int = 0
    secondary_max_num_blocks: int = 0
    secondary_free_num_blocks: int = 0
    secondary_used_num_blocks: int = 0
    secondary_evictable_num_blocks: int = 0
    secondary_peak_free_num_blocks: int = 0
    secondary_peak_used_num_blocks: int = 0
    secondary_peak_evictable_num_blocks: int = 0
    iter_alloc_total_blocks: int = 0
    iter_alloc_new_blocks: int = 0
    iter_reused_blocks: int = 0
    iter_full_reused_blocks: int = 0
    iter_partial_reused_blocks: int = 0
    iter_missed_blocks: int = 0
    iter_gen_alloc_blocks: int = 0
    iter_onboard_blocks: int = 0
    iter_onboard_bytes: int = 0
    iter_offload_blocks: int = 0
    iter_offload_bytes: int = 0
    iter_intra_device_copy_blocks: int = 0
    iter_intra_device_copy_bytes: int = 0
    iter_host_dropped_blocks: int = 0
    iter_host_dropped_bytes: int = 0

    @property
    def iter_cache_hit_rate(self) -> float:
        total = self.iter_reused_blocks + self.iter_missed_blocks
        rate = self.iter_reused_blocks / total if total else 0.0
        # The iteration-statistics API represents this field as a 32-bit float.
        return struct.unpack("f", struct.pack("f", rate))[0]


@dataclass(frozen=True, slots=True)
class KVCachePoolStatsSnapshot:
    total: int
    available: int
    evictable: int
    peak_available: int
    peak_unavailable: int
    peak_evictable: int
    slot_sizes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class KVCacheStatsMetadata:
    """Physical layout shared by captures for one manager's lifetime."""

    life_cycles: tuple[tuple[int, int, int | None, str], ...]
    cold_pool_groups: tuple[tuple[int, tuple[int, ...]], ...]
    cache_level_tiers: tuple[str, ...]


_REUSE_DELTA_FIELDS = frozenset(
    (
        "iter_reused_blocks",
        "iter_full_reused_blocks",
        "iter_partial_reused_blocks",
        "iter_missed_blocks",
    )
)


@dataclass(frozen=True, slots=True)
class KVCacheV2IterationStatsSnapshot:
    """Owned CPU values drained once, safe to materialize after the manager advances."""

    metadata: KVCacheStatsMetadata
    pools_by_level: tuple[tuple[KVCachePoolStatsSnapshot, ...], ...]
    deltas: tuple[tuple[int, tuple[tuple[str, int], ...]], ...]
    ssm_deltas: tuple[tuple[int, tuple[int, int, int, int, int, int, int]], ...]
    reused_blocks_by_level: tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]
    suspended_requests: int
    resumed_requests: int
    disk_prefetch_blocks: int
    cached_tokens_by_level: tuple[int, ...]

    def _build_stats(
        self,
        hot_pool_ids: tuple[int, ...] = (),
        cold_pool_ids: tuple[int, ...] = (),
        delta: dict[str, int] | None = None,
    ) -> KVCacheIterationStatsValues:
        hot = [self.pools_by_level[0][pool_id] for pool_id in hot_pool_ids]
        cold = [level[pool_id] for level in self.pools_by_level[1:] for pool_id in cold_pool_ids]
        return KVCacheIterationStatsValues(
            primary_max_num_blocks=sum(pool.total for pool in hot),
            primary_free_num_blocks=sum(pool.available for pool in hot),
            primary_used_num_blocks=sum(pool.total - pool.available for pool in hot),
            primary_evictable_num_blocks=sum(pool.evictable for pool in hot),
            primary_peak_free_num_blocks=sum(pool.peak_available for pool in hot),
            primary_peak_used_num_blocks=sum(pool.peak_unavailable for pool in hot),
            primary_peak_evictable_num_blocks=sum(pool.peak_evictable for pool in hot),
            secondary_max_num_blocks=sum(pool.total for pool in cold),
            secondary_free_num_blocks=sum(pool.available for pool in cold),
            secondary_used_num_blocks=sum(pool.total - pool.available for pool in cold),
            secondary_evictable_num_blocks=sum(pool.evictable for pool in cold),
            secondary_peak_free_num_blocks=sum(pool.peak_available for pool in cold),
            secondary_peak_used_num_blocks=sum(pool.peak_unavailable for pool in cold),
            secondary_peak_evictable_num_blocks=sum(pool.peak_evictable for pool in cold),
            **(delta or {}),
        )

    def build_report(self) -> KVCacheV2IterationStatsReport:
        """Expand window, physical-pool and lifecycle views without accessing engine state."""
        life_cycles = {
            life_cycle_id: (pool_id, window, kind)
            for life_cycle_id, pool_id, window, kind in self.metadata.life_cycles
        }
        pools_by_window: dict[int, set[int]] = defaultdict(set)
        windows_by_pool: dict[int, set[int]] = defaultdict(set)
        for pool_id, window, _ in life_cycles.values():
            if window is not None:
                pools_by_window[window].add(pool_id)
                windows_by_pool[pool_id].add(window)

        deltas_by_pool: dict[int, dict[str, int]] = {}
        deltas_by_window: dict[int, dict[str, int]] = {}
        reuse_by_life_cycle: dict[int, dict[str, int]] = {}
        for life_cycle_id, values in self.deltas:
            pool_id, window, _ = life_cycles[life_cycle_id]
            reuse_delta = {name: value for name, value in values if name in _REUSE_DELTA_FIELDS}
            pool_delta = {name: value for name, value in values if name not in _REUSE_DELTA_FIELDS}
            if any(pool_delta.values()):
                _add_delta(deltas_by_pool, pool_id, pool_delta)
                if window is not None:
                    _add_delta(deltas_by_window, window, pool_delta)
            if any(reuse_delta.values()):
                reuse_by_life_cycle[life_cycle_id] = reuse_delta
                if window is not None:
                    _add_delta(deltas_by_window, window, reuse_delta)

        by_window = {
            window: self._build_stats(
                tuple(
                    pool_id
                    for pool_id in pools_by_window.get(window, ())
                    if windows_by_pool[pool_id] == {window}
                ),
                delta=deltas_by_window.get(window),
            )
            for window in sorted(set(pools_by_window) | set(deltas_by_window))
        }
        pool_ids = (
            set(range(len(self.pools_by_level[0]))) | set(windows_by_pool) | set(deltas_by_pool)
        )
        by_pool = {
            pool_id: KVCacheV2PoolGroupIterationStats(
                pool_id,
                self.pools_by_level[0][pool_id].slot_sizes,
                tuple(sorted(windows_by_pool.get(pool_id, ()))),
                self._build_stats((pool_id,), delta=deltas_by_pool.get(pool_id)),
            )
            for pool_id in sorted(pool_ids)
        }

        reused_by_level = {
            life_cycle_id: (full, partial)
            for life_cycle_id, full, partial in self.reused_blocks_by_level
        }
        by_life_cycle: dict[
            int, KVCacheV2LifeCycleIterationStats | KVCacheV2SsmLifeCycleIterationStats
        ] = {}
        for life_cycle_id, delta in sorted(reuse_by_life_cycle.items()):
            pool_id, window, kind = life_cycles[life_cycle_id]
            assert kind == "attention"
            full, partial = reused_by_level.get(life_cycle_id, ((), ()))
            by_life_cycle[life_cycle_id] = KVCacheV2LifeCycleIterationStats(
                life_cycle_id,
                pool_id,
                window,
                kind,
                self._build_stats(delta=delta),
                list(full),
                list(partial),
            )
        for life_cycle_id, values in self.ssm_deltas:
            pool_id, window, kind = life_cycles[life_cycle_id]
            assert kind == "ssm" and window is None
            assert life_cycle_id not in by_life_cycle
            by_life_cycle[life_cycle_id] = KVCacheV2SsmLifeCycleIterationStats(
                life_cycle_id, pool_id, KVCacheV2SsmSnapshotIterationStats(*values)
            )

        by_cold_pool: dict[int, KVCacheV2PoolGroupIterationStats] = {}
        if len(self.pools_by_level) > 1 and self.metadata.cold_pool_groups:
            assert all(
                len(level) == len(self.metadata.cold_pool_groups)
                for level in self.pools_by_level[1:]
            )
            for pool_id, members in self.metadata.cold_pool_groups:
                windows = {
                    life_cycles[life_cycle_id][1]
                    for life_cycle_id in members
                    if life_cycles.get(life_cycle_id, (None, None, None))[1] is not None
                }
                by_cold_pool[pool_id] = KVCacheV2PoolGroupIterationStats(
                    pool_id,
                    self.pools_by_level[1][pool_id].slot_sizes,
                    tuple(sorted(windows)),
                    self._build_stats(cold_pool_ids=(pool_id,)),
                )

        return KVCacheV2IterationStatsReport(
            by_window,
            by_pool,
            dict(sorted(by_life_cycle.items())),
            by_cold_pool,
            self.suspended_requests,
            self.resumed_requests,
            self.disk_prefetch_blocks,
            list(self.cached_tokens_by_level),
            list(self.metadata.cache_level_tiers),
        )


def _add_delta(buckets: dict[int, dict[str, int]], key: int, delta: dict[str, int]) -> None:
    accumulated = buckets.setdefault(key, {})
    for name, value in delta.items():
        accumulated[name] = accumulated.get(name, 0) + value


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


def _serialize_v2_window_iteration_stats(stats) -> dict:
    """Serialize a V2 window bucket, dropping the cold-tier fields.

    A window bucket is keyed by the hot grouping, and cold levels group lifecycles independently, so
    no window owns a cold pool group. V2 reports cold blocks only in
    ``kvCacheIterationStatsByColdPoolGroup``. Derived from the full field set rather than a second
    hard-coded key list so that new fields are picked up automatically.
    """
    return {
        key: value
        for key, value in serialize_kv_cache_iteration_stats(stats).items()
        if not key.startswith("secondary")
    }


def append_kv_cache_iteration_stats(stats_dict: dict, kv_iter_stats) -> None:
    if kv_iter_stats is None:
        return
    if isinstance(kv_iter_stats, KVCacheV2IterationStatsSnapshot):
        kv_iter_stats = kv_iter_stats.build_report()
    if isinstance(kv_iter_stats, KVCacheV2IterationStatsReport):
        by_window_size = kv_iter_stats.by_window_size
        by_pool_group = kv_iter_stats.by_pool_group
        serialize_window = _serialize_v2_window_iteration_stats
    else:
        by_window_size = kv_iter_stats
        by_pool_group = None
        # Legacy V1 windows carry their own cold-tier counters; leave that payload untouched.
        serialize_window = serialize_kv_cache_iteration_stats

    stats_dict["kvCacheIterationStats"] = {
        str(window_size): serialize_window(stats) for window_size, stats in by_window_size.items()
    }
    if by_pool_group is None:
        return

    stats_dict["iterSuspendedRequests"] = kv_iter_stats.suspended_requests
    stats_dict["iterResumedRequests"] = kv_iter_stats.resumed_requests
    stats_dict["iterDiskPrefetchBlocks"] = kv_iter_stats.disk_prefetch_blocks
    stats_dict["iterCachedTokensByLevel"] = list(kv_iter_stats.cached_tokens_by_level)
    if kv_iter_stats.cache_level_tiers:
        stats_dict["kvCacheLevelTiers"] = list(kv_iter_stats.cache_level_tiers)

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

    # Keyed by cold pool-group id, not hot. A cold group spanning several hot groups is attributed to
    # none of them above, so this is the only complete account of host/disk blocks.
    if kv_iter_stats.by_cold_pool_group:
        stats_dict["kvCacheIterationStatsByColdPoolGroup"] = {
            str(cold_pool_group_id): {
                "coldPoolGroupId": stats.pool_group_id,
                "slotSize": list(stats.slot_size),
                "windowSizes": list(stats.window_sizes),
                **serialize_kv_cache_iteration_stats(
                    stats.stats, KV_CACHE_ITERATION_STATS_COLD_POOL_GROUP_KEYS
                ),
            }
            for cold_pool_group_id, stats in kv_iter_stats.by_cold_pool_group.items()
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
            # Indexed by cache level, so the arrays are as long as the configured tier list.
            if stats.full_reused_blocks_by_level:
                serialized["iterFullReusedBlocksByLevel"] = list(stats.full_reused_blocks_by_level)
            if stats.partial_reused_blocks_by_level:
                serialized["iterPartialReusedBlocksByLevel"] = list(
                    stats.partial_reused_blocks_by_level
                )
        stats_by_life_cycle[str(life_cycle_id)] = serialized
    stats_dict["kvCacheIterationStatsByLifecycle"] = stats_by_life_cycle
