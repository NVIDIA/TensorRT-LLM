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

from .._common import NDEBUG, BlockOrdinal, CacheLevel
from .._life_cycle_registry import LifeCycleId
from .._stats import (
    CountsByLevel,
    KVCacheIterationStatsDelta,
    KVCacheStatsDelta,
    ReusedBlocksByLevel,
    SsmSnapshotIterationStatsDelta,
    add_counts_by_level,
)


@dataclass(slots=True)
class _PendingAllocationSegment:
    life_cycle: LifeCycleId
    block_begin: BlockOrdinal
    block_end: BlockOrdinal
    beam_width: int
    count_as_missed: bool
    count_as_generation: bool
    record_manager_stats: bool
    record_request_stats: bool


@dataclass(slots=True)
class _PendingStatsDelta:
    global_stats: KVCacheStatsDelta
    request_stats: KVCacheStatsDelta
    iteration_stats: KVCacheIterationStatsDelta
    life_cycle: LifeCycleId | None = None

    @property
    def empty(self) -> bool:
        return self.global_stats.empty and self.request_stats.empty and self.iteration_stats.empty


@dataclass(slots=True)
class _PendingStats:
    request_stats: KVCacheStatsDelta = field(default_factory=KVCacheStatsDelta)
    global_stats: KVCacheStatsDelta = field(default_factory=KVCacheStatsDelta)
    iteration_stats_by_life_cycle: dict[LifeCycleId, KVCacheIterationStatsDelta] = field(
        default_factory=dict
    )
    ssm_snapshot_iteration_stats_by_life_cycle: dict[
        LifeCycleId, SsmSnapshotIterationStatsDelta
    ] = field(default_factory=dict)
    reused_blocks_by_level_by_life_cycle: dict[LifeCycleId, ReusedBlocksByLevel] = field(
        default_factory=dict
    )
    # Cached-token attribution for the sequence's reuse match, indexed by cache level.
    #
    # Unlike the reuse counters this is a manager-global quantity rather than a per-lifecycle one:
    # a match spans every lifecycle at once (the final SSM checkpoint summarizes the whole recurrent
    # prefix, so its tier applies to every matched token), leaving no single lifecycle to attribute
    # it to. It still rides the pending-stats lifecycle so it is committed or discarded together
    # with the counters it was derived from -- in particular, a dummy sequence's attribution is
    # dropped by the same discard_pending_stats() that drops its reuse counters.
    cached_tokens_by_level: CountsByLevel = field(default_factory=list)
    allocation_segments: list[_PendingAllocationSegment] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return (
            self.request_stats.empty
            and self.global_stats.empty
            and not self.iteration_stats_by_life_cycle
            and not self.ssm_snapshot_iteration_stats_by_life_cycle
            and not any(self.cached_tokens_by_level)
        )

    def clear(self) -> None:
        self.request_stats.clear()
        self.global_stats.clear()
        self.iteration_stats_by_life_cycle.clear()
        self.ssm_snapshot_iteration_stats_by_life_cycle.clear()
        self.reused_blocks_by_level_by_life_cycle.clear()
        self.cached_tokens_by_level = []
        self.allocation_segments.clear()

    def record_cached_tokens_by_level(self, counts: CountsByLevel) -> bool:
        if not any(counts):
            return False
        self.cached_tokens_by_level = add_counts_by_level(self.cached_tokens_by_level, counts)
        return True

    def discount_cached_tokens_by_level(self, level: CacheLevel, num_tokens: int) -> None:
        """Remove ``num_tokens`` from ``level``.

        Clamps at zero: the attribution only feeds an observability counter, so an inconsistency
        must not underflow it into a nonsense negative reading.
        """
        assert NDEBUG or level < len(self.cached_tokens_by_level)
        if level >= len(self.cached_tokens_by_level):
            return
        assert NDEBUG or self.cached_tokens_by_level[level] >= num_tokens
        self.cached_tokens_by_level[level] = max(0, self.cached_tokens_by_level[level] - num_tokens)

    def add(self, delta: _PendingStatsDelta) -> bool:
        if delta.empty:
            return False
        if not delta.global_stats.empty:
            self.global_stats.add(delta.global_stats)
        if not delta.request_stats.empty:
            self.request_stats.add(delta.request_stats)
        if not delta.iteration_stats.empty:
            assert delta.life_cycle is not None
            pending = self.iteration_stats_by_life_cycle.setdefault(
                delta.life_cycle, KVCacheIterationStatsDelta()
            )
            pending.add(delta.iteration_stats)
        return True

    def subtract(self, delta: _PendingStatsDelta) -> bool:
        if delta.empty:
            return False
        if not delta.global_stats.empty:
            self.global_stats.subtract(delta.global_stats)
        if not delta.request_stats.empty:
            self.request_stats.subtract(delta.request_stats)
        if not delta.iteration_stats.empty:
            assert delta.life_cycle is not None
            pending = self.iteration_stats_by_life_cycle.get(delta.life_cycle)
            if pending is not None:
                pending.subtract(delta.iteration_stats)
                if pending.empty:
                    del self.iteration_stats_by_life_cycle[delta.life_cycle]
        return True

    @staticmethod
    def _allocation_delta(
        segment: _PendingAllocationSegment,
        block_begin: BlockOrdinal,
        block_end: BlockOrdinal,
    ) -> _PendingStatsDelta:
        num_blocks = max(0, int(block_end) - int(block_begin)) * segment.beam_width
        manager_stats = (
            KVCacheStatsDelta(
                alloc_total_blocks=num_blocks,
                alloc_new_blocks=num_blocks,
                missed_blocks=num_blocks if segment.count_as_missed else 0,
            )
            if segment.record_manager_stats
            else KVCacheStatsDelta()
        )
        request_stats = (
            KVCacheStatsDelta(
                alloc_total_blocks=num_blocks,
                alloc_new_blocks=num_blocks,
                missed_blocks=num_blocks if segment.count_as_missed else 0,
            )
            if segment.record_request_stats
            else KVCacheStatsDelta()
        )
        iteration_stats = (
            KVCacheIterationStatsDelta(
                iter_alloc_total_blocks=num_blocks,
                iter_alloc_new_blocks=num_blocks,
                iter_missed_blocks=num_blocks if segment.count_as_missed else 0,
                iter_gen_alloc_blocks=num_blocks if segment.count_as_generation else 0,
            )
            if segment.record_manager_stats
            else KVCacheIterationStatsDelta()
        )
        return _PendingStatsDelta(manager_stats, request_stats, iteration_stats, segment.life_cycle)

    def record_allocation_range(
        self,
        life_cycle: LifeCycleId,
        block_begin: BlockOrdinal,
        block_end: BlockOrdinal,
        *,
        beam_width: int,
        count_as_missed: bool,
        count_as_generation: bool = False,
        record_manager_stats: bool,
        record_request_stats: bool,
    ) -> bool:
        if block_begin >= block_end or not (record_manager_stats or record_request_stats):
            return False
        segment = _PendingAllocationSegment(
            life_cycle=life_cycle,
            block_begin=block_begin,
            block_end=block_end,
            beam_width=beam_width,
            count_as_missed=count_as_missed,
            count_as_generation=count_as_generation,
            record_manager_stats=record_manager_stats,
            record_request_stats=record_request_stats,
        )
        if not self.add(self._allocation_delta(segment, block_begin, block_end)):
            return False
        self.allocation_segments.append(segment)
        return True

    def record_reuse(
        self,
        life_cycle: LifeCycleId,
        *,
        full_reused_blocks: int,
        partial_reused_blocks: int,
        by_level: ReusedBlocksByLevel | None = None,
        record_manager_stats: bool,
        record_request_stats: bool,
    ) -> bool:
        """Record reuse counts for one life cycle.

        ``by_level`` splits the same full/partial counts across the cache levels the reused
        pages were resident on. It rides along with the scalar counters so both are committed
        or discarded together; reuse is never rolled back (only allocation ranges are), so
        add-only is enough.
        """
        reused_blocks = full_reused_blocks + partial_reused_blocks
        if reused_blocks == 0 or not (record_manager_stats or record_request_stats):
            return False
        if record_manager_stats and by_level is not None:
            self.reused_blocks_by_level_by_life_cycle.setdefault(
                life_cycle, ReusedBlocksByLevel()
            ).add(by_level)
        return self.add(
            _PendingStatsDelta(
                global_stats=(
                    KVCacheStatsDelta(reused_blocks=reused_blocks)
                    if record_manager_stats
                    else KVCacheStatsDelta()
                ),
                request_stats=(
                    KVCacheStatsDelta(reused_blocks=reused_blocks)
                    if record_request_stats
                    else KVCacheStatsDelta()
                ),
                iteration_stats=(
                    KVCacheIterationStatsDelta(
                        iter_reused_blocks=reused_blocks,
                        iter_full_reused_blocks=full_reused_blocks,
                        iter_partial_reused_blocks=partial_reused_blocks,
                    )
                    if record_manager_stats
                    else KVCacheIterationStatsDelta()
                ),
                life_cycle=life_cycle,
            )
        )

    def record_ssm_snapshot_lookup(
        self,
        life_cycle: LifeCycleId,
        *,
        lookup_tokens: int,
        reused_tokens: int,
        tokens_per_block: int,
    ) -> bool:
        if lookup_tokens == 0:
            return False
        assert lookup_tokens > 0
        assert 0 <= reused_tokens <= lookup_tokens
        assert tokens_per_block > 0

        is_hit = reused_tokens > 0
        # Alignment describes the reusable snapshot boundary, not whether the
        # state itself is complete. Every hit represents one complete SSM
        # snapshot; token counters carry the benefit of that single lookup.
        delta = SsmSnapshotIterationStatsDelta(
            iter_snapshot_lookups=1,
            iter_snapshot_hits=int(is_hit),
            iter_snapshot_misses=int(not is_hit),
            iter_reused_tokens=reused_tokens,
            iter_unreused_tokens=lookup_tokens - reused_tokens,
            iter_aligned_snapshot_hits=int(is_hit and reused_tokens % tokens_per_block == 0),
            iter_unaligned_snapshot_hits=int(is_hit and reused_tokens % tokens_per_block != 0),
        )
        pending = self.ssm_snapshot_iteration_stats_by_life_cycle.setdefault(
            life_cycle, SsmSnapshotIterationStatsDelta()
        )
        pending.add(delta)
        return True

    def subtract_allocation_range(self, block_begin: BlockOrdinal, block_end: BlockOrdinal) -> bool:
        if block_begin >= block_end or not self.allocation_segments:
            return False
        changed = False
        idx = len(self.allocation_segments) - 1
        while idx >= 0:
            segment = self.allocation_segments[idx]
            if segment.block_end <= block_begin:
                break
            removed_begin = max(block_begin, segment.block_begin)
            removed_end = min(block_end, segment.block_end)
            if removed_begin >= removed_end:
                idx -= 1
                continue
            changed = True
            self.subtract(self._allocation_delta(segment, removed_begin, removed_end))
            if removed_begin <= segment.block_begin:
                del self.allocation_segments[idx]
            else:
                assert removed_end == segment.block_end
                segment.block_end = removed_begin
            idx -= 1
        return changed
