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

from .._common import BlockOrdinal
from .._life_cycle_registry import LifeCycleId
from .._stats import KVCacheIterationStatsDelta, KVCacheStatsDelta, SsmSnapshotIterationStatsDelta


@dataclass(slots=True)
class _PendingAllocationSegment:
    life_cycle: LifeCycleId
    block_begin: BlockOrdinal
    block_end: BlockOrdinal
    beam_width: int
    count_as_missed: bool
    count_as_generation: bool
    # Allocated for content a KV connector loaded from an external store:
    # booked as reused rather than missed.
    count_as_connector_reused: bool = False


@dataclass(slots=True)
class _PendingStatsDelta:
    global_stats: KVCacheStatsDelta
    request_stats: KVCacheStatsDelta
    iteration_stats: KVCacheIterationStatsDelta
    life_cycle: LifeCycleId | None = None
    connector_reused_blocks: int = 0

    @property
    def empty(self) -> bool:
        return (
            self.global_stats.empty
            and self.request_stats.empty
            and self.iteration_stats.empty
            and self.connector_reused_blocks == 0
        )


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
    # The connector-loaded share of the reused blocks in
    # `iteration_stats_by_life_cycle`.
    connector_reused_blocks_by_life_cycle: dict[LifeCycleId, int] = field(default_factory=dict)
    allocation_segments: list[_PendingAllocationSegment] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return (
            self.request_stats.empty
            and self.global_stats.empty
            and not self.iteration_stats_by_life_cycle
            and not self.ssm_snapshot_iteration_stats_by_life_cycle
            and not self.connector_reused_blocks_by_life_cycle
        )

    def clear(self) -> None:
        self.request_stats.clear()
        self.global_stats.clear()
        self.iteration_stats_by_life_cycle.clear()
        self.ssm_snapshot_iteration_stats_by_life_cycle.clear()
        self.connector_reused_blocks_by_life_cycle.clear()
        self.allocation_segments.clear()

    def _add_connector_reused_blocks(self, life_cycle: LifeCycleId | None, num_blocks: int) -> None:
        if num_blocks == 0:
            return
        assert life_cycle is not None
        total = self.connector_reused_blocks_by_life_cycle.get(life_cycle, 0) + num_blocks
        if total == 0:
            del self.connector_reused_blocks_by_life_cycle[life_cycle]
        else:
            self.connector_reused_blocks_by_life_cycle[life_cycle] = total

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
        self._add_connector_reused_blocks(delta.life_cycle, delta.connector_reused_blocks)
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
        self._add_connector_reused_blocks(delta.life_cycle, -delta.connector_reused_blocks)
        return True

    @staticmethod
    def _allocation_delta(
        segment: _PendingAllocationSegment,
        block_begin: BlockOrdinal,
        block_end: BlockOrdinal,
    ) -> _PendingStatsDelta:
        num_blocks = max(0, int(block_end) - int(block_begin)) * segment.beam_width
        missed_blocks = num_blocks if segment.count_as_missed else 0
        connector_reused_blocks = num_blocks if segment.count_as_connector_reused else 0
        stats = KVCacheStatsDelta(
            alloc_total_blocks=num_blocks,
            alloc_new_blocks=num_blocks,
            reused_blocks=connector_reused_blocks,
            missed_blocks=missed_blocks,
        )
        request_stats = stats.copy()
        iteration_stats = KVCacheIterationStatsDelta(
            iter_alloc_total_blocks=num_blocks,
            iter_alloc_new_blocks=num_blocks,
            iter_reused_blocks=connector_reused_blocks,
            iter_full_reused_blocks=connector_reused_blocks,
            iter_missed_blocks=missed_blocks,
            iter_gen_alloc_blocks=num_blocks if segment.count_as_generation else 0,
        )
        return _PendingStatsDelta(
            stats,
            request_stats,
            iteration_stats,
            segment.life_cycle,
            connector_reused_blocks,
        )

    @staticmethod
    def _sub_segment(
        segment: _PendingAllocationSegment,
        block_begin: BlockOrdinal,
        block_end: BlockOrdinal,
        *,
        count_as_connector_reused: bool,
    ) -> _PendingAllocationSegment:
        return _PendingAllocationSegment(
            life_cycle=segment.life_cycle,
            block_begin=block_begin,
            block_end=block_end,
            beam_width=segment.beam_width,
            count_as_missed=segment.count_as_missed and not count_as_connector_reused,
            count_as_generation=segment.count_as_generation,
            count_as_connector_reused=count_as_connector_reused,
        )

    def record_connector_reuse(self, block_begin: BlockOrdinal, block_end: BlockOrdinal) -> bool:
        """Rebook pending missed allocations in ``[block_begin, block_end)`` as reused.

        The blocks stay counted as allocated: the connector fills freshly
        allocated pages instead of the forward pass computing them.
        """
        if block_begin >= block_end:
            return False
        changed = False
        segments: list[_PendingAllocationSegment] = []
        for segment in self.allocation_segments:
            served_begin = max(block_begin, segment.block_begin)
            served_end = min(block_end, segment.block_end)
            if served_begin >= served_end or not segment.count_as_missed:
                segments.append(segment)
                continue
            changed = True
            self.subtract(self._allocation_delta(segment, served_begin, served_end))
            if segment.block_begin < served_begin:
                segments.append(
                    self._sub_segment(
                        segment, segment.block_begin, served_begin, count_as_connector_reused=False
                    )
                )
            served = self._sub_segment(
                segment, served_begin, served_end, count_as_connector_reused=True
            )
            self.add(self._allocation_delta(served, served_begin, served_end))
            segments.append(served)
            if served_end < segment.block_end:
                segments.append(
                    self._sub_segment(
                        segment, served_end, segment.block_end, count_as_connector_reused=False
                    )
                )
        self.allocation_segments = segments
        return changed

    def record_allocation_range(
        self,
        life_cycle: LifeCycleId,
        block_begin: BlockOrdinal,
        block_end: BlockOrdinal,
        *,
        beam_width: int,
        count_as_missed: bool,
        count_as_generation: bool = False,
    ) -> bool:
        if block_begin >= block_end:
            return False
        segment = _PendingAllocationSegment(
            life_cycle=life_cycle,
            block_begin=block_begin,
            block_end=block_end,
            beam_width=beam_width,
            count_as_missed=count_as_missed,
            count_as_generation=count_as_generation,
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
    ) -> bool:
        reused_blocks = full_reused_blocks + partial_reused_blocks
        if reused_blocks == 0:
            return False
        return self.add(
            _PendingStatsDelta(
                global_stats=KVCacheStatsDelta(reused_blocks=reused_blocks),
                request_stats=KVCacheStatsDelta(reused_blocks=reused_blocks),
                iteration_stats=KVCacheIterationStatsDelta(
                    iter_reused_blocks=reused_blocks,
                    iter_full_reused_blocks=full_reused_blocks,
                    iter_partial_reused_blocks=partial_reused_blocks,
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
        # Not ordered by position across life cycles once `record_connector_reuse`
        # has split segments, so every segment is examined.
        idx = len(self.allocation_segments) - 1
        while idx >= 0:
            segment = self.allocation_segments[idx]
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
