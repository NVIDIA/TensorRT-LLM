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
from .._stats import KVCacheIterationStatsDelta, KVCacheStatsDelta


@dataclass(slots=True)
class _PendingAllocationSegment:
    life_cycle: LifeCycleId
    block_begin: BlockOrdinal
    block_end: BlockOrdinal
    beam_width: int
    count_as_missed: bool
    count_as_generation: bool


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
    allocation_segments: list[_PendingAllocationSegment] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return (
            self.request_stats.empty
            and self.global_stats.empty
            and not self.iteration_stats_by_life_cycle
        )

    def clear(self) -> None:
        self.request_stats.clear()
        self.global_stats.clear()
        self.iteration_stats_by_life_cycle.clear()
        self.allocation_segments.clear()

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
        stats = KVCacheStatsDelta(
            alloc_total_blocks=num_blocks,
            alloc_new_blocks=num_blocks,
            missed_blocks=num_blocks if segment.count_as_missed else 0,
        )
        request_stats = stats.copy()
        iteration_stats = KVCacheIterationStatsDelta(
            iter_alloc_total_blocks=num_blocks,
            iter_alloc_new_blocks=num_blocks,
            iter_missed_blocks=num_blocks if segment.count_as_missed else 0,
            iter_gen_alloc_blocks=num_blocks if segment.count_as_generation else 0,
        )
        return _PendingStatsDelta(stats, request_stats, iteration_stats, segment.life_cycle)

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
