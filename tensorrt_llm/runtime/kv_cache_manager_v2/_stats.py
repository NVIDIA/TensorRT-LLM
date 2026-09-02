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

import dataclasses
from dataclasses import dataclass, fields

from ._common import CacheLevel
from ._utils import TypedIndexList


class _StatsDeltaMixin:
    __slots__ = ()

    def add(self, other) -> None:
        for field in fields(self):
            name = field.name
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def subtract(self, other) -> None:
        for field in fields(self):
            name = field.name
            setattr(self, name, getattr(self, name) - getattr(other, name))

    def clear(self) -> None:
        for field in fields(self):
            setattr(self, field.name, 0)

    def copy(self):
        return type(self)(**{field.name: getattr(self, field.name) for field in fields(self)})

    @property
    def empty(self) -> bool:
        return all(getattr(self, field.name) == 0 for field in fields(self))


@dataclass(slots=True)
class KVCacheStatsDelta(_StatsDeltaMixin):
    alloc_total_blocks: int = 0
    alloc_new_blocks: int = 0
    reused_blocks: int = 0
    missed_blocks: int = 0


@dataclass(slots=True)
class KVCacheIterationStatsDelta(_StatsDeltaMixin):
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
    # Host-tier pages released by LRU without ever being onboarded back to GPU
    # in the lifetime since they were offloaded. Counted at the drop site in
    # _storage_manager._prepare_free_slots when is_last_level(lvl).
    iter_host_dropped_blocks: int = 0
    iter_host_dropped_bytes: int = 0

    @property
    def iter_cache_hit_rate(self) -> float:
        total = self.iter_reused_blocks + self.iter_missed_blocks
        if self.iter_reused_blocks == 0 or total == 0:
            return 0.0
        return self.iter_reused_blocks / total


CountsByLevel = TypedIndexList[CacheLevel, int]
"""Counters indexed by CacheLevel: entry ``i`` belongs to the i-th configured cache tier.

The length follows the configured tier list instead of a hard-coded gpu/host/disk split, which
is what lets a deployment with a hot and a cold GPU level report them as two distinct entries.
"""


def add_counts_by_level(dst: CountsByLevel, src: CountsByLevel) -> CountsByLevel:
    """Element-wise accumulate, widening ``dst`` when ``src`` covers more levels."""
    if len(dst) < len(src):
        dst = list(dst) + [0] * (len(src) - len(dst))
    for level, value in enumerate(src):
        dst[level] += value
    return dst


@dataclass(slots=True)
class ReusedBlocksByLevel:
    """Reuse block counts split by the cache level the reused pages were resident on.

    Kept outside ``KVCacheIterationStatsDelta`` on purpose: the level count is a runtime
    quantity, while that dataclass is a fixed-field record whose field-wise add/subtract
    helpers assume scalar members.
    """

    full: CountsByLevel = dataclasses.field(default_factory=list)
    partial: CountsByLevel = dataclasses.field(default_factory=list)

    def add(self, other: "ReusedBlocksByLevel") -> None:
        self.full = add_counts_by_level(self.full, other.full)
        self.partial = add_counts_by_level(self.partial, other.partial)

    @property
    def empty(self) -> bool:
        return not any(self.full) and not any(self.partial)


@dataclass(slots=True)
class SsmSnapshotIterationStatsDelta(_StatsDeltaMixin):
    iter_snapshot_lookups: int = 0
    iter_snapshot_hits: int = 0
    iter_snapshot_misses: int = 0
    iter_reused_tokens: int = 0
    iter_unreused_tokens: int = 0
    iter_aligned_snapshot_hits: int = 0
    iter_unaligned_snapshot_hits: int = 0

    @property
    def iter_snapshot_hit_rate(self) -> float:
        if self.iter_snapshot_hits == 0 or self.iter_snapshot_lookups == 0:
            return 0.0
        return self.iter_snapshot_hits / self.iter_snapshot_lookups


_KV_CACHE_ITERATION_STATS_DELTA_FIELDS = tuple(
    field.name for field in fields(KVCacheIterationStatsDelta)
)
