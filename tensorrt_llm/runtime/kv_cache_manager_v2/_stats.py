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

import enum
from dataclasses import dataclass


class KVCacheStatsScope(enum.Enum):
    NONE = enum.auto()
    CONTEXT = enum.auto()
    GENERATION = enum.auto()


@dataclass(slots=True)
class KVCacheStatsDelta:
    alloc_total_blocks: int = 0
    alloc_new_blocks: int = 0
    reused_blocks: int = 0
    missed_blocks: int = 0

    def add(self, other: "KVCacheStatsDelta") -> None:
        self.alloc_total_blocks += other.alloc_total_blocks
        self.alloc_new_blocks += other.alloc_new_blocks
        self.reused_blocks += other.reused_blocks
        self.missed_blocks += other.missed_blocks

    def subtract(self, other: "KVCacheStatsDelta") -> None:
        self.alloc_total_blocks -= other.alloc_total_blocks
        self.alloc_new_blocks -= other.alloc_new_blocks
        self.reused_blocks -= other.reused_blocks
        self.missed_blocks -= other.missed_blocks

    def clear(self) -> None:
        self.alloc_total_blocks = 0
        self.alloc_new_blocks = 0
        self.reused_blocks = 0
        self.missed_blocks = 0

    def copy(self) -> "KVCacheStatsDelta":
        return KVCacheStatsDelta(
            alloc_total_blocks=self.alloc_total_blocks,
            alloc_new_blocks=self.alloc_new_blocks,
            reused_blocks=self.reused_blocks,
            missed_blocks=self.missed_blocks,
        )

    @property
    def empty(self) -> bool:
        return (
            self.alloc_total_blocks == 0
            and self.alloc_new_blocks == 0
            and self.reused_blocks == 0
            and self.missed_blocks == 0
        )

    @property
    def cache_hit_rate(self) -> float:
        total = self.reused_blocks + self.missed_blocks
        return 0.0 if self.reused_blocks == 0 or total == 0 else self.reused_blocks / total


@dataclass(slots=True)
class KVCacheIterationStatsDelta:
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

    def add(self, other: "KVCacheIterationStatsDelta") -> None:
        self.iter_alloc_total_blocks += other.iter_alloc_total_blocks
        self.iter_alloc_new_blocks += other.iter_alloc_new_blocks
        self.iter_reused_blocks += other.iter_reused_blocks
        self.iter_full_reused_blocks += other.iter_full_reused_blocks
        self.iter_partial_reused_blocks += other.iter_partial_reused_blocks
        self.iter_missed_blocks += other.iter_missed_blocks
        self.iter_gen_alloc_blocks += other.iter_gen_alloc_blocks
        self.iter_onboard_blocks += other.iter_onboard_blocks
        self.iter_onboard_bytes += other.iter_onboard_bytes
        self.iter_offload_blocks += other.iter_offload_blocks
        self.iter_offload_bytes += other.iter_offload_bytes
        self.iter_intra_device_copy_blocks += other.iter_intra_device_copy_blocks
        self.iter_intra_device_copy_bytes += other.iter_intra_device_copy_bytes

    def subtract(self, other: "KVCacheIterationStatsDelta") -> None:
        self.iter_alloc_total_blocks -= other.iter_alloc_total_blocks
        self.iter_alloc_new_blocks -= other.iter_alloc_new_blocks
        self.iter_reused_blocks -= other.iter_reused_blocks
        self.iter_full_reused_blocks -= other.iter_full_reused_blocks
        self.iter_partial_reused_blocks -= other.iter_partial_reused_blocks
        self.iter_missed_blocks -= other.iter_missed_blocks
        self.iter_gen_alloc_blocks -= other.iter_gen_alloc_blocks
        self.iter_onboard_blocks -= other.iter_onboard_blocks
        self.iter_onboard_bytes -= other.iter_onboard_bytes
        self.iter_offload_blocks -= other.iter_offload_blocks
        self.iter_offload_bytes -= other.iter_offload_bytes
        self.iter_intra_device_copy_blocks -= other.iter_intra_device_copy_blocks
        self.iter_intra_device_copy_bytes -= other.iter_intra_device_copy_bytes

    def copy(self) -> "KVCacheIterationStatsDelta":
        return KVCacheIterationStatsDelta(
            iter_alloc_total_blocks=self.iter_alloc_total_blocks,
            iter_alloc_new_blocks=self.iter_alloc_new_blocks,
            iter_reused_blocks=self.iter_reused_blocks,
            iter_full_reused_blocks=self.iter_full_reused_blocks,
            iter_partial_reused_blocks=self.iter_partial_reused_blocks,
            iter_missed_blocks=self.iter_missed_blocks,
            iter_gen_alloc_blocks=self.iter_gen_alloc_blocks,
            iter_onboard_blocks=self.iter_onboard_blocks,
            iter_onboard_bytes=self.iter_onboard_bytes,
            iter_offload_blocks=self.iter_offload_blocks,
            iter_offload_bytes=self.iter_offload_bytes,
            iter_intra_device_copy_blocks=self.iter_intra_device_copy_blocks,
            iter_intra_device_copy_bytes=self.iter_intra_device_copy_bytes,
        )

    @property
    def empty(self) -> bool:
        return (
            self.iter_alloc_total_blocks == 0
            and self.iter_alloc_new_blocks == 0
            and self.iter_reused_blocks == 0
            and self.iter_full_reused_blocks == 0
            and self.iter_partial_reused_blocks == 0
            and self.iter_missed_blocks == 0
            and self.iter_gen_alloc_blocks == 0
            and self.iter_onboard_blocks == 0
            and self.iter_onboard_bytes == 0
            and self.iter_offload_blocks == 0
            and self.iter_offload_bytes == 0
            and self.iter_intra_device_copy_blocks == 0
            and self.iter_intra_device_copy_bytes == 0
        )

    @property
    def iter_cache_hit_rate(self) -> float:
        total = self.iter_reused_blocks + self.iter_missed_blocks
        return (
            0.0 if self.iter_reused_blocks == 0 or total == 0 else self.iter_reused_blocks / total
        )
