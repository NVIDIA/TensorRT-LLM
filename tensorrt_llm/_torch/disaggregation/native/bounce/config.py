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
"""Bounce configuration and pluggable sizing policy. A config enables bounce; leaving it unset keeps
the per-block path. The size knob doubles as the on and off switch."""

from dataclasses import dataclass, field
from typing import Optional

_MIB = 1024 * 1024


def _round_up(a: int, b: int) -> int:
    return (a + b - 1) // b * b


@dataclass(frozen=True)
class SizingContext:
    free_bytes: int  # free at setup, after the cache pool claimed its fraction
    total_bytes: int
    chunk_bytes: int
    device_id: int


@dataclass(frozen=True)
class Sizing:
    """Returns the byte size of one region; there are two, one for sending and one for receiving."""

    def resolve(self, ctx: SizingContext) -> int:
        raise NotImplementedError


# Default size in MiB per region. Raise it to bounce larger single transfers, lower it to save
# memory. It is clamped to the free-memory budget at setup.
DEFAULT_CAPACITY_MB = 384


@dataclass(frozen=True)
class FixedSizing(Sizing):
    """A fixed capacity per region, clamped to free memory at setup."""

    capacity_mb: int = DEFAULT_CAPACITY_MB

    def resolve(self, ctx: SizingContext) -> int:
        return max(_round_up(self.capacity_mb * _MIB, ctx.chunk_bytes), ctx.chunk_bytes)


# bounce takes at most this fraction of the free memory left after the cache pool
_HEADROOM_FRACTION = 0.5


def fit_within_free(
    capacity_bytes: int,
    *,
    free_bytes: int,
    chunk_bytes: int,
    max_free_fraction: float = _HEADROOM_FRACTION,
) -> Optional[int]:
    """Clamp each region so the two together stay within the allowed fraction of free memory, rounded
    to a chunk. Returns None if not even one chunk fits."""
    budget_per_dir = (int(free_bytes * max_free_fraction) // 2 // chunk_bytes) * chunk_bytes
    if budget_per_dir < chunk_bytes:
        return None
    capacity_bytes = min(capacity_bytes, budget_per_dir)
    capacity_bytes = max(capacity_bytes, chunk_bytes)
    return capacity_bytes


@dataclass
class Config:
    sizing: Sizing = field(default_factory=FixedSizing)  # how much memory to reserve (pluggable)
    chunk_mb: int = 32  # physical chunk size; a large chunk keeps the write to a single descriptor
    # skip bounce below this many blocks (roughly 12k tokens at 128 per block); heuristic, tunable
    min_blocks: int = 96


def config_from_size(size_mb: int) -> Optional[Config]:
    """Build a bounce config from a per-region size in MiB, or None to leave bounce off when the size
    is not positive. The size is both the capacity and the on and off switch."""
    if size_mb is None or size_mb <= 0:
        return None
    return Config(sizing=FixedSizing(capacity_mb=size_mb))
