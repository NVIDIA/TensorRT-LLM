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
"""Bounce config + pluggable sizing policy (internal): a Config on
TransferWorkerConfig.bounce enables bounce, None keeps the per-block path.
config_from_size() turns the size knob into the on/off switch (size 0 => off)."""

from dataclasses import dataclass, field
from typing import Optional

_MIB = 1024 * 1024


def _round_up(a: int, b: int) -> int:
    return (a + b - 1) // b * b


@dataclass(frozen=True)
class SizingContext:
    free_bytes: int  # free at setup, after the KV pool claimed its fraction
    total_bytes: int
    chunk_bytes: int
    device_id: int


@dataclass(frozen=True)
class Sizing:
    """resolve() returns one region's byte size (there are two regions: one for send, one for recv)."""

    def resolve(self, ctx: SizingContext) -> int:
        raise NotImplementedError


# Default size in MiB for each region — one for send, one for recv; raise it to bounce larger
# single transfers, lower it to save VRAM. Clamped to the free-memory budget at setup.
DEFAULT_CAPACITY_MB = 384


@dataclass(frozen=True)
class FixedSizing(Sizing):
    """Absolute capacity for each region (send and recv) (default), clamped to free mem at setup."""

    capacity_mb: int = DEFAULT_CAPACITY_MB

    def resolve(self, ctx: SizingContext) -> int:
        return max(_round_up(self.capacity_mb * _MIB, ctx.chunk_bytes), ctx.chunk_bytes)


# Bounce takes at most this fraction of post-KV-pool free memory.
_HEADROOM_FRACTION = 0.5


def fit_within_free(
    capacity_bytes: int,
    *,
    free_bytes: int,
    chunk_bytes: int,
    max_free_fraction: float = _HEADROOM_FRACTION,
) -> Optional[int]:
    """OOM guard: clamp each region to max_free_fraction*free/2 (two regions), rounded to a chunk; None if none fits."""
    budget_per_dir = (int(free_bytes * max_free_fraction) // 2 // chunk_bytes) * chunk_bytes
    if budget_per_dir < chunk_bytes:
        return None
    capacity_bytes = min(capacity_bytes, budget_per_dir)
    capacity_bytes = max(capacity_bytes, chunk_bytes)
    return capacity_bytes


@dataclass
class Config:
    sizing: Sizing = field(default_factory=FixedSizing)  # how much VRAM (pluggable)
    chunk_mb: int = 32  # VMM physical-chunk size; large => one multi-rail descriptor
    min_blocks: int = 96  # don't bounce short transfers


def config_from_size(size_mb: int) -> Optional[Config]:
    """Build a bounce Config from a per-region capacity in MiB, or None (bounce off) when size_mb <= 0.

    The size doubles as the on/off switch (one dial instead of a separate bool); driven by the
    CacheTransceiverConfig.kv_cache_bounce_size_mb knob.
    """
    if size_mb is None or size_mb <= 0:
        return None
    return Config(sizing=FixedSizing(capacity_mb=size_mb))
