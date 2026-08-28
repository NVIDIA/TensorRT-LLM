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

import os
from dataclasses import dataclass, field
from typing import Optional

from tensorrt_llm import logger

_MIB = 1024 * 1024

# Test/advanced overrides for the size gates below (users only tune the bounce size). Read on the
# generation side, so set them there; unset uses the defaults.
# - min_bytes gates payloads that carry recurrent (mamba/KDA) state: the fallback cost scales
#   with bytes, not block count, so the gate is byte-denominated.
# - min_blocks is the legacy plain-KV gate, kept so existing bounce deployments see no behavior
#   change.
# For Kimi K3 the byte gate never rejects: the fixed ~433 MiB per-request KDA payload always
# clears the 2 MiB default, so arena capacity plus reservation backpressure is the effective
# admission control.
_MIN_BYTES_ENV = "TRTLLM_KV_CACHE_BOUNCE_MIN_BYTES"  # byte gate for recurrent-state payloads
_MIN_BLOCKS_ENV = "TRTLLM_KV_CACHE_BOUNCE_MIN_BLOCKS"  # block-count gate for plain-KV payloads


def _env_int_gate(name: str, default: int) -> int:
    """Read a gate from the env, defensively: unset or malformed falls back to the default (never
    crashing), and the value is clamped to at least 1."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(f"{name}={raw!r} is not an integer; using default {default}")
        return default
    if value < 1:
        logger.warning(f"{name}={value} < 1; clamping to 1 (bounce always clears the gate)")
        return 1
    return value


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


# Byte gate for transfers that carry recurrent (mamba/KDA) state. The cost this gate guards scales
# with BYTES, not blocks: the block-count gate (96 blocks, calibrated for 128-token blocks)
# silently skipped a 433 MiB Kimi-K3 transfer (67 blocks of 32 tokens plus the non-paged KDA state)
# and dropped it onto the ~0.4 GB/s host-staged fallback, a ~1000x cliff. Break-even is small:
# bounce adds one gather plus one scatter copy (device-local, ~hundreds of GB/s) and ~0.1 ms of
# fixed launch/reservation overhead, while the in-place path can be as slow as ~0.4 GB/s
# inter-node — 2 MiB in-place at that rate is ~5 ms vs well under 1 ms bounced. Below 2 MiB the
# fixed overhead dominates and arena slots are better kept for large transfers. Heuristic, tunable
# via TRTLLM_KV_CACHE_BOUNCE_MIN_BYTES.
DEFAULT_MIN_BYTES = 2 * _MIB


@dataclass
class Config:
    sizing: Sizing = field(default_factory=FixedSizing)  # how much memory to reserve (pluggable)
    chunk_mb: int = 32  # physical chunk size; a large chunk keeps the write to a single descriptor
    # Which gate applies depends on the payload (see VmmBounceTransport.reserve): transfers that
    # carry recurrent state use min_bytes; plain-KV transfers keep the original min_blocks gate so
    # pre-existing bounce deployments see no behavior change.
    # byte gate for recurrent-state payloads (see DEFAULT_MIN_BYTES for the rationale)
    min_bytes: int = DEFAULT_MIN_BYTES
    # block-count gate for plain-KV payloads (roughly 12k tokens at 128 per block); heuristic,
    # tunable via TRTLLM_KV_CACHE_BOUNCE_MIN_BLOCKS
    min_blocks: int = 96


def config_from_size(
    size_mb: int, min_blocks: Optional[int] = None, min_bytes: Optional[int] = None
) -> Optional[Config]:
    """Build a bounce config from a per-region size in MiB, or None to leave bounce off (size <= 0).
    Size is both the capacity and the on/off switch. min_bytes (recurrent-state payloads) and
    min_blocks (plain-KV payloads) are the gates below which a transfer stays on the per-block
    path; when unset they come from the env, else the defaults."""
    if size_mb is None or size_mb <= 0:
        return None
    if min_blocks is None:
        min_blocks = _env_int_gate(_MIN_BLOCKS_ENV, Config.min_blocks)
    if min_bytes is None:
        min_bytes = _env_int_gate(_MIN_BYTES_ENV, Config.min_bytes)
    return Config(
        sizing=FixedSizing(capacity_mb=size_mb), min_bytes=min_bytes, min_blocks=min_blocks
    )
