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


# Opt-in creation mode for the sender's gather stream (default 0 keeps the blocking stream).
# A blocking stream implicitly waits on the legacy default stream, so a gather can queue behind a
# whole forward step; a non-blocking stream does not. The gather's KV inputs are already complete
# when a send starts (see VmmBounceTransport._new_stream), so no ordering is lost.
# - 0 or unset: cudaStreamCreate (blocking, legacy behavior)
# - 1: cudaStreamCreateWithFlags(cudaStreamNonBlocking)
# - 2: cudaStreamCreateWithPriority(cudaStreamNonBlocking, greatest priority)
_SEND_STREAM_NONBLOCKING_ENV = "TRTLLM_KV_BOUNCE_SEND_STREAM_NONBLOCKING"
SEND_STREAM_BLOCKING = 0
SEND_STREAM_NONBLOCKING = 1
SEND_STREAM_NONBLOCKING_HIGH_PRIORITY = 2
_SEND_STREAM_MODES = (
    SEND_STREAM_BLOCKING,
    SEND_STREAM_NONBLOCKING,
    SEND_STREAM_NONBLOCKING_HIGH_PRIORITY,
)


def send_stream_mode_from_env() -> int:
    """Read the send-stream creation mode from the env. Unset or empty is 0; any value other than
    0, 1 or 2 raises ValueError, because silently keeping the blocking stream would hide a typo in
    the setting."""
    raw = os.environ.get(_SEND_STREAM_NONBLOCKING_ENV)
    if raw is None or raw.strip() == "":
        return SEND_STREAM_BLOCKING
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{_SEND_STREAM_NONBLOCKING_ENV}={raw!r} is not an integer; expected one of "
            f"{_SEND_STREAM_MODES}"
        ) from None
    if value not in _SEND_STREAM_MODES:
        raise ValueError(
            f"{_SEND_STREAM_NONBLOCKING_ENV}={value} is not supported; expected one of "
            f"{_SEND_STREAM_MODES}"
        )
    return value


# Opt-in role scope for the bounce regions (default "both" keeps the two regions). A context
# (prefill) worker only sends KV and a generation (decode) worker only receives it, so one of the two
# regions is never used; scoping the transport to its role gives the used region the whole budget.
# - "both" or unset: one send and one receive region (legacy)
# - "recv": receive region only (generation workers)
# - "send": send region only (context workers)
_DIRECTION_ENV = "TRTLLM_KV_BOUNCE_DIRECTION"
DIRECTION_BOTH = "both"
DIRECTION_RECV = "recv"
DIRECTION_SEND = "send"
_DIRECTIONS = (DIRECTION_BOTH, DIRECTION_RECV, DIRECTION_SEND)


def direction_from_env() -> str:
    """Read the region scope from the env. Unset or empty is "both"; any other value than "both",
    "recv" or "send" raises ValueError, so a typo cannot silently change the regions."""
    raw = os.environ.get(_DIRECTION_ENV)
    if raw is None or raw.strip() == "":
        return DIRECTION_BOTH
    value = raw.strip().lower()
    if value not in _DIRECTIONS:
        raise ValueError(
            f"{_DIRECTION_ENV}={raw!r} is not supported; expected one of {_DIRECTIONS}"
        )
    return value


# Opt-in just-in-time per-writer receive grant (default 0 keeps the dispatch-time reserve). Read on
# the generation (receiver) side, which decides; the context side follows the flag in the request.
# With it on, no receive region is reserved at dispatch: each context writer asks for a slot of its
# own size right before it gathers, and its slot is released as soon as its own fragments are
# scattered, so a slot is not held through the sender's queue or the sibling writers.
_JIT_ENV = "TRTLLM_KV_BOUNCE_JIT"
# Total time (ms) a context writer keeps asking while the receive region is full before it takes
# the per-fragment path for that one flow. Read on the context (sender) side.
_JIT_WAIT_MS_ENV = "TRTLLM_KV_BOUNCE_JIT_WAIT_MS"
DEFAULT_JIT_WAIT_MS = 1000


def bounce_jit_from_env() -> bool:
    """Read the just-in-time grant switch. Unset or empty is off; any value other than 0 or 1
    raises ValueError, so a typo cannot silently keep the dispatch-time reserve."""
    raw = os.environ.get(_JIT_ENV)
    if raw is None or raw.strip() == "":
        return False
    value = raw.strip()
    if value not in ("0", "1"):
        raise ValueError(f"{_JIT_ENV}={raw!r} is not supported; expected 0 or 1")
    return value == "1"


def bounce_jit_wait_ms_from_env() -> int:
    """Read the sender's grant wait budget in ms. Unset or empty is the default; a value that is
    not an integer of at least 0 raises ValueError."""
    raw = os.environ.get(_JIT_WAIT_MS_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_JIT_WAIT_MS
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{_JIT_WAIT_MS_ENV}={raw!r} is not an integer") from None
    if value < 0:
        raise ValueError(f"{_JIT_WAIT_MS_ENV}={value} must be >= 0")
    return value


# Opt-in batched receive scatter (default 0 keeps one scatter and one stream sync per queued item).
# Read on the generation (receiver) side. With it on, the scatter worker takes every item already
# queued behind the one it woke for, scatters all of their fragments in one batched copy on the same
# blocking scatter stream, syncs once, and then finishes the items in queue order. A queued writer
# then waits for one sync instead of one per item ahead of it.
_RECV_SCATTER_DRAIN_ENV = "TRTLLM_KV_BOUNCE_RECV_SCATTER_DRAIN"


def recv_scatter_drain_from_env() -> bool:
    """Read the batched receive scatter switch. Unset or empty is off; any value other than 0 or 1
    raises ValueError, so a typo cannot silently keep the per-item scatter."""
    raw = os.environ.get(_RECV_SCATTER_DRAIN_ENV)
    if raw is None or raw.strip() == "":
        return False
    value = raw.strip()
    if value not in ("0", "1"):
        raise ValueError(f"{_RECV_SCATTER_DRAIN_ENV}={raw!r} is not supported; expected 0 or 1")
    return value == "1"


# Opt-in small-block receive scatter (default 0 keeps the batched copy's default block). Read on the
# generation (receiver) side and applied to the receive scatter only; the send-side gather keeps the
# default block. With it on, each program of the batched copy moves one 8 KiB slice of a fragment
# instead of the default 64 KiB slice, so a fragment of a few KiB does not leave most of its
# program's lanes masked. The copy, its stream, and the sync after it are unchanged.
_RECV_SCATTER_TUNED_ENV = "TRTLLM_KV_BOUNCE_RECV_SCATTER_TUNED"


def recv_scatter_tuned_from_env() -> bool:
    """Read the small-block receive scatter switch. Unset or empty is off; any value other than 0
    or 1 raises ValueError, so a typo cannot silently keep the default block."""
    raw = os.environ.get(_RECV_SCATTER_TUNED_ENV)
    if raw is None or raw.strip() == "":
        return False
    value = raw.strip()
    if value not in ("0", "1"):
        raise ValueError(f"{_RECV_SCATTER_TUNED_ENV}={raw!r} is not supported; expected 0 or 1")
    return value == "1"


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

# Opt-in override of that fraction (default 0.5). Read where the transport is sized, so set it on the
# worker that owns the regions. A larger share lets a fixed-size region fit when less memory is free
# at setup, at the cost of less free memory for what is allocated after the regions.
_MAX_FREE_FRACTION_ENV = "TRTLLM_KV_BOUNCE_MAX_FREE_FRACTION"
_MAX_FREE_FRACTION_LIMIT = 0.9


def max_free_fraction_from_env() -> float:
    """Read the share of free memory the regions may take. Unset or empty is the default 0.5; a
    value that is not a number in (0, 0.9] raises ValueError, so a typo cannot silently resize the
    regions."""
    raw = os.environ.get(_MAX_FREE_FRACTION_ENV)
    if raw is None or raw.strip() == "":
        return _HEADROOM_FRACTION
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{_MAX_FREE_FRACTION_ENV}={raw!r} is not a number") from None
    if not 0.0 < value <= _MAX_FREE_FRACTION_LIMIT:
        raise ValueError(
            f"{_MAX_FREE_FRACTION_ENV}={value} is outside (0, {_MAX_FREE_FRACTION_LIMIT}]"
        )
    return value


def fit_within_free(
    capacity_bytes: int,
    *,
    free_bytes: int,
    chunk_bytes: int,
    max_free_fraction: float = _HEADROOM_FRACTION,
    num_regions: int = 2,
) -> Optional[int]:
    """Clamp each region so all ``num_regions`` together stay within the allowed fraction of free
    memory, rounded down to a chunk. Returns None if not even one chunk fits."""
    budget_per_dir = (
        int(free_bytes * max_free_fraction) // num_regions // chunk_bytes
    ) * chunk_bytes
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
