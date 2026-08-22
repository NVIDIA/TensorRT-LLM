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
"""Configuration for the bounce_v2 pipeline.

Mirrors the C++ ``BounceConfig``
(cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceConfig.h)
with the hybrid design's defaults (design doc Section 4.1; the 32 MiB chunk
default matches the C++ bounce v2 per product decision): 32 MiB max
chunk, 2 GiB arena, 1 MiB granularity, 8 in-flight chunks per request. The
scatter-worker knob is dropped — the hybrid has no scatter worker threads
(scatter completions are reported through the C++ completion poller and
drained by the reactor on its 1 ms tick).

The C++ ``useZeroCopyArguments`` knob is intentionally absent: the copy-plan
staging choice belongs to the bound batched-copy op, which owns the
ExecPool-equivalent streams and plan buffers (design doc Section 3.1). If the
binding needs the knob, it will live in the binding's options, not in this
config.

Where the C++ transport silently CLAMPED nonsensical values at init time
(e.g. ``maxChunkSizeBytes`` clamped to the arena's usable capacity), this
config VALIDATES instead: :meth:`BounceV2Config.validate` raises
``ValueError``. A Python deployment should fail loudly on a config the C++
would have quietly reshaped; callers that want the clamp behavior can apply
it explicitly before calling ``validate()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["BounceV2Config"]

_U32_MAX = 0xFFFFFFFF


def _round_up_pow2(value: int) -> int:
    power = 1
    while power < value:
        power <<= 1
    return power


@dataclass
class BounceV2Config:
    """Knobs for the bounce_v2 control plane and arena."""

    enabled: bool = False
    #: Total staging arena size. 2 GiB = receiver grants (8 x 128 MiB per
    #: saturating flow) + the sender eager budget (half the arena); a power of
    #: two so the buddy usable capacity equals the configured size exactly.
    arena_size_bytes: int = 2 << 30
    #: Buddy order-0 block size. Stays at 1 MiB so small-chunk packing
    #: granularity is unchanged from bounce v1/v2 (R8: tight packing).
    arena_allocation_granularity_bytes: int = 1 << 20
    #: Per-chunk byte cap. 32 MiB (matching the C++ bounce v2 default) means
    #: ~1,490 chunks/s at 50 GB/s -- still well under the reactor's batched
    #: event ceiling -- but only ~5.4 ms of pipeline slack with 8 chunks in
    #: flight; raise max_inflight_chunks_per_request to widen the GIL-stall
    #: tolerance (32 restores the ~21.5 ms slack at the same 1 GiB per-flow
    #: staging). Must fit u32 (chunk sizes are 32-bit on the wire).
    max_chunk_size_bytes: int = 32 << 20
    #: Per-request in-flight allocation cap (window depth of the pipeline).
    max_inflight_chunks_per_request: int = 8
    #: Streams available to the bound batched-copy op.
    copy_stream_count: int = 8
    #: Admission gate: bounce only when a request has at least this many
    #: descriptors ...
    min_descriptor_count: int = 1024
    #: ... and its average descriptor is at most this large (large-desc
    #: requests do fine on the standard per-desc NIXL path).
    max_average_descriptor_size_bytes: int = 16 << 10
    #: Sender-side no-progress timeout; <= 0 DISABLES timeouts entirely
    #: (used by tests that intentionally wait).
    request_timeout_ms: int = 30000
    #: Launch a chunk's gather at submit() time, before the receiver's GRANT
    #: arrives, overlapping the WANT->GRANT round-trip with the gather kernel.
    enable_eager_gather: bool = True
    #: CI-only escape hatch: fall back to plain device memory for the arena.
    disable_fabric_memory: bool = False
    #: Receiver-side lease on granted regions. Derived (``None`` default) as
    #: 2 x ``request_timeout_ms``: a dead sender emits neither DATA nor a
    #: cancel — unobservable through the protocol alone — so a flow holding
    #: regions with no progress for this long is reclaimed. The lease must
    #: EXCEED the peers' request timeout (a live sender abandons + cancels
    #: first), hence the 2x; the derivation assumes both ends run the same
    #: request timeout. Set explicitly to override.
    receiver_flow_timeout_ms: Optional[int] = None
    #: How long a receiver-reclaimed, possibly-still-being-written region
    #: stays out of the arena before reuse. Derived (``None`` default) as
    #: ``request_timeout_ms``: a one-sided RDMA write cannot be aborted, so
    #: time is the only barrier against re-granting a region a gone peer's
    #: NIC may still be writing. Set explicitly to override.
    quarantine_ms: Optional[int] = None

    def __post_init__(self) -> None:
        # Derive the lease/quarantine windows from the one user-visible
        # timeout unless explicitly overridden; disabling the request timeout
        # (<= 0) disables both.
        if self.receiver_flow_timeout_ms is None:
            self.receiver_flow_timeout_ms = (
                2 * self.request_timeout_ms if self.request_timeout_ms > 0 else 0
            )
        if self.quarantine_ms is None:
            self.quarantine_ms = self.request_timeout_ms if self.request_timeout_ms > 0 else 0

    @property
    def arena_usable_capacity_bytes(self) -> int:
        """Usable buddy capacity: granularity rounded up to a power of two,
        arena rounded down to the largest ``granularity * 2**L`` that fits.

        0 when the arena cannot fit a single granule (``validate`` rejects
        that combination).
        """
        if self.arena_size_bytes <= 0 or self.arena_allocation_granularity_bytes <= 0:
            return 0
        granule = _round_up_pow2(self.arena_allocation_granularity_bytes)
        if self.arena_size_bytes < granule:
            return 0
        usable = granule
        while usable * 2 <= self.arena_size_bytes:
            usable *= 2
        return usable

    def validate(self) -> "BounceV2Config":
        """Check the configuration for nonsensical values.

        Raises:
            ValueError: On any inconsistent field combination (zero/negative
                sizes, a chunk that could never be granted, negative derived
                timeouts, ...). See the module docstring: these are strict
                validations, not the C++ init-time clamps.

        Returns:
            ``self`` (for chaining).
        """
        if self.arena_size_bytes <= 0:
            raise ValueError("bounce_v2: arena_size_bytes must be > 0")
        if self.arena_allocation_granularity_bytes <= 0:
            raise ValueError("bounce_v2: arena_allocation_granularity_bytes must be > 0")
        if self.max_chunk_size_bytes <= 0:
            raise ValueError("bounce_v2: max_chunk_size_bytes must be > 0")
        if self.max_chunk_size_bytes > _U32_MAX:
            raise ValueError(
                "bounce_v2: max_chunk_size_bytes must be <= 4 GiB - 1 "
                "(chunk sizes are 32-bit on the wire)"
            )
        usable = self.arena_usable_capacity_bytes
        if usable == 0:
            raise ValueError("bounce_v2: arena_size_bytes cannot fit a single allocation granule")
        if self.max_chunk_size_bytes > usable:
            raise ValueError(
                f"bounce_v2: max_chunk_size_bytes ({self.max_chunk_size_bytes}) exceeds the "
                f"arena's usable buddy capacity ({usable}); such a chunk could never be granted"
            )
        if self.max_inflight_chunks_per_request <= 0:
            raise ValueError("bounce_v2: max_inflight_chunks_per_request must be > 0")
        if self.copy_stream_count <= 0:
            raise ValueError("bounce_v2: copy_stream_count must be > 0")
        if self.min_descriptor_count < 0:
            raise ValueError("bounce_v2: min_descriptor_count must be >= 0")
        if self.max_average_descriptor_size_bytes < 0:
            raise ValueError("bounce_v2: max_average_descriptor_size_bytes must be >= 0")
        if self.receiver_flow_timeout_ms is None or self.receiver_flow_timeout_ms < 0:
            raise ValueError("bounce_v2: receiver_flow_timeout_ms must be >= 0")
        if self.quarantine_ms is None or self.quarantine_ms < 0:
            raise ValueError("bounce_v2: quarantine_ms must be >= 0")
        return self
