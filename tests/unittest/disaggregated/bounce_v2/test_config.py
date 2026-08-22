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
"""Config tests: the semantic subset of bounceConfigTest.cpp.

The C++ suite is mostly env-var parsing, which the Python config does not
have; ported here are the semantic bits: defaults, the derivation of the
receiver flow lease / quarantine windows from the request timeout, and the
validate() rejections (which replace the C++ silent init-time clamps).
"""

from __future__ import annotations

import pytest
from conftest import load_bounce_v2

_b = load_bounce_v2()
BounceV2Config = _b.BounceV2Config


def test_defaults() -> None:
    cfg = BounceV2Config()
    assert cfg.enabled is False
    assert cfg.arena_size_bytes == 2 << 30
    assert cfg.arena_allocation_granularity_bytes == 1 << 20
    assert cfg.max_chunk_size_bytes == 32 << 20
    assert cfg.max_inflight_chunks_per_request == 8
    assert cfg.copy_stream_count == 8
    assert cfg.min_descriptor_count == 1024
    assert cfg.max_average_descriptor_size_bytes == 16 << 10
    assert cfg.request_timeout_ms == 30_000
    assert cfg.enable_eager_gather is True
    assert cfg.disable_fabric_memory is False
    # Derived: lease = 2x request timeout, quarantine = 1x.
    assert cfg.receiver_flow_timeout_ms == 60_000
    assert cfg.quarantine_ms == 30_000
    cfg.validate()  # the defaults are self-consistent


# C++: the derived-timeout part of BounceConfig.DescriptiveNamesParse
# (lease/quarantine derive from the request timeout, not independent knobs).
def test_lease_and_quarantine_derive_from_request_timeout() -> None:
    cfg = BounceV2Config(request_timeout_ms=900)
    assert cfg.receiver_flow_timeout_ms == 1800
    assert cfg.quarantine_ms == 900


@pytest.mark.parametrize("timeout_ms", [0, -1, -30_000])
def test_disabled_request_timeout_disables_derived_windows(timeout_ms: int) -> None:
    # request_timeout_ms <= 0 disables timeouts entirely -> both derived
    # windows are 0 (and still pass validation).
    cfg = BounceV2Config(request_timeout_ms=timeout_ms)
    assert cfg.receiver_flow_timeout_ms == 0
    assert cfg.quarantine_ms == 0
    cfg.validate()


def test_explicit_lease_and_quarantine_override_derivation() -> None:
    cfg = BounceV2Config(
        request_timeout_ms=30_000,
        receiver_flow_timeout_ms=5_000,
        quarantine_ms=7_000,
    )
    assert cfg.receiver_flow_timeout_ms == 5_000
    assert cfg.quarantine_ms == 7_000
    cfg.validate()


def test_arena_usable_capacity_rounds_like_the_buddy() -> None:
    # Granularity rounds UP to a power of two; the arena rounds DOWN to the
    # largest granularity * 2**L that fits (mirrors BuddyAllocator sizing).
    cfg = BounceV2Config(
        arena_size_bytes=1000,
        arena_allocation_granularity_bytes=256,
        max_chunk_size_bytes=256,
    )
    assert cfg.arena_usable_capacity_bytes == 512
    cfg.validate()

    cfg2 = BounceV2Config(
        arena_size_bytes=512,
        arena_allocation_granularity_bytes=100,  # rounds up to 128
        max_chunk_size_bytes=512,
    )
    assert cfg2.arena_usable_capacity_bytes == 512
    cfg2.validate()


def test_validate_returns_self_for_chaining() -> None:
    cfg = BounceV2Config()
    assert cfg.validate() is cfg


# C++ silently clamped nonsense at init; the Python config REJECTS instead.
@pytest.mark.parametrize(
    "kwargs",
    [
        {"arena_size_bytes": 0},
        {"arena_size_bytes": -1},
        {"arena_allocation_granularity_bytes": 0},
        {"arena_allocation_granularity_bytes": -1},
        {"max_chunk_size_bytes": 0},
        {"max_chunk_size_bytes": -1},
        # Chunk sizes are 32-bit on the wire: > 4 GiB - 1 is rejected.
        {"arena_size_bytes": 16 << 30, "max_chunk_size_bytes": 1 << 32},
        # A chunk larger than the arena's usable capacity could never be
        # granted (the C++ clamped it down instead).
        {"arena_size_bytes": 64 << 20, "max_chunk_size_bytes": 128 << 20},
        {"max_inflight_chunks_per_request": 0},
        {"max_inflight_chunks_per_request": -1},
        {"copy_stream_count": 0},
        {"copy_stream_count": -1},
        {"min_descriptor_count": -1},
        {"max_average_descriptor_size_bytes": -1},
        {"receiver_flow_timeout_ms": -1},
        {"quarantine_ms": -1},
        # An arena smaller than one (rounded) granule fits nothing.
        {
            "arena_size_bytes": 100,
            "arena_allocation_granularity_bytes": 256,
            "max_chunk_size_bytes": 64,
        },
    ],
    ids=[
        "arena-zero",
        "arena-neg",
        "granularity-zero",
        "granularity-neg",
        "chunk-zero",
        "chunk-neg",
        "chunk-above-u32",
        "chunk-above-usable",
        "inflight-zero",
        "inflight-neg",
        "streams-zero",
        "streams-neg",
        "min-desc-neg",
        "avg-desc-neg",
        "lease-neg",
        "quarantine-neg",
        "arena-below-granule",
    ],
)
def test_validate_rejects_nonsense(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        BounceV2Config(**kwargs).validate()


def test_max_chunk_exactly_u32_max_and_usable_is_accepted() -> None:
    # Exactly 4 GiB - 1 is allowed on the wire, provided the arena's usable
    # capacity covers it (usable must be a power of two >= the chunk).
    cfg = BounceV2Config(
        arena_size_bytes=8 << 30,
        max_chunk_size_bytes=(1 << 32) - 1,
    )
    cfg.validate()
    # And a chunk equal to the usable capacity is the largest grantable one.
    cfg2 = BounceV2Config(
        arena_size_bytes=256 << 20,
        max_chunk_size_bytes=256 << 20,
    )
    assert cfg2.arena_usable_capacity_bytes == 256 << 20
    cfg2.validate()
