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

"""Unit tests for KV cache storage-tier hit attribution in _setup_for_reuse.

Validates that _setup_for_reuse correctly partitions reused tokens into gpu,
host, and disk tiers across single and multiple attention lifecycles, handles
partial blocks, and reconciles integer division residue.
"""

from types import SimpleNamespace
from typing import Optional

from tensorrt_llm.runtime.kv_cache_manager_v2._common import (
    CacheLevel,
    CacheTier,
    GPU_LEVEL,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._core._kv_cache import _KVCache
from tensorrt_llm.runtime.kv_cache_manager_v2._life_cycle_registry import (
    AttnLifeCycle,
    LifeCycleId,
)

ATTN_LC0 = LifeCycleId(0)
ATTN_LC1 = LifeCycleId(1)
HOST_LEVEL = CacheLevel(GPU_LEVEL + 1)
DISK_LEVEL = CacheLevel(GPU_LEVEL + 2)


class _MockPage:
    """Mock BlockPage with hold semantics and cache level."""

    def __init__(self, cache_level: CacheLevel) -> None:
        self.cache_level = cache_level

    def hold(self) -> "_MockPage":
        return self


class _MockBlock:
    """Mock block exposing pages by lifecycle and token sequence."""

    def __init__(self, pages_by_lc: dict[LifeCycleId, _MockPage], tokens: list[int]) -> None:
        self.pages_by_lc = pages_by_lc
        self.tokens = tokens

    def get_page(self, lc_id: LifeCycleId) -> Optional[_MockPage]:
        return self.pages_by_lc.get(lc_id)


class _MockLifeCycles:
    """Mock life cycle registry."""

    def __init__(self, lcs: dict[LifeCycleId, AttnLifeCycle]) -> None:
        self._lcs = lcs
        self.ssm_life_cycle_id = None
        self.size = len(lcs)

    def items(self):
        return self._lcs.items()

    def __getitem__(self, idx: LifeCycleId) -> AttnLifeCycle:
        return self._lcs[idx]


def _create_test_cache(tokens_per_block: int, lcs: dict[LifeCycleId, AttnLifeCycle]):
    """Create a duck-typed _KVCache stand-in for testing _setup_for_reuse."""
    life_cycles = _MockLifeCycles(lcs)
    manager = SimpleNamespace(
        tokens_per_block=tokens_per_block,
        cache_tier_list=[CacheTier.GPU_MEM, CacheTier.HOST_MEM, CacheTier.DISK],
        _life_cycles=life_cycles,
        mark_stats_dirty=lambda _id: None,
    )
    cache = SimpleNamespace(
        id=1,
        manager=manager,
        beam_width=1,
        _base_page_indices=[[]],
        _should_record_manager_stats=lambda: False,
        _should_record_request_stats=lambda: False,
    )
    cache._block = lambda ordinal, beam_idx: cache._blocks[ordinal].pages[beam_idx]
    cache._get_matched_tokens = lambda m: _KVCache._get_matched_tokens(cache, m)
    return cache


def test_setup_for_reuse_mixed_tiers_single_lifecycle() -> None:
    """Mixed tiers across blocks with a partial final block."""
    tokens_per_block = 64
    lcs = {ATTN_LC0: AttnLifeCycle(None, 0)}
    cache = _create_test_cache(tokens_per_block, lcs)

    # 3 blocks: block0 = GPU (64 tokens), block1 = Host (64 tokens), block2 = Disk (32 tokens)
    b0 = _MockBlock({ATTN_LC0: _MockPage(GPU_LEVEL)}, list(range(64)))
    b1 = _MockBlock({ATTN_LC0: _MockPage(HOST_LEVEL)}, list(range(64)))
    b2 = _MockBlock({ATTN_LC0: _MockPage(DISK_LEVEL)}, list(range(32)))

    match = SimpleNamespace(
        blocks=[b0, b1, b2],
        num_tokens=160,
    )

    _setup = getattr(_KVCache, "_setup_for_reuse").__get__(cache)
    _setup(match)

    assert cache._reused_tokens_by_tier == {"gpu": 64, "host": 64, "disk": 32}
    assert sum(cache._reused_tokens_by_tier.values()) == 160


def test_setup_for_reuse_multi_lifecycle_with_residue() -> None:
    """Multiple attention lifecycles with split page tiers and odd token residue."""
    tokens_per_block = 64
    lcs = {
        ATTN_LC0: AttnLifeCycle(None, 0),
        ATTN_LC1: AttnLifeCycle(None, 0),
    }
    cache = _create_test_cache(tokens_per_block, lcs)

    # Block 0: LC0 on GPU, LC1 on Host (64 tokens -> 32 GPU, 32 Host)
    # Block 1 (partial, 15 tokens): LC0 on Host, LC1 on Disk (15 // 2 = 7 each, 1 residue to GPU)
    b0 = _MockBlock(
        {
            ATTN_LC0: _MockPage(GPU_LEVEL),
            ATTN_LC1: _MockPage(HOST_LEVEL),
        },
        list(range(64)),
    )
    b1 = _MockBlock(
        {
            ATTN_LC0: _MockPage(HOST_LEVEL),
            ATTN_LC1: _MockPage(DISK_LEVEL),
        },
        list(range(15)),
    )

    match = SimpleNamespace(
        blocks=[b0, b1],
        num_tokens=79,
    )

    _setup = getattr(_KVCache, "_setup_for_reuse").__get__(cache)
    _setup(match)

    # Expected:
    # Block 0: 32 GPU, 32 Host
    # Block 1: 7 Host, 7 Disk
    # Residue: 79 - (32 + 39 + 7) = 1 -> assigned to GPU
    # Total: GPU = 33, Host = 39, Disk = 7
    assert cache._reused_tokens_by_tier["gpu"] == 33
    assert cache._reused_tokens_by_tier["host"] == 39
    assert cache._reused_tokens_by_tier["disk"] == 7
    assert sum(cache._reused_tokens_by_tier.values()) == 79


def test_setup_for_reuse_sliding_window_stale_range() -> None:
    """Sliding window attention where older blocks fall in the stale range."""
    tokens_per_block = 64
    # Window of 64 tokens means only the latest 1 block is kept active; older blocks are stale
    lcs = {ATTN_LC0: AttnLifeCycle(window_size=64, num_sink_blocks=0)}
    cache = _create_test_cache(tokens_per_block, lcs)

    # Block 0 on Host (stale, outside window), Block 1 on Disk (active, within window)
    b0 = _MockBlock({ATTN_LC0: _MockPage(HOST_LEVEL)}, list(range(64)))
    b1 = _MockBlock({ATTN_LC0: _MockPage(DISK_LEVEL)}, list(range(64)))

    match = SimpleNamespace(
        blocks=[b0, b1],
        num_tokens=128,
    )

    _setup = getattr(_KVCache, "_setup_for_reuse").__get__(cache)
    _setup(match)

    # Both blocks are matched and reused:
    # Block 0 is on Host (64 tokens) and Block 1 is on Disk (64 tokens).
    # Even though Block 0 is stale (out of window) and not copied into active pages,
    # its reused tokens are correctly attributed to the Host tier instead of falling into GPU.
    assert cache._reused_tokens_by_tier["host"] == 64
    assert cache._reused_tokens_by_tier["disk"] == 64
    assert cache._reused_tokens_by_tier["gpu"] == 0
    assert sum(cache._reused_tokens_by_tier.values()) == 128

    # Verify that only the non-stale block (block 1) had its page holder loaded into active blocks
    beam_idx = 0
    assert cache._blocks[0].pages[beam_idx][ATTN_LC0] is None
    assert cache._blocks[1].pages[beam_idx][ATTN_LC0] is not None

