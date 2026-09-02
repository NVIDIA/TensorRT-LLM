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
"""Request-level storage-tier attribution of cached prompt tokens.

The KV cache managers record ordered ``(tier, num_tokens)`` runs over the matched
prefix; ``split_cached_tokens_by_tier`` turns them into the per-request
``cached_tokens_by_tier`` dict for whatever prefix length the engine ends up
skipping. The sum of the dict must always equal the aggregate ``cached_tokens``.
"""

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache_stats import (
    KV_CACHE_ITERATION_STATS_REUSE_KEYS,
    split_cached_tokens_by_tier,
)
from tensorrt_llm.metrics.enums import CACHE_TIER_LABELS

pytestmark = pytest.mark.cpu_only

GPU, HOST, DISK, REMOTE, NONE = range(5)


def test_tier_labels_are_stable_wire_values():
    # The indices are shared with the C++ KvCacheTier enum and the manager segments.
    assert CACHE_TIER_LABELS == ("gpu", "host", "disk", "remote", "none")


def test_full_prefix_split():
    segments = [(GPU, 64), (HOST, 64), (DISK, 32)]
    assert split_cached_tokens_by_tier(segments, 160) == {"gpu": 64, "host": 64, "disk": 32}


def test_prefix_shorter_than_matched_trims_from_the_end():
    """The engine may skip fewer tokens than the manager matched (block alignment,
    last prompt token); the trailing tokens are dropped, never re-attributed."""
    segments = [(GPU, 64), (HOST, 64), (DISK, 32)]
    assert split_cached_tokens_by_tier(segments, 100) == {"gpu": 64, "host": 36}
    assert split_cached_tokens_by_tier(segments, 64) == {"gpu": 64}
    assert split_cached_tokens_by_tier(segments, 1) == {"gpu": 1}


def test_repeated_tiers_are_merged():
    assert split_cached_tokens_by_tier([(HOST, 16), (GPU, 8), (HOST, 16)], 40) == {
        "host": 32,
        "gpu": 8,
    }


def test_no_attribution_or_no_cached_tokens_gives_empty_dict():
    assert split_cached_tokens_by_tier(None, 10) == {}
    assert split_cached_tokens_by_tier([], 10) == {}
    assert split_cached_tokens_by_tier([(GPU, 8)], 0) == {}


def test_remote_and_none_tiers_round_trip():
    segments = [(REMOTE, 128), (NONE, 16)]
    assert split_cached_tokens_by_tier(segments, 144) == {"remote": 128, "none": 16}


def test_sum_invariant_for_every_prefix_length():
    segments = [(GPU, 5), (NONE, 3), (HOST, 7), (DISK, 2), (REMOTE, 11)]
    total = sum(n for _, n in segments)
    for length in range(total + 1):
        split = split_cached_tokens_by_tier(segments, length)
        assert sum(split.values()) == length
        assert set(split) <= set(CACHE_TIER_LABELS)


def test_iteration_stats_reuse_keys_include_tiers():
    for key in (
        "iterReusedBlocksGpu",
        "iterReusedBlocksHost",
        "iterReusedBlocksDisk",
        "iterReusedBlocksRemote",
    ):
        assert key in KV_CACHE_ITERATION_STATS_REUSE_KEYS
