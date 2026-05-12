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
"""Tests for _stats_serializer with kvCacheIterationStats injection."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm.executor.base_worker import BaseWorker


def _make_mock_iteration_stats(kv_cache_stats_json=None):
    """Create a mock IterationStats object with to_json_str()."""
    base = {
        "iter": 1,
        "iterLatencyMS": 10.5,
        "gpuMemUsage": 1024,
        "cpuMemUsage": 0,
        "pinnedMemUsage": 0,
    }
    if kv_cache_stats_json is not None:
        base["kvCacheStats"] = kv_cache_stats_json

    mock = MagicMock()
    mock.to_json_str.return_value = json.dumps(base)
    return mock


def _make_mock_kv_iter_stats(
    window_size=16,
    primary_used=10,
    primary_max=20,
    reused=5,
    full_reused=4,
    partial_reused=1,
    missed=3,
    gen_alloc=2,
):
    """Create a mock KvCacheIterationStats nanobind object."""
    s = SimpleNamespace(
        primary_max_num_blocks=primary_max,
        primary_free_num_blocks=primary_max - primary_used,
        primary_used_num_blocks=primary_used,
        secondary_max_num_blocks=0,
        secondary_free_num_blocks=0,
        secondary_used_num_blocks=0,
        iter_alloc_total_blocks=reused + missed,
        iter_alloc_new_blocks=missed,
        iter_reused_blocks=reused,
        iter_full_reused_blocks=full_reused,
        iter_partial_reused_blocks=partial_reused,
        iter_missed_blocks=missed,
        iter_cache_hit_rate=reused / (reused + missed) if (reused + missed) > 0 else 0.0,
        iter_gen_alloc_blocks=gen_alloc,
        iter_onboard_blocks=1,
        iter_onboard_bytes=4096,
        iter_offload_blocks=0,
        iter_offload_bytes=0,
        iter_intra_device_copy_blocks=2,
        iter_intra_device_copy_bytes=8192,
    )
    return {window_size: s}


class TestStatsSerializer:
    def test_serializer_without_kv_iter_stats(self):
        """Legacy 2-tuple and 3-tuple with None should produce same output."""
        iter_stats = _make_mock_iteration_stats()

        # 3-tuple with None kv_iter_stats
        result = BaseWorker._stats_serializer((iter_stats, None, None))
        d = json.loads(result)
        assert "iter" in d
        assert "kvCacheIterationStats" not in d

    def test_serializer_with_kv_iter_stats(self):
        """KvCacheIterationStats should appear when provided."""
        iter_stats = _make_mock_iteration_stats(
            kv_cache_stats_json={"maxNumBlocks": 20, "usedNumBlocks": 10}
        )
        kv_iter = _make_mock_kv_iter_stats(
            window_size=16,
            primary_used=10,
            primary_max=20,
            reused=5,
            full_reused=4,
            partial_reused=1,
            missed=3,
            gen_alloc=2,
        )

        result = BaseWorker._stats_serializer((iter_stats, None, kv_iter))
        d = json.loads(result)

        # Existing kvCacheStats should still be present
        assert "kvCacheStats" in d

        # New kvCacheIterationStats should be present
        assert "kvCacheIterationStats" in d
        iter_kv = d["kvCacheIterationStats"]
        assert "16" in iter_kv  # window size key as string

        ws_stats = iter_kv["16"]
        assert ws_stats["primaryMaxNumBlocks"] == 20
        assert ws_stats["primaryUsedNumBlocks"] == 10
        assert ws_stats["primaryFreeNumBlocks"] == 10
        assert ws_stats["iterReusedBlocks"] == 5
        assert ws_stats["iterFullReusedBlocks"] == 4
        assert ws_stats["iterPartialReusedBlocks"] == 1
        assert ws_stats["iterMissedBlocks"] == 3
        assert ws_stats["iterGenAllocBlocks"] == 2
        assert ws_stats["iterOnboardBlocks"] == 1
        assert ws_stats["iterOnboardBytes"] == 4096
        assert ws_stats["iterOffloadBlocks"] == 0
        assert ws_stats["iterOffloadBytes"] == 0
        assert ws_stats["iterIntraDeviceCopyBlocks"] == 2
        assert ws_stats["iterIntraDeviceCopyBytes"] == 8192
        assert ws_stats["iterCacheHitRate"] == pytest.approx(5 / 8)

    def test_serializer_multiple_window_sizes(self):
        """Multiple window sizes should all appear in output."""
        iter_stats = _make_mock_iteration_stats()
        kv_iter = _make_mock_kv_iter_stats(
            window_size=16,
            primary_used=5,
            primary_max=10,
            reused=2,
            full_reused=2,
            partial_reused=0,
            missed=1,
            gen_alloc=0,
        )
        # Add a second window size
        kv_iter[64] = _make_mock_kv_iter_stats(
            window_size=64,
            primary_used=8,
            primary_max=16,
            reused=3,
            full_reused=1,
            partial_reused=2,
            missed=2,
            gen_alloc=1,
        )[64]

        result = BaseWorker._stats_serializer((iter_stats, None, kv_iter))
        d = json.loads(result)

        iter_kv = d["kvCacheIterationStats"]
        assert "16" in iter_kv
        assert "64" in iter_kv
        assert iter_kv["16"]["primaryMaxNumBlocks"] == 10
        assert iter_kv["64"]["primaryMaxNumBlocks"] == 16

    def test_serializer_with_request_stats(self):
        """Request stats and kv iter stats should coexist."""
        iter_stats = _make_mock_iteration_stats()
        kv_iter = _make_mock_kv_iter_stats()

        req_stat = MagicMock()
        req_stat.to_json_str.return_value = json.dumps({"id": 42})

        result = BaseWorker._stats_serializer((iter_stats, [req_stat], kv_iter))
        d = json.loads(result)

        assert "requestStats" in d
        assert len(d["requestStats"]) == 1
        assert d["requestStats"][0]["id"] == 42
        assert "kvCacheIterationStats" in d

    def test_serializer_none_on_off_interval(self):
        """When kv_iter_stats is None (off-interval), field should be absent."""
        iter_stats = _make_mock_iteration_stats()

        result = BaseWorker._stats_serializer((iter_stats, None, None))
        d = json.loads(result)
        assert "kvCacheIterationStats" not in d

    def test_serializer_legacy_2_tuple(self):
        """Legacy 2-tuple without third element should work."""
        iter_stats = _make_mock_iteration_stats()

        result = BaseWorker._stats_serializer((iter_stats, None))
        d = json.loads(result)
        assert "kvCacheIterationStats" not in d

    def test_serializer_attention_dp_rank_tag(self):
        """ADP 4-tuple should carry the supplied attention-DP rank."""
        iter_stats = _make_mock_iteration_stats()

        result = BaseWorker._stats_serializer((iter_stats, None, None, 3))
        d = json.loads(result)
        assert d["attentionDpRank"] == 3

    def test_serializer_none_attention_dp_rank_defaults_zero(self):
        """Fixed-shape 4-tuples use None for non-ADP and serialize as rank 0."""
        iter_stats = _make_mock_iteration_stats()

        result = BaseWorker._stats_serializer((iter_stats, None, None, None))
        d = json.loads(result)
        assert d["attentionDpRank"] == 0
