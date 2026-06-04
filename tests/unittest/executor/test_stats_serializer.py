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

from tensorrt_llm._torch.pyexecutor.kv_cache_stats import (
    KVCacheV2IterationStatsReport,
    KVCacheV2LifeCycleIterationStats,
    KVCacheV2PoolGroupIterationStats,
)
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
    primary_evictable=0,
    primary_peak_free=None,
    primary_peak_used=None,
    primary_peak_evictable=None,
    secondary_max=None,
    secondary_free=0,
    secondary_used=0,
    secondary_evictable=0,
    secondary_peak_free=None,
    secondary_peak_used=None,
    secondary_peak_evictable=None,
    reused=5,
    full_reused=4,
    partial_reused=1,
    missed=3,
    gen_alloc=2,
):
    """Create a mock KvCacheIterationStats nanobind object."""
    primary_free = primary_max - primary_used
    if primary_peak_free is None:
        primary_peak_free = primary_free
    if primary_peak_used is None:
        primary_peak_used = primary_used
    if primary_peak_evictable is None:
        primary_peak_evictable = primary_evictable
    if secondary_max is None:
        secondary_max = secondary_free + secondary_used
    if secondary_peak_free is None:
        secondary_peak_free = secondary_free
    if secondary_peak_used is None:
        secondary_peak_used = secondary_used
    if secondary_peak_evictable is None:
        secondary_peak_evictable = secondary_evictable
    s = SimpleNamespace(
        primary_max_num_blocks=primary_max,
        primary_free_num_blocks=primary_free,
        primary_used_num_blocks=primary_used,
        primary_evictable_num_blocks=primary_evictable,
        primary_peak_free_num_blocks=primary_peak_free,
        primary_peak_used_num_blocks=primary_peak_used,
        primary_peak_evictable_num_blocks=primary_peak_evictable,
        secondary_max_num_blocks=secondary_max,
        secondary_free_num_blocks=secondary_free,
        secondary_used_num_blocks=secondary_used,
        secondary_evictable_num_blocks=secondary_evictable,
        secondary_peak_free_num_blocks=secondary_peak_free,
        secondary_peak_used_num_blocks=secondary_peak_used,
        secondary_peak_evictable_num_blocks=secondary_peak_evictable,
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
        iter_host_dropped_blocks=0,
        iter_host_dropped_bytes=0,
    )
    return {window_size: s}


class _FakeStorageStatistics(SimpleNamespace):
    @property
    def unavailable(self):
        return self.total - self.available


class _FakePeakStorage:
    num_pool_groups = 2
    num_cache_levels = 2

    def __init__(self):
        self._levels = []
        self.primary_stats = [
            _FakeStorageStatistics(total=10, available=8, evictable=1),
            _FakeStorageStatistics(total=10, available=9, evictable=0),
        ]
        self.secondary_stats = [
            _FakeStorageStatistics(total=5, available=4, evictable=1),
            _FakeStorageStatistics(total=5, available=5, evictable=0),
        ]

    def get_statistics(self, level):
        if int(level) == 0:
            return self.primary_stats
        return self.secondary_stats

    def destroy(self):
        pass


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
        assert ws_stats["primaryPeakFreeNumBlocks"] == 10
        assert ws_stats["primaryPeakUsedNumBlocks"] == 10
        assert ws_stats["primaryPeakEvictableNumBlocks"] == 0
        assert ws_stats["secondaryPeakFreeNumBlocks"] == 0
        assert ws_stats["secondaryPeakUsedNumBlocks"] == 0
        assert ws_stats["secondaryPeakEvictableNumBlocks"] == 0
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

    def test_serializer_with_v2_pool_group_stats(self):
        """KV cache manager V2 stats should include pool group breakdown."""
        iter_stats = _make_mock_iteration_stats()
        by_window = _make_mock_kv_iter_stats(
            window_size=16,
            primary_used=10,
            primary_max=20,
            primary_evictable=2,
            primary_peak_free=12,
            primary_peak_used=15,
            primary_peak_evictable=4,
            secondary_max=8,
            secondary_free=5,
            secondary_used=3,
            secondary_evictable=1,
            secondary_peak_free=6,
            secondary_peak_used=4,
            secondary_peak_evictable=2,
            reused=5,
            full_reused=4,
            partial_reused=1,
            missed=3,
            gen_alloc=2,
        )
        pool_group_stats = _make_mock_kv_iter_stats(
            window_size=16,
            primary_used=10,
            primary_max=20,
            primary_evictable=2,
            primary_peak_free=12,
            primary_peak_used=15,
            primary_peak_evictable=4,
            secondary_max=8,
            secondary_free=5,
            secondary_used=3,
            secondary_evictable=1,
            secondary_peak_free=6,
            secondary_peak_used=4,
            secondary_peak_evictable=2,
            reused=0,
            full_reused=0,
            partial_reused=0,
            missed=0,
            gen_alloc=2,
        )[16]
        life_cycle_stats = _make_mock_kv_iter_stats(
            window_size=16,
            primary_used=0,
            primary_max=0,
            reused=5,
            full_reused=4,
            partial_reused=1,
            missed=3,
            gen_alloc=0,
        )[16]
        kv_iter = KVCacheV2IterationStatsReport(
            by_window,
            {
                7: KVCacheV2PoolGroupIterationStats(
                    pool_group_id=7,
                    slot_size=(2 << 20,),
                    window_sizes=(16, 64),
                    stats=pool_group_stats,
                )
            },
            {
                3: KVCacheV2LifeCycleIterationStats(
                    life_cycle_id=3,
                    pool_group_id=7,
                    window_size=16,
                    kind="attention",
                    stats=life_cycle_stats,
                )
            },
        )

        result = BaseWorker._stats_serializer((iter_stats, None, kv_iter))
        d = json.loads(result)

        assert d["kvCacheIterationStats"]["16"]["iterReusedBlocks"] == 5
        assert d["kvCacheIterationStats"]["16"]["primaryPeakFreeNumBlocks"] == 12
        assert d["kvCacheIterationStats"]["16"]["primaryPeakUsedNumBlocks"] == 15
        assert d["kvCacheIterationStats"]["16"]["primaryPeakEvictableNumBlocks"] == 4
        assert d["kvCacheIterationStats"]["16"]["secondaryPeakFreeNumBlocks"] == 6
        assert d["kvCacheIterationStats"]["16"]["secondaryPeakUsedNumBlocks"] == 4
        assert d["kvCacheIterationStats"]["16"]["secondaryPeakEvictableNumBlocks"] == 2
        assert "kvCacheIterationStatsByPoolGroup" in d
        pool_group = d["kvCacheIterationStatsByPoolGroup"]["7"]
        assert pool_group["poolGroupId"] == 7
        assert pool_group["slotSize"] == [2 << 20]
        assert pool_group["windowSizes"] == [16, 64]
        assert pool_group["primaryPeakFreeNumBlocks"] == 12
        assert pool_group["primaryPeakUsedNumBlocks"] == 15
        assert pool_group["primaryPeakEvictableNumBlocks"] == 4
        assert pool_group["secondaryPeakFreeNumBlocks"] == 6
        assert pool_group["secondaryPeakUsedNumBlocks"] == 4
        assert pool_group["secondaryPeakEvictableNumBlocks"] == 2
        assert pool_group["iterGenAllocBlocks"] == 2
        assert "iterReusedBlocks" not in pool_group
        assert "iterMissedBlocks" not in pool_group
        assert "iterCacheHitRate" not in pool_group
        assert "kvCacheIterationStatsByLifecycle" in d
        life_cycle = d["kvCacheIterationStatsByLifecycle"]["3"]
        assert life_cycle["lifeCycleId"] == 3
        assert life_cycle["poolGroupId"] == 7
        assert life_cycle["windowSize"] == 16
        assert life_cycle["kind"] == "attention"
        assert life_cycle["iterReusedBlocks"] == 5
        assert life_cycle["iterMissedBlocks"] == 3
        assert "iterGenAllocBlocks" not in life_cycle

    def test_v2_peak_block_stats_reset_tracks_interval_peak(self):
        """Peak block stats should cover the interval since the previous reset."""
        from tensorrt_llm.runtime.kv_cache_manager_v2._common import GPU_LEVEL, CacheLevel
        from tensorrt_llm.runtime.kv_cache_manager_v2._core._kv_cache_manager import KVCacheManager

        storage = _FakePeakStorage()
        manager = object.__new__(KVCacheManager)
        manager._storage = storage
        manager._radix_tree = SimpleNamespace(clear=lambda: [])
        manager._reset_iteration_peak_num_blocks()

        # Some gauges rise above the reset baseline, then fall before drain.
        storage.primary_stats[0].available = 5  # primary used = 5
        storage.primary_stats[0].evictable = 3
        storage.primary_stats[1].available = 6  # primary used = 4
        storage.primary_stats[1].evictable = 4
        storage.secondary_stats[0].available = 2  # secondary used = 3
        storage.secondary_stats[0].evictable = 2
        manager._update_iteration_peak_num_blocks()
        storage.primary_stats[0].available = 7  # primary used = 3
        storage.primary_stats[0].evictable = 1
        storage.secondary_stats[0].available = 4  # secondary used = 1
        storage.secondary_stats[0].evictable = 1

        peak_by_cache_level = manager.get_and_reset_iteration_peak_block_stats()
        primary_peak = peak_by_cache_level[GPU_LEVEL]
        secondary_peak = peak_by_cache_level[CacheLevel(1)]
        assert list(primary_peak.free) == [8, 9]
        assert list(primary_peak.used) == [5, 4]
        assert list(primary_peak.evictable) == [3, 4]
        assert list(secondary_peak.free) == [4, 5]
        assert list(secondary_peak.used) == [3, 0]
        assert list(secondary_peak.evictable) == [2, 0]

        # The next interval starts from current usage, not zero.
        peak_by_cache_level = manager.get_and_reset_iteration_peak_block_stats()
        primary_peak = peak_by_cache_level[GPU_LEVEL]
        secondary_peak = peak_by_cache_level[CacheLevel(1)]
        assert list(primary_peak.free) == [7, 6]
        assert list(primary_peak.used) == [3, 4]
        assert list(primary_peak.evictable) == [1, 4]
        assert list(secondary_peak.free) == [4, 5]
        assert list(secondary_peak.used) == [1, 0]
        assert list(secondary_peak.evictable) == [1, 0]
