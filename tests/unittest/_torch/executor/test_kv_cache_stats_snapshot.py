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

import pickle

from tensorrt_llm._torch.pyexecutor.kv_cache_stats import (
    KVCachePoolStatsSnapshot,
    KVCacheStatsMetadata,
    KVCacheV2IterationStatsSnapshot,
    append_kv_cache_iteration_stats,
)


def _snapshot() -> KVCacheV2IterationStatsSnapshot:
    return KVCacheV2IterationStatsSnapshot(
        metadata=KVCacheStatsMetadata(
            life_cycles=(
                (0, 0, 32, "attention"),
                (1, 0, 64, "attention"),
                (2, 1, None, "ssm"),
            ),
            cold_pool_groups=((0, (0, 1, 2)),),
            cache_level_tiers=("gpu", "host", "disk"),
        ),
        pools_by_level=(
            (
                KVCachePoolStatsSnapshot(20, 8, 3, 18, 14, 7, (4096,)),
                KVCachePoolStatsSnapshot(10, 6, 1, 9, 5, 2, (2048,)),
            ),
            (KVCachePoolStatsSnapshot(7, 2, 1, 5, 6, 2, (4096,)),),
            (KVCachePoolStatsSnapshot(11, 5, 3, 9, 7, 4, (4096,)),),
        ),
        deltas=(
            (
                0,
                (
                    ("iter_alloc_new_blocks", 2),
                    ("iter_reused_blocks", 3),
                    ("iter_missed_blocks", 1),
                ),
            ),
            (
                1,
                (
                    ("iter_alloc_new_blocks", 4),
                    ("iter_reused_blocks", 2),
                    ("iter_missed_blocks", 2),
                ),
            ),
            (2, (("iter_onboard_blocks", 1), ("iter_onboard_bytes", 2048))),
        ),
        ssm_deltas=((2, (5, 3, 2, 24, 16, 2, 1)),),
        reused_blocks_by_level=((0, (2, 0, 0), (0, 1, 0)),),
        suspended_requests=2,
        resumed_requests=1,
        disk_prefetch_blocks=7,
        cached_tokens_by_level=(12, 8, 0),
    )


def test_shared_pool_snapshot_preserves_distinct_reporting_scopes() -> None:
    report = _snapshot().build_report()

    assert report.by_window_size[32].primary_max_num_blocks == 0
    assert report.by_window_size[64].primary_max_num_blocks == 0
    assert report.by_window_size[32].iter_cache_hit_rate == 0.75
    assert report.by_window_size[64].iter_cache_hit_rate == 0.5
    hot_pool = report.by_pool_group[0]
    assert hot_pool.window_sizes == (32, 64)
    assert hot_pool.stats.primary_used_num_blocks == 12
    assert hot_pool.stats.primary_peak_used_num_blocks == 14
    assert hot_pool.stats.iter_alloc_new_blocks == 6
    assert hot_pool.stats.iter_reused_blocks == 0
    assert hot_pool.stats.secondary_max_num_blocks == 0
    cold_pool = report.by_cold_pool_group[0]
    assert cold_pool.window_sizes == (32, 64)
    assert cold_pool.stats.secondary_max_num_blocks == 18
    assert cold_pool.stats.secondary_used_num_blocks == 11
    assert cold_pool.stats.secondary_peak_used_num_blocks == 13
    assert report.by_life_cycle[0].full_reused_blocks_by_level == [2, 0, 0]
    assert report.by_life_cycle[0].partial_reused_blocks_by_level == [0, 1, 0]
    assert report.by_life_cycle[2].snapshot_stats.iter_snapshot_hit_rate == 0.6
    assert report.by_pool_group[1].stats.iter_onboard_bytes == 2048


def test_snapshot_pickle_roundtrip_preserves_report_and_does_not_share_output() -> None:
    snapshot = _snapshot()
    expected = {}
    append_kv_cache_iteration_stats(expected, snapshot)
    received = pickle.loads(pickle.dumps(snapshot))

    first_report = received.build_report()
    first_report.by_pool_group[0].stats.primary_used_num_blocks = 999
    first_report.by_life_cycle[0].full_reused_blocks_by_level[0] = 999
    first_report.cached_tokens_by_level[0] = 999

    actual = {}
    append_kv_cache_iteration_stats(actual, received)
    assert actual == expected
    assert actual["iterSuspendedRequests"] == 2
    assert actual["iterResumedRequests"] == 1
    assert actual["iterCachedTokensByLevel"] == [12, 8, 0]


def test_ssm_only_snapshot_reports_pool_without_a_window() -> None:
    snapshot = KVCacheV2IterationStatsSnapshot(
        metadata=KVCacheStatsMetadata(((0, 0, None, "ssm"),), (), ("gpu",)),
        pools_by_level=((KVCachePoolStatsSnapshot(10, 6, 1, 8, 5, 2, (2048,)),),),
        deltas=(),
        ssm_deltas=((0, (2, 1, 1, 32, 16, 1, 0)),),
        reused_blocks_by_level=(),
        suspended_requests=0,
        resumed_requests=0,
        disk_prefetch_blocks=0,
        cached_tokens_by_level=(),
    )
    report = snapshot.build_report()
    assert report.by_window_size == {}
    assert report.by_pool_group[0].stats.primary_used_num_blocks == 4
    assert report.by_life_cycle[0].snapshot_stats.iter_reused_tokens == 32
