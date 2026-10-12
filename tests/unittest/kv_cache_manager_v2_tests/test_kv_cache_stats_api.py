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


import pytest
import torch

from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    GPU_LEVEL,
    AttentionLayerConfig,
    BufferConfig,
    GpuCacheTierConfig,
    KVCacheIterationStatsDelta,
    KVCacheManager,
    KVCacheManagerConfig,
    KVCacheStatsDelta,
    LifeCycleId,
    SsmSnapshotIterationStatsDelta,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._common import BlockOrdinal
from tensorrt_llm.runtime.kv_cache_manager_v2._core._pending_stats import _PendingStats

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _make_config(*, enable_stats: bool = True) -> KVCacheManagerConfig:
    return KVCacheManagerConfig(
        tokens_per_block=4,
        cache_tiers=[GpuCacheTierConfig(quota=4 << 20)],
        layers=[
            AttentionLayerConfig(
                layer_id=0,
                buffers=[BufferConfig(role="key", size=4096)],
            )
        ],
        enable_stats=enable_stats,
    )


def test_stats_delta_arithmetic() -> None:
    stats = KVCacheStatsDelta(4, 3, 2, 1)
    delta = KVCacheStatsDelta(1, 2, 3, 4)
    stats.add(delta)
    assert stats == KVCacheStatsDelta(5, 5, 5, 5)
    stats.subtract(delta)
    assert stats == KVCacheStatsDelta(4, 3, 2, 1)
    copied = stats.copy()
    stats.clear()
    assert stats.empty
    assert copied == KVCacheStatsDelta(4, 3, 2, 1)

    iteration = KVCacheIterationStatsDelta(iter_reused_blocks=3, iter_missed_blocks=1)
    assert iteration.iter_cache_hit_rate == 0.75
    iteration.clear()
    assert iteration.empty
    assert iteration.iter_cache_hit_rate == 0.0

    snapshot = SsmSnapshotIterationStatsDelta(
        iter_snapshot_lookups=4,
        iter_snapshot_hits=3,
        iter_snapshot_misses=1,
        iter_reused_tokens=96,
        iter_unreused_tokens=32,
        iter_aligned_snapshot_hits=2,
        iter_unaligned_snapshot_hits=1,
    )
    assert snapshot.iter_snapshot_hit_rate == 0.75
    snapshot.clear()
    assert snapshot.empty
    assert snapshot.iter_snapshot_hit_rate == 0.0


@pytest.mark.parametrize("enable_stats", [False, True])
def test_manager_stats_config_and_api(enable_stats: bool) -> None:
    manager = KVCacheManager(_make_config(enable_stats=enable_stats))
    cache = None
    try:
        assert manager.init_config.enable_stats is enable_stats
        assert manager.get_committed_stats() == KVCacheStatsDelta()
        assert manager.get_and_reset_iteration_stats() == {}
        assert manager.get_and_reset_ssm_snapshot_iteration_stats() == {}
        peak_stats = manager.get_and_reset_iteration_peak_block_stats(GPU_LEVEL)
        assert len(peak_stats) == 1
        assert peak_stats[0].available >= 0
        assert peak_stats[0].unavailable >= 0
        assert peak_stats[0].evictable >= 0

        manager.mark_stats_dirty(11)
        manager.mark_stats_dirty(None)
        assert manager.get_dirty_stats_kv_cache_ids() == {11}
        manager.mark_stats_excluded(11)
        assert manager.is_stats_excluded(11)
        assert manager.get_dirty_stats_kv_cache_ids() == set()
        manager.clear_stats_excluded(11)
        assert not manager.is_stats_excluded(11)

        cache = manager.create_kv_cache(id=17, expected_prompt_length=8)
        manager.mark_stats_dirty(17)
        assert cache.commit_pending_stats() == KVCacheStatsDelta()
        assert manager.get_dirty_stats_kv_cache_ids() == set()
        cache.discard_pending_stats()

        # Exercise the collection path: allocate blocks and commit the pending
        # stats. With stats enabled the allocation must be visible in both the
        # per-request and manager-level committed stats; with stats disabled
        # everything must stay empty.
        stream = torch.cuda.Stream()
        assert cache.resume(stream.cuda_stream)
        assert cache.resize(8)
        request_stats = cache.commit_pending_stats()
        committed = manager.get_committed_stats()
        iteration = manager.get_and_reset_iteration_stats()
        if enable_stats:
            assert request_stats.alloc_total_blocks > 0
            assert committed.alloc_total_blocks > 0
            assert any(not delta.empty for delta in iteration.values())
        else:
            assert request_stats == KVCacheStatsDelta()
            assert committed == KVCacheStatsDelta()
            assert iteration == {}
        stream.synchronize()
    finally:
        if cache is not None:
            cache.close()
        manager.shutdown()


def test_pending_connector_reuse_survives_a_split_and_a_shrink() -> None:
    life_cycles = (LifeCycleId(0), LifeCycleId(1))
    pending = _PendingStats()
    for life_cycle in life_cycles:
        assert pending.record_allocation_range(
            life_cycle, BlockOrdinal(0), BlockOrdinal(4), beam_width=1, count_as_missed=True
        )

    assert pending.record_connector_reuse(BlockOrdinal(1), BlockOrdinal(3))
    assert pending.global_stats == KVCacheStatsDelta(
        alloc_total_blocks=8, alloc_new_blocks=8, reused_blocks=4, missed_blocks=4
    )
    assert pending.request_stats == pending.global_stats
    for life_cycle in life_cycles:
        iteration = pending.iteration_stats_by_life_cycle[life_cycle]
        assert iteration.iter_reused_blocks == 2
        assert iteration.iter_full_reused_blocks == 2
        assert iteration.iter_missed_blocks == 2
    assert pending.connector_reused_blocks_by_life_cycle == {life_cycle: 2 for life_cycle in life_cycles}
    # A block is rebooked at most once.
    assert not pending.record_connector_reuse(BlockOrdinal(1), BlockOrdinal(3))

    # Shrinking drops each life cycle's tail from whichever category it was in.
    assert pending.subtract_allocation_range(BlockOrdinal(2), BlockOrdinal(4))
    assert pending.global_stats == KVCacheStatsDelta(
        alloc_total_blocks=4, alloc_new_blocks=4, reused_blocks=2, missed_blocks=2
    )
    assert pending.connector_reused_blocks_by_life_cycle == {life_cycle: 1 for life_cycle in life_cycles}

    assert pending.subtract_allocation_range(BlockOrdinal(0), BlockOrdinal(2))
    assert pending.empty
    assert pending.allocation_segments == []


def test_connector_reuse_is_committed_as_reused() -> None:
    manager = KVCacheManager(_make_config())
    cache = None
    try:
        cache = manager.create_kv_cache(id=17, expected_prompt_length=16)
        stream = torch.cuda.Stream()
        assert cache.resume(stream.cuda_stream)
        assert cache.resize(12)
        cache.record_connector_reuse(BlockOrdinal(0), BlockOrdinal(2))

        request_stats = cache.commit_pending_stats()
        assert request_stats == KVCacheStatsDelta(
            alloc_total_blocks=3, alloc_new_blocks=3, reused_blocks=2, missed_blocks=1
        )
        assert manager.get_committed_stats() == request_stats
        (iteration,) = manager.get_and_reset_iteration_stats().values()
        assert iteration.iter_reused_blocks == 2
        assert iteration.iter_full_reused_blocks == 2
        assert iteration.iter_missed_blocks == 1
        assert list(manager.get_and_reset_iteration_connector_reused_blocks().values()) == [2]
        assert manager.get_and_reset_iteration_connector_reused_blocks() == {}

        # Once committed, the allocation is no longer pending and stays a miss.
        cache.record_connector_reuse(BlockOrdinal(2), BlockOrdinal(3))
        assert cache.commit_pending_stats() == KVCacheStatsDelta()
        stream.synchronize()
    finally:
        if cache is not None:
            cache.close()
        manager.shutdown()
