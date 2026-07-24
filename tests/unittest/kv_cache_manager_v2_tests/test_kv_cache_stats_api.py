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

import os

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
    PoolGroupPeakBlockStats,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(scope="module", autouse=True)
def initialize_cuda_context() -> None:
    torch.empty(1, device="cuda")


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


def test_cpp_stats_types_are_native() -> None:
    if os.environ.get("TLLM_KV_CACHE_MANAGER_V2_BACKEND", "cpp").lower() != "cpp":
        pytest.skip("C++ backend only")

    from tensorrt_llm.bindings.internal.batch_manager import kv_cache_manager_v2 as cpp

    assert KVCacheStatsDelta is cpp.KVCacheStatsDelta
    assert KVCacheIterationStatsDelta is cpp.KVCacheIterationStatsDelta
    assert PoolGroupPeakBlockStats is cpp.PoolGroupPeakBlockStats

    stats = KVCacheStatsDelta(alloc_total_blocks=1, reused_blocks=2)
    assert repr(stats) == (
        "KVCacheStatsDelta(alloc_total_blocks=1, alloc_new_blocks=0, "
        "reused_blocks=2, missed_blocks=0)"
    )
    peak = PoolGroupPeakBlockStats(available=3, unavailable=4, evictable=5)
    assert peak == PoolGroupPeakBlockStats(3, 4, 5)
    with pytest.raises(AttributeError):
        peak.available = 6


def test_manager_accepts_uint64_max_request_id() -> None:
    manager = KVCacheManager(_make_config())
    cache = None
    cuda_graph_dummy_request_id = (1 << 64) - 1
    try:
        cache = manager.create_kv_cache(id=cuda_graph_dummy_request_id)
        assert cache.id == cuda_graph_dummy_request_id
        manager.mark_stats_dirty(cuda_graph_dummy_request_id)
        assert manager.get_dirty_stats_kv_cache_ids() == {cuda_graph_dummy_request_id}
        manager.mark_stats_excluded(cuda_graph_dummy_request_id)
        assert manager.is_stats_excluded(cuda_graph_dummy_request_id)
        assert manager.get_dirty_stats_kv_cache_ids() == set()
    finally:
        if cache is not None:
            cache.close()
        manager.shutdown()


@pytest.mark.parametrize("enable_stats", [False, True])
def test_manager_stats_config_and_api(enable_stats: bool) -> None:
    manager = KVCacheManager(_make_config(enable_stats=enable_stats))
    cache = None
    try:
        assert manager.init_config.enable_stats is enable_stats
        assert manager.get_committed_stats() == KVCacheStatsDelta()
        assert manager.get_and_reset_iteration_stats() == {}
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
