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

"""TRTLLM-15217: SSM/recurrent life cycles must appear in V2 iteration stats.

The page-movement recorders used to drop every non-attention life cycle, which
made KDA (Kimi K3) recurrent-state offload / onboard / drop invisible in
iteration statistics. These tests drive the recorders directly with a
duck-typed stand-in so they run without a GPU or an allocated cache.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm.runtime.kv_cache_manager_v2._common import GPU_LEVEL, CacheLevel
from tensorrt_llm.runtime.kv_cache_manager_v2._core._kv_cache import _KVCache
from tensorrt_llm.runtime.kv_cache_manager_v2._life_cycle_registry import (
    AttnLifeCycle,
    LifeCycleId,
    SsmLifeCycle,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._stats import KVCacheIterationStatsDelta

ATTN_LC = LifeCycleId(0)
SSM_LC = LifeCycleId(1)
PAGE_BYTES = 16
HOST_LEVEL = CacheLevel(GPU_LEVEL + 1)


def _make_recorder():
    """Duck-typed _KVCache exposing only what the stats recorders touch.

    The recording methods are bound off the real class, so the life-cycle
    filtering under test is the production implementation.
    """
    committed = []
    life_cycles = {ATTN_LC: AttnLifeCycle(None, 0), SSM_LC: SsmLifeCycle()}
    manager = SimpleNamespace(
        _life_cycles=SimpleNamespace(get_life_cycle=life_cycles.__getitem__),
        _storage=SimpleNamespace(
            get_pool_group_index=lambda life_cycle: life_cycle,
            slot_size=lambda _pool_group: [PAGE_BYTES],
        ),
        commit_stats=lambda stats, by_life_cycle: committed.append((stats, by_life_cycle)),
    )
    recorder = SimpleNamespace(manager=manager)
    recorder._should_record_stats = lambda: True
    for name in (
        "_is_attention_life_cycle",
        "_record_direct_iteration_stats",
        "_record_migrated_slots",
        "_record_dropped_pages",
    ):
        setattr(recorder, name, getattr(_KVCache, name).__get__(recorder))
    return recorder, committed


@pytest.mark.parametrize("life_cycle", [ATTN_LC, SSM_LC])
def test_offload_is_recorded_for_every_life_cycle(life_cycle: LifeCycleId) -> None:
    recorder, committed = _make_recorder()
    page = SimpleNamespace(life_cycle=life_cycle)

    recorder._record_migrated_slots([page], [object()], GPU_LEVEL, HOST_LEVEL)

    assert len(committed) == 1
    _, by_life_cycle = committed[0]
    assert set(by_life_cycle) == {life_cycle}
    assert by_life_cycle[life_cycle].iter_offload_blocks == 1
    assert by_life_cycle[life_cycle].iter_offload_bytes == PAGE_BYTES


@pytest.mark.parametrize("life_cycle", [ATTN_LC, SSM_LC])
def test_host_drop_is_recorded_for_every_life_cycle(life_cycle: LifeCycleId) -> None:
    recorder, committed = _make_recorder()
    page = SimpleNamespace(life_cycle=life_cycle)

    recorder._record_dropped_pages([page], HOST_LEVEL)

    assert len(committed) == 1
    _, by_life_cycle = committed[0]
    assert set(by_life_cycle) == {life_cycle}
    assert by_life_cycle[life_cycle].iter_host_dropped_blocks == 1
    assert by_life_cycle[life_cycle].iter_host_dropped_bytes == PAGE_BYTES


@pytest.mark.parametrize("life_cycle", [ATTN_LC, SSM_LC])
def test_direct_iteration_stats_are_recorded_for_every_life_cycle(
    life_cycle: LifeCycleId,
) -> None:
    """SSM deferred copies must reach iteration stats.

    The resume() deferred copy reports iter_intra_device_copy_* through this
    recorder for SSM life cycles too, matching the C++ backend.
    """
    recorder, committed = _make_recorder()

    recorder._record_direct_iteration_stats(
        life_cycle,
        KVCacheIterationStatsDelta(
            iter_intra_device_copy_blocks=1,
            iter_intra_device_copy_bytes=PAGE_BYTES,
        ),
    )

    assert len(committed) == 1
    _, by_life_cycle = committed[0]
    assert set(by_life_cycle) == {life_cycle}
    assert by_life_cycle[life_cycle].iter_intra_device_copy_blocks == 1
    assert by_life_cycle[life_cycle].iter_intra_device_copy_bytes == PAGE_BYTES


def test_onboard_counts_globally_only_for_attention() -> None:
    """Onboard is per-life-cycle; global cache-hit counters are attention-only.

    alloc_total_blocks / alloc_new_blocks feed the global cache-hit rate, which
    is defined over attention blocks only.
    """
    recorder, committed = _make_recorder()

    recorder._record_migrated_slots(
        [SimpleNamespace(life_cycle=SSM_LC)], [object()], HOST_LEVEL, GPU_LEVEL
    )
    recorder._record_migrated_slots(
        [SimpleNamespace(life_cycle=ATTN_LC)], [object()], HOST_LEVEL, GPU_LEVEL
    )

    assert len(committed) == 2
    ssm_stats, ssm_by_life_cycle = committed[0]
    attn_stats, attn_by_life_cycle = committed[1]

    for life_cycle, by_life_cycle in ((SSM_LC, ssm_by_life_cycle), (ATTN_LC, attn_by_life_cycle)):
        assert by_life_cycle[life_cycle].iter_onboard_blocks == 1
        assert by_life_cycle[life_cycle].iter_onboard_bytes == PAGE_BYTES

    assert ssm_stats.alloc_total_blocks == 0
    assert ssm_stats.alloc_new_blocks == 0
    assert attn_stats.alloc_total_blocks == 1
    assert attn_stats.alloc_new_blocks == 1
