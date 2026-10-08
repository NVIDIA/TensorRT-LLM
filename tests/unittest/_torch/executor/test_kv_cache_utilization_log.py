# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tensorrt_llm._torch.pyexecutor import profiling
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kv_module
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.profiling import (
    KV_POOL_LOG_INTERVAL_ENV_VAR_NAME,
    PyExecutorProfileManager,
    load_kv_pool_log_interval,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheTier

pytestmark = pytest.mark.cpu_only


def _capacity(total, free, evictable):
    # Same fields as the native and pure-Python StorageStatistics objects.
    return SimpleNamespace(total=total, free=free, evictable=evictable, available=free + evictable)


def _manager(tiers, snapshots):
    manager = object.__new__(KVCacheManagerV2)
    manager.impl = SimpleNamespace(cache_tier_list=tiers)
    manager.get_iteration_stats = Mock(
        side_effect=AssertionError("must not collect iteration stats")
    )
    manager._get_storage_statistics = Mock(side_effect=lambda level: snapshots[int(level)])
    return manager


def _executor(manager, is_v2=True):
    return Mock(
        kv_cache_manager=manager,
        _is_kv_manager_v2=is_v2,
        iter_counter=50,
        is_warmup=False,
        global_rank=0,
        dist=SimpleNamespace(rank=0),
        enable_iter_perf_stats=False,
    )


def _profiler(executor, interval, monkeypatch):
    monkeypatch.setenv(KV_POOL_LOG_INTERVAL_ENV_VAR_NAME, str(interval))
    return PyExecutorProfileManager(executor)


@pytest.mark.parametrize("include_pools", [False, True])
def test_capacity_snapshot_uses_tier_local_pools_without_iteration_stats(include_pools):
    # GPU and host pool counts differ; exclude disk. Host counts evictable pages as occupied.
    manager = _manager(
        [CacheTier.GPU_MEM, CacheTier.HOST_MEM, CacheTier.DISK],
        {
            0: [_capacity(100, 5, 5), _capacity(300, 120, 30)],
            1: [_capacity(200, 50, 100)],
        },
    )
    utilization, pools = manager.get_kv_cache_utilization(include_pools)
    assert utilization == pytest.approx(0.6)
    if include_pools:
        assert pools["gpu"] == pytest.approx([0.9, 0.5])
        assert pools["host"] == pytest.approx([0.75])
        assert manager._get_storage_statistics.call_args_list == [call(0), call(1)]
    else:
        assert pools == {}
        manager._get_storage_statistics.assert_called_once_with(0)


def test_full_host_pool_of_evictable_pages_reads_as_occupied():
    manager = _manager(
        [CacheTier.GPU_MEM, CacheTier.HOST_MEM],
        {0: [_capacity(100, 50, 0)], 1: [_capacity(200, 0, 200)]},
    )
    _, pools = manager.get_kv_cache_utilization(True)
    assert pools["host"] == pytest.approx([1.0])


@pytest.mark.parametrize("host_enabled", [False, True])
def test_zero_capacity_and_missing_host(host_enabled):
    tiers = (
        [CacheTier.GPU_MEM, CacheTier.HOST_MEM]
        if host_enabled
        else [CacheTier.GPU_MEM, CacheTier.DISK]
    )
    manager = _manager(tiers, {0: [_capacity(0, 0, 0)], 1: [_capacity(0, 0, 0)]})
    utilization, pools = manager.get_kv_cache_utilization(True)
    assert utilization is None
    assert pools == {"gpu": [None], "host": [None] if host_enabled else None}
    assert manager._get_storage_statistics.call_count == (2 if host_enabled else 1)


def test_kv_cache_stats_and_utilization_share_the_aggregate_definition():
    manager = _manager([CacheTier.GPU_MEM], {0: [_capacity(100, 5, 5), _capacity(300, 120, 30)]})
    manager.impl.get_committed_stats = Mock(
        return_value=SimpleNamespace(
            alloc_total_blocks=0, alloc_new_blocks=0, reused_blocks=0, missed_blocks=0
        )
    )
    manager.impl.get_quota = Mock(return_value=0)
    manager.tokens_per_block = 32
    manager._storage_pool_groups_by_window = Mock(return_value={})
    stats = manager.get_kv_cache_stats()
    utilization, _ = manager.get_kv_cache_utilization()
    assert (stats.max_num_blocks, stats.free_num_blocks) == (400, 160)
    assert utilization == pytest.approx(1.0 - stats.free_num_blocks / stats.max_num_blocks)


def test_construction_log_includes_cold_tier_pool_mapping(monkeypatch):
    manager = _manager([CacheTier.GPU_MEM, CacheTier.HOST_MEM, CacheTier.DISK], {})
    manager.impl.pool_group_descs = []
    manager.kv_cache_manager_py_config = SimpleNamespace(
        layers=[SimpleNamespace(layer_id=0, buffers=[SimpleNamespace(role="K")])]
    )
    indices = Mock(return_value=[0, 1, 0])
    monkeypatch.setattr(kv_module._introspection, "life_cycle_pool_group_indices", indices)
    log = Mock()
    monkeypatch.setattr(kv_module.logger, "info", log)
    manager._log_kv_cache_pool_lifecycle_mapping()
    assert indices.call_args_list == [call(manager.impl, 1)]
    assert log.call_args_list[-1] == call(
        "HOST_MEM pool_group_id -> layer_group_ids = {0: [0, 2], 1: [1]}"
    )


def test_sampled_log_reuses_gpu_snapshot_and_samples_each_iteration_once(monkeypatch):
    manager = _manager(
        [CacheTier.GPU_MEM, CacheTier.HOST_MEM],
        {0: [_capacity(100, 5, 5)], 1: [_capacity(200, 50, 100)]},
    )
    executor = _executor(manager)
    profiler = _profiler(executor, 50, monkeypatch)

    assert profiler._kv_cache_log_fields() == [
        "kv_cache_util = 0.900",
        "kv_cache_gpu_pool_util = [0:0.900]",
        "kv_cache_host_pool_util = [0:0.750]",
    ]
    assert manager._get_storage_statistics.call_args_list == [call(0), call(1)]

    manager._get_storage_statistics.reset_mock()
    # Re-entering the same iteration (retry paths) does not resample the pools.
    assert profiler._kv_cache_log_fields() == ["kv_cache_util = 0.900"]
    executor.iter_counter = 51
    assert profiler._kv_cache_log_fields() == ["kv_cache_util = 0.900"]
    assert manager._get_storage_statistics.call_args_list == [call(0), call(0)]

    executor.iter_counter = 100
    assert len(profiler._kv_cache_log_fields()) == 3


def test_warmup_iterations_skip_pool_sampling(monkeypatch):
    manager = _manager(
        [CacheTier.GPU_MEM, CacheTier.HOST_MEM],
        {0: [_capacity(100, 5, 5)], 1: [_capacity(200, 50, 100)]},
    )
    executor = _executor(manager)
    executor.is_warmup = True
    profiler = _profiler(executor, 50, monkeypatch)
    assert profiler._kv_cache_log_fields() == ["kv_cache_util = 0.900"]
    manager._get_storage_statistics.assert_called_once_with(0)


def test_log_formats_unavailable_capacity_and_absent_host(monkeypatch):
    manager = _manager([CacheTier.GPU_MEM], {0: [_capacity(0, 0, 0)]})
    executor = _executor(manager)
    profiler = _profiler(executor, 50, monkeypatch)
    assert profiler._kv_cache_log_fields() == [
        "kv_cache_util = N/A",
        "kv_cache_gpu_pool_util = [0:N/A]",
        "kv_cache_host_pool_util = N/A",
    ]


def test_disabled_interval_keeps_v2_on_the_capacity_snapshot(monkeypatch):
    manager = _manager([CacheTier.GPU_MEM, CacheTier.HOST_MEM], {0: [_capacity(100, 5, 5)]})
    manager.get_kv_cache_stats = Mock(side_effect=AssertionError("must not build KvCacheStats"))
    executor = _executor(manager)
    profiler = _profiler(executor, 0, monkeypatch)
    assert profiler._kv_cache_log_fields() == ["kv_cache_util = 0.900"]
    manager._get_storage_statistics.assert_called_once_with(0)


def test_legacy_manager_keeps_existing_stats_path(monkeypatch):
    stats = SimpleNamespace(max_num_blocks=100, free_num_blocks=25)
    manager = SimpleNamespace(get_kv_cache_stats=Mock(return_value=stats))
    executor = _executor(manager, is_v2=False)
    profiler = _profiler(executor, 50, monkeypatch)
    assert profiler._kv_cache_log_fields() == ["kv_cache_util = 0.750"]
    manager.get_kv_cache_stats.assert_called_once()


@pytest.mark.parametrize("raw,expected", [(None, 0), ("", 0), ("  ", 0), (" 50 ", 50)])
def test_load_kv_pool_log_interval_treats_unset_and_empty_as_disabled(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(KV_POOL_LOG_INTERVAL_ENV_VAR_NAME, raising=False)
    else:
        monkeypatch.setenv(KV_POOL_LOG_INTERVAL_ENV_VAR_NAME, raw)
    assert load_kv_pool_log_interval() == expected


@pytest.mark.parametrize("raw", ["-1", "50.0", "50s"])
def test_invalid_interval_fails_at_construction_and_names_the_variable(monkeypatch, raw):
    monkeypatch.setenv(KV_POOL_LOG_INTERVAL_ENV_VAR_NAME, raw)
    with pytest.raises(ValueError, match=KV_POOL_LOG_INTERVAL_ENV_VAR_NAME):
        PyExecutorProfileManager(Mock())


@pytest.mark.parametrize(
    "print_log,rank,log_ranks,expected_reads",
    [
        (False, 0, "all", 0),
        (True, 1, "0", 0),
        (True, 1, "all", 2),
        (True, 1, "1", 2),
    ],
)
def test_profile_loop_only_samples_on_logging_ranks(
    monkeypatch, print_log, rank, log_ranks, expected_reads
):
    manager = _manager(
        [CacheTier.GPU_MEM, CacheTier.HOST_MEM],
        {0: [_capacity(100, 5, 5)], 1: [_capacity(200, 50, 100)]},
    )
    executor = _executor(manager)
    executor.print_log = print_log
    executor.dist.rank = rank
    executor.profile_start_iters = set()
    executor.profile_stop_iters = set()
    executor._profile_state_lock = threading.Lock()
    executor._iter_adp_dummy_ctx_tokens = 0
    executor._iter_adp_dummy_gen_tokens = 0
    monkeypatch.setenv("TLLM_PROFILE_LOG_RANKS", log_ranks)
    monkeypatch.delenv("TLLM_TORCH_PROFILE_TRACE", raising=False)
    monkeypatch.delenv("TLLM_PROFILE_START_STOP", raising=False)
    monkeypatch.setattr(
        profiling.torch.cuda, "Event", Mock(return_value=Mock(elapsed_time=lambda _: 1))
    )
    monkeypatch.setattr(profiling, "get_calibrator", Mock(return_value=Mock()))
    monkeypatch.setattr(profiling, "get_global_profiler", lambda: None)
    monkeypatch.setattr(profiling.logger, "info", Mock())
    warning = Mock()
    monkeypatch.setattr(profiling.logger, "warning", warning)
    profiler = _profiler(executor, 50, monkeypatch)
    with profiler.profile_step() as step:
        step()  # Initialize timing.
        step()
    assert manager._get_storage_statistics.call_count == expected_reads
    # The knob warns once when print_iter_log is off; rank filtering is intentional.
    assert warning.call_count == (0 if print_log else 1)
