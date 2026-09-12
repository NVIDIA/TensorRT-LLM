# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tensorrt_llm._torch.pyexecutor import profiling
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kv_module
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.profiling import PyExecutorProfileManager
from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheTier

pytestmark = pytest.mark.cpu_only


def _capacity(total, free, evictable):
    return SimpleNamespace(total=total, available=free + evictable)


def _manager(tiers, snapshots):
    manager = object.__new__(KVCacheManagerV2)
    manager.impl = SimpleNamespace(cache_tier_list=tiers)
    manager.enable_stats = False
    manager.get_iteration_stats = Mock(
        side_effect=AssertionError("must not collect iteration stats")
    )
    manager._get_storage_statistics = Mock(side_effect=lambda level: snapshots[int(level)])
    return manager


def _executor(manager):
    return Mock(
        kv_cache_manager=manager,
        iter_counter=50,
        global_rank=0,
        dist=SimpleNamespace(rank=0),
        enable_iter_perf_stats=False,
    )


@pytest.mark.parametrize("include_pools", [False, True])
def test_capacity_snapshot_uses_tier_local_pools_without_iteration_stats(include_pools):
    # GPU and host pool counts differ; exclude disk.
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
        assert pools["host"] == pytest.approx([0.25])
        assert manager._get_storage_statistics.call_args_list == [call(0), call(1)]
    else:
        assert pools == {}
        manager._get_storage_statistics.assert_called_once_with(0)


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
    assert pools == ({"gpu": [None], "host": [None]} if host_enabled else {"gpu": [None]})
    assert manager._get_storage_statistics.call_count == (2 if host_enabled else 1)


def test_pool_mapping_uses_each_tiers_lifecycle_indices(monkeypatch):
    manager = _manager([CacheTier.GPU_MEM, CacheTier.HOST_MEM, CacheTier.DISK], {})
    indices = Mock(side_effect=lambda impl, level: {0: [0, 1, 0], 1: [0, 0, 0]}[int(level)])
    monkeypatch.setattr(kv_module._introspection, "life_cycle_pool_group_indices", indices)
    assert manager.get_kv_cache_pool_mapping() == {
        "gpu": {0: [0, 2], 1: [1]},
        "host": {0: [0, 1, 2]},
    }
    assert indices.call_args_list == [call(manager.impl, 0), call(manager.impl, 1)]


def test_sampled_log_reuses_gpu_snapshot_and_does_not_repeat_stale_host_values(monkeypatch):
    manager = _manager(
        [CacheTier.GPU_MEM, CacheTier.HOST_MEM],
        {0: [_capacity(100, 5, 5)], 1: [_capacity(200, 50, 100)]},
    )
    manager.get_kv_cache_pool_mapping = Mock(return_value={"gpu": {0: [0]}, "host": {0: [0]}})
    executor = _executor(manager)
    profiler = PyExecutorProfileManager(executor)
    log = Mock()
    monkeypatch.setattr(profiling.logger, "info", log)

    aggregate, fields = profiler._kv_cache_log_fields(50)
    assert aggregate == "0.900"
    assert fields == "kv_cache_gpu_pool_util = [0:0.900], kv_cache_host_pool_util = [0:0.250], "
    assert manager._get_storage_statistics.call_args_list == [call(0), call(1)]

    manager._get_storage_statistics.reset_mock()
    executor.iter_counter = 51
    assert profiler._kv_cache_log_fields(50) == ("0.900", "")
    manager._get_storage_statistics.assert_called_once_with(0)

    executor.iter_counter = 100
    profiler._kv_cache_log_fields(50)
    manager.get_kv_cache_pool_mapping.assert_called_once()
    log.assert_called_once()


def test_log_formats_unavailable_capacity_and_absent_host(monkeypatch):
    manager = _manager([CacheTier.GPU_MEM], {0: [_capacity(0, 0, 0)]})
    manager.get_kv_cache_pool_mapping = Mock(return_value={"gpu": {0: [0]}})
    executor = _executor(manager)
    monkeypatch.setattr(profiling.logger, "info", Mock())
    assert PyExecutorProfileManager(executor)._kv_cache_log_fields(50) == (
        "N/A",
        "kv_cache_gpu_pool_util = [0:N/A], kv_cache_host_pool_util = N/A, ",
    )


@pytest.mark.parametrize("v2,interval", [(True, 0), (False, 50)])
def test_disabled_or_legacy_log_keeps_existing_stats_path(v2, interval):
    stats = SimpleNamespace(max_num_blocks=100, free_num_blocks=25)
    manager = SimpleNamespace(get_kv_cache_stats=Mock(return_value=stats))
    if v2:
        manager.get_kv_cache_utilization = Mock(side_effect=AssertionError("must not sample"))
    executor = _executor(manager)
    assert PyExecutorProfileManager(executor)._kv_cache_log_fields(interval) == ("0.750", "")
    manager.get_kv_cache_stats.assert_called_once()


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
    manager.get_kv_cache_pool_mapping = Mock(return_value={})
    executor = _executor(manager)
    executor.print_log = print_log
    executor.dist.rank = rank
    executor.profile_start_iters = set()
    executor.profile_stop_iters = set()
    executor._profile_state_lock = threading.Lock()
    executor.is_warmup = False
    executor._iter_adp_dummy_ctx_tokens = 0
    executor._iter_adp_dummy_gen_tokens = 0
    profiler = PyExecutorProfileManager(executor)
    monkeypatch.setenv("TLLM_KV_POOL_LOG_INTERVAL", "50")
    monkeypatch.setenv("TLLM_PROFILE_LOG_RANKS", log_ranks)
    monkeypatch.delenv("TLLM_TORCH_PROFILE_TRACE", raising=False)
    monkeypatch.delenv("TLLM_PROFILE_START_STOP", raising=False)
    monkeypatch.setattr(
        profiling.torch.cuda, "Event", Mock(return_value=Mock(elapsed_time=lambda _: 1))
    )
    monkeypatch.setattr(profiling, "get_calibrator", Mock(return_value=Mock()))
    monkeypatch.setattr(profiling, "get_global_profiler", lambda: None)
    monkeypatch.setattr(profiling.logger, "info", Mock())
    with profiler.profile_step() as step:
        step()  # Initialize timing.
        step()
    assert manager._get_storage_statistics.call_count == expected_reads
