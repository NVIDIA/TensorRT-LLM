# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor import profiling
from tensorrt_llm._torch.pyexecutor.profiling import (
    KV_POOL_LOG_ENV_VAR_NAME,
    PyExecutorProfileManager,
    load_kv_pool_log_enabled,
)
from tensorrt_llm.metrics.logging import format_kv_pool_fields
from tensorrt_llm.metrics.perf import PerfMetricsCollector
from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheTier

pytestmark = pytest.mark.cpu_only

GIB = 2**30


def _capacity(total, free, evictable, slot_sizes=(GIB,)):
    # Same fields as the native and pure-Python StorageStatistics objects.
    return SimpleNamespace(
        total=total,
        free=free,
        evictable=evictable,
        available=free + evictable,
        slot_sizes=list(slot_sizes),
    )


def _manager(tiers, snapshots):
    manager = object.__new__(KVCacheManagerV2)
    manager.impl = SimpleNamespace(cache_tier_list=tiers)
    manager.get_iteration_stats = Mock(
        side_effect=AssertionError("must not collect iteration stats")
    )
    manager.get_kv_cache_stats = Mock(side_effect=AssertionError("must not build KvCacheStats"))
    manager._get_storage_statistics = Mock(side_effect=lambda level: snapshots[int(level)])
    return manager


def _executor(manager, is_v2=True):
    return Mock(
        kv_cache_manager=manager,
        _is_kv_manager_v2=is_v2,
        is_warmup=False,
        print_log=True,
        _perf_metrics=PerfMetricsCollector(),
    )


def _profiler(executor, enabled, monkeypatch):
    monkeypatch.setenv(KV_POOL_LOG_ENV_VAR_NAME, "1" if enabled else "0")
    profiler = PyExecutorProfileManager(executor)
    profiler._metrics_log_rank_enabled = True
    lines = []
    profiler.metrics_logger._write = lines.append
    return profiler, lines


def _kv_stats():
    # 60 of 100 blocks unavailable: the aggregate kv_cache_util reads 0.600.
    return SimpleNamespace(
        max_num_blocks=100, free_num_blocks=40, reused_blocks=0, missed_blocks=0, cache_hit_rate=0
    )


def _loop(collector, read_kv_cache=None, kv_refresh_interval=16):
    return collector.collect_loop(
        {
            "iter": 0,
            "globalRank": 0,
            "rank": 0,
            "numScheduledRequests": 1,
            "adpDummyCtxTokens": 0,
            "adpDummyGenTokens": 0,
            "numFetchedRequestsTotal": 1,
            "numFetchedRequestsAllRanks": 1,
            "hostStepTimeMS": 1.0,
            "prevDeviceStepTimeMS": None,
            "states": "{}",
            "numActiveRequests": 1,
            "numDummyRequests": 0,
            "numUnscheduledRequests": 0,
        },
        read_kv_cache=read_kv_cache,
        kv_refresh_interval=kv_refresh_interval,
    )


def _record(profiler):
    """A loop record whose KV snapshot was read on this loop by the profiler's collector."""
    return _loop(profiler.metrics, read_kv_cache=_kv_stats)


def test_pool_utilization_uses_tier_local_pools_without_iteration_stats():
    # GPU and host pool counts differ; exclude disk. Host counts evictable pages as occupied.
    manager = _manager(
        [CacheTier.GPU_MEM, CacheTier.HOST_MEM, CacheTier.DISK],
        {
            0: [_capacity(100, 5, 5), _capacity(300, 120, 30)],
            1: [_capacity(200, 50, 100)],
        },
    )
    pools = manager.get_kv_cache_pool_utilization()
    assert pools["gpu"] == pytest.approx([0.9, 0.5])
    assert pools["host"] == pytest.approx([0.75])
    assert manager._get_storage_statistics.call_args_list == [call(0), call(1)]


def test_full_host_pool_of_evictable_pages_reads_as_occupied():
    manager = _manager(
        [CacheTier.GPU_MEM, CacheTier.HOST_MEM],
        {0: [_capacity(100, 50, 0)], 1: [_capacity(200, 0, 200)]},
    )
    assert manager.get_kv_cache_pool_utilization()["host"] == pytest.approx([1.0])


@pytest.mark.parametrize("host_enabled", [False, True])
def test_zero_capacity_and_missing_host(host_enabled):
    tiers = (
        [CacheTier.GPU_MEM, CacheTier.HOST_MEM]
        if host_enabled
        else [CacheTier.GPU_MEM, CacheTier.DISK]
    )
    manager = _manager(tiers, {0: [_capacity(0, 0, 0)], 1: [_capacity(0, 0, 0)]})
    pools = manager.get_kv_cache_pool_utilization()
    assert pools == {"gpu": [None], "host": [None] if host_enabled else None}
    assert manager._get_storage_statistics.call_count == (2 if host_enabled else 1)


def test_gpu_pool_bytes_count_every_role_of_a_slot():
    manager = _manager(
        [CacheTier.GPU_MEM], {0: [_capacity(10, 0, 0, (100, 28)), _capacity(4, 0, 0)]}
    )
    assert manager.get_gpu_pool_bytes() == [1280, 4 * GIB]


def test_format_kv_pool_fields():
    text = format_kv_pool_fields({"gpu": [0.9, None], "host": None}, gpu_pool_gib=[1.5, 0.25])
    assert text == (
        "kv_cache_gpu_pool_gib = [0:1.500;1:0.250], "
        "kv_cache_gpu_pool_util = [0:0.900;1:N/A], "
        "kv_cache_host_pool_util = N/A, "
    )



def test_pool_fields_follow_the_kv_cache_util_refresh(monkeypatch):
    manager = _manager(
        [CacheTier.GPU_MEM, CacheTier.HOST_MEM],
        {0: [_capacity(100, 5, 5)], 1: [_capacity(200, 50, 100)]},
    )
    executor = _executor(manager)  # The profile manager holds only a weakref.
    profiler, lines = _profiler(executor, True, monkeypatch)
    collector = profiler.metrics
    read_kv_cache = Mock(side_effect=_kv_stats)
    for _ in range(4):
        profiler.expose_metrics(_loop(collector, read_kv_cache, kv_refresh_interval=2))
    assert read_kv_cache.call_count == 2
    assert len(lines) == 4
    assert "pool" not in lines[1] and "pool" not in lines[3]
    # The first sample also reports pool sizes; the pool fields follow the KV cache fields.
    assert (
        "kv_cache_util = 0.600, kv_reused_blocks = 0, kv_missed_blocks = 0, "
        "kv_hit_rate = 0.000, kv_cache_gpu_pool_gib = [0:100.000], "
        "kv_cache_gpu_pool_util = [0:0.900], kv_cache_host_pool_util = [0:0.750], "
        "currank_total_requests"
    ) in lines[0]
    assert "pool_gib" not in lines[2] and "kv_cache_gpu_pool_util = [0:0.900]" in lines[2]
    # Two samples read GPU and host utilization; the first printed line also reads GPU sizes.
    assert manager._get_storage_statistics.call_args_list == [call(0), call(1), call(0)] + [
        call(0),
        call(1),
    ]


def test_iteration_stats_snapshots_trigger_one_sample_each(monkeypatch):
    # With iteration stats enabled the executor collects rank snapshots itself.
    manager = _manager([CacheTier.GPU_MEM], {0: [_capacity(100, 5, 5)]})
    executor = _executor(manager)
    profiler, lines = _profiler(executor, True, monkeypatch)
    collector = profiler.metrics
    collector.collect_rank({"kvCacheStats": {"maxNumBlocks": 100}}, capture_iter=0)
    profiler.expose_metrics(_loop(collector))
    profiler.expose_metrics(_loop(collector))
    assert "pool_util" in lines[0] and "pool" not in lines[1]
    # A loop without any KV snapshot has nothing to line up with.
    profiler.expose_metrics(_loop(PerfMetricsCollector()))
    assert "pool" not in lines[2]


def test_pools_are_read_with_the_snapshot_not_at_print_time(monkeypatch):
    # With iteration stats the rank snapshot is taken before the line is printed, and pages
    # can be released in between (for example by deferred closes on an idle loop).
    snapshots = {0: [_capacity(100, 5, 5)]}
    manager = _manager([CacheTier.GPU_MEM], snapshots)
    executor = _executor(manager)
    profiler, lines = _profiler(executor, True, monkeypatch)
    profiler.metrics.collect_rank({"kvCacheStats": {"maxNumBlocks": 100}}, capture_iter=0)
    snapshots[0] = [_capacity(100, 100, 0)]
    profiler.expose_metrics(_loop(profiler.metrics))
    assert "kv_cache_gpu_pool_util = [0:0.900]" in lines[0]


def test_warmup_snapshot_is_not_sampled_after_warmup(monkeypatch):
    manager = _manager([CacheTier.GPU_MEM], {0: [_capacity(100, 5, 5)]})
    executor = _executor(manager)
    executor.is_warmup = True
    profiler, lines = _profiler(executor, True, monkeypatch)
    collector = profiler.metrics
    profiler.expose_metrics(_loop(collector, _kv_stats))
    executor.is_warmup = False
    profiler.expose_metrics(_loop(collector, _kv_stats))
    assert all("pool" not in line for line in lines)
    manager._get_storage_statistics.assert_not_called()


@pytest.mark.parametrize("enabled,is_v2", [(False, True), (True, False)])
def test_disabled_and_legacy_paths_read_no_pools(monkeypatch, enabled, is_v2):
    manager = _manager([CacheTier.GPU_MEM], {0: [_capacity(100, 5, 5)]})
    executor = _executor(manager, is_v2=is_v2)
    profiler, lines = _profiler(executor, enabled, monkeypatch)
    profiler.expose_metrics(_record(profiler))
    assert len(lines) == 1 and "pool" not in lines[0]
    manager._get_storage_statistics.assert_not_called()


@pytest.mark.parametrize("print_log,rank_enabled", [(False, True), (True, False)])
def test_non_logging_ranks_read_no_pools(monkeypatch, print_log, rank_enabled):
    manager = _manager([CacheTier.GPU_MEM], {0: [_capacity(100, 5, 5)]})
    executor = _executor(manager)
    executor.print_log = print_log
    profiler, lines = _profiler(executor, True, monkeypatch)
    profiler._metrics_log_rank_enabled = rank_enabled
    profiler.expose_metrics(_record(profiler))
    assert lines == []
    manager._get_storage_statistics.assert_not_called()


def test_pool_fields_stay_out_of_the_collected_record(monkeypatch):
    manager = _manager([CacheTier.GPU_MEM], {0: [_capacity(100, 5, 5)]})
    executor = _executor(manager)  # The profile manager holds only a weakref.
    profiler, lines = _profiler(executor, True, monkeypatch)
    record = _record(profiler)
    profiler.expose_metrics(record)
    assert "pool_util" in lines[0]
    assert not any("pool" in key.lower() for key in record.iteration_view())


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, False),
        ("", False),
        ("0", False),
        (" False ", False),
        ("off", False),
        ("no", False),
        ("1", True),
        (" True ", True),
        ("ON", True),
        ("yes", True),
    ],
)
def test_load_kv_pool_log_enabled(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(KV_POOL_LOG_ENV_VAR_NAME, raising=False)
    else:
        monkeypatch.setenv(KV_POOL_LOG_ENV_VAR_NAME, raw)
    warning = Mock()
    monkeypatch.setattr(profiling.logger, "warning", warning)
    assert load_kv_pool_log_enabled() is expected
    warning.assert_not_called()


@pytest.mark.parametrize("raw", ["2", "50", "ture"])
def test_unrecognized_switch_stays_disabled_with_a_warning(monkeypatch, raw):
    monkeypatch.setenv(KV_POOL_LOG_ENV_VAR_NAME, raw)
    warning = Mock()
    monkeypatch.setattr(profiling.logger, "warning", warning)
    assert load_kv_pool_log_enabled() is False
    warning.assert_called_once()
    assert KV_POOL_LOG_ENV_VAR_NAME in warning.call_args.args[0]
