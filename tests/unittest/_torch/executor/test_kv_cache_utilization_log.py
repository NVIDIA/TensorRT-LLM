# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kv_module
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.profiling import (
    KV_POOL_LOG_INTERVAL_ENV_VAR_NAME,
    PyExecutorProfileManager,
    load_kv_pool_log_interval,
)
from tensorrt_llm.metrics.logging import format_kv_pool_fields
from tensorrt_llm.metrics.perf import PerfMetricsCollector
from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheLevel, CacheTier

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


def _pool_group(index, layer_group_ids):
    variants = [SimpleNamespace(layer_group_id=g) for g in layer_group_ids]
    return SimpleNamespace(pool_group_index=index, slot_desc=SimpleNamespace(variants=variants))


def _executor(manager, is_v2=True):
    return Mock(
        kv_cache_manager=manager,
        _is_kv_manager_v2=is_v2,
        is_warmup=False,
        print_log=True,
        _perf_metrics=PerfMetricsCollector(),
    )


def _profiler(executor, interval, monkeypatch):
    monkeypatch.setenv(KV_POOL_LOG_INTERVAL_ENV_VAR_NAME, str(interval))
    profiler = PyExecutorProfileManager(executor)
    profiler._metrics_log_rank_enabled = True
    lines = []
    profiler.metrics_logger._write = lines.append
    return profiler, lines


def _record():
    return PerfMetricsCollector().collect_loop(
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
        }
    )


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


def _construction_logs(manager, monkeypatch):
    manager.kv_cache_manager_py_config = SimpleNamespace(
        layers=[SimpleNamespace(layer_id=0, buffers=[SimpleNamespace(role="K")])]
    )
    manager._format_kv_cache_pool_lifecycle_entry = Mock(return_value="role=K")
    log = Mock()
    monkeypatch.setattr(kv_module.logger, "info", log)
    manager._log_kv_cache_pool_lifecycle_mapping()
    return [c.args[0] for c in log.call_args_list]


def test_construction_log_prints_pool_sizes_and_equivalent_pool_ratio(monkeypatch):
    # Pool groups are numbered differently from layer groups; the ratio follows layer groups.
    manager = _manager(
        [CacheTier.GPU_MEM],
        {0: [_capacity(6, 0, 0), _capacity(1, 0, 0), _capacity(1, 0, 0, (GIB // 2, GIB // 2))]},
    )
    manager.impl.pool_group_descs = [
        _pool_group(0, [1]),
        _pool_group(1, [0]),
        _pool_group(2, [2]),
    ]
    lines = _construction_logs(manager, monkeypatch)
    assert lines[2] == (
        "pool_group_id=0, layer_group_ids=[1], slots=6, slot_bytes=1073741824, "
        "bytes=6442450944 (6.000 GiB), share=0.7500"
    )
    assert lines[-1] == (
        "GPU KV pools total 8.000 GiB; equivalent pool_ratio (layer_group order) = "
        "[0.1250, 0.7500, 0.1250]"
    )


def test_construction_log_skips_pool_ratio_for_shared_pool_groups(monkeypatch):
    manager = _manager([CacheTier.GPU_MEM], {0: [_capacity(2, 0, 0)]})
    manager.impl.pool_group_descs = [_pool_group(0, [0, 1])]
    lines = _construction_logs(manager, monkeypatch)
    assert lines[-1].startswith("pool_group_id=0, layer_group_ids=[0, 1], ")


def test_construction_log_includes_cold_tier_pool_mapping(monkeypatch):
    manager = _manager([CacheTier.GPU_MEM, CacheTier.HOST_MEM, CacheTier.DISK], {0: []})
    manager.impl.pool_group_descs = []
    manager.impl.get_life_cycle_pool_group_indices = Mock(return_value=[0, 1, 0])
    lines = _construction_logs(manager, monkeypatch)
    manager.impl.get_life_cycle_pool_group_indices.assert_called_once_with(CacheLevel(1))
    assert lines[-1] == "HOST_MEM pool_group_id -> layer_group_ids = {0: [0, 2], 1: [1]}"


def test_format_kv_pool_fields():
    text = format_kv_pool_fields({"gpu": [0.9, None], "host": None}, gpu_pool_gib=[1.5, 0.25])
    assert text == (
        "kv_cache_gpu_pool_gib = [0:1.500;1:0.250], "
        "kv_cache_gpu_pool_util = [0:0.900;1:N/A], "
        "kv_cache_host_pool_util = N/A, "
    )


def test_sampled_lines_read_pools_and_unsampled_lines_read_nothing(monkeypatch):
    manager = _manager(
        [CacheTier.GPU_MEM, CacheTier.HOST_MEM],
        {0: [_capacity(100, 5, 5)], 1: [_capacity(200, 50, 100)]},
    )
    executor = _executor(manager)  # The profile manager holds only a weakref.
    profiler, lines = _profiler(executor, 2, monkeypatch)
    for _ in range(4):
        profiler.expose_metrics(_record())
    assert len(lines) == 4
    assert "pool" not in lines[0] and "pool" not in lines[2]
    # The first sample also reports pool sizes; the pool fields follow kv_cache_util.
    assert (
        "kv_cache_util = N/A, kv_cache_gpu_pool_gib = [0:100.000], "
        "kv_cache_gpu_pool_util = [0:0.900], kv_cache_host_pool_util = [0:0.750], "
        "currank_total_requests"
    ) in lines[1]
    assert "pool_gib" not in lines[3] and "kv_cache_gpu_pool_util = [0:0.900]" in lines[3]
    # Two samples: GPU sizes once, then GPU and host utilization per sample.
    assert manager._get_storage_statistics.call_args_list == [call(0), call(0), call(1)] + [
        call(0),
        call(1),
    ]


@pytest.mark.parametrize(
    "interval,is_v2,is_warmup", [(0, True, False), (1, False, False), (1, True, True)]
)
def test_disabled_legacy_and_warmup_paths_read_no_pools(monkeypatch, interval, is_v2, is_warmup):
    manager = _manager([CacheTier.GPU_MEM], {0: [_capacity(100, 5, 5)]})
    executor = _executor(manager, is_v2=is_v2)
    executor.is_warmup = is_warmup
    profiler, lines = _profiler(executor, interval, monkeypatch)
    profiler.expose_metrics(_record())
    assert len(lines) == 1 and "pool" not in lines[0]
    manager._get_storage_statistics.assert_not_called()


@pytest.mark.parametrize("print_log,rank_enabled", [(False, True), (True, False)])
def test_non_logging_ranks_read_no_pools(monkeypatch, print_log, rank_enabled):
    manager = _manager([CacheTier.GPU_MEM], {0: [_capacity(100, 5, 5)]})
    executor = _executor(manager)
    executor.print_log = print_log
    profiler, lines = _profiler(executor, 1, monkeypatch)
    profiler._metrics_log_rank_enabled = rank_enabled
    profiler.expose_metrics(_record())
    assert lines == []
    manager._get_storage_statistics.assert_not_called()


def test_pool_fields_stay_out_of_the_collected_record(monkeypatch):
    manager = _manager([CacheTier.GPU_MEM], {0: [_capacity(100, 5, 5)]})
    executor = _executor(manager)  # The profile manager holds only a weakref.
    profiler, _ = _profiler(executor, 1, monkeypatch)
    record = _record()
    profiler.expose_metrics(record)
    assert not any("pool" in key.lower() for key in record.iteration_view())


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
