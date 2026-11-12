# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collection/exposure parity without a GPU, model, source drain or logger scrape."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from tensorrt_llm.metrics.collector import MetricsCollector, PrometheusExposure
from tensorrt_llm.metrics.enums import MetricNames
from tensorrt_llm.metrics.logging import LoggerExposure
from tensorrt_llm.metrics.perf import PerfMetricsCollector, PerfMetricsRecord, kv_cache_utilization

pytestmark = pytest.mark.cpu_only


def _loop(iteration: int = 0, fetched: int = 3) -> dict:
    return {
        "iter": iteration,
        "rank": 0,
        "globalRank": 4,
        "numScheduledRequests": 2,
        "adpDummyCtxTokens": 8,
        "adpDummyGenTokens": 1,
        "numFetchedRequestsTotal": fetched,
        "numFetchedRequestsAllRanks": 10,
        "hostStepTimeMS": 2.5,
        "prevDeviceStepTimeMS": None,
        "states": "decode",
        "numActiveRequests": 3,
        "numDummyRequests": 1,
        "numUnscheduledRequests": 1,
        "timestamp": 1000.0,
    }


def _prometheus() -> MetricsCollector:
    return MetricsCollector({"model_name": "test"}, registry=CollectorRegistry())


def test_direct_record_construction_owns_mutable_category_values() -> None:
    values = {"kvCacheStats": {"maxNumBlocks": 10, "usedNumBlocks": 3}}
    record = PerfMetricsRecord("rank", per_rank=values)
    values["kvCacheStats"]["usedNumBlocks"] = 9
    assert record.per_rank["kvCacheStats"]["usedNumBlocks"] == 3


def test_categories_own_the_source_and_preserve_legacy_wire_shape() -> None:
    source = {
        "iter": 7,
        "attentionDpRank": 2,
        "hostStepTimeMS": 2.0,
        "futureMeasurement": [1, 2],
        "requestStats": [{"id": 9, "numGeneratedTokens": 4}],
        "kvCacheStats": {"usedNumBlocks": 2, "maxNumBlocks": 10},
        "kvCacheIterationStatsByPoolGroup": [{"slotSize": [64]}],
    }
    record = PerfMetricsCollector.collect_iteration(source)
    assert record.iteration_stats() == source
    assert json.loads(record.iteration_json()) == source
    with pytest.raises(TypeError):
        record.iteration_view()["kvCacheStats"]["usedNumBlocks"] = 99
    assert "requestStats" in record.per_req
    assert "hostStepTimeMS" in record.per_iter
    assert "kvCacheStats" in record.per_rank
    assert record.per_iter["attentionDpRank"] == 2
    source["kvCacheStats"]["usedNumBlocks"] = 9
    source["kvCacheIterationStatsByPoolGroup"][0]["slotSize"].append(128)
    source["requestStats"][0]["numGeneratedTokens"] = 99
    assert record.per_rank["kvCacheStats"]["usedNumBlocks"] == 2
    assert record.per_rank["kvCacheIterationStatsByPoolGroup"][0]["slotSize"] == (64,)
    assert record.per_req["requestStats"][0]["numGeneratedTokens"] == 4
    with pytest.raises(TypeError):
        record.per_rank["kvCacheStats"]["usedNumBlocks"] = 20
    wire = record.iteration_stats()
    wire["kvCacheStats"]["usedNumBlocks"] = 20
    assert record.per_rank["kvCacheStats"]["usedNumBlocks"] == 2


def test_logger_and_prometheus_share_one_collection_without_extra_source_reads() -> None:
    collector = PerfMetricsCollector()
    reader = Mock(return_value=SimpleNamespace(max_num_blocks=10, free_num_blocks=4))
    record = collector.collect_loop(_loop(), read_kv_cache=reader)
    reader.assert_called_once_with()
    messages = []
    logger = LoggerExposure(messages.append)
    prom = _prometheus()
    logger.emit(record)
    prom.emit(record)
    reader.assert_called_once_with()
    assert "kv_cache_util = 0.600" in messages[0]
    assert "num_fetched = 3" in messages[0]
    assert "prev_device_step_time = N/A" in messages[0]
    assert json.loads(messages[1])["per_rank"]["captureIter"] == 0
    assert kv_cache_utilization(record.per_rank["kvCacheStats"]) == pytest.approx(0.6)
    assert prom.kv_cache_utilization.labels(
        **prom.labels, dp_rank="0"
    )._value.get() == pytest.approx(0.6)
    logger.emit(record)
    prom.emit(record)
    # Cached rank snapshots are not re-exposed by the logger.
    assert len(messages) == 3


def test_batch_snapshots_replace_logger_polling_and_retain_capture_identity() -> None:
    collector = PerfMetricsCollector()
    rank = collector.collect_rank(
        {"kvCacheStats": {"maxNumBlocks": 8, "freeNumBlocks": 2, "usedNumBlocks": 6}},
        capture_iter=7,
    )
    loop = collector.collect_loop(_loop(9))
    assert loop.per_rank is rank.per_rank
    assert loop.per_rank["captureIter"] == 7
    assert loop.per_iter["iter"] == 9
    assert loop.per_iter["loopSequence"] == 0
    messages = []
    exposure = LoggerExposure(messages.append)
    exposure.emit(rank)
    exposure.emit(loop)
    assert len(messages) == 2
    assert "kv_cache_util = 0.750" in messages[1]


def test_logger_only_kv_refresh_uses_loop_sequence_when_iteration_counter_stalls() -> None:
    collector = PerfMetricsCollector()
    reader = Mock(return_value=SimpleNamespace(max_num_blocks=10, free_num_blocks=10))
    for _ in range(17):
        collector.collect_loop(_loop(0), read_kv_cache=reader, kv_refresh_interval=16)
    assert reader.call_count == 2
    assert collector.latest_loop.per_iter["iter"] == 0
    assert collector.latest_loop.per_iter["loopSequence"] == 16


def test_request_candidates_keep_enum_keys_and_diagnostic_requests_are_not_completions() -> None:
    source = {
        "finished_reason": "stop",
        MetricNames.REQUEST_QUEUE_TIME: 0.0,
        MetricNames.SPEC_DEC_DRAFTED_PER_POS: [2, 1],
    }
    request = PerfMetricsCollector.collect_request(source)
    assert request.request_metrics() == source
    source[MetricNames.SPEC_DEC_DRAFTED_PER_POS].append(99)
    assert request.per_req[MetricNames.SPEC_DEC_DRAFTED_PER_POS] == (2, 1)
    messages = []
    LoggerExposure(messages.append).emit(request)
    assert json.loads(messages[0])["per_req"]["spec_dec_drafted_per_pos"] == [2, 1]
    prom = _prometheus()
    diagnostic = PerfMetricsCollector.collect_iteration({"requestStats": [{"id": 9}]})
    prom.emit(diagnostic)
    assert prom.counter_request_success.collect()[0].samples == []
    prom.emit(request)
    assert prom.histogram_queue_time_request.labels(**prom.labels)._sum.get() == 0
    assert (
        prom.counter_request_success.labels(**prom.labels, finished_reason="stop")._value.get() == 1
    )


def test_loop_identity_is_independent_of_delayed_batch_and_fanout_rank() -> None:
    runtime = PerfMetricsCollector()
    loop = runtime.collect_loop(_loop(9, fetched=8))
    batch = PerfMetricsCollector.collect_iteration(
        {
            "iter": 7,
            "attentionDpRank": 0,
            "executorLoop": {"kind": "loop", "per_iter": dict(loop.per_iter)},
        }
    )
    prom = _prometheus()
    prom.emit(batch)
    assert batch.per_iter["iter"] == 7
    assert batch.per_iter["executorLoop"]["per_iter"]["iter"] == 9
    # A peer row without a measured loop must not create a rank-1 zero placeholder.
    prom.emit(
        PerfMetricsCollector.collect_iteration({"attentionDpRank": 1, "numActiveRequests": 99})
    )
    assert prom.num_requests_running.collect()[0].samples == []


def test_prometheus_endpoint_uses_collected_state_without_a_source_or_logger() -> None:
    prom = _prometheus()
    assert isinstance(prom, PrometheusExposure)
    prom.emit(PerfMetricsCollector.collect_iteration({"numActiveRequests": 3}))
    first = generate_latest(prom._registry)
    second = generate_latest(prom._registry)
    assert b'trtllm_num_requests_running{model_name="test"} 3.0' in first
    assert first == second


def test_cached_rank_observations_are_not_replayed_by_multiple_loops() -> None:
    runtime = PerfMetricsCollector()
    rank = runtime.collect_rank(
        {
            "rank": 0,
            "globalRank": 4,
            "kvCacheStats": {"maxNumBlocks": 10, "usedNumBlocks": 4},
        },
        capture_iter=0,
    )
    prom = _prometheus()
    observe = Mock(wraps=prom._expose_iteration_stats)
    prom._expose_iteration_stats = observe
    prom.emit(rank)
    for iteration in range(3):
        prom.emit(runtime.collect_loop(_loop(iteration)))
    rank_calls = [call for call in observe.call_args_list if "kvCacheStats" in call.args[0]]
    assert len(rank_calls) == 1
    assert prom.kv_cache_utilization.labels(
        **prom.labels, dp_rank="0"
    )._value.get() == pytest.approx(0.4)
