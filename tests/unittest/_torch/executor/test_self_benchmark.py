# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import types

import pytest

from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.self_benchmark import (
    BenchmarkPoint,
    BenchmarkPointResult,
    SelfBenchmark,
)
from tensorrt_llm.llmapi.llm_args import SelfBenchmarkConfig, TorchLlmArgs


class _FakeRequest:

    def __init__(self):
        self.is_dummy_request = True
        self.is_attention_dp_dummy = False
        self.is_cuda_graph_dummy = False
        self.is_self_benchmark_request = False
        self.py_self_benchmark_point_id = None

    @property
    def is_dummy(self):
        return (self.is_dummy_request or self.is_attention_dp_dummy
                or self.is_cuda_graph_dummy)


class _FakeKvStats:

    def __init__(self, tokens_per_block=32):
        self.tokens_per_block = tokens_per_block
        self.max_num_blocks = 128


class _FakeKvCacheManager:

    def __init__(self, tokens_per_block=32, enable_block_reuse=True):
        self.add_dummy_calls = []
        self._tokens_per_block = tokens_per_block
        self.enable_block_reuse = enable_block_reuse

    def add_dummy_requests(self, **kwargs):
        self.add_dummy_calls.append(kwargs)
        return [_FakeRequest() for _ in kwargs["request_ids"]]

    def get_kv_cache_stats(self):
        return _FakeKvStats(tokens_per_block=self._tokens_per_block)


class _FakeResourceManager:

    def __init__(self, kv_cache_manager):
        self._kv_cache_manager = kv_cache_manager

    def get_resource_manager(self, resource_manager_type):
        if resource_manager_type == ResourceManagerType.KV_CACHE_MANAGER:
            return self._kv_cache_manager
        return None


class _FakeScheduledBatch:

    def __init__(self, requests):
        self._requests = requests

    def all_requests(self):
        return self._requests


def _make_executor(config: SelfBenchmarkConfig,
                   tokens_per_block=32,
                   enable_block_reuse=True) -> types.SimpleNamespace:
    kv_cache_manager = _FakeKvCacheManager(
        tokens_per_block=tokens_per_block,
        enable_block_reuse=enable_block_reuse)
    return types.SimpleNamespace(
        llm_args=types.SimpleNamespace(self_benchmark_config=config),
        max_input_len=16,
        max_seq_len=9,
        max_num_tokens=8,
        max_num_active_requests=4,
        max_batch_size=4,
        max_total_draft_tokens=0,
        max_beam_width=1,
        model_engine=types.SimpleNamespace(max_draft_loop_tokens=0,
                                           use_mrope=False),
        resource_manager=_FakeResourceManager(kv_cache_manager),
        dist=types.SimpleNamespace(rank=0),
    )


def test_torch_llm_args_self_benchmark_enables_iter_stats():
    args = TorchLlmArgs(model="dummy",
                        self_benchmark_config=SelfBenchmarkConfig())

    assert args.enable_iter_perf_stats is True


def test_torch_llm_args_self_benchmark_rejects_encode_only():
    with pytest.raises(ValueError, match="decoder generation workloads"):
        TorchLlmArgs(model="dummy",
                     encode_only=True,
                     self_benchmark_config=SelfBenchmarkConfig())


def test_torch_llm_args_self_benchmark_rejects_attention_dp():
    with pytest.raises(ValueError, match="attention data parallelism"):
        TorchLlmArgs(model="dummy",
                     enable_attention_dp=True,
                     self_benchmark_config=SelfBenchmarkConfig())


def test_grid_generation_uses_executor_limits():
    config = SelfBenchmarkConfig(
        mode="agg",
        prefill_isl_granularity=2,
        decode_context_granularity=2,
        decode_batch_granularity=2,
        warmup_iterations=1,
    )

    benchmark = SelfBenchmark(_make_executor(config))

    assert [point.point_type for point in benchmark._grid] == [
        "warmup",
        "prefill",
        "prefill",
        "decode",
        "decode",
        "decode",
        "decode",
    ]
    assert [point.isl for point in benchmark._grid
            if point.point_type == "prefill"] == [1, 8]
    assert [point.kv_read_tokens for point in benchmark._grid
            if point.point_type == "prefill"] == [0, 0]
    assert {(point.context_length, point.batch_size)
            for point in benchmark._grid if point.point_type == "decode"} == {
                (1, 1),
                (1, 4),
                (7, 1),
                (7, 4),
            }


def test_prefill_grid_includes_block_aligned_kv_read_axis():
    config = SelfBenchmarkConfig(
        mode="prefill",
        prefill_isl_granularity=2,
        prefill_kv_read_granularity=4,
        warmup_iterations=0,
    )

    benchmark = SelfBenchmark(_make_executor(config, tokens_per_block=4))

    assert [(point.point_type, point.isl, point.kv_read_tokens)
            for point in benchmark._grid] == [
                ("prefill", 1, 0),
                ("prefill", 8, 0),
                ("prefill_seed", 4, 4),
                ("prefill", 8, 4),
            ]
    assert benchmark._grid[2].cache_salt_id == benchmark._grid[
        3].cache_salt_id


def test_prefill_grid_omits_kv_read_axis_when_block_reuse_disabled():
    config = SelfBenchmarkConfig(
        mode="prefill",
        prefill_isl_granularity=2,
        prefill_kv_read_granularity=4,
        warmup_iterations=0,
    )

    benchmark = SelfBenchmark(
        _make_executor(config, tokens_per_block=4, enable_block_reuse=False))

    assert [(point.point_type, point.isl, point.kv_read_tokens)
            for point in benchmark._grid] == [
                ("prefill", 1, 0),
                ("prefill", 8, 0),
            ]


def test_prefill_queue_item_marks_executor_request():
    config = SelfBenchmarkConfig(mode="prefill",
                                 prefill_isl_granularity=1,
                                 warmup_iterations=0)
    benchmark = SelfBenchmark(_make_executor(config))

    items = benchmark.make_prefill_queue_items([], [])

    assert len(items) == 1
    assert items[0].id == 900_000_000
    assert items[0].request.py_is_self_benchmark_request is True
    assert items[0].request.py_self_benchmark_point_id == 0


def test_prefill_seed_and_measure_requests_share_cache_salt():
    config = SelfBenchmarkConfig(mode="prefill", warmup_iterations=0)
    benchmark = SelfBenchmark(_make_executor(config))
    benchmark._grid = [
        BenchmarkPoint(point_type="prefill_seed",
                       index=0,
                       isl=4,
                       kv_read_tokens=4,
                       cache_salt_id=123),
        BenchmarkPoint(point_type="prefill",
                       index=1,
                       isl=8,
                       kv_read_tokens=4,
                       cache_salt_id=123),
    ]

    seed_items = benchmark.make_prefill_queue_items([], [])
    benchmark._finish_current_point()
    measure_items = benchmark.make_prefill_queue_items([], [])

    assert seed_items[0].request.input_token_ids == [1, 1, 1, 1]
    assert seed_items[0].request.cache_salt_id == 123
    assert measure_items[0].request.input_token_ids == [1] * 8
    assert measure_items[0].request.cache_salt_id == 123


def test_decode_injection_uses_dummy_kv_path():
    config = SelfBenchmarkConfig(mode="decode",
                                 decode_context_granularity=1,
                                 decode_batch_granularity=1,
                                 warmup_iterations=0)
    executor = _make_executor(config)
    benchmark = SelfBenchmark(executor)

    requests = benchmark.make_decode_requests([], [])

    assert len(requests) == 4
    assert all(request.is_self_benchmark_request for request in requests)
    add_dummy_call = executor.resource_manager._kv_cache_manager.add_dummy_calls[
        0]
    assert add_dummy_call["request_ids"] == [
        900_000_000,
        900_000_001,
        900_000_002,
        900_000_003,
    ]
    assert add_dummy_call["token_nums"] == [8, 8, 8, 8]
    assert add_dummy_call["is_gen"] is True


def test_observe_iteration_sanitizes_queue_counters_and_records_result():
    config = SelfBenchmarkConfig(mode="prefill", warmup_iterations=0)
    benchmark = SelfBenchmark(_make_executor(config))
    point = BenchmarkPoint(point_type="prefill", index=0, isl=4)
    benchmark._current = BenchmarkPointResult(point=point)
    request = types.SimpleNamespace(is_self_benchmark_request=True,
                                    is_dummy=False)
    stats = {
        "numQueuedRequests": 3,
        "inflightBatchingStats": {
            "numContextRequests": 1,
            "numQueuedContextRequests": 3,
            "numQueuedCtxTokens": 128,
            "numQueuedGenRequests": 2,
            "numQueuedGenKvTokens": 256,
        },
    }

    consumed = benchmark.observe_iteration(_FakeScheduledBatch([request]),
                                           stats)

    assert consumed is True
    assert benchmark._current is None
    assert len(benchmark._results) == 1
    recorded_stats = benchmark._results[0].stats[0]
    assert recorded_stats["numQueuedRequests"] == 0
    assert recorded_stats["inflightBatchingStats"][
        "numQueuedContextRequests"] == 0
    assert recorded_stats["inflightBatchingStats"]["numQueuedCtxTokens"] == 0
    assert recorded_stats["inflightBatchingStats"]["numQueuedGenRequests"] == 0
    assert recorded_stats["inflightBatchingStats"]["numQueuedGenKvTokens"] == 0


def test_observe_iteration_records_cache_hit_validation():
    config = SelfBenchmarkConfig(mode="prefill", warmup_iterations=0)
    benchmark = SelfBenchmark(_make_executor(config))
    point = BenchmarkPoint(point_type="prefill",
                           index=0,
                           isl=8,
                           kv_read_tokens=4)
    benchmark._current = BenchmarkPointResult(point=point)
    request = types.SimpleNamespace(is_self_benchmark_request=True,
                                    py_self_benchmark_point_id=0,
                                    cached_tokens=4,
                                    is_dummy=False)
    stats = {
        "numQueuedRequests": 0,
        "inflightBatchingStats": {
            "numContextRequests": 1,
        },
    }

    consumed = benchmark.observe_iteration(_FakeScheduledBatch([request]),
                                           stats)

    assert consumed is True
    result = benchmark._results[0]
    assert result.observed_kv_read_tokens == 4
    assert result.cache_hit_validated is True
    assert result.stats[0]["selfBenchmark"] == {
        "expectedKvReadTokens": 4,
        "observedCachedTokens": 4,
        "cacheHitValidated": True,
    }


def test_write_output(tmp_path):
    output_path = tmp_path / "benchmark.json"
    config = SelfBenchmarkConfig(mode="prefill",
                                 output_path=str(output_path),
                                 warmup_iterations=0)
    benchmark = SelfBenchmark(_make_executor(config))
    benchmark._results.append(
        BenchmarkPointResult(
            point=BenchmarkPoint(point_type="prefill", index=0, isl=8),
            stats=[{
                "inflightBatchingStats": {
                    "numContextRequests": 1
                }
            }],
            observed_kv_read_tokens=0,
            cache_hit_validated=True,
        ))

    benchmark.write_output()

    with open(output_path) as f:
        data = json.load(f)
    assert data["config"]["mode"] == "prefill"
    assert data["timed_out"] is False
    assert data["limits"]["max_num_scheduled_tokens"] == 8
    assert data["limits"]["tokens_per_block"] == 32
    assert data["results"][0]["point"]["isl"] == 8
    assert data["results"][0]["observed_kv_read_tokens"] == 0
    assert data["results"][0]["cache_hit_validated"] is True
    assert data["results"][0]["iteration_stats"][0][
        "inflightBatchingStats"]["numContextRequests"] == 1
