# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import types

import pytest

import tensorrt_llm._torch.pyexecutor.self_benchmark as self_benchmark_module
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.self_benchmark import (
    BenchmarkPoint,
    BenchmarkPointResult,
    SelfBenchmark,
)
from tensorrt_llm.llmapi.llm_args import SelfBenchmarkConfig, TorchLlmArgs


class _FakeLlmRequest:
    """Stand-in for the LlmRequest produced by executor_request_to_llm_request.

    The real conversion needs torch/bindings (unavailable on CPU-only CI), so
    tests monkeypatch executor_request_to_llm_request with _fake_to_llm_request
    below. The fake just carries the deterministic per-rank inputs we assert on.
    """

    def __init__(self, req_id, executor_request):
        self.py_request_id = req_id
        self.executor_request = executor_request
        self.input_token_ids = list(executor_request.input_token_ids)
        self.cache_salt = executor_request.cache_salt
        self.is_self_benchmark_request = False
        self.py_self_benchmark_point_id = None


def _fake_to_llm_request(req_id, executor_request, child_req_ids,
                         exclude_last_generation_logits, **kwargs):
    return _FakeLlmRequest(req_id, executor_request)


@pytest.fixture(autouse=True)
def _patch_executor_request_to_llm_request(monkeypatch):
    monkeypatch.setattr(self_benchmark_module, "executor_request_to_llm_request",
                        _fake_to_llm_request)


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


def test_torch_llm_args_self_benchmark_accepts_pure_tp():
    # Per-rank deterministic replication makes pure tensor parallelism safe:
    # every TP rank builds the same forward batch in lockstep.
    args = TorchLlmArgs(model="dummy",
                        tensor_parallel_size=2,
                        self_benchmark_config=SelfBenchmarkConfig())
    assert args.parallel_config.tp_size == 2
    assert args.enable_iter_perf_stats is True


def test_torch_llm_args_self_benchmark_rejects_pipeline_parallel():
    with pytest.raises(ValueError, match="pipeline parallelism"):
        TorchLlmArgs(model="dummy",
                     pipeline_parallel_size=2,
                     self_benchmark_config=SelfBenchmarkConfig())


def test_torch_llm_args_self_benchmark_rejects_context_parallel():
    with pytest.raises(ValueError, match="context parallelism"):
        TorchLlmArgs(model="dummy",
                     context_parallel_size=2,
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


def test_prefill_request_built_per_rank_marks_llm_request():
    config = SelfBenchmarkConfig(mode="prefill",
                                 prefill_isl_granularity=1,
                                 warmup_iterations=0)
    benchmark = SelfBenchmark(_make_executor(config))

    requests = benchmark.make_prefill_requests([], [])

    # Built locally per-rank (no RequestQueueItem / broadcast). Deterministic
    # request id and benchmark marking are set on the LlmRequest itself.
    assert len(requests) == 1
    assert requests[0].py_request_id == 900_000_000
    assert requests[0].is_self_benchmark_request is True
    assert requests[0].py_self_benchmark_point_id == 0


def test_prefill_request_is_rank_independent():
    """Per-rank lockstep: non-rank-0 builds the same prefill request as rank 0.

    This guards the core TP invariant -- prefill injection must not depend on
    being rank 0. Every input is a deterministic function of the grid point, so
    a rank-1 executor produces a bit-identical request id / tokens / cache_salt.
    """
    config = SelfBenchmarkConfig(mode="prefill",
                                 prefill_isl_granularity=1,
                                 warmup_iterations=0)
    rank0 = SelfBenchmark(_make_executor(config))
    executor_rank1 = _make_executor(config)
    executor_rank1.dist = types.SimpleNamespace(rank=1)
    rank1 = SelfBenchmark(executor_rank1)

    req0 = rank0.make_prefill_requests([], [])[0]
    req1 = rank1.make_prefill_requests([], [])[0]

    assert req0.py_request_id == req1.py_request_id
    assert req0.input_token_ids == req1.input_token_ids
    assert req0.cache_salt == req1.cache_salt
    assert req0.py_self_benchmark_point_id == req1.py_self_benchmark_point_id


def test_shutdown_finishes_benchmark_without_starting_next_point(tmp_path):
    output_path = tmp_path / "benchmark.json"
    config = SelfBenchmarkConfig(mode="prefill",
                                 output_path=str(output_path),
                                 warmup_iterations=0)
    executor = _make_executor(config)
    executor.is_shutdown = True
    benchmark = SelfBenchmark(executor)

    assert benchmark.make_prefill_requests([], []) == []
    assert benchmark.active is False
    assert output_path.exists()


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

    seed_items = benchmark.make_prefill_requests([], [])
    benchmark._finish_current_point()
    measure_items = benchmark.make_prefill_requests([], [])

    assert seed_items[0].input_token_ids == [1, 1, 1, 1]
    assert seed_items[0].cache_salt == "123"
    assert measure_items[0].input_token_ids == [1] * 8
    assert measure_items[0].cache_salt == "123"


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


def test_observe_iteration_records_cache_hit_mismatch_without_skip():
    config = SelfBenchmarkConfig(mode="prefill", warmup_iterations=0)
    benchmark = SelfBenchmark(_make_executor(config))
    point = BenchmarkPoint(point_type="prefill",
                           index=0,
                           isl=8,
                           kv_read_tokens=4)
    benchmark._current = BenchmarkPointResult(point=point)
    request = types.SimpleNamespace(is_self_benchmark_request=True,
                                    py_self_benchmark_point_id=0,
                                    cached_tokens=0,
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
    assert result.skipped_reason is None
    assert result.observed_kv_read_tokens == 0
    assert result.cache_hit_validated is False
    assert result.stats[0]["selfBenchmark"] == {
        "expectedKvReadTokens": 4,
        "observedCachedTokens": 0,
        "cacheHitValidated": False,
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
    assert data["status"] == "complete"
    assert data["valid"] is True
    assert data["limits"]["max_num_scheduled_tokens"] == 8
    assert data["limits"]["tokens_per_block"] == 32
    assert data["results"][0]["point"]["isl"] == 8
    assert data["results"][0]["observed_kv_read_tokens"] == 0
    assert data["results"][0]["cache_hit_validated"] is True
    assert data["results"][0]["iteration_stats"][0][
        "inflightBatchingStats"]["numContextRequests"] == 1
