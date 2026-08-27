# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import types
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock

import pytest

import tensorrt_llm._torch.pyexecutor.self_benchmark as self_benchmark_module
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.self_benchmark import (
    BenchmarkCase,
    BenchmarkOutcome,
    BenchmarkTrialResult,
    DrainTarget,
    RunState,
    SelfBenchmark,
    TrialState,
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
        self.request_id = req_id
        self.executor_request = executor_request
        self.input_token_ids = list(executor_request.input_token_ids)
        self.cache_salt = executor_request.cache_salt
        self.cached_tokens = 0
        self.estimated_reusable_tokens = 0
        self.context_chunk_size = len(self.input_token_ids)
        self.prompt_len = max(0, len(self.input_token_ids) - 1)
        self.is_dummy_request = False
        self.is_attention_dp_dummy = False
        self.is_cuda_graph_dummy = False
        self.is_self_benchmark_request = False
        self.py_self_benchmark_trial_id = None
        self.num_draft_tokens = 0
        self.max_new_tokens = int(getattr(executor_request, "max_tokens", 1))
        self.py_beam_width = int(executor_request.sampling_config.beam_width)

    def get_num_tokens(self, beam):
        return len(self.input_token_ids)

    def get_beam_width_by_iter(self, for_next_iteration=False):
        del for_next_iteration
        return self.py_beam_width

    @property
    def is_dummy(self):
        return self.is_dummy_request or self.is_attention_dp_dummy or self.is_cuda_graph_dummy


def _fake_to_llm_request(
    req_id, executor_request, child_req_ids, exclude_last_generation_logits, **kwargs
):
    return _FakeLlmRequest(req_id, executor_request)


@pytest.fixture(autouse=True)
def _patch_executor_request_to_llm_request(monkeypatch):
    monkeypatch.setattr(
        self_benchmark_module, "executor_request_to_llm_request", _fake_to_llm_request
    )


class _FakeRequest:
    def __init__(
        self,
        request_id=900_000_000,
        *,
        total_tokens=1,
        cached_tokens=0,
        is_dummy_request=False,
        beam_width=1,
        draft_tokens=0,
        max_new_tokens=1,
    ):
        self.py_request_id = request_id
        self.request_id = request_id
        self._total_tokens = total_tokens
        self.cached_tokens = cached_tokens
        self.estimated_reusable_tokens = cached_tokens
        self.context_chunk_size = total_tokens
        self.prompt_len = max(0, total_tokens - 1)
        self.is_dummy_request = is_dummy_request
        self.is_attention_dp_dummy = False
        self.is_cuda_graph_dummy = False
        self.is_self_benchmark_request = False
        self.py_self_benchmark_trial_id = None
        self.py_beam_width = beam_width
        self.num_draft_tokens = draft_tokens
        self.max_new_tokens = max_new_tokens

    def get_num_tokens(self, beam):
        return self._total_tokens

    def get_beam_width_by_iter(self, for_next_iteration=False):
        del for_next_iteration
        return self.py_beam_width

    @property
    def is_dummy(self):
        return self.is_dummy_request or self.is_attention_dp_dummy or self.is_cuda_graph_dummy


class _FakeKvStats:
    def __init__(self, tokens_per_block=32, free_num_blocks=128):
        self.tokens_per_block = tokens_per_block
        self.max_num_blocks = 128
        self.free_num_blocks = free_num_blocks
        self.used_num_blocks = self.max_num_blocks - free_num_blocks
        self.num_free_blocks_per_window_size = {128: free_num_blocks}


class _FakeKvImpl:
    def __init__(self, capacity_limit):
        self.capacity_limit = capacity_limit
        self.capacity_probe_calls = []

    def get_kv_cache_stats(self):
        return _FakeKvStats(free_num_blocks=self.capacity_limit)


class _FakeCapacityImpl:
    def __init__(self):
        self.calls = []

    def __call__(self, requests, kv_cache_manager, peft_cache_manager, cross_kv_cache_manager):
        del peft_cache_manager, cross_kv_cache_manager
        copied = list(requests)
        kv_cache_manager.capacity_probe_calls.append(copied)
        self.calls.append((copied, kv_cache_manager))
        fitting = copied[: kv_cache_manager.capacity_limit]
        return fitting, [], []


class _FakeCapacityScheduler:
    def __init__(self, kv_cache_manager):
        self.kv_cache_manager = kv_cache_manager
        self.impl = _FakeCapacityImpl()
        self.calls = []

    def schedule_request(self, requests):
        copied = list(requests)
        self.calls.append(copied)
        return self.impl(copied, self.kv_cache_manager, None, None)


class _FakeMicroBatchScheduler:
    def __init__(self, max_batch_size, max_num_tokens):
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.impl = object()
        self.calls = []
        self.inflight_calls = []

    def schedule(self, requests, inflight_request_ids):
        self.inflight_calls.append(inflight_request_ids)
        copied = list(requests)
        self.calls.append(copied)
        contexts = []
        generation = []
        used_tokens = 0
        for request in copied:
            if request.is_dummy_request:
                cost = request.get_beam_width_by_iter(False) + request.num_draft_tokens
                destination = generation
            else:
                base_tokens = request.get_num_tokens(0)
                reusable = min(request.estimated_reusable_tokens, base_tokens)
                cost = max(1, base_tokens - reusable + request.num_draft_tokens)
                destination = contexts
            if used_tokens + cost > self.max_num_tokens:
                break
            destination.append(request)
            used_tokens += cost
            if len(contexts) + len(generation) >= self.max_batch_size:
                break
        return [], contexts, generation


class _FakeScheduler:
    def __init__(self, kv_cache_manager, max_batch_size, max_num_tokens):
        self.capacity_scheduler = _FakeCapacityScheduler(kv_cache_manager.impl)
        self.micro_batch_scheduler = _FakeMicroBatchScheduler(max_batch_size, max_num_tokens)


class _FakeKvCacheManager:
    def __init__(
        self,
        tokens_per_block=32,
        enable_block_reuse=True,
        capacity_limit=128,
        num_extra_kv_tokens=0,
        kv_reserve_draft_tokens=0,
    ):
        self.add_dummy_calls = []
        self._tokens_per_block = tokens_per_block
        self.enable_block_reuse = enable_block_reuse
        self.impl = _FakeKvImpl(capacity_limit)
        self.num_extra_kv_tokens = num_extra_kv_tokens
        self._kv_reserve_draft_tokens = kv_reserve_draft_tokens

    def add_dummy_requests(self, **kwargs):
        self.add_dummy_calls.append(kwargs)
        return [
            _FakeRequest(
                request_id,
                total_tokens=token_num,
                is_dummy_request=True,
                beam_width=kwargs["max_beam_width"],
                draft_tokens=kwargs["max_num_draft_tokens"],
            )
            for request_id, token_num in zip(kwargs["request_ids"], kwargs["token_nums"])
        ]

    def get_kv_cache_stats(self):
        return _FakeKvStats(
            tokens_per_block=self._tokens_per_block,
            free_num_blocks=self.impl.capacity_limit,
        )


class _FakeResourceManager:
    def __init__(self, kv_cache_manager, draft_kv_cache_manager=None):
        self._kv_cache_manager = kv_cache_manager
        self._draft_kv_cache_manager = draft_kv_cache_manager

    def get_resource_manager(self, resource_manager_type):
        if resource_manager_type == ResourceManagerType.KV_CACHE_MANAGER:
            return self._kv_cache_manager
        if resource_manager_type == ResourceManagerType.DRAFT_KV_CACHE_MANAGER:
            return self._draft_kv_cache_manager
        return None


class _FakeScheduledBatch:
    def __init__(
        self,
        requests=None,
        *,
        chunking=None,
        last_chunk=None,
        generation=None,
        other=None,
    ):
        self.context_requests_chunking = list(chunking or [])
        self.context_requests_last_chunk = list(last_chunk or [])
        self.generation_requests = list(generation or [])
        self._other = list(other if other is not None else requests or [])

    @property
    def context_requests(self):
        return self.context_requests_chunking + self.context_requests_last_chunk

    @property
    def batch_size(self):
        return len(self.all_requests())

    @property
    def num_encoder_requests(self):
        return 0

    @property
    def num_context_requests(self):
        return len(self.context_requests)

    @property
    def num_generation_requests(self):
        return len(self.generation_requests)

    def all_requests(self):
        return self.context_requests + self.generation_requests + self._other


def _make_executor(
    config: SelfBenchmarkConfig,
    tokens_per_block=32,
    enable_block_reuse=True,
    *,
    max_num_tokens=8,
    max_batch_size=4,
    max_beam_width=1,
    max_total_draft_tokens=0,
    target_kv_capacity=128,
    draft_kv_capacity=None,
) -> types.SimpleNamespace:
    kv_cache_manager = _FakeKvCacheManager(
        tokens_per_block=tokens_per_block,
        enable_block_reuse=enable_block_reuse,
        capacity_limit=target_kv_capacity,
        kv_reserve_draft_tokens=max_total_draft_tokens,
    )
    draft_kv_cache_manager = (
        None
        if draft_kv_capacity is None
        else _FakeKvCacheManager(
            tokens_per_block=tokens_per_block,
            enable_block_reuse=False,
            capacity_limit=draft_kv_capacity,
            kv_reserve_draft_tokens=max_total_draft_tokens,
        )
    )
    scheduler = _FakeScheduler(kv_cache_manager, max_batch_size, max_num_tokens)
    return types.SimpleNamespace(
        llm_args=types.SimpleNamespace(self_benchmark_config=config),
        max_input_len=16,
        max_seq_len=9,
        max_num_tokens=max_num_tokens,
        max_num_active_requests=max_batch_size,
        max_batch_size=max_batch_size,
        max_total_draft_tokens=max_total_draft_tokens,
        max_beam_width=max_beam_width,
        model_engine=types.SimpleNamespace(
            max_draft_loop_tokens=max_total_draft_tokens, use_mrope=False
        ),
        resource_manager=_FakeResourceManager(kv_cache_manager, draft_kv_cache_manager),
        scheduler=scheduler,
        inflight_req_ids=object(),
        dist=types.SimpleNamespace(rank=0),
    )


def _start_prefill_trial(batch_size=2, constructed_count=None, *, isl=4, kv_read_tokens=0):
    config = SelfBenchmarkConfig(mode="prefill", warmup_iterations=0)
    benchmark = SelfBenchmark(_make_executor(config))
    case = BenchmarkCase(
        case_type="prefill",
        case_id=0,
        isl=isl,
        kv_read_tokens=kv_read_tokens,
        batch_size=batch_size,
    )
    count = batch_size if constructed_count is None else constructed_count
    requests = [
        _FakeRequest(
            900_000_000 + offset,
            total_tokens=isl,
            cached_tokens=kv_read_tokens,
        )
        for offset in range(count)
    ]
    for request in requests:
        # Bound V1 validates admission before resource preparation, while the
        # reusable prefix is still represented separately from the full chunk.
        request.context_chunk_size = isl
        request.is_self_benchmark_request = True
        request.py_self_benchmark_trial_id = 0
    benchmark.start_trial(case, requests)
    return benchmark, requests


def _start_decode_trial(batch_size=2, *, context_length=4):
    config = SelfBenchmarkConfig(
        mode="decode",
        decode_context_granularity=1,
        decode_batch_granularity=1,
        warmup_iterations=0,
    )
    benchmark = SelfBenchmark(_make_executor(config))
    case = BenchmarkCase(
        case_type="decode",
        case_id=0,
        context_length=context_length,
        batch_size=batch_size,
    )
    benchmark._cases = [case]
    requests = benchmark.make_decode_requests([], [])
    return benchmark, requests


def _prefill_stats(requests, tokens):
    return {
        "inflightBatchingStats": {
            "numContextRequests": requests,
            "numCtxTokens": tokens,
            "numGenRequests": 0,
        },
        "gpuForwardTimeMS": 1.0,
    }


def _complete_current_prefill_trial(benchmark, requests):
    scheduled = _FakeScheduledBatch(last_chunk=requests)
    benchmark.observe_scheduled_batch(scheduled)
    benchmark.observe_executed_batch(
        scheduled,
        _prefill_stats(
            requests=len(requests),
            tokens=sum(request.context_chunk_size for request in requests),
        ),
    )
    benchmark.observe_finished_requests(requests)
    benchmark.observe_terminated_requests(requests)
    assert benchmark.local_outcome() == BenchmarkOutcome.COMPLETE
    benchmark.apply_global_outcome(BenchmarkOutcome.COMPLETE)


def _fake_requests(ids, *, benchmark=True, total_tokens=4, cached_tokens=0):
    requests = []
    for request_id in ids:
        request = _FakeRequest(
            request_id,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
        )
        request.is_self_benchmark_request = benchmark
        request.py_self_benchmark_trial_id = 0 if benchmark else None
        requests.append(request)
    return requests


def _scheduled_prefill(*, last_chunk_ids=(), chunking_ids=()):
    return _FakeScheduledBatch(
        last_chunk=_fake_requests(last_chunk_ids),
        chunking=_fake_requests(chunking_ids),
    )


def _scheduled_decode(*, ids=()):
    return _FakeScheduledBatch(generation=_fake_requests(ids, total_tokens=5))


def _scheduled_mixed(*, benchmark_id, user_id):
    return _FakeScheduledBatch(
        last_chunk=_fake_requests([benchmark_id]),
        other=_fake_requests([user_id], benchmark=False),
    )


class _FakeConsensusDist:
    tp_size = 2
    tp_rank = 0

    def __init__(self, rank_values, diagnostics=None):
        self.rank_values = rank_values
        self.diagnostics = diagnostics or [None, None]
        self.collectives = []

    def tp_allreduce(self, value, op):
        self.collectives.append(("allreduce", int(value), op))
        return max(int(item) for item in self.rank_values)

    def tp_allgather(self, value):
        self.collectives.append(("allgather", value))
        return self.diagnostics


def _make_sync_executor(rank_values, diagnostics=None):
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    if diagnostics is None:
        diagnostics = [
            (
                {
                    "rank": rank,
                    "outcome": int(outcome),
                    "reason": f"rank_{rank}_{outcome.name.lower()}",
                    "drain_target": (
                        DrainTarget.SKIP.value
                        if outcome == BenchmarkOutcome.SKIP
                        else DrainTarget.ABORT.value
                    ),
                }
                if outcome in (BenchmarkOutcome.SKIP, BenchmarkOutcome.ABORT)
                else None
            )
            for rank, outcome in enumerate(rank_values)
        ]
    benchmark = MagicMock(active=True)
    benchmark.local_outcome.return_value = rank_values[0]
    benchmark.local_diagnostic.return_value = diagnostics[0]
    executor = object.__new__(PyExecutor)
    executor.dist = _FakeConsensusDist(rank_values, diagnostics)
    executor.self_benchmark = benchmark
    return executor, benchmark


def _benchmark_request(request_id=900_000_000):
    return types.SimpleNamespace(
        py_request_id=request_id,
        request_id=request_id,
        py_client_id=None,
        state=None,
        is_self_benchmark_request=True,
        is_attention_dp_dummy=False,
        is_dummy_request=False,
    )


class _RecordingWaitingQueue:
    def __init__(self):
        self.added = []

    def __len__(self):
        return len(self.added)

    def add_requests(self, requests):
        self.added.extend(requests)


def _make_response_executor(request, benchmark):
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = object.__new__(PyExecutor)
    executor.active_requests = [request]
    executor.self_benchmark = benchmark
    executor.perf_manager = types.SimpleNamespace(
        get_timestamp=lambda: 0.0,
        append_step_metrics=MagicMock(),
    )
    executor.iter_counter = 1
    executor._enqueue_responses = MagicMock()
    executor._terminate_request = MagicMock()
    executor.kv_cache_transceiver = None
    executor.enable_attention_dp = False
    executor.dist = types.SimpleNamespace(rank=0, world_size=1)
    return executor


def _make_termination_executor(events):
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = object.__new__(PyExecutor)
    executor.resource_manager = types.SimpleNamespace(
        free_resources=lambda request: events.append("free_resources")
    )
    executor._prefetched_request_ids = set()
    executor.gather_all_responses = False
    executor.dist = types.SimpleNamespace(rank=0)
    executor.result_wait_queues = {}
    executor.self_benchmark = types.SimpleNamespace(
        observe_terminated_requests=lambda requests: events.append("observe_terminated")
    )
    return executor


def test_torch_llm_args_self_benchmark_enables_iter_stats():
    args = TorchLlmArgs(model="dummy", self_benchmark_config=SelfBenchmarkConfig())

    assert args.enable_iter_perf_stats is True


def test_pyexecutor_self_benchmark_creation_is_explicit_and_missing_config_safe(
    monkeypatch,
):
    import tensorrt_llm._torch.pyexecutor.py_executor as py_executor_module

    executor = object.__new__(py_executor_module.PyExecutor)
    executor.llm_args = types.SimpleNamespace()
    benchmark = object()
    benchmark_cls = MagicMock(return_value=benchmark)
    monkeypatch.setattr(py_executor_module, "SelfBenchmark", benchmark_cls)

    # AutoDeploy uses a partial SimpleNamespace without self_benchmark_config.
    assert executor._create_self_benchmark(True) is None
    benchmark_cls.assert_not_called()

    executor.llm_args.self_benchmark_config = object()
    assert executor._create_self_benchmark(False) is None
    benchmark_cls.assert_not_called()

    assert executor._create_self_benchmark(True) is benchmark
    benchmark_cls.assert_called_once_with(executor)


def test_torch_llm_args_self_benchmark_rejects_encode_only():
    with pytest.raises(ValueError, match="decoder generation workloads"):
        TorchLlmArgs(model="dummy", encode_only=True, self_benchmark_config=SelfBenchmarkConfig())


def test_torch_llm_args_self_benchmark_rejects_attention_dp():
    with pytest.raises(ValueError, match="attention data parallelism"):
        TorchLlmArgs(
            model="dummy", enable_attention_dp=True, self_benchmark_config=SelfBenchmarkConfig()
        )


def test_torch_llm_args_self_benchmark_accepts_pure_tp():
    # Per-rank deterministic replication makes pure tensor parallelism safe:
    # every TP rank builds the same forward batch in lockstep.
    args = TorchLlmArgs(
        model="dummy", tensor_parallel_size=2, self_benchmark_config=SelfBenchmarkConfig()
    )
    assert args.parallel_config.tp_size == 2
    assert args.enable_iter_perf_stats is True


def test_torch_llm_args_self_benchmark_rejects_pipeline_parallel():
    with pytest.raises(ValueError, match="pipeline parallelism"):
        TorchLlmArgs(
            model="dummy", pipeline_parallel_size=2, self_benchmark_config=SelfBenchmarkConfig()
        )


def test_torch_llm_args_self_benchmark_rejects_context_parallel():
    with pytest.raises(ValueError, match="context parallelism"):
        TorchLlmArgs(
            model="dummy", context_parallel_size=2, self_benchmark_config=SelfBenchmarkConfig()
        )


def test_case_generation_uses_executor_limits():
    config = SelfBenchmarkConfig(
        mode="agg",
        prefill_isl_granularity=2,
        prefill_batch_granularity=1,
        decode_context_granularity=2,
        decode_batch_granularity=2,
        warmup_iterations=1,
    )

    benchmark = SelfBenchmark(_make_executor(config))

    assert [case.case_type for case in benchmark._cases] == [
        "warmup",
        "prefill",
        "prefill",
        "decode",
        "decode",
        "decode",
        "decode",
    ]
    assert [case.isl for case in benchmark._cases if case.case_type == "prefill"] == [1, 8]
    assert [case.kv_read_tokens for case in benchmark._cases if case.case_type == "prefill"] == [
        0,
        0,
    ]
    assert {
        (case.context_length, case.batch_size)
        for case in benchmark._cases
        if case.case_type == "decode"
    } == {
        (1, 1),
        (1, 4),
        (6, 1),
        (6, 4),
    }


def test_prefill_cases_reserve_one_generation_slot_at_max_seq_len():
    config = SelfBenchmarkConfig(
        mode="prefill",
        prefill_isl_granularity=1,
        prefill_batch_granularity=1,
        warmup_iterations=0,
    )
    executor = _make_executor(config, max_num_tokens=128)
    executor.max_input_len = 64
    executor.max_seq_len = 64

    benchmark = SelfBenchmark(executor)
    prefill_cases = [case for case in benchmark._cases if case.case_type == "prefill"]

    assert prefill_cases
    assert max(case.isl for case in prefill_cases) == 63
    assert all(case.isl + 1 <= executor.max_seq_len for case in prefill_cases)


def test_decode_cases_reserve_input_and_sample_slots_at_max_seq_len():
    config = SelfBenchmarkConfig(
        mode="decode",
        decode_context_granularity=1,
        decode_batch_granularity=1,
        warmup_iterations=0,
    )
    executor = _make_executor(config, max_num_tokens=128)
    executor.max_input_len = 64
    executor.max_seq_len = 64

    benchmark = SelfBenchmark(executor)
    decode_cases = [case for case in benchmark._cases if case.case_type == "decode"]

    assert decode_cases
    assert max(case.context_length for case in decode_cases) == 62
    assert all(case.context_length + 2 <= executor.max_seq_len for case in decode_cases)


def test_prefill_cases_include_batch_axis_with_scheduler_caps():
    config = SelfBenchmarkConfig(
        mode="prefill",
        prefill_isl_granularity=2,
        prefill_batch_granularity=2,
        warmup_iterations=0,
    )

    benchmark = SelfBenchmark(_make_executor(config))
    prefill_cases = [case for case in benchmark._cases if case.case_type == "prefill"]

    assert [(case.isl, case.batch_size) for case in prefill_cases] == [
        (1, 1),
        (1, 4),
        (8, 1),
    ]
    assert all(case.batch_size <= 4 for case in prefill_cases)
    assert all(case.isl * case.batch_size <= 8 for case in prefill_cases)


def test_prefill_cases_include_block_aligned_kv_read_axis():
    config = SelfBenchmarkConfig(
        mode="prefill",
        prefill_isl_granularity=2,
        prefill_batch_granularity=1,
        prefill_kv_read_granularity=4,
        warmup_iterations=0,
    )

    benchmark = SelfBenchmark(_make_executor(config, tokens_per_block=4))

    assert [(case.case_type, case.isl, case.kv_read_tokens) for case in benchmark._cases] == [
        ("prefill", 1, 0),
        ("prefill", 8, 0),
        ("prefill_seed", 5, 4),
        ("prefill", 8, 4),
    ]
    assert benchmark._cases[2].cache_salt_id == benchmark._cases[3].cache_salt_id


def test_default_v1_planner_uses_bound_capacity_and_microbatch_components():
    config = SelfBenchmarkConfig(
        mode="prefill",
        prefill_isl_granularity=1,
        prefill_batch_granularity=1,
        warmup_iterations=0,
    )
    executor = _make_executor(config)

    benchmark = SelfBenchmark(executor)

    assert executor.scheduler.capacity_scheduler.calls
    assert executor.scheduler.micro_batch_scheduler.calls
    assert benchmark._case_plans
    assert {plan.planner_backend for plan in benchmark._case_plans.values()} == {"v1_bound_default"}


def test_python_v1_planner_uses_exposed_scheduler_components():
    config = SelfBenchmarkConfig(
        mode="prefill",
        prefill_isl_granularity=1,
        prefill_batch_granularity=1,
        warmup_iterations=0,
    )
    executor = _make_executor(config)
    capacity_scheduler = executor.scheduler.capacity_scheduler
    capacity_impl = capacity_scheduler.impl
    capacity_scheduler.impl = None
    capacity_scheduler.schedule_request = MagicMock(
        side_effect=lambda requests: capacity_impl(
            requests, capacity_scheduler.kv_cache_manager, None, None
        )
    )
    executor.scheduler.micro_batch_scheduler.impl = None

    benchmark = SelfBenchmark(executor)

    assert benchmark._planner_error is None
    assert capacity_scheduler.schedule_request.call_count > 0
    assert executor.scheduler.micro_batch_scheduler.calls
    assert benchmark._case_plans
    assert {plan.planner_backend for plan in benchmark._case_plans.values()} == {"v1_python"}


def test_actual_v2_manager_uses_solver_without_v1_scheduler_probes():
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    class _TestKvCacheManagerV2(KVCacheManagerV2):
        def __init__(self):
            self.enable_block_reuse = True
            self.num_extra_kv_tokens = 0
            self._kv_reserve_draft_tokens = 0
            self.availability_queries = []

        def get_num_available_tokens(
            self, *, token_num_upper_bound, batch_size=1, max_num_draft_tokens=0
        ):
            self.availability_queries.append(
                (token_num_upper_bound, batch_size, max_num_draft_tokens)
            )
            return token_num_upper_bound if batch_size <= 2 else 0

        def get_kv_cache_stats(self):
            return _FakeKvStats(tokens_per_block=32, free_num_blocks=2)

    config = SelfBenchmarkConfig(
        mode="prefill",
        prefill_isl_granularity=1,
        prefill_batch_granularity=1,
        warmup_iterations=0,
    )
    executor = _make_executor(config)
    manager = _TestKvCacheManagerV2()
    executor.resource_manager._kv_cache_manager = manager
    executor.scheduler.capacity_scheduler.schedule_request = MagicMock(
        side_effect=AssertionError("V2 planning must not probe the V1 capacity scheduler")
    )
    executor.scheduler.micro_batch_scheduler.schedule = MagicMock(
        side_effect=AssertionError("V2 planning must not probe the V1 microbatch scheduler")
    )

    benchmark = SelfBenchmark(executor)

    assert benchmark._planner_error is None
    assert manager.availability_queries
    assert {query[1] for query in manager.availability_queries} >= {1, 2, 3}
    assert benchmark._case_plans
    assert {plan.planner_backend for plan in benchmark._case_plans.values()} == {"v2_solver"}
    executor.scheduler.capacity_scheduler.schedule_request.assert_not_called()
    executor.scheduler.micro_batch_scheduler.schedule.assert_not_called()


def test_default_v1_planner_uses_executor_inflight_id_container():
    config = SelfBenchmarkConfig(
        mode="prefill",
        prefill_isl_granularity=1,
        prefill_batch_granularity=1,
        warmup_iterations=0,
    )
    executor = _make_executor(config)

    benchmark = SelfBenchmark(executor)

    assert benchmark._planner_error is None
    assert executor.scheduler.micro_batch_scheduler.inflight_calls
    assert all(
        inflight_ids is executor.inflight_req_ids
        for inflight_ids in executor.scheduler.micro_batch_scheduler.inflight_calls
    )


def test_cache_hit_prefill_planner_uses_expected_uncached_compute():
    config = SelfBenchmarkConfig(
        mode="prefill",
        prefill_isl_granularity=1,
        prefill_batch_granularity=1,
        prefill_kv_read_granularity=2,
        warmup_iterations=0,
    )
    benchmark = SelfBenchmark(_make_executor(config, tokens_per_block=4))

    case = next(
        case
        for case in benchmark._cases
        if case.case_type == "prefill" and case.isl == 8 and case.kv_read_tokens == 4
    )
    plan = benchmark._case_plans[case.case_id]

    # The measured axis can admit two requests, but its ISL=5 seed can admit
    # only one. The pair shares the lower capacity so the seed is exact too.
    assert case.batch_size == 1
    assert plan.microbatch_capacity == 2
    assert plan.token_components == {
        "base_tokens": 8,
        "reusable_tokens": 4,
        "uncached_tokens": 4,
        "beam_width": 1,
        "draft_tokens": 0,
        "compute_tokens_per_request": 4,
    }


def test_decode_planner_uses_runtime_beam_and_draft_compute_charge():
    config = SelfBenchmarkConfig(
        mode="decode",
        decode_context_granularity=1,
        decode_batch_granularity=1,
        warmup_iterations=0,
    )
    executor = _make_executor(
        config,
        max_num_tokens=10,
        max_beam_width=2,
        max_total_draft_tokens=3,
    )

    benchmark = SelfBenchmark(executor)
    case = next(case for case in benchmark._cases if case.case_type == "decode")
    plan = benchmark._case_plans[case.case_id]

    assert case.batch_size == 2
    assert plan.microbatch_capacity == 2
    assert plan.token_components["beam_width"] == 2
    assert plan.token_components["draft_tokens"] == 3
    assert plan.token_components["compute_tokens_per_request"] == 5
    assert any(
        call.get("prepare_resource") is False
        for call in executor.resource_manager._kv_cache_manager.add_dummy_calls
    )


def test_decode_planner_caps_target_and_draft_kv_independently():
    config = SelfBenchmarkConfig(
        mode="decode",
        decode_context_granularity=1,
        decode_batch_granularity=1,
        warmup_iterations=0,
    )
    executor = _make_executor(
        config,
        max_num_tokens=64,
        target_kv_capacity=4,
        draft_kv_capacity=2,
    )

    benchmark = SelfBenchmark(executor)
    case = next(case for case in benchmark._cases if case.case_type == "decode")
    plan = benchmark._case_plans[case.case_id]
    draft_manager = executor.resource_manager._draft_kv_cache_manager

    assert case.batch_size == 2
    assert plan.target_kv_capacity == 4
    assert plan.draft_kv_capacity == 2
    assert plan.limiting_constraint == "draft_kv_capacity"
    assert draft_manager.impl.capacity_probe_calls


def test_zero_warmup_decode_does_not_probe_prefill_planner(monkeypatch):
    config = SelfBenchmarkConfig(
        mode="decode",
        decode_context_granularity=1,
        decode_batch_granularity=1,
        warmup_iterations=0,
    )
    prefill_planner = MagicMock(side_effect=AssertionError("unexpected prefill planning"))
    monkeypatch.setattr(SelfBenchmark, "_plan_prefill_axis", prefill_planner)

    benchmark = SelfBenchmark(_make_executor(config))

    assert benchmark._cases
    prefill_planner.assert_not_called()


def test_tp_planner_consensus_rebuilds_cases_from_global_component_minimum():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    config = SelfBenchmarkConfig(
        mode="decode",
        decode_context_granularity=1,
        decode_batch_granularity=2,
        warmup_iterations=0,
    )
    benchmark = SelfBenchmark(_make_executor(config, max_num_tokens=64))
    local_record = benchmark.local_planner_record(rank=0)
    limiting_record = json.loads(json.dumps(local_record))
    limiting_record["rank"] = 1
    limiting_plan = limiting_record["axes"][0]["plan"]
    limiting_plan["microbatch_capacity"] = 2
    limiting_plan["proposed_capacity"] = 2
    limiting_plan["limiting_constraint"] = "microbatch_token_capacity"

    executor = object.__new__(PyExecutor)
    executor.self_benchmark = benchmark
    executor.dist = types.SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        tp_allgather=MagicMock(return_value=[local_record, limiting_record]),
    )

    executor._sync_self_benchmark_plan()

    assert [case.batch_size for case in benchmark._cases] == [1, 2]
    assert {plan.proposed_batch_size for plan in benchmark._case_plans.values()} == {1, 2}
    assert {plan.planner_origin_rank for plan in benchmark._case_plans.values()} == {1}
    executor.dist.tp_allgather.assert_called_once_with(local_record)


def test_planner_exception_is_contained_and_aborted_by_tp_consensus(tmp_path):
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    output_path = tmp_path / "planner-error.json"
    config = SelfBenchmarkConfig(
        mode="prefill",
        output_path=str(output_path),
        prefill_isl_granularity=1,
        warmup_iterations=0,
    )
    source_executor = _make_executor(config)
    source_executor.scheduler.capacity_scheduler.schedule_request = MagicMock(
        side_effect=RuntimeError("rank-local capacity failure")
    )

    benchmark = SelfBenchmark(source_executor)
    local_record = benchmark.local_planner_record(rank=0)
    rank_one_record = {
        "rank": 1,
        "ok": False,
        "error": {
            "type": "RuntimeError",
            "reason": "rank-one capacity failure",
        },
        "signature": [],
        "axes": [],
    }
    executor = object.__new__(PyExecutor)
    executor.self_benchmark = benchmark
    executor.dist = types.SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        tp_allgather=MagicMock(return_value=[local_record, rank_one_record]),
    )

    assert local_record["ok"] is False
    assert json.loads(output_path.read_text())["status"] == "running"

    executor._sync_self_benchmark_plan()

    terminal = json.loads(output_path.read_text())
    assert terminal["status"] == "aborted"
    assert {
        key: terminal["abort"][key] for key in ("trial_id", "case", "reason", "origin_rank")
    } == {
        "trial_id": None,
        "case": None,
        "reason": "planner_error: RuntimeError: rank-local capacity failure",
        "origin_rank": 0,
    }
    assert terminal["abort"]["normalized_reason"] == "planner_error"
    assert terminal["abort"]["admission"] == {}
    assert benchmark.active is False


def test_output_invalidation_failure_is_contained_before_plan_collective(monkeypatch, tmp_path):
    config = SelfBenchmarkConfig(
        mode="prefill",
        output_path=str(tmp_path / "unwritable.json"),
        prefill_isl_granularity=1,
        warmup_iterations=0,
    )
    writer = MagicMock(side_effect=OSError("rank-local disk failure"))
    monkeypatch.setattr(SelfBenchmark, "_atomic_write_json", writer)

    benchmark = SelfBenchmark(_make_executor(config))
    record = benchmark.local_planner_record(rank=1)

    assert record["ok"] is False
    assert record["error"] == {
        "type": "OSError",
        "reason": "rank-local disk failure",
    }

    benchmark.apply_planner_consensus([record])

    assert benchmark.active is False
    assert benchmark._run_state == RunState.ABORTED
    assert writer.call_count == 1


def test_terminal_write_failure_does_not_escape_plan_consensus(monkeypatch, tmp_path):
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    config = SelfBenchmarkConfig(
        mode="prefill",
        output_path=str(tmp_path / "terminal-write-failure.json"),
        prefill_isl_granularity=1,
        warmup_iterations=0,
    )
    source_executor = _make_executor(config)
    source_executor.scheduler.capacity_scheduler.schedule_request = MagicMock(
        side_effect=RuntimeError("planner failure after sentinel")
    )
    writer = MagicMock(side_effect=[None, OSError("terminal disk failure")])
    monkeypatch.setattr(SelfBenchmark, "_atomic_write_json", writer)
    benchmark = SelfBenchmark(source_executor)
    executor = object.__new__(PyExecutor)
    executor.self_benchmark = benchmark
    executor.dist = types.SimpleNamespace(tp_size=1, tp_rank=0)

    executor._sync_self_benchmark_plan()

    assert benchmark.active is False
    assert benchmark._run_state == RunState.ABORTED
    assert benchmark._artifact_write_error == {
        "stage": "terminal",
        "type": "OSError",
        "reason": "terminal disk failure",
    }
    assert writer.call_count == 2


def test_terminal_output_construction_failure_is_contained(monkeypatch, tmp_path):
    config = SelfBenchmarkConfig(
        mode="prefill",
        output_path=str(tmp_path / "terminal-construction-failure.json"),
        prefill_isl_granularity=1,
        warmup_iterations=0,
    )
    benchmark = SelfBenchmark(_make_executor(config))
    monkeypatch.setattr(
        benchmark,
        "_limits",
        MagicMock(side_effect=AttributeError("max_seq_len is not initialized")),
    )

    benchmark._finish_aborted()

    assert benchmark.active is False
    assert benchmark._run_state == RunState.ABORTED
    assert benchmark._artifact_write_error == {
        "stage": "terminal",
        "type": "AttributeError",
        "reason": "max_seq_len is not initialized",
    }


def test_tp_planner_signature_mismatch_aborts_before_first_trial(tmp_path):
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    output_path = tmp_path / "planner-mismatch.json"
    config = SelfBenchmarkConfig(
        mode="decode",
        output_path=str(output_path),
        decode_context_granularity=1,
        warmup_iterations=0,
    )
    benchmark = SelfBenchmark(_make_executor(config))
    local_record = benchmark.local_planner_record(rank=0)
    assert local_record["grid_config"] == config.model_dump(exclude={"output_path"})
    mismatched_record = json.loads(json.dumps(local_record))
    mismatched_record["rank"] = 1
    mismatched_record["signature"][0]["token_components"]["beam_width"] = 2
    executor = object.__new__(PyExecutor)
    executor.self_benchmark = benchmark
    executor.dist = types.SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        tp_allgather=MagicMock(return_value=[local_record, mismatched_record]),
    )

    executor._sync_self_benchmark_plan()

    terminal = json.loads(output_path.read_text())
    assert terminal["status"] == "aborted"
    assert terminal["abort"]["reason"] == "planner_signature_mismatch"
    assert terminal["abort"]["origin_rank"] == 0
    assert benchmark._next_case_index == 0


def test_tp_planner_grid_config_mismatch_aborts_before_first_trial(tmp_path):
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    output_path = tmp_path / "planner-config-mismatch.json"
    config = SelfBenchmarkConfig(
        mode="decode",
        output_path=str(output_path),
        decode_context_granularity=1,
        decode_batch_granularity=2,
        warmup_iterations=0,
    )
    benchmark = SelfBenchmark(_make_executor(config))
    local_record = benchmark.local_planner_record(rank=0)
    mismatched_record = json.loads(json.dumps(local_record))
    mismatched_record["rank"] = 1
    mismatched_record["grid_config"] = {
        "mode": "decode",
        "decode_context_granularity": 1,
        "decode_batch_granularity": 3,
        "warmup_iterations": 0,
    }
    executor = object.__new__(PyExecutor)
    executor.self_benchmark = benchmark
    executor.dist = types.SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        tp_allgather=MagicMock(return_value=[local_record, mismatched_record]),
    )

    executor._sync_self_benchmark_plan()

    terminal = json.loads(output_path.read_text())
    assert terminal["status"] == "aborted"
    assert terminal["abort"]["reason"] == "planner_signature_mismatch"
    assert terminal["abort"]["origin_rank"] == 0
    assert benchmark._next_case_index == 0


def test_disabled_self_benchmark_skips_planning_and_plan_collective(monkeypatch):
    import tensorrt_llm._torch.pyexecutor.py_executor as py_executor_module

    constructor = MagicMock(side_effect=AssertionError("planner should remain dormant"))
    monkeypatch.setattr(py_executor_module, "SelfBenchmark", constructor)
    executor = object.__new__(py_executor_module.PyExecutor)
    executor.llm_args = types.SimpleNamespace(self_benchmark_config=None)
    executor.dist = types.SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        tp_allgather=MagicMock(),
    )

    executor.self_benchmark = executor._create_self_benchmark(enabled=True)
    executor._sync_self_benchmark_plan()

    assert executor.self_benchmark is None
    constructor.assert_not_called()
    executor.dist.tp_allgather.assert_not_called()


def test_prefill_cases_omit_kv_read_axis_when_block_reuse_disabled():
    config = SelfBenchmarkConfig(
        mode="prefill",
        prefill_isl_granularity=2,
        prefill_batch_granularity=1,
        prefill_kv_read_granularity=4,
        warmup_iterations=0,
    )

    benchmark = SelfBenchmark(_make_executor(config, tokens_per_block=4, enable_block_reuse=False))

    assert [(case.case_type, case.isl, case.kv_read_tokens) for case in benchmark._cases] == [
        ("prefill", 1, 0),
        ("prefill", 8, 0),
    ]


def test_prefill_request_built_per_rank_marks_llm_request():
    config = SelfBenchmarkConfig(mode="prefill", prefill_isl_granularity=1, warmup_iterations=0)
    benchmark = SelfBenchmark(_make_executor(config))

    requests = benchmark.make_prefill_requests([], [])

    # Built locally per-rank (no RequestQueueItem / broadcast). Deterministic
    # request id and benchmark marking are set on the LlmRequest itself.
    assert len(requests) == 1
    assert requests[0].py_request_id == 900_000_000
    assert requests[0].is_self_benchmark_request is True
    assert requests[0].py_self_benchmark_trial_id == 0


def test_prefill_injection_uses_case_batch_size():
    config = SelfBenchmarkConfig(mode="prefill", warmup_iterations=0)
    benchmark = SelfBenchmark(_make_executor(config))
    benchmark._cases = [BenchmarkCase(case_type="prefill", case_id=0, isl=4, batch_size=3)]

    requests = benchmark.make_prefill_requests([], [])

    assert [request.py_request_id for request in requests] == [
        900_000_000,
        900_000_001,
        900_000_002,
    ]
    assert sum(len(request.input_token_ids) for request in requests) == 12
    assert len({request.cache_salt for request in requests}) == 3
    assert all(request.is_self_benchmark_request for request in requests)
    assert all(request.py_self_benchmark_trial_id == 0 for request in requests)


def test_exact_trial_completes_only_after_execution_metrics_and_release():
    benchmark, requests = _start_prefill_trial(batch_size=2)
    scheduled = _FakeScheduledBatch(last_chunk=requests)

    assert benchmark._current_trial.state == TrialState.RUNNING
    assert benchmark._run_state == RunState.RUNNING
    benchmark.observe_scheduled_batch(scheduled)
    benchmark.observe_finished_requests(requests)
    benchmark.observe_terminated_requests(requests)
    assert benchmark.local_outcome() == BenchmarkOutcome.CONTINUE

    benchmark.observe_executed_batch(scheduled, _prefill_stats(requests=2, tokens=8))
    assert benchmark.local_outcome() == BenchmarkOutcome.COMPLETE


def test_execution_validation_uses_admitted_shape_after_request_mutation():
    benchmark, requests = _start_prefill_trial(batch_size=1)
    scheduled = _FakeScheduledBatch(last_chunk=requests)

    benchmark.observe_scheduled_batch(scheduled)
    requests[0]._total_tokens += 1
    requests[0].context_chunk_size = 0

    assert benchmark.observe_executed_batch(scheduled, _prefill_stats(requests=1, tokens=4))
    assert benchmark._current_trial.pending_outcome == BenchmarkOutcome.CONTINUE
    assert benchmark._current_trial.executed_request_ids == {requests[0].py_request_id}


def test_overlap_scheduler_bubble_after_exact_admission_does_not_skip_trial():
    benchmark, requests = _start_prefill_trial(batch_size=2)

    benchmark.observe_scheduled_batch(_FakeScheduledBatch(last_chunk=requests))
    benchmark.observe_scheduled_batch(_FakeScheduledBatch())

    assert benchmark.local_outcome() == BenchmarkOutcome.CONTINUE
    assert benchmark._current_trial.scheduled_request_ids == {
        request.py_request_id for request in requests
    }


def test_draining_waits_only_for_constructed_requests():
    benchmark, requests = _start_prefill_trial(batch_size=2, constructed_count=1)
    benchmark.observe_failed_requests([], "synthetic_request_construction_failed")
    benchmark.apply_global_outcome(
        BenchmarkOutcome.ABORT,
        reason="synthetic_request_construction_failed",
        origin_rank=1,
    )

    assert benchmark._run_state == RunState.DRAINING
    assert benchmark._current_trial.state == TrialState.DRAINING
    assert benchmark.local_outcome() == BenchmarkOutcome.CONTINUE

    benchmark.observe_terminated_requests(requests)
    assert benchmark.local_outcome() == BenchmarkOutcome.COMPLETE


def test_abort_drain_publishes_canonical_terminal_diagnostic(tmp_path):
    output_path = tmp_path / "benchmark.json"
    config = SelfBenchmarkConfig(
        mode="prefill",
        output_path=str(output_path),
        warmup_iterations=0,
    )
    benchmark = SelfBenchmark(_make_executor(config))
    case = BenchmarkCase(case_type="prefill", case_id=0, isl=4)
    requests = _fake_requests([900_000_000])
    benchmark._cases = [case]
    benchmark.start_trial(case, requests)

    benchmark.observe_failed_requests(requests, "synthetic request failed")
    benchmark.apply_global_outcome(
        BenchmarkOutcome.ABORT,
        reason="synthetic request failed",
        origin_rank=1,
    )
    benchmark.observe_terminated_requests(requests)
    assert benchmark.local_outcome() == BenchmarkOutcome.COMPLETE
    benchmark.apply_global_outcome(BenchmarkOutcome.COMPLETE)

    terminal = json.loads(output_path.read_text())
    assert terminal["status"] == "aborted"
    assert terminal["valid"] is False
    assert {
        key: terminal["abort"][key] for key in ("trial_id", "case", "reason", "origin_rank")
    } == {
        "trial_id": 0,
        "case": asdict(case),
        "reason": "synthetic request failed",
        "origin_rank": 1,
    }
    assert terminal["abort"]["normalized_reason"] == "synthetic_request_failed"
    assert terminal["abort"]["admission"]["origin_rank"] == 1


def test_scheduler_and_termination_predicates_follow_trial_lifecycle():
    benchmark, requests = _start_prefill_trial(batch_size=2)
    user_request = _fake_requests([7], benchmark=False)[0]

    assert all(not benchmark.should_hold_from_scheduler(request) for request in requests)
    assert all(not benchmark.should_terminate(request) for request in requests)
    assert benchmark.should_hold_from_scheduler(user_request) is False
    assert benchmark.should_terminate(user_request) is False

    scheduled = _FakeScheduledBatch(last_chunk=requests)
    benchmark.observe_scheduled_batch(scheduled)
    assert all(benchmark.should_hold_from_scheduler(request) for request in requests)
    assert all(not benchmark.should_terminate(request) for request in requests)

    benchmark.observe_finished_requests(requests)
    assert all(benchmark.should_terminate(request) for request in requests)

    benchmark.apply_global_outcome(
        BenchmarkOutcome.SKIP,
        reason="scheduled_batch_shape_mismatch",
        origin_rank=0,
    )
    assert all(benchmark.should_hold_from_scheduler(request) for request in requests)
    assert all(benchmark.should_terminate(request) for request in requests)


@pytest.mark.parametrize(
    ("rank_values", "expected"),
    [
        (
            [BenchmarkOutcome.COMPLETE, BenchmarkOutcome.COMPLETE],
            BenchmarkOutcome.COMPLETE,
        ),
        (
            [BenchmarkOutcome.COMPLETE, BenchmarkOutcome.CONTINUE],
            BenchmarkOutcome.CONTINUE,
        ),
        (
            [BenchmarkOutcome.CONTINUE, BenchmarkOutcome.SKIP],
            BenchmarkOutcome.SKIP,
        ),
        (
            [BenchmarkOutcome.SKIP, BenchmarkOutcome.ABORT],
            BenchmarkOutcome.ABORT,
        ),
    ],
)
def test_tp_outcome_uses_ordered_max(rank_values, expected):
    executor, benchmark = _make_sync_executor(rank_values)

    executor._sync_self_benchmark_outcome()

    assert benchmark.apply_global_outcome.call_args.args[0] == expected


def test_lowest_failing_rank_supplies_canonical_diagnostic():
    diagnostics = [
        {
            "rank": 0,
            "outcome": int(BenchmarkOutcome.SKIP),
            "reason": "rank_zero_mismatch",
            "drain_target": DrainTarget.SKIP.value,
        },
        {
            "rank": 1,
            "outcome": int(BenchmarkOutcome.SKIP),
            "reason": "rank_one_mismatch",
            "drain_target": DrainTarget.SKIP.value,
        },
    ]
    executor, benchmark = _make_sync_executor(
        [BenchmarkOutcome.SKIP, BenchmarkOutcome.SKIP],
        diagnostics,
    )

    executor._sync_self_benchmark_outcome()

    benchmark.apply_global_outcome.assert_called_once_with(
        BenchmarkOutcome.SKIP,
        reason="rank_zero_mismatch",
        origin_rank=0,
        drain_target=DrainTarget.SKIP,
    )


def test_ordinary_abort_outranks_lower_rank_shutdown_interrupt():
    diagnostics = [
        {
            "rank": 0,
            "outcome": int(BenchmarkOutcome.ABORT),
            "reason": "shutdown_requested",
            "drain_target": DrainTarget.INTERRUPT.value,
        },
        {
            "rank": 1,
            "outcome": int(BenchmarkOutcome.ABORT),
            "reason": "synthetic_request_construction_failed",
            "drain_target": DrainTarget.ABORT.value,
        },
    ]
    executor, benchmark = _make_sync_executor(
        [BenchmarkOutcome.ABORT, BenchmarkOutcome.ABORT],
        diagnostics,
    )

    executor._sync_self_benchmark_outcome()

    benchmark.apply_global_outcome.assert_called_once_with(
        BenchmarkOutcome.ABORT,
        reason="synthetic_request_construction_failed",
        origin_rank=1,
        drain_target=DrainTarget.ABORT,
    )


def test_schedule_filters_held_requests_without_mutating_active_requests():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    user_request = _fake_requests([7], benchmark=False)[0]
    held_request = _fake_requests([900_000_000])[0]
    benchmark = MagicMock(active=True)
    benchmark.should_hold_from_scheduler.side_effect = lambda request: request is held_request
    scheduler_output = types.SimpleNamespace(
        context_requests=[],
        generation_requests=[],
        encoder_requests=[],
        paused_requests=[],
        fitting_disagg_gen_init_requests=[],
        num_fitting_requests=0,
    )
    executor = object.__new__(PyExecutor)
    executor.active_requests = [user_request, held_request]
    executor.inflight_req_ids = object()
    executor.self_benchmark = benchmark
    executor.scheduler = types.SimpleNamespace(
        schedule_request=MagicMock(return_value=scheduler_output)
    )
    executor.enable_attention_dp = False
    executor.attention_dp_enable_balance = False
    executor.enable_batch_waiting = False
    executor.kv_cache_manager = None
    executor.kv_cache_transceiver = None

    PyExecutor._schedule(executor)

    executor.scheduler.schedule_request.assert_called_once_with(
        [user_request], executor.inflight_req_ids
    )
    assert executor.active_requests == [user_request, held_request]


def test_drain_preserves_overlap_protected_and_user_requests():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    draining_request, protected_request = _fake_requests([900_000_000, 900_000_001])
    user_request = _fake_requests([7], benchmark=False)[0]
    benchmark = MagicMock()
    benchmark.should_terminate.return_value = True
    executor = object.__new__(PyExecutor)
    executor.self_benchmark = benchmark
    executor.active_requests = [draining_request, protected_request, user_request]
    executor._terminate_request = MagicMock()
    executor.previous_batch = types.SimpleNamespace(
        scheduled_requests=_FakeScheduledBatch(last_chunk=[protected_request])
    )

    protected_ids = PyExecutor._self_benchmark_protected_ids(executor)
    PyExecutor._drain_self_benchmark_requests(executor, protected_ids)

    assert protected_ids == {protected_request.py_request_id}
    assert executor.active_requests == [protected_request, user_request]
    executor._terminate_request.assert_called_once_with(draining_request)


def test_prepare_and_schedule_syncs_construction_and_final_shape_boundaries():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    events = []
    previous_request, current_request = _fake_requests([900_000_000, 900_000_001])
    previous_ids = {previous_request.py_request_id}
    scheduled_batch = _FakeScheduledBatch(last_chunk=[current_request])
    benchmark = MagicMock(active=True)
    benchmark.observe_scheduled_batch.side_effect = lambda batch: events.append("observe_scheduled")
    executor = types.SimpleNamespace(
        _fetch_and_activate_new_requests=lambda: events.append("fetch") or [],
        is_shutdown=False,
        should_stop_processing=False,
        _handle_control_request=lambda: None,
        kv_cache_transceiver=None,
        enable_iter_perf_stats=False,
        _pad_attention_dp_dummy_request=lambda: None,
        _prefetch_for_context_requests=lambda: None,
        drafter=None,
        _schedule=lambda: events.append("schedule") or (scheduled_batch, [], 0),
        self_benchmark=benchmark,
        _sync_self_benchmark_outcome=lambda: events.append("sync"),
        _self_benchmark_protected_ids=lambda: set(previous_ids),
        _drain_self_benchmark_requests=lambda protected_ids: events.append(
            ("drain", protected_ids)
        ),
        active_requests=[],
    )

    result, _ = PyExecutor._prepare_and_schedule_batch(executor)

    assert result is scheduled_batch
    assert events == [
        "fetch",
        "sync",
        ("drain", previous_ids),
        "schedule",
        "observe_scheduled",
        "sync",
        ("drain", previous_ids | {current_request.py_request_id}),
    ]


def test_prepare_and_schedule_skips_all_benchmark_helpers_when_disabled():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    scheduled_batch = _FakeScheduledBatch()
    sync_outcome = MagicMock()
    protected_ids = MagicMock()
    drain_requests = MagicMock()
    executor = types.SimpleNamespace(
        _fetch_and_activate_new_requests=lambda: [],
        is_shutdown=False,
        should_stop_processing=False,
        _handle_control_request=lambda: None,
        kv_cache_transceiver=None,
        enable_iter_perf_stats=False,
        _pad_attention_dp_dummy_request=lambda: None,
        _prefetch_for_context_requests=lambda: None,
        drafter=None,
        _schedule=lambda: (scheduled_batch, [], 0),
        self_benchmark=None,
        _sync_self_benchmark_outcome=sync_outcome,
        _self_benchmark_protected_ids=protected_ids,
        _drain_self_benchmark_requests=drain_requests,
        active_requests=[],
    )

    result, _ = PyExecutor._prepare_and_schedule_batch(executor)

    assert result is scheduled_batch
    sync_outcome.assert_not_called()
    protected_ids.assert_not_called()
    drain_requests.assert_not_called()


def test_prepare_and_schedule_defers_shutdown_until_interrupt_consensus():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    events = []
    scheduled_batch = _FakeScheduledBatch()
    benchmark = MagicMock(active=True)
    benchmark.request_interrupt.side_effect = lambda: events.append("interrupt")
    benchmark.observe_scheduled_batch.side_effect = lambda batch: events.append("observe_scheduled")
    executor = types.SimpleNamespace(
        _fetch_and_activate_new_requests=lambda: events.append("fetch") or [],
        is_shutdown=True,
        should_stop_processing=True,
        _handle_control_request=lambda: None,
        kv_cache_transceiver=None,
        enable_iter_perf_stats=False,
        _pad_attention_dp_dummy_request=lambda: None,
        _prefetch_for_context_requests=lambda: None,
        drafter=None,
        _schedule=lambda: events.append("schedule") or (scheduled_batch, [], 0),
        self_benchmark=benchmark,
        _sync_self_benchmark_outcome=lambda: events.append("sync"),
        _self_benchmark_protected_ids=lambda: set(),
        _drain_self_benchmark_requests=lambda protected_ids: events.append(
            ("drain", protected_ids)
        ),
        active_requests=[],
    )

    result, _ = PyExecutor._prepare_and_schedule_batch(executor)

    assert result is scheduled_batch
    assert events[:5] == [
        "fetch",
        "interrupt",
        "sync",
        ("drain", set()),
        "schedule",
    ]


def test_handle_responses_keeps_running_benchmark_request_active():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    request = _benchmark_request()
    benchmark = MagicMock()
    benchmark.should_terminate.return_value = False
    executor = _make_response_executor(request, benchmark)

    finished = PyExecutor._handle_responses(executor)

    assert finished == []
    assert executor.active_requests == [request]
    executor._terminate_request.assert_not_called()
    benchmark.should_terminate.assert_called_once_with(request)


def test_handle_responses_terminates_completed_benchmark_request():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    request = _benchmark_request()
    benchmark = MagicMock()
    benchmark.should_terminate.return_value = True
    executor = _make_response_executor(request, benchmark)

    finished = PyExecutor._handle_responses(executor)

    assert finished == [request]
    assert executor.active_requests == []
    executor._terminate_request.assert_called_once_with(request)
    benchmark.should_terminate.assert_called_once_with(request)


def test_process_previous_batch_commits_benchmark_resources_before_responses():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    events = []
    request = _benchmark_request()
    scheduled = _FakeScheduledBatch(last_chunk=[request])
    benchmark = types.SimpleNamespace(
        active=True,
        observe_finished_requests=lambda requests: events.append(("observe_finished", requests)),
    )
    executor = types.SimpleNamespace(
        previous_batch=types.SimpleNamespace(scheduled_requests=scheduled),
        self_benchmark=benchmark,
        _handle_canceled_requests=lambda: None,
        _observe_self_benchmark_finished=lambda scheduled_requests: (
            benchmark.observe_finished_requests(scheduled_requests.all_requests())
        ),
        _handle_responses=lambda emit_first_iter=True: (
            events.append(("handle_responses", emit_first_iter)) or []
        ),
        model_engine=types.SimpleNamespace(),
        resource_manager=types.SimpleNamespace(
            update_resources=lambda *args: events.append("update_resources")
        ),
        enable_early_first_token_response=False,
        enable_kv_cache_events=False,
        enable_iter_perf_stats=False,
    )

    PyExecutor._process_previous_batch(executor)

    assert events == [
        ("observe_finished", [request]),
        "update_resources",
        ("handle_responses", True),
    ]


def test_process_previous_batch_skips_benchmark_observer_when_disabled():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    observer = MagicMock()
    scheduled = _FakeScheduledBatch()
    events = []
    executor = types.SimpleNamespace(
        previous_batch=types.SimpleNamespace(scheduled_requests=scheduled),
        self_benchmark=None,
        _handle_canceled_requests=lambda: None,
        _observe_self_benchmark_finished=observer,
        _handle_responses=lambda emit_first_iter=True: (
            events.append(("handle_responses", emit_first_iter)) or []
        ),
        model_engine=types.SimpleNamespace(),
        resource_manager=types.SimpleNamespace(
            update_resources=lambda *args: events.append("update_resources")
        ),
        enable_early_first_token_response=False,
        enable_kv_cache_events=False,
        enable_iter_perf_stats=False,
    )

    PyExecutor._process_previous_batch(executor)

    observer.assert_not_called()
    assert events == [("handle_responses", True), "update_resources"]


def test_resource_release_precedes_termination_observer():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    events = []
    executor = _make_termination_executor(events)

    PyExecutor._do_terminate_request(executor, _benchmark_request())

    assert events == ["free_resources", "observe_terminated"]


def test_nonfatal_benchmark_request_error_requests_abort_before_cleanup():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    events = []
    request = _benchmark_request()
    benchmark = MagicMock()
    benchmark.observe_failed_requests.side_effect = lambda requests, reason: events.append(
        "observe_failed"
    )
    executor = types.SimpleNamespace(
        self_benchmark=benchmark,
        _error_budget=types.SimpleNamespace(
            consume=lambda message: False,
            budget=1.0,
        ),
        active_requests=[request],
        _enqueue_responses=MagicMock(),
        _terminate_request=lambda failed: events.append("free_resources"),
        _fatal_error=None,
        dist=types.SimpleNamespace(rank=0),
    )

    PyExecutor._handle_errors(
        executor,
        "synthetic request failed",
        requests=[request],
        charge_budget=False,
    )

    assert events == ["observe_failed", "free_resources"]
    executor._enqueue_responses.assert_called_once_with([])


def test_handle_responses_does_not_scan_request_markers_when_disabled():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    class TrackedUserRequest:
        def __init__(self):
            self.marker_reads = 0
            self.py_request_id = 7
            self.is_attention_dp_dummy = False
            self.py_kv_transfer_timed_out = False
            self.py_draft_tokens = []
            self.py_decoding_iter = 2
            self.return_perf_metrics = False
            self.is_finished = False

        def __getattribute__(self, name):
            if name == "is_self_benchmark_request":
                reads = object.__getattribute__(self, "marker_reads")
                object.__setattr__(self, "marker_reads", reads + 1)
                return False
            return object.__getattribute__(self, name)

        def is_generation_only_request(self):
            return False

    request = TrackedUserRequest()
    executor = _make_response_executor(request, benchmark=None)
    executor.stream_interval = 100

    PyExecutor._handle_responses(executor)

    assert request.marker_reads == 0
    assert executor.active_requests == [request]


def test_real_requests_remain_deferred_until_benchmark_terminal():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    normal_item = types.SimpleNamespace(is_normal_request=True)
    raw_queue = Queue()
    raw_queue.put(normal_item)
    waiting_queue = _RecordingWaitingQueue()
    benchmark = types.SimpleNamespace(active=True)
    executor = types.SimpleNamespace(
        control_requests=[],
        active_requests=[],
        request_accumulated=[],
        _disable_mpi=False,
        self_benchmark=benchmark,
        dist=types.SimpleNamespace(rank=0, tp_size=1, has_pp=False, cp_size=1),
        hang_detector=types.SimpleNamespace(pause=lambda: nullcontext()),
        executor_request_queue=types.SimpleNamespace(
            get_request_queue=lambda: raw_queue,
            get_from_request_queue=lambda timeout: [],
        ),
        request_broadcaster=types.SimpleNamespace(broadcast=lambda requests: (requests, None)),
        _handle_special_queue_items=lambda requests: requests,
    )

    PyExecutor._fetch_and_enqueue_requests(executor, waiting_queue, 0)
    assert executor.request_accumulated == [normal_item]
    assert waiting_queue.added == []

    benchmark.active = False
    PyExecutor._fetch_and_enqueue_requests(executor, waiting_queue, 0)
    assert executor.request_accumulated == []
    assert waiting_queue.added == [normal_item]


@pytest.mark.parametrize(
    ("scheduled", "reason"),
    [
        (
            _scheduled_prefill(last_chunk_ids=[900_000_000]),
            "scheduled_batch_size_mismatch",
        ),
        (
            _scheduled_prefill(chunking_ids=[900_000_000, 900_000_001]),
            "context_chunked",
        ),
        (_scheduled_decode(ids=[]), "scheduled_batch_size_mismatch"),
        (
            _scheduled_mixed(benchmark_id=900_000_000, user_id=7),
            "mixed_non_benchmark_request",
        ),
    ],
)
def test_non_exact_scheduled_batch_requests_skip(scheduled, reason):
    benchmark, _ = _start_prefill_trial(batch_size=2)

    benchmark.observe_scheduled_batch(scheduled)

    assert benchmark.local_outcome() == BenchmarkOutcome.SKIP
    assert benchmark.local_diagnostic(BenchmarkOutcome.SKIP, rank=0)["reason"] == reason


def test_constructor_exception_aborts_and_returns_constructed_prefix(monkeypatch):
    config = SelfBenchmarkConfig(
        mode="prefill",
        prefill_isl_granularity=1,
        prefill_batch_granularity=1,
        warmup_iterations=0,
    )
    benchmark = SelfBenchmark(_make_executor(config))
    benchmark._cases = [BenchmarkCase(case_type="prefill", case_id=0, isl=4, batch_size=2)]
    original = benchmark._make_prefill_request

    def fail_second(case, trial_id, offset):
        if offset == 1:
            raise RuntimeError("rank-local construction failure")
        return original(case, trial_id, offset)

    monkeypatch.setattr(benchmark, "_make_prefill_request", fail_second)

    requests = benchmark.make_prefill_requests([], [])

    assert len(requests) == 1
    assert benchmark._current_trial.constructed_request_ids == {900_000_000}
    assert benchmark.local_outcome() == BenchmarkOutcome.ABORT


def test_prefill_request_is_rank_independent():
    """Per-rank lockstep: non-rank-0 builds the same prefill request as rank 0.

    This guards the core TP invariant -- prefill injection must not depend on
    being rank 0. Every input is a deterministic function of the grid case, so
    a rank-1 executor produces a bit-identical request id / tokens / cache_salt.
    """
    config = SelfBenchmarkConfig(mode="prefill", prefill_isl_granularity=1, warmup_iterations=0)
    rank0 = SelfBenchmark(_make_executor(config))
    executor_rank1 = _make_executor(config)
    executor_rank1.dist = types.SimpleNamespace(rank=1)
    rank1 = SelfBenchmark(executor_rank1)

    req0 = rank0.make_prefill_requests([], [])[0]
    req1 = rank1.make_prefill_requests([], [])[0]

    assert req0.py_request_id == req1.py_request_id
    assert req0.input_token_ids == req1.input_token_ids
    assert req0.cache_salt == req1.cache_salt
    assert req0.py_self_benchmark_trial_id == req1.py_self_benchmark_trial_id


def test_shutdown_interrupts_benchmark_without_starting_next_trial(tmp_path):
    output_path = tmp_path / "benchmark.json"
    config = SelfBenchmarkConfig(mode="prefill", output_path=str(output_path), warmup_iterations=0)
    executor = _make_executor(config)
    executor.is_shutdown = True
    benchmark = SelfBenchmark(executor)

    assert benchmark.make_prefill_requests([], []) == []
    assert benchmark.local_outcome() == BenchmarkOutcome.ABORT
    diagnostic = benchmark.local_diagnostic(BenchmarkOutcome.ABORT, rank=0)
    assert diagnostic["drain_target"] == DrainTarget.INTERRUPT.value
    benchmark.apply_global_outcome(
        BenchmarkOutcome.ABORT,
        reason=diagnostic["reason"],
        origin_rank=diagnostic["rank"],
        drain_target=DrainTarget(diagnostic["drain_target"]),
    )
    assert benchmark.active is False
    assert output_path.exists()
    terminal = json.loads(output_path.read_text())
    assert terminal["status"] == "interrupted"
    assert terminal["valid"] is False


def test_interrupt_drains_live_trial_before_terminal_publication(tmp_path):
    output_path = tmp_path / "benchmark.json"
    config = SelfBenchmarkConfig(
        mode="prefill",
        output_path=str(output_path),
        warmup_iterations=0,
    )
    benchmark = SelfBenchmark(_make_executor(config))
    case = BenchmarkCase(case_type="prefill", case_id=0, isl=4)
    requests = _fake_requests([900_000_000])
    benchmark._cases = [
        case,
        BenchmarkCase(case_type="prefill", case_id=1, isl=8),
    ]
    benchmark.start_trial(case, requests)

    benchmark.request_interrupt()
    assert benchmark.local_outcome() == BenchmarkOutcome.ABORT
    diagnostic = benchmark.local_diagnostic(BenchmarkOutcome.ABORT, rank=0)
    assert diagnostic["drain_target"] == DrainTarget.INTERRUPT.value
    benchmark.apply_global_outcome(
        BenchmarkOutcome.ABORT,
        reason=diagnostic["reason"],
        origin_rank=diagnostic["rank"],
        drain_target=DrainTarget(diagnostic["drain_target"]),
    )

    assert benchmark._run_state == RunState.DRAINING
    assert json.loads(output_path.read_text())["status"] == "running"
    benchmark.observe_terminated_requests(requests)
    assert benchmark.local_outcome() == BenchmarkOutcome.COMPLETE
    benchmark.apply_global_outcome(BenchmarkOutcome.COMPLETE)

    terminal = json.loads(output_path.read_text())
    assert terminal["status"] == "interrupted"
    assert terminal["valid"] is False
    assert terminal["abort"]["reason"] == "shutdown_requested"


def test_prefill_seed_and_measure_cases_share_cache_salt():
    config = SelfBenchmarkConfig(mode="prefill", warmup_iterations=0)
    benchmark = SelfBenchmark(_make_executor(config))
    benchmark._cases = [
        BenchmarkCase(
            case_type="prefill_seed",
            case_id=0,
            isl=4,
            kv_read_tokens=4,
            batch_size=3,
            cache_salt_id=123,
        ),
        BenchmarkCase(
            case_type="prefill",
            case_id=1,
            isl=8,
            kv_read_tokens=4,
            batch_size=3,
            cache_salt_id=123,
        ),
    ]

    seed_items = benchmark.make_prefill_requests([], [])
    _complete_current_prefill_trial(benchmark, seed_items)
    measure_items = benchmark.make_prefill_requests([], [])

    assert len(seed_items) == len(measure_items) == 3
    assert all(item.input_token_ids == [1] * 4 for item in seed_items)
    assert all(item.input_token_ids == [1] * 8 for item in measure_items)
    assert [item.cache_salt for item in seed_items] == ["123:0", "123:1", "123:2"]
    assert [item.cache_salt for item in measure_items] == ["123:0", "123:1", "123:2"]


def test_prefill_seed_skip_skips_associated_measure_case(monkeypatch):
    config = SelfBenchmarkConfig(mode="prefill", warmup_iterations=0)
    benchmark = SelfBenchmark(_make_executor(config))
    warning = MagicMock()
    monkeypatch.setattr(self_benchmark_module.logger, "warning", warning)
    seed = BenchmarkCase(
        case_type="prefill_seed",
        case_id=0,
        isl=4,
        kv_read_tokens=4,
        cache_salt_id=123,
    )
    measured = BenchmarkCase(
        case_type="prefill",
        case_id=1,
        isl=8,
        kv_read_tokens=4,
        cache_salt_id=123,
    )
    following = BenchmarkCase(case_type="prefill", case_id=2, isl=2)
    benchmark._cases = [seed, measured, following]

    seed_requests = benchmark.make_prefill_requests([], [])
    benchmark._request_local_transition(BenchmarkOutcome.SKIP, "scheduled_batch_shape_mismatch")
    benchmark.apply_global_outcome(
        BenchmarkOutcome.SKIP,
        reason="scheduled_batch_shape_mismatch",
        origin_rank=1,
    )
    benchmark.observe_terminated_requests(seed_requests)
    benchmark.apply_global_outcome(BenchmarkOutcome.COMPLETE)

    assert benchmark._next_case_index == 2
    assert benchmark._next_trial_id == 2
    assert benchmark._skipped_cases == [
        {
            "trial_id": 1,
            "case": asdict(measured),
            "reason": "scheduled_batch_shape_mismatch",
            "origin_rank": 1,
        }
    ]
    assert benchmark._coverage() == {
        "expected_cases": 2,
        "completed_trials": 0,
        "skipped_cases": 1,
    }
    dependent_admission = benchmark._admission_records[1]
    assert dependent_admission["normalized_reason"] == "dependent_seed_skipped"
    assert dependent_admission["source_normalized_reason"] == ("scheduled_batch_shape_mismatch")
    assert warning.call_count == 2
    dependent_warning = warning.call_args_list[-1].args
    assert dependent_warning[2:7] == (1, 1, 1, 0, 0)
    assert dependent_warning[8:] == (
        "dependent_seed_skipped",
        "scheduled_batch_shape_mismatch",
    )

    following_requests = benchmark.make_prefill_requests([], [])
    assert following_requests
    assert benchmark._current_trial.trial_id == 2
    assert benchmark._current_trial.case == following


def test_decode_injection_uses_dummy_kv_path():
    config = SelfBenchmarkConfig(
        mode="decode", decode_context_granularity=1, decode_batch_granularity=1, warmup_iterations=0
    )
    executor = _make_executor(config)
    benchmark = SelfBenchmark(executor)

    requests = benchmark.make_decode_requests([], [])

    assert len(requests) == 4
    assert all(request.is_self_benchmark_request for request in requests)
    add_dummy_call = next(
        call
        for call in executor.resource_manager._kv_cache_manager.add_dummy_calls
        if call.get("prepare_resource", True)
    )
    assert add_dummy_call["request_ids"] == [
        900_000_000,
        900_000_001,
        900_000_002,
        900_000_003,
    ]
    expected_token_num = benchmark._cases[0].context_length + 1
    assert add_dummy_call["token_nums"] == [expected_token_num] * 4
    assert add_dummy_call["is_gen"] is True


def test_exact_decode_trial_completes_with_generation_prompt_shape():
    benchmark, requests = _start_decode_trial(batch_size=2, context_length=4)
    scheduled = _FakeScheduledBatch(generation=requests)

    assert [request.get_num_tokens(0) for request in requests] == [5, 5]
    assert [request.prompt_len for request in requests] == [4, 4]
    benchmark.observe_scheduled_batch(scheduled)
    assert benchmark.observe_executed_batch(
        scheduled,
        {
            "inflightBatchingStats": {
                "numContextRequests": 0,
                "numGenRequests": 2,
            }
        },
    )
    benchmark.observe_finished_requests(requests)
    benchmark.observe_terminated_requests(requests)

    assert benchmark.local_outcome() == BenchmarkOutcome.COMPLETE
    benchmark.apply_global_outcome(BenchmarkOutcome.COMPLETE)
    assert benchmark._run_state == RunState.COMPLETE
    assert len(benchmark._results) == 1
    assert benchmark._results[0].case.case_type == "decode"


@pytest.mark.parametrize(
    ("behavior", "outcome", "reason"),
    [
        ("missing", BenchmarkOutcome.ABORT, "KV cache manager is not available"),
        (
            "raises",
            BenchmarkOutcome.ABORT,
            "synthetic_decode_kv_allocation_failed: allocation failed",
        ),
        (
            "exhausted",
            BenchmarkOutcome.SKIP,
            "insufficient_kv_cache_for_synthetic_decode",
        ),
    ],
)
def test_decode_allocation_failures_request_canonical_outcome(
    monkeypatch, behavior, outcome, reason
):
    config = SelfBenchmarkConfig(
        mode="decode",
        decode_context_granularity=1,
        decode_batch_granularity=1,
        warmup_iterations=0,
    )
    executor = _make_executor(config)
    kv_cache_manager = executor.resource_manager._kv_cache_manager
    benchmark = SelfBenchmark(executor)
    if behavior == "missing":
        monkeypatch.setattr(
            executor.resource_manager,
            "get_resource_manager",
            lambda resource_type: (
                None if resource_type == ResourceManagerType.KV_CACHE_MANAGER else None
            ),
        )
    elif behavior == "raises":

        def _raise_allocation_error(**kwargs):
            raise RuntimeError("allocation failed")

        monkeypatch.setattr(
            kv_cache_manager,
            "add_dummy_requests",
            _raise_allocation_error,
        )
    else:
        monkeypatch.setattr(
            kv_cache_manager,
            "add_dummy_requests",
            lambda **kwargs: None,
        )

    benchmark.make_decode_requests([], [])

    assert benchmark.local_outcome() == outcome
    assert benchmark.local_diagnostic(outcome, rank=0)["reason"] == reason


@pytest.mark.parametrize("gpu_forward_time_ms", [0.0, None])
def test_process_iter_stats_propagates_gpu_forward_time_to_self_benchmark(
    gpu_forward_time_ms,
):
    from tensorrt_llm._torch.pyexecutor.py_executor import BatchState, PyExecutor

    serialized_stats = MagicMock()
    serialized_stats.to_json_str.return_value = json.dumps(
        {"inflightBatchingStats": {"numContextRequests": 1}}
    )
    observe_executed_batch = MagicMock(return_value=True)
    scheduled_requests = object()
    fake_executor = types.SimpleNamespace(
        _latest_host_step_time_ms=None,
        _latest_prev_device_step_time_ms=None,
        perf_manager=types.SimpleNamespace(
            try_compute_gpu_elapsed_time_ms=MagicMock(return_value=gpu_forward_time_ms)
        ),
        enable_iter_req_stats=False,
        enable_iter_perf_stats=True,
        _update_iter_stats=MagicMock(return_value=serialized_stats),
        self_benchmark=types.SimpleNamespace(
            active=True,
            observe_executed_batch=observe_executed_batch,
        ),
        disable_overlap_scheduler=True,
        dist=types.SimpleNamespace(pp_size=1),
        enable_attention_dp=False,
        _append_iter_stats=MagicMock(),
    )
    batch_state = BatchState(
        scheduled_requests=scheduled_requests,
        sample_state=None,
        iter_stats=object(),
    )

    PyExecutor._process_iter_stats(fake_executor, [], [], batch_state)

    observed_stats = observe_executed_batch.call_args.args[1]
    if gpu_forward_time_ms is None:
        assert "gpuForwardTimeMS" not in observed_stats
    else:
        assert observed_stats["gpuForwardTimeMS"] == gpu_forward_time_ms
    fake_executor._append_iter_stats.assert_not_called()


def test_executed_trial_sanitizes_queue_counters_and_records_result():
    benchmark, requests = _start_prefill_trial(batch_size=1)
    scheduled = _FakeScheduledBatch(last_chunk=requests)
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

    benchmark.observe_scheduled_batch(scheduled)
    assert benchmark.observe_executed_batch(scheduled, stats) is True
    benchmark.observe_finished_requests(requests)
    benchmark.observe_terminated_requests(requests)
    benchmark.apply_global_outcome(BenchmarkOutcome.COMPLETE)

    assert benchmark._current_trial is None
    assert len(benchmark._results) == 1
    recorded_stats = benchmark._results[0].iteration_stats[0]
    assert recorded_stats["numQueuedRequests"] == 0
    assert recorded_stats["inflightBatchingStats"]["numQueuedContextRequests"] == 0
    assert recorded_stats["inflightBatchingStats"]["numQueuedCtxTokens"] == 0
    assert recorded_stats["inflightBatchingStats"]["numQueuedGenRequests"] == 0
    assert recorded_stats["inflightBatchingStats"]["numQueuedGenKvTokens"] == 0


def test_executed_trial_records_cache_hit_validation():
    benchmark, requests = _start_prefill_trial(batch_size=1, kv_read_tokens=2)
    scheduled = _FakeScheduledBatch(last_chunk=requests)

    benchmark.observe_scheduled_batch(scheduled)
    benchmark.observe_executed_batch(scheduled, _prefill_stats(requests=1, tokens=2))
    benchmark.observe_finished_requests(requests)
    benchmark.observe_terminated_requests(requests)
    assert benchmark.local_outcome() == BenchmarkOutcome.COMPLETE
    benchmark.apply_global_outcome(BenchmarkOutcome.COMPLETE)

    result = benchmark._results[0]
    assert result.observed_kv_read_tokens == 2
    assert result.cache_hit_validated is True
    assert result.iteration_stats[0]["selfBenchmark"] == {
        "expectedKvReadTokens": 2,
        "observedCachedTokens": 2,
        "cacheHitValidated": True,
    }


def test_cached_prefill_signature_accepts_full_v1_context_chunk():
    benchmark, requests = _start_prefill_trial(batch_size=1, isl=8, kv_read_tokens=4)
    scheduled = _FakeScheduledBatch(last_chunk=requests)

    benchmark.observe_scheduled_batch(scheduled)

    signature = next(iter(benchmark._current_trial.expected_schedule.values()))
    assert signature.total_tokens == 8
    assert signature.expected_cached_tokens == 4
    assert signature.expected_context_chunk_size == 4
    assert requests[0].context_chunk_size == 8
    assert benchmark.local_outcome() == BenchmarkOutcome.CONTINUE


def test_cache_hit_mismatch_requests_skip_and_is_not_a_result():
    benchmark, requests = _start_prefill_trial(batch_size=1, kv_read_tokens=2)
    requests[0].cached_tokens = 0
    scheduled = _FakeScheduledBatch(last_chunk=requests)

    benchmark.observe_scheduled_batch(scheduled)
    assert benchmark.observe_executed_batch(scheduled, _prefill_stats(requests=1, tokens=2))

    assert benchmark.local_outcome() == BenchmarkOutcome.SKIP
    assert benchmark._results == []

    benchmark.apply_global_outcome(
        BenchmarkOutcome.SKIP,
        reason="cache_hit_validation_failed",
        origin_rank=0,
    )
    benchmark.observe_terminated_requests(requests)
    benchmark.apply_global_outcome(BenchmarkOutcome.COMPLETE)

    admission = benchmark._admission_records[0]
    assert admission["normalized_reason"] == "seed_cache_validation_failed"
    assert admission["source_normalized_reason"] == "seed_cache_validation_failed"
    assert admission["scheduled_batch_size"] == admission["executed_batch_size"] == 1


def test_observed_cached_tokens_requires_every_trial_request():
    benchmark, requests = _start_prefill_trial(batch_size=2, kv_read_tokens=2)
    requests[1].cached_tokens = 0
    trial = benchmark._current_trial

    assert SelfBenchmark._observed_cached_tokens(requests, trial) == 0
    assert SelfBenchmark._observed_cached_tokens(requests[:1], trial) is None


def test_terminal_schema_persists_case_plans_and_planner_reductions(tmp_path):
    output_path = tmp_path / "planner-diagnostics.json"
    config = SelfBenchmarkConfig(
        mode="prefill",
        output_path=str(output_path),
        prefill_isl_granularity=1,
        prefill_batch_granularity=2,
        warmup_iterations=0,
    )
    benchmark = SelfBenchmark(_make_executor(config, max_num_tokens=4, max_batch_size=4))

    benchmark._finish_complete()

    data = json.loads(output_path.read_text())
    assert len(data["case_plans"]) == len(data["cases"]) == 1
    case_plan = data["case_plans"][0]
    assert case_plan["case_id"] == data["cases"][0]["case_id"]
    assert case_plan["limiting_constraint"] == "microbatch_token_capacity"
    assert case_plan["token_components"]["compute_tokens_per_request"] == 4
    assert case_plan["kv_snapshot"]["target"]["free_num_blocks"] == 128
    reduction = next(event for event in data["planner_events"] if event["event"] == "axis_reduced")
    assert reduction["requested_batch_sizes"] == [1, 4]
    assert reduction["proposed_batch_sizes"] == [1]
    assert reduction["omitted_batch_sizes"] == [4]


def test_completed_trial_records_exact_admission_diagnostics():
    config = SelfBenchmarkConfig(
        mode="prefill",
        prefill_isl_granularity=2,
        prefill_batch_granularity=2,
        warmup_iterations=0,
    )
    benchmark = SelfBenchmark(_make_executor(config))
    case = next(
        case
        for case in benchmark._cases
        if case.case_type == "prefill" and case.isl == 1 and case.batch_size == 4
    )
    benchmark._cases = [case]
    benchmark._next_case_index = 0
    requests = benchmark.make_prefill_requests([], [])

    _complete_current_prefill_trial(benchmark, requests)

    admission = benchmark._results[0].admission
    assert admission["requested_batch_size"] == 4
    assert admission["proposed_batch_size"] == 4
    assert admission["scheduled_batch_size"] == 4
    assert admission["executed_batch_size"] == 4
    assert admission["microbatch_tokens_per_request"] == 1
    assert admission["microbatch_tokens_proposed"] == 4
    assert admission["limits"] == {
        "max_num_tokens": 8,
        "max_num_active_requests": 4,
        "max_batch_size": 4,
        "request_capacity": 4,
    }
    assert admission["normalized_reason"] == "exact"
    assert admission["expected_reuse_observed"] is None
    assert admission["expected_reuse_validated"] is None


def test_partial_scheduled_batch_records_normalized_reason_and_warns_once(monkeypatch):
    benchmark, requests = _start_prefill_trial(batch_size=2)
    warning = MagicMock()
    monkeypatch.setattr(self_benchmark_module.logger, "warning", warning)
    partial = _FakeScheduledBatch(last_chunk=requests[:1])

    benchmark.observe_scheduled_batch(partial)
    benchmark.observe_scheduled_batch(partial)

    trial = benchmark._current_trial
    assert trial.scheduled_batch_size == 1
    assert trial.admission_reason == "scheduled_batch_size_mismatch"
    assert warning.call_count == 1

    benchmark.apply_global_outcome(
        BenchmarkOutcome.SKIP,
        reason="scheduled_batch_size_mismatch",
        origin_rank=0,
    )
    benchmark.observe_terminated_requests(requests)
    benchmark.apply_global_outcome(BenchmarkOutcome.COMPLETE)

    admission = benchmark._admission_records[0]
    assert admission["proposed_batch_size"] == 2
    assert admission["scheduled_batch_size"] == 1
    assert admission["executed_batch_size"] == 0
    assert admission["normalized_reason"] == "scheduled_batch_size_mismatch"


def test_chunked_prefill_records_context_chunked_reason():
    benchmark, requests = _start_prefill_trial(batch_size=2)

    benchmark.observe_scheduled_batch(_FakeScheduledBatch(chunking=requests))

    assert benchmark.local_outcome() == BenchmarkOutcome.SKIP
    assert benchmark._current_trial.scheduled_batch_size == 2
    assert benchmark._current_trial.admission_reason == "context_chunked"
    assert benchmark.local_diagnostic(BenchmarkOutcome.SKIP, rank=0)["reason"] == (
        "context_chunked"
    )


def test_partial_executed_batch_records_executed_batch_size_mismatch():
    benchmark, requests = _start_prefill_trial(batch_size=2)
    scheduled = _FakeScheduledBatch(last_chunk=requests)
    benchmark.observe_scheduled_batch(scheduled)

    benchmark.observe_executed_batch(
        _FakeScheduledBatch(last_chunk=requests[:1]),
        _prefill_stats(requests=1, tokens=4),
    )

    assert benchmark.local_outcome() == BenchmarkOutcome.SKIP
    assert benchmark._current_trial.executed_batch_size == 1
    assert benchmark._current_trial.admission_reason == "executed_batch_size_mismatch"


def test_decode_kv_exhaustion_records_kv_capacity_reason(monkeypatch):
    config = SelfBenchmarkConfig(
        mode="decode",
        decode_context_granularity=1,
        decode_batch_granularity=1,
        warmup_iterations=0,
    )
    executor = _make_executor(config)
    benchmark = SelfBenchmark(executor)
    kv_cache_manager = executor.resource_manager._kv_cache_manager
    monkeypatch.setattr(kv_cache_manager, "add_dummy_requests", lambda **kwargs: None)

    benchmark.make_decode_requests([], [])

    assert benchmark._current_trial.admission_reason == "kv_capacity"
    benchmark.apply_global_outcome(
        BenchmarkOutcome.SKIP,
        reason="insufficient_kv_cache_for_synthetic_decode",
        origin_rank=0,
    )
    benchmark.apply_global_outcome(BenchmarkOutcome.COMPLETE)
    assert benchmark._admission_records[0]["normalized_reason"] == "kv_capacity"


def test_non_origin_rank_records_rank_consensus_skip(monkeypatch):
    benchmark, requests = _start_prefill_trial(batch_size=2)
    warning = MagicMock()
    monkeypatch.setattr(self_benchmark_module.logger, "warning", warning)

    benchmark.apply_global_outcome(
        BenchmarkOutcome.SKIP,
        reason="scheduled_batch_size_mismatch",
        origin_rank=1,
    )
    benchmark.observe_terminated_requests(requests)
    benchmark.apply_global_outcome(BenchmarkOutcome.COMPLETE)

    admission = benchmark._admission_records[0]
    assert admission["normalized_reason"] == "rank_consensus_skip"
    assert admission["source_normalized_reason"] == "scheduled_batch_size_mismatch"
    assert admission["origin_rank"] == 1
    warning.assert_called_once()
    assert warning.call_args.args[8:] == (
        "rank_consensus_skip",
        "scheduled_batch_size_mismatch",
    )


def test_schema_v2_output_uses_case_and_trial_id(tmp_path):
    output_path = tmp_path / "benchmark.json"
    config = SelfBenchmarkConfig(mode="prefill", output_path=str(output_path), warmup_iterations=0)
    benchmark = SelfBenchmark(_make_executor(config))
    case = BenchmarkCase(case_type="prefill", case_id=0, isl=8)
    benchmark._cases = [case]
    benchmark._results = [
        BenchmarkTrialResult(
            trial_id=3,
            case=case,
            iteration_stats=({"inflightBatchingStats": {"numContextRequests": 1}},),
            observed_kv_read_tokens=0,
            cache_hit_validated=True,
        )
    ]

    benchmark._finish_complete()

    data = json.loads(output_path.read_text())
    assert data["schema_version"] == 2
    assert data["config"]["mode"] == "prefill"
    assert data["status"] == "complete"
    assert data["valid"] is True
    assert data["coverage"] == {
        "expected_cases": 1,
        "completed_trials": 1,
        "skipped_cases": 0,
    }
    assert data["limits"]["max_num_scheduled_tokens"] == 8
    assert data["limits"]["tokens_per_block"] == 32
    assert data["results"][0]["trial_id"] == 3
    assert data["results"][0]["case"]["isl"] == 8
    # `point` mirrors `case` in Dynamo's vocabulary so the TRT-LLM self-benchmark
    # consumer (dynamo/trtllm/self_benchmark.py) can normalize results to FPM.
    assert data["results"][0]["point"] == {
        "point_type": "prefill",
        "isl": 8,
        "kv_read_tokens": 0,
        "context_length": 0,
        "batch_size": 1,
    }
    assert data["results"][0]["observed_kv_read_tokens"] == 0
    assert data["results"][0]["cache_hit_validated"] is True


@pytest.mark.parametrize(
    ("finish", "status"),
    [
        ("complete", "complete"),
        ("aborted", "aborted"),
        ("interrupted", "interrupted"),
    ],
)
def test_terminal_artifacts_use_schema_v2(tmp_path, finish, status):
    config = SelfBenchmarkConfig(
        mode="prefill",
        output_path=str(tmp_path / "benchmark.json"),
        warmup_iterations=0,
    )
    benchmark = SelfBenchmark(_make_executor(config))
    benchmark._cases = []
    output_path = Path(benchmark._output_path)
    running = json.loads(output_path.read_text())

    getattr(benchmark, f"_finish_{finish}")()

    terminal = json.loads(output_path.read_text())
    assert running["schema_version"] == terminal["schema_version"] == 2
    assert running["run_id"] == terminal["run_id"]
    assert terminal["status"] == status
    assert terminal["valid"] is (status == "complete")


def test_skipped_case_is_diagnostic_not_result(tmp_path):
    config = SelfBenchmarkConfig(
        mode="prefill",
        output_path=str(tmp_path / "benchmark.json"),
        warmup_iterations=0,
    )
    benchmark = SelfBenchmark(_make_executor(config))
    benchmark._cases = [
        BenchmarkCase(case_type="prefill", case_id=case_id, isl=1) for case_id in range(3)
    ]
    benchmark._results = [
        BenchmarkTrialResult(
            trial_id=case_id,
            case=benchmark._cases[case_id],
            iteration_stats=({},),
        )
        for case_id in (0, 2)
    ]
    benchmark._skipped_cases = [
        {
            "trial_id": 1,
            "case": asdict(benchmark._cases[1]),
            "reason": "scheduled_batch_shape_mismatch",
            "origin_rank": 1,
        }
    ]

    benchmark._finish_complete()

    data = json.loads(Path(benchmark._output_path).read_text())
    assert data["status"] == "complete"
    assert data["valid"] is False
    assert data["coverage"] == {
        "expected_cases": 3,
        "completed_trials": 2,
        "skipped_cases": 1,
    }
    assert data["skipped_cases"][0]["origin_rank"] == 1
    assert data["skipped_cases"][0]["case"] not in [entry["case"] for entry in data["results"]]
    # Skipped cases also carry the Dynamo-facing `point`/`skipped_reason` keys.
    assert data["skipped_cases"][0]["point"]["point_type"] == "prefill"
    assert data["skipped_cases"][0]["skipped_reason"] == "scheduled_batch_shape_mismatch"
