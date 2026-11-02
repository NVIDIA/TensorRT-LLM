# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contracts for switching a stopped executor from profiling to serving resources."""

import threading
from contextlib import nullcontext
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import py_executor as executor_module
from tensorrt_llm._torch.pyexecutor.py_executor import (
    PyExecutor,
    PyExecutorResources,
    _ExecutorExecutionConfig,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.sampler import AsyncWorkerMixin

pytestmark = pytest.mark.cpu_only


class _Engine(SimpleNamespace):
    @property
    def attn_metadata(self):
        return self._attn_metadata


@pytest.fixture
def executor(monkeypatch):
    events = []
    engine = _Engine(
        _attn_metadata=object(),
        _release_cuda_graphs=lambda: events.append("graphs"),
    )

    def release_engine_resources(*, release_model_owned_graphs):
        engine._release_cuda_graphs()
        engine._attn_metadata = None

    engine.release_executor_resources = Mock(side_effect=release_engine_resources)
    cache = SimpleNamespace(shutdown=lambda: events.append("cache"))
    transceiver = SimpleNamespace(shutdown=lambda: events.append("unregister"))
    ex = object.__new__(PyExecutor)
    ex._closed = False
    ex.enable_attention_dp = False
    ex._kv_connector_failed = False
    ex._resources_released = False
    ex._event_loop_error = None
    ex._fatal_error = None
    ex.model_engine = engine
    ex.draft_model_engine = None
    ex.execution_stream = object()
    ex.dist = SimpleNamespace(pp_size=1)
    ex.worker_started = True
    ex._mpi_progress_thread = None
    ex._mpi_progress_thread_exit_hook = None
    ex.worker_thread = Mock()
    ex.worker_thread.join.side_effect = lambda: events.append("join")
    ex.executor_request_queue = Mock()
    ex.executor_request_queue.enqueue_shutdown_request.side_effect = lambda: events.append("stop")
    ex.shutdown_event = threading.Event()
    ex.shutdown_event.set()
    ex._profile_manager = Mock()
    ex.hang_detector = Mock()
    ex.hang_detector.detected.return_value = False
    ex._shutdown_sleep_wakeup_listeners = Mock()
    ex.encoder_launch_executor = None
    ex.kv_connector_manager = None
    ex.kv_cache_transceiver = transceiver
    ex.resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: cache})
    ex.virtual_memory_pools = None
    ex.sampler = object()
    ex.dwdp_manager = None
    ex.pending_encoder_steps = [object()]
    ex.active_requests = [object()]
    ex.responses = {17: object()}
    ex.result_wait_queues = {17: object()}
    ex._pending_transfer_responses = [object()]
    ex._pending_response_terminations = [object()]
    ex.request_accumulated = [object()]
    ex.control_requests = [object()]
    ex.waiting_queue = [object()]
    ex._pending_adp_dummy_request = object()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("synchronize"))
    return ex, events


def test_release_preserves_model_and_orders_gpu_teardown(executor):
    ex, events = executor
    draft_engine = _Engine(
        _attn_metadata=object(),
        _release_cuda_graphs=lambda: events.append("draft_graphs"),
    )

    def release_draft_resources(*, release_model_owned_graphs):
        draft_engine._release_cuda_graphs()
        draft_engine._attn_metadata = None

    draft_engine.release_executor_resources = Mock(side_effect=release_draft_resources)
    ex.draft_model_engine = draft_engine
    ex.sampler = Mock(spec=AsyncWorkerMixin)
    ex.sampler.async_worker_enabled.return_value = True
    ex.sampler.async_worker_stop.side_effect = lambda: events.append("sampler_stop")
    engine, stream, sampler = ex.model_engine, ex.execution_stream, ex.sampler
    ex.release_resources()

    assert events == [
        "stop",
        "join",
        "sampler_stop",
        "graphs",
        "draft_graphs",
        "synchronize",
        "unregister",
        "cache",
    ]
    assert ex.draft_model_engine is draft_engine
    assert draft_engine.attn_metadata is None
    assert ex.model_engine is engine
    assert ex.execution_stream is stream
    assert ex.sampler is sampler
    assert engine.attn_metadata is None
    assert ex.resource_manager is None
    assert ex.scheduler is None
    assert ex.kv_cache_transceiver is None
    assert ex.hang_detector is None  # Drops the old transceiver's status-provider reference.
    assert not ex.responses and not ex.active_requests and not ex.result_wait_queues
    assert not ex.pending_encoder_steps
    assert not ex.request_accumulated and not ex.control_requests
    assert ex.waiting_queue is None and ex._pending_adp_dummy_request is None
    assert ex._resources_released and not ex._closed


def test_release_before_worker_start_does_not_wait_for_shutdown(executor):
    ex, events = executor
    ex.worker_started = False
    ex.shutdown_event.clear()
    ex.release_resources()
    assert events == ["graphs", "synchronize", "unregister", "cache"]
    ex.worker_thread.join.assert_not_called()


def test_final_shutdown_after_release_does_not_release_resources_twice(executor):
    ex, events = executor
    ex.release_resources()
    before = events.copy()
    with pytest.raises(RuntimeError, match="already released"):
        ex.release_resources()
    assert events == before
    engine = ex.model_engine
    engine.release_executor_resources.assert_called_once_with(release_model_owned_graphs=False)
    engine.release_executor_resources.reset_mock(side_effect=True)
    ex.shutdown()
    ex.shutdown()
    engine.release_executor_resources.assert_called_once_with(release_model_owned_graphs=True)
    assert events == before + ["synchronize"]
    assert ex._closed
    assert not hasattr(ex, "model_engine")


@pytest.mark.parametrize("closed,released", [(False, False), (True, False), (True, True)])
def test_rebind_requires_released_open_executor(executor, closed, released):
    ex, _ = executor
    ex._closed = closed
    ex._resources_released = released
    ex._initialize_execution = Mock()
    with pytest.raises(RuntimeError, match="released before rebinding"):
        ex.rebind_resources(Mock(spec=PyExecutorResources))
    ex._initialize_execution.assert_not_called()


@pytest.mark.parametrize("error_field", ["_event_loop_error", "_fatal_error"])
def test_failed_executor_cannot_resume_with_new_resources(executor, error_field):
    ex, _ = executor
    ex.release_resources()
    setattr(ex, error_field, RuntimeError("profiling failed"))
    ex._initialize_execution = Mock()
    with pytest.raises(RuntimeError, match="execution failure"):
        ex.rebind_resources(Mock(spec=PyExecutorResources))
    ex._initialize_execution.assert_not_called()


def test_rebind_resets_phase_state_and_profile_windows(executor, monkeypatch):
    ex, _ = executor
    create_waiting_queue = Mock(wraps=executor_module.create_waiting_queue)
    monkeypatch.setattr(executor_module, "create_waiting_queue", create_waiting_queue)
    monkeypatch.setenv("TLLM_PROFILE_START_STOP", "2-4")
    executor_module._load_iteration_indexes.cache_clear()
    monkeypatch.setattr(torch.cuda, "Stream", Mock())
    monkeypatch.setattr(torch.cuda, "Event", Mock())
    monkeypatch.setattr(torch.cuda, "current_stream", Mock())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(executor_module, "_distributed_warmup_guard", lambda *args: nullcontext())
    monkeypatch.setattr(executor_module.ADPRouter, "create", Mock())
    monkeypatch.setattr(executor_module, "HangDetector", Mock())
    monkeypatch.setattr(executor_module, "ExecutorRequestQueue", Mock())
    monkeypatch.setattr(executor_module, "RequestBroadcaster", Mock())
    monkeypatch.setattr(executor_module, "mpi_disabled", lambda: True)
    monkeypatch.setattr(executor_module.GpuKeepalive, "create_from_env", Mock(return_value=None))
    monkeypatch.setattr(
        executor_module, "create_executor_hang_diagnostics", Mock(return_value=(None, None))
    )
    for name in (
        "_emit_initial_stats",
        "_set_global_steady_clock_offset",
        "_initialize_remote_tail_adp",
        "_maybe_init_kv_connector_manager",
        "_validate_moe_a2a_checkpoint_support",
        "_warn_if_kv_block_budget_unchecked",
    ):
        monkeypatch.setattr(ex, name, Mock())
    ex.device_id = 0
    ex.global_rank = 0
    ex.dist = SimpleNamespace(pp_size=1, rank=0, mapping=Mock(), barrier=Mock())
    ex.execution_stream = Mock()
    ex.drafter = None
    ex.guided_decoder = None
    ex._execution_config = _ExecutorExecutionConfig(
        disable_overlap_scheduler=True,
        enable_early_first_token_response=False,
        max_input_len=128,
        max_batch_size=2,
        max_beam_width=1,
        max_draft_len=0,
        max_total_draft_tokens=0,
        garbage_collection_gen0_threshold=None,
        resource_governor_queue=None,
        hang_detection_timeout=None,
        enable_kv_pool_rebalance=False,
    )
    ex.model_engine.enable_attention_dp = False
    ex.model_engine.mm_encoder_item_scheduling_enabled = False
    ex.model_engine.enable_in_graph_sampling = False
    ex.model_engine.is_spec_decode = False
    ex.model_engine.spec_config = None
    ex.model_engine.enable_overlap_headroom = False
    ex.model_engine.warmup_scope = nullcontext
    ex.model_engine.register_beam_histories_callable = Mock()
    ex.sampler = Mock(spec=["start_beam_histories"])
    ex.model_engine.model = SimpleNamespace(model_config=SimpleNamespace(is_encoder_decoder=False))
    ex.model_engine.get_max_num_sequences = Mock(return_value=2)
    ex.model_engine.warmup = Mock()
    ex.model_engine.llm_args = SimpleNamespace(
        kv_connector_config=None,
        max_stats_len=0,
        max_num_tokens=128,
        print_iter_log=False,
        enable_iter_perf_stats=False,
        enable_iter_req_stats=False,
        iter_perf_stats_interval=1,
        stream_interval=1,
        attention_dp_config=None,
        batch_wait_timeout_ms=0,
        batch_wait_timeout_iters=0,
        batch_wait_max_tokens_ratio=0,
    )
    cache = Mock(
        event_buffer_max_size=0,
        streaming_kv_events_enabled=False,
        enable_block_reuse=False,
        tokens_per_block=16,
    )
    cache.get_kv_cache_stats.return_value = SimpleNamespace(max_num_blocks=8, tokens_per_block=16)
    resources = PyExecutorResources(
        resource_manager=ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: cache}),
        scheduler=Mock(ctx_demote_uncached_tokens=0),
        kv_cache_transceiver=None,
        peft_cache_config=None,
        virtual_memory_pools=None,
        kv_connector_manager=None,
        dwdp_manager=None,
        max_seq_len=128,
        max_num_sequences=2,
        waiting_queue_policy=executor_module.WaitingQueuePolicy.FCFS,
    )
    ex.iter_counter = 42
    ex._ctx_mla_kv_len_cap = None
    ex.profile_start_iters = {99}
    ex.profile_stop_iters = set()
    old_shutdown_event = ex.shutdown_event
    engine, stream = ex.model_engine, ex.execution_stream
    ex.release_resources()
    bound_fields = {
        "resource_manager",
        "scheduler",
        "kv_cache_transceiver",
        "peft_cache_config",
        "virtual_memory_pools",
        "kv_connector_manager",
        "dwdp_manager",
        "max_seq_len",
    }
    # Queue policy constructs a queue; sequence capacity comes from the engine.
    assert {field.name for field in fields(PyExecutorResources)} == bound_fields | {
        "waiting_queue_policy",
        "max_num_sequences",
    }, "New phase inputs need binding and lifecycle coverage in this test."
    # Stale values also catch omitted assignments for optional resources set to None.
    for name in bound_fields:
        setattr(ex, name, object())
    ex._disagg_coordinator = object()
    ex.start_worker = Mock()
    try:
        ex.rebind_resources(resources)
        assert ex.iter_counter == 0
        assert ex.profile_start_iters == {2}
        assert ex.profile_stop_iters == {4}
        assert ex._ctx_mla_kv_len_cap == "unset"
        assert ex._disagg_coordinator is None
        assert not ex.worker_started
        assert ex.shutdown_event is not old_shutdown_event
        assert not ex.shutdown_event.is_set()
        for name in bound_fields:
            assert getattr(ex, name) is getattr(resources, name), name
        create_waiting_queue.assert_called_once_with(
            resources.waiting_queue_policy, gen_admission=ex._gen_admission
        )
        engine.register_beam_histories_callable.assert_called_once_with(
            ex.sampler.start_beam_histories
        )
        assert ex.max_num_active_requests == engine.get_max_num_sequences.return_value
        assert ex.kv_cache_manager is cache
        assert ex.model_engine is engine and ex.execution_stream is stream
        assert not ex._resources_released
        engine.warmup.assert_called_once_with(resources.resource_manager)
        ex.start_worker.assert_not_called()
    finally:
        executor_module._load_iteration_indexes.cache_clear()


def test_failed_rebind_cannot_be_retried_over_partial_resources(executor):
    ex, _ = executor
    ex.release_resources()
    cache = Mock()
    resources = Mock(spec=PyExecutorResources)
    resources.resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: cache})

    def fail_warmup(bound_resources):
        ex.resource_manager = bound_resources.resource_manager
        raise RuntimeError("warmup failed")

    ex._initialize_execution = Mock(side_effect=fail_warmup)
    with pytest.raises(RuntimeError, match="warmup failed"):
        ex.rebind_resources(resources)
    with pytest.raises(RuntimeError, match="released before rebinding"):
        ex.rebind_resources(resources)
    ex._initialize_execution.assert_called_once()
    assert ex._closed
    cache.shutdown.assert_called_once_with()


def test_hung_worker_does_not_release_live_resources(executor):
    ex, events = executor
    ex.hang_detector.detected.return_value = True
    ex._pp_mpi_progress_stop = threading.Event()
    progress_thread = Mock()
    ex._mpi_progress_thread = progress_thread
    ex._mpi_progress_thread_exit_hook = progress_thread.stop
    with pytest.raises(RuntimeError, match="executor hang"):
        ex.release_resources()
    assert events == ["stop"]
    progress_thread.stop.assert_called_once_with()
    assert ex._mpi_progress_thread is None
    assert ex.resource_manager is not None
    assert not ex._resources_released


def test_transfer_shutdown_failure_keeps_cache_owned(executor):
    ex, events = executor
    manager = ex.resource_manager
    ex.kv_cache_transceiver.shutdown = Mock(side_effect=RuntimeError("DMA still active"))
    with pytest.raises(RuntimeError, match="DMA still active"):
        ex.release_resources()
    assert "cache" not in events
    assert ex.resource_manager is manager
    assert not ex._resources_released


def test_release_stops_pipeline_progress_before_gpu_teardown(executor):
    ex, events = executor
    ex.dist.pp_size = 2
    ex._pp_mpi_progress_stop = threading.Event()
    ex._pp_mpi_progress_quiesced = threading.Event()
    ex.executed_batch_queue = Mock()
    ex.broadcast_sample_state_handler = Mock()
    ex.broadcast_sample_state_handler.join.side_effect = lambda: events.append("relay_join")
    progress_thread = Mock()
    progress_thread.stop.side_effect = lambda: events.append("progress_stop")
    ex._mpi_progress_thread = progress_thread
    ex._mpi_progress_thread_exit_hook = progress_thread.stop

    ex.release_resources()

    assert events.index("join") < events.index("relay_join")
    assert events.index("relay_join") < events.index("progress_stop") < events.index("graphs")
    assert ex._pp_mpi_progress_stop.is_set()
    assert ex._pp_mpi_progress_quiesced.is_set()
    assert ex._mpi_progress_thread is None
    assert ex._mpi_progress_thread_exit_hook is None
