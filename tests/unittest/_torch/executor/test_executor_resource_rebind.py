# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contracts for switching a stopped executor from profiling to serving resources."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor, PyExecutorResources
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.sampler import AsyncWorkerMixin

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def executor(monkeypatch):
    events = []
    engine = SimpleNamespace(
        attn_metadata=object(),
        _release_cuda_graphs=lambda: events.append("graphs"),
    )
    cache = SimpleNamespace(shutdown=lambda: events.append("cache"))
    transceiver = SimpleNamespace(shutdown=lambda: events.append("unregister"))
    ex = object.__new__(PyExecutor)
    ex._closed = False
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
    draft_engine = SimpleNamespace(
        attn_metadata=object(),
        _release_cuda_graphs=lambda: events.append("draft_graphs"),
    )
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
    ex.shutdown()
    ex.shutdown()
    assert events == before
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


def test_rebind_uses_same_executor_and_does_not_start_worker(executor):
    ex, _ = executor
    engine, stream = ex.model_engine, ex.execution_stream
    ex.release_resources()
    resources = Mock(spec=PyExecutorResources)
    ex._initialize_execution = Mock()
    ex.start_worker = Mock()
    assert ex.rebind_resources(resources) is None
    ex._initialize_execution.assert_called_once_with(resources)
    ex.start_worker.assert_not_called()
    assert ex.model_engine is engine and ex.execution_stream is stream
    assert not ex._resources_released


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
