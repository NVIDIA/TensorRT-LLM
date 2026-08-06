# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import py_executor


def _runtime_executor(*, enabled=True):
    executor = object.__new__(py_executor.PyExecutor)
    executor._runtime_mem_profile_enabled = enabled
    executor._last_runtime_mem_snapshot_ns = None
    executor.iter_counter = 7
    executor.active_requests = [Mock(), Mock()]
    executor.num_scheduled_requests = 3
    executor.dist = SimpleNamespace(pp_size=1)
    executor.disable_overlap_scheduler = False
    executor.kv_cache_manager = Mock()
    executor.kv_cache_manager.get_kv_cache_stats.return_value = SimpleNamespace(
        used_num_blocks=8,
        free_num_blocks=2,
        max_num_blocks=10,
    )
    return executor


def test_runtime_mem_snapshot_is_disabled_without_overhead(monkeypatch) -> None:
    executor = _runtime_executor(enabled=False)
    monotonic_ns = Mock()
    snapshot = Mock()
    monkeypatch.setattr(py_executor.time, "monotonic_ns", monotonic_ns)
    monkeypatch.setattr(py_executor, "log_mem_snapshot", snapshot)

    executor._maybe_log_runtime_mem_snapshot()

    monotonic_ns.assert_not_called()
    executor.kv_cache_manager.get_kv_cache_stats.assert_not_called()
    snapshot.assert_not_called()


def test_runtime_mem_snapshot_is_throttled_and_has_context(monkeypatch) -> None:
    executor = _runtime_executor()
    monkeypatch.setattr(
        py_executor.time,
        "monotonic_ns",
        Mock(side_effect=[1_000_000_000, 1_500_000_000, 2_000_000_000]),
    )
    snapshot = Mock()
    monkeypatch.setattr(py_executor, "log_mem_snapshot", snapshot)

    executor._maybe_log_runtime_mem_snapshot()
    executor._maybe_log_runtime_mem_snapshot()
    executor._maybe_log_runtime_mem_snapshot()

    expected = call(
        "runtime/before_iter",
        iter=7,
        loop="overlap",
        active_requests=2,
        prev_scheduled_requests=3,
        kv_used_blocks=8,
        kv_free_blocks=2,
        kv_max_blocks=10,
    )
    assert snapshot.call_args_list == [expected, expected]
    assert executor.kv_cache_manager.get_kv_cache_stats.call_count == 2


def test_runtime_oom_logs_current_snapshot_and_history(monkeypatch) -> None:
    executor = _runtime_executor()
    snapshot = Mock()
    history = Mock()
    monkeypatch.setattr(py_executor, "log_mem_snapshot", snapshot)
    monkeypatch.setattr(py_executor, "log_mem_history", history)

    executor._log_runtime_oom(torch.OutOfMemoryError("CUDA out of memory"), "forward")

    snapshot.assert_called_once_with(
        "oom/runtime/forward",
        force=True,
        iter=7,
        active_requests=2,
    )
    history.assert_called_once_with("oom/runtime/forward")


def test_non_oom_does_not_log_memory_diagnostics(monkeypatch) -> None:
    executor = _runtime_executor()
    snapshot = Mock()
    history = Mock()
    monkeypatch.setattr(py_executor, "log_mem_snapshot", snapshot)
    monkeypatch.setattr(py_executor, "log_mem_history", history)

    executor._log_runtime_oom(RuntimeError("invalid request"), "event_loop")

    snapshot.assert_not_called()
    history.assert_not_called()


def test_handled_oom_logs_before_error_processing() -> None:
    executor = _runtime_executor()
    executor._log_runtime_oom = Mock()
    executor._fatal_error = None
    executor._error_budget = Mock(budget=1.0)
    executor._error_budget.consume.return_value = False
    executor.active_requests = []
    executor._enqueue_responses = Mock()
    calls = Mock()
    calls.attach_mock(executor._log_runtime_oom, "diagnostic")
    calls.attach_mock(executor._error_budget.consume, "budget")

    py_executor.PyExecutor._handle_errors(executor, "CUDA out of memory")

    assert calls.mock_calls[:2] == [
        call.diagnostic("CUDA out of memory", "handled"),
        call.budget("CUDA out of memory"),
    ]


def test_event_loop_oom_logs_and_reraises_original(monkeypatch) -> None:
    executor = object.__new__(py_executor.PyExecutor)
    oom = torch.OutOfMemoryError("CUDA out of memory")
    executor._is_warmup = False
    executor.garbage_collection_gen0_threshold = 0
    executor.event_loop = Mock(side_effect=oom)
    executor._log_runtime_oom = Mock()
    executor._executor_loop_cleanup = Mock()
    monkeypatch.setattr(py_executor, "host_profiler_context", lambda **_: nullcontext())
    monkeypatch.setattr(py_executor, "customized_gc_thresholds", lambda _: nullcontext())

    with pytest.raises(torch.OutOfMemoryError) as raised:
        py_executor.PyExecutor._event_loop_wrapper(executor)

    assert raised.value is oom
    executor._log_runtime_oom.assert_called_once_with(oom, "event_loop")
    executor._executor_loop_cleanup.assert_called_once_with()
