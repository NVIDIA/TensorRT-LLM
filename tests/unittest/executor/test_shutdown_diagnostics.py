# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import threading

import pytest

from tensorrt_llm.executor.shutdown_diagnostics import shutdown_watchdog, wait_for_shutdown_event
from tensorrt_llm.executor.worker import _verify_required_shm_trace_is_loaded


def test_required_shm_trace_rejects_uninstrumented_worker(monkeypatch):
    monkeypatch.setenv("TLLM_SHM_TRACE_REQUIRED", "1")
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO("python\n"))

    with pytest.raises(RuntimeError, match="not loaded"):
        _verify_required_shm_trace_is_loaded()


def test_required_shm_trace_accepts_instrumented_worker(monkeypatch):
    monkeypatch.setenv("TLLM_SHM_TRACE_REQUIRED", "1")
    process_maps = "7f00-7f01 /tmp/libnvbug6336747_shm_trace_1_0.so\n"
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO(process_maps))

    _verify_required_shm_trace_is_loaded()


def test_wait_for_shutdown_event_is_unbounded_by_default(monkeypatch):
    monkeypatch.delenv("TLLM_SHUTDOWN_TRACE_TIMEOUT_SEC", raising=False)
    event = threading.Event()
    event.set()

    wait_for_shutdown_event(event, "test", "already_set")


def test_wait_for_shutdown_event_timeout_dumps_stacks(monkeypatch, tmp_path):
    monkeypatch.setenv("TLLM_SHUTDOWN_TRACE_TIMEOUT_SEC", "0.01")
    monkeypatch.setenv("TLLM_SHUTDOWN_TRACE_DIR", str(tmp_path))

    with pytest.raises(TimeoutError, match="timed out"):
        wait_for_shutdown_event(threading.Event(), "test", "blocked")

    trace = "".join(path.read_text() for path in tmp_path.glob("shutdown_trace.*.log"))
    assert "phase=blocked state=timeout" in trace
    assert list(tmp_path.glob("shutdown_stacks.*.test.blocked.*.log"))


def test_shutdown_watchdog_records_completed_phase(monkeypatch, tmp_path):
    monkeypatch.setenv("TLLM_SHUTDOWN_TRACE_TIMEOUT_SEC", "1")
    monkeypatch.setenv("TLLM_SHUTDOWN_TRACE_DIR", str(tmp_path))
    monkeypatch.delenv("TLLM_SHUTDOWN_FORCE_EXIT", raising=False)

    with shutdown_watchdog("test", "quick"):
        pass

    trace = "".join(path.read_text() for path in tmp_path.glob("shutdown_trace.*.log"))
    assert "phase=quick state=begin" in trace
    assert "phase=quick state=end" in trace
