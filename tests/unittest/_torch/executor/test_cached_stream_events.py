# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_Q35_CACHED_STREAM_EVENTS: _forward_step's two stream-ordering events are created once and reused, and a
re-recorded event orders the waiting stream after the work recorded before the wait (negative control: without the
wait the reader can run before the writer)."""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import py_executor
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_events_created_once(monkeypatch):
    ex = object.__new__(PyExecutor)
    monkeypatch.delenv(py_executor.CACHED_STREAM_EVENTS_ENV_VAR_NAME, raising=False)
    assert ex._q35_stream_events() is None
    monkeypatch.setenv(py_executor.CACHED_STREAM_EVENTS_ENV_VAR_NAME, "1")
    events = ex._q35_stream_events()
    assert len(events) == 2 and ex._q35_stream_events() is events


def _ordered(wait: bool, events) -> bool:
    """Writer on stream a (after a long sleep), reader on stream b: True when b read the written value."""
    a, b = torch.cuda.Stream(), torch.cuda.Stream()
    x = torch.zeros(1, device="cuda")
    out = torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    with torch.cuda.stream(a):
        torch.cuda._sleep(200_000_000)
        x.fill_(1.0)
    if wait:
        events[0].record(a)
        b.wait_event(events[0])
    with torch.cuda.stream(b):
        out.copy_(x)
    torch.cuda.synchronize()
    return out.item() == 1.0


def test_reused_event_orders_streams(monkeypatch):
    monkeypatch.setenv(py_executor.CACHED_STREAM_EVENTS_ENV_VAR_NAME, "1")
    events = object.__new__(PyExecutor)._q35_stream_events()
    for _ in range(3):  # the same event, re-recorded every round
        assert _ordered(True, events)
    assert not _ordered(False, events)  # negative control


def test_forward_on_current_stream(monkeypatch):
    """TRTLLM_OVERLAP_LOOP_ON_EXECUTION_STREAM: _forward_step skips its record / wait pairs only when the caller's
    current stream is the execution stream (flag on); any other stream keeps the cross-stream ordering."""
    ex = object.__new__(PyExecutor)
    ex.execution_stream = torch.cuda.Stream()
    other = torch.cuda.Stream()
    monkeypatch.delenv(py_executor.OVERLAP_ON_EXECUTION_STREAM_ENV_VAR_NAME, raising=False)
    with torch.cuda.stream(ex.execution_stream):
        assert not ex._q35_forward_on_current_stream()
    monkeypatch.setenv(py_executor.OVERLAP_ON_EXECUTION_STREAM_ENV_VAR_NAME, "1")
    with torch.cuda.stream(ex.execution_stream):
        assert ex._q35_forward_on_current_stream()
    with torch.cuda.stream(other):
        assert not ex._q35_forward_on_current_stream()
    assert not ex._q35_forward_on_current_stream()  # the default stream
