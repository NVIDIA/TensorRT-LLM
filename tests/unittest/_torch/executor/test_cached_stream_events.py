# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_Q35_CACHED_STREAM_EVENTS: reused stream-ordering events still order the streams."""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import py_executor
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("cached", [False, True])
def test_input_copy_event_reuse(monkeypatch, cached):
    if cached:
        monkeypatch.setenv(py_executor.CACHED_STREAM_EVENTS_ENV_VAR_NAME, "1")
    else:
        monkeypatch.delenv(py_executor.CACHED_STREAM_EVENTS_ENV_VAR_NAME, raising=False)
    preparer = object.__new__(DecoderInputPreparer)
    preparer._prepare_inputs_event = None
    preparer.record_input_copy()
    first = preparer._prepare_inputs_event
    preparer.wait_for_input_copy()
    preparer.record_input_copy()
    assert (preparer._prepare_inputs_event is first) is cached
    preparer.wait_for_input_copy()


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
    executor = object.__new__(PyExecutor)
    events = executor._cached_stream_events()
    assert executor._cached_stream_events() is events
    for _ in range(3):  # the same event, re-recorded every round
        assert _ordered(True, events)
    assert not _ordered(False, events)  # negative control
