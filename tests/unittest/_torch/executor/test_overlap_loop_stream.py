# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_OVERLAP_LOOP_ON_EXECUTION_STREAM: only a speculative-decoding overlap loop runs on the
execution stream, and only then does the forward skip its cross-stream waits."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import py_executor
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor


@pytest.mark.parametrize(
    "env, speculative, on_execution_stream",
    [
        ("1", True, True),
        ("1", False, False),
        (None, True, False),
    ],
)
def test_overlap_loop_stream(monkeypatch, env, speculative, on_execution_stream):
    if env is None:
        monkeypatch.delenv(py_executor.OVERLAP_ON_EXECUTION_STREAM_ENV_VAR_NAME, raising=False)
    else:
        monkeypatch.setenv(py_executor.OVERLAP_ON_EXECUTION_STREAM_ENV_VAR_NAME, env)
    executor = object.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(spec_config=object() if speculative else None)
    executor.execution_stream = object()
    entered = []
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: entered.append(stream) or "context")

    context = executor._overlap_loop_stream()

    if on_execution_stream:
        assert context == "context" and entered == [executor.execution_stream]
    else:
        assert isinstance(context, nullcontext) and not entered


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_forward_on_current_stream(monkeypatch):
    executor = object.__new__(PyExecutor)
    executor.execution_stream = torch.cuda.Stream()
    other = torch.cuda.Stream()
    monkeypatch.delenv(py_executor.OVERLAP_ON_EXECUTION_STREAM_ENV_VAR_NAME, raising=False)
    with torch.cuda.stream(executor.execution_stream):
        assert not executor._forward_on_current_stream()
    monkeypatch.setenv(py_executor.OVERLAP_ON_EXECUTION_STREAM_ENV_VAR_NAME, "1")
    with torch.cuda.stream(executor.execution_stream):
        assert executor._forward_on_current_stream()
    with torch.cuda.stream(other):
        assert not executor._forward_on_current_stream()
    assert not executor._forward_on_current_stream()  # the default stream
