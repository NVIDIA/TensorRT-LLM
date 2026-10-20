# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_OVERLAP_LOOP_ON_EXECUTION_STREAM: the overlap loop of a speculative-decoding executor runs
with its execution stream as the current stream; every other executor keeps the thread's stream."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import py_executor
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "env, speculative, on_execution_stream",
    [
        ("1", True, True),
        ("1", False, False),
        ("0", True, False),
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
