# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Production tests of GrpcRequestManager.health_check.

The (healthy, message) pair feeds HealthCheckResponse.status, so every message
string is wire-visible and pinned byte for byte, including the first-line rule
and the 200-character cut for fatal errors.
"""

import asyncio
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

pytest.importorskip("smg_grpc_proto")

from tensorrt_llm.grpc.smg.request_manager import GrpcRequestManager  # noqa: E402


class _FakeLLM:
    """LLM double with exactly the public health surface health_check reads."""

    def __init__(
        self,
        healthy=True,
        fatal_error=None,
        encode_only=False,
        check_raises=None,
        shutdown_started=False,
        engine_dead=False,
    ):
        self.args = SimpleNamespace(encode_only=encode_only)
        self.fatal_error = fatal_error
        self.shutdown_started = shutdown_started
        self.engine_dead = engine_dead
        self._healthy = healthy
        self._check_raises = check_raises

    def check_health(self):
        if self._check_raises is not None:
            raise self._check_raises
        return self._healthy


def _health_check(llm):
    return asyncio.run(GrpcRequestManager(llm).health_check())


def test_healthy_llm_answers_ok():
    assert _health_check(_FakeLLM(healthy=True)) == (True, "OK")


def test_missing_llm_answers_not_initialized():
    assert _health_check(None) == (False, "LLM not initialized")


def test_encode_only_llm_answers_executor_not_available():
    assert _health_check(_FakeLLM(encode_only=True)) == (False, "Executor is not available")


def test_unhealthy_llm_without_fatal_error_answers_executor_unhealthy():
    assert _health_check(_FakeLLM(healthy=False)) == (False, "Executor is unhealthy")


def test_llm_after_shutdown_answers_executor_not_available():
    # The public health surface LLM.shutdown() leaves behind: no generation
    # executor, so unhealthy, shutdown started, no fatal error, engine not dead.
    llm = _FakeLLM(healthy=False, shutdown_started=True)
    assert _health_check(llm) == (False, "Executor is not available")


def test_dead_engine_during_shutdown_answers_executor_unhealthy():
    llm = _FakeLLM(healthy=False, shutdown_started=True, engine_dead=True)
    assert _health_check(llm) == (False, "Executor is unhealthy")


def test_fatal_error_during_shutdown_answers_the_fatal_error():
    llm = _FakeLLM(healthy=False, fatal_error=RuntimeError("CUDA OOM"), shutdown_started=True)
    assert _health_check(llm) == (False, "RuntimeError: CUDA OOM")


def test_fatal_error_message_names_the_exception_type():
    result = _health_check(_FakeLLM(healthy=False, fatal_error=RuntimeError("CUDA OOM")))
    assert result == (False, "RuntimeError: CUDA OOM")


def test_fatal_error_message_keeps_the_first_line_cut_to_200_characters():
    long_first_line = "x" * 300
    result = _health_check(
        _FakeLLM(healthy=False, fatal_error=RuntimeError(f"{long_first_line}\nsecond line"))
    )
    assert result == (False, f"RuntimeError: {'x' * 200}")


def test_raising_health_check_answers_error():
    assert _health_check(_FakeLLM(check_raises=ValueError("boom"))) == (False, "Error: boom")
