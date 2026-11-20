# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Production tests of the public runtime health API the serving frontends use.

Covers BaseLLM.check_health(), BaseLLM.fatal_error, BaseLLM.shutdown_started,
BaseLLM.engine_dead and the executor-side GenerationExecutor.fatal_error and
engine_dead, against the real classes (no mirrors; the LLM objects are built
with object.__new__ plus exactly the attributes BaseLLM.__init__ sets).
"""

import pytest

from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.llmapi.llm import LLM

pytestmark = pytest.mark.cpu_only


class _NoOpExecutor(GenerationExecutor):
    """Minimal concrete GenerationExecutor; health logic comes from the base."""

    def submit(self, request):
        raise NotImplementedError("health tests never submit requests")

    def abort_request(self, request_id):
        pass

    def shutdown(self):
        self.doing_shutdown = True


def _llm(executor, encode_only=False, encoder_executor=None):
    """An LLM with exactly the attributes BaseLLM.__init__ sets for health."""
    llm = object.__new__(LLM)
    llm._executor = executor
    llm._encode_only = encode_only
    llm._encoder_executor = encoder_executor
    return llm


def _proxy(engine_dead=False, fatal_error=None):
    """A proxy with exactly the two fields its engine_dead property reads."""
    proxy = object.__new__(GenerationExecutorProxy)
    proxy.workers_started = False
    proxy._engine_dead = engine_dead
    proxy._fatal_error = fatal_error
    return proxy


# ---------------------------------------------------------------------------
# BaseLLM.check_health()
# ---------------------------------------------------------------------------
def test_check_health_encode_only_with_encoder_executor():
    assert _llm(None, encode_only=True, encoder_executor=_NoOpExecutor()).check_health() is True


def test_check_health_encode_only_without_encoder_executor():
    assert _llm(None, encode_only=True, encoder_executor=None).check_health() is False


def test_check_health_healthy_executor():
    assert _llm(_NoOpExecutor()).check_health() is True


def test_check_health_fatal_error():
    executor = _NoOpExecutor()
    executor._set_fatal_error(RuntimeError("CUDA OOM"))
    assert _llm(executor).check_health() is False


def test_check_health_doing_shutdown():
    executor = _NoOpExecutor()
    executor.doing_shutdown = True
    assert _llm(executor).check_health() is False


def test_check_health_without_executor():
    assert _llm(None).check_health() is False


def test_check_health_drains_queued_fatal_error():
    """A fatal error still in the error queue is promoted by check_health()."""
    executor = _NoOpExecutor()
    queued = RuntimeError("engine died in a background thread")
    executor._error_queue.put(queued)

    llm = _llm(executor)
    assert llm.check_health() is False
    assert llm.fatal_error is queued
    assert llm.shutdown_started is True


def test_check_health_private_alias_is_the_same_function():
    """Kept for one release for out-of-tree callers that look the name up."""
    assert LLM._check_health is LLM.check_health


# ---------------------------------------------------------------------------
# BaseLLM.fatal_error
# ---------------------------------------------------------------------------
def test_fatal_error_none_without_executor():
    assert _llm(None).fatal_error is None


def test_fatal_error_none_on_healthy_executor():
    assert _llm(_NoOpExecutor()).fatal_error is None


def test_fatal_error_returns_recorded_exception_first_one_wins():
    executor = _NoOpExecutor()
    first = RuntimeError("first fatal error")
    executor._set_fatal_error(first)
    executor._set_fatal_error(RuntimeError("second fatal error"))
    assert _llm(executor).fatal_error is first


# ---------------------------------------------------------------------------
# BaseLLM.shutdown_started
# ---------------------------------------------------------------------------
def test_shutdown_started_follows_doing_shutdown():
    executor = _NoOpExecutor()
    llm = _llm(executor)
    assert llm.shutdown_started is False
    executor.doing_shutdown = True
    assert llm.shutdown_started is True


def test_shutdown_started_true_without_executor():
    assert _llm(None).shutdown_started is True


def test_shutdown_started_false_for_encode_only():
    assert _llm(None, encode_only=True, encoder_executor=_NoOpExecutor()).shutdown_started is False


# ---------------------------------------------------------------------------
# BaseLLM.engine_dead
# ---------------------------------------------------------------------------
def test_engine_dead_false_without_executor():
    assert _llm(None).engine_dead is False


def test_engine_dead_follows_the_executor():
    executor = _NoOpExecutor()
    llm = _llm(executor)
    assert llm.engine_dead is False
    executor._set_fatal_error(RuntimeError("worker died"))
    assert llm.engine_dead is True


# ---------------------------------------------------------------------------
# GenerationExecutor.fatal_error and engine_dead
# ---------------------------------------------------------------------------
def test_executor_fatal_error_property():
    executor = _NoOpExecutor()
    assert executor.fatal_error is None
    error = RuntimeError("fatal")
    executor._set_fatal_error(error)
    assert executor.fatal_error is error


def test_executor_engine_dead_property():
    executor = _NoOpExecutor()
    assert executor.engine_dead is False
    executor._set_fatal_error(RuntimeError("fatal"))
    assert executor.engine_dead is True


@pytest.mark.parametrize(
    "engine_dead,fatal_error,expected",
    [
        (False, None, False),
        (True, None, True),
        (False, RuntimeError("fatal"), True),
        (True, RuntimeError("fatal"), True),
    ],
)
def test_proxy_engine_dead_reads_both_fields(engine_dead, fatal_error, expected):
    """_mark_engine_dead can set the sticky flag before the error is recorded."""
    assert _proxy(engine_dead=engine_dead, fatal_error=fatal_error).engine_dead is expected
