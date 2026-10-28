# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Production tests of the HTTP /health handler.

Calls the production OpenAIServer._check_health and OpenAIServer.health on a
stand-in object that carries exactly the attributes they read (the shared
server fixture factory does not exist yet), with a fake generator class per
case instead of a MagicMock, which would answer any member with a truthy mock.
"""

import asyncio
import types
from unittest import mock

import pytest

from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only


class _FakeGenerator:
    """Generator double with exactly the public health surface /health reads."""

    def __init__(self, healthy=True, fatal_error=None, shutdown_started=False):
        self._healthy = healthy
        self.fatal_error = fatal_error
        self.shutdown_started = shutdown_started

    def check_health(self):
        return self._healthy


class _NoHealthGenerator:
    """A VisualGen-style generator with no health attributes at all."""


class _DeadBatcher:
    """An embedding batcher whose worker thread has exited."""

    @staticmethod
    def is_alive():
        return False


def _server(generator, is_visual_gen=False, embedding_batcher=None):
    stub = types.SimpleNamespace(
        embedding_batcher=embedding_batcher,
        generator=generator,
        _is_visual_gen=is_visual_gen,
    )
    stub._check_health = types.MethodType(OpenAIServer._check_health, stub)
    return stub


def _health(server):
    with (
        mock.patch("tensorrt_llm.serve.openai_server.signal.raise_signal") as raise_signal,
        mock.patch(
            "tensorrt_llm.serve.openai_server._record_generator_termination"
        ) as record_termination,
    ):
        response = asyncio.run(OpenAIServer.health(server))
    return response, raise_signal, record_termination


def test_health_returns_200_when_the_engine_is_healthy():
    response, raise_signal, _ = _health(_server(_FakeGenerator(healthy=True)))
    assert response.status_code == 200
    raise_signal.assert_not_called()


def test_health_returns_503_without_sigint_when_unhealthy_with_no_fatal_error():
    response, raise_signal, record_termination = _health(
        _server(_FakeGenerator(healthy=False, fatal_error=None))
    )
    assert response.status_code == 503
    assert response.body == (b"LLM is unavailable. Please check the server logs for more details.")
    raise_signal.assert_not_called()
    record_termination.assert_not_called()


def test_health_sends_sigint_exactly_once_on_a_fatal_error():
    import signal as signal_module

    generator = _FakeGenerator(
        healthy=False, fatal_error=RuntimeError("CUDA OOM"), shutdown_started=False
    )
    response, raise_signal, record_termination = _health(_server(generator))
    assert response.status_code == 503
    raise_signal.assert_called_once_with(signal_module.SIGINT)
    record_termination.assert_called_once_with(generator)


def test_health_sends_no_sigint_while_shutdown_is_already_in_progress():
    response, raise_signal, record_termination = _health(
        _server(
            _FakeGenerator(
                healthy=False, fatal_error=RuntimeError("CUDA OOM"), shutdown_started=True
            )
        )
    )
    assert response.status_code == 503
    raise_signal.assert_not_called()
    record_termination.assert_not_called()


def test_health_answers_200_for_a_generator_with_no_health_attributes():
    """A VisualGen generator has no check_health/fatal_error; /health stays 200."""
    response, raise_signal, _ = _health(_server(_NoHealthGenerator(), is_visual_gen=True))
    assert response.status_code == 200
    raise_signal.assert_not_called()


def test_health_returns_503_when_the_embedding_batcher_died():
    response, raise_signal, _ = _health(
        _server(_FakeGenerator(healthy=True), embedding_batcher=_DeadBatcher())
    )
    assert response.status_code == 503
    raise_signal.assert_not_called()
