# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for trtllm-serve /start_profile and /stop_profile endpoints.

These tests bind the unbound handler methods from ``OpenAIServer`` to a
lightweight mock instance so they can run without GPUs or model weights.
They verify that (a) the endpoints parse their request bodies correctly,
(b) default values are applied, and (c) the underlying ``generator``
object receives the expected ``start_profile`` / ``stop_profile`` calls
with the right arguments.
"""

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from tensorrt_llm.serve.openai_protocol import StartProfileRequest


def _extract_json(response) -> dict:
    """Decode a FastAPI ``JSONResponse`` body into a dict."""
    return json.loads(response.body.decode("utf-8"))


@pytest.fixture
def server_with_mock_generator():
    """Build a minimal mock ``OpenAIServer`` instance with only the bits
    needed for the profile handlers. Avoids the heavy real __init__.
    """
    # Importing here so a missing optional dep in the server module only
    # fails the test, not collection of the whole test file.
    from tensorrt_llm.serve.openai_server import OpenAIServer

    server = OpenAIServer.__new__(OpenAIServer)
    server.generator = MagicMock()
    return server


def test_start_profile_defaults(server_with_mock_generator):
    server = server_with_mock_generator
    response = asyncio.run(server.start_profile(StartProfileRequest()))

    assert response.status_code == 200
    assert _extract_json(response) == {"message": "Profiling started"}
    server.generator.start_profile.assert_called_once_with(
        output_dir=None,
        num_steps=None,
        start_step=0,
        activities=["CPU", "GPU"],
    )


def test_start_profile_none_body_uses_defaults(server_with_mock_generator):
    server = server_with_mock_generator
    # FastAPI passes ``None`` if the request body is empty; the handler
    # must accept that and fall back to the default StartProfileRequest.
    response = asyncio.run(server.start_profile(None))

    assert response.status_code == 200
    server.generator.start_profile.assert_called_once_with(
        output_dir=None,
        num_steps=None,
        start_step=0,
        activities=["CPU", "GPU"],
    )


def test_start_profile_custom_args(server_with_mock_generator):
    server = server_with_mock_generator
    request = StartProfileRequest(
        output_dir="/tmp/my_traces",
        num_steps=10,
        start_step=3,
        activities=["CUDA_PROFILER"],
    )
    response = asyncio.run(server.start_profile(request))

    assert response.status_code == 200
    server.generator.start_profile.assert_called_once_with(
        output_dir="/tmp/my_traces",
        num_steps=10,
        start_step=3,
        activities=["CUDA_PROFILER"],
    )


def test_stop_profile(server_with_mock_generator):
    server = server_with_mock_generator
    response = asyncio.run(server.stop_profile())

    assert response.status_code == 200
    assert _extract_json(response) == {"message": "Profiling stopped"}
    server.generator.stop_profile.assert_called_once_with()


def test_start_profile_backend_error_returns_500(server_with_mock_generator):
    server = server_with_mock_generator
    # A RuntimeError whose message contains neither "already in progress"
    # nor "pending" is treated as a generic backend failure (500); the
    # 409 path is covered by test_http_handler_rejects_double_start_with_409
    # in test_profile_endpoints_bugs.py.
    server.generator.start_profile.side_effect = RuntimeError("boom")
    response = asyncio.run(server.start_profile(StartProfileRequest()))

    assert response.status_code == 500
    body = _extract_json(response)
    assert "boom" in body["message"]


def test_stop_profile_backend_error_returns_500(server_with_mock_generator):
    server = server_with_mock_generator
    server.generator.stop_profile.side_effect = RuntimeError("nope")
    response = asyncio.run(server.stop_profile())

    assert response.status_code == 500
    body = _extract_json(response)
    assert "nope" in body["error"]


def _async_loop_block_test(handler_call, backend_block_s: float = 0.3) -> float:
    """Helper: run ``handler_call`` (an async lambda invoking
    ``server.start_profile()`` or ``server.stop_profile()``) while
    concurrently ticking every 10 ms in the same event loop.

    Returns the maximum gap (in seconds) between consecutive ticker
    timestamps. A gap >> 10 ms means the handler blocked the loop.
    """
    import time

    async def _body():
        ticks = []
        stop_event = asyncio.Event()

        async def _ticker():
            while not stop_event.is_set():
                ticks.append(time.monotonic())
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=0.01)
                except asyncio.TimeoutError:
                    pass

        ticker_task = asyncio.create_task(_ticker())
        # Warm up the ticker before invoking the handler so we have a
        # cadence baseline.
        await asyncio.sleep(0.05)
        await handler_call()
        await asyncio.sleep(0.05)
        stop_event.set()
        await ticker_task

        max_gap = max(b - a for a, b in zip(ticks, ticks[1:])) if len(ticks) > 1 else 0.0
        return max_gap

    return asyncio.run(_body())


def test_stop_profile_does_not_block_event_loop(server_with_mock_generator):
    """Regression for PR review §2.5: the HTTP handler must not freeze
    the asyncio event loop while the (potentially slow) backend
    ``stop_profile`` call runs. We give the mocked generator a 300 ms
    blocking ``time.sleep`` and assert the event loop kept ticking.
    """
    import time

    server = server_with_mock_generator
    backend_block_s = 0.3

    def slow_backend_stop():
        time.sleep(backend_block_s)

    server.generator.stop_profile.side_effect = slow_backend_stop

    async def _call():
        await server.stop_profile()

    max_gap = _async_loop_block_test(_call, backend_block_s=backend_block_s)
    # Allow generous margin (the ticker only fires every 10 ms; threading
    # round-trip adds ~5 ms). The bug produced gaps of >300 ms.
    assert max_gap < 0.1, (
        f"event loop blocked for {max_gap * 1000:.1f}ms during stop_profile "
        f"(backend blocked for {backend_block_s * 1000:.1f}ms). The handler "
        f"must run the blocking call via asyncio.to_thread."
    )


def test_start_profile_does_not_block_event_loop(server_with_mock_generator):
    """Regression for PR review §2.5 — same as the stop_profile case
    but for /start_profile. On the IPC-proxy path the start can block
    for up to ~60s waiting for the worker ack."""
    import time

    server = server_with_mock_generator
    backend_block_s = 0.3

    def slow_backend_start(**_):
        time.sleep(backend_block_s)

    server.generator.start_profile.side_effect = slow_backend_start

    async def _call():
        await server.start_profile(StartProfileRequest())

    max_gap = _async_loop_block_test(_call, backend_block_s=backend_block_s)
    assert max_gap < 0.1, f"event loop blocked for {max_gap * 1000:.1f}ms during start_profile"
