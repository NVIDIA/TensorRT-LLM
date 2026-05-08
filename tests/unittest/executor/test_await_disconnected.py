# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for OpenAIServer.await_disconnected keep-alive memory leak fix.

Tests cover:
- await_disconnected exits immediately when promise is already finished
- await_disconnected exits when promise finishes during polling
- await_disconnected aborts when client disconnects before finish
- await_disconnected does not abort when finished at disconnect time
- Multiple concurrent tasks behind keep-alive proxy do not accumulate

See https://github.com/NVIDIA/TensorRT-LLM/issues/13777 for the bug report.

When a reverse proxy (envoy, httpx keep-alive client) sits in front of
trtllm-serve, the TCP connection between proxy and server stays alive
across many requests.  Before the fix, ``await_disconnected`` polled
``is_disconnected()`` forever — even after generation finished — pinning
every ``RequestOutput`` in memory for the lifetime of the keep-alive
connection.  The fix adds an early-exit check on ``promise.finished``.

Tests call ``OpenAIServer.await_disconnected`` as an unbound method with a
``SimpleNamespace()`` self-stub, since the method does not access ``self``.
``asyncio.sleep`` is patched in the server module so the 1-second poll
interval does not slow down the test suite.
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tensorrt_llm.serve.openai_server import OpenAIServer

logger = logging.getLogger(__name__)

_SLEEP_PATCH = "tensorrt_llm.serve.openai_server.asyncio.sleep"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_request(*, disconnected=False):
    """Create a fake Starlette Request with a controllable is_disconnected."""
    req = SimpleNamespace()
    req.is_disconnected = AsyncMock(return_value=disconnected)
    req.client = "test-client"
    return req


def _make_promise(*, finished=False):
    """Create a fake RequestOutput / GenerationResult promise."""
    promise = MagicMock()
    promise.finished = finished
    promise.request_id = 42
    promise.abort = MagicMock()
    return promise


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestAwaitDisconnected:
    """Regression tests for the keep-alive memory leak in await_disconnected.

    The core scenario: a reverse proxy holds persistent connections to
    trtllm-serve.  ``is_disconnected()`` never returns True, so without
    the ``promise.finished`` early-exit the task runs forever and pins
    every ``RequestOutput`` in memory.

    Each test calls ``OpenAIServer.await_disconnected`` as an unbound method
    with a ``SimpleNamespace()`` self-stub and patches ``asyncio.sleep`` in
    the server module to avoid the 1-second polling delay.
    """

    @pytest.mark.asyncio
    async def test_noop_when_raw_request_is_none(self):
        """await_disconnected should be a no-op when raw_request is None."""
        self_stub = SimpleNamespace()
        promise = _make_promise()
        await OpenAIServer.await_disconnected(self_stub, None, promise)
        promise.abort.assert_not_called()

    @pytest.mark.asyncio
    async def test_exits_early_when_promise_finished_with_keepalive(self):
        """Keep-alive connection (never disconnects) + finished promise.

        The task must exit without calling abort.  This is the core
        regression test for the memory leak.
        """
        self_stub = SimpleNamespace()
        request = _make_request(disconnected=False)
        promise = _make_promise(finished=True)

        with patch(_SLEEP_PATCH, new=AsyncMock()):
            await asyncio.wait_for(
                OpenAIServer.await_disconnected(self_stub, request, promise),
                timeout=2.0,
            )
        promise.abort.assert_not_called()

    @pytest.mark.asyncio
    async def test_exits_when_promise_becomes_finished_during_polling(self):
        """promise.finished starts False and flips True after a few polls.

        Simulates normal request completion behind a keep-alive proxy.
        """
        self_stub = SimpleNamespace()
        request = _make_request(disconnected=False)
        promise = _make_promise(finished=False)
        poll_count = 0

        async def _is_disconnected():
            """Simulate keep-alive: never disconnect, finish after 3 polls."""
            nonlocal poll_count
            poll_count += 1
            if poll_count >= 3:
                promise.finished = True
            return False  # connection always alive (keep-alive proxy)

        request.is_disconnected = _is_disconnected

        with patch(_SLEEP_PATCH, new=AsyncMock()):
            await asyncio.wait_for(
                OpenAIServer.await_disconnected(self_stub, request, promise),
                timeout=2.0,
            )
        promise.abort.assert_not_called()
        assert poll_count >= 3

    @pytest.mark.asyncio
    async def test_aborts_when_client_disconnects_before_finish(self):
        """Client disconnects while generation is in progress -> abort."""
        self_stub = SimpleNamespace()
        promise = _make_promise(finished=False)
        poll_count = 0

        async def _is_disconnected():
            """Simulate client disconnect on the 3rd poll."""
            nonlocal poll_count
            poll_count += 1
            return poll_count >= 3

        request = _make_request()
        request.is_disconnected = _is_disconnected

        with patch(_SLEEP_PATCH, new=AsyncMock()):
            await asyncio.wait_for(
                OpenAIServer.await_disconnected(self_stub, request, promise),
                timeout=2.0,
            )
        promise.abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_abort_when_already_finished_at_disconnect(self):
        """Client disconnects but promise was already finished -> no abort."""
        self_stub = SimpleNamespace()
        promise = _make_promise(finished=True)
        request = _make_request(disconnected=True)

        with patch(_SLEEP_PATCH, new=AsyncMock()):
            await asyncio.wait_for(
                OpenAIServer.await_disconnected(self_stub, request, promise),
                timeout=2.0,
            )
        promise.abort.assert_not_called()

    @pytest.mark.asyncio
    async def test_keepalive_tasks_do_not_accumulate(self):
        """Spawn 100 tasks behind a keep-alive connection.

        All must complete once promise.finished is True — otherwise they
        accumulate and leak memory (the original bug).
        """
        tasks = []
        with patch(_SLEEP_PATCH, new=AsyncMock()):
            for _ in range(100):
                self_stub = SimpleNamespace()
                request = _make_request(disconnected=False)
                promise = _make_promise(finished=True)
                tasks.append(
                    asyncio.create_task(
                        OpenAIServer.await_disconnected(self_stub, request,
                                                        promise)))

            _, pending = await asyncio.wait(tasks, timeout=2.0)
        assert len(pending) == 0, (
            f"{len(pending)} await_disconnected tasks still running despite "
            f"promise.finished=True — would cause memory leak with "
            f"keep-alive proxies")
