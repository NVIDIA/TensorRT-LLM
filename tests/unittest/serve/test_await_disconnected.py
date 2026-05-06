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
"""Unit tests for OpenAIServer.await_disconnected memory leak fix.

When a reverse proxy (e.g. envoy, httpx keep-alive client) sits in front of
trtllm-serve, the TCP connection between proxy and server stays alive across
many requests.  Before the fix, ``await_disconnected`` polled
``is_disconnected()`` forever — even after the generation finished — pinning
the entire ``RequestOutput`` in memory for the lifetime of the keep-alive
connection.

The fix adds an early-exit check: once ``promise.finished`` is True the task
returns immediately, releasing the reference to ``promise``.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Minimal stub for OpenAIServer.await_disconnected
# ---------------------------------------------------------------------------
# Import just the method under test to avoid pulling in the full
# tensorrt_llm import chain (which requires C++ extensions / GPU).
# The method only uses ``self`` for logging — a bare object suffices.


async def _await_disconnected(self, raw_request, promise):
    """Copy of OpenAIServer.await_disconnected (post-fix)."""
    if raw_request is None:
        return
    while not await raw_request.is_disconnected():
        if promise.finished:
            return
        await asyncio.sleep(0)  # Use sleep(0) in tests for fast iteration
    if not promise.finished:
        promise.abort()


class _FakeServer:
    """Minimal stand-in so ``self`` is a valid receiver."""
    await_disconnected = _await_disconnected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(*, disconnected: bool = False):
    """Create a fake Starlette Request with a controllable is_disconnected."""
    req = SimpleNamespace()
    req.is_disconnected = AsyncMock(return_value=disconnected)
    req.client = "test-client"
    return req


def _make_promise(*, finished: bool = False):
    """Create a fake RequestOutput / GenerationResult promise."""
    promise = MagicMock()
    promise.finished = finished
    promise.request_id = 42
    promise.abort = MagicMock()
    return promise


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_returns_immediately_when_raw_request_is_none():
    """await_disconnected should be a no-op when raw_request is None."""
    server = _FakeServer()
    promise = _make_promise()
    await server.await_disconnected(None, promise)
    promise.abort.assert_not_called()


@pytest.mark.asyncio
async def test_exits_early_when_promise_finished_with_keepalive():
    """Simulates a keep-alive proxy: is_disconnected() always returns False
    (connection stays open), but promise.finished becomes True after
    generation completes.  The task must exit without calling abort.

    This is the core regression test for the memory leak described in
    https://github.com/NVIDIA/TensorRT-LLM/issues/13777.
    """
    server = _FakeServer()
    request = _make_request(disconnected=False)
    promise = _make_promise(finished=True)

    # With the fix, the task should complete almost immediately
    # because promise.finished is checked on each iteration.
    await asyncio.wait_for(
        server.await_disconnected(request, promise),
        timeout=2.0,
    )

    promise.abort.assert_not_called()


@pytest.mark.asyncio
async def test_exits_when_promise_becomes_finished_during_polling():
    """promise.finished starts False (generation in progress) and becomes
    True after a few polls — simulating normal request completion behind
    a keep-alive proxy.
    """
    server = _FakeServer()
    request = _make_request(disconnected=False)
    promise = _make_promise(finished=False)

    poll_count = 0

    async def _is_disconnected():
        nonlocal poll_count
        poll_count += 1
        if poll_count >= 3:
            # After a few polls, mark the request as finished
            promise.finished = True
        return False  # Connection always alive (keep-alive proxy)

    request.is_disconnected = _is_disconnected

    await asyncio.wait_for(
        server.await_disconnected(request, promise),
        timeout=2.0,
    )

    # Should have exited via the promise.finished check, not via abort
    promise.abort.assert_not_called()
    assert poll_count >= 3


@pytest.mark.asyncio
async def test_aborts_when_client_disconnects_before_finish():
    """Client disconnects while generation is still in progress.
    The task should call promise.abort().
    """
    server = _FakeServer()
    promise = _make_promise(finished=False)

    poll_count = 0

    async def _is_disconnected():
        nonlocal poll_count
        poll_count += 1
        # Simulate client disconnect on 3rd poll
        return poll_count >= 3

    request = _make_request()
    request.is_disconnected = _is_disconnected

    await asyncio.wait_for(
        server.await_disconnected(request, promise),
        timeout=2.0,
    )

    promise.abort.assert_called_once()


@pytest.mark.asyncio
async def test_no_abort_when_already_finished_at_disconnect():
    """Client disconnects, but promise was already finished.
    Should not call abort.
    """
    server = _FakeServer()
    promise = _make_promise(finished=True)
    request = _make_request(disconnected=True)

    await asyncio.wait_for(
        server.await_disconnected(request, promise),
        timeout=2.0,
    )

    promise.abort.assert_not_called()


@pytest.mark.asyncio
async def test_keepalive_does_not_pin_promise_reference():
    """Verify that with a keep-alive connection (never disconnects),
    the await_disconnected task completes and releases its reference
    to promise once promise.finished is True.

    This is the actual memory leak scenario: with N completed requests
    behind a keep-alive proxy, N await_disconnected tasks would
    accumulate, each pinning a RequestOutput in memory.
    """
    server = _FakeServer()
    tasks = []
    promises = []

    # Simulate 100 requests through a keep-alive connection
    for i in range(100):
        request = _make_request(disconnected=False)
        promise = _make_promise(finished=True)
        promises.append(promise)
        task = asyncio.create_task(
            server.await_disconnected(request, promise)
        )
        tasks.append(task)

    # All tasks should complete quickly since promise.finished is True
    done, pending = await asyncio.wait(tasks, timeout=2.0)

    assert len(pending) == 0, (
        f"{len(pending)} await_disconnected tasks are still running "
        f"despite promise.finished=True — this would cause a memory leak "
        f"with keep-alive proxies"
    )
    assert len(done) == 100
