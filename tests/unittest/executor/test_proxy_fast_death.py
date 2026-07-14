# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""EngineDeadError, the pending-result broadcast, and queue-step fast-fail (no GPU)."""

import asyncio
import queue as _queue

import pytest

from tensorrt_llm.executor import EngineDeadError
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.result import GenerationResult


def test_engine_dead_error_is_importable_and_carries_root_cause():
    cause = RuntimeError("MPI worker exited unexpectedly")
    err = EngineDeadError(cause)
    assert isinstance(err, RuntimeError)
    assert err.root_cause is cause
    assert "Engine has died" in str(err)
    assert "MPI worker exited unexpectedly" in str(err)
    # No root cause is also fine.
    assert "Engine has died" in str(EngineDeadError())


class _FakeResult:
    """Minimal stand-in for GenerationResult exposing only a `.queue`."""

    def __init__(self):
        self.queue = _queue.Queue()


def _bare_proxy():
    proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
    proxy._engine_dead = False
    proxy._results = {}
    # Set so the __del__ -> shutdown() path is a clean no-op at GC time.
    proxy.workers_started = False
    return proxy


def test_mark_engine_dead_broadcasts_to_pending_results():
    proxy = _bare_proxy()
    r1, r2 = _FakeResult(), _FakeResult()
    proxy._results = {1: r1, 2: r2}

    proxy._mark_engine_dead(RuntimeError("worker died"))

    assert proxy._engine_dead is True
    for r in (r1, r2):
        item = r.queue.get_nowait()
        assert isinstance(item, EngineDeadError)
        assert isinstance(item.root_cause, RuntimeError)


def test_mark_engine_dead_is_idempotent():
    proxy = _bare_proxy()
    r = _FakeResult()
    proxy._results = {1: r}

    proxy._mark_engine_dead(RuntimeError("first"))
    assert isinstance(r.queue.get_nowait(), EngineDeadError)

    # A second call must not enqueue another error.
    proxy._mark_engine_dead(RuntimeError("second"))
    assert r.queue.empty()


def test_handle_worker_death_broadcasts_event_driven():
    """Worker death propagates immediately via the future done-callback path."""
    proxy = _bare_proxy()
    proxy._error_queue = _queue.Queue()
    r = _FakeResult()
    proxy._results = {1: r}

    cause = RuntimeError("worker segfault")
    proxy._handle_worker_death(cause)

    # Broadcast happened event-driven: sticky flag set + EngineDeadError pushed.
    assert proxy._engine_dead is True
    item = r.queue.get_nowait()
    assert isinstance(item, EngineDeadError)
    assert item.root_cause is cause
    # Error is also recorded for the monitor loop (which drives pre_shutdown).
    assert proxy._error_queue.get_nowait() is cause


def test_result_step_raises_on_engine_dead():
    res = GenerationResult.__new__(GenerationResult)
    res.queue = _queue.Queue()
    res.queue.put(EngineDeadError(RuntimeError("boom")))
    with pytest.raises(EngineDeadError):
        res._result_step()


def test_aresult_step_raises_on_engine_dead():
    """Async path must also unblock and raise via the _SyncQueue.put notify."""
    from tensorrt_llm.llmapi.utils import AsyncQueue

    async def run():
        res = GenerationResult.__new__(GenerationResult)
        res.aqueue = AsyncQueue()
        res.queue = res.aqueue.sync_q
        res.queue.put(EngineDeadError(RuntimeError("boom")))
        with pytest.raises(EngineDeadError):
            await res._aresult_step()

    asyncio.run(run())


def test_result_stays_failed_after_engine_dead():
    """A dead engine is a sticky terminal failure surfaced on every access."""
    res = GenerationResult.__new__(GenerationResult)
    res.queue = _queue.Queue()
    res._done = False
    res._terminal_error = None
    res.queue.put(EngineDeadError(RuntimeError("boom")))

    # First access consumes the queued error and records it as terminal.
    with pytest.raises(EngineDeadError):
        res._result_step()
    assert res._done is True
    assert res.queue.empty()

    # Repeated access must keep raising (not re-block, not return a
    # successful-looking self).
    with pytest.raises(EngineDeadError):
        res.result()
    with pytest.raises(EngineDeadError):
        res.result()
    # _exception() must keep surfacing the same failure, not None.
    assert isinstance(res._exception(), EngineDeadError)


def test_aresult_stays_failed_after_engine_dead():
    from tensorrt_llm.llmapi.utils import AsyncQueue

    async def run():
        res = GenerationResult.__new__(GenerationResult)
        res.aqueue = AsyncQueue()
        res.queue = res.aqueue.sync_q
        res._done = False
        res._terminal_error = None
        res.queue.put(EngineDeadError(RuntimeError("boom")))

        with pytest.raises(EngineDeadError):
            await res._aresult_step()
        assert res._done is True
        # Repeated await must keep raising, not return successful-looking self.
        with pytest.raises(EngineDeadError):
            await res.aresult()

    asyncio.run(run())


def test_submit_fast_fails_when_engine_already_dead():
    """submit() must reject new work immediately once the engine is dead."""
    proxy = _bare_proxy()
    proxy._fatal_error = None
    proxy._engine_dead = True
    # The sticky guard is the first thing submit() does, before it touches the
    # request, so a placeholder request is never dereferenced.
    with pytest.raises(EngineDeadError):
        proxy.submit(object())


class _FakeRequest:
    """Minimal GenerationRequest stand-in for the submit() re-check test."""

    disaggregated_params = None

    def set_id(self, request_id):
        self.id = request_id


def test_submit_rechecks_engine_death_after_registration(monkeypatch):
    """Close the submit-vs-death race.

    If the engine dies between the top-of-submit guard and registering the
    result in _results (so _mark_engine_dead's one-shot sweep misses it),
    submit() must still fail fast and not leak the dangling result.
    """
    proxy = _bare_proxy()
    proxy._fatal_error = None
    proxy._engine_dead = False
    proxy._results = {}
    proxy._start_dispatch_threads = lambda: None
    proxy._get_next_client_id = lambda: 42
    proxy._get_logprob_params = lambda request: None
    proxy._handle_background_error = lambda *a, **k: None

    fake_result = _FakeResult()

    def fake_generation_result(*args, **kwargs):
        # Simulate the error-monitor thread marking the engine dead mid-submit,
        # after the top guard passed but as the result is being created.
        proxy._engine_dead = True
        return fake_result

    monkeypatch.setattr("tensorrt_llm.executor.proxy.GenerationResult", fake_generation_result)

    with pytest.raises(EngineDeadError):
        proxy.submit(_FakeRequest())

    # The raced result must not be left dangling in _results.
    assert 42 not in proxy._results


# --- Remote (TLLM_SPAWN_PROXY_PROCESS / trtllm-llmapi-launch) mode coverage ---
# RemoteMpiCommSessionClient.submit() returns no futures, so worker death must
# travel server -> client as a RemoteWorkerDeath over the control socket and be
# surfaced by the proxy's _check_remote_worker_death().


def test_remote_worker_death_roundtrip():
    from tensorrt_llm.llmapi.mpi_session import RemoteWorkerDeath

    death = RemoteWorkerDeath.from_exception(ValueError("rank 3 exploded"))
    exc = death.to_exception()
    assert isinstance(exc, RuntimeError)
    assert "ValueError" in str(exc)
    assert "rank 3 exploded" in str(exc)


def test_server_async_callback_forwards_only_failures():
    from concurrent.futures import Future

    from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionServer, RemoteWorkerDeath

    server = object.__new__(RemoteMpiCommSessionServer)
    sent = []
    server.queue = type("Q", (), {"put": lambda self, m: sent.append(m)})()

    ok = Future()
    ok.set_result(42)
    server.mpi_async_error_callback(ok)
    assert sent == []

    cancelled = Future()
    cancelled.cancel()
    server.mpi_async_error_callback(cancelled)
    assert sent == []

    failed = Future()
    failed.set_exception(RuntimeError("worker segfault"))
    server.mpi_async_error_callback(failed)
    assert len(sent) == 1 and isinstance(sent[0], RemoteWorkerDeath)
    assert sent[0].message == "worker segfault"


class _FakeZmqQueue:
    """poll()/get() stub fed with a fixed message sequence."""

    def __init__(self, messages):
        self._messages = list(messages)

    def poll(self, timeout):
        return bool(self._messages)

    def get(self):
        return self._messages.pop(0)


def _bare_remote_client(messages):
    from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionClient

    client = object.__new__(RemoteMpiCommSessionClient)  # bypass singleton
    client._is_shutdown = False
    client._pending_responses = []
    client.queue = _FakeZmqQueue(messages)
    return client


def test_client_check_worker_error_returns_death_and_buffers_others():
    from tensorrt_llm.llmapi.mpi_session import RemoteWorkerDeath

    death = RemoteWorkerDeath.from_exception(RuntimeError("boom"))
    client = _bare_remote_client([[1, 2, 3], death])

    exc = client.check_worker_error()
    assert isinstance(exc, RuntimeError) and "boom" in str(exc)
    # The non-error message was buffered for poll() (submit_sync path).
    assert client.poll() == [1, 2, 3]
    # Nothing left.
    assert client.check_worker_error() is None


def test_proxy_check_remote_worker_death_marks_engine_dead():
    proxy = _bare_proxy()
    proxy._fatal_error = None
    proxy._error_queue = _queue.Queue()
    proxy.doing_shutdown = False
    pre_shutdowns = []
    proxy.pre_shutdown = lambda: pre_shutdowns.append(1)
    r = _FakeResult()
    proxy._results = {1: r}

    death_exc = RuntimeError("Remote MPI worker died: X: boom")
    proxy.mpi_session = type("S", (), {"check_worker_error": lambda self: death_exc})()

    assert proxy._check_remote_worker_death() is True
    # Same fast-death behavior as the local-futures path:
    assert proxy._engine_dead is True
    assert isinstance(r.queue.get_nowait(), EngineDeadError)
    assert proxy._fatal_error is death_exc
    assert proxy._error_queue.get_nowait() is death_exc
    assert pre_shutdowns == [1]

    # Sessions without the hook (MpiPoolSession) are a no-op.
    proxy2 = _bare_proxy()
    proxy2.mpi_session = object()
    assert proxy2._check_remote_worker_death() is False
