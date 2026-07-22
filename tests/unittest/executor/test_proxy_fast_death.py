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
import time as _time
from concurrent.futures import Future as _Future
from unittest.mock import Mock
from unittest.mock import Mock as _Mock

import pytest

from tensorrt_llm.executor import EngineDeadError
from tensorrt_llm.executor import proxy as proxy_module
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


def test_register_worker_processes_with_session_reuse_factory(monkeypatch):
    """Session reuse replaces proxy.MpiPoolSession with a factory function."""
    pool_session = object()
    monkeypatch.setattr(proxy_module, "MpiPoolSession", lambda n_workers: pool_session)
    proxy = _bare_proxy()
    proxy.mpi_session = proxy_module.MpiPoolSession(1)
    proxy._worker_process_monitor = Mock()
    identities = [object()]

    proxy._register_worker_processes((proxy.READY_SIGNAL, None, identities))

    proxy._worker_process_monitor.register.assert_called_once_with(identities)


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


# --- Non-blocking teardown on a dead engine ---
# An abruptly-killed worker world never completes its mpi4py futures and
# never sends the result-queue shutdown sentinel, so an unbounded shutdown()
# blocks forever on f.result(), the dispatcher join, and the pool join.
# These tests pin the bounded-teardown behavior.


def _teardown_proxy(engine_dead):
    proxy = _bare_proxy()
    proxy.workers_started = True
    proxy.doing_shutdown = True  # skip pre_shutdown(); teardown path only
    proxy._engine_dead = engine_dead
    proxy._fatal_error = RuntimeError("worker died") if engine_dead else None
    proxy.dispatch_result_thread = None
    proxy.rpc_client = None
    proxy.request_queue = _Mock()
    proxy.worker_init_status_queue = _Mock()
    proxy.result_queue = _Mock()
    proxy._resource_governor_queue = None
    proxy._owns_mpi_session = True
    proxy.mpi_session = _Mock()
    proxy._handle_background_error = lambda *a, **k: None
    return proxy


def test_shutdown_does_not_block_on_dead_engine():
    """With the engine dead, never-completing futures must not hang teardown."""
    proxy = _teardown_proxy(engine_dead=True)
    pending = _Future()  # never completes: abrupt worker death
    done = _Future()
    done.set_exception(RuntimeError("captured by mpi_done_callback already"))
    proxy.mpi_futures = [pending, done]

    dispatcher = _Mock()
    dispatcher.is_alive.return_value = True
    proxy.dispatch_result_thread = dispatcher

    start = _time.monotonic()
    proxy.shutdown()
    elapsed = _time.monotonic() - start

    # One collective 5 s grace, not an unbounded f.result() per future.
    assert elapsed < 30
    assert not pending.done()
    # The dispatcher join is bounded (daemon thread is leaked, not awaited).
    dispatcher.stop.assert_called_once()
    dispatcher.join.assert_called_once_with(timeout=5.0)
    # The dead pool is not joined; the session is abandoned instead.
    proxy.mpi_session.abandon.assert_called_once_with()
    proxy.mpi_session.shutdown.assert_not_called()
    assert proxy.workers_started is False


def test_shutdown_keeps_blocking_semantics_when_engine_alive():
    """Orderly shutdown is unchanged: futures reaped, session joined."""
    proxy = _teardown_proxy(engine_dead=False)
    done = _Future()
    done.set_result(None)
    proxy.mpi_futures = [done]

    dispatcher = _Mock()
    dispatcher.is_alive.return_value = True
    proxy.dispatch_result_thread = dispatcher

    proxy.shutdown()

    dispatcher.join.assert_called_once_with(timeout=None)
    proxy.mpi_session.shutdown.assert_called_once_with()
    proxy.mpi_session.abandon.assert_not_called()
    assert proxy.workers_started is False


def test_shutdown_does_not_shut_down_external_session():
    """An externally owned session must stay alive even on a dead engine."""
    proxy = _teardown_proxy(engine_dead=True)
    proxy._owns_mpi_session = False
    proxy.mpi_futures = []

    proxy.shutdown()

    proxy.mpi_session.shutdown.assert_not_called()
    proxy.mpi_session.abandon.assert_not_called()


def test_abandon_mpi_pool_threads_unblocks_interpreter_exit():
    """Both exit-join mechanisms release a wedged pool manager thread.

    The thread is deregistered from mpi4py's THREADS_QUEUES and CPython's
    _shutdown_locks, so process exit can proceed without joining it.
    """
    import sys as _sys
    import threading as _threading
    import types as _types

    from tensorrt_llm.llmapi.mpi_session import _abandon_mpi_pool_threads

    release = _threading.Event()
    wedged = _threading.Thread(target=release.wait, name="fake_manager")
    wedged.daemon = False
    wedged.start()
    try:
        # Fake mpi4py registry module, as mpi4py would have registered it.
        fake_mod = _types.ModuleType("mpi4py.futures._lib")
        fake_mod.THREADS_QUEUES = {wedged: object()}
        prev = _sys.modules.get("mpi4py.futures._lib")
        _sys.modules["mpi4py.futures._lib"] = fake_mod
        try:
            fake_pool = _Mock()
            fake_pool._pool.thread = wedged

            _abandon_mpi_pool_threads(fake_pool)

            assert wedged not in fake_mod.THREADS_QUEUES
            shutdown_locks = getattr(_threading, "_shutdown_locks", None)
            if shutdown_locks is not None:  # CPython 3.9-3.12
                assert wedged._tstate_lock not in shutdown_locks
        finally:
            if prev is None:
                del _sys.modules["mpi4py.futures._lib"]
            else:
                _sys.modules["mpi4py.futures._lib"] = prev
    finally:
        release.set()
        wedged.join(timeout=5)


def test_abandon_mpi_pool_threads_tolerates_missing_pool():
    from tensorrt_llm.llmapi.mpi_session import _abandon_mpi_pool_threads

    _abandon_mpi_pool_threads(None)
    _abandon_mpi_pool_threads(object())  # no _pool attribute


def test_mark_engine_dead_releases_exit_joins_immediately():
    """Exit-join release must happen at detection time, not at teardown.

    CPython's exit sequence joins non-daemon threads before atexit/GC can
    run shutdown(), so a wedged pool manager thread must be deregistered
    the moment the engine is marked dead.
    """
    proxy = _bare_proxy()
    proxy.mpi_session = _Mock()

    proxy._mark_engine_dead(RuntimeError("worker died"))
    proxy.mpi_session.release_exit_joins.assert_called_once_with()

    # Sticky: a second death report must not re-release.
    proxy._mark_engine_dead(RuntimeError("again"))
    proxy.mpi_session.release_exit_joins.assert_called_once_with()


def test_mark_engine_dead_releases_external_sessions_too():
    """Ownership must not gate the exit-join release.

    The LLM API creates the session and passes it in, so the proxy does
    not own it; the release is non-destructive bookkeeping and must still
    happen.
    """
    proxy = _bare_proxy()
    proxy._owns_mpi_session = False
    proxy.mpi_session = _Mock()

    proxy._mark_engine_dead(RuntimeError("worker died"))
    proxy.mpi_session.release_exit_joins.assert_called_once_with()
    # But destructive teardown is still reserved for the owner.
    proxy.mpi_session.abandon.assert_not_called()
    proxy.mpi_session.shutdown.assert_not_called()


def test_pool_session_shutdown_never_blocks_after_release():
    """A released session never blocks, even on an explicit shutdown().

    After release_exit_joins(), a blocking shutdown() from the session
    owner must not join the dead pool.
    """
    from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

    session = MpiPoolSession.__new__(MpiPoolSession)
    # __new__ bypasses __init__ (which would spawn real MPI workers), so the
    # attributes shutdown() reads must be provided here.
    session.n_workers = 2
    session._wait_shutdown = False
    pool = _Mock()
    pool._pool.thread = None  # no real manager thread to deregister
    session.mpi_pool = pool

    session.release_exit_joins()
    session.shutdown()  # owner asks for the default blocking shutdown

    pool.shutdown.assert_called_once_with(wait=False)
