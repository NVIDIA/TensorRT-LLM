# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproducer + regression test for §2.1 of the PR review.

``GenerationExecutorProxy.stop_profile`` (and ``start_profile``) used to
``request_queue.put(...)`` and return immediately, even though the actual
``torch.profiler.stop()`` + ``export_chrome_trace()`` happens
asynchronously inside the worker subprocess. Callers that hit
``POST /stop_profile`` and then immediately tried to read the chrome
trace from disk would see an empty / partial file.

The contract documented in
``docs/source/commands/trtllm-serve/trtllm-serve.rst`` says::

    The call blocks until ``export_chrome_trace()`` has run on the
    backend thread, so by the time the handler returns the file is
    on disk.

These tests pin that contract for the IPC-proxy path.
"""

import concurrent.futures
import pickle
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import zmq

from tensorrt_llm.executor.ipc import IpcQueue
from tensorrt_llm.executor.proxy import GenerationExecutorProxy, _RequestSender
from tensorrt_llm.executor.request import (
    CancellingRequest,
    GenerationRequest,
    StartProfileRequest,
    StopProfileRequest,
)
from tensorrt_llm.executor.utils import EngineDeadError, RequestError
from tensorrt_llm.executor.worker import _dispatch_requests
from tensorrt_llm.sampling_params import SamplingParams

# These tests deliberately keep a real single-thread
# ``_profile_control_executor`` alive for the whole test body -- pinning
# every profile ZMQ op to one owner thread is precisely what they assert.
# ``pytest-threadleak`` samples threads around the *call* phase, so a
# function-scoped fixture finalizer necessarily runs too late to satisfy
# it. The ``bare_proxy`` teardown below still drains the executor, so the
# owner threads are reclaimed and do not accumulate across the module.
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


@pytest.mark.threadleak(enabled=True)
@pytest.mark.timeout(15, method="thread")
@pytest.mark.parametrize("outcome", ["success", "rejected", "fatal"])
def test_profile_flush_keeps_shared_command_ingress_live(tmp_path, outcome):
    address = (f"ipc://{tmp_path}/commands", b"k" * 32)
    entered = threading.Event()
    release = threading.Event()
    cancelled = threading.Event()
    ready = threading.Event()
    errors = queue.Queue()
    received = []

    class Worker:
        def stop_profile(self):
            entered.set()
            assert release.wait(5)
            if outcome == "rejected":
                raise RequestError("profile rejected")
            if outcome == "fatal":
                raise RuntimeError("profile failed")

        def submit(self, request):
            received.append(request.id)

        def abort_request(self, request_id):
            assert request_id == 7
            cancelled.set()

    def dispatch():
        ingress = IpcQueue(address, is_server=True, socket_type=zmq.PULL, receive_hwm=16)
        try:
            ingress.setup_lazily()
            ready.set()
            _dispatch_requests(Worker(), ingress)
        except BaseException as error:
            errors.put(error)
        finally:
            ingress.close()

    thread = threading.Thread(target=dispatch)
    thread.start()
    ack = IpcQueue(is_server=True)
    senders = [
        _RequestSender(
            IpcQueue(address, is_server=False, socket_type=zmq.PUSH),
            errors,
            batching=False,
            connect_timeout=0.5,
        )
        for _ in range(2)
    ]
    try:
        assert ready.wait(3)
        ack.setup_lazily()
        senders[0].send(pickle.dumps(StopProfileRequest(ack_addr=ack.address)), control=True)
        assert entered.wait(3)
        # Exceed the production transport's finite buffers while export is
        # held. The other frontend must keep making progress, not time out.
        for request_id in range(64):
            request = GenerationRequest([1] * (512 * 1024), SamplingParams())
            request.set_id(request_id)
            senders[1].send(pickle.dumps(request))
        senders[1].send(pickle.dumps(CancellingRequest(7)), control=True)
        assert cancelled.wait(2)
        assert received == list(range(64))
        assert errors.empty()
        assert not ack.poll(0.01), "profile acknowledged before trace export completed"
        release.set()
        if outcome == "fatal":
            error = errors.get(timeout=3)
            assert isinstance(error, RuntimeError)
            assert str(error) == "profile failed"
        else:
            assert ack.get(timeout=3) == (
                "stop",
                "profile rejected" if outcome == "rejected" else None,
            )
    finally:
        release.set()
        senders[1].close(timeout=3)
        senders[0].close(timeout=3, send_sentinel=outcome != "fatal")
        thread.join(timeout=3)
        ack.close()
    assert not thread.is_alive()
    if outcome == "fatal":
        # Closing the failed worker's ingress can independently notify both
        # frontend owners before their orderly cleanup begins.
        while not errors.empty():
            assert isinstance(errors.get_nowait(), ConnectionError)
    else:
        assert errors.empty()


@pytest.fixture
def bare_proxy():
    """Build a minimal ``GenerationExecutorProxy`` for testing.

    Provides only the queues the profile handlers touch. Avoids the
    heavy real ``__init__`` (no MPI session, no zmq sockets, no worker
    subprocess).

    Also wires a real single-thread ``_profile_control_executor`` —
    this matches the production setup, where the shared command sender
    owns writes and the profile-control executor owns profile replies.
    """
    proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
    # Set by ``GenerationExecutor.__init__`` in production; the profile
    # handlers reject control once ``pre_shutdown()`` flips it.
    proxy.doing_shutdown = False
    proxy.workers_started = False
    proxy._engine_dead = False
    proxy._fatal_error = None
    proxy.request_queue = MagicMock()
    # The fix introduces a ``profile_ack_queue`` attribute. Pre-fix this
    # attribute does not exist; we stub it as MagicMock so old code paths
    # accessing it via getattr() see a sentinel.
    proxy.profile_ack_queue = MagicMock()
    proxy.profile_ack_queue.address = ("unused-profile-ack", b"k" * 32)
    delivered = queue.Queue()

    class Commands:
        socket = SimpleNamespace(
            setsockopt=lambda *args: None,
            get_monitor_socket=lambda **kwargs: SimpleNamespace(
                poll=lambda timeout: False,
                getsockopt=lambda option: 0,
                close=lambda **kwargs: None,
            ),
            disable_monitor=lambda: None,
        )

        def setup_lazily(self):
            pass

        def put(self, message, **kwargs):
            if message is not None:
                for payload in message:
                    proxy.request_queue.put(pickle.loads(payload))
                    delivered.put(None)

        def close(self):
            pass

    proxy._request_sender = _RequestSender(Commands(), queue.Queue(), batching=False)
    original_wait = proxy._wait_profile_ack

    def wait_for_delivered_ack(kind, timeout):
        # A real worker cannot acknowledge a command before receiving it.
        # Preserve that causality with asynchronous command submission too.
        delivered.get(timeout=timeout)
        return original_wait(kind, timeout)

    proxy._wait_profile_ack = wait_for_delivered_ack
    proxy._profile_control_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="proxy_profile_control"
    )
    yield proxy
    # ``pytest-threadleak`` fails any test that leaves the owner thread
    # running, so always drain the executor -- including when the test
    # already cleared the attribute to simulate teardown.
    executor = proxy._profile_control_executor
    if executor is not None:
        executor.shutdown(wait=True)
        proxy._profile_control_executor = None
    proxy._request_sender.close(timeout=5)


def test_stop_profile_blocks_until_worker_acks(bare_proxy):
    """*Reproducer* for §2.1.

    Simulates a worker that takes 500ms to actually flush the chrome
    trace. ``stop_profile()`` must not return before that flush
    completes — otherwise the documented "trace is on disk by the time
    /stop_profile returns 200" contract is violated.

    On HEAD (pre-fix): proxy.stop_profile() returns in microseconds
    because ``request_queue.put`` is non-blocking and there is no ack
    machinery. ``elapsed`` will be < 0.01s and the assertion below
    will FAIL — that is the bug being demonstrated.

    Post-fix: the proxy waits on ``profile_ack_queue`` until the worker
    has finished processing the StopProfileRequest. ``elapsed`` will be
    ~0.5s and the assertion will PASS.
    """
    proxy = bare_proxy

    worker_flush_duration = 0.5  # seconds the "worker" pretends to flush

    def fake_worker_ack():
        # Simulate the worker subprocess: dequeue the StopProfileRequest,
        # spend ``worker_flush_duration`` flushing the chrome trace, then
        # ack via ``profile_ack_queue``.
        time.sleep(worker_flush_duration)
        # The fix has the worker push an ack object to this queue after
        # ``worker.stop_profile()`` returns. Pre-fix this code path is
        # never read by the proxy, so the ack is dropped on the floor.
        proxy.profile_ack_queue.get.return_value = ("stop", None)

    # Wire the mock so .get() blocks until the worker thread fires the ack.
    ack_event = threading.Event()

    def blocking_get(timeout=None):
        # Wait for the worker thread to finish, then return the ack.
        if not ack_event.wait(timeout=timeout):
            raise queue.Empty()
        return ("stop", None)

    proxy.profile_ack_queue.get.side_effect = blocking_get

    def worker():
        time.sleep(worker_flush_duration)
        ack_event.set()

    threading.Thread(target=worker, daemon=True).start()

    t0 = time.monotonic()
    proxy.stop_profile()
    elapsed = time.monotonic() - t0

    # The request must have been forwarded to the worker.
    assert proxy.request_queue.put.call_count == 1
    assert isinstance(proxy.request_queue.put.call_args.args[0], StopProfileRequest)

    # The crucial contract: stop_profile must not return until the
    # worker has acked. Pre-fix this assertion FAILS (elapsed << 0.5s).
    assert elapsed >= worker_flush_duration * 0.9, (
        f"stop_profile returned in {elapsed:.3f}s — that is BEFORE the "
        f"worker finished its {worker_flush_duration:.3f}s flush. The "
        f"chrome trace would not be on disk yet. (See PR review §2.1.)"
    )


def test_start_profile_blocks_until_worker_acks(bare_proxy, tmp_path):
    """/start_profile blocks until the worker acks.

    Same contract as ``/stop_profile``: the handler should not return
    200 until the worker has accepted (or rejected) the request, so
    that 409 conflicts are visible in the IPC-proxy path too.
    """
    proxy = bare_proxy

    worker_setup_duration = 0.3
    ack_event = threading.Event()

    def blocking_get(timeout=None):
        if not ack_event.wait(timeout=timeout):
            raise queue.Empty()
        return ("start", None)  # ("status", optional_error_message)

    proxy.profile_ack_queue.get.side_effect = blocking_get

    def worker():
        time.sleep(worker_setup_duration)
        ack_event.set()

    threading.Thread(target=worker, daemon=True).start()

    t0 = time.monotonic()
    proxy.start_profile(output_dir=str(tmp_path), num_steps=5)
    elapsed = time.monotonic() - t0

    assert proxy.request_queue.put.call_count == 1
    assert isinstance(proxy.request_queue.put.call_args.args[0], StartProfileRequest)

    assert elapsed >= worker_setup_duration * 0.9, (
        f"start_profile returned in {elapsed:.3f}s before worker acked "
        f"(expected >= {worker_setup_duration:.3f}s). See PR review §2.1."
    )


def test_start_profile_propagates_worker_rejection_as_runtime_error(bare_proxy):
    """Worker rejection surfaces as RuntimeError on the proxy.

    If the worker's PyExecutor rejected the start (RequestError —
    "already in progress"), the proxy must surface that as a
    ``RuntimeError`` so the HTTP handler can map it to 409. Pre-fix the
    rejection was logged on the worker and silently dropped; the HTTP
    layer always saw success.
    """
    proxy = bare_proxy

    def fake_get(timeout=None):
        # Worker reports the start was rejected by PyExecutor.
        return ("start", "Profiling is already in progress (...)")

    proxy.profile_ack_queue.get.side_effect = fake_get

    with pytest.raises(RuntimeError, match="already in progress"):
        proxy.start_profile()


@pytest.mark.parametrize("kind", ["start", "stop"])
def test_profile_control_after_shutdown_raises_runtime_error(bare_proxy, kind):
    """Profile control after shutdown fails cleanly, not with AttributeError.

    ``shutdown()`` drains ``_profile_control_executor`` and sets it to
    ``None``. A ``/start_profile`` or ``/stop_profile`` racing teardown
    would then hit ``None.submit(...)``. ``OpenAIServer`` only catches
    ``RuntimeError``/``OSError``/``ValueError``/``TimeoutError``, so an
    ``AttributeError`` escapes to FastAPI instead of producing a
    controlled response.
    """
    proxy = bare_proxy
    proxy._profile_control_executor.shutdown(wait=True)
    proxy._profile_control_executor = None

    with pytest.raises(RuntimeError, match=f"{kind}_profile"):
        getattr(proxy, f"{kind}_profile")()

    # The ZMQ queues must not be touched once the owning thread is gone.
    proxy.request_queue.put.assert_not_called()
    proxy.profile_ack_queue.get.assert_not_called()


@pytest.mark.parametrize("kind", ["start", "stop"])
def test_profile_control_during_pre_shutdown_raises_runtime_error(bare_proxy, kind):
    """Profile control is rejected as soon as ``pre_shutdown()`` starts.

    ``pre_shutdown()`` sets ``doing_shutdown`` and sends the quit sentinel
    to the workers, but ``shutdown()`` only clears
    ``_profile_control_executor`` later. In that window the executor is
    still live, so a profile call would submit and then block for the full
    ack timeout against workers that are already exiting.
    """
    proxy = bare_proxy
    proxy.doing_shutdown = True

    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match=f"{kind}_profile"):
        getattr(proxy, f"{kind}_profile")()
    # Rejected immediately rather than after the ack timeout.
    assert time.monotonic() - t0 < 1.0

    proxy.request_queue.put.assert_not_called()
    proxy.profile_ack_queue.get.assert_not_called()


def test_stop_profile_timeout_warns_does_not_hang(bare_proxy, monkeypatch):
    """Timeout produces a warning rather than wedging the event loop.

    If the worker is stuck (no ack within the timeout), the proxy
    must not hang the HTTP event loop forever. It logs a warning and
    returns so the handler can produce a 5xx response.

    ``queue.Queue.get(timeout=...)`` raises ``queue.Empty`` on timeout
    (not ``TimeoutError``), so the mock must mirror that to exercise the
    real ``except Empty`` branch in ``_wait_profile_ack``.
    """
    proxy = bare_proxy
    monkeypatch.setattr(proxy, "_STOP_PROFILE_ACK_TIMEOUT_S", 0.05)

    def never_ack(timeout=None):
        time.sleep(timeout)
        raise queue.Empty()

    proxy.profile_ack_queue.get.side_effect = never_ack

    # Should not raise; should not hang.
    t0 = time.monotonic()
    proxy.stop_profile()
    elapsed = time.monotonic() - t0
    # The TimeoutError path returns promptly.
    assert elapsed < 1.0


@pytest.mark.threadleak(enabled=True)
def test_profile_ack_wait_fails_promptly_after_worker_death():
    proxy = object.__new__(GenerationExecutorProxy)
    proxy.workers_started = False
    proxy.doing_shutdown = False
    proxy._engine_dead = False
    proxy._fatal_error = None
    proxy._results = {}
    proxy._request_sender = None
    entered = threading.Event()
    outcomes = queue.Queue()

    def wait_for_ack():
        ack = IpcQueue(is_server=True)
        original_get = ack.get

        def observed_get(*args, **kwargs):
            entered.set()
            return original_get(*args, **kwargs)

        ack.get = observed_get
        proxy.profile_ack_queue = ack
        try:
            proxy._wait_profile_ack("stop", 1.5)
            outcomes.put(None)
        except BaseException as error:
            outcomes.put(error)
        finally:
            ack.close()

    owner = threading.Thread(target=wait_for_ack)
    owner.start()
    try:
        assert entered.wait(3)
        proxy._set_fatal_error(RuntimeError("worker died during profile"))
        owner.join(0.5)
        assert not owner.is_alive(), "fatal profiling wait kept HTTP shutdown blocked"
        error = outcomes.get_nowait()
        assert isinstance(error, EngineDeadError)
        assert "worker died during profile" in str(error)
    finally:
        owner.join(3)
    assert not owner.is_alive()


def test_profile_control_pinned_to_single_owner_thread(bare_proxy, tmp_path):
    """Regression test for QiJune's review §2.

    All ZMQ socket operations triggered by ``start_profile`` /
    ``stop_profile`` must run on the same single owner thread, even
    though the public API is invoked from many different caller
    threads (``asyncio.to_thread`` does not pin to one worker).

    We capture the ``threading.current_thread()`` identity inside both
    the ``request_queue.put`` and ``profile_ack_queue.get`` mocks, and
    invoke the proxy from several caller threads in sequence. After
    the round-trip we assert:

      * Every queue operation observed exactly one owner thread.
      * That owner thread is *not* any of the caller threads — it is
        the dedicated ``proxy_profile_control`` worker created by
        ``_profile_control_executor``.
    """
    proxy = bare_proxy

    seen_threads = []

    def record_put(req):
        seen_threads.append(("put", threading.current_thread()))

    def record_get(timeout=None):
        seen_threads.append(("get", threading.current_thread()))
        # Return a matching ack so the call can complete promptly.
        # Match whatever kind the most recent put requested.
        last_req = seen_threads[-2][1] if len(seen_threads) >= 2 else None  # noqa: F841
        # We return ("start", None) and ("stop", None) alternately; the
        # caller pattern below issues start then stop, so this side_effect
        # list approach is simpler than parsing the mock call history.
        return record_get._next_ack.pop(0)

    record_get._next_ack = [("start", None), ("stop", None)]

    proxy.request_queue.put.side_effect = record_put
    proxy.profile_ack_queue.get.side_effect = record_get

    caller_threads = []

    def run_start():
        caller_threads.append(threading.current_thread())
        proxy.start_profile(output_dir=str(tmp_path), num_steps=5)

    def run_stop():
        caller_threads.append(threading.current_thread())
        proxy.stop_profile()

    # Invoke from two distinct caller threads to mirror what
    # ``asyncio.to_thread`` may do under load.
    t1 = threading.Thread(target=run_start, name="caller_1")
    t1.start()
    t1.join()
    t2 = threading.Thread(target=run_stop, name="caller_2")
    t2.start()
    t2.join()

    # 2 puts (start request, stop request) + 2 gets (each ack) = 4 ops.
    assert len(seen_threads) == 4, seen_threads

    owners = {op: {th for kind, th in seen_threads if kind == op} for op in ("put", "get")}
    assert all(len(threads) == 1 for threads in owners.values())
    assert next(iter(owners["put"])).name == "proxy_request_sender"
    assert next(iter(owners["get"])).name.startswith("proxy_profile_control")
    assert not set(caller_threads).intersection(owners["put"] | owners["get"])


# --------------------------------------------------------------------------- #
# Strengthened regression tests for QiJune's PR review §2.
#
# pyzmq sockets are not thread-safe — concurrent access from multiple threads
# (even one writer + one reader) is undefined behavior in libzmq. The fix
# routes ALL profile-control ZMQ ops through ``_profile_control_executor``
# (max_workers=1) so the FastAPI thread-pool worker never touches the socket.
# These tests pin that contract under stress.
# --------------------------------------------------------------------------- #


class _ThreadSafetyRecorder:
    """Records every queue access plus a peak concurrency counter.

    Threads call ``record()`` on enter and increment a counter under a lock;
    any peak value > 1 violates the single-owner-thread contract.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._active = {"put": 0, "get": 0}
        self.peak_concurrency = 0
        self.events = []  # list of ("op", thread)

    def record(self, op):
        with self._lock:
            self._active[op] += 1
            self.peak_concurrency = max(self.peak_concurrency, self._active[op])
            self.events.append((op, threading.current_thread()))
        try:
            # Hold the "inside-the-socket" window briefly so any
            # concurrent caller has a chance to be observed.
            time.sleep(0.001)
        finally:
            with self._lock:
                self._active[op] -= 1


def test_profile_ops_serialized_under_concurrent_callers(bare_proxy, tmp_path):
    """N concurrent callers must not produce concurrent ZMQ access.

    Mirrors what FastAPI would do under load: many request handlers each
    schedule a /start_profile or /stop_profile via ``asyncio.to_thread``
    simultaneously. If the proxy ever directly touched ZMQ sockets from
    those threads we would see ``peak_concurrency > 1``. The fix routes
    every op through ``_profile_control_executor`` (max_workers=1), so
    peak concurrency MUST stay at 1.
    """
    proxy = bare_proxy
    recorder = _ThreadSafetyRecorder()

    n_callers = 16
    # Track the kind of each put so the corresponding get returns a
    # matching ack — _wait_profile_ack uses the kind to detect stale acks
    # and may discard mismatched ones, which would otherwise exhaust a
    # pre-baked ack list.
    put_kinds = []
    state_lock = threading.Lock()

    def record_put(req):
        recorder.record("put")
        with state_lock:
            put_kinds.append("start" if isinstance(req, StartProfileRequest) else "stop")

    def record_get(timeout=None):
        recorder.record("get")
        with state_lock:
            # Pop the oldest pending put kind — guaranteed to be present
            # since the executor serializes put→get pairs.
            kind = put_kinds.pop(0)
        return (kind, None)

    proxy.request_queue.put.side_effect = record_put
    proxy.profile_ack_queue.get.side_effect = record_get

    def caller(idx):
        proxy.start_profile(output_dir=str(tmp_path), num_steps=1)
        proxy.stop_profile()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=n_callers,
        thread_name_prefix="fake_fastapi_to_thread",
    ) as pool:
        futures = [pool.submit(caller, i) for i in range(n_callers)]
        for f in futures:
            f.result()

    # 4 ops per caller (start put, start get, stop put, stop get).
    assert len(recorder.events) == 4 * n_callers, recorder.events

    # The crucial contract: ZMQ queue is never accessed by 2 threads
    # at the same time.
    assert recorder.peak_concurrency == 1, (
        f"Peak concurrent ZMQ queue access was "
        f"{recorder.peak_concurrency} — pyzmq is NOT thread-safe and "
        f"this would corrupt the socket. (See PR review §2.)"
    )

    owners = {op: {th for kind, th in recorder.events if kind == op} for op in ("put", "get")}
    assert all(len(threads) == 1 for threads in owners.values())
    assert next(iter(owners["put"])).name == "proxy_request_sender"
    assert next(iter(owners["get"])).name.startswith("proxy_profile_control")


def test_profile_control_thread_name_is_stable_across_calls(bare_proxy, tmp_path):
    """The same single owner thread serves *every* call, not just one.

    A pool with ``max_workers=1`` reuses its worker thread across submitted
    callables, so even with thousands of calls we should only ever see one
    owner thread name. This guards against an accidental future change to
    ``max_workers > 1`` (which would still pass under serialized tests but
    break the contract under concurrency).
    """
    proxy = bare_proxy
    seen_owners = []
    seen_lock = threading.Lock()
    put_kinds = []

    def record_put(req):
        with seen_lock:
            seen_owners.append(threading.current_thread())
            put_kinds.append("start" if isinstance(req, StartProfileRequest) else "stop")

    def record_get(timeout=None):
        with seen_lock:
            seen_owners.append(threading.current_thread())
            kind = put_kinds.pop(0)
        return (kind, None)

    proxy.request_queue.put.side_effect = record_put
    proxy.profile_ack_queue.get.side_effect = record_get

    n_calls = 25
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for _ in range(n_calls):
            pool.submit(proxy.start_profile, output_dir=str(tmp_path), num_steps=1).result()
            pool.submit(proxy.stop_profile).result()

    # 2 ops per call * 2 calls per iter * n_calls = 4 * n_calls ops.
    assert len(seen_owners) == 4 * n_calls

    owner_set = set(seen_owners)
    assert len(owner_set) == 2, (
        f"Owner thread changed across calls: "
        f"{sorted(t.name for t in owner_set)}. "
        f"max_workers may have been bumped > 1."
    )


def test_caller_thread_never_touches_zmq_queue(bare_proxy, tmp_path):
    """The HTTP-handler thread is never the owner of a queue op.

    This is exactly what QiJune's review asked for: ``request_queue.put`` and
    ``profile_ack_queue.get`` must NOT run on the ``asyncio.to_thread``
    worker thread. We verify by capturing the caller's thread identity and
    asserting it is disjoint from the set of owner threads observed inside
    the queue mocks.
    """
    proxy = bare_proxy

    queue_op_threads = []
    queue_lock = threading.Lock()
    put_kinds = []

    def record_put(req):
        with queue_lock:
            queue_op_threads.append(threading.current_thread())
            put_kinds.append("start" if isinstance(req, StartProfileRequest) else "stop")

    def record_get(timeout=None):
        with queue_lock:
            queue_op_threads.append(threading.current_thread())
            kind = put_kinds.pop(0)
        return (kind, None)

    proxy.request_queue.put.side_effect = record_put
    proxy.profile_ack_queue.get.side_effect = record_get

    caller_threads = []
    caller_lock = threading.Lock()

    def caller():
        with caller_lock:
            caller_threads.append(threading.current_thread())
        proxy.start_profile(output_dir=str(tmp_path), num_steps=1)
        proxy.stop_profile()

    threads = [threading.Thread(target=caller, name=f"fake_to_thread_{i}") for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    caller_set = set(caller_threads)
    queue_owner_set = set(queue_op_threads)

    # The intersection MUST be empty — caller threads must never touch
    # the ZMQ socket directly.
    leaked = caller_set & queue_owner_set
    assert not leaked, (
        f"Caller threads {[t.name for t in leaked]} appeared as ZMQ "
        f"queue-op owners — pyzmq socket leaked into the FastAPI "
        f"thread-pool worker. (See PR review §2.)"
    )

    # And every owner is the dedicated executor worker.
    for th in queue_owner_set:
        assert th.name == "proxy_request_sender" or th.name.startswith("proxy_profile_control")
