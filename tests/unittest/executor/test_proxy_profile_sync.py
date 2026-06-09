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
import queue
import threading
import time
from unittest.mock import MagicMock

import pytest

from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.request import StartProfileRequest, StopProfileRequest


def _bare_proxy():
    """Build a minimal ``GenerationExecutorProxy`` for testing.

    Provides only the queues the profile handlers touch. Avoids the
    heavy real ``__init__`` (no MPI session, no zmq sockets, no worker
    subprocess).

    Also wires a real single-thread ``_profile_control_executor`` —
    this matches the production setup, which pins all profile-related
    ZMQ socket operations to one owning thread (see
    ``GenerationExecutorProxy.__init__``).
    """
    proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
    proxy.request_queue = MagicMock()
    # The fix introduces a ``profile_ack_queue`` attribute. Pre-fix this
    # attribute does not exist; we stub it as MagicMock so old code paths
    # accessing it via getattr() see a sentinel.
    proxy.profile_ack_queue = MagicMock()
    proxy._profile_control_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="proxy_profile_control"
    )
    return proxy


def test_stop_profile_blocks_until_worker_acks():
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
    proxy = _bare_proxy()

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
            raise TimeoutError("ack not received within timeout")
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


def test_start_profile_blocks_until_worker_acks(tmp_path):
    """/start_profile blocks until the worker acks.

    Same contract as ``/stop_profile``: the handler should not return
    200 until the worker has accepted (or rejected) the request, so
    that 409 conflicts are visible in the IPC-proxy path too.
    """
    proxy = _bare_proxy()

    worker_setup_duration = 0.3
    ack_event = threading.Event()

    def blocking_get(timeout=None):
        if not ack_event.wait(timeout=timeout):
            raise TimeoutError("ack not received within timeout")
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


def test_start_profile_propagates_worker_rejection_as_runtime_error():
    """Worker rejection surfaces as RuntimeError on the proxy.

    If the worker's PyExecutor rejected the start (RequestError —
    "already in progress"), the proxy must surface that as a
    ``RuntimeError`` so the HTTP handler can map it to 409. Pre-fix the
    rejection was logged on the worker and silently dropped; the HTTP
    layer always saw success.
    """
    proxy = _bare_proxy()

    def fake_get(timeout=None):
        # Worker reports the start was rejected by PyExecutor.
        return ("start", "Profiling is already in progress (...)")

    proxy.profile_ack_queue.get.side_effect = fake_get

    with pytest.raises(RuntimeError, match="already in progress"):
        proxy.start_profile()


def test_stop_profile_timeout_warns_does_not_hang():
    """Timeout produces a warning rather than wedging the event loop.

    If the worker is stuck (no ack within the timeout), the proxy
    must not hang the HTTP event loop forever. It logs a warning and
    returns so the handler can produce a 5xx response.

    ``queue.Queue.get(timeout=...)`` raises ``queue.Empty`` on timeout
    (not ``TimeoutError``), so the mock must mirror that to exercise the
    real ``except Empty`` branch in ``_wait_profile_ack``.
    """
    proxy = _bare_proxy()

    def never_ack(timeout=None):
        raise queue.Empty()

    proxy.profile_ack_queue.get.side_effect = never_ack

    # Should not raise; should not hang.
    t0 = time.monotonic()
    proxy.stop_profile()
    elapsed = time.monotonic() - t0
    # The TimeoutError path returns promptly.
    assert elapsed < 1.0


def test_profile_control_pinned_to_single_owner_thread(tmp_path):
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
    proxy = _bare_proxy()

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

    owner_threads = {th for _, th in seen_threads}
    assert len(owner_threads) == 1, (
        f"Profile-control queue ops touched multiple threads: "
        f"{[(op, th.name) for op, th in seen_threads]}"
    )

    owner = next(iter(owner_threads))
    # The owner is the executor's worker, not either caller thread.
    assert owner not in caller_threads, (
        f"Owner thread {owner.name!r} matched a caller thread; the "
        "executor pinning is not effective."
    )
    assert owner.name.startswith("proxy_profile_control"), (
        f"Owner thread name {owner.name!r} does not match the expected "
        "'proxy_profile_control' prefix."
    )
