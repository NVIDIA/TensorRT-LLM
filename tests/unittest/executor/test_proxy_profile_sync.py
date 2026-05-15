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
    """
    proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
    proxy.request_queue = MagicMock()
    # The fix introduces a ``profile_ack_queue`` attribute. Pre-fix this
    # attribute does not exist; we stub it as MagicMock so old code paths
    # accessing it via getattr() see a sentinel.
    proxy.profile_ack_queue = MagicMock()
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


def test_start_profile_blocks_until_worker_acks():
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
    proxy.start_profile(output_dir="/tmp/x", num_steps=5)
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
    """
    proxy = _bare_proxy()

    def never_ack(timeout=None):
        raise TimeoutError("simulated worker hang")

    proxy.profile_ack_queue.get.side_effect = never_ack

    # Should not raise; should not hang.
    t0 = time.monotonic()
    proxy.stop_profile()
    elapsed = time.monotonic() - t0
    # The TimeoutError path returns promptly.
    assert elapsed < 1.0
