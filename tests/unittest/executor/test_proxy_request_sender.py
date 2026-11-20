# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for request batching between the proxy and the rank-0 worker.

``TRTLLM_PROXY_BATCH_REQUESTS=1`` makes the proxy pickle each request on the
submitting thread and send bursts of payloads as one list; the worker unpickles
each payload.
"""

import asyncio
import pickle
import queue
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import zmq
from utils.executor_stubs import bare_proxy as _bare_proxy
from utils.executor_stubs import stub_command_socket

from tensorrt_llm.executor.ipc import IpcQueue
from tensorrt_llm.executor.proxy import _create_request_sender, _RequestSender
from tensorrt_llm.executor.request import CancellingRequest
from tensorrt_llm.executor.utils import EngineDeadError, EngineUnavailableError
from tensorrt_llm.executor.worker import _incoming_requests

pytestmark = pytest.mark.cpu_only


class _RecordingQueue:
    """Request-queue stand-in that records each message as rank 0 decodes it.

    With ``block_first_put``, the first message stays in flight until
    ``release_first_put`` is set: an owner-thread write blocks for that time,
    and with ``direct`` later writes find the socket unwritable instead.
    Without ``direct``, like ``ZeroMqQueue`` with ``TLLM_LLMAPI_ZMQ_DEBUG=1``,
    it rejects a write from any thread other than the first one that wrote, so
    the production factory keeps every write on the sender thread.
    """

    _zmq_debug_enabled = True

    def __init__(self, block_first_put: bool = False, *, direct: bool = False):
        self.direct = direct
        self.messages = []
        self.socket = stub_command_socket(self._check_writer)
        self.writer = None
        self.first_put_started = threading.Event()
        self.release_first_put = threading.Event()
        if not block_first_put:
            self.release_first_put.set()

    def _check_writer(self):
        if self.direct:
            return
        if self.writer is None:
            self.writer = threading.get_ident()
        elif self.writer != threading.get_ident():
            raise RuntimeError("request queue written from a second thread")

    def setup_lazily(self):
        self._check_writer()

    def sign_pickled(self, pickled):
        return pickled

    def put_signed(self, message, *, timeout=None):
        self._check_writer()
        if self.direct:
            if self.messages and not self.release_first_put.wait(timeout):
                raise zmq.Again()
        elif not self.messages:
            self.first_put_started.set()
            self.release_first_put.wait(timeout=10)
        self.messages.append(pickle.loads(message))
        self.first_put_started.set()

    def close(self):
        self._check_writer()


def _payloads(*items):
    return [pickle.dumps(item) for item in items]


def _pre_shutdown(proxy):
    """Run ``pre_shutdown`` on a bare proxy with live workers, on a new thread."""
    proxy.workers_started = True
    shutdown = threading.Thread(target=proxy.pre_shutdown)
    shutdown.start()
    return shutdown


def _capturing_proxy():
    proxy = _bare_proxy()
    sent = []
    proxy._request_sender = SimpleNamespace(
        send=lambda payload, **kwargs: sent.append(kwargs["wait_for_capacity"])
    )
    return proxy, sent


def test_generation_admission_waits_only_for_blocking_callers():
    from tensorrt_llm.executor.request import GenerationRequest
    from tensorrt_llm.executor.utils import sync_submission
    from tensorrt_llm.sampling_params import SamplingParams

    proxy, sent = _capturing_proxy()
    request = GenerationRequest([1], SamplingParams())

    async def on_event_loop():
        proxy._send_request(request)  # an async server handler fails fast
        # A blocking generate() waits for capacity even on a thread that also
        # runs an event loop (Jupyter), so a large batch is not rejected
        # partway through when the command queue fills.
        with sync_submission([]):
            proxy._send_request(request)

    proxy._send_request(request)  # a plain blocking caller waits
    asyncio.run(on_event_loop())
    assert sent == [True, False, True]


class _Future:
    def __init__(self):
        self.aborted = False

    def abort(self):
        self.aborted = True

    def result(self):
        return self


@pytest.mark.parametrize("failure", [EngineUnavailableError, KeyboardInterrupt])
def test_failed_blocking_generate_aborts_the_requests_it_submitted(failure):
    from tensorrt_llm.executor.utils import submission_waits_for_capacity
    from tensorrt_llm.llmapi.llm import BaseLLM

    submitted, waits = [], []

    class Llm(BaseLLM):
        def generate_async(self, inputs, **kwargs):
            waits.append(submission_waits_for_capacity())
            if len(submitted) == 2:
                raise failure("command queue is full")
            submitted.append(_Future())
            return submitted[-1]

        def shutdown(self):
            pass

    llm = Llm.__new__(Llm)

    async def in_notebook():
        # A Jupyter cell calls the blocking API on the loop's thread.
        llm.generate([[1], [2], [3]], use_tqdm=False)

    with pytest.raises(failure):
        asyncio.run(in_notebook())
    assert waits == [True, True, True]
    assert [future.aborted for future in submitted] == [True, True]


def test_multimodal_encoder_generate_uses_blocking_submission():
    from tensorrt_llm.executor.utils import submission_waits_for_capacity
    from tensorrt_llm.llmapi.mm_encoder import MultimodalEncoder

    submitted, waits = [], []

    def generate_async(inputs):
        waits.append(submission_waits_for_capacity())
        if len(submitted) == 1:
            raise EngineUnavailableError("command queue is full")
        submitted.append(_Future())
        return submitted[-1]

    encoder = MultimodalEncoder.__new__(MultimodalEncoder)
    encoder.generate_async = generate_async
    encoder.shutdown = lambda: None

    async def in_notebook():
        encoder.generate([{"prompt": "a"}, {"prompt": "b"}], use_tqdm=False)

    with pytest.raises(EngineUnavailableError):
        asyncio.run(in_notebook())
    assert waits == [True, True]
    assert submitted[0].aborted


@pytest.mark.parametrize("setting", [None, "0", "1"])
def test_batching_is_opt_in(monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv("TRTLLM_PROXY_BATCH_REQUESTS", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_PROXY_BATCH_REQUESTS", setting)
    batching = setting == "1"
    q = _RecordingQueue(block_first_put=True)
    sender = _create_request_sender(q, queue.Queue())
    sender.send(pickle.dumps("a"))
    assert not q.messages, "submission waited for its command to be written"
    assert q.first_put_started.wait(10)
    sender.send(pickle.dumps("b"))
    sender.send(pickle.dumps("c"))
    q.release_first_put.set()
    sender.close(timeout=10)
    expected = ["a", _payloads("b", "c")] if batching else ["a", "b", "c"]
    assert q.messages == expected


def _owner_sender(request_queue, error_queue=None, **kwargs):
    """Create a sender that writes only on its owner thread and coalesces bursts."""
    kwargs.setdefault("batching", True)
    error_queue = queue.Queue() if error_queue is None else error_queue
    return _RequestSender(request_queue, error_queue, direct_send=False, **kwargs)


def test_lone_request_is_sent_as_itself():
    request_queue = _RecordingQueue()
    sender = _owner_sender(request_queue)
    sender.send(pickle.dumps("a"))
    sender.close(timeout=10)
    assert request_queue.messages == ["a"]


def test_burst_is_sent_as_one_list_in_order():
    request_queue = _RecordingQueue(block_first_put=True)
    sender = _owner_sender(request_queue)
    sender.send(pickle.dumps(0))
    assert request_queue.first_put_started.wait(timeout=10)
    # Everything queued while the first send is in flight goes out as one list.
    for item in range(1, 6):
        sender.send(pickle.dumps(item))
    request_queue.release_first_put.set()
    sender.close(timeout=10)
    assert request_queue.messages == [0, _payloads(1, 2, 3, 4, 5)]


def test_close_flushes_queued_requests_and_drops_later_ones():
    request_queue = _RecordingQueue(block_first_put=True)
    sender = _owner_sender(request_queue)
    sender.send(pickle.dumps("first"))
    assert request_queue.first_put_started.wait(timeout=10)
    sender.send(pickle.dumps("queued"))
    closer = threading.Thread(target=sender.close, kwargs={"timeout": 10})
    closer.start()
    request_queue.release_first_put.set()
    closer.join(timeout=10)
    assert not closer.is_alive()
    assert request_queue.messages == ["first", "queued"]
    with pytest.raises(EngineUnavailableError):
        sender.send(pickle.dumps("late"))
    assert request_queue.messages == ["first", "queued"]


def test_close_is_idempotent():
    request_queue = _RecordingQueue()
    sender = _owner_sender(request_queue)
    sender.send(pickle.dumps("a"))
    sender.close(timeout=10)
    assert not sender._thread.is_alive()
    sender.close(timeout=10)
    assert request_queue.messages == ["a"]


def test_sender_thread_sends_the_sentinel_after_the_queued_requests():
    request_queue = _RecordingQueue(block_first_put=True)
    sender = _owner_sender(request_queue)
    sender.send(pickle.dumps("first"))
    assert request_queue.first_put_started.wait(timeout=10)
    sender.send(pickle.dumps("queued"))
    result = []
    closer = threading.Thread(
        target=lambda: result.append(sender.close(timeout=10, send_sentinel=True))
    )
    closer.start()
    request_queue.release_first_put.set()
    closer.join(timeout=10)
    assert result == [None]
    # The sentinel follows the queued commands, written by the sender thread.
    assert request_queue.messages == ["first", "queued", None]


def test_shutdown_without_requests_still_uses_the_socket_owner():
    q = _RecordingQueue()
    sender = _owner_sender(q)
    assert sender.close(timeout=10, send_sentinel=True) is None
    assert q.messages == [None]
    assert q.writer != threading.get_ident()


def test_close_during_interpreter_finalization_does_not_join():
    request_queue = _RecordingQueue(block_first_put=True)
    sender = _owner_sender(request_queue)
    sender.send(pickle.dumps("first"))
    assert request_queue.first_put_started.wait(timeout=10)
    sender.send(pickle.dumps("queued"))
    # Joining would wait for the blocked send; at teardown close() must return at
    # once, and must not ask anyone to send the sentinel.
    sender._is_finalizing = lambda: True
    assert sender.close(send_sentinel=True) is None
    assert sender._thread is not None
    # Outside finalization the same sender still flushes and stops.
    sender._is_finalizing = lambda: False
    request_queue.release_first_put.set()
    sender.close(timeout=10)
    assert not sender._thread.is_alive()
    assert request_queue.messages == ["first", "queued"]


def test_pre_shutdown_flushes_batched_requests_before_the_sentinel():
    request_queue = _RecordingQueue(block_first_put=True)
    proxy = _bare_proxy()
    proxy.request_queue = request_queue
    proxy._request_sender = _owner_sender(request_queue)
    proxy._send_request("first")
    assert request_queue.first_put_started.wait(timeout=10)
    proxy._send_request("queued")

    shutdown = _pre_shutdown(proxy)
    request_queue.release_first_put.set()
    shutdown.join(timeout=10)
    assert not shutdown.is_alive()
    proxy.workers_started = False
    # The sender thread wrote everything, the sentinel included.
    assert request_queue.messages == ["first", "queued", None]
    assert request_queue.writer != shutdown.ident


@pytest.mark.parametrize("batching", [True, False])
def test_pre_shutdown_sends_the_sentinel_itself_if_nothing_else_wrote(batching):
    request_queue = _RecordingQueue()
    proxy = _bare_proxy()
    proxy.request_queue = request_queue
    proxy._request_sender = _owner_sender(request_queue, batching=batching)

    shutdown = _pre_shutdown(proxy)
    shutdown.join(timeout=10)
    assert not shutdown.is_alive()
    proxy.workers_started = False
    assert request_queue.messages == [None]
    assert request_queue.writer != shutdown.ident


@pytest.mark.timeout(20, method="thread")
@pytest.mark.parametrize("late_close", [False, True])
def test_transport_error_reaches_error_queue(monkeypatch, late_close):
    monkeypatch.setenv("TRTLLM_PROXY_BATCH_REQUESTS", "1")
    published = threading.Event()
    finish_failure = threading.Event()

    class _ErrorQueue(queue.Queue):
        def put_nowait(self, error):
            super().put_nowait(error)
            published.set()
            if not late_close:
                assert finish_failure.wait(timeout=10)

    class _FailingQueue(_RecordingQueue):
        attempts = 0

        def put_signed(self, message, *, timeout=None):
            if pickle.loads(message) is None:
                return super().put_signed(message, timeout=timeout)
            self._check_writer()
            self.attempts += 1
            self.first_put_started.set()
            assert self.release_first_put.wait(timeout=10)
            raise RuntimeError("socket closed")

    request_queue = _FailingQueue(block_first_put=True)
    error_queue = _ErrorQueue()
    sender = _create_request_sender(request_queue, error_queue)
    closer = None
    try:
        sender.send(pickle.dumps("a"))
        assert request_queue.first_put_started.wait(timeout=10)
        sender.send(pickle.dumps("queued"))
        request_queue.release_first_put.set()
        assert published.wait(timeout=10)
        if late_close:
            # The failure owner must already know how to stop its workers;
            # cleanup can arrive after that owner has closed its socket.
            sender._thread.join(timeout=3)
            assert not sender._thread.is_alive()
        responsibility = []
        closer = threading.Thread(
            target=lambda: responsibility.append(sender.close(timeout=10, send_sentinel=True))
        )
        closer.start()
        if not late_close:
            with sender._cond:
                assert sender._cond.wait_for(lambda: sender._send_sentinel, timeout=10)
        finish_failure.set()
        closer.join(timeout=10)
        assert not closer.is_alive()
        assert request_queue.attempts == 1
        assert request_queue.messages == [None]
        error = error_queue.get_nowait()
        assert isinstance(error, RuntimeError)
        assert str(error) == "socket closed"
        # close() returns the failure that stopped the sender.
        assert responsibility == [error]
    finally:
        request_queue.release_first_put.set()
        finish_failure.set()
        if closer is not None:
            closer.join(timeout=10)
        sender.close(timeout=10)


def test_batched_request_is_captured_when_sent():
    proxy = _bare_proxy()
    sent = []
    proxy._request_sender = SimpleNamespace(
        send=lambda payload, **kwargs: sent.append(payload), close=lambda **kwargs: None
    )

    request = {"max_tokens": 1}
    proxy._send_request(request)
    # A change after submission does not reach the worker.
    request["max_tokens"] = 2
    assert pickle.loads(sent[0]) == {"max_tokens": 1}

    # A request that cannot be pickled fails in its caller, and nothing is queued.
    with pytest.raises(TypeError):
        proxy._send_request({"lock": threading.Lock()})
    assert len(sent) == 1


def test_worker_unpacks_batched_requests_in_order():
    class _ListQueue:
        def __init__(self, messages):
            self._messages = list(messages)

        def get(self):
            return self._messages.pop(0)

    # A batch is a list of pickled payloads; a lone command arrives already
    # unpickled by the queue.
    request_queue = _ListQueue(["a", _payloads("b", "c"), "d", None, "after sentinel"])
    assert list(_incoming_requests(request_queue)) == ["a", "b", "c", "d"]


@pytest.mark.timeout(30, method="thread")
@pytest.mark.parametrize("direct_send", [False, True])
@pytest.mark.parametrize("push", [False, True])
def test_initial_connection_timeout_fails_accepted_work(tmp_path, push, direct_send):
    if push:
        commands = IpcQueue(
            (f"ipc://{tmp_path}/absent", b"k" * 32),
            is_server=False,
            socket_type=zmq.PUSH,
        )
    else:
        commands = IpcQueue(is_server=True)
    errors = queue.Queue()
    sender = _RequestSender(
        commands, errors, batching=not direct_send, connect_timeout=0.15, direct_send=direct_send
    )
    try:
        sender.send(pickle.dumps("accepted"))
        assert isinstance(errors.get(timeout=10), TimeoutError)
        with pytest.raises(EngineDeadError):
            sender.send(pickle.dumps("late"))
    finally:
        sender.close(timeout=2)
    assert not sender._thread.is_alive()


@pytest.mark.timeout(10, method="thread")
@pytest.mark.parametrize("direct_send", [False, True])
@pytest.mark.parametrize("transport", ["tcp", "ipc"])
def test_established_disconnect_fails_while_idle_without_replay(tmp_path, transport, direct_send):
    address = (f"ipc://{tmp_path}/commands", b"k" * 32) if transport == "ipc" else None
    peer = IpcQueue(address, is_server=True, socket_type=zmq.PULL)
    address = peer.address
    commands = IpcQueue(address, is_server=False, socket_type=zmq.PUSH)
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=False, direct_send=direct_send)
    replacement = None
    try:
        sender.send(pickle.dumps("accepted"))
        assert peer.get(timeout=2) == "accepted"
        # No subsequent send or Again may be needed to discover missing tail data.
        peer.close()
        assert isinstance(errors.get(timeout=2), ConnectionError)
        sender._thread.join(timeout=2)
        assert not sender._thread.is_alive()
        replacement = IpcQueue(address, is_server=True, socket_type=zmq.PULL)
        with pytest.raises(EngineDeadError):
            sender.send(pickle.dumps("after-disconnect"))
        assert not replacement.poll(0.1)
    finally:
        sender.close(timeout=2)
        peer.close()
        if replacement is not None:
            replacement.close()


@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("transport", ["tcp", "ipc"])
def test_stop_sentinel_after_engine_failure_survives_commands_in_flight(tmp_path, transport):
    # After a terminal failure, the stop sentinel still reaches a live rank 0
    # when a large command is in flight ahead of it, so rank 0 stops and
    # releases its workers and GPU memory.
    for attempt in range(3):
        address = (f"ipc://{tmp_path}/commands{attempt}", b"k" * 32) if transport == "ipc" else None
        peer = IpcQueue(address, is_server=True, socket_type=zmq.PULL)
        commands = IpcQueue(
            peer.address,
            is_server=False,
            socket_type=zmq.PUSH,
            send_hwm=_RequestSender.send_hwm(batching=False),
        )
        received, stop = [], threading.Event()

        def rank0_loop():
            while not stop.is_set():
                if peer.poll(0.05):
                    received.append(peer.get())

        reader = threading.Thread(target=rank0_loop, daemon=True)
        reader.start()
        sender = _RequestSender(
            commands,
            queue.Queue(),
            batching=False,
            stop_workers_on_failure=True,
            direct_send=True,
        )
        try:
            sender.send(pickle.dumps("connected"))
            deadline = time.monotonic() + 5
            while not received and time.monotonic() < deadline:
                time.sleep(0.005)
            # For example, a multimodal request still in flight.
            sender.send(pickle.dumps(b"x" * (4 << 20)))
            # What _mark_engine_dead does, then pre_shutdown's later close.
            sender.fail(RuntimeError("the engine was declared dead"))
            sender._thread.join(timeout=5)
            sender.close(timeout=7.0, send_sentinel=True)
            deadline = time.monotonic() + 3
            while None not in received and time.monotonic() < deadline:
                time.sleep(0.01)
            assert None in received, f"attempt {attempt}: stop sentinel never reached rank 0"
        finally:
            stop.set()
            reader.join(timeout=2)
            peer.close()


def _sender_behind_a_blocked_write(direct_send, error_queue=None):
    """Return a sender whose first command stays in flight until released.

    The owner-thread mode writes one command per message, like direct mode.
    """
    q = _RecordingQueue(block_first_put=True, direct=direct_send)
    error_queue = queue.Queue() if error_queue is None else error_queue
    sender = _RequestSender(q, error_queue, batching=not direct_send, direct_send=direct_send)
    sender.MAX_BATCH_SIZE = 1
    return q, sender


@pytest.mark.timeout(12, method="thread")
@pytest.mark.parametrize("direct_send", [False, True])
def test_control_commands_and_cancellations_bypass_a_full_request_queue(direct_send):
    q, sender = _sender_behind_a_blocked_write(direct_send)
    sender.MAX_PENDING_REQUESTS = 1
    try:
        sender.send(pickle.dumps("in-flight"))
        assert q.first_put_started.wait(3)
        sender.send(pickle.dumps("queued"))
        with pytest.raises(EngineUnavailableError):
            sender.send(pickle.dumps("overload"))
        # Control commands have no queue budget of their own; their callers
        # already send one at a time.
        sender.send(pickle.dumps("start-profile"), control=True)
        sender.send(pickle.dumps("stop-profile"), control=True)
        sender.send(pickle.dumps("cancel"), cancellation_id=7)
        # A second cancellation of a still-queued id is a duplicate.
        sender.send(pickle.dumps("cancel-again"), cancellation_id=7)
    finally:
        q.release_first_put.set()
        sender.close(timeout=3, send_sentinel=True)
    assert q.messages == ["in-flight", "queued", "start-profile", "stop-profile", "cancel", None]


@pytest.mark.timeout(12, method="thread")
@pytest.mark.parametrize("direct_send", [False, True])
@pytest.mark.parametrize("local_final", [False, True])
def test_proxy_retains_cancellations_when_the_request_queue_is_full(local_final, direct_send):
    proxy = _bare_proxy()
    unrelated = SimpleNamespace(queue=queue.Queue())
    proxy._results = {
        42: SimpleNamespace(queue=queue.Queue()),
        43: SimpleNamespace(queue=queue.Queue()),
        44: unrelated,
    }
    q, sender = _sender_behind_a_blocked_write(direct_send, proxy._error_queue)
    sender.MAX_PENDING_REQUESTS = 1
    proxy._request_sender = sender
    try:
        sender.send(pickle.dumps("in-flight"))
        assert q.first_put_started.wait(3)
        sender.send(pickle.dumps("queued"))
        if local_final:
            from tensorrt_llm.executor.postproc_worker import PostprocWorker

            response = PostprocWorker.Output(42, None, True, should_abort=True)
            proxy.result_queue = SimpleNamespace(get=lambda: response)
            assert proxy.dispatch_result_task()
            assert 42 not in proxy._results
        else:
            proxy.abort_request(42)
        proxy.abort_request(42)
        proxy.abort_request(43)
        assert not proxy._engine_dead
        assert unrelated.queue.empty()
        assert proxy._error_queue.empty()
    finally:
        q.release_first_put.set()
        sender.close(timeout=3)
    delivered = q.messages[2:]
    assert [request.id for request in delivered] == [42, 43]
    assert all(isinstance(request, CancellingRequest) for request in delivered)


@pytest.mark.timeout(12, method="thread")
@pytest.mark.parametrize("direct_send", [False, True])
def test_shutdown_cancels_survive_a_full_request_queue(direct_send):
    q, sender = _sender_behind_a_blocked_write(direct_send)
    sender.MAX_PENDING_REQUESTS = 1
    sender.send(pickle.dumps("in-flight"))
    assert q.first_put_started.wait(3)
    sender.send(pickle.dumps("queued"))
    closed = threading.Thread(target=lambda: sender.close(timeout=3, cancel_ids=(7, 8)))
    closed.start()
    q.release_first_put.set()
    closed.join(timeout=4)
    assert not closed.is_alive()
    assert q.messages[:2] == ["in-flight", "queued"]
    cancels = q.messages[2:]
    assert all(isinstance(request, CancellingRequest) for request in cancels)
    assert [request.id for request in cancels] == [7, 8]
    assert None not in q.messages


@pytest.mark.timeout(30, method="thread")
@pytest.mark.parametrize("direct_send", [False, True])
def test_connected_stall_longer_than_the_connect_timeout_keeps_accepted_work(direct_send):
    blocked = threading.Event()

    class ObservedQueue(IpcQueue):
        def put_signed(self, message, **kwargs):
            try:
                return super().put_signed(message, **kwargs)
            except zmq.Again:
                blocked.set()
                raise

    peer = IpcQueue(is_server=True, socket_type=zmq.PULL, receive_hwm=1)
    peer.socket.setsockopt(zmq.RCVBUF, 256 * 1024)
    commands = ObservedQueue(peer.address, is_server=False, socket_type=zmq.PUSH, send_hwm=1)
    commands.socket.setsockopt(zmq.SNDBUF, 256 * 1024)
    errors = queue.Queue()
    connect_timeout = 1.0
    sender = _RequestSender(
        commands, errors, batching=False, connect_timeout=connect_timeout, direct_send=direct_send
    )
    count = 16
    data = b"x" * (1024 * 1024)
    try:
        for i in range(count):
            sender.send(pickle.dumps((i, data)))
        assert blocked.wait(10), "real transport did not reach backpressure"
        deadline = time.monotonic() + 10
        while not sender._connected:
            assert time.monotonic() < deadline, "the command connection was not established"
            time.sleep(0.01)
        # A connected worker can stop consuming for longer than the connect
        # timeout. The sender keeps the accepted work and stays available.
        time.sleep(2 * connect_timeout)
        assert sender.available
        assert errors.empty()
        sender.send(pickle.dumps((count, data)))
        received = []
        for _ in range(count + 1):
            request_id, contents = peer.get(timeout=10)
            assert contents == data
            received.append(request_id)
        assert received == list(range(count + 1))
        assert errors.empty()
    finally:
        sender.close(timeout=2)
        peer.socket.setsockopt(zmq.LINGER, 0)
        peer.close()
    assert not sender._thread.is_alive()


@pytest.mark.timeout(10, method="thread")
@pytest.mark.parametrize("direct_send", [False, True])
@pytest.mark.parametrize("stop", ["capacity", "close", "failure"])
def test_capacity_wait_unblocks_on_progress_or_shutdown(stop, direct_send):
    commands = _RecordingQueue(block_first_put=True, direct=direct_send)
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=False, direct_send=direct_send)
    sender.MAX_PENDING_REQUESTS = 1
    outcomes = queue.Queue()
    entered = threading.Event()

    def wait_for_capacity():
        entered.set()
        try:
            sender.send(pickle.dumps("waiting"), wait_for_capacity=True)
            outcomes.put(None)
        except (EngineDeadError, EngineUnavailableError) as error:
            outcomes.put(error)

    waiter = threading.Thread(target=wait_for_capacity)
    closer = None
    try:
        sender.send(pickle.dumps("in-flight"))
        assert commands.first_put_started.wait(2)
        sender.send(pickle.dumps("queued"))
        waiter.start()
        assert entered.wait(2)
        assert outcomes.empty()
        if stop == "failure":
            sender.fail(RuntimeError("worker died"))
        elif stop == "close":
            closer = threading.Thread(target=sender.close, kwargs={"timeout": 2})
            closer.start()
        else:
            commands.release_first_put.set()
        outcome = outcomes.get(timeout=2)
        if stop == "capacity":
            assert outcome is None
        elif stop == "failure":
            assert isinstance(outcome, EngineDeadError)
        else:
            assert isinstance(outcome, EngineUnavailableError)
    finally:
        commands.release_first_put.set()
        waiter.join(timeout=2)
        if closer is not None:
            closer.join(timeout=2)
        sender.close(timeout=2)
    assert not waiter.is_alive()
    if stop == "capacity":
        assert commands.messages == ["in-flight", "queued", "waiting"]


@pytest.mark.parametrize(
    ("batching", "debug", "direct"),
    [
        ("0", False, True),
        (None, False, True),
        ("1", False, True),
        ("0", True, False),
        ("1", True, False),
    ],
)
def test_production_sender_writes_on_the_submitting_thread_unless_zmq_debug(
    monkeypatch, batching, debug, direct
):
    if batching is None:
        monkeypatch.delenv("TRTLLM_PROXY_BATCH_REQUESTS", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_PROXY_BATCH_REQUESTS", batching)
    commands = IpcQueue(is_server=True)
    commands._zmq_debug_enabled = debug
    sender = _create_request_sender(commands, queue.Queue())
    try:
        assert sender._direct_send is direct
    finally:
        sender.close(timeout=2)
        commands.close()


class _ObservedQueue(IpcQueue):
    """Real command socket that records writers and the sender thread's waits."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writers = set()
        self.owner_blocked = threading.Event()

    def put_signed(self, message, **kwargs):
        try:
            super().put_signed(message, **kwargs)
        except zmq.Again:
            if threading.current_thread().name == "proxy_request_sender":
                self.owner_blocked.set()
            raise
        self.writers.add(threading.current_thread().name)


def _deschedule_owner(sender):
    """Stop the sender thread's progress, like a thread that is denied the GIL.

    The thread is held where it waits for work, outside the condition lock, so
    callers still run. Setting the returned event lets it continue.
    """
    resume = threading.Event()
    held = threading.Event()
    wake = sender._owner_wake

    class _HeldWake:
        def wait(self, timeout=None):
            if threading.current_thread() is sender._thread:
                held.set()
                resume.wait()
            return wake.wait(timeout)

        def __getattr__(self, name):
            return getattr(wake, name)

    sender._owner_wake = _HeldWake()
    # A wait already in progress ends within IDLE_CHECK_INTERVAL.
    assert held.wait(10), "the sender thread did not reach its next wait"
    return resume


def _record_owner_checks(sender):
    """Return a list that collects the time of each sender-thread terminal check.

    The sender thread makes this check on every pass, including each idle
    connection check.
    """
    checks = []
    check_terminal = sender._check_terminal

    def recording_check_terminal():
        if threading.current_thread() is sender._thread:
            checks.append(time.monotonic())
        return check_terminal()

    sender._check_terminal = recording_check_terminal
    return checks


def _wait_for_owner_checks(checks, count):
    """Wait until ``count`` more sender-thread checks are recorded in ``checks``."""
    target = len(checks) + count
    deadline = time.monotonic() + 10
    while len(checks) < target:
        assert time.monotonic() < deadline, "the idle sender thread stopped checking"
        time.sleep(0.01)


def _command_peer(commands):
    """Connect a worker-side PAIR peer and wait until commands can reach it."""
    peer = IpcQueue(commands.address, is_server=False)
    peer.setup_lazily()
    deadline = time.monotonic() + 5
    while not commands.socket.poll(0, zmq.POLLOUT):
        assert time.monotonic() < deadline, "command peer did not connect"
        time.sleep(0.01)
    return peer


def _close_quietly(*queues):
    for ipc_queue in queues:
        if ipc_queue.socket is not None:
            ipc_queue.socket.setsockopt(zmq.LINGER, 0)
        ipc_queue.close()


@pytest.mark.timeout(20, method="thread")
def test_unbatched_admission_does_not_wait_for_the_sender_thread():
    # Callers write unbatched commands themselves. A busy event loop can deny
    # the sender thread the GIL, and a connected, idle worker must still
    # receive every command without admission filling up.
    commands = _ObservedQueue(is_server=True, send_hwm=_RequestSender.send_hwm(batching=False))
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=False, direct_send=True)
    sender.MAX_PENDING_REQUESTS = 4
    peer = _command_peer(commands)
    resume = _deschedule_owner(sender)
    count = 4 * sender.MAX_PENDING_REQUESTS
    try:
        for item in range(count):
            sender.send(pickle.dumps(item))
        assert [peer.get(timeout=2) for _ in range(count)] == list(range(count))
        assert commands.writers == {threading.current_thread().name}
        assert sender._pending_requests == 0
        assert errors.empty()
    finally:
        resume.set()
        sender.close(timeout=2)
        _close_quietly(peer)
    assert not sender._thread.is_alive()


@pytest.mark.timeout(20, method="thread")
def test_admission_after_an_unwritable_socket_does_not_wait_for_the_sender_thread():
    # After one write finds the socket unwritable, callers still write the
    # queue once the worker connects, so a descheduled sender thread does not
    # starve a connected worker. The socket holds every command, so a slow
    # libzmq I/O thread on a loaded host cannot leave commands that only the
    # held-back sender would retry.
    commands = _ObservedQueue(is_server=True, send_hwm=256)
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=False, direct_send=True)
    sender.MAX_PENDING_REQUESTS = 16
    peer = None
    resume = None
    try:
        # The worker has not connected yet, so the sender thread must retry.
        sender.send(pickle.dumps(0))
        assert commands.owner_blocked.wait(5)
        resume = _deschedule_owner(sender)
        peer = _command_peer(commands)
        count = 4 * sender.MAX_PENDING_REQUESTS
        for item in range(1, count):
            sender.send(pickle.dumps(item))
        assert [peer.get(timeout=2) for _ in range(count)] == list(range(count))
        assert errors.empty()
    finally:
        if resume is not None:
            resume.set()
        sender.close(timeout=2)
        if peer is not None:
            _close_quietly(peer)


@pytest.mark.timeout(15, method="thread")
def test_peer_loss_seen_by_a_caller_stops_admission_without_the_sender_thread():
    # A caller that observes the command peer's loss stops admission and
    # reports the sender unavailable, without waiting for the possibly
    # descheduled sender thread to record the failure.
    commands = IpcQueue(is_server=True, send_hwm=_RequestSender.send_hwm(batching=False))
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=False, direct_send=True)
    peer = _command_peer(commands)
    resume = None
    try:
        sender.send(pickle.dumps("before"))
        assert peer.get(timeout=2) == "before"
        resume = _deschedule_owner(sender)
        _close_quietly(peer)
        accepted = 0
        deadline = time.monotonic() + 5
        with pytest.raises(EngineDeadError):
            while time.monotonic() < deadline:
                sender.send(pickle.dumps("after-loss"))
                accepted += 1
                time.sleep(0.01)
        assert accepted < 50, "admission continued long after the loss was observed"
        assert not sender.available
        assert errors.empty(), "the descheduled sender thread recorded the loss"
        resume.set()
        assert isinstance(errors.get(timeout=2), ConnectionError)
    finally:
        if resume is not None:
            resume.set()
        sender.close(timeout=2)
        _close_quietly(peer)


class _FailingSendQueue(IpcQueue):
    """Real command socket whose writes fail once connected, as on a broken socket."""

    def _send_data(self, data, flags=0, routing_id=None):
        raise zmq.ZMQError(zmq.ENOTSUP)


@pytest.mark.timeout(10, method="thread")
def test_direct_transport_error_is_recorded_by_the_sender_thread():
    commands = _FailingSendQueue(is_server=True, send_hwm=_RequestSender.send_hwm(batching=False))
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=False, direct_send=True)
    peer = _command_peer(commands)
    try:
        # The caller's failed write is reported by the sender thread.
        sender.send(pickle.dumps("rejected"))
        error = errors.get(timeout=2)
        assert isinstance(error, zmq.ZMQError) and error.errno == zmq.ENOTSUP
        with pytest.raises(EngineDeadError):
            sender.send(pickle.dumps("after-failure"))
    finally:
        sender.close(timeout=2)
        _close_quietly(peer)


@pytest.mark.timeout(10, method="thread")
def test_a_failed_backlog_write_is_not_reported_as_a_full_queue():
    # With batching, a caller that finds admission full writes the backlog. If
    # that write fails, the request fails as engine death, not as a retryable
    # "queue is full" from a sender that can no longer write.
    commands = _FailingSendQueue(is_server=True, send_hwm=_RequestSender.send_hwm(batching=True))
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=True, direct_send=True)
    sender.MAX_PENDING_REQUESTS = 2
    peer = _command_peer(commands)
    resume = _deschedule_owner(sender)
    try:
        for item in range(sender.MAX_PENDING_REQUESTS):
            sender.send(pickle.dumps(item))
        with pytest.raises(EngineDeadError) as raised:
            sender.send(pickle.dumps("over-the-limit"))
        assert isinstance(raised.value.root_cause, zmq.ZMQError)
        assert not sender.available
        resume.set()
        assert isinstance(errors.get(timeout=2), zmq.ZMQError)
    finally:
        resume.set()
        sender.close(timeout=2)
        _close_quietly(peer)


@pytest.mark.timeout(30, method="thread")
def test_busy_event_loop_does_not_fill_unbatched_admission():
    # The serving shape: a coroutine that keeps the loop busy and yields
    # between submissions denies the sender thread the GIL. No submission may
    # be rejected.
    commands = IpcQueue(is_server=True, send_hwm=_RequestSender.send_hwm(batching=False))
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=False, direct_send=True)
    sender.MAX_PENDING_REQUESTS = 64
    peer = _command_peer(commands)
    count = 1000

    async def submit():
        rejected = 0
        for item in range(count):
            deadline = time.perf_counter() + 300e-6
            while time.perf_counter() < deadline:
                pass
            try:
                sender.send(pickle.dumps(item))
            except EngineUnavailableError:
                rejected += 1
            await asyncio.sleep(0)
        return rejected

    try:
        assert asyncio.run(submit()) == 0
        assert [peer.get(timeout=2) for _ in range(count)] == list(range(count))
        assert errors.empty()
    finally:
        sender.close(timeout=2)
        _close_quietly(peer)


@pytest.mark.timeout(20, method="thread")
def test_full_batched_admission_writes_the_backlog_without_the_sender_thread():
    # With TRTLLM_PROXY_BATCH_REQUESTS=1, a caller that finds admission full
    # writes the backlog itself, so a sender thread denied the GIL by a busy
    # event loop does not make admission reject requests.
    commands = _ObservedQueue(is_server=True, send_hwm=_RequestSender.send_hwm(batching=True))
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=True, direct_send=True)
    sender.MAX_PENDING_REQUESTS = 4
    peer = _command_peer(commands)
    resume = _deschedule_owner(sender)
    limit = sender.MAX_PENDING_REQUESTS
    count = 4 * limit
    try:
        for item in range(count):
            sender.send(pickle.dumps(item))
        # Each full backlog is written by the caller as one message, in order.
        for first in range(0, count - limit, limit):
            assert peer.get(timeout=2) == _payloads(*range(first, first + limit))
        assert commands.writers == {threading.current_thread().name}
        assert sender._pending_requests == limit
        # The last backlog waits for the sender thread, which coalesces it.
        resume.set()
        assert peer.get(timeout=2) == _payloads(*range(count - limit, count))
        assert errors.empty()
    finally:
        resume.set()
        sender.close(timeout=2)
        _close_quietly(peer)


@pytest.mark.timeout(15, method="thread")
def test_a_written_command_does_not_wake_the_sender_thread():
    # A send whose command the caller has written does not wake the sender
    # thread. With another CPU-bound thread, each needless wake makes the next
    # callers wait for the lock through GIL handoffs.
    # The socket holds every command, so a slow libzmq I/O thread on a loaded
    # host cannot cause real backpressure, which does need the sender thread.
    commands = IpcQueue(is_server=True, send_hwm=256)
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=False, direct_send=True)
    peer = _command_peer(commands)
    notified = []
    notify_all = sender._cond.notify_all

    def recording_notify_all():
        notified.append(threading.current_thread().name)
        notify_all()

    try:
        sender.send(pickle.dumps("connect"))
        assert peer.get(timeout=2) == "connect"
        sender._cond.notify_all = recording_notify_all
        for item in range(32):
            sender.send(pickle.dumps(item))
        assert [peer.get(timeout=2) for _ in range(32)] == list(range(32))
        assert notified == []
        assert errors.empty()
    finally:
        sender._cond.notify_all = notify_all
        sender.close(timeout=2)
        _close_quietly(peer)


def _commands(message):
    """Return the commands in one message, whether sent alone or as a batch."""
    return (
        [pickle.loads(payload) for payload in message] if isinstance(message, list) else [message]
    )


@pytest.mark.timeout(20, method="thread")
@pytest.mark.parametrize("batching", [True, False])
def test_serializing_a_message_does_not_hold_up_other_callers(batching):
    # The writer pickles and signs a message without holding the lock that
    # every send() takes, so with large prompts other callers do not wait for
    # a whole batch's serialization.
    signing, release = threading.Event(), threading.Event()
    held = {}

    class _SlowSigningQueue(IpcQueue):
        def _sign_data(self, data):
            if threading.current_thread() is held.get("thread"):
                signing.set()
                assert release.wait(10)
            return super()._sign_data(data)

    commands = _SlowSigningQueue(is_server=True, send_hwm=_RequestSender.send_hwm(batching))
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=batching, direct_send=True)
    peer = _command_peer(commands)
    first = threading.Thread(target=sender.send, args=(pickle.dumps(0),), name="first-caller")
    later = threading.Thread(
        target=lambda: [sender.send(pickle.dumps(item)) for item in range(1, 5)]
    )
    try:
        # With batching the sender thread writes; otherwise the first caller.
        held["thread"] = sender._thread if batching else first
        first.start()
        assert signing.wait(5)
        later.start()
        later.join(timeout=2)
        assert not later.is_alive(), "send() waited for another thread's serialization"
        release.set()
        first.join(timeout=5)
        received = []
        while len(received) < 5:
            received += _commands(peer.get(timeout=2))
        assert received == list(range(5))
        assert errors.empty()
    finally:
        release.set()
        for thread in (first, later):
            if thread.is_alive():
                thread.join(timeout=5)
        sender.close(timeout=2)
        _close_quietly(peer)


@pytest.mark.timeout(10, method="thread")
def test_lone_command_is_pickled_once():
    # A lone command goes out as its own signed payload, so it is pickled once
    # on the proxy and unpickled once on rank 0.
    commands = IpcQueue(is_server=True, send_hwm=_RequestSender.send_hwm(batching=False))
    sender = _RequestSender(commands, queue.Queue(), batching=False, direct_send=True)
    peer = _command_peer(commands)
    payload = pickle.dumps({"prompt_token_ids": list(range(64))})
    try:
        sender.send(payload)
        assert peer.socket.poll(2000, zmq.POLLIN)
        # The message is the payload and its 32-byte HMAC, nothing more.
        assert peer.socket.recv()[:-32] == payload
    finally:
        sender.close(timeout=2)
        _close_quietly(peer)


@pytest.mark.timeout(15, method="thread")
def test_sender_thread_is_woken_once_per_queued_burst():
    # send() wakes the sender thread only when the queue turns non-empty; the
    # thread retries queued work on its own after that.
    commands = IpcQueue(is_server=True, send_hwm=_RequestSender.send_hwm(batching=True))
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=True, direct_send=True)
    peer = _command_peer(commands)
    resume = _deschedule_owner(sender)
    wake = sender._owner_wake
    wakes = []

    class _CountingWake:
        def set(self):
            wakes.append(threading.current_thread().name)
            wake.set()

        def __getattr__(self, name):
            return getattr(wake, name)

    sender._owner_wake = _CountingWake()
    try:
        for item in range(10):
            sender.send(pickle.dumps(item))
        assert len(wakes) == 1
        resume.set()
        received = []
        while len(received) < 10:
            received += _commands(peer.get(timeout=2))
        assert received == list(range(10))
        assert errors.empty()
    finally:
        sender._owner_wake = wake
        resume.set()
        sender.close(timeout=2)
        _close_quietly(peer)


@pytest.mark.parametrize("batching", [None, "1"])
def test_unbatched_command_socket_buffers_as_many_commands_as_admission(monkeypatch, batching):
    # Unbatched, every command is one message, so the socket buffers as many
    # as admission queues and a burst does not wait for the sender thread.
    if batching is None:
        monkeypatch.delenv("TRTLLM_PROXY_BATCH_REQUESTS", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_PROXY_BATCH_REQUESTS", batching)
    proxy, _ = _single_frontend_lanes()
    try:
        expected = (
            _RequestSender.BATCHED_SEND_HWM if batching else _RequestSender.MAX_PENDING_REQUESTS
        )
        assert proxy.request_queue.socket.getsockopt(zmq.SNDHWM) == expected
    finally:
        proxy._request_sender.close(timeout=2)
        proxy.result_queue.close()
        proxy.worker_init_status_queue.close()


def test_constructor_defaults_match_the_production_sender(monkeypatch):
    monkeypatch.delenv("TRTLLM_PROXY_BATCH_REQUESTS", raising=False)
    senders = []
    try:
        for create in (_RequestSender, _create_request_sender):
            commands = IpcQueue(is_server=True)
            senders.append(create(commands, queue.Queue()))
        flags = [
            (sender._batching, sender._direct_send, sender._stop_workers_on_failure)
            for sender in senders
        ]
        assert flags == [(False, True, True), (False, True, True)]
    finally:
        for sender in senders:
            sender.close(timeout=2)


@pytest.mark.timeout(15, method="thread")
def test_peer_loss_during_close_ends_final_writes_without_a_failure():
    # When rank 0 disconnects during close, queued commands are dropped at
    # once rather than retried until the close deadline, and no engine failure
    # is reported.
    peer = IpcQueue(is_server=True, socket_type=zmq.PULL, receive_hwm=1)
    peer.socket.setsockopt(zmq.RCVBUF, 256 * 1024)
    commands = IpcQueue(peer.address, is_server=False, socket_type=zmq.PUSH, send_hwm=1)
    commands.socket.setsockopt(zmq.SNDBUF, 256 * 1024)
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=False, direct_send=True)
    data = b"x" * (1024 * 1024)
    try:
        deadline = time.monotonic() + 5
        while not sender._pending:
            assert time.monotonic() < deadline, "the command socket never filled"
            sender.send(pickle.dumps(data))
        sender.close(timeout=7.0, wait=False)
        _close_quietly(peer)
        started = time.monotonic()
        assert sender.close(timeout=7.0) is None
        assert time.monotonic() - started < 3
        assert not sender._thread.is_alive()
        assert errors.empty()
    finally:
        sender.close(timeout=2)
        _close_quietly(peer)


@pytest.mark.timeout(15, method="thread")
def test_failed_shutdown_write_marks_the_engine_dead_without_raising():
    # A stop sentinel that cannot be queued at shutdown marks the engine dead
    # but is not raised from shutdown().
    q = _RecordingQueue(block_first_put=True, direct=True)
    proxy = _bare_proxy()
    proxy._engine_dead = False
    proxy._results = {}
    proxy._error_queue = queue.Queue()
    proxy._request_sender = _RequestSender(q, proxy._error_queue)
    try:
        proxy._request_sender.send(pickle.dumps("in-flight"))
        assert q.first_put_started.wait(3)
        proxy.workers_started = True
        started = time.monotonic()
        # Rank 0 has stopped reading, so the sentinel cannot be queued.
        proxy._close_request_sender(send_sentinel=True)
        assert time.monotonic() - started < 5
        assert proxy._error_queue.empty()
        assert proxy._engine_dead
    finally:
        proxy.workers_started = False
        q.release_first_put.set()
        proxy._request_sender.close(timeout=2)
    assert q.messages == ["in-flight"]


class _PollRecordingSocket:
    """Forward to a ZMQ socket, recording each poll made under a held lock."""

    def __init__(self, socket, lock_holder, polls):
        self._socket = socket
        self._lock_holder = lock_holder
        self._polls = polls

    def __getattr__(self, name):
        return getattr(self._socket, name)

    def poll(self, *args, **kwargs):
        sender = self._lock_holder.get("sender")
        if sender is not None and sender._cond._is_owned():
            self._polls.append(threading.current_thread().name)
        return self._socket.poll(*args, **kwargs)

    def get_monitor_socket(self, **kwargs):
        monitor = self._socket.get_monitor_socket(**kwargs)
        return _PollRecordingSocket(monitor, self._lock_holder, self._polls)


@pytest.mark.timeout(15, method="thread")
def test_sends_and_idle_checks_do_not_poll_under_the_sender_lock():
    # zmq_poll releases the GIL even with a zero timeout, so neither callers
    # nor the idle sender thread poll a socket while holding the sender lock.
    # Otherwise every caller waits for that thread to get the GIL back.
    commands = IpcQueue(is_server=True, send_hwm=_RequestSender.send_hwm(batching=False))
    holder = {}
    polls = []
    commands.socket = _PollRecordingSocket(commands.socket, holder, polls)
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=False, direct_send=True)
    holder["sender"] = sender
    peer = _command_peer(commands)
    checks = _record_owner_checks(sender)
    try:
        for item in range(32):
            sender.send(pickle.dumps(item))
        assert [peer.get(timeout=2) for _ in range(32)] == list(range(32))
        _wait_for_owner_checks(checks, 3)
        assert polls == []
        assert sender.available
        assert errors.empty()
    finally:
        sender.close(timeout=2)
        _close_quietly(peer)


@pytest.mark.timeout(15, method="thread")
def test_sender_thread_stays_off_the_lock_while_callers_check_the_connection():
    # While callers send, each send checks the connection, so the sender
    # thread checks it at most once per IDLE_CHECK_INTERVAL. Its checks take
    # the lock, and with another CPU-bound thread a caller that arrives during
    # one waits for GIL handoffs.
    commands = IpcQueue(is_server=True, send_hwm=_RequestSender.send_hwm(batching=False))
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=False, direct_send=True)
    peer = _command_peer(commands)
    try:
        sender.send(pickle.dumps("connect"))
        assert peer.get(timeout=2) == "connect"
        owner_checks = _record_owner_checks(sender)
        started = time.monotonic()
        count = 0
        while time.monotonic() < started + 0.5:
            sender.send(pickle.dumps(count))
            count += 1
            time.sleep(0.002)
        active = len(owner_checks)
        active_window = time.monotonic() - started
        assert [peer.get(timeout=2) for _ in range(count)] == list(range(count))
        # Checking every POLL_INTERVAL would give about ten checks in 0.5 s.
        assert active <= active_window / sender.IDLE_CHECK_INTERVAL + 1
        # An idle connection is still checked, so its loss is still detected.
        _wait_for_owner_checks(owner_checks, 2)
        assert errors.empty()
    finally:
        sender.close(timeout=2)
        _close_quietly(peer)


def _probe(port, kind):
    if kind in ("tcp", "garbage"):
        with socket.create_connection(("127.0.0.1", port)) as probe:
            if kind == "garbage":
                probe.sendall(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
                time.sleep(0.05)
        return
    socket_type = zmq.PAIR if kind == "zmtp" else zmq.DEALER
    with zmq.Context() as context, context.socket(socket_type) as probe:
        probe.setsockopt(zmq.LINGER, 0)
        probe.connect(f"tcp://127.0.0.1:{port}")
        time.sleep(0.2)


@pytest.mark.timeout(15, method="thread")
@pytest.mark.parametrize("direct_send", [False, True])
@pytest.mark.parametrize("kind", ["tcp", "garbage", "zmtp", "incompatible"])
def test_unrelated_connection_to_command_port_does_not_fail_the_engine(kind, direct_send):
    # A local connect/close on the loopback command port that is not the
    # command peer does not fail a healthy engine.
    commands = IpcQueue(is_server=True, send_hwm=_RequestSender.send_hwm(batching=False))
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=False, direct_send=direct_send)
    peer = _command_peer(commands)
    try:
        sender.send(pickle.dumps("before"))
        assert peer.get(timeout=2) == "before"
        _probe(int(commands.address[0].rsplit(":", 1)[1]), kind)
        # Let the idle sender thread process the probe's connection events.
        time.sleep(3 * sender.IDLE_CHECK_INTERVAL)
        assert errors.empty()
        assert sender.available
        sender.send(pickle.dumps("after"))
        assert peer.get(timeout=2) == "after"
        # Losing the command peer itself remains terminal.
        _close_quietly(peer)
        assert isinstance(errors.get(timeout=2), ConnectionError)
        with pytest.raises(EngineDeadError):
            sender.send(pickle.dumps("after-disconnect"))
    finally:
        sender.close(timeout=2)
        _close_quietly(peer)


@pytest.mark.timeout(15, method="thread")
@pytest.mark.parametrize("direct_send", [False, True])
def test_probe_overlapping_the_worker_handshake_does_not_fail_the_engine(direct_send):
    # A TLS probe produces no handshake failure event. Its close, after the
    # worker's handshake completes, is not the command peer's loss.
    commands = IpcQueue(is_server=True, send_hwm=_RequestSender.send_hwm(batching=False))
    errors = queue.Queue()
    sender = _RequestSender(commands, errors, batching=False, direct_send=direct_send)
    port = int(commands.address[0].rsplit(":", 1)[1])
    probe = socket.create_connection(("127.0.0.1", port))
    peer = None
    try:
        time.sleep(0.2)
        peer = _command_peer(commands)
        sender.send(pickle.dumps("before"))
        assert peer.get(timeout=2) == "before"
        probe.sendall(bytes.fromhex("16030100c4010000c00303") + b"\x00" * 190)
        time.sleep(0.05)
        probe.close()
        time.sleep(0.5)
        assert errors.empty()
        assert sender.available
        sender.send(pickle.dumps("after"))
        assert peer.get(timeout=2) == "after"
        _close_quietly(peer)
        assert isinstance(errors.get(timeout=2), ConnectionError)
    finally:
        probe.close()
        sender.close(timeout=2)
        if peer is not None:
            _close_quietly(peer)


_HTTP_PROBE = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
_TLS_PROBE = bytes.fromhex("16030100c4010000c00303") + b"\x00" * 190


def _single_frontend_lanes():
    proxy = _bare_proxy()
    proxy._num_frontends = 1
    proxy.postproc_config = SimpleNamespace(enabled=False)
    proxy._enable_resource_governor = False
    addresses = proxy._setup_queues()
    return proxy, addresses


@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("probe_open", [True, False])
@pytest.mark.parametrize("payload", [_HTTP_PROBE, _TLS_PROBE], ids=["http", "tls"])
@pytest.mark.parametrize("lane", ["request", "result"])
def test_local_probe_before_the_worker_attaches_does_not_take_a_lane(lane, payload, probe_open):
    # An HTTP/TLS probe or port scan of a loopback lane during model load must
    # not take the PAIR lane, so rank 0, which connects after READY, still
    # receives commands and delivers results.
    proxy, addresses = _single_frontend_lanes()
    address = addresses.request_queue_addr if lane == "request" else addresses.result_queue_addr
    probe = socket.create_connection(("127.0.0.1", int(address[0].rsplit(":", 1)[1])))
    worker = None
    try:
        probe.sendall(payload)
        time.sleep(0.2)
        if not probe_open:
            probe.close()
            time.sleep(0.5)
        # Rank 0 connects after READY.
        worker = IpcQueue(address, is_server=False, name="rank0")
        worker.setup_lazily()
        time.sleep(0.5)
        if lane == "request":
            proxy._request_sender.send(pickle.dumps("first"))
            assert worker.poll(3), "a non-ZMTP probe took the command lane"
            assert worker.get() == "first"
            assert proxy._request_sender.available and proxy._error_queue.empty()
        else:
            worker.put("result")
            lane_queue = proxy.result_queue.queue
            assert lane_queue.poll(3), "a non-ZMTP probe took the result lane"
            assert lane_queue.get() == "result"
    finally:
        probe.close()
        proxy._request_sender.close(timeout=2)
        if worker is not None:
            worker.socket.setsockopt(zmq.LINGER, 0)
            worker.close()
        proxy.result_queue.close()
        proxy.worker_init_status_queue.close()


def test_command_connection_identifies_the_command_peer():
    from tensorrt_llm.executor.request_sender import _CommandConnection

    accepted, connected = zmq.EVENT_ACCEPTED, zmq.EVENT_CONNECTED
    succeeded, disconnected = zmq.EVENT_HANDSHAKE_SUCCEEDED, zmq.EVENT_DISCONNECTED
    failed = zmq.EVENT_HANDSHAKE_FAILED_NO_DETAIL

    def lost(*events):
        connection = _CommandConnection()
        return [connection.observe(event, value) for event, value in events]

    # A rejected connection, then the established peer's own loss.
    assert lost(
        (accepted, 7), (succeeded, 0), (accepted, 8), (disconnected, 8), (disconnected, 7)
    ) == [
        False,
        False,
        False,
        False,
        True,
    ]
    # A second completed handshake is not the command peer.
    assert (
        lost((accepted, 7), (succeeded, 0), (accepted, 8), (succeeded, 0), (disconnected, 8))[-1]
        is False
    )
    # Concurrent handshakes: the failed one is removed and the other remains the peer.
    assert lost(
        (accepted, 7),
        (accepted, 8),
        (succeeded, 0),
        (failed, 104),
        (disconnected, 8),
        (disconnected, 7),
    ) == [False, False, False, False, False, True]
    # Without a failure event, either concurrent candidate may be the peer.
    assert lost((accepted, 7), (accepted, 8), (succeeded, 0), (disconnected, 8))[-1] is True
    # A peer that is still attached shows the disconnected candidate was not the peer.
    connection = _CommandConnection()
    for event, value in ((accepted, 7), (accepted, 8), (succeeded, 0)):
        assert connection.observe(event, value) is False
    assert connection.observe(disconnected, 8, lambda: True) is False
    assert connection.observe(disconnected, 7, lambda: True) is True
    # A connection that disconnects before any handshake carried no commands.
    assert lost((connected, 9), (disconnected, 9)) == [False, False]
    # A completed handshake for an unobserved connection keeps every loss terminal.
    assert lost((succeeded, 0), (disconnected, 5)) == [False, True]
