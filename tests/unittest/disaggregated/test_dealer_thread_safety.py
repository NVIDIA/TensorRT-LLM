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

"""Thread-safety of the native V2 disaggregated KV-transfer control plane.

pyzmq sockets are not thread-safe.  The DEALER sockets that carry
KV_AGENT_RESULT / CANCEL_SESSION / REQUEST_DATA are reached from the executor
thread, the ZMQ listener thread and the KV worker threads, so every send must
be serialized and every send site must funnel through one choke point.

The structural tests at the bottom are the guard rail: they fail if a new send
site reaches the DEALER pool outside ``_send_to_peer()`` (the mistake that
#13075 left behind and #16116 then copied).
"""

import ast
import socket
import struct
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from tensorrt_llm._torch.disaggregation.native.messenger import ZMQMessenger
from tensorrt_llm._torch.disaggregation.native.utils import get_local_ip

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def dynamic_endpoint():
    """An endpoint on a free port, as in test_messenger.py."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
        return f"tcp://{get_local_ip()}:{port}"


def _join_workers(target, n_threads, barrier=None):
    """Run target(i) on n_threads and re-raise anything they hit.

    Without this a thread exception is only printed, and the test would fail
    later on a confusing count mismatch instead of the real cause.
    """
    errors = []

    def runner(i):
        try:
            if barrier is not None:
                barrier.wait(timeout=10)
            target(i)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=runner, args=(i,), daemon=True) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, f"worker thread raised: {errors!r}"


# --------------------------------------------------------------------------- #
# ZMQMessenger.send() must serialize the socket
# --------------------------------------------------------------------------- #
class _OverlapDetectingSocket:
    """Stand-in for a zmq socket that records how many threads are inside a send.

    The barrier makes concurrency deterministic: if sends are serialized only one
    thread ever reaches it, so it times out and breaks; if they are not, all
    threads meet inside send_multipart() and the overlap is recorded.
    """

    closed = False

    def __init__(self, n_threads, barrier_timeout=0.5):
        self.messages = []
        self.inside = 0
        self.max_inside = 0
        self._mutex = threading.Lock()
        self._barrier = threading.Barrier(n_threads)
        self._barrier_timeout = barrier_timeout

    def send_multipart(self, frames):
        with self._mutex:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
        try:
            self._barrier.wait(timeout=self._barrier_timeout)
        except threading.BrokenBarrierError:
            pass  # expected once sends are serialized
        with self._mutex:
            self.messages.append(list(frames))
            self.inside -= 1


def test_send_is_serialized_across_threads(dynamic_endpoint):
    """Two threads must never be inside the same socket's send_multipart().

    Fails on main: ZMQMessenger.send() takes no lock at all (self._lock is only
    acquired by stop()), so all threads meet at the barrier and max_inside == 4.
    """
    messenger = ZMQMessenger("ROUTER", endpoint=dynamic_endpoint)
    real_socket = messenger._socket
    fake = _OverlapDetectingSocket(n_threads=4)
    messenger._socket = fake
    try:
        _join_workers(lambda i: messenger.send([b"tag-%d" % i, b"payload-%d" % i]), n_threads=4)
    finally:
        messenger._socket = real_socket
        messenger.stop()

    assert fake.max_inside == 1, (
        f"{fake.max_inside} threads were inside send_multipart() at once; "
        "concurrent sends interleave frames on the wire"
    )
    assert len(fake.messages) == 4
    for frames in fake.messages:
        assert len(frames) == 2, f"torn message: {frames}"


def test_send_after_stop_is_a_noop(dynamic_endpoint):
    """A send racing shutdown must not touch a closed socket.

    Fails on main: send() calls send_multipart() on the closed socket and pyzmq
    raises (ZMQError/ENOTSOCK), rather than dropping the message.
    """
    messenger = ZMQMessenger("ROUTER", endpoint=dynamic_endpoint)
    messenger.stop()
    messenger.send([b"late"])  # must not raise


def test_stop_does_not_deadlock_when_a_listener_callback_sends(dynamic_endpoint):
    """stop() must not block on a listener callback that replies on the socket.

    RankInfoServer replies on its ROUTER from inside its own listener callback,
    and stop() holds self._lock across _listener_thread.join().  Serializing
    send() on self._lock would wedge the callback until the join times out.
    Passes on main (send takes no lock); this pins the naive fix out.
    """
    router = ZMQMessenger("ROUTER", endpoint=dynamic_endpoint)
    dealer = ZMQMessenger("DEALER", endpoint=dynamic_endpoint)
    in_callback = threading.Event()
    release = threading.Event()
    reply_sent = threading.Event()
    callback_errors = []

    def on_message(messages):
        in_callback.set()
        release.wait(10.0)
        try:
            router.send([messages[0], b"reply"])
            reply_sent.set()
        except Exception as e:
            # The listener swallows callback exceptions and breaks its loop, which
            # would end the join early and make this test pass vacuously.
            callback_errors.append(e)

    router.start_listener(on_message)
    dealer.send([b"ping"])
    assert in_callback.wait(10.0), "listener never ran"

    done = threading.Event()
    threading.Thread(target=lambda: (router.stop(), done.set()), daemon=True).start()
    time.sleep(0.1)  # let stop() reach the join with the callback still in flight
    release.set()

    started = time.monotonic()
    assert done.wait(20.0), "stop() deadlocked against a listener callback send"
    elapsed = time.monotonic() - started
    assert not callback_errors, f"listener callback send raised: {callback_errors!r}"
    assert reply_sent.is_set(), "the listener callback never completed its send"
    dealer.stop()
    # A wedged callback would only be released by the 5 s join timeout.
    assert elapsed < 3.0, f"stop() took {elapsed:.1f}s; the listener callback was blocked"


# --------------------------------------------------------------------------- #
# Concurrent sends must not misframe KV_AGENT_RESULT
# --------------------------------------------------------------------------- #
class _FramingSocket:
    """Models libzmq framing: parts are appended one at a time carrying a MORE flag.

    Concurrent send_multipart() calls therefore interleave into messages whose
    boundaries fall in the wrong places -- exactly what a shared DEALER does.
    """

    closed = False

    def __init__(self, n_threads, barrier_timeout=0.5):
        self.parts = []
        self._mutex = threading.Lock()
        self._barrier = threading.Barrier(n_threads)
        self._barrier_timeout = barrier_timeout

    def send_multipart(self, frames):
        for i, frame in enumerate(frames):
            with self._mutex:
                self.parts.append((frame, i < len(frames) - 1))
            try:
                self._barrier.wait(timeout=self._barrier_timeout)
            except threading.BrokenBarrierError:
                pass  # expected once sends are serialized

    def reassemble(self):
        messages, current = [], []
        for frame, more in self.parts:
            current.append(frame)
            if not more:
                messages.append(current)
                current = []
        return messages


def test_concurrent_kv_results_stay_framed(dynamic_endpoint):
    """Every KV_AGENT_RESULT must arrive as its own well-formed 2-frame message.

    Fails on main: unserialized sends interleave, so reassembly yields messages
    whose second frame is not a _KV_RESULT_PREFIX struct.  The receiver then
    raises struct.error, or unpacks a misaligned status byte and dies in
    _AGENT_RESULT_BY_CODE[status_code] (that dict has only keys 0 and 1) --
    the KV result is lost and the transfer hangs to its timeout.
    """
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")

    messenger = ZMQMessenger("ROUTER", endpoint=dynamic_endpoint)
    real_socket = messenger._socket
    fake = _FramingSocket(n_threads=4)
    messenger._socket = fake

    def worker(rank):
        for slice_id in range(3):
            messenger.send(
                tfr._make_kv_result_msg(
                    rank, 1000 + rank, slice_id, True, tfr.AgentResult.SUCCESS, transfer_size=4096
                )
            )

    try:
        _join_workers(worker, n_threads=4)
    finally:
        messenger._socket = real_socket
        messenger.stop()

    messages = fake.reassemble()
    assert len(messages) == 12, f"expected 12 messages, reassembled {len(messages)}"
    for message in messages:
        assert len(message) == 2, f"torn message with {len(message)} frames: {message}"
        assert message[0] == tfr.MessageType.KV_AGENT_RESULT
        try:
            _rank, _rid, _slice, _last, status_code, _size = tfr._KV_RESULT_PREFIX.unpack(
                message[1]
            )
        except struct.error as e:
            pytest.fail(f"misaligned KV result prefix: {e}")
        assert status_code in tfr._AGENT_RESULT_BY_CODE, (
            f"status byte {status_code} is not a valid AgentResult; "
            "_process_kv_agent_result would raise KeyError and drop the result"
        )


# --------------------------------------------------------------------------- #
# The DEALER pool itself
# --------------------------------------------------------------------------- #
class _CountingMessenger:
    """Counts DEALER constructions and is slow enough to expose a check-then-create."""

    instances = 0
    _mutex = threading.Lock()

    def __init__(self, mode, endpoint=None):
        with _CountingMessenger._mutex:
            _CountingMessenger.instances += 1
        self.mode = mode
        self.endpoint = endpoint
        self.sent = []
        time.sleep(0.05)  # widen the window between the membership test and the store

    def send(self, messages):
        self.sent.append(messages)

    def stop(self):
        pass


@contextmanager
def _pool_owner(tfr, cls_name, shutdown=False):
    """A real Sender/Receiver with __init__ bypassed, seeded with pool state only.

    object.__new__ (not SimpleNamespace) so that every method the path under test
    reaches -- _send_to_peer calls self._get_or_connect_dealer -- still resolves.
    """
    owner = object.__new__(getattr(tfr, cls_name))
    owner._dealers = {}
    owner._dealers_lock = threading.Lock()
    owner._shutdown = shutdown
    try:
        yield owner
    finally:
        # __del__ -> shutdown() would touch attributes __init__ never set.
        owner._shutdown = True


@pytest.mark.parametrize("cls_name", ["Sender", "Receiver"])
def test_dealer_pool_creates_one_dealer_per_endpoint(monkeypatch, cls_name):
    """Concurrent first-sends to one endpoint must build exactly one DEALER.

    Fails on main: the check-then-create in _get_or_connect_dealer is unlocked,
    so every thread constructs a DEALER and all but the last are dropped -- the
    dropped sockets leak a ZMQ context and their queued frames are never sent.
    """
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    monkeypatch.setattr(tfr, "ZMQMessenger", _CountingMessenger)
    _CountingMessenger.instances = 0

    with _pool_owner(tfr, cls_name) as owner:
        results = []
        _join_workers(
            lambda _i: results.append(owner._get_or_connect_dealer("tcp://peer:1234")),
            n_threads=4,
            barrier=threading.Barrier(4),
        )
        assert _CountingMessenger.instances == 1, (
            f"built {_CountingMessenger.instances} DEALERs for one endpoint"
        )
        assert len(owner._dealers) == 1
        assert all(r is owner._dealers["tcp://peer:1234"] for r in results)


@pytest.mark.parametrize("cls_name", ["Sender", "Receiver"])
def test_dealer_pool_is_not_repopulated_after_shutdown(monkeypatch, cls_name):
    """A send arriving after shutdown must not resurrect the pool.

    Fails on main: _get_or_connect_dealer has no _shutdown check, so a late send
    connects a fresh DEALER that nothing will ever stop.
    """
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    monkeypatch.setattr(tfr, "ZMQMessenger", _CountingMessenger)
    _CountingMessenger.instances = 0

    with _pool_owner(tfr, cls_name, shutdown=True) as owner:
        assert owner._get_or_connect_dealer("tcp://peer:1234") is None
        owner._send_to_peer("tcp://peer:1234", [b"CANCEL_SESSION", b"7"])  # must not raise
        assert owner._dealers == {}
        assert _CountingMessenger.instances == 0


@pytest.mark.parametrize("cls_name", ["Sender", "Receiver"])
def test_send_to_peer_rejects_a_missing_endpoint(cls_name):
    """An unregistered peer must still raise rather than silently drop."""
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    with _pool_owner(tfr, cls_name) as owner:
        with pytest.raises(ValueError, match="peer endpoint is None"):
            owner._send_to_peer(None, [b"x"])


# --------------------------------------------------------------------------- #
# Structural guard: one send idiom, one pool
# --------------------------------------------------------------------------- #
_CHOKE_POINT = "_send_to_peer"
_POOL_ACCESSOR = "_get_or_connect_dealer"


def _transfer_source():
    """Source of the transfer module actually under test.

    Read via the imported module's __file__, never a path guessed from this
    test's location: those differ whenever the tests tree and the package come
    from different checkouts, and the guard would then inspect the wrong file.
    """
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    return Path(tfr.__file__).read_text()


def _methods_by_class(tree):
    return {
        cls.name: {fn.name: fn for fn in cls.body if isinstance(fn, ast.FunctionDef)}
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef)
    }


def _enclosing_functions_calling(tree, attr_name):
    """Names of the functions whose body mentions ``self.<attr_name>``."""
    callers = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and node.attr == attr_name:
                if fn.name != attr_name:
                    callers.add(fn.name)
    return callers


@pytest.mark.parametrize("cls_name", ["Sender", "Receiver"])
def test_both_classes_expose_the_send_choke_point(cls_name):
    """Fails on main: neither class has a _send_to_peer()."""
    methods = _methods_by_class(ast.parse(_transfer_source()))
    assert cls_name in methods, f"{cls_name} not found in transfer.py"
    assert _CHOKE_POINT in methods[cls_name], (
        f"{cls_name} must route every peer send through {_CHOKE_POINT}()"
    )


def test_the_dealer_pool_is_only_reached_from_the_choke_point():
    """No send site may touch the DEALER pool directly.

    Fails on main: seven call sites reach _get_or_connect_dealer() from
    _deliver_kv_to_agent, _send_failed_result_to_receiver,
    send_cancel_to_receivers, _get_sender_info, send_cancel_to_senders and
    _request_sender_data, on three different threads.
    """
    callers = _enclosing_functions_calling(ast.parse(_transfer_source()), _POOL_ACCESSOR)
    assert callers <= {_CHOKE_POINT}, (
        f"{sorted(callers - {_CHOKE_POINT})} reach {_POOL_ACCESSOR}() directly; "
        f"go through {_CHOKE_POINT}() so the pool stays thread-safe"
    )


def test_there_is_exactly_one_dealer_pool():
    """A second, thread-local pool is what let the two idioms diverge.

    Fails on main: Sender still carries threading.local() and
    _get_or_connect_thread_dealer(), so a new send site has two idioms to copy
    and only one of them is correct on a worker thread.
    """
    source = _transfer_source()
    assert "_get_or_connect_thread_dealer" not in source, (
        "a second DEALER pool reintroduces the ambiguity that caused this bug"
    )
    assert "threading.local()" not in source, (
        "per-thread DEALER caches cannot be closed by shutdown(); "
        "ZMQMessenger.send() already serializes the shared socket"
    )
