# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pipeline-parallel request-send queue (``pp_req_send_queue.py``).

The queue tests drive ``PPRequestSendQueue`` directly. The broadcaster builds
its queue at ``DEFAULT_PP_REQ_SEND_DEPTH``; the tests below also run deeper
queues by handing the broadcaster one (``_use_depth``). The broadcaster tests run the real
``RequestBroadcaster`` over an in-process stand-in for MPI whose ``isend_object`` pickles the
payload when the send is posted (as ``pkl5`` does) and whose handle completes only once the peer
has received the message (rendezvous, the strict case for coupling and deadlocks), while
``send_object`` returns as soon as the message is queued (eager, as for the small count message).
"""

import contextlib
import pickle
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor import request_utils
from tensorrt_llm._torch.pyexecutor.pp_req_send_queue import (
    DEFAULT_PP_REQ_SEND_DEPTH,
    PP_REQ_SEND_ASYNC_ENV,
    PPRequestSendQueue,
)
from tensorrt_llm._torch.pyexecutor.request_utils import RequestBroadcaster

pytestmark = pytest.mark.cpu_only

TIMEOUT_S = 30.0

# --------------------------------------------------------------------------- depth


def _use_depth(monkeypatch, depth):
    """Broadcasters built from here on use a queue ``depth`` deep (None: the default)."""
    depth = DEFAULT_PP_REQ_SEND_DEPTH if depth is None else depth
    monkeypatch.setattr(request_utils, "PPRequestSendQueue", lambda: PPRequestSendQueue(depth))


def test_default_depth():
    assert DEFAULT_PP_REQ_SEND_DEPTH == 1
    assert PPRequestSendQueue().depth == DEFAULT_PP_REQ_SEND_DEPTH


def test_queue_rejects_non_positive_depth():
    with pytest.raises(ValueError):
        PPRequestSendQueue(0)


# --------------------------------------------------------------------------- queue semantics


class _Handle:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def wait(self):
        self.log.append(("wait", self.name))


def test_queue_waits_oldest_first_and_tracks_counts():
    log = []
    send_queue = PPRequestSendQueue(3)
    for i in range(3):
        assert not send_queue.full()
        send_queue.push(_Handle(i, log))
    assert send_queue.full() and len(send_queue) == 3
    send_queue.wait_oldest()
    send_queue.wait_oldest()
    assert log == [("wait", 0), ("wait", 1)]
    assert len(send_queue) == 1
    assert send_queue.drain() == 1
    assert log[-1] == ("wait", 2)
    assert (send_queue.num_posted, send_queue.num_waited, send_queue.max_in_flight) == (3, 3, 3)
    assert "depth=3: posted 3, waited 3, max in flight 3, in flight 0" in send_queue.summary()


def test_queue_ignores_completed_blocking_sends():
    send_queue = PPRequestSendQueue(1)
    send_queue.push(None)
    assert len(send_queue) == 0 and send_queue.num_posted == 0


def test_queue_refuses_a_push_beyond_depth():
    send_queue = PPRequestSendQueue(2)
    send_queue.push(_Handle(0, []))
    send_queue.push(_Handle(1, []))
    with pytest.raises(RuntimeError, match="in flight"):
        send_queue.push(_Handle(2, []))


def test_drain_is_a_no_op_at_depth_one():
    log = []
    send_queue = PPRequestSendQueue(1)
    send_queue.push(_Handle(0, log))
    assert send_queue.drain() == 0
    assert log == [] and len(send_queue) == 1


# --------------------------------------------------------------------------- broadcaster, scripted


class _ScriptedDist:
    """First PP rank of ``pp_size`` ranks; records every send, wait and NVTX range."""

    def __init__(self, log, pp_size=2):
        self.log = log
        self.rank = 0
        self.world_size = pp_size
        self.pp_size = pp_size
        self.has_pp = True
        self.is_first_pp_rank = True
        self.is_last_pp_rank = False
        self.next_pp_rank = 1
        self.prev_pp_rank = pp_size - 1
        self.num_sends = 0

    def tp_cp_broadcast(self, payloads, root):
        return payloads

    def isend_object(self, obj, dest, tag):
        index = self.num_sends
        self.num_sends += 1
        self.log.append(("isend", index, dest, tag))
        return _Handle(index, self.log)

    def send_object(self, obj, dest, tag):
        index = self.num_sends
        self.num_sends += 1
        self.log.append(("send", index, dest, tag))


@pytest.fixture
def nvtx_log(monkeypatch):
    log = []

    @contextlib.contextmanager
    def recording_range(name, *args, **kwargs):
        log.append(("enter", name))
        try:
            yield
        finally:
            log.append(("exit", name))

    monkeypatch.setattr(request_utils, "nvtx_range", recording_range)
    return log


def _broadcaster(monkeypatch, dist, depth, async_send=True):
    _use_depth(monkeypatch, depth)
    if async_send:
        monkeypatch.setenv(PP_REQ_SEND_ASYNC_ENV, "1")
    else:
        monkeypatch.delenv(PP_REQ_SEND_ASYNC_ENV, raising=False)
    return RequestBroadcaster(dist, SimpleNamespace(pause=contextlib.nullcontext))


def _parent_send_trace(num_payloads, tag):
    """The default one-deep send sequence: wait on the previous send, then post."""
    log = []
    for i in range(num_payloads):
        if i > 0:
            log += [
                ("enter", "wait_prev_send_requests_handler"),
                ("wait", i - 1),
                ("exit", "wait_prev_send_requests_handler"),
            ]
        log += [
            ("enter", "send_requests_to_next_pp"),
            ("isend", i, 1, tag),
            ("exit", "send_requests_to_next_pp"),
        ]
    return log


@pytest.mark.parametrize("depth", [None, 1])
def test_default_depth_matches_the_parent_send_sequence(monkeypatch, nvtx_log, depth):
    dist = _ScriptedDist(nvtx_log)
    broadcaster = _broadcaster(monkeypatch, dist, depth)
    for i in range(5):
        broadcaster._broadcast_requests([f"req{i}"], None)
    sends = [e for e in nvtx_log if e[0] != "enter" or e[1] != "tp_broadcast_requests"]
    sends = [e for e in sends if e != ("exit", "tp_broadcast_requests")]
    assert sends == _parent_send_trace(5, tag=dist.pp_size)
    # Teardown is unchanged too: the last send is never waited on.
    assert broadcaster.drain_pending_sends() == 0
    assert ("wait", 4) not in nvtx_log


def test_blocking_send_tracks_no_handle(monkeypatch, nvtx_log):
    dist = _ScriptedDist(nvtx_log)
    broadcaster = _broadcaster(monkeypatch, dist, depth=4, async_send=False)
    for i in range(3):
        broadcaster._broadcast_requests([f"req{i}"], None)
    assert [e[0] for e in nvtx_log if e[0] in ("send", "isend", "wait")] == ["send"] * 3
    assert len(broadcaster.send_requests_queue) == 0


@pytest.mark.parametrize("depth", [2, 3])
def test_deeper_queue_waits_only_on_the_oldest_when_full(monkeypatch, nvtx_log, depth):
    dist = _ScriptedDist(nvtx_log)
    broadcaster = _broadcaster(monkeypatch, dist, depth)
    num_payloads = 6
    for i in range(num_payloads):
        broadcaster._broadcast_requests([f"req{i}"], None)
    events = [e[:2] for e in nvtx_log if e[0] in ("isend", "wait")]
    expected = []
    for i in range(num_payloads):
        if i >= depth:
            expected.append(("wait", i - depth))
        expected.append(("isend", i))
    assert events == expected
    # Each wait sits inside the same NVTX range as before.
    for index, event in enumerate(nvtx_log):
        if event[0] == "wait":
            assert nvtx_log[index - 1] == ("enter", "wait_prev_send_requests_handler")
            assert nvtx_log[index + 1] == ("exit", "wait_prev_send_requests_handler")
    assert broadcaster.drain_pending_sends() == depth
    waited = [e[1] for e in nvtx_log if e[0] == "wait"]
    assert waited == list(range(num_payloads))
    assert broadcaster.send_requests_queue.max_in_flight == depth


def test_last_pp_rank_sends_nothing(monkeypatch, nvtx_log):
    dist = _ScriptedDist(nvtx_log)
    dist.is_last_pp_rank = True
    dist.is_first_pp_rank = False
    dist.recv_object = Mock(side_effect=lambda src, tag: ([f"req{src}"], None))
    broadcaster = _broadcaster(monkeypatch, dist, depth=4)
    assert broadcaster._broadcast_requests([], None) == (["req1"], None)
    assert dist.num_sends == 0 and len(broadcaster.send_requests_queue) == 0


# --------------------------------------------------------------------------- broadcaster, threaded


class _Message:
    def __init__(self, data):
        self.data = data
        self.received = threading.Event()


class _RendezvousHandle:
    def __init__(self, message):
        # Owns the serialized buffer until the wait, as mpi4py's request does.
        self.message = message

    def wait(self):
        if not self.message.received.wait(TIMEOUT_S):
            raise TimeoutError("request send never received")


class _Wire:
    """Per (source, destination, tag) FIFO channels shared by the rank threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.channels = {}

    def channel(self, src, dst, tag):
        with self.lock:
            return self.channels.setdefault((src, dst, tag), queue.Queue())


class _WireDist:
    """One TP=1 PP rank of ``pp_size`` over ``_Wire``."""

    def __init__(self, wire, rank, pp_size, recv_gate=None):
        self.wire = wire
        self.rank = rank
        self.world_size = pp_size
        self.pp_size = pp_size
        self.has_pp = pp_size > 1
        self.is_first_pp_rank = rank == 0
        self.is_last_pp_rank = rank == pp_size - 1
        self.next_pp_rank = (rank + 1) % pp_size
        self.prev_pp_rank = (rank - 1) % pp_size
        self.recv_gate = recv_gate
        self.num_isends = 0

    def tp_cp_broadcast(self, payloads, root):
        return payloads

    def tp_cp_broadcast_int64(self, values, root):
        return values

    def isend_object(self, obj, dest, tag):
        message = _Message(pickle.dumps(obj))
        self.wire.channel(self.rank, dest, tag).put(message)
        self.num_isends += 1
        return _RendezvousHandle(message)

    def send_object(self, obj, dest, tag):
        self.wire.channel(self.rank, dest, tag).put(_Message(pickle.dumps(obj)))

    def recv_object(self, src, tag):
        if self.recv_gate is not None and tag == self.pp_size:
            assert self.recv_gate.wait(TIMEOUT_S)
        message = self.wire.channel(src, self.rank, tag).get(timeout=TIMEOUT_S)
        message.received.set()
        return pickle.loads(message.data)


def _run_ranks(targets):
    errors = []

    def guard(target):
        try:
            target()
        except BaseException as error:  # noqa: BLE001 - re-raised on the test thread
            errors.append(error)

    threads = [threading.Thread(target=guard, args=(t,), daemon=True) for t in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT_S)
        assert not thread.is_alive(), "rank thread hung"
    if errors:
        raise errors[0]


@pytest.mark.parametrize("depth", [None, 1, 2, 4, 16])
@pytest.mark.parametrize("pp_size", [2, 4])
def test_every_rank_receives_every_payload_in_order(monkeypatch, depth, pp_size):
    """Rank 0 mutates each request list after posting it; later ranks see the posted snapshot."""
    _use_depth(monkeypatch, depth)
    monkeypatch.setenv(PP_REQ_SEND_ASYNC_ENV, "1")
    wire = _Wire()
    num_iterations = 25
    # Iterations with no new request skip the payload on every rank.
    counts = [0 if i % 5 == 4 else 1 + i % 3 for i in range(num_iterations)]
    received = {rank: [] for rank in range(pp_size)}
    broadcasters = {}

    def rank_loop(rank):
        dist = _WireDist(wire, rank, pp_size)
        broadcaster = RequestBroadcaster(dist, SimpleNamespace(pause=contextlib.nullcontext))
        broadcaster._collect_py_objects = lambda new_requests: ("py", len(new_requests))
        broadcasters[rank] = broadcaster
        for iteration, count in enumerate(counts):
            new_requests = [[iteration, k] for k in range(count)] if rank == 0 else []
            requests, py_objects = broadcaster.broadcast(new_requests)
            if rank == 0:
                for request in new_requests:
                    request.append("mutated after send")
            if count:
                assert py_objects == ("py", count)
            received[rank].append([list(r) for r in requests])
        broadcaster.drain_pending_sends()

    _run_ranks([lambda r=r: rank_loop(r) for r in range(pp_size)])

    expected = [[[iteration, k] for k in range(count)] for iteration, count in enumerate(counts)]
    for rank in range(1, pp_size):
        assert received[rank] == expected, f"rank {rank}"
    for rank in range(pp_size - 1):
        send_queue = broadcasters[rank].send_requests_queue
        assert send_queue.max_in_flight <= send_queue.depth
        if send_queue.depth > 1:
            assert len(send_queue) == 0


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_first_rank_runs_ahead_of_a_stalled_next_rank_by_depth_payloads(monkeypatch, depth):
    """The coupling a deeper queue removes: rank 0 posts ``depth`` payloads before it blocks."""
    _use_depth(monkeypatch, depth)
    monkeypatch.setenv(PP_REQ_SEND_ASYNC_ENV, "1")
    wire = _Wire()
    gate = threading.Event()
    num_payloads = 6
    sender = RequestBroadcaster(
        _WireDist(wire, 0, 2), SimpleNamespace(pause=contextlib.nullcontext)
    )
    receiver = RequestBroadcaster(
        _WireDist(wire, 1, 2, recv_gate=gate), SimpleNamespace(pause=contextlib.nullcontext)
    )
    posted = []
    got = []

    def send_loop():
        for i in range(num_payloads):
            sender._broadcast_requests([i], None)
            posted.append(i)
        sender.drain_pending_sends()

    def recv_loop():
        for _ in range(num_payloads):
            got.append(receiver._broadcast_requests([], None)[0])

    threads = [threading.Thread(target=f, daemon=True) for f in (send_loop, recv_loop)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + TIMEOUT_S
    while len(posted) < depth and time.monotonic() < deadline:
        time.sleep(0.01)
    # Give the sender time to get past its cap if the queue did not hold it.
    time.sleep(0.2)
    assert len(posted) == depth
    # The blocked call waits on the oldest send before it posts its own.
    assert sender.dist.num_isends == depth
    gate.set()
    for thread in threads:
        thread.join(TIMEOUT_S)
        assert not thread.is_alive()
    assert posted == list(range(num_payloads))
    assert got == [[i] for i in range(num_payloads)]
