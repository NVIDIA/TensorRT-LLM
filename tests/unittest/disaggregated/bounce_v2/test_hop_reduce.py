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
"""CPU-only tests for the hop-reduction experiment (C++ chain).

Exercises the REAL reactor + scheduler + codec over loopback pyzmq with the
mechanism layer (agent / copy pool / completion poller) replaced by pure-
Python fakes — no CUDA, no compiled binding. Raw pyzmq sockets impersonate
the remote peer (same white-box pattern as the GPU test_reactor_engine.py).

The reactor imports ``tensorrt_llm.logger``; when the full package is not
importable (source tree without the compiled wheel) a minimal stub is
installed so the standalone bounce_v2 loader keeps working.
"""

from __future__ import annotations

import importlib
import logging
import sys
import threading
import time
import types
from collections import deque

import numpy as np
import pytest


def _install_logger_stub() -> None:
    try:
        import tensorrt_llm.logger  # noqa: F401

        return
    except Exception:
        pass
    pkg = types.ModuleType("tensorrt_llm")
    logmod = types.ModuleType("tensorrt_llm.logger")
    logmod.logger = logging.getLogger("bounce_v2.hop_reduce_test")
    pkg.logger = logmod
    sys.modules["tensorrt_llm"] = pkg
    sys.modules["tensorrt_llm.logger"] = logmod


_install_logger_stub()

zmq = pytest.importorskip("zmq")

from conftest import load_bounce_v2  # noqa: E402

load_bounce_v2()
codec = importlib.import_module("bounce_v2.codec")
config_mod = importlib.import_module("bounce_v2.config")
reactor_mod = importlib.import_module("bounce_v2.reactor")
scheduler_mod = importlib.import_module("bounce_v2.scheduler")

# Reactor threads are created inside test bodies and joined by fixture
# teardown, which runs AFTER pytest-threadleak's end-of-call check — same
# rationale as the identical marker in test_reactor_engine.py.
pytestmark = pytest.mark.threadleak(enabled=False)

K_PAGE = 4096
ARENA_BASE = 0x10_0000_0000  # fake device address of arena offset 0
REMOTE_BASE = 0x20_0000_0000  # fake remote credit addresses
DEADLINE_S = 10.0


# --------------------------------------------------------------------------- #
# mechanism fakes
# --------------------------------------------------------------------------- #


class Ids:
    """One id sequence shared by pool/agent fakes.

    Mirrors the real CompletionPoller's single id namespace.
    """

    def __init__(self) -> None:
        self._n = 100
        self._mu = threading.Lock()

    def next(self) -> int:
        with self._mu:
            self._n += 1
            return self._n


class FakePoller:
    """drain()-compatible completion source; tests push rows explicitly."""

    def __init__(self) -> None:
        self._mu = threading.Lock()
        self._rows: deque[tuple[int, int, int]] = deque()

    def push(self, cid: int, kind: int, ok: int) -> None:
        with self._mu:
            self._rows.append((cid, kind, ok))

    def drain(self, timeout_ms: int) -> np.ndarray:
        with self._mu:
            rows = list(self._rows)
            self._rows.clear()
        if not rows:
            return np.empty((0, 3), dtype=np.int64)
        return np.asarray(rows, dtype=np.int64)


class FakePool:
    """submit_copy() returns a fresh completion id (never BUSY unless told)."""

    BUSY = -1

    def __init__(self, ids: Ids) -> None:
        self.max_plan_entries = 1 << 20
        self.calls: list[tuple] = []
        self._ids = ids
        self._mu = threading.Lock()

    def submit_copy(self, srcs, dsts, sizes) -> int:
        cid = self._ids.next()
        with self._mu:
            self.calls.append((cid, np.array(srcs), np.array(dsts), np.array(sizes)))
        return cid


class FakeAgent:
    """Records classic posts and (opt B) chained arms."""

    def __init__(self, ids: Ids, arm_accepts: bool = True) -> None:
        self._ids = ids
        self.arm_accepts = arm_accepts
        self.posts: list[tuple] = []
        self.arms: list[tuple] = []
        self._mu = threading.Lock()

    def register_region(self, *args, **kwargs) -> bool:
        return True

    def post_transfer_1to1(self, src, dst, nbytes, src_dev, dst_dev, peer, poller) -> int:
        xid = self._ids.next()
        with self._mu:
            self.posts.append((xid, src, dst, nbytes, peer))
        return xid

    def post_transfer_1to1_on_event(
        self, copy_id, src, dst, nbytes, src_dev, dst_dev, peer, poller
    ) -> int:
        if not self.arm_accepts:
            return -1
        xid = self._ids.next()
        with self._mu:
            self.arms.append((copy_id, xid, src, dst, nbytes, peer))
        return xid


class LegacyAgent:
    """An agent binding WITHOUT the chain primitive (old wheel)."""

    def __init__(self, ids: Ids) -> None:
        self._ids = ids
        self.posts: list[tuple] = []

    def register_region(self, *args, **kwargs) -> bool:
        return True

    def post_transfer_1to1(self, src, dst, nbytes, src_dev, dst_dev, peer, poller) -> int:
        xid = self._ids.next()
        self.posts.append((xid, src, dst, nbytes, peer))
        return xid


class FakePeer:
    """Raw pyzmq sockets impersonating a remote reactor.

    One ROUTER to receive the reactor's sends, one DEALER to inject
    control messages.
    """

    def __init__(self, ctx: "zmq.Context", name: str) -> None:
        self.name = name
        self.router = ctx.socket(zmq.ROUTER)
        self.router.setsockopt_string(zmq.ROUTING_ID, name)
        self.router.setsockopt(zmq.LINGER, 0)
        self.router.bind("tcp://127.0.0.1:*")
        self.endpoint = self.router.getsockopt_string(zmq.LAST_ENDPOINT)
        self.dealer = None

    def connect(self, ctx: "zmq.Context", target_endpoint: str) -> None:
        self.dealer = ctx.socket(zmq.DEALER)
        self.dealer.setsockopt_string(zmq.ROUTING_ID, self.name)
        self.dealer.setsockopt(zmq.LINGER, 0)
        self.dealer.connect(target_endpoint)

    def send(self, blob: bytes) -> None:
        assert self.dealer is not None, "connect() first"
        self.dealer.send(blob)

    def recv(self, timeout_s: float = DEADLINE_S) -> tuple[str, bytes]:
        if self.router.poll(int(timeout_s * 1000)):
            parts = self.router.recv_multipart(zmq.NOBLOCK)
            return parts[0].decode("utf-8"), bytes(parts[1])
        raise TimeoutError("fake peer: no message from the reactor in time")

    def recv_typed(self, msg_type: int, timeout_s: float = DEADLINE_S) -> tuple:
        """Receive until a message of ``msg_type``; returns (header, blob)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            _, blob = self.recv(timeout_s=max(deadline - time.monotonic(), 0.01))
            header = codec.decode_header(blob)
            assert header is not None, "reactor sent an unparsable message"
            if header.msg_type == msg_type:
                return header, blob
        raise TimeoutError(f"fake peer: no message of type {msg_type} in time")

    def close(self) -> None:
        self.router.close(linger=0)
        if self.dealer is not None:
            self.dealer.close(linger=0)


def make_config(**overrides) -> "config_mod.BounceV2Config":
    base = dict(
        enabled=True,
        arena_size_bytes=1 << 20,  # 256 pages
        arena_allocation_granularity_bytes=K_PAGE,
        max_chunk_size_bytes=K_PAGE,
        max_inflight_chunks_per_request=4,
        copy_stream_count=4,
        request_timeout_ms=0,  # tests wait explicitly; no timeout sweeps
    )
    base.update(overrides)
    return config_mod.BounceV2Config(**base)


class Harness:
    """One reactor with fake mechanisms + one fake remote peer."""

    def __init__(self, cfg, agent=None, name: str = "self") -> None:
        self.ids = Ids()
        self.cfg = cfg
        self.poller = FakePoller()
        self.pool = FakePool(self.ids)
        self.agent = FakeAgent(self.ids) if agent is None else agent
        self.sched = scheduler_mod.CreditScheduler(
            base_addr=ARENA_BASE,
            arena_size_bytes=cfg.arena_size_bytes,
            arena_allocation_granularity_bytes=cfg.arena_allocation_granularity_bytes,
            max_inflight_chunks_per_request=cfg.max_inflight_chunks_per_request,
        )
        self.ctx = zmq.Context(io_threads=1)
        self.reactor = reactor_mod.BounceReactor(
            self_name=name,
            config=cfg,
            device_id=0,
            raw_agent=self.agent,
            arena_base=ARENA_BASE,
            arena_bytes=cfg.arena_size_bytes,
            scheduler=self.sched,
            copy_pool=self.pool,
            poller=self.poller,
            bind_ip="127.0.0.1",
        )
        self.peer = FakePeer(self.ctx, "peerA")
        self.peer.connect(self.ctx, self.reactor.endpoint)

    def close(self) -> None:
        self.reactor.shutdown()
        self.peer.close()
        self.ctx.term()


@pytest.fixture
def harness_factory():
    made: list[Harness] = []

    def make(cfg, agent=None) -> Harness:
        h = Harness(cfg, agent=agent)
        made.append(h)
        return h

    yield make
    for h in made:
        h.close()


def wait_until(cond, timeout_s: float = DEADLINE_S, msg: str = "condition") -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.002)
    raise TimeoutError(f"timed out waiting for {msg}")


# --------------------------------------------------------------------------- #
# Opt B: C++ gather->RDMA chain (TRTLLM_BOUNCE_V2_EXP_CPP_CHAIN)
# --------------------------------------------------------------------------- #

KIND_EVENT = 0
KIND_XFER = 1
N_CHUNKS = 4


def _submit(h: Harness, n_chunks: int = N_CHUNKS):
    """Submit one request of n single-desc chunks.

    Each chunk is exactly one page, strided so the planner cannot coalesce
    them.
    """
    idx = np.arange(n_chunks, dtype=np.uint64)
    src = idx * np.uint64(2 * K_PAGE) + np.uint64(0x30_0000_0000)
    dst = idx * np.uint64(2 * K_PAGE) + np.uint64(0x40_0000_0000)
    sizes = np.full(n_chunks, K_PAGE, dtype=np.uint32)
    return h.reactor.submit(src, dst, sizes, 0, "peerA")


def _grant_all(h: Harness, rid: int, n_chunks: int = N_CHUNKS) -> None:
    credits = [
        codec.CreditEntry(REMOTE_BASE + i * K_PAGE, K_PAGE, 0, i * K_PAGE) for i in range(n_chunks)
    ]
    h.peer.send(codec.encode_grant(rid, credits))


def _ack_all_data(h: Harness, rid: int, n_chunks: int = N_CHUNKS) -> None:
    entries = []
    for _ in range(n_chunks):
        header, _blob = h.peer.recv_typed(codec.BounceMsgType.DATA)
        assert header.request_id == rid
        entries.append((header.chunk_idx, header.region_handle))
    h.peer.send(codec.encode_ack(rid, entries))


def test_cpp_chain_arms_instead_of_classic_post(harness_factory) -> None:
    """Chain enabled + capable agent: every credited chunk is ARMED.

    One completion per chunk; post_transfer_1to1 is never called, and no
    gather completion is ever delivered to Python.
    """
    h = harness_factory(make_config(enable_cpp_chain=True))
    assert h.reactor.add_peer("peerA", h.peer.endpoint)
    fut = _submit(h)
    header, _ = h.peer.recv_typed(codec.BounceMsgType.WANT)
    rid = header.request_id
    _grant_all(h, rid)
    wait_until(lambda: len(h.agent.arms) == N_CHUNKS, msg="all chunks armed")
    assert h.agent.posts == []  # classic path never used
    # Verify each arm carried the right (copy_id, staging addr, remote addr).
    copy_ids = [c[0] for c in h.pool.calls]
    for copy_id, _xid, src, dst, nbytes, peer in h.agent.arms:
        assert copy_id in copy_ids
        assert ARENA_BASE <= src < ARENA_BASE + h.cfg.arena_size_bytes
        assert dst >= REMOTE_BASE and nbytes == K_PAGE and peer == "peerA"
    # The chain resolves each chunk with ONE completion: the reserved xfer id.
    for _copy_id, xid, *_ in h.agent.arms:
        h.poller.push(xid, KIND_XFER, 1)
    _ack_all_data(h, rid)
    result = fut.result(timeout=DEADLINE_S)
    assert result.ok, result.reason
    assert h.agent.posts == []
    stats = h.reactor.stats()
    assert stats.get("tx_chain_armed") == N_CHUNKS
    assert stats.get("tx_post_classic") is None
    assert stats.get("tx_gather_eager") == N_CHUNKS  # launched before credit
    assert stats.get("tx_data_sent") == N_CHUNKS
    assert stats.get("tx_acked_chunks") == N_CHUNKS


def test_cpp_chain_arm_race_falls_back_to_classic(harness_factory) -> None:
    """Arm returns -1 (event already terminal in C++).

    The classic gather-completion -> post path must still complete the
    request.
    """
    h = harness_factory(
        make_config(enable_cpp_chain=True), agent=FakeAgent(Ids(), arm_accepts=False)
    )
    h.agent._ids = h.ids  # share the harness id space
    assert h.reactor.add_peer("peerA", h.peer.endpoint)
    fut = _submit(h)
    header, _ = h.peer.recv_typed(codec.BounceMsgType.WANT)
    rid = header.request_id
    _grant_all(h, rid)
    # Deliver the gather completions Python still owns (arm was refused).
    wait_until(lambda: len(h.pool.calls) == N_CHUNKS, msg="eager gathers")
    for copy_id, *_ in list(h.pool.calls):
        h.poller.push(copy_id, KIND_EVENT, 1)
    wait_until(lambda: len(h.agent.posts) == N_CHUNKS, msg="classic posts")
    for xid, *_ in list(h.agent.posts):
        h.poller.push(xid, KIND_XFER, 1)
    _ack_all_data(h, rid)
    result = fut.result(timeout=DEADLINE_S)
    assert result.ok, result.reason
    stats = h.reactor.stats()
    assert stats.get("tx_chain_arm_race") == N_CHUNKS  # every arm refused
    assert stats.get("tx_post_classic") == N_CHUNKS


def test_cpp_chain_disabled_never_arms(harness_factory) -> None:
    """Flag off: even a chain-capable agent is never asked to arm."""
    h = harness_factory(make_config(enable_cpp_chain=False))
    assert h.reactor.add_peer("peerA", h.peer.endpoint)
    fut = _submit(h)
    header, _ = h.peer.recv_typed(codec.BounceMsgType.WANT)
    rid = header.request_id
    _grant_all(h, rid)
    wait_until(lambda: len(h.pool.calls) == N_CHUNKS, msg="eager gathers")
    for copy_id, *_ in list(h.pool.calls):
        h.poller.push(copy_id, KIND_EVENT, 1)
    wait_until(lambda: len(h.agent.posts) == N_CHUNKS, msg="classic posts")
    assert h.agent.arms == []
    for xid, *_ in list(h.agent.posts):
        h.poller.push(xid, KIND_XFER, 1)
    _ack_all_data(h, rid)
    assert fut.result(timeout=DEADLINE_S).ok
    stats = h.reactor.stats()
    assert stats.get("tx_chain_armed") is None
    assert stats.get("tx_chain_arm_race") is None  # chain disabled: no arm attempts
    assert stats.get("tx_post_classic") == N_CHUNKS


def test_cpp_chain_legacy_binding_falls_back(harness_factory) -> None:
    """Flag on but the binding lacks post_transfer_1to1_on_event (old wheel).

    Warn + classic path, never a crash.
    """
    h = harness_factory(make_config(enable_cpp_chain=True), agent=LegacyAgent(Ids()))
    h.agent._ids = h.ids
    assert h.reactor.add_peer("peerA", h.peer.endpoint)
    fut = _submit(h)
    header, _ = h.peer.recv_typed(codec.BounceMsgType.WANT)
    rid = header.request_id
    _grant_all(h, rid)
    wait_until(lambda: len(h.pool.calls) == N_CHUNKS, msg="eager gathers")
    for copy_id, *_ in list(h.pool.calls):
        h.poller.push(copy_id, KIND_EVENT, 1)
    wait_until(lambda: len(h.agent.posts) == N_CHUNKS, msg="classic posts")
    for xid, *_ in list(h.agent.posts):
        h.poller.push(xid, KIND_XFER, 1)
    _ack_all_data(h, rid)
    assert fut.result(timeout=DEADLINE_S).ok


def test_cpp_chain_gather_failure_reports_fail_gather(harness_factory) -> None:
    """A chain dying at the gather stage must fail with FAIL_GATHER.

    It arrives as (reserved id, KIND_EVENT, 0).
    """
    h = harness_factory(make_config(enable_cpp_chain=True))
    assert h.reactor.add_peer("peerA", h.peer.endpoint)
    fut = _submit(h)
    header, _ = h.peer.recv_typed(codec.BounceMsgType.WANT)
    _grant_all(h, header.request_id)
    wait_until(lambda: len(h.agent.arms) == N_CHUNKS, msg="all chunks armed")
    h.poller.push(h.agent.arms[0][1], KIND_EVENT, 0)  # gather died in C++
    result = fut.result(timeout=DEADLINE_S)
    assert not result.ok
    assert result.reason == reactor_mod.FAIL_GATHER


def test_cpp_chain_write_failure_reports_fail_write(harness_factory) -> None:
    """A chained chunk whose post/write fails must fail with FAIL_WRITE.

    It arrives as (reserved id, KIND_XFER, 0).
    """
    h = harness_factory(make_config(enable_cpp_chain=True))
    assert h.reactor.add_peer("peerA", h.peer.endpoint)
    fut = _submit(h)
    header, _ = h.peer.recv_typed(codec.BounceMsgType.WANT)
    _grant_all(h, header.request_id)
    wait_until(lambda: len(h.agent.arms) == N_CHUNKS, msg="all chunks armed")
    h.poller.push(h.agent.arms[0][1], KIND_XFER, 0)
    result = fut.result(timeout=DEADLINE_S)
    assert not result.ok
    assert result.reason == reactor_mod.FAIL_WRITE
