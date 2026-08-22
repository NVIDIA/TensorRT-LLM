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
"""GPU-free white-box tests for the bounce_v2 reactor and the wait() watchdog.

Round-3 regression tests:
  - R1: a FAILED gather/xfer completion terminates its chunk instead of
    re-registering the just-consumed completion id as an orphan (which would
    never fire again: poller ids report exactly once) — the region must come
    back and the cancel-WANT must not wedge behind a phantom orphan write;
  - R2: queued (never-launched) scatter-backlog jobs of a reclaimed flow are
    purged on cancel-WANT / lease expiry instead of being submitted later
    into final KV addresses that may already be freed (silent corruption);
  - R3: ``BounceTransferStatus.wait()`` re-checks ``future.done()`` before
    trusting a watchdog verdict, and fails a wait against a WEDGED (alive but
    heartbeat-stale) reactor with ``FAIL_REACTOR_STALLED``.

The reactor's mechanism dependencies (batched copy pool, completion poller,
RDMA transfer agent) are replaced by in-process fakes, so no CUDA device and
no compiled transfer-agent binding are needed — only an importable
``tensorrt_llm`` package (for its logger) plus pyzmq. Peers are raw pyzmq
sockets like test_reactor_engine.FakePeer (redefined here so this module
never inherits that module's CUDA skip).
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt_llm", reason="bounce_v2 reactor unit tests import tensorrt_llm")
zmq = pytest.importorskip("zmq")

from tensorrt_llm._torch.disaggregation.bounce_v2 import codec  # noqa: E402
from tensorrt_llm._torch.disaggregation.bounce_v2 import engine as engine_mod  # noqa: E402
from tensorrt_llm._torch.disaggregation.bounce_v2.config import BounceV2Config  # noqa: E402
from tensorrt_llm._torch.disaggregation.bounce_v2.engine import BounceTransferStatus  # noqa: E402
from tensorrt_llm._torch.disaggregation.bounce_v2.plan import SCATTER_RUN_DTYPE  # noqa: E402
from tensorrt_llm._torch.disaggregation.bounce_v2.reactor import (  # noqa: E402
    _KIND_EVENT,
    _KIND_XFER,
    FAIL_GATHER,
    FAIL_REACTOR_STALLED,
    FAIL_WRITE,
    BounceReactor,
    BounceResult,
)
from tensorrt_llm._torch.disaggregation.bounce_v2.scheduler import CreditScheduler  # noqa: E402

# Reactor threads are created inside test bodies and joined by the fixture
# teardown (after pytest-threadleak's end-of-call check) — same rationale as
# the identical marker in test_reactor_engine.py.
pytestmark = pytest.mark.threadleak(enabled=False)

KIB = 1 << 10
MIB = 1 << 20
ARENA = 1 * MIB
GRAN = 4 * KIB
CHUNK = 64 * KIB
#: Fake device addresses: the arena base and the src/dst descriptor bases.
BASE = 0x1000_0000
SRC_BASE = 0x2_0000_0000
DST_BASE = 0x3_0000_0000
#: Bounded NEGATIVE wait: long enough for a 1 ms-tick reactor to have acted.
NEGATIVE_WAIT_S = 1.0


def _cfg(**kw) -> BounceV2Config:
    """Small unit config: 1 MiB arena / 64 KiB chunks, timeouts disabled.

    ``request_timeout_ms=0`` disables the sender sweep AND derives the
    receiver lease/quarantine to 0, so only the explicit paths under test can
    resolve futures / reclaim flows (override per test to enable the lease).
    """
    defaults = dict(
        enabled=True,
        arena_size_bytes=ARENA,
        arena_allocation_granularity_bytes=GRAN,
        max_chunk_size_bytes=CHUNK,
        max_inflight_chunks_per_request=4,
        copy_stream_count=2,
        min_descriptor_count=8,
        max_average_descriptor_size_bytes=16 * KIB,
        request_timeout_ms=0,
    )
    defaults.update(kw)
    return BounceV2Config(**defaults)


def _wait_until(pred, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.005)
    pytest.fail(f"timed out ({timeout_s}s) waiting for {what}")


class FakeClock:
    """Injected monotonic clock for the scheduler; advanced instead of slept."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeCopyPool:
    """BatchedCopyPool stand-in with the full REQUIRED binding surface.

    Completion ids are handed out here but reported through the FakePoller by
    the test. ``busy=True`` makes every launch/sink call return ``BUSY`` (all
    copy streams occupied), which parks receiver scatters in the reactor
    backlog and rolls sender launches back.

    Sender surface (per-request plan handle): ``register_plan`` /
    ``release_plan`` / ``launch_chunk``. ``release_plan`` records EVERY call
    (the exactly-once spy) and is idempotent; a launch on a released handle
    raises ValueError, exactly like the binding's deterministic
    launch-racing-release terminal.

    Receiver surface (raw-runs scatter sink): ``submit_scatter_runs``.
    ``sink_mode`` selects its behavior — "ok" hands out a completion id,
    "reject" returns SCATTER_REJECTED (validation failure), "raise" throws
    like a launch error. Every sink call records a copy of the raw runs blob.
    """

    BUSY = -1
    SCATTER_REJECTED = -2
    max_plan_entries = 1 << 16

    def __init__(self) -> None:
        self._mu = threading.Lock()
        self.busy = False
        self.sink_mode = "ok"
        self.plans: dict[int, tuple] = {}
        self.releases: list[int] = []
        #: Every accepted gather launch: (handle, chunk_idx, staging_base, cid).
        self.launches: list[tuple[int, int, int, int]] = []
        #: Every sink call: (region_base, region_bytes, raw runs bytes, rc).
        self.sink_calls: list[tuple[int, int, bytes, int | None]] = []
        self._next_cid = 1  # disjoint from FakeXferAgent ids (shared poller space)
        self._next_handle = 7_000

    def register_plan(self, srcs, offsets, sizes, starts) -> int:
        with self._mu:
            handle = self._next_handle
            self._next_handle += 1
            self.plans[handle] = (
                np.array(srcs),
                np.array(offsets),
                np.array(sizes),
                np.array(starts),
            )
            return handle

    def release_plan(self, handle) -> None:
        with self._mu:
            self.releases.append(int(handle))
            self.plans.pop(int(handle), None)  # idempotent

    def launch_chunk(self, handle, chunk_idx, staging_base) -> int:
        with self._mu:
            if int(handle) not in self.plans:
                raise ValueError("launch_chunk: unknown plan handle")
            if self.busy:
                return self.BUSY
            cid = self._next_cid
            self._next_cid += 1
            self.launches.append((int(handle), int(chunk_idx), int(staging_base), cid))
            return cid

    def submit_scatter_runs(self, region_base, region_bytes, runs_u8) -> int:
        with self._mu:
            if self.sink_mode == "raise":
                self.sink_calls.append((int(region_base), int(region_bytes), bytes(runs_u8), None))
                raise RuntimeError("injected sink launch failure")
            if self.sink_mode == "reject":
                rc = self.SCATTER_REJECTED
            elif self.busy:
                rc = self.BUSY
            else:
                rc = self._next_cid
                self._next_cid += 1
            self.sink_calls.append((int(region_base), int(region_bytes), bytes(runs_u8), rc))
            return rc

    def accepted(self) -> list[tuple[int, int, int, int]]:
        """Only the gather launches that actually launched (id handed out)."""
        with self._mu:
            return list(self.launches)

    def sink_accepted(self) -> list[tuple[int, int, bytes, int | None]]:
        """Only the sink calls that actually launched (id handed out)."""
        with self._mu:
            return [c for c in self.sink_calls if c[3] is not None and c[3] >= 0]


class FakePoller:
    """CompletionPoller stand-in: reports exactly what the test enqueues.

    Mirrors the binding's exactly-once contract — a delivered row is gone —
    which is precisely what the R1 fixes rely on. Rows must also carry the
    production KIND per row type (the reactor's _KIND_* mirror the binding's
    KIND_* constants): the reactor now maps a FAILED xfer row's kind to the
    failure stage — _KIND_EVENT means the C++ chain died at the gather stage
    (FAIL_GATHER), _KIND_XFER a failed RDMA write (FAIL_WRITE). Copy-pool
    completions (gather/scatter) are event rows; agent write completions are
    xfer rows, exactly like the binding.
    """

    def __init__(self) -> None:
        self._mu = threading.Lock()
        self._rows: list[tuple[int, int, int]] = []

    def complete(self, cid: int, ok: bool, kind: int = _KIND_EVENT) -> None:
        with self._mu:
            self._rows.append((int(cid), kind, 1 if ok else 0))

    def complete_xfer(self, xid: int, ok: bool) -> None:
        """An RDMA-write completion row (KIND_XFER, like registerXfer ids)."""
        self.complete(xid, ok, kind=_KIND_XFER)

    def drain(self, _timeout_ms: int) -> np.ndarray:
        with self._mu:
            rows, self._rows = self._rows, []
        if not rows:
            return np.empty((0, 3), dtype=np.int64)
        return np.asarray(rows, dtype=np.int64)


class FakeXferAgent:
    """NixlTransferAgent stand-in: records posts, ids resolve via FakePoller.

    ``launch_chunk_chained`` mirrors the binding's snapshot semantics:
    allocates the copy id from the pool fake and reserves cid +
    ``RESERVED_DELTA``. One terminal row per chained chunk, under the
    reserved id.
    """

    RESERVED_DELTA = 50_000

    def __init__(self) -> None:
        self._mu = threading.Lock()
        #: Every post: (src_addr, dst_addr, nbytes, peer, returned xfer id).
        self.posts: list[tuple[int, int, int, str, int]] = []
        #: Every chained launch: (handle, chunk_idx, staging_base, dst_ptr,
        #: nbytes, cid, reserved_id).
        self.chained: list[tuple] = []
        self._next_xid = 10_000  # disjoint from FakeCopyPool ids

    def post_transfer_1to1(self, src, dst, nbytes, _src_dev, _dst_dev, peer, _poller) -> int:
        with self._mu:
            xid = self._next_xid
            self._next_xid += 1
            self.posts.append((int(src), int(dst), int(nbytes), peer, xid))
            return xid

    def launch_chunk_chained(
        self, pool, handle, chunk_idx, staging_base, dst_ptr, nbytes, _sd, _dd, _peer, _poller
    ):
        with pool._mu:
            if int(handle) not in pool.plans:
                raise ValueError("launch_chunk_chained: unknown plan handle")
            if pool.busy:
                return (pool.BUSY, -1)
            cid = pool._next_cid
            pool._next_cid += 1
            reserved = cid + self.RESERVED_DELTA
            with self._mu:
                self.chained.append(
                    (
                        int(handle),
                        int(chunk_idx),
                        int(staging_base),
                        int(dst_ptr),
                        int(nbytes),
                        cid,
                        reserved,
                    )
                )
        return (cid, reserved)


class RawPeer:
    """Raw pyzmq peer (same shape as test_reactor_engine.FakePeer).

    A bound ROUTER receives what the reactor sends to the endpoint we
    advertise; a DEALER (routing id = our name) injects control messages into
    the reactor's ROUTER.
    """

    def __init__(self, target_endpoint: str, tag: str = "fake"):
        self.name = f"bv2u_{tag}_{uuid.uuid4().hex[:8]}"
        self._ctx = zmq.Context(io_threads=1)
        self.router = self._ctx.socket(zmq.ROUTER)
        self.router.setsockopt_string(zmq.ROUTING_ID, self.name)
        self.router.setsockopt(zmq.LINGER, 0)
        self.router.bind("tcp://127.0.0.1:*")
        self.endpoint = self.router.getsockopt_string(zmq.LAST_ENDPOINT)
        self.dealer = self._ctx.socket(zmq.DEALER)
        self.dealer.setsockopt_string(zmq.ROUTING_ID, self.name)
        self.dealer.setsockopt(zmq.LINGER, 0)
        self.dealer.connect(target_endpoint)

    def send(self, blob: bytes) -> None:
        self.dealer.send(blob)

    def recv(self, timeout_s: float):
        """One (peer, header, blob) control message, or None after timeout_s."""
        deadline = time.monotonic() + timeout_s
        while True:
            remaining_ms = int(max(deadline - time.monotonic(), 0) * 1000)
            if not self.router.poll(remaining_ms):
                return None
            parts = self.router.recv_multipart(zmq.NOBLOCK)
            if len(parts) < 2:
                continue
            header = codec.decode_header(bytes(parts[1]))
            if header is not None:
                return parts[0].decode(), header, bytes(parts[1])

    def close(self) -> None:
        self.dealer.close(linger=0)
        self.router.close(linger=0)
        self._ctx.term()


# --------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------
@pytest.fixture
def make_reactor():
    """Factory for reactor boxes wired to fakes; teardown joins the reactors."""
    boxes: list = []

    def _make(
        tag: str, *, clock: FakeClock | None = None, poller_cls=FakePoller, **cfg_kw
    ) -> SimpleNamespace:
        cfg = _cfg(**cfg_kw)
        sched_kw = {} if clock is None else {"now_fn": clock}
        sched = CreditScheduler(
            base_addr=BASE,
            arena_size_bytes=cfg.arena_size_bytes,
            arena_allocation_granularity_bytes=cfg.arena_allocation_granularity_bytes,
            max_inflight_chunks_per_request=cfg.max_inflight_chunks_per_request,
            **sched_kw,
        )
        pool, poller, agent = FakeCopyPool(), poller_cls(), FakeXferAgent()
        reactor = BounceReactor(
            self_name=f"bv2u_{tag}_{uuid.uuid4().hex[:8]}",
            config=cfg,
            device_id=0,
            raw_agent=agent,
            arena_base=BASE,
            arena_bytes=cfg.arena_size_bytes,
            scheduler=sched,
            copy_pool=pool,
            poller=poller,
            bind_ip="127.0.0.1",
            set_device_fn=None,
        )
        box = SimpleNamespace(
            reactor=reactor, sched=sched, pool=pool, poller=poller, agent=agent, cfg=cfg
        )
        boxes.append(box)
        return box

    yield _make
    for box in reversed(boxes):
        box.reactor.shutdown()


@pytest.fixture
def raw_peers():
    peers: list[RawPeer] = []

    def _make(target_endpoint: str, tag: str = "fake") -> RawPeer:
        peer = RawPeer(target_endpoint, tag)
        peers.append(peer)
        return peer

    yield _make
    for peer in peers:
        peer.close()


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _submit(box, peer_name: str, n_desc: int, desc_bytes: int = 4 * KIB):
    """Submit one request of ``n_desc`` fake VRAM descriptors.

    Eager gathers launch synchronously on this (the caller's) thread.
    """
    src_ptrs = np.asarray([SRC_BASE + i * desc_bytes for i in range(n_desc)], dtype=np.uint64)
    dst_ptrs = np.asarray([DST_BASE + i * desc_bytes for i in range(n_desc)], dtype=np.uint64)
    sizes = np.full(n_desc, desc_bytes, dtype=np.uint64)
    return box.reactor.submit(src_ptrs, dst_ptrs, sizes, 0, peer_name)


def _recv_want(peer: RawPeer, timeout_s: float = 10.0):
    """Receive one WANT; returns (rid, chunk_sizes)."""
    got = peer.recv(timeout_s)
    assert got is not None, "no WANT reached the peer"
    _, header, blob = got
    assert header.msg_type == codec.BounceMsgType.WANT
    chunk_sizes, _ep = codec.decode_want(blob, header)
    return header.request_id, chunk_sizes


def _assert_cancel(peer: RawPeer, rid: int, timeout_s: float = 10.0) -> None:
    """The next control message at the peer must be the rid's cancel-WANT."""
    got = peer.recv(timeout_s)
    assert got is not None, f"no cancel-WANT for rid={rid} reached the peer"
    _, header, blob = got
    assert header.msg_type == codec.BounceMsgType.WANT
    assert header.request_id == rid
    chunk_sizes, _ep = codec.decode_want(blob, header)
    assert codec.is_cancel_want(chunk_sizes), "expected an empty (cancel) WANT"


def _one_run_data(
    rid: int, region_handle: int, dst_addr: int, piece: int = 4 * KIB, count: int = 4
) -> bytes:
    """One-run DATA: ``count`` pieces of ``piece`` bytes, region offset 0."""
    run = np.zeros(1, dtype=SCATTER_RUN_DTYPE)
    run[0] = (0, dst_addr, piece, piece, piece, count)
    return codec.encode_data(rid, 0, 1, region_handle, run)


def _grant_all(peer: RawPeer, want_rid: int, chunk_sizes, base_addr: int = 0xA000_0000) -> list:
    """Send one batched GRANT covering every announced chunk; returns credits."""
    credits = [
        codec.CreditEntry(
            addr=base_addr + i * CHUNK, length=int(s), dev_id=0, region_handle=100 + i
        )
        for i, s in enumerate(chunk_sizes)
    ]
    peer.send(codec.encode_grant(want_rid, credits))
    return credits


def _park_scatter(box, peer: RawPeer, rid: int, dst_addr: int):
    """WANT -> GRANT -> DATA with every fake copy stream busy.

    The scatter job lands in the reactor backlog. ``box.pool.busy`` must be
    True before the DATA is sent. Returns the granted region offset.
    """
    peer.send(codec.encode_want(rid, [CHUNK], peer.endpoint))
    got = peer.recv(10.0)
    assert got is not None, f"no GRANT for rid={rid}"
    _, header, blob = got
    assert header.msg_type == codec.BounceMsgType.GRANT and header.request_id == rid
    credit = codec.decode_credits(blob, header)[0]
    assert box.pool.busy, "test bug: the pool must be busy before DATA is sent"
    backlog_before = len(box.reactor._scatter_backlog)
    peer.send(_one_run_data(rid, credit.region_handle, dst_addr))
    _wait_until(
        lambda: len(box.reactor._scatter_backlog) == backlog_before + 1,
        timeout_s=10.0,
        what=f"rid={rid} scatter job to park in the backlog",
    )
    assert box.reactor._scattering.get(credit.region_handle) is False  # parked, not orphaned
    return credit.region_handle


# --------------------------------------------------------------------------------------------
# R1: a FAILED completion terminates its chunk instead of orphaning it
# --------------------------------------------------------------------------------------------
class TestFailedCompletionTerminatesChunk:
    def test_failed_gather_releases_region_and_consumed_id_not_reregistered(
        self, make_reactor, raw_peers
    ):
        """R1.1: gather completion ok=False on a GATHERING chunk.

        The failed gather's completion id was already consumed by the drain,
        so re-registering it as orphan_gather would leak the staging region
        forever. The fixed path advances the chunk to GATHERED first: the
        region frees inside the failure, nothing is re-registered.
        """
        box = make_reactor("gfail")
        peer = raw_peers(box.reactor.endpoint, "rx")
        assert box.reactor.add_peer(peer.name, peer.endpoint)
        cap0 = box.sched.free_bytes()

        fut = _submit(box, peer.name, n_desc=16)  # one 64 KiB chunk
        gathers = box.pool.accepted()
        assert len(gathers) == 1, "eager submit should have launched exactly one gather"
        cid = gathers[0][3]
        assert box.sched.local_held_count() == 1

        rid, chunk_sizes = _recv_want(peer)
        assert len(chunk_sizes) == 1

        box.poller.complete(cid, ok=False)
        result = fut.result(timeout=10)
        assert result.ok is False
        assert result.reason == FAIL_GATHER
        # Region released inside the failure path (ordered before the resolve).
        assert box.sched.local_held_count() == 0
        assert box.sched.free_bytes() == cap0
        with box.reactor._req_mu:
            assert cid not in box.reactor._completions, "consumed id re-registered as an orphan"
            assert not box.reactor._orphan_writes
            assert not box.reactor._pending_cancels
        # No write was in flight -> the cancel-WANT goes out immediately.
        _assert_cancel(peer, rid)
        assert box.reactor.alive()

    def test_failed_write_releases_region_and_cancels_immediately(self, make_reactor, raw_peers):
        """R1.2: xfer completion ok=False on a WRITING chunk.

        The consumed xfer id must not be re-registered as orphan_xfer (it
        would never fire again, wedging ``_orphan_writes`` and deferring the
        cancel-WANT forever). With no OTHER write in flight the cancel goes
        out immediately and the region frees now.
        """
        box = make_reactor("xfail")
        peer = raw_peers(box.reactor.endpoint, "rx")
        assert box.reactor.add_peer(peer.name, peer.endpoint)
        cap0 = box.sched.free_bytes()

        fut = _submit(box, peer.name, n_desc=16)  # one 64 KiB chunk
        cid = box.pool.accepted()[0][3]
        rid, chunk_sizes = _recv_want(peer)
        assert len(chunk_sizes) == 1

        box.poller.complete(cid, ok=True)  # GATHERED
        _grant_all(peer, rid, chunk_sizes)
        _wait_until(lambda: len(box.agent.posts) == 1, 10.0, "the RDMA write to be posted")
        xid = box.agent.posts[0][4]

        box.poller.complete_xfer(xid, ok=False)
        result = fut.result(timeout=10)
        assert result.ok is False
        assert result.reason == FAIL_WRITE
        assert box.sched.local_held_count() == 0
        assert box.sched.free_bytes() == cap0
        with box.reactor._req_mu:
            assert xid not in box.reactor._completions, "consumed id re-registered as an orphan"
            assert not box.reactor._orphan_writes
            assert not box.reactor._pending_cancels, "cancel deferred behind a phantom orphan"
        # IMMEDIATE cancel: nothing else was writing when the failure landed.
        _assert_cancel(peer, rid)
        assert box.reactor.alive()

    def test_failed_write_with_other_write_in_flight_defers_only_the_other(
        self, make_reactor, raw_peers
    ):
        """R1.3: two WRITING chunks; chunk A's completion arrives ok=False.

        A's region must free NOW (its id was consumed); only B stays deferred
        as a real orphan write, holding the pending cancel. When B's
        completion later fires, the cancel goes out and B's region frees —
        the pre-existing orphan behavior, preserved.
        """
        box = make_reactor("mixed")
        peer = raw_peers(box.reactor.endpoint, "rx")
        assert box.reactor.add_peer(peer.name, peer.endpoint)
        cap0 = box.sched.free_bytes()

        fut = _submit(box, peer.name, n_desc=32)  # two 64 KiB chunks
        gathers = box.pool.accepted()
        assert len(gathers) == 2, "eager submit should have launched both gathers"
        rid, chunk_sizes = _recv_want(peer)
        assert len(chunk_sizes) == 2

        for g in gathers:
            box.poller.complete(g[3], ok=True)
        credits = _grant_all(peer, rid, chunk_sizes)
        _wait_until(lambda: len(box.agent.posts) == 2, 10.0, "both RDMA writes to be posted")
        # Map posts to chunks by the distinct remote addresses we granted.
        by_addr = {p[1]: p[4] for p in box.agent.posts}
        xid_a = by_addr[credits[0].addr]
        xid_b = by_addr[credits[1].addr]

        box.poller.complete_xfer(xid_a, ok=False)
        result = fut.result(timeout=10)
        assert result.ok is False
        assert result.reason == FAIL_WRITE
        # A released now; only B deferred (old behavior orphaned BOTH: the
        # consumed id A never fired again, so held count stuck at 2).
        _wait_until(lambda: box.sched.local_held_count() == 1, 10.0, "A's region to free")
        with box.reactor._req_mu:
            assert box.reactor._orphan_writes == {rid: 1}
            assert box.reactor._pending_cancels == {rid: peer.name}
            assert xid_a not in box.reactor._completions
        assert peer.recv(NEGATIVE_WAIT_S) is None, "cancel sent while B was still writing"

        box.poller.complete_xfer(xid_b, ok=True)  # the orphan write drains
        _wait_until(lambda: box.sched.local_held_count() == 0, 10.0, "B's region to free")
        assert box.sched.free_bytes() == cap0
        _assert_cancel(peer, rid)
        with box.reactor._req_mu:
            assert not box.reactor._orphan_writes
            assert not box.reactor._pending_cancels
        assert box.reactor.alive()


# --------------------------------------------------------------------------------------------
# R2: scatter backlog purged on cancel-WANT / lease expiry
# --------------------------------------------------------------------------------------------
class TestScatterBacklogPurge:
    def test_cancel_want_purges_queued_scatter_never_submitted(self, make_reactor, raw_peers):
        """R2.1: a cancel-WANT drops the flow's queued scatter job.

        The parked job must vanish from the backlog and ``_scattering``, its
        region frees with the flow, and once the pool frees up the job is
        NEVER submitted (it would scatter into KV addresses the cancel just
        released).
        """
        box = make_reactor("rxcancel")
        peer = raw_peers(box.reactor.endpoint, "tx")
        cap0 = box.sched.free_bytes()
        rid = 11
        box.pool.busy = True
        offset = _park_scatter(box, peer, rid, dst_addr=0x5000_0000)

        peer.send(codec.encode_cancel(rid, peer.endpoint))
        _wait_until(lambda: not box.reactor._scatter_backlog, 10.0, "the backlog purge")
        assert offset not in box.reactor._scattering
        assert box.sched.free_bytes() == cap0, "cancelled flow's region not reclaimed"
        assert box.sched.tracked_flows() == 0

        box.pool.busy = False
        time.sleep(0.3)  # generous grace: a leaked job would submit within a tick
        assert box.pool.sink_accepted() == [], "purged scatter job was still submitted"
        assert box.reactor.alive()

    def test_lease_expiry_purges_queued_scatter_and_quarantines_once(self, make_reactor, raw_peers):
        """R2.2: the lease sweep drops the stale flow's queued scatter job.

        The job is never submitted, and its region is reclaimed (quarantined)
        exactly once through the flow's held set — neither stranded as a busy
        orphan nor double-released.
        """
        clock = FakeClock()
        box = make_reactor("rxlease", clock=clock, receiver_flow_timeout_ms=500, quarantine_ms=200)
        peer = raw_peers(box.reactor.endpoint, "tx")
        cap0 = box.sched.free_bytes()
        rid = 21
        box.pool.busy = True
        offset = _park_scatter(box, peer, rid, dst_addr=0x5000_0000)
        region = box.sched.region_bytes(offset)
        assert region > 0

        clock.advance(10.0)  # way past the 500 ms lease (scheduler clock)
        _wait_until(lambda: not box.reactor._scatter_backlog, 10.0, "the lease-sweep purge")
        assert offset not in box.reactor._scattering
        assert box.sched.tracked_flows() == 0
        # Quarantined exactly once: the block stays allocated until the reap.
        assert box.sched.free_bytes() == cap0 - region

        box.pool.busy = False
        time.sleep(0.3)
        assert box.pool.sink_accepted() == [], "purged scatter job was still submitted"

        clock.advance(1.0)  # past the 200 ms quarantine
        _wait_until(lambda: box.sched.free_bytes() == cap0, 10.0, "the quarantine reap")
        assert box.sched.free_bytes() == cap0  # exactly restored: no double release
        assert box.reactor.alive()

    def test_purge_is_selective_other_flows_job_survives(self, make_reactor, raw_peers):
        """R2.3: cancelling one flow keeps the other flow's backlog job.

        The survivor is submitted exactly once when a stream frees, completes,
        and is ACKed.
        """
        box = make_reactor("rxsel")
        peer = raw_peers(box.reactor.endpoint, "tx")
        cap0 = box.sched.free_bytes()
        box.pool.busy = True
        _park_scatter(box, peer, rid=31, dst_addr=0x5000_0000)
        _park_scatter(box, peer, rid=32, dst_addr=0x6000_0000)

        peer.send(codec.encode_cancel(31, peer.endpoint))
        _wait_until(lambda: len(box.reactor._scatter_backlog) == 1, 10.0, "the selective purge")
        assert box.reactor._scatter_backlog[0].rid == 32

        box.pool.busy = False
        _wait_until(lambda: len(box.pool.sink_accepted()) == 1, 10.0, "the survivor to submit")
        _rbase, _rbytes, blob, cid = box.pool.sink_accepted()[0]
        run = np.frombuffer(blob, dtype=SCATTER_RUN_DTYPE)[0]
        assert int(run["dst_addr"]) == 0x6000_0000, "the WRONG flow's job was submitted"

        box.poller.complete(cid, ok=True)
        got = peer.recv(10.0)
        assert got is not None, "no ACK for the surviving flow's scatter"
        _, header, blob = got
        assert header.msg_type == codec.BounceMsgType.ACK
        assert header.request_id == 32
        _wait_until(lambda: box.sched.free_bytes() == cap0, 10.0, "both regions to reclaim")
        assert box.sched.tracked_flows() == 0
        time.sleep(0.2)
        assert len(box.pool.sink_accepted()) == 1, "the cancelled flow's job was also submitted"
        assert box.reactor.alive()


# --------------------------------------------------------------------------------------------
# R3: wait() done-recheck + stall watchdog (no reactor thread needed)
# --------------------------------------------------------------------------------------------
class _WatchdogReactor:
    """alive()/heartbeat_age_s() stub with optional read side effects.

    The side effects deterministically reproduce the race where the future
    resolves between the wait slice timing out and the watchdog verdict.
    """

    def __init__(self, alive: bool = True, heartbeat_age: float = 0.0) -> None:
        self._alive = alive
        self._age = heartbeat_age
        self.on_alive = None
        self.on_age = None

    def alive(self) -> bool:
        if self.on_alive is not None:
            self.on_alive()
        return self._alive

    def heartbeat_age_s(self) -> float:
        if self.on_age is not None:
            self.on_age()
        return self._age


@pytest.fixture
def fast_wait_slice(monkeypatch):
    """Shrink the 1 s watchdog slice so these tests run in milliseconds."""
    monkeypatch.setattr(engine_mod, "_WAIT_SLICE_S", 0.02)


class TestWaitWatchdog:
    def test_wait_returns_resolved_result_when_reactor_already_dead(self):
        """Spec: a future resolved BEFORE the reactor died reports its result.

        wait() reads the future first, so the watchdog verdict never runs.
        """
        fut: "Future[BounceResult]" = Future()
        fut.set_result(BounceResult(True))
        status = BounceTransferStatus(fut, _WatchdogReactor(alive=False))
        assert status.wait(5_000) is True
        assert status.last_status_str() == "SUCCESS"

    def test_wait_prefers_result_resolved_during_dead_verdict(self, fast_wait_slice):
        """R3.1 regression: the future resolves WHILE the watchdog is deciding.

        The alive() probe's side effect resolves the future just before
        reporting the reactor dead — exactly the race. The fixed wait()
        re-checks ``future.done()`` and returns the real (successful) result
        instead of fabricating FAIL_REACTOR_DEAD.
        """
        fut: "Future[BounceResult]" = Future()
        rx = _WatchdogReactor(alive=False)
        rx.on_alive = lambda: fut.done() or fut.set_result(BounceResult(True))
        status = BounceTransferStatus(fut, rx, stall_limit_s=60.0)
        assert status.wait(5_000) is True, status.last_status_str()
        assert status.last_status_str() == "SUCCESS"

    def test_wait_fails_stalled_reactor_without_killing_it(self, fast_wait_slice):
        """R3.2 regression: alive-but-wedged reactor fails the wait.

        heartbeat_age_s() far beyond the stall limit -> wait() returns False
        with FAIL_REACTOR_STALLED well before the caller deadline; the future
        stays pending and the reactor is not marked dead (only THIS wait
        fails, so the upper layer can fall back).
        """
        fut: "Future[BounceResult]" = Future()
        rx = _WatchdogReactor(alive=True, heartbeat_age=1e9)
        status = BounceTransferStatus(fut, rx, stall_limit_s=0.5)
        t0 = time.monotonic()
        assert status.wait(30_000) is False
        assert time.monotonic() - t0 < 5.0, "stall verdict rode the caller deadline"
        assert status.last_status_str() == FAIL_REACTOR_STALLED
        assert not fut.done(), "the stall verdict must not resolve the shared future"
        assert rx.alive(), "the stall verdict must not kill the reactor"

    def test_wait_prefers_result_resolved_during_stall_verdict(self, fast_wait_slice):
        """R3.3: the done() re-check also outranks the STALLED verdict."""
        fut: "Future[BounceResult]" = Future()
        rx = _WatchdogReactor(alive=True, heartbeat_age=1e9)
        rx.on_age = lambda: fut.done() or fut.set_result(BounceResult(True))
        status = BounceTransferStatus(fut, rx, stall_limit_s=0.5)
        assert status.wait(5_000) is True, status.last_status_str()
        assert status.last_status_str() == "SUCCESS"

    def test_healthy_heartbeat_keeps_waiting_until_resolve(self, fast_wait_slice):
        """A fresh heartbeat must never trip the stall verdict.

        wait() keeps slicing until the future resolves on its own.
        """
        fut: "Future[BounceResult]" = Future()
        rx = _WatchdogReactor(alive=True, heartbeat_age=0.0)
        status = BounceTransferStatus(fut, rx, stall_limit_s=0.5)
        timer = threading.Timer(0.2, fut.set_result, args=(BounceResult(True),))
        timer.start()
        try:
            assert status.wait(10_000) is True, status.last_status_str()
        finally:
            timer.cancel()
        assert status.last_status_str() == "SUCCESS"
