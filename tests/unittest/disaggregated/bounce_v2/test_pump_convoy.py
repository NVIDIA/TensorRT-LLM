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
"""GPU-free tests for the anti-convoy reactor restructure (GB200 round 41).

Pins the new machinery introduced by the no-C++-under-``_req_mu`` rewrite:

  - PUMP ownership (T1): single owner per request (``pump_busy``), work
    handed to a busy owner via ``pump_again``, ownership released ATOMICALLY
    with the ``pump_again`` re-check on the idle exit — a handoff that lands
    while the owner is mid-C++ (or between its decide and its release) must
    never be dropped (a GATHERED+credited chunk has no other retry path);
  - UNROUTED PARKING (T2): a completion row drained BEFORE its route is
    registered parks in ``_unrouted`` and is dispatched inline at
    registration; rows nobody ever claims age out (``_UNROUTED_MAX_AGE_S``)
    and the dict is hard-bounded (``_UNROUTED_MAX``) — never a wedge;
  - FAIL x IN-FLIGHT EXEC (T3): a request failing while a pump's C++ call is
    mid-execution outside the lock (``busy_op`` marked) releases each staging
    region exactly once, resolves the future with the right reason, and
    leaves the reactor alive;
  - PYTHON SCATTER FALLBACK (T5): a copy pool WITHOUT ``submit_scatter_runs``
    keeps the classic validate+expand path — ``submit_copy`` receives the
    exactly-expanded per-piece arrays.

The deterministic windows are held open by a gated fake pool whose
``submit_copy`` blocks on an event (the pump's unlocked EXECUTE phase), plus
a randomized stress run as the broad regression net for the fixed handoff
race. Reuses the fakes/helpers of test_reactor_unit.py (sibling import).
"""

from __future__ import annotations

import random
import threading
import time
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt_llm", reason="bounce_v2 pump/convoy tests import tensorrt_llm")
pytest.importorskip("zmq")

from test_reactor_unit import (  # noqa: E402  (sibling test module)
    BASE,
    CHUNK,
    KIB,
    NEGATIVE_WAIT_S,
    FakeCopyPool,
    FakePoller,
    FakeXferAgent,
    RawPeer,
    _assert_cancel,
    _cfg,
    _grant_all,
    _recv_want,
    _submit,
    _wait_until,
)

from tensorrt_llm._torch.disaggregation.bounce_v2 import codec  # noqa: E402
from tensorrt_llm._torch.disaggregation.bounce_v2 import reactor as reactor_mod  # noqa: E402
from tensorrt_llm._torch.disaggregation.bounce_v2.plan import SCATTER_RUN_DTYPE  # noqa: E402
from tensorrt_llm._torch.disaggregation.bounce_v2.reactor import (  # noqa: E402
    FAIL_PEER_DROPPED,
    BounceReactor,
)
from tensorrt_llm._torch.disaggregation.bounce_v2.scheduler import CreditScheduler  # noqa: E402

# Reactor threads are created inside test bodies and joined by the fixture
# teardown (after pytest-threadleak's end-of-call check) — same rationale as
# the identical marker in test_reactor_unit.py.
pytestmark = pytest.mark.threadleak(enabled=False)


class GatedCopyPool(FakeCopyPool):
    """FakeCopyPool whose submit_copy BLOCKS on a gate.

    Holds the pump's unlocked C++ EXECUTE window open deterministically:
    the completion id is decided (and recorded) BEFORE blocking, so the test
    can deliver its row while the submitting thread is still inside the call
    — exactly the ordering the ``_unrouted`` parking dict exists for.
    """

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.hold = threading.Event()

    def submit_copy(self, srcs, dsts, sizes) -> int:
        cid = super().submit_copy(srcs, dsts, sizes)
        self.last_cid = cid
        self.entered.set()
        assert self.hold.wait(timeout=30), "test bug: the pool gate was never released"
        return cid


class SinkFakeCopyPool(FakeCopyPool):
    """FakeCopyPool + ``submit_scatter_runs``: engages the reactor's C++ sink.

    The reactor detects the capability per instance at construction, so this
    fake pins the sink BRANCH of the receiver path without a GPU. ``mode``
    selects the sink's behavior: "ok" hands out a completion id,
    "busy"/"reject" return the binding's sentinels, "raise" throws like a
    launch error. Every call records a copy of the raw runs blob.
    """

    SCATTER_REJECTED = -2

    def __init__(self) -> None:
        super().__init__()
        self.mode = "ok"
        #: Every sink call: (region_base, region_bytes, raw runs bytes, rc).
        self.sink_calls: list[tuple[int, int, bytes, int | None]] = []

    def submit_scatter_runs(self, region_base, region_bytes, runs_u8) -> int:
        with self._mu:
            if self.mode == "raise":
                self.sink_calls.append((int(region_base), int(region_bytes), bytes(runs_u8), None))
                raise RuntimeError("injected sink launch failure")
            if self.mode == "reject":
                rc = self.SCATTER_REJECTED
            elif self.mode == "busy":
                rc = self.BUSY
            else:
                rc = self._next_cid
                self._next_cid += 1
            self.sink_calls.append((int(region_base), int(region_bytes), bytes(runs_u8), rc))
            return rc

    def sink_accepted(self) -> list[tuple[int, int, bytes, int | None]]:
        """Only the sink calls that actually launched (id handed out)."""
        with self._mu:
            return [c for c in self.sink_calls if c[3] is not None and c[3] >= 0]


class ChainFakePoller(FakePoller):
    """FakePoller + reserve/cancel: engages the reactor's two-phase chain.

    ``reserve_chain`` deterministically maps copy_id -> copy_id + 50_000 and
    never declines; ``cancel_chain`` records calls and returns ``cancel_rc``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cancels: list[int] = []
        self.cancel_rc = 0  # CANCEL_TERMINAL

    def reserve_chain(self, copy_id: int) -> int:
        return int(copy_id) + 50_000

    def cancel_chain(self, reserved_id: int) -> int:
        self.cancels.append(int(reserved_id))
        return self.cancel_rc


class GatedOpAgent(FakeXferAgent):
    """FakeXferAgent gating ONE C++ entry point (fulfill / arm / post).

    The gated method blocks on ``hold`` after setting ``entered``, then
    returns ``release_rc`` (fulfill/arm) or posts classically (post): the
    deterministic mid-EXECUTE window for the busy_op failure tests. Exposing
    ``fulfill_chain_1to1`` only matters when the poller also carries
    reserve/cancel (the reactor requires all three for two-phase mode).
    """

    def __init__(self, gate: str, release_rc: int | None = None) -> None:
        super().__init__()
        self.gate = gate
        self.release_rc = release_rc
        self.entered = threading.Event()
        self.hold = threading.Event()

    def _wait_gate(self) -> None:
        self.entered.set()
        assert self.hold.wait(timeout=30), "test bug: the agent gate was never released"

    def fulfill_chain_1to1(self, reserved_id, *args) -> int:
        if self.gate == "fulfill":
            self._wait_gate()
        return int(self.release_rc)

    def post_transfer_1to1_on_event(self, copy_id, *args) -> int:
        if self.gate == "arm":
            self._wait_gate()
        return int(self.release_rc)

    def post_transfer_1to1(self, *args) -> int:
        if self.gate == "post":
            self._wait_gate()
        return super().post_transfer_1to1(*args)


@pytest.fixture
def make_reactor():
    """Like test_reactor_unit.make_reactor, plus injectable fake instances."""
    boxes: list = []

    def _make(
        tag: str,
        *,
        pool: FakeCopyPool | None = None,
        poller: FakePoller | None = None,
        agent: FakeXferAgent | None = None,
        **cfg_kw,
    ) -> SimpleNamespace:
        cfg = _cfg(**cfg_kw)
        sched = CreditScheduler(
            base_addr=BASE,
            arena_size_bytes=cfg.arena_size_bytes,
            arena_allocation_granularity_bytes=cfg.arena_allocation_granularity_bytes,
            max_inflight_chunks_per_request=cfg.max_inflight_chunks_per_request,
        )
        pool = pool if pool is not None else FakeCopyPool()
        poller = poller if poller is not None else FakePoller()
        agent = agent if agent is not None else FakeXferAgent()
        reactor = BounceReactor(
            self_name=f"bv2pc_{tag}_{uuid.uuid4().hex[:8]}",
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
        # A still-gated fake would wedge the joining pump/reactor thread.
        if isinstance(box.pool, GatedCopyPool):
            box.pool.hold.set()
        if isinstance(box.agent, GatedOpAgent):
            box.agent.hold.set()
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


def _pin_pump_to_submitter(box) -> None:
    """Suppress the reactor tick's ``_drain_pending_posts`` (instance-level).

    The gated-pool tests need the SUBMIT thread to be the pump owner blocked
    inside the C++ window; without this, the reactor's per-tick retry pump
    can win ownership first and block the reactor thread itself in the gate
    (a livelock only a test can create — the real pool never blocks). Grant
    and gather-done handoffs still pump normally; the ungated stress test
    keeps the retry path fully active.
    """
    box.reactor._drain_pending_posts = lambda: False


def _submit_on_thread(box, peer_name: str, n_desc: int):
    """Run reactor.submit on a side thread (it blocks inside a gated pool).

    Returns (thread, result_box); ``result_box.future`` appears once submit
    returns.
    """
    result = SimpleNamespace(future=None)

    def run():
        result.future = _submit(box, peer_name, n_desc=n_desc)

    t = threading.Thread(target=run, name="bv2pc-submitter", daemon=True)
    t.start()
    return t, result


def _ack(peer: RawPeer, rid: int, chunk_idx: int, region_handle: int) -> None:
    peer.send(codec.encode_ack(rid, [(chunk_idx, region_handle)]))


# --------------------------------------------------------------------------------------------
# T1: pump ownership handoff
# --------------------------------------------------------------------------------------------
class TestPumpHandoff:
    def test_grant_during_blocked_launch_hands_work_to_owner(self, make_reactor, raw_peers):
        """T1a: work handed over mid-C++ reaches the blocked pump owner.

        The submit thread owns the pump and is blocked inside the pool's
        submit (the unlocked EXECUTE window). A GRANT lands meanwhile: the
        reactor's pump must NOT steal ownership — it sets ``pump_again`` —
        and the owner, once unblocked, must consume the handoff (attach the
        credit) instead of dropping it; the transfer then completes.
        """
        pool = GatedCopyPool()
        box = make_reactor("handoff", pool=pool)
        peer = raw_peers(box.reactor.endpoint, "rx")
        assert box.reactor.add_peer(peer.name, peer.endpoint)
        _pin_pump_to_submitter(box)

        t, result = _submit_on_thread(box, peer.name, n_desc=16)  # one 64 KiB chunk
        assert pool.entered.wait(timeout=10), "eager launch never reached the pool"
        rid, chunk_sizes = _recv_want(peer)
        assert len(chunk_sizes) == 1
        req = box.reactor._requests[rid]
        assert req.pump_busy, "the blocked submit thread must own the pump"

        credits = _grant_all(peer, rid, chunk_sizes)
        _wait_until(
            lambda: req.pump_again,
            timeout_s=10.0,
            what="the reactor's pump to hand the GRANT to the busy owner",
        )
        pool.hold.set()
        t.join(timeout=10)
        assert not t.is_alive(), "the pump owner never came back from the gated launch"
        # The owner consumed the handoff: the credit is attached, nothing lost.
        _wait_until(
            lambda: req.posted and req.posted[0].has_credit and not req.pump_busy,
            timeout_s=10.0,
            what="the owner to attach the handed-over credit and drain the pump",
        )
        assert req.pump_again is False

        # Drive the transfer home: gather done -> classic post -> write done
        # -> DATA -> ACK -> SUCCESS.
        box.poller.complete(pool.last_cid, ok=True)
        _wait_until(lambda: len(box.agent.posts) == 1, 10.0, "the RDMA write to be posted")
        box.poller.complete_xfer(box.agent.posts[0][4], ok=True)
        got = peer.recv(10.0)
        assert got is not None and got[1].msg_type == codec.BounceMsgType.DATA
        _ack(peer, rid, 0, credits[0].region_handle)
        assert result.future.result(timeout=10).ok is True
        assert box.reactor.alive()

    def test_concurrent_submit_stress_no_lost_wakeup(self, make_reactor, raw_peers):
        """T1b: randomized handoff stress — no future may ever stall.

        8 threads x 4 requests x 2 chunks against one auto-granting/acking
        raw peer, with gather/write completions delivered by a completer
        thread under randomized micro-delays: pumps race on every request
        from submit threads, the reactor's grant/gather handlers, and
        _drain_pending_posts. A lost pump wakeup (the fixed atomic-release
        race) strands a GATHERED+credited chunk; the tight request timeout
        then turns the stall into a hard, fast failure below.
        """
        box = make_reactor("stress", request_timeout_ms=5000)
        peer = raw_peers(box.reactor.endpoint, "rx")
        assert box.reactor.add_peer(peer.name, peer.endpoint)
        capacity = box.sched.arena_capacity
        stop = threading.Event()
        next_handle = [1000]

        def responder():
            """GRANT every WANT in full; ACK every DATA immediately."""
            while not stop.is_set():
                got = peer.recv(0.05)
                if got is None:
                    continue
                _, header, blob = got
                if header.msg_type == codec.BounceMsgType.WANT:
                    sizes, _ep = codec.decode_want(blob, header)
                    if codec.is_cancel_want(sizes):
                        continue
                    credits = []
                    for s in sizes:
                        h = next_handle[0]
                        next_handle[0] += 1
                        credits.append(
                            codec.CreditEntry(
                                addr=0xA000_0000 + h * CHUNK,
                                length=int(s),
                                dev_id=0,
                                region_handle=h,
                            )
                        )
                    peer.send(codec.encode_grant(header.request_id, credits))
                elif header.msg_type == codec.BounceMsgType.DATA:
                    _ack(peer, header.request_id, header.chunk_idx, header.region_handle)

        def completer():
            """Complete every accepted gather / posted write, jittered."""
            done: set[int] = set()
            rng = random.Random(41)
            while not stop.is_set():
                rows = [(c[3], False) for c in box.pool.accepted()]
                rows += [(p[4], True) for p in box.agent.posts]
                pending = [(i, x) for i, x in rows if i not in done]
                rng.shuffle(pending)
                for cid, is_xfer in pending:
                    done.add(cid)
                    if rng.random() < 0.5:
                        time.sleep(rng.uniform(0, 0.002))
                    if is_xfer:
                        box.poller.complete_xfer(cid, ok=True)
                    else:
                        box.poller.complete(cid, ok=True)
                time.sleep(0.0005)

        threads = [
            threading.Thread(target=responder, daemon=True),
            threading.Thread(target=completer, daemon=True),
        ]
        for th in threads:
            th.start()
        try:
            futures = []
            fut_mu = threading.Lock()

            def submitter(seed: int):
                rng = random.Random(seed)
                for _ in range(4):
                    fut = _submit(box, peer.name, n_desc=32)  # two 64 KiB chunks
                    with fut_mu:
                        futures.append(fut)
                    time.sleep(rng.uniform(0, 0.003))

            subs = [threading.Thread(target=submitter, args=(s,), daemon=True) for s in range(8)]
            for th in subs:
                th.start()
            for th in subs:
                th.join(timeout=30)
                assert not th.is_alive(), "a submit thread wedged (pump never returned)"
            assert len(futures) == 32
            for i, fut in enumerate(futures):
                res = fut.result(timeout=10)
                assert res.ok is True, f"request {i} stalled/failed: {res.reason}"
        finally:
            stop.set()
            for th in threads:
                th.join(timeout=10)
        _wait_until(
            lambda: box.sched.free_bytes() == capacity and box.sched.local_held_count() == 0,
            timeout_s=10.0,
            what="the arena to drain back to capacity after the stress",
        )
        assert box.reactor.alive()
        with box.reactor._req_mu:
            assert not box.reactor._requests
            assert not box.reactor._completions


# --------------------------------------------------------------------------------------------
# T2: unrouted-completion parking dict
# --------------------------------------------------------------------------------------------
class TestUnroutedParking:
    def test_row_before_route_parks_and_dispatches_at_registration(self, make_reactor, raw_peers):
        """T2a: a pre-route row parks, then dispatches at registration.

        The gated pool decides the completion id, then blocks BEFORE
        returning it to the pump: the reactor drains the row with no route
        (parks it), and only then does the pump's record phase register the
        route — which must claim the parked row and advance the chunk to
        GATHERED.
        """
        pool = GatedCopyPool()
        box = make_reactor("park", pool=pool)
        peer = raw_peers(box.reactor.endpoint, "rx")
        assert box.reactor.add_peer(peer.name, peer.endpoint)
        _pin_pump_to_submitter(box)

        t, result = _submit_on_thread(box, peer.name, n_desc=16)
        assert pool.entered.wait(timeout=10)
        cid = pool.last_cid
        rid, chunk_sizes = _recv_want(peer)

        # Deliver the gather completion while the launch is still mid-return.
        box.poller.complete(cid, ok=True)
        _wait_until(
            lambda: cid in box.reactor._unrouted,
            timeout_s=10.0,
            what="the routeless completion row to be parked",
        )
        pool.hold.set()
        t.join(timeout=10)
        assert not t.is_alive()
        req = box.reactor._requests[rid]
        _wait_until(
            lambda: req.posted and req.posted[0].state.name == "GATHERED",
            timeout_s=10.0,
            what="the parked row to be dispatched at route registration",
        )
        assert cid not in box.reactor._unrouted

        # The request stays fully serviceable: credit -> post -> ACK -> ok.
        credits = _grant_all(peer, rid, chunk_sizes)
        _wait_until(lambda: len(box.agent.posts) == 1, 10.0, "the RDMA write to be posted")
        box.poller.complete_xfer(box.agent.posts[0][4], ok=True)
        got = peer.recv(10.0)
        assert got is not None and got[1].msg_type == codec.BounceMsgType.DATA
        _ack(peer, rid, 0, credits[0].region_handle)
        assert result.future.result(timeout=10).ok is True

    def test_unclaimed_rows_age_out_without_wedging(self, make_reactor, monkeypatch):
        """T2b: parked rows nobody claims are reaped after the age limit.

        Rows of dead/cancelled requests must neither wedge the reactor nor
        linger forever — the per-tick reaper drops them (warning path) and
        the reactor stays alive.
        """
        monkeypatch.setattr(reactor_mod, "_UNROUTED_MAX_AGE_S", 0.05)
        box = make_reactor("reap")
        for cid in (424242, 424243, 424244):  # never-routed ids
            box.poller.complete(cid, ok=True)
        _wait_until(
            lambda: len(box.reactor._unrouted) == 3,
            timeout_s=10.0,
            what="the orphan rows to be parked",
        )
        _wait_until(
            lambda: not box.reactor._unrouted,
            timeout_s=10.0,
            what="the parked rows to age out",
        )
        assert box.reactor.alive()

    def test_parking_dict_is_hard_bounded(self, make_reactor, monkeypatch):
        """T2c: flooding with orphan rows never grows the dict past the cap.

        With the cap shrunk to 16, 64 routeless rows must leave exactly 16
        parked (oldest evicted with a warning) and the reactor alive.
        """
        monkeypatch.setattr(reactor_mod, "_UNROUTED_MAX", 16)
        box = make_reactor("bound")
        for cid in range(500_000, 500_064):
            box.poller.complete(cid, ok=True)
        _wait_until(
            lambda: len(box.reactor._unrouted) == 16,
            timeout_s=10.0,
            what="the flood to be parked at the cap",
        )
        time.sleep(0.1)  # more ticks: the bound must hold, not oscillate
        assert len(box.reactor._unrouted) == 16
        assert box.reactor.alive()
        # The newest 16 survived (eviction drops the oldest first).
        assert set(box.reactor._unrouted) == set(range(500_048, 500_064))


# --------------------------------------------------------------------------------------------
# T3: request failure while a pump C++ call is mid-execution
# --------------------------------------------------------------------------------------------
class TestFailDuringBusyExec:
    def test_forget_peer_during_blocked_launch_releases_exactly_once(self, make_reactor, raw_peers):
        """T3: fail with busy_op == "launch" held open -> exactly-once release.

        forget_peer fails the request while the pump is inside the pool
        submit: the fail path must SKIP the busy chunk (no ids yet), the
        pump's record phase must route the fresh id as an orphaned gather,
        and its completion must release the staging region exactly once —
        arena bytes restored to capacity, never past it, reactor alive.
        """
        pool = GatedCopyPool()
        box = make_reactor("failbusy", pool=pool)
        peer = raw_peers(box.reactor.endpoint, "rx")
        assert box.reactor.add_peer(peer.name, peer.endpoint)
        _pin_pump_to_submitter(box)
        capacity = box.sched.arena_capacity

        t, result = _submit_on_thread(box, peer.name, n_desc=16)
        assert pool.entered.wait(timeout=10)
        rid, _chunk_sizes = _recv_want(peer)

        box.reactor.forget_peer(peer.name)
        # The request dies while the launch is STILL blocked (the fail path
        # must not wait on the busy chunk)...
        _wait_until(
            lambda: rid not in box.reactor._requests,
            timeout_s=10.0,
            what="the fail path to run around the busy launch",
        )
        # ...but the staging region stays held: only the pump's record phase
        # (or its orphan completion) may release it.
        assert box.sched.local_held_count() == 1
        assert box.sched.free_bytes() == capacity - CHUNK  # exactly the staged block

        pool.hold.set()
        t.join(timeout=10)
        assert not t.is_alive()
        res = result.future.result(timeout=10)
        assert res.ok is False
        assert res.reason == FAIL_PEER_DROPPED
        # The record phase routed the never-routed id as an orphaned gather;
        # its completion releases the region EXACTLY once.
        cid = pool.last_cid
        _wait_until(
            lambda: cid in box.reactor._completions,
            timeout_s=10.0,
            what="the record phase to register the orphan route",
        )
        assert box.sched.local_held_count() == 1  # still deferred until the row
        box.poller.complete(cid, ok=True)
        _wait_until(
            lambda: box.sched.free_bytes() == capacity and box.sched.local_held_count() == 0,
            timeout_s=10.0,
            what="the orphaned gather completion to release the region",
        )
        time.sleep(0.1)  # no late double-release may push past capacity
        assert box.sched.free_bytes() == capacity
        assert box.reactor.alive()


# --------------------------------------------------------------------------------------------
# T5: receiver scatter — Python fallback when the pool lacks submit_scatter_runs
# --------------------------------------------------------------------------------------------
class TestScatterPythonFallback:
    def test_fallback_expands_runs_exactly_and_acks(self, make_reactor, raw_peers):
        """T5: no ``submit_scatter_runs`` -> classic validate+expand path.

        The fake pool lacks the C++ sink, so the reactor must expand the wire
        runs itself and call ``submit_copy`` with the exact per-piece arrays;
        the chunk then ACKs and the region recycles.
        """
        box = make_reactor("pyscatter")
        assert box.reactor._submit_scatter_fn is None  # fallback engaged
        peer = raw_peers(box.reactor.endpoint, "tx")
        cap0 = box.sched.free_bytes()

        rid = 5
        peer.send(codec.encode_want(rid, [CHUNK], peer.endpoint))
        got = peer.recv(10.0)
        assert got is not None and got[1].msg_type == codec.BounceMsgType.GRANT
        credit = codec.decode_credits(got[2], got[1])[0]
        region_base = BASE + credit.region_handle

        runs = np.zeros(2, dtype=SCATTER_RUN_DTYPE)
        # (bounce_offset, dst_addr, dst_stride, bounce_stride, piece_size, count)
        runs[0] = (0, 0x5000_0000, 8 * KIB, 4 * KIB, 4 * KIB, 3)
        runs[1] = (32 * KIB, 0x6000_0000, 2 * KIB, 2 * KIB, 2 * KIB, 2)
        peer.send(codec.encode_data(rid, 0, 1, credit.region_handle, runs))

        _wait_until(lambda: len(box.pool.accepted()) == 1, 10.0, "the expanded scatter submit")
        srcs, dsts, sizes, cid = box.pool.accepted()[0]
        exp_srcs = [region_base + i * 4 * KIB for i in range(3)]
        exp_srcs += [region_base + 32 * KIB + i * 2 * KIB for i in range(2)]
        exp_dsts = [0x5000_0000 + i * 8 * KIB for i in range(3)]
        exp_dsts += [0x6000_0000 + i * 2 * KIB for i in range(2)]
        assert srcs.tolist() == exp_srcs, "fallback expanded wrong source addresses"
        assert dsts.tolist() == exp_dsts, "fallback expanded wrong destination addresses"
        assert sizes.tolist() == [4 * KIB] * 3 + [2 * KIB] * 2

        box.poller.complete(cid, ok=True)
        got = peer.recv(10.0)
        assert got is not None and got[1].msg_type == codec.BounceMsgType.ACK
        assert got[1].request_id == rid
        _wait_until(lambda: box.sched.free_bytes() == cap0, 10.0, "the region to recycle")
        assert box.sched.tracked_flows() == 0
        assert box.reactor.alive()


# --------------------------------------------------------------------------------------------
# T6: receiver scatter — the C++-sink BRANCH, driven by a sink-capable fake
# --------------------------------------------------------------------------------------------
def _two_run_data(peer: RawPeer, rid: int, region_handle: int) -> np.ndarray:
    """Send a 2-run DATA (5 pieces) for the granted region; returns the runs."""
    runs = np.zeros(2, dtype=SCATTER_RUN_DTYPE)
    # (bounce_offset, dst_addr, dst_stride, bounce_stride, piece_size, count)
    runs[0] = (0, 0x5000_0000, 8 * KIB, 4 * KIB, 4 * KIB, 3)
    runs[1] = (32 * KIB, 0x6000_0000, 2 * KIB, 2 * KIB, 2 * KIB, 2)
    peer.send(codec.encode_data(rid, 0, 1, region_handle, runs))
    return runs


def _grant_one(peer: RawPeer, rid: int):
    """WANT one CHUNK-sized region and return the credit granted for it."""
    peer.send(codec.encode_want(rid, [CHUNK], peer.endpoint))
    got = peer.recv(10.0)
    assert got is not None and got[1].msg_type == codec.BounceMsgType.GRANT
    return codec.decode_credits(got[2], got[1])[0]


class TestScatterSinkFake:
    def test_sink_path_taken_raw_runs_and_acks(self, make_reactor, raw_peers):
        """T6a: a sink-capable pool routes DATA through submit_scatter_runs.

        The sink receives the granted region's base/bytes and the EXACT raw
        wire blob; the Python expansion path is NOT taken (submit_copy never
        called); the scatter completes, ACKs, and the region recycles.
        """
        pool = SinkFakeCopyPool()
        box = make_reactor("sinkok", pool=pool)
        assert box.reactor._submit_scatter_fn is not None  # capability detected
        peer = raw_peers(box.reactor.endpoint, "tx")
        cap0 = box.sched.free_bytes()

        credit = _grant_one(peer, rid=71)
        runs = _two_run_data(peer, 71, credit.region_handle)
        _wait_until(lambda: len(pool.sink_accepted()) == 1, 10.0, "the sink submit")
        region_base, region_bytes, blob, rc = pool.sink_accepted()[0]
        assert region_base == BASE + credit.region_handle
        assert region_bytes == box.sched.region_bytes(credit.region_handle) == CHUNK
        assert blob == runs.tobytes(), "the sink did not receive the exact raw wire runs"
        assert pool.calls == [], "the Python expansion path ran despite the sink"
        assert box.reactor.stats().get("rx_data", 0) == 1

        box.poller.complete(rc, ok=True)
        got = peer.recv(10.0)
        assert got is not None and got[1].msg_type == codec.BounceMsgType.ACK
        assert got[1].request_id == 71
        _wait_until(lambda: box.sched.free_bytes() == cap0, 10.0, "the region to recycle")
        assert not box.reactor._rx_flows
        assert box.reactor.alive()

    def test_sink_rejected_no_ack_region_released(self, make_reactor, raw_peers):
        """T6b: SCATTER_REJECTED(-2) is the no-ACK terminal.

        The region is released (the sender must time out, never believe the
        data landed), the flow accounting is decremented, rx_data is NOT
        bumped, and the reactor stays alive.
        """
        pool = SinkFakeCopyPool()
        pool.mode = "reject"
        box = make_reactor("sinkrej", pool=pool)
        peer = raw_peers(box.reactor.endpoint, "tx")
        cap0 = box.sched.free_bytes()

        credit = _grant_one(peer, rid=72)
        _two_run_data(peer, 72, credit.region_handle)
        _wait_until(lambda: len(pool.sink_calls) == 1, 10.0, "the sink validation call")
        _wait_until(lambda: box.sched.free_bytes() == cap0, 10.0, "the rejected region release")
        assert not box.reactor._rx_flows, "flow accounting leaked for the rejected chunk"
        assert box.reactor.stats().get("rx_data", 0) == 0, "rejected DATA counted as accepted"
        assert peer.recv(NEGATIVE_WAIT_S) is None, "a validation-rejected scatter was ACKed"
        assert pool.calls == []
        assert box.reactor.alive()

    def test_sink_busy_backlogs_raw_runs_then_retries(self, make_reactor, raw_peers):
        """T6c: BUSY(-1) parks the job WITH its raw runs.

        The retry resubmits the IDENTICAL blob once a context frees, then
        the scatter completes and ACKs.
        """
        pool = SinkFakeCopyPool()
        pool.mode = "busy"
        box = make_reactor("sinkbusy", pool=pool)
        peer = raw_peers(box.reactor.endpoint, "tx")
        cap0 = box.sched.free_bytes()

        credit = _grant_one(peer, rid=73)
        runs = _two_run_data(peer, 73, credit.region_handle)
        _wait_until(lambda: len(box.reactor._scatter_backlog) == 1, 10.0, "the BUSY backlog")
        assert box.reactor._scatter_backlog[0].runs is not None, "backlog job lost its raw runs"

        pool.mode = "ok"
        _wait_until(lambda: len(pool.sink_accepted()) == 1, 10.0, "the backlog retry submit")
        _region_base, _region_bytes, blob, rc = pool.sink_accepted()[0]
        assert blob == runs.tobytes(), "the retry did not carry the identical raw runs"
        assert not box.reactor._scatter_backlog

        box.poller.complete(rc, ok=True)
        got = peer.recv(10.0)
        assert got is not None and got[1].msg_type == codec.BounceMsgType.ACK
        _wait_until(lambda: box.sched.free_bytes() == cap0, 10.0, "the region to recycle")
        assert box.reactor.alive()

    def test_sink_raise_is_failed_scatter_no_crash(self, make_reactor, raw_peers):
        """T6d: a RuntimeError from the sink takes _finish_scatter(ok=False).

        No ACK, region released, flow accounting settled, reactor alive.
        """
        pool = SinkFakeCopyPool()
        pool.mode = "raise"
        box = make_reactor("sinkraise", pool=pool)
        peer = raw_peers(box.reactor.endpoint, "tx")
        cap0 = box.sched.free_bytes()

        credit = _grant_one(peer, rid=74)
        _two_run_data(peer, 74, credit.region_handle)
        _wait_until(lambda: box.sched.free_bytes() == cap0, 10.0, "the failed-launch release")
        assert not box.reactor._rx_flows
        assert peer.recv(NEGATIVE_WAIT_S) is None, "a failed scatter launch was ACKed"
        assert pool.calls == []
        assert box.reactor.alive()


# --------------------------------------------------------------------------------------------
# T7: request failure while a chunk is mid-EXECUTE for busy_op fulfill/arm/post
# --------------------------------------------------------------------------------------------
def _stage_gated_op(box, peer: RawPeer, *, gathered_first: bool):
    """Drive one chunk into the gated agent op on the REACTOR thread.

    The submit runs on THIS thread first (pool ungated) and returns before
    any GRANT exists, so the gated fulfill/arm/post can only execute on the
    reactor (grant/gather-done pump). Returns (fut, rid, req, cid).
    """
    fut = _submit(box, peer.name, n_desc=16)  # one 64 KiB chunk
    rid, chunk_sizes = _recv_want(peer)
    req = box.reactor._requests[rid]
    cid = box.pool.accepted()[0][3]
    if gathered_first:
        box.poller.complete(cid, ok=True)
        _wait_until(
            lambda: req.posted and req.posted[0].state.name == "GATHERED",
            timeout_s=10.0,
            what="the chunk to reach GATHERED before its credit",
        )
    _grant_all(peer, rid, chunk_sizes)
    assert box.agent.entered.wait(timeout=10), "the gated agent op was never entered"
    return fut, rid, req, cid


def _fail_mid_op_and_settle(box, peer: RawPeer, fut, rid: int, req, terminal_id_fn) -> None:
    """Fail the request while the agent op is blocked, then settle.

    Asserts the full busy_op fail ledger: the future resolves immediately
    with the injected reason, the cancel-WANT is DEFERRED (the in-flight op
    may still become an RDMA write on the receiver's region), the record
    phase routes the op's terminal id as an orphan write, and its row
    releases the staging region exactly once — then the deferred cancel goes
    out and every side table is empty.
    """
    capacity = box.sched.arena_capacity
    # _fail_request's documented contract: call WITHOUT _req_mu, any thread —
    # the only entry that can land mid-EXECUTE while the reactor itself is
    # inside the gated C++ call (forget_peer executes on the blocked reactor).
    box.reactor._fail_request(rid, req, FAIL_PEER_DROPPED)
    res = fut.result(timeout=10)
    assert res.ok is False
    assert res.reason == FAIL_PEER_DROPPED
    assert box.reactor._pending_cancels.get(rid) == peer.name, "busy op not counted as deferred"
    assert box.sched.local_held_count() == 1, "region released while the op was in flight"
    assert peer.recv(NEGATIVE_WAIT_S) is None, "cancel-WANT sent while the op was in flight"

    box.agent.hold.set()
    terminal_id = terminal_id_fn()
    _wait_until(
        lambda: terminal_id in box.reactor._completions,
        timeout_s=10.0,
        what="the record phase to register the orphan route",
    )
    assert box.sched.local_held_count() == 1  # still deferred until the terminal row
    box.poller.complete_xfer(terminal_id, ok=False)
    _wait_until(
        lambda: box.sched.free_bytes() == capacity and box.sched.local_held_count() == 0,
        timeout_s=10.0,
        what="the orphan terminal row to release the region",
    )
    _assert_cancel(peer, rid)  # the deferred cancel goes out at settle
    time.sleep(0.1)  # no late double-release may push past capacity
    assert box.sched.free_bytes() == capacity
    assert box.reactor.alive()
    with box.reactor._req_mu:
        assert not box.reactor._unrouted
        assert not box.reactor._orphan_writes
        assert not box.reactor._pending_cancels


class TestFailDuringBusyOp:
    @pytest.mark.parametrize("release_rc", [1, 0], ids=["armed", "declined"])
    def test_fail_mid_fulfill_releases_once_and_defers_cancel(
        self, make_reactor, raw_peers, release_rc
    ):
        """T7-fulfill: fail while fulfill_chain_1to1 is mid-execution.

        Whether the fulfill lands ARMED (write posts; terminal row after the
        write) or DECLINED (terminal row already pending for the reserved
        id), the reserved id is routed as the orphan write and settles the
        deferred cancel; cancel_chain is never called for a mid-fulfill
        chunk (its reservation is already being consumed).
        """
        poller = ChainFakePoller()
        agent = GatedOpAgent("fulfill", release_rc=release_rc)
        box = make_reactor("fulfillbusy", poller=poller, agent=agent, enable_cpp_chain=True)
        peer = raw_peers(box.reactor.endpoint, "rx")
        assert box.reactor.add_peer(peer.name, peer.endpoint)

        fut, rid, req, cid = _stage_gated_op(box, peer, gathered_first=False)
        assert req.posted[0].reserved, "the launch never took a chain reservation"
        reserved = cid + 50_000  # ChainFakePoller's deterministic mapping
        _fail_mid_op_and_settle(box, peer, fut, rid, req, lambda: reserved)
        assert poller.cancels == [], "cancel_chain called for a mid-fulfill chunk"

    @pytest.mark.parametrize("release_rc", [88_888, -1], ids=["armed", "raced"])
    def test_fail_mid_arm_releases_once_and_defers_cancel(
        self, make_reactor, raw_peers, release_rc
    ):
        """T7-arm: fail while post_transfer_1to1_on_event is mid-execution.

        An arm that succeeds routes its fresh reserved id as the orphan
        write; an arm that LOST the race routes the original copy_id the
        same way (its gather row settles the deferred cancel).
        """
        agent = GatedOpAgent("arm", release_rc=release_rc)
        box = make_reactor("armbusy", agent=agent, enable_cpp_chain=True)  # one-shot mode
        assert box.reactor._reserve_fn is None and box.reactor._chain_fn is not None
        peer = raw_peers(box.reactor.endpoint, "rx")
        assert box.reactor.add_peer(peer.name, peer.endpoint)

        fut, rid, req, cid = _stage_gated_op(box, peer, gathered_first=False)
        terminal = release_rc if release_rc >= 0 else cid
        _fail_mid_op_and_settle(box, peer, fut, rid, req, lambda: terminal)

    def test_fail_mid_post_releases_once_and_defers_cancel(self, make_reactor, raw_peers):
        """T7-post: fail while the classic post_transfer_1to1 is mid-execution.

        The synchronously-posted write's xfer id (decided only when the gate
        releases) is routed as the orphan write; its terminal row releases
        the region and lets the deferred cancel out.
        """
        agent = GatedOpAgent("post")
        box = make_reactor("postbusy", agent=agent)  # chain off: classic path
        peer = raw_peers(box.reactor.endpoint, "rx")
        assert box.reactor.add_peer(peer.name, peer.endpoint)

        fut, rid, req, _cid = _stage_gated_op(box, peer, gathered_first=True)
        assert req.posted[0].state.name == "GATHERED"

        def terminal_id():
            _wait_until(lambda: len(box.agent.posts) == 1, 10.0, "the gated post to record")
            return box.agent.posts[0][4]

        _fail_mid_op_and_settle(box, peer, fut, rid, req, terminal_id)
