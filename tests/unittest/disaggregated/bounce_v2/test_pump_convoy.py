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
"""GPU-free tests for the anti-convoy reactor restructure.

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
    leaves the reactor alive.

The deterministic windows are held open by a gated fake pool whose
``launch_chunk`` blocks on an event (the pump's unlocked EXECUTE phase),
plus a randomized stress run as the broad regression net for the fixed
handoff race. Reuses the fakes/helpers of test_reactor_unit.py (sibling
import).
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
    FAIL_GATHER,
    FAIL_PEER_DROPPED,
    BounceReactor,
)
from tensorrt_llm._torch.disaggregation.bounce_v2.scheduler import CreditScheduler  # noqa: E402

# Reactor threads are created inside test bodies and joined by the fixture
# teardown (after pytest-threadleak's end-of-call check) — same rationale as
# the identical marker in test_reactor_unit.py.
pytestmark = pytest.mark.threadleak(enabled=False)


class GatedCopyPool(FakeCopyPool):
    """FakeCopyPool whose launch_chunk BLOCKS on a gate.

    Holds the pump's unlocked C++ EXECUTE window open deterministically:
    the completion id is decided (and recorded) BEFORE blocking, so the test
    can deliver its row while the submitting thread is still inside the call
    — exactly the ordering the ``_unrouted`` parking dict exists for.
    """

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.hold = threading.Event()

    def launch_chunk(self, handle, chunk_idx, staging_base) -> int:
        cid = super().launch_chunk(handle, chunk_idx, staging_base)
        self.last_cid = cid
        self.entered.set()
        assert self.hold.wait(timeout=30), "test bug: the pool gate was never released"
        return cid


class GatedOpAgent(FakeXferAgent):
    """FakeXferAgent gating one agent entry point ("post" or "chained").

    The gated call blocks on ``hold`` after setting ``entered``: the
    deterministic mid-EXECUTE window for the busy_op failure tests. The
    chained gate holds the call open AFTER the ids are decided (the C++
    contract's "pinned plan snapshot": a release_plan during the call does
    not fail it).
    """

    def __init__(self, gate: str = "post") -> None:
        super().__init__()
        self.gate = gate
        self.entered = threading.Event()
        self.hold = threading.Event()

    def _wait_gate(self) -> None:
        self.entered.set()
        assert self.hold.wait(timeout=30), "test bug: the agent gate was never released"

    def post_transfer_1to1(self, *args) -> int:
        if self.gate == "post":
            self._wait_gate()
        return super().post_transfer_1to1(*args)

    def launch_chunk_chained(self, *args):
        out = super().launch_chunk_chained(*args)
        if self.gate == "chained":
            self._wait_gate()
        return out


class GatedPlanPool(FakeCopyPool):
    """FakeCopyPool whose launch_chunk BLOCKS on a gate BEFORE the handle check.

    A release_plan during the gate makes the resumed launch raise ValueError
    — the binding's deterministic terminal for a launch racing the
    failure-side release.
    """

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.hold = threading.Event()

    def launch_chunk(self, handle, chunk_idx, staging_base) -> int:
        self.entered.set()
        assert self.hold.wait(timeout=30), "test bug: the plan-pool gate was never released"
        return super().launch_chunk(handle, chunk_idx, staging_base)


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
        for obj in (box.pool, box.agent):
            hold = getattr(obj, "hold", None)
            if hold is not None:
                hold.set()
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
            """Complete every accepted gather/chained launch/write, jittered.

            A chained chunk's ONE row publishes under the reserved xfer id
            (KIND_XFER).
            """
            done: set[int] = set()
            rng = random.Random(41)
            while not stop.is_set():
                rows = [(c[3], False) for c in box.pool.accepted()]
                rows += [(p[4], True) for p in box.agent.posts]
                rows += [(c[6], True) for c in list(box.agent.chained)]
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
# T6: receiver scatter — the C++-sink path, driven by the sink fake
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
        """T6a: DATA routes through submit_scatter_runs.

        The sink receives the granted region's base/bytes and the EXACT raw
        wire blob; the scatter completes, ACKs, and the region recycles.
        """
        box = make_reactor("sinkok")
        pool = box.pool
        peer = raw_peers(box.reactor.endpoint, "tx")
        cap0 = box.sched.free_bytes()

        credit = _grant_one(peer, rid=71)
        runs = _two_run_data(peer, 71, credit.region_handle)
        _wait_until(lambda: len(pool.sink_accepted()) == 1, 10.0, "the sink submit")
        region_base, region_bytes, blob, rc = pool.sink_accepted()[0]
        assert region_base == BASE + credit.region_handle
        assert region_bytes == box.sched.region_bytes(credit.region_handle) == CHUNK
        assert blob == runs.tobytes(), "the sink did not receive the exact raw wire runs"
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
        box = make_reactor("sinkrej")
        pool = box.pool
        pool.sink_mode = "reject"
        peer = raw_peers(box.reactor.endpoint, "tx")
        cap0 = box.sched.free_bytes()

        credit = _grant_one(peer, rid=72)
        _two_run_data(peer, 72, credit.region_handle)
        _wait_until(lambda: len(pool.sink_calls) == 1, 10.0, "the sink validation call")
        _wait_until(lambda: box.sched.free_bytes() == cap0, 10.0, "the rejected region release")
        assert not box.reactor._rx_flows, "flow accounting leaked for the rejected chunk"
        assert box.reactor.stats().get("rx_data", 0) == 0, "rejected DATA counted as accepted"
        assert peer.recv(NEGATIVE_WAIT_S) is None, "a validation-rejected scatter was ACKed"
        assert box.reactor.alive()

    def test_sink_busy_backlogs_raw_runs_then_retries(self, make_reactor, raw_peers):
        """T6c: BUSY(-1) parks the job WITH its raw runs.

        The retry resubmits the IDENTICAL blob once a context frees, then
        the scatter completes and ACKs.
        """
        box = make_reactor("sinkbusy")
        pool = box.pool
        pool.busy = True
        peer = raw_peers(box.reactor.endpoint, "tx")
        cap0 = box.sched.free_bytes()

        credit = _grant_one(peer, rid=73)
        runs = _two_run_data(peer, 73, credit.region_handle)
        _wait_until(lambda: len(box.reactor._scatter_backlog) == 1, 10.0, "the BUSY backlog")
        assert box.reactor._scatter_backlog[0].runs is not None, "backlog job lost its raw runs"

        pool.busy = False
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
        box = make_reactor("sinkraise")
        box.pool.sink_mode = "raise"
        peer = raw_peers(box.reactor.endpoint, "tx")
        cap0 = box.sched.free_bytes()

        credit = _grant_one(peer, rid=74)
        _two_run_data(peer, 74, credit.region_handle)
        _wait_until(lambda: box.sched.free_bytes() == cap0, 10.0, "the failed-launch release")
        assert not box.reactor._rx_flows
        assert peer.recv(NEGATIVE_WAIT_S) is None, "a failed scatter launch was ACKed"
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
    def test_fail_mid_post_releases_once_and_defers_cancel(self, make_reactor, raw_peers):
        """T7-post: fail while the classic post_transfer_1to1 is mid-execution.

        The synchronously-posted write's xfer id (decided only when the gate
        releases) is routed as the orphan write; its terminal row releases
        the region and lets the deferred cancel out.
        """
        agent = GatedOpAgent("post")
        box = make_reactor("postbusy", agent=agent)  # fake pool: classic post path
        peer = raw_peers(box.reactor.endpoint, "rx")
        assert box.reactor.add_peer(peer.name, peer.endpoint)

        fut, rid, req, _cid = _stage_gated_op(box, peer, gathered_first=True)
        assert req.posted[0].state.name == "GATHERED"

        def terminal_id():
            _wait_until(lambda: len(box.agent.posts) == 1, 10.0, "the gated post to record")
            return box.agent.posts[0][4]

        _fail_mid_op_and_settle(box, peer, fut, rid, req, terminal_id)


# --------------------------------------------------------------------------------------------
# T8: pump ownership survives the finally backstop (ae8fe99bcc1 regression)
# --------------------------------------------------------------------------------------------
class TestPumpOwnershipBackstop:
    def test_busy_chunk_never_reselected_and_posted_exactly_once(self, make_reactor, raw_peers):
        """ae8fe99bcc1 regression: exactly ONE RDMA post per chunk, ever.

        The pre-fix finally backstop stomped a new legitimate owner's
        ``pump_busy`` after the atomic idle release, admitting a concurrent
        second owner that re-selected a mid-classic-post chunk and DUPLICATED
        the RDMA write (functionally invisible: the extra ACK drops as
        stale, so only a spy count can pin it). Two assertions, one per half
        of the fix:

        Assertion A — the ``busy_op`` skip in ``_next_action_locked`` (the
        reliable pre-fix discriminator): with the one chunk mid-post on the
        blocked owner (state GATHERED, ``busy_op == "post"``, ``xfer_id``
        unassigned), the decide sweep must NOT return that chunk. Pre-fix
        code matched the ``GATHERED and has_credit and xfer_id < 0`` arm and
        returned ``("post", <busy chunk>)`` — the exact re-selection a
        second owner performed.

        Assertion B — the single-owner invariant behaviorally: hammering
        ``_pump`` from several threads while the owner is blocked mid-post
        must only bounce off ``pump_busy`` / hand over via ``pump_again``;
        after release, the fake agent's post op was called EXACTLY once for
        the one chunk and the future resolves OK. (The pre-fix stomp needed
        the precise finally-after-atomic-release window, which cannot be
        forced from a test without reaching into private state — hence
        assertion A carries the pre-fix failure; this half guards the
        invariant the ``owned`` flag now guarantees.)
        """
        agent = GatedOpAgent("post")
        box = make_reactor("ownstomp", agent=agent)  # fake pool: classic post path
        peer = raw_peers(box.reactor.endpoint, "rx")
        assert box.reactor.add_peer(peer.name, peer.endpoint)

        fut, rid, req, _cid = _stage_gated_op(box, peer, gathered_first=True)
        posted = req.posted[0]
        assert posted.busy_op == "post"
        assert posted.state.name == "GATHERED" and posted.has_credit and posted.xfer_id < 0

        # --- Assertion A: the decide sweep skips the mid-EXECUTE chunk. ---
        with box.reactor._req_mu:
            action = box.reactor._next_action_locked(rid, req)
        assert action is None, f"_next_action_locked re-selected a mid-post chunk: {action!r}"
        assert posted.busy_op == "post"  # the probe did not disturb the marker

        # --- Assertion B: concurrent pumpers can only bounce or hand over. --
        hammers = [
            threading.Thread(
                target=lambda: [box.reactor._pump(rid) for _ in range(20)], daemon=True
            )
            for _ in range(3)
        ]
        for t in hammers:
            t.start()
        for t in hammers:
            t.join(timeout=10)
            assert not t.is_alive(), "a hammering _pump call wedged"
        assert len(box.agent.posts) == 0, "a second owner posted while the op was in flight"

        box.agent.hold.set()
        _wait_until(lambda: len(box.agent.posts) == 1, 10.0, "the gated post to record")
        box.poller.complete_xfer(box.agent.posts[0][4], ok=True)
        got = peer.recv(10.0)
        assert got is not None and got[1].msg_type == codec.BounceMsgType.DATA
        _ack(peer, rid, 0, posted.remote_handle)
        assert fut.result(timeout=10).ok is True
        # THE spy count: exactly one RDMA post for the one chunk, ever.
        assert len(box.agent.posts) == 1, "duplicate RDMA post for a single chunk"
        assert box.reactor.alive()


# --------------------------------------------------------------------------------------------
# Per-request plan handle — chained rows, release
# --------------------------------------------------------------------------------------------
class TestPlanHandleFakes:
    def test_chained_launch_one_row_and_release_on_completion(self, make_reactor, raw_peers):
        """P5-happy: a credited launch chains and releases the plan on ACK.

        Eager gather off -> the launch waits for its credit and goes through
        launch_chunk_chained (right handle/dst/bytes; tx_chained_launches
        bumps); the chunk's ONE row is the reserved xfer id; completing the
        request releases the plan handle exactly once.
        """
        box = make_reactor("planok", enable_eager_gather=False)
        pool, agent = box.pool, box.agent
        peer = raw_peers(box.reactor.endpoint, "rx")
        assert box.reactor.add_peer(peer.name, peer.endpoint)
        capacity = box.sched.arena_capacity

        fut = _submit(box, peer.name, n_desc=16)
        rid, chunk_sizes = _recv_want(peer)
        req = box.reactor._requests[rid]
        handle = req.plan_handle
        assert handle >= 0 and handle in pool.plans
        assert pool.launches == [] and agent.chained == []  # nothing before the credit

        credits = _grant_all(peer, rid, chunk_sizes)
        _wait_until(lambda: len(agent.chained) == 1, 10.0, "the credited chained launch")
        c_handle, c_idx, _base, c_dst, c_bytes, _cid, reserved = agent.chained[0]
        assert (c_handle, c_idx, c_dst, c_bytes) == (handle, 0, credits[0].addr, CHUNK)
        assert box.reactor.stats().get("tx_chained_launches", 0) == 1
        assert pool.launches == [], "a chained chunk also took a plain launch"

        box.poller.complete_xfer(reserved, ok=True)  # the ONE terminal row
        got = peer.recv(10.0)
        assert got is not None and got[1].msg_type == codec.BounceMsgType.DATA
        _ack(peer, rid, 0, credits[0].region_handle)
        assert fut.result(timeout=10).ok is True
        _wait_until(lambda: pool.releases == [handle], 10.0, "the ACK-complete plan release")
        _wait_until(lambda: box.sched.free_bytes() == capacity, 10.0, "the region recycle")
        s = box.reactor.stats()
        assert s["tx_xfer_events"] == 1 and s["tx_acked_chunks"] == 1
        assert s.get("tx_gather_events", 0) == 0  # the gather row was consumed in C++
        assert box.reactor.alive()

    def test_chained_gather_failure_row_maps_to_fail_gather(self, make_reactor, raw_peers):
        """P5/P3-fake: a (reserved, KIND_EVENT, 0) row means the gather died.

        The GPU suite cannot inject a real gather fault (it would poison the
        CUDA context — see the note in test_event_chain.py), so the row
        mapping is pinned here: the request fails FAIL_GATHER, the region is
        fully conserved, the plan handle is released exactly once, and the
        reactor stays alive.
        """
        box = make_reactor("plangfail", enable_eager_gather=False)
        pool, agent = box.pool, box.agent
        peer = raw_peers(box.reactor.endpoint, "rx")
        assert box.reactor.add_peer(peer.name, peer.endpoint)
        capacity = box.sched.arena_capacity

        fut = _submit(box, peer.name, n_desc=16)
        rid, chunk_sizes = _recv_want(peer)
        handle = box.reactor._requests[rid].plan_handle
        _grant_all(peer, rid, chunk_sizes)
        _wait_until(lambda: len(agent.chained) == 1, 10.0, "the credited chained launch")
        reserved = agent.chained[0][6]

        box.poller.complete(reserved, ok=False)  # KIND_EVENT: the gather failed in C++
        res = fut.result(timeout=10)
        assert res.ok is False
        assert res.reason == FAIL_GATHER, f"gather-stage row misattributed: {res.reason}"
        _wait_until(lambda: box.sched.free_bytes() == capacity, 10.0, "the region release")
        assert pool.releases == [handle], "plan handle not released exactly once on failure"
        _assert_cancel(peer, rid)  # write never posted -> the cancel goes out now
        assert not box.reactor._unrouted
        assert box.reactor.alive()

    def test_fail_between_launches_releases_plan_exactly_once(self, make_reactor, raw_peers):
        """P5a: forget_peer between launches -> ONE release_plan, ever.

        Two eagerly-launched (uncredited -> plain launch_chunk) chunks are
        GATHERING when the peer is forgotten: the fail path releases the plan
        handle exactly once, the orphaned gather rows recycle both regions,
        and the shutdown failAll must NOT release the handle again.
        """
        box = make_reactor("planfail")  # eager launches: no credit, no chaining
        pool = box.pool
        peer = raw_peers(box.reactor.endpoint, "rx")
        assert box.reactor.add_peer(peer.name, peer.endpoint)
        capacity = box.sched.arena_capacity

        fut = _submit(box, peer.name, n_desc=32)  # two 64 KiB chunks, both eager
        rid, _chunk_sizes = _recv_want(peer)
        handle = box.reactor._requests[rid].plan_handle
        assert handle >= 0
        assert len(pool.launches) == 2, "eager launches did not go through launch_chunk"
        assert [launched[0] for launched in pool.launches] == [handle, handle]

        box.reactor.forget_peer(peer.name)
        res = fut.result(timeout=10)
        assert res.ok is False and res.reason == FAIL_PEER_DROPPED
        _wait_until(lambda: pool.releases == [handle], 10.0, "the failure-side plan release")

        for _h, _ci, _base, cid in pool.launches:  # the orphaned gather rows
            box.poller.complete(cid, ok=True)
        _wait_until(lambda: box.sched.free_bytes() == capacity, 10.0, "both regions to recycle")
        assert box.reactor.alive()
        box.reactor.shutdown()  # failAll must not double-release (handle latched)
        assert pool.releases == [handle], "release_plan called more than once"

    def test_fail_mid_chained_launch_defers_cancel_and_releases_once(self, make_reactor, raw_peers):
        """P5b: fail while a CHAINED launch is mid-execution.

        busy_op == "launch_chained" counts as a deferred write (the C++ side
        may already be posting the RDMA write): the cancel-WANT is held back,
        the plan is released exactly once at the fail, the record phase
        routes the reserved id as the orphan write, and its terminal row
        recycles the region and lets the cancel out.
        """
        agent = GatedOpAgent("chained")
        box = make_reactor("planchainbusy", agent=agent, enable_eager_gather=False)
        pool = box.pool
        peer = raw_peers(box.reactor.endpoint, "rx")
        assert box.reactor.add_peer(peer.name, peer.endpoint)
        capacity = box.sched.arena_capacity

        fut = _submit(box, peer.name, n_desc=16)
        rid, chunk_sizes = _recv_want(peer)
        req = box.reactor._requests[rid]
        handle = req.plan_handle
        _grant_all(peer, rid, chunk_sizes)
        assert agent.entered.wait(timeout=10), "the gated chained launch was never entered"

        # _fail_request's documented any-thread contract (the reactor itself
        # is blocked inside the gated chained call).
        box.reactor._fail_request(rid, req, FAIL_PEER_DROPPED)
        res = fut.result(timeout=10)
        assert res.ok is False and res.reason == FAIL_PEER_DROPPED
        assert box.reactor._pending_cancels.get(rid) == peer.name, "chained busy not deferred"
        assert box.sched.local_held_count() == 1
        assert pool.releases == [handle], "plan not released exactly once at the fail"
        assert peer.recv(NEGATIVE_WAIT_S) is None, "cancel sent while the chained launch ran"

        agent.hold.set()
        reserved = agent.chained[0][6]
        _wait_until(
            lambda: reserved in box.reactor._completions,
            timeout_s=10.0,
            what="the record phase to register the orphan route for the reserved id",
        )
        box.poller.complete_xfer(reserved, ok=False)  # the guaranteed terminal row
        _wait_until(
            lambda: box.sched.free_bytes() == capacity and box.sched.local_held_count() == 0,
            timeout_s=10.0,
            what="the orphan terminal row to release the region",
        )
        _assert_cancel(peer, rid)
        assert pool.releases == [handle]  # still exactly once
        with box.reactor._req_mu:
            assert not box.reactor._unrouted
            assert not box.reactor._orphan_writes
            assert not box.reactor._pending_cancels
        assert box.reactor.alive()

    def test_launch_after_release_valueerror_lands_in_error_path(self, make_reactor, raw_peers):
        """P5c: a launch racing the failure-side release raises ValueError.

        The gate holds the plain launch open while forget_peer fails the
        request and releases the plan; the resumed launch then raises
        (unknown handle) and the launch-error path recycles the staging
        region immediately — no completion id ever exists, nothing leaks,
        the reactor stays alive.
        """
        pool = GatedPlanPool()
        box = make_reactor("planrace", pool=pool)
        peer = raw_peers(box.reactor.endpoint, "rx")
        assert box.reactor.add_peer(peer.name, peer.endpoint)
        _pin_pump_to_submitter(box)
        capacity = box.sched.arena_capacity

        t, result = _submit_on_thread(box, peer.name, n_desc=16)
        assert pool.entered.wait(timeout=10), "the eager launch never reached the gate"
        rid, _chunk_sizes = _recv_want(peer)
        handle = box.reactor._requests[rid].plan_handle

        box.reactor.forget_peer(peer.name)
        _wait_until(lambda: rid not in box.reactor._requests, 10.0, "the fail path to run")
        _wait_until(lambda: pool.releases == [handle], 10.0, "the failure-side plan release")
        assert box.sched.local_held_count() == 1  # the busy launch still holds its region
        # A plain busy launch is NOT a deferred write: no cancel is held back
        # (the cancel itself is dropped by design — forget_peer removed the
        # route before the reclaim ran).
        assert rid not in box.reactor._pending_cancels

        pool.hold.set()  # the resumed launch now raises ValueError (plan gone)
        t.join(timeout=10)
        assert not t.is_alive()
        res = result.future.result(timeout=10)
        assert res.ok is False and res.reason == FAIL_PEER_DROPPED
        _wait_until(
            lambda: box.sched.free_bytes() == capacity and box.sched.local_held_count() == 0,
            timeout_s=10.0,
            what="the launch-error path to recycle the staged region",
        )
        assert pool.launches == [], "the raced launch still allocated a completion id"
        with box.reactor._req_mu:
            assert not box.reactor._completions
        time.sleep(0.1)  # no late double-release may push past capacity
        assert box.sched.free_bytes() == capacity
        assert box.reactor.alive()
