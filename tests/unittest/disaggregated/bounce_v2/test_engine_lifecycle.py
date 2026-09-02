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
"""GPU tests for bounce_v2 peer lifecycle + concurrency (round-2 behaviors).

Regression tests for the round-2 fixes verified with the implementer:
  - B1: forget_peer/add_peer drop the DEALER route SYNCHRONOUSLY, so a
    compatible re-registration followed IMMEDIATELY by a submit succeeds
    (previously the queued async forget could close the fresh route);
  - endpoint replacement: re-registering a peer with a different endpoint
    replaces the dealer (the old idempotent-by-name add_peer kept the stale
    one);
  - sender-side abandonment lets the RECEIVER reclaim its granted regions
    (via the lease, since the cancel toward a forgotten peer is dropped);
  - multi-threaded submits and bidirectional simultaneous transfers (the
    eager budget's half-arena cap prevents a circular wait);
  - B2: a failed RDMA post during a batched credit attach fails the request
    exactly once (abandon_reason latch) instead of pumping a deleted request.

Same skip guards as test_mechanism_bindings.py; shares its helpers/fixtures
pattern with test_reactor_engine.py (imported as a sibling test module).
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("bounce_v2 lifecycle tests require a CUDA device", allow_module_level=True)

tab = pytest.importorskip(
    "tensorrt_llm.tensorrt_llm_transfer_agent_binding",
    reason="bounce_v2 lifecycle tests require the compiled tensorrt_llm wheel",
)

from test_reactor_engine import (  # noqa: E402  (sibling test module)
    DEVICE,
    KIB,
    MIB,
    WAIT_CAP_MS,
    FakePeer,
    _assert_landed,
    _cfg,
    _forged_handshake,
    _scattered_case,
    _wait_until,
)

from tensorrt_llm._torch.disaggregation.bounce_v2 import codec  # noqa: E402
from tensorrt_llm._torch.disaggregation.bounce_v2.engine import BounceEngine  # noqa: E402
from tensorrt_llm._torch.disaggregation.bounce_v2.reactor import FAIL_WRITE  # noqa: E402

# See the identical marker in test_reactor_engine.py: engines are created
# inside the test body, so their threads are still alive when pytest-threadleak
# checks at end-of-call; fixture teardown joins them right after.
pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture
def make_engine():
    """Same contract as test_reactor_engine.make_engine.

    Fixtures are not importable across test modules, so the thin factory is
    redefined.
    """
    boxes: list = []

    def _make(tag: str, **cfg_kw):
        name = f"bv2eng_{tag}_{uuid.uuid4().hex[:8]}"
        agent = tab.NixlTransferAgent(tab.BaseAgentConfig(name))
        try:
            engine = BounceEngine(agent, _cfg(**cfg_kw), DEVICE, name, bind_ip="127.0.0.1")
        except BaseException:
            agent.shutdown()
            raise
        box = SimpleNamespace(name=name, agent=agent, engine=engine)
        boxes.append(box)
        return box

    yield _make
    for box in reversed(boxes):
        box.engine.shutdown()
    torch.cuda.synchronize()
    for box in reversed(boxes):
        box.agent.shutdown()


@pytest.fixture
def fake_peers():
    peers: list[FakePeer] = []

    def _make(target_endpoint: str, tag: str = "fake") -> FakePeer:
        peer = FakePeer(target_endpoint, tag)
        peers.append(peer)
        return peer

    yield _make
    for peer in peers:
        peer.close()


def _pair(make_engine, **cfg_kw):
    src = make_engine("src", **cfg_kw)
    dst = make_engine("dst", **cfg_kw)
    src.agent.load_remote_agent(dst.name, dst.agent.get_local_agent_desc())
    assert src.engine.add_peer(dst.name, dst.engine.local_handshake_blob())
    return src, dst


# --------------------------------------------------------------------------------------------
# 1. peer route lifecycle (B1 regression territory)
# --------------------------------------------------------------------------------------------
class TestPeerRouting:
    # Pins the B1 regression: forget_peer snapshots its victim rids at call
    # time, so a submit right after a compatible re-registration must survive.
    def test_reregistration_then_immediate_submit_succeeds(self, make_engine):
        """B1 regression: compatible re-registration + immediate submit works.

        No drain wait, that is the point: the route replacement is
        synchronous and the reclaim snapshots its victims, so the fresh
        request may not be clobbered by anything the re-registration queued.
        """
        src, dst = _pair(make_engine)
        # Warm transfer proving the initial route.
        warm = _scattered_case(n_desc=32, desc_bytes=2048, seed=200)
        status = src.engine.submit(warm.src_ptrs, warm.dst_ptrs, warm.sizes, DEVICE, dst.name)
        assert status.wait(WAIT_CAP_MS), status.last_status_str()
        _assert_landed(warm)

        assert src.engine.add_peer(dst.name, dst.engine.local_handshake_blob())
        case = _scattered_case(n_desc=64, desc_bytes=4 * KIB, seed=201)
        status = src.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, dst.name)
        assert status.wait(WAIT_CAP_MS), (
            f"submit immediately after a compatible re-registration failed: "
            f"{status.last_status_str()}"
        )
        _assert_landed(case)

    def test_endpoint_replacement_dead_then_live(self, make_engine, fake_peers):
        """Re-registering a peer with a new endpoint replaces its dealer.

        Register a peer name with a DEAD endpoint, then re-register the same
        name with the LIVE endpoint: the dealer must be replaced (not kept
        idempotently by name) and the transfer succeed.
        """
        src = make_engine("src")
        dst = make_engine("dst")
        src.agent.load_remote_agent(dst.name, dst.agent.get_local_agent_desc())

        dead = fake_peers(src.engine._reactor.endpoint, "dead")  # never answers
        dead_blob = _forged_handshake(src.engine._cfg.max_chunk_size_bytes, dead.endpoint)
        assert src.engine.add_peer(dst.name, dead_blob)

        assert src.engine.add_peer(dst.name, dst.engine.local_handshake_blob())
        case = _scattered_case(n_desc=64, desc_bytes=4 * KIB, seed=210)
        status = src.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, dst.name)
        assert status.wait(WAIT_CAP_MS), (
            f"transfer after endpoint replacement failed (stale dealer kept?): "
            f"{status.last_status_str()}"
        )
        _assert_landed(case)


# --------------------------------------------------------------------------------------------
# 2. sender abandonment -> receiver-side reclamation
# --------------------------------------------------------------------------------------------
class TestReceiverReclamation:
    def test_sender_forget_midflight_receiver_reclaims_regions(self, make_engine):
        """Sender-side forget_peer mid-flight: the receiver reclaims regions.

        The cancel toward the forgotten peer is dropped (route already
        removed), so reclamation rides the receiver flow lease + quarantine —
        short timeouts keep the bound tight (lease 2x2s, quarantine 2s).
        """
        src, dst = _pair(
            make_engine,
            arena_size_bytes=32 * MIB,
            max_chunk_size_bytes=2 * MIB,
            max_inflight_chunks_per_request=2,
            request_timeout_ms=2000,
        )
        dst_free0 = dst.engine._scheduler.free_bytes()
        src_free0 = src.engine._scheduler.free_bytes()

        case = _scattered_case(n_desc=320, desc_bytes=64 * KIB, seed=220)  # >=10 chunks
        status = src.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, dst.name)
        src.engine.forget_peer(dst.name)
        status.wait(WAIT_CAP_MS)  # resolves either way (bounded), never hangs
        assert status.is_completed()

        _wait_until(
            lambda: dst.engine._scheduler.free_bytes() == dst_free0,
            timeout_s=30.0,
            what="the receiver arena to reclaim every granted region",
        )
        _wait_until(
            lambda: not dst.engine._reactor._rx_flows,
            timeout_s=10.0,
            what="the receiver's flow table to empty",
        )
        _wait_until(
            lambda: src.engine._scheduler.free_bytes() == src_free0,
            timeout_s=10.0,
            what="the sender arena to release its staging regions",
        )


# --------------------------------------------------------------------------------------------
# 3. concurrency
# --------------------------------------------------------------------------------------------
class TestConcurrency:
    def test_concurrent_submits_all_succeed_byte_exact(self, make_engine):
        """8 threads submitting to one engine pair concurrently.

        The real Sender submits from KV_TRANSFER_NUM_THREADS worker threads.
        """
        src, dst = _pair(make_engine)
        n_threads = 8
        cases = [
            _scattered_case(n_desc=64, desc_bytes=4 * KIB, seed=300 + i) for i in range(n_threads)
        ]
        barrier = threading.Barrier(n_threads)

        def run(i: int):
            torch.cuda.set_device(DEVICE)  # submit's contract: device current
            barrier.wait(timeout=30)  # maximize overlap
            c = cases[i]
            st = src.engine.submit(c.src_ptrs, c.dst_ptrs, c.sizes, DEVICE, dst.name)
            return st.wait(WAIT_CAP_MS), st.last_status_str()

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            results = list(ex.map(run, range(n_threads)))
        for i, (ok, reason) in enumerate(results):
            assert ok, f"concurrent submit {i} failed: {reason}"
        for case in cases:
            _assert_landed(case)

    def test_bidirectional_simultaneous_transfers(self, make_engine):
        """A->B and B->A at the same time, both byte-exact.

        Each side is sender AND receiver on ONE shared arena; the eager
        budget's half-arena cap must prevent the two eager senders from
        starving each other into a circular wait.
        """
        a = make_engine("a")
        b = make_engine("b")
        a.agent.load_remote_agent(b.name, b.agent.get_local_agent_desc())
        b.agent.load_remote_agent(a.name, a.agent.get_local_agent_desc())
        assert a.engine.add_peer(b.name, b.engine.local_handshake_blob())
        assert b.engine.add_peer(a.name, a.engine.local_handshake_blob())

        # 10 MiB each way through 64 MiB arenas: chunked, both directions.
        case_ab = _scattered_case(n_desc=160, desc_bytes=64 * KIB, seed=310)
        case_ba = _scattered_case(n_desc=160, desc_bytes=64 * KIB, seed=311)
        barrier = threading.Barrier(2)

        def run(engine_box, case, peer_name):
            torch.cuda.set_device(DEVICE)
            barrier.wait(timeout=30)
            st = engine_box.engine.submit(
                case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, peer_name
            )
            return st.wait(WAIT_CAP_MS), st.last_status_str()

        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_ab = ex.submit(run, a, case_ab, b.name)
            fut_ba = ex.submit(run, b, case_ba, a.name)
            ok_ab, reason_ab = fut_ab.result(timeout=120)
            ok_ba, reason_ba = fut_ba.result(timeout=120)
        assert ok_ab, f"A->B failed: {reason_ab}"
        assert ok_ba, f"B->A failed: {reason_ba}"
        _assert_landed(case_ab)
        _assert_landed(case_ba)


# --------------------------------------------------------------------------------------------
# 4. failed RDMA post during a batched credit attach (B2 regression)
# --------------------------------------------------------------------------------------------
class _FailingWriteAgent:
    """post_transfer_1to1 stub reporting every post as failed (-1).

    Counts calls. Only the reactor's write path touches ``_agent``, so
    swapping it on a live engine isolates exactly that path.
    """

    def __init__(self):
        self.calls = 0

    def post_transfer_1to1(self, *args, **kwargs):
        self.calls += 1
        return -1


class TestWriteFailure:
    def test_failed_post_during_batched_credit_attach_fails_once(self, make_engine, fake_peers):
        """B2 regression: abandon_reason latch in _fail_request_locked.

        Every chunk is eagerly GATHERED against a silent peer, then ONE
        batched GRANT covering all chunks arrives while post_transfer_1to1
        always returns -1. The FIRST failed post must fail the request
        exactly once with FAIL_WRITE and the latch must stop the credit-
        attach loop: pre-fix, the loop kept pumping the already-deleted
        request — a second failed post KeyError'd `del self._requests[rid]`
        and killed the reactor thread.
        """
        cfg_kw = dict(
            request_timeout_ms=0,  # only the immediate FAIL_WRITE can resolve
            max_chunk_size_bytes=2 * MIB,
            max_inflight_chunks_per_request=8,
        )
        eng = make_engine("tx", **cfg_kw)
        dst = make_engine("dst", **cfg_kw)  # for the healthy transfer after
        eng.agent.load_remote_agent(dst.name, dst.agent.get_local_agent_desc())
        assert eng.engine.add_peer(dst.name, dst.engine.local_handshake_blob())
        ghost = fake_peers(eng.engine._reactor.endpoint, "ghost")
        assert eng.engine.add_peer(ghost.name, _forged_handshake(2 * MIB, ghost.endpoint))

        capacity = eng.engine._scheduler.arena_capacity
        real_agent = eng.engine._reactor._agent
        stub = _FailingWriteAgent()
        eng.engine._reactor._agent = stub
        try:
            # 8 MiB over 2 MiB chunks: 4 chunks, all within the eager window.
            case = _scattered_case(n_desc=128, desc_bytes=64 * KIB, seed=400)
            status = eng.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, ghost.name)
            got = ghost.recv(10.0)
            assert got is not None, "no WANT reached the ghost receiver"
            _, header, want_blob = got
            assert header.msg_type == codec.BounceMsgType.WANT
            rid = header.request_id
            chunk_sizes, _ep = codec.decode_want(want_blob, header)
            assert len(chunk_sizes) >= 4

            def _all_gathered_creditless() -> bool:
                reqs = list(eng.engine._reactor._requests.values())
                return (
                    len(reqs) == 1
                    and len(reqs[0].posted) == len(chunk_sizes)
                    and all(p.state.name == "GATHERED" and not p.has_credit for p in reqs[0].posted)
                )

            _wait_until(
                _all_gathered_creditless,
                timeout_s=15.0,
                what="every chunk to reach GATHERED with no credit",
            )

            # One BATCHED grant for all chunks: the attach loop pairs them
            # FIFO and posts the first write, which fails.
            credits = [
                codec.CreditEntry(
                    addr=(2 * MIB) * (i + 1), length=int(s), dev_id=DEVICE, region_handle=i
                )
                for i, s in enumerate(chunk_sizes)
            ]
            t0 = time.monotonic()
            ghost.send(codec.encode_grant(rid, credits))
            assert status.wait(WAIT_CAP_MS) is False
            assert time.monotonic() - t0 < 10.0, "FAIL_WRITE resolved via timeout, not latch"
            assert status.last_status_str() == FAIL_WRITE
            assert "RDMA write failed" in status.last_status_str()
            # Exactly ONE post: the latch stopped the attach loop after the
            # first failure (pre-fix: a second call, then a reactor crash).
            assert stub.calls == 1, f"post_transfer_1to1 called {stub.calls} times"
            assert eng.engine._reactor.alive(), "reactor thread died on the failed post"
            # Every staging region (all GATHERED at failure time) came back.
            _wait_until(
                lambda: eng.engine._scheduler.local_held_count() == 0
                and eng.engine._scheduler.free_bytes() == capacity,
                timeout_s=10.0,
                what="the failed request's staging regions to be released",
            )
        finally:
            eng.engine._reactor._agent = real_agent

        # The engine is still fully serviceable with the real agent restored.
        healthy = _scattered_case(n_desc=64, desc_bytes=4 * KIB, seed=401)
        status2 = eng.engine.submit(
            healthy.src_ptrs, healthy.dst_ptrs, healthy.sizes, DEVICE, dst.name
        )
        assert status2.wait(WAIT_CAP_MS), status2.last_status_str()
        _assert_landed(healthy)
