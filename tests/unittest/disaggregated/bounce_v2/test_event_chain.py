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
"""GPU tests for the event-driven reactor fd and the two-phase C++ chain.

Feature A — completion wakeup fd (default-on when the compiled poller exposes
``set_wakeup_fd`` and the fd setup succeeds): the reactor parks in a
deadline-driven poll (up to 100 ms) instead of the legacy fixed 1 ms tick and
is woken by an fd token on every C++ publish/retire and every cross-thread
Python command. Pins: the mode engages with the real bindings, wakes are
fd-delivered (``reactor_wake_fd`` counter), the idle timeout is not pinned at
1 ms, and shutdown closes the fds behind the ``set_wakeup_fd(-1)`` fence even
with copies in flight.

Feature B — two-phase chain (``enable_cpp_chain``): gather launch RESERVES a
chain id (``poller.reserve_chain``), the credit FULFILLS it
(``agent.fulfill_chain_1to1``) — so a gather that finished BEFORE its GRANT
(the ordering the one-shot arm always lost) still takes the C++ path
(``tx_chain_fulfilled_late``). Pins: the previously-lost ordering, failure
before fulfill (timeout / forget_peer / shutdown) with region conservation,
chain-off staying byte-for-byte classic, and exactly-once completion
accounting over chained transfers.

Same skip guards as test_mechanism_bindings.py; shares its helpers/fixtures
pattern with test_reactor_engine.py (imported as a sibling test module).
"""

from __future__ import annotations

import os
import time
import uuid
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("bounce_v2 event/chain tests require a CUDA device", allow_module_level=True)

tab = pytest.importorskip(
    "tensorrt_llm.tensorrt_llm_transfer_agent_binding",
    reason="bounce_v2 event/chain tests require the compiled tensorrt_llm wheel",
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
from tensorrt_llm._torch.disaggregation.bounce_v2.reactor import (  # noqa: E402
    _POLL_MS,
    FAIL_NO_PROGRESS,
    FAIL_SHUTDOWN,
)

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
        name = f"bv2evc_{tag}_{uuid.uuid4().hex[:8]}"
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


def _stats(box) -> dict[str, int]:
    return box.engine._reactor.stats()


# --------------------------------------------------------------------------------------------
# Feature A: event-driven reactor (completion wakeup fd)
# --------------------------------------------------------------------------------------------
class TestEventDrivenReactor:
    def test_wakeup_fd_mode_engaged_and_not_spinning(self, make_engine):
        """A1: with the real bindings the fd path engages and drives wakes.

        The reactor exposes the mode as a live ``_wakeup_rfd``; after a real
        transfer the ``reactor_wake_fd`` counter shows fd-delivered wakes
        (every C++ publish writes a token), and the idle poll timeout is
        deadline-driven — well above the legacy fixed 1 ms tick.
        """
        src, dst = _pair(make_engine)
        for box in (src, dst):
            assert box.engine._reactor._wakeup_rfd is not None, (
                "event-driven mode did not engage with the compiled poller"
            )

        case = _scattered_case(n_desc=64, desc_bytes=4 * KIB, seed=901)
        status = src.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, dst.name)
        assert status.wait(WAIT_CAP_MS), status.last_status_str()
        _assert_landed(case)

        # Completion publishes wrote tokens -> at least one poll returned
        # fd-ready (not by timeout) on the sender.
        _wait_until(
            lambda: _stats(src).get("reactor_wake_fd", 0) > 0,
            timeout_s=10.0,
            what="an fd-delivered reactor wake to be counted",
        )
        # Deadline-driven idle timeout: sampled over ~150 ms it must exceed
        # the 1 ms legacy tick (with request_timeout=30 s the sender sweep
        # bounds it at 100 ms; right after a sweep the full window is open).
        samples = []
        for _ in range(30):
            samples.append(src.engine._reactor._poll_timeout_ms())
            time.sleep(0.005)
        assert max(samples) > 5 * _POLL_MS, (
            f"idle poll timeout pinned near the legacy tick: samples={samples}"
        )

    def test_shutdown_with_inflight_closes_fds_no_crash(self, make_engine, fake_peers):
        """A3: teardown with the fd path active and work in flight.

        Submit against a silent peer (gathers/reservations in flight, future
        pending), then shut the engine down: no raise, the future resolves
        FAIL_SHUTDOWN, and both wakeup fds are closed behind the
        ``set_wakeup_fd(-1)`` fence — the poller's own shutdown (which
        publishes terminal rows AFTER the fence) must not touch them.
        """
        eng = make_engine("tx", request_timeout_ms=0, enable_cpp_chain=True)
        reactor = eng.engine._reactor
        rfd, wfd = reactor._wakeup_rfd, reactor._wakeup_wfd
        assert rfd is not None and wfd is not None
        silent = fake_peers(eng.engine._reactor.endpoint, "silent")
        blob = _forged_handshake(eng.engine._cfg.max_chunk_size_bytes, silent.endpoint)
        assert eng.engine.add_peer(silent.name, blob)

        case = _scattered_case(n_desc=64, desc_bytes=64 * KIB, seed=902)
        status = eng.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, silent.name)
        assert not status.is_completed()

        eng.engine.shutdown()  # idempotent with the fixture teardown
        assert status.wait(WAIT_CAP_MS) is False
        assert status.last_status_str() == FAIL_SHUTDOWN
        assert reactor._wakeup_rfd is None and reactor._wakeup_wfd is None
        for fd in {rfd, wfd}:
            with pytest.raises(OSError):
                os.fstat(fd)
        # A late Python-side wake after teardown is a silent no-op.
        reactor._wake()


# --------------------------------------------------------------------------------------------
# Feature B: two-phase C++ chain (reserve at gather launch, fulfill on credit)
# --------------------------------------------------------------------------------------------
class TestTwoPhaseChain:
    def test_gather_before_credit_takes_cpp_path(self, make_engine, fake_peers):
        """B1: the ordering the one-shot arm always LOST is now chained.

        A hoarding fake flow exhausts the receiver arena so the real
        request's GRANT is deterministically delayed until the hoard is
        cancelled — long after the sender's eager gathers completed. The
        credits must then FULFILL the reservations inline
        (``tx_chain_fulfilled_late``) with zero classic posts, and the data
        lands byte-exact.
        """
        cfg_kw = dict(
            arena_size_bytes=128 * MIB,
            max_chunk_size_bytes=8 * MIB,
            max_inflight_chunks_per_request=16,
            enable_cpp_chain=True,
        )
        src, dst = _pair(make_engine, **cfg_kw)
        hoard = fake_peers(dst.engine._reactor.endpoint, "hoard")

        # Hoard: one fake flow takes every region of the receiver arena.
        hoard_rid = 77
        hoard.send(codec.encode_want(hoard_rid, [8 * MIB] * 16, hoard.endpoint))
        got = hoard.recv(10.0)
        assert got is not None, "hoarding WANT was never granted"
        _, header, blob = got
        assert header.msg_type == codec.BounceMsgType.GRANT
        assert len(codec.decode_credits(blob, header)) == 16
        assert dst.engine._scheduler.free_bytes() == 0, "hoard did not exhaust the arena"

        # Real transfer: 4 x 8 MiB chunks; eager gathers launch at submit but
        # NO credit can arrive while the hoard holds the arena.
        case = _scattered_case(n_desc=512, desc_bytes=64 * KIB, seed=910)
        status = src.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, dst.name)
        _wait_until(
            lambda: _stats(src).get("tx_chain_reserved", 0) >= 4,
            timeout_s=10.0,
            what="every eager gather to take a chain reservation",
        )
        # Let the gathers FINISH (device drain) and the C++ poll thread mark
        # the reservations gather-done, all before any credit exists.
        torch.cuda.synchronize()
        time.sleep(0.2)
        assert not status.is_completed(), "transfer completed while the arena was hoarded"

        # Release the hoard: the receiver reclaims and grants the real flow.
        hoard.send(codec.encode_cancel(hoard_rid, hoard.endpoint))
        assert status.wait(WAIT_CAP_MS), status.last_status_str()
        _assert_landed(case)

        s = _stats(src)
        assert s.get("tx_chain_fulfilled_late", 0) >= 1, (
            f"no fulfill found the gather already done (the lost race): {s}"
        )
        assert s.get("tx_post_classic", 0) == 0, f"a chunk fell back to the classic post path: {s}"
        assert s.get("tx_chain_fulfill_declined", 0) == 0

    def test_reserved_timeout_reclaims_and_engine_recovers(self, make_engine, fake_peers):
        """B2a: reserve -> the request FAILS (no-progress) before any fulfill.

        Every chunk is reserved-GATHERING against a silent peer; the timeout
        sweep must cancel the reservations (gathers long done -> TERMINAL:
        regions recycle immediately), restore the arena in full, keep the
        reactor alive, and leave the engine fully serviceable.
        """
        cfg_kw = dict(request_timeout_ms=2000, enable_cpp_chain=True)
        eng = make_engine("tx", **cfg_kw)
        dst = make_engine("dst", **cfg_kw)  # for the healthy transfer after
        eng.agent.load_remote_agent(dst.name, dst.agent.get_local_agent_desc())
        assert eng.engine.add_peer(dst.name, dst.engine.local_handshake_blob())
        silent = fake_peers(eng.engine._reactor.endpoint, "silent")
        blob = _forged_handshake(eng.engine._cfg.max_chunk_size_bytes, silent.endpoint)
        assert eng.engine.add_peer(silent.name, blob)
        capacity = eng.engine._scheduler.arena_capacity

        case = _scattered_case(n_desc=256, desc_bytes=64 * KIB, seed=920)  # 2 x 8 MiB chunks
        status = eng.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, silent.name)
        assert status.wait(WAIT_CAP_MS) is False
        assert status.last_status_str() == FAIL_NO_PROGRESS
        assert _stats(eng).get("tx_chain_reserved", 0) >= 1, "the chain was never reserved"
        _wait_until(
            lambda: eng.engine._scheduler.local_held_count() == 0
            and eng.engine._scheduler.free_bytes() == capacity,
            timeout_s=10.0,
            what="the failed request's reserved staging regions to be released",
        )
        assert eng.engine._reactor.alive(), "reactor died cancelling the reservations"

        healthy = _scattered_case(n_desc=64, desc_bytes=4 * KIB, seed=921)
        status2 = eng.engine.submit(
            healthy.src_ptrs, healthy.dst_ptrs, healthy.sizes, DEVICE, dst.name
        )
        assert status2.wait(WAIT_CAP_MS), status2.last_status_str()
        _assert_landed(healthy)

    def test_reserved_forget_peer_no_hang_conserves(self, make_engine, fake_peers):
        """B2b: forget_peer right after submit cancels the reservations.

        Timeouts are DISABLED, so only the forget can resolve the future:
        no hang, a real terminal reason, full region conservation, live
        reactor. (cancel_chain returns TERMINAL for a finished gather and
        REVERTED for a still-pending one — both reclaim, whichever the
        timing produced.)
        """
        eng = make_engine("tx", request_timeout_ms=0, enable_cpp_chain=True)
        silent = fake_peers(eng.engine._reactor.endpoint, "silent")
        blob = _forged_handshake(eng.engine._cfg.max_chunk_size_bytes, silent.endpoint)
        assert eng.engine.add_peer(silent.name, blob)
        capacity = eng.engine._scheduler.arena_capacity

        case = _scattered_case(n_desc=256, desc_bytes=64 * KIB, seed=930)
        status = eng.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, silent.name)
        eng.engine.forget_peer(silent.name)
        assert status.wait(WAIT_CAP_MS) is False
        assert status.last_status_str() != "<bounce: pending>"
        _wait_until(
            lambda: eng.engine._scheduler.local_held_count() == 0
            and eng.engine._scheduler.free_bytes() == capacity,
            timeout_s=10.0,
            what="the forgotten request's reserved staging regions to be released",
        )
        assert eng.engine._reactor.alive()

    def test_reserved_shutdown_resolves_no_crash(self, make_engine, fake_peers):
        """B2c: poller-shutdown termination of reserved-unfulfilled chains.

        Shutdown with reservations in flight: the future resolves
        FAIL_SHUTDOWN and nothing crashes when the poller then sweeps its
        reserved entries terminally.
        """
        eng = make_engine("tx", request_timeout_ms=0, enable_cpp_chain=True)
        silent = fake_peers(eng.engine._reactor.endpoint, "silent")
        blob = _forged_handshake(eng.engine._cfg.max_chunk_size_bytes, silent.endpoint)
        assert eng.engine.add_peer(silent.name, blob)

        case = _scattered_case(n_desc=256, desc_bytes=64 * KIB, seed=940)
        status = eng.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, silent.name)
        assert _stats(eng).get("tx_chain_reserved", 0) >= 1
        eng.engine.shutdown()  # idempotent with the fixture teardown
        assert status.wait(WAIT_CAP_MS) is False
        assert status.last_status_str() == FAIL_SHUTDOWN

    def test_chain_off_stays_classic_byte_exact(self, make_engine):
        """B3: chain disabled -> zero reserve calls, pure classic path.

        Pins the byte-for-byte claim at the observable level: no chain
        counters exist at all, every chunk went through the classic
        gather-event -> post sequence, and the data is byte-exact.
        """
        src, dst = _pair(make_engine)  # enable_cpp_chain defaults False
        case = _scattered_case(n_desc=128, desc_bytes=64 * KIB, seed=950)  # one 8 MiB chunk
        status = src.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, dst.name)
        assert status.wait(WAIT_CAP_MS), status.last_status_str()
        _assert_landed(case)

        s = _stats(src)
        for key in (
            "tx_chain_reserved",
            "tx_chain_armed",
            "tx_chain_fulfilled_late",
            "tx_chain_fulfill_declined",
            "tx_chain_arm_race",
        ):
            assert s.get(key, 0) == 0, f"chain counter {key} moved with the chain OFF: {s}"
        assert s.get("tx_post_classic", 0) >= 1
        assert s.get("tx_gather_events", 0) >= 1  # gathers reported to Python

    def test_chained_transfers_exactly_once_accounting(self, make_engine):
        """B4: chained transfers publish exactly ONE terminal row per chunk.

        Three back-to-back transfers, then the counter ledger must balance on
        both sides: every sender chunk saw exactly one write completion and
        one ACK (a duplicate row would double tx_xfer_events / rx_data; a
        lost one would fail the wait), and the routing table is empty.
        """
        src, dst = _pair(
            make_engine,
            max_chunk_size_bytes=2 * MIB,
            max_inflight_chunks_per_request=4,
            enable_cpp_chain=True,
        )
        src_free0 = src.engine._scheduler.free_bytes()
        dst_free0 = dst.engine._scheduler.free_bytes()
        for i in range(3):
            case = _scattered_case(n_desc=96, desc_bytes=64 * KIB, seed=960 + i)  # 3 chunks
            status = src.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, dst.name)
            assert status.wait(WAIT_CAP_MS), f"round {i}: {status.last_status_str()}"
            _assert_landed(case)

        # Let both arenas drain (the receiver settles asynchronously past the
        # future resolve) so the ledger below is final.
        _wait_until(
            lambda: src.engine._scheduler.free_bytes() == src_free0
            and dst.engine._scheduler.free_bytes() == dst_free0,
            timeout_s=20.0,
            what="sender+receiver arenas to drain back to capacity",
        )
        s, d = _stats(src), _stats(dst)
        chunks = s["tx_chunks"]
        assert chunks == 9
        assert s.get("tx_chain_reserved", 0) >= 1, f"the chain never engaged: {s}"
        # Exactly-once, sender side: one write completion + one DATA + one
        # ACKed chunk per chunk.
        assert s["tx_xfer_events"] == chunks, s
        assert s["tx_data_sent"] == chunks, s
        assert s["tx_acked_chunks"] == chunks, s
        # Exactly-once, receiver side: one DATA -> one scatter -> one ACK.
        assert d["rx_data"] == chunks, d
        assert d["rx_scatter_ok"] == chunks, d
        assert d["rx_ack_entries"] == chunks, d
        # Nothing left routed or pending on the sender.
        with src.engine._reactor._req_mu:
            assert not src.engine._reactor._requests
            assert not src.engine._reactor._completions
