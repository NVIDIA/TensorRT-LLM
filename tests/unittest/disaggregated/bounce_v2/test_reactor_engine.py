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
"""GPU end-to-end tests for the bounce_v2 reactor + engine (Python transport).

Same skip guards as test_mechanism_bindings.py: requires a CUDA device and
the compiled tensorrt_llm wheel with the transfer-agent binding. Everything
above the binding (BounceEngine, BounceReactor, handshake, admission) is the
pure-Python transport; peers here are real NixlTransferAgents in one process,
plus raw pyzmq sockets acting as hostile/fake peers (white-box, mirroring the
C++ bounce transport tests).

Every wait is bounded: status.wait() always gets an explicit timeout, and all
polling loops run against a deadline. Engines/agents are torn down by the
``make_engine`` factory fixture in reverse creation order (engine.shutdown()
BEFORE agent.shutdown(), with a device drain in between) — a leaked reactor
or poller thread would hang pytest.
"""

from __future__ import annotations

import time
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("bounce_v2 reactor/engine tests require a CUDA device", allow_module_level=True)

tab = pytest.importorskip(
    "tensorrt_llm.tensorrt_llm_transfer_agent_binding",
    reason="bounce_v2 reactor/engine tests require the compiled tensorrt_llm wheel",
)

import zmq  # noqa: E402  (after the module-level skips)

from tensorrt_llm._torch.disaggregation.bounce_v2 import codec  # noqa: E402
from tensorrt_llm._torch.disaggregation.bounce_v2 import engine as engine_mod  # noqa: E402
from tensorrt_llm._torch.disaggregation.bounce_v2.config import BounceV2Config  # noqa: E402
from tensorrt_llm._torch.disaggregation.bounce_v2.engine import BounceEngine  # noqa: E402
from tensorrt_llm._torch.disaggregation.bounce_v2.plan import SCATTER_RUN_DTYPE  # noqa: E402
from tensorrt_llm._torch.disaggregation.bounce_v2.reactor import (  # noqa: E402
    FAIL_NO_PROGRESS,
    FAIL_PROTOCOL,
    FAIL_REACTOR_DEAD,
    FAIL_SHUTDOWN,
)

# Engines are created lazily inside the test body (configs vary per test),
# so their reactor/poller threads are still alive when pytest-threadleak
# checks at end-of-call (fixture teardown — which joins them — runs later).
# Teardown correctness is still enforced: a leaked reactor would hang the
# session, and shutdown() joins with a bounded timeout.
pytestmark = pytest.mark.threadleak(enabled=False)

DEVICE = 0
KIB = 1 << 10
MIB = 1 << 20
#: Generous cap for waits that MUST resolve much sooner; only guards CI hangs.
WAIT_CAP_MS = 60_000
#: Bounded NEGATIVE wait: long enough for a 1 ms-tick reactor to have acted.
NEGATIVE_WAIT_S = 1.5


def _cfg(**kw) -> BounceV2Config:
    """Small test config: 64 MiB arena / 8 MiB chunks unless overridden."""
    defaults = dict(
        enabled=True,
        arena_size_bytes=64 * MIB,
        arena_allocation_granularity_bytes=MIB,
        max_chunk_size_bytes=8 * MIB,
        max_inflight_chunks_per_request=4,
        copy_stream_count=2,
        min_descriptor_count=8,
        max_average_descriptor_size_bytes=16 * KIB,
        request_timeout_ms=30_000,
    )
    defaults.update(kw)
    return BounceV2Config(**defaults)


def _rand_bytes(n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, 256, (max(n, 1),), dtype=torch.uint8, generator=gen)[:n].cuda()


def _wait_until(pred, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.02)
    pytest.fail(f"timed out ({timeout_s}s) waiting for {what}")


def _drain_reactor_cmds(reactor, timeout_s: float = 5.0) -> None:
    """Bounded wait for the reactor command queue (forget_peer state reclaim).

    The queue is popped BEFORE the commands execute, so an empty queue
    alone does not prove the last
    command finished — additionally wait for a fresh tick (a new heartbeat
    means the popping tick completed). NOTE: forget_peer snapshots its victim
    rids at call time, so requests submitted after it can NEVER be reclaim
    victims — this drain is belt-and-braces quiescence (stable white-box
    scheduler/flow state before asserting on it), not a correctness
    prerequisite for the fresh submits that follow.
    """
    _wait_until(lambda: not reactor._cmds, timeout_s, "reactor command queue to empty")
    hb = reactor._heartbeat
    _wait_until(lambda: reactor._heartbeat != hb, timeout_s, "a fresh reactor tick")


def _forged_handshake(chunk_cap: int, endpoint: str, arena_cap: int = 64 * MIB) -> bytes:
    """Handshake blob a real peer engine would produce.

    White-box: uses the engine module's wire structs so the layout can
    never drift.
    """
    ep = endpoint.encode("utf-8")
    return (
        engine_mod._HANDSHAKE_HEADER.pack(
            engine_mod._HANDSHAKE_MAGIC,
            codec.BOUNCE_VERSION,
            engine_mod._CONTROL_KIND_ZMQ,
            chunk_cap,
            arena_cap,
            len(ep),
        )
        + ep
    )


def _scattered_case(n_desc: int, desc_bytes: int, seed: int, gap: int = 160):
    """Scattered src regions (random payload) + a zeroed dst, same layout.

    Regions start at odd offsets so nothing may rely on alignment.
    """
    spread = n_desc * (desc_bytes + gap) + 256
    src = _rand_bytes(spread, seed)
    dst = torch.zeros(spread, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    offs = [3 + i * (desc_bytes + gap) for i in range(n_desc)]
    src_ptrs = np.asarray([src.data_ptr() + o for o in offs], dtype=np.uint64)
    dst_ptrs = np.asarray([dst.data_ptr() + o for o in offs], dtype=np.uint64)
    sizes = np.full(n_desc, desc_bytes, dtype=np.uint64)
    return SimpleNamespace(
        src=src,
        dst=dst,
        offs=offs,
        nbytes=desc_bytes,
        src_ptrs=src_ptrs,
        dst_ptrs=dst_ptrs,
        sizes=sizes,
        spread=spread,
    )


def _assert_landed(case) -> None:
    """Byte-exact regions AND untouched inter-region gaps on the destination."""
    torch.cuda.synchronize()
    src_cpu, dst_cpu = case.src.cpu(), case.dst.cpu()
    covered = torch.zeros(case.spread, dtype=torch.bool)
    for o in case.offs:
        assert torch.equal(dst_cpu[o : o + case.nbytes], src_cpu[o : o + case.nbytes]), (
            f"region @{o} size {case.nbytes} differs"
        )
        covered[o : o + case.nbytes] = True
    assert not dst_cpu[~covered].any(), "bounce scatter wrote outside its target regions"


class FakePeer:
    """Raw pyzmq peer, mirroring the C++ white-box transport tests.

    A bound ROUTER (to receive GRANT/ACK the engine sends back to the
    endpoint we advertise) plus a DEALER into the target engine's reactor.
    """

    def __init__(self, target_endpoint: str, tag: str = "fake"):
        self.name = f"bv2_{tag}_{uuid.uuid4().hex[:8]}"
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
def make_engine():
    """Factory for (name, agent, engine) boxes.

    Teardown in reverse creation order: engine.shutdown() -> device drain
    -> agent.shutdown().
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
    """Two engines wired sender->receiver.

    NIXL metadata is loaded one-way (like the real Sender) and the bounce
    handshake registered on the sender.
    """
    src = make_engine("src", **cfg_kw)
    dst = make_engine("dst", **cfg_kw)
    src.agent.load_remote_agent(dst.name, dst.agent.get_local_agent_desc())
    assert src.engine.add_peer(dst.name, dst.engine.local_handshake_blob())
    return src, dst


# --------------------------------------------------------------------------------------------
# 1. WANT validation (hard review requirement)
# --------------------------------------------------------------------------------------------
class TestWantValidation:
    @pytest.mark.parametrize("bad_size_kind", ["zero", "oversize"])
    def test_invalid_want_ignored_no_grant_no_stall(self, make_engine, fake_peers, bad_size_kind):
        """A raw WANT with size=0 or size>arena capacity must be dropped.

        No GRANT within a bounded negative wait, no receiver-side arena
        consumption — and a subsequent VALID flow to the SAME receiver still
        gets granted (the invalid flow never entered drain mode).
        """
        rx = make_engine("rx")
        capacity = rx.engine._scheduler.arena_capacity
        free_before = rx.engine._scheduler.free_bytes()
        attacker = fake_peers(rx.engine._reactor.endpoint, "attacker")

        bad = 0 if bad_size_kind == "zero" else capacity + MIB
        attacker.send(codec.encode_want(1, [bad], attacker.endpoint))
        assert attacker.recv(NEGATIVE_WAIT_S) is None, (
            f"receiver granted an invalid WANT (size={bad})"
        )
        assert rx.engine._scheduler.free_bytes() == free_before
        assert rx.engine._scheduler.tracked_flows() == 0

        # Valid flow from the same (formerly hostile) endpoint still grants.
        attacker.send(codec.encode_want(2, [MIB], attacker.endpoint))
        got = attacker.recv(10.0)
        assert got is not None, "valid WANT after an invalid one was never granted (stall)"
        _, header, blob = got
        assert header.msg_type == codec.BounceMsgType.GRANT
        assert header.request_id == 2
        credits = codec.decode_credits(blob, header)
        assert credits is not None and len(credits) == 1
        assert credits[0].length == MIB

    def test_invalid_want_does_not_starve_real_transfer(self, make_engine, fake_peers):
        """Invalid WANTs must not starve later real transfers.

        After an invalid WANT hits the receiver, a REAL engine-to-engine
        transfer to that receiver still completes end to end.
        """
        src, dst = _pair(make_engine)
        attacker = fake_peers(dst.engine._reactor.endpoint, "attacker")
        capacity = dst.engine._scheduler.arena_capacity
        attacker.send(codec.encode_want(7, [0, MIB], attacker.endpoint))  # 0 poisons the list
        attacker.send(codec.encode_want(8, [capacity + MIB], attacker.endpoint))
        assert attacker.recv(NEGATIVE_WAIT_S) is None

        case = _scattered_case(n_desc=64, desc_bytes=4 * KIB, seed=11)
        status = src.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, dst.name)
        assert status.wait(WAIT_CAP_MS), status.last_status_str()
        _assert_landed(case)


# --------------------------------------------------------------------------------------------
# 2. happy path end to end
# --------------------------------------------------------------------------------------------
class TestHappyPath:
    def test_scattered_multi_desc_roundtrip(self, make_engine):
        src, dst = _pair(make_engine)
        case = _scattered_case(n_desc=256, desc_bytes=1000, seed=2026)
        assert src.engine.should_use(case.sizes, dst.name)
        status = src.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, dst.name)
        assert status.wait(WAIT_CAP_MS) is True, status.last_status_str()
        assert status.is_completed()
        assert status.last_status_str() == "SUCCESS"
        assert status.last_status() == 0
        _assert_landed(case)

    def test_back_to_back_requests_same_peer(self, make_engine):
        src, dst = _pair(make_engine)
        for i in range(3):
            case = _scattered_case(n_desc=64, desc_bytes=2048, seed=100 + i)
            status = src.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, dst.name)
            assert status.wait(WAIT_CAP_MS), f"round {i}: {status.last_status_str()}"
            _assert_landed(case)


# --------------------------------------------------------------------------------------------
# 3. multi-chunk streaming + region recycling
# --------------------------------------------------------------------------------------------
class TestMultiChunk:
    def test_many_chunks_byte_exact_and_regions_recycle(self, make_engine):
        # 2 MiB chunks over 20 MiB of payload -> >= 10 chunks with only 4
        # in flight: forces streaming through the credit window.
        src, dst = _pair(
            make_engine,
            arena_size_bytes=32 * MIB,
            max_chunk_size_bytes=2 * MIB,
            max_inflight_chunks_per_request=4,
        )
        src_free0 = src.engine._scheduler.free_bytes()
        dst_free0 = dst.engine._scheduler.free_bytes()
        assert src_free0 == src.engine._scheduler.arena_capacity

        case = _scattered_case(n_desc=320, desc_bytes=64 * KIB, seed=77)
        assert int(case.sizes.sum()) == 20 * MIB
        status = src.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, dst.name)
        assert status.wait(WAIT_CAP_MS), status.last_status_str()
        _assert_landed(case)

        # Region recycling: both arenas drain back to their initial free
        # bytes (sender frees on ACK, receiver on scatter completion; both
        # are async past the future resolve, so poll bounded).
        _wait_until(
            lambda: src.engine._scheduler.free_bytes() == src_free0
            and dst.engine._scheduler.free_bytes() == dst_free0,
            timeout_s=20.0,
            what="sender+receiver arenas to drain back to capacity",
        )
        assert src.engine._scheduler.local_held_count() == 0
        assert dst.engine._scheduler.tracked_flows() == 0


# --------------------------------------------------------------------------------------------
# 4. failure paths
# --------------------------------------------------------------------------------------------
class TestFailurePaths:
    def test_dead_endpoint_times_out_with_reason(self, make_engine, fake_peers):
        """A peer whose control endpoint never answers times out with a reason.

        wait() returns False within ~request_timeout and the reason names
        the no-progress cause.
        """
        eng = make_engine("tx", request_timeout_ms=3000)
        # A live socket that never GRANTs (worse than unreachable: it accepts).
        silent = fake_peers(eng.engine._reactor.endpoint, "silent")
        blob = _forged_handshake(eng.engine._cfg.max_chunk_size_bytes, silent.endpoint)
        assert eng.engine.add_peer(silent.name, blob)

        case = _scattered_case(n_desc=16, desc_bytes=4 * KIB, seed=5)
        t0 = time.monotonic()
        status = eng.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, silent.name)
        assert status.wait(WAIT_CAP_MS) is False
        elapsed = time.monotonic() - t0
        assert elapsed < 30.0, f"timeout took {elapsed:.1f}s for a 3s request_timeout"
        assert "no GRANT/ACK progress" in status.last_status_str()
        assert status.last_status_str() == FAIL_NO_PROGRESS
        # The eager-gathered staging regions must come back after the failure.
        _wait_until(
            lambda: eng.engine._scheduler.local_held_count() == 0,
            timeout_s=10.0,
            what="failed request's staging regions to be released",
        )

    def test_forget_peer_midflight_resolves_then_fresh_transfer_works(self, make_engine):
        src, dst = _pair(
            make_engine,
            arena_size_bytes=32 * MIB,
            max_chunk_size_bytes=2 * MIB,
            max_inflight_chunks_per_request=2,
        )
        case = _scattered_case(n_desc=320, desc_bytes=64 * KIB, seed=88)  # >=10 chunks
        status = src.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, dst.name)
        src.engine.forget_peer(dst.name)
        # The future must resolve either way (SUCCESS if it won the race,
        # FAILURE otherwise) — never hang.
        resolved = status.wait(WAIT_CAP_MS)
        assert status.is_completed()
        if not resolved:
            assert status.last_status_str()  # a real reason string, not pending
            assert status.last_status_str() != "<bounce: pending>"
        assert not src.engine.has_peer(dst.name)
        assert not src.engine.should_use(case.sizes, dst.name)

        # Let forget_peer's async state reclaim finish before re-adding. The
        # rid snapshot means the fresh request below could never be a victim;
        # this is only belt-and-braces quiescence (see _drain_reactor_cmds).
        _drain_reactor_cmds(src.engine._reactor)

        # Re-add the peer: fresh transfers must fully work again.
        assert src.engine.add_peer(dst.name, dst.engine.local_handshake_blob())
        case2 = _scattered_case(n_desc=64, desc_bytes=8 * KIB, seed=89)
        status2 = src.engine.submit(case2.src_ptrs, case2.dst_ptrs, case2.sizes, DEVICE, dst.name)
        assert status2.wait(WAIT_CAP_MS), status2.last_status_str()
        _assert_landed(case2)

    def test_handshake_mismatch_disables_peer(self, make_engine):
        src, dst = _pair(make_engine)
        good = dst.engine.local_handshake_blob()
        parsed = BounceEngine._decode_handshake(good)
        assert parsed is not None
        version, kind, chunk_cap, arena_cap, endpoint = parsed
        tampered = _forged_handshake(chunk_cap + MIB, endpoint, arena_cap)

        # add_peer is REPLACEMENT: the incompatible re-registration must both
        # return False and drop the previously working route.
        assert src.engine.add_peer(dst.name, tampered) is False
        assert not src.engine.has_peer(dst.name)
        sizes = np.full(64, 1024, dtype=np.uint64)
        assert src.engine.should_use(sizes, dst.name) is False  # NIXL fallback

        # Other rejects: garbage blob, empty blob, wrong version.
        assert src.engine.add_peer(dst.name, b"\x00" * 8) is False
        assert src.engine.add_peer(dst.name, b"") is False
        assert src.engine.add_peer(dst.name, None) is False
        bad_version = bytearray(good)
        bad_version[4] ^= 0xFF  # u16 version field of the handshake header
        assert src.engine.add_peer(dst.name, bytes(bad_version)) is False

        # Route removal is synchronous and the state reclaim snapshots its
        # victim rids, so the fresh submit below is safe regardless; drain the
        # queued reclaim anyway for quiescence (see _drain_reactor_cmds).
        _drain_reactor_cmds(src.engine._reactor)

        # And the pristine blob still restores service.
        assert src.engine.add_peer(dst.name, good) is True
        case = _scattered_case(n_desc=32, desc_bytes=2048, seed=6)
        status = src.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, dst.name)
        assert status.wait(WAIT_CAP_MS), status.last_status_str()
        _assert_landed(case)

    def test_shutdown_with_inflight_resolves_failure(self, make_engine, fake_peers):
        # request_timeout_ms=0 disables the sweep: only shutdown can resolve it.
        eng = make_engine("tx", request_timeout_ms=0)
        silent = fake_peers(eng.engine._reactor.endpoint, "silent")
        blob = _forged_handshake(eng.engine._cfg.max_chunk_size_bytes, silent.endpoint)
        assert eng.engine.add_peer(silent.name, blob)

        case = _scattered_case(n_desc=16, desc_bytes=4 * KIB, seed=9)
        status = eng.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, silent.name)
        assert not status.is_completed()
        eng.engine.shutdown()  # idempotent with the fixture teardown
        assert status.wait(WAIT_CAP_MS) is False
        assert status.last_status_str() == FAIL_SHUTDOWN
        assert "shut down" in status.last_status_str()
        # Post-shutdown submits resolve immediately with the same reason.
        late = eng.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, silent.name)
        assert late.wait(WAIT_CAP_MS) is False
        assert late.last_status_str() == FAIL_SHUTDOWN

    def test_grant_mispair_fails_immediately_with_protocol_error(self, make_engine, fake_peers):
        """An undersized credit must abandon the flow immediately.

        A hostile/buggy receiver grants a credit SMALLER than the chunk it
        must back: the sender must abandon the flow IMMEDIATELY with
        FAIL_PROTOCOL — not via the timeout sweep (request_timeout_ms=0 here,
        so only the immediate-abandon path can resolve the future).
        """
        eng = make_engine("tx", request_timeout_ms=0)
        fake = fake_peers(eng.engine._reactor.endpoint, "rxfake")
        blob = _forged_handshake(eng.engine._cfg.max_chunk_size_bytes, fake.endpoint)
        assert eng.engine.add_peer(fake.name, blob)

        case = _scattered_case(n_desc=64, desc_bytes=64 * KIB, seed=13)  # one 4 MiB chunk
        status = eng.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, fake.name)
        got = fake.recv(10.0)
        assert got is not None, "no WANT reached the fake receiver"
        _, header, want_blob = got
        assert header.msg_type == codec.BounceMsgType.WANT
        rid = header.request_id
        chunk_sizes, _ep = codec.decode_want(want_blob, header)
        assert len(chunk_sizes) == 1

        # Undersized credit: an RDMA write of chunk_sizes[0] bytes into it
        # would overflow into an adjacent region on the receiver.
        undersized = codec.CreditEntry(
            addr=1 << 20, length=chunk_sizes[0] // 2, dev_id=DEVICE, region_handle=0
        )
        t0 = time.monotonic()
        fake.send(codec.encode_grant(rid, [undersized]))
        assert status.wait(WAIT_CAP_MS) is False
        assert time.monotonic() - t0 < 10.0, "mispair resolved via timeout, not immediately"
        assert status.last_status_str() == FAIL_PROTOCOL
        # The abandoned flow's staging regions must come back.
        _wait_until(
            lambda: eng.engine._scheduler.local_held_count() == 0,
            timeout_s=10.0,
            what="abandoned request's staging regions to be released",
        )

    def test_reactor_watchdog_unblocks_waiter(self, make_engine, fake_peers):
        """Watchdog (design risk #2): a dead reactor may never hang a waiter.

        A waiter blocked on a request must
        unblock once the reactor is marked dead, even with timeouts disabled.
        There is no thread-kill test hook; we flip the reactor's existing
        crash flag (`_dead`, the exact bit its exception boundary sets), which
        exercises the same waiter-side alive() polling path. Hard thread-death
        (the loop actually gone) is NOT reachable without a hook — not added
        per test rules.
        """
        eng = make_engine("tx", request_timeout_ms=0)
        silent = fake_peers(eng.engine._reactor.endpoint, "silent")
        blob = _forged_handshake(eng.engine._cfg.max_chunk_size_bytes, silent.endpoint)
        assert eng.engine.add_peer(silent.name, blob)

        case = _scattered_case(n_desc=16, desc_bytes=4 * KIB, seed=10)
        status = eng.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, silent.name)
        eng.engine._reactor._dead = True  # simulate the crash boundary having run
        t0 = time.monotonic()
        assert status.wait(WAIT_CAP_MS) is False
        assert time.monotonic() - t0 < 10.0  # a couple of 1s watchdog slices
        assert status.last_status_str() == FAIL_REACTOR_DEAD
        # A dead reactor also closes admission and refuses new submissions.
        assert eng.engine.should_use(case.sizes, silent.name) is False
        late = eng.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, silent.name)
        assert late.wait(WAIT_CAP_MS) is False
        assert late.last_status_str() == FAIL_REACTOR_DEAD


# --------------------------------------------------------------------------------------------
# 5. admission (should_use thresholds)
# --------------------------------------------------------------------------------------------
class TestAdmission:
    def test_should_use_thresholds(self, make_engine, fake_peers):
        eng = make_engine(
            "tx",
            min_descriptor_count=8,
            max_average_descriptor_size_bytes=1024,
            max_chunk_size_bytes=8 * MIB,
        )
        peer = fake_peers(eng.engine._reactor.endpoint, "peer")
        blob = _forged_handshake(8 * MIB, peer.endpoint)
        assert eng.engine.add_peer(peer.name, blob)

        ok = np.full(8, 1024, dtype=np.uint64)
        assert eng.engine.should_use(ok, peer.name) is True
        # Descriptor count below the minimum.
        assert eng.engine.should_use(ok[:7], peer.name) is False
        assert eng.engine.should_use(np.empty(0, dtype=np.uint64), peer.name) is False
        # Average descriptor size above the maximum.
        assert eng.engine.should_use(np.full(8, 1025, dtype=np.uint64), peer.name) is False
        # Average is what gates, not the max: one big desc amortized is fine...
        mixed = np.full(64, 1, dtype=np.uint64)
        mixed[0] = 32 * KIB
        assert eng.engine.should_use(mixed, peer.name) is True
        # ...unless a single descriptor exceeds one chunk (16384 descs keep
        # the average at ~513 B, so ONLY the per-desc chunk cap can reject).
        over = np.full(16384, 1, dtype=np.uint64)
        over[0] = 8 * MIB + 1
        assert eng.engine.should_use(over, peer.name) is False
        # No handshaked peer -> no bounce.
        assert eng.engine.should_use(ok, "never_handshaked_peer") is False


# --------------------------------------------------------------------------------------------
# 6. duplicate DATA / stale ACK robustness (raw-socket injection)
# --------------------------------------------------------------------------------------------
class TestProtocolRobustness:
    def test_stale_ack_ignored_and_service_continues(self, make_engine, fake_peers):
        src, dst = _pair(make_engine)
        # Hostile ACKs straight into the SENDER's reactor: unknown rid and a
        # junk entry — both must be dropped without disturbing anything.
        hostile = fake_peers(src.engine._reactor.endpoint, "hostile")
        hostile.send(codec.encode_ack(999_999, [(0, 0)]))
        hostile.send(codec.encode_ack(0, [(123, 456)]))
        time.sleep(0.2)  # let the reactor ingest them (bounded, then verified)
        case = _scattered_case(n_desc=64, desc_bytes=2048, seed=12)
        status = src.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, dst.name)
        assert status.wait(WAIT_CAP_MS), status.last_status_str()
        _assert_landed(case)

    def test_duplicate_data_not_double_acked(self, make_engine, fake_peers):
        """A replayed DATA earns no second ACK and no crash.

        Full raw-socket handshake against a live receiver: WANT -> GRANT ->
        DATA -> exactly one ACK; then the identical DATA is replayed for the
        same (now freed) region.
        """
        rx = make_engine("rx")
        fake = fake_peers(rx.engine._reactor.endpoint, "sender")
        dst_buf = torch.zeros(64 * KIB, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()

        rid = 21
        fake.send(codec.encode_want(rid, [64 * KIB], fake.endpoint))
        got = fake.recv(10.0)
        assert got is not None, "no GRANT for a valid WANT"
        _, header, blob = got
        assert header.msg_type == codec.BounceMsgType.GRANT and header.request_id == rid
        credits = codec.decode_credits(blob, header)
        assert credits is not None and len(credits) == 1
        credit = credits[0]

        piece, count = 4 * KIB, 4
        run = np.zeros(1, dtype=SCATTER_RUN_DTYPE)
        run[0] = (0, dst_buf.data_ptr(), piece, piece, piece, count)
        data = codec.encode_data(rid, 0, 1, credit.region_handle, run)
        fake.send(data)
        got = fake.recv(10.0)
        assert got is not None, "no ACK for the scattered DATA"
        _, header, blob = got
        assert header.msg_type == codec.BounceMsgType.ACK and header.request_id == rid
        entries = codec.decode_ack(blob, header)
        assert entries == [codec.AckEntry(0, credit.region_handle)]

        # Replay the identical DATA: the flow completed and the region was
        # freed, so it must be dropped — bounded negative wait for a 2nd ACK.
        fake.send(data)
        assert fake.recv(NEGATIVE_WAIT_S) is None, "duplicate DATA was ACKed again"

        # The receiver is still healthy: a fresh flow completes.
        rid2 = 22
        fake.send(codec.encode_want(rid2, [16 * KIB], fake.endpoint))
        got = fake.recv(10.0)
        assert got is not None and got[1].msg_type == codec.BounceMsgType.GRANT
        _wait_until(
            lambda: rx.engine._scheduler.tracked_flows() >= 1,
            timeout_s=5.0,
            what="the fresh flow to be tracked",
        )

    def test_data_with_out_of_bounds_run_rejected_no_ack(self, make_engine, fake_peers):
        """A scatter run reaching past its granted region is rejected.

        No scatter, NO ACK (the sender must time out rather than believe the
        data landed), and the region is released rather than leaked.
        """
        rx = make_engine("rx")
        fake = fake_peers(rx.engine._reactor.endpoint, "sender")
        free0 = rx.engine._scheduler.free_bytes()
        dst_buf = torch.zeros(64 * KIB, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()

        rid = 31
        fake.send(codec.encode_want(rid, [64 * KIB], fake.endpoint))
        got = fake.recv(10.0)
        assert got is not None
        _, header, blob = got
        credit = codec.decode_credits(blob, header)[0]

        region_bytes = rx.engine._scheduler.region_bytes(credit.region_handle)
        run = np.zeros(1, dtype=SCATTER_RUN_DTYPE)
        # bounce_offset walks off the end of the granted buddy block.
        run[0] = (region_bytes - 1, dst_buf.data_ptr(), 4 * KIB, 4 * KIB, 4 * KIB, 2)
        fake.send(codec.encode_data(rid, 0, 1, credit.region_handle, run))
        assert fake.recv(NEGATIVE_WAIT_S) is None, "out-of-bounds scatter was ACKed"
        _wait_until(
            lambda: rx.engine._scheduler.free_bytes() == free0,
            timeout_s=10.0,
            what="the rejected region to be released back to the arena",
        )
        assert not dst_buf.any(), "rejected scatter still wrote to the destination"
        # No-leak (round-2 NB fix): a no-ACK terminal must still settle the
        # flow's chunk accounting — the (peer, rid) entry may not linger in
        # the receiver reactor's _rx_flows forever.
        key = rx.engine._reactor._make_key(fake.name, rid)
        _wait_until(
            lambda: key not in rx.engine._reactor._rx_flows,
            timeout_s=10.0,
            what="the rejected flow's _rx_flows entry to be dropped",
        )
