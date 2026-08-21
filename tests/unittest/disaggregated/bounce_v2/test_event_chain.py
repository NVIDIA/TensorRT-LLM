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
"""GPU tests for the event-driven reactor fd (completion wakeup fd).

Default-on when the compiled poller exposes
``set_wakeup_fd`` and the fd setup succeeds): the reactor parks in a
deadline-driven poll (up to 100 ms) instead of the legacy fixed 1 ms tick and
is woken by an fd token on every C++ publish/retire and every cross-thread
Python command. Pins: the mode engages with the real bindings, wakes are
fd-delivered (``reactor_wake_fd`` counter), the idle timeout is not pinned at
1 ms, and shutdown closes the fds behind the ``set_wakeup_fd(-1)`` fence even
with copies in flight.

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
    pytest.skip(
        "bounce_v2 event-driven reactor tests require a CUDA device", allow_module_level=True
    )

tab = pytest.importorskip(
    "tensorrt_llm.tensorrt_llm_transfer_agent_binding",
    reason="bounce_v2 event-driven reactor tests require the compiled tensorrt_llm wheel",
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

from tensorrt_llm._torch.disaggregation.bounce_v2.engine import BounceEngine  # noqa: E402
from tensorrt_llm._torch.disaggregation.bounce_v2.reactor import (  # noqa: E402
    _POLL_MS,
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

        Submit against a silent peer (gathers in flight, future
        pending), then shut the engine down: no raise, the future resolves
        FAIL_SHUTDOWN, and both wakeup fds are closed behind the
        ``set_wakeup_fd(-1)`` fence — the poller's own shutdown (which
        publishes terminal rows AFTER the fence) must not touch them.
        """
        eng = make_engine("tx", request_timeout_ms=0)
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
# Round 45: per-request plan handle + chained credited launches (P2)
#
# NOTE (P3): a REAL chained gather failure (an invalid source pointer inside
# a registered plan) is deliberately NOT exercised here — an async gather
# kernel faulting on a bogus device address poisons the CUDA context for the
# rest of the suite, and there is no main-code fault-injection hook. The
# (reserved, KIND_EVENT, 0) -> FAIL_GATHER row mapping and its region
# conservation are pinned GPU-free with fakes instead
# (test_pump_convoy.py::TestPlanHandleFakes).
# --------------------------------------------------------------------------------------------
class TestPlanHandleTransfers:
    def test_credit_first_launches_chain_and_ledger_balances(self, make_engine):
        """P2a: with credits preceding every launch, all launches CHAIN.

        Eager gather is disabled, so no chunk can launch before its GRANT:
        every launch goes through agent.launch_chunk_chained (gather +
        C++-auto-posted RDMA write; ONE row per chunk under the reserved id).
        Pins tx_chained_launches == tx_chunks, ZERO gather rows reaching
        Python, the exactly-once ledger, and byte-exact data.
        """
        src, dst = _pair(
            make_engine,
            enable_eager_gather=False,
            max_chunk_size_bytes=2 * MIB,
            max_inflight_chunks_per_request=4,
        )
        case = _scattered_case(n_desc=128, desc_bytes=64 * KIB, seed=451)  # 8 MiB -> 4 chunks
        status = src.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, dst.name)
        assert status.wait(WAIT_CAP_MS), status.last_status_str()
        _assert_landed(case)

        s = _stats(src)
        assert s["tx_chunks"] == 4
        assert s.get("tx_chained_launches", 0) == 4, f"credited launches did not chain: {s}"
        assert s.get("tx_gather_events", 0) == 0, f"a gather row leaked to Python: {s}"
        # Exactly-once ledger: one write row + one DATA + one ACKed chunk per
        # chunk (the chained row publishes under the reserved xfer id).
        assert s["tx_xfer_events"] == 4, s
        assert s["tx_data_sent"] == 4, s
        assert s["tx_acked_chunks"] == 4, s

    def test_plan_handles_do_not_leak_across_transfers(self, make_engine):
        """P2b: plan handles are released on every terminal path.

        The pool exposes no live-plan count, so the observable proxy is:
        back-to-back transfers (default eager config) all succeed byte-exact,
        the sender's request/routing tables drain empty, both arenas return
        to capacity, and the fixture's engine shutdown (pool destruction with
        its release backstop) is clean.
        """
        src, dst = _pair(
            make_engine, max_chunk_size_bytes=2 * MIB, max_inflight_chunks_per_request=4
        )
        src_free0 = src.engine._scheduler.free_bytes()
        dst_free0 = dst.engine._scheduler.free_bytes()
        for i in range(3):
            case = _scattered_case(n_desc=96, desc_bytes=64 * KIB, seed=460 + i)  # 3 chunks
            status = src.engine.submit(case.src_ptrs, case.dst_ptrs, case.sizes, DEVICE, dst.name)
            assert status.wait(WAIT_CAP_MS), f"round {i}: {status.last_status_str()}"
            _assert_landed(case)
        _wait_until(
            lambda: src.engine._scheduler.free_bytes() == src_free0
            and dst.engine._scheduler.free_bytes() == dst_free0,
            timeout_s=20.0,
            what="sender+receiver arenas to drain back to capacity",
        )
        s = _stats(src)
        assert s["tx_chunks"] == s["tx_xfer_events"] == s["tx_acked_chunks"] == 9, s
        with src.engine._reactor._req_mu:
            assert not src.engine._reactor._requests
            assert not src.engine._reactor._completions
        assert src.engine._reactor.alive()
