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
"""GPU tests for the per-request plan handle + chained credited launches.

Pins the C++ gather->RDMA chain end to end with the real bindings: credited
launches go through ``launch_chunk_chained`` (ONE completion row per chunk
under the reserved xfer id, zero gather rows reaching Python), plan handles
are released on every terminal path, and the exactly-once ledger balances.

Same skip guards as test_mechanism_bindings.py; shares its helpers/fixtures
pattern with test_reactor_engine.py (imported as a sibling test module).
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("bounce_v2 chained-launch tests require a CUDA device", allow_module_level=True)

tab = pytest.importorskip(
    "tensorrt_llm.tensorrt_llm_transfer_agent_binding",
    reason="bounce_v2 chained-launch tests require the compiled tensorrt_llm wheel",
)

from test_reactor_engine import (  # noqa: E402  (sibling test module)
    DEVICE,
    KIB,
    MIB,
    WAIT_CAP_MS,
    _assert_landed,
    _cfg,
    _scattered_case,
    _wait_until,
)

from tensorrt_llm._torch.disaggregation.bounce_v2.engine import BounceEngine  # noqa: E402

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


def _pair(make_engine, **cfg_kw):
    src = make_engine("src", **cfg_kw)
    dst = make_engine("dst", **cfg_kw)
    src.agent.load_remote_agent(dst.name, dst.agent.get_local_agent_desc())
    assert src.engine.add_peer(dst.name, dst.engine.local_handshake_blob())
    return src, dst


def _stats(box) -> dict[str, int]:
    return box.engine._reactor.stats()


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
