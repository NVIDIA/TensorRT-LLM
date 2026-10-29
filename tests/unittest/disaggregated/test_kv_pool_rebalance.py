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
"""KV cache pool rebalance on disaggregated context servers.

CPU tests drive PyExecutor's rebalance gate, transfer drain, veto, early skip
and back-off with stubs, and check the cross-node veto's inputs. The GPU test runs kv_pool_rebalance_worker.py as two
single-rank ``mpirun`` processes (ctx and gen) on one GPU: ctx resizes its pools
between transfers, then sends from slots past the old registered extent, and gen
verifies the data bit-exact. Its ``skiptail`` variant leaves the grown tail
unregistered and expects the send to fail, which shows the check is effective.
"""

import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import types

import pytest

import tensorrt_llm._torch.pyexecutor.py_executor as pe
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar, _endpoint_host
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

PE = pe.PyExecutor

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kv_pool_rebalance_worker.py")
_PROCESS_TIMEOUT_SECONDS = 180
_TERMINATE_GRACE_SECONDS = 5
_LOG_TAIL_CHARS = 16 * 1024

# ---------------------------------------------------------------------------
# CPU: gate, drain, veto, early skip, back-off
# ---------------------------------------------------------------------------


class _Xcvr:
    """Transceiver stub: veto (layout sharing), in-flight sends and peer-wait age."""

    def __init__(self, shared=False, sessions=0, reason=None, peer_wait=0.0, peer_wait_step=0.0):
        self.shared, self.sessions, self.reason = shared, sessions, reason
        self.peer_wait, self.peer_wait_step = peer_wait, peer_wait_step

    def kv_pool_rebalance_unsupported_reason(self):
        return self.reason

    def kv_pool_layout_shared_with_peers(self):
        return self.shared

    def kv_pool_rebalance_veto_reason(self, allow_cross_node=False):
        return "a peer holds this rank's pool layout" if self.shared else None

    def has_inflight_transfers(self):
        return self.sessions > 0

    def oldest_send_waiting_for_peer_s(self):
        wait = self.peer_wait
        self.peer_wait += self.peer_wait_step
        return wait


def _executor(xcvr, peers_states=()):
    """PyExecutor stand-in; peers_states are other ranks' (veto, busy, expired, peer_wait)."""
    s = types.SimpleNamespace()
    s.kv_cache_transceiver = xcvr
    s._disagg_rebalance_disabled = False
    s._disagg_rebalance_skips = 0
    s._disagg_rebalance_retry_at = 0.0
    s._kv_pool_rebalance_allow_cross_node = False
    s.reaps = 0
    s.flushes = 0
    s.async_transfer_manager = types.SimpleNamespace(has_any_inflight_requests=lambda: False)
    s.dist = types.SimpleNamespace(
        allgather=lambda st: [st, *peers_states], cp_size=1, tp_size=1, pp_size=1
    )

    def reap(n):
        s.reaps += 1
        if xcvr is not None and xcvr.sessions:
            xcvr.sessions -= 1

    s.disagg = types.SimpleNamespace(reap_context_sends=reap)
    s._flush_pending_transfer_responses = lambda: setattr(s, "flushes", s.flushes + 1)
    s._note_disagg_drain_outcome = lambda skipped: PE._note_disagg_drain_outcome(s, skipped)
    s._in_disagg_rebalance_backoff = lambda: PE._in_disagg_rebalance_backoff(s)
    return s


def _drain(s):
    return PE._drain_disagg_transfers_for_rebalance(s)


def test_drain_without_transceiver_or_sends():
    s = _executor(None)
    assert _drain(s) and s.reaps == 0
    s = _executor(_Xcvr())
    assert _drain(s) and s.reaps == 0


def test_drain_reaps_inflight_sends():
    s = _executor(_Xcvr(sessions=3))
    assert _drain(s) and s.reaps == 3 and s.flushes == 3


@pytest.mark.parametrize(
    "xcvr, peers",
    [
        (_Xcvr(shared=True, sessions=2), ()),
        (_Xcvr(), [("peer on another node", False, False, False)]),
    ],
    ids=["this_rank_vetoes", "peer_vetoes"],
)
def test_veto_disables_rebalance(xcvr, peers):
    s = _executor(xcvr, peers)
    assert not _drain(s) and s._disagg_rebalance_disabled and s.reaps == 0


def test_peer_deadline_skips_without_disabling():
    s = _executor(_Xcvr(), [(None, True, True, False)])
    assert not _drain(s) and s.reaps == 0 and not s._disagg_rebalance_disabled


def test_early_skip_on_send_waiting_for_peer():
    s = _executor(_Xcvr(sessions=5, peer_wait=0.3))
    t0 = time.monotonic()
    assert not _drain(s) and s.reaps == 0 and time.monotonic() - t0 < 0.05
    assert not s._disagg_rebalance_disabled and s._disagg_rebalance_skips == 1
    # Another rank's waiting send skips an idle rank too.
    s = _executor(_Xcvr(), [(None, True, False, True)])
    assert not _drain(s) and s.reaps == 0 and s._disagg_rebalance_skips == 1


def test_early_skip_threshold():
    s = _executor(_Xcvr(sessions=2, peer_wait=0.1))
    assert _drain(s) and s.reaps == 2
    # A send crossing the threshold mid-drain stops it at that pass.
    s = _executor(_Xcvr(sessions=10, peer_wait=0.0, peer_wait_step=0.1))
    assert not _drain(s) and s.reaps == 3


def test_veto_wins_over_early_skip():
    s = _executor(_Xcvr(shared=True, sessions=1, peer_wait=1.0))
    assert not _drain(s) and s._disagg_rebalance_disabled


def test_backoff_doubles_and_resets():
    s = _executor(_Xcvr(sessions=1, peer_wait=1.0))
    expected = [1, 2, 4, 8, 16, 32, 60, 60]
    for want in expected:
        s.kv_cache_transceiver.sessions = 1
        t = time.monotonic()
        assert not _drain(s)
        assert abs((s._disagg_rebalance_retry_at - t) - want) < 0.05
    assert s._disagg_rebalance_skips == len(expected) and s._in_disagg_rebalance_backoff()
    s.kv_cache_transceiver.sessions, s.kv_cache_transceiver.peer_wait = 0, 0.0
    assert _drain(s) and s._disagg_rebalance_skips == 0 and not s._in_disagg_rebalance_backoff()


def test_drain_deadline_backs_off(monkeypatch):
    monkeypatch.setattr(pe, "KV_POOL_REBALANCE_DISAGG_DRAIN_S", 0.05)
    s = _executor(_Xcvr(sessions=10**9))
    assert not _drain(s) and s.reaps > 0
    assert s._disagg_rebalance_skips == 1 and s._in_disagg_rebalance_backoff()


def test_backoff_gates_agreed_need_adjustment():
    mgr = types.SimpleNamespace(impl=types.SimpleNamespace(need_adjustment=True))
    s = _executor(_Xcvr())
    s.kv_cache_manager, s.enable_attention_dp = mgr, False
    assert PE._agreed_need_adjustment(s)
    s._disagg_rebalance_retry_at = time.monotonic() + 10
    assert not PE._agreed_need_adjustment(s)
    s._disagg_rebalance_retry_at = time.monotonic() - 1
    assert PE._agreed_need_adjustment(s)
    mgr.impl.need_adjustment = False
    assert not PE._agreed_need_adjustment(s)


@pytest.mark.parametrize(
    "xcvr, disabled, expected",
    [
        (None, False, True),
        (_Xcvr(), False, True),
        (_Xcvr(reason="pipelined"), False, False),
        (_Xcvr(), True, False),
    ],
    ids=["no_transceiver", "supported", "unsupported_reason", "disabled_flag"],
)
def test_rebalance_gate(xcvr, disabled, expected):
    g = types.SimpleNamespace(
        enable_kv_pool_rebalance=True,
        kv_cache_transceiver=xcvr,
        _disagg_rebalance_disabled=disabled,
        kv_connector_manager=None,
        is_warmup=False,
        is_shutdown=False,
        drafter=None,
        iter_counter=0,
        _rebalance_check_interval=10,
    )
    assert PE._can_pause_for_rebalance(g) is expected


def test_oldest_send_waiting_for_peer():
    now = time.monotonic()

    def sess(status, age):
        return types.SimpleNamespace(status=status, created_monotonic=now - age)

    x = types.SimpleNamespace(_send_sessions={})
    assert KvCacheTransceiverV2.oldest_send_waiting_for_peer_s(x) == 0.0
    x._send_sessions = {
        1: sess(SessionStatus.READY, 5.0),
        2: sess(SessionStatus.TRANSFERRING, 9.0),
        3: sess(SessionStatus.INIT, 0.7),
        4: sess(SessionStatus.INIT, 0.2),
    }
    assert 0.7 <= KvCacheTransceiverV2.oldest_send_waiting_for_peer_s(x) < 0.75


@pytest.mark.parametrize(
    "endpoint, host",
    [
        ("tcp://10.52.32.4:44969", "10.52.32.4"),
        ("tcp://node-a:5000", "node-a"),
        ("", None),
        (None, None),
    ],
)
def test_endpoint_host(endpoint, host):
    assert _endpoint_host(endpoint) == host


def test_peer_registrar_notes_cross_node_peer():
    reg = object.__new__(PeerRegistrar)
    reg._ri = types.SimpleNamespace(self_endpoint="tcp://10.0.0.1:5000")
    reg._has_cross_node_peer = False
    for endpoint in ("tcp://10.0.0.1:6000", ""):
        reg._note_peer_host(types.SimpleNamespace(self_endpoint=endpoint))
        assert not reg.has_cross_node_peer
    reg._note_peer_host(types.SimpleNamespace(self_endpoint="tcp://10.0.0.2:5000"))
    assert reg.has_cross_node_peer
    # Never reverts.
    reg._note_peer_host(types.SimpleNamespace(self_endpoint="tcp://10.0.0.1:7000"))
    assert reg.has_cross_node_peer


@pytest.mark.parametrize(
    "registered, cross_node, allow, expected",
    [
        (False, False, False, None),
        (False, True, False, "another node"),
        (False, True, True, None),
        (True, False, False, "receives KV"),
        (True, True, True, "receives KV"),
    ],
    ids=["none", "cross_node", "cross_node_allowed", "receives_kv", "receives_kv_wins"],
)
def test_transceiver_veto_reason(registered, cross_node, allow, expected):
    x = object.__new__(KvCacheTransceiverV2)
    x._transfer_worker = types.SimpleNamespace(
        has_registered_with_peers=registered, has_cross_node_peer=cross_node
    )
    reason = x.kv_pool_rebalance_veto_reason(allow_cross_node=allow)
    assert (reason is None) if expected is None else (expected in reason)


# ---------------------------------------------------------------------------
# GPU: resize between transfers, then send from the grown tail (1 GPU)
# ---------------------------------------------------------------------------


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _log_tail(log: str) -> str:
    if len(log) <= _LOG_TAIL_CHARS:
        return log
    return (
        f"... {len(log) - _LOG_TAIL_CHARS} earlier characters omitted ...\n{log[-_LOG_TAIL_CHARS:]}"
    )


def _kill_group(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass


@pytest.mark.timeout(300)
@pytest.mark.parametrize("mode", ["refresh", "skiptail"])
def test_rebalance_integrity(mode, tmp_path):
    """refresh: sends past the old extent arrive bit-exact. skiptail: they fail."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("mpi4py")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not found on PATH")

    env = os.environ.copy()
    env.update({"MODE": mode, "ZMQ_PORT": str(_find_free_port()), "CUDA_VISIBLE_DEVICES": "0"})
    cmd = [mpirun, "--allow-run-as-root", "--oversubscribe", "-np", "1", sys.executable, WORKER]
    logs = {role: tmp_path / f"{role}.log" for role in ("ctx", "gen")}
    procs = {}
    try:
        for role in ("gen", "ctx"):
            with open(logs[role], "wb") as f:
                procs[role] = subprocess.Popen(
                    cmd + ["--role", role],
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        try:
            procs["ctx"].wait(timeout=_PROCESS_TIMEOUT_SECONDS)
            if mode == "refresh":
                procs["gen"].wait(timeout=_PROCESS_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    finally:
        for proc in procs.values():
            _kill_group(proc)
    ctx_log = logs["ctx"].read_text(errors="replace")
    gen_log = logs["gen"].read_text(errors="replace")
    detail = f"ctx log tail:\n{_log_tail(ctx_log)}\ngen log tail:\n{_log_tail(gen_log)}"

    if mode == "refresh":
        assert procs["ctx"].returncode == 0 and procs["gen"].returncode == 0, detail
        assert "[gen] RESULT PASS" in gen_log, detail
        m = re.search(r"KV_POOL_REBALANCE_EXERCISED (\d+)/(\d+)", ctx_log)
        assert m and int(m.group(1)) > 0, f"no send read past the old extent\n{detail}"
    else:
        assert procs["ctx"].returncode not in (0, None), detail
        assert "NIXL_ERR_NOT_FOUND" in ctx_log, detail
        assert "registered up to" in ctx_log, detail
