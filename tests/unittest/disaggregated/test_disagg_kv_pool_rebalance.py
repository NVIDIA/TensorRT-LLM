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
and back-off with stubs, check the cross-node veto's inputs, and check the
incremental KV pool registration's bookkeeping across resizes with a recording
fake agent. The GPU test runs kv_pool_rebalance_worker.py as two
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

import tensorrt_llm._torch.disaggregation.native.transfer as transfer
import tensorrt_llm._torch.disaggregation.transceiver as transceiver_mod
import tensorrt_llm._torch.pyexecutor.py_executor as pe
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar, _endpoint_host
from tensorrt_llm._torch.disaggregation.resource.page import CacheKind
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.runtime.kv_cache_manager_v2 import OutOfPagesError

PE = pe.PyExecutor

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kv_pool_rebalance_worker.py")
GEN_WORKER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kv_pool_rebalance_gen_worker.py"
)
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
        self.allow_cross_node_args = []

    def kv_pool_rebalance_unsupported_reason(self):
        return self.reason

    def kv_pool_layout_shared_with_peers(self):
        return self.shared

    def kv_pool_layout_fixed(self):
        # VMM pools (the path these stubs cover): a resize remaps the pools' tails.
        return False

    def kv_pool_rebalance_veto_reason(self, allow_cross_node=False):
        self.allow_cross_node_args.append(allow_cross_node)
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
    s.reap_args = []
    s.flushes = 0
    s.async_transfer_manager = types.SimpleNamespace(has_any_inflight_requests=lambda: False)
    s.dist = types.SimpleNamespace(
        allgather=lambda st: [st, *peers_states], cp_size=1, tp_size=1, pp_size=1
    )

    def reap(n):
        s.reaps += 1
        s.reap_args.append(n)
        if xcvr is not None and xcvr.sessions:
            xcvr.sessions -= 1

    s.unpublished_gen_receives = 0
    s.gen_polls = 0

    def poll_gen():
        s.gen_polls += 1
        s.unpublished_gen_receives = max(0, s.unpublished_gen_receives - 1)

    s.disagg = types.SimpleNamespace(
        reap_context_sends=reap,
        has_unpublished_gen_receives=lambda: s.unpublished_gen_receives > 0,
        poll_gen_transfers=poll_gen,
    )
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
    # Non-blocking polls: a blocking one (n > 0) can wait out a sender-future
    # timeout per pending send, far past the drain budget.
    assert s.reap_args == [0, 0, 0]


def test_drain_sleeps_within_its_budget(monkeypatch):
    sleeps = []
    monkeypatch.setattr(pe.time, "sleep", sleeps.append)
    s = _executor(_Xcvr(sessions=3))
    assert _drain(s) and len(sleeps) == 3
    assert all(0 < t <= pe.KV_POOL_REBALANCE_DISAGG_POLL_S for t in sleeps)


@pytest.mark.parametrize("allow", [False, True])
def test_drain_passes_cross_node_flag(allow):
    s = _executor(_Xcvr(sessions=2))
    s._kv_pool_rebalance_allow_cross_node = allow
    assert _drain(s)
    assert s.kv_cache_transceiver.allow_cross_node_args == [allow] * 3


def test_drain_waits_for_unpublished_gen_receives():
    """Generation side: unpublished receives count as in flight.

    With copy-based rebalancing an admitted request whose receive has not started still counts as in
    flight, and every drain pass advances receives.
    """
    s = _executor(_Xcvr())
    s.unpublished_gen_receives = 2
    assert _drain(s) and s.gen_polls == 2 and s.unpublished_gen_receives == 0


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
    # No reap: the drain stopped on its first pass instead of waiting.
    assert not _drain(s) and s.reaps == 0
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


def test_oldest_send_waiting_for_peer(monkeypatch):
    now = 1000.0
    monkeypatch.setattr(transceiver_mod.time, "monotonic", lambda: now)

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
    assert KvCacheTransceiverV2.oldest_send_waiting_for_peer_s(x) == pytest.approx(0.7)


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
        has_registered_with_peers=registered,
        has_cross_node_peer=cross_node,
        kv_pool_layout_fixed=False,
    )
    reason = x.kv_pool_rebalance_veto_reason(allow_cross_node=allow)
    assert (reason is None) if expected is None else (expected in reason)


# ---------------------------------------------------------------------------
# CPU: the executor's resize brackets adjust() with begin/end on the transceiver
# ---------------------------------------------------------------------------


class _ResizeXcvr:
    def __init__(self):
        self.calls = []

    def begin_kv_pool_resize(self):
        self.calls.append("begin")

    def end_kv_pool_resize(self):
        self.calls.append("end")


def _resize_executor(adjust):
    req = types.SimpleNamespace(py_request_id=7)
    mgr = types.SimpleNamespace(
        impl=types.SimpleNamespace(
            adjust=adjust,
            init_config=types.SimpleNamespace(
                pool_rebalance=types.SimpleNamespace(in_place_relocation=False)
            ),
        ),
        is_request_active=lambda rid: True,
        suspended=[],
        resumed=[],
    )
    mgr.suspend_request = mgr.suspended.append
    mgr.resume_request = mgr.resumed.append
    s = types.SimpleNamespace(
        kv_cache_manager=mgr,
        active_requests=[req],
        kv_cache_transceiver=_ResizeXcvr(),
        _suspend_padding_dummies_for_rebalance=lambda mgr: [],
        _resume_padding_dummies_after_rebalance=lambda mgr, dummies: None,
        _read_kv_cache_capacity=lambda: 0,
        _defer_generation_only_release=False,
    )
    return s, req


def _raise(exc):
    def adjust():
        raise exc

    return adjust


def test_resize_ends_after_out_of_pages():
    """adjust() failing after begin must still re-register (end) and resume."""
    s, req = _resize_executor(_raise(OutOfPagesError("no pages")))
    PE._rebalance_kv_pools_now(s)
    assert s.kv_cache_transceiver.calls == ["begin", "end"]
    assert s.kv_cache_manager.resumed == [req]


def test_resize_ends_before_other_errors_propagate():
    s, _ = _resize_executor(_raise(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        PE._rebalance_kv_pools_now(s)
    assert s.kv_cache_transceiver.calls == ["begin", "end"]


# ---------------------------------------------------------------------------
# CPU: incremental KV pool registration across resizes (recording fake agent)
# ---------------------------------------------------------------------------

_SEG = 1024  # small registration unit for the test
# Pool group -> (base, slot stride) of each pool; strides that do not divide
# _SEG make pool ends fall mid-segment.
_POOLS = {0: [(0x100000, 96), (0x200000, 160)], 1: [(0x300000, 256)]}


class _RecordingAgent:
    def __init__(self):
        self.live = {}  # id(descs) -> (memory type, [(addr, length), ...])
        self.registrations = []  # (addr, length), in order

    def register_memory(self, descs):
        assert id(descs) not in self.live, "registered twice"
        ranges = [(addr, length) for addr, length, _, _ in descs.descs]
        for _, live in self.live.values():
            for a, n in live:
                for addr, length in ranges:
                    assert addr + length <= a or a + n <= addr, "overlapping registration"
        self.live[id(descs)] = (descs.type, ranges)
        self.registrations.extend(ranges)

    def deregister_memory(self, descs):
        del self.live[id(descs)]

    def get_local_agent_desc(self):
        return b"agent"


class _Manager:
    """Pool groups' slot counts, the plan for the next adjust(), and adjust()."""

    def __init__(self, num_slots, with_plan=True, host_pool_groups=()):
        self.num_slots = list(num_slots)
        # Pool groups whose pools live in host memory (registered as DRAM).
        self.host_pool_groups = set(host_pool_groups)
        self.impl = types.SimpleNamespace(pool_group_descs=[], planned_slot_counts=None)
        if not with_plan:
            del self.impl.planned_slot_counts
        self._sync()

    def _sync(self):
        self.impl.pool_group_descs = [types.SimpleNamespace(num_slots=n) for n in self.num_slots]

    def page_table(self):
        return types.SimpleNamespace(
            layer_groups=[types.SimpleNamespace(kind=CacheKind.PAGED)],
            has_host_pools=bool(self.host_pool_groups),
            pool_groups=[
                types.SimpleNamespace(
                    pools=[
                        types.SimpleNamespace(
                            base_address=base,
                            slot_stride_bytes=stride,
                            num_slots=n,
                            memory_type="DRAM" if pg in self.host_pool_groups else "VRAM",
                        )
                        for base, stride in _POOLS[pg]
                    ]
                )
                for pg, n in enumerate(self.num_slots)
            ],
        )

    def lengths(self, num_slots=None):
        num_slots = self.num_slots if num_slots is None else num_slots
        return {
            base: num_slots[pg] * stride for pg, pools in _POOLS.items() for base, stride in pools
        }


def _worker(monkeypatch, mgr):
    monkeypatch.setattr(transfer, "_KV_REG_SEGMENT_BYTES", _SEG)
    monkeypatch.setattr(transfer, "build_page_table_from_manager", lambda m: m.page_table())
    monkeypatch.setattr(transfer, "KVRegionExtractorV1", lambda page_table: page_table)
    monkeypatch.setenv("TLLM_KV_INCREMENTAL_REGISTRATION", "1")
    w = object.__new__(transfer.TransferWorker)
    w._agent = _RecordingAgent()
    w._config = types.SimpleNamespace(kv_cache_manager=mgr)
    w._rank_info = types.SimpleNamespace(
        page_table=mgr.page_table(), device_id=0, transfer_engine_info=b""
    )
    w._peer_registrar = types.SimpleNamespace(replace_self_extractor=lambda extractor: None)
    w._registered_mem = []
    w._kv_registered_mem = []
    w._kv_segments = None
    w._register_kv_cache()
    return w


def _check_segments(w, lengths):
    """Each pool's segments tile [0, length) from aligned starts, all registered."""
    live = set()
    for base, length in lengths.items():
        segs = w._kv_segments[base]
        assert [start for start, _, _ in segs] == list(range(0, length, _SEG))
        assert all(end == min(start + _SEG, length) for start, end, _ in segs)
        assert (segs[-1][1] if segs else 0) == length
        live |= {id(descs) for _, _, descs in segs}
    assert set(w._agent.live) == live
    assert {id(d) for d in w._registered_mem} == live


def _resize(w, mgr, new_slots, adjust_ok=True):
    """Begin -> adjust (or a failed one) -> end, checking what begin dropped and end kept."""
    planned = getattr(mgr.impl, "planned_slot_counts", "absent")
    if planned != "absent":
        mgr.impl.planned_slot_counts = list(new_slots)
    w.begin_kv_pool_resize()
    planned_len = mgr.lengths(new_slots) if planned != "absent" else dict.fromkeys(_POOLS, 0)
    kept = {}
    for base, segs in w._kv_segments.items():
        # Nothing registered may reach past the planned length once adjust() runs.
        assert all(end <= planned_len.get(base, 0) for _, end, _ in segs)
        kept[base] = list(segs)
    if adjust_ok:
        mgr.num_slots = list(new_slots)
        mgr._sync()
    w.end_kv_pool_resize()
    _check_segments(w, mgr.lengths())
    for base, segs in kept.items():
        now = w._kv_segments[base]
        for i, seg in enumerate(segs):
            # A kept segment keeps its registration; only one that ended
            # mid-unit (the pool's old end) may be redone, from the same start.
            if seg[1] % _SEG == 0 or seg[1] == mgr.lengths()[base]:
                assert now[i] is seg
            else:
                assert i == len(segs) - 1 and now[i][0] == seg[0]
    assert w._rank_info.transfer_engine_info == b"agent"


@pytest.mark.parametrize(
    "steps",
    [
        [((40, 30), True), ((90, 30), True), ((90, 70), True)],  # grow
        [((90, 70), True), ((40, 30), True), ((5, 1), True)],  # shrink
        [((41, 30), True), ((47, 33), True), ((200, 33), True)],  # grow from mid-unit
        [((200, 80), True), ((47, 33), True), ((48, 32), True)],  # shrink to mid-unit
        [((90, 70), True), ((40, 30), False), ((40, 30), True)],  # adjust() fails after begin
        [((60, 40), True), ((0, 40), True), ((60, 40), True)],  # a pool group to zero and back
    ],
    ids=["grow", "shrink", "grow_mid_unit", "shrink_mid_unit", "adjust_fails", "to_zero"],
)
def test_incremental_registration_tracks_resizes(monkeypatch, steps):
    mgr = _Manager((64, 48))
    w = _worker(monkeypatch, mgr)
    _check_segments(w, mgr.lengths())
    for new_slots, adjust_ok in steps:
        _resize(w, mgr, new_slots, adjust_ok)


def test_incremental_registration_without_plan_reregisters_all(monkeypatch):
    mgr = _Manager((64, 48), with_plan=False)
    w = _worker(monkeypatch, mgr)
    first = len(w._agent.registrations)
    _resize(w, mgr, (90, 30))
    # No plan: begin drops every segment, end registers each pool from 0.
    assert len(w._agent.registrations) - first == sum(
        -(-length // _SEG) for length in mgr.lengths().values()
    )


def test_incremental_registration_kept_segments_not_reregistered(monkeypatch):
    # Every pool ends mid-unit: 65 * 96, 65 * 160 and 47 * 256 bytes.
    mgr = _Manager((65, 47))
    w = _worker(monkeypatch, mgr)
    old_lengths = mgr.lengths()
    first = len(w._agent.registrations)
    _resize(w, mgr, (128, 96))
    new = w._agent.registrations[first:]
    assert new
    # Growing re-registers only each pool's old mid-unit tail and what lies past it.
    for base, old_len in old_lengths.items():
        assert old_len % _SEG
        tail_start = old_len - old_len % _SEG
        assert all(addr >= base + tail_start for addr, _ in new if base <= addr < base + 0x100000)


def _pool_memory_descs(page_table, device_id, memory_type=None):
    """Whole-pool descriptors of one memory type (get_unique_pool_memory_descs)."""
    return [
        (
            pool.base_address,
            pool.num_slots * pool.slot_stride_bytes,
            device_id,
            f"{pool.base_address:x}",
        )
        for pg in page_table.pool_groups
        for pool in pg.pools
        if memory_type in (None, pool.memory_type)
    ]


def test_host_pools_fall_back_to_whole_pool_registration(monkeypatch):
    """With a pool in host memory, every pool is registered whole, one entry per memory type."""
    monkeypatch.setattr(transfer, "get_unique_pool_memory_descs", _pool_memory_descs)
    mgr = _Manager((64, 48), host_pool_groups={1})
    w = _worker(monkeypatch, mgr)

    def check():
        assert w._kv_segments is None
        live = sorted(w._agent.live.values())
        assert [t for t, _ in live] == ["DRAM", "VRAM"]
        lengths = mgr.lengths()
        for _, ranges in live:
            assert all(length == lengths[addr] for addr, length in ranges)
        assert {id(d) for d in w._kv_registered_mem} == set(w._agent.live)

    check()
    for new_slots, adjust_ok in [((90, 30), True), ((40, 70), False), ((40, 70), True)]:
        mgr.impl.planned_slot_counts = list(new_slots)
        w.begin_kv_pool_resize()
        assert not w._agent.live and not w._kv_registered_mem and not w._registered_mem
        if adjust_ok:
            mgr.num_slots = list(new_slots)
            mgr._sync()
        w.end_kv_pool_resize()
        check()


_ARENA_BASE = 0x10000000
_ARENA_BYTES = 0x200000


class _ArenaManager(_Manager):
    """CoReba layout: two elastic groups in the GPU arena, and a fixed group beside it.

    The fixed group (as with sparse attention KV cache offload) has a device pool of its own and a
    host-backed pool; the elastic groups' pools report the arena base.
    """

    _LAYOUT = {
        0: [(_ARENA_BASE, 96, "VRAM")],
        1: [(_ARENA_BASE, 256, "VRAM")],
        2: [(0x40000000, 64, "VRAM"), (0x50000000, 512, "DRAM")],
    }

    def __init__(self, num_slots):
        super().__init__(num_slots)
        self.impl.get_gpu_arena_extent = lambda: (_ARENA_BASE, _ARENA_BYTES, False)

    def page_table(self):
        return types.SimpleNamespace(
            layer_groups=[types.SimpleNamespace(kind=CacheKind.PAGED)],
            has_host_pools=True,
            pool_groups=[
                types.SimpleNamespace(
                    pools=[
                        types.SimpleNamespace(
                            base_address=base, slot_stride_bytes=stride, num_slots=n, memory_type=mt
                        )
                        for base, stride, mt in self._LAYOUT[pg]
                    ]
                )
                for pg, n in enumerate(self.num_slots)
            ],
        )


def test_arena_registers_once_beside_fixed_groups(monkeypatch):
    """The arena is one region; a fixed group's pools are registered on their own, by memory type.

    Neither changes with a rebalance: the arena never remaps and fixed groups never resize, so the
    layout is fixed and nothing is registered again.
    """
    monkeypatch.setattr(transfer, "get_unique_pool_memory_descs", _pool_memory_descs)
    mgr = _ArenaManager((40, 30, 16))
    w = _worker(monkeypatch, mgr)
    assert w.kv_pool_layout_fixed and w._kv_segments is None
    live = sorted(w._agent.live.values())
    assert live == [
        ("DRAM", [(0x50000000, 16 * 512)]),
        ("VRAM", [(_ARENA_BASE, _ARENA_BYTES), (0x40000000, 16 * 64)]),
    ]
    registrations = len(w._agent.registrations)
    mgr.impl.planned_slot_counts = [60, 10, 16]
    w.begin_kv_pool_resize()
    mgr.num_slots = [60, 10, 16]
    mgr._sync()
    w.end_kv_pool_resize()
    assert sorted(w._agent.live.values()) == live
    assert len(w._agent.registrations) == registrations


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
    # Other test modules set this at import time; the workers need the default (C++) transfer agent.
    env["TRTLLM_USE_PY_NIXL_KVCACHE"] = "0"
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
        # One deadline for both waits, inside the pytest timeout, so the log
        # tails below are always printed.
        deadline = time.monotonic() + _PROCESS_TIMEOUT_SECONDS
        try:
            procs["ctx"].wait(timeout=max(0.0, deadline - time.monotonic()))
            if mode == "refresh":
                procs["gen"].wait(timeout=max(0.0, deadline - time.monotonic()))
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


@pytest.mark.timeout(300)
def test_generation_side_rebalance_integrity(tmp_path):
    """Copy-based rebalancing: the generation side rebalances while a context peer holds its layout.

    Sends after the rebalance, into slots outside the generation side's old windows, arrive bit-exact
    with the context peer still on the page table and registration from before it.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("mpi4py")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not found on PATH")

    env = os.environ.copy()
    env.update({"ZMQ_PORT": str(_find_free_port()), "CUDA_VISIBLE_DEVICES": "0"})
    # Other test modules set this at import time; the workers need the default (C++) transfer agent.
    env["TRTLLM_USE_PY_NIXL_KVCACHE"] = "0"
    cmd = [mpirun, "--allow-run-as-root", "--oversubscribe", "-np", "1", sys.executable, GEN_WORKER]
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
            for role in ("ctx", "gen"):
                procs[role].wait(timeout=_PROCESS_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    finally:
        for proc in procs.values():
            _kill_group(proc)
    ctx_log = logs["ctx"].read_text(errors="replace")
    gen_log = logs["gen"].read_text(errors="replace")
    detail = f"ctx log tail:\n{_log_tail(ctx_log)}\ngen log tail:\n{_log_tail(gen_log)}"
    assert procs["ctx"].returncode == 0 and procs["gen"].returncode == 0, detail
    assert "[gen] RESULT PASS" in gen_log, detail
    m = re.search(r"requests received into slots outside the old window: (\d+)/(\d+)", gen_log)
    assert m and int(m.group(1)) > 0, f"no receive landed outside the old window\n{detail}"


@pytest.mark.parametrize("registered, cross_node", [(True, False), (False, True), (True, True)])
def test_fixed_layout_is_not_vetoed(registered, cross_node):
    """Copy-based rebalancing: a fixed layout is never vetoed.

    The arena is registered once and never remapped, so a peer holding this rank's layout (generation
    side) and a peer on another node are no reason to stop rebalancing.
    """
    x = object.__new__(KvCacheTransceiverV2)
    x._transfer_worker = types.SimpleNamespace(
        has_registered_with_peers=registered,
        has_cross_node_peer=cross_node,
        kv_pool_layout_fixed=True,
    )
    assert not x.kv_pool_layout_shared_with_peers()
    assert x.kv_pool_rebalance_veto_reason() is None
