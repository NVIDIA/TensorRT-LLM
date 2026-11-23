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
"""CPU-only tests for the opt-in just-in-time per-writer bounce grant (TRTLLM_KV_BOUNCE_JIT).

Covers the pure JitTransferContext state machine, the transport plumbing with a fake first-fit
allocator (including a seeded random-interleaving simulation), the wire format, the sender's grant
client against a fake DEALER, and the receiver's dispatch and grant handler with mocks.
"""

import queue
import random
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native.bounce import config as bcfg
from tensorrt_llm._torch.disaggregation.native.bounce import core as bcore

try:
    from tensorrt_llm._torch.disaggregation.native.bounce import impl as btr

    _HAVE_TRANSPORT = True
except ImportError:  # pragma: no cover - CPU-only env without CUDA bindings
    _HAVE_TRANSPORT = False

needs_transport = pytest.mark.skipif(
    not _HAVE_TRANSPORT, reason="bounce impl import needs CUDA bindings"
)

Check = bcore.JitGrantCheck
State = bcore.WriterLeaseState
Verdict = bcore.JitVerdict
_MIB = 1024 * 1024
_KEY = (7, 0)


def _tail(n=2):
    return np.arange(1, n + 1, dtype=np.int64) * 4096, np.full(n, 64, dtype=np.int64)


def _ctx(ranks=(0, 1, 2, 3), cap=1000, on_done=None):
    return bcore.JitTransferContext(
        rid_slice=_KEY,
        writer_ranks=tuple(ranks),
        writer_caps={r: cap for r in ranks},
        on_done=on_done,
    )


def _grant(ctx, rank, slot, base, nbytes=100):
    assert ctx.grant_check(rank, nbytes) is Check.ALLOCATE
    ctx.on_granted(rank, slot, base, nbytes)


def _report_ok(ctx, rank, base):
    dst, sizes = _tail()
    return ctx.record_report(rank, True, src_base=base, dst_ptrs=dst, sizes=sizes)


# ---------------------------------------------------------------------------
# knobs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [(None, False), ("", False), ("0", False), ("1", True)])
def test_bounce_jit_from_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("TRTLLM_KV_BOUNCE_JIT", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_KV_BOUNCE_JIT", raw)
    assert bcfg.bounce_jit_from_env() is expected


@pytest.mark.parametrize("raw", ["2", "true", "on", "-1"])
def test_bounce_jit_rejects_bad_value(monkeypatch, raw):
    monkeypatch.setenv("TRTLLM_KV_BOUNCE_JIT", raw)
    with pytest.raises(ValueError, match="TRTLLM_KV_BOUNCE_JIT"):
        bcfg.bounce_jit_from_env()


def test_bounce_jit_wait_ms(monkeypatch):
    monkeypatch.delenv("TRTLLM_KV_BOUNCE_JIT_WAIT_MS", raising=False)
    assert bcfg.bounce_jit_wait_ms_from_env() == 1000
    monkeypatch.setenv("TRTLLM_KV_BOUNCE_JIT_WAIT_MS", "0")
    assert bcfg.bounce_jit_wait_ms_from_env() == 0
    for bad in ("-5", "1.5", "soon"):
        monkeypatch.setenv("TRTLLM_KV_BOUNCE_JIT_WAIT_MS", bad)
        with pytest.raises(ValueError, match="TRTLLM_KV_BOUNCE_JIT_WAIT_MS"):
            bcfg.bounce_jit_wait_ms_from_env()


@needs_transport
def test_create_bounce_bad_jit_raises_instead_of_disabling(monkeypatch):
    monkeypatch.setenv("TRTLLM_KV_BOUNCE_JIT", "yes")

    def _no_build(*args, **kwargs):
        raise AssertionError("from_config must not run for a bad value")

    monkeypatch.setattr(btr.VmmBounceTransport, "from_config", classmethod(_no_build))
    with pytest.raises(ValueError, match="TRTLLM_KV_BOUNCE_JIT"):
        btr.create_bounce(SimpleNamespace(), SimpleNamespace(), device_id=0, page_table=None)


# ---------------------------------------------------------------------------
# JitTransferContext state machine (pure)
# ---------------------------------------------------------------------------


def test_grant_check_no_cases():
    ctx = _ctx(cap=1000)
    assert ctx.grant_check(9, 100) is Check.NO  # not a writer
    assert ctx.grant_check(0, 1001) is Check.NO  # oversize for its cap
    assert ctx.grant_check(0, 0) is Check.NO  # nothing to write
    assert ctx.grant_check(0, 1000) is Check.ALLOCATE
    ctx.record_report(1, True)  # tail-less success: writer 1 has reported
    assert ctx.grant_check(1, 100) is Check.NO
    ctx.abort_publication({0, 1, 2})
    assert ctx.grant_check(3, 100) is Check.NO  # publication failed
    orphan = _ctx()
    orphan.mark_orphaned()
    assert orphan.grant_check(0, 100) is Check.NO


def test_grant_is_idempotent_until_report():
    ctx = _ctx()
    _grant(ctx, 0, slot=5, base=0x1000, nbytes=200)
    assert ctx.grant_check(0, 200) is Check.REGRANT
    assert ctx.grant_check(0, 100) is Check.REGRANT  # a smaller ask fits the same slot
    assert ctx.grant_check(0, 300) is Check.NO  # a larger one does not
    assert ctx.lease(0).slot_id == 5 and ctx.lease(0).base == 0x1000


def test_per_writer_scatter_and_single_on_done():
    done = []
    ctx = _ctx(ranks=(0, 1), on_done=done.append)
    _grant(ctx, 0, 10, 0x1000)
    _grant(ctx, 1, 11, 0x2000)
    acts = _report_ok(ctx, 0, 0x1000)
    assert [a[0] for a in acts] == ["scatter"] and acts[0][1] == 0
    assert acts[0][2][0] == 0x1000
    assert ctx.lease(0).state is State.SCATTERING
    assert ctx.maybe_settle() is None
    # writer 0's slot is released as soon as its own scatter finishes
    assert ctx.finish_writer_scatter(0, True) == [("release", 10, 100)]
    assert ctx.maybe_settle() is None  # writer 1 has not reported
    _report_ok(ctx, 1, 0x2000)
    assert ctx.maybe_settle() is None  # writer 1 is scattering
    assert ctx.finish_writer_scatter(1, True) == [("release", 11, 100)]
    st = ctx.maybe_settle()
    assert st is not None and st.success and st.slot_id is None
    assert st.disposition is bcore.Disposition.RELEASE
    assert ctx.maybe_settle() is None  # exactly once


def test_one_writer_failing_fails_the_group_and_releases_its_slot():
    ctx = _ctx(ranks=(0, 1))
    _grant(ctx, 0, 10, 0x1000)
    _grant(ctx, 1, 11, 0x2000)
    assert ctx.record_report(1, False) == [("release", 11, 100)]
    _report_ok(ctx, 0, 0x1000)
    ctx.finish_writer_scatter(0, True)
    st = ctx.maybe_settle()
    assert st is not None and st.success is False


def test_scatter_failure_fails_the_group():
    ctx = _ctx(ranks=(0,))
    _grant(ctx, 0, 10, 0x1000)
    _report_ok(ctx, 0, 0x1000)
    assert ctx.finish_writer_scatter(0, False) == [("release", 10, 100)]
    assert ctx.maybe_settle().success is False


def test_tailless_success_releases_and_counts_ok():
    ctx = _ctx(ranks=(0, 1))
    _grant(ctx, 0, 10, 0x1000)  # a grant the sender gave up on (it timed out and went in place)
    assert ctx.record_report(0, True) == [("release", 10, 100)]
    assert ctx.record_report(1, True) == []  # never asked: nothing to release
    st = ctx.maybe_settle()
    assert st is not None and st.success


def test_duplicate_and_late_reports_are_ignored():
    ctx = _ctx(ranks=(0, 1))
    _grant(ctx, 0, 10, 0x1000)
    _report_ok(ctx, 0, 0x1000)
    assert ctx.record_report(0, False) == []  # duplicate: no flip to failed
    assert ctx.lease(0).ok
    ctx.finish_writer_scatter(0, True)
    ctx.record_report(1, True)
    assert ctx.maybe_settle().success
    assert ctx.record_report(1, False) == []  # late, after settle


def test_success_with_tail_but_no_grant_fails():
    ctx = _ctx(ranks=(0,))
    assert _report_ok(ctx, 0, 0x1000) == []
    assert ctx.maybe_settle().success is False


def test_success_with_wrong_base_fails():
    ctx = _ctx(ranks=(0,))
    _grant(ctx, 0, 10, 0x1000)
    assert _report_ok(ctx, 0, 0x5000) == [("release", 10, 100)]
    assert ctx.maybe_settle().success is False


def test_cancel_before_any_grant():
    ctx = _ctx()
    assert ctx.mark_orphaned() == []
    st = ctx.maybe_settle()
    assert st is not None and st.success is False and st.slot_id is None
    assert ctx.grant_check(0, 10) is Check.NO


def test_cancel_after_grant_quarantines_only_the_unreported():
    ctx = _ctx(ranks=(0, 1, 2))
    _grant(ctx, 0, 10, 0x1000)
    _grant(ctx, 1, 11, 0x2000)
    ctx.record_report(1, True)  # tail-less: released now
    acts = ctx.mark_orphaned()
    assert acts == [("quarantine", 10, 100)]
    assert ctx.maybe_settle().success is False
    assert ctx.record_report(0, True) == []  # the late report finds no slot to release


def test_cancel_during_scatter_releases_at_finish():
    done = []
    ctx = _ctx(ranks=(0, 1), on_done=done.append)
    _grant(ctx, 0, 10, 0x1000)
    _report_ok(ctx, 0, 0x1000)
    assert ctx.mark_orphaned() == []  # scattering: left alone
    assert ctx.maybe_settle() is None  # never while scattering
    assert ctx.finish_writer_scatter(0, True) == [("release", 10, 100)]
    st = ctx.maybe_settle()
    assert st is not None and st.success is False


def test_abort_publication_shrinks_the_cohort():
    ctx = _ctx(ranks=(0, 1, 2, 3))
    _grant(ctx, 0, 10, 0x1000)
    ctx.abort_publication({0, 1})
    assert ctx.grant_check(1, 10) is Check.NO
    # writer 0's write is drained by its report; partial data is not scattered
    assert _report_ok(ctx, 0, 0x1000) == [("release", 10, 100)]
    assert ctx.maybe_settle() is None  # writer 1 is published and must still report
    ctx.record_report(1, False)
    assert ctx.maybe_settle().success is False


def test_abort_publication_inconsistent_raises():
    ctx = _ctx(ranks=(0, 1))
    ctx.record_report(1, True)
    with pytest.raises(RuntimeError, match="inconsistent"):
        ctx.abort_publication({0})


# ---------------------------------------------------------------------------
# transport plumbing with a fake first-fit allocator
# ---------------------------------------------------------------------------


class _FirstFitAlloc:
    """A CPU first-fit allocator with the SlotAllocator surface the transport uses."""

    def __init__(self, capacity_bytes, phys_chunk_size, name="kv_bounce"):
        self.capacity = capacity_bytes
        self.base = 0x100000
        self.in_use = {}
        self.quarantined = {}
        self._next = 0
        self.timeouts = []

    def _occupied(self):
        return sorted(list(self.in_use.values()) + list(self.quarantined.values()))

    def reserve(self, size, timeout=None):
        self.timeouts.append(timeout)
        cursor = 0
        for s, n in self._occupied():
            if s - cursor >= size:
                break
            cursor = max(cursor, s + n)
        else:
            if self.capacity - cursor < size:
                return None
        sid = self._next
        self._next += 1
        self.in_use[sid] = (cursor, size)
        return sid, self.base + cursor

    def release(self, slot_id):
        self.in_use.pop(slot_id, None)

    def quarantine(self, slot_id, grace_s):
        entry = self.in_use.pop(slot_id, None)
        if entry is not None:
            self.quarantined[slot_id] = entry

    def reclaim_expired(self):
        n = len(self.quarantined)
        self.quarantined.clear()
        return n

    def reg_descs(self):
        return []

    def close(self):
        pass


def _transport(monkeypatch, capacity, block_bytes=1024, direction="both", jit=True):
    if jit:
        monkeypatch.setenv("TRTLLM_KV_BOUNCE_JIT", "1")
    else:
        monkeypatch.delenv("TRTLLM_KV_BOUNCE_JIT", raising=False)
    monkeypatch.setattr(btr, "SlotAllocator", _FirstFitAlloc)
    monkeypatch.setattr(btr.VmmBounceTransport, "_new_stream", lambda self, mode=0: 0)
    monkeypatch.setattr(
        btr.VmmBounceTransport,
        "_start_scatter_worker",
        lambda self, name: setattr(self, "_scatter_q", queue.Queue()),
    )
    agent = SimpleNamespace(register_memory=lambda d: None)
    return btr.VmmBounceTransport(
        agent,
        device_id=0,
        capacity_bytes=capacity,
        phys_chunk_size=32 * _MIB,
        block_bytes_per_group=[block_bytes],
        min_blocks=1,
        direction=direction,
    )


def _req(nblocks, rid=7, slice_id=0):
    return SimpleNamespace(
        block_ids_per_layer_groups=[np.arange(nblocks, dtype=np.int64)],
        unique_rid=rid,
        slice_id=slice_id,
        bounce_dst_base=None,
    )


def _drain_scatters(t, ok=True):
    t._scatter_blocking = lambda descs: None if ok else (_ for _ in ()).throw(RuntimeError("x"))
    while not t._scatter_q.empty():
        item = t._scatter_q.get_nowait()
        assert len(item) == 3
        t._scatter_jit_writer(*item)


@needs_transport
def test_transport_register_grant_scatter_settle(monkeypatch):
    t = _transport(monkeypatch, capacity=64 * 1024)
    req = _req(16)  # 16 KiB over 4 writers -> 4 KiB cap each
    assert t.register_jit(req, writer_ranks=(0, 1, 2, 3), writers_split_layers=True)
    assert req.bounce_dst_base is None  # nothing reserved at dispatch
    assert t._recv_alloc.in_use == {}
    assert t.is_bounced(_KEY) and t.writer_base(_KEY, 0) is None
    done = []
    for rank in range(4):
        verdict, base, cap = t.grant(_KEY, rank, 4096)
        assert verdict == Verdict.GRANT and cap == 4096
        assert t._recv_alloc.timeouts[-1] == 0  # never waits
        dst, sizes = _tail()
        t.record_result(_KEY, rank, dst, sizes, base, on_done=done.append if rank == 3 else None)
    assert t._scatter_q.qsize() == 4
    _drain_scatters(t)
    assert done == [True]
    assert t._recv_alloc.in_use == {} and not t.is_bounced(_KEY)
    assert t._jit_stats["inflight_bytes"] == 0 and t._jit_stats["inflight_bytes_max"] == 4 * 4096
    assert t.grant(_KEY, 0, 4096)[0] == Verdict.NO  # unknown after settle


@needs_transport
def test_transport_retry_then_grant_after_release(monkeypatch):
    t = _transport(monkeypatch, capacity=8192)
    assert t.register_jit(_req(16), writer_ranks=(0, 1, 2, 3))
    assert t.grant(_KEY, 0, 4096)[0] == Verdict.GRANT
    v1, b1, _ = t.grant(_KEY, 1, 4096)
    assert v1 == Verdict.GRANT
    assert t.grant(_KEY, 2, 4096) == (int(Verdict.RETRY), 0, 0)
    t.record_result(_KEY, 1, *_tail(), b1)
    _drain_scatters(t)  # writer 1's slot comes back
    assert t.grant(_KEY, 2, 4096)[0] == Verdict.GRANT
    assert t._jit_stats["retries"] == 1


@needs_transport
def test_transport_oversize_writer_is_refused(monkeypatch):
    t = _transport(monkeypatch, capacity=4096)
    assert not t.register_jit(_req(32), writer_ranks=(0, 1, 2, 3))  # 8 KiB per writer
    assert not t.is_bounced(_KEY)
    # the legacy whole-region reserve would also refuse it, so JIT only widens what fits
    assert not t.reserve(_req(32), 4)


@needs_transport
def test_transport_orphan_quarantines_granted(monkeypatch):
    t = _transport(monkeypatch, capacity=64 * 1024)
    assert t.register_jit(_req(16), writer_ranks=(0, 1, 2, 3))
    t.grant(_KEY, 0, 4096)
    t.orphan_reservation(_KEY)
    assert len(t._recv_alloc.quarantined) == 1 and t._recv_alloc.in_use == {}
    assert not t.is_bounced(_KEY)


@needs_transport
def test_transport_release_idle_and_legacy_unchanged(monkeypatch):
    t = _transport(monkeypatch, capacity=64 * 1024)
    assert t.register_jit(_req(16), writer_ranks=(0, 1, 2, 3))
    t.release_idle_reservation(_KEY)
    assert not t.is_bounced(_KEY)
    # the dispatch-time reserve still builds a TransferContext and the per-writer bases
    req = _req(16)
    assert t.reserve(req, 4)
    assert isinstance(t._reserved_map[_KEY], bcore.TransferContext)
    assert t.writer_base(_KEY, 1) == req.bounce_dst_base + 4096
    assert t.grant(_KEY, 0, 4096)[0] == Verdict.NO  # a legacy region is never granted


class _CountingLock:
    """A context-manager lock that counts how often it is taken."""

    def __init__(self):
        self.entered = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc):
        return False


@needs_transport
def test_transport_jit_off_registers_nothing_and_skips_the_lookup(monkeypatch):
    t = _transport(monkeypatch, capacity=64 * 1024, jit=False)
    assert not t.register_jit(_req(16), writer_ranks=(0, 1, 2, 3))
    assert not t.is_bounced(_KEY)
    # the dispatch-time reserve and its result path run as without the knob
    assert t.reserve(_req(16), 4)
    lock = _CountingLock()
    t._reserved_map_lock = lock
    assert t._is_jit(_KEY) is False
    assert lock.entered == 0  # no extra round trip under the reservation lock
    t.record_failure(_KEY, 0)
    assert lock.entered == 1  # only the legacy _apply takes it


@needs_transport
def test_transport_jit_on_looks_up_the_context_under_the_lock(monkeypatch):
    t = _transport(monkeypatch, capacity=64 * 1024, jit=True)
    assert t.register_jit(_req(16), writer_ranks=(0, 1, 2, 3))
    real = t._reserved_map_lock
    lock = _CountingLock()
    t._reserved_map_lock = lock
    assert t._is_jit(_KEY) is True
    assert lock.entered == 1
    t._reserved_map_lock = real


@needs_transport
def test_send_only_transport_cannot_register(monkeypatch):
    t = _transport(monkeypatch, capacity=64 * 1024, direction="send")
    assert not t.register_jit(_req(16), writer_ranks=(0, 1, 2, 3))
    assert t.grant(_KEY, 0, 4096)[0] == Verdict.NO


@needs_transport
def test_no_bounce_transport_jit():
    nb = btr.NoBounceTransport()
    assert nb.register_jit(_req(4), writer_ranks=(0,)) is False
    assert nb.grant(_KEY, 0, 10) == (int(Verdict.NO), 0, 0)


@needs_transport
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_interleaving_simulation(monkeypatch, seed):
    """Simulate random interleavings of grants, reports, scatters, cancels and stale grants.

    4 writers x 200 requests on a region holding 3 writer slices: no leak, every context settles
    exactly once, and never while a writer is scattering.
    """
    rng = random.Random(seed)
    writer_bytes = 4096
    t = _transport(monkeypatch, capacity=3 * writer_bytes)
    settles = {}
    live = {}  # rid -> per-writer status
    next_rid = 0
    orphans = [0]
    doomed = set()

    def on_done_for(rid, ctx):
        def cb(success):
            assert not ctx.is_scattering()
            settles[rid] = settles.get(rid, 0) + 1

        return cb

    def register():
        nonlocal next_rid
        rid = next_rid
        next_rid += 1
        assert t.register_jit(_req(16, rid=rid), writer_ranks=(0, 1, 2, 3))
        ctx = t._reserved_map[(rid, 0)]
        ctx.on_done = on_done_for(rid, ctx)
        if rng.random() < 0.1:
            doomed.add(rid)
        live[rid] = {w: {"state": "idle", "base": None, "retries": 0} for w in range(4)}

    def scatter_one():
        item = t._scatter_q.get_nowait()
        ok = rng.random() > 0.02
        t._scatter_blocking = (
            lambda descs, ok=ok: None if ok else (_ for _ in ()).throw(RuntimeError("scatter"))
        )
        t._scatter_jit_writer(*item)

    def step_writer(rid, w):
        st = live[rid][w]
        key = (rid, 0)
        if st["state"] == "idle":
            verdict, base, _cap = t.grant(key, w, writer_bytes)
            if verdict == Verdict.GRANT:
                st["state"], st["base"] = "granted", base
            elif verdict == Verdict.RETRY and st["retries"] < 50:
                st["retries"] += 1
            else:  # NO, or the wait budget ran out: the flow goes per-fragment
                t.record_result(key, w)
                st["state"] = "done"
        elif st["state"] == "granted":
            r = rng.random()
            if r < 0.05:  # a resend after a lost reply gets the same grant
                assert t.grant(key, w, writer_bytes)[1] in (st["base"], 0)
            elif r < 0.10:  # stale grant: the sender timed out and went per-fragment
                t.record_result(key, w)
                st["state"] = "done"
            elif r < 0.15:
                t.record_failure(key, w)
                st["state"] = "done"
            else:
                t.record_result(key, w, *_tail(), st["base"])
                st["state"] = "done"

    steps = 0
    while next_rid < 200 or live or not t._scatter_q.empty():
        steps += 1
        assert steps < 200_000
        if next_rid < 200 and len(live) < 6 and rng.random() < 0.3:
            register()
        if not t._scatter_q.empty() and rng.random() < 0.4:
            scatter_one()
            continue
        if not live:
            continue
        rid = rng.choice(list(live))
        if rid in doomed and rng.random() < 0.05:  # a cancel or timeout at a random point
            doomed.discard(rid)
            orphans[0] += 1
            t.orphan_reservation((rid, 0))
        pending = [w for w, s in live[rid].items() if s["state"] != "done"]
        if pending:
            step_writer(rid, rng.choice(pending))
        elif not t.is_bounced((rid, 0)):
            del live[rid]
        elif t._scatter_q.empty():
            # every writer reported and nothing is queued: the context must be settled
            ctx = t._reserved_map[(rid, 0)]
            assert ctx.is_scattering() is False
            raise AssertionError(f"request {rid} reported fully but did not settle")
    t._recv_alloc.reclaim_expired()
    assert t._recv_alloc.in_use == {}
    assert t._reserved_map == {}
    assert sorted(settles) == list(range(200)) and set(settles.values()) == {1}
    assert t._jit_stats["inflight_bytes"] == 0
    assert t._jit_stats["inflight_bytes_max"] <= 3 * writer_bytes
    # the run exercised every path it claims to
    st = t._jit_stats
    assert st["grants"] > 300 and st["retries"] > 0 and st["no"] > 0 and st["scatters"] > 0
    assert orphans[0] > 0


# ---------------------------------------------------------------------------
# wire format
# ---------------------------------------------------------------------------

tfr = None
try:
    from tensorrt_llm._torch.disaggregation.native import transfer as tfr
except ImportError:  # pragma: no cover - CPU-only env without the transfer deps
    pass

needs_transfer = pytest.mark.skipif(tfr is None, reason="transfer import needs CUDA deps")


def _recv_info(**kw):
    return tfr.RecvReqInfo(
        sender_req_id=3,
        instance_name="gen",
        instance_rank=1,
        block_ids_per_layer_groups=[np.arange(4, dtype=np.int64)],
        unique_rid=11,
        slice_id=0,
        **kw,
    )


@needs_transfer
def test_recv_req_info_bytes_unchanged_with_jit_off():
    legacy = msgpack.packb(
        {
            "sender_req_id": 3,
            "instance_name": "gen",
            "instance_rank": 1,
            "block_ids_per_layer_groups": [np.arange(4, dtype=np.int64).tobytes()],
            "unique_rid": 11,
            "dst_start_token": None,
            "aux_slot": None,
            "slice_id": 0,
            "bounce_dst_base": None,
        }
    )
    assert _recv_info().to_bytes() == legacy
    assert "bounce_jit" not in msgpack.unpackb(_recv_info().to_bytes(), raw=False)


@needs_transfer
def test_recv_req_info_round_trip_with_jit_on():
    info = tfr.RecvReqInfo.from_bytes(_recv_info(bounce_jit=True).to_bytes())
    assert info.bounce_jit is True and info.bounce_dst_base is None


@needs_transfer
def test_grant_structs_round_trip():
    req = tfr._GRANT_REQ.pack(3, 2**40 + 5, 1, 9 * 2**30, 2**63 + 7)
    assert tfr._GRANT_REQ.unpack(req) == (3, 2**40 + 5, 1, 9 * 2**30, 2**63 + 7)
    rep = tfr._GRANT_REPLY.pack(2**63 + 7, int(Verdict.RETRY), 0x7F0000000000, 4096)
    assert tfr._GRANT_REPLY.unpack(rep) == (2**63 + 7, 2, 0x7F0000000000, 4096)


# ---------------------------------------------------------------------------
# sender grant client
# ---------------------------------------------------------------------------


class _FakeDealer:
    """Scripted DEALER: ``script(nonce, attempt)`` returns the replies queued for one request."""

    def __init__(self, script):
        self.script = script
        self.sent = []
        self.inbox = []

    def send(self, frames):
        assert frames[0] == tfr.MessageType.BOUNCE_GRANT_REQUEST
        fields = tfr._GRANT_REQ.unpack(frames[1])
        self.sent.append(fields)
        self.inbox += self.script(fields[4], len(self.sent) - 1)

    def poll(self, timeout_ms):
        assert timeout_ms <= int(tfr._JIT_REPLY_TIMEOUT_S * 1000)
        return bool(self.inbox)

    def receive(self):
        return self.inbox.pop(0)


def _reply(nonce, verdict, base=0, cap=0):
    return [tfr.MessageType.BOUNCE_GRANT, tfr._GRANT_REPLY.pack(nonce, int(verdict), base, cap)]


def _sender(monkeypatch, dealer, wait_s=1.0):
    tx = object.__new__(tfr.Sender)
    tx._instance_rank = 2
    tx._bounce_jit_wait_s = wait_s
    monkeypatch.setattr(tx, "_get_or_connect_thread_dealer", lambda ep: dealer, raising=False)
    monkeypatch.setattr(tfr.time, "sleep", lambda s: None)
    return tx


def _wm():
    return SimpleNamespace(
        sizes=np.array([1000, 24], dtype=np.int64),
        unique_rid=11,
        receiver_slice_id=0,
        slice_id=0,
        peer_endpoint="tcp://gen:1",
        bounce_dst_base=None,
        bounce_dst_capacity=None,
    )


@needs_transfer
def test_sender_grant_drops_stale_nonce(monkeypatch):
    dealer = _FakeDealer(
        lambda n, a: [
            _reply(n - 1, Verdict.GRANT, 0xBAD, 1),
            _reply(n, Verdict.GRANT, 0x5000, 1024),
        ]
    )
    tx = _sender(monkeypatch, dealer)
    wm = _wm()
    assert tx._request_bounce_grant(wm) is True
    assert (wm.bounce_dst_base, wm.bounce_dst_capacity) == (0x5000, 1024)
    assert dealer.sent[0][:4] == (2, 11, 0, 1024)


@needs_transfer
def test_sender_grant_retry_then_grant(monkeypatch):
    def script(n, attempt):
        return (
            [_reply(n, Verdict.RETRY)] if attempt < 2 else [_reply(n, Verdict.GRANT, 0x6000, 1024)]
        )

    dealer = _FakeDealer(script)
    wm = _wm()
    assert _sender(monkeypatch, dealer)._request_bounce_grant(wm) is True
    assert len(dealer.sent) == 3 and len({s[4] for s in dealer.sent}) == 3  # fresh nonces
    assert wm.bounce_dst_base == 0x6000


@needs_transfer
def test_sender_grant_budget_exhausted(monkeypatch):
    dealer = _FakeDealer(lambda n, a: [_reply(n, Verdict.RETRY)])
    wm = _wm()
    assert _sender(monkeypatch, dealer, wait_s=0.0)._request_bounce_grant(wm) is False
    assert wm.bounce_dst_base is None and len(dealer.sent) == 1


@needs_transfer
def test_sender_grant_no(monkeypatch):
    dealer = _FakeDealer(lambda n, a: [_reply(n, Verdict.NO)])
    wm = _wm()
    assert _sender(monkeypatch, dealer)._request_bounce_grant(wm) is False
    assert wm.bounce_dst_base is None and len(dealer.sent) == 1


@needs_transfer
def test_sender_grant_reply_timeout(monkeypatch):
    dealer = _FakeDealer(lambda n, a: [])
    wm = _wm()
    assert _sender(monkeypatch, dealer)._request_bounce_grant(wm) is False
    assert wm.bounce_dst_base is None


# ---------------------------------------------------------------------------
# receiver dispatch and grant handler
# ---------------------------------------------------------------------------


class _MockBounce:
    enabled = True
    can_recv = True

    def __init__(self):
        self.calls = []

    def register_jit(self, recv_req, **kw):
        self.calls.append(("register_jit", kw))
        return True

    def reserve(self, recv_req, num_writers=1, **kw):
        self.calls.append(("reserve", num_writers, kw))
        recv_req.bounce_dst_base = 0x1000
        return True

    def writer_base(self, key, i):
        return 0x1000 + i * 4096

    def writer_capacity(self, key, i):
        return None

    def release_idle_reservation(self, key):
        self.calls.append(("release_idle", key))


def _receiver(monkeypatch, bounce, jit, ranks=(0, 1, 2, 3), dp_rank=0):
    overlap = SimpleNamespace(ranks=list(ranks), overlap_pp_size=len(ranks))
    peer = SimpleNamespace(
        cp_size=1,
        dp_size=1,
        page_table=None,
        sender_endpoints=[f"ep{r}" for r in range(4)],
    )
    req = _recv_info()
    rx = object.__new__(tfr.Receiver)
    rx._bounce = bounce
    rx._bounce_jit = jit
    rx._enforce_physical_ownership = False
    rx._registrar = SimpleNamespace(
        get_peer_overlap=lambda peer_ri, dp: overlap,
        self_rank_info=SimpleNamespace(cp_size=1),
        self_extractor=SimpleNamespace(page_table=None),
    )
    session = SimpleNamespace(mark_transferring=lambda slice_id: None, _sender_endpoints=set())
    sent = []
    monkeypatch.setattr(rx, "_build_recv_req_info", lambda task: req, raising=False)
    monkeypatch.setattr(rx, "_get_sender_info", lambda params: peer, raising=False)
    monkeypatch.setattr(rx, "_get_session", lambda rid: session, raising=False)
    monkeypatch.setattr(rx, "_fanin_bounce_safe", lambda *a, **k: True, raising=False)
    monkeypatch.setattr(
        rx, "_request_sender_data", lambda ep, payload: sent.append((ep, payload)), raising=False
    )
    task = SimpleNamespace(
        _params=SimpleNamespace(ctx_dp_rank=dp_rank),
        _unique_rid=11,
        slice_id=0,
        expected_transfers=None,
        _perf_timer=None,
    )
    return rx, task, sent


@needs_transfer
def test_dispatch_jit_registers_without_reserving(monkeypatch):
    bounce = _MockBounce()
    rx, task, sent = _receiver(monkeypatch, bounce, jit=True)
    rx.dispatch_task(task)
    assert [c[0] for c in bounce.calls] == ["register_jit"]
    kw = bounce.calls[0][1]
    assert kw["writer_ranks"] == (0, 1, 2, 3) and kw["writers_split_layers"] is True
    assert kw["extra_bytes"] == 0
    assert [ep for ep, _ in sent] == ["ep0", "ep1", "ep2", "ep3"]
    assert len({p for _, p in sent}) == 1  # one message for every writer
    info = tfr.RecvReqInfo.from_bytes(sent[0][1])
    assert info.bounce_jit is True and info.bounce_dst_base is None


@needs_transfer
def test_dispatch_jit_off_keeps_the_legacy_reserve(monkeypatch):
    bounce = _MockBounce()
    rx, task, sent = _receiver(monkeypatch, bounce, jit=False)
    rx.dispatch_task(task)
    assert bounce.calls == [("reserve", 4, {"extra_bytes": 0, "writers_split_layers": True})]
    infos = [tfr.RecvReqInfo.from_bytes(p) for _, p in sent]
    assert [i.bounce_dst_base for i in infos] == [0x1000 + i * 4096 for i in range(4)]
    assert not any(i.bounce_jit for i in infos)
    assert all(b"bounce_jit" not in p for _, p in sent)


@needs_transfer
def test_dispatch_jit_ineligible_under_adp_broadcast(monkeypatch):
    bounce = _MockBounce()
    rx, task, sent = _receiver(monkeypatch, bounce, jit=True, ranks=(0,), dp_rank=None)
    rx.dispatch_task(task)
    assert [c[0] for c in bounce.calls] == ["reserve"]
    assert not tfr.RecvReqInfo.from_bytes(sent[0][1]).bounce_jit


class _FakeRouter:
    def __init__(self):
        self.sent = []

    def send(self, frames, recipient=None):
        self.sent.append((recipient, frames))


class _ListenerMessenger:
    def __init__(self):
        self.handler = None

    def start_listener(self, handler):
        self.handler = handler


@needs_transfer
@pytest.mark.parametrize("jit", [False, True])
def test_listener_answers_grant_requests_only_with_jit(monkeypatch, jit):
    rx = object.__new__(tfr.Receiver)
    rx._bounce_jit = jit
    rx._messenger = _ListenerMessenger()
    handled, errors = [], []
    monkeypatch.setattr(
        rx, "_handle_bounce_grant", lambda send_id, msg: handled.append(send_id), raising=False
    )
    monkeypatch.setattr(tfr.logger, "error", lambda msg, *a, **k: errors.append(msg))
    rx._start_listener()
    payload = tfr._GRANT_REQ.pack(2, 11, 0, 4096, 77)
    assert rx._messenger.handler([b"id", tfr.MessageType.BOUNCE_GRANT_REQUEST, payload]) is True
    if jit:
        assert handled == [b"id"] and errors == []
    else:
        # without the knob the request is an unknown message type, as before the grant existed
        assert handled == []
        assert len(errors) == 1 and "unknown message type" in errors[0]


@needs_transfer
def test_handle_bounce_grant_without_session_answers_no(monkeypatch):
    rx = object.__new__(tfr.Receiver)
    rx._messenger = _FakeRouter()
    rx._bounce = SimpleNamespace(grant=lambda *a: pytest.fail("no grant without a session"))
    monkeypatch.setattr(rx, "_get_session", lambda rid: None, raising=False)
    rx._handle_bounce_grant(
        b"id", [tfr.MessageType.BOUNCE_GRANT_REQUEST, tfr._GRANT_REQ.pack(2, 11, 0, 4096, 77)]
    )
    ((recipient, frames),) = rx._messenger.sent
    assert recipient == b"id" and frames[0] == tfr.MessageType.BOUNCE_GRANT
    assert tfr._GRANT_REPLY.unpack(frames[1]) == (77, int(Verdict.NO), 0, 0)


@needs_transfer
def test_handle_bounce_grant_uses_the_result_key(monkeypatch):
    rx = object.__new__(tfr.Receiver)
    rx._messenger = _FakeRouter()
    asked = []

    def grant(key, rank, nbytes):
        asked.append((key, rank, nbytes))
        return int(Verdict.GRANT), 0x9000, nbytes

    rx._bounce = SimpleNamespace(grant=grant)
    session = SimpleNamespace(disagg_request_id=11, _kv_tasks=[SimpleNamespace(slice_id=5)])
    monkeypatch.setattr(rx, "_get_session", lambda rid: session, raising=False)
    msg = [tfr.MessageType.BOUNCE_GRANT_REQUEST, tfr._GRANT_REQ.pack(2, 11, 0, 4096, 78)]
    rx._handle_bounce_grant(b"id", msg)
    assert asked == [((11, 5), 2, 4096)]
    assert tfr._GRANT_REPLY.unpack(rx._messenger.sent[0][1][1]) == (78, 1, 0x9000, 4096)
    # a slice index out of range answers NO without asking the transport
    msg = [tfr.MessageType.BOUNCE_GRANT_REQUEST, tfr._GRANT_REQ.pack(2, 11, 3, 4096, 79)]
    rx._handle_bounce_grant(b"id", msg)
    assert len(asked) == 1
    assert tfr._GRANT_REPLY.unpack(rx._messenger.sent[1][1][1])[:2] == (79, 0)
