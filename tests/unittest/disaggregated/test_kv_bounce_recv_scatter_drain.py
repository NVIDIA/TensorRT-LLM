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
"""CPU-only tests for the opt-in batched receive scatter (TRTLLM_KV_BOUNCE_RECV_SCATTER_DRAIN).

The scatter worker runs on the test thread with the copy and the stream sync replaced by recorders,
so each test checks the exact sequence of scatter calls and syncs, the order in which completions
fire, and that every slot comes back.
"""

import queue
import random
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native.bounce import config as bcfg

try:
    from tensorrt_llm._torch.disaggregation.native.bounce import impl as btr

    _HAVE_TRANSPORT = True
except ImportError:  # pragma: no cover - CPU-only env without CUDA bindings
    _HAVE_TRANSPORT = False

needs_transport = pytest.mark.skipif(
    not _HAVE_TRANSPORT, reason="bounce impl import needs CUDA bindings"
)

_ENV = "TRTLLM_KV_BOUNCE_RECV_SCATTER_DRAIN"
_MIB = 1024 * 1024
_BLOCK = 1024
_POISON_DST = 0xBAD000


# ---------------------------------------------------------------------------
# knob
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [(None, False), ("", False), ("0", False), ("1", True)])
def test_recv_scatter_drain_from_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, raw)
    assert bcfg.recv_scatter_drain_from_env() is expected


@pytest.mark.parametrize("raw", ["2", "true", "on", "-1"])
def test_recv_scatter_drain_rejects_bad_value(monkeypatch, raw):
    monkeypatch.setenv(_ENV, raw)
    with pytest.raises(ValueError, match=_ENV):
        bcfg.recv_scatter_drain_from_env()


@needs_transport
def test_create_bounce_bad_drain_raises_instead_of_disabling(monkeypatch):
    monkeypatch.setenv(_ENV, "yes")

    def _no_build(*args, **kwargs):
        raise AssertionError("from_config must not run for a bad value")

    monkeypatch.setattr(btr.VmmBounceTransport, "from_config", classmethod(_no_build))
    with pytest.raises(ValueError, match=_ENV):
        btr.create_bounce(SimpleNamespace(), SimpleNamespace(), device_id=0, page_table=None)


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


class _Alloc:
    """A CPU bump allocator with the SlotAllocator surface the receive side uses."""

    def __init__(self, capacity_bytes, phys_chunk_size, name="kv_bounce"):
        self.capacity = capacity_bytes
        self.in_use = {}
        self.released = []
        self.quarantined = []
        self._next = 0
        self._cursor = 0x100000

    def reserve(self, size, timeout=None):
        sid = self._next
        self._next += 1
        addr = self._cursor
        self._cursor += size
        self.in_use[sid] = size
        return sid, addr

    def release(self, slot_id):
        assert slot_id in self.in_use, f"slot {slot_id} released twice or never reserved"
        del self.in_use[slot_id]
        self.released.append(slot_id)

    def quarantine(self, slot_id, grace_s):
        self.in_use.pop(slot_id, None)
        self.quarantined.append(slot_id)

    def reclaim_expired(self):
        return 0

    def reg_descs(self):
        return []

    def close(self):
        pass


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg, *a, **k):
        self.lines.append(("info", msg))

    def warning(self, msg, *a, **k):
        self.lines.append(("warning", msg))

    def error(self, msg, *a, **k):
        self.lines.append(("error", msg))

    def info_once(self, msg, *a, **k):
        pass

    def warning_once(self, msg, *a, **k):
        pass

    def of(self, level):
        return [m for lv, m in self.lines if lv == level]


def _transport(monkeypatch, drain, alloc=_Alloc):
    if drain is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, drain)
    # Some cases queue just-in-time writers, which register only with the grant on.
    monkeypatch.setenv("TRTLLM_KV_BOUNCE_JIT", "1")
    monkeypatch.setattr(btr, "SlotAllocator", alloc)
    monkeypatch.setattr(btr.VmmBounceTransport, "_new_stream", lambda self, mode=0: 0)
    monkeypatch.setattr(
        btr.VmmBounceTransport,
        "_start_scatter_worker",
        lambda self, name: setattr(self, "_scatter_q", queue.Queue()),
    )
    log = _Log()
    monkeypatch.setattr(btr, "logger", log)
    t = btr.VmmBounceTransport(
        SimpleNamespace(register_memory=lambda d: None),
        device_id=0,
        capacity_bytes=64 * _MIB,
        phys_chunk_size=32 * _MIB,
        block_bytes_per_group=[_BLOCK],
        min_blocks=1,
        direction="recv",
    )
    t._scatter_stream = 0
    t._stop = threading.Event()
    return t, log


def _patch_cuda(monkeypatch):
    """Record every scatter (its destination addresses) and every sync, in call order.

    A scatter whose destinations include _POISON_DST raises, like a copy that fails on the host side.
    """
    calls = []

    def fake_scatter(src_base, dst_ptrs, sizes, offsets, *, stream):
        dst = [int(d) for d in dst_ptrs]
        calls.append(("scatter", dst))
        if _POISON_DST in dst:
            raise RuntimeError("poisoned fragment")

    monkeypatch.setattr(btr, "scatter_contiguous", fake_scatter)
    monkeypatch.setattr(
        btr,
        "cudart",
        SimpleNamespace(
            cudaSetDevice=lambda dev: (0,),
            cudaStreamSynchronize=lambda stream: calls.append(("sync",)) or (0,),
        ),
    )
    monkeypatch.setattr(btr, "CUASSERT", lambda ret: ret)
    return calls


def _req(rid, nblocks=8):
    return SimpleNamespace(
        block_ids_per_layer_groups=[np.arange(nblocks, dtype=np.int64)],
        unique_rid=rid,
        slice_id=0,
        bounce_dst_base=None,
    )


def _tail(dst):
    return np.array(dst, dtype=np.int64), np.full(len(dst), 64, dtype=np.int64)


def _queue_region(t, rid, dst, done):
    """Queue one dispatch-time region (a (ctx, descs) item) whose single writer reported.

    Returns the region's base, which is the writer's source.
    """
    req = _req(rid)
    assert t.reserve(req, num_writers=1)
    t.record_result(
        (rid, 0),
        0,
        *_tail(dst),
        src_base=req.bounce_dst_base,
        on_done=lambda ok: done.append((rid, ok)),
    )
    return req.bounce_dst_base


def _register_jit(t, rid, ranks, done):
    assert t.register_jit(_req(rid, nblocks=8 * len(ranks)), writer_ranks=ranks)
    t._reserved_map[(rid, 0)].on_done = lambda ok: done.append((rid, ok))
    return {rank: t.grant((rid, 0), rank, 8 * _BLOCK)[1] for rank in ranks}


def _queue_jit_writer(t, rid, rank, base, dst):
    """Queue one just-in-time writer (a (ctx, descs, rank) item)."""
    t.record_result((rid, 0), rank, *_tail(dst), src_base=base)


def _run(t):
    t._scatter_q.put(None)  # poison pill behind the queued items
    t._scatter_loop()


def _queue_mixed(t, done):
    """Queue, in this order: region 1, jit 2 writer 0, region 3, jit 2 writer 1, jit 4 writer 0."""
    _queue_region(t, 1, [0x1000, 0x1100], done)
    bases2 = _register_jit(t, 2, (0, 1), done)
    bases4 = _register_jit(t, 4, (0,), done)
    _queue_jit_writer(t, 2, 0, bases2[0], [0x2000])
    _queue_region(t, 3, [0x3000], done)
    _queue_jit_writer(t, 2, 1, bases2[1], [0x2100, 0x2200])
    _queue_jit_writer(t, 4, 0, bases4[0], [0x4000])
    assert t._scatter_q.qsize() == 5
    return [[0x1000, 0x1100], [0x2000], [0x3000], [0x2100, 0x2200], [0x4000]]


# ---------------------------------------------------------------------------
# knob off: the per-item worker, unchanged
# ---------------------------------------------------------------------------


@needs_transport
@pytest.mark.parametrize("drain", [None, "", "0"])
def test_drain_off_keeps_one_scatter_and_sync_per_item(monkeypatch, drain):
    t, log = _transport(monkeypatch, drain)
    assert t._scatter_drain is False
    calls = _patch_cuda(monkeypatch)
    done = []
    per_item = _queue_mixed(t, done)
    _run(t)
    expected = []
    for dst in per_item:
        expected += [("scatter", dst), ("sync",)]
    assert calls == expected
    assert done == [(1, True), (3, True), (2, True), (4, True)]
    assert t._recv_alloc.in_use == {} and t._reserved_map == {}
    assert t._drain_stats == {"syncs": 0, "items": 0, "max_items": 0, "retried": 0}
    assert not [m for m in log.of("info") if "drain" in m]


# ---------------------------------------------------------------------------
# knob on
# ---------------------------------------------------------------------------


@needs_transport
def test_drain_batches_mixed_items_into_one_scatter_and_one_sync(monkeypatch):
    t, log = _transport(monkeypatch, "1")
    assert t._scatter_drain is True
    calls = _patch_cuda(monkeypatch)
    done = []
    syncs_seen_at_done = []
    per_item = _queue_mixed(t, done)

    # every completion fires only after the one sync
    for ctx in t._reserved_map.values():
        cb = ctx.on_done
        ctx.on_done = lambda ok, cb=cb: (
            syncs_seen_at_done.append(calls.count(("sync",))),
            cb(ok),
        )
    _run(t)
    assert calls == [("scatter", [d for dst in per_item for d in dst]), ("sync",)]
    # finished in queue order: region 1, region 3, then jit 2 after its last writer, then jit 4
    assert done == [(1, True), (3, True), (2, True), (4, True)]
    assert syncs_seen_at_done == [1, 1, 1, 1]
    assert t._recv_alloc.in_use == {} and t._reserved_map == {}
    assert t._jit_stats["inflight_bytes"] == 0
    assert t._drain_stats == {"syncs": 1, "items": 5, "max_items": 5, "retried": 0}
    assert "[kv-bounce] drain: syncs=1 items=5 max_items=5 retried=0" in log.of("info")


@needs_transport
def test_drain_finishes_jit_writers_in_queue_order(monkeypatch):
    """A jit group's slots are released one by one in queue order.

    The group settles at its last writer, not before.
    """
    t, _log = _transport(monkeypatch, "1")
    _patch_cuda(monkeypatch)
    done = []
    bases = _register_jit(t, 9, (0, 1, 2), done)
    slots = {rank: t._reserved_map[(9, 0)].lease(rank).slot_id for rank in (0, 1, 2)}
    for rank in (2, 0, 1):
        _queue_jit_writer(t, 9, rank, bases[rank], [0x9000 + rank])
    _run(t)
    assert t._recv_alloc.released == [slots[2], slots[0], slots[1]]
    assert done == [(9, True)]


@needs_transport
def test_drain_isolates_a_failing_item(monkeypatch):
    t, log = _transport(monkeypatch, "1")
    calls = _patch_cuda(monkeypatch)
    done = []
    _queue_region(t, 1, [0x1000], done)
    bases = _register_jit(t, 2, (0, 1), done)
    _queue_jit_writer(t, 2, 0, bases[0], [0x2000])
    _queue_region(t, 3, [0x3000, _POISON_DST], done)  # the bad item
    _queue_jit_writer(t, 2, 1, bases[1], [0x2100])
    _queue_region(t, 5, [0x5000], done)
    _run(t)
    batch = [0x1000, 0x2000, 0x3000, _POISON_DST, 0x2100, 0x5000]
    assert calls == [
        ("scatter", batch),  # the batch raised before its sync
        ("scatter", [0x1000]),
        ("sync",),
        ("scatter", [0x2000]),
        ("sync",),
        ("scatter", [0x3000, _POISON_DST]),  # raised again: only this item fails
        ("scatter", [0x2100]),
        ("sync",),
        ("scatter", [0x5000]),
        ("sync",),
    ]
    assert done == [(1, True), (3, False), (2, True), (5, True)]
    assert t._recv_alloc.in_use == {} and t._reserved_map == {}
    assert t._drain_stats == {"syncs": 6, "items": 5, "max_items": 5, "retried": 5}
    assert any("batched scatter of 5 items failed" in m for m in log.of("warning"))
    # slots: region 1 -> 0, jit 2 writers -> 1 and 2, region 3 -> 3
    assert [m for m in log.of("error") if "scatter failed" in m] == [
        "[kv-bounce] scatter failed (slot=3): poisoned fragment"
    ]


@needs_transport
def test_drain_failing_jit_writer_fails_only_its_group(monkeypatch):
    t, log = _transport(monkeypatch, "1")
    _patch_cuda(monkeypatch)
    done = []
    bases_a = _register_jit(t, 6, (0, 1), done)
    bases_b = _register_jit(t, 7, (0,), done)
    _queue_jit_writer(t, 6, 0, bases_a[0], [0x6000])
    _queue_jit_writer(t, 6, 1, bases_a[1], [_POISON_DST])
    _queue_jit_writer(t, 7, 0, bases_b[0], [0x7000])
    _run(t)
    assert done == [(6, False), (7, True)]
    assert t._recv_alloc.in_use == {} and t._reserved_map == {}
    assert t._jit_stats["inflight_bytes"] == 0
    assert any("jit scatter failed (rid_slice=(6, 0), writer=1)" in m for m in log.of("error"))


@needs_transport
def test_drain_single_failing_item_is_not_scattered_twice(monkeypatch):
    t, log = _transport(monkeypatch, "1")
    calls = _patch_cuda(monkeypatch)
    done = []
    _queue_region(t, 1, [_POISON_DST], done)
    _run(t)
    assert calls == [("scatter", [_POISON_DST])]  # same as the per-item worker
    assert done == [(1, False)]
    assert t._recv_alloc.in_use == {}
    assert t._drain_stats["retried"] == 0
    assert not log.of("warning")


@needs_transport
def test_drain_finish_failure_does_not_strand_later_items(monkeypatch):
    t, log = _transport(monkeypatch, "1")
    _patch_cuda(monkeypatch)
    done = []
    _queue_region(t, 1, [0x1000], done)
    _queue_region(t, 2, [0x2000], done)
    _queue_region(t, 3, [0x3000], done)
    real_apply = t._apply

    def apply(rid_slice, mutate):
        if rid_slice == (2, 0):
            raise RuntimeError("finish blew up")
        real_apply(rid_slice, mutate)

    t._apply = apply
    _run(t)
    assert done == [(1, True), (3, True)]
    assert any("finishing rid_slice=(2, 0) failed" in m for m in log.of("error"))


@needs_transport
def test_drain_exits_at_the_poison_pill_after_finishing_the_drained_items(monkeypatch):
    t, _log = _transport(monkeypatch, "1")
    calls = _patch_cuda(monkeypatch)
    done = []
    _queue_region(t, 1, [0x1000], done)
    _queue_region(t, 2, [0x2000], done)
    t._scatter_q.put(None)  # close() raced in behind two items
    _queue_region(t, 3, [0x3000], done)
    t._scatter_loop()
    assert calls == [("scatter", [0x1000, 0x2000]), ("sync",)]
    assert done == [(1, True), (2, True)]
    assert t._scatter_q.qsize() == 1  # the item behind the pill is left, as without the drain


@needs_transport
def test_drain_stat_line_is_rate_limited(monkeypatch):
    t, log = _transport(monkeypatch, "1")
    _patch_cuda(monkeypatch)
    done = []
    _queue_region(t, 1, [0x1000], done)
    assert t._scatter_drained(t._scatter_q.get_nowait()) is False
    _queue_region(t, 2, [0x2000], done)
    _queue_region(t, 3, [0x3000], done)
    assert t._scatter_drained(t._scatter_q.get_nowait()) is False
    lines = [m for m in log.of("info") if m.startswith("[kv-bounce] drain:")]
    assert lines == ["[kv-bounce] drain: syncs=1 items=1 max_items=1 retried=0"]
    assert t._drain_stats == {"syncs": 2, "items": 3, "max_items": 2, "retried": 0}
    t._drain_stats_next_log = 0.0  # the period elapsed
    _queue_region(t, 4, [0x4000], done)
    t._scatter_drained(t._scatter_q.get_nowait())
    lines = [m for m in log.of("info") if m.startswith("[kv-bounce] drain:")]
    assert lines[-1] == "[kv-bounce] drain: syncs=3 items=4 max_items=2 retried=0"


@needs_transport
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_drain_random_batches_settle_everything_once(monkeypatch, seed):
    """Random mixes of regions and jit writers, poisoned fragments and batch sizes.

    Every request settles exactly once, with failure exactly where a fragment was poisoned, and no
    slot is left.
    """
    rng = random.Random(seed)
    t, _log = _transport(monkeypatch, "1")
    _patch_cuda(monkeypatch)
    done = []
    expect_ok = {}
    pending = []  # queued-later writers of registered jit requests: (rid, rank, base, dst)
    rid = 0
    for _round in range(40):
        for _ in range(rng.randint(1, 6)):
            rid += 1
            bad = rng.random() < 0.1
            dst = [_POISON_DST if bad else 0x10000 + rid]
            if rng.random() < 0.5:
                _queue_region(t, rid, dst, done)
                expect_ok[rid] = not bad
            else:
                ranks = tuple(range(rng.randint(1, 3)))
                bases = _register_jit(t, rid, ranks, done)
                expect_ok[rid] = not bad
                for rank in ranks:
                    wdst = dst if rank == 0 else [0x20000 + rid * 4 + rank]
                    pending.append((rid, rank, bases[rank], wdst))
        rng.shuffle(pending)
        keep = []
        for item in pending:
            if rng.random() < 0.6:
                _queue_jit_writer(t, *item)
            else:
                keep.append(item)
        pending = keep
        while not t._scatter_q.empty():
            t._scatter_drained(t._scatter_q.get_nowait())
    for item in pending:
        _queue_jit_writer(t, *item)
    _run(t)
    assert sorted(r for r, _ in done) == sorted(expect_ok)
    assert dict(done) == expect_ok
    assert t._recv_alloc.in_use == {} and t._reserved_map == {}
    assert t._jit_stats["inflight_bytes"] == 0
    assert t._drain_stats["max_items"] > 1


# ---------------------------------------------------------------------------
# bytes: each item is read from its own source
# ---------------------------------------------------------------------------


def _patch_mem(monkeypatch, mem):
    """Copy bytes in ``mem`` the way the scatter does: each fragment from its absolute source."""

    def fake_scatter(src_base, dst_ptrs, sizes, offsets, *, stream):
        for d, s, n in zip(dst_ptrs, offsets, sizes):
            s = int(src_base) + int(s)
            mem[int(d) : int(d) + int(n)] = mem[s : s + int(n)]

    monkeypatch.setattr(btr, "scatter_contiguous", fake_scatter)
    monkeypatch.setattr(
        btr,
        "cudart",
        SimpleNamespace(cudaSetDevice=lambda dev: (0,), cudaStreamSynchronize=lambda stream: (0,)),
    )
    monkeypatch.setattr(btr, "CUASSERT", lambda ret: ret)


@needs_transport
@pytest.mark.parametrize("drain", [None, "1"])
def test_drain_reads_each_item_from_its_own_source(monkeypatch, drain):
    """A mixed region and jit batch lands every item's own bytes, the same as without the drain.

    Every writer's slot holds different bytes, so reading one item from another item's slot, or at
    an offset carried over from the items before it, puts the wrong bytes in the cache.
    """
    t, _log = _transport(monkeypatch, drain)
    mem = np.zeros(0x200000, dtype=np.uint8)
    _patch_mem(monkeypatch, mem)
    rng = np.random.default_rng(7)
    done = []
    expected = {}

    def write(src_base, dst):
        # the writer's fragments sit back to back at its source, 64 bytes each (see _tail)
        for k, d in enumerate(dst):
            payload = rng.integers(1, 256, 64, dtype=np.uint8)
            mem[src_base + 64 * k : src_base + 64 * (k + 1)] = payload
            expected[d] = payload

    write(_queue_region(t, 1, [0x1000, 0x1100], done), [0x1000, 0x1100])
    bases2 = _register_jit(t, 2, (0, 1), done)
    bases4 = _register_jit(t, 4, (0,), done)
    _queue_jit_writer(t, 2, 0, bases2[0], [0x2000])
    write(bases2[0], [0x2000])
    write(_queue_region(t, 3, [0x3000], done), [0x3000])
    _queue_jit_writer(t, 2, 1, bases2[1], [0x2100, 0x2200])
    write(bases2[1], [0x2100, 0x2200])
    _queue_jit_writer(t, 4, 0, bases4[0], [0x4000])
    write(bases4[0], [0x4000])
    assert t._scatter_q.qsize() == 5
    _run(t)
    assert done == [(1, True), (3, True), (2, True), (4, True)]
    for d, payload in expected.items():
        assert np.array_equal(mem[d : d + 64], payload), hex(d)
    assert t._drain_stats["max_items"] == (5 if drain == "1" else 0)


# ---------------------------------------------------------------------------
# a failure only the batch hits
# ---------------------------------------------------------------------------


def _patch_batch_only_failure(monkeypatch):
    """Like _patch_cuda, but any scatter of more than one fragment also raises.

    This is how a failed growth of the copy's metadata buffer fails a batch while every
    one-fragment item succeeds.
    """
    calls = _patch_cuda(monkeypatch)
    one_item = btr.scatter_contiguous

    def fake_scatter(src_base, dst_ptrs, sizes, offsets, *, stream):
        one_item(src_base, dst_ptrs, sizes, offsets, stream=stream)
        if len(dst_ptrs) > 1:
            raise RuntimeError("metadata buffer growth failed")

    monkeypatch.setattr(btr, "scatter_contiguous", fake_scatter)
    return calls


def _drain_pair(t, rid, done, dst=None):
    _queue_region(t, rid, [0x1000 + rid], done)
    _queue_region(t, rid + 1, dst or [0x1000 + rid + 1], done)
    assert t._scatter_drained(t._scatter_q.get_nowait()) is False


@needs_transport
def test_drain_batch_only_failure_warns_once_then_turns_batching_off(monkeypatch):
    t, log = _transport(monkeypatch, "1")
    calls = _patch_batch_only_failure(monkeypatch)
    done = []
    for k in range(btr._DRAIN_BATCH_ONLY_FAILURES_MAX):
        assert t._scatter_drain is True
        _drain_pair(t, 10 * (k + 1), done)
    assert t._scatter_drain is False
    assert all(ok for _, ok in done) and len(done) == 2 * btr._DRAIN_BATCH_ONLY_FAILURES_MAX
    # the per-drain warning is rate-limited; the counters keep every retry
    assert len([m for m in log.of("warning") if "batched scatter of 2 items failed" in m]) == 1
    assert len([m for m in log.of("warning") if "batching is off" in m]) == 1
    assert t._drain_stats["retried"] == 2 * btr._DRAIN_BATCH_ONLY_FAILURES_MAX

    # from now on the worker runs the per-item path: one scatter and one sync per item
    del calls[:]
    _queue_region(t, 90, [0x9000], done)
    _queue_region(t, 91, [0x9100], done)
    _run(t)
    assert calls == [("scatter", [0x9000]), ("sync",), ("scatter", [0x9100]), ("sync",)]
    assert done[-2:] == [(90, True), (91, True)]
    assert t._recv_alloc.in_use == {} and t._reserved_map == {}


@needs_transport
def test_drain_batch_failure_streak_resets_on_success_or_a_bad_item(monkeypatch):
    t, _log = _transport(monkeypatch, "1")
    _patch_batch_only_failure(monkeypatch)
    done = []
    streak = btr._DRAIN_BATCH_ONLY_FAILURES_MAX - 1
    for k in range(streak):
        _drain_pair(t, 10 * (k + 1), done)
    assert t._drain_batch_only_failures == streak
    # a bad item explains its batch's failure: the streak restarts
    _drain_pair(t, 50, done, dst=[_POISON_DST])
    assert done[-2:] == [(50, True), (51, False)]
    assert t._drain_batch_only_failures == 0
    for k in range(streak):
        _drain_pair(t, 60 + 10 * k, done)
    # a batch that succeeds restarts it too
    _queue_region(t, 99, [0x9900], done)
    assert t._scatter_drained(t._scatter_q.get_nowait()) is False
    assert t._drain_batch_only_failures == 0
    assert t._scatter_drain is True


# ---------------------------------------------------------------------------
# GPU: the real batched copy over a multi-item batch
# ---------------------------------------------------------------------------

# fragment sizes of the five queued items: region, jit A writer 0, region, jit A writer 1, jit B
_GPU_CASES = {
    # every fragment a multiple of 16 bytes: the batch runs as one batched copy kernel, and one
    # fragment spans more than one program (64 KiB each)
    "aligned": [[64, 4096], [48], [16, 1024, 32], [80000, 256], [256]],
    # one 40-byte fragment: the whole batch takes the per-fragment copy loop, while without the
    # drain only that item does
    "unaligned": [[64, 4096], [40], [16, 1024, 32], [80000, 256], [256]],
}


def _device_alloc(pool):
    """A bump allocator over a real device buffer, so each slot is real bounce memory."""

    class _DeviceAlloc(_Alloc):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._cursor = pool.data_ptr()
            self._end = pool.data_ptr() + pool.numel()

        def reserve(self, size, timeout=None):
            res = super().reserve(-(-size // 256) * 256, timeout)
            assert self._cursor <= self._end, "device pool too small"
            return res

    return _DeviceAlloc


def _gpu_scatter_run(monkeypatch, drain, frag_sizes, torch, bgs, real_launch):
    """Queue the five items with real device sources and destinations, then run the real worker.

    Returns the destination bytes, the expected bytes, and the batched-copy launches as (fragment
    count, whether the kernel ran).
    """
    src_pool = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    dst_pool = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    t, _log = _transport(monkeypatch, drain, alloc=_device_alloc(src_pool))
    t._scatter_stream = btr.CUASSERT(btr.cudart.cudaStreamCreate())[0]
    launches = []

    def launch(dst, src, sizes, stream, **kwargs):
        ran = real_launch(dst, src, sizes, stream, **kwargs)
        launches.append((int(dst.size), ran))
        return ran

    monkeypatch.setattr(bgs, "_launch_batched_copy", launch)

    rng = np.random.default_rng(11)
    payloads = [[rng.integers(1, 256, n, dtype=np.uint8) for n in sizes] for sizes in frag_sizes]
    dst_cursor = [0]

    def place(i, src_base):
        """Write item i's bytes at its source and pick its destinations; return its tail."""
        blob = np.concatenate(payloads[i])
        off = src_base - src_pool.data_ptr()
        src_pool[off : off + blob.size] = torch.from_numpy(blob).cuda()
        dst = []
        for n in frag_sizes[i]:
            dst.append(dst_pool.data_ptr() + dst_cursor[0])
            dst_cursor[0] += -(-(n + 64) // 256) * 256  # disjoint, 256-byte aligned
        return np.array(dst, dtype=np.int64), np.array(frag_sizes[i], dtype=np.int64)

    def nblocks(i):
        return -(-sum(frag_sizes[i]) // _BLOCK)

    done = []
    # item 0: a region
    req0 = _req(0, nblocks(0))
    assert t.reserve(req0, num_writers=1)
    tail0 = place(0, req0.bounce_dst_base)
    # jit A (writers 0 and 1, items 1 and 3) and jit B (item 4)
    per_writer = max(nblocks(1), nblocks(3))
    assert t.register_jit(_req(1, 2 * per_writer), writer_ranks=(0, 1))
    assert t.register_jit(_req(4, nblocks(4)), writer_ranks=(0,))
    base_a = {r: t.grant((1, 0), r, per_writer * _BLOCK)[1] for r in (0, 1)}
    base_b = t.grant((4, 0), 0, nblocks(4) * _BLOCK)[1]
    # item 2: a region
    req2 = _req(2, nblocks(2))
    assert t.reserve(req2, num_writers=1)
    for rid in (1, 4):
        t._reserved_map[(rid, 0)].on_done = lambda ok, rid=rid: done.append((rid, ok))

    on_done = {0: lambda ok: done.append((0, ok)), 2: lambda ok: done.append((2, ok))}
    t.record_result((0, 0), 0, *tail0, src_base=req0.bounce_dst_base, on_done=on_done[0])
    t.record_result((1, 0), 0, *place(1, base_a[0]), src_base=base_a[0])
    tail2 = place(2, req2.bounce_dst_base)
    t.record_result((2, 0), 0, *tail2, src_base=req2.bounce_dst_base, on_done=on_done[2])
    t.record_result((1, 0), 1, *place(3, base_a[1]), src_base=base_a[1])
    t.record_result((4, 0), 0, *place(4, base_b), src_base=base_b)
    assert t._scatter_q.qsize() == 5
    _run(t)
    torch.cuda.synchronize()
    btr.CUASSERT(btr.cudart.cudaStreamDestroy(t._scatter_stream))
    assert done == [(0, True), (2, True), (1, True), (4, True)]
    assert t._recv_alloc.in_use == {} and t._reserved_map == {}

    expected = np.zeros(dst_pool.numel(), dtype=np.uint8)
    cursor = 0
    for i, sizes in enumerate(frag_sizes):
        for p in payloads[i]:
            expected[cursor : cursor + p.size] = p
            cursor += -(-(p.size + 64) // 256) * 256
    return dst_pool.cpu().numpy(), expected, launches


@needs_transport
@pytest.mark.parametrize("case", sorted(_GPU_CASES))
def test_drain_gpu_batched_copy_lands_the_same_bytes(monkeypatch, case):
    """The real copy of a mixed multi-item batch lands the per-item worker's exact bytes.

    Checked on the batched kernel and on the per-fragment fallback.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from tensorrt_llm._torch.disaggregation.native.bounce import gather_scatter as bgs

    if not bgs._HAVE_TRITON:
        pytest.skip("the batched copy kernel needs Triton")
    frag_sizes = _GPU_CASES[case]
    real_launch = bgs._launch_batched_copy
    off_bytes, expected, off_launches = _gpu_scatter_run(
        monkeypatch, None, frag_sizes, torch, bgs, real_launch
    )
    on_bytes, _expected, on_launches = _gpu_scatter_run(
        monkeypatch, "1", frag_sizes, torch, bgs, real_launch
    )
    assert np.array_equal(off_bytes, expected)
    assert np.array_equal(on_bytes, off_bytes)

    aligned = [all(n % 16 == 0 for n in sizes) for sizes in frag_sizes]
    assert off_launches == [(len(sizes), ok) for sizes, ok in zip(frag_sizes, aligned)]
    assert on_launches == [(sum(len(sizes) for sizes in frag_sizes), all(aligned))]
