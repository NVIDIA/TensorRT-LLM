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
"""CPU-only unit tests for the native-disagg KV ``bounce`` module.

Covers sizing/OOM-guard math, the KV_AGENT_RESULT wire format (struct prefix +
result tail), the NoBounce no-op fallback, and the TP fan-in
reserve/writer_base/accumulate logic (with the GPU allocators/streams mocked out).
"""

import queue
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native.bounce import config as bcfg

# transport/transfer pull CUDA-binding + torch deps at import; skip gracefully when those are
# absent (CPU-only env). Catch only ImportError so a genuine bug in the module still fails CI
# instead of being silently turned into a skip.
try:
    from tensorrt_llm._torch.disaggregation.native.bounce import buffer as bbuf
    from tensorrt_llm._torch.disaggregation.native.bounce import transport as btr

    _HAVE_TRANSPORT = True
except ImportError:  # pragma: no cover - CPU-only env without CUDA bindings
    _HAVE_TRANSPORT = False

_MIB = 1024 * 1024


# --------------------------------------------------------------------------- #
# config.py — sizing + OOM guard (pure math, always runnable)
# --------------------------------------------------------------------------- #
class TestSizing:
    @pytest.mark.parametrize(
        "a,b,want", [(0, 8, 0), (1, 8, 8), (8, 8, 8), (9, 8, 16), (33, 32, 64)]
    )
    def test_round_up(self, a, b, want):
        assert bcfg._round_up(a, b) == want

    def test_fixed_sizing_rounds_up_to_chunk(self):
        chunk = 32 * _MIB
        ctx = bcfg.SizingContext(
            free_bytes=80 * _MIB, total_bytes=200 * _MIB, chunk_bytes=chunk, device_id=0
        )
        # 50 MiB requested -> rounded up to a 64 MiB (2-chunk) multiple.
        assert bcfg.FixedSizing(capacity_mb=50).resolve(ctx) == 64 * _MIB

    def test_fixed_sizing_floor_is_one_chunk(self):
        chunk = 32 * _MIB
        ctx = bcfg.SizingContext(
            free_bytes=80 * _MIB, total_bytes=200 * _MIB, chunk_bytes=chunk, device_id=0
        )
        # below one chunk still gets a whole chunk.
        assert bcfg.FixedSizing(capacity_mb=1).resolve(ctx) == chunk

    def test_fit_within_free_clamps_to_half_split_two_regions(self):
        chunk = 32 * _MIB
        # max_free_fraction=0.5 of 1024 MiB = 512 MiB, /2 regions = 256 MiB.
        got = bcfg.fit_within_free(want := 4096 * _MIB, free_bytes=1024 * _MIB, chunk_bytes=chunk)
        assert got == 256 * _MIB
        assert got < want

    def test_fit_within_free_keeps_request_when_it_fits(self):
        chunk = 32 * _MIB
        got = bcfg.fit_within_free(128 * _MIB, free_bytes=4096 * _MIB, chunk_bytes=chunk)
        assert got == 128 * _MIB

    def test_fit_within_free_none_when_too_small(self):
        chunk = 32 * _MIB
        # only 32 MiB free -> half/2 = 8 MiB < one chunk -> cannot fit.
        assert bcfg.fit_within_free(64 * _MIB, free_bytes=32 * _MIB, chunk_bytes=chunk) is None

    def test_config_defaults(self):
        cfg = bcfg.Config()
        assert isinstance(cfg.sizing, bcfg.FixedSizing)
        assert cfg.chunk_mb == 32 and cfg.min_blocks == 96


# --------------------------------------------------------------------------- #
# config.py — config on/off switch (size knob doubles as the enable flag)
# --------------------------------------------------------------------------- #
class TestConfigFromSize:
    @pytest.mark.parametrize("size_mb", [0, -1, None])
    def test_non_positive_is_off(self, size_mb):
        assert bcfg.config_from_size(size_mb) is None

    def test_positive_enables_with_capacity(self):
        cfg = bcfg.config_from_size(2048)
        assert isinstance(cfg, bcfg.Config)
        assert cfg.sizing.capacity_mb == 2048


# --------------------------------------------------------------------------- #
# wire format — KV_AGENT_RESULT struct prefix + bounce result tail
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAVE_TRANSPORT, reason="bounce.transport import needs CUDA bindings")
class TestResultTail:
    def _wm(self, dst, sizes, base):
        return SimpleNamespace(
            dst_ptrs=np.array(dst, dtype=np.int64),
            sizes=np.array(sizes, dtype=np.int64),
            bounce_dst_base=base,
        )

    def test_roundtrip_with_tail(self):
        wm = self._wm([100, 200, 300], [16, 16, 16], 0xABCD)
        # message = [msg_type, packed_prefix, *tail]; tail lives at [2:].
        msg = [b"KV_AGENT_RESULT", b"prefix"] + btr.encode_result_tail(wm)
        assert len(msg) == 5
        dst, sizes, src_base = btr.decode_result_tail(msg)
        assert np.array_equal(dst, wm.dst_ptrs)
        assert np.array_equal(sizes, wm.sizes)
        assert src_base == 0xABCD

    def test_no_tail_returns_none(self):
        # non-bounced result: only [msg_type, prefix] -> no tail.
        assert btr.decode_result_tail([b"KV_AGENT_RESULT", b"prefix"]) == (None, None, None)

    def test_encode_tail_handles_unset_base(self):
        wm = self._wm([1, 2], [8, 8], None)
        tail = btr.encode_result_tail(wm)
        assert int(np.frombuffer(tail[2], dtype=np.int64)[0]) == 0


def test_kv_result_prefix_roundtrip():
    """The KV_AGENT_RESULT binary prefix (transfer.py) must round-trip exactly."""
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    for rank, rid, sl, last, status in [
        (7, 6925227277844486, 42, True, tfr.AgentResult.SUCCESS),
        (0, 1, 0, False, tfr.AgentResult.FAILED),
        (31, 2**62, 9999, True, tfr.AgentResult.SUCCESS),
    ]:
        packed = tfr._KV_RESULT_PREFIX.pack(rank, rid, sl, last, tfr._AGENT_RESULT_CODE[status])
        r, i, s, last_out, c = tfr._KV_RESULT_PREFIX.unpack(packed)
        assert (r, i, s, last_out) == (rank, rid, sl, last)
        assert tfr._AGENT_RESULT_BY_CODE[c] is status


@pytest.mark.parametrize("result_name", ["SUCCESS", "FAILED"])
def test_make_kv_result_msg_uses_binary_frame(result_name):
    """Every KV result (success and failure) uses the binary frame so the receiver can decode it."""
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    result = getattr(tfr.AgentResult, result_name)
    msg = tfr._make_kv_result_msg(3, 12345, 7, True, result)
    assert msg[0] == tfr.MessageType.KV_AGENT_RESULT
    assert len(msg) == 2  # prefix only; no bounce tail when none is passed
    r, rid, sl, last, code = tfr._KV_RESULT_PREFIX.unpack(msg[1])
    assert (r, rid, sl, last) == (3, 12345, 7, True)
    assert tfr._AGENT_RESULT_BY_CODE[code] is result


# --------------------------------------------------------------------------- #
# NoBounce — disabled no-op fallback
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAVE_TRANSPORT, reason="bounce.transport import needs CUDA bindings")
class TestNoBounce:
    def test_noop_behaviour(self):
        nb = btr.NoBounce()
        assert nb.enabled is False
        assert nb.reserve(SimpleNamespace()) is False
        assert nb.build_request(SimpleNamespace(), 0) is None
        assert nb.writer_base(("r", 0), 1) is None
        assert nb.is_bounced(("r", 0)) is False
        nb.accumulate_and_scatter(("r", 0), 1)  # no-op, must not raise
        nb.release_reservation(("r", 0))  # no-op, must not raise
        nb.release_send(0)
        nb.close()

    def test_create_bounce_none_cfg(self):
        assert isinstance(
            btr.create_bounce(object(), None, device_id=0, page_table=None), btr.NoBounce
        )


# --------------------------------------------------------------------------- #
# Transport TP fan-in logic — reserve gate / writer_base / accumulate ordering
# (GPU allocators, streams and the scatter worker are mocked out)
# --------------------------------------------------------------------------- #
class _FakeAlloc:
    """Stand-in for SlotAllocator: hands out a fixed base, records reservations."""

    def __init__(self, capacity_bytes, phys_chunk_size, name="kv_bounce"):
        self._cap = capacity_bytes
        self.base = 0x100000
        self.next_id = 0
        self.released = []

    @property
    def capacity(self):
        return self._cap

    def reserve(self, size, timeout=None):
        if size > self._cap:
            return None
        sid = self.next_id
        self.next_id += 1
        return sid, self.base

    def release(self, slot_id):
        self.released.append(slot_id)

    def reg_descs(self):
        return []


def _make_transport(monkeypatch, recv_slot_bytes, capacity=1 << 30, min_blocks=1):
    monkeypatch.setattr(btr, "SlotAllocator", _FakeAlloc)
    monkeypatch.setattr(btr.Transport, "_new_stream", lambda self: 0)
    monkeypatch.setattr(
        btr.Transport,
        "_start_scatter_worker",
        lambda self, name: setattr(self, "_scatter_q", queue.Queue()),
    )
    agent = SimpleNamespace(register_memory=lambda d: None)
    return btr.Transport(
        agent,
        device_id=0,
        capacity_bytes=capacity,
        phys_chunk_size=32 * _MIB,
        recv_slot_bytes=recv_slot_bytes,
        min_blocks=min_blocks,
    )


def _recv_req(block_counts, rid=1, slice_id=0):
    return SimpleNamespace(
        block_ids_per_layer_groups=[SimpleNamespace(size=n) for n in block_counts],
        unique_rid=rid,
        slice_id=slice_id,
        bounce_dst_base=None,
    )


@pytest.mark.skipif(not _HAVE_TRANSPORT, reason="bounce.transport import needs CUDA bindings")
class TestFanInReserve:
    def test_reserve_stamps_base_and_per_writer(self, monkeypatch):
        t = _make_transport(monkeypatch, recv_slot_bytes=[100])
        req = _recv_req([2])  # total = 2 * 100 = 200
        assert t.reserve(req, num_writers=2) is True
        assert req.bounce_dst_base == 0x100000
        # writer i lands at base + i * (total // num_writers); per_writer = 100.
        assert t.writer_base((req.unique_rid, req.slice_id), 0) == 0x100000
        assert t.writer_base((req.unique_rid, req.slice_id), 1) == 0x100000 + 100

    def test_reserve_uneven_fanin_falls_back(self, monkeypatch):
        t = _make_transport(monkeypatch, recv_slot_bytes=[3])
        req = _recv_req([1])  # total = 3, not divisible by 2
        assert t.reserve(req, num_writers=2) is False
        assert req.bounce_dst_base is None

    def test_reserve_single_writer_ok(self, monkeypatch):
        t = _make_transport(monkeypatch, recv_slot_bytes=[3])
        req = _recv_req([1])  # total = 3, num_writers=1 -> no even-split requirement
        assert t.reserve(req, num_writers=1) is True

    def test_reserve_too_small_falls_back(self, monkeypatch):
        t = _make_transport(monkeypatch, recv_slot_bytes=[100], min_blocks=96)
        assert t.reserve(_recv_req([4]), num_writers=1) is False  # 4 < 96 blocks

    def test_reserve_unknown_slot_size_falls_back(self, monkeypatch):
        t = _make_transport(monkeypatch, recv_slot_bytes=[100])  # only 1 group known
        assert t.reserve(_recv_req([2, 2]), num_writers=1) is False  # 2nd group unknown

    def test_reserve_oversize_falls_back(self, monkeypatch):
        t = _make_transport(monkeypatch, recv_slot_bytes=[1000], capacity=500)
        assert t.reserve(_recv_req([2]), num_writers=1) is False  # total 2000 > cap 500

    def test_accumulate_orders_by_src_base(self, monkeypatch):
        t = _make_transport(monkeypatch, recv_slot_bytes=[100])
        req = _recv_req([2])
        assert t.reserve(req, num_writers=2) is True
        key = (req.unique_rid, req.slice_id)
        # writer for the HIGHER src_base arrives first; scatter must reorder by src_base.
        t.accumulate_and_scatter(
            key, 2, np.array([20], dtype=np.int64), np.array([8], dtype=np.int64), src_base=200
        )
        assert t._scatter_q.empty()  # not all writers in yet
        t.accumulate_and_scatter(
            key, 2, np.array([10], dtype=np.int64), np.array([8], dtype=np.int64), src_base=100
        )
        slot_id, addr, plan, on_done = t._scatter_q.get_nowait()
        # sorted by src_base (100 before 200) -> dst_ptrs == [10, 20]
        assert list(plan.dst_ptrs) == [10, 20]
        assert list(plan.sizes) == [8, 8]

    def test_on_done_deferred_to_worker_item(self, monkeypatch):
        # The completion cb is NOT fired inline; it rides the scatter-queue item so the worker
        # can invoke it only after the scatter has landed (closing the completion-before-scatter race).
        t = _make_transport(monkeypatch, recv_slot_bytes=[100])
        req = _recv_req([2])
        assert t.reserve(req, num_writers=1) is True
        key = (req.unique_rid, req.slice_id)
        calls = []
        t.accumulate_and_scatter(
            key,
            1,
            np.array([10], dtype=np.int64),
            np.array([8], dtype=np.int64),
            src_base=0,
            on_done=lambda ok: calls.append(ok),
        )
        slot_id, addr, plan, on_done = t._scatter_q.get_nowait()
        assert calls == []  # deferred, not fired inline
        on_done(True)  # the worker fires this after cudaStreamSynchronize
        assert calls == [True]

    def test_empty_acc_fires_on_done_inline_and_releases(self, monkeypatch):
        # Bounced SUCCESS that carried no scatter tail: nothing to copy, but the task must still
        # complete -> on_done(True) inline + slot released, nothing queued.
        t = _make_transport(monkeypatch, recv_slot_bytes=[100])
        req = _recv_req([2])
        assert t.reserve(req, num_writers=1) is True
        key = (req.unique_rid, req.slice_id)
        calls = []
        t.accumulate_and_scatter(key, 1, None, None, on_done=lambda ok: calls.append(ok))
        assert calls == [True]
        assert t._scatter_q.empty()
        assert t._recv_alloc.released  # slot freed

    def test_entry_none_fires_on_done_false(self, monkeypatch):
        # Never-reserved key (would be a lost completion): surface as failure, don't hang.
        t = _make_transport(monkeypatch, recv_slot_bytes=[100])
        calls = []
        t.accumulate_and_scatter(
            ("missing", 0),
            1,
            np.array([10], dtype=np.int64),
            np.array([8], dtype=np.int64),
            src_base=0,
            on_done=lambda ok: calls.append(ok),
        )
        assert calls == [False]

    def test_scatter_write_result_non_bounce_fires_on_done(self):
        # Non-bounced path completes inline (the in-place WRITE already landed the KV).
        calls = []
        btr.scatter_write_result(
            btr.NoBounce(), ("r", 0), 1, None, None, on_done=lambda ok: calls.append(ok)
        )
        assert calls == [True]

    def test_release_reservation_frees_slot_and_is_idempotent(self, monkeypatch):
        # FAILED/cancel must release the reserved recv slot (else the arena leaks); idempotent.
        t = _make_transport(monkeypatch, recv_slot_bytes=[100])
        req = _recv_req([2])
        assert t.reserve(req, num_writers=1) is True
        key = (req.unique_rid, req.slice_id)
        assert t.is_bounced(key) is True
        t.release_reservation(key)
        assert t.is_bounced(key) is False
        assert t._recv_alloc.released  # slot freed
        t.release_reservation(key)  # already gone -> no-op, must not raise


# --------------------------------------------------------------------------- #
# SlotAllocator — first-fit reuses out-of-order freed holes (real Buffer mocked)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAVE_TRANSPORT, reason="bounce.transport import needs CUDA bindings")
class TestSlotAllocator:
    def _alloc(self, monkeypatch, cap):
        # SlotAllocator.__init__ resolves Buffer from the buffer module's globals, so patch it there.
        monkeypatch.setattr(
            bbuf,
            "Buffer",
            lambda *a, **k: SimpleNamespace(
                size=cap, base_ptr=0x1000, reg_descs=lambda: [], close=lambda: None
            ),
        )
        return btr.SlotAllocator(cap, 512)

    def test_reuses_out_of_order_freed_hole(self, monkeypatch):
        a = self._alloc(monkeypatch, cap=1024)  # two 512-byte slots
        _sa, _ = a.reserve(512)  # [0, 512) stays live
        sb, _ = a.reserve(512)  # [512, 1024)
        a.release(sb)  # free the TAIL hole out of order
        # A bump cursor sitting at 1024 would wrap to 0, hit the live [0,512) and block; first-fit
        # must reuse [512, 1024) immediately.
        res = a.reserve(512, timeout=0.5)
        assert res is not None
        _, addr = res
        assert addr == 0x1000 + 512

    def test_blocks_until_release_when_full(self, monkeypatch):
        a = self._alloc(monkeypatch, cap=512)  # single slot
        a.reserve(512)
        assert a.reserve(512, timeout=0.05) is None  # full -> times out
