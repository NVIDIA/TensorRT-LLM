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

Covers sizing/OOM-guard math, the KV_AGENT_RESULT wire format, the NoBounceTransport
no-op fallback, and the TP fan-in reserve/record/settle logic (GPU allocators/streams mocked).
"""

import queue
from types import SimpleNamespace

import numpy as np
import pytest
from layout_test_utils import manager_layout

from tensorrt_llm._torch.disaggregation.native.bounce import config as bcfg

# core (the BounceTransport contract + the TransferContext state machine) is PURE (no CUDA / NIXL
# imports), so it is always importable on CPU.
from tensorrt_llm._torch.disaggregation.native.bounce import core as bcore

# transport/transfer pull CUDA-binding + torch deps at import; skip gracefully when those are
# absent (CPU-only env). Catch only ImportError so a genuine bug in the module still fails CI
# instead of being silently turned into a skip.
try:
    from tensorrt_llm._torch.disaggregation.native.bounce import buffer as bbuf
    from tensorrt_llm._torch.disaggregation.native.bounce import impl as btr

    _HAVE_TRANSPORT = True
except ImportError:  # pragma: no cover - CPU-only env without CUDA bindings
    _HAVE_TRANSPORT = False

from tensorrt_llm._torch.pyexecutor.kv_cache.cache_layout.layout_mapping import (
    map_buffers,
    sent_bytes,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.cache_layout.page_layout import (
    BufferDesc,
    LayerGroupDesc,
    PageLayout,
    RankTopology,
    grid_place,
    head_major,
    sectioned,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.cache_layout.page_table import ManagerLayout

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
        assert cfg.chunk_mb == 32
        # Byte gate for recurrent-state payloads; block gate for plain-KV payloads.
        assert cfg.min_bytes == bcfg.DEFAULT_MIN_BYTES == 2 * _MIB
        assert cfg.min_blocks == 96


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

    def test_min_bytes_defaults_and_overrides(self, monkeypatch):
        monkeypatch.delenv("TRTLLM_KV_CACHE_BOUNCE_MIN_BYTES", raising=False)
        assert bcfg.config_from_size(2048).min_bytes == 2 * _MIB  # keeps the Config default
        assert bcfg.config_from_size(2048, min_bytes=1).min_bytes == 1  # explicit arg overrides
        assert bcfg.config_from_size(2048, min_bytes=64 * _MIB).min_bytes == 64 * _MIB
        # The gate is tuned for tests via the env override (no user-facing config field).
        monkeypatch.setenv("TRTLLM_KV_CACHE_BOUNCE_MIN_BYTES", "1")
        assert bcfg.config_from_size(2048).min_bytes == 1  # env override
        assert bcfg.config_from_size(2048, min_bytes=250).min_bytes == 250  # arg beats env

    def test_min_blocks_defaults_and_overrides(self, monkeypatch):
        # The block-count gate (plain-KV payloads) keeps its original default of 96 and honors
        # the explicit arg and the env override.
        monkeypatch.delenv("TRTLLM_KV_CACHE_BOUNCE_MIN_BLOCKS", raising=False)
        assert bcfg.config_from_size(2048).min_blocks == 96  # keeps the Config default
        assert bcfg.config_from_size(2048, 250).min_blocks == 250  # explicit arg still overrides
        monkeypatch.setenv("TRTLLM_KV_CACHE_BOUNCE_MIN_BLOCKS", "48")
        assert bcfg.config_from_size(2048).min_blocks == 48  # env override
        assert bcfg.config_from_size(2048, 250).min_blocks == 250  # explicit arg beats env

    @pytest.mark.parametrize(
        "env,attr,default",
        [
            ("TRTLLM_KV_CACHE_BOUNCE_MIN_BYTES", "min_bytes", 2 * _MIB),
            ("TRTLLM_KV_CACHE_BOUNCE_MIN_BLOCKS", "min_blocks", 96),
        ],
    )
    def test_gate_env_is_parsed_defensively(self, monkeypatch, env, attr, default):
        # A bad value must not crash setup (falls back to the default); a non-positive value clamps to 1.
        for bad in ("", "auto", "1.5"):
            monkeypatch.setenv(env, bad)
            assert getattr(bcfg.config_from_size(2048), attr) == default
        for nonpos in ("0", "-5"):
            monkeypatch.setenv(env, nonpos)
            assert getattr(bcfg.config_from_size(2048), attr) == 1


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
    for rank, rid, sl, last, status, size in [
        (7, 6925227277844486, 42, True, tfr.AgentResult.SUCCESS, 4096),
        (0, 1, 0, False, tfr.AgentResult.FAILED, 0),
        (31, 2**62, 9999, True, tfr.AgentResult.SUCCESS, 2**40),
    ]:
        packed = tfr._KV_RESULT_PREFIX.pack(
            rank, rid, sl, last, tfr._AGENT_RESULT_CODE[status], size
        )
        r, i, s, last_out, c, sz = tfr._KV_RESULT_PREFIX.unpack(packed)
        assert (r, i, s, last_out, sz) == (rank, rid, sl, last, size)
        assert tfr._AGENT_RESULT_BY_CODE[c] is status


@pytest.mark.parametrize("result_name", ["SUCCESS", "FAILED"])
def test_make_kv_result_msg_uses_binary_frame(result_name):
    """Every KV result (success and failure) uses the binary frame so the receiver can decode it."""
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    result = getattr(tfr.AgentResult, result_name)
    msg = tfr._make_kv_result_msg(3, 12345, 7, True, result, transfer_size=8192)
    assert msg[0] == tfr.MessageType.KV_AGENT_RESULT
    assert len(msg) == 2  # prefix only; no bounce tail when none is passed
    r, rid, sl, last, code, size = tfr._KV_RESULT_PREFIX.unpack(msg[1])
    assert (r, rid, sl, last, size) == (3, 12345, 7, True, 8192)
    assert tfr._AGENT_RESULT_BY_CODE[code] is result


# --------------------------------------------------------------------------- #
# NoBounceTransport — disabled no-op fallback
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAVE_TRANSPORT, reason="bounce.transport import needs CUDA bindings")
class TestNoBounce:
    def test_noop_behaviour(self):
        nb = btr.NoBounceTransport()
        assert nb.enabled is False
        assert nb.reserve(SimpleNamespace(), [{0: 100}]) is False
        assert nb.build_request(SimpleNamespace()) is None
        assert nb.writer_base(("r", 0), 1) is None
        assert nb.is_bounced(("r", 0)) is False
        nb.record_result(("r", 0), 1)  # no-op, must not raise
        nb.record_failure(("r", 0), 1)  # no-op, must not raise
        nb.release_idle_reservation(("r", 0))  # no-op, must not raise
        nb.release_send(0)
        nb.close()

    def test_create_bounce_none_cfg(self):
        assert isinstance(
            btr.create_bounce(object(), None, device_id=0, page_table=None), btr.NoBounceTransport
        )

    def test_implements_bounce_transport_abc(self):
        # Both implementations must satisfy the explicit contract; instantiating NoBounceTransport
        # proves no abstractmethod is missing (guards the two impls against drifting apart).
        assert issubclass(btr.NoBounceTransport, bcore.BounceTransport)
        assert issubclass(btr.VmmBounceTransport, bcore.BounceTransport)
        assert isinstance(btr.NoBounceTransport(), bcore.BounceTransport)


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
        self.quarantined = []
        self.reserved_sizes = []

    @property
    def capacity(self):
        return self._cap

    def reserve(self, size, timeout=None):
        if size > self._cap:
            return None
        sid = self.next_id
        self.next_id += 1
        self.reserved_sizes.append(size)
        return sid, self.base

    def release(self, slot_id):
        self.released.append(slot_id)

    def quarantine(self, slot_id, grace_s):
        self.quarantined.append(slot_id)

    def reclaim_expired(self):
        return 0

    def reg_descs(self):
        return []


def _make_transport(monkeypatch, state_groups=(), capacity=1 << 30, min_bytes=1, min_blocks=1):
    monkeypatch.setattr(btr, "SlotAllocator", _FakeAlloc)
    monkeypatch.setattr(btr.VmmBounceTransport, "_new_stream", lambda self: 0)
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
        state_groups=state_groups,
        min_bytes=min_bytes,
        min_blocks=min_blocks,
    )


def _recv_req(block_counts, rid=1, slice_id=0):
    # Positional block tables: `n` held slots followed by one -1 hole, which
    # the reserve sizing must not count.
    return SimpleNamespace(
        block_ids_per_layer_groups=[
            np.concatenate([np.arange(n, dtype=np.int64), np.array([-1], dtype=np.int64)])
            if n
            else np.array([], dtype=np.int64)
            for n in block_counts
        ],
        unique_rid=rid,
        slice_id=slice_id,
        bounce_dst_base=None,
    )


@pytest.mark.skipif(not _HAVE_TRANSPORT, reason="bounce.transport import needs CUDA bindings")
class TestFanInReserve:
    def test_reserve_gives_each_writer_its_share_after_the_one_before(self, monkeypatch):
        t = _make_transport(monkeypatch)
        req = _recv_req([2])  # two pages; the trailing hole is not counted
        assert t.reserve(req, [{0: 30}, {0: 70}]) is True
        assert req.bounce_dst_base == 0x100000
        assert t._recv_alloc.reserved_sizes == [200]
        rid_slice = (req.unique_rid, req.slice_id)
        assert t.writer_base(rid_slice, 0) == 0x100000
        assert t.writer_base(rid_slice, 1) == 0x100000 + 60

    def test_reserve_sizes_shares_over_every_layer_group(self, monkeypatch):
        # Writers send different groups, a recurrent-state group's slot counting once.
        t = _make_transport(monkeypatch, state_groups={1})
        req = _recv_req([2, 1, 3])
        assert t.reserve(req, [{0: 100, 1: 50}, {0: 100, 2: 10}]) is True
        assert t._recv_alloc.reserved_sizes == [(200 + 50) + (200 + 30)]
        assert t.writer_base((req.unique_rid, req.slice_id), 1) == 0x100000 + 250

    def test_a_writer_with_nothing_to_send_writes_in_place(self, monkeypatch):
        t = _make_transport(monkeypatch)
        req = _recv_req([2])
        assert t.reserve(req, [{0: 100}, {}, {0: 100}]) is True
        rid_slice = (req.unique_rid, req.slice_id)
        assert [t.writer_base(rid_slice, i) for i in range(3)] == [0x100000, None, 0x100000 + 200]

    def test_reserve_single_writer_ok(self, monkeypatch):
        t = _make_transport(monkeypatch)
        req = _recv_req([1])
        assert t.reserve(req, [{0: 3}]) is True
        assert t._recv_alloc.reserved_sizes == [3]

    def test_reserve_of_nothing_falls_back(self, monkeypatch):
        t = _make_transport(monkeypatch)
        req = _recv_req([2])
        assert t.reserve(req, [{}, {1: 100}]) is False
        assert req.bounce_dst_base is None

    def test_reserve_below_min_bytes_falls_back(self, monkeypatch):
        # Recurrent-state payloads gate on BYTES: 4 blocks x 100B + 100B state = 500B < 1000B
        # falls back, regardless of how many blocks that is.
        t = _make_transport(monkeypatch, state_groups={1}, min_bytes=1000)
        req = _recv_req([4, 1])
        assert t.reserve(req, [{0: 100, 1: 100}]) is False
        assert req.bounce_dst_base is None

    def test_reserve_at_min_bytes_bounces(self, monkeypatch):
        # exactly at the threshold (9 x 100B + 100B state = 1000B >= 1000B) the transfer bounces
        t = _make_transport(monkeypatch, state_groups={1}, min_bytes=1000)
        req = _recv_req([9, 1])
        assert t.reserve(req, [{0: 100, 1: 100}]) is True
        assert req.bounce_dst_base == 0x100000

    def test_reserve_recurrent_payload_ignores_block_gate(self, monkeypatch):
        # The K3 regression shape: FEW but LARGE blocks plus recurrent state must clear the byte
        # gate even though a block-count gate of 96 fails (67 blocks x 6.5 MiB = 433 MiB).
        t = _make_transport(monkeypatch, state_groups={1}, min_bytes=2 * _MIB, min_blocks=96)
        assert t.reserve(_recv_req([67, 1]), [{0: int(6.5 * _MIB), 1: 1024}]) is True

    def test_reserve_plain_kv_uses_block_gate(self, monkeypatch):
        # Plain-KV payloads (no recurrent state) keep the original block-count gate, and the byte
        # gate does not apply to them (96 x 100B = 9600B passes despite min_bytes far above it).
        t = _make_transport(monkeypatch, min_blocks=96, min_bytes=1 << 20)
        assert t.reserve(_recv_req([4]), [{0: 100}]) is False  # 4 < 96 blocks
        assert t.reserve(_recv_req([96]), [{0: 100}]) is True

    def test_reserve_oversize_falls_back(self, monkeypatch):
        t = _make_transport(monkeypatch, capacity=500)
        assert t.reserve(_recv_req([2]), [{0: 1000}]) is False  # total 2000 > cap 500

    def test_reserve_does_not_wait_for_recv_space(self, monkeypatch):
        # The receiver reserves on the executor thread, so a full recv region must fall back to
        # the per-fragment path at once instead of waiting for space to free up.
        t = _make_transport(monkeypatch)
        timeouts = []

        def full_region(size, timeout=None):
            timeouts.append(timeout)
            return None

        monkeypatch.setattr(t._recv_alloc, "reserve", full_region)
        req = _recv_req([2])
        assert t.reserve(req, [{0: 100}]) is False
        assert timeouts == [0.0]
        assert req.bounce_dst_base is None

    def test_fanin_scatters_ordered_by_src_base(self, monkeypatch):
        t = _make_transport(monkeypatch)
        req = _recv_req([2])
        assert t.reserve(req, [{0: 50}, {0: 50}]) is True
        rid_slice = (req.unique_rid, req.slice_id)
        # writer for the HIGHER src_base reports first; scatter must reorder by src_base.
        t.record_result(
            rid_slice,
            7,
            np.array([20], dtype=np.int64),
            np.array([8], dtype=np.int64),
            src_base=200,
        )
        assert t._scatter_q.empty()  # only 1 of 2 writers terminal -> no scatter
        assert not t._recv_alloc.released  # region NOT freed while a writer is still pending
        t.record_result(
            rid_slice,
            3,
            np.array([10], dtype=np.int64),
            np.array([8], dtype=np.int64),
            src_base=100,
        )
        ctx, descs = t._scatter_q.get_nowait()
        # each tail carries its OWN src_base; sorted (100 before 200) so the scatter is deterministic.
        assert [t[0] for t in descs] == [100, 200]  # per-writer src_base preserved
        assert [list(t[1]) for t in descs] == [[10], [20]]  # dst_ptrs
        assert [list(t[2]) for t in descs] == [[8], [8]]  # sizes

    def test_fanin_fallback_writer_leaves_survivor_at_its_own_base(self, monkeypatch):
        # Regression: if one fan-in writer falls back to in-place (SUCCESS, empty tail) while a
        # sibling bounces, the survivor must be scattered from ITS OWN src_base, not packed to 0.
        t = _make_transport(monkeypatch)
        req = _recv_req([2])
        assert t.reserve(req, [{0: 50}, {0: 50}]) is True
        rid_slice = (req.unique_rid, req.slice_id)
        t.record_result(
            rid_slice, 7, None, None
        )  # writer 0 fell back to in-place: SUCCESS, no tail
        assert t._scatter_q.empty()  # only 1 of 2 writers terminal
        t.record_result(
            rid_slice,
            3,
            np.array([10], dtype=np.int64),
            np.array([8], dtype=np.int64),
            src_base=100,
        )  # writer 1 bounced to base+100
        ctx, descs = t._scatter_q.get_nowait()
        assert [t[0] for t in descs] == [100]  # only the survivor, read from base+100 (NOT 0)
        assert [list(t[1]) for t in descs] == [[10]]

    def test_fanin_failed_then_success_releases_only_after_both(self, monkeypatch):
        # A FAILED writer must not free the shared region until every writer is terminal.
        t = _make_transport(monkeypatch)
        req = _recv_req([2])
        assert t.reserve(req, [{0: 50}, {0: 50}]) is True
        rid_slice = (req.unique_rid, req.slice_id)
        t.record_failure(rid_slice, 7)  # first writer fails
        assert not t._recv_alloc.released  # region held while a sibling may still be in flight
        assert t.is_bounced(rid_slice) is True
        t.record_result(
            rid_slice, 3, np.array([10], dtype=np.int64), np.array([8], dtype=np.int64), src_base=0
        )
        # all terminal, >=1 FAILED -> no scatter, release (both drained), region freed.
        assert t._scatter_q.empty()
        assert t._recv_alloc.released
        assert not t._recv_alloc.quarantined  # FAILED has drained -> release, NOT quarantine
        assert t.is_bounced(rid_slice) is False

    def test_on_done_fires_after_scatter_lands(self, monkeypatch):
        # The completion cb rides the context and fires only when the worker records the scatter as
        # done -> the gen never observes completion before the KV is scattered into place.
        t = _make_transport(monkeypatch)
        req = _recv_req([2])
        assert t.reserve(req, [{0: 100}]) is True
        rid_slice = (req.unique_rid, req.slice_id)
        calls = []
        t.record_result(
            rid_slice,
            3,
            np.array([10], dtype=np.int64),
            np.array([8], dtype=np.int64),
            src_base=0,
            on_done=lambda ok: calls.append(ok),
        )
        ctx, descs = t._scatter_q.get_nowait()
        assert calls == []  # deferred, not fired inline
        t._apply(
            ctx.rid_slice, lambda c: c.finish_scatter(True)
        )  # worker records scatter done -> settle
        assert calls == [True]
        assert t._recv_alloc.released

    def test_empty_acc_fires_on_done_inline_and_releases(self, monkeypatch):
        # Bounced SUCCESS that carried no scatter tail: nothing to copy, but the task must still
        # complete -> on_done(True) inline + slot released, nothing queued.
        t = _make_transport(monkeypatch)
        req = _recv_req([2])
        assert t.reserve(req, [{0: 100}]) is True
        rid_slice = (req.unique_rid, req.slice_id)
        calls = []
        t.record_result(rid_slice, 3, None, None, on_done=lambda ok: calls.append(ok))
        assert calls == [True]
        assert t._scatter_q.empty()
        assert t._recv_alloc.released  # slot freed

    def test_missing_key_is_dropped(self, monkeypatch):
        # A late/duplicate result for an already-settled (popped) rid_slice is dropped: the context's own
        # settle already fired completion, so re-firing here would double-report.
        t = _make_transport(monkeypatch)
        calls = []
        t.record_result(
            ("missing", 0),
            3,
            np.array([10], dtype=np.int64),
            np.array([8], dtype=np.int64),
            src_base=0,
            on_done=lambda ok: calls.append(ok),
        )
        assert calls == []  # no-op, no callback

    def test_duplicate_writer_is_ignored(self, monkeypatch):
        # A duplicate SUCCESS from the same peer_rank must not double-count toward all-terminal.
        t = _make_transport(monkeypatch)
        req = _recv_req([2])
        assert t.reserve(req, [{0: 50}, {0: 50}]) is True
        rid_slice = (req.unique_rid, req.slice_id)
        arr = (np.array([10], dtype=np.int64), np.array([8], dtype=np.int64))
        t.record_result(rid_slice, 7, *arr, src_base=0)
        t.record_result(rid_slice, 7, *arr, src_base=0)  # duplicate of the SAME writer
        assert t._scatter_q.empty()  # still only 1 distinct writer -> not all terminal
        assert not t._recv_alloc.released

    def test_scatter_write_result_non_bounce_fires_on_done(self):
        # Non-bounced path completes inline (the in-place WRITE already landed the KV).
        calls = []
        btr.scatter_write_result(
            btr.NoBounceTransport(), ("r", 0), 1, None, None, on_done=lambda ok: calls.append(ok)
        )
        assert calls == [True]

    def test_release_idle_reservation_frees_slot_and_is_idempotent(self, monkeypatch):
        # INIT-cancel immediate release (nothing published) must free the recv slot; idempotent.
        t = _make_transport(monkeypatch)
        req = _recv_req([2])
        assert t.reserve(req, [{0: 100}]) is True
        rid_slice = (req.unique_rid, req.slice_id)
        assert t.is_bounced(rid_slice) is True
        t.release_idle_reservation(rid_slice)
        assert t.is_bounced(rid_slice) is False
        assert t._recv_alloc.released  # slot freed
        t.release_idle_reservation(rid_slice)  # already gone -> no-op, must not raise

    def test_orphan_reservation_quarantines_and_is_idempotent(self, monkeypatch):
        # Giving up on an in-flight reservation must quarantine the region (a write may still land),
        # not release or leak it; a second give-up is a no-op.
        t = _make_transport(monkeypatch)
        req = _recv_req([2])
        assert t.reserve(req, [{0: 100}]) is True
        rid_slice = (req.unique_rid, req.slice_id)
        t.orphan_reservation(rid_slice)
        assert t._recv_alloc.quarantined == [0]  # quarantined, not released
        assert t._recv_alloc.released == []
        assert t.is_bounced(rid_slice) is False  # settled and removed from the live map
        t.orphan_reservation(rid_slice)  # already gone -> no-op, must not raise
        assert t._recv_alloc.quarantined == [0]


# --------------------------------------------------------------------------- #
# Hybrid layouts (Kimi K3: KDA mamba + MLA attention) — the writer's share of KV and
# recurrent state, and the sizing and gates it drives
# --------------------------------------------------------------------------- #

# Kimi K3 geometry: 24 MLA layers in one attention layer group; 69 KDA layers
# whose per-layer recurrent state is a bf16 short-conv slot + fp32 delta slot,
# replicated per rank.
_K3_MLA_BLOCK_BYTES = 32 * 576 * 2 * 24  # tpb x (kv_lora_rank+rope) x bf16 x 24 layers = 884,736
_K3_KDA_LAYERS = 69
_K3_CONV_SLOT_BYTES = 221_184  # [3*H*hd, W-1] bf16 per layer
_K3_SSM_SLOT_BYTES = 6_291_456  # [H, hd, hd] fp32 per layer (95.5% of the state)
_K3_KDA_PAYLOAD_BYTES = (
    449_372_160  # 69 x (conv + delta) per request per rank, from the geometry above
)


_K3_MLA_LAYER_BYTES = _K3_MLA_BLOCK_BYTES // 24


def _k3_page_table(kda: bool = True) -> ManagerLayout:
    """A K3-shaped page table as the manager builds it.

    The attention layer group comes first and the KDA layer group after it, recurrent state
    replicated per rank; without ``kda``, only the attention group.
    """
    attention = list(range(24))
    kda_layers = list(range(24, 24 + _K3_KDA_LAYERS)) if kda else []
    buffers = [
        BufferDesc(layer, "key", 0, 0, i * _K3_MLA_LAYER_BYTES, _K3_MLA_LAYER_BYTES)
        for i, layer in enumerate(attention)
    ]
    for i, layer in enumerate(kda_layers):
        conv = _K3_CONV_SLOT_BYTES
        ssm = _K3_SSM_SLOT_BYTES
        buffers.append(
            BufferDesc(layer, "conv_state", 1, 0, i * conv, conv, sectioned([conv // 3] * 3, 1))
        )
        buffers.append(
            BufferDesc(layer, "ssm_state", 1, 1, i * ssm, ssm, head_major(ssm, 4, 4, ("tp", "cp")))
        )
    groups = [LayerGroupDesc("attention", None, 0, 0, tuple(attention))]
    widths = {0: (_K3_MLA_BLOCK_BYTES,)}
    if kda:
        groups.append(LayerGroupDesc("state", None, 0, 1, tuple(kda_layers)))
        widths[1] = (_K3_KDA_LAYERS * _K3_CONV_SLOT_BYTES, _K3_KDA_LAYERS * _K3_SSM_SLOT_BYTES)
    layout = PageLayout(32, widths, tuple(groups), tuple(buffers))
    return manager_layout(layout, {0: [0x100000], 1: [0x200000, 0x300000]}, kv_heads=1)


@pytest.mark.skipif(not _HAVE_TRANSPORT, reason="bounce.transport import needs CUDA bindings")
class TestHybridK3Bounce:
    def test_state_groups_name_the_recurrent_layer_groups(self):
        assert btr.state_groups(_k3_page_table()) == {1}
        assert btr.state_groups(_k3_page_table(kda=False)) == frozenset()

    def test_reserve_engages_on_k3_mixed_layout(self, monkeypatch):
        # Regression pin: a K3 recv request carries the MLA blocks and the KDA slot, and must
        # engage bounce, sized for MLA KV plus KDA state.
        page_table = _k3_page_table()
        t = _make_transport(
            monkeypatch, state_groups=btr.state_groups(page_table), min_bytes=2 * _MIB
        )
        req = _recv_req([67, 1])  # 67 MLA blocks (~2144 tokens) + the KDA slot
        share = sent_bytes(page_table.page_layout, page_table.page_layout)
        assert t.reserve(req, [share]) is True
        assert req.bounce_dst_base == 0x100000
        # the region covers the MLA KV blocks plus the appended KDA state
        assert t._recv_alloc.reserved_sizes == [67 * _K3_MLA_BLOCK_BYTES + _K3_KDA_PAYLOAD_BYTES]

    def test_reserve_counts_state_toward_min_bytes(self, monkeypatch):
        t = _make_transport(monkeypatch, state_groups={1}, min_bytes=1000)
        assert t.reserve(_recv_req([4, 1]), [{0: 100, 1: 599}]) is False  # 999 < 1000
        assert t.reserve(_recv_req([4, 1]), [{0: 100, 1: 600}]) is True  # exactly 1000

    def test_reserve_counts_state_toward_capacity(self, monkeypatch):
        # The state is part of the reservation, or the sender's coalesced write (KV + recurrent
        # state) would overrun the region into the neighboring slot.
        t = _make_transport(monkeypatch, state_groups={1}, capacity=500)
        assert t.reserve(_recv_req([2, 1]), [{0: 100, 1: 400}]) is False  # 600 > 500

    def test_reserve_fans_in_writers_holding_different_state(self, monkeypatch):
        # PP stages hold different recurrent layers: each writer's region holds its own state.
        t = _make_transport(monkeypatch, state_groups={1})
        req = _recv_req([2, 1])
        assert t.reserve(req, [{0: 100, 1: 64}, {0: 100}]) is True
        assert t._recv_alloc.reserved_sizes == [264 + 200]
        assert t.writer_base((req.unique_rid, req.slice_id), 1) == 0x100000 + 264

    def test_writer_share_matches_k3_geometry(self):
        layout = _k3_page_table().page_layout
        assert sent_bytes(layout, layout) == {0: _K3_MLA_BLOCK_BYTES, 1: _K3_KDA_PAYLOAD_BYTES}

    def test_writer_share_has_no_state_without_kda_on_either_side(self):
        layout = _k3_page_table().page_layout
        attention_only = _k3_page_table(kda=False).page_layout
        assert sent_bytes(attention_only, layout) == {0: _K3_MLA_BLOCK_BYTES}
        assert sent_bytes(layout, attention_only) == {0: _K3_MLA_BLOCK_BYTES}


# --------------------------------------------------------------------------- #
# Recurrent state under decode-CP (helix): shard grid and pairing
# --------------------------------------------------------------------------- #
class TestMambaHelixCP:
    def test_state_grid_is_tensor_times_context_ranks_cp_minor(self):
        # cp == 1 keeps the plain TP grid; cp > 1 flattens tp*cp CP-minor; attention DP splits
        # the state over no ranks.
        grid = ("tp", "cp")
        assert grid_place(RankTopology(), grid) == (1, 0)
        assert grid_place(RankTopology(cp_size=32, cp_rank=5), grid) == (32, 5)
        assert grid_place(RankTopology(tp_size=2, tp_rank=1, cp_size=4, cp_rank=3), grid) == (8, 7)
        adp = RankTopology(tp_size=4, tp_rank=2, dp_size=4, dp_rank=2, attention_dp=True)
        assert grid_place(adp, grid) == (1, 0)

    @staticmethod
    def _state_layout(topology, grid):
        """This rank's share of one recurrent-state buffer of 8 heads of 64 B.

        The heads split over the ``grid`` ranks of the tp x cp state grid.
        """
        heads = 8 // grid
        size = heads * 64
        return PageLayout(
            tokens_per_page=32,
            pool_slot_bytes={0: (size,)},
            layer_groups=(LayerGroupDesc("state", None, 0, 0, (0,)),),
            buffers=(
                BufferDesc(0, "ssm_state", 0, 0, 0, size, head_major(size, heads, 8, ("tp", "cp"))),
            ),
            topology=topology,
        )

    def _runs(self, sender_tp_rank, gen_cp_size, gen_cp_rank):
        sender = self._state_layout(RankTopology(tp_size=2, tp_rank=sender_tp_rank), 2)
        receiver = self._state_layout(
            RankTopology(cp_size=gen_cp_size, cp_rank=gen_cp_rank), gen_cp_size
        )
        return [
            (int(s), int(d), int(n))
            for runs in map_buffers(sender, receiver)
            for s, d, n in zip(runs.src_offsets, runs.dst_offsets, runs.sizes)
        ]

    def test_equal_width_unpaired_sender_sends_nothing(self):
        # Regression lock for the fan-in collapse: with equal shard widths the copy is a whole
        # slot, so an unpaired sender must send nothing instead of overwriting the receiver slot
        # with the wrong heads.
        for sender_rank in range(2):
            for gen_cp_rank in range(2):
                runs = self._runs(sender_rank, 2, gen_cp_rank)
                if sender_rank == gen_cp_rank:
                    assert runs == [(0, 0, 256)]
                else:
                    assert runs == []

    def test_narrow_to_wide_pairing_follows_covering_tree(self):
        # Sender grid 2, receiver grid 4: receiver g takes half of sender g // 2's heads.
        for sender_rank in range(2):
            for gen_cp_rank in range(4):
                runs = self._runs(sender_rank, 4, gen_cp_rank)
                if gen_cp_rank // 2 == sender_rank:
                    assert runs == [(gen_cp_rank % 2 * 128, 0, 128)], (sender_rank, gen_cp_rank)
                else:
                    assert runs == [], (sender_rank, gen_cp_rank)


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

    def test_zero_timeout_does_not_wait_when_full(self, monkeypatch):
        a = self._alloc(monkeypatch, cap=512)  # single slot
        a.reserve(512)
        monkeypatch.setattr(a._cv, "wait", lambda *args, **kwargs: pytest.fail("reserve waited"))
        assert a.reserve(512, timeout=0.0) is None


# --------------------------------------------------------------------------- #
# core.TransferContext — the pure drain-before-release state machine.
# No CUDA / NIXL / allocator, so these run on any CPU (no skipif).
# --------------------------------------------------------------------------- #
class TestLifecycle:
    def _ctx(self, num_writers, writer_bytes=100, base_addr=0x1000):
        return bcore.TransferContext(
            rid_slice=(1, 0),
            slot_id=0,
            base_addr=base_addr,
            writer_bytes=(writer_bytes,) * num_writers,
        )

    def _dst(self, v=10):
        return dict(dst_ptrs=np.array([v], dtype=np.int64), sizes=np.array([8], dtype=np.int64))

    def test_writer_base_layout(self):
        c = self._ctx(3, writer_bytes=0x64, base_addr=0x1000)
        assert [c.writer_base(i) for i in range(3)] == [0x1000, 0x1064, 0x10C8]
        uneven = bcore.TransferContext((1, 0), 0, 0x1000, writer_bytes=(0x10, 0, 0x30))
        assert [uneven.writer_base(i) for i in range(3)] == [0x1000, None, 0x1010]

    def test_single_writer_success_scatters_then_releases(self):
        c = self._ctx(1)
        assert not c.ready_to_scatter() and not c.ready_to_settle()
        c.record_writer_result(3, succeeded=True, src_base=0, **self._dst())
        assert c.ready_to_scatter()
        c.begin_scatter()
        assert not c.ready_to_settle()  # scatter not landed yet
        c.finish_scatter(True)
        assert c.ready_to_settle()
        ret = c.settle()
        assert ret.disposition is bcore.Disposition.RELEASE and ret.success is True
        assert c.state is bcore.TransferState.COMPLETED
        assert c.settle() is None  # idempotent one-shot

    def test_fanin_holds_until_all_terminal(self):
        c = self._ctx(2)
        c.record_writer_result(7, succeeded=True, src_base=0, **self._dst())
        assert not c.ready_to_scatter()  # 1/2 writers
        assert not c.ready_to_settle()  # drain-before-release
        c.record_writer_result(3, succeeded=True, src_base=100, **self._dst())
        assert c.ready_to_scatter()  # all success -> scatter

    def test_fanin_failed_then_success_releases(self):
        c = self._ctx(2)
        c.record_writer_result(7, succeeded=False)
        assert not c.ready_to_settle()  # a sibling is still pending -> hold
        c.record_writer_result(3, succeeded=True, src_base=0, **self._dst())
        assert not c.ready_to_scatter()  # >=1 FAILED -> skip scatter
        assert c.ready_to_settle()
        ret = c.settle()
        assert ret.disposition is bcore.Disposition.RELEASE  # all drained -> free, not quarantine
        assert ret.success is False
        assert c.state is bcore.TransferState.FAILED

    def test_partial_publication_waits_for_published_writers_without_scatter(self):
        c = self._ctx(2)
        c.record_writer_result(7, succeeded=True, src_base=0, **self._dst())

        c.abort_publication(published_writers={7})

        assert not c.ready_to_scatter()
        assert c.ready_to_settle()
        ret = c.settle()
        assert ret.disposition is bcore.Disposition.RELEASE
        assert ret.success is False
        assert c.state is bcore.TransferState.FAILED

    def test_failed_publication_with_no_published_writer_settles_immediately(self):
        c = self._ctx(2)

        c.abort_publication(published_writers=set())

        assert c.ready_to_settle()
        ret = c.settle()
        assert ret.disposition is bcore.Disposition.RELEASE
        assert ret.success is False

    def test_orphan_quarantines(self):
        c = self._ctx(2)
        c.record_writer_result(7, succeeded=True, src_base=0, **self._dst())
        c.mark_orphaned()  # the other writer is in-doubt
        assert c.ready_to_settle()
        ret = c.settle()
        assert ret.disposition is bcore.Disposition.QUARANTINE and ret.success is False
        assert c.state is bcore.TransferState.QUARANTINED

    def test_empty_tail_success_releases_without_scatter(self):
        c = self._ctx(1)
        c.record_writer_result(3, succeeded=True)  # no dst tail
        assert not c.ready_to_scatter()
        assert c.ready_to_settle()
        assert c.settle().disposition is bcore.Disposition.RELEASE

    def test_writers_locked_after_scatter_drops_late_writer(self):
        c = self._ctx(1)
        c.record_writer_result(3, succeeded=True, src_base=0, **self._dst())
        c.begin_scatter()  # SCATTERING -> frozen
        c.record_writer_result(9, succeeded=False)  # a late / reordered report
        assert 9 not in c._writer_ok  # dropped, cannot re-arm the state

    def test_duplicate_writer_dedup(self):
        c = self._ctx(2)
        c.record_writer_result(7, succeeded=True, src_base=0, **self._dst())
        c.record_writer_result(7, succeeded=False)  # same rank again -> ignored
        assert c._writer_ok[7] is True
        assert not c.ready_to_settle()  # still only 1 distinct writer of 2

    def test_scatter_failure_releases_as_failed(self):
        c = self._ctx(1)
        c.record_writer_result(3, succeeded=True, src_base=0, **self._dst())
        c.begin_scatter()
        c.finish_scatter(False)  # scatter kernel failed
        ret = c.settle()
        assert ret.disposition is bcore.Disposition.RELEASE and ret.success is False

    def test_orphan_after_scatter_is_ignored(self):
        # once SCATTERING, all writers already reported SUCCESS -> nothing is in doubt, so a late
        # orphan (e.g. a racing cancel) must NOT downgrade a clean transfer to quarantine.
        c = self._ctx(1)
        c.record_writer_result(3, succeeded=True, src_base=0, **self._dst())
        c.begin_scatter()
        c.mark_orphaned()  # no-op after SCATTERING
        c.finish_scatter(True)
        assert c.settle().disposition is bcore.Disposition.RELEASE


# --------------------------------------------------------------------------- #
# SlotAllocator quarantine — orphaned regions are held out of reuse for a grace
# period, then reclaimed off a timer (never handed out while a stray write could land)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAVE_TRANSPORT, reason="bounce.buffer import needs CUDA bindings")
class TestQuarantine:
    def _alloc(self, monkeypatch, cap):
        monkeypatch.setattr(
            bbuf,
            "Buffer",
            lambda *a, **k: SimpleNamespace(
                size=cap, base_ptr=0x1000, reg_descs=lambda: [], close=lambda: None
            ),
        )
        return btr.SlotAllocator(cap, 512)

    def test_quarantined_region_is_not_reused(self, monkeypatch):
        a = self._alloc(monkeypatch, cap=1024)  # two 512-byte slots
        a.reserve(512)  # [0, 512) stays live
        sb, _ = a.reserve(512)  # [512, 1024)
        a.quarantine(sb, grace_s=float("inf"))  # hold the tail out of reuse
        assert a.quarantined_bytes == 512
        # live [0,512) + quarantined [512,1024) => arena full => next reserve can't fit.
        assert a.reserve(512, timeout=0.05) is None

    def test_reclaim_returns_expired_to_free(self, monkeypatch):
        a = self._alloc(monkeypatch, cap=512)  # single slot
        s, _ = a.reserve(512)
        a.quarantine(s, grace_s=0.0)  # deadline already in the past
        assert a.reserve(512, timeout=0.05) is None  # still held until reclaimed
        assert a.reclaim_expired() == 1
        assert a.quarantined_bytes == 0
        assert a.reserve(512, timeout=0.5) is not None  # now reusable

    def test_inf_grace_never_reclaims(self, monkeypatch):
        a = self._alloc(monkeypatch, cap=512)
        s, _ = a.reserve(512)
        a.quarantine(s, grace_s=float("inf"))  # close-only reclaim
        assert a.reclaim_expired() == 0
        assert a.quarantined_bytes == 512
