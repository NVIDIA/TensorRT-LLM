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

    def test_min_blocks_defaults_and_overrides(self, monkeypatch):
        monkeypatch.delenv("TRTLLM_KV_CACHE_BOUNCE_MIN_BLOCKS", raising=False)
        assert bcfg.config_from_size(2048).min_blocks == 96  # keeps the Config default
        assert bcfg.config_from_size(2048, 1).min_blocks == 1  # explicit arg still overrides
        assert bcfg.config_from_size(2048, 250).min_blocks == 250
        # The gate is lowered for tests via the env override (no user-facing config field).
        monkeypatch.setenv("TRTLLM_KV_CACHE_BOUNCE_MIN_BLOCKS", "1")
        assert bcfg.config_from_size(2048).min_blocks == 1  # env override
        assert bcfg.config_from_size(2048, 250).min_blocks == 250  # explicit arg beats env

    def test_min_blocks_env_is_parsed_defensively(self, monkeypatch):
        # A bad value must not crash setup (falls back to the default); a non-positive value clamps to 1.
        for bad in ("", "auto", "1.5"):
            monkeypatch.setenv("TRTLLM_KV_CACHE_BOUNCE_MIN_BLOCKS", bad)
            assert bcfg.config_from_size(2048).min_blocks == 96
        for nonpos in ("0", "-5"):
            monkeypatch.setenv("TRTLLM_KV_CACHE_BOUNCE_MIN_BLOCKS", nonpos)
            assert bcfg.config_from_size(2048).min_blocks == 1


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

    def test_v2_roundtrip_uses_explicit_tail_index(self):
        wm = self._wm([100], [16], 0xABCD)
        msg = [b"KV_AGENT_RESULT", b"prefix", b"sender-endpoint"] + btr.encode_result_tail(wm)
        dst, sizes, src_base = btr.decode_result_tail(msg, tail_index=3)
        assert np.array_equal(dst, wm.dst_ptrs)
        assert np.array_equal(sizes, wm.sizes)
        assert src_base == 0xABCD

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
# fan-in safety gate — equal total//num_writers split only for uniform TP-by-head
# --------------------------------------------------------------------------- #
def test_fanin_bounce_safe_gate():
    """Restrict multi-writer equal-split bounce to uniform TP-by-head.

    PP (overlap_pp_size>1 -> unequal per-writer sizes) and duplicate_head_factor>1
    (MLA / duplicate TP heads -> some ranks don't send KV yet count in
    expected_transfers) must fall back to the per-fragment path.
    """
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    from tensorrt_llm._torch.disaggregation.resource.page import MapperKind

    safe = tfr.Receiver._fanin_bounce_safe

    def ov(dup, pp, ranks=(0,)):
        return SimpleNamespace(duplicate_head_factor=dup, overlap_pp_size=pp, ranks=list(ranks))

    def ri(lpp, page_table=None):
        return SimpleNamespace(layer_num_per_pp=lpp, page_table=page_table)

    def pt(mapper_kind):
        view = SimpleNamespace(mapper_kind=mapper_kind)
        return SimpleNamespace(layer_groups=[SimpleNamespace(pool_views=[view])])

    # single PP stage (overlap_pp_size <= 1): only duplicate_head_factor matters
    assert safe(ov(1, 1), ri([24])) is True
    assert safe(ov(1, 0), ri([24])) is True
    assert safe(ov(2, 1), ri([24])) is False  # duplicate heads / MLA -> some don't send
    # EVEN PP fan-in (equal layers per overlapping stage) -> allowed
    assert safe(ov(1, 4), ri([20, 20, 20, 20])) is True
    # UNEVEN PP fan-in -> per-writer sizes differ -> fall back
    assert safe(ov(1, 4), ri([20, 20, 20, 19])) is False
    # incomplete per-stage info (single element for a multi-stage fan-in) -> conservative fall back
    assert safe(ov(1, 4), ri([20])) is False
    # duplicate heads blocks even an otherwise-even PP split
    assert safe(ov(2, 4), ri([20, 20, 20, 20])) is False
    # replicated views (one elected sender per destination) make multi-writer
    # contributions unequal -> fall back; single-writer overlap stays safe,
    # and sharded-only view schemes are unaffected
    assert safe(ov(1, 1, ranks=(0, 1)), ri([24], pt(MapperKind.REPLICATED))) is False
    assert safe(ov(1, 1, ranks=(0,)), ri([24], pt(MapperKind.REPLICATED))) is True
    assert safe(ov(1, 1, ranks=(0, 1)), ri([24], pt(MapperKind.NHD))) is True


def test_receiver_derives_canonical_rank_bound_destination_plan():
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    if not hasattr(tfr.Receiver, "_build_bounce_destination_plan"):
        pytest.skip("minimal local transfer stub does not load Receiver implementation")
    receiver = object.__new__(tfr.Receiver)
    local_region = SimpleNamespace(memory=SimpleNamespace(ptrs=np.array([1, 2])))
    peer_region = SimpleNamespace(memory=SimpleNamespace(ptrs=np.array([3, 4])))
    mapped = SimpleNamespace(
        src=SimpleNamespace(
            memory=SimpleNamespace(
                ptrs=np.array([0x2050, 0x2000], dtype=np.int64),
                bytes_per_region=0x50,
            )
        )
    )
    mapper = SimpleNamespace(map=lambda _local, _peer: mapped)
    receiver._registrar = SimpleNamespace(
        self_extractor=SimpleNamespace(extract=lambda *_args, **_kwargs: local_region),
        peer_extractor=lambda *_args: SimpleNamespace(
            extract=lambda *_extract_args, **_extract_kwargs: peer_region
        ),
        get_pool_mapping=lambda _peer: {(0, 0): (0, 0)},
        get_kv_map=lambda *_args: mapper,
    )
    receiver_req = SimpleNamespace(block_ids_per_layer_groups=[np.array([5, 6], dtype=np.int64)])
    peer_ri = SimpleNamespace(instance_name="ctx", instance_rank=7)

    dst_ptrs, sizes = receiver._build_bounce_destination_plan(receiver_req, peer_ri)

    assert dst_ptrs.tolist() == [0x2000, 0x2050]
    assert sizes.tolist() == [0x50, 0x50]


# --------------------------------------------------------------------------- #
# NoBounceTransport — disabled no-op fallback
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAVE_TRANSPORT, reason="bounce.transport import needs CUDA bindings")
class TestNoBounce:
    def test_noop_behaviour(self):
        nb = btr.NoBounceTransport()
        assert nb.enabled is False
        assert nb.reserve(SimpleNamespace()) is False
        assert nb.build_request(SimpleNamespace()) is None
        assert nb.writer_base(("r", 0), 1) is None
        assert nb.is_bounced(("r", 0)) is False
        nb.set_completion_callback(("r", 0), lambda _ok: None)
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

    def quarantine(self, slot_id, grace_s):
        self.quarantined.append(slot_id)

    def reclaim_expired(self):
        return 0

    def reg_descs(self):
        return []


def _make_transport(
    monkeypatch,
    block_bytes_per_group,
    capacity=1 << 30,
    min_blocks=1,
    destination_pool_layouts=None,
    valid_destination_ranges=None,
):
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
        block_bytes_per_group=block_bytes_per_group,
        destination_pool_layouts=destination_pool_layouts,
        valid_destination_ranges=(
            [(0, 1 << 62)] if valid_destination_ranges is None else valid_destination_ranges
        ),
        min_blocks=min_blocks,
    )


def _recv_req(block_counts, rid=1, slice_id=0):
    return SimpleNamespace(
        block_ids_per_layer_groups=[SimpleNamespace(size=n) for n in block_counts],
        unique_rid=rid,
        slice_id=slice_id,
        bounce_dst_base=None,
    )


def _recv_req_with_ids(block_ids_per_group, rid=1, slice_id=0):
    return SimpleNamespace(
        block_ids_per_layer_groups=[
            np.asarray(block_ids, dtype=np.int64) for block_ids in block_ids_per_group
        ],
        unique_rid=rid,
        slice_id=slice_id,
        bounce_dst_base=None,
        mamba_state_index=None,
    )


@pytest.mark.skipif(not _HAVE_TRANSPORT, reason="bounce.transport import needs CUDA bindings")
class TestFanInReserve:
    def test_reserve_stamps_base_and_per_writer(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])  # total = 2 * 100 = 200
        assert t.reserve(req, num_writers=2) is True
        assert req.bounce_dst_base == 0x100000
        # writer i lands at base + i * (total // num_writers); per_writer = 100.
        assert t.writer_base((req.unique_rid, req.slice_id), 0) == 0x100000
        assert t.writer_base((req.unique_rid, req.slice_id), 1) == 0x100000 + 100

    def test_reserve_uneven_fanin_falls_back(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[3])
        req = _recv_req([1])  # total = 3, not divisible by 2
        assert t.reserve(req, num_writers=2) is False
        assert req.bounce_dst_base is None

    def test_reserve_heterogeneous_fanin_falls_back(self, monkeypatch):
        # Two groups with different per-block sizes: the equal split would overrun a sub-region, so
        # fall back (even though the total is divisible).
        t = _make_transport(monkeypatch, block_bytes_per_group=[100, 200])
        req = _recv_req([2, 2])  # total = 2*100 + 2*200 = 600
        assert t.reserve(req, num_writers=2) is False
        assert req.bounce_dst_base is None

    def test_reserve_uniform_multigroup_fanin_ok(self, monkeypatch):
        # Uniform slot bytes across present groups -> even byte split -> bounce allowed.
        t = _make_transport(monkeypatch, block_bytes_per_group=[100, 100])
        assert t.reserve(_recv_req([2, 2]), num_writers=2) is True

    def test_reserve_heterogeneous_single_writer_ok(self, monkeypatch):
        # num_writers==1 has no split, so heterogeneous slot bytes are fine.
        t = _make_transport(monkeypatch, block_bytes_per_group=[100, 200])
        assert t.reserve(_recv_req([2, 2]), num_writers=1) is True

    def test_reserve_single_writer_ok(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[3])
        req = _recv_req([1])  # total = 3, num_writers=1 -> no even-split requirement
        assert t.reserve(req, num_writers=1) is True

    def test_reserve_too_small_falls_back(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100], min_blocks=96)
        assert t.reserve(_recv_req([4]), num_writers=1) is False  # 4 < 96 blocks

    def test_reserve_unknown_slot_size_falls_back(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])  # only 1 group known
        assert t.reserve(_recv_req([2, 2]), num_writers=1) is False  # 2nd group unknown

    def test_reserve_oversize_falls_back(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[1000], capacity=500)
        assert t.reserve(_recv_req([2]), num_writers=1) is False  # total 2000 > cap 500

    def test_fanin_scatters_ordered_by_src_base(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])
        assert t.reserve(req, num_writers=2) is True
        rid_slice = (req.unique_rid, req.slice_id)
        low_base = t.bind_writer(rid_slice, 3, 0)
        high_base = t.bind_writer(rid_slice, 7, 1)
        # writer for the HIGHER src_base reports first; scatter must reorder by src_base.
        t.record_result(
            rid_slice,
            7,
            np.array([20], dtype=np.int64),
            np.array([100], dtype=np.int64),
            src_base=high_base,
        )
        assert t._scatter_q.empty()  # only 1 of 2 writers terminal -> no scatter
        assert not t._recv_alloc.released  # region NOT freed while a writer is still pending
        t.record_result(
            rid_slice,
            3,
            np.array([10], dtype=np.int64),
            np.array([100], dtype=np.int64),
            src_base=low_base,
        )
        ctx, descs = t._scatter_q.get_nowait()
        # Each tail carries its own bound source; sorting makes scatter deterministic.
        assert [item[0] for item in descs] == [low_base, high_base]
        assert [list(t[1]) for t in descs] == [[10], [20]]  # dst_ptrs
        assert [list(t[2]) for t in descs] == [[100], [100]]  # sizes

    def test_incomplete_bounced_result_is_rejected_without_releasing(self, monkeypatch):
        # Once the receiver advertises a bounce address, a tail-less SUCCESS is malformed rather
        # than an in-place fallback. It must not consume writer credit or free the shared region.
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])
        assert t.reserve(req, num_writers=2) is True
        rid_slice = (req.unique_rid, req.slice_id)
        t.bind_writer(rid_slice, 7, 0)
        t.bind_writer(rid_slice, 3, 1)
        with pytest.raises(RuntimeError, match="incomplete bounced-result scatter tail"):
            t.record_result(rid_slice, 7, None, None)
        assert t._scatter_q.empty()
        assert t._recv_alloc.released == []
        assert t.is_bounced(rid_slice) is True

    def test_fanin_failed_then_success_releases_only_after_both(self, monkeypatch):
        # A FAILED writer must not free the shared region until every writer is terminal.
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])
        assert t.reserve(req, num_writers=2) is True
        rid_slice = (req.unique_rid, req.slice_id)
        t.bind_writer(rid_slice, 7, 0)
        src_base = t.bind_writer(rid_slice, 3, 1)
        t.record_failure(rid_slice, 7)  # first writer fails
        assert not t._recv_alloc.released  # region held while a sibling may still be in flight
        assert t.is_bounced(rid_slice) is True
        t.record_result(
            rid_slice,
            3,
            np.array([10], dtype=np.int64),
            np.array([100], dtype=np.int64),
            src_base=src_base,
        )
        # all terminal, >=1 FAILED -> no scatter, release (both drained), region freed.
        assert t._scatter_q.empty()
        assert t._recv_alloc.released
        assert not t._recv_alloc.quarantined  # FAILED has drained -> release, NOT quarantine
        assert t.is_bounced(rid_slice) is False

    def test_on_done_fires_after_scatter_lands(self, monkeypatch):
        # The completion cb rides the context and fires only when the worker records the scatter as
        # done -> the gen never observes completion before the KV is scattered into place.
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])
        assert t.reserve(req, num_writers=1) is True
        rid_slice = (req.unique_rid, req.slice_id)
        src_base = t.bind_writer(rid_slice, 3, 0)
        calls = []
        t.record_result(
            rid_slice,
            3,
            np.array([10], dtype=np.int64),
            np.array([200], dtype=np.int64),
            src_base=src_base,
            on_done=lambda ok: calls.append(ok),
        )
        ctx, descs = t._scatter_q.get_nowait()
        assert calls == []  # deferred, not fired inline
        t._apply(
            ctx.rid_slice, lambda c: c.finish_scatter(True)
        )  # worker records scatter done -> settle
        assert calls == [True]
        assert t._recv_alloc.released

    def test_empty_tail_does_not_fire_callback_or_release(self, monkeypatch):
        # A missing scatter plan is not proof that the advertised remote write landed correctly.
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])
        assert t.reserve(req, num_writers=1) is True
        rid_slice = (req.unique_rid, req.slice_id)
        t.bind_writer(rid_slice, 3, 0)
        calls = []
        with pytest.raises(RuntimeError, match="incomplete bounced-result scatter tail"):
            t.record_result(rid_slice, 3, None, None, on_done=lambda ok: calls.append(ok))
        assert calls == []
        assert t._scatter_q.empty()
        assert t._recv_alloc.released == []
        assert t.is_bounced(rid_slice) is True

    def test_missing_key_is_rejected(self, monkeypatch):
        # Session tombstones filter legitimate delayed duplicates before they reach the bounce layer;
        # an unknown reservation here is a protocol error and cannot be silently accepted.
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        calls = []
        with pytest.raises(RuntimeError, match="unknown reservation"):
            t.record_result(
                ("missing", 0),
                3,
                np.array([10], dtype=np.int64),
                np.array([100], dtype=np.int64),
                src_base=0,
                on_done=lambda ok: calls.append(ok),
            )
        assert calls == []

    def test_duplicate_writer_is_ignored(self, monkeypatch):
        # A duplicate SUCCESS from the same peer_rank must not double-count toward all-terminal.
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])
        assert t.reserve(req, num_writers=2) is True
        rid_slice = (req.unique_rid, req.slice_id)
        src_base = t.bind_writer(rid_slice, 7, 0)
        t.bind_writer(rid_slice, 3, 1)
        arr = (np.array([10], dtype=np.int64), np.array([100], dtype=np.int64))
        t.record_result(rid_slice, 7, *arr, src_base=src_base)
        t.record_result(rid_slice, 7, *arr, src_base=src_base)  # duplicate of the SAME writer
        assert t._scatter_q.empty()  # still only 1 distinct writer -> not all terminal
        assert not t._recv_alloc.released

    @pytest.mark.parametrize(
        "peer_rank,source_offset,error",
        [
            (9, 0, "source identity mismatch"),
            (7, 1, "source identity mismatch"),
        ],
    )
    def test_unbound_or_wrong_source_writer_is_rejected(
        self, monkeypatch, peer_rank, source_offset, error
    ):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([1])
        assert t.reserve(req, num_writers=1)
        rid_slice = (req.unique_rid, req.slice_id)
        src_base = t.bind_writer(rid_slice, 7, 0)
        with pytest.raises(RuntimeError, match=error):
            t.record_result(
                rid_slice,
                peer_rank,
                np.array([10], dtype=np.int64),
                np.array([100], dtype=np.int64),
                src_base=src_base + source_offset,
            )
        assert t._recv_alloc.released == []
        assert t.is_bounced(rid_slice)

    @pytest.mark.parametrize(
        "dst_ptrs,sizes,error",
        [
            ([0x2000], [99], "describes 99 bytes"),
            ([0x2000, 0x2030], [60, 40], "overlap"),
            ([0x1FF0], [100], "outside the receiver-owned KV destination plan"),
            ([0x2000], [0], "must be positive"),
        ],
    )
    def test_untrusted_scatter_plan_is_rejected(self, monkeypatch, dst_ptrs, sizes, error):
        t = _make_transport(
            monkeypatch,
            block_bytes_per_group=[100],
            valid_destination_ranges=[(0x2000, 0x2100)],
        )
        req = _recv_req([1])
        assert t.reserve(req, num_writers=1)
        rid_slice = (req.unique_rid, req.slice_id)
        src_base = t.bind_writer(rid_slice, 7, 0)
        with pytest.raises(RuntimeError, match=error):
            t.record_result(
                rid_slice,
                7,
                np.array(dst_ptrs, dtype=np.int64),
                np.array(sizes, dtype=np.int64),
                src_base=src_base,
            )
        assert t._recv_alloc.released == []
        assert t.is_bounced(rid_slice)

    def test_scatter_plan_cannot_target_another_request_block(self, monkeypatch):
        # Both blocks live in the registered pool, but this request owns only
        # block 2. A sender-returned tail naming block 3 must fail closed.
        t = _make_transport(
            monkeypatch,
            block_bytes_per_group=[100],
            destination_pool_layouts=[[(0x2000, 100, 8)]],
            valid_destination_ranges=[(0x2000, 0x2320)],
        )
        req = _recv_req_with_ids([[2]])
        expected_ptr = 0x2000 + 2 * 100
        assert t.reserve(
            req,
            num_writers=1,
            expected_destination_plans={
                7: (
                    np.array([expected_ptr], dtype=np.int64),
                    np.array([100], dtype=np.int64),
                )
            },
        )
        rid_slice = (req.unique_rid, req.slice_id)
        src_base = t.bind_writer(rid_slice, 7, 0)

        with pytest.raises(RuntimeError, match="outside the receiver-owned KV destination plan"):
            t.record_result(
                rid_slice,
                7,
                np.array([0x2000 + 3 * 100], dtype=np.int64),
                np.array([100], dtype=np.int64),
                src_base=src_base,
            )

        assert t._recv_alloc.released == []
        assert t.is_bounced(rid_slice)

    @pytest.mark.parametrize(
        "dst_ptrs,error",
        [
            ([0x2000, 0x2000], "overlap or duplicate"),
            ([0x2032, 0x2000], "not in canonical address order"),
        ],
    )
    def test_exact_plan_rejects_duplicate_or_reordered_tail(self, monkeypatch, dst_ptrs, error):
        t = _make_transport(
            monkeypatch,
            block_bytes_per_group=[100],
            destination_pool_layouts=[[(0x2000, 100, 8)]],
        )
        req = _recv_req_with_ids([[0]])
        expected = {
            7: (
                np.array([0x2000, 0x2032], dtype=np.int64),
                np.array([50, 50], dtype=np.int64),
            )
        }
        assert t.reserve(req, expected_destination_plans=expected)
        rid_slice = (req.unique_rid, req.slice_id)
        src_base = t.bind_writer(rid_slice, 7, 0)

        with pytest.raises(RuntimeError, match=error):
            t.record_result(
                rid_slice,
                7,
                np.array(dst_ptrs, dtype=np.int64),
                np.array([50, 50], dtype=np.int64),
                src_base=src_base,
            )

        assert t._scatter_q.empty()
        assert t._recv_alloc.released == []
        assert t.is_bounced(rid_slice)

    @pytest.mark.parametrize("wrong_ptr", [0x2001, 0x2064])
    def test_exact_plan_rejects_rank_swapped_or_in_bounds_wrong_offset(
        self, monkeypatch, wrong_ptr
    ):
        # Both writer plans are valid, in-request ranges and collectively cover
        # the slot. A writer still cannot claim its sibling's range or shift its
        # own range within the slot: rank identity binds the exact sequence.
        t = _make_transport(
            monkeypatch,
            block_bytes_per_group=[200],
            destination_pool_layouts=[[(0x2000, 200, 8)]],
        )
        req = _recv_req_with_ids([[0]])
        expected = {
            7: (
                np.array([0x2000], dtype=np.int64),
                np.array([100], dtype=np.int64),
            ),
            3: (
                np.array([0x2064], dtype=np.int64),
                np.array([100], dtype=np.int64),
            ),
        }
        assert t.reserve(req, num_writers=2, expected_destination_plans=expected)
        rid_slice = (req.unique_rid, req.slice_id)
        src_base = t.bind_writer(rid_slice, 7, 0)
        t.bind_writer(rid_slice, 3, 1)

        with pytest.raises(RuntimeError, match="exact receiver-derived destination plan"):
            t.record_result(
                rid_slice,
                7,
                np.array([wrong_ptr], dtype=np.int64),
                np.array([100], dtype=np.int64),
                src_base=src_base,
            )

        assert t._scatter_q.empty()
        assert t._recv_alloc.released == []
        assert t.is_bounced(rid_slice)

    def test_exact_rank_bound_plans_accept_correct_fanin(self, monkeypatch):
        t = _make_transport(
            monkeypatch,
            block_bytes_per_group=[200],
            destination_pool_layouts=[[(0x2000, 200, 8)]],
        )
        req = _recv_req_with_ids([[0]])
        expected = {
            7: (
                np.array([0x2000], dtype=np.int64),
                np.array([100], dtype=np.int64),
            ),
            3: (
                np.array([0x2064], dtype=np.int64),
                np.array([100], dtype=np.int64),
            ),
        }
        assert t.reserve(req, num_writers=2, expected_destination_plans=expected)
        rid_slice = (req.unique_rid, req.slice_id)
        first_base = t.bind_writer(rid_slice, 7, 0)
        second_base = t.bind_writer(rid_slice, 3, 1)

        t.record_result(
            rid_slice,
            3,
            np.array([0x2064], dtype=np.int64),
            np.array([100], dtype=np.int64),
            src_base=second_base,
        )
        t.record_result(
            rid_slice,
            7,
            np.array([0x2000], dtype=np.int64),
            np.array([100], dtype=np.int64),
            src_base=first_base,
        )

        _ctx, descs = t._scatter_q.get_nowait()
        assert [src for src, _dst, _sizes in descs] == [first_base, second_base]

    def test_request_destination_plan_rejects_out_of_range_block_id(self, monkeypatch):
        t = _make_transport(
            monkeypatch,
            block_bytes_per_group=[100],
            destination_pool_layouts=[[(0x2000, 100, 8)]],
        )
        req = _recv_req_with_ids([[8]])

        assert t.reserve(req, num_writers=1) is False
        assert req.bounce_dst_base is None
        assert t._recv_alloc.released == [0]

    def test_mamba_request_does_not_advertise_an_incomplete_bounce_plan(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req_with_ids([[0]])
        req.mamba_state_index = 4

        assert t.reserve(req, num_writers=1) is False
        assert req.bounce_dst_base is None

    def test_build_request_releases_send_slot_when_request_creation_fails(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        monkeypatch.setattr(t, "_gather_blocking", lambda *args, **kwargs: None)

        def fail_make_write(*args, **kwargs):
            raise RuntimeError("descriptor creation failed")

        monkeypatch.setattr(t, "_make_write", fail_make_write)
        write_meta = SimpleNamespace(
            src_ptrs=np.array([0x1000], dtype=np.int64),
            dst_ptrs=np.array([0x2000], dtype=np.int64),
            sizes=np.array([100], dtype=np.int64),
        )
        with pytest.raises(RuntimeError, match="descriptor creation failed"):
            t.build_request(write_meta)
        assert t._send_alloc.released == [0]

    def test_build_request_canonicalizes_coupled_fragment_triplets(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        gathered = []
        monkeypatch.setattr(
            t,
            "_gather_blocking",
            lambda _addr, meta, _total: gathered.append(
                (meta.src_ptrs.copy(), meta.dst_ptrs.copy(), meta.sizes.copy())
            ),
        )
        monkeypatch.setattr(t, "_make_write", lambda *_args: "request")
        write_meta = SimpleNamespace(
            src_ptrs=np.array([0x3000, 0x1000, 0x2000], dtype=np.int64),
            dst_ptrs=np.array([0x2300, 0x2100, 0x2200], dtype=np.int64),
            sizes=np.array([30, 10, 20], dtype=np.int64),
        )

        request, _slot_id = t.build_request(write_meta)

        assert request == "request"
        assert list(write_meta.src_ptrs) == [0x1000, 0x2000, 0x3000]
        assert list(write_meta.dst_ptrs) == [0x2100, 0x2200, 0x2300]
        assert list(write_meta.sizes) == [10, 20, 30]
        assert [list(values) for values in gathered[0]] == [
            [0x1000, 0x2000, 0x3000],
            [0x2100, 0x2200, 0x2300],
            [10, 20, 30],
        ]

    def test_close_retains_memory_if_scatter_thread_does_not_exit(self):
        t = object.__new__(btr.VmmBounceTransport)
        stopped = []
        joined = []
        deregistered = []
        closed = []
        t._stop = SimpleNamespace(set=lambda: stopped.append(True))
        t._scatter_q = queue.Queue()
        t._scatter_thread = SimpleNamespace(
            is_alive=lambda: True, join=lambda timeout: joined.append(timeout)
        )
        t._reg_descs = ["send", "recv"]
        t._agent = SimpleNamespace(deregister_memory=lambda desc: deregistered.append(desc))
        t._send_alloc = SimpleNamespace(close=lambda: closed.append("send"))
        t._recv_alloc = SimpleNamespace(close=lambda: closed.append("recv"))

        with pytest.raises(RuntimeError, match="did not exit"):
            t.close()

        assert stopped == [True]
        assert joined == [btr._CLOSE_JOIN_S]
        assert deregistered == []
        assert closed == []

    def test_scatter_write_result_non_bounce_fires_on_done(self):
        # Non-bounced path completes inline (the in-place WRITE already landed the KV).
        calls = []
        btr.scatter_write_result(
            btr.NoBounceTransport(), ("r", 0), 1, None, None, on_done=lambda ok: calls.append(ok)
        )
        assert calls == [True]

    def test_release_idle_reservation_frees_slot_and_is_idempotent(self, monkeypatch):
        # INIT-cancel immediate release (nothing published) must free the recv slot; idempotent.
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])
        assert t.reserve(req, num_writers=1) is True
        rid_slice = (req.unique_rid, req.slice_id)
        assert t.is_bounced(rid_slice) is True
        t.release_idle_reservation(rid_slice)
        assert t.is_bounced(rid_slice) is False
        assert t._recv_alloc.released  # slot freed
        t.release_idle_reservation(rid_slice)  # already gone -> no-op, must not raise

    def test_orphan_reservation_waits_for_explicit_drain_proof(self, monkeypatch):
        # A fixed quarantine timeout cannot prove a remote RMA is done. Keep
        # the slot live until the sender ACK supplies exact drain proof.
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])
        assert t.reserve(req, num_writers=1) is True
        rid_slice = (req.unique_rid, req.slice_id)
        t.orphan_reservation(rid_slice)
        assert t._recv_alloc.quarantined == []
        assert t._recv_alloc.released == []
        assert t.is_bounced(rid_slice) is True

        t._recv_alloc.reclaim_expired()
        assert t.is_bounced(rid_slice) is True
        assert t._recv_alloc.released == []

        t.confirm_drained(rid_slice)
        assert t.is_bounced(rid_slice) is False
        assert t._recv_alloc.released == [0]
        t.confirm_drained(rid_slice)  # already gone -> idempotent


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


# --------------------------------------------------------------------------- #
# core.TransferContext — the pure drain-before-release state machine.
# No CUDA / NIXL / allocator, so these run on any CPU (no skipif).
# --------------------------------------------------------------------------- #
class TestLifecycle:
    def _ctx(self, num_writers, per_writer_bytes=100, base_addr=0x1000):
        return bcore.TransferContext(
            rid_slice=(1, 0),
            slot_id=0,
            base_addr=base_addr,
            per_writer_bytes=per_writer_bytes,
            num_writers=num_writers,
        )

    def _dst(self, v=10):
        return dict(dst_ptrs=np.array([v], dtype=np.int64), sizes=np.array([100], dtype=np.int64))

    def test_writer_base_layout(self):
        c = self._ctx(3, per_writer_bytes=0x64, base_addr=0x1000)
        assert [c.writer_base(i) for i in range(3)] == [0x1000, 0x1064, 0x10C8]

    def test_single_writer_success_scatters_then_releases(self):
        c = self._ctx(1)
        c.bind_writer(3, c.writer_base(0))
        assert not c.ready_to_scatter() and not c.ready_to_settle()
        c.record_writer_result(3, succeeded=True, src_base=c.writer_base(0), **self._dst())
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
        c.bind_writer(7, c.writer_base(0))
        c.bind_writer(3, c.writer_base(1))
        c.record_writer_result(7, succeeded=True, src_base=c.writer_base(0), **self._dst())
        assert not c.ready_to_scatter()  # 1/2 writers
        assert not c.ready_to_settle()  # drain-before-release
        c.record_writer_result(3, succeeded=True, src_base=c.writer_base(1), **self._dst())
        assert c.ready_to_scatter()  # all success -> scatter

    def test_fanin_failed_then_success_releases(self):
        c = self._ctx(2)
        c.bind_writer(7, c.writer_base(0))
        c.bind_writer(3, c.writer_base(1))
        c.record_writer_result(7, succeeded=False)
        assert not c.ready_to_settle()  # a sibling is still pending -> hold
        c.record_writer_result(3, succeeded=True, src_base=c.writer_base(1), **self._dst())
        assert not c.ready_to_scatter()  # >=1 FAILED -> skip scatter
        assert c.ready_to_settle()
        ret = c.settle()
        assert ret.disposition is bcore.Disposition.RELEASE  # all drained -> free, not quarantine
        assert ret.success is False
        assert c.state is bcore.TransferState.FAILED

    def test_orphan_waits_for_drain_proof(self):
        c = self._ctx(2)
        c.bind_writer(7, c.writer_base(0))
        c.bind_writer(3, c.writer_base(1))
        c.record_writer_result(7, succeeded=True, src_base=c.writer_base(0), **self._dst())
        c.mark_orphaned()  # the other writer is in-doubt
        assert not c.ready_to_settle()
        c.confirm_drained()
        assert c.ready_to_settle()
        ret = c.settle()
        assert ret.disposition is bcore.Disposition.RELEASE and ret.success is False
        assert c.state is bcore.TransferState.CANCELLED_DRAINED

    def test_orphan_drain_proof_fires_unconditional_settlement_callback(self):
        calls = []
        c = self._ctx(1)
        c.bind_writer(3, c.writer_base(0))
        c.set_completion_callback(lambda ok: calls.append(("settled", ok)))
        c.mark_orphaned()
        c.confirm_drained()

        settlement = c.settle()

        assert settlement is not None
        assert calls == []
        settlement.on_done(settlement.success)
        assert calls == [("settled", False)]

    def test_success_fires_result_then_unconditional_settlement_callback(self):
        calls = []
        c = self._ctx(1)
        c.bind_writer(3, c.writer_base(0))
        c.on_done = lambda ok: calls.append(("result", ok))
        c.set_completion_callback(lambda ok: calls.append(("settled", ok)))
        c.record_writer_result(3, succeeded=True, src_base=c.writer_base(0), **self._dst())
        c.begin_scatter()
        c.finish_scatter(True)

        settlement = c.settle()

        assert settlement is not None
        settlement.on_done(settlement.success)
        assert calls == [("result", True), ("settled", True)]

    def test_settlement_callback_runs_when_result_callback_raises(self):
        calls = []
        c = self._ctx(1)
        c.bind_writer(3, c.writer_base(0))

        def fail_result(_ok):
            calls.append("result")
            raise RuntimeError("boom")

        c.on_done = fail_result
        c.set_completion_callback(lambda _ok: calls.append("settled"))
        c.record_writer_result(3, succeeded=True, src_base=c.writer_base(0), **self._dst())
        c.begin_scatter()
        c.finish_scatter(True)

        settlement = c.settle()

        assert settlement is not None
        with pytest.raises(RuntimeError, match="boom"):
            settlement.on_done(settlement.success)
        assert calls == ["result", "settled"]

    def test_empty_tail_success_is_rejected(self):
        c = self._ctx(1)
        c.bind_writer(3, c.writer_base(0))
        with pytest.raises(RuntimeError, match="complete scatter tail"):
            c.record_writer_result(3, succeeded=True)
        assert not c.ready_to_settle()

    def test_writers_locked_after_scatter_rejects_unbound_late_writer(self):
        c = self._ctx(1)
        c.bind_writer(3, c.writer_base(0))
        c.record_writer_result(3, succeeded=True, src_base=c.writer_base(0), **self._dst())
        c.begin_scatter()  # SCATTERING -> frozen
        with pytest.raises(RuntimeError, match="unexpected bounce writer"):
            c.record_writer_result(9, succeeded=False)
        assert 9 not in c._writer_ok

    def test_duplicate_writer_dedup(self):
        c = self._ctx(2)
        c.bind_writer(7, c.writer_base(0))
        c.bind_writer(3, c.writer_base(1))
        c.record_writer_result(7, succeeded=True, src_base=c.writer_base(0), **self._dst())
        c.record_writer_result(7, succeeded=False)  # same rank again -> ignored
        assert c._writer_ok[7] is True
        assert not c.ready_to_settle()  # still only 1 distinct writer of 2

    def test_scatter_failure_releases_as_failed(self):
        c = self._ctx(1)
        c.bind_writer(3, c.writer_base(0))
        c.record_writer_result(3, succeeded=True, src_base=c.writer_base(0), **self._dst())
        c.begin_scatter()
        c.finish_scatter(False)  # scatter kernel failed
        ret = c.settle()
        assert ret.disposition is bcore.Disposition.RELEASE and ret.success is False

    def test_orphan_after_scatter_is_ignored(self):
        # once SCATTERING, all writers already reported SUCCESS -> nothing is in doubt, so a late
        # orphan (e.g. a racing cancel) must NOT downgrade a clean transfer to quarantine.
        c = self._ctx(1)
        c.bind_writer(3, c.writer_base(0))
        c.record_writer_result(3, succeeded=True, src_base=c.writer_base(0), **self._dst())
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
