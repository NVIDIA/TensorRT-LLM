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
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native.bounce import config as bcfg

# core (the BounceTransport contract + RecvBounceContext state machine) is PURE (no CUDA / NIXL
# imports), so it is always importable on CPU.
from tensorrt_llm._torch.disaggregation.native.bounce import core as bcore
from tensorrt_llm._torch.disaggregation.native.receive_lifecycle import (
    LifecycleAction,
    PhysicalState,
    RecvTransferRegistry,
    WriterMode,
    WriterResult,
)

# transport/transfer pull CUDA-binding + torch deps at import; skip gracefully when those are
# absent (CPU-only env). Catch only ImportError so a genuine bug in the module still fails CI
# instead of being silently turned into a skip.
try:
    from tensorrt_llm._torch.disaggregation.native.bounce import buffer as bbuf
    from tensorrt_llm._torch.disaggregation.native.bounce import gather_scatter as bgs
    from tensorrt_llm._torch.disaggregation.native.bounce import impl as btr

    _HAVE_TRANSPORT = True
except ImportError:  # pragma: no cover - CPU-only env without CUDA bindings
    _HAVE_TRANSPORT = False

_MIB = 1024 * 1024


def test_transfer_context_compatibility_alias():
    from tensorrt_llm._torch.disaggregation.native import bounce

    assert bounce.TransferContext is bounce.RecvBounceContext
    assert "TransferContext" in bounce.__all__


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
# fan-in safety gate — equal split only within one PP stage
# --------------------------------------------------------------------------- #
def test_fanin_bounce_safe_gate():
    """Require one PP stage and no TP head duplication in either direction."""
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    safe = tfr.Receiver._fanin_bounce_safe

    def ov(dup, pp, peer_dup=1):
        return SimpleNamespace(
            duplicate_head_factor=dup,
            peer_duplicate_head_factor=peer_dup,
            overlap_pp_size=pp,
        )

    # A single PP stage is safe only when neither peer duplicates KV heads.
    assert safe(ov(1, 1)) is True
    assert safe(ov(1, 0)) is False
    assert safe(ov(2, 1)) is False  # duplicate heads / MLA -> some don't send
    assert safe(ov(1, 1, peer_dup=2)) is False  # reciprocal duplication
    # Every PP fan-in falls back regardless of peer-stage metadata.
    assert safe(ov(1, 4)) is False
    assert safe(ov(2, 4)) is False


def test_fanin_bounce_rejects_uniform_peer_pp_with_mismatched_boundaries():
    """Uniform peer stages can intersect one local PP stage by unequal extents."""
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")

    # Local PP rank 0 owns layers [0, 30), while globally uniform peer stages
    # own [0, 20) and [20, 40). Their local intersections are 20 and 10 layers,
    # so an equal two-writer bounce split would authorize the wrong boundary.
    local_start, local_end = 0, 30
    peer_boundaries = ((0, 20), (20, 40))
    intersections = [
        max(0, min(local_end, end) - max(local_start, start)) for start, end in peer_boundaries
    ]
    assert intersections == [20, 10]

    overlap = SimpleNamespace(
        duplicate_head_factor=1,
        peer_duplicate_head_factor=1,
        overlap_pp_size=2,
    )
    assert tfr.Receiver._fanin_bounce_safe(overlap) is False


# --------------------------------------------------------------------------- #
# NoBounceTransport — disabled no-op fallback
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAVE_TRANSPORT, reason="bounce.transport import needs CUDA bindings")
class TestNoBounce:
    def test_noop_behaviour(self):
        nb = btr.NoBounceTransport()
        factory_calls = []
        assert nb.enabled is False
        assert (
            nb.reserve(
                SimpleNamespace(),
                destination_intervals_factory=lambda: factory_calls.append(True) or [],
            )
            is False
        )
        assert factory_calls == []
        assert nb.build_request(SimpleNamespace()) is None
        assert nb.writer_base(("r", 0), 1) is None
        assert nb.is_bounced(("r", 0)) is False
        assert nb.mark_writer_exposed(("r", 0), 1) is False
        nb.record_no_access(("r", 0), 1)
        nb.mark_logical_failure(("r", 0))
        nb.mark_protocol_conflict(("r", 0))
        nb.mark_backend_quiesced()
        assert nb.retry_settlements()
        assert nb.retry_settlements("r")
        assert nb.retry_settlements(("r", 0))
        nb.record_result(("r", 0), 1)  # no-op, must not raise
        nb.record_failure(("r", 0), 1)  # no-op, must not raise
        nb.release_idle_reservation(("r", 0))  # no-op, must not raise
        nb.release_send(0)
        nb.quarantine_send(0)
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
# page-table sizing — every attention pool view contributes its gathered extent
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAVE_TRANSPORT, reason="bounce.transport import needs CUDA bindings")
def test_block_bytes_per_group_sums_every_physical_pool_view():
    from tensorrt_llm._torch.disaggregation.resource.page import AttentionLayerGroup

    layer_group = AttentionLayerGroup(
        pool_group_idx=0,
        pool_views=[
            SimpleNamespace(pool_idx=0),
            SimpleNamespace(pool_idx=1),
            # Sender descriptor construction iterates views, so a repeated
            # physical pool must contribute its full gathered extent again.
            SimpleNamespace(pool_idx=0),
        ],
    )
    page_table = SimpleNamespace(
        layer_groups=[layer_group],
        pool_groups=[
            SimpleNamespace(pools=[SimpleNamespace(slot_bytes=96), SimpleNamespace(slot_bytes=32)])
        ],
    )

    assert btr.block_bytes_per_group(page_table) == [224]


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
        self.active = set()
        self.closed = False

    @property
    def capacity(self):
        return self._cap

    def reserve(self, size, timeout=None):
        if size > self._cap:
            return None
        sid = self.next_id
        self.next_id += 1
        self.active.add(sid)
        return sid, self.base

    def release(self, slot_id):
        self.active.discard(slot_id)
        self.released.append(slot_id)

    def quarantine(self, slot_id):
        self.active.discard(slot_id)
        self.quarantined.append(slot_id)

    @property
    def has_outstanding(self):
        return bool(self.active or self.quarantined)

    def reg_descs(self):
        return []

    def close(self):
        if self.has_outstanding:
            raise RuntimeError("live fake slots")
        self.closed = True


class _DescriptorAlloc(_FakeAlloc):
    """Fake allocator with a unique registration descriptor."""

    def __init__(self, capacity_bytes, phys_chunk_size, name="kv_bounce"):
        super().__init__(capacity_bytes, phys_chunk_size, name=name)
        self.desc = object()

    def reg_descs(self):
        return self.desc


def _make_transport(monkeypatch, block_bytes_per_group, capacity=1 << 30, min_blocks=1):
    monkeypatch.setattr(btr, "SlotAllocator", _FakeAlloc)
    monkeypatch.setattr(btr.VmmBounceTransport, "_new_stream", lambda self: 0)
    monkeypatch.setattr(
        btr.VmmBounceTransport,
        "_destroy_stream",
        lambda self, attr_name: setattr(self, attr_name, None),
    )
    monkeypatch.setattr(
        btr.VmmBounceTransport,
        "_start_scatter_worker",
        lambda self, name: setattr(self, "_scatter_q", queue.Queue()),
    )
    agent = SimpleNamespace(register_memory=lambda d: None, deregister_memory=lambda d: None)
    return btr.VmmBounceTransport(
        agent,
        device_id=0,
        capacity_bytes=capacity,
        phys_chunk_size=32 * _MIB,
        block_bytes_per_group=block_bytes_per_group,
        min_blocks=min_blocks,
    )


def _recv_req(block_counts, rid=1, slice_id=0):
    return SimpleNamespace(
        block_ids_per_layer_groups=[np.arange(n, dtype=np.int64) for n in block_counts],
        unique_rid=rid,
        slice_id=slice_id,
        bounce_dst_base=None,
    )


def _write_meta():
    return SimpleNamespace(
        src_ptrs=np.array([0x1000], dtype=np.int64),
        dst_ptrs=np.array([0x2000], dtype=np.int64),
        sizes=np.array([8], dtype=np.int64),
        bounce_dst_base=0x3000,
        dst_device_id=0,
        peer_name="peer",
    )


def _run_mock_scatter_worker(transport, monkeypatch) -> None:
    transport._scatter_ready = threading.Event()
    monkeypatch.setattr(
        btr,
        "cudart",
        SimpleNamespace(
            cudaSetDevice=lambda device_id: ("ok",),
            cudaStreamSynchronize=lambda stream: ("ok",),
        ),
    )
    monkeypatch.setattr(btr, "CUASSERT", lambda result: result[1:])
    monkeypatch.setattr(btr, "scatter_contiguous", lambda *args, **kwargs: None)
    transport._scatter_q.put(None)
    transport._scatter_loop()


@pytest.mark.skipif(not _HAVE_TRANSPORT, reason="bounce.transport import needs CUDA bindings")
class TestTransportRollback:
    def _construct(self, agent, *, capacity=1024):
        return btr.VmmBounceTransport(
            agent,
            device_id=0,
            capacity_bytes=capacity,
            phys_chunk_size=512,
            block_bytes_per_group=[8],
            min_blocks=1,
        )

    def test_constructor_rolls_back_first_allocator_when_second_fails(self, monkeypatch):
        allocators = []

        def allocator_factory(*args, **kwargs):
            if allocators:
                raise RuntimeError("recv allocator failed")
            allocator = _DescriptorAlloc(*args, **kwargs)
            allocators.append(allocator)
            return allocator

        monkeypatch.setattr(btr, "SlotAllocator", allocator_factory)
        with pytest.raises(RuntimeError, match="recv allocator failed"):
            self._construct(SimpleNamespace())
        assert allocators[0].closed

    def test_constructor_rolls_back_partial_registration(self, monkeypatch):
        allocators = []

        def allocator_factory(*args, **kwargs):
            allocator = _DescriptorAlloc(*args, **kwargs)
            allocators.append(allocator)
            return allocator

        deregistered = []

        class Agent:
            def __init__(self):
                self.register_count = 0

            def register_memory(self, desc):
                self.register_count += 1
                if self.register_count == 2:
                    raise RuntimeError("second registration failed")

            def deregister_memory(self, desc):
                deregistered.append(desc)

        monkeypatch.setattr(btr, "SlotAllocator", allocator_factory)
        with pytest.raises(RuntimeError, match="second registration failed"):
            self._construct(Agent())
        # The failing registration may have mutated the backend before raising,
        # so rollback must conservatively deregister both attempted descriptors.
        assert deregistered == [allocator.desc for allocator in allocators]
        assert all(allocator.closed for allocator in allocators)

    def test_incomplete_constructor_rollback_retains_retry_owner(self, monkeypatch):
        allocators = []

        def allocator_factory(*args, **kwargs):
            allocator = _DescriptorAlloc(*args, **kwargs)
            allocators.append(allocator)
            return allocator

        allow_deregister = False

        class Agent:
            def __init__(self):
                self.register_count = 0

            def register_memory(self, desc):
                self.register_count += 1
                if self.register_count == 2:
                    raise RuntimeError("second registration failed")

            def deregister_memory(self, desc):
                if not allow_deregister:
                    raise RuntimeError("rollback deregistration failed")

        monkeypatch.setattr(btr, "SlotAllocator", allocator_factory)
        with pytest.raises(btr.IncompleteBounceInitializationError) as raised:
            self._construct(Agent())

        owner = raised.value.owner
        assert owner in btr._INCOMPLETE_TRANSPORTS
        assert not any(allocator.closed for allocator in allocators)

        allow_deregister = True
        owner.retry_initialization_rollback()
        assert owner not in btr._INCOMPLETE_TRANSPORTS
        assert all(allocator.closed for allocator in allocators)

    def test_cleanup_pending_transport_retries_retained_owner(self):
        calls = []
        owner = SimpleNamespace(retry_initialization_rollback=lambda: calls.append(True))
        transport = btr.CleanupPendingBounceTransport(owner)

        transport.close()
        transport.close()

        assert calls == [True]

    def test_create_bounce_preserves_incomplete_cleanup_owner(self, monkeypatch):
        calls = []
        owner = SimpleNamespace(retry_initialization_rollback=lambda: calls.append(True))
        error = btr.IncompleteBounceInitializationError(owner, RuntimeError("setup failed"))
        monkeypatch.setattr(btr, "block_bytes_per_group", lambda page_table: [])
        monkeypatch.setattr(
            btr.VmmBounceTransport,
            "from_config",
            lambda *args, **kwargs: (_ for _ in ()).throw(error),
        )

        transport = btr.create_bounce(object(), object(), device_id=0, page_table=object())
        assert isinstance(transport, btr.CleanupPendingBounceTransport)
        transport.close()
        assert calls == [True]

    def test_constructor_rolls_back_registrations_when_send_stream_fails(self, monkeypatch):
        allocators = []

        def allocator_factory(*args, **kwargs):
            allocator = _DescriptorAlloc(*args, **kwargs)
            allocators.append(allocator)
            return allocator

        deregistered = []
        agent = SimpleNamespace(
            register_memory=lambda desc: None,
            deregister_memory=lambda desc: deregistered.append(desc),
        )
        monkeypatch.setattr(btr, "SlotAllocator", allocator_factory)
        monkeypatch.setattr(
            btr.VmmBounceTransport,
            "_new_stream",
            lambda self: (_ for _ in ()).throw(RuntimeError("stream failed")),
        )
        with pytest.raises(RuntimeError, match="stream failed"):
            self._construct(agent)
        assert deregistered == [allocator.desc for allocator in allocators]
        assert all(allocator.closed for allocator in allocators)

    def test_constructor_rolls_back_streams_when_thread_start_fails(self, monkeypatch):
        allocators = []

        def allocator_factory(*args, **kwargs):
            allocator = _DescriptorAlloc(*args, **kwargs)
            allocators.append(allocator)
            return allocator

        class FailingThread:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                raise RuntimeError("thread start failed")

            def is_alive(self):
                return False

        streams = iter((11, 22))
        destroyed = []

        def destroy_stream(transport, attr_name):
            stream = getattr(transport, attr_name)
            if stream is not None:
                destroyed.append(stream)
                setattr(transport, attr_name, None)

        deregistered = []
        agent = SimpleNamespace(
            register_memory=lambda desc: None,
            deregister_memory=lambda desc: deregistered.append(desc),
        )
        monkeypatch.setattr(btr, "SlotAllocator", allocator_factory)
        monkeypatch.setattr(btr.VmmBounceTransport, "_new_stream", lambda self: next(streams))
        monkeypatch.setattr(btr.VmmBounceTransport, "_destroy_stream", destroy_stream)
        monkeypatch.setattr(btr.threading, "Thread", FailingThread)
        with pytest.raises(RuntimeError, match="thread start failed"):
            self._construct(agent)
        assert destroyed == [22, 11]
        assert deregistered == [allocator.desc for allocator in allocators]
        assert all(allocator.closed for allocator in allocators)

    def test_constructor_rolls_back_when_scatter_worker_device_setup_fails(self, monkeypatch):
        allocators = []

        def allocator_factory(*args, **kwargs):
            allocator = _DescriptorAlloc(*args, **kwargs)
            allocators.append(allocator)
            return allocator

        streams = iter((11, 22))
        destroyed = []
        deregistered = []
        agent = SimpleNamespace(
            register_memory=lambda desc: None,
            deregister_memory=lambda desc: deregistered.append(desc),
        )
        monkeypatch.setattr(btr, "SlotAllocator", allocator_factory)
        monkeypatch.setattr(btr.VmmBounceTransport, "_new_stream", lambda self: next(streams))
        monkeypatch.setattr(
            btr.VmmBounceTransport,
            "_destroy_stream",
            lambda owner, attr_name: (
                destroyed.append(getattr(owner, attr_name)),
                setattr(owner, attr_name, None),
            ),
        )
        monkeypatch.setattr(
            btr,
            "cudart",
            SimpleNamespace(cudaSetDevice=lambda device_id: ("error",)),
        )
        monkeypatch.setattr(
            btr,
            "CUASSERT",
            lambda result: (_ for _ in ()).throw(RuntimeError("set device failed")),
        )

        with pytest.raises(RuntimeError, match="scatter worker failed to initialize"):
            self._construct(agent)

        assert destroyed == [22, 11]
        assert deregistered == [allocator.desc for allocator in allocators]
        assert all(allocator.closed for allocator in allocators)

    @pytest.mark.parametrize("failure_stage", ["launch", "wait", "make_write"])
    def test_send_slot_rolls_back_on_build_failure(self, monkeypatch, failure_stage):
        transport = _make_transport(monkeypatch, block_bytes_per_group=[8])
        monkeypatch.setattr(
            transport,
            "_rollback_send_slot",
            lambda slot_id: (transport._send_alloc.release(slot_id), True)[1],
        )
        if failure_stage == "launch":
            monkeypatch.setattr(
                transport,
                "_launch_gather",
                lambda *args: (_ for _ in ()).throw(RuntimeError("launch failed")),
            )
        else:
            monkeypatch.setattr(transport, "_launch_gather", lambda *args: 17)
            if failure_stage == "wait":
                monkeypatch.setattr(
                    transport,
                    "_wait_gather",
                    lambda *args: (_ for _ in ()).throw(RuntimeError("wait failed")),
                )
            else:
                monkeypatch.setattr(transport, "_wait_gather", lambda *args: None)
                monkeypatch.setattr(
                    transport,
                    "_make_write",
                    lambda *args: (_ for _ in ()).throw(RuntimeError("make write failed")),
                )

        with pytest.raises(RuntimeError):
            transport.build_request(_write_meta())
        assert transport._send_alloc.released == [0]
        assert not transport._send_alloc.has_outstanding

    def test_failed_send_slot_releases_after_stream_fence(self, monkeypatch):
        transport = _make_transport(monkeypatch, block_bytes_per_group=[8])
        slot_id, _ = transport._send_alloc.reserve(8)
        monkeypatch.setattr(
            btr,
            "cudart",
            SimpleNamespace(cudaStreamSynchronize=lambda stream: ("ok",)),
        )
        monkeypatch.setattr(btr, "CUASSERT", lambda result: result[1:])

        assert transport._rollback_send_slot(slot_id) is True

        assert transport._send_alloc.released == [slot_id]
        assert transport._send_alloc.quarantined == []

    def test_failed_send_slot_quarantines_when_stream_fence_fails(self, monkeypatch):
        transport = _make_transport(monkeypatch, block_bytes_per_group=[8])
        slot_id, _ = transport._send_alloc.reserve(8)
        monkeypatch.setattr(
            btr,
            "cudart",
            SimpleNamespace(cudaStreamSynchronize=lambda stream: ("error",)),
        )

        def cuassert(result):
            raise RuntimeError("stream fence failed")

        monkeypatch.setattr(btr, "CUASSERT", cuassert)

        assert transport._rollback_send_slot(slot_id) is False

        assert transport._send_alloc.released == []
        assert transport._send_alloc.quarantined == [slot_id]
        monkeypatch.setattr(
            transport,
            "_launch_gather",
            lambda *args: pytest.fail("an unhealthy send stream must not launch more gathers"),
        )
        assert transport.build_request(_write_meta()) is None

    def test_build_raises_distinct_error_when_gather_source_remains_in_doubt(self, monkeypatch):
        transport = _make_transport(monkeypatch, block_bytes_per_group=[8])
        monkeypatch.setattr(
            transport,
            "_launch_gather",
            lambda *args: (_ for _ in ()).throw(RuntimeError("launch failed")),
        )
        monkeypatch.setattr(transport, "_rollback_send_slot", lambda slot_id: False)

        with pytest.raises(bcore.GatherSourceInDoubtError, match="without a positive CUDA fence"):
            transport.build_request(_write_meta())

    def test_quarantine_failure_still_reports_gather_source_in_doubt(self, monkeypatch):
        transport = _make_transport(monkeypatch, block_bytes_per_group=[8])
        monkeypatch.setattr(
            transport,
            "_launch_gather",
            lambda *args: (_ for _ in ()).throw(RuntimeError("launch failed")),
        )
        monkeypatch.setattr(
            btr,
            "cudart",
            SimpleNamespace(cudaStreamSynchronize=lambda stream: ("error",)),
        )
        monkeypatch.setattr(
            btr,
            "CUASSERT",
            lambda result: (_ for _ in ()).throw(RuntimeError("stream fence failed")),
        )
        monkeypatch.setattr(
            transport._send_alloc,
            "quarantine",
            lambda slot_id: (_ for _ in ()).throw(RuntimeError("quarantine failed")),
        )

        with pytest.raises(bcore.GatherSourceInDoubtError, match="without a positive CUDA fence"):
            transport.build_request(_write_meta())

        assert not transport._send_stream_healthy
        assert transport._send_alloc.active == {0}

    def test_ambiguous_nixl_slot_is_quarantined_from_reuse(self, monkeypatch):
        transport = _make_transport(monkeypatch, block_bytes_per_group=[8])
        slot_id, _ = transport._send_alloc.reserve(8)

        transport.quarantine_send(slot_id)
        transport.release_send(slot_id)

        assert transport._send_alloc.released == [slot_id]
        assert transport._send_alloc.quarantined == [slot_id]
        assert transport._send_alloc.has_outstanding
        with pytest.raises(RuntimeError, match="outstanding arena slots"):
            transport.close()

    def test_event_record_failure_destroys_created_event(self, monkeypatch):
        transport = _make_transport(monkeypatch, block_bytes_per_group=[8])
        destroyed = []
        monkeypatch.setattr(btr, "gather_contiguous", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            btr,
            "cudart",
            SimpleNamespace(
                cudaEventCreate=lambda: ("ok", 17),
                cudaEventRecord=lambda event, stream: ("error",),
                cudaEventDestroy=lambda event: (destroyed.append(event), ("ok",))[1],
            ),
        )

        def cuassert(result):
            if result[0] == "error":
                raise RuntimeError("event record failed")
            return result[1:]

        monkeypatch.setattr(btr, "CUASSERT", cuassert)
        with pytest.raises(RuntimeError, match="event record failed"):
            transport._launch_gather(0x1000, _write_meta(), 8)
        assert destroyed == [17]

    def test_event_wait_failure_still_destroys_event(self, monkeypatch):
        transport = _make_transport(monkeypatch, block_bytes_per_group=[8])
        transport._pending_events.append(17)
        destroyed = []
        monkeypatch.setattr(
            btr,
            "cudart",
            SimpleNamespace(
                cudaEventSynchronize=lambda event: ("error",),
                cudaEventDestroy=lambda event: (destroyed.append(event), ("ok",))[1],
            ),
        )

        def cuassert(result):
            if result[0] == "error":
                raise RuntimeError("event wait failed")
            return result[1:]

        monkeypatch.setattr(btr, "CUASSERT", cuassert)
        with pytest.raises(RuntimeError, match="event wait failed"):
            transport._wait_gather(17)
        assert destroyed == [17]

    def test_close_retries_failed_event_destruction(self, monkeypatch):
        transport = _make_transport(monkeypatch, block_bytes_per_group=[8])
        transport._pending_events.append(17)
        fail_once = {"value": True}
        destroyed = []

        def destroy_event(event):
            destroyed.append(event)
            if fail_once["value"]:
                fail_once["value"] = False
                return ("error",)
            return ("ok",)

        monkeypatch.setattr(btr, "cudart", SimpleNamespace(cudaEventDestroy=destroy_event))

        def cuassert(result):
            if result[0] == "error":
                raise RuntimeError("event destroy failed")
            return result[1:]

        monkeypatch.setattr(btr, "CUASSERT", cuassert)
        monkeypatch.setattr(
            btr.VmmBounceTransport,
            "_destroy_stream",
            lambda owner, attr_name: setattr(owner, attr_name, None),
        )

        with pytest.raises(RuntimeError, match="failed to destroy 1 CUDA event"):
            transport.close()
        assert transport._pending_events == [17]
        assert not transport._send_alloc.closed

        transport.close()
        assert transport._pending_events == []
        assert destroyed == [17, 17]

    def test_destroy_stream_evicts_only_its_device_metadata(self, monkeypatch):
        transport = object.__new__(btr.VmmBounceTransport)
        transport._device_id = 1
        transport._send_stream = 17
        operations = []
        monkeypatch.setattr(
            btr,
            "cudart",
            SimpleNamespace(
                cudaStreamDestroy=lambda stream: operations.append(("destroy", stream)) or ("ok",)
            ),
        )
        monkeypatch.setattr(btr, "CUASSERT", lambda result: result[1:])
        monkeypatch.setattr(
            btr,
            "release_meta_buffers",
            lambda stream, device_id: operations.append(("release_metadata", stream, device_id)),
        )

        transport._destroy_stream("_send_stream")

        assert transport._send_stream is None
        assert operations == [("release_metadata", 17, 1), ("destroy", 17)]

    def test_metadata_cache_keys_include_device_and_can_be_evicted(self):
        dev0 = SimpleNamespace(index=0)
        dev1 = SimpleNamespace(index=1)
        assert bgs._meta_buffer_key(17, dev0) != bgs._meta_buffer_key(17, dev1)
        bgs._meta_buffers[(0, 17)] = (object(), object(), 1)
        bgs._meta_buffers[(1, 17)] = (object(), object(), 1)
        try:
            bgs.release_meta_buffers(17, 0)
            assert (0, 17) not in bgs._meta_buffers
            assert (1, 17) in bgs._meta_buffers
        finally:
            bgs._meta_buffers.pop((0, 17), None)
            bgs._meta_buffers.pop((1, 17), None)

    def test_gathers_hold_stream_metadata_until_completion(self, monkeypatch):
        transport = _make_transport(monkeypatch, block_bytes_per_group=[8])
        first_waiting = threading.Event()
        allow_first_completion = threading.Event()
        second_started = threading.Event()
        launches = []
        results = []

        def launch(*args):
            event = len(launches) + 1
            launches.append(event)
            return event

        def wait(event):
            if event == 1:
                first_waiting.set()
                assert allow_first_completion.wait(timeout=1)

        monkeypatch.setattr(transport, "_launch_gather", launch)
        monkeypatch.setattr(transport, "_wait_gather", wait)
        monkeypatch.setattr(transport, "_make_write", lambda *args: object())

        first = threading.Thread(
            target=lambda: results.append(transport.build_request(_write_meta()))
        )

        def run_second():
            second_started.set()
            results.append(transport.build_request(_write_meta()))

        second = threading.Thread(target=run_second)
        first.start()
        assert first_waiting.wait(timeout=1)
        second.start()
        assert second_started.wait(timeout=1)
        # The second launch cannot refill the shared stream metadata yet.
        assert launches == [1]
        allow_first_completion.set()
        first.join(timeout=1)
        second.join(timeout=1)

        assert not first.is_alive() and not second.is_alive()
        assert launches == [1, 2]
        assert len(results) == 2

    def test_close_retries_only_failed_deregistrations(self, monkeypatch):
        allocators = []

        def allocator_factory(*args, **kwargs):
            allocator = _DescriptorAlloc(*args, **kwargs)
            allocators.append(allocator)
            return allocator

        calls = []
        fail_once = {"value": True}

        class Agent:
            def register_memory(self, desc):
                pass

            def deregister_memory(self, desc):
                calls.append(desc)
                if desc is allocators[1].desc and fail_once["value"]:
                    fail_once["value"] = False
                    raise RuntimeError("deregister failed")

        monkeypatch.setattr(btr, "SlotAllocator", allocator_factory)
        monkeypatch.setattr(btr.VmmBounceTransport, "_new_stream", lambda self: 0)
        monkeypatch.setattr(
            btr.VmmBounceTransport,
            "_start_scatter_worker",
            lambda self, name: setattr(self, "_scatter_q", queue.Queue()),
        )
        monkeypatch.setattr(
            btr.VmmBounceTransport,
            "_destroy_stream",
            lambda self, attr_name: setattr(self, attr_name, None),
        )
        transport = self._construct(Agent())

        with pytest.raises(RuntimeError, match="failed to deregister"):
            transport.close()
        assert transport._registered_descs == [allocators[1].desc]
        assert not any(allocator.closed for allocator in allocators)

        transport.close()
        transport.close()
        assert calls.count(allocators[0].desc) == 1
        assert calls.count(allocators[1].desc) == 2
        assert all(allocator.closed for allocator in allocators)

    def test_close_retries_only_failed_stream_destruction(self, monkeypatch):
        transport = _make_transport(monkeypatch, block_bytes_per_group=[8])
        transport._scatter_stream = 22
        calls = []
        failed_once = {"value": True}

        def destroy_stream(owner, attr_name):
            stream = getattr(owner, attr_name)
            if stream is None:
                return
            calls.append(stream)
            if stream == 22 and failed_once["value"]:
                failed_once["value"] = False
                raise RuntimeError("stream destroy failed")
            setattr(owner, attr_name, None)

        monkeypatch.setattr(btr.VmmBounceTransport, "_destroy_stream", destroy_stream)
        with pytest.raises(RuntimeError, match="failed to destroy"):
            transport.close()
        assert transport._scatter_stream == 22
        assert transport._send_stream is None
        assert not transport._send_alloc.closed

        transport.close()
        transport.close()
        assert calls.count(22) == 2
        assert calls.count(0) == 1

    def test_concurrent_close_retires_resources_once(self, monkeypatch):
        transport = _make_transport(monkeypatch, block_bytes_per_group=[8])
        entered = threading.Event()
        release = threading.Event()
        second_started = threading.Event()
        stop_calls = []

        def stop_worker(owner):
            stop_calls.append(True)
            entered.set()
            assert release.wait(timeout=1)

        monkeypatch.setattr(btr.VmmBounceTransport, "_stop_scatter_worker", stop_worker)
        monkeypatch.setattr(
            btr.VmmBounceTransport,
            "_destroy_stream",
            lambda owner, attr_name: setattr(owner, attr_name, None),
        )
        errors = []

        def close(second=False):
            if second:
                second_started.set()
            try:
                transport.close()
            except Exception as error:
                errors.append(error)

        first = threading.Thread(target=close)
        second = threading.Thread(target=lambda: close(second=True))
        first.start()
        assert entered.wait(timeout=1)
        second.start()
        assert second_started.wait(timeout=1)
        release.set()
        first.join(timeout=1)
        second.join(timeout=1)

        assert not first.is_alive() and not second.is_alive()
        assert errors == []
        assert len(stop_calls) == 1


@pytest.mark.skipif(not _HAVE_TRANSPORT, reason="bounce.transport import needs CUDA bindings")
class TestReceiveReserve:
    def test_reserve_stamps_single_writer_base(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])  # total = 2 * 100 = 200
        assert t.reserve(req, writer_ranks=[7]) is True
        assert req.bounce_dst_base == 0x100000
        assert t.writer_base((req.unique_rid, req.slice_id), 7) == 0x100000
        assert t.writer_base((req.unique_rid, req.slice_id), 9) is None

    @pytest.mark.parametrize("slot_bytes", [[3], [100], [100, 100], [100, 200]])
    def test_reserve_rejects_every_multiwriter_plan(self, monkeypatch, slot_bytes):
        t = _make_transport(monkeypatch, block_bytes_per_group=slot_bytes)
        req = _recv_req([2] * len(slot_bytes))
        assert t.reserve(req, writer_ranks=[7, 3]) is False
        assert req.bounce_dst_base is None
        assert not t._recv_alloc.has_outstanding

    def test_reserve_heterogeneous_single_writer_ok(self, monkeypatch):
        # A single writer has no split, so heterogeneous slot bytes are fine.
        t = _make_transport(monkeypatch, block_bytes_per_group=[100, 200])
        assert t.reserve(_recv_req([2, 2]), writer_ranks=[7]) is True

    def test_reserve_single_writer_ok(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[3])
        req = _recv_req([1])  # total = 3, one writer -> no even-split requirement
        assert t.reserve(req, writer_ranks=[3]) is True

    def test_reserve_counts_only_valid_block_ids(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([0])
        # Prefix/SWA tables may retain BAD_PAGE_INDEX entries for evicted
        # blocks. Only the two extractable slots contribute bytes.
        req.block_ids_per_layer_groups = [np.array([4, -1, 9, -1], dtype=np.int64)]

        assert t.reserve(req, writer_ranks=[3]) is True
        ctx = t._reserved_map[(req.unique_rid, req.slice_id)]
        assert ctx.per_writer_bytes == 200

    @pytest.mark.parametrize(
        "remaining_blocks",
        [
            np.array([8, 9], dtype=np.int64),
            np.array([31], dtype=np.int64),
        ],
        ids=["prefix-trimmed", "swa-trimmed"],
    )
    def test_reserve_sizes_only_the_remaining_trimmed_suffix(self, monkeypatch, remaining_blocks):
        t = _make_transport(monkeypatch, block_bytes_per_group=[64])
        req = _recv_req([0])
        req.block_ids_per_layer_groups = [remaining_blocks]

        assert t.reserve(req, writer_ranks=[3]) is True
        ctx = t._reserved_map[(req.unique_rid, req.slice_id)]
        assert ctx.per_writer_bytes == remaining_blocks.size * 64

    def test_reserve_binds_trusted_destination_intervals(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[8])
        req = _recv_req([1])
        assert t.reserve(
            req,
            writer_ranks=[3],
            destination_intervals=[(0x2000, 8)],
        )
        ctx = t._reserved_map[(req.unique_rid, req.slice_id)]
        assert ctx._destination_intervals == ((0x2000, 0x2008),)

    def test_reserve_rejects_aliased_destination_union_before_publication(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[200])
        req = _recv_req([1])

        assert not t.reserve(
            req,
            writer_ranks=[3],
            destination_intervals=[(0x2000, 100), (0x2000, 100)],
        )

        assert req.bounce_dst_base is None
        assert t._reserved_map == {}
        assert t._recv_alloc.released == [0]
        assert not t._recv_alloc.has_outstanding

    def test_destination_intervals_factory_runs_only_after_allocator_admission(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[8])
        req = _recv_req([1])
        calls = []
        t._recv_alloc.reserve = lambda size, timeout=None: None

        assert not t.reserve(
            req,
            writer_ranks=[3],
            destination_intervals_factory=lambda: calls.append(True) or [(0x2000, 8)],
        )
        assert calls == []

    def test_destination_intervals_factory_canonicalizes_without_retaining_input(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[8])
        req = _recv_req([1])
        raw_intervals = {(0x2004, 4), (0x2000, 4)}
        calls = []

        def build_intervals():
            calls.append(t._pending_reservations)
            return raw_intervals

        assert t.reserve(
            req,
            writer_ranks=[3],
            destination_intervals_factory=build_intervals,
        )
        ctx = t._reserved_map[(req.unique_rid, req.slice_id)]
        assert calls == [1]
        assert "destination_intervals" not in ctx.__dict__
        assert ctx._destination_intervals == ((0x2000, 0x2008),)

        raw_intervals.add((0x3000, 8))
        assert ctx._destination_intervals == ((0x2000, 0x2008),)

    def test_invalid_lazy_destination_intervals_release_reserved_slot(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[8])
        req = _recv_req([1])
        assert not t.reserve(
            req,
            writer_ranks=[3],
            destination_intervals_factory=lambda: [],
        )
        assert t._recv_alloc.released == [0]
        assert not t._recv_alloc.has_outstanding

    def test_destination_intervals_factory_error_releases_reserved_slot(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[8])
        req = _recv_req([1])

        def fail_to_build_intervals():
            raise RuntimeError("manifest construction failed")

        with pytest.raises(RuntimeError, match="manifest construction failed"):
            t.reserve(
                req,
                writer_ranks=[3],
                destination_intervals_factory=fail_to_build_intervals,
            )
        assert t._recv_alloc.released == [0]
        assert not t._recv_alloc.has_outstanding

    def test_invalid_destination_intervals_release_reserved_slot(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[8])
        req = _recv_req([1])
        assert not t.reserve(req, writer_ranks=[3], destination_intervals=[(0, 8)])
        assert t._recv_alloc.released == [0]
        assert not t._recv_alloc.has_outstanding

    @pytest.mark.parametrize("writer_ranks", [[], [3, 3], [-1], [True], 1])
    def test_reserve_rejects_invalid_exact_writer_plan(self, monkeypatch, writer_ranks):
        t = _make_transport(monkeypatch, block_bytes_per_group=[3])
        req = _recv_req([1])
        assert t.reserve(req, writer_ranks=writer_ranks) is False
        assert req.bounce_dst_base is None

    def test_reserve_too_small_falls_back(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100], min_blocks=96)
        assert t.reserve(_recv_req([4]), writer_ranks=[3]) is False  # 4 < 96 blocks

    def test_reserve_unknown_slot_size_falls_back(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])  # only 1 group known
        assert t.reserve(_recv_req([2, 2]), writer_ranks=[3]) is False  # 2nd group unknown

    def test_reserve_oversize_falls_back(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[1000], capacity=500)
        assert t.reserve(_recv_req([2]), writer_ranks=[3]) is False  # total 2000 > cap 500

    def test_on_done_fires_after_scatter_lands(self, monkeypatch):
        # The completion cb rides the context and fires only when the worker records the scatter as
        # done -> the gen never observes completion before the KV is scattered into place.
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])
        assert t.reserve(req, writer_ranks=[3]) is True
        rid_slice = (req.unique_rid, req.slice_id)
        calls = []
        t.record_result(
            rid_slice,
            3,
            np.array([10], dtype=np.int64),
            np.array([200], dtype=np.int64),
            src_base=0x100000,
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
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])
        assert t.reserve(req, writer_ranks=[3]) is True
        rid_slice = (req.unique_rid, req.slice_id)
        calls = []
        t.record_result(rid_slice, 3, None, None, on_done=lambda ok: calls.append(ok))
        assert calls == [True]
        assert t._scatter_q.empty()
        assert t._recv_alloc.released  # slot freed

    def test_missing_key_is_dropped(self, monkeypatch):
        # A late/duplicate result for an already-settled (popped) rid_slice is dropped: the context's own
        # settle already fired completion, so re-firing here would double-report.
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
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
        assert t.reserve(req, writer_ranks=[3]) is True
        rid_slice = (req.unique_rid, req.slice_id)
        assert t.is_bounced(rid_slice) is True
        t.release_idle_reservation(rid_slice)
        assert t.is_bounced(rid_slice) is False
        assert t._recv_alloc.released  # slot freed
        t.release_idle_reservation(rid_slice)  # already gone -> no-op, must not raise

    def test_orphan_reservation_waits_for_backend_quiescence(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])
        assert t.reserve(req, writer_ranks=[7])
        rid_slice = (req.unique_rid, req.slice_id)

        t.orphan_reservation(rid_slice)

        assert t.is_bounced(rid_slice)
        assert t._recv_alloc.released == []
        assert t._recv_alloc.quarantined == []

        t.mark_backend_quiesced(rid_slice)

        assert not t.is_bounced(rid_slice)
        assert t._recv_alloc.released == [0]

    def test_failure_callback_fires_only_after_exposed_writer_drains(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])
        assert t.reserve(req, writer_ranks=[7])
        key = (req.unique_rid, req.slice_id)
        assert t.mark_writer_exposed(key, 7)
        calls = []
        t.mark_logical_failure(key, on_done=lambda ok: calls.append(ok))
        assert calls == []
        assert t.is_bounced(key)
        t.record_failure(key, 7)
        assert calls == [False]
        assert not t.is_bounced(key)

    def test_failed_settlement_ack_retries_without_double_release(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])
        assert t.reserve(req, writer_ranks=[7])
        key = (req.unique_rid, req.slice_id)
        calls = []

        def acknowledge(ok):
            calls.append(ok)
            if len(calls) == 1:
                raise RuntimeError("registry temporarily unavailable")

        t.record_result(key, 7, None, None, on_done=acknowledge)
        assert calls == [True]
        assert t._recv_alloc.released == [0]
        assert t.is_bounced(key)

        # Any later observation, including a duplicate result, retries the
        # durable acknowledgement without releasing the slot a second time.
        t.record_result(key, 7, None, None)
        assert calls == [True, True]
        assert t._recv_alloc.released == [0]
        assert not t.is_bounced(key)

    def test_settlement_retry_replays_registry_update_until_session_ack(self, monkeypatch):
        tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
        from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, SessionArgsBase
        from tensorrt_llm.disaggregated_params import DisaggregatedParams

        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2], rid=101)
        assert t.reserve(req, writer_ranks=[7])
        key = (req.unique_rid, req.slice_id)

        registry = RecvTransferRegistry()
        assert registry.prepare(key, {7}, has_bounce_slot=True).accepted
        assert registry.begin_publication(key, 7).publication_allowed
        assert registry.mark_published(key, 7).accepted
        ready = registry.record_result(key, 7, WriterResult.SUCCESS, WriterMode.BOUNCE)
        assert ready.actions == (LifecycleAction.START_BOUNCE_SCATTER,)

        params = DisaggregatedParams(
            disagg_request_id=key[0],
            ctx_request_id=key[0],
            ctx_dp_rank=0,
        )
        task = tfr.KVRecvTask(key[0], KVSlice(), key[1], params, aux_slot=None)
        task.status = tfr.TaskStatus.TRANSFERRING
        task._perf_timer = None

        receiver = object.__new__(tfr.Receiver)
        receiver._shutdown = True
        receiver._recv_registry = registry
        receiver._sessions_lock = threading.Lock()
        receiver._bounce_lifecycle_delivery_lock = threading.Lock()
        receiver._pending_bounce_lifecycle_deliveries = {}
        receiver._registrar = SimpleNamespace(
            self_rank_info=SimpleNamespace(instance_name="receiver", instance_rank=0)
        )

        session = object.__new__(tfr.RxSession)
        session._closed = True
        session.request_id = key[0]
        session._base_args = SessionArgsBase(params)
        session.lock = threading.Lock()
        session._kv_tasks = [task]
        session._receiver = receiver
        receiver._sessions = {key[0]: session}

        delivered_updates = []
        process_update = tfr.RxSession.process_lifecycle_update.__get__(session, tfr.RxSession)

        def fail_once(update, *, peer_rank=None):
            delivered_updates.append(update)
            if len(delivered_updates) == 1:
                raise RuntimeError("consumer temporarily unavailable")
            process_update(update, peer_rank=peer_rank)

        session.process_lifecycle_update = fail_once
        t.record_result(
            key,
            7,
            np.array([0x2000], dtype=np.int64),
            np.array([200], dtype=np.int64),
            src_base=0x100000,
            on_done=lambda succeeded: receiver._finish_bounce(key, succeeded, 7),
        )
        bounce_context, _descs = t._scatter_q.get_nowait()
        t._apply(bounce_context.rid_slice, lambda context: context.finish_scatter(True))

        assert task.status is tfr.TaskStatus.TRANSFERRING
        assert registry.context_snapshot(key).physical_state is PhysicalState.DRAINED
        assert t._recv_alloc.released == [0]
        assert t.is_bounced(key)

        assert t.retry_settlements()
        assert task.status is tfr.TaskStatus.TRANSFERRED
        assert delivered_updates[0] is delivered_updates[1]
        assert receiver._pending_bounce_lifecycle_deliveries == {}
        assert t._recv_alloc.released == [0]

    def test_concurrent_settlement_retry_does_not_duplicate_callback(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])
        assert t.reserve(req, writer_ranks=[7])
        key = (req.unique_rid, req.slice_id)
        callback_entered = threading.Event()
        allow_callback = threading.Event()
        calls = []

        def acknowledge(ok):
            calls.append(ok)
            callback_entered.set()
            assert allow_callback.wait(timeout=1)

        worker = threading.Thread(
            target=lambda: t.record_result(key, 7, None, None, on_done=acknowledge)
        )
        worker.start()
        assert callback_entered.wait(timeout=1)
        assert not t.retry_settlements()
        allow_callback.set()
        worker.join(timeout=1)

        assert not worker.is_alive()
        assert calls == [True]
        assert t._recv_alloc.released == [0]
        assert t.retry_settlements()

    def test_scatter_worker_continues_after_physical_settlement_failure(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        first_req = _recv_req([1], rid=1)
        second_req = _recv_req([1], rid=2)
        assert t.reserve(first_req, writer_ranks=[7])
        assert t.reserve(second_req, writer_ranks=[7])
        first_key = (first_req.unique_rid, first_req.slice_id)
        second_key = (second_req.unique_rid, second_req.slice_id)
        for key, dst_ptr in ((first_key, 0x2000), (second_key, 0x3000)):
            t.record_result(
                key,
                7,
                np.array([dst_ptr], dtype=np.int64),
                np.array([100], dtype=np.int64),
                src_base=0x100000,
            )

        release = t._recv_alloc.release
        failed_once = False

        def fail_first_release(slot_id):
            nonlocal failed_once
            if slot_id == 0 and not failed_once:
                failed_once = True
                raise RuntimeError("allocator temporarily unavailable")
            release(slot_id)

        t._recv_alloc.release = fail_first_release
        _run_mock_scatter_worker(t, monkeypatch)

        assert t.is_bounced(first_key)
        assert not t.is_bounced(second_key)
        assert t._recv_alloc.released == [1]
        assert t._scatter_worker_error is None
        assert t._scatter_stream_healthy
        assert t._accepting_reservations
        assert t.retry_settlements(first_key)
        assert t._recv_alloc.released == [1, 0]

    def test_scoped_settlement_retry_ignores_unrelated_request(self, monkeypatch):
        tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        first_req = _recv_req([1], rid=1)
        second_req = _recv_req([1], rid=2)
        assert t.reserve(first_req, writer_ranks=[7])
        assert t.reserve(second_req, writer_ranks=[7])
        first_key = (first_req.unique_rid, first_req.slice_id)
        second_key = (second_req.unique_rid, second_req.slice_id)
        second_calls = []

        def reject_first(_succeeded):
            raise RuntimeError("request A consumer unavailable")

        def accept_second_on_retry(succeeded):
            second_calls.append(succeeded)
            if len(second_calls) == 1:
                raise RuntimeError("request B consumer temporarily unavailable")

        t.record_result(first_key, 7, None, None, on_done=reject_first)
        t.record_result(second_key, 7, None, None, on_done=accept_second_on_retry)
        assert t.is_bounced(first_key)
        assert t.is_bounced(second_key)

        assert t.retry_settlements(second_key)
        assert not t.is_bounced(second_key)
        assert t.is_bounced(first_key)

        # Request-scoped drain must not inherit request A's retry failure.
        session = SimpleNamespace(
            disagg_request_id=second_req.unique_rid,
            _receiver=SimpleNamespace(
                _bounce=t,
                _recv_registry=SimpleNamespace(is_request_drained=lambda rid: rid == 2),
            ),
            has_untracked_receive_activity=lambda: False,
        )
        assert tfr.RxSession.resources_drained(session)
        assert not t.retry_settlements(first_req.unique_rid)
        assert not t.retry_settlements()
        assert second_calls == [True, True]

    def test_unexpected_scatter_worker_failure_closes_admission(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        first_req = _recv_req([1], rid=1)
        second_req = _recv_req([1], rid=2)
        assert t.reserve(first_req, writer_ranks=[7])
        assert t.reserve(second_req, writer_ranks=[7])
        first_key = (first_req.unique_rid, first_req.slice_id)
        second_key = (second_req.unique_rid, second_req.slice_id)
        first_logical_failures, second_calls = [], []
        for key, dst_ptr, on_done, on_logical_failure in (
            (first_key, 0x2000, None, lambda: first_logical_failures.append(True)),
            (second_key, 0x3000, lambda ok: second_calls.append(ok), None),
        ):
            t.record_result(
                key,
                7,
                np.array([dst_ptr], dtype=np.int64),
                np.array([100], dtype=np.int64),
                src_base=0x100000,
                on_done=on_done,
                on_logical_failure=on_logical_failure,
            )

        apply = t._apply
        failed_once = False

        def fail_first_state_update(rid_slice, mutate):
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                raise RuntimeError("unexpected state-machine failure")
            apply(rid_slice, mutate)

        t._apply = fail_first_state_update
        _run_mock_scatter_worker(t, monkeypatch)

        assert isinstance(t._scatter_worker_error, RuntimeError)
        assert not t._scatter_stream_healthy
        assert not t._accepting_reservations
        assert t.is_bounced(first_key)
        first = t._reserved_map[first_key]
        assert first.logical_failed
        assert first.scatter_state is bcore.ScatterState.FAILED
        assert first_logical_failures == [True]
        assert not t.is_bounced(second_key)
        assert second_calls == [False]
        assert t._recv_alloc.released == [1]
        assert not t.reserve(_recv_req([1], rid=3), writer_ranks=[7])

    def test_scatter_failure_suppresses_later_unlaunched_queue_entries(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        first_req = _recv_req([1], rid=1)
        second_req = _recv_req([1], rid=2)
        assert t.reserve(first_req, writer_ranks=[7])
        assert t.reserve(second_req, writer_ranks=[7])
        first_key = (first_req.unique_rid, first_req.slice_id)
        second_key = (second_req.unique_rid, second_req.slice_id)
        first_calls, second_calls, first_logical_failures = [], [], []
        t.record_result(
            first_key,
            7,
            np.array([0x2000], dtype=np.int64),
            np.array([100], dtype=np.int64),
            src_base=0x100000,
            on_done=lambda ok: first_calls.append(ok),
            on_logical_failure=lambda: first_logical_failures.append(True),
        )
        t.record_result(
            second_key,
            7,
            np.array([0x3000], dtype=np.int64),
            np.array([100], dtype=np.int64),
            src_base=0x100000,
            on_done=lambda ok: second_calls.append(ok),
        )
        t._scatter_ready = threading.Event()
        monkeypatch.setattr(
            btr,
            "cudart",
            SimpleNamespace(cudaSetDevice=lambda device_id: ("ok",)),
        )
        monkeypatch.setattr(btr, "CUASSERT", lambda result: result[1:])
        monkeypatch.setattr(
            btr,
            "scatter_contiguous",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("scatter failed")),
        )

        t._scatter_loop()

        assert t.is_bounced(first_key)  # the launched CUDA work remains ambiguous
        assert first_calls == []
        assert first_logical_failures == [True]
        assert not t.is_bounced(second_key)  # this entry never launched and settles as failed
        assert second_calls == [False]
        assert t._recv_alloc.released == [1]

    def test_scatter_failure_fails_request_but_retains_physical_ownership(self, monkeypatch):
        tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
        from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, SessionArgsBase
        from tensorrt_llm._torch.disaggregation.native.receive_lifecycle import LogicalState
        from tensorrt_llm.disaggregated_params import DisaggregatedParams

        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([1], rid=101)
        assert t.reserve(req, writer_ranks=[7])
        key = (req.unique_rid, req.slice_id)

        registry = RecvTransferRegistry()
        assert registry.prepare(key, {7}, has_bounce_slot=True).accepted
        assert registry.begin_publication(key, 7).publication_allowed
        assert registry.mark_published(key, 7).accepted
        registry.record_result(key, 7, WriterResult.SUCCESS, WriterMode.BOUNCE)

        params = DisaggregatedParams(
            disagg_request_id=key[0],
            ctx_request_id=key[0],
            ctx_dp_rank=0,
        )
        task = tfr.KVRecvTask(key[0], KVSlice(), key[1], params, aux_slot=None)
        task.status = tfr.TaskStatus.TRANSFERRING
        task.lifecycle_managed = True
        task._perf_timer = None

        receiver = object.__new__(tfr.Receiver)
        receiver._shutdown = True
        receiver._recv_registry = registry
        receiver._bounce = t
        receiver._sessions_lock = threading.Lock()
        receiver._bounce_lifecycle_delivery_lock = threading.Lock()
        receiver._pending_bounce_lifecycle_deliveries = {}
        receiver._pending_bounce_logical_failure_deliveries = {}
        receiver._registrar = SimpleNamespace(
            self_rank_info=SimpleNamespace(instance_name="receiver", instance_rank=0)
        )

        session = object.__new__(tfr.RxSession)
        session._closed = True
        session.request_id = key[0]
        session._base_args = SessionArgsBase(params)
        session.lock = threading.Lock()
        session._kv_tasks = [task]
        session._receiver = receiver
        session._need_aux = False
        session._terminal_status = None
        session._active_receive_dispatches = 0
        receiver._sessions = {key[0]: session}

        t.record_result(
            key,
            7,
            np.array([0x2000], dtype=np.int64),
            np.array([100], dtype=np.int64),
            src_base=0x100000,
            on_done=lambda succeeded: receiver._finish_bounce(key, succeeded, 7),
            on_logical_failure=lambda: receiver._fail_bounce_logically(key),
        )
        bounce_context, _descs = t._scatter_q.get_nowait()
        t._apply(bounce_context.rid_slice, lambda context: context.finish_scatter(False))

        snapshot = registry.context_snapshot(key)
        assert snapshot.logical_state is LogicalState.FAILED
        assert snapshot.physical_state is not PhysicalState.DRAINED
        assert task.status is tfr.TaskStatus.ERROR
        assert not tfr.RxSession.resources_drained(session)
        assert t.is_bounced(key)
        assert t._recv_alloc.released == []

    def test_backend_quiescence_callback_fires_after_slot_settlement(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])
        assert t.reserve(req, writer_ranks=[7])
        key = (req.unique_rid, req.slice_id)
        assert t.mark_writer_exposed(key, 7)
        t.mark_logical_failure(key)
        calls = []

        t.mark_backend_quiesced(key, on_done=lambda ok: calls.append(ok))

        assert calls == [False]
        assert not t.is_bounced(key)
        assert t._recv_alloc.released == [0]

    def test_close_refuses_live_receive_context(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])
        assert t.reserve(req, writer_ranks=[3])
        with pytest.raises(RuntimeError, match="live receive contexts"):
            t.close()
        assert not t.reserve(_recv_req([2], rid=2), writer_ranks=[4])

    def test_close_finishes_already_queued_scatter(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])
        assert t.reserve(req, writer_ranks=[3])
        key = (req.unique_rid, req.slice_id)
        t.record_result(
            key,
            3,
            np.array([10], dtype=np.int64),
            np.array([200], dtype=np.int64),
            src_base=0x100000,
        )

        def worker():
            while True:
                item = t._scatter_q.get()
                try:
                    if item is None:
                        return
                    ctx, _descs = item
                    t._apply(ctx.rid_slice, lambda c: c.finish_scatter(True))
                finally:
                    t._scatter_q.task_done()

        t._scatter_thread = threading.Thread(target=worker)
        t._scatter_thread.start()
        t.close()
        assert not t.is_bounced(key)
        assert t._recv_alloc.released == [0]

    def test_close_retries_settlement_created_while_draining_scatter(self, monkeypatch):
        t = _make_transport(monkeypatch, block_bytes_per_group=[100])
        req = _recv_req([2])
        assert t.reserve(req, writer_ranks=[3])
        key = (req.unique_rid, req.slice_id)
        callback_calls = []

        def acknowledge(ok):
            callback_calls.append(ok)
            if len(callback_calls) == 1:
                raise RuntimeError("registry temporarily unavailable")

        t.record_result(
            key,
            3,
            np.array([10], dtype=np.int64),
            np.array([200], dtype=np.int64),
            src_base=0x100000,
            on_done=acknowledge,
        )

        def worker():
            while True:
                item = t._scatter_q.get()
                try:
                    if item is None:
                        return
                    ctx, _descs = item
                    t._apply(ctx.rid_slice, lambda c: c.finish_scatter(True))
                finally:
                    t._scatter_q.task_done()

        t._scatter_thread = threading.Thread(target=worker)
        t._scatter_thread.start()
        t.close()

        assert callback_calls == [True, True]
        assert t._pending_settlements == {}
        assert t._recv_alloc.released == [0]

    def test_close_drains_scatter_queue_before_destroying_arenas(self):
        t = object.__new__(btr.VmmBounceTransport)
        t._init_recv_state()
        t._send_alloc = _FakeAlloc(1024, 512)
        t._recv_alloc = _FakeAlloc(1024, 512)
        t._reg_descs = [object()]
        t._agent = SimpleNamespace(
            deregister_memory=lambda d: pytest.fail("must not use a shut-down agent")
        )
        t._scatter_q = queue.Queue()
        processed = []

        def worker():
            while True:
                item = t._scatter_q.get()
                try:
                    if item is None:
                        return
                    processed.append(item)
                finally:
                    t._scatter_q.task_done()

        t._scatter_thread = threading.Thread(target=worker)
        t._scatter_thread.start()
        t._scatter_q.put("accepted-before-close")
        t.close()
        assert processed == ["accepted-before-close"]
        assert t._send_alloc.closed and t._recv_alloc.closed


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

    def test_close_atomically_rejects_racing_reservations(self, monkeypatch):
        close_started = threading.Event()
        allow_close = threading.Event()

        class BlockingBuffer:
            size = 512
            base_ptr = 0x1000

            @staticmethod
            def reg_descs():
                return []

            @staticmethod
            def close():
                close_started.set()
                assert allow_close.wait(timeout=2)

        monkeypatch.setattr(bbuf, "Buffer", lambda *a, **k: BlockingBuffer())
        allocator = btr.SlotAllocator(512, 512)
        close_thread = threading.Thread(target=allocator.close)
        close_thread.start()
        assert close_started.wait(timeout=1)

        try:
            assert allocator.reserve(512, timeout=0.1) is None
        finally:
            allow_close.set()
            close_thread.join(timeout=2)

        assert not close_thread.is_alive()

    def test_failed_buffer_close_keeps_admission_closed_and_is_retryable(self, monkeypatch):
        class FailOnceBuffer:
            size = 512
            base_ptr = 0x1000

            def __init__(self):
                self.close_calls = 0

            @staticmethod
            def reg_descs():
                return []

            def close(self):
                self.close_calls += 1
                if self.close_calls == 1:
                    raise RuntimeError("injected buffer close failure")

        buffer = FailOnceBuffer()
        monkeypatch.setattr(bbuf, "Buffer", lambda *a, **k: buffer)
        allocator = btr.SlotAllocator(512, 512)

        with pytest.raises(RuntimeError, match="injected buffer close failure"):
            allocator.close()
        assert allocator.reserve(512, timeout=0.1) is None

        allocator.close()
        assert buffer.close_calls == 2


# --------------------------------------------------------------------------- #
# core.RecvBounceContext — the pure drain-before-release state machine.
# No CUDA / NIXL / allocator, so these run on any CPU (no skipif).
# --------------------------------------------------------------------------- #
class TestLifecycle:
    def _ctx(
        self,
        writer_ranks=(3,),
        per_writer_bytes=100,
        base_addr=0x1000,
        destination_intervals=None,
    ):
        return bcore.RecvBounceContext(
            rid_slice=(1, 0),
            slot_id=0,
            base_addr=base_addr,
            per_writer_bytes=per_writer_bytes,
            writer_ranks=tuple(writer_ranks),
            destination_intervals=destination_intervals,
        )

    def _dst(self, v=10):
        return dict(
            dst_ptrs=np.array([v], dtype=np.int64),
            sizes=np.array([100], dtype=np.int64),
        )

    def test_writer_base_layout(self):
        c = self._ctx((7, 3, 11), per_writer_bytes=0x64, base_addr=0x1000)
        assert [c.writer_base(rank) for rank in (7, 3, 11)] == [0x1000, 0x1064, 0x10C8]
        with pytest.raises(KeyError):
            c.writer_base(9)

    def test_single_writer_success_scatters_then_releases(self):
        c = self._ctx()
        assert not c.ready_to_scatter() and not c.ready_to_settle()
        c.record_writer_result(3, succeeded=True, src_base=0x1000, **self._dst())
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
        c = self._ctx((7, 3))
        c.record_writer_result(7, succeeded=True, src_base=0x1000, **self._dst())
        assert not c.ready_to_scatter()  # 1/2 writers
        assert not c.ready_to_settle()  # drain-before-release
        c.record_writer_result(3, succeeded=True, src_base=0x1000 + 100, **self._dst(110))
        assert c.ready_to_scatter()  # all success -> scatter

    def test_fanin_failed_then_success_releases(self):
        c = self._ctx((7, 3))
        assert c.mark_writer_exposed(7)
        assert c.mark_writer_exposed(3)
        c.record_writer_result(7, succeeded=False)
        assert not c.ready_to_settle()  # a sibling is still pending -> hold
        c.record_writer_result(3, succeeded=True, src_base=0x1000 + 100, **self._dst())
        assert not c.ready_to_scatter()  # >=1 FAILED -> skip scatter
        assert c.ready_to_settle()
        ret = c.settle()
        assert ret.disposition is bcore.Disposition.RELEASE  # all drained -> free, not quarantine
        assert ret.success is False
        assert c.state is bcore.TransferState.FAILED

    def test_logical_failure_retains_exposed_writer_and_suppresses_scatter(self):
        c = self._ctx((7, 3))
        assert c.mark_writer_exposed(7)
        c.mark_logical_failure()  # rank 3 was never exposed, rank 7 remains in doubt
        assert c.pending_exposed_writers == (7,)
        assert c.state is bcore.TransferState.QUARANTINED
        assert not c.ready_to_settle()
        c.record_writer_result(7, succeeded=True, src_base=0x1000, **self._dst())
        assert not c.ready_to_scatter()
        ret = c.settle()
        assert ret.disposition is bcore.Disposition.RELEASE and ret.success is False

    def test_backend_quiescence_retires_in_doubt_writer(self):
        c = self._ctx((7, 3))
        assert c.mark_writer_exposed(7)
        c.mark_logical_failure()
        assert not c.ready_to_settle()
        c.mark_backend_quiesced()
        assert c.ready_to_settle()
        assert c.settle().success is False

    def test_failure_before_publication_releases_immediately(self):
        c = self._ctx((7, 3))
        c.mark_logical_failure()
        assert c.pending_exposed_writers == ()
        assert c.ready_to_settle()
        assert c.settle().success is False

    def test_protocol_conflict_before_publication_requires_backend_quiescence(self):
        c = self._ctx((7, 3))

        c.mark_protocol_conflict()

        assert c.pending_exposed_writers == ()
        assert not c.ready_to_settle()
        c.mark_backend_quiesced()
        assert c.ready_to_settle()
        assert c.settle().success is False

    def test_absent_tail_direct_fallback_succeeds_without_scatter(self):
        c = self._ctx()
        c.record_writer_result(3, succeeded=True)  # no dst tail
        assert not c.ready_to_scatter()
        assert c.ready_to_settle()
        settlement = c.settle()
        assert settlement.disposition is bcore.Disposition.RELEASE
        assert settlement.success is True

    @pytest.mark.parametrize("sizes", ([101], [0], [-1]))
    def test_invalid_scatter_extent_fails_closed(self, sizes):
        c = self._ctx(per_writer_bytes=100)
        c.record_writer_result(
            3,
            succeeded=True,
            src_base=0x1000,
            dst_ptrs=np.array([10], dtype=np.int64),
            sizes=np.array(sizes, dtype=np.int64),
        )
        assert not c.ready_to_scatter()
        assert c.ready_to_settle()
        assert c.settle().success is False

    def test_empty_scatter_tail_fails_closed(self):
        c = self._ctx()
        c.record_writer_result(
            3,
            succeeded=True,
            src_base=0x1000,
            dst_ptrs=np.array([], dtype=np.int64),
            sizes=np.array([], dtype=np.int64),
        )
        assert not c.ready_to_scatter()
        assert c.ready_to_settle()
        assert c.settle().success is False

    def test_overlapping_destinations_within_writer_fail_closed(self):
        c = self._ctx()
        c.record_writer_result(
            3,
            succeeded=True,
            src_base=0x1000,
            dst_ptrs=np.array([0x2000, 0x2030], dtype=np.int64),
            sizes=np.array([60, 40], dtype=np.int64),
        )
        assert not c.ready_to_scatter()
        assert c.ready_to_settle()
        assert c.settle().success is False

    def test_overlapping_destinations_across_writers_fail_closed(self):
        c = self._ctx((7, 3))
        c.record_writer_result(
            7,
            succeeded=True,
            src_base=0x1000,
            dst_ptrs=np.array([0x2000], dtype=np.int64),
            sizes=np.array([100], dtype=np.int64),
        )
        c.record_writer_result(
            3,
            succeeded=True,
            src_base=0x1000 + 100,
            dst_ptrs=np.array([0x2030], dtype=np.int64),
            sizes=np.array([100], dtype=np.int64),
        )
        assert c.logical_failed
        assert not c.ready_to_scatter()
        assert c.ready_to_settle()
        assert c.settle().success is False

    def test_scatter_destinations_must_stay_inside_trusted_intervals(self):
        c = self._ctx(
            per_writer_bytes=0x100,
            destination_intervals=[(0x2000, 0x100)],
        )
        c.record_writer_result(
            3,
            succeeded=True,
            src_base=0x1000,
            dst_ptrs=np.array([0x20F0], dtype=np.int64),
            sizes=np.array([0x100], dtype=np.int64),
        )
        assert not c.ready_to_scatter()
        assert c.ready_to_settle()
        assert c.settle().success is False

    def test_scatter_destinations_reject_truncated_in_range_intervals(self):
        c = self._ctx(
            per_writer_bytes=0x100,
            destination_intervals=[(0x2000, 0x100)],
        )
        c.record_writer_result(
            3,
            succeeded=True,
            src_base=0x1000,
            dst_ptrs=np.array([0x2000, 0x2080], dtype=np.int64),
            sizes=np.array([0x80, 0x20], dtype=np.int64),
        )
        assert not c.ready_to_scatter()
        assert c.ready_to_settle()
        assert c.settle().success is False

    def test_scatter_destinations_accept_exact_heterogeneous_fragment_shapes(self):
        c = self._ctx(
            per_writer_bytes=0x100,
            destination_intervals=[(0x2000, 0x100)],
        )
        c.record_writer_result(
            3,
            succeeded=True,
            src_base=0x1000,
            dst_ptrs=np.array([0x2000, 0x2040], dtype=np.int64),
            sizes=np.array([0x40, 0xC0], dtype=np.int64),
        )
        assert c.ready_to_scatter()

    @pytest.mark.parametrize(
        "destination_intervals",
        [[], [(0, 8)], [((1 << 64) - 1, 2)], [(0x2000, -1)]],
    )
    def test_invalid_trusted_destination_intervals_are_rejected(self, destination_intervals):
        with pytest.raises(ValueError, match="destination interval"):
            self._ctx(destination_intervals=destination_intervals)

    def test_unlaunched_scatter_can_be_suppressed_and_settled(self):
        c = self._ctx()
        c.record_writer_result(3, succeeded=True, src_base=0x1000, **self._dst())
        c.begin_scatter()
        c.suppress_scatter()
        assert c.ready_to_settle()
        assert c.settle().success is False

    def test_writers_locked_after_scatter_drops_late_writer(self):
        c = self._ctx()
        c.record_writer_result(3, succeeded=True, src_base=0x1000, **self._dst())
        c.begin_scatter()  # SCATTERING -> frozen
        c.record_writer_result(9, succeeded=False)  # a late / reordered report
        assert 9 not in c._writer_ok  # dropped, cannot re-arm the state

    def test_duplicate_writer_dedup(self):
        c = self._ctx((7, 3))
        c.record_writer_result(7, succeeded=True, src_base=0x1000, **self._dst())
        c.record_writer_result(7, succeeded=False)  # same rank again -> ignored
        assert c._writer_ok[7] is True
        assert not c.ready_to_settle()  # still only 1 distinct writer of 2

    def test_scatter_failure_retains_ownership_without_a_positive_fence(self):
        c = self._ctx()
        c.record_writer_result(3, succeeded=True, src_base=0x1000, **self._dst())
        c.begin_scatter()
        c.finish_scatter(False)  # scatter kernel failed
        assert not c.ready_to_settle()
        assert c.settle() is None
        assert c.scatter_state is bcore.ScatterState.FAILED
        assert c.logical_failed

    def test_logical_failure_during_scatter_waits_and_reports_failure(self):
        c = self._ctx()
        c.record_writer_result(3, succeeded=True, src_base=0x1000, **self._dst())
        c.begin_scatter()
        c.mark_logical_failure()
        assert not c.ready_to_settle()
        c.finish_scatter(True)
        assert c.settle().success is False

    def test_unexpected_rank_does_not_count(self):
        c = self._ctx((7, 3))
        assert not c.mark_writer_exposed(9)
        assert not c.record_writer_result(9, succeeded=False)
        assert c.pending_exposed_writers == ()

    def test_no_access_transition_can_complete_direct_success(self):
        c = self._ctx((7, 3))
        assert c.mark_writer_no_access(7, succeeded=True)
        c.record_writer_result(3, succeeded=True)
        assert c.ready_to_settle()
        assert c.settle().success is True


# --------------------------------------------------------------------------- #
# SlotAllocator quarantine — in-doubt regions are held until explicit quiescence evidence.
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
        a.quarantine(sb)  # hold the tail out of reuse indefinitely
        assert a.quarantined_bytes == 512
        # live [0,512) + quarantined [512,1024) => arena full => next reserve can't fit.
        assert a.reserve(512, timeout=0.05) is None

    def test_explicit_quiescence_release_returns_region_to_free(self, monkeypatch):
        a = self._alloc(monkeypatch, cap=512)  # single slot
        s, _ = a.reserve(512)
        a.quarantine(s)
        assert a.reserve(512, timeout=0.05) is None
        assert a.release_quarantined(s)
        assert a.quarantined_bytes == 0
        assert a.reserve(512, timeout=0.5) is not None  # now reusable

    def test_close_refuses_live_or_quarantined_slot(self, monkeypatch):
        a = self._alloc(monkeypatch, cap=512)
        s, _ = a.reserve(512)
        with pytest.raises(RuntimeError, match="live or quarantined"):
            a.close()
        a.quarantine(s)
        with pytest.raises(RuntimeError, match="live or quarantined"):
            a.close()
        assert a.release_quarantined(s)
        a.close()
