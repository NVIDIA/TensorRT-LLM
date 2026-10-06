# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixed-memory transfer contracts and tier-aware sparse KV transfer guards."""

import ctypes
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import msgpack
import numpy as np
import pytest

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation.base import Chunk, TokenRange
from tensorrt_llm._torch.disaggregation.base.agent import MemoryType
from tensorrt_llm._torch.disaggregation.base.region import SpecRegionPair
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.transfer import (
    RecvReqInfo,
    Sender,
    SendTaskBase,
    WriteMeta,
    _TransferBatch,
)
from tensorrt_llm._torch.disaggregation.resource import cache_reuse
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    AttentionLayerGroup,
    CacheKind,
    KVCachePageTable,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)
from tensorrt_llm._torch.disaggregation.resource.utils import get_unique_pool_memory_descs
from tensorrt_llm.runtime.kv_cache_manager_v2 import CACHE_LEVEL1, GPU_LEVEL

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("memory_type", ["VRAM", "DRAM"])
def test_pool_memory_type_roundtrip(memory_type):
    pool = PhysicalPool(0x1000, 64, 4, memory_type=memory_type)
    assert PhysicalPool.from_dict(pool.to_dict()).memory_type == memory_type
    legacy = pool.to_dict()
    del legacy["memory_type"]
    assert PhysicalPool.from_dict(legacy).memory_type == "VRAM"


def test_invalid_pool_memory_type():
    with pytest.raises(ValueError, match="VRAM or DRAM"):
        PhysicalPool(0x1000, 64, 4, memory_type="FILE")


def test_registration_separates_host_and_gpu_pools():
    entries = np.array([(0, 0, 64)], dtype=BUFFER_ENTRY_DTYPE)
    table = KVCachePageTable(
        tokens_per_block=8,
        layer_groups=[
            AttentionLayerGroup(
                pool_group_idx=0,
                pool_views=[PoolView(i, buffer_entries=entries) for i in (0, 1, 1)],
            )
        ],
        pool_groups=[
            PhysicalPoolGroup(
                [PhysicalPool(0x1000, 64, 4), PhysicalPool(0x2000, 128, 4, memory_type="DRAM")]
            )
        ],
    )
    assert get_unique_pool_memory_descs(table, 3) == [(0x1000, 256, 3, "kv_cache_memory_pool0")]
    assert get_unique_pool_memory_descs(table, 3, "DRAM") == [
        (0x2000, 512, 0, "kv_cache_memory_pool0")
    ]


def _host_page_table():
    return KVCachePageTable(
        tokens_per_block=8,
        layer_groups=[
            AttentionLayerGroup(
                pool_group_idx=0,
                pool_views=[
                    PoolView(0, buffer_entries=np.array([(0, 0, 64)], dtype=BUFFER_ENTRY_DTYPE))
                ],
            )
        ],
        pool_groups=[PhysicalPoolGroup([PhysicalPool(0x1000, 64, 8, memory_type="DRAM")])],
    )


def test_host_pool_wire_protocol_fails_closed_for_legacy_peers():
    rank = RankInfo("gen", 0, 1, 0, 1, 0, [1], [], "", b"", page_table=_host_page_table())
    wire = msgpack.unpackb(rank.to_bytes())
    assert wire["kv_memory_protocol"] == 1
    restored = RankInfo.from_bytes(rank.to_bytes())
    assert restored.page_table.pool_groups[0].pools[0].memory_type == "DRAM"
    del wire["kv_memory_protocol"]
    with pytest.raises(ValueError, match="memory protocol"):
        RankInfo.from_bytes(msgpack.packb(wire))


@pytest.mark.parametrize("slots", [[7, 1, 5], [-1, 1, 5], [8]])
def test_host_extraction_preserves_order_and_rejects_holes(slots):
    extractor = KVRegionExtractorV1(_host_page_table())
    if min(slots) < 0 or max(slots) >= 8:
        with pytest.raises(ValueError, match="valid slots"):
            extractor.extract(np.array(slots, dtype=np.int64))
    else:
        region = extractor.extract(np.array(slots, dtype=np.int64))
        np.testing.assert_array_equal(region.memory.ptrs, [0x11C0, 0x1040, 0x1140])
        assert region.memory.bytes_per_region == 64


def _write_meta(segments):
    return WriteMeta(
        task=Mock(),
        expected_transfers=1,
        peer_name="gen0",
        peer_rank=0,
        peer_endpoint="tcp://peer",
        unique_rid=42,
        src_ptrs=np.array([100, 200, 300], dtype=np.int64),
        dst_ptrs=np.array([1000, 2000, 3000], dtype=np.int64),
        sizes=np.array([8, 16, 24], dtype=np.int64),
        dst_device_id=3,
        memory_segments=segments,
    )


@pytest.mark.parametrize("src_type", ["VRAM", "DRAM"])
@pytest.mark.parametrize("dst_type", ["VRAM", "DRAM"])
def test_request_uses_independent_endpoint_memory_types(src_type, dst_type):
    meta = _write_meta(((3, src_type, dst_type),))
    request = Sender._make_agent_request(meta, 2)
    assert request.src_descs.type == getattr(MemoryType, src_type)
    assert request.dst_descs.type == getattr(MemoryType, dst_type)
    assert [tuple(desc) for desc in request.src_descs.descs] == [
        (100, 8, 2 if src_type == "VRAM" else 0),
        (200, 16, 2 if src_type == "VRAM" else 0),
        (300, 24, 2 if src_type == "VRAM" else 0),
    ]
    assert [tuple(desc) for desc in request.dst_descs.descs] == [
        (1000, 8, 3 if dst_type == "VRAM" else 0),
        (2000, 16, 3 if dst_type == "VRAM" else 0),
        (3000, 24, 3 if dst_type == "VRAM" else 0),
    ]


def test_mixed_request_preserves_every_fragment():
    batch = Sender._make_agent_request(
        _write_meta(((1, "VRAM", "DRAM"), (1, "VRAM", "VRAM"), (1, "VRAM", "DRAM"))),
        2,
    )
    assert isinstance(batch, _TransferBatch)
    assert len(batch.requests) == 2
    assert [tuple(desc) for desc in batch.requests[0].dst_descs.descs] == [
        (1000, 8, 0),
        (3000, 24, 0),
    ]
    assert [tuple(desc) for desc in batch.requests[1].dst_descs.descs] == [(2000, 16, 3)]


@pytest.mark.parametrize("segments", [((2, "VRAM", "DRAM"),), ((3, "FILE", "DRAM"),)])
def test_invalid_memory_segments_fail_before_submission(segments):
    with pytest.raises(ValueError):
        Sender._make_agent_request(_write_meta(segments), 0)


@pytest.mark.parametrize("failure", [None, "submit", "wait"])
def test_mixed_transfer_keeps_ownership_until_all_backends_finish(failure):
    sender = object.__new__(Sender)
    sender._enforce_physical_ownership = True
    sender._ownership_poison_lock = threading.Lock()
    sender._ownership_poisoned = None
    task = SendTaskBase(DisaggregatedParams(disagg_request_id=42))
    assert task.begin_physical_operation(0)
    batch = Sender._make_agent_request(_write_meta(((1, "VRAM", "DRAM"), (2, "VRAM", "VRAM"))), 2)

    def wait_first():
        assert not task.resources_drained
        return True

    first = SimpleNamespace(wait=Mock(side_effect=wait_first))
    second = SimpleNamespace(wait=Mock(return_value=failure != "wait"))
    sender._agent = SimpleNamespace(
        submit_transfer_requests=Mock(
            side_effect=[
                first,
                RuntimeError("submission failed") if failure == "submit" else second,
            ]
        )
    )
    completed, _ = sender._submit_transfer(task, 0, batch)
    assert completed is (failure is None)
    assert task.resources_drained is (failure is None)
    if failure is not None:
        assert sender._ownership_poisoned is not None
        assert task._physical_operations[0].request is batch
        assert batch.statuses[0] is first
    else:
        first.wait.assert_called_once()
        second.wait.assert_called_once()


@pytest.mark.parametrize("unsupported", [None, "bounce", "unowned"])
def test_native_host_transfer_copies_every_prompt_page(unsupported):
    """Exercise native planning/submission with a CPU byte-copy transport double."""
    src = np.arange(8 * 64, dtype=np.uint8).reshape(8, 64)
    dst = np.zeros_like(src)
    src_table, dst_table = _host_page_table(), _host_page_table()
    src_table.pool_groups[0].pools[0].base_address = src.ctypes.data
    dst_table.pool_groups[0].pools[0].base_address = dst.ctypes.data
    src_extractor, dst_extractor = KVRegionExtractorV1(src_table), KVRegionExtractorV1(dst_table)
    peer = SimpleNamespace(
        instance_name="gen",
        instance_rank=0,
        device_id=0,
        cp_size=1,
        dp_rank=0,
        self_endpoint="tcp://gen",
    )
    sender = object.__new__(Sender)
    sender._enforce_physical_ownership = unsupported != "unowned"
    sender._ownership_poison_lock = threading.Lock()
    sender._ownership_poisoned = None
    sender._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(cp_size=1),
        self_extractor=src_extractor,
        get_peer_rank_info=lambda *_: peer,
        get_peer_overlap=lambda *_: SimpleNamespace(ranks=[0]),
        peer_extractor=lambda *_: dst_extractor,
        get_pool_mapping=lambda *_: {(0, 0): (0, 0)},
        should_send_pool=lambda *_: True,
        get_kv_map=lambda *_: SimpleNamespace(map=lambda a, b: SpecRegionPair(a, b)),
    )
    task = SendTaskBase(DisaggregatedParams(disagg_request_id=42))
    task._chunk = Chunk(
        block_ids_per_layer_groups=[np.array([7, 1, 5])],
        kind_per_layer_group=[CacheKind.PAGED],
        token_range=TokenRange(0, 19),
        is_last=True,
    )
    task._prompt_len = 19
    task.slice_id = 0
    info = RecvReqInfo(
        sender_req_id=42,
        instance_name="gen",
        instance_rank=0,
        block_ids_per_layer_groups=[np.array([2, 4, 0])],
        unique_rid=42,
        bounce_dst_base=0x4000 if unsupported == "bounce" else None,
    )
    if unsupported:
        with pytest.raises(ValueError, match="bounce|ownership"):
            sender._build_kv_write_meta(task, info)
        return
    meta = sender._build_kv_write_meta(task, info)
    assert meta.memory_segments == ((3, "DRAM", "DRAM"),)
    assert meta.sizes.sum() == 3 * 64

    def submit(request):
        assert request.src_descs.type == request.dst_descs.type == MemoryType.DRAM
        for source, destination in zip(request.src_descs.descs, request.dst_descs.descs):
            assert source[1] == destination[1]
            ctypes.memmove(destination[0], source[0], source[1])
        return SimpleNamespace(wait=lambda: True)

    sender._agent = SimpleNamespace(submit_transfer_requests=submit)
    assert task.begin_physical_operation(0)
    assert sender._submit_transfer(task, 0, Sender._make_agent_request(meta, 0)) == (True, None)
    np.testing.assert_array_equal(dst[[2, 4, 0]], src[[7, 1, 5]])
    assert not dst[[1, 3, 5, 6, 7]].any()


@pytest.mark.parametrize("method", ["get_block_ids", "get_block_ordinals"])
@pytest.mark.parametrize("levels", [[GPU_LEVEL, GPU_LEVEL, None], [CACHE_LEVEL1, GPU_LEVEL, None]])
def test_gpu_adapter_checks_tiers_even_when_host_slots_are_positive(method, levels):
    snapshot = SimpleNamespace(base_page_indices=[7, 1, -1], cache_levels=levels)
    cache = SimpleNamespace(is_active=True, get_page_storage_snapshot=Mock(return_value=snapshot))
    adapter = cache_reuse._CacheReuseAdapterV2(SimpleNamespace(kv_cache_map={42: cache}))
    req = SimpleNamespace(py_request_id=42)
    if levels[0] == CACHE_LEVEL1:
        with pytest.raises(RuntimeError, match="locked GPU pages"):
            getattr(adapter, method)(req, 0, None)
    else:
        expected = [7, 1] if method == "get_block_ids" else [7, 1, -1]
        np.testing.assert_array_equal(getattr(adapter, method)(req, 0, None), expected)


def test_gpu_adapter_rejects_inactive_request():
    cache = SimpleNamespace(is_active=False)
    adapter = cache_reuse._CacheReuseAdapterV2(SimpleNamespace(kv_cache_map={42: cache}))
    with pytest.raises(RuntimeError, match="active request"):
        adapter.get_block_ids(SimpleNamespace(py_request_id=42), 0, None)


def test_mixed_transfer_waits_for_every_status_after_failure():
    batch = Sender._make_agent_request(_write_meta(((1, "VRAM", "DRAM"), (2, "VRAM", "VRAM"))), 2)
    first = SimpleNamespace(wait=Mock(return_value=False))
    second = SimpleNamespace(wait=Mock(return_value=True))
    batch.statuses = [first, second]
    assert not batch.wait()
    first.wait.assert_called_once()
    second.wait.assert_called_once()


def test_mixed_transfer_late_completion_requires_every_submitted_handle():
    batch = Sender._make_agent_request(_write_meta(((1, "VRAM", "DRAM"), (2, "VRAM", "VRAM"))), 2)
    first = SimpleNamespace(is_completed=Mock(return_value=True))
    second = SimpleNamespace(is_completed=Mock(return_value=False))
    batch.statuses = [first]
    assert not batch.is_completed()
    batch.statuses.append(second)
    assert not batch.is_completed()
    second.is_completed.return_value = True
    assert batch.is_completed()


def test_mixed_transfer_late_settlement_releases_task_only_after_all_handles_finish():
    sender = object.__new__(Sender)
    sender._enforce_physical_ownership = True
    sender._ownership_poison_lock = threading.Lock()
    sender._ownership_poisoned = None
    task = SendTaskBase(DisaggregatedParams(disagg_request_id=42))
    assert task.begin_physical_operation(0)
    batch = Sender._make_agent_request(_write_meta(((1, "VRAM", "DRAM"), (2, "VRAM", "VRAM"))), 2)
    first = SimpleNamespace(wait=lambda: True, is_completed=lambda: True)
    second = SimpleNamespace(wait=lambda: False, is_completed=Mock(return_value=False))
    sender._agent = SimpleNamespace(submit_transfer_requests=Mock(side_effect=[first, second]))
    assert sender._submit_transfer(task, 0, batch)[0] is False
    assert not task.poll_in_doubt_physical_operation(0)
    assert not task.resources_drained
    second.is_completed.return_value = True
    assert task.poll_in_doubt_physical_operation(0)
    assert task.resources_drained
