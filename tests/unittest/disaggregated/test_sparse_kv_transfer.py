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

"""Memory descriptors and logical completion for mixed host/device KV handoffs."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base import TokenRange
from tensorrt_llm._torch.disaggregation.base.agent import MemoryType
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.native import transfer as transfer_mod
from tensorrt_llm._torch.disaggregation.native.extent import Chunk
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
    KVRegionExtractorV1,
    build_page_table_from_manager,
)
from tensorrt_llm._torch.disaggregation.resource.page import (
    KVCachePageTable,
    MapperKind,
    PhysicalPool,
)
from tensorrt_llm._torch.disaggregation.resource.utils import get_unique_pool_memory_descs
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheTier

pytestmark = pytest.mark.cpu_only


def _manager(offload: bool) -> Mock:
    roles = ("compress", "compress_scale", "indexer")
    sizes = (288 * 32, 36 * 32, 144 * 32)
    layers = [
        SimpleNamespace(
            layer_id=layer,
            window_size=None,
            buffers=[
                SimpleNamespace(role=role, is_sparse=offload and role != "indexer")
                for role in roles
            ],
        )
        for layer in range(2)
    ]
    coalesced = [
        SimpleNamespace(
            single_buffer_size=size,
            buffer_ids=[SimpleNamespace(layer_id=layer, role=role) for layer in range(2)],
        )
        for role, size in zip(roles, sizes)
    ]
    impl = SimpleNamespace(
        init_config=SimpleNamespace(tokens_per_block=128, layers=layers),
        layer_grouping=((0, 1),),
        pool_group_descs=[
            SimpleNamespace(
                pool_group_index=0,
                num_slots=4,
                pools=[
                    SimpleNamespace(
                        base_address=0x100000 * (i + 1),
                        slot_bytes=2 * size,
                        # Offloaded sparse buffers are host memory; the indexer stays on the GPU.
                        storage_medium=(
                            CacheTier.HOST_MEM
                            if offload and role != "indexer"
                            else CacheTier.GPU_MEM
                        ),
                    )
                    for i, (role, size) in enumerate(zip(roles, sizes))
                ],
                slot_desc=SimpleNamespace(
                    variants=[SimpleNamespace(layer_group_id=0, coalesced_buffers=coalesced)]
                ),
            )
        ],
    )
    # The page-table builder accepts only a KVCacheManagerV2.
    return Mock(
        spec=KVCacheManagerV2,
        get_disagg_pool_view_partition=None,
        get_disagg_ignored_roles=lambda: frozenset(),
        impl=impl,
        pp_layers=[0, 1],
        num_kv_heads_per_layer=[1, 1],
        get_disagg_role_mapper_kinds=lambda: {Role.ALL: MapperKind.REPLICATED},
        get_disagg_role_layouts=lambda: {},
    )


def _rank_info(name: str, page_table: KVCachePageTable) -> RankInfo:
    return RankInfo(
        instance_name=name,
        instance_rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[
            len(
                {
                    layer.global_layer_id
                    for group in page_table.layer_groups
                    for layer in group.local_layers
                }
            )
        ],
        sender_endpoints=["sender"],
        self_endpoint="receiver",
        transfer_engine_info=b"",
        device_id=3,
        attention=AttentionInfo(1, page_table.tokens_per_block, 512, 1, False, True),
        page_table=page_table,
    )


def _sender_and_meta(
    src: KVCachePageTable,
    dst: KVCachePageTable,
    src_blocks: list[np.ndarray] | None = None,
    dst_blocks: list[np.ndarray] | None = None,
    device_id: int = 3,
    prompt_len: int | None = None,
) -> tuple[transfer_mod.Sender, transfer_mod.WriteMeta]:
    sender = object.__new__(transfer_mod.Sender)
    source = _rank_info("ctx", src)
    source.device_id = device_id
    # Exercise the metadata exchange, including memory types.
    destination = RankInfo.from_bytes(_rank_info("gen", dst).to_bytes())
    destination.device_id = device_id
    sender._registrar = PeerRegistrar(source, KVRegionExtractorV1(src))
    sender._registrar.register("gen", 0, destination)
    src_blocks = src_blocks if src_blocks is not None else [np.array([1, 3], dtype=np.int64)]
    dst_blocks = dst_blocks if dst_blocks is not None else [np.array([2, 0], dtype=np.int64)]
    if prompt_len is None:
        prompt_len = len(src_blocks[0]) * src.tokens_per_block
    chunk = Chunk(
        src_blocks, [group.kind for group in src.layer_groups], TokenRange(0, prompt_len), True
    )
    task = transfer_mod.KVSendTask(chunk, DisaggregatedParams(disagg_request_id=401), 0, prompt_len)
    task._unique_rid = 401
    req_info = transfer_mod.RecvReqInfo(1, "gen", 0, dst_blocks, 401, bounce_dst_base=0x900000)
    meta = sender._build_kv_write_meta(task, req_info)
    sender._enforce_physical_ownership = True
    sender._ownership_poisoned = None
    sender._ownership_poison_lock = threading.Lock()
    return sender, meta


@pytest.mark.parametrize("memory_type", ["VRAM", "DRAM"])
def test_pool_memory_type_roundtrip(memory_type: str) -> None:
    pool = PhysicalPool(0x1000, 64, 4, memory_type=memory_type)
    assert PhysicalPool.from_dict(pool.to_dict()) == pool
    old_payload = pool.to_dict()
    del old_payload["memory_type"]
    assert PhysicalPool.from_dict(old_payload).memory_type == "VRAM"
    with pytest.raises(ValueError, match="memory type"):
        PhysicalPool(0x1000, 64, 4, memory_type="invalid")


def test_v2_extracts_host_data_and_scales_beside_device_indexer() -> None:
    table = build_page_table_from_manager(_manager(True))
    assert [p.memory_type for p in table.pool_groups[0].pools] == ["DRAM", "DRAM", "VRAM"]
    assert table.has_host_pools
    assert not build_page_table_from_manager(_manager(False)).has_host_pools
    # Each pool's memory type is its native storage medium, not the layer config's.
    manager = _manager(True)
    manager.impl.init_config.layers[1].buffers[0].is_sparse = False
    table = build_page_table_from_manager(manager)
    assert [p.memory_type for p in table.pool_groups[0].pools] == ["DRAM", "DRAM", "VRAM"]


def test_registration_separates_memory_types_and_device_ids() -> None:
    table = build_page_table_from_manager(_manager(True))
    worker = SimpleNamespace(_rank_info=_rank_info("gen", table), _agent=Mock(), _registered_mem=[])
    transfer_mod.TransferWorker._register_kv_cache(worker)
    assert [d.type for d in worker._registered_mem] == ["VRAM", "DRAM"]
    for desc in worker._registered_mem:
        assert desc.descs == get_unique_pool_memory_descs(table, 3, desc.type)
        assert {entry[2] for entry in desc.descs} == ({0} if desc.type == "DRAM" else {3})
    assert sum(len(d.descs) for d in worker._registered_mem) == 3
    assert worker._agent.register_memory.call_count == 2


@pytest.mark.parametrize("src_offload,dst_offload", [(False, False), (False, True), (True, False)])
def test_receiver_bypasses_device_bounce_for_host_pools(
    src_offload: bool, dst_offload: bool
) -> None:
    receiver = object.__new__(transfer_mod.Receiver)
    src = _rank_info("ctx", build_page_table_from_manager(_manager(src_offload)))
    dst = _rank_info("gen", build_page_table_from_manager(_manager(dst_offload)))
    receiver._registrar = SimpleNamespace(
        self_rank_info=dst,
        self_extractor=KVRegionExtractorV1(dst.page_table),
        get_peer_overlap=lambda *_: SimpleNamespace(ranks=[0]),
    )
    receiver._enforce_physical_ownership = False
    receiver._get_sender_info = Mock(return_value=src)
    receiver._bounce = Mock(reserve=Mock(return_value=False))
    receiver._get_session = Mock(return_value=Mock(_sender_endpoints=set()))
    receiver._request_sender_data = Mock()
    task = SimpleNamespace(
        _params=DisaggregatedParams(ctx_dp_rank=0),
        _unique_rid=401,
        _perf_timer=None,
        slice_id=0,
    )
    req_info = transfer_mod.RecvReqInfo(1, "gen", 0, [np.array([0])], 401)
    receiver._build_recv_req_info = Mock(return_value=req_info)
    receiver.dispatch_task(task)
    assert receiver._bounce.reserve.call_count == int(not src_offload and not dst_offload)
    receiver._request_sender_data.assert_called_once()


@pytest.mark.parametrize("offload", [False, True])
def test_transfer_groups_keep_the_same_byte_ranges(offload: bool) -> None:
    src = build_page_table_from_manager(_manager(False))
    dst = build_page_table_from_manager(_manager(offload))
    sender, meta = _sender_and_meta(src, dst)
    assert meta.expected_transfers == 1
    assert int(meta.sizes.sum()) == 2 * sum(p.slot_bytes for p in src.pool_groups[0].pools)
    assert {(g.src_memory_type, g.dst_memory_type) for g in meta.transfer_groups} == (
        {("VRAM", "DRAM"), ("VRAM", "VRAM")} if offload else {("VRAM", "VRAM")}
    )
    assert meta.bounce_dst_base == (None if offload else 0x900000)
    request = sender._make_agent_request(meta, 3)
    requests = request.requests if offload else (request,)
    assert len(requests) == (2 if offload else 1)
    assert {r.src_descs.type for r in requests} == {MemoryType.VRAM}
    assert {r.dst_descs.type for r in requests} == (
        {MemoryType.DRAM, MemoryType.VRAM} if offload else {MemoryType.VRAM}
    )


@pytest.mark.parametrize("outcome", ["success", "wait_failure", "submission_failure"])
def test_grouped_submission_retains_every_handle_until_settled(outcome: str) -> None:
    sender, meta = _sender_and_meta(
        build_page_table_from_manager(_manager(False)),
        build_page_table_from_manager(_manager(True)),
    )
    request = sender._make_agent_request(meta, 3)
    first = Mock(wait=Mock(return_value=True), is_completed=Mock(return_value=True))
    second = Mock(
        wait=Mock(return_value=outcome == "success"),
        is_completed=Mock(return_value=outcome == "success"),
    )
    first.last_status_str.return_value = "DONE"
    second.last_status_str.return_value = "ERROR"
    sender._agent = Mock()
    sender._agent.submit_transfer_requests.side_effect = [
        first,
        RuntimeError("second submission failed") if outcome == "submission_failure" else second,
    ]
    task = meta.task
    assert task.begin_physical_operation(0)
    completed, _ = sender._submit_transfer(task, 0, request)
    assert completed == (outcome == "success")
    if completed:
        assert task.resources_drained
        first.wait.assert_called_once()
        second.wait.assert_called_once()
    else:
        operation = task._physical_operations[0]
        assert operation.request is request
        assert operation.status.statuses[0] is first
        assert not task.poll_in_doubt_physical_operation(0)
        assert not task.resources_drained
        second.is_completed.return_value = True
        assert task.poll_in_doubt_physical_operation(0) == (outcome == "wait_failure")


@pytest.mark.parametrize("enforce_ownership", [False, True])
def test_one_logical_completion_after_all_grouped_writes(enforce_ownership: bool) -> None:
    sender, meta = _sender_and_meta(
        build_page_table_from_manager(_manager(False)),
        build_page_table_from_manager(_manager(True)),
    )
    task = meta.task
    sender._enforce_physical_ownership = enforce_ownership
    sender._sessions_lock = threading.Lock()
    session = SimpleNamespace(
        kv_tasks=[task], lock=threading.Lock(), status=SessionStatus.READY, set_exception=Mock()
    )
    sender._sessions = {401: session}
    sender._device_id = 3
    sender._instance_rank = 0
    sender._bounce = Mock()
    dealer = Mock()
    sender._get_result_dealer = Mock(return_value=dealer)

    def wait() -> bool:
        assert task.transferred_count == 0
        dealer.send.assert_not_called()
        if enforce_ownership:
            assert not task.resources_drained
        return True

    sender._agent = Mock()
    sender._agent.submit_transfer_requests.side_effect = [Mock(wait=wait), Mock(wait=wait)]
    if enforce_ownership:
        assert task.begin_physical_operation(0)
    sender._deliver_kv_to_agent(meta)
    assert task.transferred_count == 1
    assert task.resources_drained
    assert task.is_done
    dealer.send.assert_called_once()
    sender._bounce.build_request.assert_not_called()


@pytest.mark.parametrize("first_completes", [False, True])
def test_legacy_grouped_submission_waits_before_submitting_next(first_completes: bool) -> None:
    sender, meta = _sender_and_meta(
        build_page_table_from_manager(_manager(False)),
        build_page_table_from_manager(_manager(True)),
    )
    sender._enforce_physical_ownership = False
    first = Mock(wait=Mock(return_value=first_completes))
    first.last_status_str.return_value = "ERROR"

    def submit(request):
        if sender._agent.submit_transfer_requests.call_count == 1:
            return first
        first.wait.assert_called_once()
        raise RuntimeError("second submission failed")

    sender._agent = Mock(submit_transfer_requests=Mock(side_effect=submit))
    request = sender._make_agent_request(meta, 3)
    if first_completes:
        with pytest.raises(RuntimeError, match="second submission failed"):
            sender._submit_transfer(meta.task, 0, request)
    else:
        assert sender._submit_transfer(meta.task, 0, request) == (False, "ERROR")
        assert sender._agent.submit_transfer_requests.call_count == 1
