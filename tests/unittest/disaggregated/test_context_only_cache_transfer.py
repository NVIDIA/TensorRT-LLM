# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transfer real CSA2 caches with different physical Context/Generation layouts."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
    CSA2CacheManager,
    CSA2CacheRole,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.transfer import RecvReqInfo, Sender
from tensorrt_llm._torch.disaggregation.resource.cache_reuse import create_cache_reuse_adapter
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.disaggregation.resource.utils import get_pool_view_global_layer_ids
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping


def _manager(context_only: bool, dp_rank: int | None = None) -> CSA2CacheManager:
    return CSA2CacheManager(
        KvCacheConfig(max_gpu_total_bytes=512 << 20, enable_block_reuse=False, dtype="fp8"),
        CacheType.SELFKONLY,
        layout=CSA2Layout(
            (0, 0) + (2,) * 18 + (1,) * 20,
            (2, 8, 14, 20),
            (2, 8, 14, 20),
            candidate_source_layer_id=20,
        ),
        context_swa_layer_limit=20 if context_only else None,
        bounded_replay_on_generation=True,
        num_layers=40,
        tokens_per_block=128,
        mapping=(
            Mapping(world_size=4, rank=dp_rank, tp_size=4, enable_attention_dp=True)
            if dp_rank is not None
            else Mapping()
        ),
        max_seq_len=8320,
        max_batch_size=1,
        max_input_len=8192,
        max_num_tokens=8192,
        dtype=DataType.BF16,
        vocab_size=1024,
    )


def _request(request_id: int, length: int, source: bool) -> LlmRequest:
    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=[i % 1024 for i in range(length)],
        sampling_config=SamplingConfig(),
        is_streaming=False,
        llm_request_type=(
            LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY
            if source
            else LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
        ),
    )
    request.py_csa2_remote_tail_mode = "source" if source else "destination"
    request.py_csa2_remote_tail_start = max(0, length - 128)
    request.py_csa2_remote_tail_split = 20
    return request


def _slice(manager: CSA2CacheManager, request: LlmRequest):
    transceiver = KvCacheTransceiverV2.__new__(KvCacheTransceiverV2)
    transceiver._kv_cache_manager = manager
    transceiver._page_table = KVRegionExtractorV1(manager).page_table
    transceiver._reuse_adapter = create_cache_reuse_adapter(manager)
    return transceiver._create_chunk(request)


def _bytes(pointer: int, count: int) -> torch.Tensor:
    return convert_to_torch_tensor(TensorWrapper(int(pointer), DataType.UINT8, [int(count)]))


def _physical_pages(manager: CSA2CacheManager, request: LlmRequest):
    """Independent model-role oracle; do not reuse transport's pool mapping."""
    for layer, role in manager._physical_roles.values():
        buffers = manager.get_buffers(layer, role)
        pages = manager.get_cache_indices(request.py_request_id, layer, role)
        yield layer, role, buffers, pages


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("length", [128, 129, 1024, 8192])
@pytest.mark.parametrize(
    "source_dp_rank,target_dp_rank",
    [pytest.param(None, None, id="tp1")]
    + [
        pytest.param(source, target, id=f"dp4-{source}-to-{target}")
        for source in range(4)
        for target in range(4)
    ],
)
@torch.inference_mode()
def test_context_only_physical_cache_transfer_preserves_logical_owners(
    length: int,
    source_dp_rank: int | None,
    target_dp_rank: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    source, target = _manager(True, source_dp_rank), _manager(False, target_dp_rank)
    source_request, target_request = _request(71, length, True), _request(72, length, False)
    prefix = source_request.py_csa2_remote_tail_start
    try:
        assert source.prepare_context(source_request)
        # A zero-prefix request computes the full prompt but transfers no KV.
        assert source.resize_context(source_request, prefix or length)
        assert target.prepare_disagg_gen_init(target_request)
        source._stream.synchronize()
        target._stream.synchronize()
        for layer, role, buffers, pages in _physical_pages(source, source_request):
            for logical, page in enumerate(pages):
                if page >= 0:
                    buffers[page].view(torch.uint8).fill_(
                        1
                        + (
                            (source_dp_rank or 0) * 53
                            + layer * 17
                            + list(CSA2CacheRole).index(role) * 7
                            + logical
                        )
                        % 250
                    )
        for _, _, buffers, pages in _physical_pages(target, target_request):
            for page in set(pages):
                if page >= 0:
                    buffers[page].view(torch.uint8).zero_()

        assert (
            source._layer_roles[20, CSA2CacheRole.GLOBAL]
            != target._layer_roles[20, CSA2CacheRole.GLOBAL]
        )
        # Exercise the real wire serialization, including attention-DP routing.
        source_info = RankInfo.from_bytes(
            RankInfo.from_kv_cache_manager(
                "context", source, torch.cuda.current_device()
            ).to_bytes()
        )
        target_info = RankInfo.from_bytes(
            RankInfo.from_kv_cache_manager(
                "generation", target, torch.cuda.current_device()
            ).to_bytes()
        )
        registrar = PeerRegistrar(source_info, KVRegionExtractorV1(source))
        for info, dp_rank in ((source_info, source_dp_rank), (target_info, target_dp_rank)):
            assert info.instance_rank == info.tp_rank == info.dp_rank == (dp_rank or 0)
            assert info.tp_size == info.dp_size == (4 if dp_rank is not None else 1)
            assert info.tp_size_per_dp_group == 1
            assert info.attention.enable_attention_dp == (dp_rank is not None)
        registrar.register("generation", target_info.instance_rank, target_info)
        assert registrar.get_peer_overlap(target_info, target_info.dp_rank).ranks == [
            target_info.instance_rank
        ]
        receiver_registrar = PeerRegistrar(target_info, KVRegionExtractorV1(target))
        assert receiver_registrar.get_peer_overlap(source_info, source_info.dp_rank).ranks == [
            source_info.instance_rank
        ]
        pool_mapping = registrar.get_pool_mapping(target_info)
        matched_global_ids = set()
        for (src_group, src_view), (dst_group, dst_view) in pool_mapping.items():
            source_group = source_info.page_table.layer_groups[src_group]
            target_group = target_info.page_table.layer_groups[dst_group]
            assert source_group.pool_views[src_view].mapper_kind == MapperKind.REPLICATED
            assert target_group.pool_views[dst_view].mapper_kind == MapperKind.REPLICATED
            source_ids = set(
                get_pool_view_global_layer_ids(source_group.pool_views[src_view], source_group)
            )
            target_ids = set(
                get_pool_view_global_layer_ids(target_group.pool_views[dst_view], target_group)
            )
            assert source_ids == target_ids
            matched_global_ids.update(source_ids)
        expected_global_ids = {
            layer * 3 + kind.value for layer, kind in source._layer_attn_to_layer_id
        }
        assert matched_global_ids == expected_global_ids
        assert len(matched_global_ids) == 20 + 4 + 3
        assert (20, CSA2CacheRole.GLOBAL) in source._physical_roles.values()
        assert (20, CSA2CacheRole.INDEX) in source._physical_roles.values()
        assert all(
            layer < 20 or role != CSA2CacheRole.SWA
            for layer, role in source._physical_roles.values()
        )

        source_slice, target_slice = _slice(source, source_request), _slice(target, target_request)
        assert source_slice.excluded_pool_views == set()
        assert len(target_slice.excluded_pool_views) == 20
        sender = Sender.__new__(Sender)
        sender._registrar = registrar
        task = SimpleNamespace(
            _chunk=source_slice,
            _prompt_len=length,
            _perf_timer=None,
            _unique_rid=991,
            slice_id=0,
        )
        receiver = RecvReqInfo(
            sender_req_id=71,
            instance_name="generation",
            instance_rank=target_info.instance_rank,
            block_ids_per_layer_groups=target_slice.block_ids_per_layer_groups,
            unique_rid=991,
        )
        write = sender._build_kv_write_meta(task, receiver)
        assert (write.sizes.size > 0) == (prefix > 0)
        assert write.peer_rank == target_info.instance_rank
        assert write.expected_transfers == 1
        # The production sender builds these exact fragments. Local GPU copies
        # execute the descriptor list without relying on a NIXL installation.
        for src, dst, count in zip(write.src_ptrs, write.dst_ptrs, write.sizes):
            _bytes(dst, count).copy_(_bytes(src, count))
        torch.cuda.synchronize()

        compared = set()
        expected_bytes = 0
        for layer, role, buffers, source_pages in _physical_pages(source, source_request):
            target_buffers = target.get_buffers(layer, role)
            target_pages = target.get_cache_indices(72, layer, role)
            window = None
            if role == CSA2CacheRole.SWA:
                window = source._swa_retention(layer)
            elif role not in (CSA2CacheRole.GLOBAL, CSA2CacheRole.INDEX):
                window = source._window(2)
            first = 0 if window is None else max(0, (prefix + 1 - window) // 128)
            for logical in range(first, (prefix + 127) // 128):
                assert source_pages[logical] >= 0 and target_pages[logical] >= 0
                torch.testing.assert_close(
                    target_buffers[target_pages[logical]].view(torch.uint8),
                    buffers[source_pages[logical]].view(torch.uint8),
                    rtol=0,
                    atol=0,
                    msg=f"layer={layer}, role={role.name}, page={logical}",
                )
                expected_bytes += buffers[source_pages[logical]].numel() * buffers.element_size()
                compared.add((layer, role))
        # Every selected physical role moves a complete page, including packed
        # scales and compressor state. DP4 must not shard these bytes by TP=4.
        assert int(write.sizes.sum()) == expected_bytes
        if prefix:
            assert (20, CSA2CacheRole.GLOBAL) in compared
            assert (20, CSA2CacheRole.INDEX) in compared
            assert sum(role == CSA2CacheRole.SWA for _, role in compared) == 20
        for layer in range(20, 40):
            buffers = target.get_buffers(layer)
            for page in set(target.get_cache_indices(72, layer, CSA2CacheRole.SWA)):
                if page >= 0:
                    assert torch.count_nonzero(buffers[page]) == 0

        # Reverse exactly the mapped fragments after erasing the source: all
        # payload bytes must survive the round trip with unequal physical IDs.
        for src, _, count in zip(write.src_ptrs, write.dst_ptrs, write.sizes):
            _bytes(src, count).zero_()
        for src, dst, count in zip(write.src_ptrs, write.dst_ptrs, write.sizes):
            _bytes(src, count).copy_(_bytes(dst, count))
            torch.testing.assert_close(_bytes(src, count), _bytes(dst, count), atol=0, rtol=0)
    finally:
        source.free_resources(source_request)
        target.free_resources(target_request)
        source.shutdown()
        target.shutdown()
