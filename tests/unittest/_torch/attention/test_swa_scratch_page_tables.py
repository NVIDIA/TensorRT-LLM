# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha,
)
from tensorrt_llm._torch.memory_buffer_utils import Buffers
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
    copy_per_layer_block_offsets_to_device,
)
from tensorrt_llm.llmapi.llm_args import EagleDecodingConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX, GPU_LEVEL, PageIndexMode


def _make_manager(
    scratch: bool,
    dtype: DataType,
    *,
    short: bool = False,
    spec_config: EagleDecodingConfig | None = None,
) -> KVCacheManagerV2:
    return KVCacheManagerV2(
        KvCacheConfig(
            max_tokens=1024,
            max_attention_window=[32, 512, 32, 512, 32, 512],
            enable_block_reuse=False,
            enable_swa_scratch_reuse=scratch,
            dtype="nvfp4" if dtype == DataType.NVFP4 else "auto",
        ),
        CacheType.SELF,
        num_layers=6,
        num_kv_heads=2,
        head_dim=128 if spec_config is not None else [64, 128, 64, 128, 64, 128],
        tokens_per_block=8,
        max_seq_len=32 if short else 512,
        max_batch_size=256,
        max_num_tokens=256,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=dtype,
        vocab_size=1024,
        spec_config=spec_config,
    )


@pytest.fixture
def make_manager(request: pytest.FixtureRequest) -> Callable[..., KVCacheManagerV2]:
    def create(*args, **kwargs) -> KVCacheManagerV2:
        manager = _make_manager(*args, **kwargs)
        request.addfinalizer(manager.shutdown)
        return manager

    return create


def _extent(manager: KVCacheManagerV2, layer: int) -> int:
    attn = SimpleNamespace(local_layer_idx=layer)
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha._attn_ref = lambda: attn
    return fmha._get_total_num_blocks(SimpleNamespace(kv_cache_manager=manager))


def _copy(manager: KVCacheManagerV2, ids: list[int], num_contexts: int) -> torch.Tensor:
    actual = torch.empty(
        manager.num_attention_op_pools,
        len(ids),
        2,
        manager.max_blocks_per_seq,
        device="cuda",
        dtype=torch.int32,
    )
    with torch.cuda.stream(manager._stream):
        manager.copy_batch_block_offsets(actual, ids, 1, num_contexts, len(ids))
    manager._stream.synchronize()
    return actual.cpu()


def test_trtllm_gen_reads_final_slot_from_per_layer_extent() -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("The trtllm-gen context kernel requires SM100-family GPUs.")
    import flashinfer

    # Two slots with three K/V layer pairs: the final layer reads pages 10 and 11.
    manager = object.__new__(KVCacheManagerV2)
    manager._use_per_layer_page_tables = True
    manager.impl = SimpleNamespace(
        get_page_index_upper_bound=lambda *_: 8,
        get_page_index_converter=lambda *_: SimpleNamespace(layer_offset=4, expansion=1),
    )
    pool = torch.zeros((12, 1, 32, 128), device="cuda", dtype=torch.bfloat16)
    pool[11].fill_(1)
    query = torch.zeros((32, 4, 128), device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(query)
    view = pool[: _extent(manager, 2)]
    flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        query=query,
        kv_cache=(view, view),
        workspace_buffer=torch.empty(128 << 20, device="cuda", dtype=torch.uint8),
        block_tables=torch.tensor([[[10], [11]]], device="cuda", dtype=torch.int32),
        seq_lens=torch.tensor([32], device="cuda", dtype=torch.int32),
        max_q_len=32,
        max_kv_len=32,
        bmm1_scale=128**-0.5,
        bmm2_scale=1.0,
        batch_size=1,
        cum_seq_lens_q=torch.tensor([0, 32], device="cuda", dtype=torch.int32),
        cum_seq_lens_kv=torch.tensor([0, 32], device="cuda", dtype=torch.int32),
        window_left=-1,
        out=output,
        kv_layout="HND",
        enable_pdl=False,
        uses_shared_paged_kv_idx=False,
        causal=True,
    )
    torch.testing.assert_close(output, torch.ones_like(output), atol=0, rtol=0)


def test_scratch_reserves_speculative_rewind_tail(make_manager) -> None:
    spec = EagleDecodingConfig(speculative_model="unused-draft", max_draft_len=9)
    manager = make_manager(True, DataType.BF16, spec_config=spec)
    scratch_config = manager.kv_cache_manager_py_config.swa_scratch_reuse
    assert scratch_config is not None
    assert scratch_config.max_rewind_len == manager.num_extra_kv_tokens == 8
    assert manager.add_dummy_requests([11], [193]) is not None
    cache = manager.kv_cache_map[11]
    assert cache.capacity == 193 + manager.num_extra_kv_tokens
    saw_scratch = False
    for layer, window in enumerate(manager.max_attention_window_vec):
        scratch_range = cache.get_swa_scratch_range(manager.impl.get_layer_group_id(layer))
        if window is not None:
            assert scratch_range
            saw_scratch = True
            assert scratch_range.end * manager.tokens_per_block + window <= 194
    assert saw_scratch
    actual = _copy(manager, [11], 1)
    for layer in range(manager.num_local_layers):
        valid = actual[layer] != BAD_PAGE_INDEX
        assert torch.all(actual[layer][valid] < _extent(manager, layer))


def test_per_layer_tables_refresh_aliased_graph_buffers(make_manager) -> None:
    manager = make_manager(True, DataType.BF16)
    assert manager.add_dummy_requests([11, 22], [65, 65], is_gen=True) is not None
    buffers = Buffers()
    shape = [manager.num_attention_op_pools, 2, 2, manager.max_blocks_per_seq]
    large = buffers.get_buffer(shape, torch.int32, "page_table", True)
    small = buffers.get_buffer([shape[0], 1, *shape[2:]], torch.int32, "page_table", True)
    assert large is not small and large.data_ptr() == small.data_ptr()

    def publish(destination, ids):
        manager.copy_batch_block_offsets(destination, ids, 1, 0, len(ids), max_blocks=9)
        torch.cuda.synchronize()

    publish(large, [11, 22])
    expected = large[..., :9].clone()
    for request_id in (22, 11):
        publish(small, [request_id])
        publish(large, [11, 22])
        torch.testing.assert_close(large[..., :9], expected)


@pytest.mark.parametrize("dtype", [DataType.BF16, DataType.FP8, DataType.NVFP4])
@pytest.mark.parametrize("long_context", [True, False])
def test_scratch_tables_match_core_converter_for_mixed_reordered_rows(
    make_manager, dtype: DataType, long_context: bool
) -> None:
    manager = make_manager(True, dtype)
    assert manager.add_dummy_requests([11, 22], [193 if long_context else 17, 17]) is not None
    assert manager.add_dummy_requests([33], [65], is_gen=True) is not None
    ids = [22, 11, 33]
    actual = _copy(manager, ids, 2)
    copy_idx = manager.index_mapper.get_copy_index(ids, 2, 1)
    saw_wraparound = saw_invalid_scratch = False
    for layer in range(manager.num_local_layers):
        group = manager.impl.get_layer_group_id(layer)
        for kv_idx, role in enumerate((Role.KEY, Role.VALUE)):
            converter = manager.impl.get_page_index_converter(layer, role)
            if dtype == DataType.NVFP4:
                scale_role = (Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE)[kv_idx]
                scale_converter = manager.impl.get_page_index_converter(layer, scale_role)
                for attr in ("scale", "layer_offset", "expansion", "scratch_pages_per_block"):
                    assert getattr(converter, attr) == getattr(scale_converter, attr)
                scale_extent = manager.impl.get_page_index_upper_bound(layer, scale_role)
                scale_extent += scale_converter.layer_offset * scale_converter.expansion
                assert scale_extent == _extent(manager, layer)
            for row, request_id in enumerate(ids):
                cache = manager.kv_cache_map[request_id]
                base = manager.host_kv_cache_block_offsets[group, int(copy_idx[row]), 0].tolist()
                core_base = cache.get_base_page_indices(group)
                assert base[: cache.num_blocks] == list(core_base[: cache.num_blocks])
                scratch_range = cache.get_swa_scratch_range(group) if row < 2 else None
                scratch_slots = cache.get_scratch_slots(group) if scratch_range else ()
                if request_id == 22:
                    assert not scratch_range
                if scratch_range:
                    saw_wraparound |= len(scratch_range) > converter.scale
                    saw_invalid_scratch |= (
                        BAD_PAGE_INDEX in base[scratch_range.beg : scratch_range.end]
                    )
                expected = converter(base, PageIndexMode.PER_LAYER, scratch_range, scratch_slots)
                torch.testing.assert_close(
                    actual[layer, row, kv_idx], torch.tensor(expected, dtype=torch.int32)
                )
                valid = actual[layer, row, kv_idx] != BAD_PAGE_INDEX
                assert torch.all(actual[layer, row, kv_idx, valid] < _extent(manager, layer))
    assert saw_wraparound == long_context
    assert saw_invalid_scratch == long_context
    if not long_context:
        materializer = manager._page_table_materializer
        native_staging = materializer.uses_device_staging
        for staging in (native_staging, True):
            materializer._use_device_staging = staging
            bounded = torch.full(
                (
                    manager.num_attention_op_pools + 1,
                    len(ids) + 2,
                    2,
                    manager.max_blocks_per_seq,
                ),
                -123,
                dtype=torch.int32,
                device="cuda",
            )
            manager.copy_batch_block_offsets(bounded, ids, 1, 2, len(ids), max_blocks=3)
            torch.testing.assert_close(
                bounded[: manager.num_attention_op_pools, : len(ids), :, :3].cpu(),
                actual[..., :3],
            )
            assert torch.all(bounded[..., 3:] == -123)
            assert torch.all(bounded[manager.num_attention_op_pools :] == -123)
            assert torch.all(bounded[:, len(ids) :] == -123)
        materializer._use_device_staging = native_staging


@pytest.mark.parametrize("scratch, short", [(True, False), (False, True), (False, False)])
def test_final_physical_slot_is_inside_attention_extent_after_resize(
    make_manager, scratch: bool, short: bool
) -> None:
    manager = make_manager(scratch, DataType.BF16, short=short)
    assert manager._use_per_layer_page_tables == (scratch or short)
    previous_extents = []
    for pool_bytes in (4 << 20, 8 << 20):
        assert manager.impl.resize(GPU_LEVEL, pool_bytes)
        manager._prepare_page_table_tensor(256)
        extents = [_extent(manager, layer) for layer in range(manager.num_local_layers)]
        assert extents != previous_extents
        previous_extents = extents
        requests = []
        for request_id in range(256):
            allocated = manager.add_dummy_requests([request_id], [1])
            if allocated is None:
                break
            requests.extend(allocated)
        # Growing admitted requests can use the slots beyond the resume watermark.
        for request in reversed(requests):
            cache = manager.kv_cache_map[request.py_request_id]
            while cache.capacity < manager.max_seq_len:
                if any(s.free == 0 for s in manager.impl.get_storage_statistics(GPU_LEVEL)):
                    break
                capacity = (
                    (cache.capacity + manager.tokens_per_block - 1) // manager.tokens_per_block + 1
                ) * manager.tokens_per_block
                assert cache.resize(capacity, cache.capacity)
        stats = manager.impl.get_storage_statistics(GPU_LEVEL)
        groups = manager.impl.get_life_cycle_pool_group_indices(GPU_LEVEL)
        assert requests and any(s.free == 0 for s in stats), "The test must fill a physical pool."
        ids = [request.py_request_id for request in reversed(requests)]
        with patch(
            "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2."
            "copy_per_layer_block_offsets_to_device",
            wraps=copy_per_layer_block_offsets_to_device,
        ) as copy_kernel:
            actual = _copy(manager, ids, len(ids))
        assert copy_kernel.call_count == int(scratch)
        saw_final_slot_with_offset = False
        for layer in range(manager.num_local_layers):
            converter = manager.impl.get_page_index_converter(layer, Role.KEY)
            upper = _extent(manager, layer)
            group = manager.impl.get_layer_group_id(layer)
            expected = stats[int(groups[group])].total * converter.scale * converter.expansion
            if not manager._use_per_layer_page_tables:
                expected -= converter.layer_offset * converter.expansion
            assert upper == expected
            pool = (
                layer
                if manager._use_per_layer_page_tables
                else manager.impl.get_layer_group_id(layer)
            )
            for row, request_id in enumerate(ids):
                count = manager.kv_cache_map[request_id].num_blocks
                entries = actual[pool, row, :, :count]
                assert torch.all(entries < upper)
                saw_final_slot_with_offset |= converter.layer_offset > 0 and bool(
                    torch.any(entries >= upper - converter.scale)
                )
        assert saw_final_slot_with_offset
        for request in requests:
            manager.free_resources(request)


def _group_lanes_by_pool(
    lane_pools: torch.Tensor, num_pools: int
) -> tuple[torch.Tensor, torch.Tensor]:
    lanes = torch.argsort(lane_pools, stable=True).to(dtype=torch.int32)
    offsets = torch.zeros(num_pools + 1, dtype=torch.int32)
    offsets[1:] = torch.cumsum(torch.bincount(lane_pools, minlength=num_pools), dim=0)
    return offsets, lanes


@pytest.mark.parametrize("source_on_device", [False, True], ids=["pinned_host", "device"])
def test_per_layer_copy_kernel_matches_reference(source_on_device: bool) -> None:
    torch.manual_seed(0)
    num_pools, rows, width, num_layers, max_blocks = 3, 7, 12, 5, 9
    # Lane ``layer * 2 + role``: layer 3 keeps K and V in different pools, and
    # pool 2 serves a single lane.
    lane_pools = torch.tensor([0, 0, 1, 1, 0, 0, 1, 2, 0, 0])
    source = torch.randint(0, 50, (num_pools, rows, 2, width), dtype=torch.int32)
    source[torch.rand(source.shape) < 0.2] = BAD_PAGE_INDEX
    copy_index = torch.tensor([5, 0, 3, 3], dtype=torch.int32)
    scales = torch.randint(1, 4, (num_layers, 2), dtype=torch.int32)
    layer_offsets = torch.randint(0, 3, (num_layers, 2), dtype=torch.int32)
    converters = torch.stack(
        (lane_pools.view(num_layers, 2).to(dtype=torch.int32), scales, layer_offsets), dim=-1
    )
    lane_offsets, lanes = _group_lanes_by_pool(lane_pools, num_pools)
    output = torch.full((num_layers, 6, 2, width), -123, dtype=torch.int32, device="cuda")
    copy_per_layer_block_offsets_to_device(
        source.cuda() if source_on_device else source.pin_memory(),
        output,
        copy_index.cuda() if source_on_device else copy_index.pin_memory(),
        converters.cuda(),
        lane_offsets.cuda(),
        lanes.cuda(),
        max_blocks,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    expected = torch.full((num_layers, 6, 2, width), -123, dtype=torch.int32)
    for lane, pool in enumerate(lane_pools.tolist()):
        layer, role = divmod(lane, 2)
        base = source[pool, copy_index.long(), 0, :max_blocks]
        expected[layer, : len(copy_index), role, :max_blocks] = torch.where(
            base == BAD_PAGE_INDEX,
            BAD_PAGE_INDEX,
            base * scales[layer, role] + layer_offsets[layer, role],
        )
    assert (expected == BAD_PAGE_INDEX).any()
    assert torch.equal(output.cpu(), expected)


def test_per_layer_copy_kernel_rejects_malformed_inputs() -> None:
    def tensor(*values_or_shape, shape=None) -> torch.Tensor:
        if shape is not None:
            return torch.zeros(shape, dtype=torch.int32, device="cuda")
        return torch.tensor(values_or_shape, dtype=torch.int32, device="cuda")

    well_formed = dict(
        input=tensor(shape=(1, 2, 2, 4)),
        output=tensor(shape=(1, 2, 2, 4)),
        copy_index=tensor(0, 1),
        converters=tensor(shape=(1, 2, 3)),
        pool_lane_offsets=tensor(0, 2),
        pool_lanes=tensor(0, 1),
        max_blocks=4,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    copy_per_layer_block_offsets_to_device(**well_formed)
    torch.cuda.synchronize()

    malformed = [
        ({"copy_index": tensor(0, 1).long()}, "copy_index must contain int32"),
        ({"max_blocks": 5}, "max_blocks 5 exceeds the table width"),
        ({"output": tensor(shape=(1, 2, 2, 8))}, "block widths must match"),
        ({"pool_lanes": tensor(0)}, "pool_lanes must list every output K/V lane"),
        ({"pool_lane_offsets": tensor(0)}, "pool_lane_offsets must hold"),
        ({"output": tensor(shape=(1, 1, 2, 4))}, "at least one row per copy_index entry"),
    ]
    for overrides, message in malformed:
        with pytest.raises(RuntimeError, match=message):
            copy_per_layer_block_offsets_to_device(**{**well_formed, **overrides})
