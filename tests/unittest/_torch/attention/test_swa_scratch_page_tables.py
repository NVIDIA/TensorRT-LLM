# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha,
)
from tensorrt_llm._torch.memory_buffer_utils import Buffers
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
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
        desc = cache.get_scratch_desc(manager.impl.get_layer_group_id(layer))
        if window is not None:
            assert desc
            saw_scratch = True
            assert desc.range.end * manager.tokens_per_block + window <= 194
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
                scratch = cache.get_scratch_desc(group) if row < 2 else None
                if request_id == 22:
                    assert not scratch
                if scratch:
                    saw_wraparound |= len(scratch.range) > converter.scale
                    saw_invalid_scratch |= (
                        BAD_PAGE_INDEX in base[scratch.range.beg : scratch.range.end]
                    )
                expected = converter(base, PageIndexMode.PER_LAYER, scratch)
                torch.testing.assert_close(
                    actual[layer, row, kv_idx], torch.tensor(expected, dtype=torch.int32)
                )
                valid = actual[layer, row, kv_idx] != BAD_PAGE_INDEX
                assert torch.all(actual[layer, row, kv_idx, valid] < _extent(manager, layer))
    assert saw_wraparound == long_context
    assert saw_invalid_scratch == long_context
    if not long_context:
        for direct_host_copy in (manager._per_layer_host_tables_pinned, False):
            manager._per_layer_host_tables_pinned = direct_host_copy
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
        actual = _copy(manager, ids, len(ids))
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
