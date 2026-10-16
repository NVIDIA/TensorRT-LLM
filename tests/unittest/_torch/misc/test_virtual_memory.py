# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest
import torch
from utils.util import get_current_process_gpu_memory

import tensorrt_llm
from tensorrt_llm._torch import virtual_memory
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import \
    KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import (BAD_PAGE_INDEX,
                                                      CacheLevel,
                                                      GpuCacheTierConfig,
                                                      KVCacheManagerConfig,
                                                      _KVCache)


@pytest.fixture(scope="function", autouse=True)
def cuda_sync_fixture():
    """
    Synchronizes CUDA to catch device errors.
    """

    torch.cuda.synchronize()
    yield
    torch.cuda.synchronize()


@pytest.fixture(scope="function", autouse=True)
def clean_cache():
    gc.collect()
    torch.cuda.empty_cache()
    yield
    gc.collect()
    torch.cuda.empty_cache()


def test_basic(process_gpu_memory_info_available):
    memory_usage_begin = get_current_process_gpu_memory()

    alloc_size = 256 * 1024 * 1024
    tag = "test_tag"

    with virtual_memory.scope(tag) as pool:
        tensor = torch.full([alloc_size], 42, dtype=torch.int8, device='cuda')
        memory_usage_materialized = get_current_process_gpu_memory()
        if process_gpu_memory_info_available:
            assert memory_usage_begin + alloc_size == memory_usage_materialized

    assert tensor[0].item() == 42

    torch.cuda.synchronize()
    virtual_memory.release_with_tag(tag)

    memory_usage_released = get_current_process_gpu_memory()
    if process_gpu_memory_info_available:
        assert memory_usage_begin == memory_usage_released

    torch.cuda.synchronize()
    virtual_memory.materialize_with_tag(tag)

    memory_usage_rematerialized = get_current_process_gpu_memory()
    if process_gpu_memory_info_available:
        assert memory_usage_begin + alloc_size == memory_usage_rematerialized

    torch.fill_(tensor, 24)
    assert tensor[0].item() == 24

    del tensor
    del pool

    memory_usage_end = get_current_process_gpu_memory()
    if process_gpu_memory_info_available:
        assert memory_usage_begin == memory_usage_end


def test_nested_scope(process_gpu_memory_info_available):
    memory_usage_begin = get_current_process_gpu_memory()

    alloc_size = 256 * 1024 * 1024
    outer_tag = "outer_tag"
    inner_tag = "inner_tag"

    with virtual_memory.scope(outer_tag) as outer_pool:
        outer_tensor = torch.full([alloc_size],
                                  42,
                                  dtype=torch.int8,
                                  device='cuda')

        with virtual_memory.scope(inner_tag) as inner_pool:
            inner_tensor = torch.full([alloc_size],
                                      24,
                                      dtype=torch.int8,
                                      device='cuda')
            memory_usage_both = get_current_process_gpu_memory()
            if process_gpu_memory_info_available:
                assert memory_usage_begin + 2 * alloc_size == memory_usage_both

        # After inner scope exits, allocations resume on the outer scope
        extra_outer = torch.full([alloc_size],
                                 99,
                                 dtype=torch.int8,
                                 device='cuda')

    assert outer_tensor[0].item() == 42
    assert inner_tensor[0].item() == 24
    assert extra_outer[0].item() == 99

    # Release only the inner tag - outer allocations stay materialized
    torch.cuda.synchronize()
    virtual_memory.release_with_tag(inner_tag)

    memory_after_inner_release = get_current_process_gpu_memory()
    if process_gpu_memory_info_available:
        assert memory_after_inner_release == memory_usage_begin + 2 * alloc_size

    assert outer_tensor[0].item() == 42
    assert extra_outer[0].item() == 99

    # Release the outer tag
    torch.cuda.synchronize()
    virtual_memory.release_with_tag(outer_tag)

    memory_after_both_release = get_current_process_gpu_memory()
    if process_gpu_memory_info_available:
        assert memory_after_both_release == memory_usage_begin

    # Re-materialize both tags
    torch.cuda.synchronize()
    virtual_memory.materialize_with_tag(inner_tag)
    virtual_memory.materialize_with_tag(outer_tag)

    memory_rematerialized = get_current_process_gpu_memory()
    if process_gpu_memory_info_available:
        assert memory_rematerialized == memory_usage_begin + 3 * alloc_size

    torch.fill_(outer_tensor, 1)
    torch.fill_(inner_tensor, 2)
    torch.fill_(extra_outer, 3)
    assert outer_tensor[0].item() == 1
    assert inner_tensor[0].item() == 2
    assert extra_outer[0].item() == 3

    del outer_tensor, inner_tensor, extra_outer
    del outer_pool, inner_pool

    memory_usage_end = get_current_process_gpu_memory()
    if process_gpu_memory_info_available:
        assert memory_usage_end == memory_usage_begin


def test_restore():
    alloc_size = 1024 * 1024
    tag = "test_tag"

    with virtual_memory.scope(tag, virtual_memory.RestoreMode.PINNED) as pool:
        tensor = torch.full([alloc_size], 42, dtype=torch.int8, device='cuda')

    assert tensor[0].item() == 42

    torch.cuda.synchronize()
    virtual_memory.release_with_tag(tag)

    torch.cuda.synchronize()

    virtual_memory.materialize_with_tag(tag)
    torch.cuda.synchronize()

    assert tensor[0].item() == 42

    del tensor
    del pool


class _KVCacheManagerV2WithColdGpu(KVCacheManagerV2):

    def _build_cache_config(
            self, config: KVCacheManagerConfig) -> KVCacheManagerConfig:
        config = super()._build_cache_config(config)
        gpu_quota = config.cache_tiers[0].quota
        config.cache_tiers = [
            config.cache_tiers[0],
            GpuCacheTierConfig(quota=gpu_quota),
            *config.cache_tiers[1:],
        ]
        return config


@pytest.mark.parametrize("manager_cls,restore_mode", [
    (KVCacheManager, virtual_memory.RestoreMode.NONE),
    *[(manager_cls, mode)
      for manager_cls in (KVCacheManagerV2, _KVCacheManagerV2WithColdGpu)
      for mode in
      (virtual_memory.RestoreMode.NONE, virtual_memory.RestoreMode.MEMSET,
       virtual_memory.RestoreMode.CPU, virtual_memory.RestoreMode.PINNED)],
])
def test_kv_cache_manager(process_gpu_memory_info_available, manager_cls,
                          restore_mode):
    is_v2 = issubclass(manager_cls, KVCacheManagerV2)
    kv_cache_params = {
        "kv_cache_config": KvCacheConfig(max_tokens=1024),
        "kv_cache_type": CacheType.SELF,
        "num_layers": 8,
        "num_kv_heads": 256,
        "head_dim": 64,
        "tokens_per_block": 64,
        "max_seq_len": 1024,
        "max_batch_size": 1,
        "mapping": Mapping(world_size=1, tp_size=1, rank=0),
        "dtype": tensorrt_llm.bindings.DataType.FP8,
    }
    if is_v2:
        # Exercise the static device-side page-index conversion tables too.
        kv_cache_params["head_dim"] = [64, 32] * 4
        kv_cache_params["max_beam_width"] = 2
        kv_cache_params["kv_cache_config"].host_cache_size = (
            0 if restore_mode == virtual_memory.RestoreMode.NONE else 8 << 20
            if restore_mode == virtual_memory.RestoreMode.MEMSET else 1 << 30)

    mgr = manager_cls(**kv_cache_params)
    if is_v2:
        # Warm one-time process-wide native page migration resources.
        mgr.add_dummy_requests([123], token_nums=[128], is_gen=True)
        mgr.release_memory(restore_mode)
        mgr.materialize_memory()
    mgr.shutdown()
    del mgr

    memory_usage_begin = get_current_process_gpu_memory()

    tag = "test_tag"
    cache_size = torch.empty(
        [
            2,  # KV
            8,  # Layers
            256,  # Heads
            1024,  # Tokens
            64,  # Head dim
        ],
        dtype=torch.float8_e4m3fn,
        device='meta')

    alloc_size = cache_size.nelement()
    if is_v2:
        alloc_size = alloc_size * 3 // 4

    with virtual_memory.scope(tag, restore_mode) as pool:
        mgr = manager_cls(**kv_cache_params)
        assert isinstance(mgr, manager_cls)
        if is_v2:
            assert mgr._use_per_layer_page_tables
            metadata = mgr._device_attention_op_scales
            expected_metadata = metadata.cpu()
            gpu_levels = [
                CacheLevel(level)
                for level, tier in enumerate(mgr.impl.init_config.cache_tiers)
                if isinstance(tier, GpuCacheTierConfig)
            ]
            gpu_capacities = [[
                stat.total for stat in mgr.impl.get_storage_statistics(level)
            ] for level in gpu_levels]
            all_capacities = [[
                stat.total
                for stat in mgr.impl.get_storage_statistics(CacheLevel(level))
            ] for level in range(len(mgr.impl.init_config.cache_tiers))]
            gpu_pool_bytes = sum(
                stat.total * sum(stat.slot_sizes) for level in gpu_levels
                for stat in mgr.impl.get_storage_statistics(level))
            assert gpu_pool_bytes >= alloc_size * len(gpu_levels)
            alloc_size = gpu_pool_bytes
        memory_usage_materialized = get_current_process_gpu_memory()
        if process_gpu_memory_info_available:
            assert memory_usage_materialized >= memory_usage_begin + alloc_size

    buffers = mgr.get_buffers(0)
    pointers = [buffer.data_ptr() for buffer in buffers]
    resource = None
    if is_v2:
        from types import SimpleNamespace

        resource = SimpleNamespace(
            release_memory=lambda tags, manager=mgr: manager.release_memory(
                restore_mode),
            materialize_memory=lambda tags, manager=mgr: manager.
            materialize_memory(),
        )
        dummy = mgr.add_dummy_requests([123],
                                       token_nums=[128],
                                       is_gen=True,
                                       max_beam_width=2)[0]
        index = mgr.index_mapper.get_index(dummy.py_request_id)
    for cycle in range(2):
        for buffer in buffers:
            buffer.fill_(cycle + 1)
        torch.cuda.synchronize()
        if is_v2:
            old_cache = mgr.kv_cache_map[dummy.py_request_id]
            old_shape = (old_cache.capacity, old_cache.history_length,
                         old_cache.beam_width, old_cache.is_decoding)
        assert virtual_memory.release_with_tag(tag, resource=resource) > 0
        if is_v2:
            assert mgr.index_mapper.get_index(dummy.py_request_id) == index
            assert mgr.kv_cache_map[dummy.py_request_id] is old_cache
            assert old_cache.status != _KVCache.Status.CLOSED
            assert not old_cache.is_active
            assert all(stat.total == 0 for level in gpu_levels
                       for stat in mgr.impl.get_storage_statistics(level))
            if restore_mode in (virtual_memory.RestoreMode.NONE,
                                virtual_memory.RestoreMode.MEMSET):
                assert all(stat.total == 0
                           for level in range(len(all_capacities))
                           for stat in mgr.impl.get_storage_statistics(
                               CacheLevel(level)))
        memory_usage_released = get_current_process_gpu_memory()
        if process_gpu_memory_info_available:
            assert memory_usage_materialized - memory_usage_released >= alloc_size
        assert virtual_memory.materialize_with_tag(tag, resource=resource) > 0
        torch.cuda.synchronize()
        memory_usage_rematerialized = get_current_process_gpu_memory()
        if process_gpu_memory_info_available:
            assert memory_usage_rematerialized - memory_usage_released >= alloc_size
        assert [buffer.data_ptr() for buffer in mgr.get_buffers(0)] == pointers
        if is_v2:
            assert [[
                stat.total for stat in mgr.impl.get_storage_statistics(level)
            ] for level in gpu_levels] == gpu_capacities
            assert [[
                stat.total
                for stat in mgr.impl.get_storage_statistics(CacheLevel(level))
            ] for level in range(len(all_capacities))] == all_capacities
            assert torch.equal(metadata.cpu(), expected_metadata)
            assert mgr.index_mapper.get_index(dummy.py_request_id) == index
            cache = mgr.kv_cache_map[dummy.py_request_id]
            assert cache.is_active
            assert cache is old_cache
            assert (cache.capacity, cache.history_length, cache.beam_width,
                    cache.is_decoding) == old_shape
            # Shared prompt blocks use beam 0 via cache indirection; the writable tail has both beams.
            first = cache.get_base_page_indices(0, 0)
            second = cache.get_base_page_indices(0, 1)
            assert first[0] != BAD_PAGE_INDEX
            assert second[0] == BAD_PAGE_INDEX
            assert first[1] != BAD_PAGE_INDEX and second[1] != BAD_PAGE_INDEX
            assert first[1] != second[1]
        if restore_mode != virtual_memory.RestoreMode.NONE:
            expected = 0 if restore_mode == virtual_memory.RestoreMode.MEMSET else cycle + 1
            if is_v2 and restore_mode != virtual_memory.RestoreMode.MEMSET:
                pages = mgr.get_batch_cache_indices([dummy.py_request_id],
                                                    layer_idx=0)[0]
                assert pages
                for page in pages:
                    assert torch.all(
                        mgr.get_buffers(0)[page].cpu().float() == expected)
            else:
                for buffer in buffers:
                    assert torch.all(buffer.cpu().float() == expected)

    del buffer, buffers
    if is_v2:
        del metadata, expected_metadata
    # Destruction must also be valid while the physical pages are released.
    virtual_memory.release_with_tag(tag, resource=resource)

    mgr.shutdown()
    if is_v2:
        del cache, old_cache, dummy, resource
    del mgr
    del pool

    memory_usage_end = get_current_process_gpu_memory()
    if process_gpu_memory_info_available:
        assert memory_usage_begin == memory_usage_end


@pytest.mark.parametrize("restore_mode,disk_bytes", [
    (virtual_memory.RestoreMode.CPU, 0),
    (virtual_memory.RestoreMode.PINNED, 16 << 20),
])
def test_kv_sleep_preserves_held_pages_under_cold_tier_pressure(
        tmp_path, restore_mode, disk_bytes):
    from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheLevel, ReuseScope

    mgr = KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            host_cache_size=16 << 20,
            disk_cache_size=disk_bytes,
            disk_cache_path=str(tmp_path),
        ),
        kv_cache_type=CacheType.SELF,
        num_layers=1,
        num_kv_heads=128,
        head_dim=1024,
        tokens_per_block=4,
        max_seq_len=16,
        max_batch_size=1,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )
    try:
        assert mgr.impl.resize(CacheLevel(0), 14 << 20)
        assert sum(stat.total
                   for stat in mgr.impl.get_storage_statistics()) == 7
        dummy = mgr.add_dummy_requests([123], token_nums=[4], is_gen=True)[0]
        held = mgr.kv_cache_map[dummy.py_request_id]
        # Each block occupies 2 MiB. Fill GPU and cold tiers with distinct prefixes.
        prefixes = [[i * 4 + j for j in range(4)] for i in range(24)]
        for tokens in prefixes:
            cache = mgr.impl.create_kv_cache(ReuseScope(), tokens)
            assert cache.resume(mgr._stream.cuda_stream)
            assert cache.resize(4)
            cache.commit(tokens)
            cache.close()
        reusable_before = sum(
            mgr.impl.probe_reuse(ReuseScope(), tokens) for tokens in prefixes)
        assert reusable_before > 0
        pages = mgr.get_batch_cache_indices([dummy.py_request_id],
                                            layer_idx=0)[0]
        for page in pages:
            mgr.get_buffers(0)[page].fill_(7)
        torch.cuda.synchronize()
        slots = [desc.num_slots for desc in mgr.impl.pool_group_descs]
        mgr.release_memory(restore_mode)
        assert all(stat.total == 0
                   for stat in mgr.impl.get_storage_statistics())
        mgr.materialize_memory()
        assert [desc.num_slots for desc in mgr.impl.pool_group_descs] == slots
        assert mgr.kv_cache_map[dummy.py_request_id] is held
        pages = mgr.get_batch_cache_indices([dummy.py_request_id],
                                            layer_idx=0)[0]
        assert pages
        for page in pages:
            assert torch.all(mgr.get_buffers(0)[page].cpu() == 7)
        reusable_after = sum(
            mgr.impl.probe_reuse(ReuseScope(), tokens) for tokens in prefixes)
        assert 0 < reusable_after < reusable_before
    finally:
        mgr.shutdown()


@pytest.mark.parametrize(
    "restore_mode",
    [virtual_memory.RestoreMode.CPU, virtual_memory.RestoreMode.MEMSET])
def test_cuda_graph(process_gpu_memory_info_available, restore_mode):
    from types import SimpleNamespace

    def work(input: torch.Tensor) -> torch.Tensor:
        intermediate = input + input
        output = input + intermediate
        return output

    g = torch.cuda.CUDAGraph()
    tag = "cuda_graph"

    with virtual_memory.scope(tag, restore_mode) as pool:
        manager = KVCacheManagerV2(
            kv_cache_config=KvCacheConfig(max_gpu_total_bytes=4 << 20,
                                          host_cache_size=8 << 20),
            kv_cache_type=CacheType.SELF,
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            tokens_per_block=4,
            max_seq_len=16,
            max_batch_size=1,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=tensorrt_llm.bindings.DataType.FLOAT,
        )
        hole = manager.add_dummy_requests([122], token_nums=[4], is_gen=True)[0]
        dummy = manager.add_dummy_requests([123], token_nums=[4],
                                           is_gen=True)[0]
        manager.free_resources(hole)
        page = manager.get_batch_cache_indices([dummy.py_request_id],
                                               layer_idx=0)[0][0]
        kv_pool = manager.get_buffers(0)
        pool_address = kv_pool.data_ptr()
        block_offsets = torch.zeros(
            (manager.num_attention_op_pools, 1, 2, manager.max_blocks_per_seq),
            dtype=torch.int32,
            device="cuda")
        manager.copy_batch_block_offsets(block_offsets, [dummy.py_request_id],
                                         1, 0, 1)
        static_output = torch.zeros(1024, dtype=kv_pool.dtype, device="cuda")
        resource = SimpleNamespace(
            release_memory=lambda tags: manager.release_memory(restore_mode),
            materialize_memory=lambda tags: manager.materialize_memory(),
        )

        with torch.cuda.graph(g):
            # Attention offsets index individual K/V pages, while get_buffers groups them by slot.
            static_input = kv_pool.flatten(0, 1).index_select(
                0, block_offsets[0, 0, 0, :1].long()).view(-1)[:1024]
            static_output.copy_(work(static_input))

    kv_pool[page].fill_(1.0)
    g.replay()

    torch.cuda.synchronize()
    assert static_output[0].item() == 3.0

    memory_usage_before = get_current_process_gpu_memory()

    torch.cuda.synchronize()
    virtual_memory.release_with_tag(tag, resource=resource)

    memory_usage_released = get_current_process_gpu_memory()
    if process_gpu_memory_info_available:
        assert memory_usage_released < memory_usage_before

    torch.cuda.synchronize()
    virtual_memory.materialize_with_tag(tag, resource=resource)
    restored_page = manager.get_batch_cache_indices([dummy.py_request_id],
                                                    layer_idx=0)[0][0]
    assert manager.get_buffers(0).data_ptr() == pool_address
    # The freed hole forces this test through the changed-slot path.
    assert restored_page != page
    manager.copy_batch_block_offsets(block_offsets, [dummy.py_request_id], 1, 0,
                                     1)

    if restore_mode == virtual_memory.RestoreMode.CPU:
        assert torch.all(kv_pool[restored_page] == 1.0)
    else:
        assert torch.all(kv_pool[restored_page] == 0.0)
    kv_pool[page].fill_(7.0)
    kv_pool[restored_page].fill_(1.0)
    torch.fill_(static_output, 0.0)
    g.replay()

    torch.cuda.synchronize()
    assert static_output[0].item() == 3.0

    del static_input, static_output, block_offsets, kv_pool, g, pool
    manager.shutdown()
