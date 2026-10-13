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


@pytest.mark.parametrize("manager_cls,restore_mode", [
    (KVCacheManager, virtual_memory.RestoreMode.NONE),
    *[(KVCacheManagerV2, mode) for mode in
      (virtual_memory.RestoreMode.NONE, virtual_memory.RestoreMode.MEMSET,
       virtual_memory.RestoreMode.CPU, virtual_memory.RestoreMode.PINNED)],
])
def test_kv_cache_manager(process_gpu_memory_info_available, manager_cls,
                          restore_mode):
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
    if manager_cls is KVCacheManagerV2:
        # Exercise the static device-side page-index conversion tables too.
        kv_cache_params["head_dim"] = [64, 32] * 4

    mgr = manager_cls(**kv_cache_params)
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
    if manager_cls is KVCacheManagerV2:
        alloc_size = alloc_size * 3 // 4

    with virtual_memory.scope(tag, restore_mode) as pool:
        mgr = manager_cls(**kv_cache_params)
        assert isinstance(mgr, manager_cls)
        if manager_cls is KVCacheManagerV2:
            assert mgr._use_per_layer_page_tables
            metadata = mgr._device_attention_op_scales
            expected_metadata = metadata.cpu()
        memory_usage_materialized = get_current_process_gpu_memory()
        if process_gpu_memory_info_available:
            assert memory_usage_materialized >= memory_usage_begin + alloc_size

    buffers = mgr.get_buffers(0)
    pointers = [buffer.data_ptr() for buffer in buffers]
    resource = None
    if manager_cls is KVCacheManagerV2:
        from types import SimpleNamespace

        resource = SimpleNamespace(
            release_memory=lambda tags, manager=mgr: manager.release_memory(
                restore_mode),
            materialize_memory=lambda tags, manager=mgr: manager.
            materialize_memory(),
        )
        dummy = mgr.add_dummy_requests([123], token_nums=[128], is_gen=True)[0]
        index = mgr.index_mapper.get_index(dummy.py_request_id)
    for cycle in range(2):
        for buffer in buffers:
            buffer.fill_(cycle + 1)
        torch.cuda.synchronize()
        if manager_cls is KVCacheManagerV2:
            old_cache = mgr.kv_cache_map[dummy.py_request_id]
        assert virtual_memory.release_with_tag(tag, resource=resource) > 0
        if manager_cls is KVCacheManagerV2:
            assert mgr.index_mapper.get_index(dummy.py_request_id) == index
            assert all(stat.total == 0
                       for stat in mgr.impl.get_storage_statistics())
        memory_usage_released = get_current_process_gpu_memory()
        if process_gpu_memory_info_available:
            assert memory_usage_materialized - memory_usage_released >= alloc_size
        assert virtual_memory.materialize_with_tag(tag, resource=resource) > 0
        torch.cuda.synchronize()
        memory_usage_rematerialized = get_current_process_gpu_memory()
        if process_gpu_memory_info_available:
            assert memory_usage_rematerialized - memory_usage_released >= alloc_size
        assert [buffer.data_ptr() for buffer in mgr.get_buffers(0)] == pointers
        if manager_cls is KVCacheManagerV2:
            assert torch.equal(metadata.cpu(), expected_metadata)
            assert mgr.index_mapper.get_index(dummy.py_request_id) == index
            cache = mgr.kv_cache_map[dummy.py_request_id]
            assert cache.is_active
            assert (cache
                    is old_cache) == (restore_mode
                                      in (virtual_memory.RestoreMode.CPU,
                                          virtual_memory.RestoreMode.PINNED))
        if restore_mode != virtual_memory.RestoreMode.NONE:
            expected = 0 if restore_mode == virtual_memory.RestoreMode.MEMSET else cycle + 1
            if manager_cls is KVCacheManagerV2 and restore_mode != virtual_memory.RestoreMode.MEMSET:
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
    if manager_cls is KVCacheManagerV2:
        del metadata, expected_metadata
    # Destruction must also be valid while the physical pages are released.
    virtual_memory.release_with_tag(tag, resource=resource)

    mgr.shutdown()
    if manager_cls is KVCacheManagerV2:
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


@pytest.mark.parametrize("restore_mode", [
    virtual_memory.RestoreMode.NONE,
    virtual_memory.RestoreMode.MEMSET,
    virtual_memory.RestoreMode.CPU,
    virtual_memory.RestoreMode.PINNED,
])
def test_kv_sleep_restores_caches_above_admission_threshold(restore_mode):
    from tensorrt_llm.runtime.kv_cache_manager_v2 import ReuseScope

    mgr = KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            host_cache_size=16 << 20,
            max_util_for_resume=0.5,
        ),
        kv_cache_type=CacheType.SELF,
        num_layers=1,
        num_kv_heads=128,
        head_dim=1024,
        tokens_per_block=4,
        max_seq_len=32,
        max_batch_size=3,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )
    try:
        mgr.add_dummy_requests([123, 1, 2], token_nums=[4, 4, 4], is_gen=True)
        capacities = {123: 8, 1: 12, 2: 12}
        for req_id, capacity in capacities.items():
            assert mgr.kv_cache_map[req_id].resize(capacity)
        assert sum(stat.total
                   for stat in mgr.impl.get_storage_statistics()) == 8
        torch.cuda.synchronize()

        # In any order, the first two caches exceed the admission threshold.
        mgr.release_memory(restore_mode)
        mgr.materialize_memory()
        for req_id, capacity in capacities.items():
            cache = mgr.kv_cache_map[req_id]
            assert cache.is_active
            assert cache.capacity == capacity

        probe = mgr.impl.create_kv_cache(ReuseScope())
        try:
            # Normal admission still observes the utilization threshold.
            assert not probe.resume(mgr._stream.cuda_stream)
            resident = mgr.kv_cache_map[123]
            resident.suspend()
            assert probe.resume(check_utilization=False)
            assert probe.resize(4)
            probe.suspend()
            assert resident.resume(check_utilization=False)
            # Eight occupied slots leave no room for the held probe page.
            assert not probe.resume(check_utilization=False)
            assert not probe.is_active
        finally:
            probe.close()
    finally:
        mgr.shutdown()


def test_cuda_graph(process_gpu_memory_info_available):

    def work(input: torch.Tensor) -> torch.Tensor:
        intermediate = input + input
        output = input + intermediate
        return output

    g = torch.cuda.CUDAGraph()
    tag = "cuda_graph"

    with virtual_memory.scope(tag) as pool:
        static_input = torch.ones(1024, dtype=torch.float32, device='cuda')
        static_output = torch.zeros(1024, dtype=torch.float32, device='cuda')

        with torch.cuda.graph(g):
            static_output.copy_(work(static_input))

    torch.fill_(static_input, 1.0)
    g.replay()

    torch.cuda.synchronize()
    assert static_output[0].item() == 3.0

    memory_usage_before = get_current_process_gpu_memory()

    torch.cuda.synchronize()
    virtual_memory.release_with_tag(tag)

    memory_usage_released = get_current_process_gpu_memory()
    if process_gpu_memory_info_available:
        assert memory_usage_released < memory_usage_before

    torch.cuda.synchronize()
    virtual_memory.materialize_with_tag(tag)

    torch.fill_(static_input, 1.0)
    torch.fill_(static_output, 0.0)
    g.replay()

    torch.cuda.synchronize()
    assert static_output[0].item() == 3.0

    del static_input, static_output, g, pool
