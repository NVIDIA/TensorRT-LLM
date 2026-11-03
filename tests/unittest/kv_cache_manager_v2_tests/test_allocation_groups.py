# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import dataclasses
from importlib.util import find_spec

import pytest
import torch

if find_spec("kv_cache_manager_v2") is not None:
    import kv_cache_manager_v2 as kv
else:
    from tensorrt_llm.runtime import kv_cache_manager_v2 as kv

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _config(explicit: bool) -> kv.KVCacheManagerConfig:
    return kv.KVCacheManagerConfig(
        tokens_per_block=4,
        cache_tiers=[kv.GpuCacheTierConfig(quota=8 << 20)],
        layers=[
            kv.AttentionLayerConfig(
                layer_id=layer,
                buffers=[
                    kv.BufferConfig(
                        role=role,
                        size=size,
                        allocation_group_id=17 if explicit else None,
                    )
                    for role, size in (("key", 4096), ("value", 2048))
                ],
            )
            for layer in range(2)
        ],
        allocation_groups=[kv.AllocationGroupConfig(17, kv.AllocationUnit.TOKEN_BLOCK)]
        if explicit
        else [],
    )


def test_allocation_groups_survive_config_replace_and_copy():
    config = _config(True)
    updated = dataclasses.replace(config, enable_stats=False)
    cloned = copy.deepcopy(updated)
    assert not cloned.enable_stats
    assert len(cloned.allocation_groups) == 1
    group = cloned.allocation_groups[0]
    assert group.allocation_group_id == 17
    assert group.allocation_unit == kv.AllocationUnit.TOKEN_BLOCK
    assert group.reuse_policy == kv.ReusePolicy.REQUIRED
    assert all(
        buffer.allocation_group_id == 17 for layer in cloned.layers for buffer in layer.buffers
    )


def test_allocation_lifetime_fields_survive_copy_and_descriptor_equality():
    group = kv.AllocationGroupConfig(
        31,
        kv.AllocationUnit.FIXED,
        reuse_policy=kv.ReusePolicy.PRIVATE,
        initially_active=False,
        suspend_policy=kv.SuspendPolicy.RELEASE,
    )
    cloned = copy.deepcopy(group)
    assert not cloned.initially_active
    assert cloned.suspend_policy == kv.SuspendPolicy.RELEASE
    desc = kv.KVCacheDesc(32, 16, active_allocation_groups=[31])
    assert copy.deepcopy(desc) == desc
    assert desc.active_allocation_groups == [31]
    assert desc != kv.KVCacheDesc(32, 16)
    desc.active_allocation_groups = []
    assert desc == kv.KVCacheDesc(32, 16, active_allocation_groups=[])
    desc.active_allocation_groups = None
    assert desc == kv.KVCacheDesc(32, 16)


@requires_cuda
def test_deferred_group_is_released_while_held_groups_keep_their_slots():
    config = kv.KVCacheManagerConfig(
        tokens_per_block=4,
        cache_tiers=[kv.GpuCacheTierConfig(quota=16 << 20)],
        layers=[
            kv.AttentionLayerConfig(
                layer_id=layer,
                buffers=[
                    kv.BufferConfig(role="state", size=4096 << layer, allocation_group_id=group)
                ],
            )
            for layer, group in enumerate((101, 202, 303))
        ],
        allocation_groups=[
            kv.AllocationGroupConfig(
                group,
                kv.AllocationUnit.FIXED,
                reuse_policy=kv.ReusePolicy.PRIVATE,
                initially_active=group != 303,
                suspend_policy=kv.SuspendPolicy.RELEASE if group == 303 else kv.SuspendPolicy.HOLD,
            )
            for group in (101, 202, 303)
        ],
        constraints=[kv.BatchDesc([kv.KVCacheDesc(0, 0, active_allocation_groups=[303])])],
        commit_min_snapshot=True,
        max_util_for_resume=1.0,
    )
    manager = kv.KVCacheManager(config)
    stream = torch.cuda.Stream()
    cache = manager.create_kv_cache()
    try:
        groups = [manager.get_buffer_group_id(layer, "state") for layer in range(3)]
        pool = manager.get_life_cycle_pool_group_indices()[groups[2]]
        before = manager.get_storage_statistics()[pool].free
        assert cache.resume(stream.cuda_stream)
        held = [cache.get_ssm_block_base_index(group) for group in groups[:2]]
        assert cache.get_ssm_block_base_index(groups[2]) == -1
        assert manager.get_storage_statistics()[pool].free == before
        assert cache.activate_groups([303])
        assert cache.get_ssm_block_base_index(groups[2]) >= 0
        assert manager.get_storage_statistics()[pool].free == before - 1
        cache.suspend()
        assert manager.get_storage_statistics()[pool].free == before
        assert cache.resume(stream.cuda_stream)
        assert [cache.get_ssm_block_base_index(group) for group in groups[:2]] == held
        assert cache.get_ssm_block_base_index(groups[2]) >= 0
        cache.suspend()
        assert cache.resume(stream.cuda_stream, active_allocation_groups=[])
        assert cache.get_ssm_block_base_index(groups[2]) == -1
        assert manager.get_storage_statistics()[pool].free == before
    finally:
        cache.close()
        manager.shutdown()


def _pool_layout(manager: kv.KVCacheManager) -> tuple:
    return tuple(
        (
            pool.num_slots,
            tuple(
                (
                    variant.layer_group_id,
                    tuple(
                        (
                            buffer.single_buffer_size,
                            tuple((entry.layer_id, entry.role) for entry in buffer.buffer_ids),
                        )
                        for buffer in variant.coalesced_buffers
                    ),
                )
                for variant in pool.slot_desc.variants
            ),
        )
        for pool in manager.pool_group_descs
    )


@requires_cuda
def test_explicit_token_group_matches_implicit_layout_and_prefix_reuse():
    results = []
    for explicit in (False, True):
        stream = torch.cuda.Stream()
        manager = kv.KVCacheManager(_config(explicit))
        caches = []
        try:
            layout = _pool_layout(manager)
            group = manager.get_buffer_group_id(0, "key")
            assert all(
                manager.get_buffer_group_id(layer, role) == group
                for layer in range(2)
                for role in ("key", "value")
            )
            source = manager.create_kv_cache(input_tokens=[])
            caches.append(source)
            assert source.resume(stream.cuda_stream)
            assert source.resize(8)
            source.commit(list(range(8)))
            pages = source.get_base_page_indices(group).tolist()
            source.close()

            reader = manager.create_kv_cache(input_tokens=list(range(8)))
            caches.append(reader)
            assert reader.num_committed_tokens == 8
            assert reader.resume(stream.cuda_stream)
            assert reader.get_base_page_indices(group).tolist()[:2] == pages[:2]
            reader.suspend()
            assert reader.resume(stream.cuda_stream)
            assert reader.get_base_page_indices(group).tolist()[:2] == pages[:2]
            results.append((layout, reader.num_committed_tokens))
        finally:
            for cache in reversed(caches):
                cache.close()
            manager.shutdown()
    assert results[0] == results[1]


@requires_cuda
def test_equal_explicit_groups_keep_independent_membership():
    config = _config(False)
    layers = list(config.layers)
    layers[0] = kv.AttentionLayerConfig(
        layer_id=0,
        buffers=[
            kv.BufferConfig(role="key", size=4096, allocation_group_id=31),
            kv.BufferConfig(role="value", size=4096, allocation_group_id=47),
        ],
    )
    config.layers = layers
    config.allocation_groups = [
        kv.AllocationGroupConfig(group, kv.AllocationUnit.TOKEN_BLOCK) for group in (31, 47)
    ]
    stream = torch.cuda.Stream()
    manager = kv.KVCacheManager(config)
    caches = []
    try:
        key = manager.get_buffer_group_id(0, "key")
        value = manager.get_buffer_group_id(0, "value")
        implicit = manager.get_buffer_group_id(1, "key")
        assert len({key, value, implicit}) == 3
        assert manager.get_buffer_group_id(1, "value") == implicit
        with pytest.raises(RuntimeError, match="allocation groups"):
            manager.get_layer_group_id(0)
        for group, role in ((key, "key"), (value, "value")):
            assert [(entry.layer_id, entry.role) for entry in manager.buffer_grouping[group]] == [
                (0, role)
            ]
        cache = manager.create_kv_cache(input_tokens=[])
        caches.append(cache)
        assert cache.resume(stream.cuda_stream)
        assert cache.resize(8)
        key_pages = cache.get_base_page_indices(key)
        value_pages = cache.get_base_page_indices(value)
        # Equal layouts may share a physical pool, but each group owns distinct slots.
        pool_groups = manager.get_life_cycle_pool_group_indices()
        if pool_groups[key] == pool_groups[value]:
            assert set(key_pages[:2]).isdisjoint(value_pages[:2])
    finally:
        for cache in reversed(caches):
            cache.close()
        manager.shutdown()


@requires_cuda
@pytest.mark.parametrize("layer_type", [kv.AttentionLayerConfig, kv.SsmLayerConfig])
def test_explicit_fixed_group_ignores_layer_kind_and_stays_private(layer_type):
    config = kv.KVCacheManagerConfig(
        tokens_per_block=4,
        cache_tiers=[kv.GpuCacheTierConfig(quota=8 << 20)],
        layers=[
            kv.AttentionLayerConfig(layer_id=0, buffers=[kv.BufferConfig(role="key", size=4096)]),
            layer_type(
                layer_id=1,
                buffers=[kv.BufferConfig(role="state", size=4096, allocation_group_id=29)],
            ),
        ],
        allocation_groups=[
            kv.AllocationGroupConfig(
                29, kv.AllocationUnit.FIXED, reuse_policy=kv.ReusePolicy.PRIVATE
            )
        ],
        commit_min_snapshot=True,
    )
    stream = torch.cuda.Stream()
    manager = kv.KVCacheManager(config)
    caches = []
    try:
        fixed = manager.get_buffer_group_id(1, "state")
        source = manager.create_kv_cache(input_tokens=[])
        caches.append(source)
        assert source.resume(stream.cuda_stream)
        slot = source.get_ssm_block_base_index(fixed)
        assert slot >= 0
        assert source.resize(8)
        assert source.get_ssm_block_base_index(fixed) == slot
        source.commit(list(range(8)))

        reader = manager.create_kv_cache(input_tokens=list(range(8)))
        caches.append(reader)
        assert reader.num_committed_tokens == 8
        assert not reader.reuse_status[fixed].complete
        assert reader.resume(stream.cuda_stream)
        assert reader.get_ssm_block_base_index(fixed) != slot
        source.suspend()
        assert source.resume(stream.cuda_stream)
        assert source.get_ssm_block_base_index(fixed) == slot
    finally:
        for cache in reversed(caches):
            cache.close()
        manager.shutdown()
