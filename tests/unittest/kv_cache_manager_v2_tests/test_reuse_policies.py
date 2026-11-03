# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from importlib.util import find_spec

import pytest
import torch

if find_spec("kv_cache_manager_v2") is not None:
    import kv_cache_manager_v2 as kv
else:
    from tensorrt_llm.runtime import kv_cache_manager_v2 as kv


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _config(alignment=1):
    return kv.KVCacheManagerConfig(
        tokens_per_block=4,
        cache_tiers=[kv.GpuCacheTierConfig(quota=16 << 20)],
        layers=[
            kv.AttentionLayerConfig(
                layer_id=0,
                buffers=[
                    kv.BufferConfig(role=role, size=size, allocation_group_id=group)
                    for group, role, size in (
                        (10, "key", 4096),
                        (20, "first", 8192),
                        (30, "second", 16384),
                        (40, "private", 2048),
                    )
                ],
            )
        ],
        allocation_groups=[
            kv.AllocationGroupConfig(10, kv.AllocationUnit.TOKEN_BLOCK, reuse_alignment=alignment),
            kv.AllocationGroupConfig(20, kv.AllocationUnit.FIXED),
            kv.AllocationGroupConfig(30, kv.AllocationUnit.FIXED),
            kv.AllocationGroupConfig(
                40,
                kv.AllocationUnit.FIXED,
                reuse_policy=kv.ReusePolicy.PRIVATE,
                reuse_alignment=101,
            ),
        ],
        commit_min_snapshot=True,
    )


def test_reuse_alignment_survives_config_copy():
    config = copy.deepcopy(_config(8))
    assert config.allocation_groups[0].reuse_alignment == 8
    assert config.allocation_groups[1].reuse_alignment == 1
    config.validate()


@pytest.mark.parametrize("alignment", [0, -1])
def test_reuse_alignment_must_be_positive(alignment):
    with pytest.raises(RuntimeError, match="reuse_alignment must be positive"):
        _config(alignment)


@requires_cuda
def test_selective_snapshots_require_a_common_published_endpoint():
    manager = kv.KVCacheManager(_config())
    stream = torch.cuda.Stream()
    caches = []
    try:
        first = manager.get_buffer_group_id(0, "first")
        second = manager.get_buffer_group_id(0, "second")
        private = manager.get_buffer_group_id(0, "private")
        source = manager.create_kv_cache(input_tokens=[])
        caches.append(source)
        assert source.resume(stream.cuda_stream)
        assert source.resize(20)
        source.commit(list(range(4)))
        source.commit(list(range(4, 8)), snapshot_groups=[first])
        source.commit(list(range(8, 12)), snapshot_groups=[second])
        source.commit(list(range(12, 16)), snapshot_groups=[])
        reader = manager.create_kv_cache(input_tokens=list(range(16)))
        caches.append(reader)
        assert reader.num_committed_tokens == 4
        assert reader.reuse_status[first].complete
        assert reader.reuse_status[second].complete
        assert not reader.reuse_status[private].complete
        assert reader.resume(stream.cuda_stream)
        reader.close()

        source.commit(list(range(16, 20)), snapshot_groups=[first, second], is_end=True)
        reader = manager.create_kv_cache(input_tokens=list(range(20)))
        caches.append(reader)
        assert reader.num_committed_tokens == 20
        assert reader.resume(stream.cuda_stream)
        source.commit([], is_end=True)
    finally:
        for cache in reversed(caches):
            cache.close()
        manager.shutdown()


@requires_cuda
def test_alignment_does_not_round_a_snapshot_to_an_unwritten_state():
    manager = kv.KVCacheManager(_config(4))
    stream = torch.cuda.Stream()
    caches = []
    try:
        source = manager.create_kv_cache(input_tokens=[])
        caches.append(source)
        assert source.resume(stream.cuda_stream)
        assert source.resize(11)
        source.commit(list(range(6)))
        reader = manager.create_kv_cache(input_tokens=list(range(6)))
        caches.append(reader)
        assert reader.num_committed_tokens == 0
        reader.close()
        source.commit(list(range(6, 8)))
        source.commit(list(range(8, 11)))
        reader = manager.create_kv_cache(input_tokens=list(range(11)))
        caches.append(reader)
        assert reader.num_committed_tokens == 8
    finally:
        for cache in reversed(caches):
            cache.close()
        manager.shutdown()


@requires_cuda
def test_invalid_snapshot_selection_leaves_progress_unchanged():
    manager = kv.KVCacheManager(_config())
    stream = torch.cuda.Stream()
    source = manager.create_kv_cache(input_tokens=[])
    try:
        first = manager.get_buffer_group_id(0, "first")
        key = manager.get_buffer_group_id(0, "key")
        private = manager.get_buffer_group_id(0, "private")
        assert source.resume(stream.cuda_stream)
        assert source.resize(4)
        for selection in ([first, first], [key], [private], [-1], [1000]):
            with pytest.raises(RuntimeError, match="snapshot_groups"):
                source.commit(list(range(4)), snapshot_groups=selection)
            assert source.num_committed_tokens == 0
            assert source.history_length == 0
        source.commit(list(range(4)), snapshot_groups=None)
        assert source.num_committed_tokens == 4
    finally:
        source.close()
        manager.shutdown()


@requires_cuda
def test_snapshot_selection_excludes_delayed_and_rebuilt_private_state():
    config = _config()
    groups = list(config.allocation_groups)
    groups[-1].initially_active = False
    groups[-1].suspend_policy = kv.SuspendPolicy.RELEASE
    config.allocation_groups = groups
    manager = kv.KVCacheManager(config)
    stream = torch.cuda.Stream()
    caches = []
    try:
        private = manager.get_buffer_group_id(0, "private")
        source = manager.create_kv_cache(input_tokens=[])
        caches.append(source)
        assert source.resume(stream.cuda_stream)
        assert source.resize(12)
        assert source.get_ssm_block_base_index(private) < 0
        source.commit(list(range(4)))
        assert source.activate_groups([40])
        with pytest.raises(RuntimeError, match="shareable fixed"):
            source.commit(list(range(4, 8)), snapshot_groups=[private])
        assert source.num_committed_tokens == 4
        source.commit(list(range(4, 8)))
        source.suspend()
        assert not source.is_active
        assert source.resume(stream.cuda_stream)
        assert source.get_ssm_block_base_index(private) >= 0
        source.commit(list(range(8, 12)))
        reader = manager.create_kv_cache(input_tokens=list(range(12)))
        caches.append(reader)
        assert reader.num_committed_tokens == 12
        assert reader.resume(stream.cuda_stream)
        assert reader.get_ssm_block_base_index(private) < 0
    finally:
        for cache in reversed(caches):
            cache.close()
        manager.shutdown()
