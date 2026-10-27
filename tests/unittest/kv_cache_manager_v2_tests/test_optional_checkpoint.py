# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python bindings for native OPTIONAL selection; storage contracts live in C++."""

import time
from datetime import timedelta

import pytest
import torch

from tensorrt_llm.bindings.executor import KvCacheRetentionConfig
from tensorrt_llm.runtime import kv_cache_manager_v2 as kv

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("priority", [None, 80])
def test_reuse_status_and_optional_selection_bindings(priority):
    layers = [
        kv.AttentionLayerConfig(
            layer_id=i,
            buffers=[kv.BufferConfig(role="key", size=4096)],
            sliding_window_size=None if i == 0 else 4 if i == 1 else 2,
            reuse_policy=kv.ReusePolicy.OPTIONAL if i else kv.ReusePolicy.REQUIRED,
        )
        for i in range(3)
    ]
    config = kv.KVCacheManagerConfig(
        tokens_per_block=4, cache_tiers=[kv.GpuCacheTierConfig(quota=4 << 20)], layers=layers
    )
    stream = torch.cuda.Stream()
    manager = kv.KVCacheManager(config)
    caches = []
    tokens = list(range(8))
    try:
        source = manager.create_kv_cache(
            None,
            [],
            expected_prompt_length=8,
            kv_cache_retention_config=KvCacheRetentionConfig(
                [
                    KvCacheRetentionConfig.TokenRangeRetentionConfig(
                        0, None, 60, timedelta(seconds=1)
                    )
                ]
            ),
        )
        caches.append(source)
        assert source.resume(stream.cuda_stream)
        assert source.resize(8)
        source.commit(tokens)
        groups = [manager.get_layer_group_id(i) for i in range(3)]
        source_indices = [source.get_base_page_indices(group).copy() for group in groups]
        source.close()
        retention = (
            KvCacheRetentionConfig(
                [KvCacheRetentionConfig.TokenRangeRetentionConfig(0, None, priority)]
            )
            if priority is not None
            else None
        )
        reader = manager.create_kv_cache(None, tokens, kv_cache_retention_config=retention)
        caches.append(reader)
        global_group, encoder, peer = groups
        for layer, group in enumerate(groups):
            status = reader.reuse_status[group]
            assert status.group_id == group
            assert status.policy == config.layers[layer].reuse_policy
            assert status.complete and status.endpoint == 8
            assert list(status.coverage) == ([(0, 8)] if layer == 0 else [(4, 8)])
        for invalid in ([global_group], [-1], [len(reader.reuse_status)]):
            with pytest.raises(ValueError):
                reader.resume(stream.cuda_stream, optional_reuse_groups=invalid)
            assert not reader.is_active
        assert reader.resume(stream.cuda_stream, optional_reuse_groups=[encoder])
        assert reader.get_base_page_indices(encoder)[1] == source_indices[1][1]
        assert reader.get_base_page_indices(peer)[1] != source_indices[2][1]
        assert reader.get_page_priorities(encoder)[1] == (60 if priority is None else priority)
        assert reader.get_page_priorities(peer)[1] == (35 if priority is None else priority)
        time.sleep(1.05)
        manager.refresh_retention()
        assert reader.get_page_priorities(encoder)[1] == (60 if priority is None else priority)
        reader.suspend()
        assert reader.resume(stream.cuda_stream)
        assert reader.get_base_page_indices(encoder)[1] == source_indices[1][1]
    finally:
        for cache in reversed(caches):
            cache.close()
        manager.shutdown()


@pytest.mark.parametrize("length", [7, 8])
@pytest.mark.parametrize("select_optional", [False, True])
def test_ssm_layer_reuse_policies(length, select_optional):
    layers = [
        kv.AttentionLayerConfig(layer_id=0, buffers=[kv.BufferConfig(role="key", size=4096)]),
        *[
            kv.SsmLayerConfig(
                layer_id=i,
                buffers=[
                    kv.BufferConfig(role="state", size=4096),
                    kv.BufferConfig(role="conv", size=4096),
                ],
                reuse_policy=policy,
            )
            for i, policy in enumerate(
                [kv.ReusePolicy.REQUIRED, kv.ReusePolicy.OPTIONAL, kv.ReusePolicy.PRIVATE],
                start=1,
            )
        ],
    ]
    config = kv.KVCacheManagerConfig(
        tokens_per_block=4,
        cache_tiers=[kv.GpuCacheTierConfig(quota=8 << 20)],
        layers=layers,
        commit_min_snapshot=True,
    )
    manager = kv.KVCacheManager(config)
    stream = torch.cuda.Stream()
    tokens = list(range(length))
    caches = []
    try:
        source = manager.create_kv_cache(input_tokens=[])
        caches.append(source)
        assert source.resume(stream.cuda_stream)
        assert source.resize(length)
        source.commit(tokens, is_end=True)
        source.close()
        required, optional, private = [manager.get_layer_group_id(i) for i in (1, 2, 3)]
        assert len({required, optional, private}) == 3
        reader = manager.create_kv_cache(input_tokens=tokens)
        caches.append(reader)
        assert reader.num_committed_tokens == length
        assert reader.reuse_status[required].policy == kv.ReusePolicy.REQUIRED
        assert reader.reuse_status[optional].complete
        assert list(reader.reuse_status[optional].coverage) == [(length, length)]
        assert not reader.reuse_status[private].complete
        assert reader.resume(
            stream.cuda_stream, optional_reuse_groups=[optional] if select_optional else []
        )
        indices = [
            reader.get_ssm_block_base_index(group) for group in (required, optional, private)
        ]
        assert all(index >= 0 for index in indices)
        reader.suspend()
        assert reader.resume(stream.cuda_stream)
        assert indices == [
            reader.get_ssm_block_base_index(group) for group in (required, optional, private)
        ]
    finally:
        for cache in reversed(caches):
            cache.close()
        manager.shutdown()


@pytest.mark.parametrize("length", [7, 8])
def test_ssm_buffer_reuse_policies(length):
    replay = kv.BufferConfig(role="old_x", size=2048, reuse_policy=kv.ReusePolicy.PRIVATE)
    assert replay.reuse_policy == kv.ReusePolicy.PRIVATE
    assert kv.BufferConfig(role="state", size=4096).reuse_policy is None
    layers = [
        kv.AttentionLayerConfig(layer_id=0, buffers=[kv.BufferConfig(role="key", size=4096)]),
        *[
            kv.SsmLayerConfig(
                layer_id=i,
                buffers=[
                    kv.BufferConfig(role="state", size=4096),
                    kv.BufferConfig(role="conv", size=4096),
                    kv.BufferConfig(role="old_x", size=2048, reuse_policy=kv.ReusePolicy.PRIVATE),
                ],
            )
            for i in (1, 2)
        ],
    ]
    config = kv.KVCacheManagerConfig(
        tokens_per_block=4,
        cache_tiers=[kv.GpuCacheTierConfig(quota=8 << 20)],
        layers=layers,
        commit_min_snapshot=True,
    )
    assert config.layers[1].buffers[2].reuse_policy == kv.ReusePolicy.PRIVATE
    manager = kv.KVCacheManager(config)
    stream = torch.cuda.Stream()
    tokens = list(range(length))
    caches = []
    try:
        required = manager.get_layer_group_id(1, "state")
        private = manager.get_layer_group_id(layer_id=1, role="old_x")
        assert required != private
        assert manager.get_layer_group_id(2, "conv") == required
        assert manager.get_layer_group_id(2, "old_x") == private
        assert manager.get_layer_group_id(0) == manager.get_layer_group_id(0, "key")
        for layer_id in (1, 2):
            with pytest.raises(ValueError, match="layer groups"):
                manager.get_layer_group_id(layer_id)
        assert manager.num_layers == 3
        assert sorted(manager.layer_ids) == [0, 1, 2]
        assert sorted(manager.layer_grouping[required]) == [1, 2]
        assert sorted(manager.layer_grouping[private]) == [1, 2]

        source = manager.create_kv_cache(input_tokens=[])
        caches.append(source)
        assert source.resume(stream.cuda_stream)
        assert source.resize(length)
        source.commit(tokens, is_end=True)
        source.close()
        reader = manager.create_kv_cache(input_tokens=tokens)
        caches.append(reader)
        assert reader.num_committed_tokens == length
        assert reader.reuse_status[required].complete
        assert not reader.reuse_status[private].complete
        assert reader.resume(stream.cuda_stream)
        indices = [reader.get_ssm_block_base_index(group) for group in (required, private)]
        assert all(index >= 0 for index in indices)
        reader.suspend()
        assert reader.resume(stream.cuda_stream)
        assert indices == [reader.get_ssm_block_base_index(group) for group in (required, private)]
    finally:
        for cache in reversed(caches):
            cache.close()
        manager.shutdown()


@pytest.mark.parametrize("policy", [kv.ReusePolicy.PRIVATE, kv.ReusePolicy.OPTIONAL])
def test_attention_buffer_reuse_policy_must_match_layer(policy):
    def make_config(buffer_policy):
        return kv.KVCacheManagerConfig(
            tokens_per_block=4,
            cache_tiers=[kv.GpuCacheTierConfig(quota=4 << 20)],
            layers=[
                kv.AttentionLayerConfig(
                    layer_id=0,
                    buffers=[kv.BufferConfig(role="key", size=4096, reuse_policy=buffer_policy)],
                )
            ],
        )

    make_config(kv.ReusePolicy.REQUIRED)
    with pytest.raises(ValueError, match="only supported for SSM layers"):
        make_config(policy)
