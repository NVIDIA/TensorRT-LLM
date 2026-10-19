# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python bindings for native OPTIONAL selection; storage contracts live in C++."""

import pytest
import torch

from tensorrt_llm.runtime import kv_cache_manager_v2 as kv

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def test_reuse_status_and_optional_selection_bindings():
    layers = [
        kv.AttentionLayerConfig(
            layer_id=i,
            buffers=[kv.BufferConfig(role="key", size=4096)],
            sliding_window_size=None if i == 0 else 4 if i == 1 else 2,
            reuse_policy=kv.AttentionReusePolicy.OPTIONAL
            if i
            else kv.AttentionReusePolicy.REQUIRED,
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
        source = manager.create_kv_cache(None, [])
        caches.append(source)
        assert source.resume(stream.cuda_stream)
        assert source.resize(8)
        source.commit(tokens)
        reader = manager.create_kv_cache(None, tokens)
        caches.append(reader)
        groups = [manager.get_layer_group_id(i) for i in range(3)]
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
        assert reader.get_base_page_indices(encoder)[1] == source.get_base_page_indices(encoder)[1]
        assert reader.get_base_page_indices(peer)[1] != source.get_base_page_indices(peer)[1]
        reader.suspend()
        assert reader.resume(stream.cuda_stream)
        assert reader.get_base_page_indices(encoder)[1] == source.get_base_page_indices(encoder)[1]
    finally:
        for cache in reversed(caches):
            cache.close()
        manager.shutdown()
