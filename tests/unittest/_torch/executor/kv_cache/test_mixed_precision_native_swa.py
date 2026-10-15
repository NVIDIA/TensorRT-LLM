# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local GPU QA for the unchanged native full-history + SWA reuse contract."""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mixed_precision_cache import (
    MixedPrecisionKVCacheManager,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime import kv_cache_manager_v2 as kv

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture
def manager(request):
    manager = MixedPrecisionKVCacheManager(
        KvCacheConfig(
            max_tokens=16384,
            host_cache_size=0,
            enable_block_reuse=True,
            pool_ratio=[0.75, 0.25],
            block_reuse_config={"policy": request.param},
        ),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=2,
        head_dim=64,
        tokens_per_block=128,
        max_seq_len=1024,
        max_batch_size=2,
        mapping=Mapping(),
        dtype=DataType.FP8,
        first_n=128,
        last_n=128,
    )
    yield manager
    manager.shutdown()


@pytest.mark.parametrize("manager", ["all_reusable", "per_request"], indirect=True)
def test_native_common_reuse_and_window_retention(manager):
    config = manager.kv_cache_manager_py_config
    assert [layer.reuse_policy for layer in config.layers] == [kv.AttentionReusePolicy.REQUIRED] * 2
    assert config.layers[1].num_sink_tokens == 128
    assert config.layers[1].sliding_window_size == 129
    assert (
        MixedPrecisionKVCacheManager.prepare_context_cache is KVCacheManagerV2.prepare_context_cache
    )
    assert (
        MixedPrecisionKVCacheManager._resume_context_cache is KVCacheManagerV2._resume_context_cache
    )
    assert MixedPrecisionKVCacheManager.update_resources is KVCacheManagerV2.update_resources
    assert [layer.layer_id for layer in manager._cold_page_codec_config(config).layers] == [0]

    native = manager.impl
    stream = torch.cuda.current_stream().cuda_stream
    tokens = list(range(640))
    caches = []
    try:
        source = native.create_kv_cache(None, [])
        caches.append(source)
        assert source.resume(stream) and source.resize(len(tokens))
        source.commit(tokens)
        source.suspend()
        # Native SWA discards middle working mappings, but keeps sink and tail.
        high_group = native.get_layer_group_id(manager.high_layers[0])
        assert source.resume(stream, is_decoding=True)
        pages = list(source.get_base_page_indices(high_group))
        assert pages[0] >= 0 and pages[4] >= 0
        assert pages[1:4] == [kv.BAD_PAGE_INDEX] * 3
        source.close()
        caches.remove(source)

        full = native.create_kv_cache(None, tokens)
        caches.append(full)
        assert full.num_committed_tokens == 640
        assert full.resume(stream)
        full.close()
        caches.remove(full)

        # This endpoint needs a different high tail. No Python rematch or replay.
        shortened = native.create_kv_cache(None, tokens[:512])
        caches.append(shortened)
        expected = 128 if config.commit_min_snapshot else 512
        assert shortened.num_committed_tokens == expected
        assert shortened.resume(stream)
    finally:
        for cache in reversed(caches):
            cache.close()


@pytest.mark.parametrize("manager", ["per_request"], indirect=True)
def test_short_sequence_uses_high_values_for_every_page(manager):
    native = manager.impl
    cache = native.create_kv_cache(None, [])
    try:
        assert cache.resume(torch.cuda.current_stream().cuda_stream) and cache.resize(192)
        cache.commit(list(range(192)))
        group = native.get_layer_group_id(manager.high_layers[0])
        assert all(page >= 0 for page in cache.get_base_page_indices(group))
    finally:
        cache.close()
