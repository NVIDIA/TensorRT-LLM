# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local GPU QA for the unchanged native full-history + SWA reuse contract."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    MixedPrecisionKVCacheManager,
    Role,
)
from tensorrt_llm._utils import get_size_in_bytes
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime import kv_cache_manager_v2 as kv

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture
def manager(request):
    policy = request.param if isinstance(request.param, str) else "per_request"
    history_dtype, high_dtype = (
        (DataType.NVFP4, DataType.BF16) if isinstance(request.param, str) else request.param
    )
    manager = MixedPrecisionKVCacheManager(
        KvCacheConfig(
            max_tokens=65536,
            host_cache_size=0,
            enable_block_reuse=True,
            block_reuse_config={"policy": policy},
        ),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=2,
        head_dim=64,
        tokens_per_block=128,
        max_seq_len=1024,
        max_batch_size=2,
        mapping=Mapping(),
        dtype=high_dtype,
        low_dtype=history_dtype,
        compute_dtype=torch.bfloat16,
        high_low_pool_ratio=1 / 3,
        first_n=128,
        last_n=128,
    )
    yield manager
    manager.shutdown()


@pytest.mark.parametrize(
    "manager",
    [
        (DataType.NVFP4, DataType.FP8),
        (DataType.NVFP4, DataType.BF16),
        (DataType.NVFP4, DataType.HALF),
    ],
    indirect=True,
)
def test_extra_pool_uses_existing_dtype_layout(manager):
    """Allocation coverage, not a claim that every dtype has an Attention reader."""
    high = manager.kv_cache_manager_py_config.layers[1]
    for buffer in high.buffers:
        expected = get_size_in_bytes(2 * 64 * 128, manager.high_dtype)
        assert buffer.size == expected
        assert manager.impl.get_page_stride(high.layer_id, buffer.role) >= expected
    has_scales = any(buffer.role == Role.KEY_BLOCK_SCALE for buffer in high.buffers)
    assert has_scales == (manager.high_dtype == DataType.NVFP4)


@pytest.mark.parametrize("manager", ["per_request"], indirect=True)
def test_recurrent_state_layout_is_unchanged(manager):
    state = kv.SsmLayerConfig(
        layer_id=1,
        buffers=[
            kv.BufferConfig(role=kv.DataRole("ssm_state"), size=1024),
            kv.BufferConfig(role=kv.DataRole("conv_state"), size=256),
        ],
    )
    config = manager.kv_cache_manager_py_config
    with patch.object(manager, "high_layers", {}), patch.object(manager, "high_low_pool_ratio", 1):
        configured = manager._build_cache_config(
            replace(config, layers=[config.layers[0], state], initial_pool_ratio=[0.5, 0.5])
        )
        retained = configured.layers[1]
        assert isinstance(retained, kv.SsmLayerConfig)
        assert retained.layer_id == state.layer_id
        assert [(buffer.role, buffer.size) for buffer in retained.buffers] == [
            (buffer.role, buffer.size) for buffer in state.buffers
        ]
        assert manager.high_layers == {0: 2}
        assert len(configured.layers) == 3
        # Original KV:SSM=1:1, then high:low=1:1 within KV, never three equal shares.
        assert configured.initial_pool_ratio == pytest.approx([0.25, 0.5, 0.25])


@pytest.mark.parametrize("manager", ["per_request"], indirect=True)
def test_pool_ratio_splits_native_groups_not_individual_layers(manager):
    config = manager.kv_cache_manager_py_config
    key = kv.BufferConfig(role=Role.KEY, size=1024)
    layers = [kv.AttentionLayerConfig(layer_id=i, buffers=[key]) for i in range(2)]
    layers.append(kv.AttentionLayerConfig(layer_id=2, buffers=[key], sliding_window_size=513))
    high = [
        kv.AttentionLayerConfig(
            layer_id=i, buffers=[key], sliding_window_size=129, num_sink_tokens=128
        )
        for i in (3, 4)
    ]
    original = replace(config, layers=layers, initial_pool_ratio=[0.6, 0.4])
    with (
        patch.object(manager, "high_layers", {0: 3, 1: 4}),
        patch.object(manager, "high_low_pool_ratio", 2),
    ):
        ratio = manager._split_mixed_precision_pool_ratio(original, [*layers, *high])
    assert ratio == pytest.approx([0.2, 0.4, 0.4])


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


@pytest.mark.parametrize("family", ("dsa", "deepseek_v4"))
@pytest.mark.parametrize("high_dtype", (DataType.HALF, DataType.BF16, DataType.FP8))
def test_model_manager_layouts_keep_native_nvfp4_and_auxiliary_buffers(family, high_dtype):
    """Use the real model managers; this tests allocation, not model inference."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
        DeepseekV4CacheManager,
    )
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType,
    )
    from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import DSACacheManagerV2
    from tensorrt_llm.llmapi.llm_args import (
        DeepSeekSparseAttentionConfig,
        DeepSeekV4SparseAttentionConfig,
    )

    kwargs = dict(
        kv_cache_config=KvCacheConfig(
            max_gpu_total_bytes=128 << 20,
            host_cache_size=0,
            enable_block_reuse=True,
            enable_swa_scratch_reuse=False,
            block_reuse_config={"policy": "per_request"},
        ),
        kv_cache_type=CacheType.SELFKONLY,
        num_kv_heads=1,
        tokens_per_block=128,
        max_seq_len=1024,
        max_batch_size=2,
        max_num_tokens=1024,
        mapping=Mapping(),
    )
    if family == "dsa":
        manager_cls = DSACacheManagerV2
        kwargs.update(
            num_layers=1,
            head_dim=576,
            sparse_attention_config=DeepSeekSparseAttentionConfig(index_head_dim=128),
            pretrained_config=SimpleNamespace(qk_rope_head_dim=64, num_hidden_layers=1),
        )
    else:
        manager_cls = DeepseekV4CacheManager
        kwargs.update(
            num_layers=3,
            head_dim=512,
            vocab_size=1024,
            max_input_len=1024,
            sparse_attn_config=DeepSeekV4SparseAttentionConfig(
                index_head_dim=128, window_size=128, compress_ratios=[1, 4, 128]
            ),
        )

    def layout(layers):
        return [
            (
                int(layer.layer_id),
                [(buffer.role, buffer.size, buffer.is_sparse) for buffer in layer.buffers],
                layer.sliding_window_size,
                layer.num_sink_tokens,
                layer.reuse_policy,
            )
            for layer in layers
        ]

    reference = manager_cls(dtype=DataType.NVFP4, **kwargs)
    try:
        original = layout(reference.kv_cache_manager_py_config.layers)
    finally:
        reference.shutdown()

    mixed_cls = MixedPrecisionKVCacheManager.for_manager(manager_cls)
    manager = mixed_cls(
        dtype=high_dtype,
        low_dtype=DataType.NVFP4,
        compute_dtype=torch.bfloat16,
        first_n=128,
        last_n=128,
        **kwargs,
    )
    cache = None
    try:
        assert isinstance(manager, manager_cls)
        assert mixed_cls.prepare_context_cache is manager_cls.prepare_context_cache
        configured = manager.kv_cache_manager_py_config.layers
        assert layout(configured[: len(original)]) == original
        assert len(manager.high_layouts) == (1 if family == "dsa" else 2)
        cache = manager.impl.create_kv_cache(None, [])
        assert cache.resume(torch.cuda.current_stream().cuda_stream) and cache.resize(640)
        manager.kv_cache_map[0] = cache
        for layer_id, spec in manager.high_layouts.items():
            if family == "dsa":
                assert spec.roles == (Role.KEY,)
                assert spec.rows_per_block == 128
                assert spec.head_dim == 576
            else:
                assert spec.roles == (DeepseekV4AttentionType.COMPRESS.role,)
                assert spec.rows_per_block in (32, 1)
                assert spec.head_dim == 512
            high_id = manager.high_layers[layer_id]
            high = configured[high_id]
            assert high.reuse_policy == kv.AttentionReusePolicy.REQUIRED
            assert high.num_sink_tokens == 128 and high.sliding_window_size == 129
            for buffer in high.buffers:
                assert buffer.size == get_size_in_bytes(
                    spec.rows_per_block * spec.num_heads * spec.head_dim, high_dtype
                )
                pool, tables = manager.get_high_precision_rows(layer_id, buffer.role, [0])
                assert pool.shape[1:] == (spec.rows_per_block, spec.num_heads, spec.head_dim)
                assert len(tables[0]) == 5
                assert all(0 <= index < pool.shape[0] for index in tables[0])
    finally:
        manager.kv_cache_map.pop(0, None)
        if cache is not None:
            cache.close()
        manager.shutdown()
