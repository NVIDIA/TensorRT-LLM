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
from tensorrt_llm._utils import binding_to_torch_dtype, get_size_in_bytes
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


@pytest.mark.parametrize("dtype", (DataType.HALF, DataType.BF16, DataType.FP8))
def test_native_swa_and_shared_kv_keep_original_storage(dtype):
    """Only the ordinary full-history layer receives low/high KV pools."""
    config = KvCacheConfig(
        max_tokens=65536,
        enable_block_reuse=True,
        max_attention_window=[1024, 513, 1024, 1024],
        pool_ratio=[0.5, 0.5],
    )
    args = dict(
        num_layers=4,
        num_kv_heads=2,
        head_dim=64,
        tokens_per_block=128,
        max_seq_len=1024,
        max_batch_size=2,
        mapping=Mapping(),
        dtype=dtype,
        # The last layer reads full Attention KV from layer 2, as in Gemma4.
        pretrained_config=SimpleNamespace(
            num_hidden_layers=4,
            num_kv_shared_layers=1,
            layer_types=["full_attention", "sliding_attention", "full_attention", "full_attention"],
        ),
    )
    native = KVCacheManagerV2(config, CacheType.SELF, **args)
    mixed = None
    try:
        mixed = MixedPrecisionKVCacheManager(
            config,
            CacheType.SELF,
            compute_dtype=torch.bfloat16,
            high_low_pool_ratio=1,
            first_n=128,
            last_n=128,
            **args,
        )
        assert mixed.dtype == native.dtype == dtype
        assert mixed.high_layers == {0: 4}
        assert mixed.get_buffers(0).dtype == torch.uint8
        assert mixed.get_buffers(0).shape[-1] == 32
        assert mixed.get_cache_bytes_per_token() == sum(
            mixed.get_layer_bytes_per_token(layer, Role.ALL) for layer in range(4)
        )
        for layer in (1, 2, 3):
            original = native.kv_cache_manager_py_config.layers[layer]
            retained = mixed.kv_cache_manager_py_config.layers[layer]
            assert retained.sliding_window_size == original.sliding_window_size
            assert retained.reuse_policy == original.reuse_policy
            assert [(b.role, b.size) for b in retained.buffers] == [
                (b.role, b.size) for b in original.buffers
            ]
            buffer = mixed.get_buffers(layer)
            assert buffer.dtype == binding_to_torch_dtype(dtype)
            assert buffer.shape[1:] == native.get_buffers(layer).shape[1:]
    finally:
        if mixed is not None:
            mixed.shutdown()
        native.shutdown()


@pytest.mark.parametrize("manager", ["per_request"], indirect=True)
@pytest.mark.parametrize(
    "state",
    [
        kv.SsmLayerConfig(
            layer_id=1,
            buffers=[
                kv.BufferConfig(role=kv.DataRole("ssm_state"), size=1024),
                kv.BufferConfig(role=kv.DataRole("conv_state"), size=256),
            ],
        ),
        kv.AttentionLayerConfig(
            layer_id=1,
            buffers=[kv.BufferConfig(role=role, size=1024) for role in (Role.KEY, Role.VALUE)],
            sliding_window_size=513,
        ),
    ],
    ids=("recurrent", "native_swa"),
)
def test_native_state_layout_is_unchanged(manager, state):
    config = manager.kv_cache_manager_py_config
    with patch.object(manager, "high_layers", {}), patch.object(manager, "high_low_pool_ratio", 1):
        configured = manager._build_cache_config(
            replace(config, layers=[config.layers[0], state], initial_pool_ratio=[0.5, 0.5])
        )
        retained = configured.layers[1]
        assert type(retained) is type(state)
        assert retained.layer_id == state.layer_id
        assert [(buffer.role, buffer.size) for buffer in retained.buffers] == [
            (buffer.role, buffer.size) for buffer in state.buffers
        ]
        if isinstance(state, kv.AttentionLayerConfig):
            assert retained.sliding_window_size == state.sliding_window_size
            assert retained.num_sink_tokens == state.num_sink_tokens
            assert retained.reuse_policy == state.reuse_policy
        assert manager.high_layers == {0: 2}
        assert len(configured.layers) == 3
        # Split the full-history KV budget without changing the native state's share.
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
