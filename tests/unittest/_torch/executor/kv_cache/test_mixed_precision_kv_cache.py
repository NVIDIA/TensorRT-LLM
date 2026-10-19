# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for mixed-precision KV allocation, reuse and transfer."""

from contextlib import closing
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    MixedPrecisionKVCacheManager,
    Role,
)
from tensorrt_llm._utils import binding_to_torch_dtype
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime import kv_cache_manager_v2 as kv

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12),
    reason="Mixed-precision NVFP4 KV requires Blackwell",
)


@pytest.fixture
def make_manager():
    managers = []

    def create(manager_type=MixedPrecisionKVCacheManager, config=None, **kwargs):
        options = dict(
            num_layers=1,
            num_kv_heads=2,
            head_dim=64,
            tokens_per_block=128,
            max_seq_len=1024,
            max_batch_size=2,
            mapping=Mapping(),
            dtype=DataType.BF16,
        )
        if manager_type is MixedPrecisionKVCacheManager:
            options.update(first_n=1, last_n=65)
        options.update(kwargs)
        config = config or KvCacheConfig(
            max_tokens=65536, enable_block_reuse=True, block_reuse_config={"policy": "per_request"}
        )
        manager = manager_type(config, CacheType.SELF, **options)
        managers.append(manager)
        return manager

    yield create
    for manager in reversed(managers):
        manager.shutdown()


@pytest.mark.parametrize("dtype", (DataType.FP8, DataType.BF16, DataType.HALF))
def test_extra_pool_uses_existing_dtype_layout(make_manager, dtype):
    manager = make_manager(dtype=dtype)
    assert manager.high_low_pool_ratio == 1 / 5
    assert manager.kv_cache_manager_py_config.initial_pool_ratio == pytest.approx([5 / 6, 1 / 6])
    for pool, storage_dtype, width in (
        ("low", torch.uint8, 40),
        ("high", binding_to_torch_dtype(dtype), 64),
    ):
        nhd = manager.get_buffers(0, "NHD", mixed_precision_pool=pool)
        hnd = manager.get_buffers(0, "HND", mixed_precision_pool=pool)
        assert nhd.data_ptr() == hnd.data_ptr()
        assert nhd.dtype == hnd.dtype == storage_dtype
        assert nhd.shape[1:] == (2, 128, 2, width)
        assert hnd.shape[1:] == (2, 2, 128, width)


def test_mixed_precision_preserves_native_and_shared_kv_storage(make_manager):
    config = KvCacheConfig(
        max_tokens=65536,
        enable_block_reuse=True,
        max_attention_window=[1024, 513, 1024, 1024],
        pool_ratio=[0.5, 0.5],
    )
    options = dict(
        config=config,
        num_layers=4,
        pretrained_config=SimpleNamespace(
            num_hidden_layers=4,
            num_kv_shared_layers=1,
            layer_types=["full_attention", "sliding_attention", "full_attention", "full_attention"],
        ),
    )
    native = make_manager(KVCacheManagerV2, **options)
    mixed = make_manager(high_low_pool_ratio=1, **options)
    assert mixed.high_layers == {0: 4}
    for layer in (1, 2, 3):
        original = native.kv_cache_manager_py_config.layers[layer]
        retained = mixed.kv_cache_manager_py_config.layers[layer]
        assert retained.sliding_window_size == original.sliding_window_size
        assert retained.reuse_policy == original.reuse_policy
        assert [(b.role, b.size) for b in retained.buffers] == [
            (b.role, b.size) for b in original.buffers
        ]
        assert mixed.get_buffers(layer).dtype == native.get_buffers(layer).dtype
        assert mixed.get_buffers(layer).shape[1:] == native.get_buffers(layer).shape[1:]


@pytest.mark.parametrize(
    "window,sink,expected",
    [(513, 0, [0.2, 0.4, 0.4]), (129, 128, [0.2, 0.8]), (129, 1, [0.2, 0.8])],
)
def test_pool_ratio_splits_native_groups_not_individual_layers(
    make_manager, window, sink, expected
):
    manager = make_manager(high_low_pool_ratio=2)
    key = kv.BufferConfig(role=Role.KEY, size=1024)
    layers = [kv.AttentionLayerConfig(layer_id=i, buffers=[key]) for i in range(2)]
    layers.append(
        kv.AttentionLayerConfig(
            layer_id=2, buffers=[key], sliding_window_size=window, num_sink_tokens=sink
        )
    )
    original = replace(
        manager.kv_cache_manager_py_config, layers=layers, initial_pool_ratio=[0.6, 0.4]
    )
    high = [
        kv.AttentionLayerConfig(
            layer_id=i, buffers=[key], sliding_window_size=129, num_sink_tokens=128
        )
        for i in (3, 4)
    ]
    manager.high_layers = {0: 3, 1: 4}
    ratio = manager._split_mixed_precision_pool_ratio(original, [*layers, *high])
    assert ratio == pytest.approx(expected)


def test_native_common_reuse_and_window_retention(make_manager):
    manager = make_manager()
    config = manager.kv_cache_manager_py_config
    assert [layer.reuse_policy for layer in config.layers] == [kv.AttentionReusePolicy.REQUIRED] * 2
    assert (config.layers[1].num_sink_tokens, config.layers[1].sliding_window_size) == (1, 66)
    native = manager.impl
    stream = torch.cuda.current_stream().cuda_stream
    tokens = list(range(640))
    high_group = native.get_layer_group_id(manager.high_layers[0])
    with closing(native.create_kv_cache(None, [])) as source:
        assert source.resume(stream) and source.resize(len(tokens))
        source.commit(tokens)
        source.suspend()
        assert source.resume(stream, is_decoding=True)
        pages = list(source.get_base_page_indices(high_group))
        assert pages[0] >= 0 and pages[4] >= 0
        assert pages[1:4] == [kv.BAD_PAGE_INDEX] * 3
    with closing(native.create_kv_cache(None, tokens)) as full:
        assert full.num_committed_tokens == 640
        assert full.resume(stream)
    with closing(native.create_kv_cache(None, tokens[:512])) as shortened:
        # Reuse follows the native snapshot policy and required high tail.
        assert shortened.num_committed_tokens == (128 if config.commit_min_snapshot else 512)
        assert shortened.resume(stream)


def test_disagg_exports_both_pools_and_pairs_sink_tail(make_manager):
    import numpy as np

    from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
    from tensorrt_llm._torch.disaggregation.native.transfer import Sender
    from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1

    manager = make_manager()
    rank = RankInfo.from_kv_cache_manager("mixed", manager, device_id=0)
    table = rank.page_table
    assert [group.local_layers[0].global_layer_id for group in table.layer_groups] == [0, 1]
    assert [len(group.pool_views) for group in table.layer_groups] == [1, 1]
    registrar = PeerRegistrar(rank, KVRegionExtractorV1(table))
    peer = RankInfo.from_bytes(rank.to_bytes())
    peer.instance_name = "decode"
    assert registrar.get_pool_mapping(peer) == {(0, 0): (0, 0), (1, 0): (1, 0)}
    source, destination = Sender._pair_ordinals(
        np.array([4, -1, -1, -1, 7]), np.array([2, -1, -1, -1, 6])
    )
    assert source.tolist() == [4, 7]
    assert destination.tolist() == [2, 6]
