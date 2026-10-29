# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small real-manager checks for logical layers and physical KV ownership."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManagerV2
from tensorrt_llm._torch.attention.backends.sparse.mistral_large4 import MistralLarge4CacheManager
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as wrapper
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import (
    DeepSeekSparseAttentionConfig,
    DSparkDecodingConfig,
    KvCacheConfig,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import PageIndexMode

from .test_mistral_large4_cache_manager import _SHARING, _SOURCES, _model_config


def make_manager(
    kind: str,
    *,
    enable_stats: bool = False,
    spec_config: DSparkDecodingConfig | None = None,
    layer_mask: list[bool] | None = None,
) -> DSACacheManagerV2:
    """Use a few slots on device and host; no model weights are needed."""
    if kind == "ml4":
        cls = MistralLarge4CacheManager
        config = _model_config().pretrained_config
        num_layers = 72
    else:
        cls = DSACacheManagerV2
        num_layers = 4
        config = SimpleNamespace(num_hidden_layers=num_layers)
        if kind == "glm":
            config.index_topk_pattern = ["F", "S", "F", "S"]
    return cls(
        kv_cache_config=KvCacheConfig(
            max_tokens=256,
            host_cache_size=32 << 20,
            enable_block_reuse=True,
        ),
        kv_cache_type=CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=32,
        max_seq_len=128,
        max_num_tokens=64,
        max_batch_size=2,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.FP8,
        sparse_attention_config=DeepSeekSparseAttentionConfig(
            index_n_heads=32,
            index_head_dim=128,
            index_topk=2048,
        ),
        pretrained_config=config,
        enable_stats=enable_stats,
        spec_config=spec_config,
        layer_mask=layer_mask,
    )


@pytest.mark.parametrize("draft_len", [1, 3, 7])
@pytest.mark.parametrize("layer_mask", [None, [True] * 72])
def test_ml4_speculation_reserves_tokens_without_adding_target_layers(
    draft_len: int, layer_mask: list[bool] | None
) -> None:
    """Verification reserves draft capacity while preserving the shared latent layout."""
    manager = make_manager(
        "ml4", spec_config=DSparkDecodingConfig(max_draft_len=draft_len), layer_mask=layer_mask
    )
    try:
        assert manager._kv_reserve_draft_tokens == draft_len
        assert manager._generation_kv_capacity_headroom == 2 * draft_len
        assert manager.pp_layers == list(range(72))
        assert manager.get_cache_bytes_per_token() == 63 * 656 + 44 * 132
        assert manager.get_latent_pool().shape[1:] == (63, 32, 656)
        for layer, source in enumerate(_SOURCES):
            assert manager.latent_pool_row(layer) == manager.latent_pool_row(source)
            assert manager.has_buffer(layer, Role.KEY) == (layer == source)
    finally:
        manager.shutdown()


@pytest.mark.parametrize("layer_mask", [[True] * 71, [True] * 73, [False] + [True] * 71])
def test_ml4_rejects_masks_that_change_logical_target_layers(layer_mask: list[bool]) -> None:
    with pytest.raises(NotImplementedError, match="derives its layer mask"):
        make_manager("ml4", layer_mask=layer_mask)


@pytest.mark.parametrize("kind", ["deepseek", "glm", "ml4"])
def test_physical_roles_and_layer_views(kind: str) -> None:
    manager = make_manager(kind)
    try:
        ml4 = kind == "ml4"
        sources = _SOURCES if ml4 else list(range(4))
        owners = [i for i, source in enumerate(sources) if i == source]
        index_layers = (
            [i for i, role in enumerate(_SHARING) if role == "full"]
            if ml4
            else ([0, 2] if kind == "glm" else list(range(4)))
        )
        token_bytes = 656 if ml4 else 576
        pool = manager.get_unique_primary_pool()
        assert pool.shape == (manager.blocks_in_primary_pool, len(owners), 1, 32 * token_bytes)
        assert manager.pp_layers == list(range(len(sources)))
        assert (
            manager.get_cache_bytes_per_token()
            == len(owners) * token_bytes + len(index_layers) * 132
        )
        assert manager.indexer_k_cache_page_scale == len(index_layers)
        if ml4:
            assert manager.kv_owner_layers == owners
            assert manager.kv_source_layers == sources
            assert manager.get_latent_pool().stride(0) // 656 == 2016
        inventory = {(int(buf.layer_id), buf.role) for buf in manager.impl.all_buffer_ids}
        for layer, source in enumerate(sources):
            assert manager.has_buffer(layer, Role.KEY) == ((layer, Role.KEY) in inventory)
            assert manager.has_buffer(layer, Role.INDEX_KEY) == (
                (layer, Role.INDEX_KEY) in inventory
            )
            assert manager.has_buffer(layer, Role.KEY) == (layer in owners)
            assert manager.has_buffer(layer, Role.INDEX_KEY) == (layer in index_layers)
            if layer in owners:
                ordinal = owners.index(layer)
                assert manager.get_primary_pool_page_index_params(layer) == (len(owners), ordinal)
                view = manager.get_buffers(layer)
                assert view.data_ptr() - pool.data_ptr() == ordinal * 32 * token_bytes
                view.view(torch.uint8)[0, 0, 0, 0, 0] = layer
            else:
                assert manager.get_buffers(layer) is None
                assert manager._primary_pool_page_index_params[layer] is None
                with pytest.raises(ValueError, match="owns no"):
                    manager.get_layer_page_index_scale(layer)
                with pytest.raises(ValueError, match="owns no"):
                    manager.get_primary_pool_page_index_params(layer)
            if ml4:
                assert manager.latent_pool_row(layer) == owners.index(source)

        index_base = manager.impl.get_mem_pool_base_address(
            index_layers[0], Role.INDEX_KEY, PageIndexMode.SHARED
        )
        for ordinal, layer in enumerate(index_layers):
            view = manager.get_indexer_k_cache_buffers(layer)
            assert view.shape[1:] == (32, 1, 132)
            assert view.stride() == (32 * 132, 132, 132, 1)
            assert view.data_ptr() - index_base == ordinal * 32 * 132
            converter = manager.impl.get_page_index_converter(layer, Role.INDEX_KEY)
            assert (int(converter.scale), int(converter.layer_offset)) == (
                len(index_layers),
                ordinal,
            )
            view[0, 0, 0, 0] = layer
        for ordinal, layer in enumerate(owners):
            assert pool.view(torch.uint8)[0, ordinal, 0, 0].item() == layer
        for layer in index_layers:
            assert manager.get_indexer_k_cache_buffers(layer)[0, 0, 0, 0].item() == layer
        if ml4:
            assert manager.get_latent_pool()[0, manager.latent_pool_row(39), 0, 0].item() == 35
            assert manager.get_indexer_k_cache_buffers(39)[0, 0, 0, 0].item() == 39
            assert manager.get_indexer_k_cache_buffers(35)[0, 0, 0, 0].item() == 35

        # The existing batch API reads stable slots in request order.
        slots = [manager.index_mapper.add_new_sequence(i) for i in (101, 202)]
        try:
            for slot, blocks in zip(slots, ([1, 2, 0, 0], [2, 3, 0, 0])):
                manager.host_kv_cache_block_offsets[0, slot, 0, :4] = torch.tensor(blocks)
            torch.testing.assert_close(
                manager.get_pool_block_indices(2, request_ids=[202, 101], num_contexts=2)[:, :4],
                torch.tensor([[2, 3, 0, 0], [1, 2, 0, 0]], dtype=torch.int32),
            )
        finally:
            for request_id in (101, 202):
                manager.index_mapper.remove_sequence(request_id)
    finally:
        manager.shutdown()


def test_ml4_host_slots_preserve_latent_and_index_pages() -> None:
    """Force real host transfers and check every byte of both physical pools."""
    manager = make_manager("ml4", enable_stats=True)
    cache = manager.impl.create_kv_cache()
    pressure = None
    try:
        stream = torch.cuda.current_stream().cuda_stream
        assert cache.resume(stream)
        cache.capacity = 192
        blocks = list(cache.get_base_page_indices(0))
        assert len(blocks) == 6
        latent = manager.get_latent_pool()
        indexes = [i for i, role in enumerate(_SHARING) if role == "full"]
        expected = []
        for ordinal, block in enumerate(blocks):
            latent[block].copy_(
                (torch.arange(latent[block].numel(), device="cuda") + ordinal)
                .to(torch.uint8)
                .view_as(latent[block])
            )
            expected.append(latent[block].clone())
            for layer in indexes:
                manager.get_indexer_k_cache_buffers(layer)[block * 44].fill_(layer + ordinal)
        torch.cuda.synchronize()
        cache.commit(list(range(192)))
        cache.suspend()
        manager.impl.get_and_reset_iteration_stats()
        # Another live request forces suspended pages into the host tier.
        pressure = manager.impl.create_kv_cache()
        assert pressure.resume(stream)
        pressure.capacity = 192
        offload_stats = manager.impl.get_and_reset_iteration_stats()
        offload = dict(
            iterOffloadBlocks=sum(s.iter_offload_blocks for s in offload_stats.values()),
            iterOffloadBytes=sum(s.iter_offload_bytes for s in offload_stats.values()),
        )
        assert offload["iterOffloadBlocks"] > 0
        assert offload["iterOffloadBytes"] == offload["iterOffloadBlocks"] * 32 * 47136
        pressure.close()
        pressure = None
        assert cache.resume(stream)
        torch.cuda.synchronize()
        onboard_stats = manager.impl.get_and_reset_iteration_stats()
        onboard = dict(
            iterOnboardBlocks=sum(s.iter_onboard_blocks for s in onboard_stats.values()),
            iterOnboardBytes=sum(s.iter_onboard_bytes for s in onboard_stats.values()),
        )
        assert onboard["iterOnboardBlocks"] > 0
        assert onboard["iterOnboardBytes"] == onboard["iterOnboardBlocks"] * 32 * 47136
        for ordinal, block in enumerate(cache.get_base_page_indices(0)):
            torch.testing.assert_close(latent[block], expected[ordinal])
            for layer in indexes:
                page = manager.get_indexer_k_cache_buffers(layer)[block * 44]
                assert torch.all(page == layer + ordinal).item()
        print("ML4_HOST_SLOT_ROUNDTRIP", dict(offload=offload, onboard=onboard))
    finally:
        if pressure is not None:
            pressure.close()
        cache.close()
        manager.shutdown()


def test_ml4_pool_rows_follow_physical_order(monkeypatch) -> None:
    """Swap real native allocations without changing the model's logical layers."""
    native_manager = wrapper.KVCacheManagerPy

    def swapped_manager(config, **kwargs):
        layers = list(config.layers)
        layers[0], layers[1] = layers[1], layers[0]
        return native_manager(replace(config, layers=layers), **kwargs)

    monkeypatch.setattr(wrapper, "KVCacheManagerPy", swapped_manager)
    manager = make_manager("ml4")
    try:
        pool = manager.get_latent_pool()
        assert manager.kv_owner_layers[:2] == [0, 1]
        assert manager.latent_pool_row(0) == 1
        assert manager.latent_pool_row(1) == 0
        assert pool.is_contiguous()
        assert pool.stride(0) // 656 == 2016
        for layer in manager.kv_owner_layers:
            row = manager.latent_pool_row(layer)
            view = manager.get_buffers(layer)
            assert view.data_ptr() - pool.data_ptr() == row * 32 * 656
            view.view(torch.uint8)[0, 0, 0, 0, 0] = layer
        for layer, source in enumerate(manager.kv_source_layers):
            assert pool[0, manager.latent_pool_row(layer), 0, 0].item() == source
        assert manager.latent_pool_row(39) == manager.latent_pool_row(35)
    finally:
        manager.shutdown()
