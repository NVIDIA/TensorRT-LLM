# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physical SWA pruning, cache sizing and logical-layer compatibility."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2 import cache_manager as cache_module
from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
    CSA2CacheManager,
    CSA2CacheRole,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX


def _layout() -> CSA2Layout:
    return CSA2Layout(
        (0, 0) + (2,) * 18 + (1,) * 20,
        (2, 8, 14, 20),
        (2, 8, 14, 20, 24, 28, 32, 36),
        candidate_source_layer_id=20,
    )


def _declarative_manager(limit: int | None) -> CSA2CacheManager:
    manager = object.__new__(CSA2CacheManager)
    manager.context_swa_layer_limit = limit
    manager.layout = _layout()
    manager.pp_layers = list(range(40))
    manager.tokens_per_block = 128
    manager.max_draft_len = manager.reuse_match_backoff = 0
    manager._decoder_replay_window = 0
    manager._encoder_replay_enabled = False
    manager._build_cache_config(SimpleNamespace(swa_scratch_reuse=None))
    return manager


@pytest.mark.cpu_only
@pytest.mark.parametrize("limit", [0, -1, 40, 41, 20.5, True, "20"])
def test_invalid_context_swa_limit_fails_before_allocation(limit) -> None:
    with pytest.raises(ValueError, match="context SWA layer limit"):
        CSA2CacheManager(
            KvCacheConfig(dtype="fp8"),
            CacheType.SELFKONLY,
            num_layers=40,
            tokens_per_block=128,
            mapping=Mapping(),
            layout=_layout(),
            context_swa_layer_limit=limit,
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("limit", [None, 20])
def test_prune_only_physical_swa_groups(limit: int | None) -> None:
    manager = _declarative_manager(limit)
    built = manager._build_cache_config(SimpleNamespace(swa_scratch_reuse=None))
    retained = 40 if limit is None else limit
    assert manager.pp_layers == list(range(40))
    assert len(built.layers) == retained + 7
    identities = set(manager._physical_roles.values())
    assert {layer for layer, role in identities if role == CSA2CacheRole.SWA} == set(
        range(retained)
    )
    for role in (CSA2CacheRole.GLOBAL, CSA2CacheRole.INDEX):
        assert {layer for layer, item in identities if item == role} == {2, 8, 14, 20}
        assert manager._layer_roles[39, role] == manager._layer_roles[20, role]
    for role in (CSA2CacheRole.COMPRESSOR_KV, CSA2CacheRole.COMPRESSOR_SCORE):
        assert {layer for layer, item in identities if item == role} == {2, 8, 14}
    sizes, windows = manager._get_runtime_cache_size_layer_components()
    assert sizes.count(528) == retained
    assert sum(size for size, window in zip(sizes, windows) if window is None) == 890
    assert sizes.count(4096) == 3
    assert (
        sum(buffer.size for layer in built.layers for buffer in layer.buffers) == sum(sizes) * 128
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("tokens_per_block", [128, 256])
def test_static_sizing_removes_only_swa_reservation(tokens_per_block: int) -> None:
    layout = _layout()
    text_config = SimpleNamespace(
        compress_ratios=layout.compress_ratios,
        kv_source_layer_ids=layout.kv_source_layer_ids,
        index_source_layer_ids=layout.index_source_layer_ids,
        candidate_source_layer_id=layout.candidate_source_layer_id,
        candidate_topk_blocks=layout.candidate_topk_blocks,
        candidate_block_size=layout.candidate_block_size,
        index_topk=layout.index_topk,
        sliding_window=layout.window_size,
    )
    model = SimpleNamespace(pretrained_config=text_config, get_num_attention_layers=lambda: 40)
    full = CSA2CacheManager.get_cache_size_per_token(
        model, Mapping(), tokens_per_block=tokens_per_block, max_batch_size=32
    )
    model.extra_attrs = {"csa2_context_swa_layer_limit": 20}
    pruned = CSA2CacheManager.get_cache_size_per_token(
        model, Mapping(), tokens_per_block=tokens_per_block, max_batch_size=32
    )
    # The GLOBAL token slope stays fixed. Every omitted SWA layer had two
    # pages reserved per request, including the page-boundary headroom.
    assert full[0] == pruned[0] == 890
    assert full[1] - pruned[1] == 20 * 2 * tokens_per_block * 528 * 32


@pytest.mark.cpu_only
def test_logical_page_tables_and_null_pointers(monkeypatch) -> None:
    manager = _declarative_manager(20)
    manager.num_local_layers = 40
    manager.num_pools = 3
    manager.max_batch_size = 2
    manager.max_beam_width = 1
    manager.max_blocks_per_seq = 8
    manager.impl = SimpleNamespace(get_mem_pool_base_address=lambda *args: 4096 + int(args[0]))
    monkeypatch.setattr(cache_module, "prefer_pinned", lambda: False)
    # The device page-table plan needs real converters; pointers do not.
    monkeypatch.setattr(manager, "_init_batch_page_plan", lambda: None)
    manager._prepare_page_table_tensor(2)
    assert manager.kv_cache_pool_pointers.shape == (40, 2)
    assert torch.all(manager.kv_cache_pool_pointers[:20, 0] != 0)
    assert torch.count_nonzero(manager.kv_cache_pool_pointers[20:]) == 0
    assert manager.kv_cache_pool_mapping[:, 0].tolist() == list(range(40))
    with pytest.raises(ValueError, match="has no SWA cache"):
        manager.get_buffers(20)
    with pytest.raises(ValueError, match="has no SWA cache"):
        manager.get_cache_indices(1, 39, CSA2CacheRole.SWA)

    def compute(request_ids, num_contexts):
        # Stands in for the device conversion: SWA tables of the retained layers.
        tables = [
            [[r + layer, 88] + [BAD_PAGE_INDEX] * 6 for r in request_ids] for layer in range(20)
        ]
        manager._batch_page_tables = torch.tensor(tables, dtype=torch.int32)

    manager.compute_batch_page_tables = compute
    destination = torch.full((40, 2, 2, 8), 99, dtype=torch.int32)
    manager._swa_publication_token = manager._swa_publication = None
    manager.copy_batch_block_offsets(destination, [1], 1, 1, 1)
    assert destination[:20, 0, 0, 0].tolist() == list(range(1, 21))
    assert torch.all(destination[20:] == BAD_PAGE_INDEX)


@pytest.mark.cpu_only
def test_fresh_fill_uses_physical_roles_and_preserves_relocated_committed_pages(
    monkeypatch,
) -> None:
    manager = _declarative_manager(20)
    manager._fresh_page_fill = 0
    manager._fresh_fill_announced = True
    manager._fresh_pages_filled = {}
    manager.kv_cache_map = {1: SimpleNamespace(num_committed_tokens=1)}
    # Model layer 20's GLOBAL role remains, despite no SWA at that layer.
    buffers = {
        identity: torch.full((16, 1, 4), 7, dtype=torch.uint8)
        for identity in manager._physical_roles.values()
    }
    page_tables = {identity: [1, 2, 3, 4, BAD_PAGE_INDEX, BAD_PAGE_INDEX] for identity in buffers}
    manager.get_buffers = lambda layer, role: buffers[layer, role]
    manager.get_cache_indices = lambda request, layer, role: page_tables[layer, role]
    manager._get_page_index_converter = lambda *args: SimpleNamespace(expansion=2)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    manager._fill_fresh_kv_pages(1)
    for buffer in buffers.values():
        assert torch.all(buffer[[1, 2]] == 7)  # ceil(committed / 128) * expansion
        assert torch.all(buffer[[3, 4]] == 0)
        buffer.fill_(9)
    for identity in page_tables:
        page_tables[identity] = [1, 2, 8, 9, 10, 11]
    manager._fill_fresh_kv_pages(1)
    for buffer in buffers.values():
        assert torch.all(buffer[[1, 2, 8, 9]] == 9)  # protected or relocated
        assert torch.all(buffer[[10, 11]] == 0)  # newly materialized ordinals


@pytest.mark.parametrize("guard", [False, True])
def test_real_context_allocator_omits_decoder_swa(monkeypatch, guard: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CSA2 runtime cache storage requires CUDA")
    monkeypatch.setenv("TRTLLM_KV_GUARD_PAGE", "nan" if guard else "")
    monkeypatch.setenv("TRTLLM_KV_FRESH_PAGE_FILL", "zero")
    manager = CSA2CacheManager(
        KvCacheConfig(
            max_gpu_total_bytes=128 << 20, host_cache_size=0, enable_block_reuse=False, dtype="fp8"
        ),
        CacheType.SELFKONLY,
        num_layers=40,
        tokens_per_block=128,
        max_seq_len=2048,
        max_batch_size=2,
        max_num_tokens=1024,
        mapping=Mapping(),
        dtype=DataType.BF16,
        vocab_size=8192,
        layout=_layout(),
        context_swa_layer_limit=20,
    )
    try:
        assert manager.num_local_layers == 40
        assert len(manager.impl.init_config.layers) == 27
        assert len(list(manager.impl.all_buffer_ids)) == 34
        assert torch.all(manager.kv_cache_pool_pointers[20:] == 0)
        assert set(manager._guard_page_by_layer) == (set(range(20)) if guard else set())
        assert manager.get_num_free_blocks() > 0
        assert manager.blocks_in_primary_pool > 0
        quota = manager._get_quota_from_max_tokens(1024)
        assert manager._get_max_tokens_from_quota(quota) == pytest.approx(1024)
        cache = manager._create_kv_cache(90, None, [])
        assert cache is not None and manager._resume_and_restore(90, cache)
        assert cache.resize(513)
        manager._fill_fresh_kv_pages(90)
        for owner in (2, 8, 14, 20):
            assert manager.get_cache_indices(90, owner, CSA2CacheRole.GLOBAL)
            assert manager.get_buffers(owner, CSA2CacheRole.GLOBAL).numel() > 0
        for layer in range(20, 40):
            with pytest.raises(ValueError, match="has no SWA cache"):
                manager.get_buffers(layer)
        destination = torch.empty(
            (40, 2, 2, manager.max_blocks_per_seq), dtype=torch.int32, device="cuda"
        )
        manager.copy_batch_block_offsets(destination, [90], 1, 1, 1)
        assert torch.all(destination[20:] == BAD_PAGE_INDEX)
        assert sum(role == CSA2CacheRole.SWA for _, role in manager._batch_page_index) == 20
        manager.check_invalid_values_in_kv_cache(fill_with_zero=True)
        assert not manager.check_invalid_values_in_kv_cache()
    finally:
        for request_id in list(manager.kv_cache_map):
            manager.free_resources(SimpleNamespace(py_request_id=request_id))
        manager.shutdown()
