# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 index-K cache formats in the V2 pool: config, page views, capacity and disagg."""

from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    INDEX_KEY_NVFP4,
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.index_k_nvfp4 import (
    INDEX_K_NVFP4_PAGE_BYTES,
)
from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import AttentionPolicy
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import build_page_table_from_manager
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.disaggregation.resource.utils import get_layer_byte_ranges
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import (
    Eagle3DecodingConfig,
    KvCacheConfig,
    MiniMaxM3SparseAttentionConfig,
)
from tensorrt_llm.mapping import Mapping

# MiniMax-M3's target layers as one TP2 rank holds them: layers 0-2 dense
# (FP8 K/V), layers 3-59 sparse (NVFP4 K/V), two KV heads per layer.
NUM_LAYERS = 60
SPARSE_LAYERS = list(range(3, NUM_LAYERS))
NUM_KV_HEADS = 2
HEAD_DIM = 128
TOKENS_PER_BLOCK = 128
# 128 rows of 64 E2M1 data bytes, then 128 rows of 8 E4M3 scale bytes.
NVFP4_PAGE_BYTES = 9216

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA cache pools"
)


def _make_manager(
    indexer_kv_dtype: str,
    gpu_bytes: int = 256 << 20,
    *,
    eagle3: bool = False,
    mapping: Optional[Mapping] = None,
) -> MiniMaxM3KVCacheManagerV2:
    torch.cuda.init()
    return MiniMaxM3KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=gpu_bytes,
            host_cache_size=0,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=256,
        max_batch_size=2,
        max_num_tokens=256,
        mapping=mapping or Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.NVFP4,
        vocab_size=16,
        # Eagle3 appends one shared draft layer (FP8 K/V, no index-K) after the target layers.
        spec_config=(
            Eagle3DecodingConfig(max_draft_len=3, speculative_model="unused") if eagle3 else None
        ),
        sparse_attention_config=MiniMaxM3SparseAttentionConfig(
            implementation="msa", indexer_kv_dtype=indexer_kv_dtype
        ),
    )


def _nonzero_bytes(view: torch.Tensor) -> int:
    try:
        raw = view.view(torch.uint8)
    except RuntimeError:
        raw = view.contiguous().view(torch.uint8)
    return int(torch.count_nonzero(raw).item())


def _kv_and_scale_views(manager: MiniMaxM3KVCacheManagerV2) -> list[torch.Tensor]:
    views = [manager.get_buffers(layer_idx) for layer_idx in manager.layer_offsets]
    views += [
        manager.get_block_scale_buffers(layer_idx)
        for layer_idx in manager.layer_offsets
        if manager.is_nvfp4_layer(layer_idx)
    ]
    return views


def _view_layout(view: torch.Tensor) -> tuple:
    # Strides of size-1 dims carry no layout information.
    strides = tuple(stride for stride, size in zip(view.stride(), view.shape) if size != 1)
    return view.dtype, tuple(view.shape), view.data_ptr(), strides


@pytest.mark.cpu_only
def test_nvfp4_indexer_config_has_the_fp8_requirements() -> None:
    cfg = MiniMaxM3SparseAttentionConfig(implementation="msa", indexer_kv_dtype="nvfp4")
    assert cfg.to_sparse_params().indexer_kv_dtype == "nvfp4"
    fused = MiniMaxM3SparseAttentionConfig(
        implementation="msa", indexer_kv_dtype="nvfp4", fuse_qkv_index_projection=True
    )
    assert fused.to_sparse_params().fuse_qkv_index_projection

    with pytest.raises(ValueError, match=r"indexer_kv_dtype='nvfp4' currently requires the 'msa'"):
        MiniMaxM3SparseAttentionConfig(implementation="triton", indexer_kv_dtype="nvfp4")
    with pytest.raises(ValueError, match=r"'nvfp4' requires sparse_disable_index_value=True"):
        MiniMaxM3SparseAttentionConfig(
            implementation="msa", indexer_kv_dtype="nvfp4", sparse_disable_index_value=False
        )
    with pytest.raises(ValueError, match=r"requires indexer_kv_dtype='fp8' or 'nvfp4'"):
        MiniMaxM3SparseAttentionConfig(implementation="msa", fuse_qkv_index_projection=True)
    with pytest.raises(ValueError, match=r"indexer_kv_dtype"):
        MiniMaxM3SparseAttentionConfig(implementation="msa", indexer_kv_dtype="mxfp4")


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("indexer_kv_dtype", "kwargs", "match"),
    [
        ("mxfp4", {}, r"must be 'bf16', 'fp8' or 'nvfp4', got 'mxfp4'"),
        ("nvfp4", {"sparse_index_dim": 64}, r"tokens_per_block=128, got 64 and 128\."),
        ("nvfp4", {"tokens_per_block": 64}, r"tokens_per_block=128, got 128 and 64\."),
        (
            "nvfp4",
            {"disable_index_value_layer_ids": []},
            r"'nvfp4' requires disable_index_value=True for every sparse layer",
        ),
    ],
    ids=["unknown_dtype", "nvfp4_index_dim_64", "nvfp4_64_token_pages", "nvfp4_index_value"],
)
def test_cache_manager_rejects_unsupported_index_k_formats(
    indexer_kv_dtype: str, kwargs: dict, match: str
) -> None:
    # Rejected before any pool is allocated.
    options = {"num_layers": 8, "tokens_per_block": TOKENS_PER_BLOCK, **kwargs}
    with pytest.raises(ValueError, match=match):
        MiniMaxM3KVCacheManagerV2(
            sparse_attention_config=SimpleNamespace(indexer_kv_dtype=indexer_kv_dtype), **options
        )


@requires_cuda
@pytest.mark.parametrize(
    ("indexer_kv_dtype", "role", "page_bytes", "dtype"),
    [
        ("bf16", Role.INDEX_KEY, 32768, torch.bfloat16),
        ("fp8", Role.INDEX_KEY, 16384, torch.float8_e4m3fn),
        ("nvfp4", INDEX_KEY_NVFP4, NVFP4_PAGE_BYTES, torch.uint8),
    ],
)
def test_each_sparse_layer_registers_one_index_k_page(
    indexer_kv_dtype: str, role: str, page_bytes: int, dtype: torch.dtype
) -> None:
    manager = _make_manager(indexer_kv_dtype)
    try:
        assert manager.index_k_page_bytes(TOKENS_PER_BLOCK) == page_bytes
        if indexer_kv_dtype == "nvfp4":
            # The page the NVFP4-E writer and scorers address.
            assert page_bytes == INDEX_K_NVFP4_PAGE_BYTES
        for layer_idx in range(NUM_LAYERS):
            if layer_idx not in SPARSE_LAYERS:
                assert manager.get_index_k_buffer(layer_idx) is None
                continue
            local = manager.layer_offsets[layer_idx]
            assert manager.impl.get_page_stride(local, role) == page_bytes
            for kv_layout in ("HND", "NHD"):
                view = manager.get_index_k_buffer(layer_idx, kv_layout)
                assert view.dtype == dtype
                base_view = KVCacheManagerV2.get_index_k_buffer(
                    manager, layer_idx, head_dim=HEAD_DIM, dtype=dtype, kv_layout=kv_layout
                )
                if indexer_kv_dtype == "nvfp4":
                    # NVFP4-E pages replace the INDEX_KEY buffer rather than adding to it.
                    assert base_view is None
                else:
                    # BF16 and FP8 keep the base INDEX_KEY views in both layouts.
                    assert _view_layout(view) == _view_layout(base_view)
    finally:
        manager.shutdown()


@requires_cuda
@pytest.mark.parametrize(
    ("eagle3", "pp_rank"),
    [(False, None), (True, None), (False, 0), (False, 1)],
    ids=["tp2", "tp2_eagle3", "pp2_rank0", "pp2_rank1"],
)
def test_nvfp4_index_k_views_are_zeroed_disjoint_page_rows(
    eagle3: bool, pp_rank: Optional[int]
) -> None:
    mapping = None if pp_rank is None else Mapping(world_size=2, rank=pp_rank, tp_size=1, pp_size=2)
    manager = _make_manager("nvfp4", eagle3=eagle3, mapping=mapping)
    try:
        local_sparse_layers = [i for i in SPARSE_LAYERS if i in manager.layer_offsets]
        assert local_sparse_layers
        assert (NUM_LAYERS in manager.layer_offsets) == eagle3
        views = {}
        for layer_idx, local in manager.layer_offsets.items():
            view = manager.get_index_k_buffer(layer_idx)
            if layer_idx not in local_sparse_layers:
                # Dense and shared draft layers have no index-K pages.
                assert view is None
                continue
            num_slots = manager.get_buffers(layer_idx).shape[0]
            pages_per_slot = manager.impl.get_page_index_converter(local, INDEX_KEY_NVFP4).scale
            assert view.dtype == torch.uint8
            assert tuple(view.shape) == (num_slots, NVFP4_PAGE_BYTES)
            # Row s is slot s's page. No K/V or scale page is 9216 B, so a pool slot holds
            # exactly one index-K page per local sparse layer and the row stride spans it.
            assert pages_per_slot == len(local_sparse_layers)
            assert view.stride() == (pages_per_slot * NVFP4_PAGE_BYTES, 1)
            assert view.data_ptr() == manager.impl.get_mem_pool_base_address(local, INDEX_KEY_NVFP4)
            # Packed pages have no head axis: both layouts are the same view.
            assert manager.get_index_k_buffer(layer_idx, "NHD") is view
            assert manager.get_index_k_buffer(layer_idx, "HND") is view
            views[layer_idx] = view
        kv_views = _kv_and_scale_views(manager)

        torch.cuda.synchronize()
        assert all(_nonzero_bytes(view) == 0 for view in (*views.values(), *kv_views))

        # No K/V data or block-scale write reaches an index-K page...
        for view in kv_views:
            view.view(torch.uint8).fill_(0x5A)
        assert all(_nonzero_bytes(view) == 0 for view in views.values())
        for view in kv_views:
            view.view(torch.uint8).zero_()
        # ...and a distinct word per (layer, page row) survives only if no two rows alias
        # and no row reaches a K/V data or block-scale page.
        words_per_page = NVFP4_PAGE_BYTES // 4
        patterns = {}
        for layer_idx, view in views.items():
            rows = torch.arange(view.shape[0], device=view.device, dtype=torch.int32)
            patterns[layer_idx] = (rows + (layer_idx << 16))[:, None].expand(-1, words_per_page)
            view.view(torch.int32).copy_(patterns[layer_idx])
        for layer_idx, view in views.items():
            assert torch.equal(view.view(torch.int32), patterns[layer_idx])
        assert all(_nonzero_bytes(view) == 0 for view in kv_views)
    finally:
        manager.shutdown()


@requires_cuda
def test_capacity_follows_index_k_bytes_per_token() -> None:
    num_sparse = len(SPARSE_LAYERS)
    num_dense = NUM_LAYERS - num_sparse
    # The TP2 production layout per token: NVFP4 K/V data and E4M3 scales on sparse layers, FP8
    # K/V on the dense layers and the shared Eagle3 draft layer. Index-K takes FP8 25,760 B to
    # NVFP4-E 22,568 B.
    main_kv_bytes = (
        2
        * NUM_KV_HEADS
        * (num_sparse * (HEAD_DIM // 2 + HEAD_DIM // 16) + (num_dense + 1) * HEAD_DIM)
    )
    bytes_per_token = {
        "fp8": main_kv_bytes + num_sparse * HEAD_DIM,
        "nvfp4": main_kv_bytes + num_sparse * NVFP4_PAGE_BYTES // TOKENS_PER_BLOCK,
    }
    num_slots = {}
    for indexer_kv_dtype, token_bytes in bytes_per_token.items():
        manager = _make_manager(indexer_kv_dtype, gpu_bytes=4 << 30, eagle3=True)
        try:
            assert manager.get_index_k_buffer(NUM_LAYERS) is None
            slot_bytes = sum(
                int(pool.slot_bytes)
                for group in manager.impl.pool_group_descs
                for pool in group.pools
            )
            assert slot_bytes == token_bytes * TOKENS_PER_BLOCK
            num_slots[indexer_kv_dtype] = manager.get_index_k_buffer(SPARSE_LAYERS[0]).shape[0]
        finally:
            manager.shutdown()
    # One GPU budget: slots scale inversely with bytes per token, up to pool rounding.
    assert num_slots["nvfp4"] / num_slots["fp8"] == pytest.approx(
        bytes_per_token["fp8"] / bytes_per_token["nvfp4"], rel=5e-3
    )


@requires_cuda
def test_nvfp4_index_k_is_a_replicated_role_of_its_own() -> None:
    fp8_manager = _make_manager("fp8")
    nvfp4_manager = _make_manager("nvfp4")
    try:
        assert fp8_manager.get_disagg_role_mapper_kinds() == {
            Role.ALL: MapperKind.HND,
            Role.INDEX_KEY: MapperKind.REPLICATED,
        }
        assert nvfp4_manager.get_disagg_role_mapper_kinds() == {
            Role.ALL: MapperKind.HND,
            INDEX_KEY_NVFP4: MapperKind.REPLICATED,
        }

        nvfp4_table = build_page_table_from_manager(nvfp4_manager)
        pool_views = nvfp4_table.layer_groups[0].pool_views
        (index_view,) = [view for view in pool_views if view.mapper_kind == MapperKind.REPLICATED]
        assert index_view.pool_role == frozenset({INDEX_KEY_NVFP4})
        # No K/V or scale page is 9216 B, so the index-K pages fill a pool of their own.
        assert all(
            view.pool_idx != index_view.pool_idx for view in pool_views if view is not index_view
        )
        offset_by_local_layer, bytes_per_layer = get_layer_byte_ranges(index_view)
        assert bytes_per_layer == NVFP4_PAGE_BYTES
        # Each layer's transferred bytes are the pages its index-K view addresses.
        page_addrs = {
            nvfp4_manager.layer_offsets[i]: nvfp4_manager.get_index_k_buffer(i).data_ptr()
            for i in SPARSE_LAYERS
        }
        first_page_addr = min(page_addrs.values())
        assert offset_by_local_layer == {
            local: addr - first_page_addr for local, addr in page_addrs.items()
        }

        # A peer with another index-K format fails the handshake.
        AttentionPolicy.validate_peer_compatible(nvfp4_table, nvfp4_table)
        with pytest.raises(ValueError, match=r"replicated roles differ"):
            AttentionPolicy.validate_peer_compatible(
                nvfp4_table, build_page_table_from_manager(fp8_manager)
            )
    finally:
        nvfp4_manager.shutdown()
        fp8_manager.shutdown()
