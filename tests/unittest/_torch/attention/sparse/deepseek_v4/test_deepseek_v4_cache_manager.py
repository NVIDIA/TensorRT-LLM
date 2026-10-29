# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock, patch

import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.sparse import offload as sparse_offload
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4CacheManager
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import (
    cache_manager as deepseek_v4_cache,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
    DEEPSEEK_V4_SLIDING_ATTENTION,
    DeepseekV4AttentionType,
    compress_ratio_has_attention,
)
from tensorrt_llm._torch.attention.backends.sparse.offload import SparseOffloadPaddingRegion
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
    KVRegionExtractorV1,
    build_page_table_from_manager,
)
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page.nvfp4_quantization import (
    Nvfp4ColdPageQuantizationCompression,
)
from tensorrt_llm._torch.pyexecutor._util import CacheCost
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kv_cache_v2_module
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    CUDA_GRAPH_DUMMY_REQUEST_ID,
    BlockReusePolicy,
    KVCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.sparse_offload_memory import (
    sparse_arena_bytes,
    sparse_arena_shapes,
    sparse_history_bytes_per_block,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import binding_to_torch_dtype
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal import kv_cache_compression as native_kvcc
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import (
    ColdPageQuantizationCompressionConfig,
    DeepSeekV4SparseAttentionConfig,
    DraftTargetDecodingConfig,
    KvCacheConfig,
    MTPDecodingConfig,
    SparseKvCacheOffloadConfig,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    BAD_PAGE_INDEX,
    AttentionLayerConfig,
    BatchDesc,
    BufferConfig,
    BufferId,
    CacheTier,
    GpuCacheTierConfig,
    IndexTableSpec,
    KVCacheDesc,
    KVCacheManagerConfig,
    PageIndexMode,
    SparseCacheConfig,
    _introspection,
)

_RequestCache = Dict[
    Tuple[int, DeepseekV4AttentionType],  # (layer index, attention type)
    Tuple[torch.Tensor, torch.Tensor | None],  # (values tensor, scales tensor)
]


# Sparse offload cache descriptors and history lifecycle.


class _SparseRuntime:
    """Expose native buffer configuration without allocating a cache manager."""

    def __init__(self, tokens_per_block=128):
        self.groups = {41: 13, 19: 13, 42: 9, 20: 9, 43: 5, 21: 5, 7: 14}
        self.sparse = {41, 19}
        self.expansions = {41: tokens_per_block // 4, 19: tokens_per_block // 4}
        self.pool_indices = {}
        self.layers = {}
        for layer_id in (41, 19):
            self.configure_buffers(layer_id, [(DeepseekV4AttentionType.COMPRESS.role, 0)])

    def sparse_index_table_specs(self):
        """One request per (layer, pool) run, as the native manager derives them.

        The fake's sparse buffers are one per layer, so each is its own run.
        """
        return [
            IndexTableSpec([BufferId(layer_id, role)])
            for layer_id in sorted(self.sparse)
            for role in (DeepseekV4AttentionType.COMPRESS.role,)
        ]

    def configure_buffers(self, layer_id, roles_and_pools):
        self.layers[layer_id] = AttentionLayerConfig(
            layer_id=layer_id,
            buffers=[
                BufferConfig(role=role, size=1024, is_sparse=True) for role, _ in roles_and_pools
            ],
            sparse_cache=SparseCacheConfig(
                num_selected_blocks=8, max_cached_units=16, unit_tokens=4
            ),
        )
        for role, pool in roles_and_pools:
            self.pool_indices[BufferId(layer_id, role)] = pool

    @property
    def init_config(self):
        return SimpleNamespace(
            layers=list(self.layers.values()),
            constraints=[
                SimpleNamespace(kv_caches=[None] * 2),
                SimpleNamespace(kv_caches=[None] * 6),
            ],
        )

    @property
    def pool_group_descs(self):
        count = max(self.pool_indices.values()) + 1
        coalesced = [SimpleNamespace(buffer_ids=[]) for _ in range(count)]
        # Storage's buffer order need not be declaration order. The adapter
        # must use this only to locate the pool, then preserve declaration order.
        for buffer_id, pool in reversed(list(self.pool_indices.items())):
            coalesced[pool].buffer_ids.append(buffer_id)
        variant = SimpleNamespace(coalesced_buffers=coalesced)
        return [SimpleNamespace(slot_desc=SimpleNamespace(variants=[variant]))]

    def is_sparse(self, layer_id, role):
        return layer_id in self.sparse

    def get_layer_group_id(self, layer_id):
        return self.groups[layer_id]

    def get_sparse_cache_units_per_page(self, layer_id):
        return self.expansions[layer_id]

    def get_sparse_cache_base_address(self, layer_id, role):
        return 100000 + layer_id * 4096 + self.pool_indices[BufferId(layer_id, role)] * 1024


def _manager(tokens_per_block=128):
    manager = object.__new__(DeepseekV4CacheManager)
    manager.kv_cache_type = CacheTypeCpp.SELFKONLY
    manager._enable_kv_cache_offload = True
    manager._sparse_batch = None
    manager._sparse_batch_members = {}
    manager._sparse_batch_prepared = False
    manager.max_batch_size = 4
    manager._sparse_offload_topk = 8
    manager.max_beam_width = 1
    manager.max_draft_len = 0
    manager.max_total_draft_tokens = 0
    manager.enable_block_reuse = False
    manager.kv_compression_manages_history = False
    manager._has_cp_helix = False
    manager.is_draft = False
    manager._use_nvfp4_compress = False
    manager._nvfp4_residual_dim = 0
    manager._sparse_arena_buffers = {}
    manager.dtype = DataType.BF16
    manager.head_dim = 512
    manager.use_fp8_ds_mla = False
    manager.pp_layers = [2, 3, 4]
    manager._compress_ratios = [1, 4, 4, 128, 4]
    manager.tokens_per_block = tokens_per_block
    manager.max_blocks_per_seq = 6
    manager._layer_attn_to_layer_id = {
        (2, DeepseekV4AttentionType.COMPRESS): 41,
        (4, DeepseekV4AttentionType.COMPRESS): 19,
        (2, DeepseekV4AttentionType.INDEXER_COMPRESS): 42,
        (4, DeepseekV4AttentionType.INDEXER_COMPRESS): 20,
        (2, DeepseekV4AttentionType.SWA): 43,
        (4, DeepseekV4AttentionType.SWA): 21,
        (3, DeepseekV4AttentionType.COMPRESS): 7,
    }
    manager.impl = _SparseRuntime(tokens_per_block)
    # The arenas are the Batch's, and this manager has no native one; answer for it.
    manager._sparse_arena_base = lambda layer_id, role: manager.impl.get_sparse_cache_base_address(
        layer_id, role
    )
    manager.kv_cache_map = {
        10: SimpleNamespace(id=10, history_length=tokens_per_block - 1, pages=[0, 9, -1]),
        20: SimpleNamespace(id=20, history_length=tokens_per_block, pages=[0, 0, 12]),
        30: SimpleNamespace(id=30, history_length=tokens_per_block + 1, pages=[8, 7, 0]),
    }
    for cache in manager.kv_cache_map.values():
        cache.num_blocks = len(cache.pages)
        cache.capacity = cache.num_blocks * tokens_per_block
        cache.beam_width = 1
    return manager


@pytest.mark.cpu_only
@pytest.mark.parametrize("tokens_per_block", [128, 256])
def test_sparse_offload_descriptors_use_runtime_ids(tokens_per_block):
    manager = _manager(tokens_per_block)
    descriptors = manager.get_sparse_offload_descriptors()
    assert list(descriptors) == [2, 4]
    assert [d.buffer_ids[0].layer_id for d in descriptors.values()] == [41, 19]
    for descriptor in descriptors.values():
        assert descriptor.buffer_ids[0].role == DeepseekV4AttentionType.COMPRESS.role
        assert descriptor.group_id == 13
        assert descriptor.unit_tokens == 4
        assert descriptor.data_buffer_index == 0
        assert descriptor.arena_row_counts == (
            manager.max_batch_size * 2 * (tokens_per_block // 4),
        )
        assert descriptor.arena_base_addresses == (
            manager.impl.get_sparse_cache_base_address(*descriptor.buffer_ids[0]),
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "invalid",
    ["expansion", "not_sparse", "unconfigured", "width", "footer"],
)
def test_sparse_offload_rejects_incompatible_layout(invalid):
    manager = _manager()
    if invalid == "expansion":
        manager.impl.expansions[41] = 1
    elif invalid == "not_sparse":
        manager.impl.sparse.remove(41)
    elif invalid == "footer":
        manager.use_fp8_ds_mla = True
    else:
        config = manager.impl.layers[41]
        if invalid == "unconfigured":
            manager.impl.layers[41] = AttentionLayerConfig(
                layer_id=config.layer_id, buffers=config.buffers
            )
        elif invalid == "width":
            cache = config.sparse_cache
            cache.num_selected_blocks += 1
            config.sparse_cache = cache
    with pytest.raises((ValueError, NotImplementedError)):
        manager.get_sparse_offload_descriptors()


@pytest.mark.cpu_only
def test_sparse_offload_accepts_a_dense_buffer_in_a_sparse_layer():
    """An indexer sits beside the data it selects over; only the latter is fetched."""
    manager = _manager()
    config = manager.impl.layers[41]
    buffers = config.buffers
    buffers.append(BufferConfig(role="ordinary", size=16))
    config.buffers = buffers
    descriptors = manager.get_sparse_offload_descriptors()
    assert descriptors
    roles = {b.role for d in descriptors.values() for b in d.buffer_ids}
    assert "ordinary" not in roles


@pytest.mark.cpu_only
def test_sparse_offload_accepts_a_dense_layer_in_a_sparse_group():
    manager = _manager()
    manager.impl.groups[42] = 13
    role = DeepseekV4AttentionType.INDEXER_COMPRESS.role
    manager.impl.layers[42] = AttentionLayerConfig(
        layer_id=42, buffers=[BufferConfig(role=role, size=1024)]
    )
    manager.impl.pool_indices[BufferId(42, role)] = 1

    descriptors = manager.get_sparse_offload_descriptors()

    assert set(descriptors) == {2, 4}
    assert all(buffer.layer_id != 42 for d in descriptors.values() for buffer in d.buffer_ids)


@pytest.mark.cpu_only
def test_sparse_offload_descriptors_include_all_roles_in_native_order():
    manager = _manager()
    manager._use_nvfp4_compress = True
    main_role = DeepseekV4AttentionType.COMPRESS.role
    scale_role = deepseek_v4_cache.COMPRESS_BLOCK_SCALE_ROLE
    manager.impl.configure_buffers(
        41, [(main_role, 1), (scale_role, 0), ("scale_padding", 0), ("data_padding", 1)]
    )
    manager.impl.configure_buffers(19, [(main_role, 1), (scale_role, 0)])
    descriptor = manager.get_sparse_offload_descriptors()[2]
    assert [buffer.role for buffer in descriptor.buffer_ids] == [
        scale_role,
        "scale_padding",
        main_role,
        "data_padding",
    ]
    assert all(buffer.layer_id == 41 for buffer in descriptor.buffer_ids)
    assert descriptor.data_buffer_index == 2
    assert descriptor.scale_buffer_index == 0


@pytest.mark.cpu_only
def test_sparse_offload_padding_regions_hold_units_not_pages():
    manager = _manager()
    manager._use_nvfp4_compress = True
    main = DeepseekV4AttentionType.COMPRESS.role
    scale = deepseek_v4_cache.COMPRESS_BLOCK_SCALE_ROLE
    manager.impl.configure_buffers(
        41, [(main, 1), (scale, 0), ("scale_padding", 0), ("data_padding", 1)]
    )
    manager.impl.configure_buffers(19, [(main, 1), (scale, 0)])
    units_per_page = manager.tokens_per_block // 4
    for layer in manager.impl.layers.values():
        buffers = layer.buffers
        for buffer in buffers:
            # A sparse buffer's size is a whole page of 288-byte rows, or 36-byte scales.
            buffer.size = (36 if "scale" in buffer.role else 288) * units_per_page
        layer.buffers = buffers

    descriptor = manager.get_sparse_offload_descriptors()[2]

    # Sixteen cached units round up to one page of 32, and every lease holds one
    # region per buffer of the pool: three here, the padding role second.
    assert descriptor.padding_regions == tuple(
        SparseOffloadPaddingRegion(
            base_address=manager.impl.get_sparse_cache_base_address(41, role) + 32 * row_bytes,
            bytes_per_lease=32 * row_bytes,
            lease_stride=3 * 32 * row_bytes,
        )
        for role, row_bytes in (("scale_padding", 36), ("data_padding", 288))
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("row_sizes", [(256, 32), (288, 36)])
@pytest.mark.parametrize("max_units", [16, 65])
def test_sparse_arena_padding_initialization_covers_all_leases(row_sizes, max_units):
    manager = _manager()
    manager._use_nvfp4_compress = True
    manager._stream = torch.cuda.Stream()
    manager._nvfp4_residual_dim = row_sizes[0] * 2 - manager.head_dim
    main = DeepseekV4AttentionType.COMPRESS.role
    scale = deepseek_v4_cache.COMPRESS_BLOCK_SCALE_ROLE
    manager.impl.configure_buffers(
        41, [(main, 1), (scale, 0), ("scale_padding", 0), ("data_padding", 1)]
    )
    manager.impl.configure_buffers(19, [(main, 1), (scale, 0)])
    expansion = manager.tokens_per_block // 4
    for layer in manager.impl.layers.values():
        buffers = layer.buffers
        for buffer in buffers:
            # A sparse buffer's size is a whole page of rows.
            buffer.size = row_sizes["scale" in buffer.role] * expansion
        layer.buffers = buffers
        cache = layer.sparse_cache
        cache.max_cached_units = max_units if layer.layer_id == 19 else 16
        layer.sparse_cache = cache
    rows = (max_units + expansion - 1) // expansion * expansion
    # The Batch owns the arenas and leases one row of them per batch row, so the
    # fixture's arena is as wide as its batch limit. Two model buffers and one
    # padding buffer share each pool.
    leases = manager.max_batch_size
    arenas = [
        torch.full((leases, 3, rows, size), 37, dtype=torch.uint8, device="cuda")
        for size in row_sizes
    ]
    manager.impl.get_sparse_cache_base_address = lambda layer, role: arenas[
        "scale" in role
    ].data_ptr()
    descriptors = manager.get_sparse_offload_descriptors()
    manager._initialize_sparse_arena_buffers(descriptors)
    data_view, scale_view = manager.get_sparse_compress_buffers(2)
    assert manager.get_sparse_compress_buffers(4)[0] is data_view
    assert manager.get_sparse_compress_buffers(4)[1] is scale_view
    for view, arena, size in zip((data_view, scale_view), arenas, row_sizes):
        assert view.shape == (leases * 3 * rows, 1, size)
        assert view.is_contiguous() and view.data_ptr() == arena.data_ptr()
        torch.testing.assert_close(view.view(torch.uint8).reshape_as(arena), arena)
    manager._stream.wait_stream(torch.cuda.current_stream())
    manager._initialize_sparse_arena_padding(descriptors)
    for arena in arenas:
        expected = torch.full_like(arena, 37)
        expected[:, 1] = 0
        torch.testing.assert_close(arena, expected)
    # Rebuilding graph metadata describes the same regions without reinitializing
    # an arena that may already contain live request data.
    for arena in arenas:
        arena.fill_(19)
    manager.get_sparse_offload_descriptors()
    assert all(torch.all(arena == 19) for arena in arenas)
    assert torch.all(data_view == 19)
    assert torch.all(scale_view.view(torch.uint8) == 19)


@pytest.mark.cpu_only
@pytest.mark.parametrize("cache_dtype", [DataType.BF16, DataType.FP8, DataType.UINT8])
@pytest.mark.parametrize("max_units", [16, 65])
def test_sparse_arena_views_use_pool_capacity_and_share_storage(cache_dtype, max_units):
    manager = _manager()
    manager.dtype = cache_dtype
    manager._use_nvfp4_compress = cache_dtype == DataType.UINT8
    manager._nvfp4_residual_dim = 64
    manager._stream = SimpleNamespace(device="cuda:0")
    main = DeepseekV4AttentionType.COMPRESS.role
    scale = deepseek_v4_cache.COMPRESS_BLOCK_SCALE_ROLE
    roles = [(main, 1), ("data_padding", 1)]
    if manager._use_nvfp4_compress:
        roles += [(scale, 0), ("scale_padding", 0)]
    manager.impl.configure_buffers(41, roles)
    manager.impl.configure_buffers(
        19, [(main, 1)] + ([(scale, 0)] if manager._use_nvfp4_compress else [])
    )
    config = manager.impl.layers[19]
    cache = config.sparse_cache
    cache.max_cached_units = max_units
    config.sparse_cache = cache
    manager.impl.get_sparse_cache_base_address = lambda layer, role: (
        8192 if "scale" in role else 4096
    )
    descriptors = manager.get_sparse_offload_descriptors()
    rows = manager.max_batch_size * 3 * ((max_units + 31) // 32 * 32)
    with (
        patch.object(torch.cuda, "device"),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        patch.object(
            deepseek_v4_cache,
            "convert_to_torch_tensor",
            side_effect=lambda wrapper: torch.empty(wrapper.shape, dtype=wrapper.dtype),
        ) as convert,
    ):
        manager._initialize_sparse_arena_buffers(descriptors)
    data, scales = manager.get_sparse_compress_buffers(2)
    assert data.shape == (rows, 1, 288 if manager._use_nvfp4_compress else 512)
    assert data.dtype == binding_to_torch_dtype(cache_dtype)
    assert data.is_contiguous()
    assert manager.get_sparse_compress_buffers(4)[0] is data
    if manager._use_nvfp4_compress:
        assert scales.shape == (rows, 1, 36)
        assert scales.dtype == torch.float8_e4m3fn
        assert manager.get_sparse_compress_buffers(4)[1] is scales
    else:
        assert scales is None
    assert convert.call_count == (2 if manager._use_nvfp4_compress else 1)
    assert convert.call_args_list[0].args[0].data_ptr() == 4096

    with (
        patch.object(torch.cuda, "device"),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
        patch.object(deepseek_v4_cache, "convert_to_torch_tensor") as convert,
    ):
        with pytest.raises(RuntimeError, match="before graph capture"):
            manager._initialize_sparse_arena_buffers(descriptors)
        convert.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize("tokens_per_block", [16, 128, 256])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("residual_dim", [0, 64])
def test_compress_storage_views_match_allocation_size(
    tokens_per_block: int, sparse: bool, residual_dim: int
) -> None:
    manager = _manager(tokens_per_block)
    manager._use_nvfp4_compress = True
    manager._nvfp4_residual_dim = residual_dim
    manager.index_head_dim = 128
    manager.compressed_block_sizes = [
        tokens_per_block // ratio for ratio in manager._compress_ratios
    ]
    if not sparse:
        manager.impl.sparse.clear()
    main = DeepseekV4AttentionType.COMPRESS.role
    scale = deepseek_v4_cache.COMPRESS_BLOCK_SCALE_ROLE
    addresses = {main: 4096, scale: 8192}
    manager.impl.get_mem_pool_base_address = Mock(
        side_effect=lambda layer, role, mode: addresses[role]
    )
    # A sparse buffer's page index space is in pages, and a page holds
    # compressed_block_size rows.
    manager.impl.get_page_index_upper_bound = Mock(return_value=3)
    manager.compress_pool_ptrs = {4: addresses[main]}
    manager.compress_scale_pool_ptrs = {4: addresses[scale]}
    with patch.object(
        deepseek_v4_cache,
        "convert_to_torch_tensor",
        side_effect=lambda wrapper: torch.empty(wrapper.shape, dtype=wrapper.dtype),
    ) as convert:
        data = manager.get_buffers(2, DeepseekV4AttentionType.COMPRESS)
        scales = manager.get_compress_scale_buffers(2)
        flat_data, flat_scales = manager.get_compress_pool_buffers(4)

    rows_per_page = tokens_per_block // 4
    storage_dim = manager.head_dim + residual_dim
    assert data.shape == (3, rows_per_page, storage_dim // 2)
    assert scales.shape == (3, rows_per_page, storage_dim // 16)
    assert data.dtype == torch.uint8
    assert scales.dtype == torch.float8_e4m3fn
    assert data.numel() == flat_data.numel() == 3 * (tokens_per_block // 4) * storage_dim // 2
    assert scales.numel() == flat_scales.numel() == 3 * (tokens_per_block // 4) * storage_dim // 16
    assert [call.args[0].data_ptr() for call in convert.call_args_list] == [4096, 8192, 4096, 8192]


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.cpu_only),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
        ),
    ],
)
@pytest.mark.parametrize("nvfp4", [False, True])
@pytest.mark.parametrize("fill_with_zero", [False, True])
def test_sparse_warmup_cleanup_covers_arenas_and_leaves_host_history(
    monkeypatch: pytest.MonkeyPatch, device: str, nvfp4: bool, fill_with_zero: bool
) -> None:
    manager = _manager()
    manager._use_nvfp4_compress = nvfp4
    manager._nvfp4_residual_dim = 64
    manager.index_head_dim = 128
    manager.compressed_block_sizes = [
        manager.tokens_per_block // ratio for ratio in manager._compress_ratios
    ]
    manager._layer_attn_to_layer_id = {
        (2, DeepseekV4AttentionType.COMPRESS): 41,
        (4, DeepseekV4AttentionType.COMPRESS): 19,
    }
    main = DeepseekV4AttentionType.COMPRESS.role
    scale = deepseek_v4_cache.COMPRESS_BLOCK_SCALE_ROLE
    manager._guard_page_by_layer = {(41, main): 5}
    manager._guard_page_value = 17
    backings = []

    def allocate(rows: int, width: int, dtype: torch.dtype, *, host: bool) -> torch.Tensor:
        storage = torch.full(
            (rows + 2, 1, width),
            37,
            dtype=torch.uint8 if dtype == torch.float8_e4m3fn else dtype,
            device="cpu" if host else device,
            pin_memory=host and device == "cuda",
        )
        backings.append((storage, storage[[0, -1]].clone()))
        return storage[1:-1].view(dtype)

    data_dtype = torch.uint8 if nvfp4 else torch.bfloat16
    data_width = (manager.head_dim + manager.nvfp4_residual_dim) // 2 if nvfp4 else manager.head_dim
    host_rows = 3 * (manager.tokens_per_block // 4)
    storage = {}
    for layer in (41, 19):
        storage[layer, main] = allocate(host_rows, data_width, data_dtype, host=True)
        if nvfp4:
            storage[layer, scale] = allocate(host_rows, 36, torch.float8_e4m3fn, host=True)
    arena_data = allocate(11, data_width, data_dtype, host=False)
    arena_scales = allocate(11, 36, torch.float8_e4m3fn, host=False) if nvfp4 else None
    manager._sparse_arena_buffers = {2: (arena_data, arena_scales), 4: (arena_data, arena_scales)}
    if not nvfp4:
        arena_data[0, 0, 0] = torch.nan
    buffers = list(storage.values()) + [arena_data] + ([arena_scales] if nvfp4 else [])
    originals = [buffer.view(torch.uint8).clone() for buffer in buffers]
    manager.impl.get_mem_pool_base_address = Mock(
        side_effect=lambda layer, role, mode: storage[layer, role].data_ptr()
    )
    manager.impl.get_page_index_upper_bound = Mock(
        return_value=host_rows // manager.compressed_block_sizes[2]
    )
    by_address = {buffer.data_ptr(): buffer for buffer in storage.values()}
    convert = deepseek_v4_cache.convert_to_torch_tensor

    def convert_storage(wrapper: deepseek_v4_cache.TensorWrapper) -> torch.Tensor:
        buffer = by_address[wrapper.data_ptr()]
        # Check the native allocation bound before exposing a raw pointer to CUDA.
        assert wrapper.numel() == buffer.numel()
        assert wrapper.dtype == buffer.dtype
        return convert(wrapper) if device == "cuda" else buffer.reshape(wrapper.shape)

    monkeypatch.setattr(deepseek_v4_cache, "convert_to_torch_tensor", convert_storage)
    if device == "cpu":
        manager._stream = SimpleNamespace(device="cpu")
        cleanup_stream = Mock()
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device: cleanup_stream)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
        monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    else:
        manager._stream = torch.cuda.Stream()
        manager._stream.wait_stream(torch.cuda.current_stream())

    with patch.object(torch, "isnan", wraps=torch.isnan) as isnan:
        assert manager.check_invalid_values_in_kv_cache(fill_with_zero) == (not nvfp4)
    if device == "cpu":
        cleanup_stream.wait_stream.assert_called_once_with(manager._stream)
    scanned = [call.args[0].data_ptr() for call in isnan.call_args_list]
    for arena in (arena_data, arena_scales):
        if arena is not None:
            assert scanned.count(arena.data_ptr()) == 1
    # Nothing reads host history before a transfer or a write-back fills it, so
    # the scan neither reads nor clears it, guard page included.
    assert not {buffer.data_ptr() for buffer in storage.values()} & set(scanned)
    for buffer, original in zip(buffers, originals):
        cleared = fill_with_zero and all(buffer is not host for host in storage.values())
        expected = torch.zeros_like(original) if cleared else original
        torch.testing.assert_close(buffer.view(torch.uint8), expected)
    for backing, edges in backings:
        torch.testing.assert_close(backing[[0, -1]].view(torch.uint8), edges.view(torch.uint8))


@pytest.mark.cpu_only
@pytest.mark.parametrize("invalid", ["count", "position", "missing_scales"])
def test_sparse_offload_descriptors_reject_unpaired_nvfp4_arenas(invalid):
    manager = _manager()
    manager._use_nvfp4_compress = True
    main = DeepseekV4AttentionType.COMPRESS.role
    scale = deepseek_v4_cache.COMPRESS_BLOCK_SCALE_ROLE
    manager.impl.configure_buffers(41, [(main, 1), (scale, 0)])
    manager.impl.configure_buffers(19, [(main, 1), (scale, 0)])
    if invalid == "count":
        # An extra buffer in a later layer changes every request's lease stride.
        manager.impl.configure_buffers(19, [(main, 1), (scale, 0), ("scale_padding", 0)])
    elif invalid == "position":
        # Equal pool sizes alone do not align the corresponding buffer positions.
        manager.impl.configure_buffers(
            41, [("data_padding", 1), (main, 1), (scale, 0), ("scale_padding", 0)]
        )
    else:
        manager.impl.configure_buffers(41, [(main, 1)])
    with pytest.raises(ValueError, match="block-scale buffer|data and scale arenas"):
        manager.get_sparse_offload_descriptors()


class _HistoryCache:
    def __init__(self, tokens_per_block: int, cuda_stream: int) -> None:
        self.id, self.beam_width, self.num_blocks = 20, 1, 3
        self.capacity, self.history_length = 3 * tokens_per_block, 0
        self.cuda_stream, self.is_active = cuda_stream, True
        self.pages = [0, 7, 8]
        self.calls: list[tuple[int | None, int | None]] = []
        self.fail_resize = False

    def resize(self, capacity: int | None, history_length: int | None) -> bool:
        self.calls.append((capacity, history_length))
        if self.fail_resize:
            return False
        if capacity is not None:
            self.capacity = capacity
        if history_length is not None:
            self.history_length = history_length
        return True


def _history_case(
    tokens_per_block: int = 128, *, stream: torch.cuda.Stream | None = None
) -> tuple[DeepseekV4CacheManager, _HistoryCache, SimpleNamespace, SimpleNamespace]:
    manager = _manager(tokens_per_block)
    manager._stream = stream or Mock(cuda_stream=19, device=None)
    manager.block_reuse_policy = BlockReusePolicy.ALL_REUSABLE
    manager.conversation_manager = None
    manager._allocated_draft_lens = {}
    manager._inflight_draft_lens = {}
    cache = _HistoryCache(tokens_per_block, manager._stream.cuda_stream)
    manager.kv_cache_map = {20: cache}
    request = SimpleNamespace(
        py_request_id=20,
        context_current_position=tokens_per_block - 1,
        context_remaining_length=0,
        is_dummy_request=False,
        is_disagg_generation_init_state=False,
        py_num_accepted_draft_tokens=0,
        py_rewind_len=0,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        max_beam_num_tokens=tokens_per_block,
    )
    batch = SimpleNamespace(context_requests=[request], generation_requests=[])
    return manager, cache, request, batch


@pytest.mark.cpu_only
@pytest.mark.parametrize("tokens_per_block", [128, 256])
def test_completed_history_advances_without_reuse_or_capacity_growth(tokens_per_block: int) -> None:
    manager, cache, request, batch = _history_case(tokens_per_block)
    caller_stream = object()
    with (
        patch("torch.cuda.current_stream", return_value=caller_stream),
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
    ):
        manager.update_context_resources(batch)
        assert cache.history_length == tokens_per_block - 1
        assert cache.calls == [(None, tokens_per_block - 1)]
        batch.context_requests, batch.generation_requests = [], [request]
        for completed in range(tokens_per_block, 2 * tokens_per_block + 2):
            # What scheduling records for the step, as try_allocate_generation does.
            manager._allocated_draft_lens[request.py_request_id] = 0
            manager._inflight_draft_lens.setdefault(request.py_request_id, []).append(0)
            manager.update_forward_resources(batch, wait_for_input_copy=Mock())
            assert cache.history_length == completed
            assert cache.calls[-1] == (None, completed)
            assert request.py_request_id not in manager._allocated_draft_lens
            assert request.py_request_id not in manager._inflight_draft_lens
            # Sampling has appended one output token; that token has no KV yet.
            request.max_beam_num_tokens = completed + 1
            manager.update_resources(batch)
            assert cache.history_length == completed
            assert cache.calls[-1] == (None, completed)
        assert manager._stream.wait_stream.call_count == tokens_per_block + 3
        manager._stream.wait_stream.assert_called_with(caller_stream)
        assert len(cache.calls) == tokens_per_block + 3


@pytest.mark.cpu_only
@pytest.mark.parametrize("context", [False, True])
@pytest.mark.parametrize(
    "problem", ["resize_failure", "wrong_stream", "capture", "suspended", "removed"]
)
def test_history_update_rejects_failures_and_skips_inactive(context: bool, problem: str) -> None:
    manager, cache, request, batch = _history_case()
    update = manager.update_context_resources if context else manager.update_forward_resources
    update_kwargs = {} if context else {"wait_for_input_copy": Mock()}
    if not context:
        batch.context_requests, batch.generation_requests = [], [request]
    if problem == "resize_failure":
        cache.fail_resize = True
    elif problem == "wrong_stream":
        cache.cuda_stream += 1
    elif problem == "suspended":
        cache.is_active = False
    elif problem == "removed":
        manager.kv_cache_map.clear()
    with (
        patch("torch.cuda.current_stream"),
        patch("torch.cuda.is_current_stream_capturing", return_value=problem == "capture"),
    ):
        if problem in ("suspended", "removed"):
            update(batch, **update_kwargs)
            manager._stream.wait_stream.assert_not_called()
        else:
            with pytest.raises((ValueError, RuntimeError)):
                update(batch, **update_kwargs)
    assert cache.history_length == 0
    assert len(cache.calls) == (1 if problem == "resize_failure" else 0)


@pytest.mark.cpu_only
def test_sparse_descriptors_allow_different_groups_and_pool_geometry() -> None:
    manager = _manager()
    manager.impl.groups[19] = 100
    manager.impl.configure_buffers(19, [(DeepseekV4AttentionType.COMPRESS.role, 1)])
    descriptors = manager.get_sparse_offload_descriptors()
    assert descriptors[2].group_id == 13
    assert descriptors[4].group_id == 100


@pytest.mark.cpu_only
@pytest.mark.parametrize("gen_init", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_prefill_rejected_from_scheduled_request(gen_init, enabled):
    """Offload serves generation only, so a context request that runs a forward is refused.

    A disaggregated gen-init request is the exception: it is resized and its KV
    transferred before it joins the Batch, so it runs no forward this step.
    """
    manager = _manager()
    manager._enable_kv_cache_offload = enabled
    batch = SimpleNamespace(
        context_requests=[
            SimpleNamespace(
                py_request_id=20,
                is_first_context_chunk=True,
                is_last_context_chunk=True,
                is_disagg_generation_init_state=gen_init,
            )
        ],
        generation_requests=[],
    )
    with (
        patch.object(KVCacheManagerV2, "prepare_resources") as prepare,
        patch.object(manager, "_prepare_sparse_batch") as publish,
    ):
        if enabled and not gen_init:
            with pytest.raises(NotImplementedError, match="generation only"):
                manager.prepare_resources(batch)
            prepare.assert_not_called()
            publish.assert_not_called()
        else:
            manager.prepare_resources(batch)
            prepare.assert_called_once_with(batch)
            # Nothing runs a forward here, so no batch is published either way.
            publish.assert_not_called()


class _SchedulingBatch:
    """Model native Batch's exclusive page-index ownership and published snapshots."""

    def __init__(self, manager, cuda_stream, max_batch_size, max_blocks_per_seq, index_tables=()):
        self.cuda_stream = cuda_stream
        self.max_batch_size = max_batch_size
        self.max_blocks_per_seq = max_blocks_per_seq
        self.index_tables = list(index_tables)
        self.rows = [None] * max_batch_size
        self.snapshots = []
        self.upload_pending = False
        self.flushes = []
        self.joins = []
        self.leaves = []

    def row_of(self, cache):
        return self.rows.index(cache) if cache in self.rows else -1

    def flush_sparse_units(self, new_history_lengths):
        # Recorded rather than performed: what matters here is that the write-back is asked
        # for while the members are still seated, and with each row's own next history.
        assert len(new_history_lengths) == self.max_batch_size
        for row, cache in enumerate(self.rows):
            if cache is not None:
                assert new_history_lengths[row] >= cache.history_length
        self.flushes.append(list(new_history_lengths))

    def add_kv_cache(self, cache):
        assert not self.upload_pending
        assert cache.page_index_owner == "runtime"
        row = self.rows.index(None)
        self.rows[row] = cache
        cache.page_index_owner = "batch"
        self.joins.append(cache.id)
        return row

    def remove_kv_cache(self, cache):
        assert not self.upload_pending
        assert cache.page_index_owner == "batch"
        self.rows[self.rows.index(cache)] = None
        cache.page_index_owner = "cache"
        self.leaves.append(cache.id)

    def coalesced_page_table(self, pool_id: int) -> torch.Tensor:
        return torch.tensor(
            [
                list(cache.pages) + [BAD_PAGE_INDEX] * (self.max_blocks_per_seq - len(cache.pages))
                if cache is not None
                else [BAD_PAGE_INDEX] * self.max_blocks_per_seq
                for cache in self.rows
            ],
            dtype=torch.int32,
        )

    def publish(self):
        assert not self.upload_pending
        self.snapshots.append(
            [
                (cache.id, cache.history_length, cache.capacity, tuple(cache.pages))
                if cache is not None
                else (None, 0, 0, (BAD_PAGE_INDEX,) * self.max_blocks_per_seq)
                for cache in self.rows
            ]
        )
        self.upload_pending = True

    def wait_published(self):
        self.upload_pending = False


@pytest.fixture
def sparse_scheduling_case():
    manager, cache, request, _ = _history_case()
    manager.max_batch_size = 4
    cache.page_index_owner = "runtime"
    second = _HistoryCache(manager.tokens_per_block, manager._stream.cuda_stream)
    second.id = 10
    second.history_length = manager.tokens_per_block
    second.page_index_owner = "runtime"
    manager.kv_cache_map[10] = second
    request.is_first_context_chunk = request.is_last_context_chunk = True
    other_request = SimpleNamespace(py_request_id=10, is_disagg_generation_init_state=False)
    scheduled = SimpleNamespace(context_requests=[], generation_requests=[request, other_request])
    manager.num_pools = 2
    manager.index_mapper = Mock()
    slots = {20: 3, 10: 1}
    manager.index_mapper.get_index.side_effect = slots.__getitem__
    manager.host_kv_cache_block_offsets = torch.full((2, 4, 1, 6), -99, dtype=torch.int32)

    def restore(request_id, request_cache):
        assert request_cache.page_index_owner == "cache"
        request_cache.page_index_owner = "runtime"

    with (
        patch.object(sparse_offload, "Batch", side_effect=_SchedulingBatch) as create_batch,
        patch.object(KVCacheManagerV2, "prepare_resources"),
        patch.object(manager, "_set_page_index_bufs", side_effect=restore) as restore_buffers,
        patch("torch.cuda.current_stream", return_value=Mock(cuda_stream=23)) as current_stream,
        patch("torch.cuda.is_current_stream_capturing", return_value=False) as capturing,
    ):
        yield SimpleNamespace(
            manager=manager,
            cache=cache,
            request=request,
            scheduled=scheduled,
            slots=slots,
            create_batch=create_batch,
            restore_buffers=restore_buffers,
            current_stream=current_stream,
            capturing=capturing,
        )


@pytest.mark.cpu_only
def test_sparse_batch_publishes_schedule_and_reuses_allocation(sparse_scheduling_case):
    case = sparse_scheduling_case
    manager = case.manager
    manager.prepare_resources(case.scheduled)
    batch = manager.get_sparse_batch([20, 10, CUDA_GRAPH_DUMMY_REQUEST_ID])
    assert [row[0] for row in batch.snapshots[-1]] == [20, 10, None, None]
    assert batch.snapshots[-1][0][1:] == (0, 384, (0, 7, 8))
    assert batch.snapshots[-1][1][1:] == (128, 384, (0, 7, 8))
    assert batch.snapshots[-1][2] == (None, 0, 0, (-1,) * manager.max_blocks_per_seq)
    case.current_stream.return_value.wait_stream.assert_called_with(manager._stream)

    # A new schedule order moves nothing: each request keeps its row and lease.
    case.scheduled.generation_requests.reverse()
    manager.prepare_resources(case.scheduled)
    assert manager.get_sparse_batch([10, 20]) is batch
    assert [row[0] for row in batch.snapshots[-1]] == [20, 10, None, None]
    call = case.create_batch.call_args
    assert call.args == (manager.impl, manager._stream.cuda_stream)
    assert call.kwargs["max_batch_size"] == 4
    assert call.kwargs["max_blocks_per_seq"] == 6
    # By content: IndexTableSpec has no value equality, so comparing against a freshly
    # derived list would only ever compare object identity and never hold.
    assert [
        [(buffer.layer_id, buffer.role) for buffer in request.buffers]
        for request in call.kwargs["index_tables"]
    ] == [
        [(layer_id, DeepseekV4AttentionType.COMPRESS.role)]
        for layer_id in sorted(manager.impl.sparse)
    ]
    case.restore_buffers.assert_not_called()
    assert batch.joins == [20, 10]
    assert not batch.leaves


@pytest.mark.cpu_only
@pytest.mark.parametrize("max_beam_width", [1, 2])
def test_sparse_batch_retains_continuing_members_and_refreshes_legacy_tables(
    sparse_scheduling_case: SimpleNamespace, max_beam_width: int
) -> None:
    case = sparse_scheduling_case
    manager = case.manager
    manager.max_beam_width = max_beam_width
    manager.host_kv_cache_block_offsets = torch.full(
        (2, 4 * max_beam_width, 1, 6), -99, dtype=torch.int32
    )
    manager.prepare_resources(case.scheduled)
    batch = manager.get_sparse_batch([20, 10])
    case.cache.pages = [4, 8, 9]
    case.scheduled.generation_requests.reverse()
    manager.prepare_resources(case.scheduled)

    assert batch.joins == [20, 10]
    assert not batch.leaves
    assert manager.get_sparse_batch([10, 20]) is batch
    # Only the columns below the members' block counts are copied; past them the
    # runtime's binding keeps its own BAD_PAGE_INDEX, here the -99 the test filled.
    for pool_id in range(manager.num_pools):
        assert manager.host_kv_cache_block_offsets[pool_id, 3 * max_beam_width, 0].tolist() == [
            4,
            8,
            9,
            -99,
            -99,
            -99,
        ]
        assert manager.host_kv_cache_block_offsets[pool_id, max_beam_width, 0].tolist() == [
            0,
            7,
            8,
            -99,
            -99,
            -99,
        ]

    case.scheduled.generation_requests = [case.request]
    manager.prepare_resources(case.scheduled)
    assert manager.get_sparse_batch([20]) is batch
    assert batch.joins == [20, 10]
    assert batch.leaves == [10]
    assert batch.rows == [case.cache, None, None, None]
    assert case.cache.page_index_owner == "batch"


@pytest.mark.cpu_only
@pytest.mark.parametrize("warmup", [False, True])
def test_sparse_batch_metadata_joins_scheduler_or_prepares_warmup(sparse_scheduling_case, warmup):
    case = sparse_scheduling_case
    if not warmup:
        case.manager.prepare_resources(case.scheduled)
    case.manager.prepare_sparse_batch([20, 10, CUDA_GRAPH_DUMMY_REQUEST_ID])
    batch = case.manager.get_sparse_batch([20, 10])
    case.manager.prepare_sparse_batch([20, 10])
    # Metadata may order the published requests differently; nothing is republished.
    case.manager.prepare_sparse_batch([10, 20])
    assert len(batch.snapshots) == 1
    assert [row[0] for row in batch.snapshots[-1]] == [20, 10, None, None]
    case.current_stream.return_value.wait_stream.assert_called_with(case.manager._stream)
    with pytest.raises(ValueError, match="do not match"):
        case.manager.prepare_sparse_batch([20])


@pytest.mark.cpu_only
@pytest.mark.parametrize("dummy_count", [0, 1, 4])
def test_sparse_batch_clears_rows_for_empty_or_padded_schedule(sparse_scheduling_case, dummy_count):
    case = sparse_scheduling_case
    case.manager.prepare_resources(case.scheduled)
    case.scheduled.generation_requests = [
        SimpleNamespace(
            py_request_id=CUDA_GRAPH_DUMMY_REQUEST_ID, is_disagg_generation_init_state=False
        )
    ] * dummy_count
    case.manager.prepare_resources(case.scheduled)
    batch = case.manager.get_sparse_batch([CUDA_GRAPH_DUMMY_REQUEST_ID] * dummy_count)
    assert all(row == (None, 0, 0, (-1,) * 6) for row in batch.snapshots[-1])
    assert all(cache.page_index_owner == "runtime" for cache in case.manager.kv_cache_map.values())


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "inactive",
        "beam",
        "stream",
        "history",
        "blocks",
        "duplicate",
        "padding",
        "size",
        "capture",
    ],
)
def test_sparse_batch_rejects_invalid_schedule_before_binding(sparse_scheduling_case, problem):
    case = sparse_scheduling_case
    if problem == "missing":
        del case.manager.kv_cache_map[20]
    elif problem == "inactive":
        case.cache.is_active = False
    elif problem == "beam":
        case.cache.beam_width = 2
    elif problem == "stream":
        case.cache.cuda_stream += 1
    elif problem == "history":
        case.cache.history_length = case.cache.capacity + 1
    elif problem == "blocks":
        case.cache.num_blocks = case.manager.max_blocks_per_seq + 1
    elif problem == "duplicate":
        case.scheduled.generation_requests = [case.request, case.request]
    elif problem == "padding":
        case.scheduled.generation_requests.insert(
            0,
            SimpleNamespace(
                py_request_id=CUDA_GRAPH_DUMMY_REQUEST_ID, is_disagg_generation_init_state=False
            ),
        )
    elif problem == "size":
        case.manager.max_batch_size = 1
    else:
        case.capturing.return_value = True
    with pytest.raises((ValueError, RuntimeError)):
        case.manager.prepare_resources(case.scheduled)
    case.create_batch.assert_not_called()
    case.restore_buffers.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "request_ids",
    [
        [20],
        [20, 20],
        [20, 10, 30],
        [20, -1, 10],
        [20, 10] + [-1] * 3,
        [CUDA_GRAPH_DUMMY_REQUEST_ID, 20, 10],
        [20, 10] + [CUDA_GRAPH_DUMMY_REQUEST_ID] * 3,
    ],
)
def test_sparse_batch_checks_consumer_requests(sparse_scheduling_case, request_ids):
    case = sparse_scheduling_case
    case.manager.prepare_resources(case.scheduled)
    with pytest.raises(ValueError, match="do not match"):
        case.manager.get_sparse_batch(request_ids)


@pytest.mark.cpu_only
@pytest.mark.parametrize("request_ids", [[20, 10], [10, 20], [10, 20, CUDA_GRAPH_DUMMY_REQUEST_ID]])
def test_sparse_batch_accepts_members_in_any_order(sparse_scheduling_case, request_ids):
    case = sparse_scheduling_case
    case.manager.prepare_resources(case.scheduled)
    assert case.manager.get_sparse_batch(request_ids) is case.manager._sparse_batch


def _add_scheduled_request(case: SimpleNamespace, request_id: int, slot: int) -> _HistoryCache:
    cache = _HistoryCache(case.manager.tokens_per_block, case.manager._stream.cuda_stream)
    cache.id = request_id
    cache.page_index_owner = "runtime"
    case.manager.kv_cache_map[request_id] = cache
    case.slots[request_id] = slot
    return cache


@pytest.mark.cpu_only
def test_sparse_batch_rows_are_slots_that_arrivals_refill(sparse_scheduling_case):
    """A departure leaves a free row, which the next arrival takes; nobody else moves."""
    case = sparse_scheduling_case
    manager = case.manager
    _add_scheduled_request(case, 30, slot=0)
    third = SimpleNamespace(py_request_id=30, is_disagg_generation_init_state=False)
    other = case.scheduled.generation_requests[1]

    def schedule(*requests):
        case.scheduled.generation_requests = list(requests)
        manager.prepare_resources(case.scheduled)
        return {rid: member.row for rid, member in manager._sparse_batch_members.items()}

    assert schedule(case.request, other) == {20: 0, 10: 1}
    batch = manager._sparse_batch
    assert schedule(other) == {10: 1}
    assert batch.rows == [None, manager.kv_cache_map[10], None, None]
    # The arrival runs first in the input but takes the lowest free row.
    assert schedule(third, other) == {10: 1, 30: 0}
    assert manager.get_sparse_batch([30, 10]) is batch
    assert schedule(other, third) == {10: 1, 30: 0}
    assert batch.joins == [20, 10, 30]
    assert batch.leaves == [20]


@pytest.mark.cpu_only
@pytest.mark.parametrize("hole", [False, True])
def test_sparse_batch_snapshot_follows_rows_with_or_without_holes(sparse_scheduling_case, hole):
    case = sparse_scheduling_case
    manager = case.manager
    third = _add_scheduled_request(case, 30, slot=0)
    third.pages = [5, 6, 7]
    scheduled = case.scheduled.generation_requests + [
        SimpleNamespace(py_request_id=30, is_disagg_generation_init_state=False)
    ]
    case.scheduled.generation_requests = scheduled
    manager.prepare_resources(case.scheduled)
    assert {rid: member.row for rid, member in manager._sparse_batch_members.items()} == {
        20: 0,
        10: 1,
        30: 2,
    }
    if hole:
        # Request 10 leaves row 1 free, so rows 0 and 2 are gathered.
        case.scheduled.generation_requests = [scheduled[2], scheduled[0]]
    # A member that grew writes its new pages through its Batch row.
    case.cache.pages = [4, 8, 9, 11]
    case.cache.num_blocks = 4
    third.pages = [5, 6, 12]
    manager.prepare_resources(case.scheduled)
    host = manager.host_kv_cache_block_offsets
    for pool_id in range(manager.num_pools):
        assert host[pool_id, 3, 0].tolist() == [4, 8, 9, 11, -99, -99]
        assert host[pool_id, 0, 0].tolist() == [5, 6, 12, -1, -99, -99]
        if not hole:
            assert host[pool_id, 1, 0].tolist() == [0, 7, 8, -1, -99, -99]


def _gen_init_request(request_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=request_id,
        is_first_context_chunk=True,
        is_last_context_chunk=True,
        is_disagg_generation_init_state=True,
    )


@pytest.mark.cpu_only
def test_sparse_batch_ignores_disagg_gen_init_preparation(sparse_scheduling_case):
    """Gen-init requests are prepared before their KV transfer and run no forward.

    Their admission bypasses the batch budget, so a burst can exceed the Batch rows.
    """
    case = sparse_scheduling_case
    burst = case.manager.max_batch_size + 2
    case.scheduled.context_requests = [_gen_init_request(100 + i) for i in range(burst)]
    case.scheduled.generation_requests = []

    case.manager.prepare_resources(case.scheduled)

    case.create_batch.assert_not_called()
    with pytest.raises(RuntimeError, match="must be prepared"):
        case.manager.get_sparse_batch([])


@pytest.mark.cpu_only
def test_disagg_gen_init_retains_continuing_sparse_cache_leases(
    sparse_scheduling_case: SimpleNamespace,
) -> None:
    case = sparse_scheduling_case
    case.manager.prepare_resources(case.scheduled)
    batch = case.manager.get_sparse_batch([20, 10])
    case.manager.prepare_resources(
        SimpleNamespace(context_requests=[_gen_init_request(100)], generation_requests=[])
    )
    assert [cache.id for cache in batch.rows if cache is not None] == [20, 10]
    assert not batch.leaves
    case.manager.prepare_resources(case.scheduled)
    assert case.manager.get_sparse_batch([20, 10]) is batch
    assert batch.joins == [20, 10]


@pytest.mark.cpu_only
def test_releasing_unseated_request_preserves_the_published_batch(
    sparse_scheduling_case: SimpleNamespace,
) -> None:
    case = sparse_scheduling_case
    case.manager.prepare_resources(case.scheduled)
    batch = case.manager.get_sparse_batch([20, 10])
    with patch.object(KVCacheManagerV2, "free_resources"):
        case.manager.free_resources(_gen_init_request(100))
    assert case.manager.get_sparse_batch([20, 10]) is batch
    assert not batch.leaves


@pytest.mark.cpu_only
def test_sparse_batch_publishes_only_requests_that_run_a_forward(sparse_scheduling_case):
    case = sparse_scheduling_case
    case.scheduled.context_requests = [_gen_init_request(100)]

    case.manager.prepare_resources(case.scheduled)

    batch = case.manager.get_sparse_batch([20, 10])
    assert [row[0] for row in batch.snapshots[-1]] == [20, 10, None, None]


@pytest.mark.cpu_only
@pytest.mark.parametrize("failure", ["add", "publish"])
def test_sparse_batch_restores_ownership_after_publication_failure(sparse_scheduling_case, failure):
    case = sparse_scheduling_case
    case.manager.prepare_resources(case.scheduled)
    batch = case.manager.get_sparse_batch([20, 10])
    case.manager._release_sparse_batch()
    if failure == "add":
        add = batch.add_kv_cache

        def fail_second(cache):
            if cache.id == 10:
                raise RuntimeError("injected add failure")
            return add(cache)

        batch.add_kv_cache = fail_second
    else:
        batch.publish = Mock(side_effect=RuntimeError("injected publish failure"))
    with pytest.raises(RuntimeError, match="injected"):
        case.manager.prepare_resources(case.scheduled)
    assert batch.rows == [None] * 4
    assert all(cache.page_index_owner == "runtime" for cache in case.manager.kv_cache_map.values())
    with pytest.raises(RuntimeError, match="must be prepared"):
        case.manager.get_sparse_batch([20, 10])


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "operation",
    [
        "update_context_resources",
        "suspend_request",
        "free_resources",
        "release_index_slot",
        "shutdown",
    ],
)
def test_sparse_batch_detaches_before_cache_lifecycle_mutations(sparse_scheduling_case, operation):
    case = sparse_scheduling_case
    manager = case.manager
    manager.prepare_resources(case.scheduled)
    batch = manager.get_sparse_batch([20, 10])

    def mutate(*args):
        if operation == "shutdown":
            assert batch.rows == [None] * 4
            assert all(
                cache.page_index_owner == "runtime" for cache in manager.kv_cache_map.values()
            )
        else:
            assert batch.rows == [None, manager.kv_cache_map[10], None, None]
            assert case.cache.page_index_owner == "runtime"
            assert manager.kv_cache_map[10].page_index_owner == "batch"

    with patch.object(KVCacheManagerV2, operation, side_effect=mutate) as parent:
        if operation.startswith("update_"):
            case.scheduled.context_requests = [case.request]
            getattr(manager, operation)(case.scheduled)
        elif operation == "release_index_slot":
            manager.release_index_slot(20)
        elif operation == "shutdown":
            manager.shutdown()
        else:
            getattr(manager, operation)(case.request)
        parent.assert_called_once()
    with pytest.raises(RuntimeError, match="must be prepared"):
        manager.get_sparse_batch([20, 10])


@pytest.mark.cpu_only
@pytest.mark.parametrize("context", [False, True])
def test_sparse_batch_publishes_updated_history_on_next_step(sparse_scheduling_case, context):
    case = sparse_scheduling_case
    manager, cache, request = case.manager, case.cache, case.request
    cache.history_length, cache.capacity = 7, 8
    request.context_current_position = 8
    request.max_beam_num_tokens = 9  # The sampled output token does not yet have KV.
    # Offload serves generation only, so the context side of this is reached solely by a
    # disaggregated gen-init request: resized and transferred before it joins the Batch, it
    # runs no forward and nothing is published for it until it becomes a generation request.
    request.is_disagg_generation_init_state = context
    case.scheduled.context_requests = [request] if context else []
    case.scheduled.generation_requests = [] if context else [request]
    manager.prepare_resources(case.scheduled)
    batch = None
    if not context:
        batch = manager.get_sparse_batch([20])
        assert batch.snapshots[-1][0][1:3] == (7, 8)

    if context:
        manager.update_context_resources(case.scheduled)
    else:
        manager.update_forward_resources(case.scheduled, wait_for_input_copy=Mock())
        manager.update_resources(case.scheduled)
    assert cache.history_length == cache.capacity == 8
    assert cache.page_index_owner == ("runtime" if context else "batch")
    if batch is not None:
        assert not batch.upload_pending

    # Scheduling reserves the next input token after the completed history update.
    assert cache.resize(9, None)
    request.is_disagg_generation_init_state = False
    case.scheduled.context_requests = []
    case.scheduled.generation_requests = [request]
    manager.prepare_resources(case.scheduled)
    stepped = manager.get_sparse_batch([20])
    if batch is not None:
        assert stepped is batch
    assert stepped.snapshots[-1][0][1:3] == (8, 9)


@pytest.mark.cpu_only
@pytest.mark.parametrize("history_length", [7, 127, 255])
def test_sparse_overlap_history_advances_before_delayed_samples(
    sparse_scheduling_case: SimpleNamespace, history_length: int
) -> None:
    case = sparse_scheduling_case
    manager, cache, request = case.manager, case.cache, case.request
    case.scheduled.generation_requests = [request]
    cache.history_length, cache.capacity = history_length, history_length + 1
    request.max_beam_num_tokens = history_length + 1
    manager.prepare_resources(case.scheduled)
    batch = manager.get_sparse_batch([20])

    # Forward N produces KV before the executor consumes sampling result N.
    manager.update_forward_resources(case.scheduled, wait_for_input_copy=Mock())
    assert cache.history_length == history_length + 1
    assert cache.page_index_owner == "batch"

    assert cache.resize(history_length + 2, None)
    manager.prepare_resources(case.scheduled)
    assert batch.snapshots[-1][0][1:3] == (history_length + 1, history_length + 2)
    calls = list(cache.calls)
    # A delayed update must not detach the newly published Batch.
    manager.update_resources(case.scheduled)
    assert manager.get_sparse_batch([20]) is batch
    assert cache.calls == calls

    manager.update_forward_resources(case.scheduled, wait_for_input_copy=Mock())
    assert cache.history_length == history_length + 2
    calls = list(cache.calls)
    # The overlap executor now consumes N's sample after forward N+1.
    # Its CPU token count describes an older KV frontier.
    request.max_beam_num_tokens = history_length + 2
    manager.update_resources(case.scheduled)
    assert cache.history_length == history_length + 2
    assert cache.calls == calls

    assert cache.resize(history_length + 3, None)
    manager.prepare_resources(case.scheduled)
    assert batch.snapshots[-1][0][1:3] == (history_length + 2, history_length + 3)


@pytest.mark.cpu_only
def test_sparse_history_does_not_advance_without_forward(
    sparse_scheduling_case: SimpleNamespace,
) -> None:
    case = sparse_scheduling_case
    manager, cache, request = case.manager, case.cache, case.request
    case.scheduled.generation_requests = [request]
    cache.history_length, cache.capacity = 7, 8
    manager.prepare_resources(case.scheduled)
    batch = manager.get_sparse_batch([20])
    # Resource preparation alone can be followed by can_queue=False.
    manager.update_resources(case.scheduled)
    manager.prepare_resources(case.scheduled)
    assert cache.history_length == 7
    assert not cache.calls
    assert batch.snapshots[-1][0][1:3] == (7, 8)


@pytest.mark.cpu_only
@pytest.mark.parametrize("copy_fails", [False, True])
def test_sparse_forward_waits_for_input_copy_before_history_update(
    sparse_scheduling_case: SimpleNamespace, copy_fails: bool
) -> None:
    case = sparse_scheduling_case
    manager, cache, request = case.manager, case.cache, case.request
    case.scheduled.generation_requests = [request]
    cache.history_length, cache.capacity = 7, 8
    manager.prepare_resources(case.scheduled)
    batch = manager.get_sparse_batch([20])

    def wait_for_input_copy() -> None:
        assert manager.get_sparse_batch([20]) is batch
        assert cache.page_index_owner == "batch"
        assert not cache.calls
        if copy_fails:
            raise RuntimeError("input copy failed")

    wait = Mock(side_effect=wait_for_input_copy)
    if copy_fails:
        with pytest.raises(RuntimeError, match="input copy failed"):
            manager.update_forward_resources(case.scheduled, wait_for_input_copy=wait)
        assert manager.get_sparse_batch([20]) is batch
        assert cache.history_length == 7
        assert not cache.calls
    else:
        manager.update_forward_resources(case.scheduled, wait_for_input_copy=wait)
        assert cache.history_length == 8
        assert cache.page_index_owner == "batch"
    wait.assert_called_once_with()


@pytest.mark.cpu_only
@pytest.mark.parametrize("history_length", [3, 127])
def test_sparse_forward_keeps_flush_proof_attached_until_history_advances(
    sparse_scheduling_case: SimpleNamespace, history_length: int
) -> None:
    case = sparse_scheduling_case
    manager, cache, request = case.manager, case.cache, case.request
    case.scheduled.generation_requests = [request]
    cache.history_length, cache.capacity = history_length, history_length + 1
    manager.prepare_resources(case.scheduled)
    batch = manager.get_sparse_batch([20])
    manager._stream.wait_stream.reset_mock()
    flush = batch.flush_sparse_units
    resize = cache.resize

    def flush_after_forward(histories):
        manager._stream.wait_stream.assert_called_once_with(case.current_stream.return_value)
        flush(histories)

    def advance_attached(capacity, history):
        row = batch.row_of(cache)
        assert row >= 0, "The native flush watermark belongs to the attached Batch row"
        assert batch.flushes[-1][row] >= history
        assert not batch.upload_pending, "History mutation must not race the staged table upload"
        return resize(capacity, history)

    with (
        patch.object(batch, "flush_sparse_units", side_effect=flush_after_forward),
        patch.object(cache, "resize", side_effect=advance_attached),
    ):
        manager.update_forward_resources(case.scheduled, wait_for_input_copy=Mock())

    assert cache.history_length == history_length + 1
    assert cache.page_index_owner == "batch"
    assert batch.row_of(cache) == 0
    assert batch.joins == [20]
    assert not batch.leaves


@pytest.mark.cpu_only
def test_sparse_forward_requires_reserved_input_capacity() -> None:
    manager, cache, request, batch = _history_case()
    batch.context_requests, batch.generation_requests = [], [request]
    cache.history_length = cache.capacity
    with (
        patch("torch.cuda.current_stream"),
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
        pytest.raises(ValueError, match="no reserved input token"),
    ):
        manager.update_forward_resources(batch, wait_for_input_copy=Mock())
    assert not cache.calls


@pytest.mark.cpu_only
def test_sparse_scheduling_keeps_a_reserved_input_token() -> None:
    """An attention-DP dummy arrives holding its input token; the scheduler must not grow it."""
    manager, cache, request, batch = _history_case()
    batch.context_requests, batch.generation_requests = [], [request]
    # add_dummy_requests(is_gen=True) reserves the step's input token up front.
    cache.history_length, cache.capacity = 0, 1

    with patch.object(KVCacheManagerV2, "try_allocate_generation") as grow:
        assert manager.try_allocate_generation(request)
    grow.assert_not_called()
    assert not cache.calls
    assert request.py_request_id not in manager._allocated_draft_lens

    # A fleet-wide skipped step (can_queue=False) has no growth to give back.
    manager.revert_allocate_generation(request)
    assert not cache.calls

    # The forward consumes exactly the reserved token.
    with (
        patch("torch.cuda.current_stream"),
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
    ):
        manager.update_forward_resources(batch, wait_for_input_copy=Mock())
    assert cache.history_length == cache.capacity == 1


@pytest.mark.cpu_only
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("reserved", [False, True])
def test_generation_growth_defers_to_the_base_policy(offload: bool, reserved: bool) -> None:
    """Only an offload cache that already holds its input token skips growth.

    Without offload, history follows sampling and lags one step under the overlap
    scheduler, so capacity beyond history is normal there and must still grow.
    """
    manager, cache, request, _ = _history_case()
    manager._enable_kv_cache_offload = offload
    cache.history_length = 7
    cache.capacity = 8 if reserved else 7

    with patch.object(KVCacheManagerV2, "try_allocate_generation", return_value=True) as grow:
        assert manager.try_allocate_generation(request)
    assert grow.call_count == (0 if offload and reserved else 1)


@pytest.mark.cpu_only
def test_non_offload_history_still_follows_sampling() -> None:
    manager, cache, request, batch = _history_case()
    manager._enable_kv_cache_offload = False
    batch.context_requests, batch.generation_requests = [], [request]
    wait = Mock()
    manager.update_forward_resources(batch, wait_for_input_copy=wait)
    wait.assert_not_called()
    assert not cache.calls
    manager.update_resources(batch)
    assert cache.history_length == request.max_beam_num_tokens - 1
    manager._stream.wait_stream.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize("offload", [False, True])
def test_sparse_batch_not_allocated_without_local_sparse_layers(sparse_scheduling_case, offload):
    case = sparse_scheduling_case
    case.manager._enable_kv_cache_offload = offload
    if offload:
        case.manager.pp_layers = [3]
    case.manager.prepare_resources(case.scheduled)
    case.create_batch.assert_not_called()
    case.current_stream.assert_not_called()


def _cache_config_case(
    *,
    enabled: bool,
    tokens_per_block: int = 128,
    pp_layers: list[int] | None = None,
    dtype: DataType = DataType.BF16,
    nvfp4_residual_dim: int = 0,
    index_topk: int | None = 512,
    indexer_k_dtype: str = "fp8",
    cache_factor: float = 2.0,
) -> tuple[DeepseekV4CacheManager, KVCacheManagerConfig]:
    manager = object.__new__(DeepseekV4CacheManager)
    manager._enable_kv_cache_offload = enabled
    manager._sparse_offload_topk = index_topk
    manager._sparse_kv_offload = (
        SparseKvCacheOffloadConfig(gpu_cache_factor=cache_factor) if enabled else None
    )
    manager.pp_layers = [0, 1, 2, 3] if pp_layers is None else pp_layers
    manager._compress_ratios = [1, 4, 128, 4]
    manager.compressed_block_sizes = [tokens_per_block // r for r in manager._compress_ratios]
    manager.tokens_per_block = tokens_per_block
    manager.head_dim = 512
    manager.index_head_dim = 128
    manager.dtype = DataType.FP8 if dtype == DataType.NVFP4 else dtype
    manager._indexer_k_dtype = indexer_k_dtype
    manager._use_nvfp4_compress = dtype == DataType.NVFP4
    manager._nvfp4_residual_dim = nvfp4_residual_dim
    manager.use_fp8_ds_mla = False
    manager._swa_window_size = 128
    manager.max_seq_len = 1024
    manager.max_batch_size = 2
    manager.is_estimating_kv_cache = False
    manager.max_cuda_graph_batch_size = None
    manager.max_attention_window_vec = [None] * len(manager.pp_layers)
    manager.max_draft_len = manager._max_draft_len = 0
    manager.num_extra_kv_tokens = 0
    config = KVCacheManagerConfig(
        tokens_per_block=tokens_per_block,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 20)],
        layers=[],
    )
    return manager, config


@pytest.mark.cpu_only
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("tokens_per_block", [128, 256])
@pytest.mark.parametrize("pp_layers", [[0, 1, 2, 3], [2, 3], [0, 2]])
@pytest.mark.parametrize("index_topk", [1, 512])
@pytest.mark.parametrize("indexer_k_dtype", ["fp8", "fp4"])
@pytest.mark.parametrize(
    ("dtype", "nvfp4_residual_dim", "data_row_bytes", "scale_row_bytes"),
    [
        (DataType.BF16, 0, 1024, None),
        (DataType.FP8, 0, 512, None),
        (DataType.NVFP4, 0, 256, 32),
        (DataType.NVFP4, 64, 288, 36),
    ],
)
def test_sparse_offload_cache_roles(
    enabled: bool,
    tokens_per_block: int,
    pp_layers: list[int],
    index_topk: int,
    indexer_k_dtype: str,
    dtype: DataType,
    nvfp4_residual_dim: int,
    data_row_bytes: int,
    scale_row_bytes: int | None,
) -> None:
    manager, config = _cache_config_case(
        enabled=enabled,
        tokens_per_block=tokens_per_block,
        pp_layers=pp_layers,
        dtype=dtype,
        nvfp4_residual_dim=nvfp4_residual_dim,
        index_topk=index_topk,
        indexer_k_dtype=indexer_k_dtype,
    )
    result = manager._build_cache_config(config)

    assert result.tokens_per_block == tokens_per_block
    assert len(result.layers) == manager._num_manager_layers
    for layer in result.layers:
        for buffer in layer.buffers:
            if not buffer.is_sparse:
                assert buffer.tokens_per_block_override is None
                assert buffer.sparse_unit_chunks == 1
    layers = {layer.layer_id: layer for layer in result.layers}
    for (model_layer, attn_type), layer_id in manager._layer_attn_to_layer_id.items():
        assert model_layer in pp_layers
        assert manager._manager_layer_id_to_layer_attn[layer_id, attn_type.role] == (
            model_layer,
            attn_type,
        )
        buffer = next(b for b in layers[layer_id].buffers if b.role == attn_type.role)
        ratio = manager._compress_ratios[model_layer]
        assert buffer.is_sparse == (
            enabled and ratio == 4 and attn_type == DeepseekV4AttentionType.COMPRESS
        )
        if attn_type == DeepseekV4AttentionType.COMPRESS:
            row_sizes = {attn_type.role: data_row_bytes}
            if scale_row_bytes is not None:
                row_sizes[deepseek_v4_cache.COMPRESS_BLOCK_SCALE_ROLE] = scale_row_bytes
            compressed_rows = tokens_per_block // ratio
            for role, row_bytes in row_sizes.items():
                role_buffer = next(b for b in layers[layer_id].buffers if b.role == role)
                if enabled and ratio == 4:
                    assert role_buffer.is_sparse
                    # A sparse buffer's size is its whole page, as for any other buffer;
                    # the layer's sparse_cache says how many units it divides into.
                    assert role_buffer.tokens_per_block_override is None
                    assert role_buffer.sparse_unit_chunks == 1
                    assert role_buffer.size == compressed_rows * row_bytes
                    cache = layers[layer_id].sparse_cache
                    assert cache is not None
                    assert cache.num_selected_blocks == index_topk
                    # Two selections plus the row being written, in whole pages of
                    # compressed rows.
                    cells = -(-(2 * index_topk + 1) // compressed_rows)
                    assert cache.max_cached_units == cells * compressed_rows
                    assert cache.unit_tokens == 4
                else:
                    assert role_buffer.size == compressed_rows * row_bytes
            if ratio == 4:
                indexer_layer_id = manager._layer_attn_to_layer_id[
                    model_layer, DeepseekV4AttentionType.INDEXER_COMPRESS
                ]
                # COMPRESS and INDEXER_COMPRESS are one layer either way: they are one
                # attention computation, and only the first is fetched through the cache.
                assert layer_id == indexer_layer_id


@pytest.mark.cpu_only
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("pp_layers", [[0, 1], [0, 2]])
def test_sparse_offload_cache_config_footer_layout(enabled: bool, pp_layers: list[int]) -> None:
    manager, config = _cache_config_case(enabled=enabled, pp_layers=pp_layers, dtype=DataType.FP8)
    manager.use_fp8_ds_mla = True
    if enabled and 1 in pp_layers:
        with pytest.raises(NotImplementedError, match="separate data and footer-scale regions"):
            manager._build_cache_config(config)
    else:
        result = manager._build_cache_config(config)
        assert all(not buffer.is_sparse for layer in result.layers for buffer in layer.buffers)


@pytest.mark.cpu_only
@pytest.mark.parametrize(("cache_factor", "rows_per_layer"), [(1.0, 544), (1.5, 800), (3.0, 1568)])
def test_sparse_offload_cache_factor_scales_whole_cells(
    cache_factor: float, rows_per_layer: int
) -> None:
    manager, config = _cache_config_case(enabled=True, cache_factor=cache_factor)
    result = manager._build_cache_config(config)

    caches = [layer.sparse_cache for layer in result.layers if layer.sparse_cache is not None]
    assert caches
    # The factor scales the 512 selected rows, the row being written adds one, and
    # the total rounds up to cells of 32 rows.
    assert {cache.max_cached_units for cache in caches} == {rows_per_layer}


@pytest.mark.cpu_only
@pytest.mark.parametrize("index_topk", [None, 0, -1])
def test_sparse_offload_cache_config_requires_positive_topk(index_topk: int | None) -> None:
    manager, config = _cache_config_case(enabled=True, index_topk=index_topk)
    with pytest.raises(ValueError, match="positive index_topk"):
        manager._build_cache_config(config)


@skip_pre_blackwell
@pytest.mark.parametrize("tokens_per_block", [128, 256])
@pytest.mark.parametrize("dtype", [DataType.BF16, DataType.FP8, DataType.NVFP4])
def test_sparse_offload_initializes_host_history_and_gpu_arena(
    tokens_per_block: int, dtype: DataType
) -> None:
    sparse_config = DeepSeekV4SparseAttentionConfig(
        enable_kv_cache_offload=True,
        compress_ratios=[4, 128],
        index_topk=8,
        kv_cache_offload_config={"host_history_size": 64 << 20},
    )
    manager = DeepseekV4CacheManager(
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, max_tokens=1024, host_cache_size=0),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=len(sparse_config.compress_ratios),
        head_dim=512,
        max_batch_size=1,
        tokens_per_block=tokens_per_block,
        max_seq_len=512,
        max_num_tokens=512,
        vocab_size=128,
        mapping=Mapping(),
        dtype=dtype,
        sparse_attn_config=sparse_config,
    )
    try:
        assert manager.get_host_backed_pool_groups()
        page_table = build_page_table_from_manager(manager)
        assert {pool.memory_type for group in page_table.pool_groups for pool in group.pools} == {
            "DRAM",
            "VRAM",
        }
        assert set(manager.get_sparse_offload_descriptors()) == {0}
        assert set(manager._sparse_arena_buffers) == {0}
        data, scales = manager._sparse_arena_buffers[0]
        assert data.is_cuda
        if dtype == DataType.NVFP4:
            assert scales is not None and scales.is_cuda
        else:
            assert scales is None
    finally:
        manager.shutdown()


@pytest.mark.parametrize(("avg_seq_len", "expected"), [(None, 1024), (256, 256)])
def test_typical_seq_len_preserves_deepseek_v4_fallback(
    avg_seq_len: int | None, expected: int
) -> None:
    manager = object.__new__(DeepseekV4CacheManager)
    manager.max_seq_len = 1024

    assert manager._get_typical_seq_len(KvCacheConfig(avg_seq_len=avg_seq_len)) == expected


@pytest.mark.cpu_only
@pytest.mark.parametrize("pool_ratio", [None, [1.0]])
@pytest.mark.parametrize(
    "estimating,graph_batch_size,window,expected_batch_size",
    [
        (False, 2, None, 3),
        (True, 2, None, 2),
        (True, None, None, 3),
        (True, 0, None, 3),
        (True, 4, None, 3),
        (True, 2, 128, 3),
    ],
)
def test_build_cache_config_long_decode_constraint(
    pool_ratio: list[float] | None,
    estimating: bool,
    graph_batch_size: int | None,
    window: int | None,
    expected_batch_size: int,
) -> None:
    manager = object.__new__(DeepseekV4CacheManager)
    manager.pp_layers = [0]
    manager._compress_ratios = [1]
    manager.dtype = DataType.BF16
    manager._use_nvfp4_compress = False
    manager.head_dim = 512 + 64
    manager.index_head_dim = 128
    manager._indexer_k_dtype = "fp8"
    manager.use_fp8_ds_mla = False
    manager._swa_window_size = 128
    manager.tokens_per_block = 128
    manager.max_seq_len = 1024
    manager.max_batch_size = 3
    manager.is_estimating_kv_cache = estimating
    manager.max_cuda_graph_batch_size = graph_batch_size
    manager.max_attention_window_vec = [window]
    manager.max_draft_len = manager._max_draft_len = 4
    manager.num_extra_kv_tokens = 3

    context_constraint = BatchDesc([KVCacheDesc(capacity=259, history_length=0)])
    base_config = KVCacheManagerConfig(
        tokens_per_block=128,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 20)],
        layers=[],
        constraints=[context_constraint] if pool_ratio is None else [],
        initial_pool_ratio=pool_ratio,
    )

    config = manager._build_cache_config(base_config)

    if pool_ratio is None:
        assert config.constraints == [
            context_constraint,
            BatchDesc(
                [KVCacheDesc(capacity=1024, history_length=1023)]
                + [KVCacheDesc(capacity=8, history_length=0)] * (expected_batch_size - 1)
            ),
        ]
        assert base_config.constraints == [context_constraint]
    else:
        assert config.initial_pool_ratio == pool_ratio
        assert config.constraints == []


def test_cache_size_estimation_uses_model_attention_layer_count():
    class FakeModelConfig:
        sparse_attention_config = SimpleNamespace(
            index_head_dim=128,
            compress_ratios=[1, 4, 1, 128],
            indexer_k_dtype="fp8",
            window_size=128,
            enable_kv_cache_offload=False,
        )
        pretrained_config = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
        quant_config = None

        def get_num_attention_layers(self) -> int:
            return len(self.sparse_attention_config.compress_ratios)

    size_per_token = DeepseekV4CacheManager.get_cache_size_per_token(
        FakeModelConfig(),
        Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        tokens_per_block=128,
        is_disagg=True,
    )

    cost = CacheCost.from_raw(size_per_token)
    assert cost.slope > 0
    assert cost.intercept == 0


def test_quota_from_max_tokens_models_context_swa_scratch():
    manager = object.__new__(DeepseekV4CacheManager)
    manager._enable_kv_cache_offload = False
    manager.pp_layers = [0, 1]
    manager._compress_ratios = [4, 4]
    manager.dtype = DataType.BF16
    manager._use_nvfp4_compress = False
    manager.head_dim = 512 + 64
    manager.index_head_dim = 128
    manager._indexer_k_dtype = "fp8"
    manager.use_fp8_ds_mla = False
    manager._swa_window_size = 128
    manager._max_draft_len = 0
    manager._max_num_tokens = 1024
    manager.tokens_per_block = 128
    manager.max_batch_size = 2

    manager.enable_swa_scratch_reuse = True
    small_quota = manager._get_quota_from_max_tokens(1)
    large_quota = manager._get_quota_from_max_tokens(4096)
    scratch_delta = large_quota - small_quota

    assert small_quota < large_quota
    assert small_quota > manager._get_extra_quota_padding()
    assert manager._get_max_tokens_from_quota(small_quota) == 1
    assert manager._get_max_tokens_from_quota(large_quota) == 4096

    manager.enable_swa_scratch_reuse = False
    no_scratch_small_quota = manager._get_quota_from_max_tokens(1)
    no_scratch_large_quota = manager._get_quota_from_max_tokens(4096)
    no_scratch_delta = no_scratch_large_quota - no_scratch_small_quota

    assert no_scratch_small_quota < no_scratch_large_quota
    assert no_scratch_delta > scratch_delta
    assert manager._get_max_tokens_from_quota(no_scratch_large_quota) == 4096


def test_needed_resource_uses_context_swa_scratch_slope():
    manager = object.__new__(DeepseekV4CacheManager)
    manager._enable_kv_cache_offload = False
    manager.pp_layers = [0, 1]
    manager._compress_ratios = [4, 4]
    manager.dtype = DataType.BF16
    manager._use_nvfp4_compress = False
    manager.head_dim = 512 + 64
    manager.index_head_dim = 128
    manager._indexer_k_dtype = "fp8"
    manager.use_fp8_ds_mla = False
    manager._swa_window_size = 128
    manager.tokens_per_block = 128
    manager.num_extra_kv_tokens = 0
    manager.enable_swa_scratch_reuse = True

    context_request = SimpleNamespace(
        is_context_init_state=True,
        is_generation_in_progress_state=False,
        is_generation_to_complete_state=False,
        is_disagg_generation_init_state=False,
        state=LlmRequestState.CONTEXT_INIT,
        prompt_len=10,
        max_new_tokens=100,
    )
    longer_context_request = SimpleNamespace(
        is_context_init_state=True,
        is_generation_in_progress_state=False,
        is_generation_to_complete_state=False,
        is_disagg_generation_init_state=False,
        state=LlmRequestState.CONTEXT_INIT,
        prompt_len=11,
        max_new_tokens=100,
    )
    generation_request = SimpleNamespace(
        is_context_init_state=False,
        is_generation_in_progress_state=True,
        is_generation_to_complete_state=False,
        is_disagg_generation_init_state=False,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        prompt_len=10,
        max_new_tokens=20,
    )
    longer_generation_request = SimpleNamespace(
        is_context_init_state=False,
        is_generation_in_progress_state=True,
        is_generation_to_complete_state=False,
        is_disagg_generation_init_state=False,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        prompt_len=10,
        max_new_tokens=21,
    )

    non_sliding_attn_size_per_token = manager.get_cache_bytes_per_token()
    context_bytes = manager.get_needed_resource_to_completion(context_request)
    longer_context_bytes = manager.get_needed_resource_to_completion(longer_context_request)
    generation_bytes = manager.get_needed_resource_to_completion(generation_request)
    longer_generation_bytes = manager.get_needed_resource_to_completion(longer_generation_request)

    assert non_sliding_attn_size_per_token > 0
    assert context_bytes > context_request.prompt_len * non_sliding_attn_size_per_token
    assert longer_context_bytes - context_bytes > non_sliding_attn_size_per_token
    assert longer_generation_bytes - generation_bytes == non_sliding_attn_size_per_token


def _view_fp8_as_uint8(buffer: torch.Tensor) -> torch.Tensor:
    """View an FP8 buffer as uint8. Non-FP8 buffers are returned as-is."""
    if buffer.dtype == torch.float8_e4m3fn:
        return buffer.view(torch.uint8)
    return buffer


@pytest.fixture(params=[False, True], ids=["scratch_reuse_disabled", "scratch_reuse_enabled"])
def scratch_reuse_enabled(request) -> bool:
    return request.param


@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(80000)
class TestDeepseekV4CacheManager:
    # deepseek_v4 specific param
    head_dim = 512
    index_head_dim = 128
    window_size = 128
    vocab_size = 129280
    sparse_layer_ratio = 4
    overlap_compress_layer_ratio = 4

    # indexer quantization config
    indexer_dtype = DataType.FP8
    indexer_scale_dtype = DataType.FLOAT

    # cache manager specific param
    tokens_per_block = 128

    @staticmethod
    def _attention_op_block_offsets_shape(
        cache_manager: DeepseekV4CacheManager, num_seqs: int
    ) -> tuple[int, int, int, int]:
        return (
            cache_manager.num_attention_op_pools,
            num_seqs,
            2,
            cache_manager.max_blocks_per_seq,
        )

    @staticmethod
    def _sliding_block_tables_shape(
        cache_manager: DeepseekV4CacheManager, num_seqs: int
    ) -> tuple[int, int, int, int]:
        return (
            cache_manager.num_local_layers,
            len(DEEPSEEK_V4_SLIDING_ATTENTION),
            num_seqs,
            cache_manager.max_blocks_per_seq,
        )

    def _is_compress_layer(self, compress_ratio: int) -> bool:
        """Check if a layer uses compression based on its compress ratio.

        Args:
            compress_ratio: The compression ratio for the layer

        Returns:
            True if the layer uses compression (ratio > 1)
        """
        return compress_ratio > 1

    def _is_sparse_layer(self, compress_ratio: int) -> bool:
        """Check if a layer uses sparse attention based on its compress ratio.

        Args:
            compress_ratio: The compression ratio for the layer

        Returns:
            True if the layer uses sparse attention (ratio == 4)
        """
        return compress_ratio == self.sparse_layer_ratio

    def _is_overlap_compressor(self, compress_ratio: int) -> bool:
        """Check if a layer uses overlap compressor based on its compress ratio.

        Args:
            compress_ratio: The compression ratio for the layer

        Returns:
            True if the layer uses overlap compressor (ratio == 4)
        """
        return compress_ratio == self.overlap_compress_layer_ratio

    def _get_window_size(self, compress_ratio: int, attn_type: DeepseekV4AttentionType) -> int:
        """Get the window size for a layer based on its compress ratio and attention type.

        Args:
            compress_ratio: The compression ratio for the layer
            attn_type: The attention type

        Returns:
            The window size for the layer
        """
        state_factor = 2 if self._is_overlap_compressor(compress_ratio) else 1
        if attn_type == DeepseekV4AttentionType.SWA:
            return self.window_size
        elif attn_type in [
            DeepseekV4AttentionType.COMPRESSOR_KV,
            DeepseekV4AttentionType.COMPRESSOR_SCORE,
            DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
            DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
        ]:
            return state_factor * compress_ratio
        elif attn_type in [
            DeepseekV4AttentionType.COMPRESS,
            DeepseekV4AttentionType.INDEXER_COMPRESS,
        ]:
            return None

    def _create_deepseek_v4_cache_manager(
        self,
        tokens_per_block: int,
        max_batch_size: int,
        max_seq_len: int,
        compress_ratios: List[int],
        dtype: DataType,
        compressor_dtype: DataType,
        max_input_len: Optional[int] = None,
        is_draft: bool = False,
        tp_size: int = 1,
        enable_attention_dp: bool = False,
        spec_config: object | None = None,
        indexer_k_dtype: str | None = None,
        enable_swa_scratch_reuse: bool = True,
        cold_page_codec_provider: object | None = None,
        host_cache_size: int | None = None,
        max_gpu_total_bytes: int = 0,
    ) -> Tuple[DeepseekV4CacheManager, DeepSeekV4SparseAttentionConfig]:
        """Helper to create a DeepseekV4CacheManager for testing."""

        # Create sparse attention config
        config_kwargs = {}
        if indexer_k_dtype is not None:
            config_kwargs["indexer_k_dtype"] = indexer_k_dtype
        sparse_attn_config = DeepSeekV4SparseAttentionConfig(
            index_head_dim=self.index_head_dim,
            window_size=self.window_size,
            compress_ratios=compress_ratios,
            **config_kwargs,
        )

        # Create KV cache config
        if max_input_len is None:
            max_input_len = max_seq_len
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=max_seq_len * max_batch_size,
            event_buffer_max_size=0,
            enable_swa_scratch_reuse=enable_swa_scratch_reuse,
            host_cache_size=host_cache_size,
            max_gpu_total_bytes=max_gpu_total_bytes,
        )

        # Create mapping (single GPU, no parallelism)
        mapping = Mapping(
            world_size=tp_size,
            rank=0,
            tp_size=tp_size,
            pp_size=1,
            enable_attention_dp=enable_attention_dp,
        )

        # Create cache manager
        cache_manager = DeepseekV4CacheManager(
            kv_cache_config=kv_cache_config,
            kv_cache_type=CacheTypeCpp.SELFKONLY,
            num_layers=len(compress_ratios),
            num_kv_heads=1,
            head_dim=self.head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            mapping=mapping,
            dtype=dtype,
            compressor_dtype=compressor_dtype,
            vocab_size=self.vocab_size,
            max_num_tokens=max_batch_size * (max_input_len + 1),
            sparse_attn_config=sparse_attn_config,
            is_draft=is_draft,
            spec_config=spec_config,
            cold_page_codec_provider=cold_page_codec_provider,
        )

        return cache_manager, sparse_attn_config

    def test_nvfp4_cold_page_codec_accepts_real_csa_hca_lifecycle(self):
        provider = Nvfp4ColdPageQuantizationCompression(
            ColdPageQuantizationCompressionConfig(),
            pretrained_config=SimpleNamespace(model_type="deepseek_v4"),
        )
        with patch.object(
            native_kvcc,
            "create_python_cold_page_codec",
            wraps=native_kvcc.create_python_cold_page_codec,
        ) as create_codec:
            cache_manager, _ = self._create_deepseek_v4_cache_manager(
                tokens_per_block=256,
                max_batch_size=1,
                max_seq_len=256,
                compress_ratios=[4, 128],
                dtype=DataType.BF16,
                compressor_dtype=DataType.FLOAT,
                indexer_k_dtype="fp4",
                cold_page_codec_provider=provider,
                host_cache_size=64 << 20,
            )

        try:
            assert create_codec.call_count == 1
            csa_layer_id = cache_manager._layer_attn_to_layer_id[
                0, DeepseekV4AttentionType.COMPRESS
            ]
            hca_layer_id = cache_manager._layer_attn_to_layer_id[
                1, DeepseekV4AttentionType.COMPRESS
            ]
            assert cache_manager.impl.get_layer_group_id(
                csa_layer_id
            ) == cache_manager.impl.get_layer_group_id(hca_layer_id)

            codec_state = create_codec.call_args.args[1]
            assert set(codec_state.layer_ids) == {int(csa_layer_id), int(hca_layer_id)}
            assert len(codec_state.lifecycle_metadata) == 1
            metadata = codec_state.lifecycle_metadata[0]
            assert metadata.num_buffers == 3
            assert metadata.integers[:3, 1].tolist() == [2, 1, 1]
            assert metadata.cold_page_bytes == 27136
        finally:
            cache_manager.shutdown()

    @pytest.mark.parametrize(
        ("dtype", "skip_rope_quantization", "residual_dim", "cold_page_bytes"),
        [
            (DataType.BF16, False, None, 13568),
            (DataType.FP8, False, None, 13056),
            (DataType.BF16, False, 0, 12416),
            (DataType.FP8, False, 0, 11904),
            (DataType.BF16, True, None, 15360),
            (DataType.FP8, True, None, 12800),
        ],
    )
    def test_nvfp4_cold_page_codec_migrates_real_csa_hca_through_host(
        self,
        dtype: DataType,
        skip_rope_quantization: bool,
        residual_dim: int | None,
        cold_page_bytes: int,
    ) -> None:
        prompt_len = 64 * self.tokens_per_block
        pressure_len = 65 * self.tokens_per_block
        compress_ratios = [4, 128]
        # Leave the switch unset in default cases; test the opt-in lossless layout too.
        if skip_rope_quantization:
            compression_config = ColdPageQuantizationCompressionConfig(skip_rope_quantization=True)
        elif residual_dim == 0:
            compression_config = ColdPageQuantizationCompressionConfig(nvfp4_residual_dim=0)
        else:
            compression_config = ColdPageQuantizationCompressionConfig()
        provider = Nvfp4ColdPageQuantizationCompression(
            compression_config,
            pretrained_config=SimpleNamespace(model_type="deepseek_v4"),
        )
        requests: list[LlmRequest] = []

        with (
            patch.object(
                native_kvcc,
                "create_python_cold_page_codec",
                wraps=native_kvcc.create_python_cold_page_codec,
            ) as create_codec,
            patch.object(
                native_kvcc,
                "nvfp4_cold_page_encode",
                wraps=native_kvcc.nvfp4_cold_page_encode,
            ) as encode,
            patch.object(
                native_kvcc,
                "nvfp4_cold_page_decode",
                wraps=native_kvcc.nvfp4_cold_page_decode,
            ) as decode,
        ):
            cache_manager, sparse_attn_config = self._create_deepseek_v4_cache_manager(
                tokens_per_block=self.tokens_per_block,
                max_batch_size=1,
                max_seq_len=pressure_len,
                compress_ratios=compress_ratios,
                dtype=dtype,
                compressor_dtype=DataType.FLOAT,
                indexer_k_dtype="fp4",
                enable_swa_scratch_reuse=False,
                cold_page_codec_provider=provider,
                host_cache_size=64 << 20,
                # Below what the constraints need, so the manager sizes every pool group at
                # its floor: room for one max-length request, which the pressure request
                # then has to evict.
                max_gpu_total_bytes=1 << 20,
            )

            try:
                assert create_codec.call_count == 1
                metadata = create_codec.call_args.args[1].lifecycle_metadata[0]
                expected_transform = 2 if not skip_rope_quantization and residual_dim is None else 0
                assert metadata.integers[:3, 1].tolist() == [expected_transform, 1, 1]
                first = self._create_request(request_id=0, prompt_len=prompt_len)
                requests.append(first)
                expected = self._create_random_cache(
                    seq_len=prompt_len,
                    head_dim=self.head_dim,
                    sparse_attn_config=sparse_attn_config,
                    dtype=binding_to_torch_dtype(dtype),
                    compressor_dtype=binding_to_torch_dtype(DataType.FLOAT),
                )
                expected_csa, _ = expected[0, DeepseekV4AttentionType.COMPRESS]
                quantized_elements = 448 if skip_rope_quantization else self.head_dim
                quantized_values = torch.linspace(
                    -1.0,
                    1.0,
                    quantized_elements,
                    dtype=torch.float32,
                    device=expected_csa.device,
                ).expand(expected_csa.size(0), -1)
                if dtype == DataType.FP8:
                    quantized_values = quantized_values.to(torch.float8_e4m3fn).view(torch.uint8)
                else:
                    quantized_values = quantized_values.to(expected_csa.dtype)
                expected_csa[:, :quantized_elements] = quantized_values
                assert cache_manager.prepare_context(first)
                assert cache_manager.resize_context(first, first.context_chunk_size)
                self._write_request_prefill(first, prompt_len, cache_manager, expected)

                scheduled_batch = ScheduledRequests()
                scheduled_batch.context_requests_last_chunk = [first]
                first.context_current_position = prompt_len
                first.add_new_token(prompt_len, 0)
                cache_manager.update_context_resources(scheduled_batch)
                cache_manager.update_resources(scheduled_batch)
                torch.cuda.synchronize()

                first_cache = cache_manager.kv_cache_map[first.py_request_id]
                cache_manager.suspend_request(first)

                # The pressure request displaces the suspended one only when the GPU tier holds
                # no slack beyond its slot floors, which depends on how the allocator packed the
                # pools. Shrink the tier to the least the floors allow; resize refuses anything
                # smaller.
                gpu_quota = cache_manager.impl.get_quota(0)
                grain = _introspection.tier_allocation_granularity(CacheTier.GPU_MEM, gpu_quota)
                while cache_manager.impl.resize(0, gpu_quota - grain):
                    gpu_quota -= grain

                pressure = self._create_request(request_id=1, prompt_len=pressure_len)
                requests.append(pressure)
                assert cache_manager.prepare_context(pressure)
                assert cache_manager.resize_context(pressure, pressure.context_chunk_size)
                torch.cuda.synchronize()

                cold_counts = _introspection.active_page_stats(first_cache)[0]
                assert cold_counts[1] > 0
                assert encode.call_count > 0
                encoded_pages = sum(call.args[1] for call in encode.call_args_list)
                assert encoded_pages > 0
                assert all(call.args[5] == 3 for call in encode.call_args_list)
                assert all(call.args[7] == cold_page_bytes for call in encode.call_args_list)

                cache_manager.free_resources(pressure)
                assert cache_manager.resume_request(first)
                torch.cuda.synchronize()

                assert _introspection.active_page_stats(first_cache)[0][1] == 0
                assert decode.call_count > 0
                assert sum(call.args[1] for call in decode.call_args_list) == encoded_pages

                actual = self._read_request(
                    first,
                    prompt_len,
                    cache_manager,
                    compress_ratios,
                )
                expected_csa, _ = expected[0, DeepseekV4AttentionType.COMPRESS]
                actual_csa, _ = actual[0, DeepseekV4AttentionType.COMPRESS]
                expected_quantized = expected_csa[:, :quantized_elements]
                actual_quantized = actual_csa[:, :quantized_elements]
                assert not torch.equal(actual_csa[:, :448], expected_csa[:, :448])
                if dtype == DataType.FP8:
                    expected_quantized = expected_quantized.view(torch.float8_e4m3fn)
                    actual_quantized = actual_quantized.view(torch.float8_e4m3fn)
                torch.testing.assert_close(
                    actual_quantized.float(),
                    expected_quantized.float(),
                    rtol=0.25,
                    atol=0.02,
                )
                if skip_rope_quantization:
                    assert torch.equal(actual_csa[:, 448:], expected_csa[:, 448:])
                elif residual_dim == 0:
                    assert not torch.equal(actual_csa[:, 448:], expected_csa[:, 448:])

                expected_indexer = expected[0, DeepseekV4AttentionType.INDEXER_COMPRESS]
                actual_indexer = actual[0, DeepseekV4AttentionType.INDEXER_COMPRESS]
                assert torch.equal(actual_indexer[0], expected_indexer[0])
                assert torch.equal(actual_indexer[1], expected_indexer[1])

                expected_hca, _ = expected[1, DeepseekV4AttentionType.COMPRESS]
                actual_hca, _ = actual[1, DeepseekV4AttentionType.COMPRESS]
                assert torch.equal(actual_hca, expected_hca)
            finally:
                for request in requests:
                    if request.py_request_id in cache_manager.kv_cache_map:
                        cache_manager.free_resources(request)
                cache_manager.shutdown()

    def _create_request(self, request_id: int, prompt_len: int) -> LlmRequest:
        """Helper to create a test LlmRequest.

        Args:
            request_id: Unique request identifier
            prompt_len: Prompt length (number of tokens)

        Returns:
            LlmRequest instance
        """
        input_tokens = list(range(prompt_len))
        request = LlmRequest(
            request_id=request_id,
            max_new_tokens=1024,
            input_tokens=input_tokens,
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )

        return request

    def _rand_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if dtype in (torch.uint8, torch.float8_e4m3fn):
            # Use uint8 for both uint8 and FP8 (same 1-byte layout)
            return torch.randint(0, 255, shape, dtype=torch.uint8, device=device)
        else:
            return torch.randn(shape, dtype=dtype, device=device) * 1000.0

    def _create_random_cache(
        self,
        seq_len: int,
        head_dim: int,
        sparse_attn_config: DeepSeekV4SparseAttentionConfig,
        dtype: torch.dtype,
        compressor_dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> _RequestCache:
        """Helper to create random cache values for all layers and attention types.

        Args:
            seq_len: Sequence length
            head_dim: Head dimension for regular attention
            sparse_attn_config: Sparse attention configuration

        Returns:
            Dictionary mapping (layer_idx, attn_type) to (values, scales) tuples.
            scales is None for non-quantized attention types.
        """
        device = device or torch.device("cuda")
        cache: _RequestCache = {}

        for layer, ratio in enumerate(sparse_attn_config.compress_ratios):
            is_overlap = self._is_overlap_compressor(ratio)

            cache[layer, DeepseekV4AttentionType.SWA] = (
                self._rand_tensor((seq_len, head_dim), dtype, device),
                None,
            )

            if self._is_compress_layer(ratio):
                compressor_dim = 2 * head_dim if is_overlap else head_dim
                cache[layer, DeepseekV4AttentionType.COMPRESS] = (
                    self._rand_tensor((seq_len // ratio, head_dim), dtype, device),
                    None,
                )
                cache[layer, DeepseekV4AttentionType.COMPRESSOR_KV] = (
                    self._rand_tensor((seq_len, compressor_dim), compressor_dtype, device),
                    None,
                )
                cache[layer, DeepseekV4AttentionType.COMPRESSOR_SCORE] = (
                    self._rand_tensor((seq_len, compressor_dim), compressor_dtype, device),
                    None,
                )

            if self._is_sparse_layer(ratio):
                # Indexer KV cache stores raw quantized bytes plus raw scale bytes.
                indexer_dim = sparse_attn_config.index_head_dim
                indexer_num_tokens = seq_len // ratio
                indexer_k_dtype = sparse_attn_config.indexer_k_dtype
                value_dim, scale_dim, scale_dtype = self._indexer_cache_layout(indexer_k_dtype)
                indexer_values = self._rand_tensor(
                    (indexer_num_tokens, value_dim), torch.uint8, device
                )
                indexer_scales = self._rand_tensor(
                    (indexer_num_tokens, scale_dim), scale_dtype, device
                )
                cache[layer, DeepseekV4AttentionType.INDEXER_COMPRESS] = (
                    indexer_values,
                    indexer_scales,
                )

                indexer_compressor_dim = 2 * indexer_dim if is_overlap else indexer_dim
                cache[layer, DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV] = (
                    self._rand_tensor((seq_len, indexer_compressor_dim), compressor_dtype, device),
                    None,
                )
                cache[layer, DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE] = (
                    self._rand_tensor((seq_len, indexer_compressor_dim), compressor_dtype, device),
                    None,
                )

        return cache

    def _indexer_cache_layout(self, indexer_k_dtype: str) -> Tuple[int, int, torch.dtype]:
        if indexer_k_dtype == "fp8":
            return self.index_head_dim, self.index_head_dim // 128, torch.float32
        if indexer_k_dtype == "fp4":
            return self.index_head_dim // 2, self.index_head_dim // 32, torch.uint8
        raise ValueError(f"Unsupported indexer_k_dtype {indexer_k_dtype!r}")

    def _split_blockwise_buffer(
        self,
        buffer: torch.Tensor,
        indexer_k_dtype: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split an indexer K cache buffer into value and scale buffers.

        Args:
            buffer: The raw indexer K cache buffer.

        Returns:
            Tuple of (value_buffer, scale_buffer)
        """
        num_blocks, tokens_per_block, bytes_per_token = buffer.shape
        bytes_per_block = bytes_per_token * tokens_per_block
        value_dim, scale_dim, scale_dtype = self._indexer_cache_layout(indexer_k_dtype)
        scale_bytes = scale_dim * scale_dtype.itemsize

        # Get value buffer
        value_shape = (num_blocks, tokens_per_block, value_dim)
        value_stride = (bytes_per_block, value_dim, 1)
        value_buffer = buffer.as_strided(value_shape, value_stride, 0).view(torch.uint8)

        # Get scale buffer
        scale_shape = (num_blocks, tokens_per_block, scale_bytes)
        scale_stride = (bytes_per_block, scale_bytes, 1)
        scale_offset = value_dim * tokens_per_block
        scale_buffer = buffer.as_strided(scale_shape, scale_stride, scale_offset).view(scale_dtype)

        return value_buffer, scale_buffer

    def _prefill_write_paged_cache(
        self,
        buffer: torch.Tensor,
        block_indices: List[int],
        values: torch.Tensor,
    ) -> None:
        """Write context values to a paged cache buffer.

        Args:
            buffer: The cache buffer to write to (shape: [num_blocks, tokens_per_block, dim_per_token])
            block_indices: List of block indices to write to
            values: Values to write (shape: [seq_len, dim_per_token])
        """
        assert buffer.size(2) == values.size(1), f"{buffer.size(2)=} != {values.size(1)=}"
        tokens_per_block = buffer.size(1)
        seq_len, dim_per_token = values.shape

        num_blocks = (seq_len + tokens_per_block - 1) // tokens_per_block
        assert all(idx != BAD_PAGE_INDEX for idx in block_indices[:num_blocks]), (
            f"{block_indices[:num_blocks]=} contains BAD_PAGE_INDEX"
        )

        if seq_len % tokens_per_block != 0:
            # pad the values to the nearest multiple of tokens_per_block
            pad_len = tokens_per_block - (seq_len % tokens_per_block)
            values = torch.cat(
                [
                    values,
                    self._rand_tensor((pad_len, dim_per_token), values.dtype, values.device),
                ],
                dim=0,
            )

        values_blocks = values.reshape(num_blocks, tokens_per_block, dim_per_token)
        buffer[block_indices[:num_blocks]] = values_blocks

    def _decode_write_paged_cache(
        self,
        buffer: torch.Tensor,
        block_indices: List[int],
        token_idx: int,
        value: torch.Tensor,
    ) -> None:
        """Simulate the decode phrase. Write one new token to the cache.

        Args:
            buffer: The cache buffer to write to (shape: [num_blocks, tokens_per_block, dim_per_token])
            block_indices: List of block indices to write to
            token_idx: Index of the new token to write
            value: Value to write (shape: [dim_per_token])
        """
        assert buffer.size(2) == value.size(0), f"{buffer.size(2)=} != {value.size(0)=}"
        num_blocks, tokens_per_block, _ = buffer.shape

        block_idx = token_idx // tokens_per_block
        block_offset = token_idx % tokens_per_block
        assert block_idx < num_blocks, f"{block_idx=} >= {num_blocks=}"
        assert block_indices[block_idx] != BAD_PAGE_INDEX, (
            f"{block_indices[block_idx]=} == BAD_PAGE_INDEX"
        )

        buffer[block_indices[block_idx], block_offset] = value

    def _read_paged_cache(
        self, buffer: torch.Tensor, block_indices: List[int], seq_len: int, window_size: int | None
    ) -> torch.Tensor:
        """Read values from a paged cache buffer.

        Args:
            buffer: The cache buffer to read from (shape: [num_blocks, tokens_per_block, dim_per_token])
            block_indices: List of block/page indices to read from
            seq_len: Sequence length
            window_size: sliding window size to read from the cache

        Returns:
            Tensor containing the read values (shape: [seq_len, dim_per_token] or [window_size, dim_per_token]
            if window_size is given and seq_len > window_size)
        """
        _, tokens_per_block, dim_per_token = buffer.shape

        # check if all blocks within the window are valid
        end_block_idx = (seq_len + tokens_per_block - 1) // tokens_per_block
        if window_size is not None:
            start_block_idx = (seq_len - window_size + tokens_per_block - 1) // tokens_per_block
        else:
            start_block_idx = 0
        assert all(idx != BAD_PAGE_INDEX for idx in block_indices[start_block_idx:end_block_idx]), (
            f"{block_indices[start_block_idx:end_block_idx]=} contains BAD_PAGE_INDEX"
        )

        # read values from the cache
        values = buffer[block_indices].reshape(-1, dim_per_token)[:seq_len]
        if window_size is not None and seq_len > window_size:
            values = values[-window_size:]
        return values

    def _get_page_indices(
        self,
        req: LlmRequest,
        cache_manager: DeepseekV4CacheManager,
        layer_idx: int,
        attn_type: DeepseekV4AttentionType,
        num_contexts: int = 1,
    ) -> torch.Tensor:
        if attn_type == DeepseekV4AttentionType.COMPRESS:
            compress_block_tables = torch.empty(
                1,
                cache_manager.max_blocks_per_seq,
                dtype=torch.int32,
                device="cpu",
            )
            cache_manager.copy_batch_compress_block_tables(
                compress_block_tables,
                [req.py_request_id],
                compress_ratio=cache_manager._compress_ratios[layer_idx],
                beam_width=1,
                num_contexts=num_contexts,
                num_seqs=1,
            )
            return compress_block_tables[0]

        if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
            host_block_table = torch.empty(
                1,
                cache_manager.max_blocks_per_seq,
                dtype=torch.int32,
                device="cpu",
            )
            cache_manager.copy_batch_indexer_compress_block_tables(
                host_block_table,
                [req.py_request_id],
                beam_width=1,
                num_contexts=num_contexts,
                num_seqs=1,
            )
            return host_block_table[0]

        sliding_block_tables = torch.empty(
            self._sliding_block_tables_shape(cache_manager, 1),
            dtype=torch.int32,
            device="cuda",
        )
        with torch.cuda.stream(cache_manager._stream):
            cache_manager.compute_sliding_block_tables(
                [req.py_request_id],
                num_contexts=num_contexts,
            )
            cache_manager.copy_batch_sliding_block_tables(
                sliding_block_tables,
                [req.py_request_id],
                num_contexts=num_contexts,
                num_seqs=1,
            )
        cache_manager._stream.synchronize()
        return sliding_block_tables.cpu()[
            cache_manager.layer_offsets[layer_idx],
            attn_type.value,
            0,
        ]

    def _prepare_mixed_copy_batch(
        self,
        cache_manager: DeepseekV4CacheManager,
        prompt_len: int,
    ) -> tuple[list[LlmRequest], int]:
        requests = [self._create_request(request_id, prompt_len) for request_id in range(3)]
        for req in requests:
            assert cache_manager.prepare_context(req)
            assert cache_manager.resize_context(req, req.context_chunk_size)

        gen_req = requests[-1]
        scheduled_batch = ScheduledRequests()
        scheduled_batch.context_requests_last_chunk = [gen_req]
        gen_req.context_current_position = prompt_len
        gen_req.add_new_token(prompt_len, 0)
        cache_manager.update_context_resources(scheduled_batch)
        cache_manager.update_resources(scheduled_batch)
        assert cache_manager.try_allocate_generation(gen_req)
        return requests, len(requests) - 1

    def _reference_copy_batch_page_indices(
        self,
        cache_manager: DeepseekV4CacheManager,
        request_ids: list[int],
        num_contexts: int,
        layer_idx: int,
        attn_type: DeepseekV4AttentionType,
        page_index_mode: PageIndexMode,
    ) -> torch.Tensor:
        layer_id = cache_manager._layer_attn_to_layer_id[layer_idx, attn_type]
        pool_id = cache_manager.layer_to_pool_mapping_dict[layer_id]
        converter = cache_manager.impl.get_page_index_converter(layer_id, attn_type.role)
        copy_idx = cache_manager.index_mapper.get_copy_index(request_ids, num_contexts, 1)

        expected = torch.full(
            (len(request_ids), cache_manager.max_blocks_per_seq),
            BAD_PAGE_INDEX,
            dtype=torch.int32,
            device="cpu",
        )
        for row, request_id in enumerate(request_ids):
            base_indices = cache_manager.host_kv_cache_block_offsets[
                pool_id,
                int(copy_idx[row]),
                0,
            ].tolist()
            scratch_range = None
            scratch_slots = []
            if row < num_contexts and page_index_mode == PageIndexMode.PER_LAYER:
                cache = cache_manager.kv_cache_map[request_id]
                scratch_range = cache.get_swa_scratch_range(pool_id) or None
                scratch_slots = cache.get_scratch_slots(pool_id)
            converted = converter(base_indices, page_index_mode, scratch_range, scratch_slots)
            expected[row, : len(converted)] = torch.tensor(converted, dtype=torch.int32)
        return expected

    def _write_request_prefill(
        self,
        req: LlmRequest,
        prompt_len: int,
        cache_manager: DeepseekV4CacheManager,
        cache_values: _RequestCache,
    ) -> None:
        """Write cache values for a request to the cache manager.

        Args:
            req: The request to write cache for
            prompt_len: Prompt length
            cache_manager: The cache manager instance
            cache_values: Request's cache to write
        """
        compress_ratios = cache_manager._compress_ratios
        for (layer_idx, attn_type), (values, scales) in cache_values.items():
            page_indices = self._get_page_indices(
                req,
                cache_manager,
                layer_idx,
                attn_type,
            )

            if attn_type in [
                DeepseekV4AttentionType.COMPRESS,
                DeepseekV4AttentionType.INDEXER_COMPRESS,
            ]:
                seq_len = prompt_len // compress_ratios[layer_idx]
            else:
                seq_len = prompt_len

            buffer = _view_fp8_as_uint8(cache_manager.get_buffers(layer_idx, attn_type))
            if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
                values_buffer, scales_buffer = self._split_blockwise_buffer(
                    buffer, cache_manager._indexer_k_dtype
                )
                self._prefill_write_paged_cache(
                    buffer=values_buffer,
                    block_indices=page_indices,
                    values=values[:seq_len],
                )
                self._prefill_write_paged_cache(
                    buffer=scales_buffer,
                    block_indices=page_indices,
                    values=scales[:seq_len],
                )
            else:
                self._prefill_write_paged_cache(
                    buffer=buffer,
                    block_indices=page_indices,
                    values=values[:seq_len],
                )

    def _write_request_decode(
        self,
        req: LlmRequest,
        token_idx: int,
        cache_manager: DeepseekV4CacheManager,
        cache_values: _RequestCache,
    ) -> None:
        """Simulate the decode phrase. Write one new token to the cache.

        Args:
            req: The request to write cache for
            token_idx: Index of the new token to write
            cache_manager: The cache manager instance
            cache_values: Request's cache to write
        """
        compress_ratios = cache_manager._compress_ratios
        for (layer_idx, attn_type), (values, scales) in cache_values.items():
            block_indices = self._get_page_indices(
                req,
                cache_manager,
                layer_idx,
                attn_type,
            )

            compressed_token_idx = token_idx
            if attn_type in [
                DeepseekV4AttentionType.COMPRESS,
                DeepseekV4AttentionType.INDEXER_COMPRESS,
            ]:
                if (token_idx + 1) % compress_ratios[layer_idx] != 0:
                    # skip if current token will not trigger compression
                    continue
                compressed_token_idx = token_idx // compress_ratios[layer_idx]

            buffer = _view_fp8_as_uint8(cache_manager.get_buffers(layer_idx, attn_type))
            if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
                values_buffer, scales_buffer = self._split_blockwise_buffer(
                    buffer, cache_manager._indexer_k_dtype
                )
                self._decode_write_paged_cache(
                    buffer=values_buffer,
                    block_indices=block_indices,
                    token_idx=compressed_token_idx,
                    value=values[compressed_token_idx],
                )
                self._decode_write_paged_cache(
                    buffer=scales_buffer,
                    block_indices=block_indices,
                    token_idx=compressed_token_idx,
                    value=scales[compressed_token_idx],
                )
            else:
                self._decode_write_paged_cache(
                    buffer=buffer,
                    block_indices=block_indices,
                    token_idx=compressed_token_idx,
                    value=values[compressed_token_idx],
                )

    def _read_request(
        self,
        req: LlmRequest,
        seq_len: int,
        cache_manager: DeepseekV4CacheManager,
        compress_ratios: List[int],
    ) -> _RequestCache:
        """Read cache values for a request from the cache manager.

        Args:
            req: The request to read cache for
            seq_len: Sequence length
            cache_manager: The cache manager instance
            compress_ratios: Compression ratios for each layer

        Returns:
            Request's cache
        """
        cache_values: _RequestCache = {}
        for layer, ratio in enumerate(compress_ratios):
            attn_types = [DeepseekV4AttentionType.SWA]
            if self._is_compress_layer(ratio):
                attn_types.extend(
                    [
                        DeepseekV4AttentionType.COMPRESS,
                        DeepseekV4AttentionType.COMPRESSOR_KV,
                        DeepseekV4AttentionType.COMPRESSOR_SCORE,
                    ]
                )
            if self._is_sparse_layer(ratio):
                attn_types.extend(
                    [
                        DeepseekV4AttentionType.INDEXER_COMPRESS,
                        DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
                        DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
                    ]
                )

            # read cache values for each attention type
            for attn_type in attn_types:
                page_indices = self._get_page_indices(
                    req,
                    cache_manager,
                    layer,
                    attn_type,
                )
                if attn_type in [
                    DeepseekV4AttentionType.COMPRESS,
                    DeepseekV4AttentionType.INDEXER_COMPRESS,
                ]:
                    attn_len = seq_len // ratio
                else:
                    attn_len = seq_len
                window_size = self._get_window_size(ratio, attn_type)

                buffer = _view_fp8_as_uint8(cache_manager.get_buffers(layer, attn_type))
                if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
                    values_buffer, scales_buffer = self._split_blockwise_buffer(
                        buffer, cache_manager._indexer_k_dtype
                    )
                    values = self._read_paged_cache(
                        buffer=values_buffer,
                        block_indices=page_indices,
                        seq_len=attn_len,
                        window_size=window_size,
                    )
                    scales = self._read_paged_cache(
                        buffer=scales_buffer,
                        block_indices=page_indices,
                        seq_len=attn_len,
                        window_size=window_size,
                    )
                else:
                    values = self._read_paged_cache(
                        buffer=buffer,
                        block_indices=page_indices,
                        seq_len=attn_len,
                        window_size=window_size,
                    )
                    scales = None

                cache_values[layer, attn_type] = (values, scales)

        return cache_values

    def _assert_cache_equal(
        self, seq_len: int, compress_ratios: List[int], expect: _RequestCache, actual: _RequestCache
    ) -> None:
        """Assert that two cache dictionaries contain equal values.

        Args:
            seq_len: Sequence length
            compress_ratios: Compression ratios for each layer
            expected: Expected cache values
            actual: Actual cache values read from cache manager
        """
        # Check that keys match
        assert set(expect.keys()) == set(actual.keys()), (
            f"Cache keys don't match. Expected: {set(expect.keys())}, Actual: {set(actual.keys())}"
        )

        # Check each tensor value
        for layer_idx, attn_type in expect.keys():
            if attn_type in [
                DeepseekV4AttentionType.COMPRESS,
                DeepseekV4AttentionType.INDEXER_COMPRESS,
            ]:
                attn_len = seq_len // compress_ratios[layer_idx]
            else:
                attn_len = seq_len

            expect_values, expect_scales = expect[layer_idx, attn_type]
            actual_values, actual_scales = actual[layer_idx, attn_type]

            # Slice to attention length
            expect_values = expect_values[:attn_len]
            if expect_scales is not None:
                expect_scales = expect_scales[:attn_len]

            # Apply window size if applicable
            window_size = self._get_window_size(compress_ratios[layer_idx], attn_type)
            if window_size is not None:
                expect_values = expect_values[-window_size:]
                if expect_scales is not None:
                    expect_scales = expect_scales[-window_size:]

            # Assert values match
            torch.testing.assert_close(
                actual_values,
                expect_values,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Mismatch for layer {layer_idx}, attention type {attn_type.name} (values)",
            )

            # Assert scales match (both should be None or both should be tensors)
            if expect_scales is None:
                assert actual_scales is None, (
                    f"Expected no scales for layer {layer_idx}, attention type {attn_type.name}, "
                    f"but got scales with shape {actual_scales.shape}"
                )
            else:
                assert actual_scales is not None, (
                    f"Expected scales for layer {layer_idx}, attention type {attn_type.name}, "
                    f"but got None"
                )
                torch.testing.assert_close(
                    actual_scales,
                    expect_scales,
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"Mismatch for layer {layer_idx}, attention type {attn_type.name} (scales)",
                )

    def test_max_num_tokens_is_used_by_base_config(self):
        max_batch_size = 2
        max_seq_len = 1024
        max_input_len = 127
        max_num_tokens = max_batch_size * (max_input_len + 1)
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            max_input_len=max_input_len,
            compress_ratios=[1, 4],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
        )

        assert cache_manager.kv_cache_manager_py_config.typical_step == BatchDesc(
            [
                KVCacheDesc(capacity=max_num_tokens, history_length=0),
                KVCacheDesc(capacity=max_seq_len, history_length=max_seq_len - 1),
            ]
        )

    def test_indexer_cache_layout_default(self):
        """DeepSeek-V4 defaults to FP4 indexer K cache on Blackwell+."""
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=512,
            compress_ratios=[4],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
        )
        try:
            buffer = cache_manager.get_buffers(0, DeepseekV4AttentionType.INDEXER_COMPRESS)
            assert buffer.dtype == torch.uint8
            assert buffer.shape[-1] == 64 + 4
            assert cache_manager.quant_block_size == 32
        finally:
            cache_manager.shutdown()

    def test_indexer_cache_layout_fp8(self):
        """The legacy FP8 blockwise indexer K cache remains available."""
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=512,
            compress_ratios=[4],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            indexer_k_dtype="fp8",
        )
        try:
            buffer = cache_manager.get_buffers(0, DeepseekV4AttentionType.INDEXER_COMPRESS)
            assert buffer.dtype == torch.float8_e4m3fn
            assert buffer.shape[-1] == 128 + 4
            assert cache_manager.quant_block_size == 128
        finally:
            cache_manager.shutdown()

    @pytest.mark.parametrize("compress_ratios", [[1, 4, 128]])
    @pytest.mark.parametrize(
        "dtype,compressor_dtype", [(DataType.BF16, DataType.FLOAT), (DataType.FP8, DataType.FLOAT)]
    )
    @pytest.mark.parametrize("prompt_lens", [[512, 128, 160], [1024, 2048, 4096]])
    @pytest.mark.parametrize("num_generation_steps", [2, 100])
    def test_write_read_cache(
        self,
        compress_ratios: List[int],
        prompt_lens: List[int],
        num_generation_steps: int,
        dtype: DataType,
        compressor_dtype: DataType,
        scratch_reuse_enabled: bool,
    ):
        max_batch_size = len(prompt_lens)
        max_seq_len = max(prompt_lens) + num_generation_steps + 1
        max_input_len = max(prompt_lens)
        # Create cache manager and sparse attention config
        cache_manager, sparse_attn_config = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            compress_ratios=compress_ratios,
            dtype=dtype,
            compressor_dtype=compressor_dtype,
            max_input_len=max_input_len,
            enable_swa_scratch_reuse=scratch_reuse_enabled,
        )
        assert cache_manager.enable_swa_scratch_reuse == scratch_reuse_enabled

        # Create requests and their cache values
        requests = list[LlmRequest]()
        try:
            cache_values = dict[int, _RequestCache]()
            for req_id, prompt_len in enumerate(prompt_lens):
                req = self._create_request(req_id, prompt_len)
                requests.append(req)

                # Generate random cache values for this request
                cache_values[req_id] = self._create_random_cache(
                    seq_len=prompt_len + num_generation_steps + 1,
                    head_dim=self.head_dim,
                    sparse_attn_config=sparse_attn_config,
                    dtype=binding_to_torch_dtype(dtype),
                    compressor_dtype=binding_to_torch_dtype(compressor_dtype),
                )

            # Simulate the prefill phrase
            scheduled_batch = ScheduledRequests()
            scheduled_batch.context_requests_last_chunk = requests
            for req in requests:
                cache_manager.prepare_context(req)
                cache_manager.resize_context(req, req.context_chunk_size)

            # Write context to cache
            for req in requests:
                self._write_request_prefill(
                    req=req,
                    prompt_len=prompt_lens[req.py_request_id],
                    cache_manager=cache_manager,
                    cache_values=cache_values[req.py_request_id],
                )

            # Update requests state and call update_resources
            for req in requests:
                req.context_current_position = prompt_lens[req.py_request_id]
                req.add_new_token(prompt_lens[req.py_request_id], 0)
            cache_manager.update_context_resources(scheduled_batch)
            cache_manager.update_resources(scheduled_batch)

            # Read context from cache and verify
            for req in requests:
                actual_cache_values = self._read_request(
                    req=req,
                    seq_len=prompt_lens[req.py_request_id],
                    cache_manager=cache_manager,
                    compress_ratios=compress_ratios,
                )
                self._assert_cache_equal(
                    seq_len=prompt_lens[req.py_request_id],
                    compress_ratios=compress_ratios,
                    expect=cache_values[req.py_request_id],
                    actual=actual_cache_values,
                )

            # Simulate the decode phrase
            for i in range(num_generation_steps):
                seq_lens = [prompt_len + i + 1 for prompt_len in prompt_lens]
                scheduled_batch = ScheduledRequests()
                scheduled_batch.generation_requests = requests
                for req in requests:
                    cache_manager.try_allocate_generation(req)

                # Write new token to cache
                for req in requests:
                    self._write_request_decode(
                        req=req,
                        token_idx=seq_lens[req.py_request_id] - 1,
                        cache_manager=cache_manager,
                        cache_values=cache_values[req.py_request_id],
                    )

                # Read context from cache and verify
                for req in requests:
                    actual_cache_values = self._read_request(
                        req=req,
                        seq_len=seq_lens[req.py_request_id],
                        cache_manager=cache_manager,
                        compress_ratios=compress_ratios,
                    )
                    self._assert_cache_equal(
                        seq_len=seq_lens[req.py_request_id],
                        compress_ratios=compress_ratios,
                        expect=cache_values[req.py_request_id],
                        actual=actual_cache_values,
                    )

                for req in requests:
                    req.add_new_token(seq_lens[req.py_request_id], 0)
                cache_manager.update_resources(scheduled_batch)
        finally:
            try:
                for req in requests:
                    cache_manager.free_resources(req)
            finally:
                cache_manager.shutdown()

    @pytest.mark.parametrize("compress_ratios", [[1, 4, 128]])
    @pytest.mark.parametrize(
        "dtype,compressor_dtype", [(DataType.BF16, DataType.FLOAT), (DataType.FP8, DataType.FLOAT)]
    )
    def test_pool_mapping_tensors(
        self, compress_ratios: List[int], dtype: DataType, compressor_dtype: DataType
    ):
        # Create cache manager and sparse attention config
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=4,
            max_seq_len=1024,
            compress_ratios=compress_ratios,
            dtype=dtype,
            compressor_dtype=compressor_dtype,
        )

        try:
            expected_mapping = torch.stack(
                [
                    torch.arange(cache_manager.num_local_layers, dtype=torch.int32),
                    torch.zeros(cache_manager.num_local_layers, dtype=torch.int32),
                ],
                dim=1,
            )
            expected_pointers = torch.tensor(
                [
                    [
                        cache_manager.impl.get_mem_pool_base_address(
                            cache_manager._layer_attn_to_layer_id[
                                pp_layer, DeepseekV4AttentionType.SWA
                            ],
                            DeepseekV4AttentionType.SWA.role,
                            PageIndexMode.PER_LAYER,
                        ),
                        0,
                    ]
                    for pp_layer in cache_manager.pp_layers
                ],
                dtype=torch.int64,
            )

            assert cache_manager.kv_cache_pool_mapping.shape == expected_mapping.shape
            assert cache_manager.kv_cache_pool_mapping.dtype == expected_mapping.dtype
            assert torch.equal(cache_manager.kv_cache_pool_mapping, expected_mapping)
            assert cache_manager.kv_cache_pool_pointers.shape == expected_pointers.shape
            assert cache_manager.kv_cache_pool_pointers.dtype == expected_pointers.dtype
            assert torch.equal(cache_manager.kv_cache_pool_pointers, expected_pointers)
        finally:
            cache_manager.shutdown()

    def test_disagg_page_table_accepts_deepseek_v4_roles(self):
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=512,
            compress_ratios=[1, 4, 128],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
        )

        try:
            page_table = build_page_table_from_manager(cache_manager)
            saw_pool_view = False
            for layer_group in page_table.layer_groups:
                for pool_view in layer_group.pool_views:
                    saw_pool_view = True
                    # All DSv4 pools publish per-layer buffer_entries (INDEXED);
                    # the FLAT layout is reserved for the DSA (DeepSeek v3.2)
                    # indexer K cache pool.
                    assert pool_view.mapper_kind == MapperKind.INDEXED
                    # pool_role carries the manager-native role-name strings,
                    # which are all DSv4 attention-type names like
                    # "deepseek_v4_swa", "deepseek_v4_compress", etc.
                    assert pool_view.pool_role
                    for role_name in pool_view.pool_role:
                        assert role_name.startswith("deepseek_v4_"), role_name
            assert saw_pool_view
        finally:
            cache_manager.shutdown()

    def test_disagg_pool_mapping_prefers_exact_pool_view_layer_set(self):
        ctx_cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=512,
            compress_ratios=[1, 4, 128],
            dtype=DataType.FP8,
            compressor_dtype=DataType.FLOAT,
            tp_size=2,
        )
        gen_cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=512,
            compress_ratios=[1, 4, 128],
            dtype=DataType.FP8,
            compressor_dtype=DataType.FLOAT,
            tp_size=2,
            enable_attention_dp=True,
        )

        try:
            ctx_rank_info = RankInfo.from_kv_cache_manager("ctx", ctx_cache_manager, 0)
            gen_rank_info = RankInfo.from_kv_cache_manager("gen", gen_cache_manager, 0)
            registrar = PeerRegistrar(gen_rank_info, KVRegionExtractorV1(gen_cache_manager))
            registrar.register("ctx", 0, ctx_rank_info)

            pool_mapping = registrar.get_pool_mapping(ctx_rank_info)
            assert pool_mapping
            for self_pool_key, peer_pool_key in pool_mapping.items():
                registrar.get_kv_map(ctx_rank_info, self_pool_key, peer_pool_key)
        finally:
            ctx_cache_manager.shutdown()
            gen_cache_manager.shutdown()

    def test_dsv4_block_tables(self, scratch_reuse_enabled: bool):
        prompt_len = self.tokens_per_block * 2
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=prompt_len,
            compress_ratios=[4, 4],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=scratch_reuse_enabled,
        )
        assert cache_manager.enable_swa_scratch_reuse == scratch_reuse_enabled

        req = self._create_request(0, prompt_len)
        allocated = False
        try:
            assert cache_manager.prepare_context(req)
            assert cache_manager.resize_context(req, req.context_chunk_size)
            allocated = True

            attention_types = {
                DeepseekV4AttentionType.COMPRESS,
                DeepseekV4AttentionType.INDEXER_COMPRESS,
                DeepseekV4AttentionType.COMPRESSOR_KV,
            }
            sliding_block_tables_cuda = torch.empty(
                self._sliding_block_tables_shape(cache_manager, 1),
                dtype=torch.int32,
                device="cuda",
            )
            compress_block_table = torch.empty(
                1,
                cache_manager.max_blocks_per_seq,
                dtype=torch.int32,
                device="cpu",
            )
            host_indexer_compress_block_table = torch.empty(
                1,
                cache_manager.max_blocks_per_seq,
                dtype=torch.int32,
                device="cpu",
            )
            with torch.cuda.stream(cache_manager._stream):
                cache_manager.compute_sliding_block_tables(
                    [req.py_request_id],
                    num_contexts=1,
                )
                cache_manager.copy_batch_sliding_block_tables(
                    sliding_block_tables_cuda,
                    [req.py_request_id],
                    num_contexts=1,
                    num_seqs=1,
                )
            cache_manager.copy_batch_compress_block_tables(
                compress_block_table,
                [req.py_request_id],
                compress_ratio=4,
                beam_width=1,
                num_contexts=1,
                num_seqs=1,
            )
            cache_manager.copy_batch_indexer_compress_block_tables(
                host_indexer_compress_block_table,
                [req.py_request_id],
                beam_width=1,
                num_contexts=1,
                num_seqs=1,
            )
            cache_manager._stream.synchronize()
            sliding_block_tables = sliding_block_tables_cuda.cpu()
            for attn_type in attention_types:
                layer0_buffer = _view_fp8_as_uint8(cache_manager.get_buffers(0, attn_type))
                layer1_buffer = _view_fp8_as_uint8(cache_manager.get_buffers(1, attn_type))
                attn_len = prompt_len
                if attn_type in [
                    DeepseekV4AttentionType.COMPRESS,
                    DeepseekV4AttentionType.INDEXER_COMPRESS,
                ]:
                    attn_len //= 4
                num_blocks = (attn_len + layer0_buffer.shape[1] - 1) // layer0_buffer.shape[1]
                if attn_type == DeepseekV4AttentionType.COMPRESS:
                    layer0_offsets = compress_block_table[0, :num_blocks].tolist()
                    layer1_offsets = compress_block_table[0, :num_blocks].tolist()
                elif attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
                    layer0_offsets = host_indexer_compress_block_table[0, :num_blocks].tolist()
                    layer1_offsets = host_indexer_compress_block_table[0, :num_blocks].tolist()
                else:
                    layer0_offsets = sliding_block_tables[
                        cache_manager.layer_offsets[0],
                        attn_type.value,
                        0,
                        :num_blocks,
                    ].tolist()
                    layer1_offsets = sliding_block_tables[
                        cache_manager.layer_offsets[1],
                        attn_type.value,
                        0,
                        :num_blocks,
                    ].tolist()
                assert all(offset != BAD_PAGE_INDEX for offset in layer0_offsets)
                assert all(offset != BAD_PAGE_INDEX for offset in layer1_offsets)

                layer0_values = torch.full(
                    (num_blocks, layer0_buffer.shape[1], layer0_buffer.shape[-1]),
                    1,
                    dtype=layer0_buffer.dtype,
                    device=layer0_buffer.device,
                )
                layer1_values = torch.full_like(layer0_values, 2)

                layer0_buffer[layer0_offsets] = layer0_values
                layer1_buffer[layer1_offsets] = layer1_values

                torch.testing.assert_close(layer0_buffer[layer0_offsets], layer0_values)
                torch.testing.assert_close(layer1_buffer[layer1_offsets], layer1_values)
        finally:
            if allocated:
                cache_manager.free_resources(req)
            cache_manager.shutdown()

    def test_copy_batch_block_offsets_matches_python_converter(self, scratch_reuse_enabled: bool):
        prompt_len = self.tokens_per_block * 2 + 1
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=3,
            max_seq_len=1024,
            compress_ratios=[1, 4, 128, 4],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=scratch_reuse_enabled,
        )
        assert cache_manager.enable_swa_scratch_reuse == scratch_reuse_enabled

        requests = []
        try:
            requests, num_contexts = self._prepare_mixed_copy_batch(cache_manager, prompt_len)
            request_ids = [req.py_request_id for req in requests]
            actual = torch.empty(
                self._attention_op_block_offsets_shape(cache_manager, len(request_ids)),
                dtype=torch.int32,
                device="cuda",
            )
            sliding_block_tables = torch.empty(
                self._sliding_block_tables_shape(cache_manager, len(request_ids)),
                dtype=torch.int32,
                device="cuda",
            )

            with torch.cuda.stream(cache_manager._stream):
                cache_manager.compute_sliding_block_tables(
                    request_ids,
                    num_contexts=num_contexts,
                )
                cache_manager.copy_batch_block_offsets(
                    actual,
                    request_ids,
                    beam_width=1,
                    num_contexts=num_contexts,
                    num_seqs=len(request_ids),
                )
                cache_manager.copy_batch_sliding_block_tables(
                    sliding_block_tables,
                    request_ids,
                    num_contexts=num_contexts,
                    num_seqs=len(request_ids),
                )

            expected = torch.full(
                (
                    cache_manager.num_attention_op_pools,
                    len(request_ids),
                    cache_manager.max_blocks_per_seq,
                ),
                BAD_PAGE_INDEX,
                dtype=torch.int32,
                device="cpu",
            )
            for local_layer_idx, pp_layer in enumerate(cache_manager.pp_layers):
                ref = self._reference_copy_batch_page_indices(
                    cache_manager,
                    request_ids,
                    num_contexts,
                    pp_layer,
                    DeepseekV4AttentionType.SWA,
                    PageIndexMode.PER_LAYER,
                )
                expected[local_layer_idx, : len(request_ids)] = ref

            # DSV4 AttentionOp only consumes the key table.
            cache_manager._stream.synchronize()
            actual_cpu = actual.cpu()
            sliding_block_tables_cpu = sliding_block_tables.cpu()
            torch.testing.assert_close(
                actual_cpu[:, : len(request_ids), 0],
                expected[:, : len(request_ids)],
            )
            torch.testing.assert_close(
                actual_cpu[:, : len(request_ids), 0],
                sliding_block_tables_cpu[
                    :,
                    DeepseekV4AttentionType.SWA.value,
                    : len(request_ids),
                ],
            )
        finally:
            for req in requests:
                cache_manager.free_resources(req)
            cache_manager.shutdown()

    def test_copy_batch_sliding_block_tables_matches_python_converter(
        self, scratch_reuse_enabled: bool
    ):
        prompt_len = self.tokens_per_block * 2 + 1
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=3,
            max_seq_len=1024,
            compress_ratios=[1, 4, 128, 4],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=scratch_reuse_enabled,
        )
        assert cache_manager.enable_swa_scratch_reuse == scratch_reuse_enabled

        requests = []
        try:
            requests, num_contexts = self._prepare_mixed_copy_batch(cache_manager, prompt_len)
            request_ids = [req.py_request_id for req in requests]
            actual = torch.empty(
                self._sliding_block_tables_shape(cache_manager, len(request_ids)),
                dtype=torch.int32,
                device="cuda",
            )

            with torch.cuda.stream(cache_manager._stream):
                cache_manager.compute_sliding_block_tables(
                    request_ids,
                    num_contexts=num_contexts,
                )
                cache_manager.copy_batch_sliding_block_tables(
                    actual,
                    request_ids,
                    num_contexts=num_contexts,
                    num_seqs=len(request_ids),
                )

            expected = torch.full(
                self._sliding_block_tables_shape(cache_manager, len(request_ids)),
                BAD_PAGE_INDEX,
                dtype=torch.int32,
                device="cpu",
            )
            for pp_layer in cache_manager.pp_layers:
                compress_ratio = cache_manager._compress_ratios[pp_layer]
                local_layer_idx = cache_manager.layer_offsets[pp_layer]
                for attn_type in DEEPSEEK_V4_SLIDING_ATTENTION:
                    if not compress_ratio_has_attention(compress_ratio, attn_type):
                        continue
                    expected[local_layer_idx, attn_type.value, : len(request_ids)] = (
                        self._reference_copy_batch_page_indices(
                            cache_manager,
                            request_ids,
                            num_contexts,
                            pp_layer,
                            attn_type,
                            PageIndexMode.PER_LAYER,
                        )
                    )

            cache_manager._stream.synchronize()
            torch.testing.assert_close(
                actual.cpu()[:, :, : len(request_ids)],
                expected[:, :, : len(request_ids)],
            )
        finally:
            for req in requests:
                cache_manager.free_resources(req)
            cache_manager.shutdown()

    def test_copy_batch_compress_block_tables_matches_python_converter(
        self, scratch_reuse_enabled: bool
    ):
        prompt_len = self.tokens_per_block * 2 + 1
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=3,
            max_seq_len=1024,
            compress_ratios=[1, 4, 128, 4],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=scratch_reuse_enabled,
        )
        assert cache_manager.enable_swa_scratch_reuse == scratch_reuse_enabled

        requests = []
        try:
            requests, num_contexts = self._prepare_mixed_copy_batch(cache_manager, prompt_len)
            request_ids = [req.py_request_id for req in requests]

            for compress_ratio in (4, 128):
                actual = torch.empty(
                    len(request_ids),
                    cache_manager.max_blocks_per_seq,
                    dtype=torch.int32,
                    device="cpu",
                )

                cache_manager.copy_batch_compress_block_tables(
                    actual,
                    request_ids,
                    compress_ratio=compress_ratio,
                    beam_width=1,
                    num_contexts=num_contexts,
                    num_seqs=len(request_ids),
                )

                pp_layer = next(
                    layer
                    for layer in cache_manager.pp_layers
                    if cache_manager._compress_ratios[layer] == compress_ratio
                )
                expected = self._reference_copy_batch_page_indices(
                    cache_manager,
                    request_ids,
                    num_contexts,
                    pp_layer,
                    DeepseekV4AttentionType.COMPRESS,
                    PageIndexMode.SHARED,
                )
                torch.testing.assert_close(actual, expected)
        finally:
            for req in requests:
                cache_manager.free_resources(req)
            cache_manager.shutdown()

    def test_copy_batch_indexer_compress_block_tables_matches_python_converter(
        self, scratch_reuse_enabled: bool
    ):
        prompt_len = self.tokens_per_block * 2 + 1
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=3,
            max_seq_len=1024,
            compress_ratios=[1, 4, 128, 4],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=scratch_reuse_enabled,
        )
        assert cache_manager.enable_swa_scratch_reuse == scratch_reuse_enabled

        requests = []
        try:
            requests, num_contexts = self._prepare_mixed_copy_batch(cache_manager, prompt_len)
            request_ids = [req.py_request_id for req in requests]
            actual = torch.empty(
                len(request_ids),
                cache_manager.max_blocks_per_seq,
                dtype=torch.int32,
                device="cpu",
            )

            cache_manager.copy_batch_indexer_compress_block_tables(
                actual,
                request_ids,
                beam_width=1,
                num_contexts=num_contexts,
                num_seqs=len(request_ids),
            )

            pp_layer = next(
                layer
                for layer in cache_manager.pp_layers
                if compress_ratio_has_attention(
                    cache_manager._compress_ratios[layer],
                    DeepseekV4AttentionType.INDEXER_COMPRESS,
                )
            )
            expected = self._reference_copy_batch_page_indices(
                cache_manager,
                request_ids,
                num_contexts,
                pp_layer,
                DeepseekV4AttentionType.INDEXER_COMPRESS,
                PageIndexMode.SHARED,
            )
            torch.testing.assert_close(actual, expected)
            assert num_contexts == 2
        finally:
            for req in requests:
                cache_manager.free_resources(req)
            cache_manager.shutdown()

    def test_swa_scratch_reuse_enabled_by_default_for_main_manager(self):
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=1024,
            compress_ratios=[1],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
        )

        try:
            assert cache_manager.enable_swa_scratch_reuse
            assert cache_manager.kv_cache_manager_py_config.enable_swa_scratch_reuse
            assert cache_manager.kv_cache_manager_py_config.swa_scratch_reuse is not None
            assert cache_manager.kv_cache_manager_py_config.swa_scratch_reuse.max_rewind_len == 0
            assert cache_manager.num_attention_op_pools == cache_manager.num_local_layers
        finally:
            cache_manager.shutdown()

    def test_swa_scratch_reuse_disabled_by_config_for_main_manager(self):
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=1024,
            compress_ratios=[1],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=False,
        )

        try:
            assert not cache_manager.enable_swa_scratch_reuse
            assert not cache_manager.kv_cache_manager_py_config.enable_swa_scratch_reuse
            assert cache_manager.kv_cache_manager_py_config.swa_scratch_reuse is None
            assert cache_manager.num_attention_op_pools == cache_manager.num_local_layers
        finally:
            cache_manager.shutdown()

    def test_swa_scratch_reuse_uses_extra_kv_tokens_for_rewind(self):
        spec_config = DraftTargetDecodingConfig(max_draft_len=7, speculative_model="draft")
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=1024,
            compress_ratios=[1],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            spec_config=spec_config,
            enable_swa_scratch_reuse=True,
        )

        try:
            scratch_reuse = cache_manager.kv_cache_manager_py_config.swa_scratch_reuse
            assert scratch_reuse is not None
            assert scratch_reuse.max_rewind_len == spec_config.max_draft_len - 1
        finally:
            cache_manager.shutdown()

    def test_draft_cache_manager_disables_swa_scratch_reuse(self):
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=1024,
            compress_ratios=[1],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            is_draft=True,
            enable_swa_scratch_reuse=True,
        )

        try:
            assert not cache_manager.enable_swa_scratch_reuse
            assert not cache_manager.kv_cache_manager_py_config.enable_swa_scratch_reuse
            assert cache_manager.kv_cache_manager_py_config.swa_scratch_reuse is None
            assert cache_manager.num_attention_op_pools == cache_manager.num_local_layers
        finally:
            cache_manager.shutdown()

    def test_context_request_enable_scratch_reuse_until_generation(self):
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=1024,
            compress_ratios=[1],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=True,
        )

        req = self._create_request(request_id=0, prompt_len=self.tokens_per_block + 1)
        allocated = False
        try:
            assert cache_manager.prepare_context(req)
            assert cache_manager.resize_context(req, req.context_chunk_size)
            allocated = True

            kv_cache = cache_manager.kv_cache_map[req.py_request_id]
            assert kv_cache.enable_swa_scratch_reuse

            scheduled_batch = ScheduledRequests()
            scheduled_batch.context_requests_last_chunk = [req]
            req.context_current_position = req.prompt_len
            req.add_new_token(req.prompt_len, 0)
            cache_manager.update_context_resources(scheduled_batch)
            assert not kv_cache.enable_swa_scratch_reuse

            assert cache_manager.try_allocate_generation(req)
            assert not kv_cache.enable_swa_scratch_reuse
        finally:
            if allocated:
                cache_manager.free_resources(req)
            cache_manager.shutdown()

    def test_disagg_generation_init_disables_swa_scratch_reuse(self):
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=1024,
            compress_ratios=[1],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=True,
        )

        req = self._create_request(request_id=0, prompt_len=self.tokens_per_block + 1)
        req.state = LlmRequestState.DISAGG_GENERATION_INIT
        allocated = False
        try:
            assert cache_manager.prepare_disagg_gen_init(req)
            allocated = True

            kv_cache = cache_manager.kv_cache_map[req.py_request_id]
            assert not kv_cache.enable_swa_scratch_reuse
            assert kv_cache.history_length == req.prompt_len
        finally:
            if allocated:
                cache_manager.free_resources(req)
            cache_manager.shutdown()

    def test_disagg_generation_init_rejects_context_apis(self):
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=1024,
            compress_ratios=[1],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=True,
        )
        req = self._create_request(request_id=0, prompt_len=self.tokens_per_block + 1)
        req.state = LlmRequestState.DISAGG_GENERATION_INIT
        try:
            with pytest.raises(AssertionError, match="prepare_disagg_gen_init"):
                cache_manager.prepare_context(req)
            with pytest.raises(AssertionError, match="prepare_disagg_gen_init"):
                cache_manager.resize_context(req, req.context_remaining_length)
        finally:
            cache_manager.shutdown()

    @pytest.mark.parametrize("compress_ratios", [[1], [1, 4, 128]])
    @pytest.mark.parametrize("long_token_num", [127, 128, 129, 513])
    def test_dummy_generation_requests_with_swa_scratch_reuse(
        self, compress_ratios: list[int], long_token_num: int
    ) -> None:
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=2,
            max_seq_len=1024,
            compress_ratios=compress_ratios,
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=True,
        )

        requests = []
        token_nums = [1, long_token_num]
        try:
            requests = cache_manager.add_dummy_requests(
                request_ids=[0, 1],
                token_nums=token_nums,
                is_gen=True,
            )
            assert requests is not None
            assert len(requests) == len(token_nums)

            short_kv_cache = cache_manager.kv_cache_map[requests[0].py_request_id]
            long_kv_cache = cache_manager.kv_cache_map[requests[1].py_request_id]
            assert not short_kv_cache.enable_swa_scratch_reuse
            assert not long_kv_cache.enable_swa_scratch_reuse
            assert short_kv_cache.history_length == 0
            assert short_kv_cache.capacity == token_nums[0]
            assert long_kv_cache.history_length == token_nums[1] - 1
            assert long_kv_cache.capacity == token_nums[1]
        finally:
            for req in requests:
                cache_manager.free_resources(req)
            cache_manager.shutdown()

    @pytest.mark.parametrize("compress_ratios", [[1, 4, 128]])
    @pytest.mark.parametrize(
        "dtype,compressor_dtype", [(DataType.BF16, DataType.FLOAT), (DataType.FP8, DataType.FLOAT)]
    )
    @pytest.mark.parametrize("invalid", [False, True])
    @pytest.mark.parametrize("fill_with_zero", [False, True])
    def test_check_invalid_values_in_kv_cache(
        self,
        compress_ratios: List[int],
        dtype: DataType,
        compressor_dtype: DataType,
        invalid: bool,
        fill_with_zero: bool,
    ):
        """Test invalid value detection and optional zero-fill behavior in KV cache."""
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=4,
            max_seq_len=1024,
            compress_ratios=compress_ratios,
            dtype=dtype,
            compressor_dtype=compressor_dtype,
        )

        needs_invalid_cleanup = False
        try:
            # Fresh cache (zero-initialized) should have no invalid values
            result = cache_manager.check_invalid_values_in_kv_cache()
            assert not result, "Fresh cache should have no invalid values"

            if invalid:
                # Inject invalid into a float buffer so NaN/Inf checks are supported.
                layer_idx = next(i for i, ratio in enumerate(compress_ratios) if ratio > 1)
                buffer = cache_manager.get_buffers(layer_idx, DeepseekV4AttentionType.COMPRESSOR_KV)
                buffer[0, 0, 0] = torch.nan
                needs_invalid_cleanup = True

            result = cache_manager.check_invalid_values_in_kv_cache(fill_with_zero=fill_with_zero)
            if invalid and fill_with_zero:
                needs_invalid_cleanup = False
            assert result == invalid, (
                f"Expected invalid={invalid} from check_invalid_values_in_kv_cache, got {result}"
            )

            # Verify whether invalid values remain after the check.
            post_check = cache_manager.check_invalid_values_in_kv_cache()
            expected_post_check = invalid and not fill_with_zero
            assert post_check == expected_post_check, (
                f"Expected post-check invalid={expected_post_check}, got {post_check}"
            )

            if expected_post_check:
                # Cleanup for shutdown path when zero-fill wasn't requested above.
                cache_manager.check_invalid_values_in_kv_cache(fill_with_zero=True)
                needs_invalid_cleanup = False
        finally:
            try:
                if needs_invalid_cleanup:
                    cache_manager.check_invalid_values_in_kv_cache(fill_with_zero=True)
            finally:
                cache_manager.shutdown()


# Offload memory accounting.

V4_FLASH_COMPRESS_RATIOS = [1, 1] + [4, 128] * 20 + [4]


def _v4_flash_offload_manager(
    *, max_batch_size: int = 256, cache_factor: float = 2.0
) -> DeepseekV4CacheManager:
    manager, _ = _cache_config_case(
        enabled=True,
        pp_layers=list(range(len(V4_FLASH_COMPRESS_RATIOS))),
        dtype=DataType.NVFP4,
        nvfp4_residual_dim=64,
        indexer_k_dtype="fp4",
        cache_factor=cache_factor,
    )
    manager._compress_ratios = V4_FLASH_COMPRESS_RATIOS
    manager.compressed_block_sizes = [
        manager.tokens_per_block // r for r in V4_FLASH_COMPRESS_RATIOS
    ]
    manager.max_batch_size = max_batch_size
    manager._max_num_tokens = 8192
    manager.enable_swa_scratch_reuse = False
    manager._generation_kv_capacity_headroom = 1
    return manager


def _v4_flash_offload_config(manager: DeepseekV4CacheManager) -> KVCacheManagerConfig:
    """Return the manager's cache config."""
    return manager._build_cache_config(
        KVCacheManagerConfig(
            tokens_per_block=manager.tokens_per_block,
            cache_tiers=[GpuCacheTierConfig(quota=1 << 30)],
            layers=[],
            constraints=[BatchDesc([KVCacheDesc(capacity=0, history_length=0)] * 513)],
        )
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(("cache_factor", "lease_bytes"), [(1.0, 3_793_024), (2.0, 7_362_688)])
def test_offload_splits_v4_flash_nvfp4_history_by_memory_kind(
    cache_factor: float, lease_bytes: int
) -> None:
    manager = _v4_flash_offload_manager(cache_factor=cache_factor)
    config = _v4_flash_offload_config(manager)

    # 21 CSA layers keep 81 B per token in host memory: 288 B of NVFP4 data and
    # 36 B of scales per row of four tokens.
    assert sparse_history_bytes_per_block(config.layers) == 128 * 21 * 81
    # The GPU keeps the MXFP4 indexer keys (17 B per token) and the HCA history
    # (324 B per 128 tokens): 407.625 B per token, reported in whole bytes.
    assert manager.get_cache_bytes_per_token() == 408
    (arena,) = sparse_arena_shapes(config.layers, tokens_per_block=manager.tokens_per_block)
    assert arena.lease_bytes == lease_bytes
    assert sparse_arena_bytes(config, 513) == arena.arena_bytes(513)

    manager._enable_kv_cache_offload = False
    config = _v4_flash_offload_config(manager)
    assert manager.get_cache_bytes_per_token() == 2098
    assert sparse_history_bytes_per_block(config.layers) == 0
    assert sparse_arena_bytes(config, 513) == 0


@pytest.mark.cpu_only
def test_offload_scratch_counts_both_attention_metadata_copies() -> None:
    manager = _v4_flash_offload_manager()
    config = _v4_flash_offload_config(manager)

    # A copy holds 64 MiB of FP8 rows dequantized from NVFP4, [256, 512, 512], with a
    # zeroed tail of one FMHA KV tile (128 rows of 512), and int32 indices: the
    # selections [256, 512], the fetch outputs [256, 2, 512] and write destinations
    # [256, 2] of the data and scale buffers, the scatter start positions [256], the
    # Batch row map [256] and the active count.
    per_copy = (256 * 512 + 128) * 512 + 4 * (256 * 512 + 256 * 2 * 512 + 256 * 2 + 256 + 256 + 1)
    assert per_copy == 68_751_364
    # The eager attention metadata holds one, and the CUDA graph metadata share one.
    assert manager.get_sparse_offload_scratch_bytes(config) == 2 * per_copy

    manager._enable_kv_cache_offload = False
    assert manager.get_sparse_offload_scratch_bytes(_v4_flash_offload_config(manager)) == 0


@pytest.mark.cpu_only
def test_offload_counts_gpu_history_and_window_pages_exactly() -> None:
    manager = _v4_flash_offload_manager()
    max_tokens = 1 << 20

    # Every window can span two pages: 43 SWA layers, 21 CSA and 20 HCA
    # compressor states.
    assert manager._get_cache_bytes_for_tokens(0, context=False) == 81_657_856
    # 52,176 GPU bytes per 128 tokens, with no row rounded down.
    assert manager._get_cache_bytes_for_tokens(max_tokens, context=False) == (
        max_tokens * 52_176 // 128 + 81_657_856
    )
    assert manager.get_request_device_bytes(max_tokens) == (
        manager._get_cache_bytes_for_tokens(max_tokens, context=False)
    )
    quota = manager._get_quota_from_max_tokens(max_tokens)
    assert manager._get_max_tokens_from_quota(quota) == pytest.approx(max_tokens)

    # Without offload the estimates are unchanged: whole bytes per token and one
    # page per window.
    manager._enable_kv_cache_offload = False
    assert manager.get_cache_bytes_per_token() == 2098
    assert manager._get_cache_bytes_for_tokens(0, context=False) == 40_828_928
    assert manager._get_cache_bytes_for_tokens(max_tokens, context=False) == (
        max_tokens * 2098 + 40_828_928
    )


@pytest.mark.cpu_only
def test_sparse_offload_rejects_kv_pool_rebalance() -> None:
    sparse_config = DeepSeekV4SparseAttentionConfig(enable_kv_cache_offload=True)
    with (
        patch.object(deepseek_v4_cache.KVCacheManagerV2, "__init__") as initialize_cache,
        pytest.raises(ValueError, match="enable_kv_pool_rebalance"),
    ):
        DeepseekV4CacheManager(
            kv_cache_config=KvCacheConfig(enable_block_reuse=False, enable_kv_pool_rebalance=True),
            kv_cache_type=CacheTypeCpp.SELFKONLY,
            num_layers=len(sparse_config.compress_ratios),
            max_batch_size=1,
            tokens_per_block=128,
            max_seq_len=256,
            vocab_size=128,
            mapping=Mapping(),
            sparse_attn_config=sparse_config,
        )

    initialize_cache.assert_not_called()


@pytest.mark.cpu_only
def test_sparse_offload_rejects_speculative_decoding() -> None:
    sparse_config = DeepSeekV4SparseAttentionConfig(enable_kv_cache_offload=True)
    with (
        patch.object(deepseek_v4_cache.KVCacheManagerV2, "__init__") as initialize_cache,
        pytest.raises(ValueError, match="does not support speculative decoding"),
    ):
        DeepseekV4CacheManager(
            kv_cache_config=KvCacheConfig(enable_block_reuse=False),
            kv_cache_type=CacheTypeCpp.SELFKONLY,
            num_layers=len(sparse_config.compress_ratios),
            max_batch_size=1,
            tokens_per_block=128,
            max_seq_len=256,
            vocab_size=128,
            mapping=Mapping(),
            sparse_attn_config=sparse_config,
            spec_config=MTPDecodingConfig(max_draft_len=1),
        )

    initialize_cache.assert_not_called()


@pytest.mark.cpu_only
def test_offload_capacity_report_names_the_binding_limit() -> None:
    manager = _v4_flash_offload_manager()
    manager.max_seq_len = 1 << 17
    manager.mapping = Mapping()
    manager.kv_cache_manager_py_config = _v4_flash_offload_config(manager)
    # Half of a 100 GiB share of host memory, the automatic budget on a node of
    # four ranks with 400 GiB available.
    manager._host_history_budget = 50 << 30
    # The native group holds 4,096 slots: four requests of 128K tokens. Its pool of
    # sparse pages holds 217,728 host bytes per slot; its indexer pool is on the GPU.
    sparse_layer = next(
        layer for layer in manager.kv_cache_manager_py_config.layers if layer.sparse_cache
    )
    sparse_buffer = next(buffer for buffer in sparse_layer.buffers if buffer.is_sparse)
    dense_buffer = next(buffer for buffer in sparse_layer.buffers if not buffer.is_sparse)
    host_pool = SimpleNamespace(
        num_slots=4096,
        slot_desc=SimpleNamespace(
            variants=[
                SimpleNamespace(
                    coalesced_buffers=[
                        SimpleNamespace(
                            buffer_ids=[
                                SimpleNamespace(layer_id=sparse_layer.layer_id, role=buffer.role)
                            ]
                        )
                        for buffer in (sparse_buffer, dense_buffer)
                    ]
                )
            ]
        ),
        pools=[
            SimpleNamespace(pool_index=0, slot_bytes=217_728),
            SimpleNamespace(pool_index=1, slot_bytes=52_176),
        ],
    )
    manager.get_host_backed_pool_groups = lambda: [host_pool]
    # The device quota counts only device-backed pools.
    manager.impl = SimpleNamespace(get_quota=lambda level: 100 << 30)

    with (
        patch.object(kv_cache_v2_module.logger, "info") as info,
        patch.object(kv_cache_v2_module.logger, "warning") as warning,
    ):
        manager._log_sparse_kv_offload_capacity(KvCacheConfig())

    report = info.call_args.args[0]
    # The DeepSeek-V4 estimator sizes each 128K-token request at 128.8 MiB of GPU pools.
    assert "128.8MiB per request" in report
    assert "4 by host" in report
    assert "limited by host" in report
    # One lease per Batch row, not per constrained request: max_batch_size, not 513.
    assert "256 leases" in report
    # The 50 GiB host budget holds 240 requests of 128K tokens.
    assert "A max_batch_size of 240 fits" in report
    assert "Only 4 of max_batch_size=256" in warning.call_args.args[0]


if __name__ == "__main__":
    tester = TestDeepseekV4CacheManager()
    print("=== FP8, prompt_lens=[1024, 2048, 4096], steps=100 ===")
    tester.test_write_read_cache([1, 4, 128], [1024, 2048, 4096], 100, DataType.FP8, DataType.FLOAT)
    print("Test passed")
