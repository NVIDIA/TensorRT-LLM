# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-backed main K/V blocks for MiniMax-M3's MSA decode path."""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.kv_cache.sparse_offload_memory import sparse_cache_config
from tensorrt_llm._utils import TensorWrapper, binding_to_torch_dtype, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferId,
    KVCacheManagerConfig,
)

from ..offload import (
    SPARSE_OFFLOAD_STATE_COPIES,
    SparseOffloadCacheMixin,
    SparseOffloadLayerDescriptor,
)

__all__ = ["MiniMaxM3SparseOffloadMixin"]

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig


class MiniMaxM3SparseOffloadMixin(SparseOffloadCacheMixin):
    """Keep dense layers and index keys resident while offloading sparse K/V blocks."""

    def _init_m3_sparse_offload(self, config: MiniMaxM3SparseAttentionConfig | None) -> None:
        sizing = None if config is None else config.get_kv_cache_offload_config()
        self._init_sparse_offload(enabled=sizing is not None, topk=None)
        self._m3_topk_blocks = 0 if config is None else config.sparse_topk_blocks
        self._m3_offload_descriptors: dict[int, SparseOffloadLayerDescriptor] | None = None
        self._m3_arena_views: dict[tuple[int, bool], torch.Tensor] = {}

    def offloads_layer(self, layer_idx: int) -> bool:
        return self._enable_kv_cache_offload and layer_idx in self.sparse_layer_ids

    def _build_cache_config(self, config: KVCacheManagerConfig) -> KVCacheManagerConfig:
        config = super()._build_cache_config(config)
        if not self._enable_kv_cache_offload:
            return config
        if self.tokens_per_block != 128:
            raise ValueError("MiniMax-M3 KV cache offload requires 128-token pages")
        if self.dtype not in (DataType.FP8, DataType.NVFP4, DataType.BF16, DataType.HALF):
            raise ValueError(f"Unsupported MiniMax-M3 offload KV dtype: {self.dtype}")
        heads = {
            self.num_kv_heads_per_layer[self.layer_offsets[layer]]
            for layer in self.sparse_layer_ids
            if layer in self.layer_offsets
        }
        if len(heads) != 1:
            raise ValueError("MiniMax-M3 offload requires uniform sparse-layer KV head counts")
        self._sparse_offload_topk = self._m3_topk_blocks * heads.pop()
        layers = []
        roles = {Role.KEY, Role.VALUE, Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE}
        for layer in config.layers:
            if not isinstance(layer, AttentionLayerConfig) or not self.offloads_layer(
                self.pp_layers[int(layer.layer_id)]
            ):
                layers.append(layer)
                continue
            sparse_layer = copy(layer)
            buffers = []
            for buffer in layer.buffers:
                sparse_buffer = copy(buffer)
                if buffer.role in roles:
                    sparse_buffer.is_sparse = True
                buffers.append(sparse_buffer)
            sparse_layer.buffers = buffers
            sparse_layer.sparse_cache = sparse_cache_config(
                num_selected_units=self._sparse_offload_topk,
                unit_tokens=self.tokens_per_block,
                tokens_per_block=self.tokens_per_block,
                gpu_cache_factor=self._sparse_kv_offload.gpu_cache_factor,
                include_partial_units=True,
            )
            layers.append(sparse_layer)
        return replace(config, layers=layers)

    def get_sparse_offload_descriptors(self) -> dict[int, SparseOffloadLayerDescriptor]:
        if self._m3_offload_descriptors is None:
            layout = self._sparse_arena_layout()
            self._m3_offload_descriptors = {
                layer: replace(
                    self._build_sparse_offload_descriptor(
                        layout,
                        self.layer_offsets[layer],
                        Role.KEY,
                        unit_tokens=self.tokens_per_block,
                        scale_role=Role.KEY_BLOCK_SCALE if self.is_nvfp4_layer(layer) else None,
                        name=f"MiniMax-M3 layer {layer}",
                    ),
                    padding_regions=(),
                )
                for layer in self.pp_layers
                if self.offloads_layer(layer)
            }
        return self._m3_offload_descriptors

    def _m3_sparse_arena_view(self, layer_idx: int, *, scales: bool = False) -> torch.Tensor:
        """A strided [arena pages, K/V, heads, 128, dim] view addressed by K's indices.

        Each lease stores all K cells followed by all V cells. Their displacement
        is constant across leases, so both planes can use K's returned page index.
        NVFP4 scale arenas have the same buffer positions as their data arenas.
        """
        key = (layer_idx, scales)
        if key in self._m3_arena_views:
            return self._m3_arena_views[key]
        descriptor = self.get_sparse_offload_descriptors()[layer_idx]
        layout = self._sparse_arena_layout()
        local = self.layer_offsets[layer_idx]
        k_role, v_role = (
            (Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE) if scales else (Role.KEY, Role.VALUE)
        )
        k_id, v_id = BufferId(local, k_role), BufferId(local, v_role)
        k_index, v_index = descriptor.buffer_ids.index(k_id), descriptor.buffer_ids.index(v_id)
        if layout.pool_indices[k_id] != layout.pool_indices[v_id]:
            raise ValueError("MiniMax-M3 sparse K/V must share one arena pool")
        if scales and layout.positions[v_id] != layout.positions[BufferId(local, Role.VALUE)]:
            raise ValueError("MiniMax-M3 value data and scales must have matching arena indices")
        units = layout.max_units_by_group[descriptor.group_id]
        offset = (layout.positions[v_id][1] - layout.positions[k_id][1]) * units
        if (
            offset <= 0
            or descriptor.arena_base_addresses[k_index] != descriptor.arena_base_addresses[v_index]
        ):
            raise ValueError("MiniMax-M3 sparse value cells must follow key cells")
        heads = self.num_kv_heads_per_layer[local]
        dim = self.head_dim_per_layer[local]
        if scales:
            dim, dtype = dim // 16, torch.uint8
        elif self.is_nvfp4_layer(layer_idx):
            dim, dtype = dim // 2, torch.int8
        else:
            dtype = binding_to_torch_dtype(self.dtype)
        page = self.tokens_per_block
        elements = heads * page * dim
        shape = (descriptor.arena_row_counts[k_index] - offset, 2, heads, page, dim)
        strides = (elements, offset * elements, page * dim, dim, 1)
        view = convert_to_torch_tensor(
            TensorWrapper(descriptor.arena_base_addresses[k_index], dtype, shape, strides=strides)
        )
        self._m3_arena_views[key] = view
        return view

    def get_sparse_offload_scratch_bytes(self, config: KVCacheManagerConfig) -> int:
        shared = super().get_sparse_offload_scratch_bytes(config)
        if not shared:
            return 0
        # Deduplicated per-head selections, write slots, request IDs, unit lengths,
        # and the logical-block to arena-page table, for eager and graph metadata.
        entries = self.max_batch_size * (
            self._sparse_offload_topk + 3 + self._get_max_blocks_per_seq(self.max_seq_len)
        )
        return shared + SPARSE_OFFLOAD_STATE_COPIES * entries * torch.int32.itemsize
