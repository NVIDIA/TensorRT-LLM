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
"""KV cache management for MiniMax-M3 sparse attention.

Provides:
  * :class:`MiniMaxM3SparseIndexCache` — plain-tensor side cache used by
    algorithm-only unit tests (no pyexecutor dependency required for
    construction).
  * :class:`MiniMaxM3KVCacheManagerV2` — :class:`KVCacheManagerV2`
    subclass that registers a per-sparse-layer ``Role.INDEX_KEY`` paged
    buffer alongside the standard K/V buffers.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._utils import TensorWrapper, binding_to_torch_dtype, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig
from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX
from tensorrt_llm.runtime.kv_cache_manager_v2._config import DataRole

from ....pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2, Role


class MiniMaxM3SparseIndexCache:
    """Plain-tensor side cache for the M3 sparse index branch.

    Slot layout matches the main ``KVCacheManagerV2`` paged buffer
    geometry so the same ``req_to_token`` mapping addresses both
    caches. One ``[num_slots, 1, sparse_index_dim]`` index-K buffer is
    allocated per sparse layer; index-V is allocated only for layers
    not listed in ``disable_index_value_layer_ids``.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        sparse_layer_ids: List[int],
        disable_index_value_layer_ids: List[int],
        num_slots: int,
        sparse_index_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if num_slots <= 0:
            raise ValueError(f"num_slots must be > 0, got {num_slots}")
        if sparse_index_dim <= 0:
            raise ValueError(f"sparse_index_dim must be > 0, got {sparse_index_dim}")
        self.num_layers = int(num_layers)
        self.sparse_layer_ids = sorted(int(i) for i in sparse_layer_ids)
        self.disable_index_value_layer_ids = set(int(i) for i in disable_index_value_layer_ids)
        self.num_slots = int(num_slots)
        self.sparse_index_dim = int(sparse_index_dim)
        self.dtype = dtype
        self.device = device

        self._index_k: dict[int, torch.Tensor] = {}
        self._index_v: dict[int, torch.Tensor] = {}
        for layer_idx in self.sparse_layer_ids:
            if not (0 <= layer_idx < self.num_layers):
                raise ValueError(f"sparse layer_idx {layer_idx} outside [0, {self.num_layers})")
            self._index_k[layer_idx] = torch.zeros(
                (self.num_slots, 1, self.sparse_index_dim),
                dtype=dtype,
                device=device,
            )
            if layer_idx not in self.disable_index_value_layer_ids:
                self._index_v[layer_idx] = torch.zeros(
                    (self.num_slots, 1, self.sparse_index_dim),
                    dtype=dtype,
                    device=device,
                )

    def has_index_value(self, layer_idx: int) -> bool:
        return layer_idx in self._index_v

    def get_index_k_buffer(self, layer_idx: int) -> torch.Tensor:
        if layer_idx not in self._index_k:
            raise KeyError(
                f"layer_idx {layer_idx} is not a sparse layer; "
                f"sparse layers: {self.sparse_layer_ids}"
            )
        return self._index_k[layer_idx]

    def get_index_v_buffer(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._index_v.get(layer_idx)

    def set_index_k(self, layer_idx: int, out_cache_loc: torch.Tensor, idx_k: torch.Tensor) -> None:
        """Write ``idx_k`` into the index-K cache at ``out_cache_loc``."""
        buf = self.get_index_k_buffer(layer_idx)
        if idx_k.shape[1] != 1:
            raise ValueError(
                f"index K is replicated single-head; expected shape "
                f"[N, 1, {self.sparse_index_dim}], got {tuple(idx_k.shape)}"
            )
        buf.index_copy_(0, out_cache_loc.to(torch.long), idx_k.to(buf.dtype))

    def set_index_v(self, layer_idx: int, out_cache_loc: torch.Tensor, idx_v: torch.Tensor) -> None:
        """Write ``idx_v`` into the index-V cache (only when allocated)."""
        buf = self.get_index_v_buffer(layer_idx)
        if buf is None:
            raise RuntimeError(
                f"layer {layer_idx} has disable_index_value=True; index V is not allocated"
            )
        if idx_v.shape[1] != 1:
            raise ValueError(
                f"index V is replicated single-head; expected shape "
                f"[N, 1, {self.sparse_index_dim}], got {tuple(idx_v.shape)}"
            )
        buf.index_copy_(0, out_cache_loc.to(torch.long), idx_v.to(buf.dtype))


class MiniMaxM3KVCacheManagerV2(KVCacheManagerV2):
    """KVCacheManagerV2 subclass with a V2-managed paged index-K cache
    per sparse layer.

    Each sparse local layer registers a ``Role.INDEX_KEY``
    :class:`BufferConfig` via :meth:`_extra_buffers_per_layer` so the
    index-K cache participates in the V2 paged lifecycle (allocation,
    free, prefix reuse) and shares ``life_cycle_id`` with main K/V.

    The optional index-V branch is kept as a plain CUDA tensor for the
    rare ``disable_index_value=False`` test path. M3 production sets
    ``disable_index_value=True`` everywhere, so ``_index_v_buffers``
    stays empty in production.

    Constructor extras (forwarded kwargs go to :class:`KVCacheManagerV2`):
      * ``sparse_layer_ids`` — layer indices using sparse attention.
      * ``disable_index_value_layer_ids`` — subset whose index-V is
        omitted.
      * ``sparse_index_dim`` — width of the index-K/V vectors.
    """

    def __init__(
        self,
        *args,
        sparse_layer_ids=None,
        disable_index_value_layer_ids=None,
        sparse_index_dim: Optional[int] = None,
        **kwargs,
    ):
        # Resolve M3 sparse-layer metadata from explicit kwargs first,
        # then from ``sparse_attn_config``, then from the M3 checkpoint
        # convention (layers 0..2 dense, 3..N-1 sparse,
        # disable_index_value=True, sparse_index_dim=128).
        sparse_attn_config = kwargs.get("sparse_attn_config")
        num_layers = kwargs.get("num_layers")

        if sparse_index_dim is None:
            sparse_index_dim = int(getattr(sparse_attn_config, "sparse_index_dim", 0) or 0) or 128
        if sparse_layer_ids is None:
            if num_layers is not None:
                sparse_layer_ids = list(range(3, int(num_layers)))
            else:
                sparse_layer_ids = []
        if disable_index_value_layer_ids is None:
            disable_index_value_layer_ids = list(sparse_layer_ids)

        # Must be set BEFORE super().__init__ — the base
        # ``_build_cache_config`` invokes ``_extra_buffers_per_layer``
        # which reads these attributes.
        self.sparse_layer_ids = sorted(int(i) for i in sparse_layer_ids)
        self.disable_index_value_layer_ids = set(int(i) for i in disable_index_value_layer_ids)
        self.sparse_index_dim = int(sparse_index_dim)

        super().__init__(*args, **kwargs)

        index_v_layer_ids = set(self.sparse_layer_ids) - self.disable_index_value_layer_ids
        if self.is_disagg and index_v_layer_ids:
            raise ValueError(
                "MiniMax M3 disaggregated serving requires disable_index_value=True "
                "for every sparse layer because the optional test-only index-V cache "
                "is not managed or transferred by KVCacheManagerV2; enabled layers="
                f"{sorted(index_v_layer_ids)}"
            )

        # Optional plain-tensor index-V cache for non-disabled sparse
        # layers (test-only; production has disable_index_value=True
        # on every sparse layer).
        num_total_slots = self._compute_num_total_slots()
        torch_dtype = self._torch_dtype_for_index_cache()
        device = torch.device("cuda")
        self._index_v_buffers: dict[int, torch.Tensor] = {}
        for layer_idx in self.sparse_layer_ids:
            if layer_idx not in self.layer_offsets:
                continue
            if layer_idx not in self.disable_index_value_layer_ids:
                self._index_v_buffers[layer_idx] = torch.zeros(
                    (num_total_slots, 1, self.sparse_index_dim),
                    dtype=torch_dtype,
                    device=device,
                )

    def _extra_buffers_per_layer(self, *, tokens_per_block):
        """Register a per-sparse-layer ``Role.INDEX_KEY`` :class:`BufferConfig`.

        ``size`` is bytes per **block**: ``1 * sparse_index_dim *
        elem_bytes * tokens_per_block``. Keyed by **local** layer id —
        the base ``_build_cache_config`` iterates local ids, so keying
        by global ids would silently skip registration on non-trivial
        PP ranks.
        """
        torch_dtype = self._torch_dtype_for_index_cache()
        elem_bytes = torch.tensor([], dtype=torch_dtype).element_size()
        bytes_per_token = 1 * self.sparse_index_dim * elem_bytes
        size_per_block = bytes_per_token * tokens_per_block
        return {
            self.layer_offsets[layer_id]: [BufferConfig(role=Role.INDEX_KEY, size=size_per_block)]
            for layer_id in self.sparse_layer_ids
            if layer_id in self.layer_offsets
        }

    def get_disagg_role_mapper_kinds(self) -> dict[DataRole, MapperKind]:
        """Declare MiniMax M3's token-major K/V and replicated index-K."""
        return {
            Role.ALL: MapperKind.NHD,
            Role.INDEX_KEY: MapperKind.REPLICATED,
        }

    def _compute_num_total_slots(self) -> int:
        """Total token slots across all blocks in the main K pool.

        Sizes the plain-tensor index-V cache only; the V2-managed
        index-K cache pulls its slot count from ``Role.INDEX_KEY``'s
        page upper bound.
        """
        if not self.layer_offsets:
            return int(self.max_batch_size * self.max_seq_len)
        any_layer_offset = next(iter(self.layer_offsets.values()))
        page_upper = self.impl.get_page_index_upper_bound(any_layer_offset, Role.KEY)
        kv_factor = 1 if self.kv_cache_type == CacheTypeCpp.SELFKONLY else 2
        return int((page_upper // kv_factor) * self.tokens_per_block)

    def _torch_dtype_for_index_cache(self) -> torch.dtype:
        """Match the main cache dtype where possible, fall back to bf16."""
        if self.dtype == DataType.HALF:
            return torch.float16
        if self.dtype == DataType.FLOAT:
            return torch.float32
        return torch.bfloat16

    def get_index_k_buffer(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Return the V2-managed paged index-K view for ``layer_idx``.

        Shape: ``[num_pages, tokens_per_block, 1, sparse_index_dim]``.
        Reads/writes decompose ``slot = (page, within)`` and use
        multi-dim fancy indexing; writes propagate to pool storage.
        """
        return super().get_index_k_buffer(
            layer_idx,
            num_heads=1,
            head_dim=self.sparse_index_dim,
            dtype=self._torch_dtype_for_index_cache(),
        )

    def get_index_v_buffer(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Plain-tensor index-V cache for non-disabled sparse layers."""
        return self._index_v_buffers.get(layer_idx)

    def has_index_value(self, layer_idx: int) -> bool:
        return layer_idx in self._index_v_buffers

    def get_buffers(self, layer_idx: int, kv_layout: str = "NHD") -> Optional[torch.Tensor]:
        """Return a paged K+V view with strides spanning the coalesced pool.

        The base :meth:`KVCacheManagerV2.get_buffers` produces a
        ``[num_pages, kv_factor, ...]`` view with contiguous strides
        that assume the slot holds exactly one layer's K+V. In M3's
        pool the slot packs K+V for *all* layers of the group
        (``scale >= 2 * num_layers_in_group``), so the base view's
        dim-0 stride does not reach the next slot's K for this layer.
        (When INDEX_KEY's per-block size coincides with K/V's, it is
        coalesced into the same pool and contributes to ``scale`` too.)

        The override builds a ``[num_slots, scale, ...]`` view rooted
        at K's base, then slices ``[:, :2]`` to extract K+V. The slice
        preserves the dim-0 stride (``scale * page_stride``), so
        ``view[s, 0/1, ...]`` lands on this layer's K/V at slot ``s``.
        """
        if kv_layout not in ("NHD", "HND"):
            raise ValueError(f"Unsupported kv_layout: {kv_layout}")
        if self.kv_cache_type == CacheTypeCpp.SELFKONLY:
            raise NotImplementedError(
                "MiniMaxM3KVCacheManagerV2.get_buffers does not support SELFKONLY cache type"
            )

        layer_offset = self.layer_offsets[layer_idx]
        addr_key = self.impl.get_mem_pool_base_address(layer_offset, Role.KEY)
        addr_value = self.impl.get_mem_pool_base_address(layer_offset, Role.VALUE)
        page_stride_key = self.impl.get_page_stride(layer_offset, Role.KEY)
        page_stride_value = self.impl.get_page_stride(layer_offset, Role.VALUE)
        # V2 always lays V immediately after K within the per-layer
        # contribution to a slot. The slice ``[:, :2]`` depends on this.
        assert addr_key + page_stride_value == addr_value, (
            f"MiniMaxM3 get_buffers requires addr_K + page_stride "
            f"== addr_V (V immediately after K in slot); got "
            f"addr_K={addr_key} page_stride_V={page_stride_value} "
            f"addr_V={addr_value} for layer {layer_idx}."
        )
        assert page_stride_key == page_stride_value, (
            f"MiniMaxM3 get_buffers requires equal K and V page "
            f"strides; got K={page_stride_key} V="
            f"{page_stride_value}."
        )

        converter = self.impl.get_page_index_converter(layer_offset, Role.KEY)
        scale = int(converter.scale)
        layer_offset_pages = int(converter.layer_offset)
        page_upper_K = self.impl.get_page_index_upper_bound(layer_offset, Role.KEY)
        num_slots_total = page_upper_K + layer_offset_pages
        assert num_slots_total % scale == 0, (
            f"V2 storage inconsistency: page_upper_K + "
            f"layer_offset_pages = {num_slots_total} is not "
            f"divisible by scale = {scale}."
        )
        num_slots = num_slots_total // scale

        element_per_container = 1
        dtype = self.dtype
        if dtype == DataType.NVFP4:
            element_per_container = 2
            torch_dtype = torch.int8
        else:
            torch_dtype = binding_to_torch_dtype(dtype)

        layer_head_dim = self.head_dim_per_layer[layer_offset]
        num_kv_heads = self.num_kv_heads_per_layer[layer_offset]

        if kv_layout == "NHD":
            full_slot_shape = [
                num_slots,
                scale,
                self.tokens_per_block,
                num_kv_heads,
                layer_head_dim // element_per_container,
            ]
        else:
            full_slot_shape = [
                num_slots,
                scale,
                num_kv_heads,
                self.tokens_per_block,
                layer_head_dim // element_per_container,
            ]

        full_view = convert_to_torch_tensor(TensorWrapper(addr_key, torch_dtype, full_slot_shape))
        return full_view[:, :2]

    def _kv_pool_mapping_offset(self, layer_id, layer_group_id, key_base_addr) -> int:
        """Pool-mapping offset from layer position in the pool group.

        The base formula ``exact_div(addr_offset, key_bytes * kv_factor *
        tokens_per_block)`` assumes each layer contributes exactly K+V to
        its pool slot. When index-K coalesces into the K/V pool the layer
        stride is non-uniform (sparse layers add an INDEX_KEY sub-page),
        so derive ``offset`` from ``self.impl.layer_grouping`` instead.
        The M3 forward path uses :meth:`get_buffers` /
        :meth:`get_index_k_buffer` rather than this mapping, so the
        offset just needs to be consistent (layer position in group).
        """
        layers_in_group = list(self.impl.layer_grouping[int(layer_group_id)])
        return layers_in_group.index(int(layer_id))

    def _get_batch_cache_indices_by_pool_id(
        self,
        request_ids,
        *,
        pool_id: int = 0,
        is_kv_aggregate: bool = True,
        num_blocks_per_seq: Optional[Sequence[int]] = None,
    ):
        """Return page indices; padded entries remain ``BAD_PAGE_INDEX`` (-1).

        The base method converts slot ids to V1-style block ids via
        ``base_idx * index_scales[pool_id] // kv_factor``, which is
        only correct when each layer contributes exactly K+V. M3's slot
        packs K+V for all layers of the group, so the scale breaks the
        V1 conversion and produces out-of-bounds block ids during V2
        warmup.

        Bypass the conversion: the M3 forward path indexes paged
        views (built by :meth:`get_buffers` /
        :meth:`get_index_k_buffer`) directly by slot id.
        ``BAD_PAGE_INDEX`` slots remain ``-1`` here because disaggregation's
        :class:`KVRegionExtractorV1` filters ``region_ids >= 0``.
        :meth:`get_block_ids_per_seq` maps them to zero for the attention
        metadata's padded tensor.

        Args:
            request_ids: Request IDs whose page-index rows are returned.
            pool_id: V2 pool whose page indices are requested.
            is_kv_aggregate: Kept for compatibility with the base virtual method.
            num_blocks_per_seq: Optional per-request truncation limits. When
                omitted, preserve the full padded width required by MiniMax
                CUDA-graph metadata initialization.
        """
        res = []
        for req_idx, req_id in enumerate(request_ids):
            kv_cache = self.kv_cache_map[req_id]
            base_page_indices = kv_cache.get_base_page_indices(pool_id)
            if num_blocks_per_seq is not None:
                num_blocks = min(kv_cache.num_blocks, num_blocks_per_seq[req_idx])
                base_page_indices = base_page_indices[:num_blocks]
            res.append(list(base_page_indices))
        return res

    def get_block_ids_per_seq(self, request_ids):
        """Return per-request slot ids matching the per-layer paged view's dim-0.

        Drops the base's final ``i // num_local_layers`` step (paired
        with the base ``index_scales`` multiplication that's also
        bypassed here). Pads with ``0`` to preserve shape.
        """
        block_ids_per_seq = self.get_batch_cache_indices(request_ids)
        block_ids_per_seq_tensors = [
            torch.tensor(
                [i if i != BAD_PAGE_INDEX else 0 for i in sublist],
                dtype=torch.int,
            )
            for sublist in block_ids_per_seq
        ]
        padded_tensor = torch.nn.utils.rnn.pad_sequence(
            block_ids_per_seq_tensors,
            batch_first=True,
            padding_value=0,
        )
        return padded_tensor


def get_minimax_m3_kv_cache_manager_cls():
    """Backward-compatible accessor; prefer importing the class directly."""
    return MiniMaxM3KVCacheManagerV2


__all__ = [
    "MiniMaxM3KVCacheManagerV2",
    "MiniMaxM3SparseIndexCache",
    "get_minimax_m3_kv_cache_manager_cls",
]
