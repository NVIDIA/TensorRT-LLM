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
    buffer alongside the standard K/V buffers; shared Eagle3 draft layers get
    their own virtual attention-op pools.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._utils import (
    TensorWrapper,
    binding_to_torch_dtype,
    convert_to_torch_tensor,
    prefer_pinned,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
    copy_batch_block_offsets_to_device,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, PageIndexMode
from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX
from tensorrt_llm.runtime.kv_cache_manager_v2._config import DataRole


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


def shared_draft_layer_count(spec_config, layer_mask) -> int:
    """How many one-model draft layers the base manager appends to this one.

    Same rule as ``get_pp_layers``: only with a speculative config and no
    ``layer_mask`` (a separate draft manager setup always passes a mask).
    """
    if spec_config is None or layer_mask is not None:
        return 0
    from tensorrt_llm._torch.speculative.utils import get_num_spec_layers

    return int(get_num_spec_layers(spec_config))


def derive_shared_draft_layout(
    num_layers: Optional[int],
    num_kv_heads,
    num_draft: int,
) -> Tuple[List[int], Optional[int]]:
    """Locate the draft layers the base manager appends after the target's.

    ``num_layers`` is the target count; a per-layer ``num_kv_heads`` list (a
    drafter with a different head count) already includes the draft tail and
    pins the total. Returns ``(draft_layer_ids, num_target_layers)``, or
    ``([], None)`` when nothing pins the range.
    """
    num_draft = max(0, int(num_draft))
    if isinstance(num_kv_heads, (list, tuple)):
        total = len(num_kv_heads)
        if num_layers is not None:
            total = max(total, int(num_layers))
    elif num_layers is not None:
        total = int(num_layers) + num_draft
    else:
        return [], None
    num_target = total - num_draft
    return list(range(num_target, total)), num_target


def extend_attention_op_pools_for_shared_draft_layers(
    pool_pointers: torch.Tensor,
    pool_mapping: torch.Tensor,
    num_pools: int,
    draft_layers: Sequence[Tuple[int, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """Give each shared draft layer its own attention-op pool rooted at its K page.

    ``draft_layers`` holds ``(local_layer_idx, key_base_addr, sub_pages_per_slot)``.
    With ``index_scale = sub_pages_per_slot`` and ``kv_offset = 1``, slot ``s``
    maps to page ``s * scale`` (K) and ``s * scale + 1`` (V). Returns
    ``(pool_pointers, pool_mapping, index_scales, kv_offsets, op_pools)`` with
    ``op_pools = [(attention_op_pool_id, source_storage_pool_id)]``.
    """
    pointer_rows = pool_pointers.tolist()
    mapping_rows = pool_mapping.tolist()
    nested = pool_pointers.dim() == 3  # NVFP4 carries [data, scale] pointer pairs
    index_scales: List[int] = []
    kv_offsets: List[int] = []
    op_pools: List[Tuple[int, int]] = []
    for i, (local_layer_idx, key_base_addr, sub_pages_per_slot) in enumerate(draft_layers):
        op_pool_id = num_pools + i
        source_pool_id = int(mapping_rows[local_layer_idx][0])
        pointer_rows.append([[key_base_addr, 0], [0, 0]] if nested else [key_base_addr, 0])
        mapping_rows[local_layer_idx] = [op_pool_id, 0]
        index_scales.append(int(sub_pages_per_slot))
        kv_offsets.append(1)
        op_pools.append((op_pool_id, source_pool_id))
    pinned = prefer_pinned()
    return (
        torch.tensor(pointer_rows, dtype=pool_pointers.dtype, device="cpu", pin_memory=pinned),
        torch.tensor(mapping_rows, dtype=pool_mapping.dtype, device="cpu", pin_memory=pinned),
        torch.tensor(index_scales, dtype=torch.int32, device="cpu", pin_memory=pinned),
        torch.tensor(kv_offsets, dtype=torch.int32, device="cpu", pin_memory=pinned),
        op_pools,
    )


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

    Shared Eagle3 draft layers: the base manager appends them after the target
    layers. They run the generic TRTLLM attention op, which needs a uniform
    per-layer stride inside a pool. M3's pool is not uniform: here an index-K
    page is as large as a K or V page, so V2 packs K, V and index-K of all
    layers into one pool (3 sub-pages per sparse layer, 2 per dense or draft
    layer). M3's own kernels don't care (they use :meth:`get_buffers`), but the
    draft layer would read the wrong pages. So each draft layer gets its own
    virtual attention-op pool rooted at its K page, like
    ``DeepseekV4CacheManager`` does for SWA layers; see
    :meth:`_prepare_page_table_tensor`. Can go once V2 keeps index-K in its
    own pool.
    """

    _main_kv_mapper_kind = MapperKind.NHD
    # Virtual attention-op pools for shared draft layers:
    # (attention_op_pool_id, source_storage_pool_id) plus their copy parameters.
    _draft_op_pools: Tuple[Tuple[int, int], ...] = ()
    _draft_index_scales: Optional[torch.Tensor] = None
    _draft_kv_offsets: Optional[torch.Tensor] = None
    # Extra page sizes trtllm-gen may use with this manager (see
    # FlashInferTrtllmGenFmha); set with the virtual pools.
    trtllm_gen_extra_tokens_per_block: frozenset = frozenset()

    def __init__(
        self,
        *args,
        sparse_layer_ids=None,
        disable_index_value_layer_ids=None,
        sparse_index_dim: Optional[int] = None,
        **kwargs,
    ):
        # Resolve M3 sparse-layer metadata from explicit kwargs first, then
        # from the executor's ``sparse_attention_config`` keyword, then from
        # the M3 checkpoint convention (layers 0..2 dense, 3..N-1 sparse,
        # disable_index_value=True, sparse_index_dim=128). Honoring the
        # executor keyword also makes non-default sparse_index_dim values
        # authoritative for the cache layout instead of falling back to 128.
        # Peeked (not popped) so the base __init__ still receives them.
        sparse_attention_config = kwargs.get("sparse_attention_config")
        num_layers = kwargs.get("num_layers")
        implementation = getattr(sparse_attention_config, "implementation", "triton")
        self._main_kv_mapper_kind = MapperKind.HND if implementation == "msa" else MapperKind.NHD

        if sparse_index_dim is None:
            sparse_index_dim = getattr(sparse_attention_config, "sparse_index_dim", None)
            if sparse_index_dim is None:
                sparse_index_dim = 128
        sparse_index_dim = int(sparse_index_dim)
        if sparse_index_dim <= 0:
            raise ValueError(
                f"MiniMax M3 sparse_index_dim must be greater than 0, got {sparse_index_dim}."
            )
        # Shared draft layers sit above the target layers and have no index-K
        # cache, so the sparse-layer default stops at the target range.
        self._shared_draft_layer_ids, num_target_layers = derive_shared_draft_layout(
            num_layers,
            kwargs.get("num_kv_heads"),
            shared_draft_layer_count(kwargs.get("spec_config"), kwargs.get("layer_mask")),
        )
        if sparse_layer_ids is None:
            if num_target_layers is not None:
                sparse_layer_ids = list(range(3, num_target_layers))
            else:
                sparse_layer_ids = []
        if disable_index_value_layer_ids is None:
            disable_index_value_layer_ids = list(sparse_layer_ids)

        # Must be set BEFORE super().__init__ — the base
        # ``_build_base_config`` invokes ``_extra_buffers_per_layer``
        # which reads these attributes.
        self.sparse_layer_ids = sorted(int(i) for i in sparse_layer_ids)
        self.disable_index_value_layer_ids = set(int(i) for i in disable_index_value_layer_ids)
        self.sparse_index_dim = sparse_index_dim
        self.indexer_kv_dtype = str(getattr(sparse_attention_config, "indexer_kv_dtype", "bf16"))
        if self.indexer_kv_dtype not in ("bf16", "fp8"):
            raise ValueError(
                "MiniMax M3 indexer_kv_dtype must be 'bf16' or 'fp8', got "
                f"{self.indexer_kv_dtype!r}."
            )
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

    def _prepare_page_table_tensor(self, index_mapper_capacity: int) -> None:
        """Base pool tables plus one virtual attention-op pool per shared draft layer.

        See the class docstring for why. The virtual pool reuses the source
        pool's slot ids; :meth:`copy_batch_block_offsets` scales them.
        """
        super()._prepare_page_table_tensor(index_mapper_capacity)
        draft_layers = [
            layer_idx
            for layer_idx in self._shared_draft_layer_ids
            if layer_idx in self.layer_offsets
        ]
        if self.is_draft or not draft_layers:
            return
        if self.enable_swa_scratch_reuse:
            raise NotImplementedError(
                "MiniMax-M3 shared Eagle3 draft layers do not support SWA scratch reuse."
            )
        geometry = []
        for layer_idx in draft_layers:
            key_base_addr, _dtype, _num_slots, sub_pages_per_slot, _shape = self._kv_slot_geometry(
                layer_idx
            )
            geometry.append((self.layer_offsets[layer_idx], key_base_addr, sub_pages_per_slot))
        (
            self.kv_cache_pool_pointers,
            self.kv_cache_pool_mapping,
            self._draft_index_scales,
            self._draft_kv_offsets,
            op_pools,
        ) = extend_attention_op_pools_for_shared_draft_layers(
            self.kv_cache_pool_pointers, self.kv_cache_pool_mapping, self.num_pools, geometry
        )
        self._draft_op_pools = tuple(op_pools)
        self.num_attention_op_pools = self.num_pools + len(op_pools)
        # Draft layers run at the target's 128-token pages. trtllm-gen has P128
        # kernels for their dense-GQA shapes but not for every shape, so opt in
        # here rather than in the global allowlist.
        if self.tokens_per_block == 128:
            self.trtllm_gen_extra_tokens_per_block = frozenset({128})
        logger.info(
            f"[unified-kv] draft layers {draft_layers} share the target KV cache manager; "
            f"attention-op pools {[pool for pool, _ in op_pools]} address their pages."
        )

    def copy_batch_block_offsets(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
        max_blocks: Optional[int] = None,
    ) -> None:
        super().copy_batch_block_offsets(
            dst_tensor, request_ids, beam_width, num_contexts, num_seqs, max_blocks=max_blocks
        )
        if not self._draft_op_pools:
            return
        # Fill each virtual pool from its source pool's slot ids with the draft
        # layer's scale.
        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts, beam_width)
        for i, (op_pool_id, source_pool_id) in enumerate(self._draft_op_pools):
            copy_batch_block_offsets_to_device(
                self.host_kv_cache_block_offsets[source_pool_id : source_pool_id + 1],
                dst_tensor[op_pool_id : op_pool_id + 1],
                copy_idx,
                self._draft_index_scales[i : i + 1],
                self._draft_kv_offsets[i : i + 1],
                self._stream.cuda_stream,
            )

    def _extra_buffers_per_layer(self, *, tokens_per_block):
        """Register a per-sparse-layer ``Role.INDEX_KEY`` :class:`BufferConfig`.

        ``size`` is bytes per **block**: ``1 * sparse_index_dim *
        elem_bytes * tokens_per_block``. Keyed by **local** layer id —
        the base ``_build_base_config`` iterates local ids, so keying
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
        """Declare the backend's main K/V layout and replicated index-K."""
        return {
            Role.ALL: self._main_kv_mapper_kind,
            Role.INDEX_KEY: MapperKind.REPLICATED,
        }

    def _main_kv_layout_name(self) -> str:
        """Return the tensor-view layout name for the selected mapper kind."""
        return "HND" if self._main_kv_mapper_kind == MapperKind.HND else "NHD"

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
        """Return the independently configured index-cache storage dtype."""
        if self.indexer_kv_dtype == "fp8":
            return torch.float8_e4m3fn
        return torch.bfloat16

    def get_index_k_buffer(
        self, layer_idx: int, kv_layout: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """Return the V2-managed paged index-K view for ``layer_idx``.

        NHD shape is ``[num_pages, tokens_per_block, 1, sparse_index_dim]``;
        HND shape is ``[num_pages, 1, tokens_per_block, sparse_index_dim]``.
        When omitted, ``kv_layout`` follows the selected sparse backend.
        """
        if kv_layout is None:
            kv_layout = self._main_kv_layout_name()
        return super().get_index_k_buffer(
            layer_idx,
            num_heads=1,
            head_dim=self.sparse_index_dim,
            dtype=self._torch_dtype_for_index_cache(),
            kv_layout=kv_layout,
        )

    def get_index_v_buffer(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Plain-tensor index-V cache for non-disabled sparse layers."""
        return self._index_v_buffers.get(layer_idx)

    def has_index_value(self, layer_idx: int) -> bool:
        return layer_idx in self._index_v_buffers

    def _kv_slot_geometry(
        self, layer_idx: int, kv_layout: Optional[str] = None
    ) -> Tuple[int, torch.dtype, int, int, List[int]]:
        """Where a layer's K/V live in the coalesced pool.

        Returns ``(addr_key, torch_dtype, num_slots, scale, page_shape)``:
        ``scale`` sub-pages per slot, this layer's K at sub-page 0 and V at
        sub-page 1 from ``addr_key``. Used by :meth:`get_buffers` and the draft
        layers' virtual pools. ``kv_layout`` defaults to the backend's layout.
        """
        if kv_layout is None:
            kv_layout = self._main_kv_layout_name()
        if kv_layout not in ("NHD", "HND"):
            raise ValueError(f"Unsupported kv_layout: {kv_layout}")
        if self.kv_cache_type == CacheTypeCpp.SELFKONLY:
            raise NotImplementedError(
                "MiniMaxM3KVCacheManagerV2 does not support the SELFKONLY cache type"
            )

        layer_offset = self.layer_offsets[layer_idx]
        addr_key = self.impl.get_mem_pool_base_address(layer_offset, Role.KEY)
        addr_value = self.impl.get_mem_pool_base_address(layer_offset, Role.VALUE)
        page_stride_key = self.impl.get_page_stride(layer_offset, Role.KEY)
        page_stride_value = self.impl.get_page_stride(layer_offset, Role.VALUE)
        # V2 always lays V immediately after K within the per-layer
        # contribution to a slot. The slice ``[:, :2]`` depends on this.
        assert addr_key + page_stride_value == addr_value, (
            f"MiniMaxM3 requires addr_K + page_stride "
            f"== addr_V (V immediately after K in slot); got "
            f"addr_K={addr_key} page_stride_V={page_stride_value} "
            f"addr_V={addr_value} for layer {layer_idx}."
        )
        assert page_stride_key == page_stride_value, (
            f"MiniMaxM3 requires equal K and V page "
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
        containers = layer_head_dim // element_per_container

        if kv_layout == "NHD":
            page_shape = [self.tokens_per_block, num_kv_heads, containers]
        else:
            page_shape = [num_kv_heads, self.tokens_per_block, containers]
        return addr_key, torch_dtype, num_slots, scale, page_shape

    def get_buffers(
        self, layer_idx: int, kv_layout: Optional[str] = None
    ) -> Optional[torch.Tensor]:
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
        When omitted, ``kv_layout`` follows the selected sparse backend.
        """
        addr_key, torch_dtype, num_slots, scale, page_shape = self._kv_slot_geometry(
            layer_idx, kv_layout
        )
        full_slot_shape = [num_slots, scale, *page_shape]
        full_view = convert_to_torch_tensor(TensorWrapper(addr_key, torch_dtype, full_slot_shape))
        return full_view[:, :2]

    def _kv_pool_mapping_offset(self, layer_id, layer_group_id, key_base_addr) -> int:
        """Pool-mapping offset from the layer's physical position in its pool.

        The base formula ``exact_div(addr_offset, key_bytes * kv_factor *
        tokens_per_block)`` assumes each layer contributes exactly K+V to
        its pool slot. When index-K coalesces into the K/V pool the layer
        stride is non-uniform (sparse layers add an INDEX_KEY sub-page),
        so no uniform-stride offset exists. The M3 forward path uses
        :meth:`get_buffers` / :meth:`get_index_k_buffer` rather than this
        mapping, so the offset just needs to be a consistent per-layer
        position. Rank the group's layers by their K base address instead
        of by ``layer_grouping`` iteration order: the ordering of
        ``layer_grouping`` is not a V2 API contract, while the address
        rank always reflects the physical slot layout (and keeps the
        NVFP4 ``block_scale_offset == offset`` cross-check in the base
        pool-mapping loop meaningful).
        """
        layers_by_addr = sorted(
            self.impl.layer_grouping[int(layer_group_id)],
            key=lambda lid: self.impl.get_mem_pool_base_address(
                lid, Role.KEY, PageIndexMode.SHARED
            ),
        )
        return layers_by_addr.index(int(layer_id))

    def _get_batch_cache_indices_by_pool_id(
        self,
        request_ids,
        *,
        pool_id: int = 0,
        is_kv_aggregate: bool = True,
        num_blocks_per_seq: Optional[Sequence[int]] = None,
        index_scale: Optional[int] = None,
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
            index_scale: Kept for compatibility with the base virtual method;
                M3 bypasses the V1 block-id conversion entirely, so any
                caller-supplied scale is ignored alongside ``index_scales``.
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

        The rows are written through a numpy view of a single zero-filled,
        pinned result, so the attention metadata builders ship it to the device
        in one asynchronous copy.
        """
        block_ids_per_seq = self.get_batch_cache_indices(request_ids)
        batch = len(block_ids_per_seq)
        max_blocks = max((len(block_ids) for block_ids in block_ids_per_seq), default=0)
        padded_tensor = torch.zeros(
            (batch, max_blocks), dtype=torch.int32, pin_memory=prefer_pinned()
        )
        rows = padded_tensor.numpy()
        for row, block_ids in zip(rows, block_ids_per_seq):
            row[: len(block_ids)] = block_ids
        # BAD_PAGE_INDEX marks padding, which this tensor reports as 0.
        rows[rows == BAD_PAGE_INDEX] = 0
        return padded_tensor


def get_minimax_m3_kv_cache_manager_cls():
    """Backward-compatible accessor; prefer importing the class directly."""
    return MiniMaxM3KVCacheManagerV2


__all__ = [
    "MiniMaxM3KVCacheManagerV2",
    "MiniMaxM3SparseIndexCache",
    "derive_shared_draft_layout",
    "extend_attention_op_pools_for_shared_draft_layers",
    "get_minimax_m3_kv_cache_manager_cls",
    "shared_draft_layer_count",
]
