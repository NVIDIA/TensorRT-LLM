# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V2 hybrid cache manager for QSA sparse attention."""

from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import Tensor, replicated
from tensorrt_llm._utils import TensorWrapper, binding_to_torch_dtype, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, DataRole, PageIndexMode

from .constants import (
    QSA_INDEX_K_CACHE_DTYPE,
    QSA_MAIN_KV_ROLES,
    QSA_POSITION_CACHE_DTYPE,
    QSA_POSITION_COORDINATE_AXES,
    QSA_SPARSE_KV_CACHE_DTYPES,
)
from .params import QSASparseParams

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig

# Storage-layout constants; model geometry comes from QSASparseParams.
_IDENTITY_PAGE_INDEX_EXPANSION = 1
_SHARED_POSITION_ROLE_INDEX = 0
_INDEX_K_ELEMENT_BYTES = torch.empty((), dtype=QSA_INDEX_K_CACHE_DTYPE).element_size()
_POSITION_ELEMENT_BYTES = torch.empty((), dtype=QSA_POSITION_CACHE_DTYPE).element_size()
# NVFP4 packs two e2m1 values per byte and keeps one e4m3 scale byte per
# 16-element block.
_NVFP4_VALUES_PER_BYTE = 2
_NVFP4_BLOCK_SIZE = 16
# Granularity by which the packed side-plane page grows until its size cannot
# be a main-cache page; keeps every slot's planes 16-byte aligned.
_SIDE_PLANES_SIZE_STEP = 16

# Per-token RoPE/mRoPE coordinates used when raw index keys are compressed.
# They are request state shared by all local QSA layers.
QSA_INDEX_POSITION = DataRole("qsa_index_position")
# Every local layer's index-K page followed by the position page, as one
# buffer per lifecycle slot. Registered instead of the per-layer INDEX_KEY and
# the position role when either page could be the size of a K/V or block-scale
# page (see `_extra_buffers_per_layer`).
QSA_SIDE_PLANES = DataRole("qsa_side_planes")


class QSAMambaHybridCacheManagerV2(MambaHybridCacheManagerV2):
    """Hybrid GDN/KV manager with lifecycle-coupled sparse side buffers.

    QSA adds a per-layer index-K cache and request-wide position coordinates.
    These buffers share the main K/V allocation, eviction, and prefix-reuse
    lifecycle. The parent hybrid manager owns GDN/Mamba recurrent state.

    Index geometry is resolved from the same serving/checkpoint configuration
    as the indexer. A local fallback could otherwise give the indexer and its
    paged cache different shapes or page strides.

    V2 storage coalesces the buffers of one lifecycle by page size alone. The
    attention op addresses a layer's K/V pages and its NVFP4 block-scale pages
    with one block-offset table, so a side page that lands in the K/V pool
    (NVFP4 at 32 tokens per block: index-K 8192 B == K == V) changes the K/V
    slot stride but not the block-scale one, and any side page in the K/V pool
    breaks the K/V pool mapping. When a side page could be the size of a main
    page, the side planes are therefore registered as one packed per-slot buffer
    whose size can be no main-cache page, and this manager serves their
    per-layer views itself. Both the choice and the packed size are the same on
    every TP and attention-DP layout, so disaggregated peers agree on the roles.
    """

    def __init__(
        self,
        *args,
        sparse_attention_config: Optional["SparseAttentionConfig"] = None,
        pretrained_config: Optional[object] = None,
        layer_mask: Optional[list[bool]] = None,
        **kwargs,
    ) -> None:
        if sparse_attention_config is None:
            raise ValueError("sparse_attention_config is required for the QSA cache manager")
        if layer_mask is None:
            raise ValueError("QSA cache allocation requires the full-attention layer mask")
        sparse_params = sparse_attention_config.to_sparse_params(
            pretrained_config=pretrained_config
        )
        if not isinstance(sparse_params, QSASparseParams):
            raise ValueError("QSA cache manager requires QSA sparse parameters")

        # Match the cache's width and head count exactly to the indexer. V2's
        # page-index converter already handles roles that share a physical pool.
        self.qsa_index_dim = sparse_params.index_head_dim
        self.qsa_index_kv_heads = sparse_params.index_kv_heads
        # layer_mask uses global IDs; layer_offsets later selects this PP rank.
        self.qsa_sparse_layer_ids = [
            layer_idx for layer_idx, enabled in enumerate(layer_mask) if enabled
        ]
        self.qsa_position_layer_id: Optional[int] = None
        # Bytes of one packed side-plane page, or None while the side planes
        # are registered as per-layer roles. Set by `_extra_buffers_per_layer`.
        self.qsa_side_planes_size: Optional[int] = None
        super().__init__(*args, layer_mask=layer_mask, **kwargs)

    def _page_buffer_tensors(
        self, layer_id: int, role: str, size: int, tokens: int, state: bool
    ) -> tuple[Tensor, ...]:
        """Index positions are request-wide coordinates, so every rank holds the same bytes; so do
        the packed side planes and the index keys, which project one head alike on every rank.
        They carry no head axis, which splitting them by the rank's KV heads would assume."""
        if role in (QSA_INDEX_POSITION, QSA_SIDE_PLANES, Role.INDEX_KEY):
            return replicated(size)
        return super()._page_buffer_tensors(layer_id, role, size, tokens, state)

    @cached_property
    def qsa_local_sparse_layer_ids(self) -> List[int]:
        """Sparse layers this rank owns, in model order."""
        return [
            layer_id for layer_id in self.qsa_sparse_layer_ids if layer_id in self.layer_offsets
        ]

    @cached_property
    def qsa_shared_topk_slots(self) -> Dict[int, int]:
        """Row each sparse layer captures into in the shared MTP Top-K buffers.

        Every sparse layer runs its own indexer projection over its own index
        cache, so a captured selection is only valid for the layer that
        produced it.
        """
        return {layer_id: slot for slot, layer_id in enumerate(self.qsa_local_sparse_layer_ids)}

    def _extra_buffers_per_layer(
        self,
        *,
        tokens_per_block: int,
    ) -> dict[int, list[BufferConfig]]:
        """Register per-layer index K and lifecycle-aligned position pages.

        Both are registered as their own roles unless either page could have
        the size of a K/V or block-scale page at some split of the KV heads over
        ranks (`_main_cache_page_sizes`), which V2 would coalesce into that pool.
        They are then packed, in local sparse-layer order with the position page
        last, into one `QSA_SIDE_PLANES` buffer on the first local sparse layer,
        grown until its size is none of those page sizes. Neither the choice nor
        the size depends on the KV heads this rank holds, so peers of different
        TP or attention-DP layouts register the same transfer roles.
        """
        index_size = (
            self.qsa_index_kv_heads * self.qsa_index_dim * _INDEX_K_ELEMENT_BYTES * tokens_per_block
        )
        position_size = QSA_POSITION_COORDINATE_AXES * _POSITION_ELEMENT_BYTES * tokens_per_block
        local_sparse_layers = self.qsa_local_sparse_layer_ids
        # Coordinates are request-wide, so all local indexers use one view.
        # Register the position role once; duplicating it on every sparse layer
        # wastes one three-axis int32 page per layer without adding state.
        self.qsa_position_layer_id = next(iter(local_sparse_layers), None)
        self.qsa_side_planes_size = None
        if self.qsa_position_layer_id is None:
            return {}
        position_local_idx = self.layer_offsets[self.qsa_position_layer_id]
        main_page_sizes = self._main_cache_page_sizes(tokens_per_block)
        if index_size in main_page_sizes or position_size in main_page_sizes:
            packed_size = len(local_sparse_layers) * index_size + position_size
            while packed_size in main_page_sizes:
                packed_size += _SIDE_PLANES_SIZE_STEP
            self.qsa_side_planes_size = packed_size
            return {position_local_idx: [BufferConfig(role=QSA_SIDE_PLANES, size=packed_size)]}
        result = {
            self.layer_offsets[layer_id]: [BufferConfig(role=Role.INDEX_KEY, size=index_size)]
            for layer_id in local_sparse_layers
        }
        result[position_local_idx].append(BufferConfig(role=QSA_INDEX_POSITION, size=position_size))
        return result

    def _main_cache_page_sizes(self, tokens_per_block: int) -> set[int]:
        """Every size, in bytes, a local layer's K/V or block-scale page can have
        at any split of that layer's KV heads over ranks: one head per rank up to
        all of them. The set is the same on every TP and attention-DP layout."""
        global_layer = {local: layer for layer, local in self.layer_offsets.items()}
        sizes = set()
        for local_idx in range(self.num_local_layers):
            heads = self.num_kv_heads_per_layer[local_idx]
            total = (
                self.num_kv_heads
                if isinstance(self.num_kv_heads, int)
                else self.num_kv_heads[global_layer[local_idx]]
            )
            for role in self._get_buffer_roles_for_layer(local_idx):
                page = self.get_layer_bytes_per_token(local_idx, role) * tokens_per_block
                if page:
                    sizes.update(page // heads * h for h in range(1, total + 1))
        return sizes

    def _side_planes(self) -> torch.Tensor:
        """uint8 ``[slots, qsa_side_planes_size]`` over the packed side-plane pool."""
        local_idx = self.layer_offsets[self.qsa_position_layer_id]
        page_stride = self.impl.get_page_stride(local_idx, QSA_SIDE_PLANES)
        if page_stride != self.qsa_side_planes_size:
            raise RuntimeError(
                "QSA side-plane page stride mismatch: "
                f"expected {self.qsa_side_planes_size}, got {page_stride}"
            )
        converter = self.impl.get_page_index_converter(local_idx, QSA_SIDE_PLANES)
        if (
            int(converter.scale) != 1
            or int(converter.layer_offset) != 0
            or int(converter.expansion) != _IDENTITY_PAGE_INDEX_EXPANSION
        ):
            raise RuntimeError(
                "QSA side planes share their V2 pool with another buffer: "
                f"scale={int(converter.scale)}, layer_offset={int(converter.layer_offset)}, "
                f"expansion={int(converter.expansion)}"
            )
        addr = self.impl.get_mem_pool_base_address(local_idx, QSA_SIDE_PLANES, PageIndexMode.SHARED)
        num_slots = self.impl.get_page_index_upper_bound(local_idx, QSA_SIDE_PLANES)
        return convert_to_torch_tensor(
            TensorWrapper(addr, torch.uint8, [num_slots, self.qsa_side_planes_size])
        )

    def get_index_k_buffer(
        self,
        layer_idx: int,
        kv_layout: str = "NHD",
    ) -> Optional[torch.Tensor]:
        """Return the index-K view using the indexer's resolved geometry.

        ``[slots, tokens_per_block, heads, dim]`` (NHD) or ``[slots, heads,
        tokens_per_block, dim]`` (HND), indexed by lifecycle slot.
        """
        if self.qsa_side_planes_size is not None:
            if kv_layout not in ("NHD", "HND"):
                raise ValueError(f"Unsupported kv_layout: {kv_layout}")
            if layer_idx not in self.qsa_local_sparse_layer_ids:
                return None
            ordinal = self.qsa_local_sparse_layer_ids.index(layer_idx)
            index_size = (
                self.qsa_index_kv_heads
                * self.qsa_index_dim
                * _INDEX_K_ELEMENT_BYTES
                * self.tokens_per_block
            )
            planes = self._side_planes()
            plane = planes[:, ordinal * index_size : (ordinal + 1) * index_size]
            if kv_layout == "NHD":
                page_shape = [self.tokens_per_block, self.qsa_index_kv_heads, self.qsa_index_dim]
            else:
                page_shape = [self.qsa_index_kv_heads, self.tokens_per_block, self.qsa_index_dim]
            return plane.view(QSA_INDEX_K_CACHE_DTYPE).view(planes.shape[0], *page_shape)
        return super().get_index_k_buffer(
            layer_idx,
            num_heads=self.qsa_index_kv_heads,
            head_dim=self.qsa_index_dim,
            dtype=QSA_INDEX_K_CACHE_DTYPE,
            kv_layout=kv_layout,
        )

    def get_buffers(
        self,
        layer_idx: int,
        kv_layout: str = "NHD",
    ) -> Optional[torch.Tensor]:
        """Return this layer's adjacent K/V roles with the physical slot stride.

        QSA tables store lifecycle slot IDs. Preserving V2's coalesced stride is
        therefore required even when several layers share one physical pool.

        ``[slots, 2, heads, tokens_per_block, width]`` (HND) or ``[slots, 2,
        tokens_per_block, heads, width]`` (NHD), K then V. Half, bf16 and fp8
        pools are viewed in their element type at ``width = head_dim``. An
        NVFP4 pool is viewed as uint8 at ``width = head_dim // 2``: two e2m1
        values per byte, element ``2i`` in the low nibble; its block scales are
        `get_block_scale_buffers`. Any other format is the parent's layout.
        """
        if self.dtype in QSA_SPARSE_KV_CACHE_DTYPES:
            return self._paired_role_view(
                layer_idx,
                (Role.KEY, Role.VALUE),
                binding_to_torch_dtype(self.dtype),
                1,
                kv_layout,
            )
        if self.dtype == DataType.NVFP4:
            return self._paired_role_view(
                layer_idx, (Role.KEY, Role.VALUE), torch.uint8, _NVFP4_VALUES_PER_BYTE, kv_layout
            )
        return super().get_buffers(layer_idx, kv_layout)

    def get_block_scale_buffers(
        self,
        layer_idx: int,
        kv_layout: str = "NHD",
    ) -> Optional[torch.Tensor]:
        """Return an NVFP4 pool's e4m3 block scales for this layer, K then V.

        uint8 ``[slots, 2, heads, tokens_per_block, head_dim // 16]`` (HND) or
        ``[slots, 2, tokens_per_block, heads, head_dim // 16]`` (NHD) with the
        same slot indexing as `get_buffers`: one scale byte per 16 values. The
        attention op writes K's scales token-major within a head page, and V's
        in groups of four tokens, ``[tokens_per_block / 4, head_dim / 16, 4]``
        per head; the view is the page as bytes and leaves that order to the
        reader. None for a layer this rank does not hold.
        """
        if self.dtype != DataType.NVFP4:
            raise ValueError(f"block scales exist only for an NVFP4 pool, not {self.dtype}")
        return self._paired_role_view(
            layer_idx,
            (Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE),
            torch.uint8,
            _NVFP4_BLOCK_SIZE,
            kv_layout,
        )

    def _paired_role_view(
        self,
        layer_idx: int,
        roles: tuple[DataRole, DataRole],
        dtype: torch.dtype,
        values_per_element: int,
        kv_layout: str,
    ) -> Optional[torch.Tensor]:
        """Slot-indexed view over a layer's adjacent K-side and V-side pages."""
        if kv_layout not in ("NHD", "HND"):
            raise ValueError(f"Unsupported kv_layout: {kv_layout}")
        if layer_idx not in self.layer_offsets:
            return None
        if self.kv_cache_type == CacheTypeCpp.SELFKONLY:
            raise NotImplementedError("QSA sparse attention requires both K and V cache buffers")

        key_role, value_role = roles
        layer_offset = self.layer_offsets[layer_idx]
        addr_key = self.impl.get_mem_pool_base_address(layer_offset, key_role, PageIndexMode.SHARED)
        addr_value = self.impl.get_mem_pool_base_address(
            layer_offset, value_role, PageIndexMode.SHARED
        )
        key_stride = self.impl.get_page_stride(layer_offset, key_role)
        value_stride = self.impl.get_page_stride(layer_offset, value_role)
        if addr_key + key_stride != addr_value or key_stride != value_stride:
            raise RuntimeError(
                "QSA K/V buffers are not adjacent equal-sized V2 roles: "
                f"layer={layer_idx}, roles={key_role}/{value_role}, "
                f"key_stride={key_stride}, value_stride={value_stride}"
            )

        converter = self.impl.get_page_index_converter(layer_offset, key_role)
        scale = int(converter.scale)
        if scale < QSA_MAIN_KV_ROLES:
            raise RuntimeError(
                f"QSA K/V roles do not share one lifecycle slot: layer={layer_idx}, {scale=}"
            )
        if int(converter.expansion) != _IDENTITY_PAGE_INDEX_EXPANSION:
            raise RuntimeError("QSA does not support expanded V2 page indices")
        layer_offset_pages = int(converter.layer_offset)
        page_upper = self.impl.get_page_index_upper_bound(layer_offset, key_role)
        num_slots_total = page_upper + layer_offset_pages
        if num_slots_total % scale != 0:
            raise RuntimeError(
                "QSA K/V page mapping is inconsistent: "
                f"{num_slots_total=} is not divisible by {scale=}"
            )
        num_slots = num_slots_total // scale

        head_dim = self.head_dim_per_layer[layer_offset]
        num_heads = self.num_kv_heads_per_layer[layer_offset]
        width = head_dim // values_per_element
        if kv_layout == "NHD":
            page_shape = [self.tokens_per_block, num_heads, width]
        else:
            page_shape = [num_heads, self.tokens_per_block, width]
        page_bytes = self.tokens_per_block * num_heads * width * dtype.itemsize
        if page_bytes != key_stride:
            raise RuntimeError(
                f"QSA {key_role} page is {key_stride} B, but {page_shape} {dtype} is "
                f"{page_bytes} B: layer={layer_idx}"
            )
        full_view = convert_to_torch_tensor(
            TensorWrapper(
                addr_key,
                dtype,
                [num_slots, scale, *page_shape],
            )
        )
        return full_view[:, :QSA_MAIN_KV_ROLES]

    @torch.compiler.disable
    def get_qsa_position_buffer(self) -> Optional[torch.Tensor]:
        """Return per-token three-axis RoPE/mRoPE position coordinates.

        int32 ``[slots, tokens_per_block, 3]``, indexed by lifecycle slot.
        """
        if self.qsa_position_layer_id is None:
            return None
        if self.qsa_side_planes_size is not None:
            position_size = (
                QSA_POSITION_COORDINATE_AXES * _POSITION_ELEMENT_BYTES * self.tokens_per_block
            )
            planes = self._side_planes()
            start = len(self.qsa_local_sparse_layer_ids) * (
                self.qsa_index_kv_heads
                * self.qsa_index_dim
                * _INDEX_K_ELEMENT_BYTES
                * self.tokens_per_block
            )
            plane = planes[:, start : start + position_size]
            return plane.view(QSA_POSITION_CACHE_DTYPE).view(
                planes.shape[0], self.tokens_per_block, QSA_POSITION_COORDINATE_AXES
            )
        local_idx = self.layer_offsets[self.qsa_position_layer_id]
        addr = self.impl.get_mem_pool_base_address(
            local_idx,
            QSA_INDEX_POSITION,
            PageIndexMode.SHARED,
        )
        page_stride = self.impl.get_page_stride(local_idx, QSA_INDEX_POSITION)
        expected_stride = (
            QSA_POSITION_COORDINATE_AXES * _POSITION_ELEMENT_BYTES * self.tokens_per_block
        )
        if page_stride != expected_stride:
            raise RuntimeError(
                "QSA position-cache page stride mismatch: "
                f"expected {expected_stride}, got {page_stride}"
            )
        converter = self.impl.get_page_index_converter(
            local_idx,
            QSA_INDEX_POSITION,
        )
        page_upper = self.impl.get_page_index_upper_bound(
            local_idx,
            QSA_INDEX_POSITION,
        )
        num_pages_with_offset = page_upper + int(converter.layer_offset)
        scale = int(converter.scale)
        if scale <= 0:
            raise RuntimeError(f"QSA received an invalid position page-index scale: {scale}")
        if int(converter.expansion) != _IDENTITY_PAGE_INDEX_EXPANSION:
            raise RuntimeError("QSA does not support expanded V2 position page indices")
        if num_pages_with_offset % scale != 0:
            raise RuntimeError("QSA position-cache page mapping is inconsistent")
        num_slots = num_pages_with_offset // scale
        full_view = convert_to_torch_tensor(
            TensorWrapper(
                addr,
                QSA_POSITION_CACHE_DTYPE,
                [
                    num_slots,
                    scale,
                    self.tokens_per_block,
                    QSA_POSITION_COORDINATE_AXES,
                ],
            )
        )
        return full_view[:, _SHARED_POSITION_ROLE_INDEX]

    def _build_pool_mapping_tensors(self):
        """Refuse an NVFP4 layout whose K/V and block-scale slots differ.

        The attention op reads a layer's K/V pages and their block scales
        through one block-offset table: a page index scaled for the K/V pool
        addresses the scale pool too. A layer whose two page-index scales
        differ would read and write other slots' scales without any error.
        """
        if self.dtype == DataType.NVFP4:
            for layer_id in self.pp_layers:
                local_idx = self.layer_offsets[layer_id]
                if self._is_local_mamba_layer(local_idx):
                    continue
                data_scale = self.impl.get_page_index_scale(local_idx, Role.KEY)
                scale_scale = self.impl.get_page_index_scale(local_idx, Role.KEY_BLOCK_SCALE)
                if data_scale != scale_scale:
                    raise RuntimeError(
                        "NVFP4 K/V and block-scale pools hold a different number of pages "
                        f"per lifecycle slot (layer {layer_id}: {data_scale} vs {scale_scale}); "
                        "another buffer was coalesced into one of them, and the attention op "
                        "would address the block scales of other slots"
                    )
        return super()._build_pool_mapping_tensors()

    def get_qsa_attention_pool_layout(self) -> tuple[int, int]:
        """Return the shared attention-pool index and V2 page-index scale.

        One graph-stable QSA block table is reused by every local sparse layer,
        so their main-K roles must resolve lifecycle slots through the same
        attention pool and converter.
        """
        if self.qsa_position_layer_id is None:
            raise RuntimeError("QSA cache manager has no local sparse layer")
        local_idx = self.layer_offsets[self.qsa_position_layer_id]
        pool_id = self.layer_to_pool_mapping_dict[local_idx]
        if pool_id >= self.num_attention_op_pools:
            raise RuntimeError(
                "QSA K/V pool is not represented in the attention block table: "
                f"pool {pool_id}, attention pools {self.num_attention_op_pools}"
            )
        converter = self.impl.get_page_index_converter(local_idx, Role.KEY)
        scale = int(converter.scale)
        for layer_id in self.qsa_sparse_layer_ids:
            if layer_id not in self.layer_offsets:
                continue
            candidate_idx = self.layer_offsets[layer_id]
            candidate_pool = self.layer_to_pool_mapping_dict[candidate_idx]
            candidate_converter = self.impl.get_page_index_converter(candidate_idx, Role.KEY)
            if candidate_pool != pool_id or int(candidate_converter.scale) != scale:
                raise RuntimeError(
                    "QSA local layers do not share one attention page mapping: "
                    f"layer {layer_id} uses pool/scale "
                    f"{candidate_pool}/{int(candidate_converter.scale)}, expected "
                    f"{pool_id}/{scale}"
                )
            if int(candidate_converter.expansion) != _IDENTITY_PAGE_INDEX_EXPANSION:
                raise RuntimeError("QSA does not support expanded V2 page indices")
        if scale <= 0:
            raise RuntimeError(f"QSA received an invalid V2 page-index scale: {scale}")
        return pool_id, scale


__all__ = ["QSA_INDEX_POSITION", "QSA_SIDE_PLANES", "QSAMambaHybridCacheManagerV2"]
