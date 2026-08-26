# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V2 hybrid cache manager for QSA sparse attention."""

from __future__ import annotations

from typing import List, Optional

import torch

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._utils import TensorWrapper, binding_to_torch_dtype, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, PageIndexMode
from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX
from tensorrt_llm.runtime.kv_cache_manager_v2._config import DataRole

QSA_INDEX_POSITION = DataRole("qsa_index_position")


class QSAMambaHybridCacheManagerV2(MambaHybridCacheManagerV2):
    """Hybrid GDN/KV manager with lifecycle-coupled sparse side buffers.

    Each full-attention layer owns a replicated BF16 raw/compressed index-K
    buffer. One local full-attention layer additionally owns a three-axis
    int32 position view. The position bytes are registered on every sparse
    layer so all full-attention layers retain an identical V2 pool layout; the
    indexers share the first local layer's view. All buffers use the same V2
    lifecycle as the corresponding main K/V pages.
    """

    def __init__(
        self,
        *args,
        qsa_index_dim: int = 128,
        layer_mask: Optional[list[bool]] = None,
        **kwargs,
    ) -> None:
        if qsa_index_dim <= 0:
            raise ValueError(f"qsa_index_dim must be positive, got {qsa_index_dim}")
        self.qsa_index_dim = int(qsa_index_dim)
        # Keep INDEX_KEY out of the main K/V coalesced pool. At TP4 the
        # logical 128-wide BF16 index key is exactly the same size as each
        # local K/V role. A one-element storage pad gives V2 an unambiguous
        # pool/stride while the modeling view below remains 128-wide.
        self.qsa_index_storage_dim = self.qsa_index_dim + 1
        self.qsa_sparse_layer_ids = (
            [layer_idx for layer_idx, enabled in enumerate(layer_mask) if enabled]
            if layer_mask is not None
            else []
        )
        self.qsa_position_layer_id: Optional[int] = None
        super().__init__(*args, layer_mask=layer_mask, **kwargs)

    def _extra_buffers_per_layer(self, *, tokens_per_block: int):
        elem_bytes = torch.tensor([], dtype=torch.bfloat16).element_size()
        index_size = self.qsa_index_storage_dim * elem_bytes * tokens_per_block
        position_elem_bytes = 4
        local_sparse_layers = [
            layer_id for layer_id in self.qsa_sparse_layer_ids if layer_id in self.layer_offsets
        ]
        self.qsa_position_layer_id = local_sparse_layers[0] if local_sparse_layers else None
        result = {
            self.layer_offsets[layer_id]: [BufferConfig(role=Role.INDEX_KEY, size=index_size)]
            for layer_id in local_sparse_layers
        }
        for layer_id in local_sparse_layers:
            local_idx = self.layer_offsets[layer_id]
            result[local_idx].append(
                BufferConfig(
                    role=QSA_INDEX_POSITION,
                    size=3 * position_elem_bytes * tokens_per_block,
                )
            )
        return result

    def get_index_k_buffer(
        self,
        layer_idx: int,
        kv_layout: str = "NHD",
    ) -> Optional[torch.Tensor]:
        storage = super().get_index_k_buffer(
            layer_idx,
            num_heads=1,
            head_dim=self.qsa_index_storage_dim,
            dtype=torch.bfloat16,
            kv_layout=kv_layout,
        )
        if storage is None:
            return None
        return storage[..., : self.qsa_index_dim]

    def get_buffers(
        self,
        layer_idx: int,
        kv_layout: str = "NHD",
    ) -> Optional[torch.Tensor]:
        """Return a per-lifecycle-slot K/V view for exact sparse attention.

        V2 may coalesce the K/V pairs of several attention layers.  QSA's
        side table stores lifecycle slot IDs, so preserve the complete physical
        slot stride and select this layer's adjacent K/V roles. INDEX_KEY uses
        padded storage and therefore remains outside the K/V coalesced group.
        """
        if kv_layout not in ("NHD", "HND"):
            raise ValueError(f"Unsupported kv_layout: {kv_layout}")
        if self.kv_cache_type == CacheTypeCpp.SELFKONLY:
            raise NotImplementedError("QSA sparse attention requires both K and V cache buffers")

        layer_offset = self.layer_offsets[layer_idx]
        addr_key = self.impl.get_mem_pool_base_address(layer_offset, Role.KEY)
        addr_value = self.impl.get_mem_pool_base_address(layer_offset, Role.VALUE)
        key_stride = self.impl.get_page_stride(layer_offset, Role.KEY)
        value_stride = self.impl.get_page_stride(layer_offset, Role.VALUE)
        if addr_key + value_stride != addr_value or key_stride != value_stride:
            raise RuntimeError(
                "QSA K/V buffers are not adjacent equal-sized V2 roles: "
                f"layer={layer_idx}, key_stride={key_stride}, "
                f"value_stride={value_stride}"
            )

        converter = self.impl.get_page_index_converter(layer_offset, Role.KEY)
        scale = int(converter.scale)
        layer_offset_pages = int(converter.layer_offset)
        page_upper = self.impl.get_page_index_upper_bound(layer_offset, Role.KEY)
        num_slots_total = page_upper + layer_offset_pages
        if num_slots_total % scale != 0:
            raise RuntimeError(
                "QSA K/V page mapping is inconsistent: "
                f"{num_slots_total=} is not divisible by {scale=}"
            )
        num_slots = num_slots_total // scale

        element_per_container = 1
        dtype = self.dtype
        if dtype == DataType.NVFP4:
            element_per_container = 2
            torch_dtype = torch.int8
        else:
            torch_dtype = binding_to_torch_dtype(dtype)
        head_dim = self.head_dim_per_layer[layer_offset]
        num_heads = self.num_kv_heads_per_layer[layer_offset]
        if kv_layout == "NHD":
            page_shape = [self.tokens_per_block, num_heads, head_dim // element_per_container]
        else:
            page_shape = [num_heads, self.tokens_per_block, head_dim // element_per_container]
        full_view = convert_to_torch_tensor(
            TensorWrapper(
                addr_key,
                torch_dtype,
                [num_slots, scale, *page_shape],
            )
        )
        return full_view[:, :2]

    def get_qsa_position_buffer(self) -> Optional[torch.Tensor]:
        """Return ``[slots, tokens_per_block, 3]`` position coordinates."""
        if self.qsa_position_layer_id is None:
            return None
        local_idx = self.layer_offsets[self.qsa_position_layer_id]
        addr = self.impl.get_mem_pool_base_address(
            local_idx,
            QSA_INDEX_POSITION,
            PageIndexMode.SHARED,
        )
        page_stride = self.impl.get_page_stride(local_idx, QSA_INDEX_POSITION)
        expected_stride = 3 * 4 * self.tokens_per_block
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
        if num_pages_with_offset % scale != 0:
            raise RuntimeError("QSA position-cache page mapping is inconsistent")
        num_slots = num_pages_with_offset // scale
        full_view = convert_to_torch_tensor(
            TensorWrapper(
                addr,
                torch.int32,
                [num_slots, scale, self.tokens_per_block, 3],
            )
        )
        return full_view[:, 0]

    def get_qsa_slot_block_indices(
        self,
        request_ids: List[int],
    ) -> List[List[int]]:
        """Return lifecycle slot IDs shared by main and sparse side buffers."""
        if self.qsa_position_layer_id is None:
            raise RuntimeError("QSA cache manager has no local sparse layer")
        local_idx = self.layer_offsets[self.qsa_position_layer_id]
        pool_id = self.layer_to_pool_mapping_dict[local_idx]
        result = []
        for request_id in request_ids:
            cache = self.kv_cache_map[request_id]
            pages = cache.get_base_page_indices(pool_id)[: cache.num_blocks]
            result.append([0 if page == BAD_PAGE_INDEX else int(page) for page in pages])
        return result


__all__ = ["QSAMambaHybridCacheManagerV2"]
