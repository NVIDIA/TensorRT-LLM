# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V2 hybrid cache manager for QSA sparse attention."""

from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._utils import TensorWrapper, binding_to_torch_dtype, convert_to_torch_tensor
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, PageIndexMode
from tensorrt_llm.runtime.kv_cache_manager_v2._config import DataRole

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

# Per-token RoPE/mRoPE coordinates used when raw index keys are compressed.
# They are request state shared by all local QSA layers.
QSA_INDEX_POSITION = DataRole("qsa_index_position")


class QSAMambaHybridCacheManagerV2(MambaHybridCacheManagerV2):
    """Hybrid GDN/KV manager with lifecycle-coupled sparse side buffers.

    QSA adds a per-layer index-K cache and request-wide position coordinates.
    These buffers share the main K/V allocation, eviction, and prefix-reuse
    lifecycle. The parent hybrid manager owns GDN/Mamba recurrent state.

    Index geometry is resolved from the same serving/checkpoint configuration
    as the indexer. A local fallback could otherwise give the indexer and its
    paged cache different shapes or page strides.
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
        super().__init__(*args, layer_mask=layer_mask, **kwargs)

    def _extra_buffers_per_layer(
        self,
        *,
        tokens_per_block: int,
    ) -> dict[int, list[BufferConfig]]:
        """Register per-layer index K and lifecycle-aligned position pages."""
        index_size = (
            self.qsa_index_kv_heads * self.qsa_index_dim * _INDEX_K_ELEMENT_BYTES * tokens_per_block
        )
        local_sparse_layers = [
            layer_id for layer_id in self.qsa_sparse_layer_ids if layer_id in self.layer_offsets
        ]
        # Coordinates are request-wide, so all local indexers use one view.
        # Register the position role once; duplicating it on every sparse layer
        # wastes one three-axis int32 page per layer without adding state.
        self.qsa_position_layer_id = next(iter(local_sparse_layers), None)
        result = {
            self.layer_offsets[layer_id]: [BufferConfig(role=Role.INDEX_KEY, size=index_size)]
            for layer_id in local_sparse_layers
        }
        if self.qsa_position_layer_id is not None:
            local_idx = self.layer_offsets[self.qsa_position_layer_id]
            result[local_idx].append(
                BufferConfig(
                    role=QSA_INDEX_POSITION,
                    size=(
                        QSA_POSITION_COORDINATE_AXES * _POSITION_ELEMENT_BYTES * tokens_per_block
                    ),
                )
            )
        return result

    def get_index_k_buffer(
        self,
        layer_idx: int,
        kv_layout: str = "NHD",
    ) -> Optional[torch.Tensor]:
        """Return the index-K view using the indexer's resolved geometry."""
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
        """
        if self.dtype not in QSA_SPARSE_KV_CACHE_DTYPES:
            # The parent owns packed data and scale-page layouts such as
            # NVFP4. QSA returns to the regular backend for these formats.
            return super().get_buffers(layer_idx, kv_layout)
        if kv_layout not in ("NHD", "HND"):
            raise ValueError(f"Unsupported kv_layout: {kv_layout}")
        if layer_idx not in self.layer_offsets:
            return None
        if self.kv_cache_type == CacheTypeCpp.SELFKONLY:
            raise NotImplementedError("QSA sparse attention requires both K and V cache buffers")

        layer_offset = self.layer_offsets[layer_idx]
        addr_key = self.impl.get_mem_pool_base_address(layer_offset, Role.KEY, PageIndexMode.SHARED)
        addr_value = self.impl.get_mem_pool_base_address(
            layer_offset, Role.VALUE, PageIndexMode.SHARED
        )
        key_stride = self.impl.get_page_stride(layer_offset, Role.KEY)
        value_stride = self.impl.get_page_stride(layer_offset, Role.VALUE)
        if addr_key + key_stride != addr_value or key_stride != value_stride:
            raise RuntimeError(
                "QSA K/V buffers are not adjacent equal-sized V2 roles: "
                f"layer={layer_idx}, key_stride={key_stride}, "
                f"value_stride={value_stride}"
            )

        converter = self.impl.get_page_index_converter(layer_offset, Role.KEY)
        scale = int(converter.scale)
        if scale < QSA_MAIN_KV_ROLES:
            raise RuntimeError(
                f"QSA K/V roles do not share one lifecycle slot: layer={layer_idx}, {scale=}"
            )
        if int(converter.expansion) != _IDENTITY_PAGE_INDEX_EXPANSION:
            raise RuntimeError("QSA does not support expanded V2 page indices")
        layer_offset_pages = int(converter.layer_offset)
        page_upper = self.impl.get_page_index_upper_bound(layer_offset, Role.KEY)
        num_slots_total = page_upper + layer_offset_pages
        if num_slots_total % scale != 0:
            raise RuntimeError(
                "QSA K/V page mapping is inconsistent: "
                f"{num_slots_total=} is not divisible by {scale=}"
            )
        num_slots = num_slots_total // scale

        torch_dtype = binding_to_torch_dtype(self.dtype)
        head_dim = self.head_dim_per_layer[layer_offset]
        num_heads = self.num_kv_heads_per_layer[layer_offset]
        if kv_layout == "NHD":
            page_shape = [self.tokens_per_block, num_heads, head_dim]
        else:
            page_shape = [num_heads, self.tokens_per_block, head_dim]
        full_view = convert_to_torch_tensor(
            TensorWrapper(
                addr_key,
                torch_dtype,
                [num_slots, scale, *page_shape],
            )
        )
        return full_view[:, :QSA_MAIN_KV_ROLES]

    @torch.compiler.disable
    def get_qsa_position_buffer(self) -> Optional[torch.Tensor]:
        """Return per-token three-axis RoPE/mRoPE position coordinates."""
        if self.qsa_position_layer_id is None:
            return None
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


__all__ = ["QSAMambaHybridCacheManagerV2"]
