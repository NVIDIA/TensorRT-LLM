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
"""Kimi KDA hybrid cache manager with a DSA indexer K-cache on its MLA layers."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Optional

import torch

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import (
    _get_indexer_k_cache_bytes_per_token,
    _get_indexer_k_cache_size_per_token,
    derive_indexer_k_cache_layer_mask,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.params import DSAParams
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, DataRole, PageIndexMode

from .cache_manager import KimiK3HybridCacheManager

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig


def _uses_nvfp4_ds_mla(kv_cache_config) -> bool:
    """Whether the cache config selects the inline NVFP4 latent records (``nvfp4_ds_mla``)."""
    return getattr(kv_cache_config, "dtype", "auto") == "nvfp4_ds_mla"


def _nvfp4_ds_mla_head_dim(head_dim, dtype) -> int:
    """Storage head_dim of the ``nvfp4_ds_mla`` latent pool: one record of one-byte elements per token."""
    from ...attention.backends.sparse.nvfp4_ds_mla_kv import NOPE_DIM, ROPE_DIM, TOKEN_BYTES

    if head_dim is not None and int(head_dim) != NOPE_DIM + ROPE_DIM:
        raise ValueError(
            f"nvfp4_ds_mla requires kv_lora_rank + qk_rope_head_dim = {NOPE_DIM + ROPE_DIM}, got {head_dim}"
        )
    if dtype is not None and dtype != DataType.FP8:
        raise ValueError(f"nvfp4_ds_mla stores its records in a one-byte (FP8) pool, got {dtype}")
    return TOKEN_BYTES


class KimiK3DSAHybridCacheManager(KimiK3HybridCacheManager):
    """Kimi K3 KDA/MLA cache plus a DSA indexer K-cache (Kimi K3.1).

    Kimi K3.1 adds a DeepSeek sparse-attention indexer to its MLA layers. The
    indexer keys live in one extra ``Role.INDEX_KEY`` buffer per indexer layer,
    so they share allocation, reuse and release with the latent KV pages. The
    DSA accessors mirror :class:`DSACacheManagerV2`, except that local layer
    indices also cover the KDA layers, which own neither a K page nor an
    indexer buffer.
    """

    # Read by ``is_dsa_cache_manager``: the DSA backend and metadata may use
    # this manager's indexer K-cache and primary-pool page mapping.
    provides_dsa_indexer_k_cache = True

    @override
    def __init__(
        self,
        *args,
        sparse_attention_config: "SparseAttentionConfig",
        pretrained_config=None,
        **kwargs,
    ) -> None:
        """Build the hybrid manager with an indexer buffer per DSA layer.

        Args:
            sparse_attention_config: DSA sparse-attention config of the model.
            pretrained_config: Text config; resolves indexer sharing per layer.
            **kwargs: ``KimiK3HybridCacheManager`` arguments; ``layer_mask``
                marks the MLA layers.
        """
        if sparse_attention_config is None:
            raise ValueError("sparse_attention_config is required for the Kimi DSA cache")
        sparse_params = sparse_attention_config.to_sparse_params(
            pretrained_config=pretrained_config
        )
        if not isinstance(sparse_params, DSAParams):
            raise ValueError("The Kimi DSA cache requires DSA sparse parameters")
        layer_mask = kwargs.get("layer_mask")
        if layer_mask is None:
            raise ValueError("The Kimi DSA cache needs layer_mask to place the indexer buffers")

        # Read by _extra_buffers_per_layer during the base construction.
        self.quant_block_size = 128
        self.index_head_dim = sparse_params.index_head_dim
        self.indexer_k_dtype = sparse_params.indexer_k_dtype
        self.use_fp4 = self.indexer_k_dtype == "fp4"
        self.use_fp8_ds_mla = False
        self.mla_kv_cache_residual_dim = 0
        # kv_cache_config.dtype "nvfp4_ds_mla": every MLA token is one 352-byte inline NVFP4 record
        # (sparse/nvfp4_ds_mla_kv.py) in a one-byte (FP8) container pool.
        kv_cache_config = args[9] if len(args) > 9 else kwargs.get("kv_cache_config")
        self.use_nvfp4_ds_mla = _uses_nvfp4_ds_mla(kv_cache_config)
        if self.use_nvfp4_ds_mla:
            kwargs["head_dim"] = _nvfp4_ds_mla_head_dim(kwargs.get("head_dim"), kwargs.get("dtype"))
        full_indexer_mask = derive_indexer_k_cache_layer_mask(
            sparse_attention_config, pretrained_config, len(layer_mask)
        )
        self._indexer_layer_ids = frozenset(
            layer_idx
            for layer_idx, (is_attention, has_indexer) in enumerate(
                zip(layer_mask, full_indexer_mask)
            )
            if is_attention and has_indexer
        )
        self._unique_primary_pool: Optional[torch.Tensor] = None

        super().__init__(*args, **kwargs)

        # A unified standalone drafter (DSpark) registers its layers as model layers; they own
        # neither a coalesced MLA K page nor an indexer buffer, so the DSA views skip them.
        self._attention_local_layers = [
            local_layer_idx
            for local_layer_idx in range(self.num_local_layers)
            if not self._is_standalone_draft_layer(local_layer_idx)
            and not self._is_local_mamba_layer(local_layer_idx)
        ]
        if not self._attention_local_layers:
            raise ValueError("DSA requires at least one local MLA layer")
        self._primary_pool_page_index_params = {}
        for local_layer_idx in self._attention_local_layers:
            converter = self.impl.get_page_index_converter(local_layer_idx, Role.KEY)
            self._primary_pool_page_index_params[local_layer_idx] = (
                int(converter.scale),
                int(converter.layer_offset),
            )
        indexer_local_layers = [
            local_layer_idx
            for local_layer_idx, has_indexer in enumerate(self.indexer_k_cache_local_layer_mask)
            if has_indexer
        ]
        indexer_converters = [
            self.impl.get_page_index_converter(local_layer_idx, Role.INDEX_KEY)
            for local_layer_idx in indexer_local_layers
        ]
        if any(int(converter.expansion) != 1 for converter in indexer_converters):
            raise ValueError("DSA indexer K-cache does not support page expansion")
        # Each cache view starts at its SHARED base address, so the converter's
        # layer offset is already represented by the base pointer.
        self.indexer_k_cache_page_scale = (
            int(indexer_converters[0].scale) if indexer_converters else 1
        )
        if any(
            int(converter.scale) != self.indexer_k_cache_page_scale
            for converter in indexer_converters[1:]
        ):
            raise ValueError("DSA requires a uniform shared INDEX_KEY page mapping across layers")
        self._attention_pool_id = self.layer_to_pool_mapping_dict[self._attention_local_layers[0]]
        self._num_attention_slots = self._count_attention_slots()
        self.indexer_k_cache_pool_per_layer = {
            local_layer_idx: self._get_indexer_k_cache_pool_data(local_layer_idx)
            for local_layer_idx in indexer_local_layers
        }

    @property
    def indexer_k_cache_local_layer_mask(self) -> list[bool]:
        """Indexer ownership of each local layer; KDA layers own none."""
        return [layer_idx in self._indexer_layer_ids for layer_idx in self.pp_layers]

    @override
    def _extra_buffers_per_layer(self, *, tokens_per_block: int) -> dict[int, list[BufferConfig]]:
        """One ``Role.INDEX_KEY`` buffer per local indexer layer."""
        return {
            self.layer_offsets[layer_idx]: [
                BufferConfig(
                    role=Role.INDEX_KEY,
                    size=self._indexer_bytes_per_token() * tokens_per_block,
                )
            ]
            for layer_idx in sorted(self._indexer_layer_ids)
            if layer_idx in self.layer_offsets
        }

    def _indexer_bytes_per_token(self) -> int:
        return _get_indexer_k_cache_bytes_per_token(
            self.index_head_dim, self.quant_block_size, self.indexer_k_dtype
        )

    @override
    def get_layer_bytes_per_token(self, local_layer_idx: int, data_role: DataRole) -> int:
        index_bytes = (
            self._indexer_bytes_per_token()
            if local_layer_idx < len(self.pp_layers)
            and self.pp_layers[local_layer_idx] in self._indexer_layer_ids
            else 0
        )
        if data_role == Role.INDEX_KEY:
            return index_bytes
        cache_bytes = super().get_layer_bytes_per_token(local_layer_idx, data_role)
        return cache_bytes + index_bytes if data_role == Role.ALL else cache_bytes

    @override
    def _attention_cache_bytes_per_token(self) -> int:
        if self._draft_layer_sizes:
            # The base sum also covers the standalone draft layers with their page expansion;
            # get_layer_bytes_per_token above still adds the index K bytes of the MLA layers.
            return super()._attention_cache_bytes_per_token()
        return sum(
            self.get_layer_bytes_per_token(local_layer_idx, Role.ALL)
            for local_layer_idx in range(self.num_local_layers)
        )

    @classmethod
    @override
    def get_cache_size_per_token(
        cls,
        model_config,
        mapping: Mapping,
        *,
        max_batch_size: int,
        kv_cache_config: KvCacheConfig,
        tokens_per_block: int = 32,
        max_seq_len: Optional[int] = None,
        **kwargs,
    ) -> tuple[int, int]:
        slope, fixed_cost = super().get_cache_size_per_token(
            model_config,
            mapping,
            max_batch_size=max_batch_size,
            kv_cache_config=kv_cache_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            **kwargs,
        )
        index_bytes_per_token = _get_indexer_k_cache_size_per_token(model_config, mapping)
        if _uses_nvfp4_ds_mla(kv_cache_config):
            # The base estimate sizes the MLA latent at the one-byte container's head_dim
            # (kv_lora_rank + qk_rope_head_dim); a record is 352 bytes.
            from ...attention.backends.sparse.nvfp4_ds_mla_kv import TOKEN_BYTES
            from ...pyexecutor.kv_cache.mamba_cache_manager.common import (
                _get_local_mamba_cache_layout,
            )

            config = model_config.pretrained_config
            head_dim = int(config.kv_lora_rank) + int(config.qk_rope_head_dim)
            # Target MLA layers of this rank only: draft layers keep their own storage dtype.
            _, _, local_mla_layers = _get_local_mamba_cache_layout(
                model_config,
                mapping,
                spec_config=kwargs.get("spec_config"),
                use_separate_draft_kv_cache=True,
            )
            slope -= local_mla_layers * (head_dim - TOKEN_BYTES)
        state_config = kv_cache_config.mamba_state_config
        seq_limit = max_seq_len if max_seq_len is not None else float("inf")
        has_snapshots = kv_cache_config.enable_block_reuse and (
            0 < state_config.periodic_snapshot_interval <= seq_limit
            or any(
                offset <= seq_limit
                for offset in state_config.additional_snapshot_offsets_from_start
            )
            or any(
                offset < seq_limit for offset in state_config.additional_snapshot_offsets_from_end
            )
        )
        # V2 reserves one partial attention page per resident lineage when
        # snapshots are reachable; include the indexer section of those pages.
        if has_snapshots:
            fixed_cost += (
                max_batch_size * mapping.pp_size * tokens_per_block * index_bytes_per_token
            )
        return slope + index_bytes_per_token, fixed_cost

    def get_primary_pool_page_index_params(self, local_layer_idx: int) -> tuple[int, int]:
        """Return the V2 page scale and layer offset of an MLA layer's K page."""
        return self._primary_pool_page_index_params[local_layer_idx]

    def _count_attention_slots(self) -> int:
        """Physical slot count of the MLA K pool (not its V2 page bound)."""
        local_layer_idx = self._attention_local_layers[0]
        converter = self.impl.get_page_index_converter(local_layer_idx, Role.KEY)
        page_upper = self.impl.get_page_index_upper_bound(local_layer_idx, Role.KEY)
        expansion = int(converter.expansion)
        assert page_upper % expansion == 0
        num_pages_with_offset = page_upper // expansion + int(converter.layer_offset)
        scale = int(converter.scale)
        assert num_pages_with_offset % scale == 0
        return num_pages_with_offset // scale

    def _get_indexer_k_cache_pool_data(self, local_layer_idx: int) -> torch.Tensor:
        """Return a contiguous shared-page view for one indexer K-cache."""
        address = self.impl.get_mem_pool_base_address(
            local_layer_idx, Role.INDEX_KEY, PageIndexMode.SHARED
        )
        page_upper = self.impl.get_page_index_upper_bound(local_layer_idx, Role.INDEX_KEY)
        flat_page_size = self.tokens_per_block * self._indexer_bytes_per_token()
        return convert_to_torch_tensor(
            TensorWrapper(address, torch.uint8, [page_upper, flat_page_size])
        )

    def get_indexer_k_cache_buffers(self, layer_idx: int) -> torch.Tensor:
        """Return the page-indexed indexer K-cache view for a global layer."""
        local_layer_idx = self.layer_offsets[layer_idx]
        pool = self.indexer_k_cache_pool_per_layer.get(local_layer_idx)
        assert pool is not None, f"Layer {layer_idx} owns no DSA indexer K-cache."
        return pool.view(pool.shape[0], self.tokens_per_block, 1, self._indexer_bytes_per_token())

    def get_pool_block_indices(
        self,
        num_seqs: int,
        *,
        request_ids: Optional[list[int]] = None,
        num_contexts: int = 0,
        beam_width: int = 1,
    ) -> torch.Tensor:
        """Read the MLA pool's V2 stable slots in current-batch order."""
        if request_ids is None:
            raise ValueError("KimiK3DSAHybridCacheManager requires request_ids to map slots")
        if len(request_ids) != num_seqs:
            raise ValueError(f"Expected {num_seqs} request IDs, got {len(request_ids)}")
        copy_idx = self.index_mapper.get_copy_index(list(request_ids), num_contexts, beam_width)
        copy_idx = copy_idx.to(device="cpu", dtype=torch.long)
        num_cols = self._batch_block_columns(request_ids)
        block_indices = self.host_kv_cache_block_offsets[
            self._attention_pool_id, copy_idx, 0, :num_cols
        ]
        return block_indices.clamp(min=0, max=self._num_attention_slots - 1).to(torch.int32)

    def _batch_block_columns(self, request_ids: list[int]) -> int:
        """Block-table columns that can hold a page of this batch.

        A request's pages cover its KV capacity (every token any kernel reads, draft slots
        included), so columns past the largest capacity hold no page of the batch. Reading only
        those keeps the per-step host gather proportional to the batch's context instead of
        ``max_seq_len``.
        """
        width = self.host_kv_cache_block_offsets.shape[-1]
        capacity = 0
        for request_id in request_ids:
            kv_cache = self.kv_cache_map.get(request_id)
            if kv_cache is None:
                return width
            capacity = max(capacity, int(kv_cache.capacity))
        return min(width, max(1, -(-capacity // self.tokens_per_block)))

    def get_unique_primary_pool(self) -> torch.Tensor:
        """Return the MLA K pool as ``[slots, mla_layers, 1, elements]``."""
        if self._unique_primary_pool is not None:
            return self._unique_primary_pool
        if self.kv_factor != 1:
            raise ValueError("DSA requires a SELFKONLY KV cache")
        attention_layers = self._attention_local_layers
        first_layer = attention_layers[0]
        first_converter = self.impl.get_page_index_converter(first_layer, Role.KEY)
        if int(first_converter.expansion) != 1:
            raise ValueError("DSA MLA K-cache does not support page expansion")
        if int(first_converter.scale) != len(attention_layers):
            raise ValueError("DSA requires one uniformly coalesced K page per local MLA layer")
        page_stride = self.impl.get_page_stride(first_layer, Role.KEY)
        base_address = self.impl.get_mem_pool_base_address(
            first_layer, Role.KEY, PageIndexMode.SHARED
        )
        for ordinal, local_layer_idx in enumerate(attention_layers):
            converter = self.impl.get_page_index_converter(local_layer_idx, Role.KEY)
            if int(converter.scale) != int(first_converter.scale) or int(converter.expansion) != 1:
                raise ValueError("DSA requires a uniform page-index mapping across MLA layers")
            if int(converter.layer_offset) != ordinal:
                raise ValueError("DSA requires K pages to follow local MLA-layer order")
            address = self.impl.get_mem_pool_base_address(
                local_layer_idx, Role.KEY, PageIndexMode.SHARED
            )
            if int(address) != int(base_address + ordinal * page_stride):
                raise ValueError("DSA requires contiguous per-layer K pages in each slot")

        head_dim = self.head_dim_per_layer[first_layer]
        num_heads = self.num_kv_heads_per_layer[first_layer]
        # NVFP4 codes are packed two per byte; their E4M3 scales live in a
        # separate role. DSA gathers both through the manager's pool pointers.
        packed = self.dtype == DataType.NVFP4
        elements_per_layer = self.tokens_per_block * num_heads * head_dim // (2 if packed else 1)
        shape = [self._num_attention_slots, len(attention_layers), 1, elements_per_layer]
        self._unique_primary_pool = convert_to_torch_tensor(
            TensorWrapper(base_address, torch.int8 if packed else self.dtype, shape)
        )
        return self._unique_primary_pool

    @override
    def shutdown(self) -> None:
        self.indexer_k_cache_pool_per_layer = {}
        self._unique_primary_pool = None
        super().shutdown()
