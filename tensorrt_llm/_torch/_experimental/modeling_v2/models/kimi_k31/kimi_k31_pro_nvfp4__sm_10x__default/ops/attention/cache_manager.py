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

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, DataRole, PageIndexMode

from ....ops.attention.dsa_cache import derive_indexer_k_cache_layer_mask
from ....ops.attention.dsa_params import DSAParams, k31_dsa_params
from ..ssm.hybrid_cache import KimiK3HybridCacheManager
from .cache_policy import get_kimi_k31_fp8_kv_layer_ids
from .cache_utils import _get_indexer_k_cache_bytes_per_token, _get_indexer_k_cache_size_per_token

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig


class KimiK3DSAHybridCacheManager(KimiK3HybridCacheManager):
    """Kimi K3 KDA/MLA cache plus a DSA indexer K-cache (Kimi K3.1).

    Kimi K3.1 adds a DeepSeek sparse-attention indexer to its MLA layers. The
    indexer keys live in one extra ``Role.INDEX_KEY`` buffer per indexer layer,
    so they share allocation, reuse and release with the latent KV pages. The
    DSA accessors mirror the engine's V2 DSA cache manager, except that local layer
    indices also cover the KDA layers, which own neither a K page nor an
    indexer buffer.
    """

    # Read by ``is_dsa_cache_manager``: the DSA backend and metadata may use
    # this manager's indexer K-cache and primary-pool page mapping.
    provides_dsa_indexer_k_cache = True

    @classmethod
    def extra_init_kwargs(cls, model_config) -> dict:
        """Constructor arguments the engine adds for this manager: the DSA config and the text
        config (``_create_kv_cache_manager``)."""
        pretrained_config = model_config.pretrained_config
        return {
            "sparse_attention_config": model_config.sparse_attention_config,
            "pretrained_config": getattr(pretrained_config, "text_config", None)
            or pretrained_config,
        }

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
        sparse_params = k31_dsa_params(
            sparse_attention_config.to_sparse_params(pretrained_config=pretrained_config),
            pretrained_config,
        )
        if not isinstance(sparse_params, DSAParams):
            raise ValueError("The Kimi DSA cache requires DSA sparse parameters")
        if sparse_params.indexer_k_dtype not in ("fp8", "fp4"):
            raise ValueError(
                "The Kimi K3.1 DSA cache stores FP8 indexer keys; got "
                f"indexer_k_dtype={sparse_params.indexer_k_dtype!r}"
            )
        layer_mask = kwargs.get("layer_mask")
        if layer_mask is None:
            raise ValueError("The Kimi DSA cache needs layer_mask to place the indexer buffers")

        # Read by _extra_buffers_per_layer during the base construction.
        self.quant_block_size = 128
        self.index_head_dim = sparse_params.index_head_dim
        self.indexer_k_dtype = sparse_params.indexer_k_dtype
        self.use_fp4 = sparse_params.indexer_k_dtype == "fp4"
        self.use_fp8_ds_mla = False
        self.mla_kv_cache_residual_dim = 0
        self._fp8_mla_global_layer_ids = get_kimi_k31_fp8_kv_layer_ids(
            pretrained_config, nvfp4_enabled=kwargs.get("dtype") == DataType.NVFP4
        )
        if self._fp8_mla_global_layer_ids:
            if kwargs.get("dtype") != DataType.NVFP4:
                raise ValueError("Kimi K3 FP8 KV fallback requires NVFP4 cache storage.")
            logger.info(
                "Kimi K3 DSA layers using FP8 KV-cache fallback: "
                f"{sorted(self._fp8_mla_global_layer_ids)}."
            )
        self._primary_pools: dict[int, torch.Tensor] = {}
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

        # Draft layers share admission and transfer with the target, but own no DSA indexer pages.
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
            if self._get_layer_dtype_by_local_id(local_layer_idx) == DataType.NVFP4:
                scale_converter = self.impl.get_page_index_converter(
                    local_layer_idx, Role.KEY_BLOCK_SCALE
                )
                if (converter.scale, converter.layer_offset, converter.expansion) != (
                    scale_converter.scale,
                    scale_converter.layer_offset,
                    scale_converter.expansion,
                ):
                    raise ValueError("NVFP4 DSA data and scale pages must use the same indices.")
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
        if any(
            self.layer_to_pool_mapping_dict[layer] != self._attention_pool_id
            for layer in self._attention_local_layers
        ):
            raise ValueError("Kimi DSA MLA layers must share one request-slot lifecycle.")
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
    def _get_layer_dtype_by_local_id(self, local_layer_id: int) -> DataType:
        if (
            local_layer_id < len(self.pp_layers)
            and self.pp_layers[local_layer_id] in self._fp8_mla_global_layer_ids
        ):
            return DataType.FP8
        return super()._get_layer_dtype_by_local_id(local_layer_id)

    def get_kv_cache_dtype(self, layer_idx: int | None = None) -> DataType:
        """Return the storage dtype for a global MLA layer."""
        if layer_idx is None:
            return self.dtype
        return self._get_layer_dtype_by_local_id(self.layer_offsets[layer_idx])

    @override
    def _get_buffer_roles_for_layer(self, local_layer_idx: int) -> list[DataRole]:
        if (
            local_layer_idx < len(self.pp_layers)
            and self.pp_layers[local_layer_idx] in self._fp8_mla_global_layer_ids
        ):
            return [Role.KEY]
        return super()._get_buffer_roles_for_layer(local_layer_idx)

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
        if (
            local_layer_idx < len(self.pp_layers)
            and self.pp_layers[local_layer_idx] in self._fp8_mla_global_layer_ids
        ):
            cache_bytes = (
                self.num_kv_heads_per_layer[local_layer_idx]
                * self.head_dim_per_layer[local_layer_idx]
                if data_role in (Role.KEY, Role.ALL)
                else 0
            )
        else:
            cache_bytes = super().get_layer_bytes_per_token(local_layer_idx, data_role)
        return cache_bytes + index_bytes if data_role == Role.ALL else cache_bytes

    @override
    def _attention_cache_bytes_per_token(self) -> int:
        if self._draft_layer_sizes:
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
        config = model_config.pretrained_config
        config = getattr(config, "text_config", None) or config
        quant_config = model_config.quant_config
        fp8_layers = get_kimi_k31_fp8_kv_layer_ids(
            config,
            nvfp4_enabled=quant_config is not None and quant_config.quant_mode.has_fp4_kv_cache(),
        )
        local_fp8_layers = fp8_layers.intersection(mapping.pp_layers(config.num_hidden_layers))
        head_dim = int(config.kv_lora_rank + config.qk_rope_head_dim)
        # Kimi DSA uses a latent-only NVFP4 cache without the dense MLA HP tail.
        fallback_bytes_per_token = len(local_fp8_layers) * (
            head_dim - head_dim // 2 - head_dim // 16
        )
        extra_bytes_per_token = index_bytes_per_token + fallback_bytes_per_token
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
        # snapshots are reachable; include their indexer and FP8 fallback bytes.
        if has_snapshots:
            fixed_cost += (
                max_batch_size * mapping.pp_size * tokens_per_block * extra_bytes_per_token
            )
        return slope + extra_bytes_per_token, fixed_cost

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

    def get_allocated_block_counts(self, request_ids: list[int]) -> torch.Tensor:
        """Pages each request of the batch holds (its KV capacity in blocks), in batch order.

        A request this manager does not hold counts the whole block-table row, as in
        ``_batch_block_columns``.
        """
        width = self.host_kv_cache_block_offsets.shape[-1]
        counts = []
        for request_id in request_ids:
            kv_cache = self.kv_cache_map.get(request_id)
            if kv_cache is None:
                counts.append(width)
            else:
                counts.append(min(width, -(-int(kv_cache.capacity) // self.tokens_per_block)))
        return torch.tensor(counts, dtype=torch.int32)

    def get_primary_pool(self, local_layer_idx: int) -> torch.Tensor:
        """Return this layer's physical K pool as ``[slots, pages, 1, elements]``.

        Different storage dtypes have distinct physical pools in the same
        lifecycle. Slot IDs remain shared with the indexer; page scales and
        offsets describe the layers coalesced in each physical pool.
        """
        if self.kv_factor != 1:
            raise ValueError("DSA requires a SELFKONLY KV cache")
        converter = self.impl.get_page_index_converter(local_layer_idx, Role.KEY)
        if int(converter.expansion) != 1:
            raise ValueError("DSA MLA K-cache does not support page expansion")
        base_address = self.impl.get_mem_pool_base_address(
            local_layer_idx, Role.KEY, PageIndexMode.PER_LAYER
        )
        if base_address in self._primary_pools:
            return self._primary_pools[base_address]
        head_dim = self.head_dim_per_layer[local_layer_idx]
        num_heads = self.num_kv_heads_per_layer[local_layer_idx]
        dtype = self._get_layer_dtype_by_local_id(local_layer_idx)
        # NVFP4 codes are packed two per byte; their E4M3 scales live in a
        # separate role. DSA gathers both through the manager's pool pointers.
        packed = dtype == DataType.NVFP4
        elements_per_layer = self.tokens_per_block * num_heads * head_dim // (2 if packed else 1)
        shape = [self._num_attention_slots, int(converter.scale), 1, elements_per_layer]
        pool = convert_to_torch_tensor(
            TensorWrapper(base_address, torch.int8 if packed else dtype, shape)
        )
        self._primary_pools[base_address] = pool
        return pool

    def get_unique_primary_pool(self) -> torch.Tensor:
        """Return the primary pool when all MLA layers share one storage layout."""
        pools = [self.get_primary_pool(layer) for layer in self._attention_local_layers]
        if any(pool.data_ptr() != pools[0].data_ptr() for pool in pools[1:]):
            raise ValueError("Mixed Kimi DSA storage requires a layer-specific primary pool.")
        return pools[0]

    @override
    def shutdown(self) -> None:
        self.indexer_k_cache_pool_per_layer = {}
        self._unique_primary_pool = None
        self._primary_pools.clear()
        super().shutdown()


def select_k31_kv_cache_manager_cls(
    model_config, kv_cache_config, is_disagg: bool = False, cache_transceiver_config=None
):
    """The cache manager of a Kimi K3.1 engine with DSA: KDA recurrent state, MLA latent cache and
    one indexer K-cache buffer per MLA layer in one KV cache manager V2. None without DSA (the
    engine's routes then apply)."""
    sparse_config = model_config.sparse_attention_config
    if getattr(sparse_config, "algorithm", None) != "dsa":
        return None
    if kv_cache_config.use_kv_cache_manager_v2 is not True:
        raise ValueError(
            "Kimi K3.1 DSA requires use_kv_cache_manager_v2=True (the indexer K-cache is a V2 "
            "extra buffer)."
        )
    if is_disagg:
        # Only the Python NIXL transceiver moves KDA state and the INDEX_KEY buffers.
        from tensorrt_llm._torch.pyexecutor._util import _resolve_disagg_transceiver_route

        backend, runtime = _resolve_disagg_transceiver_route(cache_transceiver_config)
        if runtime != "PYTHON" or backend != "NIXL":
            raise ValueError(
                "Kimi K3.1 disaggregated serving requires cache_transceiver_config "
                "backend='NIXL' with transceiver_runtime='PYTHON' (got "
                f"backend={backend!r}, transceiver_runtime={runtime!r})."
            )
    if kv_cache_config.enable_block_reuse and kv_cache_config.enable_kv_pool_rebalance:
        raise ValueError(
            "V2 Mamba block reuse is not compatible with enable_kv_pool_rebalance because the "
            "rebalancer does not yet model retained recurrent-state snapshots."
        )
    return KimiK3DSAHybridCacheManager
