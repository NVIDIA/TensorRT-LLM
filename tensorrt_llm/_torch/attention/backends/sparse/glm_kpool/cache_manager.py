# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GLM hybrid cache manager for recurrent, latent KV, and indexer state."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.config_utils import (
    extract_mamba_kv_cache_params,
    unwrap_glm5_next_text_config,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import get_pp_layers
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, DataRole

from .params import glm_kpool_cache_row_dim


class Glm5NextCacheManager(MambaHybridCacheManagerV2):
    """Manage KDA state, latent KV and indexer buffers in one V2 lifecycle.

    KDA uses the inherited recurrent/conv pools; sparse MLA uses SELFKONLY pages.
    An extra BF16 INDEX_KEY buffer stores [k | gate | pool key] per sparse layer.
    Registering it through _extra_buffers_per_layer shares allocation, reuse,
    release and disaggregated transfer with the base manager.
    """

    def __init__(
        self,
        *args,
        sparse_layer_ids: Sequence[int] | None = None,
        index_state_dim: int = 0,
        **kwargs,
    ) -> None:
        """Build the hybrid manager with one indexer buffer per sparse layer.

        Args:
            sparse_layer_ids: Global layer ids that carry an indexer buffer.
                Defaults to every attention layer in ``layer_mask``: each GLM
                attention layer, including appended MTP layers, is sparse.
            index_state_dim: BF16 width of one indexer cache row, see
                :func:`glm_kpool_cache_row_dim`.
        """
        # Set before super().__init__: the base _build_base_config calls
        # _extra_buffers_per_layer, which reads both of these.
        if sparse_layer_ids is None:
            layer_mask = kwargs.get("layer_mask")
            if layer_mask is None:
                raise ValueError(
                    "Glm5NextCacheManager needs layer_mask or sparse_layer_ids "
                    "to place the indexer buffers"
                )
            sparse_layer_ids = [i for i, is_attention in enumerate(layer_mask) if is_attention]
        self.sparse_layer_ids = sorted(int(i) for i in sparse_layer_ids)
        self.index_state_dim = int(index_state_dim)
        if self.sparse_layer_ids and self.index_state_dim <= 0:
            raise ValueError(
                "glm5_next sparse layers need a positive index_state_dim "
                f"(got {self.index_state_dim})"
            )
        super().__init__(*args, **kwargs)

    def _extra_buffers_per_layer(self, *, tokens_per_block: int) -> dict[int, list[BufferConfig]]:
        """One ``Role.INDEX_KEY`` buffer per sparse layer, keyed by local id."""
        return {
            self.layer_offsets[layer_id]: [
                BufferConfig(
                    role=Role.INDEX_KEY,
                    size=self.get_layer_bytes_per_token(
                        self.layer_offsets[layer_id], Role.INDEX_KEY
                    )
                    * tokens_per_block,
                )
            ]
            for layer_id in self.sparse_layer_ids
            if layer_id in self.layer_offsets
        }

    def get_layer_bytes_per_token(self, local_layer_idx: int, data_role: DataRole) -> int:
        index_bytes = (
            self.index_state_dim * torch.bfloat16.itemsize
            if self.pp_layers[local_layer_idx] in self.sparse_layer_ids
            else 0
        )
        if data_role == Role.INDEX_KEY:
            return index_bytes
        cache_bytes = super().get_layer_bytes_per_token(local_layer_idx, data_role)
        return cache_bytes + index_bytes if data_role == Role.ALL else cache_bytes

    def _attention_cache_bytes_per_token(self) -> int:
        return sum(
            self.get_layer_bytes_per_token(local_layer_idx, Role.ALL)
            for local_layer_idx in range(self.num_local_layers)
        )

    @staticmethod
    def get_cache_size_per_token(
        model_config: ModelConfig,
        mapping: Mapping,
        *,
        max_batch_size: int,
        kv_cache_config: KvCacheConfig,
        tokens_per_block: int = 32,
        max_seq_len: int | None = None,
        **kwargs,
    ) -> tuple[int, int]:
        slope, fixed_cost = MambaHybridCacheManagerV2.get_cache_size_per_token(
            model_config,
            mapping,
            max_batch_size=max_batch_size,
            kv_cache_config=kv_cache_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            **kwargs,
        )
        spec_config = kwargs.get("spec_config")
        params = extract_mamba_kv_cache_params(
            model_config.pretrained_config,
            spec_config=spec_config,
            quant_config=model_config.quant_config,
        )
        kda_mask, attention_mask = params.get_layer_masks(
            is_draft=kwargs.get("is_draft", False),
            use_separate_draft_kv_cache=kwargs.get("use_separate_draft_kv_cache", False),
        )
        layer_mask = [kda or attention for kda, attention in zip(kda_mask, attention_mask)]
        local_layers, _ = get_pp_layers(
            sum(layer_mask), mapping, spec_config=spec_config, layer_mask=layer_mask
        )
        local_attention_layers = sum(attention_mask[layer] for layer in local_layers)
        config = unwrap_glm5_next_text_config(model_config.pretrained_config)
        # All GLM full-attention layers, including MTP layers, use the sparse indexer.
        # Indexer pages remain BF16 even when latent KV is quantized.
        index_bytes_per_token = (
            local_attention_layers
            * glm_kpool_cache_row_dim(int(config.index_head_dim))
            * torch.bfloat16.itemsize
        )
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
        # V2 reserves one partial attention page per resident lineage when snapshots
        # are reachable. Include the indexer section of those retained pages.
        if has_snapshots:
            fixed_cost += (
                max_batch_size * mapping.pp_size * tokens_per_block * index_bytes_per_token
            )
        return slope + index_bytes_per_token, fixed_cost

    def get_index_state_buffer(self, layer_idx: int) -> torch.Tensor | None:
        """Paged indexer state for ``layer_idx``, NHD-shaped."""
        return self.get_index_k_buffer(
            layer_idx,
            num_heads=1,
            head_dim=self.index_state_dim,
            dtype=torch.bfloat16,
            kv_layout="NHD",
        )

    def get_batch_slot_tables(self, request_ids: Sequence[int]) -> list[list[int]]:
        """Return raw slot IDs shared by the sparse layers' latent and indexer views.

        PP ranks without local sparse layers return empty rows.
        """
        local_layers = [i for i in self.sparse_layer_ids if i in self.layer_offsets]
        if not local_layers:
            return [[] for _ in request_ids]
        pools = {self.layer_to_pool_mapping_dict[self.layer_offsets[i]] for i in local_layers}
        if len(pools) != 1:
            raise ValueError(
                f"glm5_next sparse layers span V2 layer groups {sorted(pools)}; "
                "the slot-indexed latent/index views require a single group"
            )
        return self.get_batch_cache_indices(
            list(request_ids), layer_idx=local_layers[0], raw_indices=True
        )

    def get_latent_state_buffer(self, layer_idx: int) -> torch.Tensor | None:
        """Return a slot-major [slots, tokens_per_block, num_kv_heads, head_dim] view.

        get_buffers exposes per-page strides. Coalesced layers instead share raw slot
        IDs, so fold the page converter's scale into the slot stride, matching the
        INDEX_KEY view. No payload is copied.
        """
        pages = self.get_buffers(layer_idx)
        if pages is None:
            return None
        flat = pages[:, 0]  # [pages, tokens, heads, dim]
        converter = self.impl.get_page_index_converter(self.layer_offsets[layer_idx], Role.KEY)
        # ``scale`` is the buffers-per-slot count of the coalesced pool;
        # ``within_slot`` is where this layer's buffer sits inside a slot.
        scale, within_slot = int(converter.scale), int(converter.layer_offset)
        kv_factor = int(self.kv_factor)
        if scale <= kv_factor:
            return flat
        if scale % kv_factor:
            raise ValueError(
                f"glm5_next layer {layer_idx}: page-index scale {scale} is not a "
                f"multiple of kv_factor {kv_factor}, so the latent pool has no "
                "slot-major view"
            )
        # ``get_buffers`` already folds the kv_factor axis in, so its dim 0
        # counts kv-aggregate pages and the slot stride is the scale in those
        # units. Slot counts are derived the way get_index_k_buffer derives
        # them, so both pools expose the same slot space.
        stride = scale // kv_factor
        slots = (flat.shape[0] + within_slot) // stride
        # Slot s lives at pool page ``within_slot + s * stride``; the last one
        # stays in range because a layer's offset inside a coalesced slot is
        # always smaller than the slot itself.
        if within_slot >= stride or (slots - 1) * stride >= flat.shape[0]:
            raise ValueError(
                f"glm5_next layer {layer_idx}: {slots} slots at stride {stride} do "
                f"not fit {flat.shape[0]} pages from slot offset {within_slot}"
            )
        return torch.as_strided(
            flat,
            size=(slots, *flat.shape[1:]),
            stride=(stride * flat.stride(0), *flat.stride()[1:]),
            storage_offset=flat.storage_offset(),
        )
