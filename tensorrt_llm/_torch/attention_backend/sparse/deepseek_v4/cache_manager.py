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

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from tensorrt_llm._torch.pyexecutor import llm_request
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import GPU_LEVEL, KVCacheManagerV2
from tensorrt_llm._utils import (
    TensorWrapper,
    convert_to_torch_tensor,
    get_size_in_bytes,
    nvtx_range_debug,
    prefer_pinned,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime import ModelConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BatchDesc,
    BufferConfig,
    DataRole,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    KVCacheDesc,
    LayerId,
    PageIndexMode,
    ScratchDesc,
    SwaScratchReuseConfig,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManagerConfig as KVCacheManagerConfigPy
from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX

from .compressor import KVCacheDtype
from .deepseek_v4 import (
    DEEPSEEK_V4_NON_SLIDING_ATTENTION,
    DEEPSEEK_V4_SLIDING_ATTENTION,
    DEEPSEEK_V4_SPARSE_RATIO,
    DeepseekV4AttentionType,
    compress_ratio_has_attention,
    get_attn_dim,
    get_token_bytes,
    is_overlap_compressor,
)


def _estimate_non_sliding_attn_size_per_token(
    head_dim: int,
    index_head_dim: int,
    compress_ratios: List[int],
    has_fp8_kv_cache,
    indexer_k_dtype: str = "fp8",
) -> int:
    total_bytes = 0
    for compress_ratio in compress_ratios:
        for attn_type in DEEPSEEK_V4_NON_SLIDING_ATTENTION:
            if compress_ratio_has_attention(compress_ratio, attn_type):
                total_bytes += _get_attn_bytes_per_token(
                    head_dim,
                    index_head_dim,
                    compress_ratio,
                    attn_type,
                    has_fp8_kv_cache,
                    indexer_k_dtype=indexer_k_dtype,
                )
    return total_bytes


def _estimate_swa_cache_size(
    head_dim: int,
    index_head_dim: int,
    compress_ratios: List[int],
    has_fp8_kv_cache,
    tokens_per_block: int,
    swa_window_size: int | None,
    *,
    context: bool,
    scratch: bool,
    indexer_k_dtype: str = "fp8",
) -> Tuple[int, int]:
    tokens_per_block = int(tokens_per_block)
    size_per_token = 0
    size_per_request = 0
    scratch_keys = set()
    for compress_ratio in compress_ratios:
        for attn_type in DEEPSEEK_V4_SLIDING_ATTENTION:
            if not compress_ratio_has_attention(compress_ratio, attn_type):
                continue
            if attn_type == DeepseekV4AttentionType.SWA:
                if swa_window_size is None:
                    continue
                window_size = swa_window_size
            else:
                state_factor = 2 if is_overlap_compressor(compress_ratio) else 1
                window_size = state_factor * compress_ratio
            if window_size <= 0:
                continue
            window_tokens = (
                (int(window_size) + tokens_per_block - 1) // tokens_per_block
            ) * tokens_per_block
            token_bytes = _get_attn_bytes_per_token(
                head_dim,
                index_head_dim,
                compress_ratio,
                attn_type,
                has_fp8_kv_cache,
                indexer_k_dtype=indexer_k_dtype,
            )
            if not context:
                size_per_request += window_tokens * token_bytes
            elif not scratch:
                size_per_token += token_bytes
            else:
                scratch_key = (attn_type, compress_ratio)
                if scratch_key in scratch_keys:
                    size_per_request += window_tokens * token_bytes
                else:
                    scratch_keys.add(scratch_key)
                    size_per_token += token_bytes
    return size_per_token, size_per_request


def _get_attn_bytes_per_token(
    head_dim: int,
    index_head_dim: int,
    compress_ratio: int,
    attn_type: DeepseekV4AttentionType,
    has_fp8_kv_cache: bool,
    indexer_k_dtype: str = "fp8",
) -> int:
    token_bytes = get_token_bytes(
        head_dim,
        index_head_dim,
        compress_ratio,
        attn_type,
        has_fp8_kv_cache,
        indexer_k_dtype=indexer_k_dtype,
    )
    if attn_type in [DeepseekV4AttentionType.COMPRESS, DeepseekV4AttentionType.INDEXER_COMPRESS]:
        token_bytes //= compress_ratio
    return token_bytes


def _get_index_mode(attn_type: DeepseekV4AttentionType) -> PageIndexMode:
    if attn_type in DEEPSEEK_V4_SLIDING_ATTENTION:
        return PageIndexMode.PER_LAYER
    else:
        return PageIndexMode.SHARED


class DeepseekV4CacheManager(KVCacheManagerV2):
    # This tensor is for compatibility with AttentionOp, it only contains swa attention.
    # kv_cache_pool_pointers contains one virtual attention-op pool per local
    # SWA layer, shape: [num_local_layers, 2]. The second column is always 0.
    kv_cache_pool_pointers: torch.Tensor
    # This tensor is for compatibility with AttentionOp, it only contains swa attention.
    # kv_cache_pool_mapping contains pool id and layer offset for each layer's swa attention,
    # shape: [num_local_layers, 2]
    kv_cache_pool_mapping: torch.Tensor
    # The block size of the (indexer) compressed cache.
    # For other attention types, block size is tokens_per_block.
    compressed_block_sizes: List[int]

    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: int = 1,
        max_batch_size: int,
        max_beam_width: int = 1,
        tokens_per_block: int,
        max_seq_len: int,
        vocab_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.BF16,
        compressor_dtype: DataType = DataType.FLOAT,
        sparse_attn_config: Optional[DeepSeekV4SparseAttentionConfig] = None,
        max_input_len: Optional[int] = None,
        max_num_tokens: Optional[int] = None,
        **kwargs,
    ) -> None:
        if sparse_attn_config is None:
            sparse_attn_config = kwargs.pop("sparse_attention_config", None)
        if sparse_attn_config is None and kwargs.get("model_config") is not None:
            sparse_attn_config = kwargs["model_config"].sparse_attention_config
        if sparse_attn_config is None:
            raise ValueError(
                "sparse_attn_config or sparse_attention_config is required "
                "for DeepseekV4CacheManager"
            )

        # DeepSeek-V4 specific attributes initialization
        assert kv_cache_type == CacheTypeCpp.SELFKONLY, "DeepSeek-V4 only supports SELFKONLY"
        assert num_kv_heads == 1, "DeepSeek-V4 only supports num_kv_heads == 1"
        assert len(sparse_attn_config.compress_ratios) >= num_layers, (
            "The length of compress ratios must be >= the number of layers"
        )
        assert dtype in [DataType.BF16, DataType.FP8], (
            f"Unsupported dtype: {dtype}, only support BF16 and FP8"
        )
        assert compressor_dtype == DataType.FLOAT, (
            f"Unsupported compressor dtype: {compressor_dtype}, only support FP32/TF32"
        )

        assert tokens_per_block in [128, 256], (
            f"DeepseekV4CacheManager requires tokens_per_block in [128, 256], got {tokens_per_block}. "
            f"Set kv_cache_config.tokens_per_block to 128 or 256."
        )

        self.index_head_dim = sparse_attn_config.index_head_dim
        self._compress_ratios = sparse_attn_config.compress_ratios
        # When MTP is enabled, enlarge the sliding window sizes by
        # max_draft_len so that rewinding rejected draft tokens can still
        # reach the KV entries that would otherwise have been evicted by the
        # sliding-window policy.
        spec_config = kwargs.get("spec_config", None)
        self._max_draft_len = spec_config.max_draft_len if spec_config is not None else 0
        self._swa_window_size = sparse_attn_config.window_size
        self._compressor_dtype = compressor_dtype
        # If MTP is enabled, append compress ratios for MTP virtual layers.
        # MTP adds (max_draft_len - 1) extra layers that mirror the last real
        # layer's attention pattern.  Only NEW entries are appended; existing
        # per-layer ratios are never modified, so a real layer with ratio==1
        # stays SWA-only.
        if self._max_draft_len > 0:
            self._compress_ratios = self._compress_ratios + [self._compress_ratios[-1]] * (
                self._max_draft_len - 1
            )
        self.compressed_block_sizes = [tokens_per_block // ratio for ratio in self._compress_ratios]

        self._init_indexer_dtype(sparse_attn_config)

        # _build_cache_config() needs them to build constraints
        self._max_input_len = max_input_len
        self._max_num_tokens = max_num_tokens

        # General initialization
        super().__init__(
            kv_cache_config,
            kv_cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            mapping=mapping,
            dtype=dtype,
            **kwargs,
        )
        self.is_vswa = True  # DeepSeek-V4 must has VSWA

        # DeepSeek-V4 expects cache of all layers with the same attention type and compress ratio
        # to be in the same pool and have the same scale.
        self._assert_layer_pool_scale()

        # For DeepSeek-V4 Attention, the base pointer for SWA pool
        # Use first PP layer instead of hardcoded 0 for pipeline parallelism.
        first_pp_layer = self.pp_layers[0]
        self.swa_pool_ptr = self.impl.get_mem_pool_base_address(
            self._layer_attn_to_layer_id[first_pp_layer, DeepseekV4AttentionType.SWA],
            DeepseekV4AttentionType.SWA.role,
            PageIndexMode.PER_LAYER,
        )

        self.compress_pool_ptrs = {}
        # Find first PP layer with each compress ratio for pool pointer lookup.
        pp_compress_ratios = [self._compress_ratios[layer] for layer in self.pp_layers]
        if 4 in pp_compress_ratios:  # indexer compressor
            first_layer_with_4 = self.pp_layers[pp_compress_ratios.index(4)]
            self.compress_pool_ptrs[4] = self.impl.get_mem_pool_base_address(
                self._layer_attn_to_layer_id[first_layer_with_4, DeepseekV4AttentionType.COMPRESS],
                DeepseekV4AttentionType.COMPRESS.role,
                PageIndexMode.SHARED,
            )
        if 128 in pp_compress_ratios:  # compressor
            first_layer_with_128 = self.pp_layers[pp_compress_ratios.index(128)]
            self.compress_pool_ptrs[128] = self.impl.get_mem_pool_base_address(
                self._layer_attn_to_layer_id[
                    first_layer_with_128, DeepseekV4AttentionType.COMPRESS
                ],
                DeepseekV4AttentionType.COMPRESS.role,
                PageIndexMode.SHARED,
            )

    def _format_kv_cache_pool_lifecycle_entry(self, layer_id: LayerId, role: DataRole) -> str:
        layer_semantics = self._manager_layer_id_to_layer_attn.get((layer_id, role))
        if layer_semantics is None:
            return super()._format_kv_cache_pool_lifecycle_entry(layer_id, role)

        model_layer_idx, attn_type = layer_semantics
        attr = self.impl._storage.get_buffer_attr(layer_id, role)
        pool_group_id = self.impl._storage.get_pool_group_index(attr.life_cycle_id)
        lifecycle = self.impl._life_cycles.get_life_cycle(attr.life_cycle_id)
        return (
            f"deepseek_role={attn_type.name}, "
            f"compress_ratio={self._compress_ratios[model_layer_idx]}, "
            f"pool_group_id={int(pool_group_id)}, "
            f"lifecycle_id={int(attr.life_cycle_id)}, lifecycle={lifecycle}"
        )

    def get_buffers(self, layer_idx: int, attn_type: DeepseekV4AttentionType) -> torch.Tensor:
        """
        Get the buffers for a specific layer and attention type.

        Args:
            layer_idx: The layer index
            attn_type: The attention type

        Returns:
            The buffer tensor (shape: [num_blocks, tokens_per_block, attn_dim])
            For blockwise FP8 layers, shape is [num_blocks, tokens_per_block, attn_dim + scale_size]
        """
        layer_id = self._layer_attn_to_layer_id[(layer_idx, attn_type)]
        data_role = attn_type.role
        page_index_mode = _get_index_mode(attn_type)
        addr = self.impl.get_mem_pool_base_address(layer_id, data_role, page_index_mode)

        block_size = self.tokens_per_block
        if attn_type in [
            DeepseekV4AttentionType.COMPRESS,
            DeepseekV4AttentionType.INDEXER_COMPRESS,
        ]:
            block_size = self.compressed_block_sizes[layer_idx]

        attn_dim = get_attn_dim(
            self.head_dim, self.index_head_dim, self._compress_ratios[layer_idx], attn_type
        )
        if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
            # Indexer always pack data + per-block scales into the same row.
            dim_per_token = self._indexer_data_size + self._indexer_scale_size
        else:
            dim_per_token = attn_dim

        page_index_upper_bound = self.impl.get_page_index_upper_bound(layer_id, data_role)
        if page_index_mode == PageIndexMode.PER_LAYER:
            converter = self.impl.get_page_index_converter(layer_id, data_role)
            if converter.layer_offset is not None:
                page_index_upper_bound += converter.layer_offset * converter.expansion

        shape = (page_index_upper_bound, block_size, dim_per_token)

        dtype = self.dtype
        # (indexer) compressor kv and score use compressor_dtype
        if attn_type in [
            DeepseekV4AttentionType.COMPRESSOR_KV,
            DeepseekV4AttentionType.COMPRESSOR_SCORE,
            DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
            DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
        ]:
            dtype = self._compressor_dtype
        elif attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
            dtype = self._indexer_dtype

        return convert_to_torch_tensor(TensorWrapper(addr, dtype, shape))

    def _get_window_size(
        self, compress_ratio: int, attn_type: DeepseekV4AttentionType
    ) -> int | None:
        if attn_type == DeepseekV4AttentionType.SWA:
            base_window_size = self._swa_window_size
        elif attn_type in (
            DeepseekV4AttentionType.COMPRESSOR_KV,
            DeepseekV4AttentionType.COMPRESSOR_SCORE,
            DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
            DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
        ):
            state_factor = 2 if is_overlap_compressor(compress_ratio) else 1
            base_window_size = state_factor * compress_ratio
        else:
            return None
        return base_window_size + self._max_draft_len

    def _prepare_page_table_tensor(self, index_mapper_capacity: int) -> None:
        # Tensors for compatibility with AttentionOp, only contains swa attention.
        # SWA uses per-layer page indices, so each SWA layer has a virtual
        # attention-op pool.
        # shape: [num_local_layers, 2]
        self.num_attention_op_pools = self.num_local_layers
        self.kv_cache_pool_pointers = torch.tensor(
            [
                [
                    self.impl.get_mem_pool_base_address(
                        self._layer_attn_to_layer_id[pp_layer, DeepseekV4AttentionType.SWA],
                        DeepseekV4AttentionType.SWA.role,
                        PageIndexMode.PER_LAYER,
                    ),
                    0,
                ]
                for pp_layer in self.pp_layers
            ],
            dtype=torch.int64,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        # shape: [num_local_layers, 2]
        self.kv_cache_pool_mapping = torch.tensor(
            [[local_layer_idx, 0] for local_layer_idx in range(self.num_local_layers)],
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.host_kv_cache_block_offsets = torch.empty(
            self.num_pools,
            index_mapper_capacity * self.max_beam_width,
            2,  # key and value
            self.max_blocks_per_seq,
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
            device="cpu",
        )
        staging_capacity = self.max_batch_size * self.max_beam_width
        self._host_compress_block_tables_staging = {
            compress_ratio: torch.full(
                (staging_capacity, self.max_blocks_per_seq),
                BAD_PAGE_INDEX,
                dtype=torch.int32,
                pin_memory=prefer_pinned(),
                device="cpu",
            )
            for compress_ratio in set(self._compress_ratios)
            if compress_ratio_has_attention(compress_ratio, DeepseekV4AttentionType.COMPRESS)
        }

        # layer offsets per layer and attn, shape [num_local_layers, len(DEEPSEEK_V4_SLIDING_ATTENTION)].
        self._layer_offsets = torch.full(
            (self.num_local_layers, len(DEEPSEEK_V4_SLIDING_ATTENTION)),
            -1,
            dtype=torch.int32,
            device="cpu",
        )
        # Pool ids per layer and sliding attention type, shape [num_local_layers, num_sliding_attention_types].
        self._layer_attn_pool_ids = torch.full(
            (self.num_local_layers, len(DEEPSEEK_V4_SLIDING_ATTENTION)),
            -1,
            dtype=torch.int32,
            device="cpu",
        )
        # Scales per layer and sliding attention type, shape [num_local_layers, num_sliding_attention_types].
        self._layer_attn_scales = torch.ones(
            (self.num_local_layers, len(DEEPSEEK_V4_SLIDING_ATTENTION)),
            dtype=torch.int32,
            device="cpu",
        )
        # Scratch pages per block per layer and sliding attention type, shape
        # [num_local_layers, num_sliding_attention_types].
        self._scratch_pages = torch.zeros(
            (self.num_local_layers, len(DEEPSEEK_V4_SLIDING_ATTENTION)),
            dtype=torch.int32,
            device="cpu",
        )
        self._csa_compress_pool_id = None
        self._csa_compress_scale = None
        self._csa_indexer_compress_pool_id = None
        self._csa_indexer_compress_scale = None
        self._hca_compress_pool_id = None
        self._hca_compress_scale = None

        for layer_idx in self.pp_layers:
            compress_ratio = self._compress_ratios[layer_idx]
            if compress_ratio == 4:
                compress_layer_id = self._layer_attn_to_layer_id[
                    layer_idx, DeepseekV4AttentionType.COMPRESS
                ]
                compress_converter = self.impl.get_page_index_converter(
                    compress_layer_id, DeepseekV4AttentionType.COMPRESS.role
                )
                self._csa_compress_pool_id = self.layer_to_pool_mapping_dict[compress_layer_id]
                self._csa_compress_scale = int(compress_converter.scale)
                indexer_layer_id = self._layer_attn_to_layer_id[
                    layer_idx, DeepseekV4AttentionType.INDEXER_COMPRESS
                ]
                indexer_converter = self.impl.get_page_index_converter(
                    indexer_layer_id, DeepseekV4AttentionType.INDEXER_COMPRESS.role
                )
                self._csa_indexer_compress_pool_id = self.layer_to_pool_mapping_dict[
                    indexer_layer_id
                ]
                self._csa_indexer_compress_scale = int(indexer_converter.scale)
            elif compress_ratio == 128:
                compress_layer_id = self._layer_attn_to_layer_id[
                    layer_idx, DeepseekV4AttentionType.COMPRESS
                ]
                compress_converter = self.impl.get_page_index_converter(
                    compress_layer_id, DeepseekV4AttentionType.COMPRESS.role
                )
                self._hca_compress_pool_id = self.layer_to_pool_mapping_dict[compress_layer_id]
                self._hca_compress_scale = int(compress_converter.scale)

            local_layer_idx = self.layer_offsets[layer_idx]
            for attn_type in DEEPSEEK_V4_SLIDING_ATTENTION:
                if not compress_ratio_has_attention(self._compress_ratios[layer_idx], attn_type):
                    continue
                layer_id = self._layer_attn_to_layer_id[layer_idx, attn_type]
                pool_id = self.layer_to_pool_mapping_dict[layer_id]
                converter = self.impl.get_page_index_converter(layer_id, attn_type.role)
                self._layer_attn_pool_ids[local_layer_idx, attn_type.value] = pool_id
                self._layer_attn_scales[local_layer_idx, attn_type.value] = converter.scale
                self._layer_offsets[local_layer_idx, attn_type.value] = converter.layer_offset
                self._scratch_pages[local_layer_idx, attn_type.value] = (
                    converter.scratch_pages_per_block
                )

        device = torch.device("cuda", torch.cuda.current_device())
        self._device_kv_cache_block_offsets_input = torch.empty_like(
            self.host_kv_cache_block_offsets,
            device=device,
        )
        self._precomputed_sliding_block_tables = torch.empty(
            (
                self.num_local_layers,
                len(DEEPSEEK_V4_SLIDING_ATTENTION),
                self.host_kv_cache_block_offsets.size(1),
                self.max_blocks_per_seq,
            ),
            dtype=torch.int32,
            device=device,
        )
        self._device_copy_idx_staging = torch.zeros(
            self.host_kv_cache_block_offsets.size(1),
            dtype=torch.int32,
            device=device,
        )
        self._device_num_contexts = torch.empty((), dtype=torch.int32, device=device)
        self._device_layer_offsets = self._layer_offsets.to(device=device)
        self._device_layer_attn_pool_ids = self._layer_attn_pool_ids.to(
            device=device,
            dtype=torch.long,
        )
        self._device_layer_attn_scales = self._layer_attn_scales.to(device=device)
        self._device_scratch_pages = self._scratch_pages.to(device=device)
        self._device_valid_sliding_pool = self._device_layer_attn_pool_ids >= 0
        self._device_block_positions = torch.arange(
            self.max_blocks_per_seq,
            dtype=torch.int32,
            device=device,
        )

        if self.enable_swa_scratch_reuse:
            valid_scales = self._layer_attn_scales[self._layer_attn_pool_ids >= 0]
            min_scale = int(valid_scales.min().item()) if valid_scales.numel() > 0 else 1
            max_scratch_pages = int(self._scratch_pages.max().item())
            self._max_scratch_slots = max(
                1,
                (self.max_blocks_per_seq * max_scratch_pages + min_scale - 1) // min_scale,
            )
            scratch_slots_shape = (
                self.num_pools,
                staging_capacity,
                self._max_scratch_slots,
            )
            self._host_scratch_begs_staging = torch.empty(
                self.num_pools,
                staging_capacity,
                dtype=torch.int32,
                pin_memory=prefer_pinned(),
                device="cpu",
            )
            self._host_scratch_ends_staging = torch.empty(
                self.num_pools,
                staging_capacity,
                dtype=torch.int32,
                pin_memory=prefer_pinned(),
                device="cpu",
            )
            self._host_scratch_slots_staging = torch.empty(
                scratch_slots_shape,
                dtype=torch.int32,
                pin_memory=prefer_pinned(),
                device="cpu",
            )
            self._device_scratch_begs_staging = torch.empty(
                self.num_pools,
                staging_capacity,
                dtype=torch.int32,
                device=device,
            )
            self._device_scratch_ends_staging = torch.empty(
                self.num_pools,
                staging_capacity,
                dtype=torch.int32,
                device=device,
            )
            self._device_scratch_slots_staging = torch.empty(
                scratch_slots_shape,
                dtype=torch.int32,
                device=device,
            )

    @property
    def blocks_in_primary_pool(self) -> int:
        first_pp_layer = self.pp_layers[0]
        swa_layer_id = self._layer_attn_to_layer_id[first_pp_layer, DeepseekV4AttentionType.SWA]
        return self.impl.get_page_index_upper_bound(swa_layer_id, DeepseekV4AttentionType.SWA.role)

    def get_num_free_blocks(self) -> int:
        # This method reports primary-pool capacity while the manager is empty.
        # DSV4 does not allocate the generic Role.KEY buffer, so use SWA's
        # model-specific DataRole for warmup capacity estimation.
        assert len(self.kv_cache_map) == 0, (
            "get_num_free_blocks is only used when the kv cache manager is empty"
        )
        max_num_pages = max(
            self.impl.get_page_index_upper_bound(
                self._layer_attn_to_layer_id[layer_idx, DeepseekV4AttentionType.SWA],
                DeepseekV4AttentionType.SWA.role,
            )
            for layer_idx in self.pp_layers
        )
        return max_num_pages

    def get_cache_indices(
        self,
        request_id: int,
        layer_idx: int,
        attn_type: DeepseekV4AttentionType,
    ) -> List[int]:
        """
        Get the cache block indices for a batch of requests at a specific layer and attention type.

        Args:
            request_id: The request id
            layer_idx: The layer index
            attn_type: The attention type

        Returns:
            The cache block indices, shape (max_blocks_per_seq,)
        """
        layer_id = self._layer_attn_to_layer_id[(layer_idx, attn_type)]
        data_role = attn_type.role
        pool_id = self.layer_to_pool_mapping_dict[layer_id]
        kv_cache = self.kv_cache_map[request_id]
        base_indices = kv_cache.get_base_page_indices(pool_id).tolist()
        converter = self.impl.get_page_index_converter(layer_id, data_role)
        page_index_mode = _get_index_mode(attn_type)
        return converter(
            base_indices,
            page_index_mode,
            kv_cache.get_scratch_desc(pool_id),
        )

    def _get_extra_quota_padding(self) -> int:
        """Ensure each attention type has minimal space when max_tokens is small."""
        return len(DeepseekV4AttentionType) * (2 << 20)

    def _get_quota_from_max_tokens(self, max_tokens: int) -> int:
        compress_ratios = [self._compress_ratios[layer] for layer in self.pp_layers]
        has_fp8_kv_cache = self.dtype == DataType.FP8
        non_sliding_attn_size_per_token = _estimate_non_sliding_attn_size_per_token(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            indexer_k_dtype=self._indexer_k_dtype,
        )
        (
            context_swa_size_per_token,
            _,
        ) = _estimate_swa_cache_size(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            self.tokens_per_block,
            self._swa_window_size,
            context=True,
            scratch=self.enable_swa_scratch_reuse,
            indexer_k_dtype=self._indexer_k_dtype,
        )
        (
            generation_swa_size_per_token,
            generation_swa_size_per_request,
        ) = _estimate_swa_cache_size(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            self.tokens_per_block,
            self._swa_window_size,
            context=False,
            scratch=False,
            indexer_k_dtype=self._indexer_k_dtype,
        )
        max_context_tokens = (
            self._max_num_tokens if self._max_num_tokens is not None else max_tokens
        )
        context_tokens = min(max_tokens, max_context_tokens)
        generation_tokens = max_tokens - context_tokens
        generation_quota = (
            max_tokens * non_sliding_attn_size_per_token
            + generation_tokens * generation_swa_size_per_token
            + self.max_batch_size * generation_swa_size_per_request
        )
        context_extra_quota = context_tokens * context_swa_size_per_token
        padding = self._get_extra_quota_padding()
        return int(generation_quota + context_extra_quota + padding)

    def _get_max_tokens_from_quota(self, quota: int) -> float:
        compress_ratios = [self._compress_ratios[layer] for layer in self.pp_layers]
        has_fp8_kv_cache = self.dtype == DataType.FP8
        non_sliding_attn_size_per_token = _estimate_non_sliding_attn_size_per_token(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            indexer_k_dtype=self._indexer_k_dtype,
        )
        context_swa_size_per_token, _ = _estimate_swa_cache_size(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            self.tokens_per_block,
            self._swa_window_size,
            context=True,
            scratch=self.enable_swa_scratch_reuse,
            indexer_k_dtype=self._indexer_k_dtype,
        )
        (
            generation_swa_size_per_token,
            generation_swa_size_per_request,
        ) = _estimate_swa_cache_size(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            self.tokens_per_block,
            self._swa_window_size,
            context=False,
            scratch=False,
            indexer_k_dtype=self._indexer_k_dtype,
        )
        padding = self._get_extra_quota_padding()
        size_per_batch = self.max_batch_size * generation_swa_size_per_request + padding
        if quota < size_per_batch:
            return 0
        context_size_per_token = non_sliding_attn_size_per_token + context_swa_size_per_token
        if self._max_num_tokens is None:
            return (quota - size_per_batch) / context_size_per_token

        context_limit_quota = self._max_num_tokens * context_size_per_token + size_per_batch
        if quota <= context_limit_quota:
            return (quota - size_per_batch) / context_size_per_token

        generation_size_per_token = non_sliding_attn_size_per_token + generation_swa_size_per_token
        if generation_size_per_token <= 0:
            return float("inf")
        return self._max_num_tokens + (quota - context_limit_quota) / generation_size_per_token

    def _build_cache_config(
        self,
        kv_cache_config: KvCacheConfig,
        *,
        tokens_per_block: int,
        vocab_size: int | None,
        cache_tiers: List[GpuCacheTierConfig | HostCacheTierConfig],
    ) -> KVCacheManagerConfigPy:
        """
        Create the cache manager config for DeepSeek-V4.
        """
        layers: List[AttentionLayerConfig] = []
        layer_attn_to_layer_id: Dict[Tuple[int, DeepseekV4AttentionType], LayerId] = {}
        manager_layer_id_to_layer_attn: Dict[
            Tuple[LayerId, DataRole], Tuple[int, DeepseekV4AttentionType]
        ] = {}

        def _add_layer(
            layer_idx: int,
            attention_types: List[DeepseekV4AttentionType],
            sliding_window_size: int | None,
        ) -> None:
            layer_id = LayerId(len(layers))
            for attn_type in attention_types:
                layer_attn_to_layer_id[layer_idx, attn_type] = layer_id
                manager_layer_id_to_layer_attn[layer_id, attn_type.role] = (
                    layer_idx,
                    attn_type,
                )
            layer_config = AttentionLayerConfig(
                layer_id=layer_id,
                buffers=[
                    BufferConfig(
                        role=attn_type.role,
                        size=self._get_attn_bytes_per_block(attn_type, layer_idx),
                    )
                    for attn_type in attention_types
                ],
                sliding_window_size=sliding_window_size,
                num_sink_tokens=None,
            )
            layers.append(layer_config)

        # create the layer config for DeepSeek-V4
        for layer in self.pp_layers:
            compress_ratio = self._compress_ratios[layer]
            if compress_ratio == 1:
                _add_layer(
                    layer,
                    [DeepseekV4AttentionType.SWA],
                    self._get_window_size(compress_ratio, DeepseekV4AttentionType.SWA),
                )
            elif compress_ratio == DEEPSEEK_V4_SPARSE_RATIO:
                _add_layer(
                    layer,
                    [DeepseekV4AttentionType.SWA],
                    self._get_window_size(compress_ratio, DeepseekV4AttentionType.SWA),
                )
                _add_layer(
                    layer,
                    [
                        DeepseekV4AttentionType.COMPRESS,
                        DeepseekV4AttentionType.INDEXER_COMPRESS,
                    ],
                    None,
                )
                _add_layer(
                    layer,
                    [
                        DeepseekV4AttentionType.COMPRESSOR_KV,
                        DeepseekV4AttentionType.COMPRESSOR_SCORE,
                        DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
                        DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
                    ],
                    self._get_window_size(compress_ratio, DeepseekV4AttentionType.COMPRESSOR_KV),
                )
            elif compress_ratio == 128:
                _add_layer(
                    layer,
                    [
                        DeepseekV4AttentionType.SWA,
                        DeepseekV4AttentionType.COMPRESSOR_KV,
                        DeepseekV4AttentionType.COMPRESSOR_SCORE,
                    ],
                    self._get_window_size(compress_ratio, DeepseekV4AttentionType.SWA),
                )
                _add_layer(layer, [DeepseekV4AttentionType.COMPRESS], None)
            else:
                raise ValueError(f"Unsupported DeepSeek-V4 compress ratio {compress_ratio}.")

        # the mapping from layer index and attention type to layer id
        self._layer_attn_to_layer_id = layer_attn_to_layer_id
        self._manager_layer_id_to_layer_attn = manager_layer_id_to_layer_attn
        # number of layers in the KVCacheManagerPy
        self._num_manager_layers = len(layers)

        # Build constraints and typical_step for better pool ratio.
        max_batch_size = self.max_batch_size
        max_seq_len = self.max_seq_len
        max_num_tokens = self._max_num_tokens
        max_draft_len = self._max_draft_len
        typical_step = None
        constraints = []
        if kv_cache_config.pool_ratio is None:
            typical_seq_len = (
                kv_cache_config.avg_seq_len
                if kv_cache_config.avg_seq_len is not None
                else max_seq_len
            )
            if typical_seq_len > max_seq_len:
                raise ValueError(
                    f"kv_cache_config.avg_seq_len ({typical_seq_len}) must be less than or "
                    f"equal to max_seq_len ({max_seq_len})"
                )

            # For aggregated serving in large batch size:
            # Use 1 context request + (max_batch_size - 1) generation requests as
            # the typical step. An all-generation typical_step over-provisions the
            # compressed-cache pool at the expense of the SWA pool, starving the
            # SWA pool and artificially capping the achievable batch size.
            ctx_capacity = max_num_tokens if max_num_tokens is not None else typical_seq_len
            generation_history_length = max(0, typical_seq_len - max_draft_len - 1)
            typical_step = BatchDesc(
                kv_caches=[
                    KVCacheDesc(capacity=ctx_capacity, history_length=0),
                ]
                + [
                    KVCacheDesc(
                        capacity=typical_seq_len,
                        history_length=generation_history_length,
                    )
                ]
                * (max_batch_size - 1),
            )

            # Constraint 1: cuda graph generation warmup — one decode request that has
            # accumulated to the tail of max_seq_len. Using history_length=max_seq_len-1
            # (instead of 0) lets SWA / SSM pools collapse to their windowed working set,
            # while full-cache pools still need max_seq_len/tokens_per_block blocks
            # because they don't age.
            constraints.append(
                BatchDesc([KVCacheDesc(capacity=max_seq_len, history_length=max_seq_len - 1)])
            )

            # Constraint 2: general / chunked-prefill warmup — one fresh context request
            # at max_num_tokens (the per-iteration token budget).
            if max_num_tokens is not None:
                constraints.append(
                    BatchDesc([KVCacheDesc(capacity=max_num_tokens, history_length=0)])
                )

        scratch_reuse_config = None
        if self.enable_swa_scratch_reuse:
            # Context requests will allocate num_extra_kv_tokens tokens for spec decoding.
            # Cache manager should not take them into account when calculating scratch range.
            # Therefore set max_rewind_len to num_extra_kv_tokens.
            scratch_reuse_config = SwaScratchReuseConfig(max_rewind_len=self.num_extra_kv_tokens)

        return KVCacheManagerConfigPy(
            tokens_per_block=tokens_per_block,
            vocab_size=vocab_size,
            cache_tiers=cache_tiers,
            max_util_for_resume=kv_cache_config.max_util_for_resume,
            swa_scratch_reuse=scratch_reuse_config,
            layers=layers,
            typical_step=typical_step,
            constraints=constraints,
            enable_stats=self.enable_stats,
            initial_pool_ratio=kv_cache_config.pool_ratio,
        )

    def _init_indexer_dtype(self, sparse_attn_config: DeepSeekV4SparseAttentionConfig) -> None:
        # Indexer compressor cache layout. Two modes are supported:
        #   - "fp8" (FP8 blockwise): 1 byte per value + 1 fp32 scale per 128
        #     values.
        #   - "fp4" (MXFP4 blockwise): ½ byte per value (two FP4 codes packed
        #     per byte) + 1 ue8m0 byte per 32 values. At index_head_dim=128
        #     this halves the per-token indexer-K footprint vs FP8.
        self._indexer_k_dtype = sparse_attn_config.indexer_k_dtype
        if self._indexer_k_dtype == "fp8":
            self._indexer_cache_dtype = KVCacheDtype.FP8_BLOCKWISE
            self._indexer_dtype = DataType.FP8
            self.quant_block_size = 128
            self._indexer_data_size = self.index_head_dim
            self._indexer_scale_size = get_size_in_bytes(
                self.index_head_dim // self.quant_block_size, DataType.FLOAT
            )
        elif self._indexer_k_dtype == "fp4":
            assert self.index_head_dim == 128, (
                f"FP4 indexer K cache requires index_head_dim=128, got {self.index_head_dim}."
            )
            self._indexer_cache_dtype = KVCacheDtype.MXFP4_BLOCKWISE
            # Pool dtype is uint8 because PyTorch can't allocate float4
            # backing storage; downstream consumers reinterpret these
            # raw bytes as packed E2M1 + UE8M0 exponents.
            self._indexer_dtype = DataType.UINT8
            self.quant_block_size = 32
            # Two E2M1 codes pack into one byte → half the data footprint.
            self._indexer_data_size = self.index_head_dim // 2
            # 1 UE8M0 byte per 32-element block.
            self._indexer_scale_size = self.index_head_dim // self.quant_block_size
        else:
            raise ValueError(
                f"Unsupported indexer_k_dtype "
                f"{sparse_attn_config.indexer_k_dtype!r}; expected "
                "'fp8' or 'fp4'."
            )
        # FP4 indexer flag mirrors `DSACacheManager.use_fp4` so the shared
        # base Indexer can branch without knowing the V4-specific enum.
        self.use_fp4 = self._indexer_k_dtype == "fp4"
        assert self.index_head_dim % self.quant_block_size == 0, (
            f"indexer_head_dim {self.index_head_dim} must be divisible by {self.quant_block_size}"
        )

    def _assert_layer_pool_scale(self) -> None:
        attn_ratio_to_pool_id = defaultdict[DeepseekV4AttentionType, dict[int, int]](lambda: {})
        attn_ratio_to_scale = defaultdict[DeepseekV4AttentionType, dict[int, int]](lambda: {})

        comb = [
            (attn_type, layer_idx)
            for attn_type in DeepseekV4AttentionType
            for layer_idx in self.pp_layers
            if compress_ratio_has_attention(self._compress_ratios[layer_idx], attn_type)
        ]
        for attn_type, layer_idx in comb:
            compress_ratio = self._compress_ratios[layer_idx]
            layer_id = self._layer_attn_to_layer_id[layer_idx, attn_type]
            pool_id = self.layer_to_pool_mapping_dict[layer_id]
            converter = self.impl.get_page_index_converter(layer_id, attn_type.role)
            assert converter.expansion == 1, "DeepSeek-V4 page index expansion must be 1"
            scale = converter.scale

            # check if the pool id is consistent
            if compress_ratio in attn_ratio_to_pool_id[attn_type]:
                other_pool_id = attn_ratio_to_pool_id[attn_type][compress_ratio]
                assert other_pool_id == pool_id, (
                    f"Layer {layer_idx} with compress ratio {compress_ratio}, "
                    f"its attention type {attn_type.name} has pool id {pool_id}, "
                    f"but another layer with the same compress ratio and attention type has pool id {other_pool_id}."
                    "DeepSeek-V4 expects they share the same pool."
                )
            else:
                attn_ratio_to_pool_id[attn_type][compress_ratio] = pool_id

            # check if the scale is consistent
            if compress_ratio in attn_ratio_to_scale[attn_type]:
                other_scale = attn_ratio_to_scale[attn_type][compress_ratio]
                assert other_scale == scale, (
                    f"Layer {layer_idx} with compress ratio {compress_ratio}, "
                    f"its attention type {attn_type.name} has scale {scale}, "
                    f"but another layer with the same compress ratio and attention type has scale {other_scale}."
                    "DeepSeek-V4 expects they share the same scale."
                )
            else:
                attn_ratio_to_scale[attn_type][compress_ratio] = scale

        # check if all swa attentions are in the same pool and have the same scale
        swa_pool_ids = set(attn_ratio_to_pool_id[DeepseekV4AttentionType.SWA].values())
        swa_scales = set(attn_ratio_to_scale[DeepseekV4AttentionType.SWA].values())
        assert len(swa_pool_ids) == 1, "All swa attentions must be in the same pool"
        assert len(swa_scales) == 1, "All swa attentions must have the same scale"

        # Ensure all compress ratios have SWA entries, not just PP-local ones.
        # The attention metadata uses compress_ratio=1 as a hardcoded SWA key,
        # but with pipeline parallelism a PP stage may not have any layers with
        # ratio 1. Since all SWA layers share the same pool, we populate entries
        # for every compress ratio in the model.
        swa_pool_id = next(iter(swa_pool_ids))
        swa_scale = next(iter(swa_scales))
        for ratio in set(self._compress_ratios):
            if ratio not in attn_ratio_to_pool_id[DeepseekV4AttentionType.SWA]:
                attn_ratio_to_pool_id[DeepseekV4AttentionType.SWA][ratio] = swa_pool_id
                attn_ratio_to_scale[DeepseekV4AttentionType.SWA][ratio] = swa_scale

    def _get_attn_bytes_per_block(
        self,
        attn_type: DeepseekV4AttentionType,
        layer_idx: int,
    ) -> int:
        """
        Get the cache bytes per block for a specific attention type and layer.
        """
        has_fp8_kv_cache = self.dtype == DataType.FP8
        token_bytes = get_token_bytes(
            self.head_dim,
            self.index_head_dim,
            self._compress_ratios[layer_idx],
            attn_type,
            has_fp8_kv_cache,
            indexer_k_dtype=self._indexer_k_dtype,
        )

        block_size = self.tokens_per_block
        if attn_type in [
            DeepseekV4AttentionType.COMPRESS,
            DeepseekV4AttentionType.INDEXER_COMPRESS,
        ]:
            block_size = self.compressed_block_sizes[layer_idx]

        return token_bytes * block_size

    def get_cache_bytes_per_token(self) -> int:
        """Get the average cache bytes per token for DeepSeek-V4."""
        has_fp8_kv_cache = self.dtype == DataType.FP8
        compress_ratios = [self._compress_ratios[layer] for layer in self.pp_layers]
        return _estimate_non_sliding_attn_size_per_token(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            indexer_k_dtype=self._indexer_k_dtype,
        )

    def get_max_resource_count(self) -> int:
        # Keep scheduler capacity tied to physical GPU KV quota in bytes.
        return int(self.impl.get_quota(GPU_LEVEL))

    def _is_context_request(self, request: llm_request.LlmRequest) -> bool:
        if request.is_context_init_state:
            return True
        return request.state == llm_request.LlmRequestState.CONTEXT_INIT

    def _is_generation_request(self, request: llm_request.LlmRequest) -> bool:
        if (
            request.is_generation_in_progress_state
            or request.is_generation_to_complete_state
            or request.is_disagg_generation_init_state
        ):
            return True
        return request.state in (
            llm_request.LlmRequestState.GENERATION_IN_PROGRESS,
            llm_request.LlmRequestState.GENERATION_TO_COMPLETE,
        )

    def _get_context_bytes(self, request: llm_request.LlmRequest) -> int:
        prompt_len = max(0, request.prompt_len)
        total_tokens = prompt_len + self.num_extra_kv_tokens
        return self._get_cache_bytes_for_tokens(total_tokens, context=True)

    def _get_generation_bytes(self, request: llm_request.LlmRequest) -> int:
        prompt_len = max(0, request.prompt_len)
        max_new_tokens = max(0, request.max_new_tokens)
        total_tokens = prompt_len + max_new_tokens + self.num_extra_kv_tokens
        return self._get_cache_bytes_for_tokens(total_tokens, context=False)

    def _get_cache_bytes_for_tokens(self, total_tokens: int, *, context: bool) -> int:
        has_fp8_kv_cache = self.dtype == DataType.FP8
        compress_ratios = [self._compress_ratios[layer] for layer in self.pp_layers]
        non_sliding_attn_size_per_token = _estimate_non_sliding_attn_size_per_token(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            indexer_k_dtype=self._indexer_k_dtype,
        )
        swa_size_per_token, swa_size_per_request = _estimate_swa_cache_size(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            self.tokens_per_block,
            self._swa_window_size,
            context=context,
            scratch=self.enable_swa_scratch_reuse,
            indexer_k_dtype=self._indexer_k_dtype,
        )
        return int(
            total_tokens * (non_sliding_attn_size_per_token + swa_size_per_token)
            + swa_size_per_request
        )

    def get_needed_resource_to_completion(self, request: llm_request.LlmRequest) -> int:
        if self._is_generation_request(request):
            return self._get_generation_bytes(request)
        if self._is_context_request(request):
            return self._get_context_bytes(request)
        raise ValueError(f"Unsupported request state: {request.state}")

    def get_layer_bytes_per_token(
        self,
        local_layer_idx: int,
        data_role: DataRole,
    ) -> int:
        raise NotImplementedError(
            "DeepSeek-V4 doesn't support get_layer_bytes_per_token, use _get_attn_bytes_per_block"
        )

    def get_indexer_k_cache_buffers(self, layer_idx: int) -> torch.Tensor:
        """
        Get the buffers for the indexer k cache for a specific layer.
        """
        buffer = self.get_buffers(layer_idx, DeepseekV4AttentionType.INDEXER_COMPRESS).unsqueeze(2)
        return buffer.view(torch.uint8)

    def _compute_shared_block_table(
        self, pool_id: int, scale: int, copy_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the shared offset for one pool and copy index.
        Return shape: [num_seqs, max_blocks_per_seq]
        """
        base = self.host_kv_cache_block_offsets[pool_id, copy_idx, 0, :]
        return torch.where(base == BAD_PAGE_INDEX, BAD_PAGE_INDEX, base * scale)

    def _copy_idx_to_device(self, copy_idx: torch.Tensor) -> torch.Tensor:
        num_tables = copy_idx.size(0)
        device_copy_idx = self._device_copy_idx_staging[:num_tables]
        device_copy_idx.copy_(copy_idx, non_blocking=True)
        # Keep the compiled graph independent of the active table count.
        return self._device_copy_idx_staging

    def _copy_scratch_metadata_to_device(
        self,
        scratch_descs_by_pool: list[list[ScratchDesc | None]],
        num_contexts: int,
        host_begs_staging: torch.Tensor,
        host_ends_staging: torch.Tensor,
        host_slots_staging: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shape: [num_pools, num_contexts]
        host_begs = host_begs_staging[:, :num_contexts]
        # shape: [num_pools, num_contexts]
        host_ends = host_ends_staging[:, :num_contexts]
        # shape: [num_pools, num_contexts, num_slots]
        host_slots = host_slots_staging[:, :num_contexts, :]
        host_begs.zero_()
        host_ends.zero_()
        host_slots.zero_()

        for pool_idx, scratch_descs in enumerate(scratch_descs_by_pool):
            for context_idx, desc in enumerate(scratch_descs):
                if desc is None:
                    continue
                slot_ids = desc.slot_ids
                if len(slot_ids) > self._max_scratch_slots:
                    raise RuntimeError(
                        f"Scratch slot count {len(slot_ids)} exceeds staging capacity "
                        f"{self._max_scratch_slots}"
                    )
                host_begs[pool_idx, context_idx] = int(desc.range.beg)
                host_ends[pool_idx, context_idx] = int(desc.range.end)
                for slot_idx, slot_id in enumerate(slot_ids):
                    host_slots[pool_idx, context_idx, slot_idx] = int(slot_id)

        self._device_scratch_begs_staging.copy_(host_begs_staging, non_blocking=True)
        self._device_scratch_ends_staging.copy_(host_ends_staging, non_blocking=True)
        self._device_scratch_slots_staging.copy_(host_slots_staging, non_blocking=True)
        # Keep scratch tensor shapes fixed; the device-side mask gates num_contexts.
        return (
            self._device_scratch_begs_staging,
            self._device_scratch_ends_staging,
            self._device_scratch_slots_staging,
        )

    @nvtx_range_debug("dsv4_compute_sliding_block_tables")
    def compute_sliding_block_tables(
        self,
        request_ids: List[int],
        num_contexts: int,
    ) -> None:
        """Compute all per-layer sliding-window block tables for this batch."""
        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts, 1)
        num_tables = copy_idx.size(0)
        self._num_tables = num_tables

        scratch_descs_by_pool = None
        if self.enable_swa_scratch_reuse and num_contexts > 0:
            scratch_descs_by_pool = [
                [
                    self.kv_cache_map[req].get_scratch_desc(pool_id)
                    for req in request_ids[:num_contexts]
                ]
                for pool_id in range(self.num_pools)
            ]

        device_copy_idx = self._copy_idx_to_device(copy_idx)
        self._device_kv_cache_block_offsets_input.copy_(
            self.host_kv_cache_block_offsets,
            non_blocking=True,
        )

        if scratch_descs_by_pool is not None:
            scratch_begs, scratch_ends, scratch_slots = self._copy_scratch_metadata_to_device(
                scratch_descs_by_pool,
                num_contexts,
                self._host_scratch_begs_staging,
                self._host_scratch_ends_staging,
                self._host_scratch_slots_staging,
            )
            self._device_num_contexts.fill_(num_contexts)
            torch.ops.trtllm.deepseek_v4_compute_sliding_block_tables_with_scratch(
                self._device_kv_cache_block_offsets_input,
                device_copy_idx,
                self._device_layer_attn_pool_ids,
                self._device_valid_sliding_pool,
                self._device_layer_attn_scales,
                self._device_layer_offsets,
                self._device_scratch_pages,
                scratch_begs,
                scratch_ends,
                scratch_slots,
                self._device_num_contexts,
                self._precomputed_sliding_block_tables,
            )
        else:
            torch.ops.trtllm.deepseek_v4_compute_sliding_block_tables(
                self._device_kv_cache_block_offsets_input,
                device_copy_idx,
                self._device_layer_attn_pool_ids,
                self._device_valid_sliding_pool,
                self._device_layer_attn_scales,
                self._device_layer_offsets,
                self._precomputed_sliding_block_tables,
            )

    @nvtx_range_debug("dsv4_copy_batch_block_offsets")
    def copy_batch_block_offsets(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
    ) -> None:
        """For compatibility with AttentionOp, copy only the SWA block offsets."""
        assert beam_width == 1, "DSV4 only supports beam width 1 now"
        assert dst_tensor.is_cuda, "copy_batch_block_offsets expects a CUDA destination"
        dst_tensor.fill_(BAD_PAGE_INDEX)
        dst_tensor[:, : self._num_tables, 0, :].copy_(
            self._precomputed_sliding_block_tables[
                :, DeepseekV4AttentionType.SWA.value, : self._num_tables, :
            ],
            non_blocking=True,
        )

    @nvtx_range_debug("dsv4_copy_batch_sliding_block_tables")
    def copy_batch_sliding_block_tables(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        num_contexts: int,
        num_seqs: int,
    ) -> None:
        """
        Copy the per-layer block tables for attentions managed in sliding-window mode to the GPU tensor.
        """
        assert dst_tensor.is_cuda, "copy_batch_sliding_block_tables expects a CUDA destination"
        dst_tensor.fill_(BAD_PAGE_INDEX)
        dst_tensor[:, :, : self._num_tables, :].copy_(
            self._precomputed_sliding_block_tables[:, :, : self._num_tables, :],
            non_blocking=True,
        )

    @nvtx_range_debug("dsv4_copy_batch_compress_block_tables")
    def copy_batch_compress_block_tables(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        compress_ratio: int,
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
    ) -> None:
        """Build the COMPRESS block table for one compression ratio and copy it to the destination."""
        assert beam_width == 1, "DSV4 only supports beam width 1 now"
        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts, beam_width)
        staging = self._host_compress_block_tables_staging[compress_ratio]
        if compress_ratio == 4:
            pool_id = self._csa_compress_pool_id
            scale = self._csa_compress_scale
        elif compress_ratio == 128:
            pool_id = self._hca_compress_pool_id
            scale = self._hca_compress_scale
        else:
            raise ValueError(
                f"Unsupported compress ratio {compress_ratio} for copy_batch_compress_block_tables"
            )

        if pool_id is None or scale is None:
            raise RuntimeError(
                f"Missing COMPRESS pool metadata for compress ratio {compress_ratio}"
            )
        staging[:num_seqs] = self._compute_shared_block_table(pool_id, scale, copy_idx)
        dst_tensor[:num_seqs].copy_(staging[:num_seqs], non_blocking=True)

    @nvtx_range_debug("dsv4_copy_batch_indexer_compress_block_tables")
    def copy_batch_indexer_compress_block_tables(
        self,
        host_block_table: torch.Tensor,
        request_ids: List[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
    ) -> None:
        """Build the shared INDEXER_COMPRESS compatibility block table."""
        assert beam_width == 1, "DSV4 only supports beam width 1 now"
        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts, beam_width)
        pool_id = self._csa_indexer_compress_pool_id
        scale = self._csa_indexer_compress_scale
        if pool_id is None or scale is None:
            raise RuntimeError("Missing INDEXER_COMPRESS pool metadata")
        host_block_table[:num_seqs] = self._compute_shared_block_table(pool_id, scale, copy_idx)

    @staticmethod
    def get_cache_size_per_token(model_config: ModelConfig, mapping: Mapping, **kwargs):
        config = model_config.pretrained_config
        head_dim = config.kv_lora_rank + config.qk_rope_head_dim
        index_head_dim = model_config.sparse_attention_config.index_head_dim
        pp_layers = mapping.pp_layers(model_config.get_num_attention_layers())
        compress_ratios = [
            model_config.sparse_attention_config.compress_ratios[layer] for layer in pp_layers
        ]
        quant_config = model_config.quant_config
        if quant_config is not None:
            has_fp8_kv_cache = quant_config.quant_mode.has_fp8_kv_cache()
        else:
            has_fp8_kv_cache = False
        indexer_k_dtype = model_config.sparse_attention_config.indexer_k_dtype
        non_sliding_attn_size_per_token = _estimate_non_sliding_attn_size_per_token(
            head_dim,
            index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            indexer_k_dtype=indexer_k_dtype,
        )
        swa_size_per_token, swa_size_per_request = _estimate_swa_cache_size(
            head_dim,
            index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            kwargs["tokens_per_block"],
            model_config.sparse_attention_config.window_size,
            context=False,
            scratch=False,
            indexer_k_dtype=indexer_k_dtype,
        )
        max_batch_size = int(kwargs.get("max_batch_size") or 0)
        return (
            non_sliding_attn_size_per_token + swa_size_per_token,
            swa_size_per_request * max_batch_size,
        )

    def check_invalid_values_in_kv_cache(self, fill_with_zero: bool = False) -> bool:
        some_checks_unavailable = False
        has_invalid_values = torch.tensor(
            [False], dtype=torch.bool, device=torch.cuda.current_device()
        )
        buffers_handled = set()

        # Handle each attention buffer from start to end to traverse the whole
        # KV cache. Multiple attention buffers can now share one cache layer.
        for (layer, attn), layer_id in self._layer_attn_to_layer_id.items():
            data_role = attn.role
            buffer_key = (layer_id, data_role)
            if buffer_key in buffers_handled:
                continue
            buffer = self.get_buffers(layer, attn)
            # process in chunks of 256 pages to avoid OoM
            for i in range(0, buffer.shape[0], 256):
                buffer_slice = buffer[i : i + 256]
                try:
                    has_invalid_values.logical_or_(torch.isnan(buffer_slice).any())
                    has_invalid_values.logical_or_(torch.isinf(buffer_slice).any())
                except NotImplementedError:
                    some_checks_unavailable = True
            if fill_with_zero:
                buffer.zero_()
            buffers_handled.add(buffer_key)
        torch.cuda.synchronize()

        if some_checks_unavailable:
            logger.warning(
                "`torch.isnan` or `torch.isinf` is not implemented for current kv cache dtype, "
                "related checks are skipped"
            )
        return bool(has_invalid_values)
