# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from tensorrt_llm._torch.pyexecutor import llm_request
from tensorrt_llm._torch.pyexecutor.resource_manager import GPU_LEVEL, KVCacheManagerV2
from tensorrt_llm._utils import (
    TensorWrapper,
    convert_to_torch_tensor,
    get_size_in_bytes,
    nvtx_range,
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
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManagerConfig as KVCacheManagerConfigPy
from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX

from .compressor import KVCacheDtype
from .deepseek_v4 import (
    DEEPSEEK_V4_SLIDING_ATTENTION,
    DEEPSEEK_V4_SPARSE_RATIO,
    DeepseekV4AttentionType,
    compress_ratio_has_attention,
    get_attn_dim,
    get_token_bytes,
    is_overlap_compressor,
)

# Keep DSV4 scratch reuse opt-in so per-layer block tables can be tested
# without scratch-page remapping in the default path.
DSV4_ENABLE_SWA_SCRATCH_REUSE_ENV = "TRTLLM_DSV4_ENABLE_SWA_SCRATCH_REUSE"


@dataclass(frozen=True)
class _SwaCopyConstants:
    pool_id: int
    scale: int
    scratch_pages: int
    layer_offsets: torch.Tensor
    layer_offsets_i64: torch.Tensor
    layer_ids: torch.Tensor


@dataclass(frozen=True)
class _SlidingCopyGroup:
    attention_type_value: int
    pool_id: int
    scale: int
    scratch_pages: int
    layer_indices: torch.Tensor
    layer_offsets: torch.Tensor
    layer_offsets_i64: torch.Tensor
    layer_ids: torch.Tensor


@dataclass(frozen=True)
class _SharedCopyConstants:
    pool_id: int
    scale: int


def _estimate_bytes_per_token(
    head_dim: int,
    index_head_dim: int,
    compress_ratios: List[int],
    has_fp8_kv_cache,
    attn_types: set[DeepseekV4AttentionType] | None = None,
    indexer_k_dtype: str = "fp8",
) -> int:
    total_bytes = 0
    for ratio in compress_ratios:
        for attn in DeepseekV4AttentionType:
            if attn_types is not None and attn not in attn_types:
                continue
            if compress_ratio_has_attention(ratio, attn):
                total_bytes += _get_attn_bytes_per_token(
                    head_dim,
                    index_head_dim,
                    ratio,
                    attn,
                    has_fp8_kv_cache,
                    indexer_k_dtype=indexer_k_dtype,
                )
    return total_bytes


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


def _enable_swa_scratch_reuse_from_env() -> bool:
    value = os.environ.get(DSV4_ENABLE_SWA_SCRATCH_REUSE_ENV, "0")
    return value.strip() == "1"


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
        sparse_attn_config: DeepSeekV4SparseAttentionConfig,
        max_input_len: Optional[int] = None,
        max_num_tokens: Optional[int] = None,
        **kwargs,
    ) -> None:
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
            enable_swa_scratch_reuse=_enable_swa_scratch_reuse_from_env(),
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
        self._host_attention_op_block_offsets_staging = torch.empty(
            self.num_attention_op_pools,
            index_mapper_capacity * self.max_beam_width,
            2,  # key and value
            self.max_blocks_per_seq,
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
            device="cpu",
        )
        self._host_per_layer_block_tables_staging = torch.empty(
            self.num_local_layers,
            len(DEEPSEEK_V4_SLIDING_ATTENTION),
            index_mapper_capacity * self.max_beam_width,
            self.max_blocks_per_seq,
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
            device="cpu",
        )
        self._host_compress_block_tables_staging = {
            compress_ratio: torch.empty(
                index_mapper_capacity * self.max_beam_width,
                self.max_blocks_per_seq,
                dtype=torch.int32,
                pin_memory=prefer_pinned(),
                device="cpu",
            )
            for compress_ratio in sorted(set(self._compress_ratios))
            if compress_ratio_has_attention(compress_ratio, DeepseekV4AttentionType.COMPRESS)
        }

        # Page-index converter metadata is fixed once the page tables are
        # prepared. Cache the tensor forms here so copy_batch_* only handles
        # per-request base indices and scratch ranges.
        first_swa_layer_id = self._layer_attn_to_layer_id[
            self.pp_layers[0], DeepseekV4AttentionType.SWA
        ]
        first_swa_converter = self.impl.get_page_index_converter(
            first_swa_layer_id, DeepseekV4AttentionType.SWA.role
        )
        swa_offsets = [first_swa_converter.layer_offset]
        for pp_layer in self.pp_layers[1:]:
            layer_id = self._layer_attn_to_layer_id[pp_layer, DeepseekV4AttentionType.SWA]
            converter = self.impl.get_page_index_converter(
                layer_id, DeepseekV4AttentionType.SWA.role
            )
            swa_offsets.append(converter.layer_offset)
        swa_layer_offsets = torch.tensor(swa_offsets, dtype=torch.int32, device="cpu").view(
            -1, 1, 1
        )
        self._swa_copy_constants = _SwaCopyConstants(
            pool_id=self.layer_to_pool_mapping_dict[first_swa_layer_id],
            scale=first_swa_converter.scale,
            scratch_pages=first_swa_converter.scratch_pages_per_block,
            layer_offsets=swa_layer_offsets,
            layer_offsets_i64=swa_layer_offsets.to(dtype=torch.int64),
            layer_ids=torch.arange(len(swa_offsets), dtype=torch.long, device="cpu").view(-1, 1, 1),
        )

        layer_groups: Dict[Tuple[int, int, int, int], List[Tuple[int, int]]] = defaultdict(list)
        for pp_layer in self.pp_layers:
            local_layer_idx = self.layer_offsets[pp_layer]
            compress_ratio = self._compress_ratios[pp_layer]
            for attention_type in DEEPSEEK_V4_SLIDING_ATTENTION:
                if not compress_ratio_has_attention(compress_ratio, attention_type):
                    continue
                layer_id = self._layer_attn_to_layer_id[pp_layer, attention_type]
                pool_id = self.layer_to_pool_mapping_dict[layer_id]
                converter = self.impl.get_page_index_converter(layer_id, attention_type.role)
                layer_groups[
                    (
                        attention_type.value,
                        pool_id,
                        converter.scale,
                        converter.scratch_pages_per_block,
                    )
                ].append((local_layer_idx, converter.layer_offset))

        self._sliding_copy_groups = []
        for (
            attention_type_value,
            pool_id,
            scale,
            scratch_pages,
        ), layers in layer_groups.items():
            layer_offsets = torch.tensor(
                [offset for _, offset in layers],
                dtype=torch.int32,
                device="cpu",
            ).view(-1, 1, 1)
            self._sliding_copy_groups.append(
                _SlidingCopyGroup(
                    attention_type_value=attention_type_value,
                    pool_id=pool_id,
                    scale=scale,
                    scratch_pages=scratch_pages,
                    layer_indices=torch.tensor(
                        [layer_idx for layer_idx, _ in layers],
                        dtype=torch.long,
                        device="cpu",
                    ),
                    layer_offsets=layer_offsets,
                    layer_offsets_i64=layer_offsets.to(dtype=torch.int64),
                    layer_ids=torch.arange(len(layers), dtype=torch.long, device="cpu").view(
                        -1, 1, 1
                    ),
                )
            )

        self._compress_copy_constants = {}
        for compress_ratio in self._host_compress_block_tables_staging:
            pp_layer = next(
                (
                    layer
                    for layer in self.pp_layers
                    if self._compress_ratios[layer] == compress_ratio
                ),
                None,
            )
            if pp_layer is None:
                continue
            layer_id = self._layer_attn_to_layer_id[pp_layer, DeepseekV4AttentionType.COMPRESS]
            converter = self.impl.get_page_index_converter(
                layer_id, DeepseekV4AttentionType.COMPRESS.role
            )
            self._compress_copy_constants[compress_ratio] = _SharedCopyConstants(
                pool_id=self.layer_to_pool_mapping_dict[layer_id],
                scale=converter.scale,
            )

        indexer_layer = next(
            (
                layer
                for layer in self.pp_layers
                if compress_ratio_has_attention(
                    self._compress_ratios[layer], DeepseekV4AttentionType.INDEXER_COMPRESS
                )
            ),
            None,
        )
        self._indexer_compress_copy_constants = None
        if indexer_layer is not None:
            layer_id = self._layer_attn_to_layer_id[
                indexer_layer, DeepseekV4AttentionType.INDEXER_COMPRESS
            ]
            converter = self.impl.get_page_index_converter(
                layer_id, DeepseekV4AttentionType.INDEXER_COMPRESS.role
            )
            self._indexer_compress_copy_constants = _SharedCopyConstants(
                pool_id=self.layer_to_pool_mapping_dict[layer_id],
                scale=converter.scale,
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

    def _get_cache_quota(self, max_tokens: int) -> int:
        quota = int(max_tokens * self.get_cache_bytes_per_token())
        # Add extra quota to ensure sufficient space for small max_tokens cases.
        quota += len(DeepseekV4AttentionType) * (2 << 20)
        return quota

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

        # For aggregated serving in large batch size:
        # Use 1 context request + (max_batch_size - 1) generation requests as
        # the typical step. An all-generation typical_step over-provisions the
        # compressed-cache pool at the expense of the SWA pool, starving the
        # SWA pool and artificially capping the achievable batch size.
        ctx_capacity = max_num_tokens if max_num_tokens is not None else max_seq_len
        typical_step = BatchDesc(
            kv_caches=[
                KVCacheDesc(capacity=ctx_capacity, history_length=0),
            ]
            + [KVCacheDesc(capacity=max_seq_len, history_length=max_seq_len - max_draft_len - 1)]
            * (max_batch_size - 1),
        )

        constraints = []
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
            constraints.append(BatchDesc([KVCacheDesc(capacity=max_num_tokens, history_length=0)]))

        return KVCacheManagerConfigPy(
            tokens_per_block=tokens_per_block,
            vocab_size=vocab_size,
            cache_tiers=cache_tiers,
            max_util_for_resume=kv_cache_config.max_util_for_resume,
            enable_stats=self.enable_stats,
            enable_swa_scratch_reuse=self.enable_swa_scratch_reuse,
            layers=layers,
            typical_step=typical_step,
            constraints=constraints,
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

        self._attn_ratio_to_pool_id = attn_ratio_to_pool_id
        self._attn_ratio_to_scale = attn_ratio_to_scale

    def _get_attn_bytes_per_block(
        self,
        attn_type: DeepseekV4AttentionType,
        layer_idx: int,
    ) -> int:
        """
        Get the cache bytes per token for a specific attention type and layer.
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
        return _estimate_bytes_per_token(
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
        return total_tokens * self.get_cache_bytes_per_token()

    def _get_generation_bytes(self, request: llm_request.LlmRequest) -> int:
        prompt_len = max(0, request.prompt_len)
        max_new_tokens = max(0, request.max_new_tokens)
        total_tokens = prompt_len + max_new_tokens + self.num_extra_kv_tokens
        has_fp8_kv_cache = self.dtype == DataType.FP8
        total_bytes = 0
        for layer in self.pp_layers:
            compress_ratio = self._compress_ratios[layer]
            for attn_type in DeepseekV4AttentionType:
                if not compress_ratio_has_attention(compress_ratio, attn_type):
                    continue
                token_bytes = _get_attn_bytes_per_token(
                    self.head_dim,
                    self.index_head_dim,
                    compress_ratio,
                    attn_type,
                    has_fp8_kv_cache,
                    indexer_k_dtype=self._indexer_k_dtype,
                )
                attn_tokens = total_tokens
                window_size = self._get_window_size(compress_ratio, attn_type)
                if window_size is not None:
                    attn_tokens = window_size
                total_bytes += attn_tokens * token_bytes
        return total_bytes

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

    def _get_batch_base_indices(
        self,
        pool_id: int,
        copy_idx: torch.Tensor,
        num_seqs: int,
        base_indices_cache: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if base_indices_cache is not None and pool_id in base_indices_cache:
            return base_indices_cache[pool_id]

        # host_kv_cache_block_offsets is indexed as [pool, copy_slot, kv_role, block].
        # DSV4 uses SELFKONLY, so column 0 is the only page-index table we need.
        batch_copy_idx = copy_idx[:num_seqs].to(
            device=self.host_kv_cache_block_offsets.device, dtype=torch.long
        )
        base_indices = self.host_kv_cache_block_offsets[pool_id].index_select(0, batch_copy_idx)[
            :, 0, :
        ]
        if base_indices_cache is not None:
            base_indices_cache[pool_id] = base_indices
        return base_indices

    def _get_batch_scratch_tensors(
        self,
        request_ids: List[int],
        pool_id: int,
        num_blocks: int,
        device: torch.device,
        scratch_tensors_cache: Optional[Dict[int, Optional[Tuple[torch.Tensor, ...]]]],
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        if scratch_tensors_cache is not None and pool_id in scratch_tensors_cache:
            return scratch_tensors_cache[pool_id]

        scratch_rows = []
        scratch_begs = []
        scratch_lengths = []
        scratch_slot_ids = []
        for row, request_id in enumerate(request_ids):
            scratch = self.kv_cache_map[request_id].get_scratch_desc(pool_id)
            if not scratch:
                continue
            beg = int(scratch.range.beg)
            end = min(int(scratch.range.end), num_blocks)
            if beg >= end:
                continue
            scratch_rows.append(row)
            scratch_begs.append(beg)
            scratch_lengths.append(end - beg)
            scratch_slot_ids.append(list(scratch.slot_ids))

        if not scratch_rows:
            if scratch_tensors_cache is not None:
                scratch_tensors_cache[pool_id] = None
            return None

        # Pack ragged scratch metadata into tensors once per pool. active/rows/columns
        # identify the [seq, block] positions that should use scratch slots instead
        # of the normal base page indices.
        max_length = max(scratch_lengths)
        max_num_slots = max(len(slot_ids) for slot_ids in scratch_slot_ids)
        slot_ids = torch.tensor(
            [
                row_slot_ids + [BAD_PAGE_INDEX] * (max_num_slots - len(row_slot_ids))
                for row_slot_ids in scratch_slot_ids
            ],
            dtype=torch.int64,
            device=device,
        )
        block_pos = torch.arange(max_length, dtype=torch.int64, device=device).view(1, -1)
        lengths = torch.tensor(scratch_lengths, dtype=torch.int64, device=device).view(-1, 1)
        active = block_pos < lengths
        rows = torch.tensor(scratch_rows, dtype=torch.long, device=device).view(-1, 1)
        columns = (
            torch.tensor(scratch_begs, dtype=torch.long, device=device).view(-1, 1) + block_pos
        )
        scratch_tensors = (
            slot_ids,
            block_pos,
            active,
            rows.expand(-1, max_length),
            columns,
        )
        if scratch_tensors_cache is not None:
            scratch_tensors_cache[pool_id] = scratch_tensors
        return scratch_tensors

    @nvtx_range("dsv4_copy_batch_block_offsets")
    def copy_batch_block_offsets(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
    ) -> None:
        """For compatibility with AttentionOp, copy only the SWA block offsets."""
        assert beam_width == 1, "beam_width must be 1 for KVCacheManagerV2"
        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts, beam_width)
        assert copy_idx.shape[0] == num_seqs
        assert self._host_attention_op_block_offsets_staging is not None

        staging = self._host_attention_op_block_offsets_staging
        staging[: self.num_attention_op_pools, :num_seqs].fill_(BAD_PAGE_INDEX)
        request_ids = request_ids[:num_seqs]
        # Only leading context rows can have scratch reuse enabled; generation
        # rows must keep the normal base page indices.
        context_request_ids = request_ids[:num_contexts]
        base_indices_cache: Dict[int, torch.Tensor] = {}
        scratch_tensors_cache: Dict[int, Optional[Tuple[torch.Tensor, ...]]] = {}
        constants = self._swa_copy_constants

        base_indices = self._get_batch_base_indices(
            constants.pool_id, copy_idx, num_seqs, base_indices_cache
        )
        scratch_tensors = None
        if context_request_ids:
            scratch_tensors = self._get_batch_scratch_tensors(
                context_request_ids,
                constants.pool_id,
                base_indices.size(1),
                base_indices.device,
                scratch_tensors_cache,
            )
        # Broadcast [layers, 1, 1] layer offsets over [1, seqs, blocks] base
        # indices, building all SWA layer tables in one tensor operation.
        layer_base_indices = base_indices.unsqueeze(0)
        converted = torch.where(
            layer_base_indices != BAD_PAGE_INDEX,
            layer_base_indices * constants.scale + constants.layer_offsets,
            layer_base_indices.expand(constants.layer_offsets.size(0), -1, -1),
        )
        if scratch_tensors is not None:
            # Scratch blocks replace a range of base indices. block_pos maps each
            # scratch block to a scratch slot and sub-page offset, then the
            # boolean mask scatters those scratch indices back into converted.
            slot_ids, block_pos, active, rows, columns = scratch_tensors
            total_offset = block_pos * constants.scratch_pages
            slot_idx = total_offset // constants.scale
            selected_slot_ids = slot_ids.gather(
                1,
                slot_idx.expand(slot_ids.size(0), -1).clamp(max=slot_ids.size(1) - 1),
            )
            scratch_index = (
                selected_slot_ids.unsqueeze(0) * constants.scale
                + (total_offset.view(1, 1, -1) % constants.scale + constants.layer_offsets_i64)
                % constants.scale
            )
            active = active.unsqueeze(0).expand(constants.layer_offsets.size(0), -1, -1)
            converted[
                constants.layer_ids.expand_as(active)[active],
                rows.unsqueeze(0).expand_as(active)[active],
                columns.unsqueeze(0).expand_as(active)[active],
            ] = scratch_index[active].to(dtype=converted.dtype)
        if converted.size(2) > self.max_blocks_per_seq:
            raise ValueError(
                f"Converted page indices length {converted.size(2)} exceeds "
                f"max_blocks_per_seq {self.max_blocks_per_seq}"
            )
        staging[: self.num_attention_op_pools, :num_seqs, 0, : converted.size(2)].copy_(converted)
        staging[: self.num_attention_op_pools, :num_seqs, 1, :] = staging[
            : self.num_attention_op_pools, :num_seqs, 0, :
        ]
        dst_tensor[: self.num_attention_op_pools, :num_seqs].copy_(
            staging[: self.num_attention_op_pools, :num_seqs], non_blocking=True
        )

    @nvtx_range("dsv4_copy_batch_sliding_block_tables")
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
        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts, 1)
        assert copy_idx.shape[0] == num_seqs
        assert self._host_per_layer_block_tables_staging is not None
        staging = self._host_per_layer_block_tables_staging
        staging[: self.num_local_layers, :, :num_seqs].fill_(BAD_PAGE_INDEX)
        request_ids = request_ids[:num_seqs]
        # Only leading context rows can have scratch reuse enabled; generation
        # rows must keep the normal base page indices.
        context_request_ids = request_ids[:num_contexts]
        base_indices_cache: Dict[int, torch.Tensor] = {}
        scratch_tensors_cache: Dict[int, Optional[Tuple[torch.Tensor, ...]]] = {}

        for group in self._sliding_copy_groups:
            base_indices = self._get_batch_base_indices(
                group.pool_id, copy_idx, num_seqs, base_indices_cache
            )
            scratch_tensors = None
            if context_request_ids:
                scratch_tensors = self._get_batch_scratch_tensors(
                    context_request_ids,
                    group.pool_id,
                    base_indices.size(1),
                    base_indices.device,
                    scratch_tensors_cache,
                )
            # Convert this whole layer group at once: [group_layers, seqs, blocks].
            layer_base_indices = base_indices.unsqueeze(0)
            converted = torch.where(
                layer_base_indices != BAD_PAGE_INDEX,
                layer_base_indices * group.scale + group.layer_offsets,
                layer_base_indices.expand(group.layer_offsets.size(0), -1, -1),
            )
            if scratch_tensors is not None:
                # Apply scratch overrides only for rows that actually have a
                # scratch range in this pool.
                slot_ids, block_pos, active, rows, columns = scratch_tensors
                total_offset = block_pos * group.scratch_pages
                slot_idx = total_offset // group.scale
                selected_slot_ids = slot_ids.gather(
                    1,
                    slot_idx.expand(slot_ids.size(0), -1).clamp(max=slot_ids.size(1) - 1),
                )
                scratch_index = (
                    selected_slot_ids.unsqueeze(0) * group.scale
                    + (total_offset.view(1, 1, -1) % group.scale + group.layer_offsets_i64)
                    % group.scale
                )
                active = active.unsqueeze(0).expand(group.layer_offsets.size(0), -1, -1)
                converted[
                    group.layer_ids.expand_as(active)[active],
                    rows.unsqueeze(0).expand_as(active)[active],
                    columns.unsqueeze(0).expand_as(active)[active],
                ] = scratch_index[active].to(dtype=converted.dtype)
            if converted.size(2) > self.max_blocks_per_seq:
                raise ValueError(
                    f"Converted page indices length {converted.size(2)} exceeds "
                    f"max_blocks_per_seq {self.max_blocks_per_seq}"
                )
            staging[
                group.layer_indices,
                group.attention_type_value,
                :num_seqs,
                : converted.size(2),
            ] = converted

        dst_tensor[: self.num_local_layers, :, :num_seqs].copy_(
            staging[: self.num_local_layers, :, :num_seqs], non_blocking=True
        )

    @nvtx_range("dsv4_copy_batch_compress_block_tables")
    def copy_batch_compress_block_tables(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        compress_ratio: int,
        num_contexts: int,
        num_seqs: int,
    ) -> None:
        """Build the COMPRESS block table for one compression ratio and copy it to the destination."""
        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts, 1)
        assert copy_idx.shape[0] == num_seqs
        staging = self._host_compress_block_tables_staging[compress_ratio]
        staging[:num_seqs].fill_(BAD_PAGE_INDEX)

        constants = self._compress_copy_constants.get(compress_ratio)
        if constants is None:
            dst_tensor[:num_seqs].copy_(staging[:num_seqs], non_blocking=True)
            return

        base_indices = self._get_batch_base_indices(constants.pool_id, copy_idx, num_seqs)
        # COMPRESS uses shared page indices in DSV4. No layer offset or
        # scratch-page replacement is needed, so the conversion stays 2D.
        converted = torch.where(
            base_indices != BAD_PAGE_INDEX,
            base_indices * constants.scale,
            base_indices,
        )
        if converted.size(1) > self.max_blocks_per_seq:
            raise ValueError(
                f"Converted page indices length {converted.size(1)} exceeds "
                f"max_blocks_per_seq {self.max_blocks_per_seq}"
            )
        staging[:num_seqs, : converted.size(1)].copy_(converted)

        dst_tensor[:num_seqs].copy_(staging[:num_seqs], non_blocking=True)

    @nvtx_range("dsv4_copy_batch_indexer_compress_block_tables")
    def copy_batch_indexer_compress_block_tables(
        self,
        host_block_table: torch.Tensor,
        request_ids: List[int],
        num_seqs: int,
    ) -> None:
        """Build the shared INDEXER_COMPRESS compatibility block table."""
        copy_idx = self.index_mapper.get_copy_index(request_ids, 0, 1)
        assert copy_idx.shape[0] == num_seqs
        assert host_block_table.device.type == "cpu"
        host_block_table[:num_seqs].fill_(BAD_PAGE_INDEX)

        constants = self._indexer_compress_copy_constants
        if constants is None:
            return

        base_indices = self._get_batch_base_indices(constants.pool_id, copy_idx, num_seqs)
        # INDEXER_COMPRESS is also a shared table and does not use scratch
        # pages in DSV4.
        converted = torch.where(
            base_indices != BAD_PAGE_INDEX,
            base_indices * constants.scale,
            base_indices,
        )
        if converted.size(1) > self.max_blocks_per_seq:
            raise ValueError(
                f"Converted page indices length {converted.size(1)} exceeds "
                f"max_blocks_per_seq {self.max_blocks_per_seq}"
            )
        host_block_table[:num_seqs, : converted.size(1)].copy_(converted)

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
        return _estimate_bytes_per_token(
            head_dim,
            index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            indexer_k_dtype=indexer_k_dtype,
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
