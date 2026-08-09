# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense Sparse Attention (DSA) backend for TRT-LLM with indexer-based TopK selection."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager, get_pp_layers
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor, get_size_in_bytes
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, DataRole, PageIndexMode

from .params import DSAParams

ModelConfig = tensorrt_llm.bindings.ModelConfig

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig, SparseAttentionConfig


def _get_indexer_k_cache_bytes_per_token(
    index_head_dim: int, quant_block_size: int, use_fp4: bool
) -> int:
    """Return the raw indexer K-cache footprint for one token."""
    data_bytes = index_head_dim // 2 if use_fp4 else index_head_dim
    scale_bytes = index_head_dim // quant_block_size * 4
    return data_bytes + scale_bytes


def _get_indexer_k_cache_size_per_token(
    model_config: ModelConfig,
    mapping: Mapping,
    num_layers: Optional[int] = None,
) -> int:
    """Estimate the indexer-only cache cost across local attention layers."""
    sparse_attention_config = model_config.sparse_attention_config
    if sparse_attention_config is None:
        raise ValueError("sparse_attention_config is required for DSA cache")
    sparse_params = sparse_attention_config.to_sparse_params(
        pretrained_config=model_config.pretrained_config
    )
    if not isinstance(sparse_params, DSAParams):
        raise ValueError("DSA cache requires DSA sparse parameters")

    num_attention_layers = KVCacheManager._resolve_num_attention_layers(
        model_config, mapping, num_layers
    )
    bytes_per_layer = _get_indexer_k_cache_bytes_per_token(
        sparse_params.index_head_dim,
        128,
        sparse_params.indexer_k_dtype == "fp4",
    )
    return num_attention_layers * bytes_per_layer


def derive_indexer_k_cache_layer_mask(
    sparse_attention_config: "SparseAttentionConfig",
    pretrained_config,
    num_layers: int,
) -> List[bool]:
    return [
        bool(
            getattr(
                sparse_attention_config.to_sparse_params(
                    pretrained_config=pretrained_config, layer_idx=layer_idx
                ),
                "is_full_indexer_layer",
                True,
            )
        )
        for layer_idx in range(num_layers)
    ]


class DSACacheManager(KVCacheManager):
    """KV cache manager for DSA with additional indexer K-cache pools."""

    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        layer_mask: Optional[List[bool]] = None,
        max_num_tokens: int = 8192,
        model_config: Optional[ModelConfig] = None,
        max_beam_width: int = 1,
        sparse_attention_config: Optional["SparseAttentionConfig"] = None,
        pretrained_config=None,
        **kwargs,
    ) -> None:
        """Initialize cache manager with indexer K-cache pool per layer."""
        if sparse_attention_config is None:
            sparse_attention_config = kwargs.pop("sparse_attn_config", None)
        if sparse_attention_config is None and model_config is not None:
            sparse_attention_config = model_config.sparse_attention_config
        if sparse_attention_config is None:
            raise ValueError("sparse_attention_config is required for DSA cache")
        sparse_params = sparse_attention_config.to_sparse_params(
            pretrained_config=pretrained_config
        )
        if not isinstance(sparse_params, DSAParams):
            raise ValueError("DSA cache requires DSA sparse parameters")
        self.quant_block_size = 128
        self.index_head_dim = sparse_params.index_head_dim
        # FP4 mode packs the indexer K cache as head_dim/2 data bytes + 4
        # scale bytes (vs. head_dim + 4 for FP8). The C++ WindowBlockManager
        # allocates the pool with this smaller stride when the flag is set.
        self.use_fp4 = sparse_params.indexer_k_dtype == "fp4"

        from tensorrt_llm._torch.speculative import get_num_spec_layers

        total_num_layers = len(layer_mask) if layer_mask is not None else num_layers
        if spec_config is not None and layer_mask is None:
            total_num_layers += get_num_spec_layers(spec_config)
        indexer_k_cache_layer_mask = derive_indexer_k_cache_layer_mask(
            sparse_attention_config, pretrained_config, total_num_layers
        )

        super().__init__(
            kv_cache_config,
            kv_cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype,
            spec_config=spec_config,
            layer_mask=layer_mask,
            max_num_tokens=max_num_tokens,
            model_config=model_config,
            max_beam_width=max_beam_width,
            enable_indexer_k_cache=True,
            indexer_k_cache_quant_block_size=128,
            indexer_k_cache_index_head_dim=self.index_head_dim,
            indexer_k_cache_use_fp4=self.use_fp4,
            indexer_k_cache_layer_mask=indexer_k_cache_layer_mask,
            **kwargs,
        )
        self.num_blocks = self.blocks_in_primary_pool
        # V1 stores one INDEX_KEY page per physical pool slot.
        self.indexer_k_cache_page_scale = 1

        # Indexer K cache pool for DSA attention
        # Shape: [num_blocks, self.tokens_per_block * (index_head_dim + scale_size)]
        # Non-interleaved layout: [fp8_tok0 | fp8_tok1 | ... | scale_tok0 | scale_tok1 | ...]
        # Store FP8-quantized k values from the indexer.
        # Shared-indexer layers do not own a row in the masked indexer pool.
        local_mask = self.indexer_k_cache_local_layer_mask
        self.indexer_k_cache_pool_per_layer = [
            self.get_indexer_k_cache_pool_data(local_offset) if local_mask[local_offset] else None
            for local_offset in range(self.num_local_layers)
        ]
        num_full = sum(local_mask)
        if num_full < self.num_local_layers:
            logger.info(
                f"[DSACacheManager] Indexer k-cache: {num_full} of "
                f"{self.num_local_layers} local layers own an indexer k-cache."
            )

    def get_primary_pool_page_index_params(self, local_layer_idx: int) -> Tuple[int, int]:
        """Return V1 page scale and layer offset for sparse MLA indices."""
        return self.num_local_layers, local_layer_idx

    def get_indexer_k_cache_buffers(self, layer_idx: int):
        """Get indexer k cache buffer from a specific layer pool."""
        block_size = self.tokens_per_block
        per_token_size = _get_indexer_k_cache_bytes_per_token(
            self.index_head_dim, self.quant_block_size, self.use_fp4
        )
        layer_offset = self.layer_offsets[layer_idx]
        pool = self.indexer_k_cache_pool_per_layer[layer_offset]
        assert pool is not None, (
            f"Layer {layer_idx} is a shared-indexer layer and owns no indexer "
            f"k-cache; only full-indexer layers may access it."
        )
        return pool.view(self.num_blocks, block_size, 1, per_token_size)

    def get_pool_block_indices(
        self,
        num_seqs: int,
        *,
        request_ids: Optional[List[int]] = None,
        num_contexts: int = 0,
        beam_width: int = 1,
    ) -> torch.Tensor:
        """Decode V1 block offsets into physical memory-pool block indices."""
        del request_ids, num_contexts, beam_width
        encoded = self.host_kv_cache_block_offsets[0, :num_seqs, 0, :]
        max_pool_idx = self.blocks_in_primary_pool - 1
        return (encoded // self.num_local_layers).clamp(min=0, max=max_pool_idx).to(torch.int32)

    def get_batch_indexer_k_cache_indices(self, request_ids: List[int]) -> List[List[int]]:
        """
        Get the indices for the indexer k cache for a specific batch of requests.
        """
        # All of layers share the same cache indices, so we use layer index 0.
        return self.get_batch_cache_indices(request_ids, 0)

    def shutdown(self):
        """Release indexer K-cache pool references before C++ buffer cleanup."""
        # Clear Python references BEFORE C++ frees the underlying CUDA buffers
        self.indexer_k_cache_pool_per_layer = []
        super().shutdown()

    @staticmethod
    def get_cache_size_per_token(
        model_config: ModelConfig, mapping: Mapping, num_layers: Optional[int] = None, **kwargs
    ):
        """Estimate total cache bytes per token including indexer K-cache overhead."""
        config = model_config.pretrained_config
        sparse_attention_config = model_config.sparse_attention_config
        if sparse_attention_config is None:
            raise ValueError("sparse_attention_config is required for DSA cache")
        sparse_params = sparse_attention_config.to_sparse_params(
            pretrained_config=model_config.pretrained_config
        )
        if not isinstance(sparse_params, DSAParams):
            raise ValueError("DSA cache requires DSA sparse parameters")
        index_head_dim = sparse_params.index_head_dim
        quant_block_size = 128
        # Under FP4 the indexer stores two E2M1 codes per byte, so the
        # per-token data footprint halves (132 B -> 68 B at index_head_dim=128);
        # the scale bytes are unchanged (4 per token, one int32 holding four
        # UE8M0 exponents at quant_block_size=32 after packing).
        use_fp4 = sparse_params.indexer_k_dtype == "fp4"
        indexer_data_dim = index_head_dim // 2 if use_fp4 else index_head_dim

        # get kv cache dtype bytes
        mem_per_token = 2
        quant_config = model_config.quant_config
        if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache():
            mem_per_token = 1

        # get head dim
        head_dim = config.kv_lora_rank + config.qk_rope_head_dim

        num_attention_layers = KVCacheManager._resolve_num_attention_layers(
            model_config, mapping, num_layers
        )
        # MLA latent K cache: stored at the KV cache dtype (BF16/FP8).
        mem_per_token *= num_attention_layers * head_dim

        if num_layers is not None:
            num_indexer_layers = max(num_layers, 1)
        else:
            local_layer_ids = mapping.pp_layers(model_config.get_num_attention_layers())
            num_indexer_layers = sum(
                1
                for layer_id in local_layer_ids
                if getattr(
                    sparse_attention_config.to_sparse_params(
                        pretrained_config=config, layer_idx=layer_id
                    ),
                    "is_full_indexer_layer",
                    True,
                )
            )

        # Indexer K cache: physically allocated as raw UINT8 in
        # WindowBlockManager::allocatePools (poolDtype = kUINT8), so we assume
        # 1 byte/element here -- it is NOT scaled by the KV cache dtype (unlike
        # the latent above). The data-portion byte count already reflects fp8 vs
        # fp4 via indexer_data_dim.
        indexer_bytes_per_token = num_indexer_layers * (
            indexer_data_dim + index_head_dim // quant_block_size * 4
        )
        mem_per_token += indexer_bytes_per_token
        return mem_per_token

    def get_cache_bytes_per_token(self):
        """Compute actual cache bytes per token from instance configuration."""
        # MLA latent K cache: stored at the KV cache dtype (self.dtype). The
        # indexer K cache is added separately below.
        cache_size_per_token = math.ceil(
            self.kv_factor * sum(self.num_kv_heads_per_layer) * self.head_dim
        )

        if self.dtype not in (
            DataType.FP8,
            DataType.HALF,
            DataType.BF16,
            DataType.FLOAT,
            DataType.NVFP4,
        ):
            raise ValueError(f"Cannot support {self.dtype} KV cache.")

        cache_size_bytes_per_token = get_size_in_bytes(cache_size_per_token, self.dtype)
        if self.dtype == DataType.NVFP4:
            cache_size_bytes_per_token += self.calculate_scaling_factor_size_bytes(
                cache_size_per_token, quant_vector_size=16, scaling_factor_dtype=DataType.FP8
            )

        # Indexer K cache: physically allocated as raw UINT8 in
        # WindowBlockManager::allocatePools (poolDtype = kUINT8), so we assume
        # 1 byte/element here -- it is NOT scaled by the KV cache dtype (unlike
        # the latent above). Under FP4 the indexer data portion is halved (two
        # E2M1 codes per byte); the scale bytes are unchanged. Shared-indexer
        # layers contribute no bytes because they do not own a cache row.
        indexer_data_dim = self.index_head_dim // 2 if self.use_fp4 else self.index_head_dim
        local_mask = self.indexer_k_cache_local_layer_mask
        if local_mask is not None:
            num_indexer_layers = sum(
                kv_heads
                for kv_heads, has_indexer in zip(self.num_kv_heads_per_layer, local_mask)
                if has_indexer
            )
        else:
            num_indexer_layers = sum(self.num_kv_heads_per_layer)
        indexer_bytes_per_token = num_indexer_layers * (
            indexer_data_dim + self.index_head_dim // self.quant_block_size * 4
        )
        cache_size_bytes_per_token += indexer_bytes_per_token

        return cache_size_bytes_per_token


class DSACacheManagerV2(KVCacheManagerV2):
    """KVCacheManagerV2-backed cache manager with a DSA indexer K-cache."""

    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        layer_mask: Optional[List[bool]] = None,
        max_num_tokens: int = 8192,
        model_config: Optional[ModelConfig] = None,
        max_beam_width: int = 1,
        sparse_attention_config: Optional["SparseAttentionConfig"] = None,
        pretrained_config=None,
        **kwargs,
    ) -> None:
        if sparse_attention_config is None:
            sparse_attention_config = kwargs.pop("sparse_attn_config", None)
        if sparse_attention_config is None and model_config is not None:
            sparse_attention_config = model_config.sparse_attention_config
        if sparse_attention_config is None:
            raise ValueError("sparse_attention_config is required for DSA cache")
        sparse_params = sparse_attention_config.to_sparse_params(
            pretrained_config=pretrained_config
        )
        if not isinstance(sparse_params, DSAParams):
            raise ValueError("DSA cache requires DSA sparse parameters")

        self.quant_block_size = 128
        self.index_head_dim = sparse_params.index_head_dim
        self.use_fp4 = sparse_params.indexer_k_dtype == "fp4"
        self._unique_primary_pool: Optional[torch.Tensor] = None

        from tensorrt_llm._torch.speculative import get_num_spec_layers

        total_num_layers = len(layer_mask) if layer_mask is not None else num_layers
        if spec_config is not None and layer_mask is None:
            total_num_layers += get_num_spec_layers(spec_config)
        indexer_k_cache_layer_mask = derive_indexer_k_cache_layer_mask(
            sparse_attention_config, pretrained_config, total_num_layers
        )
        pp_layers, _ = get_pp_layers(
            num_layers, mapping, spec_config=spec_config, layer_mask=layer_mask
        )
        self.indexer_k_cache_local_layer_mask = [
            indexer_k_cache_layer_mask[layer_idx] for layer_idx in pp_layers
        ]

        super().__init__(
            kv_cache_config,
            kv_cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype,
            spec_config=spec_config,
            layer_mask=layer_mask,
            max_num_tokens=max_num_tokens,
            model_config=model_config,
            max_beam_width=max_beam_width,
            **kwargs,
        )
        if self.num_local_layers == 0:
            raise ValueError("DSA requires at least one local attention layer")
        primary_converters = [
            self.impl.get_page_index_converter(local_layer_idx, Role.KEY)
            for local_layer_idx in range(self.num_local_layers)
        ]
        self._primary_pool_page_index_params = [
            (int(converter.scale), int(converter.layer_offset)) for converter in primary_converters
        ]
        full_indexer_local_layers = [
            local_layer_idx
            for local_layer_idx, has_indexer in enumerate(self.indexer_k_cache_local_layer_mask)
            if has_indexer
        ]
        indexer_converters = [
            self.impl.get_page_index_converter(local_layer_idx, Role.INDEX_KEY)
            for local_layer_idx in full_indexer_local_layers
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
        self.num_blocks = self.blocks_in_primary_pool
        self.indexer_k_cache_pool_per_layer = [
            self._get_indexer_k_cache_pool_data(local_layer_idx)
            if self.indexer_k_cache_local_layer_mask[local_layer_idx]
            else None
            for local_layer_idx in range(self.num_local_layers)
        ]
        num_full = sum(self.indexer_k_cache_local_layer_mask)
        if num_full < self.num_local_layers:
            logger.info(
                f"[DSACacheManagerV2] Indexer k-cache: {num_full} of "
                f"{self.num_local_layers} local layers own an indexer k-cache."
            )

    def get_primary_pool_page_index_params(self, local_layer_idx: int) -> Tuple[int, int]:
        """Return the formal V2 page scale and layer offset for sparse MLA."""
        return self._primary_pool_page_index_params[local_layer_idx]

    def _extra_buffers_per_layer(self, *, tokens_per_block: int) -> dict[int, List[BufferConfig]]:
        return {
            local_layer_idx: [
                BufferConfig(
                    role=Role.INDEX_KEY,
                    size=self.get_layer_bytes_per_token(local_layer_idx, Role.INDEX_KEY)
                    * tokens_per_block,
                )
            ]
            for local_layer_idx in range(self.num_local_layers)
            if self.indexer_k_cache_local_layer_mask[local_layer_idx]
        }

    @property
    def blocks_in_primary_pool(self) -> int:
        """Return the physical slot count rather than the V2 page bound."""
        converter = self.impl.get_page_index_converter(0, Role.KEY)
        page_upper = self.impl.get_page_index_upper_bound(0, Role.KEY)
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
        flat_page_size = self.tokens_per_block * self.get_layer_bytes_per_token(
            local_layer_idx, Role.INDEX_KEY
        )
        return convert_to_torch_tensor(
            TensorWrapper(address, torch.uint8, [page_upper, flat_page_size])
        )

    def get_indexer_k_cache_buffers(self, layer_idx: int) -> torch.Tensor:
        """Return the page-indexed indexer K-cache view for a global layer."""
        layer_offset = self.layer_offsets[layer_idx]
        pool = self.indexer_k_cache_pool_per_layer[layer_offset]
        assert pool is not None, (
            f"Layer {layer_idx} is a shared-indexer layer and owns no indexer "
            f"k-cache; only full-indexer layers may access it."
        )
        per_token_size = self.get_layer_bytes_per_token(layer_offset, Role.INDEX_KEY)
        return pool.view(pool.shape[0], self.tokens_per_block, 1, per_token_size)

    def get_pool_block_indices(
        self,
        num_seqs: int,
        *,
        request_ids: Optional[List[int]] = None,
        num_contexts: int = 0,
        beam_width: int = 1,
    ) -> torch.Tensor:
        """Read V2 stable slots in current-batch order as physical block IDs."""
        if request_ids is None:
            raise ValueError("DSACacheManagerV2 requires request_ids to map stable slots")
        if len(request_ids) != num_seqs:
            raise ValueError(f"Expected {num_seqs} request IDs, got {len(request_ids)}")
        copy_idx = self.index_mapper.get_copy_index(list(request_ids), num_contexts, beam_width)
        copy_idx = copy_idx.to(device="cpu", dtype=torch.long)
        block_indices = self.host_kv_cache_block_offsets[0, copy_idx, 0, :]
        return block_indices.clamp(min=0, max=self.num_blocks - 1).to(torch.int32)

    def get_unique_primary_pool(self) -> torch.Tensor:
        """Return the uniform MLA K pool in the V1-compatible layout."""
        if self._unique_primary_pool is not None:
            return self._unique_primary_pool
        if self.num_local_layers == 0:
            raise ValueError("DSA requires at least one local attention layer")
        if self.kv_factor != 1:
            raise ValueError("DSA requires a SELFKONLY KV cache")

        first_head_dim = self.head_dim_per_layer[0]
        first_num_heads = self.num_kv_heads_per_layer[0]
        if any(head_dim != first_head_dim for head_dim in self.head_dim_per_layer):
            raise ValueError("DSA requires a uniform KV head dimension")
        if any(num_heads != first_num_heads for num_heads in self.num_kv_heads_per_layer):
            raise ValueError("DSA requires a uniform KV head count")

        first_converter = self.impl.get_page_index_converter(0, Role.KEY)
        if int(first_converter.expansion) != 1:
            raise ValueError("DSA MLA K-cache does not support page expansion")
        if int(first_converter.layer_offset) != 0:
            raise ValueError("The first DSA layer must start at pool offset 0")
        if int(first_converter.scale) != self.num_local_layers:
            raise ValueError("DSA requires one uniformly coalesced K page per local layer")

        page_stride = self.impl.get_page_stride(0, Role.KEY)
        base_address = self.impl.get_mem_pool_base_address(0, Role.KEY, PageIndexMode.SHARED)
        for local_layer_idx in range(1, self.num_local_layers):
            converter = self.impl.get_page_index_converter(local_layer_idx, Role.KEY)
            if int(converter.scale) != int(first_converter.scale) or int(converter.expansion) != 1:
                raise ValueError("DSA requires a uniform page-index mapping across layers")
            if int(converter.layer_offset) != local_layer_idx:
                raise ValueError("DSA requires K pages to follow local-layer order")
            address = self.impl.get_mem_pool_base_address(
                local_layer_idx, Role.KEY, PageIndexMode.SHARED
            )
            expected_address = base_address + int(converter.layer_offset) * page_stride
            if int(address) != int(expected_address):
                raise ValueError("DSA requires contiguous per-layer K pages in each slot")

        element_per_container = 2 if self.dtype == DataType.NVFP4 else 1
        dtype = torch.int8 if self.dtype == DataType.NVFP4 else self.dtype
        elements_per_layer = (
            self.tokens_per_block * first_num_heads * first_head_dim // element_per_container
        )
        shape = [
            self.blocks_in_primary_pool,
            self.num_local_layers,
            1,
            elements_per_layer,
        ]
        self._unique_primary_pool = convert_to_torch_tensor(
            TensorWrapper(base_address, dtype, shape)
        )
        return self._unique_primary_pool

    def get_layer_bytes_per_token(self, local_layer_idx: int, data_role: DataRole) -> int:
        if data_role == Role.INDEX_KEY:
            return _get_indexer_k_cache_bytes_per_token(
                self.index_head_dim, self.quant_block_size, self.use_fp4
            )
        cache_bytes = super().get_layer_bytes_per_token(local_layer_idx, data_role)
        if data_role == Role.ALL and self.indexer_k_cache_local_layer_mask[local_layer_idx]:
            cache_bytes += self.get_layer_bytes_per_token(local_layer_idx, Role.INDEX_KEY)
        return cache_bytes

    def get_cache_bytes_per_token(self) -> int:
        return sum(
            self.get_layer_bytes_per_token(local_layer_idx, Role.ALL)
            for local_layer_idx in range(self.num_local_layers)
        )

    @staticmethod
    def get_cache_size_per_token(
        model_config: ModelConfig,
        mapping: Mapping,
        num_layers: Optional[int] = None,
        **kwargs,
    ):
        return DSACacheManager.get_cache_size_per_token(
            model_config, mapping, num_layers=num_layers, **kwargs
        )

    def shutdown(self) -> None:
        self.indexer_k_cache_pool_per_layer = []
        self._unique_primary_pool = None
        super().shutdown()


def is_dsa_cache_manager(cache_manager: object) -> bool:
    """Return whether a manager uses native DSA indexer page mapping."""
    return isinstance(cache_manager, (DSACacheManager, DSACacheManagerV2))
