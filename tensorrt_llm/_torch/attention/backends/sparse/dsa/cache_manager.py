# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense Sparse Attention (DSA) backend for TRT-LLM with indexer-based TopK selection."""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.kv_cache.sparse_offload_memory import sparse_cache_config
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import (
    TensorWrapper,
    convert_to_torch_tensor,
    get_size_in_bytes,
    get_sm_version,
    is_sm_100f,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    DataRole,
    PageIndexMode,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManagerConfig as KVCacheManagerConfigPy

from ..offload import (
    SPARSE_OFFLOAD_STATE_COPIES,
    SparseOffloadCacheMixin,
    SparseOffloadLayerDescriptor,
)
from .params import DSAParams

# A DSA selection names single tokens of the MLA latent.
DSA_SPARSE_UNIT_TOKENS = 1

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


def _resolve_fp8_ds_mla_head_dim(
    kv_cache_config: KvCacheConfig,
    tokens_per_block: int,
    head_dim: int,
    dtype: DataType,
) -> Tuple[bool, int]:
    """Validate and lower the inline-scale cache layout to storage elements."""
    # The Python LLM API config owns ``dtype``; direct cache-manager users and
    # older fixtures may still pass the mirrored pybind config, which does not
    # expose Python-only fields.
    use_fp8_ds_mla = getattr(kv_cache_config, "dtype", "auto") == "fp8_ds_mla"
    if not use_fp8_ds_mla:
        return False, head_dim

    from ..inline_scale_kv import PAGE_SIZE, TOKEN_BYTES

    if tokens_per_block != PAGE_SIZE:
        raise ValueError(
            f"FlashInfer DSA FMHA requires tokens_per_block={PAGE_SIZE}, got {tokens_per_block}."
        )
    if head_dim != 576:
        raise ValueError(f"inline-scale KV layout requires head_dim=576, got {head_dim}.")
    elem_bytes = {DataType.BF16: 2, DataType.FP8: 1}.get(dtype)
    if elem_bytes is None:
        raise ValueError(f"inline-scale KV layout requires BF16 or FP8, got {dtype}.")
    if TOKEN_BYTES % elem_bytes != 0:
        raise ValueError("inline-scale KV token size must align to the cache dtype.")
    return True, TOKEN_BYTES // elem_bytes


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


def _get_full_indexer_cache_size_per_token(
    model_config: ModelConfig,
    mapping: Mapping,
    num_layers: Optional[int] = None,
) -> int:
    """Estimate the indexer K-cache cost of the local full-indexer layers.

    Shared-indexer layers own no indexer row. With ``num_layers`` every one of
    those layers is costed as a full-indexer layer.
    """
    sparse_attention_config = model_config.sparse_attention_config
    config = model_config.pretrained_config
    sparse_params = sparse_attention_config.to_sparse_params(pretrained_config=config)
    if not isinstance(sparse_params, DSAParams):
        raise ValueError("DSA cache requires DSA sparse parameters")
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
    # Indexer K cache: physically allocated as raw UINT8, so it is NOT scaled by
    # the KV cache dtype (unlike the latent). Under FP4 the indexer stores two
    # E2M1 codes per byte; the scale bytes are unchanged.
    return num_indexer_layers * _get_indexer_k_cache_bytes_per_token(
        sparse_params.index_head_dim, 128, sparse_params.indexer_k_dtype == "fp4"
    )


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
        if sparse_attention_config.get_kv_cache_offload_config() is not None:
            raise ValueError(
                "DSA KV cache offload requires kv_cache_config.use_kv_cache_manager_v2=True."
            )
        self.quant_block_size = 128
        self.index_head_dim = sparse_params.index_head_dim
        # V1 pool geometry is fixed by the legacy C++ manager. Residual
        # quantization is enabled by DSACacheManagerV2, whose buffer sizes are
        # role-configurable.
        self.mla_kv_cache_residual_dim = 0
        # FP4 mode packs the indexer K cache as head_dim/2 data bytes + 4
        # scale bytes (vs. head_dim + 4 for FP8). The C++ WindowBlockManager
        # allocates the pool with this smaller stride when the flag is set.
        self.use_fp4 = sparse_params.indexer_k_dtype == "fp4"
        self.use_fp8_ds_mla, head_dim = _resolve_fp8_ds_mla_head_dim(
            kv_cache_config, tokens_per_block, head_dim, dtype
        )

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

        # Get latent KV-cache bytes. NVFP4 stores two values per data byte
        # plus one E4M3 block scale for every 16 values.
        quant_config = model_config.quant_config
        head_dim = config.kv_lora_rank + config.qk_rope_head_dim

        num_attention_layers = KVCacheManager._resolve_num_attention_layers(
            model_config, mapping, num_layers
        )
        kv_cache_config = kwargs.get("kv_cache_config")
        if (
            kv_cache_config is not None
            and getattr(kv_cache_config, "dtype", "auto") == "fp8_ds_mla"
        ):
            from ..inline_scale_kv import TOKEN_BYTES

            mem_per_token = num_attention_layers * TOKEN_BYTES
        elif quant_config is not None and quant_config.quant_mode.has_fp4_kv_cache():
            mem_per_token = num_attention_layers * (head_dim // 2 + math.ceil(head_dim / 16))
        else:
            # MLA latent K cache: stored at the KV cache dtype (BF16/FP8).
            bytes_per_element = (
                1 if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache() else 2
            )
            mem_per_token = num_attention_layers * head_dim * bytes_per_element

        return mem_per_token + _get_full_indexer_cache_size_per_token(
            model_config, mapping, num_layers
        )

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


class DSACacheManagerV2(SparseOffloadCacheMixin, KVCacheManagerV2):
    """KVCacheManagerV2-backed cache manager with a DSA indexer K-cache.

    With KV cache offload, every layer's MLA latent (``KEY``, with its
    ``KEY_BLOCK_SCALE`` under NVFP4) keeps its history in host memory and is
    read through a per-request GPU cache of single-token units; the indexer
    K-cache (``INDEX_KEY``) stays on the GPU. Full-indexer and shared-indexer
    layers differ only in where their selection comes from.
    """

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
        residual_config = (
            pretrained_config
            if pretrained_config is not None
            else getattr(model_config, "pretrained_config", None)
        )
        rope_head_dim = int(getattr(residual_config, "qk_rope_head_dim", 0))
        residual_quantization_enabled = (
            os.environ.get("TRTLLM_NVFP4_MLA_RESIDUAL_QUANTIZATION", "1") == "1"
        )
        self.mla_kv_cache_residual_dim = (
            rope_head_dim if dtype == DataType.NVFP4 and residual_quantization_enabled else 0
        )
        if self.mla_kv_cache_residual_dim % 16 != 0:
            raise ValueError(
                "NVFP4 MLA residual dimension must be divisible by 16, got "
                f"{self.mla_kv_cache_residual_dim}"
            )
        if self.mla_kv_cache_residual_dim > head_dim:
            raise ValueError(
                "NVFP4 MLA residual dimension cannot exceed the KV head dimension: "
                f"{self.mla_kv_cache_residual_dim} > {head_dim}"
            )
        self._unique_primary_pool: Optional[torch.Tensor] = None
        self.use_fp8_ds_mla, head_dim = _resolve_fp8_ds_mla_head_dim(
            kv_cache_config, tokens_per_block, head_dim, dtype
        )
        sparse_kv_offload_config = sparse_attention_config.get_kv_cache_offload_config()
        self._init_sparse_offload(
            enabled=sparse_kv_offload_config is not None, topk=sparse_params.index_topk
        )
        if sparse_kv_offload_config is not None:
            self._check_dsa_offload_support(dtype)

        from tensorrt_llm._torch.speculative import get_num_spec_layers

        total_num_layers = len(layer_mask) if layer_mask is not None else num_layers
        if spec_config is not None and layer_mask is None:
            total_num_layers += get_num_spec_layers(spec_config)
        self._indexer_k_cache_layer_mask = derive_indexer_k_cache_layer_mask(
            sparse_attention_config, pretrained_config, total_num_layers
        )
        self._selection_leader_cache: dict[int, int | None] | None = None

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
            sparse_kv_offload_config=sparse_kv_offload_config,
            **kwargs,
        )
        if self.num_local_layers == 0:
            raise ValueError("DSA requires at least one local attention layer")
        self._sparse_offload_descriptors: dict[int, SparseOffloadLayerDescriptor] = {}
        self._sparse_latent_write_pools: dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._sparse_latent_nvfp4_arenas: dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        if self._enable_kv_cache_offload:
            self._sparse_offload_descriptors = self.get_sparse_offload_descriptors()
            self._initialize_sparse_arena_padding(self._sparse_offload_descriptors)
            self._sparse_latent_write_pools = {
                layer: self._latent_arena_pool_tensors(descriptor)
                for layer, descriptor in self._sparse_offload_descriptors.items()
            }
            if self.dtype == DataType.NVFP4:
                self._sparse_latent_nvfp4_arenas = self._latent_nvfp4_arena_views(
                    self._sparse_offload_descriptors
                )
        primary_converters = [
            self.impl.get_page_index_converter(local_layer_idx, Role.KEY)
            for local_layer_idx in range(self.num_local_layers)
        ]
        self._primary_pool_page_index_params = [
            (int(converter.scale), int(converter.layer_offset)) for converter in primary_converters
        ]
        local_indexer_mask = self.indexer_k_cache_local_layer_mask
        full_indexer_local_layers = [
            local_layer_idx
            for local_layer_idx, has_indexer in enumerate(local_indexer_mask)
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
            self._get_indexer_k_cache_pool_data(local_layer_idx) if has_indexer else None
            for local_layer_idx, has_indexer in enumerate(local_indexer_mask)
        ]
        num_full = sum(local_indexer_mask)
        if num_full < self.num_local_layers:
            logger.info(
                f"[DSACacheManagerV2] Indexer k-cache: {num_full} of "
                f"{self.num_local_layers} local layers own an indexer k-cache."
            )

    @property
    def indexer_k_cache_local_layer_mask(self) -> list[bool]:
        """Project indexer ownership onto the wrapper's current local layers."""
        # The base wrapper may remove empty layers while building the cache.
        return [self._indexer_k_cache_layer_mask[layer_idx] for layer_idx in self.pp_layers]

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
            for local_layer_idx, has_indexer in enumerate(self.indexer_k_cache_local_layer_mask)
            if has_indexer
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
        if self._enable_kv_cache_offload:
            raise RuntimeError(
                "With KV cache offload the MLA latent's pages are host memory; attention "
                "reads it through the sparse cache arena"
            )
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
        storage_head_dim = first_head_dim + self.mla_kv_cache_residual_dim
        elements_per_layer = (
            self.tokens_per_block * first_num_heads * storage_head_dim // element_per_container
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
        if self.dtype == DataType.NVFP4 and self.mla_kv_cache_residual_dim > 0:
            if data_role == Role.KEY:
                cache_bytes += self.mla_kv_cache_residual_dim // 2
            elif data_role == Role.KEY_BLOCK_SCALE:
                cache_bytes += self.mla_kv_cache_residual_dim // 16
            elif data_role == Role.ALL:
                cache_bytes += self.mla_kv_cache_residual_dim // 2
                cache_bytes += self.mla_kv_cache_residual_dim // 16
        if data_role == Role.ALL and self._sparse_kv_offload is not None:
            # Role.ALL is what the device estimators count; the latent's pages
            # are host memory.
            for role in self._sparse_latent_roles():
                cache_bytes -= self.get_layer_bytes_per_token(local_layer_idx, role)
        if (
            data_role == Role.ALL
            and self._indexer_k_cache_layer_mask[self.pp_layers[local_layer_idx]]
        ):
            cache_bytes += self.get_layer_bytes_per_token(local_layer_idx, Role.INDEX_KEY)
        return cache_bytes

    def _check_dsa_offload_support(self, dtype: DataType) -> None:
        """Refuse offload on the paths that have no arena read or write yet.

        The GPU is checked first: elsewhere it decides which KV cache dtypes are
        allowed at all, so a dtype error on an unsupported GPU would mislead.
        """
        if not is_sm_100f():
            raise NotImplementedError(
                "DSA KV cache offload reads the latent through the trtllm-gen sparse MLA "
                f"kernels, which run on the SM100 family only; this GPU is SM{get_sm_version()}."
            )
        if self.use_fp8_ds_mla:
            raise NotImplementedError(
                "DSA KV cache offload does not support kv_cache_config.dtype='fp8_ds_mla'."
            )
        if dtype not in (DataType.FP8, DataType.BF16, DataType.NVFP4):
            raise NotImplementedError(
                f"DSA KV cache offload supports an FP8, BF16 or NVFP4 MLA latent, got {dtype}."
            )
        if self._sparse_offload_topk is None or self._sparse_offload_topk <= 0:
            raise ValueError("DSA KV cache offload requires a positive index_topk")

    def _sparse_latent_roles(self) -> Tuple[DataRole, ...]:
        """The latent buffers offload keeps in host memory: data, and NVFP4 block scales."""
        if self.dtype == DataType.NVFP4:
            return (Role.KEY, Role.KEY_BLOCK_SCALE)
        return (Role.KEY,)

    def _selection_leaders(self) -> dict[int, int | None]:
        """Per model layer, the model layer whose indexer selects for it.

        A group is a run of local layers starting at one that owns an indexer;
        the layers after it reuse its top-k, so the whole run is handed one
        selection and shares one resident set in the cache. ``None`` where the
        run holds a single layer, which is every layer of a model with no
        indexer sharing -- a layer that selects for itself never shares.

        Keyed and valued by model layer, because the local numbering is compacted
        after the cache config is built and this answer has to be the same on both
        sides of that. Computed once for the same reason.
        """
        if self._selection_leader_cache is None:
            leader_of: dict[int, int | None] = {}
            leader = self.pp_layers[0]
            for local, layer in enumerate(self.pp_layers):
                # The first layer leads whether or not the mask says it owns an
                # indexer: a run has to start somewhere, and
                # derive_indexer_k_cache_layer_mask already forces layer 0 to be full.
                if self._indexer_k_cache_layer_mask[layer] or local == 0:
                    leader = layer
                leader_of[layer] = leader
            members = Counter(leader_of.values())
            self._selection_leader_cache = {
                layer: leader if members[leader] > 1 else None
                for layer, leader in leader_of.items()
            }
        return self._selection_leader_cache

    def _build_cache_config(self, config: KVCacheManagerConfigPy) -> KVCacheManagerConfigPy:
        """With offload, give every layer's MLA latent host-backed history and a GPU cache."""
        if self._sparse_kv_offload is None:
            return config
        # Named by the leading local layer, so the layers of one indexer resolve once
        # between them. An unshared layer names nothing and keeps a row to itself.
        leaders = self._selection_leaders()

        def selection_id(local_layer_idx: int) -> int | None:
            leader = leaders[self.pp_layers[local_layer_idx]]
            # Negated so the id cannot land in the runtime's layer-id space, where an
            # unset id reads as the layer's own and a leading model layer's number may
            # well be some other layer's local one. The id only has to be equal within
            # a group and distinct between them; it is not a layer reference.
            return None if leader is None else -(leader + 1)

        def cache_for(local_layer_idx: int):
            return sparse_cache_config(
                num_selected_units=self._sparse_offload_topk,
                unit_tokens=DSA_SPARSE_UNIT_TOKENS,
                tokens_per_block=self.tokens_per_block,
                gpu_cache_factor=self._sparse_kv_offload.gpu_cache_factor,
                selection_id=selection_id(local_layer_idx),
            )

        sparse_roles = self._sparse_latent_roles()
        layers = []
        for layer in config.layers:
            if not isinstance(layer, AttentionLayerConfig):
                layers.append(layer)
                continue
            # A sparse buffer's size is still a whole page; the layer's sparse_cache
            # divides it into one-token units of the single-head latent.
            buffers = [
                BufferConfig(role=buffer.role, size=buffer.size, is_sparse=True)
                if buffer.role in sparse_roles
                else buffer
                for buffer in layer.buffers
            ]
            layers.append(
                AttentionLayerConfig(
                    layer_id=layer.layer_id,
                    buffers=buffers,
                    sliding_window_size=layer.sliding_window_size,
                    num_sink_tokens=layer.num_sink_tokens,
                    sparse_cache=cache_for(int(layer.layer_id)),
                )
            )
        return replace(config, layers=layers)

    def get_sparse_offload_descriptors(self) -> dict[int, SparseOffloadLayerDescriptor]:
        """Describe each local layer's MLA latent arena, by global layer index."""
        if not self._enable_kv_cache_offload:
            raise ValueError("Sparse offload metadata requires an offload-enabled cache manager")
        layout = self._sparse_arena_layout()
        leaders = self._selection_leaders()
        return {
            layer: self._build_sparse_offload_descriptor(
                layout,
                self.layer_offsets[layer],
                Role.KEY,
                unit_tokens=DSA_SPARSE_UNIT_TOKENS,
                scale_role=Role.KEY_BLOCK_SCALE if self.dtype == DataType.NVFP4 else None,
                selection_owner=leaders[layer],
                name=f"Layer {layer}",
            )
            for layer in self.pp_layers
        }

    def _latent_arena_pool_tensors(
        self, descriptor: SparseOffloadLayerDescriptor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a one-pool view of a layer's latent arena for the MLA cache writer.

        The writer addresses its cache through host pool pointers [pool, 2] and a
        layer-to-pool mapping [layer, 2]. Pool 0 starts at the arena base, so a
        one-token page whose block offset is the write row lands in that row.
        NVFP4 pointers are [pool, 2, 2], data then block scales; the scales'
        arena pairs its rows with the data's, so the same row addresses both.
        """
        data_base = descriptor.arena_base_addresses[descriptor.data_buffer_index]
        if descriptor.scale_buffer_index is None:
            pointers = [[data_base, 0]]
        else:
            scale_base = descriptor.arena_base_addresses[descriptor.scale_buffer_index]
            pointers = [[[data_base, scale_base], [0, 0]]]
        pool_pointers = torch.tensor(pointers, dtype=torch.int64, device="cpu")
        pool_mapping = torch.zeros((1, 2), dtype=torch.int32, device="cpu")
        return pool_pointers, pool_mapping

    def get_sparse_latent_write_pool(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the host pool pointers and mapping that address a layer's latent arena."""
        return self._sparse_latent_write_pools[layer_idx]

    def _latent_nvfp4_arena_views(
        self, descriptors: dict[int, SparseOffloadLayerDescriptor]
    ) -> dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """View each layer's NVFP4 latent arenas as the rows the NVFP4 gather reads.

        Data views are packed bytes [rows, 1, data bytes] and scale views E4M3
        [rows, 1, scale bytes], where a row is one token's unit and an arena row
        index addresses it. All layers' buffers of one pool share its region,
        and so its view.
        """
        views: dict[int, torch.Tensor] = {}

        def view(
            descriptor: SparseOffloadLayerDescriptor,
            index: int,
            dtype: torch.dtype,
            row_bytes: int,
        ) -> torch.Tensor:
            base = descriptor.arena_base_addresses[index]
            if base not in views:
                rows = descriptor.arena_row_counts[index]
                views[base] = convert_to_torch_tensor(
                    TensorWrapper(base, dtype, (rows, 1, row_bytes))
                )
            return views[base]

        arenas = {}
        for layer, descriptor in descriptors.items():
            local_layer_idx = self.layer_offsets[layer]
            arenas[layer] = (
                view(
                    descriptor,
                    descriptor.data_buffer_index,
                    torch.uint8,
                    self.get_layer_bytes_per_token(local_layer_idx, Role.KEY),
                ),
                view(
                    descriptor,
                    descriptor.scale_buffer_index,
                    torch.float8_e4m3fn,
                    self.get_layer_bytes_per_token(local_layer_idx, Role.KEY_BLOCK_SCALE),
                ),
            )
        return arenas

    def get_sparse_latent_nvfp4_arenas(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a layer's NVFP4 latent data and block-scale arena views, indexed by arena row."""
        return self._sparse_latent_nvfp4_arenas[layer_idx]

    def _sparse_offload_nvfp4_head_dim(self) -> Optional[int]:
        return self.head_dim if self.dtype == DataType.NVFP4 else None

    def get_sparse_offload_scratch_bytes(self, config: KVCacheManagerConfigPy) -> int:
        """Count the offload state, the latent writer's one-token page table and the
        indices the sparse attention reads.

        Per Batch row, the page table holds a K and a V offset and the write slots
        one entry, since an offloading step has one query token per request, and the
        attention reads ``topk`` merged indices. Both metadata copies hold them.
        """
        scratch = super().get_sparse_offload_scratch_bytes(config)
        if scratch:
            writer_entries = 3 * self.max_batch_size
            attn_index_entries = self.max_batch_size * self._sparse_offload_topk
            scratch += (
                SPARSE_OFFLOAD_STATE_COPIES
                * (writer_entries + attn_index_entries)
                * torch.int32.itemsize
            )
        return scratch

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
        sparse_attention_config = model_config.sparse_attention_config
        if (
            sparse_attention_config is not None
            and sparse_attention_config.get_kv_cache_offload_config() is not None
        ):
            # The latent's history lives in host memory; only the indexer K-cache
            # takes device bytes per token.
            return _get_full_indexer_cache_size_per_token(model_config, mapping, num_layers)
        cache_bytes = DSACacheManager.get_cache_size_per_token(
            model_config, mapping, num_layers=num_layers, **kwargs
        )
        quant_config = model_config.quant_config
        residual_quantization_enabled = (
            os.environ.get("TRTLLM_NVFP4_MLA_RESIDUAL_QUANTIZATION", "1") == "1"
        )
        if (
            quant_config is not None
            and quant_config.quant_mode.has_fp4_kv_cache()
            and residual_quantization_enabled
        ):
            config = model_config.pretrained_config
            residual_dim = int(getattr(config, "qk_rope_head_dim", 0))
            num_attention_layers = KVCacheManager._resolve_num_attention_layers(
                model_config, mapping, num_layers
            )
            cache_bytes += num_attention_layers * (residual_dim // 2 + math.ceil(residual_dim / 16))
        return cache_bytes

    def shutdown(self) -> None:
        self.indexer_k_cache_pool_per_layer = []
        self._unique_primary_pool = None
        super().shutdown()


def is_dsa_cache_manager(cache_manager: object) -> bool:
    """Return whether a manager uses native DSA indexer page mapping."""
    return isinstance(cache_manager, (DSACacheManager, DSACacheManagerV2))
