# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense Sparse Attention (DSA) backend for TRT-LLM with indexer-based TopK selection."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Union

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.mapping import Mapping

from .params import DSAParams

ModelConfig = tensorrt_llm.bindings.ModelConfig

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig, SparseAttentionConfig


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
            **kwargs,
        )
        self.num_blocks = self.blocks_in_primary_pool

        # Indexer K cache pool for DSA attention
        # Shape: [num_blocks, self.tokens_per_block * (index_head_dim + scale_size)]
        # Non-interleaved layout: [fp8_tok0 | fp8_tok1 | ... | scale_tok0 | scale_tok1 | ...]
        # Store FP8-quantized k values from the indexer
        self.indexer_k_cache_pool_per_layer = [
            self.get_indexer_k_cache_pool_data(layer_idx)
            for layer_idx in range(self.num_local_layers)
        ]

    def get_indexer_k_cache_buffers(self, layer_idx: int):
        """Get indexer k cache buffer from a specific layer pool."""
        block_size = self.tokens_per_block
        data_bytes = self.index_head_dim // 2 if self.use_fp4 else self.index_head_dim
        per_token_size = data_bytes + self.index_head_dim // self.quant_block_size * 4
        layer_offset = self.layer_offsets[layer_idx]
        return self.indexer_k_cache_pool_per_layer[layer_offset].view(
            self.num_blocks, block_size, 1, per_token_size
        )

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

        # Indexer K cache: physically allocated as raw UINT8 in
        # WindowBlockManager::allocatePools (poolDtype = kUINT8), so we assume
        # 1 byte/element here -- it is NOT scaled by the KV cache dtype (unlike
        # the latent above). The data-portion byte count already reflects fp8 vs
        # fp4 via indexer_data_dim.
        indexer_bytes_per_token = num_attention_layers * (
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
        # E2M1 codes per byte); the scale bytes are unchanged.
        indexer_data_dim = self.index_head_dim // 2 if self.use_fp4 else self.index_head_dim
        indexer_bytes_per_token = sum(self.num_kv_heads_per_layer) * (
            indexer_data_dim + self.index_head_dim // self.quant_block_size * 4
        )
        cache_size_bytes_per_token += indexer_bytes_per_token

        return cache_size_bytes_per_token
