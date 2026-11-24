# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer K-cache helpers of the Kimi K3.1 Rubin targets.

The family cache manager (``ops/ssm/rubin_kda_dsa_cache_manager.py``) stores FP8 or FP4 index
keys by dtype name (``indexer_k_dtype``) and a K pool per storage dtype (``get_primary_pool``).
These helpers size the index keys by that name; ``get_primary_pool`` returns a layer's K pool
from such a manager, or the one pool of the engine's DSA cache managers.
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention.backends.sparse.dsa.params import DSAParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.mapping import Mapping

ModelConfig = tensorrt_llm.bindings.ModelConfig


def get_indexer_k_data_bytes_per_token(index_head_dim: int, indexer_k_dtype: str) -> int:
    """Return the indexer K payload bytes for one token, excluding its scale."""
    if indexer_k_dtype == "fp4":
        return index_head_dim // 2
    return index_head_dim


def get_indexer_k_cache_bytes_per_token(
    index_head_dim: int, quant_block_size: int, indexer_k_dtype: str
) -> int:
    """Return the raw indexer K-cache footprint for one token: the keys and their scale words."""
    data_bytes = get_indexer_k_data_bytes_per_token(index_head_dim, indexer_k_dtype)
    scale_bytes = index_head_dim // quant_block_size * 4
    return data_bytes + scale_bytes


def get_indexer_k_cache_size_per_token(
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
    bytes_per_layer = get_indexer_k_cache_bytes_per_token(
        sparse_params.index_head_dim, 128, sparse_params.indexer_k_dtype
    )
    return num_attention_layers * bytes_per_layer


def get_primary_pool(kv_cache_manager, local_layer_idx: int) -> torch.Tensor:
    """Return the physical K pool a local MLA layer's pages live in.

    The family cache manager keeps a pool per storage dtype (FP8 KV layers beside NVFP4 ones);
    the engine's DSA managers keep one pool for every layer.
    """
    get_pool = getattr(kv_cache_manager, "get_primary_pool", None)
    if get_pool is not None:
        return get_pool(local_layer_idx)
    return kv_cache_manager.get_unique_primary_pool()


__all__ = [
    "get_indexer_k_cache_bytes_per_token",
    "get_indexer_k_cache_size_per_token",
    "get_indexer_k_data_bytes_per_token",
    "get_primary_pool",
]
