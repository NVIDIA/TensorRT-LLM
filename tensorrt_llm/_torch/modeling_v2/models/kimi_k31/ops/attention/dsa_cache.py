# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer K-cache sizing helpers and the DSA cache-manager check.

The Kimi K3.1 hybrid KDA + MLA cache manager (ssm_kda_dsa_cache_manager) owns the indexer K
cache; these helpers size it and let the DSA metadata recognize it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_params import (
    K31DSAParams,
    k31_dsa_params,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.mapping import Mapping

ModelConfig = tensorrt_llm.bindings.ModelConfig

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig


def _get_indexer_k_cache_bytes_per_token(
    index_head_dim: int, quant_block_size: int, indexer_k_dtype: str
) -> int:
    """Return the raw indexer K-cache footprint for one token: the FP8 keys (one byte per element) and
    their 4-byte scale word(s) (one FP32 per 128 values)."""
    if indexer_k_dtype != "fp8":
        raise ValueError(f"the Kimi K3.1 DSA indexer takes FP8 index keys, got {indexer_k_dtype!r}")
    data_bytes = index_head_dim
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
    sparse_params = k31_dsa_params(
        sparse_attention_config.to_sparse_params(pretrained_config=model_config.pretrained_config)
    )
    if not isinstance(sparse_params, K31DSAParams):
        raise ValueError("DSA cache requires DSA sparse parameters")

    num_attention_layers = KVCacheManager._resolve_num_attention_layers(
        model_config, mapping, num_layers
    )
    bytes_per_layer = _get_indexer_k_cache_bytes_per_token(
        sparse_params.index_head_dim, 128, sparse_params.indexer_k_dtype
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


def is_dsa_cache_manager(cache_manager: object) -> bool:
    """Return whether a manager holds the DSA indexer K cache (the Kimi K3.1 hybrid manager sets
    ``provides_dsa_indexer_k_cache``; the drafters' KV cache managers do not)."""
    return (
        hasattr(cache_manager, "provides_dsa_indexer_k_cache")
        and cache_manager.provides_dsa_indexer_k_cache is True
    )
