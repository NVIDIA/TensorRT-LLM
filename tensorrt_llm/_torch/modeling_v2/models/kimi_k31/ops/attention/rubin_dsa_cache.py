# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer K-cache sizing of the Kimi K3.1 Rubin targets.

The family cache manager (``ops/ssm/rubin_kda_dsa_cache_manager.py``) stores FP8 index keys, one
byte per value plus one FP32 scale word per quantization block; these helpers size them.
"""

from __future__ import annotations

from typing import Optional

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention.backends.sparse.dsa.params import DSAParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.mapping import Mapping

ModelConfig = tensorrt_llm.bindings.ModelConfig


def get_indexer_k_cache_bytes_per_token(index_head_dim: int, quant_block_size: int) -> int:
    """Return the indexer K-cache footprint of one token: the FP8 keys and their scale words."""
    return index_head_dim + index_head_dim // quant_block_size * 4


def get_indexer_k_cache_size_per_token(
    model_config: ModelConfig,
    mapping: Mapping,
    num_layers: Optional[int] = None,
) -> int:
    """Estimate the indexer-only cache cost across local attention layers.

    Raises:
        ValueError: The config has no DSA parameters, or its index keys are not FP8.
    """
    sparse_attention_config = model_config.sparse_attention_config
    if sparse_attention_config is None:
        raise ValueError("sparse_attention_config is required for DSA cache")
    sparse_params = sparse_attention_config.to_sparse_params(
        pretrained_config=model_config.pretrained_config
    )
    if not isinstance(sparse_params, DSAParams):
        raise ValueError("DSA cache requires DSA sparse parameters")
    if sparse_params.indexer_k_dtype != "fp8":
        raise ValueError(
            f"The Kimi K3.1 Rubin DSA cache stores FP8 index keys, not {sparse_params.indexer_k_dtype}."
        )

    num_attention_layers = KVCacheManager._resolve_num_attention_layers(
        model_config, mapping, num_layers
    )
    return num_attention_layers * get_indexer_k_cache_bytes_per_token(
        sparse_params.index_head_dim, 128
    )


__all__ = [
    "get_indexer_k_cache_bytes_per_token",
    "get_indexer_k_cache_size_per_token",
]
