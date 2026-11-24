# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer K-cache sizing helpers and the DSA cache-manager check.

The Kimi K3.1 hybrid KDA + MLA cache manager (ssm_kda_dsa_cache_manager) owns the indexer K
cache; these helpers size it and let the DSA metadata recognize it.
"""

from __future__ import annotations

from typing import Optional

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import (
    is_dsa_cache_manager as _engine_is_dsa_cache_manager,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.mapping import Mapping

from ....ops.attention.dsa_params import DSAParams, k31_dsa_params

ModelConfig = tensorrt_llm.bindings.ModelConfig


def _get_indexer_k_cache_bytes_per_token(
    index_head_dim: int, quant_block_size: int, indexer_k_dtype: str
) -> int:
    """Return the raw indexer K-cache footprint for one token: the FP8 keys and their
    4-byte scale words."""
    if indexer_k_dtype not in ("fp8", "fp4"):
        raise ValueError(f"the Kimi K3.1 DSA indexer takes FP8 index keys, got {indexer_k_dtype!r}")
    if indexer_k_dtype == "fp4":
        return index_head_dim // 2 + index_head_dim // 32
    scale_bytes = index_head_dim // quant_block_size * 4
    return index_head_dim + scale_bytes


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
    if not isinstance(sparse_params, DSAParams):
        raise ValueError("DSA cache requires DSA sparse parameters")

    num_attention_layers = KVCacheManager._resolve_num_attention_layers(
        model_config, mapping, num_layers
    )
    bytes_per_layer = _get_indexer_k_cache_bytes_per_token(
        sparse_params.index_head_dim, 128, sparse_params.indexer_k_dtype
    )
    return num_attention_layers * bytes_per_layer


def is_dsa_cache_manager(cache_manager: object) -> bool:
    """Return whether a manager uses native DSA indexer page mapping.

    Besides the engine's DSA managers, a hybrid manager that implements the same
    indexer K-cache accessors opts in with ``provides_dsa_indexer_k_cache``.
    """
    return _engine_is_dsa_cache_manager(cache_manager) or getattr(
        cache_manager, "provides_dsa_indexer_k_cache", False
    )
