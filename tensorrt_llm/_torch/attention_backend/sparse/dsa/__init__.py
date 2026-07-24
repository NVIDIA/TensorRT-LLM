# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek Sparse Attention backend package."""

from .backend import DSATrtllmAttention
from .cache_manager import DSACacheManager
from .indexer import (
    _DG_SCHEDULE_BLOCK_KV,
    HAS_FAST_HADAMARD,
    Indexer,
    IndexerParams,
    IndexerPrefillChunkMetadata,
    RotaryEmbedding,
    _compute_slot_mappings,
    _effective_compress_ratio_divisor,
    _pick_dsl_expand,
    _select_indexer_compress_ratio,
    compute_cu_seqlen_kv_bounds_with_cache,
    rotate_activation,
    split_prefill_chunks,
    transform_local_topk_and_prepare_pool_view,
    warmup_heuristic_topk_decode,
)
from .metadata import DSAtrtllmAttentionMetadata
from .params import DSABackendForwardArgs, DSAMetadataParams, DSAParams

__all__ = [
    "HAS_FAST_HADAMARD",
    "DSABackendForwardArgs",
    "DSACacheManager",
    "DSAMetadataParams",
    "DSAParams",
    "DSATrtllmAttention",
    "DSAtrtllmAttentionMetadata",
    "Indexer",
    "IndexerParams",
    "IndexerPrefillChunkMetadata",
    "RotaryEmbedding",
    "_DG_SCHEDULE_BLOCK_KV",
    "_compute_slot_mappings",
    "_effective_compress_ratio_divisor",
    "_pick_dsl_expand",
    "_select_indexer_compress_ratio",
    "compute_cu_seqlen_kv_bounds_with_cache",
    "rotate_activation",
    "split_prefill_chunks",
    "transform_local_topk_and_prepare_pool_view",
    "warmup_heuristic_topk_decode",
]
