# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 sparse attention package.

Layered as:

  * :mod:`.kernels`       -- OpenAI Triton kernels (per-block max
                              score, masked softmax for sparse GQA).
  * :mod:`.metadata`      -- ``MiniMaxM3SparseConfig`` /
                              ``MiniMaxM3SparseAttentionMetadata``
                              dataclasses, CUDA-graph-stable buffer
                              allocator + builder, and the
                              :class:`AttentionMetadata` subclass
                              factory.
  * :mod:`.cache_manager` -- standalone side index cache used by tests
                              and the :class:`KVCacheManagerV2`
                              subclass factory.
  * :mod:`.backend`       -- the algorithm itself (vectorized
                              paged-cache helpers, prefill / decode
                              entry points, the thin
                              :class:`MiniMaxM3SparseAttention`
                              orchestrator) and the
                              :class:`AttentionBackend` subclass
                              factory.

This package's public surface re-exports the names callers
historically imported from ``...sparse.minimax_m3`` so external
importers (the model code, ``sparse.utils``, focused tests) keep
working unchanged.
"""

# Re-export the algorithm-internal helpers focused unit tests reach
# into so the package preserves the surface the monolithic module
# exposed. These are not part of ``__all__`` (still package-private)
# but stay importable as ``from ...minimax_m3 import _write_main_kv_slots``.
from .backend import (  # noqa: F401
    MiniMaxM3SparseAttention,
    _compute_index_attn_chunk_q,
    _compute_sparse_gqa_chunk_q,
    _gather_paged_batched,
    _index_attention_and_select,
    _write_main_kv_slots,
    _write_main_kv_slots_to_pool,
    get_minimax_m3_attention_backend_cls,
    minimax_m3_sparse_decode,
    minimax_m3_sparse_prefill,
)
from .cache_manager import (
    MiniMaxM3KVCacheManagerV2,
    MiniMaxM3SparseIndexCache,
    get_minimax_m3_kv_cache_manager_cls,
)
from .metadata import (
    MiniMaxM3SparseAttentionMetadata,
    MiniMaxM3SparseConfig,
    allocate_minimax_m3_static_buffers,
    build_runtime_metadata_from_kv_manager,
    get_minimax_m3_attention_metadata_cls,
    replace_metadata,
)
from .msa_backend import (
    get_minimax_m3_attention_backend_cls_with_msa,
    get_minimax_m3_msa_attention_backend_cls,
    minimax_m3_msa_sparse_decode,
    minimax_m3_msa_sparse_prefill,
)

__all__ = [
    "MiniMaxM3KVCacheManagerV2",
    "MiniMaxM3SparseAttention",
    "MiniMaxM3SparseAttentionMetadata",
    "MiniMaxM3SparseConfig",
    "MiniMaxM3SparseIndexCache",
    "allocate_minimax_m3_static_buffers",
    "build_runtime_metadata_from_kv_manager",
    "get_minimax_m3_attention_backend_cls",
    "get_minimax_m3_attention_backend_cls_with_msa",
    "get_minimax_m3_attention_metadata_cls",
    "get_minimax_m3_kv_cache_manager_cls",
    "get_minimax_m3_msa_attention_backend_cls",
    "minimax_m3_msa_sparse_decode",
    "minimax_m3_msa_sparse_prefill",
    "minimax_m3_sparse_decode",
    "minimax_m3_sparse_prefill",
    "replace_metadata",
]
