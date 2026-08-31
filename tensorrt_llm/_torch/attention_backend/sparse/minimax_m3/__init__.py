# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 sparse attention package.

Layered as:

  * :mod:`.triton_metadata` -- ``MiniMaxM3TritonSparseAttentionMetadata``
                                dataclass, CUDA-graph-stable buffer
                                allocator + builder, and the
                                :class:`AttentionMetadata` subclass
                                factory for the Triton reference path.
  * :mod:`.cache_manager`   -- standalone side index cache used by tests
                                and the :class:`KVCacheManagerV2`
                                subclass factory. Shared by both backends.
  * :mod:`.common`          -- backend-neutral config bundles, the paged
                                KV-slot writer, block-priority sentinels, and
                                the paged-cache slot mapping builder shared by
                                both backends.
  * :mod:`.msa_utils`       -- MSA-only (fmha_sm100) helpers: import guard,
                                kernel precondition constants, HND paged-cache
                                adapters, main-KV writer, page-table builder,
                                valid-block counting, and top-k selection.
  * :mod:`.triton_kernels`  -- OpenAI Triton kernels (per-block max
                                score, masked softmax for sparse GQA).
  * :mod:`.triton_backend`  -- the Triton reference algorithm (vectorized
                                paged-cache helpers, prefill / decode
                                entry points, the thin
                                :class:`MiniMaxM3TritonSparseAttention`
                                orchestrator) and its
                                :class:`AttentionBackend` subclass
                                factory.
  * :mod:`.msa_backend`     -- the MSA (fmha_sm100) backend, its flat
                                metadata, and the backend factory.
  * :mod:`.msa_indexer`     -- the MSA proxy scoring + top-k block
                                selection submodule.
  * :mod:`.msa_availability`-- SM100 and fmha_sm100 gating for the MSA
                                path.

This package's public surface re-exports the names callers
historically imported from ``...sparse.minimax_m3`` so external
importers (the model code, ``sparse.utils``, focused tests) keep
working unchanged.
"""

# The dense Triton oracle in the model imports these paged-cache helpers, so
# they stay importable from the package. They are package-private and are not
# part of __all__. Every other backend/metadata/config symbol is imported
# directly from its defining submodule by the code that needs it.
from .cache_manager import MiniMaxM3KVCacheManagerV2
from .msa_backend import MiniMaxM3MsaSparseAttention
from .triton_backend import (  # noqa: F401
    MiniMaxM3SparseRuntimeBackend,
    _gather_paged_batched,
    _write_main_kv_slots_to_pool,
)

__all__ = [
    "MiniMaxM3KVCacheManagerV2",
    "MiniMaxM3MsaSparseAttention",
    "MiniMaxM3SparseRuntimeBackend",
]
