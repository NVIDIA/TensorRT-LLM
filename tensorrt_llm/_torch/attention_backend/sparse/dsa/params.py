# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense Sparse Attention (DSA) backend for TRT-LLM with indexer-based TopK selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

import tensorrt_llm
import tensorrt_llm.bindings

from ..params import SparseMetadataParams, SparseParams

ModelConfig = tensorrt_llm.bindings.ModelConfig

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class DSAMetadataParams(SparseMetadataParams):
    """DSA metadata settings derived from user config."""

    indexer_max_chunk_size: int
    max_sparse_topk: Optional[int]
    enable_indexer_skip: bool
    enable_heuristic_topk: bool
    use_cute_dsl_paged_mqa_logits: bool
    q_split_threshold: int


@dataclass(frozen=True)
class DSAParams(SparseParams):
    """DSA sparse attention backend parameters."""

    algorithm: Literal["dsa"] = field(init=False, default="dsa")
    index_n_heads: Optional[int] = None
    index_head_dim: Optional[int] = None
    index_topk: Optional[int] = None
    indexer_max_chunk_size: Optional[int] = None
    skip_indexer_for_short_seqs: bool = True
    use_cute_dsl_topk: bool = False
    use_cute_dsl_paged_mqa_logits: bool = False
    q_split_threshold: int = 8192
    indexer_rope_interleave: bool = False
    enable_heuristic_topk: bool = False
    indexer_k_dtype: Literal["fp8", "fp4"] = "fp8"
    # Cross-layer indexer sharing: whether this layer runs its own indexer
    # ("full") or reuses the previous full layer's top-k ("shared"). Always
    # True for a dense per-layer indexer (e.g. DeepSeek-V3.2).
    is_full_indexer_layer: bool = True

    @property
    def indices_block_size(self) -> int:
        return 1
