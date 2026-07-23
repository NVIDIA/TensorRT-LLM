# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA parameter types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal, Optional

import torch

import tensorrt_llm
import tensorrt_llm.bindings

from ..params import SparseBackendForwardArgs, SparseMetadataParams, SparseParams

ModelConfig = tensorrt_llm.bindings.ModelConfig

if TYPE_CHECKING:
    pass


@dataclass(kw_only=True, slots=True)
class DSABackendForwardArgs(SparseBackendForwardArgs):
    """DSA inputs passed from the MLA module to its backend."""

    indexer_intermediates: Optional[List[torch.Tensor]] = None


@dataclass(frozen=True)
class DSAMetadataParams(SparseMetadataParams):
    """DSA metadata parameters."""

    indexer_max_chunk_size: int
    max_sparse_topk: Optional[int]
    index_head_dim: int
    enable_indexer_skip: bool
    enable_heuristic_topk: bool
    use_cute_dsl_topk: bool
    use_cute_dsl_paged_mqa_logits: bool
    q_split_threshold: int
    has_shared_indexer_layers: bool = False


@dataclass(frozen=True)
class DSAParams(SparseParams):
    """DSA backend parameters."""

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
    # Shared layers reuse the preceding full layer's top-k.
    is_full_indexer_layer: bool = True

    @property
    def indices_block_size(self) -> int:
        return 1
