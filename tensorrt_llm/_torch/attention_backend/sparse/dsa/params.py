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


def is_gvr_cute_dsl_supported(
    *,
    is_cute_dsl_available: bool,
    is_cute_dsl_rubin_available: bool,
    sm_version: int,
    use_self_sampling_topk: bool,
) -> bool:
    """Return whether the CuTe DSL stack can run GVR on this architecture."""
    return is_cute_dsl_available and (
        sm_version in (100, 103)
        or (sm_version == 107 and is_cute_dsl_rubin_available and use_self_sampling_topk)
    )


def use_self_sampling_gvr(
    *,
    enable_heuristic_topk: bool,
    use_self_sampling_topk: bool,
    index_topk: int | None,
    compress_ratio: int,
    is_cute_dsl_available: bool,
    sm_version: int,
    is_cute_dsl_rubin_available: bool = False,
) -> bool:
    """Return whether the two-level dispatch picks the self-sampling engine.

    Shared by the indexer (per-layer TopK construction) and the attention
    metadata (prior-state allocation and warmup) so both sides of the
    dispatch agree.
    """
    return (
        enable_heuristic_topk
        and use_self_sampling_topk
        and is_gvr_cute_dsl_supported(
            is_cute_dsl_available=is_cute_dsl_available,
            is_cute_dsl_rubin_available=is_cute_dsl_rubin_available,
            sm_version=sm_version,
            use_self_sampling_topk=True,
        )
        and index_topk in (512, 1024, 2048)
        and compress_ratio in (1, 4)
    )


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
    mtp_index_share: bool = False
    use_self_sampling_topk: bool = True
    use_gvr_emission: bool = False
    use_gvr_locality_domain: bool = False


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
    # Second-level GVR dispatch: hint-free self-sampling engine (True) vs
    # temporal previous-step-hint engines (False). Only meaningful when
    # enable_heuristic_topk is set.
    use_self_sampling_topk: bool = True
    # Emission block-skip for the temporal-hint engine; only meaningful with
    # enable_heuristic_topk=True and use_self_sampling_topk=False on FP4.
    use_gvr_emission: bool = False
    # Prototype Rubin-only row sharding for self-sampling GVR V2. The logits
    # producer remains full-device; only non-overlapping Top-K row slices are
    # submitted to the two locality-domain streams.
    use_gvr_locality_domain: bool = False
    indexer_k_dtype: Literal["fp8", "fp4"] = "fp8"
    # Shared layers reuse the preceding full layer's top-k.
    is_full_indexer_layer: bool = True
    mtp_index_share: bool = False

    @property
    def indices_block_size(self) -> int:
        return 1
