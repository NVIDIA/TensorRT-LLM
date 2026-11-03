# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA parameter types."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal, Optional

import torch

import tensorrt_llm
import tensorrt_llm.bindings

from ..params import SparseBackendForwardArgs, SparseMetadataParams, SparseParams

ModelConfig = tensorrt_llm.bindings.ModelConfig

if TYPE_CHECKING:
    pass

# Set while a Kimi K3.1 model (Kimi K3 MLA layers with a DSA indexer) is built in this process; its tuned DSA
# decode / prefill paths key on it so that DeepSeek-V3.2, GLM-5 and DeepSeek-V4 keep the generic ones.
_KIMI_K31_DSA = False


def mark_kimi_k31_dsa() -> None:
    """Record that this process serves Kimi K3.1 (called by its MLA layers before they build the indexer)."""
    global _KIMI_K31_DSA
    _KIMI_K31_DSA = True


def is_kimi_k31_dsa() -> bool:
    """Whether this process serves Kimi K3.1."""
    return _KIMI_K31_DSA


def disagg_role() -> str:
    """This server's disaggregated role: "context", "generation", or "" (aggregated serving).

    trtllm-serve exports it from --server_role.
    """
    return os.environ.get("TRTLLM_DISAGG_ROLE", "")


@dataclass(kw_only=True, slots=True)
class DSABackendForwardArgs(SparseBackendForwardArgs):
    """DSA inputs passed from the MLA module to its backend."""

    indexer_intermediates: Optional[List[torch.Tensor]] = None
    # Paged Top-K of a generation-only call that the MLA module already predicted (on its auxiliary stream,
    # concurrently with the query absorption); the backend then skips its own prediction.
    precomputed_attn_indices: Optional[torch.Tensor] = None


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
    gvr_deterministic_topk: bool = False


@dataclass(frozen=True)
class DSAParams(SparseParams):
    """DSA backend parameters."""

    algorithm: Literal["dsa"] = field(init=False, default="dsa")
    index_n_heads: Optional[int] = None
    index_head_dim: Optional[int] = None
    index_topk: Optional[int] = None
    # Rotated channels of each indexer head. None means the MLA's
    # qk_rope_head_dim; 0 means a position-free (NoPE) indexer.
    index_rope_dim: Optional[int] = None
    # Epsilon of the indexer key LayerNorm. None means DeepSeek-V3.2's 1e-6.
    index_k_norm_eps: Optional[float] = None
    indexer_max_chunk_size: Optional[int] = None
    skip_indexer_for_short_seqs: bool = True
    use_cute_dsl_topk: bool = False
    use_cute_dsl_paged_mqa_logits: bool = True
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
    # Deterministic self-sampling GVR: repeatable index set and output order
    # (only meaningful with enable_heuristic_topk=True, use_self_sampling_topk=True).
    gvr_deterministic_topk: bool = False
    indexer_k_dtype: Literal["fp8", "fp4", "bf16"] = "fp8"
    # Shared layers reuse the preceding full layer's top-k.
    is_full_indexer_layer: bool = True
    mtp_index_share: bool = False

    @property
    def indices_block_size(self) -> int:
        return 1
