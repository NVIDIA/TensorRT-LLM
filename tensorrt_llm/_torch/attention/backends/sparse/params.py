# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared sparse attention parameter types."""

import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch

from ..cpp_schema import cpp_field
from ..extensions import MetadataExtension

_INDEXER_MQA_LOGITS_DEFAULT_ELEM_BUDGET = 1 << 31
_INDEXER_MQA_LOGITS_BYTES_PER_ELEMENT = 4


def get_indexer_mqa_logits_elem_budget() -> int:
    """Return the per-call Indexer MQA-logits cap used by the runtime."""
    return int(
        os.environ.get(
            "TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET", _INDEXER_MQA_LOGITS_DEFAULT_ELEM_BUDGET
        )
    )


def get_indexer_mqa_logits_workspace_bytes(
    max_num_tokens: Optional[int] = None, max_seq_len: Optional[int] = None
) -> int:
    """Return the reachable maximum bytes for one FP32 MQA-logits tile."""
    elem_budget = get_indexer_mqa_logits_elem_budget()
    if max_num_tokens is not None and max_seq_len is not None:
        elem_budget = min(elem_budget, max_num_tokens * max_seq_len)
    return elem_budget * _INDEXER_MQA_LOGITS_BYTES_PER_ELEMENT


# Buffers of MTPIndexShareExtension that the requiring algorithm allocates.
_INDEX_SHARE_BUFFERS = frozenset({"block_table", "req_idx_per_token", "shared_topk_indices"})


class MTPIndexShareExtension(MetadataExtension):
    """Draft-loop state shared by the sparse algorithms that reuse one MTP selection.

    The speculative driver brackets an MTP draft loop through the setters; the
    algorithm's indexer captures its Top-K selection into ``shared_topk_indices``
    once and reuses it while ``indexer_skip_topk`` is set. The algorithm that
    requires this extension (DSA or QSA) also keeps the page tables and the
    packed-row request map here, so the shared selection and the tables it
    indexes are one state.
    """

    fields = frozenset(
        {
            "block_table",
            "in_mtp_draft_loop",
            "indexer_skip_topk",
            "mtp_num_accepted",
            "req_idx_per_token",
            "shared_topk_indices",
            "shared_topk_captured_slots",
        }
    )
    methods = frozenset(
        {
            "bind_buffers",
            "set_in_mtp_draft_loop",
            "set_mtp_num_accepted",
            "set_skip_topk",
            "reset_mtp_index_share",
        }
    )

    def __init__(self, metadata, *, source=None, **config) -> None:
        super().__init__(metadata, source=source, **config)
        # Page table of the algorithm's indexer cache; allocated by the
        # requiring algorithm.
        self.block_table: Optional[torch.Tensor] = None
        # Request row of each packed token; allocated by the requiring algorithm.
        self.req_idx_per_token: Optional[torch.Tensor] = None
        # Captured selections reused by the draft loop, and the capture slots
        # that hold a selection captured in the current loop.
        self.shared_topk_indices: Optional[torch.Tensor] = None
        self.shared_topk_captured_slots: set[int] = set()
        self.in_mtp_draft_loop = False
        self.indexer_skip_topk = False
        self.mtp_num_accepted: Optional[torch.Tensor] = None
        # A metadata copy starts outside any draft loop.
        self.shared_topk_captured_slots = set()
        self.in_mtp_draft_loop = False
        self.indexer_skip_topk = False
        self.mtp_num_accepted = None

    adopts_target_state = True
    # The draft loop rebuilds the packed-row request map; the captured
    # selections are the draft loop's own.
    draft_bound = frozenset(
        {
            "block_table",
            "in_mtp_draft_loop",
            "indexer_skip_topk",
            "mtp_num_accepted",
        }
    )
    draft_written = frozenset({"req_idx_per_token"})
    draft_private = frozenset({"shared_topk_indices", "shared_topk_captured_slots"})

    def bind_buffers(self, **buffers: Optional[torch.Tensor]) -> None:
        """Bind buffers the requiring algorithm allocated, by field name.

        Accepts ``block_table``, ``req_idx_per_token`` and
        ``shared_topk_indices``; None unbinds one.
        """
        unknown = set(buffers) - _INDEX_SHARE_BUFFERS
        if unknown:
            raise TypeError(f"MTPIndexShareExtension has no buffers {sorted(unknown)}")
        for name, buffer in buffers.items():
            setattr(self, name, buffer)

    def set_in_mtp_draft_loop(self, active: bool) -> None:
        """Mark whether a draft loop is running.

        Readiness is per capture slot: a layer may reuse only the row it
        captured itself in this loop, so every transition clears it.
        """
        self.in_mtp_draft_loop = active
        self.shared_topk_captured_slots.clear()

    def set_mtp_num_accepted(self, num_accepted: Optional[torch.Tensor]) -> None:
        """Supply the per-request accepted counts used to pick the capture row."""
        self.mtp_num_accepted = num_accepted

    def set_skip_topk(self, skip: bool) -> None:
        """Ask the indexer to reuse its captured selection instead of scoring."""
        self.indexer_skip_topk = skip

    def reset_mtp_index_share(self) -> None:
        """Clear the draft-loop state the indexer reads."""
        self.set_skip_topk(False)
        self.set_in_mtp_draft_loop(False)
        self.set_mtp_num_accepted(None)


class MTPIndexShareUser(MetadataExtension):
    """An algorithm whose indexer keeps its tables and draft-loop state in ``MTPIndexShareExtension``."""

    requires = (MTPIndexShareExtension,)
    fields = frozenset({"index_share"})

    @property
    def index_share(self) -> MTPIndexShareExtension:
        """The draft-loop state and tables shared with the MTP driver."""
        return self.metadata.ext(MTPIndexShareExtension)


def use_self_sampling_gvr(
    *,
    enable_heuristic_topk: bool,
    use_self_sampling_topk: bool,
    index_topk: int | None,
    compress_ratio: int,
    is_cute_dsl_available: bool,
    sm_version: int,
) -> bool:
    """Return whether the two-level dispatch picks the self-sampling engine.

    Shared by the indexer (per-layer TopK construction) and the attention
    metadata (prior-state allocation and warmup) so both sides of the
    dispatch agree.
    """
    return (
        enable_heuristic_topk
        and use_self_sampling_topk
        and is_cute_dsl_available
        and sm_version in (100, 103)
        and index_topk in (512, 1024, 2048)
        and compress_ratio in (1, 4)
    )


class SparseParams:
    """Base parameters for a sparse attention backend."""

    algorithm: str


class SparseMetadataParams:
    """Base parameters for sparse attention metadata."""


class SparseAlgorithmExtension(MetadataExtension):
    """Base of a sparse algorithm's metadata state; a metadata carries at most one."""

    fields = frozenset({"sparse_metadata_params"})
    init_fields = {"sparse_metadata_params": None}
    #: The algorithm's settings for allocating and updating its state.
    sparse_metadata_params: Optional[SparseMetadataParams] = None


@dataclass(kw_only=True, slots=True)
class SparseBackendForwardArgs:
    """Sparse inputs passed from an attention module to its backend."""

    # Shared by algorithms that accept precomputed top-k indices.
    topk_indices: Optional[torch.Tensor] = cpp_field(dtype=torch.int32)
    # Complete block-sparse routing payload predicted by the module before the
    # core forward; the default backend hook hands it through unchanged.
    block_sparse_inputs: Optional["BlockSparseForwardInputs"] = None


@dataclass(frozen=True, slots=True)
class BlockSparseForwardInputs:
    """Block geometry and live routing payload for one attention call.

    Exactly one routing representation is present. Canonical BSR uses
    ``block_indptr`` and ``block_indices``; packed bitmask routing uses
    ``exact_block_bits``. Paired K/V summaries enable proxy routes without
    encoding an algorithm name in this shared carrier. Both summaries are
    block means over the tokens present, including a partial final block.
    """

    q_block_size: int
    kv_block_size: int
    max_blocks_per_row: Optional[int] = None
    block_indptr: Optional[torch.Tensor] = None
    block_indices: Optional[torch.Tensor] = None
    exact_block_bits: Optional[torch.Tensor] = None
    k_summary: Optional[torch.Tensor] = None
    v_summary: Optional[torch.Tensor] = None
    kv_valid_bits: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        has_bsr = self.block_indptr is not None
        if has_bsr != (self.block_indices is not None):
            raise ValueError("block_indptr and block_indices must be provided together")
        if has_bsr == (self.exact_block_bits is not None):
            raise ValueError("exactly one route representation must be provided")
        if has_bsr and self.max_blocks_per_row is None:
            raise ValueError("BSR routes require max_blocks_per_row")
        if (self.k_summary is None) != (self.v_summary is None):
            raise ValueError("k_summary and v_summary must be provided together")

    @property
    def sparse_format(self) -> Literal["bsr", "bitmask"]:
        """Routing representation selected by the live payload."""
        return "bitmask" if self.exact_block_bits is not None else "bsr"

    @property
    def use_proxy_routes(self) -> bool:
        """Whether unselected blocks are represented by K/V summaries."""
        return self.k_summary is not None


@dataclass(kw_only=True, slots=True)
class SparseRuntimeParams:
    """Complete per-attention sparse runtime state consumed by FMHA/``AttentionOp``."""

    # Sparse index inputs shared by multiple algorithms.
    sparse_kv_indices: Optional[torch.Tensor] = cpp_field(dtype=torch.int32)
    sparse_kv_offsets: Optional[torch.Tensor] = cpp_field(dtype=torch.int32)
    sparse_attn_indices: Optional[torch.Tensor] = cpp_field(dtype=torch.int32)
    # Per-query offsets, or backend-specific secondary sparse indices
    # (DeepSeek-V4 fp8_ds_mla compressed-pool indices).
    sparse_attn_offsets: Optional[torch.Tensor] = cpp_field(dtype=torch.int32)
    sparse_attn_indices_block_size: int = 0
    sparse_attn_kv_lens: Optional[torch.Tensor] = cpp_field(dtype=torch.int32)
    aux_kv_cache_pool_ptr: Optional[int] = None

    # SkipSoftmax prefill threshold; kernels divide it by context length.
    threshold_scale_factor_prefill: float = 0.0
    # SkipSoftmax decode threshold; diffusion models leave it at zero.
    threshold_scale_factor_decode: float = 0.0
    block_sparse_inputs: Optional[BlockSparseForwardInputs] = None


__all__ = [
    "BlockSparseForwardInputs",
    "SparseBackendForwardArgs",
    "SparseMetadataParams",
    "SparseParams",
    "SparseRuntimeParams",
]
