# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal sparse-attention parameters for QSA."""

from dataclasses import dataclass

from ..params import SparseMetadataParams, SparseParams
from .constants import QSA_INDEX_KV_HEADS


def _validate_selection_geometry(token_topk: int, compress_ratio: int) -> None:
    if token_topk <= 0:
        raise ValueError("QSA token_topk must be positive")
    if compress_ratio < 2:
        # Ratio one provides no sparse-index reduction and is not a supported
        # QSA selection geometry.
        raise ValueError("QSA sparse attention requires compress_ratio >= 2")
    if token_topk % compress_ratio != 0:
        raise ValueError("QSA token_topk must be divisible by compress_ratio")


@dataclass(kw_only=True, slots=True)
class QSASparseParams(SparseParams):
    """Runtime QSA geometry resolved from the checkpoint's ``indexer_*`` fields."""

    # Sparse-attention registry identifier.
    algorithm: str = "qsa"
    # Number of query heads used to score candidate token groups.
    index_n_heads: int
    # Number of key heads used by the indexer; QSA currently requires one.
    index_kv_heads: int
    # Per-head dimension of the indexer's query and key projections.
    index_head_dim: int
    # Sparse selection budget, expressed in original-token units.
    token_topk: int
    # Number of consecutive original tokens represented by one indexed group.
    compress_ratio: int
    # Requested dense cutoff; the effective cutoff never falls below token_topk.
    seq_len_threshold: int | None = None

    def __post_init__(self) -> None:
        values = {
            "index_n_heads": self.index_n_heads,
            "index_kv_heads": self.index_kv_heads,
            "index_head_dim": self.index_head_dim,
            "token_topk": self.token_topk,
            "compress_ratio": self.compress_ratio,
        }
        if any(value <= 0 for value in values.values()):
            raise ValueError(f"QSA sparse parameters must be positive: {values}")
        if self.seq_len_threshold is None:
            self.seq_len_threshold = self.token_topk
        elif self.seq_len_threshold <= 0:
            raise ValueError("QSA seq_len_threshold must be positive")
        if self.index_kv_heads != QSA_INDEX_KV_HEADS:
            raise ValueError("QSA sparse attention requires one index KV head")
        _validate_selection_geometry(self.token_topk, self.compress_ratio)

    @property
    def block_topk(self) -> int:
        """Return the number of compressed groups selected per query."""
        return self.token_topk // self.compress_ratio

    @property
    def expanded_topk(self) -> int:
        """Return the maximum raw-token width after group expansion."""
        return self.token_topk + self.compress_ratio - 1

    @property
    def indices_block_size(self) -> int:
        """Return one because each expanded index addresses one raw token."""
        return 1

    @property
    def dense_seq_len_threshold(self) -> int:
        """Return a cutoff no smaller than the raw-token selection budget."""
        if self.seq_len_threshold is None:
            raise RuntimeError("QSA sequence threshold was not initialized")
        # Below token_topk, QSA can select the entire prefix but adds indexer
        # overhead without reducing attention work, so retain dense attention.
        return max(self.token_topk, self.seq_len_threshold)


@dataclass(kw_only=True, slots=True)
class QSASparseMetadataParams(SparseMetadataParams):
    """Layer-invariant metadata geometry for the QSA backend."""

    # Sparse selection budget, expressed in original-token units.
    token_topk: int
    # Number of consecutive original tokens represented by one indexed group.
    compress_ratio: int

    def __post_init__(self) -> None:
        _validate_selection_geometry(self.token_topk, self.compress_ratio)

    @property
    def block_topk(self) -> int:
        """Return the fixed compressed-group output width for Top-K."""
        return self.token_topk // self.compress_ratio
