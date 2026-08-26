# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal sparse-attention parameters for QSA."""

from dataclasses import dataclass

from ..params import SparseMetadataParams, SparseParams


@dataclass(kw_only=True, slots=True)
class QSASparseParams(SparseParams):
    """Validated runtime geometry for the QSA sparse indexer."""

    algorithm: str = "qsa"
    index_n_heads: int
    index_kv_heads: int
    index_head_dim: int
    token_topk: int
    compress_ratio: int
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
        if self.index_kv_heads != 1:
            raise ValueError("QSA sparse attention requires one index KV head")
        if self.compress_ratio < 2:
            raise ValueError("QSA sparse attention requires compress_ratio >= 2")
        if self.token_topk % self.compress_ratio != 0:
            raise ValueError("QSA token_topk must be divisible by compress_ratio")

    @property
    def block_topk(self) -> int:
        return self.token_topk // self.compress_ratio

    @property
    def expanded_topk(self) -> int:
        return self.token_topk + self.compress_ratio - 1

    @property
    def indices_block_size(self) -> int:
        return 1

    @property
    def dense_seq_len_threshold(self) -> int:
        """Return the effective dense cutoff for a sparse-attention batch."""
        assert self.seq_len_threshold is not None
        return max(self.token_topk, self.seq_len_threshold)


@dataclass(kw_only=True, slots=True)
class QSASparseMetadataParams(SparseMetadataParams):
    """Layer-invariant metadata geometry for the QSA backend."""

    token_topk: int
    compress_ratio: int
