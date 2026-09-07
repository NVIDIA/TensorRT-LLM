# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3-specific recurrent-state metadata."""

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata

KDA_PREFILL_CHUNK_SIZE = 64


@torch.compile(dynamic=True, fullgraph=True)
def _prepare_kda_chunk_indices(
    sequence_lengths: torch.Tensor,
    chunk_indices: torch.Tensor,
) -> torch.Tensor:
    """Fill FLA-compatible chunk indices without reading CUDA counts on CPU."""
    chunk_counts = (sequence_lengths + KDA_PREFILL_CHUNK_SIZE - 1).div(
        KDA_PREFILL_CHUNK_SIZE, rounding_mode="floor"
    )
    num_chunks = chunk_indices.shape[0]
    # Supplying output_size keeps repeat_interleave from synchronizing to read
    # sum(chunk_counts) from the device.
    sequence_indices = torch.repeat_interleave(
        torch.arange(sequence_lengths.shape[0], dtype=torch.long, device=sequence_lengths.device),
        chunk_counts,
        output_size=num_chunks,
    )
    chunk_starts = torch.cumsum(chunk_counts, dim=0) - chunk_counts
    local_chunk_indices = (
        torch.arange(num_chunks, dtype=torch.long, device=sequence_lengths.device)
        - chunk_starts[sequence_indices]
    )
    chunk_indices[:, 0].copy_(sequence_indices)
    chunk_indices[:, 1].copy_(local_chunk_indices)
    return chunk_indices


class KimiK3MambaMetadata(Mamba2Metadata):
    """Mamba metadata extended with Kimi K3 KDA preparation."""

    def __init__(self, max_batch_size: int, chunk_size: int, max_num_tokens: int) -> None:
        super().__init__(max_batch_size, chunk_size, max_num_tokens)
        # Every chunk contains at least one token, so this is a hard upper bound.
        self._kda_chunk_indices = torch.empty((max_num_tokens, 2), dtype=torch.long, device="cuda")
        self.kda_chunk_indices: torch.Tensor | None = None
        self.kda_varlen_is_aligned: bool | None = None
        self.kda_single_sequence_length: int | None = None

    def prepare(self, attn_metadata: AttentionMetadata) -> None:
        context_lengths = attn_metadata.seq_lens[: attn_metadata.num_contexts].tolist()
        assert all(length > 0 for length in context_lengths), (
            f"Expected positive context lengths, got {context_lengths}"
        )

        super().prepare(attn_metadata)

        self.kda_varlen_is_aligned = (
            all(length % KDA_PREFILL_CHUNK_SIZE == 0 for length in context_lengths)
            if context_lengths
            else None
        )
        self.kda_single_sequence_length = context_lengths[0] if len(context_lengths) == 1 else None

        if not context_lengths:
            self.kda_chunk_indices = None
            return

        chunk_counts = [
            (length + KDA_PREFILL_CHUNK_SIZE - 1) // KDA_PREFILL_CHUNK_SIZE
            for length in context_lengths
        ]
        num_chunks = sum(chunk_counts)
        assert num_chunks <= self._kda_chunk_indices.shape[0], (
            f"Expected at most {self._kda_chunk_indices.shape[0]} KDA chunks, got {num_chunks}"
        )
        self.kda_chunk_indices = self._kda_chunk_indices[:num_chunks]

        _prepare_kda_chunk_indices(
            attn_metadata.seq_lens_cuda[: len(context_lengths)],
            self.kda_chunk_indices,
        )
