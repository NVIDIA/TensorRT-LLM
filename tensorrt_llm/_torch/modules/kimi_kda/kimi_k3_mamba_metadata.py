# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3-specific recurrent-state metadata."""

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._utils import prefer_pinned

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

    # The KDA prefill kernel keys its compile cache on chunk alignment.
    prefill_chunk_alignment = KDA_PREFILL_CHUNK_SIZE

    def __init__(self, max_batch_size: int, chunk_size: int, max_num_tokens: int) -> None:
        super().__init__(max_batch_size, chunk_size, max_num_tokens)
        # Every chunk contains at least one token, so this is a hard upper bound.
        self._kda_chunk_indices = torch.empty((max_num_tokens, 2), dtype=torch.long, device="cuda")
        self.kda_chunk_indices: torch.Tensor | None = None
        self.kda_varlen_is_aligned: bool | None = None
        self.kda_single_sequence_length: int | None = None
        # KDA's CuTe verify kernel requires a 16-byte-aligned DLPack pointer.
        # Keep the backing allocation stable so metadata preparation remains
        # safe for CUDA graph capture and replay.
        self._generation_state_indices = torch.empty(
            max_batch_size, dtype=torch.int32, device="cuda"
        )
        self.generation_state_indices: torch.Tensor | None = None

        # KDA accepted-draft counts in generation-batch order. The backing
        # allocation stays fixed across CUDA graph replays.
        self.prev_num_accepted_tokens = torch.zeros(
            max_batch_size, dtype=torch.int32, device="cuda"
        )
        self._prev_count_staging = torch.empty(
            (max_batch_size, 2), dtype=torch.int32, device="cuda"
        )

    def prepare_prev_num_accepted_tokens(
        self,
        accepted_draft_tokens: list[int],
        previous_slots: list[int | None],
        *,
        previous_new_tokens_lens: torch.Tensor | None = None,
    ) -> None:
        """Fill the stable replay buffer from GPU sampler slots or CPU counts.

        ``accepted_draft_tokens`` contains CPU fallback counts for the full
        generation batch. Each ``previous_slots`` entry selects a GPU sampler
        result, or is None to use that row's CPU count. Unused buffer rows
        are cleared.
        """
        num_generations = len(accepted_draft_tokens)
        assert len(previous_slots) == num_generations <= self.max_batch_size
        self.prev_num_accepted_tokens.zero_()
        if num_generations == 0:
            return

        # Pack counts and slots into one contiguous asynchronous H2D copy.
        # Fresh pinned storage cannot be overwritten by later preparation
        # while the upload is pending.
        staging = torch.tensor(
            [
                (count, -1 if slot is None else slot)
                for count, slot in zip(accepted_draft_tokens, previous_slots)
            ],
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
        )
        device_inputs = self._prev_count_staging[:num_generations]
        device_inputs.copy_(staging, non_blocking=True)
        host_counts, slots = device_inputs.unbind(dim=1)
        counts = self.prev_num_accepted_tokens[:num_generations]
        if any(slot is not None for slot in previous_slots):
            assert previous_new_tokens_lens is not None
            sampled = previous_new_tokens_lens.index_select(0, slots.clamp(min=0))
            # Sampler lengths include the golden token; replay only the drafts.
            sampled.sub_(1).clamp_(min=0)
            torch.where(slots >= 0, sampled, host_counts, out=counts)
        else:
            counts.copy_(host_counts, non_blocking=True)

    def prepare(self, attn_metadata: AttentionMetadata) -> None:
        context_lengths = attn_metadata.seq_lens[: attn_metadata.num_contexts].tolist()
        assert all(length > 0 for length in context_lengths), (
            f"Expected positive context lengths, got {context_lengths}"
        )

        super().prepare(attn_metadata)

        self.generation_state_indices = None
        batch_size = attn_metadata.seq_lens.shape[0]
        generation_state_indices = self.state_indices[attn_metadata.num_contexts : batch_size]
        if (
            getattr(attn_metadata.kv_cache_manager, "use_kda_replay_update", False)
            and generation_state_indices.numel() > 0
            and generation_state_indices.data_ptr() % 16
        ):
            aligned_state_indices = self._generation_state_indices[
                : generation_state_indices.shape[0]
            ]
            aligned_state_indices.copy_(generation_state_indices)
            self.generation_state_indices = aligned_state_indices

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
