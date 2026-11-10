# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-prepared KDA metadata: packed FlashInfer prefill schedule and replay slots."""

from itertools import accumulate

import torch

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._utils import prefer_pinned


class KimiK3MambaMetadata(Mamba2Metadata):
    """Reuse stable int32 slots/starts and int64 offsets from Mamba metadata."""

    prefill_chunk_alignment = 16

    def __init__(self, max_batch_size: int, chunk_size: int, max_num_tokens: int) -> None:
        super().__init__(max_batch_size, chunk_size, max_num_tokens)
        # Each request's slot in the KDA replay buffers. A cache manager that
        # keeps replay state in a region of its own fills this static buffer;
        # otherwise the replay buffers share the recurrent-state slots and
        # this aliases ``state_indices``.
        self.replay_state_indices = self.state_indices
        self._replay_state_indices = torch.zeros(max_batch_size, dtype=torch.int32, device="cuda")
        self._replay_state_indices_cpu = torch.zeros(
            max_batch_size, dtype=torch.int32, pin_memory=prefer_pinned()
        )
        self._replay_state_indices_buffer = DeviceCopyBuffer(
            self._replay_state_indices, pin_memory=prefer_pinned()
        )
        self.kda_max_seqlen = 0
        # Align both rows to 16 bytes. One transfer refreshes the metadata shared
        # by all layers; device storage stays fixed for CUDA graph replay.
        width = ((max_batch_size + 1 + 3) // 4) * 4
        self._kda_schedule = torch.zeros((2, width), dtype=torch.int32, device="cuda")
        self.kda_seq_order = self._kda_schedule[0, :0]
        self.kda_cu_chunks = self._kda_schedule[1, :1]

        # KDA's CuTe verify kernel requires a 16-byte-aligned DLPack pointer.
        # Keep the backing allocations stable so metadata preparation remains
        # safe for CUDA graph capture and replay.
        self._generation_state_indices = torch.empty(
            max_batch_size, dtype=torch.int32, device="cuda"
        )
        self.generation_state_indices: torch.Tensor | None = None
        # The generation requests' replay slots, aligned the same way when
        # they are not the state slots.
        self._generation_replay_state_indices = torch.empty(
            max_batch_size, dtype=torch.int32, device="cuda"
        )
        self.generation_replay_state_indices: torch.Tensor | None = None

    def prepare(self, attn_metadata: AttentionMetadata) -> None:
        context_lengths = attn_metadata.seq_lens[: attn_metadata.num_contexts].tolist()
        assert all(length > 0 for length in context_lengths), (
            f"Expected positive context lengths, got {context_lengths}"
        )

        super().prepare(attn_metadata)
        self._prepare_replay_state_indices(attn_metadata)

        self.generation_state_indices = None
        self.generation_replay_state_indices = None
        batch_size = attn_metadata.seq_lens.shape[0]
        num_contexts = attn_metadata.num_contexts
        if (
            getattr(attn_metadata.kv_cache_manager, "use_kda_replay_update", False)
            and batch_size > num_contexts
        ):
            self.generation_state_indices = self._aligned_generation_indices(
                self.state_indices, self._generation_state_indices, num_contexts, batch_size
            )
            if self.replay_state_indices is self.state_indices:
                self.generation_replay_state_indices = self.generation_state_indices
            else:
                self.generation_replay_state_indices = self._aligned_generation_indices(
                    self.replay_state_indices,
                    self._generation_replay_state_indices,
                    num_contexts,
                    batch_size,
                )

        self.kda_max_seqlen = max(context_lengths, default=0)
        n = len(context_lengths)
        self.kda_seq_order = self._kda_schedule[0, :n]
        self.kda_cu_chunks = self._kda_schedule[1, : n + 1]
        if not n:
            return
        order = sorted(range(n), key=lambda i: (-context_lengths[i], i))
        chunks = list(accumulate(((length + 15) // 16 for length in context_lengths), initial=0))
        width = self._kda_schedule.shape[1]
        # Use a fresh pinned source so another prepare cannot overwrite an
        # in-flight transfer. The pinned allocator retains it until DMA ends.
        host_schedule = torch.tensor(
            [order + [0] * (width - n), chunks + [0] * (width - n - 1)],
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self._kda_schedule.copy_(host_schedule, non_blocking=True)

    def _prepare_replay_state_indices(self, attn_metadata: AttentionMetadata) -> None:
        """Fill the static replay-slot buffer, or alias the state slots."""
        kv_cache_manager = attn_metadata.kv_cache_manager
        request_ids = attn_metadata.request_ids
        if getattr(kv_cache_manager, "replay_layer_group_id", None) is None or request_ids is None:
            self.replay_state_indices = self.state_indices
            return
        batch_size = attn_metadata.seq_lens.shape[0]
        indices = kv_cache_manager.get_replay_state_indices(request_ids[:batch_size])
        assert len(indices) == batch_size, (
            f"get_replay_state_indices() returned {len(indices)} entries for "
            f"a batch of {batch_size} requests."
        )
        self._replay_state_indices_cpu[:batch_size].copy_(
            torch.as_tensor(indices, dtype=self._replay_state_indices_cpu.dtype)
        )
        self._replay_state_indices_buffer.update(self._replay_state_indices_cpu[:batch_size])
        self.replay_state_indices = self._replay_state_indices

    @staticmethod
    def _aligned_generation_indices(
        indices: torch.Tensor, aligned: torch.Tensor, num_contexts: int, batch_size: int
    ) -> torch.Tensor | None:
        """Copy ``indices[num_contexts:batch_size]`` into ``aligned`` when it is misaligned.

        Returns the aligned copy, or ``None`` when the slice is already 16-byte aligned.
        """
        generation_indices = indices[num_contexts:batch_size]
        if generation_indices.data_ptr() % 16 == 0:
            return None
        aligned_indices = aligned[: generation_indices.shape[0]]
        aligned_indices.copy_(generation_indices)
        return aligned_indices
