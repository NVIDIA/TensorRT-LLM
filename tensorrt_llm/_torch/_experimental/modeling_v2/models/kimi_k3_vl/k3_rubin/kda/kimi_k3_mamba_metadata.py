# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-prepared scheduling metadata for packed KDA prefill.

Kimi K3 Rubin merge: main's FlashInfer KDA prefill metadata (``kda_max_seqlen``,
``kda_seq_order``, ``kda_cu_chunks``, the 16-byte aligned ``generation_state_indices``)
plus the Rubin agentic branch's metadata for its CuTe DSL prefill (the package's default
prefill, ``trtllm::k3r_kda_prefill``): the FLA-style ``kda_chunk_indices`` (filled on the
device), ``kda_varlen_is_aligned`` / ``kda_single_sequence_length``, chunk alignment 64 for
the warmup, the int64 ``state_indices_long`` mirror (staged through the ULL input commit where
the engine has one) and the mid-chunk snapshot plan (``kda_snapshot``, built from the cache
manager's ``get_kda_midchunk_snapshots``: ``KimiK3HybridCacheManager`` plans snapshots with
TRTLLM_KDA_MIDCHUNK_SNAPSHOT=1 and commits only the ones fetched here; otherwise it stays
``None``).
"""

import sys
from dataclasses import dataclass
from itertools import accumulate
from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._utils import prefer_pinned

# Granularity of the branch's mid-chunk snapshot offsets (the CuTe DSL prefill kernel's chunk).
KDA_PREFILL_CHUNK_SIZE = 64

# The engine's batched input commit module (TRTLLM_BATCHED_INPUT_COMMIT / TRTLLM_K3_ULL_STEP_PREP).
_INPUT_COMMIT_MODULE = "tensorrt_llm._torch.pyexecutor.input_commit"


def _active_input_commit():
    """The executor's open input commit, or ``None``.

    The module is looked up rather than imported, so the package also runs on engines without
    it. Only metadata whose ``state_indices`` were staged into the commit
    (``state_indices_staged``) stages the int64 mirror too; ``Mamba2Metadata`` uploads the state
    indices with a direct copy, so with it ``state_indices`` is already written, in stream order,
    when the mirror is copied.
    """
    module = sys.modules.get(_INPUT_COMMIT_MODULE)
    return None if module is None else module.active_input_commit()


@dataclass
class KDASnapshotPlan:
    """Mid-chunk recurrent-state snapshots requested for one context batch.

    Snapshot ``j`` (ordered by context sequence) makes the KDA layers store
    its sequence's recurrent state and convolution window after the
    sequence's first ``offset_j`` tokens of this step (a multiple of
    ``KDA_PREFILL_CHUNK_SIZE``, strictly inside the step) into pool slot
    ``slots[j]``. Per-sequence vectors hold ``-1`` for sequences without a
    snapshot. All tensors live on the device.
    """

    num_snapshots: int
    # int64 [n]: pool slot that receives each snapshot.
    slots: torch.Tensor
    # int64 [n]: packed token index of the first token after each snapshot point.
    token_indices: torch.Tensor
    # int32 [num_contexts]: KDA chunk at whose start the snapshot is taken.
    chunks: torch.Tensor
    # int32 [num_contexts]: snapshot row in the pool itself (fp32 pools).
    pool_rows: torch.Tensor
    # int32 [num_contexts]: snapshot row in a staged copy holding the
    # sequences' rows first and the snapshot rows after them (bf16 pools).
    staged_rows: torch.Tensor


def build_kda_snapshot_plan(
    snapshots: List[Tuple[int, int, int]],
    context_lengths: List[int],
    device: torch.device,
) -> Optional[KDASnapshotPlan]:
    """Build the device plan for ``(context index, slot, offset)`` snapshots."""
    if not snapshots:
        return None
    num_contexts = len(context_lengths)
    starts = [0] * num_contexts
    for i in range(1, num_contexts):
        starts[i] = starts[i - 1] + context_lengths[i - 1]
    chunks = [-1] * num_contexts
    pool_rows = [-1] * num_contexts
    staged_rows = [-1] * num_contexts
    slots, token_indices = [], []
    for j, (seq, slot, offset) in enumerate(sorted(snapshots)):
        if not 0 <= seq < num_contexts or chunks[seq] != -1:
            raise ValueError(f"Invalid KDA snapshot sequence index {seq}")
        if offset % KDA_PREFILL_CHUNK_SIZE or not 0 < offset < context_lengths[seq]:
            raise ValueError(
                f"KDA snapshot offset {offset} must be a multiple of "
                f"{KDA_PREFILL_CHUNK_SIZE} inside the {context_lengths[seq]}-token step"
            )
        chunks[seq] = offset // KDA_PREFILL_CHUNK_SIZE
        pool_rows[seq] = slot
        staged_rows[seq] = num_contexts + j
        slots.append(slot)
        token_indices.append(starts[seq] + offset)
    num_snapshots = len(slots)
    pin = device.type == "cuda" and prefer_pinned()
    snapshot_vectors = torch.tensor([*slots, *token_indices], dtype=torch.int64, pin_memory=pin).to(
        device, non_blocking=True
    )
    # The kernel reads the per-sequence vectors as 16-byte aligned int32 arrays.
    stride = (num_contexts + 3) // 4 * 4
    padding = [-1] * (stride - num_contexts)
    sequence_vectors = torch.tensor(
        [*chunks, *padding, *pool_rows, *padding, *staged_rows, *padding],
        dtype=torch.int32,
        pin_memory=pin,
    ).to(device, non_blocking=True)
    return KDASnapshotPlan(
        num_snapshots=num_snapshots,
        slots=snapshot_vectors[:num_snapshots],
        token_indices=snapshot_vectors[num_snapshots:],
        chunks=sequence_vectors[:num_contexts],
        pool_rows=sequence_vectors[stride : stride + num_contexts],
        staged_rows=sequence_vectors[2 * stride : 2 * stride + num_contexts],
    )


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
    """Reuse stable int32 slots/starts and int64 offsets from Mamba metadata."""

    # The CuTe DSL prefill kernel keys its compile cache on chunk alignment (64); the warmup
    # primes both sides of it. A 64-aligned length is 16-aligned too, so main's FlashInfer
    # prefill (alignment 16; the fallback) gets both of its variants from the same two passes.
    prefill_chunk_alignment = KDA_PREFILL_CHUNK_SIZE

    def __init__(self, max_batch_size: int, chunk_size: int, max_num_tokens: int) -> None:
        super().__init__(max_batch_size, chunk_size, max_num_tokens)
        self.kda_max_seqlen = 0
        # Align both rows to 16 bytes. One transfer refreshes the metadata shared
        # by all layers; device storage stays fixed for CUDA graph replay.
        width = ((max_batch_size + 1 + 3) // 4) * 4
        self._kda_schedule = torch.zeros((2, width), dtype=torch.int32, device="cuda")
        self.kda_seq_order = self._kda_schedule[0, :0]
        self.kda_cu_chunks = self._kda_schedule[1, :1]

        # KDA's CuTe verify kernel requires a 16-byte-aligned DLPack pointer.
        # Keep the backing allocation stable so metadata preparation remains
        # safe for CUDA graph capture and replay.
        self._generation_state_indices = torch.empty(
            max_batch_size, dtype=torch.int32, device="cuda"
        )
        self.generation_state_indices: torch.Tensor | None = None

        # Rubin agentic branch: an int64 mirror of the state indices, refreshed once per step.
        self._state_indices_long = torch.zeros(max_batch_size, dtype=torch.long, device="cuda")
        self.state_indices_long = self._state_indices_long[:0]
        # The CuTe DSL prefill's varlen chunk table ([num_chunks, 2]: sequence, chunk in it),
        # filled on the device once per step instead of in each KDA layer.
        self._kda_chunk_indices = torch.empty((0, 2), dtype=torch.long, device="cuda")
        self.kda_chunk_indices: torch.Tensor | None = None
        self.kda_varlen_is_aligned: bool | None = None
        self.kda_single_sequence_length: int | None = None
        self.kda_snapshot: KDASnapshotPlan | None = None

    def prepare(self, attn_metadata: AttentionMetadata) -> None:
        context_lengths = attn_metadata.seq_lens[: attn_metadata.num_contexts].tolist()
        assert all(length > 0 for length in context_lengths), (
            f"Expected positive context lengths, got {context_lengths}"
        )

        super().prepare(attn_metadata)
        self.kda_snapshot = None

        self.generation_state_indices = None
        batch_size = attn_metadata.seq_lens.shape[0]
        input_commit = _active_input_commit()
        if input_commit is not None and getattr(self, "state_indices_staged", False):
            # The int32 indices are written by the commit too; stage the same values.
            input_commit.stage(
                self._state_indices_long[:batch_size], self.state_indices_cpu[:batch_size]
            )
        else:
            self._state_indices_long[:batch_size].copy_(self.state_indices[:batch_size])
        self.state_indices_long = self._state_indices_long[:batch_size]

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

        self.kda_max_seqlen = max(context_lengths, default=0)
        n = len(context_lengths)
        self.kda_seq_order = self._kda_schedule[0, :n]
        self.kda_cu_chunks = self._kda_schedule[1, : n + 1]
        if not n:
            self.kda_chunk_indices = None
            return

        # Mid-chunk snapshots (TRTLLM_KDA_MIDCHUNK_SNAPSHOT): duck-typed on the cache manager.
        # KimiK3HybridCacheManager returns this step's planned snapshots, and commits a pending
        # snapshot page only when its plan was fetched here.
        get_snapshots = getattr(attn_metadata.kv_cache_manager, "get_kda_midchunk_snapshots", None)
        if get_snapshots is not None and attn_metadata.request_ids is not None:
            self.kda_snapshot = build_kda_snapshot_plan(
                get_snapshots(attn_metadata.request_ids[:n]),
                context_lengths,
                self._state_indices_long.device,
            )

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

        # The CuTe DSL prefill's varlen chunk table (the branch's, 64-token chunks).
        num_chunks = sum(
            (length + KDA_PREFILL_CHUNK_SIZE - 1) // KDA_PREFILL_CHUNK_SIZE
            for length in context_lengths
        )
        if self._kda_chunk_indices.shape[0] < num_chunks:
            self._kda_chunk_indices = torch.empty((num_chunks, 2), dtype=torch.long, device="cuda")
        self.kda_chunk_indices = self._kda_chunk_indices[:num_chunks]
        _prepare_kda_chunk_indices(attn_metadata.seq_lens_cuda[:n], self.kda_chunk_indices)
