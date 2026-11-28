# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA metadata of the Kimi K3.1 Rubin targets: the engine's packed-prefill metadata plus the
step's mid-chunk recurrent-state snapshot plan.

A target names :class:`RubinKdaMetadata` as its ``mamba_metadata_cls``. The family's KDA prefill
stores the snapshots the plan names (``kda_snapshot``); the family cache manager hands out the
snapshot points (``get_kda_midchunk_snapshots``) on the prefill's chunk grid.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata
from tensorrt_llm._utils import prefer_pinned

KDA_PREFILL_CHUNK_SIZE = 64


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


class RubinKdaMetadata(KimiK3MambaMetadata):
    """The engine's KDA metadata plus the step's mid-chunk snapshot plan (``kda_snapshot``)."""

    def __init__(self, max_batch_size: int, chunk_size: int, max_num_tokens: int) -> None:
        super().__init__(max_batch_size, chunk_size, max_num_tokens)
        self.kda_snapshot: KDASnapshotPlan | None = None

    def prepare(self, attn_metadata: AttentionMetadata) -> None:
        super().prepare(attn_metadata)
        self.kda_snapshot = None
        get_snapshots = getattr(attn_metadata.kv_cache_manager, "get_kda_midchunk_snapshots", None)
        if get_snapshots is None or attn_metadata.request_ids is None:
            return
        context_lengths = attn_metadata.seq_lens[: attn_metadata.num_contexts].tolist()
        self.kda_snapshot = build_kda_snapshot_plan(
            get_snapshots(attn_metadata.request_ids[: len(context_lengths)]),
            context_lengths,
            self._kda_schedule.device,
        )
