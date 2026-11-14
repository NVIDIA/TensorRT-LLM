# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KDA step metadata of the Kimi K3 targets (``mamba_metadata_cls``).

``K3MambaMetadata`` extends the engine's ``Mamba2Metadata`` (stable int32 slots / starts and
int64 offsets) with what the KDA layers read once per step instead of once per layer:

* main's FlashInfer KDA prefill schedule (``kda_max_seqlen``, ``kda_seq_order``,
  ``kda_cu_chunks``) and the 16-byte aligned ``generation_state_indices``;
* the CuTe DSL prefill's metadata: the FLA-style ``kda_chunk_indices`` (filled on the device),
  ``kda_varlen_is_aligned`` / ``kda_single_sequence_length``, chunk alignment 64 for the warmup,
  and the int64 ``state_indices_long`` mirror (packed into the step's input upload while the
  engine packs them, ``pyexecutor.step_inputs``);
* ``kda_snapshot``, the recurrent-state snapshots the prefill takes inside this step's chunk of a
  context request (:class:`KDASnapshotPlan`), from one of two producers of the cache manager:

  - **mid-chunk snapshots**: ``kv_cache_manager.get_kda_midchunk_snapshots(request_ids)`` returns
    ``(context index, pool slot, offset)`` for each context request whose chunk holds a snapshot
    point; the prefill writes the state and convolution window after the first ``offset`` tokens
    into that slot. A target asks for them by declaring the family's
    ``ops/cache/k3_sm107_cache_manager.KimiK3Sm107MidchunkCacheManager`` (alignment
    ``KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT``, the prefill kernel's chunk) as
    ``RuntimeHooks.kv_cache_manager_cls``. Fetching the plan is the
    handshake: the manager commits a snapshot page only when the plan was fetched this step, and
    turns mid-chunk planning off when a planned snapshot is never fetched.
  - **in-forward snapshots** (``MambaHybridCacheManagerV2`` with block reuse):
    ``Mamba2Metadata.prepare`` hands its plan to :meth:`K3MambaMetadata._prepare_inforward_split`;
    the live slots keep the state at the point and staging rows the chunk-end state.

  A step takes snapshots from at most one of them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import accumulate
from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor import step_inputs
from tensorrt_llm._utils import prefer_pinned

__all__ = [
    "KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT",
    "KDA_PREFILL_CHUNK_SIZE",
    "K3MambaMetadata",
    "KDASnapshotPlan",
    "build_kda_snapshot_plan",
]


# Granularity of the mid-chunk snapshot offsets (the CuTe DSL prefill kernel's chunk).
KDA_PREFILL_CHUNK_SIZE = 64
#: The step alignment of the mid-chunk snapshots the family's cache manager plans
#: (``KimiK3Sm107MidchunkCacheManager``): offsets are multiples of the prefill chunk.
KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT = KDA_PREFILL_CHUNK_SIZE


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
    # In-forward snapshots: int64 [n] rows of the cache manager's staging tensors
    # (``inforward_snapshot_stage_views``) that receive each sequence's chunk-end state;
    # ``slots`` are then the sequences' live slots, which keep the state at the snapshot point.
    # None: the snapshots go to pool slots of their own.
    stage_rows: Optional[torch.Tensor] = None


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


class K3MambaMetadata(Mamba2Metadata):
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

        # An int64 mirror of the state indices, refreshed once per step.
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

        # Mamba2Metadata.prepare builds this step's in-forward plan (_prepare_inforward_split).
        self.kda_snapshot = None
        super().prepare(attn_metadata)

        self.generation_state_indices = None
        batch_size = attn_metadata.seq_lens.shape[0]
        if step_inputs.is_open() and self._state_indices_aliased_ptr is None:
            # ``Mamba2Metadata.prepare`` wrote the int32 indices from ``state_indices_cpu`` (it
            # did not alias a cache manager buffer): the pack uploads the same values, widened on
            # the host, at the graph replay.
            step_inputs.copy_(
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

        self._fetch_midchunk_snapshots(attn_metadata, context_lengths)

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

        # The CuTe DSL prefill's varlen chunk table (64-token chunks).
        num_chunks = sum(
            (length + KDA_PREFILL_CHUNK_SIZE - 1) // KDA_PREFILL_CHUNK_SIZE
            for length in context_lengths
        )
        if self._kda_chunk_indices.shape[0] < num_chunks:
            self._kda_chunk_indices = torch.empty((num_chunks, 2), dtype=torch.long, device="cuda")
        self.kda_chunk_indices = self._kda_chunk_indices[:num_chunks]
        _prepare_kda_chunk_indices(attn_metadata.seq_lens_cuda[:n], self.kda_chunk_indices)

    def _fetch_midchunk_snapshots(
        self, attn_metadata: AttentionMetadata, context_lengths: List[int]
    ) -> None:
        """This step's mid-chunk snapshots as the ``kda_snapshot`` plan (see the module).

        The cache manager's ``get_kda_midchunk_snapshots(request_ids)`` takes the context
        requests' ids in batch order and returns ``(context index, pool slot, offset)`` per
        snapshot; a manager without the method (or a step without request ids) plans none.
        """
        get_snapshots = getattr(attn_metadata.kv_cache_manager, "get_kda_midchunk_snapshots", None)
        if get_snapshots is None or attn_metadata.request_ids is None:
            return
        plan = build_kda_snapshot_plan(
            get_snapshots(attn_metadata.request_ids[: len(context_lengths)]),
            context_lengths,
            self._state_indices_long.device,
        )
        if plan is None:
            return
        if self.kda_snapshot is not None:
            raise RuntimeError(
                "Kimi K3 KDA: the cache manager planned both mid-chunk and in-forward snapshots "
                "for one step"
            )
        self.kda_snapshot = plan

    def _prepare_inforward_split(
        self, attn_metadata: AttentionMetadata, kv_cache_manager, plan, num_contexts, batch_size
    ) -> None:
        """This step's in-forward snapshots (``MambaHybridCacheManagerV2`` with block reuse) as a
        ``kda_snapshot`` plan with staging rows.

        A planned request's chunk runs whole: the CuTe DSL prefill stores the recurrent state at
        the snapshot point P (``num_before`` tokens into the chunk, a multiple of the KDA chunk),
        and the layer leaves the state and convolution window at P in the request's live slot
        and the chunk-end ones in its staging row, as ``finish_inforward_snapshots`` expects
        (``K3KDALinearAttention.forward_prefill``).
        """
        if batch_size != num_contexts:
            raise RuntimeError(
                "In-forward KDA snapshot planned for a batch with generation requests"
            )
        width = kv_cache_manager.conv_state_shape[-1]
        context_lengths = attn_metadata.seq_lens[:num_contexts].tolist()
        request_ids = attn_metadata.request_ids[:num_contexts]
        snapshots, stage_rows = [], []
        for seq, (rid, length) in enumerate(zip(request_ids, context_lengths)):
            entry = plan.get(rid)
            if entry is None:
                continue
            before = entry.num_before
            if (
                before + entry.num_after != length
                or before < width
                or before % KDA_PREFILL_CHUNK_SIZE
            ):
                raise RuntimeError(
                    f"In-forward KDA snapshot: request {rid} chunk of {length} tokens does "
                    f"not match the plan ({before} + {entry.num_after}, window {width})"
                )
            snapshots.append((seq, kv_cache_manager.state_index(rid), before))
            stage_rows.append(entry.row)
        if not snapshots:
            return
        device = self._state_indices_long.device
        rows = torch.tensor(stage_rows, dtype=torch.int64, pin_memory=prefer_pinned())
        self.kda_snapshot = replace(
            build_kda_snapshot_plan(snapshots, context_lengths, device),
            stage_rows=rows.to(device, non_blocking=True),
        )
