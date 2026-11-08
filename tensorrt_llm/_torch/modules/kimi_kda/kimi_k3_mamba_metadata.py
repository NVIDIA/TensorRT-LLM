# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-prepared scheduling metadata for packed FlashInfer KDA prefill."""

from itertools import accumulate
from typing import NamedTuple, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._utils import prefer_pinned


class KdaInforwardSnapshot(NamedTuple):
    """One step's in-forward snapshots of the KDA layers (``KimiK3MambaMetadata``)."""

    rows: torch.Tensor  # int32 [N]: staging row of context sequence b, -1 for none
    chunks: torch.Tensor  # int32 [N]: P lies 64 * chunks[b] tokens into sequence b's chunk
    live: torch.Tensor  # int64 [k]: live state slots of the k snapshotted sequences
    stage_rows: torch.Tensor  # int64 [k]: their staging rows
    conv_win: torch.Tensor  # int64 [k * (W - 1)]: positions of the conv window ending at P


class KimiK3MambaMetadata(Mamba2Metadata):
    """Reuse stable int32 slots/starts and int64 offsets from Mamba metadata."""

    prefill_chunk_alignment = 16
    # This step's in-forward snapshots (None: none).
    kda_inforward_snapshot: Optional[KdaInforwardSnapshot] = None

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

    def prepare(self, attn_metadata: AttentionMetadata) -> None:
        context_lengths = attn_metadata.seq_lens[: attn_metadata.num_contexts].tolist()
        assert all(length > 0 for length in context_lengths), (
            f"Expected positive context lengths, got {context_lengths}"
        )

        self.kda_inforward_snapshot = None
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

    def _prepare_inforward_split(
        self, attn_metadata: AttentionMetadata, kv_cache_manager, plan, num_contexts, batch_size
    ) -> None:
        """The KDA form of the in-forward snapshots: no split. The prefill kernel stores the
        state at P into the request's staging row in its pass over the whole chunk, and the
        layer then swaps it with the live chunk-end state (``KimiKDALinearAttention``)."""
        from ...custom_ops.cute_dsl_kimi_k3_custom_ops import KDA_PREFILL_SNAPSHOT_CHUNK

        if batch_size != num_contexts:
            raise RuntimeError(
                "In-forward KDA snapshot planned for a batch with generation requests"
            )
        width = kv_cache_manager.conv_state_shape[-1]
        rows, chunks, live, stage_rows, conv_win = [], [], [], [], []
        start = 0
        request_ids = attn_metadata.request_ids[:num_contexts]
        for rid, length in zip(request_ids, attn_metadata.seq_lens[:num_contexts].tolist()):
            entry = plan.get(rid)
            if entry is None:
                rows.append(-1)
                chunks.append(0)
            else:
                before = entry.num_before
                if (
                    before + entry.num_after != length
                    or before < width
                    or before % KDA_PREFILL_SNAPSHOT_CHUNK
                ):
                    raise RuntimeError(
                        f"In-forward KDA snapshot: request {rid} chunk of {length} tokens does "
                        f"not match the plan ({before} + {entry.num_after}, window {width})"
                    )
                rows.append(entry.row)
                chunks.append(before // KDA_PREFILL_SNAPSHOT_CHUNK)
                live.append(kv_cache_manager.state_index(rid))
                stage_rows.append(entry.row)
                conv_win.extend(range(start + before - width, start + before))
            start += length
        if not live:
            return
        # Fresh pinned sources: no later host write can race the copies.
        pinned = prefer_pinned()
        rows_chunks = torch.tensor([rows, chunks], dtype=torch.int32, pin_memory=pinned).to(
            "cuda", non_blocking=True
        )
        indices = torch.tensor(
            live + stage_rows + conv_win, dtype=torch.int64, pin_memory=pinned
        ).to("cuda", non_blocking=True)
        k = len(live)
        self.kda_inforward_snapshot = KdaInforwardSnapshot(
            rows_chunks[0], rows_chunks[1], indices[:k], indices[k : 2 * k], indices[2 * k :]
        )
