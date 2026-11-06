# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token embedding lookup from a replicated BF16 table in one kernel, via the trtllm
glm_fused_embedding op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: GLM-5.2's vocabulary and hidden size: the table every rank holds whole.
VOCAB = 154880
HIDDEN = 6144


class GlmFusedEmbedding(OpWrapper):
    """`out[..., :] = weight[input_ids[...], :]`, and a zero row for an id outside
    `[0, weight.shape[0])`.

    The guard is the Embedding module's for an unsharded table: the -1 that
    pads rejected draft tokens, or any other out-of-range id, reads as a zero
    row instead of out of bounds. Rows are bit copies of the table. Returns a
    new tensor of shape `input_ids.shape + (hidden,)` in the table's dtype; no
    input is modified.

    `weight` is the whole table, [vocab, hidden], the same on every rank: the
    lookup is local and needs no collective. A vocab-parallel shard is not a
    valid `weight` (see `note`).

    One kernel on the current stream: one thread per 16-byte vector, a grid of
    tokens x hidden / 1024 CTAs of 128 threads (8 x 6 at the c1 verify
    forward). Under programmatic dependent launch it waits for its producer
    before reading the ids, then releases its dependents at once and writes
    `out`, so the kernel after it must wait before reading `out`.
    """

    ARCHS = frozenset({Arch.SM_107})

    # The target looks up the verify forward's ids (one request's last accepted
    # token and its 7 drafts at c1) in the replicated table; the engine's ids
    # are int32. Prefill forwards (gen-server warm-up, PrefillTarget) look up a
    # whole chunk.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="c1 verify forward: 8 int32 ids into the replicated 154880 x 6144 table",
            spec=dict(tokens=8, ids_dtype=torch.int32, ids="random"),
        ),
        Cell(
            why="decode without speculation: 1 id, a grid of 1 x 6 CTAs",
            spec=dict(tokens=1, ids_dtype=torch.int32, ids="random"),
        ),
        Cell(
            why="rejected-draft padding: -1 and ids >= vocab read as zero rows; rows 0, vocab-1",
            spec=dict(tokens=8, ids_dtype=torch.int32, ids="edges"),
        ),
        Cell(
            why="int64 ids: the second id type the kernel is instantiated for",
            spec=dict(tokens=8, ids_dtype=torch.int64, ids="edges"),
        ),
        Cell(
            why="prefill chunk: 8192 ids, a grid of 8192 x 6 CTAs",
            spec=dict(tokens=8192, ids_dtype=torch.int32, ids="random"),
        ),
    )

    note = """
    `weight`: BF16, contiguous, 2-D, 16-byte aligned, hidden a multiple of 8;
    ids int32 or int64 on the table's device, any shape (made contiguous). The
    op raises on anything else.

    A vocab-parallel shard passed as `weight` is not detectable from the
    tensors and answers wrongly: the op indexes it with the global ids (rows
    of the wrong tokens, zeros past the shard) instead of masking to the
    shard's range and summing over ranks. The target keeps the table whole.

    Bitwise equal to the Embedding module's masked F.embedding chain for an
    unsharded table.
    """

    def __call__(self, input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.ops.trtllm.glm_fused_embedding(input_ids, weight)

    def reference(self, input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """The table rows of the in-range ids, zero rows elsewhere: a gather,
        nothing accumulated or rounded."""
        valid = (input_ids >= 0) & (input_ids < weight.shape[0])
        rows = torch.where(valid, input_ids, torch.zeros_like(input_ids)).long()
        return torch.nn.functional.embedding(rows, weight).masked_fill(~valid.unsqueeze(-1), 0)

    def is_valid(self, input_ids: torch.Tensor, weight: torch.Tensor) -> None:
        """Nothing to refuse: every input the op accepts it computes as stated.

        The op raises on every dtype, layout, alignment and device it does not
        serve. What is left is whether `weight` is the whole table (see
        `note`), which no check on the tensors can see.
        """
        return None


fused_embedding = GlmFusedEmbedding()
