# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token embedding lookup from the whole (TP1) table in one Triton kernel, ids outside the
vocab reading as zero rows; the kernel is in ``_masked_embedding_gather``."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._masked_embedding_gather import masked_embedding_gather as _gather


class MaskedEmbeddingGather(OpWrapper):
    """`out[..., :] = weight[input_ids[...], :]`, and a +0.0 row for an id outside
    `[0, weight.shape[0])`.

    Returns a new tensor of shape `input_ids.shape + (hidden,)` in the table's dtype; rows
    are bit copies of the table. No input is modified.

    `weight` is the whole table `[vocab, hidden]` (TP1): the guard is the Embedding
    module's for an unsharded table, so any id outside the vocab (a -1 pad, an id past the
    end) is a zero row rather than an out-of-bounds read.
    """

    ARCHS = frozenset({Arch.SM_120})

    # spec: tokens; ids ("random": uniform in-vocab | "edges": -1, 0, vocab - 1, vocab,
    # vocab + 5 and the int32 extremes first, then random). The engine's ids are int32.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode 1 token (c1): a grid of 1 x 5 programs",
            spec=dict(tokens=1, ids="random"),
        ),
        Cell(why="decode 128 tokens (concurrency 128)", spec=dict(tokens=128, ids="random")),
        Cell(
            why="out-of-vocab ids read as zero rows; rows 0 and vocab - 1",
            spec=dict(tokens=16, ids="edges"),
        ),
        Cell(
            why="prefill 8192 tokens (max_num_tokens): 8192 x 5 programs",
            spec=dict(tokens=8192, ids="random"),
        ),
    )

    note = """
    Metadata consumed: the step's input ids.

    Kernel: one Triton program per (token, 1024-column block) -- 5 blocks at hidden
    5120 -- 4 warps, no PDL, JIT-compiled on first use. The id is loaded and widened
    to int64, the row offset `row * weight.stride(0)` is int64, an invalid id's row
    load is masked off and stores zeros.

    Preconditions: integer ids on the table's device; a 2-D table with unit column
    stride. Loud: CPU tensors (Triton refuses the pointer), a 1-D table (IndexError
    before launch). `is_valid` covers what the kernel takes silently: float ids
    (truncated to an index), a strided 1-D id view (flattened as a view and read as
    if dense), a table with a column stride.

    Numerics: a gather; bitwise the Embedding module's masked `F.embedding` chain for
    an unsharded table (`pre_comm_embedding_ops` at TP1).

    Certified: the checkpoint's 248320 x 5120 bf16 table, int32 ids at 1, 16, 128
    and 8192 tokens. Not certified: int64 ids, a vocab-parallel shard (the op masks
    to the whole table's range; a shard would need `[start, end)` and a sum).
    """

    def __call__(self, input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return _gather(input_ids, weight, 0, weight.shape[0])

    def reference(self, input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """The table rows of the in-range ids, zero rows elsewhere: a gather, nothing
        accumulated or rounded."""
        valid = (input_ids >= 0) & (input_ids < weight.shape[0])
        rows = torch.where(valid, input_ids, torch.zeros_like(input_ids)).long()
        return torch.nn.functional.embedding(rows, weight).masked_fill(~valid.unsqueeze(-1), 0)

    def is_valid(self, input_ids: torch.Tensor, weight: torch.Tensor) -> None:
        assert not input_ids.dtype.is_floating_point, (
            f"input_ids must be an integer tensor; got {input_ids.dtype}, which the kernel "
            "truncates to an index"
        )
        assert input_ids.is_contiguous(), (
            "input_ids must be contiguous; a strided 1-D view is flattened as a view and "
            "read as if it were dense"
        )
        assert weight.stride(-1) == 1, (
            f"the table's hidden dim must be contiguous; got stride {weight.stride(-1)}, "
            "which the kernel reads as 1"
        )


masked_embedding_gather = MaskedEmbeddingGather()
