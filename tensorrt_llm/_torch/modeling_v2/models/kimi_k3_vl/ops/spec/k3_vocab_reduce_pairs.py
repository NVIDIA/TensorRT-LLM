# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Second half of vocab-parallel token selection: the exchanged ``(value, global index)`` pairs
of every rank reduced to one token per row (Triton)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_spec.vocab_parallel_kernels import reduce_pairs
from ._reference import argmax_pairs, interleaved_pairs


class K3VocabReducePairs(OpWrapper):
    """``out[r]`` = the global index of the pair the ``torch.argmax`` order selects among slots
    ``s < n_slots`` of row ``r`` (value at ``pairs.view(-1)[r * stride_row + s * stride_slot]``,
    index right after it). ``out`` is ``[rows]`` of any integer dtype; returns ``out``."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # The MNNVL exchange of the whole helix group (every rank's per-split pairs, interleaved);
    # int64 for the target tokens, int32 for the draft tokens of a drafter without a Markov head.
    CELLS = (
        Cell(
            why="helix16 ULL c1: 8 rows, 16 ranks x 5 splits = 80 slots",
            spec=dict(tp=16, rows=8, out_dtype="int64", ties=False),
        ),
        Cell(
            why="helix16 ULL c4: 32 rows, 80 slots, int32 draft tokens",
            spec=dict(tp=16, rows=32, out_dtype="int32", ties=False),
        ),
        Cell(
            why="helix8 c24/c42: 16 rows, 8 ranks x 10 splits = 80 slots",
            spec=dict(tp=8, rows=16, out_dtype="int64", ties=False),
        ),
        Cell(
            why="helix8 c75/c174: 64 rows (two programs), ties across ranks and NaN rows",
            spec=dict(tp=8, rows=64, out_dtype="int64", ties=True),
        ),
        Cell(
            why="helix8 c240: 120 rows (20 x 6, a partial program), 8 ranks x 3 splits",
            spec=dict(tp=8, rows=120, out_dtype="int64", ties=False),
        ),
    )

    note = """
    The pairs come from ``k3_vocab_local_select`` through the exchange: MNNVL / one rank lay
    them out interleaved per row (slot stride 2), NCCL as rank blocks (slot stride 2 * rows).
    A NaN wins; equal values go to the lower index, which is the lower rank's lower column, so
    the result equals ``torch.argmax`` over the gathered logits. 32 rows per program.
    """

    def __call__(
        self,
        pairs: torch.Tensor,
        stride_row: int,
        stride_slot: int,
        n_slots: int,
        out: torch.Tensor,
    ) -> torch.Tensor:
        return reduce_pairs(pairs, stride_row, stride_slot, n_slots, out)

    def reference(
        self,
        pairs: torch.Tensor,
        stride_row: int,
        stride_slot: int,
        n_slots: int,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """The new ``out`` (a new tensor of its shape and dtype); ``out`` is not written."""
        vals, idx = interleaved_pairs(pairs, stride_row, stride_slot, n_slots, out.shape[0])
        _, best = argmax_pairs(vals, idx)
        return best.to(out.dtype)


k3_vocab_reduce_pairs = K3VocabReducePairs()
