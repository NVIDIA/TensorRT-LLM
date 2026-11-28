# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The helix combine mask of a token-sparse generation batch in one launch (Triton).

A helix rank's partial attention output of a row must stay out of the combine when the rank owns
no KV of the row (rank-local bound 0) or when its share of the row's Top-K is empty. The helix8
target builds this mask here when the key-shard merge did not write it (the fallback Top-K, or a
step without per-token bounds).
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ....kernels.attention.rubin_dsa_kernels import helix_zero_or_empty_rows


class RubinHelixZeroOrEmptyRows(OpWrapper):
    """``(bounds == 0) | (row_lens == 0)`` as a new bool ``[R]`` tensor: ``bounds`` (each row's
    rank-local KV bound) and ``row_lens`` (each row's kept Top-K entries on this rank), contiguous
    1-D integer tensors of one length ``R``. No host read: CUDA-graph safe."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="helix8 GEN b16 k5: 96 rows, some with no local KV, some with an empty share",
            spec=dict(rows=96, zero_frac=0.2, empty_frac=0.2),
        ),
        Cell(why="one row", spec=dict(rows=1, zero_frac=0.0, empty_frac=1.0)),
        Cell(
            why="more rows than one program (1024)",
            spec=dict(rows=1500, zero_frac=0.1, empty_frac=0.1),
        ),
    )
    note = """
    The same values as ``bounds == 0`` and ``row_lens == 0`` OR-ed in torch (two compares and an
    OR, three launches), in one launch.
    """

    def __call__(self, bounds: torch.Tensor, row_lens: torch.Tensor) -> torch.Tensor:
        return helix_zero_or_empty_rows(bounds, row_lens)

    def reference(self, bounds: torch.Tensor, row_lens: torch.Tensor) -> torch.Tensor:
        return (bounds == 0) | (row_lens == 0)

    def is_valid(self, bounds: torch.Tensor, row_lens: torch.Tensor) -> None:
        """Nothing to add: the launcher raises on every layout it would read wrongly."""
        return None


rubin_helix_zero_or_empty_rows = RubinHelixZeroOrEmptyRows()
