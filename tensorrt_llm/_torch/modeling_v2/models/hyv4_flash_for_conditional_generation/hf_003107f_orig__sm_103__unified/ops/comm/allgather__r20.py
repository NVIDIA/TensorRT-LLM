# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather every rank's rows into one tensor, via the trtllm allgather op, at
this checkpoint's width and row counts.

Local variant of ``catalog/comm/allgather.py`` (``__r20``) owned by the
Hy4-Flash unified target: same op call, same reference and same guards, with
``CELLS`` replaced by the shapes this target calls it with -- trailing 3072 and
the 1-row, 64-row and 5120-row ends. The shared entry and its cells are
unchanged; this file is what the target imports.

The op is a compiled-in C++ kernel (``trtllm::allgather`` and
``trtllm::allgather_pg``, registered in
``cpp/tensorrt_llm/thop/allgatherOp.cpp``), not a JIT kernel: there is no
architecture-specific compilation to record, and the op runs whichever wheel
the interpreter loaded.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class Allgather(OpWrapper):
    """Gather `input` from every rank in `group`, concatenated along dim 0.

    `sizes` gives each rank's `input.shape[0]` in ascending rank order, or is
    `None` when every rank holds the same number of rows. The result is a
    freshly allocated tensor on every rank; `input` is left untouched.
    """

    #: Driven on the run's sm_103 device only. The shared entry also carries
    #: SM_100 from its own certification; that arch is not measured here and
    #: this target's routing admits sm_103 alone.
    ARCHS = frozenset({Arch.SM_103})

    # The Hy4-Flash surface: trailing 3072 (the text hidden size), at the row
    # counts the attention-DP row exchange feeds the gather -- one decode row, a
    # 64-row batch, and a full 5120-token prefill chunk -- with the even split
    # the engine's equalizing steps present.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one row per rank, trailing 3072",
            spec=dict(rows=1, trailing=(3072,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="64-row end: a padded batch, trailing 3072",
            spec=dict(rows=64, trailing=(3072,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="prefill chunk: 5120 rows, trailing 3072",
            spec=dict(rows=5120, trailing=(3072,), dtype=torch.bfloat16, sizes=None),
        ),
    )

    note = """
    The one call site this target reaches is the attention-DP row exchange in
    `LatentMoe.forward` (dep4): each rank's rows are gathered in rank order, the
    rank's expert window runs over the union, and the partial sums come back to
    the rows' own ranks by reduce-scatter. The gathered rows are bf16 of
    trailing 3072 -- the cells' width -- at row counts from one decode row to a
    full prefill chunk.

    `sizes` is `None` when the engine has equalized the ranks' row counts, which
    it does for a graph-eligible decode step; an eager attention-DP step hands
    the group's own ragged counts instead, so the ragged form is reached in
    production too. It is not a cell: the cells hold the even split, and the
    ragged split is carried by this target's op matrix (`check_ragged_gather`,
    the ragged dtype checks, the eager replays) rather than by a cell. Dtypes
    other than bf16 likewise.

    CUDA graph behaviour -- capture at every engine batch size, replays
    surviving eager calls of other shapes, the captured sizes vector being held
    -- is not a cell axis either: it is about the op across capture and replay
    rather than about an input configuration, and the matrix carries it as its
    own named checks.
    """

    def __call__(
        self,
        input: torch.Tensor,
        sizes: Optional[List[int]],
        group: List[int],
        process_group: Optional[object] = None,
    ) -> torch.Tensor:
        """`process_group` selects the communicator flavor, mirroring the
        product router in `_torch/distributed/ops.py`: None runs the
        MPI-session symbol -- the one a dep4 serving world calls -- and a boxed
        c10d ProcessGroup runs the `_pg` symbol the non-MPI backends use.
        """
        if process_group is None:
            return torch.ops.trtllm.allgather(input, sizes, group)
        return torch.ops.trtllm.allgather_pg(input, sizes, group, process_group)

    def reference(
        self,
        per_rank_inputs: Sequence[torch.Tensor],
        sizes: Optional[List[int]] = None,
        group: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """The concatenation the collective is supposed to make.

        The documented exception to mirroring `__call__`: a collective's output
        is not a function of what the calling rank holds, so a reference given
        only `input` could not state anything at all. It takes the group's
        inputs in ascending rank order instead. The matrix builds that list on
        one rank from the seed convention every rank shares, which is what keeps
        the reference arithmetic rather than a second collective.
        """
        return torch.cat(list(per_rank_inputs), dim=0)

    def is_valid(
        self,
        input: torch.Tensor,
        sizes: Optional[List[int]],
        group: List[int],
        process_group: Optional[object] = None,
    ) -> None:
        # Each guard is for a violation measured to end in silent corruption or
        # a process kill rather than an error.
        assert input.dim() >= 1, (
            "input must have at least one dimension; a 0-d tensor segfaults inside "
            "AllgatherOp::run_list"
        )
        assert input.is_contiguous(), (
            "input must be contiguous; the op reads it as packed memory and a "
            "strided view is silently gathered from the wrong elements"
        )
        assert sizes is None or len(sizes) == len(group), (
            f"len(sizes)={len(sizes) if sizes is not None else None} must equal "
            f"len(group)={len(group)}; the op sizes its output from `sizes` alone, "
            "so a short list silently drops the trailing ranks and a long one "
            "appends uninitialized rows"
        )


allgather = Allgather()
