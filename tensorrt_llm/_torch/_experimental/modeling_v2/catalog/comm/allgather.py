# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather every rank's rows into one tensor, via the trtllm allgather op."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

_STEP5 = dict(trailing=(4096,), dtype=torch.bfloat16)


class Allgather(OpWrapper):
    """Gather `input` from every rank in `group`, concatenated along dim 0.

    `sizes` gives each rank's `input.shape[0]` in ascending rank order, or is
    `None` when every rank holds the same number of rows. The result is a
    freshly allocated tensor on every rank; `input` is left untouched.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    # The first three cells: deepseek-r1 at dep4 and the gpt-oss default target
    # under attention DP, which both pad every rank to a common height and pass
    # `sizes=None`. Row counts span what an attention-DP rank feeds its
    # expert-parallel MoE call, from one decode token to a full prefill chunk.
    # They carry no `world` key and run at world 4.
    #
    # After them, Step-5-Preview's attention-DP gather of the MoE input rows
    # (hidden 4096, bf16) at dep4 and at the disaggregated dep2 world. Per rank
    # a call sees one decode token, the 128-request decode batch
    # (`max_batch_size`), or one 8192-token prefill chunk (`max_num_tokens`, the
    # envelope's largest row count); the ragged cells are the unpadded form,
    # where each rank holds only its own requests' rows.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one row per rank, hidden 7168",
            spec=dict(rows=1, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="a small padded batch, the common case between decode and prefill",
            spec=dict(rows=32, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="prefill chunk height",
            spec=dict(rows=2048, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="step5 dep4 decode: one row per rank, hidden 4096",
            spec=dict(world=4, rows=1, sizes=None, **_STEP5),
        ),
        Cell(
            why="step5 dep4 padded decode batch, 128 rows per rank",
            spec=dict(world=4, rows=128, sizes=None, **_STEP5),
        ),
        Cell(
            why="step5 dep4 prefill chunk, 8192 rows per rank (envelope max)",
            spec=dict(world=4, rows=8192, sizes=None, **_STEP5),
        ),
        Cell(
            why="step5 dep4 ragged decode rows [128, 1, 37, 64]",
            spec=dict(world=4, rows=None, sizes=[128, 1, 37, 64], **_STEP5),
        ),
        Cell(
            why="step5 dep4 ragged at the envelope max [8192, 1, 8192, 512]",
            spec=dict(world=4, rows=None, sizes=[8192, 1, 8192, 512], **_STEP5),
        ),
        Cell(
            why="step5 dep2 padded decode batch, 128 rows per rank",
            spec=dict(world=2, rows=128, sizes=None, **_STEP5),
        ),
        Cell(
            why="step5 dep2 prefill chunk, 8192 rows per rank (envelope max)",
            spec=dict(world=2, rows=8192, sizes=None, **_STEP5),
        ),
        Cell(
            why="step5 dep2 ragged prefill against decode [8192, 17]",
            spec=dict(world=2, rows=None, sizes=[8192, 17], **_STEP5),
        ),
    )

    note = """
    Both symbols `__call__` routes to carry every cell, at every cell's world
    size: `torch.ops.trtllm.allgather` (process_group=None) with the ranks
    started by mpirun, so its communicator comes from the MPI session as in a
    served world, and `torch.ops.trtllm.allgather_pg` with a boxed c10d group.
    Every cell is a bitwise gate (a gather computes nothing), driven twice in
    a row in one process. Beyond the cells the matrix covers other dtypes,
    zero-row ranks, trailing dims, rank subsets and the guards. deepseek-r1 and
    gpt-oss always pass `sizes=None`; the ragged form is a cell only at
    Step-5's unpadded attention-DP gather.

    Inside the op, a uniform gather (sizes None, or a sizes vector whose
    entries are all equal) is one ncclAllGather on the MPI symbol; a ragged one
    is one ncclBroadcast per root inside one NCCL group. The `_pg` symbol
    runs c10d's allgather, into per-rank output slices when ragged.

    Caller preconditions: every rank passes the same `sizes` and `group`, and
    rank r's `input` holds exactly `sizes[r]` rows. A rank holding a different
    row count than its `sizes` entry is not probed here (on the MPI symbol the
    root broadcasts `sizes[r]` rows from its buffer whatever its length).
    On the MPI symbol the communicator for a rank set is built on that set's
    first call and cached for the process, so the first call is eager (an
    engine's warm-up forward); the matrix captures only after it.

    CUDA graph behaviour -- capture at every engine batch size, replays
    surviving eager calls of other shapes, the captured sizes vector being held
    -- is not a cell axis: it is about the op across capture and replay
    rather than about an input configuration, and the matrix carries it as its
    own named checks under both symbols, including Step-5's decode graphs (its
    19 capture batch sizes, one gather per MoE layer, 88 per graph, hidden
    4096). A captured graph replays the `sizes` split it was captured with, so
    a graph-captured attention-DP step pads every rank to one height.

    Hard limits (computed, not probed): both symbols sum `sizes` in a 32-bit
    int, so a total above 2^31 - 1 rows wraps; the output reaches 2^31
    elements at 524288 total rows of width 4096. The Step-5 envelope's largest
    gather is 4 x 8192 = 32768 rows (2^27 elements, 256 MiB in bf16).

    ARCHS: the three deepseek/gpt-oss cells were driven on sm_100 and sm_103.
    The Step-5 cells (`step5 ...`, worlds 2 and 4, both symbols) were driven on
    sm_103 only and are not certified on sm_100 until they run there.
    """

    def __call__(
        self,
        input: torch.Tensor,
        sizes: Optional[List[int]],
        group: List[int],
        process_group: Optional[object] = None,
    ) -> torch.Tensor:
        """Run the gather on the symbol `process_group` selects.

        This mirrors the product router in `_torch/distributed/ops.py`: None
        runs the MPI-session symbol -- the one a dep4 serving world calls -- and
        a boxed c10d ProcessGroup runs the `_pg` symbol the non-MPI backends use.
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
