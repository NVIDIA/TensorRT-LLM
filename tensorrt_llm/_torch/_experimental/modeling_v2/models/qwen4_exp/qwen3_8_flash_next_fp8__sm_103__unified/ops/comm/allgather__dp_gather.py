# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather every rank's rows into one tensor, via the trtllm allgather op.

This target's copy of the catalog entry `catalog/comm/allgather`, certified at the
attention-DP gather of its MoE round trip (`_expert_sum_op`'s `round_trip`, from
`_moe_block` and `_mtp_moe_block`)."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The checkpoint's hidden size: the width of every row the round trip gathers.
HIDDEN = 2560


def _dp_gather(world: int, rows: int) -> dict:
    """One rank's uniform gather in the round trip: `rows` bf16 rows of HIDDEN over a
    `world`-rank attention-DP group, every rank padded to the group's largest count."""
    return dict(world=world, rows=rows, trailing=(HIDDEN,), dtype=torch.bfloat16, sizes=None)


class Allgather(OpWrapper):
    """Gather `input` from every rank in `group`, concatenated along dim 0.

    `sizes` gives each rank's `input.shape[0]` in ascending rank order, or is
    `None` when every rank holds the same number of rows. The result is a
    freshly allocated tensor on every rank; `input` is left untouched.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The catalog entry's own cells (deepseek-r1 and gpt-oss callers, hidden 7168) are kept
    # as they stand; they carry no `world` and run at every world size the test drives.
    # Then this target's: under attention DP (dep2, dep4) each rank pads its rows to the
    # group's largest count (`_dp_pad_rows`) and gathers them with `sizes=None` before its
    # expert window. Rows per rank run from one decode token to a max_num_tokens chunk
    # (8192, the envelope's largest row), with 128 (the largest captured batch) and 512 (a
    # dep4 MTP3 verify: 128 requests x 4 rows) between.
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
        Cell(why="dep2 decode: one row per rank, hidden 2560", spec=_dp_gather(2, 1)),
        Cell(why="dep2 at the largest captured batch, 128 rows per rank", spec=_dp_gather(2, 128)),
        Cell(
            why="dep2 prefill: a max_num_tokens chunk, 8192 rows per rank",
            spec=_dp_gather(2, 8192),
        ),
        Cell(why="dep4 decode: one row per rank, hidden 2560", spec=_dp_gather(4, 1)),
        Cell(why="dep4 at the largest captured batch, 128 rows per rank", spec=_dp_gather(4, 128)),
        Cell(why="dep4 MTP3 verify of 128 requests: 512 rows per rank", spec=_dp_gather(4, 512)),
        Cell(
            why="dep4 prefill: a max_num_tokens chunk, 8192 rows per rank",
            spec=_dp_gather(4, 8192),
        ),
    )

    note = """
    Source: compiled-in C++, `torch.ops.trtllm.allgather` and `torch.ops.trtllm.allgather_pg`
    (`cpp/tensorrt_llm/thop/allgatherOp.cpp`: `ncclAllGather` for the uniform form, one
    `ncclBroadcast` per root for the ragged one). Nothing is JIT-compiled.

    Two symbols, two communicators. `process_group=None` runs the MPI-session symbol, whose
    NCCL communicator `getComm` (`cpp/tensorrt_llm/common/opUtils.cpp`) builds on a group's
    first call and caches per group; a boxed c10d ProcessGroup runs the `_pg` symbol on that
    group's communicator. This target's round trip passes None under MPI (`mpirun` and
    `trtllm-llmapi-launch`, how this run serves) and the TP group's boxed process group when
    `mpi_disabled()`. The test drives every cell, the CUDA-graph checks and the guards through
    both, each at world size 2 (dep2) and 4 (dep4), on one node. Measured on GB300: every
    gather bitwise equal to `reference` on both transports at both sizes (21 checks over
    torch.distributed, 19 over the MPI session), the world-2 job in about 35 s and the
    world-4 job in about 43 s.

    Precondition on the MPI-session symbol: a group's first call builds its communicator
    (an MPI exchange of the NCCL unique id) and cannot be captured into a CUDA graph. Make
    it eagerly first, as the engine's warm-up forward does before graph capture. The `_pg`
    symbol's communicator exists from `init_process_group`, and its cold first call is
    captured in the test.

    The ragged form (`sizes` as an explicit per-rank vector) and dtypes other
    than bf16 work and are exercised by the op matrix, but no shipped target
    reaches them, so they are not cells. This target pads to the group's largest
    count and always passes `sizes=None`.

    CUDA graph behaviour is not a cell axis: it is about the op across capture and replay
    rather than about an input configuration, and the matrix carries it as its own named
    checks. It certifies this target's captured shapes. Every per-rank row count a decode
    graph holds is captured, the engine's batch sizes (1, 2, 4, 8, 16, 24, ..., 128) and
    their x4 (an MTP3 verify's rows), with all 31 graphs alive in one pool and replayed
    interleaved, and 48 gathers per graph, one per trunk MoE layer.

    Not certified: world sizes other than 2 and 4; a group spanning nodes; a rank subset
    on the MPI-session symbol (the round trip's group is always the whole attention-DP
    group); hidden widths other than 2560 and 7168; `sizes` vectors under capture (a
    graph replays the split it captured, so a captured step must be uniform). The
    catalog entry's own three cells are driven here on sm_103 only.
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
