# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather every rank's rows into one tensor, via the trtllm allgather op.

The attention-DP gather of the deepseek-r1 and gpt-oss default targets and of
the qwen4_exp unified target, whose cells (hidden 2560, worlds 4 and 2) are
merged into this entry's.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

#: The qwen4_exp unified target's MoE input under attention DP: hidden 2560,
#: bf16, every rank padded to the group's largest row count, so `sizes` is None.
_QWEN4_UNIFIED = dict(trailing=(2560,), dtype=torch.bfloat16, sizes=None)


class Allgather(OpWrapper):
    """Gather `input` from every rank in `group`, concatenated along dim 0.

    `sizes` gives each rank's `input.shape[0]` in ascending rank order, or is
    `None` when every rank holds the same number of rows. The result is a
    freshly allocated tensor on every rank; `input` is left untouched.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    # deepseek-r1 at dep4 and the gpt-oss default target under attention DP
    # are the callers, and both always pass `sizes=None`: each pads every rank
    # to a common height first, so the ragged form the op also supports is
    # not something a shipped target reaches. Row counts span what an
    # attention-DP rank feeds its expert-parallel MoE call, from one decode
    # token to a full prefill chunk.
    #
    # The qwen4_exp unified target gathers the same way, before each of its 48
    # MoE layers: at dep4 (world 4) and at disagg's dep2 generation engine
    # (world 2). Its cells carry `world`; a cell without one is driven at every
    # world size the matrix runs.
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
            why="qwen4 dep4 decode: one row per rank, hidden 2560",
            spec=dict(world=4, rows=1, **_QWEN4_UNIFIED),
        ),
        Cell(
            why="qwen4 dep4 decode at concurrency 128: 32 rows per rank",
            spec=dict(world=4, rows=32, **_QWEN4_UNIFIED),
        ),
        Cell(
            why="qwen4 dep4 prefill: a max_num_tokens 8192-row chunk per rank",
            spec=dict(world=4, rows=8192, **_QWEN4_UNIFIED),
        ),
        Cell(
            why="qwen4 dep2 (disagg generation) decode: one row per rank",
            spec=dict(world=2, rows=1, **_QWEN4_UNIFIED),
        ),
        Cell(
            why="qwen4 dep2 decode at concurrency 128: 64 rows per rank",
            spec=dict(world=2, rows=64, **_QWEN4_UNIFIED),
        ),
        Cell(
            why="qwen4 dep2 prefill: a max_num_tokens 8192-row chunk per rank",
            spec=dict(world=2, rows=8192, **_QWEN4_UNIFIED),
        ),
        # The per-rank counts between those: dep4 captures decode graphs at
        # 1, 2, 4, 8, 16, 24, 32 rows per rank, and an eager step pads every
        # rank to the group's largest count, which can be any value up to 8192.
        Cell(
            why="qwen4 dep4 decode graph at 2 rows per rank, the smallest batch past one",
            spec=dict(world=4, rows=2, **_QWEN4_UNIFIED),
        ),
        Cell(
            why="qwen4 dep4 decode graph at 24 rows per rank, a captured batch not a power of two",
            spec=dict(world=4, rows=24, **_QWEN4_UNIFIED),
        ),
        Cell(
            why="qwen4 dep4 eager at 33 rows per rank, one past the largest captured batch",
            spec=dict(world=4, rows=33, **_QWEN4_UNIFIED),
        ),
        Cell(
            why="qwen4 dep4 eager prefill padded to 1000 rows per rank, an arbitrary group max",
            spec=dict(world=4, rows=1000, **_QWEN4_UNIFIED),
        ),
        Cell(
            why="qwen4 dep4 eager prefill padded to 4097 rows per rank, one past a power of two",
            spec=dict(world=4, rows=4097, **_QWEN4_UNIFIED),
        ),
        Cell(
            why="qwen4 dep2 decode at 24 rows per rank, a captured batch not a power of two",
            spec=dict(world=2, rows=24, **_QWEN4_UNIFIED),
        ),
        Cell(
            why="qwen4 dep2 eager prefill padded to 4097 rows per rank, one past a power of two",
            spec=dict(world=2, rows=4097, **_QWEN4_UNIFIED),
        ),
    )

    note = """
    The ragged form (`sizes` as an explicit per-rank vector) and dtypes other
    than bf16 work and are exercised by the op matrix, but no shipped target
    reaches them, so they are not cells.

    CUDA graph behaviour -- capture at every engine batch size, replays
    surviving eager calls of other shapes, the captured sizes vector being held
    -- is not a cell axis either: it is about the op across capture and replay
    rather than about an input configuration, and the matrix carries it as its
    own named checks.

    Two symbols, two C++ paths. `process_group=None` runs `allgather`
    (`AllgatherOp` in thop/allgatherOp.cpp: `ncclAllGather` on the NCCL
    communicator `getComm(group)` builds over the MPI session and caches per
    group); a boxed ProcessGroup runs `allgather_pg` (`AllgatherPgOp`: the
    c10d ProcessGroup's allgather). A served engine under mpirun or
    trtllm-llmapi-launch calls the first. The matrix drives every cell through
    both, at world 2 and 4: its spawned ranks run `allgather_pg` and every
    other check; its mpirun ranks run `allgather` through the cells, the
    uniform and ragged gathers, CUDA graphs at the engine batch sizes (48
    gathers in each graph, one per MoE layer of the qwen4_exp checkpoint, which
    also covers the 29 of a 30-layer deepseek-shaped decode graph), eager calls
    between replays, the engine's attention-DP host step and stream switching.
    Both are compiled-in C++ (libth_common), no JIT.

    The MPI symbol builds its group's communicator on the group's first call
    (an MPI send/recv of the NCCL unique id, then ncclCommInitRank), so that
    call cannot be the first one inside a CUDA-graph capture; an engine's eager
    warm-up makes it. Read from `getComm`, not probed.

    Row counts. The qwen4 cells sample the per-rank counts the attention-DP
    round trip gathers. At world 4 they are 1, 2, 24 and 32, decode graph
    batches, then 33, 1000, 4097 and 8192, eager steps padded to the group's
    largest count. At world 2 they are 1, 24, 64, 4097 and 8192. Per-rank
    message sizes run from 5 KiB to 40 MiB. The matrix's CUDA-graph checks also
    capture and replay every batch the qwen4_exp deployment captures (1, 2, 4,
    8, 16, then 24 to 128 in steps of 8) and the default list at
    `max_batch_size = 256` (1 to 32, 64, 128, 256) at hidden 2560, through both
    symbols. A count between the cells is not one. The cells without `world`
    (hidden 7168, the deepseek-r1 and gpt-oss callers) run at both world sizes.

    Hard limits. The uniform form passes `input.numel()` to NCCL as a size_t
    count. The ragged form sums `sizes`, and the trailing dims' product, in an
    int accumulator (`std::accumulate` seeded with an int), so past 2^31 - 1
    gathered rows or trailing elements it overflows; no target uses that form.
    The largest cell (8192 rows x 4 ranks x 2560) is 83,886,080 output
    elements, 160 MiB; 2^31 elements at trailing 2560 is 838,861 gathered rows.

    Not certified: a split MPI session (the per-engine session of disagg under
    trtllm-llmapi-launch; the matrix's MPI world is COMM_WORLD), world 8, more
    than 8192 rows per rank. On sm_100 only the cells without `world`, at world
    4 through `allgather_pg`, were driven; the qwen4_exp cells (those with
    `world`), world 2 and the MPI symbol were driven on sm_103 only.
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
