# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum across ranks and keep this rank's slice, via the trtllm reducescatter op.

The attention-DP reduce-scatter of the deepseek-r1 and gpt-oss default targets
and of the qwen4 unified target, whose cells (hidden 2560, worlds 4 and 2) are
merged into this entry's.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

#: The dtypes whose NCCL mapping performs the sum the caller means by it.
#: `getDtypeMap` in `cpp/tensorrt_llm/common/opUtils.cpp` is the whole table:
#: these entries land on a native NCCL float or int type, so `ncclSum` is a
#: float or integer sum of the values. Every other entry, and every dtype the
#: table omits, is refused below -- see `is_valid` for the two distinct ways
#: they go wrong.
_SUMMABLE_DTYPES = frozenset(
    {
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
        torch.int8,
        torch.uint8,
    }
)


class Reducescatter(OpWrapper):
    """Sum `input` across every rank in `group`, then keep this rank's slice.

    Every rank passes a tensor of the same full shape; the elementwise sum is
    split along dim 0 in ascending rank order and rank `i` of the group gets
    slice `i`. `sizes` gives each rank's slice height in that order, or is
    `None` when the split is even. The result is a freshly allocated tensor on
    every rank; `input` is left untouched.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    # deepseek-r1 at dep4 and the gpt-oss default target under attention DP
    # are the callers, and both always pass `sizes=None`: each pads to a
    # common height first. The row counts are the full heights an
    # attention-DP step reduces, which is the rank count times a slice.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one row per rank after the split, hidden 7168",
            spec=dict(rows_per_rank=1, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="a small padded batch",
            spec=dict(rows_per_rank=32, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="prefill chunk height",
            spec=dict(rows_per_rank=2048, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        # Qwen3.8-Flash-Next's MoE block under attention DP: dep4, and dep2 for
        # the disaggregated engines. Each rank's rows are padded to the group
        # maximum, gathered, run through this rank's 128- (dep4) or 256-expert
        # (dep2) window and reduce-scattered back at hidden 2560. Per-rank rows
        # span the decode capture share (dep4 [1..32], dep2 [1..64]) and one
        # max_num_tokens prefill chunk. `round_trip` pads to
        # max(all_rank_num_tokens): a decode graph to its captured batch (dep4
        # captures 1, 2, 4, 8, 16, 24, 32), an eager step to any count up to
        # 8192, so cells also sit between the capture batches and off every
        # power of two.
        Cell(
            why="qwen4 dep4 decode: one padded row per rank, hidden 2560",
            spec=dict(world=4, rows_per_rank=1, trailing=(2560,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="qwen4 dep4 decode at the largest per-rank capture share, 128 / 4 = 32 rows",
            spec=dict(
                world=4, rows_per_rank=32, trailing=(2560,), dtype=torch.bfloat16, sizes=None
            ),
        ),
        Cell(
            why="qwen4 dep4 prefill: one 8192-row max_num_tokens chunk per rank, 32768 gathered",
            spec=dict(
                world=4, rows_per_rank=8192, trailing=(2560,), dtype=torch.bfloat16, sizes=None
            ),
        ),
        Cell(
            why="qwen4 dep2 (disagg) decode: one padded row per rank, a two-term sum",
            spec=dict(world=2, rows_per_rank=1, trailing=(2560,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="qwen4 dep2 (disagg) decode at the largest per-rank capture share, 64 rows",
            spec=dict(
                world=2, rows_per_rank=64, trailing=(2560,), dtype=torch.bfloat16, sizes=None
            ),
        ),
        Cell(
            why="qwen4 dep2 (disagg) prefill: one 8192-row max_num_tokens chunk per rank",
            spec=dict(
                world=2, rows_per_rank=8192, trailing=(2560,), dtype=torch.bfloat16, sizes=None
            ),
        ),
        Cell(
            why="qwen4 dep4 decode graph at its second capture batch, 2 rows per rank",
            spec=dict(world=4, rows_per_rank=2, trailing=(2560,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="qwen4 dep4 decode graph at 24 rows per rank, a capture batch off the powers of two",
            spec=dict(
                world=4, rows_per_rank=24, trailing=(2560,), dtype=torch.bfloat16, sizes=None
            ),
        ),
        Cell(
            why="qwen4 dep4 eager step padded to 33 rows per rank, one past the largest capture",
            spec=dict(
                world=4, rows_per_rank=33, trailing=(2560,), dtype=torch.bfloat16, sizes=None
            ),
        ),
        Cell(
            why="qwen4 dep4 eager prefill padded to 1000 rows per rank, between 32 and 8192",
            spec=dict(
                world=4, rows_per_rank=1000, trailing=(2560,), dtype=torch.bfloat16, sizes=None
            ),
        ),
        Cell(
            why="qwen4 dep4 eager prefill at 4097 rows per rank, odd and one past 2^12",
            spec=dict(
                world=4, rows_per_rank=4097, trailing=(2560,), dtype=torch.bfloat16, sizes=None
            ),
        ),
        Cell(
            why="qwen4 dep2 (disagg) decode graph at 24 rows per rank",
            spec=dict(
                world=2, rows_per_rank=24, trailing=(2560,), dtype=torch.bfloat16, sizes=None
            ),
        ),
        Cell(
            why="qwen4 dep2 (disagg) eager prefill at 4097 rows per rank, odd and one past 2^12",
            spec=dict(
                world=2, rows_per_rank=4097, trailing=(2560,), dtype=torch.bfloat16, sizes=None
            ),
        ),
    )

    note = """
    The ragged form and dtypes other than bf16 work and are exercised by the op
    matrix, but no shipped target reaches them, so they are not cells. CUDA
    graph behaviour is carried by the matrix's own named checks rather than as
    a cell axis.

    `world` is the group size a cell is driven at. A cell without one (the
    deepseek-r1 / gpt-oss cells, hidden 7168) is driven at world 4 only, the
    dep4 layout it was certified at; the matrix runs at worlds 2 and 4. The
    qwen4 cells (hidden 2560, worlds 4 and 2) were driven on sm_103 only, one
    GB300 tray, and gate bitwise against `reference` on payloads whose every
    partial sum is exact. On sm_100 only the cells without `world` and world 4
    were driven; the qwen4 cells and world 2 are not certified there.

    The op is compiled C++ (`cpp/tensorrt_llm/thop/reducescatterOp.cpp`), with
    no JIT path. Its two symbols are separate classes. The matrix drives
    `reducescatter_pg` (`ReducescatterPgOp`, c10d `ProcessGroup::reduce_scatter`)
    because it starts its ranks without MPI. A served engine under `mpirun` or
    `trtllm-llmapi-launch` passes `process_group=None` and runs `reducescatter`
    (`ReducescatterOp`: `ncclReduceScatter` on the MPI-session communicator, one
    `ncclReduce` per root for a ragged split), which the matrix does not drive.
    That communicator is `getComm`'s (`cpp/tensorrt_llm/common/opUtils.cpp`),
    built with `NCCL_RUNTIME_CONNECT=0`, `NCCL_GRAPH_REGISTER=0`,
    `graphUsageMode=1` and NVLS off where the fabric cannot bind it, not with
    c10d's defaults, so NCCL may pick another algorithm there.

    Numerics, measured through `_pg` on GB300: the sum is in the input dtype,
    deterministic, and a sequential ring-order chain starting at the
    destination's successor. At world 2 that chain is one bf16 addition, so the
    result is the correctly rounded sum, bitwise. At world 4 about a third of
    the elements differ from the correctly rounded sum, all within the
    sequential-summation bound.
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
            return torch.ops.trtllm.reducescatter(input, sizes, group)
        return torch.ops.trtllm.reducescatter_pg(input, sizes, group, process_group)

    def reference(
        self,
        contributions_to_this_slice: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """The sum of what every rank sent to this rank's slice.

        The documented exception to mirroring `__call__`, for the same reason as
        allgather: a collective's output is not a function of what the calling
        rank holds. It takes the one block each rank contributed to *this*
        position, in ascending rank order -- not their whole inputs, because a
        rank can regenerate exactly those WORLD blocks from the shared seed
        convention and never has to materialize anybody else's full tensor.

        Accumulated in fp32 (int64 for integer dtypes) and rounded once. The op
        sums in the input dtype in whatever order NCCL picks; on a cell's payload
        that costs nothing, because every partial sum there is exact in the input
        dtype either way -- which is what lets the matrix gate it bitwise.
        """
        first = contributions_to_this_slice[0]
        acc_dtype = torch.float32 if first.dtype.is_floating_point else torch.int64
        acc = torch.zeros(first.shape, dtype=acc_dtype, device=first.device)
        for block in contributions_to_this_slice:
            acc += block.to(acc_dtype)
        return acc.to(first.dtype)

    def is_valid(
        self,
        input: torch.Tensor,
        sizes: Optional[List[int]],
        group: List[int],
        process_group: Optional[object] = None,
    ) -> None:
        # Each guard is for a violation measured to end in silent corruption, a
        # wedge, or a process kill rather than an error.
        assert input.dim() >= 1, (
            "input must have at least one dimension; a 0-d tensor segfaults inside "
            "ReducescatterOp::run_list"
        )
        assert input.is_contiguous(), (
            "input must be contiguous; the op reads it as packed memory and a "
            "strided view is silently reduced over the wrong elements"
        )
        # An allowlist rather than a list of known-bad dtypes, because the two
        # ways a dtype goes wrong here are both silent and the second is not
        # confined to this call:
        #
        #   mapped to the wrong arithmetic -- float8_e4m3fn and bool both land
        #   on ncclInt8, so ncclSum adds their raw bytes. fp8 comes back as
        #   nonsense and bool behaves like OR, neither with an error.
        #
        #   not in the map at all -- float64, quint4x2 and the rest reach
        #   `TorchUtils::dataType()`, which throws. That call is *inside* the
        #   group: reducescatterOp.cpp opens with ncclGroupStart() on line 21,
        #   converts on line 25, and the matching ncclGroupEnd() on line 57 is
        #   never reached. The communicator is left mid-group, and the
        #   collectives after this one return invalid data. A guard here turns
        #   that into a Python error before the group opens.
        assert input.dtype in _SUMMABLE_DTYPES, (
            f"{input.dtype} is not summable by this op; it is accepted and then "
            f"either summed as raw bytes or thrown on inside an open NCCL group, "
            f"which corrupts later collectives. Reduce in one of "
            f"{sorted(str(d) for d in _SUMMABLE_DTYPES)} and cast afterwards"
        )
        if sizes is None:
            assert input.shape[0] % len(group) == 0, (
                f"input.shape[0]={input.shape[0]} must be divisible by "
                f"len(group)={len(group)} when sizes is None; the op takes "
                "shape[0] // len(group) rows per rank and silently drops the remainder"
            )
        else:
            assert len(sizes) == len(group), (
                f"len(sizes)={len(sizes)} must equal len(group)={len(group)}; a short "
                "list raises on the highest rank and wedges the others, a long one "
                "reduces more rows than the input holds and returns garbage"
            )
            assert sum(sizes) == input.shape[0], (
                f"sum(sizes)={sum(sizes)} must equal input.shape[0]={input.shape[0]}; "
                "the op sizes the reduction from `sizes` alone, so a short split "
                "silently drops the trailing rows and a long one reads past the input"
            )


reducescatter = Reducescatter()
