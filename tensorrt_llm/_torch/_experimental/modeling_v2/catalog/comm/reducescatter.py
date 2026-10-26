# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum across ranks and keep this rank's slice, via the trtllm reducescatter op."""

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

    # The first three cells: deepseek-r1 at dep4 and the gpt-oss default target
    # under attention DP, which both pad every rank to a common height and pass
    # `sizes=None`. The row counts are the full heights an attention-DP step
    # reduces, which is the rank count times a slice. They carry no `world` key
    # and run at world 4.
    #
    # After them, Step-5-Preview's attention-DP reduce-scatter of the routed
    # expert output (hidden 4096) back to each rank's rows, at dep4 and at the
    # disaggregated dep2 world, in bf16 and fp32. `rows_per_rank` is the slice
    # each rank keeps under `sizes=None`; a ragged cell passes `sizes`, one row
    # count per rank, the unpadded form where each rank keeps only its own
    # requests' rows. Per rank a call sees one decode token, the 128-request
    # decode batch (`max_batch_size`) or one 8192-token prefill chunk
    # (`max_num_tokens`, the envelope's largest row count).
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
        Cell(
            why="step5 dep4 decode: one row per rank, bf16",
            spec=dict(world=4, rows_per_rank=1, trailing=(4096,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="step5 dep4 padded decode batch, 128 rows per rank, bf16",
            spec=dict(
                world=4, rows_per_rank=128, trailing=(4096,), dtype=torch.bfloat16, sizes=None
            ),
        ),
        Cell(
            why="step5 dep4 prefill chunk, 8192 rows per rank, bf16",
            spec=dict(
                world=4, rows_per_rank=8192, trailing=(4096,), dtype=torch.bfloat16, sizes=None
            ),
        ),
        Cell(
            why="step5 dep4 ragged decode [128, 1, 37, 64], fp32",
            spec=dict(
                world=4,
                rows_per_rank=None,
                trailing=(4096,),
                dtype=torch.float32,
                sizes=[128, 1, 37, 64],
            ),
        ),
        Cell(
            why="step5 dep4 prefill chunk 8192 per rank, fp32 (envelope max)",
            spec=dict(
                world=4, rows_per_rank=8192, trailing=(4096,), dtype=torch.float32, sizes=None
            ),
        ),
        Cell(
            why="step5 dep2 padded decode batch, 128 rows per rank, fp32",
            spec=dict(
                world=2, rows_per_rank=128, trailing=(4096,), dtype=torch.float32, sizes=None
            ),
        ),
        Cell(
            why="step5 dep2 ragged prefill against decode [8192, 17], bf16",
            spec=dict(
                world=2,
                rows_per_rank=None,
                trailing=(4096,),
                dtype=torch.bfloat16,
                sizes=[8192, 17],
            ),
        ),
    )

    note = """
    Every cell is driven through both symbols `__call__` routes to: the MPI-session
    `reducescatter` (`process_group=None`, ranks started by mpirun, the symbol a
    serving world launched under MPI calls) and `reducescatter_pg` (a boxed c10d
    group, MPI disabled), each at the cell's world size. A cell without `world`
    is a world-4 cell. deepseek-r1 and gpt-oss always pass `sizes=None`; the
    ragged form is a cell only at Step-5's unpadded attention-DP step. fp16 and
    the integer dtypes work and are exercised by the op matrix at hidden 2560,
    but no target's call site reaches them. CUDA graph behaviour is carried by
    the matrix's own named checks rather than as a cell axis.

    Numerics, on standard-normal payloads at hidden 4096 and 1, 128 and 8192
    rows per rank, bf16 and fp32, both symbols, worlds 2 and 4: the result is
    deterministic and bitwise equal to a sequential sum in the input dtype, in
    ring order starting at the destination's successor. At world 2 that is one
    rounded add, the correctly rounded sum. At world 4 it is not (the matrix
    asserts more than 5% of bf16 elements differ from an fp32 sum rounded once,
    at hidden 2560). The error stays inside
    `(world-1) * u * sum|x|` (u = 2^-8 bf16, 2^-24 fp32); the worst measured
    ratio is 0.9419 for bf16 and 0.9139 for fp32 at world 4.

    The MPI symbol builds its NCCL communicator for a rank set on that set's
    first call, so that first call must be eager (a serving engine's warm-up
    forward is that call); the `_pg` symbol's communicator exists from
    `init_process_group` and its first call captures directly. Under CUDA graphs
    a captured call holds the `sizes` it was captured with, so a captured step
    passes padded (uniform) rows. Step-5's decode graphs are driven as captured:
    19 batch sizes [1, 2, 4, 8, 16, ..., 128] with 88 reduce-scatters each
    (one per MoE layer) in one pool, replayed shuffled and in bursts.

    ARCHS: the three deepseek/gpt-oss cells were driven on sm_100 and sm_103.
    The Step-5 cells (`step5 ...`, worlds 2 and 4, both symbols) were driven on
    sm_103 only and are not certified on sm_100 until they run there.

    Not certified here: rows per rank above 128 under capture at hidden 4096
    (the dep4 tune list reaches 256), and fp16 or integer dtypes at hidden 4096.

    Hard limits: the op's element count crosses 2^31 at 524288 total rows of
    hidden 4096 (2^31 / 4096), 16x the envelope's 32768 (4 x 8192). Not probed.
    """

    def __call__(
        self,
        input: torch.Tensor,
        sizes: Optional[List[int]],
        group: List[int],
        process_group: Optional[object] = None,
    ) -> torch.Tensor:
        """Reduce-scatter through the symbol `process_group` selects.

        `process_group` selects the communicator flavor, mirroring the product
        router in `_torch/distributed/ops.py`: None runs the MPI-session symbol
        -- the one a dep4 serving world calls -- and a boxed c10d ProcessGroup
        runs the `_pg` symbol the non-MPI backends use.
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
