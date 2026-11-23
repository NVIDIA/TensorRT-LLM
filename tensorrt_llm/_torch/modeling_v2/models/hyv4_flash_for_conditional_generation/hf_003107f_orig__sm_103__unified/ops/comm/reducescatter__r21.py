# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum across ranks and keep this rank's slice, via the trtllm reducescatter op,
at this checkpoint's trailing width and row ends.

Local variant of ``catalog/comm/reducescatter.py`` (``__r21``) owned by the
Hy4-Flash unified target: same op call, same reference and same guards, with
``CELLS`` replaced by the shapes this target calls it with. The shared entry and
its cells are unchanged; this file is what the target imports.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

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

    #: Driven on the run's sm_103 device only. The shared entry also carries
    #: SM_100 from its own certification; that arch is not measured here.
    ARCHS = frozenset({Arch.SM_103})

    # The Hy4-Flash EP combine's reduce-scatter: the 3072-wide hidden, one row
    # per rank at decode through the 5120-row prefill chunk. Every cell takes
    # the even split (`sizes=None`), the form an equalized step presents; a
    # ragged attention-DP step splits by the group's own row counts instead, and
    # that form is not a cell here.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one row per rank after the split, hidden 3072",
            spec=dict(rows_per_rank=1, trailing=(3072,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="64-row end: a decode batch's per-rank rows, hidden 3072",
            spec=dict(rows_per_rank=64, trailing=(3072,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="prefill chunk: 5120 rows per rank, hidden 3072",
            spec=dict(rows_per_rank=5120, trailing=(3072,), dtype=torch.bfloat16, sizes=None),
        ),
    )

    note = """
    Local variant (`__r21`) of `catalog/comm/reducescatter.py`, owned by the
    Hy4-Flash unified target. Same op call, reference and guards as the shared
    entry; the shared entry's 7168-wide cells stay with it.

    The one call site this target reaches is the attention-DP row exchange in
    `LatentMoe.forward` (dep4): the group's expert partials, each over the union
    of the ranks' rows, are summed here and this rank keeps its own rows. `sizes`
    is `None` when the engine has equalized the ranks' row counts (a
    graph-eligible decode step) and the group's ragged counts otherwise, which
    an eager attention-DP step hands instead.

    `__call__` dispatches to `torch.ops.trtllm.reducescatter` (the MPI-session
    symbol) or `torch.ops.trtllm.reducescatter_pg` (a boxed c10d ProcessGroup,
    the routing the product's non-MPI mode uses in
    `_torch/distributed/ops.py`). Both are compiled-in C++ kernels
    (`cpp/tensorrt_llm/thop/reducescatterOp.cpp`, `TORCH_LIBRARY_FRAGMENT(trtllm)`)
    that drive NCCL, so the path that runs is the wheel's binary, not a JIT
    compilation.

    The matrix beside the target's test drives every cell above at world size 4
    on sm_103, and gates each one bitwise against `reference` on a payload whose
    every partial sum is exact in bf16.

    Not certified: world size 8 across two nodes (the tep8 member; the matrix
    spawns one rank per visible device of one node), the ragged `sizes` form and
    dtypes other than bf16 (exercised by the matrix's own checks, not cells), a
    trailing width other than 3072, and rows_per_rank other than 1/64/5120.
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
