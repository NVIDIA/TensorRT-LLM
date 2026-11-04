# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum across ranks and keep this rank's slice, via the trtllm reducescatter op.

This target's own copy of `catalog/comm/reducescatter`, certified at the call it makes:
the scatter half of the attention-DP MoE round trip (`_expert_sum_op.round_trip`), which
hands each dep2 / dep4 rank the sum of every rank's expert-window partial over its rows."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

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

#: Qwen3.8-Flash-Next's hidden size: the width of every window partial the round trip sums.
_HIDDEN = 2560


def _dp_scatter(world: int, rows_per_rank: int) -> dict:
    """One uniform reduce-scatter of `world * rows_per_rank` bf16 rows of the hidden size,
    over an attention-DP group of `world` ranks (the whole job: dep2 or dep4)."""
    return dict(
        world=world,
        rows_per_rank=rows_per_rank,
        trailing=(_HIDDEN,),
        dtype=torch.bfloat16,
        sizes=None,
    )


class Reducescatter(OpWrapper):
    """Sum `input` across every rank in `group`, then keep this rank's slice.

    Every rank passes a tensor of the same full shape; the elementwise sum is
    split along dim 0 in ascending rank order and rank `i` of the group gets
    slice `i`. `sizes` gives each rank's slice height in that order, or is
    `None` when the split is even. The result is a freshly allocated tensor on
    every rank; `input` is left untouched.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The round trip pads every rank's rows to the group's largest count
    # (`_dp_pad_rows`; a captured step is already uniform), so it always takes the
    # uniform form, `sizes=None`, over the whole job: [world * rows, 2560] bf16 in,
    # [rows, 2560] out. Rows per rank: 1 and 128 bound the captured decode batches
    # (1..128), 512 is dep4's MTP3 verify (128 requests x 4 rows, also the first
    # draft step's MoE), and 8192 is max_num_tokens -- the largest a rank holds.
    CELLS: tuple[Cell, ...] = (
        Cell(why="dep2 decode: one row per rank, 2 ranks", spec=_dp_scatter(2, 1)),
        Cell(
            why="dep2 decode at the largest captured batch, 128 rows per rank",
            spec=_dp_scatter(2, 128),
        ),
        Cell(
            why="dep2 prefill chunk: max_num_tokens 8192 rows per rank", spec=_dp_scatter(2, 8192)
        ),
        Cell(why="dep4 decode: one row per rank, 4 ranks", spec=_dp_scatter(4, 1)),
        Cell(
            why="dep4 decode at the largest captured batch, 128 rows per rank",
            spec=_dp_scatter(4, 128),
        ),
        Cell(why="dep4 MTP3 verify at batch 128: 4 x 128 rows per rank", spec=_dp_scatter(4, 512)),
        Cell(
            why="dep4 prefill chunk: max_num_tokens 8192 rows per rank", spec=_dp_scatter(4, 8192)
        ),
    )

    note = """
    Source: compiled-in C++, `cpp/tensorrt_llm/thop/reducescatterOp.cpp`, registered as
    `trtllm::reducescatter` (class `ReducescatterOp`: one `ncclReduceScatter` for an even
    split, a grouped `ncclReduce` per rank for a ragged one) and `trtllm::reducescatter_pg`
    (class `ReducescatterPgOp`, c10d's `ProcessGroup::reduce_scatter`). Nothing is
    JIT-compiled.

    The certified path is the MPI-session symbol, `process_group=None`: what the target
    calls when its engine runs under MPI, as a served dep2 / dep4 engine does. Its
    communicator comes from `getComm(group)` (`cpp/tensorrt_llm/common/opUtils.cpp`),
    built on the group's first call -- the root's NCCL unique id sent over the MPI
    session, then `ncclCommInitRank`, with `NCCL_RUNTIME_CONNECT=0` and
    `NCCL_GRAPH_REGISTER=0` -- and cached per rank set, so the round trip's all-gather and
    this scatter share one communicator. That first call cannot be stream-captured; the
    engine's warm-up forward makes it eagerly before any graph is captured, and the
    matrix does the same.

    Measured on GB300 by the matrix
    (`tests/unittest/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_103__unified/comm/`),
    one mpirun job per world size:
      cells     bitwise against `reference` on payloads whose every partial sum is exact
                (k/8, |k| <= 31), guard armed: dep2 at 1, 128 and 8192 rows per rank,
                dep4 at 1, 128, 512 and 8192.
      numerics  on standard-normal bf16 payloads at every cell's size the result is
                deterministic and bitwise the sequential sum in the input dtype, in ring
                order from the destination's successor. At two ranks that is the correctly
                rounded sum (no element differs); at four, 0.329-0.341 of elements differ
                from it. Every element stays inside (world - 1) * 2^-8 * sum|x_r|, at most
                0.996 of it at dep2 and 0.908 at dep4.
      graphs    captured at all 19 decode batch sizes (1, 2, 4, 8..128) in one pool and
                replayed in four orders, synchronized and as a burst; 48 round trips per
                graph (gather, a window, this scatter: the trunk's MoE layers); dep MTP3's
                graph per batch b, 49 round trips over 4b rows per rank and 2 over b (up to
                512 rows); replays interleaved with eager prefill-sized calls (up to 8192
                rows per rank) and ragged calls on the same communicator. All bitwise.
      padding   the round trip's padded rows (each rank padded to the group's largest
                count) come back zero and are dropped; up to 8192 rows against 3.

    The ragged form, the other dtypes in `_SUMMABLE_DTYPES`, rank subsets as `group`,
    and the call-order behaviour (calls pair by position on the communicator; a swapped
    pair is wrong on every rank, deterministically, and realigns once both calls are made)
    are driven by the matrix too, though the target passes neither `sizes` nor a subset.

    Not certified: the `_pg` symbol at these cells (the target calls it only with MPI
    disabled; the catalog entry's matrix drives it at hidden 2560 over four ranks, not as
    cells); a capture of the group's first call; rows past 8192 per rank; dep MTP depths
    other than 3 (MTP7's 8 rows per request); groups spanning nodes; sm_100.
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
        #   `TorchUtils::dataType()`, which throws. In `ReducescatterOp::run_list`
        #   that call sits between `ncclGroupStart()` and `ncclGroupEnd()`, so the
        #   group is never closed: the communicator is left mid-group, and the
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
