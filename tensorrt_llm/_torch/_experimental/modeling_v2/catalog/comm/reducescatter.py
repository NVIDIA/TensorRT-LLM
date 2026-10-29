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

# Mistral Large 4 NVFP4 at dep4: the combine after the routed experts of every MoE
# layer (1..71). Each of the four ranks passes the runner's finalized
# [4 x T_max, 6144] bf16 partial and keeps its T_max rows, sizes=None, on the
# MPI-session communicator a dep4 world launched under mpirun reaches.
_DEP4_COMBINE = dict(
    trailing=(6144,),
    dtype=torch.bfloat16,
    sizes=None,
    world_size=4,
    process_group=None,
)
# The decode batches the dep4 engine captures per rank (padded), and the
# combines one captured decode step issues.
_DEP4_GRAPH_ROWS = (1, 2, 4, 8, 16)
_DEP4_MOE_LAYERS = 71

# Mistral Large 4 NVFP4 at dep8: the same combine over the eight attention-DP ranks
# of two GB300 nodes of four (one NVL72 segment, so NCCL forms an MNNVL clique
# of 8), group [0..7]. Each rank passes the runner's finalized [8 x T_max, 6144]
# bf16 partial its 16-expert window wrote for every gathered token and keeps
# its T_max rows, sizes=None, on the MPI-session communicator an eight-rank
# world launched under mpirun across both nodes reaches. `nodes` is the number
# of hosts the ranks span.
_DEP8_COMBINE = dict(
    trailing=(6144,),
    dtype=torch.bfloat16,
    sizes=None,
    world_size=8,
    process_group=None,
    nodes=2,
)
# The decode batches a dep8 engine captures per rank -- the gates' padded
# 1/2/4/8/16 and the LLM API's unpadded default at max_batch_size 16, every
# count 1..16 -- and the combines one captured decode step issues.
_DEP8_GRAPH_ROWS = tuple(range(1, 17))
_DEP8_MOE_LAYERS = 71


class Reducescatter(OpWrapper):
    """Sum `input` across every rank in `group`, then keep this rank's slice.

    Every rank passes a tensor of the same full shape; the elementwise sum is
    split along dim 0 in ascending rank order and rank `i` of the group gets
    slice `i`. `sizes` gives each rank's slice height in that order, or is
    `None` when the split is even. The result is a freshly allocated tensor on
    every rank; `input` is left untouched.
    """

    ARCHS = frozenset({Arch.SM_103})

    # deepseek-r1 and Mistral Large 4 NVFP4 at dep4 and dep8 are the callers, and
    # all pass `sizes=None`: they pad to a common height first. The row counts
    # are the full heights an attention-DP step reduces, which is the rank count
    # times a slice. The first three cells drove the `_pg` symbol (a boxed c10d
    # group); every cell stating `process_group=None` drove the MPI-session
    # symbol, at the world size it states.
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
        *(
            Cell(
                why=f"dep4 MoE combine, decode graph at {rows} row(s) per rank: captured with "
                f"{_DEP4_MOE_LAYERS} combines (MoE layers 1..71) beside the other captured "
                "batches in one pool; every replay on new data bitwise equal to the reference "
                "and to eager",
                spec=dict(
                    rows_per_rank=rows,
                    graph_rows=_DEP4_GRAPH_ROWS,
                    sites=_DEP4_MOE_LAYERS,
                    **_DEP4_COMBINE,
                ),
            )
            for rows in _DEP4_GRAPH_ROWS
        ),
        Cell(
            why="dep4 MoE combine, eager rows_per_rank every value 1..2048: every protocol and "
            "channel-count switch NCCL's ring makes at this width (LL through 154 rows, LL128 "
            "155..990, SIMPLE from 991), each bitwise equal to the reference",
            spec=dict(rows_per_rank=2048, sweep_rows=(1, 2048), sweep_step=1, **_DEP4_COMBINE),
        ),
        Cell(
            why="dep4 MoE combine, eager rows_per_rank 2049..32768: a stride of 61 rows through "
            "the SIMPLE range plus every 2^k - 1, 2^k and 2^k + 1 up to max_num_tokens 32768 "
            "(input 131072 x 6144), each bitwise equal to the reference",
            spec=dict(
                rows_per_rank=32768,
                sweep_rows=(2049, 32768),
                sweep_step=61,
                sweep_also=(4095, 4096, 4097, 8191, 8192, 8193, 16383, 16384, 16385, 32767),
                **_DEP4_COMBINE,
            ),
        ),
        Cell(
            why="dep4 MoE combine at max_num_tokens on every rank (rows_per_rank 32768): a "
            "random payload stays inside the bf16 sequential-sum band of the reference and "
            "repeats bitwise; the input is left untouched and the output is fresh",
            spec=dict(rows_per_rank=32768, payload="random", **_DEP4_COMBINE),
        ),
        *(
            Cell(
                why=f"dep8 MoE combine, decode graph at {rows} row(s) per rank, 8 ranks over two "
                f"nodes: captured with {_DEP8_MOE_LAYERS} combines (MoE layers 1..71) beside the "
                "other captured batches 1..16 in one pool, after an eager warmup of the group; "
                "every replay on new data bitwise equal to the reference and to eager",
                spec=dict(
                    rows_per_rank=rows,
                    graph_rows=_DEP8_GRAPH_ROWS,
                    sites=_DEP8_MOE_LAYERS,
                    **_DEP8_COMBINE,
                ),
            )
            for rows in _DEP8_GRAPH_ROWS
        ),
        Cell(
            why="dep8 MoE combine, eager rows_per_rank every value 1..2048, 8 ranks over two "
            "nodes: every protocol and channel-count switch NCCL's ring makes at this width (LL "
            "through 96 rows, LL128 97..1190, SIMPLE from 1191), each bitwise equal to the "
            "reference",
            spec=dict(rows_per_rank=2048, sweep_rows=(1, 2048), sweep_step=1, **_DEP8_COMBINE),
        ),
        Cell(
            why="dep8 MoE combine, eager rows_per_rank 2049..32768, 8 ranks over two nodes, one "
            "NCCL tuning bucket (ring, SIMPLE, 32 channels): a stride of 61 rows plus every "
            "2^k - 1, 2^k and 2^k + 1 and rows_per_rank 21845/21846/21847, where the input passes "
            "2^31 bytes, up to max_num_tokens 32768 (input 262144 x 6144, 3,221,225,472 bytes), "
            "each bitwise equal to the reference",
            spec=dict(
                rows_per_rank=32768,
                sweep_rows=(2049, 32768),
                sweep_step=61,
                sweep_also=(
                    4095,
                    4096,
                    4097,
                    8191,
                    8192,
                    8193,
                    16383,
                    16384,
                    16385,
                    21845,
                    21846,
                    21847,
                    32767,
                ),
                **_DEP8_COMBINE,
            ),
        ),
        Cell(
            why="dep8 MoE combine at max_num_tokens on every rank (rows_per_rank 32768), 8 ranks "
            "over two nodes: a random payload stays inside the bf16 sequential-sum band of the "
            "reference at W = 8 and repeats bitwise; the input is left untouched and the output "
            "is fresh",
            spec=dict(rows_per_rank=32768, payload="random", **_DEP8_COMBINE),
        ),
    )

    note = """
    Two communicator flavours, one per symbol (see `__call__`). The cells that
    state `process_group=None` drove `torch.ops.trtllm.reducescatter` on the
    MPI-session communicator: at `world_size=4` four ranks of one node started
    by mpirun, as a dep4 world is; at `world_size=8` eight ranks over two GB300
    nodes of four (`nodes=2`, one NVL72 segment) started by the session's MPI
    launcher across both nodes, as a dep8 world is.
    `getComm` (`cpp/tensorrt_llm/common/opUtils.cpp`) builds and caches one
    NCCL communicator per `group` set at that group's first call, through the
    MPI session, and sets NCCL_RUNTIME_CONNECT=0 and NCCL_GRAPH_REGISTER=0 where
    they are unset. The cells without the key drove `reducescatter_pg` on a
    boxed c10d group.

    Preconditions on the MPI-session symbol: every rank of `group` makes the
    same calls in the same order at the same shape. Calls pair by position on
    the communicator; the op matrix's call-order checks measure, on the `_pg`
    symbol, that a mispairing at equal byte counts returns wrong data silently.
    The group's first call builds the communicator, so it is made eagerly: the
    graph cells warm the group before capture, as the engine's warmup does.
    Capturing a group's first call is not certified.

    Numerics, measured on 4 sm_103 ranks at width 6144 bf16. NCCL's tuner (its
    TUNING log) picks the RING algorithm at every rows_per_rank 1..1100 and at
    every power of two up to 32768: LL through 154 rows, LL128 from 155, SIMPLE
    from 991, with the channel count changing many times below 622 rows; the
    captured batches 1/2/4/8/16 get eager's algorithm, protocol and channels.
    On random payloads at 1, 16, 256, 2048 and 32768 rows the result is the
    bf16 chain in ring order from the destination's successor, deterministic,
    and bitwise the same on both symbols. It stays inside
    (W - 1) * 2^-8 * sum_r |x_r| of the exact sum (largest ratio 0.956, at 32768
    rows) and differs from an fp32 sum rounded once in about a third of the
    elements.

    Every rows_per_rank 1..32768 at width 6144 was driven once on the
    MPI-session symbol, bitwise equal to the reference: 39.1 s, past one cell's
    budget, so the cells keep every value to 2048 and a stride above it.

    Measured on 8 sm_103 ranks over two GB300 nodes at width 6144 bf16: the
    eight ranks form one MNNVL clique (NCCL reports nNodes 1 and localRanks 8),
    and every one of the communicator's 32 rings runs 0 1 2 ... 7, each link
    P2P/MNNVL. Rank 0's TUNING log (NCCL logs the choice on rank 0 only), every
    rows_per_rank to 2048 and every 8th to 32768: the RING algorithm at every
    size -- NVLS is available on the devices and never chosen for this op -- LL
    through 96 rows, LL128 from 97, SIMPLE from 1191, with the channel count
    moving between 3 and 32 below 311 rows and 32 from there on; the captured
    batches 1..16 get eager's algorithm, protocol and channels. On random
    payloads at 1, 16, 256, 2048, 21846 and 32768 rows the result is again the
    bf16 chain in ring order from the destination's successor, deterministic,
    on all eight ranks. It stays inside (W - 1) * 2^-8 * sum_r |x_r| of the
    exact sum, W = 8 (largest ratio 0.655, at 2048 rows; 0.649 at 32768 rows
    over three payloads), and differs from an fp32 sum rounded once in about
    half of the elements.

    Every rows_per_rank 1..32768 was driven once on the MPI-session symbol at
    world 8 and each slice was the reference bit for bit on all eight ranks
    (77.2 s), so the world-8 cells too keep every value to 2048 and a stride
    above it.

    The ragged form and dtypes other than bf16 work and are exercised by the op
    matrix on the `_pg` symbol, but no shipped target reaches them, so they are
    not cells. Not certified on the MPI-session symbol: a ragged `sizes`, other
    dtypes and widths, a group other than all four ranks of one node or all
    eight ranks of two nodes, other world sizes and node counts, and
    rows_per_rank above 32768.

    Hard limits (sm_103), the uniform split at width 6144 bf16
      input bytes   4 * rows_per_rank * 12288 vs 2^31   reached at rows_per_rank 43691
                    not reached by dep4 (32768: 1,610,612,736 bytes)   not probed
      input numel   4 * rows_per_rank * 6144 vs 2^31    reached at rows_per_rank 87382
                    not reached   not probed
      NCCL count    output.numel() as size_t; no index passes through fp32

    Hard limits (sm_103), the uniform split at width 6144 bf16, world 8 over two nodes
      input bytes   8 * rows_per_rank * 12288 vs 2^31   reached at rows_per_rank 21846,
                    inside dep8 (32768: 3,221,225,472 bytes)   probed at every
                    rows_per_rank 21845..32768 once, and kept as cells at 21845, 21846,
                    21847, a stride of 61 and 32767/32768   passed, bitwise on all
                    eight ranks
      input numel   8 * rows_per_rank * 6144 vs 2^31    reached at rows_per_rank 43691
                    not reached by dep8 (32768: 1,610,612,736)   not probed
      NCCL count    output.numel() as size_t; no index passes through fp32
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
