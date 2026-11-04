# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the Qwen3.8-Flash-Next-FP8 target's reducescatter__dp_scatter.

Adapted from the catalog entry's matrix (`tests/unittest/_torch/modeling_v2/comm/
_reducescatter_op_matrix.py`) to the call this target makes: the scatter half of the
attention-DP MoE round trip, `_expert_sum_op.round_trip` -- this rank's rows padded to the
group's largest count, all-gathered, run through this rank's expert window, and
reduce-scattered back -- at dep2 and dep4.

That call runs under MPI: a served dep2 / dep4 engine is started by `mpirun`, so the target
passes `process_group=None` and the entry calls the MPI-session symbol
`torch.ops.trtllm.reducescatter`, whose communicator `getComm(group)` builds on the group's
first call. The catalog matrix drives the `_pg` symbol over a torch.distributed group instead
-- a different C++ class (`ReducescatterPgOp`, c10d's `reduce_scatter`) -- so this one starts
its ranks under `mpirun` and drives the MPI-session symbol throughout.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _reducescatter__dp_scatter_op_matrix.py --world-size 4

The launcher re-executes this file under `mpirun -n <world size>`, one rank per device named
in CUDA_VISIBLE_DEVICES, under a deadline it enforces by killing the process group: a
collective that breaks hangs rather than raising. Every `CHECKS` entry runs in order on every
rank, and the job must exit 0.

Not a pytest module, despite the `check_*` bodies: they are one fixed sequence inside a single
job, each reading rank state only `_run_one_rank` binds, and several assert on communicator
state the previous one left behind. Neither this file's name nor its function names match
pytest's collection patterns. The collected entry point is
`test_fp8_unified_op_matrices.py` beside this file, which starts one job per world size and
turns its exit code into an assertion.
"""

import argparse
import ctypes
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

assert torch.cuda.is_available(), "reducescatter__dp_scatter requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 1800

#: The Slurm step variables a nested `mpirun` must not inherit: with them every rank reads
#: the launching step's single task and believes it is rank 0 of a world of 1.
_SLURM_STEP_VARS = (
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "SLURM_LOCALID",
    "SLURM_NODEID",
    "SLURM_STEP_ID",
    "SLURM_TASK_PID",
    "SLURM_STEP_NUM_TASKS",
)

HIDDEN = 2560  # Qwen3.8-Flash-Next hidden_size: the width of every window partial

# The per-rank row counts a decode CUDA graph is captured at: the engine's default
# `cuda_graph_config` batch sizes at max_batch_size 128 ([1, 2, 4] and every multiple of 8
# up to 128). Under attention DP a decode step pads every rank to the captured batch, so a
# captured call's per-rank rows *are* that batch size, and its input is WORLD times that.
GRAPH_BATCH_SIZES: Tuple[int, ...] = (1, 2, 4) + tuple(range(8, 129, 8))

# Round trips inside one decode graph: every one of the checkpoint's 48 layers is a MoE
# layer, each completed by one gather / window / scatter round trip.
MOE_LAYERS = 48

# dep4's MTP3 decode step, per captured batch size b: the trunk's 48 MoE layers run over
# the (1 + 3) * b verify rows, the draft layer's first step over the same rows and its two
# later steps over b rows (`_draft_pad_rows`: trunk tokens first, then one row per request).
MTP3_SITE_ROWS: Tuple[int, ...] = (4,) * (MOE_LAYERS + 1) + (1, 1)

# bf16 unit roundoff: the format carries 8 significand bits, so eps = 2^-8 and
# a round-to-nearest step is off by at most 2^-9 of the running value. Used by
# the accumulation-order test's error budget.
BF16_U = 2.0**-8

# Bound inside the rank body, never in the launcher: importing the entry pulls in
# tensorrt_llm, and the launcher must hold no MPI or CUDA state of its own.
reducescatter: Any = None
allgather: Any = None
entry: Any = None
validating: Any = None
COMM: Any = None
RANK = 0
WORLD = 1
GROUP: List[int] = [0]
TIMINGS: Dict[str, float] = {}


def _barrier() -> None:
    torch.cuda.synchronize()
    COMM.Barrier()


def _block(
    rank: int,
    dest: int,
    rows: int,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
    trailing: Tuple[int, ...] = (HIDDEN,),
    amp: int = 31,
) -> torch.Tensor:
    """The part of `rank`'s input that ends up on group position `dest`.

    Every rank's input is the concatenation of one block per destination, so a
    rank can regenerate exactly the WORLD blocks its own result is the sum of
    without materializing anybody's whole input -- cuRAND is deterministic for a
    given (seed, shape, dtype). That is what lets the reference below be
    arithmetic rather than a second collective.

    The float value set is multiples of 1/8 with |x| <= amp/8. bf16, fp16 and
    fp32 all represent k/8 exactly for |k| <= 255, and a sum of WORLD <= 4 of them
    reaches |k| <= 4 * 31 = 124, so **every partial sum is exact in every
    summation order**. That is what lets the assertions be bitwise despite the
    op summing in the input dtype.
    """
    gen = torch.Generator(device="cuda").manual_seed((seed * 977 + rank) * 131 + dest + 1)
    shape = (rows, *trailing)
    if dtype in (torch.uint8, torch.int8, torch.int32, torch.int64):
        lo, hi = (0, 8) if dtype is torch.uint8 else (-15, 16)
        return torch.randint(lo, hi, shape, generator=gen, device="cuda", dtype=torch.int32).to(
            dtype
        )
    raw = torch.randint(-amp, amp + 1, shape, generator=gen, device="cuda", dtype=torch.int32)
    return (raw.float() / 8.0).to(dtype)


def _input(
    rank: int,
    sizes: Sequence[int],
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
    trailing: Tuple[int, ...] = (HIDDEN,),
    amp: int = 31,
) -> torch.Tensor:
    """One rank's whole contribution: every destination's block, concatenated."""
    return torch.cat(
        [_block(rank, d, n, seed, dtype, trailing, amp) for d, n in enumerate(sizes)],
        dim=0,
    )


def _ref(
    sizes: Sequence[int],
    pos: int,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
    trailing: Tuple[int, ...] = (HIDDEN,),
    ranks: Optional[Sequence[int]] = None,
    amp: int = 31,
) -> torch.Tensor:
    """Arithmetic reference: the sum this group position is supposed to get.

    Accumulated in fp32 (int64 for integer dtypes) on this rank alone, from the
    same seeds every rank uses. Never from a second collective.
    """
    ranks = list(range(WORLD)) if ranks is None else list(ranks)
    # Delegates to the entry: what the sum is supposed to be is the entry's claim.
    return entry.reducescatter.reference(
        [_block(r, pos, sizes[pos], seed, dtype, trailing, amp) for r in ranks]
    )


def _assert_bitwise(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    """Gate of exactly zero, on payloads whose every partial sum is exact.

    The reduction is a sum in the input dtype, but `_block`'s value set makes every
    intermediate representable, so any difference at all is a wrong reduction or a
    wrong slice, not rounding.
    """
    try:
        assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
    except AssertionError as exc:
        raise AssertionError(f"{where}: {exc}") from exc


def _sizes_vectors() -> List[List[int]]:
    """Per-rank row counts an attention-DP step produces, adapted to WORLD."""
    return [
        [1] * WORLD,  # uniform, but stated as an explicit sizes vector
        [1 + 4 * r for r in range(WORLD)],  # steadily uneven decode batches
        [7 * (WORLD - r) for r in range(WORLD)],  # uneven the other way
        [0] + [3 + 2 * r for r in range(WORLD - 1)],  # one rank with no rows
        [2048] + [1 + 388 * r for r in range(WORLD - 1)],  # prefill-sized, lopsided
    ]


def _mispaired_ref(sizes: Sequence[int], pos: int, seeds: Sequence[int]) -> torch.Tensor:
    """The sum a call actually computes when the ranks disagree on call order.

    `seeds[r]` is the payload rank `r` happened to be holding when this position on
    the communicator came round. With `_block`'s value set every partial sum is
    exact, so the prediction is bitwise regardless of accumulation order.
    """
    acc = torch.zeros((sizes[pos], HIDDEN), dtype=torch.float32, device="cuda")
    for rank, seed in enumerate(seeds):
        acc += _block(rank, pos, sizes[pos], seed).float()
    return acc.to(torch.bfloat16)


def _rand(rank: int, dest: int, rows: int, seed: int) -> torch.Tensor:
    """Payload with no exactness property, for the accumulation-order tests."""
    gen = torch.Generator(device="cuda").manual_seed((seed * 977 + rank) * 131 + dest + 1)
    return torch.randn((rows, HIDDEN), generator=gen, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )


def _rand_input(rank: int, rows: int, seed: int) -> torch.Tensor:
    """One rank's whole contribution, drawn from the inexact value set."""
    return torch.cat([_rand(rank, d, rows, seed) for d in range(WORLD)], dim=0)


def _ring_chain(rows: int, pos: int, seeds: Sequence[int]) -> torch.Tensor:
    """Ring-order sequential sum for group position `pos`, rounded every step:
    `x_{pos+1} + x_{pos+2} + ... + x_{pos+G-1} + x_pos`, indices mod WORLD, where
    `x_r` is rank `r`'s block for `pos` drawn from `seeds[r]`."""
    first = (pos + 1) % WORLD
    chain = _rand(first, pos, rows, seeds[first]).clone()
    for k in range(2, WORLD + 1):
        rank = (pos + k) % WORLD
        chain = chain + _rand(rank, pos, rows, seeds[rank])
    return chain


def _swapped_pair(
    first: torch.Tensor, second: torch.Tensor, sizes: Optional[List[int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Issue two calls, with rank 0 issuing them in the opposite order.

    Returns the results of communicator **positions** 0 and 1 -- which is not the
    same as the results of `first` and `second`, and that is the point.
    """
    if RANK == 0:
        return (
            reducescatter(second, sizes, GROUP),
            reducescatter(first, sizes, GROUP),
        )
    return reducescatter(first, sizes, GROUP), reducescatter(second, sizes, GROUP)


def _reduce_scatter_on_a_side_stream(x: torch.Tensor, side: torch.cuda.Stream) -> torch.Tensor:
    """Issue one call with `side` current, joined to the caller on both ends."""
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        out = reducescatter(x, None, GROUP)
    torch.cuda.current_stream().wait_stream(side)
    return out


def _fraction_wrong(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Share of elements that differ, over the rows the two tensors share."""
    rows = min(got.shape[0], ref.shape[0])
    assert rows > 0
    return (got[:rows] != ref[:rows]).float().mean().item()


def _window_scale(rank: int) -> float:
    """Rank `rank`'s stand-in expert window: a scale of (rank + 1) / 4.

    With own rows from `_block(..., amp=7)` (k/8, |k| <= 7) every window partial is
    k (r + 1) / 32 and every partial sum over the group is k m / 32 with m <= 10, so
    |k m| <= 70 stays exact in bf16 and a round trip can be gated bitwise.
    """
    return float(rank + 1) / 4.0


def _window_total() -> float:
    return sum(_window_scale(r) for r in range(WORLD))


def _round_trip(mine: torch.Tensor, pad_rows: int = 0) -> torch.Tensor:
    """The target's `_expert_sum_op.round_trip` with a scale for the expert window:
    pad, gather (the MPI-session all-gather over the same group), run the window over
    the gathered rows, reduce-scatter, drop the padding."""
    rows = mine.shape[0]
    padded = mine if not pad_rows else torch.nn.functional.pad(mine, [0, 0, 0, pad_rows])
    gathered = allgather(padded, None, GROUP)
    partial = (gathered.float() * _window_scale(RANK)).to(torch.bfloat16)
    out = reducescatter(partial, None, GROUP)
    return out if not pad_rows else out[:rows]


def _warm_up_off_capture_stream(body: Callable[[], Any], reps: int = 2) -> None:
    """Run `body` on a side stream, then rejoin, so a capture can follow.

    The MPI-session symbol builds its communicator on the group's first call, which
    a capture cannot contain, and torch wants the work warmed on a non-default stream.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(reps):
            body()
    torch.cuda.current_stream().wait_stream(side)
    _barrier()


def _capture_per_batch_size(
    site_rows: Sequence[int], seed: int, round_trip: bool
) -> Tuple[Dict[int, List[torch.Tensor]], Dict[int, Any], Dict[int, List[torch.Tensor]]]:
    """One graph per GRAPH_BATCH_SIZES entry, one shared pool, all left alive.

    That is the state a graph runner ends up in: every capture after the first goes
    into the first one's pool, and none of them is ever destroyed. Site `s` of the
    graph for batch b runs over `site_rows[s] * b` rows per rank and reads its own
    persistent input buffer, so the payload of every site can be moved independently:
    the whole `[WORLD * rows, HIDDEN]` input of a bare reduce-scatter, or this rank's
    own `[rows, HIDDEN]` rows of a round trip. Returns (inputs, graphs, outputs), all
    keyed by the batch size.
    """
    inputs: Dict[int, List[torch.Tensor]] = {}
    graphs: Dict[int, Any] = {}
    outputs: Dict[int, List[torch.Tensor]] = {}
    pool = None
    for b in GRAPH_BATCH_SIZES:
        xs = [_site_payload(m * b, seed + 13 * s, round_trip) for s, m in enumerate(site_rows)]

        def body(xs: List[torch.Tensor] = xs) -> List[torch.Tensor]:
            if round_trip:
                return [_round_trip(x) for x in xs]
            return [reducescatter(x, None, GROUP) for x in xs]

        # Persistent output buffers, allocated OUTSIDE the shared pool: the op's own
        # output lands in the pool, where a later graph's capture reuses the same
        # address, so under a burst of replays with no sync between them the pooled
        # outputs all end up holding the last writer's result. Copying each into a
        # dedicated buffer inside the capture is ordered after this graph's
        # collective on the one stream, so it reads the right value first.
        persistent = [
            torch.empty(m * b, HIDDEN, dtype=torch.bfloat16, device="cuda") for m in site_rows
        ]

        def capture_body(persistent: List[torch.Tensor] = persistent) -> List[torch.Tensor]:
            produced = body()
            for dst, src in zip(persistent, produced):
                dst.copy_(src)
            return persistent

        _warm_up_off_capture_stream(body)
        graph = torch.cuda.CUDAGraph()
        ctx = torch.cuda.graph(graph) if pool is None else torch.cuda.graph(graph, pool=pool)
        with ctx:
            outputs[b] = capture_body()
        # Without this a capture that returned nothing would make every
        # `_verify_replay` below an empty loop, i.e. a vacuous pass.
        assert len(outputs[b]) == len(site_rows), (b, len(outputs[b]), len(site_rows))
        if pool is None:
            pool = graph.pool()
        inputs[b], graphs[b] = xs, graph
        _barrier()
    return inputs, graphs, outputs


def _site_payload(rows: int, seed: int, round_trip: bool) -> torch.Tensor:
    if round_trip:
        return _block(RANK, 0, rows, seed, amp=7)
    return _input(RANK, [rows] * WORLD, seed)


def _site_expected(x: torch.Tensor, rows: int, seed: int, round_trip: bool) -> torch.Tensor:
    if round_trip:
        return (x.float() * _window_total()).to(torch.bfloat16)
    return _ref([rows] * WORLD, RANK, seed)


def _replay_orders() -> List[Tuple[str, List[int]]]:
    """Four ways a served batch size moves through the captured set."""
    asc = list(GRAPH_BATCH_SIZES)
    shuffled = list(asc)
    random.Random(1234).shuffle(shuffled)
    # The largest jump still available at every step: 128, 1, 120, 2, 112, 4...
    extremes: List[int] = []
    lo, hi = 0, len(asc) - 1
    while lo <= hi:
        extremes.append(asc[hi])
        if lo != hi:
            extremes.append(asc[lo])
        lo, hi = lo + 1, hi - 1
    return [
        ("ascending", asc),
        ("descending", list(reversed(asc))),
        ("shuffled", shuffled),
        ("extremes", extremes),
    ]


def _fresh_payload(
    inputs: Dict[int, List[torch.Tensor]],
    b: int,
    site_rows: Sequence[int],
    seed: int,
    round_trip: bool,
) -> None:
    """Overwrite every site's persistent input buffer for this batch size."""
    for s, (x, m) in enumerate(zip(inputs[b], site_rows)):
        x.copy_(_site_payload(m * b, seed + 13 * s, round_trip))


def _verify_replay(
    inputs: Dict[int, List[torch.Tensor]],
    outputs: Dict[int, List[torch.Tensor]],
    b: int,
    site_rows: Sequence[int],
    seed: int,
    round_trip: bool,
    where: str,
) -> None:
    """Every site's captured output tensor holds this payload's result, bitwise."""
    for s, (x, out, m) in enumerate(zip(inputs[b], outputs[b], site_rows)):
        _assert_bitwise(
            out, _site_expected(x, m * b, seed + 13 * s, round_trip), f"{where} site={s}"
        )


def _replay_two_ways(
    inputs: Dict[int, List[torch.Tensor]],
    graphs: Dict[int, Any],
    outputs: Dict[int, List[torch.Tensor]],
    orders: List[Tuple[str, List[int]]],
    site_rows: Sequence[int],
    seed: int,
    round_trip: bool,
) -> int:
    """For each order: one replay at a time with a synchronize and a full check
    before the next graph runs, then the whole order issued back to back with no
    synchronization in between -- every graph queued as one run of work, the only
    pass that can catch a hazard needing the next call to already be in flight.
    Each replay gets a payload no earlier call used, so a graph that silently did
    not re-run keeps the previous answer and fails the bitwise gate."""
    for name, order in orders:
        for pos, b in enumerate(order):
            seed += 1000
            _fresh_payload(inputs, b, site_rows, seed, round_trip)
            graphs[b].replay()
            torch.cuda.synchronize()
            _verify_replay(
                inputs, outputs, b, site_rows, seed, round_trip, f"{name}/checked@{pos} b={b}"
            )
        _barrier()
        staged: Dict[int, int] = {}
        for b in order:
            seed += 1000
            staged[b] = seed
            _fresh_payload(inputs, b, site_rows, seed, round_trip)
        for b in order:
            graphs[b].replay()
        torch.cuda.synchronize()
        for pos, b in enumerate(order):
            _verify_replay(
                inputs, outputs, b, site_rows, staged[b], round_trip, f"{name}/burst@{pos} b={b}"
            )
        _barrier()
    return seed


def check_cuda_graph_captures_after_an_eager_first_call() -> None:
    """The group's first call is eager; a capture after it replays bitwise.

    The MPI-session symbol builds its NCCL communicator on the group's first call
    (`getComm`: the root's unique id sent over MPI, then `ncclCommInitRank`), which a
    stream capture cannot contain. A served engine's first round trip is eager -- its
    warm-up forward runs before any graph is captured -- and so is this job's. Every
    later check reuses the communicator this call builds. Replaying after a fresh
    input must reflect that input, bitwise.
    """
    rows = 16
    sizes = [rows] * WORLD
    seed = 100
    x = _input(RANK, sizes, seed)
    _assert_bitwise(reducescatter(x, None, GROUP), _ref(sizes, RANK, seed), "eager first call")
    _barrier()
    if RANK == 0:
        # `getComm` sets these with C setenv before building the communicator, which
        # os.environ (a snapshot taken at start-up) does not see.
        getenv = ctypes.CDLL(None).getenv
        getenv.restype = ctypes.c_char_p
        settings = {
            name: (getenv(name.encode()) or b"<unset>").decode()
            for name in ("NCCL_NVLS_ENABLE", "NCCL_RUNTIME_CONNECT", "NCCL_GRAPH_REGISTER")
        }
        print(f"[rank 0] NCCL {torch.cuda.nccl.version()}, communicator built with {settings}")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = reducescatter(x, None, GROUP)
    # Capture records the op, it does not run it; the first replay fills `captured`.
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(captured, _ref(sizes, RANK, seed), "first capture")

    # A second replay after a refilled input reflects that input, not the first.
    x.copy_(_input(RANK, sizes, seed + 500))
    _barrier()
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(captured, _ref(sizes, RANK, seed + 500), "replay after refill")
    del graph
    _barrier()


def check_certified_cells() -> None:
    """Drive the entry's cells at this job's world size, inside `validating`.

    The rest of this file says things no cell can -- CUDA graph behaviour, call
    order, which stream the call lands on. This one says the plain thing: for every
    configuration the entry claims at this world size, the slice is the sum,
    bitwise, and the guard admits the input this target passes.
    """
    cells = [c for c in entry.reducescatter.CELLS if c.spec["world"] == WORLD]
    assert cells, f"the entry certifies no cell at world size {WORLD}: this job would be vacuous"
    for i, cell in enumerate(cells):
        spec = cell.spec
        rows, trailing, dtype = spec["rows_per_rank"], spec["trailing"], spec["dtype"]
        seed = 9000 + 13 * i
        sizes = [rows] * WORLD
        x = _input(RANK, sizes, seed, dtype, trailing)
        with validating(entry.reducescatter):
            out = reducescatter(x, spec["sizes"], GROUP)
        assert out.shape == (rows, *trailing), (cell.why, out.shape)
        assert out.dtype is dtype and out.is_contiguous(), (cell.why, out.dtype)
        _assert_bitwise(out, _ref(sizes, RANK, seed, dtype, trailing), cell.why)
        _barrier()
        if RANK == 0:
            print(
                f"[rank 0] cell {cell.why!r}: [{WORLD * rows}, {trailing[0]}] bitwise", flush=True
            )


def check_uniform_reduce_scatter() -> None:
    """sizes=None: every rank sends WORLD * rows rows and keeps `rows` of them."""
    for rows in (1, 2, 8, 32, 2048):
        seed = 1000 + rows
        sizes = [rows] * WORLD
        x = _input(RANK, sizes, seed)
        assert x.shape == (rows * WORLD, HIDDEN), x.shape
        out = reducescatter(x, None, GROUP)
        assert out.shape == (rows, HIDDEN), out.shape
        assert out.dtype is x.dtype and out.device == x.device
        assert out.is_contiguous()
        _assert_bitwise(out, _ref(sizes, RANK, seed), f"uniform rows={rows}")
        _barrier()


def check_ragged_reduce_scatter() -> None:
    """sizes=[...]: the split is uneven, this rank keeps sizes[my position].

    The target pads to the group's largest count and never passes `sizes`; the
    ragged form is the op's other path, driven so a caller knows it agrees.
    """
    for i, sizes in enumerate(_sizes_vectors()):
        seed = 3000 + 7 * i
        x = _input(RANK, sizes, seed)
        assert x.shape == (sum(sizes), HIDDEN), x.shape
        out = reducescatter(x, sizes, GROUP)
        assert out.shape == (sizes[RANK], HIDDEN), (sizes, out.shape)
        assert out.dtype is x.dtype and out.is_contiguous()
        _assert_bitwise(out, _ref(sizes, RANK, seed), f"ragged sizes={sizes}")
        _barrier()


def check_output_is_fresh_and_input_is_untouched() -> None:
    """The op allocates its result; the caller keeps ownership of `input`."""
    rows, seed = 8, 4100
    sizes = [rows] * WORLD
    x = _input(RANK, sizes, seed)
    before = x.clone()
    out = reducescatter(x, None, GROUP)
    torch.cuda.synchronize()
    assert out.data_ptr() != x.data_ptr(), "output aliased the input buffer"
    _assert_bitwise(x, before, "input after the call")

    # Clobbering the input afterwards cannot reach the result.
    x.fill_(-7.0)
    torch.cuda.synchronize()
    _assert_bitwise(out, _ref(sizes, RANK, seed), "output after input clobber")
    _barrier()


def check_trailing_dims_are_preserved() -> None:
    """dim 0 is the scatter axis; every other dim is carried through untouched."""
    cases: List[Tuple[Tuple[int, ...], bool]] = [
        ((2, HIDDEN // 2), False),
        ((2, HIDDEN // 2), True),
        ((), False),  # 1-D: a flat vector scatters into WORLD pieces
        ((), True),
    ]
    for i, (trailing, ragged) in enumerate(cases):
        seed = 5000 + 11 * i
        sizes = [1 + 4 * r for r in range(WORLD)] if ragged else [6] * WORLD
        x = _input(RANK, sizes, seed, trailing=trailing)
        out = reducescatter(x, sizes if ragged else None, GROUP)
        assert out.shape == (sizes[RANK], *trailing), (trailing, sizes, out.shape)
        _assert_bitwise(
            out,
            _ref(sizes, RANK, seed, trailing=trailing),
            f"trailing={trailing} ragged={ragged}",
        )
        _barrier()


def check_dtypes_reduce_arithmetically() -> None:
    """The dtypes whose sum this op actually computes, in both forms."""
    ragged = [1 + 4 * r for r in range(WORLD)]
    uniform = [16] * WORLD
    for i, dtype in (
        (0, torch.float16),
        (1, torch.float32),
        (2, torch.int32),
        (3, torch.int64),
        (4, torch.uint8),
        (5, torch.int8),
    ):
        seed = 6000 + 13 * i
        x = _input(RANK, uniform, seed, dtype)
        out = reducescatter(x, None, GROUP)
        assert out.dtype is dtype and out.shape == (16, HIDDEN)
        _assert_bitwise(out, _ref(uniform, RANK, seed, dtype), f"uniform {dtype}")
        _barrier()

        xr = _input(RANK, ragged, seed, dtype)
        outr = reducescatter(xr, ragged, GROUP)
        assert outr.dtype is dtype and outr.shape == (ragged[RANK], HIDDEN)
        _assert_bitwise(outr, _ref(ragged, RANK, seed, dtype), f"ragged {dtype}")
        _barrier()


def check_group_selects_a_rank_subset() -> None:
    """`group` names the communicator's ranks; the slice index is the position in it.

    Over the MPI session a subset is its own communicator (`getComm` keys its cache
    on the rank set), built by the member ranks alone. The second subset excludes
    rank 0, so a rank that indexed the split by its global rank instead of by its
    position in the group would read a different slice. Ranks outside a subset do
    not call. At world size 2 the only subset is the whole group.
    """
    subsets = [[0, 1]]
    if WORLD >= 4:
        subsets.append([WORLD - 2, WORLD - 1])
    for i, subset in enumerate(subsets):
        rows, seed = 4, 7100 + 31 * i
        sizes = [rows] * len(subset)
        if RANK in subset:
            pos = subset.index(RANK)
            out = reducescatter(_input(RANK, sizes, seed), None, subset)
            assert out.shape == (rows, HIDDEN), out.shape
            _assert_bitwise(out, _ref(sizes, pos, seed, ranks=subset), f"subset {subset} uniform")
            ragged = [2, 6] if len(subset) == 2 else [2] * len(subset)
            outr = reducescatter(_input(RANK, ragged, seed), ragged, subset)
            assert outr.shape == (ragged[pos], HIDDEN), outr.shape
            _assert_bitwise(outr, _ref(ragged, pos, seed, ranks=subset), f"subset {subset} ragged")
        _barrier()


def check_group_order_does_not_change_the_output() -> None:
    """The split is ordered by ascending rank, whatever order `group` lists."""
    rows, seed = 4, 7200
    sizes = [rows] * WORLD
    x = _input(RANK, sizes, seed)
    out = reducescatter(x, None, list(reversed(GROUP)))
    _assert_bitwise(out, _ref(sizes, RANK, seed), "reversed group list")
    _barrier()


def check_reduction_is_deterministic_and_accumulates_in_the_input_dtype() -> None:
    """What the sum is, exactly -- the fact the target's accuracy gates rest on.

    On payloads with no exactness property (standard normal, cast to bf16), the
    expert windows' partials being bf16 too:

    1. It is **deterministic**: eight identical eager calls return bitwise
       identical tensors, and so do eight replays of a captured call.
    2. It is a **sequential summation in the input dtype, in ring order** starting
       at the destination's successor -- matched bitwise on every element. That
       order is NCCL's choice for this topology; if this assertion ever fails, the
       reduction order changed and the entry's note has to be re-measured rather
       than this gate loosened.
    3. So at two ranks it is the correctly rounded sum (one rounding of two bf16
       addends), and above two it is **not**: a measurable share of elements
       differs from an fp32 reduction rounded once.
    4. Its distance from the exact sum stays inside the classical bound for a
       sequential sum of WORLD terms, `(WORLD-1) * u * sum_r |x_r|`, u = 2^-8.
    """
    for rows in (1, 8, 256):
        seed = 8000 + rows
        x = torch.cat([_rand(RANK, d, rows, seed) for d in range(WORLD)], dim=0)
        outs = [reducescatter(x, None, GROUP) for _ in range(8)]
        torch.cuda.synchronize()
        for k, o in enumerate(outs[1:], start=1):
            _assert_bitwise(o, outs[0], f"repeat {k} at rows={rows}")

        mine = [_rand(r, RANK, rows, seed) for r in range(WORLD)]
        exact = torch.zeros(rows, HIDDEN, dtype=torch.float32, device="cuda")
        absum = torch.zeros(rows, HIDDEN, dtype=torch.float32, device="cuda")
        for chunk in mine:
            exact += chunk.float()
            absum += chunk.float().abs()

        # (2) a sequential chain in ring order, starting at my successor
        chain = mine[(RANK + 1) % WORLD].clone()
        for k in range(2, WORLD + 1):
            chain = chain + mine[(RANK + k) % WORLD]
        _assert_bitwise(outs[0], chain, f"ring-order chain at rows={rows}")

        # (3) one rounding at two ranks, more than one above
        differs = (outs[0] != exact.to(torch.bfloat16)).float().mean().item()
        if WORLD == 2:
            assert differs == 0.0, (
                f"rows={rows}: {differs:.4f} of elements differ from the correctly rounded "
                "sum of two addends"
            )
        else:
            assert differs > 0.05, (
                f"rows={rows}: only {differs:.4f} of elements differ from an fp32 "
                "reduction -- the op may have started accumulating in fp32"
            )

        # (4) inside the classical sequential-summation error bound
        err = (outs[0].float() - exact).abs()
        budget = (WORLD - 1) * BF16_U * absum
        over = (err > budget).sum().item()
        ratio = (err / budget.clamp_min(1e-30)).max().item()
        assert over == 0, (
            f"rows={rows}: {over} elements outside the sequential-summation bound, "
            f"worst ratio {ratio:.4f}"
        )
        if RANK == 0:
            print(
                f"[rank 0] world {WORLD} rows={rows}: {differs:.4f} of elements differ from "
                f"the correctly rounded sum; worst error / bound {ratio:.4f}",
                flush=True,
            )
        _barrier()

    # Determinism under capture, and the captured result equals the eager one.
    rows, seed = 16, 8500
    x = torch.cat([_rand(RANK, d, rows, seed) for d in range(WORLD)], dim=0)
    eager = reducescatter(x, None, GROUP)
    _warm_up_off_capture_stream(lambda: reducescatter(x, None, GROUP))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = reducescatter(x, None, GROUP)
    for k in range(8):
        graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise(captured, eager, f"replay {k} against the eager result")
    del graph
    _barrier()


def check_cell_sized_reductions_on_window_like_payloads() -> None:
    """The numerics of each certified cell's own call, on payloads with no exactness
    property: what a window partial actually is.

    NCCL picks its algorithm and protocol by message size, so the order the small
    calls above are matched in need not hold at a cell's [WORLD * rows, 2560] (up to
    168 MB at dep4's 8192 rows per rank). It does: per cell at this world size, three
    repeats agree bitwise and the result is bitwise the ring-order chain -- at two
    ranks also the correctly rounded sum, above two not -- inside the
    sequential-summation bound.
    """
    cells = [c for c in entry.reducescatter.CELLS if c.spec["world"] == WORLD]
    for i, cell in enumerate(cells):
        rows = cell.spec["rows_per_rank"]
        seed = 12000 + 17 * i
        x = _rand_input(RANK, rows, seed)
        outs = [reducescatter(x, None, GROUP) for _ in range(3)]
        torch.cuda.synchronize()
        for k, o in enumerate(outs[1:], start=1):
            _assert_bitwise(o, outs[0], f"{cell.why}: repeat {k}")
        mine = [_rand(r, RANK, rows, seed) for r in range(WORLD)]
        exact = torch.zeros(rows, HIDDEN, dtype=torch.float32, device="cuda")
        absum = torch.zeros(rows, HIDDEN, dtype=torch.float32, device="cuda")
        for chunk in mine:
            exact += chunk.float()
            absum += chunk.float().abs()
        chain = _ring_chain(rows, RANK, [seed] * WORLD)
        off_chain = _fraction_wrong(outs[0], chain)
        off_rounded = _fraction_wrong(outs[0], exact.to(torch.bfloat16))
        err = (outs[0].float() - exact).abs()
        budget = (WORLD - 1) * BF16_U * absum
        ratio = (err / budget.clamp_min(1e-30)).max().item()
        assert (err > budget).sum().item() == 0, (
            f"{cell.why}: outside the sequential-summation bound, worst ratio {ratio:.4f}"
        )
        every = COMM.allgather([off_chain, off_rounded, ratio])
        if RANK == 0:
            worst = [max(v[j] for v in every) for j in range(3)]
            print(
                f"[rank 0] {cell.why!r}: worst over ranks {worst[0]:.4f} of elements off the "
                f"ring-order chain, {worst[1]:.4f} off the correctly rounded sum, error / "
                f"bound {worst[2]:.4f}",
                flush=True,
            )
        _assert_bitwise(outs[0], chain, f"{cell.why}: the ring-order chain")
        if WORLD == 2:
            assert off_rounded == 0.0, f"{cell.why}: {off_rounded:.4f} off the rounded sum"
        else:
            assert off_rounded > 0.05, f"{cell.why}: only {off_rounded:.4f} off the rounded sum"
        _barrier()


def check_uniform_form_and_an_explicit_even_split_agree_bitwise() -> None:
    """`sizes=None` and an explicit even `sizes` vector return the same bits, on
    payloads where a different reduction order would show."""
    for rows in (1, 8, 256, 2048):
        seed = 9000 + rows
        x = torch.cat([_rand(RANK, d, rows, seed) for d in range(WORLD)], dim=0)
        a = reducescatter(x, None, GROUP)
        b = reducescatter(x, [rows] * WORLD, GROUP)
        torch.cuda.synchronize()
        _assert_bitwise(b, a, f"explicit even split at rows={rows}")
        _barrier()


def check_round_trip_with_a_gather_returns_each_rank_its_own_rows() -> None:
    """The attention-DP MoE round trip, eager, as the target runs it.

    Each rank holds a different row count (an eager prefill step), pads to the
    group's largest (`_dp_pad_rows`), gathers the group's rows with the MPI-session
    all-gather over the same communicator, applies its own expert window (a scale),
    reduce-scatters the partials in the uniform form and drops its padding. The
    claim: rank i gets back exactly the rows it contributed, holding the sum of the
    windows -- and the padding rows it dropped are zero.

    The all-gather is a **fixture**, never the reference; the gathered tensor is
    checked against a local `torch.cat`, so a wrong gather cannot be mistaken for a
    right reduce-scatter. The reference is this rank's own rows times the sum of
    the window scales, computed locally.
    """
    for sizes in ([1 + 4 * r for r in range(WORLD)], [8192] + [3] * (WORLD - 1)):
        seed = 10500 + sizes[0]
        own = [_block(r, 0, sizes[r], seed, amp=7) for r in range(WORLD)]
        mine = own[RANK]
        pad = max(sizes)
        padded = torch.nn.functional.pad(mine, [0, 0, 0, pad - sizes[RANK]])
        gathered = allgather(padded, None, GROUP)
        torch.cuda.synchronize()
        expect_gathered = torch.cat(
            [torch.nn.functional.pad(o, [0, 0, 0, pad - o.shape[0]]) for o in own], dim=0
        )
        _assert_bitwise(gathered, expect_gathered, f"gather fixture sizes={sizes}")
        partial = (gathered.float() * _window_scale(RANK)).to(torch.bfloat16)
        out = reducescatter(partial, None, GROUP)
        torch.cuda.synchronize()
        assert out.shape == (pad, HIDDEN), out.shape
        expected = (mine.float() * _window_total()).to(torch.bfloat16)
        _assert_bitwise(out[: sizes[RANK]], expected, f"round trip sizes={sizes}")
        assert not bool(out[sizes[RANK] :].any()), f"padding rows came back non-zero {sizes}"
        _assert_bitwise(_round_trip(mine, pad - sizes[RANK]), expected, f"_round_trip {sizes}")
        _barrier()

    # Ragged form, and the bare identity: reduce-scattering a gather returns WORLD copies.
    sizes = [1 + 4 * r for r in range(WORLD)]
    mine = _block(RANK, 0, sizes[RANK], 10700, amp=7)
    out2 = reducescatter(allgather(mine, sizes, GROUP), sizes, GROUP)
    torch.cuda.synchronize()
    _assert_bitwise(out2, (mine.float() * WORLD).to(torch.bfloat16), "rs(ag(x)), ragged")
    _barrier()


def check_cuda_graph_at_every_engine_batch_size() -> None:
    """One reduce-scatter per graph, captured at all 19 engine batch sizes, then
    replayed interleaved in four orders, checked and in a burst."""
    site_rows = (1,)
    inputs, graphs, outputs = _capture_per_batch_size(site_rows, 11000, round_trip=False)
    assert len(graphs) == len(GRAPH_BATCH_SIZES) == 19
    seed = _replay_two_ways(
        inputs, graphs, outputs, _replay_orders(), site_rows, 110000, round_trip=False
    )
    # Discrimination for a gate of exactly 0: what a replay that did not happen
    # leaves behind is the previous payload's result. The two consecutive
    # references have to be far apart for that to be caught, which this checks.
    prev = _ref([16] * WORLD, RANK, seed - 1000)
    last = _ref([16] * WORLD, RANK, seed)
    margin = (last.float() - prev.float()).abs().max().item()
    assert margin > 4.0, f"stale-replay margin {margin}"
    del graphs, outputs, inputs
    _barrier()


def check_cuda_graph_round_trip_in_every_moe_layer_of_every_batch_size_graph() -> None:
    """The decode graph this checkpoint captures under attention DP, at all 19 batch
    sizes: 48 round trips (gather, window, scatter) per graph, 912 captured
    reduce-scatters and as many all-gathers on one communicator, one memory pool.
    The sites are independent -- each reads its own buffer and every one gets a
    distinct payload -- so a site that returned a neighbour's rows fails the bitwise
    gate. Two replay orders: the axis this adds is the site count, not the order."""
    site_rows = (1,) * MOE_LAYERS
    inputs, graphs, outputs = _capture_per_batch_size(site_rows, 21000, round_trip=True)
    _replay_two_ways(
        inputs, graphs, outputs, _replay_orders()[2:], site_rows, 310000, round_trip=True
    )
    del graphs, outputs, inputs
    _barrier()


def check_cuda_graph_mtp3_decode_step_in_every_batch_size_graph() -> None:
    """dep4's MTP3 decode graph at all 19 batch sizes: per batch b, 49 round trips
    over 4b rows per rank (the trunk's 48 layers over the verify rows, then the draft
    layer's first step) and 2 over b rows (its later steps) -- up to 512 rows per
    rank, the largest captured reduce-scatter this target issues."""
    inputs, graphs, outputs = _capture_per_batch_size(MTP3_SITE_ROWS, 31000, round_trip=True)
    _replay_two_ways(
        inputs, graphs, outputs, _replay_orders()[2:], MTP3_SITE_ROWS, 410000, round_trip=True
    )
    del graphs, outputs, inputs
    _barrier()


def check_cuda_graph_replays_survive_eager_calls_of_other_shapes() -> None:
    """Between two decode replays, a server runs work no graph holds: a prefill
    chunk eagerly at a row count outside the captured set (up to max_num_tokens per
    rank), and a ragged reduce-scatter on the same communicator the graphs baked in."""
    site_rows = (1,)
    inputs, graphs, outputs = _capture_per_batch_size(site_rows, 41000, round_trip=False)
    seed = 510000
    for i, b in enumerate(reversed(GRAPH_BATCH_SIZES)):
        wide = (2048, 8192, 1500)[i % 3]
        wide_sizes = [wide] * WORLD
        _assert_bitwise(
            reducescatter(_input(RANK, wide_sizes, 610000 + i), None, GROUP),
            _ref(wide_sizes, RANK, 610000 + i),
            f"eager uniform {wide} rows",
        )
        ragged = [(11 * i + 3 * r) % 97 for r in range(WORLD)]
        _assert_bitwise(
            reducescatter(_input(RANK, ragged, 710000 + i), ragged, GROUP),
            _ref(ragged, RANK, 710000 + i),
            f"eager ragged {ragged}",
        )
        seed += 1000
        _fresh_payload(inputs, b, site_rows, seed, round_trip=False)
        graphs[b].replay()
        torch.cuda.synchronize()
        _verify_replay(inputs, outputs, b, site_rows, seed, False, f"after-eager@{i} b={b}")
        _barrier()
    del graphs, outputs, inputs
    _barrier()


def check_cuda_graph_holds_the_sizes_vector_it_captured() -> None:
    """`sizes` is a host argument: a replay re-runs the split it was captured with,
    which is why a captured step needs padded (uniform) row counts."""
    first = [1 + 4 * r for r in range(WORLD)]
    second = [7 * (WORLD - r) for r in range(WORLD)]
    seed = 81000
    buffers, graphs, captured = {}, {}, {}
    pool = None
    for tag, sizes in (("first", first), ("second", second)):
        x = _input(RANK, sizes, seed)
        _warm_up_off_capture_stream(lambda x=x, sizes=sizes: reducescatter(x, sizes, GROUP))
        graph = torch.cuda.CUDAGraph()
        ctx = torch.cuda.graph(graph) if pool is None else torch.cuda.graph(graph, pool=pool)
        with ctx:
            captured[tag] = reducescatter(x, sizes, GROUP)
        pool = graph.pool() if pool is None else pool
        buffers[tag], graphs[tag] = x, graph
        assert captured[tag].shape == (sizes[RANK], HIDDEN), captured[tag].shape
        _barrier()

    for round_seed in (82000, 83000):
        for tag, sizes in (("second", second), ("first", first)):
            buffers[tag].copy_(_input(RANK, sizes, round_seed))
            graphs[tag].replay()
            torch.cuda.synchronize()
            _assert_bitwise(
                captured[tag],
                _ref(sizes, RANK, round_seed),
                f"ragged replay {tag} sizes={sizes}",
            )
            _barrier()
    del graphs, captured, buffers
    _barrier()


def check_the_gate_discriminates_a_wrong_reduce_scatter() -> None:
    """The bitwise gate rejects every plausible wrong result, by a wide margin:
    the slice of a neighbouring group position, one rank's contribution missing, no
    reduction at all, and the right data cut at the wrong offsets under a rotated
    ragged split. Every one has the right shape, so a shape check alone would pass
    all four."""
    rows, seed = 8, 91000
    sizes = [rows] * WORLD
    uniform = reducescatter(_input(RANK, sizes, seed), None, GROUP)
    _assert_bitwise(uniform, _ref(sizes, RANK, seed), "uniform baseline")
    ragged = [1 + 4 * r for r in range(WORLD)]
    rag_out = reducescatter(_input(RANK, ragged, seed), ragged, GROUP)
    _assert_bitwise(rag_out, _ref(ragged, RANK, seed), "ragged baseline")

    mine = [_block(r, RANK, rows, seed) for r in range(WORLD)]
    missing_one = mine[0].float()
    for c in mine[1:-1]:
        missing_one = missing_one + c.float()
    wrong = [
        ("neighbouring slice", uniform, _ref(sizes, (RANK + 1) % WORLD, seed)),
        ("one rank never contributed", uniform, missing_one.to(torch.bfloat16)),
        ("not reduced at all", uniform, mine[RANK]),
        (
            "neighbouring slice, ragged split",
            rag_out,
            _ref(ragged, (RANK + 1) % WORLD, seed),
        ),
    ]
    for name, got, variant in wrong:
        n = min(variant.shape[0], got.shape[0])
        assert n > 0, name
        diff = (variant[:n].float() - got[:n].float()).abs()
        fraction = (diff != 0).float().mean().item()
        assert fraction > 0.9, f"{name}: only {fraction:.4f} of elements differ"
        assert diff.max().item() > 1.0, f"{name}: max difference {diff.max().item()}"
    _barrier()


def check_calls_pair_by_position_and_a_swapped_pair_realigns() -> None:
    """Calls pair by their **position** on the communicator, not by intent.

    One rank issuing two same-shaped calls in the opposite order to everybody else
    is not detected: both calls return, at the right shape, finite, with no
    diagnostic. What each rank gets back is the sum of whatever payloads met at that
    position -- asserted bitwise against a locally computed mix. Both `sizes` forms
    are driven (the even form is one `ncclReduceScatter`, the ragged form a grouped
    `ncclReduce` per rank). A swapped *pair* leaves the positions aligned once both
    calls are made, so the plain call at the end of each round is bitwise correct.
    """
    rows = 12
    ragged = [1 + 4 * r for r in range(WORLD)]
    for tag, sizes in (("even", None), ("ragged", ragged)):
        split = [rows] * WORLD if sizes is None else sizes
        base = 160000 if sizes is None else 165000
        first_seed, second_seed = base, base + 500
        first = _input(RANK, split, first_seed)
        second = _input(RANK, split, second_seed)
        _barrier()

        pos0, pos1 = _swapped_pair(first, second, sizes)
        torch.cuda.synchronize()
        for name, got in (("pos0", pos0), ("pos1", pos1)):
            assert got.shape == (split[RANK], HIDDEN), (tag, name, got.shape)
            assert bool(torch.isfinite(got.float()).all()), f"{tag} {name}: not finite"

        _assert_bitwise(
            pos0,
            _mispaired_ref(split, RANK, [second_seed] + [first_seed] * (WORLD - 1)),
            f"{tag}: position 0 is the mix of second(rank 0) and first(others)",
        )
        _assert_bitwise(
            pos1,
            _mispaired_ref(split, RANK, [first_seed] + [second_seed] * (WORLD - 1)),
            f"{tag}: position 1 is the mix of first(rank 0) and second(others)",
        )

        intended = _ref(split, RANK, second_seed if RANK == 0 else first_seed)
        wrong = _fraction_wrong(pos0, intended)
        assert wrong > 0.9, f"{tag}: only {wrong:.4f} of elements differ from intent"
        _barrier()

        _assert_bitwise(
            reducescatter(_input(RANK, split, base + 900), sizes, GROUP),
            _ref(split, RANK, base + 900),
            f"{tag}: plain call after a swapped pair",
        )
        _barrier()


def check_a_mispaired_result_is_deterministic_rather_than_noise() -> None:
    """A mispaired call is bitwise identical across repeats and equal to the
    ring-order chain of the addends that met: the mispairing changes which tensors
    are summed and nothing else, so re-running the step reproduces the wrong answer
    and a divergence cannot be found by looking for run-to-run instability."""
    rows = 12
    first_seed, second_seed = 162000, 162500
    first = _rand_input(RANK, rows, first_seed)
    second = _rand_input(RANK, rows, second_seed)
    _barrier()

    seen: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(4):
        pos0, pos1 = _swapped_pair(first, second, None)
        torch.cuda.synchronize()
        seen.append((pos0, pos1))
        _barrier()
    for k, (pos0, pos1) in enumerate(seen[1:], start=1):
        _assert_bitwise(pos0, seen[0][0], f"repeat {k} at position 0")
        _assert_bitwise(pos1, seen[0][1], f"repeat {k} at position 1")

    _assert_bitwise(
        seen[0][0],
        _ring_chain(rows, RANK, [second_seed] + [first_seed] * (WORLD - 1)),
        "position 0 is the ring-order sum of the mispaired addends",
    )
    _assert_bitwise(
        seen[0][1],
        _ring_chain(rows, RANK, [first_seed] + [second_seed] * (WORLD - 1)),
        "position 1 is the ring-order sum of the mispaired addends",
    )
    aligned = _ring_chain(rows, RANK, [second_seed if RANK == 0 else first_seed] * WORLD)
    wrong = _fraction_wrong(seen[0][0], aligned)
    assert wrong > 0.9, f"only {wrong:.4f} of elements differ from the aligned sum"
    _barrier()

    _assert_bitwise(
        reducescatter(_input(RANK, [rows] * WORLD, 163000), None, GROUP),
        _ref([rows] * WORLD, RANK, 163000),
        "plain call after four swapped pairs",
    )
    _barrier()


def check_an_extra_call_on_one_rank_misaligns_until_the_counts_match() -> None:
    """Rank 0 issues one call the others never issue, then all ranks issue four
    calls they intend to agree on: every one of the five positions is mispaired by
    exactly one place, and stays so until the other ranks' catch-up call equalizes
    the counts. There is no resynchronization point inside a stream of collectives."""
    rows, calls = 10, 4
    sizes = [rows] * WORLD
    shared = [170000 + 100 * k for k in range(calls)]
    extra_seed, catchup_seed = 179000, 179500
    _barrier()

    outs: List[torch.Tensor] = []
    if RANK == 0:
        outs.append(reducescatter(_input(RANK, sizes, extra_seed), None, GROUP))
        for seed in shared:
            outs.append(reducescatter(_input(RANK, sizes, seed), None, GROUP))
    else:
        for seed in shared:
            outs.append(reducescatter(_input(RANK, sizes, seed), None, GROUP))
        outs.append(reducescatter(_input(RANK, sizes, catchup_seed), None, GROUP))
    torch.cuda.synchronize()
    assert len(outs) == calls + 1

    rank0_order = [extra_seed] + shared
    others_order = list(shared) + [catchup_seed]
    for pos in range(calls + 1):
        mix = [rank0_order[pos]] + [others_order[pos]] * (WORLD - 1)
        _assert_bitwise(
            outs[pos], _mispaired_ref(sizes, RANK, mix), f"off-by-one at position {pos}"
        )
        intended = _ref(sizes, RANK, rank0_order[pos] if RANK == 0 else others_order[pos])
        wrong = _fraction_wrong(outs[pos], intended)
        assert wrong > 0.9, f"position {pos}: only {wrong:.4f} of elements differ"
    _barrier()

    _assert_bitwise(
        reducescatter(_input(RANK, sizes, 178000), None, GROUP),
        _ref(sizes, RANK, 178000),
        "plain call after the call counts were equalized",
    )
    _barrier()


def check_the_stream_the_call_lands_on_is_not_part_of_the_match() -> None:
    """The op runs on whatever stream is current, and ranks need not agree: what
    pairs the calls is their order on the communicator. Driven on the exact value
    set and on inexact payloads against the ring-order chain."""
    side = torch.cuda.Stream()
    rows = 24
    sizes = [rows] * WORLD
    cases = (
        ("every rank on a side stream", lambda i: True),
        ("only rank 0 on a side stream", lambda i: RANK == 0),
        ("ranks alternating opposite", lambda i: (i + RANK) % 2 == 0),
    )
    for c, (name, use_side) in enumerate(cases):
        for i in range(4):
            seed = 150000 + 1000 * c + 13 * i
            x = _input(RANK, sizes, seed)
            out = (
                _reduce_scatter_on_a_side_stream(x, side)
                if use_side(i)
                else reducescatter(x, None, GROUP)
            )
            _assert_bitwise(out, _ref(sizes, RANK, seed), f"{name} i={i}")
        _barrier()

    for c, (name, use_side) in enumerate(cases):
        for i in range(4):
            seed = 155000 + 1000 * c + 13 * i
            x = _rand_input(RANK, rows, seed)
            out = (
                _reduce_scatter_on_a_side_stream(x, side)
                if use_side(i)
                else reducescatter(x, None, GROUP)
            )
            torch.cuda.synchronize()
            _assert_bitwise(out, _ring_chain(rows, RANK, [seed] * WORLD), f"{name} ring i={i}")
        _barrier()


def _assert_refused(args: Tuple[Any, ...], what: str, match: Optional[str] = None) -> None:
    """The guard refuses `args` before the op is called."""
    try:
        with validating(entry.reducescatter):
            reducescatter(*args)
    except AssertionError as exc:
        if match is not None:
            assert match in str(exc), str(exc)
    else:
        raise AssertionError(f"the guard accepted {what}")


def check_the_guard_refuses_what_the_op_gets_silently_wrong() -> None:
    """Every `is_valid` refusal, each before the op is reached: a float8 input
    (summed as raw bytes), a non-contiguous view (reduced over the wrong elements),
    a 0-d tensor (segfaults in `ReducescatterOp::run_list`), a sizes list of the
    wrong length (wedges, or reads past the input), and a split that does not cover
    the input (drops rows, or reads past it)."""
    rows, seed = 4, 92000
    fp8 = torch.full((rows * WORLD, 8), float(RANK + 1), device="cuda").to(torch.float8_e4m3fn)
    _assert_refused((fp8, None, GROUP), "a float8_e4m3fn input", "not summable")
    padded = torch.zeros(rows * WORLD, HIDDEN * 2, dtype=torch.bfloat16, device="cuda")
    view = padded[:, :HIDDEN]
    assert not view.is_contiguous()
    _assert_refused((view, None, GROUP), "a non-contiguous input", "contiguous")
    _assert_refused((torch.tensor(1.0, device="cuda"), None, GROUP), "a 0-d input")
    x = _input(RANK, [rows] * WORLD, seed)
    for bad in ([rows] * (WORLD - 1), [rows] * (WORLD + 1)):
        _assert_refused((x, bad, GROUP), f"sizes of length {len(bad)}", "len(sizes)")
    odd = _input(RANK, [1] * (WORLD * 2 + 1), seed)
    _assert_refused((odd, None, GROUP), "a row count the group does not divide", "divisible")
    for short in ([2] * WORLD, [6] * WORLD):
        _assert_refused((x, short, GROUP), f"an uncovered split {short}", "sum(sizes)")
    _barrier()


def check_the_guard_refuses_every_dtype_it_cannot_sum() -> None:
    """`bool` maps to ncclInt8 like float8_e4m3fn (its sum reads as OR); `float64`
    is not in `getDtypeMap` and throws inside the open NCCL group. The guard refuses
    both before the group opens, so the communicator stays usable -- checked."""
    rows, seed = 8, 97000
    sizes = [rows] * WORLD
    for dtype in (torch.bool, torch.float64):
        x = torch.ones(4 * WORLD, 8, dtype=torch.float32, device="cuda").to(dtype)
        _assert_refused((x, None, GROUP), str(dtype), "not summable")
        _barrier()
    _assert_bitwise(
        reducescatter(_input(RANK, sizes, seed), None, GROUP),
        _ref(sizes, RANK, seed),
        "the group after the guard refused an unsummable dtype",
    )
    _barrier()


CHECKS = (
    # Stays first: it issues the group's first call, which builds the communicator
    # every later check uses.
    check_cuda_graph_captures_after_an_eager_first_call,
    check_certified_cells,
    check_uniform_reduce_scatter,
    check_ragged_reduce_scatter,
    check_output_is_fresh_and_input_is_untouched,
    check_trailing_dims_are_preserved,
    check_dtypes_reduce_arithmetically,
    check_group_selects_a_rank_subset,
    check_group_order_does_not_change_the_output,
    check_reduction_is_deterministic_and_accumulates_in_the_input_dtype,
    check_cell_sized_reductions_on_window_like_payloads,
    check_uniform_form_and_an_explicit_even_split_agree_bitwise,
    check_round_trip_with_a_gather_returns_each_rank_its_own_rows,
    check_cuda_graph_at_every_engine_batch_size,
    check_cuda_graph_round_trip_in_every_moe_layer_of_every_batch_size_graph,
    check_cuda_graph_mtp3_decode_step_in_every_batch_size_graph,
    check_cuda_graph_replays_survive_eager_calls_of_other_shapes,
    check_cuda_graph_holds_the_sizes_vector_it_captured,
    check_the_gate_discriminates_a_wrong_reduce_scatter,
    # The call-order block. Each of these deliberately disagrees about call order
    # and each restores alignment before it returns -- the plain call every one of
    # them ends on is what proves it.
    check_calls_pair_by_position_and_a_swapped_pair_realigns,
    check_a_mispaired_result_is_deterministic_rather_than_noise,
    check_an_extra_call_on_one_rank_misaligns_until_the_counts_match,
    check_the_stream_the_call_lands_on_is_not_part_of_the_match,
    check_the_guard_refuses_what_the_op_gets_silently_wrong,
    check_the_guard_refuses_every_dtype_it_cannot_sum,
)


def _run_one_rank() -> int:
    """Body of one MPI rank: run every check in order; a failing rank aborts the
    job, since a rank that leaves a collective early wedges every other rank in it."""
    global RANK, WORLD, GROUP, COMM, reducescatter, allgather, validating, entry
    os.environ.pop("TLLM_DISABLE_MPI", None)
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK, WORLD = COMM.Get_rank(), COMM.Get_size()
    assert WORLD >= 2, f"a collective needs at least 2 ranks, got {WORLD}"
    torch.cuda.set_device(RANK % torch.cuda.device_count())

    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.comm import (  # noqa: E501
        reducescatter__dp_scatter as entry_module,
    )

    entry = entry_module
    # `process_group` left at None: the MPI-session symbol, the target's call under mpirun.
    reducescatter = entry.reducescatter

    def _allgather(input, sizes, group):
        return torch.ops.trtllm.allgather(input, sizes, group)

    allgather = _allgather

    # Run as an MPI rank, not collected by pytest, so the repo's `__extra_import_path__`
    # mechanism is not in play: reach `_validating` under the modeling_v2 test root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from _validating import validating as validating_context

    validating = validating_context
    GROUP = list(range(WORLD))

    for check in CHECKS:
        start = time.monotonic()
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            COMM.Abort(13)
        TIMINGS[check.__name__] = time.monotonic() - start
        if RANK == 0:
            print(f"[rank 0] passed {check.__name__} in {TIMINGS[check.__name__]:.1f}s", flush=True)
    _barrier()
    print(f"[rank {RANK}] world {WORLD}: {len(CHECKS)} checks passed", flush=True)
    sys.stdout.flush()
    # Finalize MPI, then leave without static teardown: destroying communicators that
    # captured collectives into CUDA graphs is not what this job certifies.
    MPI.Finalize()
    os._exit(0)


def _spawn(world_size: int) -> None:
    """Re-execute this file under `mpirun -n world_size`, one rank per visible device,
    under a deadline enforced by killing the whole process group.

    `--oversubscribe` because the launching shell may itself sit in a one-task Slurm
    allocation, and the Slurm step variables are dropped for the same reason: with
    them every rank reads the step's single task and runs as rank 0 of 1.
    """
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns, e.g. 0,1,2,3"
    assert 2 <= world_size <= len(visible), (
        f"world size {world_size} needs 2..{len(visible)} visible devices (one rank per device)"
    )
    env = {k: v for k, v in os.environ.items() if k not in _SLURM_STEP_VARS}
    env.pop("TLLM_DISABLE_MPI", None)
    command = [
        "mpirun",
        "--allow-run-as-root",
        "--oversubscribe",
        "-n",
        str(world_size),
        sys.executable,
        str(Path(__file__).resolve()),
        _WORKER_FLAG,
    ]
    print(f"[launcher] {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        code = process.wait(timeout=DEADLINE_S)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise AssertionError(
            f"the {world_size}-rank run did not finish in {DEADLINE_S}s (wedged)"
        ) from None
    assert code == 0, f"the {world_size}-rank run exited {code}"


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=None)
    args, _ = parser.parse_known_args()
    visible_count = len(
        [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    )
    _spawn(args.world_size or visible_count)
    print("OK")
