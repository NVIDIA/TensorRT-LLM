# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the reducescatter catalog entry.

A collective cannot be exercised in one process, so this script is its own
launcher: run it plainly and it starts one worker process per device, with one
rank per device named in CUDA_VISIBLE_DEVICES, under a deadline the parent
enforces by killing the workers. A wedged collective hangs rather than
raising — and this op wedges rather than raising when the ranks disagree about
the split — so the deadline is what keeps a broken kernel from taking the
calling run down with it.

The entry's `__call__` routes to one of two symbols, and a run certifies one
of them, chosen by `--transport`:

* `mpi` (`torch.ops.trtllm.reducescatter`, `process_group=None`): the symbol
  a serving world launched under MPI calls. The ranks are started by
  `mpirun`, so the op's communicator is built from the MPI session exactly as
  in serving.
* `pg` (`torch.ops.trtllm.reducescatter_pg`, a boxed c10d group): the symbol
  the MPI-disabled backends call. The ranks are spawned directly and
  rendezvous through torch.distributed alone.

The world size is the number of visible devices, and a run drives the cells
whose `world` matches it (a cell without one is a world-4 cell).

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _reducescatter_op_matrix.py --transport=mpi
    CUDA_VISIBLE_DEVICES=0,1 python _reducescatter_op_matrix.py --transport=pg

A world that spans GB300 trays is started instead as one srun task per rank,
`python _reducescatter_op_matrix.py --srun`, with `MASTER_ADDR` and `MASTER_PORT`
naming the first node (`_rank_job.adopt_srun_env`); that path runs the `pg`
transport.

Not a pytest module, despite the `check_*` bodies. They are one fixed
sequence inside a single multi-rank job rather than independent cases: each
reads module-global rank state that only `_run_one_rank` binds, and several
assert on communicator state the previous one left behind. Collected as tests
they would run at world size 1 against unbound globals — so neither this
file's name nor its function names match pytest's collection patterns, which
is what keeps it uncollectable however pytest is pointed at this tree.

The collected entry point is `test_modeling_v2_reducescatter_op_matrix.py`
beside this file: it starts this job once per world size and transport and
turns its exit code into an assertion. Both halves are started by file path:
the launcher must not import `tensorrt_llm` (that would give it CUDA or MPI
state of its own), and the ranks reach the entry by absolute import, so
neither needs a package.
"""

import functools
import os
import random
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

assert torch.cuda.is_available(), "reducescatter requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
_TRANSPORT_ENV = "MV2_REDUCESCATTER_MATRIX_TRANSPORT"
TRANSPORTS = ("mpi", "pg")
DEADLINE_S = 1800

HIDDEN = 2560  # DeepSeek-V3-Lite hidden size, the caller that motivates this

# The row counts one engine's decode graphs are captured at. trtllm's graph
# runner instantiates one graph per configured batch size, and under attention
# data parallelism a decode call's per-rank row count *is* that batch size:
# with `cuda_graph_config.max_batch_size = 256` and the default list that is 35
# graphs at [1..32, 64, 128, 256], all alive at once in one memory pool and
# replayed interleaved as the served batch size moves. This op's input carries
# every rank's rows, so its captured input is WORLD times that.
GRAPH_BATCH_SIZES: Tuple[int, ...] = tuple(range(1, 33)) + (64, 128, 256)

# Reduce-scatters inside one decode graph for this checkpoint: 30 layers with
# `first_k_dense_replace = 1`, so 29 expert-parallel MoE calls, each followed
# by one reduce-scatter of the four expert windows' partial outputs.
SITES_PER_DECODE_GRAPH = 29

# bf16 unit roundoff: the format carries 8 significand bits, so eps = 2^-8 and
# a round-to-nearest step is off by at most 2^-9 of the running value. Used by
# the accumulation-order test's error budget.
BF16_U = 2.0**-8

# Step-5-Preview's decode graphs: the run's `cuda_graph_config` batch sizes,
# which under attention DP are the per-rank row counts, and one reduce-scatter
# per MoE layer (`moe_layer_list` is layers 3..90 of 92; 0, 1, 2 and 91 are
# dense), each returning the routed experts' output to the rows a rank owns.
STEP5_HIDDEN = 4096
STEP5_GRAPH_BATCH_SIZES: Tuple[int, ...] = (1, 2, 4) + tuple(range(8, 129, 8))
STEP5_MOE_SITES_PER_DECODE_GRAPH = 88

# Unit roundoff per dtype the Step-5 site reduces in, for the sequential
# summation bound of check_step5_numerics.
UNIT_ROUNDOFF = {torch.bfloat16: 2.0**-8, torch.float32: 2.0**-24}

# Bound inside the rank body, never in the launcher: importing the entry
# pulls in tensorrt_llm, which calls MPI_Init at import, and the launcher must
# carry no MPI or CUDA state.
reducescatter: Any = None
entry: Any = None
validating: Any = None
TRANSPORT = "pg"
PG_BOXED: Any = None
SUBGROUPS: Dict[Tuple[int, ...], Any] = {}
RANK = 0
WORLD = 1
GROUP: List[int] = [0]


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
    without materializing anybody's whole input — cuRAND is deterministic for a
    given (seed, shape, dtype). That is what lets the reference below be
    arithmetic rather than a second collective.

    The float value set is multiples of 1/8 with |x| <= amp/8. bf16, fp16 and
    fp32 all represent k/8 exactly for |k| <= 255, and a sum of WORLD of them
    reaches |k| <= 4 * 31 = 124, so **every partial sum is exact in every
    summation order**. That is what lets the assertions be bitwise despite the
    op summing in the input dtype (see check_reduction_is_deterministic_and_
    accumulates_in_the_input_dtype for what happens when they are not).
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
    # Delegates to the entry: what the sum is supposed to be is the entry's
    # claim, and computing it here as well would be a second place to fix.
    return entry.reducescatter.reference(
        [_block(r, pos, sizes[pos], seed, dtype, trailing, amp) for r in ranks]
    )


def _assert_bitwise(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    """Gate of exactly zero, on payloads whose every partial sum is exact.

    Tightened from the default dtype-aware tolerances rather than loosened: the
    reduction is a sum in the input dtype, but `_block`'s value set makes every
    intermediate representable, so any difference at all is a wrong reduction
    or a wrong slice, not rounding.
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

    `seeds[r]` is the payload rank `r` happened to be holding when this
    position on the communicator came round. With `_block`'s value set every
    partial sum is exact, so the prediction is bitwise regardless of the order
    the addends are accumulated in.
    """
    acc = torch.zeros((sizes[pos], HIDDEN), dtype=torch.float32, device="cuda")
    for rank, seed in enumerate(seeds):
        acc += _block(rank, pos, sizes[pos], seed).float()
    return acc.to(torch.bfloat16)


def _rand(
    rank: int,
    dest: int,
    rows: int,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
    hidden: int = HIDDEN,
) -> torch.Tensor:
    """Payload with no exactness property, for the accumulation-order tests."""
    gen = torch.Generator(device="cuda").manual_seed((seed * 977 + rank) * 131 + dest + 1)
    return torch.randn((rows, hidden), generator=gen, device="cuda", dtype=torch.float32).to(dtype)


def _rand_input(
    rank: int,
    rows: int,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
    hidden: int = HIDDEN,
) -> torch.Tensor:
    """One rank's whole contribution, drawn from the inexact value set."""
    return torch.cat([_rand(rank, d, rows, seed, dtype, hidden) for d in range(WORLD)], dim=0)


def _ring_chain(rows: int, pos: int, seeds: Sequence[int]) -> torch.Tensor:
    """Ring-order sequential sum for group position `pos`, rounded every step.

    The order this op reduces in (certified by
    check_reduction_is_deterministic_and_accumulates_in_the_input_dtype):
    `x_{pos+1} + x_{pos+2} + ... + x_{pos+G-1} + x_pos`, indices mod WORLD,
    where `x_r` is rank `r`'s block for `pos` drawn from `seeds[r]`. Per-rank
    seeds, so a mispaired call's exact bits can be predicted too — on payloads
    where a different accumulation order would give different bits.
    """
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

    Returns the results of communicator **positions** 0 and 1 — which is not
    the same as the results of `first` and `second`, and that is the point.
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


def _warm_up_off_capture_stream(body, reps: int = 2) -> None:
    """Run `body` on a side stream, then rejoin, so a capture can follow.

    Two things have to be done before a capture and cannot be done inside one:
    the group's NCCL communicator has to exist (see
    check_cuda_graph_captures_a_cold_first_call), and torch wants the work
    warmed on a non-default stream.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(reps):
            body()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    dist.barrier()


def _capture_per_batch_size(
    sites: int = 1,
    seed: int = 11000,
    batch_sizes: Sequence[int] = GRAPH_BATCH_SIZES,
    hidden: int = HIDDEN,
) -> Tuple[Dict[int, List[torch.Tensor]], Dict[int, Any], Dict[int, List[torch.Tensor]]]:
    """One graph per `batch_sizes` entry, one shared pool, all left alive.

    That is the state a graph runner ends up in: every capture after the first
    goes into the first one's pool, and none of them is ever destroyed.
    `sites` reduce-scatters go into each graph, each reading its own persistent
    input buffer, so the payload of every site can be moved independently.
    Returns (inputs, graphs, outputs), all keyed by the per-rank row count.
    """
    inputs: Dict[int, List[torch.Tensor]] = {}
    graphs: Dict[int, Any] = {}
    outputs: Dict[int, List[torch.Tensor]] = {}
    pool = None
    for rows in batch_sizes:
        xs = [_input(RANK, [rows] * WORLD, seed + 13 * s, trailing=(hidden,)) for s in range(sites)]

        def body(xs: List[torch.Tensor] = xs) -> List[torch.Tensor]:
            return [reducescatter(x, None, GROUP) for x in xs]

        # Persistent output buffers, allocated OUTSIDE the shared pool: the op's
        # own output lands in the pool, where a later graph's capture reuses the
        # same address, so under a burst of replays with no sync between them the
        # pooled outputs all end up holding the last writer's result. Copying each
        # into a dedicated buffer inside the capture fixes that -- the copy is
        # ordered after this graph's reduce-scatter on the one stream, so it reads
        # the right value before the next graph overwrites the pool. This is what
        # lets the burst pass test the collective's ordering rather than die on
        # pool aliasing.
        persistent = [
            torch.empty(rows, hidden, dtype=torch.bfloat16, device="cuda") for _ in range(sites)
        ]

        def capture_body() -> List[torch.Tensor]:
            produced = body()
            for dst, src in zip(persistent, produced):
                dst.copy_(src)
            return persistent

        _warm_up_off_capture_stream(body)
        graph = torch.cuda.CUDAGraph()
        ctx = torch.cuda.graph(graph) if pool is None else torch.cuda.graph(graph, pool=pool)
        with ctx:
            outputs[rows] = capture_body()
        # Without this a capture that returned nothing would make every
        # `_verify_replay` below an empty loop, i.e. a vacuous pass.
        assert len(outputs[rows]) == sites, (
            f"rows={rows}: captured {len(outputs[rows])} outputs, expected {sites}"
        )
        if pool is None:
            pool = graph.pool()
        inputs[rows], graphs[rows] = xs, graph
        dist.barrier()
    return inputs, graphs, outputs


def _replay_orders(
    batch_sizes: Sequence[int] = GRAPH_BATCH_SIZES,
) -> List[Tuple[str, List[int]]]:
    """Four ways a served batch size moves through the captured set."""
    asc = list(batch_sizes)
    shuffled = list(asc)
    random.Random(1234).shuffle(shuffled)
    # The largest jump still available at every step: 256, 1, 128, 2, 64, 3...
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
    inputs: Dict[int, List[torch.Tensor]], rows: int, seed: int, hidden: int = HIDDEN
) -> None:
    """Overwrite every site's persistent input buffer for this row count."""
    for site, x in enumerate(inputs[rows]):
        x.copy_(_input(RANK, [rows] * WORLD, seed + 13 * site, trailing=(hidden,)))


def _verify_replay(
    outputs: Dict[int, List[torch.Tensor]],
    rows: int,
    seed: int,
    where: str,
    hidden: int = HIDDEN,
) -> None:
    """Every site's captured output tensor holds this payload's result, bitwise."""
    for site, out in enumerate(outputs[rows]):
        _assert_bitwise(
            out,
            _ref([rows] * WORLD, RANK, seed + 13 * site, trailing=(hidden,)),
            f"{where} site={site}",
        )


def check_cuda_graph_captures_a_cold_first_call() -> None:
    """A group's first-ever call captures straight into a graph, no warm-up.

    The communicator is built by `init_process_group`, before any op runs, so a
    capture never has to build one inside itself -- the one thing a capture
    cannot do. (The MPI-session path built the communicator lazily on first
    use, so its first call could not be captured; this path has no such trap,
    which is the behaviour this certifies.) Replaying after a fresh input must
    reflect that input, bitwise.
    """
    rows = 16
    sizes = [rows] * WORLD
    seed = 100
    x = _input(RANK, sizes, seed)
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = reducescatter(x, None, GROUP)
    # Capture records the op, it does not run it; the first replay is what fills
    # `captured` from this input.
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(captured, _ref(sizes, RANK, seed), "cold first capture")

    # A second replay after a refilled input reflects that input, not the first.
    x.copy_(_input(RANK, sizes, seed + 500))
    torch.cuda.synchronize()
    dist.barrier()
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(captured, _ref(sizes, RANK, seed + 500), "replay after refill")
    del graph
    dist.barrier()


def check_mpi_symbol_first_call_is_eager_then_captures() -> None:
    """The MPI symbol's first call for a group is eager; a capture follows it.

    The MPI symbol builds its NCCL communicator for a rank set on that set's
    first call (`getComm`, cached per rank set for the life of the process),
    so that first call cannot sit inside a capture. A serving engine's
    warm-up forward is that call. After it, a capture with no further warm-up
    replays bitwise, and a refilled input is reflected by the next replay.
    """
    rows, seed = 16, 100
    sizes = [rows] * WORLD
    x = _input(RANK, sizes, seed)
    _assert_bitwise(reducescatter(x, None, GROUP), _ref(sizes, RANK, seed), "first eager")
    torch.cuda.synchronize()
    dist.barrier()

    x.copy_(_input(RANK, sizes, seed + 200))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = reducescatter(x, None, GROUP)
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(captured, _ref(sizes, RANK, seed + 200), "capture after eager")

    x.copy_(_input(RANK, sizes, seed + 500))
    torch.cuda.synchronize()
    dist.barrier()
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(captured, _ref(sizes, RANK, seed + 500), "replay after refill")
    del graph
    dist.barrier()


def cell_world(cell: Any) -> int:
    """The world size a cell is certified at.

    The deepseek-r1 and gpt-oss cells carry no `world`; they run at 4.
    """
    return cell.spec.get("world", 4)


def cell_split(cell: Any) -> List[int]:
    """Each rank's slice height, in rank order, for one cell."""
    spec = cell.spec
    if spec["sizes"] is None:
        return [spec["rows_per_rank"]] * cell_world(cell)
    return list(spec["sizes"])


def check_certified_cells() -> None:
    """Drive the entry's own cell list at this world size, inside `validating`.

    The rest of this file says things no cell can -- CUDA graph behaviour, call
    order, which stream the call lands on. This one says the plain thing: for
    every configuration the entry claims, the slice is the sum, bitwise, and the
    guard admits the input a shipped target passes. A uniform cell passes
    `sizes=None` and every rank keeps `rows_per_rank` rows; a ragged cell passes
    `sizes` and rank r keeps `sizes[r]` rows.

    The cells run twice in a row. The second pass reuses the communicator and
    the caching allocator's blocks the first pass left, so a call that read a
    stale output block or a previous cell's rows fails it.
    """
    cells = [(i, c) for i, c in enumerate(entry.reducescatter.CELLS) if cell_world(c) == WORLD]
    # A world size with no cell would make this check a vacuous pass.
    assert cells, f"no cell of the entry is at world {WORLD}"
    for repeat in range(2):
        for i, cell in cells:
            spec = cell.spec
            trailing, dtype = spec["trailing"], spec["dtype"]
            split = cell_split(cell)
            assert len(split) == WORLD, (cell.why, split)
            seed = 9000 + 13 * i + 100000 * repeat
            x = _input(RANK, split, seed, dtype, trailing)
            with validating(entry.reducescatter):
                out = reducescatter(x, spec["sizes"], GROUP)
            assert out.shape == (split[RANK], *trailing), (cell.why, out.shape)
            assert out.dtype is dtype and out.is_contiguous(), (cell.why, out.dtype)
            _assert_bitwise(
                out, _ref(split, RANK, seed, dtype, trailing), f"{cell.why} pass={repeat}"
            )
            dist.barrier()
    if RANK == 0:
        print(
            f"[rank 0] transport={TRANSPORT} world={WORLD}: "
            f"{len(cells)} cells bitwise, two passes: " + "; ".join(c.why for _, c in cells),
            flush=True,
        )


def check_step5_numerics() -> None:
    """What the sum is at Step-5's widths and dtypes, on inexact payloads.

    The cells gate on a value set where every partial sum is exact, so they
    cannot see the accumulation. Here the payload is standard normal, cast to
    the cell's dtype, at hidden 4096 and the Step-5 row counts:

    1. It is **deterministic**: four identical calls return identical bits.
    2. It **is** a sequential summation in the input dtype, in ring order
       starting at the destination's successor, matched bitwise
       (`_ring_chain`). At world 2 that is one rounded add, the correctly
       rounded sum.
    3. Its distance from the exact (fp64) sum is inside the classical bound for
       a sequential sum of WORLD terms, `(WORLD-1) * u * sum_r |x_r|` with
       `u` the dtype's unit roundoff.

    A failure of (2) means NCCL picked a different reduction order for this
    payload; the bound (3) is what a caller may rely on in that case, so it is
    asserted separately rather than folded into (2).
    """
    for dtype in (torch.bfloat16, torch.float32):
        for rows in (1, 128, 8192):
            seed = 120000 + rows + (0 if dtype is torch.bfloat16 else 50000)
            hidden = STEP5_HIDDEN
            x = _rand_input(RANK, rows, seed, dtype, hidden)
            outs = [reducescatter(x, None, GROUP) for _ in range(4)]
            torch.cuda.synchronize()
            for k, o in enumerate(outs[1:], start=1):
                _assert_bitwise(o, outs[0], f"{dtype} repeat {k} at rows={rows}")

            mine = [_rand(r, RANK, rows, seed, dtype, hidden) for r in range(WORLD)]
            first = (RANK + 1) % WORLD
            chain = mine[first].clone()
            for k in range(2, WORLD + 1):
                chain = chain + mine[(RANK + k) % WORLD]
            _assert_bitwise(outs[0], chain, f"{dtype} ring-order chain at rows={rows}")

            exact = torch.zeros(rows, hidden, dtype=torch.float64, device="cuda")
            absum = torch.zeros(rows, hidden, dtype=torch.float64, device="cuda")
            for chunk in mine:
                exact += chunk.double()
                absum += chunk.double().abs()
            err = (outs[0].double() - exact).abs()
            budget = (WORLD - 1) * UNIT_ROUNDOFF[dtype] * absum
            over = (err > budget).sum().item()
            ratio = (err / budget.clamp_min(1e-300)).max().item()
            assert over == 0, (
                f"{dtype} rows={rows}: {over} elements outside the sequential-summation "
                f"bound, worst ratio {ratio:.4f}"
            )
            if RANK == 0:
                print(
                    f"[rank 0] numerics {dtype} rows={rows}: ring-order bitwise, "
                    f"worst error/bound {ratio:.4f}",
                    flush=True,
                )
            del x, outs, mine, chain, exact, absum, err, budget
            dist.barrier()


def check_uniform_reduce_scatter() -> None:
    """sizes=None: every rank sends WORLD * rows rows and keeps `rows` of them.

    Decode-like (1, 2, 8, 32) through prefill-like (2048) per-rank row counts,
    which is the whole span an attention-DP rank gets back from its
    expert-parallel MoE call.
    """
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
        dist.barrier()


def check_ragged_reduce_scatter() -> None:
    """sizes=[...]: the split is uneven, this rank keeps sizes[my position].

    Attention data parallelism produces exactly this — each rank owns its own
    requests — so the ragged form is the one a non-padded step takes, including
    the case where one rank has no rows at all.
    """
    for i, sizes in enumerate(_sizes_vectors()):
        seed = 3000 + 7 * i
        x = _input(RANK, sizes, seed)
        assert x.shape == (sum(sizes), HIDDEN), x.shape
        out = reducescatter(x, sizes, GROUP)
        assert out.shape == (sizes[RANK], HIDDEN), (sizes, out.shape)
        assert out.dtype is x.dtype and out.is_contiguous()
        _assert_bitwise(out, _ref(sizes, RANK, seed), f"ragged sizes={sizes}")
        dist.barrier()


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
    dist.barrier()


def check_trailing_dims_are_preserved() -> None:
    """Dim 0 is the scatter axis; every other dim is carried through untouched."""
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
        dist.barrier()


def check_dtypes_reduce_arithmetically() -> None:
    """The dtypes whose sum this op actually computes, in both forms.

    fp16 and fp32 alongside bf16, and the integer widths, which reduce as
    integer sums (kept small enough here not to overflow the narrow ones).
    float8 is a separate test: it is accepted and summed as raw bytes.
    """
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
        dist.barrier()

        xr = _input(RANK, ragged, seed, dtype)
        outr = reducescatter(xr, ragged, GROUP)
        assert outr.dtype is dtype and outr.shape == (ragged[RANK], HIDDEN)
        _assert_bitwise(outr, _ref(ragged, RANK, seed, dtype), f"ragged {dtype}")
        dist.barrier()


def check_group_selects_a_rank_subset() -> None:
    """`group` names the communicator's ranks; the slice index is the position in it.

    The second subset is the discriminating one: it excludes rank 0, so a rank
    that indexed the split by its global rank instead of by its position in the
    group would read a different slice (and rank WORLD-1 would read past the
    end). Ranks outside the subset must not call.
    """
    subsets = [[0, 1]]
    if WORLD >= 4:
        subsets.append([WORLD - 2, WORLD - 1])
    for i, subset in enumerate(subsets):
        rows, seed = 4, 7100 + 31 * i
        sizes = [rows] * len(subset)
        if RANK in subset:
            # new_group returns a sentinel (not a group) on non-member ranks.
            pg = SUBGROUPS[tuple(sorted(subset))].boxed()
            pos = subset.index(RANK)
            out = reducescatter(_input(RANK, sizes, seed), None, subset, process_group=pg)
            assert out.shape == (rows, HIDDEN), out.shape
            _assert_bitwise(out, _ref(sizes, pos, seed, ranks=subset), f"subset {subset} uniform")
            ragged = [2, 6] if len(subset) == 2 else [2] * len(subset)
            outr = reducescatter(_input(RANK, ragged, seed), ragged, subset, process_group=pg)
            assert outr.shape == (ragged[pos], HIDDEN), outr.shape
            _assert_bitwise(outr, _ref(ragged, pos, seed, ranks=subset), f"subset {subset} ragged")
        dist.barrier()


def check_group_order_does_not_change_the_output() -> None:
    """The split is ordered by ascending rank, whatever order `group` lists."""
    rows, seed = 4, 7200
    sizes = [rows] * WORLD
    x = _input(RANK, sizes, seed)
    out = reducescatter(x, None, list(reversed(GROUP)))
    _assert_bitwise(out, _ref(sizes, RANK, seed), "reversed group list")
    dist.barrier()


def check_reduction_is_deterministic_and_accumulates_in_the_input_dtype() -> None:
    """What the sum is, exactly — the fact a target's accuracy gate rests on.

    On payloads with no exactness property (standard normal, cast to bf16):

    1. It is **deterministic**: eight identical eager calls return bitwise
       identical tensors, and so do eight replays of a captured call.
    2. It is **not** the correctly rounded exact sum. The op accumulates in the
       input dtype, so about a third of the elements differ from an fp32
       reduction rounded once at world 4. At world 2 the one rounded add is
       that sum, and every element matches it.
    3. It **is** a sequential summation in the input dtype, in ring order
       starting at the destination's successor — matched bitwise on every
       element here. That order is NCCL's algorithm choice for this topology;
       if this assertion ever fails, the reduction order changed and the
       contract's Numerics note has to be re-measured rather than this gate
       loosened.
    4. Its distance from the exact sum stays inside the classical bound for a
       sequential sum of WORLD terms, `(WORLD-1) * u * sum_r |x_r|` with
       `u = 2^-8` for bf16 — the bound holds elementwise with the largest
       observed ratio 0.89 (measured at 1, 8, 64, 256 and 2048 rows).
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

        # (2) not the correctly rounded exact sum -- except at world 2, where
        # the one rounded add of two bf16 values is that sum.
        differs = (outs[0] != exact.to(torch.bfloat16)).float().mean().item()
        if WORLD > 2:
            assert differs > 0.05, (
                f"rows={rows}: only {differs:.4f} of elements differ from an fp32 "
                "reduction — the op may have started accumulating in fp32"
            )
        else:
            assert differs == 0, f"rows={rows}: {differs:.4f} differ from one rounded add"

        # (3) a sequential chain in ring order, starting at my successor
        chain = mine[(RANK + 1) % WORLD].clone()
        for k in range(2, WORLD + 1):
            chain = chain + mine[(RANK + k) % WORLD]
        _assert_bitwise(outs[0], chain, f"ring-order chain at rows={rows}")

        # (4) inside the classical sequential-summation error bound
        err = (outs[0].float() - exact).abs()
        budget = (WORLD - 1) * BF16_U * absum
        over = (err > budget).sum().item()
        assert over == 0, (
            f"rows={rows}: {over} elements outside the sequential-summation "
            f"bound, worst ratio {(err / budget.clamp_min(1e-30)).max().item():.4f}"
        )
        dist.barrier()

    # Determinism under capture, and the captured result equals the eager one.
    rows, seed = 16, 8500
    x = torch.cat([_rand(RANK, d, rows, seed) for d in range(WORLD)], dim=0)
    eager = reducescatter(x, None, GROUP)
    torch.cuda.synchronize()
    _warm_up_off_capture_stream(lambda: reducescatter(x, None, GROUP))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = reducescatter(x, None, GROUP)
    for k in range(8):
        graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise(captured, eager, f"replay {k} against the eager result")
    del graph
    dist.barrier()


def check_uniform_form_and_an_explicit_even_split_agree_bitwise() -> None:
    """`sizes=None` and an explicit even `sizes` vector return the same bits.

    Worth pinning because the two are not obliged to take the same path — and
    because an attention-DP target switches between them (padded, captured
    decode steps pass None; eager steps pass the counts), so a caller needs to
    know that the switch does not perturb the arithmetic. Driven on payloads
    with no exactness property, where a different reduction order would show.
    """
    for rows in (1, 8, 256, 2048):
        seed = 9000 + rows
        x = torch.cat([_rand(RANK, d, rows, seed) for d in range(WORLD)], dim=0)
        a = reducescatter(x, None, GROUP)
        b = reducescatter(x, [rows] * WORLD, GROUP)
        torch.cuda.synchronize()
        _assert_bitwise(b, a, f"explicit even split at rows={rows}")
        dist.barrier()


def check_round_trip_with_a_gather_returns_each_rank_its_own_rows() -> None:
    """The attention-DP MoE round trip, end to end, against a local reference.

    Gather every rank's tokens, let each rank apply its own expert window to
    the whole token set, reduce-scatter the four partial outputs. The claim
    under test is that the scatter undoes the gather: rank i gets back exactly
    the rows rank i contributed, holding the sum of the four windows.

    `torch.ops.trtllm.allgather_pg` appears here as a **fixture**, to build the
    input the way the target will — never as the reference. The reference is
    this rank's own rows times the sum of the four window scales, computed
    locally. The gathered tensor is also checked against a local `torch.cat`,
    so a wrong gather cannot be mistaken for a right reduce-scatter.
    """
    sizes = [1 + 4 * r for r in range(WORLD)]
    seed = 10500
    # Scales are k/4 and payloads k/8 with |k| <= 7, so every product and every
    # partial sum stays exactly representable and the gate can be bitwise.
    own = [_block(r, 0, sizes[r], seed, amp=7) for r in range(WORLD)]
    mine = own[RANK]
    gathered = torch.ops.trtllm.allgather_pg(mine, sizes, GROUP, PG_BOXED)
    torch.cuda.synchronize()
    _assert_bitwise(gathered, torch.cat(own, dim=0), "gather fixture")

    scale = float(RANK + 1) / 4.0
    partial = (gathered.float() * scale).to(torch.bfloat16)
    out = reducescatter(partial, sizes, GROUP)
    torch.cuda.synchronize()
    total_scale = sum((r + 1) / 4.0 for r in range(WORLD))
    assert out.shape == mine.shape, (out.shape, mine.shape)
    _assert_bitwise(out, (mine.float() * total_scale).to(torch.bfloat16), "round trip")
    dist.barrier()

    # The bare identity: reduce-scattering a gather returns WORLD copies summed.
    out2 = reducescatter(torch.ops.trtllm.allgather_pg(mine, sizes, GROUP, PG_BOXED), sizes, GROUP)
    torch.cuda.synchronize()
    _assert_bitwise(out2, (mine.float() * WORLD).to(torch.bfloat16), "rs(ag(x))")
    dist.barrier()


def check_cuda_graph_at_every_engine_batch_size() -> None:
    """Captured at all 35 engine batch sizes, then replayed interleaved.

    Capturing one row count proves nothing about an engine, which holds every
    graph in GRAPH_BATCH_SIZES alive at once and replays them in whatever order
    the served batch size takes. Each replay here gets a payload no earlier
    call used, so a graph that silently did not re-run keeps the previous
    answer and fails the bitwise gate.

    Two passes per order: one replay at a time with a synchronize and a full
    check before the next graph runs, then the whole order issued back to back
    with no synchronization in between — 35 differently sized collectives
    queued as one run of work, which is the only pass that can catch a hazard
    that needs the next call to already be in flight.
    """
    inputs, graphs, outputs = _capture_per_batch_size()
    assert len(graphs) == len(GRAPH_BATCH_SIZES) == 35
    seed = 110000
    for name, order in _replay_orders():
        for pos, rows in enumerate(order):
            seed += 1000
            _fresh_payload(inputs, rows, seed)
            graphs[rows].replay()
            torch.cuda.synchronize()
            _verify_replay(outputs, rows, seed, f"{name}/checked@{pos} rows={rows}")
        dist.barrier()
        staged: Dict[int, int] = {}
        for rows in order:
            seed += 1000
            staged[rows] = seed
            _fresh_payload(inputs, rows, seed)
        for rows in order:
            graphs[rows].replay()
        torch.cuda.synchronize()
        for pos, rows in enumerate(order):
            _verify_replay(outputs, rows, staged[rows], f"{name}/burst@{pos} rows={rows}")
        dist.barrier()
    # Discrimination for a gate of exactly 0: what a replay that did not happen
    # leaves behind is the previous payload's result. The two consecutive
    # references have to be far apart for that to be caught, which this checks.
    prev = _ref([16] * WORLD, RANK, seed - 1000)
    last = _ref([16] * WORLD, RANK, seed)
    margin = (last.float() - prev.float()).abs().max().item()
    assert margin > 8.0, f"stale-replay margin {margin}"
    del graphs, outputs, inputs
    dist.barrier()


def check_cuda_graph_one_site_per_moe_layer_in_every_batch_size_graph() -> None:
    """The decode graph this checkpoint captures, at all 35 batch sizes.

    1015 captured reduce-scatters, one memory pool. The sites are independent —
    each reads its own buffer and every one gets a distinct payload, so a site
    that returned a neighbour's data would fail the bitwise gate. Two replay
    orders rather than four: the per-order cost is 29x the single-site test's
    and the axis this one adds is site count, not order.
    """
    inputs, graphs, outputs = _capture_per_batch_size(sites=SITES_PER_DECODE_GRAPH, seed=21000)
    seed = 310000
    for name, order in _replay_orders()[2:]:
        for pos, rows in enumerate(order):
            seed += 1000
            _fresh_payload(inputs, rows, seed)
            graphs[rows].replay()
            torch.cuda.synchronize()
            _verify_replay(outputs, rows, seed, f"{name}/checked@{pos} rows={rows}")
        dist.barrier()
        staged: Dict[int, int] = {}
        for rows in order:
            seed += 1000
            staged[rows] = seed
            _fresh_payload(inputs, rows, seed)
        for rows in order:
            graphs[rows].replay()
        torch.cuda.synchronize()
        for pos, rows in enumerate(order):
            _verify_replay(outputs, rows, staged[rows], f"{name}/burst@{pos} rows={rows}")
        dist.barrier()
    del graphs, outputs, inputs
    dist.barrier()


def check_cuda_graph_step5_decode_graphs() -> None:
    """Step-5's decode graphs: 19 batch sizes, 88 reduce-scatters each, hidden 4096.

    The target captures one decode graph per `STEP5_GRAPH_BATCH_SIZES` entry
    and each holds one reduce-scatter per MoE layer, so 1672 captured calls
    share one pool. Every site reads its own buffer and gets a distinct payload
    per replay, so a site that returned a neighbour's slice, or a replay that
    did not re-run, fails the bitwise gate. Shuffled and extremes orders, each
    checked one replay at a time and then as an unsynchronised burst.
    """
    sizes = STEP5_GRAPH_BATCH_SIZES
    assert len(sizes) == 19 and sizes[-1] == 128, sizes
    hidden = STEP5_HIDDEN
    inputs, graphs, outputs = _capture_per_batch_size(
        sites=STEP5_MOE_SITES_PER_DECODE_GRAPH, seed=51000, batch_sizes=sizes, hidden=hidden
    )
    assert len(graphs) == len(sizes)
    seed = 910000
    for name, order in _replay_orders(sizes)[2:]:
        for pos, rows in enumerate(order):
            seed += 1000
            _fresh_payload(inputs, rows, seed, hidden)
            graphs[rows].replay()
            torch.cuda.synchronize()
            _verify_replay(outputs, rows, seed, f"step5 {name}/checked@{pos} rows={rows}", hidden)
        dist.barrier()
        staged: Dict[int, int] = {}
        for rows in order:
            seed += 1000
            staged[rows] = seed
            _fresh_payload(inputs, rows, seed, hidden)
        for rows in order:
            graphs[rows].replay()
        torch.cuda.synchronize()
        for pos, rows in enumerate(order):
            _verify_replay(
                outputs, rows, staged[rows], f"step5 {name}/burst@{pos} rows={rows}", hidden
            )
        dist.barrier()
    del graphs, outputs, inputs
    dist.barrier()


def check_cuda_graph_replays_survive_eager_calls_of_other_shapes() -> None:
    """Between two decode replays, a server runs work no graph holds.

    A prefill runs eagerly at a row count far outside the captured set, and a
    ragged reduce-scatter — the other split this op has — runs eagerly on the
    same communicator the graphs baked in.
    """
    inputs, graphs, outputs = _capture_per_batch_size(seed=41000)
    seed = 510000
    for i, rows in enumerate(reversed(GRAPH_BATCH_SIZES)):
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
        _fresh_payload(inputs, rows, seed)
        graphs[rows].replay()
        torch.cuda.synchronize()
        _verify_replay(outputs, rows, seed, f"after-eager@{i} rows={rows}")
        dist.barrier()
    del graphs, outputs, inputs
    dist.barrier()


def check_cuda_graph_holds_the_sizes_vector_it_captured() -> None:
    """`sizes` is a host argument: a replay re-runs the split it was captured with.

    Two graphs with different sizes vectors are captured into one pool and
    replayed in both orders; each keeps its own row split and its own output
    length. This is why a graph-captured decode step needs padded (uniform)
    row counts — the ragged split cannot be varied per replay.
    """
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
        dist.barrier()

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
            dist.barrier()
    del graphs, captured, buffers
    dist.barrier()


def check_the_gate_discriminates_a_wrong_reduce_scatter() -> None:
    """The bitwise gate rejects every plausible wrong result, by a wide margin.

    A gate of exactly 0 cannot be too loose, but it can be blind: if every
    rank's contribution looked alike, a missing addend or a wrong slice would
    pass it. These are the four ways this op could plausibly be wrong — the
    slice of a neighbouring group position, one rank's contribution missing,
    no reduction at all (this rank's own slice returned untouched), and the
    right data cut at the wrong offsets under a rotated ragged split. Every one
    of them has the right shape, so shape checking alone would let all four
    through.
    """
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
        # A wrong ragged slice has a different row count, so the comparison is
        # over the rows the two share — the point is the values, not the shape.
        n = min(variant.shape[0], got.shape[0])
        assert n > 0, name
        diff = (variant[:n].float() - got[:n].float()).abs()
        fraction = (diff != 0).float().mean().item()
        assert fraction > 0.9, f"{name}: only {fraction:.4f} of elements differ"
        assert diff.max().item() > 1.0, f"{name}: max difference {diff.max().item()}"
    dist.barrier()


def check_calls_pair_by_position_and_a_swapped_pair_realigns() -> None:
    """Calls pair by their **position** on the communicator, not by intent.

    One rank issuing two same-shaped calls in the opposite order to everybody
    else is not detected: both calls return, at the right shape, with finite
    values and no diagnostic of any kind. What each rank gets back is the sum
    of whatever payloads happened to meet at that position — asserted here
    bitwise against a locally computed mix, which is the strongest form of
    "silently wrong" there is.

    This op differs from the sibling all-gather in *how much* it corrupts. A
    gather hands each rank one block per rank, so a single disagreeing rank
    spoils one block: 1/4 of the elements at world size 4. Here the mispaired
    rank's payload is an addend of **every** element of every rank's slice, so
    every rank is wrong nearly everywhere (measured 0.982-0.993 against the
    floor of 0.9 below; the residue is elements where the two payloads happened
    to agree).

    Both `sizes` forms are driven, because they are not the same NCCL traffic:
    the even form issues one `ncclReduceScatter` while the ragged form issues a
    grouped `ncclReduce` per rank. The group pairs atomically — the ragged form
    mispairs exactly like the even one, rather than interleaving per-root.

    Finally, a swapped *pair* leaves the positions aligned once both calls are
    made, so the plain call at the end of each round is bitwise correct.
    """
    rows = 12
    ragged = [1 + 4 * r for r in range(WORLD)]
    for tag, sizes in (("even", None), ("ragged", ragged)):
        split = [rows] * WORLD if sizes is None else sizes
        base = 160000 if sizes is None else 165000
        first_seed, second_seed = base, base + 500
        first = _input(RANK, split, first_seed)
        second = _input(RANK, split, second_seed)
        dist.barrier()

        pos0, pos1 = _swapped_pair(first, second, sizes)
        torch.cuda.synchronize()
        for name, got in (("pos0", pos0), ("pos1", pos1)):
            assert got.shape == (split[RANK], HIDDEN), (tag, name, got.shape)
            assert bool(torch.isfinite(got.float()).all()), f"{tag} {name}: not finite"

        # What each position actually computed: rank 0 was one call out of step.
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

        # ...and that is not what this rank asked for, on every rank.
        intended = _ref(split, RANK, second_seed if RANK == 0 else first_seed)
        wrong = _fraction_wrong(pos0, intended)
        assert wrong > 0.9, f"{tag}: only {wrong:.4f} of elements differ from intent"
        dist.barrier()

        _assert_bitwise(
            reducescatter(_input(RANK, split, base + 900), sizes, GROUP),
            _ref(split, RANK, base + 900),
            f"{tag}: plain call after a swapped pair",
        )
        dist.barrier()


def check_a_mispaired_result_is_deterministic_rather_than_noise() -> None:
    """Because this op computes, "wrong" could have meant "unreproducible".

    It does not. On payloads with no exactness property, a mispaired call is
    bitwise identical across repeats *and* bitwise equal to the ring-order
    chain of the addends that met — the same accumulation order the aligned
    call uses, applied to the wrong operands. So the mispairing perturbs which
    tensors are summed and nothing else.

    That is the worse outcome for a caller, not the better one: re-running the
    step reproduces the wrong answer exactly, so a divergence cannot be found
    by looking for run-to-run instability.
    """
    rows = 12
    first_seed, second_seed = 162000, 162500
    first = _rand_input(RANK, rows, first_seed)
    second = _rand_input(RANK, rows, second_seed)
    dist.barrier()

    seen: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(4):
        pos0, pos1 = _swapped_pair(first, second, None)
        torch.cuda.synchronize()
        seen.append((pos0, pos1))
        dist.barrier()
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
    # Discrimination for those two bitwise gates: the aligned result of the
    # same call is a different tensor nearly everywhere.
    aligned = _ring_chain(rows, RANK, [second_seed if RANK == 0 else first_seed] * WORLD)
    wrong = _fraction_wrong(seen[0][0], aligned)
    assert wrong > 0.9, f"only {wrong:.4f} of elements differ from the aligned sum"
    dist.barrier()

    _assert_bitwise(
        reducescatter(_input(RANK, [rows] * WORLD, 163000), None, GROUP),
        _ref([rows] * WORLD, RANK, 163000),
        "plain call after four swapped pairs",
    )
    dist.barrier()


def check_an_extra_call_on_one_rank_misaligns_until_the_counts_match() -> None:
    """An odd number of extra calls does not realign; a swapped pair does.

    Rank 0 issues one call the others never issue, then all ranks issue four
    calls they intend to agree on. Every one of those five positions is
    mispaired by exactly one place, and it stays that way — there is no
    resynchronization point inside a stream of collectives, so the divergence
    is permanent rather than transient.

    The other ranks issue one catch-up call at the end, which is what makes
    this test terminate: the counts have to match for the last position to
    complete at all. Once they do, alignment is restored and the plain call at
    the end is bitwise correct. A caller with one rank permanently ahead gets
    the wedge instead, at whatever point the process next waits on the device.
    """
    rows, calls = 10, 4
    sizes = [rows] * WORLD
    shared = [170000 + 100 * k for k in range(calls)]
    extra_seed, catchup_seed = 179000, 179500
    dist.barrier()

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

    # Position p carried rank 0's p-th issued payload and everybody else's.
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
    dist.barrier()

    _assert_bitwise(
        reducescatter(_input(RANK, sizes, 178000), None, GROUP),
        _ref(sizes, RANK, 178000),
        "plain call after the call counts were equalized",
    )
    dist.barrier()


def check_the_stream_the_call_lands_on_is_not_part_of_the_match() -> None:
    """The op runs on whatever stream is current, and ranks need not agree.

    A serving engine moves the current stream under the model: the same forward
    runs on torch's graph-capture stream while a decode graph is being captured
    and on the serving stream otherwise, and a target cannot pin either. So
    "must the ranks agree on the stream?" is a precondition question. They do
    not have to — what pairs the calls is their order on the communicator.
    Three shapes: every rank on a side stream, one rank on a side stream while
    the others stay on the default, and ranks alternating in opposite patterns
    so that at every call index they disagree.

    Driven twice: once on the exact value set, where a wrong pairing shows, and
    once on payloads with no exactness property against the ring-order chain,
    where a changed *accumulation* order would show as well. Neither moves.
    """
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
        dist.barrier()

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
        dist.barrier()


def check_wrapper_guards_a_float8_input() -> None:
    """The wrapper rejects float8_e4m3fn.

    The op does not sum it as floats, so a caller doing post-quantization
    dispatch would get a wrong reduction.
    """
    rows = 4
    x = torch.full((rows * WORLD, 8), float(RANK + 1), dtype=torch.float32, device="cuda").to(
        torch.float8_e4m3fn
    )
    try:
        with validating(entry.reducescatter):
            reducescatter(x, None, GROUP)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrapper accepted a float8_e4m3fn input")
    dist.barrier()


def check_the_guard_refuses_every_dtype_it_cannot_sum() -> None:
    """The allowlist, driven on both ways a dtype goes wrong.

    `bool` maps to ncclInt8 exactly as float8_e4m3fn does, so ncclSum adds its
    raw bytes and the result reads as OR rather than as an error. `float64` is
    not in `getDtypeMap` at all, and the conversion that rejects it runs
    *inside* the group the op has already opened, which leaves the
    communicator mid-group (the entry's `is_valid` comment). The raw op is
    deliberately not called with either.

    The last assertion is the point: the group still reduces correctly after
    the guard refused, because `is_valid` raised before `ncclGroupStart`.
    """
    rows, seed = 8, 97000
    sizes = [rows] * WORLD

    for dtype in (torch.bool, torch.float64):
        x = torch.ones(4 * WORLD, 8, dtype=torch.float32, device="cuda").to(dtype)
        try:
            with validating(entry.reducescatter):
                reducescatter(x, None, GROUP)
        except AssertionError as exc:
            assert "not summable" in str(exc), str(exc)
        else:
            raise AssertionError(f"the guard accepted {dtype}")
        dist.barrier()

    _assert_bitwise(
        reducescatter(_input(RANK, sizes, seed), None, GROUP),
        _ref(sizes, RANK, seed),
        "the group after the guard refused an unsummable dtype",
    )
    dist.barrier()


def check_wrapper_guards_a_non_contiguous_input() -> None:
    """The wrapper rejects a non-contiguous input.

    The op reads it as packed memory and would reduce the wrong elements.
    """
    rows, seed = 8, 92000
    values = _input(RANK, [rows] * WORLD, seed)
    padded = torch.zeros(rows * WORLD, HIDDEN * 2, dtype=torch.bfloat16, device="cuda")
    padded[:, :HIDDEN] = values
    view = padded[:, :HIDDEN]
    assert not view.is_contiguous()

    try:
        with validating(entry.reducescatter):
            reducescatter(view, None, GROUP)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrapper accepted a non-contiguous input")
    dist.barrier()


def check_wrapper_guards_a_zero_dim_input() -> None:
    """A 0-d input segfaults inside the op, so the wrapper stops it first.

    The op is deliberately not called here: the crash is in
    `ReducescatterOp::run_list` and kills every rank in the job.
    """
    try:
        with validating(entry.reducescatter):
            reducescatter(torch.tensor(1.0, device="cuda"), None, GROUP)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrapper accepted a 0-d input")
    dist.barrier()


def check_wrapper_guards_a_sizes_list_of_the_wrong_length() -> None:
    """Neither wrong length is survivable, so the wrapper stops both.

    The raw op is deliberately not called with either. A list one entry short
    raises `IndexError: vector::_M_range_check: __n (which is WORLD) >=
    this->size()` on the highest rank while the others sit in the collective,
    which cannot be recovered from inside the job. A list one entry long makes
    the op reduce `sum(sizes)` rows out of an input that holds fewer — an
    out-of-bounds read; it returned a correctly shaped tensor of garbage (NaN)
    when this test was brought up, and reading past a live allocation is not
    something to repeat on every run.
    """
    rows, seed = 4, 93000
    sizes = [rows] * WORLD
    long_list = [rows] * (WORLD + 1)
    for bad in ([rows] * (WORLD - 1), long_list):
        try:
            with validating(entry.reducescatter):
                reducescatter(_input(RANK, sizes, seed), bad, GROUP)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"wrapper accepted sizes of length {len(bad)}")
    dist.barrier()


def check_wrapper_guards_a_split_that_does_not_cover_the_input() -> None:
    """The wrapper rejects a split that does not cover the input.

    Rows the split does not reach are silently dropped, and a split summing
    past the input is an out-of-bounds read.
    """
    seed = 94000
    odd = WORLD * 2 + 1
    x = _input(RANK, [1] * odd, seed)  # a row count len(group) does not divide
    y = _input(RANK, [4] * WORLD, seed)
    short_split = [2] * WORLD
    for x_, sizes_ in ((x, None), (y, short_split), (y, [6] * WORLD)):
        try:
            with validating(entry.reducescatter):
                reducescatter(x_, sizes_, GROUP)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"wrapper accepted an uncovered split {sizes_}")
    dist.barrier()


def check_the_group_still_works_after_the_negative_tests() -> None:
    """The raises above leave the communicator usable — checked, not assumed."""
    rows, seed = 8, 95000
    sizes = [rows] * WORLD
    out = reducescatter(_input(RANK, sizes, seed), None, GROUP)
    _assert_bitwise(out, _ref(sizes, RANK, seed), "after the negative tests")
    dist.barrier()


# The communicator-establishing check stays first: it is the only one that can
# observe GROUP's first-ever call, and every later check needs the
# communicator it leaves. Which one depends on the symbol under test.
FIRST_CHECK = {
    "pg": check_cuda_graph_captures_a_cold_first_call,
    "mpi": check_mpi_symbol_first_call_is_eager_then_captures,
}

CHECKS = (
    check_certified_cells,
    check_step5_numerics,
    check_uniform_reduce_scatter,
    check_ragged_reduce_scatter,
    check_output_is_fresh_and_input_is_untouched,
    check_trailing_dims_are_preserved,
    check_dtypes_reduce_arithmetically,
    check_group_selects_a_rank_subset,
    check_group_order_does_not_change_the_output,
    check_reduction_is_deterministic_and_accumulates_in_the_input_dtype,
    check_uniform_form_and_an_explicit_even_split_agree_bitwise,
    check_round_trip_with_a_gather_returns_each_rank_its_own_rows,
    check_cuda_graph_at_every_engine_batch_size,
    check_cuda_graph_one_site_per_moe_layer_in_every_batch_size_graph,
    check_cuda_graph_step5_decode_graphs,
    check_cuda_graph_replays_survive_eager_calls_of_other_shapes,
    check_cuda_graph_holds_the_sizes_vector_it_captured,
    check_the_gate_discriminates_a_wrong_reduce_scatter,
    # The call-order block. Each of these deliberately disagrees about call
    # order and each restores alignment before it returns — the plain call
    # every one of them ends on is what proves it.
    check_calls_pair_by_position_and_a_swapped_pair_realigns,
    check_a_mispaired_result_is_deterministic_rather_than_noise,
    check_an_extra_call_on_one_rank_misaligns_until_the_counts_match,
    check_the_stream_the_call_lands_on_is_not_part_of_the_match,
    check_wrapper_guards_a_float8_input,
    check_wrapper_guards_a_non_contiguous_input,
    check_wrapper_guards_a_zero_dim_input,
    check_wrapper_guards_a_sizes_list_of_the_wrong_length,
    check_wrapper_guards_a_split_that_does_not_cover_the_input,
    check_the_guard_refuses_every_dtype_it_cannot_sum,
    check_the_group_still_works_after_the_negative_tests,
)


def _rank_identity(transport: str) -> Tuple[int, int]:
    """(rank, world) from whoever started this process.

    `mpirun` names the rank in its own variables; torch.distributed's env://
    rendezvous below reads RANK and WORLD_SIZE, so they are set from them.
    """
    if transport == "mpi":
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMIX_RANK", "-1")))
        world = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0"))
        assert rank >= 0 and world > 0, "mpi transport: mpirun named no rank for this process"
        os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), str(world)
        return rank, world
    return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])


def _run_one_rank(device: Optional[int] = None) -> int:
    """Body of one rank: run every check.

    A failing rank exits nonzero and the launcher (or mpirun) stops its
    siblings.
    """
    global RANK, WORLD, GROUP, PG_BOXED, SUBGROUPS, TRANSPORT
    global reducescatter, validating, entry
    TRANSPORT = os.environ[_TRANSPORT_ENV]
    assert TRANSPORT in TRANSPORTS, TRANSPORT
    if TRANSPORT == "pg":
        os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    else:
        os.environ.pop("TLLM_DISABLE_MPI", None)
    rank, world = _rank_identity(TRANSPORT)
    # One node: the rank owns device `rank`. Across nodes (--srun) the caller
    # passes the node-local index. Never read LOCAL_RANK here: a spawned rank
    # inherits the launching shell's value, the same on every rank, and NCCL
    # then refuses the group with "invalid usage".
    if device is None:
        device = rank
    torch.cuda.set_device(device)
    # device_id keeps NCCL communicator teardown off the implicit-device path
    # that measurably wedges a watchdog at exit.
    dist.init_process_group("nccl", init_method="env://", device_id=torch.device("cuda", device))

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import reducescatter as entry
    from tensorrt_llm._utils import mpi_disabled

    PG_BOXED = dist.distributed_c10d._get_default_group().boxed()
    if TRANSPORT == "pg":
        # The wrapper hands the boxed group to the _pg symbol -- the routing
        # the product's non-MPI mode does in _torch/distributed/ops.py. The
        # partial is only the call surface: the guard is armed on the entry
        # itself, and `reference` and `CELLS` are read off `entry.reducescatter`.
        assert mpi_disabled()
        reducescatter = functools.partial(entry.reducescatter, process_group=PG_BOXED)
    else:
        # The MPI-session symbol: its communicator comes from the MPI world
        # mpirun started, the one a dep4 serving world runs in.
        assert not mpi_disabled()
        from mpi4py import MPI

        assert MPI.COMM_WORLD.Get_rank() == rank and MPI.COMM_WORLD.Get_size() == world, (
            MPI.COMM_WORLD.Get_rank(),
            MPI.COMM_WORLD.Get_size(),
            rank,
            world,
        )
        reducescatter = functools.partial(entry.reducescatter, process_group=None)

    # This file is run as a worker process, not collected by pytest, so the
    # repo's `__extra_import_path__` mechanism is not in play and the shared
    # helper (tests/unittest/_torch/modeling_v2/_validating.py) is reached the
    # plain way. The mutation is confined to these rank processes.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _validating import validating as _validating_cm

    validating = _validating_cm
    RANK = rank
    WORLD = world
    GROUP = list(range(WORLD))
    assert WORLD >= 2, f"a collective needs at least 2 ranks, got {WORLD}"
    # A rank subset is a real sub-ProcessGroup, not just a shorter `group`
    # list: the subset check runs the _pg symbol on the communicator it is
    # handed under either transport. Every rank must enter new_group for each
    # subset, in the same order, or it deadlocks -- so build them all here.
    for _ranks in ([0, 1], [WORLD - 2, WORLD - 1]):
        key = tuple(sorted(set(_ranks)))
        if key not in SUBGROUPS:
            SUBGROUPS[key] = dist.new_group(list(key))

    checks = (FIRST_CHECK[TRANSPORT],) + CHECKS
    for check in checks:
        started = time.monotonic()
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # Exit hard rather than return: a rank that leaves a collective
            # early wedges every sibling in it, and the launcher kills the
            # rest the moment it sees this nonzero exit.
            os._exit(13)
        if RANK == 0:
            print(f"[rank 0] {check.__name__}: {time.monotonic() - started:.1f}s", flush=True)
    dist.barrier()
    torch.cuda.synchronize()
    print(
        f"[rank {RANK}] transport={TRANSPORT} world={WORLD}: {len(checks)} checks passed",
        flush=True,
    )
    sys.stdout.flush()
    if TRANSPORT == "mpi":
        # mpirun reports a rank that leaves without finalizing MPI as failed.
        # Finalizing does not touch the NCCL communicators.
        from mpi4py import MPI

        MPI.COMM_WORLD.Barrier()
        MPI.Finalize()
    # Exit without tearing the process group down: destroying a communicator
    # that captured collectives into CUDA graphs measurably wedges NCCL's
    # watchdog, and these are throwaway processes the launcher owns anyway.
    os._exit(0)


def _scrubbed_env(**extra: str) -> Dict[str, str]:
    """This process's environment for the ranks, minus the launching step's MPI.

    Drop the launching srun step's PMIx/PMI/OMPI namespace from the children:
    they inherit it and are then registered as participants of that step, so
    the first one to exit (hard, as a failing rank does) trips srun's pmix
    errhandler, which SIGKILLs the whole step -- launcher and real error with
    it -- before anything is reported. The `mpi` transport's own mpirun starts
    a fresh MPI world from this clean environment.
    """
    env = dict(os.environ, **extra)
    return {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}


def _spawn_ranks(transport: str) -> None:
    """Start one rank per claimed device and wait for them under the deadline.

    `pg`: one worker process per device, rendezvousing through
    torch.distributed's env://; each gets its own session, and on the first
    nonzero exit or the deadline every session is killed. `mpi`: one mpirun,
    which owns its ranks; on the deadline it is asked to stop them (SIGTERM)
    before it is killed.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, (
        "set CUDA_VISIBLE_DEVICES to the devices this run owns, "
        "e.g. export CUDA_VISIBLE_DEVICES=0,1,2,3"
    )
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size >= 2, (
        f"CUDA_VISIBLE_DEVICES names {world_size} device(s); a collective test needs at least 2"
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = _scrubbed_env(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        **{_TRANSPORT_ENV: transport},
    )
    me = str(Path(__file__).resolve())
    print(
        f"[launcher] transport={transport}: {world_size} ranks, rendezvous on :{port}",
        flush=True,
    )

    if transport == "mpi":
        env.pop("TLLM_DISABLE_MPI", None)
        forwarded = [
            arg
            for key in (
                "MASTER_ADDR",
                "MASTER_PORT",
                "CUDA_VISIBLE_DEVICES",
                "PYTHONPATH",
                "PYTHONNOUSERSITE",
                "TMPDIR",
                _TRANSPORT_ENV,
            )
            if key in env
            for arg in ("-x", key)
        ]
        cmd = ["mpirun", "-n", str(world_size), "--oversubscribe", "--allow-run-as-root"]
        mpirun = subprocess.Popen(
            cmd + forwarded + [sys.executable, me, _WORKER_FLAG],
            env=env,
            start_new_session=True,
        )
        try:
            code = mpirun.wait(timeout=DEADLINE_S)
        except subprocess.TimeoutExpired:
            os.killpg(mpirun.pid, signal.SIGTERM)
            try:
                mpirun.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(mpirun.pid, signal.SIGKILL)
                mpirun.wait()
            raise AssertionError(f"the {world_size}-rank mpirun did not finish in {DEADLINE_S}s")
        if code != 0:
            raise AssertionError(f"the {world_size}-rank mpirun exited {code}")
        return

    children = [
        subprocess.Popen(
            [sys.executable, me, _WORKER_FLAG],
            env=dict(env, RANK=str(r), TLLM_DISABLE_MPI="1"),
            start_new_session=True,
        )
        for r in range(world_size)
    ]

    def _kill_all() -> None:
        for p in children:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for p in children:
            p.wait()

    deadline = time.monotonic() + DEADLINE_S
    live = set(range(world_size))
    failure = None
    while live and failure is None:
        if time.monotonic() > deadline:
            failure = f"did not finish in {DEADLINE_S}s (wedged)"
            break
        for r in sorted(live):
            code = children[r].poll()
            if code is None:
                continue
            live.discard(r)
            if code != 0:
                failure = f"rank {r} exited {code}"
                break
        time.sleep(0.5)
    if failure is not None:
        _kill_all()
        raise AssertionError(f"the {world_size}-rank run failed: {failure}")


def _transport_arg(argv: Sequence[str]) -> str:
    chosen = [a.split("=", 1)[1] for a in argv if a.startswith("--transport=")]
    assert len(chosen) == 1 and chosen[0] in TRANSPORTS, (
        f"pass exactly one --transport=<{'|'.join(TRANSPORTS)}>, got {chosen}"
    )
    return chosen[0]


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    if "--srun" in sys.argv:
        import _rank_job

        _device = _rank_job.adopt_srun_env()
        os.environ[_TRANSPORT_ENV] = "pg"
        sys.exit(_run_one_rank(device=_device))
    _spawn_ranks(_transport_arg(sys.argv[1:]))
    print("OK")
