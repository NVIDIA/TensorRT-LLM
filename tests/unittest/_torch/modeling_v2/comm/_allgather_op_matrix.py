# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the allgather catalog entry.

A collective cannot be exercised in one process, so this script is its own
launcher: run it plainly and it starts one rank per device named in
CUDA_VISIBLE_DEVICES, under a deadline the parent enforces by killing the
ranks. A wedged collective hangs rather than raising — and this op wedges
rather than raising for every rank-argument-disagreement case probed — so the
deadline is what keeps a broken kernel from taking the calling run down with
it.

Beyond the lockstep surface (uniform / ragged / dtypes / CUDA graphs) the
`_pg` job covers what a serving engine adds: the engine's own attention-DP
synchronisation interleaved with the op, the engine's stream switching, and
what disagreeing on call order actually does.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _allgather_op_matrix.py

That runs two jobs, in this order, and both must exit 0:

- the `_pg` job: plain worker processes that rendezvous through
  torch.distributed's env://, every `CHECKS` entry on `world_size` ranks. It
  drives the cells that do not state `process_group`, through the
  `allgather_pg` symbol on a boxed c10d group;
- the MPI-session job: the same number of ranks started by `mpirun`, every
  `MPI_CHECKS` entry. It drives the cells that state `process_group=None` at
  that world size, through `torch.ops.trtllm.allgather` on the communicator
  the MPI session builds -- the symbol and communicator a dep4, tep4 or tp4
  world launched under mpirun reaches. Its `mpirun` gets an environment with
  the launching allocation's PMI/PMIx/Open MPI/Slurm variables removed, so it
  starts its own ranks on this node rather than joining the allocation's step.

The world-8 cells -- the dep8 gather and the tep8 q-split gather, eight ranks
over two nodes of four -- are a third job, the MPI-session checks again on
ranks an MPI launcher starts across both nodes, one per device, as the dep8
and tep8 engines' ranks are. One process cannot place ranks on another node,
so the launcher is the environment's (the collected test takes it from
`MODELING_V2_MPI_LAUNCH`):

    <launcher> python _allgather_op_matrix.py --mpi-rank-worker

Each MPI-session job drives the cells of its own world size and node count,
and runs each check on the kind of cell it holds: the four-rank job the dep4
gather's and the tep4 and tp4 q-split's, the eight-rank job the dep8 gather's
and the tep8 q-split's. A rank that wedges inside a collective is ended by
faulthandler at `MPI_DEADLINE_S`, and the launcher ends its siblings.

Not a pytest module, despite the `check_*` bodies. They are one fixed
sequence inside a single job rather than independent cases: each reads
module-global rank state that only the rank body binds, and several assert
on communicator state the previous one left behind. Collected as tests they
would run at world size 1 against unbound globals — so neither this file's
name nor its function names match pytest's collection patterns, which is what
keeps it uncollectable however pytest is pointed at this tree.

The collected entry point is `test_modeling_v2_allgather_op_matrix.py` beside
this file: it starts these jobs and turns their exit codes into assertions.
Both halves are started by file path: the launcher must not import
`tensorrt_llm` (that would give it CUDA and MPI state of its own), and the
ranks reach the catalog by absolute import, so neither needs a package.
"""

import faulthandler
import functools
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

assert torch.cuda.is_available(), "allgather requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
_MPI_WORKER_FLAG = "--mpi-rank-worker"
# Each job's own deadline; `_rank_job` reads their sum off DEADLINE_S. Each
# MPI-session rank also ends itself at MPI_DEADLINE_S, and the collected test
# allows the world-8 job's launcher that plus a grace.
PG_DEADLINE_S = 1200
MPI_DEADLINE_S = 600
DEADLINE_S = PG_DEADLINE_S + MPI_DEADLINE_S

HIDDEN = 2560  # DeepSeek-V3-Lite hidden size, the caller that motivates this

# The row counts one engine's decode graphs are captured at. trtllm's graph
# runner instantiates one graph per configured batch size, and under attention
# data parallelism a decode call's per-rank row count *is* that batch size:
# with `cuda_graph_config.max_batch_size = 256` and the default list that is 35
# graphs at [1..32, 64, 128, 256], all alive at once in one memory pool and
# replayed interleaved as the served batch size moves. So the surface a
# captured gather has to be correct over is the whole set plus the transitions
# between its members, not one representative row count.
GRAPH_BATCH_SIZES: Tuple[int, ...] = tuple(range(1, 33)) + (64, 128, 256)

# Gathers inside one decode graph for this checkpoint: 30 layers with
# `first_k_dense_replace = 1`, so 29 expert-parallel MoE calls, each preceded
# by one gather of the rank's own rows into the full token set.
SITES_PER_DECODE_GRAPH = 29

# Bound inside the rank body, never in the launcher: importing the entry
# pulls in tensorrt_llm, which calls MPI_Init at import, and the launcher has
# to stay free of MPI and CUDA state.
allgather: Any = None
entry: Any = None
MPI_COMM: Any = None
RANK = 0
WORLD = 1
#: Distinct hosts the MPI-session job's ranks run on.
NODES = 1
GROUP: List[int] = [0]
# The engine's own cross-rank synchronisation object, built from the real
# Mapping a serving engine builds under attention data parallelism. Bound in
# the rank body for the same reason as the rest.
HOST_GROUP: Any = None
SUBGROUPS: Dict[Tuple[int, ...], Any] = {}
PG_BOXED: Any = None


def _host_allgather(obj: Any) -> List[Any]:
    """One host-side cross-rank step, the shape of the engine's per-step sync.

    A serving engine agrees the per-rank token counts once per step with a
    host-side collective (`Distributed.tp_allgather`; TorchDist rides this same
    torch.distributed machinery). A dedicated gloo group stands in for the
    engine object here: constructing the engine's own `TorchDist` needs ray
    importable, which a plain development machine does not owe this test.
    """
    gathered: List[Any] = [None] * WORLD
    dist.all_gather_object(gathered, obj, group=HOST_GROUP)
    return gathered


def _payload(
    rank: int,
    rows: int,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
    trailing: Tuple[int, ...] = (HIDDEN,),
) -> torch.Tensor:
    """One rank's contribution: values distinct per (rank, seed), exact in dtype.

    Every rank can regenerate every other rank's payload — cuRAND is
    deterministic for a given (seed, shape, dtype) — which is what lets the
    reference below be arithmetic rather than a second collective. The float
    value sets are multiples of 1/8 (bounded at 2.0 for e4m3, which only
    represents that spacing below it), so a payload survives the dtype cast
    exactly and every assertion in this file can be bitwise.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed * 977 + rank + 1)
    shape = (rows, *trailing)
    if dtype is torch.uint8:
        return torch.randint(0, 256, shape, generator=gen, device="cuda", dtype=torch.int32).to(
            torch.uint8
        )
    if dtype in (torch.int32, torch.int64):
        return torch.randint(
            -(2**20), 2**20, shape, generator=gen, device="cuda", dtype=torch.int64
        ).to(dtype)
    amp = 16 if dtype is torch.float8_e4m3fn else 120
    raw = torch.randint(-amp, amp + 1, shape, generator=gen, device="cuda", dtype=torch.int32)
    return (raw.float() / 8.0).to(dtype)


def _gather_ref(
    sizes: Sequence[int],
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
    trailing: Tuple[int, ...] = (HIDDEN,),
    ranks: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Arithmetic reference: the concatenation the collective is supposed to make.

    Built on this rank alone from the same seeds every rank uses, in ascending
    rank order. Never from a second collective.
    """
    ranks = list(range(WORLD)) if ranks is None else list(ranks)
    # Delegates to the entry: what the concatenation is supposed to be is the
    # entry's claim, and building it here again would be a second place to fix
    # if it were ever wrong.
    return entry.allgather.reference(
        [_payload(r, sizes[i], seed, dtype, trailing) for i, r in enumerate(ranks)]
    )


def _assert_bitwise(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    """Gate of exactly zero: the op moves bytes, so nothing may differ.

    Tightened from the default dtype-aware tolerances rather than loosened —
    a gather computes nothing, so any difference at all is a wrong gather.
    """
    try:
        assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        if out.dtype is torch.float8_e4m3fn:
            # assert_close has no e4m3 band; a byte view is the exact gate anyway.
            out, ref = (out.view(torch.uint8), ref.view(torch.uint8))
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


def _adp_step_counts(steps: int = 8) -> List[List[int]]:
    """Per-rank token counts a serving engine hands its layers, step by step.

    One seeded host RNG, so the vector is identical on every rank without a
    collective — which is also what lets the engine's own `tp_allgather` of it
    be checked against something rather than trusted. The mix is what a live
    attention-DP engine produces: steps where it has equalized the per-rank
    batch (it does that whenever a decode step is graph-eligible), ragged
    decode steps, and ragged prefill-sized steps.
    """
    rng = random.Random(90210)
    out: List[List[int]] = []
    for step in range(steps):
        if step % 3 == 0:
            out.append([rng.choice([1, 2, 8, 17, 32])] * WORLD)
        elif step % 3 == 1:
            out.append([rng.randint(1, 32) for _ in range(WORLD)])
        else:
            out.append([rng.randint(1, 1024) for _ in range(WORLD)])
    return out


def _gather_on_a_side_stream(x: torch.Tensor, side: torch.cuda.Stream) -> torch.Tensor:
    """Issue one gather with `side` current, joined on both ends."""
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        out = allgather(x, None, GROUP)
    torch.cuda.current_stream().wait_stream(side)
    return out


def _warm_up_off_capture_stream(body, reps: int = 2) -> None:
    """Run `body` on a side stream, then rejoin, so a capture can follow.

    Two things have to be done before a capture and cannot be done inside one:
    torch wants the work warmed on a non-default stream
    before a capture (the communicator itself already exists -- see
    check_cuda_graph_captures_a_cold_first_call).
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
    sites: int = 1, seed: int = 11000
) -> Tuple[Dict[int, List[torch.Tensor]], Dict[int, Any], Dict[int, List[torch.Tensor]]]:
    """One graph per GRAPH_BATCH_SIZES entry, one shared pool, all left alive.

    That is the state a graph runner ends up in: every capture after the first
    goes into the first one's pool, and none of them is ever destroyed.
    `sites` gathers go into each graph, each reading its own persistent input
    buffer, so the payload of every site can be moved independently.
    Returns (inputs, graphs, outputs), all keyed by row count.
    """
    inputs: Dict[int, List[torch.Tensor]] = {}
    graphs: Dict[int, Any] = {}
    outputs: Dict[int, List[torch.Tensor]] = {}
    pool = None
    for rows in GRAPH_BATCH_SIZES:
        xs = [_payload(RANK, rows, seed + 13 * s) for s in range(sites)]

        def body(xs: List[torch.Tensor] = xs) -> List[torch.Tensor]:
            return [allgather(x, None, GROUP) for x in xs]

        # Persistent output buffers outside the shared pool; see the reduce-scatter
        # matrix for why -- a burst of replays with no sync lets pooled outputs
        # alias, and the copy inside the capture, ordered after this graph's
        # gather, reads the right value before the next graph overwrites the pool.
        persistent = [
            torch.empty(rows * WORLD, HIDDEN, dtype=torch.bfloat16, device="cuda")
            for _ in range(sites)
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


def _replay_orders() -> List[Tuple[str, List[int]]]:
    """Four ways a served batch size moves through the captured set."""
    asc = list(GRAPH_BATCH_SIZES)
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


def _fresh_payload(inputs: Dict[int, List[torch.Tensor]], rows: int, seed: int) -> None:
    """Overwrite every site's persistent input buffer for this row count."""
    for site, x in enumerate(inputs[rows]):
        x.copy_(_payload(RANK, rows, seed + 13 * site))


def _verify_replay(
    outputs: Dict[int, List[torch.Tensor]], rows: int, seed: int, where: str
) -> None:
    """Every site's captured output tensor holds this payload's gather, bitwise."""
    for site, out in enumerate(outputs[rows]):
        _assert_bitwise(out, _gather_ref([rows] * WORLD, seed + 13 * site), f"{where} site={site}")


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
    seed = 100
    x = _payload(RANK, rows, seed)
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = allgather(x, None, GROUP)
    # Capture records the op, it does not run it; the first replay is what fills
    # `captured` from this input.
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(captured, _gather_ref([rows] * WORLD, seed), "cold first capture")

    # A second replay after a refilled input reflects that input, not the first.
    x.copy_(_payload(RANK, rows, seed + 500))
    torch.cuda.synchronize()
    dist.barrier()
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(captured, _gather_ref([rows] * WORLD, seed + 500), "replay after refill")
    del graph
    dist.barrier()


def _on_mpi_session(cell: Any) -> bool:
    """Whether `cell` drives the MPI-session symbol: it states `process_group=None`."""
    return "process_group" in cell.spec and cell.spec["process_group"] is None


def check_certified_cells() -> None:
    """Drive the entry's `_pg` cells, inside `validating`.

    The other checks in this job say things no cell can -- CUDA graph
    behaviour, call-order disagreement, which stream the call lands on. This one
    says the plain thing: for every configuration these cells claim, the
    gather is the concatenation, and the guard admits the input a shipped
    target passes.
    """
    driven = 0
    for i, cell in enumerate(entry.allgather.CELLS):
        if _on_mpi_session(cell):
            continue  # the MPI-session job drives it (`MPI_CHECKS`)
        driven += 1
        spec = cell.spec
        rows, trailing, dtype = spec["rows"], spec["trailing"], spec["dtype"]
        seed = 9000 + 13 * i
        x = _payload(RANK, rows, seed, dtype, trailing)
        with validating(entry.allgather):
            out = allgather(x, spec["sizes"], GROUP)
        assert out.shape == (rows * WORLD, *trailing), (cell.why, out.shape)
        _assert_bitwise(out, _gather_ref([rows] * WORLD, seed, dtype, trailing), cell.why)
        dist.barrier()
    assert driven > 0, "no cell drives the _pg symbol"


def check_uniform_gather() -> None:
    """sizes=None: every rank holds `rows` rows, the result is WORLD * rows.

    Decode-like (1, 2, 8, 32) through prefill-like (2048) row counts, which is
    the whole span an attention-DP rank feeds its expert-parallel MoE call.
    """
    for rows in (1, 2, 8, 32, 2048):
        seed = 1000 + rows
        x = _payload(RANK, rows, seed)
        out = allgather(x, None, GROUP)
        assert out.shape == (rows * WORLD, HIDDEN), out.shape
        assert out.dtype is x.dtype and out.device == x.device
        assert out.is_contiguous()
        _assert_bitwise(out, _gather_ref([rows] * WORLD, seed), f"uniform rows={rows}")
        dist.barrier()


def check_ragged_gather() -> None:
    """sizes=[...]: per-rank row counts differ, the result is sum(sizes).

    Attention data parallelism produces exactly this — each rank owns its own
    requests — so the ragged form is the one a non-padded step takes, including
    the case where one rank has no rows at all.
    """
    for i, sizes in enumerate(_sizes_vectors()):
        seed = 3000 + 7 * i
        x = _payload(RANK, sizes[RANK], seed)
        out = allgather(x, sizes, GROUP)
        assert out.shape == (sum(sizes), HIDDEN), (sizes, out.shape)
        assert out.dtype is x.dtype and out.is_contiguous()
        _assert_bitwise(out, _gather_ref(sizes, seed), f"ragged sizes={sizes}")
        dist.barrier()


def check_output_is_fresh_and_input_is_untouched() -> None:
    """The op allocates its result; the caller keeps ownership of `input`."""
    rows, seed = 8, 4100
    x = _payload(RANK, rows, seed)
    before = x.clone()
    out = allgather(x, None, GROUP)
    torch.cuda.synchronize()
    assert out.data_ptr() != x.data_ptr(), "output aliased the input buffer"
    _assert_bitwise(x, before, "input after the call")

    # Clobbering the input afterwards cannot reach the result.
    x.fill_(-7.0)
    torch.cuda.synchronize()
    _assert_bitwise(out, _gather_ref([rows] * WORLD, seed), "output after input clobber")
    dist.barrier()


def check_trailing_dims_are_preserved() -> None:
    """dim 0 is the gather axis; every other dim is carried through untouched."""
    cases: List[Tuple[Tuple[int, ...], Optional[List[int]]]] = [
        ((2, HIDDEN // 2), None),
        ((2, HIDDEN // 2), [1 + 4 * r for r in range(WORLD)]),
        ((), None),  # 1-D: a flat vector gathers to WORLD * rows elements
        ((), [1 + 4 * r for r in range(WORLD)]),
    ]
    for i, (trailing, sizes) in enumerate(cases):
        seed = 5000 + 11 * i
        rows = 6 if sizes is None else sizes[RANK]
        x = _payload(RANK, rows, seed, trailing=trailing)
        out = allgather(x, sizes, GROUP)
        total = rows * WORLD if sizes is None else sum(sizes)
        assert out.shape == (total, *trailing), (trailing, sizes, out.shape)
        _assert_bitwise(
            out,
            _gather_ref([rows] * WORLD if sizes is None else sizes, seed, trailing=trailing),
            f"trailing={trailing} sizes={sizes}",
        )
        dist.barrier()


def check_dtypes_move_bitwise() -> None:
    """The payload dtypes an attention-DP dispatch moves, in both forms.

    bf16 hidden states, plus what a post-quantization dispatch carries next to
    them: packed NVFP4 bytes and their scale factors (uint8), fp8 activations,
    int32 expert ids and fp32 routing scales. Nothing is computed, so each is
    a byte-for-byte move — the multi-tensor sibling op `allgather_list` exists
    for gathering them in one call and is not this entry.
    """
    sizes = [1 + 4 * r for r in range(WORLD)]
    for i, dtype in (
        (0, torch.float16),
        (1, torch.float32),
        (2, torch.int32),
        (3, torch.uint8),
        (4, torch.float8_e4m3fn),
    ):
        seed = 6000 + 13 * i
        x = _payload(RANK, 16, seed, dtype)
        out = allgather(x, None, GROUP)
        assert out.dtype is dtype and out.shape == (16 * WORLD, HIDDEN)
        _assert_bitwise(out, _gather_ref([16] * WORLD, seed, dtype), f"uniform {dtype}")
        dist.barrier()

        xr = _payload(RANK, sizes[RANK], seed, dtype)
        outr = allgather(xr, sizes, GROUP)
        assert outr.dtype is dtype and outr.shape == (sum(sizes), HIDDEN)
        _assert_bitwise(outr, _gather_ref(sizes, seed, dtype), f"ragged {dtype}")
        dist.barrier()


def check_group_selects_a_rank_subset() -> None:
    """`group` names the communicator's ranks; ranks outside it must not call."""
    subset = [0, 1]
    rows, seed = 4, 7100
    if RANK in subset:
        # new_group returns a sentinel (not a group) on ranks outside the subset,
        # so only a member may box it.
        pg = SUBGROUPS[tuple(subset)].boxed()
        out = allgather(_payload(RANK, rows, seed), None, subset, process_group=pg)
        assert out.shape == (rows * len(subset), HIDDEN), out.shape
        _assert_bitwise(out, _gather_ref([rows] * len(subset), seed, ranks=subset), "subset gather")
    dist.barrier()


def check_group_order_does_not_change_the_output_order() -> None:
    """The result is ordered by ascending rank, whatever order `group` lists."""
    rows, seed = 4, 7200
    x = _payload(RANK, rows, seed)
    out = allgather(x, None, list(reversed(GROUP)))
    _assert_bitwise(out, _gather_ref([rows] * WORLD, seed), "reversed group list")
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
    # leaves behind is the previous payload's gather. Measured by mutating this
    # file to skip one row count's refresh — the checked pass then failed on all
    # four ranks at that row count, 163158 of 163840 elements wrong (99.6%),
    # greatest absolute difference 30.0. The assertion below keeps that
    # separation from silently shrinking.
    prev = _gather_ref([16] * WORLD, seed - 1000)
    last = _gather_ref([16] * WORLD, seed)
    margin = (last.float() - prev.float()).abs().max().item()
    assert margin > 8.0, f"stale-replay margin {margin}"
    del graphs, outputs, inputs
    dist.barrier()


def check_cuda_graph_one_site_per_moe_layer_in_every_batch_size_graph() -> None:
    """The decode graph this checkpoint captures, at all 35 batch sizes.

    1015 captured gathers, one memory pool. The sites are independent — each
    reads its own buffer and every one gets a distinct payload, so a site that
    returned a neighbour's data would fail the bitwise gate. Two replay orders
    rather than four: the per-order cost is 29x the single-site test's and the
    axis this one adds is site count, not order.
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


def check_cuda_graph_replays_survive_eager_calls_of_other_shapes() -> None:
    """Between two decode replays, a server runs work no graph holds.

    A prefill runs eagerly at a row count far outside the captured set, and a
    ragged gather — the other transport this op has — runs eagerly on the same
    communicator the graphs baked in.
    """
    inputs, graphs, outputs = _capture_per_batch_size(seed=41000)
    seed = 510000
    for i, rows in enumerate(reversed(GRAPH_BATCH_SIZES)):
        wide = (2048, 8192, 1500)[i % 3]
        _assert_bitwise(
            allgather(_payload(RANK, wide, 610000 + i), None, GROUP),
            _gather_ref([wide] * WORLD, 610000 + i),
            f"eager uniform {wide} rows",
        )
        ragged = [(11 * i + 3 * r) % 97 for r in range(WORLD)]
        _assert_bitwise(
            allgather(_payload(RANK, ragged[RANK], 710000 + i), ragged, GROUP),
            _gather_ref(ragged, 710000 + i),
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
        x = _payload(RANK, sizes[RANK], seed)
        _warm_up_off_capture_stream(lambda x=x, sizes=sizes: allgather(x, sizes, GROUP))
        graph = torch.cuda.CUDAGraph()
        ctx = torch.cuda.graph(graph) if pool is None else torch.cuda.graph(graph, pool=pool)
        with ctx:
            captured[tag] = allgather(x, sizes, GROUP)
        pool = graph.pool() if pool is None else pool
        buffers[tag], graphs[tag] = x, graph
        assert captured[tag].shape == (sum(sizes), HIDDEN), captured[tag].shape
        dist.barrier()

    for round_seed in (82000, 83000):
        for tag, sizes in (("second", second), ("first", first)):
            buffers[tag].copy_(_payload(RANK, sizes[RANK], round_seed))
            graphs[tag].replay()
            torch.cuda.synchronize()
            _assert_bitwise(
                captured[tag],
                _gather_ref(sizes, round_seed),
                f"ragged replay {tag} sizes={sizes}",
            )
            dist.barrier()
    del graphs, captured, buffers
    dist.barrier()


def check_the_gate_discriminates_a_wrong_gather() -> None:
    """The bitwise gate rejects every plausible wrong gather, by a wide margin.

    A gate of exactly 0 cannot be too loose, but it can be blind: if every
    rank's payload looked alike, a misordered or short gather would pass it.
    These are the four ways this op could plausibly be wrong — rank blocks in
    the wrong order, one rank's block standing in for another's, a rank that
    never contributed, and the right data landing at the wrong offsets under a
    rotated ragged split. Every one of them has the right shape, so shape
    checking alone would let all four through.

    Measured on the certified path (world size 4, bf16, hidden 2560), as
    (fraction of elements differing, greatest absolute difference): reversed
    (0.9958, 30.0), one block substituted (0.2489, 30.0), one rank missing
    (0.2489, 15.0), rotated split (0.9602, 30.0) — against a gate of 0. The
    two 0.2489 figures sit just under the arithmetic ceiling of 1/WORLD: a
    variant that corrupts one rank's block cannot move more than a quarter of
    a 4-rank gather, which is why the floors below are per variant.
    """
    rows, seed = 8, 91000
    uniform = allgather(_payload(RANK, rows, seed), None, GROUP)
    blocks = [_payload(r, rows, seed) for r in range(WORLD)]
    _assert_bitwise(uniform, torch.cat(blocks, dim=0), "uniform baseline")

    sizes = [1 + 4 * r for r in range(WORLD)]
    ragged = allgather(_payload(RANK, sizes[RANK], seed), sizes, GROUP)
    _assert_bitwise(ragged, _gather_ref(sizes, seed), "ragged baseline")

    rotated_split = sizes[1:] + sizes[:1]
    one_block = 0.9 / WORLD  # a variant that corrupts a single rank's block
    wrong = [
        ("rank blocks reversed", uniform, torch.cat(blocks[::-1], dim=0), 0.9),
        (
            "rank 0's block twice",
            uniform,
            torch.cat([blocks[0]] + blocks[1:-1] + [blocks[0]], dim=0),
            one_block,
        ),
        (
            "one rank never contributed",
            uniform,
            torch.cat(blocks[:-1] + [torch.zeros_like(blocks[-1])], dim=0),
            one_block,
        ),
        (
            "ragged split rotated",
            ragged,
            torch.cat([_payload(r, rotated_split[r], seed) for r in range(WORLD)], 0),
            0.9,
        ),
    ]
    for name, got, variant, floor in wrong:
        assert variant.shape == got.shape, (name, variant.shape, got.shape)
        diff = (variant.float() - got.float()).abs()
        fraction = (diff != 0).float().mean().item()
        assert fraction > floor, f"{name}: only {fraction:.4f} of elements differ"
        assert diff.max().item() > 8.0, f"{name}: max difference {diff.max().item()}"
    dist.barrier()


def check_the_engines_own_cross_rank_step_is_not_on_this_communicator() -> None:
    """What a serving engine synchronises with, next to what this op uses.

    Under attention data parallelism the engine agrees the per-rank token
    counts once per step with a **host-side** collective, not with the device
    collective this op runs. The two never share an ordering; see
    `_host_allgather` for what stands in for the engine object here.
    """
    counts = [3 + 2 * r for r in range(WORLD)]
    assert list(_host_allgather(counts[RANK])) == counts
    dist.barrier()


def check_interleaved_with_the_engines_attention_dp_synchronisation() -> None:
    """The op inside a forward, between the engine's own cross-rank steps.

    The shape a served forward has: once per step the engine agrees the
    per-rank token counts on the host (the previous test's collective, which is
    where `attn_metadata.all_rank_num_tokens` comes from), then the model
    issues one gather per expert-parallel layer on rows padded to
    `max(all_rank_num_tokens)`.

    Nothing synchronises here. Every step's payloads and gathers, and the host
    collective between steps, are issued back to back for all 8 steps before a
    single result is read — 232 gathers per rank in flight, which is what makes
    the launch queue saturate the way it does in a served forward rather than
    in a lockstep test loop. All of them are checked afterwards, bitwise.
    """
    steps = _adp_step_counts()
    pending: List[Tuple[int, int, int, int, torch.Tensor]] = []
    for step, counts in enumerate(steps):
        # The engine-shaped step-level sync. It is a host collective, so the
        # device work queued above it stays in flight across it — that
        # interleaving is the point of this test.
        observed = list(_host_allgather(counts[RANK]))
        assert observed == counts, (step, observed, counts)
        rows = max(counts)
        for layer in range(SITES_PER_DECODE_GRAPH):
            seed = 130000 + 1000 * step + 13 * layer
            out = allgather(_payload(RANK, rows, seed), None, GROUP)
            pending.append((step, layer, rows, seed, out))
    assert len(pending) == len(steps) * SITES_PER_DECODE_GRAPH
    for step, layer, rows, seed, out in pending:
        _assert_bitwise(
            out,
            _gather_ref([rows] * WORLD, seed),
            f"adp step={step} layer={layer} rows={rows}",
        )
    dist.barrier()


def check_the_stream_the_call_lands_on_is_not_part_of_the_match() -> None:
    """The op runs on whatever stream is current, and ranks need not agree.

    A serving engine moves the current stream under the model: the same
    forward runs on torch's graph-capture stream while a decode graph is being
    captured and on the serving stream otherwise, and a target cannot pin
    either. So "must the ranks agree on the stream?" is a precondition
    question. They do not have to — what pairs the calls is their order on the
    communicator. Three shapes: every rank on a side stream, one rank on a side
    stream while the others stay on the default, and ranks alternating in
    opposite patterns so that at every call index they disagree.
    """
    side = torch.cuda.Stream()
    rows = 24
    cases = [
        ("every rank on a side stream", lambda i: True),
        ("only rank 0 on a side stream", lambda i: RANK == 0),
        ("ranks alternating opposite", lambda i: (i + RANK) % 2 == 0),
    ]
    for c, (name, use_side) in enumerate(cases):
        for i in range(4):
            seed = 150000 + 1000 * c + 13 * i
            x = _payload(RANK, rows, seed)
            out = _gather_on_a_side_stream(x, side) if use_side(i) else allgather(x, None, GROUP)
            _assert_bitwise(out, _gather_ref([rows] * WORLD, seed), f"{name} i={i}")
        dist.barrier()


def check_wrapper_guards_a_non_contiguous_input() -> None:
    """The wrapper rejects a non-contiguous input: the op reads it as packed
    memory and would gather the wrong elements."""
    rows, seed = 8, 92000
    values = _payload(RANK, rows, seed)
    padded = torch.zeros(rows, HIDDEN * 2, dtype=torch.bfloat16, device="cuda")
    padded[:, :HIDDEN] = values
    view = padded[:, :HIDDEN]
    assert not view.is_contiguous()

    try:
        with validating(entry.allgather):
            allgather(view, None, GROUP)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrapper accepted a non-contiguous input")
    dist.barrier()


def check_wrapper_guards_a_sizes_list_of_the_wrong_length() -> None:
    """The wrapper rejects a `sizes` list that is not one entry per rank: the op
    sizes its output from `sizes` alone, so a short list drops the trailing
    ranks and a long one appends uninitialized rows."""
    rows, seed = 4, 93000
    for sizes in ([rows] * (WORLD - 1), [rows] * (WORLD + 1)):
        try:
            with validating(entry.allgather):
                allgather(_payload(RANK, rows, seed), sizes, GROUP)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"wrapper accepted sizes of length {len(sizes)}")
    dist.barrier()


def check_wrapper_guards_a_zero_dim_input() -> None:
    """A 0-d input segfaults inside the op, so the wrapper stops it first.

    The op is deliberately not called here: the crash is in
    `AllgatherOp::run_list` and kills every rank in the job.
    """
    try:
        with validating(entry.allgather):
            allgather(torch.tensor(1.0, device="cuda"), None, GROUP)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrapper accepted a 0-d input")
    dist.barrier()


def check_call_order_disagreement_corrupts_silently() -> None:
    """Ranks that disagree on the *order* of two equal-sized gathers get wrong
    data back, with no error and no hang.

    Argument disagreement wedges (the contract's first precondition). Order
    disagreement does not, as long as the two calls carry the same number of
    bytes: what pairs the calls is their position on the communicator, so a
    swapped pair pairs each rank's first call with the others' second, every
    call returns a correctly *shaped* tensor and nothing reports anything. That
    is the failure mode a target has to design against — it surfaces as an
    accuracy loss, not as a crash.

    Measured on the certified path (world size 4, bf16, hidden 2560): the rank
    that swapped gets exactly 3/4 of its elements wrong in both results, every
    other rank exactly 1/4 — the disagreeing rank's block — against floors of
    0.7 and 0.2 here. Nothing raised and nothing hung, in 20 probe iterations
    out of 20 and in every run of this test.

    Kept last, and it puts the communicator back: a swapped *pair* leaves the
    per-communicator positions aligned once both calls are made, so the final
    assertion is a plain gather coming back bitwise correct.
    """
    rows, seed = 12, 160000
    first, second = _payload(RANK, rows, seed), _payload(RANK, rows, seed + 500)
    ref_first = _gather_ref([rows] * WORLD, seed)
    ref_second = _gather_ref([rows] * WORLD, seed + 500)
    dist.barrier()

    if RANK == 0:
        got_second = allgather(second, None, GROUP)
        got_first = allgather(first, None, GROUP)
    else:
        got_first = allgather(first, None, GROUP)
        got_second = allgather(second, None, GROUP)
    torch.cuda.synchronize()

    floor = 0.7 if RANK == 0 else 0.2
    for name, got, ref in (
        ("first", got_first, ref_first),
        ("second", got_second, ref_second),
    ):
        assert got.shape == ref.shape, (name, got.shape, ref.shape)
        wrong = (got != ref).float().mean().item()
        assert wrong > floor, f"{name}: only {wrong:.4f} of elements differ"
    dist.barrier()

    x = _payload(RANK, rows, seed + 900)
    _assert_bitwise(
        allgather(x, None, GROUP),
        _gather_ref([rows] * WORLD, seed + 900),
        "gather after an order disagreement",
    )
    dist.barrier()


CHECKS = (
    # Stays first: it is the only test that can observe GROUP's first-ever
    # call, and every later test needs the communicator it builds.
    check_cuda_graph_captures_a_cold_first_call,
    check_certified_cells,
    check_uniform_gather,
    check_ragged_gather,
    check_output_is_fresh_and_input_is_untouched,
    check_trailing_dims_are_preserved,
    check_dtypes_move_bitwise,
    check_group_selects_a_rank_subset,
    check_group_order_does_not_change_the_output_order,
    check_cuda_graph_at_every_engine_batch_size,
    check_cuda_graph_one_site_per_moe_layer_in_every_batch_size_graph,
    check_cuda_graph_replays_survive_eager_calls_of_other_shapes,
    check_cuda_graph_holds_the_sizes_vector_it_captured,
    check_the_gate_discriminates_a_wrong_gather,
    check_the_engines_own_cross_rank_step_is_not_on_this_communicator,
    check_interleaved_with_the_engines_attention_dp_synchronisation,
    check_the_stream_the_call_lands_on_is_not_part_of_the_match,
    check_wrapper_guards_a_non_contiguous_input,
    check_wrapper_guards_a_sizes_list_of_the_wrong_length,
    check_wrapper_guards_a_zero_dim_input,
    # Stays last: it deliberately disagrees on call order, and although a
    # swapped pair realigns the communicator (its final assertion proves it),
    # nothing after it should depend on that.
    check_call_order_disagreement_corrupts_silently,
)


def _run_one_rank() -> int:
    """Body of one spawned rank: run every test; a failing rank exits nonzero
    and the launcher kills its siblings."""
    global RANK, WORLD, GROUP, HOST_GROUP, SUBGROUPS, PG_BOXED, allgather, validating, entry
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    # device_id keeps NCCL communicator teardown off the implicit-device path
    # that measurably wedges a watchdog at exit.
    dist.init_process_group("nccl", init_method="env://", device_id=torch.device("cuda", rank))

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import allgather as entry

    # One partial instead of a per-call argument: every check's call site keeps
    # the certified shape, and the wrapper hands the boxed group to the _pg
    # symbol -- the same routing the product's non-MPI mode does in
    # _torch/distributed/ops.py. The partial is only the call surface: the
    # guard is armed on the entry itself, and `reference` and
    # `CELLS` are read off `entry.allgather` directly.
    PG_BOXED = dist.distributed_c10d._get_default_group().boxed()
    allgather = functools.partial(entry.allgather, process_group=PG_BOXED)

    # This file is run as a spawned worker process, not collected by pytest,
    # so the repo's `__extra_import_path__` mechanism is not in play and the
    # sibling helper has to be reached the plain way. The mutation is confined
    # to these rank processes; the pytest process that launched them never sees
    # it, which is what its own sys.path check is there to protect.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _validating import validating as _validating_cm

    validating = _validating_cm
    RANK = rank
    WORLD = world
    GROUP = list(range(WORLD))
    assert WORLD >= 2, f"a collective needs at least 2 ranks, got {WORLD}"
    HOST_GROUP = dist.new_group(backend="gloo")
    # A rank subset is a real sub-ProcessGroup on this path, not just a shorter
    # `group` list: the _pg op gathers on the communicator it is handed, so a
    # 2-rank gather needs a 2-rank group. Every rank must enter new_group for
    # each subset, in the same order, or it deadlocks -- so build them all here.
    for _ranks in ([0, 1], [WORLD - 2, WORLD - 1]):
        key = tuple(sorted(set(_ranks)))
        if key not in SUBGROUPS:
            SUBGROUPS[key] = dist.new_group(list(key))

    for check in CHECKS:
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
    dist.barrier()
    torch.cuda.synchronize()
    print(f"[rank {RANK}] {len(CHECKS)} checks passed", flush=True)
    # Exit without tearing the process group down: destroying a communicator
    # that captured collectives into CUDA graphs measurably wedges NCCL's
    # watchdog, and these are throwaway processes the launcher owns anyway.
    sys.stdout.flush()
    os._exit(0)


# ---- the MPI-session job ------------------------------------------------------
#
# Everything from here to `MPI_CHECKS` drives the cells that state
# `process_group=None`, on the symbol and the communicator a dep4, tep4, tp4,
# dep8 or tep8 serving world reaches: ranks started by mpirun,
# `torch.ops.trtllm.allgather`, and the NCCL communicator tensorrt_llm's MPI
# session builds for `group` at its first call. These ranks have no
# torch.distributed group; they synchronize through the MPI session itself. One
# job drives the cells of its own world size: four ranks of one node
# (`_mpirun_ranks`), or eight over two nodes (started by the environment's
# launcher).

#: The entry's own `__call__` and `reference`, counted (`_count_entry_calls`).
SEEN: Dict[str, int] = {"__call__": 0, "reference": 0}
#: What each MPI-session cell ran through the entry: [calls, references].
DRIVEN: Dict[int, List[int]] = {}


def _mpi_cells() -> List[Tuple[int, Any]]:
    """(index, cell) of every cell this MPI-session job drives: its world size's."""
    return [
        (i, c)
        for i, c in enumerate(entry.allgather.CELLS)
        if _on_mpi_session(c) and c.spec["world_size"] == WORLD
    ]


def _mpi_sync() -> None:
    """Every rank's device work done, then every rank here."""
    torch.cuda.synchronize()
    MPI_COMM.Barrier()


def _count_entry_calls() -> None:
    """What `_validating.certifying` does under pytest, for this rank body.

    The entry's own `__call__` (with `is_valid` interposed first, as
    `validating` does) and its own `reference` are wrapped to count, so every
    cell can assert that both ran for it rather than a local copy.
    """
    cls = type(entry.allgather)
    call, ref = cls.__dict__["__call__"], cls.__dict__["reference"]

    def counted_call(self, *args, **kwargs):
        SEEN["__call__"] += 1
        self.is_valid(*args, **kwargs)
        return call(self, *args, **kwargs)

    def counted_reference(self, *args, **kwargs):
        SEEN["reference"] += 1
        return ref(self, *args, **kwargs)

    cls.__call__ = counted_call
    cls.reference = counted_reference


def _account(i: int, before: Dict[str, int]) -> None:
    """Add what the entry ran since `before` to cell `i`."""
    got = DRIVEN.setdefault(i, [0, 0])
    got[0] += SEEN["__call__"] - before["__call__"]
    got[1] += SEEN["reference"] - before["reference"]


def _gather(x: torch.Tensor, spec: Dict[str, Any]) -> torch.Tensor:
    """The call the dep4 and dep8 targets make, with the arguments the cell states."""
    return entry.allgather(x, spec["sizes"], GROUP, process_group=spec["process_group"])


def _width(spec: Dict[str, Any]) -> int:
    assert len(spec["trailing"]) == 1 and spec["dtype"] is torch.bfloat16, spec
    assert spec["world_size"] == WORLD and spec["sizes"] is None, (spec, WORLD)
    # A cell without `nodes` is one node's, as the dep4 cells are.
    assert spec.get("nodes", 1) == NODES, (spec, NODES)
    return spec["trailing"][0]


def _rows(rank: int, shape: Tuple[int, ...], seed: int) -> torch.Tensor:
    """`rank`'s rows of `shape`, regenerable on every rank from the shared seed.

    Multiples of 1/8 with |x| <= 31/8, 63 values: two ranks' rows agree in about
    one element in 63, so a block from the wrong rank or at the wrong offset
    changes nearly every element it covers.
    """
    gen = torch.Generator(device="cuda").manual_seed((seed * 977 + rank) * 131 + 7)
    raw = torch.randint(-31, 32, shape, generator=gen, device="cuda", dtype=torch.int8)
    return raw.to(torch.bfloat16).div_(8.0)


def _bits(rank: int, shape: Tuple[int, ...], seed: int) -> torch.Tensor:
    """Every bf16 bit pattern equally likely: NaNs, infinities, -0.0, subnormals."""
    gen = torch.Generator(device="cuda").manual_seed((seed * 977 + rank) * 131 + 11)
    raw = torch.randint(-(2**15), 2**15, shape, generator=gen, device="cuda", dtype=torch.int32)
    return raw.to(torch.int16).view(torch.bfloat16)


def _differs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Whether two bf16 tensors differ in any bit: a device bool, compared as int64 words."""
    assert a.shape == b.shape and a.dtype == b.dtype == torch.bfloat16, (a.shape, b.shape)
    return torch.ne(a.reshape(-1).view(torch.int64), b.reshape(-1).view(torch.int64)).any()


def _fraction_moved(a: torch.Tensor, b: torch.Tensor) -> float:
    """The fraction of elements whose bits differ between two bf16 tensors."""
    return (a.view(torch.int16) != b.view(torch.int16)).float().mean().item()


def _sweep_rows(spec: Dict[str, Any]) -> List[int]:
    """The row counts a sweep cell names, largest first."""
    lo, hi = spec["sweep_rows"]
    rows = set(range(lo, hi + 1, spec["sweep_step"])) | set(spec.get("sweep_also", ())) | {hi}
    assert spec["rows"] == hi and all(lo <= r <= hi for r in rows), spec
    # Largest first, so the caching allocator sizes its blocks for the top once.
    return sorted(rows, reverse=True)


def check_mpi_session_is_the_launchers_world() -> None:
    """The ranks are mpirun's, one device each, and the MPI session spans them all.

    `getComm` builds the communicator through the MPI session, so a session of
    one process per rank -- what a launch without the ranks in place gives --
    would wedge the first call instead of failing it. Over several nodes each
    node's ranks hold that node's devices 0..n-1, in rank order, as the
    launcher places them.
    """
    from tensorrt_llm._utils import mpi_rank, mpi_world_size

    assert os.environ.get("TLLM_DISABLE_MPI") != "1", "the MPI-session job runs with MPI"
    assert mpi_world_size() == MPI_COMM.Get_size() == WORLD, (mpi_world_size(), WORLD)
    assert mpi_rank() == MPI_COMM.Get_rank() == RANK, (mpi_rank(), RANK)
    placed = MPI_COMM.allgather((socket.gethostname(), torch.cuda.current_device()))
    hosts = list(dict.fromkeys(host for host, _ in placed))
    assert len(hosts) == NODES and WORLD % NODES == 0, placed
    per_node = WORLD // NODES
    assert [hosts.index(h) for h, _ in placed] == [r // per_node for r in range(WORLD)], placed
    assert [d for _, d in placed] == [r % per_node for r in range(WORLD)], placed
    if RANK == 0:
        print(
            f"[mpi] {torch.cuda.get_device_name()} sm_"
            f"{''.join(map(str, torch.cuda.get_device_capability()))}, NCCL "
            f"{torch.cuda.nccl.version()}; {WORLD} ranks on {NODES} node(s): {placed}",
            flush=True,
        )
    cells = _mpi_cells()
    assert cells, "no cell states process_group=None"
    for _, cell in cells:
        if _on_split(cell):
            _split_width(cell.spec)
        else:
            _width(cell.spec)
    _mpi_sync()


def check_mpi_eager_sweep_cells() -> None:
    """Every sweep cell: at each row count it names, the gather is the reference's, bitwise.

    One full-height block per rank, regenerated for every rank of the world
    from the shared seeds. A row count passes the leading `rows` rows of this rank's
    block, as a rank whose rows were padded to that T_max does. These are the
    group's first calls, so the first of them builds the communicator, eagerly.
    A mismatch is agreed across the ranks before any rank raises, so the
    collective that counts it is made by all of them.
    """
    for i, cell in _mpi_cells():
        spec = cell.spec
        if "sweep_rows" not in spec:
            continue
        width = _width(spec)
        rows_list = _sweep_rows(spec)
        top = rows_list[0]
        seed = 20000 + 13 * i
        full = [_rows(r, (top, width), seed) for r in range(WORLD)]
        mine = full[RANK]
        bad = torch.zeros(len(rows_list), dtype=torch.bool, device="cuda")
        before = dict(SEEN)
        _mpi_sync()
        for k, rows in enumerate(rows_list):
            out = _gather(mine[:rows], spec)
            assert out.shape == (WORLD * rows, width), (cell.why, rows, out.shape)
            bad[k] = _differs(out, entry.allgather.reference([block[:rows] for block in full]))
        torch.cuda.synchronize()
        everyone = MPI_COMM.allgather(bad.nonzero().flatten().tolist())
        failing = sorted({rows_list[k] for got in everyone for k in got})
        if failing:
            rows = failing[0]
            out = _gather(mine[:rows], spec)
            ref = entry.allgather.reference([block[:rows] for block in full])
            raise AssertionError(
                f"{cell.why}: {len(failing)} of {len(rows_list)} row counts differ from the "
                f"reference on some rank, the smallest {rows} here in "
                f"{_fraction_moved(out, ref):.4f} of the elements"
            )
        _account(i, before)
        del full, mine
        _mpi_sync()


def check_mpi_the_gate_discriminates_a_wrong_gather() -> None:
    """The bitwise gate, on the sweep cells' payload, rejects every plausible wrong gather.

    At the top of the first sweep cell: rank blocks in the wrong order, rank 0's
    block standing in for the last rank's, the last rank never contributing
    (its block left zero), and every block taken one row late. Each has the
    right shape, so only the payload can tell them apart; each moves nearly
    every element of the blocks it gets wrong, which is at most 1/WORLD of the
    result when it gets one block wrong. A job holding only q-split cells skips
    it: the q-split's own discrimination check is in `check_mpi_split_cells`.
    """
    if _only_split_cells():
        return
    i, cell = next((i, c) for i, c in _mpi_cells() if "sweep_rows" in c.spec)
    spec = cell.spec
    width, rows = _width(spec), spec["rows"]
    before = dict(SEEN)
    blocks = [_rows(r, (rows + 1, width), 25000) for r in range(WORLD)]
    got = [block[:rows] for block in blocks]
    out = _gather(got[RANK], spec)
    assert not bool(_differs(out, entry.allgather.reference(got))), "baseline differs"
    one_block = 0.9 / WORLD
    last = WORLD - 1
    fractions = []
    for name, wrong, floor in (
        ("rank blocks reversed", torch.cat(got[::-1]), 0.9),
        (f"rank 0's block in rank {last}'s place", torch.cat(got[:-1] + got[:1]), one_block),
        (
            f"rank {last} never contributed",
            torch.cat(got[:-1] + [torch.zeros_like(got[-1])]),
            one_block,
        ),
        ("every block one row late", torch.cat([block[1:] for block in blocks]), 0.9),
    ):
        assert wrong.shape == out.shape, (name, wrong.shape, out.shape)
        fraction = _fraction_moved(out, wrong)
        assert fraction > floor, f"{name}: only {fraction:.4f} of elements differ"
        fractions.append(round(fraction, 4))
    _account(i, before)
    everyone = MPI_COMM.allgather(fractions)
    if RANK == 0:
        print(
            f"[mpi] wrong gathers at rows {rows} move (reversed, substituted, missing, late) "
            f"{everyone[0]} of the elements on rank 0, min over ranks "
            f"{[min(f[k] for f in everyone) for k in range(4)]}",
            flush=True,
        )
    _mpi_sync()


def check_mpi_bits_payload_cells() -> None:
    """Every `payload="bits"` cell: the dep4 or dep8 padding skew at its row count, bitwise.

    Rank r's leading `valid_rows[r]` rows are arbitrary bf16 bit patterns and
    the rest is +0.0, the zero padding the target writes up to T_max. The op
    moves bytes, so the gather is the concatenation bit for bit -- NaN payloads
    and -0.0 included, which a float comparison could not show. Four calls
    return the same bits in four distinct tensors, none of them the input, and
    the input is unchanged.
    """
    for i, cell in _mpi_cells():
        spec = cell.spec
        if spec.get("payload") != "bits" or _on_split(cell):
            continue  # the q-split's bits cell runs in `check_mpi_split_cells`
        width, rows, valid = _width(spec), spec["rows"], tuple(spec["valid_rows"])
        assert len(valid) == WORLD and max(valid) == rows, (cell.why, valid)
        seed = 30000 + 13 * i
        blocks = []
        for r in range(WORLD):
            block = torch.zeros(rows, width, dtype=torch.bfloat16, device="cuda")
            block[: valid[r]] = _bits(r, (valid[r], width), seed)
            blocks.append(block)
        mine = blocks[RANK]
        kept = mine.clone()
        before = dict(SEEN)
        _mpi_sync()
        outs = [_gather(mine, spec) for _ in range(4)]
        ref = entry.allgather.reference(blocks)
        for k, out in enumerate(outs):
            assert out.shape == (WORLD * rows, width) and out.is_contiguous(), (cell.why, k)
            assert not bool(_differs(out, ref)), f"{cell.why}: call {k} differs from the reference"
        pointers = {out.data_ptr() for out in outs}
        assert len(pointers) == len(outs) and mine.data_ptr() not in pointers, cell.why
        assert not bool(_differs(mine, kept)), f"{cell.why}: the input was written"
        # The payload holds every class of bit pattern the cell names.
        full = blocks[valid.index(rows)]
        tiny = torch.finfo(torch.bfloat16).tiny
        counts = [
            int(torch.isnan(full).sum()),
            int(torch.isinf(full).sum()),
            int(((full == 0) & torch.signbit(full)).sum()),
            int(((full != 0) & (full.abs() < tiny)).sum()),
        ]
        assert min(counts) > 0, f"{cell.why}: (NaN, inf, -0.0, subnormal) counts {counts}"
        if RANK == 0:
            print(
                f"[mpi] {cell.why[:44]}: (NaN, inf, -0.0, subnormal) elements in the full "
                f"block {counts}, moved bitwise",
                flush=True,
            )
        _account(i, before)
        del blocks, mine, kept, outs, ref, full
        _mpi_sync()


def check_mpi_graph_cells() -> None:
    """The graph cells together: the decode graphs a dep4 or dep8 engine holds, one pool.

    One graph per captured batch, largest first as the engine's graph runner
    captures, each with `sites` gathers that read their own persistent input
    and copy into their own persistent output (outside the pool, whose blocks a
    later capture reuses). The group is warmed eagerly at each batch before its
    capture, as the engine's warmup does. Then four replay orders, each twice:
    one replay at a time on fresh payloads checked bitwise against the
    reference, then every graph's payload staged and all replayed back to back
    with no synchronization. Last, arbitrary bit patterns: each replay bitwise
    the reference and the same calls made eagerly. A job holding only q-split
    cells (context steps, never captured) has no graph and skips it.
    """
    if _only_split_cells():
        return
    cells = [(i, c) for i, c in _mpi_cells() if "graph_rows" in c.spec]
    assert cells, "no graph cell"
    spec0 = cells[0][1].spec
    pool_rows, sites, width = tuple(spec0["graph_rows"]), spec0["sites"], _width(spec0)
    by_rows = {c.spec["rows"]: i for i, c in cells}
    assert sorted(by_rows) == sorted(pool_rows), (sorted(by_rows), pool_rows)
    for _, c in cells:
        assert c.spec["sites"] == sites and tuple(c.spec["graph_rows"]) == pool_rows, c.why

    inputs: Dict[int, torch.Tensor] = {}
    outputs: Dict[int, torch.Tensor] = {}
    graphs: Dict[int, Any] = {}
    pool = None
    for rows in sorted(pool_rows, reverse=True):
        spec = entry.allgather.CELLS[by_rows[rows]].spec
        before = dict(SEEN)
        x = _rows(RANK, (sites, rows, width), 40000 + rows)
        out = torch.empty(sites, WORLD * rows, width, dtype=torch.bfloat16, device="cuda")

        def body(x: torch.Tensor = x, out: torch.Tensor = out, spec: Dict[str, Any] = spec) -> None:
            for s in range(sites):
                out[s].copy_(_gather(x[s], spec))

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                body()
        torch.cuda.current_stream().wait_stream(side)
        _mpi_sync()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            body()
        pool = graph.pool() if pool is None else pool
        inputs[rows], outputs[rows], graphs[rows] = x, out, graph
        _account(by_rows[rows], before)
        _mpi_sync()

    def payload(rank: int, rows: int, seed: int, bits: bool) -> torch.Tensor:
        return (_bits if bits else _rows)(rank, (sites, rows, width), seed)

    def stage(rows: int, seed: int, bits: bool = False) -> None:
        inputs[rows].copy_(payload(RANK, rows, seed, bits))

    def verify(rows: int, seed: int, where: str, bits: bool = False) -> None:
        before = dict(SEEN)
        everyone = [payload(r, rows, seed, bits) for r in range(WORLD)]
        ref = torch.stack(
            [entry.allgather.reference([p[s] for p in everyone]) for s in range(sites)]
        )
        _account(by_rows[rows], before)
        if bool(_differs(outputs[rows], ref)):
            bad = [s for s in range(sites) if bool(_differs(outputs[rows][s], ref[s]))]
            raise AssertionError(f"{where} rows={rows}: sites {bad} differ from the reference")

    asc = sorted(pool_rows)
    shuffled = list(asc)
    random.Random(1234).shuffle(shuffled)
    extremes = [r for pair in zip(reversed(asc), asc) for r in pair][: len(asc)]
    seed = 50000
    for name, order in (
        ("ascending", asc),
        ("descending", list(reversed(asc))),
        ("shuffled", shuffled),
        ("extremes", extremes),
    ):
        assert sorted(order) == asc, (name, order)
        for pos, rows in enumerate(order):
            seed += 1
            stage(rows, seed)
            _mpi_sync()
            graphs[rows].replay()
            torch.cuda.synchronize()
            verify(rows, seed, f"{name}/checked@{pos}")
        _mpi_sync()
        staged: Dict[int, int] = {}
        for rows in order:
            seed += 1
            staged[rows] = seed
            stage(rows, seed)
        _mpi_sync()
        for rows in order:
            graphs[rows].replay()
        torch.cuda.synchronize()
        for pos, rows in enumerate(order):
            verify(rows, staged[rows], f"{name}/burst@{pos}")
        _mpi_sync()

    # A replay that did not happen leaves the previous payload's gather; the
    # gate of exactly zero sees it in every element where consecutive payloads
    # differ.
    stale = _fraction_moved(payload(RANK, 16, seed - 1, False), payload(RANK, 16, seed, False))
    assert stale > 0.9, f"consecutive payloads differ in only {stale:.4f} of elements"

    for rows in shuffled:
        seed += 1
        stage(rows, seed, bits=True)
        _mpi_sync()
        graphs[rows].replay()
        torch.cuda.synchronize()
        verify(rows, seed, "bit patterns", bits=True)
        before = dict(SEEN)
        spec = entry.allgather.CELLS[by_rows[rows]].spec
        eager = torch.stack([_gather(inputs[rows][s], spec) for s in range(sites)])
        _account(by_rows[rows], before)
        if bool(_differs(outputs[rows], eager)):
            raise AssertionError(f"bit patterns, rows={rows}: a replay differs from eager")
        _mpi_sync()
    graphs.clear()
    _mpi_sync()


def check_mpi_round_trip_through_a_combine_in_one_graph() -> None:
    """The dep4 or dep8 decode step's collective sequence, captured: gather, window, combine.

    Per MoE layer the target gathers every rank's rows, lets this rank's expert
    window act on all of them and combines the world's partials, all on the one
    communicator `getComm` caches for the group. Here each of the `sites`
    layers gathers through the entry, scales by a per-rank factor standing in
    for the window, and combines with `torch.ops.trtllm.reducescatter` -- a
    fixture standing in for the combine, never the reference -- every batch's
    graph in one pool. The gathered rows are checked against the entry's
    reference, and rank i gets back its own rows times the sum of the world's
    factors, computed locally, so a wrong combine cannot hide behind a right
    gather or the reverse. A job holding only q-split cells has no graph and
    skips it.
    """
    if _only_split_cells():
        return
    cells = [(i, c) for i, c in _mpi_cells() if "graph_rows" in c.spec]
    spec = cells[0][1].spec
    by_rows = {c.spec["rows"]: i for i, c in cells}
    sites, width, pool_rows = spec["sites"], _width(spec), spec["graph_rows"]
    # Payload k/8 with |k| <= 7 and factors (r + 1) / 4: every product and every
    # partial sum is exact in bf16, so both gates are bitwise. At world 8 a
    # partial sum is k * s / 32 with s at most 1 + 2 + ... + 8 = 36, so |k * s|
    # stays at or below 252, inside bf16's exact integers.
    factor = (RANK + 1) / 4.0
    total = sum((r + 1) / 4.0 for r in range(WORLD))
    pool = None
    for rows in sorted(pool_rows, reverse=True):
        own = torch.empty(sites, rows, width, dtype=torch.bfloat16, device="cuda")
        gathered = torch.empty(sites, WORLD * rows, width, dtype=torch.bfloat16, device="cuda")
        back = torch.empty(sites, rows, width, dtype=torch.bfloat16, device="cuda")

        def body() -> None:
            for s in range(sites):
                g = _gather(own[s], spec)
                gathered[s].copy_(g)
                back[s].copy_(torch.ops.trtllm.reducescatter(torch.mul(g, factor), None, GROUP))

        def payload(rank: int, seed: int) -> torch.Tensor:
            gen = torch.Generator(device="cuda").manual_seed(seed * 131 + rank)
            raw = torch.randint(-7, 8, (sites, rows, width), generator=gen, device="cuda")
            return raw.to(torch.bfloat16).div_(8.0)

        before = dict(SEEN)
        own.copy_(payload(RANK, 70000 + rows))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            body()
        torch.cuda.current_stream().wait_stream(side)
        _mpi_sync()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            body()
        pool = graph.pool() if pool is None else pool
        for seed in (71000 + rows, 72000 + rows):
            own.copy_(payload(RANK, seed))
            _mpi_sync()
            graph.replay()
            torch.cuda.synchronize()
            everyone = [payload(r, seed) for r in range(WORLD)]
            ref = torch.stack(
                [entry.allgather.reference([p[s] for p in everyone]) for s in range(sites)]
            )
            assert not bool(_differs(gathered, ref)), f"rows={rows}: the gather is wrong"
            expected = (own.float() * total).to(torch.bfloat16)
            assert not bool(_differs(back, expected)), f"rows={rows}: the round trip is wrong"
            _mpi_sync()
        _account(by_rows[rows], before)
        del graph
    _mpi_sync()


# ---- the q-split selection gather (the cells stating `q_split`) ---------------
#
# Mistral Large 4 NVFP4 tep4 and tp4 (four ranks of one node) and tep8 (eight ranks
# over two nodes) under chunked prefill: once a step's tc context rows reach the
# engine's q_split_threshold, TP rank r of WORLD selects for rows
# [q_begin, q_end) = [tc * r // WORLD, tc * (r + 1) // WORLD) only, and every
# selecting indexer layer gathers the shares -- `topk_buf[q_begin:q_end]`, sizes
# = q_sizes -- into rows [0, tc) of the step's selection buffer, over the nodes
# the ranks span. A selection row holds key positions, so the payloads are int32
# in [-1, 2^20) unless a cell asks for every bit pattern (`bits`) or for rows
# shaped as the top-k writes them (`indices`). Each job drives the q-split cells
# of its own world size: `_q_split` and every share below follow WORLD.

#: The engine's default q_split_threshold: the smallest tc it splits.
Q_SPLIT_THRESHOLD = 8192
#: A value no selection holds (positions are >= -1, salts stay below 2^30): a
#: row still holding it was never written.
SPLIT_SENTINEL = -(2**31)


def _on_split(cell: Any) -> bool:
    """Whether `cell` is one of the q-split gather's: it states `q_split`."""
    return bool(cell.spec.get("q_split"))


def _split_cells() -> List[Tuple[int, Any]]:
    """(index, cell) of every q-split cell this job drives, all on the MPI-session symbol."""
    return [(i, c) for i, c in _mpi_cells() if _on_split(c)]


def _only_split_cells() -> bool:
    """Whether this job's cells are all the q-split's (then it has no gather sweep or graph)."""
    return all(_on_split(c) for _, c in _mpi_cells())


def _q_split(tc: int, rank: int) -> Tuple[int, int, List[int]]:
    """`rank`'s (q_begin, q_end) of `tc` context rows and the q_sizes, as the engine splits."""
    sizes = [(r + 1) * tc // WORLD - r * tc // WORLD for r in range(WORLD)]
    return tc * rank // WORLD, tc * (rank + 1) // WORLD, sizes


def _split_width(spec: Dict[str, Any]) -> int:
    assert len(spec["trailing"]) == 1 and spec["dtype"] is torch.int32, spec
    assert spec["world_size"] == WORLD and spec["process_group"] is None, (spec, WORLD)
    # A cell without `nodes` is one node's, as the tep4 and tp4 cells are.
    assert spec.get("nodes", 1) == NODES, (spec, NODES)
    return spec["trailing"][0]


def _split_tokens(spec: Dict[str, Any]) -> List[int]:
    """The tc values a q-split cell names, largest first, checked against its keys.

    A sweep names its totals as `sweep_tokens` or `sweep_totals`; its sizes are
    the floor split at each (`sizes="floor_split"`, or no `sizes`).
    """
    if "sweep_tokens" in spec or "sweep_totals" in spec:
        lo, hi = spec["sweep_tokens"] if "sweep_tokens" in spec else spec["sweep_totals"]
        tcs = set(range(lo, hi + 1, spec["sweep_step"])) | set(spec.get("sweep_also", ())) | {hi}
        assert all(lo <= tc <= hi for tc in tcs), spec
        assert spec.get("sizes", "floor_split") == "floor_split", spec
    else:
        tcs = {spec["tokens"]}
        assert list(spec["sizes"]) == _q_split(spec["tokens"], RANK)[2], spec
    assert min(tcs) >= Q_SPLIT_THRESHOLD, spec
    assert spec["rows"] == max(max(_q_split(tc, RANK)[2]) for tc in tcs), spec
    return sorted(tcs, reverse=True)


def _selections(tc: int, seed: int, width: int, bits: bool = False) -> torch.Tensor:
    """A step's [tc, width] selection rows, the same on every rank for one seed.

    Rank r's input is its rows [q_begin, q_end) of these, so the gather is
    supposed to give back exactly this tensor. Key positions in [-1, 2^20): two
    rows agree in about one element in a million, so a block from the wrong rank
    or at the wrong offset changes nearly every element it covers. With `bits`,
    every int32 bit pattern equally likely.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed * 7919 + 3)
    if bits:
        shape = (tc, width)
        raw = torch.randint(-(2**31), 2**31, shape, generator=gen, device="cuda", dtype=torch.int64)
        return raw.to(torch.int32)
    return torch.randint(-1, 2**20, (tc, width), generator=gen, device="cuda", dtype=torch.int32)


def _shares(rows: torch.Tensor, tc: int) -> List[torch.Tensor]:
    """Every rank's share of a step's `tc` selection rows, in rank order."""
    return [rows[_q_split(tc, r)[0] : _q_split(tc, r)[1]] for r in range(WORLD)]


def _salted(shares: List[torch.Tensor], salt: int) -> List[torch.Tensor]:
    return [torch.bitwise_xor(share, salt) for share in shares]


def _words_differ(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Whether two int32 tensors differ in any bit: a device bool, compared as int64 words."""
    assert a.shape == b.shape and a.dtype == b.dtype == torch.int32, (a.shape, b.shape, a.dtype)
    return torch.ne(a.reshape(-1).view(torch.int64), b.reshape(-1).view(torch.int64)).any()


def _int_fraction_moved(a: torch.Tensor, b: torch.Tensor) -> float:
    """The fraction of elements that differ between two int32 tensors."""
    return torch.ne(a, b).float().mean().item()


def _split_gather(x: torch.Tensor, sizes: List[int]) -> torch.Tensor:
    """The call the tep4, tp4 and tep8 targets make: the share, q_sizes, the TP group, MPI."""
    return entry.allgather(x, sizes, GROUP, process_group=None)


def check_mpi_split_sizes_are_the_engines() -> None:
    """Every q-split cell's tc splits as the engine splits it, by the engine's own function.

    `_split_prefill_queries` (the DSA indexer's) at the default threshold, on
    this rank's Mappings at tp WORLD, each with attention DP off and the job's
    GROUP as its TP group: the default one and the one with the MoE
    tensor-parallel over the same ranks (tep4's and tp4's at four ranks), and
    the one with moe_ep over every rank and moe_tp 1 (tep8's at eight ranks, as
    the tep8 target builds it). Unsplit below 8192, and at every tc a cell names
    the (q_begin, q_end, q_sizes) `_q_split` gives. So the cells' shares are the
    targets', and a change to the engine's split fails here first.
    """
    if not _split_cells():
        return
    from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import _split_prefill_queries
    from tensorrt_llm.mapping import Mapping

    checked = 0
    for mapping in (
        Mapping(world_size=WORLD, rank=RANK, tp_size=WORLD),
        Mapping(world_size=WORLD, rank=RANK, tp_size=WORLD, moe_tp_size=WORLD, moe_ep_size=1),
        Mapping(world_size=WORLD, rank=RANK, tp_size=WORLD, moe_ep_size=WORLD, moe_tp_size=1),
    ):
        assert mapping.tp_rank == RANK and not mapping.enable_attention_dp, mapping.tp_rank
        assert list(mapping.tp_group) == GROUP, (list(mapping.tp_group), GROUP)
        for tc in (1, Q_SPLIT_THRESHOLD - 1):
            assert _split_prefill_queries(tc, mapping, Q_SPLIT_THRESHOLD) == (0, tc, None), tc
        for _, cell in _split_cells():
            for tc in _split_tokens(cell.spec):
                got = _split_prefill_queries(tc, mapping, Q_SPLIT_THRESHOLD)
                assert tuple(got) == _q_split(tc, RANK), (cell.why[:44], tc, got)
                checked += 1
    assert checked > 0, "no q-split cell"
    _mpi_sync()


def check_mpi_split_cells() -> None:
    """Every q-split cell at one tc: the target's call on its share, bitwise the reference.

    Each rank stages its share of a step's selection rows at [q_begin, q_end) of
    a sentinel-filled [tc, width] buffer and passes that view, as the target
    passes `topk_buf[q_begin:q_end]`. Three calls on three payloads (a `bits`
    cell: four calls on one payload of every int32 bit pattern), each the
    reference's concatenation of the shares bit for bit, in a fresh tensor
    outside the buffer, with the buffer left as staged.
    """
    for i, cell in _split_cells():
        spec = cell.spec
        if "tokens" not in spec or "sites" in spec:
            continue
        width = _split_width(spec)
        (tc,) = _split_tokens(spec)
        begin, end, sizes = _q_split(tc, RANK)
        bits = spec.get("payload") == "bits"
        seeds = [60000 + 13 * i] * 4 if bits else [60000 + 13 * i + k for k in range(3)]
        before = dict(SEEN)
        outs = []
        rows = None
        for k, seed in enumerate(seeds):
            prev, rows = rows, _selections(tc, seed, width, bits)
            buf = torch.full((tc, width), SPLIT_SENTINEL, dtype=torch.int32, device="cuda")
            buf[begin:end] = rows[begin:end]
            kept = buf.clone()
            _mpi_sync()
            out = _split_gather(buf[begin:end], sizes)
            where = f"{cell.why[:44]}: call {k}"
            assert out.shape == (tc, width) and out.is_contiguous(), (where, out.shape)
            ref = entry.allgather.reference(_shares(rows, tc))
            assert not bool(_words_differ(out, ref)), f"{where} differs from the reference"
            assert not bool(_words_differ(buf, kept)), f"{where} wrote its input buffer"
            span = (buf.data_ptr(), buf.data_ptr() + buf.numel() * buf.element_size())
            assert not span[0] <= out.data_ptr() < span[1], f"{where}: the result is in the buffer"
            outs.append(out)
        assert len({out.data_ptr() for out in outs}) == len(outs), cell.why[:44]
        if bits:
            assert bool(((rows < -1) | (rows >= 2**20)).any()), (
                "the bits payload holds no wide value"
            )
        if tc == Q_SPLIT_THRESHOLD + 1:
            _split_gate_discriminates(rows, prev, tc, out)
        _account(i, before)
        del outs, rows, prev, buf, kept
        _mpi_sync()


def _split_gate_discriminates(
    rows: torch.Tensor, stale: torch.Tensor, tc: int, out: torch.Tensor
) -> None:
    """The bitwise gate, on a ragged split's payload, rejects every plausible wrong gather.

    Shares in reversed rank order, the last rank's share never arriving (left
    zero), every share one row late, and the previous call's rows (a call that
    did not happen). Each has the right shape; each moves nearly every element
    of the blocks it gets wrong, at most 1/WORLD of the result when it gets one
    block wrong.
    """
    shares = _shares(rows, tc)
    late = []
    for r in range(WORLD):
        begin, end, _ = _q_split(tc, r)
        late.append(torch.cat([rows[begin + 1 : end], rows[end % tc : end % tc + 1]]))
    missing = shares[:-1] + [torch.zeros_like(shares[-1])]
    fractions = []
    for name, wrong, floor in (
        ("shares in reversed rank order", torch.cat(shares[::-1]), 0.9),
        (f"rank {WORLD - 1}'s share never arrived", torch.cat(missing), 0.9 / WORLD),
        ("every share one row late", torch.cat(late), 0.9),
        ("the previous call's rows", stale, 0.9),
    ):
        assert wrong.shape == out.shape, (name, wrong.shape, out.shape)
        fraction = _int_fraction_moved(out, wrong)
        assert fraction > floor, f"{name}: only {fraction:.4f} of elements differ"
        fractions.append(round(fraction, 4))
    everyone = MPI_COMM.allgather(fractions)
    if RANK == 0:
        print(
            f"[mpi] q-split gate at tc {tc}: wrong gathers move (reversed, missing, late, "
            f"stale) {everyone[0]} of the elements on rank 0, min over ranks "
            f"{[min(f[k] for f in everyone) for k in range(4)]}",
            flush=True,
        )


def check_mpi_split_sweep_cells() -> None:
    """Every q-split sweep cell: at each tc it names, the target's call, bitwise the reference.

    One [top, width] payload regenerated on every rank, and at each tc a salt
    XORed into it -- a different one per call, below 2^30 so no selection becomes
    the sentinel -- so a call that did not happen, leaving a block the caching
    allocator hands back with an earlier call's rows, differs in every element.
    Rank r stages its share of the salted rows at [q_begin, q_end) of a buffer
    and passes that view. Verdicts stay on the device and are read once; a
    mismatch is agreed across the ranks before any rank raises.
    """
    for i, cell in _split_cells():
        spec = cell.spec
        if "sweep_tokens" not in spec:
            continue
        width = _split_width(spec)
        tcs = _split_tokens(spec)
        top = tcs[0]
        base = _selections(top, 61000 + 13 * i, width)
        buf = torch.empty(top, width, dtype=torch.int32, device="cuda")
        bad = torch.zeros(len(tcs), dtype=torch.bool, device="cuda")
        before = dict(SEEN)
        _mpi_sync()
        for k, tc in enumerate(tcs):
            begin, end, sizes = _q_split(tc, RANK)
            salt = (k + 1) * 40503 % 2**30
            torch.bitwise_xor(base[begin:end], salt, out=buf[begin:end])
            out = _split_gather(buf[begin:end], sizes)
            assert out.shape == (tc, width), (cell.why[:44], tc, out.shape)
            ref = entry.allgather.reference(_salted(_shares(base[:tc], tc), salt))
            bad[k] = _words_differ(out, ref)
        torch.cuda.synchronize()
        everyone = MPI_COMM.allgather(bad.nonzero().flatten().tolist())
        failing = sorted({tcs[k] for got in everyone for k in got})
        if failing:
            raise AssertionError(
                f"{cell.why[:44]}: {len(failing)} of {len(tcs)} tc differ from the reference "
                f"on some rank, the smallest {failing[0]}"
            )
        if RANK == 0:
            print(
                f"[mpi] q-split sweep: {len(tcs)} tc {tcs[-1]}..{tcs[0]}, each bitwise", flush=True
            )
        _account(i, before)
        del base, buf, bad
        _mpi_sync()


def check_mpi_split_step_cells() -> None:
    """Every q-split step cell: `sites` gathers in layer order, as the target issues them.

    One selection buffer per rank, as the target keeps one per step. Per layer
    its rows [0, tc) are set to the sentinel, the rank's share of that layer's
    selections is written at [q_begin, q_end), and the view is gathered with the
    result copied into rows [0, tc) -- the target's `topk_buf[:tc].copy_(
    allgather(topk_buf[q_begin:q_end], q_sizes, ...))` -- then the rows a layer
    reads are kept. Nothing synchronizes between layers. Afterwards every layer's
    rows are checked against the reference bitwise, and for the sentinel: a row
    the gather did not write would still hold it.
    """
    for i, cell in _split_cells():
        spec = cell.spec
        if "sites" not in spec or spec.get("payload") == "indices":
            continue  # an `indices` step runs in `check_mpi_split_copy_back_cells`
        width, sites = _split_width(spec), spec["sites"]
        (tc,) = _split_tokens(spec)
        begin, end, sizes = _q_split(tc, RANK)
        base = _selections(tc, 62000 + 13 * i, width)
        salts = [(997 * i + layer + 1) * 40503 % 2**30 for layer in range(sites)]
        topk = torch.empty(tc, width, dtype=torch.int32, device="cuda")
        read = torch.empty(sites, tc, width, dtype=torch.int32, device="cuda")
        before = dict(SEEN)
        _mpi_sync()
        for layer, salt in enumerate(salts):
            topk.fill_(SPLIT_SENTINEL)
            torch.bitwise_xor(base[begin:end], salt, out=topk[begin:end])
            topk[:tc].copy_(_split_gather(topk[begin:end], sizes))
            read[layer].copy_(topk)
        bad = torch.zeros(sites, dtype=torch.bool, device="cuda")
        holes = torch.zeros(sites, dtype=torch.bool, device="cuda")
        for layer, salt in enumerate(salts):
            ref = entry.allgather.reference(_salted(_shares(base, tc), salt))
            bad[layer] = _words_differ(read[layer], ref)
            holes[layer] = torch.eq(read[layer], SPLIT_SENTINEL).any()
        torch.cuda.synchronize()
        everyone = MPI_COMM.allgather(
            (bad.nonzero().flatten().tolist(), holes.nonzero().flatten().tolist())
        )
        wrong = sorted({layer for got, _ in everyone for layer in got})
        unwritten = sorted({layer for _, got in everyone for layer in got})
        if wrong or unwritten:
            raise AssertionError(
                f"{cell.why[:44]}: layers {wrong} differ from the reference and layers "
                f"{unwritten} hold unwritten rows, on some rank"
            )
        _account(i, before)
        del read, topk, base, bad, holes
        _mpi_sync()


def _indices(tc: int, seed: int, width: int, top: int) -> torch.Tensor:
    """A step's [tc, width] selection rows as the top-k leaves them, the same on every rank.

    Row j holds key positions in [0, top] in its first `seen` slots and -1 in
    the rest: a query that sees fewer than `width` keys -- early in a request,
    or in a chunk with no key at all -- leaves a -1 tail. Every third row sees
    `width` keys or more and holds both 0 and `top`, every 97th from row 1 sees
    none, and the others see 0..width keys, uniformly. Two rows' positions agree
    in about one slot in a million, so a block from the wrong rank or at the
    wrong offset changes every slot where either holds a position.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed * 7919 + 5)
    rows = torch.randint(0, top + 1, (tc, width), generator=gen, device="cuda", dtype=torch.int32)
    seen = torch.randint(0, width + 1, (tc, 1), generator=gen, device="cuda", dtype=torch.int32)
    seen[::3] = width
    seen[1::97] = 0
    rows[::3, 0] = 0
    rows[::3, -1] = top
    slots = torch.arange(width, device="cuda", dtype=torch.int32)
    return rows.masked_fill_(slots >= seen, -1)


def _indices_payload_and_gate(rows: torch.Tensor, other: torch.Tensor, tc: int, top: int) -> None:
    """An `indices` payload holds what it claims, and the bitwise gate sees a wrong gather on it.

    The payload: positions 0 and `top` present, nothing outside [-1, top], rows
    with -1 tails, rows of -1 only and rows without a tail. The gate: shares in
    reversed rank order, every share one row late, another call's rows at this
    tc, and the last rank's share never arriving (left zero) each move more than
    half the slots of the blocks they get wrong (1/WORLD of the result for one
    block); the rows that agree are the -1 tails both sides share.
    """
    assert int(rows.min()) == -1 and int(rows.max()) == top, (int(rows.min()), int(rows.max()))
    assert bool((rows == 0).any()), "no position 0"
    tails = rows[:, -1] == -1
    empty = rows[:, 0] == -1
    counts = [int((~tails).sum()), int((tails & ~empty).sum()), int(empty.sum())]
    assert min(counts) > 0, f"(full, tailed, empty) rows {counts}"
    shares = _shares(rows, tc)
    late = []
    for r in range(WORLD):
        begin, end, _ = _q_split(tc, r)
        late.append(torch.cat([rows[begin + 1 : end], rows[end % tc : end % tc + 1]]))
    fractions = []
    for name, wrong, floor in (
        ("shares in reversed rank order", torch.cat(shares[::-1]), 0.5),
        ("every share one row late", torch.cat(late), 0.5),
        ("another call's rows", other, 0.5),
        (
            f"rank {WORLD - 1}'s share never arrived",
            torch.cat(shares[:-1] + [torch.zeros_like(shares[-1])]),
            0.5 / WORLD,
        ),
    ):
        assert wrong.shape == rows.shape, (name, wrong.shape, rows.shape)
        fraction = _int_fraction_moved(rows, wrong)
        assert fraction > floor, f"{name}: only {fraction:.4f} of slots differ"
        fractions.append(round(fraction, 4))
    everyone = MPI_COMM.allgather(fractions)
    if RANK == 0:
        print(
            f"[mpi] indices payload at tc {tc}: (full, tailed, empty) rows {counts}, positions "
            f"-1..{top}; wrong gathers move (reversed, late, other call, missing) "
            f"{everyone[0]} of the slots on rank 0, min over ranks "
            f"{[min(f[k] for f in everyone) for k in range(4)]}",
            flush=True,
        )


def check_mpi_split_copy_back_cells() -> None:
    """Every `payload="indices"` q-split cell, through the target's copy-back.

    The tp4 target's own form of the call, on rows shaped as its top-k leaves
    them (`_indices`). One selection buffer per rank, as the target keeps one
    per step: per call its rows [0, tc) are set to the sentinel, the rank's
    share of a fresh payload is written at [q_begin, q_end), and the view is
    gathered with the result copied into rows [0, tc) -- `topk_buf[:tc].copy_(
    allgather(topk_buf[q_begin:q_end], q_sizes, ...))`. A `sites` cell makes
    that many calls at its tc in layer order with nothing synchronizing between
    them, keeps the rows each layer reads and checks them afterwards; a sweep
    cell makes one call per tc, largest first, each checked as it goes. Every
    call's rows must be bitwise the reference's concatenation of the shares and
    hold no sentinel. Verdicts stay on the device and are read once; a mismatch
    is agreed across the ranks before any rank raises.
    """
    for i, cell in _split_cells():
        spec = cell.spec
        if spec.get("payload") != "indices":
            continue
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        width, top = _split_width(spec), spec["top_position"]
        tcs = _split_tokens(spec)
        sites = spec.get("sites", 1)
        calls = [(tc, layer) for tc in tcs for layer in range(sites)]
        seed = (64000 + 13 * i) * 4096
        topk = torch.empty(tcs[0], width, dtype=torch.int32, device="cuda")
        read = None
        if sites > 1:
            read = torch.empty(sites, tcs[0], width, dtype=torch.int32, device="cuda")
        bad = torch.zeros(len(calls), dtype=torch.bool, device="cuda")
        holes = torch.zeros(len(calls), dtype=torch.bool, device="cuda")
        _indices_payload_and_gate(
            _indices(tcs[0], seed, width, top), _indices(tcs[0], seed - 1, width, top), tcs[0], top
        )
        # The control: with the gather withheld, the other ranks' rows still hold the sentinel.
        begin, end, _ = _q_split(tcs[0], RANK)
        topk.fill_(SPLIT_SENTINEL)
        topk[begin:end].copy_(_indices(tcs[0], seed - 1, width, top)[begin:end])
        assert bool(torch.eq(topk, SPLIT_SENTINEL).any()), "a withheld gather left no sentinel"
        before = dict(SEEN)
        _mpi_sync()
        for k, (tc, layer) in enumerate(calls):
            begin, end, sizes = _q_split(tc, RANK)
            rows = _indices(tc, seed + k, width, top)
            topk[:tc].fill_(SPLIT_SENTINEL)
            topk[begin:end].copy_(rows[begin:end])
            topk[:tc].copy_(_split_gather(topk[begin:end], sizes))
            if read is not None:
                read[layer, :tc].copy_(topk[:tc])
                continue
            bad[k] = _words_differ(topk[:tc], entry.allgather.reference(_shares(rows, tc)))
            holes[k] = torch.eq(topk[:tc], SPLIT_SENTINEL).any()
        if read is not None:
            for k, (tc, layer) in enumerate(calls):
                ref = entry.allgather.reference(_shares(_indices(tc, seed + k, width, top), tc))
                bad[k] = _words_differ(read[layer, :tc], ref)
                holes[k] = torch.eq(read[layer, :tc], SPLIT_SENTINEL).any()
        torch.cuda.synchronize()
        everyone = MPI_COMM.allgather(
            (bad.nonzero().flatten().tolist(), holes.nonzero().flatten().tolist())
        )
        wrong = sorted({calls[k] for got, _ in everyone for k in got})
        unwritten = sorted({calls[k] for _, got in everyone for k in got})
        if wrong or unwritten:
            raise AssertionError(
                f"{cell.why[:44]}: {len(wrong)} of {len(calls)} (tc, layer) calls differ from "
                f"the reference, the first {wrong[:4]}, and {len(unwritten)} hold unwritten "
                f"rows, the first {unwritten[:4]}, on some rank"
            )
        if RANK == 0:
            print(
                f"[mpi] {cell.why[:44]}: {len(calls)} copy-back calls at tc "
                f"{tcs[-1]}..{tcs[0]}, each bitwise, none unwritten; "
                f"{time.monotonic() - started:.2f}s, peak allocated "
                f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                flush=True,
            )
        _account(i, before)
        del topk, read, bad, holes
        _mpi_sync()


def check_mpi_split_guards_a_share_that_is_not_its_sizes_entry() -> None:
    """`sizes[i]` must be the i-th rank's input height, and the op never checks it.

    Run in a job that holds q-split cells, on its world. With `per` = 8192 //
    WORLD rows (2048 at four ranks, 1024 at eight), unguarded -- the op itself,
    at tc 8193's q_sizes [per, ..., per, per + 1] ([2048, 2048, 2048, 2049] at
    four ranks): the last rank's input one row short of its entry, a leading
    view of a buffer whose next row is a marker row, gives a result whose last
    row is the marker, a row no rank passed; rank 0's input one row long, its
    share then the marker, gives the shares' concatenation with the marker
    nowhere in it. With every entry per + 1 and every input `per` rows (equal
    entries: one ncclAllGather of `input.numel()`), the blocks land at
    `per`-row offsets rather than where `sizes` puts them, and the result's
    last WORLD rows are never written (the caching allocator's previous block
    there, sentinel-filled, shows through when it is reused). Every rank gets
    the same wrong result, and nothing raises.

    Guarded: with every rank's input one row short of its entry, and with every
    entry per + 1 over `per`-row inputs, the entry's `is_valid` refuses the call
    on every rank, so no rank enters the collective. Each refusal is first asked
    of `is_valid` alone and agreed across the ranks, so a rank it admitted
    cannot sit in the collective alone.
    """
    if not _split_cells():
        return
    width, tc = 2048, Q_SPLIT_THRESHOLD + 1
    per, last = Q_SPLIT_THRESHOLD // WORLD, WORLD - 1
    rows = _selections(tc, 63000, width)
    begin, end, sizes = _q_split(tc, RANK)
    assert sizes == [per] * last + [per + 1], sizes
    marker = torch.full((1, width), 0x5A5A5A5A, dtype=torch.int32, device="cuda")
    op = torch.ops.trtllm.allgather

    # One row short on the last rank: the op reads the row after its input.
    share = rows[begin : end - 1] if RANK == last else rows[begin:end]
    x = torch.cat([share, marker])[: share.shape[0]]
    _mpi_sync()
    out = op(x, sizes, GROUP)
    torch.cuda.synchronize()
    assert out.shape == (tc, width), out.shape
    assert not bool(_words_differ(out[: tc - 1], rows[: tc - 1])), "short: the passed rows moved"
    assert bool((out[tc - 1] == marker[0]).all()), "short: the result's last row is not the marker"

    # One row long on rank 0: its last row is dropped.
    x = torch.cat([rows[begin:end], marker]) if RANK == 0 else rows[begin:end].clone()
    _mpi_sync()
    out = op(x, sizes, GROUP)
    torch.cuda.synchronize()
    assert not bool(_words_differ(out, rows)), "long: the result is not the shares"
    assert not bool((out == marker[0]).all(dim=1).any()), "long: the extra row arrived"

    # Every entry per + 1, every input `per` rows: one ncclAllGather of input.numel().
    equal = [per + 1] * WORLD
    x = rows[per * RANK : per * RANK + per].clone()
    bait = torch.full((sum(equal), width), SPLIT_SENTINEL, dtype=torch.int32, device="cuda")
    bait_at = bait.data_ptr()
    del bait
    _mpi_sync()
    out = op(x, equal, GROUP)
    torch.cuda.synchronize()
    assert out.shape == (sum(equal), width), out.shape
    packed = WORLD * per
    assert not bool(_words_differ(out[:packed], rows[:packed])), "equal: not packed"
    assert bool(_words_differ(out[per + 1 : 2 * per + 1], rows[per : 2 * per])), (
        "equal: rank 1 placed"
    )
    reused = out.data_ptr() == bait_at
    if reused:
        assert bool((out[packed:] == SPLIT_SENTINEL).all()), "equal: the tail was written"
    if RANK == 0:
        print(
            f"[mpi] q-split heights at world {WORLD}: short -> the row after the input, long -> "
            f"cut, equal entries -> packed at {per}-row offsets (tail unwritten, sentinel block "
            f"reused: {reused})",
            flush=True,
        )
    _mpi_sync()

    for name, x, sizes_ in (
        ("every input one row short", rows[begin : end - 1].clone(), sizes),
        (
            f"every entry {per + 1} over {per}-row inputs",
            rows[per * RANK : per * RANK + per].clone(),
            equal,
        ),
    ):
        try:
            entry.allgather.is_valid(x, sizes_, GROUP, process_group=None)
        except AssertionError:
            refused = True
        else:
            refused = False
        everyone = MPI_COMM.allgather(refused)
        assert all(everyone), f"{name}: is_valid admitted it on ranks {everyone}"
        try:
            _split_gather(x, sizes_)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"{name}: the entry made the call")
        _mpi_sync()


def check_mpi_every_cell_was_driven() -> None:
    """Each MPI-session cell ran through the entry's own `__call__` and `reference`."""
    for i, cell in _mpi_cells():
        calls, refs = DRIVEN.get(i, [0, 0])
        assert calls > 0 and refs > 0, (
            f"cell {cell.why!r}: the entry's __call__ / reference ran {calls} / {refs} times"
        )


MPI_CHECKS = (
    check_mpi_session_is_the_launchers_world,
    # Before any capture: the group's first call builds the communicator.
    check_mpi_eager_sweep_cells,
    check_mpi_the_gate_discriminates_a_wrong_gather,
    check_mpi_bits_payload_cells,
    # The q-split gather's cells (tep4 and tp4 at four ranks, tep8 at eight):
    # context steps, so eager only.
    check_mpi_split_sizes_are_the_engines,
    check_mpi_split_cells,
    check_mpi_split_sweep_cells,
    check_mpi_split_step_cells,
    check_mpi_split_copy_back_cells,
    check_mpi_split_guards_a_share_that_is_not_its_sizes_entry,
    check_mpi_graph_cells,
    check_mpi_round_trip_through_a_combine_in_one_graph,
    check_mpi_every_cell_was_driven,
)


def _run_one_mpi_rank() -> int:
    """Body of one mpirun rank: run every MPI check; a failing rank exits nonzero
    without finalizing, and mpirun then ends its siblings and fails. A rank
    still running at `MPI_DEADLINE_S` -- wedged in a collective a sibling never
    joined -- dumps its stack and exits."""
    global RANK, WORLD, NODES, GROUP, MPI_COMM, entry
    faulthandler.dump_traceback_later(MPI_DEADLINE_S, exit=True)
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    world = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]))

    from mpi4py import MPI

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import allgather as module
    from tensorrt_llm._utils import mpi_comm

    entry = module
    MPI_COMM = mpi_comm()
    RANK, WORLD = rank, world
    NODES = len(set(MPI_COMM.allgather(socket.gethostname())))
    GROUP = list(range(WORLD))
    assert WORLD >= 2, f"a collective needs at least 2 ranks, got {WORLD}"
    _count_entry_calls()

    for check in MPI_CHECKS:
        started = time.monotonic()
        try:
            check()
        except BaseException:
            import traceback

            print(f"[mpi rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(13)
        if RANK == 0:
            print(f"[mpi] {check.__name__}: {time.monotonic() - started:.1f}s", flush=True)
    _mpi_sync()
    print(
        f"[mpi rank {RANK}] {len(MPI_CHECKS)} checks passed over {len(_mpi_cells())} cells at "
        f"world {WORLD} on {NODES} node(s); peak allocated "
        f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, reserved "
        f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB",
        flush=True,
    )
    sys.stdout.flush()
    faulthandler.cancel_dump_traceback_later()
    # mpirun fails a rank that exits without finalizing. The communicators are
    # left as they are, as in the `_pg` job: these processes end here.
    MPI.Finalize()
    os._exit(0)


def _session_members(sid: int) -> List[int]:
    """Every process in session `sid`, read from /proc."""
    members = []
    for entry_dir in Path("/proc").iterdir():
        if not entry_dir.name.isdigit():
            continue
        try:
            stat = (entry_dir / "stat").read_text()
        except OSError:
            continue
        # pid (comm) state ppid pgrp session ...; comm may hold spaces.
        fields = stat[stat.rindex(")") + 2 :].split()
        if int(fields[3]) == sid:
            members.append(int(entry_dir.name))
    return members


def _end_mpi_job(job: Any) -> None:
    """End mpirun and every rank it started.

    mpirun runs as a session leader; its ranks stay in that session but each in
    a process group of its own, so killing mpirun's group would leave them
    running. mpirun is asked first, then the session's members are killed.
    """
    if job.poll() is None:
        job.terminate()
        try:
            job.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    for pid in _session_members(job.pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if job.poll() is None:
        job.kill()
    job.wait()


def _mpirun_ranks(world_size: int) -> None:
    """Start the MPI-session job under mpirun, one rank per device, and wait for it.

    The environment loses the launching allocation's PMI/PMIx/Open MPI/Slurm
    variables, so this mpirun starts its own ranks on this node instead of
    joining the allocation's step (where every rank would read rank 0 of one),
    and `--oversubscribe` lets it place them whatever slot count it detects.
    """
    mpirun = shutil.which("mpirun")
    assert mpirun is not None, "the MPI-session cells start their ranks with mpirun; none on PATH"
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_")) and k != "TLLM_DISABLE_MPI"
    }
    command = [mpirun, "-n", str(world_size), "--oversubscribe", "--allow-run-as-root"]
    command += [sys.executable, str(Path(__file__).resolve()), _MPI_WORKER_FLAG]
    print(f"[launcher] starting {world_size} MPI-session ranks under mpirun", flush=True)
    job = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        code = job.wait(timeout=MPI_DEADLINE_S)
    except subprocess.TimeoutExpired:
        _end_mpi_job(job)
        raise AssertionError(
            f"the {world_size}-rank MPI-session run did not finish in {MPI_DEADLINE_S}s (wedged)"
        ) from None
    if code != 0:
        _end_mpi_job(job)
        raise AssertionError(f"the {world_size}-rank MPI-session run failed: mpirun exited {code}")


def _spawn_ranks() -> None:
    """Run the `_pg` job, one worker process per claimed device and no external
    launcher, then the MPI-session job under mpirun.

    The `_pg` ranks rendezvous through torch.distributed's env:// method, so the
    only thing they need is a free port. Each child gets its own session; on the
    first nonzero exit or on the deadline, every session is killed, which is
    what keeps one wedged collective from outliving the run.
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
    env = dict(
        os.environ,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        TLLM_DISABLE_MPI="1",
    )

    # Drop the launching srun step's PMIx/PMI/OMPI namespace from the children:
    # they inherit it and are then registered as participants of that step, so
    # the first one to exit (hard, as a failing rank does) trips srun's pmix
    # errhandler, which SIGKILLs the whole step -- launcher and real error with
    # it -- before anything is reported. These ranks rendezvous through
    # torch.distributed's own env://, not pmix.
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    print(f"[launcher] spawning {world_size} ranks, rendezvous on :{port}", flush=True)
    children = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
            env=dict(env, RANK=str(r)),
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

    deadline = time.monotonic() + PG_DEADLINE_S
    live = set(range(world_size))
    failure = None
    while live and failure is None:
        if time.monotonic() > deadline:
            failure = f"did not finish in {PG_DEADLINE_S}s (wedged)"
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
    _mpirun_ranks(world_size)


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    if _MPI_WORKER_FLAG in sys.argv:
        sys.exit(_run_one_mpi_rank())
    _spawn_ranks()
    print("OK")
