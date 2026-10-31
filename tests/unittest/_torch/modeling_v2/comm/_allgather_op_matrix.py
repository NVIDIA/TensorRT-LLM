# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the allgather catalog entry.

The cells are filtered by the world size the job runs at (a cell without
`world` runs at every size), the CUDA-graph batch sizes are the union of the
default capture list and the qwen4_exp unified target's, a decode graph holds
one gather per MoE layer of the qwen4_exp checkpoint (48, more than the 29 of a
30-layer deepseek-shaped graph), and the served path's checks run a second time
through the MPI-session symbol.

A collective cannot be exercised in one process, so this script is its own
launcher, with one rank per device named in CUDA_VISIBLE_DEVICES and two
transports, run one after the other, each under its own deadline:

  pg   it spawns one worker process per device; the ranks rendezvous through
       torch.distributed and call `allgather_pg`. Every check runs here.
  mpi  it re-executes itself under `mpirun -n <devices>` (the in-tree
       `_lockstep.spawn` launch, plus the scrub below); the ranks are an MPI
       world and call `allgather`, the symbol a served engine calls. The checks
       a served dep4 / dep2 forward exercises run here (`MPI_CHECKS`).

A wedged collective hangs rather than raising — and this op wedges rather than
raising for every rank-argument-disagreement case probed — so the deadline is
what keeps a broken kernel from taking the calling run down with it.

Beyond the lockstep surface (uniform / ragged / dtypes / CUDA graphs) the
file covers what a serving engine adds: the engine's own attention-DP
synchronisation interleaved with the op, the engine's stream switching, and
what disagreeing on call order actually does.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _allgather_op_matrix.py
    CUDA_VISIBLE_DEVICES=0,1 python _allgather_op_matrix.py --transport mpi

Not a pytest module, despite the `check_*` bodies. They are one fixed
sequence inside a single job rather than independent cases: each reads
module-global rank state that only the rank bodies bind, and several assert
on communicator state the previous one left behind. Collected as tests they
would run at world size 1 against unbound globals — so neither this file's
name nor its function names match pytest's collection patterns, which is what
keeps it uncollectable however pytest is pointed at this tree.

The collected entry point is `test_modeling_v2_allgather_op_matrix.py`
beside this file: it starts this job at world 2 and 4 and turns its exit code
into an assertion. Both halves are started by file path: the launcher must not
import `tensorrt_llm` (that calls MPI_Init, and the launcher must carry no
CUDA nor torch.distributed state of its own), and the ranks reach the catalog by
absolute import, so neither needs a package.
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

assert torch.cuda.is_available(), "allgather requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
_MPI_WORKER_FLAG = "--mpi-rank-worker"

#: Each transport's own budget, enforced by this launcher.
PG_DEADLINE_S = 1200
MPI_DEADLINE_S = 900
#: The whole job's budget: both transports, one after the other. `_rank_job`
#: reads it to set its own timeout above this one.
DEADLINE_S = PG_DEADLINE_S + MPI_DEADLINE_S

HIDDEN = 2560  # DeepSeek-V3-Lite's and the qwen4_exp checkpoint's MoE input width

# The row counts one engine's decode graphs are captured at. trtllm's graph
# runner instantiates one graph per configured batch size, and under attention
# data parallelism a decode call's per-rank row count *is* that batch size,
# all alive at once in one memory pool and replayed interleaved as the served
# batch size moves. So the surface a captured gather has to be correct over is
# the whole set plus the transitions between its members, not one
# representative row count. The default list at `max_batch_size = 256` is 35
# graphs (1..32, 64, 128, 256); the qwen4_exp deployment captures 19 at
# `max_batch_size = 128` (1, 2, 4, 8, 16, 24, ..., 120, 128). Both: 45 sizes.
_QWEN4_CAPTURE_SIZES = (1, 2, 4, 8, 16) + tuple(range(24, 129, 8))
GRAPH_BATCH_SIZES: Tuple[int, ...] = tuple(
    sorted(set(range(1, 33)) | {64, 128, 256} | set(_QWEN4_CAPTURE_SIZES))
)

# Gathers inside one decode graph, each preceding one expert-parallel MoE call
# with a gather of the rank's own rows into the full token set. The qwen4_exp
# checkpoint has 48, all its layers being MoE; a 30-layer deepseek-shaped
# checkpoint with `first_k_dense_replace = 1` has 29. One graph holding the
# larger count covers the smaller.
SITES_PER_DECODE_GRAPH = 48

# Bound inside the rank body, never in the launcher: importing the entry
# pulls in tensorrt_llm, which calls MPI_Init at import, and an
# MPI-initialized launcher could not start the mpi transport's mpirun.
allgather: Any = None
entry: Any = None
RANK = 0
WORLD = 1
GROUP: List[int] = [0]
# The engine's own cross-rank synchronisation object, built from the real
# Mapping a serving engine builds under attention data parallelism. Bound in
# the rank body for the same reason as the rest.
HOST_GROUP: Any = None
SUBGROUPS: Dict[Tuple[int, ...], Any] = {}
PG_BOXED: Any = None
# The MPI world, on the mpi transport only; None on the pg transport.
MPI_COMM: Any = None


def _barrier() -> None:
    """Align the ranks: torch.distributed's barrier on the pg transport, the MPI
    world's on the mpi transport (after the device work this rank queued)."""
    if MPI_COMM is None:
        dist.barrier()
    else:
        torch.cuda.synchronize()
        MPI_COMM.Barrier()


def _host_allgather(obj: Any) -> List[Any]:
    """One host-side cross-rank step, the shape of the engine's per-step sync.

    A serving engine agrees the per-rank token counts once per step with a
    host-side collective (`Distributed.tp_allgather`: MPI's allgather under
    mpirun, which is what the mpi transport calls; TorchDist rides
    torch.distributed). On the pg transport a dedicated gloo group stands in
    for the engine object: constructing the engine's own `TorchDist` needs ray
    importable, which a plain development machine does not owe this test.
    """
    if MPI_COMM is not None:
        return list(MPI_COMM.allgather(obj))
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
    _barrier()


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
        _barrier()
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
    _barrier()

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
    _barrier()
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(captured, _gather_ref([rows] * WORLD, seed + 500), "replay after refill")
    del graph
    _barrier()


def check_certified_cells() -> None:
    """Drive the entry's own cell list, inside `validating`.

    The other checks in this file say things no cell can -- CUDA graph
    behaviour, call-order disagreement, which stream the call lands on. This one
    says the plain thing: for every configuration the entry claims to be
    certified over, the gather is the concatenation, and the guard admits the
    input a shipped target passes.

    A cell with a `world` runs at that world size only; one without runs at
    every world size the job is started at. The seed is the cell's position in
    `CELLS`, so a cell draws the same payload at every world and transport.
    """
    cells = [
        (i, cell)
        for i, cell in enumerate(entry.allgather.CELLS)
        if cell.spec.get("world", WORLD) == WORLD
    ]
    assert any("world" in cell.spec for _, cell in cells), (
        f"no cell names world size {WORLD}; no target calls this op at the size this job runs"
    )
    for i, cell in cells:
        spec = cell.spec
        rows, trailing, dtype = spec["rows"], spec["trailing"], spec["dtype"]
        seed = 9000 + 13 * i
        x = _payload(RANK, rows, seed, dtype, trailing)
        with validating(entry.allgather):
            out = allgather(x, spec["sizes"], GROUP)
        assert out.shape == (rows * WORLD, *trailing), (cell.why, out.shape)
        _assert_bitwise(out, _gather_ref([rows] * WORLD, seed, dtype, trailing), cell.why)
        _barrier()
    if RANK == 0:
        driven = [(cell.spec["rows"], cell.spec["trailing"][-1]) for _, cell in cells]
        print(f"[rank 0] world {WORLD}: {len(cells)} cells bitwise, (rows, hidden) {driven}")


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
        _barrier()


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
        _barrier()


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
    _barrier()


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
        _barrier()


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
        _barrier()

        xr = _payload(RANK, sizes[RANK], seed, dtype)
        outr = allgather(xr, sizes, GROUP)
        assert outr.dtype is dtype and outr.shape == (sum(sizes), HIDDEN)
        _assert_bitwise(outr, _gather_ref(sizes, seed, dtype), f"ragged {dtype}")
        _barrier()


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
    _barrier()


def check_group_order_does_not_change_the_output_order() -> None:
    """The result is ordered by ascending rank, whatever order `group` lists."""
    rows, seed = 4, 7200
    x = _payload(RANK, rows, seed)
    out = allgather(x, None, list(reversed(GROUP)))
    _assert_bitwise(out, _gather_ref([rows] * WORLD, seed), "reversed group list")
    _barrier()


def check_cuda_graph_at_every_engine_batch_size() -> None:
    """Captured at all 45 engine batch sizes, then replayed interleaved.

    Capturing one row count proves nothing about an engine, which holds every
    graph in GRAPH_BATCH_SIZES alive at once and replays them in whatever order
    the served batch size takes. Each replay here gets a payload no earlier
    call used, so a graph that silently did not re-run keeps the previous
    answer and fails the bitwise gate.

    Two passes per order: one replay at a time with a synchronize and a full
    check before the next graph runs, then the whole order issued back to back
    with no synchronization in between — 45 differently sized collectives
    queued as one run of work, which is the only pass that can catch a hazard
    that needs the next call to already be in flight.
    """
    inputs, graphs, outputs = _capture_per_batch_size()
    assert len(graphs) == len(GRAPH_BATCH_SIZES) == 45
    seed = 110000
    for name, order in _replay_orders():
        for pos, rows in enumerate(order):
            seed += 1000
            _fresh_payload(inputs, rows, seed)
            graphs[rows].replay()
            torch.cuda.synchronize()
            _verify_replay(outputs, rows, seed, f"{name}/checked@{pos} rows={rows}")
        _barrier()
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
        _barrier()
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
    _barrier()


def check_cuda_graph_one_site_per_moe_layer_in_every_batch_size_graph() -> None:
    """The decode graph this checkpoint captures, at all 45 batch sizes.

    2160 captured gathers (45 graphs x 48 MoE layers), one memory pool. The
    sites are independent — each reads its own buffer and every one gets a
    distinct payload, so a site that returned a neighbour's data would fail the
    bitwise gate. Two replay orders rather than four: the per-order cost is 48x
    the single-site test's and the axis this one adds is site count, not order.
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
        _barrier()
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
        _barrier()
    del graphs, outputs, inputs
    _barrier()


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
        _barrier()
    del graphs, outputs, inputs
    _barrier()


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
        _barrier()

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
            _barrier()
    del graphs, captured, buffers
    _barrier()


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

    The rotated split's floor follows the world size too. Its first row is
    rank 0's first row drawn from the same generator stream at another height,
    which comes out identical (0.9602 at world 4 is 27 of 28 rows differing),
    so the most it can move is 1 - 1/sum(sizes): 5 of 6 rows at world 2. From
    world 4 up that ceiling leaves room for the flat 0.9 floor, which is kept
    there; below it the floor is 0.9 of the ceiling.
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
            0.9 if WORLD >= 4 else 0.9 * (1 - 1 / sum(sizes)),
        ),
    ]
    for name, got, variant, floor in wrong:
        assert variant.shape == got.shape, (name, variant.shape, got.shape)
        diff = (variant.float() - got.float()).abs()
        fraction = (diff != 0).float().mean().item()
        assert fraction > floor, f"{name}: only {fraction:.4f} of elements differ"
        assert diff.max().item() > 8.0, f"{name}: max difference {diff.max().item()}"
    _barrier()


def check_the_engines_own_cross_rank_step_is_not_on_this_communicator() -> None:
    """What a serving engine synchronises with, next to what this op uses.

    Under attention data parallelism the engine agrees the per-rank token
    counts once per step with a **host-side** collective, not with the device
    collective this op runs. The two never share an ordering; see
    `_host_allgather` for what stands in for the engine object here.
    """
    counts = [3 + 2 * r for r in range(WORLD)]
    assert list(_host_allgather(counts[RANK])) == counts
    _barrier()


def check_interleaved_with_the_engines_attention_dp_synchronisation() -> None:
    """The op inside a forward, between the engine's own cross-rank steps.

    The shape a served forward has: once per step the engine agrees the
    per-rank token counts on the host (the previous test's collective, which is
    where `attn_metadata.all_rank_num_tokens` comes from), then the model
    issues one gather per expert-parallel layer on rows padded to
    `max(all_rank_num_tokens)`.

    Nothing synchronises here. Every step's payloads and gathers, and the host
    collective between steps, are issued back to back for all 8 steps before a
    single result is read — 384 gathers per rank in flight, which is what makes
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
    _barrier()


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
        _barrier()


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
    _barrier()


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
    _barrier()


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
    _barrier()


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
    other rank exactly 1/4 — the disagreeing rank's block. At world size 8 the
    same split is 7/8 and 1/8. The floors are those fractions less 0.05 (0.7
    and 0.2 at world 4). Nothing raised and nothing hung, in 20 probe
    iterations out of 20 and in every run of this test.

    Kept last, and it puts the communicator back: a swapped *pair* leaves the
    per-communicator positions aligned once both calls are made, so the final
    assertion is a plain gather coming back bitwise correct.
    """
    rows, seed = 12, 160000
    first, second = _payload(RANK, rows, seed), _payload(RANK, rows, seed + 500)
    ref_first = _gather_ref([rows] * WORLD, seed)
    ref_second = _gather_ref([rows] * WORLD, seed + 500)
    _barrier()

    if RANK == 0:
        got_second = allgather(second, None, GROUP)
        got_first = allgather(first, None, GROUP)
    else:
        got_first = allgather(first, None, GROUP)
        got_second = allgather(second, None, GROUP)
    torch.cuda.synchronize()

    # The swapping rank pairs with every other rank wrongly; the others pair
    # wrongly only with the swapping rank's block.
    floor = ((WORLD - 1) / WORLD if RANK == 0 else 1 / WORLD) - 0.05
    for name, got, ref in (
        ("first", got_first, ref_first),
        ("second", got_second, ref_second),
    ):
        assert got.shape == ref.shape, (name, got.shape, ref.shape)
        wrong = (got != ref).float().mean().item()
        assert wrong > floor, f"{name}: only {wrong:.4f} of elements differ"
    _barrier()

    x = _payload(RANK, rows, seed + 900)
    _assert_bitwise(
        allgather(x, None, GROUP),
        _gather_ref([rows] * WORLD, seed + 900),
        "gather after an order disagreement",
    )
    _barrier()


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

# What a served dep4 / dep2 forward does with the MPI-session symbol: the
# cells, uniform and ragged gathers, a fresh output, CUDA graphs at every
# engine batch size with one gather per MoE layer, eager calls between
# replays, the engine's host step between gathers, and the stream moving
# under the call. The rest of `CHECKS` concern the wrapper's guards, the
# reference's discrimination or the pg symbol's sub-groups, none of which
# depends on the symbol a served engine reaches.
MPI_CHECKS = (
    # Stays first: the MPI symbol builds its group's communicator on the
    # group's first call, which cannot happen inside a capture, so the first
    # call is an eager one.
    check_certified_cells,
    check_uniform_gather,
    check_ragged_gather,
    check_output_is_fresh_and_input_is_untouched,
    check_cuda_graph_at_every_engine_batch_size,
    check_cuda_graph_one_site_per_moe_layer_in_every_batch_size_graph,
    check_cuda_graph_replays_survive_eager_calls_of_other_shapes,
    check_interleaved_with_the_engines_attention_dp_synchronisation,
    check_the_stream_the_call_lands_on_is_not_part_of_the_match,
)


def _bind_entry(rank: int, world: int, process_group: Optional[object]) -> None:
    """Bind the rank state every check reads: the entry, its call surface and
    the guard harness."""
    global RANK, WORLD, GROUP, allgather, validating, entry
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import allgather as entry

    # One partial instead of a per-call argument: every check's call site keeps
    # the certified shape, and the wrapper hands `process_group` to the symbol
    # it selects -- the same routing the product does in
    # _torch/distributed/ops.py: a boxed group runs the _pg symbol, None the
    # MPI-session one. The partial is only the call surface: the guard is
    # armed on the entry itself, and `reference` and `CELLS` are read off
    # `entry.allgather` directly.
    allgather = functools.partial(entry.allgather, process_group=process_group)

    # This file is run as a spawned worker process, not collected by pytest,
    # so the repo's `__extra_import_path__` mechanism is not in play and the
    # sibling helper has to be reached the plain way. The mutation is confined
    # to these rank processes; the pytest process that launched them never sees
    # it, which is what its own sys.path check is there to protect.
    # comm -> modeling_v2, where _validating is.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _validating import validating as _validating_cm

    validating = _validating_cm
    RANK = rank
    WORLD = world
    GROUP = list(range(WORLD))
    assert WORLD >= 2, f"a collective needs at least 2 ranks, got {WORLD}"


def _run_checks(checks: Sequence[Any], transport: str, fail) -> None:
    """Run `checks` in order; rank 0 reports each one's wall time. A failing
    rank calls `fail`, which must not return: a rank that leaves a collective
    early wedges every sibling in it."""
    start = time.monotonic()
    for check in checks:
        t0 = time.monotonic()
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__} ({transport})", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            fail()
        if RANK == 0:
            print(
                f"[rank 0] {transport} world {WORLD}: passed {check.__name__} "
                f"in {time.monotonic() - t0:.1f}s",
                flush=True,
            )
    _barrier()
    torch.cuda.synchronize()
    print(
        f"[rank {RANK}] {transport} world {WORLD}: {len(checks)} checks passed "
        f"in {time.monotonic() - start:.1f}s",
        flush=True,
    )


def _run_one_rank(device: Optional[int] = None) -> int:
    """Body of one spawned rank (pg transport): run every check; a failing rank
    exits nonzero and the launcher kills its siblings."""
    global HOST_GROUP, SUBGROUPS, PG_BOXED
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    # Spawned on one node, the rank owns device `rank`. Across nodes (--srun)
    # the caller passes the node-local index. Never read LOCAL_RANK here: a
    # spawned rank inherits the launching shell's value, the same on every
    # rank, and NCCL then refuses the group with "invalid usage".
    if device is None:
        device = rank
    torch.cuda.set_device(device)
    # device_id keeps NCCL communicator teardown off the implicit-device path
    # that measurably wedges a watchdog at exit.
    dist.init_process_group("nccl", init_method="env://", device_id=torch.device("cuda", device))

    PG_BOXED = dist.distributed_c10d._get_default_group().boxed()
    _bind_entry(rank, world, PG_BOXED)
    HOST_GROUP = dist.new_group(backend="gloo")
    # A rank subset is a real sub-ProcessGroup on this path, not just a shorter
    # `group` list: the _pg op gathers on the communicator it is handed, so a
    # 2-rank gather needs a 2-rank group. Every rank must enter new_group for
    # each subset, in the same order, or it deadlocks -- so build them all here.
    for _ranks in ([0, 1], [WORLD - 2, WORLD - 1]):
        key = tuple(sorted(set(_ranks)))
        if key not in SUBGROUPS:
            SUBGROUPS[key] = dist.new_group(list(key))

    # Exit hard rather than return: the launcher kills the rest the moment it
    # sees this nonzero exit.
    _run_checks(CHECKS, "pg", lambda: os._exit(13))
    # Exit without tearing the process group down: destroying a communicator
    # that captured collectives into CUDA graphs measurably wedges NCCL's
    # watchdog, and these are throwaway processes the launcher owns anyway.
    sys.stdout.flush()
    os._exit(0)


def _run_one_mpi_rank() -> int:
    """Body of one mpirun rank (mpi transport): the MPI world is the group, and
    the entry runs the MPI-session symbol a served engine calls."""
    global MPI_COMM
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    # The launcher started one rank per device it names, in order.
    torch.cuda.set_device(rank % torch.cuda.device_count())
    _bind_entry(rank, world, None)
    MPI_COMM = comm
    # A failing rank aborts the MPI world, which ends every rank and mpirun.
    _run_checks(MPI_CHECKS, "mpi", lambda: comm.Abort(13))
    # Finalize MPI, so mpirun reads a clean exit, then leave without running the
    # NCCL communicators' destructors, for the reason the pg ranks give.
    sys.stdout.flush()
    MPI.Finalize()
    os._exit(0)


def _visible_world() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, (
        "set CUDA_VISIBLE_DEVICES to the devices this run owns, "
        "e.g. export CUDA_VISIBLE_DEVICES=0,1,2,3"
    )
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size >= 2, (
        f"CUDA_VISIBLE_DEVICES names {world_size} device(s); a collective test needs at least 2"
    )
    return world_size


def _spawn_ranks(world_size: int, deadline_s: int) -> None:
    """pg transport: start one worker process per claimed device; no external
    launcher.

    The ranks rendezvous through torch.distributed's env:// method, so the only
    thing this needs is a free port. Each child gets its own session; on the
    first nonzero exit or on the deadline, every session is killed, which is
    what keeps one wedged collective from outliving the run.
    """
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
    print(f"[launcher] pg: spawning {world_size} ranks, rendezvous on :{port}", flush=True)
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

    deadline = time.monotonic() + deadline_s
    live = set(range(world_size))
    failure = None
    while live and failure is None:
        if time.monotonic() > deadline:
            failure = f"did not finish in {deadline_s}s (wedged)"
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
        raise AssertionError(f"the {world_size}-rank pg run failed: {failure}")


def _inherited_from_a_launch(key: str) -> bool:
    """An environment variable that places this process inside some launch.

    mpirun reads them as the allocation it runs in: inside a one-task Slurm
    allocation the SLURM_ ones make it refuse to start more ranks than the one
    task (or start ranks that each believe they are rank 0 of a world of 1),
    and an enclosing PMIx/OMPI namespace would register its ranks with that
    launch. The MCA settings and the run-as-root consent are configuration,
    not placement, and are kept.
    """
    if key.startswith("SLURM_"):
        return True
    if key.startswith(("PMIX_MCA_", "OMPI_MCA_", "OMPI_ALLOW_RUN_AS_ROOT")):
        return False
    return key.startswith(("PMIX_", "PMI_", "OMPI_"))


def _spawn_mpi_ranks(world_size: int, deadline_s: int) -> None:
    """mpi transport: re-execute this file under mpirun, one rank per device,
    killed with its process group at the deadline.

    `_lockstep.spawn`'s launch, with `--oversubscribe` and the environment
    scrubbed of any enclosing launch (see `_inherited_from_a_launch`), so it
    also starts inside a scheduler allocation that granted this process one
    task.
    """
    env = {k: v for k, v in os.environ.items() if not _inherited_from_a_launch(k)}
    env.pop("TLLM_DISABLE_MPI", None)
    command = [
        "mpirun",
        "--allow-run-as-root",
        "--oversubscribe",
        "-n",
        str(world_size),
        sys.executable,
        str(Path(__file__).resolve()),
        _MPI_WORKER_FLAG,
    ]
    print(f"[launcher] mpi: {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        code = process.wait(timeout=deadline_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise AssertionError(
            f"the {world_size}-rank mpi run did not finish in {deadline_s}s (wedged)"
        ) from None
    assert code == 0, f"the {world_size}-rank mpi run exited {code}"


def _argument(name: str, default: str) -> str:
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    if _MPI_WORKER_FLAG in sys.argv:
        sys.exit(_run_one_mpi_rank())
    if "--srun" in sys.argv:
        # _rank_job is beside this file, in comm/.
        sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
        import _rank_job

        sys.exit(_run_one_rank(device=_rank_job.adopt_srun_env()))
    WORLD_SIZE = _visible_world()
    requested = _argument("--world-size", str(WORLD_SIZE))
    assert int(requested) == WORLD_SIZE, (
        f"--world-size {requested}, but CUDA_VISIBLE_DEVICES names {WORLD_SIZE} devices"
    )
    TRANSPORT = _argument("--transport", "both")
    assert TRANSPORT in ("both", "pg", "mpi"), f"--transport {TRANSPORT}: both, pg or mpi"
    if TRANSPORT in ("both", "pg"):
        _spawn_ranks(WORLD_SIZE, PG_DEADLINE_S)
    if TRANSPORT in ("both", "mpi"):
        _spawn_mpi_ranks(WORLD_SIZE, MPI_DEADLINE_S)
    print("OK")
