# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the `cute_dsl_allreduce_add_rmsnorm` catalog entry.

Every cell names its world size and the workspace it runs on (`max_rows`), and
each job drives the cells of its own world size: one workspace per distinct
`max_rows` among them is built, the Mistral Large 4 FP8 cells on that target's
9216-row workspace and the Mistral Large 4 NVFP4 cells on those targets'
32768-row one (world 4: tep4 and tp4; world 8: tep8 and tp8), which is how
each target builds it. One check reads the 32768-row workspace's peer buffers
where their offsets pass 2^31 elements
(`check_peer_buffer_offsets_past_2_31_land_in_place`); at world 8 they pass
2^32.

The Mistral Large 4 cells are driven the way those targets serve: their decode
steps replay captured CUDA graphs and their prefill steps run eagerly, all on
one workspace. So each decode cell is one step's graph -- 144 launches, two
per layer, carrying one residual through them in place -- and its replays are
interleaved with the prefill cells' eager calls: a decode launch then reuses a
slot a two-shot call of up to the whole workspace filled, which is what the
per-slot vector counts in the device state exist to clear.

A collective cannot be exercised in one process. Two jobs, both this file:

- the world-4 cells, on one node. Run plainly, this script is its own
  launcher: it spawns one worker process per device named in
  CUDA_VISIBLE_DEVICES (four), under a deadline the parent enforces by
  killing every worker's session. A Lamport collective whose ranks disagree
  hangs rather than raising, so the deadline is what keeps a broken kernel
  from taking the calling run down with it. The workers are plain processes,
  no MPI: the workspace brings its process group up from the rendezvous
  address the launcher puts in the environment.

      CUDA_VISIBLE_DEVICES=0,1,2,3 python _cute_dsl_allreduce_add_rmsnorm_op_matrix.py

- the world-8 cells, over two nodes of four devices. Started once per rank by
  an MPI launcher with `--mpi-rank-worker`, as the engine's ranks are
  started: each rank takes the device of its node-local rank, drops any
  rendezvous address its node's environment holds, and the workspace brings
  its process group up from rank 0's address broadcast over
  `MPI.COMM_WORLD`. One process cannot place ranks on another node, so the
  launcher is the environment's (the collected test takes it from
  `MODELING_V2_MPI_LAUNCH`):

      <launcher> python _cute_dsl_allreduce_add_rmsnorm_op_matrix.py --mpi-rank-worker

  A Lamport collective one rank misses hangs every other rank. So every
  rank-local assertion of this job is agreed over `MPI.COMM_WORLD` before any
  rank runs another collective: a failure on one rank fails every rank at the
  same point, and each exits nonzero. A rank that wedges inside a collective
  never reaches the agreement; `MPI_DEADLINE_S` ends it (faulthandler dumps
  its stack and exits), and the launcher ends its siblings.

Not a pytest module, despite the `check_*` bodies: they are one ordered
sequence inside a single job, reading rank state only `_run_one_rank` binds,
and the slot-counter checks depend on every launch before them. The collected
entry point is `test_modeling_v2_cute_dsl_allreduce_add_rmsnorm.py`
beside this file. This file imports only torch and the standard library at
module scope: `_rank_job` executes it to read `DEADLINE_S`.
"""

import faulthandler
import math
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
import types
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist

_WORKER_FLAG = "--rank-worker"
_MPI_WORKER_FLAG = "--mpi-rank-worker"
#: The world-4 job: two workspaces of four CuTe DSL compiles each per rank, then the
#: cells, the largest of which build fp64 references over 32768 x 6144: the whole job
#: measured 51 s with a third workspace and its cells besides, so this is some twelve
#: times that. Read by `_rank_job.run`.
DEADLINE_S = 600
#: The world-8 job's own deadline, each rank's: one workspace, sixteen captured steps
#: replayed three times each between eager calls of up to 32768 rows. Read by the
#: collected test, which allows its launcher this plus a grace.
MPI_DEADLINE_S = 900

#: The spawned job's world size: the world-4 cells, on the devices of one node.
WORLD_SIZE = 4
#: The MPI job's: the world-8 cells, over two nodes.
MPI_WORLD_SIZE = 8
HIDDEN = 6144
#: Mistral Large 4's `rms_norm_eps`.
EPS = 1e-5
#: Launches per captured decode graph: Mistral Large 4's two closing collectives per
#: layer over its 72 layers, one decode step.
STEP_LAUNCHES = 144
#: Each decode graph is replayed this many times, after a different prefill cell
#: each time. Between two replays of one graph a round launches, per graph cell,
#: one eager call and one 144-launch graph, 145 launches, 1 mod 3: the slot counter
#: moves by the number of graph cells mod 3 (5 graphs per Mistral Large 4 TP4
#: workspace, 16 at TP8), so while that is not a multiple of 3 the three replays
#: start on three different slots.
GRAPH_ROUNDS = 3

# The payload scales. The world's partials sum to std 1 (each std 1/sqrt(world):
# 0.5 at TP4), the residual is std 1, so the sum and the residual are the same
# size and neither hides the other's rounding. The norm weight is 1 +- 1/8, the
# spread the band derivations below assume.
_RESIDUAL_STD = 1.0
_WEIGHT_SPREAD = 0.25
#: Scales the payload so a row's mean square is about 2e-6, near eps: see
#: `check_the_workspace_eps_reaches_the_kernel`.
_SMALL = 1e-3

# Bound in the rank body, never at module scope: importing the entry pulls in
# tensorrt_llm, and `_rank_job` executes this file to read DEADLINE_S.
RANK = 0
WORLD = 1
#: Distinct hosts the job's ranks run on.
NODES = 1
#: `MPI.COMM_WORLD` in the MPI job, where rank-local failures are agreed; None in the
#: spawned job, whose launcher ends every sibling of a failing rank itself.
COMM: Any = None
TWOSHOT_ABOVE = 0
op: Any = None
validating: Any = None
assert_within_ulp: Any = None
#: `max_rows` -> (the workspace, its slot-counter mirror).
WORKSPACES: Dict[int, Tuple[Any, "_Mirror"]] = {}
#: How often the entry's own `__call__` and `reference` ran (see `_count_entry_calls`).
SEEN: Dict[str, int] = {"__call__": 0, "reference": 0}


# ---- agreeing on a rank-local failure ---------------------------------------


def _agree(where: str, error: Optional[str]) -> None:
    """Every rank's verdict on a rank-local block; any failure fails every rank here."""
    errors = COMM.allgather(error)
    failed = [(r, e) for r, e in enumerate(errors) if e is not None]
    if failed:
        detail = "\n".join(f"  rank {r}: {e[:1500]}" for r, e in failed)
        raise AssertionError(
            f"{where}: failed on rank(s) {[r for r, _ in failed]} of {WORLD}\n{detail}"
        )


@contextmanager
def _rank_local(where: str) -> Iterator[None]:
    """A block of rank-local assertions, ahead of the next collective.

    In the MPI job its outcome is agreed across the ranks on the way out, so
    every rank leaves it together, raising or not; a failing rank cannot leave
    its siblings inside a collective it skipped. In the spawned job an
    assertion propagates as it is, and the launcher ends the siblings.
    """
    if COMM is None:
        yield
        return
    error = None
    try:
        yield
    except Exception as exc:  # noqa: BLE001 -- reported on every rank, re-raised below
        error = f"{type(exc).__name__}: {exc}"
        print(f"[rank {RANK}] {where}:\n{traceback.format_exc()}", flush=True)
    _agree(where, error)


# ---- the host mirror of a workspace's device slot counter ----------------


class _Mirror:
    """What one workspace's device slot state must read after the launches so far.

    `state` is `[entry counter, current slot, rows*hidden/8 last published per
    slot (3), release epoch]`. Every launch of any of the workspace's kernels
    takes the current slot, records how many 16-byte vectors it published
    there, and moves the slot on by one mod 3; the entry counter returns to 0
    when the launch completes. The epoch moves only under the `strict` PDL
    policy, which no kernel of the workspace uses.
    """

    def __init__(self, ws: Any) -> None:
        self.ws = ws
        self.slot = 0
        self.counts = [0, 0, 0]

    def launched(self, rows: int) -> None:
        self.counts[self.slot] = rows * HIDDEN // 8
        self.slot = (self.slot + 1) % 3

    def check(self, where: str) -> None:
        torch.cuda.synchronize()
        got = self.ws.state.tolist()
        want = [0, self.slot, *self.counts, 0]
        assert got == want, f"{where}: device slot state {got}, expected {want}"


# ---- payloads -------------------------------------------------------------


def _seed(*parts: Any) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % 2**31


def _randn(rows: int, seed: int, std: float) -> torch.Tensor:
    # Every rank regenerates every rank's payload from the seed: the CUDA
    # generator is deterministic per (seed, shape), so the reference is
    # arithmetic on this rank rather than a second collective.
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, HIDDEN, generator=g, device="cuda") * std).to(torch.bfloat16)


def _partials(rows: int, seed: int, scale: float = 1.0) -> List[torch.Tensor]:
    """Every rank's partial, in rank order."""
    std = 1.0 / math.sqrt(WORLD)
    return [_randn(rows, seed * 977 + r + 1, std * scale) for r in range(WORLD)]


def _residual(rows: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    """The residual stream: replicated, identical on every rank, as in the target."""
    return _randn(rows, seed * 977 + 101, _RESIDUAL_STD * scale)


def _weight(seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed * 977 + 202)
    u = torch.rand(HIDDEN, generator=g, device="cuda") - 0.5
    return (1.0 + u * _WEIGHT_SPREAD).to(torch.bfloat16)


# ---- gates ----------------------------------------------------------------


def _assert_same_on_every_rank(t: torch.Tensor, where: str) -> None:
    """Every rank sums in the same order, so every rank's result is bit-identical."""
    bits = t.contiguous().view(torch.uint8).to(torch.int32)
    lo, hi = bits.clone(), bits.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    assert torch.equal(lo, hi), f"{where}: the ranks hold different results"


def _gate_norm_one_shot(norm: torch.Tensor, ref_norm: torch.Tensor, where: str) -> None:
    """A one-shot norm against the fp64 reference: 2 ulp element-wise, 0.5 RMS (see `_gate`)."""
    try:
        assert_within_ulp(norm, ref_norm, element_ulp=2.0, rms_ulp=0.5)
    except AssertionError as exc:
        raise AssertionError(f"{where}: {exc}") from None


def _gathered(partials: List[torch.Tensor]) -> torch.Tensor:
    """What the two-shot gather carries: the fp32 sum in rank order, rounded once to bf16."""
    acc = torch.zeros(partials[0].shape, dtype=torch.float32, device=partials[0].device)
    for p in partials:
        acc += p.float()
    return acc.to(torch.bfloat16)


def _refs_per_check(rows: int) -> int:
    """How many times `_gate` calls the entry's `reference` at this row count."""
    return 1 if rows <= TWOSHOT_ABOVE else 2


def _gate(
    rows: int,
    norm: torch.Tensor,
    new_residual: torch.Tensor,
    inputs: Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor],
    ws: Any,
    where: str,
    report: bool = False,
) -> None:
    """Gate one call's outputs against the entry's `reference` on its inputs; the bands, and why.

    One-shot (rows <= TWOSHOT_ABOVE): the residual is bf16(fp32 sum in rank
    order + residual), the reference's arithmetic exactly, so it is bit-exact.
    The norm differs from the fp64 reference only through the kernel's fp32
    sum of squares, its approximate rsqrt and its fp32 products, about 1e-6
    relative before the bf16 rounding. That moves an element only where it
    sits on a bf16 rounding boundary, and then by one ulp of that element:
    at most 2^-7 of it, 2 units of 2^-8 against the row's largest. So 2 ulp
    element-wise; and since about 1e-6 / 2^-8 of the elements move, 0.5 ulp
    relative RMS is some ten times what that rate gives.

    Two-shot (rows > TWOSHOT_ABOVE): the reduced rows are gathered as bf16, so
    the residual is bf16(bf16(sum) + residual), rounded twice. Against the
    single rounding that is at most half an ulp of the sum plus an ulp of the
    result: 1 + 2 units of the row's largest when the sum is no larger than
    the residual stream it lands in (both std 1 here, at every world size), so
    4 ulp element-wise. Double rounding moves roughly a quarter to a half of
    the elements by one ulp, about 1 ulp relative RMS, so 2. And the residual
    must NOT be bit-exact: that is what shows the two-shot kernel ran. The
    norm inherits the residual's difference scaled by the weight, at most
    4 x (9/7) units with the weight in 1 +- 1/8, plus its own 2: 8 ulp
    element-wise, 2 RMS.

    That band admits a sum taken in bf16 pairwise: measured 3.2-3.4 ulp
    element-wise, 0.64 RMS on the residual at 9216 rows (TP4). So the two-shot
    call is also gated against its own arithmetic step for step: the entry's
    `reference` given the bf16 gathered sum as the one partial computes
    bf16(bf16(sum) + residual) and its norm. Measured bit-exact on the
    residual at 3073, 4096, 9215 and 9216 rows and on every rank at TP4, with
    the norm within 1.35 ulp element-wise, 0.0045 RMS -- the one-shot band
    holds. The bf16 pairwise sum differs from it in about a fifth of the
    elements.

    `report` prints, on rank 0, how many elements differ from the reference.
    """
    partials, residual, weight = inputs
    ref_norm, ref_residual = op.reference(partials, residual, weight, norm, ws)
    if report and RANK == 0:
        print(
            f"[rank 0] {where}: {int((norm != ref_norm).sum())} norm and "
            f"{int((new_residual != ref_residual).sum())} residual elements of "
            f"{norm.numel()} differ from the reference",
            flush=True,
        )
    if rows <= TWOSHOT_ABOVE:
        assert torch.equal(new_residual, ref_residual), f"{where}: one-shot residual not exact"
        _gate_norm_one_shot(norm, ref_norm, where)
        return
    assert_within_ulp(new_residual, ref_residual, element_ulp=4.0, rms_ulp=2.0)
    assert not torch.equal(new_residual, ref_residual), (
        f"{where}: a two-shot residual rounded once: the one-shot kernel ran above "
        f"{TWOSHOT_ABOVE} rows"
    )
    assert_within_ulp(norm, ref_norm, element_ulp=8.0, rms_ulp=2.0)
    step_norm, step_residual = op.reference([_gathered(partials)], residual, weight, norm, ws)
    if report and RANK == 0:
        print(
            f"[rank 0] {where}: {int((norm != step_norm).sum())} norm and "
            f"{int((new_residual != step_residual).sum())} residual elements differ from "
            f"the step-for-step reference",
            flush=True,
        )
    assert torch.equal(new_residual, step_residual), (
        f"{where}: the two-shot residual is not bf16(bf16(fp32 sum in rank order) + residual)"
    )
    _gate_norm_one_shot(norm, step_norm, where)


# ---- one call of the entry, with its inputs and its check -----------------


class _Call:
    """One entry call on one workspace: owns its buffers, so it can be refilled and replayed."""

    def __init__(self, rows: int, max_rows: int) -> None:
        self.rows, self.max_rows = rows, max_rows
        self.ws, self.mirror = WORKSPACES[max_rows]
        dev = "cuda"
        self.partial = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.residual = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.weight = torch.empty(HIDDEN, dtype=torch.bfloat16, device=dev)
        self._inputs: Optional[tuple] = None

    def fill(self, seed: int, scale: float = 1.0) -> None:
        """Fresh inputs for `seed`; NaN in the output, so an unwritten element fails."""
        partials = _partials(self.rows, seed, scale)
        residual, weight = _residual(self.rows, seed, scale), _weight(seed)
        self.partial.copy_(partials[RANK])
        self.residual.copy_(residual)
        self.weight.copy_(weight)
        self.out.fill_(float("nan"))
        self._inputs = (partials, residual, weight)

    def launch(self) -> None:
        assert op(self.partial, self.residual, self.weight, self.out, self.ws) is self.out

    def check(self, where: str, report: bool = False) -> None:
        """The slot state and the gate on this rank, agreed; then bit-identity across ranks."""
        where = f"{where}: {self.rows} rows on the {self.max_rows}-row workspace"
        with _rank_local(where):
            self.mirror.check(where)
            _gate(self.rows, self.out, self.residual, self._inputs, self.ws, where, report)
        _assert_same_on_every_rank(self.out, where)
        _assert_same_on_every_rank(self.residual, where)


def _count_entry_calls() -> None:
    """What `_validating.certifying` does under pytest, for this rank body.

    The entry's own `__call__` (with `is_valid` interposed first, as
    `validating` does) and its own `reference` are wrapped to count, so every
    cell can assert that both ran for it rather than a local copy.
    """
    cls = type(op)
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


def _drove(i: int, before: Dict[str, int], calls: int, refs: int, driven: Dict[int, int]) -> None:
    """Record that cell `i` ran through the entry's own `__call__` and `reference`."""
    got = (SEEN["__call__"] - before["__call__"], SEEN["reference"] - before["reference"])
    assert got == (calls, refs), (
        f"cell {op.CELLS[i].why!r}: the entry's __call__ / reference ran {got} times, "
        f"expected {(calls, refs)}"
    )
    driven[i] = driven.get(i, 0) + 1


def _mine() -> List[int]:
    """The cells this job drives: those at its world size."""
    return [i for i, c in enumerate(op.CELLS) if c.spec["world_size"] == WORLD]


# ---- checks ---------------------------------------------------------------


def check_every_cell_names_this_configuration() -> None:
    """The constants this matrix builds tensors from are the ones every cell states.

    Every cell is at one of the two jobs' world sizes, and this job's cells
    state the topology it runs on: `nodes` (1 when absent) distinct hosts.
    """
    with _rank_local("the cells' configuration"):
        assert _mine(), f"no cell is at world size {WORLD}"
        for cell in op.CELLS:
            spec = cell.spec
            assert spec["world_size"] in (WORLD_SIZE, MPI_WORLD_SIZE), (cell.why, spec)
            assert spec["hidden"] == HIDDEN and spec["eps"] == EPS, (cell.why, spec)
            assert 1 <= spec["rows"] <= spec["max_rows"], (cell.why, spec)
            if spec.get("graph"):
                assert spec["rows"] <= TWOSHOT_ABOVE, (cell.why, "a decode graph cell is one-shot")
        for i in _mine():
            cell = op.CELLS[i]
            nodes = cell.spec.get("nodes", 1)
            assert nodes == NODES, (cell.why, f"this job spans {NODES} node(s)")


def check_the_workspaces_agreed_and_launched_nothing() -> None:
    """Every rank agreed the same multicast decision, and building launched no kernel."""
    for max_rows, (ws, mirror) in WORKSPACES.items():
        flags: List[Any] = [None] * WORLD
        dist.all_gather_object(flags, ws.multicast_base != 0)
        with _rank_local(f"the {max_rows}-row workspace"):
            assert len(set(flags)) == 1, f"ranks disagree on multicast: {flags}"
            assert ws.max_rows == max_rows and ws.hidden == HIDDEN and ws.world == WORLD
            assert ws.eps == EPS
            mirror.check(f"after building the {max_rows}-row workspace")
        if RANK == 0:
            print(
                f"[rank 0] {max_rows}-row workspace: multicast publish "
                f"{'on' if flags[0] else 'off'}",
                flush=True,
            )


def _eager(i: int, rnd: int, driven: Dict[int, int]) -> None:
    cell = op.CELLS[i]
    rows, max_rows = cell.spec["rows"], cell.spec["max_rows"]
    where = f"cell {cell.why!r}, round {rnd}"
    call = _Call(rows, max_rows)
    call.fill(_seed(cell.why, rnd))
    torch.cuda.synchronize()
    dist.barrier()
    before = dict(SEEN)
    call.launch()
    call.mirror.launched(rows)
    call.check(where, report=rnd == 0)
    with _rank_local(where):
        _drove(i, before, calls=1, refs=_refs_per_check(rows), driven=driven)
    dist.barrier()


class _Step:
    """A decode cell: one step's `STEP_LAUNCHES` launches at its row count, captured once.

    As in the target's forward, one residual buffer is carried through every
    launch in place, each launch with its own partial, output and norm weight
    (each layer's norms differ), and nothing between the launches: launch
    `k + 1` reads the residual launch `k` wrote. On a one-shot row count each
    residual is exact, so the chained reference stays exact to the last launch.
    """

    def __init__(self, i: int) -> None:
        self.i = i
        cell = op.CELLS[i]
        self.rows, self.max_rows = cell.spec["rows"], cell.spec["max_rows"]
        self.ws, self.mirror = WORKSPACES[self.max_rows]
        shape, bf16 = (self.rows, HIDDEN), torch.bfloat16
        self.residual = torch.empty(shape, dtype=bf16, device="cuda")
        self.partials = torch.empty((STEP_LAUNCHES, *shape), dtype=bf16, device="cuda")
        self.outs = torch.empty((STEP_LAUNCHES, *shape), dtype=bf16, device="cuda")
        self.weights = torch.empty((STEP_LAUNCHES, HIDDEN), dtype=bf16, device="cuda")
        self._inputs: Optional[tuple] = None
        self.fill(_seed(cell.why, "capture"))
        torch.cuda.synchronize()
        dist.barrier()
        before = SEEN["__call__"]
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            for k in range(STEP_LAUNCHES):
                out = self.outs[k]
                assert op(self.partials[k], self.residual, self.weights[k], out, self.ws) is out
        with _rank_local(f"capturing {cell.why!r}"):
            captured = SEEN["__call__"] - before
            assert captured == STEP_LAUNCHES, captured
        dist.barrier()

    def fill(self, seed: int) -> None:
        """Fresh inputs for `seed`; NaN in the outputs, so an unwritten element fails."""
        partials = [_partials(self.rows, _seed(seed, k)) for k in range(STEP_LAUNCHES)]
        residual = _residual(self.rows, seed)
        weights = [_weight(_seed(seed, k)) for k in range(STEP_LAUNCHES)]
        self.partials.copy_(torch.stack([p[RANK] for p in partials]))
        self.residual.copy_(residual)
        self.weights.copy_(torch.stack(weights))
        self.outs.fill_(float("nan"))
        self._inputs = (partials, residual, weights)

    def replay(self, rnd: int, driven: Dict[int, int]) -> None:
        cell = op.CELLS[self.i]
        where = f"cell {cell.why!r}, replay {rnd}"
        self.fill(_seed(cell.why, rnd))
        torch.cuda.synchronize()
        dist.barrier()
        before = dict(SEEN)
        self.graph.replay()
        for _ in range(STEP_LAUNCHES):
            self.mirror.launched(self.rows)
        with _rank_local(where):
            self.mirror.check(where)
            partials, residual, weights = self._inputs
            differ = 0
            for k in range(STEP_LAUNCHES):
                ref_norm, residual = op.reference(
                    partials[k], residual, weights[k], self.outs[k], self.ws
                )
                differ += int((self.outs[k] != ref_norm).sum())
                _gate_norm_one_shot(self.outs[k], ref_norm, f"{where}, launch {k}")
            assert torch.equal(self.residual, residual), (
                f"{where}: the residual carried through {STEP_LAUNCHES} launches is not exact"
            )
            if RANK == 0:
                print(
                    f"[rank 0] {where}: {differ} norm elements of "
                    f"{STEP_LAUNCHES * self.rows * HIDDEN} differ from the reference, "
                    f"the carried residual exact",
                    flush=True,
                )
        _assert_same_on_every_rank(self.outs, where)
        _assert_same_on_every_rank(self.residual, where)
        # A replay runs the kernels the capture recorded and no Python: the
        # entry's `__call__` ran (and its guard) at capture, counted there.
        with _rank_local(where):
            _drove(self.i, before, calls=0, refs=STEP_LAUNCHES, driven=driven)
        dist.barrier()


def check_every_cell_on_its_workspace() -> None:
    """Drive every cell of this job's world size, through the entry's own `__call__` and `reference`.

    Eager cells of a workspace without graph cells run in order and then in
    reverse, so a small call follows the largest one and has to clear the slot
    it filled. A workspace with graph cells captures each once, then replays
    each `GRAPH_ROUNDS` times, each replay after an eager cell's call (a
    different one per round): a decode step after a prefill chunk, sharing one
    workspace, as the target serves. The slot state is checked after every
    call and every replay.
    """
    driven: Dict[int, int] = {}
    for max_rows, (_, mirror) in WORKSPACES.items():
        mine = [i for i in _mine() if op.CELLS[i].spec["max_rows"] == max_rows]
        eager = [i for i in mine if not op.CELLS[i].spec.get("graph")]
        graphs = [_Step(i) for i in mine if op.CELLS[i].spec.get("graph")]
        with _rank_local(f"the {max_rows}-row workspace's graphs"):
            mirror.check(f"after capturing {len(graphs)} graphs: capturing launches nothing")
        if not graphs:
            for rnd, order in enumerate((eager, eager[::-1])):
                for i in order:
                    _eager(i, rnd, driven)
            continue
        with _rank_local(f"the {max_rows}-row workspace's rounds"):
            assert eager, "graph cells are replayed between eager cells"
            assert len(graphs) % 3, "the replays of a graph would all start on one slot"
        for rnd in range(GRAPH_ROUNDS):
            for k, g in enumerate(graphs):
                _eager(eager[(k + rnd) % len(eager)], rnd, driven)
                g.replay(rnd, driven)
        for g in graphs:
            g.graph.reset()
    with _rank_local("every cell driven"):
        undriven = [op.CELLS[i].why for i in _mine() if i not in driven]
        assert not undriven, f"cells never driven: {undriven}"
    if RANK == 0:
        print(f"[rank 0] cells driven: {driven}", flush=True)


# ---- peer-buffer offsets past 2^31 -----------------------------------------

#: Bitwise bf16 -0.0: the in-band "not yet arrived" mark a cleared vector holds.
_NOT_ARRIVED = -32768
#: Bitwise bf16 1.0: a stale vector that reads as arrived.
_STALE = 0x3F80


def _probe_call(rows: int, max_rows: int, *seed: Any) -> "_Call":
    """One call outside any cell, on the workspace's current slot, gated like a cell's."""
    call = _Call(rows, max_rows)
    call.fill(_seed("offsets", *seed))
    torch.cuda.synchronize()
    dist.barrier()
    call.launch()
    call.mirror.launched(rows)
    call.check(f"offsets probe {seed}")
    dist.barrier()
    return call


def check_peer_buffer_offsets_past_2_31_land_in_place() -> None:
    """At 32768 rows a peer buffer passes 2^31 elements; read the buffer where it does.

    Each rank's buffer is `[3 slots, world regions, max_rows x 6144]` bf16. At
    max_rows 32768 that is 2,415,919,104 elements (4.5 GiB) at world 4 and
    4,831,838,208 (9 GiB, past 2^32 elements) at world 8, and slot 2's last
    region starts at element (2 x world + world - 1) x 201,326,592:
    2,214,592,512 at world 4, past 2^31 elements and 2^32 bytes;
    4,630,511,616 at world 8, past 2^32 elements and 2^33 bytes. A one-shot
    launch publishes the rank's partial into region <its rank> of every
    peer's buffer, and every launch clears the next slot's regions; two-shot
    moves data through regions 0 and 1 only (on slot 2 below 2^31 at world 4,
    from element 3,221,225,472 at world 8) but its clear covers every region.
    A 32-bit offset there would wrap to an address outside the buffer. The
    cells reach slot 2 as well, gated on their outputs; this reads the buffer
    itself:

    - a one-shot call at 3072 rows (the most one-shot publishes) on slot 2:
      every peer's region of this rank's buffer holds that peer's partial bit
      for bit (-0.0 is published as +0.0), the last rank's furthest out;
    - a one-row call on slot 0, then one on slot 1, which clears slot 2:
      those regions read -0.0 again;
    - a 32768-row two-shot call on slot 2, gated; then every region of slot 2
      overwritten with a ready-looking 1.0, out to the buffer's last element;
      a one-row call on slot 0 and a two-shot call on slot 1, whose clear
      covers slot 2's regions to 32768 rows: every element of slot 2 reads
      -0.0;
    - a one-shot call on slot 2 after that, gated: no stale vector was read.
    """
    probed = []
    for max_rows, (ws, mirror) in WORKSPACES.items():
        numel = max_rows * HIDDEN
        if 3 * WORLD * numel <= 2**31:
            continue
        bits = ws.buffer.view(torch.int16)
        last = (2 * WORLD + WORLD - 1) * numel
        with _rank_local(f"the {max_rows}-row workspace's buffer"):
            assert tuple(bits.shape) == (3, WORLD, numel), bits.shape
            assert last > 2**31, "slot 2's last region starts below 2^31"
        peers = [s for s in range(WORLD) if s != RANK]

        while mirror.slot != 2:
            _probe_call(1, max_rows, "to slot 2", mirror.slot)
        rows = TWOSHOT_ABOVE
        partials = _probe_call(rows, max_rows, "one-shot publish")._inputs[0]
        with _rank_local(f"{max_rows}-row workspace, slot 2, one-shot publish"):
            for s in peers:
                want = partials[s].view(torch.int16).reshape(-1)
                want = torch.where(want == _NOT_ARRIVED, torch.zeros_like(want), want)
                assert torch.equal(bits[2, s, : rows * HIDDEN], want), (
                    f"{max_rows}-row workspace, slot 2, region {s} (from element "
                    f"{(2 * WORLD + s) * numel}): rank {s}'s partial is not where it was "
                    f"published"
                )
        _probe_call(1, max_rows, "one-shot clear", 0)
        _probe_call(1, max_rows, "one-shot clear", 1)
        torch.cuda.synchronize()
        with _rank_local(f"{max_rows}-row workspace, slot 2, one-shot clear"):
            for s in peers:
                left = int((bits[2, s, : rows * HIDDEN] != _NOT_ARRIVED).sum())
                assert left == 0, (
                    f"{max_rows}-row workspace, slot 2, region {s}: {left} published "
                    f"elements not cleared by the launch on slot 1"
                )
            assert mirror.slot == 2, mirror.slot

        _probe_call(max_rows, max_rows, "two-shot, whole workspace")
        bits[2].fill_(_STALE)
        torch.cuda.synchronize()
        with _rank_local(f"{max_rows}-row workspace, slot 2 overwritten"):
            assert int((bits[2] == _STALE).sum()) == WORLD * numel
        dist.barrier()
        _probe_call(1, max_rows, "two-shot clear", 0)
        _probe_call(TWOSHOT_ABOVE + 1, max_rows, "two-shot clear", 1)
        torch.cuda.synchronize()
        with _rank_local(f"{max_rows}-row workspace, slot 2, two-shot clear"):
            left = int((bits[2] != _NOT_ARRIVED).sum())
            assert left == 0, (
                f"{max_rows}-row workspace: {left} of slot 2's {WORLD * numel} elements, last "
                f"element {3 * WORLD * numel - 1}, not cleared by the two-shot launch on slot 1"
            )
        _probe_call(16, max_rows, "slot 2 after the clear")
        probed.append(max_rows)
        if RANK == 0:
            print(
                f"[rank 0] {max_rows}-row workspace: rank {WORLD - 1}'s one-shot publish on "
                f"slot 2 from element {last} in place and cleared; a two-shot call on slot 2 "
                f"(regions 0 and 1 from element {2 * WORLD * numel}) gated; slot 2 cleared to "
                f"element {3 * WORLD * numel - 1}",
                flush=True,
            )
    with _rank_local("offsets probed"):
        assert probed, "no workspace passes 2^31 elements: the 32768-row cells are gone"


def check_the_workspace_eps_reaches_the_kernel() -> None:
    """`eps` is compiled into the kernels at build; rows whose mean square is near it show which.

    At the cells' payload (mean square about 2) eps 1e-5 moves the norm by
    2.5e-6 relative, below any bf16 step, so no cell can tell 1e-5 from
    another eps. Here the payload is scaled by `_SMALL`, putting the mean
    square at 2e-6: measured, the kernel's norm then sits 129 to 500 ulp from
    a reference at eps 1e-6, 1e-4 or 0, against 3 ulp at 1e-5. The gate passes
    at the workspace's eps and must fail at 1e-6, one-shot and two-shot.
    """
    wrong = types.SimpleNamespace(eps=1e-6)
    for max_rows, (ws, mirror) in WORKSPACES.items():
        for rows in (16, max_rows):
            where = f"eps: {rows} rows on the {max_rows}-row workspace"
            call = _Call(rows, max_rows)
            call.fill(_seed("eps", max_rows, rows), scale=_SMALL)
            torch.cuda.synchronize()
            dist.barrier()
            call.launch()
            mirror.launched(rows)
            with _rank_local(where):
                mirror.check(where)
                _gate(rows, call.out, call.residual, call._inputs, ws, where)
                try:
                    _gate(rows, call.out, call.residual, call._inputs, wrong, where)
                except AssertionError:
                    pass
                else:
                    raise AssertionError(f"{where}: a reference at eps 1e-6 passes the gate too")
            dist.barrier()


def check_guards_refuse_a_row_count_above_the_budget() -> None:
    """Above `max_rows` the kernel would write past every peer's buffer.

    The guard raises before anything launches, on every rank alike, so the
    group stays in step.
    """
    for max_rows, (_, mirror) in WORKSPACES.items():
        big = _Call(max_rows + 1, max_rows)
        with _rank_local(f"{max_rows + 1} rows on the {max_rows}-row workspace"):
            try:
                with validating(op):
                    big.launch()
            except AssertionError as exc:
                assert "budget" in str(exc), exc
            else:
                raise AssertionError(f"{max_rows + 1} rows accepted by a {max_rows}-row workspace")
            mirror.check(f"after the refused call on the {max_rows}-row workspace")
        dist.barrier()


CHECKS = (
    check_every_cell_names_this_configuration,
    check_the_workspaces_agreed_and_launched_nothing,
    check_every_cell_on_its_workspace,
    check_peer_buffer_offsets_past_2_31_land_in_place,
    check_the_workspace_eps_reaches_the_kernel,
    check_guards_refuse_a_row_count_above_the_budget,
)


def _run_one_rank(mpi: bool) -> int:
    """Body of one rank: run every check; a failing rank exits nonzero.

    Spawned (`mpi` False): the launcher kills the siblings of a failing rank.
    Under MPI: the failure was agreed, so every rank fails the same check and
    exits nonzero, and the launcher reports it; a rank wedged past
    `MPI_DEADLINE_S` is ended by faulthandler.
    """
    global RANK, WORLD, NODES, COMM, TWOSHOT_ABOVE, op, validating, assert_within_ulp
    if mpi:
        faulthandler.dump_traceback_later(MPI_DEADLINE_S, exit=True)
        # A node's environment can hold a rendezvous address of its own (container hooks
        # derive MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE per node from the Slurm job), and a
        # launcher need not forward one node's to the other: ranks that kept it waited on a
        # store their siblings never joined. Every rank drops it, so all of them take
        # `ensure_process_group`'s MPI case, rank 0's address broadcast over COMM_WORLD.
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
            os.environ.pop(key, None)
        from mpi4py import MPI

        COMM = MPI.COMM_WORLD
        rank, world = COMM.Get_rank(), COMM.Get_size()
        local = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        hosts = COMM.allgather(socket.gethostname())
        NODES = len(set(hosts))
    else:
        os.environ.setdefault("TLLM_DISABLE_MPI", "1")
        rank = local = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        hosts = [socket.gethostname()] * world
    torch.cuda.set_device(local)

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import _lamport
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.cute_dsl_allreduce_add_rmsnorm import (  # noqa: E501
        cute_dsl_allreduce_add_rmsnorm,
    )

    # This file runs as a rank process, not under pytest, so the repo's
    # `__extra_import_path__` is not in play: reach the shared helper the
    # plain way (tests/unittest/_torch/modeling_v2). Confined to these rank
    # processes.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _validating import assert_within_ulp as _within
    from _validating import validating as _validating_cm

    RANK, WORLD = rank, world
    want = MPI_WORLD_SIZE if mpi else WORLD_SIZE
    assert WORLD == want, f"this job drives the world-{want} cells, got {WORLD} ranks"
    if RANK == 0:
        dev = torch.cuda.current_device()
        print(
            f"[rank 0] {torch.cuda.get_device_name(dev)} "
            f"sm_{''.join(map(str, torch.cuda.get_device_capability(dev)))}; "
            f"{WORLD} ranks on {NODES} node(s): {hosts}",
            flush=True,
        )
    # Spawned: the group comes up from the address the launcher exported.
    # Under MPI: no address is exported, so rank 0's is broadcast over
    # MPI.COMM_WORLD, as the engine's ranks bring it up.
    _lamport.ensure_process_group(rank, world)
    assert dist.is_initialized() and dist.get_world_size() == world
    op = cute_dsl_allreduce_add_rmsnorm
    TWOSHOT_ABOVE = _lamport.TWOSHOT_ABOVE
    validating, assert_within_ulp = _validating_cm, _within
    for max_rows in sorted({op.CELLS[i].spec["max_rows"] for i in _mine()}):
        t0 = time.monotonic()
        free = torch.cuda.mem_get_info()[0]
        ws = _lamport.build_workspace(HIDDEN, EPS, max_rows, rank, world)
        WORKSPACES[max_rows] = (ws, _Mirror(ws))
        torch.cuda.synchronize()
        if RANK == 0:
            print(
                f"[rank 0] built the {max_rows}-row workspace in {time.monotonic() - t0:.1f}s, "
                f"{(free - torch.cuda.mem_get_info()[0]) / 2**30:.2f} GiB of device memory "
                f"taken; buffer {tuple(ws.buffer.shape)}",
                flush=True,
            )
    _count_entry_calls()

    for check in CHECKS:
        t0 = time.monotonic()
        try:
            check()
        except BaseException:
            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # Exit hard rather than return: a rank that leaves a collective
            # early wedges every sibling in it. The spawned launcher kills the
            # rest the moment it sees this nonzero exit; under MPI every rank
            # agreed the failure and exits here too.
            os._exit(13)
        if RANK == 0:
            print(f"[rank 0] {check.__name__}: {time.monotonic() - t0:.1f}s", flush=True)
    dist.barrier()
    torch.cuda.synchronize()
    print(
        f"[rank {RANK}] {len(CHECKS)} checks passed; peak allocated "
        f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
        flush=True,
    )
    sys.stdout.flush()
    if mpi:
        faulthandler.cancel_dump_traceback_later()
        # mpirun fails a rank that exits without finalizing.
        dist.destroy_process_group()
        COMM.Barrier()
        MPI.Finalize()
    # Exit without tearing the symmetric memory down: these are throwaway
    # processes their launcher owns.
    os._exit(0)


def _spawn_ranks() -> None:
    """Start one worker process per claimed device; no external launcher.

    The workers find each other through the address and port exported here.
    Each child gets its own session; on the first nonzero exit or on the
    deadline, every session is killed, which is what keeps one wedged
    collective from outliving the run.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, (
        "set CUDA_VISIBLE_DEVICES to the devices this run owns, "
        "e.g. export CUDA_VISIBLE_DEVICES=0,1,2,3"
    )
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size == WORLD_SIZE, (
        f"CUDA_VISIBLE_DEVICES names {world_size} device(s); this job needs {WORLD_SIZE}"
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
    # Drop a launching srun step's PMIx/PMI/OMPI namespace from the children:
    # inherited, it registers them as participants of that step, and the first
    # hard exit trips srun's errhandler, which kills the whole step before
    # anything is reported. These ranks rendezvous through torch.distributed.
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


if __name__ == "__main__":
    if _MPI_WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank(mpi=True))
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank(mpi=False))
    _spawn_ranks()
    print("OK")
