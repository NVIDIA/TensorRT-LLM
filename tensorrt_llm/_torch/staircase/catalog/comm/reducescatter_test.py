# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the reducescatter catalog entry.

A collective cannot be exercised in one process, so this script is its own
launcher: run it plainly and it re-executes itself under `mpirun` with one
rank per device named in CUDA_VISIBLE_DEVICES, under a deadline the parent
enforces by killing the whole process group. A wedged collective hangs
rather than raising — and this op wedges rather than raising when the ranks
disagree about the split — so the deadline is what keeps a broken kernel
from taking the calling run down with it.

    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python catalog/comm/reducescatter_test.py

That runs two jobs, in this order. The first is the test proper: every
`TESTS` entry on `world_size` ranks, and it must exit 0. The second is a
four-rank sub-job of its own that deliberately mispairs this op against an
all-gather of a *different* byte count, which is the one call-order
divergence that hangs instead of returning wrong data — it cannot live in
the first job, because a job that wedges never reports. The launcher
certifies it out of band: every rank marks the file system before issuing
the pair, none marks it afterwards, and the job is ended by its own
watchdog. That sub-job doubles as the harness's positive control, since it
is a real collective deadlock this launcher has to survive.
"""

import os
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

assert torch.cuda.is_available(), "reducescatter requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
_WEDGE_FLAG = "--wedge-worker"
DEADLINE_S = 1800

# The wedge sub-job's budget. A pair that is going to return does so in
# microseconds, so a rank still inside it GRACE seconds later is wedged; the
# watchdog then ends its own process, because a rank blocked in a NCCL kernel
# that will never complete has no other way out. CAP covers the sub-job's
# import and communicator bring-up before the pair is issued (~50 s measured)
# plus the driver's teardown of four contexts holding a spinning kernel.
WEDGE_GRACE_S = 60
WEDGE_CAP_S = 420

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

# Bound inside the rank body, never in the launcher: importing the entry
# pulls in tensorrt_llm, which calls MPI_Init at import, and an
# MPI-initialized process cannot launch `mpirun` — measured on this host,
# mpirun then exits 1 with no output from any rank.
reducescatter: Any = None
COMM: Any = None
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
    op summing in the input dtype (see test_reduction_is_deterministic_and_
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
    acc_dtype = torch.float32 if dtype.is_floating_point else torch.int64
    acc = torch.zeros((sizes[pos], *trailing), dtype=acc_dtype, device="cuda")
    for r in ranks:
        acc += _block(r, pos, sizes[pos], seed, dtype, trailing, amp).to(acc_dtype)
    return acc.to(dtype)


def _assert_bitwise(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    """Gate of exactly zero, on payloads whose every partial sum is exact.

    Tightened from the default dtype-aware tolerances rather than loosened: the
    reduction is a sum in the input dtype, but `_block`'s value set makes every
    intermediate representable, so any difference at all is a wrong reduction
    or a wrong slice, not rounding.
    """
    torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=lambda built: f"{where}: {built}")


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
    """Ring-order sequential sum for group position `pos`, rounded every step.

    The order this op reduces in (certified by
    test_reduction_is_deterministic_and_accumulates_in_the_input_dtype):
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
    test_cuda_graph_capture_of_a_first_call_raises), and torch wants the work
    warmed on a non-default stream.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(reps):
            body()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    COMM.Barrier()


def _capture_per_batch_size(
    sites: int = 1, seed: int = 11000
) -> Tuple[Dict[int, List[torch.Tensor]], Dict[int, Any], Dict[int, List[torch.Tensor]]]:
    """One graph per GRAPH_BATCH_SIZES entry, one shared pool, all left alive.

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
    for rows in GRAPH_BATCH_SIZES:
        xs = [_input(RANK, [rows] * WORLD, seed + 13 * s) for s in range(sites)]

        def body(xs: List[torch.Tensor] = xs) -> List[torch.Tensor]:
            return [reducescatter(x, None, GROUP) for x in xs]

        _warm_up_off_capture_stream(body)
        graph = torch.cuda.CUDAGraph()
        ctx = torch.cuda.graph(graph) if pool is None else torch.cuda.graph(graph, pool=pool)
        with ctx:
            outputs[rows] = body()
        # Without this a capture that returned nothing would make every
        # `_verify_replay` below an empty loop, i.e. a vacuous pass.
        assert len(outputs[rows]) == sites, (
            f"rows={rows}: captured {len(outputs[rows])} outputs, expected {sites}"
        )
        if pool is None:
            pool = graph.pool()
        inputs[rows], graphs[rows] = xs, graph
        COMM.Barrier()
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
        x.copy_(_input(RANK, [rows] * WORLD, seed + 13 * site))


def _verify_replay(
    outputs: Dict[int, List[torch.Tensor]], rows: int, seed: int, where: str
) -> None:
    """Every site's captured output tensor holds this payload's result, bitwise."""
    for site, out in enumerate(outputs[rows]):
        _assert_bitwise(out, _ref([rows] * WORLD, RANK, seed + 13 * site), f"{where} site={site}")


def test_cuda_graph_capture_of_a_first_call_raises() -> None:
    """A group's first-ever call cannot be captured; the build inside fails.

    Must run before anything else touches GROUP — the failure is specifically
    the NCCL communicator being built inside the capture, and it only happens
    once per rank set per process. The failure is survivable, and the two
    assertions after it are the workaround a caller needs: one eager call
    first, then capture.
    """
    rows = 16
    sizes = [rows] * WORLD
    x = _input(RANK, sizes, 100)
    ref = _ref(sizes, RANK, 100)
    COMM.Barrier()

    graph = torch.cuda.CUDAGraph()
    resting_stream = torch.cuda.current_stream()
    inner: Optional[BaseException] = None
    outer: Optional[BaseException] = None
    try:
        with torch.cuda.graph(graph):
            try:
                reducescatter(x, None, GROUP)
            except RuntimeError as exc:
                inner = exc
                raise
    except RuntimeError as exc:
        outer = exc
    finally:
        # torch's context manager ends the capture before restoring the
        # stream, so a capture that fails at capture_end leaves its own
        # stream current. Put the resting one back by hand.
        torch.cuda.set_stream(resting_stream)
    del graph

    assert inner is not None, "the op accepted a first call inside a capture"
    assert "NCCL error" in str(inner) and "opUtils.cpp" in str(inner), str(inner)
    assert outer is not None, "capturing a group's first-ever call was accepted"
    assert "operation failed due to a previous error during capture" in str(outer), str(outer)

    # Survivable: the group works eagerly straight afterwards...
    torch.cuda.synchronize()
    COMM.Barrier()
    _assert_bitwise(reducescatter(x, None, GROUP), ref, "eager after failed capture")
    COMM.Barrier()

    # ...and a capture taken after that replays correctly.
    _warm_up_off_capture_stream(lambda: reducescatter(x, None, GROUP))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = reducescatter(x, None, GROUP)
    x.copy_(_input(RANK, sizes, 700))
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(captured, _ref(sizes, RANK, 700), "replay after warm-up")
    del graph
    COMM.Barrier()


def test_uniform_reduce_scatter() -> None:
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
        COMM.Barrier()


def test_ragged_reduce_scatter() -> None:
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
        COMM.Barrier()


def test_output_is_fresh_and_input_is_untouched() -> None:
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
    COMM.Barrier()


def test_trailing_dims_are_preserved() -> None:
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
        COMM.Barrier()


def test_dtypes_reduce_arithmetically() -> None:
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
        COMM.Barrier()

        xr = _input(RANK, ragged, seed, dtype)
        outr = reducescatter(xr, ragged, GROUP)
        assert outr.dtype is dtype and outr.shape == (ragged[RANK], HIDDEN)
        _assert_bitwise(outr, _ref(ragged, RANK, seed, dtype), f"ragged {dtype}")
        COMM.Barrier()


def test_group_selects_a_rank_subset() -> None:
    """`group` names MPI session ranks; the slice index is the position in it.

    The second subset is the discriminating one: it excludes rank 0, so a rank
    that indexed the split by its MPI rank instead of by its position in the
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
            pos = subset.index(RANK)
            out = reducescatter(_input(RANK, sizes, seed), None, subset)
            assert out.shape == (rows, HIDDEN), out.shape
            _assert_bitwise(out, _ref(sizes, pos, seed, ranks=subset), f"subset {subset} uniform")
            ragged = [2, 6] if len(subset) == 2 else [2] * len(subset)
            outr = reducescatter(_input(RANK, ragged, seed), ragged, subset)
            assert outr.shape == (ragged[pos], HIDDEN), outr.shape
            _assert_bitwise(outr, _ref(ragged, pos, seed, ranks=subset), f"subset {subset} ragged")
        COMM.Barrier()


def test_group_order_does_not_change_the_output() -> None:
    """The split is ordered by ascending rank, whatever order `group` lists."""
    rows, seed = 4, 7200
    sizes = [rows] * WORLD
    x = _input(RANK, sizes, seed)
    out = reducescatter(x, None, list(reversed(GROUP)))
    _assert_bitwise(out, _ref(sizes, RANK, seed), "reversed group list")
    COMM.Barrier()


def test_reduction_is_deterministic_and_accumulates_in_the_input_dtype() -> None:
    """What the sum is, exactly — the fact a target's accuracy gate rests on.

    On payloads with no exactness property (standard normal, cast to bf16):

    1. It is **deterministic**: eight identical eager calls return bitwise
       identical tensors, and so do eight replays of a captured call.
    2. It is **not** the correctly rounded exact sum. The op accumulates in the
       input dtype, so about a third of the elements differ from an fp32
       reduction rounded once.
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

        # (2) not the correctly rounded exact sum
        differs = (outs[0] != exact.to(torch.bfloat16)).float().mean().item()
        assert differs > 0.05, (
            f"rows={rows}: only {differs:.4f} of elements differ from an fp32 "
            "reduction — the op may have started accumulating in fp32"
        )

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
        COMM.Barrier()

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
    COMM.Barrier()


def test_uniform_form_and_an_explicit_even_split_agree_bitwise() -> None:
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
        COMM.Barrier()


def test_round_trip_with_a_gather_returns_each_rank_its_own_rows() -> None:
    """The attention-DP MoE round trip, end to end, against a local reference.

    Gather every rank's tokens, let each rank apply its own expert window to
    the whole token set, reduce-scatter the four partial outputs. The claim
    under test is that the scatter undoes the gather: rank i gets back exactly
    the rows rank i contributed, holding the sum of the four windows.

    `torch.ops.trtllm.allgather` appears here as a **fixture**, to build the
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
    gathered = torch.ops.trtllm.allgather(mine, sizes, GROUP)
    torch.cuda.synchronize()
    _assert_bitwise(gathered, torch.cat(own, dim=0), "gather fixture")

    scale = float(RANK + 1) / 4.0
    partial = (gathered.float() * scale).to(torch.bfloat16)
    out = reducescatter(partial, sizes, GROUP)
    torch.cuda.synchronize()
    total_scale = sum((r + 1) / 4.0 for r in range(WORLD))
    assert out.shape == mine.shape, (out.shape, mine.shape)
    _assert_bitwise(out, (mine.float() * total_scale).to(torch.bfloat16), "round trip")
    COMM.Barrier()

    # The bare identity: reduce-scattering a gather returns WORLD copies summed.
    out2 = reducescatter(torch.ops.trtllm.allgather(mine, sizes, GROUP), sizes, GROUP)
    torch.cuda.synchronize()
    _assert_bitwise(out2, (mine.float() * WORLD).to(torch.bfloat16), "rs(ag(x))")
    COMM.Barrier()


def test_cuda_graph_at_every_engine_batch_size() -> None:
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
        COMM.Barrier()
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
        COMM.Barrier()
    # Discrimination for a gate of exactly 0: what a replay that did not happen
    # leaves behind is the previous payload's result. The two consecutive
    # references have to be far apart for that to be caught, which this checks.
    prev = _ref([16] * WORLD, RANK, seed - 1000)
    last = _ref([16] * WORLD, RANK, seed)
    margin = (last.float() - prev.float()).abs().max().item()
    assert margin > 8.0, f"stale-replay margin {margin}"
    del graphs, outputs, inputs
    COMM.Barrier()


def test_cuda_graph_one_site_per_moe_layer_in_every_batch_size_graph() -> None:
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
        COMM.Barrier()
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
        COMM.Barrier()
    del graphs, outputs, inputs
    COMM.Barrier()


def test_cuda_graph_replays_survive_eager_calls_of_other_shapes() -> None:
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
        COMM.Barrier()
    del graphs, outputs, inputs
    COMM.Barrier()


def test_cuda_graph_holds_the_sizes_vector_it_captured() -> None:
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
        COMM.Barrier()

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
            COMM.Barrier()
    del graphs, captured, buffers
    COMM.Barrier()


def test_the_gate_discriminates_a_wrong_reduce_scatter() -> None:
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
    COMM.Barrier()


def test_calls_pair_by_position_and_a_swapped_pair_realigns() -> None:
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
        COMM.Barrier()

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
        COMM.Barrier()

        _assert_bitwise(
            reducescatter(_input(RANK, split, base + 900), sizes, GROUP),
            _ref(split, RANK, base + 900),
            f"{tag}: plain call after a swapped pair",
        )
        COMM.Barrier()


def test_a_mispaired_result_is_deterministic_rather_than_noise() -> None:
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
    COMM.Barrier()

    seen: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(4):
        pos0, pos1 = _swapped_pair(first, second, None)
        torch.cuda.synchronize()
        seen.append((pos0, pos1))
        COMM.Barrier()
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
    COMM.Barrier()

    _assert_bitwise(
        reducescatter(_input(RANK, [rows] * WORLD, 163000), None, GROUP),
        _ref([rows] * WORLD, RANK, 163000),
        "plain call after four swapped pairs",
    )
    COMM.Barrier()


def test_an_extra_call_on_one_rank_misaligns_until_the_counts_match() -> None:
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
    COMM.Barrier()

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
    COMM.Barrier()

    _assert_bitwise(
        reducescatter(_input(RANK, sizes, 178000), None, GROUP),
        _ref(sizes, RANK, 178000),
        "plain call after the call counts were equalized",
    )
    COMM.Barrier()


def test_mispaired_against_an_all_gather_of_equal_byte_count_corrupts_silently() -> None:
    """A different collective at the same byte count is not detected either.

    What pairs is position, not the identity of the op: with rank 0 issuing
    `allgather` where the others issue `reducescatter`, both calls return, at
    the shapes their own arguments imply, with finite values and no error. Two
    calls of this pair move the same number of elements as NCCL counts them —
    an even reduce-scatter of `[WORLD*n, H]` and an all-gather of `[n, H]` both
    carry `n*H` — which is what keeps it from hanging. The unequal-byte-count
    version of exactly this mispairing wedges instead, and is certified by the
    launcher's sub-job rather than here.

    Measured on the certified path: the reduce-scatter comes back 0.984-0.989
    wrong and the gather 0.246-0.741 wrong (rank-dependent, since a gather's
    blocks are spoiled individually), both bitwise stable across repeats and
    across processes. The floors below are 0.9 and 0.2. `allgather` is a
    fixture here, never a reference — every reference in this file is
    arithmetic.

    The pair realigns: a plain reduce-scatter afterwards is bitwise correct,
    checked after each repeat.
    """
    rows = 16
    sizes = [rows] * WORLD
    rs_seed, ag_seed = 190000, 190500
    rs_in = _input(RANK, sizes, rs_seed)
    ag_in = _block(RANK, 0, rows, ag_seed)
    assert rs_in.shape[0] // WORLD * HIDDEN == ag_in.numel(), "byte counts differ"
    rs_ref = _ref(sizes, RANK, rs_seed)
    ag_ref = torch.cat([_block(r, 0, rows, ag_seed) for r in range(WORLD)], dim=0)
    COMM.Barrier()

    seen: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for repeat in range(2):
        if RANK == 0:
            got_ag = torch.ops.trtllm.allgather(ag_in, None, GROUP)
            got_rs = reducescatter(rs_in, None, GROUP)
        else:
            got_rs = reducescatter(rs_in, None, GROUP)
            got_ag = torch.ops.trtllm.allgather(ag_in, None, GROUP)
        torch.cuda.synchronize()
        assert got_rs.shape == (rows, HIDDEN), got_rs.shape
        assert got_ag.shape == (rows * WORLD, HIDDEN), got_ag.shape
        assert bool(torch.isfinite(got_rs.float()).all()), "reduce-scatter not finite"
        assert bool(torch.isfinite(got_ag.float()).all()), "gather not finite"
        rs_wrong = _fraction_wrong(got_rs, rs_ref)
        ag_wrong = _fraction_wrong(got_ag, ag_ref)
        assert rs_wrong > 0.9, f"repeat {repeat}: reduce-scatter {rs_wrong:.4f} wrong"
        assert ag_wrong > 0.2, f"repeat {repeat}: gather {ag_wrong:.4f} wrong"
        seen.append((got_rs, got_ag))
        COMM.Barrier()

        _assert_bitwise(
            reducescatter(_input(RANK, sizes, 191000 + repeat), None, GROUP),
            _ref(sizes, RANK, 191000 + repeat),
            f"plain call after cross-op repeat {repeat}",
        )
        COMM.Barrier()

    _assert_bitwise(seen[1][0], seen[0][0], "cross-op reduce-scatter across repeats")
    _assert_bitwise(seen[1][1], seen[0][1], "cross-op gather across repeats")
    COMM.Barrier()


def test_the_stream_the_call_lands_on_is_not_part_of_the_match() -> None:
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
        COMM.Barrier()

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
        COMM.Barrier()


def test_float8_is_summed_as_raw_bytes() -> None:
    """float8_e4m3fn is accepted and reduced as unsigned bytes, not as floats.

    Measured, and the reason the wrapper rejects the dtype: an all-gather moves
    fp8 activations correctly, so a caller doing post-quantization dispatch
    would reasonably expect the return leg to work too. It does not — the sum
    is of the e4m3 *bit patterns*, wrapping at 256.
    """
    rows = 4
    x = torch.full((rows * WORLD, 8), float(RANK + 1), dtype=torch.float32, device="cuda").to(
        torch.float8_e4m3fn
    )
    raw = torch.ops.trtllm.reducescatter(x, None, GROUP)
    torch.cuda.synchronize()
    byte_sum = torch.zeros(rows, 8, dtype=torch.int64, device="cuda")
    for r in range(WORLD):
        one = torch.full((rows, 8), float(r + 1), dtype=torch.float32, device="cuda").to(
            torch.float8_e4m3fn
        )
        byte_sum += one.view(torch.uint8).to(torch.int64)
    _assert_bitwise(
        raw.view(torch.uint8),
        (byte_sum % 256).to(torch.uint8),
        "float8 reduced as bytes",
    )
    # ...and that is nowhere near the float sum it looks like it should be.
    assert abs(raw.float()[0, 0].item() - sum(range(1, WORLD + 1))) > 1.0
    COMM.Barrier()

    try:
        reducescatter(x, None, GROUP)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrapper accepted a float8_e4m3fn input")
    COMM.Barrier()


def test_unsupported_dtypes_raise_and_poison_every_later_collective() -> None:
    """fp64 and the other float8 formats raise — and the raise is terminal.

    Runs last, and has to: the raise leaves NCCL's group state unbalanced (the
    op opens a group before it converts the dtype and never closes it), so
    every collective the process issues afterwards returns garbage. Measured
    here for this op; measured in probing for the sibling all-gather and
    all-reduce on the same group too, so it is the process that is finished,
    not this entry's communicator.

    The last assertion therefore asserts a defect. If it ever fails, upstream
    has balanced the group and the contract's *Preconditions* wording — "not
    survivable" — has to be re-measured, not this gate loosened.
    """
    rows, seed = 8, 96000
    sizes = [rows] * WORLD
    _assert_bitwise(
        reducescatter(_input(RANK, sizes, seed), None, GROUP),
        _ref(sizes, RANK, seed),
        "baseline before the unsupported dtypes",
    )
    COMM.Barrier()

    for dtype in (torch.float64, torch.float8_e5m2):
        x = torch.ones(4 * WORLD, 8, dtype=torch.float32, device="cuda").to(dtype)
        try:
            reducescatter(x, None, GROUP)
        except RuntimeError as exc:
            assert "unsupported data type" in str(exc), str(exc)
        else:
            raise AssertionError(f"{dtype} was accepted")
        COMM.Barrier()

    after = reducescatter(_input(RANK, sizes, seed), None, GROUP)
    torch.cuda.synchronize()
    wrong = (after != _ref(sizes, RANK, seed)).float().mean().item()
    assert wrong > 0.5, (
        f"only {wrong:.4f} of elements are wrong after the unsupported-dtype "
        "raise — the raise used to destroy every later collective in the "
        "process; re-measure the contract's dtype precondition"
    )
    COMM.Barrier()


def test_wrapper_guards_a_non_contiguous_input() -> None:
    """The wrapper's assert stands where the op itself is silently wrong."""
    rows, seed = 8, 92000
    sizes = [rows] * WORLD
    values = _input(RANK, sizes, seed)
    padded = torch.zeros(rows * WORLD, HIDDEN * 2, dtype=torch.bfloat16, device="cuda")
    padded[:, :HIDDEN] = values
    view = padded[:, :HIDDEN]
    assert torch.equal(view, values) and not view.is_contiguous()

    raw = torch.ops.trtllm.reducescatter(view, None, GROUP)
    torch.cuda.synchronize()
    # It reduced `rows * WORLD * HIDDEN` packed elements from the start of
    # `padded`, which is the first half of the rows interleaved with the zero
    # half — so the result is neither the right sum nor the right rows.
    assert raw.shape == (rows, HIDDEN), raw.shape
    ref = _ref(sizes, RANK, seed)
    assert (raw != ref).float().mean().item() > 0.4
    COMM.Barrier()

    try:
        reducescatter(view, None, GROUP)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrapper accepted a non-contiguous input")
    COMM.Barrier()


def test_wrapper_guards_a_zero_dim_input() -> None:
    """A 0-d input segfaults inside the op, so the wrapper stops it first.

    The op is deliberately not called here: the crash is in
    `ReducescatterOp::run_list` and kills every rank in the job.
    """
    try:
        reducescatter(torch.tensor(1.0, device="cuda"), None, GROUP)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrapper accepted a 0-d input")
    COMM.Barrier()


def test_wrapper_guards_a_sizes_list_of_the_wrong_length() -> None:
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
            reducescatter(_input(RANK, sizes, seed), bad, GROUP)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"wrapper accepted sizes of length {len(bad)}")
    COMM.Barrier()


def test_wrapper_guards_a_split_that_does_not_cover_the_input() -> None:
    """Rows the split does not reach are silently dropped, not flagged.

    Two ways to get there, both exercised on the raw op because both are
    survivable: a row count that `len(group)` does not divide, and a `sizes`
    vector summing to less than the input holds. A vector summing to *more*
    than the input holds is not exercised — the op reads past the end of the
    buffer — and the wrapper rejects it on the same assert.
    """
    seed = 94000
    odd = WORLD * 2 + 1
    x = _input(RANK, [1] * odd, seed)  # `odd` blocks of one row each
    assert x.shape == (odd, HIDDEN)
    raw = torch.ops.trtllm.reducescatter(x, None, GROUP)
    torch.cuda.synchronize()
    assert raw.shape == (odd // WORLD, HIDDEN), raw.shape  # the last row vanished
    COMM.Barrier()

    short_split = [2] * WORLD
    y = _input(RANK, [4] * WORLD, seed)
    raw2 = torch.ops.trtllm.reducescatter(y, short_split, GROUP)
    torch.cuda.synchronize()
    assert raw2.shape == (2, HIDDEN), raw2.shape
    COMM.Barrier()

    for x_, sizes_ in ((x, None), (y, short_split), (y, [6] * WORLD)):
        try:
            reducescatter(x_, sizes_, GROUP)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"wrapper accepted an uncovered split {sizes_}")
    COMM.Barrier()


def test_the_group_still_works_after_the_negative_tests() -> None:
    """The raises above leave the communicator usable — checked, not assumed."""
    rows, seed = 8, 95000
    sizes = [rows] * WORLD
    out = reducescatter(_input(RANK, sizes, seed), None, GROUP)
    _assert_bitwise(out, _ref(sizes, RANK, seed), "after the negative tests")
    COMM.Barrier()


TESTS = (
    # Stays first: it is the only test that can observe GROUP's first-ever
    # call, and every later test needs the communicator it builds.
    test_cuda_graph_capture_of_a_first_call_raises,
    test_uniform_reduce_scatter,
    test_ragged_reduce_scatter,
    test_output_is_fresh_and_input_is_untouched,
    test_trailing_dims_are_preserved,
    test_dtypes_reduce_arithmetically,
    test_group_selects_a_rank_subset,
    test_group_order_does_not_change_the_output,
    test_reduction_is_deterministic_and_accumulates_in_the_input_dtype,
    test_uniform_form_and_an_explicit_even_split_agree_bitwise,
    test_round_trip_with_a_gather_returns_each_rank_its_own_rows,
    test_cuda_graph_at_every_engine_batch_size,
    test_cuda_graph_one_site_per_moe_layer_in_every_batch_size_graph,
    test_cuda_graph_replays_survive_eager_calls_of_other_shapes,
    test_cuda_graph_holds_the_sizes_vector_it_captured,
    test_the_gate_discriminates_a_wrong_reduce_scatter,
    # The call-order block. Each of these deliberately disagrees about call
    # order and each restores alignment before it returns — the plain call
    # every one of them ends on is what proves it.
    test_calls_pair_by_position_and_a_swapped_pair_realigns,
    test_a_mispaired_result_is_deterministic_rather_than_noise,
    test_an_extra_call_on_one_rank_misaligns_until_the_counts_match,
    test_mispaired_against_an_all_gather_of_equal_byte_count_corrupts_silently,
    test_the_stream_the_call_lands_on_is_not_part_of_the_match,
    test_float8_is_summed_as_raw_bytes,
    test_wrapper_guards_a_non_contiguous_input,
    test_wrapper_guards_a_zero_dim_input,
    test_wrapper_guards_a_sizes_list_of_the_wrong_length,
    test_wrapper_guards_a_split_that_does_not_cover_the_input,
    test_the_group_still_works_after_the_negative_tests,
    # Stays last: the raise it asserts leaves every later collective in the
    # process returning garbage, so nothing can run after it.
    test_unsupported_dtypes_raise_and_poison_every_later_collective,
)


def _run_one_rank() -> int:
    """Body of one MPI rank: run every test, abort the job if any fails."""
    global COMM, RANK, WORLD, GROUP, reducescatter
    from mpi4py import MPI

    from . import reducescatter as entry

    reducescatter = entry.reducescatter
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    WORLD = COMM.Get_size()
    GROUP = list(range(WORLD))
    assert WORLD >= 2, f"a collective needs at least 2 ranks, got {WORLD}"
    torch.cuda.set_device(RANK)

    for test in TESTS:
        try:
            test()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {test.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # Abort rather than return: a rank that leaves a collective early
            # wedges every other rank in it.
            COMM.Abort(1)
    COMM.Barrier()
    print(f"[rank {RANK}] {len(TESTS)} tests passed", flush=True)
    return 0


def _mark(kind: str) -> None:
    """Record that this rank reached `kind`, durably, for the launcher to read."""
    path = os.path.join(os.environ["RS_WEDGE_MARKS"], f"{kind}.{RANK}")
    with open(path, "w") as handle:
        handle.write(f"{time.time():.3f}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _wedge_watchdog() -> None:
    """End this rank once the mispaired pair has had its chance to return.

    Runs in a daemon thread, which can only make progress because the rank is
    blocked in `torch.cuda.synchronize()` with the GIL released. There is no
    gentler exit: the rank is waiting on a NCCL kernel that will never
    complete, so no exception can be raised into it and no collective can be
    cancelled.
    """
    time.sleep(WEDGE_GRACE_S)
    _mark("wedged")
    os._exit(7)


def _run_wedge_rank() -> int:
    """Body of one rank of the sub-job that certifies the wedge.

    Same mispairing as
    test_mispaired_against_an_all_gather_of_equal_byte_count_corrupts_silently,
    with one difference: the two calls carry **different** element counts
    (`rows*HIDDEN` against `5*HIDDEN`). That is the case NCCL cannot serve out
    of the buffers it was given, and it hangs rather than returning wrong data.

    The rank marks the file system either side of the pair so the launcher can
    tell a wedge from a crash: every rank reaching `ready` and none reaching
    `returned` is the wedge, and it means nothing without the first half.
    """
    global COMM, RANK, WORLD, GROUP, reducescatter
    from mpi4py import MPI

    from . import reducescatter as entry

    reducescatter = entry.reducescatter
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    WORLD = COMM.Get_size()
    GROUP = list(range(WORLD))
    torch.cuda.set_device(RANK)

    # One aligned call first: the group's communicator has to already exist, so
    # that what wedges below is the mispaired pair and not the bootstrap.
    rows, seed = 16, 970000
    sizes = [rows] * WORLD
    _assert_bitwise(
        reducescatter(_input(RANK, sizes, seed), None, GROUP),
        _ref(sizes, RANK, seed),
        "wedge sub-job warm-up",
    )
    COMM.Barrier()

    rs_in = _input(RANK, sizes, seed + 1000)
    gather_in = _block(RANK, 0, 5, seed + 2000)
    assert gather_in.numel() != rs_in.shape[0] // WORLD * HIDDEN, "counts must differ"
    _mark("ready")
    COMM.Barrier()
    # Started after the barrier, so the grace covers the pair and nothing else.
    threading.Thread(target=_wedge_watchdog, daemon=True).start()

    if RANK == 0:
        torch.ops.trtllm.allgather(gather_in, None, GROUP)
        reducescatter(rs_in, None, GROUP)
    else:
        reducescatter(rs_in, None, GROUP)
        torch.ops.trtllm.allgather(gather_in, None, GROUP)
    torch.cuda.synchronize()
    _mark("returned")
    print(f"[rank {RANK}] the mispaired pair RETURNED", flush=True)
    return 0


def _mpirun(world_size: int, flag: str, env: Optional[Dict[str, str]] = None) -> Any:
    """Start one mpirun job in its own process group, so it can be killed."""
    command = [
        "mpirun",
        "-n",
        str(world_size),
        sys.executable,
        "-m",
        "tensorrt_llm._torch.staircase.catalog.comm.reducescatter_test",
        flag,
    ]
    print(f"[launcher] {' '.join(command)}", flush=True)
    return subprocess.Popen(command, start_new_session=True, env=env)


def _certify_the_unequal_byte_count_wedge(world_size: int) -> None:
    """Second job: the one call-order divergence that hangs instead of lying.

    It cannot be a `TESTS` entry, because the job that runs it never reports.
    So it runs on its own, and the evidence is the marks its ranks leave: all
    of them entered the mispaired pair, none came out of it within
    WEDGE_GRACE_S, and the job ended by its own watchdog rather than by
    completing. A crash before the pair fails this too — the `ready` count is
    what separates the two.
    """
    marks = tempfile.mkdtemp(prefix="reducescatter_wedge_")
    started = time.time()
    process = _mpirun(world_size, _WEDGE_FLAG, env=dict(os.environ, RS_WEDGE_MARKS=marks))
    try:
        code: Optional[int] = process.wait(timeout=WEDGE_CAP_S)
        ended = "its own watchdog"
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        code, ended = None, f"the launcher, after {WEDGE_CAP_S}s"

    seen = os.listdir(marks)
    ready = [m for m in seen if m.startswith("ready.")]
    returned = [m for m in seen if m.startswith("returned.")]
    wedged = [m for m in seen if m.startswith("wedged.")]
    for name in seen:
        os.remove(os.path.join(marks, name))
    os.rmdir(marks)
    print(
        f"[launcher] wedge sub-job: exit={code} ready={len(ready)} "
        f"returned={len(returned)} wedged={len(wedged)} "
        f"ended by {ended} in {time.time() - started:.0f}s",
        flush=True,
    )

    assert len(ready) == world_size, (
        f"only {len(ready)}/{world_size} ranks reached the mispaired pair — the "
        "sub-job failed before it could be mispaired, so nothing was certified"
    )
    assert not returned, (
        "the mispaired pair returned at unequal byte counts; it used to wedge, "
        "so the contract's call-order table has to be re-measured"
    )
    assert len(wedged) == world_size, (
        f"{len(wedged)}/{world_size} ranks were still inside the pair after "
        f"{WEDGE_GRACE_S}s; expected every one of them"
    )


def _spawn_ranks() -> None:
    """Re-exec this file under mpirun, one rank per claimed device."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, (
        "set CUDA_VISIBLE_DEVICES to the devices this run owns, "
        "e.g. export CUDA_VISIBLE_DEVICES=0,1,2,3"
    )
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size >= 2, (
        f"CUDA_VISIBLE_DEVICES names {world_size} device(s); a collective test needs at least 2"
    )
    # Own process group so the deadline can kill wedged grandchildren too.
    process = _mpirun(world_size, _WORKER_FLAG)
    try:
        code = process.wait(timeout=DEADLINE_S)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise AssertionError(
            f"the {world_size}-rank run did not finish in {DEADLINE_S}s (wedged)"
        ) from None
    assert code == 0, f"the {world_size}-rank run exited {code}"
    _certify_the_unequal_byte_count_wedge(world_size)


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    if _WEDGE_FLAG in sys.argv:
        sys.exit(_run_wedge_rank())
    _spawn_ranks()
    print("OK")
