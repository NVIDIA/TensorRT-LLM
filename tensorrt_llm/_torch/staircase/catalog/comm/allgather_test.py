# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the allgather catalog entry.

A collective cannot be exercised in one process, so this script is its own
launcher: run it plainly and it re-executes itself under `mpirun` with one
rank per device named in CUDA_VISIBLE_DEVICES, under a deadline the parent
enforces by killing the whole process group. A wedged collective hangs
rather than raising — and this op wedges rather than raising for every
rank-argument-disagreement case probed — so the deadline is what keeps a
broken kernel from taking the calling run down with it.

Beyond the lockstep surface (uniform / ragged / dtypes / CUDA graphs) the
file covers what a serving engine adds: the engine's own attention-DP
synchronisation interleaved with the op, the engine's stream switching, and
what disagreeing on call order actually does.

    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python catalog/comm/allgather_test.py
"""

import os
import random
import signal
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

assert torch.cuda.is_available(), "allgather requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 1200

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
# pulls in tensorrt_llm, which calls MPI_Init at import, and an
# MPI-initialized process cannot launch `mpirun` — measured on this host,
# mpirun then exits 1 with no output from any rank.
allgather: Any = None
COMM: Any = None
RANK = 0
WORLD = 1
GROUP: List[int] = [0]
# The engine's own cross-rank synchronisation object, built from the real
# Mapping a serving engine builds under attention data parallelism. Bound in
# the rank body for the same reason as the rest.
DIST: Any = None
MPI: Any = None


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
    return torch.cat(
        [_payload(r, sizes[i], seed, dtype, trailing) for i, r in enumerate(ranks)],
        dim=0,
    )


def _assert_bitwise(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    """Gate of exactly zero: the op moves bytes, so nothing may differ.

    Tightened from the default dtype-aware tolerances rather than loosened —
    a gather computes nothing, so any difference at all is a wrong gather.
    """
    if out.dtype is torch.float8_e4m3fn:
        # torch.testing cannot compare float8 tensors; for a pure data move the
        # byte pattern is the honest gate anyway.
        out, ref = out.view(torch.uint8), ref.view(torch.uint8)
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
        x.copy_(_payload(RANK, rows, seed + 13 * site))


def _verify_replay(
    outputs: Dict[int, List[torch.Tensor]], rows: int, seed: int, where: str
) -> None:
    """Every site's captured output tensor holds this payload's gather, bitwise."""
    for site, out in enumerate(outputs[rows]):
        _assert_bitwise(out, _gather_ref([rows] * WORLD, seed + 13 * site), f"{where} site={site}")


def test_cuda_graph_capture_of_a_first_call_raises() -> None:
    """A group's first-ever call cannot be captured; the build inside fails.

    Must run before anything else touches GROUP — the failure is specifically
    the NCCL communicator being built inside the capture, and it only happens
    once per rank set per process. The failure is survivable, and the two
    assertions after it are the workaround a caller needs: one eager call
    first, then capture.
    """
    rows = 16
    x = _payload(RANK, rows, 100)
    ref = _gather_ref([rows] * WORLD, 100)
    COMM.Barrier()

    graph = torch.cuda.CUDAGraph()
    resting_stream = torch.cuda.current_stream()
    raised: Optional[BaseException] = None
    try:
        with torch.cuda.graph(graph):
            allgather(x, None, GROUP)
    except RuntimeError as exc:
        raised = exc
    finally:
        # torch's context manager ends the capture before restoring the
        # stream, so a capture that fails at capture_end leaves its own
        # stream current. Put the resting one back by hand.
        torch.cuda.set_stream(resting_stream)
    del graph

    assert raised is not None, "capturing a group's first-ever call was accepted"
    chain, exc = [], raised
    while exc is not None and len(chain) < 10:
        chain.append(str(exc))
        exc = exc.__context__
    text = "\n".join(chain)
    assert "operation failed due to a previous error during capture" in text, text
    assert "NCCL error" in text and "opUtils.cpp" in text, text

    # Survivable: the group works eagerly straight afterwards...
    torch.cuda.synchronize()
    COMM.Barrier()
    _assert_bitwise(allgather(x, None, GROUP), ref, "eager after failed capture")
    COMM.Barrier()

    # ...and a capture taken after that warm-up replays correctly.
    _warm_up_off_capture_stream(lambda: allgather(x, None, GROUP))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = allgather(x, None, GROUP)
    x.copy_(_payload(RANK, rows, 700))
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(captured, _gather_ref([rows] * WORLD, 700), "replay after warm-up")
    del graph
    COMM.Barrier()


def test_uniform_gather() -> None:
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
        COMM.Barrier()


def test_ragged_gather() -> None:
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
        COMM.Barrier()


def test_output_is_fresh_and_input_is_untouched() -> None:
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
    COMM.Barrier()


def test_trailing_dims_are_preserved() -> None:
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
        COMM.Barrier()


def test_dtypes_move_bitwise() -> None:
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
        COMM.Barrier()

        xr = _payload(RANK, sizes[RANK], seed, dtype)
        outr = allgather(xr, sizes, GROUP)
        assert outr.dtype is dtype and outr.shape == (sum(sizes), HIDDEN)
        _assert_bitwise(outr, _gather_ref(sizes, seed, dtype), f"ragged {dtype}")
        COMM.Barrier()


def test_group_selects_a_rank_subset() -> None:
    """`group` names MPI session ranks; ranks outside it must not call."""
    subset = [0, 1]
    rows, seed = 4, 7100
    if RANK in subset:
        out = allgather(_payload(RANK, rows, seed), None, subset)
        assert out.shape == (rows * len(subset), HIDDEN), out.shape
        _assert_bitwise(out, _gather_ref([rows] * len(subset), seed, ranks=subset), "subset gather")
    COMM.Barrier()


def test_group_order_does_not_change_the_output_order() -> None:
    """The result is ordered by ascending rank, whatever order `group` lists."""
    rows, seed = 4, 7200
    x = _payload(RANK, rows, seed)
    out = allgather(x, None, list(reversed(GROUP)))
    _assert_bitwise(out, _gather_ref([rows] * WORLD, seed), "reversed group list")
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
    COMM.Barrier()


def test_cuda_graph_one_site_per_moe_layer_in_every_batch_size_graph() -> None:
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
        x = _payload(RANK, sizes[RANK], seed)
        _warm_up_off_capture_stream(lambda x=x, sizes=sizes: allgather(x, sizes, GROUP))
        graph = torch.cuda.CUDAGraph()
        ctx = torch.cuda.graph(graph) if pool is None else torch.cuda.graph(graph, pool=pool)
        with ctx:
            captured[tag] = allgather(x, sizes, GROUP)
        pool = graph.pool() if pool is None else pool
        buffers[tag], graphs[tag] = x, graph
        assert captured[tag].shape == (sum(sizes), HIDDEN), captured[tag].shape
        COMM.Barrier()

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
            COMM.Barrier()
    del graphs, captured, buffers
    COMM.Barrier()


def test_the_gate_discriminates_a_wrong_gather() -> None:
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
    COMM.Barrier()


def test_the_engines_own_cross_rank_step_is_not_on_this_communicator() -> None:
    """What a serving engine synchronises with, next to what this op uses.

    Under attention data parallelism the engine agrees the per-rank token
    counts once per step, and that is the only cross-rank traffic it issues in
    this configuration. It does it with `MPIDist.tp_allgather` — a **host-side
    MPI** collective on a sub-communicator the engine builds itself — not with
    a device collective, and not on the MPI session communicator this op
    resolves `group` against. The two never share an ordering.
    """
    assert type(DIST).__name__ == "MPIDist", type(DIST).__name__
    tp_comm = DIST.tp_comm
    assert isinstance(tp_comm, MPI.Comm), type(tp_comm)
    # Same ranks in the same order, but a communicator of its own: MPI's own
    # comparison says congruent, never identical.
    assert MPI.Comm.Compare(tp_comm, MPI.COMM_WORLD) != MPI.IDENT
    assert tp_comm.Get_size() == WORLD and tp_comm.Get_rank() == RANK
    counts = [3 + 2 * r for r in range(WORLD)]
    assert list(DIST.tp_allgather(counts[RANK])) == counts
    COMM.Barrier()


def test_interleaved_with_the_engines_attention_dp_synchronisation() -> None:
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
        # The engine's step-level sync, on the object the engine builds. It is
        # a host collective, so the device work queued above it stays in
        # flight across it — that interleaving is the point of this test.
        observed = list(DIST.tp_allgather(counts[RANK]))
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
    COMM.Barrier()


def test_the_stream_the_call_lands_on_is_not_part_of_the_match() -> None:
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
        COMM.Barrier()


def test_wrapper_guards_a_non_contiguous_input() -> None:
    """The wrapper's assert stands where the op itself is silently wrong."""
    rows, seed = 8, 92000
    values = _payload(RANK, rows, seed)
    padded = torch.zeros(rows, HIDDEN * 2, dtype=torch.bfloat16, device="cuda")
    padded[:, :HIDDEN] = values
    view = padded[:, :HIDDEN]
    assert torch.equal(view, values) and not view.is_contiguous()

    raw = torch.ops.trtllm.allgather(view, None, GROUP)
    torch.cuda.synchronize()
    # It read `rows * HIDDEN` packed elements from the start of `padded`, which
    # is this rank's first rows/2 values interleaved with the zero half — so
    # every second row of every rank's block comes back zero.
    assert raw.shape == (rows * WORLD, HIDDEN), raw.shape
    assert raw[1::2].abs().max().item() == 0.0, "expected the zero half to show"
    assert (raw != _gather_ref([rows] * WORLD, seed)).float().mean().item() > 0.4
    COMM.Barrier()

    try:
        allgather(view, None, GROUP)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrapper accepted a non-contiguous input")
    COMM.Barrier()


def test_wrapper_guards_a_sizes_list_of_the_wrong_length() -> None:
    """A short `sizes` list silently drops the trailing ranks."""
    rows, seed = 4, 93000
    short = [rows] * (WORLD - 1)
    raw = torch.ops.trtllm.allgather(_payload(RANK, rows, seed), short, GROUP)
    torch.cuda.synchronize()
    assert raw.shape == (rows * (WORLD - 1), HIDDEN), raw.shape
    _assert_bitwise(raw, _gather_ref(short, seed, ranks=range(WORLD - 1)), "short sizes list")
    COMM.Barrier()

    for sizes in (short, [rows] * (WORLD + 1)):
        try:
            allgather(_payload(RANK, rows, seed), sizes, GROUP)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"wrapper accepted sizes of length {len(sizes)}")
    COMM.Barrier()


def test_wrapper_guards_a_zero_dim_input() -> None:
    """A 0-d input segfaults inside the op, so the wrapper stops it first.

    The op is deliberately not called here: the crash is in
    `AllgatherOp::run_list` and kills every rank in the job.
    """
    try:
        allgather(torch.tensor(1.0, device="cuda"), None, GROUP)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrapper accepted a 0-d input")
    COMM.Barrier()


def test_call_order_disagreement_corrupts_silently() -> None:
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
    COMM.Barrier()

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
    COMM.Barrier()

    x = _payload(RANK, rows, seed + 900)
    _assert_bitwise(
        allgather(x, None, GROUP),
        _gather_ref([rows] * WORLD, seed + 900),
        "gather after an order disagreement",
    )
    COMM.Barrier()


TESTS = (
    # Stays first: it is the only test that can observe GROUP's first-ever
    # call, and every later test needs the communicator it builds.
    test_cuda_graph_capture_of_a_first_call_raises,
    test_uniform_gather,
    test_ragged_gather,
    test_output_is_fresh_and_input_is_untouched,
    test_trailing_dims_are_preserved,
    test_dtypes_move_bitwise,
    test_group_selects_a_rank_subset,
    test_group_order_does_not_change_the_output_order,
    test_cuda_graph_at_every_engine_batch_size,
    test_cuda_graph_one_site_per_moe_layer_in_every_batch_size_graph,
    test_cuda_graph_replays_survive_eager_calls_of_other_shapes,
    test_cuda_graph_holds_the_sizes_vector_it_captured,
    test_the_gate_discriminates_a_wrong_gather,
    test_the_engines_own_cross_rank_step_is_not_on_this_communicator,
    test_interleaved_with_the_engines_attention_dp_synchronisation,
    test_the_stream_the_call_lands_on_is_not_part_of_the_match,
    test_wrapper_guards_a_non_contiguous_input,
    test_wrapper_guards_a_sizes_list_of_the_wrong_length,
    test_wrapper_guards_a_zero_dim_input,
    # Stays last: it deliberately disagrees on call order, and although a
    # swapped pair realigns the communicator (its final assertion proves it),
    # nothing after it should depend on that.
    test_call_order_disagreement_corrupts_silently,
)


def _run_one_rank() -> int:
    """Body of one MPI rank: run every test, abort the job if any fails."""
    global COMM, RANK, WORLD, GROUP, allgather, DIST, MPI
    from mpi4py import MPI as _MPI

    from tensorrt_llm._torch.distributed import Distributed
    from tensorrt_llm.mapping import Mapping

    from . import allgather as entry

    allgather = entry.allgather
    MPI = _MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    WORLD = COMM.Get_size()
    GROUP = list(range(WORLD))
    assert WORLD >= 2, f"a collective needs at least 2 ranks, got {WORLD}"
    torch.cuda.set_device(RANK)
    # The engine's own cross-rank object, from the Mapping a serving engine
    # builds for this topology: attention DP over `WORLD` ranks with the MoE
    # expert-parallel over the same set. Real state, not a stand-in — it is the
    # class the executor calls every step.
    # `Distributed.get` is declared to return the abstract base; the concrete
    # class under mpirun is MPIDist and only it carries `tp_comm`, which the
    # tests below assert on — hence the widening.
    engine_dist: Any = Distributed.get(
        Mapping(
            world_size=WORLD,
            rank=RANK,
            tp_size=WORLD,
            moe_ep_size=WORLD,
            enable_attention_dp=True,
        )
    )
    DIST = engine_dist

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
    command = [
        "mpirun",
        "-n",
        str(world_size),
        sys.executable,
        "-m",
        "tensorrt_llm._torch.staircase.catalog.comm.allgather_test",
        _WORKER_FLAG,
    ]
    print(f"[launcher] {' '.join(command)}", flush=True)
    # Own process group so the deadline can kill wedged grandchildren too.
    process = subprocess.Popen(command, start_new_session=True)
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
    _spawn_ranks()
    print("OK")
