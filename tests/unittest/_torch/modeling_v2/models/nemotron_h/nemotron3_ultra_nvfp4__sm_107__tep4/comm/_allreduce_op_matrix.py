# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the Nemotron-3 Ultra tep4 target's allreduce entry.

A collective cannot be exercised in one process, so this script is its own
launcher: run it plainly and it spawns one worker process per device, with one
rank per device named in CUDA_VISIBLE_DEVICES, under a deadline the parent
enforces by killing the whole process group. A wedged collective hangs rather
than raising, so the deadline is what keeps a broken kernel from taking the
calling run down with it.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _allreduce_op_matrix.py

The ranks reach the op through the `_pg` symbol on a torch.distributed NCCL
group, as the other collective matrices do, and set up what the engine's
TorchDist sets up before a model runs: `init_pg` registers the process groups
with the C++ side (allreduce_pg reads the world group on every call), and
`get_allreduce_workspace(mapping)` -- the function `AllReduce(mapping)` calls
-- allocates the IPC workspace the ONESHOT / TWOSHOT / AUTO strategies need.

Not a pytest module, despite the `check_*` bodies. They are one fixed sequence
inside a single 4-rank job rather than independent cases: each reads
module-global rank state that only `_run_one_rank` binds. Neither this file's
name nor its function names match pytest's collection patterns.

The collected entry point is `test_modeling_v2_allreduce_op_matrix.py` beside
this file: it starts this job and turns its exit code into an assertion. Both
halves are started by file path: the launcher must not import `tensorrt_llm`
(that calls MPI_Init, and an MPI-initialized process cannot start the
children cleanly), and the ranks reach the entry by absolute import.
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

assert torch.cuda.is_available(), "allreduce requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 1800

HIDDEN = 8192  # Ultra's hidden size: every all-reduce this target makes is [T, 8192] bf16

# The decode graphs the Ultra accuracy gate captures:
# CudaGraphConfig(max_batch_size=32, enable_padding=True) gives these seven,
# all alive at once in one memory pool and replayed interleaved as the served
# batch size moves. Decode is one token per sequence, so rows == batch size.
GRAPH_BATCH_SIZES: Tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32)

# All-reduces inside one decode graph: one per layer of the 108 -- 48 mamba
# out_proj partials, 48 MoE (latent + shared expert) partials, 12 attention
# o_proj partials.
SITES_PER_DECODE_GRAPH = 108

# bf16 unit roundoff: 8 significand bits, so a round-to-nearest step is off by
# at most 2^-8 of the running value.
BF16_U = 2.0**-8

# Bound inside the rank body, never in the launcher: importing the entry
# pulls in tensorrt_llm, which calls MPI_Init at import.
allreduce: Any = None
entry: Any = None
validating: Any = None
S: Any = None  # tensorrt_llm.functional.AllReduceStrategy
RANK = 0
WORLD = 1
GROUP: List[int] = [0]
PG_BOXED: Any = None
WORKSPACE: Optional[torch.Tensor] = None


def _payload(rank: int, rows: int, seed: int) -> torch.Tensor:
    """One rank's contribution, exact: multiples of 1/8 with |x| <= 31/8.

    bf16 holds k/8 exactly for |k| <= 255, and a sum of WORLD of these reaches
    |k| <= 124, so **every partial sum is exact in every summation order** --
    which is what lets every assertion on these payloads be bitwise, whatever
    order a strategy adds in. Every rank can regenerate every other rank's
    payload (cuRAND is deterministic for a given seed and shape), so the
    reference is arithmetic rather than a second collective.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed * 977 + rank + 1)
    raw = torch.randint(-31, 32, (rows, HIDDEN), generator=gen, device="cuda", dtype=torch.int32)
    return (raw.float() / 8.0).to(torch.bfloat16)


def _rand(rank: int, rows: int, seed: int) -> torch.Tensor:
    """A contribution with no exactness property: standard normal in bf16."""
    gen = torch.Generator(device="cuda").manual_seed(seed * 977 + rank + 1)
    return torch.randn((rows, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)


def _ref(rows: int, seed: int) -> torch.Tensor:
    # Delegates to the entry: what the sum is supposed to be is the entry's
    # claim, and computing it here as well would be a second place to fix.
    return entry.allreduce.reference([_payload(r, rows, seed) for r in range(WORLD)])


def _ws(strategy) -> Optional[torch.Tensor]:
    return None if strategy == S.NCCL else WORKSPACE


def _assert_bitwise(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    """Gate of exactly zero."""
    try:
        assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
    except AssertionError as exc:
        raise AssertionError(f"{where}: {exc}") from exc


def _assert_within_summation_bound(
    out: torch.Tensor, xs: Sequence[torch.Tensor], where: str
) -> None:
    """Every element inside the classical bound for summing WORLD terms.

    Any summation order -- sequential, ring or tree -- rounding to nearest at
    each step lands within gamma_{n-1} * sum|x_r| of the exact sum, gamma_k =
    k*u / (1 - k*u), u = 2^-8 for bf16. The exact sum is formed in float64,
    where four bf16 values add without rounding.
    """
    exact = sum(x.double() for x in xs)
    absum = sum(x.double().abs() for x in xs)
    k = WORLD - 1
    budget = (k * BF16_U / (1 - k * BF16_U)) * absum
    err = (out.double() - exact).abs()
    over = (err > budget).sum().item()
    worst = (err / budget.clamp_min(1e-300)).max().item()
    assert over == 0, (
        f"{where}: {over} elements outside the summation bound, worst at {worst:.3f} of it"
    )


def _assert_kernel_arithmetic(
    out: torch.Tensor, xs: Sequence[torch.Tensor], kernel: str, where: str
) -> None:
    """What the implementation the cell names computes, on an inexact payload.

    ONESHOT and TWOSHOT add in fp32 in ascending rank order and round once:
    that is `reference`, bit for bit. NCCL rounds at every hop, so it is not --
    measured, a third of the elements differ -- and is held to the bound
    instead. The second half doubles as the check that a strategy expected to
    run a custom kernel did not silently fall back to NCCL, and vice versa.

    Discrimination, measured by mutating the run rather than this file: with
    ALL_REDUCE_FUSION_KERNEL_ACC_FP32=0 (the custom kernels add in bf16) the
    first ONESHOT cell failed on all four ranks with 33.2% of elements wrong;
    with TRTLLM_ALLREDUCE_FUSION_WORKSPACE_SIZE=4096 (every custom call falls
    back to NCCL) it failed with 32.6% wrong -- against a gate of 0.
    """
    _assert_within_summation_bound(out, xs, where)
    ref = entry.allreduce.reference(xs)
    if kernel in ("oneshot", "twoshot"):
        _assert_bitwise(out, ref, f"{where} (fp32 sum in rank order, rounded once)")
    else:
        assert kernel == "nccl", kernel
        differs = (out != ref).float().mean().item()
        assert differs > 0.05, (
            f"{where}: only {differs:.4f} of elements differ from the fp32 sum rounded once; "
            "that is not NCCL's per-hop rounding -- a different kernel ran"
        )


def _assert_same_on_every_rank(out: torch.Tensor, where: str) -> None:
    """Bit-identical on every rank: TP-replicated activations stay replicated."""
    gathered = [torch.empty_like(out) for _ in range(WORLD)]
    dist.all_gather(gathered, out.contiguous())
    for r, other in enumerate(gathered):
        _assert_bitwise(other, gathered[0], f"{where}: rank {r} against rank 0")


def _warm_up_off_capture_stream(body, reps: int = 2) -> None:
    """Run `body` on a side stream, then rejoin, so a capture can follow."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(reps):
            body()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    dist.barrier()


def _capture_per_batch_size(
    strategy, sites: int, seed: int
) -> Tuple[Dict[int, List[torch.Tensor]], Dict[int, Any], Dict[int, List[torch.Tensor]]]:
    """One graph per GRAPH_BATCH_SIZES entry, one shared pool, all left alive.

    That is the state a graph runner ends up in. `sites` all-reduces go into
    each graph, each reading its own persistent input and copying into its own
    persistent output inside the capture (pooled outputs may alias across
    graphs once replays are issued back to back). Returns (inputs, graphs,
    outputs), keyed by row count.
    """
    inputs: Dict[int, List[torch.Tensor]] = {}
    graphs: Dict[int, Any] = {}
    outputs: Dict[int, List[torch.Tensor]] = {}
    pool = None
    ws = _ws(strategy)
    for rows in GRAPH_BATCH_SIZES:
        xs = [_payload(RANK, rows, seed + 13 * s) for s in range(sites)]

        def body(xs: List[torch.Tensor] = xs) -> List[torch.Tensor]:
            return [allreduce(x, GROUP, strategy, ws) for x in xs]

        persistent = [
            torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device="cuda") for _ in range(sites)
        ]

        def capture_body() -> List[torch.Tensor]:
            for dst, src in zip(persistent, body()):
                dst.copy_(src)
            return persistent

        _warm_up_off_capture_stream(body)
        graph = torch.cuda.CUDAGraph()
        ctx = torch.cuda.graph(graph) if pool is None else torch.cuda.graph(graph, pool=pool)
        with ctx:
            outputs[rows] = capture_body()
        assert len(outputs[rows]) == sites, (rows, len(outputs[rows]))
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
    for site, x in enumerate(inputs[rows]):
        x.copy_(_payload(RANK, rows, seed + 13 * site))


def _verify_replay(
    outputs: Dict[int, List[torch.Tensor]], rows: int, seed: int, where: str
) -> None:
    for site, out in enumerate(outputs[rows]):
        _assert_bitwise(out, _ref(rows, seed + 13 * site), f"{where} site={site}")


def _replay_every_order(name: str, inputs, graphs, outputs, seed: int) -> int:
    """Checked replays one at a time, then the whole order issued as a burst.

    Each replay gets a payload no earlier call used, so a graph that silently
    did not re-run keeps the previous answer and fails the bitwise gate.
    """
    for order_name, order in _replay_orders():
        for pos, rows in enumerate(order):
            seed += 1000
            _fresh_payload(inputs, rows, seed)
            graphs[rows].replay()
            torch.cuda.synchronize()
            _verify_replay(outputs, rows, seed, f"{name} {order_name}/checked@{pos} rows={rows}")
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
            _verify_replay(
                outputs, rows, staged[rows], f"{name} {order_name}/burst@{pos} rows={rows}"
            )
        dist.barrier()
    return seed


def check_cuda_graph_captures_a_cold_first_call() -> None:
    """The op's first-ever call captures straight into a graph, no warm-up.

    Under the `_pg` symbol the NCCL communicator is built by
    `init_process_group`, before any op runs, so a capture never has to build
    one inside itself. (Under the MPI-session symbol the communicator is built
    on the group's first call, which therefore must not be a capture -- the
    engine's warm-up forward makes it.) Replaying after a fresh input must
    reflect that input, bitwise.
    """
    rows, seed = 16, 100
    x = _payload(RANK, rows, seed)
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = allreduce(x, GROUP, S.NCCL, None)
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(captured, _ref(rows, seed), "cold first capture")
    x.copy_(_payload(RANK, rows, seed + 500))
    torch.cuda.synchronize()
    dist.barrier()
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(captured, _ref(rows, seed + 500), "replay after refill")
    del graph
    dist.barrier()


def check_certified_cells() -> None:
    """Drive the entry's own cell list, inside `validating`.

    Per cell: the exact sum on an exact payload, bitwise; then on a random
    payload, the named kernel's arithmetic, determinism over three calls, and
    bit-identical results on every rank.
    """
    for i, cell in enumerate(entry.allreduce.CELLS):
        spec = cell.spec
        rows, strategy, kernel = spec["rows"], spec["strategy"], spec["kernel"]
        assert spec["hidden"] == HIDDEN and spec["dtype"] is torch.bfloat16, spec
        seed = 9000 + 13 * i
        x = _payload(RANK, rows, seed)
        with validating(entry.allreduce):
            out = allreduce(x, GROUP, strategy, _ws(strategy))
        assert out.shape == x.shape and out.dtype is x.dtype and out.is_contiguous(), cell.why
        _assert_bitwise(out, _ref(rows, seed), f"{cell.why} (exact payload)")

        xs = [_rand(r, rows, seed) for r in range(WORLD)]
        outs = [allreduce(xs[RANK], GROUP, strategy, _ws(strategy)) for _ in range(3)]
        torch.cuda.synchronize()
        for k, o in enumerate(outs[1:], start=1):
            _assert_bitwise(o, outs[0], f"{cell.why}: repeat {k}")
        _assert_kernel_arithmetic(outs[0], xs, kernel, cell.why)
        _assert_same_on_every_rank(outs[0], cell.why)
        dist.barrier()


def check_output_is_fresh_and_input_is_untouched() -> None:
    """The op allocates its result; the caller keeps ownership of `input`."""
    for strategy in (S.NCCL, S.ONESHOT):
        rows, seed = 8, 4100
        x = _payload(RANK, rows, seed)
        before = x.clone()
        out = allreduce(x, GROUP, strategy, _ws(strategy))
        torch.cuda.synchronize()
        assert out.data_ptr() != x.data_ptr(), f"{strategy.name}: output aliased the input"
        _assert_bitwise(x, before, f"{strategy.name}: input after the call")
        x.fill_(-7.0)
        torch.cuda.synchronize()
        _assert_bitwise(out, _ref(rows, seed), f"{strategy.name}: output after input clobber")
        dist.barrier()


def check_cuda_graph_at_every_engine_batch_size() -> None:
    """Every decode graph, 108 all-reduces each, replayed interleaved.

    For NCCL and for AUTO -- which at these row counts runs the one-shot
    kernel, so 756 captured one-shot calls share one workspace and its flags
    across seven graphs. Checked replays, then each order as a burst of seven
    graphs issued back to back with no synchronization: the only pass that
    can catch a hazard needing the next replay already in flight.
    """
    seed = 110000
    for strategy in (S.NCCL, S.AUTO):
        inputs, graphs, outputs = _capture_per_batch_size(strategy, SITES_PER_DECODE_GRAPH, 11000)
        assert len(graphs) == len(GRAPH_BATCH_SIZES)
        seed = _replay_every_order(strategy.name, inputs, graphs, outputs, seed)
        del graphs, outputs, inputs
        dist.barrier()
    # Discrimination for a gate of exactly 0: a replay that did not happen
    # leaves the previous payload's sum behind, which differs from this one's
    # almost everywhere.
    prev, last = _ref(16, seed - 1000), _ref(16, seed)
    assert (prev != last).float().mean().item() > 0.9 and (
        prev.float() - last.float()
    ).abs().max() > 4.0


def check_cuda_graph_replays_survive_eager_calls_of_other_shapes() -> None:
    """Between two decode replays, a server runs prefill work no graph holds.

    AUTO graphs (the one-shot kernel) replayed between eager AUTO calls at
    prefill heights -- 8192 rows (NCCL), 2048 (two-shot), 300 (one-shot) --
    and eager NCCL calls, all on the same workspace and communicator the
    graphs baked in.
    """
    inputs, graphs, outputs = _capture_per_batch_size(S.AUTO, 1, 41000)
    seed = 510000
    for i, rows in enumerate(reversed(GRAPH_BATCH_SIZES)):
        wide = (8192, 2048, 300)[i % 3]
        _assert_bitwise(
            allreduce(_payload(RANK, wide, 610000 + i), GROUP, S.AUTO, WORKSPACE),
            _ref(wide, 610000 + i),
            f"eager AUTO {wide} rows",
        )
        _assert_bitwise(
            allreduce(_payload(RANK, 37, 710000 + i), GROUP, S.NCCL, None),
            _ref(37, 710000 + i),
            "eager NCCL 37 rows",
        )
        seed += 1000
        _fresh_payload(inputs, rows, seed)
        graphs[rows].replay()
        torch.cuda.synchronize()
        _verify_replay(outputs, rows, seed, f"after-eager@{i} rows={rows}")
        dist.barrier()
    del graphs, outputs, inputs
    dist.barrier()


def check_the_gates_discriminate_a_wrong_sum() -> None:
    """Both gates reject every plausible wrong all-reduce, by a wide margin.

    A gate of exactly 0 cannot be too loose, but it can be blind. The ways an
    all-reduce could plausibly be wrong -- a rank that never contributed, one
    counted twice, one rank's input standing in for another's -- all have the
    right shape. Each is rejected by the bitwise gate on the exact payload and
    by the summation bound on the random one.
    """
    rows, seed = 8, 91000
    exact = [_payload(r, rows, seed) for r in range(WORLD)]
    inexact = [_rand(r, rows, seed) for r in range(WORLD)]
    good = allreduce(exact[RANK], GROUP, S.NCCL, None)
    _assert_bitwise(good, entry.allreduce.reference(exact), "baseline")
    variants = {
        "a rank never contributed": lambda xs: xs[:-1],
        "rank 0 counted twice": lambda xs: [xs[0]] + xs,
        "rank 0's input in rank 1's place": lambda xs: [xs[0], xs[0]] + xs[2:],
    }
    for name, wrong in variants.items():
        bad = entry.allreduce.reference(wrong(exact))
        differs = (bad != good).float().mean().item()
        assert differs > 0.9, f"{name}: only {differs:.4f} of elements differ"
        try:
            _assert_within_summation_bound(entry.allreduce.reference(wrong(inexact)), inexact, name)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"the summation bound admitted '{name}'")
    dist.barrier()


def check_the_stream_the_call_lands_on_is_not_part_of_the_match() -> None:
    """The op runs on whatever stream is current, and ranks need not agree.

    The same forward runs on torch's capture stream while a graph is being
    captured and on the serving stream otherwise. What pairs the calls is
    their order on the communicator, not the stream. Checked for NCCL and for
    the one-shot kernel.
    """
    side = torch.cuda.Stream()
    rows = 24
    cases = [
        ("every rank on a side stream", lambda i: True),
        ("only rank 0 on a side stream", lambda i: RANK == 0),
        ("ranks alternating opposite", lambda i: (i + RANK) % 2 == 0),
    ]
    for strategy in (S.NCCL, S.ONESHOT):
        for c, (name, use_side) in enumerate(cases):
            for i in range(4):
                seed = 150000 + 1000 * c + 13 * i
                x = _payload(RANK, rows, seed)
                if use_side(i):
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        out = allreduce(x, GROUP, strategy, _ws(strategy))
                    torch.cuda.current_stream().wait_stream(side)
                else:
                    out = allreduce(x, GROUP, strategy, _ws(strategy))
                _assert_bitwise(out, _ref(rows, seed), f"{strategy.name} {name} i={i}")
            dist.barrier()


def check_wrapper_guards_a_non_contiguous_input() -> None:
    """A strided view is summed from the wrong elements by the one-shot kernel.

    Measured first, on every rank at once (a collective), then refused by the
    guard without reaching the op.
    """
    rows, seed = 8, 92000
    wide = torch.cat([_payload(RANK, rows, seed), _payload(RANK, rows, seed + 1)], dim=1)
    view = wide[:, :HIDDEN]
    assert not view.is_contiguous()
    silent = allreduce(view, GROUP, S.ONESHOT, WORKSPACE)
    differs = (silent != _ref(rows, seed)).float().mean().item()
    assert differs > 0.3, (
        f"the one-shot kernel now reads a strided view correctly ({differs:.4f} differ)"
    )
    dist.barrier()
    try:
        with validating(entry.allreduce):
            allreduce(view, GROUP, S.NCCL, None)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrapper accepted a non-contiguous input")
    dist.barrier()


def check_wrapper_guards_a_float8_input() -> None:
    """The guard refuses float8 without reaching the op.

    Under the MPI-session symbol float8 maps to ncclInt8 and the raw bytes are
    added (measured outside this matrix: 99.8% of results equal the int8 byte
    sum). The `_pg` symbol hands the dtype to c10d instead, so what this
    matrix's own path would return says nothing about the engine's.
    """
    rows, seed = 4, 93000
    xf = (_payload(RANK, rows, seed).float() / 4).to(torch.float8_e4m3fn)
    try:
        with validating(entry.allreduce):
            allreduce(xf, GROUP, S.NCCL, None)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrapper accepted a float8 input")
    dist.barrier()


def check_call_order_disagreement_corrupts_silently() -> None:
    """Ranks that disagree on the order of two equal-sized all-reduces get a
    wrong sum back, with no error and no hang.

    What pairs the calls is their position on the communicator, so a swapped
    pair sums rank 0's second input with everyone else's first. Every rank
    receives the same wrong tensor, which is what makes this invisible from
    inside a forward: a target must issue its all-reduces in the same order on
    every rank. Kept last; a swapped pair realigns the communicator once both
    calls are made, which the final plain call proves.
    """
    rows, seed = 12, 160000
    first, second = _payload(RANK, rows, seed), _payload(RANK, rows, seed + 500)
    ref_first, ref_second = _ref(rows, seed), _ref(rows, seed + 500)
    dist.barrier()
    if RANK == 0:
        got_second = allreduce(second, GROUP, S.NCCL, None)
        got_first = allreduce(first, GROUP, S.NCCL, None)
    else:
        got_first = allreduce(first, GROUP, S.NCCL, None)
        got_second = allreduce(second, GROUP, S.NCCL, None)
    torch.cuda.synchronize()
    for name, got, ref in (("first", got_first, ref_first), ("second", got_second, ref_second)):
        wrong = (got != ref).float().mean().item()
        assert wrong > 0.5, f"{name}: only {wrong:.4f} of elements differ"
    dist.barrier()
    x = _payload(RANK, rows, seed + 900)
    _assert_bitwise(
        allreduce(x, GROUP, S.NCCL, None), _ref(rows, seed + 900), "after an order disagreement"
    )
    dist.barrier()


CHECKS = (
    # Stays first: it is the only check that can observe the op's first-ever call.
    check_cuda_graph_captures_a_cold_first_call,
    check_certified_cells,
    check_output_is_fresh_and_input_is_untouched,
    check_cuda_graph_at_every_engine_batch_size,
    check_cuda_graph_replays_survive_eager_calls_of_other_shapes,
    check_the_gates_discriminate_a_wrong_sum,
    check_the_stream_the_call_lands_on_is_not_part_of_the_match,
    check_wrapper_guards_a_non_contiguous_input,
    check_wrapper_guards_a_float8_input,
    # Stays last: it deliberately disagrees on call order.
    check_call_order_disagreement_corrupts_silently,
)


class _HandleExchange:
    """Stands in for the engine's `Distributed` object while the workspace is built.

    `get_allreduce_workspace` swaps IPC handles with `Distributed.get(mapping)
    .tp_allgather`; under TLLM_DISABLE_MPI that is TorchDist, whose
    constructor needs ray. The exchange it performs is an all_gather_object
    over the TP group, which a gloo group does here.
    """

    def __init__(self, group: Any) -> None:
        self._group = group

    def tp_allgather(self, obj: Any, **_: Any) -> List[Any]:
        out: List[Any] = [None] * WORLD
        dist.all_gather_object(out, obj, group=self._group)
        return out


def _run_one_rank() -> int:
    """Body of one spawned rank: run every check; a failing rank exits nonzero
    and the launcher kills its siblings."""
    global RANK, WORLD, GROUP, PG_BOXED, WORKSPACE, S, allreduce, validating, entry
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    # device_id keeps NCCL communicator teardown off the implicit-device path
    # that measurably wedges a watchdog at exit.
    dist.init_process_group("nccl", init_method="env://", device_id=torch.device("cuda", rank))

    from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.nemotron3_ultra_nvfp4__sm_107__tep4.ops.comm import (  # noqa: E501
        allreduce as entry,
    )
    from tensorrt_llm._torch.distributed import communicator
    from tensorrt_llm._torch.distributed.ops import get_allreduce_workspace
    from tensorrt_llm._utils import torch_pybind11_abi
    from tensorrt_llm.bindings.internal.process_group import init_pg
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    RANK, WORLD, GROUP, S = rank, world, list(range(world)), AllReduceStrategy
    assert WORLD >= 2, f"a collective needs at least 2 ranks, got {WORLD}"

    # What the engine's TorchDist does before any all-reduce: register the
    # world and node-local groups with the C++ side. allreduce_pg reads the
    # world group on every call and dies on a null one without it.
    local = dist.new_group(ranks=GROUP, backend="cuda:nccl,cpu:gloo")
    init_pg(dist.group.WORLD, local, torch_pybind11_abi())

    # The workspace AllReduce(mapping) holds, from the function it calls.
    exchange = _HandleExchange(dist.new_group(backend="gloo"))
    communicator.Distributed.get = staticmethod(lambda mapping: exchange)
    WORKSPACE = get_allreduce_workspace(
        Mapping(world_size=world, rank=rank, tp_size=world, gpus_per_node=world)
    )

    # One partial instead of per-call arguments: every check's call site keeps
    # the certified shape, and the wrapper hands the boxed group and this rank
    # to the _pg symbol -- the routing the product's non-MPI mode does in
    # _torch/distributed/ops.py. The guard is armed on the entry itself, and
    # `reference` and `CELLS` are read off `entry.allreduce` directly.
    PG_BOXED = dist.distributed_c10d._get_default_group().boxed()
    allreduce = functools.partial(entry.allreduce, process_group=PG_BOXED, rank=rank)

    # A spawned worker, not collected by pytest, so `__extra_import_path__` is
    # not in play; the mutation is confined to these rank processes.
    # comm -> <target> -> nemotron_h -> models -> modeling_v2, where _validating lives.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from _validating import validating as _validating_cm

    validating = _validating_cm

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


def _spawn_ranks() -> None:
    """Start one worker process per claimed device; no external launcher.

    The ranks rendezvous through torch.distributed's env:// method, so the only
    thing this needs is a free port. Each child gets its own session; on the
    first nonzero exit or on the deadline, every session is killed.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, (
        "set CUDA_VISIBLE_DEVICES to the devices this run owns, e.g. export CUDA_VISIBLE_DEVICES=0,1,2,3"
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
    # errhandler, which SIGKILLs the whole step before anything is reported.
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
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    _spawn_ranks()
    print("OK")
