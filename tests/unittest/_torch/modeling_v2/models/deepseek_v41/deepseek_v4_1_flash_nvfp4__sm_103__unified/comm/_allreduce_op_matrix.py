# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the DeepSeek-V4.1 target's allreduce entry.

A collective cannot be exercised in one process, so this script is its own
launcher: run it plainly and, for each suite, it spawns one worker process per
device named in CUDA_VISIBLE_DEVICES, rendezvoused through torch.distributed's
env://, under a deadline it enforces by killing every worker's session. A
broken collective hangs rather than raising, so the deadline is what keeps a
wedged kernel from taking the calling run down with it.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _allreduce_op_matrix.py [--group-size G]

The ranks run with MPI disabled, so the entry takes the ProcessGroup path the
product takes then (`allreduce_pg`, bound once through `process_group`): the
group's communicator is the Mapping's `tp_group_pg`, and the IPC workspace is
the one `get_allreduce_workspace` builds from that Mapping.

The surface is the plain summing all-reduce (`op = 0`) over the payloads a
DeepSeek-V4.1 tensor-parallel forward hands it: fp32 `[T, 5120]` partial sums
(row-parallel output projection, routed-expert output) and bf16 rows
(`[T, 5120]`, and the `[T, 24, 256]` engram rows whose shards are zero off
their owning rank), on every strategy the op implements for a plain sum,
eagerly and replayed from CUDA graphs. Every cell of the entry's `CELLS` for
this rank's layout is driven here.

The group is a parameter, as the target's collective group is: the world is
cut into contiguous blocks of `--group-size` ranks (default: the whole world),
which is how a pipeline-parallel engine lays out its stages' TP groups
(`Mapping.tp_group`), and every block runs the same checks at once on its own
group. The layouts this covers:

- world 4, group 4 (tep4 / tp4 on one node) and world 2, group 2
  (pp2 x tep2 / tp2 stage groups, run alone), from this launcher;
- world 4 in groups of 2 (the pp2 x tep2 / tp2 layout itself: groups [0, 1]
  and [2, 3], each with the workspace a pp2 mapping builds), from this
  launcher with `--group-size 2`;
- world 8 over two NVL72 trays (tep8), and world 8 in groups of 4 (the
  pp2 x tep4 / tp4 layout, one group per tray), from an external two-node
  `torchrun` that starts the rank body directly (the command is in the
  entry's `note`): `--rank-worker <suite> [--group-size G]`.

A group whose ranks do not share one host gets no IPC workspace (the target's
`ar_one_node`), so there the fp32 sums go to NCCL (strategy 0) exactly as the
target sends them, the workspace strategies are not run, and the two mixed
suites (which are about the workspace) do not apply.

It runs as five jobs, each a fresh multi-rank process set with its own IPC
workspace: the fp32 suite (every strategy on fp32, with bf16 rows only on the
two workspace-free strategies, as a DeepSeek-V4.1 forward interleaves them),
the bf16 suite (every strategy on bf16), the core suite (the DeepSeek-V4.1 core
target's two call forms -- fp32 [T, 5120] on AUTO (NCCL across nodes) at every
row count 1..32768, bf16 [T, 6144] on NCCL at every row count 1..16384 -- and
decode graphs at every per-rank row count G and 6G of the capture batch set),
and two mixed suites, which replay the defect that forces that split -- the
ONESHOT kernel sizes the reset of its previous call's buffer in that call's
element count, so a workspace whose ONESHOT path sees two element sizes reads
stale data -- and assert that this harness sees it (see the entry's `note`);
the second does it with the core target's own layer sequence and the engram
rows on the wrong strategy.

Not a pytest module, despite the `check_*` bodies. They are one fixed
sequence inside a single multi-rank job: each reads module-global rank state
that only `_run_one_rank` binds. Collected as tests they would run at world
size 1 against unbound globals, so neither this file's name nor its function
names match pytest's collection patterns.

The collected entry point is `test_modeling_v2_allreduce.py` beside this file:
it starts this job and turns its exit code into an assertion. Both halves are
started by file path: the launcher must carry no CUDA or torch.distributed
state of its own, and the ranks reach the entry by absolute import.
"""

import hashlib
import os
import random
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

assert torch.cuda.is_available(), "allreduce requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
_GROUP_FLAG = "--group-size"
# Each suite's own deadline. The core suite sweeps every row count up to 32768
# and captures 84 decode graphs, so it gets the largest share.
SUITE_DEADLINE_S: Dict[str, int] = {
    "fp32": 600,
    "bf16": 600,
    "core": 2400,
    "mixed": 600,
    "mixed_core": 600,
}
DEADLINE_S = sum(SUITE_DEADLINE_S.values())
"""Budget for the whole launcher: the sum of the suites' deadlines."""

SUITES: Tuple[str, ...] = tuple(SUITE_DEADLINE_S)

HIDDEN = 5120  # DeepSeek-V4.1-Flash hidden size
ENGRAM_ROW = (24, 256)  # one engram value row per token

# Every strategy the op implements for a plain sum, by the integer it takes.
ALL_STRATEGIES: Dict[int, str] = {
    0: "NCCL",
    8: "NCCL_SYMMETRIC",
    3: "AUTO",
    4: "ONESHOT",
    5: "TWOSHOT",
    1: "MIN_LATENCY",
}
# The ones that read the IPC workspace; the other two run with workspace=None.
WORKSPACE_STRATEGIES = (3, 4, 5, 1)
# The strategies this group runs: all six on one node, the two workspace-free
# ones when the group spans nodes (no IPC workspace exists there). Rebound in
# the rank body.
STRATEGIES: Dict[int, str] = dict(ALL_STRATEGIES)

# The fp32 [T, 5120], bf16 [T, 5120] and bf16 [T, 24, 256] row counts every
# strategy runs are the entry's cells (`allreduce.CELLS`), driven by
# `check_certified_cells_*`. The row-sharded engram check runs the 3-D form's
# row counts.
ENGRAM_ROWS: Tuple[int, ...] = (1, 8, 32, 4096)

# One decode graph of a TP4 DeepSeek-V4.1 forward holds two fp32 all-reduces
# per decoder layer (output projection and routed experts) over 40 layers.
SITES_PER_DECODE_GRAPH = 80
ENGRAM_SITE_EVERY = 10  # plus one bf16 engram row sum per this many fp32 sites
GRAPH_ROWS: Tuple[int, ...] = (1, 2, 3, 4, 8, 16, 32, 64, 128)

# The DeepSeek-V4.1 core target's two call forms under tensor parallelism
# (`_allreduce32` and `_engram`): fp32 [T, 5120] partial sums on AUTO with the
# workspace when the group is on one node (NCCL when it spans nodes: the
# target's `ar_one_node`), and the engram rows as a 2-D bf16 [T, 24 * 256] view
# on NCCL.
ENGRAM_HIDDEN = 6144
AUTO, NCCL = 3, 0
CORE_FP32 = AUTO  # the fp32 sums' strategy for this group; rebound in the rank body
# Rows per rank. The fp32 sums run on every layer, and layers 21-39 of a
# tensor-parallel context step can run a decoder pass over more rows than
# max_num_tokens: the finishing requests' 128-row tails plus the generation
# rows, up to 128 x 256 = 32768. The engram sums run on layers 1 and 14 only,
# ahead of that pass, so their rows stop at max_num_tokens (16384).
N_MAX = 32768
ENGRAM_N_MAX = 16384
SWEEP_SLACK = 509  # extra table rows (prime): every call reads its rows at its own offset
# What the fp32 [T, 5120] sums resolve to, per group size, on sm_103 (the op's
# lookup table, `selectStrategyLookUpTable`, is indexed by the group size and
# buckets T by floor(log2(T)) and hidden 5120 into its 4096 column; the 64 MiB
# workspace check sends [3277, 5120] fp32 and everything above to NCCL). Read
# from the op's own debug log (`AllReduceOp runtime strategy for rank 0:`).
# A group that spans nodes calls NCCL directly.
AUTO_FP32_BANDS_BY_GROUP: Dict[int, Tuple[Tuple[int, int, str], ...]] = {
    2: ((1, 3276, "ONESHOT"), (3277, N_MAX, "NCCL")),
    4: ((1, 1023, "ONESHOT"), (1024, 3276, "TWOSHOT"), (3277, N_MAX, "NCCL")),
}
NCCL_BANDS: Tuple[Tuple[int, int, str], ...] = ((1, N_MAX, "NCCL"),)
AUTO_FP32_BANDS: Tuple[Tuple[int, int, str], ...] = AUTO_FP32_BANDS_BY_GROUP[
    4
]  # rebound in the rank body
# One decode forward: 40 layers, an attention and a MoE fp32 sum each, plus the
# engram rows' bf16 sum ahead of the attention sum on layers 1 and 14.
CORE_LAYERS = 40
ENGRAM_LAYER_IDS = (1, 14)
# The capture batch set, per rank: every G in 1..32 plus the padded default list
# of cuda_graph_config {enable_padding: true, max_batch_size: 256} (1, 2, 4, 8,
# 16..128 step 8, 192, 256) -- 46 values. A captured step is generation-only
# with next_n 1 (DSpark off) or 6 (DSpark on), so the target's sums see T = G
# and T = 6G (82 values, up to 1536). The batches G 1..8, 16, 24, 32 with 5G and
# 6G of those stay in as well: they add 35 and 160.
GSET: Tuple[int, ...] = tuple(range(1, 33)) + tuple(range(40, 129, 8)) + (192, 256)
_GATE_BATCHES: Tuple[int, ...] = tuple(range(1, 9)) + (16, 24, 32)
CORE_CAPTURE_ROWS: Tuple[int, ...] = tuple(
    sorted(
        set(GSET)
        | {6 * g for g in GSET}
        | set(_GATE_BATCHES)
        | {5 * b for b in _GATE_BATCHES}
        | {6 * b for b in _GATE_BATCHES}
    )
)
# Eager prefill-sized sums replayed between the decode graphs: row counts the
# runtime dumps showed, the decoder pass's maximum, and a row count on each side
# of AUTO's edges. The engram form runs only where its rows reach.
PREFILL_BETWEEN_REPLAYS: Tuple[int, ...] = (1018, 1935, 2048, 3277, 10444, 16384, 32768)

# Bound inside the rank body, never in the launcher: the launcher must carry no
# CUDA or torch.distributed state, and importing the entry pulls in
# tensorrt_llm. RANK and WORLD are this rank's position in its group and the
# group's size; HOST_GROUP is a gloo group over the group (for host-side checks
# only); GROUP is the group's global ranks, which is what the op takes; LAYOUT
# names the cells' layout this job runs.
allreduce: Any = None
validating: Any = None
HOST_GROUP: Any = None
RANK = 0
WORLD = 1
GROUP: List[int] = [0]
WORKSPACE: Optional[torch.Tensor] = None
ONE_NODE = True  # every rank of the group on one host: the IPC workspace exists
LAYOUT = ""


def _barrier() -> None:
    """Host-side barrier over this rank's group."""
    dist.barrier(group=HOST_GROUP)


def _host_allgather(obj: Any) -> List[Any]:
    """One host-side value from every rank of this rank's group, in group order."""
    gathered: List[Any] = [None] * WORLD
    dist.all_gather_object(gathered, obj, group=HOST_GROUP)
    return gathered


def _payload(rank: int, shape: Sequence[int], seed: int, dtype: torch.dtype) -> torch.Tensor:
    """One rank's contribution, distinct per (rank, seed), whose sums are exact.

    Every rank regenerates every other rank's payload from the same seeds,
    which is what keeps the reference arithmetic rather than a second
    collective. fp32 values are multiples of 1/256 below 4096 in magnitude, so
    any sum of four of them, in any order, is exact in fp32 (at most 24
    significant bits); bf16 values are multiples of 1/8 of at most 4, so every
    partial sum of four is a multiple of 1/8 of at most 16 and exact in bf16's
    8 significant bits. Every assertion on these payloads can be bitwise.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed * 977 + rank + 1)
    if dtype == torch.float32:
        k = torch.randint(
            -(2**20), 2**20 + 1, tuple(shape), generator=gen, device="cuda", dtype=torch.int64
        )
        return (k.double() / 256.0).float()
    k = torch.randint(-32, 33, tuple(shape), generator=gen, device="cuda", dtype=torch.int32)
    return (k.float() / 8.0).to(dtype)


def _sum_ref(shape: Sequence[int], seed: int, dtype: torch.dtype) -> torch.Tensor:
    """The arithmetic sum of every rank's `_payload`, exact in `dtype` here.

    Delegates to the entry: what the sum is supposed to be is the entry's
    claim, built on this rank alone from the seeds every rank shares.
    """
    return allreduce.reference([_payload(r, shape, seed, dtype) for r in range(WORLD)], GROUP)[0]


def _random_parts(shape: Sequence[int], seed: int, dtype: torch.dtype) -> List[torch.Tensor]:
    """Unrestricted normal payloads for every rank, the shape real partials have."""
    parts = []
    for r in range(WORLD):
        gen = torch.Generator(device="cuda").manual_seed(seed * 131 + r)
        parts.append(
            torch.randn(tuple(shape), generator=gen, device="cuda", dtype=torch.float32).to(dtype)
        )
    return parts


def _call(x: torch.Tensor, strategy: int, trigger_completion_at_end: bool = True) -> torch.Tensor:
    ws = WORKSPACE if strategy in WORKSPACE_STRATEGIES else None
    outs = allreduce(
        x, None, None, None, None, ws, GROUP, strategy, 0, 1e-6, trigger_completion_at_end
    )
    assert isinstance(outs, list) and len(outs) == 1, (
        f"op=0 must return one tensor, got {len(outs)}"
    )
    return outs[0]


def _assert_bitwise(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    """Gate of exactly zero, against a reference that is exact here.

    The gate itself is the entry's `compare`; `where` only labels the failure.
    """
    try:
        allreduce.compare([out], [ref])
    except AssertionError as exc:
        raise AssertionError(f"{where}: {exc}") from exc


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha1(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _assert_same_bits_on_every_rank(out: torch.Tensor, where: str) -> None:
    """Every rank must hold the same bytes. A host-side check of the result,
    not a source of the reference."""
    digests = _host_allgather(_digest(out))
    assert len(set(digests)) == 1, f"{where}: ranks disagree: {digests}"


def _layout_cells(suite: str) -> List[Any]:
    """The entry's cells for this job's layout and `suite`, in declaration order."""
    cells = [c for c in allreduce.CELLS if c.spec["layout"] == LAYOUT and c.spec["suite"] == suite]
    assert cells, f"no {suite} cells for layout {LAYOUT}"
    return cells


def _drive_certified_cells(suite: str) -> None:
    """Every cell of `suite` at this layout: every row count on its strategy.

    Per call form, row count outer and strategy inner, so the strategies take
    turns on one workspace as a serving engine's mixed traffic does. Inside
    `validating`, so the guard is shown to admit every input a cell names.
    Each result must be fresh (not aliasing the input), contiguous, on the
    input's device, of the input's shape and dtype, the exact sum, and the
    input untouched.
    """
    cells = _layout_cells(suite)
    forms: List[Tuple[torch.dtype, Tuple[int, ...]]] = []
    for c in cells:
        form = (c.spec["dtype"], tuple(c.spec["trailing"]))
        if form not in forms:
            forms.append(form)
    for dtype, trailing in forms:
        of_form = [
            c for c in cells if (c.spec["dtype"], tuple(c.spec["trailing"])) == (dtype, trailing)
        ]
        rows_of_form = tuple(of_form[0].spec["rows"])
        assert all(tuple(c.spec["rows"]) == rows_of_form for c in of_form), [c.why for c in of_form]
        for rows in rows_of_form:
            for cell in of_form:
                strategy = cell.spec["strategy"]
                name = ALL_STRATEGIES[strategy]
                assert strategy in STRATEGIES, f"{cell.why}: {name} does not apply to this group"
                shape = (rows, *trailing)
                base = 1000 if dtype == torch.float32 else 2000 + len(shape)
                seed = base + rows * 16 + strategy
                x = _payload(RANK, shape, seed, dtype)
                before = x.clone()
                with validating(allreduce):
                    out = _call(x, strategy)
                torch.cuda.synchronize()
                where = f"{cell.why} rows={rows}"
                assert out.shape == x.shape and out.dtype == dtype, (where, out.shape, out.dtype)
                assert out.is_contiguous() and out.device == x.device, where
                assert out.data_ptr() != x.data_ptr(), f"{where}: result aliases the input"
                _assert_bitwise(out, _sum_ref(shape, seed, dtype), where)
                assert torch.equal(x, before), f"{where}: input was modified"
    _barrier()


def check_certified_cells_fp32() -> None:
    """fp32 [T, 5120]: every strategy returns the exact sum, fresh, input untouched."""
    _drive_certified_cells("fp32")


def check_certified_cells_bf16() -> None:
    """bf16 [T, 5120] and the 3-D engram shape [T, 24, 256], on every strategy."""
    _drive_certified_cells("bf16")


def _engram_rows_owned_by_one_rank_come_back_bitwise(strategies: Sequence[int]) -> None:
    """Row-sharded lookups: each row is nonzero on exactly one rank.

    That is how a vocabulary-sharded table is combined -- the owner holds the
    real row and every other rank a zero row -- so the sum is `row + 0 + 0 + 0`
    and must be the owner's row bit for bit, in bf16, whatever the strategy
    and however unrestricted the values are.
    """
    for rows in ENGRAM_ROWS:
        shape = (rows, *ENGRAM_ROW)
        owner = torch.Generator(device="cpu").manual_seed(rows)
        owners = torch.randint(0, WORLD, (rows,), generator=owner).cuda()
        full = _random_parts(shape, 3000 + rows, torch.bfloat16)[0]
        for strategy in strategies:
            name = STRATEGIES[strategy]
            mine = torch.where((owners == RANK).view(rows, 1, 1), full, torch.zeros_like(full))
            out = _call(mine, strategy)
            torch.cuda.synchronize()
            where = f"engram rows={rows} {name}"
            assert torch.equal(out.view(torch.int16), full.view(torch.int16)), (
                f"{where}: not the owner's row bits"
            )
            _assert_same_bits_on_every_rank(out, where)
    _barrier()


def check_engram_rows_owned_by_one_rank_on_the_workspace_free_strategies() -> None:
    _engram_rows_owned_by_one_rank_come_back_bitwise((0, 8))


def check_engram_rows_owned_by_one_rank_on_every_strategy() -> None:
    _engram_rows_owned_by_one_rank_come_back_bitwise(tuple(STRATEGIES))


def check_fp32_on_the_workspace_interleaved_with_bf16_on_nccl() -> None:
    """The DeepSeek-V4.1 mix: fp32 sums on a workspace strategy, bf16 rows on NCCL.

    A tensor-parallel layer sums its fp32 output-projection and routed-expert
    partials and, on engram layers, a bf16 row; the bf16 sum goes through a
    workspace-free strategy, so the workspace's ONESHOT path only ever sees
    fp32. Decode- and prefill-sized steps alternate, as a serving engine
    alternates them.
    """
    rows_cycle = (1, 8, 32, 4096, 3, 128, 2048, 1)
    step = 0
    for strategy in WORKSPACE_STRATEGIES:
        for bf16_strategy in (0, 8):
            for rows in rows_cycle:
                step += 1
                seeds = (7000 + step * 3, 7001 + step * 3, 7002 + step * 3)
                wo = _call(_payload(RANK, (rows, HIDDEN), seeds[0], torch.float32), strategy)
                eng = _call(
                    _payload(RANK, (rows, *ENGRAM_ROW), seeds[1], torch.bfloat16), bf16_strategy
                )
                moe = _call(_payload(RANK, (rows, HIDDEN), seeds[2], torch.float32), strategy)
                torch.cuda.synchronize()
                where = f"interleave {STRATEGIES[strategy]}+{STRATEGIES[bf16_strategy]} rows={rows}"
                _assert_bitwise(
                    wo, _sum_ref((rows, HIDDEN), seeds[0], torch.float32), f"{where} fp32#1"
                )
                _assert_bitwise(
                    eng, _sum_ref((rows, *ENGRAM_ROW), seeds[1], torch.bfloat16), f"{where} bf16"
                )
                _assert_bitwise(
                    moe, _sum_ref((rows, HIDDEN), seeds[2], torch.float32), f"{where} fp32#2"
                )
    _barrier()


def check_fp32_partial_sums_are_reduced_in_fp32() -> None:
    """Unrestricted fp32 partials: the result is the fp32 sum, on every rank alike.

    Four fp32 addends in some order: the default fp32 tolerances of
    `assert_close` cover the at-most-three roundings any summation order
    makes. And every rank must hold the same bits -- a rank that disagrees
    with its peers would feed a different residual stream into the next
    layer on each rank. The reference is the entry's (the correctly rounded
    sum); the gate is not the entry's zero-tolerance `compare`, which is for
    payloads whose sums are exact.
    """
    for rows in (1, 32, 4096):
        shape = (rows, HIDDEN)
        parts = _random_parts(shape, 4000 + rows, torch.float32)
        ref = allreduce.reference(parts, GROUP)[0]
        for strategy, name in STRATEGIES.items():
            out = _call(parts[RANK].clone(), strategy)
            torch.cuda.synchronize()
            where = f"fp32 random rows={rows} {name}"
            torch.testing.assert_close(out, ref, msg=lambda m: f"{where}: {m}")
            _assert_same_bits_on_every_rank(out, where)
    _barrier()


def check_trigger_completion_at_end_false_gives_the_same_bits() -> None:
    """The flag moves the programmatic-launch trigger, not the arithmetic."""
    for rows in (1, 8, 2048):
        for strategy in WORKSPACE_STRATEGIES:
            seed = 5000 + rows + strategy
            shape = (rows, HIDDEN)
            out = _call(
                _payload(RANK, shape, seed, torch.float32),
                strategy,
                trigger_completion_at_end=False,
            )
            torch.cuda.synchronize()
            _assert_bitwise(
                out,
                _sum_ref(shape, seed, torch.float32),
                f"trigger=False rows={rows} {STRATEGIES[strategy]}",
            )
    _barrier()


def check_cuda_graph_decode_graphs_replay_in_any_order() -> None:
    """One graph per decode row count, 80 fp32 all-reduce sites each, shared pool.

    Every tenth fp32 site is followed by a bf16 engram-row sum on NCCL, the
    composition a DeepSeek-V4.1 decode graph has.

    Per strategy: every graph is captured into one memory pool and kept alive,
    then replayed shuffled, twice, each time with a fresh payload copied into
    the captured inputs and an eager call of an unrelated size in between (the
    workspace strategies keep per-call state in device memory, and an eager
    call moves it between replays). Every site's output must be the exact sum
    of what its input held at replay time.
    """
    rng = random.Random(1234)  # identical on every rank, no collective needed
    distinct = 8  # inputs per graph, cycled over the sites
    for strategy, name in STRATEGIES.items():
        # The group's communicator and, for the workspace strategies, the
        # workspace state must exist before a capture; warm on a side stream.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for rows in GRAPH_ROWS:
                _call(_payload(RANK, (rows, HIDDEN), 1, torch.float32), strategy)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        _barrier()

        pool = None
        graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        inputs: Dict[int, List[torch.Tensor]] = {}
        outputs: Dict[int, List[torch.Tensor]] = {}
        engram_inputs: Dict[int, List[torch.Tensor]] = {}
        engram_outputs: Dict[int, List[torch.Tensor]] = {}
        for rows in GRAPH_ROWS:
            inputs[rows] = [
                _payload(RANK, (rows, HIDDEN), 7, torch.float32) for _ in range(distinct)
            ]
            engram_inputs[rows] = [
                _payload(RANK, (rows, *ENGRAM_ROW), 8, torch.bfloat16) for _ in range(2)
            ]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                outputs[rows], engram_outputs[rows] = [], []
                for i in range(SITES_PER_DECODE_GRAPH):
                    outputs[rows].append(_call(inputs[rows][i % distinct], strategy))
                    if i % ENGRAM_SITE_EVERY == ENGRAM_SITE_EVERY - 1:
                        # a bf16 engram row, on NCCL: see the interleave check
                        engram_outputs[rows].append(
                            _call(engram_inputs[rows][len(engram_outputs[rows]) % 2], 0)
                        )
            pool = graph.pool() if pool is None else pool
            graphs[rows] = graph
            torch.cuda.synchronize()
            _barrier()

        for round_ in range(2):
            order = list(GRAPH_ROWS)
            rng.shuffle(order)
            for rows in order:
                seeds = [60000 + round_ * 1000 + rows * 10 + i for i in range(distinct)]
                for i, seed in enumerate(seeds):
                    inputs[rows][i].copy_(_payload(RANK, (rows, HIDDEN), seed, torch.float32))
                engram_seeds = [65000 + round_ * 1000 + rows * 10 + i for i in range(2)]
                for i, seed in enumerate(engram_seeds):
                    engram_inputs[rows][i].copy_(
                        _payload(RANK, (rows, *ENGRAM_ROW), seed, torch.bfloat16)
                    )
                graphs[rows].replay()
                eager_rows = rng.choice((33, 777, 1500))
                eager_seed = 70000 + round_ * 1000 + rows
                eager = _call(
                    _payload(RANK, (eager_rows, HIDDEN), eager_seed, torch.float32), strategy
                )
                torch.cuda.synchronize()
                where = f"graph {name} rows={rows} round={round_}"
                refs = [_sum_ref((rows, HIDDEN), seed, torch.float32) for seed in seeds]
                for site, out in enumerate(outputs[rows]):
                    _assert_bitwise(out, refs[site % distinct], f"{where} site={site}")
                engram_refs = [
                    _sum_ref((rows, *ENGRAM_ROW), seed, torch.bfloat16) for seed in engram_seeds
                ]
                for site, out in enumerate(engram_outputs[rows]):
                    _assert_bitwise(out, engram_refs[site % 2], f"{where} engram site={site}")
                _assert_bitwise(
                    eager,
                    _sum_ref((eager_rows, HIDDEN), eager_seed, torch.float32),
                    f"{where} eager",
                )
                _barrier()
        del graphs, outputs, inputs, engram_outputs, engram_inputs
        torch.cuda.synchronize()
        _barrier()


def check_the_gate_discriminates_a_wrong_reduce() -> None:
    """Deliberately wrong reductions, each of which the gates above must reject.

    A zero-tolerance gate cannot be too loose, but it can be blind; these are
    the ways an all-reduce of these payloads plausibly goes wrong, each with
    the right shape and dtype. The last one is the real op run on a wrong
    input -- the fp32 partials rounded to bf16 before the reduction, which is
    what an all-reduce in the activation dtype does to them -- and it must
    fail the fp32 gate that the correct call passes.

    Each variant is formed in fp64 and rounded once to fp32, the output's
    dtype, which is what the entry's `compare` sees.
    """
    rows, seed = 32, 88000
    shape = (rows, HIDDEN)
    blocks = [_payload(r, shape, seed, torch.float32).double() for r in range(WORLD)]
    good = _call(_payload(RANK, shape, seed, torch.float32), 0)
    torch.cuda.synchronize()
    exact = sum(blocks)
    _assert_bitwise(good, _sum_ref(shape, seed, torch.float32), "baseline")

    wrong = {
        "one rank never contributed": sum(blocks[:-1]),
        "one rank counted twice": exact + blocks[0],
        "mean instead of sum": exact / WORLD,
        "partials rounded to bf16": sum(b.to(torch.bfloat16).double() for b in blocks),
    }
    for label, variant in wrong.items():
        variant = variant.float()
        fraction = (variant != good).float().mean().item()
        assert fraction > 0.9, f"{label}: only {fraction:.4f} of elements differ"
        try:
            _assert_bitwise(good, variant, label)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"{label}: the gate accepted a wrong reduction")

    parts = _random_parts(shape, 89000, torch.float32)
    ref = allreduce.reference(parts, GROUP)[0]
    in_bf16 = _call(parts[RANK].to(torch.bfloat16), 0).float()
    torch.cuda.synchronize()
    try:
        torch.testing.assert_close(in_bf16, ref)
    except AssertionError:
        pass
    else:
        raise AssertionError("an all-reduce of bf16-rounded partials passed the fp32 gate")
    _barrier()


def check_wrapper_guards_a_non_contiguous_input() -> None:
    """The wrapper rejects a strided view: the op reads packed memory, and on
    the MPI path a strided view was reduced over the wrong elements.

    Only the guard is checked here. On this path strategy 0 sums a clone of
    the input, which is contiguous, so the bare op does not show the defect the
    guard is for; the guard is what this test protects.
    """
    base = _payload(RANK, (8, 2 * HIDDEN), 91000, torch.float32)
    view = base[:, ::2]
    assert not view.is_contiguous()
    try:
        with validating(allreduce):
            _call(view, 0)
    except AssertionError as err:
        assert "contiguous" in str(err), err
    else:
        raise AssertionError("the wrapper accepted a non-contiguous input")
    _barrier()


def check_wrapper_guards_float8() -> None:
    """The wrapper rejects e4m3, which the MPI path accepted and summed as
    bytes rather than floats.

    Only the guard is checked here, for the same reason as above: what the
    ProcessGroup's own allreduce does with a float8 tensor is not this entry's
    contract.
    """
    x = torch.full((4, HIDDEN), float(RANK + 1), device="cuda").to(torch.float8_e4m3fn)
    try:
        with validating(allreduce):
            _call(x, 0)
    except AssertionError as err:
        assert "float8_e4m3fn" in str(err), err
    else:
        raise AssertionError("the wrapper accepted a float8_e4m3fn input")
    _barrier()


def check_oneshot_reads_stale_data_after_an_element_size_change() -> None:
    """A known defect, made visible: the harness must see it for a clean run to count.

    ONESHOT resets the buffer its previous call used, sizing the reset as that
    call's element count times the CURRENT call's element size. So after an
    fp32 call, a bf16 call resets only half of the fp32 call's bytes, and the
    call that next reuses that buffer takes the unreset half for data a peer
    has already written. With one rank arriving late, every other rank sums
    that rank's contribution to an earlier call instead of this one.

    The identical sequence with fp32 throughout is the control, and it must be
    exact; the mixed one must be wrong on every on-time rank. This is why the
    other suites give each element size its own workspace, and why the
    contract routes the bf16 rows of a mixed-dtype forward through NCCL.
    """
    rows = 128
    late = WORLD - 1

    def sequence(mid_dtype: torch.dtype, seed: int) -> float:
        _call(_payload(RANK, (rows, HIDDEN), seed, torch.float32), 4)
        _call(_payload(RANK, (1, HIDDEN), seed + 1, mid_dtype), 4)
        _call(_payload(RANK, (1, HIDDEN), seed + 2, mid_dtype), 4)
        x = _payload(RANK, (rows, HIDDEN), seed + 3, torch.float32)
        torch.cuda.synchronize()
        _barrier()
        if RANK == late:
            torch.cuda._sleep(200_000_000)  # hold this rank's push back
        out = _call(x, 4)
        torch.cuda.synchronize()
        ref = _sum_ref((rows, HIDDEN), seed + 3, torch.float32)
        return (out != ref).float().mean().item()

    for trial in range(3):
        fractions = _host_allgather(sequence(torch.float32, 95000 + trial * 10))
        assert all(f == 0.0 for f in fractions), (
            f"control (fp32 throughout) trial {trial}: {fractions}"
        )
    for trial in range(3):
        fractions = _host_allgather(sequence(torch.bfloat16, 96000 + trial * 10))
        on_time = [f for r, f in enumerate(fractions) if r != late]
        assert all(f > 0.9 for f in on_time), (
            f"mixed trial {trial}: the stale read did not show on every on-time rank: {fractions}"
        )
    _barrier()


def _sweep_table(
    hidden: int, seed: int, dtype: torch.dtype, n_max: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This rank's payload rows and their exact sum over the group, n_max + SWEEP_SLACK rows.

    The sum is exact in `dtype` (see `_payload`), so it is stored in `dtype`.
    """
    shape = (n_max + SWEEP_SLACK, hidden)
    return _payload(RANK, shape, seed, dtype), _sum_ref(shape, seed, dtype)


def _sweep_offsets(k: int) -> Tuple[int, int, int]:
    """Row offsets of step k's three calls into their tables.

    Each call reads its rows at its own offset, so no two nearby calls hand the
    op the same values at the same row: a result left over from an earlier
    call (the stale read of the mixed suite) cannot pass for this one.
    """
    return (k * 131) % SWEEP_SLACK, (k * 173 + 61) % SWEEP_SLACK, (k * 197 + 263) % SWEEP_SLACK


def check_core_tp_sums_at_every_row_count() -> None:
    """Every T in 1..N_MAX, the core target's layer sequence, twice over.

    AUTO picks its transport from (T, hidden), and NCCL its algorithm and
    protocol from the message size, so every row count is run, not sampled:
    per step, the engram rows' bf16 [T, 6144] sum on NCCL (T up to
    ENGRAM_N_MAX, the engine's max_num_tokens), the attention's fp32 [T, 5120]
    sum on CORE_FP32, the MoE's fp32 [T, 5120] sum on CORE_FP32 -- the order a
    tensor-parallel engram layer issues them in, the fp32 ones on one workspace
    when the group has one. Above ENGRAM_N_MAX a step is a decoder-pass layer
    (21-39): the two fp32 sums only, up to N_MAX. First with T ascending, then
    in a seeded shuffle, so every band of AUTO_FP32_BANDS is entered from every
    other band (a prefill chunk between decode steps). Every element of every
    result is compared with the exact sum; the mismatch counts stay on the
    device and are read once per pass.

    This is the driver of the entry's two core cells for this layout: their
    strategy, shape and row range are asserted to be the ones swept here.
    """
    core = {c.spec["dtype"]: c.spec for c in _layout_cells("core")}
    assert set(core) == {torch.float32, torch.bfloat16}, list(core)
    for dtype, strategy, trailing, n_max in (
        (torch.float32, CORE_FP32, (HIDDEN,), N_MAX),
        (torch.bfloat16, NCCL, (ENGRAM_HIDDEN,), ENGRAM_N_MAX),
    ):
        spec = core[dtype]
        assert spec["strategy"] == strategy, (dtype, spec["strategy"], strategy)
        assert tuple(spec["trailing"]) == trailing, (dtype, spec["trailing"])
        assert list(spec["rows"]) == list(range(1, n_max + 1)), (dtype, spec["rows"])
    with validating(allreduce):
        # The guard admits both core forms; the sweep itself runs unguarded.
        _call(_payload(RANK, (1, HIDDEN), 1, torch.float32), CORE_FP32)
        _call(_payload(RANK, (1, ENGRAM_HIDDEN), 1, torch.bfloat16), NCCL)
    torch.cuda.synchronize()
    _barrier()

    wo, wo_sum = _sweep_table(HIDDEN, 120001, torch.float32, N_MAX)
    moe, moe_sum = _sweep_table(HIDDEN, 120002, torch.float32, N_MAX)
    eng, eng_sum = _sweep_table(ENGRAM_HIDDEN, 120003, torch.bfloat16, ENGRAM_N_MAX)

    # The gate is not blind to a stale result: at every band edge, the right
    # answer of a step differs on nearly every element from the answers of the
    # calls before it on the same transport (the previous three fp32 sums on
    # the workspace, the previous two bf16 sums).
    k = 1000
    for t in sorted(
        {1, 8, ENGRAM_N_MAX, N_MAX} | {edge for band in AUTO_FP32_BANDS for edge in band[:2]}
    ):
        o_wo = _sweep_offsets(k)[0]
        o_eng = _sweep_offsets(k)[1]
        wo_now = wo_sum[o_wo : o_wo + t]
        earlier = [
            ("fp32", wo_now, moe_sum, _sweep_offsets(k - 1)[2]),
            ("fp32", wo_now, wo_sum, _sweep_offsets(k - 1)[0]),
            ("fp32", wo_now, moe_sum, _sweep_offsets(k - 2)[2]),
        ]
        if t <= ENGRAM_N_MAX:
            earlier += [
                ("bf16", eng_sum[o_eng : o_eng + t], eng_sum, _sweep_offsets(k - 1)[1]),
                ("bf16", eng_sum[o_eng : o_eng + t], eng_sum, _sweep_offsets(k - 2)[1]),
            ]
        for kind, now, table, o in earlier:
            fraction = (now != table[o : o + t]).float().mean().item()
            assert fraction > 0.9, f"{kind} T={t}: a stale result differs on only {fraction:.4f}"

    ascending = list(range(1, N_MAX + 1))
    shuffled = list(ascending)
    random.Random(4242).shuffle(shuffled)
    k = 0
    for name, order in (("ascending", ascending), ("shuffled", shuffled)):
        bad = torch.zeros(3, N_MAX + 1, dtype=torch.int64, device="cuda")
        calls = 0
        for t in order:
            k += 1
            o_wo, o_eng, o_moe = _sweep_offsets(k)
            if t <= ENGRAM_N_MAX:
                e = _call(eng[o_eng : o_eng + t], NCCL)
                assert e.shape == (t, ENGRAM_HIDDEN) and e.dtype == torch.bfloat16, (
                    t,
                    e.shape,
                    e.dtype,
                )
                bad[1, t] += (e != eng_sum[o_eng : o_eng + t]).sum()
                calls += 1
            a = _call(wo[o_wo : o_wo + t], CORE_FP32)
            b = _call(moe[o_moe : o_moe + t], CORE_FP32)
            assert a.shape == (t, HIDDEN) and a.dtype == torch.float32, (t, a.shape, a.dtype)
            assert b.shape == (t, HIDDEN) and b.dtype == torch.float32, (t, b.shape, b.dtype)
            bad[0, t] += (a != wo_sum[o_wo : o_wo + t]).sum()
            bad[2, t] += (b != moe_sum[o_moe : o_moe + t]).sum()
            calls += 2
        torch.cuda.synchronize()
        wrong = bad.nonzero().tolist()
        fp32_name = STRATEGIES[CORE_FP32]
        labels = (f"attention fp32 {fp32_name}", "engram bf16 NCCL", f"MoE fp32 {fp32_name}")
        assert not wrong, f"{name}: wrong sums at " + ", ".join(
            f"{labels[i]} T={t} ({bad[i, t].item()} elements)" for i, t in wrong[:20]
        )
        if RANK == 0:
            print(
                f"[rank 0] core sweep {name}: fp32 {fp32_name} T=1..{N_MAX}, engram bf16 NCCL T=1..{ENGRAM_N_MAX}, "
                f"{calls} calls, every element exact",
                flush=True,
            )
        _barrier()
    del wo, wo_sum, moe, moe_sum, eng, eng_sum
    torch.cuda.empty_cache()
    _barrier()


def _core_forward(
    f_in: List[torch.Tensor], e_in: List[torch.Tensor]
) -> List[Tuple[str, int, torch.Tensor]]:
    """One decode forward's all-reduce sites, in the core target's order.

    The 80 fp32 sites cycle over the buffers of `f_in`; engram layer j reads
    `e_in[j]`. Returns (kind, buffer index, result) per site.
    """
    sites: List[Tuple[str, int, torch.Tensor]] = []
    fp32_site = 0
    for layer in range(CORE_LAYERS):
        if layer in ENGRAM_LAYER_IDS:
            j = ENGRAM_LAYER_IDS.index(layer)
            sites.append(("bf16", j, _call(e_in[j], NCCL)))
        for _ in range(2):  # attention output projection, then routed experts
            i = fp32_site % len(f_in)
            sites.append(("fp32", i, _call(f_in[i], CORE_FP32)))
            fp32_site += 1
    return sites


def check_core_tp_decode_graphs_replay_bitwise_equal_to_eager() -> None:
    """One decode graph per CORE_CAPTURE_ROWS entry, the core target's 82 sites each.

    The engine's state after warm-up: every graph captured (largest first) into
    one shared pool and kept alive. Then three passes over all of them in
    shuffled order, each with new data copied into the captured inputs:

    1. exact payloads, one graph at a time, with an eager prefill-sized sum
       (PREFILL_BETWEEN_REPLAYS, rotating) on both call forms between replays:
       every site bitwise equal to the exact sum;
    2. exact payloads, every graph replayed back to back with nothing
       synchronizing, then all checked;
    3. unrestricted normal payloads, where the sum order shows in the bits:
       every site bitwise equal to an eager call on the same buffer afterwards.

    Before every replay of passes 1 and 2, every site's output is checked to
    differ from the new exact sum on more than 90% of its elements, so a replay
    that did not run (a stale output) cannot pass.
    """
    distinct = 8  # fp32 input buffers per graph, cycled over its 80 fp32 sites
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for t in CORE_CAPTURE_ROWS:
            _call(_payload(RANK, (t, HIDDEN), 1, torch.float32), CORE_FP32)
            _call(_payload(RANK, (t, ENGRAM_HIDDEN), 2, torch.bfloat16), NCCL)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    _barrier()

    pool = None
    graphs: Dict[int, torch.cuda.CUDAGraph] = {}
    f_in: Dict[int, List[torch.Tensor]] = {}
    e_in: Dict[int, List[torch.Tensor]] = {}
    sites: Dict[int, List[Tuple[str, int, torch.Tensor]]] = {}
    for t in sorted(CORE_CAPTURE_ROWS, reverse=True):
        f_in[t] = [_payload(RANK, (t, HIDDEN), 30 + i, torch.float32) for i in range(distinct)]
        e_in[t] = [
            _payload(RANK, (t, ENGRAM_HIDDEN), 50 + j, torch.bfloat16)
            for j in range(len(ENGRAM_LAYER_IDS))
        ]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            sites[t] = _core_forward(f_in[t], e_in[t])
        assert len(sites[t]) == 2 * CORE_LAYERS + len(ENGRAM_LAYER_IDS), len(sites[t])
        pool = graph.pool() if pool is None else pool
        graphs[t] = graph
        torch.cuda.synchronize()
        _barrier()

    def stage_exact(t: int, seed: int) -> Dict[Tuple[str, int], torch.Tensor]:
        refs = {}
        for i, x in enumerate(f_in[t]):
            x.copy_(_payload(RANK, (t, HIDDEN), seed + i, torch.float32))
            refs[("fp32", i)] = _sum_ref((t, HIDDEN), seed + i, torch.float32)
        for j, x in enumerate(e_in[t]):
            x.copy_(_payload(RANK, (t, ENGRAM_HIDDEN), seed + 100 + j, torch.bfloat16))
            refs[("bf16", j)] = _sum_ref((t, ENGRAM_HIDDEN), seed + 100 + j, torch.bfloat16)
        return refs

    def verify(t: int, refs: Dict[Tuple[str, int], torch.Tensor], where: str) -> None:
        bad = torch.zeros((), dtype=torch.int64, device="cuda")
        for kind, i, out in sites[t]:
            bad += (out != refs[(kind, i)]).sum()
        assert bad.item() == 0, (
            f"{where} T={t}: {bad.item()} wrong elements over {len(sites[t])} sites"
        )

    stale_worst = [1.0]

    def assert_stale_is_visible(
        t: int, refs: Dict[Tuple[str, int], torch.Tensor], where: str
    ) -> None:
        """Before the replay: what the outputs hold now must fail the gate."""
        fractions = torch.stack(
            [(out != refs[(kind, i)]).double().mean() for kind, i, out in sites[t]]
        )
        worst = fractions.min().item()
        stale_worst[0] = min(stale_worst[0], worst)
        assert worst > 0.9, (
            f"{where} T={t}: a stale output differs from the new sum on only {worst:.4f}"
        )

    rng = random.Random(777)
    order = list(CORE_CAPTURE_ROWS)
    # Pass 1: checked one graph at a time, prefill-sized eager traffic in between.
    rng.shuffle(order)
    for pos, t in enumerate(order):
        refs = stage_exact(t, 200000 + 1000 * t)
        assert_stale_is_visible(t, refs, "pass 1 (before replay)")
        graphs[t].replay()
        prefill = PREFILL_BETWEEN_REPLAYS[pos % len(PREFILL_BETWEEN_REPLAYS)]
        seed = 300000 + pos
        eager_f = _call(_payload(RANK, (prefill, HIDDEN), seed, torch.float32), CORE_FP32)
        eager_e = None
        if prefill <= ENGRAM_N_MAX:
            eager_e = _call(_payload(RANK, (prefill, ENGRAM_HIDDEN), seed, torch.bfloat16), NCCL)
        torch.cuda.synchronize()
        verify(t, refs, "pass 1 (checked)")
        _assert_bitwise(
            eager_f,
            _sum_ref((prefill, HIDDEN), seed, torch.float32),
            f"pass 1 eager fp32 T={prefill}",
        )
        if eager_e is not None:
            _assert_bitwise(
                eager_e,
                _sum_ref((prefill, ENGRAM_HIDDEN), seed, torch.bfloat16),
                f"pass 1 eager bf16 T={prefill}",
            )
        del eager_f, eager_e
        _barrier()
    # Pass 2: every graph back to back, nothing synchronizing between replays.
    rng.shuffle(order)
    staged = {t: stage_exact(t, 400000 + 1000 * t) for t in order}
    for t in order:
        assert_stale_is_visible(t, staged[t], "pass 2 (before replay)")
    for t in order:
        graphs[t].replay()
    torch.cuda.synchronize()
    for t in order:
        verify(t, staged[t], "pass 2 (burst)")
    del staged
    _barrier()
    # Pass 3: unrestricted payloads; the replay must be bit for bit what an
    # eager call on the same buffer returns.
    rng.shuffle(order)
    for pos, t in enumerate(order):
        for i, x in enumerate(f_in[t]):
            x.copy_(_random_parts((t, HIDDEN), 500000 + 1000 * t + i, torch.float32)[RANK])
        for j, x in enumerate(e_in[t]):
            x.copy_(_random_parts((t, ENGRAM_HIDDEN), 600000 + 1000 * t + j, torch.bfloat16)[RANK])
        graphs[t].replay()
        torch.cuda.synchronize()
        eager = {("fp32", i): _call(x, CORE_FP32) for i, x in enumerate(f_in[t])}
        eager.update({("bf16", j): _call(x, NCCL) for j, x in enumerate(e_in[t])})
        torch.cuda.synchronize()
        for s, (kind, i, out) in enumerate(sites[t]):
            assert torch.equal(
                out.view(torch.int32 if kind == "fp32" else torch.int16),
                eager[(kind, i)].view(torch.int32 if kind == "fp32" else torch.int16),
            ), f"pass 3 T={t} site={s} ({kind}): replay differs from eager on the same data"
        _assert_same_bits_on_every_rank(sites[t][0][2], f"pass 3 T={t} first site")
        _barrier()
    if RANK == 0:
        print(
            f"[rank 0] core graphs: {len(graphs)} captured at {list(CORE_CAPTURE_ROWS)}; before every replay of "
            "passes 1-2 a stale output "
            f"differed from the new reference on at least {stale_worst[0]:.4f} of its elements",
            flush=True,
        )
    for held in (graphs, sites, f_in, e_in):
        held.clear()
    torch.cuda.synchronize()
    _barrier()


def check_engram_rows_on_auto_corrupt_the_next_fp32_sum() -> None:
    """The wrong strategy for the engram rows, run: the control for the core rule.

    The core target's layer order is MoE fp32 sum, then on an engram layer the
    engram rows' bf16 sum, the attention fp32 sum and the MoE fp32 sum. With
    the engram sum on NCCL (the target's choice) all four are exact even with
    one rank late to the last one. With the engram sum on AUTO -- the same
    workspace as the fp32 sums -- the one-shot kernel's reset after the bf16
    call covers half of the fp32 buffer before it, and the last fp32 sum, which
    reuses that buffer, returns the stale half on every on-time rank.
    Runs in a process of its own: it leaves the workspace inconsistent.
    """
    late = WORLD - 1

    def layer(engram_strategy: int, t: int, seed: int) -> Tuple[List[float], float]:
        prev = _call(_payload(RANK, (t, HIDDEN), seed, torch.float32), AUTO)
        eng = _call(_payload(RANK, (t, ENGRAM_HIDDEN), seed + 1, torch.bfloat16), engram_strategy)
        attn = _call(_payload(RANK, (t, HIDDEN), seed + 2, torch.float32), AUTO)
        torch.cuda.synchronize()
        head = [
            (prev != _sum_ref((t, HIDDEN), seed, torch.float32)).float().mean().item(),
            (eng != _sum_ref((t, ENGRAM_HIDDEN), seed + 1, torch.bfloat16)).float().mean().item(),
            (attn != _sum_ref((t, HIDDEN), seed + 2, torch.float32)).float().mean().item(),
        ]
        x = _payload(RANK, (t, HIDDEN), seed + 3, torch.float32)
        torch.cuda.synchronize()
        _barrier()
        if RANK == late:
            torch.cuda._sleep(200_000_000)  # hold this rank's push back
        moe = _call(x, AUTO)
        torch.cuda.synchronize()
        return head, (moe != _sum_ref((t, HIDDEN), seed + 3, torch.float32)).float().mean().item()

    rows = (1, 8)
    for t in rows:
        for trial in range(3):
            head, last = layer(NCCL, t, 130000 + 10 * trial + 100 * t)
            got = _host_allgather((head, last))
            assert all(h == [0.0, 0.0, 0.0] and tail == 0.0 for h, tail in got), (
                f"engram on NCCL, T={t} trial {trial}: expected every sum exact, got {got}"
            )
    for t in rows:
        for trial in range(3):
            _, last = layer(AUTO, t, 140000 + 10 * trial + 100 * t)
            got = _host_allgather(last)
            on_time = [f for r, f in enumerate(got) if r != late]
            assert all(f > 0.9 for f in on_time), (
                f"engram on AUTO, T={t} trial {trial}: the stale read did not show on every on-time rank: {got}"
            )
    _barrier()


SUITE_CHECKS: Dict[str, Tuple[Callable[[], None], ...]] = {
    "fp32": (
        check_certified_cells_fp32,
        check_fp32_partial_sums_are_reduced_in_fp32,
        check_trigger_completion_at_end_false_gives_the_same_bits,
        check_engram_rows_owned_by_one_rank_on_the_workspace_free_strategies,
        check_fp32_on_the_workspace_interleaved_with_bf16_on_nccl,
        check_cuda_graph_decode_graphs_replay_in_any_order,
        check_the_gate_discriminates_a_wrong_reduce,
        check_wrapper_guards_a_non_contiguous_input,
        check_wrapper_guards_float8,
    ),
    "bf16": (
        check_certified_cells_bf16,
        check_engram_rows_owned_by_one_rank_on_every_strategy,
    ),
    # The DeepSeek-V4.1 core target's tensor-parallel surface:
    # every row count, and every capture batch size, of its two call forms.
    "core": (
        check_core_tp_sums_at_every_row_count,
        check_core_tp_decode_graphs_replay_bitwise_equal_to_eager,
    ),
    # Each in a process of its own: they deliberately leave the workspace's
    # ONESHOT state inconsistent.
    "mixed": (check_oneshot_reads_stale_data_after_an_element_size_change,),
    "mixed_core": (check_engram_rows_on_auto_corrupt_the_next_fp32_sum,),
}


# Checks and suites about the IPC workspace. A group that spans nodes has none
# (the target's `ar_one_node` is False there and it sums on NCCL), so for such a
# group these are not run: the checks are dropped from their suite, and the
# suites refuse to start.
WORKSPACE_CHECKS = (
    check_trigger_completion_at_end_false_gives_the_same_bits,
    check_fp32_on_the_workspace_interleaved_with_bf16_on_nccl,
)
WORKSPACE_SUITES = ("mixed", "mixed_core")


def _suite_checks(suite: str) -> Tuple[Callable[[], None], ...]:
    """The checks of `suite` that apply to this group."""
    if ONE_NODE:
        return SUITE_CHECKS[suite]
    assert suite not in WORKSPACE_SUITES, (
        f"suite {suite} is about the IPC workspace, which a group spanning nodes does not have"
    )
    return tuple(c for c in SUITE_CHECKS[suite] if c not in WORKSPACE_CHECKS)


def _engine_dist(mapping: Any) -> Any:
    """The engine's TorchDist for `mapping`, with its cluster probe read from hostnames.

    TorchDist asks Ray for each rank's node and GPU; these ranks are not Ray
    actors, so the same (node, device) list comes from a host-side gather
    instead. Everything else is the engine's own code: the device mesh the
    Mapping's process groups come from, the per-node local group, `init_pg`
    (which the `_pg` op resolves ranks and the node topology through), and
    `tp_allgather` (which the IPC workspace exchanges its handles through).
    """
    from tensorrt_llm._torch.distributed.communicator import TorchDist

    class _TorchDistOnHostnames(TorchDist):
        def _get_cluster_info(self):
            if self.cluster_info is None:
                info: List[Any] = [None] * dist.get_world_size()
                dist.all_gather_object(info, (socket.gethostname(), torch.cuda.current_device()))
                self.cluster_info = info
            return self.cluster_info

    return _TorchDistOnHostnames(mapping)


def _run_one_rank(suite: str, group_size: Optional[int]) -> int:
    """Body of one rank: run the suite's checks; a failing rank exits nonzero.

    The world is cut into contiguous groups of `group_size` ranks (all of it
    when None), as `Mapping.tp_group` cuts a pipeline-parallel world into its
    stages' TP groups; every group runs the suite at once on its own ranks.
    The rank's device is its node-local rank (LOCAL_RANK under torchrun, the
    global rank under this file's own launcher), so the same body runs under
    both. Rendezvous is torch.distributed's env:// (RANK, WORLD_SIZE,
    MASTER_ADDR, MASTER_PORT), with MPI disabled.
    """
    global \
        HOST_GROUP, \
        RANK, \
        WORLD, \
        GROUP, \
        WORKSPACE, \
        ONE_NODE, \
        STRATEGIES, \
        CORE_FP32, \
        AUTO_FP32_BANDS, \
        LAYOUT, \
        allreduce, \
        validating
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    world_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", world_rank))
    torch.cuda.set_device(local_rank)
    # The backend pair the engine's Ray workers initialize: the `_pg` op's node
    # topology probe exchanges host values over the world group, which needs a
    # CPU backend beside NCCL. device_id keeps NCCL communicator teardown off the
    # implicit-device path.
    dist.init_process_group(
        backend="cuda:nccl,cpu:gloo",
        init_method="env://",
        device_id=torch.device("cuda", local_rank),
    )

    from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.comm import (  # noqa: E501
        allreduce as entry,
    )
    from tensorrt_llm._torch.distributed import communicator
    from tensorrt_llm._torch.distributed.ops import get_allreduce_workspace
    from tensorrt_llm.mapping import Mapping

    # This file runs as a spawned rank process, not under pytest, so the repo's
    # `__extra_import_path__` mechanism is not in play and the shared helper is
    # reached the plain way. The mutation is confined to these rank processes.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from _validating import validating as _validating_cm

    allreduce, validating = entry.allreduce, _validating_cm
    WORLD = group_size or world_size
    assert WORLD >= 2 and world_size % WORLD == 0, (
        f"a collective needs at least 2 ranks, and the world ({world_size}) must split into "
        f"groups of {WORLD}"
    )
    stage = world_rank // WORLD
    RANK = world_rank - stage * WORLD
    GROUP = list(range(stage * WORLD, (stage + 1) * WORLD))

    hosts: List[Any] = [None] * world_size
    dist.all_gather_object(hosts, socket.gethostname())
    nodes = len(set(hosts))
    assert world_size % nodes == 0, f"uneven ranks per node: {hosts}"
    ONE_NODE = len({hosts[r] for r in GROUP}) == 1
    layouts = {
        (c.spec["world_size"], c.spec["group_size"], c.spec["nodes"]): c.spec["layout"]
        for c in allreduce.CELLS
    }
    assert (world_size, WORLD, nodes) in layouts, (
        f"no cells at world size {world_size}, group size {WORLD}, {nodes} node(s)"
    )
    LAYOUT = layouts[(world_size, WORLD, nodes)]

    # One gloo group per block for the host-side checks. Every rank must enter
    # new_group for every block, in the same order, or it deadlocks.
    for first in range(0, world_size, WORLD):
        block = dist.new_group(list(range(first, first + WORLD)), backend="gloo")
        if first == stage * WORLD:
            HOST_GROUP = block

    # The Mapping the target builds for its group (a pipeline-parallel world
    # passes its pp_size), and the engine's TorchDist over it. The IPC
    # workspace reaches the engine's communicator object through
    # `Distributed.get`, which would build a Ray-backed one; it gets this one.
    mapping = Mapping(
        world_size=world_size,
        rank=world_rank,
        gpus_per_node=world_size // nodes,
        tp_size=WORLD,
        pp_size=world_size // WORLD,
    )
    engine_dist = _engine_dist(mapping)

    def _the_engine_dist(m: Any) -> Any:
        assert m == mapping, f"the workspace asked for another Mapping: {m}"
        return engine_dist

    communicator.Distributed.get = staticmethod(_the_engine_dist)
    assert list(mapping.tp_group) == GROUP, (mapping.tp_group, GROUP)
    # One binding instead of a per-call argument: every check's call site
    # keeps the certified shape, and the wrapper hands the group's boxed
    # ProcessGroup to the `_pg` symbol -- the routing the product's non-MPI
    # mode does in _torch/distributed/ops.py.
    allreduce.bind_const(process_group=mapping.tp_group_pg.boxed())

    if ONE_NODE:
        assert WORLD in AUTO_FP32_BANDS_BY_GROUP, (
            f"AUTO's bands at group size {WORLD} were never read"
        )
        STRATEGIES, CORE_FP32, AUTO_FP32_BANDS = (
            dict(ALL_STRATEGIES),
            AUTO,
            AUTO_FP32_BANDS_BY_GROUP[WORLD],
        )
        # Real state: the IPC workspace the target builds for its group.
        WORKSPACE = get_allreduce_workspace(mapping)
    else:
        STRATEGIES = {k: v for k, v in ALL_STRATEGIES.items() if k not in WORKSPACE_STRATEGIES}
        CORE_FP32, AUTO_FP32_BANDS, WORKSPACE = NCCL, NCCL_BANDS, None
    checks = _suite_checks(suite)
    # One eager call before any check (and so before any capture).
    _call(torch.zeros(1, HIDDEN, device="cuda"), 0)
    torch.cuda.synchronize()
    major, minor = torch.cuda.get_device_capability()
    who: List[Any] = [None] * world_size
    dist.all_gather_object(
        who,
        f"world rank {world_rank} host {socket.gethostname()} device {local_rank} "
        f"({torch.cuda.get_device_name()} sm_{major}{minor}) group {GROUP} position {RANK} "
        f"entry={entry.__file__}",
    )
    if world_rank == 0:
        print(
            f"[rank 0] suite {suite} layout {LAYOUT}: sm_{major}{minor} "
            f"{torch.cuda.get_device_name()} world_size={world_size} group_size={WORLD} "
            f"groups={[list(range(g, g + WORLD)) for g in range(0, world_size, WORLD)]} "
            f"one_node={ONE_NODE} strategies={list(STRATEGIES.values())} "
            f"core fp32 on {STRATEGIES[CORE_FP32]} entry={entry.__file__}",
            flush=True,
        )
        print("\n".join(f"[rank 0]   {line}" for line in who), flush=True)
        dropped = [c.__name__ for c in SUITE_CHECKS[suite] if c not in checks]
        if dropped:
            print(
                f"[rank 0] suite {suite}: not run for a group spanning nodes "
                f"(no IPC workspace): {dropped}",
                flush=True,
            )

    for check in checks:
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {world_rank}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # Exit hard rather than return: a rank that leaves a collective
            # early wedges every sibling in it, and the launcher kills the rest
            # the moment it sees this nonzero exit.
            os._exit(13)
        if world_rank == 0:
            print(f"[rank 0] passed {check.__name__}", flush=True)
    dist.barrier()
    torch.cuda.synchronize()
    print(f"[rank {world_rank}] suite {suite}: {len(checks)} checks passed", flush=True)
    # Exit without tearing the process groups down: destroying a communicator
    # that captured collectives into CUDA graphs can wedge NCCL's watchdog, and
    # these are throwaway processes the launcher owns.
    sys.stdout.flush()
    os._exit(0)


def _group_size_arg() -> Optional[int]:
    """`--group-size G` from the command line, or None (the whole world)."""
    if _GROUP_FLAG not in sys.argv:
        return None
    return int(sys.argv[sys.argv.index(_GROUP_FLAG) + 1])


def _run_suite(suite: str, world_size: int, group_size: Optional[int]) -> None:
    """One suite: `world_size` fresh worker processes, under the suite's deadline.

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
    # Drop the launching step's PMIx/PMI/OMPI/Slurm namespace from the
    # children: they would otherwise register as participants of that step,
    # and the first hard exit (a failing rank's) trips its errhandler, which
    # kills the whole step before anything is reported.
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    group = [] if group_size is None else [_GROUP_FLAG, str(group_size)]
    command = [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG, suite, *group]
    print(f"[launcher] {world_size} x {' '.join(command)} (rendezvous on :{port})", flush=True)
    children = [
        subprocess.Popen(
            command, env=dict(env, RANK=str(r), LOCAL_RANK=str(r)), start_new_session=True
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

    per_suite_s = SUITE_DEADLINE_S[suite]
    deadline = time.monotonic() + per_suite_s
    live = set(range(world_size))
    failure = None
    while live and failure is None:
        if time.monotonic() > deadline:
            failure = f"did not finish in {per_suite_s}s (wedged)"
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
        raise AssertionError(f"the {world_size}-rank {suite} suite failed: {failure}")


def _spawn_ranks(group_size: Optional[int]) -> None:
    """Every suite in turn, one worker process per claimed device; no external launcher."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, (
        "set CUDA_VISIBLE_DEVICES to the devices this run owns, "
        "e.g. export CUDA_VISIBLE_DEVICES=0,1,2,3"
    )
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size >= 2, (
        f"CUDA_VISIBLE_DEVICES names {world_size} device(s); a collective test needs at least 2"
    )
    for suite in SUITES:
        _run_suite(suite, world_size, group_size)


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank(sys.argv[sys.argv.index(_WORKER_FLAG) + 1], _group_size_arg()))
    _spawn_ranks(_group_size_arg())
    print("OK")
