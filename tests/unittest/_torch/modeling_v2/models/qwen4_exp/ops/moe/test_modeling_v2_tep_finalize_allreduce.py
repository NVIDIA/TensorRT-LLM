# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`tep_finalize_allreduce` on 4 ranks against its reference, bit for bit.

Four processes, one GPU each, an NCCL group. Each rank holds a window's FC2 rows in a permuted
order, the routing weights and a slot -> row table in which the other ranks' slots are -1 (one
row has no local slot), with -0.0 lanes and values over 2^-12..2^8. For each cell's row count:

* eagerly, three launches on slots 0, 1, 2: every rank's output equals the reference (every
  rank's finalize in fp32 slot order, -0.0 as +0.0, summed in fp32 in rank order), and the four
  ranks' outputs are equal;
* replayed from one CUDA graph of three launches (the slot sequence a step bakes), over fresh
  operands copied into the graph's buffers, with one rank (the same choice on every rank) held
  back 3-40 us before each replay: every replay equals the reference.
* replayed back to back, with no host synchronization between replays, from one CUDA graph of
  `CYCLES` x 3 launches (each slot reused `CYCLES` times a replay, and again by the next replay)
  with a sleep before each launch on a rotating rank, so a rank runs ahead into a slot while
  another still reads the slot before it: every launch of every replay equals the reference
  (counted on the device, read once at the end).

Both release points (`early` False / True) run. `QWEN4_EXP_FINAR_REPLAYS` raises the replays
(default 300 a cell and release point) for a longer stress run.
"""

import os
import random
import socket

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import tep_finalize_allreduce as op

WORLD = op.WORLD
REPLAYS = int(os.environ.get("QWEN4_EXP_FINAR_REPLAYS", "300"))
#: Slot cycles a back-to-back graph holds.
CYCLES = 4
ROWS = sorted({cell.spec["rows"] for cell in op.tep_finalize_allreduce.CELLS})

pytestmark = [
    pytest.mark.no_xdist,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.device_count() < WORLD
        or torch.cuda.get_device_capability() != (10, 3),
        reason="the fused finalize all-reduce is certified on 4 SM103 GPUs of one NVLink domain",
    ),
]


def _operands(gen, rank: int, rows: int, device):
    """One rank's `(fc2, wts, perm)`: slot (t, k) belongs to rank (3 t + k) mod 4, row 0 to none
    of this rank's; the FC2 rows carry spare rows the table never names."""
    owner = (torch.arange(rows).unsqueeze(1) * 3 + torch.arange(op.TOP_K).unsqueeze(0)) % WORLD
    owner[0] = (rank + 1) % WORLD
    mine = owner == rank
    count = int(mine.sum())
    perm = torch.full([rows, op.TOP_K], -1, dtype=torch.int32)
    perm[mine] = torch.randperm(count + 5, generator=gen)[:count].to(torch.int32)
    scale = torch.exp2(torch.randint(-12, 9, [count + 5, 1], generator=gen).float())
    fc2 = (torch.randn([count + 5, op.HIDDEN], generator=gen) * scale).to(torch.bfloat16)
    fc2[:, ::37] = -0.0
    wts = (torch.rand([rows, op.TOP_K], generator=gen) * 0.5 + 0.01).to(torch.bfloat16)
    return fc2.to(device), wts.to(device), perm.to(device)


def _reference(ops_mine):
    import torch.distributed as dist

    everyone = [None] * WORLD
    dist.all_gather_object(everyone, tuple(t.cpu() for t in ops_mine))
    device = ops_mine[0].device
    return op.tep_finalize_allreduce.reference(
        [tuple(t.to(device) for t in ops) for ops in everyone]
    )


def _check(name: str, got: torch.Tensor, want: torch.Tensor) -> None:
    same = got.view(torch.int16) == want.view(torch.int16)
    assert bool(same.all()), f"{name}: {int((~same).sum())} of {same.numel()} lanes differ"


def _same_on_every_rank(name: str, got: torch.Tensor) -> None:
    import torch.distributed as dist

    everyone = [torch.empty_like(got) for _ in range(WORLD)]
    dist.all_gather(everyone, got)
    for r, other in enumerate(everyone):
        _check(f"{name}: rank {r}", got, other)


def _back_to_back(gen, rank: int, rows: int, finar, device, tag: str) -> None:
    """`REPLAYS` replays of one graph of `CYCLES` x 3 launches with no host synchronization
    between them; launch i sleeps first on rank i mod 4. Mismatched lanes are counted on the
    device."""
    launches = op.SLOTS * CYCLES
    static = [_operands(gen, rank, rows, device) for _ in range(launches)]
    want = [_reference(ops) for ops in static]
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        outs = []
        for i in range(launches):
            if i % WORLD == rank:
                torch.cuda._sleep(20000)  # about 10 us
            outs.append(op.tep_finalize_allreduce(*static[i], rows, i % op.SLOTS, finar))
    torch.cuda.current_stream().wait_stream(stream)
    bad = torch.zeros((), dtype=torch.int64, device=device)
    for _ in range(REPLAYS):
        graph.replay()
        for got, ref in zip(outs, want):
            bad += (got.view(torch.int16) != ref.view(torch.int16)).sum()
    assert int(bad.item()) == 0, f"back to back {tag}: {int(bad.item())} lanes differ"


def _rank_main(rank: int, port: int) -> None:
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=WORLD)
    try:
        device = torch.device("cuda", rank)
        gen = torch.Generator().manual_seed(20261125 + rank)
        rnd = random.Random(20261125)  # the same late rank on every rank
        for early in (False, True):
            finar = op.build(rank, use_pdl=True, early=early)
            for rows in ROWS:
                tag = f"{rows} rows, early={early}"
                for slot in range(op.SLOTS):
                    ops_mine = _operands(gen, rank, rows, device)
                    got = op.tep_finalize_allreduce(*ops_mine, rows, slot, finar)
                    _check(f"eager {tag}, slot {slot}", got, _reference(ops_mine))
                    _same_on_every_rank(f"eager {tag}, slot {slot}", got)

                static = [_operands(gen, rank, rows, device) for _ in range(op.SLOTS)]
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
                    outs = [
                        op.tep_finalize_allreduce(*static[s], rows, s, finar)
                        for s in range(op.SLOTS)
                    ]
                torch.cuda.current_stream().wait_stream(stream)
                for i in range(REPLAYS):
                    fresh = [_operands(gen, rank, rows, device) for _ in range(op.SLOTS)]
                    for dst, src in zip(static, fresh):
                        # A rank's operand shapes depend on the row count alone.
                        for d, s_ in zip(dst, src):
                            d.copy_(s_)
                    late = rnd.randrange(WORLD)
                    if late == rank:
                        torch.cuda._sleep(int(rnd.uniform(3, 40) * 1900))
                    else:
                        rnd.uniform(3, 40)
                    graph.replay()
                    for s in range(op.SLOTS):
                        _check(f"replay {i} {tag}, slot {s}", outs[s], _reference(static[s]))
                _back_to_back(gen, rank, rows, finar, device, tag)
        torch.cuda.synchronize(device)
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_fused_finalize_allreduce_matches_its_reference():
    import torch.multiprocessing as mp

    mp.spawn(_rank_main, args=(_free_port(),), nprocs=WORLD, join=True)
