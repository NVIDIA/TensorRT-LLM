# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`tep_finalize_allreduce.build_tp_sum` (the plain Lamport TP sum) on 4 ranks, bit for bit.

Four processes, one GPU each, an NCCL group. Each rank holds a bf16 partial `[rows, 2560]` with
-0.0 lanes and values over 2^-12..2^8, at a row pitch wider than the row (a column view, as the
output projection's partial can be). For rows 1, 4, 16, 20 and 32:

* eagerly, three launches on slots 0, 1, 2: every rank's output equals the reference (each
  rank's partial, -0.0 as +0.0, summed in fp32 in rank order, rounded once), and the four ranks'
  outputs are equal;
* replayed from one CUDA graph of three launches over fresh operands, with one rank held back
  3-40 us before each replay;
* replayed back to back from one CUDA graph of `CYCLES` x 3 launches with a sleep before each
  launch on a rotating rank (each slot reused within and across replays): every launch of every
  replay equals the reference (counted on the device).

Both PDL release points (`early` False / True) run."""

import os
import random
import socket

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import tep_finalize_allreduce as op

WORLD = op.WORLD
REPLAYS = int(os.environ.get("QWEN4_EXP_TPSUM_REPLAYS", "200"))
CYCLES = 4
ROWS = (1, 4, 16, 20, op.TP_SUM_MAX_ROWS)

pytestmark = [
    pytest.mark.no_xdist,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.device_count() < WORLD
        or torch.cuda.get_device_capability() != (10, 3),
        reason="the Lamport TP sum is certified on 4 SM103 GPUs of one NVLink domain",
    ),
]


def _partial(gen, rows: int, device):
    scale = torch.exp2(torch.randint(-12, 9, [rows, 1], generator=gen).float())
    wide = (torch.randn([rows, op.HIDDEN + 64], generator=gen) * scale).to(torch.bfloat16)
    wide[:, ::37] = -0.0
    return wide.to(device)[:, : op.HIDDEN]


def _reference(mine):
    import torch.distributed as dist

    everyone = [None] * WORLD
    dist.all_gather_object(everyone, mine.cpu())
    return op.tp_sum_reference([t.to(mine.device) for t in everyone])


def _check(name: str, got: torch.Tensor, want: torch.Tensor) -> None:
    same = got.view(torch.int16) == want.view(torch.int16)
    assert bool(same.all()), f"{name}: {int((~same).sum())} of {same.numel()} lanes differ"


def _rank_main(rank: int, port: int) -> None:
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=WORLD)
    try:
        device = torch.device("cuda", rank)
        gen = torch.Generator().manual_seed(20261126 + rank)
        rnd = random.Random(20261126)
        for early in (False, True):
            s = op.build_tp_sum(rank, use_pdl=True, early=early)
            for rows in ROWS:
                tag = f"{rows} rows, early={early}"
                for slot in range(op.SLOTS):
                    x = _partial(gen, rows, device)
                    got = s(x, slot)
                    _check(f"eager {tag}, slot {slot}", got, _reference(x))
                    everyone = [torch.empty_like(got) for _ in range(WORLD)]
                    dist.all_gather(everyone, got)
                    for r, other in enumerate(everyone):
                        _check(f"eager {tag}, slot {slot}, rank {r}", got, other)
                static = [_partial(gen, rows, device) for _ in range(op.SLOTS)]
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
                    outs = [s(static[k], k) for k in range(op.SLOTS)]
                torch.cuda.current_stream().wait_stream(stream)
                for i in range(REPLAYS):
                    for d in static:
                        d.copy_(_partial(gen, rows, device))
                    late = rnd.randrange(WORLD)
                    delay = rnd.uniform(3, 40)
                    if late == rank:
                        torch.cuda._sleep(int(delay * 1900))
                    graph.replay()
                    for k in range(op.SLOTS):
                        _check(f"replay {i} {tag}, slot {k}", outs[k], _reference(static[k]))
                launches = op.SLOTS * CYCLES
                fixed = [_partial(gen, rows, device) for _ in range(launches)]
                want = [_reference(x) for x in fixed]
                graph = torch.cuda.CUDAGraph()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
                    outs = []
                    for i in range(launches):
                        if i % WORLD == rank:
                            torch.cuda._sleep(20000)
                        outs.append(s(fixed[i], i % op.SLOTS))
                torch.cuda.current_stream().wait_stream(stream)
                bad = torch.zeros((), dtype=torch.int64, device=device)
                for _ in range(REPLAYS):
                    graph.replay()
                    for got, ref in zip(outs, want):
                        bad += (got.view(torch.int16) != ref.view(torch.int16)).sum()
                assert int(bad.item()) == 0, f"back to back {tag}: {int(bad.item())} lanes differ"
        torch.cuda.synchronize(device)
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_plain_tp_sum_matches_its_reference():
    import torch.multiprocessing as mp

    mp.spawn(_rank_main, args=(_free_port(),), nprocs=WORLD, join=True)


@pytest.mark.cpu_only
def test_takes():
    x = torch.empty(32, op.HIDDEN + 64, dtype=torch.bfloat16)[:, : op.HIDDEN]
    assert op.TepTpSum.takes(x)
    assert not op.TepTpSum.takes(torch.empty(33, op.HIDDEN, dtype=torch.bfloat16))
    assert not op.TepTpSum.takes(torch.empty(4, op.HIDDEN, dtype=torch.float32))
    assert not op.TepTpSum.takes(torch.empty(4, 2048, dtype=torch.bfloat16))
