# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`dep_lamport_round_trip` on 4 ranks against NCCL's ring collectives, bit for bit.

Four processes, one GPU each, an NCCL group with `NCCL_ALGO=Ring`. For each cell's row count,
eagerly and then replayed from one CUDA graph (the three slots rotating across replays):

* the gather of every rank's rows equals `all_gather_into_tensor` of the rows with -0.0 lanes as
  +0.0 (the transport's marker), and the entry's reference;
* the reduce-scatter of every rank's partials equals `reduce_scatter_tensor` of the same
  sanitized partials (the ring's sum order, bf16 after every add), and the reference.

The inputs carry -0.0 lanes on purpose. Past `MAX_ROWS` rows a rank `takes` refuses the call
(the caller keeps NCCL). `QWEN4_EXP_LAMPORT_REPLAYS` raises the graph replays (default 500 a cell)
for a longer stress run.
"""

import os
import socket

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import dep_lamport_round_trip as op

WORLD = 4
HIDDEN = 2560
REPLAYS = int(os.environ.get("QWEN4_EXP_LAMPORT_REPLAYS", "500"))
ROWS = sorted({cell.spec["rows"] for cell in op.dep_lamport_round_trip.CELLS})

pytestmark = [
    pytest.mark.no_xdist,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.device_count() < WORLD
        or torch.cuda.get_device_capability() != (10, 3),
        reason="the Lamport round trip is certified on 4 SM103 GPUs of one NVLink domain",
    ),
]


def _inputs(gen, rows: int, device) -> torch.Tensor:
    """bf16 `[rows, HIDDEN]` with about 1% -0.0 lanes."""
    x = torch.randn([rows, HIDDEN], generator=gen, device=device).to(torch.bfloat16)
    zero = torch.rand([rows, HIDDEN], generator=gen, device=device) < 0.01
    return torch.where(zero, torch.full_like(x, -0.0), x)


def _check(name: str, got: torch.Tensor, want: torch.Tensor) -> None:
    same = got.view(torch.int16) == want.view(torch.int16)
    assert bool(same.all()), f"{name}: {int((~same).sum())} of {same.numel()} lanes differ"


def _round(trip, rows, x, y):
    """One gather and one reduce-scatter through the entry, and both through NCCL."""
    import torch.distributed as dist

    gathered = op.dep_lamport_round_trip(x, trip, False)
    reduced = op.dep_lamport_round_trip(y, trip, True)
    nccl_gathered = torch.empty([WORLD * rows, HIDDEN], dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(nccl_gathered, op._sanitized(x))
    nccl_reduced = torch.empty([rows, HIDDEN], dtype=y.dtype, device=y.device)
    dist.reduce_scatter_tensor(nccl_reduced, op._sanitized(y))
    return gathered, reduced, nccl_gathered, nccl_reduced


def _rank_main(rank: int, port: int) -> None:
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), NCCL_ALGO="Ring")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=WORLD)
    try:
        device = torch.device("cuda", rank)
        trip = op.build(HIDDEN, op.MAX_ROWS, rank, WORLD)
        probe = torch.empty([1, HIDDEN], dtype=torch.bfloat16, device=device)
        assert trip.takes(op.MAX_ROWS, probe) and not trip.takes(op.MAX_ROWS + 1, probe)
        gen = torch.Generator(device=device).manual_seed(20261125 + rank)
        for rows in ROWS:
            # Eager: every rank's inputs gathered once more for the reference.
            x, y = _inputs(gen, rows, device), _inputs(gen, WORLD * rows, device)
            gathered, reduced, nccl_gathered, nccl_reduced = _round(trip, rows, x, y)
            xs = [torch.empty_like(x) for _ in range(WORLD)]
            ys = [torch.empty_like(y) for _ in range(WORLD)]
            dist.all_gather(xs, x)
            dist.all_gather(ys, y)
            _check(f"gather {rows} rows vs NCCL", gathered, nccl_gathered)
            _check(
                f"gather {rows} rows vs reference",
                gathered,
                op.dep_lamport_round_trip.reference(xs, rank, False),
            )
            _check(f"reduce-scatter {rows} rows vs NCCL", reduced, nccl_reduced)
            _check(
                f"reduce-scatter {rows} rows vs reference",
                reduced,
                op.dep_lamport_round_trip.reference(ys, rank, True),
            )

            # Replayed from one graph: fresh inputs copied into its static buffers.
            sx, sy = torch.zeros_like(x), torch.zeros_like(y)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    g_out = op.dep_lamport_round_trip(sx, trip, False)
                    r_out = op.dep_lamport_round_trip(sy, trip, True)
            torch.cuda.current_stream().wait_stream(stream)
            for i in range(REPLAYS):
                x, y = _inputs(gen, rows, device), _inputs(gen, WORLD * rows, device)
                sx.copy_(x)
                sy.copy_(y)
                graph.replay()
                nccl_gathered = torch.empty_like(g_out)
                dist.all_gather_into_tensor(nccl_gathered, op._sanitized(x))
                nccl_reduced = torch.empty_like(r_out)
                dist.reduce_scatter_tensor(nccl_reduced, op._sanitized(y))
                _check(f"replay {i}: gather {rows} rows", g_out, nccl_gathered)
                _check(f"replay {i}: reduce-scatter {rows} rows", r_out, nccl_reduced)
        torch.cuda.synchronize(device)
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_round_trip_matches_nccl_ring():
    import torch.multiprocessing as mp

    mp.spawn(_rank_main, args=(_free_port(),), nprocs=WORLD, join=True)
