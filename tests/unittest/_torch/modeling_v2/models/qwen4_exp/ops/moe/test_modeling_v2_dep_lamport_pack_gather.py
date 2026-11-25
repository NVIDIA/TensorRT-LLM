# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`dep_lamport_pack_gather` on 4 and 8 ranks against `fp4_quantize` of the gathered rows.

One process a GPU, the host-index Lamport workspace of `dep_lamport_round_trip` over torch
symmetric memory. Every rank draws every rank's rows and logits from per-rank seeds, so each
holds the group's reference: the codes and linear block scales bitwise `fp4_quantize` of the
group's rows with -0.0 lanes as +0.0 (the codes as the transport delivers them: a 16-bit half
0x8000, an E2M1 -0 after three +0, as 0x0000), the logits bitwise the group's with -0.0 as +0.0. Eagerly,
inside regions that also hold the workspace's reduce-scatters (the slots rotate under both), and
replayed from one CUDA graph. The 8-rank cells need eight SM103 GPUs in one node.
"""

import os
import socket

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import dep_lamport_pack_gather as op
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import dep_lamport_round_trip as rt

HIDDEN, EXPERTS = 2560, 512
REPLAYS = int(os.environ.get("QWEN4_EXP_LAMPORT_REPLAYS", "100"))
CELLS = op.dep_lamport_pack_gather.CELLS
WORLDS = sorted({c.spec["world"] for c in CELLS})

pytestmark = [pytest.mark.no_xdist]


def _skip(world: int):
    return pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.device_count() < world
        or torch.cuda.get_device_capability() != (10, 3),
        reason=f"the packed gather runs on {world} SM103 GPUs of one NVLink domain here",
    )


def _draw(world: int, rows: int, seed: int, device):
    """Every rank's `(x, logits)`, from per-rank seeds: -0.0 lanes in both."""
    out = []
    for p in range(world):
        gen = torch.Generator(device=device).manual_seed(seed * 64 + p)
        x = torch.randn([rows, HIDDEN], generator=gen, device=device).to(torch.bfloat16)
        lg = torch.randn([rows, EXPERTS], generator=gen, device=device).to(torch.bfloat16)
        x[:, ::97] = -0.0
        lg[:, ::61] = -0.0
        out.append((x, lg))
    return out


def _check(tag, got, xs, lgs, gs):
    want = op.dep_lamport_pack_gather.reference(xs, lgs, gs)
    for name, a, b in zip(("codes", "scales", "logits"), got, want):
        same = a.reshape(-1).view(torch.uint8) == b.reshape(-1).view(torch.uint8)
        assert bool(same.all()), f"{tag}: {name}: {int((~same).sum())} bytes differ"


def _rank_main(rank: int, port: int, world: int) -> None:
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        device = torch.device("cuda", rank)
        steps = rt.LamportSteps()
        trip = rt.build(HIDDEN, rt.MAX_ROWS, rank, world, steps=steps)
        pack = op.PackGather(trip, HIDDEN, EXPERTS)
        gs = torch.tensor([448.0 * 6.0 / 3.0], dtype=torch.float32, device=device)
        for spec in (c.spec for c in CELLS if c.spec["world"] == world):
            rows = spec["rows"]
            for k in range(4):
                drawn = _draw(world, rows, 7 * rows + k, device)
                xs, lgs = [d[0] for d in drawn], [d[1] for d in drawn]
                y = torch.randn([world * rows, HIDDEN], device=device).to(torch.bfloat16)
                with steps.region():
                    got = op.dep_lamport_pack_gather(xs[rank], lgs[rank], gs, pack)
                    # k reduce-scatters after it in the same region: every residue of the base.
                    for _ in range(k):
                        rt.dep_lamport_round_trip(y, trip, True)
                _check(f"{world} ranks, {rows} rows, round {k}", got, xs, lgs, gs)
            # Replayed from one graph.
            sx = torch.zeros([rows, HIDDEN], dtype=torch.bfloat16, device=device)
            slg = torch.zeros([rows, EXPERTS], dtype=torch.bfloat16, device=device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream), steps.region():
                    out = op.dep_lamport_pack_gather(sx, slg, gs, pack)
            torch.cuda.current_stream().wait_stream(stream)
            for i in range(REPLAYS):
                drawn = _draw(world, rows, 1000 + i, device)
                sx.copy_(drawn[rank][0])
                slg.copy_(drawn[rank][1])
                graph.replay()
                xs, lgs = [d[0] for d in drawn], [d[1] for d in drawn]
                _check(f"replay {i}, {rows} rows", out, xs, lgs, gs)
        torch.cuda.synchronize(device)
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("world", [pytest.param(w, marks=_skip(w), id=f"w{w}") for w in WORLDS])
def test_pack_gather_matches_quantized_gather(world):
    import torch.multiprocessing as mp

    mp.spawn(_rank_main, args=(_free_port(), world), nprocs=world, join=True)


@pytest.mark.cpu_only
def test_a_row_is_1232_lanes():
    """Host only: 80 code, 10 scale and 64 logit vectors a row (154 x 8 = 1,232 bf16 lanes)."""
    assert op.vectors_a_row(HIDDEN, EXPERTS) == 154
    assert op.threads(HIDDEN, EXPERTS) == 192


@pytest.mark.cpu_only
def test_transported_codes_zero_only_the_marker_halves():
    """Host only: a code half 0x8000 (little-endian bytes 0x00, 0x80) becomes 0x0000; a -0 nibble
    anywhere else, and every other byte, is unchanged (the full3 dep8 build check declined 15 of 27
    row counts on raw `fp4_quantize` codes, each with one to three such halves)."""
    codes = torch.tensor(
        [
            [0x00, 0x80, 0x80, 0x00, 0x08, 0x00, 0x00, 0x88],
            [0x00, 0x80, 0x01, 0x80, 0x00, 0x80, 0xFF, 0x7F],
        ],
        dtype=torch.uint8,
    )
    want = torch.tensor(
        [
            [0x00, 0x00, 0x80, 0x00, 0x08, 0x00, 0x00, 0x88],
            [0x00, 0x00, 0x01, 0x80, 0x00, 0x00, 0xFF, 0x7F],
        ],
        dtype=torch.uint8,
    )
    assert torch.equal(op.transported_codes(codes), want)
    assert torch.equal(op.transported_codes(codes[:, :4]), want[:, :4])
