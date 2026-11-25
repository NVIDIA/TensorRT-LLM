# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`dep_lamport_pack_gather` on 4 and 8 ranks against `fp4_quantize` of the gathered rows.

One process a GPU, the host-index Lamport workspace of `dep_lamport_round_trip` over torch
symmetric memory. Every rank draws every rank's rows and logits from per-rank seeds, so each
holds the group's reference: the codes and linear block scales bitwise `fp4_quantize` of the
group's rows with -0.0 lanes as +0.0 (the codes as the transport delivers them: a 16-bit half
0x8000, an E2M1 -0 after three +0, as 0x0000), the logits bitwise the group's with -0.0 as +0.0,
or with the owner top-k (`ownk` cells) the routing rows bitwise `routed_rows` of
`renorm_moe_routing_op` over those logits. Eagerly, inside regions that also hold the workspace's
reduce-scatters (the slots rotate under both), and replayed from one CUDA graph. The 8-rank cells
need eight SM103 GPUs in one node. One GPU: the owner's routing statements (`route_one_token` +
`_routing_words`) in a one-warp-a-row kernel over logits of every kind, against the same
reference.
"""

import os
import socket

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import dep_lamport_pack_gather as op
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import dep_lamport_round_trip as rt
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import (
    moe_route_front__decode as front,
)

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
        # Coarse logits on every other row (ties inside the top-10 and at its edge).
        lg[1::2] = (lg[1::2] * 4).round().div(4)
        lg[:, ::61] = -0.0
        out.append((x, lg))
    return out


def _check(tag, got, xs, lgs, gs, ownk=False):
    want = op.dep_lamport_pack_gather.reference(xs, lgs, gs, ownk)
    for name, a, b in zip(("codes", "scales", "routing" if ownk else "logits"), got, want):
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
        packs = {ownk: op.PackGather(trip, HIDDEN, EXPERTS, ownk=ownk) for ownk in (False, True)}
        gs = torch.tensor([448.0 * 6.0 / 3.0], dtype=torch.float32, device=device)
        for spec in (c.spec for c in CELLS if c.spec["world"] == world):
            rows, ownk = spec["rows"], spec["ownk"]
            pack = packs[ownk]
            for k in range(4):
                drawn = _draw(world, rows, 7 * rows + k, device)
                xs, lgs = [d[0] for d in drawn], [d[1] for d in drawn]
                y = torch.randn([world * rows, HIDDEN], device=device).to(torch.bfloat16)
                with steps.region():
                    got = op.dep_lamport_pack_gather(xs[rank], lgs[rank], gs, pack)
                    # k reduce-scatters after it in the same region: every residue of the base.
                    for _ in range(k):
                        rt.dep_lamport_round_trip(y, trip, True)
                _check(
                    f"{world} ranks, {rows} rows, ownk {ownk}, round {k}", got, xs, lgs, gs, ownk
                )
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
                _check(f"replay {i}, {rows} rows, ownk {ownk}", out, xs, lgs, gs, ownk)
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
    """Host only: 80 code, 10 scale and 64 logit vectors a row (154 x 8 = 1,232 bf16 lanes); with
    the owner top-k 3 routing vectors instead of the logits (93) and one routing warp (160
    threads)."""
    assert op.vectors_a_row(HIDDEN, EXPERTS) == 154
    assert op.threads(HIDDEN, EXPERTS) == 192
    assert op.vectors_a_row(HIDDEN, EXPERTS, ownk=True) == 93
    assert op.threads(HIDDEN, EXPERTS, ownk=True) == 160
    assert 3 * 8 == front.IDS_ROW_SLOTS == op._mrf.IDS_ROW_SLOTS


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


def _owner_rows_fn(rows: int):
    """A one-warp-a-row kernel running the owner top-k gather's routing statements: the row's
    logits sanitized as the gather loads them, `route_one_token`, `_routing_words`, lanes 0..2
    storing the 3 vectors."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int64
    from cutlass.cute.runtime import make_fake_stream

    from tensorrt_llm._torch.kernels.blackwell import moe_decode_front as mdf
    from tensorrt_llm._torch.kernels.blackwell import moe_route_front as mrf

    @cute.kernel
    def kernel(lg_ptr: Int64, out_ptr: Int64):
        lane, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        src = lg_ptr + (Int64(row) * Int64(EXPERTS) + Int64(lane) * Int64(16)) * Int64(2)
        m0, m1, m2, m3 = mdf._ld_v4(src)
        m4, m5, m6, m7 = mdf._ld_v4(src + Int64(16))
        packed = tuple(op._clean(m) for m in (m0, m1, m2, m3, m4, m5, m6, m7))
        sel, w = mrf.route_one_token(packed, lane, 10)
        c0, c1, c2, c3 = op._routing_words(sel, w, lane)
        if lane < 3:
            dst = out_ptr + Int64(row) * Int64(2 * front.IDS_ROW_SLOTS) + Int64(lane) * Int64(16)
            mdf._st_v4(dst, c0, c1, c2, c3)

    @cute.jit
    def launch(lg_ptr: Int64, out_ptr: Int64, stream: cuda.CUstream, ROWS: cutlass.Constexpr[int]):
        kernel(lg_ptr, out_ptr).launch(grid=(ROWS, 1, 1), block=(32, 1, 1), stream=stream)

    z = Int64(0)
    return cute.compile(launch, z, z, make_fake_stream(), rows, options="--enable-tvm-ffi")


def _logits_of_every_kind(rows: int, seed: int, device) -> torch.Tensor:
    """Plain rows, coarse rows (bf16 ties), rows whose top-10 sits in one lane's 16 experts (the
    full-sort fallback), +-0.0 rows (ties of sanitized -0.0 lanes) and rows with -0.0 lanes."""
    gen = torch.Generator(device=device).manual_seed(seed)
    lg = torch.randn([rows, EXPERTS], generator=gen, device=device)
    kind = torch.arange(rows, device=device) % 5
    lane = (torch.arange(EXPERTS, device=device) // 16)[None, :]
    owner = (torch.arange(rows, device=device) % 32)[:, None]
    lg = torch.where((kind == 1)[:, None], (lg * 2).round() / 2, lg)
    lg = torch.where((kind == 2)[:, None] & (lane == owner), lg + 8.0, lg)
    zeros = torch.zeros_like(lg)
    zeros[:, ::3] = -0.0
    zeros[:, ::37] = 1.0
    lg = torch.where((kind == 3)[:, None], zeros, lg)
    lg = lg.to(torch.bfloat16)
    lg[kind == 4, ::7] = -0.0
    return lg.contiguous()


@pytest.mark.parametrize(
    "rows", [pytest.param(r, marks=_skip(1), id=f"rows{r}") for r in (5, 96, 112)]
)
def test_owner_routing_matches_the_route_fronts_routing(rows):
    # An explicit index: the raw-stream query refuses `torch.device("cuda").index` (None).
    device = torch.device("cuda", torch.cuda.current_device())
    fn = _owner_rows_fn(rows)
    for seed in range(3):
        lg = _logits_of_every_kind(rows, 31 * rows + seed, device)
        out = torch.full([rows, front.IDS_ROW_SLOTS], -1, dtype=torch.int16, device=device)
        fn(lg.data_ptr(), out.data_ptr(), torch._C._cuda_getCurrentRawStream(device.index))
        torch.cuda.synchronize()
        ids, wts = front.renorm_moe_routing_op(rt._sanitized(lg), front.TOP_K, torch.bfloat16)
        want = front.routed_rows(ids, wts)
        same = out == want
        assert bool(same.all()), f"seed {seed}: {int((~same).sum())} slots differ"
