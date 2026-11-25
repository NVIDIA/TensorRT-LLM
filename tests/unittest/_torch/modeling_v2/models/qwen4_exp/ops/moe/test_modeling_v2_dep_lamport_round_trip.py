# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`dep_lamport_round_trip` on 4 and 8 ranks against NCCL's ring collectives, bit for bit.

One process a GPU, an NCCL group with `NCCL_ALGO=Ring` inside one node (the ring then visits the
ranks in rank order, the transport's sum order). For each cell's world size, row count and slot
protocol (device counter, or host index under `LamportSteps`), eagerly and then replayed from one
CUDA graph (the three slots rotating across replays):

* the gather of every rank's rows equals `all_gather_into_tensor` of the rows with -0.0 lanes as
  +0.0 (the transport's marker), and the entry's reference;
* the reduce-scatter of every rank's partials equals `reduce_scatter_tensor` of the same
  sanitized partials (the ring's sum order, bf16 after every add), and the reference.

The host-index cells run each round inside a region with a varying number of launches (one to
four round trips), so the slot base moves by every residue mod 3, and also one launch outside any
region (its own advance). The 8-rank cells run on one node of eight SM103 GPUs. Their workspace,
and once more the 4-rank host-index one (so a 4-GPU tray runs it too), is built the way the NVFP4
target's dep8 builds it over `PeerBuffers`: every rank's mapped buffer as a raw device pointer,
wrapped by `buffers_from_pointers`, then `build_on` (here the pointers are torch symmetric
memory's; the target's are MNNVL memory's).

The inputs carry -0.0 lanes on purpose. Past `MAX_ROWS` rows a rank `takes` refuses the call
(the caller keeps NCCL). `QWEN4_EXP_LAMPORT_REPLAYS` raises the graph replays (default 500 a cell)
for a longer stress run.
"""

import os
import socket

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import dep_lamport_round_trip as op

HIDDEN = 2560
REPLAYS = int(os.environ.get("QWEN4_EXP_LAMPORT_REPLAYS", "500"))
CELLS = op.dep_lamport_round_trip.CELLS
GROUPS = sorted({(c.spec["world"], c.spec["step"]) for c in CELLS})

pytestmark = [pytest.mark.no_xdist]


def _skip(world: int):
    return pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.device_count() < world
        or torch.cuda.get_device_capability() != (10, 3),
        reason=f"the Lamport round trip runs on {world} SM103 GPUs of one NVLink domain here",
    )


def _inputs(gen, rows: int, device) -> torch.Tensor:
    """bf16 `[rows, HIDDEN]` with about 1% -0.0 lanes."""
    x = torch.randn([rows, HIDDEN], generator=gen, device=device).to(torch.bfloat16)
    zero = torch.rand([rows, HIDDEN], generator=gen, device=device) < 0.01
    return torch.where(zero, torch.full_like(x, -0.0), x)


def _check(name: str, got: torch.Tensor, want: torch.Tensor) -> None:
    same = got.view(torch.int16) == want.view(torch.int16)
    assert bool(same.all()), f"{name}: {int((~same).sum())} of {same.numel()} lanes differ"


def _nccl(x, y, world: int):
    import torch.distributed as dist

    rows = x.shape[0]
    gathered = torch.empty([world * rows, HIDDEN], dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(gathered, op._sanitized(x))
    reduced = torch.empty([rows, HIDDEN], dtype=y.dtype, device=y.device)
    dist.reduce_scatter_tensor(reduced, op._sanitized(y))
    return gathered, reduced


def _build(rank: int, world: int, step: bool, via_pointers: bool):
    """`build` (torch symmetric memory), or with `via_pointers` the dep8 target's entry: every
    rank's mapped buffer as a device pointer through `buffers_from_pointers`, then `build_on`."""
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    steps = op.LamportSteps() if step else None
    if not via_pointers:
        return op.build(HIDDEN, op.MAX_ROWS, rank, world, steps=steps), steps
    device = torch.device("cuda", rank)
    shape = op.workspace_shape(HIDDEN, op.MAX_ROWS, world)
    buffer = symm_mem.empty(shape, dtype=torch.bfloat16, device=device)
    handle = symm_mem.rendezvous(buffer, group=dist.group.WORLD.group_name)
    mapped = [handle.get_buffer(p, buffer.shape, buffer.dtype) for p in range(world)]
    buffers = op.buffers_from_pointers(
        [b.data_ptr() for b in mapped], HIDDEN, op.MAX_ROWS, world, rank
    )
    assert [b.data_ptr() for b in buffers] == [b.data_ptr() for b in mapped]
    trip = op.build_on(
        HIDDEN, op.MAX_ROWS, rank, world, buffers, steps=steps, keepalive=(handle, buffer, mapped)
    )
    torch.cuda.synchronize(device)
    dist.barrier()
    return trip, steps


def _rank_main(rank: int, port: int, world: int, step: bool, via_pointers: bool) -> None:
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), NCCL_ALGO="Ring")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        device = torch.device("cuda", rank)
        trip, steps = _build(rank, world, step, via_pointers)
        assert (trip.steps is not None) == step
        probe = torch.empty([1, HIDDEN], dtype=torch.bfloat16, device=device)
        assert trip.takes(op.MAX_ROWS, probe) and not trip.takes(op.MAX_ROWS + 1, probe)
        gen = torch.Generator(device=device).manual_seed(20261125 + rank)
        rows_list = sorted(
            c.spec["rows"] for c in CELLS if (c.spec["world"], c.spec["step"]) == (world, step)
        )
        for rows in rows_list:
            # Eager, 1..4 round trips a region (host index) or one at a time.
            for trips in range(1, 5 if step else 2):
                with steps.region() if step else _nothing():
                    rounds = []
                    for _ in range(trips):
                        x, y = _inputs(gen, rows, device), _inputs(gen, world * rows, device)
                        gathered = op.dep_lamport_round_trip(x, trip, False)
                        reduced = op.dep_lamport_round_trip(y, trip, True)
                        rounds.append((x, y, gathered, reduced))
                for k, (x, y, gathered, reduced) in enumerate(rounds):
                    nccl_gathered, nccl_reduced = _nccl(x, y, world)
                    tag = f"{world} ranks, {rows} rows, {trips} trips, trip {k}"
                    _check(f"gather vs NCCL ({tag})", gathered, nccl_gathered)
                    _check(f"reduce-scatter vs NCCL ({tag})", reduced, nccl_reduced)
                    xs = [torch.empty_like(x) for _ in range(world)]
                    ys = [torch.empty_like(y) for _ in range(world)]
                    dist.all_gather(xs, x)
                    dist.all_gather(ys, y)
                    ref = op.dep_lamport_round_trip.reference
                    _check(f"gather vs reference ({tag})", gathered, ref(xs, rank, False))
                    _check(f"reduce-scatter vs reference ({tag})", reduced, ref(ys, rank, True))
            if step:
                # A launch outside any region takes index 0 and its own advance.
                x, y = _inputs(gen, rows, device), _inputs(gen, world * rows, device)
                gathered = op.dep_lamport_round_trip(x, trip, False)
                reduced = op.dep_lamport_round_trip(y, trip, True)
                nccl_gathered, nccl_reduced = _nccl(x, y, world)
                _check(f"gather outside a region ({rows} rows)", gathered, nccl_gathered)
                _check(f"reduce-scatter outside a region ({rows} rows)", reduced, nccl_reduced)

            # Replayed from one graph: fresh inputs copied into its static buffers.
            sx = torch.zeros([rows, HIDDEN], dtype=torch.bfloat16, device=device)
            sy = torch.zeros([world * rows, HIDDEN], dtype=torch.bfloat16, device=device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    with steps.region() if step else _nothing():
                        g_out = op.dep_lamport_round_trip(sx, trip, False)
                        r_out = op.dep_lamport_round_trip(sy, trip, True)
            torch.cuda.current_stream().wait_stream(stream)
            for i in range(REPLAYS):
                x, y = _inputs(gen, rows, device), _inputs(gen, world * rows, device)
                sx.copy_(x)
                sy.copy_(y)
                graph.replay()
                nccl_gathered, nccl_reduced = _nccl(x, y, world)
                _check(f"replay {i}: gather {rows} rows", g_out, nccl_gathered)
                _check(f"replay {i}: reduce-scatter {rows} rows", r_out, nccl_reduced)
        torch.cuda.synchronize(device)
    finally:
        dist.destroy_process_group()


class _nothing:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


#: `(world, step, via_pointers)`: every cell group; the 8-rank ones and one more 4-rank host-index
#: run over `buffers_from_pointers` + `build_on` (the dep8 target's entry).
_BUILDS = [(w, s, w == 8) for w, s in GROUPS] + [(4, True, True)]


@pytest.mark.parametrize(
    "world, step, via_pointers",
    [
        pytest.param(
            w,
            s,
            p,
            marks=_skip(w),
            id=f"w{w}-{'step' if s else 'counter'}{'-pointers' if p and w == 4 else ''}",
        )
        for w, s, p in _BUILDS
    ],
)
def test_round_trip_matches_nccl_ring(world, step, via_pointers):
    import torch.multiprocessing as mp

    mp.spawn(_rank_main, args=(_free_port(), world, step, via_pointers), nprocs=world, join=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device pointer")
def test_buffers_from_pointers_are_bf16_views_of_the_pointed_memory():
    """`buffers_from_pointers` hands `build_on` bf16 `workspace_shape` tensors over the very
    bytes it was pointed at, on any one GPU (the 4- and 8-rank cells above need a tray)."""
    world, rows = 2, 3
    shape = op.workspace_shape(HIDDEN, rows, world)
    owned = [torch.randn(shape, device="cuda").to(torch.bfloat16) for _ in range(world)]
    views = op.buffers_from_pointers(
        [b.data_ptr() for b in owned], HIDDEN, rows, world, torch.cuda.current_device()
    )
    for view, buf in zip(views, owned):
        assert view.dtype is torch.bfloat16 and tuple(view.shape) == shape
        assert view.data_ptr() == buf.data_ptr()
        assert torch.equal(view.view(torch.int16), buf.view(torch.int16))
