# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Certification matrix for the FP8 target's `ce_exchange__sp_pair` copy-engine exchange at the
tep2 / tp2 pair under mpirun: setup failing on one rank yields None on both, then eager exchanges
and graph replays with new inputs, byte-exact, then the gather (`begin_gather` / `put_gather` /
`end_gather`, the MoE block's codes and routing rows) eagerly and replayed from a graph between
plain exchanges: both ranks' rows in rank order.
Run as `CUDA_VISIBLE_DEVICES=0,1 python _ce_exchange__sp_pair_op_matrix.py --world-size 2`."""

import argparse
import sys

DEADLINE_S = 600
CERTIFIED_WORLDS = (2,)
_WORKER_FLAG = "--rank-worker"
HIDDEN = 2560
CHAIN = 6
_PKG = (
    "tensorrt_llm._torch.modeling_v2.models.qwen4_exp."
    "qwen3_8_flash_next_fp8__sm_103__unified.ops.comm."
)


def _worker() -> None:
    import torch
    from mpi4py import MPI

    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    assert world == 2
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")
    ce_mod = __import__(_PKG + "ce_exchange__sp_pair", fromlist=["x"])
    mapping = Mapping(world_size=2, rank=rank, gpus_per_node=2, tp_size=2)
    peer = 1 - rank

    def fill(t, r, it):
        g = torch.Generator(device=dev).manual_seed(1000 * it + r)
        if t.dtype == torch.int32:
            t.copy_(
                torch.randint(
                    -(2**31), 2**31 - 1, t.shape, generator=g, device=dev, dtype=torch.int32
                )
            )
        else:
            t.copy_(torch.randn(t.shape, generator=g, device=dev).to(t.dtype))

    # check_failed_setup: one rank's allocation, then one rank's open of the peer's handle,
    # fails; both ranks get None (no hang), and each frees what it allocated.
    from cuda.bindings import runtime

    def fail(*_):
        return (runtime.cudaError_t.cudaErrorMemoryAllocation, 0)

    for name, failing in (("cudaMalloc", 1), ("cudaIpcOpenMemHandle", 0)):
        torch.cuda.synchronize()
        free_before = torch.cuda.mem_get_info()[0]
        real = getattr(runtime, name)
        if rank == failing:
            setattr(runtime, name, fail)
        try:
            got = ce_mod.CopyEngineExchange.create(mapping, 512 << 20)
        finally:
            setattr(runtime, name, real)
        assert got is None, f"rank {rank}: built an exchange although {name} failed on {failing}"
        torch.cuda.synchronize()
        leaked = free_before - torch.cuda.mem_get_info()[0]
        assert leaked < 64 << 20, f"rank {rank}: {leaked} bytes kept after {name} failed"
        comm.Barrier()

    # check_create: the buffer holds the largest cell, 8192 int32 rows of 2051.
    capacity = 72 << 20
    ce = ce_mod.CopyEngineExchange.create(mapping, capacity)
    assert ce is not None, f"rank {rank}: the exchange was not built"

    # check_eager: every certified cell, three rounds interleaved.
    cells = [c.spec for c in ce_mod.ce_exchange.CELLS if c.spec["world"] == world]
    assert cells, f"no cell is certified at world size {world}"
    for it in range(3 * len(cells)):
        spec = cells[it % len(cells)]
        x = torch.empty(spec["rows"], spec["width"], dtype=spec["dtype"], device=dev)
        fill(x, rank, it)
        got = ce_mod.ce_exchange(ce, x).clone()
        want = torch.empty_like(x)
        fill(want, peer, it)
        torch.cuda.synchronize()
        assert torch.equal(got, want), f"rank {rank}: eager exchange {it} ({spec}) differs"
    comm.Barrier()

    # check_graph
    xs = [torch.empty(4096, HIDDEN, dtype=torch.bfloat16, device=dev) for _ in range(CHAIN)]
    outs = [torch.empty_like(x) for x in xs]

    def body():
        for k in range(CHAIN):
            outs[k].copy_(ce.exchange(xs[k]))

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        body()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    for rep in range(30):
        for k in range(CHAIN):
            fill(xs[k], rank, 100 + rep * CHAIN + k)
        graph.replay()
        torch.cuda.synchronize()
        for k in range(CHAIN):
            want = torch.empty_like(xs[k])
            fill(want, peer, 100 + rep * CHAIN + k)
            assert torch.equal(outs[k], want), f"rank {rank}: replay {rep} exchange {k} differs"
    comm.Barrier()

    # check_gather: `begin_gather` / `put_gather` / `end_gather` as `PrefillTarget._moe_ce_gather`
    # calls them (e4m3 codes as bf16 pairs, then the 70-wide routing rows), between plain
    # exchanges: each receive buffer holds both ranks' rows in rank order, an all-gather's bytes.
    def gather_ce(mine):
        # mine: this rank's tensors, each put at its region (rows in rank order) after the last
        offsets, at = [], 0
        for t in mine:
            nbytes = t.numel() * t.element_size()
            offsets.append(at)
            at += -(-(2 * nbytes) // 256) * 256
        ce.begin_gather()
        for t, off in zip(mine, offsets):
            ce.put_gather(t.view(torch.uint8), off + rank * t.numel() * t.element_size())
        ce.end_gather()
        views = []
        for t, off in zip(mine, offsets):
            nbytes = t.numel() * t.element_size()
            v = ce.received(off, 2 * nbytes).view(t.dtype).view(2 * t.shape[0], *t.shape[1:])
            v[rank * t.shape[0] : (rank + 1) * t.shape[0]].copy_(t)
            views.append(v)
        return views

    def gathered(t, it):
        # both ranks' `fill(t, r, it)` in rank order
        whole = torch.empty(2 * t.shape[0], *t.shape[1:], dtype=t.dtype, device=dev)
        for r in range(2):
            fill(whole[r * t.shape[0] : (r + 1) * t.shape[0]], r, it)
        return whole

    for it in range(8):
        rows = (4096, 77, 1024, 2048)[it % 4]
        a = torch.empty(rows, HIDDEN // 2, dtype=torch.bfloat16, device=dev)
        b = torch.empty(rows, 70, dtype=torch.bfloat16, device=dev)
        fill(a, rank, 500 + it)
        fill(b, rank, 600 + it)
        got = [v.clone() for v in gather_ce([a, b])]
        want = [gathered(a, 500 + it), gathered(b, 600 + it)]
        torch.cuda.synchronize()
        for g, w in zip(got, want):
            assert torch.equal(g, w), f"rank {rank}: eager gather {it} differs"
        # a plain exchange between gathers
        y = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
        fill(y, rank, 700 + it)
        got = ce.exchange(y).clone()
        want = torch.empty_like(y)
        fill(want, peer, 700 + it)
        torch.cuda.synchronize()
        assert torch.equal(got, want), f"rank {rank}: exchange after gather {it} differs"
    comm.Barrier()
    ga = torch.empty(4096, HIDDEN // 2, dtype=torch.bfloat16, device=dev)
    gb = torch.empty(4096, 70, dtype=torch.bfloat16, device=dev)
    gx = torch.empty(4096, HIDDEN, dtype=torch.bfloat16, device=dev)
    gouts = [
        torch.empty(8192, HIDDEN // 2, dtype=torch.bfloat16, device=dev),
        torch.empty(8192, 70, dtype=torch.bfloat16, device=dev),
        torch.empty_like(gx),
    ]

    def gather_body():
        va, vb = gather_ce([ga, gb])
        gouts[0].copy_(va)
        gouts[1].copy_(vb)
        gouts[2].copy_(ce.exchange(gx))

    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        gather_body()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    gather_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gather_graph):
        gather_body()
    for rep in range(20):
        fill(ga, rank, 800 + rep)
        fill(gb, rank, 900 + rep)
        fill(gx, rank, 1000 + rep)
        gather_graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(gouts[0], gathered(ga, 800 + rep)), f"rank {rank}: graph gather {rep}"
        assert torch.equal(gouts[1], gathered(gb, 900 + rep)), f"rank {rank}: graph gather {rep}"
        want = torch.empty_like(gx)
        fill(want, peer, 1000 + rep)
        assert torch.equal(gouts[2], want), f"rank {rank}: graph exchange after gather {rep}"
    comm.Barrier()

    if rank == 0:
        print("ce_exchange__sp_pair: all checks passed", flush=True)
    MPI.Finalize()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=2)
    a, _ = p.parse_known_args()
    if _WORKER_FLAG in sys.argv:
        _worker()
        return
    assert a.world_size in CERTIFIED_WORLDS, a.world_size
    import _launch  # beside this file

    _launch.mpirun_ranks(__file__, a.world_size, [_WORKER_FLAG], DEADLINE_S)


if __name__ == "__main__":
    main()
