# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Certification matrix for the FP8 target's `ce_exchange__sp_pair` copy-engine exchange at the
tep2 / tp2 pair under mpirun: eager exchanges and graph replays with new inputs, byte-exact.
Run as `CUDA_VISIBLE_DEVICES=0,1 python _ce_exchange__sp_pair_op_matrix.py --world-size 2`."""

import argparse
import sys

DEADLINE_S = 600
CERTIFIED_WORLDS = (2,)
_WORKER_FLAG = "--rank-worker"
HIDDEN = 2560
CHAIN = 6
_PKG = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp."
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
