# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU checks of the buffered PP transport (``pp_buffered.py``).

Each test spawns ``world_size`` ranks (2 or 4 GPUs of one node), one PP stage per
GPU, and drives ``pp_send_tensors`` / ``pp_recv_tensors`` exactly as a stage
boundary does (an FP32 residual and its BF16 next-layer attention input):

* ``test_staging_ring_delivers_every_message``: stamped payloads of varying rows
  (1 .. 8192 at hidden 8192, i.e. up to 384 MiB) flow through the
  pipeline with uneven per-stage compute between them; every stage checks every
  byte it receives, and the counters show every message took the staging ring.
* ``test_captured_sends_fall_back_to_nccl_and_interleave``: a CUDA graph captures
  one boundary (NCCL on both ends) and is replayed between eager ring messages.
* ``test_default_is_the_stock_communicator``: without the buffered request ``init_pp_comm``
  builds ``PPCommNCCL``.

Run directly for the overlap microbench (a measurement on the production
transport): ``mpirun -n 4 python test_pp_buffered_p2p_gpu.py bench --ring 4``
prints JSON with the per-stage period of the zero-transfer ring, the stock
``PPCommNCCL`` and the staging ring under a calibrated compute block.

The two transport checks also run directly on ``N`` launched ranks, without the
``MPIPoolExecutor`` spawn (for launchers such as ``srun --mpi=pmix`` that cannot
spawn): ``srun -n N python test_pp_buffered_p2p_gpu.py ring`` and ``... capture``
print one JSON line on rank 0 and exit non-zero when a check fails.
"""

import argparse
import json
import pickle
import sys
import time
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN = 8192


def _setup(world_size: int, buffered: bool):
    import tensorrt_llm
    from tensorrt_llm._torch.distributed import communicator
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    mapping = Mapping(
        world_size=world_size, rank=rank, gpus_per_node=world_size, pp_size=world_size
    )
    communicator._pp_comm = None
    communicator.init_pp_comm(mapping, buffered=buffered)
    return mapping, communicator


def _payload(stage: int, message: int, rows: int):
    base = float((stage * 131 + message * 7) % 1000)
    residual = torch.arange(rows * HIDDEN, device="cuda", dtype=torch.float32)
    residual = residual.view(rows, HIDDEN) + base
    attn_input = torch.full((rows, HIDDEN), base / 3.0, device="cuda").to(torch.bfloat16)
    return [residual, attn_input]


def _busy(ms: float, a: torch.Tensor):
    """Roughly ``ms`` of GEMM work on the current stream (uncalibrated)."""
    for _ in range(max(1, int(ms))):
        a = torch.tanh(a @ a)
    return a


def _run(fn, world_size, *args):
    try:
        return fn(world_size, *args)
    except Exception:  # noqa: BLE001 - reported to the parent
        traceback.print_exc()
        raise


def _ring_delivers(world_size: int, messages: int):
    from tensorrt_llm._torch.distributed.pp_buffered import BufferedPPComm

    mapping, communicator = _setup(world_size, buffered=True)
    comm = communicator._pp_comm
    assert isinstance(comm, BufferedPPComm), type(comm)
    rows_cycle = [1, 7, 4096, 8192, 300, 8192, 2048]
    stage = mapping.pp_rank
    work = torch.randn(1024, 1024, device="cuda") / 32
    bad = []
    for message in range(messages):
        rows = rows_cycle[message % len(rows_cycle)]
        if stage > 0:
            inbox = [
                torch.empty((rows, HIDDEN), device="cuda", dtype=torch.float32),
                torch.empty((rows, HIDDEN), device="cuda", dtype=torch.bfloat16),
            ]
            communicator.pp_recv_tensors(inbox)
            want = _payload(stage - 1, message, rows)
            if not all(torch.equal(a, b) for a, b in zip(inbox, want)):
                bad.append(message)
        # Uneven compute: stages and messages differ, so the ring fills and drains.
        work = _busy(1 + ((message * 3 + stage * 5) % 7), work)
        if stage < world_size - 1:
            communicator.pp_send_tensors(_payload(stage, message, rows))
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    return {
        "rank": mapping.rank,
        "bad": bad,
        "send_seq": comm.send_seq,
        "recv_seq": comm.recv_seq,
        "fallback": comm.fallback_sends + comm.fallback_recvs,
    }


def _capture_interleave(world_size: int):
    mapping, communicator = _setup(world_size, buffered=True)
    comm = communicator._pp_comm
    stage, rows = mapping.pp_rank, 64
    static = _payload(stage, 999, rows) if stage < world_size - 1 else None
    inbox = [
        torch.empty((rows, HIDDEN), device="cuda", dtype=torch.float32),
        torch.empty((rows, HIDDEN), device="cuda", dtype=torch.bfloat16),
    ]
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    MPI.COMM_WORLD.Barrier()
    with torch.cuda.stream(side), torch.cuda.graph(graph, stream=side):
        if stage > 0:
            communicator.pp_recv_tensors(inbox)
        if stage < world_size - 1:
            communicator.pp_send_tensors(static)
    torch.cuda.synchronize()
    bad = []
    for message in range(6):
        # Eager message on the ring, then one graph replay over NCCL.
        if stage > 0:
            eager_in = [torch.empty_like(t) for t in inbox]
            communicator.pp_recv_tensors(eager_in)
            want = _payload(stage - 1, message, rows)
            if not all(torch.equal(a, b) for a, b in zip(eager_in, want)):
                bad.append(("eager", message))
        if stage < world_size - 1:
            communicator.pp_send_tensors(_payload(stage, message, rows))
        graph.replay()
        torch.cuda.synchronize()
        if stage > 0 and not all(
            torch.equal(a, b) for a, b in zip(inbox, _payload(stage - 1, 999, rows))
        ):
            bad.append(("graph", message))
    MPI.COMM_WORLD.Barrier()
    return {
        "rank": mapping.rank,
        "bad": bad,
        "fallback": comm.fallback_sends + comm.fallback_recvs,
        "ring": comm.send_seq + comm.recv_seq,
    }


def _default_is_stock(world_size: int):
    _, communicator = _setup(world_size, buffered=False)
    return type(communicator._pp_comm).__name__


def _spawn(world_size, fn, *args):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    with MPIPoolExecutor(max_workers=world_size) as executor:
        return list(executor.map(_run, *zip(*[(fn, world_size, *args)] * world_size)))


@pytest.mark.parametrize("world_size", [2, 4])
def test_staging_ring_delivers_every_message(world_size):
    results = _spawn(world_size, _ring_delivers, 40)
    for r in results:
        assert r["bad"] == [], r
        assert r["fallback"] == 0, r
        stage = r["rank"]
        assert r["send_seq"] == (40 if stage < world_size - 1 else 0)
        assert r["recv_seq"] == (40 if stage > 0 else 0)


@pytest.mark.parametrize("world_size", [2, 4])
def test_captured_sends_fall_back_to_nccl_and_interleave(world_size):
    results = _spawn(world_size, _capture_interleave)
    for r in results:
        assert r["bad"] == [], r
        # Capture went through NCCL on every edge end; the eager messages took the ring.
        stage = r["rank"]
        ends = (stage > 0) + (stage < world_size - 1)
        assert r["fallback"] == ends, r
        assert r["ring"] == 6 * ends, r


def test_default_is_the_stock_communicator():
    assert set(_spawn(2, _default_is_stock)) == {"PPCommNCCL"}


# --------------------------------------------------------------------------- bench


def _bench(args) -> dict:
    """Per-stage period over ``micro_batches`` of a ``compute_ms`` block per stage.

    Variants: ``zero`` (the ring of hand-offs without bytes: a 4-byte message on
    the staging ring), ``stock`` (``PPCommNCCL``), ``buffered`` (staging ring).
    Closed loop: stage 0 starts micro-batch j only after the last stage finished
    j - ring (a host message from the last rank), like the PP executor's relay.
    """
    import tensorrt_llm
    from tensorrt_llm._torch.distributed.communicator import PPCommNCCL
    from tensorrt_llm._torch.distributed.pp_buffered import BufferedPPComm
    from tensorrt_llm.mapping import Mapping

    world = MPI.COMM_WORLD
    rank, world_size = world.Get_rank(), world.Get_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    mapping = Mapping(
        world_size=world_size, rank=rank, gpus_per_node=world_size, pp_size=world_size
    )
    stock = PPCommNCCL(mapping)
    buffered = BufferedPPComm(mapping, PPCommNCCL(mapping))
    a = torch.randn(args.gemm, args.gemm, device="cuda", dtype=torch.bfloat16)

    def block():
        nonlocal a
        for _ in range(args.gemm_iters):
            a = (a @ a) * (1.0 / args.gemm)

    # Calibrate: GEMMs per block for compute_ms.
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(5):
        block()
    torch.cuda.synchronize()
    per_block_ms = (time.perf_counter() - start) * 200.0
    args.gemm_iters = max(1, int(args.gemm_iters * args.compute_ms / per_block_ms))
    tensorrt_llm.logger.info(f"rank {rank}: {args.gemm_iters} GEMMs per block")

    results = {}
    for variant in args.variants.split(","):
        rows = 1 if variant == "zero" else args.rows
        send = [
            torch.ones((rows, HIDDEN), device="cuda", dtype=torch.float32),
            torch.ones((rows, HIDDEN), device="cuda", dtype=torch.bfloat16),
        ]
        recv = [torch.empty_like(t) for t in send]
        comm = stock if variant == "stock" else buffered
        world.Barrier()
        torch.cuda.synchronize()
        starts, ends = [], []
        for j in range(args.micro_batches):
            if rank == 0 and j >= args.ring:
                world.recv(source=world_size - 1, tag=7)
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            if rank > 0:
                if variant == "stock":
                    for t in recv:
                        comm.recv(t)
                else:
                    comm.recv_tensors(recv)
            block()
            if rank < world_size - 1:
                if variant == "stock":
                    for t in send:
                        comm.send(t)
                else:
                    comm.send_tensors(send)
            end.record()
            starts.append(begin)
            ends.append(end)
            if rank == world_size - 1:
                end.synchronize()
                world.send(j, dest=0, tag=7)
        torch.cuda.synchronize()
        # Drain the closed-loop tokens the last rank sent past the end.
        if rank == 0:
            for _ in range(min(args.ring, args.micro_batches)):
                world.recv(source=world_size - 1, tag=7)
        world.Barrier()
        span = starts[0].elapsed_time(ends[-1])
        warm = args.warmup
        period = starts[warm].elapsed_time(ends[-1]) / (args.micro_batches - warm)
        results[variant] = {"span_ms": span, "period_ms": period}
    gathered = world.gather(results, root=0)
    if rank == 0:
        summary = {
            "ring": args.ring,
            "rows": args.rows,
            "compute_ms": args.compute_ms,
            "per_rank": gathered,
        }
        for variant in args.variants.split(","):
            summary[f"{variant}_period_max_ms"] = max(r[variant]["period_ms"] for r in gathered)
        if "zero_period_max_ms" in summary:
            for variant in ("stock", "buffered"):
                key = f"{variant}_period_max_ms"
                if key in summary:
                    summary[f"{variant}_exposed_ms"] = summary[key] - summary["zero_period_max_ms"]
        print(json.dumps(summary, indent=1))
        return summary
    return {}


def _check_direct(mode: str) -> int:
    """Run one transport check on the launched ranks; rank 0 asserts and reports."""
    world = MPI.COMM_WORLD
    world_size = world.Get_size()
    if mode == "ring":
        result = _ring_delivers(world_size, 40)
    else:
        result = _capture_interleave(world_size)
    gathered = world.gather(result, root=0)
    if world.Get_rank() != 0:
        return 0
    failures = []
    for r in gathered:
        stage = r["rank"]
        ends = (stage > 0) + (stage < world_size - 1)
        if r["bad"]:
            failures.append(r)
        elif mode == "ring" and (
            r["fallback"] != 0
            or r["send_seq"] != (40 if stage < world_size - 1 else 0)
            or r["recv_seq"] != (40 if stage > 0 else 0)
        ):
            failures.append(r)
        elif mode == "capture" and (r["fallback"] != ends or r["ring"] != 6 * ends):
            failures.append(r)
    print(
        json.dumps(
            {"mode": mode, "world_size": world_size, "ok": not failures, "per_rank": gathered}
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("ring", "capture"):
        sys.exit(_check_direct(sys.argv[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["bench"])
    parser.add_argument("--ring", type=int, default=4)
    parser.add_argument("--micro-batches", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--compute-ms", type=float, default=85.0)
    parser.add_argument("--gemm", type=int, default=8192)
    parser.add_argument("--gemm-iters", type=int, default=8)
    parser.add_argument("--variants", default="zero,stock,buffered")
    _bench(parser.parse_args())
