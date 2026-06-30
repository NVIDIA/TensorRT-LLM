"""Multi-rank tests for the async Ulysses A2A op.

Exercises the production ``ulysses_a2a_async_prepare`` / ``ulysses_a2a_async_push``
/ ``ulysses_a2a_async_barrier`` chain end-to-end across multiple ranks. Two test
surfaces:

1. ``test_slot_ring_wraparound`` — eager mode (cudaMemcpyBatchAsync path).
   Loops the pair more than ``kNumSlots`` times so the slot ring wraps multiple
   times, asserts byte-exact match vs ``all_to_all_4d`` on every iteration.
   Catches off-by-one slot-reuse bugs.

2. ``test_capture_smoke`` — under-capture mode (per-peer cudaMemcpyAsync path).
   Warms up the slot out-of-capture, then captures a ``torch.cuda.CUDAGraph``
   containing one prepare+async pair, replays it K times with fresh inputs.
   Smoke-tests the cuda_graph path used by the e2e production benchmark.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_ulysses_async.py -v
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    import sys
    from pathlib import Path

    from tensorrt_llm._torch.distributed import all_to_all_4d

    # Spawn distributed workers via a helper that retries with a fresh master
    # port when the c10d rendezvous TCPStore loses the bind race (EADDRINUSE).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _visual_gen_dist_utils import spawn_with_retry

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


# Loop count must comfortably exceed kNumSlots so the ring wraps at least
# twice. kNumSlots is 3 today; 8 iterations = ~2.67 full rotations.
NUM_ITERS = 8


def _init_dist(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _slot_ring_logic(rank: int, world_size: int):
    """Loop the async A2A prepare/async pair NUM_ITERS times; assert byte-exact
    match against all_to_all_4d on every iteration.

    Stream pattern mirrors production (parallel.py:_issue_async): Phase 1 on
    the default (compute) stream, Phase 2 on a dedicated side stream gated by
    an event recorded after Phase 1.
    """
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(1234 + rank)

    # Production-scale shapes (LTX-2 ws=4: H=32, D=128, S_local=720 region).
    B, S_local, H, D = 2, 128, world_size * 8, 128
    pg = dist.group.WORLD
    pg_boxed = pg.boxed()
    side_stream = torch.cuda.Stream(device=device)

    for it in range(NUM_ITERS):
        # Fresh input each iteration (different bytes) so a stale slot would
        # produce a stale value and the byte-exact compare would catch it.
        torch.manual_seed(1000 * it + rank)
        x = torch.randn(B, S_local, H, D, dtype=torch.bfloat16, device=device)

        # Phase 1 on default stream.
        recv_5d, send_h = torch.ops.trtllm.ulysses_a2a_async_prepare(x, pg_boxed)
        ev = torch.cuda.Event()
        ev.record()
        # Phase 2 on side stream.
        with torch.cuda.stream(side_stream):
            ev.wait()
            torch.ops.trtllm.ulysses_a2a_async_push(send_h, pg_boxed)
            torch.ops.trtllm.ulysses_a2a_async_barrier(pg_boxed)
        ev_done = torch.cuda.Event()
        with torch.cuda.stream(side_stream):
            ev_done.record()
        torch.cuda.current_stream().wait_event(ev_done)
        torch.cuda.synchronize()

        # Async op produces [P, B, S_local, H/P, D]; permute to [B, P*S_local, H/P, D]
        # to match the all_to_all_4d output layout.
        async_out = (
            recv_5d.permute(1, 0, 2, 3, 4)
            .contiguous()
            .view(B, world_size * S_local, H // world_size, D)
        )

        ref = all_to_all_4d(x, scatter_dim=2, gather_dim=1, process_group=pg)

        assert torch.equal(async_out, ref), (
            f"rank {rank} iter {it}: async A2A ≠ all_to_all_4d reference "
            f"(slot ring wrap = {it % 3})"
        )


def _capture_smoke_logic(rank: int, world_size: int):
    """Smoke-test the under-capture branch: warm up slots out-of-capture, then
    capture a torch.cuda.CUDAGraph containing one prepare+async pair and replay
    K times with fresh inputs. Exercises the per-peer cudaMemcpyAsync loop
    (production cuda_graph path) instead of the eager cudaMemcpyBatchAsync.
    """
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(1234 + rank)

    B, S_local, H, D = 2, 128, world_size * 8, 128
    P = world_size
    pg = dist.group.WORLD
    pg_boxed = pg.boxed()
    side_stream = torch.cuda.Stream(device=device)

    # Static buffers (graph captures pointers, not values).
    x_static = torch.randn(B, S_local, H, D, dtype=torch.bfloat16, device=device)
    out_static = torch.empty(B, P * S_local, H // P, D, dtype=torch.bfloat16, device=device)

    # Warmup: must allocate ALL kNumSlots ring entries out-of-capture (allocation
    # is not capture-safe; the cudaStreamIsCapturing guard in getOrAllocSlot
    # enforces this). One warmup call only allocates 1 slot — the captured
    # _prepare advances mNextIdx and would hit an unallocated slot. So warm up
    # >= kNumSlots times. Mirror production's stream pattern (Phase 2 on side
    # stream) so cudaMemcpyBatchAsync sees the steady-state stream context.
    K_NUM_SLOTS = 3  # mirrors AsyncUlyssesOp::kNumSlots
    for _ in range(K_NUM_SLOTS):
        recv_w, sh_w = torch.ops.trtllm.ulysses_a2a_async_prepare(x_static, pg_boxed)
        ev_w = torch.cuda.Event()
        ev_w.record()
        with torch.cuda.stream(side_stream):
            ev_w.wait()
            torch.ops.trtllm.ulysses_a2a_async_push(sh_w, pg_boxed)
            torch.ops.trtllm.ulysses_a2a_async_barrier(pg_boxed)
        ev_w_done = torch.cuda.Event()
        with torch.cuda.stream(side_stream):
            ev_w_done.record()
        torch.cuda.current_stream().wait_event(ev_w_done)
        torch.cuda.synchronize()
        del recv_w, sh_w

    # Capture.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        recv_5d, send_h = torch.ops.trtllm.ulysses_a2a_async_prepare(x_static, pg_boxed)
        ev = torch.cuda.Event()
        ev.record()
        with torch.cuda.stream(side_stream):
            ev.wait()
            torch.ops.trtllm.ulysses_a2a_async_push(send_h, pg_boxed)
            torch.ops.trtllm.ulysses_a2a_async_barrier(pg_boxed)
        ev_done = torch.cuda.Event()
        with torch.cuda.stream(side_stream):
            ev_done.record()
        torch.cuda.current_stream().wait_event(ev_done)
        captured_out = recv_5d.permute(1, 0, 2, 3, 4).contiguous().view(B, P * S_local, H // P, D)
        out_static.copy_(captured_out)

    # Replay K times. K > kNumSlots*2 to exercise wrap-around under capture.
    K = 8
    for it in range(K):
        torch.manual_seed(2000 * it + rank)
        new_x = torch.randn(B, S_local, H, D, dtype=torch.bfloat16, device=device)
        x_static.copy_(new_x)
        g.replay()
        torch.cuda.synchronize()
        ref = all_to_all_4d(new_x, scatter_dim=2, gather_dim=1, process_group=pg)
        assert torch.equal(out_static, ref), (
            f"rank {rank} capture replay {it}: async A2A ≠ all_to_all_4d reference"
        )


def _multi_pg_logic(rank: int, world_size: int):
    """Two distinct PGs spanning the same ranks but with different group_names;
    alternate calls between them; verify each gets its own slot ring and
    byte-exact output vs all_to_all_4d on its own group.
    """
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(4242 + rank)

    B, S_local, H, D = 2, 64, world_size * 8, 128
    # Two PGs, same membership (all ranks), but new_group assigns distinct
    # group_names — so getOrCreateOp must cache one op per PG-name and
    # set_group_info must register both.
    pg_a = dist.new_group(ranks=list(range(world_size)))
    pg_b = dist.new_group(ranks=list(range(world_size)))
    assert pg_a.group_name != pg_b.group_name, "new_group should yield distinct names"
    pg_a_boxed = pg_a.boxed()
    pg_b_boxed = pg_b.boxed()
    side_stream = torch.cuda.Stream(device=device)

    def _issue(pg_boxed, pg_obj, x):
        recv_5d, send_h = torch.ops.trtllm.ulysses_a2a_async_prepare(x, pg_boxed)
        ev = torch.cuda.Event()
        ev.record()
        with torch.cuda.stream(side_stream):
            ev.wait()
            torch.ops.trtllm.ulysses_a2a_async_push(send_h, pg_boxed)
            torch.ops.trtllm.ulysses_a2a_async_barrier(pg_boxed)
        ev_done = torch.cuda.Event()
        with torch.cuda.stream(side_stream):
            ev_done.record()
        torch.cuda.current_stream().wait_event(ev_done)
        torch.cuda.synchronize()
        async_out = (
            recv_5d.permute(1, 0, 2, 3, 4)
            .contiguous()
            .view(B, world_size * S_local, H // world_size, D)
        )
        ref = all_to_all_4d(x, scatter_dim=2, gather_dim=1, process_group=pg_obj)
        return async_out, ref

    # Alternate between the two PGs for 2*kNumSlots iterations to exercise
    # both slot rings wrapping.
    for it in range(NUM_ITERS):
        torch.manual_seed(3000 * it + rank)
        x = torch.randn(B, S_local, H, D, dtype=torch.bfloat16, device=device)
        pg_boxed, pg_obj, tag = (pg_a_boxed, pg_a, "A") if it % 2 == 0 else (pg_b_boxed, pg_b, "B")
        async_out, ref = _issue(pg_boxed, pg_obj, x)
        assert torch.equal(async_out, ref), (
            f"rank {rank} iter {it} PG={tag}: async A2A ≠ all_to_all_4d reference"
        )


def _worker(rank, world_size, port):
    try:
        _init_dist(rank, world_size, port)
        _slot_ring_logic(rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _worker_capture(rank, world_size, port):
    try:
        _init_dist(rank, world_size, port)
        _capture_smoke_logic(rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _worker_multi_pg(rank, world_size, port):
    try:
        _init_dist(rank, world_size, port)
        _multi_pg_logic(rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(world_size: int, test_fn: Callable):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")
    spawn_with_retry(
        lambda port: mp.spawn(
            test_fn,
            args=(world_size, port),
            nprocs=world_size,
            join=True,
        )
    )


def test_slot_ring_wraparound():
    """Loop _prepare/_async ≥ 2*kNumSlots iterations on ws=2; assert byte-exact
    match against all_to_all_4d on each iteration (eager mode).
    """
    _run(2, _worker)


def test_capture_smoke():
    """Capture a CUDAGraph containing one _prepare/_async pair on ws=2; replay
    K iterations with fresh inputs; assert byte-exact match against
    all_to_all_4d (per-peer cudaMemcpyAsync path under capture).
    """
    _run(2, _worker_capture)


def test_multi_pg():
    """Two distinct ProcessGroups on ws=2; alternate _prepare/_async calls
    between them; assert byte-exact match against each group's all_to_all_4d.

    Exercises PG-name caching in ``getOrCreateOp`` and ``set_group_info``
    re-registration across multiple groups — each PG has its own group_name,
    so each must yield its own cached ``AsyncUlyssesOp`` instance + slot ring.
    """
    _run(2, _worker_multi_pg)
