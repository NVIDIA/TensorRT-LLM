# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU matrix for ``ops/comm/reducescatter__r21.py``; run plainly, it is its own launcher.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _reducescatter__r21_op_matrix.py

It spawns one rank per visible device and kills them all on the first failure or at
the deadline, because a broken collective hangs rather than raising.
"""

import functools
import os
import random
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.distributed as dist

assert torch.cuda.is_available(), "reducescatter requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 900

HIDDEN = 3072
# The decode graph batch sizes this checkpoint's engine captures (per-rank rows).
GRAPH_BATCH_SIZES = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64)

# Bound in the rank body: importing the entry pulls in tensorrt_llm, which the launcher must not.
reducescatter: Any = None
entry: Any = None
validating: Any = None
RANK = 0
WORLD = 1
GROUP: List[int] = [0]


def _block(rank, dest, rows, seed, trailing=(HIDDEN,), dtype=torch.bfloat16) -> torch.Tensor:
    """The part of `rank`'s input that lands on group position `dest`.

    Multiples of 1/8 with |x| <= 31/8: every partial sum of WORLD <= 8 of them is exact
    in bf16 in any order, so the gate can be bitwise.
    """
    gen = torch.Generator(device="cuda").manual_seed((seed * 977 + rank) * 131 + dest + 1)
    shape = (rows, *trailing)
    raw = torch.randint(-31, 32, shape, generator=gen, device="cuda", dtype=torch.int32)
    return (raw.float() / 8.0).to(dtype)


def _input(rank, sizes: Sequence[int], seed, trailing=(HIDDEN,), dtype=torch.bfloat16):
    blocks = [_block(rank, d, n, seed, trailing, dtype) for d, n in enumerate(sizes)]
    return torch.cat(blocks, dim=0)


def _ref(sizes: Sequence[int], seed, trailing=(HIDDEN,), dtype=torch.bfloat16) -> torch.Tensor:
    return entry.reducescatter.reference(
        [_block(r, RANK, sizes[RANK], seed, trailing, dtype) for r in range(WORLD)]
    )


def _assert_bitwise(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    try:
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
    except AssertionError as exc:
        raise AssertionError(f"{where}: {exc}") from exc


def check_certified_cells() -> None:
    for i, cell in enumerate(entry.reducescatter.CELLS):
        spec = cell.spec
        rows, trailing, dtype = spec["rows_per_rank"], spec["trailing"], spec["dtype"]
        seed = 9000 + 13 * i
        sizes = [rows] * WORLD
        x = _input(RANK, sizes, seed, trailing, dtype)
        with validating(entry.reducescatter):
            out = reducescatter(x, spec["sizes"], GROUP)
        _assert_bitwise(out, _ref(sizes, seed, trailing, dtype), cell.why)
        dist.barrier()


def check_ragged_reduce_scatter() -> None:
    """The split an eager attention-DP step passes, including a rank with no rows."""
    for i, sizes in enumerate(
        (
            [1 + 4 * r for r in range(WORLD)],
            [7 * (WORLD - r) for r in range(WORLD)],
            [0] + [3 + 2 * r for r in range(WORLD - 1)],
            [2048] + [1 + 388 * r for r in range(WORLD - 1)],
        )
    ):
        seed = 3000 + 7 * i
        out = reducescatter(_input(RANK, sizes, seed), sizes, GROUP)
        _assert_bitwise(out, _ref(sizes, seed), f"ragged sizes={sizes}")
        dist.barrier()


def check_cuda_graph_at_every_engine_batch_size() -> None:
    """One graph per decode batch size in a shared pool, replayed back to back on fresh inputs."""
    inputs: Dict[int, torch.Tensor] = {}
    outputs: Dict[int, torch.Tensor] = {}
    graphs: Dict[int, Any] = {}
    pool = None
    for rows in GRAPH_BATCH_SIZES:
        x = _input(RANK, [rows] * WORLD, 11000)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                reducescatter(x, None, GROUP)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        dist.barrier()
        # Outside the pool: pooled outputs alias across graphs under back-to-back replays.
        persistent = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
        graph = torch.cuda.CUDAGraph()
        ctx = torch.cuda.graph(graph) if pool is None else torch.cuda.graph(graph, pool=pool)
        with ctx:
            persistent.copy_(reducescatter(x, None, GROUP))
        if pool is None:
            pool = graph.pool()
        inputs[rows], outputs[rows], graphs[rows] = x, persistent, graph
        dist.barrier()

    shuffled = list(GRAPH_BATCH_SIZES)
    random.Random(1234).shuffle(shuffled)
    seed = 110000
    for name, order in (("descending", list(reversed(GRAPH_BATCH_SIZES))), ("shuffled", shuffled)):
        staged = {}
        for rows in order:
            seed += 1000
            staged[rows] = seed
            inputs[rows].copy_(_input(RANK, [rows] * WORLD, seed))
        for rows in order:
            graphs[rows].replay()
        torch.cuda.synchronize()
        for rows in order:
            ref = _ref([rows] * WORLD, staged[rows])
            _assert_bitwise(outputs[rows], ref, f"{name} rows={rows}")
        dist.barrier()


CHECKS = (
    check_certified_cells,
    check_ragged_reduce_scatter,
    check_cuda_graph_at_every_engine_batch_size,
)


def _run_one_rank() -> int:
    global RANK, WORLD, GROUP, reducescatter, validating, entry
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    # The rank owns device `rank`. Never LOCAL_RANK: every spawned rank inherits one value.
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method="env://", device_id=torch.device("cuda", rank))

    from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.comm import (  # noqa: E501
        reducescatter__r21 as entry,
    )

    # The `_pg` symbol, as the non-MPI router in `_torch/distributed/ops.py` calls it.
    reducescatter = functools.partial(
        entry.reducescatter, process_group=dist.distributed_c10d._get_default_group().boxed()
    )

    # parents[5] is tests/unittest/_torch/modeling_v2, where `_validating.py` lives.
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
    from _validating import validating as _validating_cm

    validating = _validating_cm
    RANK, WORLD = rank, world
    GROUP = list(range(WORLD))
    assert WORLD >= 2, f"a collective needs at least 2 ranks, got {WORLD}"

    for check in CHECKS:
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # A rank that leaves a collective early wedges its siblings; exit hard.
            os._exit(13)
    dist.barrier()
    torch.cuda.synchronize()
    print(f"[rank {RANK}] {len(CHECKS)} checks passed", flush=True)
    # No process-group teardown: destroying a communicator that captured graphs can wedge NCCL's watchdog.
    sys.stdout.flush()
    os._exit(0)


def _spawn_ranks() -> None:
    """One worker per visible device, each in its own session, all killed on the first failure or at the deadline."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns"
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size >= 2, f"a collective test needs at least 2 devices, got {world_size}"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(
        os.environ,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        TLLM_DISABLE_MPI="1",
    )
    # An inherited srun PMI namespace lets one failing rank kill the whole step unreported.
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    print(f"[launcher] spawning {world_size} ranks, rendezvous on :{port}", flush=True)
    children = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
            env=dict(env, RANK=str(r)),
            start_new_session=True,
        )
        for r in range(world_size)
    ]
    deadline = time.monotonic() + DEADLINE_S
    live = set(range(world_size))
    failure = None
    while live and failure is None:
        if time.monotonic() > deadline:
            failure = f"did not finish in {DEADLINE_S}s (wedged)"
            break
        for r in sorted(live):
            code = children[r].poll()
            if code is None:
                continue
            live.discard(r)
            if code != 0:
                failure = f"rank {r} exited {code}"
                break
        time.sleep(0.5)
    if failure is not None:
        for p in children:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for p in children:
            p.wait()
        raise AssertionError(f"the {world_size}-rank run failed: {failure}")


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    _spawn_ranks()
    print("OK")
