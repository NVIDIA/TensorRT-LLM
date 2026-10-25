# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Certify `comm/allgather_cast`: every rank's fp32 rows equal `reference` (the all-gather, the vocab concat and
`.float()`) bit for bit, on every cell certified at this run's world size.

Every rank builds every rank's shard from one seed, so each rank checks its own output against the reference over
all shards. Per cell: random logits with adversarial bits planted (-0.0, +-inf, NaN payloads, denormals, the
0x80000000 pair word), several seeds, inside `validating`. Also: a captured call replayed with fresh contents copied
into its static input before every replay (the epoch and the buffer rotation advance on the device); call-unique
contents while the grid alternates between 16 CTAs and the picked grid; `supports` refuses more rows than the
workspace holds, fp32 shards and partial 16-byte vectors.

World 4 (one GB300 tray) is spawned from one process:

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _allgather_cast_op_matrix.py

World 8 spans two trays and runs as one srun task per rank, like `_mnnvl_allreduce_op_matrix.py`:

    MASTER_ADDR=<first node> MASTER_PORT=<port> python _allgather_cast_op_matrix.py --srun

under ``srun --ntasks=8 --ntasks-per-node=4 --kill-on-bad-exit=1``.
"""

import importlib.util
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist

assert torch.cuda.is_available(), "allgather_cast requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 900
SEEDS = 4
GPUS_PER_NODE = 4
"""A GB300 tray. The mapping needs it to see a world-8 group span two nodes."""
# bf16 bit patterns planted into every shard: -0.0, +inf, -inf, NaN payloads, denormals.
SPECIAL_BITS = (-32768, 0x7F80, -128, 0x7FC1, -63, 0x0001, -32767, 0x7FFF)

RANK = 0
WORLD = 1
# Bound inside the rank body: `entry` is the module, `op` its `allgather_cast` instance (the call surface,
# `reference`, `is_valid` and `CELLS`), `WS` the workspace arguments of every call, per shard width.
entry: Any = None
op: Any = None
validating: Any = None
WS: dict = {}


def _shards(seed: int, rows: int, cols: int):
    """Every rank's bf16 [rows, cols] shard (the same on every rank), adversarial bits planted."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for _ in range(WORLD):
        x = (torch.randn(rows, cols, device="cuda", generator=g) * 3).bfloat16()
        w = x.view(torch.int16)
        # Distinct columns: scatter_ with a repeated index keeps an arbitrary writer, which can differ between
        # devices, and every rank must build the same shards.
        cols_hit = torch.randperm(cols, generator=g, device="cuda")[:64]
        special = torch.tensor(SPECIAL_BITS, dtype=torch.int16, device="cuda").repeat(8)
        w[:, cols_hit] = special
        w[:, 0] = 0
        w[:, 1] = -32768  # the int32 word 0x80000000
        out.append(x)
    return out


def _call(shard: torch.Tensor, grid: int = 0) -> torch.Tensor:
    return op(shard, **WS[shard.shape[1]], grid=grid)


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.view(torch.int32), b.view(torch.int32))


def check_certified_cells() -> None:
    cells = [c for c in op.CELLS if c.spec["world"] == WORLD]
    assert cells, f"no cell is certified at world size {WORLD}"
    for cell in cells:
        rows, cols = cell.spec["rows"], cell.spec["cols"]
        for seed in range(SEEDS):
            shards = _shards(seed, rows, cols)
            with validating(op):
                got = _call(shards[RANK])
            assert _bits_equal(got, op.reference(shards)), (
                f"{cell.why}: seed {seed} differs on rank {RANK}"
            )
        dist.barrier()
        if RANK == 0:
            print(f"[cell] world={WORLD} {cell.why}: bitwise", flush=True)


def check_graph_replays_fresh_contents() -> None:
    cell = [c for c in op.CELLS if c.spec["world"] == WORLD][0]
    rows, cols = cell.spec["rows"], cell.spec["cols"]
    static = _shards(100, rows, cols)[RANK].clone()
    _call(static)
    torch.cuda.synchronize()
    graph_out = {}
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            graph_out["t"] = _call(static)
    torch.cuda.current_stream().wait_stream(s)
    for i in range(6):
        shards = _shards(200 + i, rows, cols)
        static.copy_(shards[RANK])
        g.replay()
        torch.cuda.synchronize()
        assert _bits_equal(graph_out["t"], op.reference(shards)), (
            f"replay {i} returned stale or wrong rows"
        )
    dist.barrier()


def check_alternating_grids() -> None:
    cell = [c for c in op.CELLS if c.spec["world"] == WORLD][0]
    rows, cols = cell.spec["rows"], cell.spec["cols"]
    for i in range(24):
        shards = _shards(300 + i, rows, cols)
        got = _call(shards[RANK], grid=16 if i % 2 else 0)
        assert _bits_equal(got, op.reference(shards)), (
            f"call {i} (grid {16 if i % 2 else 'picked'}) differs"
        )
    dist.barrier()


def check_supports_refuses() -> None:
    cols = [c for c in op.CELLS if c.spec["world"] == WORLD][0].spec["cols"]
    shard = torch.zeros(entry.MAX_ROWS + 1, cols, dtype=torch.bfloat16, device="cuda")
    assert not entry.supports(shard, WORLD), "more rows than the workspace holds"
    assert not entry.supports(shard[:4].float(), WORLD), "an fp32 shard"
    assert not entry.supports(
        torch.zeros(4, cols - 4, dtype=torch.bfloat16, device="cuda"), WORLD
    ), "a partial 16-byte vector"
    assert not entry.supports(shard[:4, :-8], WORLD), "a strided view"
    dist.barrier()


CHECKS = (
    check_certified_cells,
    check_graph_replays_fresh_contents,
    check_alternating_grids,
    check_supports_refuses,
)


def _run_one_rank(device: int) -> int:
    global RANK, WORLD, WS, entry, op, validating
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(device)
    dist.init_process_group(
        "cuda:nccl,cpu:gloo", init_method="env://", device_id=torch.device("cuda", device)
    )
    RANK, WORLD = rank, world

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import (
        allgather_cast as entry_module,
    )
    from tensorrt_llm.mapping import Mapping

    entry = entry_module
    op = entry.allgather_cast
    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=GPUS_PER_NODE, tp_size=world)
    for cols in sorted({c.spec["cols"] for c in op.CELLS if c.spec["world"] == world}):
        WS[cols] = entry.workspace_args(mapping, cols, dist.group.WORLD)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _validating import validating as _validating_cm

    validating = _validating_cm

    for check in CHECKS:
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(13)
        print(f"[rank {RANK}] {check.__name__} passed", flush=True)
    dist.barrier()
    torch.cuda.synchronize()
    sys.stdout.flush()
    os._exit(0)


def _spawn_ranks() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns"
    world_size = len([d for d in visible.split(",") if d.strip()])
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size),
               TLLM_DISABLE_MPI="1")  # fmt: skip
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    children = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
            env=dict(env, RANK=str(r), LOCAL_RANK=str(r)),
            start_new_session=True,
        )  # fmt: skip
        for r in range(world_size)
    ]

    def _kill_all() -> None:
        for p in children:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for p in children:
            p.wait()

    deadline = time.monotonic() + DEADLINE_S
    live = set(range(world_size))
    failure: Optional[str] = None
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
        _kill_all()
        raise AssertionError(f"the {world_size}-rank run failed: {failure}")


def _rank_job():
    """`_rank_job.py` beside this file, by path (a sibling directory ships its own module of that name)."""
    path = Path(__file__).resolve().with_name("_rank_job.py")
    spec = importlib.util.spec_from_file_location("_comm_rank_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    if "--srun" in sys.argv:
        sys.exit(_run_one_rank(device=_rank_job().adopt_srun_env()))
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank(device=int(os.environ["RANK"])))
    _spawn_ranks()
    print("OK")
