# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank plumbing shared by this target's MNNVL op matrices.

A matrix is its own launcher. Run plainly, it spawns one worker per device named in
CUDA_VISIBLE_DEVICES (``spawn_ranks``) and kills them all on the first failure or at its deadline,
because a broken Lamport exchange hangs rather than raising. Run with ``--rank-worker`` it is one
rank (``init_rank``). The workers rendezvous through torch.distributed's env:// method without
MPI, and the MNNVL workspaces are built over the TP process group, as in a non-MPI serving engine.

The c1 world, 16 ranks on 4 nodes of 4 GPUs on one MNNVL fabric, is started by hand, one worker
per GPU, for example inside a 4-node allocation:

    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
    export MASTER_PORT=29511 TRTLLM_ENABLE_PDL=1
    srun -N4 --ntasks-per-node=4 python _mnnvl_hop_op_matrix.py --rank-worker

``init_rank`` then reads the rank, the world size and the local rank from Slurm. The checks that
need an oracle only a 1 MiB one-shot all-reduce provides skip the shapes above it.

Only torch is imported at module scope: the launcher must carry no tensorrt_llm state.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.distributed as dist

WORKER_FLAG = "--rank-worker"

# Lamport data per rank up to which the built-in MNNVL all-reduce runs one-shot, with the same
# reduction order as the hops: the shapes where it serves as a bitwise oracle.
BUILTIN_ONE_SHOT_BYTES = 1 << 20


def spawn_ranks(script: Path, deadline_s: int) -> None:
    """Start one worker per device in CUDA_VISIBLE_DEVICES and wait for all of them.

    The workers rendezvous through env:// on a free local port. Each gets its own session, so
    that on the first nonzero exit or at the deadline every session is killed.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns, e.g. 0,1,2,3"
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size >= 2, f"CUDA_VISIBLE_DEVICES names {world_size} device(s); need 2 or more"
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
    # The children rendezvous through env://, not through the launching step's PMIx: drop its
    # namespace, or the first hard exit trips srun's errhandler before anything is reported.
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    print(f"[launcher] spawning {world_size} ranks, rendezvous on :{port}", flush=True)
    children = [
        subprocess.Popen(
            [sys.executable, str(script), WORKER_FLAG],
            env=dict(env, RANK=str(r), LOCAL_RANK=str(r)),
            start_new_session=True,
        )
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

    deadline = time.monotonic() + deadline_s
    live = set(range(world_size))
    failure = None
    while live and failure is None:
        if time.monotonic() > deadline:
            failure = f"did not finish in {deadline_s}s (wedged)"
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


def init_rank() -> Tuple[int, int, Any]:
    """Bind this worker's GPU, join the process group, and return ``(rank, world, mapping)``.

    ``mapping`` is the pure tensor-parallel mapping over the whole world, the c1 server's.
    """
    os.environ["TLLM_DISABLE_MPI"] = "1"
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    # device_id keeps NCCL teardown off the implicit-device path; gloo serves the host
    # agreements of the workspace construction.
    dist.init_process_group(
        "cuda:nccl,cpu:gloo",
        init_method="env://",
        rank=rank,
        world_size=world,
        device_id=torch.device("cuda", local_rank),
    )
    from tensorrt_llm.mapping import Mapping

    gpus_per_node = min(world, torch.cuda.device_count())
    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=gpus_per_node, tp_size=world)
    return rank, world, mapping


def run_checks(checks, rank: int) -> None:
    """Run every check in order; a failing rank exits hard, and the launcher kills its peers."""
    for check in checks:
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {rank}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # A rank that leaves a collective early wedges its peers: exit at once.
            os._exit(13)
        dist.barrier()
    torch.cuda.synchronize()
    print(f"[rank {rank}] {len(checks)} checks passed", flush=True)
    sys.stdout.flush()
    # Process groups and MNNVL workspaces are torn down with the process.
    os._exit(0)


def normal(shape, seed: int, scale: float = 1.0, dtype: torch.dtype = torch.bfloat16):
    """Normal values times ``scale`` in ``dtype``, the same on every rank for one seed."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, generator=gen, device="cuda") * scale).to(dtype)


def eighths(shape, seed: int, bound: int, dtype: torch.dtype = torch.bfloat16):
    """Multiples of 1/8 in [-bound / 8, bound / 8], exact in BF16 and in their sums up to 32."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    raw = torch.randint(-bound, bound + 1, shape, generator=gen, device="cuda")
    return (raw.float() / 8.0).to(dtype)


_INTEGER_OF_SIZE = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def bits(t: torch.Tensor) -> torch.Tensor:
    """The tensor's bits as integers of its element size."""
    return t.view(_INTEGER_OF_SIZE[t.element_size()])


def assert_bitwise(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    """Gate of exactly zero, on the bits (NaN payloads and signed zeros included)."""
    assert out.dtype == ref.dtype, f"{where}: dtype {out.dtype} != {ref.dtype}"
    assert out.shape == ref.shape, f"{where}: shape {tuple(out.shape)} != {tuple(ref.shape)}"
    differ = bits(out) != bits(ref)
    if bool(differ.any()):
        first = tuple(int(i) for i in differ.nonzero()[0])
        raise AssertionError(
            f"{where}: {int(differ.sum())} of {out.numel()} elements differ, the first at {first}: "
            f"{out[first].item()} != {ref[first].item()}"
        )


def assert_same_on_every_rank(t: torch.Tensor, where: str) -> None:
    """Every rank of the world holds the same bits in ``t`` as rank 0."""
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t.contiguous())
    for r, other in enumerate(gathered):
        assert_bitwise(other, gathered[0], f"{where}: rank {r} vs rank 0")


def lamport_flags(buffer_flags: torch.Tensor) -> list[int]:
    """The nine Lamport flag words, as host integers."""
    torch.cuda.synchronize()
    return [int(v) for v in buffer_flags.view(torch.int32).tolist()]


def assert_rotated(before: list[int], after: list[int], bytes_to_clear: int, where: str) -> None:
    """One call later the current buffer is the dirty one, of one stage, with its bytes to clear."""
    expected = [
        (before[0] + 1) % 3,
        before[0],
        before[2],
        1,
        bytes_to_clear,
        0,
        0,
        0,
        0,
    ]
    assert after == expected, f"{where}: flags {after} after {before}, expected {expected}"
