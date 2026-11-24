# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank plumbing for the K3.1 ULL target's collective entries.

An entry's matrix, ``_<entry>_matrix.py`` beside this file, is its own launcher: run plainly it
spawns one worker per device named in ``CUDA_VISIBLE_DEVICES`` and kills them all on the first
failure or at its deadline (a broken Lamport exchange hangs rather than raising); run with
``--rank-worker`` it is one rank. Workers rendezvous through torch.distributed's env:// method with
MPI disabled and build their multicast workspaces over the TP process group. ``run`` starts a
matrix from pytest in a fresh interpreter (it must carry no CUDA or torch.distributed state) and
turns its exit code into an assertion.

Only the standard library and torch are imported at module scope: the launcher must carry no
tensorrt_llm state.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

WORKER_FLAG = "--rank-worker"
WORLD_SIZE = 4
"""One GB300 tray. The served world (16 ranks over 4 trays) is not reachable from one node."""

_LAUNCHER_GRACE_S = 120


def visible_devices() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [d for d in visible.split(",") if d.strip()]
    return [str(i) for i in range(torch.cuda.device_count())]


def run(entry: str) -> None:
    """Run ``_<entry>_matrix.py``'s launcher over the first ``WORLD_SIZE`` visible devices."""
    launcher = Path(__file__).resolve().with_name(f"_{entry}_matrix.py")
    devices = visible_devices()
    assert len(devices) >= WORLD_SIZE, f"{len(devices)} device(s) visible, need {WORLD_SIZE}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(devices[:WORLD_SIZE]))
    package = importlib.util.find_spec("tensorrt_llm")
    assert package is not None and package.origin is not None, "tensorrt_llm is not importable"
    package_parent = str(Path(package.origin).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (package_parent, env.get("PYTHONPATH", "")) if p)
    spec = importlib.util.spec_from_file_location(launcher.stem, launcher)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    completed = subprocess.run(
        [sys.executable, str(launcher)],
        env=env,
        capture_output=True,
        text=True,
        timeout=module.DEADLINE_S + _LAUNCHER_GRACE_S,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{launcher.name} exited {completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout[-20000:]}\n"
            f"--- stderr ---\n{completed.stderr[-20000:]}"
        )
    print(completed.stdout[-4000:])


def spawn_ranks(script: Path, deadline_s: int) -> None:
    """One worker per device in ``CUDA_VISIBLE_DEVICES``, each in its own session; on the first
    nonzero exit or at the deadline every session is killed."""
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
    # The workers rendezvous through env://, not through the launching step's PMIx.
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


def init_rank(helix: bool = False):
    """Bind this worker's GPU, join the process group; ``(rank, world, mapping)`` of a pure TP
    group over the whole world, or with ``helix`` of a Helix CP group with the FIFO transport (the
    device mesh is built once per process: one kind of group per worker)."""
    os.environ["TLLM_DISABLE_MPI"] = "1"
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "cuda:nccl,cpu:gloo",
        init_method="env://",
        rank=rank,
        world_size=world,
        device_id=torch.device("cuda", local_rank),
    )
    from tensorrt_llm.mapping import CpType, Mapping

    gpus_per_node = min(world, torch.cuda.device_count())
    if helix:
        cp_config = {"cp_type": CpType.HELIX, "use_nccl_for_alltoall": False, "fifo_version": 2}
        mapping = Mapping(
            world_size=world,
            rank=rank,
            gpus_per_node=gpus_per_node,
            cp_size=world,
            cp_config=cp_config,
        )
    else:
        mapping = Mapping(world_size=world, rank=rank, gpus_per_node=gpus_per_node, tp_size=world)
    return rank, world, mapping


def run_checks(checks, rank: int) -> None:
    """Every check in order; a failing rank exits hard, and the launcher kills its peers."""
    for check in checks:
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {rank}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(13)
        dist.barrier()
        if rank == 0:
            print(f"[rank 0] passed {check.__name__}", flush=True)
    torch.cuda.synchronize()
    print(f"[rank {rank}] {len(checks)} checks passed", flush=True)
    sys.stdout.flush()
    os._exit(0)


def eighths(shape, seed: int, bound: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Multiples of 1/8 in [-bound / 8, bound / 8]: exact in BF16, and so are sums of up to
    ``256 / bound`` of them, in any order."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    raw = torch.randint(-bound, bound + 1, shape, generator=gen, device="cuda")
    return (raw.float() / 8.0).to(dtype)


def normal(shape, seed: int, scale: float = 1.0, dtype: torch.dtype = torch.bfloat16):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, generator=gen, device="cuda") * scale).to(dtype)


def assert_bitwise(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    assert out.dtype == ref.dtype and out.shape == ref.shape, (
        f"{where}: {out.dtype} {tuple(out.shape)} != {ref.dtype} {tuple(ref.shape)}"
    )
    ints = {2: torch.int16, 4: torch.int32}[out.element_size()]
    differ = out.view(ints) != ref.view(ints)
    if bool(differ.any()):
        first = tuple(int(i) for i in differ.nonzero()[0])
        raise AssertionError(
            f"{where}: {int(differ.sum())} of {out.numel()} elements differ, the first at {first}: "
            f"{out[first].item()} != {ref[first].item()}"
        )
