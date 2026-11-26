# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank launchers for the collective matrices beside it, which import it from launcher code only.
Module scope imports only the standard library, since `_rank_job` runs a matrix's module scope
from pytest. The deadlines are there because a broken collective hangs rather than raising."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

#: Slurm step variables an mpirun inside a one-task step must not inherit: with them OpenMPI
#: sizes its slots from the step (one) and every rank reads the step's own rank.
SLURM_STEP_VARS = (
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "SLURM_LOCALID",
    "SLURM_NODEID",
    "SLURM_STEP_ID",
    "SLURM_TASK_PID",
    "SLURM_STEP_NUM_TASKS",
)


def visible_devices() -> list[str]:
    """The devices this run owns: CUDA_VISIBLE_DEVICES, which must name them."""
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns, e.g. 0,1,2,3"
    return visible


def _claim(world: int) -> str:
    devices = visible_devices()
    assert 2 <= world <= len(devices), (
        f"world size {world} needs 2..{len(devices)} visible devices (one rank per device)"
    )
    return ",".join(devices[:world])


def mpirun_ranks(entry_file: str, world: int, args: Sequence[str], deadline_s: int) -> None:
    """Re-execute `entry_file` as `world` MPI ranks on the first `world` visible devices, without
    `TLLM_DISABLE_MPI` (the ranks need tensorrt_llm's MPI session); raise on a nonzero exit or
    past `deadline_s`, killing the process group."""
    command = [
        "mpirun",
        "--allow-run-as-root",
        "--oversubscribe",
        "-n",
        str(world),
        sys.executable,
        str(Path(entry_file).resolve()),
        *args,
    ]
    env = {
        k: v for k, v in os.environ.items() if k not in SLURM_STEP_VARS and k != "TLLM_DISABLE_MPI"
    }
    env["CUDA_VISIBLE_DEVICES"] = _claim(world)
    print(f"[launcher] {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        code = process.wait(timeout=deadline_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise AssertionError(
            f"the {world}-rank MPI run did not finish in {deadline_s}s (wedged)"
        ) from None
    assert code == 0, f"the {world}-rank MPI run exited {code}"


def spawn_pg_ranks(entry_file: str, world: int, args: Sequence[str], deadline_s: int) -> None:
    """Start `entry_file` as `world` torch.distributed ranks (`env://` on a free local port), each
    in its own session; kill them all and raise on the first nonzero exit or past `deadline_s`."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=_claim(world),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world),
        TLLM_DISABLE_MPI="1",
    )
    # Drop the srun step's PMIx/PMI/OMPI namespace: a child that inherits it joins the step, and
    # its hard exit makes srun SIGKILL the whole step before any error is reported.
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    print(f"[launcher] spawning {world} torch.distributed ranks, rendezvous on :{port}", flush=True)
    children = [
        subprocess.Popen(
            [sys.executable, str(Path(entry_file).resolve()), *args],
            env=dict(env, RANK=str(r)),
            start_new_session=True,
        )
        for r in range(world)
    ]
    deadline = time.monotonic() + deadline_s
    live = set(range(world))
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
        for p in children:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for p in children:
            p.wait()
        raise AssertionError(f"the {world}-rank torch.distributed run failed: {failure}")
