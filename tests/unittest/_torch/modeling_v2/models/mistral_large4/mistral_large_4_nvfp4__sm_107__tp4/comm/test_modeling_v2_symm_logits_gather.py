# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the sm_107 tp4 target's `symm_logits_gather_softcap`.

The matrix is `_symm_logits_gather_op_matrix.py` beside this file, which drives the entry's
`CELLS` in two jobs, one test each:

- the world-4 cells (tp4): its own 4-rank launcher over 4 devices of one node, started here in
  a fresh interpreter, which must carry no CUDA or torch.distributed state of its own;
- the world-8 cells (tp8): 8 ranks over two nodes of 4 devices, one per device, started by an
  MPI launcher as the tp8 engine's ranks are. A process cannot place ranks on another node by
  itself, so this test takes that launcher from `MODELING_V2_MPI_LAUNCH` -- a command prefix
  that starts the command after it once per rank over the job's two nodes, four per node --
  and skips without it.
"""

import importlib.util
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires VR200/SM107",
)

_HERE = Path(__file__).resolve().parent
_MATRIX = _HERE / "_symm_logits_gather_op_matrix.py"
#: Names the launcher that starts the world-8 cells' ranks.
_LAUNCH_ENV = "MODELING_V2_MPI_LAUNCH"
#: Headroom over the ranks' own deadline, so their message wins the race.
_LAUNCHER_GRACE_S = 300
_WORLD_SIZE = 4
_DROPPED = ("TLLM_DISABLE_MPI",)
_DROPPED_PREFIXES = ("PMIX_", "PMI_", "OMPI_", "SLURM_", "PRTE_")


def _env() -> dict:
    # tests/unittest/_torch/modeling_v2/models/mistral_large4/<target>/comm -> repo root.
    repo_root = _HERE.parents[7]
    assert (repo_root / "tensorrt_llm").is_dir(), f"expected the repo root at {repo_root}"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)
    return env


def _constants():
    """The matrix's module-level deadlines, read without importing tensorrt_llm."""
    spec = importlib.util.spec_from_file_location(_MATRIX.stem, _MATRIX)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Spawns 4 rank processes over the visible devices; under xdist several workers would fight for
# the same GPUs, so this runs alone, like the other collective entries.
@pytest.mark.no_xdist
def test_symm_logits_gather_world4() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [d for d in visible.split(",") if d.strip()]
        if visible
        else [str(i) for i in range(torch.cuda.device_count())]
    )
    if len(devices) < _WORLD_SIZE:
        pytest.skip(f"the world-4 cells need {_WORLD_SIZE} devices, {len(devices)} visible")
    env = dict(_env(), CUDA_VISIBLE_DEVICES=",".join(devices[:_WORLD_SIZE]))
    completed = subprocess.run(
        [sys.executable, str(_MATRIX)],
        env=env,
        capture_output=True,
        text=True,
        timeout=_constants().DEADLINE_S + _LAUNCHER_GRACE_S,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{_MATRIX.name} exited {completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
        )


@pytest.mark.no_xdist
def test_symm_logits_gather_world8_mpi_ranks() -> None:
    launch = os.environ.get(_LAUNCH_ENV, "").strip()
    if not launch:
        pytest.skip(
            f"the world-8 cells need 8 ranks over two nodes, which this process cannot start; "
            f"set {_LAUNCH_ENV} to a launcher that starts them"
        )
    deadline = _constants().MPI_DEADLINE_S + _LAUNCHER_GRACE_S
    env = {
        k: v for k, v in _env().items() if k not in _DROPPED and not k.startswith(_DROPPED_PREFIXES)
    }
    command = [*shlex.split(launch), sys.executable, str(_MATRIX), "--mpi-rank-worker"]
    print(f"[launcher] {shlex.join(command)}", flush=True)
    # The launcher's own session, so a wedged job is ended as a whole.
    job = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        code = job.wait(timeout=deadline)
    except subprocess.TimeoutExpired:
        os.killpg(job.pid, signal.SIGTERM)
        try:
            job.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(job.pid, signal.SIGKILL)
            job.wait()
        raise AssertionError(f"the world-8 job did not finish in {deadline}s (wedged)") from None
    assert code == 0, f"the world-8 job exited {code}: {shlex.join(command)}"
