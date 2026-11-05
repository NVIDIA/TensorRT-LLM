# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the `cute_dsl_allreduce_add_rmsnorm` entry.

The matrix is `_cute_dsl_allreduce_add_rmsnorm_op_matrix.py` beside this file,
which drives the entry's `CELLS` in two jobs, one test each:

- the world-4 cells: its own 4-rank launcher over 4 devices of one node,
  which `_rank_job` starts in a fresh interpreter and whose exit code it turns
  into an assertion;
- the world-8 cells (Mistral Large 4 NVFP4 tep8 and tp8): 8 ranks over two nodes
  of 4 devices, one rank per device, started by an MPI launcher as the
  engine's ranks are. A process cannot place ranks on another node by itself,
  so this test takes that launcher from `MODELING_V2_MPI_LAUNCH` -- a command
  prefix that starts the command after it once per rank over the job's two
  nodes, four per node -- and skips without it, as on the one 4-GPU node
  pre-merge runs this file on. On a two-node compute session whose node-0
  scheduler holds both nodes for a job, that is
  `bash <its infra>/scripts/mn/mn_mpirun.sh 4`.
"""

import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

__extra_import_path__ = ["../../../../comm"]
import _rank_job  # noqa: E402 — needs the path declared above

_HERE = Path(__file__).resolve().parent
_MATRIX = _HERE / "_cute_dsl_allreduce_add_rmsnorm_op_matrix.py"
#: Names the launcher that starts the world-8 cells' ranks.
_LAUNCH_ENV = "MODELING_V2_MPI_LAUNCH"
#: Headroom over the ranks' own deadline, so their faulthandler dump wins the race.
_LAUNCHER_GRACE_S = 300
# The engine's ranks run with MPI on; and the launcher must not join an enclosing
# MPI job or Slurm step. (Each rank drops any rendezvous address itself: a node's
# own environment can hold one this process never sees.)
_DROPPED = ("TLLM_DISABLE_MPI",)
_DROPPED_PREFIXES = ("PMIX_", "PMI_", "OMPI_", "SLURM_", "PRTE_")


# Spawns 4 rank processes over the visible devices. Under xdist several
# workers would fight for the same GPUs, so this must run alone, like the
# other collective entries.
def _sm103_devices() -> int:
    """How many of the devices the job would claim are sm_103."""
    if not torch.cuda.is_available():
        return 0
    claimed = range(min(torch.cuda.device_count(), _rank_job.WORLD_SIZE))
    return sum(torch.cuda.get_device_capability(i) == (10, 3) for i in claimed)


@pytest.mark.no_xdist
def test_cute_dsl_allreduce_add_rmsnorm_op_matrix() -> None:
    if _sm103_devices() < _rank_job.WORLD_SIZE:
        pytest.skip(f"the Lamport all-reduce entry runs on {_rank_job.WORLD_SIZE} sm_103 GPUs")
    _rank_job.run("cute_dsl_allreduce_add_rmsnorm", directory=_HERE)


@pytest.mark.no_xdist
def test_cute_dsl_allreduce_add_rmsnorm_world8_mpi_ranks() -> None:
    """The world-8 cells: the matrix's MPI job, under the environment's launcher."""
    launch = os.environ.get(_LAUNCH_ENV, "").strip()
    if not launch:
        pytest.skip(
            f"the world-8 cells need 8 ranks over two nodes, which this process cannot "
            f"start; set {_LAUNCH_ENV} to a launcher that starts them"
        )
    constants = _rank_job._load_launcher_constants(_MATRIX)
    # tests/unittest/_torch/modeling_v2/models/mistral_large4/ops/comm -> repo root.
    repo_root = _HERE.parents[7]
    assert (repo_root / "tensorrt_llm").is_dir(), f"expected the repo root at {repo_root}"
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _DROPPED and not k.startswith(_DROPPED_PREFIXES)
    }
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)
    command = [*shlex.split(launch), sys.executable, str(_MATRIX), "--mpi-rank-worker"]
    print(f"[launcher] {shlex.join(command)}", flush=True)
    # The launcher's own session, so a wedged job is ended as a whole: mpirun
    # takes SIGTERM down to every rank it started, on every node.
    job = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        code = job.wait(timeout=constants.MPI_DEADLINE_S + _LAUNCHER_GRACE_S)
    except subprocess.TimeoutExpired:
        os.killpg(job.pid, signal.SIGTERM)
        try:
            job.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(job.pid, signal.SIGKILL)
            job.wait()
        raise AssertionError(
            f"the world-8 job did not finish in "
            f"{constants.MPI_DEADLINE_S + _LAUNCHER_GRACE_S}s (wedged)"
        ) from None
    assert code == 0, f"the world-8 job exited {code}: {shlex.join(command)}"
