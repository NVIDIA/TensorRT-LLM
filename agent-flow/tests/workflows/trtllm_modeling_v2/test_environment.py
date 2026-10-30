# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the resolved launch environment."""

from __future__ import annotations

from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import (
    MULTI_TASK_SLURM_SCRUB,
    RunEnvironment,
)


def test_every_command_scrubs_the_allocations_launcher_environment():
    """Otherwise every rank is rank 0 of a world of 1, and the run hangs.

    `trtllm-llmapi-launch` reads `SLURM_PROCID` before
    `OMPI_COMM_WORLD_RANK` and `SLURM_NTASKS` before `OMPI_COMM_WORLD_SIZE`.
    Inside a one-task allocation those are `0` and `1` for every rank, so all
    four become `mgmn_leader_node` and wait forever for workers.

    mpi4py auto-initializes whenever PMI variables are present. A spool script
    runs inside the session's own allocation, so they are, and a lone probe
    would join a world it never asked for.
    """
    env = RunEnvironment(kind="session_spool", python="/p", world_size=4, spool="/s")
    command = env.command("/p x.py")
    for name in ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID", "SLURM_NODEID"):
        assert f"-u {name}" in command, name
    assert command.index(MULTI_TASK_SLURM_SCRUB) < command.index("mpirun")
    single = env.command("/p probe.py", single_process=True)
    assert "${!PMIX@}" in single
    assert "exec /p probe.py" in single
