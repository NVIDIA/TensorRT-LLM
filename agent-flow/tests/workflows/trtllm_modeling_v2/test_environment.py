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


def test_a_single_task_scrub_unsets_mpi_bootstrap_and_keeps_mpi_settings():
    import subprocess

    from agent_flow.workflows.trtllm_modeling_v2.infra.compute import environment as env_mod

    scrub = env_mod.single_task_scrub(["OMPI_MCA_pml"])
    script = f"{scrub}; env | grep -E '^(OMPI|PMIX|PMI|SLURM)_' | sort"
    env = {
        "PATH": "/usr/bin:/bin",
        "OMPI_COMM_WORLD_RANK": "0",
        "PMIX_RANK": "0",
        "SLURM_NTASKS": "4",
        "SLURM_PROCID": "0",
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        "OMPI_MCA_pml": "ob1",
    }
    left = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True).stdout
    assert left.split() == [
        "OMPI_ALLOW_RUN_AS_ROOT=1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1",
        "OMPI_MCA_pml=ob1",
        "SLURM_PROCID=0",
    ]


def test_ranks_find_the_launcher_beside_the_interpreter():
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment

    command = RunEnvironment("local", "/opt/venv/bin/python", world_size=4).command("python x.py")
    assert 'PATH=/opt/venv/bin:"$PATH" env -u' in command and "trtllm-llmapi-launch" in command
    bare = RunEnvironment("local", "python3", world_size=4).command("python x.py")
    assert "PATH=" not in bare.split("mpirun")[0]
