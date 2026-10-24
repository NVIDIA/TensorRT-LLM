# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the resolved launch environment.

The module resolves how GPU work is launched into one string before any
agent runs, so the tests care about exactly one thing: that the string is
right, and that the prompt block quotes the same string the orchestrator
will run.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra import environment
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import (
    MULTI_TASK_SLURM_SCRUB,
    MULTI_TASK_SLURM_VARS,
    RunEnvironment,
    resolve_environment,
)

_LOCAL_TASK = {
    "environment": {
        "kind": "local",
        "python": "/opt/venv/bin/python",
        "env": {"TRTLLM_MODELING_V2": "require", "TMPDIR": "/scratch/tmp"},
    }
}


def test_local_prefix_carries_devices_and_env():
    env = resolve_environment(_LOCAL_TASK, devices=[2, 3], world_size=2)
    prefix = env.prefix
    assert prefix.startswith("CUDA_VISIBLE_DEVICES=2,3 ")
    assert "TRTLLM_MODELING_V2=require" in prefix
    assert "TMPDIR=/scratch/tmp" in prefix


def test_local_env_order_is_stable():
    """A prefix that reorders between calls makes two identical runs look different."""
    env = resolve_environment(_LOCAL_TASK, devices=[0], world_size=1)
    assert env.prefix == env.prefix
    assert env.prefix.index("TMPDIR") < env.prefix.index("TRTLLM_MODELING_V2")


def test_local_prefix_without_devices_omits_the_mask():
    env = resolve_environment(_LOCAL_TASK, devices=[], world_size=1)
    assert "CUDA_VISIBLE_DEVICES" not in env.prefix


def test_only_a_local_run_owns_its_devices():
    """One answer for preflight's claim, the smith's device and a gate's files.

    A kind added later is the scheduler's until someone says otherwise.
    """
    from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import ENVIRONMENT_KINDS

    assert not environment.scheduler_owns_devices("local")
    assert {k for k in ENVIRONMENT_KINDS if not environment.scheduler_owns_devices(k)} == {"local"}
    assert environment.scheduler_owns_devices("a_kind_added_later")


def test_command_prefixes_a_string():
    env = resolve_environment(_LOCAL_TASK, devices=[1], world_size=1)
    assert env.command("trtllm-eval --model x gsm8k").endswith("trtllm-eval --model x gsm8k")
    assert env.command("trtllm-eval").startswith("CUDA_VISIBLE_DEVICES=1")


def test_command_quotes_a_sequence():
    env = resolve_environment(_LOCAL_TASK, devices=[], world_size=1)
    assert "'a b'" in env.command(["echo", "a b"])


def test_python_command_uses_the_declared_interpreter():
    env = resolve_environment(_LOCAL_TASK, devices=[0], world_size=1)
    assert "/opt/venv/bin/python script.py" in env.python_command("script.py")


def test_env_values_with_spaces_are_quoted():
    task = {
        "environment": {
            "kind": "local",
            "python": "/p",
            "env": {"SOME_FLAGS": "-a -b c"},
        }
    }
    env = resolve_environment(task, devices=[], world_size=1)
    assert "SOME_FLAGS='-a -b c'" in env.prefix


def test_describe_quotes_the_same_prefix_the_orchestrator_runs():
    """The prompt block and the gate invocation must not be able to drift."""
    env = resolve_environment(_LOCAL_TASK, devices=[0, 1], world_size=2)
    described = env.describe()
    assert env.prefix in described
    assert env.python in described
    assert "world size 2" in described
    assert "symptom to report" in described


def test_world_size_is_at_least_one():
    env = resolve_environment(_LOCAL_TASK, devices=[], world_size=0)
    assert env.world_size == 1


def test_environment_is_frozen():
    env = RunEnvironment(kind="local", python="/p")
    try:
        env.kind = "session_spool"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("RunEnvironment must be immutable once resolved")


# ------------------------------------------------- the multi-rank launch rule


def test_spool_and_local_apply_the_same_rule_through_mpirun():
    """`mpirun` creates the ranks and the launcher assigns leader and workers."""
    env = RunEnvironment(kind="session_spool", python="/venv/bin/python", world_size=4, spool="/s")
    command = env.command("/venv/bin/python x.py")
    assert "mpirun -n 4" in command
    assert "trtllm-llmapi-launch" in command


def test_a_single_rank_spool_command_is_left_alone():
    env = RunEnvironment(kind="session_spool", python="/venv/bin/python", world_size=1, spool="/s")
    assert "mpirun" not in env.command("/venv/bin/python x.py")


# ------------------------- mpirun inside someone else's Slurm allocation
#
# Two defects, one cause: a `session_spool` job is ONE Slurm task that owns
# the node and runs `mpirun` itself, so both Open MPI's slot accounting and
# `trtllm-llmapi-launch`'s rank derivation read the allocation and get the
# wrong answer. Neither is a property of the model, the target or the run.


def test_mpirun_oversubscribes_because_the_allocation_is_one_task():
    """`-n 4` against a one-task allocation dies before any Python runs.

    Open MPI takes its slot count from the resource manager and refuses with
    *"There are not enough slots available in the system"*. The allocation
    holds the whole node, so the slot count is bookkeeping.
    """
    env = RunEnvironment(kind="session_spool", python="/venv/bin/python", world_size=4, spool="/s")
    assert "mpirun -n 4 --oversubscribe --allow-run-as-root" in env.command("/venv/bin/python x.py")


def test_mpirun_scrubs_the_slurm_task_vars_the_launcher_reads_first():
    """Otherwise every rank is rank 0 of a world of 1, and the run hangs.

    `trtllm-llmapi-launch` reads `SLURM_PROCID` before
    `OMPI_COMM_WORLD_RANK` and `SLURM_NTASKS` before `OMPI_COMM_WORLD_SIZE`.
    Inside a one-task allocation those are `0` and `1` for every rank, so all
    four become `mgmn_leader_node` and wait forever for workers.
    """
    env = RunEnvironment(kind="session_spool", python="/venv/bin/python", world_size=4, spool="/s")
    command = env.command("/venv/bin/python x.py")
    for name in ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID", "SLURM_NODEID"):
        assert f"-u {name}" in command, name
    assert command.index(MULTI_TASK_SLURM_SCRUB) < command.index("mpirun")


def test_the_scrub_names_every_var_the_step_sets_for_its_task():
    """A partial scrub still leaves the launcher a Slurm answer to prefer."""
    assert set(MULTI_TASK_SLURM_VARS) == {
        "SLURM_PROCID",
        "SLURM_NTASKS",
        "SLURM_LOCALID",
        "SLURM_NODEID",
        "SLURM_STEP_ID",
        "SLURM_TASK_PID",
        "SLURM_STEP_NUM_TASKS",
    }


def test_a_single_rank_spool_command_gets_neither_workaround():
    """Both are properties of `mpirun`, and at one task there is no `mpirun`."""
    env = RunEnvironment(kind="session_spool", python="/p", world_size=1, spool="/s")
    command = env.command("/p x.py", single_process=True)
    assert "--oversubscribe" not in command
    assert "env -u SLURM_PROCID" not in command


def test_a_single_process_command_at_world_size_4_gets_neither():
    """The probe is re-asked at one task, so it leaves the `mpirun` rules behind."""
    env = RunEnvironment(kind="session_spool", python="/p", world_size=4, spool="/s")
    command = env.python_command("probe.py", single_process=True)
    assert "--oversubscribe" not in command
    assert "env -u SLURM_PROCID" not in command


def test_the_spool_prompt_quotes_the_line_the_orchestrator_will_run():
    """The prompt is the agent's only instruction, so a stale copy is a defect.

    An agent handed the bare `mpirun` line reproduces both failures by hand,
    and the second of them presents as an engine that never boots.
    """
    env = RunEnvironment(kind="session_spool", python="/p", world_size=4, spool="/s")
    block = env.describe()
    assert "mpirun -n 4 --oversubscribe --allow-run-as-root" in block
    assert MULTI_TASK_SLURM_SCRUB in block
    assert "Copy that line whole" in block


def test_a_multi_rank_spool_prompt_tells_the_agent_how_to_launch():
    """The prompt is the agent's only instruction; it must carry the rule."""
    env = RunEnvironment(kind="session_spool", python="/p", world_size=4, spool="/s")
    block = env.describe()
    assert "mpirun -n 4" in block
    assert "trtllm-llmapi-launch" in block
    assert "MPI_ERR_SPAWN" in block
    assert "Do not work around that by reducing the world size" in block


def test_a_single_rank_spool_prompt_stays_quiet_about_it():
    """A rule a run cannot break is noise in that run's prompt."""
    env = RunEnvironment(kind="session_spool", python="/p", world_size=1, spool="/s")
    assert "mpirun" not in env.describe()


# ------------------------------------------- work that is not engine work

# A read-only probe is launched by the same module and by a different rule.
# The rule is the one that already exists for one task -- no launcher, MPI
# scrubbed -- asked for by name rather than re-derived by the caller.


def test_a_single_process_command_drops_the_multi_rank_launcher():
    """The inventory probe wants one process at any world size.

    It imports the build, enumerates it and exits. Sent through the engine's
    launch rules at world size 4 it would become
    `mpirun -n 4 ... trtllm-llmapi-launch`, which needs an MPI the probe has
    no use for; where `mpirun` is not on `PATH` the probe dies with rc=127
    and the plan phase is handed an empty survey.
    """
    env = RunEnvironment(kind="session_spool", python="/venv/bin/python", world_size=4, spool="/s")
    command = env.python_command("probe.py", single_process=True)
    assert "mpirun" not in command
    assert "trtllm-llmapi-launch" not in command
    assert "/venv/bin/python probe.py" in command


def test_a_single_process_command_still_scrubs_the_mpi_environment():
    """mpi4py auto-initializes whenever PMI variables are present.

    A spool script runs inside the session's own allocation, so they are,
    and a lone probe would join a world it never asked for.
    """
    env = RunEnvironment(kind="session_spool", python="/p", world_size=4, spool="/s")
    command = env.command("/p probe.py", single_process=True)
    assert "${!PMIX@}" in command
    assert "exec /p probe.py" in command


# --------------------------------------------------------------- PATH merging


def test_a_declared_path_extends_the_inherited_one_rather_than_deleting_it():
    """The line that would otherwise produce `mpirun: command not found`.

    A spec pins its own interpreter by putting its venv first. Written as a
    plain assignment that also deletes `/usr/local/mpi/bin`, and the failure
    arrives as rc=127 inside a launcher line several layers away.
    """
    task = {
        "environment": {
            "kind": "local",
            "python": "/venv/bin/python",
            "env": {"PATH": "/venv/bin:/usr/local/bin:/usr/bin:/bin"},
        }
    }
    prefix = resolve_environment(task, devices=[], world_size=1).prefix
    assert prefix.split()[0] == 'PATH=/venv/bin:/usr/local/bin:/usr/bin:/bin:"$PATH"'


def test_the_declared_entries_still_win_the_lookup():
    """Prepending, not appending: the whole point of declaring a PATH."""
    task = {"environment": {"kind": "local", "python": "/p", "env": {"PATH": "/venv/bin"}}}
    prefix = resolve_environment(task, devices=[], world_size=1).prefix
    assert prefix.index("/venv/bin") < prefix.index("$PATH")


def test_a_path_that_already_names_the_inherited_one_is_not_doubled():
    """Both spellings of the suffix an author may write by hand."""
    for declared in ("/venv/bin:$PATH", "/venv/bin:${PATH}"):
        task = {"environment": {"kind": "local", "python": "/p", "env": {"PATH": declared}}}
        prefix = resolve_environment(task, devices=[], world_size=1).prefix
        assert prefix.split()[0] == 'PATH=/venv/bin:"$PATH"'


def test_other_variables_are_still_replaced_outright():
    """Only PATH decides *which program runs*; nothing else gets the exception."""
    task = {
        "environment": {
            "kind": "local",
            "python": "/p",
            "env": {"TRTLLM_MODELING_V2": "require", "TMPDIR": "/scratch/tmp"},
        }
    }
    prefix = resolve_environment(task, devices=[], world_size=1).prefix
    assert "$" not in prefix


# ------------------------- the orchestrator must outlive the compute session


def _spool_env(tmp_path, kind="session_spool"):
    return RunEnvironment(kind=kind, python="/p", spool=str(tmp_path))


def test_sharing_an_allocation_with_the_watcher_warns(tmp_path, monkeypatch):
    """A session loss takes the orchestrator with it and discards answered work.

    The orchestrator is GPU-free and blocks for hours; the compute session is
    GPU-heavy and dies — preempted, torn down by a worker teardown, or killed
    outright when an agent's `pkill -9` catches the watcher, which IS the
    allocation's main process. Putting the first inside the second couples
    their lifetimes for no benefit at all.
    """
    (tmp_path / "watcher.jobid").write_text("500470\n", encoding="utf-8")
    monkeypatch.setenv("SLURM_JOB_ID", "500470")
    message = environment.shares_allocation_with_spool(_spool_env(tmp_path))
    assert message and "500470" in message
    assert "--gpus-per-node=0" in message


def test_a_different_allocation_is_silent(tmp_path, monkeypatch):
    """The correct configuration must not be warned at, or the warning is noise."""
    (tmp_path / "watcher.jobid").write_text("500470\n", encoding="utf-8")
    monkeypatch.setenv("SLURM_JOB_ID", "500807")
    assert environment.shares_allocation_with_spool(_spool_env(tmp_path)) is None


@pytest.mark.parametrize("contents", ["", "unknown\n"])
def test_an_unknown_watcher_job_is_silent(tmp_path, monkeypatch, contents):
    """Unknown is not a warning. Absent evidence is absent, not damning."""
    (tmp_path / "watcher.jobid").write_text(contents, encoding="utf-8")
    monkeypatch.setenv("SLURM_JOB_ID", "500470")
    assert environment.shares_allocation_with_spool(_spool_env(tmp_path)) is None


def test_a_missing_marker_is_silent(tmp_path, monkeypatch):
    """A spool whose watcher wrote no marker, or is not running yet."""
    monkeypatch.setenv("SLURM_JOB_ID", "500470")
    assert environment.shares_allocation_with_spool(_spool_env(tmp_path)) is None


def test_outside_slurm_and_outside_a_spool_it_is_silent(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    (tmp_path / "watcher.jobid").write_text("500470\n", encoding="utf-8")
    assert environment.shares_allocation_with_spool(_spool_env(tmp_path)) is None

    monkeypatch.setenv("SLURM_JOB_ID", "500470")
    assert environment.shares_allocation_with_spool(_spool_env(tmp_path, kind="local")) is None


def test_the_watcher_writes_the_marker_as_its_own_file(tmp_path):
    """`watcher.jobid` is the evidence, and it must not share the heartbeat line.

    Submitters parse `watcher.heartbeat` as `alive=<timestamp>`, so anything
    extra on that line breaks every one of them. A check nothing writes checks
    nothing, so this pins both halves: the marker is written, and it is written
    somewhere else.
    """
    from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import Scheduler

    Scheduler(tmp_path, environ={"SLURM_JOB_ID": "500470"}).heartbeat()
    assert (tmp_path / "watcher.jobid").read_text(encoding="utf-8").strip() == "500470"
    beat = (tmp_path / "watcher.heartbeat").read_text(encoding="utf-8").strip()
    assert beat.startswith("alive=") and "500470" not in beat
    assert Scheduler(tmp_path / "other", environ={}).slurm_job == "unknown"


def test_a_job_the_spool_started_is_recognised_as_inside_the_session(tmp_path, monkeypatch):
    """The scheduler scrubs SLURM_* from its jobs and hands the job id over by name."""
    (tmp_path / "watcher.jobid").write_text("500470\n", encoding="utf-8")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("MV2FLOW_SPOOL_SESSION_JOBID", "500470")
    assert "500470" in (environment.shares_allocation_with_spool(_spool_env(tmp_path)) or "")


def test_the_spool_prompt_asks_for_devices_instead_of_picking_them():
    """Two scripts that both take device 0 contend, so the scheduler allocates devices."""
    block = RunEnvironment(kind="session_spool", python="/p", world_size=1, spool="/s").describe()
    assert "--gpus" in block and "spoolctl.py submit" in block
    assert "Never set `CUDA_VISIBLE_DEVICES` yourself" in block
    assert "cancel <name>" in block and "never `kill`" in block
    assert "export CUDA_VISIBLE_DEVICES=0" not in block


# ------------------------------- image import is the parallelism lever


def _img_env(image, kind="session_spool"):
    return RunEnvironment(kind=kind, python="/p", container_image=image, spool="/s")


def test_a_registry_reference_warns_with_the_command_to_fix_it():
    """Every allocation re-imports a 26 GB image: 5-7 minutes, 25-35 under load.

    That cost is what forces work onto one shared session, and a shared session
    has one failure domain -- a leaked server, a process-group kill or a PMIx
    collision takes down everything else.
    """
    msg = environment.image_import_warning(_img_env("registry.example/ns/trtllm:tag"))
    assert msg and "enroot import" in msg
    assert "5-7 minutes" in msg
    assert "25-35 minutes" in msg
    assert "trtllm:tag" in msg


def test_an_existing_squashfs_is_silent(tmp_path):
    """A correctly configured run must not be warned at."""
    sqsh = tmp_path / "trtllm.sqsh"
    sqsh.write_bytes(b"x")
    assert environment.image_import_warning(_img_env(str(sqsh))) is None


def test_a_squashfs_path_that_does_not_exist_is_a_different_warning(tmp_path):
    """Pyxis fails to start rather than falling back to a registry."""
    msg = environment.image_import_warning(_img_env(str(tmp_path / "absent.sqsh")))
    assert msg and "does not exist" in msg
    assert "enroot import" not in msg


def test_a_local_run_needs_no_container_and_is_silent():
    assert environment.image_import_warning(_img_env("registry/x:y", kind="local")) is None
    assert environment.image_import_warning(_img_env("")) is None


def test_a_one_rank_command_runs_its_worker_in_process_and_says_so_in_the_prompt():
    """At one rank nothing creates the worker; spawning it is forbidden under pyxis."""
    one = resolve_environment(_LOCAL_TASK, devices=[0], world_size=1)
    assert "TLLM_WORKER_USE_SINGLE_PROCESS=1" in one.command("engine")
    assert "TLLM_WORKER_USE_SINGLE_PROCESS=1" in one.describe()
    many = resolve_environment(_LOCAL_TASK, devices=[0, 1], world_size=2)
    assert "TLLM_WORKER_USE_SINGLE_PROCESS" not in many.command("engine")
    assert "TLLM_WORKER_USE_SINGLE_PROCESS" not in many.describe()
    assert "TLLM_WORKER_USE_SINGLE_PROCESS=1" in replace(many, world_size=1).prefix
    kept = replace(one, env={**one.env, "TLLM_WORKER_USE_SINGLE_PROCESS": "0"})
    assert "TLLM_WORKER_USE_SINGLE_PROCESS=0" in kept.prefix
