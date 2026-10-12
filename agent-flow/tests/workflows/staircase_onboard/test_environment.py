# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the resolved launch environment.

This is the module that replaces four hops of agent-side environment
discovery with one string, so the tests care about exactly one thing: that
the string is right, and that the prompt block quotes the same string the
orchestrator will run.
"""

from __future__ import annotations

import pytest

from agent_flow.workflows.staircase_onboard import environment
from agent_flow.workflows.staircase_onboard.environment import (
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

_SLURM_TASK = {
    "environment": {
        "kind": "slurm",
        "python": "/opt/venv/bin/python",
        "env": {"TRTLLM_MODELING_V2": "require"},
        "partition": "batch",
        "account": "coreai",
        "container_image": "/img/trtllm.sqsh",
        "mounts": ["/lustre:/lustre", "/scratch:/scratch"],
        "time_limit": "04:00:00",
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


def test_slurm_prefix_is_an_srun_line():
    env = resolve_environment(_SLURM_TASK, devices=[], world_size=4)
    prefix = env.prefix
    assert prefix.startswith("srun ")
    assert "--partition=batch" in prefix
    assert "--account=coreai" in prefix
    assert "--time=04:00:00" in prefix
    assert "--gres=gpu:4" in prefix
    assert "--container-image=/img/trtllm.sqsh" in prefix
    assert "--container-mounts=/lustre:/lustre,/scratch:/scratch" in prefix
    assert "--export=ALL,TRTLLM_MODELING_V2=require" in prefix


def test_slurm_prefix_never_sets_cuda_visible_devices():
    """Under Slurm the allocation is the claim; masking it hides our own devices."""
    env = resolve_environment(_SLURM_TASK, devices=[0, 1, 2, 3], world_size=4)
    assert "CUDA_VISIBLE_DEVICES" not in env.prefix


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


def test_describe_tells_slurm_runs_the_devices_are_allocated():
    env = resolve_environment(_SLURM_TASK, devices=[], world_size=4)
    assert "allocated by Slurm" in env.describe()


def test_world_size_is_at_least_one():
    env = resolve_environment(_LOCAL_TASK, devices=[], world_size=0)
    assert env.world_size == 1


def test_environment_is_frozen():
    env = RunEnvironment(kind="local", python="/p")
    try:
        env.kind = "slurm"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("RunEnvironment must be immutable once resolved")


# ------------------------------------------------- the multi-rank launch rule

# These pin TensorRT-LLM's own launch rules, carried from
# `.claude/skills/trtllm-test-script-builder/`. They assert exact strings on
# purpose: the whole failure mode is a plausible-looking srun line that is
# missing one flag, and a loose assertion would pass on exactly that.


def _slurm(world_size, gpus_per_node=4, **over):
    kwargs = dict(
        kind="slurm",
        python="/venv/bin/python",
        world_size=world_size,
        gpus_per_node=gpus_per_node,
        partition="batch",
        account="acct",
        container_image="/img.sqsh",
        mounts=("/lustre:/lustre",),
        time_limit="04:00:00",
        env={"TRTLLM_MODELING_V2": "require"},
    )
    kwargs.update(over)
    return RunEnvironment(**kwargs)


def test_one_task_per_rank_is_emitted():
    """The flag whose absence causes MPI_ERR_SPAWN.

    Without `--ntasks`, srun launches one task, the MPI world size is 1, and
    `need_spawn_mpi_workers` (`llmapi/mpi_session.py`) is
    `get_mpi_world_size() == 1 and model_world_size > 1` — true — so the LLM
    API tries to create the other ranks dynamically, which pyxis forbids.
    """
    command = _slurm(4).command("python x.py")
    assert "--ntasks=4" in command
    assert "--ntasks-per-node=4" in command
    assert "--nodes=1" in command


def test_gres_is_per_node_not_per_world():
    """They differ the moment a run spans nodes, and `--gres` is per node."""
    env = _slurm(8, gpus_per_node=4)
    assert env.nodes == 2
    command = env.command("python x.py")
    assert "--gres=gpu:4" in command
    assert "--gres=gpu:8" not in command
    assert "--ntasks=8" in command


def test_multi_rank_wraps_the_payload_in_the_launcher():
    """`trtllm-llmapi-launch` makes rank 0 the leader and the rest workers."""
    command = _slurm(4).command("/venv/bin/python x.py")
    assert "bash -c 'trtllm-llmapi-launch /venv/bin/python x.py'" in command
    assert "--mpi=pmix" in command


def test_a_single_task_scrubs_mpi_instead_of_wrapping():
    """One task is the special case, and for a specific reason.

    mpi4py auto-initializes MPI whenever PMI variables are present, and inside
    a PMIx-managed step they always are — so a single-task job initializes MPI
    for no reason and then dies spawning a worker. Clearing the variables lets
    it skip MPI init entirely.
    """
    command = _slurm(1).command("/venv/bin/python x.py")
    assert "trtllm-llmapi-launch" not in command
    assert "--mpi=" not in command  # `none` is expressed by omitting the flag
    assert "${!PMIX@}" in command
    assert "exec /venv/bin/python x.py" in command


def test_the_scrub_keeps_the_rank_identifiers():
    """They identify the rank rather than bootstrap MPI, and are read later."""
    command = _slurm(1).command("python x.py")
    assert "SLURM_PROCID" in command
    assert "SLURM_LOCALID" in command


def test_an_explicit_pmix_variant_is_honoured():
    """Clusters report `pmix`, `pmix_v3`, `pmix_v5`; `srun --mpi=list` says."""
    assert "--mpi=pmix_v5" in _slurm(4, mpi="pmix_v5").command("python x.py")


def test_a_container_name_is_emitted_so_pyxis_can_reuse_the_import():
    """A 26 GB image costs 4-8 minutes per job without it."""
    assert "--container-name=trtllm-run" in _slurm(4, container_name="trtllm-run").command("x")
    assert "--container-name" not in _slurm(4).command("x")


def test_the_emitted_line_matches_the_deployment_guide_shape():
    """Checked against `docs/source/deployment-guide/` two-node example."""
    command = _slurm(8, gpus_per_node=4).command("python3 -m tensorrt_llm.commands.serve M")
    for fragment in (
        "--nodes=2",
        "--ntasks=8",
        "--ntasks-per-node=4",
        "--mpi=pmix",
        "--gres=gpu:4",
        "trtllm-llmapi-launch python3 -m tensorrt_llm.commands.serve M",
    ):
        assert fragment in command, fragment


def test_spool_and_local_apply_the_same_rule_through_mpirun():
    """There is no srun inside a spool script, but the engine rule is the same."""
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
# The `slurm` branch must keep neither workaround.


def test_mpirun_oversubscribes_because_the_allocation_is_one_task():
    """`-n 4` against a one-task allocation dies before any Python runs.

    Open MPI takes its slot count from the resource manager and refuses with
    *"There are not enough slots available in the system"*. The allocation
    holds the whole node, so the slot count is bookkeeping.
    """
    env = RunEnvironment(kind="session_spool", python="/venv/bin/python", world_size=4, spool="/s")
    assert "mpirun -n 4 --oversubscribe --allow-run-as-root" in env.command("/venv/bin/python x.py")


def test_the_slurm_branch_never_oversubscribes():
    """There a slot shortage is a mis-sized allocation reporting itself.

    `srun --ntasks` genuinely sizes the world under `slurm`, so the flag
    would hide the one failure that is worth seeing.
    """
    assert "--oversubscribe" not in _slurm(4).command("/venv/bin/python x.py")
    assert "--oversubscribe" not in _slurm(8, gpus_per_node=4).command("python3 -m x")


def test_mpirun_scrubs_the_slurm_task_vars_the_launcher_reads_first():
    """Otherwise every rank is rank 0 of a world of 1, and the run hangs.

    `trtllm-llmapi-launch` reads `SLURM_PROCID` before
    `OMPI_COMM_WORLD_RANK` and `SLURM_NTASKS` before `OMPI_COMM_WORLD_SIZE`.
    Inside a one-task allocation those are `0` and `1` for every rank, so all
    four became `mgmn_leader_node` and waited forever for workers.
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


def test_the_slurm_branch_never_scrubs_the_task_vars():
    """Under `srun` the same scrub breaks a launch that is already correct.

    `SLURM_PROCID` really is the rank there and `SLURM_NTASKS` really is the
    world size; the one-task branch's own scrub deliberately spares them.
    """
    for command in (_slurm(4).command("/venv/bin/python x.py"), _slurm(1).command("/p x.py")):
        assert "env -u SLURM_PROCID" not in command
        assert "-u SLURM_NTASKS" not in command


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
    launch rules at world size 4 it became
    `mpirun -n 4 ... trtllm-llmapi-launch`, which needs an MPI the run's
    `PATH` had deleted -- so the probe died with rc=127 and the plan phase
    was handed an empty survey.
    """
    env = RunEnvironment(kind="session_spool", python="/venv/bin/python", world_size=4, spool="/s")
    command = env.python_command("probe.py", single_process=True)
    assert "mpirun" not in command
    assert "trtllm-llmapi-launch" not in command
    assert "/venv/bin/python probe.py" in command


def test_a_single_process_command_still_scrubs_the_mpi_environment():
    """mpi4py auto-initializes whenever PMI variables are present.

    A spool script runs inside the session's own allocation, so they are,
    and a lone probe that joins a world it never asked for is the same
    failure the one-task Slurm branch exists to prevent.
    """
    env = RunEnvironment(kind="session_spool", python="/p", world_size=4, spool="/s")
    command = env.command("/p probe.py", single_process=True)
    assert "${!PMIX@}" in command
    assert "exec /p probe.py" in command


def test_a_single_process_slurm_command_sizes_the_step_for_one_task():
    """Shrinking the payload without shrinking the step runs N copies of it."""
    command = _slurm(4).python_command("probe.py", single_process=True)
    assert "--ntasks=1" in command
    assert "--ntasks-per-node=1" in command
    assert "--gres=gpu:1" in command
    assert "--mpi=" not in command
    assert "trtllm-llmapi-launch" not in command
    assert "${!PMIX@}" in command


def test_engine_work_is_untouched_by_the_new_distinction():
    """The gates genuinely do want a rank per device; only the probe does not."""
    command = _slurm(4).python_command("x.py")
    assert "--ntasks=4" in command
    assert "trtllm-llmapi-launch /venv/bin/python x.py" in command


# --------------------------------------------------------------- PATH merging


def test_a_declared_path_extends_the_inherited_one_rather_than_deleting_it():
    """The line that produced `mpirun: command not found`.

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
    assert prefix == 'PATH=/venv/bin:/usr/local/bin:/usr/bin:/bin:"$PATH"'


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
        assert prefix == 'PATH=/venv/bin:"$PATH"'


def test_the_merge_reaches_the_slurm_export_list_too():
    """`--export` carries the same assignments and the same deletion."""
    command = _slurm(4, env={"PATH": "/venv/bin"}).command("python x.py")
    assert '--export=ALL,PATH=/venv/bin:"$PATH"' in command


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
    """Five session losses across two runs, each discarding answered work.

    The orchestrator is GPU-free and blocks for hours; the compute session is
    GPU-heavy and dies — preempted, torn down by a worker teardown, or killed
    outright when an agent's `pkill -9` catches the watcher, which IS the
    allocation's main process. Putting the first inside the second couples
    their lifetimes for no benefit at all.
    """
    (tmp_path / "watcher.jobid").write_text("559470\n", encoding="utf-8")
    monkeypatch.setenv("SLURM_JOB_ID", "559470")
    message = environment.shares_allocation_with_spool(_spool_env(tmp_path))
    assert message and "559470" in message
    assert "--gpus-per-node=0" in message


def test_a_different_allocation_is_silent(tmp_path, monkeypatch):
    """The correct configuration must not be warned at, or the warning is noise."""
    (tmp_path / "watcher.jobid").write_text("559470\n", encoding="utf-8")
    monkeypatch.setenv("SLURM_JOB_ID", "561807")
    assert environment.shares_allocation_with_spool(_spool_env(tmp_path)) is None


@pytest.mark.parametrize("contents", ["", "unknown\n"])
def test_an_unknown_watcher_job_is_silent(tmp_path, monkeypatch, contents):
    """Unknown is not a warning. Absent evidence is absent, not damning."""
    (tmp_path / "watcher.jobid").write_text(contents, encoding="utf-8")
    monkeypatch.setenv("SLURM_JOB_ID", "559470")
    assert environment.shares_allocation_with_spool(_spool_env(tmp_path)) is None


def test_a_missing_marker_is_silent(tmp_path, monkeypatch):
    """A spool whose watcher predates this file, or is not running yet."""
    monkeypatch.setenv("SLURM_JOB_ID", "559470")
    assert environment.shares_allocation_with_spool(_spool_env(tmp_path)) is None


def test_outside_slurm_and_outside_a_spool_it_is_silent(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    (tmp_path / "watcher.jobid").write_text("559470\n", encoding="utf-8")
    assert environment.shares_allocation_with_spool(_spool_env(tmp_path)) is None

    monkeypatch.setenv("SLURM_JOB_ID", "559470")
    assert environment.shares_allocation_with_spool(_spool_env(tmp_path, kind="slurm")) is None


def test_the_watcher_writes_the_marker_as_its_own_file():
    """`watcher.jobid` is the evidence, and it must not share the heartbeat line.

    `scripts/spool_submit.sh` pipes `watcher.heartbeat` straight into `date -d`
    after stripping `alive=`, so anything extra on that line breaks every
    submitter. A check nothing writes checks nothing, so this pins both halves:
    the marker is written, and it is written somewhere else.
    """
    from pathlib import Path

    watcher = Path(environment.__file__).with_name("spool_watcher.sh").read_text(encoding="utf-8")
    assert '"$SPOOL/watcher.jobid"' in watcher
    assert "${SLURM_JOB_ID:-unknown}" in watcher
    for line in watcher.splitlines():
        if "watcher.heartbeat" in line and "echo" in line:
            assert "SLURM_JOB_ID" not in line


# ------------------------------- per-workload substrate


_WORKLOAD_BASE = {
    "environment": {
        "kind": "session_spool",
        "python": "/p",
        "spool": "/s",
        "env": {"A": "1", "B": "2"},
        "partition": "batch",
    }
}


def test_without_an_override_nothing_changes():
    """Absent a `workloads` block, asking as a gate resolves to the base."""
    plain = resolve_environment(_WORKLOAD_BASE)
    gated = resolve_environment(_WORKLOAD_BASE, workload="gate")
    assert plain == gated


def test_a_gate_override_replaces_scalars_and_merges_env():
    """Short work on the shared session; a 10-15 minute gate in its own job."""
    spec = {
        "environment": {
            **_WORKLOAD_BASE["environment"],
            "workloads": {"gate": {"kind": "slurm", "env": {"B": "override", "C": "3"}}},
        }
    }
    gate = resolve_environment(spec, workload="gate")
    assert gate.kind == "slurm"
    # Scalars the override does not mention are inherited, so a gate that only
    # needs a different substrate does not restate mounts and accounts.
    assert gate.partition == "batch"
    # `env` merges key-by-key rather than replacing wholesale.
    assert gate.env == {"A": "1", "B": "override", "C": "3"}


def test_an_override_for_another_workload_does_not_leak():
    """A gate override must not change the substrate of everything else."""
    spec = {
        "environment": {
            **_WORKLOAD_BASE["environment"],
            "workloads": {"gate": {"kind": "slurm"}},
        }
    }
    assert resolve_environment(spec).kind == "session_spool"
    assert resolve_environment(spec, workload=None).kind == "session_spool"


def test_the_workload_set_is_closed():
    """An override the orchestrator never dispatches under is a typo.

    It would silently never apply, and nothing downstream would say so —
    `task_schema._validate_workloads` rejects it for exactly that reason.
    """
    assert environment.WORKLOAD_KINDS == ("gate",)


# ------------------------------- image import is the parallelism lever


def _img_env(image, kind="session_spool"):
    return RunEnvironment(kind=kind, python="/p", container_image=image, spool="/s")


def test_a_registry_reference_warns_with_the_command_to_fix_it():
    """Every allocation re-imports a 26 GB image: 5-7 minutes, 25-35 under load.

    That cost is what forces work onto one shared session, and a shared session
    has one failure domain -- which is how a leaked server, a process-group
    kill and a PMIx collision each took down everything else in one run.
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


def test_every_container_bearing_kind_is_checked():
    """`slurm`, `slurm_broker` and `session_spool` all hand the image to pyxis."""
    for kind in ("slurm", "slurm_broker", "session_spool"):
        assert environment.image_import_warning(_img_env("registry/x:y", kind=kind))
