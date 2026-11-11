# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running on the orchestrator's own host: commands, the launch environment, the CLI, the monitor."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import time
from pathlib import Path
from unittest import mock

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2 import cli
from agent_flow.workflows.trtllm_modeling_v2.infra import monitor as monitor_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import executor
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import (
    TIMEOUT_RETURNCODE,
    CommandSpec,
    LocalCommandExecutor,
    ended_early,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.files import write_json_atomic
from agent_flow.workflows.trtllm_modeling_v2.infra.monitor import (
    ORCHESTRATOR_EXIT,
    ORCHESTRATOR_HEARTBEAT,
)
from agent_flow.workflows.trtllm_modeling_v2.run import workflow as workflow_mod
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import (
    TaskSchemaError,
    load_and_validate_task_yaml,
)


def test_a_command_that_outlives_its_deadline_is_timed_out_not_merely_failed():
    started = time.monotonic()
    record = LocalCommandExecutor().execute(CommandSpec(argv="sleep 30", timeout_s=1))
    assert record.timed_out
    assert record.returncode == TIMEOUT_RETURNCODE
    assert not record.ok
    assert time.monotonic() - started < 20
    # A verdict names a stall by its cause, not by a time limit the job never reached.
    assert ended_early(record, 1) == "timed out after 1s"
    stalled = dataclasses.replace(record, stalled="no progress for 601s after a failure line")
    assert ended_early(stalled, 14400) == "stalled: no progress for 601s after a failure line"


def test_a_local_command_runs_with_the_environment_given_and_says_how_it_ended():
    run = executor.shell_runner_for(
        executor.select_command_executor(RunEnvironment(kind="local", python="/p"))
    )
    done = run("printf '%s' \"$MV2_TEST_VALUE\"; exit 3", {"MV2_TEST_VALUE": "a b"}, 30)
    assert (done.returncode, done.stdout, done.ok, done.timed_out) == (3, "a b", False, False)
    assert run("true", None, 30).ok
    # `local` is the one kind the workflow runs commands for.
    with pytest.raises(ValueError, match="has no executor here"):
        executor.select_command_executor(RunEnvironment(kind="session_spool", python="/p"))


def test_a_spec_written_for_compute_sessions_is_refused_with_what_to_do(task_file):
    spec = yaml.safe_load(task_file.read_text())
    spec["environment"].update(kind="session_spool", spool="/s/spool", session_wait="1h")
    task_file.write_text(yaml.safe_dump(spec))
    with pytest.raises(TaskSchemaError) as refused:
        load_and_validate_task_yaml(task_file)
    said = str(refused.value)
    assert "'environment.spool', 'environment.session_wait'" in said
    assert "Start the orchestrator where the GPUs are" in said and "`launch` prefix" in said


def test_every_command_scrubs_the_allocations_launcher_environment():
    """Otherwise every rank is rank 0 of a world of 1, and the run hangs.

    `trtllm-llmapi-launch` reads `SLURM_PROCID` before
    `OMPI_COMM_WORLD_RANK` and `SLURM_NTASKS` before `OMPI_COMM_WORLD_SIZE`.
    Inside a one-task allocation those are `0` and `1` for every rank, so all
    four become `mgmn_leader_node` and wait forever for workers.

    mpi4py auto-initializes whenever PMI variables are present. An orchestrator
    started inside an allocation has them, and a lone probe would join a world
    it never asked for.
    """
    env = RunEnvironment(kind="local", python="/p", world_size=4, devices=(0, 1, 2, 3))
    command = env.command("/p x.py")
    for name in ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID", "SLURM_NODEID"):
        assert f"-u {name}" in command, name
    assert command.index("env -u SLURM_PROCID ") < command.index("mpirun -n 4 ")
    assert command.startswith("CUDA_VISIBLE_DEVICES=0,1,2,3 ")
    single = env.command("/p probe.py", single_process=True)
    assert "${!PMIX@}" in single
    assert "exec /p probe.py" in single


def test_the_launch_prefix_preserves_shell_quotes():
    env = RunEnvironment(kind="local", python="/p", devices=(0,), env={"EXAMPLE": "a b'$c"})
    found = subprocess.run(
        ["bash", "-c", f"{env.prefix} printenv EXAMPLE"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert found.stdout == "a b'$c\n"


def test_monitor_cli_tells_a_busy_orchestrator_from_one_that_has_exited(
    tmp_path, monkeypatch, capsys
):
    monitor = cli.SUBCOMMANDS["monitor"]
    monkeypatch.delenv("MV2FLOW_WORKSPACE", raising=False)
    assert monitor(["status"]) == 2
    assert "pass --workspace or set MV2FLOW_WORKSPACE" in capsys.readouterr().err
    assert monitor(["--workspace", str(tmp_path / "missing"), "status"]) == 1
    assert "not a trtllm_modeling_v2 workspace" in capsys.readouterr().err

    (tmp_path / "task.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("MV2FLOW_WORKSPACE", str(tmp_path))
    assert monitor(["status", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data == {
        "workspace": str(tmp_path.resolve()),
        "orchestrator": {"state": "unknown", "detail": "no orchestrator heartbeat"},
        "run": None,
    }
    dispatch = tmp_path / "dispatch"
    dispatch.mkdir()
    write_json_atomic(dispatch / ORCHESTRATOR_HEARTBEAT, {"pid": 1, "host": "h", "at": time.time()})
    assert monitor_mod.orchestrator_state(tmp_path)[0] == "running"
    # From any node that sees the workspace: only the files are read, and a heartbeat that
    # stopped says the orchestrator is gone.
    late = time.time() + monitor_mod.ORCHESTRATOR_STALE_S + 1
    state, said = monitor_mod.orchestrator_state(tmp_path, now=late)
    assert state == "exited" and "pid 1 on h stopped heart-beating" in said
    assert monitor(["wait", "--timeout", "0"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "TIMEOUT  nothing new"
    # A closed run changes nothing else, so the waiter must notice the exit.
    write_json_atomic(dispatch / ORCHESTRATOR_EXIT, {"pid": 1, "host": "h", "at": time.time() + 1})
    assert monitor(["--workspace", str(tmp_path), "wait", "--timeout", "0"]) == 0
    assert capsys.readouterr().out.splitlines()[0].startswith("ORCHESTRATOR_EXITED  pid 1")
    # An exit older than the heartbeat is an earlier process's: this one is running.
    write_json_atomic(dispatch / ORCHESTRATOR_EXIT, {"pid": 1, "host": "h", "at": time.time() - 60})
    assert monitor_mod.orchestrator_state(tmp_path)[0] == "running"


def test_monitor_status_says_where_the_run_stands(tmp_path, capsys):
    (tmp_path / "task.yaml").write_text("{}\n", encoding="utf-8")
    (tmp_path / ".trtllm_modeling_v2_state.json").write_text(
        json.dumps({"version": 4, "onboarded": {"commit": "c0ffee1234567890"}})
    )
    tasks = [
        {"id": "survey", "status": "done"},
        {"id": "core", "status": "running"},
        {"id": "mtp", "status": "pending"},
    ]
    (tmp_path / "tasks.yaml").write_text(yaml.safe_dump({"tasks": tasks}))
    assert cli.SUBCOMMANDS["monitor"](["--workspace", str(tmp_path), "status"]) == 0
    line = capsys.readouterr().out.splitlines()[1]
    assert line == (
        "run: onboarded at c0ffee123456; tasks 1 done, 1 pending, 1 running; running core"
    )
    (tmp_path / ".trtllm_modeling_v2_state.json").write_text(json.dumps({"version": 4}))
    assert cli.SUBCOMMANDS["monitor"](["--workspace", str(tmp_path), "status"]) == 0
    assert capsys.readouterr().out.splitlines()[1].startswith("run: not onboarded; tasks ")


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_the_command_line_dry_runs_a_spec_and_its_exit_code_says_how_it_went(
    tmp_path, task_file, repo, monkeypatch, capsys
):
    # The CLI commits as the checkout's own user, and proves it with one trial commit.
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "T")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    for name in ("MV2FLOW_GIT_NAME", "MV2FLOW_GIT_EMAIL"):
        monkeypatch.delenv(name, raising=False)
    # The CLI pins its identity in the process environment and starts a heartbeat thread:
    # neither outlives this test.
    before = dict(os.environ)
    beats: list[str] = []

    class _Heartbeat:
        def mark_exit(self):
            beats.append("exit")

    monkeypatch.setattr(cli, "start_orchestrator_heartbeat", lambda workspace, period: _Heartbeat())

    def run(*argv):
        with mock.patch.dict(os.environ), pytest.raises(SystemExit) as exited:
            cli.main(list(argv))
        return exited.value.code

    assert run("--help") == 0
    assert "--dry-run" in capsys.readouterr().out
    workspace = tmp_path / "ws"
    argv = ["--task", str(task_file), "--workspace", str(workspace), "--dry-run"]
    assert run(*argv, "--max-rounds", "9") == 0
    said = capsys.readouterr()
    assert "--dry-run: preflight passed" in said.out
    assert "note: --max-rounds is ignored: " in said.err
    assert (workspace / "anchors.yaml").is_file() and not (workspace / "tasks.yaml").exists()

    # A spec that states no model for its roles is a spec error: 2, before anything runs.
    spec = yaml.safe_load(task_file.read_text())
    del spec["agents"]
    task_file.write_text(yaml.safe_dump(spec))
    other = tmp_path / "ws-no-agents"
    assert run("--task", str(task_file), "--workspace", str(other), "--dry-run") == 2
    assert "missing agents.defaults.model: " in capsys.readouterr().err

    # A refused spec still says which of its keys were ignored on the way.
    spec["profile"], spec["feature"] = "smoke", ["mtp"]
    task_file.write_text(yaml.safe_dump(spec))
    assert run("--task", str(task_file), "--workspace", str(other), "--dry-run") == 2
    said = capsys.readouterr().err
    assert "note: 'profile' is ignored: " in said and "unknown key 'feature'" in said
    del spec["profile"], spec["feature"]

    # A checkpoint that cannot be read fails preflight: 1, and the journal says why.
    spec["agents"] = {"defaults": {"backend": "claude-code", "model": "claude-test-model"}}
    (Path(spec["checkpoint_path"]) / "config.json").write_text("{not json")
    task_file.write_text(yaml.safe_dump(spec))
    failed = tmp_path / "ws-no-anchor"
    assert run("--task", str(task_file), "--workspace", str(failed), "--dry-run") == 1
    journal = (failed / "journal.md").read_text()
    assert "preflight failed" in journal and "the checkpoint is not readable" in journal
    # A workspace an earlier version of the flow ran in is refused in one line, with what to do.
    old = tmp_path / "ws-old"
    old.mkdir()
    (old / ".trtllm_modeling_v2_state.json").write_text('{"version": 3, "stage": "execution"}')
    assert run("--task", str(task_file), "--workspace", str(old)) == 2
    said = capsys.readouterr().err
    assert "error: " in said and "start over with --clean, or take another --workspace" in said
    # The exit status says what the run came to: 0 onboarded, whatever else is not done.
    for onboarded, code in ((True, 0), (False, 1)):
        came_to = lambda self, task, onboarded=onboarded: onboarded  # noqa: E731
        monkeypatch.setattr(workflow_mod.TrtllmModelingV2Workflow, "run", came_to)
        workspace = tmp_path / f"ws-run-{code}"  # each start holds its workspace's lock
        assert run("--task", str(task_file), "--workspace", str(workspace)) == code
    assert dict(os.environ) == before
