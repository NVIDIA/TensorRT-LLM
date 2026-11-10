# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running on the orchestrator's own host: commands, the launch environment, the CLI, the monitor."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import time
from unittest import mock

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2 import cli
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import executor, server_runtime
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import (
    TIMEOUT_RETURNCODE,
    CommandSpec,
    LocalCommandExecutor,
    ended_early,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import (
    MULTI_TASK_SLURM_SCRUB,
    RunEnvironment,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.files import write_json_atomic
from agent_flow.workflows.trtllm_modeling_v2.infra.monitor import status
from agent_flow.workflows.trtllm_modeling_v2.infra.monitor.heartbeat import (
    ORCHESTRATOR_EXIT,
    ORCHESTRATOR_HEARTBEAT,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.monitor.status import load_config
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
    assert "Use `kind: local`" in said and "inside the allocation" in said


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
    assert command.index(MULTI_TASK_SLURM_SCRUB) < command.index("mpirun")
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
    assert f"```\n{env.prefix}\n```" in env.describe()


def test_hung_gpu_inspection_returns_unknown_instead_of_blocking_teardown(monkeypatch):
    def hung(argv, **kwargs):
        assert kwargs["timeout"] == 30
        raise subprocess.TimeoutExpired(argv, 30)

    monkeypatch.setattr(server_runtime.subprocess, "run", hung)
    assert server_runtime.device_memory_used_mib() is None
    assert server_runtime.device_holders() == set()


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
    assert status.orchestrator_state(load_config(tmp_path))[0] == "running"
    assert monitor(["wait", "--timeout", "0"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "TIMEOUT  nothing new"
    # A closed run changes nothing else, so the waiter must notice the exit.
    write_json_atomic(dispatch / ORCHESTRATOR_EXIT, {"pid": 1, "host": "h", "at": time.time() + 1})
    assert monitor(["--workspace", str(tmp_path), "wait", "--timeout", "0"]) == 0
    assert capsys.readouterr().out.splitlines()[0].startswith("ORCHESTRATOR_EXITED  pid 1")


def test_monitor_status_says_where_the_run_stands(tmp_path, capsys):
    (tmp_path / "task.yaml").write_text("{}\n", encoding="utf-8")
    (tmp_path / ".trtllm_modeling_v2_state.json").write_text(
        json.dumps({"version": 3, "stage": "execution", "done": False})
    )
    tasks = [
        {"id": "survey", "status": "done"},
        {"id": "core", "status": "running"},
        {"id": "mtp", "status": "pending"},
    ]
    (tmp_path / "tasks.yaml").write_text(yaml.safe_dump({"tasks": tasks}))
    (tmp_path / "certification.yaml").write_text(
        yaml.safe_dump({"certified": False, "round": 1, "commit": "c0ffee1234567890"})
    )
    assert cli.SUBCOMMANDS["monitor"](["--workspace", str(tmp_path), "status"]) == 0
    line = capsys.readouterr().out.splitlines()[1]
    assert line == (
        "run: stage execution; tasks 1 done, 1 pending, 1 running; running core; "
        "certification round 1: not certified c0ffee123456"
    )


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
    assert run("--task", str(task_file), "--workspace", str(workspace), "--dry-run") == 0
    assert "--dry-run: preflight passed" in capsys.readouterr().out
    assert (workspace / "contract.yaml").is_file() and not (workspace / "tasks.yaml").exists()

    # A spec that states no model for its roles is a spec error: 2, before anything runs.
    spec = yaml.safe_load(task_file.read_text())
    del spec["agents"]
    task_file.write_text(yaml.safe_dump(spec))
    other = tmp_path / "ws-no-agents"
    assert run("--task", str(task_file), "--workspace", str(other), "--dry-run") == 2
    assert "states no model for" in capsys.readouterr().err

    # A spec whose accuracy anchor cannot be read fails preflight: 1, and the journal says why.
    spec["agents"] = {"defaults": {"backend": "claude-code", "model": "claude-test-model"}}
    spec["accuracy"]["model_key"] = "nobody/no-such-model"
    task_file.write_text(yaml.safe_dump(spec))
    failed = tmp_path / "ws-no-anchor"
    assert run("--task", str(task_file), "--workspace", str(failed), "--dry-run") == 1
    assert "preflight failed" in (failed / "journal.md").read_text()
    assert dict(os.environ) == before
