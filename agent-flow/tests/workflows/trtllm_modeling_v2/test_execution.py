# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the orchestrator's turns and commands actually go.

Two things are being protected here and they are different in kind.

The first is that the mapping from a ``task.yaml`` environment kind to a
mechanism lives in one place, outside the code that owns resume and gate
sequencing, so adding a mechanism does not mean editing the state machine.

The second is that the abstraction is *load-bearing* rather than decorative.
An interface nothing calls is worse than no interface, because it reads as
structure. So several of these assert the real call path end to end: a run
configured for a spool really does put its gate command into that spool's
`pending` directory, through `CommandExecutor`, with no branch in the
workflow.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gates as gates_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import (
    CommandResult,
    CommandSpec,
    InProcessTurnExecutor,
    LocalCommandExecutor,
    SpoolCommandExecutor,
    TurnRequest,
    select_command_executor,
    select_turn_executor,
    shell_runner_for,
    wait_for_session,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.transport import ExternalAgentTransport
from agent_flow.workflows.trtllm_modeling_v2.onboard.survey import MARKER, Inventory
from agent_flow.workflows.trtllm_modeling_v2.run import workflow as workflow_mod
from agent_flow.workflows.trtllm_modeling_v2.run.stages import preflight as preflight_stage
from agent_flow.workflows.trtllm_modeling_v2.run.state import SCHEMA_VERSION, load_state
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

from .test_workflow import Script, _git, _install, healthy_smi

# ------------------------------------------------------- executor selection


def _environment(kind: str, **kwargs) -> RunEnvironment:
    return RunEnvironment(kind=kind, python="/opt/venv/bin/python", **kwargs)


def test_local_resolves_to_a_subprocess_here():
    assert isinstance(select_command_executor(_environment("local")), LocalCommandExecutor)


def test_a_session_spool_resolves_to_the_spool_it_names(tmp_path):
    executor = select_command_executor(_environment("session_spool", spool=str(tmp_path)))
    assert isinstance(executor, SpoolCommandExecutor)
    assert executor.root == tmp_path


def test_a_session_spool_with_no_spool_falls_back_rather_than_crashing():
    """The schema already refuses this; resolution must not be the check.

    Two places enforcing one rule means the second one eventually disagrees
    with the first.
    """
    assert isinstance(select_command_executor(_environment("session_spool")), LocalCommandExecutor)


# ------------------------------------------------- the shell runner adapters


def test_the_shell_adapter_produces_the_shape_the_gates_take():
    run = shell_runner_for(LocalCommandExecutor())
    result = run("echo hi", None, 30)
    assert isinstance(result, CommandResult)
    assert result.ok and result.stdout.strip() == "hi"


def test_the_shell_adapter_keeps_a_timeout_distinct_from_a_failure():
    run = shell_runner_for(LocalCommandExecutor())
    assert run("exit 5", None, 30).returncode == 5
    timed = run("sleep 30", None, 1)
    assert timed.timed_out and not timed.ok


def test_the_shell_adapter_passes_the_environment_through():
    run = shell_runner_for(LocalCommandExecutor())
    assert run("echo $MV2FLOW_ADAPTER", {"MV2FLOW_ADAPTER": "yes"}, 30).stdout.strip() == "yes"


def test_an_argv_is_executed_without_a_shell():
    result = LocalCommandExecutor().execute(CommandSpec(argv=("echo", "a b"), timeout_s=30))
    assert result.stdout == "a b\n"


def test_a_missing_binary_is_127_not_a_failure_of_the_probe():
    spec = CommandSpec(argv=("mv2flow-no-such-binary-xyz",), timeout_s=30)
    assert LocalCommandExecutor().execute(spec).returncode == 127


def test_the_gate_default_runner_goes_through_the_local_executor(monkeypatch):
    """There is one implementation of "run a command here"."""
    seen: list[CommandSpec] = []
    real = LocalCommandExecutor.execute

    def spy(self, spec):
        seen.append(spec)
        return real(self, spec)

    monkeypatch.setattr(LocalCommandExecutor, "execute", spy)
    gates_mod._default_shell_runner("echo hi", {"A": "b"}, 42)

    assert len(seen) == 1
    assert seen[0].argv == "echo hi"
    assert seen[0].env == {"A": "b"}
    assert seen[0].timeout_s == 42


# -------------------------------------------------------- turn executor choice


def test_no_transport_means_this_process_calls_the_layer():
    executor = select_turn_executor(None, lambda request: (lambda prompt: "hi"))
    assert isinstance(executor, InProcessTurnExecutor)
    assert executor.run(TurnRequest(role="smith", what="x", prompt="p")).text == "hi"


def test_a_turn_executor_is_used_as_given(tmp_path):
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    assert select_turn_executor(transport, lambda request: None) is transport


# ------------------------------------- the orchestrator does not know the names


def _orchestrator_source() -> str:
    """`workflow.py` and its stages: the code that sequences a run."""
    from agent_flow.workflows.trtllm_modeling_v2.run.stages import (
        assemble,
        common,
        deliver,
        plan,
        preflight,
        smith,
    )

    modules = (workflow_mod, common, preflight, plan, smith, assemble, deliver)
    return "\n".join(inspect.getsource(module) for module in modules)


def test_the_orchestrator_neither_names_nor_builds_a_platform_mechanism():
    """A mechanism should be removable without reading the state machine."""
    source = _orchestrator_source()
    for name in (
        "session_spool",
        "SpoolCommandExecutor",
        "LocalCommandExecutor",
        "srun",
    ):
        assert name not in source, f"the orchestrator names the mechanism {name!r}"
    assert "environment.kind" not in source
    # A task's kind is the graph's; a mechanism's kind is not the orchestrator's to compare.
    for kind in ("local", "session_spool", "slurm"):
        assert f'.kind == "{kind}"' not in source and f'kind != "{kind}"' not in source


def test_the_orchestrator_never_reads_the_spool_directly():
    """Where gate output lives is the executor's business, not the stages'."""
    assert '"spool"' not in _orchestrator_source()


def test_run_turn_has_no_transport_branch():
    """Both transports answer `TurnExecutor.run`, so stage sequencing cannot tell them apart."""
    source = inspect.getsource(TrtllmModelingV2Workflow._run_turn)
    assert "_agent_transport" not in source
    assert "TurnRequest(" in source
    assert ".run(request)" in source


def test_the_workflow_resolves_its_turn_executor_once(tmp_path):
    workflow = TrtllmModelingV2Workflow(workspace=tmp_path / "ws")
    assert isinstance(workflow._turns, InProcessTurnExecutor)

    transport = ExternalAgentTransport(tmp_path / "ws2", poll_s=0.01)
    assert (
        TrtllmModelingV2Workflow(workspace=tmp_path / "ws2", agent_transport=transport)._turns
        is transport
    )


def test_agent_turn_timeout_bounds_in_process_turns_for_every_role(tmp_path):
    bounded = TrtllmModelingV2Workflow(workspace=tmp_path / "ws", agent_turn_timeout=900)
    assert bounded._turns.timeout_s == 900 and bounded._turns.role_timeouts == {}

    default = TrtllmModelingV2Workflow(workspace=tmp_path / "ws2")
    assert default._turns.role_timeouts


# ------------------------------------------------- the real call path, wired


@pytest.fixture
def spool_task_file(tmp_path, repo, checkpoint):  # noqa: F811 -- fixtures by name
    spool = tmp_path / "spool"
    spool.mkdir()
    spec = {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "parallel": "tp1",
        "trtllm_repo_path": str(repo),
        "target": {"family": "gpt_oss", "checkpoint_segment": "gpt_oss_120b"},
        "environment": {
            "kind": "session_spool",
            "python": "/opt/venv/bin/python",
            "spool": str(spool),
            "env": {
                "TMPDIR": "/scratch/mv2flow-run-tmp",
                "CLAUDE_CODE_TMPDIR": "/scratch/mv2flow-run-tmp",
            },
        },
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml",
            "model_key": "openai/gpt-oss-120b",
            "tolerance": 5.0,
        },
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


def test_a_spool_run_puts_its_commands_in_the_spool(tmp_path, spool_task_file, monkeypatch):
    """End to end: preflight resolves the executor, and gates use it.

    `--dry-run` stops before the first agent turn, which is exactly far
    enough to have resolved the executor. It does still take the inventory
    survey, and that survey goes through the same spool — so it is stubbed
    out here rather than left to block for its full deadline on a spool
    nothing is watching. What this test is about is the mapping from an
    environment kind to a mechanism, not the survey.
    """
    monkeypatch.setattr(preflight_stage, "run_inventory_probe", lambda *a, **k: Inventory())
    workflow = TrtllmModelingV2Workflow(
        workspace=tmp_path / "ws",
        command_runner=healthy_smi(),
        dry_run=True,
    )
    _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(spool_task_file)

    assert isinstance(workflow._command_executor, SpoolCommandExecutor)
    assert workflow._command_executor.root == tmp_path / "spool"

    # And the gate runner really goes there: nothing is watching, so it
    # times out, but the work landed as a file.
    result = workflow._shell_runner("echo gate-command", None, 0)
    assert result.timed_out
    scripts = sorted((tmp_path / "spool" / "pending").glob("*.sh"))
    assert scripts and "echo gate-command" in scripts[-1].read_text(encoding="utf-8")


def test_the_checkout_identity_is_never_a_spool_job(tmp_path, spool_task_file, repo, monkeypatch):
    """The resume marker is read here, on a spool run too.

    It would otherwise be a job queued behind the gates, holding the node to
    answer a question about a checkout this process can already read.
    """
    _git(repo, "init", "-q", ".")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "target")
    monkeypatch.setattr(preflight_stage, "run_inventory_probe", lambda *a, **k: Inventory())
    workflow = TrtllmModelingV2Workflow(
        workspace=tmp_path / "ws",
        command_runner=healthy_smi(),
        dry_run=True,
    )
    _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(spool_task_file)
    assert isinstance(workflow._command_executor, SpoolCommandExecutor)

    pending = tmp_path / "spool" / "pending"
    queued = sorted(pending.glob("*.sh"))
    assert workflow._target_marker().startswith("tree:")
    assert sorted(pending.glob("*.sh")) == queued


def test_a_command_lost_to_its_session_is_noted_and_waited_on_through_the_executor(
    tmp_path, spool_task_file, monkeypatch
):
    """The resolved runner tells the loop when a session, not a command, ended it."""
    monkeypatch.setattr(preflight_stage, "run_inventory_probe", lambda *a, **k: Inventory())
    workflow = TrtllmModelingV2Workflow(
        workspace=tmp_path / "ws", command_runner=healthy_smi(), dry_run=True
    )
    _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(spool_task_file)

    # Nothing has ever watched this spool.
    result = workflow._shell_runner("echo gate-command", None, 0)
    assert "no session watches" in result.infrastructure
    assert workflow._lost_commands == [result.infrastructure]
    assert wait_for_session(workflow._command_executor, 0.1) is False
    assert wait_for_session(LocalCommandExecutor(), 3600) is False


def test_an_injected_runner_still_wins_over_the_resolved_one(
    tmp_path, spool_task_file, monkeypatch
):
    """An operator debugging a gate, and every test in this suite."""
    calls: list[str] = []

    def injected(command, env=None, timeout=0):
        calls.append(command)
        return CommandResult(returncode=0, stdout="ok")

    workflow = TrtllmModelingV2Workflow(
        workspace=tmp_path / "ws",
        command_runner=healthy_smi(),
        shell_runner=injected,
        dry_run=True,
    )
    _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(spool_task_file)

    assert workflow._shell_runner is injected
    assert workflow._shell_runner("x", None, 1).stdout == "ok"
    assert calls[-1] == "x"
    # The dry run's inventory probe went through it as well, which is the
    # same claim stated twice: nothing resolved a second runner behind the
    # injected one's back. Everything the run itself issued is that probe.
    assert calls[:-1] and all(MARKER in command for command in calls[:-1])


# ------------------------------------------------------- checkpoint compatibility


def test_the_checkpoint_schema_version_is_pinned():
    """A checkpoint from the staged pipeline (version 1) is refused, not misread."""
    assert SCHEMA_VERSION == 2


def test_a_version_2_checkpoint_without_optional_fields_loads(tmp_path):
    minimal = {
        "version": 2,
        "task_path": str(tmp_path / "task.yaml"),
        "stage": "planned",
        "done": False,
        "verdict": "",
        "slug": "gpt-oss-120b-tp1",
        "branch": "trtllm-modeling-v2/gpt-oss-120b-tp1",
        "devices": [0],
        "human_stops": 0,
    }
    path = tmp_path / ".trtllm_modeling_v2_state.json"
    path.write_text(json.dumps(minimal), encoding="utf-8")
    state = load_state(path)
    assert state.stage == "planned" and state.devices == [0] and state.tune == {}


def test_the_dispatch_directory_layout_did_not_move(tmp_path):
    """A resume shares its mailbox with the driver that was already running."""
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    assert transport.root == Path(tmp_path) / "dispatch"


def test_a_commit_outside_the_target_changes_the_marker(tmp_path):
    """A kernel task's catalog commit makes every verdict on the target stale.

    The marker fingerprints the commit the checkout is at, whose snapshot the
    checks measure; the run's own workspace inside the checkout is not part
    of it, or every journal line would make every verdict stale.
    """
    repo = tmp_path / "checkout"
    (repo / "tensorrt_llm" / "catalog").mkdir(parents=True)
    (repo / "tensorrt_llm" / "target.py").write_text("T = 1\n")
    (repo / "tensorrt_llm" / "catalog" / "norm.py").write_text("N = 1\n")
    _git(repo, "init", "-q", ".")
    _git(repo, "config", "user.name", "T")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    workflow = TrtllmModelingV2Workflow(workspace=repo / "ws")
    workflow.task_data = {"trtllm_repo_path": str(repo)}
    first = workflow._target_marker()
    assert first.startswith("tree:")

    (repo / "ws").mkdir(exist_ok=True)
    (repo / "ws" / "journal.md").write_text("an event\n")
    assert workflow._target_marker() == first

    (repo / "tensorrt_llm" / "catalog" / "norm.py").write_text("N = 2\n")
    assert workflow._target_marker() == first
    _git(repo, "commit", "-q", "-am", "a kernel task lands")
    assert workflow._target_marker() not in ("", first)


def test_two_gates_started_together_in_one_pid_namespace_get_different_job_names(
    monkeypatch, tmp_path
):
    from agent_flow.workflows.trtllm_modeling_v2.infra.execution import spool as spool_mod

    # Two sandboxed CLIs: same pid, same millisecond, each a fresh counter.
    monkeypatch.setattr(spool_mod.os, "getpid", lambda: 2)
    monkeypatch.setattr(spool_mod.time, "time", lambda: 1_792_000_000.0)
    first = spool_mod.SpoolCommandExecutor(tmp_path)._name()
    second = spool_mod.SpoolCommandExecutor(tmp_path)._name()
    assert first != second
