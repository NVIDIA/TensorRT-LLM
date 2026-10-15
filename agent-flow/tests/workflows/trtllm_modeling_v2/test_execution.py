# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the orchestrator's turns and commands actually go.

Two things are being protected here and they are different in kind.

The first is that the mapping from a ``task.yaml`` environment kind to a
mechanism lives in one place. Before this it was inline in
``_run_preflight``, twice, in the file that also owns resume and gate
sequencing — so adding the next mechanism meant editing the state machine.

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

from agent_flow.runtime import (
    CommandSpec,
    DelegatedCommandExecutor,
    InProcessTurnExecutor,
    LocalCommandExecutor,
    SlurmStepExecutor,
    SpoolCommandExecutor,
    TurnOutcome,
    TurnRequest,
)
from agent_flow.workflows.trtllm_modeling_v2 import gates as gates_mod
from agent_flow.workflows.trtllm_modeling_v2 import workflow as workflow_mod
from agent_flow.workflows.trtllm_modeling_v2.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.execution import (
    command_runner_for,
    select_command_executor,
    select_turn_executor,
    shell_runner_for,
)
from agent_flow.workflows.trtllm_modeling_v2.preflight import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.state import SCHEMA_VERSION, load_state
from agent_flow.workflows.trtllm_modeling_v2.survey import MARKER, Inventory
from agent_flow.workflows.trtllm_modeling_v2.transport import ExternalAgentTransport
from agent_flow.workflows.trtllm_modeling_v2.workflow import TrtllmModelingV2Workflow

from .test_workflow import Script, _install, healthy_smi

# ------------------------------------------------------- executor selection


def _environment(kind: str, **kwargs) -> RunEnvironment:
    return RunEnvironment(kind=kind, python="/opt/venv/bin/python", **kwargs)


def test_local_resolves_to_a_subprocess_here():
    assert isinstance(select_command_executor(_environment("local")), LocalCommandExecutor)


def test_slurm_resolves_to_a_step_executor_that_declares_itself_distributed():
    executor = select_command_executor(_environment("slurm", world_size=8))
    assert isinstance(executor, SlurmStepExecutor)
    assert executor.capabilities.distributed
    assert executor.capabilities.max_world_size == 8


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


def test_the_broker_resolves_to_a_delegated_executor(tmp_path):
    """The orchestrator cannot submit here, so a turn does — and only that.

    The verdict still comes from files the job writes; see
    `tests/runtime/test_delegated_executor.py` for the property that an
    agent's report cannot move it.
    """
    executor = select_command_executor(
        _environment("slurm_broker", container_image="img", partition="p"),
        turns=InProcessTurnExecutor(lambda r: None),
        workspace=tmp_path,
    )
    assert isinstance(executor, DelegatedCommandExecutor)
    assert executor.root == tmp_path / "broker"


def test_the_broker_submission_is_a_script_path_not_an_interpolated_command(tmp_path):
    """The template quotes its payload; a gate command is full of quotes."""
    executor = select_command_executor(
        _environment("slurm_broker", container_image="img", partition="p"),
        turns=InProcessTurnExecutor(lambda r: None),
        workspace=tmp_path,
    )
    call = executor.render_submission("/w/broker/gate-1.sh")
    assert 'command=["bash", "-lc", "bash /w/broker/gate-1.sh"])' in call
    assert 'image="img"' in call


def test_without_a_turn_executor_the_broker_falls_back_rather_than_crashing():
    """A `--dry-run` has no turn executor to hand over and must still resolve."""
    executor = select_command_executor(_environment("slurm_broker"))
    assert type(executor) is LocalCommandExecutor


# ------------------------------------------------ the legacy runner adapters


def test_the_shell_adapter_produces_the_shape_the_gates_take():
    run = shell_runner_for(LocalCommandExecutor())
    result = run("echo hi", None, 30)
    assert isinstance(result, gates_mod.ShellResult)
    assert result.ok and result.stdout.strip() == "hi"


def test_the_shell_adapter_keeps_a_timeout_distinct_from_a_failure():
    run = shell_runner_for(LocalCommandExecutor())
    assert run("exit 5", None, 30).returncode == 5
    timed = run("sleep 30", None, 1)
    assert timed.timed_out and not timed.ok


def test_the_shell_adapter_passes_the_environment_through():
    run = shell_runner_for(LocalCommandExecutor())
    assert run("echo $MV2FLOW_ADAPTER", {"MV2FLOW_ADAPTER": "yes"}, 30).stdout.strip() == "yes"


def test_the_probe_adapter_produces_the_shape_preflight_takes():
    run = command_runner_for(LocalCommandExecutor())
    result = run(["echo", "a b"], None, 30)
    assert isinstance(result, CommandResult)
    assert result.stdout == "a b\n"


def test_the_probe_adapter_reports_a_missing_binary_as_127():
    run = command_runner_for(LocalCommandExecutor())
    assert run(["mv2flow-no-such-binary-xyz"], None, 30).returncode == 127


def test_the_gate_default_runner_goes_through_the_local_executor(monkeypatch):
    """Not merely equivalent to it -- actually it.

    The point of Phase 2 is that there is one implementation of "run a
    command here". A second one living on in `gates.py` would be the thing
    the refactor was for.
    """
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


def test_the_spool_gate_runner_goes_through_the_spool_executor(tmp_path):
    """`spool_runner` keeps its name and signature and owns none of the protocol."""
    run = gates_mod.spool_runner(tmp_path, poll_s=0.01, name_prefix="probe")
    result = run("echo hi", None, 0)  # nothing is watching, so it times out
    assert result.timed_out
    script = sorted((tmp_path / "pending").glob("probe-*.sh"))[-1]
    assert script.read_text(encoding="utf-8").rstrip().endswith("echo hi")


# -------------------------------------------------------- turn executor choice


def test_no_transport_means_this_process_calls_the_layer():
    executor = select_turn_executor(None, lambda request: (lambda prompt: "hi"))
    assert isinstance(executor, InProcessTurnExecutor)
    assert executor.run(TurnRequest(role="smith", what="x", prompt="p")).text == "hi"


def test_a_turn_executor_is_used_as_given(tmp_path):
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    assert select_turn_executor(transport, lambda request: None) is transport


def test_a_pre_turn_request_callable_is_still_accepted(tmp_path):
    """A keyword-callable transport still works.

    That is how the external transport was wired before the turn types
    existed, and a caller outside this package has no way to know it was
    supposed to change.
    """
    seen: dict = {}

    def legacy(*, role, what, prompt, system_prompt="", decision_file=None):
        seen.update(role=role, what=what, prompt=prompt, system_prompt=system_prompt)
        return "legacy answer"

    executor = select_turn_executor(legacy, lambda request: None)
    outcome = executor.run(TurnRequest(role="auditor", what="audit", prompt="p", system_prompt="S"))
    assert isinstance(outcome, TurnOutcome)
    assert outcome.text == "legacy answer"
    assert seen == {"role": "auditor", "what": "audit", "prompt": "p", "system_prompt": "S"}


# ------------------------------------- the workflow no longer knows the names


def test_the_workflow_neither_names_nor_builds_a_platform_mechanism():
    """A mechanism should be removable without reading the state machine."""
    source = inspect.getsource(workflow_mod)
    for name in (
        "session_spool",
        "slurm_broker",
        "spool_runner",
        "SpoolCommandExecutor",
        "LocalCommandExecutor",
        "SlurmStepExecutor",
        "srun",
    ):
        assert name not in source, f"workflow.py still names the mechanism {name!r}"
    assert "environment.kind" not in source
    assert ".kind ==" not in source


def test_the_one_remaining_platform_read_is_the_auditor_log_hint():
    """Pinned rather than hidden, because it is real and it is not Phase 2's.

    After a resume the gate log is empty (it is in memory), so the auditor is
    told where the dumps are on disk instead of being handed an empty
    section. That location is a property of where commands ran, so it does
    belong to the executor — `ExecutionRecord.log_path` already carries it
    per command. Routing the *glob* through the executor needs the evidence
    plane that Phase 5 introduces, and doing it here would mean inventing
    half of one.

    So: exactly one read, in exactly one method. If a second appears, this
    fails and the question gets asked again.
    """
    source = inspect.getsource(workflow_mod)
    assert source.count('"spool"') == 1
    assert '"spool"' in inspect.getsource(TrtllmModelingV2Workflow._auditor_dump)


def test_run_turn_has_no_transport_branch():
    """Both transports answer `TurnExecutor.run`, so there is nothing to branch on.

    This is the Phase 1 acceptance criterion stated as a test: the file that
    owns stage sequencing must not be able to tell the two apart.
    """
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
    monkeypatch.setattr(workflow_mod, "run_inventory_probe", lambda *a, **k: Inventory())
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


def test_an_injected_runner_still_wins_over_the_resolved_one(
    tmp_path, spool_task_file, monkeypatch
):
    """An operator debugging a gate, and every test in this suite."""
    calls: list[str] = []

    def injected(command, env=None, timeout=0):
        calls.append(command)
        return gates_mod.ShellResult(returncode=0, stdout="ok")

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


def test_the_checkpoint_schema_did_not_move():
    """Phase 1 and 2 must be resumable from a checkpoint written before them."""
    assert SCHEMA_VERSION == 1


def test_a_checkpoint_written_before_the_refactor_still_loads(tmp_path):
    """Byte-for-byte what the pre-refactor `save_state` wrote."""
    legacy = {
        "version": 1,
        "task_path": str(tmp_path / "task.yaml"),
        "stage": "assemble",
        "done": False,
        "verdict": "",
        "slug": "gpt-oss-120b-tp1",
        "branch": "trtllm-modeling-v2/gpt-oss-120b-tp1",
        "devices": [0],
        "plan_round": 1,
        "plan_approved": True,
        "queue": [{"n": 1, "mode": "onboard", "target": "norm/rms", "status": "done"}],
        "unit_index": 0,
        "unit_turns": 2,
        "units_done": [],
        "audit_clean": False,
        "human_stops": 0,
    }
    path = tmp_path / ".trtllm_modeling_v2_state.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    state = load_state(path)
    assert state.stage == "assemble"
    assert state.plan_approved is True
    assert state.queue[0]["target"] == "norm/rms"
    assert state.unit_turns == 2


def test_the_dispatch_directory_layout_did_not_move(tmp_path):
    """A resume shares its mailbox with the driver that was already running."""
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    assert transport.root == Path(tmp_path) / "dispatch"
