# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight's agent probe: a turn that cannot do its work stops the run before it spends anything.

Both failures it exists for looked like a normal turn: a CLI refused at
start-up, and a Bash tool that could not start its sandbox while the model
reported, in words, that it had tried. No real agent runs here; the layers
are fakes that do -- or fail to do -- what the probe asks, and the probe is
judged by the file, exactly as it judges a real turn.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import anyio
import pytest
import yaml

from agent_flow.agent_runtime import AgentConfig
from agent_flow.backends import claude_code
from agent_flow.logger import get_logger
from agent_flow.workflows.trtllm_modeling_v2.agents.roles import ROLES
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import TURNS_DIRNAME
from agent_flow.workflows.trtllm_modeling_v2.infra.transport import ExternalAgentTransport
from agent_flow.workflows.trtllm_modeling_v2.onboard.survey import Inventory
from agent_flow.workflows.trtllm_modeling_v2.run import agent_probe
from agent_flow.workflows.trtllm_modeling_v2.run.agent_probe import (
    AGENT_PROBE_DIR,
    AGENT_PROBE_FRESH_S,
    AGENT_PROBE_RECORD,
    AGENT_PROBE_SKIP_ENV,
    ProbedAgent,
    distinct_agents,
    format_failures,
    hint_for,
    probe_agents,
    probe_command,
    probe_inputs,
    read_probe_file,
)
from agent_flow.workflows.trtllm_modeling_v2.run.stages import preflight as preflight_stage

from .test_workflow import _workflow

_COMMAND = re.compile(r"^    (\{ printf .*)$", re.M)
_NONCE = re.compile(r"printf '%s\\n' (\S+);")
_OUT = re.compile(r"> (\S+) 2>&1$")


@pytest.fixture(autouse=True)
def _probe_runs(monkeypatch):
    """Undo the suite's skip, and pin the CLI identity the probe records."""
    monkeypatch.delenv(AGENT_PROBE_SKIP_ENV, raising=False)
    monkeypatch.setattr(agent_probe, "launched_cli", lambda: ("/opt/claude", "2.1.296"))


def _command(prompt: str) -> str:
    match = _COMMAND.search(prompt)
    assert match, prompt
    return match.group(1)


class ShellLayer:
    """A turn whose Bash tool works: it runs the command it was given."""

    def __init__(self, agent=None, workspace=None) -> None:
        self.agent, self.workspace = agent, workspace
        self.prompts: list[str] = []
        self.closed = False

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        # In the layer's working directory, as the CLI runs it.
        done = subprocess.run(
            ["bash", "-c", _command(prompt)], capture_output=True, text=True, cwd=self.workspace
        )
        return f"exit {done.returncode}"

    def __exit__(self, *args) -> None:
        self.closed = True


class ScriptedLayer(ShellLayer):
    """A turn that writes the file itself, as a model with only a Write tool could.

    ``head`` is the third line it writes; ``hashed`` whether it can produce the
    nonce's SHA-256, which only a shell does.
    """

    def __init__(self, *args, head: str = "c0ffee1", hashed: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.head, self.hashed = head, hashed

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        command = _command(prompt)
        nonce = _NONCE.search(command).group(1)
        out = Path(_OUT.search(command).group(1))
        digest = hashlib.sha256(nonce.encode()).hexdigest() if self.hashed else "f" * 64
        out.write_text(f"{nonce}\n{digest}  -\n{self.head}\n")
        return "exit 0"


class SayingLayer(ShellLayer):
    """A turn whose Bash tool failed, and whose model said so and ended normally."""

    def __init__(self, *args, said: str = "", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.said = said

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.said


class RefusedLayer(ShellLayer):
    """A CLI the organization refuses: it writes why to stderr and the turn raises."""

    def __call__(self, prompt: str) -> str:
        claude_code._forward_stderr(
            "Claude Code 2.1.259 is older than the minimum version required by your organization"
        )
        raise RuntimeError("Command failed with exit code 1\nError output: Check stderr output")


class StalledLayer(ShellLayer):
    """A turn that never ends; only its bound stops it."""

    async def aforward(self, prompt: str) -> str:
        await anyio.sleep(3600)
        return "never"


def _git_repo(path: Path) -> tuple[Path, str]:
    path.mkdir(parents=True, exist_ok=True)
    for args in (
        ["init", "-q"],
        [
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "base",
        ],
    ):
        subprocess.run(["git", "-C", str(path), *args], check=True)
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return path, head


def _agent(model: str = "claude-opus-5-5", effort: str | None = "xhigh") -> ProbedAgent:
    return ProbedAgent(AgentConfig("claude-code", model, effort), ("planner", "smith"))


# ----------------------------------------------------------- what is probed


def test_every_role_on_the_driving_agent_is_one_probe():
    [probed] = distinct_agents({})
    assert probed.roles == ROLES
    assert probed.agent.backend == "claude-code"


def test_a_role_routed_elsewhere_is_probed_on_its_own_agent():
    task = {
        "agents": {
            "defaults": {"model": "claude-opus-5-5", "reasoning_effort": "xhigh"},
            "roles": {"plan_reviewer": {"backend": "codex", "model": "gpt-6-astra"}},
        }
    }
    probed = distinct_agents(task)
    assert len(probed) == 2
    by_backend = {p.agent.backend: p for p in probed}
    assert by_backend["codex"].roles == ("plan_reviewer",)
    assert by_backend["claude-code"].agent.reasoning_effort == "xhigh"
    assert "plan_reviewer" not in by_backend["claude-code"].roles
    assert by_backend["claude-code"].label == "claude-code claude-opus-5-5 (xhigh)"


# ------------------------------------------------------- the file is the verdict


def test_the_command_writes_a_file_the_probe_accepts(tmp_path):
    repo, head = _git_repo(tmp_path / "checkout")
    out = tmp_path / "1.out"
    subprocess.run(["bash", "-c", probe_command("n0nce", out, str(repo))], check=True)
    assert read_probe_file(out, "n0nce", head, str(repo)) == ""
    # Without preflight's own reading of HEAD, any object id is accepted.
    assert read_probe_file(out, "n0nce", "", str(repo)) == ""


def test_a_file_never_written_is_a_failure_whatever_the_turn_said(tmp_path):
    problem = read_probe_file(tmp_path / "1.out", "n0nce", "", "/repo")
    assert "did not run the command" in problem


def test_a_file_a_shell_did_not_write_is_a_failure(tmp_path):
    out = tmp_path / "1.out"
    out.write_text("n0nce\n" + "f" * 64 + "  -\nc0ffee1\n")
    assert "no shell wrote it" in read_probe_file(out, "n0nce", "c0ffee1", "/repo")
    out.write_text("another\n")
    assert "does not start with this probe's nonce" in read_probe_file(out, "n0nce", "", "/r")


def test_a_shell_that_cannot_read_the_checkout_is_a_failure_with_gits_words(tmp_path):
    out = tmp_path / "1.out"
    digest = hashlib.sha256(b"n0nce").hexdigest()
    out.write_text(f"n0nce\n{digest}  -\nfatal: detected dubious ownership in repository\n")
    problem = read_probe_file(out, "n0nce", "", "/repo")
    assert "`git -C /repo rev-parse HEAD` printed fatal: detected dubious ownership" in problem
    out.write_text(f"n0nce\n{digest}  -\n{'a' * 40}\n")
    assert "(preflight reads " + "b" * 40 + ")" in read_probe_file(out, "n0nce", "b" * 40, "/r")


# ------------------------------------------------------------- the probe turn


def test_a_working_agent_passes_and_leaves_its_turn_on_disk(tmp_path):
    repo, head = _git_repo(tmp_path / "checkout")
    layers: list[ShellLayer] = []

    def make(agent, workspace):
        layers.append(ShellLayer(agent, workspace))
        return layers[-1]

    [outcome] = probe_agents([_agent()], tmp_path / "ws", repo=str(repo), head=head, layer_for=make)
    assert outcome.ok, outcome.problem
    assert layers[0].closed and layers[0].workspace == tmp_path / "ws"
    assert outcome.files == str(tmp_path / "ws" / AGENT_PROBE_DIR / "1" / "001-agent_probe.*")
    turns = tmp_path / "ws" / AGENT_PROBE_DIR / "1"
    request = json.loads((turns / "001-agent_probe.request.json").read_text())
    assert request["what"] == "claude-code claude-opus-5-5 (xhigh)"
    assert request["agent"]["reasoning_effort"] == "xhigh"
    assert (turns / "001-agent_probe.result.md").read_text() == "exit 0"
    assert json.loads((turns / "001-agent_probe.done").read_text())["outcome"] == "answered"


def test_the_command_names_absolute_paths_for_a_turn_that_runs_in_the_workspace(
    tmp_path, monkeypatch
):
    repo, head = _git_repo(tmp_path / "checkout")
    (tmp_path / "ws").mkdir()
    monkeypatch.chdir(tmp_path)
    [outcome] = probe_agents(
        [_agent()], Path("ws"), repo="checkout", head=head, layer_for=ShellLayer
    )
    assert outcome.ok, outcome.problem


def test_every_agent_is_probed_and_each_turn_has_its_own_file(tmp_path):
    repo, head = _git_repo(tmp_path / "checkout")
    agents = [_agent(), ProbedAgent(AgentConfig("codex", "gpt-6-astra"), ("plan_reviewer",))]
    outcomes = probe_agents(
        agents, tmp_path / "ws", repo=str(repo), head=head, layer_for=ShellLayer
    )
    assert [o.ok for o in outcomes] == [True, True]
    turns = tmp_path / "ws" / AGENT_PROBE_DIR
    assert sorted(str(p.relative_to(turns)) for p in turns.rglob("*.out")) == [
        "1/probe.out",
        "2/probe.out",
    ]
    assert len(list(turns.rglob("*.request.json"))) == 2


def test_a_bash_tool_that_failed_in_words_fails_the_probe_with_those_words(tmp_path):
    said = (
        "The Bash tool failed: Error: connect EPERM /tmp/claude/srt-mux-1234.sock. "
        "The command could not be run."
    )
    [outcome] = probe_agents(
        [_agent()], tmp_path, repo="/repo", layer_for=lambda a, w: SayingLayer(said=said)
    )
    assert not outcome.ok and "did not run the command" in outcome.problem
    assert outcome.said == said
    report = format_failures([outcome], environ={"SANDBOX_RUNTIME": "1"})
    assert "srt-mux-1234.sock" in report
    assert "not from inside another agent's session" in report
    assert "This process runs inside one: SANDBOX_RUNTIME is set." in report
    both = format_failures([outcome], environ={"SANDBOX_RUNTIME": "1", "CLAUDECODE": "1"})
    assert "SANDBOX_RUNTIME and CLAUDECODE are set." in both
    assert "--agent-transport external" in report


def test_a_refused_cli_fails_the_probe_with_its_stderr_and_the_cli_to_set(tmp_path):
    [outcome] = probe_agents(
        [_agent()], tmp_path, repo="/repo", layer_for=lambda a, w: RefusedLayer()
    )
    assert outcome.problem.startswith("the turn failed: RuntimeError: Command failed")
    assert any("older than the minimum version" in line for line in outcome.stderr)
    report = format_failures([outcome])
    assert "the CLI's stderr: Claude Code 2.1.259 is older than the minimum version" in report
    assert "(/opt/claude, 2.1.296) is refused. Set AGENT_FLOW_CLAUDE_CLI" in report
    # The turn's record has the stderr too.
    activity = (tmp_path / AGENT_PROBE_DIR / "1" / "001-agent_probe.activity.log").read_text()
    assert "stderr: Claude Code 2.1.259 is older" in activity


def test_a_probe_turn_that_never_ends_is_stopped_at_its_bound(tmp_path):
    started = time.monotonic()
    [outcome] = probe_agents(
        [_agent()], tmp_path, repo="/repo", layer_for=lambda a, w: StalledLayer(), timeout_s=0.2
    )
    assert time.monotonic() - started < 10
    assert "TurnTimeout" in outcome.problem and "no answer within 0.2s" in outcome.problem


def test_the_hints_follow_the_evidence():
    assert "Log in with `claude`" in hint_for("Not logged in · Please run /login", environ={})
    assert "AGENT_FLOW_CLAUDE_CLI" not in hint_for("Not logged in", environ={})
    plain = hint_for("something else entirely", environ={})
    assert plain.count("\n- ") == 0 and "--agent-transport external" in plain
    assert f"{AGENT_PROBE_SKIP_ENV}=1 skips it" in plain


def test_what_a_pass_stands_for(tmp_path, monkeypatch):
    task = {"trtllm_repo_path": str(tmp_path)}
    agents = distinct_agents(task)
    env = {"HOME": "/home/u"}
    inputs = probe_inputs(task, agents, environ=env)
    assert set(inputs) == {"agents", "cli", "process", "checkout"}
    assert probe_inputs(task, agents, environ=env) == inputs

    def changed(**kwargs) -> set[str]:
        now = probe_inputs(
            kwargs.get("task", task), kwargs.get("agents", agents), environ=kwargs.get("env", env)
        )
        return {name for name in now if now[name] != inputs[name]}

    assert changed(env={**env, "SANDBOX_RUNTIME": "1"}) == {"process"}
    assert changed(task={"trtllm_repo_path": "/elsewhere"}) == {"checkout"}
    other = distinct_agents({"agents": {"defaults": {"reasoning_effort": "low"}}})
    assert changed(agents=other) == {"agents"}
    monkeypatch.setattr(agent_probe, "launched_cli", lambda: ("/usr/local/bin/claude", "2.2.0"))
    assert changed() == {"cli"}


# ------------------------------------------------------------- in preflight


class Probes:
    """The layers preflight's probe builds, counted; ``layer`` makes each one."""

    def __init__(self, layer=ScriptedLayer) -> None:
        self.layer = layer
        self.made: list[ShellLayer] = []

    def __call__(self, agent, workspace):
        self.made.append(self.layer(agent, workspace))
        return self.made[-1]


def _preflight(tmp_path, task_file, monkeypatch, probes: Probes, **kwargs):
    monkeypatch.setattr(agent_probe, "make_probe_layer", probes)
    workflow = _workflow(tmp_path, **kwargs)
    log = get_logger().console
    state = workflow._init_state(task_file, log)
    return workflow._run_preflight(state, log), workflow


def _journal(tmp_path) -> str:
    return (tmp_path / "ws" / "journal.md").read_text(encoding="utf-8")


def test_a_new_run_probes_its_agents_and_records_the_pass(tmp_path, task_file, monkeypatch):
    probes = Probes()
    passed, workflow = _preflight(tmp_path, task_file, monkeypatch, probes)
    assert passed and len(probes.made) == 1
    record = json.loads((tmp_path / "ws" / AGENT_PROBE_RECORD).read_text())
    assert set(record["inputs"]) == {"agents", "cli", "process", "checkout"}
    assert record["agents"][0]["roles"] == list(ROLES)
    assert "**agent probe passed** — run because this is a new run" in _journal(tmp_path)
    # Role turns in this process now leave their files beside the run.
    assert workflow._turns.records == tmp_path / "ws" / TURNS_DIRNAME


def test_a_failed_probe_stops_the_run_before_any_planner_turn(tmp_path, task_file, monkeypatch):
    said = "Bash failed: Error: connect EPERM /tmp/claude/srt-mux-9.sock"
    probes = Probes(lambda a, w: SayingLayer(a, w, said=said))
    shells: list[str] = []
    passed, workflow = _preflight(
        tmp_path,
        task_file,
        monkeypatch,
        probes,
        skip_substrate_gate=False,
        shell_runner=lambda command, env=None, timeout=0: shells.append(command),
    )
    assert not passed
    journal = _journal(tmp_path)
    assert "agent probe failed" in journal and "srt-mux-9.sock" in journal
    assert "a plain terminal or a tmux session on the login node" in journal
    assert not (tmp_path / "ws" / AGENT_PROBE_RECORD).exists()
    # Nothing past it ran: not the substrate gate, not the survey.
    assert shells == []


def test_a_resume_in_the_same_environment_stands_on_the_pass(tmp_path, task_file, monkeypatch):
    _preflight(tmp_path, task_file, monkeypatch, Probes())
    probes = Probes()
    passed, workflow = _preflight(tmp_path, task_file, monkeypatch, probes)
    assert passed and workflow.resume and probes.made == []
    assert "**agent probe not run** — passed 0.0 h ago in this environment" in _journal(tmp_path)


def test_a_resume_probes_again_once_the_agents_change_or_the_pass_is_old(
    tmp_path, task_file, monkeypatch
):
    _preflight(tmp_path, task_file, monkeypatch, Probes())
    spec = yaml.safe_load(task_file.read_text())
    spec["agents"] = {"defaults": {"reasoning_effort": "high"}}
    task_file.write_text(yaml.safe_dump(spec))
    probes = Probes()
    _preflight(tmp_path, task_file, monkeypatch, probes)
    assert len(probes.made) == 1
    assert "run because the agents differs from the last pass's" in _journal(tmp_path)

    path = tmp_path / "ws" / AGENT_PROBE_RECORD
    record = json.loads(path.read_text())
    path.write_text(json.dumps({**record, "at": time.time() - AGENT_PROBE_FRESH_S - 60}))
    probes = Probes()
    _preflight(tmp_path, task_file, monkeypatch, probes)
    assert len(probes.made) == 1 and "past the 12 h window" in _journal(tmp_path)


def test_a_dry_run_probes_too(tmp_path, task_file, monkeypatch):
    probes = Probes()
    monkeypatch.setattr(agent_probe, "make_probe_layer", probes)
    monkeypatch.setattr(preflight_stage, "run_inventory_probe", lambda *a, **k: Inventory())
    with _workflow(tmp_path, dry_run=True) as workflow:
        workflow.run(task_file)
    assert len(probes.made) == 1
    assert (tmp_path / "ws" / AGENT_PROBE_RECORD).is_file()


def test_the_external_transport_is_not_probed(tmp_path, task_file, monkeypatch):
    probes = Probes()
    transport = ExternalAgentTransport(tmp_path / "ws", poll_s=0.01)
    passed, _ = _preflight(tmp_path, task_file, monkeypatch, probes, agent_transport=transport)
    assert passed and probes.made == []
    assert not (tmp_path / "ws" / AGENT_PROBE_RECORD).exists()


def test_the_probe_can_be_skipped_by_name(tmp_path, task_file, monkeypatch):
    monkeypatch.setenv(AGENT_PROBE_SKIP_ENV, "1")
    probes = Probes()
    passed, _ = _preflight(tmp_path, task_file, monkeypatch, probes)
    assert passed and probes.made == []
    assert f"**agent probe not run** — ${AGENT_PROBE_SKIP_ENV} is set" in _journal(tmp_path)
