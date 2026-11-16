# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 roles behavior and regression checks."""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
import tomllib
from dataclasses import asdict, replace
from importlib.resources import files
from pathlib import Path
from typing import Any

import anyio
import pytest
import yaml
from jinja2 import StrictUndefined, UndefinedError

from agent_flow.backends import claude_code
from agent_flow.logger import get_logger
from agent_flow.types import AgentTextEvent, SessionInitEvent, ThinkingEvent, ToolCallEvent
from agent_flow.workflows.trtllm_modeling_v2.agents import prompts, roles
from agent_flow.workflows.trtllm_modeling_v2.agents.decisions import names
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import _common, _render, kernel, survey
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts.tags import TAG_PROMPTS, task_tags
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts.worker import WORKER_BASE
from agent_flow.workflows.trtllm_modeling_v2.agents.turns import (
    InProcessTurnExecutor,
    TurnEnded,
    TurnRequest,
    TurnTimeout,
)
from agent_flow.workflows.trtllm_modeling_v2.checks import check_config
from agent_flow.workflows.trtllm_modeling_v2.operators.coverage import QueueItem
from agent_flow.workflows.trtllm_modeling_v2.run.execution.handoff import (
    current_decision,
    decision_path,
    mark_decision_dispatch,
    read_decision,
)
from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import ExecutionStage
from agent_flow.workflows.trtllm_modeling_v2.run.graph import WORKER_KINDS, Graph, Task
from agent_flow.workflows.trtllm_modeling_v2.run.startup import agent_probe, bootstrap
from agent_flow.workflows.trtllm_modeling_v2.run.startup.agent_probe import (
    AGENT_PROBE_RECORD,
    AGENT_PROBE_SKIP_ENV,
)

from ._helpers import _workflow


def test_a_claude_role_keeps_its_stop_hook_and_a_codex_role_gets_none(tmp_path):
    """The hook refuses a turn that wrote nothing; the file is validated when read.

    Codex refuses per-client Claude SDK hooks, so a Stop hook would fail the turn.
    """
    workflow = _workflow(tmp_path)
    kb = {"kb": {"type": "http", "url": "https://kb.test/mcp"}}
    workflow.task_data = {
        "agents": {
            "defaults": {"backend": "claude-code", "model": "claude-test-model"},
            "roles": {
                "assembler": {"reasoning_effort": "high"},
                "surveyor": {
                    "backend": "codex",
                    "model": "codex-test-model",
                    "extra_mcp_servers": kb,
                },
            },
        }
    }
    layer = workflow._make_agent("assembler", "system")
    backend = layer.config.backend
    assert (backend.kind, backend.model, backend.reasoning_effort) == (
        "claude-code",
        "claude-test-model",
        "high",
    )
    assert backend.hooks and backend.hooks["Stop"]
    assert backend.cwd == workflow.workspace
    assert layer.config.required_tools == ()
    # Nothing wakes a worker after its turn ends: the tool to be woken by is not offered.
    assert layer.config.disallowed_tools == ("Monitor",)

    layer = workflow._make_agent("surveyor", "system")
    backend = layer.config.backend
    assert (backend.kind, backend.model) == ("codex", "codex-test-model")
    assert backend.hooks is None
    assert backend.extra_mcp_servers == kb
    assert layer.config.required_tools == () and layer.config.disallowed_tools == ()
    # Without the user's plugins, and, once the agent probe has read them, their skills.
    assert backend.plugins is False and backend.disabled_skills == ()
    workflow._foreign_skills = {"codex": ("imagegen", "cluster-jobs")}
    backend = workflow._make_agent("surveyor", "system").config.backend
    assert backend.disabled_skills == ("imagegen", "cluster-jobs")


def test_a_role_whose_model_the_spec_does_not_state_is_refused():
    """Nothing is inherited from the launching session or the environment."""
    spec = {"agents": {"defaults": {"model": "m"}, "roles": {"surveyor": {"backend": "codex"}}}}
    with pytest.raises(roles.AgentConfigError, match="states no model for surveyor:"):
        roles.resolve_role_agents(spec)
    with pytest.raises(roles.AgentConfigError, match="surveyor, kernel_writer, assembler:"):
        roles.resolve_role_agents({})
    spec["agents"]["roles"]["surveyor"]["model"] = "codex-m"
    agents = roles.resolve_role_agents(spec)
    assert (agents["surveyor"].backend, agents["surveyor"].model) == ("codex", "codex-m")
    assert (agents["assembler"].backend, agents["assembler"].model) == ("claude-code", "m")


@pytest.fixture
def _probe_runs(monkeypatch):
    """Undo the suite's skip, and pin the CLI identity the probe records."""
    monkeypatch.delenv(AGENT_PROBE_SKIP_ENV, raising=False)
    monkeypatch.setattr(agent_probe, "launched_cli", lambda: ("/opt/claude", "2.1.296"))


class SayingLayer:
    """A turn whose Bash tool failed, and whose model said so and ended normally."""

    def __init__(self, agent, workspace, said: str) -> None:
        self.said = said

    async def aforward(self, prompt: str) -> str:
        return self.said

    def __exit__(self, *args) -> None:
        pass


@pytest.mark.usefixtures("_probe_runs")
def test_a_failed_probe_stops_the_run_before_any_agent_turn(tmp_path, task_file, monkeypatch):
    said = "Bash failed: Error: connect EPERM /tmp/claude/srt-mux-9.sock"
    monkeypatch.setattr(agent_probe, "make_probe_layer", lambda a, w: SayingLayer(a, w, said))
    shells: list[str] = []
    workflow = _workflow(
        tmp_path,
        shell_runner=lambda command, env=None, timeout=0: shells.append(command),
    )
    log = get_logger().console
    assert not bootstrap._run_preflight(workflow, workflow._init_state(task_file, log), log)
    journal = (tmp_path / "ws" / "journal.md").read_text(encoding="utf-8")
    assert "agent probe failed" in journal and "srt-mux-9.sock" in journal
    # The file never written fails the probe, whatever the turn said.
    assert "did not run the command" in journal
    assert "workflow does not change that policy" in journal
    assert not (tmp_path / "ws" / AGENT_PROBE_RECORD).exists()
    # Nothing past it ran: not the substrate gate, not the survey.
    assert shells == []


@dataclasses.dataclass(frozen=True)
class _Config:
    """The parts of ``AgentLayerConfig`` the executor touches."""

    on_activity: Any = None


class StreamingLayer:
    """An ``AgentLayer`` as the executor sees it: a config with an observer, and ``aforward``.

    Each turn streams its session, a tool call, a thinking block and its text
    through the config's observer, as ``AgentLayer._execute`` does, and writes
    ``stderr`` as the CLI's stderr reader would. ``hold`` keeps a turn open until set.
    """

    def __init__(self, *, stderr: str = "", hold: threading.Event | None = None) -> None:
        self.config = _Config()
        self.stderr = stderr
        self.hold = hold
        self.cancelled = False

    def _observe(self, kind: str, event: Any) -> None:
        if self.config.on_activity is not None:
            self.config.on_activity(kind, event)

    async def aforward(self, prompt: str) -> str:
        session = SessionInitEvent(
            backend="codex", model="resolved-model", reasoning_effort="medium", session_id="s1"
        )
        self._observe("session", session)
        self._observe("tool", ToolCallEvent(name="Bash", input={"command": f"echo {prompt}"}))
        self._observe("thinking", ThinkingEvent(text="first line\nsecond line"))
        if self.stderr:
            claude_code._forward_stderr(self.stderr)
        try:
            while self.hold is not None and not self.hold.is_set():
                await anyio.sleep(0.01)
        except anyio.get_cancelled_exc_class():
            self.cancelled = True
            raise
        self._observe("text", AgentTextEvent(text=f"done with {prompt}"))
        return f"answer to {prompt}"


def _request(role: str = "kernel_writer", what: str = "k12", prompt: str = "go") -> TurnRequest:
    return TurnRequest(
        role=role,
        what=what,
        prompt=prompt,
        system_prompt="ROLE PROMPT",
        decision_file="decisions/core-kernel_writer-1.yaml",
        model="m",
    )


def test_a_turn_past_its_bound_is_stopped_and_counts_as_a_failed_attempt(tmp_path):
    layer = StreamingLayer(hold=threading.Event())
    executor = InProcessTurnExecutor(lambda r: layer, timeout_s=0.2, records=tmp_path)
    started = time.monotonic()
    with pytest.raises(
        TurnTimeout, match=r"001-kernel_writer: no answer within 0.2s \(the kernel_writer turn"
    ) as raised:
        executor.run(_request())
    assert time.monotonic() - started < 5
    # Cancelled, not abandoned: the CLI an AgentLayer started is stopped with it.
    assert layer.cancelled
    done = json.loads((tmp_path / "001-kernel_writer.done").read_text())
    assert done["outcome"] == "timed out" and "001-kernel_writer.activity.log" in done["error"]
    # What an unanswered turn raises is no receipt: the orchestrator counts the attempt
    # failed and starts the next one while the task's budget lasts.
    assert isinstance(raised.value, TurnTimeout)


def test_a_turn_the_run_ends_is_stopped_as_at_its_limit_and_says_it_was_ended(tmp_path):
    """The session that watches a run answers a task while its agent is still in its turn."""
    layer = StreamingLayer(hold=threading.Event())
    executor = InProcessTurnExecutor(lambda r: layer, timeout_s=60, records=tmp_path)
    assert executor.end("k12") is False, "no turn is in flight yet"
    raised: list[BaseException] = []

    def turn() -> None:
        try:
            executor.run(_request())
        except BaseException as exc:  # what the task's thread would see
            raised.append(exc)

    thread = threading.Thread(target=turn)
    thread.start()
    deadline = time.monotonic() + 5
    while not executor.end("k12"):
        assert time.monotonic() < deadline, "the turn never started"
        time.sleep(0.01)
    thread.join(10)
    assert not thread.is_alive()
    # Cancelled as at its own limit: the CLI is stopped with it, and the record says which.
    (exc,) = raised
    assert isinstance(exc, TurnEnded) and "the turn was ended before it answered" in str(exc)
    assert layer.cancelled
    assert json.loads((tmp_path / "001-kernel_writer.done").read_text())["outcome"] == "ended"
    assert executor.end("k12") is False, "the turn is over"

    # Watching for the end changes nothing else: a turn that raises is still seen as
    # what it raised, with its own words.
    class Refusing:
        async def aforward(self, prompt):
            raise ValueError("the backend said no")

    with pytest.raises(ValueError, match="the backend said no"):
        InProcessTurnExecutor(lambda r: Refusing(), timeout_s=5).run(_request())


def _completed(workspace, *, task="core-tp", attempt=2):
    decision = mark_decision_dispatch(
        workspace,
        "assembler",
        5,
        unit="core",
        identity={"task_id": task, "task_attempt": attempt},
    )

    class Answer:
        async def aforward(self, prompt):
            decision.write_text("decision: ready\nsummary: measured on the worker tree\n")
            return "ready"

    records = workspace / "turns"
    original = InProcessTurnExecutor(lambda request: Answer(), records=records)
    original.run(
        TurnRequest(
            role="assembler", what=task, work=task, prompt="assemble", decision_file=str(decision)
        )
    )
    return original, decision


def _resumed(workspace):
    def unexpected(request):
        raise AssertionError("a completed worker turn must not start an agent")

    return InProcessTurnExecutor(unexpected, records=workspace / "turns")


def _relaunched(tmp_path, *, attempt):
    """A workflow relaunched on a running task's second attempt, after a turn of ``attempt``."""
    workspace = tmp_path / "ws"
    _, answer = _completed(workspace, attempt=attempt)
    Graph(tasks=[Task(id="core-tp", kind="feature", goal="g", status="running", attempts=2)]).save(
        workspace / "graph.yaml"
    )
    workflow = _workflow(tmp_path)
    workflow._turns = _resumed(workspace)
    return workflow, answer


def test_a_relaunch_adopts_without_another_agent_call_or_decision_copy(tmp_path):
    workflow, answer = _relaunched(tmp_path, attempt=2)
    workspace = workflow.workspace
    result = ExecutionStage(workflow)._dispatch_task("assembler", "core", "core-tp", "assemble")
    assert result.decision == "ready"
    assert result.result_file.is_file()
    assert len(list((workspace / "turns").glob("*.request.json"))) == 1
    # Read where the earlier dispatch reserved it: no path is reserved again.
    assert [p.name for p in (workspace / "decisions").glob("*.yaml")] == [answer.name]
    assert result.data["decision"] == "ready"
    assert "decision adopted on resume" in workflow.journal_path.read_text()


def _write(workspace: Path, agent: str, iteration: int, payload: dict) -> Path:
    path = decision_path(workspace, agent, iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_a_decision_older_than_the_marker_is_refused(tmp_path):
    stale = _write(tmp_path, "assembler", 1, {"summary": "s", "decision": "blocked"})
    os.utime(stale, (1_000_000, 1_000_000))  # written long before this dispatch
    stale.with_name(f".{stale.stem}.dispatched").write_text("", encoding="utf-8")
    assert read_decision(stale, "assembler") is None


def test_a_previous_turns_decision_is_never_read_as_this_ones(tmp_path):
    """A new dispatch is reserved a new path: the earlier verdict is neither read nor overwritten."""
    old = _write(tmp_path, "assembler", 1, {"summary": "s", "decision": "blocked"})
    before = old.read_text()
    target = mark_decision_dispatch(tmp_path, "assembler", 1)

    assert target != old
    assert current_decision(tmp_path, "assembler", 1) == target
    assert read_decision(target, "assembler") is None
    target.write_text("summary: s\ndecision: ready\n", encoding="utf-8")
    assert read_decision(target, "assembler")["decision"] == "ready"
    assert old.read_text() == before


def test_missing_context_raises() -> None:
    assert _render._ENVIRONMENT.undefined is StrictUndefined
    with pytest.raises(UndefinedError):
        _render.render_template("common/notes.md.j2")


def test_markdown_and_user_notes_are_neither_escaped_nor_rendered_again() -> None:
    notes = "<tag> & a > b; {{ not_a_variable }} {% not_a_statement %}"
    assert _render._ENVIRONMENT.autoescape is False
    for prompt in asdict(prompts.build_trtllm_modeling_v2_prompts(notes=f"  {notes}\n")).values():
        assert notes in prompt
        assert "&lt;tag&gt;" not in prompt
    assert _common.notes_block(" \n ") == ""


def test_decision_file_extra_keys_scope_and_role_endings() -> None:
    plain = _common.decision_file_block("surveyor")
    extra = _common.decision_file_block("assembler", ("check", "next_question"))
    assert "<name>-surveyor-<turn>.yaml" in plain and "<unit>" not in plain
    assert "It also carries" not in plain
    # Each key is shown in the form its value takes: a sentence where a number belongs
    # once cost a finished kernel row its attempt.
    assert 'check: {id: "<the card\'s check id>"' in extra
    assert 'next_question: "<the word that names what answers it>: <your one question' in extra
    assert "The file must parse as YAML" in extra
    # `decision` is the first key of the example, and the two values are the only ones.
    assert plain.index("decision: ready") < plain.index("summary: |")
    assert _common.scope_block("full") == ""
    assert "ignore the vision tensors" in _common.scope_block("text_only")
    assert set(asdict(prompts.DEFAULT_PROMPTS)) == {"surveyor", "kernel_writer", "assembler"}
    for role, prompt in asdict(prompts.DEFAULT_PROMPTS).items():
        assert names(role) == ("ready", "blocked")
        for decision in names(role):
            assert f"- `{decision}`:" in prompt
        for key in ("tried", "command", "result", "ruled_out", "next_question"):
            assert key in prompt
        assert "SPDX-License-Identifier" not in prompt
    # A row of the target is briefed by its own short section, not by the whole target's;
    # a new attempt adds its section.
    assert task_tags("feature", ("investigate",)) == ("feature", "investigate")
    assert [task_tags(kind) for kind in ("weights", "routing", "state", "assemble")] == [
        ("weights",),
        ("routing",),
        ("state",),
        ("assemble",),
    ]
    assert set(TAG_PROMPTS) - {"investigate"} <= set(WORKER_KINDS)


def test_workers_wait_for_results_and_write_decisions_before_ending() -> None:
    for role in ("surveyor", "kernel_writer", "assembler"):
        text = getattr(prompts.DEFAULT_PROMPTS, role)
        assert WORKER_BASE in text
    # The roles that commit read what holds for the code they test; the surveyor does not.
    assert _common.MODELING_V2_NATIVE in prompts.DEFAULT_PROMPTS.kernel_writer
    assert _common.MODELING_V2_NATIVE in prompts.DEFAULT_PROMPTS.assembler
    assert _common.MODELING_V2_NATIVE not in prompts.DEFAULT_PROMPTS.surveyor


@pytest.fixture
def kernel_context() -> kernel.KernelContext:
    return kernel.KernelContext(
        item=QueueItem(4, "onboard", "norm/rmsnorm", surface="bf16", note="certify rows"),
        workspace="/run",
        repo="/repo",
        arch="sm_103",
        target_rel="modeling_v2/models/family/target",
        vocabulary_path="/run/vocabulary.yaml",
        note="Finish the previous attempt.",
        goal="## Your task: `k4-rmsnorm` (kernel, attempt 1 of 3)\n\n**What it is for.** certify norm",
        task_id="k4-rmsnorm",
        task_kind="kernel",
        landing="ops/norm/rmsnorm__r4",
        paths=("entry.py", "test.py"),
        shared=("catalog/norm/rmsnorm.py", "catalog/test_rmsnorm.py"),
    )


def test_no_kernel_writer_is_told_a_device_the_watcher_assigns_it(
    kernel_context: kernel.KernelContext,
) -> None:
    """On one host too: concurrent writers get distinct devices from the spool, not an index."""
    ctx = kernel_context
    text = kernel.kernel_prompt(ctx)
    assert "CUDA_VISIBLE_DEVICES=" not in text
    assert text.index(ctx.note) < text.index("## Your task")
    assert text.index("## Your task") < text.index("## Tag `kernel`")
    assert "tags:" not in text
    assert "from .ops.norm.rmsnorm__r4 import rmsnorm" in text
    assert text.index("`entry.py`") < text.index("`test.py`")
    assert "The surface to certify: bf16" in text


@pytest.mark.parametrize("mode", ["onboard", "verify"])
def test_a_kernel_dispatch_says_where_the_entry_lands(
    kernel_context: kernel.KernelContext, mode: str
) -> None:
    ctx = replace(kernel_context, item=replace(kernel_context.item, mode=mode))
    text = kernel.kernel_prompt(ctx)
    assert ("Write the entry there" in text) is (mode == "onboard")
    assert ("It starts as a copy of the entry's module" in text) is (mode != "onboard")


def test_survey_capture_variants(monkeypatch) -> None:
    capture = check_config.CaptureList((1, 8), False, source="tp1.yaml")
    monkeypatch.setattr(
        check_config,
        "launch_captures",
        lambda _: [
            check_config.LaunchCapture("core", "core", "tp1.yaml", capture),
            check_config.LaunchCapture("same", "core", "same.yaml", capture),
            check_config.LaunchCapture("disabled", "core", "disabled.yaml", None),
        ],
    )
    # Performance work is not the flow's: the survey plans no serving config.
    text = survey.capture_lists_block({})
    assert "`core`: 2 batches [1, 8], unpadded" in text
    assert "same: the same list as their deployment" in text
    assert "`disabled` (a server of deployment `core`): no CUDA graphs in any check" in text
    for old in ("tune", "variant", "unit", "gate"):
        assert old not in text


def test_all_template_resources_are_declared_as_package_data() -> None:
    root = Path(__file__).resolve().parents[3]
    package = "agent_flow.workflows.trtllm_modeling_v2"
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    package_root = root / Path(*package.split("."))
    patterns = config["tool"]["setuptools"]["package-data"][package]
    declared = {path for pattern in patterns for path in package_root.glob(pattern)}
    templates = set((package_root / "agents/prompts/templates").rglob("*.md.j2"))
    assert templates and templates <= declared
    assert any(dep.lower().startswith("jinja2>=") for dep in config["project"]["dependencies"])
    resources = files(prompts) / "templates"
    loaded = set(_render._ENVIRONMENT.list_templates())
    assert loaded == {
        str(path.relative_to(package_root / "agents/prompts/templates")) for path in templates
    }
    for name in loaded:
        assert (resources / name).is_file()
        _render._ENVIRONMENT.get_template(name)
