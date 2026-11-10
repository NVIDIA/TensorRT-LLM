# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 roles behavior and regression checks."""

from __future__ import annotations

import dataclasses
import json
import threading
import time
import tomllib
from dataclasses import asdict, replace
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from jinja2 import StrictUndefined, UndefinedError

from agent_flow.backends import claude_code
from agent_flow.logger import get_logger
from agent_flow.types import AgentTextEvent, SessionInitEvent, ThinkingEvent, ToolCallEvent
from agent_flow.workflows.trtllm_modeling_v2.agents import prompts, roles
from agent_flow.workflows.trtllm_modeling_v2.agents.decisions import (
    decision_path,
    names,
    read_decision,
)
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import _common, _render, kernel, survey
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts.tags import TAG_PROMPTS, task_tags
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts.worker import WORKER_BASE
from agent_flow.workflows.trtllm_modeling_v2.agents.turns import (
    InProcessTurnExecutor,
    TurnOutcome,
    TurnRequest,
    TurnTimeout,
)
from agent_flow.workflows.trtllm_modeling_v2.checks import check_config
from agent_flow.workflows.trtllm_modeling_v2.operators.coverage import QueueItem
from agent_flow.workflows.trtllm_modeling_v2.run.pipeline import ROLE_BY_KIND, TAGS_BY_KIND
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
                "planner": {
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

    layer = workflow._make_agent("planner", "system")
    backend = layer.config.backend
    assert (backend.kind, backend.model) == ("codex", "codex-test-model")
    assert backend.hooks is None
    assert backend.extra_mcp_servers == kb
    assert layer.config.required_tools == ()
    # Without the user's plugins, and, once the agent probe has read them, their skills.
    assert backend.plugins is False and backend.disabled_skills == ()
    workflow._foreign_skills = {"codex": ("imagegen", "cluster-jobs")}
    backend = workflow._make_agent("planner", "system").config.backend
    assert backend.disabled_skills == ("imagegen", "cluster-jobs")


def test_a_role_whose_model_the_spec_does_not_state_is_refused():
    """Nothing is inherited from the launching session or the environment."""
    spec = {"agents": {"defaults": {"model": "m"}, "roles": {"planner": {"backend": "codex"}}}}
    with pytest.raises(roles.AgentConfigError, match="states no model for planner:"):
        roles.resolve_role_agents(spec)
    with pytest.raises(roles.AgentConfigError, match="surveyor, kernel_writer"):
        roles.resolve_role_agents({})
    spec["agents"]["roles"]["planner"]["model"] = "codex-m"
    agents = roles.resolve_role_agents(spec)
    assert (agents["planner"].backend, agents["planner"].model) == ("codex", "codex-m")
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
def test_a_failed_probe_stops_the_run_before_any_planner_turn(tmp_path, task_file, monkeypatch):
    said = "Bash failed: Error: connect EPERM /tmp/claude/srt-mux-9.sock"
    monkeypatch.setattr(agent_probe, "make_probe_layer", lambda a, w: SayingLayer(a, w, said))
    shells: list[str] = []
    workflow = _workflow(
        tmp_path,
        skip_substrate_gate=False,
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


def test_a_turn_past_its_bound_is_stopped_and_says_so(tmp_path):
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
    # What the pipeline records of it: the turn's own failure, with where its activity is.
    assert "the turn was stopped" in str(raised.value)


class _Executor:
    """Answers each turn, and records its prompt and the layer it was bound to."""

    def __init__(self):
        self.layers = []
        self.prompts = []

    def bound(self, layer):
        self.layers.append(layer)
        return SimpleNamespace(run=self.run)

    def run(self, request):
        self.prompts.append(request.prompt)
        return TurnOutcome(role=request.role, text="ok")


class _Layer:
    def __init__(self, fork_from=None):
        self.fork_from = fork_from
        self.closed = False

    def __exit__(self, *args):
        self.closed = True


def _primed(tmp_path, monkeypatch, primer):
    """A workflow whose kernel_writer has ``primer`` ready, its executor, and the forks it made."""
    executor, made = _Executor(), []
    workflow = _workflow(tmp_path, primers=True)
    workflow.task_data = {"agents": {"defaults": {"model": "claude-test-model"}}}
    workflow._turns = executor
    workflow._primers["kernel_writer"] = primer
    workflow._primer_ready["kernel_writer"] = threading.Event()
    workflow._primer_ready["kernel_writer"].set()

    def make_agent(*args, fork_from=None, **kwargs):
        made.append(fork_from)
        return _Layer(fork_from)

    monkeypatch.setattr(workflow, "_make_agent", make_agent)
    return workflow, executor, made


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
    extra = _common.decision_file_block("assembler", ("blocked_reason", "uncertified"))
    assert "<task>-<attempt>.yaml" in plain
    assert "This role also carries" not in plain
    assert "  blocked_reason: ...\n  uncertified: ...\n```" in extra
    assert _common.scope_block("full") == ""
    assert "ignore the vision tensors" in _common.scope_block("text_only")
    for role, prompt in asdict(prompts.DEFAULT_PROMPTS).items():
        for decision in names(role):
            assert decision in prompt
        assert "SPDX-License-Identifier" not in prompt
    assert task_tags("feature", ("fix", "assemble")) == ("assemble", "feature", "fix")
    # Every kind of task the pipeline runs has a role, and a section for each of its tags.
    assert set(TAGS_BY_KIND) | {"kernel"} == set(ROLE_BY_KIND)
    assert {tag for tags in TAGS_BY_KIND.values() for tag in tags} | {"kernel"} == set(TAG_PROMPTS)
    for tag, text in TAG_PROMPTS.items():
        assert text.startswith(f"## Tag `{tag}`")


def test_each_attempt_of_a_task_has_its_own_decision_file_and_only_a_valid_one_is_read(tmp_path):
    first = decision_path(tmp_path, "core", 1)
    assert first == tmp_path / "decisions" / "core-1.yaml" and first.parent.is_dir()
    assert decision_path(tmp_path, "core", 2) != first
    assert decision_path(tmp_path, "core", 1, turn=2).name == "core-1.2.yaml"
    assert decision_path(tmp_path, "k1/rms norm", 1).name == "k1_rms_norm-1.yaml"
    assert read_decision(first, "assembler") is None  # no file is no decision
    first.write_text("summary: s\ndecision: LANDED\n")
    assert read_decision(first, "assembler") is None  # another role's decision is none
    first.write_text("decision: [not, a, mapping")
    assert read_decision(first, "assembler") is None
    first.write_text("summary: s\ndecision: ' UNIT_READY '\nchecks: [boot]\n")
    assert read_decision(first, "assembler") == {
        "summary": "s",
        "decision": "UNIT_READY",
        "checks": ["boot"],
    }


def test_workers_wait_for_results_and_write_decisions_before_ending() -> None:
    for role in ("surveyor", "kernel_writer", "assembler"):
        text = getattr(prompts.DEFAULT_PROMPTS, role)
        assert WORKER_BASE in text
        assert "nothing wakes you after it" in text
        assert "While a job your answer needs is queued or running" in text
        assert "keep\n  polling or waiting in bounded calls" in text
        assert "write your decision file before ending the turn" in text
    layout = " ".join(_common.MODELING_V2_LAYOUT.split())
    assert "static metadata and routing" in layout
    assert "tests that require a GPU" in layout
    assert "do not treat that glob as a CPU-only test suite" in layout


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
        goal="certify norm",
        task_id="k4-rmsnorm",
        task_kind="kernel",
        tags=("fix",),
        tier=4,
        landing="ops/norm/rmsnorm__r4",
        paths=("entry.py", "test.py"),
        shared=("catalog/norm/rmsnorm.py", "catalog/test_rmsnorm.py"),
        environment_kind="local",
        devices=4,
    )


@pytest.mark.parametrize("environment,devices", [("local", 1), ("local", 4)])
def test_kernel_writers_are_spread_over_the_hosts_devices(
    kernel_context: kernel.KernelContext, environment: str, devices: int
) -> None:
    ctx = replace(kernel_context, environment_kind=environment, devices=devices)
    text = kernel.kernel_prompt(ctx)
    assert ("CUDA_VISIBLE_DEVICES=0" in text) is (devices > 1)
    assert text.index(ctx.note) < text.index("## Your task")
    assert text.index("## Tag `kernel`") < text.index("## Tag `fix`")
    assert "from .ops.norm.rmsnorm__r4 import rmsnorm" in text
    assert text.index("`entry.py`") < text.index("`test.py`")
    assert "The surface to certify: bf16" in text


@pytest.mark.parametrize(
    ("tier", "mode", "expected", "absent"),
    [
        (None, "verify", "no rung is recorded", "no implementation yet"),
        (1, "onboard", "Sourcing: **tier 1", "no implementation yet"),
        ("3", "verify", "FlashInfer callable directly", "no implementation yet"),
        (4, "onboard", "no implementation yet", "FlashInfer callable directly"),
        (5, "verify", "Sourcing: **tier 5", "no implementation yet"),
    ],
)
def test_kernel_sourcing_and_landing_branches(
    kernel_context: kernel.KernelContext, tier: object, mode: str, expected: str, absent: str
) -> None:
    ctx = replace(kernel_context, tier=tier, item=replace(kernel_context.item, mode=mode))
    text = kernel.kernel_prompt(ctx)
    assert expected in text
    assert absent not in text
    assert ("Write the entry there" in text) is (mode == "onboard")
    assert ("Start the variant as a copy" in text) is (mode != "onboard")


def test_survey_capture_lists_name_units_and_variants(monkeypatch) -> None:
    capture = check_config.CaptureList((1, 8), False, source="unit.yaml")
    monkeypatch.setattr(
        check_config,
        "launch_captures",
        lambda _: [
            check_config.LaunchCapture("core", "core", "unit.yaml", capture),
            check_config.LaunchCapture("same", "core", "same.yaml", capture),
            check_config.LaunchCapture("disabled", "core", "disabled.yaml", None),
        ],
    )
    text = survey.capture_lists_block({})
    assert "`core`: 2 batches [1, 8], unpadded" in text
    assert "variants same: their unit's list" in text
    assert "variant `disabled` (unit `core`): no CUDA graphs at any gate" in text
    assert "tune" not in text


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
