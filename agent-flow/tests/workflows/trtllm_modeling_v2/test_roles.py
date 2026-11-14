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
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
import yaml
from jinja2 import StrictUndefined, UndefinedError

from agent_flow.backends import claude_code
from agent_flow.logger import get_logger
from agent_flow.types import AgentTextEvent, SessionInitEvent, ThinkingEvent, ToolCallEvent
from agent_flow.workflows.trtllm_modeling_v2.agents import prompts, roles, slots
from agent_flow.workflows.trtllm_modeling_v2.agents.decisions import names
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import _common, _render, kernel, survey
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts.tags import (
    TAG_PROMPTS,
    TAG_SUMMARIES,
    task_tags,
)
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts.worker import WORKER_BASE
from agent_flow.workflows.trtllm_modeling_v2.agents.slots import AgentGate, Priority
from agent_flow.workflows.trtllm_modeling_v2.agents.turns import (
    InProcessTurnExecutor,
    TurnOutcome,
    TurnRequest,
    TurnTimeout,
)
from agent_flow.workflows.trtllm_modeling_v2.checks import check_config
from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as catalog_paths
from agent_flow.workflows.trtllm_modeling_v2.operators.coverage import QueueItem
from agent_flow.workflows.trtllm_modeling_v2.run.execution import kernel_task
from agent_flow.workflows.trtllm_modeling_v2.run.execution.handoff import (
    current_decision,
    decision_path,
    mark_decision_dispatch,
    read_decision,
)
from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import ExecutionStage
from agent_flow.workflows.trtllm_modeling_v2.run.graph import WORKER_TAGS, Graph, Task
from agent_flow.workflows.trtllm_modeling_v2.run.startup import agent_probe, bootstrap, primers
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
def _quick_polls(monkeypatch):
    """Background and side turns look again every 0.1s here, not every 5s."""
    monkeypatch.setattr(slots, "_BACKGROUND_POLL_S", 0.1)


class _Turns:
    """Turns that hold their slot until released, started one thread each."""

    def __init__(self, gate: AgentGate) -> None:
        self.gate = gate
        self.release = threading.Event()
        self.admitted: list[str] = []
        self._lock = threading.Lock()

    def start(self, name: str, priority: Priority) -> None:
        def hold():
            with self.gate.slot(priority):
                with self._lock:
                    self.admitted.append(name)
                self.release.wait()

        threading.Thread(target=hold, daemon=True).start()

    def settle(self) -> None:
        time.sleep(0.2)


@pytest.mark.usefixtures("_quick_polls")
def test_the_planner_owns_the_reserved_slot_and_takes_no_other():
    turns = _Turns(AgentGate(8))
    for i in range(8):
        turns.start(f"task{i}", Priority.REQUIRED)
    turns.settle()
    assert len(turns.admitted) == 7  # the eighth task turn waits: the last slot is the planner's
    turns.start("critic", Priority.SIDE)
    turns.settle()
    assert "critic" not in turns.admitted
    turns.start("planner", Priority.PLANNER)
    turns.settle()
    assert "planner" in turns.admitted and len(turns.admitted) == 8
    turns.release.set()
    # Arriving first, the planner still leaves every task slot to the tasks.
    first = _Turns(AgentGate(4))
    first.start("planner", Priority.PLANNER)
    first.settle()
    for i in range(3):
        first.start(f"task{i}", Priority.REQUIRED)
    first.settle()
    assert len(first.admitted) == 4
    first.release.set()


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


def test_a_turn_past_its_bound_is_stopped_and_retried_as_an_infrastructure_failure(tmp_path):
    # Imported here: importing the orchestrator first walks into the stages' import cycle.
    from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import _not_the_task

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
    # What an unanswered turn raises: the orchestrator retries it as the run's failure.
    assert _not_the_task(f"raised {type(raised.value).__name__}: {raised.value}")


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
            decision.write_text("decision: UNIT_READY\nsummary: measured on the worker tree\n")
            return "UNIT_READY"

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
    assert result.decision == "UNIT_READY"
    assert result.result_file.is_file()
    assert len(list((workspace / "turns").glob("*.request.json"))) == 1
    # Read where the earlier dispatch reserved it: no path is reserved again.
    assert [p.name for p in (workspace / "decisions").glob("*.yaml")] == [answer.name]
    assert result.data["decision"] == "UNIT_READY"
    assert "decision adopted on resume" in workflow.journal_path.read_text()


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


def test_a_codex_role_keeps_its_primer_thread_for_later_forks(tmp_path, monkeypatch):
    executor = _Executor()
    workflow = _workflow(tmp_path, primers=True)
    workflow._turns = executor
    workflow.task_data = {"agents": {"defaults": {"backend": "codex", "model": "codex-test-model"}}}

    def make_agent(name, system_prompt, *, requires_decision=True, session_mode="stateless",
                   fork_from=None):  # fmt: skip
        layer = _Layer(fork_from)
        layer.session_id = "thread-1"
        return layer

    monkeypatch.setattr(workflow, "_make_agent", make_agent)
    ready = threading.Event()
    workflow._primer_ready["kernel_writer"] = ready
    workflow._build_primer("kernel_writer", "v", ready)
    assert ready.is_set()
    assert "your session is copied to start every later kernel_writer turn" in executor.prompts[-1]
    assert workflow._primers["kernel_writer"].session_id == "thread-1"
    start = workflow._turn_start("kernel_writer", "implement")
    assert start.forked and start.layer.fork_from == "thread-1"
    assert primers.load(workflow.workspace)["kernel_writer"].session_id == "thread-1"


def test_a_failed_fork_keeps_the_summary_fallback_across_restart(tmp_path, monkeypatch):
    summary = tmp_path / "briefing.md"
    summary.write_text("saved source reading")
    primer = primers.Primer("kernel_writer", "old-thread", "a" * 40, "v", str(summary))
    workflow, executor, _ = _primed(tmp_path, monkeypatch, primer)
    original = executor.run

    def run(request):
        if request.prompt.startswith("**You are now working your own turn"):
            raise RuntimeError("thread/fork failed before the turn started")
        return original(request)

    monkeypatch.setattr(executor, "run", run)
    assert workflow._run_turn("kernel_writer", "verify", "do the entry") == "ok"
    assert str(summary) in executor.prompts[-1]
    kept = primers.load(workflow.workspace)["kernel_writer"]
    assert kept.session_id == "" and kept.summary == str(summary)


def _write(workspace: Path, agent: str, iteration: int, payload: dict) -> Path:
    path = decision_path(workspace, agent, iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_a_decision_older_than_the_marker_is_refused(tmp_path):
    stale = _write(tmp_path, "assembler", 1, {"summary": "s", "decision": "BLOCKED"})
    os.utime(stale, (1_000_000, 1_000_000))  # written long before this dispatch
    stale.with_name(f".{stale.stem}.dispatched").write_text("", encoding="utf-8")
    assert read_decision(stale, "assembler") is None


def test_a_previous_turns_decision_is_never_read_as_this_ones(tmp_path):
    """A new dispatch is reserved a new path: the earlier verdict is neither read nor overwritten."""
    old = _write(tmp_path, "assembler", 1, {"summary": "s", "decision": "BLOCKED"})
    before = old.read_text()
    target = mark_decision_dispatch(tmp_path, "assembler", 1)

    assert target != old
    assert current_decision(tmp_path, "assembler", 1) == target
    assert read_decision(target, "assembler") is None
    target.write_text("summary: s\ndecision: UNIT_READY\n", encoding="utf-8")
    assert read_decision(target, "assembler")["decision"] == "UNIT_READY"
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
    extra = _common.decision_file_block("assembler", ("blocked_reason", "uncertified"))
    assert "<unit>-surveyor-<turn>.yaml" in plain
    assert "This role also carries" not in plain
    assert "  blocked_reason: ...\n  uncertified: ...\n```" in extra
    assert _common.scope_block("full") == ""
    assert "ignore the vision tensors" in _common.scope_block("text_only")
    for role, prompt in asdict(prompts.DEFAULT_PROMPTS).items():
        for decision in names(role):
            assert decision in prompt
        assert "SPDX-License-Identifier" not in prompt
    assert task_tags("feature", ("investigate", "assemble")) == (
        "assemble",
        "feature",
        "investigate",
    )
    assert tuple(TAG_PROMPTS) == WORKER_TAGS == tuple(TAG_SUMMARIES)
    for tag, text in TAG_PROMPTS.items():
        assert text.startswith(f"## Tag `{tag}`")


def test_workers_wait_for_results_and_write_decisions_before_ending() -> None:
    for role in ("surveyor", "kernel_writer", "assembler", "tuner"):
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
        tags=("tune",),
        tier=4,
        landing="ops/norm/rmsnorm__r4",
        paths=("entry.py", "test.py"),
        shared=("catalog/norm/rmsnorm.py", "catalog/test_rmsnorm.py"),
        environment_kind="local",
        devices=4,
    )


@pytest.mark.parametrize("environment,devices", [("local", 1), ("local", 4), ("session_spool", 4)])
def test_kernel_device_assignment_respects_scheduler_and_device_zero(
    kernel_context: kernel.KernelContext, environment: str, devices: int
) -> None:
    ctx = replace(kernel_context, environment_kind=environment, devices=devices)
    text = kernel.kernel_prompt(ctx)
    assert ("CUDA_VISIBLE_DEVICES=0" in text) is (environment == "local" and devices > 1)
    assert text.index(ctx.note) < text.index("## Your task")
    assert text.index("## Tag `kernel`") < text.index("## Tag `tune`")
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


@pytest.mark.parametrize(
    "name",
    ["fp4_block_scale_moe_runner", "mxe4m3_mxe2m1_block_scale_moe_runner", "noaux_tc_op"],
)
@pytest.mark.parametrize("tests_directory", ["moe", "moe/ops"])
@pytest.mark.parametrize("shared_landing", [False, True])
def test_kernel_prompt_protects_shared_tests_in_task_checkout(
    tmp_path: Path, name: str, tests_directory: str, shared_landing: bool
) -> None:
    repo = tmp_path / "checkout"
    source = f"{catalog_paths.CATALOG_DIR}/moe/{name}.py"
    test = f"{catalog_paths.CATALOG_TESTS_DIR}/{tests_directory}/test_modeling_v2_{name}.py"
    for relative in (source, test):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Shared entry contract.\n", encoding="utf-8")

    target = f"{catalog_paths.MODELING_V2_PACKAGE}/models/family/target"
    ref = f"verify:moe/{name}"
    item = catalog_paths.queue_item_from_ref(ref, "scope_probe")
    assert item is not None
    local_source = f"{target}/ops/moe/{name}__scope_probe.py"
    task = Task(
        id="scope_probe",
        kind="kernel",
        goal=f"Verify {name}",
        ref=ref,
        touches=[source if shared_landing else local_source],
    )
    execution = SimpleNamespace(
        workflow=SimpleNamespace(
            task_data={"trtllm_repo_path": str(repo), "arch": "sm_100"},
            _preflight=SimpleNamespace(target_rel=target),
            workspace=tmp_path / "run",
            vocabulary_path=tmp_path / "run/vocabulary.yaml",
        )
    )
    landing, paths = kernel_task._landing(execution, item, task)
    text = kernel_task._kernel_writer_prompt(execution, item, task=task)
    if shared_landing:
        assert landing == item.target
        assert paths == (source, test)
        assert "Leave " not in text
    else:
        assert landing == item.variant
        assert paths is not None and paths[0] == local_source
        assert f"Leave `{source}` and `{test}`\nas they are:" in text


def test_survey_capture_variants_and_tune_identity_files(tmp_path: Path, monkeypatch) -> None:
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
    identity = tmp_path / "identity.yaml"
    identity.write_text("cuda_graph_config: null\n", encoding="utf-8")
    missing = tmp_path / "missing.yaml"
    text = survey.capture_lists_block(
        {
            "perf": {
                "enabled": True,
                "configs": [
                    {"name": "identity", "identity_configs": [str(identity), str(missing)]},
                    {"name": "derived", "workload": {"concurrency": [1, 8]}},
                    {"name": "overlay", "llm_api_options": {"cuda_graph_config": None}},
                ],
            }
        }
    )
    assert "`core`: 2 batches [1, 8], unpadded" in text
    assert "variants same: their unit's list" in text
    assert "variant `disabled` (unit `core`): no CUDA graphs at any gate" in text
    assert f"tune `identity`: no CUDA graphs, from {identity}" in text
    assert f"{missing} (not readable here)" in text
    assert "`capture_batch_sizes` up to concurrency 8" in text
    assert "tune `overlay`: no CUDA graphs, from its `llm_api_options`" in text


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


@pytest.mark.parametrize("invalid", ["yaml", "capture"])
def test_invalid_tune_configuration_is_reported_without_losing_other_configs(tmp_path, invalid):
    if invalid == "yaml":
        path = tmp_path / "invalid.yaml"
        path.write_text("cuda_graph_config: [\n")
        bad = {"name": "bad", "identity_configs": [str(path)]}
    else:
        bad = {"name": "bad", "llm_api_options": {"cuda_graph_config": "invalid"}}
    text = survey.capture_lists_block(
        {"perf": {"enabled": True, "configs": [bad, {"name": "good"}]}}
    )
    assert "tune `bad`: cannot be resolved" in text
    assert "repair this configuration before measuring" in text
    assert "tune `good`:" in text
