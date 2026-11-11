# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 roles behavior and regression checks."""

from __future__ import annotations

import dataclasses
import json
import re
import threading
import time
import tomllib
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
from agent_flow.workflows.trtllm_modeling_v2.agents import prompts, roles
from agent_flow.workflows.trtllm_modeling_v2.agents.decisions import (
    decision_path,
    decision_problem,
    read_decision,
)
from agent_flow.workflows.trtllm_modeling_v2.agents.turns import (
    InProcessTurnExecutor,
    TurnOutcome,
    TurnRequest,
    TurnTimeout,
)
from agent_flow.workflows.trtllm_modeling_v2.checks.boot import resolve_boot_spec
from agent_flow.workflows.trtllm_modeling_v2.operators.coverage import (
    VERDICT_AFTER_ADDITIONS,
    VocabularyError,
    parse_vocabulary,
)
from agent_flow.workflows.trtllm_modeling_v2.run import start
from agent_flow.workflows.trtllm_modeling_v2.run.pipeline import ROLE_BY_KIND
from agent_flow.workflows.trtllm_modeling_v2.run.startup import agent_probe
from agent_flow.workflows.trtllm_modeling_v2.run.startup.agent_probe import AGENT_PROBE_SKIP_ENV
from agent_flow.workflows.trtllm_modeling_v2.spec.deployments import Deployment, Server
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import (
    TaskSchemaError,
    load_and_validate_task_yaml,
)

from ._helpers import _workflow, facts


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

    layer = workflow._make_agent("surveyor", "system")
    backend = layer.config.backend
    assert (backend.kind, backend.model) == ("codex", "codex-test-model")
    assert backend.hooks is None
    assert backend.extra_mcp_servers == kb
    assert layer.config.required_tools == ()
    # Without the user's plugins, and, once the agent probe has read them, their skills.
    assert backend.plugins is False and backend.disabled_skills == ()
    workflow._foreign_skills = {"codex": ("imagegen", "cluster-jobs")}
    backend = workflow._make_agent("surveyor", "system").config.backend
    assert backend.disabled_skills == ("imagegen", "cluster-jobs")


def test_a_role_whose_model_the_spec_does_not_state_is_one_line_of_the_schema(task_file):
    """Nothing is inherited from the launching session or the environment."""
    spec = {"agents": {"defaults": {"model": "m"}, "roles": {"surveyor": {"backend": "codex"}}}}
    assert roles.unstated_models(spec) == ["surveyor"]
    assert roles.unstated_models({}) == list(roles.ROLES)
    spec["agents"]["roles"]["surveyor"]["model"] = "codex-m"
    assert roles.unstated_models(spec) == []
    agents = roles.resolve_role_agents(spec)
    assert (agents["surveyor"].backend, agents["surveyor"].model) == ("codex", "codex-m")
    assert (agents["assembler"].backend, agents["assembler"].model) == ("claude-code", "m")
    # Said when the spec loads, with every other line of the schema, before anything starts.
    data = yaml.safe_load(task_file.read_text())
    data["agents"] = {"defaults": {"backend": "claude-code"}}
    task_file.write_text(yaml.safe_dump(data))
    with pytest.raises(TaskSchemaError) as raised:
        load_and_validate_task_yaml(task_file)
    lines = [line for line in str(raised.value).splitlines() if "agents.defaults.model" in line]
    assert len(lines) == 1 and "unstated for surveyor, kernel_writer, assembler" in lines[0]
    assert "fix: `agents: {defaults: {backend: claude-code, model: <model>}}`" in lines[0]


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
def test_a_failed_probe_stops_the_run_before_any_turn(tmp_path, task_file, monkeypatch):
    said = "Bash failed: Error: connect EPERM /tmp/claude/srt-mux-9.sock"
    monkeypatch.setattr(agent_probe, "make_probe_layer", lambda a, w: SayingLayer(a, w, said))
    shells: list[str] = []
    workflow = _workflow(
        tmp_path, shell_runner=lambda command, env=None, timeout=0: shells.append(command)
    )
    log = get_logger().console
    assert not start.start(workflow, workflow._init_state(task_file, log), log)
    journal = (tmp_path / "ws" / "journal.md").read_text(encoding="utf-8")
    assert "agent probe failed" in journal and "srt-mux-9.sock" in journal
    # The file never written fails the probe, whatever the turn said.
    assert "did not run the command" in journal
    assert "workflow does not change that policy" in journal
    # Nothing past it ran: not the survey's inventory probe.
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


def test_each_attempt_of_a_task_has_its_own_decision_file_and_only_a_valid_one_is_read(tmp_path):
    first = decision_path(tmp_path, "core", 1)
    assert first == tmp_path / "decisions" / "core-1.yaml" and first.parent.is_dir()
    assert decision_path(tmp_path, "core", 2) != first
    assert decision_path(tmp_path, "core", 1, turn=2).name == "core-1.2.yaml"
    assert decision_path(tmp_path, "k1/rms norm", 1).name == "k1_rms_norm-1.yaml"
    assert read_decision(first) is None  # no file is no receipt
    assert decision_problem(first) == f"there is no file at `{first}`"
    # One receipt for every worker: ready or blocked, and nothing else.
    first.write_text("summary: s\ndecision: UNIT_READY\n")
    assert read_decision(first) is None
    assert "`decision` is 'UNIT_READY'" in decision_problem(first)
    assert "it must be `ready` or `blocked`" in decision_problem(first)
    first.write_text("decision: [not, a, mapping")
    assert read_decision(first) is None and "is not valid YAML" in decision_problem(first)
    first.write_text("summary: s\n")
    assert "has no `decision` key" in decision_problem(first)
    first.write_text("summary: s\ndecision: ' blocked '\nnot_done: [boot]\n")
    assert read_decision(first) == {"summary": "s", "decision": "blocked", "not_done": ["boot"]}


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
    loaded = set(prompts._TEMPLATES.list_templates())
    assert loaded == {
        str(path.relative_to(package_root / "agents/prompts/templates")) for path in templates
    }
    for name in loaded:
        assert (resources / name).is_file()
        prompts._TEMPLATES.get_template(name)


# -------------------------------------------------------------------------- the prompts

_SECTIONS = {
    "config": "## Task: make the tree load the checkpoint's config",
    "survey": "## Task: survey",
    "kernel": "## Task: one operator entry (queue row 4)",
    "core": "## Task: core, the first deployment `kv_fp8`",
    "deployment": "## Task: deployment `kv_fp8`",
    "fix": "## Task: the final fix",
}
#: What a turn's message holds besides the run's own facts, for the size limits.
_QUIET = dict(python="", host_facts="", inventory="", history=[], spec_notes="")
_CARDS = ("disaggregated", "multi_node", "drafter", "multimodal", "kv_dtype", "state_dtype")


def test_every_kind_of_task_has_a_role_its_section_and_the_one_receipt() -> None:
    assert set(_SECTIONS) == set(ROLE_BY_KIND)
    for kind, heading in _SECTIONS.items():
        said = prompts.dispatch(kind, facts())
        # The priorities come first, then the task, last the receipt: ready or blocked.
        assert said.startswith("## Priorities, in this order\n\n1. Onboard: ")
        assert said.index(heading) < said.index("## The receipt")
        assert "Begin with `decision: ready` or `decision: blocked`: " in said
        assert "```yaml\ndecision: ready\nsummary: |" in said
        assert "to exactly this path. The run reads nothing else.\n\n`/ws/decisions/" in said
        assert "list" not in said.split("## Priorities")[0] and "SPDX" not in said
        assert "Never change an anchor, a tolerance or a check" in said
        # The prefix of any other GPU command, as the run resolved it.
        assert "## GPU commands" in said and "under `/opt/venv/bin/python`" in said
        assert "\n    CUDA_VISIBLE_DEVICES=0 TMPDIR=/scratch/tmp\n" in said
        assert "with this prefix, on the device(s) the run claimed (" in said
        assert "{{" not in said and "{%" not in said
    kernel = prompts.dispatch("kernel", facts())
    assert "- `entry`: the entry's path as committed" in kernel and "- `tier`: " in kernel
    assert "- `entry`: " not in prompts.dispatch("core", facts())
    # A fact a template names and nobody supplied raises: no prompt goes out with a hole.
    assert prompts._TEMPLATES.undefined is StrictUndefined
    with pytest.raises(UndefinedError):
        prompts.dispatch("core", {})
    # What the operator wrote is sent as written, neither escaped nor rendered again.
    notes = "<tag> & a > b; {{ not_a_variable }} {% not_a_statement %}"
    assert notes in prompts.dispatch("core", facts(spec_notes=notes))
    assert "## Notes from the spec" not in prompts.dispatch("core", facts())
    scoped = prompts.dispatch("core", facts(text_only=True))
    assert "## Scope: text only" in scoped and "skips the vision tensors" in scoped


def test_a_kernel_task_names_its_row_its_landing_and_its_sourcing() -> None:
    said = prompts.dispatch("kernel", facts())
    assert "**verify** the entry `norm/rmsnorm` for `sm_103`." in said
    assert "Surface to certify, the value the entry's cells exclude today: bf16" in said
    assert "Cells to certify, each the `spec` of one `Cell`: `[{'rows': 2048}]`" in said
    assert "Sourcing: tier 3: a kernel of a library the engine pins (FlashInfer)" in said
    assert "the target imports `from .ops.norm.rmsnorm__r4 import rmsnorm`" in said
    assert said.index("rmsnorm__r4.py`") < said.index("test_modeling_v2_rmsnorm__r4.py`")
    assert "First copy `catalog/norm/rmsnorm.py` and " in said
    assert "Leave the shared files as they are." in said
    assert "cd /ws/worktrees/k4 && PYTHONPATH=/ws/worktrees/k4 python -m pytest" in said
    new = dict(onboard=True, mode="onboard", tier=None, shared="", surface="", cells="")
    onboard = prompts.dispatch("kernel", facts(**new))
    assert "No entry wraps this op: write one." in onboard and "First copy" not in onboard
    assert "no tier is recorded" in onboard and "Surface to certify" not in onboard
    survey = prompts.dispatch("survey", facts(worktree="", gate=""))
    assert all(f"     {tier}. {text}" in survey for tier, text in prompts.TIERS.items())
    # The survey has no worktree and runs no check: its message says neither.
    assert "## Worktree and commits" not in survey and "gate <boot|" not in survey


def test_a_deployment_loads_the_one_card_its_options_ask_for() -> None:
    assert prompts.cards(Deployment(name="tp1", server=Server(options={}))) == []
    quantized = {"kv_cache_config": {"dtype": "fp8"}}
    pd = Deployment(
        name="pd",
        ctx=Server(options={}, launch="srun -N1"),
        gen=Server(options=quantized),
        inputs=("text", "image"),
    )
    assert prompts.cards(pd) == ["disaggregated", "multi_node", "multimodal", "kv_dtype"]
    state = {"dtype": "auto", "mamba_ssm_cache_dtype": "float16"}
    drafted = {"speculative_config": {"decoding_type": "MTP"}, "kv_cache_config": state}
    assert prompts.cards(Deployment(name="mtp", server=Server(options=drafted))) == [
        "drafter",
        "state_dtype",
    ]
    said = prompts.dispatch("deployment", facts(cards=prompts.cards(pd)))
    assert "### Card: disaggregated serving" in said and "### Card: KV-cache dtype" not in said
    assert "also asks for what these cards cover: `multi_node`, `multimodal`, `kv_dtype`" in said
    assert "/flow/trtllm_modeling_v2/agents/prompts/templates/cards/" in said
    assert "- Read the card below before you change code." in said
    assert "Read the card below" not in prompts.dispatch("deployment", facts())
    # What the first GPU run of a disaggregated deployment paid for is on its card.
    assert "state `cache_transceiver_config` with a `backend`" in said
    assert "`UCX_TLS=tcp` stops the NIXL backend from starting" in said
    assert '`UCX_TLS: "^ib"` worked on GB300' in said
    for card in _CARDS:
        text = prompts.dispatch("deployment", facts(cards=[card]))
        # A card opens with its check and the command, then says what "supported" means.
        opening = text.split("### Card: ")[0].rstrip().splitlines()[-1]
        assert opening.startswith("Check: `/ws/bin/trtllm-modeling-v2 gate <boot|"), card
        assert "**Supported means:**" in text.split("### Card: ")[1]


def test_one_turn_loads_about_twelve_hundred_words_and_the_texts_stay_small() -> None:
    def words(kind: str, **changes) -> int:
        return len(prompts.dispatch(kind, facts(**_QUIET, **changes)).split())

    sizes = {kind: words(kind) for kind in _SECTIONS if kind != "survey"}
    sizes["survey"] = words("survey", worktree="", gate="")  # it has no worktree and no check
    sizes.update({card: words("deployment", cards=[card]) for card in _CARDS})
    # Counted with the fixture's paths and commands in it, which are not standing text.
    assert max(sizes.values()) <= 1270, sizes
    assert all(size <= 1200 for kind, size in sizes.items() if kind != "kernel"), sizes
    # What is maintained: the base, the receipt, the six sections and the six cards; and
    # with them what a message says of the run (its facts, a repair, the host's file).
    root = Path(prompts.__file__).parent / "templates"
    texts = {p.name: len(p.read_text().split()) - 13 for p in root.rglob("*.md.j2")}
    run_facts = sum(texts.pop(name) for name in ("run.md.j2", "repair.md.j2", "host_facts.md.j2"))
    assert len(texts) == 14 and sum(texts.values()) <= 4320, texts
    assert run_facts <= 1210, run_facts


def test_what_a_text_says_of_the_code_is_what_the_code_does() -> None:
    """A weak model reads each word literally: a sentence about the code is held to the code."""
    # A nonzero pytest exit is a failure for the run's own test jobs: never "ignore it".
    host = (Path(prompts.__file__).parent / "templates" / "host_facts.md.j2").read_text()
    assert "never on the exit code" not in host
    assert "a nonzero exit is a failure, as it is for the run's own test jobs" in host
    # `scope: text_only` is about the target; a deployment that states images is sent one.
    said = prompts.dispatch("core", facts(text_only=True))
    assert "A check sends an image only to a deployment whose `inputs` name one." in said
    assert "send text prompts only" not in said
    sees = Deployment(name="vl", server=Server(options={}), inputs=("text", "image"))
    reads = Deployment(name="t", server=Server(options={}))
    scope = {"scope": "text_only"}
    assert resolve_boot_spec(scope, sees).has_images
    assert not resolve_boot_spec(scope, reads).has_images
    # The survey: the keys of a queue row code checks, and that it counts the cells, no more.
    survey = prompts.dispatch("survey", facts())
    assert (
        "Of a row it checks `mode`, `depends_on`, `tier`, `unit`, `site`, a `verify` row's "
        "`surface` and the number of `cells`; the rest the workers read." in survey
    )
    assert "- A `verify` row needs `surface`; it lists its `cells`, at most 64." in survey
    row = {"n": 1, "mode": "verify", "target": "norm/rms", "surface": "dtype=bf16"}

    def parsed(**changes):
        document = {"verdict": VERDICT_AFTER_ADDITIONS, "queue": [{**row, **changes}]}
        return parse_vocabulary(document)

    assert parsed() and parsed(cells=[{"anything": ["a", "worker", "reads"]}] * 64)
    for wrong, why in (
        (dict(surface=""), "verify dispatch with no `surface`"),
        (dict(cells=[{}] * 65), "lists 65 cells, more than 64"),
        (dict(mode="check"), "mode is 'check'"),
        (dict(depends_on="1"), "depends_on must be a list"),
        (dict(tier=9), "tier is 9"),
        (dict(site="A B"), "site is 'A B'"),
    ):
        with pytest.raises(VocabularyError, match=re.escape(why)):
            parsed(**wrong)
    # The kernel: one sentence parts a silent fallback from torch ops as the stated tier,
    # and a large cell is marked with the names the tree has.
    kernel = prompts.dispatch("kernel", facts())
    assert (
        "No silent torch fallback: it certifies nothing; torch ops as the stated "
        "implementation (tier 4) say so in `note`." in kernel
    )
    assert "No torch fallback" not in kernel and "composition of torch" not in kernel
    assert (
        'a larger one gets `"large": True` in its `spec` and `pytest.mark.high_cuda_memory`.'
        in (kernel)
    )
    assert "spec.large" not in kernel
    assert prompts.TIERS[4].startswith("no implementation exists: compose one from torch ops")


def test_the_texts_name_only_paths_the_tree_has() -> None:
    """A prompt is a map of paths: one the tree lost sends a worker nowhere."""
    root = Path(__file__).resolve().parents[4]
    if not (root / "tensorrt_llm").is_dir():
        pytest.skip("not inside a TensorRT-LLM checkout")
    text = "\n".join(p.read_text() for p in Path(prompts.__file__).parent.rglob("*.j2"))
    named = {prompts.MV2_README, prompts.BUILD_GUIDE}
    for found in re.findall(r"`((?:tensorrt_llm|tests|cpp|examples)/[^`\s]+)`", text):
        path = found.split("::")[0].rstrip(".,;:)")
        if not any(ch in path for ch in "*<{["):
            named.add(path)
    missing = sorted(p for p in named if not (root / p).exists())
    assert len(named) > 12 and not missing, f"the texts name paths this tree lacks: {missing}"
    # A marker a worker is told to use is one the tree's pytest.ini registers, and the key
    # of a cell's `spec` it is told to set is one the tree's tests read.
    marks = set(re.findall(r"pytest\.mark\.(\w+)", text))
    registered = (root / "tests/unittest/pytest.ini").read_text()
    assert marks == {"high_cuda_memory"} and "\n    high_cuda_memory:" in registered
    tests = root / "tests/unittest/_torch/modeling_v2"
    assert any('spec.get("large")' in p.read_text() for p in tests.rglob("test_*.py"))
