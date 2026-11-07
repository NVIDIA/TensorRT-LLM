# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 integration behavior and regression checks."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import boot as boot_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import verdict as check_result
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.git import (
    CommitIdentity,
    CommitPolicy,
    git_output,
)
from agent_flow.workflows.trtllm_modeling_v2.operators.coverage import QueueItem
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.execution import handoff as r
from agent_flow.workflows.trtllm_modeling_v2.run.execution import kernel_task as worker
from agent_flow.workflows.trtllm_modeling_v2.run.execution.handoff import (
    TaskResult,
    decision_file_instruction,
)
from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import ExecutionStage
from agent_flow.workflows.trtllm_modeling_v2.run.graph import Task
from agent_flow.workflows.trtllm_modeling_v2.run.records.journal import Journal
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run.validation import runtime_facts as rf
from agent_flow.workflows.trtllm_modeling_v2.run.validation.check_runner import CheckRunner
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow
from agent_flow.workflows.trtllm_modeling_v2.run.workspace import snapshots as snapshots_mod
from agent_flow.workflows.trtllm_modeling_v2.run.workspace.integration import (
    REBASE_RETRIES,
    SPARSE_CONE,
    Integration,
)

from ._helpers import (
    Script,
    _git,
    _install,
    _workflow,
    passing_shell,
    planner_graph_path,
    unit_tasks,
    write_graph,
)

_integration_ENV = {"GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.com"}


def _integration_git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@example.com", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _integration_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _integration_git(repo, "init", "-q", "-b", "main")
    (repo / "a.py").write_text("A = 1\n")
    (repo / "b.py").write_text("B = 1\n")
    _integration_git(repo, "add", ".")
    _integration_git(repo, "commit", "-q", "-m", "base")
    return repo


def _integration_commit(tree: Path, name: str, text: str, message: str) -> None:
    (tree / name).write_text(text)
    _integration_git(tree, "commit", "-q", "-am", f"{message}\n\nSigned-off-by: T <t@example.com>")


def test_uncommitted_work_or_a_commit_that_breaks_the_rules_goes_back_to_its_worker(tmp_path):
    """A plain `git commit` in a worktree is audited once, before it can land."""
    repo = _integration_repo(tmp_path)
    policy = CommitPolicy(identity=CommitIdentity("T", "t@example.com"), allowed_paths=("a.py",))
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _integration_ENV, policy=policy)
    tree = integration.open("k1")
    (tree / "a.py").write_text("A = 2\n")
    outcome = integration.integrate("k1")
    assert not outcome.merged and "uncommitted changes" in outcome.detail
    assert not outcome.base  # nothing for a rebase to resolve
    _integration_git(tree, "commit", "-q", "-am", "unsigned")
    outcome = integration.integrate("k1")
    assert not outcome.merged and outcome.repairable
    assert "no Signed-off-by" in outcome.detail and "rebase --signoff" in outcome.detail
    assert (repo / "a.py").read_text() == "A = 1\n"
    _integration_git(tree, "rebase", "-q", "--signoff", _integration_git(repo, "rev-parse", "HEAD"))
    assert integration.integrate("k1").merged


class _Journal:
    def __init__(self):
        self.events: list[str] = []

    def event(self, kind, text):
        self.events.append(text)


class _integration_Stage(ExecutionStage):
    """What a task's worktree turn uses of the workflow, with a scripted worker."""

    def __init__(self, workspace, integration, worker):
        super().__init__(
            SimpleNamespace(workspace=workspace, _tls=threading.local(), journal=_Journal())
        )
        self._integration = integration
        self.worker = worker
        #: The worktree block each turn was given, the note on a rebase included.
        self.blocks: list[str] = []

    def _work_task_body(self, task):
        self.blocks.append(self._tls.worktree)
        return TaskResult(self.worker(len(self.blocks)))


def test_a_conflicting_task_goes_back_to_its_worker_and_then_integrates(tmp_path):
    repo = _integration_repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _integration_ENV)
    first, second = integration.open("k1"), integration.open("k2")
    _integration_commit(first, "a.py", "A = 2\n", "a")
    # k2's own change predates k1's landing, so its worktree keeps its base.
    _integration_commit(second, "a.py", "A = 3\n", "a again")
    assert integration.integrate("k1").merged
    head = _integration_git(repo, "rev-parse", "HEAD")

    def worker(turn):
        if turn == 1:
            return "UNIT_READY"
        # What the note asks for: rebase onto the head, keep both changes, continue.
        rebased = subprocess.run(["git", "-C", str(second), "rebase", head], capture_output=True)
        assert rebased.returncode != 0
        (second / "a.py").write_text("A = 2\nA3 = 3\n")
        _integration_git(second, "add", "a.py")
        _integration_git(second, "-c", "core.editor=true", "rebase", "--continue")
        return "UNIT_READY"

    stage = _integration_Stage(tmp_path / "ws", integration, worker)
    assert (
        stage._work_task_turn(g.Task(id="k2", kind="feature", goal="mtp")).decision == "UNIT_READY"
    )
    assert (repo / "a.py").read_text() == "A = 2\nA3 = 3\n"
    assert _integration_git(repo, "rev-list", "--merges", "--count", "HEAD") == "0"
    assert not second.exists()  # integrated, so closed
    first_block, rebase_turn = stage.blocks
    assert "did not integrate" not in first_block
    assert rebase_turn.startswith(first_block)
    assert "did not integrate" in rebase_turn and f"rebase {head}" in rebase_turn
    assert "`a.py`" in rebase_turn and "BLOCKED" in rebase_turn
    events = stage.workflow.journal.events
    assert "conflicts with the integration head" in events[0]
    assert events[1] == f"k2: back to its assembler to rebase onto {head[:10]} and resolve " + (
        f"(1 of {REBASE_RETRIES})"
    )
    assert events[2].startswith("k2: integrated")


IN_CONE = (
    "README.md",
    "setup.py",
    "tensorrt_llm/__init__.py",
    "tensorrt_llm/models/modeling.py",
    "tests/conftest.py",
    "tests/unittest/conftest.py",
    "tests/unittest/_torch/test_modeling.py",
    "tests/unittest/llmapi/apps/openai_server.py",
    "tests/unittest/llmapi/test_llm_args.py",
    "tests/unittest/utils/util.py",
    "tests/test_common/hooks.py",
    "tests/integration/defs/conftest.py",
    "tests/integration/test_lists/l0.yml",
    "examples/llm-api/quickstart_advanced.py",
    "examples/visual_gen/cat.png",
)


OUTSIDE = (
    "cpp/CMakeLists.txt",
    "cpp/kernels/rmsnorm.cu",
    "3rdparty/lib/lib.h",
    "docs/guide/page.md",
    "docs/other/notes.md",
    "scripts/build.py",
    "tests/unittest/llmapi/other/test_unrelated.py",
)


def _integration_sparse_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _integration_git(repo, "init", "-q", "-b", "main")
    for path in IN_CONE + OUTSIDE:
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(f"{path} 1\n")
    (repo / ".gitignore").write_text("tensorrt_llm/*.so\n")
    _integration_git(repo, "add", ".")
    _integration_git(repo, "commit", "-q", "-m", "base")
    (repo / "tensorrt_llm" / "libnative.so").write_text("elf")  # the native build
    return repo


def test_a_commit_from_a_sparse_worktree_leaves_every_file_outside_the_cone_intact(tmp_path):
    repo = _integration_sparse_repo(tmp_path)
    integration = Integration(repo, tmp_path / "ws", "mv2/run", _integration_ENV)
    tree = integration.open("k1")
    # The checkout moves on outside the cone while the task works, so integrating is a real rebase.
    (repo / "cpp" / "kernels" / "rmsnorm.cu").write_text("rmsnorm 2\n")
    (repo / "docs" / "new.md").write_text("new\n")
    _integration_git(repo, "add", ".")
    _integration_git(repo, "commit", "-q", "-m", "native")
    expected = {p: (repo / p).read_text() for p in _integration_git(repo, "ls-files").split()}
    (tree / "tensorrt_llm" / "models" / "modeling.py").write_text("modeling 2\n")
    _integration_git(tree, "commit", "-q", "-am", "model\n\nSigned-off-by: T <t@example.com>")

    outcome = integration.integrate("k1")
    assert outcome.merged and outcome.commits == 1
    expected["tensorrt_llm/models/modeling.py"] = "modeling 2\n"
    assert set(_integration_git(repo, "ls-files").split()) == set(expected)
    for path, text in expected.items():
        assert (repo / path).read_text() == text, path
        assert _integration_git(repo, "show", f"HEAD:{path}") + "\n" == text, path
    assert _integration_git(repo, "rev-list", "--merges", "--count", "HEAD") == "0"
    integration.close("k1")
    assert not tree.exists()


def test_the_cone_holds_every_top_level_package_the_package_imports():
    """A sparse worktree must import: each repository package ``tensorrt_llm`` imports is in it."""
    import re as _re

    root = Path(__file__).resolve().parents[4]
    package = root / "tensorrt_llm"
    if not package.is_dir():
        pytest.skip("this checkout has no tensorrt_llm/ beside agent-flow/")
    top = {p.name for p in root.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    pattern = _re.compile(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", _re.MULTILINE)
    imported = set()
    for source in package.rglob("*.py"):
        imported.update(pattern.findall(source.read_text(encoding="utf-8", errors="replace")))
    needed = sorted((top & imported) - {"tensorrt_llm"})
    assert needed, "expected tensorrt_llm to import at least one sibling package"
    assert [name for name in needed if name not in SPARSE_CONE] == []


def _journal(tmp_path: Path) -> Journal:
    journal = Journal(path=tmp_path / "journal.md")
    journal.open(slug="t")
    return journal


class _receipts_Stage(ExecutionStage):
    """What a task's turn uses of the workflow; the turn itself is scripted."""

    def __init__(self, workspace: Path, integration=None):
        super().__init__(
            SimpleNamespace(
                workspace=workspace,
                _tls=threading.local(),
                journal=_journal(workspace),
                _git=git_output,
            )
        )
        self._integration = integration


def test_a_finished_task_keeps_a_receipt_with_the_commits_it_integrated(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _integration_git(repo, "init", "-q", "-b", "main")
    (repo / "a.py").write_text("A = 1\n")
    _integration_git(repo, "add", ".")
    _integration_git(repo, "commit", "-q", "-m", "base")
    ws = tmp_path / "ws"
    ws.mkdir()
    integration = Integration(
        repo, ws, "mv2/run", {"GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.com"}
    )

    class Worker(_receipts_Stage):
        def _work_task_body(self, task):
            tree = integration.tree(task.id)
            for n in (1, 2):
                (tree / "a.py").write_text(f"A = {n + 1}\n")
                _integration_git(tree, "commit", "-q", "-am", f"mtp: step {n}")
            # The turn as the workflow leaves it: its report, its decision file.
            decision = ws / "decisions" / "mtp-assembler-1.yaml"
            journal = self.workflow.journal
            report = journal.record_dispatch(
                "assembler", task.id, f"Do {task.id}.\n\n{decision_file_instruction(decision)}"
            )
            decision.parent.mkdir(parents=True)
            summary = "mtp at tp1.\n\nFor the planner: depth 3 only."
            decision.write_text(yaml.safe_dump({"decision": "UNIT_READY", "summary": summary}))
            journal.record_result(report, "UNIT_READY; mtp at tp1.")
            return r.TaskResult.answered("assembler", decision, report)

    stage = Worker(ws, integration)
    task = g.Task(id="mtp", kind="feature", goal="mtp", ref="mtp")
    assert stage._work_task(task) == "UNIT_READY"
    receipt = r.read(ws / "tasks", "mtp")
    assert receipt["decision"] == "UNIT_READY" and receipt["summary"] == "mtp at tp1."
    assert [c.split(" ", 1)[1] for c in receipt["commits"]] == ["mtp: step 2", "mtp: step 1"]
    assert receipt["commits"][0].split()[0] == _integration_git(
        repo, "rev-parse", "--short", "HEAD"
    )
    assert receipt["decision_file"] == str(ws / "decisions" / "mtp-assembler-1.yaml")
    assert Path(receipt["result_file"]).name.endswith("-assembler-mtp.md")

    # A process that never integrated it knows only the branch tip.
    later = _receipts_Stage(ws, integration)
    [tip] = later._landed_commits("mtp")
    assert tip.endswith("mtp: step 2 (the branch tip)")
    # A task that did not finish leaves none.
    later._keep_receipt(g.Task(id="core", kind="assemble", goal="g"), r.TaskResult("BLOCKED", "x"))
    assert r.read(ws / "tasks", "core") is None


_BIND = "modeling_v2 x: bound MambaHybridCacheManagerV2 #1: 897 recurrent slots, 32 per slot\n"


def _binding_shell():
    """Every command passes, and logs the recurrent pool its engine bound."""
    inner = passing_shell()

    def run(command, env=None, timeout=0):
        result = inner(command, env, timeout)
        return CommandResult(returncode=result.returncode, stdout=result.stdout + _BIND)

    return run


def test_a_dependent_starts_with_the_receipt_and_the_measured_facts(
    tmp_path, task_file, monkeypatch
):
    planner_prompts: list[str] = []
    dependent_prompts: list[str] = []

    def planner(turn, prompt):
        planner_prompts.append(prompt)
        if turn > 1:
            return {"decision": "DONE", "summary": "the contract should hold"}
        tasks = unit_tasks(planner_graph_path(prompt).parent)
        tasks.append(
            {"id": "why-core", "kind": "investigate", "goal": "why core binds", "ref": "core",
             "depends_on": ["core"]}
        )  # fmt: skip
        write_graph(prompt, tasks)
        return {"decision": "GRAPH_UPDATED", "summary": "core, then an investigation of it"}

    def assembler(turn, prompt):
        if "## Your task: `why-core`" in prompt:
            dependent_prompts.append(prompt)
            return {"decision": "UNIT_READY", "summary": "found it"}
        summary = "core at tp1.\n\nFor the planner: the conv pool binds 897 slots."
        return {"decision": "UNIT_READY", "summary": summary}

    workflow = _workflow(tmp_path, shell_runner=_binding_shell())
    script = Script(None, planner=planner, assembler=assembler)
    _install(monkeypatch, workflow, script)
    with workflow:
        workflow.run(task_file)

    ws = tmp_path / "ws"
    receipt = r.read(ws / "tasks", "core")
    assert receipt["decision"] == "UNIT_READY" and receipt["summary"] == "core at tp1."
    [prompt] = dependent_prompts
    assert r.HEADING in prompt and "### `core` (assemble): UNIT_READY" in prompt
    assert rf.HEADING in prompt and "- core@tp1:" in prompt and "897 recurrent slots" in prompt
    facts = yaml.safe_load((ws / rf.FACTS_FILENAME).read_text())
    member = facts["members"]["core@tp1"]
    assert member["facts"]["recurrent_slots_per_rank"] == 897
    assert {e["gate"] for e in member["engines"]} == {"boot", "accuracy[gsm8k]"}
    # The first graph is planned before any gate has measured an engine.
    assert rf.HEADING not in planner_prompts[0]
    envelope = json.loads((ws / "tasks" / "why-core.json").read_text())
    assert envelope["receipts"][0]["task"] == "core"


def _snapshots_commit(repo: Path, name: str, text: str) -> str:
    (repo / name).parent.mkdir(parents=True, exist_ok=True)
    (repo / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"change {name}")
    return _git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture
def snapshot_repo(tmp_path) -> Path:
    """A checkout with a package, native code, a quickstart and a built library git ignores."""
    repo = tmp_path / "checkout"
    (repo / "tensorrt_llm" / "libs").mkdir(parents=True)
    (repo / "tensorrt_llm" / "libs" / "libth_common.so").write_bytes(b"\x7fELF")
    (repo / ".gitignore").write_text("*.so\n")
    _git(repo, "init", "-q", ".")
    for name in ("tensorrt_llm/__init__.py", "cpp/kernel.cu", "examples/llm-api/quickstart.py"):
        _snapshots_commit(repo, name, "x = 1\n")
    return repo


@pytest.mark.parametrize("imported_from", ["snapshot", "checkout"])
def test_a_check_measures_one_commit_while_integrations_move_the_checkout(
    tmp_path, snapshot_repo, monkeypatch, imported_from
):
    repo = snapshot_repo
    head = _git(repo, "rev-parse", "HEAD").strip()
    seen: list[str] = []

    def shell(command, env=None, timeout=0):
        seen.append(command)
        if len(seen) == 1:  # a task lands between the legs
            _snapshots_commit(repo, "tensorrt_llm/landed.py", "z = 3\n")
        root = command.split("cd ", 1)[1].split(" ", 1)[0] if imported_from == "snapshot" else repo
        return CommandResult(
            returncode=0,
            stdout=f"[mv2flow-resolve] pid=7 rank=0 module=tensorrt_llm "
            f"origin={root}/tensorrt_llm/__init__.py\n",
        )

    def two_legs(environment, **kwargs):
        for leg in ("A", "B"):
            kwargs["runner"](f"leg {leg}", None, 10)
        return check_result.GateResult(name="boot", member="tp1", passed=True, detail="legs agree")

    monkeypatch.setattr(boot_checks, "run_boot_gate", two_legs)
    monkeypatch.setattr(boot_checks, "resolve_boot_spec", lambda *args: None)
    workflow = _workflow(tmp_path, shell_runner=shell, git_runner=None)
    workflow.task_data = {"trtllm_repo_path": str(repo)}
    workflow.environment = RunEnvironment(kind="local", python="/py", world_size=1)
    workflow._snapshots = snapshots_mod.Snapshots(repo, workflow.workspace / "snapshots")
    marker = workflow._target_marker()
    check = contract_mod.Check("core", "boot", "tp1", contract_mod.TIER_INNER)

    result = CheckRunner(workflow)._run_contract_check(check, None)
    status = contract_mod.ContractStatus.load(workflow.workspace)
    verdict = status.record("core", result, marker="the target when the check was queued")

    snapshot = workflow._snapshots.tree(head)
    assert seen and all(command.startswith(f"cd {snapshot} && ") for command in seen)
    assert verdict.marker == result.marker == marker and verdict.tree_state == head[:7]
    assert not snapshot.exists()  # the checkout moved on and nothing holds it
    journal = workflow.journal_path.read_text()
    if imported_from == "snapshot":
        assert result.passed and verdict.imported == [head, head] and not verdict.void
        return
    # Each leg imported the checkout an integration moved: void, measured once more, never passed.
    assert len(seen) == 2 * (snapshots_mod.VOID_RERUNS + 1) and "void verdict" in journal
    assert not result.passed and not verdict.passed and str(repo) in verdict.void
    assert "[VOID] boot@tp1" in result.summary()
    rendered = contract_mod.evaluate(contract_mod.Contract(checks=(check,)), status).render()
    assert f"- VOID {check.key}: " in rendered


_FILES = {
    "tensorrt_llm/models/target.py": "T = 1\n",
    "tensorrt_llm/README.md": "the package\n",
    "tests/integration/defs/accuracy/references/gsm8k.yaml": "m: [{accuracy: 50}]\n",
    "examples/llm-api/quickstart_advanced.py": "print('q')\n",
}


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / "checkout"
    for name, text in _FILES.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    _git(repo, "init", "-q", ".")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _workflow_marker_workflow(repo: Path) -> TrtllmModelingV2Workflow:
    workflow = TrtllmModelingV2Workflow(workspace=repo / "ws")
    workflow.task_data = {
        "trtllm_repo_path": str(repo),
        "accuracy": {"reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml"},
    }
    return workflow


def _workflow_marker_commit(repo: Path, name: str, text: str) -> None:
    (repo / name).parent.mkdir(parents=True, exist_ok=True)
    (repo / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"change {name}")


@pytest.mark.parametrize(
    "name,seen",
    [
        ("tests/unittest/test_new.py", False),
        ("tensorrt_llm/README.md", False),
        ("examples/llm-api/quickstart_advanced.py", False),
        ("tensorrt_llm/models/target.py", True),
    ],
)
def test_only_a_commit_a_gate_can_see_changes_the_marker(checkout, name, seen):
    workflow = _workflow_marker_workflow(checkout)
    before = workflow._target_marker()
    assert before.startswith("tree:")
    _workflow_marker_commit(checkout, name, "edited\n")
    assert (workflow._target_marker() != before) is seen


def _planned_writer(tmp_path, monkeypatch, decision="LANDED"):
    wf = _workflow(tmp_path)
    wf._turns.keep_records(wf.workspace / "turns")
    execution = ExecutionStage(wf)
    item = QueueItem(n=7, mode="onboard", target="norm/x")
    monkeypatch.setattr(execution, "_queue_item", lambda task: item)
    monkeypatch.setattr(
        worker,
        "_kernel_writer_prompt",
        lambda execution, item, note, task=None: f"`norm/x`\n{note}",
    )
    calls = _install(
        monkeypatch,
        wf,
        Script(None, kernel_writer=lambda turn, prompt: {"decision": decision, "entry": "norm/x"}),
    )
    return execution, calls


def test_failed_receipt_integrates_only_entry_changes(tmp_path, monkeypatch):
    execution, _ = _planned_writer(tmp_path, monkeypatch)
    item = QueueItem(n=7, mode="onboard", target="norm/x")
    task = Task(id="kernel-x", kind="kernel", goal="Certify norm/x", ref="7")
    paths = ["tensorrt_llm/_torch/_experimental/modeling_v2/catalog/norm/x.py"]
    integrated, closed = [], []
    execution._integration = SimpleNamespace(
        changed_paths=lambda task_id: paths,
        branch=lambda task_id: "task/kernel-x",
        integrate=lambda task_id: (
            integrated.append(task_id) or SimpleNamespace(merged=True, commits=1, detail="")
        ),
        close=closed.append,
    )
    assert "are integrated" in worker._land_failed_receipt(execution, item, task, "FAILED_RECEIPT")
    assert integrated == closed == [task.id]
    paths.append("tensorrt_llm/_torch/pyexecutor/executor.py")
    assert "change more" in worker._land_failed_receipt(execution, item, task, "FAILED_RECEIPT")
    assert integrated == closed == [task.id]
    assert worker._land_failed_receipt(execution, item, task, "LANDED") == ""


def test_the_planner_hears_why_an_assembler_blocked_and_a_macros_rows(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run.execution.handoff import TaskResult
    from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import ExecutionStage

    from ._helpers import _workflow

    stage = ExecutionStage(_workflow(tmp_path))
    said = stage._blocked_report(
        {
            "decision": "BLOCKED",
            "blocked_reason": "vocabulary_gap",
            "missing_entry": "attention/x",
            "summary": "the call has no entry",
        }
    )
    assert said == "blocked_reason vocabulary_gap; missing_entry attention/x; the call has no entry"
    rows = stage._macro_rows(
        TaskResult(
            "MACRO",
            data={
                "constituents": [{"target": "gemm/a", "tier": 1}, {"target": "norm/b", "tier": 2}]
            },
        )
    )
    assert "`gemm/a` (tier 1); `norm/b` (tier 2)" in rows
