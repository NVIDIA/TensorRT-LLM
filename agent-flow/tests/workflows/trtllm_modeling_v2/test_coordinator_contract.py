# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The coordinator's contract, case by case, as the verifier of this step wrote it from the design.

Each case states one obligation at a public boundary: the task compiler, the loop, the
landing, `resolve`, the published records. A crash is a raise at a persistence boundary
followed by a reload, as a new controller would do.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import anchors, gate_cli
from agent_flow.workflows.trtllm_modeling_v2.operators import rows
from agent_flow.workflows.trtllm_modeling_v2.run import escalation as esc
from agent_flow.workflows.trtllm_modeling_v2.run import evidence, resolve
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run import orchestrator as orch
from agent_flow.workflows.trtllm_modeling_v2.run.execution import handoff
from agent_flow.workflows.trtllm_modeling_v2.run.execution import tasks as execution
from agent_flow.workflows.trtllm_modeling_v2.run.execution.handoff import TaskResult
from agent_flow.workflows.trtllm_modeling_v2.run.records import report
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import (
    ONBOARDED_FILENAME,
    TrtllmModelingV2Workflow,
)
from agent_flow.workflows.trtllm_modeling_v2.run.workspace.integration import Integration

from ._helpers import _git

_CORE = c.Check("core", "deployment", "tp1")


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "ws"
    path.mkdir()
    return path


@pytest.fixture
def repo(tmp_path):
    """A tiny real repository: one package, one commit."""
    path = tmp_path / "repo"
    (path / "package").mkdir(parents=True)
    (path / "package/__init__.py").write_text("")
    (path / "package/base.py").write_text("VALUE=1\n")
    _git(path, "init", "-q", "-b", "main")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "base")
    return path


def _passed(check):
    return SimpleNamespace(name=check.gate, member=check.member, passed=True, detail="")


def _loop(workspace, tasks=None, *, contract=c.Contract(()), **services):
    """A controller on ``workspace``; with ``tasks`` its graph is written first."""
    if tasks is not None:
        g.Graph(tasks=list(tasks)).save(workspace / g.GRAPH_FILENAME)
    services.setdefault("work", lambda task: "ready")
    services.setdefault("check", _passed)
    return orch.GraphOrchestrator(
        orch.Services(**services),
        contract,
        c.ContractStatus.load(workspace),
        workspace,
        poll_s=0.005,
    )


@pytest.fixture
def blocked(workspace):
    """A graph with one blocked row, and the join that waits for it."""
    task = g.Task("row", "feature", "implement", status="blocked", attempts=2, budget=2,
                  touches=["package/row.py"], check="cpu:row", last="cannot proceed",
                  block={"by": esc.WORKER, "kind": "blocked"})  # fmt: skip
    graph = g.Graph([task, g.Task("join", "check", "join", depends_on=["row"])])
    graph.save(workspace / g.GRAPH_FILENAME)
    return graph


@pytest.fixture
def spec(workspace, repo):
    data = {
        "trtllm_repo_path": str(repo),
        "environment": {"kind": "local", "env": {"VALUE": "old"}},
        "deployments": [{"name": "core", "options": {"modeling_v2": "require"}, "benchmarks": []}],
        "accuracy": {"floor": {"k": 2}},
    }
    (workspace / "task.yaml").write_text(yaml.safe_dump(data))
    return data


# ------------------------------------------------------------- the loop's verdict
def test_a_task_is_recorded_as_running_before_its_work_starts(workspace, monkeypatch):
    started, seen, save = threading.Event(), [], g.Graph.save
    loop = _loop(workspace, [g.Task("row", "feature", "x")],
                 work=lambda task: started.set() or "ready")  # fmt: skip

    def crash_on_running(graph, path):
        if any(t.status == "running" for t in graph.tasks):
            seen.append(started.wait(0.5))
            raise RuntimeError("crash before the running record")
        return save(graph, path)

    monkeypatch.setattr(g.Graph, "save", crash_on_running)
    with pytest.raises(RuntimeError):
        loop.run()
    assert seen == [False]


def test_a_task_whose_worker_blocked_is_not_run_again_after_a_crash(workspace, monkeypatch):
    calls, save = [], g.Graph.save

    def crash_on_blocked(graph, path):
        if any(t.status == "blocked" for t in graph.tasks):
            raise RuntimeError("crash before the block was saved")
        return save(graph, path)

    def work(task):
        calls.append(task.id)
        return "blocked\nneed environment"

    monkeypatch.setattr(g.Graph, "save", crash_on_blocked)
    with pytest.raises(RuntimeError):
        _loop(workspace, [g.Task("row", "feature", "x")], work=work).run()
    monkeypatch.setattr(g.Graph, "save", save)
    assert g.Graph.load(workspace / g.GRAPH_FILENAME).by_id()["row"].status == "running"
    # What is on disk is reconciled before anything is dispatched: the attempt's receipt,
    # written once by its worker, says it blocked.
    left = {"decision": "blocked", "attempt": 1, "observed": "need environment",
            "next_question": "edit-env: which TMPDIR?"}  # fmt: skip
    resumed = _loop(workspace, work=work, receipt=lambda task: left)
    row = resumed.graph.by_id()["row"]
    assert row.status == "blocked" and row.block["question"] == "edit-env: which TMPDIR?"
    assert resumed.run() == "escalated" and calls == ["row"]
    # With no such receipt the attempt runs again, and blocks again.
    other = workspace.parent / "none"
    other.mkdir()
    monkeypatch.setattr(g.Graph, "save", crash_on_blocked)
    with pytest.raises(RuntimeError):
        _loop(other, [g.Task("row", "feature", "x")], work=work).run()
    monkeypatch.setattr(g.Graph, "save", save)
    assert _loop(other, work=work).run() == "escalated" and calls == ["row", "row", "row"]


def test_a_skeleton_that_was_not_committed_stops_the_fan_out_with_the_reason(workspace):
    started = []

    def skeleton(files):
        raise OSError("skeleton not committed")

    loop = _loop(workspace, [g.Task("row", "feature", "x", touches=["package/new/row.py"])],
                 work=lambda task: started.append(task.id) or "ready", skeleton=skeleton)  # fmt: skip
    assert loop.run() == "escalated" and not started
    row = loop.graph.by_id()["row"]
    assert row.status == "blocked" and row.block["kind"] == "skeleton_failed"
    assert "skeleton not committed" in row.last and "`resolve retry row" in row.block["question"]
    # Nothing remembers that it was asked: with the cause gone, a retry asks again and runs.
    asked = []
    esc.leave_answer(workspace, "row", "retry", {"note": "the disk has room again"}, 1)
    again = _loop(workspace, work=lambda task: started.append(task.id) or "ready",
                  skeleton=lambda files: asked.append(list(files)) or "")  # fmt: skip
    assert again.run() == "delivered" and started == ["row"]
    assert asked == [["package/new/row.py"]]


def test_an_onboarded_publication_that_failed_is_asked_again_at_the_next_start(workspace):
    calls = []

    def mark(task):
        calls.append(task.id)
        raise OSError("publication interrupted")

    first = g.Task("session-core", "check", "certify", ref="core", check="deployment:core",
                   satisfies=[_CORE.key])  # fmt: skip
    services = dict(key=lambda check, commit="": "candidate", onboarded=mark,
                    deployments=lambda: [orch.Deployment("core", checks=(_CORE.key,))])  # fmt: skip
    contract = c.Contract((_CORE,))
    assert _loop(workspace, [first], contract=contract, **services).run() == "delivered"
    assert calls == ["session-core"]
    _loop(workspace, contract=contract, **services).run()
    assert calls == ["session-core", "session-core"]


class _Stage(execution.ExecutionStage):
    """What a landing uses of a run, on a real repository.

    The worker is scripted, and the coordinator's own run of a check finds no gate: only
    what is in the ledger counts.
    """

    def __init__(self, workspace, repo, worker, data=None):
        super().__init__(
            SimpleNamespace(
                workspace=workspace,
                task_path=workspace / "task.yaml",
                task_data={"trtllm_repo_path": str(repo), **(data or {})},
                _tls=threading.local(),
                journal=SimpleNamespace(event=lambda *a: None),
                _gate_process=lambda *a: (2, "no evidence"),
            )
        )
        identity = {"GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.invalid"}
        self._integration = Integration(repo, workspace, "test-run", env=identity)
        self.worker, self.turns = worker, 0

    def _work_task_body(self, task):
        self.turns += 1
        return self.worker(self, task)


def _commit(stage, task, path="package/row.py"):
    tree = stage._integration.tree(task.id)
    (tree / path).parent.mkdir(parents=True, exist_ok=True)
    (tree / path).write_text("VALUE=2\n")
    _git(tree, "add", path)
    _git(tree, "commit", "-q", "-m", "row change")


def test_a_finding_of_form_is_told_once_and_then_lands_with_the_work(workspace, repo):
    """Told at hand-in, with one turn beside the budget; it holds no landing and blocks nothing."""
    from agent_flow.workflows.trtllm_modeling_v2.operators import coverage

    target = f"{coverage.MODELING_V2_PACKAGE}/models/fam/target"
    entry = f"{target}/ops/attention/paged_kv_binding.py"
    helper = f"{target}/ops/attention/_paged_kv_binding.py"
    told = []

    def stage_with(worker):
        stage = _Stage(workspace, repo, worker)
        stage.workflow._preflight = SimpleNamespace(target_rel=target)
        return stage

    def stubborn(stage, task):
        told.append(stage._tls.worktree)
        if stage.turns == 1:
            _commit(stage, task, entry)  # a bare module where an entry lies
        return TaskResult("ready")

    # Before the run knows its target there is nothing to hold a file to.
    task = g.Task("state", "state", "x", touches=[target], check="", ref="state", budget=1)
    plain = _Stage(workspace, repo, stubborn)
    assert plain._file_rules(task, plain._integration, repo) == []
    # One attempt was all its budget: the turn it is told in is given beside it.
    stage = stage_with(stubborn)
    result = stage._work_task_turn(task)
    assert stage.turns == 2 and (task.attempts, task.budget) == (2, 2)
    assert "## Your work is accepted as it computes; its form has findings" in told[-1]
    assert f"- entry:{entry}: " in told[-1] and "no OpWrapper subclass" in told[-1]
    assert "test_modeling_v2_paged_kv_binding.py" in told[-1]
    # Where the note asks for the entry's test is in its write scope before it is told:
    # a whole run's task obeyed the rule and was refused by a scope that named another file.
    wanted = (
        f"{coverage.model_ops_tests_dir(target)}/attention/test_modeling_v2_paged_kv_binding.py"
    )
    assert task.touches[0] == target and wanted in task.touches
    # It stands after that turn: the work lands anyway, with the finding said and whose it is.
    assert result.decision == "ready" and (repo / entry).is_file()
    assert "landed with 2 finding(s) of form, which `delivery-files` repairs" in result.note
    # The delivery's task finds it in the tree, with no record kept of it: it is derived.
    delivery = g.Task(rows.DELIVERY_ROW, "feature", "x", touches=[target], ref=rows.DELIVERY_ROW)
    stage.workflow._head = lambda: _git(repo, "rev-parse", "HEAD")
    assert stage._delivery_owed(delivery, repo) == []  # no receipt names where the run began
    (workspace / "tasks").mkdir()
    first = _git(repo, "rev-list", "--max-parents=0", "HEAD").split()[0]
    landed = _git(repo, "rev-list", "--reverse", f"{first}..HEAD").split()[0]
    (workspace / "tasks" / "state.1.receipt.yaml").write_text(
        yaml.safe_dump({"task": "state", "commits": [f"{landed} the entry"]})
    )
    owed = stage._delivery_owed(delivery, repo)
    assert len(owed) == 2 and owed[0].startswith(f"entry:{entry}: ")
    assert "## What the delivery requires of the target's files" in stage._delivery_note(
        delivery, repo
    )

    # The first deployment's assembly, before `onboarded`: the milestone waits for that
    # landing and for nothing of form. No turn: it lands, and the delivery's task is told.
    assembly = g.Task("assemble-core", "assemble", "x", touches=[target], check="", budget=1)
    waited = stage_with(lambda stage, task: _commit(stage, task, f"{target}/ops/gemm/proj.py")
                        or TaskResult("ready"))  # fmt: skip
    waited._holds_onboarded = lambda task_id: task_id == "assemble-core"
    result = waited._work_task_turn(assembly)
    assert waited.turns == 1 and (assembly.attempts, assembly.budget) == (1, 1)
    assert result.decision == "ready" and (repo / target / "ops/gemm/proj.py").is_file()
    assert "finding(s) of form, not told first (`onboarded` waits for this landing)" in result.note

    # A finding is said when the work has landed. A check that does not pass lands
    # nothing: its failure is all the task hears, and all an escalation would quote.
    unproven = g.Task("unproven", "weights", "x", touches=[target], check="cpu:unproven")

    def once(stage, task):
        if stage.turns == 1:
            _commit(stage, task, f"{target}/ops/moe/route.py")
        return TaskResult("ready")

    result = stage_with(once)._work_task_turn(unproven)
    assert result.decision != "ready" and not (repo / target / "ops/moe/route.py").exists()
    assert "finding(s) of form" not in result.note

    # A worker that puts it right in its turn lands with no finding.
    def repairs(stage, task):
        tree = stage._integration.tree(task.id)
        if stage.turns == 1:
            _commit(stage, task, f"{target}/ops/norm/bare.py")
        else:
            _git(tree, "mv", f"{target}/ops/norm/bare.py", f"{target}/ops/norm/_bare.py")
            _git(tree, "commit", "-q", "-m", "a helper, named as one")
        return TaskResult("ready")

    other = g.Task("other", "state", "x", touches=[target], check="", ref="other")
    fixed = stage_with(repairs)
    result = fixed._work_task_turn(other)
    assert fixed.turns == 2 and result.decision == "ready" and "finding" not in result.note
    assert (repo / target / "ops/norm/_bare.py").is_file()
    # A helper beside the entries is none: one turn, no word.
    quiet = g.Task("helper", "state", "x", touches=[target], check="", ref="helper")
    silent = stage_with(lambda stage, task: _commit(stage, task, helper) or TaskResult("ready"))
    assert silent._work_task_turn(quiet).decision == "ready" and silent.turns == 1


def test_a_targets_import_of_modeling_v1_code_is_a_finding_of_form_and_a_requirement(tmp_path):
    """The guide: a target reads and may copy Modeling V1 code; it does not import, subclass or call it."""
    from agent_flow.workflows.trtllm_modeling_v2.checks import delivery
    from agent_flow.workflows.trtllm_modeling_v2.spec import task_schema

    from .test_v1_rule import _checkout

    repo = _checkout(tmp_path)
    target = "engine/core/v2/models/toy/toy_1b"
    (repo / target / "copied.py").write_text("class Linear:\n    pass  # read, and copied\n")
    (repo / target / "unsure.py").write_text("from engine.core.modules.linear import split\n")
    files = [f"{target}/{name}" for name in ("modeling.py", "weights.py", "copied.py", "unsure.py")]
    checks = {c.name: c for c in delivery.check_v1_imports(repo, [*files, f"{target}/gone.py"])}
    # One failing check for the file that imports what the guide names; a copy has none.
    broken = checks[f"imports:{target}/modeling.py"]
    assert (
        not broken.passed
        and "`ToyForCausalLM` from `engine.core.models.modeling_toy`" in broken.detail
    )
    assert "may copy it into its own operations; it does not import" in broken.detail
    # A name from a listed place that is none of the kinds the guide names: said, not failed.
    unsure = checks[f"imports:{target}/unsure.py"]
    assert unsure.passed and unsure.detail.startswith("not certain, for a person: ")
    assert set(checks) == {broken.name, unsure.name}
    # Told when the task hands in, and landed with the work, as every finding of form.
    for command in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", "base"]):
        _git(repo, *command)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    told = []

    def worker(stage, task):
        told.append(stage._tls.worktree)
        if stage.turns == 1:
            tree = stage._integration.tree(task.id)
            (tree / target / "modeling.py").write_text(
                (tree / target / "modeling.py").read_text() + "\nX = 1\n"
            )
            _git(tree, "commit", "-q", "-a", "-m", "the model")
        return TaskResult("ready")

    def landed(data, name):
        stage = _Stage(workspace, repo, worker, data)
        stage.workflow._preflight = SimpleNamespace(target_rel=target)
        task = g.Task(name, "assemble", "x", touches=[target], check="", ref=name)
        result = stage._work_task_turn(task)
        return stage, result

    stage, result = landed({}, "first")
    assert stage.turns == 2 and result.decision == "ready"
    assert f"- imports:{target}/modeling.py: " in told[-1]
    assert "landed with 1 finding(s) of form" in result.note and "imports:" in result.note
    # A requirement like any other: the spec says it, and the outer session may change it.
    assert delivery.v1_imports_refused({}) and delivery.v1_imports_refused({"target": {}})
    waived = {"target": {"v1_imports": "allowed"}}
    assert not delivery.v1_imports_refused(waived)
    spec = {"target": {}}
    esc.set_spec(spec, "target.v1_imports", "allowed")
    assert spec == waived
    stage, result = landed(waived, "second")
    assert stage.turns == 1 and "finding" not in result.note
    errors = []
    task_schema._validate_target({"v1_imports": "sometimes"}, errors)
    assert errors and "`refused`" in errors[0] and "`allowed`" in errors[0]
    task_schema._validate_target({"family": "toy", "v1_imports": "allowed"}, ok := [])
    assert ok == []


def test_the_deliverys_task_takes_no_turn_when_nothing_is_open_and_is_told_what_stands(
    workspace, repo
):
    """Its check is the delivery's own: the test is where the delivery looks, and a list names it."""
    from agent_flow.workflows.trtllm_modeling_v2.operators import coverage

    target = f"{coverage.MODELING_V2_PACKAGE}/models/fam/target"
    # The anchor is declared with its source, as the intake guide says to where the tree
    # has no reference row for the model.
    declared = {"name": "gsm8k", "anchor": 41.5, "source": "the report of run 7,\n full split"}
    scored = {"name": "core", "options": {}, "benchmarks": [declared]}
    spec = {"deployments": [scored], "accuracy": {"model_key": "org/model"}}
    told = []

    def stage_with(worker, data):
        stage = _Stage(workspace, repo, worker, data)
        stage.workflow._preflight = SimpleNamespace(target_rel=target, family="fam")
        stage.workflow._head = lambda: _git(repo, "rev-parse", "HEAD")
        return stage

    def write(stage, task, files):
        told.append(stage._tls.worktree)
        tree = stage._integration.tree(task.id)
        for path, text in files.items():
            (tree / path).parent.mkdir(parents=True, exist_ok=True)
            (tree / path).write_text(text)
        _git(tree, "add", "-A")
        if _git(tree, "status", "--porcelain"):
            _git(tree, "commit", "-q", "-m", "the accuracy test for CI")

    # A run none of whose deployments scores a benchmark owes no CI file. With no finding
    # recorded either, its delivery task is done as it starts: no turn, no tree.
    unscored = stage_with(lambda stage, task: TaskResult("ready"), {"deployments": [
        {"name": "core", "options": {}}]})  # fmt: skip
    assert unscored._ci_files() == ()
    idle = g.Task(rows.DELIVERY_ROW, "feature", "x", touches=[target], ref=rows.DELIVERY_ROW)
    result = unscored._work_task_turn(idle)
    assert result.decision == "ready" and unscored.turns == 0
    assert "nothing the delivery requires of the target's files is open" in result.note

    stage = stage_with(None, spec)
    test, lists, reference = stage._ci_files()
    assert test == "tests/integration/defs/accuracy/test_modeling_v2_fam.py"
    assert lists == "tests/integration/test_lists/test-db"
    assert reference == "tests/integration/defs/accuracy/references/gsm8k.yaml"
    # One list of rules: what the delivery's check fails for CI on this tree is what its
    # task is owed, in the same words. A whole run's delivery asked for the anchor's row,
    # which its task was never told of and could not write inside its scope.
    from agent_flow.workflows.trtllm_modeling_v2.checks import delivery as delivery_rules
    from agent_flow.workflows.trtllm_modeling_v2.run.validation import completion

    final = delivery_rules.verify_delivery(
        repo_root=repo, target_rel=target, family="fam", **completion.ci_arguments(stage.workflow)
    )
    asked = [
        f"{c.name}: {c.detail}"
        for c in final.checks
        if c.name in completion.CI_DELIVERY_CHECKS and not c.passed
    ]
    made = g.Task(rows.DELIVERY_ROW, "feature", "x", touches=[test, lists, reference, target],
                  ref=rows.DELIVERY_ROW)  # fmt: skip
    assert stage._delivery_owed(made, repo) == asked and len(asked) == 3
    # For a declared anchor the row is known, and the rule's own line says it: the key,
    # the score, the count and the source to keep as a comment.
    (row,) = [line for line in asked if line.startswith("accuracy:reference: ")]
    assert (
        f"missing: {repo / reference}. The spec declares it: add under the key `org/model` " in row
    )
    assert (
        "the entry `accuracy: 41.5` for `core`, with a comment above it that keeps the count "
        in row
    )
    assert "(1319 samples) and the source (the report of run 7, full split)" in row

    def task():
        return g.Task(rows.DELIVERY_ROW, "feature", "x", touches=[test, lists, reference, target],
                      ref=rows.DELIVERY_ROW)  # fmt: skip

    # The run owes the CI files and the tree has neither: its dispatch says what stands.
    # It writes the test alone, is told once, and what stands lands with it: the delivery
    # requires it at the end, and nothing before that is held for it.
    alone = stage_with(
        lambda stage, task: write(stage, task, {test: "T = 1\n"}) or TaskResult("ready"),
        spec,
    )
    result = alone._work_task_turn(task())
    assert alone.turns == 2 and result.decision == "ready" and (repo / test).is_file()
    assert "## What the delivery requires of the target's files" in told[0]
    assert "- tests:accuracy-file: missing: " in told[0] and "- tests:test-db: " in told[0]
    # Told of the anchor's row with the rule's own words, asked of its tree.
    assert "- accuracy:reference: missing: " in told[0]
    assert row.partition(". The spec declares it: ")[2] in told[0]
    assert "its form has findings" in told[1] and "CI would never run it" in told[1]
    assert "landed with 2 finding(s) of form" in result.note and "tests:test-db" in result.note
    # The entry is added: nothing stands, and the next start of the task takes no turn.
    entry = "- condition: {}\n  tests:\n  - accuracy/test_modeling_v2_fam.py\n"
    anchor = "org/model:\n  # 1319 samples; the report of run 7, full split\n  - accuracy: 41.5\n"
    both = stage_with(
        lambda stage, task: (
            write(stage, task, {f"{lists}/l0.yml": entry, reference: anchor}) or TaskResult("ready")
        ),
        spec,
    )
    result = both._work_task_turn(task())
    assert both.turns == 1 and result.decision == "ready" and "finding" not in result.note
    done = stage_with(lambda stage, task: TaskResult("ready"), spec)
    assert done._work_task_turn(task()).decision == "ready" and done.turns == 0
    # Sent back by the outer session with a note, it takes a turn whatever it finds open
    # itself: the note is the reason. A whole run's `retry run --task delivery-files
    # --note <the row to add>` started attempt 2 and closed it at once, and the note
    # reached nobody.
    sent = task()
    sent.attempts, sent.last = 2, f"{esc.ANSWERED} `run` with `retry`: add the row of the anchor"
    assert esc.sent_back(sent) and done._delivery_owed(sent, repo) == []
    result = done._work_task_turn(sent)
    assert done.turns == 1 and result.decision == "ready" and "no turn" not in result.note
    # An attempt the loop itself started again finds nothing open, and takes none.
    sent.last = "its checks failed: core:boot@tp1"
    assert done._work_task_turn(sent).decision == "ready" and done.turns == 1


def test_a_commit_outside_the_write_scope_goes_back_once_on_the_tasks_own_budget(workspace, repo):
    def worker(stage, task):
        if stage.turns == 1:
            _commit(stage, task, "package/foreign.py")
        return TaskResult("ready")

    task = g.Task("row", "feature", "x", touches=["package/row.py"], check="cpu:row")
    stage = _Stage(workspace, repo, worker)
    result = stage._work_task_turn(task)
    assert (stage.turns, result.decision, task.attempts) == (2, "blocked", 2)
    assert not (repo / "package/foreign.py").exists()
    # With one attempt, there is none to send it back with: one counter, not two.
    other = workspace.parent / "one"
    other.mkdir()
    only = g.Task("solo", "feature", "x", touches=["package/row.py"], check="cpu:solo", budget=1)
    stage = _Stage(other, repo, worker)
    result = stage._work_task_turn(only)
    assert (stage.turns, result.decision, only.attempts) == (1, "blocked", 1)
    assert "with no attempt left" in result.data["observed"]


def test_the_final_tests_are_selected_from_the_change_and_the_tasks_and_state_their_devices(
    workspace, repo, monkeypatch
):
    from agent_flow.workflows.trtllm_modeling_v2.run import keys
    from agent_flow.workflows.trtllm_modeling_v2.run.validation import regression

    # Before the run: the package's suite holds another model's test.
    suite = f"{regression.PACKAGE_TESTS[0]}/test_modeling_v2_gemm.py"
    (repo / suite).parent.mkdir(parents=True)
    (repo / suite).write_text("")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "the package's suite")
    base = _git(repo, "rev-parse", "HEAD")
    changed = {
        "package/target/ops/rms.py": "R = 1\n",
        "tests/target/test_modeling_v2_rms.py": "",
        "package/shared/norm.py": "N = 1\n",
        "tests/shared/test_modeling_v2_norm.py": "",
        "docs/guide.md": "words\n",
    }
    for path, text in changed.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "what the run's tasks landed")
    head = _git(repo, "rev-parse", "HEAD")
    # What the run knows of it: the receipts name its first commit, the graph its tasks and
    # what each was given, the rows the devices each one's check states.
    (workspace / "tasks").mkdir()
    (workspace / "tasks" / "k1.1.receipt.yaml").write_text(
        yaml.safe_dump({"task": "k1", "commits": [f"{head} what the run's tasks landed"]})
    )
    kernel = g.Task("k1", "kernel", "an entry", status="done", ref="1",
                    touches=["package/target/ops/rms.py", "tests/target/test_modeling_v2_rms.py"])  # fmt: skip
    shared = g.Task("norm", "feature", "a shared norm", status="done", ref="norm",
                    touches=["package/shared/norm.py", "tests/shared/test_modeling_v2_norm.py"])  # fmt: skip
    skipped = g.Task("extra", "feature", "left out", status="skipped", ref="extra",
                     touches=["package/shared/extra.py", "tests/shared/test_modeling_v2_extra.py"])  # fmt: skip
    g.Graph([kernel, shared, skipped]).save(workspace / g.GRAPH_FILENAME)
    rows.save_work(
        workspace,
        [rows.Row("1", "operator", "an entry", gpus=2), rows.Row("norm", "feature", "a norm")],
    )
    stage = _Stage(workspace, repo, lambda stage, task: TaskResult("ready"))
    stage._own_paths = ("package/target", "tests/target")
    workflow = stage.workflow
    workflow._head = lambda: _git(repo, "rev-parse", "HEAD")
    workflow._key = lambda check_id, commit="", extra=None: keys.key(
        workflow.task_data, workspace, commit, check_id, inputs_of=lambda _: "", extra=extra
    )
    ran = []

    def run_final_tests(**said):
        ran.append(said)
        return regression.TestsResult(
            said["head"], said["base"], "CLEAN", [], [], len(said["files"]), 1, said["scope"],
            said["run_job"].gpus, said["run_job"].nodes,
        )  # fmt: skip

    def job_runner(workflow, name, gpus=1, nodes=0, identity=""):
        job = SimpleNamespace(name=name, gpus=gpus, nodes=nodes, identity=identity)
        return None, job, "python"

    monkeypatch.setattr(regression, "run_final_tests", run_final_tests)
    monkeypatch.setattr(regression, "job_runner", job_runner)
    result = stage._final_tests(head)
    (said,) = ran
    # The target's own tests and the tests of the task that was given the shared file: its
    # reach is known, so the package's suite is not run. Compared with where the run began.
    assert said["files"] == [
        "tests/shared/test_modeling_v2_norm.py",
        "tests/target/test_modeling_v2_rms.py",
    ]
    assert (said["head"], said["base"]) == (head, base) and "collected suite" not in said["scope"]
    # One job name, and the devices the rows state: the kernel row's two, not a node's.
    assert (said["run_job"].name, said["run_job"].gpus) == ("final-tests", 2)
    assert result.passed and result.gpus == 2 and said["run_job"].nodes == 0
    # Each job carries what its results are kept under beside the commit: the native build
    # and the runtime. Nothing is missing.
    assert len(said["run_job"].identity) == 16
    assert said["missing"] == []
    # Its result is a row of the ledger under the tests' key on this candidate and selection.
    (row,) = [v for v in evidence.verdicts(workspace) if v.gate == "final-tests"]
    assert row.passed and row.marker.startswith("key:") and not row.unverified
    # The same candidate and selection: not run a second time.
    again = stage._final_tests(head)
    assert len(ran) == 1 and again.passed and again.scope == result.scope
    # A shared file no task with tests of its own was given: its reach is unknown, and the
    # package's collected suite runs with the rest, once.
    (repo / "package/shared/other.py").write_text("O = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "a late change to shared code")
    later = _git(repo, "rev-parse", "HEAD")
    assert stage._final_tests(later).passed and len(ran) == 2
    assert suite in ran[1]["files"] and "collected suite, once" in ran[1]["scope"]
    assert "package/shared/other.py is unknown" in ran[1]["scope"]
    assert stage._final_tests(later).passed and len(ran) == 2
    # Another runtime is another identity and another key: the passed row is not this one's.
    workflow.task_data["environment"] = {"python": "/opt/other/python"}
    assert stage._final_tests(later).passed and len(ran) == 3
    assert ran[2]["run_job"].identity != ran[1]["run_job"].identity
    # A row that states whole nodes: the final tests' job states them too, as its own
    # check did. And a test a done task declared that the candidate does not hold is
    # named, not dropped.
    wide = g.Task("wide", "feature", "a two-node path", status="done", ref="wide",
                  touches=["package/target/wide.py", "tests/target/test_modeling_v2_wide.py"])  # fmt: skip
    g.Graph([kernel, shared, skipped, wide]).save(workspace / g.GRAPH_FILENAME)
    rows.save_work(workspace, [rows.Row("wide", "feature", "a two-node path", nodes=2)])
    assert stage._final_tests(later).passed and len(ran) == 4
    assert (ran[3]["run_job"].nodes, ran[3]["run_job"].gpus) == (2, 2)
    assert ran[3]["missing"] == ["tests/target/test_modeling_v2_wide.py"]


_HARNESS_TEST = """from harness import LlmapiAccuracyTestHarness


class TestTarget(LlmapiAccuracyTestHarness):
    def test_{benchmark}(self):
        {task}(self.MODEL_NAME).evaluate(self.llm)
"""


def test_an_accuracy_test_a_task_names_is_left_to_ci_with_what_a_session_measured(
    workspace, repo, monkeypatch
):
    """Each of its cases is a load and a whole benchmark, which the session measured."""
    from agent_flow.workflows.trtllm_modeling_v2.run import keys
    from agent_flow.workflows.trtllm_modeling_v2.run.validation import regression

    base = _git(repo, "rev-parse", "HEAD")
    # Told by the class it is built on, wherever it is: no path of the engine is named.
    scored, other = "ci/defs/test_target_accuracy.py", "ci/defs/test_target_mmlu.py"
    landed = {
        "package/target/ops/rms.py": "R = 1\n",
        "tests/target/test_modeling_v2_rms.py": "def test_rms():\n    pass\n",
        scored: _HARNESS_TEST.format(benchmark="gsm8k", task="GSM8K"),
        other: _HARNESS_TEST.format(benchmark="mmlu", task="MMLU"),
    }
    for path, text in landed.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "what the run's tasks landed")
    head = _git(repo, "rev-parse", "HEAD")
    (workspace / "tasks").mkdir()
    (workspace / "tasks" / "k1.1.receipt.yaml").write_text(
        yaml.safe_dump({"task": "k1", "commits": [f"{head} what the run's tasks landed"]})
    )
    row = g.Task("delivery-files", "feature", "the delivered accuracy test", status="done",
                 ref="delivery-files", touches=[scored, other])  # fmt: skip
    g.Graph([row]).save(workspace / g.GRAPH_FILENAME)
    assert sorted(regression.accuracy_tests(repo, head, sorted(landed))) == [scored, other]
    data = {"deployments": [{"name": "tp1", "options": {}, "timeout": 900}]}
    stage = _Stage(workspace, repo, lambda stage, task: TaskResult("ready"), data)
    stage._own_paths = ("package/target", "tests/target")
    workflow = stage.workflow
    workflow._head = lambda: head
    workflow._key = lambda check_id, commit="", extra=None: keys.key(
        workflow.task_data, workspace, commit, check_id, inputs_of=lambda _: "", extra=extra
    )
    # What the ledger holds: the session of `tp1` measured gsm8k under its key on the
    # candidate; a line under another key measured other content and stands for nothing.
    workflow._contract = c.Contract((c.Check("tp1", "deployment", "tp1"),))
    workflow._check_key = lambda check, commit="": f"key:{check.unit}-on-{commit[:6]}"
    status = c.ContractStatus.load(workspace)
    workflow._contract_status = lambda: status
    here = f"key:tp1-on-{head[:6]}"
    score = SimpleNamespace(name="accuracy[gsm8k]", member="tp1", passed=True, detail="41.9")
    status.record("tp1", score, marker=here)
    old = SimpleNamespace(name="accuracy[mmlu]", member="tp1", passed=True, detail="70")
    status.record("tp1", old, marker="key:tp1-on-another")
    assert stage._measured(head) == {"gsm8k": ["tp1"]}
    ran = []

    def run_final_tests(**said):
        ran.append(said)
        return regression.TestsResult(
            said["head"], said["base"], "CLEAN", jobs=1, left=[f"{scored}: left to CI"]
        )

    monkeypatch.setattr(regression, "run_final_tests", run_final_tests)
    monkeypatch.setattr(
        regression, "job_runner", lambda *a, **kw: (None, SimpleNamespace(gpus=1, nodes=0), "py")
    )
    assert stage._final_tests(head).passed
    # Both stay in the selection (a task names them), marked as collected and not run:
    # one with what the session measured, one with nothing measured here.
    assert ran[0]["base"] == base and {scored, other} <= set(ran[0]["files"])
    assert ran[0]["ci"] == {scored: "the session measured `gsm8k` on `tp1`", other: ""}
    # Their row is the one record of their result: a pass on the same candidate and
    # selection reads it there, with what was left to CI, and runs nothing.
    again = stage._final_tests(head)
    assert len(ran) == 1 and again.passed and again.left == [f"{scored}: left to CI"]


def test_a_repository_with_no_test_to_run_is_not_verified_never_passed(workspace, repo):
    from agent_flow.workflows.trtllm_modeling_v2.run.validation import regression

    g.Graph([g.Task("row", "feature", "x", status="done", ref="row",
                    touches=["package/target/row.py", "tests/target/test_modeling_v2_row.py"])]
            ).save(workspace / g.GRAPH_FILENAME)  # fmt: skip
    stage = _Stage(workspace, repo, lambda stage, task: TaskResult("ready"))
    stage._own_paths = ("package/target", "tests/target")
    stage.workflow._head = lambda: _git(repo, "rev-parse", "HEAD")
    result = stage._final_tests(_git(repo, "rev-parse", "HEAD"))
    assert result.verdict == regression.NOT_VERIFIED and result.partial and not result.passed
    assert result.unverified == [
        "tests/target/test_modeling_v2_row.py is declared by a task and is not in the candidate"
    ]
    # With nothing declared either: it says that there is nothing to run.
    g.Graph([]).save(workspace / g.GRAPH_FILENAME)
    assert stage._final_tests(_git(repo, "rev-parse", "HEAD")).unverified == [
        "no test file is selected: the target has no test to run"
    ]


def test_a_pass_judged_by_other_options_does_not_certify_the_new_ones(workspace, repo):
    def worker(stage, task):
        _commit(stage, task)
        stage.workflow.task_data["deployments"] = [
            {"name": "core", "options": {"max_batch_size": 1}}
        ]
        pin = stage._integration.pin(task.id)
        marker = evidence.key_of(stage.workflow.task_data, workspace, pin.commit, "deployment:core")
        old = SimpleNamespace(name="deployment", member="tp1", passed=True, detail="batch 1")
        c.ContractStatus.load(workspace).record("core", old, marker=marker, inputs="old-options")
        stage.workflow.task_data["deployments"][0]["options"]["max_batch_size"] = 64
        return TaskResult("ready")

    task = g.Task("row", "assemble", "x", touches=["package/row.py"], check="deployment:core")
    result = _Stage(workspace, repo, worker)._work_task_turn(task)
    assert result.decision != "ready" and not (repo / "package/row.py").exists()


def test_a_rows_devices_nodes_and_wall_reach_its_job(workspace, repo):
    wide = rows.Row("op", "operator", "verify", nodes=2, wall=37, check="kernel:op")
    two = rows.Row("r", "operator", "verify", gpus=2, wall=37, check="kernel:r")
    rows.save_work(workspace, [wide, two])
    stage = _Stage(workspace, repo, lambda *a: TaskResult("ready"))
    for row, option in ((wide, "--nodes"), (two, "--gpus")):
        task = g.Task(row.id, "kernel", "verify", ref=row.id, check=row.check)
        argv = execution.check_argv(row.check, **stage._check_places(task, tree="/tree"))
        assert argv[argv.index(option) + 1] == "2" and argv[argv.index("--timeout") + 1] == "37"
        # The front door takes what the card says.
        assert gate_cli._parse(argv).name == "kernel"


def test_a_live_spec_change_refreshes_the_contract_the_run_executes(workspace, repo):
    workflow = TrtllmModelingV2Workflow(workspace=workspace)
    workflow.task_data = {
        "trtllm_repo_path": str(repo),
        "checkpoint_path": str(repo),
        "environment": {"kind": "local"},
        "deployments": [{"name": "core", "options": {"max_batch_size": 1}}],
    }
    workflow.task_path = workspace / "task.yaml"
    workflow._contract = c.compile_contract(workflow.task_data)
    before = workflow._contract.checks[0].inputs
    workflow.task_data["deployments"][0]["options"]["max_batch_size"] = 64
    workflow._respec(["deployments.core.options.max_batch_size"])
    actual = c.compile_contract(workflow.task_data).checks[0].inputs
    assert actual != before and workflow._contract.checks[0].inputs == actual
    # The contract's file follows, so a check run now is recorded with what it is judged by.
    written = yaml.safe_load((workspace / c.CONTRACT_FILENAME).read_text())
    assert written["checks"][0]["inputs"] == actual
    # And a controller that runs is given it.
    loop = _loop(workspace, [], contract=c.Contract(()))
    loop.recontract(workflow._contract)
    assert loop.contract is workflow._contract
    assert loop.validation["required_checks"] == [k.key for k in workflow._contract.required()]


# ------------------------------------------------- answers: whole, once, and crash-atomic
def _answer(workspace, graph, verb, args, name="row", hooks=None, save=None):
    """Leave an answer and apply what waits, as a controller's pass does; what it became."""
    task = graph.by_id().get(name)
    esc.leave_answer(workspace, name, verb, args, task.attempts if task else 0)
    esc.apply_pending(
        workspace,
        graph,
        hooks or resolve.offline_hooks(workspace, graph),
        save or (lambda: graph.save(workspace / g.GRAPH_FILENAME)),
    )
    return esc.read_answer(workspace, name)


_CHILD = {"id": "child", "kind": "feature", "outcome": "part", "required": ["proof"],
          "writes": ["package/child.py"]}  # fmt: skip


@pytest.mark.parametrize(
    "verb,args",
    [
        ("retry", {"note": "retry safely"}),
        ("skip", {"note": "scope revised"}),
        ("edit-env", {"set": {"environment.env.VALUE": "new"}, "note": "fix environment"}),
        ("allow-shared", {"path": "shared/", "note": "needed"}),
        ("change-requirement", {"set": {"deployments.core.options.modeling_v2": "off"},
                                "note": "incorrect requirement"}),
        ("pool-retry", {"note": "queue recovered"}),
        ("pool-adjust", {"set": {"qos": "QOS"}, "note": "site revised"}),
        ("split", {"rows": [_CHILD], "note": "smaller work"}),
    ],
)  # fmt: skip
def test_each_verb_applies_once_and_a_restart_finds_nothing_to_apply(
    workspace, blocked, spec, verb, args
):
    hooks = resolve.offline_hooks(workspace, blocked)
    hooks.pool_retry = lambda *a: "retried"
    hooks.pool_adjust = lambda: "adjusted"
    name = "pool" if verb.startswith("pool-") else "row"
    assert _answer(workspace, blocked, verb, args, name, hooks)["applied"]
    before = (workspace / g.GRAPH_FILENAME).read_bytes()
    restarted = g.Graph.load(workspace / g.GRAPH_FILENAME)
    assert esc.apply_pending(workspace, restarted, hooks, lambda: None) == []
    assert (workspace / g.GRAPH_FILENAME).read_bytes() == before
    if verb == "split":
        parent = restarted.by_id()["row"]
        assert parent.kind == "check" and "child" in parent.depends_on and not parent.touches
        assert {r.id for r in rows.load_work(workspace)} == {"child"}
    if verb == "change-requirement":
        (change,) = esc.spec_changes(workspace)
        assert (change["from"], change["to"], change["why"]) == (
            "require", "off", "incorrect requirement",
        )  # fmt: skip


@pytest.mark.parametrize(
    "verb,changes",
    [
        ("change-requirement", {"accuracy.floor.k": 3, "deployments.no-such.options.x": 1}),
        ("edit-env", {"environment.env.VALUE": "new", "deployments.missing.env.X": "bad"}),
    ],
)
def test_a_set_of_keys_is_changed_whole_or_not_at_all(workspace, blocked, spec, verb, changes):
    import copy

    original = copy.deepcopy(spec)
    settled = _answer(
        workspace, blocked, verb, {"set": changes, "note": "revision"},
        hooks=esc.Hooks(task_data=spec),
    )  # fmt: skip
    assert not settled["applied"]
    assert spec == original and not esc.spec_changes(workspace)
    # Nothing of it comes back when the spec is read anew at the next start.
    esc.reapply_changes(workspace, original)
    assert original == spec and blocked.by_id()["row"].status == "blocked"


def test_a_refused_retry_changes_nothing(workspace, blocked):
    before = blocked.by_id()["row"].as_dict()
    settled = _answer(
        workspace, blocked, "retry", {"task": "absent", "note": "fix"}, hooks=esc.Hooks()
    )
    assert not settled["applied"] and blocked.by_id()["row"].as_dict() == before


def test_an_anchor_change_is_recorded_once_when_the_graph_save_after_it_was_lost(
    workspace, blocked, spec
):
    anchors.change(workspace, "core", "gsm8k", value=90, reason="original")
    args = {
        "deployment": "core",
        "benchmark": "gsm8k",
        "anchor": 20,
        "note": "corrected protocol",
    }

    def lost():
        raise OSError("save failed")

    with pytest.raises(OSError):
        _answer(workspace, blocked, "change-requirement", args, save=lost)
    # The answer was not settled: the process that starts next finds it and the task as
    # they were, applies it, and the change it had already recorded is not recorded twice.
    restarted = g.Graph.load(workspace / g.GRAPH_FILENAME)
    (said,) = esc.apply_pending(
        workspace,
        restarted,
        resolve.offline_hooks(workspace, restarted),
        lambda: restarted.save(workspace / g.GRAPH_FILENAME),
    )
    assert "was not applied" not in said
    made = [x for x in anchors.changes(workspace) if x["reason"] == "corrected protocol"]
    assert len(made) == 1 and restarted.by_id()["row"].status == "pending"


def test_a_requirement_changed_on_another_deployment_reopens_that_deployments_session(
    workspace, blocked, spec
):
    target = g.Task("session-core", "check", "certify", status="done", ref="core",
                    check="deployment:core", satisfies=["core:deployment@tp1"])  # fmt: skip
    blocked.tasks.append(target)
    blocked.save(workspace / g.GRAPH_FILENAME)
    old = SimpleNamespace(name="deployment", member="tp1", passed=True, detail="old options")
    c.ContractStatus.load(workspace).record("core", old, marker="candidate")
    settled = _answer(workspace, blocked, "change-requirement",
                      {"set": {"deployments.core.options.modeling_v2": "off"},
                       "note": "corrected requirement"})  # fmt: skip
    assert settled["applied"]
    latest = evidence.lookup(workspace, "deployment:core", "candidate")
    assert target.status == "pending" and latest is not None and not latest.passed


def test_a_receipt_is_published_whole_and_once(workspace):
    # Of two writers of one attempt's receipt, the first to publish stands, whole.
    receipts = [{"task": "row", "attempt": 1, "decision": "ready", "summary": text}
                for text in ("first", "second")]  # fmt: skip
    gate = threading.Barrier(2)

    def write(row):
        gate.wait(5)
        handoff.write(workspace / "tasks", row)

    threads = [threading.Thread(target=write, args=(row,)) for row in receipts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=6)
    kept = handoff.read(workspace / "tasks", "row", 1)
    assert kept in receipts
    handoff.write(workspace / "tasks", {**receipts[0], "summary": "later"})
    assert handoff.read(workspace / "tasks", "row", 1) == kept
    assert not list((workspace / "tasks").glob(".*"))


def test_a_split_keeps_or_refuses_every_dependency_and_a_refused_one_keeps_no_row(
    workspace, blocked, spec
):
    # A child that waits for what waits for its parent: the edge is kept, so it is refused.
    cyclic = [{"id": "child", "kind": "feature", "outcome": "x", "required": ["x"],
               "depends_on": ["join"]}]  # fmt: skip
    settled = _answer(workspace, blocked, "split", {"rows": cyclic, "note": "bad"})
    assert not settled["applied"]
    assert rows.row_of(workspace, "child") is None and "child" not in blocked.by_id()
    # A cycle among the children themselves: refused, and no row of it is kept.
    pair = [{"id": a, "kind": "feature", "outcome": a, "required": [a], "depends_on": [b]}
            for a, b in (("child-a", "child-b"), ("child-b", "child-a"))]  # fmt: skip
    settled = _answer(workspace, blocked, "split", {"rows": pair, "note": "split"})
    assert not settled["applied"] and not rows.load_work(workspace)
    # An admissible one: the parent is the join, its rows are kept, and a restart finds both.
    good = [{"id": "child-a", "kind": "feature", "outcome": "a", "required": ["a"],
             "writes": ["package/a.py"]},
            {"id": "child-b", "kind": "feature", "outcome": "b", "required": ["b"],
             "depends_on": ["child-a"], "writes": ["package/b.py"]}]  # fmt: skip
    blocked.by_id()["row"].satisfies = ["core:deployment@tp1"]
    settled = _answer(workspace, blocked, "split", {"rows": good, "note": "split"})
    assert settled["applied"]
    reloaded = g.Graph.load(workspace / g.GRAPH_FILENAME).by_id()
    assert reloaded["row"].kind == "check" and reloaded["row"].satisfies == ["core:deployment@tp1"]
    assert reloaded["row"].depends_on == ["child-a", "child-b"]
    assert reloaded["child-b"].depends_on == ["child-a"]
    assert {r.id for r in rows.load_work(workspace)} == {"child-a", "child-b"}


# ------------------------------------------------- the compiler: typed rows, deltas, files
def test_what_a_split_leaves_of_its_task_is_a_join_that_runs_nothing(workspace):
    """A `check` task with no check of its own waits for what it depends on and is done as it starts."""
    ran = []
    tasks = [g.Task("row", "feature", "x"),
             g.Task("join", g.CHECK_KIND, "the rows of a split", depends_on=["row"])]  # fmt: skip
    loop = _loop(workspace, tasks, work=lambda task: ran.append(task.id) or "ready")
    assert loop.run() == "delivered" and ran == ["row"]
    assert loop.graph.by_id()["join"].status == "done"


def test_a_row_that_states_no_files_is_given_its_own_and_never_a_directory():
    parsed = rows.typed_rows({"rows": [{"id": name, "kind": "weights", "outcome": "one piece",
                                        "required": ["proof"]} for name in ("a", "b-two")]})  # fmt: skip
    compiled = orch.compile_tasks(parsed, [], own_paths=["package/", "tests/package/"])
    by = {task.id: task for task in compiled.tasks}
    assert by["a"].touches == ["package/a.py", "tests/package/test_modeling_v2_a.py"]
    assert by["b-two"].touches == ["package/b_two.py", "tests/package/test_modeling_v2_b_two.py"]
    assert len(g.next_to_start(g.Graph(compiled.tasks))) == 2


def test_an_onboarded_ref_the_spec_names_is_resolved_or_the_run_is_refused(repo):
    head = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/mv2/earlier/onboarded", head)
    named = {"trtllm_repo_path": str(repo), "onboarded": "refs/mv2/earlier/onboarded"}
    assert execution.onboarded_baseline(named) == head
    assert execution.onboarded_baseline({"trtllm_repo_path": str(repo)}) == ""
    with pytest.raises(ValueError, match="names no commit"):
        execution.onboarded_baseline({**named, "onboarded": "definitely-not-a-git-ref"})
    # The survey of such a run is told of the baseline by its ref; a first run's is told nothing.
    assert f"`onboarded: {named['onboarded']}`" in execution._baseline_block(named)
    assert execution._baseline_block({"trtllm_repo_path": str(repo)}) == ""


def test_the_report_never_attributes_a_later_change_to_onboarded(workspace):
    import json

    def pinned(**more):
        record = {"commit": "a" * 40, "ref": "refs/mv2/run/onboarded", "session": "session-core"}
        (workspace / ONBOARDED_FILENAME).write_text(json.dumps({**record, **more}))

    change = {"key": "accuracy.floor.k", "from": 2, "to": 3, "why": "later revision",
              "at": "2099-01-01", "class": "requirement"}  # fmt: skip
    (workspace / "spec_changes.jsonl").write_text(json.dumps(change) + "\n")
    pinned(at=1, requirements=["`core:gsm8k`: declared 41.77"], changes=0)
    text = report.render(workspace, verdict="partial")
    assert "holds under the requirements as changed" not in text
    assert "1 of these came after `onboarded` was pinned (`accuracy.floor.k`)" in text
    assert "Under the requirements in force then: `core:gsm8k`: declared 41.77." in text
    # One pinned after the change does hold under it; one whose time is not recorded says so.
    pinned(at=5e9, requirements=[], changes=1)
    assert "`onboarded` was pinned after all of these" in report.render(workspace, verdict="x")
    pinned(at=None, recovered_at=7.0, requirements=None, changes=None)
    text = report.render(workspace, verdict="x")
    assert "when it was pinned, and under which requirements, is not recorded" in text
    assert "is attributed to none of them" in text


# ------------------------------------------------- what the prompts step asked of this code


def test_a_receipt_that_does_not_parse_says_which_line_and_the_retry_carries_it(tmp_path):
    """A whole run lost a finished kernel row to `tier: Tier 1: the wrapper's ...`."""
    path = tmp_path / "decisions" / "target-kernel_writer-3.yaml"
    path.parent.mkdir()
    path.write_text("decision: ready\nsummary: certified\ntier: Tier 1: the wrapper's own op\n")
    result = TaskResult.answered("kernel_writer", path, None)
    assert result.decision == "" and result.data == {}
    assert f"the receipt `{path}` is not valid YAML at line 3, column 13" in result.note
    assert "mapping values are not allowed here" in result.note
    # A file that is a receipt says nothing; neither does one that is not there.
    path.write_text("decision: ready\nsummary: certified\ntier: 1\n")
    assert handoff.decision_problem(path, "kernel_writer") == ""
    assert TaskResult.answered("kernel_writer", path, None).note == ""
    assert handoff.decision_problem(tmp_path / "none.yaml", "kernel_writer") == ""
    path.write_text("decision: done\n")
    assert "`decision: 'done'`, which is not one of" in handoff.decision_problem(
        path, "kernel_writer"
    )
    path.write_text("- ready\n")
    assert "is not a mapping" in handoff.decision_problem(path, "kernel_writer")
    # The loop's retry line, and so the next attempt's card, carries the parser's words.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    problem = "the receipt `d.yaml` is not valid YAML at line 3, column 13: mapping values"
    logged = []
    loop = _loop(workspace, [g.Task("row", "feature", "x")], log=logged.append,
                 work=lambda task: f"\n{problem}" if task.attempts == 1 else "ready")  # fmt: skip
    loop.run()
    said = next(line for line in logged if line.startswith("[retry] row"))
    assert f"its turn ended with no receipt: {problem}" in said


def test_a_dispatch_names_its_dependencies_in_one_line_each():
    receipts = [
        {"task": "survey", "kind": "survey", "decision": "ready", "summary": "a long summary " * 40,
         "references": ["a/b.py", "c/d.py"], "decision_file": "/ws/decisions/plan-surveyor-1.yaml"},
        {"task": "k1-rms", "kind": "kernel", "decision": "ready", "not_done": ["fp8 cell", "tp4"],
         "check": {"id": "kernel:1", "command": "gate kernel --row 1", "result": "[PASS] kernel:1"},
         "decision_file": "/ws/decisions/target-kernel_writer-1.yaml"},
    ]  # fmt: skip
    text = handoff.brief(receipts)
    lines = [line for line in text.splitlines() if line.startswith("- ")]
    assert lines == [
        "- `survey` (survey): ready; `/ws/decisions/plan-surveyor-1.yaml`",
        "- `k1-rms` (kernel): ready; check kernel:1: [PASS] kernel:1; 2 thing(s) left not done; "
        "`/ws/decisions/target-kernel_writer-1.yaml`",
    ]
    assert (
        text.startswith(handoff.HEADING) and "a long summary" not in text and "a/b.py" not in text
    )
    assert handoff.brief([]) == ""


def test_a_row_says_what_to_look_for_in_its_own_words_and_its_tests_are_told_by_name():
    """What to look for is one list on the card, under whichever key the surveyor wrote it."""
    entry = {"id": "decode", "kind": "feature", "outcome": "x",
             "leads": ["`replay`: replays a captured graph, the nearest decode path"],
             "definitions": ["capture"], "search_questions": ["Which path is captured?"]}  # fmt: skip
    (row,) = rows.typed_rows({"rows": [entry]})
    assert row.leads == (entry["leads"][0], "`capture`", "Which path is captured?")
    # What must hold is the surveyor's to state: a row with none has its check to pass.
    assert row.required == () and row.check == "cpu:decode"
    assert rows.Row.from_dict(row.as_dict()) == row
    assert sorted(row.as_dict()) == ["check", "id", "kind", "leads", "outcome", "wall"]
    # A test is what pytest collects by name: not a helper beside it, and one outside `tests/`.
    paths = ["tests/unit/test_modeling_v2_rms.py", "tests/unit/helpers.py", "tests/unit/conftest.py",
             "tests/lists/core.yml", "pkg/norm/rms_test.py", "pkg/norm/rms.py"]  # fmt: skip
    assert rows.test_files(paths) == ["tests/unit/test_modeling_v2_rms.py", "pkg/norm/rms_test.py"]
