# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What wakes the improver, and how: batched, one turn in flight, beside the run.

A task's end wakes it when it is a certification that failed, a stuck
worker or a repeat of the same work, not of the same kind. The watcher wakes
it for what the workspace shows: a re-certification, a ready task waiting,
a planner revision that changed nothing, a reported flow issue, a void
verdict. Nothing the run does waits for an improver turn but the run's close.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.agents import lessons as L
from agent_flow.workflows.trtllm_modeling_v2.run import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.stages import improve as I


def _t(tid, kind="feature", ref="", attempts=1, status="pending", **kw):
    return g.Task(
        id=tid, kind=kind, goal=f"do {tid}", ref=ref, attempts=attempts, status=status, **kw
    )


class Host(I.ImproveStage):
    """The improve stage with a workspace, a journal and a scripted improver turn."""

    def __init__(self, tmp_path, answer=None):
        self.workspace = tmp_path / "ws"
        self.workspace.mkdir(exist_ok=True)
        (self.workspace / "journal.md").write_text("x", encoding="utf-8")
        self.journal_path = self.workspace / "journal.md"
        self.lessons_dir = tmp_path / "lessons"
        # A run measuring the workflow: its improver turns as the run goes.
        self.task_data = {"knowledge": "measurement"}
        self.events: list[tuple[str, str]] = []
        self.journal = SimpleNamespace(
            event=lambda what, detail="": self.events.append((what, detail))
        )
        self.prompts: list[str] = []
        self.answer = answer or (lambda prompt: "NOTHING_NEW")
        self.improver_batch_s = 0.0

    def _dispatch_task(self, role, unit, what, prompt):
        assert role == "improver"
        self.prompts.append(prompt)
        return self.answer(prompt)

    def idle(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while self.__dict__.get("_improver_thread") is not None:
            assert time.monotonic() < deadline, "the improver's thread did not end"
            time.sleep(0.01)


@pytest.fixture
def host(tmp_path):
    return Host(tmp_path)


def _graph(ws, tasks, revisions=("orchestrator: survey",)):
    graph = g.Graph(tasks=tasks, revisions=[g.Revision(i + 1, r) for i, r in enumerate(revisions)])
    graph.save(ws / g.GRAPH_FILENAME)
    return graph


# ---------------------------------------------------------------- lineage


@pytest.mark.parametrize(
    "first, second",
    [
        (_t("tune", "tune"), _t("tune-2", "tune")),
        (_t("tune-2", "tune"), _t("tune-3", "tune")),
        (_t("core", "assemble", "core"), _t("core-retry", "assemble", "core")),
        (_t("mtp", "feature", "mtp"), _t("mtp-v2", "feature", "mtp")),
        # A kernel is its queue row, whatever the planner calls it.
        (_t("k46-thop", "kernel", "46"), _t("thop-unpadded", "kernel", "46")),
    ],
)
def test_a_task_written_again_under_a_new_id_keeps_its_lineage(first, second):
    assert g.lineage(first) == g.lineage(second)


@pytest.mark.parametrize(
    "first, second",
    [
        (_t("k-12", "kernel", "12"), _t("k-46", "kernel", "46")),
        (_t("audit-core", "review", "core"), _t("review-core", "review", "core")),
        (_t("mtp", "feature", "core"), _t("kv-fp8", "feature", "core")),
        (_t("core", "assemble", "core"), _t("core-tp", "assemble", "core")),
        (_t("core-2", "assemble", "core"), _t("core-2", "review", "core")),
    ],
)
def test_unrelated_work_of_one_kind_or_unit_is_another_lineage(first, second):
    assert g.lineage(first) != g.lineage(second)


# ------------------------------------------------------------ a task's end


def test_a_failed_receipt_wakes_it_the_first_time(host):
    assert host._wakes_improver(_t("k-12", "kernel", "12"), "FAILED_RECEIPT\nthe op diverged")


def test_a_first_ordinary_failure_does_not_wake_it(host):
    assert not host._wakes_improver(_t("tune", "tune"), "TUNE_FAILED")
    assert not host._wakes_improver(_t("k-12", "kernel", "12"), "MACRO")


@pytest.mark.parametrize("decision", ["", "BLOCKED", "INTEGRATION_FAILED\nrebase conflicted"])
def test_an_environment_shaped_failure_wakes_it_the_first_time(host, decision):
    assert host._wakes_improver(_t("core", "assemble", "core"), decision)


def test_a_finished_task_never_wakes_it(host):
    task = _t("core", "assemble", "core", attempts=3)
    assert not host._wakes_improver(task, "UNIT_READY")
    assert not host._wakes_improver(task, "UNIT_READY\nthe tokenizer needs a template")


def test_the_same_kernel_failing_again_under_a_new_id_wakes_it(host):
    assert not host._wakes_improver(_t("k46-thop", "kernel", "46"), "MACRO")
    assert host._wakes_improver(_t("k46-thop-unpadded", "kernel", "46"), "MACRO")


def test_a_review_reject_on_its_first_attempt_never_wakes_it(host):
    """An audit's job is to find gaps; a review that rejects is doing its job."""
    assert not host._wakes_improver(_t("audit-core", "review", "core"), "REJECT")
    assert not host._wakes_improver(_t("review-core", "review", "core"), "REJECT")
    assert not host._wakes_improver(_t("review-core-2", "review", "core"), "REJECT")
    assert host._wakes_improver(_t("review-core", "review", "core", attempts=2), "REJECT")


def test_a_retried_task_failing_again_wakes_it_in_a_fresh_process(tmp_path):
    """``attempts`` is in the graph, so a restart between the two failures changes nothing."""
    assert Host(tmp_path)._wakes_improver(_t("tune", "tune", attempts=3), "TUNE_FAILED")


def test_the_same_attempt_failing_twice_in_one_process_wakes_it(host):
    """An infrastructure retry runs the task again as the same attempt."""
    task = _t("core", "assemble", "core")
    assert not host._wakes_improver(task, "raised TimeoutError: turn")
    assert host._wakes_improver(task, "raised TimeoutError: turn")


def test_a_task_end_never_waits_for_the_improver_turn(tmp_path):
    release, started = threading.Event(), threading.Event()

    def slow(prompt):
        started.set()
        assert release.wait(10)
        return "NOTHING_NEW"

    host = Host(tmp_path, slow)
    began = time.monotonic()
    host._improver_task_ended(_t("k-12", "kernel", "12"), "FAILED_RECEIPT\ncells diverged")
    assert time.monotonic() - began < 1.0
    assert started.wait(10)
    release.set()
    host.idle()
    [prompt] = host.prompts
    assert "task `k-12` (kernel, role smith) ended FAILED_RECEIPT" in prompt
    assert "**a task ended without finishing**" in prompt


def test_without_a_lessons_store_nothing_wakes_it(host):
    host.lessons_dir = None
    host._improver_task_ended(_t("k-12", "kernel", "12"), "FAILED_RECEIPT")
    assert host._improver_tick() == [] and host.prompts == []


# ---------------------------------------------------------- batched, one in flight


def test_wakes_are_batched_and_one_turn_is_in_flight(tmp_path):
    release = threading.Event()
    in_flight, most = [0], [0]
    lock = threading.Lock()

    def turn(prompt):
        with lock:
            in_flight[0] += 1
            most[0] = max(most[0], in_flight[0])
        if len(host.prompts) == 1:
            assert release.wait(10)
        with lock:
            in_flight[0] -= 1
        return "NOTHING_NEW"

    host = Host(tmp_path, turn)
    host._improver_wake(I.Wake("check_void", "a", "first"))
    deadline = time.monotonic() + 10
    while not host.prompts:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    for key in ("b", "c", "d"):
        host._improver_wake(I.Wake("check_void", key, f"wake {key}"))
    host._improver_wake(I.Wake("check_void", "b", "seen already"))
    release.set()
    host.idle()
    assert len(host.prompts) == 2 and most[0] == 1
    assert all(f"wake {k}" in host.prompts[1] for k in "bcd")
    assert "seen already" not in host.prompts[1]
    assert ("improver woken", "3 a check's verdict was void") in host.events


def test_the_first_wake_waits_for_others_to_join_it(tmp_path):
    host = Host(tmp_path)
    host.improver_batch_s = 0.3
    host._improver_wake(I.Wake("check_void", "a", "first"))
    host._improver_wake(I.Wake("flow_issue", "b", "second"))
    host.idle()
    assert len(host.prompts) == 1 and "second" in host.prompts[0]


def test_a_wake_no_turn_read_survives_a_restart(tmp_path, monkeypatch):
    before = Host(tmp_path)
    monkeypatch.setattr(before, "_improver_loop", lambda: None)  # the process stops first
    before._improver_wake(I.Wake("check_void", "a", "left unread"))
    saved = json.loads((before.workspace / "improve" / I.WAKES_FILENAME).read_text())
    assert [w["key"] for w in saved["pending"]] == ["a"]

    after = Host(tmp_path)
    after._improver_resume()
    after.idle()
    assert "left unread" in after.prompts[0]
    saved = json.loads((after.workspace / "improve" / I.WAKES_FILENAME).read_text())
    assert saved["pending"] == [] and saved["seen"] == ["a"]
    after._improver_wake(I.Wake("check_void", "a", "again"))
    after.idle()
    assert len(after.prompts) == 1  # seen in the process before


def test_the_close_reads_every_wake_left(tmp_path, monkeypatch):
    host = Host(tmp_path)
    host.improver_batch_s = 3600.0
    host._improver_wake(I.Wake("ready_waiting", "a", "waited an hour"))
    host._improve("The run closed with verdict `delivered`.")
    [prompt] = host.prompts
    assert "verdict `delivered`" in prompt and "waited an hour" in prompt
    assert json.loads((host.workspace / "improve" / I.WAKES_FILENAME).read_text())["pending"] == []


def test_a_turn_that_raised_drops_its_wakes_and_ends_its_thread(tmp_path):
    def broken(prompt):
        raise RuntimeError("transport down")

    host = Host(tmp_path, broken)
    host._improver_wake(I.Wake("check_void", "a", "x"))
    host.idle()
    assert host.events[-1][0] == "improver turn raised"
    assert "RuntimeError: transport down" in host.events[-1][1]


def test_what_python_refused_is_shown_to_the_next_turn(tmp_path):
    def propose(prompt):
        proposal = {"lessons": [{"scope": "role:nobody", "trigger": "t", "lesson": "l"}]}
        (host.workspace / L.PROPOSED_FILENAME).write_text(yaml.safe_dump(proposal))
        return "LESSONS_PROPOSED"

    host = Host(tmp_path, propose)
    host._improve("first")
    host.answer = lambda prompt: "NOTHING_NEW"
    host._improve("second")
    assert "## Your last proposal" not in host.prompts[0]
    refused = host.prompts[1].split("## Your last proposal", 1)[1]
    assert "no role 'nobody'" in refused


def test_the_request_names_the_reported_issues_and_how_to_check_a_proposal(host):
    host._improve("closed")
    prompt = host.prompts[0]
    assert L.REPORTED_ISSUES_FILENAME in prompt and "(none yet)" in prompt
    assert "## Patch drafts" in prompt
    if L.flow_layout() is not None:
        assert "check-proposal" in prompt and f"archive {L.RUNNING_HEAD}" in prompt


def _file_issue(ws, **kw):
    issue = {
        "lesson": "role:planner:abc",
        "scope": "role:planner",
        "symptom": "a ready task waits for a slot nobody holds",
        "suspected": "run/orchestrator.py:_start_ready",
        "proposed_fix": "free the slot",
        **kw,
    }
    (ws / L.FLOW_ISSUES_FILENAME).write_text(yaml.safe_dump({"issues": [issue]}))


def test_a_wake_an_open_issue_answers_is_recorded_on_it_and_opens_no_turn(host):
    ws = host.workspace
    host.improver_batch_s = 3600.0
    _file_issue(ws, wakes=["ready_waiting"])
    assert host._improver_wake(I.Wake("ready_waiting", "ready:a#1", "task `a` waited 31 min")) == []
    [new] = host._improver_wake(I.Wake("planner_noop", "noop:3", "revision 3 changed nothing"))
    assert new.key == "noop:3"
    [issue] = L.load_issues(ws)
    assert len(issue.recurrences) == 1 and "task `a` waited 31 min" in issue.recurrences[0]
    pending = json.loads((ws / "improve" / I.WAKES_FILENAME).read_text())["pending"]
    assert [w["key"] for w in pending] == ["noop:3"]
    assert any(what == "improver wake answered" for what, _ in host.events)
    # A refresh of the issue keeps what was recorded on it.
    L.record_issues(ws, [L.FlowIssue.from_dict({**issue.__dict__, "wakes": ["planner_noop"]})])
    [issue] = L.load_issues(ws)
    assert issue.wakes == ["ready_waiting", "planner_noop"] and len(issue.recurrences) == 1


def test_a_wake_that_recurs_after_its_issue_was_fixed_wakes_it(host):
    host.improver_batch_s = 3600.0
    _file_issue(host.workspace, wakes=["ready_waiting"], fixed_by="abc1234")
    [new] = host._improver_wake(I.Wake("ready_waiting", "ready:a#1", "task `a` waited 31 min"))
    assert new.key == "ready:a#1" and L.load_issues(host.workspace)[0].recurrences == []


def test_the_request_shows_each_wakes_signature(host):
    host.improver_batch_s = 3600.0
    host._improver_task_ended(_t("k-12", "kernel", "12"), "FAILED_RECEIPT")
    host._improve("closed")
    assert (
        f"signature `task_failed:{g.lineage(_t('k-12', 'kernel', '12'))}:FAILED_RECEIPT`"
        in (host.prompts[0])
    )


# ------------------------------------------------------------- the watcher


def test_a_ready_task_waiting_half_an_hour_wakes_it_once(host):
    ws = host.workspace
    _graph(
        ws,
        [
            _t("core", "assemble", "core", status="running", touches=["a/model.py"]),
            _t("mtp", "feature", "mtp", touches=["a/model.py"], gpus=4),
            _t("later", "feature", "x", depends_on=["core"]),
        ],
    )
    host.improver_batch_s = 3600.0
    assert host._improver_tick(now=1000.0) == []
    assert host._improver_tick(now=1000.0 + I.READY_WAIT_S - 1) == []
    [wake] = host._improver_tick(now=1000.0 + I.READY_WAIT_S)
    assert wake.kind == "ready_waiting" and wake.key == "ready:mtp#1"
    assert "ready to start for 30 min" in wake.detail and "asks 4 GPU(s)" in wake.detail
    assert "shares paths with `core`" in wake.detail
    assert host._improver_tick(now=1000.0 + 2 * I.READY_WAIT_S) == []


def test_a_task_that_started_is_no_longer_waiting(host):
    since: dict[str, float] = {}
    graph = g.Graph(tasks=[_t("mtp")])
    assert I.ready_waits(graph, since, 0.0, 60.0) == [] and since == {"mtp#1": 0.0}
    graph.tasks[0].status = "running"
    assert I.ready_waits(graph, since, 120.0, 60.0) == [] and since == {}


def test_a_campaign_or_a_nice_to_have_task_waits_by_design():
    since: dict[str, float] = {}
    graph = g.Graph(tasks=[_t("tune", "tune"), _t("fp8-kv", priority="nice_to_have")])
    assert I.ready_waits(graph, since, 0.0, 60.0) == []
    assert I.ready_waits(graph, since, 7200.0, 60.0) == [] and since == {}


def test_a_re_certification_of_a_certified_entry_wakes_it(host):
    ws = host.workspace
    host.improver_batch_s = 3600.0
    _graph(
        ws,
        [
            _t("k-attn", "kernel", "verify:attention/paged", status="done"),
            _t("k-attn-depths", "kernel", "verify:attention/paged"),
            _t("k-norm", "kernel", "onboard:norm/rms"),
            _t("k-conv", "kernel", "verify:conv/causal", status="failed"),
            _t("k-conv-2", "kernel", "verify:conv/causal"),
        ],
    )
    wakes = [w for w in host._improver_tick(now=0.0) if w.kind == "recertification"]
    assert [w.key for w in wakes] == ["recert:k-attn-depths"]
    assert "certifies `attention/paged` again, which `k-attn` certified" in wakes[0].detail


def test_a_queue_row_names_its_entry():
    rows = {"7": "attention/paged", "8": "attention/paged"}
    graph = g.Graph(
        tasks=[_t("k7", "kernel", "7", status="done"), _t("k8", "kernel", "8", status="running")]
    )
    [wake] = I.recertifications(graph, lambda t: rows.get(t.ref, ""))
    assert wake.key == "recert:k8"


def test_a_promote_certifies_nothing_again(host):
    """A promote merges landed variants' cells into the catalog and adds none: never a re-certification.

    However many entries its ref names, a pending promote does not wake the improver, and a done
    one is not a certification a later ``verify:`` repeats; a ``verify:`` after a row still wakes it.
    """
    ws = host.workspace
    host.improver_batch_s = 3600.0
    host.vocabulary_path = ws / "vocabulary.yaml"
    host.vocabulary_path.write_text(
        yaml.safe_dump(
            {
                "verdict": "assemblable_after_additions",
                "queue": [{"n": 16, "mode": "verify", "target": "ssm/chunk", "surface": "s"}],
            }
        ),
        encoding="utf-8",
    )
    _graph(
        ws,
        [
            _t("k16", "kernel", "16", status="done"),
            _t("promote-3", "kernel", "promote:ssm/chunk"),
            _t("promote", "kernel", "promote:ssm/chunk,ssm/conv"),
            _t("promote-0", "kernel", "promote:moe/route", status="done"),
            _t("k-route", "kernel", "verify:moe/route"),
            _t("k16c", "kernel", "verify:ssm/chunk@r16"),
        ],
    )
    wakes = [w.key for w in host._improver_tick(now=0.0) if w.kind == "recertification"]
    assert wakes == ["recert:k16c"]


def test_rows_on_one_shared_entry_at_their_own_sites_are_not_certified_again(host):
    """A row's ``site`` names its variant; a later task at the same site, or none, repeats it."""
    ws = host.workspace
    host.improver_batch_s = 3600.0
    host.vocabulary_path = ws / "vocabulary.yaml"
    row = {"mode": "verify", "target": "attention/thop", "surface": "s"}
    host.vocabulary_path.write_text(
        yaml.safe_dump(
            {
                "verdict": "assemblable_after_additions",
                "queue": [{"n": 20, "site": "vision", **row}],
            }
        ),
        encoding="utf-8",
    )
    _graph(
        ws,
        [
            _t("k20", "kernel", "20", status="done"),
            _t("k-vision-2", "kernel", "verify:attention/thop@vision"),
            _t("k-thop", "kernel", "verify:attention/thop"),
        ],
    )
    wakes = {w.key: w for w in host._improver_tick(now=0.0) if w.kind == "recertification"}
    assert sorted(wakes) == ["recert:k-thop", "recert:k-vision-2"]
    said = wakes["recert:k-vision-2"].detail
    assert "certifies `attention/thop@vision` again, which `k20` certified" in said


def test_a_planner_revision_that_changed_nothing_wakes_it(host):
    ws = host.workspace
    host.improver_batch_s = 3600.0
    tasks = [_t("core", "assemble", "core"), _t("mtp", "feature", "mtp")]
    _graph(ws, tasks, ["orchestrator: seed", "plan"])
    assert host._improver_tick(now=0.0) == []
    # The orchestrator started core meanwhile; the planner wrote the plan back.
    tasks[0].status = "running"
    _graph(ws, tasks, ["orchestrator: seed", "plan", "events handled"])
    [wake] = host._improver_tick(now=1.0)
    assert wake.kind == "planner_noop" and "revision 3 left the plan as it was" in wake.detail
    # A revision that adds a task, or retries one, is no no-op.
    tasks.append(_t("review", "review", "core"))
    _graph(ws, tasks, ["orchestrator: seed", "plan", "events handled", "review core"])
    assert host._improver_tick(now=2.0) == []
    tasks[1].attempts = 2
    _graph(ws, tasks, ["orchestrator: seed", "plan", "events handled", "review core", "retry"])
    assert host._improver_tick(now=3.0) == []


def _verdict(ws, **kw):
    record = {"unit": "core", "gate": "boot", "member": "tp4", "passed": False, **kw}
    with (ws / contract_mod.STATUS_FILENAME).open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def test_a_void_verdict_wakes_it_once_per_check_and_cause(host):
    ws = host.workspace
    host.improver_batch_s = 3600.0
    _verdict(ws, passed=True)
    _verdict(ws, void="rank 1 imported /trees/abc1234def instead of the snapshot")
    [wake] = host._improver_tick(now=0.0)
    assert wake.kind == "check_void" and "`core:boot@tp4` measured nothing" in wake.detail
    _verdict(ws, void="rank 3 imported /trees/0fedcba987 instead of the snapshot")
    assert host._improver_tick(now=1.0) == []
    _verdict(ws, void="the snapshot could not be made")
    assert [w.kind for w in host._improver_tick(now=2.0)] == ["check_void"]
    assert host._improver_tick(now=3.0) == []
    # A line still being written is read once it is whole.
    with (ws / contract_mod.STATUS_FILENAME).open("a") as fh:
        fh.write('{"unit": "core", "gate": "witness", "member": "tp4", "void": "half')
    assert host._improver_tick(now=4.0) == []


def test_a_reported_flow_issue_wakes_it(host):
    ws = host.workspace
    L.report_flow_issue(
        ws,
        by="driver",
        symptom="the planner was woken by events a revision had answered",
        evidence=["journal.md"],
        impact="20 min per turn",
        suspected="run/orchestrator.py:_drop_answered",
    )
    with (ws / L.REPORTED_ISSUES_FILENAME).open("a") as fh:
        fh.write("free text from a worker\n")
    host.improver_batch_s = 3600.0
    wakes = host._improver_tick(now=0.0)
    assert [w.key for w in wakes] == ["reported:0", "reported:1"]
    assert "line 1 of `flow_issues.reported.jsonl`: driver reported" in wakes[0].detail
    assert "impact: 20 min per turn" in wakes[0].detail and "`journal.md`" in wakes[0].detail
    assert "free text from a worker" in wakes[1].detail
    assert host._improver_tick(now=1.0) == []


def test_the_watcher_reads_the_workspace_while_the_run_is_under_way(tmp_path):
    host = Host(tmp_path)
    L.report_flow_issue(host.workspace, symptom="a gate leaked a server")
    host.improver_watch_s = 3600.0
    with host._improver_watching():
        deadline = time.monotonic() + 10
        while not host.prompts:
            assert time.monotonic() < deadline
            time.sleep(0.01)
    host.idle()
    assert "a gate leaked a server" in host.prompts[0]


def test_a_proposal_answered_after_its_process_stopped_is_merged_not_deleted(tmp_path):
    """A reload between an improver's answer and its reading must not lose the proposal."""
    from agent_flow.workflows.trtllm_modeling_v2.infra.transport import (
        DISPATCH_DIRNAME,
        ExternalAgentTransport,
    )

    host = Host(tmp_path)
    ws = host.workspace
    dispatch = ws / DISPATCH_DIRNAME
    dispatch.mkdir()
    decision = ws / "decisions" / "improve-improver-1.yaml"
    decision.parent.mkdir()
    decision.write_text("decision: LESSONS_PROPOSED\nsummary: one lesson\n", encoding="utf-8")
    (dispatch / "098-improver.request.json").write_text(
        json.dumps(
            {
                "id": "098-improver",
                "role": "improver",
                "what": "improve",
                "decision_file": str(decision),
                "requested_at": time.time() - 600,
            }
        ),
        encoding="utf-8",
    )
    (dispatch / "098-improver.result.md").write_text("proposed", encoding="utf-8")
    (dispatch / "098-improver.done").write_text("1\n", encoding="utf-8")
    proposal = {
        "lessons": [
            {
                "scope": "role:planner",
                "trigger": "late t",
                "lesson": "late l",
                "evidence": "journal.md",
            }
        ]
    }
    (ws / L.PROPOSED_FILENAME).write_text(yaml.safe_dump(proposal), encoding="utf-8")
    host._turns = ExternalAgentTransport(ws, poll_s=0.01, timeout_s=5)

    host._improve("after the reload")

    assert any(what == "lessons carried over" for what, _ in host.events), host.events
    assert "merged before this turn" in host.prompts[0], host.events
    store = L.LessonStore(host.lessons_dir)
    assert any(lesson.trigger == "late t" for lesson in store.load("role:planner"))


def test_a_production_runs_wakes_wait_for_the_turn_at_its_close(tmp_path):
    """The user's run gives no agent slot to the flow's own improvement while it works."""
    host = Host(tmp_path)
    host.task_data = {}
    host._improver_wake(I.Wake("ready_waiting", "a", "waited an hour"))
    time.sleep(0.2)
    assert host.prompts == [] and host.__dict__.get("_improver_thread") is None
    host._improve("The run closed with verdict `delivered`.")
    [prompt] = host.prompts
    assert "waited an hour" in prompt
