# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for reports generated from the ledger, and for the consistency check.

The reflection on the DeepSeek-V4.1 run lists ten contradictions between its
records (its section 6). The first two are reproduced here as fixtures:

1. a summary saying "16/16 PASS, no finding" while the final regression
   table held a FINDING on ``accept:tp4``;
2. a progress page listing an A/B as pending after it had closed, under a
   stale branch, target commit and status.

Generated reports cannot make either mistake, and the check catches both
when a human writes them.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.staircase_onboard import report
from agent_flow.workflows.staircase_onboard.freshness import complete_regression, record_regression
from agent_flow.workflows.staircase_onboard.gates import GateResult
from agent_flow.workflows.staircase_onboard.ledger import (
    Ledger,
    VerdictContext,
    native_build,
    record_gate_result,
    tree_identity,
)
from agent_flow.workflows.staircase_onboard.report import (
    FINAL_REPORT,
    OVERLAY_FILENAME,
    PROGRESS_HTML,
    PROGRESS_JSON,
    build_progress,
    check_claims,
    check_consistency,
    check_items,
    check_progress,
    collect_facts,
    headline,
    narrative,
    render_final_report,
    write_reports,
)

from .test_delivery import TARGET_REL, _build_repo


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    root = _build_repo(tmp_path)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@e.st")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "seed")
    return root


@pytest.fixture
def workspace(tmp_path, repo):
    ws = tmp_path / "ws"
    ws.mkdir()
    spec = {
        "trtllm_repo_path": str(repo),
        "features": ["dspark"],
        "environment": {"kind": "local", "python": "/py", "env": {}},
    }
    (ws / "task.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    return ws


class _Finding:
    """A gate result flagged for a human, the way an acceptance gate would flag one."""

    name = "acceptance"
    passed = True
    detail = "-0.1216 against the dspark gate"
    member = "tp4"
    measured = 2.71
    finding = True


def _record(ws, repo, result, unit="core"):
    return record_gate_result(
        Ledger.in_workspace(ws),
        result,
        VerdictContext(repo_root=str(repo), target_rel=TARGET_REL, unit=unit),
    )


def _gate(name, member="tp4", passed=True, **kw):
    kw.setdefault("detail", "ok")
    return GateResult(name=name, passed=passed, member=member, **kw)


def _sixteen_with_a_finding(ws, repo):
    """The shape of the DeepSeek-V4.1 final regression: all green but one FINDING."""
    for member in ("tep4", "dep4", "tp4"):
        _record(ws, repo, _gate("boot", member))
        _record(ws, repo, _gate("accuracy[gsm8k]", member, measured=92.5, reference=92.3))
    _record(ws, repo, _Finding(), unit="dspark")
    ledger = Ledger.in_workspace(ws)
    regression = complete_regression(ledger.latest(), tree_identity(repo), native_build(repo))
    record_regression(ledger, regression)


# --------------------------------------------------------------- generation


def test_the_headline_is_counted_not_written(workspace, repo):
    _sixteen_with_a_finding(workspace, repo)
    facts = collect_facts(workspace)
    text = headline(facts)
    assert "7 verdict(s): 6 PASS, 1 FINDING, 0 FAIL" in text
    assert "bound to HEAD" in text
    assert check_claims(text, facts, where="headline") == []


def test_progress_json_carries_every_fact_from_the_ledger(workspace, repo):
    _sixteen_with_a_finding(workspace, repo)
    facts = collect_facts(workspace)
    doc = build_progress(facts, workspace=workspace)
    assert doc["generated_from"]["ledger_digest"] == Ledger.in_workspace(workspace).digest()
    assert len(doc["gates"]["rows"]) == 7
    assert [f["gate"] for f in doc["findings"]] == ["acceptance"]
    assert doc["regression"]["binding"]["ok"]
    assert check_progress(doc, facts) == []


def test_a_stale_verdict_shows_as_stale_and_queued(workspace, repo):
    _record(workspace, repo, _gate("boot"))
    (repo / TARGET_REL / "modeling.py").write_text("# rewritten\n", encoding="utf-8")
    doc = build_progress(collect_facts(workspace))
    (row,) = doc["gates"]["rows"]
    assert row["fresh"] == "STALE"
    assert any("code changed" in r for r in row["reasons"])
    assert doc["rerun_queue"][0]["gate"] == "boot"
    assert doc["summary"]["gates"]["stale"] == 1


def test_the_page_is_offline_and_escapes_what_it_shows(workspace, repo):
    _record(workspace, repo, _gate("boot", detail="<script>x</script>"))
    doc = write_reports(workspace)
    page = (workspace / PROGRESS_HTML).read_text()
    assert "<script>x</script>" not in page
    assert "http://" not in page and "https://" not in page
    assert "boot" in page and doc["meta"]["title"] in page
    assert json.loads((workspace / PROGRESS_JSON).read_text())["consistency"]["ok"]


def test_overlay_sections_are_rendered(workspace, repo):
    (workspace / OVERLAY_FILENAME).write_text(
        yaml.safe_dump(
            {
                "meta": {"model": "Some-Model", "hardware": "4x GB300"},
                "features": [{"feature": "TP4", "status": "DONE", "notes": "core"}],
                "workflow_feedback": [{"id": "F041", "title": "late checks", "status": "OPEN"}],
            }
        ),
        encoding="utf-8",
    )
    doc = write_reports(workspace)
    page = (workspace / PROGRESS_HTML).read_text()
    assert doc["meta"]["model"] == "Some-Model"
    assert "Feature scorecard" in page and "F041" in page


# ----------------------------------------------- the contradictions that happened


def test_no_finding_over_a_finding_is_a_contradiction(workspace, repo):
    """Reflection section 6, row 1."""
    _sixteen_with_a_finding(workspace, repo)
    facts = collect_facts(workspace)
    problems = check_claims("Final regression: 7/7 PASS, no finding.", facts, where="report")
    assert [p.claim for p in problems] == ["no finding"]
    assert "acceptance@tp4 [dspark]" in problems[0].fact


def test_n_of_n_pass_over_a_stale_verdict_is_a_contradiction(workspace, repo):
    _record(workspace, repo, _gate("boot"))
    _record(workspace, repo, _gate("accuracy[gsm8k]"))
    (repo / TARGET_REL / "weights.py").write_text("MANIFEST = (1,)\n", encoding="utf-8")
    facts = collect_facts(workspace)
    problems = check_claims("All gates pass; 2/2 PASS on the final tree.", facts, where="r")
    assert {p.claim for p in problems} == {"All gates pass", "2/2 PASS"}


def test_an_n_of_n_that_counts_neither_set_is_a_contradiction(workspace, repo):
    _record(workspace, repo, _gate("boot"))
    facts = collect_facts(workspace)
    (problem,) = check_claims("16/16 PASS", facts, where="r")
    assert "number 1" in problem.fact


def test_a_partial_count_is_not_a_claim_of_completeness(workspace, repo):
    _record(workspace, repo, _gate("boot", passed=False))
    assert check_claims("1/2 PASS so far", collect_facts(workspace), where="r") == []


def test_a_final_tree_that_is_not_the_regression_tree_is_a_contradiction(workspace, repo):
    """Reflection rows 5 and 8: hashes in the report that no regression ran on."""
    _sixteen_with_a_finding(workspace, repo)
    facts = collect_facts(workspace)
    head = _git(repo, "rev-parse", "HEAD").strip()
    assert check_claims(f"the final tree is `{head[:10]}`", facts, where="r") == []
    (problem,) = check_claims("the final tree is `e8c78be0a1`", facts, where="r")
    assert "last full regression ran on" in problem.fact


def test_a_pending_item_that_is_in_fact_closed(workspace, repo):
    """Reflection row 2: the A/B still listed as pending after it closed."""
    _sixteen_with_a_finding(workspace, repo)
    _record(workspace, repo, _gate("acceptance", "tp4"), unit="dspark")
    facts = collect_facts(workspace)
    items = [
        {
            "id": "G4",
            "title": "acceptance A/B",
            "status": "IN PROGRESS",
            "evidence": [{"gate": "acceptance", "unit": "dspark"}],
        }
    ]
    (problem,) = check_items(items, facts)
    assert "in fact closed" in problem.fact


def test_a_closed_item_needs_fresh_passing_evidence(workspace, repo):
    _record(workspace, repo, _gate("boot", "tp4"))
    _record(workspace, repo, _gate("accuracy[gsm8k]", "tp4", passed=False))
    facts = collect_facts(workspace)
    items = [
        {"id": "V1", "status": "DONE", "evidence": {"unit": "core"}},
        {"id": "V2", "status": "DONE", "evidence": {"gate": "witness*"}},
        {"id": "V3", "status": "DONE"},  # narrative only: not checked
    ]
    problems = check_items(items, facts)
    assert [p.where for p in problems] == ["item V1", "item V2"]
    assert "FAIL" in problems[0].fact
    assert "matches no verdict" in problems[1].fact


def test_a_hand_edited_or_outdated_page_is_a_contradiction(workspace, repo):
    _record(workspace, repo, _gate("boot", passed=False))
    write_reports(workspace)
    doc = json.loads((workspace / PROGRESS_JSON).read_text())
    doc["gates"]["rows"][0]["result"] = "PASS"
    doc["rerun_queue"].append({"gate": "accuracy[gsm8k]", "unit": "core", "member": "tp4"})
    (workspace / PROGRESS_JSON).write_text(json.dumps(doc))
    _record(workspace, repo, _gate("accuracy[gsm8k]"))  # the ledger moves on

    problems = check_consistency(workspace)
    facts = [p.fact for p in problems]
    assert any("the page is stale" in f for f in facts)
    assert any("the ledger says FAIL" in f for f in facts)
    assert any("in fact closed" in f for f in facts)


# ------------------------------------------------------------- final report


def test_the_final_report_regenerates_its_tables_and_keeps_the_narrative(workspace, repo):
    _record(workspace, repo, _gate("boot"))
    first = render_final_report(collect_facts(workspace))
    assert "<!-- BEGIN GENERATED: gate-table -->" in first
    written = first + "\nThe operator's conclusion: core is ready.\n"

    _record(workspace, repo, _gate("accuracy[gsm8k]", measured=91.0, reference=92.0))
    second = render_final_report(collect_facts(workspace), written)
    assert "accuracy[gsm8k]" in second
    assert "The operator's conclusion: core is ready." in second
    assert "accuracy[gsm8k]" not in narrative(second)


def test_a_region_missing_from_an_existing_report_is_appended():
    text = render_final_report(report.Facts(), "# My report\n\nprose\n")
    assert text.startswith("# My report\n\nprose\n")
    for name in ("summary", "gate-table", "findings", "freshness", "regression"):
        assert f"<!-- BEGIN GENERATED: {name} -->" in text


def test_the_final_reports_narrative_is_checked(workspace, repo):
    _sixteen_with_a_finding(workspace, repo)
    write_reports(workspace, final_report=True)
    path = workspace / FINAL_REPORT
    path.write_text(path.read_text() + "\nConclusion: no finding.\n", encoding="utf-8")
    (problem,) = check_consistency(workspace)
    assert problem.where == FINAL_REPORT


def test_report_check_cli_exit_codes(workspace, repo, capsys):
    _sixteen_with_a_finding(workspace, repo)
    assert report.main([str(workspace)]) == 0
    assert report.main([str(workspace), "--check"]) == 0
    note = workspace / "summary.md"
    note.write_text("16/16 PASS, no finding\n", encoding="utf-8")
    assert report.main([str(workspace), "--check", str(note)]) == 1
    assert "CONTRADICTION" in capsys.readouterr().out


# ------------------------------------------------------------ in the workflow


def _run(tmp_path, task_file, monkeypatch, **kwargs):
    from agent_flow.workflows.staircase_onboard.state import load_state

    from .test_workflow import Script, _install, _plan_then_vocabulary, _workflow

    workflow = _workflow(tmp_path, **kwargs)
    _install(monkeypatch, workflow, Script(None))
    _plan_then_vocabulary(workflow, tmp_path, monkeypatch, [])
    with workflow:
        workflow.run(task_file)
    return load_state(tmp_path / "ws" / ".staircase_onboard_state.json")


def test_a_run_writes_its_reports_from_the_ledger(tmp_path, task_file, monkeypatch):
    state = _run(tmp_path, task_file, monkeypatch)
    ws = tmp_path / "ws"
    assert state.verdict == "delivered"
    doc = json.loads((ws / PROGRESS_JSON).read_text())
    assert doc["meta"]["status"] == "DELIVERED"
    gates = {r["gate"].split("[")[0] for r in doc["gates"]["rows"] if r["unit"]}
    assert gates == {"boot", "accuracy"}
    assert any(r["gate"].startswith("delivery:") for r in doc["gates"]["rows"])
    assert doc["pipeline_stages"][-1]["status"] == "DELIVERED"
    assert (ws / PROGRESS_HTML).is_file()
    assert "<!-- BEGIN GENERATED: gate-table -->" in (ws / FINAL_REPORT).read_text()
    assert check_consistency(ws) == []


def test_delivery_refuses_a_report_that_contradicts_the_ledger(tmp_path, task_file, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / OVERLAY_FILENAME).write_text(
        yaml.safe_dump(
            {"items": [{"id": "W1", "status": "DONE", "evidence": {"gate": "witness*"}}]}
        ),
        encoding="utf-8",
    )
    state = _run(tmp_path, task_file, monkeypatch)
    assert state.verdict == "delivery-failed:evidence"
    assert "contradict the ledger" in (ws / "journal.md").read_text()


def test_clean_archives_the_previous_runs_evidence(tmp_path, task_file, monkeypatch):
    _run(tmp_path, task_file, monkeypatch)
    ws = tmp_path / "ws"
    before = (ws / "ledger.jsonl").read_text()
    _run(tmp_path, task_file, monkeypatch, clean=True)
    archived = list((ws / "archive").glob("*-ledger.jsonl"))
    assert len(archived) == 1 and archived[0].read_text() == before
