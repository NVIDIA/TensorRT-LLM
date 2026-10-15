# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the typed ``certification_gap`` decision and its automatic fan-out.

The history behind it, from the DeepSeek-V4.1 run: three reviewer REJECTs of a
sound core target, each on certification alone ("the target code is sound and
needs no change"), each costing a reviewer turn plus an assembler turn; an
assembler that could only say ``knowledge_gap`` for four verifies it needed,
which closed the run; and four verify smiths dispatched by hand. Each case
below is one of those, now handled by the orchestrator.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2 import progress as progress_mod
from agent_flow.workflows.trtllm_modeling_v2 import remediation as R
from agent_flow.workflows.trtllm_modeling_v2.state import load_state

from .test_workflow import Script, _install, _workflow, _write_vocabulary

_GAP = {
    "entry": "activation/silu_and_mul",
    "values": ["2", "3", "6", "8192"],
    "axis": "rows",
    "reason": "MoE expert MLP at the verify step: 6·G rows",
}


# ---------------------------------------------------------------- parsing


def test_a_gap_row_needs_only_an_entry():
    rows, dropped = R.parse_certification_gaps(
        [
            {"entry": "moe/runner", "values": [16384]},
            {"entry": "norm/rms", "value": "3", "axis": "T", "arch": "sm_100", "reason": "why"},
            {"values": ["1"]},
            "not a row",
        ],
        default_arch="sm_103",
    )
    assert dropped == 2
    assert (rows[0].arch, rows[0].axis, rows[0].values) == (
        "sm_103",
        R.DEFAULT_GAP_AXIS,
        ("16384",),
    )
    assert (rows[1].arch, rows[1].axis, rows[1].contract_line) == ("sm_100", "T", "why")


def test_the_auditor_may_mix_uncovered_and_gap_rows():
    entry = {
        "uncovered": [{"entry": "a/x", "arch": "sm_103", "axis": "rows", "values": ["5"]}],
        R.CERTIFICATION_GAP_KEY: [{"entry": "b/y", "values": ["7"]}, {"values": []}],
    }
    rows, dropped = R.parse_audit_rows(entry, default_arch="sm_103")
    assert [r.entry for r in rows] == ["a/x", "b/y"] and dropped == 1


def test_records_are_normalized_at_the_persistence_boundary():
    records = R.normalize_certification_gap_records(
        [{"entry": " moe/runner ", "values": [1, " 2 ", ""], "reason": " r "}, {"entry": ""}, 3]
    )
    assert records == [{"entry": "moe/runner", "values": ["1", "2"], "reason": "r"}]


def _item(entry: str, status: str = R.QUEUE_PENDING, values=("1",)) -> R.RemediationItem:
    return R.RemediationItem(entry=entry, arch="sm_103", axis="rows", values=values, status=status)


def test_new_gaps_are_dispatched_and_certified_ones_are_not():
    budget = R.RemediationBudget()
    queue = [_item("a/done", R.QUEUE_CERTIFIED)]
    incoming = (_item("a/done"), _item("b/new"))
    merge = R.merge_into_queue(queue, incoming, budget=budget)
    decision = R.decide_gap_dispatch(merge, budget=budget, incoming=incoming)
    assert decision.action == "dispatch"
    assert [i.entry for i in decision.dispatch] == ["b/new"]
    assert "already certified by an earlier verify" in decision.detail


def test_a_gap_whose_verify_already_failed_stops():
    budget = R.RemediationBudget()
    queue = [_item("a/x", R.QUEUE_FAILED_RECEIPT)]
    incoming = (_item("a/x", values=("2",)),)
    merge = R.merge_into_queue(queue, incoming, budget=budget)
    decision = R.decide_gap_dispatch(merge, budget=budget, incoming=incoming)
    assert decision.stopping and decision.verdict == R.STOP_VERIFICATION_FAILED


def test_nothing_new_is_clean():
    budget = R.RemediationBudget()
    queue = [_item("a/x", R.QUEUE_CERTIFIED)]
    incoming = (_item("a/x"),)
    merge = R.merge_into_queue(queue, incoming, budget=budget)
    assert R.decide_gap_dispatch(merge, budget=budget, incoming=incoming).action == "clean"


# --------------------------------------------------------------- decisions


def test_the_three_roles_carry_the_gap_through_a_decision_file(tmp_path):
    ctx = progress_mod.ProgressContext(path=tmp_path / "progress.yaml")
    progress_mod.init_progress_file(ctx.path)
    for role, decision in (
        ("assembler", "UNIT_READY"),
        ("reviewer", "APPROVE"),
        ("auditor", "CLEAN"),
    ):
        path = progress_mod.decision_path(tmp_path, role, 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                {"decision": decision, "summary": "s", "certification_gap": [_GAP, {"x": 1}]}
            ),
            encoding="utf-8",
        )
        assert progress_mod.collect_decision_file(ctx, tmp_path, role, 1) == decision
        entry = progress_mod.latest_entry(ctx.path, role)
        assert entry["certification_gap"] == [
            {
                "entry": _GAP["entry"],
                "values": _GAP["values"],
                "axis": "rows",
                "reason": _GAP["reason"],
            }
        ]
    assert "certification_gap" in progress_mod.BLOCKED_REASONS


def test_the_tools_offer_the_gap_to_exactly_the_three_roles():
    for role in R.CERTIFICATION_GAP_ROLES:
        assert R.CERTIFICATION_GAP_KEY in progress_mod.EXTRA_KEYS_BY_AGENT[role]
    for role in ("smith", "plan_drafter", "plan_reviewer"):
        assert R.CERTIFICATION_GAP_KEY not in progress_mod.EXTRA_KEYS_BY_AGENT.get(role, ())


# ---------------------------------------------------------------- workflow


def _landing_smith(turn, prompt):
    target = prompt.split("the catalog entry `", 1)[1].split("`", 1)[0]
    return {"decision": "LANDED", "summary": f"certified {target}", "entry": target}


def _run(tmp_path, task_file, monkeypatch, **roles):
    workflow = _workflow(tmp_path)
    roles.setdefault("smith", _landing_smith)
    calls = _install(monkeypatch, workflow, Script(None, **roles))
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)
    return calls, load_state(tmp_path / "ws" / ".trtllm_modeling_v2_state.json")


def _roles(calls):
    return [role for role, _ in calls]


def test_an_assembler_gap_is_verified_while_the_unit_moves_on(tmp_path, task_file, monkeypatch):
    calls, state = _run(
        tmp_path,
        task_file,
        monkeypatch,
        assembler=lambda turn, prompt: {
            "decision": "UNIT_READY",
            "summary": "core assembled; silu rows uncovered",
            "certification_gap": [_GAP],
        },
    )
    assert _roles(calls) == [
        "plan_drafter",
        "plan_reviewer",
        "assembler",
        "smith",
        "reviewer",
        "auditor",
    ]
    smith = next(prompt for role, prompt in calls if role == "smith")
    assert "**verify** the catalog entry `activation/silu_and_mul`" in smith
    assert "2, 3, 6, 8192" in smith and "6·G rows" in smith
    assert state.verdict == "delivered"
    cycle = R.RemediationState.from_dict(state.remediation)
    assert [(i.entry, i.status, i.capture_id) for i in cycle.queue] == [
        ("activation/silu_and_mul", R.QUEUE_CERTIFIED, "assembler:core")
    ]


def test_a_certification_only_review_does_not_reopen_the_assembler(
    tmp_path, task_file, monkeypatch
):
    calls, state = _run(
        tmp_path,
        task_file,
        monkeypatch,
        reviewer=lambda turn, prompt: {
            "decision": "APPROVE",
            "summary": "the target code is sound; one entry needs certifying",
            "certification_gap": [_GAP],
        },
    )
    assert _roles(calls).count("assembler") == 1
    assert _roles(calls).count("smith") == 1
    assert state.verdict == "delivered"


def test_a_blocked_assembler_gets_its_verifies_and_is_resumed(tmp_path, task_file, monkeypatch):
    def assembler(turn, prompt):
        if turn == 1:
            return {
                "decision": "BLOCKED",
                "summary": "cannot pick the chunk size without knowing",
                "blocked_reason": "certification_gap",
                "certification_gap": [_GAP, {"entry": "moe/runner", "values": ["16384"]}],
            }
        return {"decision": "UNIT_READY", "summary": "assembled"}

    calls, state = _run(tmp_path, task_file, monkeypatch, assembler=assembler)
    roles = _roles(calls)
    assert roles.index("smith") < len(roles) - roles[::-1].index("assembler") - 1
    assert roles.count("smith") == 2 and roles.count("assembler") == 2
    assert state.verdict == "delivered"
    # Two surfaces, two distinct dispatch numbers: neither reads the other's decision.
    smiths = [prompt for role, prompt in calls if role == "smith"]
    numbers = sorted(int(p.split("Dispatch ", 1)[1].split(":", 1)[0]) for p in smiths)
    assert numbers == [9000, 9001]


def test_a_gap_that_fails_to_certify_goes_back_to_the_assembler(tmp_path, task_file, monkeypatch):
    def assembler(turn, prompt):
        entry = {"decision": "UNIT_READY", "summary": "assembled"}
        if turn == 1:
            entry["certification_gap"] = [_GAP]
        return entry

    calls, state = _run(
        tmp_path,
        task_file,
        monkeypatch,
        assembler=assembler,
        smith=lambda turn, prompt: {
            "decision": "FAILED_RECEIPT",
            "summary": "fails at 8192 rows",
            "entry": "activation/silu_and_mul",
        },
    )
    assert _roles(calls).count("assembler") == 2
    corrections = (Path(tmp_path) / "ws" / "corrections.md").read_text()
    assert "did not certify: activation/silu_and_mul[rows]" in corrections
    assert state.verdict == "delivered"


def test_a_blocked_gap_with_no_rows_ends_the_run_with_its_reason(tmp_path, task_file, monkeypatch):
    calls, state = _run(
        tmp_path,
        task_file,
        monkeypatch,
        assembler=lambda turn, prompt: {
            "decision": "BLOCKED",
            "summary": "needs verifies",
            "blocked_reason": "certification_gap",
        },
    )
    assert state.verdict == "certification-gap-unnamed"
    assert "smith" not in _roles(calls)


def test_an_auditor_gap_runs_the_remediation_cycle(tmp_path, task_file, monkeypatch):
    calls, state = _run(
        tmp_path,
        task_file,
        monkeypatch,
        auditor=lambda turn, prompt: (
            {"decision": "UNCOVERED", "summary": "one gap", "certification_gap": [_GAP]}
            if turn == 1
            else {"decision": "CLEAN", "summary": "covered"}
        ),
    )
    assert _roles(calls).count("auditor") == 2 and _roles(calls).count("smith") == 1
    assert state.verdict == "delivered"


def test_the_roles_are_told_to_report_gaps_rather_than_reject():
    from agent_flow.workflows.trtllm_modeling_v2.prompts import DEFAULT_PROMPTS

    assert "is a certification gap, not a REJECT" in DEFAULT_PROMPTS.reviewer
    assert "`certification_gap`" in DEFAULT_PROMPTS.assembler
    assert "`certification_gap`" in DEFAULT_PROMPTS.auditor
