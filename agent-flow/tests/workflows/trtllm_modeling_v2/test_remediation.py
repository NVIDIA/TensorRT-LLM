# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The convergence rules of the UNCOVERED remediation cycle.

These are the tests the whole module exists for. The loop spends GPU time,
so "does it stop, and does it stop for a nameable reason" is the property
that matters, and it is decidable on a laptop because every rule here is a
pure function over plain data.

The four failure modes being ruled out, each of which is a real way a
naive `while auditor == UNCOVERED` loop burns a night:

  * one smith per value instead of one per surface;
  * re-dispatching a finding that is already queued;
  * re-dispatching a surface that was already certified;
  * running forever because the auditor keeps saying the same thing.
"""

from __future__ import annotations

from agent_flow.workflows.trtllm_modeling_v2 import remediation as R
from agent_flow.workflows.trtllm_modeling_v2.progress import parse_uncovered_records
from agent_flow.workflows.trtllm_modeling_v2.state import is_remediable

# ------------------------------------------------------------------- parsing


def test_a_row_without_entry_arch_or_axis_is_dropped_not_guessed():
    """A guessed axis merges unrelated surfaces and certifies neither."""
    rows, dropped = R.parse_uncovered_rows(
        [
            {"entry": "attention/x", "arch": "sm_103", "axis": "batch_size", "values": [1]},
            {"entry": "attention/x", "arch": "sm_103"},  # no axis
            {"arch": "sm_103", "axis": "batch_size"},  # no entry
            "not a mapping",
        ]
    )
    assert len(rows) == 1
    assert dropped == 3


def test_a_single_value_is_accepted_in_either_spelling():
    rows, _ = R.parse_uncovered_rows([{"entry": "e", "arch": "a", "axis": "x", "value": 7}])
    assert rows[0].values == ("7",)


def test_values_are_canonicalized_so_two_spellings_hash_alike():
    """``[12, 3, 5]`` and ``[5, 3, 12]`` are the same gap."""
    one, _ = R.parse_uncovered_rows(
        [{"entry": "e", "arch": "a", "axis": "x", "values": [12, 3, 5]}]
    )
    two, _ = R.parse_uncovered_rows(
        [{"entry": "e", "arch": "a", "axis": "x", "values": [5, 12, 3]}]
    )
    assert one[0].values == two[0].values == ("3", "5", "12")


def test_numeric_values_sort_numerically_not_lexically():
    """Lexical order would put 12 before 3 and change the identity."""
    rows, _ = R.parse_uncovered_rows(
        [{"entry": "e", "arch": "a", "axis": "x", "values": [2, 31, 9]}]
    )
    assert rows[0].values == ("2", "9", "31")


def test_the_decision_file_parser_agrees_with_the_typed_one():
    """`progress` normalizes for persistence; `remediation` types it. Same rule."""
    records = parse_uncovered_records(
        [
            {"entry": "e", "arch": "a", "axis": "x", "values": [3], "contract_line": "L"},
            {"entry": "e", "arch": "a"},
        ]
    )
    assert len(records) == 1
    rows, dropped = R.parse_uncovered_rows(records)
    assert dropped == 0 and rows[0].contract_line == "L"


# -------------------------------------------------------------- canonicalize


def _rows(*specs):
    return [
        R.UncoveredRow(entry=e, arch=a, axis=x, values=tuple(str(v) for v in vals))
        for e, a, x, vals in specs
    ]


def test_seven_uncovered_values_on_one_entry_are_one_dispatch():
    """The whole reason the queue is made of surfaces rather than values."""
    rows = _rows(
        ("attention/paged", "sm_103", "batch_size", [3, 5, 7]),
        ("attention/paged", "sm_103", "batch_size", [9, 12, 15, 31]),
    )
    items = R.canonicalize(rows)
    assert len(items) == 1
    assert items[0].values == ("3", "5", "7", "9", "12", "15", "31")


def test_different_axes_on_one_entry_stay_separate():
    """They are different tests and different contract lines."""
    items = R.canonicalize(
        _rows(
            ("attention/paged", "sm_103", "batch_size", [3]),
            ("attention/paged", "sm_103", "num_pages", [64]),
        )
    )
    assert len(items) == 2


def test_the_same_surface_on_two_architectures_is_two_items():
    """Receipts are per-architecture; absent means unknown."""
    items = R.canonicalize(
        _rows(
            ("norm/rms", "sm_103", "hidden", [4096]),
            ("norm/rms", "sm_100", "hidden", [4096]),
        )
    )
    assert len(items) == 2


def test_identity_is_stable_across_processes():
    a = R.canonicalize(_rows(("e", "a", "x", [1, 2])), target_commit="c", capture_id="k")
    b = R.canonicalize(_rows(("e", "a", "x", [2, 1])), target_commit="c", capture_id="k")
    assert a[0].identity == b[0].identity


def test_a_new_commit_or_capture_makes_it_new_work():
    """A wrapper change can open a runtime branch that did not exist before."""
    base = R.canonicalize(_rows(("e", "a", "x", [1])), target_commit="c1", capture_id="k")
    newer = R.canonicalize(_rows(("e", "a", "x", [1])), target_commit="c2", capture_id="k")
    other = R.canonicalize(_rows(("e", "a", "x", [1])), target_commit="c1", capture_id="k2")
    assert base[0].identity != newer[0].identity != other[0].identity


def test_the_brief_carries_the_whole_surface_and_the_contract_line():
    rows = [
        R.UncoveredRow(
            entry="e",
            arch="sm_103",
            axis="batch_size",
            values=("3", "5"),
            contract_line="certified at 1, 2, 4",
        )
    ]
    brief = R.canonicalize(rows)[0].brief()
    assert "batch_size" in brief and "3, 5" in brief
    assert "certified at 1, 2, 4" in brief
    assert "sm_103 receipt" in brief


# --------------------------------------------------------------------- merge


def test_genuinely_new_work_is_appended():
    merge = R.merge_into_queue([], R.canonicalize(_rows(("e", "a", "x", [1]))))
    assert len(merge.appended) == 1 and merge.has_new_work


def test_a_finding_already_queued_is_not_dispatched_twice():
    first = R.canonicalize(_rows(("e", "a", "x", [1, 2])))
    merge = R.merge_into_queue(first, first)
    assert merge.duplicates and not merge.appended and not merge.has_new_work


def test_a_wider_surface_widens_the_open_item_rather_than_adding_one():
    """Two dispatches on one entry would collide on the same file."""
    queue = R.canonicalize(_rows(("e", "a", "x", [1])))
    merge = R.merge_into_queue(queue, R.canonicalize(_rows(("e", "a", "x", [1, 5]))))
    assert len(merge.queue) == 1
    assert merge.widened and merge.queue[0].values == ("1", "5")


def test_a_certified_surface_coming_back_is_a_regression_not_a_redispatch():
    """Re-running the same smith cannot fix a stale receipt or a bad read."""
    queue = [
        R.canonicalize(_rows(("e", "a", "x", [1])))[0].__class__(
            entry="e", arch="a", axis="x", values=("1",), status=R.QUEUE_CERTIFIED
        )
    ]
    merge = R.merge_into_queue(queue, R.canonicalize(_rows(("e", "a", "x", [1]))))
    assert merge.regressions and not merge.appended


def test_a_surface_whose_attempt_failed_is_not_retried():
    queue = [
        R.RemediationItem(
            entry="e", arch="a", axis="x", values=("1",), status=R.QUEUE_FAILED_RECEIPT
        )
    ]
    merge = R.merge_into_queue(queue, R.canonicalize(_rows(("e", "a", "x", [1, 2]))))
    assert merge.exhausted and not merge.appended and not merge.widened


def test_the_new_item_cap_reports_rather_than_silently_dropping():
    """Proceeding on a partial set would converge on an incomplete audit."""
    incoming = R.canonicalize(_rows(*[(f"e{i}", "a", "x", [1]) for i in range(5)]))
    merge = R.merge_into_queue([], incoming, budget=R.RemediationBudget(max_new_items=2))
    assert len(merge.appended) == 2
    assert len(merge.exhausted) == 3


# ----------------------------------------------------------------- signature


def test_the_signature_ignores_closed_items():
    """Only unresolved work decides whether the cycle is shrinking."""
    open_only = [R.RemediationItem(entry="e", arch="a", axis="x", values=("1",))]
    plus_done = open_only + [
        R.RemediationItem(entry="z", arch="a", axis="x", values=("9",), status=R.QUEUE_CERTIFIED)
    ]
    assert R.unresolved_signature(open_only) == R.unresolved_signature(plus_done)


def test_an_empty_queue_has_no_signature():
    assert R.unresolved_signature([]) == ""


# --------------------------------------------------------------- round plan


def _merge_of(*specs, queue=()):
    return R.merge_into_queue(list(queue), R.canonicalize(_rows(*specs)))


def test_nothing_open_means_the_audit_is_closed():
    decision = R.decide_round(R.merge_into_queue([], []), budget=R.RemediationBudget(), round_no=1)
    assert decision.action == "clean"


def test_new_work_dispatches_capped_at_the_parallel_budget():
    merge = _merge_of(*[(f"e{i}", "a", "x", [1]) for i in range(6)])
    decision = R.decide_round(
        merge, budget=R.RemediationBudget(max_parallel_verifiers=2), round_no=1
    )
    assert decision.action == "dispatch"
    assert len(decision.dispatch) == 6
    assert decision.workers == 2


def test_a_regression_stops_the_cycle_whatever_else_is_true():
    """Evidence that cannot be trusted is not repaired by more rounds."""
    queue = [
        R.RemediationItem(entry="e", arch="a", axis="x", values=("1",), status=R.QUEUE_CERTIFIED)
    ]
    merge = _merge_of(("e", "a", "x", [1]), ("new", "a", "x", [2]), queue=queue)
    decision = R.decide_round(merge, budget=R.RemediationBudget(), round_no=1)
    assert decision.stopping
    assert decision.verdict == R.STOP_NO_PROGRESS
    assert "certified in an earlier round" in decision.detail


def test_an_auditor_repeating_itself_stops_the_cycle():
    queue = R.canonicalize(_rows(("e", "a", "x", [1])))
    merge = R.merge_into_queue(queue, queue)
    decision = R.decide_round(merge, budget=R.RemediationBudget(), round_no=2)
    assert decision.stopping and decision.verdict == R.STOP_NO_PROGRESS


def test_spending_the_round_budget_stops_with_its_own_verdict():
    merge = _merge_of(("e", "a", "x", [1]))
    decision = R.decide_round(merge, budget=R.RemediationBudget(max_rounds=1), round_no=2)
    assert decision.stopping and decision.verdict == R.STOP_BUDGET


def test_an_unchanged_unresolved_set_stops_even_when_rows_look_new():
    """Monotone convergence is the contract, not a heuristic."""
    merge = _merge_of(("e", "a", "x", [1]))
    signature = R.unresolved_signature(merge.queue)
    decision = R.decide_round(
        merge, budget=R.RemediationBudget(), round_no=1, previous_signature=signature
    )
    assert decision.stopping and decision.verdict == R.STOP_NO_PROGRESS
    assert "byte-identical" in decision.detail


# ------------------------------------------------------------------ outcomes


def test_a_landed_verification_certifies_its_surface():
    queue = R.canonicalize(_rows(("e", "a", "x", [1])))
    updated, failed = R.apply_outcomes(queue, {queue[0].identity: "LANDED"})
    assert updated[0].status == R.QUEUE_CERTIFIED
    assert updated[0].attempts == 1
    assert not failed


def test_a_failed_receipt_is_terminal_and_reported():
    queue = R.canonicalize(_rows(("e", "a", "x", [1])))
    updated, failed = R.apply_outcomes(queue, {queue[0].identity: "FAILED_RECEIPT"})
    assert updated[0].status == R.QUEUE_FAILED_RECEIPT and len(failed) == 1


def test_a_verification_that_said_nothing_counts_as_blocked_not_as_success():
    """Treating silence as success is how an uncertified surface reaches a merge."""
    queue = R.canonicalize(_rows(("e", "a", "x", [1])))
    updated, failed = R.apply_outcomes(queue, {})
    assert updated[0].status == R.QUEUE_BLOCKED and len(failed) == 1


def test_already_closed_items_are_left_alone():
    queue = [
        R.RemediationItem(
            entry="e", arch="a", axis="x", values=("1",), status=R.QUEUE_CERTIFIED, attempts=1
        )
    ]
    updated, failed = R.apply_outcomes(queue, {})
    assert updated[0].status == R.QUEUE_CERTIFIED and updated[0].attempts == 1
    assert not failed


# ----------------------------------------------------------------- checkpoint


def test_the_cycle_survives_a_resume_byte_for_byte():
    """A resume that forgets the queue re-dispatches work it already did."""
    queue = list(R.canonicalize(_rows(("e", "a", "x", [3, 1])), target_commit="c"))
    state = R.RemediationState(queue=queue, round_no=2, signature="abc123")
    restored = R.RemediationState.from_dict(state.as_dict())
    assert restored.round_no == 2 and restored.signature == "abc123"
    assert [i.identity for i in restored.queue] == [i.identity for i in queue]


def test_a_checkpoint_from_before_the_cycle_existed_loads_as_empty():
    assert R.RemediationState.from_dict(None).queue == []
    assert R.RemediationState.from_dict({}).round_no == 0


def test_every_stop_verdict_is_re_enterable():
    """Each names a repair made outside the run, and the queue is persisted."""
    for verdict in (R.STOP_NO_PROGRESS, R.STOP_BUDGET, R.STOP_VERIFICATION_FAILED):
        assert is_remediable(verdict), verdict


# ------------------------------------------------------ the cycle, end to end
#
# Everything above is a pure function. These drive the real orchestrator, so
# they are what catches the wiring: a queue that is never checkpointed, a
# verify dispatch that never runs, a loop that re-audits without re-reading
# delivery.

import json  # noqa: E402 -- the end-to-end half needs the harness below

import pytest  # noqa: E402

from agent_flow.workflows.trtllm_modeling_v2.state import load_state  # noqa: E402

from .test_workflow import Script, _install, _workflow, _write_vocabulary  # noqa: E402

_ROW = {
    "entry": "attention/thop_attention",
    "arch": "sm_103",
    "axis": "batch_size",
    "values": ["3", "5", "7", "9", "12", "15", "31"],
    "contract_line": "certified at batch sizes 1, 2, 4, 8 on sm_103",
}


def _drive(tmp_path, task_file, monkeypatch, *, auditor, smith=None):
    """Run the whole workflow with a scripted auditor and smith."""
    workflow = _workflow(tmp_path)
    script = Script(None, auditor=auditor, **({"smith": smith} if smith else {}))
    calls = _install(monkeypatch, workflow, script)
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)
    return workflow, calls


def _state(tmp_path):
    return load_state(tmp_path / "ws" / ".trtllm_modeling_v2_state.json")


def test_a_converging_cycle_certifies_and_then_delivers(tmp_path, task_file, monkeypatch):
    """The loop this module exists to create: find, fix, re-check, ship.

    Before this, round one ended the run and a human did the rest.
    """
    workflow, calls = _drive(
        tmp_path,
        task_file,
        monkeypatch,
        auditor=lambda turn, prompt: (
            {"decision": "UNCOVERED", "summary": "7 values outside", "uncovered": [_ROW]}
            if turn == 1
            else {"decision": "CLEAN", "summary": "all covered"}
        ),
        smith=lambda turn, prompt: {
            "decision": "LANDED",
            "summary": "certified attention/thop_attention",
            "entry": "attention/thop_attention",
        },
    )
    state = _state(tmp_path)
    assert state.done and state.verdict == "delivered"
    assert state.audit_clean

    # Two auditor turns and exactly one verify dispatch.
    assert [n for n, _ in calls].count("auditor") == 2
    assert [n for n, _ in calls].count("smith") == 1

    cycle = R.RemediationState.from_dict(state.remediation)
    assert [i.status for i in cycle.queue] == [R.QUEUE_CERTIFIED]


def test_seven_uncovered_values_produce_one_smith_not_seven(tmp_path, task_file, monkeypatch):
    """The cost saving that motivated the whole design."""
    _, calls = _drive(
        tmp_path,
        task_file,
        monkeypatch,
        auditor=lambda turn, prompt: (
            {"decision": "UNCOVERED", "summary": "s", "uncovered": [_ROW]}
            if turn == 1
            else {"decision": "CLEAN", "summary": "s"}
        ),
        smith=lambda turn, prompt: {
            "decision": "LANDED",
            "summary": "ok",
            "entry": "attention/thop_attention",
        },
    )
    smith_prompts = [p for n, p in calls if n == "smith"]
    assert len(smith_prompts) == 1
    # ...and that one smith was handed the whole surface.
    assert "3, 5, 7, 9, 12, 15, 31" in smith_prompts[0]
    assert "certified at batch sizes 1, 2, 4, 8" in smith_prompts[0]


def test_an_auditor_that_repeats_itself_stops_instead_of_looping(tmp_path, task_file, monkeypatch):
    """The failure the bound exists for: a cycle that cannot converge.

    The auditor reports the identical surface forever and the smith keeps
    claiming success. Without the convergence rule this runs until someone
    notices.
    """
    _, calls = _drive(
        tmp_path,
        task_file,
        monkeypatch,
        auditor=lambda turn, prompt: {
            "decision": "UNCOVERED",
            "summary": "same as last time",
            "uncovered": [_ROW],
        },
        smith=lambda turn, prompt: {
            "decision": "LANDED",
            "summary": "ok",
            "entry": "attention/thop_attention",
        },
    )
    state = _state(tmp_path)
    assert state.verdict == R.STOP_NO_PROGRESS
    # It stopped early rather than spending the round budget.
    assert [n for n, _ in calls].count("smith") == 1
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "certified in an earlier round" in journal


def test_a_verification_that_cannot_land_stops_with_its_own_verdict(
    tmp_path, task_file, monkeypatch
):
    _, calls = _drive(
        tmp_path,
        task_file,
        monkeypatch,
        auditor=lambda turn, prompt: {"decision": "UNCOVERED", "summary": "s", "uncovered": [_ROW]},
        smith=lambda turn, prompt: {
            "decision": "FAILED_RECEIPT",
            "summary": "the test does not pass at batch 31",
            "entry": "attention/thop_attention",
        },
    )
    state = _state(tmp_path)
    assert state.verdict == R.STOP_VERIFICATION_FAILED
    assert [n for n, _ in calls].count("smith") == 1
    cycle = R.RemediationState.from_dict(state.remediation)
    assert cycle.queue[0].status == R.QUEUE_FAILED_RECEIPT


def test_an_uncovered_with_no_structured_rows_keeps_the_old_behaviour(
    tmp_path, task_file, monkeypatch
):
    """Backward compatible on purpose.

    An auditor that reports prose only has said something real; the run must
    not treat it as clean. It ends exactly as it did before the cycle
    existed.
    """
    _drive(
        tmp_path,
        task_file,
        monkeypatch,
        auditor=lambda turn, prompt: {
            "decision": "UNCOVERED",
            "summary": "batch 257 outside the column",
        },
    )
    assert _state(tmp_path).verdict == "audit-uncovered"


def test_the_queue_is_checkpointed_so_a_resume_does_not_redo_it(tmp_path, task_file, monkeypatch):
    """The manual loop's real cost was re-deriving what was already known."""
    _drive(
        tmp_path,
        task_file,
        monkeypatch,
        auditor=lambda turn, prompt: {"decision": "UNCOVERED", "summary": "s", "uncovered": [_ROW]},
        smith=lambda turn, prompt: {
            "decision": "FAILED_RECEIPT",
            "summary": "no",
            "entry": "attention/thop_attention",
        },
    )
    raw = json.loads((tmp_path / "ws" / ".trtllm_modeling_v2_state.json").read_text())
    assert raw["remediation"]["queue"], "the queue never reached the checkpoint"
    assert raw["remediation"]["round_no"] >= 1
    restored = R.RemediationState.from_dict(raw["remediation"])
    assert restored.queue[0].entry == "attention/thop_attention"


@pytest.mark.parametrize("budget", [{"max_rounds": 0}, {"max_new_items": -1}])
def test_a_budget_of_zero_or_less_is_refused_by_the_schema(budget, tmp_path):
    """There is no spelling for 'unlimited', by design."""
    import yaml

    from agent_flow.workflows.trtllm_modeling_v2.task_schema import (
        TaskSchemaError,
        load_and_validate_task_yaml,
    )

    spec = {
        "checkpoint_path": str(tmp_path),
        "arch": "sm_103",
        "parallel": "tp1",
        "trtllm_repo_path": str(tmp_path),
        "catalog_remediation": budget,
    }
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(TaskSchemaError, match="catalog_remediation"):
        load_and_validate_task_yaml(path)
