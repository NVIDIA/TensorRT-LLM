# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A smith failure only a later unit pays for must not stop core.

Measured on the DeepSeek-V4.1 run: the plan queued 33 dispatches, six of them
needed only by later feature units, and the smith phase closed the run
`smith-blocked` on ANY failure -- before core, which needed none of those six,
had started. Queuing them up front is still right, since they run alongside
core's for free; what changes is what their failure blocks.

Same harness as `test_workflow.py`: no agent is constructed, and the fake
writes the typed decision the real role would. The smith here is scripted per
target and per attempt, so a test can say "fails once, lands on the retry".
"""

from __future__ import annotations

import json
import re

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.state import (
    ITEM_DEFERRED,
    ITEM_DONE,
    ITEM_FAILED,
    STAGE_ASSEMBLE,
    STAGE_SMITH,
    load_state,
)

from .test_workflow import Script, _install, _plan_then_vocabulary, _workflow

_STATE = ".trtllm_modeling_v2_state.json"


def _features(task_file, features):
    spec = yaml.safe_load(task_file.read_text())
    spec["features"] = list(features)
    task_file.write_text(yaml.safe_dump(spec), encoding="utf-8")


def _smith(outcomes: dict[str, list[str]] | None = None):
    """A smith scripted per target and per attempt; ``LANDED`` when unscripted.

    It stamps the dispatch number it was given as its ``iteration``, which is
    what lets the orchestrator tell its entry apart from a concurrent
    sibling's -- the real collector does the same.
    """
    outcomes = outcomes or {}
    attempts: dict[str, int] = {}

    def smith(turn, prompt):
        n = int(re.search(r"Dispatch (\d+):", prompt).group(1))
        target = re.search(r"the catalog entry `([^`]+)`", prompt).group(1)
        attempts[target] = attempts.get(target, 0) + 1
        script = outcomes.get(target, ["LANDED"])
        decision = script[min(attempts[target], len(script)) - 1]
        return {"iteration": n, "decision": decision, "summary": f"{decision} {target}"}

    return smith


def _smith_calls(calls, target=None):
    return [
        (index, prompt)
        for index, (role, prompt) in enumerate(calls)
        if role == "smith" and (target is None or f"catalog entry `{target}`" in prompt)
    ]


def _assembler_index(calls, unit):
    return next(
        index
        for index, (role, prompt) in enumerate(calls)
        if role == "assembler" and f"Unit: **{unit}**" in prompt
    )


def _run(tmp_path, task_file, monkeypatch, queue, smith, **roles):
    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None, smith=smith, **roles))
    _plan_then_vocabulary(workflow, tmp_path, monkeypatch, queue)
    with workflow:
        workflow.run(task_file)
    return calls, load_state(tmp_path / "ws" / _STATE)


def _row(state, n):
    return next(row for row in state.queue if row["n"] == n)


# ----------------------------------------------------------- core still blocks


def test_a_core_failure_still_blocks_before_assembly(tmp_path, task_file, monkeypatch):
    """An untagged row is core's, and core cannot assemble without it."""
    _features(task_file, ["mtp"])
    queue = [
        {"n": 1, "mode": "onboard", "target": "norm/core_only"},
        {"n": 2, "mode": "onboard", "target": "moe/drafter_moe", "unit": "mtp"},
    ]
    calls, state = _run(
        tmp_path, task_file, monkeypatch, queue, _smith({"norm/core_only": ["BLOCKED"]})
    )

    assert "assembler" not in [role for role, _ in calls]
    assert state.verdict == "smith-blocked"
    assert _row(state, 1)["status"] == ITEM_FAILED


def test_a_dependency_of_a_core_row_is_core_whatever_its_tag(tmp_path, task_file, monkeypatch):
    """Core's own entries were built on it, so deferring it would start core without it."""
    _features(task_file, ["mtp"])
    queue = [
        {"n": 1, "mode": "onboard", "target": "quant/shared_packing", "unit": "mtp"},
        {"n": 2, "mode": "onboard", "target": "attn/core_logits", "depends_on": [1]},
    ]
    calls, state = _run(
        tmp_path, task_file, monkeypatch, queue, _smith({"quant/shared_packing": ["BLOCKED"]})
    )

    assert state.verdict == "smith-blocked"
    assert "assembler" not in [role for role, _ in calls]
    # The core row that waits on it is never dispatched: there is nothing to
    # build it on.
    assert not _smith_calls(calls, "attn/core_logits")


# ------------------------------------------------- feature-only rows defer


def test_a_feature_only_failure_is_deferred_and_core_proceeds(tmp_path, task_file, monkeypatch):
    _features(task_file, ["mtp"])
    queue = [
        {"n": 1, "mode": "onboard", "target": "norm/core_only"},
        {"n": 2, "mode": "onboard", "target": "moe/drafter_moe", "unit": "mtp"},
    ]
    calls, state = _run(
        tmp_path,
        task_file,
        monkeypatch,
        queue,
        _smith({"moe/drafter_moe": ["BLOCKED", "LANDED"]}),
    )

    assert "Unit: **core**" in calls[_assembler_index(calls, "core")][1]
    assert state.units_done == ["core", "mtp"]
    assert state.verdict == "delivered"
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "smith dispatches deferred" in journal
    assert "2:moe/drafter_moe → mtp" in journal


def test_a_deferred_row_is_retried_once_before_its_unit(tmp_path, task_file, monkeypatch):
    """After core is green, before the first mtp turn -- and told what failed."""
    _features(task_file, ["mtp"])
    queue = [
        {"n": 1, "mode": "onboard", "target": "norm/core_only"},
        {"n": 2, "mode": "onboard", "target": "moe/drafter_moe", "unit": "mtp"},
    ]
    calls, state = _run(
        tmp_path,
        task_file,
        monkeypatch,
        queue,
        _smith({"moe/drafter_moe": ["BLOCKED", "LANDED"]}),
    )

    attempts = _smith_calls(calls, "moe/drafter_moe")
    assert len(attempts) == 2
    retry_at, retry_prompt = attempts[1]
    assert _assembler_index(calls, "core") < retry_at < _assembler_index(calls, "mtp")
    assert "This is a retry" in retry_prompt
    assert "smith decided BLOCKED" in retry_prompt
    # Pointed at the report the first attempt left, by its real path.
    report = re.search(r"previous attempt's report is `([^`]+)`", retry_prompt).group(1)
    assert "smith-onboard-moe-drafter-moe" in report
    first_report = open(report, encoding="utf-8").read()
    assert "Dispatch 2: **onboard** the catalog entry `moe/drafter_moe`" in first_report
    assert "This is a retry" not in first_report
    # The first attempt was an ordinary dispatch: no retry note.
    assert "This is a retry" not in attempts[0][1]
    assert _row(state, 2)["status"] == ITEM_DONE


def test_a_retry_that_fails_again_closes_the_run_naming_its_unit(tmp_path, task_file, monkeypatch):
    _features(task_file, ["mtp"])
    queue = [
        {"n": 1, "mode": "onboard", "target": "norm/core_only"},
        {"n": 2, "mode": "onboard", "target": "moe/drafter_moe", "unit": "mtp"},
    ]
    calls, state = _run(
        tmp_path, task_file, monkeypatch, queue, _smith({"moe/drafter_moe": ["BLOCKED"]})
    )

    assert state.verdict == "smith-blocked:mtp"
    # Core was built and gated; the stop is at mtp, before its first turn.
    assert state.units_done == ["core"]
    assert [p for r, p in calls if r == "assembler" and "Unit: **mtp**" in p] == []
    assert len(_smith_calls(calls, "moe/drafter_moe")) == 2, "one retry, not more"
    assert _row(state, 2)["status"] == ITEM_FAILED


def test_a_row_waiting_on_a_deferred_row_is_deferred_undispatched(tmp_path, task_file, monkeypatch):
    """It is dispatched for the first time at its unit, after what it waits on lands."""
    _features(task_file, ["mtp", "eplb"])
    queue = [
        {"n": 1, "mode": "onboard", "target": "norm/core_only"},
        {"n": 2, "mode": "onboard", "target": "moe/drafter_moe", "unit": "mtp"},
        {
            "n": 3,
            "mode": "onboard",
            "target": "moe/balancer_slots",
            "unit": "eplb",
            "depends_on": [2],
        },
    ]
    calls, state = _run(
        tmp_path,
        task_file,
        monkeypatch,
        queue,
        _smith({"moe/drafter_moe": ["BLOCKED", "LANDED"]}),
    )

    waiting = _smith_calls(calls, "moe/balancer_slots")
    assert len(waiting) == 1, "held back in the smith phase, then dispatched once"
    # Dispatched at eplb's start: after mtp was assembled, before eplb's turn.
    assert _assembler_index(calls, "mtp") < waiting[0][0] < _assembler_index(calls, "eplb")
    # Its first attempt, so there is no earlier report to point it at.
    assert "This is a retry" not in waiting[0][1]
    assert state.verdict == "delivered"
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "3:moe/balancer_slots → eplb (not dispatched: waits on [2])" in journal


# ---------------------------------------------------------------------- resume


def test_a_resume_keeps_deferrals_and_does_not_redispatch_landed_rows(
    tmp_path, task_file, monkeypatch
):
    _features(task_file, ["mtp"])
    queue = [
        {"n": 1, "mode": "onboard", "target": "norm/core_only"},
        {"n": 2, "mode": "onboard", "target": "moe/drafter_moe", "unit": "mtp"},
    ]

    def die(turn, prompt):
        raise KeyboardInterrupt

    first = _workflow(tmp_path)
    _install(
        monkeypatch,
        first,
        Script(None, smith=_smith({"moe/drafter_moe": ["BLOCKED"]}), reviewer=die),
    )
    _plan_then_vocabulary(first, tmp_path, monkeypatch, queue)
    with pytest.raises(KeyboardInterrupt), first:
        first.run(task_file)

    saved = load_state(tmp_path / "ws" / _STATE)
    assert saved.stage == STAGE_ASSEMBLE
    assert _row(saved, 1)["status"] == ITEM_DONE
    assert _row(saved, 2)["status"] == ITEM_DEFERRED
    assert _row(saved, 2)["deferred_to"] == "mtp"

    second = _workflow(tmp_path)
    calls = _install(monkeypatch, second, Script(None, smith=_smith()))
    with second:
        second.run(task_file)

    assert not _smith_calls(calls, "norm/core_only"), "a landed row is never re-dispatched"
    retried = _smith_calls(calls, "moe/drafter_moe")
    assert len(retried) == 1
    assert "This is a retry" in retried[0][1]
    assert retried[0][0] < _assembler_index(calls, "mtp")
    assert load_state(tmp_path / "ws" / _STATE).verdict == "delivered"


def test_a_resume_after_the_retry_landed_dispatches_no_smith(tmp_path, task_file, monkeypatch):
    """The retry is owed once. A restart inside the unit it served pays nothing again."""
    _features(task_file, ["mtp"])
    queue = [
        {"n": 1, "mode": "onboard", "target": "norm/core_only"},
        {"n": 2, "mode": "onboard", "target": "moe/drafter_moe", "unit": "mtp"},
    ]

    def assembler(turn, prompt):
        if "Unit: **mtp**" in prompt:
            raise KeyboardInterrupt
        return {"decision": "UNIT_READY", "summary": "assembled"}

    first = _workflow(tmp_path)
    _install(
        monkeypatch,
        first,
        Script(
            None,
            smith=_smith({"moe/drafter_moe": ["BLOCKED", "LANDED"]}),
            assembler=assembler,
        ),
    )
    _plan_then_vocabulary(first, tmp_path, monkeypatch, queue)
    with pytest.raises(KeyboardInterrupt), first:
        first.run(task_file)
    assert _row(load_state(tmp_path / "ws" / _STATE), 2)["status"] == ITEM_DONE

    second = _workflow(tmp_path)
    calls = _install(monkeypatch, second, Script(None, smith=_smith()))
    with second:
        second.run(task_file)

    assert _smith_calls(calls) == []
    assert load_state(tmp_path / "ws" / _STATE).verdict == "delivered"


def test_a_retry_interrupted_in_flight_is_still_owed(tmp_path, task_file, monkeypatch):
    """A deferred row checkpointed `running` died mid-retry; its sibling that landed did not.

    The walk marks a batch `running` and checkpoints before it dispatches, so
    that is the state a crash inside the retry batch leaves behind.
    """
    _features(task_file, ["mtp"])
    queue = [
        {"n": 1, "mode": "onboard", "target": "norm/core_only"},
        {"n": 2, "mode": "onboard", "target": "moe/drafter_moe", "unit": "mtp"},
        {"n": 3, "mode": "onboard", "target": "moe/drafter_router", "unit": "mtp"},
    ]

    def die(turn, prompt):
        raise KeyboardInterrupt

    first = _workflow(tmp_path)
    blocked = {"moe/drafter_moe": ["BLOCKED"], "moe/drafter_router": ["BLOCKED"]}
    _install(monkeypatch, first, Script(None, smith=_smith(blocked), reviewer=die))
    _plan_then_vocabulary(first, tmp_path, monkeypatch, queue)
    with pytest.raises(KeyboardInterrupt), first:
        first.run(task_file)

    state_path = tmp_path / "ws" / _STATE
    data = json.loads(state_path.read_text(encoding="utf-8"))
    for row in data["queue"]:
        if row["n"] == 2:
            row["status"] = ITEM_DONE
        if row["n"] == 3:
            assert row["status"] == ITEM_DEFERRED
            row["status"] = "running"
    state_path.write_text(json.dumps(data), encoding="utf-8")

    second = _workflow(tmp_path)
    calls = _install(monkeypatch, second, Script(None, smith=_smith()))
    with second:
        second.run(task_file)

    assert not _smith_calls(calls, "norm/core_only")
    assert not _smith_calls(calls, "moe/drafter_moe")
    router = _smith_calls(calls, "moe/drafter_router")
    assert len(router) == 1
    assert router[0][0] < _assembler_index(calls, "mtp")
    assert load_state(state_path).verdict == "delivered"


# ------------------------------- applying the change to a run already closed


def _strip_units(state_path):
    """Make the checkpoint look like one written before `unit` existed."""
    data = json.loads(state_path.read_text(encoding="utf-8"))
    for row in data["queue"]:
        row.pop("unit", None)
    state_path.write_text(json.dumps(data), encoding="utf-8")


def _close_smith_blocked(tmp_path, task_file, monkeypatch, queue):
    workflow = _workflow(tmp_path)
    _install(monkeypatch, workflow, Script(None, smith=_smith({"moe/drafter_moe": ["BLOCKED"]})))
    _plan_then_vocabulary(workflow, tmp_path, monkeypatch, queue)
    with workflow:
        workflow.run(task_file)
    state_path = tmp_path / "ws" / _STATE
    closed = load_state(state_path)
    assert closed.verdict == "smith-blocked" and closed.stage == STAGE_SMITH
    _strip_units(state_path)
    return state_path


def _tag_vocabulary(tmp_path, tags):
    path = tmp_path / "ws" / "vocabulary.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    for row in doc["queue"]:
        if row["n"] in tags:
            row["unit"] = tags[row["n"]]
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")


def test_tagging_a_closed_runs_vocabulary_re_enters_it(tmp_path, task_file, monkeypatch):
    """The operator's path for a run whose checkpoint predates units.

    The run closed `smith-blocked` over a feature-only row. Tagging that row in
    `vocabulary.yaml` and re-running is the whole repair: the failed row is
    deferred without a second dispatch now, the row the close left unstarted
    is dispatched, core assembles, and the retry happens at mtp.
    """
    _features(task_file, ["mtp"])
    queue = [
        {"n": 1, "mode": "onboard", "target": "norm/core_only"},
        {"n": 2, "mode": "onboard", "target": "moe/drafter_moe"},
        {"n": 3, "mode": "onboard", "target": "attn/core_after", "depends_on": [1]},
    ]
    state_path = _close_smith_blocked(tmp_path, task_file, monkeypatch, queue)
    assert _row(load_state(state_path), 3)["status"] == "pending"
    _tag_vocabulary(tmp_path, {2: "mtp"})

    second = _workflow(tmp_path)
    calls = _install(monkeypatch, second, Script(None, smith=_smith()))
    with second:
        second.run(task_file)

    assert not _smith_calls(calls, "norm/core_only")
    assert len(_smith_calls(calls, "attn/core_after")) == 1
    retried = _smith_calls(calls, "moe/drafter_moe")
    assert len(retried) == 1, "deferred without a second smith-phase dispatch"
    assert _assembler_index(calls, "core") < retried[0][0] < _assembler_index(calls, "mtp")
    final = load_state(state_path)
    assert final.verdict == "delivered"
    assert _row(final, 2)["unit"] == "mtp"


def test_a_closed_run_stays_closed_when_core_needed_the_failure(tmp_path, task_file, monkeypatch):
    """No tag, or a tag that does not cover the failure: re-running still refuses."""
    _features(task_file, ["mtp"])
    queue = [
        {"n": 1, "mode": "onboard", "target": "norm/core_only"},
        {"n": 2, "mode": "onboard", "target": "moe/drafter_moe"},
    ]
    state_path = _close_smith_blocked(tmp_path, task_file, monkeypatch, queue)

    second = _workflow(tmp_path)
    calls = _install(monkeypatch, second, Script(None, smith=_smith()))
    with second:
        second.run(task_file)

    assert calls == []
    assert load_state(state_path).verdict == "smith-blocked"


def test_a_tag_naming_a_unit_the_run_does_not_build_is_not_applied(
    tmp_path, task_file, monkeypatch
):
    """Untagged stays core -- the blocking direction -- rather than a guess."""
    _features(task_file, ["mtp"])
    queue = [
        {"n": 1, "mode": "onboard", "target": "norm/core_only"},
        {"n": 2, "mode": "onboard", "target": "moe/drafter_moe"},
    ]
    state_path = _close_smith_blocked(tmp_path, task_file, monkeypatch, queue)
    _tag_vocabulary(tmp_path, {2: "dspark"})

    second = _workflow(tmp_path)
    calls = _install(monkeypatch, second, Script(None, smith=_smith()))
    with second:
        second.run(task_file)

    assert calls == []
    assert "unit" not in _row(load_state(state_path), 2)


# ------------------------------------------------------------------ plan phase


def test_the_plan_phase_rejects_a_unit_the_run_does_not_build(tmp_path, task_file, monkeypatch):
    """Caught as a re-draft, where the drafter can fix it, not at the first failure."""
    _features(task_file, ["mtp"])
    queue = [{"n": 1, "mode": "onboard", "target": "moe/drafter_moe", "unit": "dspark"}]
    workflow = _workflow(tmp_path, max_plan_rounds=0)
    calls = _install(monkeypatch, workflow, Script(None))
    _plan_then_vocabulary(workflow, tmp_path, monkeypatch, queue)
    with workflow:
        workflow.run(task_file)

    assert "smith" not in [role for role, _ in calls]
    assert load_state(tmp_path / "ws" / _STATE).verdict == "plan-failed"
    assert "not one of this run's assembly units" in workflow._vocabulary_errors
