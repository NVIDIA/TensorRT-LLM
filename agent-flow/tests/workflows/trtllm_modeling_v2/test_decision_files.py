# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the file-based decision protocol.

This mode exists because an enterprise MCP configuration
(``/etc/claude-code/managed-mcp.json``) makes the Agent SDK refuse every
dynamically-registered server — including the in-process one the typed
decision tools live in. On such a host those tools cannot be created at all,
and a run that needs them dies on its first dispatch with an opaque
``ProcessError: exit code 1``.

The guarantee the tools provide is a **closed value**: the orchestrator
branches on ``decision`` and never on prose. In this mode validation on read
enforces it instead of the tool schema, and it has to be just as strict there
or the mode is a downgrade pretending not to be one.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import build_trtllm_modeling_v2_prompts
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts._common import decision_file_block
from agent_flow.workflows.trtllm_modeling_v2.run import progress as progress_mod
from agent_flow.workflows.trtllm_modeling_v2.run.progress import (
    DECISIONS_BY_AGENT,
    EXTRA_KEYS_BY_AGENT,
    ProgressContext,
    collect_decision_file,
    current_decision,
    decision_is_stale,
    decision_path,
    init_progress_file,
    latest_decision,
    latest_entry,
    mark_decision_dispatch,
    reserve_decision,
)


def _context(tmp_path: Path) -> ProgressContext:
    progress = tmp_path / "progress.yaml"
    init_progress_file(progress)
    return ProgressContext(path=progress)


def _write(tmp_path: Path, agent: str, iteration: int, payload: dict) -> None:
    path = decision_path(tmp_path, agent, iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_a_valid_decision_is_recorded_and_readable_downstream(tmp_path):
    """The whole point: `latest_decision` callers read it like any other entry."""
    ctx = _context(tmp_path)
    _write(tmp_path, "plan_drafter", 1, {"summary": "wrote both files", "decision": "DRAFT_READY"})

    assert collect_decision_file(ctx, tmp_path, "plan_drafter", 1) == "DRAFT_READY"
    assert (
        latest_decision(ctx.path, "plan_drafter", DECISIONS_BY_AGENT["plan_drafter"])
        == "DRAFT_READY"
    )
    assert latest_entry(ctx.path, "plan_drafter")["summary"] == "wrote both files"


def test_a_decision_outside_the_enum_is_refused(tmp_path):
    """The tool schema refuses it; so must the reader.

    Accepting it would turn the orchestrator's one closed input into
    whatever a role felt like writing — which is the failure the typed
    tools exist to prevent.
    """
    ctx = _context(tmp_path)
    _write(tmp_path, "plan_drafter", 1, {"summary": "s", "decision": "LOOKS_GOOD_TO_ME"})
    assert collect_decision_file(ctx, tmp_path, "plan_drafter", 1) is None
    assert latest_entry(ctx.path, "plan_drafter") is None


def test_another_roles_decision_is_not_accepted(tmp_path):
    """`APPROVE` is real — for the reviewer. It is not a drafter decision."""
    ctx = _context(tmp_path)
    _write(tmp_path, "plan_drafter", 1, {"summary": "s", "decision": "APPROVE"})
    assert collect_decision_file(ctx, tmp_path, "plan_drafter", 1) is None


def test_a_missing_file_is_none_rather_than_an_exception(tmp_path):
    """A role that ended without deciding is a case the caller handles."""
    ctx = _context(tmp_path)
    assert collect_decision_file(ctx, tmp_path, "smith", 3) is None


def test_malformed_yaml_is_refused_without_raising(tmp_path):
    ctx = _context(tmp_path)
    path = decision_path(tmp_path, "smith", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("decision: [unclosed\n", encoding="utf-8")
    assert collect_decision_file(ctx, tmp_path, "smith", 1) is None


def test_role_specific_extra_keys_survive(tmp_path):
    """The assembler's `blocked_reason` is control flow, not commentary.

    `_handle_blocked_assembler` branches on it — a BLOCKED entry that lost
    it is a run that cannot triage itself.
    """
    ctx = _context(tmp_path)
    _write(
        tmp_path,
        "assembler",
        1,
        {
            "summary": "no entry for the gated delta rule",
            "decision": "BLOCKED",
            "blocked_reason": "vocabulary_gap",
            "missing_entry": "gdn_custom_op_inplace",
        },
    )
    assert collect_decision_file(ctx, tmp_path, "assembler", 1) == "BLOCKED"
    entry = latest_entry(ctx.path, "assembler")
    assert entry["blocked_reason"] == "vocabulary_gap"
    assert entry["missing_entry"] == "gdn_custom_op_inplace"


def test_empty_extra_keys_are_dropped_rather_than_stored_blank(tmp_path):
    ctx = _context(tmp_path)
    _write(tmp_path, "smith", 1, {"summary": "s", "decision": "LANDED", "entry": "   "})
    collect_decision_file(ctx, tmp_path, "smith", 1)
    assert "entry" not in latest_entry(ctx.path, "smith")


def test_the_iteration_is_the_orchestrators_not_the_agents(tmp_path):
    """A role could write any iteration into the file; it is not asked to.

    Stamping it here keeps a mistyped or hallucinated number from
    reordering the record.
    """
    ctx = _context(tmp_path)
    _write(tmp_path, "smith", 7, {"summary": "s", "decision": "LANDED", "iteration": 99})
    collect_decision_file(ctx, tmp_path, "smith", 7)
    assert latest_entry(ctx.path, "smith")["iteration"] == 7


# ---------------------------------------------- a stale decision is not a decision


def _write_decision(workspace: Path, agent: str, iteration: int, decision: str) -> Path:
    target = decision_path(workspace, agent, iteration)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"summary: s\ndecision: {decision}\n", encoding="utf-8")
    return target


def test_a_decision_written_after_the_marker_is_collected(tmp_path):
    """The ordinary case: a turn answered its own dispatch."""
    ctx = _context(tmp_path)
    mark_decision_dispatch(tmp_path, "reviewer", 1)
    _write_decision(tmp_path, "reviewer", 1, "APPROVE")
    assert collect_decision_file(ctx, tmp_path, "reviewer", 1) == "APPROVE"


def test_a_decision_older_than_the_marker_is_refused(tmp_path):
    """The failure the marker exists for, on a path that was not reserved.

    A workspace can hold a marker and a decision at the same path, in the
    wrong order. The decision predates the dispatch that wants it, so it
    belongs to a previous turn and is refused.
    """
    ctx = _context(tmp_path)
    stale = _write_decision(tmp_path, "reviewer", 1, "REJECT")
    os.utime(stale, (1_000_000, 1_000_000))  # written long before this dispatch
    marker = stale.with_name(f".{stale.stem}.dispatched")
    marker.write_text("", encoding="utf-8")

    assert decision_is_stale(tmp_path, "reviewer", 1)
    assert collect_decision_file(ctx, tmp_path, "reviewer", 1) is None
    # and nothing was appended, so `latest_decision` cannot read it either
    assert latest_decision(ctx.path, "reviewer", DECISIONS_BY_AGENT["reviewer"]) is None


def test_a_previous_turns_decision_is_never_read_as_this_ones(tmp_path):
    """Turns that share one path read back verdicts from earlier passes.

    A new dispatch is reserved a new path, so the earlier verdict is neither
    read as this turn's nor overwritten by it.
    """
    ctx = _context(tmp_path)
    old = _write_decision(tmp_path, "reviewer", 1, "REJECT")
    target = mark_decision_dispatch(tmp_path, "reviewer", 1)

    assert target != old
    assert collect_decision_file(ctx, tmp_path, "reviewer", 1) is None
    target.write_text("summary: s\ndecision: APPROVE\n", encoding="utf-8")
    assert collect_decision_file(ctx, tmp_path, "reviewer", 1) == "APPROVE"
    assert old.read_text() == "summary: s\ndecision: REJECT\n"


def test_without_a_marker_no_decision_is_refused(tmp_path):
    """A run that never stamped one must not have its decisions refused.

    Absence means "this run does not use the marker", not "stale" — the same
    three-state rule the rest of the workflow applies to receipts.
    """
    ctx = _context(tmp_path)
    _write_decision(tmp_path, "reviewer", 2, "APPROVE")
    assert not decision_is_stale(tmp_path, "reviewer", 2)
    assert collect_decision_file(ctx, tmp_path, "reviewer", 2) == "APPROVE"


def test_a_missing_decision_is_not_stale(tmp_path):
    """A stamped turn that was never answered is the caller's "no decision"."""
    mark_decision_dispatch(tmp_path, "assembler", 3)
    assert not decision_is_stale(tmp_path, "assembler", 3)


def test_the_marker_does_not_collide_with_the_decision_it_guards(tmp_path):
    """A dotfile beside it, so a marker is never mistaken for an answer."""
    mark_decision_dispatch(tmp_path, "smith", 4)
    written = sorted(p.name for p in decision_path(tmp_path, "smith", 4).parent.iterdir())
    assert written == [".core-smith-4.dispatched"]


def test_staleness_is_judged_by_mtime_not_by_the_clock():
    """Both files live under the workspace, so one filesystem orders them.

    An agent may answer from a node whose clock is skewed by minutes; a
    wall-clock comparison would reject its perfectly good decision.
    """
    assert "_predates_dispatch" in inspect.getsource(decision_is_stale)
    src = inspect.getsource(progress_mod._predates_dispatch)
    assert "st_mtime" in src
    assert "time.time" not in src and "datetime.now" not in src


# ------------------------------------------ named by unit, never overwritten


def test_decision_files_are_named_by_unit_role_and_turn(tmp_path):
    assert decision_path(tmp_path, "assembler", 2, unit="multimodal").name == (
        "multimodal-assembler-2.yaml"
    )
    assert decision_path(tmp_path, "reviewer", 1, unit="core").name == "core-reviewer-1.yaml"
    assert decision_path(tmp_path, "smith", 7, unit="dspark").name == "dspark-smith-7.yaml"
    # Plan is not per unit; it files under its stage.
    assert decision_path(tmp_path, "plan_drafter", 1, unit="mtp").name == (
        "plan-plan_drafter-1.yaml"
    )


def test_two_units_turn_one_do_not_share_a_file(tmp_path):
    """Turn 1 of the multimodal assembler must not overwrite core's turn 1."""
    core = mark_decision_dispatch(tmp_path, "assembler", 1, unit="core")
    core.write_text("summary: core\ndecision: UNIT_READY\n", encoding="utf-8")
    multimodal = mark_decision_dispatch(tmp_path, "assembler", 1, unit="multimodal")
    multimodal.write_text("summary: mm\ndecision: BLOCKED\n", encoding="utf-8")
    assert core != multimodal
    assert "core" in core.read_text()


def test_a_reused_turn_number_gets_a_new_attempt_not_the_old_file(tmp_path):
    """A re-entered unit restarts its turn count; a retried smith keeps its number."""
    first = mark_decision_dispatch(tmp_path, "smith", 7)
    first.write_text("summary: s\ndecision: FAILED_RECEIPT\n", encoding="utf-8")
    second = mark_decision_dispatch(tmp_path, "smith", 7)
    third = mark_decision_dispatch(tmp_path, "smith", 7)
    assert [first.name, second.name, third.name] == [
        "core-smith-7.yaml",
        "core-smith-7.2.yaml",
        "core-smith-7.3.yaml",
    ]
    assert current_decision(tmp_path, "smith", 7) == third


def test_concurrent_reservations_never_share_a_path(tmp_path):
    import threading

    got: list[Path] = []
    lock = threading.Lock()

    def reserve():
        path = reserve_decision(tmp_path, "reviewer", 1, unit="core")
        with lock:
            got.append(path)

    threads = [threading.Thread(target=reserve) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(got)) == 16


def test_the_collected_entry_names_its_file(tmp_path):
    ctx = _context(tmp_path)
    path = mark_decision_dispatch(tmp_path, "reviewer", 2, unit="mtp")
    path.write_text("summary: s\ndecision: APPROVE\n", encoding="utf-8")
    assert collect_decision_file(ctx, tmp_path, "reviewer", 2, unit="mtp") == "APPROVE"
    assert latest_entry(ctx.path, "reviewer")["decision_file"] == "mtp-reviewer-2.yaml"


def test_a_dispatch_prompt_names_the_reserved_path(tmp_path, task_file, monkeypatch):
    """The role is told the one exact file, including the unit and any attempt suffix."""
    from .test_workflow import Script, _install, _workflow

    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None))
    workflow._turns_by_key = {}
    workflow._dispatch_task("reviewer", "mtp", "mtp", "Review it.")
    workflow._dispatch_task("reviewer", "mtp", "mtp", "Review it again.")
    prompts = [prompt for role, prompt in calls if role == "reviewer"]
    assert "mtp-reviewer-1.yaml`" in prompts[0]
    assert "mtp-reviewer-2.yaml`" in prompts[1]


# ------------------------------------------------------------------ prompts


def test_each_role_is_shown_only_its_own_options():
    """A role shown another role's enum will eventually pick from it."""
    block = decision_file_block("assembler", DECISIONS_BY_AGENT["assembler"])
    assert "UNIT_READY" in block and "BLOCKED" in block
    for foreign in ("DRAFT_READY", "CEILING", "LANDED", "FAILED_RECEIPT"):
        assert foreign not in block, foreign


def test_every_deciding_role_gets_its_decision_block():
    """The tuner is the exception: it writes the file its request names."""
    bundle = build_trtllm_modeling_v2_prompts()
    for role, decisions in DECISIONS_BY_AGENT.items():
        if role == "tuner":
            assert "decisions/<unit>-tuner-<turn>.yaml" not in bundle.tuner
            continue
        prompt = getattr(bundle, role)
        assert "## The decision file" in prompt, role
        assert f"decisions/<unit>-{role}-<turn>.yaml" in prompt, role
        for value in decisions:
            assert f"`{value}`" in prompt, (role, value)


def test_roles_with_extra_keys_are_told_about_them():
    bundle = build_trtllm_modeling_v2_prompts()
    for role, keys in EXTRA_KEYS_BY_AGENT.items():
        for key in keys:
            assert key in getattr(bundle, role), (role, key)


# ------------------------------------------------- the workspace spec copy


def test_a_resumed_run_refreshes_the_spec_the_agents_read(tmp_path):
    """Two copies of task.yaml must not drift apart.

    `_init_state` re-reads and resolves the operator's file every run, so the
    workspace copy — which every role is told to read as "the run's spec" —
    is rewritten to match. Otherwise a resumed run gives the gates one spec
    and the assembler another: both self-consistent, neither correct, and
    nothing would say so.
    """
    import yaml as _yaml

    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    workspace = tmp_path / "ws"
    workspace.mkdir()
    stale = {"features": [], "environment": {"env": {}}}
    (workspace / "task.yaml").write_text(_yaml.safe_dump(stale), encoding="utf-8")

    source = tmp_path / "task.yaml"
    source.write_text("", encoding="utf-8")

    workflow = TrtllmModelingV2Workflow(workspace)
    workflow.task_data = {"features": ["mtp"], "environment": {"env": {"PATH": "/venv/bin"}}}
    workflow._refresh_workspace_spec(source, None)

    written = _yaml.safe_load((workspace / "task.yaml").read_text(encoding="utf-8"))
    assert written["features"] == ["mtp"]
    assert written["environment"]["env"]["PATH"] == "/venv/bin"


def test_refreshing_is_a_no_op_when_the_spec_already_matches(tmp_path):
    """Rewriting an identical file would churn its mtime for no reason."""
    import yaml as _yaml

    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    workspace = tmp_path / "ws"
    workspace.mkdir()
    data = {"features": ["mtp"]}
    (workspace / "task.yaml").write_text(
        _yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    before = (workspace / "task.yaml").stat().st_mtime_ns

    source = tmp_path / "task.yaml"
    source.write_text("", encoding="utf-8")

    workflow = TrtllmModelingV2Workflow(workspace)
    workflow.task_data = data
    workflow._refresh_workspace_spec(source, None)

    assert (workspace / "task.yaml").stat().st_mtime_ns == before
