# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The decision file: the closed value a turn's verdict reaches the orchestrator as."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.run.execution.handoff import (
    current_decision,
    decision_dispatch_matches,
    decision_path,
    mark_decision_dispatch,
    read_decision,
)
from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import ExecutionStage


def _write(workspace: Path, agent: str, iteration: int, payload: dict) -> Path:
    path = decision_path(workspace, agent, iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_a_decision_is_read_with_what_it_says_only_inside_the_enum(tmp_path):
    path = _write(
        tmp_path, "surveyor", 1, {"summary": "wrote both files", "decision": "DRAFT_READY"}
    )
    assert read_decision(path, "surveyor") == {
        "summary": "wrote both files",
        "decision": "DRAFT_READY",
    }
    path = _write(tmp_path, "surveyor", 2, {"summary": "s", "decision": "LOOKS_GOOD_TO_ME"})
    assert read_decision(path, "surveyor") is None
    # The assembler's `blocked_reason` is control flow, not commentary: role extras survive.
    blocked = {
        "summary": "no entry for the gated delta rule",
        "decision": "BLOCKED",
        "blocked_reason": "vocabulary_gap",
        "missing_entry": "gdn_custom_op_inplace",
    }
    assert read_decision(_write(tmp_path, "assembler", 1, blocked), "assembler") == blocked


# ---------------------------------------------- a stale decision is not a decision


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


# ------------------------------------------ named by unit, never overwritten


def test_a_reused_turn_number_gets_a_new_attempt_not_the_old_file(tmp_path):
    first = mark_decision_dispatch(tmp_path, "kernel_writer", 7)
    first.write_text("summary: s\ndecision: FAILED_RECEIPT\n", encoding="utf-8")
    second = mark_decision_dispatch(tmp_path, "kernel_writer", 7)
    third = mark_decision_dispatch(tmp_path, "kernel_writer", 7)
    assert [first.name, second.name, third.name] == [
        "core-kernel_writer-7.yaml",
        "core-kernel_writer-7.2.yaml",
        "core-kernel_writer-7.3.yaml",
    ]
    assert current_decision(tmp_path, "kernel_writer", 7) == third


def test_concurrent_reservations_never_share_a_path(tmp_path):
    got: list[Path] = []
    lock = threading.Lock()

    def reserve():
        path = mark_decision_dispatch(tmp_path, "assembler", 1, unit="core")
        with lock:
            got.append(path)

    threads = [threading.Thread(target=reserve) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(got)) == 16


@pytest.mark.parametrize("payload", [None, "", "{", "[]", '{"task_id": "x"}', "\xff"])
def test_unusable_markers_do_not_establish_dispatch_identity(tmp_path, payload):
    path = decision_path(tmp_path, "kernel_writer", 7)
    marker = path.with_name(f".{path.stem}.dispatched")
    if payload is not None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes(payload.encode("latin-1"))
    assert not decision_dispatch_matches(path, {"task_id": "x", "task_attempt": 3})


def test_a_dispatch_prompt_names_the_reserved_path(tmp_path, task_file, monkeypatch):
    """The role is told the one exact file, including the unit and any attempt suffix."""
    from .test_workflow import Script, _install, _workflow

    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None))
    execution = ExecutionStage(workflow)
    execution._dispatch_task("assembler", "mtp", "mtp", "Assemble it.")
    execution._dispatch_task("assembler", "mtp", "mtp", "Assemble it again.")
    prompts = [prompt for role, prompt in calls if role == "assembler"]
    assert "mtp-assembler-1.yaml`" in prompts[0]
    assert "mtp-assembler-2.yaml`" in prompts[1]
