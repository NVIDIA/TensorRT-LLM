# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The decision file: the closed value a turn's verdict reaches the orchestrator as."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.run.execution.handoff import (
    current_decision,
    decision_path,
    mark_decision_dispatch,
    read_decision,
)


def _write(workspace: Path, agent: str, iteration: int, payload: dict) -> Path:
    path = decision_path(workspace, agent, iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


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
