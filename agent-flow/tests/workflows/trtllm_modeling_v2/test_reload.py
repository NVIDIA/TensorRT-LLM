# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A reload drains the orchestrator, then re-executes it; the next process resumes where it was."""

from __future__ import annotations

import json

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.driver import reload as reload_mod
from agent_flow.workflows.trtllm_modeling_v2.run import drain as drain_mod

from .test_workflow import Script, _install, _workflow


class Replaced(Exception):
    """What the fake re-exec raises: a real one never returns, as the process is gone."""


def test_a_planned_run_reloads_between_turns_and_resumes_where_it_was(
    tmp_path, task_file, monkeypatch
):
    """The planner's turn is answered while a reload waits; the next process runs the task."""
    asked = []

    def planner(turn, prompt):
        from .test_workflow import plan_every_unit

        if turn == 1:
            reload_mod.request(tmp_path / "ws", by="driver")
        return plan_every_unit(turn, prompt)

    def replaced():
        asked.append(json.loads((tmp_path / "ws" / reload_mod.RECORD_FILENAME).read_text()))
        raise Replaced()

    first = _workflow(tmp_path)
    first.reexec = replaced
    calls = _install(monkeypatch, first, Script(None, planner=planner))
    with first, pytest.raises(Replaced):
        first.run(task_file)
    assert [name for name, _ in calls] == ["planner"], "no task starts while it drains"
    assert asked and asked[0]["phase"] == reload_mod.PHASE_REEXEC
    journal = (tmp_path / "ws" / "journal.md").read_text(encoding="utf-8")
    assert "**reload**" in journal

    second = _workflow(tmp_path)
    second.reexec = replaced
    calls = _install(monkeypatch, second, Script(None))
    with second:
        second.run(task_file)
    assert [name for name, _ in calls] == ["assembler"], "the plan is not asked again"
    assert (reload_mod.read_record(tmp_path / "ws") or {}).get("phase") == (
        reload_mod.PHASE_RELOADED
    )
    journal = (tmp_path / "ws" / "journal.md").read_text(encoding="utf-8")
    assert "**restart (reload)**" in journal and "nothing was cut short" in journal
    starts = [
        json.loads(line)
        for line in (tmp_path / "ws" / reload_mod.RESTARTS_FILENAME).read_text().splitlines()
    ]
    assert [s["after"] for s in starts] == [drain_mod.AFTER_FIRST, drain_mod.AFTER_RELOAD]
