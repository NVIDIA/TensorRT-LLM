# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A campaign gives way to full support: told to stop, it measures nothing new and wraps up."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune import stage as stage_mod

from .test_tune_wall_budget import _campaign, _candidate, _Clock, _ledger, _stage


def _stopping(spec, state, dispatch, clock, logs, stop):
    stage = _stage(spec, state, dispatch, clock, logs)
    stage.should_stop = lambda: stop["now"]
    return stage


def test_told_to_stop_before_it_measured_anything_it_ends_at_once(tmp_path):
    spec = _campaign(tmp_path)
    clock, turns, logs = _Clock(), [], []
    state = {"phase": "campaign", "noise_spread": {}, "noise_samples": 0}
    stop = {"now": True}
    stage = _stopping(
        spec, state, lambda role, what, prompt: turns.append(what) or "", clock, logs, stop
    )
    assert stage.run() == "tuned"
    assert turns == []  # no roadmap, no candidate
    assert state["ended"]["by"] == "support-complete" and "stopped" in state
    assert any("nothing was kept: no final measurement" in m for m in logs)
    assert not (stage.data / "t-final").exists()
    assert "stopped" in [r["kind"] for r in _ledger(spec)]
    summary = (spec.workdir / "summary.md").read_text()
    assert "full support held on the target, so the campaign wrapped up" in summary


def test_what_is_measuring_finishes_and_a_turn_answered_after_the_stop_is_dropped(tmp_path):
    """What measures when full support holds is kept; a turn answering after it is dropped.

    fast-path is measuring when full support holds: it is judged and kept. side-path's turn
    answers after the stop: nothing measures it, and no further roadmap is asked for.
    """
    spec = _campaign(tmp_path)
    clock, turns, logs = _Clock(), [], []
    stop = {"now": False}

    def dispatch(role, what, prompt):
        turns.append(what)
        if what == "roadmap":
            items = [{"slug": "fast-path"}, {"slug": "side-path"}, {"slug": "third-path"}]
            (spec.workdir / "roadmap.yaml").write_text(yaml.safe_dump(items))
        elif what == "fast-path":
            _candidate(spec, stage, what, "FAST = True\n")
        elif what == "side-path":
            _candidate(spec, stage, what, "FAST = False\nSIDE = 1\n")
            stop["now"] = True  # full support holds while this turn runs
        return ""

    state = {"phase": "campaign", "noise_spread": {}, "noise_samples": 0}
    stage = _stopping(spec, state, dispatch, clock, logs, stop)
    assert stage.run() == "tuned"
    assert turns == ["roadmap", "fast-path", "side-path"]
    assert state["kept"] == ["fast-path"] and state["ended"]["by"] == "support-complete"
    candidates = json.loads((spec.workdir / "campaign_state.json").read_text())["candidates"]
    assert "side-path" not in candidates or candidates["side-path"]["status"] == "issued"
    assert any("side-path: answered after the run was fully supported" in m for m in logs)
    # Something was kept: the best tree is measured once more.
    assert (stage.data / "t-final" / "meta.json").exists()


def test_the_task_reports_that_full_support_ended_the_campaign(tmp_path, monkeypatch):
    spec = _campaign(tmp_path)
    base = subprocess.run(
        ["git", "-C", str(spec.repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    clock = _Clock()
    given = []

    def stage_for(wf, state, base_commit="", attempt=1, should_stop=lambda: False):
        given.append(should_stop)
        spec.base_commit = base_commit
        stage = _stage(spec, state.tune, lambda *a: "", clock, [])
        stage.should_stop = should_stop
        return stage

    monkeypatch.setattr(stage_mod, "_stage_for", stage_for)
    state = SimpleNamespace(tune={"phase": "campaign", "noise_spread": {}, "noise_samples": 0})
    decision, line = stage_mod.run_tune_task(None, state, base, should_stop=lambda: True)
    assert decision == "TUNED" and given[0]()
    assert "full support held on the target, so the campaign wrapped up" in line
    assert Path(line.split("summary `")[1].rstrip("`")).is_file()
