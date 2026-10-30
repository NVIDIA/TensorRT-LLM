# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target's own geometry, once its core boots, and the one re-certification batch it asks for."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from agent_flow.workflows.trtllm_modeling_v2.run import geometry
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run import runtime_facts as rf

from .test_geometry_probe import _engine_output, _Runner, _stage
from .test_orchestrator import PlannerAnswer, _orchestrator, _services, _task

_KEYS = ["core:boot@tp1", "core:accuracy[gsm8k]@tp1", "core:envelope@tp1"]


def _settle(stage, members: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        probe = stage.__dict__.get("_reprobe")
        if len(stage._reprobed) == members and probe is not None and probe.done:
            if not stage._reprobe_pending:
                return
        time.sleep(0.02)
    raise AssertionError("the target probes did not finish")


def test_the_targets_geometry_is_probed_once_per_width_in_batches(tmp_path: Path):
    geometry.start_probe(_stage(tmp_path, _Runner())).wait(10)  # the stock path's estimate
    narrow = _engine_output(batches=[1, 2, 4, 8, 16, 32, 64])
    wide = _engine_output(batches=[1, 2, 4, 8, 16, 32, 64, 128])
    release = threading.Event()
    runner = _Runner(answers={"--tp_size=1": narrow, "--tp_size=4": wide})
    stage = _stage(tmp_path, runner, gate=release)
    notes: list[str] = []

    assert geometry.reprobe_target(stage, "tp1", notes.append)
    assert not geometry.reprobe_target(stage, "tp1", notes.append)  # once per width
    assert geometry.reprobe_target(stage, "tep4", notes.append)  # waits for the running probe
    release.set()
    _settle(stage, members=2)

    assert len(notes) == 2, "one event per probe, the second for the width that waited"
    assert "the target's own geometry at tp1" in notes[0]
    assert "cuda_graph_batches [64] (the earlier probe: [1, 2, 4, 8, 16, 32])" in notes[0]
    assert "Re-certify, as one batch" in notes[0]
    # What the first probe already reported is not asked for again.
    assert "the target's own geometry at tep4" in notes[1]
    assert "cuda_graph_batches [128]" in notes[1] and "[64]" not in notes[1].split("(")[0]
    legs = rf.geometry(stage.workspace)["legs"]
    assert legs["core@tp1"]["path"] == "require" and legs["core@tep4"]["path"] == "require"
    assert "path" not in legs["core@dep4"], "the other widths keep the earlier probe's legs"


def test_a_target_that_measures_what_was_estimated_asks_for_nothing(tmp_path: Path):
    before = {"legs": {"core@tp1": {"status": rf.MEASURED, "numbers": {"max_num_tokens": 4096}}}}
    same = {"legs": {"core@tp1": {"status": rf.MEASURED, "numbers": {"max_num_tokens": 4096}}}}
    assert geometry.recertification_note(before, same, ["tp1"], ["catalog/a/b.py"]) == ""
    grown = {"legs": {"core@tp1": {"status": rf.MEASURED, "numbers": {"max_num_tokens": 8192}}}}
    note = geometry.recertification_note(before, grown, ["tp1"], ["catalog/a/b.py"])
    assert "max_num_tokens [8192] (the earlier probe: [4096])" in note
    assert "`catalog/a/b.py`" in note


def test_what_the_stage_learns_outside_a_task_reaches_the_planner(tmp_path: Path):
    turns = []

    def plan(turn):
        turns.append(turn)
        if len(turns) == 1:
            return PlannerAnswer(
                "GRAPH_UPDATED", "plan", g.Graph(tasks=[_task("core", "assemble", _KEYS)])
            )
        return PlannerAnswer("DONE", "green")

    services = _services(plan)
    queued = ["the target's own geometry at tp1 has values no certification was made against"]
    services.stage_notes = lambda: [queued.pop()] if queued else []
    assert _orchestrator(tmp_path, services).run() == "delivered"
    assert any("the target's own geometry" in e for t in turns for e in t.events)
