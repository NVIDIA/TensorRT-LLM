# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``perf.campaign.max_hours``: a campaign past its wall budget issues nothing new and ends."""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune.campaign import CandidateState
from agent_flow.workflows.trtllm_modeling_v2.tune.stage import (
    TuneSpec,
    TuneStage,
    validate_perf_block,
)

from .test_tune_stage import FakeRunner, _contract, _git, _repo

HOUR = 3600.0


@pytest.mark.parametrize("hours", [8, 0.5, None])
def test_a_budget_is_a_positive_number_of_hours_or_none(hours):
    assert validate_perf_block({"perf": {"campaign": {"max_hours": hours}}}) == []


@pytest.mark.parametrize("hours", [0, -1, "8h", True])
def test_anything_else_is_refused(hours):
    errors = validate_perf_block({"perf": {"campaign": {"max_hours": hours}}})
    assert errors == [
        "perf.campaign.max_hours: a positive number of hours, e.g. 8; omit it for no budget"
    ]


class _Clock:
    """The wall clock the stage reads; a turn may move it on."""

    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def _campaign(tmp_path: Path, **campaign):
    """A one-session campaign that starts at the campaign phase, as the rebase test's does."""
    repo = _repo(tmp_path)
    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump({"modeling_v2": "require"}))
    task = {
        "trtllm_repo_path": str(repo),
        "checkpoint_path": "/ckpt",
        **_contract(tmp_path),
        "perf": {
            "identity_configs": [str(identity)],
            "workload": {"isl": 8192, "osl": 1024, "concurrency": [1, 128]},
            "spools": ["s1"],
            "env_script": "/env.sh",
            "reference": "none",
            "campaign": {"max_parallel_tuners": 1, "kernel_ledger": "off", **campaign},
        },
    }
    return TuneSpec.from_task(task, tmp_path / "ws", target="t/")


def _effect(config, tree):
    text = Path(tree, "modeling.py").read_text() if tree else ""
    fast = 1.2 if "FAST = True" in text else 1.0
    side = 1.1 if "SIDE" in text else 1.0
    return 100.0 * fast * side, 2000.0


def _stage(spec, state, dispatch, clock, logs):
    return TuneStage(
        spec,
        state,
        dispatch=dispatch,
        checkpoint=lambda: None,
        runner_factory=lambda spools, prefix: FakeRunner(spools, prefix, _effect),
        poll_s=0,
        log=logs.append,
        clock=clock,
    )


def _ledger(spec) -> list[dict]:
    path = spec.workdir / "perf_ledger.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _candidate(spec, stage, what: str, text: str) -> None:
    """What a tuner's turn leaves: a commit in its worktree and a CANDIDATE decision."""
    wt = spec.workdir / "worktrees" / what
    (wt / "modeling.py").write_text(text)
    _git(wt, "commit", "-q", "-am", what)
    commits = _git(wt, "rev-list", "--reverse", f"{stage.branch_prefix}/best..HEAD").split()
    (spec.workdir / "candidates" / what / "decision.yaml").write_text(
        yaml.safe_dump({"decision": "CANDIDATE", "worktree": str(wt), "commits": commits})
    )


def test_a_spent_budget_ends_the_campaign_with_what_it_has_and_says_so(tmp_path):
    spec = _campaign(tmp_path, max_hours=8)
    clock, turns, logs = _Clock(), [], []
    state = {"phase": "campaign", "noise_spread": {}, "noise_samples": 0,
             "started_at": clock.now - 9 * HOUR}  # fmt: skip
    stage = _stage(spec, state, lambda role, what, prompt: turns.append(what) or "", clock, logs)
    assert stage.run() == "tuned"
    assert turns == []  # no roadmap, so no candidate
    assert state["ended"]["by"] == "budget" and state["kept"] == []
    assert state["budget_spent"]["after_h"] == 9.0
    kinds = [r["kind"] for r in _ledger(spec)]
    assert "budget_spent" in kinds and "campaign_end" in kinds
    ended = (
        "the wall budget of 8 h (`perf.campaign.max_hours`) ended the campaign, spent after 9.0 h"
    )
    assert stage._end_text() == ended
    assert f"Ended: {ended}." in (spec.workdir / "summary.md").read_text()
    assert any("the wall budget of 8 h (perf.campaign.max_hours) is spent" in m for m in logs)

    # Entered again after it ended (an explicit `tune`), the campaign has a budget of its own.
    state["phase"] = "campaign"
    again = _stage(spec, state, lambda role, what, prompt: turns.append(what) or "", clock, logs)
    assert again.run() == "tuned"
    assert turns == ["roadmap", "roadmap-2"] and state["ended"]["by"] == "roadmaps"
    assert state["started_at"] == clock.now and "budget_spent" not in state


def test_what_is_in_flight_finishes_and_nothing_new_is_issued(tmp_path):
    """A candidate issued before the budget ran out is measured and kept; nothing follows it.

    side-path's turn runs past the budget while fast-path is measured: fast-path is still
    judged and kept, side-path (issued before) is taken up and conflicts with that keep, and
    no rebase is asked for it; third-path is never issued, nor a second roadmap.
    """
    spec = _campaign(tmp_path, max_hours=8)
    clock, turns, logs = _Clock(), [], []

    def dispatch(role, what, prompt):
        turns.append(what)
        if what == "roadmap":
            items = [{"slug": "fast-path"}, {"slug": "side-path"}, {"slug": "third-path"}]
            (spec.workdir / "roadmap.yaml").write_text(yaml.safe_dump(items))
        elif what == "fast-path":
            _candidate(spec, stage, what, "FAST = True\n")
        elif what == "side-path":
            _candidate(spec, stage, what, "FAST = False\nSIDE = 1\n")
            clock.now += 9 * HOUR  # this turn runs past the budget
        return ""

    state = {"phase": "campaign", "noise_spread": {}, "noise_samples": 0}
    stage = _stage(spec, state, dispatch, clock, logs)
    assert stage.run() == "tuned"
    assert turns == ["roadmap", "fast-path", "side-path"]
    assert state["kept"] == ["fast-path"] and state["ended"]["by"] == "budget"
    candidates = json.loads((spec.workdir / "campaign_state.json").read_text())["candidates"]
    assert candidates["side-path"]["status"] == "conflict"
    assert any("side-path" in m and "no rebase is asked for" in m for m in logs)
    assert Path(state["best_tree"], "modeling.py").read_text() == "FAST = True\n"


def test_a_turn_answered_before_a_restart_is_still_taken_up_past_the_budget(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.tune.worktree import open_candidate

    spec = _campaign(tmp_path, max_hours=8)
    clock, turns = _Clock(), []
    state = {"phase": "campaign", "noise_spread": {}, "noise_samples": 0,
             "started_at": clock.now - 9 * HOUR}  # fmt: skip
    stage = _stage(spec, state, lambda role, what, prompt: turns.append(what) or "", clock, [])
    items = [{"slug": "fast-path"}, {"slug": "side-path"}]
    (spec.workdir / "roadmap.yaml").write_text(yaml.safe_dump(items))
    # fast-path's turn answered while no orchestrator watched; side-path's never did.
    wt = open_candidate(
        spec.repo, spec.workdir / "worktrees", "fast-path", "HEAD", prefix=stage.branch_prefix
    )
    (wt / "modeling.py").write_text("FAST = True\n")
    _git(wt, "commit", "-q", "-am", "fast-path")
    decision = spec.workdir / "candidates" / "fast-path" / "decision.yaml"
    decision.parent.mkdir(parents=True, exist_ok=True)
    decision.write_text(
        yaml.safe_dump(
            {
                "decision": "CANDIDATE",
                "worktree": str(wt),
                "commits": [_git(wt, "rev-parse", "HEAD")],
            }
        )
    )
    assert stage.run() == "tuned"
    assert turns == []  # no roadmap turn and no candidate turn
    assert state["kept"] == ["fast-path"] and state["ended"]["by"] == "budget"


def test_without_a_budget_the_campaign_runs_its_roadmaps_out(tmp_path):
    spec = _campaign(tmp_path, max_hours=100)
    clock, turns = _Clock(), []

    def dispatch(role, what, prompt):
        turns.append(what)
        if what == "roadmap":
            (spec.workdir / "roadmap.yaml").write_text(yaml.safe_dump([{"slug": "fast-path"}]))
        elif what == "fast-path":
            _candidate(spec, stage, what, "FAST = True\n")
            clock.now += 9 * HOUR  # well inside 100 h
        return ""

    state = {"phase": "campaign", "noise_spread": {}, "noise_samples": 0}
    stage = _stage(spec, state, dispatch, clock, [])
    assert stage.run() == "tuned"
    assert turns == ["roadmap", "fast-path", "roadmap-2"]
    assert state["kept"] == ["fast-path"] and state["ended"]["by"] == "roadmaps"
    assert "budget_spent" not in state and stage._end_text() == ""
    assert "Ended:" not in (spec.workdir / "summary.md").read_text()


def test_a_split_past_the_budget_asks_for_no_variant(tmp_path):
    spec = _campaign(tmp_path, max_hours=8)
    stage = _stage(spec, {"phase": "campaign"}, lambda *a: "", _Clock(), [])
    closed: list[tuple[str, str]] = []
    cut = CandidateState("b-gated", variant_of="b")  # its turn was cut off by a restart
    camp = SimpleNamespace(
        state=SimpleNamespace(
            candidates={
                "a": CandidateState("a", status="split", split={"mode": "dp"}),
                "b": CandidateState(
                    "b", status="split", split={"mode": "tp", "variant": "b-gated"}
                ),
                "b-gated": cut,
            }
        ),
        close_split=lambda slug, why: closed.append((slug, why)),
    )
    futures: dict = {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        stage._split_turns(camp, pool, futures, budget=12, spent=True)
    assert futures == {}
    assert closed == [("a", "the wall budget is spent"), ("b", "the wall budget is spent")]
    assert cut.status == "dropped" and "the wall budget is spent" in cut.reason


def test_the_task_reports_that_the_budget_ended_the_campaign(tmp_path, monkeypatch):
    import agent_flow.workflows.trtllm_modeling_v2.tune.stage as stage_mod

    spec = _campaign(tmp_path, max_hours=2)
    repo = Path(spec.repo)
    base = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    clock = _Clock()

    def stage_for(wf, state, base_commit="", attempt=1, should_stop=lambda: False):
        spec.base_commit = base_commit
        return _stage(spec, state.tune, lambda *a: "", clock, [])

    monkeypatch.setattr(stage_mod, "_stage_for", stage_for)
    state = SimpleNamespace(
        tune={
            "phase": "campaign",
            "noise_spread": {},
            "noise_samples": 0,
            "started_at": clock.now - 3 * HOUR,
        }  # fmt: skip
    )
    decision, line = stage_mod.run_tune_task(None, state, base)
    assert decision == "TUNED"
    assert "the wall budget of 2 h (`perf.campaign.max_hours`) ended the campaign" in line
    assert line.index("goal not reached") < line.index("wall budget") < line.index("summary")
