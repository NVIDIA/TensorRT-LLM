# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``perf.campaign.max_hours``: a campaign past its wall budget issues nothing new and ends."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune.spec import TuneSpec
from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneStage

from .test_tune_stage import FakeRunner, _contract, _git, _microbench, _repo

HOUR = 3600.0


class _Clock:
    """The wall clock the stage reads; a turn may move it on."""

    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def _campaign(tmp_path: Path, **campaign):
    """A one-session campaign that starts at the campaign phase."""
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
            "sol": {"enabled": False},
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
    stage = TuneStage(
        spec,
        state,
        dispatch=dispatch,
        checkpoint=lambda: None,
        runner_factory=lambda spools, prefix: FakeRunner(spools, prefix, _effect),
        poll_s=0,
        log=logs.append,
        clock=clock,
    )
    assert stage.ground()
    return stage


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
        yaml.safe_dump(
            {
                "decision": "CANDIDATE",
                "worktree": str(wt),
                "commits": commits,
                "microbench": _microbench(spec.workdir / "candidates" / what),
            }
        )
    )


def test_a_spent_budget_ends_the_campaign_with_what_it_has_and_says_so(tmp_path):
    spec = _campaign(tmp_path, max_hours=8)
    clock, turns, logs = _Clock(), [], []
    state = {"phase": "campaign",
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

    state = {"phase": "campaign"}
    stage = _stage(spec, state, dispatch, clock, logs)
    assert stage.run() == "tuned"
    assert turns == ["roadmap", "fast-path", "side-path"]
    assert state["kept"] == ["fast-path"] and state["ended"]["by"] == "budget"
    candidates = json.loads((spec.workdir / "campaign_state.json").read_text())["candidates"]
    assert candidates["side-path"]["status"] == "conflict"
    assert any("side-path" in m and "no rebase is asked for" in m for m in logs)
    assert Path(state["best_tree"], "modeling.py").read_text() == "FAST = True\n"
