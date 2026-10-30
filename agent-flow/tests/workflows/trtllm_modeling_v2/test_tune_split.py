# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A LOSS split by member: its classification, the variant's verdict, and the campaign's variant."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneStage

from .test_tune_stage import FakeRunner, _git
from .test_tune_wall_budget import _campaign as _spec

# ------------------------------------------------------------------- the campaign


def _effect(config, tree):
    """Con 1 is faster with FAST; con 128 is slower with SLOW_HI unless GATED."""
    text = Path(tree, "modeling.py").read_text() if tree else ""
    lat = 100.0 * (1.2 if "FAST = True" in text else 1.0)
    thr = 2000.0 * (0.9 if "SLOW_HI" in text and "GATED" not in text else 1.0)
    return lat, thr


def _campaign(tmp_path: Path, variant_text: str):
    spec = _spec(tmp_path, max_roadmaps=1)
    turns: list[str] = []

    def dispatch(role, what, prompt):
        turns.append(what)
        ws = spec.workdir
        if what == "sol":
            return ""
        if what == "roadmap":
            (ws / "roadmap.yaml").write_text(yaml.safe_dump([{"slug": "fast-path"}]))
            return ""
        wt = ws / "worktrees" / what
        if what == "fast-path":
            (wt / "modeling.py").write_text("FAST = True\nSLOW_HI = True\n")
        else:
            (wt / "modeling.py").write_text((wt / "modeling.py").read_text() + variant_text)
        _git(wt, "commit", "-q", "-am", what)
        commits = _git(wt, "rev-list", "--reverse", f"{stage.branch_prefix}/best..HEAD").split()
        decision = {
            "decision": "CANDIDATE",
            "worktree": str(wt),
            "commits": commits,
            "perf_scope": "e2e",
        }
        (ws / "candidates" / what / "decision.yaml").write_text(yaml.safe_dump(decision))
        return ""

    log: list[str] = []
    state: dict = {"phase": "campaign"}
    stage = TuneStage(spec, state, dispatch=dispatch, checkpoint=lambda: None,
                      runner_factory=lambda spools, prefix: FakeRunner(spools, prefix, _effect),
                      poll_s=0, log=log.append)  # fmt: skip
    assert stage.ground()
    assert stage.run() == "tuned"
    camp = json.loads((spec.workdir / "campaign_state.json").read_text())["candidates"]
    kinds = [json.loads(line)["kind"] for line in (spec.workdir / "perf_ledger.jsonl").open()]
    return spec, state, turns, camp, kinds, log


def test_a_split_candidate_gets_one_member_gated_variant_that_is_kept(tmp_path):
    spec, state, turns, camp, kinds, log = _campaign(tmp_path, "GATED = True\n")
    assert turns == ["roadmap", "fast-path", "fast-path-gated"]
    assert camp["fast-path"]["status"] == "split"
    assert camp["fast-path"]["split"]["mode"] == (
        "high concurrency (>= 128 requests per rank in a step)"
    )
    assert camp["fast-path-gated"]["variant_of"] == "fast-path"
    assert state["kept"] == ["fast-path-gated"]
    best = Path(state["best_tree"], "modeling.py").read_text()
    assert "FAST = True" in best and "GATED = True" in best
    assert {"candidate_split", "split_variant", "variant_verdict", "candidate_kept"} <= set(kinds)
    assert any("fast-path: split: loses only at con128" in m for m in log), log
    request = (spec.workdir / "candidates" / "fast-path-gated" / "request.md").read_text()
    assert "the requests this rank runs in the step >= 128" in request
    assert str(spec.workdir / "worktrees" / "fast-path-gated") in request
