# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A failed attempt is a result: its findings reach later turns; a fixable loss is revised once."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune.campaign import CandidateState, revisable
from agent_flow.workflows.trtllm_modeling_v2.tune.spec import TuneSpec
from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneStage

from .test_tune_stage import FakeRunner, _contract, _effect, _git, _microbench, _repo

FOUND = {"what": "the tiled kernel loses at b1", "evidence": "micro A/B", "next": "keep b1 eager"}


def _slowed(config, tree):
    lat, thr = _effect(config, tree)
    if tree and Path(tree, "modeling.py").read_text().startswith("SLOW"):
        lat /= 1.25
    return lat, thr


def test_a_fixable_loss_is_revised_once_and_every_later_request_carries_what_was_found(tmp_path):
    repo = _repo(tmp_path)
    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump({"modeling_v2": "require", "max_batch_size": 128}))
    task = {
        "trtllm_repo_path": str(repo), "checkpoint_path": "/ckpt", **_contract(tmp_path),
        "perf": {"identity_configs": [str(identity)], "spools": ["s1"], "env_script": "/env.sh",
                 "workload": {"isl": 8192, "osl": 1024, "concurrency": [1, 128]},
                 "sol": {"enabled": False}, "campaign": {"max_parallel_tuners": 1}},
    }  # fmt: skip
    spec = TuneSpec.from_task(
        task, tmp_path / "ws", target="tensorrt_llm/_torch/_experimental/modeling_v2/models/f/t/"
    )
    turns, asks = [], {}

    def dispatch(role, what, prompt):
        turns.append(what)
        if what.startswith("roadmap"):
            asks[what] = (spec.workdir / f"{what}-request.md").read_text()
            items = [{"slug": "tiled", "kind": "kernel", "hypothesis": "tile the GEMM"},
                     {"slug": "fused-norm", "kind": "fusion", "hypothesis": "fuse the norm"}]  # fmt: skip
            (spec.workdir / f"{what}.yaml").write_text(
                yaml.safe_dump(items if what == "roadmap" else [])
            )
            return "roadmap"
        slug = what.removeprefix("revise-")
        out, wt = spec.workdir / "candidates" / slug, spec.workdir / "worktrees" / slug
        if slug == "fused-norm":
            asks[slug] = (out / "request.md").read_text()
            decision = {"decision": "DROPPED", "kind": "fusion", "worktree": str(wt),
                        "findings": [FOUND]}  # fmt: skip
        else:
            asks[what] = (out / ("revise-request.md" if what != slug else "request.md")).read_text()
            (wt / "modeling.py").write_text("FAST = True\n" if what != slug else "SLOW = True\n")
            _git(wt, "commit", "-q", "-am", what)
            decision = {"decision": "CANDIDATE", "kind": "kernel", "worktree": str(wt), "findings": [FOUND],
                        "commits": [_git(wt, "rev-parse", "HEAD")], "microbench": _microbench(out)}  # fmt: skip
        (out / "decision.yaml").write_text(yaml.safe_dump(decision))
        return "done"

    state: dict = {}
    stage = TuneStage(spec, state, dispatch=dispatch, checkpoint=lambda: None, poll_s=0,
                      runner_factory=lambda spools, prefix: FakeRunner(spools, prefix, _slowed),
                      log=lambda m: None)  # fmt: skip
    assert stage.run() == "tuned"
    assert turns == ["roadmap", "tiled", "fused-norm", "revise-tiled", "roadmap-2"]
    assert state["kept"] == ["tiled"]
    # The revise carries the stage's verdict and numbers; the next roadmap, every attempt.
    revise = asks["revise-tiled"]
    assert "Kind: REVISE, once." in revise and "projects con1 -11.11%" in revise
    assert "**`tiled`** (kernel, micro): **being revised**" in revise
    assert "local speedup: decode-b1-bf16-graph -20.0%" in revise and FOUND["next"] in revise
    again = asks["roadmap-2"]
    assert "**`tiled`** (kernel, micro): **kept**" in again
    assert "**`fused-norm`** (fusion, micro): **dropped**" in again
    assert "found: the tiled kernel loses at b1 (micro A/B); next: keep b1 eager" in again
    assert (spec.workdir / "attempts.md").read_text().startswith("# Attempts so far")
    ledger = [
        json.loads(line) for line in (spec.workdir / "perf_ledger.jsonl").read_text().splitlines()
    ]
    (revised,) = [r for r in ledger if r["kind"] == "candidate_revise"]
    assert revised["slug"] == "tiled" and revised["outcome"] == "lost"


def test_only_a_failure_the_code_can_fix_is_revised():
    def st(**kw):
        return CandidateState("c", decision="CANDIDATE", **kw)

    assert revisable(st(status="reverted", verdict="LOSS", reason="micro A/B projects con1 -3%"))
    assert revisable(st(status="blocked", reason="microbench.share: every share is 0"))
    assert not revisable(st(status="reverted", verdict="NEUTRAL"))  # a measured "no gain"
    assert not revisable(st(status="reverted", verdict="LOSS", reason="the checkpoint at x ..."))
    assert not revisable(st(status="blocked", reason="microbench A/B ran on different hosts"))
    assert not revisable(st(status="reverted", verdict="LOSS", scope="e2e"))  # that is a split
    assert not revisable(st(status="reverted", verdict="LOSS", revisions=1))
