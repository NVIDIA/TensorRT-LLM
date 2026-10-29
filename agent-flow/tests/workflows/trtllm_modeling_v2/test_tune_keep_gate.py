# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness in the campaign: keeps wait for a checkpoint; accuracy runs once, at the end."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune.campaign import Campaign
from agent_flow.workflows.trtllm_modeling_v2.tune.measure import BenchSpec
from agent_flow.workflows.trtllm_modeling_v2.tune.records import NoiseFloor
from agent_flow.workflows.trtllm_modeling_v2.tune.worktree import open_candidate

from .test_tune_stage import FakeRunner, _effect, _git, _keep_gate, _microbench, _repo

UNIFIED = {"parallel": "unified", "members": ["tp1", "tp2", "tep4", "dep4"]}


def _fast(tree) -> bool:
    return bool(tree) and Path(tree, "modeling.py").read_text().startswith("FAST = True")


def _campaign(tmp_path: Path, gates, msgs: list, effect=_effect, keep_gate=None, **kw) -> Campaign:
    """One candidate, 20 % faster in its micro-bench at half of each point's step."""
    repo = _repo(tmp_path)
    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump({"modeling_v2": "require"}))
    bench = BenchSpec(bench_py="b", model="m", env_script="e", isl=8192, osl=1024,
                      concurrencies=[1, 128], tree=str(repo))  # fmt: skip
    work = tmp_path / "camp"
    gate = keep_gate or replace(_keep_gate(tmp_path, **UNIFIED), tests=("tests/unittest/t",))
    camp = Campaign(workdir=work, repo=repo, bench=bench, base_configs=[identity],
                    perf_runner=FakeRunner(["s1"], "camp", effect), gate_runner=gates,
                    keep_gate=gate, noise=NoiseFloor(spread={"con1.tok_s_user": 0.001}),
                    log=msgs.append, **kw)  # fmt: skip
    camp.state.best_tree = str(open_candidate(repo, work / "wt", "best", "HEAD"))
    camp.state.best_commit = _git(Path(camp.state.best_tree), "rev-parse", "HEAD")
    wt = open_candidate(repo, work / "wt", "fast-path", "HEAD")
    (wt / "modeling.py").write_text("FAST = True\n")
    _git(wt, "commit", "-q", "-am", "fast")
    decision = tmp_path / "fast-path.yaml"
    decision.write_text(yaml.safe_dump({"decision": "CANDIDATE", "worktree": str(wt),
                                        "microbench": _microbench(tmp_path / "microbench"),
                                        "commits": [_git(wt, "rev-parse", "HEAD")]}))  # fmt: skip
    camp.submit_candidate("fast-path", decision)
    return camp


def _run(camp: Campaign, steps: int = 30) -> list[dict]:
    for _ in range(steps):
        if not camp.step():
            break
    ledger = camp.workdir / "perf_ledger.jsonl"
    return [json.loads(line) for line in ledger.read_text().splitlines() if '"checkpoint"' in line]


def test_a_keep_waits_for_the_checkpoint_and_a_changed_first_token_rolls_it_back(tmp_path):
    flipped = lambda tree: {9: -0.1, 7: -2.5} if _fast(tree) else {7: -0.1, 9: -2.5}  # noqa: E731
    gates = FakeRunner(["g1"], "campg", _effect, first_token=flipped)
    msgs: list = []
    camp = _campaign(tmp_path, gates, msgs)
    good = camp.state.best_commit
    assert camp.keep_gate.members == ("dep4",)  # one member: accuracy runs rarely
    (ck,) = _run(camp)
    st = camp.state.candidates["fast-path"]
    assert st.status == "reverted" and camp.state.kept == [] and camp.state.best_commit == good
    assert (Path(camp.state.best_tree) / "modeling.py").read_text() == "FAST = False\n"
    assert "first tokens agree on 0% of 16 prompts" in st.reason
    assert ck["verdict"] == "LOSS" and ck["e2e"] == "WIN" and ck["correctness"] == "FAIL"
    # No gate per keep: the checkpoint's suite and the first tokens of both trees, no GSM8K.
    assert sorted(label.split("-")[0] for label in gates.submitted) == ["gate", "logits", "logits"]
    assert camp.final_accuracy(lambda: None) is None  # nothing kept, nothing to score


def test_a_held_checkpoint_scales_later_projections_by_what_it_measured(tmp_path):
    def diluted(config, tree):  # served, the kernel's 20 % is 4.3 % of the step
        lat, thr = _effect(config, tree)
        return (lat / 1.15 if config and _fast(tree) else lat), thr

    msgs: list = []
    camp = _campaign(tmp_path, FakeRunner(["g1"], "campg", _effect), msgs, effect=diluted)
    (ck,) = _run(camp)
    assert camp.state.kept == ["fast-path"] and ck["verdict"] != "LOSS"
    assert round(ck["projected"]["con1"], 4) == 0.0909 and round(ck["measured"]["con1"], 3) == 0.043
    assert round(camp.state.calibration, 2) == 0.48 == round(ck["calibration"], 2)
    assert camp.state.checkpoint == {"good": camp.state.best_commit}
    assert any("projections are scaled by 0.48" in m for m in msgs)


def test_a_control_that_cannot_answer_leaves_the_keeps_standing_until_more_arrive(tmp_path):
    mute = lambda tree: {7: -0.1, 9: -2.5} if _fast(tree) else {}  # noqa: E731
    camp = _campaign(tmp_path, FakeRunner(["g1"], "campg", _effect, first_token=mute), [])
    (ck,) = _run(camp)
    assert ck["verdict"] == "ERROR" and camp.state.kept == ["fast-path"]
    assert ck["reason"].startswith("logits-") and "0 of 16 prompts answered" in ck["reason"]
    assert camp.state.checkpoint["retry_after"] == 4  # three more keeps, then again


def test_the_final_tree_is_scored_once_with_the_suite_when_keeps_await_a_checkpoint(tmp_path):
    gates = FakeRunner(["g1"], "campg", _effect, score=lambda label, tree: 94.0)
    camp = _campaign(tmp_path, gates, [], checkpoint_gain=1.0)
    assert _run(camp) == [] and camp.state.kept == ["fast-path"] and gates.submitted == []
    final = camp.final_accuracy(lambda: None)
    assert final["verdict"] == "FAIL" and final["reason"].startswith(
        "dep4: measured 94.0000 < 95.3"
    )
    sha = camp.state.best_commit[:10]
    assert gates.submitted == [f"gate-final-{sha}-dep4", f"gate-final-{sha}-tests"]
