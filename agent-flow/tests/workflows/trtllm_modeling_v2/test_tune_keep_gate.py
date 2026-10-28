# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The keep gate: a candidate is kept only when every part of its gate passes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune.campaign import Campaign
from agent_flow.workflows.trtllm_modeling_v2.tune.keep_gate import KEEP_FILE
from agent_flow.workflows.trtllm_modeling_v2.tune.measure import BenchSpec
from agent_flow.workflows.trtllm_modeling_v2.tune.records import NoiseFloor
from agent_flow.workflows.trtllm_modeling_v2.tune.worktree import open_candidate

from .test_tune_stage import FakeRunner, _effect, _git, _keep_gate, _microbench, _repo

UNIFIED = {"parallel": "unified", "members": ["tp1", "tp2", "tep4", "dep4"]}


def _campaign(tmp_path: Path, gates, keep_gate, msgs: list) -> Campaign:
    """One candidate wins its local A/B and goes to the keep gate."""
    repo = _repo(tmp_path)
    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump({"modeling_v2": "require"}))
    bench = BenchSpec(bench_py="b", model="m", env_script="e", isl=8192, osl=1024,
                      concurrencies=[1, 128], tree=str(repo))  # fmt: skip
    work = tmp_path / "camp"
    camp = Campaign(workdir=work, repo=repo, bench=bench, base_configs=[identity],
                    perf_runner=FakeRunner(["s1"], "camp", _effect), gate_runner=gates,
                    keep_gate=keep_gate, noise=NoiseFloor(spread={"con1.tok_s_user": 0.001}),
                    log=msgs.append)  # fmt: skip
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


def _run(camp: Campaign, steps: int = 30) -> None:
    for _ in range(steps):
        if not camp.step():
            break


def test_every_member_passes_or_nothing_is_kept_and_the_verdict_records_why(tmp_path):
    class Gates(FakeRunner):
        rounds: list = []

        def submit(self, spool, label, command, **kw):
            self.rounds.append(label)
            return super().submit(spool, label, command, **kw)

    gates = Gates(["g1", "g2"], "campg", _effect,
                  score=lambda label, tree: 94.0 if label.endswith("-dep4") else 96.5)  # fmt: skip
    gate = replace(_keep_gate(tmp_path, **UNIFIED), tests=("tests/unittest/t",))
    msgs: list = []
    camp = _campaign(tmp_path, gates, gate, msgs)
    _run(camp)
    st = camp.state.candidates["fast-path"]
    assert st.status == "reverted" and camp.state.kept == []
    assert st.reason.startswith("keep gate FAIL: dep4: measured 94.0000 < 95.3286")
    assert [label.rsplit("-", 1)[1] for label in gates.rounds] == ["dep4", "tp1", "tests"]
    assert {gates.where[label] for label in gates.rounds} == {"g1", "g2"}  # side by side
    key = gates.rounds[0].removeprefix("gate-").rsplit("-", 1)[0]
    keep = json.loads((camp.workdir / "gates" / key / KEEP_FILE).read_text())
    assert keep["verdict"] == "FAIL" and keep["scores"] == {"tp1": 96.5, "dep4": 94.0}
    assert keep["anchor"] == 96.36 and keep["floor"] == pytest.approx(95.3286, abs=1e-3)
    assert keep["members"] == ["tp1", "dep4"] and keep["unit"] == "core"
    assert keep["tree"] == st.eval_tree and keep["commit"] == _git(
        Path(st.eval_tree), "rev-parse", "HEAD"
    )
    gated = [json.loads(line) for line in (camp.workdir / "perf_ledger.jsonl").read_text().splitlines()
             if '"candidate_gate"' in line]  # fmt: skip
    assert gated[0]["score"] == 94.0 and gated[0]["anchor"] == 96.36
