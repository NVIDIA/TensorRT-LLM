# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A LOSS split by member: its classification, the variant's verdict, and the campaign's variant."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune.records import (
    PARTS_FILE,
    Measurement,
    NoiseFloor,
    compare,
)
from agent_flow.workflows.trtllm_modeling_v2.tune.split import SPLIT, classify, judge_variant
from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneSpec, TuneStage

from .test_tune_stage import FakeRunner, _contract, _git, _repo

#: The members of a campaign over parallel modes: their LLM-API config and concurrencies.
MEMBERS = {
    "tp1": ({"tensor_parallel_size": 1}, (1, 16)),
    "tp4": ({"tensor_parallel_size": 4}, (1, 16)),
    "tep4": ({"tensor_parallel_size": 4, "moe_expert_parallel_size": 4}, (16, 128)),
    "dep2": (
        {"tensor_parallel_size": 2, "moe_expert_parallel_size": 2, "enable_attention_dp": True},
        (64, 128),
    ),
    "dep4": (
        {"tensor_parallel_size": 4, "moe_expert_parallel_size": 4, "enable_attention_dp": True},
        (128, 256),
    ),
}
NOISE = NoiseFloor(spread={}, default=0.001)
RULE = {"min_gain": 0.02, "noise_factor": 3.0}


def _measure(root: Path, label: str, gains: dict | None = None, tpot: dict | None = None):
    """A measurement over :data:`MEMBERS`: ``gains[site]`` scales every metric at ``member.conN``."""
    gains, tpot = gains or {}, tpot or {}
    for name, (config, cons) in MEMBERS.items():
        part = root / label / name
        part.mkdir(parents=True, exist_ok=True)
        rows = []
        for con in cons:
            g = gains.get(f"{name}.con{con}", 1.0)
            rows.append({"concurrency": con, "tok_s_user": 100.0 * g, "output_tok_s_gpu": 1000.0 * g,
                         "mean_tpot_ms": tpot.get(f"{name}.con{con}", 10.0)})  # fmt: skip
        (part / "summary.json").write_text(json.dumps(rows))
        meta = {"label": name, "config": {"modeling_v2": "require", **config}, "hostname": "h1"}
        (part / "meta.json").write_text(json.dumps(meta))
    index = {"parts": list(MEMBERS), "constraints": {"dep4": {"tpot_ms_max": 30.0}}}
    (root / label / PARTS_FILE).write_text(json.dumps(index))
    (root / label / "meta.json").write_text(json.dumps({"label": label, "hostname": "h1"}))
    return Measurement.load(root / label)


def _shape(tmp_path: Path, gains: dict, tpot: dict | None = None):
    ctrl = _measure(tmp_path, "ctrl")
    cand = _measure(tmp_path, "cand", gains, tpot)
    assert compare(cand, ctrl, NOISE, **RULE).verdict == "LOSS"
    return classify(cand, ctrl, NOISE, **RULE)


WINS_OUTSIDE_DEP4 = {"tp1.con1": 1.06, "tp4.con16": 1.08, "tep4.con128": 1.03, "dep2.con64": 1.03}


def test_losses_at_one_attention_dp_member_split_by_its_traits(tmp_path):
    shape = _shape(tmp_path, {**WINS_OUTSIDE_DEP4, "dep4.con128": 0.979, "dep4.con256": 0.928})
    assert shape.kind == SPLIT
    assert shape.mode == "attention DP on and EP size 4"
    assert shape.condition == "mapping.enable_attention_dp and mapping.moe_ep_size == 4"
    assert shape.members == ("dep4",) and shape.region == ("dep4.con128", "dep4.con256")
    assert "dep4.con256.output_tok_s_gpu" in shape.losses and "tp1.con1.tok_s_user" in shape.wins


def test_the_variant_wins_only_with_its_losses_gone_and_its_wins_held(tmp_path):
    ctrl = _measure(tmp_path, "ctrl")
    cand = _measure(tmp_path, "cand", {**WINS_OUTSIDE_DEP4, "dep4.con256": 0.93})
    split = classify(cand, ctrl, NOISE, **RULE).as_dict()
    original = compare(cand, ctrl, NOISE, **RULE).deltas

    def verdict(label, gains):
        variant = compare(_measure(tmp_path, label, gains), ctrl, NOISE, **RULE)
        return judge_variant(variant, split, original, NOISE, **RULE)

    win, why = verdict("gated", WINS_OUTSIDE_DEP4)
    assert win and "attention DP on and EP size 4" in why
    win, why = verdict("still-loses", {**WINS_OUTSIDE_DEP4, "dep4.con256": 0.97})
    assert not win and why.startswith("LOSS")
    win, why = verdict("gave-back", {**WINS_OUTSIDE_DEP4, "tp4.con16": 1.01})
    assert not win and "tp4.con16.output_tok_s_gpu +1.0% (the candidate +8.0%)" in why
    # A narrow win that comes in under the threshold still holds its gain within the loss floor.
    win, _ = verdict("held", {**WINS_OUTSIDE_DEP4, "dep2.con64": 1.025, "tep4.con128": 1.025})
    assert win
    win, why = verdict("neutral", {})
    assert not win and why.startswith("NEUTRAL")


# ------------------------------------------------------------------- the campaign


def _effect(config, tree):
    """Con 1 is faster with FAST; con 128 is slower with SLOW_HI unless GATED."""
    text = Path(tree, "modeling.py").read_text() if tree else ""
    lat = 100.0 * (1.2 if "FAST = True" in text else 1.0)
    thr = 2000.0 * (0.9 if "SLOW_HI" in text and "GATED" not in text else 1.0)
    return lat, thr


def _campaign(tmp_path: Path, variant_text: str, **campaign):
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
            "campaign": {"max_parallel_tuners": 1, "kernel_ledger": "off", "max_roadmaps": 1,
                         **campaign},
        },
    }  # fmt: skip
    spec = TuneSpec.from_task(task, tmp_path / "ws", target="t/")
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
        decision = {"decision": "CANDIDATE", "worktree": str(wt), "commits": commits}
        (ws / "candidates" / what / "decision.yaml").write_text(yaml.safe_dump(decision))
        return ""

    log: list[str] = []
    state: dict = {"phase": "campaign", "noise_spread": {}, "noise_samples": 0}
    stage = TuneStage(spec, state, dispatch=dispatch, checkpoint=lambda: None,
                      runner_factory=lambda spools, prefix: FakeRunner(spools, prefix, _effect),
                      poll_s=0, log=log.append)  # fmt: skip
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
