# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A LOSS split by member: its classification, the variant's request, and the variant's verdict."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune.records import (
    PARTS_FILE,
    Measurement,
    NoiseFloor,
    compare,
)
from agent_flow.workflows.trtllm_modeling_v2.tune.requests import (
    CampaignContext,
    render_split_request,
)
from agent_flow.workflows.trtllm_modeling_v2.tune.split import (
    MIXED,
    SPLIT,
    WHOLE,
    classify,
    judge_variant,
)
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
    """ssm-decode-fused: loses at dep4 only; dep2 (attention DP) and tep4 (EP 4) win."""
    shape = _shape(tmp_path, {**WINS_OUTSIDE_DEP4, "dep4.con128": 0.979, "dep4.con256": 0.928})
    assert shape.kind == SPLIT
    assert shape.mode == "attention DP on and EP size 4"
    assert shape.condition == "mapping.enable_attention_dp and mapping.moe_ep_size == 4"
    assert shape.members == ("dep4",) and shape.region == ("dep4.con128", "dep4.con256")
    assert "dep4.con256.output_tok_s_gpu" in shape.losses and "tp1.con1.tok_s_user" in shape.wins


def test_a_loss_at_high_concurrency_splits_by_the_requests_per_rank(tmp_path):
    """moe-finalize-latent-norm: dep4 wins at 128 and loses at 256; dep2 wins at 128 (64 per rank)."""
    shape = _shape(tmp_path, {"dep4.con128": 1.04, "dep4.con256": 0.98, "dep2.con128": 1.02})
    assert shape.kind == SPLIT and shape.lost == ("dep4.con256",)
    assert shape.mode == "EP size 4 and high concurrency (>= 64 requests per rank in a step)"
    assert "the requests this rank runs in the step >= 64" in shape.condition
    assert "dep2.con128" not in shape.region and "dep4.con128" not in shape.region


def test_one_trait_is_preferred_and_the_narrowest_wins_a_tie(tmp_path):
    """moe-permute-fused: loses at tp1 con1, wins at tp4 and dep4.

    TP size 1 and one request per rank cover as many sites; the parallel trait comes first.
    """
    shape = _shape(
        tmp_path, {"tp1.con1": 0.99, "tp4.con16": 1.04, "dep4.con128": 1.056, "dep4.con256": 1.058}
    )
    assert shape.kind == SPLIT and shape.terms == (("tp_size", "==", 1),)
    assert shape.region == ("tp1.con1", "tp1.con16")


def test_a_loss_with_no_win_is_a_plain_loss(tmp_path):
    shape = _shape(tmp_path, {"dep4.con256": 0.9})
    assert shape.kind == WHOLE and shape.why == "it wins nowhere"


def test_losses_and_wins_that_share_every_mode_are_mixed(tmp_path):
    """tp1 and dep4 lose, tp4 at one request per rank wins.

    No trait, nor two, holds both losses and no win.
    """
    shape = _shape(tmp_path, {"tp1.con1": 0.95, "dep4.con256": 0.95, "tp4.con1": 1.05})
    assert shape.kind == MIXED and "no parallel mode" in shape.why


def test_a_site_that_both_wins_and_loses_is_mixed(tmp_path):
    ctrl = _measure(tmp_path, "ctrl")
    _measure(tmp_path, "cand", {"tp1.con1": 1.05})
    rows = json.loads((tmp_path / "cand" / "dep4" / "summary.json").read_text())
    rows[1]["output_tok_s_gpu"] *= 1.05
    rows[1]["tok_s_user"] *= 0.95
    (tmp_path / "cand" / "dep4" / "summary.json").write_text(json.dumps(rows))
    shape = classify(Measurement.load(tmp_path / "cand"), ctrl, NOISE, **RULE)
    assert shape.kind == MIXED and shape.why == "dep4.con256 both won and lost"


def test_a_broken_constraint_is_a_loss_at_its_site(tmp_path):
    shape = _shape(tmp_path, WINS_OUTSIDE_DEP4, tpot={"dep4.con256": 35.0})
    assert shape.kind == SPLIT and shape.lost == ("dep4.con256",)
    assert shape.broken and shape.broken[0].startswith("dep4@con256: mean_tpot_ms")


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


def test_the_variant_request_names_the_losing_mode_and_its_condition(tmp_path):
    ctrl = _measure(tmp_path, "ctrl")
    cand = _measure(tmp_path, "cand", {**WINS_OUTSIDE_DEP4, "dep4.con256": 0.93})
    split = classify(cand, ctrl, NOISE, **RULE).as_dict()
    ctx = CampaignContext(
        repo=tmp_path, package=tmp_path / "pkg", campaign_dir=tmp_path / "camp",
        perf_root=tmp_path / "camp", target_dir="t/", target_tests="tests/t/", model="/ckpt",
        dev_spools=[tmp_path / "dev"], env_script=tmp_path / "env.sh", embargo=["x/"],
        workload="w", standings_table="s",
    )  # fmt: skip
    path = render_split_request(
        ctx, slug="ssm-decode-fused", variant="ssm-decode-fused-gated", worktree=tmp_path / "wt",
        base="0123456789abcdef", branch="tune/r/ssm-decode-fused-gated", best_branch="tune/r/best",
        split=split, deltas=compare(cand, ctrl, NOISE, **RULE).deltas,
        original=tmp_path / "camp" / "candidates" / "ssm-decode-fused" / "decision.yaml",
    )  # fmt: skip
    text = path.read_text()
    assert path == tmp_path / "camp" / "candidates" / "ssm-decode-fused-gated" / "request.md"
    assert "# Tuner request: ssm-decode-fused-gated" in text and "Kind: SPLIT" in text
    assert "**attention DP on and EP size 4**" in text
    assert "`dep4.con256.output_tok_s_gpu` -7.0%" in text and "Members that lost: `dep4`" in text
    assert "Where `mapping.enable_attention_dp and mapping.moe_ep_size == 4`" in text
    assert (
        "`variant_of: ssm-decode-fused`" in text and "rev-list --reverse tune/r/best..HEAD" in text
    )


# ------------------------------------------------------------------- the campaign


def _effect(config, tree):
    """Con 1 is faster with FAST; con 128 is slower with SLOW_HI unless GATED."""
    text = Path(tree, "modeling.py").read_text() if tree else ""
    lat = 100.0 * (1.2 if "FAST = True" in text else 1.0)
    thr = 2000.0 * (0.9 if "SLOW_HI" in text and "GATED" not in text else 1.0)
    if "WORSE" in text:
        thr *= 0.9
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


def test_a_variant_that_still_loses_is_reverted_and_never_split_again(tmp_path):
    _, state, turns, camp, _, log = _campaign(tmp_path, "WORSE = True\n")
    assert turns == ["roadmap", "fast-path", "fast-path-gated"]
    assert camp["fast-path-gated"]["status"] == "reverted" and not camp["fast-path-gated"]["split"]
    assert camp["fast-path-gated"]["reason"].startswith("variant of fast-path: LOSS")
    assert state["kept"] == []
    assert any("member-gated variant of fast-path: not a win" in m for m in log), log


def test_a_split_past_the_budget_asks_for_no_variant(tmp_path):
    _, state, turns, camp, kinds, _ = _campaign(tmp_path, "GATED = True\n", max_iterations=1)
    assert turns == ["roadmap", "fast-path"]
    assert camp["fast-path"]["status"] == "reverted"
    assert "no variant: the budget of 1 candidates is spent" in camp["fast-path"]["reason"]
    assert "split_closed" in kinds and state["kept"] == []


def test_split_variants_off_reverts_the_split(tmp_path):
    _, _, turns, camp, _, log = _campaign(tmp_path, "GATED = True\n", split_variants=False)
    assert turns == ["roadmap", "fast-path"] and camp["fast-path"]["status"] == "reverted"
    assert any("split variants are off" in m for m in log), log


def test_a_variant_turn_cut_off_by_a_restart_is_taken_or_asked_again(tmp_path):
    repo = _repo(tmp_path)
    task = {"trtllm_repo_path": str(repo), "checkpoint_path": "/ckpt", **_contract(tmp_path),
            "perf": {"workload": {"concurrency": [1, 128]}, "spools": ["s1"], "env_script": "/e"}}  # fmt: skip
    spec = TuneSpec.from_task(task, tmp_path / "ws", target="t/")
    stage = TuneStage(spec, {}, dispatch=lambda *a: "", checkpoint=lambda: None,
                      runner_factory=lambda spools, prefix: FakeRunner(spools, prefix, _effect),
                      poll_s=0, log=lambda m: None)  # fmt: skip
    out = spec.workdir / "candidates" / "a-gated"
    out.mkdir(parents=True)
    (out / "decision.yaml").write_text(yaml.safe_dump({"decision": "CANDIDATE", "worktree": ""}))

    def cand(slug, status, **kw):
        return SimpleNamespace(slug=slug, status=status, **kw)

    candidates = {
        "a": cand("a", "split", split={"variant": "a-gated"}),
        "a-gated": cand("a-gated", "issued"),
        "b": cand("b", "split", split={"variant": "b-gated"}),
        "b-gated": cand("b-gated", "issued"),
        "c": cand("c", "split", split={}),
    }
    submitted, closed, asked = [], [], {}
    camp = SimpleNamespace(
        state=SimpleNamespace(candidates=candidates),
        submit_candidate=lambda slug, path: submitted.append(slug),
        close_split=lambda slug, why: closed.append((slug, why)),
    )
    pool = SimpleNamespace(submit=lambda fn, st, variant, standings: variant)
    stage._split_turns(camp, pool, asked, budget=5)
    assert submitted == ["a-gated"]  # its decision is on disk
    assert asked == {"b-gated": "b-gated"}  # cut off: asked again
    assert closed == [("c", "the budget of 5 candidates is spent")]  # five candidates already
