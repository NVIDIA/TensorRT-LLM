# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tune stage end to end on CPU: fake spool runners, a fake tuner, a real git repo."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from agent_flow.workflows.staircase_onboard.tune.stage import TuneSpec, TuneStage, validate_perf_block


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "T")
    _git(repo, "config", "user.email", "t@example.com")
    (repo / "modeling.py").write_text("FAST = False\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


class FakeRunner:
    """Finishes every job on the next poll, writing what bench.py / gate_accuracy.py would."""

    def __init__(self, spools, prefix, effect):
        self.spools, self.prefix, self.effect = list(spools), prefix, effect
        self.inflight: dict[str, tuple[str, str]] = {}
        self.jobs: dict[str, tuple[str, str | None]] = {}

    def adopt(self, label):
        return False

    def cancel(self, label):
        return self.inflight.pop(f"{self.prefix}-{label}", None) is not None

    def free_spools(self):
        busy = {s for s, _ in self.inflight.values()}
        return [s for s in self.spools if s not in busy]

    def submit(self, spool, label, command, *, wall="2h", tree=None, gpus="all"):
        name = f"{self.prefix}-{label}"
        self.inflight[name] = (spool, label)
        self.jobs[label] = (command, tree)
        return name

    def poll(self):
        done = []
        for name, (spool, label) in list(self.inflight.items()):
            command, tree = self.jobs[label]
            tokens = command.split()
            if "gate_accuracy.py" in command:
                out = Path(tokens[tokens.index("--out") + 1])
                out.mkdir(parents=True, exist_ok=True)
                (out / "result.json").write_text(json.dumps({"verdict": "PASS", "score": 96.0, "floor": 95.3}))
            else:
                configs = [Path(tokens[i + 1]) for i, t in enumerate(tokens) if t == "--config"]
                merged: dict = {}
                for c in configs:
                    merged.update(yaml.safe_load(c.read_text()) or {})
                lat, thr = self.effect(merged, tree)
                d = Path(tokens[tokens.index("--out-root") + 1]) / tokens[tokens.index("--label") + 1]
                d.mkdir(parents=True, exist_ok=True)
                (d / "summary.json").write_text(json.dumps([
                    {"concurrency": 1, "tok_s_user": lat, "output_tok_s_gpu": lat, "mean_tpot_ms": 10, "mean_ttft_ms": 200},
                    {"concurrency": 128, "tok_s_user": thr / 128, "output_tok_s_gpu": thr, "mean_tpot_ms": 50, "mean_ttft_ms": 4000},
                ]))
                status = Path(spool, "status.json")
                host = json.loads(status.read_text())["host"] if status.exists() else ""
                (d / "meta.json").write_text(json.dumps({"label": d.name, "modeling_v2": merged.get("modeling_v2", "require"),
                                                         "tree": {}, "hostname": host}))
            del self.inflight[name]
            done.append((label, spool, 0))
        return done


def _effect(config, tree):
    lat, thr = 100.0, 2000.0
    if config.get("modeling_v2") == "off":
        lat, thr = 120.0, 2300.0
    if (config.get("cuda_graph_config") or {}).get("enable_padding"):
        thr *= 1.05
    if tree and Path(tree, "modeling.py").read_text().startswith("FAST = True"):
        lat *= 1.2
    return lat, thr


def test_perf_block_validation():
    assert validate_perf_block({}) == []
    assert validate_perf_block({"perf": {"workload": {"isl": 8192, "concurrency": [1, 128]}}}) == []
    errors = validate_perf_block({"perf": {"bogus": 1, "workload": {"concurrency": []}, "reference": "x"}})
    assert any("bogus" in e for e in errors) and any("concurrency" in e for e in errors) and any("reference" in e for e in errors)


def test_the_stage_runs_ground_ladder_campaign_and_wrap(tmp_path):
    repo = _repo(tmp_path)
    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump({"modeling_v2": "require", "max_batch_size": 128}))
    task = {
        "trtllm_repo_path": str(repo),
        "checkpoint_path": "/ckpt",
        "perf": {
            "identity_configs": [str(identity)],
            "workload": {"isl": 8192, "osl": 1024, "concurrency": [1, 128]},
            "spools": ["s1", "s2"],
            "env_script": "/env.sh",
            "ladder": {"only": ["cg-padding", "gc-gen0-100k"]},
            "campaign": {"max_parallel_tuners": 2, "accuracy": {"task": "gsm8k", "anchor": 96.36}},
        },
    }
    spec = TuneSpec.from_task(task, tmp_path / "ws", target="tensorrt_llm/_torch/_experimental/modeling_v2/models/f/t/")
    turns = []

    def dispatch(role, what, prompt):
        turns.append(what)
        if what == "roadmap-2":
            (spec.workdir / "roadmap-2.yaml").write_text("[]")
            return "nothing left"
        if what == "roadmap":
            (spec.workdir / "roadmap.yaml").write_text(yaml.safe_dump([
                {"slug": "fast-path", "kind": "fusion", "hypothesis": "make it fast"},
                {"slug": "no-op", "kind": "kernel", "hypothesis": "a source that does not win"},
            ]))
            return "roadmap written"
        out = spec.workdir / "candidates" / what
        wt = spec.workdir / "worktrees" / what
        if what == "fast-path":
            (wt / "modeling.py").write_text("FAST = True\n")
            _git(wt, "commit", "-q", "-am", "fast path")
            decision = {"decision": "CANDIDATE", "worktree": str(wt), "commits": [_git(wt, "rev-parse", "HEAD")]}
        else:
            decision = {"decision": "DROPPED", "worktree": str(wt), "microbench": {"table": "slower"}}
        (out / "decision.yaml").write_text(yaml.safe_dump(decision))
        return "done"

    state: dict = {}
    stage = TuneStage(spec, state, dispatch=dispatch, checkpoint=lambda: None,
                      runner_factory=lambda spools, prefix: FakeRunner(spools, prefix, _effect), poll_s=0, log=lambda m: None)
    assert stage.run() == "tuned"
    assert state["phase"] == "done"
    assert state["ladder_kept"] == ["cg-padding"]
    assert state["kept"] == ["fast-path"]
    assert set(turns) == {"roadmap", "fast-path", "no-op", "roadmap-2"}  # the spent roadmap is refreshed once
    best = Path(state["best_tree"])
    assert (best / "modeling.py").read_text() == "FAST = True\n"
    summary = (spec.workdir / "summary.md").read_text()
    assert "t-final" in summary and "t-reference-tuned" in summary
    kinds = [json.loads(line)["kind"] for line in (spec.workdir / "perf_ledger.jsonl").read_text().splitlines()]
    assert {"noise_floor", "tuner_decision", "candidate_e2e", "candidate_gate", "candidate_kept"} <= set(kinds)
    # resume after completion re-measures nothing
    again = TuneStage(spec, state, dispatch=dispatch, checkpoint=lambda: None,
                      runner_factory=lambda spools, prefix: FakeRunner(spools, prefix, _effect), poll_s=0, log=lambda m: None)
    assert again.run() == "tuned"


def test_a_config_change_re_measures_against_a_control_of_that_config_and_host(tmp_path):
    """Controls are keyed by tree, config AND host: after the ladder's overlay is adopted a
    candidate measured at the old config is re-measured, and judged only against a control
    of the new config from its own host (a 4% slower host must not turn a win into a loss)."""
    from agent_flow.workflows.staircase_onboard.tune.campaign import Campaign
    from agent_flow.workflows.staircase_onboard.tune.ladder import BenchSpec
    from agent_flow.workflows.staircase_onboard.tune.records import NoiseFloor
    from agent_flow.workflows.staircase_onboard.tune.worktree import open_candidate

    repo = _repo(tmp_path)
    spools = []
    for name, host in (("s1", "fast"), ("s2", "slow")):
        d = tmp_path / name
        d.mkdir()
        (d / "status.json").write_text(json.dumps({"host": host}))
        spools.append(str(d))

    class HostRunner(FakeRunner):
        def poll(self):
            done = super().poll()
            for label, spool, rc in done:
                if "gate" in label:
                    continue
                tokens = self.jobs[label][0].split()
                d = Path(tokens[tokens.index("--out-root") + 1]) / label
                meta = json.loads((d / "meta.json").read_text())
                meta["hostname"] = json.loads((Path(spool) / "status.json").read_text())["host"]
                (d / "meta.json").write_text(json.dumps(meta))
                if meta["hostname"] == "slow":
                    rows = json.loads((d / "summary.json").read_text())
                    for r in rows:
                        r["tok_s_user"] *= 0.96
                        r["output_tok_s_gpu"] *= 0.96
                    (d / "summary.json").write_text(json.dumps(rows))
            return done

    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump({"modeling_v2": "require"}))
    overlay = tmp_path / "ladder.yaml"
    overlay.write_text(yaml.safe_dump({"cuda_graph_config": {"enable_padding": True}}))
    bench = BenchSpec(bench_py="b", model="m", env_script="e", isl=8192, osl=1024, concurrencies=[1, 128], tree=str(repo))
    perf = HostRunner(spools, "camp", _effect)
    gates = HostRunner([spools[0]], "campg", _effect)
    work = tmp_path / "camp"
    camp = Campaign(workdir=work, repo=repo, bench=bench, base_configs=[identity], perf_runner=perf, gate_runner=gates,
                    gate_cmd=lambda tree, configs, out: f"python gate_accuracy.py --out {out}",
                    noise=NoiseFloor(spread={"con1.tok_s_user": 0.001}), log=lambda m: None)
    camp.state.best_tree = str(open_candidate(repo, work / "wt", "best", "HEAD"))
    camp.state.best_commit = _git(Path(camp.state.best_tree), "rev-parse", "HEAD")
    wt = open_candidate(repo, work / "wt", "fast-path", "HEAD")
    (wt / "modeling.py").write_text("FAST = True\n")
    _git(wt, "commit", "-q", "-am", "fast")
    decision = tmp_path / "decision.yaml"
    decision.write_text(yaml.safe_dump({"decision": "CANDIDATE", "worktree": str(wt), "commits": [_git(wt, "rev-parse", "HEAD")]}))
    camp.submit_candidate("fast-path", decision)
    camp.step()                                   # controls queued on both hosts, nothing judged yet
    camp.state.config_overlays = [str(overlay)]   # the ladder's overlay is adopted mid-campaign
    for _ in range(30):
        if not camp.step():
            break
    st = camp.state.candidates["fast-path"]
    assert st.status == "kept", camp.summary()
    assert st.eval_config == str(overlay)
    ctrl = json.loads((work / "data" / st.control_label / "meta.json").read_text())
    cand = json.loads((work / "data" / st.e2e_label / "meta.json").read_text())
    assert ctrl["hostname"] == cand["hostname"]
    assert st.control_label != "" and "-id-" not in st.control_label   # a control of the new config
    assert round(st.deltas["con1.tok_s_user"], 3) == 0.2


def test_a_candidate_that_conflicts_with_a_keep_is_rebased_and_retried(tmp_path):
    repo = _repo(tmp_path)
    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump({"modeling_v2": "require"}))
    task = {
        "trtllm_repo_path": str(repo), "checkpoint_path": "/ckpt",
        "perf": {"identity_configs": [str(identity)], "workload": {"isl": 8192, "osl": 1024, "concurrency": [1, 128]},
                 "spools": ["s1"], "env_script": "/env.sh", "reference": "none", "ladder": {"enabled": False},
                 "campaign": {"max_parallel_tuners": 1, "accuracy": {"task": "gsm8k", "anchor": 96.36}}},
    }
    spec = TuneSpec.from_task(task, tmp_path / "ws", target="t/")
    turns = []

    def effect(config, tree):
        text = Path(tree, "modeling.py").read_text() if tree else ""
        return 100.0 * (1.2 if "FAST" in text and "= True" in text else 1.0) * (1.1 if "SIDE" in text else 1.0), 2000.0

    def dispatch(role, what, prompt):
        turns.append(what)
        ws = spec.workdir
        if what == "roadmap":
            (ws / "roadmap.yaml").write_text(yaml.safe_dump([{"slug": "fast-path"}, {"slug": "side-path"}]))
            return ""
        if what.startswith("roadmap-"):
            assert "side-path" in Path(ws / f"{what}-request.md").read_text()  # told what was tried
            return ""
        slug = what.removeprefix("rebase-")
        wt = ws / "worktrees" / slug
        if what == "fast-path":
            (wt / "modeling.py").write_text("FAST = True\n")
        elif what == "side-path":
            (wt / "modeling.py").write_text("FAST = False\nSIDE = 1\n")
        else:  # the rebase turn: resolve by keeping both sides
            subprocess.run(["git", "-C", str(wt), "rebase", "tune/best"], capture_output=True)
            (wt / "modeling.py").write_text("FAST = True\nSIDE = 1\n")
            _git(wt, "add", "modeling.py")
            subprocess.run(["git", "-C", str(wt), "-c", "core.editor=true", "rebase", "--continue"], capture_output=True)
        if what in ("fast-path", "side-path"):
            _git(wt, "commit", "-q", "-am", what)
        commits = _git(wt, "rev-list", "--reverse", "tune/best..HEAD").split()
        (ws / "candidates" / slug / "decision.yaml").write_text(
            yaml.safe_dump({"decision": "CANDIDATE", "worktree": str(wt), "commits": commits}))
        return ""

    state: dict = {"phase": "campaign", "noise_spread": {}, "noise_samples": 0}
    stage = TuneStage(spec, state, dispatch=dispatch, checkpoint=lambda: None,
                      runner_factory=lambda spools, prefix: FakeRunner(spools, prefix, effect), poll_s=0, log=lambda m: None)
    assert stage.run() == "tuned"
    assert "rebase-side-path" in turns, turns
    assert state["kept"] == ["fast-path", "side-path"], state
    assert Path(state["best_tree"], "modeling.py").read_text() == "FAST = True\nSIDE = 1\n"


def test_winners_on_one_best_are_stacked_measured_and_gated_once(tmp_path):
    """Two independent winners: one combined measurement and one gate, both kept --
    instead of keeping one and re-measuring the other on the moved best."""
    from agent_flow.workflows.staircase_onboard.tune.campaign import Campaign
    from agent_flow.workflows.staircase_onboard.tune.ladder import BenchSpec
    from agent_flow.workflows.staircase_onboard.tune.records import NoiseFloor
    from agent_flow.workflows.staircase_onboard.tune.worktree import open_candidate

    repo = _repo(tmp_path)
    (repo / "attn.py").write_text("FUSED = False\n")
    _git(repo, "add", "attn.py")
    _git(repo, "commit", "-q", "-m", "attn")

    def effect(config, tree):
        lat = 100.0
        if tree and "FAST = True" in Path(tree, "modeling.py").read_text():
            lat *= 1.20
        if tree and "FUSED = True" in Path(tree, "attn.py").read_text():
            lat *= 1.10
        return lat, 2000.0

    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump({"modeling_v2": "require"}))
    bench = BenchSpec(bench_py="b", model="m", env_script="e", isl=8192, osl=1024, concurrencies=[1, 128], tree=str(repo))
    perf, gates = FakeRunner(["s1"], "camp", effect), FakeRunner(["g1"], "campg", effect)
    work = tmp_path / "camp"
    camp = Campaign(workdir=work, repo=repo, bench=bench, base_configs=[identity], perf_runner=perf, gate_runner=gates,
                    gate_cmd=lambda tree, configs, out: f"python gate_accuracy.py --out {out}",
                    noise=NoiseFloor(spread={"con1.tok_s_user": 0.001}), log=lambda m: None)
    camp.state.best_tree = str(open_candidate(repo, work / "wt", "best", "HEAD"))
    camp.state.best_commit = _git(Path(camp.state.best_tree), "rev-parse", "HEAD")
    for slug, path, text in (("fast-path", "modeling.py", "FAST = True\n"), ("fused-attn", "attn.py", "FUSED = True\n")):
        wt = open_candidate(repo, work / "wt", slug, "HEAD")
        (wt / path).write_text(text)
        _git(wt, "commit", "-q", "-am", slug)
        d = tmp_path / f"{slug}.yaml"
        d.write_text(yaml.safe_dump({"decision": "CANDIDATE", "worktree": str(wt), "commits": [_git(wt, "rev-parse", "HEAD")]}))
        camp.submit_candidate(slug, d)
    for _ in range(40):
        if not camp.step():
            break
    assert {s: c.status for s, c in camp.state.candidates.items()} == {"fast-path": "kept", "fused-attn": "kept"}
    kinds = [json.loads(l)["kind"] for l in (work / "perf_ledger.jsonl").read_text().splitlines()]
    assert "batch_formed" in kinds and "batch_kept" in kinds
    assert sum(1 for k in kinds if k == "candidate_gate") == 1      # one gate for the stack
    best = Path(camp.state.best_tree)
    assert (best / "modeling.py").read_text() == "FAST = True\n" and (best / "attn.py").read_text() == "FUSED = True\n"


def test_a_batch_stacks_the_compatible_winners_with_the_largest_summed_gain(tmp_path):
    """The largest single winner conflicts with two smaller ones that stack together:
    the batch takes the pair, and the large one is rebased onto the batch while it is
    measured, waits, and is measured on the new best once the batch is kept."""
    import os

    from agent_flow.workflows.staircase_onboard.tune.campaign import Campaign
    from agent_flow.workflows.staircase_onboard.tune.ladder import BenchSpec
    from agent_flow.workflows.staircase_onboard.tune.records import NoiseFloor
    from agent_flow.workflows.staircase_onboard.tune.worktree import open_candidate

    repo = _repo(tmp_path)
    (repo / "attn.py").write_text("FUSED = False\n")
    _git(repo, "add", "attn.py")
    _git(repo, "commit", "-q", "-m", "attn")

    def effect(config, tree):
        lat = 100.0
        text = Path(tree, "modeling.py").read_text() + Path(tree, "attn.py").read_text() if tree else ""
        for flag, gain in (("FAST = True", 1.20), ("FUSED = True", 1.10), ("BIG = True", 1.25)):
            if flag in text:
                lat *= gain
        return lat, 2000.0

    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump({"modeling_v2": "require"}))
    bench = BenchSpec(bench_py="b", model="m", env_script="e", isl=8192, osl=1024, concurrencies=[1, 128], tree=str(repo))
    perf, gates = FakeRunner(["s1"], "camp", effect), FakeRunner(["g1"], "campg", effect)
    work = tmp_path / "camp"
    camp = Campaign(workdir=work, repo=repo, bench=bench, base_configs=[identity], perf_runner=perf, gate_runner=gates,
                    gate_cmd=lambda tree, configs, out: f"python gate_accuracy.py --out {out}",
                    noise=NoiseFloor(spread={"con1.tok_s_user": 0.001}), log=lambda m: None)
    camp.state.best_tree = str(open_candidate(repo, work / "wt", "best", "HEAD"))
    camp.state.best_commit = _git(Path(camp.state.best_tree), "rev-parse", "HEAD")
    edits = {"fast-path": {"modeling.py": "FAST = True\n"}, "fused-attn": {"attn.py": "FUSED = True\n"},
             "big-one": {"modeling.py": "BIG = True\n", "attn.py": "BIG_ATTN = True\n"}}
    wts = {}
    for slug, files in edits.items():
        wt = wts[slug] = open_candidate(repo, work / "wt", slug, "HEAD")
        for path, text in files.items():
            (wt / path).write_text(text)
        _git(wt, "commit", "-q", "-am", slug)
        d = tmp_path / f"{slug}.yaml"
        d.write_text(yaml.safe_dump({"decision": "CANDIDATE", "worktree": str(wt), "commits": [_git(wt, "rev-parse", "HEAD")]}))
        camp.submit_candidate(slug, d)
    for _ in range(40):
        camp.step()
        if camp.state.candidates["big-one"].status == "conflict":
            break
    big = camp.state.candidates["big-one"]
    assert big.status == "conflict" and big.rebase_onto == "tune/eval-batch"
    assert sorted(camp.state.batch["members"]) == ["fast-path", "fused-attn"]

    # The rebase turn: onto the batch's commit, both sides kept.
    wt = wts["big-one"]
    env = {**os.environ, "GIT_EDITOR": "true"}
    subprocess.run(["git", "-C", str(wt), "rebase", big.rebase_ref], capture_output=True, text=True, env=env)
    (wt / "modeling.py").write_text("FAST = True\nBIG = True\n")
    (wt / "attn.py").write_text("FUSED = True\nBIG_ATTN = True\n")
    _git(wt, "add", "-A")
    subprocess.run(["git", "-C", str(wt), "rebase", "--continue"], check=True, capture_output=True, env=env)
    d = tmp_path / "big-one.yaml"
    d.write_text(yaml.safe_dump({"decision": "CANDIDATE", "worktree": str(wt),
                                 "commits": _git(wt, "rev-list", "--reverse", f"{big.rebase_ref}..HEAD").splitlines()}))
    assert camp.submit_candidate("big-one", d).status == "waiting"
    for _ in range(60):
        if not camp.step():
            break
    assert {s: c.status for s, c in camp.state.candidates.items()} == {"fast-path": "kept", "fused-attn": "kept", "big-one": "kept"}
    best = Path(camp.state.best_tree)
    assert (best / "modeling.py").read_text() == "FAST = True\nBIG = True\n"
    formed = [json.loads(l) for l in (work / "perf_ledger.jsonl").read_text().splitlines() if '"batch_formed"' in l]
    assert formed[0]["excluded"] == ["big-one"]


def test_a_finished_candidates_eval_checkout_is_reused(tmp_path):
    """A fresh eval worktree is a full checkout (minutes on a network filesystem);
    a reverted candidate's checkout is moved and re-pointed instead."""
    from agent_flow.workflows.staircase_onboard.tune.campaign import Campaign
    from agent_flow.workflows.staircase_onboard.tune.ladder import BenchSpec
    from agent_flow.workflows.staircase_onboard.tune.records import NoiseFloor
    from agent_flow.workflows.staircase_onboard.tune.worktree import open_candidate

    repo = _repo(tmp_path)
    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump({"modeling_v2": "require"}))
    bench = BenchSpec(bench_py="b", model="m", env_script="e", isl=8192, osl=1024, concurrencies=[1, 128], tree=str(repo))
    perf, gates = FakeRunner(["s1"], "camp", _effect), FakeRunner(["g1"], "campg", _effect)
    work = tmp_path / "camp"
    camp = Campaign(workdir=work, repo=repo, bench=bench, base_configs=[identity], perf_runner=perf, gate_runner=gates,
                    gate_cmd=lambda tree, configs, out: f"python gate_accuracy.py --out {out}",
                    noise=NoiseFloor(spread={"con1.tok_s_user": 0.001}), log=lambda m: None)
    camp.state.best_tree = str(open_candidate(repo, work / "wt", "best", "HEAD"))
    camp.state.best_commit = _git(Path(camp.state.best_tree), "rev-parse", "HEAD")

    def submit(slug, text):
        wt = open_candidate(repo, work / "wt", slug, "HEAD")
        (wt / "modeling.py").write_text(text)
        _git(wt, "commit", "-q", "-am", slug)
        d = tmp_path / f"{slug}.yaml"
        d.write_text(yaml.safe_dump({"decision": "CANDIDATE", "worktree": str(wt), "commits": [_git(wt, "rev-parse", "HEAD")]}))
        return camp.submit_candidate(slug, d)

    submit("no-gain", "FAST = False\nX = 1\n")
    for _ in range(20):
        if not camp.step():
            break
    assert camp.state.candidates["no-gain"].status == "reverted"
    st = submit("fast-path", "FAST = True\n")
    assert Path(st.eval_tree) == work / "wt" / "eval-fast-path"
    assert not (work / "wt" / "eval-no-gain").exists()
    assert (Path(st.eval_tree) / "modeling.py").read_text() == "FAST = True\n"
    for _ in range(20):
        if not camp.step():
            break
    assert camp.state.candidates["fast-path"].status == "kept"


def test_one_control_is_queued_per_waiting_measurement_not_per_host(tmp_path):
    """A new best tree needs a same-host control before a candidate is judged. The
    candidate is queued behind one control on one node; the other nodes stay free."""
    from agent_flow.workflows.staircase_onboard.tune.campaign import Campaign
    from agent_flow.workflows.staircase_onboard.tune.ladder import BenchSpec
    from agent_flow.workflows.staircase_onboard.tune.records import NoiseFloor
    from agent_flow.workflows.staircase_onboard.tune.worktree import open_candidate

    repo = _repo(tmp_path)
    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump({"modeling_v2": "require"}))
    bench = BenchSpec(bench_py="b", model="m", env_script="e", isl=8192, osl=1024, concurrencies=[1, 128], tree=str(repo))
    spools = []
    for i in range(4):
        sp = tmp_path / f"s{i}"
        sp.mkdir()
        (sp / "status.json").write_text(json.dumps({"host": f"node{i}"}))
        spools.append(str(sp))
    perf, gates = FakeRunner(spools, "camp", _effect), FakeRunner(["g1"], "campg", _effect)
    work = tmp_path / "camp"
    camp = Campaign(workdir=work, repo=repo, bench=bench, base_configs=[identity], perf_runner=perf, gate_runner=gates,
                    gate_cmd=lambda tree, configs, out: f"python gate_accuracy.py --out {out}",
                    noise=NoiseFloor(spread={"con1.tok_s_user": 0.001}), log=lambda m: None)
    camp.state.best_tree = str(open_candidate(repo, work / "wt", "best", "HEAD"))
    camp.state.best_commit = _git(Path(camp.state.best_tree), "rev-parse", "HEAD")
    wt = open_candidate(repo, work / "wt", "fast-path", "HEAD")
    (wt / "modeling.py").write_text("FAST = True\n")
    _git(wt, "commit", "-q", "-am", "fast")
    d = tmp_path / "fast-path.yaml"
    d.write_text(yaml.safe_dump({"decision": "CANDIDATE", "worktree": str(wt), "commits": [_git(wt, "rev-parse", "HEAD")]}))
    camp.submit_candidate("fast-path", d)
    camp.step()
    queued = [(sp, label) for sp, label in perf.inflight.values()]
    assert sorted(label.split("-")[0] for _, label in queued) == ["ctl", "e2e"]
    assert len({sp for sp, _ in queued}) == 1          # behind its control, on the same node
    for _ in range(20):
        if not camp.step():
            break
    assert camp.state.candidates["fast-path"].status == "kept"
