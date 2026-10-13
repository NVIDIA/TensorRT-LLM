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
                (d / "meta.json").write_text(json.dumps({"label": d.name, "modeling_v2": merged.get("modeling_v2", "require"), "tree": {}}))
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
    assert set(turns) == {"roadmap", "fast-path", "no-op"}
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
