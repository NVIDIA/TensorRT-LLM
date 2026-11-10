# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune campaign regression tests."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import replace
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.compute.executor import TUNER
from agent_flow.workflows.trtllm_modeling_v2.tune.campaign import Campaign
from agent_flow.workflows.trtllm_modeling_v2.tune.measure import BenchSpec
from agent_flow.workflows.trtllm_modeling_v2.tune.records import NoiseFloor
from agent_flow.workflows.trtllm_modeling_v2.tune.spec import TuneSpec
from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneStage
from agent_flow.workflows.trtllm_modeling_v2.tune.worktree import open_candidate


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _contract(tmp_path: Path, anchor: float = 96.36, **accuracy) -> dict:
    """The contract fields a keep gate reads: a tp1 target anchored to a gsm8k row."""
    ref = tmp_path / "references" / "gsm8k.yaml"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(yaml.safe_dump({"org/model": [{"accuracy": anchor}]}))
    return {
        "parallel": "tp1",
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": str(ref),
            "model_key": "org/model",
            **accuracy,
        },
    }


def _keep_gate(tmp_path: Path, campaign: dict | None = None, **contract):
    """The keep gate :func:`_contract` defines, with the campaign's ``accuracy`` block."""
    from agent_flow.workflows.trtllm_modeling_v2.tune.keep_gate import resolve_keep_gate

    task = {"trtllm_repo_path": str(tmp_path), "checkpoint_path": "/ckpt", **_contract(tmp_path)}
    return resolve_keep_gate({**task, **contract}, campaign or {}, workspace=tmp_path)


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
    """Finishes every job on the next poll, leaving what bench.py, trtllm-eval or pytest would.

    A keep gate's member scores ``score(label, tree)`` (96.0 by default, above the floor of
    :func:`_contract`'s anchor); its suite passes.
    """

    def __init__(self, spools, prefix, effect, score=None, first_token=None):
        self.spools, self.prefix, self.effect = (list(spools), prefix, effect)
        self.score = score or (lambda label, tree: 96.0)
        self.first_token = first_token or (lambda tree: {7: -0.1, 9: -2.5})
        self.inflight: dict[str, tuple[str, str]] = {}
        self.jobs: dict[str, tuple[str, str | None]] = {}
        self.outputs: dict[str, str] = {}
        self.receipts: dict[str, dict] = {}
        self.submitted: list[str] = []

    def adopt(self, label, *, active_only=False):
        return False

    def place(self, gpus, wall="1h"):
        load = {s: sum((1 for sp, _ in self.inflight.values() if sp == s)) for s in self.spools}
        return min(self.spools, key=lambda s: load[s])

    def widest(self):
        return 0

    def output(self, label):
        return (self.outputs.get(label, ""), self.receipts.get(label, {}))

    def queued(self, label):
        return any((queued == label for _, queued in self.inflight.values()))

    def cancel(self, label):
        return self.inflight.pop(f"{self.prefix}-{label}", None) is not None

    def free_spools(self):
        busy = {s for s, _ in self.inflight.values()}
        return [s for s in self.spools if s not in busy]

    def alive(self, spool):
        return spool in self.spools

    def submit(self, spool, label, command, *, wall="2h", tree=None, gpus="all"):
        self.submitted.append(label)
        name = f"{self.prefix}-{label}"
        self.inflight[name] = (spool, label)
        self.jobs[label] = (command, tree)
        self.where = getattr(self, "where", {})
        self.where[label] = spool
        return name

    def poll(self):
        done = []
        for name, (spool, label) in list(self.inflight.items()):
            command, tree = self.jobs[label]
            tokens = shlex.split(command)
            status = Path(spool, "status.json")
            host = json.loads(status.read_text())["host"] if status.exists() else "h1"
            self.receipts[label] = {"rc": 0, "host": host, "tree": tree}
            if "--harness-sha256" in tokens:
                result = Path(tokens[tokens.index("--output") + 1])
                result.parent.mkdir(parents=True, exist_ok=True)
                latency = 100 / self.effect({}, tree)[0]
                result.write_text(
                    json.dumps(
                        {
                            "commit": tokens[tokens.index("--commit") + 1],
                            "harness_sha256": tokens[tokens.index("--harness-sha256") + 1],
                            "hostname": host,
                            "module": str(Path(tree) / "tensorrt_llm/__init__.py"),
                            "latency_us": {"decode-b1-bf16-graph": latency},
                            "numerics": {"decode-b1-bf16-graph": {"exact": True, "error": 0.0}},
                        }
                    )
                )
            elif "engine_driver.py" in command:
                spec = json.loads(Path(tokens[tokens.index("--spec") + 1]).read_text())
                row = [[tok, lp, None] for tok, lp in self.first_token(tree).items()]
                self.outputs[label] = "".join(
                    (
                        f"MV2FLOW_RECORD {json.dumps({'index': i, 'id': r['id'], 'top_logprobs': [row]})}\n"
                        for i, r in enumerate(spec["requests"])
                    )
                )
            elif "trtllm-eval " in command:
                score = self.score(label, tree)
                self.outputs[label] = f"[I] lm-eval gsm8k average accuracy: {score}\n"
            elif " -m pytest " in command:
                self.outputs[label] = "3 passed in 0.01s\n"
            else:
                configs = [Path(tokens[i + 1]) for i, t in enumerate(tokens) if t == "--config"]
                merged: dict = {}
                for c in configs:
                    merged.update(yaml.safe_load(c.read_text()) or {})
                lat, thr = self.effect(merged, tree)
                d = (
                    Path(tokens[tokens.index("--out-root") + 1])
                    / tokens[tokens.index("--label") + 1]
                )
                d.mkdir(parents=True, exist_ok=True)
                (d / "summary.json").write_text(
                    json.dumps(
                        [
                            {
                                "concurrency": 1,
                                "tok_s_user": lat,
                                "output_tok_s_gpu": lat,
                                "mean_tpot_ms": 10,
                                "mean_ttft_ms": 200,
                            },
                            {
                                "concurrency": 128,
                                "tok_s_user": thr / 128,
                                "output_tok_s_gpu": thr,
                                "mean_tpot_ms": 50,
                                "mean_ttft_ms": 4000,
                            },
                        ]
                    )
                )
                (d / "meta.json").write_text(
                    json.dumps(
                        {
                            "label": d.name,
                            "modeling_v2": merged.get("modeling_v2", "require"),
                            "tree": {
                                "commit": _git(
                                    Path(tree or tokens[tokens.index("--tree") + 1]),
                                    "rev-parse",
                                    "HEAD",
                                )
                            },
                            "hostname": host,
                            "isl": int(tokens[tokens.index("--isl") + 1]),
                            "osl": int(tokens[tokens.index("--osl") + 1]),
                            "concurrencies": [
                                int(c) for c in tokens[tokens.index("--concurrency") + 1].split(",")
                            ],
                            "config": merged,
                        }
                    )
                )
            del self.inflight[name]
            done.append((label, spool, 0))
        return done


def _kernel_ledger(item_slug: str) -> dict:
    """A perf_optimize kernel ledger: two kernels, the hot one's 'faster' answered by ``item_slug``."""

    def row(kernel, share, faster):
        overlap = {
            "disposition": "dismissed",
            "concurrent_with": "serialized on the main stream",
            "ref": "no independent work",
        }
        return {
            "kernel": kernel,
            "full_name": f"void {kernel}_kernel",
            "share_pct": share,
            "ncu": "unavailable: nsys-only roadmap turn",
            "bound": "memory",
            "elimination": {
                "disposition": "dismissed",
                "why_it_runs": "its output feeds the next layer",
                "ref": "mandatory",
            },
            "faster": faster,
            "fusion": {
                "disposition": "dismissed",
                "neighbors": "norm -> THIS -> gemm",
                "ref": "multi-consumer",
            },
            "overlap": overlap,
        }

    return {
        "version": 1,
        "source": "profiles/roadmap",
        "coverage": {
            "enumerated_share_pct": 90.0,
            "other_share_pct": 10.0,
            "min_share_pct": 1.0,
            "gpu_busy_pct": 97.0,
        },
        "kernels": [
            row("moe_gemm", 60.0, {"disposition": "item", "ref": item_slug}),
            row("rmsnorm", 30.0, {"disposition": "dismissed", "ref": "at its bandwidth floor"}),
        ],
    }


def _microbench(path: Path) -> dict:
    script = path / "bench.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("def benchmark():\n    return {'decode-b1-bf16-graph': lambda: None}\n")
    share = {"decode-b1-bf16-graph": {"con1": 0.5, "con128": 0.5}}
    return {"script": str(script), "cases": ["decode-b1-bf16-graph"], "share": share}


def _effect(config, tree):
    lat, thr = (100.0, 2000.0)
    if config.get("modeling_v2") == "off":
        lat, thr = (120.0, 2300.0)
    if (config.get("cuda_graph_config") or {}).get("enable_padding"):
        thr *= 1.05
    if tree and Path(tree, "modeling.py").read_text().startswith("FAST = True"):
        lat *= 1.2
    return (lat, thr)


def _write_sol(workdir: Path, work: float = 10) -> None:
    """Synthetic work/capacity inputs, not device specifications or fitted ceilings."""
    from agent_flow.workflows.trtllm_modeling_v2.tune.sol import calculate

    keys = ("con1.tok_s_user", "con128.output_tok_s_gpu", "con128.tok_s_user")
    inputs = {
        "hardware": {
            "compute_flops_s": {"value": 1000000, "source": "synthetic test capacity"},
            "memory_bytes_s": {"value": 1000000, "source": "synthetic test capacity"},
            "assumptions": "synthetic one-device work, not a hardware specification",
        },
        "headline": {
            k: {
                "flops": work,
                "bytes": work,
                "output_tokens": 1,
                "gpu_count": 1,
                "assumptions": "synthetic whole-interval work",
            }
            for k in keys
        },
    }
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "sol.yaml").write_text(yaml.safe_dump(calculate(inputs)))


def test_the_stage_runs_ground_campaign_and_wrap(tmp_path):
    repo = _repo(tmp_path)
    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump({"modeling_v2": "require", "max_batch_size": 128}))
    task = {
        "trtllm_repo_path": str(repo),
        "checkpoint_path": "/ckpt",
        **_contract(tmp_path),
        "perf": {
            "identity_configs": [str(identity)],
            "workload": {"isl": 8192, "osl": 1024, "concurrency": [1, 128]},
            "spools": ["s1", "s2"],
            "env_script": "/env.sh",
            "campaign": {"max_parallel_tuners": 2},
        },
    }
    spec = TuneSpec.from_task(
        task, tmp_path / "ws", target="tensorrt_llm/_torch/_experimental/modeling_v2/models/f/t/"
    )
    turns = []
    asks = {}

    def dispatch(role, what, prompt):
        turns.append(what)
        if what.startswith("sol-"):
            _write_sol(spec.workdir / "sol" / what.removeprefix("sol-"))
            return "ceilings written"
        if what == "roadmap-2":
            (spec.workdir / "roadmap-2.yaml").write_text("[]")
            return "nothing left"
        if what == "roadmap":
            asks["roadmap"] = (spec.workdir / "roadmap-request.md").read_text()
            (spec.workdir / "roadmap.yaml").write_text(
                yaml.safe_dump(
                    [
                        {"slug": "fast-path", "kind": "fusion", "hypothesis": "make it fast"},
                        {
                            "slug": "no-op",
                            "kind": "kernel",
                            "hypothesis": "a source that does not win",
                        },
                    ]
                )
            )
            (spec.workdir / "roadmap-kernel_ledger.yaml").write_text(
                yaml.safe_dump(_kernel_ledger("fast-path"))
            )
            return "roadmap written"
        out = spec.workdir / "candidates" / what
        wt = spec.workdir / "worktrees" / what
        if what == "fast-path":
            (wt / "modeling.py").write_text("FAST = True\n")
            _git(wt, "commit", "-q", "-am", "fast path")
            decision = {
                "decision": "CANDIDATE",
                "worktree": str(wt),
                "commits": [_git(wt, "rev-parse", "HEAD")],
                "microbench": _microbench(out),
                "sources": [
                    {"source": "flashinfer", "ref": "flashinfer.fused_x", "outcome": "used"}
                ],
            }
        else:
            decision = {
                "decision": "DROPPED",
                "worktree": str(wt),
                "microbench": {"table": "slower"},
            }
        (out / "decision.yaml").write_text(yaml.safe_dump(decision))
        return "done"

    state: dict = {}
    runners = []

    def runner_factory(spools, prefix):
        runner = FakeRunner(spools, prefix, _effect)
        runners.append(runner)
        return runner

    stage = TuneStage(
        spec,
        state,
        dispatch=dispatch,
        checkpoint=lambda: None,
        runner_factory=runner_factory,
        poll_s=0,
        log=lambda m: None,
    )
    assert stage.run() == "tuned"
    submitted = [label for runner in runners for label in runner.submitted]
    assert submitted.count("t-baseline") == submitted.count("t-final") == 1
    assert not any((label.startswith(("t-baseline-r", "e2e-")) for label in submitted))
    assert len([label for label in submitted if label.startswith("micro-")]) == 2
    assert sum((label.startswith("ckpt-") for label in submitted)) == 1
    assert sum((label.startswith("logits-") for label in submitted)) == 2
    assert any((label.startswith("gate-final-") and label.endswith("-tp1") for label in submitted))
    records = [
        json.loads(line) for line in (spec.workdir / "perf_ledger.jsonl").read_text().splitlines()
    ]
    used = [
        r["sources"]
        for r in records
        if r["kind"] == "tuner_decision" and r.get("slug") == "fast-path"
    ]
    assert used and used[0][0]["source"] == "flashinfer"
    kl = [r for r in records if r["kind"] == "kernel_ledger"]
    assert kl and kl[0]["rows"] == 2 and (kl[0]["items_per_question"]["faster"] == 1)
    assert state["phase"] == "done"
    assert state["kept"] == ["fast-path"]
    assert set(turns) == {
        "sol-analytic",
        "sol-code",
        "sol-profile",
        "roadmap",
        "fast-path",
        "no-op",
        "roadmap-2",
    }
    assert set(turns[:3]) == {"sol-analytic", "sol-code", "sol-profile"}
    assert state["sol"]["method"] == "roofline"
    assert "Rank by headroom" in asks["roadmap"] and "## Speed of light" in asks["roadmap"]
    best = Path(state["best_tree"])
    assert (best / "modeling.py").read_text() == "FAST = True\n"
    summary = (spec.workdir / "summary.md").read_text()
    assert "t-final" in summary and "t-final % of SOL" in summary
    unmerged = (spec.workdir.parent / "UNMERGED_OPTIMIZATIONS.md").read_text()
    assert "## `fast-path`" in unmerged and "- accuracy gate:" in unmerged
    assert f"`{stage.branch_prefix}/best`" in unmerged
    assert {
        "baseline",
        "tuner_decision",
        "candidate_microbench",
        "candidate_kept",
        "checkpoint",
        "final_accuracy",
    } <= {r["kind"] for r in records}
    (ck,) = [r for r in records if r["kind"] == "checkpoint"]
    assert ck["verdict"] == "WIN" and ck["correctness"] == "PASS" and (ck["since"] == ["fast-path"])
    assert round(ck["projected"]["con1"], 4) == 0.0909 and round(ck["measured"]["con1"], 2) == 0.2
    again = TuneStage(
        spec,
        state,
        dispatch=dispatch,
        checkpoint=lambda: None,
        runner_factory=lambda spools, prefix: FakeRunner(spools, prefix, _effect),
        poll_s=0,
        log=lambda m: None,
    )
    assert again.run() == "tuned"


FOUND = {"what": "the tiled kernel loses at b1", "evidence": "micro A/B", "next": "keep b1 eager"}


def _slowed(config, tree):
    lat, thr = _effect(config, tree)
    if tree and Path(tree, "modeling.py").read_text().startswith("SLOW"):
        lat /= 1.25
    return (lat, thr)


def test_a_fixable_loss_is_revised_once_and_every_later_request_carries_what_was_found(tmp_path):
    repo = _repo(tmp_path)
    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump({"modeling_v2": "require", "max_batch_size": 128}))
    task = {
        "trtllm_repo_path": str(repo),
        "checkpoint_path": "/ckpt",
        **_contract(tmp_path),
        "perf": {
            "identity_configs": [str(identity)],
            "spools": ["s1"],
            "env_script": "/env.sh",
            "workload": {"isl": 8192, "osl": 1024, "concurrency": [1, 128]},
            "sol": {"enabled": False},
            "campaign": {"max_parallel_tuners": 1},
        },
    }
    spec = TuneSpec.from_task(
        task, tmp_path / "ws", target="tensorrt_llm/_torch/_experimental/modeling_v2/models/f/t/"
    )
    turns, asks = ([], {})

    def dispatch(role, what, prompt):
        turns.append(what)
        if what.startswith("roadmap"):
            asks[what] = (spec.workdir / f"{what}-request.md").read_text()
            items = [
                {"slug": "tiled", "kind": "kernel", "hypothesis": "tile the GEMM"},
                {"slug": "fused-norm", "kind": "fusion", "hypothesis": "fuse the norm"},
            ]
            (spec.workdir / f"{what}.yaml").write_text(
                yaml.safe_dump(items if what == "roadmap" else [])
            )
            return "roadmap"
        slug = what.removeprefix("revise-")
        out, wt = (spec.workdir / "candidates" / slug, spec.workdir / "worktrees" / slug)
        if slug == "fused-norm":
            asks[slug] = (out / "request.md").read_text()
            decision = {
                "decision": "DROPPED",
                "kind": "fusion",
                "worktree": str(wt),
                "findings": [FOUND],
            }
        else:
            asks[what] = (out / ("revise-request.md" if what != slug else "request.md")).read_text()
            (wt / "modeling.py").write_text("FAST = True\n" if what != slug else "SLOW = True\n")
            _git(wt, "commit", "-q", "-am", what)
            decision = {
                "decision": "CANDIDATE",
                "kind": "kernel",
                "worktree": str(wt),
                "findings": [FOUND],
                "commits": [_git(wt, "rev-parse", "HEAD")],
                "microbench": _microbench(out),
            }
        (out / "decision.yaml").write_text(yaml.safe_dump(decision))
        return "done"

    state: dict = {}
    stage = TuneStage(
        spec,
        state,
        dispatch=dispatch,
        checkpoint=lambda: None,
        poll_s=0,
        runner_factory=lambda spools, prefix: FakeRunner(spools, prefix, _slowed),
        log=lambda m: None,
    )
    assert stage.run() == "tuned"
    assert turns == ["roadmap", "tiled", "fused-norm", "revise-tiled", "roadmap-2"]
    assert state["kept"] == ["tiled"]
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


UNIFIED = {"parallel": "unified", "members": ["tp1", "tp2", "tep4", "dep4"]}


def _fast(tree) -> bool:
    return bool(tree) and Path(tree, "modeling.py").read_text().startswith("FAST = True")


def _campaign(tmp_path: Path, gates, msgs: list, effect=_effect, keep_gate=None, **kw) -> Campaign:
    """One candidate, 20 % faster in its micro-bench at half of each point's step."""
    repo = _repo(tmp_path)
    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump({"modeling_v2": "require"}))
    bench = BenchSpec(
        bench_py="b",
        model="m",
        env_script="e",
        isl=8192,
        osl=1024,
        concurrencies=[1, 128],
        tree=str(repo),
    )
    work = tmp_path / "camp"
    gate = keep_gate or replace(_keep_gate(tmp_path, **UNIFIED), tests=("tests/unittest/t",))
    camp = Campaign(
        workdir=work,
        repo=repo,
        bench=bench,
        base_configs=[identity],
        perf_runner=FakeRunner(["s1"], "camp", effect),
        gate_runner=gates,
        keep_gate=gate,
        noise=NoiseFloor(spread={"con1.tok_s_user": 0.001}),
        log=msgs.append,
        **kw,
    )
    camp.state.best_tree = str(open_candidate(repo, work / "wt", "best", "HEAD"))
    camp.state.best_commit = _git(Path(camp.state.best_tree), "rev-parse", "HEAD")
    wt = open_candidate(repo, work / "wt", "fast-path", "HEAD")
    (wt / "modeling.py").write_text("FAST = True\n")
    _git(wt, "commit", "-q", "-am", "fast")
    decision = tmp_path / "fast-path.yaml"
    decision.write_text(
        yaml.safe_dump(
            {
                "decision": "CANDIDATE",
                "worktree": str(wt),
                "microbench": _microbench(tmp_path / "microbench"),
                "commits": [_git(wt, "rev-parse", "HEAD")],
            }
        )
    )
    camp.submit_candidate("fast-path", decision)
    return camp


def _run(camp: Campaign, steps: int = 30) -> list[dict]:
    for _ in range(steps):
        if not camp.step():
            break
    ledger = camp.workdir / "perf_ledger.jsonl"
    return [json.loads(line) for line in ledger.read_text().splitlines() if '"checkpoint"' in line]


def test_a_control_that_cannot_answer_leaves_the_keeps_standing_until_more_arrive(tmp_path):
    def mute(tree):
        return {7: -0.1, 9: -2.5} if _fast(tree) else {}

    camp = _campaign(tmp_path, FakeRunner(["g1"], "campg", _effect, first_token=mute), [])
    (ck,) = _run(camp)
    assert ck["verdict"] == "ERROR" and camp.state.kept == ["fast-path"]
    assert ck["reason"].startswith("logits-") and "0 of 16 prompts answered" in ck["reason"]
    assert camp.state.checkpoint["retry_after"] == 4


def test_a_checkpoints_suite_job_holds_as_many_devices_as_the_widest_member(tmp_path):
    """The checkpoint's gate drops the members (accuracy runs on the final tree), not their width.

    The target's suite carries multi-rank tests; on one device they fail at every checkpoint
    and every keep since the last one is rolled back.
    """
    gates = FakeRunner(["g1"], "campg", _effect)
    asked: dict[str, str] = {}
    submit = gates.submit

    def recording(spool, label, command, **kw):
        asked[label] = kw.get("gpus", "all")
        return submit(spool, label, command, **kw)

    gates.submit = recording
    camp = _campaign(tmp_path, gates, [])
    assert camp.keep_gate.members == ("dep4",) and camp.keep_gate.suite_devices == 4
    suite = camp.keep_gate.tests_only()
    assert suite.members == () and not suite.ungated
    assert suite.suite_devices == 4 and suite.needs("tests")[0] == 4
    (ck,) = _run(camp)
    (label,) = [label for label in asked if label.startswith("gate-ckpt-")]
    assert label.endswith("-tests") and asked[label] == "4"
    assert ck["correctness"] == "PASS" and camp.state.kept == ["fast-path"]


def test_the_suite_width_leaves_the_campaigns_evidence_inputs_as_they_were(tmp_path):
    """The suite's width restates the members' width and changes nothing the gate judges.

    It stays out of the evidence inputs, so a campaign resumed across a change of that field
    keeps its controls and measurements instead of measuring them all again.
    """
    camp = _campaign(tmp_path, FakeRunner(["g1"], "campg", _effect), [])
    before = camp.state.inputs
    assert before
    camp.keep_gate = replace(camp.keep_gate, suite_width=camp.keep_gate.suite_devices)
    camp.refresh_inputs()
    assert camp.state.inputs == before


HOUR = 3600.0


class _Clock:
    """The wall clock the stage reads; a turn may move it on."""

    def __init__(self) -> None:
        self.now = 1000000.0

    def __call__(self) -> float:
        return self.now


def _budget_spec(tmp_path: Path, **campaign):
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
            # A deferral ends the run at once unless a test waits it out.
            "campaign": {
                "max_parallel_tuners": 1,
                "kernel_ledger": "off",
                "max_retries": 0,
                **campaign,
            },
        },
    }
    return TuneSpec.from_task(task, tmp_path / "ws", target="t/")


def _budget_effect(config, tree):
    text = Path(tree, "modeling.py").read_text() if tree else ""
    fast = 1.2 if "FAST = True" in text else 1.0
    side = 1.1 if "SIDE" in text else 1.0
    return (100.0 * fast * side, 2000.0)


def _stage(spec, state, dispatch, clock, logs):
    stage = TuneStage(
        spec,
        state,
        dispatch=dispatch,
        checkpoint=lambda: None,
        runner_factory=lambda spools, prefix: FakeRunner(spools, prefix, _budget_effect),
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
    spec = _budget_spec(tmp_path, max_hours=8)
    clock, turns, logs = (_Clock(), [], [])
    state = {"phase": "campaign", "started_at": clock.now - 9 * HOUR}
    stage = _stage(spec, state, lambda role, what, prompt: turns.append(what) or "", clock, logs)
    assert stage.run() == "tuned"
    assert turns == []
    assert state["ended"]["by"] == "budget" and state["kept"] == []
    assert state["budget_spent"]["after_h"] == 9.0
    kinds = [r["kind"] for r in _ledger(spec)]
    assert "budget_spent" in kinds and "campaign_end" in kinds
    ended = (
        "the wall budget of 8 h (`perf.campaign.max_hours`) ended the campaign, spent after 9.0 h"
    )
    assert stage._end_text() == ended
    assert f"Ended: {ended}." in (spec.workdir / "summary.md").read_text()
    assert any(("the wall budget of 8 h (perf.campaign.max_hours) is spent" in m for m in logs))
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
    spec = _budget_spec(tmp_path, max_hours=8)
    clock, turns, logs = (_Clock(), [], [])

    def dispatch(role, what, prompt):
        turns.append(what)
        if what == "roadmap":
            items = [{"slug": "fast-path"}, {"slug": "side-path"}, {"slug": "third-path"}]
            (spec.workdir / "roadmap.yaml").write_text(yaml.safe_dump(items))
        elif what == "fast-path":
            _candidate(spec, stage, what, "FAST = True\n")
        elif what == "side-path":
            _candidate(spec, stage, what, "FAST = False\nSIDE = 1\n")
            clock.now += 9 * HOUR
        return ""

    state = {"phase": "campaign"}
    stage = _stage(spec, state, dispatch, clock, logs)
    assert stage.run() == "tuned"
    assert turns == ["roadmap", "fast-path", "side-path"]
    assert state["kept"] == ["fast-path"] and state["ended"]["by"] == "budget"
    candidates = json.loads((spec.workdir / "campaign_state.json").read_text())["candidates"]
    assert candidates["side-path"]["status"] == "conflict"
    assert any(("side-path" in m and "no rebase is asked for" in m for m in logs))
    assert Path(state["best_tree"], "modeling.py").read_text() == "FAST = True\n"


def _split_effect(config, tree):
    """Con 1 is faster with FAST; con 128 is slower with SLOW_HI unless GATED."""
    text = Path(tree, "modeling.py").read_text() if tree else ""
    lat = 100.0 * (1.2 if "FAST = True" in text else 1.0)
    thr = 2000.0 * (0.9 if "SLOW_HI" in text and "GATED" not in text else 1.0)
    return (lat, thr)


def _split_campaign(tmp_path: Path, variant_text: str):
    spec = _budget_spec(tmp_path, max_roadmaps=1)
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
    stage = TuneStage(
        spec,
        state,
        dispatch=dispatch,
        checkpoint=lambda: None,
        runner_factory=lambda spools, prefix: FakeRunner(spools, prefix, _split_effect),
        poll_s=0,
        log=log.append,
    )
    assert stage.ground()
    assert stage.run() == "tuned"
    camp = json.loads((spec.workdir / "campaign_state.json").read_text())["candidates"]
    kinds = [json.loads(line)["kind"] for line in (spec.workdir / "perf_ledger.jsonl").open()]
    return (spec, state, turns, camp, kinds, log)


def test_a_split_candidate_gets_one_member_gated_variant_that_is_kept(tmp_path):
    spec, state, turns, camp, kinds, log = _split_campaign(tmp_path, "GATED = True\n")
    assert turns == ["roadmap", "fast-path", "fast-path-gated"]
    assert camp["fast-path"]["status"] == "split"
    assert (
        camp["fast-path"]["split"]["mode"]
        == "high concurrency (>= 128 requests per rank in a step)"
    )
    assert camp["fast-path-gated"]["variant_of"] == "fast-path"
    assert state["kept"] == ["fast-path-gated"]
    best = Path(state["best_tree"], "modeling.py").read_text()
    assert "FAST = True" in best and "GATED = True" in best
    assert {"candidate_split", "split_variant", "variant_verdict", "candidate_kept"} <= set(kinds)
    assert any(("fast-path: split: loses only at con128" in m for m in log)), log
    request = (spec.workdir / "candidates" / "fast-path-gated" / "request.md").read_text()
    assert "the requests this rank runs in the step >= 128" in request
    assert str(spec.workdir / "worktrees" / "fast-path-gated") in request
    assert f"--priority {TUNER}" in request
