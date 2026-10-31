# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tune stage end to end on CPU: fake spool runners, a fake tuner, a real git repo."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune.spec import TuneSpec
from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneStage


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
        self.spools, self.prefix, self.effect = list(spools), prefix, effect
        self.score = score or (lambda label, tree: 96.0)
        #: A tree's first-token log-probs, the same on every tree unless a test says otherwise.
        self.first_token = first_token or (lambda tree: {7: -0.1, 9: -2.5})
        self.inflight: dict[str, tuple[str, str]] = {}
        self.jobs: dict[str, tuple[str, str | None]] = {}
        self.outputs: dict[str, str] = {}
        self.receipts: dict[str, dict] = {}
        self.submitted: list[str] = []

    def adopt(self, label):
        return False

    def place(self, gpus, wall="1h"):
        load = {s: sum(1 for sp, _ in self.inflight.values() if sp == s) for s in self.spools}
        return min(self.spools, key=lambda s: load[s])

    def widest(self):
        return 0

    def output(self, label):
        return self.outputs.get(label, ""), self.receipts.get(label, {})

    def queued(self, label):
        return any(queued == label for _, queued in self.inflight.values())

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
                        }
                    )
                )
            elif "engine_driver.py" in command:
                spec = json.loads(Path(tokens[tokens.index("--spec") + 1]).read_text())
                row = [[tok, lp, None] for tok, lp in self.first_token(tree).items()]
                self.outputs[label] = "".join(
                    f"MV2FLOW_RECORD {json.dumps({'index': i, 'id': r['id'], 'top_logprobs': [row]})}\n"
                    for i, r in enumerate(spec["requests"])
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
    lat, thr = 100.0, 2000.0
    if config.get("modeling_v2") == "off":
        lat, thr = 120.0, 2300.0
    if (config.get("cuda_graph_config") or {}).get("enable_padding"):
        thr *= 1.05
    if tree and Path(tree, "modeling.py").read_text().startswith("FAST = True"):
        lat *= 1.2
    return lat, thr


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
        if what.startswith("sol-"):  # three analyses that agree: no adjudication
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
    assert not any(label.startswith(("t-baseline-r", "e2e-")) for label in submitted)
    assert len([label for label in submitted if label.startswith("micro-")]) == 2
    # The keep projected past the checkpoint's gain: e2e against the tree before it, its suite,
    # the first tokens of both; GSM8K once, on the final tree, at one member.
    assert sorted(label.split("-")[0] for label in submitted[4:-1]) == [
        "ckpt", "ctl", "gate", "logits", "logits"
    ]  # fmt: skip
    assert submitted[-1].startswith("gate-final-") and submitted[-1].endswith("-tp1")
    records = [
        json.loads(line) for line in (spec.workdir / "perf_ledger.jsonl").read_text().splitlines()
    ]
    used = [
        r["sources"]
        for r in records
        if r["kind"] == "tuner_decision" and r.get("slug") == "fast-path"
    ]
    assert (
        used and used[0][0]["source"] == "flashinfer"
    )  # the ledger keeps where each candidate came from
    kl = [r for r in records if r["kind"] == "kernel_ledger"]
    assert (
        kl and kl[0]["rows"] == 2 and kl[0]["items_per_question"]["faster"] == 1
    )  # perf_optimize's ledger, validated
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
    }  # the spent roadmap is refreshed once
    assert set(turns[:3]) == {"sol-analytic", "sol-code", "sol-profile"}
    assert state["sol"]["method"] == "roofline"
    assert "Rank by headroom" in asks["roadmap"] and "## Speed of light" in asks["roadmap"]
    best = Path(state["best_tree"])
    assert (best / "modeling.py").read_text() == "FAST = True\n"
    summary = (spec.workdir / "summary.md").read_text()
    assert "t-final" in summary and "t-final % of SOL" in summary
    # What the campaign kept is listed for a human: the delivered target does not carry it.
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
    assert ck["verdict"] == "WIN" and ck["correctness"] == "PASS" and ck["since"] == ["fast-path"]
    assert round(ck["projected"]["con1"], 4) == 0.0909 and round(ck["measured"]["con1"], 2) == 0.2
    # resume after completion re-measures nothing
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
