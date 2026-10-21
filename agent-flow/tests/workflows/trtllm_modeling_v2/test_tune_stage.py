# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tune stage end to end on CPU: fake spool runners, a fake tuner, a real git repo."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneSpec, TuneStage


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

    def __init__(self, spools, prefix, effect, score=None):
        self.spools, self.prefix, self.effect = list(spools), prefix, effect
        self.score = score or (lambda label, tree: 96.0)
        self.inflight: dict[str, tuple[str, str]] = {}
        self.jobs: dict[str, tuple[str, str | None]] = {}
        self.outputs: dict[str, str] = {}

    def adopt(self, label):
        return False

    def place(self, gpus, wall="1h"):
        load = {s: sum(1 for sp, _ in self.inflight.values() if sp == s) for s in self.spools}
        return min(self.spools, key=lambda s: load[s])

    def widest(self):
        return 0

    def output(self, label):
        return self.outputs.get(label, ""), {}

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
            tokens = command.split()
            if "trtllm-eval " in command:
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
                status = Path(spool, "status.json")
                host = json.loads(status.read_text())["host"] if status.exists() else ""
                (d / "meta.json").write_text(
                    json.dumps(
                        {
                            "label": d.name,
                            "modeling_v2": merged.get("modeling_v2", "require"),
                            "tree": {},
                            "hostname": host,
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


def _effect(config, tree):
    lat, thr = 100.0, 2000.0
    if config.get("modeling_v2") == "off":
        lat, thr = 120.0, 2300.0
    if (config.get("cuda_graph_config") or {}).get("enable_padding"):
        thr *= 1.05
    if tree and Path(tree, "modeling.py").read_text().startswith("FAST = True"):
        lat *= 1.2
    return lat, thr


def _write_sol(workdir: Path, value: float = 1e9, method: str = "calculator") -> None:
    """What a SOL turn writes: ceilings far above anything the fake runner measures."""
    peaks = workdir / "sol" / "peaks.json"
    peaks.parent.mkdir(parents=True, exist_ok=True)
    peaks.write_text("{}")
    keys = ("con1.tok_s_user", "con128.output_tok_s_gpu", "con128.tok_s_user")
    (workdir / "sol.yaml").write_text(
        yaml.safe_dump(
            {
                "method": method,
                "peaks": str(peaks),
                "ceilings": {
                    k: {"value": value, "bound": "memory", "derivation": "bytes / bw"} for k in keys
                },
            }
        )
    )


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
        if what == "sol":
            _write_sol(spec.workdir)
            return "ceilings written"
        if what == "roadmap":
            asks["roadmap"] = (spec.workdir / "roadmap-request.md").read_text()
        if what == "roadmap-2":
            (spec.workdir / "roadmap-2.yaml").write_text("[]")
            return "nothing left"
        if what == "roadmap":
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
    stage = TuneStage(
        spec,
        state,
        dispatch=dispatch,
        checkpoint=lambda: None,
        runner_factory=lambda spools, prefix: FakeRunner(spools, prefix, _effect),
        poll_s=0,
        log=lambda m: None,
    )
    assert stage.run() == "tuned"
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
        "sol",
        "roadmap",
        "fast-path",
        "no-op",
        "roadmap-2",
    }  # the spent roadmap is refreshed once
    assert turns[0] == "sol" and state["sol"]["method"] == "calculator"
    assert "Rank by headroom" in asks["roadmap"] and "## Speed of light" in asks["roadmap"]
    best = Path(state["best_tree"])
    assert (best / "modeling.py").read_text() == "FAST = True\n"
    summary = (spec.workdir / "summary.md").read_text()
    assert "t-final" in summary and "t-final % of SOL" in summary
    # What the campaign kept is listed for a human: the delivered target does not carry it.
    unmerged = (spec.workdir.parent / "UNMERGED_OPTIMIZATIONS.md").read_text()
    assert "## `fast-path`" in unmerged and "- accuracy gate:" in unmerged
    assert f"`{stage.branch_prefix}/best`" in unmerged
    kinds = [
        json.loads(line)["kind"]
        for line in (spec.workdir / "perf_ledger.jsonl").read_text().splitlines()
    ]
    assert {
        "noise_floor",
        "tuner_decision",
        "candidate_e2e",
        "candidate_gate",
        "candidate_kept",
    } <= set(kinds)
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
