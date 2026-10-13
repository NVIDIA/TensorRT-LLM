# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Modeling iterations: the orchestrator half of a tuner candidate.

The tuner claims (``decision.yaml``: CANDIDATE + micro-bench evidence); this
module verifies and acts, the same split the gates have -- execution may move
to an agent, adjudication stays in Python:

1. **e2e**: a whole-node bench job against the candidate's worktree
   (``#spool: tree=<worktree>``, so ``import tensorrt_llm`` resolves there),
   at the campaign's current config.
2. **judge**: :func:`records.compare` against the control measured at the same
   config on the current best tree. WIN / LOSS / NEUTRAL.
3. **gate**: on WIN, ``gate_accuracy.py`` (gsm8k, ``modeling_v2: require``
   forced) against the same worktree and config.
4. **keep**: the candidate's commits are cherry-picked onto ``tune/best``;
   later candidates and the control are measured on that tree. A conflict
   is reported, never resolved silently.

State lives in ``<workdir>/campaign_state.json``; every measurement and verdict
is appended to ``<workdir>/perf_ledger.jsonl``.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from .ladder import BenchSpec, SpoolRunner
from .records import Measurement, NoiseFloor, PerfLedger, compare

TUNER_DECISIONS = ("CANDIDATE", "DROPPED", "NEED_SMITH", "NEED_RUNTIME", "NEED_REBUILD", "BLOCKED", "STOP")


@dataclass
class CandidateState:
    slug: str
    status: str = "issued"  # issued | e2e | gate | kept | reverted | dropped | blocked | conflict
    decision: str = ""
    worktree: str = ""
    commits: list[str] = field(default_factory=list)
    e2e_label: str = ""
    control_label: str = ""
    verdict: str = ""
    reason: str = ""
    deltas: dict = field(default_factory=dict)
    gate: dict = field(default_factory=dict)
    kept_commits: list[str] = field(default_factory=list)


@dataclass
class CampaignState:
    best_tree: str = ""
    best_commit: str = ""
    config_overlays: list[str] = field(default_factory=list)
    control_label: str = ""
    candidates: dict[str, CandidateState] = field(default_factory=dict)
    kept: list[str] = field(default_factory=list)
    # Same-host controls, keyed "<best commit>|<overlays>|<host>": hosts differ by
    # more than most candidates move, so a candidate is judged on its own host.
    controls: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "CampaignState":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        st = cls(**{k: v for k, v in data.items() if k != "candidates"})
        st.candidates = {k: CandidateState(**v) for k, v in data.get("candidates", {}).items()}
        return st

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2) + "\n")
        tmp.replace(path)


def read_decision(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    decision = str(data.get("decision", "")).strip().upper()
    if decision not in TUNER_DECISIONS:
        raise ValueError(f"{path}: decision {decision!r} is not one of {TUNER_DECISIONS}")
    data["decision"] = decision
    return data


def git(tree: Path | str, *args: str, check: bool = True) -> str:
    out = subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True)
    if check and out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {out.stderr.strip()}")
    return out.stdout.strip()


class Campaign:
    def __init__(
        self,
        *,
        workdir: Path,
        repo: Path,
        bench: BenchSpec,
        base_configs: Sequence[Path],
        perf_runner: SpoolRunner,
        gate_runner: SpoolRunner,
        gate_cmd: Callable[[Path, Sequence[Path], Path], str],
        noise: NoiseFloor,
        min_gain: float = 0.01,
        noise_factor: float = 3.0,
        log: Callable[[str], None] = print,
    ) -> None:
        self.workdir = workdir
        self.repo = repo
        self.bench = bench
        self.base_configs = list(base_configs)
        self.perf = perf_runner
        self.gates = gate_runner
        self.gate_cmd = gate_cmd
        self.noise = noise
        self.min_gain = min_gain
        self.noise_factor = noise_factor
        self.log = log
        self.data = workdir / "data"
        self.data.mkdir(parents=True, exist_ok=True)
        self.state_path = workdir / "campaign_state.json"
        self.ledger = PerfLedger(workdir / "perf_ledger.jsonl")
        self.state = CampaignState.load(self.state_path)
        for st in self.state.candidates.values():
            for label in (st.e2e_label,) if st.status == "e2e" else ():
                if label and not (self.data / label / "meta.json").exists():
                    self.perf.adopt(label)
            if st.status == "gate":
                self.gates.adopt(f"gate-{st.slug}")

    # -- configuration ---------------------------------------------------------------
    def configs(self) -> list[Path]:
        return self.base_configs + [Path(p) for p in self.state.config_overlays]

    def _measurement(self, label: str) -> Measurement | None:
        path = self.data / label
        if (path / "summary.json").exists() and (path / "meta.json").exists():
            return Measurement.load(path)
        return None

    def _control_key(self, host: str) -> str:
        return f"{self.state.best_commit[:10]}|{','.join(self.state.config_overlays)}|{host}"

    @staticmethod
    def spool_host(spool: str) -> str:
        try:
            return str(json.loads((Path(spool) / "status.json").read_text()).get("host") or "")
        except (OSError, ValueError):
            return ""

    def _control_for(self, host: str) -> Measurement | None:
        label = self.state.controls.get(self._control_key(host))
        m = self._measurement(label) if label else None
        if m is None and self.state.control_label:
            fallback = self._measurement(self.state.control_label)
            if fallback is not None and fallback.meta.get("hostname") == host:
                return fallback
        return m

    def _ensure_control(self, spool: str) -> bool:
        """True when ``spool``'s host has a control for the current best tree and config;
        otherwise queue one there (and return False)."""
        host = self.spool_host(spool)
        if self._control_for(host) is not None:
            return True
        key = self._control_key(host)
        label = f"ctl-{self.state.best_commit[:10]}-{host or Path(spool).name}"
        self.state.controls.setdefault(key, label)
        if f"{self.perf.prefix}-{label}" not in self.perf.inflight and not (self.data / label / "meta.json").exists():
            cmd = self.bench.command(label, self.data, self.configs())
            self.perf.submit(spool, label, cmd, wall=self.bench.wall, tree=self.state.best_tree)
            self.log(f"[campaign] control {label} queued on {spool}")
        return False

    def set_control(self, label_dir: Path) -> None:
        """Adopt an existing measurement of the best tree at the current config as the control."""
        link = self.data / label_dir.name
        if not link.exists():
            link.symlink_to(label_dir.resolve(), target_is_directory=True)
        self.state.control_label = label_dir.name
        self.state.save(self.state_path)

    # -- the loop ----------------------------------------------------------------------
    def submit_candidate(self, slug: str, decision_path: Path) -> CandidateState:
        st = self.state.candidates.setdefault(slug, CandidateState(slug))
        data = read_decision(decision_path)
        st.decision = data["decision"]
        st.worktree = str(data.get("worktree") or "")
        st.commits = [str(c) for c in data.get("commits") or []]
        self.ledger.append("tuner_decision", slug=slug, decision=st.decision, candidate_kind=data.get("kind"),
                           hypothesis=data.get("hypothesis"), microbench=data.get("microbench"),
                           projection=data.get("projection"))
        if st.decision != "CANDIDATE":
            st.status = {"DROPPED": "dropped", "BLOCKED": "blocked"}.get(st.decision, st.decision.lower())
            self.state.save(self.state_path)
            return st
        tree = Path(st.worktree)
        if git(tree, "status", "--porcelain", "-uno"):
            st.status, st.reason = "blocked", "worktree has uncommitted changes; the candidate is its commits"
            self.state.save(self.state_path)
            return st
        st.e2e_label = f"e2e-{slug}-{git(tree, 'rev-parse', '--short=10', 'HEAD')}"
        st.control_label = self.state.control_label
        st.status = "e2e"
        self.state.save(self.state_path)
        return st

    def step(self) -> bool:
        """Advance every candidate as far as possible; True while any is in flight."""
        finished = {label for label, _, _ in self.perf.poll()} | {label for label, _, _ in self.gates.poll()}
        busy = False
        for st in self.state.candidates.values():
            if st.status == "e2e":
                busy = True
                m = self._measurement(st.e2e_label)
                if m is None and f"{self.perf.prefix}-{st.e2e_label}" not in self.perf.inflight:
                    if st.e2e_label in finished:
                        st.status, st.reason = "reverted", "e2e measurement failed (see the label's .partial dir)"
                        continue
                    for spool in self.perf.free_spools():
                        if not self._ensure_control(spool):
                            continue
                        cmd = self.bench.command(st.e2e_label, self.data, self.configs())
                        self.perf.submit(spool, st.e2e_label, cmd, wall=self.bench.wall, tree=st.worktree)
                        self.log(f"[campaign] e2e {st.slug} queued on {spool}")
                        break
                    continue
                if m is None:
                    continue
                ctrl = self._control_for(str(m.meta.get("hostname") or ""))
                if ctrl is None:
                    ctrl = self._measurement(st.control_label or self.state.control_label)
                st.control_label = ctrl.label if ctrl else ""
                self.ledger.measurement(m, candidate=st.slug)
                cmp = compare(m, ctrl, self.noise, min_gain=self.min_gain, noise_factor=self.noise_factor)
                st.verdict, st.reason, st.deltas = cmp.verdict, cmp.reason, dict(cmp.deltas)
                self.ledger.append("candidate_e2e", slug=st.slug, **cmp.as_dict())
                self.log(f"[campaign] {st.slug}: {cmp.verdict} {cmp.reason}")
                st.status = "gate" if cmp.verdict == "WIN" else "reverted"
            if st.status == "gate":
                busy = True
                out = self.workdir / "gates" / st.slug
                result = out / "result.json"
                job = f"gate-{st.slug}"
                if not result.exists() and f"{self.gates.prefix}-{job}" not in self.gates.inflight:
                    if job in finished:
                        st.status, st.reason = "reverted", f"accuracy gate did not produce a result; see {out}"
                        continue
                    free = self.gates.free_spools()
                    if free:
                        self.gates.submit(free[0], job, self.gate_cmd(Path(st.worktree), self.configs(), out),
                                          wall="1h", tree=st.worktree, gpus="1")
                        self.log(f"[campaign] gate {st.slug} queued on {free[0]}")
                    continue
                if not result.exists():
                    continue
                st.gate = json.loads(result.read_text())
                self.ledger.append("candidate_gate", slug=st.slug, **st.gate)
                if st.gate.get("verdict") == "PASS":
                    self._keep(st)
                else:
                    st.status, st.reason = "reverted", f"accuracy {st.gate.get('score')} < floor {st.gate.get('floor')}"
        self.state.save(self.state_path)
        return busy

    def _keep(self, st: CandidateState) -> None:
        best = Path(self.state.best_tree)
        before = git(best, "rev-parse", "HEAD")
        out = subprocess.run(["git", "-C", str(best), "cherry-pick", *st.commits], capture_output=True, text=True)
        if out.returncode != 0:
            git(best, "cherry-pick", "--abort", check=False)
            st.status, st.reason = "conflict", f"cherry-pick onto tune/best failed: {out.stderr.strip()[:300]}"
            self.ledger.append("candidate_conflict", slug=st.slug, detail=st.reason)
            return
        st.kept_commits = git(best, "rev-list", "--reverse", f"{before}..HEAD").splitlines()
        st.status = "kept"
        self.state.best_commit = git(best, "rev-parse", "HEAD")
        self.state.kept.append(st.slug)
        self.ledger.append("candidate_kept", slug=st.slug, commits=st.kept_commits, best=self.state.best_commit,
                           deltas=st.deltas, gate=st.gate)
        self.log(f"[campaign] KEEP {st.slug} -> tune/best {self.state.best_commit[:10]}")

    def summary(self) -> str:
        lines = [f"best={self.state.best_commit[:10]} control={self.state.control_label} kept={self.state.kept}"]
        for st in self.state.candidates.values():
            d = ", ".join(f"{k} {v:+.1%}" for k, v in st.deltas.items())
            g = f" gate={st.gate.get('score')}" if st.gate else ""
            lines.append(f"  {st.slug:24s} {st.status:9s} {st.decision:10s} {st.verdict:7s} {d}{g} {st.reason[:80]}")
        return "\n".join(lines)


def gate_command(package: Path, env_script: Path, model: str, anchor: float, task: str = "gsm8k"):
    def build(tree: Path, configs: Sequence[Path], out: Path) -> str:
        argv = ["python", str(package / "tune" / "gate_accuracy.py"), "--model", model, "--out", str(out),
                "--task", task, "--anchor", str(anchor)]
        for c in configs:
            argv += ["--config", str(c)]
        return f"source {shlex.quote(str(env_script))}\ncd {shlex.quote(str(tree))}\n" + " ".join(shlex.quote(a) for a in argv)

    return build
