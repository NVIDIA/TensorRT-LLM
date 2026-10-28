# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One measured A/B verification, then the correctness keep gate.

Isolated changes use a micro/module harness; system-wide changes use serving.
Verification and keeps are serialized while tuner turns remain parallel, avoiding
stale-stack remeasurements. The stage validates cumulative e2e at wrap-up.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import yaml

from ..agents.decisions import names as decision_names
from ..infra.compute.spoolctl import SpoolError
from ..infra.git import add_worktree, git_checked, link_native
from .harness import microbench
from .inert import inert_members
from .keep_gate import ERROR, FAIL, KEEP_FILE, PART_FILE, PASS, KeepGate
from .measure import BenchSpec, SpoolRunner, load_measurement, spool_host, validate_measurement
from .records import TTFT_TOLERANCE, Comparison, Measurement, NoiseFloor, PerfLedger, compare
from .split import MIXED, SPLIT, classify, judge_variant

TUNER_DECISIONS = decision_names("tuner")
SYSTEM_KINDS = {"host", "overlap", "serving", "communication", "collective"}


@dataclass
class CandidateState:
    slug: str
    status: str = "issued"
    decision: str = ""
    worktree: str = ""
    commits: list[str] = field(default_factory=list)
    scope: str = "micro"
    microbench: dict = field(default_factory=dict)
    e2e_label: str = ""
    control_label: str = ""
    verdict: str = ""
    reason: str = ""
    deltas: dict = field(default_factory=dict)
    gate: dict = field(default_factory=dict)
    kept_commits: list[str] = field(default_factory=list)
    eval_tree: str = ""
    eval_base: str = ""
    rebase_ref: str = ""
    split: dict = field(default_factory=dict)
    variant_of: str = ""
    changed_members: list[str] = field(default_factory=list)
    path_marker: str = ""
    controls: list[str] = field(default_factory=list)
    inert: dict = field(default_factory=dict)


@dataclass
class CampaignState:
    best_tree: str = ""
    best_commit: str = ""
    control_label: str = ""
    candidates: dict[str, CandidateState] = field(default_factory=dict)
    kept: list[str] = field(default_factory=list)
    # One reusable serving control per best revision and host.
    controls: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "CampaignState":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        st = cls(
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "candidates"}
        )
        for key, value in data.get("candidates", {}).items():
            candidate = CandidateState(
                **{k: v for k, v in value.items() if k in CandidateState.__dataclass_fields__}
            )
            if "scope" not in value:
                candidate.scope = "e2e"
                if candidate.status in ("e2e", "gate", "won", "waiting"):
                    candidate.status = "ready"
            st.candidates[key] = candidate
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
        keep_gate: KeepGate,
        noise: NoiseFloor,
        noise_commit: str | None = None,
        min_gain: float = 0.01,
        noise_factor: float = 3.0,
        ttft_tolerance: float | None = TTFT_TOLERANCE,
        branch_prefix: str = "tune",
        log: Callable[[str], None] = print,
        split_variants: bool = True,
    ) -> None:
        self.workdir, self.repo, self.bp = workdir, repo, branch_prefix
        self.bench, self.base_configs = bench, list(base_configs)
        self.perf, self.gates, self.keep_gate = perf_runner, gate_runner, keep_gate
        self.noise, self.noise_commit = noise, noise_commit
        self.min_gain, self.noise_factor, self.ttft_tolerance = (
            min_gain,
            noise_factor,
            ttft_tolerance,
        )
        self.split_variants, self.log = split_variants, log
        self.data = workdir / "data"
        self.data.mkdir(parents=True, exist_ok=True)
        self.state_path = workdir / "campaign_state.json"
        self.ledger = PerfLedger(workdir / "perf_ledger.jsonl")
        self.state = CampaignState.load(self.state_path)
        self._gate_done: dict[str, int] = {}
        for st in self.state.candidates.values():
            if st.status == "e2e":
                self.perf.adopt(st.e2e_label)
                if st.control_label:
                    self.perf.adopt(st.control_label)
            elif st.status == "micro":
                for side in ("control", "candidate"):
                    self.perf.adopt(self._micro_label(st, side))
            elif st.status == "gate" and st.eval_tree:
                self._adopt_gate(self._gate_key(st))

    def _control_key(self, host: str) -> str:
        return f"{self.state.best_commit}|{host}"

    def _floor(self) -> NoiseFloor:
        if self.noise_commit is None or self.state.best_commit.startswith(self.noise_commit[:10]):
            return self.noise
        return NoiseFloor()

    def set_control(self, label_dir: Path) -> None:
        m = load_measurement(self.data, label_dir.name)
        if m is not None:
            self.state.controls[self._control_key(str(m.meta.get("hostname") or ""))] = (
                label_dir.name
            )
            self.state.control_label = label_dir.name
        self.state.save(self.state_path)

    def submit_candidate(
        self, slug: str, decision_path: Path, *, data: dict | None = None
    ) -> CandidateState:
        st = self.state.candidates.setdefault(slug, CandidateState(slug))
        data = read_decision(decision_path) if data is None else data
        st.decision = data["decision"]
        st.worktree = str(data.get("worktree") or "")
        st.commits = [str(c) for c in data.get("commits") or []]
        st.changed_members = [str(m) for m in data.get("changed_members") or []]
        st.path_marker = str(data.get("path_marker") or "").strip()
        st.scope = str(data.get("perf_scope") or "micro")
        st.microbench = dict(data.get("microbench") or {})
        self.ledger.append(
            "tuner_decision",
            slug=slug,
            decision=st.decision,
            candidate_kind=data.get("kind"),
            hypothesis=data.get("hypothesis"),
            microbench=st.microbench,
            projection=data.get("projection"),
            sources=data.get("sources"),
            perf_scope=st.scope,
        )
        if st.decision != "CANDIDATE":
            st.status = {"DROPPED": "dropped", "BLOCKED": "blocked"}[st.decision]
        else:
            tree = Path(st.worktree)
            changed = git_checked(
                tree, "diff", "--name-only", f"{self._best_head()}...HEAD"
            ).splitlines()
            system = data.get("kind") in SYSTEM_KINDS or any(
                path.endswith((".yaml", ".yml"))
                or any(
                    part in path.split("/")
                    for part in ("pyexecutor", "runtime", "serve", "distributed")
                )
                for path in changed
            )
            if st.scope not in ("micro", "module", "e2e"):
                st.status, st.reason = "blocked", "perf_scope must be micro, module or e2e"
            elif system and st.scope != "e2e":
                st.status, st.reason = (
                    "blocked",
                    "serving, host, overlap or communication changes require perf_scope: e2e",
                )
            elif git_checked(tree, "status", "--porcelain", "-uno") or not st.commits:
                st.status, st.reason = "blocked", "candidate must name commits in a clean worktree"
            elif st.scope != "e2e":
                try:
                    script = Path(st.microbench["script"])
                    cases = st.microbench["cases"]
                    if (
                        not script.is_absolute()
                        or not isinstance(cases, list)
                        or not cases
                        or any(not isinstance(c, str) or not c.strip() for c in cases)
                        or len(set(cases)) != len(cases)
                    ):
                        raise ValueError(
                            "microbench requires an absolute script and unique nonempty case names"
                        )
                    st.microbench["sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
                    st.status = "ready"
                except (KeyError, OSError, TypeError, ValueError) as exc:
                    st.status, st.reason = "blocked", f"no runnable micro/module benchmark: {exc}"
            else:
                st.status = "ready"
        self.state.save(self.state_path)
        return st

    def _stack(self, st: CandidateState) -> bool:
        """Build the current best with all pending candidate commits, before verification."""
        head = self._best_head()
        pending = git_checked(
            Path(st.worktree),
            "rev-list",
            "--reverse",
            "--cherry-pick",
            "--right-only",
            "--no-merges",
            f"{head}...HEAD",
        ).splitlines()
        if not pending:
            st.status, st.reason = "dropped", "the best tree already carries this candidate"
            return False
        st.commits = pending
        tree = Path(self.state.best_tree).parent / f"eval-{st.slug}"
        branch = f"{self.bp}/eval-{st.slug}"
        if (tree / ".git").exists():
            git_checked(tree, "cherry-pick", "--abort", check=False)
            git_checked(tree, "checkout", "-q", "-f", "-B", branch, head)
            git_checked(tree, "reset", "-q", "--hard", head)
        else:
            add_worktree(self.repo, tree, "-B", branch, str(tree), head)
        link_native(self.repo, tree)
        out = subprocess.run(
            ["git", "-C", str(tree), "cherry-pick", *st.commits], capture_output=True, text=True
        )
        if out.returncode != 0:
            git_checked(tree, "cherry-pick", "--abort", check=False)
            st.status, st.reason = (
                "conflict",
                f"does not stack on tune/best {head[:10]}: {out.stderr.strip()[:300]}",
            )
            st.rebase_ref = head
            self.ledger.append("candidate_conflict", slug=st.slug, best=head, detail=st.reason)
            return False
        st.eval_tree, st.eval_base = str(tree), head
        st.e2e_label = f"e2e-{self._gate_key(st)}"
        return True

    def _gate_key(self, st: CandidateState) -> str:
        return f"{st.slug}-{git_checked(Path(st.eval_tree), 'rev-parse', '--short=10', 'HEAD')}"

    def step(self) -> bool:
        try:
            return self._step()
        except OSError as exc:
            if exc.errno not in (28, 122):
                raise
            self.log(f"[campaign] {exc.strerror}; retrying on the next poll")
            return True
        except SpoolError as exc:
            self.log(f"[campaign] {exc}; retrying on the next poll")
            return True

    def _gate_dir(self, key: str) -> Path:
        return self.workdir / "gates" / key

    def _adopt_gate(self, key: str) -> None:
        for part in self.keep_gate.parts():
            if not (self._gate_dir(key) / part / PART_FILE).exists():
                self.gates.adopt(f"gate-{key}-{part}")

    def _cancel_gate(self, key: str) -> bool:
        return any([self.gates.cancel(f"gate-{key}-{part}") for part in self.keep_gate.parts()])

    @staticmethod
    def _write(path: Path, data: dict) -> dict:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str) + "\n")
        return data

    def _requeue(self, key: str, where: Path, verdict: dict) -> bool:
        """Retry a correctness gate with no verdict, never a measured performance result."""
        attempts = len(list(where.glob("attempt-*.json")))
        if attempts >= (2 if verdict.get("infrastructure") else 1):
            return False
        self._write(where / f"attempt-{attempts + 1}.json", verdict)
        part = verdict.get("part")
        self.ledger.append(
            "retry", slug=key, what=f"gate {part}", rc=verdict.get("rc"), attempt=attempts + 1
        )
        return True

    def _gate(self, key: str, tree: Path) -> dict | None:
        """Run every correctness part on the stacked tree; only all PASS permits a keep."""
        out = self._gate_dir(key)
        if (out / KEEP_FILE).exists():
            return json.loads((out / KEEP_FILE).read_text())
        gate = self.keep_gate
        parts: dict[str, dict] = {}
        todo: list[str] = []
        running = False
        for part in gate.parts():
            where, job = out / part, f"gate-{key}-{part}"
            if (where / PART_FILE).exists():
                parts[part] = json.loads((where / PART_FILE).read_text())
            elif self.gates.queued(job):
                running = True
            elif job in self._gate_done:
                output, record = self.gates.output(job)
                rc = self._gate_done.pop(job)
                verdict = gate.judge(part, tree, where, rc=rc, output=output, record=record)
                if verdict["verdict"] in (PASS, FAIL) or not self._requeue(key, where, verdict):
                    parts[part] = self._write(where / PART_FILE, verdict)
                else:
                    todo.append(part)
            else:
                todo.append(part)
        failed = any(v.get("verdict") == FAIL for v in parts.values())
        for part in [] if failed else todo:
            where, job = out / part, f"gate-{key}-{part}"
            gpus, wall = gate.needs(part)
            spool = self.gates.place(gpus, wall)
            if spool is None:
                widest = self.gates.widest()
                if gpus > widest > 0:
                    parts[part] = self._write(
                        where / PART_FILE,
                        {
                            "verdict": ERROR,
                            "part": part,
                            "detail": f"needs {gpus} GPUs; the widest dev session hands out {widest}",
                        },
                    )
                else:
                    running = True
                continue
            prepared = gate.prepare(part, tree, where)
            if isinstance(prepared, dict):
                parts[part] = self._write(where / PART_FILE, prepared)
                if prepared.get("verdict") == FAIL:
                    failed = True
                    break
                continue
            self.gates.submit(
                spool,
                job,
                prepared.script,
                wall=prepared.wall,
                tree=str(tree),
                gpus=str(prepared.gpus),
            )
            self.log(f"[campaign] gate {key}: {part} queued on {spool}")
            running = True
        if running and not failed:
            return None
        if running:
            self._cancel_gate(key)
        verdict = self._write(out / KEEP_FILE, gate.verdict(parts, tree))
        self.log(f"[campaign] gate {key}: {verdict['verdict']} {verdict['reason'][:400]}")
        return verdict

    def _step(self) -> bool:
        self._gate_done = {label: rc for label, _, rc in self.gates.poll()}
        done = {label: rc for label, _, rc in self.perf.poll()}
        active = next(
            (st for st in self.state.candidates.values() if st.status in ("micro", "e2e", "gate")),
            None,
        )
        if active is None:
            active = next(
                (st for st in self.state.candidates.values() if st.status == "ready"), None
            )
            if active is not None and self._stack(active):
                active.status = "e2e" if active.scope == "e2e" else "micro"
                active.control_label = ""
                active.microbench.pop("spool", None)
        if active is not None:
            if active.status == "micro":
                self._micro(active, done)
            elif active.status == "e2e":
                self._e2e(active, done)
            if active.status == "gate":
                gate = self._gate(self._gate_key(active), Path(active.eval_tree))
                if gate is not None:
                    active.gate = gate
                    self.ledger.append("candidate_gate", slug=active.slug, **gate)
                    if gate["verdict"] == PASS:
                        self._keep(active)
                    else:
                        active.status, active.reason = (
                            "reverted",
                            f"keep gate {gate['verdict']}: {gate['reason']}",
                        )
        self.state.save(self.state_path)
        return any(
            st.status in ("ready", "micro", "e2e", "gate") for st in self.state.candidates.values()
        )

    def _micro_label(self, st: CandidateState, side: str) -> str:
        return f"micro-{self._gate_key(st)}-{side}"

    def _micro(self, st: CandidateState, done: dict[str, int]) -> None:
        results = {}
        for side, tree, commit in (
            ("control", self.state.best_tree, st.eval_base),
            ("candidate", st.eval_tree, git_checked(Path(st.eval_tree), "rev-parse", "HEAD")),
        ):
            label = self._micro_label(st, side)
            path = self.data / label / "result.json"
            if self.perf.queued(label):
                return
            receipt = self.data / label / "receipt.json"
            if label in done:
                _, record = self.perf.output(label)
                if done[label] != 0 or record.get("rc") != 0 or record.get("tree") != tree:
                    st.status, st.reason = "blocked", f"{label}: failed job or wrong-tree receipt"
                    return
                self._write(receipt, record)
            if receipt.exists():
                try:
                    record = json.loads(receipt.read_text())
                    result = microbench.load_result(
                        path,
                        commit=commit,
                        harness=st.microbench["sha256"],
                        cases=st.microbench["cases"],
                    )
                    if (
                        record.get("rc") != 0
                        or record.get("tree") != tree
                        or record.get("host") != result["hostname"]
                    ):
                        raise ValueError("microbench receipt does not match the measured host/tree")
                    if not Path(result["module"]).resolve().is_relative_to(Path(tree).resolve()):
                        raise ValueError("microbench imported another checkout")
                    results[side] = result
                except (KeyError, OSError, TypeError, ValueError) as exc:
                    st.status, st.reason = "blocked", f"invalid micro/module evidence: {exc}"
                    return
                continue
            free = self.perf.free_spools()
            spool = st.microbench.get("spool")
            if spool and spool not in free:
                if not self.perf.alive(spool):
                    st.status, st.reason = "blocked", "microbench host left before A/B completed"
                return
            if not free:
                return
            spool = spool or free[0]
            st.microbench["spool"] = spool
            path.unlink(missing_ok=True)
            args = [
                "python",
                str(Path(microbench.__file__).resolve()),
                "--script",
                st.microbench["script"],
                "--output",
                str(path),
                "--tree",
                tree,
                "--commit",
                commit,
                "--harness-sha256",
                st.microbench["sha256"],
            ]
            command = f"source {shlex.quote(self.bench.env_script)}\n" + shlex.join(args)
            self.perf.submit(spool, label, command, tree=tree, wall=self.bench.wall)
            return
        if results["control"]["hostname"] != results["candidate"]["hostname"]:
            st.status, st.reason = "blocked", "microbench A/B ran on different hosts"
            return
        st.deltas = microbench.deltas(
            results["candidate"]["latency_us"], results["control"]["latency_us"]
        )
        threshold = max(self.min_gain, self.noise_factor * NoiseFloor().default)
        st.verdict = (
            "LOSS"
            if any(v <= -threshold for v in st.deltas.values())
            else "WIN"
            if any(v >= threshold for v in st.deltas.values())
            else "NEUTRAL"
        )
        st.reason = f"{st.scope} A/B, assumed noise band {threshold:.1%}; no e2e claim"
        self.ledger.append(
            "candidate_microbench",
            slug=st.slug,
            verdict=st.verdict,
            deltas=st.deltas,
            evidence=results,
            reason=st.reason,
        )
        st.e2e_label = ""
        st.status = "gate" if st.verdict == "WIN" else "reverted"

    def _e2e(self, st: CandidateState, done: dict[str, int]) -> None:
        free = self.perf.free_spools()
        if not st.control_label:
            if not free:
                return
            spool = next(
                (sp for sp in free if self._control_key(spool_host(sp)) in self.state.controls),
                free[0],
            )
            st.microbench["spool"] = spool
            key = self._control_key(spool_host(spool))
            st.control_label = (
                self.state.controls.get(key)
                or f"ctl-{st.eval_base[:10]}-{spool_host(spool) or Path(spool).name}"
            )
            self.state.controls[key] = st.control_label
        spool = st.microbench["spool"]
        for label, tree in ((st.control_label, self.state.best_tree), (st.e2e_label, st.eval_tree)):
            if self.perf.queued(label):
                return
            if label in done and done[label] != 0:
                st.status, st.reason = (
                    "blocked",
                    f"{label}: serving measurement failed rc={done[label]}",
                )
                return
            try:
                measured = load_measurement(self.data, label)
                if measured is not None:
                    validate_measurement(
                        measured,
                        self.bench,
                        self.base_configs,
                        git_checked(Path(tree), "rev-parse", "HEAD"),
                    )
                    continue
            except (KeyError, OSError, TypeError, ValueError) as exc:
                st.status, st.reason = "blocked", f"invalid serving evidence: {exc}"
                return
            if label in done:
                st.status, st.reason = "blocked", f"{label}: missing serving measurement"
                return
            if spool not in free:
                if not self.perf.alive(spool):
                    st.status, st.reason = "blocked", "serving host left before A/B completed"
                return
            self.perf.submit(
                spool,
                label,
                self.bench.command(label, self.data, self.base_configs),
                wall=self.bench.wall,
                tree=tree,
            )
            return
        m = load_measurement(self.data, st.e2e_label)
        ctrl = load_measurement(self.data, st.control_label)
        if m.meta.get("hostname") != ctrl.meta.get("hostname"):
            st.status, st.reason = "blocked", "serving A/B ran on different hosts"
            return
        noise = self._floor()
        inert = inert_members([m], [ctrl], changed=st.changed_members, marker=st.path_marker)
        cmp = compare(
            m,
            ctrl,
            noise,
            min_gain=self.min_gain,
            noise_factor=self.noise_factor,
            inert=inert,
            ttft_tolerance=self.ttft_tolerance,
        )
        st.verdict, st.reason, st.deltas = cmp.verdict, cmp.reason, cmp.judged()
        st.controls, st.inert = [ctrl.label], inert
        self.ledger.measurement(m, candidate=st.slug)
        self.ledger.append("candidate_e2e", slug=st.slug, **cmp.as_dict(), controls=st.controls)
        st.status = "gate" if cmp.verdict == "WIN" else "reverted"
        if st.variant_of:
            st.status = "gate" if self._judge_variant(st, cmp, noise) == "won" else "reverted"
        elif cmp.verdict == "LOSS":
            st.status = self._split(st, m, ctrl, noise, cmp.inert)

    def _split(
        self,
        st: CandidateState,
        m: Measurement,
        ctrl: Measurement,
        noise: NoiseFloor,
        inert: Sequence[str] = (),
    ) -> str:
        shape = classify(
            m, ctrl, noise, min_gain=self.min_gain, noise_factor=self.noise_factor, inert=inert
        )
        self.ledger.append("candidate_split", slug=st.slug, **shape.as_dict())
        if shape.kind == MIXED:
            self.log(f"[campaign] {st.slug}: not split by member ({shape.why}); reverted")
        if shape.kind != SPLIT or not self.split_variants:
            return "reverted"
        st.split = {
            **shape.as_dict(),
            "base": git_checked(Path(st.eval_tree), "rev-parse", "HEAD"),
            "best": st.eval_base,
            "variant": "",
        }
        st.reason = f"split ({shape.mode}): {st.reason}"
        self.log(
            f"[campaign] {st.slug}: split: loses only at {', '.join(shape.lost)} ({shape.mode})"
        )
        return "split"

    def open_variant(self, slug: str) -> str:
        st = self.state.candidates[slug]
        variant = st.split.get("variant") or ""
        if not variant:
            variant, n = f"{slug}-gated", 1
            while variant in self.state.candidates:
                n += 1
                variant = f"{slug}-gated-{n}"
            st.split["variant"] = variant
            self.state.candidates[variant] = CandidateState(variant, variant_of=slug)
            self.ledger.append(
                "split_variant",
                slug=slug,
                variant=variant,
                mode=st.split.get("mode"),
                condition=st.split.get("condition"),
            )
            self.state.save(self.state_path)
        return variant

    def close_split(self, slug: str, why: str) -> None:
        st = self.state.candidates[slug]
        st.status, st.reason = "reverted", f"split ({st.split.get('mode')}), no variant: {why}"
        self.ledger.append("split_closed", slug=slug, why=why)
        self.state.save(self.state_path)

    def _judge_variant(self, st: CandidateState, cmp: Comparison, noise: NoiseFloor) -> str:
        orig = self.state.candidates.get(st.variant_of)
        win, why = judge_variant(
            cmp,
            orig.split if orig else {},
            orig.deltas if orig else {},
            noise,
            min_gain=self.min_gain,
            noise_factor=self.noise_factor,
        )
        self.ledger.append(
            "variant_verdict", slug=st.slug, variant_of=st.variant_of, win=win, why=why
        )
        if not win:
            st.reason = f"variant of {st.variant_of}: {why}"
        return "won" if win else "reverted"

    def _best_head(self) -> str:
        return git_checked(Path(self.state.best_tree), "rev-parse", "HEAD")

    def _keep(self, st: CandidateState) -> None:
        best = Path(self.state.best_tree)
        before = self._best_head()
        if before != st.eval_base:
            st.status, st.reason = "blocked", "best tree moved outside the serialized verification"
            return
        out = subprocess.run(
            [
                "git",
                "-C",
                str(best),
                "merge",
                "--ff-only",
                git_checked(Path(st.eval_tree), "rev-parse", "HEAD"),
            ],
            capture_output=True,
            text=True,
        )
        if out.returncode != 0:
            st.status, st.reason = (
                "conflict",
                f"could not move tune/best: {out.stderr.strip()[:300]}",
            )
            st.rebase_ref = before
            return
        st.kept_commits = git_checked(best, "rev-list", "--reverse", f"{before}..HEAD").splitlines()
        st.status = "kept"
        self.state.best_commit = self._best_head()
        self.state.kept.append(st.slug)
        if st.scope == "e2e":
            self.set_control(self.data / st.e2e_label)
        self.ledger.append(
            "candidate_kept",
            slug=st.slug,
            commits=st.kept_commits,
            best=self.state.best_commit,
            deltas=st.deltas,
            perf_scope=st.scope,
            gate=st.gate,
            ungated=bool(st.gate.get("ungated")),
        )
        ungated = " (UNGATED: perf.campaign.accuracy.gate is off)" if st.gate.get("ungated") else ""
        self.log(f"[campaign] KEEP {st.slug} -> tune/best {self.state.best_commit[:10]}{ungated}")
