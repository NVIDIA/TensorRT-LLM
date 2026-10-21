# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Modeling iterations: the orchestrator half of a tuner candidate.

The tuner claims (``decision.yaml``: CANDIDATE + micro-bench evidence); this
module verifies and acts:

1. **e2e**: a whole-node bench job against the candidate's worktree
   (``#spool: tree=<worktree>``, so ``import tensorrt_llm`` resolves there),
   at the campaign's current config, started on an idle node.
2. **judge**: :func:`records.compare` against the mean of the controls of the
   current best tree measured on the same node at the same config -- A before
   the first candidate there, A' after it -- under the best tree's own noise
   floor (:meth:`Campaign._floor`). Members the candidate provably does not change
   (:mod:`.inert`) are judged nowhere. A LOSS that one disturbed run explains, or a
   loss at an inert member, is measured once more on the same node, interleaved
   with a control. A LOSS whose losses share one parallel mode and that wins
   elsewhere is ``split`` (:mod:`.split`).
3. **gate**: on WIN, the keep gate (``keep_gate.py``) on the stacked tree.
4. **keep**: the candidate's commits are cherry-picked onto ``tune/best``. A
   conflict is reported, never resolved silently.

State lives in ``<workdir>/campaign_state.json``; every measurement and verdict
is appended to ``<workdir>/perf_ledger.jsonl``.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import yaml

from ..agents.decisions import names as decision_names
from ..infra.substrate.spoolctl import SpoolError
from ..infra.vcs import MERGE_TREE_MERGE_BASE, git_checked, git_version
from .inert import inert_members
from .keep_gate import ERROR, FAIL, KEEP_FILE, PART_FILE, PASS, KeepGate
from .measure import INFRA_RCS, BenchSpec, SpoolRunner, load_measurement, spool_host
from .records import (
    TTFT_TOLERANCE,
    Comparison,
    Measurement,
    NoiseFloor,
    PerfLedger,
    compare,
    remeasure_reason,
)
from .split import MIXED, SPLIT, classify, judge_variant
from .worktree import add_worktree, link_native

TUNER_DECISIONS = decision_names("tuner")
#: Controls of the best tree a host has before a candidate measured there is judged: A, and A'.
PAIR = 2


@dataclass
class CandidateState:
    slug: str
    # issued | e2e | won | waiting | gate | kept | reverted | dropped | blocked | conflict | split
    status: str = "issued"
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
    # The candidate is measured stacked on the best tree as it stood at
    # ``eval_base``; a keep that moves the best tree first re-stacks it.
    eval_tree: str = ""
    eval_base: str = ""
    # A conflict names the branch to rebase onto and the commit it stood at:
    # ``tune/best``, or ``tune/eval-batch`` for a winner left out of a pending
    # batch (rebased onto the batch while the batch is measured).
    rebase_onto: str = ""
    rebase_ref: str = ""
    # Failed runs of the current e2e label / gate.
    retries: int = 0
    # A LOSS whose losses share one parallel mode (``split.py``): the mode, its sites, the
    # commit it was measured at (``base``) and the member-gated variant asked for (``variant``).
    split: dict = field(default_factory=dict)
    # A member-gated variant names the split candidate it gates; it is never split itself.
    variant_of: str = ""
    # From the decision: the members whose engine runs the change, and the line it prints
    # when its path is taken (``inert.py``).
    changed_members: list[str] = field(default_factory=list)
    path_marker: str = ""
    # The judgment: the controls it was judged against, and the members found inert.
    controls: list[str] = field(default_factory=list)
    inert: dict = field(default_factory=dict)
    # A verdict measured once more before it counts: the candidate's further labels on its
    # host, and {why, controls: the host's controls the re-measure waits for, first: the
    # comparison that asked for it, failed}.
    samples: list[str] = field(default_factory=list)
    remeasure: dict = field(default_factory=dict)


@dataclass
class CampaignState:
    best_tree: str = ""
    best_commit: str = ""
    control_label: str = ""
    candidates: dict[str, CandidateState] = field(default_factory=dict)
    kept: list[str] = field(default_factory=list)
    # Same-host controls, keyed "<best commit>|<host>": hosts differ by
    # more than most candidates move, so a candidate is judged on its own host. The
    # first control of each key; ``samples`` holds every one, in the order they ran.
    controls: dict[str, str] = field(default_factory=dict)
    samples: dict[str, list[str]] = field(default_factory=dict)
    # Winners on one best tree measured together: {members, base, tree, label, status}.
    batch: dict = field(default_factory=dict)

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


def _ungated(gate: dict) -> str:
    """What a keep's log line says when no accuracy gate stood before it."""
    return " (UNGATED: perf.campaign.accuracy.gate is off)" if gate.get("ungated") else ""


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
        self.workdir = workdir
        self.repo = repo
        self.bp = branch_prefix
        self.bench = bench
        self.base_configs = list(base_configs)
        self.perf = perf_runner
        self.gates = gate_runner
        self.keep_gate = keep_gate
        #: A floor measured before the campaign (the ground repeats), and the commit it was
        #: measured on: it joins the floor only while that commit is the best tree. ``None``
        #: declares it for every tree.
        self.noise = noise
        self.noise_commit = noise_commit
        self.min_gain = min_gain
        self.noise_factor = noise_factor
        #: How much slower a first token may come at concurrency 1; None: not judged.
        self.ttft_tolerance = ttft_tolerance
        #: A split LOSS asks for a member-gated variant (``perf.campaign.split_variants``).
        self.split_variants = split_variants
        self.log = log
        self.data = workdir / "data"
        self.data.mkdir(parents=True, exist_ok=True)
        self.state_path = workdir / "campaign_state.json"
        self.ledger = PerfLedger(workdir / "perf_ledger.jsonl")
        self.state = CampaignState.load(self.state_path)
        self._rcs: dict[str, int] = {}
        #: Gate jobs the last poll found finished: label -> return code.
        self._gate_done: dict[str, int] = {}
        for st in self.state.candidates.values():
            for label in (st.e2e_label, *st.samples) if st.status == "e2e" else ():
                if label and not (self.data / label / "meta.json").exists():
                    self.perf.adopt(label)
            if st.status == "gate" and (st.eval_tree or st.worktree):
                # Gate jobs are named by the tree gated (see the gate step).
                sha = git_checked(
                    Path(st.eval_tree or st.worktree),
                    "rev-parse",
                    "--short=10",
                    "HEAD",
                    check=False,
                )
                if sha:
                    self._adopt_gate(f"{st.slug}-{sha}")
        b = self.state.batch
        if (
            b.get("status") == "e2e"
            and b.get("label")
            and not (self.data / b["label"] / "meta.json").exists()
        ):
            self.perf.adopt(b["label"])
        elif b.get("status") == "gate" and b.get("tree"):
            sha = git_checked(Path(b["tree"]), "rev-parse", "--short=10", "HEAD", check=False)
            if sha:
                self._adopt_gate(f"batch-{sha}")
        for labels in self.state.samples.values():
            for label in labels:
                if not (self.data / label / "meta.json").exists():
                    self.perf.adopt(label)

    # -- configuration ---------------------------------------------------------------
    def _control_key(self, host: str) -> str:
        return f"{self.state.best_commit[:10]}|{host}"

    def _spool_of(self, host: str) -> str | None:
        """The perf session on ``host`` now, or None once no session reports it."""
        return next((sp for sp in self.perf.spools if spool_host(sp) == host), None)

    def _controls(self, host: str) -> list[Measurement]:
        """The measurements of the best tree at the current config on ``host``, in the order they ran."""
        found = (
            load_measurement(self.data, label)
            for label in self.state.samples.get(self._control_key(host), [])
        )
        return [m for m in found if m is not None]

    def _register(self, host: str, label: str) -> None:
        """``label`` measured (or measures) the best tree at the current config on ``host``."""
        key = self._control_key(host)
        labels = self.state.samples.setdefault(key, [])
        if label not in labels:
            labels.append(label)
        self.state.controls.setdefault(key, labels[0])

    def _floor(self) -> NoiseFloor:
        """The noise floor of the best tree at the current config, per site.

        Measured on this tree, never inherited from an older best: the spread of each
        host's controls of it (two or more on one node), the widest per site, and the
        floor the campaign was opened with while its tree is the best.
        """
        prefix = self._control_key("")
        floors = []
        for key, labels in self.state.samples.items():
            if not key.startswith(prefix):
                continue
            found = [m for m in (load_measurement(self.data, label) for label in labels) if m]
            if len(found) >= 2:
                floors.append(NoiseFloor.from_measurements(found))
        if self.noise.spread and (
            self.noise_commit is None
            or (self.noise_commit and self.state.best_commit.startswith(self.noise_commit[:10]))
        ):
            floors.append(self.noise)
        return NoiseFloor.widest(floors)

    def _sample_label(self, spool: str, n: int) -> str:
        """The ``n``-th control of the best tree at the current config on ``spool``'s host."""
        label = f"ctl-{self.state.best_commit[:10]}-{spool_host(spool) or Path(spool).name}"
        return label if n == 1 else f"{label}-s{n}"

    def _ensure_controls(self, spool: str, want: int = PAIR) -> bool:
        """True when ``spool``'s host has ``want`` controls of the best tree at the current config.

        Otherwise the next one is queued there once the node is idle and runs nothing of the
        campaign's (and False is returned): A before the first candidate measured on that
        node, A' right after it, A'' after a re-measure.
        """
        host = spool_host(spool)
        labels = self.state.samples.get(self._control_key(host), [])
        measured = [label for label in labels if (self.data / label / "meta.json").exists()]
        if len(measured) >= want:
            return True
        if any(self.perf.queued(label) for label in labels) or spool not in self.perf.free_spools():
            return False
        missing = [label for label in labels if label not in measured]
        label = missing[0] if missing else self._sample_label(spool, len(labels) + 1)
        self._register(host, label)
        cmd = self.bench.command(label, self.data, self.base_configs)
        self.perf.submit(spool, label, cmd, wall=self.bench.wall, tree=self.state.best_tree)
        self.log(f"[campaign] control {label} queued on {spool}")
        return False

    def _unplaced(self) -> int:
        """Measurements waiting for a node: candidates (and a batch) neither measured nor queued."""
        labels = [
            st.e2e_label
            for st in self.state.candidates.values()
            if st.status == "e2e" and st.e2e_label
        ]
        if self.state.batch.get("status") == "e2e" and self.state.batch.get("label"):
            labels.append(self.state.batch["label"])
        return sum(
            1
            for label in labels
            if not self.perf.queued(label) and load_measurement(self.data, label) is None
        )

    def _hosts_starting(self) -> int:
        """Hosts whose first control of the best tree at the current config is in flight."""
        prefix = self._control_key("")
        return sum(
            1
            for key, labels in self.state.samples.items()
            if key.startswith(prefix)
            and any(self.perf.queued(label) for label in labels)
            and not any((self.data / label / "meta.json").exists() for label in labels)
        )

    def _queue_measurement(self, label: str, tree: str) -> str | None:
        """Queue ``label`` on an idle node whose host has a control of the best tree; its spool.

        A host with both controls (A and A') first, then one with A, whose A' follows the
        measurement there. With neither, a control goes first on an idle node (and None is
        returned): one per measurement waiting for a node, never on every host at once.
        """
        free = self.perf.free_spools()
        if not free:
            return None
        counts = {sp: len(self._controls(spool_host(sp))) for sp in free}
        ready = sorted((sp for sp in free if counts[sp]), key=lambda sp: -min(counts[sp], PAIR))
        if ready:
            cmd = self.bench.command(label, self.data, self.base_configs)
            self.perf.submit(ready[0], label, cmd, wall=self.bench.wall, tree=tree)
            return ready[0]
        if self._hosts_starting() < self._unplaced():
            self._ensure_controls(free[0], 1)
        return None

    def set_control(self, label_dir: Path) -> None:
        """Adopt a measurement under ``data/`` of the best tree at the current config as a control.

        It is one of its host's controls; the first adopted is named in ``control_label``.
        """
        if not self.state.control_label:
            self.state.control_label = label_dir.name
        m = load_measurement(self.data, label_dir.name)
        if m is not None:
            self._register(str(m.meta.get("hostname") or ""), label_dir.name)
        self.state.save(self.state_path)

    # -- the loop ----------------------------------------------------------------------
    def submit_candidate(self, slug: str, decision_path: Path) -> CandidateState:
        st = self.state.candidates.setdefault(slug, CandidateState(slug))
        data = read_decision(decision_path)
        st.decision = data["decision"]
        st.worktree = str(data.get("worktree") or "")
        st.commits = [str(c) for c in data.get("commits") or []]
        changed = data.get("changed_members") or []
        st.changed_members = [str(m) for m in changed] if isinstance(changed, list) else []
        st.path_marker = str(data.get("path_marker") or "").strip()
        self.ledger.append(
            "tuner_decision",
            slug=slug,
            decision=st.decision,
            candidate_kind=data.get("kind"),
            hypothesis=data.get("hypothesis"),
            microbench=data.get("microbench"),
            projection=data.get("projection"),
            sources=data.get("sources"),
            changed_members=st.changed_members,
            path_marker=st.path_marker,
        )
        if st.decision != "CANDIDATE":
            st.status = {"DROPPED": "dropped", "BLOCKED": "blocked"}.get(
                st.decision, st.decision.lower()
            )
            self.state.save(self.state_path)
            return st
        tree = Path(st.worktree)
        if tree.is_dir() and self.state.best_tree:
            # A candidate built on another candidate's branch carries that branch's
            # commits too: stack everything since it left the best tree.
            best_head = self._best_head()
            # Commits on the candidate that the best tree does not already carry,
            # by patch identity: a parent candidate kept earlier was cherry-picked
            # onto the best tree under another SHA and must not be applied twice.
            since = git_checked(
                tree,
                "rev-list",
                "--reverse",
                "--cherry-pick",
                "--right-only",
                "--no-merges",
                f"{best_head}...HEAD",
                check=False,
            ).splitlines()
            if since and set(st.commits) <= set(since) | {c[:10] for c in since}:
                st.commits = since
        if git_checked(tree, "status", "--porcelain", "-uno"):
            st.status, st.reason = (
                "blocked",
                "worktree has uncommitted changes; the candidate is its commits",
            )
            self.state.save(self.state_path)
            return st
        b = self.state.batch
        if (
            b.get("tip")
            and subprocess.run(
                ["git", "-C", str(tree), "merge-base", "--is-ancestor", b["tip"], "HEAD"]
            ).returncode
            == 0
        ):
            # Rebased onto a batch still being measured: it stacks once the batch is kept.
            st.status, st.reason = "waiting", f"rebased onto batch {b['label']}"
            self.state.save(self.state_path)
            return st
        if (
            st.rebase_onto == "tune/eval-batch"
            and st.rebase_ref
            and st.rebase_ref != b.get("tip")
            and subprocess.run(
                ["git", "-C", str(tree), "merge-base", "--is-ancestor", st.rebase_ref, "HEAD"]
            ).returncode
            == 0
            and subprocess.run(
                [
                    "git",
                    "-C",
                    str(tree),
                    "merge-base",
                    "--is-ancestor",
                    st.rebase_ref,
                    self._best_head(),
                ]
            ).returncode
            != 0
        ):
            # Rebased onto a batch that was not kept: it carries the batch's commits.
            st.status, st.reason = "conflict", "rebased onto a batch that was not kept"
            st.rebase_onto, st.rebase_ref = "tune/best", self._best_head()
            self.state.save(self.state_path)
            return st
        if not self._stack(st):
            self.state.save(self.state_path)
            return st
        st.control_label = self.state.control_label
        st.status = "e2e"
        self.state.save(self.state_path)
        return st

    def _stack(self, st: CandidateState) -> bool:
        """Build ``eval-<slug>``: the current best tree with the candidate's commits on top."""
        head = self._best_head()
        if st.worktree and Path(st.worktree).is_dir():
            # Re-derive against the best tree as it is now: whatever it already
            # carries (by patch identity) is not applied again.
            pending = git_checked(
                Path(st.worktree),
                "rev-list",
                "--reverse",
                "--cherry-pick",
                "--right-only",
                "--no-merges",
                f"{head}...HEAD",
                check=False,
            ).splitlines()
            if pending:
                st.commits = pending
        eval_tree = Path(self.state.best_tree).parent / f"eval-{st.slug}"
        branch = f"{self.bp}/eval-{st.slug}"
        if (eval_tree / ".git").exists():
            # Reuse the checkout: moving it to the new head rewrites only the files that differ.
            git_checked(eval_tree, "cherry-pick", "--abort", check=False)
            # Forced: an interrupted earlier checkout (a full filesystem) leaves
            # local changes that would refuse a plain one.
            git_checked(eval_tree, "checkout", "-q", "-f", "-B", branch, head)
            git_checked(eval_tree, "reset", "-q", "--hard", head)
        elif (spare := self._spare_eval_tree(st.slug)) is not None:
            # A finished candidate's checkout, renamed and moved to the new head.
            git_checked(self.repo, "worktree", "move", str(spare), str(eval_tree))
            git_checked(eval_tree, "checkout", "-q", "-f", "-B", branch, head)
            git_checked(eval_tree, "reset", "-q", "--hard", head)
            # Ignored files too: __pycache__ written at the old path makes pytest import
            # stale modules and fail collection; the native links are re-made below.
            git_checked(eval_tree, "clean", "-q", "-fdx")
        else:
            add_worktree(self.repo, eval_tree, "-B", branch, str(eval_tree), head)
        link_native(self.repo, eval_tree)
        out = subprocess.run(
            ["git", "-C", str(eval_tree), "cherry-pick", *st.commits],
            capture_output=True,
            text=True,
        )
        if out.returncode != 0:
            git_checked(eval_tree, "cherry-pick", "--abort", check=False)
            st.status, st.reason = (
                "conflict",
                f"does not stack on tune/best {head[:10]}: {out.stderr.strip()[:300]}",
            )
            st.rebase_onto, st.rebase_ref = "tune/best", head
            self.ledger.append("candidate_conflict", slug=st.slug, best=head, detail=st.reason)
            return False
        st.eval_tree, st.eval_base = str(eval_tree), head
        self._label(st)
        return True

    def _spare_eval_tree(self, slug: str) -> Path | None:
        """The eval checkout of a candidate that is finished with it (kept, reverted, dropped)."""
        for other in self.state.candidates.values():
            if (
                other.slug != slug
                and other.status in ("kept", "reverted", "dropped", "split")
                and other.eval_tree
            ):
                tree = Path(other.eval_tree)
                if (tree / ".git").exists() and not self.perf.queued(other.e2e_label):
                    other.eval_tree = ""
                    return tree
        return None

    def _label(self, st: CandidateState) -> None:
        st.retries = 0
        st.samples, st.remeasure, st.controls, st.inert = [], {}, [], {}
        st.e2e_label = (
            f"e2e-{st.slug}-{git_checked(Path(st.eval_tree), 'rev-parse', '--short=10', 'HEAD')}"
        )

    def step(self) -> bool:
        """Advance every candidate as far as possible; True while any is in flight.

        A full filesystem (a shared quota other users fill) fails the loop's own
        small writes for minutes at a time: the step is retried on the next poll.
        """
        try:
            return self._step()
        except OSError as exc:
            if exc.errno not in (28, 122):  # ENOSPC, EDQUOT
                raise
            self.log(f"[campaign] {exc.strerror}; retrying on the next poll")
            return True
        except SpoolError as exc:
            # A session died between choosing it and queuing on it (preempted).
            self.log(f"[campaign] {exc}; retrying on the next poll")
            return True

    def _retry(self, st: CandidateState, rc: int, what: str) -> bool:
        """True when a failed run is re-queued: once for any failure, twice for the substrate's own."""
        limit = 2 if rc in INFRA_RCS else 1
        if st.retries >= limit:
            return False
        st.retries += 1
        self.ledger.append("retry", slug=st.slug, what=what, rc=rc, attempt=st.retries)
        self.log(
            f"[campaign] {st.slug}: {what} failed (rc={rc}); re-queued (attempt {st.retries + 1})"
        )
        return True

    # -- the keep gate -----------------------------------------------------------------
    def _gate_dir(self, key: str) -> Path:
        return self.workdir / "gates" / key

    def _adopt_gate(self, key: str) -> None:
        """Re-register the parts of gate ``key`` an earlier process left in flight."""
        for part in self.keep_gate.parts():
            if not (self._gate_dir(key) / part / PART_FILE).exists():
                self.gates.adopt(f"gate-{key}-{part}")

    def _cancel_gate(self, key: str) -> bool:
        """Stop every part of gate ``key`` still in flight; True when one was."""
        return any([self.gates.cancel(f"gate-{key}-{part}") for part in self.keep_gate.parts()])

    @staticmethod
    def _write(path: Path, data: dict) -> dict:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str) + "\n")
        return data

    def _requeue(self, key: str, where: Path, verdict: dict) -> bool:
        """Set aside a run of a gate part that reached no verdict; True when it may run again.

        Once for any such run, twice for the substrate's own failures.
        """
        attempts = len(list(where.glob("attempt-*.json")))
        if attempts >= (2 if verdict.get("infrastructure") else 1):
            return False
        self._write(where / f"attempt-{attempts + 1}.json", verdict)
        part = verdict.get("part")
        self.ledger.append(
            "retry", slug=key, what=f"gate {part}", rc=verdict.get("rc"), attempt=attempts + 1
        )
        self.log(
            f"[campaign] gate {key}: {part} reached no verdict ({str(verdict.get('detail'))[:200]}); re-queued"
        )
        return True

    def _gate(self, key: str, tree: Path) -> dict | None:
        """Advance the keep gate of ``tree`` under ``gates/<key>``: its verdict, or None while it runs.

        Every part without a verdict is queued as soon as a dev session can take it, so the
        members and the suite run side by side. A part that fails ends the gate and stops
        the others. A run that reached no verdict (the substrate ended it, the suite could
        not collect) is queued again (:meth:`_requeue`); past that it stands as an ERROR,
        and only a gate whose every part passed keeps.
        """
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
                    parts[part] = self._write(where / PART_FILE, {
                        "verdict": ERROR, "part": part,
                        "detail": f"needs {gpus} GPUs; the widest dev session hands out {widest}",
                    })  # fmt: skip
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
            self.gates.submit(spool, job, prepared.script, wall=prepared.wall, tree=str(tree),
                              gpus=str(prepared.gpus))  # fmt: skip
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
        gated = self.gates.poll()
        polled = [*self.perf.poll(), *gated]
        finished = {label for label, _, _ in polled}
        self._rcs = {label: rc for label, _, rc in polled}
        self._gate_done = {label: rc for label, _, rc in gated}
        busy = False
        for st in self.state.candidates.values():
            if st.status == "e2e":
                busy = True
                m = load_measurement(self.data, st.e2e_label)
                if m is None and not self.perf.queued(st.e2e_label):
                    if st.e2e_label in finished and not self._retry(
                        st, self._rcs[st.e2e_label], "e2e"
                    ):
                        st.status, st.reason = (
                            "reverted",
                            "e2e measurement failed (see the label's .partial dir)",
                        )
                        continue
                    if st.e2e_label in finished:
                        continue
                    if st.eval_base and st.eval_base != self._best_head():
                        # Not measured yet and the best tree moved: stack on it first.
                        if not self._stack(st):
                            continue
                    spool = self._queue_measurement(st.e2e_label, st.eval_tree or st.worktree)
                    if spool:
                        self.log(f"[campaign] e2e {st.slug} queued on {spool}")
                    continue
                if m is None:
                    continue
                best_moved = bool(st.eval_base) and st.eval_base != self._best_head()
                if best_moved:
                    # The best tree moved while this was measured: judge it on the new best.
                    self.log(f"[campaign] {st.slug}: best moved, re-stacking")
                    self._stack(st)
                    continue
                judged = self._judged(st, m)
                if judged is None:
                    continue
                cmp, cand, ctrl, noise = judged
                st.verdict, st.reason, st.deltas = cmp.verdict, cmp.reason, cmp.judged()
                self.ledger.append(
                    "candidate_e2e", slug=st.slug, **cmp.as_dict(), host=m.meta.get("hostname"),
                    controls=st.controls, samples=[st.e2e_label, *st.samples], inert_why=st.inert,
                    remeasured=st.remeasure.get("why", ""),
                    noise={"spread": noise.spread, "samples": noise.samples},
                )  # fmt: skip
                self.log(f"[campaign] {st.slug}: {cmp.verdict} {cmp.reason}")
                # A winner waits for the other measurements on this best tree: several
                # winners are stacked and measured once (see _batch).
                st.status = "won" if cmp.verdict == "WIN" else "reverted"
                if st.variant_of:
                    st.status = self._judge_variant(st, cmp, noise)
                elif cmp.verdict == "LOSS":
                    st.status = self._split(st, cand, ctrl, noise, cmp.inert)
                st.retries = 0
            if st.status == "gate":
                busy = True
                # Keyed by the tree gated: a candidate re-stacked on a new best is a
                # different tree, and its earlier gate says nothing about this one.
                tree = Path(st.eval_tree or st.worktree)
                key = f"{st.slug}-{git_checked(tree, 'rev-parse', '--short=10', 'HEAD')}"
                if (
                    not (self._gate_dir(key) / KEEP_FILE).exists()
                    and st.eval_base
                    and st.eval_base != self._best_head()
                ):
                    # Another keep moved the best tree under this gate: its verdict would be
                    # about a tree that will never be kept. Stop it, re-stack, measure again.
                    if self._cancel_gate(key):
                        self.log(
                            f"[campaign] {st.slug}: best moved during its gate; gate cancelled, re-stacking"
                        )
                    if self._stack(st):
                        st.status, st.gate = "e2e", {}
                    continue
                gate = self._gate(key, tree)
                if gate is None:
                    continue
                st.gate = gate
                self.ledger.append("candidate_gate", slug=st.slug, **gate)
                if gate["verdict"] == PASS:
                    self._keep(st)
                else:
                    st.status, st.reason = (
                        "reverted",
                        f"keep gate {gate['verdict']}: {gate['reason']}",
                    )
        busy = self._batch() or busy
        self.state.save(self.state_path)
        return busy

    # -- the judgment -------------------------------------------------------------------
    def _judged(
        self, st: CandidateState, m: Measurement
    ) -> tuple[Comparison, Measurement, Measurement, NoiseFloor] | None:
        """``(comparison, candidate, control, floor)`` once ``st`` can be judged, else None.

        The candidate is judged on the host it ran on, against the mean of that host's
        controls of the best tree (at least A and A'), under the best tree's floor
        (:meth:`_floor`). A verdict that one disturbed run explains is measured once more on
        the same node, interleaved: the candidate again (B'), then a control (A''). A loss
        repeated at a member it does not change is reported as the host's noise.
        """
        host = str(m.meta.get("hostname") or "")
        spool = self._spool_of(host)
        extra = [load_measurement(self.data, label) for label in st.samples]
        if None in extra:
            label = st.samples[extra.index(None)]
            if self.perf.queued(label):
                return None
            if label in self._rcs and not self._retry(st, self._rcs[label], "re-measure"):
                self.log(f"[campaign] {st.slug}: its re-measure failed; judged on its first run")
                self.ledger.append("remeasure_failed", slug=st.slug, label=label)
                st.samples, st.remeasure["failed"] = [], True
                return self._judged(st, m)
            if label in self._rcs:
                return None
            if spool is None:
                self._unpair(st, host, "its node left before the re-measure")
                return None
            if spool in self.perf.free_spools():
                cmd = self.bench.command(label, self.data, self.base_configs)
                self.perf.submit(spool, label, cmd, wall=self.bench.wall,
                                 tree=st.eval_tree or st.worktree)  # fmt: skip
                self.log(f"[campaign] {st.slug}: re-measure {label} queued on {spool}")
            return None
        want = int(st.remeasure.get("controls") or PAIR) if st.samples else PAIR
        controls = self._controls(host)
        if len(controls) < want:
            if spool is None:
                self._unpair(st, host, "its node left before its controls were measured")
            else:
                self._ensure_controls(spool, want)
            return None
        noise = self._floor()
        samples = [m, *extra]
        inert = inert_members(samples, controls, changed=st.changed_members, marker=st.path_marker)
        cand, ctrl = Measurement.mean(samples), Measurement.mean(controls)
        cmp = compare(cand, ctrl, noise, min_gain=self.min_gain, noise_factor=self.noise_factor,
                      inert=inert, ttft_tolerance=self.ttft_tolerance)  # fmt: skip
        st.control_label, st.controls, st.inert = (
            controls[0].label,
            [c.label for c in controls],
            inert,
        )
        if not st.remeasure:
            why = remeasure_reason(
                cmp, noise, min_gain=self.min_gain, noise_factor=self.noise_factor,
                ttft_tolerance=self.ttft_tolerance or TTFT_TOLERANCE,
            )  # fmt: skip
            if why:
                st.remeasure = {"why": why, "controls": len(controls) + 1, "first": cmp.as_dict()}
                st.samples, st.retries = [f"{st.e2e_label}-r2"], 0
                self.ledger.append("remeasure", slug=st.slug, host=host, why=why, **cmp.as_dict())
                self.log(
                    f"[campaign] {st.slug}: {cmp.verdict} not counted yet ({why}); measured again "
                    f"on {host}, then a control"
                )
                return None
        elif extra:
            again = compare(extra[-1], ctrl, noise, min_gain=self.min_gain,
                            noise_factor=self.noise_factor, inert=inert,
                            ttft_tolerance=self.ttft_tolerance)  # fmt: skip
            first = set((st.remeasure.get("first") or {}).get("inert_losses") or ())
            repeated = [site for site in again.inert_losses if site in first]
            if repeated:
                self.ledger.append("host_noise", slug=st.slug, host=host, sites=repeated,
                                   inert=inert, controls=st.controls)  # fmt: skip
                self.log(
                    f"[campaign] {st.slug}: lost again at {', '.join(repeated)}, where it changes "
                    f"nothing: the noise of {host}, not held against it"
                )
        for sample in samples:
            self.ledger.measurement(sample, candidate=st.slug)
        return cmp, cand, ctrl, noise

    def _unpair(self, st: CandidateState, host: str, why: str) -> None:
        """Set ``st``'s measurements aside: it is measured again on a node that can pair it."""
        moved = []
        for label in [st.e2e_label, *st.samples]:
            path = self.data / label
            if path.exists():
                aside = path.with_name(f"{label}.unpaired-{int(time.time())}")
                path.rename(aside)
                moved.append(str(aside))
        self.ledger.append("unpaired", slug=st.slug, host=host, why=why, moved_to=moved)
        self.log(f"[campaign] {st.slug}: {why} ({host}); measured again on another node")
        st.samples, st.remeasure, st.retries = [], {}, 0

    # -- a LOSS split by member ---------------------------------------------------------
    def _split(
        self,
        st: CandidateState,
        m: Measurement,
        ctrl: Measurement,
        noise: NoiseFloor,
        inert: Sequence[str] = (),
    ) -> str:
        """``split`` when the LOSS's losses share a parallel mode no win is in, else ``reverted``."""
        shape = classify(m, ctrl, noise, min_gain=self.min_gain, noise_factor=self.noise_factor,
                         inert=inert)  # fmt: skip
        self.ledger.append("candidate_split", slug=st.slug, **shape.as_dict())
        if shape.kind == MIXED:
            self.log(f"[campaign] {st.slug}: not split by member ({shape.why}); reverted")
        if shape.kind != SPLIT:
            return "reverted"
        where = f"{', '.join(shape.lost)} ({shape.mode})"
        if not self.split_variants:
            self.log(
                f"[campaign] {st.slug}: loses only at {where}; split variants are off; reverted"
            )
            return "reverted"
        tree = Path(st.eval_tree or st.worktree)
        st.split = {
            **shape.as_dict(),
            "base": git_checked(tree, "rev-parse", "HEAD"),
            "best": st.eval_base,
            "variant": "",
        }
        st.reason = f"split ({shape.mode}): {st.reason}"
        self.log(
            f"[campaign] {st.slug}: split: loses only at {where}, wins at {', '.join(shape.won)}; "
            "a member-gated variant goes to a tuner"
        )
        return "split"

    def open_variant(self, slug: str) -> str:
        """Register the member-gated variant of split candidate ``slug``; its slug.

        One per candidate. It is a candidate of its own: it counts against the budget, and is
        judged by :func:`split.judge_variant`.
        """
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
            self.log(
                f"[campaign] {slug}: member-gated variant {variant} asked for: the old path "
                f"where {st.split.get('condition')}"
            )
            self.state.save(self.state_path)
        return variant

    def close_split(self, slug: str, why: str) -> None:
        """A split no variant will follow: the candidate is reverted."""
        st = self.state.candidates[slug]
        st.status, st.reason = "reverted", f"split ({st.split.get('mode')}), no variant: {why}"
        self.ledger.append("split_closed", slug=slug, why=why)
        self.log(f"[campaign] {slug}: split, no variant asked for: {why}; reverted")
        self.state.save(self.state_path)

    def _judge_variant(self, st: CandidateState, cmp: Comparison, noise: NoiseFloor) -> str:
        """``won`` or ``reverted``: a variant wins only with its candidate's losses gone and its wins held."""
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
        self.log(
            f"[campaign] {st.slug}: member-gated variant of {st.variant_of}: "
            + ("WIN, " if win else "not a win, ")
            + why
        )
        if not win:
            st.reason = f"variant of {st.variant_of}: {why}"
        return "won" if win else "reverted"

    def _best_head(self) -> str:
        return git_checked(Path(self.state.best_tree), "rev-parse", "HEAD")

    def _batch(self) -> bool:
        """Decide what the winners on the current best tree become: one gate, or one stack."""
        head = self._best_head()
        cands = self.state.candidates
        b = self.state.batch
        if b and b.get("status") == "single":
            # One winner gating alone, the winners that conflict with it rebased onto it.
            (st1,) = (cands[s] for s in b["members"])
            if st1.status == "gate":
                return True
            self._end_batch(kept=st1.status == "kept" and head == b.get("tip"))
            b = {}
        if b and b.get("base") != head:
            for slug in b.get("members", []):
                if cands[slug].status == "won":
                    cands[slug].status, cands[slug].e2e_label = "e2e", ""
            self._end_batch(kept=False)
            b = {}
        for st in cands.values():
            if st.status == "won" and st.eval_base != head:
                st.status, st.e2e_label = "e2e", ""  # the best moved: re-stack
        winners = sorted(
            (st for st in cands.values() if st.status == "won"),
            key=lambda st: max(st.deltas.values() or [0.0]),
            reverse=True,
        )
        if not b:
            waiting = [
                st
                for st in cands.values()
                if st.status == "e2e"
                and st.eval_base == head
                and st.e2e_label
                and load_measurement(self.data, st.e2e_label) is None
            ]
            if not winners or waiting:
                return bool(winners)
            if len(winners) == 1:
                winners[0].status = "gate"
                return True
            # Winners written in parallel collide in the forward file. Stack the
            # compatible set with the largest summed gain (greedy by gain can pick
            # one large winner that excludes three others); the rest rebase onto it.
            chosen = self._compatible_set(head, winners)
            if len(chosen) < 2:
                single = (chosen or winners)[0]
                single.status = "gate"
                # The others conflict with it: rebase them onto its stacked tree while it
                # is gated, as for a batch, instead of gating them on a tree it will move.
                tip = (
                    git_checked(Path(single.eval_tree), "rev-parse", "HEAD")
                    if single.eval_tree
                    else ""
                )
                others = [st for st in winners if st is not single]
                if tip and others:
                    self.state.batch = {
                        "members": [single.slug],
                        "base": head,
                        "tree": single.eval_tree,
                        "label": single.e2e_label,
                        "status": "single",
                        "tip": tip,
                    }
                    for st in others:
                        st.status, st.e2e_label = "conflict", ""
                        st.reason = f"does not stack with {single.slug} on {head[:10]}"
                        st.rebase_onto, st.rebase_ref = "tune/eval-batch", tip
                        self.ledger.append(
                            "candidate_conflict",
                            slug=st.slug,
                            best=head,
                            batch=single.slug,
                            detail=st.reason,
                        )
                    self.log(
                        f"[campaign] {single.slug} gated alone; to rebase onto it: {[st.slug for st in others]}"
                    )
                return True
            tree = Path(self.state.best_tree).parent / "eval-batch"
            if (tree / ".git").exists():
                git_checked(tree, "cherry-pick", "--abort", check=False)
                git_checked(tree, "checkout", "-q", "-f", "-B", f"{self.bp}/eval-batch", head)
                git_checked(tree, "reset", "-q", "--hard", head)
            else:
                add_worktree(self.repo, tree, "-B", f"{self.bp}/eval-batch", str(tree), head)
            link_native(self.repo, tree)
            members = []
            for st in chosen:
                tip = git_checked(tree, "rev-parse", "HEAD")
                out = subprocess.run(
                    ["git", "-C", str(tree), "cherry-pick", *st.commits],
                    capture_output=True,
                    text=True,
                )
                if out.returncode != 0:
                    git_checked(tree, "cherry-pick", "--abort", check=False)
                    git_checked(tree, "reset", "-q", "--hard", tip)
                    continue  # stays `won`: kept alone or re-stacked after the batch
                members.append(st.slug)
            if len(members) < 2:
                (cands[members[0]] if members else winners[0]).status = "gate"
                return True

            label = f"e2e-batch-{git_checked(tree, 'rev-parse', '--short=10', 'HEAD')}"
            tip = git_checked(tree, "rev-parse", "HEAD")
            self.state.batch = b = {
                "members": members,
                "base": head,
                "tree": str(tree),
                "label": label,
                "status": "e2e",
                "tip": tip,
            }
            excluded = [st.slug for st in winners if st.slug not in members]
            for slug in excluded:
                st = cands[slug]
                st.status, st.e2e_label = "conflict", ""
                st.reason = f"does not stack with batch {members} on {head[:10]}"
                st.rebase_onto, st.rebase_ref = "tune/eval-batch", tip
                self.ledger.append(
                    "candidate_conflict", slug=slug, best=head, batch=label, detail=st.reason
                )
            self.ledger.append(
                "batch_formed", members=members, excluded=excluded, base=head, label=label
            )
            self.log(
                f"[campaign] batch {members} stacked on {head[:10]}"
                + (f"; to rebase onto it: {excluded}" if excluded else "")
            )
        if b["status"] == "e2e":
            m = load_measurement(self.data, b["label"])
            if m is None:
                if b["label"] in self._rcs and self._batch_failed(b, winners, "e2e", b["label"]):
                    return True
                if not self.perf.queued(b["label"]):
                    spool = self._queue_measurement(b["label"], b["tree"])
                    if spool:
                        self.log(f"[campaign] batch e2e queued on {spool}")
                return True
            host = str(m.meta.get("hostname") or "")
            controls = self._controls(host)
            if len(controls) < PAIR:
                spool = self._spool_of(host)
                if spool is not None:
                    self._ensure_controls(spool, PAIR)
                else:
                    # Its node left before its controls: measured again on one that can pair it.
                    aside = self.data / f"{b['label']}.unpaired-{int(time.time())}"
                    (self.data / b["label"]).rename(aside)
                    self.ledger.append("unpaired", slug="batch", host=host, moved_to=[str(aside)])
                return True
            noise = self._floor()
            self.ledger.measurement(m, candidate="batch")
            cmp = compare(m, Measurement.mean(controls), noise, min_gain=self.min_gain,
                          noise_factor=self.noise_factor,
                          ttft_tolerance=self.ttft_tolerance)  # fmt: skip
            best_single = {}
            for slug in b["members"]:
                for k, v in cands[slug].deltas.items():
                    best_single[k] = max(best_single.get(k, v), v)
            # The stack must hold every metric's best single gain, within the loss floor.
            holds = cmp.verdict == "WIN" and all(
                cmp.deltas.get(k, 0.0) >= v - max(0.01, self.noise_factor * noise.of(k))
                for k, v in best_single.items()
                if v > 0
            )
            self.ledger.append(
                "batch_e2e",
                members=b["members"],
                holds=holds,
                best_single=best_single,
                **cmp.as_dict(),
                controls=[c.label for c in controls],
                noise={"spread": noise.spread, "samples": noise.samples},
            )
            self.log(f"[campaign] batch {b['members']}: {cmp.verdict} {cmp.reason}; holds={holds}")
            if not holds:
                winners[0].status = "gate"
                self._end_batch(kept=False)
                return True
            b["status"], b["deltas"] = "gate", dict(cmp.deltas)
        if b["status"] == "gate":
            tree = Path(b["tree"])
            gate = self._gate(f"batch-{git_checked(tree, 'rev-parse', '--short=10', 'HEAD')}", tree)
            if gate is None:
                return True
            self.ledger.append("candidate_gate", slug="batch", batch=b["members"], **gate)
            if gate["verdict"] != PASS:
                winners[0].status = "gate"  # fall back to the strongest single
                self._end_batch(kept=False)
                return True
            best = Path(self.state.best_tree)
            before = git_checked(best, "rev-parse", "HEAD")
            out = subprocess.run(
                [
                    "git",
                    "-C",
                    str(best),
                    "merge",
                    "--ff-only",
                    git_checked(Path(b["tree"]), "rev-parse", "HEAD"),
                ],
                capture_output=True,
                text=True,
            )
            if out.returncode != 0:
                winners[0].status = "gate"
                self._end_batch(kept=False)
                return True
            self.state.best_commit = git_checked(best, "rev-parse", "HEAD")
            for slug in b["members"]:
                st = cands[slug]
                st.status, st.gate = "kept", gate
                self.state.kept.append(slug)
            # The batch's measurement is the new best tree's first control on its host.
            self._register(
                str(load_measurement(self.data, b["label"]).meta.get("hostname") or ""), b["label"]
            )
            self.ledger.append(
                "batch_kept",
                members=b["members"],
                best=self.state.best_commit,
                deltas=b.get("deltas"),
                commits=git_checked(best, "rev-list", "--reverse", f"{before}..HEAD").splitlines(),
                gate=gate,
                ungated=bool(gate.get("ungated")),
            )
            self.log(
                f"[campaign] KEEP batch {b['members']} -> tune/best {self.state.best_commit[:10]}"
                + _ungated(gate)
            )
            for other in cands.values():
                if other.status != "e2e":
                    continue
                if other.e2e_label and self.perf.cancel(other.e2e_label):
                    other.e2e_label = ""
                for label in other.samples:
                    self.perf.cancel(label)
            self._end_batch(kept=True)
        return True

    def _batch_failed(
        self, b: dict, winners: Sequence[CandidateState], what: str, key: str
    ) -> bool:
        """A batch run that failed is re-queued once (twice for the substrate's own failures).

        After that the batch is abandoned and its strongest member is gated alone.
        """
        rc = self._rcs.get(key, 1)
        b["retries"] = int(b.get("retries", 0)) + 1
        self.ledger.append("retry", slug="batch", what=what, rc=rc, attempt=b["retries"])
        if b["retries"] <= (2 if rc in INFRA_RCS else 1):
            self.log(f"[campaign] batch {what} failed (rc={rc}); re-queued")
            return False
        self.log(
            f"[campaign] batch {what} failed {b['retries']} times; gating {winners[0].slug} alone"
        )
        winners[0].status = "gate"
        self._end_batch(kept=False)
        return True

    def _end_batch(self, *, kept: bool) -> None:
        """Release the candidates rebased onto the batch: stack them on the new best, or rebase again."""
        head = self._best_head()
        for st in self.state.candidates.values():
            if st.status != "waiting":
                continue
            if kept and self._stack(st):
                st.status, st.control_label = "e2e", self.state.control_label
            elif not kept:
                st.status, st.reason = (
                    "conflict",
                    f"rebased onto batch {self.state.batch.get('label')}, which was not kept",
                )
                st.rebase_onto, st.rebase_ref = "tune/best", head
        self.state.batch = {}

    def _pick(self, tip: str, commits: Sequence[str]) -> str | None:
        """Cherry-pick ``commits`` onto ``tip`` in the object store only (no checkout): the new tip, or None.

        A git before ``merge-tree --merge-base`` (2.40) picks the candidate as
        one change instead: a merge of its last commit, whose base git finds
        itself, which is a cherry-pick of the whole only where that base is
        where the candidate's branch began. Elsewhere it does not stack.
        """
        if not commits:
            return tip
        version = git_version()
        if version is not None and version < MERGE_TREE_MERGE_BASE:
            start = git_checked(self.repo, "rev-parse", f"{commits[0]}^", check=False)
            base = git_checked(self.repo, "merge-base", tip, commits[-1], check=False)
            if not start or base != start:
                return None
            steps = [(commits[-1], [])]
        else:
            steps = [(c, [f"--merge-base={c}^"]) for c in commits]
        for c, merge_base in steps:
            out = subprocess.run(
                ["git", "-C", str(self.repo), "merge-tree", "--write-tree", *merge_base, tip, c],
                capture_output=True,
                text=True,
            )
            if out.returncode != 0:
                return None
            tip = git_checked(
                self.repo, "commit-tree", out.stdout.split()[0], "-p", tip, "-m", f"stack {c}"
            )
        return tip

    def _compatible_set(self, head: str, winners: Sequence[CandidateState]) -> list[CandidateState]:
        """The winners that cherry-pick together onto ``head`` with the largest summed gain."""
        gain = [max(st.deltas.values() or [0.0]) for st in winners]
        best: tuple[list[CandidateState], float] = ([], -1.0)

        def walk(i: int, tip: str, chosen: list[CandidateState], total: float) -> None:
            nonlocal best
            if total + sum(g for g in gain[i:] if g > 0) <= best[1]:
                return
            if i == len(winners):
                best = (chosen, total)
                return
            nxt = self._pick(tip, winners[i].commits)
            if nxt is not None:
                walk(i + 1, nxt, [*chosen, winners[i]], total + gain[i])
            walk(i + 1, tip, chosen, total)

        walk(0, head, [], 0.0)
        return best[0]

    def _keep(self, st: CandidateState) -> None:
        best = Path(self.state.best_tree)
        before = git_checked(best, "rev-parse", "HEAD")
        if st.eval_tree and before != st.eval_base:
            # Another keep landed while this one was gated: measure it again on the new best.
            self.log(f"[campaign] {st.slug}: best moved during its gate, re-stacking")
            if self._stack(st):
                st.status, st.gate = "e2e", {}
            return
        source = git_checked(Path(st.eval_tree), "rev-parse", "HEAD") if st.eval_tree else ""
        out = subprocess.run(
            ["git", "-C", str(best), "merge", "--ff-only", source] if source
            else ["git", "-C", str(best), "cherry-pick", *st.commits],
            capture_output=True, text=True,
        )  # fmt: skip
        if out.returncode != 0:
            git_checked(best, "cherry-pick", "--abort", check=False)
            st.status, st.reason = (
                "conflict",
                f"could not move tune/best: {out.stderr.strip()[:300]}",
            )
            st.rebase_onto, st.rebase_ref = "tune/best", before
            self.ledger.append("candidate_conflict", slug=st.slug, detail=st.reason)
            return
        st.kept_commits = git_checked(best, "rev-list", "--reverse", f"{before}..HEAD").splitlines()
        st.status = "kept"
        # Every other measurement in flight was stacked on the old best and would be
        # re-measured anyway: stop it now and re-stack on the next pass.
        for other in self.state.candidates.values():
            if other.slug != st.slug and other.status == "won":
                other.status, other.e2e_label = "e2e", ""
            if other.slug != st.slug and other.status == "e2e" and other.e2e_label:
                if self.perf.cancel(other.e2e_label):
                    self.log(f"[campaign] {other.slug}: cancelled its measurement on the old best")
                    other.e2e_label = ""
                for label in other.samples:  # a re-measure on the old best
                    self.perf.cancel(label)
        self.state.best_commit = git_checked(best, "rev-parse", "HEAD")
        # The keep's own measurements are measurements of the new best tree at the
        # current config on its host: that host's first controls, for free.
        for label in [st.e2e_label, *st.samples]:
            kept = load_measurement(self.data, label)
            if kept is not None:
                self._register(str(kept.meta.get("hostname") or ""), label)
        self.state.kept.append(st.slug)
        self.ledger.append(
            "candidate_kept",
            slug=st.slug,
            commits=st.kept_commits,
            best=self.state.best_commit,
            deltas=st.deltas,
            gate=st.gate,
            ungated=bool(st.gate.get("ungated")),
        )
        self.log(
            f"[campaign] KEEP {st.slug} -> tune/best {self.state.best_commit[:10]}"
            + _ungated(st.gate)
        )
